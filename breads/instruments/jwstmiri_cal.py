import os.path
from copy import copy
from glob import glob
import astropy.io.fits as pyfits

import numpy as np
import stpsf as webbpsf
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.interpolate import interp1d

from copy import copy, deepcopy
import warnings
from astropy.utils.exceptions import AstropyUserWarning

import breads.utils as utils
from breads.utils import broaden, rotate_coordinates, find_closest_leftnright_elements
from breads.jwst_tools.flat_miri_utils import beta_masking_inverse_slice
from breads.jwst_tools.splines import normalize_rows
from scipy.ndimage import generic_filter
from breads.instruments.jwst_IFUs import JWST_IFUs
from breads.jwst_tools.default_miri import wv_ref_dict, sub_band_from_name


class JWSTMiri_cal(JWST_IFUs):
    def __init__(self, filename=None, channel_reduction='1', utils_dir=None, save_utils=True,
                 load_utils=True,
                 preproc_task_list=None,
                 verbose=True, wv_ref=None):
        """JWST MIRI/MRS 2D calibrated data class.

        Parameters
        ----------
        filename: str
            Path to the calibrated FITS file.
        channel_reduction: str or int (default='1')
            MIRI channel selected for reduction. (MIRI/MRS calibrated images always have two channels).
        utils_dir: str or None
            Path to the folder saving the intermediate products of each preprocessing step.
        save_utils: bool (default=True)
            Whether to save intermediate products.
        load_utils: bool (default=True)
            Whether to load intermediate products.
        preproc_task_list: list or None
            List of preprocessing tasks to run.
        verbose: bool (default=True)
            If True, the code is returning more printing.

        About the "preproc_task_list" parameter. Each task should be a list containing:
            task[0] = the name of the class method
            task[1] = a dictionary with any relevant method arguments (but not including save_utils, see task[2])
                If not defined, it assumes no parameters are needed (task[1] = {}).
            task[2] = a boolean saying if the outputs should be saved in the utils folder.
                Default to class save_utils if not defined for the task.
                If it is a string instead, it will be saved with the string as the filename.
            task[3] = a boolean saying if we should attempt to load the data from the utils folder.
                Default to class load_utils if not defined for the task.
        """

        super().__init__(filename, utils_dir, verbose)
        self.ifu_name = 'miri'
        self._init_miri_channel_band(channel_reduction)
        self._init_mask_channel(wv_ref)
        self._transpose_data()
        self._init_additional_default_filenames()
        self.opmode = "IFU" #only option for MIRI

        super().run_preproc_list(save_utils=save_utils, load_utils=load_utils, preproc_task_list=preproc_task_list)

    def _init_miri_channel_band(self, channel_reduction):
        """Initialize attributes relative to MIRI channel and band.
        Also retrieve the pixelscale from crds and spectral resolution.

        Parameters
        ----------
        channel_reduction : str or int
            Must be set to '1', '2' '3' or '4'. It selects the targeted channel we want to use for the reduction.
        """

        self.sub_band_name = self.priheader["BAND"]
        self.combined_channel = self.priheader['CHANNEL']

        if type(channel_reduction) is str:
            self.channel_reduction = channel_reduction
        if type(channel_reduction) is int:
            self.channel_reduction = str(channel_reduction)

        if self.channel_reduction not in self.combined_channel:
            raise ValueError(
                f'Channel {self.channel_reduction} set for reduction but the cal files channel {self.combined_channel} do not match')

        self.sub_band = sub_band_from_name(self.sub_band_name)
        self.combined_channel_band = self.combined_channel + self.sub_band
        self.channel_band = self.channel_reduction + self.sub_band
        self.breads_header['CHANNEL_BAND'] = self.channel_band
        photom_file = self._get_crds_reference()
        if self.verbose:
            print(f"Photom file selected: {photom_file}")
        with fits.open(photom_file) as hdul_photom:
            self.area2d = hdul_photom['PIXSIZ'].data  # in steradian
        if self.channel_reduction == '1' or self.channel_reduction == '4':
            self.pixelscale = np.sqrt(np.nanmedian(self.area2d[10:-10, 20:500]) * 4.25e10)  # steradian to arcsec
        else:
            self.pixelscale = np.sqrt(np.nanmedian(self.area2d[10:-10, 500:-20]) * 4.25e10)

        self._set_mrs_spectral_resolution()

    def _get_crds_reference(self):
        """Return the local path of the CRDS best reference file for photometric calibration.
        """
        import crds
        import jwst.datamodels

        with jwst.datamodels.open(self.filename) as model:
            parameters = model.get_crds_parameters()
        return crds.getreferences(parameters, reftypes=['photom'], context=self.priheader.get('CRDS_CTX'),
                                  observatory='jwst')['photom']

    def _init_mask_channel(self, wv_ref):
        """Initializes a mask to hide channel data that is not useful for the desired reduction.

        Parameters
        ---------
        wv_ref : float or None
            Wavelength reference to later compute the quick monochromatic webbpsf
        """
        self.mask_channel = np.full_like(self.data, np.nan)
        if self.channel_reduction == '1' or self.channel_reduction == '4':
            self.mask_channel[:, :507] = 1
        elif self.channel_reduction == '2' or self.channel_reduction == '3':
            self.mask_channel[:, 507:] = 1
        self.data *= self.mask_channel
        self.wavelengths *= self.mask_channel
        self.wv_ref = wv_ref_dict[self.channel_band]
        self.breads_header['WV_REF'] = self.wv_ref

    def _init_wave_wcs(self, filename):
        hdulist = fits.open(filename)
        print(hdulist)
        try:
            self.wavelengths = hdulist['WAVELENGTH'].data
        except(Exception) as e:
            import jwst.datamodels
            model = jwst.datamodels.open(filename)

            # Pixel grid
            yy, xx = np.mgrid[0:1024, 0:1032]

            # Vectorized WCS transforms
            _, _, self.wavelengths = model.meta.wcs.transform('detector', 'world', xx, yy)
        return self.wavelengths

    def _init_wcs(self, filename):
        """Hook for MIRI subclass to compute World Coordinate System"""
        hdulist = fits.open(filename)
        try:
            self.wavelengths = hdulist['WAVELENGTH'].data
            self.ra_array = hdulist['RA_ARRAY'].data
            self.dec_array = hdulist['DEC_ARRAY'].data
            self.alpha = hdulist['ALPHA'].data
            self.beta = hdulist['BETA'].data
            self.trace_id_map = hdulist['TRACE_ID'].data
            print("WCS loaded")

        except(Exception) as e:

            import jwst.datamodels
            model = jwst.datamodels.open(filename)

            # Pixel grid
            yy, xx = np.mgrid[0:1024, 0:1032]

            # Vectorized WCS transforms
            self.ra_array, self.dec_array, self.wavelengths = model.meta.wcs.transform('detector', 'world', xx, yy)
            self.alpha, self.beta, _ = model.meta.wcs.transform('detector', 'alpha_beta', xx, yy)
            beta_values = np.unique(self.beta[~np.isnan(self.beta)])
            self.trace_id_map = np.zeros_like(self.data)  # TODO make the actual function to do the trace_id mapping
            '''for i, beta_value in enumerate(beta_values):
                self.trace_id_map[self.beta==beta_value] = int(self.channel) * 100 + i'''

            hdus = [
                ("RA_ARRAY", self.ra_array),
                ("DEC_ARRAY", self.dec_array),
                ("WAVELENGTH", self.wavelengths),
                ("ALPHA", self.alpha),
                ("BETA", self.beta),
                ("TRACE_ID", self.trace_id_map),
            ]

            for name, data in hdus:
                hdu = fits.ImageHDU(data=data, name=name)
                hdulist.append(hdu)

            hdulist.writeto(self.filename, overwrite=True)

            print("WCS computed")

            if hasattr(self, "mask_channel"):
                self.wavelengths *= self.mask_channel

        self.wavelengths = self.wavelengths.transpose()
        self.ra_array = self.ra_array.transpose()
        self.dec_array = self.dec_array.transpose()
        self.alpha = self.alpha.transpose()
        self.beta = self.beta.transpose()
        self.trace_id_map = self.trace_id_map.transpose()


        return self.ra_array, self.dec_array, self.wavelengths, self.area2d

    def _init_additional_default_filenames(self):
        """Initializes the default filenames used only for nirspec preprocessing."""
        self.default_filenames["compute_odd_even_correction"] = \
                os.path.join(self.utils_dir, os.path.basename(self.filename).replace(".fits", "_odd_even_correction.fits"))

    def _save_coordinates_arrays(self, save_utils):
        """ Save the computed sky coordinates into the save utils directory.
        This will save in a fits the relative coordinates in arcsec, the 2D pixelscale in steradian and the ID of the traces

        Format of the utils file:
        No.    Name      Ver    Type      Cards   Dimensions   Format
          0  PRIMARY       1 PrimaryHDU     264   ()
          1  WAVE          1 ImageHDU        73   (2048, 2048)   float32
          2  X             1 ImageHDU         9   (2048, 2048)   float64
          3  Y             1 ImageHDU         9   (2048, 2048)   float64
          4  AREA2D        1 ImageHDU         9   (2048, 2048)   float32
          5  TRACE_ID_MAP    1 ImageHDU         8   (2048, 2048)   float64
          6  BREADS        1 ImageHDU        11   ()

        Parameters
        ----------
        save_utils : str or None
            directory to save the computed coordinates. If None, the coordinates are saved in default utils directory.
        """
        if isinstance(save_utils, str):
            out_filename = save_utils
        else:
            out_filename = self.default_filenames["compute_coordinates_arrays"]

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(header=self.priheader))
        hdulist.append(pyfits.ImageHDU(data=self.wavelengths.transpose(), header=self.extheader, name='WAVE'))
        hdulist.append(pyfits.ImageHDU(data=self.x.transpose(),
                                       header=pyfits.Header({'BUNIT': self.breads_header['COORUNIT']}),
                                        name='X'))
        hdulist.append(pyfits.ImageHDU(data=self.y.transpose(),
                                       header=pyfits.Header({'BUNIT': self.breads_header['COORUNIT']}),
                                        name='Y'))
        hdulist.append(pyfits.ImageHDU(data=self.y.transpose(),
                                       header=pyfits.Header({'BUNIT': self.breads_header['COORUNIT']}),
                                       name='Y'))
        hdulist.append(pyfits.ImageHDU(data=self.area2d.transpose(),
                                       header=pyfits.Header({'BUNIT': self.breads_header['AREAUNIT']}),
                                       name='AREA2D'))
        hdulist.append(pyfits.ImageHDU(data=self.trace_id_map.transpose(), name='TRACE_ID_MAP'))
        hdulist.append(pyfits.ImageHDU(header=self.breads_header, name='BREADS'))
        hdulist.writeto(out_filename, overwrite=True)
        hdulist.close()
        if self.verbose:
            print(f"  Saved the computed coordinates arrays to {out_filename}")

    def reload_coordinates_arrays(self, load_filename=None):
        """ Reload coordinates arrays

        This updates the attributes self.x, self.y, self.area2d.

        Parameters
        ----------
        load_filename : str or None
            Filename to load coordinates from. If None, will use a default filename.
        """
        if load_filename is None:
            load_filename = self.default_filenames["compute_coordinates_arrays"]
        if len(glob(load_filename)) ==0:
            return None

        with pyfits.open(load_filename) as hdulist:
            wave_array = hdulist["WAVE"].data.transpose()
            x = hdulist['X'].data.transpose()
            y = hdulist['Y'].data.transpose()
            area2d = hdulist['AREA2D'].data.transpose()
            try:
                self.trace_id_map = hdulist['TRACE_ID_MAP'].data.transpose()
            except KeyError:
                print("Old reduction of coordinates. Could not find hdulist['TRACE_ID_MAP'].data. Please reprocess.")
            try:
                hdr_area2d = hdulist['AREA2D'].header['BUNIT']
            except KeyError:
                print("Warning - Could not find data unit for hdulist['AREA2D']. \n *_relcoords.fits intermediate files seems to be computed from an old breads version")
                mean_area2d = np.nanmedian(area2d)
                if mean_area2d < 1e-12: #Unit surely in steradian
                    print("Unit seems to be in steradian")
                else:
                    print("Unit seems to be in arcsec^2, converting to steradian for compatibility")
                    arcsec2_to_steradian = (2. * np.pi / (360. * 3600.)) ** 2
                    area2d *= arcsec2_to_steradian
            self.breads_header['COORDS'] = hdulist['BREADS'].header['COORDS']
            self.breads_header['COORUNIT'] = hdulist['X'].header['BUNIT']
            self.breads_header['AREAUNIT'] = hdulist['AREA2D'].header['BUNIT']

        self.x, self.y, self.area2d = x, y, area2d
        return wave_array, x, y, area2d



    def _set_mrs_spectral_resolution(self):
        """Spectral resolution table for each MIRI channel/band"""
        R_table = {
            ('1A'): 2700,
            ('1B'): 3000,
            ('1C'): 3200,
            ('2A'): 2600,
            ('2B'): 2800,
            ('2C'): 3000,
            ('3A'): 2400,
            ('3B'): 2600,
            ('3C'): 2800,
            ('4A'): 2200,
            ('4B'): 2400,
            ('4C'): 2600,
        }

        key = (self.channel_band)

        try:
            self.R = R_table[key]
        except KeyError:
            raise ValueError(f"Invalid channel/band inputs : {key}")

    def _transpose_data(self):
        self.data = self.data.transpose()
        self.noise = self.noise.transpose()
        self.bad_pixels = self.bad_pixels.transpose()
        self.area2d = self.area2d.transpose()


    def _save_coordinates_arrays(self, save_utils):
        """ Save the computed sky coordinates into the save utils directory.
        This will save in a fits the relative coordinates in arcsec, the 2D pixelscale in steradian and the ID of the traces

        Format of the utils file:
        No.    Name      Ver    Type      Cards   Dimensions   Format
          0  PRIMARY       1 PrimaryHDU     264   ()
          1  WAVE          1 ImageHDU        73   (2048, 2048)   float32
          2  X             1 ImageHDU         9   (2048, 2048)   float64
          3  Y             1 ImageHDU         9   (2048, 2048)   float64
          4  AREA2D        1 ImageHDU         9   (2048, 2048)   float32
          5  TRACE_ID_MAP    1 ImageHDU         8   (2048, 2048)   float64
          6  BREADS        1 ImageHDU        11   ()

        Parameters
        ----------
        save_utils : str or None
            directory to save the computed coordinates. If None, the coordinates are saved in default utils directory.
        """
        if isinstance(save_utils, str):
            out_filename = save_utils
        else:
            out_filename = self.default_filenames["compute_coordinates_arrays"]

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(header=self.priheader))
        hdulist.append(pyfits.ImageHDU(data=self.wavelengths.transpose(), header=self.extheader, name='WAVE'))
        hdulist.append(pyfits.ImageHDU(data=self.x.transpose(),
                                       header=pyfits.Header({'BUNIT': self.breads_header['COORUNIT']}),
                                        name='X'))
        hdulist.append(pyfits.ImageHDU(data=self.y.transpose(),
                                       header=pyfits.Header({'BUNIT': self.breads_header['COORUNIT']}),
                                        name='Y'))
        hdulist.append(pyfits.ImageHDU(data=self.alpha.transpose(),
                                       header=pyfits.Header({'BUNIT': self.breads_header['COORUNIT']}),
                                       name='ALPHA'))
        hdulist.append(pyfits.ImageHDU(data=self.beta.transpose(),
                                       header=pyfits.Header({'BUNIT': self.breads_header['COORUNIT']}),
                                       name='BETA'))
        hdulist.append(pyfits.ImageHDU(data=self.area2d.transpose(),
                                       header=pyfits.Header({'BUNIT': self.breads_header['AREAUNIT']}),
                                       name='AREA2D'))
        hdulist.append(pyfits.ImageHDU(data=self.trace_id_map.transpose(), name='TRACE_ID_MAP'))
        hdulist.append(pyfits.ImageHDU(header=self.breads_header, name='BREADS'))
        hdulist.writeto(out_filename, overwrite=True)
        hdulist.close()
        if self.verbose:
            print(f"  Saved the computed coordinates arrays to {out_filename}")

    def reload_coordinates_arrays(self, load_filename=None):
        """ Reload coordinates arrays

        This updates the attributes self.x, self.y, self.area2d.

        Parameters
        ----------
        load_filename : str or None
            Filename to load coordinates from. If None, will use a default filename.
        """
        if load_filename is None:
            load_filename = self.default_filenames["compute_coordinates_arrays"]
        if len(glob(load_filename)) ==0:
            return None

        with pyfits.open(load_filename) as hdulist:
            wave_array = hdulist["WAVE"].data.transpose()
            x = hdulist['X'].data.transpose()
            y = hdulist['Y'].data.transpose()
            alpha = hdulist['ALPHA'].data.transpose()
            beta = hdulist['BETA'].data.transpose()
            area2d = hdulist['AREA2D'].data.transpose()
            try:
                self.trace_id_map = hdulist['TRACE_ID_MAP'].data.transpose()
            except KeyError:
                print("Old reduction of coordinates. Could not find hdulist['TRACE_ID_MAP'].data. Please reprocess.")
            try:
                hdr_area2d = hdulist['AREA2D'].header['BUNIT']
            except KeyError:
                print("Warning - Could not find data unit for hdulist['AREA2D']. \n *_relcoords.fits intermediate files seems to be computed from an old breads version")
                mean_area2d = np.nanmedian(area2d)
                if mean_area2d < 1e-12: #Unit surely in steradian
                    print("Unit seems to be in steradian")
                else:
                    print("Unit seems to be in arcsec^2, converting to steradian for compatibility")
                    arcsec2_to_steradian = (2. * np.pi / (360. * 3600.)) ** 2
                    area2d *= arcsec2_to_steradian
            self.breads_header['COORDS'] = hdulist['BREADS'].header['COORDS']
            self.breads_header['COORUNIT'] = hdulist['X'].header['BUNIT']
            self.breads_header['AREAUNIT'] = hdulist['AREA2D'].header['BUNIT']

        self.wavelengths, self.x, self.y, self.alpha, self.beta, self.area2d = wave_array, x, y, alpha, beta, area2d
        return wave_array, x, y, area2d

    def compute_med_filt_badpix(self, save_utils=False):
        """ Quick bad pixel identification.

        The data is first high-pass filtered row by row with a median filter with a window size of 50 (window_size)
        pixels. The median absolute deviation (MAP) is then calculated row by row, and any pixel deviating by more than
        50x the MAP are identified as bad.

        Only returns (or save) the newly identified bad pixels, the ones already included in self.bad_pixels won't be in
        new_badpix.
        But this map is automatically applied to self.bad_pixels:  self.bad_pixels *= new_badpix

        Parameters
        ----------
        save_utils : bool or string
            Save the computed bad pixel map (nans=bad) into the utils directory
            Default filename (set save_utils as a string instead of bool to override filename):
            os.path.join(self.utils_dir, os.path.basename(self.filename).replace(".fits", "_med_filt_badpix.fits"))

        Returns
        -------
        new_badpix : np.array
            nans = bad.
        """

        if self.verbose:
            print("Initializing row_err and bad_pixels for MIRI")
        new_badpix = np.ones(self.bad_pixels.shape)

        for rowid in range(self.bad_pixels.shape[0]):
            row_flux = np.copy(self.data[rowid, :])

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                smooth = generic_filter(row_flux, np.nanmedian, size=50)
                row_flux = row_flux / smooth

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyUserWarning)
                clipped_data = sigma_clip(row_flux, sigma=3, maxiters=3)
                new_badpix[rowid, clipped_data.mask] = np.nan

        self.bad_pixels *= new_badpix

        if save_utils:
            self._save_med_filt_badpix(save_utils, new_badpix)

        return new_badpix

    def _save_med_filt_badpix(self, save_utils, new_badpix):
        """Save the bad pixel map computed via median filtering.

        Format:
        Name        Ver    Type          Cards   Dimensions      Format
        --------    ---    ----          -----   ----------      ------
        PRIMARY       1    PrimaryHDU     268    ()
        BADPIXEL      1    ImageHDU        73    (2048, 2048)    float64
        BREADS        1    ImageHDU        13    ()

        Parameters
        ----------
        save_utils: str or None
            Path to save the bad pixel map computed via median filtering. if None, by default path and filename are used.
        new_badpix: 2d np.array
            New bad pixel map computed via median filtering.

        """
        if isinstance(save_utils, str):
            out_filename = save_utils
        else:
            out_filename = self.default_filenames["compute_med_filt_badpix"]

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(header=self.priheader))
        hdulist.append(pyfits.ImageHDU(data=new_badpix.transpose(), header=self.extheader, name='BADPIXEL'))
        hdulist.append(pyfits.ImageHDU(header=self.breads_header, name='BREADS'))
        hdulist.writeto(out_filename, overwrite=True)
        hdulist.close()
        if self.verbose:
            print(f"  Saved the quick bad pixel map to {out_filename}")

    def reload_med_filt_badpix(self, load_filename=None):
        """ Reload and apply bad pixel map from med_filt_badpix.

        Parameters
        ----------
        load_filename : str or None
            Loading directory. If None, will use self.default_filenames["compute_med_filt_badpix"]

        Returns
        -------
        new_badpix : np.array
            nans = bad.

        """
        if load_filename is None:
            load_filename = self.default_filenames["compute_med_filt_badpix"]
        if len(glob(load_filename)) == 0:
            return None

        with pyfits.open(load_filename) as hdulist:
            new_badpix = hdulist['BADPIXEL'].data.transpose()

        self.bad_pixels *= new_badpix

        return new_badpix

    def _get_webbpsf_model_inputs(self, image_mask, pixel_scale):
        """Hook for MIRI subclass, returns webbpsf parameters for MRS simulation"""
        miri = webbpsf.MIRI()
        miri.load_wss_opd_by_date(self.priheader["DATE-BEG"])  # Load telescope state as of our observation date
        miri.image_mask = image_mask  # optional: model opaque field stop outside of the IFU aperture
        miri.pixelscale = self.pixelscale  # Optional: set this manually to match the drizzled cube sampling, rather than the default
        return miri

    def _get_webbpsf_fit_inputs(self):
        """
        Hook for MIRI, returns input for webbPSF fit
        """
        return (
            np.copy(self.bad_pixels),
            np.copy(self.data),
            np.copy(self.noise),
            np.copy(self.x),
            np.copy(self.y),
            np.copy(np.abs(self.wavelengths - self.webbpsf_wv0)),
        )

    def _get_starspectrum_input(self, spec_R_sampling, x_nodes, N_nodes):
        """Hook for MIRI, returns input for continuum normalized star spectrum computation."""
        im = np.copy(self.data)
        im_wvs = np.copy(self.wavelengths)
        err = np.copy(self.noise)
        if spec_R_sampling is None:
            spec_R_sampling = self.R * 4
        if x_nodes is None:
            x_nodes = np.linspace(np.nanmin(im_wvs), np.nanmax(im_wvs), N_nodes, endpoint=True)
        bad_pixels = np.copy(self.bad_pixels)

        return im, im_wvs, err, bad_pixels, spec_R_sampling, x_nodes

    def compute_odd_even_correction(self, save_utils=False, mppool=None,
        spec_R_sampling=None, threshold_badpix=10, wv_nodes=None, N_nodes=40, iterative=True):

        """ Compute star spectrum normalized by the continuum

        Parameters
        ----------
        save_utils : Boolean
            Save the intermediate star subtraction step products.
        im : str or None
        TODO : Is it still useful to keep those optional args? It feels like it was for testing purposes...
        im_wvs : str or None
        TODO : same
        err : str or None
        TODO : same
        mppool : multiprocessing.Pool or None (optional)
            If None, the computation is done without parallelization.
        spec_R_sampling : float or None (optional)
            Spectral resolution to sample the continuum-normalized star spectrum
            If None, the spectral resolution will be set to 4 times the instrumental spectral resolution of the IFU.
        threshold_badpix : float (optional)
            Hard threshold for bad pixel flagging. Thresholding is done by comparing the continuum normalized row with its median standard deviation.
        x_nodes : 1d array or None (optional)
            If x_nodes is specified, this wavelength spacing (in micron) will be used to do the splines fitting.
            If None, N_nodes will set an evenly nodes spacing.
        N_nodes : int or None (optional, default is 40)
            Number of nodes to use for fitting splines for the continuum star spectrum estimation.
        iterative : Boolean (optional)
            If True, the fitting procedure is iteratively applied. It helps identifies potential additional bad pixels flagging and to have a better regularization.

        Returns
        -------
        new_wavelengths : 1d numpy array (N_wavelengths)
            New wavelengths axis of the combined high-frequency star spectrum (micron)
        combined_fluxes : 1d numpy array (N_wavelengths)
            Combined high-frequency star spectrum (MJy or MJy/sr)
        combined_errors : 1d numpy array (N_wavelengths)
            Combined flux errors (MJy or MJy/sr)
        spline_cont0 : 2d numpy array (N_detector_rows, N_detector_cols)
            Star continuum fitted by splines for each spectral trace of the detector.
        spline_paras0 : 2d numpy array (N_nodes, N_traces)
            Linear parameters returned by the continuum spline fitting routine for each spectral trace of the detector.
        x_nodes : 1d numpy array (N_nodes)
            Nodes spacing in the wavelength dimension (in micron).

        """
        im = np.copy(self.data)
        im_wvs = np.copy(self.wavelengths)
        err = np.copy(self.noise)
        if spec_R_sampling is None:
            spec_R_sampling = self.R * 4
        if wv_nodes is None:
            wv_nodes = np.linspace(np.nanmin(im_wvs), np.nanmax(im_wvs), N_nodes, endpoint=True)
        bad_pixels = np.copy(self.bad_pixels)


        if self.verbose:
            print(f"Computing stellar spectrum (continuum normalized)")

        stellar_features = None

        # Define the regularization in the form of priors on the value of the flux at the position of the spline nodes
        reg_mean_map0 = np.zeros((im.shape[0], np.size(wv_nodes)))  # The flux values at each position of the nodes
        reg_std_map0 = np.zeros(
            (im.shape[0], np.size(wv_nodes)))  # The corresponding width of the Gaussian prior at each nodes
        for rowid, row in enumerate(im):
            row_bp = bad_pixels[rowid, :]
            if np.nansum(np.isfinite(row * row_bp)) == 0:
                continue
            # Set the prior to the median value of the row for each node
            median_row = np.nanmedian(row * row_bp)
            stddev_row = np.nanstd(row * row_bp)
            reg_mean_map0[rowid, :] = median_row
            # Set the width of the prior to its mean to have fairly unconstraining priors
            reg_std_map0[rowid, :] = np.max([np.abs(median_row), stddev_row])

        if reg_mean_map0 is None and reg_std_map0 is None:
            regularization = False
        else:
            regularization = True
        spline_cont0, _, new_badpixs, new_res, spline_paras0 = normalize_rows(im, im_wvs, noise=err,
                                                                              badpixs=bad_pixels,
                                                                              wv_nodes=wv_nodes, mppool=mppool,
                                                                              threshold=threshold_badpix,
                                                                              stellar_features=stellar_features,
                                                                              regularization=regularization,
                                                                              reg_mean_map=reg_mean_map0,
                                                                              reg_std_map=reg_std_map0)
        if iterative:
            if regularization:
                reg_mean_map1 = copy(spline_paras0)
                where_nan = np.where(np.isnan(reg_mean_map1))
                reg_mean_map1[where_nan] = reg_mean_map0[where_nan]
                reg_std_map1 = np.abs(reg_mean_map1)
            else:
                reg_mean_map1 = None
                reg_std_map1 = None
            spline_cont0, _, new_badpixs, new_res, spline_paras0 = normalize_rows(im, im_wvs, noise=err,
                                                                                  badpixs=new_badpixs,
                                                                                  wv_nodes=wv_nodes, mppool=mppool,
                                                                                  threshold=threshold_badpix,
                                                                                  stellar_features=stellar_features,
                                                                                  regularization=regularization,
                                                                                  reg_mean_map=reg_mean_map1,
                                                                                  reg_std_map=reg_std_map1)

        # _get_masked_normalized_object does the continuum normalization and allows for different behavior between NIRSpec and MIRI
        continuum, normalized_im, normalized_err = spline_cont0, im/spline_cont0, err/spline_cont0 #self._get_masked_normalized_object(spline_cont0, im, err)

        weights = 1/normalized_err ** 2

        # mask NaNs / inf
        mask = (
                np.isfinite(normalized_im)
                & np.isfinite(weights)
                & (weights > 0)
        )

        row_mean = np.zeros(normalized_im.shape[1])

        for i in range(normalized_im.shape[1]):
            good = mask[:, i]

            if np.sum(good) < 10: #at least 10 pixels valid for a column or skip it
                continue

            row_mean[i] = np.average(
                normalized_im[good, i],
                weights=weights[good, i]
            )

        normalized_im = normalized_im / row_mean[None, :]

        self.data /= row_mean[None, :]

        if save_utils:
            self._save_odd_even_correction(save_utils, row_mean, normalized_im)


    def _save_odd_even_correction(self, save_utils, row_mean, normalized_im):
        if isinstance(save_utils, str):
            out_filename = save_utils
        else:
            out_filename = self.default_filenames["compute_odd_even_correction"]

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU())
        hdulist.append(pyfits.ImageHDU(data=row_mean, name='MEAN_CORR'))
        hdulist.append(pyfits.ImageHDU(data=normalized_im, name='CONT_NORM_IM'))
        hdulist.writeto(out_filename, overwrite=True)
        hdulist.close()


    def reload_odd_even_correction(self, load_filename=None):
        if load_filename is None:
            load_filename = self.default_filenames["compute_odd_even_correction"]
        if len(glob(load_filename)) == 0:
            return None

        with pyfits.open(load_filename) as hdulist:
            row_mean = hdulist['MEAN_CORR'].data

        self.data /= row_mean[:, None]

        return row_mean

    def _get_masked_normalized_object(self, continuum, im, err):
        """Hook for MIRI, create a mask to only keep high SNR data before computing continuum normalized star spectrum."""
        continuum = np.copy(continuum)

        mask_brightest_slices = beta_masking_inverse_slice(self.data.transpose(), self.beta.transpose(), self.channel_reduction,
                                                           N_slices=4)
        mask_brightest_slices[mask_brightest_slices == 0] = np.nan
        continuum *= mask_brightest_slices.transpose()

        continuum[np.where(continuum / err < 50)] = np.nan
        continuum[np.where(np.isnan((self.bad_pixels)))] = np.nan

        normalized_im = im / continuum
        normalized_err = err / continuum

        return continuum, normalized_im, normalized_err

    def _set_bad_pixels(self, bad_pixels):
        """Hook for MIRI to set the bad pixels map. Used only in JWST_IFUs.compute_starspectrum()."""
        self.bad_pixels = bad_pixels

    def _save_starspectrum_contnorm(self, save_utils, new_wavelengths, combined_fluxes, combined_errors, spline_cont0, spline_paras0, wv_nodes, normalized_im, stellar_features):
        """Save continuum normalized star spectrum results."""
        if isinstance(save_utils, str):
            out_filename = save_utils
        else:
            out_filename = self.default_filenames["compute_starspectrum_contnorm"]

        self.breads_header["STCONTFN"] = out_filename
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(header=self.priheader))
        hdulist.append(pyfits.ImageHDU(data=new_wavelengths, header=self.extheader, name="WAVE"))
        hdulist.append(pyfits.ImageHDU(data=combined_fluxes, name='COM_FLUXES'))
        hdulist.append(pyfits.ImageHDU(data=combined_errors, name='COM_ERRORS'))
        hdulist.append(pyfits.ImageHDU(data=spline_cont0.transpose(), name='SPLINE_CONT0'))
        hdulist.append(pyfits.ImageHDU(data=spline_paras0, name='SPLINE_PARAS0'))
        hdulist.append(pyfits.ImageHDU(data=wv_nodes, name='wv_nodes'))
        hdulist.append(pyfits.ImageHDU(data=normalized_im.transpose(), name='CONT_NORM_IM'))
        if stellar_features is not None:
            hdulist.append(pyfits.ImageHDU(data=stellar_features, name='STELLAR_FEATURES'))
        hdulist.append(pyfits.ImageHDU(header=self.breads_header, name='BREADS'))
        hdulist.writeto(out_filename, overwrite=True)
        hdulist.close()

    def reload_starspectrum_contnorm(self, load_filename=None):
        """ Reload star spectrum normalized by continuum

        Parameters
        ----------
        load_filename : str or None
            Filename to load spectrum data from, or leave None to use default filename

        Returns
        -------
        new_wavelengths : 1d numpy array (N_wavelengths)
            New wavelengths axis of the combined high-frequency star spectrum (micron)
        combined_fluxes : 1d numpy array (N_wavelengths)
            Combined continuum normalized spectrum of the star.
        combined_errors : 1d numpy array (N_wavelengths)
            error vector for combined_fluxes
        spline_cont0 : 2d numpy array (N_detector_rows, N_detector_cols)
            Star continuum fitted by splines for each spectral trace of the detector.
        spline_paras0 : 2d numpy array (N_nodes, N_traces)
            Linear parameters returned by the continuum spline fitting routine for each spectral trace of the detector.
        wv_nodes : 1d numpy array (N_nodes)
            Nodes spacing in the wavelength dimension (in micron).

        Also sets self.wv_nodes and self.star_func according to values in the reloaded file headers.

        """
        if load_filename is None:
            load_filename = self.default_filenames["compute_starspectrum_contnorm"]
        if len(glob(load_filename)) ==0:
            return None

        hdulist = pyfits.open(load_filename)
        new_wavelengths = hdulist["WAVE"].data
        combined_fluxes = hdulist["COM_FLUXES"].data
        combined_errors = hdulist["COM_ERRORS"].data
        spline_cont0 = hdulist["SPLINE_CONT0"].data.transpose()
        spline_paras0 = hdulist["SPLINE_PARAS0"].data.transpose()
        try:
            wv_nodes = hdulist['wv_nodes'].data
        except:
            wv_nodes = hdulist['x_nodes'].data
        self.breads_header['STCONTRS'] = hdulist['BREADS'].header['STCONTRS']
        self.breads_header['STCONTTH'] = hdulist['BREADS'].header['STCONTTH']
        self.breads_header["STCONTFN"] = load_filename
        hdulist.close()

        self.check_and_update_nodes(wv_nodes)
        self.star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False, fill_value=1)
        return new_wavelengths,combined_fluxes,combined_errors,spline_cont0,spline_paras0,wv_nodes

    def _get_starsub_inputs(self, load_starspectrum_contnorm):
        """Hook for MIRI, return input for subtracting star spectrum from the data."""
        if load_starspectrum_contnorm is None:
            load_starspectrum_contnorm = self.default_filenames["compute_starspectrum_contnorm"]

        hdulist = pyfits.open(load_starspectrum_contnorm)
        spline_paras0 = hdulist["SPLINE_PARAS0"].data

        if 'STELLAR_FEATURES' in hdulist:
            stellar_features0 = hdulist['STELLAR_FEATURES'].data
        else:
            stellar_features0 = None
        hdulist.close()

        wherenan = np.where(np.isnan(spline_paras0))
        reg_mean_map = copy(spline_paras0)

        reg_mean_map[wherenan] = np.tile(np.nanmedian(spline_paras0, axis=1)[:, None], (1, spline_paras0.shape[1]))[wherenan]
        reg_std_map = np.abs(spline_paras0)
        reg_std_map[wherenan] = np.tile(np.nanmax(np.abs(spline_paras0), axis=1)[:, None], (1, spline_paras0.shape[1]))[wherenan]
        reg_std_map = reg_std_map
        reg_std_map = np.clip(reg_std_map, 1e-11, np.inf)

        im = np.copy(self.data)
        im_wvs = np.copy(self.wavelengths)
        err = np.copy(self.noise)

        bad_pixels = np.copy(self.bad_pixels)

        return im, im_wvs, err, bad_pixels, reg_mean_map, reg_std_map, stellar_features0

    def _save_starsubtraction(self, save_utils, subtracted_im, im, star_model, spline_paras0, starsub_dir):
        """Hook for MIRI, save intermediate output used for the star subtraction from the data."""

        if isinstance(save_utils, str):
            out_filename = save_utils
        else:
            out_filename = self.default_filenames["compute_starsubtraction"]

        _breads_header = copy(self.breads_header)
        _breads_header["DATA_HPF"] = True
        _breads_header["HPF_TYPE"] = "spline1d"

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(header=self.priheader))
        hdulist.append(pyfits.ImageHDU(data=subtracted_im.transpose(), header=self.extheader, name="IM_SUB"))
        hdulist.append(pyfits.ImageHDU(data=im.transpose(), name='IM'))
        hdulist.append(pyfits.ImageHDU(data=star_model.transpose(), name='STARMODEL'))
        hdulist.append(pyfits.ImageHDU(data=self.bad_pixels.transpose(), name='BADPIX'))
        hdulist.append(pyfits.ImageHDU(data=spline_paras0, name='SPLINE_PARAS0'))
        hdulist.append(pyfits.ImageHDU(data=self.wv_nodes, name='wv_nodes'))
        hdulist.append(pyfits.ImageHDU(header=_breads_header, name='BREADS'))
        hdulist.writeto(out_filename, overwrite=True)

        if starsub_dir is not None:
            if not os.path.exists(os.path.join(self.utils_dir, starsub_dir)):
                os.makedirs(os.path.join(self.utils_dir, starsub_dir))
            hdulist_sc = pyfits.open(self.filename)
            du = self.breads_header["DATAUNIT"]
            bu = self.extheader["BUNIT"].strip()
            if du == 'MJy' and bu == 'MJy':
                hdulist_sc["SCI"].data = subtracted_im.transpose()
            if du == 'MJy/sr' and bu == 'MJy/sr':
                hdulist_sc["SCI"].data = subtracted_im.transpose()
            if du == 'MJy/sr' and bu == 'MJy':
                hdulist_sc["SCI"].data = (subtracted_im * self.area2d).transpose()
            if du == 'MJy' and bu == 'MJy/sr':
                hdulist_sc["SCI"].data = (subtracted_im / self.area2d).transpose()
            hdulist_sc["DQ"].data[np.where(np.isnan(self.bad_pixels.transpose()))] = 1
            hdulist_sc.writeto(os.path.join(self.utils_dir, starsub_dir, os.path.basename(self.filename)),
                               overwrite=True)
            hdulist_sc.close()

    def reload_starsubtraction(self, load_filename=None):
        """ Reload Star Subtraction

        Parameters
        ----------
        load_filename : str or None
            Filename to load PSF subtracted data from, or leave None to use default filename

        Returns
        -------
        subtracted_im, star_model, spline_paras0, wv_nodes

        Also modifies self.bad_pixels

        """
        if load_filename is None:
            load_filename = self.default_filenames["compute_starsubtraction"]
        if len(glob(load_filename)) ==0:
            return None

        hdulist = pyfits.open(load_filename)
        subtracted_im = hdulist["IM_SUB"].data.transpose()
        star_model = hdulist["STARMODEL"].data.transpose()
        fmderived_bad_pixels = hdulist['BADPIX'].data.transpose()
        spline_paras0 = hdulist["SPLINE_PARAS0"].data
        wv_nodes = hdulist['wv_nodes'].data
        self.breads_header["DATA_HPF"] = hdulist["BREADS"].header["DATA_HPF"]
        self.breads_header["HPF_TYPE"] = hdulist["BREADS"].header["HPF_TYPE"]
        self.breads_header['STSUBTH'] = hdulist["BREADS"].header["STSUBTH"]
        self.breads_header['STCONTRS'] = hdulist['BREADS'].header['STCONTRS']
        self.breads_header['STCONTTH'] = hdulist['BREADS'].header['STCONTTH']
        self.breads_header["STCONTFN"] = hdulist['BREADS'].header['STCONTFN']
        hdulist.close()

        self.check_and_update_nodes(wv_nodes)

        self.bad_pixels = self.bad_pixels * fmderived_bad_pixels
        self.data = subtracted_im

        return subtracted_im, star_model, spline_paras0, wv_nodes

    def _init_regwvs_obj(self, regwvs_dataobj, Ntraces, Nwv):
        """Hook for MIRI, initialize the interpolated calibrated data over regular wavelength grid object."""
        regwvs_dataobj.x = np.full((Ntraces, Nwv), np.nan)
        regwvs_dataobj.y = np.full((Ntraces, Nwv), np.nan)
        regwvs_dataobj.wavelengths = np.full((Ntraces, Nwv), np.nan)
        regwvs_dataobj.leftnright_wavelengths = np.full((2, Ntraces, Nwv), np.nan)
        regwvs_dataobj.data = np.full((Ntraces, Nwv), np.nan)
        regwvs_dataobj.noise = np.full((Ntraces, Nwv), np.nan)
        regwvs_dataobj.bad_pixels = np.full((Ntraces, Nwv), np.nan)
        regwvs_dataobj.area2d = np.full((Ntraces, Nwv), np.nan)

    def _get_interpdata_shapes(self, wv_sampling):
        """Hook for MIRI, return the number of spectral traces and the number of wavelength bins for the interpolated data."""
        Ntraces, Nwv = self.data.shape[0], np.size(wv_sampling)
        return Ntraces, Nwv

    def _get_where_finite(self, trace_id):
        """Hook for MIRI, return the index of the spectral trace where the data are not NaN."""
        wvs_finite = np.where(np.isfinite(self.wavelengths[trace_id, :]))
        where_finite = np.where(np.isfinite(self.bad_pixels[trace_id, :]))
        return wvs_finite, where_finite

    def _interpdata_regwvs_trace(self, regwvs_dataobj, wv_sampling, wvs_finite, where_finite, trace_id):
        """Hook for MIRI, method to interpolate the data over regular wavelength grid."""
        if self.combined_channel == '34':
            print("Flipping the wavelength axis for channel 34")
            regwvs_dataobj.x[trace_id] = np.interp(wv_sampling,
                                                              np.flip(self.wavelengths[trace_id, wvs_finite[0]]),
                                                              np.flip(self.x[trace_id, wvs_finite[0]]),
                                                              left=np.nan,
                                                              right=np.nan)
            regwvs_dataobj.y[trace_id] = np.interp(wv_sampling,
                                                               np.flip(self.wavelengths[trace_id, wvs_finite[0]]),
                                                               np.flip(self.y[trace_id, wvs_finite[0]]),
                                                               left=np.nan,
                                                               right=np.nan)

            regwvs_dataobj.wavelengths[trace_id] = wv_sampling
            regwvs_dataobj.area2d[trace_id] = np.interp(wv_sampling, np.flip(self.wavelengths[trace_id, wvs_finite[0]]),
                                                               np.flip(self.area2d[trace_id, wvs_finite[0]]),
                                                               left=np.nan,
                                                               right=np.nan)
            badpix_mask = np.isfinite(np.flip(self.bad_pixels[trace_id])).astype(float)
            regwvs_dataobj.bad_pixels[trace_id] = np.interp(wv_sampling,
                                                            np.flip(self.wavelengths[trace_id, wvs_finite[0]]),
                                                            np.flip(badpix_mask[wvs_finite]), left=0, right=0)

            # following little section written by chatgpt to find the left and right wavelengths in the original data
            v_left, v_right = find_closest_leftnright_elements(np.flip(self.wavelengths[trace_id, wvs_finite[0]]),
                                                               wv_sampling)

            regwvs_dataobj.leftnright_wavelengths[0, trace_id] = v_left
            regwvs_dataobj.leftnright_wavelengths[1, trace_id] = v_right

            where_finite = np.where(np.isfinite(np.flip(self.bad_pixels[trace_id])))

            regwvs_dataobj.data[trace_id] = np.interp(wv_sampling, np.flip(self.wavelengths[trace_id, wvs_finite[0]]),
                                                      np.flip(self.data[trace_id, wvs_finite[0]]), left=np.nan, right=np.nan)
            regwvs_dataobj.noise[trace_id] = np.interp(wv_sampling, np.flip(self.wavelengths[trace_id, wvs_finite[0]]),
                                                       np.flip(self.noise[trace_id, wvs_finite[0]]), left=np.nan,
                                                       right=np.nan)

        else:
            regwvs_dataobj.x[trace_id] = np.interp(wv_sampling, self.wavelengths[trace_id, wvs_finite[0]],
                                                              self.x[trace_id, wvs_finite[0]], left=np.nan,
                                                              right=np.nan)
            regwvs_dataobj.y[trace_id] = np.interp(wv_sampling, self.wavelengths[trace_id, wvs_finite[0]],
                                                               self.y[trace_id, wvs_finite[0]], left=np.nan,
                                                               right=np.nan)

            regwvs_dataobj.wavelengths[trace_id] = wv_sampling
            regwvs_dataobj.area2d[trace_id] = np.interp(wv_sampling, self.wavelengths[trace_id, wvs_finite[0]],
                                                        self.area2d[trace_id, wvs_finite[0]], left=np.nan, right=np.nan)
            badpix_mask = np.isfinite(self.bad_pixels[trace_id]).astype(float)
            regwvs_dataobj.bad_pixels[trace_id] = np.interp(wv_sampling, self.wavelengths[trace_id, wvs_finite[0]],
                                                            badpix_mask[wvs_finite], left=0, right=0)

            # following little section written by chatgpt to find the left and right wavelengths in the original data
            v_left, v_right = find_closest_leftnright_elements(self.wavelengths[trace_id, wvs_finite[0]], wv_sampling)

            regwvs_dataobj.leftnright_wavelengths[0, trace_id] = v_left
            regwvs_dataobj.leftnright_wavelengths[1, trace_id] = v_right

            where_finite = np.where(np.isfinite(self.bad_pixels[trace_id]))

            regwvs_dataobj.data[trace_id] = np.interp(wv_sampling, self.wavelengths[trace_id, wvs_finite[0]],
                                                      self.data[trace_id, wvs_finite[0]], left=np.nan, right=np.nan)
            regwvs_dataobj.noise[trace_id] = np.interp(wv_sampling, self.wavelengths[trace_id, wvs_finite[0]],
                                                       self.noise[trace_id, wvs_finite[0]], left=np.nan, right=np.nan)

    def _save_interpdata_regwvs(self, save_utils, regwvs_dataobj):
        """ Save the interpolation data in a fits file.

        No.    Name      Ver    Type      Cards   Dimensions   Format
          0  PRIMARY       1 PrimaryHDU     268   ()
          1  INTERP_DATA    1 ImageHDU        73   (2197, 2048)   float64
          2  INTERP_ERR    1 ImageHDU         8   (2197, 2048)   float64
          3  INTERP_X      1 ImageHDU         8   (2197, 2048)   float64
          4  INTERP_Y      1 ImageHDU         8   (2197, 2048)   float64
          5  INTERP_WAVE    1 ImageHDU         8   (2197, 2048)   float64
          6  INTERP_BADPIX    1 ImageHDU         8   (2197, 2048)   float64
          7  INTERP_AREA2D    1 ImageHDU         8   (2197, 2048)   float64
          8  INTERP_LEFTNRIGHT    1 ImageHDU         9   (2197, 2048, 2)   float64
          9  BREADS        1 ImageHDU        21   ()
        """
        if isinstance(save_utils, str):
            out_filename = save_utils
        else:
            out_filename = self.default_filenames["compute_interpdata_regwvs"]
        if bool(self.breads_header["DATA_HPF"]) and not out_filename.endswith("_starsub.fits"):
            out_filename = out_filename.replace(".fits", "_starsub.fits")

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(header=self.priheader))
        hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.data.transpose(),header=self.extheader, name='INTERP_DATA'))
        hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.noise.transpose(), name='INTERP_ERR'))
        hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.x.transpose(), name='INTERP_X'))
        hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.y.transpose(), name='INTERP_Y'))
        hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.wavelengths.transpose(), name='INTERP_WAVE'))
        hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.bad_pixels.transpose(), name='INTERP_BADPIX'))
        hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.area2d.transpose(), name='INTERP_AREA2D'))
        hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.leftnright_wavelengths.transpose(), name='INTERP_LEFTNRIGHT'))
        hdulist.append(pyfits.ImageHDU(header=self.breads_header,name="BREADS"))
        hdulist.writeto(out_filename, overwrite=True)
        hdulist.close()


    def reload_interpdata_regwvs(self, load_filename=None):
        """ Reload interpolated data onto regular wavelengths

        Parameters
        ----------
        load_filename

        Returns
        -------

        """
        if "regwvs" in self.breads_header['COORDS']:
            raise Exception("This data object is already interpolated. Won't interpolate again.")

        if load_filename is None:
            load_filename = self.default_filenames["compute_interpdata_regwvs"]
        if self.breads_header["DATA_HPF"] and not load_filename.endswith("_starsub.fits"):
            load_filename = load_filename.replace(".fits", "_starsub.fits")
        if len(glob(load_filename)) ==0:
            return None

        with pyfits.open(load_filename) as hdulist:
            self.data = hdulist['INTERP_DATA'].data.transpose()
            self.noise = hdulist['INTERP_ERR'].data.transpose()
            self.x  = hdulist['INTERP_X'].data.transpose()
            self.y = hdulist['INTERP_Y'].data.transpose()
            self.wavelengths  = hdulist['INTERP_WAVE'].data.transpose()
            self.bad_pixels  = hdulist['INTERP_BADPIX'].data.transpose()
            self.area2d = hdulist['INTERP_AREA2D'].data.transpose()
            try:
                self.leftnright_wavelengths = hdulist['INTERP_LEFTNRIGHT'].data.transpose()
            except KeyError:
                pass
            self.breads_header['COORDS'] = hdulist['BREADS'].header['COORDS']
            self.breads_header['DATAUNIT'] = hdulist['BREADS'].header['DATAUNIT']

        self.wv_sampling = np.nanmedian(self.wavelengths, axis=0)

        return self

    def reload_advanced_badpix(self, load_filename=None):
        """ Reload advanced bad pixel map computed by compute_starsubtraction().

        Parameters
        ----------
        load_filename : str or None
            Filename to load PSF subtracted data from, or leave None to use default filename

        Returns
        -------
        bad_pixels
            Also modifies self.bad_pixels

        """
        if load_filename is None:
            load_filename = self.default_filenames["compute_starsubtraction"]
        if len(glob(load_filename)) == 0:
            return None

        hdulist = pyfits.open(load_filename)
        fmderived_bad_pixels = hdulist['BADPIX'].data.transpose()
        hdulist.close()

        self.bad_pixels = self.bad_pixels * fmderived_bad_pixels
        return self.bad_pixels