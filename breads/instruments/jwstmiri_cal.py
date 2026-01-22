import os.path
from copy import copy
from glob import glob
import astropy.io.fits as pyfits

import numpy as np
import stpsf as webbpsf
from astropy.io import fits
from astropy.stats import sigma_clip

from copy import copy, deepcopy
import warnings
from astropy.utils.exceptions import AstropyUserWarning

import breads.utils as utils
from breads.utils import broaden, rotate_coordinates, find_closest_leftnright_elements
from breads.jwst_tools.flat_miri_utils import beta_masking_inverse_slice
from scipy.ndimage import generic_filter
from breads.instruments.jwst_IFUs import JWST_IFUs


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
        self.opmode = "IFU" #only option for MIRI
        super()._init_pipeline(save_utils=save_utils, load_utils=load_utils, preproc_task_list=preproc_task_list)

    def _init_miri_channel_band(self, channel_reduction):
        """Initialize attributes relative to MIRI channel and band.
        Also retrieve the pixelscale from crds and spectral resolution.

        Parameters
        ----------
        channel_reduction : str or int
            Must be set to '1', '2' '3' or '4'. It selects the targeted channel we want to use for the reduction.
        """

        self.band = self.priheader["BAND"]
        self.channel = self.priheader['CHANNEL']

        if type(channel_reduction) is str:
            self.channel_reduction = channel_reduction
        if type(channel_reduction) is int:
            self.channel_reduction = str(channel_reduction)

        if self.channel_reduction not in self.channel:
            raise ValueError(
                f'Channel {self.channel_reduction} set for reduction but the cal files channel {self.channel} do not match')

        self.band_aka = self.channel
        self.band_reduction_aka = self.channel_reduction
        if self.band == 'SHORT':
            self.band_aka += 'A'
            self.band_reduction_aka += 'A'
        elif self.band == 'MEDIUM':
            self.band_aka += 'B'
            self.band_reduction_aka += 'B'
        elif self.band == 'LONG':
            self.band_aka += 'C'
            self.band_reduction_aka += 'C'

        path_photom_crds = os.path.join(self.crds_dir, 'references/jwst/miri/')

        fitsfile_crds = os.listdir(path_photom_crds)
        for file_crds in fitsfile_crds:
            if 'photom' in file_crds:
                hdu_photom = fits.open(os.path.join(path_photom_crds, file_crds))
                hdr_photom = hdu_photom[0].header
                if hdr_photom['DETECTOR'] == 'MIRIMAGE':
                    continue
                if hdr_photom['CHANNEL'] == self.channel and hdr_photom['BAND'] == self.band:
                    area2d_fits = file_crds
                    print(f"Photom file selected: {file_crds}")
                hdu_photom.close()

        self.area2d = fits.open(os.path.join(path_photom_crds, area2d_fits))['PIXSIZ'].data #in steradian
        if channel_reduction == '1' or channel_reduction == '4':
            self.pixelscale = np.sqrt(np.nanmedian(self.area2d[10:-10, 20:500]) * 4.25e10)  # steradian to arcsec
        else:
            self.pixelscale = np.sqrt(np.nanmedian(self.area2d[10:-10, 500:-20]) * 4.25e10)

        self._set_mrs_spectral_resolution()

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
        if wv_ref is not None:
            self.wv_ref = wv_ref
        else:
            self.wv_ref = np.nanmin(self.wavelengths)

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
            model = jwst.datamodels.open(hdulist)

            # Pixel grid
            yy, xx = np.mgrid[0:1024, 0:1032]

            # Vectorized WCS transforms
            self.ra_array, self.dec_array, self.wavelengths = model.meta.wcs.transform('detector', 'world', xx, yy)
            self.alpha, self.beta, _ = model.meta.wcs.transform('detector', 'alpha_beta', xx, yy)
            self.trace_id_map = np.zeros_like(self.data)  # TODO make the actual function to do the trace_id mapping

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

        return self.ra_array, self.dec_array, self.wavelengths, self.area2d



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

        key = (self.band_reduction_aka)

        try:
            self.R = R_table[key]
        except KeyError:
            raise ValueError(f"Invalid channel/band inputs : {key}")

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

        for colid in range(self.bad_pixels.shape[1]):
            col_flux = np.copy(self.data[:, colid])

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                smooth = generic_filter(col_flux, np.nanmedian, size=50)
                col_flux = col_flux / smooth

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyUserWarning)
                clipped_data = sigma_clip(col_flux, sigma=3, maxiters=3)
                new_badpix[clipped_data.mask, colid] = np.nan

        self.bad_pixels *= new_badpix

        if save_utils:
            self._save_med_filt_badpix(save_utils, new_badpix)

        return new_badpix

    def _get_webbpsf_model_inputs(self, image_mask, pixel_scale):
        """Hook for MIRI subclass, returns webbpsf parameters for MRS simulation"""
        miri = webbpsf.MIRI()
        miri.load_wss_opd_by_date(self.priheader["DATE-BEG"])  # Load telescope state as of our observation date
        miri.image_mask = image_mask  # optional: model opaque field stop outside of the IFU aperture
        miri.pixelscale = self.pixelscale  # Optional: set this manually to match the drizzled cube sampling, rather than the default
        return miri

    def insert_psf_model(self, save_utils=False, centroid=None, OWA=None, spectrum_func=None, out_folder="insert_psf",
                         mode=None):
        if not hasattr(self, "webbpsf_interp"):
            raise Exception("WebbPSF not found. Please run compute_quick_webbpsf_model or compute_webbpsf_model first.")

        if mode is None:
            mode = "quick_webbpsf"

        if centroid is None:
            centroid = [0, 0]

        if OWA is None:
            where_finite = np.where(np.isfinite(self.dra_as_array))
        else:
            separation_arr = np.sqrt(self.dra_as_array ** 2 + self.ddec_as_array ** 2)
            where_finite = np.where(np.isfinite(self.dra_as_array) * (separation_arr < OWA))

        _dra_as_array, _ddec_as_array = self.get_sky_coords()
        x = _dra_as_array[where_finite]
        y = _ddec_as_array[where_finite]
        w = self.wavelengths[where_finite]

        if mode == "quick_webbpsf":
            model_vec = self.webbpsf_interp((centroid[0] - x) * self.webbpsf_wv0 / w,
                                            (centroid[1] - y) * self.webbpsf_wv0 / w)
        else:
            raise Exception("Unknown mode {0} to inject PSF".format(mode))

        if spectrum_func is not None:
            model_vec *= spectrum_func(w)

        model_im = np.full(self.data.shape, np.nan)
        model_im[where_finite] = model_vec

        if save_utils:
            if isinstance(save_utils, str):
                out_filename = save_utils
            else:
                if not os.path.exists(os.path.join(self.utils_dir, out_folder)):
                    os.makedirs(os.path.join(self.utils_dir, out_folder))
                out_filename = os.path.join(self.utils_dir, out_folder, os.path.basename(self.filename))

            hdulist_sc = pyfits.open(self.filename)
            bu = self.extheader["BUNIT"].strip()
            if bu == 'MJy':
                hdulist_sc["SCI"].data = model_im
            if bu == 'MJy/sr':
                hdulist_sc["SCI"].data = model_im / self.area2d
            hdulist_sc.writeto(out_filename, overwrite=True)
            hdulist_sc.close()

        if self.data_unit == 'MJy':
            return model_im
        elif self.data_unit == 'MJy/sr':
            return model_im / self.area2d


    def _get_webbpsf_fit_inputs(self):
        """
        Hook for MIRI, returns input for webbPSF fit
        """
        return (
            np.copy(self.bad_pixels).transpose(),
            np.copy(self.data).transpose(),
            np.copy(self.noise).transpose(),
            np.copy(self.dra_as_array).transpose(),
            np.copy(self.ddec_as_array).transpose(),
            np.copy(np.abs(self.wavelengths - self.webbpsf_wv0)).transpose(),
        )

    def _get_starspectrum_input(self, im, im_wvs, err, spec_R_sampling, x_nodes, N_nodes):
        """Hook for MIRI, returns input for continuum normalized star spectrum computation."""
        if im is None:
            im = np.copy(self.data).transpose()
        if im_wvs is None:
            im_wvs = np.copy(self.wavelengths).transpose()
        if err is None:
            err = np.copy(self.noise).transpose()
        if spec_R_sampling is None:
            spec_R_sampling = self.R * 4
        if x_nodes is None:
            x_nodes = np.linspace(np.nanmin(im_wvs), np.nanmax(im_wvs), N_nodes, endpoint=True)
        bad_pixels = np.copy(self.bad_pixels).transpose()

        return im, im_wvs, err, bad_pixels, spec_R_sampling, x_nodes

    def _get_masked_normalized_object(self, continuum, im, err):
        """Hook for MIRI, create a mask to only keep high SNR data before computing continuum normalized star spectrum."""
        continuum = np.copy(continuum)

        mask_brightest_slices = beta_masking_inverse_slice(self.data, self.beta, int(self.channel_reduction),
                                                           N_slices=4)
        mask_brightest_slices[mask_brightest_slices == 0] = np.nan
        continuum *= mask_brightest_slices.transpose()

        continuum[np.where(continuum / err < 50)] = np.nan
        continuum[np.where(np.isnan((self.bad_pixels).transpose()))] = np.nan

        normalized_im = im / continuum
        normalized_err = err / continuum

        return continuum, normalized_im, normalized_err

    def _set_bad_pixels(self, bad_pixels):
        """Hook for MIRI to set the bad pixels map. Used only in JWST_IFUs.compute_starspectrum()."""
        self.bad_pixels = bad_pixels.transpose() #Transposing to be consistent with the parent's class method output return.

    def _save_starspectrum_contnorm(self, save_utils, new_wavelengths, combined_fluxes, combined_errors, spline_cont0, spline_paras0, x_nodes):
        """Save continuum normalized star spectrum results."""
        if isinstance(save_utils, str):
            out_filename = save_utils
        else:
            out_filename = self.default_filenames["compute_starspectrum_contnorm"]

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=new_wavelengths))
        hdulist.append(pyfits.ImageHDU(data=combined_fluxes, name='COM_FLUXES'))
        hdulist.append(pyfits.ImageHDU(data=combined_errors, name='COM_ERRORS'))
        hdulist.append(pyfits.ImageHDU(data=spline_cont0.transpose(), name='SPLINE_CONT0'))
        hdulist.append(pyfits.ImageHDU(data=spline_paras0, name='SPLINE_PARAS0'))
        hdulist.append(pyfits.ImageHDU(data=x_nodes, name='x_nodes'))
        hdulist.writeto(out_filename, overwrite=True)
        hdulist.close()

    def _get_starsub_inputs(self, load_starspectrum_contnorm, im, im_wvs, err):
        """Hook for MIRI, return input for subtracting star spectrum from the data."""
        if load_starspectrum_contnorm is None:
            load_starspectrum_contnorm = self.default_filenames["compute_starspectrum_contnorm"]

        hdulist = pyfits.open(load_starspectrum_contnorm)
        spline_paras0 = hdulist["SPLINE_PARAS0"].data
        hdulist.close()

        wherenan = np.where(np.isnan(spline_paras0))
        reg_mean_map = copy(spline_paras0)

        reg_mean_map[wherenan] = np.tile(np.nanmedian(spline_paras0, axis=1)[:, None], (1, spline_paras0.shape[1]))[wherenan]
        reg_std_map = np.abs(spline_paras0)
        reg_std_map[wherenan] = np.tile(np.nanmax(np.abs(spline_paras0), axis=1)[:, None], (1, spline_paras0.shape[1]))[wherenan]
        reg_std_map = reg_std_map
        reg_std_map = np.clip(reg_std_map, 1e-11, np.inf)

        if im is None:
            im = np.copy(self.data).transpose()
        if im_wvs is None:
            im_wvs = np.copy(self.wavelengths).transpose()
        if err is None:
            err = np.copy(self.noise).transpose()

        bad_pixels = np.copy(self.bad_pixels).transpose()

        return im, im_wvs, err, bad_pixels, reg_mean_map, reg_std_map

    def _save_starsubtraction(self, save_utils, subtracted_im, im, star_model, spline_paras0, starsub_dir):
        """Hook for MIRI, save intermediate output used for the star subtraction from the data."""

        if isinstance(save_utils, str):
            out_filename = save_utils
        else:
            out_filename = self.default_filenames["compute_starsubtraction"]

        subtracted_im = subtracted_im.transpose()
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=subtracted_im))
        hdulist.append(pyfits.ImageHDU(data=im.transpose(), name='IM'))
        hdulist.append(pyfits.ImageHDU(data=star_model.transpose(), name='STARMODEL'))
        hdulist.append(pyfits.ImageHDU(data=self.bad_pixels, name='BADPIX'))
        hdulist.append(pyfits.ImageHDU(data=spline_paras0, name='SPLINE_PARAS0'))
        hdulist.append(pyfits.ImageHDU(data=self.x_nodes, name='x_nodes'))
        hdulist.writeto(out_filename, overwrite=True)

        if starsub_dir is not None:
            if not os.path.exists(os.path.join(self.utils_dir, starsub_dir)):
                os.makedirs(os.path.join(self.utils_dir, starsub_dir))
            hdulist_sc = pyfits.open(self.filename)
            du = self.data_unit
            bu = self.extheader["BUNIT"].strip()
            if du == 'MJy' and bu == 'MJy':
                hdulist_sc["SCI"].data = subtracted_im
            if du == 'MJy/sr' and bu == 'MJy/sr':
                hdulist_sc["SCI"].data = subtracted_im
            if du == 'MJy/sr' and bu == 'MJy':
                hdulist_sc["SCI"].data = subtracted_im * self.area2d
            if du == 'MJy' and bu == 'MJy/sr':
                hdulist_sc["SCI"].data = subtracted_im / self.area2d
            hdulist_sc["DQ"].data[np.where(np.isnan(self.bad_pixels))] = 1
            hdulist_sc.writeto(os.path.join(self.utils_dir, starsub_dir, os.path.basename(self.filename)),
                               overwrite=True)
            hdulist_sc.close()

    def _init_regwvs_obj(self, regwvs_dataobj, Ntraces, Nwv):
        """Hook for MIRI, initialize the interpolated calibrated data over regular wavelength grid object."""
        regwvs_dataobj.dra_as_array = np.full((Nwv, Ntraces), np.nan)
        regwvs_dataobj.ddec_as_array = np.full((Nwv, Ntraces), np.nan)
        regwvs_dataobj.wavelengths = np.full((Nwv, Ntraces), np.nan)
        regwvs_dataobj.leftnright_wavelengths = np.full((2, Nwv, Ntraces), np.nan)
        regwvs_dataobj.data = np.full((Nwv, Ntraces), np.nan)
        regwvs_dataobj.noise = np.full((Nwv, Ntraces), np.nan)
        regwvs_dataobj.bad_pixels = np.full((Nwv, Ntraces), np.nan)
        regwvs_dataobj.area2d = np.full((Nwv, Ntraces), np.nan)

    def _get_interpdata_shapes(self, _data, wv_sampling):
        """Hook for MIRI, return the number of spectral traces and the number of wavelength bins for the interpolated data."""
        Ntraces, Nwv = _data.shape[1], np.size(wv_sampling)
        return Ntraces, Nwv

    def _get_where_finite(self, trace_id):
        """Hook for MIRI, return the index of the spectral trace where the data are not NaN."""
        wvs_finite = np.where(np.isfinite(self.wavelengths[:, trace_id]))
        where_finite = np.where(np.isfinite(self.bad_pixels[:, trace_id]))
        return wvs_finite, where_finite

    def _interpdata_regwvs_trace(self, regwvs_dataobj, wv_sampling, _data, wvs_finite, where_finite, trace_id):
        """Hook for MIRI, method to interpolate the data over regular wavelength grid."""
        if self.channel == '34':
            print("Flipping the wavelength axis for channel 34")
            regwvs_dataobj.dra_as_array[:, trace_id] = np.interp(wv_sampling,
                                                              np.flip(self.wavelengths[wvs_finite[0], trace_id]),
                                                              np.flip(self.dra_as_array[wvs_finite[0], trace_id]),
                                                              left=np.nan,
                                                              right=np.nan)
            regwvs_dataobj.ddec_as_array[:, trace_id] = np.interp(wv_sampling,
                                                               np.flip(self.wavelengths[wvs_finite[0], trace_id]),
                                                               np.flip(self.ddec_as_array[wvs_finite[0], trace_id]),
                                                               left=np.nan,
                                                               right=np.nan)

            regwvs_dataobj.wavelengths[:, trace_id] = wv_sampling
            regwvs_dataobj.area2d[:, trace_id] = np.interp(wv_sampling, np.flip(self.wavelengths[wvs_finite[0], trace_id]),
                                                        np.flip(self.area2d[wvs_finite[0], trace_id]), left=np.nan,
                                                        right=np.nan)
            badpix_mask = np.isfinite(np.flip(self.bad_pixels[:, trace_id])).astype(float)
            regwvs_dataobj.bad_pixels[:, trace_id] = np.interp(wv_sampling,
                                                            np.flip(self.wavelengths[wvs_finite[0], trace_id]),
                                                            np.flip(badpix_mask[wvs_finite]), left=0, right=0)

            # following little section written by chatgpt to find the left and right wavelengths in the original data
            v_left, v_right = find_closest_leftnright_elements(np.flip(self.wavelengths[wvs_finite[0], trace_id]),
                                                               wv_sampling)

            regwvs_dataobj.leftnright_wavelengths[0, :, trace_id] = v_left
            regwvs_dataobj.leftnright_wavelengths[1, :, trace_id] = v_right

            where_finite = np.where(np.isfinite(np.flip(self.bad_pixels[:, trace_id])))

            regwvs_dataobj.data[:, trace_id] = np.interp(wv_sampling, np.flip(self.wavelengths[where_finite[0], trace_id]),
                                                      np.flip(_data[where_finite[0], trace_id]), left=np.nan, right=np.nan)
            regwvs_dataobj.noise[:, trace_id] = np.interp(wv_sampling, np.flip(self.wavelengths[where_finite[0], trace_id]),
                                                       np.flip(self.noise[where_finite[0], trace_id]), left=np.nan,
                                                       right=np.nan)

        else:
            regwvs_dataobj.dra_as_array[:, trace_id] = np.interp(wv_sampling, self.wavelengths[wvs_finite[0], trace_id],
                                                              self.dra_as_array[wvs_finite[0], trace_id], left=np.nan,
                                                              right=np.nan)
            regwvs_dataobj.ddec_as_array[:, trace_id] = np.interp(wv_sampling, self.wavelengths[wvs_finite[0], trace_id],
                                                               self.ddec_as_array[wvs_finite[0], trace_id], left=np.nan,
                                                               right=np.nan)

            regwvs_dataobj.wavelengths[:, trace_id] = wv_sampling
            regwvs_dataobj.area2d[:, trace_id] = np.interp(wv_sampling, self.wavelengths[wvs_finite[0], trace_id],
                                                        self.area2d[wvs_finite[0], trace_id], left=np.nan, right=np.nan)
            badpix_mask = np.isfinite(self.bad_pixels[:, trace_id]).astype(float)
            regwvs_dataobj.bad_pixels[:, trace_id] = np.interp(wv_sampling, self.wavelengths[wvs_finite[0], trace_id],
                                                            badpix_mask[wvs_finite], left=0, right=0)

            # following little section written by chatgpt to find the left and right wavelengths in the original data
            v_left, v_right = find_closest_leftnright_elements(self.wavelengths[wvs_finite[0], trace_id], wv_sampling)

            regwvs_dataobj.leftnright_wavelengths[0, :, trace_id] = v_left
            regwvs_dataobj.leftnright_wavelengths[1, :, trace_id] = v_right

            where_finite = np.where(np.isfinite(self.bad_pixels[:, trace_id]))

            regwvs_dataobj.data[:, trace_id] = np.interp(wv_sampling, self.wavelengths[where_finite[0], trace_id],
                                                      _data[where_finite[0], trace_id], left=np.nan, right=np.nan)
            regwvs_dataobj.noise[:, trace_id] = np.interp(wv_sampling, self.wavelengths[where_finite[0], trace_id],
                                                       self.noise[where_finite[0], trace_id], left=np.nan, right=np.nan)

    def reload_interpdata_regwvs(self, load_filename=None):
        """ Reload interpolated data onto regular wavelengths

        Parameters
        ----------
        load_filename

        Returns
        -------

        """
        if "regwvs" in self.coords:
            raise Exception("This data object is already interpolated. Won't interpolate again.")

        if load_filename is None:
            load_filename = self.default_filenames["compute_interpdata_regwvs"]
        if len(glob(load_filename)) ==0:
            return None
        regwvs_dataobj = deepcopy(self)

        regwvs_dataobj.coords = self.coords + " regwvs"

        with pyfits.open(load_filename) as hdulist:
            regwvs_dataobj.data = hdulist[0].data
            regwvs_dataobj.noise = hdulist['INTERP_ERR'].data
            regwvs_dataobj.dra_as_array  = hdulist['INTERP_RA'].data
            regwvs_dataobj.ddec_as_array = hdulist['INTERP_DEC'].data
            regwvs_dataobj.wavelengths  = hdulist['INTERP_WAVE'].data
            regwvs_dataobj.bad_pixels  = hdulist['INTERP_BADPIX'].data
            regwvs_dataobj.area2d = hdulist['INTERP_AREA2D'].data
            try:
                regwvs_dataobj.leftnright_wavelengths = hdulist['INTERP_LEFTNRIGHT'].data
            except KeyError:
                pass

        regwvs_dataobj.wv_sampling = np.nanmedian(regwvs_dataobj.wavelengths, axis=1)

        return regwvs_dataobj
