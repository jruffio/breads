import os.path
import matplotlib.pyplot as plt
from breads.instruments.instrument import Instrument
import breads.utils as utils
from warnings import warn
from astropy.io import fits
import astropy.io.fits as pyfits
from copy import copy, deepcopy
from breads.utils import broaden
from astropy import units as u
from glob import glob
import itertools
from scipy.interpolate import interp1d
from astropy.stats import sigma_clip
from breads.utils import get_spline_model
from scipy.optimize import lsq_linear
import stpsf as webbpsf
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import curve_fit
# from stdatamodels.jwst import datamodels
from jwst import datamodels
from scipy.signal import convolve2d
import scipy.linalg as la
from scipy.optimize import minimize
from astropy import constants as const
from scipy.stats import median_abs_deviation
from tqdm import tqdm

from breads.utils import rotate_coordinates, find_closest_leftnright_elements
from breads.jwst_tools.fit_miri_psf_centroid import fit_trace
from breads.jwst_tools.flat_miri_utils import beta_masking_inverse_slice
import matplotlib.tri as tri
import numpy as np
from scipy.ndimage import generic_filter
from multiprocessing import Pool
from scipy.interpolate import splev, splrep
from astropy.stats import sigma_clip

import inspect


class JWSTMiri_cal(Instrument):
    def __init__(self, filename=None, channel_reduction='1', utils_dir=None, save_utils=True,
                 load_utils=True,
                 preproc_task_list=None,
                 verbose=True, wv_ref=None):
        """JWST MIRI/MRS 2D calibrated data.


        Parameters
        ----------
        filename
        crds_dir
        utils_dir
        save_utils
        load_utils
        preproc_task_list
        verbose
        """
        super().__init__('jwstnirspec_cal', verbose=verbose)

        self.wv_ref = None
        self.opmode = None
        self.data_unit = None
        self.extheader = None
        self.priheader = None
        self.crds_dir = None
        self.utils_dir = None
        self.filename = None

        if filename is None:
            warning_text = "No data file provided. " + \
                           "Please manually add data or use JWSTMiri_cal.read_data_file()"
            warn(warning_text)
        else:
            self.read_data_file(filename, channel_reduction=channel_reduction, utils_dir=utils_dir,
                                save_utils=save_utils,
                                load_utils=load_utils,
                                preproc_task_list=preproc_task_list, wv_ref=wv_ref)

    def read_data_file(self, filename, channel_reduction='1', utils_dir=None, save_utils=True, load_utils=True,
                       preproc_task_list=None, wv_ref=None):
        """Read JWST MIRI/MRS 2D Cal file.  Also checks validity at the end

        Parameters
        ----------
        filename
        crds_dir
        utils_dir
        save_utils
        load_utils
        preproc_task_list

        Returns
        -------

        """
        if self.verbose:
            print(f"Reading data from {filename}")
        self.filename = filename
        if utils_dir is None:
            self.utils_dir = os.path.dirname(self.filename)
        else:
            self.utils_dir = utils_dir

        self.crds_dir = os.getenv('CUSTOM_CRDS_PATH')

        ## Part 1: Loading information from the FITS file and its header metadata
        hdulist_sc = pyfits.open(self.filename)
        self.priheader = hdulist_sc[0].header
        self.extheader = hdulist_sc[1].header
        self.data_unit = self.extheader["BUNIT"].strip()  # MJy/sr or MJy
        self.detector = self.priheader["DETECTOR"]
        self.band = self.priheader["BAND"]
        self.channel = self.priheader['CHANNEL']
        self.channel_reduction = channel_reduction

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

        path_photom_crds = os.path.join(self.crds_dir, 'references/jwst/miri/', self.band_aka)

        fitsfile_crds = os.listdir(path_photom_crds)
        for file_crds in fitsfile_crds:
            if 'photom' in file_crds:
                sr2d_fits = file_crds

        self.sr2d = fits.open(os.path.join(path_photom_crds, sr2d_fits))['PIXSIZ'].data

        # high level decision flag
        self.opmode = "IFU"
        self.data = hdulist_sc["SCI"].data

        rnoise_var = hdulist_sc["VAR_RNOISE"].data
        pnoise_var = hdulist_sc["VAR_POISSON"].data
        self.noise = np.sqrt(rnoise_var + pnoise_var)

        dq = hdulist_sc["DQ"].data
        self.load_wavelength(hdulist_sc)

        self.mask_channel = np.zeros_like(self.data) + np.nan
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

        ny, nx = self.data.shape
        # hdulist_sc.close()

        # Simplifying bad pixel map following convention in this package as: nan = bad, 1 = good
        self.bad_pixels = np.ones((ny, nx))
        # Pixels marked as "do not use" are marked as bad (nan = bad, 1 = good):
        self.bad_pixels[np.where(untangle_dq(dq, verbose=self.verbose)[0, :, :])] = np.nan
        self.bad_pixels[np.where(np.isnan(self.data))] = np.nan

        # Removing any data with zero noise
        where_zero_noise = np.where(self.noise == 0)
        self.noise[where_zero_noise] = np.nan
        self.bad_pixels[where_zero_noise] = np.nan

        self.east2V2_deg = -(float(self.extheader["ROLL_REF"]) + float(self.extheader["V3I_YANG"]))
        self.bary_RV = 0  # Already corrected in wavecal for JWST. float(self.extheader["VELOSYS"])/1000 # in km/s
        self.R = 2700

        self.default_filenames = {}
        self.default_filenames["compute_med_filt_badpix"] = \
            os.path.join(self.utils_dir, os.path.basename(self.filename).replace(".fits", "_roughbadpix.fits"))
        self.default_filenames["compute_coordinates_arrays"] = \
            os.path.join(self.utils_dir, os.path.basename(self.filename).replace(".fits", "_relcoords.fits"))
        splitbasename = os.path.basename(filename).split("_")
        self.default_filenames["compute_webbpsf_model"] = \
            os.path.join(self.utils_dir,
                         splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[3] + "_webbpsf.fits")
        self.default_filenames["compute_quick_webbpsf_model"] = \
            os.path.join(self.utils_dir,
                         splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[3] + "_quick_webbpsf.fits")
        self.default_filenames["compute_new_coords_from_webbPSFfit"] = \
            os.path.join(self.utils_dir, os.path.basename(self.filename).replace(".fits", "_newcen_wpsf.fits"))
        self.default_filenames["compute_charge_bleeding_mask"] = \
            os.path.join(self.utils_dir, os.path.basename(self.filename).replace(".fits", "_barmask.fits"))
        self.default_filenames["compute_starspectrum_contnorm"] = \
            os.path.join(self.utils_dir, os.path.basename(self.filename).replace(".fits", "_starspec_contnorm.fits"))
        self.default_filenames["compute_starspectrum_contnorm_2dspline"] = \
            os.path.join(self.utils_dir, os.path.basename(self.filename).replace(".fits", "_starspec_2dcontnorm.fits"))
        self.default_filenames["compute_starsubtraction"] = \
            os.path.join(self.utils_dir, os.path.basename(self.filename).replace(".fits", "_starsub.fits"))
        self.default_filenames["compute_starsubtraction_2dspline"] = \
            os.path.join(self.utils_dir, os.path.basename(self.filename).replace(".fits", "_2dstarsub.fits"))
        self.default_filenames["compute_interpdata_regwvs"] = \
            os.path.join(self.utils_dir, os.path.basename(self.filename).replace(".fits", "_regwvs.fits"))

        # Mini "pipeline-like" sequence, running a list of task (ie, methods) specified in preproc_task_list
        if preproc_task_list is None:
            preproc_task_list = []
        for task in preproc_task_list:
            # Each task should be a list containning:
            # task[0] = the name of the class method
            # task[1] = a dictionary with any relevant method arguments (but not including save_utils, see task[2])
            # If not defined, it assumes no parameters are needed (task[1] = {}).
            # task[2] = a boolean saying if the outputs should be saved in the utils folder.
            # Default to class save_utils if not defined for the task.
            # If it is a string instead, it will be saved with the string as the filename.
            # task[3] = a boolean saying if we should attempt to load the data from the utils folder.
            # Default to class load_utils if not defined for the task.
            task_name = task[0]
            if len(task) > 1:
                dict_paras = task[1]
            else:
                dict_paras = {}
            if len(task) > 2:
                save_task = task[2]
            else:
                save_task = save_utils
            if len(task) > 3:
                load_task = task[3]
            else:
                load_task = load_utils

            # If save_task is not a string, the default filename is assumed
            if isinstance(save_task, str):
                task_out_filename = save_task
            else:
                if task_name in self.default_filenames.keys():
                    task_out_filename = self.default_filenames[task_name]
                else:
                    task_out_filename = None

            if load_task and task_out_filename is not None and os.path.exists(task_out_filename):
                # Loading data instead because this task has already been done and it is available in the utils folder.
                if self.verbose:
                    print(f"Loading data for {task_name} cached in {task_out_filename}")

                func = getattr(self, task_name.replace("compute_", "reload_"))
                func(load_filename=task_out_filename)
            else:
                # Run task
                if self.verbose:
                    print(f"Running {task_name} with parameters:")
                    print(f"\t save_utils: {save_task}")
                    for para_name in dict_paras.keys():
                        print(f"\t {para_name}: {dict_paras[para_name]}")

                func = getattr(self, task_name)
                func(save_utils=save_task, **dict_paras)

        self.valid_data_check()
        return

    def load_wavelength(self, hdulist):

        try:
            self.wavelengths = hdulist['WAVELENGTH'].data
            self.ra_array = hdulist['RA_ARRAY'].data
            self.dec_array = hdulist['DEC_ARRAY'].data

        except(Exception) as e:
            model = datamodels.open(hdulist)
            self.wavelengths = np.zeros(self.data.shape) + np.nan
            ny, nx = self.data.shape[0], self.data.shape[1]
            n_cores = 20
            args = [('detector', 'world', j, i) for j in range(nx) for i in range(ny)]
            path = inspect.getfile(model.meta.wcs.transform)
            print("Path du fichier source:", os.path.abspath(path))
            with Pool() as pool:
                results = pool.starmap(model.meta.wcs.transform, args)
            self.ra_array = np.array([res[0] for res in results]).reshape(nx, ny).transpose()
            self.dec_array = np.array([res[1] for res in results]).reshape(nx, ny).transpose()
            self.wavelengths = np.array([res[2] for res in results]).reshape(nx, ny).transpose()

            hdu_ra = fits.ImageHDU(data=self.ra_array)
            hdu_ra.header['EXTNAME'] = 'RA_ARRAY'
            hdulist.append(hdu_ra)
            hdu_dec = fits.ImageHDU(data=self.dec_array)
            hdu_dec.header['EXTNAME'] = 'DEC_ARRAY'
            hdulist.append(hdu_dec)
            hdu_wave = fits.ImageHDU(data=self.wavelengths)
            hdu_wave.header['EXTNAME'] = 'WAVELENGTH'
            hdulist.append(hdu_wave)

            hdulist.writeto(self.filename, overwrite=True)

        print("Wavelength map loaded")

    def compute_med_filt_badpix(self, save_utils=False, window_size=50, mad_threshold=50):
        """ Quick bad pixel identification.
        The data is first high-pass filtered row by row with a median filter with a window size of 50 (window_size)
        pixels. The median absolute deviation (MAP) is then calculated row by row, and any pixel deviating by more than
        50x the MAP are identified as bad.

        Only returns (or save) the newly identified bad pixels, the ones already included in self.bad_pixels won't be in
        new_badpix.
        But this map is automatically applied to self.bad_pixels:  self.bad_pixels *= new_badpix

        Parameters
        ----------
        save_utils : bool
            Save the computed bad pixel map (nans=bad) into the utils directory
            Default filename (set save_utils as a string instead of bool to override filename):
            os.path.join(self.utils_dir, os.path.basename(self.filename).replace(".fits", "_med_filt_badpix.fits"))

        Returns
        -------
        new_badpix : np.array
            nans = bad.
        """
        old_badpix = np.copy(self.bad_pixels)

        if self.verbose:
            print("Initializing row_err and bad_pixels")
        new_badpix = np.ones(self.bad_pixels.shape)

        for colid in range(self.bad_pixels.shape[1]):
            col_flux = np.copy(self.data[:, colid])
            col_flux = col_flux / generic_filter(col_flux, np.nanmedian, size=50)
            clipped_data = sigma_clip(col_flux, sigma=3, maxiters=3)
            new_badpix[clipped_data.mask, colid] = np.nan

        self.bad_pixels *= new_badpix

        if save_utils:
            if isinstance(save_utils, str):
                out_filename = save_utils
            else:
                out_filename = self.default_filenames["compute_med_filt_badpix"]

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=self.bad_pixels))
            hdulist.append(pyfits.ImageHDU(data=self.bad_pixels, name='SELF_BADPIX'))
            hdulist.append(pyfits.ImageHDU(data=old_badpix, name='OLD_BADPIX'))
            try:
                hdulist.writeto(out_filename, overwrite=True)
            except TypeError:
                hdulist.writeto(out_filename, clobber=True)
            hdulist.close()
            if self.verbose:
                print(f"Saved the quick bad pixel map to {out_filename}")
        return new_badpix

    def reload_med_filt_badpix(self, load_filename=None):
        """ Reload and apply bad pixel map from med_filt_badpix.
        Parameters
        ----------
        filename : str
            If filename is None, default is:
            os.path.join(self.utils_dir, os.path.basename(self.filename).replace(".fits", "_med_filt_badpix.fits"))

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
            new_badpix = hdulist[0].data
        self.bad_pixels *= new_badpix
        return new_badpix

    def compute_coordinates_arrays(self, save_utils=False, center_with_targname=True, targname=None, fit_centroid=True):
        """ Determine the relative coordinates in the focal plane relative to the target.
        Compute the coordinates {wavelen, delta_ra, delta_dec, area} for each pixel in a 2D image

        Parameters
        ----------
        save_utils : bool
            Save the computed coordinates into the utils directory

        Returns
        -------
        wavelen_array: in microns
        dra_as_array: in arcsec
        ddec_as_array: in arcsec
        area2d: in arcsec^2

        """
        import jwst.datamodels, jwst.assign_wcs
        from jwst.photom.photom import DataSet
        if self.verbose:
            print(f"Computing coordinates arrays.")

        hdulist = pyfits.open(self.filename)  # open file

        # Compute 2D wavelength and pixel area arrays for the whole image
        # Use WCS to compute RA, Dec for each pixel

        if center_with_targname:
            # Calculate the updated SkyCoord object for the desired date
            if targname is None:
                targname = hdulist[0].header["TARGNAME"]
            host_coord = utils.propagate_coordinates_at_epoch(targname, hdulist[0].header["DATE-OBS"])
            host_ra_deg = host_coord.ra.deg
            host_dec_deg = host_coord.dec.deg
            print("host_ra_deg", host_ra_deg, "host_dec_deg", host_dec_deg)

            if fit_centroid:
                hdu = fits.open(self.filename)
                _, _, offset_ra_arcsec, offset_dec_arcsec = fit_trace(hdu, self.band_reduction_aka, self.crds_dir,
                                                                      everyn=10)
                hdu.close()
                offset_ra_deg = -(host_ra_deg - offset_ra_arcsec / 3600)
                offset_dec_deg = -(host_dec_deg - offset_dec_arcsec / 3600)

                print("ra offset", offset_ra_deg * 3600, "dec offset", offset_dec_deg * 3600)
            else:
                offset_ra_deg, offset_dec_deg = 0, 0

            dra_as_array = (self.ra_array - host_ra_deg - offset_ra_deg) * 3600 * np.cos(np.radians(self.dec_array))
            ddec_as_array = (self.dec_array - host_dec_deg - offset_dec_deg) * 3600
        else:
            dra_as_array = self.ra_array
            ddec_as_array = self.dec_array

        if save_utils:
            if isinstance(save_utils, str):
                out_filename = save_utils
            else:
                out_filename = self.default_filenames["compute_coordinates_arrays"]

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=self.wavelengths))
            hdulist.append(pyfits.ImageHDU(data=dra_as_array, name='DELTA_RA'))
            hdulist.append(pyfits.ImageHDU(data=ddec_as_array, name='DELTA_DEC'))
            hdulist.append(pyfits.ImageHDU(data=self.sr2d, name='AREA2D'))
            try:
                hdulist.writeto(out_filename, overwrite=True)
            except TypeError:
                hdulist.writeto(out_filename, clobber=True)
            hdulist.close()
            if self.verbose:
                print(f"  Saved the computed coordinates arrays to {out_filename}")
        self.dra_as_array, self.ddec_as_array = dra_as_array, ddec_as_array
        self.coords = "sky"

        return self.wavelengths, dra_as_array, ddec_as_array, self.sr2d

    def reload_coordinates_arrays(self, load_filename=None):
        if load_filename is None:
            load_filename = self.default_filenames["compute_coordinates_arrays"]
        if len(glob(load_filename)) == 0:
            return None
        with pyfits.open(load_filename) as hdulist:
            wavelen_array = hdulist[0].data
            dra_as_array = hdulist['DELTA_RA'].data
            ddec_as_array = hdulist['DELTA_DEC'].data
            area2d = hdulist['AREA2D'].data
        self.dra_as_array, self.ddec_as_array, self.area2d = dra_as_array, ddec_as_array, area2d
        self.coords = "sky"
        return wavelen_array, dra_as_array, ddec_as_array, area2d

    def set_coords2ifu(self, load_filename=None):
        ifuX, ifuY = self.getifucoords()
        self.dra_as_array, self.ddec_as_array = ifuX, ifuY
        if "regwvs" in self.coords:
            self.coords = "ifu regwvs"
        else:
            self.coords = "ifu"
        return ifuX, ifuY

    def set_coords2sky(self, load_filename=None):
        dra_as_array, ddec_as_array = self.getskycoords()
        self.dra_as_array, self.ddec_as_array = dra_as_array, ddec_as_array
        if "regwvs" in self.coords:
            self.coords = "sky regwvs"
        else:
            self.coords = "sky"
        return dra_as_array, ddec_as_array

    def convert_MJy_per_sr_to_MJy(self, save_utils=False, data_in_MJy_per_sr=None):
        if data_in_MJy_per_sr is not None:  # TODO peut etre supprimer ces boucles de conditions, le return est la seule chose qui change
            return data_in_MJy_per_sr * self.sr2d
        else:
            if self.data_unit != "MJy/sr":
                raise Exception("Data should in MJy/sr to be converted from MJy/sr to MJy")

            self.data = self.data * (self.sr2d)
            self.noise = self.noise * (self.sr2d)
            self.data_unit = "MJy"
            return self.data, self.noise

    def apply_coords_offset(self, save_utils=False, coords_offset=None):
        """ Offset coordinates in the class:
        self.dra_as_array -= coords_offset[0]
        self.ddec_as_array -= coords_offset[1]

        Can only call this method after compute_coordinates_arrays has been run for dra_as_array/ddec_as_array to be
        defined.
        Load/save feature not applicable here.

        Parameters
        ----------
        save_utils : bool
            Does not do anything
        coords_offset: List
            (offset ra, offset dec) in arcsec

        Returns
        -------
        dra_as_array: in arcsec, new relative RA after offset
        ddec_as_array: in arcsec, new relative declination after offset

        """
        # TODO commenter l'equation polynomiale en jeu
        if coords_offset is None:
            coords_offset = [0, 0]
        if self.verbose:
            print(f"Applying relative coordinate offset {coords_offset}")
        if isinstance(coords_offset[0], list) or isinstance(coords_offset[0], np.ndarray):
            self.dra_as_array -= np.polyval(coords_offset[0], self.wavelengths - np.nanmedian(self.wavelengths))
        else:
            self.dra_as_array -= coords_offset[0]
        if isinstance(coords_offset[1], list) or isinstance(coords_offset[1], np.ndarray):
            self.ddec_as_array -= np.polyval(coords_offset[1], self.wavelengths - np.nanmedian(self.wavelengths))
        else:
            self.ddec_as_array -= coords_offset[1]
        return self.dra_as_array, self.ddec_as_array

    def reload_webbpsf_model(self, load_filename=None):
        """ Reload a previously-computed WebbPSF model PSF from a FITS file

        Parameters
        ----------
        filename : str
            Filename of saved PSF

        Returns
        -------

        """
        if load_filename is None:
            load_filename = self.default_filenames["compute_webbpsf_model"]
        if len(glob(load_filename)) == 0:
            return None

        hdulist = pyfits.open(load_filename)

        wpsfs = hdulist[0].data
        wpsfs_header = hdulist[0].header
        wepsfs = hdulist[1].data
        webbpsf_wvs = hdulist[2].data
        webbpsf_X = hdulist[3].data
        webbpsf_Y = hdulist[4].data
        wpsf_pixelscale = wpsfs_header["PIXELSCL"]
        wpsf_oversample = wpsfs_header["oversamp"]

        if not hasattr(self, "wv_sampling"):
            self.wv_sampling = webbpsf_wvs

        hdulist.close()
        # Need to return a bunch of stuff here:

        self.webbpsf_spaxel_area = (wpsf_pixelscale) ** 2
        psf_wv0_id = np.argmin(np.abs(webbpsf_wvs - np.nanmedian(self.wavelengths)))
        self.webbpsf_im = wepsfs[psf_wv0_id]  # /np.nanmax(wepsfs[psf_wv0_id,:,:])
        self.webbpsf_X = webbpsf_X
        self.webbpsf_Y = webbpsf_Y
        self.webbpsf_wv0 = webbpsf_wvs[psf_wv0_id]
        wX, wY = rotate_coordinates(self.webbpsf_X.flatten(), self.webbpsf_Y.flatten(), -self.east2V2_deg, flipx=True)
        self.webbpsf_interp = CloughTocher2DInterpolator((wX, wY), self.webbpsf_im.flatten(), fill_value=0.0)

        return wpsfs, wpsfs_header, wepsfs, webbpsf_wvs, webbpsf_X, webbpsf_Y, wpsf_oversample, wpsf_pixelscale

    def compute_webbpsf_model(self, image_mask=None, pixelscale=0.1, oversample=10, parallelize=False, wv_sampling=None, fov=6,
                              save_utils=False):
        """ Compute WebbPSF simulated PSFs for the NIRSpec IFU

        Parameters
        ----------
        image_mask : str or None
            image mask to use in webbpsf calculations. Default is None since we generally do not wish the edges of the
            IFU aperture in the simulated PSF
        pixelscale : float
            Pixelscale to use for simulated PSF
        oversample : int
            Oversampling factor
        parallelize : bool
            Use multiprocessing to parallelize operations?
        mppool : multiprocessing.pool
            This must be supplied if parallelize is set to True
        save_utils : bool
            Save in the utils directory

        Returns
        -------

        """

        if self.verbose:
            print("Computing PSFs. This has to iterate over many wavelengths, so is slow.")

        if wv_sampling is None:
            if not hasattr(self, "wv_sampling"):
                self.wv_sampling = self.get_regwvs_sampling()
            wv_sampling = self.wv_sampling

        self.wv_sampling = wv_sampling
        nwavelen = np.size(wv_sampling)
        miri = webbpsf.MIRI()
        print("Loading telescope state as of observation date")
        miri.load_wss_opd_by_date(self.priheader["DATE-BEG"])  # Load telescope state as of our observation date
        print("other stuff")
        miri.image_mask = image_mask  # optional: model opaque field stop outside of the IFU aperture
        miri.pixelscale = pixelscale  # Optional: set this manually to match the drizzled cube sampling, rather than the default
        print("end other stuff")
        if not parallelize:
            outarr_not_created = True
            for wv_id, wv in enumerate(wv_sampling):
                print(
                    f"Current index of wavelength {wv_id}, Current wavelength {wv}, Total number of wavelength {nwavelen}")
                paras = miri, wv, oversample, self.opmode
                out = _get_wpsf_task(paras, fov=fov)

                if outarr_not_created:
                    wpsfs = np.zeros((nwavelen, out[0].shape[0], out[0].shape[1]))
                    wepsfs = np.zeros((nwavelen, out[0].shape[0], out[0].shape[1]))
                    outarr_not_created = False

                wpsfs[wv_id, :, :] = out[0]
                wepsfs[wv_id, :, :] = out[1]
        else:  # Parallelized version
            raise Exception(
                "Parallelized version of compute_webbpsf_model does not work, run with parallelize = False.")

        wepsfs *= oversample ** 2

        halffov_x = pixelscale / oversample * wpsfs.shape[2] / 2.0
        halffov_y = pixelscale / oversample * wpsfs.shape[1] / 2.0
        x = np.linspace(-halffov_x, halffov_x, wpsfs.shape[2], endpoint=True)
        y = np.linspace(-halffov_y, halffov_y, wpsfs.shape[1], endpoint=True)
        webbpsf_X, webbpsf_Y = np.meshgrid(x, y)

        wpsfs_header = {"PIXELSCL": pixelscale, "im_mask": image_mask,
                        "oversamp": oversample, "DATE-BEG": self.priheader["DATE-BEG"]}
        if save_utils:
            if isinstance(save_utils, str):
                out_filename = save_utils
            else:
                out_filename = self.default_filenames["compute_webbpsf_model"]

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=wpsfs, header=pyfits.Header(cards=wpsfs_header)))
            hdulist.append(pyfits.ImageHDU(data=wepsfs, name='EPSFS'))
            hdulist.append(pyfits.ImageHDU(data=wv_sampling, name='WAVELEN'))
            hdulist.append(pyfits.ImageHDU(data=webbpsf_X, name='X'))
            hdulist.append(pyfits.ImageHDU(data=webbpsf_Y, name='Y'))
            try:
                hdulist.writeto(out_filename, overwrite=True)
            except TypeError:
                hdulist.writeto(out_filename, clobber=True)
            hdulist.close()
            if self.verbose:
                print(f"  Saved the computed PSFs to {out_filename}")

        self.webbpsf_spaxel_area = (pixelscale) ** 2
        psf_wv0_id = np.argmin(np.abs(wv_sampling - np.nanmedian(self.wavelengths)))
        self.webbpsf_im = wepsfs[psf_wv0_id]  # /np.nanmax(wepsfs[psf_wv0_id,:,:])
        self.webbpsf_X = webbpsf_X
        self.webbpsf_Y = webbpsf_Y
        self.webbpsf_wv0 = wv_sampling[psf_wv0_id]
        wX, wY = rotate_coordinates(self.webbpsf_X.flatten(), self.webbpsf_Y.flatten(), -self.east2V2_deg, flipx=True)
        self.webbpsf_interp = CloughTocher2DInterpolator((wX, wY), self.webbpsf_im.flatten(), fill_value=0.0)

        return wpsfs, wpsfs_header, wepsfs, wv_sampling, webbpsf_X, webbpsf_Y, oversample, pixelscale

    def reload_quick_webbpsf_model(self, load_filename=None):
        """ Reload a previously-computed quick WebbPSF model PSF from a FITS file

        Parameters
        ----------
        filename : str
            Filename of saved PSF

        Returns
        -------

        """
        if load_filename is None:
            load_filename = self.default_filenames["compute_quick_webbpsf_model"]
        if len(glob(load_filename)) == 0:
            return None

        hdulist = pyfits.open(load_filename)

        wpsfs = hdulist[0].data
        wpsfs_header = hdulist[0].header
        wepsfs = hdulist[1].data
        webbpsf_X = hdulist[2].data
        webbpsf_Y = hdulist[3].data
        wpsf_pixelscale = wpsfs_header["PIXELSCL"]
        wpsf_oversample = wpsfs_header["oversamp"]
        self.webbpsf_wv0 = wpsfs_header["WAVE"]

        hdulist.close()
        # Need to return a bunch of stuff here:

        self.webbpsf_spaxel_area = (wpsf_pixelscale) ** 2
        self.webbpsf_im = wepsfs
        self.webbpsf_X = webbpsf_X
        self.webbpsf_Y = webbpsf_Y
        wX, wY = rotate_coordinates(self.webbpsf_X.flatten(), self.webbpsf_Y.flatten(), -self.east2V2_deg, flipx=True)
        self.webbpsf_interp = CloughTocher2DInterpolator((wX, wY), self.webbpsf_im.flatten(), fill_value=0.0)

        return wpsfs, wpsfs_header, wepsfs, webbpsf_X, webbpsf_Y, wpsf_oversample, wpsf_pixelscale

    def compute_quick_webbpsf_model(self, image_mask=None, pixelscale=0.1, oversample=10, fov=6, save_utils=False):
        """ Compute WebbPSF simulated PSFs at the MEDIAN WAVELENGTH ONLY for the NIRSpec IFU

        Parameters
        ----------
        image_mask : str or None
            image mask to use in webbpsf calculations. Default is None since we generally do not wish the edges of the
            IFU aperture in the simulated PSF
        pixelscale : float
            Pixelscale to use for simulated PSF
        oversample : int
            Oversampling factor
        save_utils : bool
            Save in the utils directory

        Returns
        -------

        """

        if self.verbose:
            print("Computing PSFs. This has to iterate over many wavelengths, so is slow.")

        self.webbpsf_wv0 = np.nanmedian(self.wavelengths)

        miri = webbpsf.MIRI()
        miri.load_wss_opd_by_date(self.priheader["DATE-BEG"])  # Load telescope state as of our observation date
        miri.image_mask = image_mask  # optional: model opaque field stop outside of the IFU aperture
        miri.pixelscale = pixelscale  # Optional: set this manually to match the drizzled cube sampling, rather than the default

        paras = miri, self.webbpsf_wv0, oversample, self.opmode
        out = _get_wpsf_task(paras, fov=fov)
        wpsfs = out[0]
        wepsfs = out[1]

        wepsfs *= oversample ** 2

        # print(psf_array_shape,pixelscale)

        halffov_x = pixelscale / oversample * wpsfs.shape[1] / 2.0
        halffov_y = pixelscale / oversample * wpsfs.shape[0] / 2.0

        x = np.linspace(-halffov_x, halffov_x, wpsfs.shape[1], endpoint=True)
        y = np.linspace(-halffov_y, halffov_y, wpsfs.shape[0], endpoint=True)
        webbpsf_X, webbpsf_Y = np.meshgrid(x, y)

        wpsfs_header = {"PIXELSCL": pixelscale, "im_mask": image_mask,
                        "oversamp": oversample, "DATE-BEG": self.priheader["DATE-BEG"],
                        "WAVE": self.webbpsf_wv0}
        if save_utils:
            if isinstance(save_utils, str):
                out_filename = save_utils
            else:
                out_filename = self.default_filenames["compute_quick_webbpsf_model"]

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=wpsfs, header=pyfits.Header(cards=wpsfs_header)))
            hdulist.append(pyfits.ImageHDU(data=wepsfs, name='EPSFS'))
            hdulist.append(pyfits.ImageHDU(data=webbpsf_X, name='X'))
            hdulist.append(pyfits.ImageHDU(data=webbpsf_Y, name='Y'))
            try:
                hdulist.writeto(out_filename, overwrite=True)
            except TypeError:
                hdulist.writeto(out_filename, clobber=True)
            hdulist.close()
            if self.verbose:
                print(f"  Saved the computed PSFs to {out_filename}")

        self.webbpsf_spaxel_area = (pixelscale) ** 2
        self.webbpsf_im = wepsfs
        self.webbpsf_X = webbpsf_X
        self.webbpsf_Y = webbpsf_Y
        wX, wY = rotate_coordinates(self.webbpsf_X.flatten(), self.webbpsf_Y.flatten(), -self.east2V2_deg, flipx=True)
        self.webbpsf_interp = CloughTocher2DInterpolator((wX, wY), self.webbpsf_im.flatten(), fill_value=0.0)

        return wpsfs, wpsfs_header, wepsfs, webbpsf_X, webbpsf_Y, oversample, pixelscale

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

        _dra_as_array, _ddec_as_array = self.getskycoords()
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

        arcsec2_to_sr = (2. * np.pi / (360. * 3600.)) ** 2

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
                hdulist_sc["SCI"].data = model_im / (self.area2d * arcsec2_to_sr)
            try:
                hdulist_sc.writeto(out_filename, overwrite=True)
            except TypeError:
                hdulist_sc.writeto(out_filename, clobber=True)
            hdulist_sc.close()

        if self.data_unit == 'MJy':
            return model_im
        elif self.data_unit == 'MJy/sr':
            return model_im / (self.area2d * arcsec2_to_sr)

    def compute_new_coords_from_webbPSFfit(self, save_utils=False, IWA=None, OWA=None, apply_offset=True):
        """ Update coordinates after fitting a webbPSF at the median wavelength of the data.
        This is the wavelength at which the WebbPSF was saved in the class.

        It does not interpolate the data at that wavelength, only grabs the closest pixel.

        Parameters
        ----------
        save_utils : bool
            Save in the utils directory

        Returns
        -------

        """
        if IWA is None:
            IWA = 0
        if OWA is None:
            OWA = 1.5

        # rough centroid fit
        fit_cen, fit_angle = True, False
        linear_interp = True
        init_paras = np.array([0, 0])

        mask = copy(self.bad_pixels)
        diff_wv_map = np.abs(self.wavelengths - self.webbpsf_wv0)
        mask[np.where(diff_wv_map > np.nanmedian(self.wavelengths) / self.R)] = np.nan
        allnans_rows = np.where(np.nansum(np.isfinite(diff_wv_map), axis=1) == 0)
        diff_wv_map[allnans_rows, :] = 0
        # diff_wv_map[np.where(np.isnan(diff_wv_map))] = 0
        argmin_ids = np.nanargmin(diff_wv_map, axis=1)
        print(argmin_ids)

        paras = linear_interp, self.webbpsf_im, self.webbpsf_X, self.webbpsf_Y, self.east2V2_deg, True, \
            self.dra_as_array[np.arange(self.data.shape[0]), argmin_ids], \
            self.ddec_as_array[np.arange(self.data.shape[0]), argmin_ids], \
            self.data[np.arange(self.data.shape[0]), argmin_ids], \
            self.noise[np.arange(self.data.shape[0]), argmin_ids], \
            mask[np.arange(self.data.shape[0]), argmin_ids], \
            IWA, OWA, fit_cen, fit_angle, init_paras
        out, _ = _fit_wpsf_task(paras)
        ra_offset, dec_offset, angle_offset = out[0, 2::]

        if save_utils:
            if isinstance(save_utils, str):
                out_filename = save_utils
            else:
                out_filename = self.default_filenames["compute_new_coords_from_webbPSFfit"]

            wpsfs_header = {"RA_CEN": ra_offset, "DEC_CEN": dec_offset, "ANGLE": angle_offset}
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(header=pyfits.Header(cards=wpsfs_header)))
            hdulist.writeto(out_filename, overwrite=True)
            hdulist.close()
            if self.verbose:
                print(f"  Saved the computed PSFs to {out_filename}")

        if apply_offset:
            self.dra_as_array -= ra_offset
            self.ddec_as_array -= dec_offset
        return ra_offset, dec_offset

    def reload_new_coords_from_webbPSFfit(self, load_filename=None, apply_offset=True):
        """ Reapply a previously-computed centroid shift based a WebbPSF fit.

        Parameters
        ----------
        load_filename : str
            Filename of fits file to load

        Returns
        -------

        """
        if load_filename is None:
            load_filename = self.default_filenames["compute_new_coords_from_webbPSFfit"]
        if len(glob(load_filename)) == 0:
            return None

        hdulist = pyfits.open(load_filename)
        ra_offset = hdulist[0].header["RA_CEN"]
        dec_offset = hdulist[0].header["DEC_CEN"]
        angle_offset = hdulist[0].header["ANGLE"]
        hdulist.close()

        if apply_offset:
            self.dra_as_array -= ra_offset
            self.ddec_as_array -= dec_offset
        return ra_offset, dec_offset

    def compute_charge_bleeding_mask(self, save_utils=False, threshold2mask=0.15):
        if self.verbose:
            print(f"Computing charge bleeding mask. Will save to {0}".format(
                self.default_filenames['compute_charge_bleeding_mask']))
        ifuX, ifuY = self.getifucoords()

        bar_mask = np.ones(self.bad_pixels.shape)
        bar_mask[np.where(np.abs(ifuX) < threshold2mask)] = np.nan
        if save_utils:
            if isinstance(save_utils, str):
                out_filename = save_utils
            else:
                out_filename = self.default_filenames["compute_charge_bleeding_mask"]

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=bar_mask))
            try:
                hdulist.writeto(out_filename, overwrite=True)
            except TypeError:
                hdulist.writeto(out_filename, clobber=True)
            hdulist.close()

        # self.bad_pixels *= bar_mask
        return bar_mask  # TODO comprendre la fonction et commenter

    def reload_charge_bleeding_mask(self, load_filename=None):
        if load_filename is None:
            load_filename = self.default_filenames["compute_charge_bleeding_mask"]
        if len(glob(load_filename)) == 0:
            return None

        hdulist = pyfits.open(load_filename)
        bar_mask = hdulist[0].data
        hdulist.close()

        # self.bad_pixels *= bar_mask #TODO uncomment this, debug purpose only for now
        return bar_mask

    def compute_starspectrum_contnorm(self, save_utils=False, im=None, im_wvs=None, err=None, mppool=None,
                                      spec_R_sampling=None, threshold_badpix=10, x_nodes=None, N_nodes=40,
                                      iterative=False, star_hf_subtraction=False):
        if im is None:
            im = self.data
        if im_wvs is None:
            im_wvs = self.wavelengths
        if err is None:
            err = self.noise
        if spec_R_sampling is None:
            spec_R_sampling = self.R * 4  # TODO changer R avant
        if x_nodes is None:
            x_nodes = np.linspace(np.nanmin(im_wvs), np.nanmax(im_wvs), N_nodes, endpoint=True)

        if self.verbose:
            print(f"Computing stellar spectrum (continuum normalized)")

        reg_mean_map0 = np.zeros((self.data.shape[1], np.size(x_nodes)))
        reg_std_map0 = np.zeros((self.data.shape[1], np.size(x_nodes)))
        for colid in range(self.data.shape[1]):
            col = self.data[:, colid]
            col_bp = self.bad_pixels[:, colid]
            if np.nansum(np.isfinite(col * col_bp)) == 0:
                continue
            reg_mean_map0[colid, :] = np.nanmedian(col * col_bp)
            reg_std_map0[colid, :] = reg_mean_map0[colid, :]

        spline_cont0, _, new_badpixs, new_res, spline_paras0 = normalize_columns(im, im_wvs, noise=err,
                                                                                 badpixs=self.bad_pixels,
                                                                                 x_nodes=x_nodes, mypool=mppool,
                                                                                 threshold=threshold_badpix,
                                                                                 regularization=True,
                                                                                 reg_mean_map=reg_mean_map0,
                                                                                 reg_std_map=reg_std_map0)
        if iterative:
            reg_mean_map1 = copy(spline_paras0)
            where_nan = np.where(np.isnan(reg_mean_map1))
            reg_mean_map1[where_nan] = reg_mean_map0[where_nan]
            reg_std_map1 = np.abs(reg_mean_map1)
            spline_cont0, _, new_badpixs, new_res, spline_paras0 = normalize_columns(im, im_wvs, noise=err,
                                                                                     badpixs=new_badpixs,
                                                                                     x_nodes=x_nodes, mypool=mppool,
                                                                                     threshold=threshold_badpix,
                                                                                     regularization=True,
                                                                                     reg_mean_map=reg_mean_map1,
                                                                                     reg_std_map=reg_std_map1)

        continuum = copy(spline_cont0)
        print(int(self.channel_reduction), self.band_aka)

        mask_brightest_slices = beta_masking_inverse_slice(self.data, int(self.channel_reduction), self.band_aka, N_slices=4)
        mask_brightest_slices[mask_brightest_slices==0] = np.nan
        continuum *= mask_brightest_slices

        continuum[np.where(continuum / err < 50)] = np.nan
        # continuum[np.where(continuum < np.median(continuum))] = np.nan
        continuum[np.where(np.isnan(self.bad_pixels))] = np.nan

        normalized_im = im / continuum
        normalized_err = err / continuum

        # normalized_im *= mask_brightest_slices
        # normalized_err *= mask_brightest_slices

        new_wavelengths, combined_fluxes, combined_errors = combine_spectrum(im_wvs.flatten(),
                                                                             normalized_im.flatten(),
                                                                             normalized_err.flatten(),
                                                                             np.nanmedian(im_wvs) / (spec_R_sampling))

        if save_utils:
            if isinstance(save_utils, str):
                out_filename = save_utils
            else:
                out_filename = self.default_filenames["compute_starspectrum_contnorm"]

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=new_wavelengths))
            hdulist.append(pyfits.ImageHDU(data=combined_fluxes, name='COM_FLUXES'))
            hdulist.append(pyfits.ImageHDU(data=combined_errors, name='COM_ERRORS'))
            hdulist.append(pyfits.ImageHDU(data=spline_cont0, name='SPLINE_CONT0'))
            hdulist.append(pyfits.ImageHDU(data=spline_paras0, name='SPLINE_PARAS0'))
            hdulist.append(pyfits.ImageHDU(data=x_nodes, name='x_nodes'))
            hdulist.append(pyfits.ImageHDU(data=im_wvs, name='WAVELENGTHS'))
            hdulist.append(pyfits.ImageHDU(data=normalized_im, name='NORMALIZED_IMAGE'))
            hdulist.append(pyfits.ImageHDU(data=normalized_err, name='NORMALIZED_ERR'))
            hdulist.append(pyfits.ImageHDU(data=err, name='ERR'))
            hdulist.append(pyfits.ImageHDU(data=self.bad_pixels, name='BAD_PIX'))
            hdulist.append(pyfits.ImageHDU(data=new_res, name='RES'))
            hdulist.append(pyfits.ImageHDU(data=im, name='IMAGE'))
            try:
                hdulist.writeto(out_filename, overwrite=True)
            except TypeError:
                hdulist.writeto(out_filename, clobber=True)
            hdulist.close()

        self.x_nodes = x_nodes
        self.star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False, fill_value=1)
        if not star_hf_subtraction:
            print("[WARNING] star_hf_subtraction set to False: star_func interpolation function is set to f=1")
            self.star_func = interp1d(new_wavelengths, np.ones_like(new_wavelengths), kind="linear", bounds_error=False,
                                      fill_value=1)
        return new_wavelengths, combined_fluxes, combined_errors, spline_cont0, spline_paras0, x_nodes

    def reload_starspectrum_contnorm(self, load_filename=None):
        if load_filename is None:
            load_filename = self.default_filenames["compute_starspectrum_contnorm"]
        if len(glob(load_filename)) == 0:
            return None

        hdulist = pyfits.open(load_filename)
        new_wavelengths = hdulist[0].data
        combined_fluxes = hdulist[1].data
        combined_errors = hdulist[2].data
        spline_cont0 = hdulist[3].data
        spline_paras0 = hdulist[4].data
        x_nodes = hdulist[5].data
        hdulist.close()

        self.x_nodes = x_nodes
        self.star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False, fill_value=1)

        return new_wavelengths, combined_fluxes, combined_errors, spline_cont0, spline_paras0, x_nodes

    def compute_starsubtraction(self, save_utils=False, im=None, im_wvs=None, err=None, threshold_badpix=10,
                                mppool=None, starsub_dir="starsub1d", load_starspectrum_contnorm=None):
        if self.verbose:
            print(f"Computing star subtraction.")

        if load_starspectrum_contnorm is None:
            load_filename = self.default_filenames["compute_starspectrum_contnorm"]
            hdulist = pyfits.open(load_filename)
            spline_paras0 = hdulist[4].data
            hdulist.close()
        else:
            hdulist = pyfits.open(load_starspectrum_contnorm)
            spline_paras0 = hdulist[4].data
            hdulist.close()

        wherenan = np.where(np.isnan(spline_paras0))
        reg_mean_map = copy(spline_paras0)
        reg_mean_map[wherenan] = np.tile(np.nanmedian(spline_paras0, axis=1)[:, None], (1, spline_paras0.shape[1]))[
            wherenan]
        reg_std_map = np.abs(spline_paras0)
        reg_std_map[wherenan] = np.tile(np.nanmax(np.abs(spline_paras0), axis=1)[:, None], (1, spline_paras0.shape[1]))[
            wherenan]
        reg_std_map = reg_std_map
        reg_std_map = np.clip(reg_std_map, 1e-11, np.inf)

        if im is None:
            im = self.data
        if im_wvs is None:
            im_wvs = self.wavelengths
        if err is None:
            err = self.noise

        star_model, _, new_badpixs, subtracted_im, spline_paras0 = normalize_columns(im, im_wvs, noise=err,
                                                                                     badpixs=self.bad_pixels,
                                                                                     x_nodes=self.x_nodes,
                                                                                     star_model=self.star_func(im_wvs),
                                                                                     threshold=threshold_badpix,
                                                                                     mypool=mppool,
                                                                                     regularization=True,
                                                                                     reg_mean_map=reg_mean_map,
                                                                                     reg_std_map=reg_std_map)
        self.bad_pixels = self.bad_pixels * new_badpixs
        star_model, _, new_badpixs, subtracted_im, spline_paras0 = normalize_columns(im, im_wvs, noise=err,
                                                                                     badpixs=self.bad_pixels,
                                                                                     x_nodes=self.x_nodes,
                                                                                     star_model=self.star_func(im_wvs),
                                                                                     threshold=threshold_badpix,
                                                                                     mypool=mppool,
                                                                                     regularization=True,
                                                                                     reg_mean_map=reg_mean_map,
                                                                                     reg_std_map=reg_std_map)
        self.bad_pixels = self.bad_pixels * new_badpixs

        subtracted_im[np.where(np.isnan(subtracted_im))] = 0

        if save_utils:
            if isinstance(save_utils, str):
                out_filename = save_utils
            else:
                out_filename = self.default_filenames["compute_starsubtraction"]

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=subtracted_im))
            hdulist.append(pyfits.ImageHDU(data=im, name='IM'))
            hdulist.append(pyfits.ImageHDU(data=star_model, name='STARMODEL'))
            hdulist.append(pyfits.ImageHDU(data=self.bad_pixels, name='BADPIX'))
            hdulist.append(pyfits.ImageHDU(data=spline_paras0, name='SPLINE_PARAS0'))
            hdulist.append(pyfits.ImageHDU(data=self.x_nodes, name='x_nodes'))
            try:
                hdulist.writeto(out_filename, overwrite=True)
            except TypeError:
                hdulist.writeto(out_filename, clobber=True)

            if starsub_dir is not None:
                if not os.path.exists(os.path.join(self.utils_dir, starsub_dir)):
                    print(f"Creating star subtraction folder at {starsub_dir}")
                    os.makedirs(os.path.join(self.utils_dir, starsub_dir))
                else:
                    print(f"Using star subtraction folder: {starsub_dir}")
                hdulist_sc = pyfits.open(self.filename)
                du = self.data_unit
                bu = self.extheader["BUNIT"].strip()
                arcsec2_to_sr = (2. * np.pi / (360. * 3600.)) ** 2
                if du == 'MJy' and bu == 'MJy':
                    hdulist_sc["SCI"].data = subtracted_im
                if du == 'MJy/sr' and bu == 'MJy/sr':
                    hdulist_sc["SCI"].data = subtracted_im
                if du == 'MJy/sr' and bu == 'MJy':
                    hdulist_sc["SCI"].data = subtracted_im * (self.sr2d)
                if du == 'MJy' and bu == 'MJy/sr':
                    hdulist_sc["SCI"].data = subtracted_im / (self.sr2d)
                hdulist_sc["DQ"].data[np.where(np.isnan(self.bad_pixels))] = 1
                try:
                    hdulist_sc.writeto(os.path.join(self.utils_dir, starsub_dir, os.path.basename(self.filename)),
                                       overwrite=True)
                except TypeError:
                    hdulist_sc.writeto(os.path.join(self.utils_dir, starsub_dir, os.path.basename(self.filename)),
                                       clobber=True)
                hdulist_sc.close()
        return subtracted_im, star_model, spline_paras0, self.x_nodes

    def reload_starsubtraction(self, load_filename=None):
        if load_filename is None:
            load_filename = self.default_filenames["compute_starsubtraction"]
        if len(glob(load_filename)) == 0:
            return None

        hdulist = pyfits.open(load_filename)
        subtracted_im = hdulist[0].data
        star_model = hdulist[2].data
        fmderived_bad_pixels = hdulist[3].data
        spline_paras0 = hdulist[4].data
        x_nodes = hdulist[5].data
        hdulist.close()

        self.bad_pixels = self.bad_pixels * fmderived_bad_pixels
        return subtracted_im, star_model, spline_paras0, x_nodes

    def compute_interpdata_regwvs(self, save_utils=False, wv_sampling=None, replace_data=None):
        """Interpolate onto a regular wavelength sampling.

        Parameters
        ----------
        wv_sampling
        modelfit
        out_filename
        load_interpdata_regwvs

        Returns
        -------

        """
        if "regwvs" in self.coords:
            raise Exception("This data object is already interpolated. Won't interpolate again.")

        if wv_sampling is None:
            if not hasattr(self, "wv_sampling"):
                self.wv_sampling = self.get_regwvs_sampling()
            wv_sampling = self.wv_sampling
        else:
            self.wv_sampling = wv_sampling

        # wv_sampling = np.nanmedian(self.wavelengths[3:-3], axis=1)
        # self.wv_sampling = wv_sampling
        Nwv = np.size(wv_sampling)

        regwvs_dataobj = deepcopy(self)

        regwvs_dataobj.coords = self.coords + " regwvs"

        if replace_data is not None:
            _data = replace_data
        else:
            _data = self.data

        regwvs_dataobj.dra_as_array = np.zeros((Nwv, _data.shape[1])) + np.nan
        regwvs_dataobj.ddec_as_array = np.zeros((Nwv, _data.shape[1])) + np.nan
        regwvs_dataobj.wavelengths = np.zeros((Nwv, _data.shape[1])) + np.nan
        regwvs_dataobj.leftnright_wavelengths = np.zeros((2, Nwv, _data.shape[1])) + np.nan
        regwvs_dataobj.data = np.zeros((Nwv, _data.shape[1])) + np.nan
        regwvs_dataobj.noise = np.zeros((Nwv, _data.shape[1])) + np.nan
        regwvs_dataobj.bad_pixels = np.zeros((Nwv, _data.shape[1])) + np.nan
        regwvs_dataobj.area2d = np.zeros((Nwv, _data.shape[1])) + np.nan

        for colid in range(_data.shape[1]):
            wvs_finite = np.where(np.isfinite(self.wavelengths[:, colid]))
            if np.size(wvs_finite[0]) == 0:
                continue
            regwvs_dataobj.dra_as_array[:, colid] = np.interp(wv_sampling, self.wavelengths[wvs_finite[0], colid],
                                                              self.dra_as_array[wvs_finite[0], colid], left=np.nan,
                                                              right=np.nan)
            regwvs_dataobj.ddec_as_array[:, colid] = np.interp(wv_sampling, self.wavelengths[wvs_finite[0], colid],
                                                               self.ddec_as_array[wvs_finite[0], colid], left=np.nan,
                                                               right=np.nan)

            regwvs_dataobj.wavelengths[:, colid] = wv_sampling
            regwvs_dataobj.area2d[:, colid] = np.interp(wv_sampling, self.wavelengths[wvs_finite[0], colid],
                                                        self.area2d[wvs_finite[0], colid], left=np.nan, right=np.nan)
            badpix_mask = np.isfinite(self.bad_pixels[:, colid]).astype(float)
            regwvs_dataobj.bad_pixels[:, colid] = np.interp(wv_sampling, self.wavelengths[wvs_finite[0], colid],
                                                            badpix_mask[wvs_finite], left=0, right=0)

            # following little section written by chatgpt to find the left and right wavelengths in the original data
            v_left, v_right = find_closest_leftnright_elements(self.wavelengths[wvs_finite[0], colid], wv_sampling)

            regwvs_dataobj.leftnright_wavelengths[0, :, colid] = v_left
            regwvs_dataobj.leftnright_wavelengths[1, :, colid] = v_right

            where_finite = np.where(np.isfinite(self.bad_pixels[:, colid]))
            if np.size(where_finite[0]) == 0:
                # print("No ref points")
                continue
            regwvs_dataobj.data[:, colid] = np.interp(wv_sampling, self.wavelengths[where_finite[0], colid],
                                                      _data[where_finite[0], colid], left=np.nan, right=np.nan)
            regwvs_dataobj.noise[:, colid] = np.interp(wv_sampling, self.wavelengths[where_finite[0], colid],
                                                       self.noise[where_finite[0], colid], left=np.nan, right=np.nan)

        where_bad = np.where(regwvs_dataobj.bad_pixels != 1.0)
        regwvs_dataobj.data[where_bad] = np.nan
        regwvs_dataobj.noise[where_bad] = np.nan
        regwvs_dataobj.bad_pixels[where_bad] = np.nan
        regwvs_dataobj.wvs_ori = np.copy(self.wavelengths)

        if save_utils:
            if isinstance(save_utils, str):
                out_filename = save_utils
            else:
                out_filename = self.default_filenames["compute_interpdata_regwvs"]

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=regwvs_dataobj.data))
            hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.noise, name='INTERP_ERR'))
            hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.dra_as_array, name='INTERP_RA'))
            hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.ddec_as_array, name='INTERP_DEC'))
            hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.wavelengths, name='INTERP_WAVE'))
            hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.bad_pixels, name='INTERP_BADPIX'))
            hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.area2d, name='INTERP_AREA2D'))
            hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.leftnright_wavelengths, name='INTERP_LEFTNRIGHT'))
            hdulist.append(pyfits.ImageHDU(data=self.wavelengths, name='WAVE_ORIGINAL'))
            try:
                hdulist.writeto(out_filename, overwrite=True)
            except TypeError:
                hdulist.writeto(out_filename, clobber=True)
            hdulist.close()
        return regwvs_dataobj

    def reload_interpdata_regwvs(self, load_filename=None):
        if "regwvs" in self.coords:
            raise Exception("This data object is already interpolated. Won't interpolate again.")

        if load_filename is None:
            load_filename = self.default_filenames["compute_interpdata_regwvs"]
        if len(glob(load_filename)) == 0:
            return None
        regwvs_dataobj = deepcopy(self)

        regwvs_dataobj.coords = self.coords + " regwvs"

        with pyfits.open(load_filename) as hdulist:
            regwvs_dataobj.data = hdulist[0].data
            regwvs_dataobj.noise = hdulist['INTERP_ERR'].data
            regwvs_dataobj.dra_as_array = hdulist['INTERP_RA'].data
            regwvs_dataobj.ddec_as_array = hdulist['INTERP_DEC'].data
            regwvs_dataobj.wavelengths = hdulist['INTERP_WAVE'].data
            regwvs_dataobj.bad_pixels = hdulist['INTERP_BADPIX'].data
            regwvs_dataobj.area2d = hdulist['INTERP_AREA2D'].data
            regwvs_dataobj.wvs_ori = hdulist['WAVE_ORIGINAL'].data
            try:
                regwvs_dataobj.leftnright_wavelengths = hdulist['INTERP_LEFTNRIGHT'].data
            except:
                pass

        regwvs_dataobj.wv_sampling = np.nanmedian(regwvs_dataobj.wavelengths, axis=1)
        return regwvs_dataobj

    def mask_interp_elements_too_far_from_bin_edges(self, dwv_threshold):
        if "regwvs" not in self.coords:
            raise Exception("'regwvs' in self.coords. This data object needs to be interpolated first.")
        dist_to_bin_edges = np.nanmin(np.abs(self.leftnright_wavelengths - self.wavelengths), axis=0)
        mask = dist_to_bin_edges > dwv_threshold
        self.bad_pixels[np.where(mask)] = np.nan
        return mask

    def getifucoords(self, ras=None, decs=None):
        """ Get IFU coordinates

        Parameters
        ----------
        ras
        decs

        Returns
        -------
        ifuX, ifuY : arrays

        """

        if ras is not None and decs is not None:
            ifuX, ifuY = rotate_coordinates(ras, decs, self.east2V2_deg, flipx=False)
        else:
            if "ifu" in self.coords:
                ifuX, ifuY = self.dra_as_array, self.ddec_as_array
            elif "sky" in self.coords:
                ifuX, ifuY = rotate_coordinates(self.dra_as_array, self.ddec_as_array, self.east2V2_deg, flipx=False)

        return ifuX, ifuY

    def getskycoords(self, ifux=None, ifuy=None):
        """ Get sky coordinates

        Parameters
        ----------
        ras
        decs

        Returns
        -------
        dra_as_array, ddec_as_array : arrays

        """
        if ifux is not None and ifuy is not None:
            dra_as_array, ddec_as_array = rotate_coordinates(ifux, ifuy, -self.east2V2_deg, flipx=False)
        else:
            if "sky" in self.coords:
                dra_as_array, ddec_as_array = self.dra_as_array, self.ddec_as_array
            elif "ifu" in self.coords:
                dra_as_array, ddec_as_array = rotate_coordinates(self.dra_as_array, self.ddec_as_array,
                                                                 -self.east2V2_deg, flipx=False)
        return dra_as_array, ddec_as_array

    def broaden(self, wvs, spectrum, loc=None, mppool=None):
        """ Broaden a spectrum to the resolution of this data object using the resolution attribute (self.R).

        LSF is assumed to be a 1D gaussian.
        The broadening is technically fiber dependent so you need to specify which fiber calibration to use.

        Args:
            wvs: Wavelength sampling of the spectrum to be broadened.
            spectrum: 1D spectrum to be broadened.
            loc: To be ignored. Could be used in the future to specify (x,y) position if field dependent resolution is
                available.
            mypool: Multiprocessing pool to parallelize the code. If None (default), non parallelization is applied.
                E.g. mppool = mp.Pool(processes=10) # 10 is the number processes

        Return:
            Broadened spectrum
        """
        return broaden(wvs, spectrum, self.R, mppool=mppool)

    def get_regwvs_sampling(self):
        """ Get a regular wavelength sampling

        Determines the min and max values, and median step size, for the provided wavelengths array
        Computes a regular grid using that median sampling

        Returns
        -------
        wv_sampling : array
            Even regular wavelength values

        """
        wv_min, wv_max = np.nanmin(self.wavelengths), np.nanmax(self.wavelengths)
        sampling_dw = np.nanmedian(self.wavelengths[:, 1::] - self.wavelengths[:, 0:self.wavelengths.shape[1] - 1])
        wv_sampling = np.arange(wv_min, wv_max, sampling_dw)
        return wv_sampling

    def where_point_source(self, radec_as, rad_as):
        return where_point_source(self, radec_as, rad_as)


##########################FUNCTIONS###################################

def _get_wpsf_task(paras, fov=6):
    """ Run WebbPSF for a single wavelength. Utility function for compute_webbpsf_model."""
    miri, center_wv, wpsf_oversample, opmode = paras
    if opmode == 'IFU':
        kernel = np.ones((wpsf_oversample, wpsf_oversample))
    else:
        raise Exception('OPMODE unknown')

    ext = 'OVERSAMP'
    slicepsf_wv0 = miri.calc_psf(monochromatic=center_wv * 1e-6,  # Wavelength, in **METERS**
                                 fov_arcsec=fov,  # angular size to simulate PSF over
                                 oversample=wpsf_oversample,
                                 # output pixel scale relative to the pixelscale set above
                                 add_distortion=False)  # skip an extra computation step that's not relevant for IFU
    webbpsfim = slicepsf_wv0[ext].data
    smoothed_im = convolve2d(webbpsfim, kernel, mode='same') / wpsf_oversample ** 2
    return webbpsfim, smoothed_im


def untangle_dq(arr, verbose=True):
    """Reshape and unpack Data Quality array from ints using a bitmask to a datacube of individual bits

    Got help from ChatGPT.

    Parameters
    ----------
    arr

    Returns
    -------

    """

    if verbose:
        print("Unpacking data quality bitmasks")
        print("DQ array is of type", arr.dtype)
    # Assume arr is your input numpy array of shape (ny, nx)
    ny, nx = arr.shape

    # Create a new numpy array of shape (32, ny, nx) to hold the cube
    cube = np.zeros((32, ny, nx), dtype=bool)

    # Create a mask array to extract the individual bits of each integer in the input array
    mask = np.array([1 << i for i in range(32)], dtype=arr.dtype)

    # Use NumPy's bitwise AND operator to extract the individual bits of each integer
    bits = (arr[..., np.newaxis] & mask[np.newaxis, np.newaxis, :]) > 0

    # Transpose the bits array and assign it to the first dimension of the cube array
    cube[:, :, :] = bits.transpose(2, 0, 1)
    return cube


def _task_normcolumns(paras):
    im_rows, im_wvs_rows, noise_rows, badpix_rows, x_nodes, star_model, threshold, star_sub_mode, regularization, reg_mean_map, reg_std_map = paras

    new_im_rows = np.array(copy(im_rows), '<f4')  # .byteswap().newbyteorder()
    new_noise_rows = copy(noise_rows)
    new_badpix_rows = copy(badpix_rows)
    res = np.zeros(im_rows.shape) + np.nan
    paras_out = np.zeros((im_rows.shape[1], np.size(x_nodes))) + np.nan
    plot = True
    for k in range(im_rows.shape[1]):
        M_spline = get_spline_model(x_nodes, im_wvs_rows[:, k], spline_degree=3)

        where_data_finite = np.where(np.isfinite(badpix_rows[:, k]) * np.isfinite(im_rows[:, k]) * \
                                     np.isfinite(noise_rows[:, k]) * (noise_rows[:, k] != 0) * \
                                     np.isfinite(star_model[:, k]))

        where_badpix = np.where(np.isfinite(badpix_rows[:, k]))
        where_nan_im = np.where(np.isfinite(im_rows[:, k]))
        where_noise = np.where(np.isfinite(noise_rows[:, k]))
        where_0 = np.where(noise_rows[:, k] != 0)

        if np.size(where_data_finite[0]) == 0:
            res[:, k] = np.nan
            continue

        d = im_rows[where_data_finite[0], k]
        d_err = noise_rows[where_data_finite[0], k]

        M = M_spline[where_data_finite[0], :] * star_model[where_data_finite[0], k, None]

        if regularization:
            validpara = np.where(np.nansum(M > np.nanmax(M) * 0.00001, axis=0) != 0)
        else:
            validpara = np.where(np.nansum(M > np.nanmax(M) * 0.01, axis=0) != 0)

        if len(validpara[0]) > 0:
            M = M[:, validpara[0]]
        else:
            return new_im_rows, new_noise_rows, new_badpix_rows, res, paras_out

        if regularization:
            d_reg, s_reg = reg_mean_map[k, :], reg_std_map[k, :]
            s_reg = s_reg[validpara]
            d_reg = d_reg[validpara]
            where_reg = np.where(np.isfinite(s_reg))
            s_reg = s_reg[where_reg]
            d_reg = d_reg[where_reg]
            M_reg = np.zeros((np.size(where_reg[0]), M.shape[1]))
            M_reg[np.arange(np.size(where_reg[0])), where_reg[0]] = 1
            M4fit = np.concatenate([M, M_reg], axis=0)
            d4fit = np.concatenate([d, d_reg])
            s4fit = np.concatenate([d_err, s_reg])
        else:
            d4fit, M4fit, s4fit = d, M, d_err

        bounds_min = [-np.inf, ] * M.shape[1]
        bounds_max = [np.inf, ] * M.shape[1]
        try:
            p = lsq_linear(M4fit / s4fit[:, None], d4fit / s4fit, bounds=(bounds_min, bounds_max)).x
            paras_out[k, validpara[0]] = p

            m = np.dot(M, p)
            res[where_data_finite[0], k] = d - m
            new_im_rows[where_data_finite[0], k] = m
            new_noise_rows[where_data_finite[0], k] = d_err
            norm_res_row = np.zeros(im_rows.shape[0]) + np.nan
            norm_res_row[where_data_finite] = (d - m) / d_err

            meddev = median_abs_deviation(norm_res_row[where_data_finite])
            where_bad = np.where((np.abs(norm_res_row) / meddev > threshold) | np.isnan(norm_res_row))
            new_badpix_rows[where_bad[0], k] = np.nan

        except(Exception):
            print("exception")
            new_im_rows[:, k] = np.nan
            new_noise_rows[:, k] = np.nan
            res[:, k] = np.nan
            paras_out[k, :] = np.nan

    return new_im_rows, new_noise_rows, badpix_rows, res, paras_out  # TODO change to new_badpix_rows


def normalize_columns(image, im_wvs, noise=None, badpixs=None, star_model=None, nodes=40, mypool=None,
                      threshold=10, star_sub_mode=False, x_nodes=None, regularization=True,
                      reg_mean_map=None, reg_std_map=None):
    """Normalize columns


    Parameters
    ----------
    image
    im_wvs
    noise
    badpixs
    star_model
    nodes
    mypool
    threshold
    star_sub_mode
    x_nodes
    regularization
    reg_mean_map
    reg_std_map

    Returns
    -------

    """
    if noise is None:
        noise = np.ones(image.shape)
    if badpixs is None:
        badpixs = np.ones(image.shape)
    if star_model is None:
        star_model = np.ones(image.shape)
    # print(f"[DEBUG] star model shape for input {image.shape}")
    print(f"[DEBUG] reg_mean_map shape for input start0 {reg_mean_map.shape}")
    if x_nodes is None:
        x_nodes = np.linspace(np.nanmin(im_wvs), np.nanmax(im_wvs), nodes, endpoint=True)

    new_image = copy(image)

    new_noise = copy(noise)
    new_badpixs = copy(badpixs)
    new_res = np.zeros(image.shape) + np.nan
    new_spline_paras = np.zeros((image.shape[1], np.size(x_nodes)))

    # if chunk is too small, don't parallelize
    parallel_flag = True
    if (mypool is not None):
        numthreads = mypool._processes
        chunk_size = image.shape[1] // (3 * numthreads)
        if chunk_size == 0:
            parallel_flag = False

    if (mypool is None) or (parallel_flag == False):
        paras = new_image, im_wvs, new_noise, new_badpixs, x_nodes, star_model, threshold, star_sub_mode, regularization, reg_mean_map, reg_std_map
        outputs = _task_normcolumns(paras)
        new_image, new_noise, new_badpixs, new_res, new_spline_paras = outputs
    else:
        numthreads = mypool._processes
        chunk_size = image.shape[1] // (3 * numthreads)
        N_chunks = image.shape[1] // chunk_size
        col_ids = np.arange(image.shape[1])

        col_indices_list = []
        image_list = []
        wvs_list = []
        noise_list = []
        badpixs_list = []
        starmodel_list = []
        if regularization:
            reg_mean_map_list, reg_std_map_list = [], []
        for k in range(N_chunks - 1):
            _col_valid_pix = col_ids[(k * chunk_size):((k + 1) * chunk_size)]
            col_indices_list.append(_col_valid_pix)

            _new_image = new_image[:, (k * chunk_size):((k + 1) * chunk_size)]
            _im_wvs = im_wvs[:, (k * chunk_size):((k + 1) * chunk_size)]
            _new_noise = new_noise[:, (k * chunk_size):((k + 1) * chunk_size)]
            _new_badpixs = new_badpixs[:, (k * chunk_size):((k + 1) * chunk_size)]
            _star_model = star_model[:, (k * chunk_size):((k + 1) * chunk_size)]
            if regularization:
                reg_mn_chunk = reg_mean_map[(k * chunk_size):((k + 1) * chunk_size), :]
                reg_std_chunk = reg_std_map[(k * chunk_size):((k + 1) * chunk_size), :]

            image_list.append(_new_image)
            wvs_list.append(_im_wvs)
            noise_list.append(_new_noise)
            badpixs_list.append(_new_badpixs)
            starmodel_list.append(_star_model)
            if regularization:
                reg_mean_map_list.append(reg_mn_chunk)
                reg_std_map_list.append(reg_std_chunk)

        _col_valid_pix = col_ids[((N_chunks - 1) * chunk_size):image.shape[1]]
        col_indices_list.append(_col_valid_pix)

        _new_image = new_image[:, ((N_chunks - 1) * chunk_size):image.shape[1]]
        _im_wvs = im_wvs[:, ((N_chunks - 1) * chunk_size):image.shape[1]]
        _new_noise = new_noise[:, ((N_chunks - 1) * chunk_size):image.shape[1]]
        _new_badpixs = new_badpixs[:, ((N_chunks - 1) * chunk_size):image.shape[1]]
        _star_model = star_model[:, ((N_chunks - 1) * chunk_size):image.shape[1]]
        if regularization:
            reg_mn_chunk = reg_mean_map[((N_chunks - 1) * chunk_size):image.shape[1], :]
            reg_std_chunk = reg_std_map[((N_chunks - 1) * chunk_size):image.shape[1], :]

        image_list.append(_new_image)
        wvs_list.append(_im_wvs)
        noise_list.append(_new_noise)
        badpixs_list.append(_new_badpixs)
        starmodel_list.append(_star_model)
        if regularization:
            reg_mean_map_list.append(reg_mn_chunk)
            reg_std_map_list.append(reg_std_chunk)

        if not regularization:
            outputs_list = mypool.map(_task_normcolumns, zip(image_list, wvs_list, noise_list, badpixs_list,
                                                             itertools.repeat(x_nodes),
                                                             starmodel_list,
                                                             itertools.repeat(threshold),
                                                             itertools.repeat(star_sub_mode),
                                                             itertools.repeat(False),
                                                             itertools.repeat(None),
                                                             itertools.repeat(None)))
        else:
            outputs_list = mypool.map(_task_normcolumns, zip(image_list, wvs_list, noise_list, badpixs_list,
                                                             itertools.repeat(x_nodes),
                                                             starmodel_list,
                                                             itertools.repeat(threshold),
                                                             itertools.repeat(star_sub_mode),
                                                             itertools.repeat(regularization),
                                                             reg_mean_map_list, reg_std_map_list))
        for col_indices, outputs in zip(col_indices_list, outputs_list):
            out_im_rows, out_noise_rows, out_badpixs_rows, out_res, spline_paras = outputs
            new_image[:, col_indices] = out_im_rows
            new_noise[:, col_indices] = out_noise_rows
            new_badpixs[:, col_indices] = out_badpixs_rows
            new_res[:, col_indices] = out_res
            new_spline_paras[col_indices, :] = spline_paras

    return new_image, new_noise, new_badpixs, new_res, new_spline_paras


def fit_webbpsf(sc_im, sc_im_wvs, noise, bad_pixels, dra_as_array, ddec_as_array, interpolator, psf_wv0, fix_cen=None):
    """Fit a webbpsf model to an image

    Parameters
    ----------
    sc_im
    sc_im_wvs
    noise
    bad_pixels
    dra_as_array
    ddec_as_array
    interpolator
    psf_wv0
    fix_cen

    Returns
    -------

    """
    wv_min, wv_max = np.nanmin(sc_im_wvs), np.nanmax(sc_im_wvs)
    wv_sampling = np.exp(np.arange(np.log(wv_min), np.log(wv_max), np.log(1 + 0.5 / 2700.)))

    dist2host_as = np.sqrt(dra_as_array ** 2 + ddec_as_array ** 2)

    psfsub_sc_im = np.zeros(sc_im.shape) + np.nan
    psfsub_model_im = np.zeros(sc_im.shape)
    bestfit_paras = np.zeros((4, np.size(wv_sampling))) + np.nan
    for wv_id, left_wv in enumerate(wv_sampling):

        center_wv = left_wv * (1 + 0.25 / 2700)
        right_wv = left_wv * (1 + 0.5 / 2700)
        print(left_wv, center_wv, right_wv, wv_min, wv_max)

        where_fit_mask = np.where(
            np.isfinite(sc_im) * (noise != 0) * (np.isfinite(bad_pixels)) * (left_wv < sc_im_wvs) * (
                    sc_im_wvs < right_wv) * (dist2host_as < 1.0))  # *(dist2host_as>0.5)
        where_sc_mask = np.where(np.isfinite(sc_im) * (noise != 0) * (left_wv < sc_im_wvs) * (sc_im_wvs < right_wv))
        Xfit = dra_as_array[where_fit_mask]
        Yfit = ddec_as_array[where_fit_mask]
        Zfit = sc_im[where_fit_mask]
        Zerr2_fit = (noise[where_fit_mask]) ** 2

        Xsc = dra_as_array[where_sc_mask]
        Ysc = ddec_as_array[where_sc_mask]
        Zsc = sc_im[where_sc_mask]

        if (np.size(where_fit_mask[0]) < 377 / 4) or (np.size(where_sc_mask[0]) < 736 / 2):
            print("Not enough points", wv_id, center_wv, np.size(where_fit_mask[0]), np.size(where_sc_mask[0]))
            bestfit_paras[:, wv_id] = np.array([center_wv, np.nan, np.nan, np.nan])
            psfsub_model_im[where_sc_mask] = np.nan
            psfsub_sc_im[where_sc_mask] = np.nan
            continue

        if fix_cen is None:
            m0 = interpolator(Xfit * psf_wv0 / center_wv, Yfit * psf_wv0 / center_wv)
            a0 = np.nansum(Zfit * m0 / Zerr2_fit) / np.nansum(m0 ** 2 / Zerr2_fit)

            # Define the function to fit
            def myfunc(coords, xc, yc, A):
                _x, _y = coords[0], coords[1]
                znew = A * interpolator(_x - xc, _y - yc)
                return znew

            # Define the initial parameter values for the fit
            p0 = [0, 0, a0]
            # Fit the data to the function
            try:
                params, _ = curve_fit(myfunc, np.array([Xfit * psf_wv0 / center_wv, Yfit * psf_wv0 / center_wv]), Zfit,
                                      p0=p0, method='lm', ftol=1e-6, xtol=1e-6)
            except:
                print("curve_fit failed", wv_id, center_wv, np.size(where_fit_mask[0]), np.size(where_sc_mask[0]))
                bestfit_paras[:, wv_id] = np.array([center_wv, np.nan, np.nan, np.nan])
                psfsub_model_im[where_sc_mask] = np.nan
                psfsub_sc_im[where_sc_mask] = np.nan
                continue
            # Extract the optimized parameter values
            xc, yc, a = params

        else:
            m0 = interpolator((Xfit - fix_cen[0]) * psf_wv0 / center_wv, (Yfit - fix_cen[1]) * psf_wv0 / center_wv)
            a0 = np.nansum(Zfit * m0 / Zerr2_fit) / np.nansum(m0 ** 2 / Zerr2_fit)
            xc, yc, a = 0, 0, a0
        # print(xc, yc, a)
        psfmodel = a * interpolator(Xsc * psf_wv0 / center_wv - xc, Ysc * psf_wv0 / center_wv - yc)
        psfsub_Zsc = Zsc - psfmodel

        bestfit_paras[:, wv_id] = np.array(
            [center_wv, xc * center_wv / psf_wv0, yc * center_wv / psf_wv0, a * interpolator(0, 0)])
        psfsub_model_im[where_sc_mask] = psfmodel
        psfsub_sc_im[where_sc_mask] = psfsub_Zsc

    return bestfit_paras, psfsub_model_im, psfsub_sc_im


def where_point_source(dataobj, radec_as, rad_as):
    ra, dec = radec_as
    dist2pointsource_as = np.sqrt((dataobj.dra_as_array - ra) ** 2 + (dataobj.ddec_as_array - dec) ** 2)
    return np.where(dist2pointsource_as < rad_as)


def PCA_wvs_axis_miri(wavelengths, im, im_err, im_badpixs, bin_size, N_KL=5):
    ny, nx = im.shape
    mask = im / im_err

    new_wvs = np.arange(np.nanmin(wavelengths * im_badpixs), np.nanmax(wavelengths * im_badpixs), bin_size)
    nz = np.size(new_wvs)
    new_im = np.zeros((nz, nx)) + np.nan
    for k in range(nx):

        x = wavelengths[:, k]
        y = im[:, k] / im_err[:, k]
        q = im_badpixs[:, k]
        s = im_err[:, k]

        where_finite = np.where(np.isfinite(q) * np.isfinite(y) * (s != 0.0))
        if np.size(where_finite[0]) < ny // 4:
            continue
        f = interp1d(x[where_finite], y[where_finite], bounds_error=False, fill_value=np.nan, kind="linear")
        new_im[:, k] = f(new_wvs)

    # new_im[np.where(np.sum(np.isfinite(new_im), axis=0) < 100)[0], :] = np.nan #TODO dur de comprendre donc j'ai peut etre inverse les dimensions
    new_im = new_im[:, np.where(np.sum(np.isfinite(new_im), axis=0) != 0)[0]]
    print(f"[DEBUG] shape PCA norm {new_im.shape} {np.nanstd(new_im, axis=0)[:, None].shape} ")
    stds = np.nanstd(new_im, axis=0)
    new_im = new_im / stds[np.newaxis, :]

    where_nan = np.where(np.isnan(new_im))
    new_im[where_nan] = 0

    X = new_im.transpose()
    C = np.cov(X)

    tot_basis = C.shape[0]
    tmp_res_numbasis = np.clip(np.abs(N_KL) - 1, 0,
                               tot_basis - 1)  # clip values, for output consistency we'll keep duplicates
    max_basis = np.max(
        tmp_res_numbasis) + 1  # maximum number of eigenvectors/KL basis we actually need to use/calculate
    evals, evecs = la.eigh(C, subset_by_index=[tot_basis - max_basis, tot_basis - 1])
    evals = np.copy(evals[::-1])
    evecs = np.copy(evecs[:, ::-1], order='F')  # fortran order to improve memory caching in matrix multiplication
    # calculate the KL basis vectors
    kl_basis = np.dot(X.T, evecs)
    kls = kl_basis * (1. / np.sqrt(evals * (nz - 1)))[None, :]  # multiply a value for each row
    print(f"[DEBUG] PCA component shape {kls.shape}")

    return new_wvs, kls


def combine_spectrum(wavelengths, fluxes, errors, bin_size):
    """
    Combines the spectrum by combining the flux values in each bin using a weighted mean.
    Calculates and returns the new combined flux errors.

    :param wavelengths: 1D array of wavelengths.
    :param fluxes: 1D array of fluxes.
    :param errors: 1D array of flux errors.
    :param bin_size: scalar value specifying the wavelength bin size.
    :return: tuple containing three 1D arrays: the new wavelength array, the combined flux values, and the new combined flux errors.
    """

    # Remove NaN values from the input arrays
    nan_mask = np.logical_or(np.isnan(wavelengths), np.isnan(fluxes))
    where_mask = np.where(~nan_mask)
    wavelengths = wavelengths[where_mask]
    fluxes = fluxes[where_mask]
    errors = errors[where_mask]

    # Sort the arrays by wavelength
    sort_indices = np.argsort(wavelengths)
    wavelengths = wavelengths[sort_indices]
    fluxes = fluxes[sort_indices]
    errors = errors[sort_indices]

    print("[DEBUG] combining spectrum", fluxes.shape, np.nanmax(fluxes))
    print("[DEBUG] combining spectrum", wavelengths.shape, np.nanmax(wavelengths))

    # Determine the number of bins
    num_bins = int((wavelengths[-1] - wavelengths[0]) / bin_size) + 1

    # Initialize arrays to store the combined flux values and errors
    combined_fluxes = np.zeros(num_bins)
    combined_errors = np.zeros(num_bins)
    new_wavelengths = np.zeros(num_bins)

    # Loop through each bin
    for i in range(num_bins):
        # Determine the wavelength range for the bin
        bin_start = wavelengths[0] + i * bin_size
        bin_end = bin_start + bin_size

        # Find the flux values and errors that fall within the bin
        bin_mask = np.logical_and(wavelengths >= bin_start, wavelengths < bin_end)
        if np.sum(bin_mask.astype(int)) == 0:
            # Store the combined flux and error in the appropriate arrays
            combined_fluxes[i] = np.nan
            combined_errors[i] = np.nan
            new_wavelengths[i] = bin_start + bin_size / 2.0
            continue
        bin_fluxes = fluxes[bin_mask]
        bin_errors = errors[bin_mask]

        # Do sigma clipping
        tmp_snr = (bin_fluxes - np.nanmedian(bin_fluxes)) / bin_errors
        where_tmp_snr_finite = np.where(np.isfinite(tmp_snr))
        if np.size(where_tmp_snr_finite[0]) == 0:
            combined_fluxes[i] = np.nan
            combined_errors[i] = np.nan
            new_wavelengths[i] = bin_start + bin_size / 2.0
            continue
        mask = np.full(tmp_snr.shape, False, dtype=bool)
        mask[where_tmp_snr_finite] = sigma_clip(tmp_snr[where_tmp_snr_finite], 3, masked=True).mask
        where_valid = np.where(~mask)
        bin_fluxes = bin_fluxes[where_valid]
        bin_errors = bin_errors[where_valid]

        # Calculate the weighted mean of the flux values and errors in the bin
        weights = 1.0 / bin_errors ** 2
        if len(weights) > 0:
            weighted_flux = np.sum(weights * bin_fluxes) / np.sum(weights)
            weighted_error = 1.0 / np.sqrt(np.sum(weights))
        else:
            weighted_flux = np.nan
            weighted_error = np.nan

        # Store the combined flux and error in the appropriate arrays
        combined_fluxes[i] = weighted_flux
        combined_errors[i] = weighted_error
        new_wavelengths[i] = bin_start + bin_size / 2.0

    return new_wavelengths, combined_fluxes, combined_errors


def combine_spectrum_1dspline(wavelengths, fluxes, errors, bin_size, oversampling=10):
    new_wavelengths, combined_fluxes, combined_errors = combine_spectrum(wavelengths, fluxes, errors, bin_size)
    star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False, fill_value=1)
    err_func = interp1d(new_wavelengths, combined_errors, kind="linear", bounds_error=False, fill_value=1)

    tmp = (fluxes - star_func(wavelengths)) / errors
    tmp_std = np.nanstd(tmp)
    where_outliers = np.where(np.abs(tmp) > (5 * tmp_std))
    fluxes[where_outliers] = np.nan

    # Remove NaN values from the input arrays
    nan_mask = np.logical_or(np.isnan(wavelengths), np.isnan(fluxes))
    where_mask = np.where(~nan_mask)
    wavelengths = wavelengths[where_mask]
    fluxes = fluxes[where_mask]
    errors = errors[where_mask]

    # Sort the arrays by wavelength
    sort_indices = np.argsort(wavelengths)
    wavelengths = wavelengths[sort_indices]
    fluxes = fluxes[sort_indices]
    errors = errors[sort_indices]

    spl = splrep(wavelengths, fluxes, k=3, t=new_wavelengths[1:(np.size(new_wavelengths) - 1)], task=-1, s=None,
                 w=1 / errors)

    hd_wvs = np.arange(new_wavelengths[0], new_wavelengths[-1], bin_size / oversampling)
    return hd_wvs, splev(hd_wvs, spl), err_func(hd_wvs), spl
    # return spl


# Define the function to fit
def mycostfunc(paras, _x, _y, data, error, _webbpsf_interp):
    if len(paras) == 2:
        xc, yc = paras
        th = 0
    else:
        xc, yc, th = paras
    _x_diff, _y_diff = rotate_coordinates(_x - xc, _y - yc, -th, flipx=False)
    znew = _webbpsf_interp(_x_diff, _y_diff)
    # znew = psf_func[0](_x - xc, _y - yc,grid=False )
    A = np.nansum(data * znew / error ** 2) / np.nansum(znew ** 2 / error ** 2)
    res = data - A * znew
    chi2 = np.nansum((res / error) ** 2)
    return chi2


def filter_big_triangles(X, Y, max_edge_length):
    points = np.array([X, Y]).T
    # Create triangulation
    triangulation = tri.Triangulation(points[:, 0], points[:, 1])

    # Calculate triangle edge lengths
    edge_lengths = np.linalg.norm(
        points[triangulation.triangles[:, [0, 1, 2, 0]], :] - points[triangulation.triangles[:, [1, 2, 0, 1]], :],
        axis=2)

    # Check maximum edge length constraint
    valid_triangles = np.all(edge_lengths <= max_edge_length, axis=1)

    # Filter out sliver triangles
    filtered_triangles = triangulation.triangles[valid_triangles]

    return filtered_triangles


def _fit_wpsf_task(paras):
    if len(paras) == 16:
        linear_interp, wepsf, wifuX, wifuY, east2V2_deg, flipx, _X, _Y, _Z, _Zerr, _Zbad, IWA, OWA, fit_cen, fit_angle, init_paras = paras
        ann_width, padding, sector_area = None, 0.0, None
    else:
        linear_interp, wepsf, wifuX, wifuY, east2V2_deg, flipx, _X, _Y, _Z, _Zerr, _Zbad, IWA, OWA, fit_cen, fit_angle, init_paras, ann_width, padding, sector_area = paras
    # R = np.sqrt((X)**2+ (Y)**2)
    # Xrav, Yrav, Zrav, Zerrrav, Zbadrav = _X.ravel(), _Y.ravel(), _Z.ravel(), _Zerr.ravel(), _Zbad.ravel()
    _R = np.sqrt((_X - init_paras[0]) ** 2 + (_Y - init_paras[1]) ** 2)
    _PA = np.arctan2(_X - init_paras[0], _Y - init_paras[1]) % (2 * np.pi)

    iterator_sectors = []
    if ann_width is None:
        rad_bounds = [(IWA, OWA)]
    else:
        rad_bounds = [(rmin, rmin + ann_width) for rmin in np.arange(IWA, OWA, ann_width)]
    # rad_bounds = [(1.6, 1.8)]
    # rad_bounds = [(0.5, 3.0)]
    # sector_area = 1000.0
    # print(rad_bounds)
    for [r_min, r_max] in rad_bounds:
        # equivalent to using floor but casting as well
        if ann_width is None:
            curr_sep_N_subsections = 1
        else:
            curr_sep_N_subsections = np.max([int(np.pi * (r_max ** 2 - r_min ** 2) / sector_area), 1])
        # divide annuli into subsections : change method to defined the section. Now identical to parallelized
        dphi = 2 * np.pi / curr_sep_N_subsections
        phi_bounds_list = [[dphi * phi_i, dphi * (phi_i + 1)] for phi_i in
                           range(curr_sep_N_subsections)]
        phi_bounds_list[-1][1] = 2 * np.pi
        # dphi = 2 * np.pi / curr_sep_N_subsections
        # phi_bounds_list = [[dphi * phi_i, dphi * (phi_i + 1)] for phi_i in range(curr_sep_N_subsections)]
        # phi_bounds_list[-1][1] = 2 * np.pi
        # for phi_bound in phi_bounds_list:
        #     print(((r_min,r_max),phi_bound) )
        iterator_sectors.extend([((r_min, r_max), phi_bound) for phi_bound in phi_bounds_list])
    tot_sectors = len(iterator_sectors)

    out_paras = np.zeros((tot_sectors, 5)) + np.nan
    out_model = np.zeros(_Z.shape) + np.nan
    # print(rad_bounds)
    # print(iterator_sectors)
    # exit()
    for sector_id, sector in enumerate(iterator_sectors):
        # if sector_id < 14:
        #     continue
        # print("sector",sector)
        # exit()
        rmin, rmax = sector[0]
        pamin, pamax = sector[1]
        padding2 = padding / 2.0
        if pamin < pamax:
            deltaphi = pamax - pamin + 2 * padding / np.mean([rmin, rmax])
            # deltaphi2 = pamax - pamin + 2 * padding2 / np.mean([rmin, rmax])
        else:
            deltaphi = (2 * np.pi - (pamin - pamax)) + 2 * padding / np.mean([rmin, rmax])
            # deltaphi2 = (2 * np.pi - (pamin - pamax)) + 2 * padding2 / np.mean([rmin, rmax])

        # If the length or the arc is higher than 2*pi, simply pick the entire circle.
        if deltaphi >= 2 * np.pi:
            pamin_pad = 0
            pamax_pad = 2 * np.pi
        else:
            pamin_pad = ((pamin) - padding / np.mean([rmin, rmax])) % (2.0 * np.pi)
            pamax_pad = ((pamax) + padding / np.mean([rmin, rmax])) % (2.0 * np.pi)

        rmin_pad = np.max([rmin - padding, 0.0])
        rmax_pad = rmax + padding

        if pamin_pad < pamax_pad:
            fit_sector = (rmin_pad <= _R) & (_R < rmax_pad) & (pamin_pad <= _PA) & (_PA < pamax_pad) & np.isfinite(
                _Zbad)
        else:
            fit_sector = (rmin_pad <= _R) & (_R < rmax_pad) & ((pamin_pad <= _PA) | (_PA < pamax_pad)) & np.isfinite(
                _Zbad)
        if pamin < pamax:
            sc_sector = (rmin <= _R) & (_R < rmax) & (pamin <= _PA) & (_PA < pamax)  # & np.isfinite(_Zbad)
        else:
            sc_sector = (rmin <= _R) & (_R < rmax) & ((pamin <= _PA) | (_PA < pamax))  # & np.isfinite(_Zbad)

        where_fit = np.where(fit_sector)
        if np.size(where_fit[0]) < 1:
            continue
        X, Y, Z, Zerr, Zbad = _X[where_fit], _Y[where_fit], _Z[where_fit], _Zerr[where_fit], _Zbad[where_fit]

        where_sc = np.where(sc_sector)
        if np.size(where_sc[0]) < 1:
            continue
        Xsc, Ysc = _X[where_sc], _Y[where_sc]

        where_wepsf_finite = np.where(np.isfinite(wepsf) * np.isfinite(wifuX) * np.isfinite(wifuY))

        if np.size(where_wepsf_finite[0]) < 3:
            continue
        wX, wY, wZ = wifuX[where_wepsf_finite], wifuY[where_wepsf_finite], wepsf[where_wepsf_finite]
        wX, wY = rotate_coordinates(wX, wY, -east2V2_deg, flipx=flipx)

        if linear_interp:
            webbpsf_interp = LinearNDInterpolator((wX, wY), wZ, fill_value=0.0)
        else:
            webbpsf_interp = CloughTocher2DInterpolator((wX, wY), wZ, fill_value=0.0)

        if fit_angle:
            p0 = np.array([0.0, 0.0, 0.0])
            simplex_init_steps = np.array([0.05, 0.05, 1 / 1000])
        else:
            p0 = np.array([0.0, 0.0])
            simplex_init_steps = np.array([0.05, 0.05])

        if init_paras is not None:
            p0 = np.array(init_paras)
        m0 = webbpsf_interp(X - p0[0], Y - p0[1])
        a0 = np.nansum(Z * m0 / Zerr ** 2) / np.nansum(m0 ** 2 / Zerr ** 2)
        initial_simplex = np.concatenate([p0[None, :], p0[None, :] + np.diag(simplex_init_steps)], axis=0)

        chi20 = mycostfunc(p0, X, Y, Z, Zerr, webbpsf_interp)
        # Define the initial parameter values for the fit
        # Fit the data to the function
        try:
            if fit_cen:
                out = minimize(mycostfunc, p0, args=(X, Y, Z, Zerr, webbpsf_interp), method="Nelder-Mead", bounds=None,
                               options={"xatol": np.inf, "fatol": chi20 * 1e-12, "maxiter": 5e3,
                                        "initial_simplex": initial_simplex, "disp": False})
                if fit_angle:
                    xc, yc, th = out.x
                    wX, wY = rotate_coordinates(wX, wY, th, flipx=False)
                    webbpsf_interp = CloughTocher2DInterpolator((wX, wY), wZ, fill_value=0.0)
                else:
                    xc, yc = out.x
                    th = 0.0
            else:
                xc, yc, th = p0[0], p0[1], 0
            m0 = webbpsf_interp(X - xc, Y - yc)
            a = np.nansum(Z * m0 / Zerr ** 2) / np.nansum(m0 ** 2 / Zerr ** 2)
        except:
            a, xc, yc, th = np.nan, np.nan, np.nan, np.nan

        out_paras[sector_id, :] = np.array([a0, a, xc, yc, th])
        out_model[where_sc] = a * webbpsf_interp(Xsc - xc, Ysc - yc)
    return out_paras, out_model


def _interp_psf(paras):
    linear_interp, wepsf, wifuX, wifuY, wv_id, east2V2_deg = paras
    # print(wv_id)
    wX, wY, wZ = wifuX.ravel(), wifuY.ravel(), wepsf.flatten()
    wX, wY = rotate_coordinates(wX, wY, -east2V2_deg, flipx=True)

    wherepsffinite = np.where(np.isfinite(wZ))
    wX, wY, wZ = wX[wherepsffinite], wY[wherepsffinite], wZ[wherepsffinite]
    if linear_interp:
        webbpsf_interp = LinearNDInterpolator((wX, wY), wZ, fill_value=0.0)
    else:
        webbpsf_interp = CloughTocher2DInterpolator((wX, wY), wZ, fill_value=0.0)

    return webbpsf_interp


def fitpsf_miri(combdataobj, psfs, psfX, psfY, out_filename=None, IWA=0, OWA=np.inf, mppool=None,
                init_centroid=None, fit_cen=True, fit_angle=False,
                ann_width=None, padding=None, sector_area=None, RDI_folder_suffix=None,
                linear_interp=True, rotate_psf=0.0, flipx=False, psf_spaxel_area=None,
                debug_init=None, debug_end=None):
    """
    Fit a model PSF (psfs, psfX, psfY) to a combined dataset (dataobj_list).

    Args:
        dataobj_list:
        psfs:
        psfX:
        psfY:
        out_filename:
        IWA:
        OWA:
        mppool:
        init_centroid:
        fit_cen:
        fit_angle:
        ann_width:
        padding:
        sector_area:
        RDI_folder_suffix:
        linear_interp:
        rotate_psf:
        flipx:
        psf_spaxel_area:

    Returns:

    """
    if RDI_folder_suffix is None:
        RDI_folder_suffix = ""
    if padding is None:
        padding = 0.0

    print("Make sure interpdata_regwvs was already done ")
    all_interp_ra = combdataobj.dra_as_array
    all_interp_dec = combdataobj.ddec_as_array
    all_interp_wvs = combdataobj.wavelengths
    all_interp_flux = combdataobj.data
    all_interp_err = combdataobj.noise
    all_interp_badpix = combdataobj.bad_pixels
    all_interp_area2d = combdataobj.area2d

    wv_sampling = combdataobj.wv_sampling
    all_interp_flux = all_interp_flux / all_interp_area2d * psf_spaxel_area
    all_interp_err = all_interp_err / all_interp_area2d * psf_spaxel_area

    all_interp_psfmodel = np.zeros(all_interp_flux.shape) + np.nan
    all_interp_psfsub = np.zeros(all_interp_flux.shape) + np.nan

    if init_centroid is None:
        init_paras = np.array([0, 0])
    else:
        init_paras = np.array(init_centroid)

    print(len(glob(out_filename)), out_filename)

    # only process frames with wavelength index between debug_init and debug_end
    if debug_init is None:
        debug_init = 0
    if debug_end is None:
        debug_end = np.size(wv_sampling)
    print(debug_init, debug_end)

    wpsf_angle_offset = 0
    bestfit_coords_defined = False
    if mppool is None:
        print(f"[DEBUG] mpool is None")
        for wv_id, wv in enumerate(wv_sampling):
            if not (wv_id >= debug_init and wv_id < debug_end):
                continue
            print(wv_id, wv, np.size(wv_sampling))
            paras = linear_interp, psfs[wv_id], psfX[wv_id], psfY[wv_id], rotate_psf - wpsf_angle_offset, flipx, \
                all_interp_ra[:, wv_id], all_interp_dec[:, wv_id], all_interp_flux[:, wv_id], all_interp_err[:,
                                                                                              wv_id], all_interp_badpix[
                                                                                                      :, wv_id], \
                IWA, OWA, fit_cen, fit_angle, init_paras, ann_width, padding, sector_area
            out = _fit_wpsf_task(paras)

            if not bestfit_coords_defined:
                bestfit_coords = np.zeros(
                    (out[0].shape[0], np.size(wv_sampling), 5)) + np.nan  # flux_init, flux,ra,dec,angle
                bestfit_coords_defined = True
            bestfit_coords[:, wv_id, :] = out[0]
            all_interp_psfmodel[:, wv_id] = out[1]
            all_interp_psfsub[:, wv_id] = all_interp_flux[:, wv_id] - out[1]

    else:
        print(f"[DEBUG] mpool is not None")
        output_lists = mppool.map(_fit_wpsf_task,
                                  zip(itertools.repeat(linear_interp),
                                      psfs, psfX, psfY,
                                      itertools.repeat(rotate_psf - wpsf_angle_offset),
                                      itertools.repeat(flipx),
                                      all_interp_ra.T[debug_init:debug_end],
                                      all_interp_dec.T[debug_init:debug_end],
                                      all_interp_flux.T[debug_init:debug_end],
                                      all_interp_err.T[debug_init:debug_end],
                                      all_interp_badpix.T[debug_init:debug_end],
                                      itertools.repeat(IWA),
                                      itertools.repeat(OWA),
                                      itertools.repeat(fit_cen),
                                      itertools.repeat(fit_angle),
                                      itertools.repeat(init_paras),
                                      itertools.repeat(ann_width),
                                      itertools.repeat(padding),
                                      itertools.repeat(sector_area)))

        for out_id, out in enumerate(output_lists):
            print(out_id, len(output_lists))
            if not bestfit_coords_defined:
                bestfit_coords = np.zeros(
                    (out[0].shape[0], np.size(wv_sampling), 5)) + np.nan  # flux_init, flux,ra,dec,angle
                bestfit_coords_defined = True
            bestfit_coords[:, debug_init + out_id, :] = out[0]
            all_interp_psfmodel[:, debug_init + out_id] = out[1]
            all_interp_psfsub[:, debug_init + out_id] = all_interp_flux[:, debug_init + out_id] - out[1]

    all_interp_psfsub = all_interp_psfsub * all_interp_area2d / psf_spaxel_area
    all_interp_psfmodel = all_interp_psfmodel * all_interp_area2d / psf_spaxel_area
    all_interp_err = all_interp_err * all_interp_area2d / psf_spaxel_area

    if out_filename is not None:
        wpsfsfit_header = {"INIT_ANG": wpsf_angle_offset,
                           "INIT_RA": init_paras[0], "INIT_DEC": init_paras[1]}
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=bestfit_coords, header=pyfits.Header(cards=wpsfsfit_header)))
        try:
            hdulist.writeto(out_filename, overwrite=True)
        except TypeError:
            hdulist.writeto(out_filename, clobber=True)
        hdulist.close()

        RDI_psfsub_dir = os.path.join(os.path.dirname(out_filename), "RDI_psfsub" + RDI_folder_suffix)
        if not os.path.exists(RDI_psfsub_dir):
            os.makedirs(RDI_psfsub_dir)
        RDI_model_dir = os.path.join(os.path.dirname(out_filename), "RDI_model" + RDI_folder_suffix)
        if not os.path.exists(RDI_model_dir):
            os.makedirs(RDI_model_dir)

        for obj_id, filename in enumerate(combdataobj.filelist):
            nx = combdataobj.data.shape[1]
            ny = combdataobj.data.shape[0] // len(combdataobj.filelist)
            interpdata_filename = os.path.join(combdataobj.utils_dir,
                                               os.path.basename(filename).replace(".fits", "_regwvs.fits"))
            _interpdata_psfsub_filename = interpdata_filename.replace(".fits", "_psfsub" + RDI_folder_suffix + ".fits")
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=all_interp_psfsub[(ny * obj_id):(ny * (obj_id + 1)), :]))
            hdulist.append(
                pyfits.ImageHDU(data=all_interp_psfmodel[(ny * obj_id):(ny * (obj_id + 1)), :], name='INTERP_MOD'))
            hdulist.append(
                pyfits.ImageHDU(data=all_interp_err[(ny * obj_id):(ny * (obj_id + 1)), :], name='INTERP_ERR'))
            hdulist.append(pyfits.ImageHDU(data=all_interp_ra[(ny * obj_id):(ny * (obj_id + 1)), :], name='INTERP_RA'))
            hdulist.append(
                pyfits.ImageHDU(data=all_interp_dec[(ny * obj_id):(ny * (obj_id + 1)), :], name='INTERP_DEC'))
            hdulist.append(
                pyfits.ImageHDU(data=all_interp_wvs[(ny * obj_id):(ny * (obj_id + 1)), :], name='INTERP_WAVE'))
            hdulist.append(
                pyfits.ImageHDU(data=all_interp_badpix[(ny * obj_id):(ny * (obj_id + 1)), :], name='INTERP_BADPIX'))
            hdulist.append(
                pyfits.ImageHDU(data=all_interp_area2d[(ny * obj_id):(ny * (obj_id + 1)), :], name='INTERP_AREA2D'))
            try:
                hdulist.writeto(_interpdata_psfsub_filename, overwrite=True)
            except TypeError:
                hdulist.writeto(_interpdata_psfsub_filename, clobber=True)
            hdulist.close()

            hdulist_sc = pyfits.open(filename)
            wvs_ori = combdataobj.wvs_ori  # hdulist_sc["WAVELENGTH"].data
            ny_ori, nx_ori = wvs_ori.shape

            new_model = np.zeros((ny_ori, nx_ori)) + np.nan
            new_badpix = np.zeros((ny_ori, nx_ori)) + np.nan
            new_area2d = np.zeros((ny_ori, nx_ori)) + np.nan
            for colid in range(nx_ori):
                print(f"[BIG DEBUG] psfmodel shape {all_interp_psfmodel.shape}")
                print(
                    f"[DEBUG] shapes for psf fit: {wvs_ori[:, colid].shape}, {wv_sampling.shape}, {all_interp_psfmodel[ny * obj_id:ny * (obj_id + 1), colid].shape}")
                new_model[:, colid] = np.interp(wvs_ori[:, colid], wv_sampling,
                                                all_interp_psfmodel[ny * obj_id:ny * (obj_id + 1), colid], left=np.nan,
                                                right=np.nan)
                badpix_mask = np.isfinite(all_interp_badpix[ny * obj_id:ny * (obj_id + 1), colid]).astype(float)
                new_badpix[:, colid] = np.interp(wvs_ori[:, colid], wv_sampling, badpix_mask, left=np.nan, right=np.nan)
                new_area2d[:, colid] = np.interp(wvs_ori[:, colid], wv_sampling,
                                                 all_interp_area2d[ny * obj_id:ny * (obj_id + 1), colid], left=np.nan,
                                                 right=np.nan)
            where_bad = np.where(new_badpix != 1.0)

            arcsec2_to_sr = (2. * np.pi / (360. * 3600.)) ** 2
            du = combdataobj.data_unit
            bu = hdulist_sc[1].header["BUNIT"].strip()
            if du == 'MJy' and bu == 'MJy':
                pass
            if du == 'MJy/sr' and bu == 'MJy/sr':
                pass
            if du == 'MJy/sr' and bu == 'MJy':
                new_model *= (new_area2d * arcsec2_to_sr)
            if du == 'MJy' and bu == 'MJy/sr':
                new_model /= (new_area2d * arcsec2_to_sr)
            ######

            tmp_sub = hdulist_sc["SCI"].data - new_model
            hdulist_sc["SCI"].data = tmp_sub
            hdulist_sc["DQ"].data[where_bad] = 1
            # Write the new HDU list to a new FITS file

            psfsub_filename = os.path.join(RDI_psfsub_dir, os.path.basename(filename))
            try:
                hdulist_sc.writeto(psfsub_filename, overwrite=True)
            except TypeError:
                hdulist_sc.writeto(psfsub_filename, clobber=True)

            hdulist_sc["SCI"].data = new_model
            psfmod_filename = os.path.join(RDI_model_dir, os.path.basename(filename))
            try:
                hdulist_sc.writeto(psfmod_filename, overwrite=True)
            except TypeError:
                hdulist_sc.writeto(psfmod_filename, clobber=True)

            hdulist_sc.close()


def matchedfilter_bb(fitpsf_filename, dataobj_list, psfs, psfX, psfY, ra_vec, dec_vec, planet_f, out_filename=None,
                     load=True, linear_interp=True, mppool=None, aper_radius=0.5, rv=0):
    print("Make sure interpdata_regwvs was already done ")
    dataobj0 = dataobj_list[0]
    wv_sampling = dataobj0.wv_sampling
    east2V2_deg = dataobj0.east2V2_deg

    comp_spec = planet_f(wv_sampling * (1 - (rv) / const.c.to('km/s').value)) * (u.W / u.m ** 2 / u.um)
    comp_spec = comp_spec * dataobj0.aper_to_epsf_peak_f(wv_sampling)  # normalized to peak flux
    comp_spec = comp_spec * (wv_sampling * u.um) ** 2 / const.c  # from  Flambda to Fnu
    comp_spec = comp_spec.to(u.MJy).value

    ra_grid, dec_grid = np.meshgrid(ra_vec, dec_vec)
    r_grid = np.sqrt(ra_grid ** 2 + dec_grid ** 2)
    PA_grid = np.arctan2(ra_grid, dec_grid) % 2 * np.pi

    flux_map = np.zeros(ra_grid.shape)
    fluxerr_map = np.zeros(ra_grid.shape)

    all_interp_ra, all_interp_dec, all_interp_wvs, all_interp_flux, all_interp_err, all_interp_badpix, all_interp_area2d = \
        dataobj0.interpdata_regwvs(wv_sampling=None, modelfit=False, out_filename=dataobj0.interpdata_regwvs_filename,
                                   load_interpdata_regwvs=True)
    # print(all_interp_ra.shape)
    # plt.scatter(all_interp_ra[:,1000],all_interp_dec[:,1000],s=all_interp_flux[:,1000]/np.nanmedian(all_interp_flux))
    # plt.show()
    if len(dataobj_list) > 1:
        for dataobj in dataobj_list[1::]:
            interp_ra, interp_dec, interp_wvs, interp_flux, interp_err, interp_badpix, interp_area2d = \
                dataobj.interpdata_regwvs(wv_sampling=None, modelfit=False,
                                          out_filename=dataobj.interpdata_regwvs_filename, load_interpdata_regwvs=True)
            # print(all_interp_ra.shape)
            # plt.scatter(interp_ra[:,1000],interp_dec[:,1000],s=interp_flux[:,1000]/np.nanmedian(interp_flux))
            # plt.show()
            all_interp_ra = np.concatenate((all_interp_ra, interp_ra), axis=0)
            all_interp_dec = np.concatenate((all_interp_dec, interp_dec), axis=0)
            all_interp_wvs = np.concatenate((all_interp_wvs, interp_wvs), axis=0)
            all_interp_flux = np.concatenate((all_interp_flux, interp_flux), axis=0)
            all_interp_err = np.concatenate((all_interp_err, interp_err), axis=0)
            all_interp_badpix = np.concatenate((all_interp_badpix, interp_badpix), axis=0)
            all_interp_area2d = np.concatenate((all_interp_area2d, interp_area2d), axis=0)
    with pyfits.open(fitpsf_filename) as hdulist:
        # bestfit_coords = hdulist[0].data
        # wpsf_angle_offset = hdulist[0].header["INIT_ANG"]
        # wpsf_ra_offset = hdulist[0].header["INIT_RA"]
        # wpsf_dec_offset = hdulist[0].header["INIT_DEC"]
        all_interp_psfsub = hdulist[1].data
        # all_interp_psfmodel = hdulist[2].data
    psf_interp_list = []
    print("create psf model")
    # debug_init= 800
    # debug_end = 900
    debug_init = 0
    debug_end = np.size(wv_sampling)
    if 0 or mppool is None:
        for wv_id, wv in enumerate(wv_sampling):
            if not (wv_id > debug_init and wv_id < debug_end):
                psf_interp_list.append(0)
                continue
            print(wv_id, wv, np.size(wv_sampling))
            paras = linear_interp, psfs[wv_id, :, :], psfX[wv_id, :, :], psfY[wv_id, :, :], wv_id, east2V2_deg
            out = _interp_psf(paras)
            # plt.imshow(psfs[wv_id, :, :],origin="lower")
            # plt.imshow(out(ra_grid,dec_grid),origin="lower")
            # plt.show()
            psf_interp_list.append(out)
    else:
        output_lists = mppool.map(_interp_psf, zip(itertools.repeat(linear_interp), psfs[debug_init:debug_end, :, :],
                                                   psfX[debug_init:debug_end, :, :], psfY[debug_init:debug_end, :, :],
                                                   np.arange(np.size(wv_sampling))[debug_init:debug_end],
                                                   itertools.repeat(east2V2_deg)))
        for k in range(debug_init):
            psf_interp_list.append(0)
        for wv_id, (wv, out) in enumerate(zip(wv_sampling, output_lists)):
            print(wv_id, np.size(wv_sampling))
            psf_interp_list.append(out)

        # output_lists = mppool.map(_interp_psf,zip(itertools.repeat(linear_interp),psfs, psfX, psfY,np.arange(np.size(wv_sampling)),itertools.repeat(east2V2_deg)))
        #
        # for wv_id, (wv, out) in enumerate(zip(wv_sampling, output_lists)):
        #     print(wv_id, np.size(wv_sampling))
        #     psf_interp_list.append(out)

    print("done creating psf model")

    for ra_id, ra in enumerate(ra_vec):
        for dec_id, dec in enumerate(dec_vec):
            print(ra, dec)
            sampled_psf = np.zeros(all_interp_flux.shape) + np.nan
            for wv_id, wv in enumerate(wv_sampling):
                # print(wv)
                if not (wv_id > debug_init and wv_id < debug_end):
                    continue
                X = all_interp_ra[:, wv_id]
                Y = all_interp_dec[:, wv_id]
                R = np.sqrt((X - ra) ** 2 + (Y - dec) ** 2)
                where_finite = np.where(
                    np.isfinite(all_interp_badpix[:, wv_id]) * np.isfinite(X) * np.isfinite(Y) * (R < aper_radius))
                X = X[where_finite]
                Y = Y[where_finite]
                sampled_psf[where_finite[0], wv_id] = psf_interp_list[wv_id](X - ra, Y - dec)
                # print(ra,dec)
                # # plt.scatter(X,Y,s=psf_interp_list[wv_id](X,Y)/np.nanmedian(psf_interp_list[wv_id](X,Y)))#,s=sampled_psf[:,wv_id]/np.nanmedian(sampled_psf[:,wv_id])
                # plt.scatter(X,Y,s=sampled_psf[where_finite[0],wv_id]/np.nanmedian(sampled_psf[where_finite[0],wv_id]))#,s=sampled_psf[:,wv_id]/np.nanmedian(sampled_psf[:,wv_id])
                # plt.show()
                # exit()

            sampled_psf = (sampled_psf * comp_spec[None, :]) * all_interp_area2d / dataobj_list[0].webbpsf_spaxel_area

            deno = np.nansum(sampled_psf ** 2 / all_interp_err ** 2)
            mfflux = np.nansum(sampled_psf * all_interp_psfsub / all_interp_err ** 2) / deno
            mffluxerr = 1 / np.sqrt(deno)

            res = all_interp_psfsub - mfflux * sampled_psf
            noise_factor = np.nanstd(res / all_interp_err)

            flux_map[dec_id, ra_id] = mfflux
            fluxerr_map[dec_id, ra_id] = mffluxerr * noise_factor
            # where_finite = np.where(np.isfinite(all_interp_badpix))
            # X = all_interp_ra[where_finite]
            # Y = all_interp_dec

    snr_map = flux_map / fluxerr_map
    if out_filename is not None:
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=flux_map))
        hdulist.append(pyfits.ImageHDU(data=fluxerr_map, name='FLUXERR'))
        hdulist.append(pyfits.ImageHDU(data=snr_map, name='SNR'))
        hdulist.append(pyfits.ImageHDU(data=ra_grid, name='RA'))
        hdulist.append(pyfits.ImageHDU(data=dec_grid, name='DEC'))
        try:
            hdulist.writeto(out_filename, overwrite=True)
        except TypeError:
            hdulist.writeto(out_filename, clobber=True)
        hdulist.close()
    return snr_map, flux_map, fluxerr_map, ra_grid, dec_grid


def _build_cube_task(inputs):
    combdataobj, psf_interp_paras, wv_id, wv, ra_vec, dec_vec, aper_radius, N_pix_min = inputs

    wv_sampling = combdataobj.wv_sampling
    east2V2_deg = combdataobj.east2V2_deg
    all_interp_ra = combdataobj.dra_as_array.transpose()
    all_interp_dec = combdataobj.ddec_as_array.transpose()
    all_interp_flux = combdataobj.data.transpose()
    all_interp_err = combdataobj.noise.transpose()
    all_interp_badpix = combdataobj.bad_pixels.transpose()

    rprint("computing build_cube parallel {} {} {}          ".format(wv_id, wv, np.size(wv_sampling)))
    psf_interp = _interp_psf(psf_interp_paras)

    outs = []
    for ra_id, ra in enumerate(ra_vec):
        for dec_id, dec in enumerate(dec_vec):

            X = all_interp_ra[:, wv_id]
            Y = all_interp_dec[:, wv_id]
            Z = all_interp_flux[:, wv_id]
            Zerr = all_interp_err[:, wv_id]
            R = np.sqrt((X - ra) ** 2 + (Y - dec) ** 2)
            Zerr_masking = Zerr / median_abs_deviation(Zerr[np.where(np.isfinite(Zerr))])
            where_finite = np.where(
                np.isfinite(all_interp_badpix[:, wv_id]) * (Zerr_masking < 5e1) * np.isfinite(X) * np.isfinite(Y) * (
                        R < aper_radius))

            if np.size(where_finite[0]) < N_pix_min:
                outs.append([ra_id, ra, dec_id, dec, np.nan, np.nan])  # changed from continue
            else:
                X = X[where_finite]
                Y = Y[where_finite]
                Z = Z[where_finite]

                Zerr = Zerr[where_finite]
                M = psf_interp(X - ra, Y - dec)

                deno = np.nansum(M ** 2 / Zerr ** 2)
                mfflux = np.nansum(M * Z / Zerr ** 2) / deno
                mffluxerr = 1 / np.sqrt(deno)

                res = Z - mfflux * M
                noise_factor = np.nanstd(res / Zerr)
                outs.append([ra_id, ra, dec_id, dec, mfflux, mffluxerr * noise_factor])
    return outs


import sys


def rprint(string):
    sys.stdout.write('\r' + str(string))
    sys.stdout.flush()


def build_cube(combdataobj, psfs, psfX, psfY, ra_vec, dec_vec, out_filename=None,
               linear_interp=True, mppool=None, aper_radius=0.5,
               debug_init=None, debug_end=None, N_pix_min=None):
    print("[LOG] BUILD CUBE MIRI")
    if "regwvs" not in combdataobj.coords:
        raise Exception(
            "This data object needs to be interpolated on regular wavelength grid. See dataobj.compute_interpdata_regwvs")
    if mppool is not None:
        # raise Exception("Parallelization not implemented yet.")
        # print('passed Exception block. Testing parallelization... setting parallel_flag = True')
        print('Parallelization not supported at the moment for MIRI build cube')
        parallel_flag = False
    else:
        parallel_flag = False
    # plt.imshow(combdataobj.data * combdataobj.bad_pixels, interpolation="nearest", origin="lower")
    # print("coucou")
    # plt.show()
    wv_sampling = combdataobj.wv_sampling
    east2V2_deg = combdataobj.east2V2_deg
    all_interp_ra = combdataobj.dra_as_array.transpose()
    all_interp_dec = combdataobj.ddec_as_array.transpose()
    all_interp_flux = combdataobj.data.transpose()
    all_interp_err = combdataobj.noise.transpose()
    all_interp_badpix = combdataobj.bad_pixels.transpose()
    # plt.imshow(all_interp_flux*all_interp_badpix, interpolation="nearest",origin="lower")
    # plt.show()

    if hasattr(combdataobj, "filelist"):
        N_dithers = len(combdataobj.filelist)
    else:
        N_dithers = 1

    ra_grid, dec_grid = np.meshgrid(ra_vec, dec_vec)

    flux_cube = np.zeros((np.size(wv_sampling), ra_grid.shape[0], ra_grid.shape[1])) + np.nan
    fluxerr_cube = np.zeros((np.size(wv_sampling), ra_grid.shape[0], ra_grid.shape[1])) + np.nan

    print("create psf model")
    # test  = np.where((wv_sampling>3.2465)*(wv_sampling<3.250))
    # test  = np.where((wv_sampling>4.8)*(wv_sampling<4.85))

    # only process frames with wavelength index between debug_init and debug_end
    if debug_init is None:
        debug_init = 0
    if debug_end is None:
        debug_end = np.size(wv_sampling)
    print(debug_init, debug_end)

    if parallel_flag:
        # do the same thing as below just inside a pool
        if N_pix_min is None:
            N_pix_min = (np.pi * aper_radius ** 2 / (0.01) * N_dithers) / 4
        # step 1 prepare list of inputs
        inputs = []
        for wv_id, wv in enumerate(wv_sampling):
            if not (wv_id >= debug_init and wv_id < debug_end):
                continue
            rprint("prepping build_cube inputs {} {} {}          ".format(wv_id, wv, np.size(wv_sampling)))
            psf_interp_paras = linear_interp, psfs[wv_id, :, :], psfX[wv_id, :, :], psfY[wv_id, :,
                                                                                    :], wv_id, east2V2_deg
            inputs.append([combdataobj, psf_interp_paras, wv_id, wv, ra_vec, dec_vec, aper_radius, N_pix_min])
        print()
        # step 2 map _build_cube_task over input list
        print('starting pool.map()')
        outputs = mppool.map(_build_cube_task, inputs)

        # step 3 iterate over outputs and save values
        for j, inp in enumerate(inputs):
            rprint('outputs {} {} {}          '.format(wv_id, wv, np.size(wv_sampling)))

            combdataobj, psf_interp_paras, wv_id, wv, ra_vec, dec_vec, aper_radius, N_pix_min = inp
            outs = outputs[j]
            for o in outs:
                ra_id, ra, dec_id, dec, flux, err = o

                flux_cube[wv_id, dec_id, ra_id] = flux
                fluxerr_cube[wv_id, dec_id, ra_id] = err
        print()

    else:
        print("Doing the job without parallelization")
        if N_pix_min is None:
            N_pix_min = 0  # (np.pi * aper_radius ** 2 / (0.01) * N_dithers) / 4
        for wv_id, wv in enumerate(wv_sampling):
            if not (wv_id >= debug_init and wv_id < debug_end):
                continue
            print("build_cube", wv_id, wv, np.size(wv_sampling))
            psf_interp_paras = linear_interp, psfs[wv_id, :, :], psfX[wv_id, :, :], psfY[wv_id, :,
                                                                                    :], wv_id, east2V2_deg
            psf_interp = _interp_psf(psf_interp_paras)
            print("ra vec", ra_vec)
            for ra_id, ra in enumerate(ra_vec):
                for dec_id, dec in enumerate(dec_vec):
                    print("ra, dec", ra, dec)
                    X = all_interp_ra[:, wv_id]
                    Y = all_interp_dec[:, wv_id]
                    Z = all_interp_flux[:, wv_id]
                    Zerr = all_interp_err[:, wv_id]
                    R = np.sqrt((X - ra) ** 2 + (Y - dec) ** 2)
                    Zerr_masking = Zerr / median_abs_deviation(Zerr[np.where(np.isfinite(Zerr))])
                    where_finite = np.where(
                        np.isfinite(all_interp_badpix[:, wv_id]) * (Zerr_masking < 5e1) * np.isfinite(X) * np.isfinite(
                            Y) * (R < aper_radius))
                    # print(np.size(where_finite[0]),30*N_dithers)
                    # if np.size(where_finite[0]) < 30 * N_dithers:
                    # print(np.size(where_finite[0]), N_pix_min)
                    if np.size(where_finite[0]) < N_pix_min:
                        print("NOT ENOUGHT PIXELS")
                        continue
                    X = X[where_finite]
                    Y = Y[where_finite]
                    Z = Z[where_finite]
                    # Zp=Zp[where_finite]
                    Zerr = Zerr[where_finite]
                    M = psf_interp(X - ra, Y - dec)
                    # # print(wv,ra,dec)
                    # # # plt.scatter(X,Y,s=psf_interp_list[wv_id](X,Y)/np.nanmedian(psf_interp_list[wv_id](X,Y)))#,s=sampled_psf[:,wv_id]/np.nanmedian(sampled_psf[:,wv_id])
                    # plt.figure(10)
                    # plt.subplot(1,2,1)
                    # plt.scatter(X,Y,s=M/np.nanmedian(M))#,s=sampled_psf[:,wv_id]/np.nanmedian(sampled_psf[:,wv_id])
                    # plt.subplot(1,2,2)
                    # plt.scatter(X,Y,s=Z/np.nanmedian(Z))
                    # plt.show()
                    # # plt.figure(11)
                    # #
                    # # # plt.scatter(np.sqrt(X**2+Y**2),M,label="M")
                    # # plt.subplot(1,3,1)
                    # # plt.scatter(np.sqrt(X**2+Y**2),Z,label="Z")
                    # # # plt.subplot(1,3,2)
                    # # # plt.scatter(np.sqrt(X**2+Y**2),Zp,label="Zp")
                    # # plt.subplot(1,3,3)
                    # # plt.scatter(np.sqrt(X**2+Y**2),Zerr,label="Zerr")
                    # # plt.legend()
                    # # plt.show()
                    # # # exit()

                    deno = np.nansum(M ** 2 / Zerr ** 2)
                    mfflux = np.nansum(M * Z / Zerr ** 2) / deno
                    mffluxerr = 1 / np.sqrt(deno)

                    print(f"Interp res for {ra}, {dec}:", deno, mfflux, mffluxerr)

                    res = Z - mfflux * M
                    noise_factor = np.nanstd(res / Zerr)

                    flux_cube[wv_id, dec_id, ra_id] = mfflux
                    fluxerr_cube[wv_id, dec_id, ra_id] = mffluxerr * noise_factor

                # snr_vec = flux_cube[:, dec_id, ra_id] / fluxerr_cube[:, dec_id, ra_id]
                # snr_vec = snr_vec - generic_filter(snr_vec, np.nanmedian, size=50)
                # snr_vec = snr_vec / median_abs_deviation(snr_vec[np.where(np.isfinite(snr_vec))])
                # where_outliers = np.where(snr_vec > 10)
                # flux_cube[where_outliers[0], dec_id, ra_id] = np.nan
                # fluxerr_cube[where_outliers[0], dec_id, ra_id] = np.nan

    if out_filename is not None:
        if debug_init != 0 or debug_end != np.size(wv_sampling):
            out_filename = out_filename.replace(".fits", "_from{0}to{1}.fits".format(debug_init, debug_end))
        print("saving", out_filename)
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=flux_cube))
        hdulist.append(pyfits.ImageHDU(data=fluxerr_cube, name='FLUXERR_CUBE'))
        hdulist.append(pyfits.ImageHDU(data=ra_grid, name='RA'))
        hdulist.append(pyfits.ImageHDU(data=dec_grid, name='DEC'))
        hdulist.append(pyfits.ImageHDU(data=wv_sampling, name='WAVE'))
        try:
            hdulist.writeto(out_filename, overwrite=True)
        except TypeError:
            hdulist.writeto(out_filename, clobber=True)
        hdulist.close()
    return flux_cube, fluxerr_cube, ra_grid, dec_grid

    # dataobj0 = dataobj_list[0]
    # wv_sampling = dataobj0.wv_sampling
    # east2V2_deg = dataobj0.east2V2_deg


def cube_matchedfilter(flux_cube, fluxerr_cube, wv_sampling, ra_grid, dec_grid, planet_f, rv=0, out_filename=None,
                       outlier_threshold=None):
    comp_spec = planet_f(wv_sampling * (1 - (rv) / const.c.to('km/s').value)) * (u.W / u.m ** 2 / u.um)
    # comp_spec = comp_spec * dataobj0.aper_to_epsf_peak_f(wv_sampling)  # normalized to peak flux
    comp_spec = comp_spec * (wv_sampling * u.um) ** 2 / const.c  # from  Flambda to Fnu
    comp_spec = comp_spec.to(u.MJy).value

    ra_vec = ra_grid[0, :]
    dec_vec = dec_grid[:, 0]
    # r_grid = np.sqrt(ra_grid ** 2 + dec_grid ** 2)
    # PA_grid = np.arctan2(ra_grid, dec_grid) % 2 * np.pi

    flux_map = np.zeros(ra_grid.shape) + np.nan
    fluxerr_map = np.zeros(ra_grid.shape) + np.nan

    for ra_id, ra in enumerate(ra_vec):
        print("ra", ra)
        for dec_id, dec in enumerate(dec_vec):
            # if ra <-1.0 or dec<-1.0:
            #     continue
            # if ra >-0.9 or dec>-0.9:
            #     continue

            if outlier_threshold is not None:
                snr_vec = flux_cube[:, dec_id, ra_id] / fluxerr_cube[:, dec_id, ra_id]
                snr_vec = (snr_vec - generic_filter(snr_vec, np.nanmedian, size=50)) / median_abs_deviation(
                    snr_vec[np.where(np.isfinite(snr_vec))])
                where_outliers = np.where(snr_vec > outlier_threshold)
                flux_cube[where_outliers[0], dec_id, ra_id] = np.nan
                fluxerr_cube[where_outliers[0], dec_id, ra_id] = np.nan

            deno = np.nansum(comp_spec ** 2 / fluxerr_cube[:, dec_id, ra_id] ** 2)
            bbflux = np.nansum(comp_spec * flux_cube[:, dec_id, ra_id] / fluxerr_cube[:, dec_id, ra_id] ** 2) / deno
            bbfluxerr = 1 / np.sqrt(deno)

            res = flux_cube[:, dec_id, ra_id] - bbflux * comp_spec
            noise_factor = np.nanstd(res / fluxerr_cube[:, dec_id, ra_id])

            flux_map[dec_id, ra_id] = bbflux
            fluxerr_map[dec_id, ra_id] = bbfluxerr * noise_factor

    snr_map = flux_map / fluxerr_map
    if out_filename is not None:
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=flux_map))
        hdulist.append(pyfits.ImageHDU(data=fluxerr_map, name='FLUXERR'))
        hdulist.append(pyfits.ImageHDU(data=snr_map, name='SNR'))
        hdulist.append(pyfits.ImageHDU(data=ra_grid, name='RA'))
        hdulist.append(pyfits.ImageHDU(data=dec_grid, name='DEC'))
        try:
            hdulist.writeto(out_filename, overwrite=True)
        except TypeError:
            hdulist.writeto(out_filename, clobber=True)
        hdulist.close()
    return snr_map, flux_map, fluxerr_map, ra_grid, dec_grid


def get_contnorm_spec_miri(dataobj_list, out_filename=None, load_utils=False, mppool=None, spec_R_sampling=None,
                           spline2d=False,
                           masking_radius=None, masking_ifu_location=None, interpolation=None):
    if interpolation is None:
        interpolation = "linear"
    if 1 and load_utils and len(glob(out_filename)):
        print(len(glob(out_filename)), out_filename)
        with pyfits.open(out_filename) as hdulist:
            new_wavelengths = hdulist[0].data
            combined_fluxes = hdulist[1].data
            combined_errors = hdulist[2].data
    else:
        wvs_list = []
        normalized_im_list = []
        normalized_err_list = []
        for dataobj in dataobj_list:
            reload_outputs = dataobj.reload_starspectrum_contnorm()
            if reload_outputs is None:
                reload_outputs = dataobj.compute_starspectrum_contnorm(save_utils=False, mppool=mppool)
            new_wavelengths, combined_fluxes, combined_errors, spline_cont0, spline_paras0, x_nodes = reload_outputs

            spline_cont0[np.where(spline_cont0 / dataobj.noise < 5)] = np.nan
            spline_cont0 = copy(spline_cont0)
            spline_cont0[np.where(spline_cont0 < np.median(spline_cont0))] = np.nan
            spline_cont0[np.where(np.isnan(dataobj.bad_pixels))] = np.nan

            if masking_ifu_location is not None:
                im_ifux, im_ifuy = dataobj.getifucoords()
                dist_map = np.sqrt((im_ifux - masking_ifu_location[0]) ** 2 + (im_ifuy - masking_ifu_location[1]) ** 2)
                spline_cont0[np.where(dist_map < masking_radius)] = np.nan
            normalized_im = dataobj.data / spline_cont0
            normalized_err = dataobj.noise / spline_cont0

            wvs_list.extend(dataobj.wavelengths.flatten())
            normalized_im_list.extend(normalized_im.flatten())
            normalized_err_list.extend(normalized_err.flatten())
        if spec_R_sampling is None:
            spec_R_sampling = 4 * dataobj.R
        if interpolation == "linear":
            new_wavelengths, combined_fluxes, combined_errors = combine_spectrum(np.array(wvs_list),
                                                                                 np.array(normalized_im_list),
                                                                                 np.array(normalized_err_list),
                                                                                 np.nanmedian(wvs_list) / (
                                                                                     spec_R_sampling))
        elif interpolation == "spline":
            new_wavelengths, combined_fluxes, combined_errors, spl = combine_spectrum_1dspline(np.array(wvs_list),
                                                                                               np.array(
                                                                                                   normalized_im_list),
                                                                                               np.array(
                                                                                                   normalized_err_list),
                                                                                               np.nanmedian(
                                                                                                   wvs_list) / (
                                                                                                   spec_R_sampling),
                                                                                               oversampling=10)

        if out_filename is not None:
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=new_wavelengths))
            hdulist.append(pyfits.ImageHDU(data=combined_fluxes, name='COM_FLUXES'))
            hdulist.append(pyfits.ImageHDU(data=combined_errors, name='COM_ERRORS'))
            try:
                hdulist.writeto(out_filename, overwrite=True)
            except TypeError:
                hdulist.writeto(out_filename, clobber=True)
            hdulist.close()
    return new_wavelengths, combined_fluxes, combined_errors
    #
    # plt.scatter(dataobj.wavelengths.flatten(), normalized_im.flatten())
    # plt.ylim([0, 2])
    # plt.show()


def _build_cube_task_para(inputs):
    """ Worker function for creating a cube slice, for one single wavelength.
    Called from build_cube(); not intended to be called directly by users.

    Parameters
    ----------
    inputs : tuple containing many parameters
        X, Y, Z, Zerr, Zbp, wv_sampling, east2V2_deg, psf_interp_paras, wv_id, wv, ra_vec, dec_vec, aper_radius, N_pix_min

    Returns
    -------
    outs : list of lists
        Complex nested bunch of stuff... TODO figure out and document

    """
    X, Y, Z, Zerr, Zbp, wv_sampling, east2V2_deg, psf_interp_paras, wv_id, wv, ra_vec, dec_vec, aper_radius, N_pix_min = inputs

    psf_interp = _interp_psf(psf_interp_paras)

    outs = []
    for ra_id, ra in enumerate(ra_vec):
        for dec_id, dec in enumerate(dec_vec):

            R = np.sqrt((X - ra) ** 2 + (Y - dec) ** 2)
            Zerr_masking = Zerr / median_abs_deviation(Zerr[np.where(np.isfinite(Zerr))])
            where_finite = np.where(
                np.isfinite(Zbp) * (Zerr_masking < 1e1) * np.isfinite(X) * np.isfinite(Y) * (R < aper_radius))
            where_finite_bp = np.where(np.isfinite(Zbp))
            where_finite_masking = np.where((Zerr_masking < 1e1))
            where_radius = np.where((R < aper_radius))
            if np.size(where_finite[0]) < N_pix_min:
                outs.append([ra_id, dec_id, np.nan, np.nan])  # changed from continue
            else:
                X_fin = X[where_finite]
                Y_fin = Y[where_finite]
                Z_fin = Z[where_finite]

                Zerr_fin = Zerr[where_finite]
                M = psf_interp(X_fin - ra, Y_fin - dec)

                deno = np.nansum(M ** 2 / Zerr_fin ** 2)
                mfflux = np.nansum(M * Z_fin / Zerr_fin ** 2) / deno
                mffluxerr = 1 / np.sqrt(deno)

                res = Z_fin - mfflux * M
                noise_factor = np.nanstd(res / Zerr_fin)
                outs.append([ra_id, dec_id, mfflux, mffluxerr * noise_factor])
    return outs


def build_cube_para(combdataobj, psfs, psfX, psfY, ra_vec, dec_vec, out_filename=None,
                    linear_interp=True, mppool=None, aper_radius=0.5,
                    debug_init=None, debug_end=None, N_pix_min=None):
    """ Build a datacube, based on the forward modeling processed results

    Parameters
    ----------
    combdataobj
    psfs
    psfX
    psfY
    ra_vec
    dec_vec
    out_filename
    linear_interp : bool
        Use linear interpolation (TODO document what is being interpolated ?)
    mppool : multiprocessing.Pool or None
        if a multiprocessing Pool is supplied, the calculation will use that pool to run in parallel.
        Otherwise it will run in serial on a single process.
    aper_radius : float
        Aperture radius
    debug_init : int or None
        Minimum wavelength image to limit the calculation. Optional, for debugging.
    debug_end : int or None
        Maximum wavelength image to limit the calculation. Optional, for debugging.
    N_pix_min

    Returns
    -------
    flux_cube, fluxerr_cube, ra_grid, dec_grid

    """
    print("[LOG] BUILD CUBE MIRI PARA")

    if "regwvs" not in combdataobj.coords:
        raise Exception(
            "This data object needs to be interpolated on regular wavelength grid. See dataobj.compute_interpdata_regwvs")

    if mppool is not None:
        print('Setting parallel_flag = True')
        parallel_flag = True
    else:
        print('Setting parallel_flag = False')
        parallel_flag = False

    wv_sampling = combdataobj.wv_sampling
    east2V2_deg = combdataobj.east2V2_deg
    all_interp_ra = combdataobj.dra_as_array.transpose()
    all_interp_dec = combdataobj.ddec_as_array.transpose()
    all_interp_flux = combdataobj.data.transpose()
    all_interp_err = combdataobj.noise.transpose()
    all_interp_badpix = combdataobj.bad_pixels.transpose()
    fits.writeto("all_interp_ra.fits", all_interp_ra, overwrite=True)
    print(all_interp_ra.shape, np.nanmin(all_interp_ra), np.nanmax(all_interp_dec))
    if hasattr(combdataobj, "filelist"):
        N_dithers = len(combdataobj.filelist)
    else:
        N_dithers = 1

    ra_grid, dec_grid = np.meshgrid(ra_vec, dec_vec)

    flux_cube = np.zeros((np.size(wv_sampling), ra_grid.shape[0], ra_grid.shape[1])) + np.nan
    fluxerr_cube = np.zeros((np.size(wv_sampling), ra_grid.shape[0], ra_grid.shape[1])) + np.nan

    # only process frames with wavelength index between debug_init and debug_end
    if debug_init is None:
        debug_init = 0
    if debug_end is None:
        debug_end = np.size(wv_sampling)
    print(f'Processing wavelength indices in range: {debug_init} to {debug_end}')

    if N_pix_min is None:
        N_pix_min = (np.pi * aper_radius ** 2 / 0.01 * N_dithers) / 4

    # step 1 prepare list of inputs
    inputs = []
    for wv_id, wv in enumerate(wv_sampling):
        if not (debug_init <= wv_id < debug_end):
            continue
        rprint("prepping build_cube inputs... id: {} wave: {}".format(wv_id, wv))

        psf_interp_paras = linear_interp, psfs[wv_id, :, :], psfX[wv_id, :, :], psfY[wv_id, :, :], wv_id, east2V2_deg

        X = all_interp_ra[:, wv_id]
        Y = all_interp_dec[:, wv_id]
        Z = all_interp_flux[:, wv_id]
        Zerr = all_interp_err[:, wv_id]
        Zbp = all_interp_badpix[:, wv_id]

        inputs.append([X, Y, Z, Zerr, Zbp, wv_sampling, east2V2_deg,
                       psf_interp_paras,
                       wv_id, wv, ra_vec, dec_vec, aper_radius, N_pix_min])

    # step 2 map _build_cube_task over input list
    if parallel_flag:
        print('starting parallel _build_cube_task...')
        # Iterate calculation in parallel, showing a progress bar of percentage completion
        outputs = list(tqdm(mppool.imap(_build_cube_task_para, inputs), total=len(inputs), ncols=100))
    else:
        print('starting serial _build_cube_task...')
        outputs = []
        # Iterate calculation serially, also showing a progress bar of percentage completion
        for inp in tqdm(inputs, total=len(inputs), ncols=100):
            outputs.append(_build_cube_task_para(inp))

    # step 3 iterate over outputs and save values
    for j, inp in enumerate(inputs):
        X, Y, Z, Zerr, Zbp, wv_sampling, east2V2_deg, psf_interp_paras, wv_id, wv, ra_vec, dec_vec, aper_radius, N_pix_min = inp
        rprint('cubing outputs... id: {} wave: {}'.format(wv_id, wv))
        outs = outputs[j]
        for o in outs:
            ra_id, dec_id, flux, err = o
            flux_cube[wv_id, dec_id, ra_id] = flux
            fluxerr_cube[wv_id, dec_id, ra_id] = err

    if out_filename is not None:
        if debug_init != 0 or debug_end != np.size(wv_sampling):
            out_filename = out_filename.replace(".fits", "_from{0}to{1}.fits".format(debug_init, debug_end))
        print("saving", out_filename)
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=flux_cube))
        hdulist.append(pyfits.ImageHDU(data=fluxerr_cube, name='FLUXERR_CUBE'))
        hdulist.append(pyfits.ImageHDU(data=ra_grid, name='RA'))
        hdulist.append(pyfits.ImageHDU(data=dec_grid, name='DEC'))
        hdulist.append(pyfits.ImageHDU(data=wv_sampling, name='WAVE'))
        hdulist.writeto(out_filename, overwrite=True)
        hdulist.close()
    return flux_cube, fluxerr_cube, ra_grid, dec_grid
