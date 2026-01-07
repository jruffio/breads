import os.path
from copy import copy
from glob import glob
import astropy.io.fits as pyfits

import numpy as np
import stpsf as webbpsf
from astropy.io import fits
from astropy.stats import sigma_clip

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

        super().__init__(filename, utils_dir, verbose)
        self.ifu_name = 'miri'
        #self._init_wcs(filename)
        self._init_miri_channel_band(channel_reduction)
        self._init_mask_channel(wv_ref)
        self.opmode = "IFU"
        super()._init_pipeline(save_utils=save_utils, load_utils=load_utils, preproc_task_list=preproc_task_list)

    def _init_miri_channel_band(self, channel_reduction):
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
        mad_threshold
        window_size
        save_utils : bool or string
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
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyUserWarning)
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
            hdulist.writeto(out_filename, overwrite=True)
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
            hdulist_sc.writeto(out_filename, overwrite=True)
            hdulist_sc.close()

        if self.data_unit == 'MJy':
            return model_im
        elif self.data_unit == 'MJy/sr':
            return model_im / (self.area2d * arcsec2_to_sr)


    def _get_webbpsf_fit_inputs(self):
        """
        Hook to override for MIRI, returns input for webbPSF fit
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
        print("MIRI hook starspectrum")
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
        continuum = np.copy(continuum)
        print(int(self.channel_reduction), self.band_aka)

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
        self.bad_pixels = bad_pixels.transpose()

    def _save_starspectrum_contnorm(self, save_utils, new_wavelengths, combined_fluxes, combined_errors, spline_cont0, spline_paras0, x_nodes):
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

    def _init_regwvs_obj(self, regwvs_dataobj, Ntraces, Nwv):
        regwvs_dataobj.dra_as_array = np.full((Nwv, Ntraces), np.nan)
        regwvs_dataobj.ddec_as_array = np.full((Nwv, Ntraces), np.nan)
        regwvs_dataobj.wavelengths = np.full((Nwv, Ntraces), np.nan)
        regwvs_dataobj.leftnright_wavelengths = np.full((2, Nwv, Ntraces), np.nan)
        regwvs_dataobj.data = np.full((Nwv, Ntraces), np.nan)
        regwvs_dataobj.noise = np.full((Nwv, Ntraces), np.nan)
        regwvs_dataobj.bad_pixels = np.full((Nwv, Ntraces), np.nan)
        regwvs_dataobj.area2d = np.full((Nwv, Ntraces), np.nan)

    def _get_interpdata_shapes(self, _data, wv_sampling):
        Ntraces, Nwv = _data.shape[1], np.size(wv_sampling)
        return Ntraces, Nwv

    def _get_where_finite(self, trace_id):
        wvs_finite = np.where(np.isfinite(self.wavelengths[:, trace_id]))
        where_finite = np.where(np.isfinite(self.bad_pixels[:, trace_id]))
        return wvs_finite, where_finite

    def _interpdata_regwvs_trace(self, regwvs_dataobj, wv_sampling, _data, wvs_finite, where_finite, trace_id):

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



##########################FUNCTIONS###################################

'''def _get_wpsf_task(paras, fov=6):
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
        psfmodel = a * interpolator(Xsc * psf_wv0 / center_wv - xc, Ysc * psf_wv0 / center_wv - yc)
        psfsub_Zsc = Zsc - psfmodel

        bestfit_paras[:, wv_id] = np.array(
            [center_wv, xc * center_wv / psf_wv0, yc * center_wv / psf_wv0, a * interpolator(0, 0)])
        psfsub_model_im[where_sc_mask] = psfmodel
        psfsub_sc_im[where_sc_mask] = psfsub_Zsc

    return bestfit_paras, psfsub_model_im, psfsub_sc_im


def where_point_source(dataobj, radec_as, rad_as):
    """Which pixels in a point cloud are within some given radius of a given location?

    Parameters
    ----------
    dataobj : data object
        Point cloud data object
    radec_as : tuple of floats
        RA, Dec coordinates of interest
    rad_as : float
        Radius in arcseconds

    Returns
    -------

    """
    ra, dec = radec_as
    dist2pointsource_as = np.sqrt((dataobj.dra_as_array - ra) ** 2 + (dataobj.ddec_as_array - dec) ** 2)
    return np.where(dist2pointsource_as < rad_as)


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
        iterator_sectors.extend([((r_min, r_max), phi_bound) for phi_bound in phi_bounds_list])
    tot_sectors = len(iterator_sectors)

    out_paras = np.zeros((tot_sectors, 5)) + np.nan
    out_model = np.zeros(_Z.shape) + np.nan
    for sector_id, sector in enumerate(iterator_sectors):
        rmin, rmax = sector[0]
        pamin, pamax = sector[1]
        padding2 = padding / 2.0
        if pamin < pamax:
            deltaphi = pamax - pamin + 2 * padding / np.mean([rmin, rmax])
        else:
            deltaphi = (2 * np.pi - (pamin - pamax)) + 2 * padding / np.mean([rmin, rmax])

        # If the length or the arc is higher than 2*pi, simply pick the entire circle.
        if deltaphi >= 2 * np.pi:
            pamin_pad = 0
            pamax_pad = 2 * np.pi
        else:
            pamin_pad = (pamin - padding / np.mean([rmin, rmax])) % (2.0 * np.pi)
            pamax_pad = (pamax + padding / np.mean([rmin, rmax])) % (2.0 * np.pi)

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
    """ Interpolate PSF

    Parameters
    ----------
    paras : tuple
        Contains the following:
        linear_interp, wepsf, wifuX, wifuY, wv_id, east2V2_deg


    Returns
    -------
    webbpsf_interp : Interpolator object
        a scipy.interpolate Interpolator object for interpolating a PSF onto
        specified coordinates.

    """
    linear_interp, wepsf, wifuX, wifuY, wv_id, east2V2_deg = paras
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

    # only process frames with wavelength index between debug_init and debug_end
    if debug_init is None:
        debug_init = 0
    if debug_end is None:
        debug_end = np.size(wv_sampling)
    print(debug_init, debug_end)

    wpsf_angle_offset = 0
    bestfit_coords_defined = False
    if mppool is None:
        for wv_id, wv in enumerate(wv_sampling):
            if not (wv_id >= debug_init and wv_id < debug_end):
                continue
            print(wv_id, wv, np.size(wv_sampling))
            paras = linear_interp, psfs[wv_id], psfX[wv_id], psfY[wv_id], rotate_psf - wpsf_angle_offset, flipx, \
                all_interp_ra[wv_id, :], all_interp_dec[wv_id, :], all_interp_flux[wv_id, :], all_interp_err[
                                                                                              wv_id, :], all_interp_badpix[wv_id, :], \
                IWA, OWA, fit_cen, fit_angle, init_paras, ann_width, padding, sector_area
            out = _fit_wpsf_task(paras)

            if not bestfit_coords_defined:
                bestfit_coords = np.zeros(
                    (out[0].shape[0], np.size(wv_sampling), 5)) + np.nan  # flux_init, flux, ra, dec, angle
                bestfit_coords_defined = True
            bestfit_coords[:, wv_id, :] = out[0]
            all_interp_psfmodel[wv_id, :] = out[1]
            all_interp_psfsub[wv_id, :] = all_interp_flux[wv_id, :] - out[1]


    else:
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
                    (out[0].shape[0], np.size(wv_sampling), 5)) + np.nan  # flux_init, flux, ra, dec, angle
                bestfit_coords_defined = True
            bestfit_coords[:, debug_init + out_id, :] = out[0]
            all_interp_psfmodel[debug_init + out_id, :] = out[1]
            all_interp_psfsub[debug_init + out_id, :] = all_interp_flux[debug_init + out_id, :] - out[1]

    all_interp_psfsub = all_interp_psfsub * all_interp_area2d / psf_spaxel_area
    all_interp_psfmodel = all_interp_psfmodel * all_interp_area2d / psf_spaxel_area
    all_interp_err = all_interp_err * all_interp_area2d / psf_spaxel_area

    if out_filename is not None:
        wpsfsfit_header = {"INIT_ANG": wpsf_angle_offset,
                           "INIT_RA": init_paras[0], "INIT_DEC": init_paras[1]}
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=bestfit_coords, header=pyfits.Header(cards=wpsfsfit_header)))
        hdulist.writeto(out_filename, overwrite=True)
        hdulist.close()

        RDI_psfsub_dir = os.path.join(os.path.dirname(out_filename), "RDI_psfsub" + RDI_folder_suffix)
        if not os.path.exists(RDI_psfsub_dir):
            os.makedirs(RDI_psfsub_dir)
        RDI_model_dir = os.path.join(os.path.dirname(out_filename), "RDI_model" + RDI_folder_suffix)
        if not os.path.exists(RDI_model_dir):
            os.makedirs(RDI_model_dir)

        for obj_id, filename in enumerate(combdataobj.filelist):
            ny = combdataobj.data.shape[1] // len(combdataobj.filelist)
            nx = combdataobj.data.shape[0]
            interpdata_filename = os.path.join(combdataobj.utils_dir,
                                               os.path.basename(filename).replace(".fits", "_regwvs.fits"))
            _interpdata_psfsub_filename = interpdata_filename.replace(".fits", "_psfsub" + RDI_folder_suffix + ".fits")
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=all_interp_psfsub[:, (ny * obj_id):(ny * (obj_id + 1))]))
            hdulist.append(
                pyfits.ImageHDU(data=all_interp_psfmodel[:, (ny * obj_id):(ny * (obj_id + 1))], name='INTERP_MOD'))
            hdulist.append(
                pyfits.ImageHDU(data=all_interp_err[:, (ny * obj_id):(ny * (obj_id + 1))], name='INTERP_ERR'))
            hdulist.append(pyfits.ImageHDU(data=all_interp_ra[:, (ny * obj_id):(ny * (obj_id + 1))], name='INTERP_RA'))
            hdulist.append(
                pyfits.ImageHDU(data=all_interp_dec[:, (ny * obj_id):(ny * (obj_id + 1))], name='INTERP_DEC'))
            hdulist.append(
                pyfits.ImageHDU(data=all_interp_wvs[:, (ny * obj_id):(ny * (obj_id + 1))], name='INTERP_WAVE'))
            hdulist.append(
                pyfits.ImageHDU(data=all_interp_badpix[:, (ny * obj_id):(ny * (obj_id + 1))], name='INTERP_BADPIX'))
            hdulist.append(
                pyfits.ImageHDU(data=all_interp_area2d[:, (ny * obj_id):(ny * (obj_id + 1))], name='INTERP_AREA2D'))
            hdulist.writeto(_interpdata_psfsub_filename, overwrite=True)


            hdulist_sc = pyfits.open(filename)
            wvs_ori = combdataobj.wvs_ori  # hdulist_sc["WAVELENGTH"].data
            ny_ori, nx_ori = wvs_ori.shape

            new_model = np.zeros((ny_ori, nx_ori)) + np.nan
            new_badpix = np.zeros((ny_ori, nx_ori)) + np.nan
            new_area2d = np.zeros((ny_ori, nx_ori)) + np.nan
            for colid in range(nx_ori):
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
            hdulist_sc.writeto(psfsub_filename, overwrite=True)

            hdulist_sc["SCI"].data = new_model
            psfmod_filename = os.path.join(RDI_model_dir, os.path.basename(filename))
            hdulist_sc.writeto(psfmod_filename, overwrite=True)

            hdulist_sc.close()


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
        hdulist.writeto(out_filename, overwrite=True)
        hdulist.close()
    return flux_cube, fluxerr_cube, ra_grid, dec_grid

    # dataobj0 = dataobj_list[0]
    # wv_sampling = dataobj0.wv_sampling
    # east2V2_deg = dataobj0.east2V2_deg

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
            where_finite = np.where(np.isfinite(Zbp) * np.isfinite(X) * np.isfinite(Y) * (R < aper_radius))
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
'''