import itertools
import os.path
from copy import copy
from glob import glob

import astropy.io.fits as pyfits
import matplotlib.tri as tri
import numpy as np
import scipy.linalg as la
import stpsf as webbpsf

from scipy.interpolate import interp1d, splev, splrep
from scipy.ndimage import generic_filter, median_filter
from scipy.optimize import lsq_linear
from scipy.stats import median_abs_deviation
from tqdm import tqdm


from breads.utils import get_spline_model
from breads.instruments.jwst_IFUs import JWST_IFUs
from breads.instruments.jwst_IFUs import crop_trace_edges, set_nans, filter_big_triangles, combine_spectrum



class JWSTNirspec_cal(JWST_IFUs):
    def __init__(self, filename=None, utils_dir=None, save_utils=True,
                 load_utils=True,
                 preproc_task_list = None,
                 verbose=True):
        """JWST NIRSpec 2D calibrated data class.


        Parameters
        ----------
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
        wv_ref : float
            Reference wavelength. If not set, the shortest wavelength will be used by default.



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
        self.ifu_name = 'nirspec'
        super().__init__(filename, utils_dir, verbose)
        self._init_additional_default_filenames()
        self.R = 2700 #TODO change
        super()._init_pipeline(save_utils=save_utils, load_utils=load_utils, preproc_task_list=preproc_task_list)


    def _init_wcs(self, filename):
        "Hook for nirspec subclass to compute World Coordinates System."
        from stdatamodels.jwst import datamodels
        import jwst.assign_wcs
        from jwst.photom.photom import DataSet
        from gwcs import wcstools

        hdulist = pyfits.open(filename)
        calfile = jwst.datamodels.open(hdulist)  # save time opening by passing the already opened file
        photom_dataset = DataSet(calfile)

        # Compute 2D wavelength and pixel area arrays for the whole image
        # Use WCS to compute RA, Dec for each pixel

        self.trace_id_map = np.zeros(self.data.shape) + np.nan

        if self.opmode == "FIXEDSLIT":
            print('Using FixedSlit methods...')

            pxarea_as2 = calfile.slits[0].meta.photometry.pixelarea_arcsecsq
            area2d = np.ones(self.data.shape) * pxarea_as2  # constant area

            if len(calfile.slits) != 1:
                raise Exception("Multiple slits in data model not implemented.")
            slitwcs = calfile.slits[0].meta.wcs
            x, y = wcstools.grid_from_bounding_box(slitwcs.bounding_box, step=(1, 1), center=True)
            ra_array, dec_array, wavelen_array = slitwcs(x, y)

            self.trace_id_map[np.where(np.isfinite(ra_array))] = 0

        elif self.opmode == "IFU":
            ## Determine pixel areas for each pixel, retrieved from a CRDS reference file
            area_fname = hdulist[0].header["R_AREA"].replace("crds://",
                                                             os.path.join(self.crds_dir, "references", "jwst",
                                                                          "nirspec") + os.path.sep)
            # Load the pixel area table for the IFU slices
            area_model = datamodels.open(area_fname)
            area_data = area_model.area_table

            wave2d, area2d, dqmap = photom_dataset.calc_nrs_ifu_sens2d(area_data)
            area2d[np.where(area2d == 1)] = np.nan
            wcses = jwst.assign_wcs.nrs_ifu_wcs(calfile)  # returns a list of 30 WCSes, one per slice. This is slow.

            #Initializing coordinates arrays
            ra_array = np.zeros(self.data.shape) + np.nan
            dec_array = np.zeros(self.data.shape) + np.nan
            wavelen_array = np.zeros(self.data.shape) + np.nan

            print(f"Computing coords for {len(wcses)} slices...")
            for i in tqdm(range(len(wcses)), total=len(wcses), ncols=100):
                # Set up 2D X, Y index arrays spanning across the full area of the slice WCS
                xmin = max(int(np.round(wcses[i].bounding_box.intervals[0][0])), 0)
                xmax = int(np.round(wcses[i].bounding_box.intervals[0][1]))
                ymin = max(int(np.round(wcses[i].bounding_box.intervals[1][0])), 0)
                ymax = int(np.round(wcses[i].bounding_box.intervals[1][1]))

                x = np.arange(xmin, xmax)
                x = x.reshape(1, x.shape[0]) * np.ones((ymax - ymin, 1))
                y = np.arange(ymin, ymax)
                y = y.reshape(y.shape[0], 1) * np.ones((1, xmax - xmin))

                # Transform all those pixels to RA, Dec, wavelength
                skycoords, speccoord = wcses[i](x, y, with_units=True)

                ra_array[ymin:ymax, xmin:xmax] = skycoords.ra
                dec_array[ymin:ymax, xmin:xmax] = skycoords.dec
                wavelen_array[ymin:ymax, xmin:xmax] = speccoord

                self.trace_id_map[ymin:ymax, xmin:xmax][np.where(np.isfinite(ra_array[ymin:ymax, xmin:xmax]))] = i

        arcsec2_to_steradians = (2.*np.pi/(360.*3600.))**2

        self.wavelengths = wavelen_array
        self.ra_array = ra_array
        self.dec_array = dec_array
        self.area2d = area2d * arcsec2_to_steradians #convert area2d in steradians

        return ra_array, dec_array, wavelen_array, area2d


    def _init_additional_default_filenames(self):
        """Initializes the default filenames used only for nirspec preprocessing."""
        self.default_filenames["compute_charge_bleeding_mask"] = \
                os.path.join(self.utils_dir, os.path.basename(self.filename).replace(".fits", "_barmask.fits"))
        self.default_filenames["compute_starspectrum_contnorm_2dspline"] = \
                os.path.join(self.utils_dir, os.path.basename(self.filename).replace(".fits", "_starspec_2dcontnorm.fits"))
        self.default_filenames["compute_starsubtraction_2dspline"] = \
                os.path.join(self.utils_dir, os.path.basename(self.filename).replace(".fits", "_2dstarsub.fits"))

    def compute_med_filt_badpix(self, save_utils=False, window_size=50, mad_threshold=50, crop_Npix_from_trace_edges=0):
        """ Quick bad pixel identification.

        The data is first high-pass filtered row by row with a median filter with a window size of 50 (window_size)
        pixels. The median absolute deviation (MAP) is then calculated row by row, and any pixel deviating by more than
        50x the MAP are identified as bad.

        Only returns (or save) the newly identified bad pixels, the ones already included in self.bad_pixels won't be in
        new_badpix.
        But this map is automatically applied to self.bad_pixels:  self.bad_pixels *= new_badpix

        Parameters
        ----------
        crop_Npix_from_trace_edges
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

        if self.verbose:
            print("Initializing row_err and bad_pixels for nirspec")
        new_badpix = np.ones(self.bad_pixels.shape)
        for rowid in range(self.bad_pixels.shape[0]):
            row_err = self.noise[rowid,:]
            row_err = row_err - generic_filter(row_err, np.nanmedian, size=window_size)
            row_err_masking = row_err/median_abs_deviation(row_err[np.where(np.isfinite(self.bad_pixels[rowid,:]))])
            new_badpix[rowid,np.where((row_err_masking>mad_threshold))[0]] = np.nan
        self.bad_pixels *= new_badpix

        if crop_Npix_from_trace_edges != 0:
            if hasattr(self, "trace_id_map"):
                self.bad_pixels = crop_trace_edges(self.bad_pixels, N_pix=crop_Npix_from_trace_edges,trace_id_map=self.trace_id_map)
            else:
                self.bad_pixels = crop_trace_edges(self.bad_pixels, N_pix=crop_Npix_from_trace_edges)

        if save_utils:
            self._save_med_filt_badpix(save_utils, new_badpix)

        return new_badpix


    def _get_webbpsf_model_inputs(self, image_mask, pixel_scale):
        """Hook for nirspec subclass, returns webbpsf parameters for nirspec simulation"""
        nrs = webbpsf.NIRSpec()
        nrs.load_wss_opd_by_date(self.priheader["DATE-BEG"])  # Load telescope state as of our observation date
        nrs.image_mask = image_mask  # optional: model opaque field stop outside of the IFU aperture
        nrs.pixelscale = pixel_scale
        return nrs


    def compute_charge_bleeding_mask(self, save_utils=False, threshold2mask=0.15):
        """ Compute charge bleeding bar mask

        Parameters
        ----------
        save_utils: bool (default is False)
            if True, save the computed charge bleeding mask.
        threshold2mask: float (default is 0.15) in arcsec
            Separation threshold to mask the traces sitting in the charge bleeding region.

        Returns
        -------

        """
        if self.verbose:
            print(f"Computing charge bleeding mask. Will save to {0}".format(self.default_filenames['compute_charge_bleeding_mask']))
        ifuX, ifuY = self.get_ifu_coords()

        bar_mask = np.ones(self.bad_pixels.shape)
        bar_mask[np.where(np.abs(ifuX) < threshold2mask)] = np.nan
        if save_utils:
            self.save_charge_bleeding_mask(save_utils, bar_mask)

        self.bad_pixels *= bar_mask
        return bar_mask

    def save_charge_bleeding_mask(self, save_utils, bar_mask):
        """Save charge bleeding bar mask"""
        if isinstance(save_utils, str):
            out_filename = save_utils
        else:
            out_filename = self.default_filenames["compute_charge_bleeding_mask"]

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=bar_mask))
        hdulist.writeto(out_filename, overwrite=True)
        hdulist.close()

    def reload_charge_bleeding_mask(self, load_filename=None):
        """ Reload charge bleeding bar mask

        Parameters
        ----------
        load_filename : str or None
            Filename to load mask data from, or leave None to use default filename

        Returns
        -------
        bar_mask : ndarray

        Also modifies self.badpixels, by multiplying that times the bar_mask

        """
        if load_filename is None:
            load_filename = self.default_filenames["compute_charge_bleeding_mask"]
        if len(glob(load_filename)) ==0:
            return None

        hdulist = pyfits.open(load_filename)
        bar_mask = hdulist[0].data
        hdulist.close()

        self.bad_pixels *= bar_mask
        return bar_mask

    #herenow
    def compute_starspectrum_contnorm_2dspline(self,  save_utils=False,im=None, im_wvs=None, err=None, mppool=None,
                                               spec_R_sampling=None, threshold_badpix=10,
                                               wv_nodes=None,N_wvs_nodes=20,ifuy_nodes=None,delta_ifuy=0.05,
                                               apply_new_bad_pixels = False, iterative = True,independent_trace = True):
        """ Compute star spectrum continuum normalized by 2d spline

        Parameters
        ----------
        save_utils
        im
        im_wvs
        err
        mppool
        spec_R_sampling
        threshold_badpix
        wv_nodes
        N_wvs_nodes
        ifuy_nodes
        delta_ifuy
        apply_new_bad_pixels
        iterative
        independent_trace

        Returns
        -------

        """
        if im is None:
            im = self.data
        if im_wvs is None:
            im_wvs = self.wavelengths
        if err is None:
            err = self.noise
        if spec_R_sampling is None:
            spec_R_sampling = self.R*4
        if wv_nodes is None:
            wv_nodes = np.linspace(np.nanmin(im_wvs), np.nanmax(im_wvs), N_wvs_nodes, endpoint=True)

        _, im_ifuy = self.get_ifu_coords()

        if ifuy_nodes is None:
            ifuy_min, ifuy_max = np.nanmin(im_ifuy), np.nanmax(im_ifuy)
            ifuy_min, ifuy_max = np.floor(ifuy_min * 10) / 10, np.ceil(ifuy_max * 10) / 10
            ifuy_nodes = np.arange(ifuy_min, ifuy_max + 0.1, delta_ifuy)

        if self.verbose:
            print(f"Computing stellar spectrum with 2d spline (continuum normalized)")

        if independent_trace:
            _trace_id_map = self.trace_id_map
        else:
            _trace_id_map = np.zeros(self.trace_id_map.shape)

        if 1:
            unique_trace_ids = np.unique(_trace_id_map[np.where(np.isfinite(_trace_id_map))])

            ifuy_nodes_grid, wv_nodes_grid = np.meshgrid(ifuy_nodes, wv_nodes, indexing="ij")

            # Define the window size
            w = 10
            window_size = (1, w)
            # Apply median filter
            data_all_LPF = median_filter(self.data * self.bad_pixels, size=window_size, mode='constant', cval=np.nan)

            reg_mean_map0 = np.zeros((np.size(unique_trace_ids),np.size(ifuy_nodes),np.size(wv_nodes))) + np.nan
            for traceid in range(np.size(unique_trace_ids)):

                where_good = np.where((_trace_id_map == traceid) *np.isfinite(data_all_LPF) * np.isfinite(im_ifuy) * np.isfinite(self.wavelengths))
                X = im_ifuy[where_good]
                Y = self.wavelengths[where_good]
                Z = data_all_LPF[where_good]
                filtered_triangles = filter_big_triangles(X * self.wv_ref / Y, Y, 0.2)
                # Create filtered triangulation
                filtered_tri = tri.Triangulation(X * self.wv_ref / Y, Y, triangles=filtered_triangles)
                # Perform LinearTriInterpolator for filtered triangulation
                pointcloud_interp = tri.LinearTriInterpolator(filtered_tri, Z)

                reg_mean_map0[traceid,:,:] = pointcloud_interp(ifuy_nodes_grid, wv_nodes_grid)

            # replace nans in horizontal rows by extending the last value
            for traceid in range(reg_mean_map0.shape[0]):
                for k in range(reg_mean_map0.shape[1]):
                    row = reg_mean_map0[traceid,k, :]
                    finite_indices = np.where(np.isfinite(row))[0]
                    if len(finite_indices) == 0:
                        continue
                    min_id = np.min(finite_indices)
                    max_id = np.max(finite_indices)
                    reg_mean_map0[traceid,k, 0:min_id] = reg_mean_map0[traceid,k, min_id]
                    reg_mean_map0[traceid,k, max_id + 1::] = reg_mean_map0[traceid,k, max_id]

            # replace nans in  vertical columns by extending the last value
            for traceid in range(reg_mean_map0.shape[0]):
                for l in range(reg_mean_map0.shape[2]):
                    col = reg_mean_map0[traceid,:, l]
                    finite_indices = np.where(np.isfinite(col))[0]
                    if len(finite_indices) == 0:
                        continue
                    min_id = np.min(finite_indices)
                    max_id = np.max(finite_indices)
                    reg_mean_map0[traceid,0:min_id, l] = reg_mean_map0[traceid,min_id, l]
                    reg_mean_map0[traceid,max_id + 1::, l] = reg_mean_map0[traceid,max_id, l]

            reg_std_map0 = np.abs(reg_mean_map0)#/2

        spline_cont0, _, new_badpixs, new_res, spline_paras0 = normalize_slices_2dspline(im,
                                                                                         im_wvs,
                                                                                         im_ifuy,
                                                                                         noise=err,
                                                                                         badpixs=self.bad_pixels,
                                                                                         trace_id_map = _trace_id_map,
                                                                                         wv_nodes = wv_nodes,
                                                                                         ifuy_nodes=ifuy_nodes,
                                                                                         threshold=threshold_badpix,
                                                                                         use_set_nans=False,
                                                                                         reg_mean_map=reg_mean_map0,
                                                                                         reg_std_map=reg_std_map0,
                                                                                         wv_ref=self.wv_ref,
                                                                                         mypool=mppool)
        if iterative:
            reg_mean_map1 = copy(spline_paras0)
            where_nan = np.where(np.isnan(reg_mean_map1))
            reg_mean_map1[where_nan] = reg_mean_map0[where_nan]
            reg_std_map1 = np.abs(reg_mean_map1)/2
            spline_cont0, _, new_badpixs, new_res, spline_paras0 = normalize_slices_2dspline(im,
                                                                                             im_wvs,
                                                                                             im_ifuy,
                                                                                             noise=err,
                                                                                             badpixs=self.bad_pixels*new_badpixs,
                                                                                             trace_id_map = _trace_id_map,
                                                                                             wv_nodes = wv_nodes,
                                                                                             ifuy_nodes=ifuy_nodes,
                                                                                             threshold=threshold_badpix,
                                                                                             use_set_nans=False,
                                                                                             reg_mean_map=reg_mean_map1,
                                                                                             reg_std_map=reg_std_map1,
                                                                                             wv_ref=self.wv_ref,
                                                                                             mypool=mppool)
        continuum = copy(spline_cont0)
        continuum[np.where(continuum / err < 5)] = np.nan
        continuum[np.where(continuum < np.median(continuum))] = np.nan
        continuum[np.where(np.isnan(self.bad_pixels))] = np.nan
        normalized_im = im / continuum
        normalized_err = err / continuum

        new_wavelengths, combined_fluxes, combined_errors = combine_spectrum(im_wvs.flatten(),
                                                                             normalized_im.flatten(),
                                                                             normalized_err.flatten(),
                                                                             np.nanmedian(im_wvs) / spec_R_sampling)

        if apply_new_bad_pixels:
            self.bad_pixels *= new_badpixs

        if save_utils:
            if isinstance(save_utils,str):
                out_filename = save_utils
            else:
                out_filename = self.default_filenames["compute_starspectrum_contnorm_2dspline"]

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=new_wavelengths))
            hdulist.append(pyfits.ImageHDU(data=combined_fluxes, name='COM_FLUXES'))
            hdulist.append(pyfits.ImageHDU(data=combined_errors, name='COM_ERRORS'))
            hdulist.append(pyfits.ImageHDU(data=spline_cont0, name='SPLINE_CONT0'))
            hdulist.append(pyfits.ImageHDU(data=spline_paras0, name='SPLINE_PARAS0'))
            hdulist.append(pyfits.ImageHDU(data=new_badpixs, name='NEW_BADPIX'))
            hdulist.append(pyfits.ImageHDU(data=wv_nodes, name='wv_nodes'))
            hdulist.append(pyfits.ImageHDU(data=ifuy_nodes, name='ifuy_nodes'))
            hdulist.writeto(out_filename, overwrite=True)
            hdulist.close()

        self.wv_nodes = wv_nodes
        self.ifuy_nodes = ifuy_nodes
        self.star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False, fill_value=1)
        return new_wavelengths,combined_fluxes,combined_errors,spline_cont0,spline_paras0,wv_nodes,ifuy_nodes

    def reload_starspectrum_contnorm_2dspline(self, load_filename=None, apply_new_bad_pixels = False):
        """ Reload star spectrum continuum normalized by 2d spline

        Parameters
        ----------
        load_filename : str or None
            Filename to load spectrum data from, or leave None to use default filename
        apply_new_bad_pixels : bool
            If set, multiply self.badpixels times the NEW_BADPIX extension of the spectrum


        Returns
        -------
        new_wavelengths,combined_fluxes,combined_errors,spline_cont0,spline_paras0,wv_nodes,ifuy_nodes

        """
        if load_filename is None:
            load_filename = self.default_filenames["compute_starspectrum_contnorm_2dspline"]
        if len(glob(load_filename)) ==0:
            return None

        hdulist = pyfits.open(load_filename)
        new_wavelengths = hdulist[0].data
        combined_fluxes = hdulist['COM_FLUXES'].data
        combined_errors = hdulist['COM_ERRORS'].data
        spline_cont0 = hdulist['SPLINE_CONT0'].data
        spline_paras0 = hdulist['SPLINE_PARAS0'].data
        new_badpixs = hdulist['NEW_BADPIX'].data
        wv_nodes = hdulist['wv_nodes'].data
        ifuy_nodes = hdulist['ifuy_nodes'].data
        hdulist.close()

        if apply_new_bad_pixels:
            self.bad_pixels *= new_badpixs

        self.wv_nodes = wv_nodes
        self.ifuy_nodes = ifuy_nodes
        self.star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False, fill_value=1)
        return new_wavelengths,combined_fluxes,combined_errors,spline_cont0,spline_paras0,wv_nodes,ifuy_nodes


    def compute_starsubtraction_2dspline(self,  save_utils=False, im=None, im_wvs=None, err=None, threshold_badpix=10,
                                mppool=None,starsub_dir="starsub2d", iterative = True,independent_trace = True):
        """ Compute Star Subtraction with 2D Spline

        Parameters
        ----------
        save_utils
        im
        im_wvs
        err
        threshold_badpix
        mppool
        starsub_dir
        iterative
        independent_trace

        Returns
        -------
        subtracted_im, star_model, spline_paras0, self.wv_nodes, self.ifuy_nodes

        """

        if self.verbose:
            print(f"Computing star subtraction 2d spline.")

        if im is None:
            im = self.data
        if im_wvs is None:
            im_wvs = self.wavelengths
        if err is None:
            err = self.noise

        _, im_ifuy = self.get_ifu_coords()

        if independent_trace:
            _trace_id_map = self.trace_id_map
        else:
            _trace_id_map = np.zeros(self.trace_id_map.shape)

        if 1:
            unique_trace_ids = np.unique(_trace_id_map[np.where(np.isfinite(_trace_id_map))])

            ifuy_nodes_grid, wv_nodes_grid = np.meshgrid(self.ifuy_nodes, self.wv_nodes, indexing="ij")

            # Define the window size
            w = 10
            window_size = (1, w)
            # Apply median filter
            data_all_LPF = median_filter(self.data * self.bad_pixels, size=window_size, mode='constant', cval=np.nan)

            reg_mean_map0 = np.zeros((np.size(unique_trace_ids),np.size(self.ifuy_nodes),np.size(self.wv_nodes))) + np.nan
            for traceid in range(np.size(unique_trace_ids)):

                where_good = np.where((_trace_id_map == traceid) *np.isfinite(data_all_LPF) * np.isfinite(im_ifuy) * np.isfinite(self.wavelengths))
                X = im_ifuy[where_good]
                Y = self.wavelengths[where_good]
                Z = data_all_LPF[where_good]
                filtered_triangles = filter_big_triangles(X * self.wv_ref / Y, Y, 0.2)
                # Create filtered triangulation
                filtered_tri = tri.Triangulation(X * self.wv_ref / Y, Y, triangles=filtered_triangles)
                # Perform LinearTriInterpolator for filtered triangulation
                pointcloud_interp = tri.LinearTriInterpolator(filtered_tri, Z)

                reg_mean_map0[traceid,:,:] = pointcloud_interp(ifuy_nodes_grid, wv_nodes_grid)

            # replace nans in horizontal rows by extending the last value
            for traceid in range(reg_mean_map0.shape[0]):
                for k in range(reg_mean_map0.shape[1]):
                    row = reg_mean_map0[traceid,k, :]
                    finite_indices = np.where(np.isfinite(row))[0]
                    if len(finite_indices) == 0:
                        continue
                    min_id = np.min(finite_indices)
                    max_id = np.max(finite_indices)
                    reg_mean_map0[traceid,k, 0:min_id] = reg_mean_map0[traceid,k, min_id]
                    reg_mean_map0[traceid,k, max_id + 1::] = reg_mean_map0[traceid,k, max_id]

            # replace nans in  vertical columns by extending the last value
            for traceid in range(reg_mean_map0.shape[0]):
                for l in range(reg_mean_map0.shape[2]):
                    col = reg_mean_map0[traceid,:, l]
                    finite_indices = np.where(np.isfinite(col))[0]
                    if len(finite_indices) == 0:
                        continue
                    min_id = np.min(finite_indices)
                    max_id = np.max(finite_indices)
                    reg_mean_map0[traceid,0:min_id, l] = reg_mean_map0[traceid,min_id, l]
                    reg_mean_map0[traceid,max_id + 1::, l] = reg_mean_map0[traceid,max_id, l]

            reg_std_map0 = np.abs(reg_mean_map0)/2

        if self.verbose:
            print(f"Running 2d spline fit for the first time")
        star_model, _, new_badpixs, subtracted_im, spline_paras0 = normalize_slices_2dspline(im,
                                                                                         im_wvs,
                                                                                         im_ifuy,
                                                                                         noise=err,
                                                                                         badpixs=self.bad_pixels,
                                                                                         star_model=self.star_func(im_wvs),
                                                                                         trace_id_map = _trace_id_map,
                                                                                         wv_nodes = self.wv_nodes,
                                                                                         ifuy_nodes=self.ifuy_nodes,
                                                                                         threshold=threshold_badpix,
                                                                                         use_set_nans=False,
                                                                                         reg_mean_map=reg_mean_map0,
                                                                                         reg_std_map=reg_std_map0,
                                                                                         wv_ref=self.wv_ref,
                                                                                         mypool=mppool)
        if iterative:
            reg_mean_map1 = copy(spline_paras0)
            where_nan = np.where(np.isnan(reg_mean_map1))
            reg_mean_map1[where_nan] = reg_mean_map0[where_nan]
            reg_std_map1 = np.abs(reg_mean_map1)/2
            if self.verbose:
                print(f"Running 2d spline fit for the second time after removing outliers")
            star_model, _, new_badpixs, subtracted_im, spline_paras0 = normalize_slices_2dspline(im,
                                                                                             im_wvs,
                                                                                             im_ifuy,
                                                                                             noise=err,
                                                                                             badpixs=self.bad_pixels*new_badpixs,
                                                                                             star_model=self.star_func(im_wvs),
                                                                                             trace_id_map = _trace_id_map,
                                                                                             wv_nodes = self.wv_nodes,
                                                                                             ifuy_nodes=self.ifuy_nodes,
                                                                                             threshold=threshold_badpix,
                                                                                             use_set_nans=False,
                                                                                             reg_mean_map=reg_mean_map1,
                                                                                             reg_std_map=reg_std_map1,
                                                                                             wv_ref=self.wv_ref,
                                                                                             mypool=mppool)
        self.bad_pixels = self.bad_pixels * new_badpixs


        subtracted_im[np.where(np.isnan(subtracted_im))] = 0

        if save_utils:
            if isinstance(save_utils,str):
                out_filename = save_utils
            else:
                out_filename = self.default_filenames["compute_starsubtraction_2dspline"]

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=subtracted_im))
            hdulist.append(pyfits.ImageHDU(data=im, name='IM'))
            hdulist.append(pyfits.ImageHDU(data=star_model, name='STARMODEL'))
            hdulist.append(pyfits.ImageHDU(data=self.bad_pixels, name='BADPIX'))
            hdulist.append(pyfits.ImageHDU(data=spline_paras0, name='SPLINE_PARAS0'))
            hdulist.append(pyfits.ImageHDU(data=self.wv_nodes, name='wv_nodes'))
            hdulist.append(pyfits.ImageHDU(data=self.ifuy_nodes, name='ifuy_nodes'))
            hdulist.writeto(out_filename, overwrite=True)

            if starsub_dir is not None:
                if not os.path.exists(os.path.join(self.utils_dir,starsub_dir)):
                    os.makedirs(os.path.join(self.utils_dir,starsub_dir))
                hdulist_sc = pyfits.open(self.filename)
                du = self.data_unit
                bu = self.extheader["BUNIT"].strip()

                if du == 'MJy'    and bu == 'MJy':
                    hdulist_sc["SCI"].data = subtracted_im
                if du == 'MJy/sr' and bu == 'MJy/sr':
                    hdulist_sc["SCI"].data = subtracted_im
                if du == 'MJy/sr' and bu == 'MJy':
                    hdulist_sc["SCI"].data = subtracted_im* self.area2d
                if du == 'MJy' and bu == 'MJy/sr':
                    hdulist_sc["SCI"].data = subtracted_im/self.area2d
                hdulist_sc["DQ"].data[np.where(np.isnan(self.bad_pixels))] = 1
                hdulist_sc.writeto(os.path.join(self.utils_dir,starsub_dir, os.path.basename(self.filename)), overwrite=True)
                hdulist_sc.close()
        return subtracted_im, star_model, spline_paras0, self.wv_nodes, self.ifuy_nodes


    def reload_starsubtraction_2dspline(self, load_filename=None):
        """ Reload Star Subtraction with 2D spline

        Parameters
        ----------
        load_filename : str or None
            Filename to load PSF subtracted data from, or leave None to use default filename

        Returns
        -------
        subtracted_im, star_model, spline_paras0, wv_nodes, ifuy_nodes

        """
        if load_filename is None:
            load_filename = self.default_filenames["compute_starsubtraction_2dspline"]
        if len(glob(load_filename)) ==0:
            return None

        hdulist = pyfits.open(load_filename)
        subtracted_im = hdulist[0].data
        star_model = hdulist[2].data
        fmderived_bad_pixels = hdulist[3].data
        spline_paras0 = hdulist[4].data
        wv_nodes = hdulist[5].data
        ifuy_nodes = hdulist[6].data
        hdulist.close()

        self.bad_pixels = self.bad_pixels * fmderived_bad_pixels
        self.wv_nodes = wv_nodes
        self.ifuy_nodes = ifuy_nodes
        return subtracted_im, star_model, spline_paras0, wv_nodes, ifuy_nodes


    def mask_interp_elements_too_far_from_bin_edges(self, dwv_threshold):
        """ Mask interpolated elements too far from the edge bins

        Parameters
        ----------
        dwv_threshold

        Returns
        -------
        mask : ndarray
            Mask of which pixels are masked

        Also modifies self.bad_pixels

        """
        if "regwvs" not in self.coords:
            raise Exception("'regwvs' in self.coords. This data object needs to be interpolated first.")
        dist_to_bin_edges = np.nanmin(np.abs(self.leftnright_wavelengths-self.wavelengths),axis=0)
        mask = dist_to_bin_edges>dwv_threshold
        self.bad_pixels[np.where(mask)] = np.nan
        return mask


def _task_normslice_2dspline(paras):
    """ Worker function for normalizing slices via 2d spline, for use in parallelized calculations

    Parameters
    ----------
    paras : tuple containing many values
        im, im_wvs, im_ifuy, noise, badpix, wv_nodes,ifuy_nodes, wv_ref, star_model, threshold, reg_mean_map, reg_std_map

    Returns
    -------

    """
    im, im_wvs, im_ifuy, noise, badpix, wv_nodes,ifuy_nodes, wv_ref, star_model, threshold, reg_mean_map, reg_std_map = paras

    new_im = np.zeros(im.shape)+np.nan#np.array(copy(im), '<f4')  # .byteswap().newbyteorder()
    new_noise = copy(noise)
    new_badpix = copy(badpix)
    res = np.zeros(im.shape) + np.nan

    bool_map = np.isfinite(new_badpix) * np.isfinite(im) * np.isfinite(noise) * (noise != 0) * np.isfinite(star_model) * np.isfinite(im_ifuy)
    where_data_finite = np.where(bool_map)
    if np.size(where_data_finite[0]) != 0:
        ravel_im_ifuy = im_ifuy[where_data_finite]
    ravel_im_wvs = im_wvs[where_data_finite]
    # M_spline_ifuy = get_spline_model(ifuy_nodes, ravel_im_ifuy, spline_degree=3)
    M_spline_ifuy = get_spline_model(ifuy_nodes, ravel_im_ifuy/ravel_im_wvs*wv_ref, spline_degree=3)
    M_spline_wvs = get_spline_model(wv_nodes, ravel_im_wvs, spline_degree=3)
    M_spline_ifuy_repeated = np.repeat(M_spline_ifuy, np.size(wv_nodes), axis=1)
    M_spline_wvs_tiled = np.tile(M_spline_wvs, (1, np.size(ifuy_nodes)))
    M_2dspline = M_spline_ifuy_repeated * M_spline_wvs_tiled

    d = im[where_data_finite]
    d_err = noise[where_data_finite]

    M = M_2dspline * star_model[where_data_finite][:, None]

    validpara = np.where(np.nansum(M > np.nanmax(M) * 0.005, axis=0) != 0)
    M = M[:, validpara[0]]

    if 1:
        d_reg, s_reg = np.ravel(reg_mean_map), np.ravel(reg_std_map)
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

    p = lsq_linear(M4fit / s4fit[:, None], d4fit / s4fit).x
    m = np.dot(M, p)
    res[where_data_finite] = d - m
    new_im[where_data_finite] = m
    new_noise[where_data_finite] = d_err
    norm_res = copy(res)
    norm_res[where_data_finite] = norm_res[where_data_finite] / d_err

    meddev = median_abs_deviation(norm_res[where_data_finite])
    where_bad = np.where((np.abs(norm_res) / meddev > threshold) | np.isnan(norm_res))
    new_badpix[where_bad] = np.nan

    paras_out = np.zeros((np.size(ifuy_nodes), np.size(wv_nodes))) + np.nan
    paras_out = np.ravel(paras_out)
    paras_out[validpara] = p
    paras_out = np.reshape(paras_out,(np.size(ifuy_nodes), np.size(wv_nodes)))

    return new_im, new_noise, new_badpix, res, paras_out


def normalize_slices_2dspline(image, im_wvs,im_ifuy, noise=None, badpixs=None,trace_id_map=None,
                              star_model=None,  mypool=None,
                              threshold=10, use_set_nans=False,
                              N_wvs_nodes=20, wv_nodes=None, delta_ifuy=0.05, ifuy_nodes=None,
                              reg_mean_map=None, reg_std_map=None, wv_ref = None):
    """ Normalize sliaces using a 2D spline

    Parameters
    ----------
    image
    im_wvs
    im_ifuy
    noise
    badpixs
    trace_id_map
    star_model
    mypool
    threshold
    use_set_nans
    N_wvs_nodes
    wv_nodes
    delta_ifuy
    ifuy_nodes
    reg_mean_map
    reg_std_map
    wv_ref

    Returns
    -------
    new_image, new_noise, new_badpixs, new_res, new_spline_paras

    """

    if noise is None:
        noise = np.ones(image.shape)
    if badpixs is None:
        badpixs = np.ones(image.shape)
    if star_model is None:
        star_model = np.ones(image.shape)
    if trace_id_map is None:
        trace_id_map = np.zeros(image.shape)


    if wv_nodes is None:
        wv_nodes = np.linspace(np.nanmin(im_wvs), np.nanmax(im_wvs), N_wvs_nodes, endpoint=True)

    if ifuy_nodes is None:
        ifuy_min, ifuy_max = np.nanmin(im_ifuy), np.nanmax(im_ifuy)
        ifuy_min, ifuy_max = np.floor(ifuy_min * 10) / 10, np.ceil(ifuy_max * 10) / 10
        ifuy_nodes = np.arange(ifuy_min, ifuy_max + 0.1, delta_ifuy)

    if wv_ref is None:
        wv_ref = np.nanmin(im_wvs)

    parallel_flag = True

    unique_trace_ids = np.unique(trace_id_map[np.where(np.isfinite(trace_id_map))])

    new_image = copy(image)
    if use_set_nans:
        new_image = set_nans(image, 40)
    new_noise = copy(noise)
    new_res = np.zeros(image.shape) + np.nan
    new_badpixs = np.zeros(image.shape) + np.nan
    new_spline_paras = np.zeros((np.size(unique_trace_ids),np.size(ifuy_nodes), np.size(wv_nodes)))

    scaled_cloud = im_ifuy * wv_ref / im_wvs
    bool_map = (scaled_cloud<np.min(ifuy_nodes)) | (scaled_cloud>np.max(ifuy_nodes)) | \
               (im_wvs<np.min(wv_nodes)) | (im_wvs>np.max(wv_nodes))
    where2mask = np.where(bool_map)
    badpixs_nodes_mask = np.ones(badpixs.shape)
    badpixs_nodes_mask[where2mask] = np.nan

    if (mypool is None) or (parallel_flag == False):
        print("\tPerforming serial normalize_slices_2dspline...")
        for id,trace_id in enumerate(unique_trace_ids):
            trace_mask = trace_id_map == trace_id
            where_in_trace = np.where(trace_mask)
            tmp_badpixs = np.zeros(badpixs.shape)+np.nan
            tmp_badpixs[where_in_trace] = (badpixs_nodes_mask*badpixs)[where_in_trace]
            row_id_min,row_id_max = np.min(where_in_trace[0]),np.max(where_in_trace[0])

            paras = new_image[row_id_min:row_id_max,:], im_wvs[row_id_min:row_id_max,:], im_ifuy[row_id_min:row_id_max,:], \
                new_noise[row_id_min:row_id_max,:], tmp_badpixs[row_id_min:row_id_max,:], wv_nodes,ifuy_nodes,wv_ref, \
                star_model[row_id_min:row_id_max,:], threshold, reg_mean_map[id], reg_std_map[id]

            outputs = _task_normslice_2dspline(paras)
            partial_new_image, partial_new_noise, partial_new_badpixs, partial_new_res, partial_new_spline_paras = outputs

            new_image[row_id_min:row_id_max,:] = partial_new_image
            new_noise[row_id_min:row_id_max,:] = partial_new_noise
            new_badpixs[row_id_min:row_id_max,:] = partial_new_badpixs
            new_res[row_id_min:row_id_max,:] = partial_new_res
            new_spline_paras[id,:,:] = partial_new_spline_paras
    else:
        print("\tPerforming parallelized normalize_slices_2dspline...")
        row_indices_list = []
        image_list = []
        wvs_list = []
        im_ifuy_list = []
        noise_list = []
        badpixs_list = []
        star_model_list = []
        for id, trace_id in enumerate(unique_trace_ids):
            trace_mask = (trace_id_map == trace_id)
            where_in_trace = np.where(trace_mask)
            tmp_badpixs = np.zeros(badpixs.shape)+np.nan
            tmp_badpixs[where_in_trace] = (badpixs_nodes_mask*badpixs)[where_in_trace]
            row_id_min,row_id_max = np.min(where_in_trace[0]),np.max(where_in_trace[0])
            row_indices_list.append((row_id_min,row_id_max ))

            image_list.append(new_image[row_id_min:row_id_max, :])
            wvs_list.append(im_wvs[row_id_min:row_id_max, :])
            im_ifuy_list.append(im_ifuy[row_id_min:row_id_max, :])
            noise_list.append(new_noise[row_id_min:row_id_max, :])
            badpixs_list.append(tmp_badpixs[row_id_min:row_id_max, :])
            star_model_list.append(star_model[row_id_min:row_id_max, :])

        outputs_list = mypool.map(_task_normslice_2dspline, zip(image_list, wvs_list, im_ifuy_list, noise_list, badpixs_list,
                                                      itertools.repeat(wv_nodes),
                                                      itertools.repeat(ifuy_nodes),
                                                      itertools.repeat(wv_ref),
                                                      star_model_list,
                                                      itertools.repeat(threshold),
                                                      reg_mean_map,reg_std_map))
        for id,((row_id_min,row_id_max), outputs) in enumerate(zip(row_indices_list, outputs_list)):
            partial_new_image, partial_new_noise, partial_new_badpixs, partial_new_res, partial_new_spline_paras = outputs
            new_image[row_id_min:row_id_max, :] = partial_new_image
            new_noise[row_id_min:row_id_max, :] = partial_new_noise
            new_badpixs[row_id_min:row_id_max, :] = partial_new_badpixs
            new_res[row_id_min:row_id_max, :] = partial_new_res
            new_spline_paras[id, :, :] = partial_new_spline_paras

    return new_image, new_noise, new_badpixs, new_res, new_spline_paras


def PCA_detec(im, im_err, im_badpixs, N_KL=5):
    """ Detect (something??) using Princople Component Analyses

    Parameters
    ----------
    im
    im_err
    im_badpixs
    N_KL : int
        Number of KL modes

    Returns
    -------
    kls

    """
    im_cp = im * im_badpixs / im_err

    new_im = im_cp[np.where(np.nansum(np.isfinite(im_cp), axis=1) > im.shape[1] // 2)[0], :]
    ny, nx = new_im.shape
    med_spec = np.nanmedian(new_im, axis=0)
    where_nan = np.where(np.isnan(new_im))
    new_im[where_nan] = med_spec[where_nan[1]]

    X = new_im
    X = X[np.where(np.nansum(X, axis=1) != 0)[0], :]
    X = X / np.nanstd(X, axis=1)[:, None]
    X[np.where(np.isnan(X))] = np.tile(np.nanmedian(X, axis=0)[None, :], (X.shape[0], 1))[np.where(np.isnan(X))]
    X[np.where(np.isnan(X))] = 0

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
    kls = kl_basis * (1. / np.sqrt(evals * (nx - 1)))[None, :]  # multiply a value for each row
    print(kls.shape)

    return kls


def PCA_wvs_axis(wavelengths, im, im_err, im_badpixs, bin_size, N_KL=5):
    """Perform PCA along the wavelength axis

    Parameters
    ----------
    wavelengths
    im
    im_err
    im_badpixs
    bin_size
    N_KL : int
        Number of KL modes

    Returns
    -------
    new_wvs, kls

    """
    ny, nx = im.shape

    new_wvs = np.arange(np.nanmin(wavelengths*im_badpixs), np.nanmax(wavelengths*im_badpixs), bin_size)
    nz = np.size(new_wvs)
    new_im = np.zeros((ny, nz))+  np.nan
    for k in range(ny):
        x = wavelengths[k]
        y = im[k]/im_err[k]
        q = im_badpixs[k]
        s = im_err[k]

        where_finite = np.where(np.isfinite(q) * np.isfinite(y) * (s != 0.0))
        if np.size(where_finite[0]) < nx // 4:
            continue
        f = interp1d(x[where_finite], y[where_finite], bounds_error=False, fill_value=np.nan, kind="linear")
        new_im[k, :] = f(new_wvs)
    new_im[:,np.where(np.sum(np.isfinite(new_im),axis=0)<100)[0]]=np.nan
    new_im = new_im[np.where(np.sum(np.isfinite(new_im),axis=1)!=0)[0],:]

    new_im = new_im / np.nanstd(new_im, axis=1)[:, None]
    where_nan = np.where(np.isnan(new_im))
    new_im[where_nan] = 0

    X = new_im
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
    print(kls.shape)

    return new_wvs, kls


def combine_spectrum_1dspline(wavelengths, fluxes, errors, bin_size, oversampling=10):
    """Combine a spectrum using a 1d epline

    Parameters
    ----------
    wavelengths
    fluxes
    errors
    bin_size
    oversampling

    Returns
    -------
    hd_wvs
    splev(hd_wvs, spl)
    err_func(hd_wvs)
    spl

    """
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

    spl = splrep(wavelengths, fluxes, k=3, t=new_wavelengths[1:(np.size(new_wavelengths)-1)], task=-1, s=None, w=1 / errors)

    hd_wvs = np.arange(new_wavelengths[0],new_wavelengths[-1], bin_size / oversampling)
    return hd_wvs, splev(hd_wvs, spl),err_func(hd_wvs),spl
