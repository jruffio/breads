import os.path

from matplotlib.pyplot import axis
import matplotlib.pyplot as plt
from breads.instruments.instrument import Instrument
import breads.utils as utils
from warnings import warn
import astropy.io.fits as pyfits
import numpy as np
import ctypes
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.time import Time
from copy import copy
from breads.utils import broaden
from breads.calibration import SkyCalibration
import multiprocessing as mp
import pandas as pd
import astropy
import jwst.datamodels, jwst.assign_wcs
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.simbad import Simbad
from astropy.time import Time
from glob import glob
import itertools
from scipy.interpolate import interp1d
from astropy.stats import sigma_clip
from breads.utils import get_spline_model
from scipy.optimize import lsq_linear
import astropy
import webbpsf
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import curve_fit
from jwst.photom.photom import DataSet
from stdatamodels.jwst import datamodels
from scipy.signal import convolve2d
import scipy.linalg as la
from scipy.optimize import minimize
from astropy import constants as const
from scipy.stats import median_abs_deviation

from breads.utils import rotate_coordinates

import matplotlib.tri as tri
import numpy as np
from scipy.ndimage import generic_filter





class jwstnirpsec_cal(Instrument):
    def __init__(self, filename=None, crds_dir=None, utils_dir=None, save_utils=True,
                 load_utils=True,
                 load_coords=True,
                 load_interpdata_regwvs=True,
                 external_dir=None, mppool=None,
                 mask_charge_bleeding=True, compute_wpsf=True, compute_starspec_contnorm=True, compute_starsub=True,
                 compute_interp_regwvs=True, fit_wpsf=True,init_fit_psf=False,
                 spec_R_sampling=None, N_nodes=None, recenter_from_webbpsf=True, coords_offset=None,
                 regwvs_sampling=None, wpsffit_IWA=0.0, wpsffit_OWA=1.0,threshold_badpix=10,
                 apply_chargediff_mask=True):
        super().__init__('jwstnirpsec')
        if filename is None:
            warning_text = "No data file provided. " + \
                           "Please manually add data or use jwstnirpsec.read_data_file()"
            warn(warning_text)
        else:
            self.read_data_file(filename, crds_dir=crds_dir, utils_dir=utils_dir, save_utils=save_utils,
                                external_dir=external_dir,
                                load_utils=load_utils,
                                load_coords=load_coords,
                                load_interpdata_regwvs=load_interpdata_regwvs,
                                mppool=mppool,
                                mask_charge_bleeding=mask_charge_bleeding, compute_wpsf=compute_wpsf,
                                compute_starspec_contnorm=compute_starspec_contnorm, compute_starsub=compute_starsub,
                                compute_interp_regwvs=compute_interp_regwvs, fit_wpsf=fit_wpsf,init_fit_psf=init_fit_psf,
                                spec_R_sampling=spec_R_sampling, N_nodes=N_nodes,
                                recenter_from_webbpsf=recenter_from_webbpsf,
                                coords_offset=coords_offset, regwvs_sampling=regwvs_sampling,
                                wpsffit_IWA=wpsffit_IWA, wpsffit_OWA=wpsffit_OWA,threshold_badpix=threshold_badpix,
                       apply_chargediff_mask=apply_chargediff_mask)

    def read_data_file(self, filename, crds_dir=None, utils_dir=None, save_utils=True, external_dir=None,
                       load_utils=True,
                       load_coords=True,
                       load_interpdata_regwvs=True,
                       mppool=None,
                       mask_charge_bleeding=True, compute_wpsf=True, compute_starspec_contnorm=True,
                       compute_starsub=True, compute_interp_regwvs=True, fit_wpsf=True,init_fit_psf=False,
                       spec_R_sampling=None, N_nodes=None, recenter_from_webbpsf=True, max_MJy=1e-5,
                       coords_offset=None, regwvs_sampling=None, wpsffit_IWA=0.0, wpsffit_OWA=1.0,threshold_badpix=10,
                       apply_chargediff_mask=True):
        """
        Read OSIRIS spectral cube, also checks validity at the end
        """
        self.filename = filename
        if utils_dir is None:
            utils_dir = os.path.dirname(self.filename)
        hdulist_sc = pyfits.open(self.filename)
        priheader = hdulist_sc[0].header
        extheader = hdulist_sc[1].header
        im = hdulist_sc["SCI"].data
        err = hdulist_sc["ERR"].data
        dq = hdulist_sc["DQ"].data
        im_wvs = hdulist_sc["WAVELENGTH"].data
        ny, nx = im.shape

        self.compute_wpsf = compute_wpsf
        self.compute_starspec_contnorm = compute_starspec_contnorm
        self.compute_starsub = compute_starsub
        self.compute_interp_regwvs = compute_interp_regwvs
        self.fit_wpsf = fit_wpsf

        self.wavelengths = im_wvs
        self.data = im
        self.noise = err
        where_zero_noise = np.where(self.noise == 0)
        self.noise[where_zero_noise] = np.nan
        self.bad_pixels = np.ones((ny, nx))
        self.bad_pixels[np.where(untangle_dq(dq)[0, :, :])] = np.nan
        self.bad_pixels[np.where(np.isnan(self.data))] = np.nan
        # self.bad_pixels[np.where(self.data<=0)] = np.nan
        self.bad_pixels[where_zero_noise] = np.nan
        # self.bad_pixels[np.where(np.abs(self.data)>max_MJy)] = np.nan
        self.starsub_filename = os.path.join(utils_dir, os.path.basename(filename).replace(".fits", "_starsub.fits"))
        if not(self.compute_starsub and load_utils and len(glob(self.starsub_filename))):
            for rowid in range(self.bad_pixels.shape[0]):
                row_err = self.noise[rowid,:]
                row_err = row_err - generic_filter(row_err, np.nanmedian, size=50)
                row_err_masking = row_err/median_abs_deviation(row_err[np.where(np.isfinite(self.bad_pixels[rowid,:]))])
                # if rowid == 305:
                #     plt.plot(row_err_masking)
                #     plt.ylim([-10,50])
                #     plt.show()
                self.bad_pixels[rowid,np.where((row_err_masking>5e1))[0]] = np.nan

        # todo this is definitely an approximation, curvature of the trace? can be done better?
        self.delta_wavelengths = self.wavelengths[:, 1::] - self.wavelengths[:, 0:self.wavelengths.shape[1] - 1]
        self.delta_wavelengths = np.concatenate([self.delta_wavelengths, self.delta_wavelengths[:, -1][:, None]],
                                                axis=1)

        self.priheader = priheader
        self.extheader = extheader
        self.east2V2_deg = -(float(extheader["ROLL_REF"]) + float(extheader["V3I_YANG"]))
        self.wpsf_ra_offset, self.wpsf_dec_offset, self.wpsf_angle_offset = 0.0, 0.0, 0.0
        self.wpsffit_IWA, self.wpsffit_OWA = wpsffit_IWA, wpsffit_OWA
        linear_interp = True
        self.crds_dir = crds_dir

        self.bary_RV = 0  # Already corrected in wavecal for JWST. float(self.extheader["VELOSYS"])/1000 # in km/s
        self.R = 2700
        if spec_R_sampling is None:
            spec_R_sampling = 4 * self.R
        if N_nodes is None:
            self.N_nodes = 40
        else:
            self.N_nodes = N_nodes

        self.coords_filename = os.path.join(utils_dir, os.path.basename(filename).replace(".fits", "_relcoords.fits"))
        print(len(glob(self.coords_filename)), self.coords_filename)
        if 1 and load_utils and load_coords and len(glob(self.coords_filename)):
            with pyfits.open(self.coords_filename) as hdulist:
                wavelen_array = hdulist[0].data
                dra_as_array = hdulist[1].data
                ddec_as_array = hdulist[2].data
                area2d = hdulist[3].data
        else:
            Simbad.add_votable_fields("pmra")  # Store proper motion in RA
            Simbad.add_votable_fields("pmdec")  # Store proper motion in Dec.

            result_table = Simbad.query_object(self.priheader["TARGNAME"])
            # Get the coordinates and proper motion from the result table
            ra = result_table["RA"][0]
            dec = result_table["DEC"][0]
            pm_ra = result_table["PMRA"][0]
            pm_dec = result_table["PMDEC"][0]

            # Create a SkyCoord object with the coordinates and proper motion
            HD19467_coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg),
                                     pm_ra_cosdec=pm_ra * u.mas / u.year,
                                     pm_dec=pm_dec * u.mas / u.year,
                                     frame='icrs', obstime='J2000.0')
            desired_date = self.priheader["DATE-OBS"]  # '2023-01-25'  # Example date in ISO format
            # Convert the desired date to an astropy Time object
            t = Time(desired_date)
            # Calculate the updated SkyCoord object for the desired date
            host_coord = HD19467_coord.apply_space_motion(new_obstime=t)
            host_ra_deg = host_coord.ra.deg
            host_dec_deg = host_coord.dec.deg

            calfile = jwst.datamodels.open(filename)
            photom_dataset = DataSet(calfile)
            area_fname = self.priheader["R_AREA"].replace("crds://", os.path.join(self.crds_dir, "references", "jwst","nirspec") + os.path.sep)
            # print(area_fname)
            # exit()
            # Load the pixel area table for the IFU slices
            area_model = datamodels.open(area_fname)
            area_data = area_model.area_table

            # Compute 2D wavelength and pixel area arrays for the whole image
            wave2d, area2d, dqmap = photom_dataset.calc_nrs_ifu_sens2d(area_data)
            area2d[np.where(area2d == 1)] = np.nan

            wcses = jwst.assign_wcs.nrs_ifu_wcs(calfile)  # returns a list of 30 WCSes, one per slice. This is slow.

            ra_array = np.zeros((2048, 2048)) + np.nan
            dec_array = np.zeros((2048, 2048)) + np.nan
            wavelen_array = np.zeros((2048, 2048)) + np.nan

            slicer_x_array = np.zeros((2048, 2048)) + np.nan
            slicer_y_array = np.zeros((2048, 2048)) + np.nan
            slicer_w_array = np.zeros((2048, 2048)) + np.nan

            for i in range(30):
                print(f"Computing coords for slice {i}")

                # Set up 2D X, Y index arrays spanning across the full area of the slice WCS
                xmin = max(int(np.round(wcses[i].bounding_box.intervals[0][0])), 0)
                xmax = int(np.round(wcses[i].bounding_box.intervals[0][1]))
                ymin = max(int(np.round(wcses[i].bounding_box.intervals[1][0])), 0)
                ymax = int(np.round(wcses[i].bounding_box.intervals[1][1]))
                # print(xmax, xmin,ymax, ymin,ymax - ymin,xmax - xmin)

                x = np.arange(xmin, xmax)
                x = x.reshape(1, x.shape[0]) * np.ones((ymax - ymin, 1))
                y = np.arange(ymin, ymax)
                y = y.reshape(y.shape[0], 1) * np.ones((1, xmax - xmin))

                # Transform all those pixels to RA, Dec, wavelength
                skycoords, speccoord = wcses[i](x, y, with_units=True)
                # print(skycoords.ra)

                ra_array[ymin:ymax, xmin:xmax] = skycoords.ra
                dec_array[ymin:ymax, xmin:xmax] = skycoords.dec
                wavelen_array[ymin:ymax, xmin:xmax] = speccoord

                # Transform all those pixels to the slicer plane
                slice_transform = wcses[i].get_transform('detector', 'slicer')

                sx, sy, sw = slice_transform(x, y)

                slicer_x_array[ymin:ymax, xmin:xmax] = sx
                slicer_y_array[ymin:ymax, xmin:xmax] = sy
                slicer_w_array[ymin:ymax, xmin:xmax] = sw

            # # print(ra_array)
            # print(host_ra_deg)
            # print(np.nanmedian(ra_array))
            # print(np.nanmedian(slicer_x_array))
            # exit()
            dra_as_array = (ra_array - host_ra_deg) * 3600 * np.cos(np.radians(dec_array))
            ddec_as_array = (dec_array - host_dec_deg) * 3600

            if save_utils:
                hdulist = pyfits.HDUList()
                hdulist.append(pyfits.PrimaryHDU(data=wavelen_array))
                hdulist.append(pyfits.ImageHDU(data=dra_as_array))
                hdulist.append(pyfits.ImageHDU(data=ddec_as_array))
                hdulist.append(pyfits.ImageHDU(data=area2d))
                try:
                    hdulist.writeto(self.coords_filename, overwrite=True)
                except TypeError:
                    hdulist.writeto(self.coords_filename, clobber=True)
                hdulist.close()

        self.dra_as_array, self.ddec_as_array, self.area2d = dra_as_array, ddec_as_array, area2d
        # print(np.nanmedian(self.dra_as_array[1::,:]-self.dra_as_array[0:self.dra_as_array.shape[0]-1,:]))
        # print(np.nanmedian(self.ddec_as_array[1::,:]-self.ddec_as_array[0:self.ddec_as_array.shape[0]-1,:]))
        # return
        # exit()
        # print(coords_offset)
        if coords_offset is not None:
            # print("coucou")
            self.dra_as_array -= coords_offset[0]
            self.ddec_as_array -= coords_offset[1]
        # exit()
        if regwvs_sampling is None:
            wv_sampling = self.get_regwvs_sampling()
        else:
            wv_sampling = regwvs_sampling
        self.wv_sampling = wv_sampling

        # self.webbpsf_filename = os.path.join(utils_dir,self.priheader["DATE-OBS"]+"_"+self.priheader["FILTER"]+"_"+self.priheader["GRATING"]+"_webbpsf.fits")
        splitbasename = os.path.basename(filename).split("_")
        self.webbpsf_filename = os.path.join(utils_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[3] + "_webbpsf.fits")
        print(len(glob(self.webbpsf_filename)), self.webbpsf_filename)
        # wpsf_image_mask = 'IFU'
        wpsf_image_mask = None
        wpsf_pixelscale = 0.1
        wpsf_oversample = 10
        source_offset_x = 0.0
        source_offset_y = 0.0  # -0.7
        if self.compute_wpsf and load_utils and len(glob(self.webbpsf_filename)):
            # print("coucou0")
            # with pyfits.open(self.webbpsf_filename) as hdulist:
            hdulist = pyfits.open(self.webbpsf_filename)
            wpsfs = hdulist[0].data
            wpsfs_header = hdulist[0].header
            wepsfs = hdulist[1].data
            webbpsf_wvs = hdulist[2].data
            webbpsf_X = hdulist[3].data
            webbpsf_Y = hdulist[4].data
            wpsf_pixelscale = wpsfs_header["PIXELSCL"]
            wpsf_oversample = wpsfs_header["oversamp"]

            # self.aper_to_epsf_peak_f = interp1d(webbpsf_wvs, aper_to_epsf_peak, bounds_error=False,fill_value=np.nan)
            try:
                aper_to_epsf_peak = hdulist[5].data
                # print("coucou0")
                self.aper_to_epsf_peak_f = interp1d(webbpsf_wvs, aper_to_epsf_peak, bounds_error=False,fill_value=np.nan)
            except:
                print("coucou except")
                webbpsf_R = np.sqrt(webbpsf_X ** 2 + webbpsf_Y ** 2)
                mask_aper = np.ones(webbpsf_R.shape)
                mask_aper[np.where(webbpsf_R > 3)] = np.nan
                tiled_mask_aper = np.tile(mask_aper[None, :, :], (wpsfs.shape[0], 1, 1))
                aper_phot_webb_psf = np.nansum(wpsfs * tiled_mask_aper,axis=(1, 2))
                peak_webb_epsf = np.nanmax(wepsfs, axis=(1, 2))

                self.aper_to_epsf_peak_f = interp1d(webbpsf_wvs, peak_webb_epsf / aper_phot_webb_psf, bounds_error=False, fill_value=np.nan)

                wpsfs_header = {"PIXELSCL": wpsf_pixelscale, "im_mask": wpsfs_header["im_mask"],
                                "oversamp": wpsf_oversample, "DATE-BEG": wpsfs_header["DATE-BEG"],
                                "offset_x": wpsfs_header["offset_x"], "offset_y": wpsfs_header["offset_y"]}
                _hdulist = pyfits.HDUList()
                _hdulist.append(pyfits.PrimaryHDU(data=wpsfs, header=pyfits.Header(cards=wpsfs_header)))
                _hdulist.append(pyfits.ImageHDU(data=wepsfs))
                _hdulist.append(pyfits.ImageHDU(data=webbpsf_wvs))
                _hdulist.append(pyfits.ImageHDU(data=webbpsf_X))
                _hdulist.append(pyfits.ImageHDU(data=webbpsf_Y))
                _hdulist.append(pyfits.ImageHDU(data=peak_webb_epsf / aper_phot_webb_psf))
                try:
                    _hdulist.writeto(self.webbpsf_filename, overwrite=True)
                except TypeError:
                    _hdulist.writeto(self.webbpsf_filename, clobber=True)
                _hdulist.close()
                # exit()
            hdulist.close()
        elif self.compute_wpsf:
            nrs = webbpsf.NIRSpec()
            nrs.load_wss_opd_by_date(priheader["DATE-BEG"])  # Load telescope state as of our observation date
            nrs.image_mask = wpsf_image_mask  # optional: model opaque field stop outside of the IFU aperture
            nrs.pixelscale = wpsf_pixelscale  # Optional: set this manually to match the drizzled cube sampling, rather than the default
            nrs.options["source_offset_x"] = source_offset_x
            nrs.options["source_offset_x"] = source_offset_y
            if 1:
                # wv_sampling = wv_sampling[0:10]
                outarr_not_created = True
                for wv_id, wv in enumerate(wv_sampling):
                    print(wv_id, wv, np.size(wv_sampling))
                    paras = nrs, wv, wpsf_oversample
                    out = _get_wpsf_task(paras)
                    # plt.imshow(out[0])
                    # plt.show()
                    if outarr_not_created:
                        wpsfs = np.zeros((np.size(wv_sampling), out[0].shape[0], out[0].shape[1]))
                        wepsfs = np.zeros((np.size(wv_sampling), out[0].shape[0], out[0].shape[1]))
                        outarr_not_created = False
                    wpsfs[wv_id, :, :] = out[0]
                    wepsfs[wv_id, :, :] = out[1]
            # else:
            #     output_lists = mppool.map(_get_wpsf_task, zip(itertools.repeat(nrs),
            #                                                   wv_sampling,
            #                                                  itertools.repeat(wpsf_oversample)))
            #
            #     outarr_not_created = True
            #     for wv_id, (wv,out) in enumerate(zip(wv_sampling,output_lists)):
            #         if outarr_not_created:
            #             wpsfs = np.zeros((np.size(wv_sampling),out[0].shape[0],out[0].shape[1]))
            #             wepsfs = np.zeros((np.size(wv_sampling),out[0].shape[0],out[0].shape[1]))
            #             outarr_not_created=False
            #         wpsfs[wv_id,:,:] = out[0]
            #         wepsfs[wv_id,:,:] = out[1]

            # print(psf_array_shape,pixelscale)
            halffov_x = wpsf_pixelscale / wpsf_oversample * wpsfs.shape[2] / 2.0
            halffov_y = wpsf_pixelscale / wpsf_oversample * wpsfs.shape[1] / 2.0
            x = np.linspace(-halffov_x, halffov_x, wpsfs.shape[2], endpoint=True)
            y = np.linspace(-halffov_y, halffov_y, wpsfs.shape[1], endpoint=True)
            webbpsf_X, webbpsf_Y = np.meshgrid(x, y)

            webbpsf_R = np.sqrt(webbpsf_X ** 2 + webbpsf_Y ** 2)
            mask_aper = np.ones(webbpsf_R.shape)
            mask_aper[np.where(webbpsf_R > 3)] = np.nan
            tiled_mask_aper = np.tile(mask_aper[None, :, :], (wpsfs.shape[0], 1, 1))
            aper_phot_webb_psf = np.nansum(wpsfs * tiled_mask_aper,axis=(1, 2))
            peak_webb_epsf = np.nanmax(wepsfs, axis=(1, 2))
            self.aper_to_epsf_peak_f = interp1d(wv_sampling, peak_webb_epsf / aper_phot_webb_psf, bounds_error=False, fill_value=np.nan)

            # plt.imshow(wpsfs[0])
            # plt.show()
            if save_utils:
                wpsfs_header = {"PIXELSCL": wpsf_pixelscale, "im_mask": wpsf_image_mask,
                                "oversamp": wpsf_oversample, "DATE-BEG": priheader["DATE-BEG"],
                                "offset_x": source_offset_x, "offset_y": source_offset_y}
                hdulist = pyfits.HDUList()
                hdulist.append(pyfits.PrimaryHDU(data=wpsfs, header=pyfits.Header(cards=wpsfs_header)))
                hdulist.append(pyfits.ImageHDU(data=wepsfs))
                hdulist.append(pyfits.ImageHDU(data=wv_sampling))
                hdulist.append(pyfits.ImageHDU(data=webbpsf_X))
                hdulist.append(pyfits.ImageHDU(data=webbpsf_Y))
                hdulist.append(pyfits.ImageHDU(data=peak_webb_epsf / aper_phot_webb_psf))
                try:
                    hdulist.writeto(self.webbpsf_filename, overwrite=True)
                except TypeError:
                    hdulist.writeto(self.webbpsf_filename, clobber=True)
                hdulist.close()

                webbpsf_wvs = wv_sampling
        # # exit()
        # # print(np.nanmin(webbpsf_X),np.nanmin(webbpsf_Y))
        # # exit()
        # print("coucou11")
        # # wepsfs = wepsfs * (wpsf_pixelscale / wpsf_oversample) ** 2 #slow
        # print("coucou22")
        # webbpsf_R = np.sqrt(webbpsf_X ** 2 + webbpsf_Y ** 2)
        # print("coucou33")
        # # if 1:
        # #     mask_aper = np.ones(webbpsf_R.shape)
        # #     mask_aper[np.where(webbpsf_R > 0.75)] = np.nan
        # #     tiled_mask_aper = np.tile(mask_aper[None, :, :], (wpsfs.shape[0], 1, 1))
        # #     aper_phot_webb_psf = np.nansum(wpsfs * tiled_mask_aper, axis=(1, 2)) * (wpsf_pixelscale / wpsf_oversample) ** 2
        # # plt.plot(webbpsf_wvs,aper_phot_webb_psf,label="psf0")
        # mask_aper = np.ones(webbpsf_R.shape)
        # print("coucou44")
        # mask_aper[np.where(webbpsf_R > 3)] = np.nan
        # print("coucou55")
        # tiled_mask_aper = np.tile(mask_aper[None, :, :], (wpsfs.shape[0], 1, 1))
        # print("coucou66")
        # aper_phot_webb_psf = np.nansum(wpsfs * tiled_mask_aper, axis=(1, 2)) #* (wpsf_pixelscale / wpsf_oversample) ** 2 #slow
        # # plt.plot(webbpsf_wvs,aper_phot_webb_psf,label="psf")
        # # # plt.plot(webbpsf_wvs,np.nansum((wepsfs*tiled_mask_aper)[:,::10,::10],axis=(1,2)),label="epsf")
        # # if 1:
        # #     mask_aper = np.ones(webbpsf_R.shape)
        # #     mask_aper[np.where(webbpsf_R > 3)] = np.nan
        # #     tiled_mask_aper = np.tile(mask_aper[None, :, :], (wpsfs.shape[0], 1, 1))
        # #     aper_phot_webb_psf2 = np.nansum(wpsfs * tiled_mask_aper, axis=(1, 2)) * (wpsf_pixelscale / wpsf_oversample) ** 2
        # # plt.plot(webbpsf_wvs,aper_phot_webb_psf/aper_phot_webb_psf2,label="psf2")
        # # plt.legend()
        # # plt.show()
        # print("coucou77")
        # peak_webb_epsf = np.nanmax(wepsfs, axis=(1, 2))
        # print("coucou88")
        # self.aper_to_epsf_peak_f = interp1d(webbpsf_wvs, peak_webb_epsf / aper_phot_webb_psf, bounds_error=False,fill_value=np.nan)
        # # # plt.plot(peak_webb_epsf,label="peak_webb_epsf")
        # # # plt.plot(aper_phot_webb_psf,label="aper_phot_webb_psf")
        # # plt.plot(webbpsf_wvs,self.aper_to_epsf_peak_f(webbpsf_wvs))
        # # plt.legend()
        # # plt.show()
        # print("coucou99")
        # self.webbpsf_spaxel_area = (wpsf_pixelscale) ** 2
        # # print(wpsf_oversample)
        # # # kernel = np.ones((wpsf_oversample, wpsf_oversample))
        # # # smoothed_im = convolve2d(kernel, kernel, mode='same')
        # # # plt.imshow(smoothed_im)
        # # # plt.show()
        # # plt.plot(webbpsf_wvs,np.nansum((wepsfs*tiled_mask_aper)[:,::10,::10]/peak_webb_epsf[:,None,None],axis=(1,2))/self.aper_to_epsf_peak_f(webbpsf_wvs),label="1")
        # # plt.plot(webbpsf_wvs,np.nansum((wepsfs*tiled_mask_aper)[:,::10,::10]/peak_webb_epsf[:,None,None],axis=(1,2))*self.aper_to_epsf_peak_f(webbpsf_wvs),label="2")
        # # plt.legend()
        # # plt.show()
        # print("coucou2")


        self.webbpsf_spaxel_area = (wpsf_pixelscale) ** 2
        psf_wv0_id = np.size(webbpsf_wvs)//2
        self.webbpsf_im = wepsfs[psf_wv0_id]/np.nanmax(wepsfs[psf_wv0_id,:,:])
        self.webbpsf_X = webbpsf_X
        self.webbpsf_Y = webbpsf_Y
        self.webbpsf_wv0 = webbpsf_wvs[psf_wv0_id]
        wX, wY = rotate_coordinates(self.webbpsf_X.flatten(), self.webbpsf_Y.flatten(), -self.east2V2_deg, flipx=True)
        self.webbpsf_interp = CloughTocher2DInterpolator((wX, wY), self.webbpsf_im.flatten(),fill_value=0.0)


        self.barmask_filename = os.path.join(utils_dir, os.path.basename(filename).replace(".fits", "_barmask.fits"))
        print(len(glob(self.barmask_filename)), self.barmask_filename)
        if 1 and mask_charge_bleeding and load_utils and len(glob(self.barmask_filename)):
            with pyfits.open(self.barmask_filename) as hdulist:
                self.bar_mask = hdulist[0].data
                self.wpsf_angle_offset = hdulist[0].header["INIT_ANG"]
                self.wpsf_ra_offset = hdulist[0].header["INIT_RA"]
                self.wpsf_dec_offset = hdulist[0].header["INIT_DEC"]
            # plt.figure(1)
            # plt.subplot(1,2,1)
            # plt.imshow(self.data,interpolation="nearest",origin="lower")
            # plt.subplot(1,2,2)
            # plt.imshow(bar_mask,interpolation="nearest",origin="lower")
            # plt.show()

        elif mask_charge_bleeding:
            # rough centroid fit
            wv_id = np.size(wv_sampling) // 2
            fit_cen, fit_angle = True, False
            # init_paras = np.array([self.wpsf_ra_offset, self.wpsf_dec_offset, self.wpsf_angle_offset])
            init_paras = np.array([self.wpsf_ra_offset, self.wpsf_dec_offset])
            # paras = wepsfs[wv_id,:,:], webbpsf_X, webbpsf_Y,self.east2V2_deg,\
            # interp_ra[:,wv_id], interp_dec[:,wv_id], interp_flux[:,wv_id], interp_err[:,wv_id],interp_badpix[:,wv_id],\
            # IWA,OWA,fit_cen,fit_angle,init_paras
            wheredata2fit = np.where(
                (self.wavelengths > (wv_sampling[0] + 0.2)) * (self.wavelengths < (wv_sampling[-1] - 0.2)))
            paras = linear_interp, wepsfs[wv_id, :, :], webbpsf_X, webbpsf_Y, self.east2V2_deg,True, \
                self.dra_as_array[wheredata2fit] / self.wavelengths[wheredata2fit] * wv_sampling[wv_id], \
                self.ddec_as_array[wheredata2fit] / self.wavelengths[wheredata2fit] * wv_sampling[wv_id], \
                self.data[wheredata2fit], self.noise[wheredata2fit], self.bad_pixels[wheredata2fit], \
                self.wpsffit_IWA, self.wpsffit_OWA, fit_cen, fit_angle, init_paras
            out, _ = _fit_wpsf_task(paras)
            # print(out)
            init_paras = np.array([out[0,2], out[0,3]])
            paras = linear_interp, wepsfs[wv_id, :, :], webbpsf_X, webbpsf_Y, self.east2V2_deg,True, \
                self.dra_as_array[wheredata2fit] / self.wavelengths[wheredata2fit] * wv_sampling[wv_id], \
                self.ddec_as_array[wheredata2fit] / self.wavelengths[wheredata2fit] * wv_sampling[wv_id], \
                self.data[wheredata2fit], self.noise[wheredata2fit], self.bad_pixels[wheredata2fit], \
                self.wpsffit_IWA, self.wpsffit_OWA, fit_cen, fit_angle, init_paras
            out, _ = _fit_wpsf_task(paras)
            # print(out)
            # exit()
            # out = [0.00014386691687160602, -0.13301339368089535 ,-0.08327523187469463 ,0.001383654850629076]
            self.wpsf_ra_offset, self.wpsf_dec_offset, self.wpsf_angle_offset = out[0,2::]
            init_paras = [self.wpsf_ra_offset, self.wpsf_dec_offset]

            indices_within_threshold = find_bleeding_bar(self.dra_as_array - self.wpsf_ra_offset,
                                                         self.ddec_as_array - self.wpsf_dec_offset,
                                                         threshold2mask=0.15)
            self.bar_mask = np.ones(self.bad_pixels.shape)
            self.bar_mask[indices_within_threshold] = np.nan
            if save_utils:
                hdulist = pyfits.HDUList()
                wpsfsfit_header = {"INIT_ANG": self.wpsf_angle_offset,
                                   "INIT_RA": self.wpsf_ra_offset, "INIT_DEC": self.wpsf_dec_offset}
                hdulist = pyfits.HDUList()
                hdulist.append(pyfits.PrimaryHDU(data=self.bar_mask, header=pyfits.Header(cards=wpsfsfit_header)))
                try:
                    hdulist.writeto(self.barmask_filename, overwrite=True)
                except TypeError:
                    hdulist.writeto(self.barmask_filename, clobber=True)
                hdulist.close()
        else:
            self.bar_mask = np.ones(self.bad_pixels.shape)
        # exit()


        self.starspec_contnorm_filename = os.path.join(utils_dir, os.path.basename(filename).replace(".fits",
                                                                                                     "_starspec_contnorm.fits"))
        print(len(glob(self.starspec_contnorm_filename)), self.starspec_contnorm_filename)
        if 0 and self.compute_starspec_contnorm and load_utils and len(glob(self.starspec_contnorm_filename)):
            with pyfits.open(self.starspec_contnorm_filename) as hdulist:
                new_wavelengths = hdulist[0].data
                combined_fluxes = hdulist[1].data
                combined_errors = hdulist[2].data
                spline_paras0 = hdulist[4].data

            # plt.plot(new_wavelengths,combined_fluxes)
            # plt.show()
        elif self.compute_starspec_contnorm:

            reg_mean_map0 = np.zeros((self.data.shape[0], self.N_nodes))
            reg_std_map0 = np.zeros((self.data.shape[0], self.N_nodes))
            for rowid, row in enumerate(self.data):
                row_wvs = self.wavelengths[rowid, :]
                row_bp = self.bad_pixels[rowid, :]
                if np.nansum(np.isfinite(row * row_bp)) == 0:
                    continue
                reg_mean_map0[rowid, :] = np.nanmedian(row * row_bp)
                reg_std_map0[rowid, :] = reg_mean_map0[rowid, :]

            # print(im.shape, im_wvs.shape,err.shape, self.bad_pixels.shape)
            spline_cont0, _, new_badpixs, new_res,spline_paras0 = normalize_rows(im, im_wvs, noise=err, badpixs=self.bad_pixels*self.bar_mask,
                                                                   nodes=self.N_nodes, mypool=mppool,threshold=threshold_badpix,
                                                                   use_set_nans=False,
                                                                     regularization=True,reg_mean_map=reg_mean_map0,reg_std_map=reg_std_map0)
            spline_cont0, _, new_badpixs, new_res,spline_paras0 = normalize_rows(im, im_wvs, noise=err, badpixs=new_badpixs,
                                                                   nodes=self.N_nodes, mypool=mppool,threshold=threshold_badpix,
                                                                   use_set_nans=False,
                                                                     regularization=True,reg_mean_map=spline_paras0,reg_std_map=spline_paras0)

            spline_cont0[np.where(spline_cont0 / err < 5)] = np.nan
            spline_cont0 = copy(spline_cont0)
            spline_cont0[np.where(spline_cont0 < np.median(spline_cont0))] = np.nan
            spline_cont0[np.where(np.isnan(self.bad_pixels*self.bar_mask))] = np.nan
            normalized_im = im / spline_cont0
            normalized_err = err / spline_cont0

            # plt.imshow(normalized_im,origin="lower")
            # plt.clim([0.999,1.001])
            # plt.show()

            # comb_spec = np.nansum(normalized_im/normalized_err**2,axis=(1,2))/np.nansum(1/normalized_err**2,axis=(1,2))
            # comb_err = 1/np.sqrt(np.nansum(1/normalized_err**2,axis=(1,2)))
            new_wavelengths, combined_fluxes, combined_errors = combine_spectrum(im_wvs.flatten(),
                                                                                 normalized_im.flatten(),
                                                                                 normalized_err.flatten(),
                                                                                 np.nanmedian(im_wvs) / (
                                                                                     spec_R_sampling))
            # plt.plot(combined_fluxes,label="1")
            # plt.show()
            # plt.plot(new_wavelengths, combined_fluxes,label="1")
            #
            #
            # spline_cont,_,new_badpixs,new_res = normalize_rows(im, im_wvs,noise=err, badpixs=self.bad_pixels,nodes=self.N_nodes,mypool=mppool,use_set_nans=False)
            # spline_cont[np.where(spline_cont/err<5)] = np.nan
            # spline_cont[np.where(spline_cont<np.median(spline_cont))] = np.nan
            # normalized_im = im/spline_cont
            # normalized_err = err/spline_cont
            #
            # # comb_spec = np.nansum(normalized_im/normalized_err**2,axis=(1,2))/np.nansum(1/normalized_err**2,axis=(1,2))
            # # comb_err = 1/np.sqrt(np.nansum(1/normalized_err**2,axis=(1,2)))
            # new_wavelengths, combined_fluxes, combined_errors = combine_spectrum(im_wvs.flatten(), normalized_im.flatten(), normalized_err.flatten(), np.nanmedian(im_wvs)/(4*self.R))

            # plt.figure(1)
            # plt.scatter(im_wvs.flatten(), normalized_im.flatten(), s=0.5)
            # plt.plot(new_wavelengths, combined_fluxes,color="red",label="2")
            # plt.ylim([0,2])
            # plt.legend()
            # plt.figure(2)
            # plt.subplot(1,3,1)
            # plt.imshow(self.data,origin="lower")
            # plt.clim([0,1e-10])
            # plt.subplot(1,3,2)
            # plt.imshow(new_badpixs,origin="lower")
            # plt.subplot(1,3,3)
            # plt.imshow(spline_cont0,origin="lower")
            # plt.clim([0,1e-10])
            # plt.show()

            if save_utils:
                hdulist = pyfits.HDUList()
                hdulist.append(pyfits.PrimaryHDU(data=new_wavelengths))
                hdulist.append(pyfits.ImageHDU(data=combined_fluxes))
                hdulist.append(pyfits.ImageHDU(data=combined_errors))
                hdulist.append(pyfits.ImageHDU(data=spline_cont0))
                hdulist.append(pyfits.ImageHDU(data=spline_paras0))
                try:
                    hdulist.writeto(self.starspec_contnorm_filename, overwrite=True)
                except TypeError:
                    hdulist.writeto(self.starspec_contnorm_filename, clobber=True)
                hdulist.close()
        if self.compute_starspec_contnorm:
            self.star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False, fill_value=1)
        # plt.plot(new_wavelengths,combined_fluxes)
        # plt.plot(new_wavelengths,self.star_func(new_wavelengths),linestyle="--")
        # plt.show()

        print(len(glob(self.starsub_filename)), self.starsub_filename)
        if 1 and self.compute_starsub and load_utils and len(glob(self.starsub_filename)):
            with pyfits.open(self.starsub_filename) as hdulist:
                subtracted_im = hdulist[0].data
                star_model = hdulist[2].data
                fmderived_bad_pixels = hdulist[3].data

                # plt.figure(1)
                # plt.subplot(2,2,1)
                # plt.imshow(self.bad_pixels,interpolation="nearest",origin="lower")
                # plt.subplot(2,2,2)
                # plt.imshow(fmderived_bad_pixels,interpolation="nearest",origin="lower")
                # plt.subplot(2,2,3)
                # plt.imshow(subtracted_im,interpolation="nearest",origin="lower")
                # plt.clim([-1e-10,1e-10])
                # plt.subplot(2,2,4)
                # plt.imshow(self.star_func(im_wvs),interpolation="nearest",origin="lower")
                #
                # rowid = 255
                # plt.figure(2)
                # plt.subplot(2,1,1)
                # plt.plot(fmderived_bad_pixels[rowid,:])
                # plt.subplot(2,1,2)
                # plt.plot(self.data[rowid,:],label="data")
                # plt.plot(star_model[rowid,:]*self.bad_pixels[rowid,:]*fmderived_bad_pixels[rowid,:],label="star_model")
                # plt.plot(subtracted_im[rowid,:],label="subtracted_im")
                # plt.plot(self.noise[rowid,:],label="noise")
                # plt.plot(threshold_badpix*self.noise[rowid,:],label="5*noise")
                # plt.ylim([-1e-9,1e-8])
                # plt.legend()
                #
                # rowid = 1600
                # plt.figure(3)
                # plt.subplot(2,1,1)
                # plt.plot(fmderived_bad_pixels[rowid,:])
                # plt.subplot(2,1,2)
                # plt.plot(self.data[rowid,:],label="data")
                # plt.plot(star_model[rowid,:]*self.bad_pixels[rowid,:]*fmderived_bad_pixels[rowid,:],label="star_model")
                # plt.plot(subtracted_im[rowid,:],label="subtracted_im")
                # plt.plot(self.noise[rowid,:],label="noise")
                # plt.plot(threshold_badpix*self.noise[rowid,:],label="5*noise")
                # plt.ylim([-1e-9,1e-8])
                # plt.legend()
                # plt.show()

                self.bad_pixels = self.bad_pixels * fmderived_bad_pixels
        elif self.compute_starsub:
            # print(np.sum(np.isnan(self.bad_pixels)))
            star_model, _, new_badpixs, subtracted_im,spline_paras0 = normalize_rows(im, im_wvs, noise=err, badpixs=self.bad_pixels,
                                                                       nodes=self.N_nodes,
                                                                       star_model=self.star_func(im_wvs),
                                                                       threshold=threshold_badpix, use_set_nans=False,
                                                                       mypool=mppool,
                                                                     regularization=True,reg_mean_map=spline_paras0,reg_std_map=spline_paras0)
            # print(np.sum(np.isnan(new_badpixs)))
            # exit()
            self.bad_pixels = self.bad_pixels * new_badpixs
            star_model, _, new_badpixs, subtracted_im,_ = normalize_rows(im, im_wvs, noise=err, badpixs=self.bad_pixels,
                                                                       nodes=self.N_nodes,
                                                                       star_model=self.star_func(im_wvs),
                                                                       threshold=threshold_badpix, use_set_nans=False,
                                                                       mypool=mppool,
                                                                     regularization=True,reg_mean_map=spline_paras0,reg_std_map=spline_paras0)
            self.bad_pixels = self.bad_pixels * new_badpixs

            # print(np.sum(np.isnan(new_badpixs)))
            subtracted_im[np.where(np.isnan(subtracted_im))] = 0

            hdulist_sc["SCI"].data = subtracted_im
            if save_utils:
                hdulist = pyfits.HDUList()
                hdulist.append(pyfits.PrimaryHDU(data=subtracted_im))
                hdulist.append(pyfits.ImageHDU(data=im))
                hdulist.append(pyfits.ImageHDU(data=star_model))
                hdulist.append(pyfits.ImageHDU(data=self.bad_pixels))
                try:
                    hdulist.writeto(self.starsub_filename, overwrite=True)
                except TypeError:
                    hdulist.writeto(self.starsub_filename, clobber=True)
                hdulist.close()

                if not os.path.exists(os.path.join(utils_dir, "starsub")):
                    os.makedirs(os.path.join(utils_dir, "starsub"))
                try:
                    hdulist_sc.writeto(os.path.join(utils_dir, "starsub", os.path.basename(filename)), overwrite=True)
                except TypeError:
                    hdulist_sc.writeto(os.path.join(utils_dir, "starsub", os.path.basename(filename)), clobber=True)
                hdulist_sc.close()
        # exit()

        if apply_chargediff_mask:
            self.bad_pixels = self.bad_pixels * self.bar_mask

        # self.interpdata_regwvs_filename = os.path.join(savedir,
        #                                 os.path.basename(self.filename).replace(".fits", "_regwvs_modelfit.fits"))
        self.interpdata_regwvs_filename = os.path.join(utils_dir,
                                                       os.path.basename(self.filename).replace(".fits", "_regwvs.fits"))
        if self.compute_interp_regwvs:
            interp_ra, interp_dec, interp_wvs, interp_flux, interp_err, interp_badpix, interp_area2d = \
                self.interpdata_regwvs(wv_sampling=wv_sampling, modelfit=False,
                                       out_filename=self.interpdata_regwvs_filename,
                                       load_interpdata_regwvs=load_interpdata_regwvs)


        # self.webbpsf_filename = os.path.join(utils_dir,self.priheader["DATE-OBS"]+"_"+self.priheader["FILTER"]+"_"+self.priheader["GRATING"]+"_webbpsf.fits")
        splitbasename = os.path.basename(filename).split("_")
        self.webbpsffit_filename = os.path.join(utils_dir,
                                                os.path.basename(filename).replace(".fits", "_webbpsffit_freecen.fits"))
        print(len(glob(self.webbpsffit_filename)), self.webbpsffit_filename)
        if 1 and self.fit_wpsf and load_utils and len(glob(self.webbpsffit_filename)):
            with pyfits.open(self.webbpsffit_filename) as hdulist:
                bestfit_coords = hdulist[0].data
                self.wpsf_angle_offset = hdulist[0].header["INIT_ANG"]
                self.wpsf_ra_offset = hdulist[0].header["INIT_RA"]
                self.wpsf_dec_offset = hdulist[0].header["INIT_DEC"]
        elif self.fit_wpsf:
            if init_fit_psf:
                # rough centroid fit
                wv_id = 1000
                fit_cen, fit_angle = True, False
                # init_paras = np.array([self.wpsf_ra_offset, self.wpsf_dec_offset, self.wpsf_angle_offset])
                init_paras = np.array([self.wpsf_ra_offset, self.wpsf_dec_offset])
                # paras = linear_interp,wepsfs[wv_id,:,:], webbpsf_X, webbpsf_Y,self.east2V2_deg,\
                    # interp_ra[:,wv_id], interp_dec[:,wv_id], interp_flux[:,wv_id], interp_err[:,wv_id],interp_badpix[:,wv_id],\
                    # IWA,OWA,fit_cen,fit_angle,init_paras
                wheredata2fit = np.where(
                    (self.wavelengths > (wv_sampling[0] + 0.2)) * (self.wavelengths < (wv_sampling[-1] - 0.2)))
                paras = linear_interp, wepsfs[wv_id, :, :], webbpsf_X, webbpsf_Y, self.east2V2_deg,True, \
                    self.dra_as_array[wheredata2fit] / self.wavelengths[wheredata2fit] * wv_sampling[wv_id], \
                    self.ddec_as_array[wheredata2fit] / self.wavelengths[wheredata2fit] * wv_sampling[wv_id], \
                    self.data[wheredata2fit], self.noise[wheredata2fit], self.bad_pixels[wheredata2fit], \
                    self.wpsffit_IWA, 10.0, fit_cen, fit_angle, init_paras
                out = _fit_wpsf_task(paras)
                print(out)
                # out = [0.00014386691687160602, -0.13301339368089535 ,-0.08327523187469463 ,0.001383654850629076]
                self.wpsf_ra_offset, self.wpsf_dec_offset, self.wpsf_angle_offset = out[0,2::]

            # wv_id=1000
            # plt.scatter(interp_ra[:,wv_id],interp_dec[:,wv_id],s=interp_flux[:,wv_id]/np.nanmean(interp_flux[:,wv_id]))
            # plt.show()

            #
            # cp_interp_badpix = copy(interp_badpix)
            # if mask_charge_bleeding:
            #     indices_within_threshold = find_bleeding_bar(interp_ra-self.wpsf_ra_offset,interp_dec-self.wpsf_dec_offset,threshold2mask=0.15)
            #     cp_interp_badpix[indices_within_threshold] = np.nan

            init_paras = [self.wpsf_ra_offset, self.wpsf_dec_offset]
            # print(init_paras)
            fit_cen, fit_angle = True, False
            bestfit_coords = np.zeros((np.size(wv_sampling), 5)) + np.nan  # flux_init, flux,ra,dec,angle
            if 0:
                for wv_id, wv in enumerate(wv_sampling):
                    if wv_id < 1146:  # or wv_id>300
                        continue
                    # if wv_id % 50 != 0:
                    #     continue
                    print(wv_id, wv, np.size(wv_sampling))
                    paras = linear_interp, wepsfs[wv_id, :,:], webbpsf_X, webbpsf_Y, self.east2V2_deg - self.wpsf_angle_offset,True, \
                        interp_ra[:, wv_id], interp_dec[:, wv_id], interp_flux[:, wv_id], interp_err[:,wv_id], interp_badpix[:,wv_id], \
                        self.wpsffit_IWA, self.wpsffit_OWA, fit_cen, fit_angle, init_paras
                    out, _ = _fit_wpsf_task(paras)
                    # plt.imshow(out[0])
                    # plt.show()
                    bestfit_coords[wv_id, :] = out[0]
                    exit()
            else:
                # paras = wepsfs[wv_id,:,:], webbpsf_X, webbpsf_Y,self.east2V2_deg - self.wpsf_angle_offset,\
                #     interp_ra[:,wv_id], interp_dec[:,wv_id], interp_flux[:,wv_id], interp_err[:,wv_id],interp_badpix[:,wv_id],\
                #     self.wpsffit_IWA,self.wpsffit_OWA,fit_cen,fit_angle,init_paras
                output_lists = mppool.map(_fit_wpsf_task, zip(itertools.repeat(linear_interp),
                                                              wepsfs, itertools.repeat(webbpsf_X),
                                                              itertools.repeat(webbpsf_Y),
                                                              itertools.repeat(self.east2V2_deg - self.wpsf_angle_offset),
                                                              itertools.repeat(True),
                                                              interp_ra.T,
                                                              interp_dec.T,
                                                              interp_flux.T,
                                                              interp_err.T,
                                                              interp_badpix.T,
                                                              itertools.repeat(self.wpsffit_IWA),
                                                              itertools.repeat(self.wpsffit_OWA),
                                                              itertools.repeat(fit_cen),
                                                              itertools.repeat(fit_angle),
                                                              itertools.repeat(init_paras)))

                for wv_id, (wv, out) in enumerate(zip(wv_sampling, output_lists)):
                    print(wv_id, np.size(wv_sampling))
                    bestfit_coords[wv_id, :] = out[0]

            # plt.imshow(wpsfs[0])
            # plt.show()
            if save_utils:
                wpsfsfit_header = {"INIT_ANG": self.wpsf_angle_offset,
                                   "INIT_RA": self.wpsf_ra_offset, "INIT_DEC": self.wpsf_dec_offset}
                hdulist = pyfits.HDUList()
                hdulist.append(pyfits.PrimaryHDU(data=bestfit_coords, header=pyfits.Header(cards=wpsfsfit_header)))
                try:
                    hdulist.writeto(self.webbpsffit_filename, overwrite=True)
                except TypeError:
                    hdulist.writeto(self.webbpsffit_filename, clobber=True)
                hdulist.close()

        # plt.subplot(2, 2, 1)
        # # plt.plot(np.arange(np.size(wv_sampling)), bestfit_coords[:, 0],color="blue")
        # # plt.plot(np.arange(np.size(wv_sampling)), bestfit_coords[:, 1],color="red")
        # plt.plot(wv_sampling, bestfit_coords[:, 0], color="blue")
        # plt.plot(wv_sampling, bestfit_coords[:, 1], color="red")
        # plt.ylabel("flux")
        # plt.subplot(2, 2, 2)
        # # plt.plot(np.arange(np.size(wv_sampling)), np.zeros(np.size(wv_sampling))+self.wpsf_angle_offset,color="blue")
        # # plt.plot(np.arange(np.size(wv_sampling)), bestfit_coords[:, 4],color="red")
        # plt.plot(wv_sampling, np.zeros(np.size(wv_sampling)) + self.wpsf_angle_offset, color="blue")
        # plt.plot(wv_sampling, bestfit_coords[:, 4], color="red")
        # # self.wpsf_ra_offset, self.wpsf_dec_offset,
        # plt.ylabel("angle")
        # plt.subplot(2, 2, 3)
        # # plt.plot(np.arange(np.size(wv_sampling)), np.zeros(np.size(wv_sampling))+self.wpsf_ra_offset,color="blue")
        # # plt.plot(np.arange(np.size(wv_sampling)), bestfit_coords[:, 2],color="red")
        # plt.plot(wv_sampling, np.zeros(np.size(wv_sampling)) + self.wpsf_ra_offset, color="blue")
        # plt.plot(wv_sampling, bestfit_coords[:, 2], color="red")
        # plt.ylabel("x")
        # plt.subplot(2, 2, 4)
        # # plt.plot(np.arange(np.size(wv_sampling)), np.zeros(np.size(wv_sampling))+self.wpsf_dec_offset,color="blue")
        # # plt.plot(np.arange(np.size(wv_sampling)), bestfit_coords[:, 3],color="red")
        # plt.plot(wv_sampling, np.zeros(np.size(wv_sampling)) + self.wpsf_dec_offset, color="blue")
        # plt.plot(wv_sampling, bestfit_coords[:, 3], color="red")
        # plt.ylabel("y")
        # plt.show()
        # exit()

        # if apply_coords_corr:
        #     self.east2V2_deg = self.east2V2_deg - self.wpsf_angle_offset
        #     self.dra_as_array = self.dra_as_array - self.wpsf_ra_offset
        #     self.ddec_as_array = self.ddec_as_array - self.wpsf_dec_offset

        # interp_webbpsf = RegularGridInterpolator((webbpsf_wvs,),wepsfs/peak_webb_epsf[:,None,None],method="linear",bounds_error=False,fill_value=0.0)

        # photref_filename = "/scr3/jruffio/data/JWST/nirspec/HD_19467/breads/utils/photref_combine_395h_calib_g395h-f290lp_s3d.fits"
        # with pyfits.open(photref_filename) as hdulist:
        #     photref_cube = hdulist["SCI"].data
        #     photref_pixscale = hdulist["SCI"].header["CDELT1"]*3600
        #     # print(photref_pixscale)
        #     # exit()
        # from spectral_cube import SpectralCube
        # tmpcube=SpectralCube.read(photref_filename,hdu=1)
        # photref_wvs=np.array(tmpcube.spectral_axis)
        #
        # photref_cube[np.where(np.abs(photref_cube)>1e-5)] = np.nan
        # cube_sum_im = np.nansum(photref_cube,axis=0)
        # kc,lc = np.unravel_index(np.nanargmax(cube_sum_im),cube_sum_im.shape)
        # ks = np.arange(photref_cube.shape[1])-kc
        # ls = np.arange(photref_cube.shape[2])-lc
        # fluxref_X, fluxref_Y = np.meshgrid(ls,ks)
        # fluxref_R = np.sqrt(fluxref_X**2+fluxref_Y**2)
        # mask= np.ones(fluxref_R.shape)
        # mask[np.where(fluxref_R>10)] = np.nan
        # mask_tiled = np.tile(mask[None,:,:],(photref_cube.shape[0],1,1))
        # # plt.imshow(cube_sum_im)
        # # plt.plot(np.nanmax(cube*mask_tiled,axis=(1,2)))
        # # plt.plot(np.nanmax(photref_cube*mask_tiled,axis=(1,2))/np.nansum(photref_cube*mask_tiled,axis=(1,2)))
        # plt.plot(photref_wvs,(photref_cube*mask_tiled)[:,kc,lc]/np.nansum(photref_cube*mask_tiled,axis=(1,2)))
        # plt.plot(photref_wvs,self.aper_to_epsf_peak_f(photref_wvs))
        # plt.show()

        # p = photref_cube[np.nanargmin(np.abs(photref_wvs-4.5)),kc,:]
        # plt.plot((np.arange(np.size(p))-np.nanargmax(p))*photref_pixscale,p/np.nanmax(p),label="photref")
        # print(webb_epsf_ims.shape)
        # p = interp_webbpsf([4.5,])[0][webb_epsf_ims.shape[1]//2,:]
        # plt.plot((np.arange(np.size(p))-np.nanargmax(p))*webbpsf_pixelscale,p/np.nanmax(p),label="webbpsf")
        # plt.legend()
        # plt.show()
        # print()

        # ra_offset = -1.38 # ra offset in as
        # dec_offset = -0.92 # dec offset in as
        # dist2host_as = np.sqrt((self.dra_as_array-ra_offset) ** 2 + (self.ddec_as_array-dec_offset) ** 2)
        # mask_comp  = dist2host_as<0.2

        # plt.figure(1)
        # plt.imshow(subtracted_im,origin="lower")
        # plt.clim([-1e-11,1e-11])
        # contour = plt.gca().contour(mask_comp.astype(int), colors='cyan', levels=[0.5])
        # plt.show()

        # print(np.nanmedian(bestfit_paras[3,:]))
        # exit()

        hdulist_sc.close()
        self.valid_data_check()

    def getifucoords(self, ras=None, decs=None):
        # plt.scatter(self.dra_as_array[:,500],self.ddec_as_array[:,500],c="red",s=100*self.data[:,500]/np.nanmax(self.data[:,500]))
        if ras is not None and decs is not None:
            ifuX, ifuY = rotate_coordinates(ras, decs, self.east2V2_deg, flipx=False)
        else:
            ifuX, ifuY = rotate_coordinates(self.dra_as_array, self.ddec_as_array, self.east2V2_deg, flipx=False)
        # plt.scatter(ifuX[:,500],ifuY[:,500],c="blue",s=100*self.data[:,500]/np.nanmax(self.data[:,500]))
        # plt.show()
        # exit()
        return ifuX, ifuY

    def broaden(self, wvs, spectrum, loc=None, mppool=None):
        """
        Broaden a spectrum to the resolution of this data object using the resolution attribute (self.R).
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
        wv_min, wv_max = np.nanmin(self.wavelengths), np.nanmax(self.wavelengths)
        sampling_dw = np.nanmedian(self.wavelengths[:, 1::] - self.wavelengths[:, 0:self.wavelengths.shape[1] - 1])
        wv_sampling = np.arange(wv_min, wv_max, sampling_dw)
        return wv_sampling

    def interpdata_regwvs(self, wv_sampling=None, modelfit=True, out_filename=None, load_interpdata_regwvs=True):

        print(len(glob(out_filename)), out_filename)
        if 1 and load_interpdata_regwvs and len(glob(out_filename)):
            with pyfits.open(out_filename) as hdulist:
                interp_flux = hdulist[0].data
                interp_err = hdulist[1].data
                interp_ra = hdulist[2].data
                interp_dec = hdulist[3].data
                interp_wvs = hdulist[4].data
                interp_badpix = hdulist[5].data
                interp_area2d = hdulist[6].data
        else:
            if wv_sampling is None:
                wv_sampling = self.get_regwvs_sampling()
            Nwv = np.size(wv_sampling)

            if modelfit:
                with pyfits.open(self.starsub_filename) as hdulist:
                    data2interp = hdulist[2].data
            else:
                data2interp = self.data

            interp_ra = np.zeros((self.data.shape[0], Nwv))
            interp_dec = np.zeros((self.data.shape[0], Nwv))
            interp_wvs = np.zeros((self.data.shape[0], Nwv))
            interp_flux = np.zeros((self.data.shape[0], Nwv))
            interp_err = np.zeros((self.data.shape[0], Nwv))
            interp_badpix = np.zeros((self.data.shape[0], Nwv))
            interp_area2d = np.zeros((self.data.shape[0], Nwv))

            for wv_id, center_wv in enumerate(wv_sampling):
                # if wv_id != 200:
                #     continue
                # if left_wv<4.41:
                #     continue
                # if left_wv<4.4 or left_wv>4.6:
                #     continue
                # if center_wv<3.4 or center_wv>3.45:
                #     continue
                # print(wv_id, Nwv, center_wv)
                for rowid in range(self.data.shape[0]):
                    where_finite = np.where(np.isfinite(self.bad_pixels[rowid, :]))
                    if np.size(where_finite[0]) == 0:
                        # print("No ref points")
                        continue

                    w = self.wavelengths[rowid, :]
                    interp_ra[rowid, wv_id] = np.interp(center_wv, w, self.dra_as_array[rowid, :], left=np.nan,
                                                        right=np.nan)
                    interp_dec[rowid, wv_id] = np.interp(center_wv, w, self.ddec_as_array[rowid, :], left=np.nan,
                                                         right=np.nan)
                    interp_wvs[rowid, wv_id] = center_wv
                    interp_flux[rowid, wv_id] = np.interp(center_wv, w, data2interp[rowid, :], left=np.nan,
                                                          right=np.nan)
                    interp_err[rowid, wv_id] = np.interp(center_wv, w, self.noise[rowid, :], left=np.nan, right=np.nan)
                    badpix_mask = np.isfinite(self.bad_pixels[rowid, :]).astype(float)
                    interp_badpix[rowid, wv_id] = np.interp(center_wv, w, badpix_mask, left=np.nan, right=np.nan)
                    interp_area2d[rowid, wv_id] = np.interp(center_wv, w, self.area2d[rowid, :], left=np.nan,
                                                            right=np.nan)

            where_bad = np.where(interp_badpix != 1.0)
            interp_ra[where_bad] = np.nan
            interp_dec[where_bad] = np.nan
            interp_wvs[where_bad] = np.nan
            interp_flux[where_bad] = np.nan
            interp_err[where_bad] = np.nan
            interp_badpix[where_bad] = np.nan
            interp_area2d[where_bad] = np.nan

            if out_filename is not None:
                hdulist = pyfits.HDUList()
                hdulist.append(pyfits.PrimaryHDU(data=interp_flux))
                hdulist.append(pyfits.ImageHDU(data=interp_err))
                hdulist.append(pyfits.ImageHDU(data=interp_ra))
                hdulist.append(pyfits.ImageHDU(data=interp_dec))
                hdulist.append(pyfits.ImageHDU(data=interp_wvs))
                hdulist.append(pyfits.ImageHDU(data=interp_badpix))
                hdulist.append(pyfits.ImageHDU(data=interp_area2d))
                try:
                    hdulist.writeto(out_filename, overwrite=True)
                except TypeError:
                    hdulist.writeto(out_filename, clobber=True)
                hdulist.close()

        return interp_ra, interp_dec, interp_wvs, interp_flux, interp_err, interp_badpix, interp_area2d


def untangle_dq(arr):
    # # Assume arr is your input numpy array of shape (ny, nx)
    # ny, nx = arr.shape
    #
    # # Create a new numpy array of shape (32, ny, nx) to hold the cube
    # cube = np.zeros((32, ny, nx), dtype=bool)
    #
    # # Loop over each element of the input array
    # for i in range(ny):
    #     for j in range(nx):
    #         # Get the binary representation of the integer at this location
    #         binary_str = bin(arr[i,j])[2:].zfill(32)
    #
    #         # Loop over each bit in the binary string and set the corresponding element of the cube
    #         for k in range(32):
    #             cube[k,i,j] = bool(int(binary_str[k]))
    print(arr.dtype)
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


def set_nans(arr, n):
    # remove the edges of the spectrum
    # for i in range(arr.shape[0]):
    #     row = arr[i, :]
    #     first_real_idx = np.argmax(~np.isnan(row))
    #     last_real_idx = np.argmax(~np.isnan(row[::-1]))
    #     last_real_idx = row.shape[0] - last_real_idx - 1
    #     arr[i, :first_real_idx + n] = np.nan
    #     arr[i, last_real_idx - n + 1:] = np.nan
    # return arr
    # Create a copy of the input array
    arr_copy = np.copy(arr)

    # Find the indices of the first and last non-nan values
    mask = ~np.isnan(arr_copy)
    first_real_idx = np.argmax(mask, axis=1)
    last_real_idx = arr_copy.shape[1] - np.argmax(mask[:, ::-1], axis=1) - 1

    # Set the first n and last n non-nan values to nan
    for i in range(arr_copy.shape[0]):
        arr_copy[i, :first_real_idx[i] + n] = np.nan
        arr_copy[i, last_real_idx[i] - n + 1:] = np.nan

    return arr_copy


def _task_normrows(paras):
    im_rows, im_wvs_rows, noise_rows, badpix_rows, x_knots, star_model, threshold, star_sub_mode,regularization,reg_mean_map,reg_std_map = paras

    new_im_rows = np.array(copy(im_rows), '<f4')  # .byteswap().newbyteorder()
    new_noise_rows = copy(noise_rows)
    new_badpix_rows = copy(badpix_rows)
    res = np.zeros(im_rows.shape) + np.nan
    paras_out = np.zeros((im_rows.shape[0],np.size(x_knots))) + np.nan
    for k in range(im_rows.shape[0]):
        # print(k)
        # if k == 255:
        #     print("coucou")
        # else:
        #     continue
        M_spline = get_spline_model(x_knots, im_wvs_rows[k, :], spline_degree=3)

        # # plt.subplot(2,1,1)
        # plt.plot(im_rows[k,:])
        # plt.plot(noise_rows[k,:])
        # # plt.subplot(2,1,2)
        # # plt.plot(np.isfinite(med_spec),label="np.isfinite(med_spec)")
        # # plt.plot(noise_rows[k,:]==0,label="noise_rows[k,:]==0")
        # # plt.plot(np.isfinite(im_rows[k,:]),label="np.isfinite(im_rows[k,:])")
        # # plt.plot(np.isfinite(noise_rows[k,:]),label="np.isfinite(noise_rows[k,:])")
        # plt.legend()
        # plt.show()

        where_data_finite = np.where(
            np.isfinite(badpix_rows[k, :]) * np.isfinite(im_rows[k, :]) * np.isfinite(noise_rows[k, :]) * (
                        noise_rows[k, :] != 0) * np.isfinite(star_model[k, :]))
        # if k == 1512:
        #     print("coucou")
        # else:
        #     res[:,k] = np.nan
        #     continue
        # print(k,np.size(where_data_finite[0]))
        if np.size(where_data_finite[0]) == 0:
            res[k, :] = np.nan
            continue

        d = im_rows[k, where_data_finite[0]]
        d_err = noise_rows[k, where_data_finite[0]]

        M = M_spline[where_data_finite[0], :] * star_model[k, where_data_finite[0], None]

        if regularization:
            validpara = np.where(np.nansum(M > np.nanmax(M) * 0.00001, axis=0) != 0)
        else:
            validpara = np.where(np.nansum(M > np.nanmax(M) * 0.01, axis=0) != 0)
        M = M[:, validpara[0]]
        # plt.subplot(2,1,1)
        # plt.plot(im_rows[k,:],label="data")
        # plt.plot(noise_rows[k,:],label="noise")
        # plt.ylim([np.nanmin(d),np.nanmax(d)])
        # plt.legend()
        # plt.subplot(2,1,2)
        # print(M.shape)
        # for k in range(M.shape[-1]-1):
        #     plt.plot(M[:,k+1])
        # plt.show()

        #regularization,reg_mean_map,reg_std_map
        if regularization:
            d_reg, s_reg = reg_mean_map[k,:],reg_std_map[k,:]
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
            d4fit,M4fit,s4fit = d,M,d_err

        # bounds_min = [0, ]* M.shape[1]
        bounds_min = [-np.inf, ] * M.shape[1]
        bounds_max = [np.inf, ] * M.shape[1]
        p = lsq_linear(M4fit / s4fit[:, None], d4fit / s4fit, bounds=(bounds_min, bounds_max)).x
        paras_out[k,validpara[0]] = p
        # p,chi2,rank,s = np.linalg.lstsq(M/d_err[:,None],d/d_err,rcond=None)
        m = np.dot(M, p)
        res[k, where_data_finite[0]] = d - m
        new_im_rows[k, where_data_finite[0]] = m
        new_noise_rows[k, where_data_finite[0]] = d_err
        norm_res_row = np.zeros(im_rows.shape[1]) + np.nan
        norm_res_row[where_data_finite] = (d - m) / d_err

        # where_bad = np.where((np.abs(res[:,k])>3*np.nanstd(res[:,k])) | np.isnan(res[:,k]))
        # meddev=median_abs_deviation(res[k,where_data_finite[0]])
        # where_bad = np.where((np.abs(res[k,:])>threshold*meddev) | np.isnan(res[k,:]))
        meddev = median_abs_deviation(norm_res_row[where_data_finite])
        where_bad = np.where((np.abs(norm_res_row) / meddev > threshold) | np.isnan(norm_res_row))
        new_badpix_rows[k, where_bad[0]] = np.nan
        # where_bad = np.where(np.isnan(np.correlate(new_badpix_rows[k,:] ,np.ones(2),mode="same")))
        # new_badpix_rows[k,where_bad[0]] = np.nan
        # new_badpix_rows[k,np.where(np.isnan(new_badpix_rows[k,:]))[0]] = 1

        if 0:
            # plt.figure(2)
            # plt.plot(new_im_rows[k,:],label="d")
            # # plt.plot(new_noise_rows[k,:],label="n")
            # plt.legend()

            plt.figure(1)
            plt.subplot(3, 1, 1)
            plt.plot(d, label="d")
            # m0 = med_spec[where_data_finite[0],None]
            # plt.plot(m0/np.nansum(m0)*np.nansum(d),label="m0")
            plt.plot(m, label="m")
            plt.plot(d_err, label="err")
            plt.plot(d - m, label="res")
            plt.plot(d / d * threshold * meddev, label="threshold")
            plt.legend()

            plt.subplot(3, 1, 2)
            for l in range(M.shape[1]):
                plt.plot(M[:, l])
            plt.subplot(3, 1, 3)
            ratio = im_rows[k, :] / new_im_rows[k, :]
            ratio[np.where(new_im_rows[k, :] / noise_rows[k, :] < 10)] = np.nan
            plt.plot(ratio)
            plt.show()

            # plt.plot(new_data_arr[where_data_finite[0],k],label="new d",linestyle="--")
            # plt.legend()
            # plt.figure(2)
            # plt.plot(new_badpix_arr[where_data_finite[0],k],label="bad pix",linestyle="-")
            # plt.show()

    return new_im_rows, new_noise_rows, new_badpix_rows, res,paras_out


def normalize_rows(image, im_wvs, noise=None, badpixs=None, star_model=None, nodes=40, mypool=None, nan_mask_boxsize=3,
                   threshold=10, star_sub_mode=False, use_set_nans=True,x_knots=None,regularization=False,reg_mean_map=None,reg_std_map=None):

    if noise is None:
        noise = np.ones(image.shape)
    if badpixs is None:
        badpixs = np.ones(image.shape)
    if star_model is None:
        star_model = np.ones(image.shape)

    if x_knots is None:
        x_knots = np.linspace(np.nanmin(im_wvs), np.nanmax(im_wvs), nodes, endpoint=True)

    new_image = copy(image)
    if use_set_nans:
        new_image = set_nans(image, 40)
    new_noise = copy(noise)
    new_badpixs = copy(badpixs)
    new_res = np.zeros(image.shape) + np.nan
    new_spline_paras = np.zeros((image.shape[0],np.size(x_knots)))

    if mypool is None:
        paras = new_image, im_wvs, new_noise, new_badpixs, x_knots, star_model, threshold, star_sub_mode,regularization,reg_mean_map,reg_std_map
        outputs = _task_normrows(paras)
        new_image, new_noise, new_badpixs, new_res,new_spline_paras = outputs
    else:
        numthreads = mypool._processes
        chunk_size = image.shape[0] // (3 * numthreads)
        N_chunks = image.shape[0] // chunk_size
        row_ids = np.arange(image.shape[0])

        row_indices_list = []
        image_list = []
        wvs_list = []
        noise_list = []
        badpixs_list = []
        starmodel_list = []
        if regularization:
            reg_mean_map_list, reg_std_map_list = [],[]
        for k in range(N_chunks - 1):
            _row_valid_pix = row_ids[(k * chunk_size):((k + 1) * chunk_size)]
            row_indices_list.append(_row_valid_pix)

            _new_image = new_image[(k * chunk_size):((k + 1) * chunk_size), :]
            _im_wvs = im_wvs[(k * chunk_size):((k + 1) * chunk_size), :]
            _new_noise = new_noise[(k * chunk_size):((k + 1) * chunk_size), :]
            _new_badpixs = new_badpixs[(k * chunk_size):((k + 1) * chunk_size), :]
            _star_model = star_model[(k * chunk_size):((k + 1) * chunk_size), :]
            # regularization=None,reg_mean_map=None,reg_std_map=None
            if regularization:
                reg_mn_chunk= reg_mean_map[(k * chunk_size):((k + 1) * chunk_size), :]
                reg_std_chunk = reg_std_map[(k * chunk_size):((k + 1) * chunk_size), :]

            image_list.append(_new_image)
            wvs_list.append(_im_wvs)
            noise_list.append(_new_noise)
            badpixs_list.append(_new_badpixs)
            starmodel_list.append(_star_model)
            if regularization:
                reg_mean_map_list.append(reg_mn_chunk)
                reg_std_map_list.append(reg_std_chunk)

        _row_valid_pix = row_ids[((N_chunks - 1) * chunk_size):image.shape[0]]
        row_indices_list.append(_row_valid_pix)

        _new_image = new_image[((N_chunks - 1) * chunk_size):image.shape[0], :]
        _im_wvs = im_wvs[((N_chunks - 1) * chunk_size):image.shape[0], :]
        _new_noise = new_noise[((N_chunks - 1) * chunk_size):image.shape[0], :]
        _new_badpixs = new_badpixs[((N_chunks - 1) * chunk_size):image.shape[0], :]
        _star_model = star_model[((N_chunks - 1) * chunk_size):image.shape[0], :]
        if regularization:
            reg_mn_chunk = reg_mean_map[((N_chunks - 1) * chunk_size):image.shape[0], :]
            reg_std_chunk = reg_std_map[((N_chunks - 1) * chunk_size):image.shape[0], :]

        image_list.append(_new_image)
        wvs_list.append(_im_wvs)
        noise_list.append(_new_noise)
        badpixs_list.append(_new_badpixs)
        starmodel_list.append(_star_model)
        if regularization:
            reg_mean_map_list.append(reg_mn_chunk)
            reg_std_map_list.append(reg_std_chunk)

        # paras = new_image,im_wvs,new_noise,new_badpixs,x_knots,med_spec,chunks,threshold
        if not regularization:
            outputs_list = mypool.map(_task_normrows, zip(image_list, wvs_list, noise_list, badpixs_list,
                                                          itertools.repeat(x_knots),
                                                          starmodel_list,
                                                          itertools.repeat(threshold),
                                                          itertools.repeat(star_sub_mode),
                                                          itertools.repeat(False),
                                                          itertools.repeat(None),
                                                          itertools.repeat(None)))
        else:
            outputs_list = mypool.map(_task_normrows, zip(image_list, wvs_list, noise_list, badpixs_list,
                                                          itertools.repeat(x_knots),
                                                          starmodel_list,
                                                          itertools.repeat(threshold),
                                                          itertools.repeat(star_sub_mode),
                                                          itertools.repeat(regularization),
                                                          reg_mean_map_list,reg_std_map_list))
        for row_indices, outputs in zip(row_indices_list, outputs_list):
            out_im_rows, out_noise_rows, out_badpixs_rows, out_res,spline_paras = outputs
            new_image[row_indices, :] = out_im_rows
            new_noise[row_indices, :] = out_noise_rows
            new_badpixs[row_indices, :] = out_badpixs_rows
            new_res[row_indices, :] = out_res
            new_spline_paras[row_indices, :] = spline_paras

    return new_image, new_noise, new_badpixs, new_res,new_spline_paras
    #
    # return bestfit_paras, psfsub_model_im, psfsub_sc_im


def fit_webbpsf(sc_im, sc_im_wvs, noise, bad_pixels, dra_as_array, ddec_as_array, interpolator, psf_wv0, fix_cen=None):
    wv_min, wv_max = np.nanmin(sc_im_wvs), np.nanmax(sc_im_wvs)
    wv_sampling = np.exp(np.arange(np.log(wv_min), np.log(wv_max), np.log(1 + 0.5 / 2700.)))

    dist2host_as = np.sqrt(dra_as_array ** 2 + ddec_as_array ** 2)

    # Zinterp = interpolator(X.ravel()*wv_min/wv_max, Y.ravel()*wv_min/wv_max)
    # # Zinterp = griddata((X.ravel()/wv_min, Y.ravel()/wv_min), im_min.ravel(), (X.ravel()/wv_max, Y.ravel()/wv_max), method='cubic')
    # im_max_interp = Zinterp.reshape(psf_array_shape)
    # im_max_interp = im_max_interp/np.nansum(im_max_interp)*np.nansum(im_max)
    #
    #
    # interpolator = RegularGridInterpolator((x, y), im_min)
    # Zinterp = interpolator(((X.ravel()*wv_min/wv_max, Y.ravel()*wv_min/wv_max)))
    # im_max_interp2 = Zinterp.reshape(psf_array_shape)
    # im_max_interp2 = im_max_interp2/np.nansum(im_max_interp2)*np.nansum(im_max)
    #
    # print(X.shape)
    # print(Y.shape)
    # extent = [-halffov_x, halffov_x, -halffov_y, halffov_y]
    # plt.subplot(1,3,1)
    # plt.imshow(im_max,
    #     extent=extent,
    #     cmap="viridis",
    #     interpolation="nearest",
    #     origin='lower'
    # )
    # plt.subplot(1,3,2)
    # plt.imshow((im_max-im_max_interp)/np.nanmax(im_max),
    #            extent=extent,
    #            cmap="viridis",
    #            interpolation="nearest",
    #            origin='lower'
    #            )
    # plt.colorbar()
    # plt.subplot(1,3,3)
    # plt.imshow((im_max-im_max_interp2)/np.nanmax(im_max),
    #            extent=extent,
    #            cmap="viridis",
    #            interpolation="nearest",
    #            origin='lower'
    #            )
    # plt.colorbar()
    # plt.show()
    #
    # print(psf_array_shape)
    # print(pixelscale)
    # exit()

    psfsub_sc_im = np.zeros(sc_im.shape) + np.nan
    psfsub_model_im = np.zeros(sc_im.shape)
    bestfit_paras = np.zeros((4, np.size(wv_sampling))) + np.nan
    for wv_id, left_wv in enumerate(wv_sampling):
        # if left_wv<4.5:
        #     continue
        center_wv = left_wv * (1 + 0.25 / 2700)
        right_wv = left_wv * (1 + 0.5 / 2700)
        print(left_wv, center_wv, right_wv, wv_min, wv_max)

        # sc_im, sc_im_wvs,noise, sc_dq
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

        print(np.size(where_fit_mask[0]), 377 / 4, np.size(where_sc_mask[0]), 736 / 2)
        # exit()
        if (np.size(where_fit_mask[0]) < 377 / 4) or (np.size(where_sc_mask[0]) < 736 / 2):
            print("Not enough points", wv_id, center_wv, np.size(where_fit_mask[0]), np.size(where_sc_mask[0]))
            bestfit_paras[:, wv_id] = np.array([center_wv, np.nan, np.nan, np.nan])
            psfsub_model_im[where_sc_mask] = np.nan
            psfsub_sc_im[where_sc_mask] = np.nan
            continue

        # plt.figure(1)
        # plt.scatter(Xfit,Zfit,label="Zsc")
        # plt.scatter(Xfit,a0*m0,label="psfmodel")
        # plt.legend()
        #
        # plt.figure(2)
        # plt.subplot(1,2,1)
        # plt.scatter(Xfit,Yfit,c=Zfit,s=Zfit/np.nanmedian(Zfit),label="Zsc")
        # plt.subplot(1,2,2)
        # plt.scatter(Xfit,Yfit,c=a0*m0,s=a0*m0/np.nanmedian(Zfit),label="psfmodel")
        # plt.legend()
        # plt.show()
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

        # plt.figure(1)
        # plt.subplot(1,2,1)
        # plt.scatter(Xsc,Zsc,label="Zsc")
        # plt.scatter(Xsc,psfmodel,label="psfmodel")
        # plt.scatter(Xsc,psfsub_Zsc,label="psfsub_Zsc")
        # plt.scatter(0,a * interpolator(0,0))
        # plt.legend()
        # plt.subplot(1,2,2)
        # plt.scatter(Xfit,Zfit,label="Zsc")
        # plt.scatter(Xfit,a * interpolator(Xfit*psf_wv0/center_wv - xc,Yfit*psf_wv0/center_wv - yc),label="psfmodel")
        # plt.scatter(Xfit,Zfit-a * interpolator(Xfit*psf_wv0/center_wv - xc,Yfit*psf_wv0/center_wv - yc),label="psfsub_Zsc")
        # plt.scatter(0,a * interpolator(0,0))
        # plt.legend()
        # plt.show()

        bestfit_paras[:, wv_id] = np.array(
            [center_wv, xc * center_wv / psf_wv0, yc * center_wv / psf_wv0, a * interpolator(0, 0)])
        psfsub_model_im[where_sc_mask] = psfmodel
        psfsub_sc_im[where_sc_mask] = psfsub_Zsc

    return bestfit_paras, psfsub_model_im, psfsub_sc_im


import numpy as np


def where_point_source(dataobj, radec_as, rad_as):
    ra, dec = radec_as
    dist2pointsource_as = np.sqrt((dataobj.dra_as_array - ra) ** 2 + (dataobj.ddec_as_array - dec) ** 2)
    return np.where(dist2pointsource_as < rad_as)


def PCA_detec(im, im_err, im_badpixs, N_KL=5):
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

    # print(X.shape)
    C = np.cov(X)
    # print(C.shape)
    # exit()
    tot_basis = C.shape[0]
    tmp_res_numbasis = np.clip(np.abs(N_KL) - 1, 0,
                               tot_basis - 1)  # clip values, for output consistency we'll keep duplicates
    max_basis = np.max(
        tmp_res_numbasis) + 1  # maximum number of eigenvectors/KL basis we actually need to use/calculate
    evals, evecs = la.eigh(C, eigvals=(tot_basis - max_basis, tot_basis - 1))
    check_nans = np.any(evals <= 0)  # alternatively, check_nans = evals[0] <= 0
    evals = np.copy(evals[::-1])
    evecs = np.copy(evecs[:, ::-1], order='F')  # fortran order to improve memory caching in matrix multiplication
    # calculate the KL basis vectors
    kl_basis = np.dot(X.T, evecs)
    kls = kl_basis * (1. / np.sqrt(evals * (nx - 1)))[None, :]  # multiply a value for each row
    print(kls.shape)

    return kls


def PCA_wvs_axis(wavelengths, im, im_err, im_badpixs, bin_size, N_KL=5):
    ny, nx = im.shape
    # print(im.shape)
    # exit()
    mask = im/im_err

    # print(np.where(mask*im_badpixs>10))
    #
    # plt.imshow(mask*im_badpixs,origin="lower")
    # plt.clim([-10,10])
    # plt.show()

    new_wvs = np.arange(np.nanmin(wavelengths*im_badpixs), np.nanmax(wavelengths*im_badpixs), bin_size)
    nz = np.size(new_wvs)
    new_im = np.zeros((ny, nz))+  np.nan
    for k in range(ny):
        # print(k)
        # if k <200:
        #     continue
        x = wavelengths[k]
        y = im[k]/im_err[k]
        q = im_badpixs[k]
        s = im_err[k]

        where_finite = np.where(np.isfinite(q) * np.isfinite(y) * (s != 0.0))
        if np.size(where_finite[0]) < nx // 4:
            continue
        f = interp1d(x[where_finite], y[where_finite], bounds_error=False, fill_value=np.nan, kind="linear")
        new_im[k, :] = f(new_wvs)
        # plt.plot(x,y)
        # plt.plot(new_wvs,new_im[k,:])
        # plt.show()
        # exit()
    new_im[:,np.where(np.sum(np.isfinite(new_im),axis=0)<100)[0]]=np.nan
    med_spec = np.nanmedian(new_im, axis=0)
    # plt.plot(med_spec)
    # plt.show()
    # new_im[where_nan] = med_spec[where_nan[1]]
    # print(new_im.shape)
    # plt.plot(np.sum(np.isfinite(new_im),axis=1))
    # plt.show()
    new_im = new_im[np.where(np.sum(np.isfinite(new_im),axis=1)!=0)[0],:]
    # print(new_im.shape)
    # exit()
    # for k in range(new_im.shape[0]):
    #     plt.plot(new_im[k])
    # plt.show()

    new_im = new_im / np.nanstd(new_im, axis=1)[:, None]
    where_nan = np.where(np.isnan(new_im))
    new_im[where_nan] = 0

    X = new_im
    # X = X[np.where(np.nansum(X, axis=1) != 0)[0], :]
    # X = X / np.nanstd(X, axis=1)[:, None]
    # X[np.where(np.isnan(X))] = np.tile(np.nanmedian(X, axis=0)[None, :], (X.shape[0], 1))[np.where(np.isnan(X))]
    # X[np.where(np.isnan(X))] = 0

    # print(X.shape)
    C = np.cov(X)
    # print(C.shape)
    # exit()
    tot_basis = C.shape[0]
    tmp_res_numbasis = np.clip(np.abs(N_KL) - 1, 0,
                               tot_basis - 1)  # clip values, for output consistency we'll keep duplicates
    max_basis = np.max(
        tmp_res_numbasis) + 1  # maximum number of eigenvectors/KL basis we actually need to use/calculate
    evals, evecs = la.eigh(C, eigvals=(tot_basis - max_basis, tot_basis - 1))
    check_nans = np.any(evals <= 0)  # alternatively, check_nans = evals[0] <= 0
    evals = np.copy(evals[::-1])
    evecs = np.copy(evecs[:, ::-1], order='F')  # fortran order to improve memory caching in matrix multiplication
    # calculate the KL basis vectors
    kl_basis = np.dot(X.T, evecs)
    kls = kl_basis * (1. / np.sqrt(evals * (nz - 1)))[None, :]  # multiply a value for each row
    print(kls.shape)

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


def find_bleeding_bar(ra_arr, dec_arr, threshold2mask=0.15):
    nx = ra_arr.shape[1]
    dist2host_as_col = np.sqrt((ra_arr[:, nx // 2]) ** 2 + (dec_arr[:, nx // 2]) ** 2)
    k = np.nanargmin(dist2host_as_col)
    x, y = ra_arr[k - 10:k + 10, nx // 2], dec_arr[k - 10:k + 10, nx // 2]
    where_finite = np.where(np.isfinite(x) * np.isfinite(y))
    bleeding_slope = np.polyfit(x[where_finite], y[where_finite], 1)[0]
    # Calculate the distances of each point to the line
    distances2bleeding = np.abs(dec_arr - bleeding_slope * ra_arr) / np.sqrt(1 + bleeding_slope ** 2)
    # Find the indices of the points within the threshold distance
    return np.where(distances2bleeding < threshold2mask)


def _get_wpsf_task(paras):
    nrs, center_wv, wpsf_oversample = paras
    kernel = np.ones((wpsf_oversample, wpsf_oversample))
    ext = 'OVERSAMP'
    slicepsf_wv0 = nrs.calc_psf(monochromatic=center_wv * 1e-6,  # Wavelength, in **METERS**
                                fov_arcsec=6,  # angular size to simulate PSF over
                                oversample=wpsf_oversample,
                                # output pixel scale relative to the pixelscale set above
                                add_distortion=False)  # skip an extra computation step that's not relevant for IFU
    webbpsfim = slicepsf_wv0[ext].data
    smoothed_im = convolve2d(webbpsfim, kernel, mode='same')
    return webbpsfim, smoothed_im


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


def filter_big_triangles(X,Y, max_edge_length):
    points = np.array([X,Y]).T
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
        linear_interp, wepsf, wifuX, wifuY, east2V2_deg,flipx, _X, _Y, _Z, _Zerr, _Zbad, IWA, OWA, fit_cen, fit_angle, init_paras = paras
        ann_width, padding, sector_area = None, 0.0, None
    else:
        linear_interp, wepsf, wifuX, wifuY, east2V2_deg,flipx, _X, _Y, _Z, _Zerr, _Zbad, IWA, OWA, fit_cen, fit_angle, init_paras, ann_width, padding, sector_area = paras
    # R = np.sqrt((X)**2+ (Y)**2)
    # Xrav, Yrav, Zrav, Zerrrav, Zbadrav = _X.ravel(), _Y.ravel(), _Z.ravel(), _Zerr.ravel(), _Zbad.ravel()
    _R = np.sqrt((_X - init_paras[0]) ** 2 + (_Y - init_paras[1]) ** 2)
    _PA = np.arctan2(_X- init_paras[0], _Y- init_paras[1]) % (2 * np.pi)

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
    out_model = np.zeros(_Z.shape)+ np.nan
    # print(iterator_sectors)
    # exit()
    for sector_id, sector in enumerate(iterator_sectors):
        # print(sector)
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

        # if deltaphi2 >= 2 * np.pi:
        #     pamin_pad2 = 0
        #     pamax_pad2 = 2 * np.pi
        # else:
        #     pamin_pad2 = ((pamin) - padding2 / np.mean([rmin, rmax])) % (2.0 * np.pi)
        #     pamax_pad2 = ((pamax) + padding2 / np.mean([rmin, rmax])) % (2.0 * np.pi)

        rmin_pad = np.max([rmin - padding, 0.0])
        rmax_pad = rmax + padding
        # rmin_pad2 = np.max([rmin - padding2, 0.0])
        # rmax_pad2 = rmax + padding2
        #
        if pamin_pad < pamax_pad:
            fit_sector = (rmin_pad < _R) & (_R < rmax_pad) & (pamin_pad < _PA) & (_PA < pamax_pad) & np.isfinite(_Zbad)
        else:
            fit_sector = (rmin_pad < _R) & (_R < rmax_pad) & ((pamin_pad < _PA) | (_PA < pamax_pad)) & np.isfinite(_Zbad)
        if pamin < pamax:
            sc_sector = (rmin < _R) & (_R < rmax) & (pamin < _PA) & (_PA < pamax) & np.isfinite(_Zbad)
        else:
            sc_sector = (rmin < _R) & (_R < rmax) & ((pamin < _PA) | (_PA < pamax)) & np.isfinite(_Zbad)

        where_fit = np.where(fit_sector)
        if np.size(where_fit[0])<1:
            continue
        X, Y, Z, Zerr, Zbad = _X[where_fit], _Y[where_fit], _Z[where_fit], _Zerr[where_fit], _Zbad[where_fit]
        # Zerr = Z/10.
        where_sc = np.where(sc_sector)
        if np.size(where_sc[0])<1:
            continue
        # Xsc, Ysc, Zsc, Zerrsc, Zbadsc = _X[where_sc], _Y[where_sc], _Z[where_sc], _Zerr[where_sc], _Zbad[where_sc]
        Xsc, Ysc = _X[where_sc], _Y[where_sc]

        where_wepsf_finite = np.where(np.isfinite(wepsf))
        # print(np.size(where_wepsf_finite[0]))
        if np.size(where_wepsf_finite[0])<3:
            continue
        wX, wY, wZ = wifuX[where_wepsf_finite], wifuY[where_wepsf_finite], wepsf[where_wepsf_finite]
        wX, wY = rotate_coordinates(wX, wY, -east2V2_deg, flipx=flipx)

        # plt.scatter(wX, wY,s=100*wZ/np.nanmax(wZ))
        # plt.show()

        # if linear_interp:
        #     filtered_triangles = filter_big_triangles(wX, wY,0.2)
        #     # Create filtered triangulation
        #     filtered_tri = tri.Triangulation(wX, wY, triangles=filtered_triangles)
        #
        #     # Perform LinearTriInterpolator for filtered triangulation
        #     webbpsf_interp = tri.LinearTriInterpolator(filtered_tri, wZ)
        #
        #     # webbpsf_interp = LinearNDInterpolator(filtered_tri.points, wZ, fill_value=0.0)
        # else:
        #     webbpsf_interp = CloughTocher2DInterpolator(filtered_tri.points, wZ, fill_value=0.0)
        if linear_interp:
            webbpsf_interp = LinearNDInterpolator((wX, wY), wZ, fill_value=0.0)
        else:
            webbpsf_interp = CloughTocher2DInterpolator((wX, wY), wZ, fill_value=0.0)
        # slice_currmodel = np.zeros(slice_flux.shape)
        # slice_all_ampls = np.zeros((Nit_max + 1, np.size(dec_vec), np.size(ra_vec))) + np.nan
        # slice_ampls_err = np.zeros((np.size(dec_vec), np.size(ra_vec))) + np.nan

        # #PA_V3   =    64.98555793945364 / [deg] Position angle of telescope V3 axis
        if 0:
            fluxfinite = np.isfinite(Zbad)
            wherefluxfinite = np.where(fluxfinite)
            X, Y, Z, Zerr, Zbad = X[wherefluxfinite], Y[wherefluxfinite], Z[wherefluxfinite], Zerr[wherefluxfinite], \
            Zbad[wherefluxfinite]
            dra = 0.01
            ddec = 0.01
            ra_vec = np.arange(-2.5, 2.1, dra)
            dec_vec = np.arange(-3.0, 1.9, ddec)
            ra_grid, dec_grid = np.meshgrid(ra_vec, dec_vec)
            r_grid = np.sqrt(ra_grid ** 2 + dec_grid ** 2)
            interpolator_sc = LinearNDInterpolator((X, Y), Z, fill_value=0.0)
            plt.subplot(1, 2, 1)
            plt.imshow(interpolator_sc(ra_grid, dec_grid), interpolation="nearest", origin="lower",
                       extent=[ra_vec[0] - dra / 2., ra_vec[-1] + dra / 2., dec_vec[0] - ddec / 2.,
                               dec_vec[-1] + ddec / 2.])
            plt.clim([-1e-9, 1e-9])
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.imshow(webbpsf_interp(ra_grid, dec_grid), interpolation="nearest", origin="lower",
                       extent=[ra_vec[0] - dra / 2., ra_vec[-1] + dra / 2., dec_vec[0] - ddec / 2.,
                               dec_vec[-1] + ddec / 2.])
            plt.clim([-2e-9, 2e-9])
            # plt.clim([-1e-5, 1e-5])
            plt.colorbar()
            webbpsf_interp = CloughTocher2DInterpolator((rotate_coordinates(wX, wY, 65 + 138.5, flipx=True)), wZ,
                                                        fill_value=0.0)
            # plt.subplot(2,2,3)
            # plt.imshow(webbpsf_interp(ra_grid, dec_grid),interpolation="nearest",origin="lower",extent=[ra_vec[0]-dra/2.,ra_vec[-1]+dra/2.,dec_vec[0]-ddec/2.,dec_vec[-1]+ddec/2.])
            # plt.clim([-0.5e-2,0.5e-2])
            # webbpsf_interp = CloughTocher2DInterpolator((rotate_coordinates(wX, wY, -65-138.5,flipx=True)),wZ, fill_value=0.0)
            # plt.subplot(2,2,4)
            # plt.imshow(webbpsf_interp(ra_grid, dec_grid),interpolation="nearest",origin="lower",extent=[ra_vec[0]-dra/2.,ra_vec[-1]+dra/2.,dec_vec[0]-ddec/2.,dec_vec[-1]+ddec/2.])
            # plt.clim([-0.5e-2,0.5e-2])
            plt.show()

        # wherefluxfinite4fit = np.where(np.isfinite(Zbad)*(R > IWA) * (R < OWA))
        # X,Y,Z,Zerr,Zbad = X[wherefluxfinite4fit], Y[wherefluxfinite4fit], Z[wherefluxfinite4fit],Zerr[wherefluxfinite4fit], Zbad[wherefluxfinite4fit]
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
        # print(a,xc, yc, th)
        # 0.00014184273586106017 -0.1364760635130025 -0.08421747641555134 0.0013115802927045803
        # 0.0001436426668425129 -0.1330706312425748 -0.0827898233517135 0.001335870184394232

        if 0:
            print(a, xc, yc, th)
            print(np.nansum(m0)/4.)
            plt.figure(1)
            plt.scatter(X, Y, s=Z / np.nanmedian(Z))
            plt.axis('equal')

            plt.figure(3)
            plt.scatter(X,Z,c="orange")
            plt.scatter(X,a*m0,c="blue")
            plt.scatter(X,Z-a*m0,c="grey")

            dra = 0.01
            ddec = 0.01
            ra_vec = np.arange(-2.5, 2.1, dra)
            dec_vec = np.arange(-3.0, 1.9, ddec)
            ra_grid, dec_grid = np.meshgrid(ra_vec, dec_vec)
            r_grid = np.sqrt(ra_grid ** 2 + dec_grid ** 2)
            interpolator_sc = LinearNDInterpolator((X, Y), Z, fill_value=0.0)
            plt.figure(2)
            plt.subplot(1, 3, 1)
            lim = 1e-10
            plt.imshow(interpolator_sc(ra_grid, dec_grid), interpolation="nearest", origin="lower",
                       extent=[ra_vec[0] - dra / 2., ra_vec[-1] + dra / 2., dec_vec[0] - ddec / 2.,
                               dec_vec[-1] + ddec / 2.])
            plt.clim([-lim, lim])
            plt.colorbar()
            plt.subplot(1, 3, 2)
            plt.imshow(a * webbpsf_interp(ra_grid - xc, dec_grid - yc), interpolation="nearest", origin="lower",
                       extent=[ra_vec[0] - dra / 2., ra_vec[-1] + dra / 2., dec_vec[0] - ddec / 2.,
                               dec_vec[-1] + ddec / 2.])
            # plt.imshow(a * webbpsf_interp(ra_grid, dec_grid), interpolation="nearest", origin="lower",
            #            extent=[ra_vec[0] - dra / 2., ra_vec[-1] + dra / 2., dec_vec[0] - ddec / 2.,
            #                    dec_vec[-1] + ddec / 2.])
            plt.clim([-lim, lim])
            plt.colorbar()
            plt.subplot(1, 3, 3)
            plt.imshow(interpolator_sc(ra_grid, dec_grid) - a * webbpsf_interp(ra_grid - xc, dec_grid - yc),
                       interpolation="nearest", origin="lower",
                       extent=[ra_vec[0] - dra / 2., ra_vec[-1] + dra / 2., dec_vec[0] - ddec / 2.,
                               dec_vec[-1] + ddec / 2.])
            plt.clim([-lim, lim])
            plt.colorbar()
            # webbpsf_interp = CloughTocher2DInterpolator((rotate_coordinates(wX, wY, 65+138.5,flipx=True)),wZ, fill_value=0.0)
            # plt.subplot(2,2,3)
            # plt.imshow(webbpsf_interp(ra_grid, dec_grid),interpolation="nearest",origin="lower",extent=[ra_vec[0]-dra/2.,ra_vec[-1]+dra/2.,dec_vec[0]-ddec/2.,dec_vec[-1]+ddec/2.])
            # plt.clim([-0.5e-2,0.5e-2])
            # webbpsf_interp = CloughTocher2DInterpolator((rotate_coordinates(wX, wY, -65-138.5,flipx=True)),wZ, fill_value=0.0)
            # plt.subplot(2,2,4)
            # plt.imshow(webbpsf_interp(ra_grid, dec_grid),interpolation="nearest",origin="lower",extent=[ra_vec[0]-dra/2.,ra_vec[-1]+dra/2.,dec_vec[0]-ddec/2.,dec_vec[-1]+ddec/2.])
            # plt.clim([-0.5e-2,0.5e-2])
            plt.show()

        out_paras[sector_id, :] = np.array([a0, a, xc, yc, th])
        out_model[where_sc] = a * webbpsf_interp(Xsc - xc, Ysc - yc)
    return out_paras, out_model


def _interp_psf(paras):
    linear_interp, wepsf, wifuX, wifuY, wv_id, east2V2_deg = paras
    print(wv_id)
    wX, wY, wZ = wifuX.ravel(), wifuY.ravel(), wepsf.flatten()
    wX, wY = rotate_coordinates(wX, wY, -east2V2_deg, flipx=True)

    wherepsffinite = np.where(np.isfinite(wZ))
    wX, wY, wZ = wX[wherepsffinite], wY[wherepsffinite], wZ[wherepsffinite]
    if linear_interp:
        webbpsf_interp = LinearNDInterpolator((wX, wY), wZ, fill_value=0.0)
    else:
        webbpsf_interp = CloughTocher2DInterpolator((wX, wY), wZ, fill_value=0.0)

    return webbpsf_interp



def fitpsf(dataobj_list, psfs, psfX, psfY, out_filename=None, load=True, IWA=0, OWA=np.inf, mppool=None,
           init_centroid=None, run_init=False,fit_cen=True, fit_angle = False,
           ann_width=None, padding=0.0, sector_area=None, RDI_folder_suffix=None,
           linear_interp=True,rotate_psf=0.0,flipx=False,psf_spaxel_area=None):
    if RDI_folder_suffix is None:
        RDI_folder_suffix = ""

    print("Make sure interpdata_regwvs was already done ")
    dataobj0 = dataobj_list[0]

    all_interp_ra, all_interp_dec, all_interp_wvs, all_interp_flux, all_interp_err, all_interp_badpix,all_interp_area2d = \
        dataobj0.interpdata_regwvs(wv_sampling=None, modelfit=False, out_filename=dataobj0.interpdata_regwvs_filename,
                                   load_interpdata_regwvs=True)
    # print(all_interp_ra.shape)
    # plt.scatter(all_interp_ra[:,1000],all_interp_dec[:,1000],s=100*all_interp_flux[:,1000]/np.nanmax(all_interp_flux[:,1000]))
    # plt.xlim([init_centroid[0]-0.5,init_centroid[0]+0.5])
    # plt.ylim([init_centroid[1]-0.5,init_centroid[1]+0.5])
    # plt.show()
    wv_sampling = dataobj0.wv_sampling
    if len(dataobj_list) > 1:
        for dataobj in dataobj_list[1::]:
            interp_ra, interp_dec, interp_wvs, interp_flux, interp_err, interp_badpix, interp_area2d = \
                dataobj.interpdata_regwvs(wv_sampling=None, modelfit=False,
                                          out_filename=dataobj.interpdata_regwvs_filename, load_interpdata_regwvs=True)
            # print(all_interp_ra.shape)
            # plt.scatter(interp_ra[:,1000],interp_dec[:,1000],s=100*interp_flux[:,1000]/np.nanmax(interp_flux[:,1000]))
            # plt.xlim([init_centroid[0]-1,init_centroid[0]+1])
            # plt.ylim([init_centroid[1]-1,init_centroid[1]+1])
            # plt.show()
            all_interp_ra = np.concatenate((all_interp_ra, interp_ra), axis=0)
            all_interp_dec = np.concatenate((all_interp_dec, interp_dec), axis=0)
            all_interp_wvs = np.concatenate((all_interp_wvs, interp_wvs), axis=0)
            all_interp_flux = np.concatenate((all_interp_flux, interp_flux), axis=0)
            all_interp_err = np.concatenate((all_interp_err, interp_err), axis=0)
            all_interp_badpix = np.concatenate((all_interp_badpix, interp_badpix), axis=0)
            all_interp_area2d = np.concatenate((all_interp_area2d, interp_area2d), axis=0)
    # plt.show()
    # print(all_interp_ra.shape)
    # plt.scatter(all_interp_ra[:,1000],all_interp_dec[:,1000],s=all_interp_flux[:,1000]/np.nanmedian(all_interp_flux))
    # plt.show()
    # print(np.nanmedian(all_interp_area2d),psf_spaxel_area)
    # exit()
    all_interp_flux = all_interp_flux/all_interp_area2d*psf_spaxel_area
    all_interp_err = all_interp_err/all_interp_area2d*psf_spaxel_area

    all_interp_psfmodel = np.zeros(all_interp_flux.shape) + np.nan
    all_interp_psfsub = np.zeros(all_interp_flux.shape) + np.nan

    if init_centroid is None:
        init_paras = np.array([0, 0])
    else:
        init_paras = np.array(init_centroid)

    print(len(glob(out_filename)), out_filename)
    if 0 and load and len(glob(out_filename)):
        with pyfits.open(out_filename) as hdulist:
            bestfit_coords = hdulist[0].data
            wpsf_angle_offset = hdulist[0].header["INIT_ANG"]
            wpsf_ra_offset = hdulist[0].header["INIT_RA"]
            wpsf_dec_offset = hdulist[0].header["INIT_DEC"]
            all_interp_psfsub = hdulist[1].data
            all_interp_psfmodel = hdulist[2].data
            # print(all_interp_psfsub.shape)
            # exit()
        # plt.imshow(all_interp_psfsub,origin="lower")
        # plt.show()

        #     RDI_psfsub_dir = os.path.join(os.path.dirname(out_filename), "RDI_psfsub"+RDI_folder_suffix)
        #     if not os.path.exists(RDI_psfsub_dir):
        #         os.makedirs(RDI_psfsub_dir)
        #     RDI_model_dir = os.path.join(os.path.dirname(out_filename), "RDI_model"+RDI_folder_suffix)
        #     if not os.path.exists(RDI_model_dir):
        #         os.makedirs(RDI_model_dir)
        #
        #     for obj_id,dataobj in enumerate(dataobj_list):
        #         ny = dataobj.data.shape[0]
        #         _interpdata_regwvs_filename = os.path.join(os.path.dirname(dataobj.interpdata_regwvs_filename),"RDI_psfsub"+RDI_folder_suffix,os.path.basename(dataobj.interpdata_regwvs_filename))
        #         hdulist = pyfits.HDUList()
        #         hdulist.append(pyfits.PrimaryHDU(data=all_interp_psfsub[(ny * obj_id):(ny * (obj_id+1)), :]))
        #         hdulist.append(pyfits.ImageHDU(data=all_interp_err[(ny * obj_id):(ny * (obj_id+1)), :]))
        #         hdulist.append(pyfits.ImageHDU(data=all_interp_ra[(ny * obj_id):(ny * (obj_id+1)), :]))
        #         hdulist.append(pyfits.ImageHDU(data=all_interp_dec[(ny * obj_id):(ny * (obj_id+1)), :]))
        #         hdulist.append(pyfits.ImageHDU(data=all_interp_wvs[(ny * obj_id):(ny * (obj_id+1)), :]))
        #         hdulist.append(pyfits.ImageHDU(data=all_interp_badpix[(ny * obj_id):(ny * (obj_id+1)), :]))
        #         hdulist.append(pyfits.ImageHDU(data=all_interp_area2d[(ny * obj_id):(ny * (obj_id+1)), :]))
        #         try:
        #             hdulist.writeto(_interpdata_regwvs_filename, overwrite=True)
        #         except TypeError:
        #             hdulist.writeto(_interpdata_regwvs_filename, clobber=True)
        #         hdulist.close()
        #
        #         hdulist_sc = pyfits.open(dataobj.filename)
        #         new_model = np.zeros(dataobj.data.shape)+ np.nan
        #         new_badpix = np.zeros(dataobj.data.shape)+ np.nan
        #         for rowid in range(interp_flux.shape[0]):
        #             new_model[rowid, :] = np.interp(dataobj.wavelengths[rowid, :],wv_sampling, all_interp_psfmodel[(ny * obj_id+rowid), :],left=np.nan, right=np.nan)
        #             badpix_mask = np.isfinite(all_interp_psfmodel[(ny * obj_id+rowid), :]).astype(float)
        #             new_badpix[rowid, :] = np.interp(dataobj.wavelengths[rowid, :], wv_sampling, badpix_mask,left=np.nan, right=np.nan)
        #
        #         dataobj.bad_pixels[np.where(new_badpix != 1.0)] = np.nan
        #
        #         where_mask = np.where(np.isfinite(dataobj.bad_pixels))
        #         where_bad = np.where(np.isnan(dataobj.bad_pixels))
        #         hdulist_sc["SCI"].data[where_mask] = dataobj.data[where_mask] - new_model[where_mask]
        #         hdulist_sc["DQ"].data[where_bad] = 1
        #         # Write the new HDU list to a new FITS file
        #
        #         psfsub_filename = os.path.join(RDI_psfsub_dir, os.path.basename(dataobj.filename))
        #         try:
        #             hdulist_sc.writeto(psfsub_filename, overwrite=True)
        #         except TypeError:
        #             hdulist_sc.writeto(psfsub_filename, clobber=True)
        #
        #         hdulist_sc["SCI"].data[where_mask] = new_model[where_mask]
        #         psfmod_filename = os.path.join(RDI_model_dir, os.path.basename(dataobj.filename))
        #         try:
        #             hdulist_sc.writeto(psfmod_filename, overwrite=True)
        #         except TypeError:
        #             hdulist_sc.writeto(psfmod_filename, clobber=True)
        #
        #         hdulist_sc.close()
        # exit()
    else:
        if run_init:
            raise Exception("These initializations are bad")
            # rough centroid fit
            wv_id = np.size(wv_sampling) // 2
            _fit_cen, _fit_angle = True, False
            # init_paras = np.array([self.wpsf_ra_offset, self.wpsf_dec_offset, self.wpsf_angle_offset])
            # init_paras = np.array([wpsf_ra_offset, wpsf_dec_offset])
            # print(init_paras)
            print("init started", init_paras)
            # wheredata2fit = np.where(
            #     (self.wavelengths > (wv_sampling[0] + 0.2)) * (self.wavelengths < (wv_sampling[-1] - 0.2)))
            # paras = linear_interp, wepsfs[wv_id, :, :], webbpsf_X, webbpsf_Y, self.east2V2_deg, \
            #     self.dra_as_array[wheredata2fit] / self.wavelengths[wheredata2fit] * wv_sampling[wv_id], \
            #     self.ddec_as_array[wheredata2fit] / self.wavelengths[wheredata2fit] * wv_sampling[wv_id], \
            #     self.data[wheredata2fit], self.noise[wheredata2fit], self.bad_pixels[wheredata2fit], \
            #     self.wpsffit_IWA, 10.0, fit_cen, fit_angle, init_paras
            paras = linear_interp, psfs[wv_id], psfX[wv_id], psfY[wv_id], rotate_psf,flipx, \
                all_interp_ra[:, wv_id], all_interp_dec[:, wv_id], all_interp_flux[:, wv_id], all_interp_err[:, wv_id], all_interp_badpix[:,wv_id], \
                0, 10, _fit_cen, _fit_angle, init_paras
            # paras = psfs[wv_id,:,:], psfX, psfY,rotate_psf,\
            #     all_interp_ra/all_interp_wvs*wv_sampling[wv_id], all_interp_dec/all_interp_wvs*wv_sampling[wv_id], all_interp_flux, all_interp_err,all_interp_badpix,\
            #     IWA,OWA,_fit_cen,_fit_angle,init_paras
            out, _ = _fit_wpsf_task(paras)
            # out = [0.00014386691687160602, -0.13301339368089535 ,-0.08327523187469463 ,0.001383654850629076]
            wpsf_ra_offset, wpsf_dec_offset, wpsf_angle_offset = out[0,2::]
            print("init done", out)
            init_paras = [wpsf_ra_offset, wpsf_dec_offset]
        else:
            init_paras = init_centroid
            wpsf_angle_offset = 0
        # exit()
        #
        # cp_interp_badpix = copy(interp_badpix)
        # if mask_charge_bleeding:
        #     indices_within_threshold = find_bleeding_bar(interp_ra-self.wpsf_ra_offset,interp_dec-self.wpsf_dec_offset,threshold2mask=0.15)
        #     cp_interp_badpix[indices_within_threshold] = np.nan

        # print(init_paras)

        bestfit_coords_defined = False
        if 0 or mppool is None:
            for wv_id, wv in enumerate(wv_sampling):
                # if wv_id < 1000:  # or wv_id>300
                #     continue
                # if wv_id != np.argmin(np.abs(wv_sampling-4.592)):  # or wv_id>300
                #     continue
                # if wv_id != np.argmin(np.abs(wv_sampling-3.106)):  # or wv_id>300
                # if wv_id != np.argmin(np.abs(wv_sampling-3.1325)):  # or wv_id>300
                # if wv_id != np.argmin(np.abs(wv_sampling-3.66)):  # or wv_id>300
                #     continue
                # if wv_id % 50 != 0:
                #     continue
                print(wv_id, wv, np.size(wv_sampling))
                paras = linear_interp, psfs[wv_id], psfX[wv_id], psfY[wv_id], rotate_psf - wpsf_angle_offset,flipx, \
                    all_interp_ra[:, wv_id], all_interp_dec[:, wv_id], all_interp_flux[:, wv_id], all_interp_err[:,wv_id], all_interp_badpix[:, wv_id], \
                    IWA, OWA, fit_cen, fit_angle, init_paras, ann_width, padding, sector_area
                out = _fit_wpsf_task(paras)
                # plt.imshow(out[0])
                # plt.show()
                # print(out[0].shape,out[1].shape)
                # exit()
                if not bestfit_coords_defined:
                    bestfit_coords = np.zeros((out[0].shape[0],np.size(wv_sampling), 5)) + np.nan  # flux_init, flux,ra,dec,angle
                    bestfit_coords_defined=True
                bestfit_coords[:,wv_id, :] = out[0]
                all_interp_psfmodel[:, wv_id] = out[1]
                all_interp_psfsub[:, wv_id] = all_interp_flux[:, wv_id] - out[1]

                plt.figure(1)
                plt.scatter(all_interp_ra[:, wv_id], all_interp_flux[:, wv_id], label="interp_flux")
                plt.scatter(all_interp_ra[:, wv_id], out[1], label="model")
                plt.scatter(all_interp_ra[:, wv_id], all_interp_flux[:, wv_id] - out[1], label="res")

                plt.figure(2)
                plt.scatter(all_interp_ra[:, wv_id], (all_interp_flux[:, wv_id] - out[1]) / all_interp_err[:, wv_id],
                            label="res")
                plt.show()
            exit()
        else:
            # paras = psfs[wv_id,:,:], psfX, psfY,self.east2V2_deg - self.wpsf_angle_offset,\
            #     interp_ra[:,wv_id], interp_dec[:,wv_id], interp_flux[:,wv_id], interp_err[:,wv_id],interp_badpix[:,wv_id],\
            #     IWA,OWA,fit_cen,fit_angle,init_paras
            output_lists = mppool.map(_fit_wpsf_task,
                                      zip(itertools.repeat(linear_interp),
                                          psfs, psfX, psfY,
                                          itertools.repeat(rotate_psf - wpsf_angle_offset),
                                          itertools.repeat(flipx),
                                          all_interp_ra.T,
                                          all_interp_dec.T,
                                          all_interp_flux.T,
                                          all_interp_err.T,
                                          all_interp_badpix.T,
                                          itertools.repeat(IWA),
                                          itertools.repeat(OWA),
                                          itertools.repeat(fit_cen),
                                          itertools.repeat(fit_angle),
                                          itertools.repeat(init_paras),
                                          itertools.repeat(ann_width),
                                          itertools.repeat(padding),
                                          itertools.repeat(sector_area)))

            for wv_id, (wv, out) in enumerate(zip(wv_sampling, output_lists)):
                print(wv_id, np.size(wv_sampling))
                if not bestfit_coords_defined:
                    bestfit_coords = np.zeros((out[0].shape[0],np.size(wv_sampling), 5)) + np.nan  # flux_init, flux,ra,dec,angle
                    bestfit_coords_defined=True
                bestfit_coords[:,wv_id, :] = out[0]
                all_interp_psfmodel[:, wv_id] = out[1]
                all_interp_psfsub[:, wv_id] = all_interp_flux[:, wv_id] - out[1]

        # plt.imshow(wpsfs[0])
        # plt.show()
        if out_filename is not None:
            wpsfsfit_header = {"INIT_ANG": wpsf_angle_offset,
                               "INIT_RA": init_paras[0], "INIT_DEC": init_paras[1]}
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=bestfit_coords, header=pyfits.Header(cards=wpsfsfit_header)))
            hdulist.append(pyfits.ImageHDU(data=(all_interp_psfsub/psf_spaxel_area)*all_interp_area2d))
            hdulist.append(pyfits.ImageHDU(data=all_interp_psfmodel))
            try:
                hdulist.writeto(out_filename, overwrite=True)
            except TypeError:
                hdulist.writeto(out_filename, clobber=True)
            hdulist.close()

            RDI_psfsub_dir = os.path.join(os.path.dirname(out_filename), "RDI_psfsub"+RDI_folder_suffix)
            if not os.path.exists(RDI_psfsub_dir):
                os.makedirs(RDI_psfsub_dir)
            RDI_model_dir = os.path.join(os.path.dirname(out_filename), "RDI_model"+RDI_folder_suffix)
            if not os.path.exists(RDI_model_dir):
                os.makedirs(RDI_model_dir)

            for obj_id,dataobj in enumerate(dataobj_list):
                ny = dataobj.data.shape[0]
                _interpdata_regwvs_filename = os.path.join(os.path.dirname(dataobj.interpdata_regwvs_filename),"RDI_psfsub"+RDI_folder_suffix,os.path.basename(dataobj.interpdata_regwvs_filename))
                hdulist = pyfits.HDUList()
                hdulist.append(pyfits.PrimaryHDU(data=all_interp_psfsub[(ny * obj_id):(ny * (obj_id+1)), :]))
                hdulist.append(pyfits.ImageHDU(data=all_interp_err[(ny * obj_id):(ny * (obj_id+1)), :]))
                hdulist.append(pyfits.ImageHDU(data=all_interp_ra[(ny * obj_id):(ny * (obj_id+1)), :]))
                hdulist.append(pyfits.ImageHDU(data=all_interp_dec[(ny * obj_id):(ny * (obj_id+1)), :]))
                hdulist.append(pyfits.ImageHDU(data=all_interp_wvs[(ny * obj_id):(ny * (obj_id+1)), :]))
                hdulist.append(pyfits.ImageHDU(data=all_interp_badpix[(ny * obj_id):(ny * (obj_id+1)), :]))
                hdulist.append(pyfits.ImageHDU(data=all_interp_area2d[(ny * obj_id):(ny * (obj_id+1)), :]))
                try:
                    hdulist.writeto(_interpdata_regwvs_filename, overwrite=True)
                except TypeError:
                    hdulist.writeto(_interpdata_regwvs_filename, clobber=True)
                hdulist.close()

                hdulist_sc = pyfits.open(dataobj.filename)
                new_model = np.zeros(dataobj.data.shape)+ np.nan
                new_badpix = np.zeros(dataobj.data.shape)+ np.nan
                for rowid in range(interp_flux.shape[0]):
                    new_model[rowid, :] = np.interp(dataobj.wavelengths[rowid, :],wv_sampling, all_interp_psfmodel[(ny * obj_id+rowid), :],left=np.nan, right=np.nan)
                    badpix_mask = np.isfinite(all_interp_psfmodel[(ny * obj_id+rowid), :]).astype(float)
                    new_badpix[rowid, :] = np.interp(dataobj.wavelengths[rowid, :], wv_sampling, badpix_mask,left=np.nan, right=np.nan)

                dataobj.bad_pixels[np.where(new_badpix != 1.0)] = np.nan

                where_mask = np.where(np.isfinite(dataobj.bad_pixels))
                where_bad = np.where(np.isnan(dataobj.bad_pixels))
                hdulist_sc["SCI"].data[where_mask] = dataobj.data[where_mask] - new_model[where_mask]
                hdulist_sc["DQ"].data[where_bad] = 1
                # Write the new HDU list to a new FITS file

                psfsub_filename = os.path.join(RDI_psfsub_dir, os.path.basename(dataobj.filename))
                try:
                    hdulist_sc.writeto(psfsub_filename, overwrite=True)
                except TypeError:
                    hdulist_sc.writeto(psfsub_filename, clobber=True)

                hdulist_sc["SCI"].data[where_mask] = new_model[where_mask]
                psfmod_filename = os.path.join(RDI_model_dir, os.path.basename(dataobj.filename))
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

            res = all_interp_psfsub - mfflux*sampled_psf
            noise_factor = np.nanstd(res/all_interp_err)

            flux_map[dec_id, ra_id] = mfflux
            fluxerr_map[dec_id, ra_id] = mffluxerr*noise_factor
            # where_finite = np.where(np.isfinite(all_interp_badpix))
            # X = all_interp_ra[where_finite]
            # Y = all_interp_dec

    snr_map = flux_map / fluxerr_map
    if out_filename is not None:
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=flux_map))
        hdulist.append(pyfits.ImageHDU(data=fluxerr_map))
        hdulist.append(pyfits.ImageHDU(data=snr_map))
        hdulist.append(pyfits.ImageHDU(data=ra_grid))
        hdulist.append(pyfits.ImageHDU(data=dec_grid))
        try:
            hdulist.writeto(out_filename, overwrite=True)
        except TypeError:
            hdulist.writeto(out_filename, clobber=True)
        hdulist.close()
    return snr_map, flux_map, fluxerr_map, ra_grid, dec_grid


    # dataobj0 = dataobj_list[0]
    # wv_sampling = dataobj0.wv_sampling
    # east2V2_deg = dataobj0.east2V2_deg
def matchedfilter_perwv(wv_sampling,east2V2_deg,
                all_interp_ra,all_interp_dec,all_interp_flux,all_interp_err,all_interp_badpix,N_dithers,
                        psfs, psfX, psfY, ra_vec, dec_vec, planet_f, out_filename=None,
                  load=True, linear_interp=True, mppool=None, aper_radius=0.5, rv=0,noise_prop2flux=False,
                        psf_interp_list=None):
    print("Make sure interpdata_regwvs was already done ")

    comp_spec = planet_f(wv_sampling * (1 - (rv) / const.c.to('km/s').value)) * (u.W / u.m ** 2 / u.um)
    # comp_spec = comp_spec * dataobj0.aper_to_epsf_peak_f(wv_sampling)  # normalized to peak flux
    comp_spec = comp_spec * (wv_sampling * u.um) ** 2 / const.c  # from  Flambda to Fnu
    comp_spec = comp_spec.to(u.MJy).value

    ra_grid, dec_grid = np.meshgrid(ra_vec, dec_vec)
    r_grid = np.sqrt(ra_grid ** 2 + dec_grid ** 2)
    PA_grid = np.arctan2(ra_grid, dec_grid) % 2 * np.pi

    flux_cube = np.zeros((np.size(wv_sampling),ra_grid.shape[0],ra_grid.shape[1])) +np.nan
    fluxerr_cube = np.zeros((np.size(wv_sampling),ra_grid.shape[0],ra_grid.shape[1])) +np.nan
    flux_map = np.zeros(ra_grid.shape) +np.nan
    fluxerr_map = np.zeros(ra_grid.shape) +np.nan

    print("create psf model")
    # test  = np.where((wv_sampling>3.2465)*(wv_sampling<3.250))
    # test  = np.where((wv_sampling>4.8)*(wv_sampling<4.85))
    # debug_init = test[0][0]
    # debug_end = test[0][-1]
    # debug_init= 800
    # debug_end = 900
    debug_init = 0
    debug_end = np.size(wv_sampling)
    print(debug_init,debug_end)
    if psf_interp_list is None:
        psf_interp_list = []
        if 0 or mppool is None:
            for wv_id, wv in enumerate(wv_sampling):
                if not (wv_id >= debug_init and wv_id < debug_end):
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
            print("ra, dec",ra, dec)
            for wv_id, wv in enumerate(wv_sampling):
                # print(wv)
                if not (wv_id >= debug_init and wv_id < debug_end):
                    continue
                X = all_interp_ra[:, wv_id]
                Y = all_interp_dec[:, wv_id]
                Z = all_interp_flux[:, wv_id]
                Zerr = all_interp_err[:, wv_id]
                R = np.sqrt((X - ra) ** 2 + (Y - dec) ** 2)
                Zerr_masking = Zerr/median_abs_deviation(Zerr[np.where(np.isfinite(Zerr))])
                where_finite = np.where(np.isfinite(all_interp_badpix[:, wv_id])*(Zerr_masking<5e1) * np.isfinite(X) * np.isfinite(Y) * (R < aper_radius))
                # print(np.size(where_finite[0]),30*N_dithers)
                if np.size(where_finite[0])<30*N_dithers:
                    continue
                X = X[where_finite]
                Y = Y[where_finite]
                Z = Z[where_finite]
                # Zp=Zp[where_finite]
                Zerr = Zerr[where_finite]
                M = psf_interp_list[wv_id](X - ra, Y - dec)
                # print(wv,ra,dec)
                # # plt.scatter(X,Y,s=psf_interp_list[wv_id](X,Y)/np.nanmedian(psf_interp_list[wv_id](X,Y)))#,s=sampled_psf[:,wv_id]/np.nanmedian(sampled_psf[:,wv_id])
                # plt.figure(10)
                # plt.subplot(1,2,1)
                # plt.scatter(X,Y,s=M/np.nanmedian(M))#,s=sampled_psf[:,wv_id]/np.nanmedian(sampled_psf[:,wv_id])
                # plt.subplot(1,2,2)
                # plt.scatter(X,Y,s=Z/np.nanmedian(Z))
                #
                # plt.figure(11)
                #
                # # plt.scatter(np.sqrt(X**2+Y**2),M,label="M")
                # plt.subplot(1,3,1)
                # plt.scatter(np.sqrt(X**2+Y**2),Z,label="Z")
                # # plt.subplot(1,3,2)
                # # plt.scatter(np.sqrt(X**2+Y**2),Zp,label="Zp")
                # plt.subplot(1,3,3)
                # plt.scatter(np.sqrt(X**2+Y**2),Zerr,label="Zerr")
                # plt.legend()
                # plt.show()
                # # exit()


                deno = np.nansum(M ** 2 / Zerr ** 2)
                mfflux = np.nansum(M * Z / Zerr ** 2) / deno
                mffluxerr = 1 / np.sqrt(deno)

                res = Z - mfflux * M
                noise_factor = np.nanstd(res / Zerr)

                flux_cube[wv_id, dec_id, ra_id] = mfflux
                fluxerr_cube[wv_id, dec_id, ra_id] = mffluxerr*noise_factor

            # where_finite = np.where(np.isfinite(all_interp_badpix))
            # X = all_interp_ra[where_finite]
            # Y = all_interp_dec

            snr_vec = flux_cube[:, dec_id, ra_id] / fluxerr_cube[:, dec_id, ra_id]
            snr_vec = snr_vec - generic_filter(snr_vec, np.nanmedian, size=50)
            snr_vec = snr_vec / median_abs_deviation(snr_vec[np.where(np.isfinite(snr_vec))])
            where_outliers = np.where(snr_vec > 10)
            flux_cube[where_outliers[0], dec_id, ra_id] = np.nan
            fluxerr_cube[where_outliers[0], dec_id, ra_id] = np.nan

            deno = np.nansum(comp_spec** 2 / fluxerr_cube[:, dec_id, ra_id] ** 2)
            bbflux = np.nansum(comp_spec * flux_cube[:, dec_id, ra_id] / fluxerr_cube[:, dec_id, ra_id] ** 2) / deno
            bbfluxerr = 1 / np.sqrt(deno)

            res = flux_cube[:, dec_id, ra_id] - bbflux*comp_spec
            noise_factor = np.nanstd(res/fluxerr_cube[:, dec_id, ra_id])

            flux_map[dec_id, ra_id] = bbflux
            fluxerr_map[dec_id, ra_id] = bbfluxerr*noise_factor

    snr_map = flux_map / fluxerr_map
    if out_filename is not None:
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=flux_cube))
        hdulist.append(pyfits.ImageHDU(data=fluxerr_cube))
        hdulist.append(pyfits.ImageHDU(data=flux_map))
        hdulist.append(pyfits.ImageHDU(data=fluxerr_map))
        hdulist.append(pyfits.ImageHDU(data=snr_map))
        hdulist.append(pyfits.ImageHDU(data=ra_grid))
        hdulist.append(pyfits.ImageHDU(data=dec_grid))
        try:
            hdulist.writeto(out_filename, overwrite=True)
        except TypeError:
            hdulist.writeto(out_filename, clobber=True)
        hdulist.close()
    return (flux_cube,fluxerr_cube,snr_map, flux_map, fluxerr_map, ra_grid, dec_grid),psf_interp_list


def get_contnorm_spec(dataobj_list, out_filename=None, load_utils=True, mppool=None, spec_R_sampling=None):
    print(len(glob(out_filename)), out_filename)
    if 1 and load_utils and len(glob(out_filename)):
        with pyfits.open(out_filename) as hdulist:
            new_wavelengths = hdulist[0].data
            combined_fluxes = hdulist[1].data
            combined_errors = hdulist[2].data
    else:
        wvs_list = []
        normalized_im_list = []
        normalized_err_list = []
        for dataobj in dataobj_list:
            if 1 and load_utils and len(glob(dataobj.starspec_contnorm_filename)):
                with pyfits.open(dataobj.starspec_contnorm_filename) as hdulist:
                    spline_cont0 = hdulist[3].data
            else:
                spline_cont0, _, new_badpixs, new_res,_ = normalize_rows(dataobj.data, dataobj.wavelengths,
                                                                       noise=dataobj.noise, badpixs=dataobj.bad_pixels,
                                                                       nodes=dataobj.N_nodes, mypool=mppool,
                                                                       use_set_nans=False)

                spline_cont0[np.where(spline_cont0 / dataobj.noise < 5)] = np.nan
                spline_cont0 = copy(spline_cont0)
                spline_cont0[np.where(spline_cont0 < np.median(spline_cont0))] = np.nan
                spline_cont0[np.where(np.isnan(dataobj.bad_pixels))] = np.nan
            normalized_im = dataobj.data / spline_cont0
            normalized_err = dataobj.noise / spline_cont0
            wvs_list.extend(dataobj.wavelengths.flatten())
            normalized_im_list.extend(normalized_im.flatten())
            normalized_err_list.extend(normalized_err.flatten())
        if spec_R_sampling is None:
            spec_R_sampling = 4 * dataobj.R
        new_wavelengths, combined_fluxes, combined_errors = combine_spectrum(np.array(wvs_list),
                                                                             np.array(normalized_im_list),
                                                                             np.array(normalized_err_list),
                                                                             np.nanmedian(wvs_list) / (spec_R_sampling))
        #
        # plt.scatter(dataobj.wavelengths.flatten(), normalized_im.flatten())
        # plt.ylim([0, 2])
        # plt.show()

        if out_filename is not None:
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=new_wavelengths))
            hdulist.append(pyfits.ImageHDU(data=combined_fluxes))
            hdulist.append(pyfits.ImageHDU(data=combined_errors))
            try:
                hdulist.writeto(out_filename, overwrite=True)
            except TypeError:
                hdulist.writeto(out_filename, clobber=True)
            hdulist.close()
    return new_wavelengths, combined_fluxes, combined_errors
