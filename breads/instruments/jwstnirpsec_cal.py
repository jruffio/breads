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
import jwst.datamodels,jwst.assign_wcs
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.simbad import Simbad
from astropy.time import Time
from glob import glob
from scipy.stats import median_abs_deviation
import itertools
from  scipy.interpolate import interp1d
from astropy.stats import sigma_clip
from breads.utils import get_spline_model
from scipy.optimize import lsq_linear
import astropy
import webbpsf
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.optimize import curve_fit
from jwst.photom.photom import DataSet
from stdatamodels.jwst import datamodels
from scipy.signal import convolve2d

class jwstnirpsec_cal(Instrument):
    def __init__(self, filename=None, utils_dir = None, save_utils=True, load_utils=True,mppool=None):
        super().__init__('jwstnirpsec')
        if filename is None:
            warning_text = "No data file provided. " + \
            "Please manually add data or use jwstnirpsec.read_data_file()"
            warn(warning_text)
        else:
            self.read_data_file(filename, utils_dir = utils_dir, save_utils=save_utils, load_utils=load_utils,mppool=mppool)

    def read_data_file(self, filename, utils_dir = None, save_utils=True, load_utils=True,mppool=None,mask_charge_bleeding=True):
        """
        Read OSIRIS spectral cube, also checks validity at the end
        """
        if utils_dir is None:
            utils_dir = os.path.dirname(filename)
        with pyfits.open(filename) as hdulist:
            priheader = hdulist[0].header
            extheader = hdulist[1].header
            im = hdulist["SCI"].data
            err = hdulist["ERR"].data
            dq = hdulist["DQ"].data
            im_wvs = hdulist["WAVELENGTH"].data
        ny, nx = im.shape

        self.wavelengths = im_wvs
        self.data = im
        self.noise = err
        self.bad_pixels = np.ones((ny, nx))
        self.bad_pixels[np.where(untangle_dq(dq)[0,:,:])] = np.nan
        self.bad_pixels[np.where(np.isnan(self.data))] = np.nan

        # todo this is definitely an approximation, curvature of the trace? can be done better?
        self.delta_wavelengths = self.wavelengths[:,1::]-self.wavelengths[:,0:self.wavelengths.shape[1]-1]
        self.delta_wavelengths = np.concatenate([self.delta_wavelengths,self.delta_wavelengths[:,-1][:,None]],axis=1)


        self.priheader = priheader
        self.extheader = extheader

        self.bary_RV = 0#float(self.extheader["VELOSYS"])/1000 # in km/s
        self.R = 2700

        coords_filename = os.path.join(utils_dir,os.path.basename(filename).replace(".fits","_relcoords.fits"))
        if load_utils and len(glob(coords_filename)):
            with pyfits.open(coords_filename) as hdulist:
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
                                     pm_ra_cosdec=pm_ra*u.mas/u.year,
                                     pm_dec=pm_dec*u.mas/u.year,
                                     frame='icrs', obstime='J2000.0')
            desired_date = self.priheader["DATE-OBS"]#'2023-01-25'  # Example date in ISO format
            # Convert the desired date to an astropy Time object
            t = Time(desired_date)
            # Calculate the updated SkyCoord object for the desired date
            host_coord = HD19467_coord.apply_space_motion(new_obstime=t)
            host_ra_deg = host_coord.ra.deg
            host_dec_deg = host_coord.dec.deg

            calfile = jwst.datamodels.open(filename)
            photom_dataset = DataSet(calfile)
            #todo fix this, pick up the correct file based on headers
            area_fname = "/scr3/jruffio/data/JWST/crds_cache/references/jwst/nirspec/jwst_nirspec_area_0034.fits"
            # Load the pixel area table for the IFU slices
            area_model = datamodels.open(area_fname)
            area_data = area_model.area_table

            # Compute 2D wavelength and pixel area arrays for the whole image
            wave2d, area2d, dqmap = photom_dataset.calc_nrs_ifu_sens2d(area_data)
            area2d[np.where(area2d==1)] = np.nan

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
                xmin =  max(int(np.round(wcses[i].bounding_box.intervals[0][0])) , 0)
                xmax =  int(np.round(wcses[i].bounding_box.intervals[0][1]))
                ymin =  max(int(np.round(wcses[i].bounding_box.intervals[1][0])), 0)
                ymax =  int(np.round(wcses[i].bounding_box.intervals[1][1]))

                nx = xmax-xmin
                ny = ymax-ymin

                x = np.arange(xmin,xmax)
                x = x.reshape(1, x.shape[0]) * np.ones((ny,1))
                y = np.arange(ymin,ymax)
                y = y.reshape(y.shape[0], 1) * np.ones((1,nx))


                # Transform all those pixels to RA, Dec, wavelength
                skycoords, speccoord = wcses[i](x,y, with_units=True)


                ra_array[ymin:ymax, xmin:xmax] = skycoords.ra
                dec_array[ymin:ymax, xmin:xmax] = skycoords.dec
                wavelen_array[ymin:ymax, xmin:xmax] = speccoord

                # Transform all those pixels to the slicer plane
                slice_transform = wcses[i].get_transform('detector', 'slicer')

                sx, sy, sw = slice_transform(x, y)

                slicer_x_array[ymin:ymax, xmin:xmax] = sx
                slicer_y_array[ymin:ymax, xmin:xmax] = sy
                slicer_w_array[ymin:ymax, xmin:xmax] = sw

            dra_as_array = (ra_array-host_ra_deg)*3600
            ddec_as_array = (dec_array-host_dec_deg)*3600

            if save_utils:
                hdulist = pyfits.HDUList()
                hdulist.append(pyfits.PrimaryHDU(data=wavelen_array))
                hdulist.append(pyfits.ImageHDU(data=dra_as_array))
                hdulist.append(pyfits.ImageHDU(data=ddec_as_array))
                hdulist.append(pyfits.ImageHDU(data=area2d))
                try:
                    hdulist.writeto(coords_filename, overwrite=True)
                except TypeError:
                    hdulist.writeto(coords_filename, clobber=True)
                hdulist.close()

        self.dra_as_array,self.ddec_as_array,self.area2d = dra_as_array,ddec_as_array,area2d

        if mask_charge_bleeding:
            dist2host_as = np.sqrt((self.dra_as_array)**2+(self.ddec_as_array)**2)
            k,l = np.unravel_index(np.nanargmin(dist2host_as),dist2host_as.shape)
            self.bad_pixels[k-150:k+150,:] = np.nan

        starspec_contnorm_filename = os.path.join(utils_dir,os.path.basename(filename).replace(".fits","_starspec_contnorm.fits"))
        if load_utils and len(glob(starspec_contnorm_filename)):
            with pyfits.open(starspec_contnorm_filename) as hdulist:
                new_wavelengths = hdulist[0].data
                combined_fluxes = hdulist[1].data
                combined_errors = hdulist[2].data
        else:
            spline_cont,_,new_dq,new_res = normalize_rows(im, im_wvs,noise=err, dq=dq,mypool=mppool)

            spline_cont[np.where(spline_cont<np.median(spline_cont))] = np.nan
            normalized_im = im/spline_cont
            normalized_err = err/spline_cont

            # comb_spec = np.nansum(normalized_im/normalized_err**2,axis=(1,2))/np.nansum(1/normalized_err**2,axis=(1,2))
            # comb_err = 1/np.sqrt(np.nansum(1/normalized_err**2,axis=(1,2)))
            new_wavelengths, combined_fluxes, combined_errors = combine_spectrum(im_wvs.flatten(), normalized_im.flatten(), normalized_err.flatten(), 4./10000.)

            if save_utils:
                hdulist = pyfits.HDUList()
                hdulist.append(pyfits.PrimaryHDU(data=new_wavelengths))
                hdulist.append(pyfits.ImageHDU(data=combined_fluxes))
                hdulist.append(pyfits.ImageHDU(data=combined_errors))
                try:
                    hdulist.writeto(starspec_contnorm_filename, overwrite=True)
                except TypeError:
                    hdulist.writeto(starspec_contnorm_filename, clobber=True)
                hdulist.close()
        self.star_func = interp1d(new_wavelengths,combined_fluxes,bounds_error=False,fill_value=1)

        # webbpsf_filename = os.path.join(utils_dir,self.priheader["DATE-OBS"]+"_"+self.priheader["FILTER"]+"_"+self.priheader["GRATING"]+"_webbpsf.fits")
        splitbasename = os.path.basename(filename).split("_")
        webbpsf_filename = os.path.join(utils_dir,splitbasename[0]+"_"+splitbasename[1]+"_"+splitbasename[3]+"_webbpsf.fits")
        oversample=10
        if load_utils and len(glob(webbpsf_filename)):
            with pyfits.open(webbpsf_filename) as hdulist:
                webb_psf_ims = hdulist[0].data
                webb_epsf_ims = hdulist[1].data
                webbpsf_wvs = hdulist[2].data
                webbpsf_X = hdulist[3].data
                webbpsf_Y = hdulist[4].data
                webbpsf_pixelscale = hdulist[3].header["PIXELSCL"]
                # webbpsf_pixelscale = 0.01111

                # hdulist2 = pyfits.HDUList()
                # hdulist2.append(pyfits.PrimaryHDU(data=webb_psf_ims))
                # hdulist2.append(pyfits.PrimaryHDU(data=webb_epsf_ims*100))
                # hdulist2.append(pyfits.ImageHDU(data=webbpsf_wvs))
                # hdulist2.append(pyfits.ImageHDU(data=webbpsf_X))
                # hdulist2.append(pyfits.ImageHDU(data=webbpsf_Y))
                # try:
                #     hdulist2.writeto(webbpsf_filename.replace(".fits","2.fits"), overwrite=True)
                # except TypeError:
                #     hdulist2.writeto(webbpsf_filename.replace(".fits","2.fits"), clobber=True)
                # hdulist2.close()
                # exit()
        else:
            wv_min,wv_max = np.nanmin(self.wavelengths), np.nanmax(self.wavelengths)
            webbpsf_wvs = np.arange(wv_min-0.05,wv_max+0.1,0.05)
            kernel = np.ones((oversample, oversample))
            # kernel /= kernel.sum()
            webb_psf_ims = []
            webb_epsf_ims = []
            for webbpsf_wv0 in webbpsf_wvs:
                nrs = webbpsf.NIRSpec()
                nrs.load_wss_opd_by_date(priheader["DATE-BEG"]) # Load telescope state as of our observation date
                nrs.image_mask = 'IFU' # optional: model opaque field stop outside of the IFU aperture
                nrs.pixelscale = 0.1   # Optional: set this manually to match the drizzled cube sampling, rather than the default
                ext='OVERSAMP'
                slicepsf_wv0 = nrs.calc_psf(monochromatic=webbpsf_wv0*1e-6,   # Wavelength, in **METERS**
                                            fov_arcsec=3,           # angular size to simulate PSF over
                                            oversample=oversample,           # output pixel scale relative to the pixelscale set above
                                            add_distortion=False)   # skip an extra computation step that's not relevant for IFU
                im = slicepsf_wv0[ext].data
                if 0:
                    pixelscale = slicepsf_wv0[ext].header['PIXELSCL']
                    print(pixelscale)
                    psf_array_shape = slicepsf_wv0[ext].data.shape
                    halffov_x = pixelscale * psf_array_shape[1] / 2.0
                    halffov_y = pixelscale * psf_array_shape[0] / 2.0
                    x = np.linspace(-halffov_x, halffov_x, psf_array_shape[1],endpoint=True)
                    y = np.linspace(-halffov_y, halffov_y, psf_array_shape[0],endpoint=True)
                    print(halffov_x,halffov_y)
                    print(x,y)
                    exit()
                smoothed_im = convolve2d(im, kernel, mode='same')
                # plt.plot(im[270//2,:])
                # plt.plot(smoothed_im[270//2,:])
                # plt.show()
                webb_psf_ims.append(im)
                webb_epsf_ims.append(smoothed_im)
            webb_psf_ims = np.array(webb_psf_ims)
            webb_epsf_ims = np.array(webb_epsf_ims)

            psf_array_shape = slicepsf_wv0[ext].data.shape
            webbpsf_pixelscale = slicepsf_wv0[ext].header['PIXELSCL']

            # print(psf_array_shape,pixelscale)
            halffov_x = webbpsf_pixelscale * psf_array_shape[1] / 2.0
            halffov_y = webbpsf_pixelscale * psf_array_shape[0] / 2.0
            x = np.linspace(-halffov_x, halffov_x, psf_array_shape[1],endpoint=True)
            y = np.linspace(-halffov_y, halffov_y, psf_array_shape[0],endpoint=True)
            webbpsf_X, webbpsf_Y = np.meshgrid(x, y)

            if save_utils:
                hdulist = pyfits.HDUList()
                hdulist.append(pyfits.PrimaryHDU(data=webb_psf_ims))
                hdulist.append(pyfits.PrimaryHDU(data=webb_epsf_ims))
                hdulist.append(pyfits.ImageHDU(data=webbpsf_wvs))
                hdulist.append(pyfits.ImageHDU(data=webbpsf_X,
                                               header=pyfits.Header(cards={"PIXELSCL": webbpsf_pixelscale})))
                hdulist.append(pyfits.ImageHDU(data=webbpsf_Y,
                                               header=pyfits.Header(cards={"PIXELSCL": webbpsf_pixelscale})))
                try:
                    hdulist.writeto(webbpsf_filename, overwrite=True)
                except TypeError:
                    hdulist.writeto(webbpsf_filename, clobber=True)
                hdulist.close()
                # exit()

        webb_epsf_ims = webb_epsf_ims*webbpsf_pixelscale**2
        webbpsf_R = np.sqrt(webbpsf_X**2+webbpsf_Y**2)
        mask_aper = np.ones(webbpsf_R.shape)
        mask_aper[np.where(webbpsf_R>1.5)] = np.nan
        tiled_mask_aper = np.tile(mask_aper[None,:,:],(webb_psf_ims.shape[0],1,1))
        aper_phot_webb_psf = np.nansum(webb_psf_ims*tiled_mask_aper,axis=(1,2))*webbpsf_pixelscale**2
        # aper_phot_webb_epsf = np.nansum((webb_epsf_ims*tiled_mask_aper)[:,::10,::10],axis=(1,2))*pix_area
        # aper_phot_webb_epsf2 = np.nansum((webb_epsf_ims*tiled_mask_aper)[:,5::10,5::10],axis=(1,2))*pix_area
        peak_webb_epsf = np.nanmax(webb_epsf_ims,axis=(1,2))
        # plt.plot(aper_phot_webb_psf,label="aper_phot_webb_psf")
        # plt.plot(aper_phot_webb_epsf,label="aper_phot_webb_epsf")
        # plt.plot(aper_phot_webb_epsf2,label="aper_phot_webb_epsf2")
        # plt.legend()
        # plt.plot(webbpsf_wvs,peak_webb_epsf/aper_phot_webb_psf)
        # plt.show()
        self.aper_to_epsf_peak_f = interp1d(webbpsf_wvs,peak_webb_epsf/aper_phot_webb_psf,bounds_error=False,fill_value=np.nan)

        interp_webbpsf = RegularGridInterpolator((webbpsf_wvs,),webb_epsf_ims/peak_webb_epsf[:,None,None],method="linear",bounds_error=False,fill_value=0.0)
        # print(webbpsf_wvs[10])
        # print(myinterpgrid([webbpsf_wvs[10],])[0].shape)
        # plt.figure(1)
        # plt.subplot(1,2,1)
        # plt.imshow((webb_epsf_ims/peak_webb_epsf[:,None,None])[10,:,:])
        # plt.subplot(1,2,2)
        # plt.imshow(interp_webbpsf([webbpsf_wvs[10],])[0])
        #
        # plt.figure(2)
        # plt.plot(webbpsf_X[300//2,:],webb_psf_ims[10,300//2,:])
        # plt.plot(webbpsf_X[300//2,:],(webb_epsf_ims/peak_webb_epsf[:,None,None])[10,300//2,:])
        # plt.plot(webbpsf_X[300//2,:],interp_webbpsf([webbpsf_wvs[10],])[0][300//2,:],"--")
        # plt.show()
        # exit()

        psf_wv0_id = np.size(webbpsf_wvs)//2
        self.webbpsf_im = webb_epsf_ims[psf_wv0_id]/peak_webb_epsf[psf_wv0_id]
        self.webbpsf_X = webbpsf_X
        self.webbpsf_Y = webbpsf_Y
        self.webbpsf_spaxel_area = (webbpsf_pixelscale*oversample)**2
        self.webbpsf_wv0 = webbpsf_wvs[psf_wv0_id]
        self.webbpsf_interp = CloughTocher2DInterpolator((self.webbpsf_X.flatten(), self.webbpsf_Y.flatten()), self.webbpsf_im.flatten(),fill_value=0.0)


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



        webbpsffit_filename = os.path.join(utils_dir,os.path.basename(filename).replace(".fits","_webbpsffit.fits"))
        if load_utils and len(glob(webbpsffit_filename)):
            with pyfits.open(webbpsffit_filename) as hdulist:
                bestfit_paras = hdulist[0].data
                psf_model = hdulist[1].data
                psfsub_im = hdulist[2].data
                bestfit_paras2 = hdulist[3].data
                psf_model2 = hdulist[4].data
                psfsub_im2 = hdulist[5].data

                # _wvs = bestfit_paras[0,:]
                # xcs = bestfit_paras[1,:]
                # ycs = bestfit_paras[2,:]
                # stellar_spec = bestfit_paras[3,:]
                # plt.subplot(3,1,1)
                # plt.plot(_wvs,xcs)
                # plt.xlabel("Wavelength (um)")
                # plt.ylabel("x position (as)")
                # plt.subplot(3,1,2)
                # plt.plot(_wvs,ycs)
                # plt.xlabel("Wavelength (um)")
                # plt.ylabel("y position (as)")
                # plt.subplot(3,1,3)
                # plt.plot(_wvs,stellar_spec/self.aper_to_epsf_peak_f(_wvs))
                # plt.tight_layout()
                # plt.show()

                # photfilter = "/scr3/jruffio/data/JWST/nirspec/HD_19467/breads/utils/JWST_NIRCam."+"F460M"+".dat"
                # filter_arr = np.loadtxt(photfilter)
                # trans_wvs = filter_arr[:,0]/1e4
                # trans = filter_arr[:,1]
                # _dwvs = _wvs[1::]-_wvs[0:np.size(_wvs)-1]
                # _dwvs = np.append(_dwvs,_dwvs[-1])
                # # plt.plot(trans_wvs,trans)
                # # plt.show()
                # photfilter_f = interp1d(trans_wvs,trans,bounds_error=False,fill_value=0)
                # # plt.plot(_wvs,stellar_spec)
                # # plt.show()
                # c = (2.998e8 *u.m/u.s)
                # test = photfilter_f(_wvs)*(stellar_spec/self.aper_to_epsf_peak_f(_wvs))*1e6*u.Jy*c/(_wvs*u.um)**2
                # filter_norm = np.nansum(_dwvs*u.um*photfilter_f(_wvs))
                # plt.plot(test.to(u.W/u.m**2/u.um))
                # Fl = np.nansum(_dwvs*u.um*test.to(u.W/u.m**2/u.um))/filter_norm # u.W*u.m**-2/u.um
                # print(Fl,(Fl*(4.6*u.um)).to(u.W*u.m**-2))
                # Fnu = Fl*(4.6*u.um)**2/c
                # print(Fnu)
                # print(Fnu.to(u.W*u.m**-2/u.Hz))
                # print(Fnu.to(u.Jy))
                # plt.show()
        else:
            bestfit_paras,psf_model,psfsub_im = fit_webbpsf(self.data, self.wavelengths,self.noise, self.bad_pixels,self.dra_as_array,self.ddec_as_array,
                                                       self.webbpsf_interp,self.webbpsf_wv0,fix_cen = None)
            bestfit_paras2,psf_model2,psfsub_im2 = fit_webbpsf(self.data, self.wavelengths,self.noise, self.bad_pixels,self.dra_as_array,self.ddec_as_array,
                                                            self.webbpsf_interp,self.webbpsf_wv0,fix_cen=(np.nanmedian(bestfit_paras[1,:]),np.nanmedian(bestfit_paras[2,:])))
            # bestfit_paras2[1:3,:]=bestfit_paras[1:3,:]
            if save_utils:
                hdulist = pyfits.HDUList()
                hdulist.append(pyfits.PrimaryHDU(data=bestfit_paras))
                hdulist.append(pyfits.ImageHDU(data=psf_model))
                hdulist.append(pyfits.ImageHDU(data=psfsub_im))
                hdulist.append(pyfits.PrimaryHDU(data=bestfit_paras2))
                hdulist.append(pyfits.ImageHDU(data=psf_model2))
                hdulist.append(pyfits.ImageHDU(data=psfsub_im2))
                try:
                    hdulist.writeto(webbpsffit_filename, overwrite=True)
                except TypeError:
                    hdulist.writeto(webbpsffit_filename, clobber=True)
                hdulist.close()

                # hdulist_sc["SCI"].data = psfsub_sc_im
                # hdulist_sc["DQ"].data = psfsub_sc_dq
                # # Write the new HDU list to a new FITS file
                # out_filename = os.path.join(star_sub_folder,os.path.basename(sc_filename))
                # try:
                #     hdulist_sc.writeto(out_filename, overwrite=True)
                # except TypeError:
                #     hdulist_sc.writeto(out_filename, clobber=True)
                # hdulist_sc.close()

        self.dra_as_array,self.ddec_as_array = self.dra_as_array - np.nanmedian(bestfit_paras[1,:]),self.ddec_as_array - np.nanmedian(bestfit_paras[2,:])
        # print(np.nanmedian(bestfit_paras[3,:]))
        # exit()


        self.valid_data_check()

    def broaden(self, wvs,spectrum, loc=None,mppool=None):
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
    cube[:,:,:] = bits.transpose(2, 0, 1)
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
        arr_copy[i, :first_real_idx[i]+n] = np.nan
        arr_copy[i, last_real_idx[i]-n+1:] = np.nan

    return arr_copy

def _task_normrows(paras):
    im_rows,im_wvs_rows,noise_rows,badpix_rows,x_knots,star_model,chunks,threshold,star_sub_mode = paras


    new_im_rows = np.array(copy(im_rows), '<f4')#.byteswap().newbyteorder()
    new_noise_rows = copy(noise_rows)
    new_badpix_rows = copy(badpix_rows)
    res = np.zeros(im_rows.shape) + np.nan
    for k in range(im_rows.shape[0]):
        # print(k)
        # if k == 605:
        #     print("coucou")
        # else:
        #     continue
        M_spline = get_spline_model(x_knots,im_wvs_rows[k,:],spline_degree=3)

        # plt.subplot(2,1,1)
        # plt.plot(im_rows[k,:])
        # plt.plot(noise_rows[k,:])
        # plt.subplot(2,1,2)
        # plt.plot(np.isfinite(med_spec),label="np.isfinite(med_spec)")
        # plt.plot(noise_rows[k,:]==0,label="noise_rows[k,:]==0")
        # plt.plot(np.isfinite(im_rows[k,:]),label="np.isfinite(im_rows[k,:])")
        # plt.plot(np.isfinite(noise_rows[k,:]),label="np.isfinite(noise_rows[k,:])")
        # plt.legend()
        # plt.show()

        where_data_finite = np.where(np.isfinite(im_rows[k,:])*np.isfinite(noise_rows[k,:])*np.isfinite(star_model[k,:]))
        # if k == 1512:
        #     print("coucou")
        # else:
        #     res[:,k] = np.nan
        #     continue
        # print(k,np.size(where_data_finite[0]))
        if np.size(where_data_finite[0]) == 0:
            res[k,:] = np.nan
            continue
        d = im_rows[k,where_data_finite[0]]
        d_err = noise_rows[k,where_data_finite[0]]

        M = M_spline[where_data_finite[0],:]*star_model[k,where_data_finite[0],None]

        validpara = np.where(np.nansum(M>np.nanmax(M)*1e-6,axis=0)!=0)
        M = M[:,validpara[0]]

        # bounds_min = [0, ]* M.shape[1]
        bounds_min = [-np.inf, ]* M.shape[1]
        bounds_max = [np.inf, ] * M.shape[1]
        p = lsq_linear(M/d_err[:,None],d/d_err,bounds=(bounds_min, bounds_max)).x
        # p,chi2,rank,s = np.linalg.lstsq(M/d_err[:,None],d/d_err,rcond=None)
        m = np.dot(M,p)
        res[k,where_data_finite[0]] = d-m
        new_im_rows[k,where_data_finite[0]] = m
        new_noise_rows[k,where_data_finite[0]] = d_err


        # where_bad = np.where((np.abs(res[:,k])>3*np.nanstd(res[:,k])) | np.isnan(res[:,k]))
        meddev=median_abs_deviation(res[k,where_data_finite[0]])
        where_bad = np.where((np.abs(res[k,:])>threshold*meddev) | np.isnan(res[k,:]))
        new_badpix_rows[k,where_bad[0]] = 1
        # where_bad = np.where(np.isnan(np.correlate(new_badpix_rows[k,:] ,np.ones(2),mode="same")))
        # new_badpix_rows[k,where_bad[0]] = np.nan
        # new_badpix_rows[k,np.where(np.isnan(new_badpix_rows[k,:]))[0]] = 1

        if 0:
            plt.figure(2)
            plt.plot(new_im_rows[k,:],label="d")
            # plt.plot(new_noise_rows[k,:],label="n")
            plt.legend()


            plt.figure(1)
            plt.subplot(2,1,1)
            plt.plot(d,label="d")
            # m0 = med_spec[where_data_finite[0],None]
            # plt.plot(m0/np.nansum(m0)*np.nansum(d),label="m0")
            plt.plot(m,label="m")
            plt.plot(d_err,label="err")
            plt.plot(d-m,label="res")
            plt.plot(d/d*threshold*meddev,label="threshold")
            plt.legend()

            plt.subplot(2,1,2)
            for l in range(M.shape[1]):
                plt.plot(M[:,l])
            plt.show()


            # plt.plot(new_data_arr[where_data_finite[0],k],label="new d",linestyle="--")
            # plt.legend()
            # plt.figure(2)
            # plt.plot(new_badpix_arr[where_data_finite[0],k],label="bad pix",linestyle="-")
            # plt.show()


    return new_im_rows,new_noise_rows,new_badpix_rows,res

def normalize_rows(image, im_wvs,noise=None, dq=None,star_model=None,chunks=40,mypool=None,nan_mask_boxsize=3,threshold=100,star_sub_mode = False):
    if noise is None:
        noise = np.ones(image.shape)
    if dq is None:
        dq = np.zeros(image.shape)
    if star_model is None:
        star_model = np.ones(image.shape)

    x_knots = np.linspace(np.nanmin(im_wvs),np.nanmax(im_wvs),chunks+1,endpoint=True)

    new_image = copy(image)
    new_image = set_nans(image, 40)
    new_noise = copy(noise)
    new_dq = copy(dq)
    new_res = np.zeros(image.shape) + np.nan


    if mypool is None:
        paras = new_image,im_wvs,new_noise,new_dq,x_knots,star_model,chunks,threshold,star_sub_mode
        outputs = _task_normrows(paras)
        new_image,new_noise,new_dq,new_res = outputs
    else:
        numthreads = mypool._processes
        chunk_size = image.shape[0]//(3*numthreads)
        N_chunks = image.shape[0]//chunk_size
        row_ids = np.arange(image.shape[0])


        row_indices_list = []
        image_list = []
        wvs_list = []
        noise_list = []
        dq_list = []
        starmodel_list = []
        for k in range(N_chunks-1):
            _row_valid_pix = row_ids[(k*chunk_size):((k+1)*chunk_size)]
            row_indices_list.append(_row_valid_pix)

            _new_image = new_image[(k*chunk_size):((k+1)*chunk_size),:]
            _im_wvs = im_wvs[(k*chunk_size):((k+1)*chunk_size),:]
            _new_noise = new_noise[(k*chunk_size):((k+1)*chunk_size),:]
            _new_dq = new_dq[(k*chunk_size):((k+1)*chunk_size),:]
            _star_model = star_model[(k*chunk_size):((k+1)*chunk_size),:]

            image_list.append(_new_image)
            wvs_list.append(_im_wvs)
            noise_list.append(_new_noise)
            dq_list.append(_new_dq)
            starmodel_list.append(_star_model)

        _row_valid_pix = row_ids[((N_chunks-1)*chunk_size):image.shape[0]]
        row_indices_list.append(_row_valid_pix)

        _new_image = new_image[((N_chunks-1)*chunk_size):image.shape[0],:]
        _im_wvs = im_wvs[((N_chunks-1)*chunk_size):image.shape[0],:]
        _new_noise = new_noise[((N_chunks-1)*chunk_size):image.shape[0],:]
        _new_dq = new_dq[((N_chunks-1)*chunk_size):image.shape[0],:]
        _star_model = star_model[((N_chunks-1)*chunk_size):image.shape[0],:]

        image_list.append(_new_image)
        wvs_list.append(_im_wvs)
        noise_list.append(_new_noise)
        dq_list.append(_new_dq)
        starmodel_list.append(_star_model)


        # paras = new_image,im_wvs,new_noise,new_dq,x_knots,med_spec,chunks,threshold
        outputs_list = mypool.map(_task_normrows, zip(image_list,wvs_list,noise_list,dq_list,
                                                      itertools.repeat(x_knots),
                                                      starmodel_list,
                                                      itertools.repeat(chunks),
                                                      itertools.repeat(threshold),
                                                      itertools.repeat(star_sub_mode)))
        for row_indices,outputs in zip(row_indices_list,outputs_list):
            out_im_rows,out_noise_rows,out_dq_rows,out_res = outputs
            new_image[row_indices,:] = out_im_rows
            new_noise[row_indices,:] = out_noise_rows
            new_dq[row_indices,:] = out_dq_rows
            new_res[row_indices,:] = out_res

    return new_image,new_noise,new_dq,new_res


def fit_webbpsf(sc_im, sc_im_wvs,noise, bad_pixels,dra_as_array,ddec_as_array,interpolator,psf_wv0,fix_cen=None ):
    wv_min,wv_max = np.nanmin(sc_im_wvs), np.nanmax(sc_im_wvs)
    wv_sampling = np.exp(np.arange(np.log(wv_min),np.log(wv_max),np.log(1+0.5/2700.)))

    dist2host_as = np.sqrt(dra_as_array**2+ddec_as_array**2)

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

    psfsub_sc_im = np.zeros(sc_im.shape)+np.nan
    psfsub_model_im = np.zeros(sc_im.shape)
    bestfit_paras = np.zeros((4,np.size(wv_sampling)))+np.nan
    for wv_id, left_wv in enumerate(wv_sampling):
        # if left_wv<4.5:
        #     continue
        center_wv = left_wv*(1+0.25/2700)
        right_wv = left_wv*(1+0.5/2700)
        print(left_wv,center_wv,right_wv,wv_min,wv_max)

        #sc_im, sc_im_wvs,noise, sc_dq
        where_fit_mask = np.where(np.isfinite(sc_im)*(noise!=0)*(np.isfinite(bad_pixels))*(left_wv<sc_im_wvs)*(sc_im_wvs<right_wv)*(dist2host_as<1.0))#*(dist2host_as>0.5)
        where_sc_mask = np.where(np.isfinite(sc_im)*(noise!=0)*(left_wv<sc_im_wvs)*(sc_im_wvs<right_wv))
        Xfit = dra_as_array[where_fit_mask]
        Yfit = ddec_as_array[where_fit_mask]
        Zfit = sc_im[where_fit_mask]
        Zerr2_fit =(noise[where_fit_mask])**2

        Xsc = dra_as_array[where_sc_mask]
        Ysc = ddec_as_array[where_sc_mask]
        Zsc = sc_im[where_sc_mask]

        if (np.size(where_fit_mask[0]) < 377/2) or (np.size(where_sc_mask[0]) < 736/2) :
            print("Not enough points",wv_id,center_wv,np.size(where_fit_mask[0]),np.size(where_sc_mask[0]))
            bestfit_paras[:,wv_id] = np.array([center_wv,np.nan,np.nan,np.nan])
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
            m0 = interpolator(Xfit*psf_wv0/center_wv, Yfit*psf_wv0/center_wv)
            a0 = np.nansum(Zfit*m0/Zerr2_fit)/np.nansum(m0**2/Zerr2_fit)
            # Define the function to fit
            def myfunc(coords, xc, yc, A):
                _x, _y = coords[0],coords[1]
                znew = A*interpolator(_x - xc,_y - yc)
                return znew
            # Define the initial parameter values for the fit
            p0 = [0,0,a0]
            # Fit the data to the function
            try:
                params, _ = curve_fit(myfunc, np.array([Xfit*psf_wv0/center_wv,Yfit*psf_wv0/center_wv]), Zfit, p0=p0, method='lm', ftol=1e-6, xtol=1e-6)
            except:
                print("curve_fit failed",wv_id,center_wv,np.size(where_fit_mask[0]),np.size(where_sc_mask[0]))
                bestfit_paras[:,wv_id] = np.array([center_wv,np.nan,np.nan,np.nan])
                psfsub_model_im[where_sc_mask] = np.nan
                psfsub_sc_im[where_sc_mask] = np.nan
                continue
            # Extract the optimized parameter values
            xc, yc, a = params

        else:
            m0 = interpolator((Xfit-fix_cen[0])*psf_wv0/center_wv, (Yfit-fix_cen[1])*psf_wv0/center_wv)
            a0 = np.nansum(Zfit*m0/Zerr2_fit)/np.nansum(m0**2/Zerr2_fit)
            xc, yc, a =  0,0,a0
        # print(xc, yc, a)
        psfmodel = a * interpolator(Xsc*psf_wv0/center_wv - xc,Ysc*psf_wv0/center_wv - yc)
        psfsub_Zsc = Zsc-psfmodel

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

        bestfit_paras[:,wv_id] = np.array([center_wv,xc*center_wv/psf_wv0,yc*center_wv/psf_wv0,a * interpolator(0,0)])
        psfsub_model_im[where_sc_mask] = psfmodel
        psfsub_sc_im[where_sc_mask] = psfsub_Zsc

    return bestfit_paras,psfsub_model_im,psfsub_sc_im

import numpy as np

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
    wavelengths = wavelengths[~nan_mask]
    fluxes = fluxes[~nan_mask]
    errors = errors[~nan_mask]

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
        tmp_snr = (bin_fluxes-np.nanmedian(bin_fluxes))/bin_errors
        mask = sigma_clip(tmp_snr,3,masked=True).mask
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