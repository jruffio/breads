import numpy as np
import os
from glob import glob
import multiprocessing as mp
from  scipy.interpolate import interp1d

import astropy.io.fits as fits
import astropy.units as u
from astropy import constants as const
from breads.instruments.jwstnirspec_cal import JWSTNirspec_cal
from breads.instruments.jwstnirspec_cal import fitpsf

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Qt5Agg')



if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass
    wv_sampling_nrs1 = np.arange(2.859509, 4.1012874, 0.0006763935)
    wv_sampling_nrs2 = np.arange(4.081285,5.278689,0.0006656647)

    ####################################################################################################################
    ## To update:
    ################
    detector = "nrs1" # "nrs1" or "nrs2" (run once for both)

    numthreads = 32 # Number of threads for paralelization
    mypool = mp.Pool(processes=numthreads)

    os.environ["WEBBPSF_PATH"] = "/stow/jruffio/data/webbPSF/webbpsf-data"
    crds_dir="/stow/jruffio/data/JWST/crds_cache/"
    # utils directory where utility files will be saved
    utils_dir = "/stow/jruffio/data/JWST/nirspec/A0_TYC 4433-1800-1/breads/20231101_utils/"
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)
    # output directory
    out_dir = "/stow/jruffio/data/JWST/nirspecA0_TYC 4433-1800-1/breads/20231101_out/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if 1: # TYC 4433-1800-1 (A0 photometric calibrator)
        # Download files on MAST program id. 1128
        filelist = glob("/stow/jruffio/data/JWST/nirspec/A0_TYC 4433-1800-1/20230612_stage2_notelegraph/jw01128009001_03108_*_"+detector+"_cal.fits")
        if detector == "nrs1":
            coords_offset = [-0.25973700664819993, 0.7535417070247359] #offset of the coordinates because point cloud is not centered
        elif detector == "nrs2":
            coords_offset = [-0.2679950725373308, 0.7554649479920329]
        out_filename = os.path.join(out_dir, "TYC_4433-1800-1_spectrum_microns_MJy.png")

        # Download calspec spectrum fo star 1808347 (this is TYC_4433-1800-1) from
        # https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec
        stis_spectrum = "/stow/jruffio/models/TYC_4433-1800-1/1808347_stiswfc_004.fits"

    # if 0: # VHS1256
    #     filelist = glob("/stow/jruffio/data/JWST/nirspec/ERS_VHS1256/20230612_stage2_notelegraph/jw01386013001_03104_*_"+detector+"_cal.fits")
    #     if detector == "nrs1":
    #         coords_offset = [-0.626, -0.542] #offset of the coordinates because point cloud is not centered
    #     else:
    #         coords_offset = [-0.63, -0.55]
    #     out_filename = os.path.join(out_dir, "VHS1256b_spectrum_microns_MJy.png")
    ####################################################################################################################

    ####################################################################################################################
    ## Extract the spectrum

    if detector == "nrs1":
        wv_sampling = wv_sampling_nrs1
    elif detector == "nrs2":
        wv_sampling = wv_sampling_nrs2

    ## Look for science exoposures
    filelist.sort()
    for filename in filelist:
        print(filename)
    print("N files: {0}".format(len(filelist)))


    splitbasename = os.path.basename(filelist[0]).split("_")
    RDI_folder_suffix = "_webbpsf_v2" # This is called RDI even though you might not be doing RDI, but PSF fitting and RDI are the same thing
    fitpsf_filename = os.path.join(utils_dir,splitbasename[0]+"_"+splitbasename[1]+"_"+splitbasename[3]+"_fitpsf"+RDI_folder_suffix+".fits")

    if 1: # 1D extraction: run once for each detector
        # once ran, can be disabled with "if 0:" to plot next

        # Read and preprocess science exposures
        dataobj_list = []
        cen_list = []
        for filename in filelist:
            print(filename)
            dataobj = JWSTNirspec_cal(filename,crds_dir=crds_dir,utils_dir=utils_dir,save_utils=True, load_utils=True,mppool=mypool,
                                      regwvs_sampling = wv_sampling,coords_offset=None,
                                      load_interpdata_regwvs=True,
                                      wpsffit_IWA=0.0,wpsffit_OWA=0.5,
                                      mask_charge_bleeding=False,compute_wpsf=True,compute_starspec_contnorm=True,
                                      compute_starsub=True,compute_interp_regwvs=True,fit_wpsf=False,
                                      threshold_badpix=10,init_fit_psf=False)
            dataobj_list.append(dataobj)
            cen_list.append([dataobj.wpsf_ra_offset,dataobj.wpsf_dec_offset])

        with fits.open(dataobj.webbpsf_filename) as hdulist:
            wpsfs = hdulist[0].data
            wpsfs_header = hdulist[0].header
            wepsfs = hdulist[1].data #*(wpsfs_header["PIXELSCL"]/wpsfs_header["oversamp"])**2
            peak_webb_epsf = np.nanmax(wepsfs, axis=(1, 2))
            wepsfs = wepsfs / peak_webb_epsf[:, None, None]
            wepsfs = wepsfs * dataobj_list[0].aper_to_epsf_peak_f(wv_sampling)[:, None, None]
            webbpsf_wvs = hdulist[2].data
            webbpsf_X = hdulist[3].data
            webbpsf_Y = hdulist[4].data
            webbpsf_X = np.tile(webbpsf_X[None,:,:],(wepsfs.shape[0],1,1))
            webbpsf_Y = np.tile(webbpsf_Y[None,:,:],(wepsfs.shape[0],1,1))

        IWA = 0.0 # inner working angle of fitting region
        OWA = 0.5 # outer working angle of fitting region
        # if using sectors (mostly for RDI)
        ann_width = None #annulus width
        padding = 0.0 # padding of the sectors (I need to check how relevant this is now)
        sector_area = None # in arcsec^2, rough area of the sectors

        # The only requirement for the PSF point cloud is that the three arrays:
        # wepsfs,webbpsf_X,webbpsf_Y (respectively the normalized flux, dRA and dDEC coordinates)
        # are 2D array with:
        # 1st dim: wavelength following the sampling of "wv_sampling"
        # 2nd dim: any length, list of point (irregularly sampled)
        #
        # This means it can easily be replaced by a point cloud from a reference star instead of WebbPSF

        fitpsf(dataobj_list,wepsfs,webbpsf_X,webbpsf_Y, out_filename=fitpsf_filename,load=False,IWA = IWA,OWA = OWA,
               mppool=mypool,init_centroid=coords_offset,run_init=False,ann_width=ann_width,padding=padding,
               sector_area=sector_area,RDI_folder_suffix=RDI_folder_suffix,rotate_psf=dataobj_list[0].east2V2_deg,
               flipx=True,psf_spaxel_area=dataobj_list[0].webbpsf_spaxel_area)
        # exit()


    if 1: # plot spectrum
        hdulist = fits.open(stis_spectrum)
        from astropy.table import Table
        stis_table = Table(fits.getdata(stis_spectrum,1))
        stis_wvs =  (np.array(stis_table["WAVELENGTH"]) *u.Angstrom).to(u.um).value # angstroms -> mum
        stis_spec = np.array(stis_table["FLUX"]) * u.erg /u.s/u.cm**2/u.Angstrom # erg s-1 cm-2 A-1
        stis_spec = stis_spec.to(u.W*u.m**-2/u.um)
        stis_spec_Fnu = stis_spec*(stis_wvs*u.um)**2/const.c # from Flambda back to Fnu
        stis_spec_Fnu = stis_spec_Fnu.to(u.MJy).value

        color_list = ["#ff9900", "#006699", "#6600ff"]
        if detector == "nrs1":
            fitpsf_filename_nrs1 = fitpsf_filename
            fitpsf_filename_nrs2 = fitpsf_filename.replace("nrs1","nrs2")
        else:
            fitpsf_filename_nrs1 = fitpsf_filename.replace("nrs2","nrs1")
            fitpsf_filename_nrs2 = fitpsf_filename

        flux2save_wvs = []
        flux2save = []

        for det_id,(_detector,_fitpsf_filename,wv_sampling) in enumerate(zip(["nrs1","nrs2"],[fitpsf_filename_nrs1,fitpsf_filename_nrs2],[wv_sampling_nrs1,wv_sampling_nrs2])):
            if len(glob(_fitpsf_filename)) == 0:
                continue
            with fits.open(_fitpsf_filename) as hdulist:
                bestfit_coords = hdulist[0].data
                wpsf_angle_offset = hdulist[0].header["INIT_ANG"]
                wpsf_ra_offset = hdulist[0].header["INIT_RA"]
                wpsf_dec_offset = hdulist[0].header["INIT_DEC"]
                all_interp_psfsub = hdulist[1].data
                all_interp_psfmodel = hdulist[2].data

            print(bestfit_coords.shape)
            fontsize=12
            plt.figure(1,figsize=(12,10))
            gs = gridspec.GridSpec(5,1, height_ratios=[1,0.5,0.3,1,0.5], width_ratios=[1])
            gs.update(left=0.075, right=0.95, bottom=0.07, top=0.95, wspace=0.0, hspace=0.0)
            ax1 = plt.subplot(gs[2*det_id+det_id, 0])

            flux2save_wvs.extend(wv_sampling)
            flux2save.extend(bestfit_coords[0,:,0])

            plt.plot(wv_sampling,bestfit_coords[0,:,0]*1e9,linestyle="-",color=color_list[0],label="Fixed centroid",linewidth=1)
            plt.plot(wv_sampling,bestfit_coords[0,:,1]*1e9,linestyle="--",color=color_list[2],label="Free centroid",linewidth=1)
            plt.plot(wv_sampling,bestfit_coords[0,:,0]*1e9*np.polyval([-0.03128495, 1.09007397], wv_sampling),linestyle="-.",color=color_list[1],label="Fixed centroid - Corrected",linewidth=2)

            plt.plot(stis_wvs,stis_spec_Fnu*1e9,linestyle=":",color="black",label="CALSPEC",linewidth=2)
            plt.xlim([wv_sampling[0],wv_sampling[-1]])
            plt.ylabel("Flux density (mJy)",fontsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)
            plt.legend(loc="upper right")
            plt.text(0.01,0.01, '4 dithers - TYC 4433-1800-1', fontsize=fontsize, ha='left', va='bottom',color="black", transform=plt.gca().transAxes)
            plt.xticks([])

            ax1 = plt.subplot(gs[2*det_id+det_id+1, 0])
            calspec_func = interp1d(stis_wvs,stis_spec_Fnu*1e9)
            plt.plot(wv_sampling,bestfit_coords[0,:,0]*1e9*np.polyval([-0.03128495, 1.09007397], wv_sampling)-calspec_func(wv_sampling),linestyle="-.",color=color_list[1],label="Fixed centroid - Corrected",linewidth=1)
            plt.xlim([wv_sampling[0],wv_sampling[-1]])
            plt.xlabel("Wavelength ($\mu$m)",fontsize=fontsize)
            plt.ylabel("Difference (mJy)",fontsize=fontsize)
            plt.gca().tick_params(axis='x', labelsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)
            plt.legend(loc="upper right")

            plt.figure(2,figsize=(6,6))
            plt.subplot(2,1,1)
            plt.plot(wv_sampling,bestfit_coords[0,:,2],label=_detector)
            plt.xlabel("Wavelength ($\mu$m)",fontsize=fontsize)
            plt.ylabel("$\Delta$RA (as)",fontsize=fontsize)
            plt.gca().tick_params(axis='x', labelsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)
            plt.ylim([-0.280,-0.250])
            plt.legend(loc="upper right")
            plt.xlim([2.83,5.28])
            plt.text(0.01,0.01, '4 dithers - TYC 4433-1800-1', fontsize=fontsize, ha='left', va='bottom',color="black", transform=plt.gca().transAxes)# \n 36-166 $\mu$Jy
            plt.legend(loc="upper right")
            print("centroid",np.nanmedian(bestfit_coords[0,:,2]),np.nanmedian(bestfit_coords[0,:,3]))

            plt.subplot(2,1,2)
            plt.plot(wv_sampling,bestfit_coords[0,:,3])
            plt.xlabel("Wavelength ($\mu$m)",fontsize=fontsize)
            plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
            plt.gca().tick_params(axis='x', labelsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)
            plt.ylim([0.745,0.760])
            plt.xlim([2.83,5.28])

        plt.figure(1)
        out_filename = os.path.join(out_dir, "photocalib.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)
        plt.savefig(out_filename.replace(".png", ".pdf"),bbox_inches='tight')

        plt.figure(2)
        out_filename = os.path.join(out_dir, "centroid.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)
        plt.savefig(out_filename.replace(".png", ".pdf"),bbox_inches='tight')


        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(data=flux2save_wvs))
        hdulist.append(fits.ImageHDU(data=flux2save))
        try:
            hdulist.writeto(out_filename.replace(".png", ".fits"), overwrite=True)
        except TypeError:
            hdulist.writeto(out_filename.replace(".png", ".fits"), clobber=True)
        hdulist.close()

        with fits.open(out_filename.replace(".png", ".fits")) as hdulist:
            wvs = hdulist[0].data
            spec = hdulist[1].data
        plt.figure(3)
        plt.title("test reading spectrum fits file")
        plt.plot(wvs,spec)


        plt.show()

    exit()