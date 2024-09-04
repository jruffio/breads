
import os
import time
from glob import glob
from copy import copy

import numpy as np
from scipy.stats import median_abs_deviation
from astropy.io import fits

import jwst
from jwst.pipeline import Detector1Pipeline, Spec2Pipeline

from breads.fit import fitfm
import datetime

###########################################################################
#                       JWST reduction tools
#
# This module contains utility functions for JWST reductions, particularly 
# for invoking the JWST pipeline with some customizations and additions for
# tuned for the kind of processing we want to do with breads. 
#
# Top-level function is "run_complete_stage1_2_clean_reduction"
# This invokes lower-level functions for:
#   run_stage1
#   run_noise_clean
#   run_stage2

def find_files_to_process(input_dir, filetype='uncal.fits'):
    """ Utility function to find files of a given type """

    if "jw0" in filetype:
        files = glob(os.path.join(input_dir,filetype))
    else:
        files = glob(os.path.join(input_dir,"jw0*_"+filetype))
    files.sort()
    for file in files:
        print(file)
    print('Found ' + str(len(files)) + ' input files to process')
    return files

###########################################################################
# Functions for invoking the pipeline

def run_stage1(uncal_files, output_dir, overwrite=False,maximum_cores="all"):
    """ Run pipeline stage 1, with some customizations for reductions
    intended to be used with breads for IFU high contrast

    Currently only tested on NIRSpec IFU data
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    time0 = time.perf_counter()
    print(time0)

    rate_files = []

    for i, file in enumerate(uncal_files):
        print(f"Processing file {i+1} of {len(uncal_files)}.")

        outname = os.path.join(output_dir, os.path.basename(file).replace('uncal.fits', 'rate.fits'))
        rate_files.append(outname)

        #print(os.path.join(output_dir, outname))
        if os.path.exists(outname) and not overwrite:
            print(f"Output file {outname} already exists in output dir;\n\tskipping {file}.")
            continue

        det1 = Detector1Pipeline() # Instantiate the pipeline

        #defining used pipeline steps
        # This version only shows the step parameters which are changes from defaults.
        step_parameters = {
            # group_scale - run with defaults
            # dq_init - run with defaults
            'saturation': {'n_pix_grow_sat': 0},    # check for saturated pixels, but do not expand to adjacent pixels
            # ipc - run with defaults
            # superbias - run with defaults
            # linearity - run with defaults
            'persistence': {'skip' : True},         # This step does nothing; there are no nonzero parameters in the reference files yet
            # dark_current : run with defaults
            'jump': {'maximum_cores': maximum_cores},       # parallelize
            'ramp_fit': {'maximum_cores': maximum_cores},   # parallelize
            # gain_scale : run with defaults
        }

        det1.call(file, save_results=True, output_dir = output_dir,
                  steps=step_parameters)


        # Print out the time benchmark
        time1 = time.perf_counter()
        print(f"Runtime so far: {time1 - time0:0.4f} seconds")

    time1 = time.perf_counter()
    print(f"Total Runtime: {time1 - time0:0.4f} seconds")
    return rate_files



def run_stage2(rate_files, output_dir, skip_cubes=True, overwrite=False):
    """ Run pipeline stage 2, with some customizations for reductions
    intended to be used with breads for IFU high contrast

    Currently only tested on NIRSpec IFU data

    """


    # We need to check that the desired output directories exist, and if not create them
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

        # Start a timer to keep track of runtime
    time0 = time.perf_counter()
    print(time0)

    cal_files = []

    for fid,rate_file in enumerate(rate_files):
        print(fid,rate_file)

        # Setting up steps and running the Spec2 portion of the pipeline.

        outname = os.path.join(output_dir, os.path.basename(rate_file).replace('rate.fits', 'cal.fits'))
        cal_files.append(outname)
        if os.path.exists(outname) and not overwrite:
            print(f"Output file {outname} already exists;\n\tskipping {rate_file}.")
            continue

        spec2 = Spec2Pipeline()
        #spec2.output_dir = spec2_dir

        step_parameters = {
            # spec2.assign_wcs.skip = False
            # spec2.bkg_subtract.skip = False
            # spec2.imprint_subtract.skip = False
            # spec2.msa_flagging.skip = False
            # # spec2.srctype.source_type = 'POINT'
            # spec2.flat_field.skip = False
            # spec2.pathloss.skip = False
            # spec2.photom.skip = False
            'cube_build' : {'skip' : skip_cubes},   # We do not want or need interpolated cubes
            'extract_1d' : {'skip' : True},
            # spec3.cube_build.coord_system = 'skyalign'
            # spec2.cube_build.coord_system='ifualign'
        }
        spec2.save_bsub = True

        #choose what results to save and from what steps
        #spec2.save_results = True
        spec2.call(rate_file, save_results=True, output_dir = output_dir,
                   steps = step_parameters)

        # Print out the time benchmark
        time1 = time.perf_counter()
        print(f"Runtime so far: {time1 - time0:0.4f} seconds")

    time1 = time.perf_counter()
    print(f"Total Runtime: {time1 - time0:0.4f} seconds")
    return cal_files

###########################################################################
#  Function for centroid calibration

import multiprocessing as mp
from breads.instruments.jwstnirspec_cal import JWSTNirspec_cal
from breads.instruments.jwstnirspec_multiple_cals import JWSTNirspec_multiple_cals
from breads.instruments.jwstnirspec_cal import fitpsf
import matplotlib.gridspec as gridspec # GRIDSPEC !
import matplotlib.pyplot as plt

# # Definition of the wavelength sampling on which the detector images are interpolated (for each detector)
# wv_for_cent_calib_dict = {}
# #"G140H","G235H","G395H"
# wv_for_cent_calib_dict["G140H nrs1"] = np.arange(0.96646905,1.4494654,0.003)
# wv_for_cent_calib_dict["G140H nrs2"] = []
# wv_for_cent_calib_dict["G235H nrs1"] = []#0.005
# wv_for_cent_calib_dict["G235H nrs2"] = []
# wv_for_cent_calib_dict["G395H nrs1"] = np.arange(2.859, 4.103, 0.01)
# wv_for_cent_calib_dict["G395H nrs2"] = np.arange(4.081, 5.280, 0.01)

def run_coordinate_recenter(cal_files, utils_dir,crds_dir, init_centroid = (0,0),wv_sampling=None, N_wvs_nodes=40,
                             mask_charge_transfer_radius = None,
                             IWA=0.3,OWA=1.0,
                             debug_init=None,debug_end=None,
                             numthreads = 16,
                             save_plots=False):

    mypool = mp.Pool(processes=numthreads)

    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)

    # Science data: List of stage 2 cal.fits files
    for filename in cal_files:
        print(filename)
    print("N files: {0}".format(len(cal_files)))

    hdulist_sc = fits.open(cal_files[0])
    grating = hdulist_sc[0].header["GRATING"].strip()
    detector = hdulist_sc[0].header["DETECTOR"].strip().lower()
    if wv_sampling is None:
        wv_sampling = np.arange(np.nanmin(hdulist_sc["WAVELENGTH"].data),
                                np.nanmax(hdulist_sc["WAVELENGTH"].data),
                                np.nanmedian(hdulist_sc["WAVELENGTH"].data)/300)
    hdulist_sc.close()

    #     wv_for_cent_calib_dict[grating+" "+detector]

    regwvs_dataobj_list= []
    for filename in cal_files[0::]:
        print(filename)
        if detector not in filename:
            raise Exception("The files in cal_files should all be for the same detector")

        preproc_task_list = []
        preproc_task_list.append(["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}, True, True])
        preproc_task_list.append(["compute_coordinates_arrays"])
        preproc_task_list.append(["convert_MJy_per_sr_to_MJy"])
        preproc_task_list.append(["compute_starspectrum_contnorm", {"N_nodes": N_wvs_nodes,
                                                                     "threshold_badpix": 100,
                                                                     "mppool": mypool}, True, True])
        preproc_task_list.append(["compute_starsubtraction", {"starsub_dir": "starsub1d_tmp",
                                                              "threshold_badpix": 10,
                                                              "mppool": mypool}, True, True])
        preproc_task_list.append(["compute_interpdata_regwvs", {"wv_sampling": wv_sampling}, True, False])

        dataobj = JWSTNirspec_cal(filename, crds_dir=crds_dir, utils_dir=utils_dir,
                                  save_utils=True,load_utils=True,preproc_task_list=preproc_task_list)
        regwvs_dataobj_list.append(dataobj.reload_interpdata_regwvs())

    regwvs_combdataobj = JWSTNirspec_multiple_cals(regwvs_dataobj_list)
    if mask_charge_transfer_radius is not None:
        regwvs_combdataobj.compute_charge_bleeding_mask(threshold2mask=mask_charge_transfer_radius)

    # Load the webbPSF model (or compute if it does not yet exist)
    webbpsf_reload = regwvs_combdataobj.reload_webbpsf_model()
    if webbpsf_reload is None:
        webbpsf_reload = regwvs_combdataobj.compute_webbpsf_model(wv_sampling=regwvs_combdataobj.wv_sampling,
                                                                  image_mask=None,
                                                                  pixelscale=0.1, oversample=10,
                                                                  parallelize=False, mppool=mypool,
                                                                  save_utils=True)
    wpsfs, wpsfs_header, wepsfs, webbpsf_wvs, webbpsf_X, webbpsf_Y, wpsf_oversample, wpsf_pixelscale = webbpsf_reload
    webbpsf_X = np.tile(webbpsf_X[None, :, :], (wepsfs.shape[0], 1, 1))
    webbpsf_Y = np.tile(webbpsf_Y[None, :, :], (wepsfs.shape[0], 1, 1))


    # Definie the filename of the output file saved by fitpsf
    splitbasename = os.path.basename(regwvs_combdataobj.filename).split("_")
    filename_suffix = "_webbpsf"
    fitpsf_filename = os.path.join(utils_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[3] + "_fitpsf" + filename_suffix + ".fits")

    # Fit a model PSF (WebbPSF) to the combined point cloud of dataobj_list
    # Save output as fitpsf_filename
    ann_width = None
    padding = 0.0
    sector_area = None
    where_center_disk = regwvs_combdataobj.where_point_source((0.0,0.0),IWA)
    regwvs_combdataobj.bad_pixels[where_center_disk] = np.nan

    # l0 = 100
    # plt.scatter(regwvs_combdataobj.dra_as_array[:, l0], regwvs_combdataobj.ddec_as_array[:, l0],
    #             c=10 * regwvs_combdataobj.data[:, l0] / np.nanmax(regwvs_combdataobj.data[:, l0]), s=1)
    # plt.show()
    fitpsf(regwvs_combdataobj,wepsfs,webbpsf_X,webbpsf_Y, out_filename=fitpsf_filename,IWA = 0.0,OWA = OWA,
           mppool=mypool,init_centroid=init_centroid,ann_width=ann_width,padding=padding,
           sector_area=sector_area,RDI_folder_suffix=filename_suffix,rotate_psf=regwvs_combdataobj.east2V2_deg,
           flipx=True,psf_spaxel_area=(wpsf_pixelscale) ** 2,debug_init=debug_init,debug_end=debug_end)

    with fits.open(fitpsf_filename) as hdulist:
        bestfit_coords = hdulist[0].data
        wpsf_angle_offset = hdulist[0].header["INIT_ANG"]
        wpsf_ra_offset = hdulist[0].header["INIT_RA"]
        wpsf_dec_offset = hdulist[0].header["INIT_DEC"]

    x2fit = wv_sampling - np.nanmedian(wv_sampling)
    y2fit = bestfit_coords[0, :, 2]
    # if detector == "nrs1":
    #     _wv_min, _wv_max = 3.0, 4.0
    # elif detector == "nrs2":
    #     _wv_min, _wv_max = 4.3, 5.2
    _wv_min = wv_sampling[0]+0.1*(wv_sampling[-1]-wv_sampling[0])
    _wv_max = wv_sampling[-1]-0.1*(wv_sampling[-1]-wv_sampling[0])
    print(_wv_min, _wv_max)
    wherefinite = np.where(np.isfinite(y2fit) * (wv_sampling > _wv_min) * (wv_sampling < _wv_max))
    poly_p_RA = np.polyfit(x2fit[wherefinite], y2fit[wherefinite], deg=2)
    print("RA correction " + detector, poly_p_RA)

    x2fit = wv_sampling - np.nanmedian(wv_sampling)
    y2fit = bestfit_coords[0, :, 3]
    wherefinite=np.where(np.isfinite(y2fit)*(wv_sampling>_wv_min)*(wv_sampling<_wv_max))
    poly_p_dec = np.polyfit(x2fit[wherefinite],y2fit[wherefinite], deg=2)
    # plt.scatter(x2fit[wherefinite],y2fit[wherefinite],label=detector)
    print("Dec correction "+detector, poly_p_dec)

    if save_plots:
        color_list = ["#ff9900", "#006699", "#6600ff", "#006699", "#ff9900", "#6600ff"]
        print(bestfit_coords.shape)
        fontsize=12
        plt.figure(figsize=(12,10))
        plt.subplot(3,1,1)

        plt.plot(wv_sampling,bestfit_coords[0,:,0]*1e9,linestyle="-",color=color_list[0],label="Fixed centroid",linewidth=1)
        plt.plot(wv_sampling,bestfit_coords[0,:,1]*1e9,linestyle="--",color=color_list[2],label="Free centroid",linewidth=1)
        plt.xlim([wv_sampling[0],wv_sampling[-1]])
        plt.xlabel("Wavelength ($\mu$m)",fontsize=fontsize)
        plt.ylabel("Flux density (mJy)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right")

        plt.subplot(3,1,2)
        plt.plot(wv_sampling,bestfit_coords[0,:,2],label="bestfit centroid")
        poly_model = np.polyval(poly_p_RA,wv_sampling - np.nanmedian(wv_sampling))
        plt.plot(wv_sampling,poly_model,label="polyfit")
        plt.plot(wv_sampling,bestfit_coords[0,:,2]-poly_model,label="residuals")
        plt.xlabel("Wavelength ($\mu$m)",fontsize=fontsize)
        plt.ylabel("$\Delta$RA (as)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right")

        plt.subplot(3,1,3)
        plt.plot(wv_sampling,bestfit_coords[0,:,3],label="bestfit centroid")
        poly_model = np.polyval(poly_p_dec,wv_sampling - np.nanmedian(wv_sampling))
        plt.plot(wv_sampling,poly_model,label="polyfit")
        plt.plot(wv_sampling,bestfit_coords[0,:,3]-poly_model,label="residuals")
        plt.xlabel("Wavelength ($\mu$m)",fontsize=fontsize)
        plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        plt.tight_layout()

        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y%m%d_%H%M%S")

        out_filename = os.path.join(utils_dir, formatted_datetime+"_centroid_calibration.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)
        plt.savefig(out_filename.replace(".png", ".pdf"),bbox_inches='tight')

        # out_filename = os.path.join(out_png, "HR8799_spectrum_microns_MJy_"+obsnum+".png")
        # hdulist = fits.HDUList()
        # hdulist.append(fits.PrimaryHDU(data=flux2save_wvs))
        # hdulist.append(fits.ImageHDU(data=flux2save))
        # try:
        #     hdulist.writeto(out_filename, overwrite=True)
        # except TypeError:
        #     hdulist.writeto(out_filename, clobber=True)
        # hdulist.close()

    return poly_p_RA,poly_p_dec


###########################################################################
#  Functions for noise cleaning

from breads.utils import get_spline_model
def fm_column_background(nonlin_paras, cubeobj, nodes=20,
                         fix_parameters=None,
                         return_where_finite=False,
                         fit_diffus=False,
                         regularization=None,
                         badpixfraction=0.75,
                         M_spline=None):
    """ Forward model column background, for use in forward_model_noise_clean


    """
    if fix_parameters is not None:
        _nonlin_paras = np.array(fix_parameters)
        _nonlin_paras[np.where(np.array(fix_parameters) is None)] = nonlin_paras
    else:
        _nonlin_paras = nonlin_paras

    if M_spline is None:
        if type(nodes) is int:
            N_nodes = nodes
            x_knots = np.linspace(0, np.size(cubeobj.data), N_nodes, endpoint=True).tolist()
        elif type(nodes) is list or type(nodes) is np.ndarray:
            x_knots = nodes
            if type(nodes[0]) is list or type(nodes[0]) is np.ndarray:
                N_nodes = np.sum([np.size(n) for n in nodes])
            else:
                N_nodes = np.size(nodes)
        else:
            raise ValueError("Unknown format for nodes.")
    else:
        N_nodes = M_spline.shape[1]

    # Number of linear parameters
    N_linpara = N_nodes
    if fit_diffus:
        N_linpara += 1

    data = cubeobj.data
    noise = cubeobj.noise
    bad_pixels = cubeobj.bad_pixels

    where_trace_finite = np.where(np.isfinite(data)*np.isfinite(bad_pixels)*(noise!=0))
    d = data[where_trace_finite]
    s = noise[where_trace_finite]

    # print("coucou")
    # print(np.size(where_trace_finite[0]), (1-badpixfraction) * np.sum(new_mask), vsini < 0)
    if np.size(where_trace_finite[0]) <= (1-badpixfraction) * np.size(data):
        # don't bother to do a fit if there are too many bad pixels
        return np.array([]), np.array([]).reshape(0,N_linpara), np.array([])
    else:
        x = np.arange(np.size(cubeobj.data))
        if M_spline is None:
            M = get_spline_model(x_knots, x, spline_degree=3)
        else:
            M =copy(M_spline)
        # print(M_spline.shape)

        if fit_diffus:
            diffus_center = _nonlin_paras[0]
            diffus_scale = _nonlin_paras[1]
            diffus_model = np.exp(-np.abs(x-diffus_center)/diffus_scale)
            # diffus_model = np.exp(-(x-diffus_center)**2/diffus_scale**2)
            # diffus_model = 1-np.abs(x-diffus_center)/diffus_scale
            M_diffus = diffus_model[:,None]
            M = np.concatenate([M_diffus,M], axis=1)

        M = M[where_trace_finite[0],:]

        extra_outputs = {}
        if regularization == "default":
            s_reg = np.array(np.nanmax(d) + np.zeros(N_nodes))
            d_reg = np.array(0 + np.zeros(N_nodes))
            if fit_diffus:
                s_reg = np.concatenate([[np.nan],s_reg])
                d_reg = np.concatenate([[np.nan],d_reg])
            extra_outputs["regularization"] = (d_reg, s_reg)
        elif regularization == "user":
            raise Exception("user defined regularisation not yet implemented")
            extra_outputs["regularization"] = (d_reg, s_reg)

        if return_where_finite:
            extra_outputs["where_trace_finite"] = where_trace_finite

        if len(extra_outputs) >= 1:
            return d, M, s, extra_outputs
        else:
            return d, M, s


def forward_model_noise_clean(rate_file, cal_file_dir, clean_dir, N_nodes = 40):
    """ Clean 1/f stripe noise from NIRSpec IFU data. Inspired by NSClean but implemented independently.

    The way it works:
        subtraction done on rate.fits
        Use the cal.fits to retrieve the mask of the IFU slices
        Fit detector columns one at time. I just fit a smooth continuum (using my splines) to the masked detector column, and also masking the region around the star more aggressively I believe
        subtract the fitted continuum
        Save new rate.fits
    """

    x = np.arange(2048)
    x_knots = np.linspace(0, 2048, N_nodes, endpoint=True).tolist()
    M_spline = get_spline_model(x_knots, x, spline_degree=3)


    basename = os.path.basename(rate_file)
    cal_filename = os.path.join(cal_file_dir, basename.replace("_rate.fits","_cal.fits"))
    print(cal_filename)
    print(glob(cal_filename))
    with fits.open(cal_filename) as hdul:
        cal_im = hdul["SCI"].data

    # Get data
    hdul = fits.open(rate_file)


    priheader = hdul[0].header
    im = hdul["SCI"].data
    im_ori = copy(im)
    noise = hdul["ERR"].data
    dq = hdul["DQ"].data

    from breads.instruments.jwstnirspec_cal import untangle_dq
    # Simplifying bad pixel map following convention in this package as: nan = bad, 1 = good
    # bad_pixels = np.ones((ny, nx))
    bad_pixels=np.zeros(cal_im.shape)+np.nan
    bad_pixels[np.where(np.isnan(cal_im))] = 1
    # Pixels marked as "do not use" are marked as bad (nan = bad, 1 = good):
    bad_pixels[np.where(untangle_dq(dq, verbose=True)[0, :, :])] = np.nan
    bad_pixels[np.where(np.isnan(im))] = np.nan
    im[np.where(np.isnan(im))] = 0

    #Removing any data with zero noise
    where_zero_noise = np.where(noise == 0)
    noise[where_zero_noise] = np.nan
    bad_pixels[where_zero_noise] = np.nan

    if "nrs1" in rate_file:
        for rowid in range(im.shape[0]):
            finite_ids = np.where(np.isfinite(cal_im[rowid,0:450]))[0]
            if len(finite_ids) != 0 :
                id_to_mask = np.min(finite_ids)
                bad_pixels[rowid,0:id_to_mask] = np.nan
    elif "nrs2" in rate_file:
        for rowid in range(im.shape[0]):
            finite_ids = np.where(np.isfinite(cal_im[rowid,1550::]))[0]
            if len(finite_ids) != 0 :
                id_to_mask = np.max(finite_ids)
                bad_pixels[rowid,1550+id_to_mask::] = np.nan
    bad_pixels[np.where(np.abs(im)>5)] = np.nan

    from breads.instruments.instrument import Instrument
    data = Instrument()
    new_im = np.zeros(im.shape)
    for colid in range(im.shape[1]):
        # print(colid)
        # colid=300
        data.data = copy(im[:,colid])
        data.noise = copy(noise[:,colid])
        data.bad_pixels = copy(bad_pixels[:,colid])

        fm_paras = {"badpixfraction":0.99,"nodes":N_nodes,"fix_parameters": [],"regularization":"default","fit_diffus":False,"M_spline":M_spline}
        # fm_paras = {"badpixfraction":0.99,"nodes":N_nodes,"fix_parameters": nonlin_paras,"regularization":"default","fit_diffus":True}
        log_prob, log_prob_H0, rchi2, linparas, linparas_err = fitfm([],data,fm_column_background,fm_paras)
        if not np.isfinite(log_prob):
            continue
        d_masked, M, s,extra_outputs = fm_column_background([],data,return_where_finite=True,**fm_paras)
        where_finite = extra_outputs["where_trace_finite"]
        data.bad_pixels = np.ones(data.data.shape)
        d, M, s,_ = fm_column_background([],data,return_where_finite=True,**fm_paras)
        # print(data.data.shape,d.shape,np.size(where_finite[0]),np.size(d_masked))
        d_masked_canvas = np.zeros(d.shape)+np.nan
        d_masked_canvas[where_finite] = d_masked

        m = np.dot(M,linparas)
        new_im[:,colid] = im_ori[:,colid]-m

        data.bad_pixels = bad_pixels[:,colid]
        mad = median_abs_deviation((d_masked_canvas - m)[np.where(np.isfinite(d_masked_canvas))])
        data.bad_pixels[np.where(np.abs(d_masked_canvas-m)>5*mad)] = np.nan
        # plt.plot(data.bad_pixels)
        # plt.show()

        # Redo the fit with outliers removed
        log_prob, log_prob_H0, rchi2, linparas, linparas_err = fitfm([],data,fm_column_background,fm_paras)
        if not np.isfinite(log_prob):
            continue
        d_masked, M, s,extra_outputs = fm_column_background([],data,return_where_finite=True,**fm_paras)
        where_finite = extra_outputs["where_trace_finite"]
        data.bad_pixels = np.ones(data.data.shape)
        d, M, s,_ = fm_column_background([],data,return_where_finite=True,**fm_paras)
        # print(data.data.shape,d.shape,np.size(where_finite[0]),np.size(d_masked))
        d_masked_canvas = np.zeros(d.shape)+np.nan
        d_masked_canvas[where_finite] = d_masked

        m = np.dot(M,linparas)
        new_im[:,colid] = im_ori[:,colid]-m

        # plt.figure(figsize=(12,4))
        # plt.subplot(2,1,1)
        # plt.plot(d,label="original data")
        # plt.plot(d_masked_canvas,label="masked data")
        # plt.plot(m,label="model",linestyle="--")
        # plt.ylim([-5,100])
        # plt.legend()
        # plt.subplot(2,1,2)
        # plt.plot(d_masked_canvas-m,label="original data")
        # plt.plot(m-m+5*mad)
        # plt.plot(m-m-5*mad)
        # plt.show()

    priheader['comment'] = 'Detector correlated noise removed by custom code'
    hdul[0].header = priheader
    hdul["SCI"].data = new_im
    hdul.writeto(os.path.join(clean_dir, os.path.basename(rate_file)), overwrite=True)
    hdul.close()



def run_noise_clean(rate_files, stage2_dir, clean_dir, overwrite=False):
    """Invoke forward model noise removal for a list of rate files
    """
    # We need to check that the desired output directories exist, and if not create them
    if not os.path.exists(clean_dir):
        os.makedirs(clean_dir)

    # Start a timer to keep track of runtime
    time0 = time.perf_counter()
    print(time0)


    cleaned_rate_files = []

    for fid,rate_file in enumerate(rate_files):

        outname = os.path.join(clean_dir, os.path.basename(rate_file))
        cleaned_rate_files.append(outname)
        if os.path.exists(outname) and not overwrite:
            print(f"Output file {outname} already exists in the cleaned output directory; skipping {rate_file}.")
            continue


        print(f"Processing file {fid+1} of {len(rate_files)}: {rate_file}")

        forward_model_noise_clean(rate_file, cal_file_dir=stage2_dir, clean_dir=clean_dir)

        # Print out the time benchmark
        time1 = time.perf_counter()
        print(f"Runtime so far: {time1 - time0:0.4f} seconds\n")
    time1 = time.perf_counter()
    print(f"Total Runtime: {time1 - time0:0.4f} seconds")

    return cleaned_rate_files

############################################################################
#  Function to invoke all reduction steps in one go


def run_complete_stage1_2_clean_reduction(input_dir, output_root_dir=None, overwrite=False):
    """Overarching top-level function to invoke stage1, stage2, and 1/f noise cleaning code

    This will run the complete reduction from uncal files to cal files. It will take a while.

    If files already exist, repeat reductions are skipped, unless overwrite is set True
    """

    if output_root_dir is None:
        output_root_dir = input_dir

    # Set up subdirectory paths
    det1_dir = os.path.join(output_root_dir,"stage1")    # Detector1 pipeline outputs will go here
    spec2_dir = os.path.join(output_root_dir,"stage2")   # Initial spec2 pipeline outputs will go here
    clean_det1_dir = os.path.join(output_root_dir,"stage1_clean")   # noise-cleaned Detector1 pipeline outputs will go here
    clean_spec2_dir = os.path.join(output_root_dir,"stage2_clean")  # noise-cleaned Spec2 pipeline outputs will go here

    # Find input rate files
    uncal_files = find_files_to_process(input_dir, 'uncal.fits')

    # Run all reduction steps
    rate_files = run_stage1(uncal_files, output_dir=det1_dir, overwrite=overwrite)
    cleaned_rate_files = run_noise_clean(rate_files, spec2_dir, clean_det1_dir, overwrite=overwrite)
    cleaned_cal_files = run_stage2(cleaned_rate_files, output_dir=clean_spec2_dir, overwrite=overwrite)

    return cleaned_cal_files
