
import os
import time
from glob import glob
from copy import copy

import numpy as np
from scipy.stats import median_abs_deviation
from astropy.io import fits

import jwst
from jwst.pipeline import Detector1Pipeline, Spec2Pipeline

import multiprocessing as mp
import matplotlib.pyplot as plt
import datetime
from scipy.ndimage import generic_filter
from scipy.ndimage import convolve1d
from  scipy.interpolate import interp1d
import matplotlib.tri as tri

from breads.instruments.instrument import Instrument
from breads.instruments.jwstnirspec_cal import JWSTNirspec_cal
from breads.instruments.jwstnirspec_cal import untangle_dq
from breads.instruments.jwstnirspec_cal import crop_trace_edges
from breads.instruments.jwstnirspec_cal import where_point_source
from breads.instruments.jwstnirspec_cal import fitpsf
from breads.instruments.jwstnirspec_cal import get_contnorm_spec
from breads.instruments.jwstnirspec_cal import filter_big_triangles
from breads.instruments.jwstnirspec_multiple_cals import JWSTNirspec_multiple_cals
from breads.fit import fitfm


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
                             save_plots=False,
                            filename_suffix = "_webbpsf",
                            overwrite = False):

    mypool = mp.Pool(processes=numthreads)

    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)

    # Science data: List of stage 2 cal.fits files
    for filename in cal_files:
        print(filename)
    print("N files: {0}".format(len(cal_files)))

    # Definie the filename of the output file saved by fitpsf
    splitbasename = os.path.basename(cal_files[0]).split("_")
    fitpsf_filename = os.path.join(utils_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[3] + "_fitpsf" + filename_suffix + ".fits")
    poly2d_centroid_filename = os.path.join(utils_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[3] + "_poly2d_centroid" + filename_suffix + ".txt")


    hdulist_sc = fits.open(cal_files[0])
    grating = hdulist_sc[0].header["GRATING"].strip()
    detector = hdulist_sc[0].header["DETECTOR"].strip().lower()
    if wv_sampling is None:
        wv_sampling = np.arange(np.nanmin(hdulist_sc["WAVELENGTH"].data),
                                np.nanmax(hdulist_sc["WAVELENGTH"].data),
                                np.nanmedian(hdulist_sc["WAVELENGTH"].data)/300)
    hdulist_sc.close()

    if not overwrite:
        if len(glob(poly2d_centroid_filename)) == 1:
            output = np.loadtxt(poly2d_centroid_filename, delimiter=' ')
            poly_p_RA, poly_p_dec = output[0],output[1]
            print("RA correction " + detector, poly_p_RA)
            print("Dec correction "+detector, poly_p_dec)
            return poly_p_RA, poly_p_dec

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

    np.savetxt(poly2d_centroid_filename, [poly_p_RA,poly_p_dec], delimiter=' ')

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
        # plt.savefig(out_filename.replace(".png", ".pdf"),bbox_inches='tight')

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
                         regularization=None,
                         badpixfraction=0.75,
                         M_spline=None,
                         spline_reg_std=1.0):
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


        M = M[where_trace_finite[0],:]

        extra_outputs = {}
        if regularization == "default":
            s_reg = np.zeros(N_nodes)+spline_reg_std#np.array(np.nanmax(d) + np.zeros(N_nodes))
            d_reg = np.zeros(N_nodes)
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



def fm_charge_transfer(nonlin_paras, cubeobj, charge_transfer_mask=None,nodes=40, fix_parameters=None,return_where_finite=False,
             regularization=None,badpixfraction=0.75, M_spline=None,spline_reg_std=1.0):
    if fix_parameters is not None:
        _nonlin_paras = np.array(fix_parameters)
        _nonlin_paras[np.where(np.array(fix_parameters)==None)] = nonlin_paras
    else:
        _nonlin_paras = nonlin_paras

    if M_spline is None:
        if type(nodes) is int:
            N_nodes = nodes
            x_knots = np.linspace(np.nanmin(cubeobj.wavelengths), np.nanmax(cubeobj.wavelengths), N_nodes, endpoint=True).tolist()
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


    where_finite = np.where(np.isfinite(cubeobj.data)*np.isfinite(cubeobj.bad_pixels)*(cubeobj.noise!=0))
    d = cubeobj.data[where_finite]
    s = cubeobj.noise[where_finite]


    # print("coucou")
    # print(np.size(where_trace_finite[0]), (1-badpixfraction) * np.sum(new_mask), vsini < 0)
    if np.size(where_finite[0]) <= (1-badpixfraction) * np.size(cubeobj.data):
        # don't bother to do a fit if there are too many bad pixels
        return np.array([]), np.array([]).reshape(0,N_linpara), np.array([])
    else:
        where_finite_wvs = np.where(np.isfinite(cubeobj.wavelengths))
        if M_spline is None:
            M_tmp = get_spline_model(x_knots, cubeobj.wavelengths[where_finite_wvs], spline_degree=3)
        else:
            M_tmp =copy(M_spline)
        M_entire_image = np.zeros((cubeobj.data.shape[0],cubeobj.data.shape[1],M_tmp.shape[1]))
        M_entire_image[where_finite_wvs[0],where_finite_wvs[1],:] = M_tmp
        M_entire_image = M_entire_image*charge_transfer_mask[:,:,None]
        M_entire_image[np.where(np.isnan(M_entire_image))]=0
        # print(M_spline.shape)
        # # plt.figure()
        # # plt.imshow(charge_transfer_mask,origin="lower")
        # plt.figure()
        # plt.imshow(M_entire_image[:,:,20],origin="lower")
        # plt.figure()
        # plt.plot(M_entire_image[:,1150,20])
        # plt.show()

        kernel_scale = _nonlin_paras[0]
        x = np.arange(-cubeobj.data.shape[0],cubeobj.data.shape[0]+1)
        charge_transfer_kernel = 1/(1+x**2/kernel_scale**2) #Lorentzian Function

        # tmp = M_entire_image[:, 1150:1250, :]
        # M_entire_image_convolved = convolve1d(tmp, weights=charge_transfer_kernel, axis=0, mode='constant')
        # plt.figure()
        # plt.plot(M_entire_image_convolved[:,0,2])
        # plt.figure()
        # plt.plot(tmp[:,0,2])
        # plt.show()

        # Convolve each column with the kernel
        M_entire_image_convolved = convolve1d(M_entire_image, weights=charge_transfer_kernel, axis=0, mode='constant')
        M = M_entire_image_convolved[where_finite[0],where_finite[1],:]
        # plt.imshow(M_entire_image_convolved[:,:,2],origin="lower")
        # plt.show()

        extra_outputs = {}
        if regularization == "default":
            s_reg = np.zeros(N_nodes)+spline_reg_std#np.array(np.nanmax(d) + np.zeros(N_nodes))
            d_reg = np.zeros(N_nodes)
            extra_outputs["regularization"] = (d_reg, s_reg)
        elif regularization == "user":
            raise Exception("user defined regularisation not yet implemented")
            extra_outputs["regularization"] = (d_reg, s_reg)

        if return_where_finite:
            extra_outputs["where_finite"] = where_finite

        if len(extra_outputs) >= 1:
            return d, M, s, extra_outputs
        else:
            return d, M, s

def forward_model_noise_clean(rate_file, cal_file_dir, clean_dir, crds_dir, N_nodes = 40,model_charge_transfer=False,
                              utils_dir=None,coords_offset=(0,0)):
    """ Clean 1/f stripe noise from NIRSpec IFU data. Inspired by NSClean but implemented independently.

    The way it works:
        subtraction done on rate.fits
        Use the cal.fits to retrieve the mask of the IFU slices
        Fit detector columns one at time. I just fit a smooth continuum (using my splines) to the masked detector column, and also masking the region around the star more aggressively I believe
        subtract the fitted continuum
        Save new rate.fits
    """


    basename = os.path.basename(rate_file)
    cal_filename = os.path.join(cal_file_dir, basename.replace("_rate.fits","_cal.fits"))
    if len(glob(cal_filename)) == 0:
        raise Exception("Could not find the corresponding cal file. Please run stage 2 without cleaning first.")

    cal_dataobj = JWSTNirspec_cal(cal_filename, crds_dir=crds_dir, utils_dir=utils_dir,
                                  save_utils=True, load_utils=True)
    out = cal_dataobj.reload_coordinates_arrays()
    if out is None:
        cal_dataobj.compute_coordinates_arrays(save_utils=True)
    cal_dataobj.apply_coords_offset(coords_offset=coords_offset)

    ra_im, dec_im = cal_dataobj.getskycoords()
    sep_im = np.sqrt(ra_im ** 2 + dec_im ** 2)
    cal_im = cal_dataobj.data

    hdulist_cal = fits.open(cal_dataobj.filename)
    dq = hdulist_cal["DQ"].data
    hdulist_cal.close()



    print(cal_filename)
    print(glob(cal_filename))
    with fits.open(cal_filename) as hdul:
        cal_im = hdul["SCI"].data

    # Get data. Read rate.fits file
    hdul = fits.open(rate_file)
    priheader = hdul[0].header
    extheader = hdul[1].header
    im = hdul["SCI"].data
    # im_ori = copy(im)
    noise = hdul["ERR"].data
    dq = hdul["DQ"].data
    ny, nx = im.shape

    cal_mask = np.ones(cal_im.shape)
    cal_mask[np.where(np.isnan((sep_im)))] = np.nan

    # Simplifying bad pixel map following convention in this package as: nan = bad, 1 = good
    bad_pixels = np.full(cal_im.shape, np.nan)  # array full of nans
    # We select only the background pixels:
    bad_pixels[np.where(np.isnan(cal_mask))] = 1  # every pixel that is not in a cal slice is actually good here
    # Pixels marked as "do not use" are marked as bad (nan = bad, 1 = good):
    bad_pixels[np.where(untangle_dq(dq, verbose=True)[0, :, :])] = np.nan
    bad_pixels[np.where(np.isnan(im))] = np.nan
    # Removing any data with zero noise
    where_zero_noise = np.where(noise == 0)
    noise[where_zero_noise] = np.nan
    bad_pixels[where_zero_noise] = np.nan

    im[np.where(np.isnan(im))] = 0

    ## Extent the slices masks to the edge of the detector
    if "nrs1" in rate_file:
        for rowid in range(im.shape[0]):
            finite_ids = np.where(np.isfinite(sep_im[rowid, 0:450]))[0]
            if len(finite_ids) != 0:
                id_to_mask = np.min(finite_ids)
                bad_pixels[rowid, 0:id_to_mask] = np.nan
                sep_im[rowid, 0:id_to_mask] = sep_im[rowid, id_to_mask]
    elif "nrs2" in rate_file:
        for rowid in range(im.shape[0]):
            finite_ids = np.where(np.isfinite(sep_im[rowid, 1550::]))[0]
            if len(finite_ids) != 0:
                id_to_mask = np.max(finite_ids)
                bad_pixels[rowid, 1550 + id_to_mask::] = np.nan
                sep_im[rowid, 1550 + id_to_mask::] = sep_im[rowid, 1550 + id_to_mask]

    mad_threshold = 5
    window_size = 50
    new_badpix = np.ones(bad_pixels.shape)
    for rowid in range(bad_pixels.shape[0]):
        row_data = im[rowid, :] - generic_filter(im[rowid, :] * bad_pixels[rowid, :], np.nanmedian, size=window_size)
        row_data_masking = row_data / median_abs_deviation(row_data[np.where(np.isfinite(bad_pixels[rowid, :]))])
        new_badpix[rowid, np.where((row_data_masking > mad_threshold))[0]] = np.nan
    bad_pixels *= new_badpix


    if model_charge_transfer:
        data = Instrument()

        data.data = copy(im)
        data.noise = copy(noise)
        data.bad_pixels = copy(bad_pixels)
        data.wavelengths = copy(cal_dataobj.wavelengths)

        cal_model_filename = os.path.join(utils_dir, "RDI_model_webbpsf", basename.replace("_rate.fits", "_cal.fits"))
        if len(glob(cal_model_filename)) == 0:
            raise Exception(
                "Could not find the corresponding RDI model webbpsf cal file. Please run run_coordinate_recenter(...) first.")
        with fits.open(cal_model_filename) as hdul_model:
            webbPSF_im = hdul_model["SCI"].data
        saturated_mask = np.full(cal_dataobj.data.shape, np.nan)
        saturated_mask[np.where(untangle_dq(dq, verbose=False)[1, :, :])] = 1
        saturated_mask[np.where((sep_im > 0.5))] = np.nan
        # Define the saturation threshold in Mjy/sr below. This is definitely not ideal, probably not accurate.
        # Will probably need to fix later.
        saturation_threshold = 1e5  # Mjy/sr

        charge_transfer_mask = (webbPSF_im - saturation_threshold) * saturated_mask
        charge_transfer_mask[np.where(np.isnan(charge_transfer_mask))] = 0.0
        charge_transfer_mask = np.clip(charge_transfer_mask, 0, np.inf)

        # Define the spline nodes for fitting the background in each detector column
        N_nodes_charge_transfer = 5  # number of nodes in the column
        x_knots_charge_transfer = np.linspace(np.nanmin(cal_dataobj.wavelengths), np.nanmax(cal_dataobj.wavelengths),
                                              N_nodes_charge_transfer,
                                              endpoint=True).tolist()
        where_finite_wvs = np.where(np.isfinite(cal_dataobj.wavelengths))
        M_spline_charge_transfer = get_spline_model(x_knots_charge_transfer, cal_dataobj.wavelengths[where_finite_wvs],
                                                    spline_degree=3)

        fix_parameters = [10]  # width of the lorentzian
        fm_paras = {"charge_transfer_mask": charge_transfer_mask, "fix_parameters": fix_parameters,
                    "regularization": None, "badpixfraction": 0.75, "M_spline": M_spline_charge_transfer,
                    "spline_reg_std": 1.0}
        nonlin_paras = []
        out_log_prob, _, rchi2, linparas, linparas_err = fitfm(nonlin_paras, data, fm_charge_transfer, fm_paras,
                                                               computeH0=False, scale_noise=False)
        d_masked, M, s, extra_outputs = fm_charge_transfer(nonlin_paras, data, return_where_finite=True, **fm_paras)
        where_finite = extra_outputs["where_finite"]
        d_masked_canvas = np.zeros(data.data.shape) + np.nan
        d_masked_canvas[where_finite] = d_masked
        data.bad_pixels = np.ones(data.data.shape)
        d, M, s, _ = fm_charge_transfer(nonlin_paras, data, return_where_finite=True, **fm_paras)
        model_canvas = np.dot(M, linparas)
        model_canvas = np.reshape(model_canvas, data.data.shape)

        ################################
        # remove lorentzian model
        im -= model_canvas
        ################################

    x = np.arange(2048)
    x_knots_column = np.linspace(0, 2048, N_nodes, endpoint=True).tolist()
    M_spline_column = get_spline_model(x_knots_column, x, spline_degree=3)

    data = Instrument()
    new_im = np.zeros(im.shape)
    for colid in range(im.shape[1]):
        # print(colid)
        # colid=300
        data.data = copy(im[:, colid])
        data.noise = copy(noise[:, colid])
        data.bad_pixels = copy(bad_pixels[:, colid])

        nonlin_paras = []
        fm_paras = {"badpixfraction": 0.99, "nodes": N_nodes, "fix_parameters": [],
                    "regularization": "default", "M_spline": M_spline_column}
        if 1:  # optimize non linear parameter
            out_log_prob, _, rchi2, linparas, linparas_err = fitfm(nonlin_paras, data, fm_column_background, fm_paras,
                                                                   computeH0=False, scale_noise=False)
            if not np.isfinite(out_log_prob):
                continue
            d_masked, M, s, extra_outputs = fm_column_background(nonlin_paras, data, return_where_finite=True,
                                                                 **fm_paras)
            where_finite = extra_outputs["where_trace_finite"]
            data.bad_pixels = np.ones(data.data.shape)
            d, M, s, _ = fm_column_background(nonlin_paras, data, return_where_finite=True, **fm_paras)
            # print(data.data.shape,d.shape,np.size(where_finite[0]),np.size(d_masked))
            d_masked_canvas = np.zeros(d.shape) + np.nan
            d_masked_canvas[where_finite] = d_masked

            m = np.dot(M, linparas)
            mad = median_abs_deviation(((d_masked_canvas - m))[np.where(np.isfinite(d_masked_canvas))])

            data.bad_pixels = bad_pixels[:, colid]
            data.bad_pixels[np.where(np.abs(d_masked_canvas - m) > 5 * mad)] = np.nan

            # plt.plot(d_masked_canvas,label="d_masked_canvas")
            # plt.plot(m,label="m")
            # plt.plot(d_masked_canvas - m,label="r")
            # plt.plot(np.zeros(m.shape)+5 * mad,label="5 * mad")
            # plt.plot(np.zeros(m.shape)-5 * mad,label="-5 * mad")
            # plt.legend()
            # plt.show()
        if 1:  # optimize non linear parameter
            out_log_prob, _, rchi2, linparas, linparas_err = fitfm(nonlin_paras, data, fm_column_background, fm_paras,
                                                                   computeH0=False, scale_noise=False)
            if not np.isfinite(out_log_prob):
                continue
            d_masked, M, s, extra_outputs = fm_column_background(nonlin_paras, data, return_where_finite=True,
                                                                 **fm_paras)
            where_finite = extra_outputs["where_trace_finite"]
            data.bad_pixels = np.ones(data.data.shape)
            d, M, s, _ = fm_column_background(nonlin_paras, data, return_where_finite=True, **fm_paras)
            # print(data.data.shape,d.shape,np.size(where_finite[0]),np.size(d_masked))
            d_masked_canvas = np.zeros(d.shape) + np.nan
            d_masked_canvas[where_finite] = d_masked

            m = np.dot(M, linparas)

        new_im[:, colid] = im[:, colid] - m

    priheader['comment'] = 'Detector correlated noise removed by custom code'
    hdul[0].header = priheader
    hdul["SCI"].data = new_im
    new_rate_file = os.path.join(clean_dir, os.path.basename(rate_file))
    hdul.writeto(new_rate_file, overwrite=True)
    hdul.close()
    return new_rate_file



def run_noise_clean(rate_files, stage2_dir, clean_dir, crds_dir, N_nodes = 40,model_charge_transfer=False,
                              utils_dir=None,coords_offset=(0,0),overwrite=False):
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

        # forward_model_noise_clean(rate_file, cal_file_dir=stage2_dir, clean_dir=clean_dir)
        forward_model_noise_clean(rate_file, stage2_dir, clean_dir,crds_dir,
                                                      N_nodes=N_nodes,
                                                      model_charge_transfer=model_charge_transfer, utils_dir=utils_dir,
                                                      coords_offset=coords_offset)
        # exit()
        # Print out the time benchmark
        time1 = time.perf_counter()
        print(f"Runtime so far: {time1 - time0:0.4f} seconds\n")
    time1 = time.perf_counter()
    print(f"Total Runtime: {time1 - time0:0.4f} seconds")

    return cleaned_rate_files

def compute_normalized_stellar_spectrum(cal_files,utils_dir,crds_dir,coords_offset=(0,0),wv_nodes=None,
                                        mask_charge_transfer_radius=None,mppool=None,
                                        ra_dec_point_sources=None,overwrite=False):
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)

    hdulist_sc = fits.open(cal_files[0])
    detector = hdulist_sc[0].header["DETECTOR"].strip().lower()
    if wv_nodes is None:
        wv_nodes = np.linspace(np.nanmin(hdulist_sc["WAVELENGTH"].data),
                               np.nanmax(hdulist_sc["WAVELENGTH"].data),
                               40, endpoint=True)
    hdulist_sc.close()

    splitbasename = os.path.basename(cal_files[0]).split("_")
    combined_contnorm_spec_filename = os.path.join(utils_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + detector + "_starspec_contnorm_combined_1dspline.fits")

    if not overwrite:
        if len(glob(combined_contnorm_spec_filename)):
            with fits.open(combined_contnorm_spec_filename) as hdulist:
                new_wavelengths = hdulist[0].data
                combined_fluxes = hdulist[1].data
                combined_errors = hdulist[2].data
                combined_star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False, fill_value=1)
            return combined_star_func

    dataobj_list = []
    for filename in cal_files:
        print(filename)

        preproc_task_list = []
        preproc_task_list.append(["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}, True, True])
        preproc_task_list.append(["compute_coordinates_arrays"])
        preproc_task_list.append(["convert_MJy_per_sr_to_MJy"])
        preproc_task_list.append(["apply_coords_offset", {"coords_offset": coords_offset}])
        preproc_task_list.append(["compute_starspectrum_contnorm", {"x_nodes": wv_nodes,
                                                                    "threshold_badpix": 100,
                                                                    "mppool": mppool}, True, True])
        preproc_task_list.append(["compute_starsubtraction", {"starsub_dir": "starsub1d_tmp",
                                                              "threshold_badpix": 10,
                                                              "mppool": mppool}, True, True])

        dataobj = JWSTNirspec_cal(filename, crds_dir=crds_dir, utils_dir=utils_dir,
                                  save_utils=True, load_utils=True, preproc_task_list=preproc_task_list)

        # Do some masking
        dataobj.bad_pixels = crop_trace_edges(dataobj.bad_pixels, N_pix=1, trace_id_map=dataobj.trace_id_map)
        if mask_charge_transfer_radius is not None:
            dataobj.compute_charge_bleeding_mask(threshold2mask=mask_charge_transfer_radius)
        # mask planets before computing the star spectrum
        if ra_dec_point_sources is not None:
            for ra_pl,dec_pl in ra_dec_point_sources:
                where_pl = where_point_source(dataobj, [ra_pl / 1000., dec_pl / 1000.], 0.16)
                dataobj.bad_pixels[where_pl] = np.nan

        dataobj_list.append(dataobj)

    new_wavelengths, combined_fluxes, combined_errors = get_contnorm_spec(dataobj_list, spline2d=False,
                                                                          load_utils=False,
                                                                          out_filename=combined_contnorm_spec_filename,
                                                                          spec_R_sampling=2700 * 4,
                                                                          interpolation="linear")

    combined_star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False, fill_value=1)
    return combined_star_func

def compute_starlight_subtraction(cal_files,utils_dir,crds_dir,wv_nodes=None,combined_star_func=None,coords_offset=(0,0),mppool=None):

    hdulist_sc = fits.open(cal_files[0])
    detector = hdulist_sc[0].header["DETECTOR"].strip().lower()
    if wv_nodes is None:
        wv_nodes = np.linspace(np.nanmin(hdulist_sc["WAVELENGTH"].data),
                               np.nanmax(hdulist_sc["WAVELENGTH"].data),
                               40, endpoint=True)
    hdulist_sc.close()

    dataobj_list = []
    for filename in cal_files[0::]:
        print(filename)

        preproc_task_list = []
        preproc_task_list.append(["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}, True, True])
        preproc_task_list.append(["compute_coordinates_arrays"])
        preproc_task_list.append(["convert_MJy_per_sr_to_MJy"])
        preproc_task_list.append(["apply_coords_offset", {"coords_offset": coords_offset}])
        if combined_star_func is None:
            preproc_task_list.append(["compute_starspectrum_contnorm", {"x_nodes": wv_nodes,
                                                                        "threshold_badpix": 100,
                                                                        "mppool": mppool}, True, True])

        dataobj = JWSTNirspec_cal(filename, crds_dir=crds_dir, utils_dir=utils_dir,
                                  save_utils=True, load_utils=True, preproc_task_list=preproc_task_list)
        if combined_star_func is not None:
            dataobj.reload_starspectrum_contnorm()
            dataobj.star_func = combined_star_func

        outputs = dataobj.reload_starsubtraction()
        # outputs = None
        if outputs is None:
            outputs = dataobj.compute_starsubtraction(save_utils=True, starsub_dir="starsub1d",
                                                      threshold_badpix=10, mppool=mppool)
        subtracted_im, star_model, spline_paras0, _wv_nodes = outputs

        dataobj_list.append(dataobj)

    return dataobj_list

def get_combined_regwvs(dataobj_list,wv_sampling=None,mask_charge_transfer_radius=None,use_starsub1d = False):
    regwvs_dataobj_list = []
    for dataobj in dataobj_list:

        if use_starsub1d:
            starsub_filename = os.path.join(dataobj.utils_dir,"starsub1d",os.path.basename(dataobj.filename))
            starsub_dataobj = JWSTNirspec_cal(starsub_filename,crds_dir=dataobj.crds_dir, utils_dir=dataobj.utils_dir)
            if dataobj.data_unit == 'MJy':
                replace_data = dataobj.convert_MJy_per_sr_to_MJy(data_in_MJy_per_sr=starsub_dataobj.data)
            elif dataobj.data_unit == "MJy/sr":
                replace_data = starsub_dataobj.data
            regwvs_filename = dataobj.default_filenames["compute_interpdata_regwvs"].replace("_regwvs.fits","_starsub1d_regwvs.fits")
        else:
            replace_data = None
            regwvs_filename = dataobj.default_filenames["compute_interpdata_regwvs"]

        regwvs_dataobj = dataobj.reload_interpdata_regwvs(load_filename=regwvs_filename)
        if regwvs_dataobj is None:
            regwvs_dataobj = dataobj.compute_interpdata_regwvs(save_utils=regwvs_filename,wv_sampling=wv_sampling,replace_data=replace_data)
        regwvs_dataobj_list.append(regwvs_dataobj)

    regwvs_combdataobj = JWSTNirspec_multiple_cals(regwvs_dataobj_list)
    if mask_charge_transfer_radius is not None:
        regwvs_combdataobj.compute_charge_bleeding_mask(threshold2mask=mask_charge_transfer_radius)

    return regwvs_combdataobj

def save_combined_regwvs(regwvs_combdataobj,out_filename):
    hdulist = fits.HDUList()
    hdulist.append(fits.ImageHDU(data=regwvs_combdataobj.data, name='DATA'))
    hdulist.append(fits.ImageHDU(data=regwvs_combdataobj.noise, name='ERR'))
    hdulist.append(fits.ImageHDU(data=regwvs_combdataobj.dra_as_array, name='RA'))
    hdulist.append(fits.ImageHDU(data=regwvs_combdataobj.ddec_as_array, name='DEC'))
    hdulist.append(fits.ImageHDU(data=regwvs_combdataobj.wavelengths, name='WAVE'))
    hdulist.append(fits.ImageHDU(data=regwvs_combdataobj.wv_sampling, name='WV_SAMPLING'))
    hdulist.append(fits.ImageHDU(data=regwvs_combdataobj.bad_pixels, name='BADPIX'))
    try:
        hdulist.writeto(out_filename, overwrite=True)
    except TypeError:
        hdulist.writeto(out_filename, clobber=True)
    hdulist.close()

def get_2D_point_cloud_interpolator(regwvs_combdataobj,wv0):
    if isinstance(regwvs_combdataobj, str):
        with fits.open(regwvs_combdataobj) as hdulist:
            data = hdulist["DATA"].data
            dra_as_array = hdulist["RA"].data
            ddec_as_array = hdulist["DEC"].data
            bad_pixels = hdulist["BADPIX"].data
            wv_sampling = hdulist["WV_SAMPLING"].data
    else:
        data = regwvs_combdataobj.data
        dra_as_array = regwvs_combdataobj.dra_as_array
        ddec_as_array = regwvs_combdataobj.ddec_as_array
        bad_pixels = regwvs_combdataobj.bad_pixels
        wv_sampling =  regwvs_combdataobj.wv_sampling

    wv0_index = np.argmin(np.abs(wv_sampling-wv0))

    where_good = np.where(np.isfinite(bad_pixels[:, wv0_index]))
    X = dra_as_array[where_good[0], wv0_index]
    Y = ddec_as_array[where_good[0], wv0_index]
    Z = data[where_good[0], wv0_index]
    filtered_triangles = filter_big_triangles(X, Y, 0.2)
    # Create filtered triangulation
    filtered_tri = tri.Triangulation(X, Y, triangles=filtered_triangles)
    # Perform LinearTriInterpolator for filtered triangulation
    pointcloud_interp = tri.LinearTriInterpolator(filtered_tri, Z)

    return pointcloud_interp

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
