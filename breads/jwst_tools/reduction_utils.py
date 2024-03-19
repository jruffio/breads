
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

    files = glob(os.path.join(input_dir,"jw0*_"+filetype))
    files.sort()
    for file in files:
        print(file)
    print('Found ' + str(len(files)) + ' input files to process')
    return files

###########################################################################
# Functions for invoking the pipeline

def run_stage1(uncal_files, output_dir, overwrite=False):
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
            'jump': {'maximum_cores': 'all'},       # parallelize
            'ramp_fit': {'maximum_cores': 'all'},   # parallelize
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
