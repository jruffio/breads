import os
import time
from glob import glob
from copy import copy
import fnmatch

import numpy as np
from scipy.stats import median_abs_deviation
from astropy.io import fits
from tqdm import tqdm
from multiprocess import Pool
import itertools
import h5py

import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.gridspec as gridspec
import datetime
from scipy.ndimage import generic_filter, gaussian_filter
from scipy.ndimage import convolve1d
from scipy.ndimage import convolve
from scipy.ndimage import correlate
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
import matplotlib.tri as tri
import warnings
import matplotlib.gridspec as gridspec
from astropy import constants as const
from astropy import units as u
from astropy.table import Table
from scipy.optimize import minimize
from astropy.convolution import convolve, Box2DKernel

try:
    import jwst
    _HAS_OPTIONAL_DEPENDENCY_JWST = True

    from jwst.pipeline import Detector1Pipeline, Spec2Pipeline, Spec3Pipeline
    from jwst.associations import asn_from_list as afl  # Tools for creating association files
    from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
except ImportError:
    _HAS_OPTIONAL_DEPENDENCY_JWST = False

from breads.instruments.instrument import Instrument
from breads.instruments.jwstnirspec_cal import JWSTNirspec_cal
from breads.instruments.jwst_IFUs import untangle_dq
from breads.instruments.jwstnirspec_multiple_cals import JWSTNirspec_multiple_cals
from breads.fit import fitfm
from breads.utils import get_spline_model,get_breads_commit
import breads.jwst_tools.plotting
from breads.jwst_tools.plotting import filter_big_triangles
from breads.jwst_tools.fitpsf import fitpsf,project_psf_model
from breads.jwst_tools.spectra import combine_spectrum,combine_spectrum_1dspline
import breads.jwst_tools.default_nirspec as default
from breads.jwst_tools.build_cube import build_cube
from breads.jwst_tools.splines import evaluate_3dspline_pointcloud
from breads.fm.hc_atmgrid_splinefm_jwst_ifu_cal import hc_atmgrid_splinefm_jwst_ifu_cal
from breads.instruments.jwstnirspec_cal import PCA_wvs_axis
from breads.grid_search import grid_search

from collections import defaultdict


###########################################################################
#                       JWST reduction tools
#
# This module contains utility functions for JWST reductions, particularly
# for invoking the JWST pipeline with some customizations and additions for
# tuned for the kind of processing we want to do with breads.

def find_files_to_process(input_dir, filetype='uncal.fits', exp_numbers=None, verbose=True):
    """ Utility function to find files of a given type

    Parameters
    ----------
    input_dir : str
        Input directory to search for files
    filetype : str
        Filename match pattern. Either a simple ending string like 'uncal.fits' or a more
        complex regular expression search pattern. This will be used to search the
        input directory for all FITS files matching this pattern.
    exp_numbers :  list or ndarray of ints
        Optional list of exposure numbers. The list of files will be filtered to contain
        only this subset of exposure numbers.
    verbose : bool
        Be more verbose in outputs

    Returns
    -------
    files : list of str
        List of filenames found in input_dir matching the filetype (and exp_numbers, if provided)
    """

    search_pattern = filetype if filetype.startswith('jw') else  "jw*_" + filetype
    files = glob(os.path.join(input_dir, search_pattern))
    files.sort()
    if verbose:
        print(f"Searching in {input_dir} for files matching {search_pattern}")
        print('\tFound ' + str(len(files)) + ' input files to process')
        for file in files:
            print("\t" + os.path.basename(file))

    if exp_numbers is not None:
        # Use fnmatch to filter only the wanted exposure numbers
        files = [f for f in files if any(fnmatch.fnmatch(os.path.basename(f), "jw*_*_{0:05d}_*".format(num)) for num in exp_numbers)]

    return files

def check_instrument_grating(uncal_files):
    """ Read the GRATING keyword for a list of uncal files, and verify that all have the same GRATING
    value. This is useful because forward modeling should be done for only one grating at a time, but
    some JWST observations may use multiple gratings in different activities within the observation

    Parameters
    ----------
    uncal_files : list of str
        filenames

    Returns
    -------
    grating : str
        GRATING keyword value
    """

    gratings = [fits.getheader(f)['GRATING'] for f in uncal_files]
    gratings = set(gratings)
    if len(gratings) > 1:
        raise RuntimeError(f"The specified list of files contains multiple different GRATING values: {gratings}. "
                            "Adjust your input file selection criteria to select files all with the same spectral"
                            "grating value.")
    else:
        return list(gratings)[0]

###########################################################################
# Functions for invoking the pipeline

def run_stage1(uncal_files, output_dir, overwrite=False, maximum_cores="all", save_plots=True):
    warnings.warn("run_stage1 is deprecated. Please use run_stage1_nirspec instead.")
    return run_stage1_nirspec(uncal_files, output_dir, overwrite=overwrite, maximum_cores=maximum_cores,
                              save_plots=save_plots)

def run_stage1_nirspec(uncal_files, output_dir, overwrite=False, maximum_cores="all", save_plots=True):
    """ Run pipeline stage 1 for JWST/NIRSpec, with some customizations for reductions
    intended to be used with breads for IFU high contrast

    For each input file, before doing any reduction, the expected output filename
    is inferred, and it checks whether that output file already exists.
    If so, then it is NOT reduced again by default.
    Set the overwrite flag to True to re-reduce files

    Parameters
    ----------
    uncal_files : list of strings
        Filenames of uncal files to reduce
    output_dir : string
        Directory path for where to put the output files
    overwrite : bool
        Re-reduce and overwrite outputs whether they existed already. If the processed files already exists, it returns the file list of those.
        Default is to SKIP re-reducing anything already reduced.
    maximum_cores : string
        Passed to JWST pipeline functions that use multiprocessing, such as ramp fit
    save_plots : bool
        Whether to save the output plots. Default is True.
    """
    from jwst.pipeline import Detector1Pipeline

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    time0 = time.perf_counter()

    rate_files = []
    N_files_processed = 0
    for i, file in enumerate(uncal_files):
        print(f"Stage 1 Processing file {i + 1} of {len(uncal_files)}.")

        outname = os.path.join(output_dir, os.path.basename(file).replace('uncal.fits', 'rate.fits'))
        rate_files.append(outname)

        if os.path.exists(outname) and not overwrite:
            print(f"\tStage 1 Output file {os.path.basename(outname)} already exists in output dir;\n\tskipping {os.path.basename(file)}.")
            continue
        N_files_processed += 1

        det1 = Detector1Pipeline()  # Instantiate the pipeline

        # defining used pipeline steps
        # This version only shows the step parameters which are changes from defaults.
        step_parameters = {
            # group_scale - run with defaults
            # dq_init - run with defaults
            'saturation': {'n_pix_grow_sat': 0},  # check for saturated pixels, but do not expand to adjacent pixels
            # ipc - run with defaults
            # superbias - run with defaults
            # linearity - run with defaults
            'persistence': {'skip': True},
            # This step does nothing; there are no nonzero parameters in the reference files yet
            # dark_current : run with defaults
            'jump': {'maximum_cores': maximum_cores},  # parallelize
            'ramp_fit': {'maximum_cores': maximum_cores},  # parallelize
            # gain_scale : run with defaults
        }

        det1.call(file, save_results=True, output_dir=output_dir,
                  steps=step_parameters)

        # Print out the time benchmark
        time1 = time.perf_counter()
        print(f"\tStage 1 Runtime so far: {time1 - time0:0.4f} seconds")

    time1 = time.perf_counter()
    print(f"Stage 1 Total Runtime: {time1 - time0:0.4f} seconds")

    if save_plots and N_files_processed > 0:
        breads.jwst_tools.plotting.plot_2d_image_set(rate_files,
                                                     output_dir = output_dir,
                                                     suptitle="Stage 1 pipeline reduction results",
                                                     plot_label = 'stage1')

    return rate_files

def run_stage2(uncal_files, output_dir, skip_cubes=True, overwrite=False, TA=False, nsclean_skip=False, save_plots=True):
    warnings.warn("run_stage2 is deprecated. Please use run_stage2_nirspec instead.")
    return run_stage2_nirspec(uncal_files, output_dir, skip_cubes=skip_cubes, overwrite=overwrite, TA=TA,
                              cleanflicker_skip=nsclean_skip, save_plots=save_plots)

def run_stage2_nirspec(rate_files, output_dir, skip_cubes=True, overwrite=False, TA=False, cleanflicker_skip=True, save_plots=True):
    """
    Run pipeline stage 2 for JWST/NIRSpec, with some customizations for reductions
    intended to be used with breads for IFU high contrast


    Parameters
    ----------
    rate_files : list of strings
        Filenames of stage 1 rate files to reduce
    output_dir : string
        Directory path for where to put the output files
    skip_cubes : bool
        Skip the cube building step, since we do not need the interpolated cubes for our purposes. Default is True.
    overwrite : bool
        Re-reduce and overwrite outputs whether they existed already. If the processed files already exists, it returns the file list of those.
        Default is to SKIP re-reducing anything already reduced.
    TA : bool
        If these are target acquisition images, then skip the pathloss step, since pathloss correction is not appropriate for TA images. Default is False.
    cleanflicker_skip : bool
        Skip the clean flicker noise step, which is designed to clean 1/f noise from the data; This step can be very slow.
        Note that BREADS has its own customized noise cleaning procedure.
        Default is True.
    save_plots : bool
            Whether to save plots of the stage 2 results. Default is True.

    Returns
    -------
    cal_files : list of str
        List of cal filenames produced by stage 2
    """
    from jwst.pipeline import Spec2Pipeline

    # We need to check that the desired output directories exist, and if not create them
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

        # Start a timer to keep track of runtime
    time0 = time.perf_counter()

    cal_files = []
    N_files_processed = 0
    for fid, rate_file in enumerate(rate_files):
        print(f"Stage 2 Processing file {fid + 1} of {len(rate_files)}.")

        # Setting up steps and running the Spec2 portion of the pipeline.

        outname = os.path.join(output_dir, os.path.basename(rate_file).replace('rate.fits', 'cal.fits'))
        cal_files.append(outname)
        if os.path.exists(outname) and not overwrite:
            print(f"\tStage 2 Output file {os.path.basename(outname)} already exists in output dir;\n\tskipping {os.path.basename(rate_file)}.")
            continue
        N_files_processed += 1

        spec2 = Spec2Pipeline()

        pathloss_skip = TA  # For target acq images, skip the pathloss step, otherwise don't skip it.

        step_parameters = {
            # spec2.assign_wcs.skip = False
            # spec2.bkg_subtract.skip = False
            # spec2.imprint_subtract.skip = False
            # spec2.msa_flagging.skip = False
            # # spec2.srctype.source_type = 'POINT'
            # spec2.flat_field.skip = False
            # spec2.pathloss.skip = False
            'clean_flicker_noise':{'skip':cleanflicker_skip},
            'pathloss':{'skip':pathloss_skip},
            # spec2.photom.skip = False
            'cube_build': {'skip': skip_cubes},  # We do not want or need interpolated cubes
            'extract_1d': {'skip': True},
            # spec3.cube_build.coord_system = 'skyalign'
            # spec2.cube_build.coord_system='ifualign'
        }
        spec2.save_bsub = True

        # choose what results to save and from what steps
        spec2.call(rate_file, save_results=True, output_dir=output_dir,
                   steps=step_parameters)

        # Print out the time benchmark
        time1 = time.perf_counter()
        print(f"\tStage 2 Runtime so far: {time1 - time0:0.4f} seconds")

    time1 = time.perf_counter()
    print(f"Stage 2 Total Runtime: {time1 - time0:0.4f} seconds")
    if save_plots and N_files_processed > 0:
        breads.jwst_tools.plotting.plot_2d_image_set(cal_files,
                                                     output_dir = output_dir,
                                                     suptitle="Stage 2 pipeline reduction results",
                                                     plot_label = 'stage2')

    return cal_files


###########################################################################
#  Functions for noise cleaning


def fm_column_background(nonlin_paras, cubeobj, nodes=20,
                         fix_parameters=None,
                         return_where_finite=False,
                         regularization=None,
                         badpixfraction=0.75,
                         M_spline=None,
                         spline_reg_std=1.0):
    """
    BREADS forward model column background, for use in forward_model_noise_clean
    """
    if fix_parameters is not None:
        _nonlin_paras = np.array(fix_parameters)
        _nonlin_paras[np.where(np.array(fix_parameters) is None)] = nonlin_paras
    else:
        _nonlin_paras = nonlin_paras

    if M_spline is None:
        if type(nodes) is int:
            n_nodes = nodes
            x_knots = np.linspace(0, np.size(cubeobj.data), n_nodes, endpoint=True).tolist()
        elif type(nodes) is list or type(nodes) is np.ndarray:
            x_knots = nodes
            if type(nodes[0]) is list or type(nodes[0]) is np.ndarray:
                n_nodes = np.sum([np.size(n) for n in nodes])
            else:
                n_nodes = np.size(nodes)
        else:
            raise ValueError("Unknown format for nodes.")
    else:
        n_nodes = M_spline.shape[1]

    # Number of linear parameters
    n_linpara = n_nodes

    data = cubeobj.data
    noise = cubeobj.noise
    bad_pixels = cubeobj.bad_pixels

    where_trace_finite = np.where(np.isfinite(data) * np.isfinite(bad_pixels) * (noise != 0))
    d = data[where_trace_finite]
    s = noise[where_trace_finite]

    if np.size(where_trace_finite[0]) <= (1 - badpixfraction) * np.size(data):
        # don't bother to do a fit if there are too many bad pixels
        return np.array([]), np.array([]).reshape(0, n_linpara), np.array([])
    else:
        x = np.arange(np.size(cubeobj.data))
        if M_spline is None:
            m_spline = get_spline_model(x_knots, x, spline_degree=3)
        else:
            m_spline = copy(M_spline)

        m_spline = m_spline[where_trace_finite[0], :]

        extra_outputs = {}
        if regularization == "default":
            s_reg = np.zeros(n_nodes) + spline_reg_std
            d_reg = np.zeros(n_nodes)
            extra_outputs["regularization"] = (d_reg, s_reg)
        elif regularization == "user":
            raise Exception("user defined regularisation not yet implemented")
            extra_outputs["regularization"] = (d_reg, s_reg)

        if return_where_finite:
            extra_outputs["where_trace_finite"] = where_trace_finite

        if len(extra_outputs) >= 1:
            return d, m_spline, s, extra_outputs
        else:
            return d, m_spline, s

def _task_1f_noise_nirspec_col(args):
    data_col, noise_col, bad_pixels_col,N_nodes,m_spline_column = args

    data = Instrument()
    data.data =copy(data_col)
    data.data[np.where(~np.isfinite(data.data))] = 0
    data.noise = copy(noise_col)
    data.noise[np.where(data.noise == 0)] = np.nanmedian(data.noise)
    data.bad_pixels = copy(bad_pixels_col)

    nonlin_paras = []
    fm_paras = {"badpixfraction": 0.99, "nodes": N_nodes, "fix_parameters": None,
                "regularization": "default", "M_spline": m_spline_column}

    # The 1/f noise spline fit is done twice.
    # The first time is only to identify bad pixel better from sigma clipping of the residuals.

    ####
    # First fit:
    out_log_prob, rchi2, linparas, linparas_err = fitfm(nonlin_paras, data, fm_column_background, fm_paras,
                                                        scale_noise=False)
    if not np.isfinite(out_log_prob):
        return np.zeros(data_col.shape)
    d_masked, m, s, extra_outputs = fm_column_background(nonlin_paras, data, return_where_finite=True,
                                                         **fm_paras)
    where_finite = extra_outputs["where_trace_finite"]
    data.bad_pixels = np.ones(data.data.shape)
    d, m, s, _ = fm_column_background(nonlin_paras, data, return_where_finite=True, **fm_paras)
    d_masked_canvas = np.zeros(data_col.shape) + np.nan
    d_masked_canvas[where_finite] = d_masked

    m = np.dot(m, linparas)
    res = d_masked_canvas - m

    # identify outliers
    mad = median_abs_deviation(res[where_finite])
    data.bad_pixels = bad_pixels_col
    data.bad_pixels[np.where(np.abs(res) > 5 * mad)] = np.nan

    ####
    # After identifying residual outliers, redo the fit a final time:
    out_log_prob, rchi2, linparas, linparas_err = fitfm(nonlin_paras, data, fm_column_background, fm_paras,
                                                        scale_noise=False)
    if not np.isfinite(out_log_prob):
        return np.zeros(data_col.shape)
    d_masked, m, s, extra_outputs = fm_column_background(nonlin_paras, data, return_where_finite=True,
                                                         **fm_paras)
    where_finite = extra_outputs["where_trace_finite"]
    data.bad_pixels = np.ones(data.data.shape)
    d, m, s, _ = fm_column_background(nonlin_paras, data, return_where_finite=True, **fm_paras)
    d_masked_canvas = np.zeros(data_col.shape) + np.nan
    d_masked_canvas[where_finite] = d_masked

    return np.dot(m, linparas)

def fit_1f_noise_nirspec(im,noise,bkg_bad_pixels,N_nodes=40,mppool=None):
    # Define the column-wise spline model, which will be used to model the smooth vertical 1/f noise variation
    # We are using the BREADS's linear least square forward modeling framework
    x = np.arange(2048)
    x_knots_column = np.linspace(0, 2048, N_nodes, endpoint=True).tolist()
    m_spline_column = get_spline_model(x_knots_column, x, spline_degree=3)

    model_1f_noise = np.full_like(im,np.nan)
    if mppool is None:
        for colid in range(im.shape[1]):
            args = (im[:, colid],noise[:, colid],bkg_bad_pixels[:, colid],N_nodes,m_spline_column)
            noise_1f_col = _task_1f_noise_nirspec_col(args)
            model_1f_noise[:,colid] = noise_1f_col
    else:
        results = list(tqdm(
            mppool.imap(_task_1f_noise_nirspec_col,zip(im.T,noise.T,bkg_bad_pixels.T,
                                                       itertools.repeat(N_nodes),
                                                       itertools.repeat(m_spline_column))),
            total=im.shape[1]
        ))

        for colid, col_result in enumerate(results):
            model_1f_noise[:, colid] = col_result

    return model_1f_noise



def _charge_transfer_model_col_fun(psf_image_col, tau=None, cutoff=0, power=1, kernel_radius=256):

    extra_charges = psf_image_col - cutoff
    extra_charges[np.where((extra_charges < 0) | ~np.isfinite(extra_charges))] = 0.0


    vecy = np.arange(3 * (kernel_radius * 2 + 1))
    vecy -= vecy[np.size(vecy) // 2]
    vecy = np.abs(vecy)
    if tau is None:
        kernel = 1 / (vecy ** power)
    else:
        kernel = np.exp(-vecy / (tau / np.log(2))) / (vecy ** power)
    kernel[np.size(vecy) // 2] = 0.0

    ny = np.size(extra_charges)
    _charge_transfer_model = np.zeros(ny)

    _charge_transfer_model = convolve1d(extra_charges, weights=kernel / np.nansum(kernel), mode='constant')

    return _charge_transfer_model[::3]


def _chi2_charge_transfer_col(paras, data_col, bad_pixels_col, new_model_col, tau, kernel_radius=1024):
    cutoff, power = paras
    if tau is not None and tau <= 0:
        return np.inf
    if cutoff < 0:
        return np.inf
    if power <= 1.6 or power > 2.0:
        return np.inf
    _charge_transfer_model = _charge_transfer_model_col_fun(new_model_col, tau=tau, cutoff=cutoff, power=power,
                                                            kernel_radius=kernel_radius)
    scale = np.nansum(data_col * _charge_transfer_model) / np.nansum((_charge_transfer_model * bad_pixels_col) ** 2)
    res = data_col - scale * _charge_transfer_model
    chi2 = np.nansum(res ** 2)
    return chi2


def _task_charge_transfer_nirspec_col(_args):
    data_col, bkg_bad_pixels_col, new_model_col,noise_ratio = _args
    if noise_ratio <0.8:
        return np.zeros(np.shape(data_col))
    if np.nansum(new_model_col) == 0 or np.nansum(bkg_bad_pixels_col) == 0:
        return np.zeros(np.shape(data_col))
    tau = None  # No exponential decay term
    cutoff0 = np.min([2000,np.nanmax(new_model_col)/2.,10*np.nanmax(np.abs(data_col*bkg_bad_pixels_col))])
    # print(cutoff0,[2000,np.nanmax(new_model_col)/2.,10*np.nanmax(np.abs(data_col*bkg_bad_pixels_col))])
    # cutoff0=3
    power0 = 1.8
    kernel_radius=1024

    paras0 = [cutoff0, power0]  # your initial guesses
    cutoff, power = paras0
    paras0 = np.array(paras0)
    simplex_init_steps = [cutoff0 / 2., power0 / 5.]
    initial_simplex = np.concatenate([paras0[None, :], paras0[None, :] + np.diag(simplex_init_steps)], axis=0)

    result = minimize(_chi2_charge_transfer_col, paras0, args=(data_col, bkg_bad_pixels_col, new_model_col,
                                                               tau, kernel_radius), method='Nelder-Mead',
                      options={"maxiter": 1e2, "initial_simplex": initial_simplex, "disp": False})
    cutoff, power = result.x
    # print(result.nit, [cutoff, power],paras0)
    # if power < 1.0 or power > 3.0:
    #     return np.zeros(np.shape(data_col))
    charge_transfer_model_col = _charge_transfer_model_col_fun(new_model_col, tau=tau, cutoff=cutoff, power=power,kernel_radius=kernel_radius)
    denum = np.nansum((charge_transfer_model_col * bkg_bad_pixels_col) ** 2)
    if denum > 0:
        scale = np.nansum(data_col * charge_transfer_model_col) / denum
    else:
        scale = 0.0

    return scale * charge_transfer_model_col,[cutoff, power]


def fit_charge_transfer_nirspec(rate_dataobj,bkg_bad_pixels,rn_noise,poisson_noise,targetname=None,mppool=None,use_stpsf=False,use_breadspsf=True,init_centroid=None):
    im = copy(rate_dataobj.data)
    wvs = rate_dataobj.wavelengths
    ny_ori, nx_ori = im.shape

    grating = rate_dataobj.priheader['GRATING'].strip()
    detector = rate_dataobj.priheader['DETECTOR'].strip().lower()
    wv_sampling = default.wv_sampling_dict[grating][detector]
    wv_nodes = default.nodes_dict[grating][detector]

    rate_dataobj.default_filenames["compute_med_filt_badpix"] = rate_dataobj.default_filenames["compute_med_filt_badpix"].replace(".fits","_rate_cleaning_tmp.fits")
    rate_dataobj.default_filenames["compute_starspectrum_contnorm"] = rate_dataobj.default_filenames["compute_starspectrum_contnorm"].replace(".fits","_rate_cleaning_tmp.fits")
    rate_dataobj.default_filenames["compute_starsubtraction"] = rate_dataobj.default_filenames["compute_starsubtraction"].replace(".fits","_rate_cleaning_tmp.fits")
    rate_dataobj.default_filenames["compute_advanced_badpix"] = rate_dataobj.default_filenames["compute_advanced_badpix"].replace(".fits","_rate_cleaning_tmp.fits")
    rate_dataobj.default_filenames["compute_interpdata_regwvs"] = rate_dataobj.default_filenames["compute_interpdata_regwvs"].replace(".fits","_rate_cleaning_tmp.fits")

    preproc_task_list = [
        ["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}, False, True],
        ["compute_coordinates_arrays", {'targname': targetname}, False, True],
        ["compute_starspectrum_contnorm", {"wv_nodes": wv_nodes, "threshold_badpix": 100, "iterative": True,
                                           "mppool": mppool}, True, True],
        ["compute_advanced_badpix", {"threshold_badpix": 10, "mppool": mppool, "iterative": False}, False, True],
        ["compute_interpdata_regwvs", {"wv_sampling": wv_sampling}, False, True]]
    rate_dataobj.run_preproc_list(preproc_task_list=preproc_task_list)

    if 1: # Add intermediate rows to limit undersampling issue. Similar idea to Law+2026.
        ny0,nx0 = rate_dataobj.data.shape
        ny_hd, nx_hd = ny0*3, nx0
        ind_vec_hd = np.arange(ny_hd)
        ind_vec = ind_vec_hd[::3]
        x_new = np.full((ny_hd, nx_hd), np.nan)
        y_new = np.full((ny_hd, nx_hd), np.nan)
        w_new = np.full((ny_hd, nx_hd), np.nan)
        d_new = np.full((ny_hd, nx_hd), np.nan)
        e_new = np.full((ny_hd, nx_hd), np.nan)
        bp_new = np.full((ny_hd, nx_hd), np.nan)
        for colid in range(nx0):
            x_new[:, colid] = np.interp(ind_vec_hd,ind_vec,rate_dataobj.x[:, colid],left=np.nan, right=np.nan)
            y_new[:, colid] = np.interp(ind_vec_hd,ind_vec,rate_dataobj.y[:, colid],left=np.nan, right=np.nan)
            w_new[:, colid] = np.interp(ind_vec_hd,ind_vec,rate_dataobj.wavelengths[:, colid],left=np.nan, right=np.nan)
        d_new[::3,:] = rate_dataobj.data
        e_new[::3,:] = rate_dataobj.noise
        bp_new[::3,:] = rate_dataobj.bad_pixels
        rate_dataobj.x = x_new
        rate_dataobj.y = y_new
        rate_dataobj.wavelengths = w_new
        rate_dataobj.data = d_new
        rate_dataobj.noise = e_new
        rate_dataobj.bad_pixels = bp_new

    # Derive the first guess centroids from a narrow wavelength region
    if init_centroid is None:
        frac_init_bandpass = 0.005
        IWA_init,OWA_init = 0.0, 1.0
        bandpass = rate_dataobj.wv_sampling[-1] - rate_dataobj.wv_sampling[0]
        midwv = (rate_dataobj.wv_sampling[-1] + rate_dataobj.wv_sampling[0]) / 2.0
        debug_wv_range = [midwv - frac_init_bandpass * bandpass, midwv + frac_init_bandpass * bandpass]

        # _fitpsf_filename = os.path.join(rate_dataobj.utils_dir, os.path.basename(rate_dataobj.filename).replace(".fits","_fitspsf_init_charge_transfer.fits"))
        bestfit_paras, _, _, _ = fitpsf(rate_dataobj, use_stpsf=use_stpsf,use_breadspsf=use_breadspsf,
                                        IWA=IWA_init, OWA=OWA_init, out_filename=None,
                                        overwrite=False, mppool=mppool, debug_wv_range=debug_wv_range,
                                                               poly_deg_coords=0)
        init_centroid = (np.nanmedian(bestfit_paras[0, :, 2]), np.nanmedian(bestfit_paras[0, :, 3]))
    ######
    # Fit the whole wavelength range
    IWA,OWA = 0.0, 1.0
    _fitpsf_filename = os.path.join(rate_dataobj.utils_dir, os.path.basename(rate_dataobj.filename).replace(".fits","_fitspsf_charge_transfer.fits"))
    bestfit_paras, data, bestfit_model, residuals = fitpsf(rate_dataobj, use_stpsf=use_stpsf,use_breadspsf=use_breadspsf,
                                                           init_centroid=init_centroid,
                                                           IWA=IWA, OWA=OWA, out_filename=_fitpsf_filename,
                                                           overwrite=False, mppool=mppool, debug_wv_range=None,
                                                               poly_deg_coords=1)

    # reinterpolate back onto the data point cloud sampling (instead of "regwvs")
    w_ori_new = np.full((ny_hd, nx_ori), np.nan)
    for colid in range(nx_ori):
        w_ori_new[:, colid] = np.interp(ind_vec_hd, ind_vec, wvs[:, colid], left=np.nan, right=np.nan)
    new_model = np.full((ny_hd, nx_ori), np.nan)
    for rowid in range(ny_hd):
        new_model[rowid, :] = np.interp(w_ori_new[rowid, :],rate_dataobj.wv_sampling,bestfit_model[rowid, :],
                                        left=np.nan, right=np.nan)

    noise_ratio = poisson_noise/rn_noise*bkg_bad_pixels
    # kernel = np.ones((50,50))
    # noise_ratio_smooth = correlate(noise_ratio, kernel, mode='constant', cval=np.nan)
    # noise_ratio_smooth = generic_filter(noise_ratio,
    #                                     function = np.nanmedian,
    #                                     size = 50,  # equivalent to your 50x50 kernel
    #                                     mode = 'constant',
    #                                     cval = np.nan)  # pad edges with NaN (same as your original)

    # convolve handles NaNs natively - NaN pixels are interpolated over
    noise_ratio_smooth = convolve(noise_ratio, Box2DKernel(10), boundary='fill', fill_value=np.nan)
    noise_ratio_vec = np.nanmax(noise_ratio_smooth, axis=0)
    window_size = 100
    noise_ratio_vec_smooth = generic_filter(noise_ratio_vec, np.nanmedian, size=window_size)
    # plt.figure()
    # plt.imshow(w_ori_new,origin="lower")
    # plt.figure()
    # plt.imshow(new_model,origin="lower")
    # plt.figure()
    # plt.imshow(bestfit_model,origin="lower")
    # plt.show()

    data = im * bkg_bad_pixels
    charge_transfer_model = np.zeros(data.shape)

    bestfit_paras_arr = np.full([2,data.shape[1]], np.nan)
    if mppool is None:
        for colid in range(data.shape[1]):
        # for colid in [484]:
            _args = (data[:,colid],bkg_bad_pixels[:,colid],new_model[:,colid],noise_ratio_vec_smooth[colid])
            charge_transfer_model_col,paras = _task_charge_transfer_nirspec_col(_args)
            charge_transfer_model[:,colid] = charge_transfer_model_col
            print(colid,paras)
            bestfit_paras_arr[:,colid] = paras
            # bestfit_paras_arr[0,:] => cutoff for each column
            # bestfit_paras_arr[1,:] => power
    else:
        results = list(tqdm(
            mppool.imap(_task_charge_transfer_nirspec_col,zip(data.T,bkg_bad_pixels.T,new_model.T,noise_ratio_vec_smooth)),
            total=data.shape[1]
        ))

        for colid, col_result in enumerate(results):
            charge_transfer_model[:, colid] = col_result[0]
            bestfit_paras_arr[:, colid] = col_result[1]



    # for colid in [484]:
    #     plt.figure(figsize=(12,8))
    #     plt.title(f"colid: {colid}")
    #     plt.plot(im[:,colid], label="im",linestyle="-")
    #     plt.plot(data[:,colid], label="data",linestyle="--")
    #     plt.plot(new_model[::3,colid], label="new_model",linestyle="-")
    #     plt.plot(charge_transfer_model[:,colid], label="charge_transfer_model",linestyle="--")
    #     plt.plot(data[:,colid]-charge_transfer_model[:,colid], label="res",linestyle="--")
    #     plt.legend()
    # plt.figure()
    # plt.plot(noise_ratio_vec_smooth)
    # print("coucou3")
    # plt.show()
    if 0:
        cutoff_vec = copy(bestfit_paras_arr[0, :])
        power_vec = copy(bestfit_paras_arr[1, :])
        if "nrs1" in detector:
            cutoff_vec[0:300] = np.nan
            cutoff_vec[1900::] = np.nan
            power_vec[0:300] = np.nan
            power_vec[1900::] = np.nan
        else:
            cutoff_vec[0:150] = np.nan
            cutoff_vec[(2048-300)::] = np.nan
            power_vec[0:150] = np.nan
            power_vec[(2048-300)::] = np.nan
        window_size = 30
        cutoff_smooth = generic_filter(cutoff_vec, np.nanmedian, size=window_size)
        power_smooth = generic_filter(power_vec, np.nanmedian, size=window_size)
        if "nrs1" in detector:
            cutoff_smooth[0:300] = np.full_like(cutoff_smooth[0:300],cutoff_smooth[300])
            cutoff_smooth[1900::] = np.full_like(cutoff_smooth[1900::],cutoff_smooth[1900-1])
            power_smooth[0:300] = np.full_like(power_smooth[0:300],power_smooth[300])
            power_smooth[1900::] = np.full_like(power_smooth[1900::],power_smooth[1900-1])
        else:
            cutoff_smooth[0:150] = np.full_like(cutoff_smooth[0:150],cutoff_smooth[150])
            cutoff_smooth[(2048-300)::] = np.full_like(cutoff_smooth[(2048-300)::],cutoff_smooth[2048-300-1])
            power_smooth[0:150] = np.full_like(power_smooth[0:150],power_smooth[150])
            power_smooth[(2048-300)::] = np.full_like(power_smooth[(2048-300)::],power_smooth[2048-300-1])
        plt.figure()
        plt.subplot(2,1,1)
        plt.title("cutoff")
        plt.plot(bestfit_paras_arr[0,:])
        plt.plot(cutoff_smooth)
        plt.subplot(2,1,2)
        plt.title("power")
        plt.plot(bestfit_paras_arr[1,:])
        plt.plot(power_smooth)
        # plt.show()
        for colid in range(data.shape[1]):
            if np.nansum(new_model[:,colid]) == 0 or np.nansum(bkg_bad_pixels[:,colid]) == 0:
                charge_transfer_model[:,colid] = np.zeros(np.shape(data[:,colid]))
            print(colid)
            cutoff,power = cutoff_smooth[colid],power_smooth[colid]
            charge_transfer_model_col = _charge_transfer_model_col_fun(new_model[:,colid], tau=None, cutoff=cutoff, power=power,kernel_radius=1024)
            denum = np.nansum((charge_transfer_model_col * bkg_bad_pixels[:,colid]) ** 2)
            if denum > 0:
                scale = np.nansum(data[:,colid]* bkg_bad_pixels[:,colid] * charge_transfer_model_col) / denum
            else:
                scale = 0.0
            charge_transfer_model[:,colid] = scale * charge_transfer_model_col

    # for colid in [500]:
    #     plt.figure(figsize=(12,8))
    #     plt.title(f"colid: {colid}")
    #     plt.plot(im[:,colid], label="im",linestyle="-")
    #     plt.plot(data[:,colid], label="data",linestyle="--")
    #     plt.plot(new_model[::3,colid], label="new_model",linestyle="-")
    #     plt.plot(charge_transfer_model[:,colid], label="charge_transfer_model",linestyle="--")
    #     plt.plot(data[:,colid]-charge_transfer_model[:,colid], label="res",linestyle="--")
    #     plt.legend()

    # model_too_low_to_matter = np.where(np.nanmax(charge_transfer_model/rn_noise,axis=0)<100)[0]
    # charge_transfer_model[:,model_too_low_to_matter] = 0

    # print("ratio",,np.nanmax(new_model[:,750],axis=0),np.nanmax(rn_noise[:,750],axis=0))
    # print("ratio",np.nanmax(charge_transfer_model[:,270]/rn_noise[:,270],axis=0),np.nanmax(new_model[:,270],axis=0),np.nanmax(rn_noise[:,270],axis=0))

    # for colid in [100,1240,1242]:
    #     plt.figure(figsize=(12,8))
    #     plt.title(f"colid: {colid}")
    #     plt.plot(im[:,colid], label="im",linestyle="-")
    #     plt.plot(data[:,colid], label="data",linestyle="--")
    #     plt.plot(new_model[::3,colid], label="new_model",linestyle="-")
    #     plt.plot(charge_transfer_model[:,colid], label="charge_transfer_model",linestyle="--")
    #     plt.plot(data[:,colid]-charge_transfer_model[:,colid], label="res",linestyle="--")
    #     plt.legend()

    # print("coucou2")
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.title("cutoff")
    # plt.plot(bestfit_paras_arr[0,:])
    # plt.subplot(2,1,2)
    # plt.title("power")
    # plt.plot(bestfit_paras_arr[1,:])
    #

    # plt.figure(figsize=(12,8))
    # plt.plot(im[:,459], label="ori im")
    # plt.plot(new_model[::3,459], label="psf model")
    # plt.plot(data[:,459], label="data",linestyle="--")
    # plt.plot(charge_transfer_model[:,459], label="charge_transfer_model",linestyle="--")
    # plt.legend()
    # plt.show()
    # exit()

    return charge_transfer_model

def _get_bkg_bad_pixels(rate_dataobj,cal_trace_id_map,dq_rate,extend_sat=1):
    # kernel = np.ones((3, 3))
    # mask = copy(cal_trace_id_map)
    # mask[np.where(~np.isfinite(cal_trace_id_map))] = 0
    # mask = correlate(mask, kernel, mode='constant', cval=0.0)
    # mask[np.where(mask==0)] = np.nan

    # plt.subplot(1,2,1)
    # plt.imshow(cal_trace_id_map,origin="lower")
    # plt.subplot(1,2,2)
    # plt.imshow(mask,origin="lower")
    # plt.show()

    detector = rate_dataobj.priheader['DETECTOR'].strip().lower()
    im = rate_dataobj.data
    noise = rate_dataobj.noise

    # Simplifying bad pixel map following convention in this package as: nan = bad, 1 = good
    bkg_bad_pixels = np.full(rate_dataobj.data.shape, np.nan)  # array full of nans
    # We select only the background pixels:
    bkg_bad_pixels[np.where(np.isnan(cal_trace_id_map))] = 1  # every pixel that is not in a cal slice is actually good here
    # Pixels marked as "do not use" are marked as bad (nan = bad, 1 = good):
    untangle_dq_rate = untangle_dq(dq_rate, verbose=True)
    do_not_use_dq_rate = untangle_dq_rate[0, :, :]
    saturated_dq_rate = untangle_dq_rate[1, :, :]
    bkg_bad_pixels[np.where(do_not_use_dq_rate)] = np.nan
    bkg_bad_pixels[np.where(np.isnan(im))] = np.nan
    # Removing any data with zero noise
    where_zero_noise = np.where(noise == 0)
    noise[where_zero_noise] = np.nan
    bkg_bad_pixels[where_zero_noise] = np.nan


    if "nrs1" in detector:
        finite_mask = np.isfinite(cal_trace_id_map[:, 0:450])
        has_finite = finite_mask.any(axis=1)
        # argmax on bool finds the first True (leftmost finite)
        id_to_mask = np.argmax(finite_mask, axis=1)  # shape (nrows,)
        col_idx = np.arange(450)
        mask2d = col_idx[None, :] < id_to_mask[:, None]  # shape (nrows, 450)
        mask2d &= has_finite[:, None]
        bkg_bad_pixels[:, 0:450][mask2d] = np.nan

    elif "nrs2" in detector:
        finite_mask = np.isfinite(cal_trace_id_map[:, 1550:])
        has_finite = finite_mask.any(axis=1)
        # flip to find last True via argmax on reversed array
        id_to_mask = (finite_mask.shape[1] - 1) - np.argmax(finite_mask[:, ::-1], axis=1)
        col_idx = np.arange(finite_mask.shape[1])
        mask2d = col_idx[None, :] > id_to_mask[:, None]
        mask2d &= has_finite[:, None]
        bkg_bad_pixels[:, 1550:][mask2d] = np.nan

    # Extend the slices masks to the edge of the detector, because there is still real flux there at the edge of the spectral filter.
    # There is some hard coded stuff here, but hopefully nothing too dangerous.
    # if "nrs1" in detector:
    #     for rowid in range(im.shape[0]):
    #         finite_ids = np.where(np.isfinite(_cal_trace_id_map[rowid, 0:450]))[0]
    #         if len(finite_ids) != 0:
    #             id_to_mask = np.min(finite_ids)
    #             bkg_bad_pixels[rowid, 0:id_to_mask] = np.nan
    # elif "nrs2" in detector:
    #     for rowid in range(im.shape[0]):
    #         finite_ids = np.where(np.isfinite(_cal_trace_id_map[rowid, 1550::]))[0]
    #         if len(finite_ids) != 0:
    #             id_to_mask = np.max(finite_ids)
    #             bkg_bad_pixels[rowid, 1550 + id_to_mask::] = np.nan


    # identify additional bad pixels from sliding median window and sigma clipping
    mad_threshold = 5
    window_size = 50
    new_badpix = np.ones(bkg_bad_pixels.shape)
    for rowid in range(bkg_bad_pixels.shape[0]):
        row_data = im[rowid, :] - generic_filter(im[rowid, :] * bkg_bad_pixels[rowid, :], np.nanmedian, size=window_size)
        row_data_masking = row_data / median_abs_deviation(row_data[np.where(np.isfinite(bkg_bad_pixels[rowid, :]))])
        new_badpix[rowid, np.where((row_data_masking > mad_threshold))[0]] = np.nan
    bkg_bad_pixels *= new_badpix

    # plt.figure()
    # plt.imshow(bkg_bad_pixels,origin="lower")
    # plt.figure()
    # plt.imshow(saturated_dq_rate,origin="lower")

    kernel = np.ones((2*extend_sat+1, 2*extend_sat+2))
    mask = saturated_dq_rate.astype(float)
    mask = correlate(mask, kernel, mode='constant', cval=0.0)
    bkg_bad_pixels[np.where(mask!=0)] = np.nan

    # plt.figure()
    # plt.imshow(mask,origin="lower")
    # plt.figure()
    # plt.imshow(bkg_bad_pixels,origin="lower")
    # plt.show()

    return bkg_bad_pixels

def clean_rate_nirspec_per_file(rate_file, cal_file_dir, clean_dir, N_nodes=40,
                                clean_1f_noise=True,model_charge_transfer=False,
                              utils_dir=None, init_centroid=None,mppool=None,targetname=None,extend_sat=2,
                                verbose=True):
    """
    Remove the 1/f noise  and/or the charge transferfrom rate files of the NIRSpec IFU.
    Inspired by NSClean but different implementation using column-wise splines.
    The cleaned rate.fits files are saved in output_dir.
    An initial reduction of the rate.fits and cal.fits (stage 1 and 2) need to be available before running this function.

    The way it works:

    - subtraction done on rate.fits
    - Use the cal.fits to retrieve the mask of the IFU slices
    - Fit detector columns one at time. I just fit a smooth continuum (using my splines) to the masked detector
    - column, and also masking the region around the star more aggressively I believe
    - subtract the fitted continuum
    - Save new rate.fits

    Parameters
    ----------
    rate_file : string
        Filename of rate file to reduce
    stage2_dir : string
        Directory where the cal.fits files (from the stage 2 pipeline) corresponding to the same input rate files can be found.
        The function will
    output_dir : string
        Directory path for where to put the output files
    N_nodes : integer
        Number of spline nodes to use for calculating the charge transfer. Default is 40.
    model_charge_transfer : boolean
        Model the charge transfer originating from the saturated pixels. Default is False.
    utils_dir : string
        Directory where the BREADS utils files will be loaded/saved.
    init_centroid : tuple
        (ra, dec) in arcseconds as a best rough guess of the coordinate of the central star.

    Returns
    -------
    new_rate_file : string
        Filename of the new rate file with the 1/f noise and optionally charge transfer cleaned.

    """
    basename = os.path.basename(rate_file)

    # Look for the cal file corresponding to the rate file being processed.
    # We will use the information stored in the cal file later: eg the mask of science slices or the wcs header.
    cal_filename = os.path.join(cal_file_dir, basename.replace("_rate.fits", "_cal.fits"))
    if len(glob(cal_filename)) == 0:
        raise Exception("Could not find the corresponding cal file. Please run stage 2 without cleaning first.")

    # Get data. Read rate.fits file
    hdul = fits.open(rate_file)
    priheader = hdul[0].header
    im = hdul["SCI"].data
    new_rate_im = copy(im)
    noise = hdul["ERR"].data
    rn_noise = np.sqrt(hdul["VAR_RNOISE"].data)
    poisson_noise = np.sqrt(hdul["VAR_POISSON"].data)
    dq_rate = hdul["DQ"].data
    ny, nx = im.shape

    rate_dataobj = JWSTNirspec_cal(cal_filename, utils_dir=utils_dir)
    preproc_task_list = [["compute_coordinates_arrays", {'targname': targetname}, True, True]]
    rate_dataobj.run_preproc_list(preproc_task_list=preproc_task_list)
    cal_trace_id_map = rate_dataobj.trace_id_map
    # cal_im = copy(rate_dataobj.data)
    rate_dataobj.data = copy(im)
    rate_dataobj.noise = noise

    bkg_bad_pixels = _get_bkg_bad_pixels(rate_dataobj, cal_trace_id_map, dq_rate, extend_sat=extend_sat)
    # _tmp = os.path.join(rate_dataobj.utils_dir,os.path.basename(rate_dataobj.filename)+"_bad_pixels_tmp.npy")
    # if not os.path.exists(_tmp):
    #     bkg_bad_pixels = _get_bkg_bad_pixels(rate_dataobj,cal_trace_id_map, dq_rate,extend_sat=extend_sat)
    #     np.save(_tmp,bkg_bad_pixels)
    # else:
    #     bkg_bad_pixels = np.load(_tmp)

    priheader.add_history('Processed with BREADS (https://github.com/jruffio/breads)')

    if model_charge_transfer:
        if verbose:
            print("Modeling and subtracting charge transfer from saturated pixels...")
        charge_transfer_model = fit_charge_transfer_nirspec(rate_dataobj,bkg_bad_pixels,rn_noise,poisson_noise,targetname=targetname,mppool=mppool,
                                                            use_stpsf=False,use_breadspsf=True,init_centroid=init_centroid)
        priheader.add_history('Subtracted charge transfer')

        new_rate_im -= charge_transfer_model

    # plt.figure()
    # plt.imshow(charge_transfer_model)
    # plt.figure()
    # plt.imshow(new_rate_im)
    # plt.show()
    if clean_1f_noise:
        if verbose:
            print("Modeling and subtracting 1/f noise...")
        model_1f_noise = fit_1f_noise_nirspec(new_rate_im,noise,bkg_bad_pixels,N_nodes,mppool=mppool)
        priheader.add_history('Applied 1/f noise subtraction using column-wise spline')

        new_rate_im -= model_1f_noise

    # add breads processing stamps in the fits header
    priheader['BREDDATE'] = (datetime.date.today().isoformat(), 'Date of BREADS processing')
    breads_commit = get_breads_commit()
    priheader['BREDCOMI'] = (breads_commit[:40], 'BREADS git commit hash')

    # Save cleaned file
    hdul[0].header = priheader
    hdul["SCI"].data = new_rate_im
    new_rate_file = os.path.join(clean_dir, os.path.basename(rate_file))
    hdul.writeto(new_rate_file, overwrite=True)
    hdul.close()

    return new_rate_file

def clean_rate_nirspec(rate_files, stage2_dir, output_dir, N_nodes=40, clean_1f_noise=True,model_charge_transfer=False,
                    utils_dir=None, init_centroid=None, overwrite=False, save_plots=True,targetname=None, mppool=None,
                       extend_sat=2):
    """
    Remove the 1/f noise and optionally the charge transfer from rate files. The cleaned rate.fits files are saved in output_dir.
    An initial reduction of the rate.fits and cal.fits (stage 1 and 2) need to be available before running this function.


    Parameters
    ----------
    rate_files : list of strings
        Filenames of rate files to reduce
    stage2_dir : string
        Directory where the cal.fits files (from the stage 2 pipeline) corresponding to the same input rate files can be found.
        The function will
    output_dir : string
        Directory path for where to put the output files
    N_nodes : integer
        Number of spline nodes to use for calculating the charge transfer. Default is 40.
    model_charge_transfer : boolean
        Model the charge transfer originating from the saturated pixels. Default is False.
    utils_dir : string
        Directory where the BREADS utils files will be loaded/saved.
    init_centroid : tuple
        (ra, dec) in arcseconds as a best rough guess of the coordinate of the central star.
    overwrite : bool
        Re-reduce and overwrite outputs whether they existed already. If the processed files already exists, it returns the file list of those.
        Default is to SKIP re-reducing anything already reduced.
    save_plots : bool
        Whether to save the plots of the cleaned up rate files. Default is True.


    Returns
    -------
    cleaned_rate_files : list of strings
        Filenames of cleaned rate files.

    """
    # We need to check that the desired output directories exist, and if not create them
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)

    # Start a timer to keep track of runtime
    time0 = time.perf_counter()

    # will store the filename of the cleaned up rate fits files.
    cleaned_rate_files = []
    N_files_processed = 0
    for fid, rate_file in enumerate(rate_files):
        print(f"Noise Clean: Processing file {fid + 1} of {len(rate_files)}: {os.path.basename(rate_file)}")

        # output filepath of the cleaner rate fits
        outname = os.path.join(output_dir, os.path.basename(rate_file))
        cleaned_rate_files.append(outname)
        # skip if already processed
        if os.path.exists(outname) and not overwrite: #todo undo
            print(f"\tOutput file {os.path.basename(outname)} already exists in the cleaned output directory; skipping {os.path.basename(rate_file)}.")
            continue

        N_files_processed += 1
        clean_rate_nirspec_per_file(rate_file, stage2_dir, output_dir,N_nodes=N_nodes,
                                  clean_1f_noise=clean_1f_noise,model_charge_transfer=model_charge_transfer, utils_dir=utils_dir,
                                  init_centroid=init_centroid,targetname=targetname, mppool=mppool,extend_sat=extend_sat)
        # Print out the time benchmark
        time1 = time.perf_counter()
        print(f"\tNoise Clean Runtime so far: {time1 - time0:0.4f} seconds\n")
    time1 = time.perf_counter()
    print(f"Noise Clean Total Runtime: {time1 - time0:0.4f} seconds")

    if save_plots and N_files_processed > 0:
        breads.jwst_tools.plotting.plot_2d_image_sets_side_by_side(rate_files, cleaned_rate_files,
                                                                   output_dir=output_dir,
                                                                   suptitle="Noise Cleaning results. Left = Before, Right = After.",
                                                                   plot_label='noiseclean')

    return cleaned_rate_files


###########################################################################
#  Function for centroid calibration

def recenter_coordinates_of_sequence_nirspec(cal_files, utils_dir,combined_contnorm_spec_filename,fitpsf_filename,
                                             init_centroid=None,
                                             wv_sampling=None,
                                             mask_charge_transfer_radius=None,ra_dec_point_sources=None,
                                             IWA_init=0.0, OWA_init=1.0, frac_init_bandpass=0.005,
                                             IWA=0.0, OWA=0.5,
                                             mppool=None,
                                             overwrite=False,
                                             targetname=None,
                                             save_pickle=False,
                                             load_pickle=False,
                                             stis_spectrum=None,
                                             use_stpsf = True,
                                             use_breadspsf = None,
                                             poly_deg_coords = 2):
    """


    Parameters
    ----------

    Returns
    -------

    """
    poly_centroid_filename = fitpsf_filename.replace(".fits", "_poly_centroid_IWA{0:.2f}_OWA{1:.2f}.txt".format(IWA,OWA))
    poly_fluxcal_filename = fitpsf_filename.replace(".fits", "_poly_fluxcal_IWA{0:.2f}_OWA{1:.2f}.txt".format(IWA,OWA))
    if overwrite or len(glob(poly_centroid_filename)) == 0:

        if use_breadspsf is not None:
            use_stpsf = False

        if not os.path.exists(utils_dir):
            os.makedirs(utils_dir)

        if init_centroid is None:
            _tmp_centroid = np.array([0,0])
        else:
            _tmp_centroid = np.array(init_centroid)


        grating = fits.getheader(cal_files[0])['GRATING'].strip()
        detector = fits.getheader(cal_files[0])['DETECTOR'].strip().lower()

        if wv_sampling is None:
            wv_sampling = default.wv_sampling_dict[grating][detector]

        splitbasename = os.path.basename(cal_files[0]).split("_")
        pickle_filename = os.path.join(utils_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[3] + "_regwvs" + ".pkl")

        if load_pickle and len(glob(pickle_filename)) >= 1:
            regwvs_combdataobj = JWSTNirspec_multiple_cals.load(pickle_filename)
        else:
            preproc_task_list = [
                ["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}, True, True],
                ["compute_coordinates_arrays", {'targname': targetname}, True, True],
                ["compute_advanced_badpix", {"threshold_badpix": 10, "mppool": mppool,
                                             "combined_contnorm_filename": combined_contnorm_spec_filename}, True, True],
                ["compute_interpdata_regwvs", {"wv_sampling": wv_sampling}, True, True]]
            dataobj_list = []
            for filename in cal_files:
                dataobj = JWSTNirspec_cal(filename, utils_dir=utils_dir)
                dataobj.run_preproc_list(preproc_task_list=preproc_task_list)

                # Applying some rough coordinate correction before applying the masks
                if mask_charge_transfer_radius is not None or ra_dec_point_sources is not None:
                    dataobj.apply_coords_offset(_tmp_centroid)

                # Do some masking
                if mask_charge_transfer_radius is not None:
                    dataobj.compute_charge_bleeding_mask(threshold2mask=mask_charge_transfer_radius)
                # mask planets before computing the star spectrum
                if ra_dec_point_sources is not None:
                    for ra_pl, dec_pl in ra_dec_point_sources:
                        where_pl = dataobj.where_point_source([ra_pl / 1000., dec_pl / 1000.], 0.16)
                        dataobj.bad_pixels[where_pl] = np.nan

                # undoing the rough coordinate correction because we don't want this to bias the centroid later
                if mask_charge_transfer_radius is not None or ra_dec_point_sources is not None:
                    dataobj.apply_coords_offset(-_tmp_centroid)

                dataobj_list.append(dataobj)

            # combined data object
            regwvs_combdataobj = JWSTNirspec_multiple_cals(dataobj_list)
            regwvs_combdataobj.plot_2D_point_cloud_html(save_plot=True, overlay_pointcloud=False)
            if save_pickle:
                regwvs_combdataobj.save()

        # Derive the first guess centrois from a narrow wavelength region
        if init_centroid is None:
            bandpass = regwvs_combdataobj.wv_sampling[-1] - regwvs_combdataobj.wv_sampling[0]
            midwv = (regwvs_combdataobj.wv_sampling[-1] + regwvs_combdataobj.wv_sampling[0]) / 2.0
            debug_wv_range = [midwv - frac_init_bandpass * bandpass, midwv + frac_init_bandpass * bandpass]
            bestfit_paras, _, _, _ = fitpsf(regwvs_combdataobj, use_stpsf=use_stpsf,use_breadspsf=use_breadspsf,
                                            IWA=IWA_init, OWA=OWA_init, out_filename=None,
                                            overwrite=False, mppool=mppool, debug_wv_range=debug_wv_range,
                                            poly_deg_coords=0)
            init_centroid = (np.nanmedian(bestfit_paras[0, :, 2]), np.nanmedian(bestfit_paras[0, :, 3]))

        ######
        # Fit the whole wavelength range
        bestfit_paras, data, bestfit_model, residuals = fitpsf(regwvs_combdataobj, use_stpsf=use_stpsf,use_breadspsf=use_breadspsf,
                                                               init_centroid=init_centroid,
                                                               IWA=IWA, OWA=OWA, out_filename=fitpsf_filename,
                                                               overwrite=False, mppool=mppool, debug_wv_range=None,
                                                               stis_spectrum=stis_spectrum, poly_deg_coords=poly_deg_coords)
    return poly_centroid_filename,poly_fluxcal_filename


def recenter_coordinates_per_frame_nirspec(cal_files, utils_dir,combined_contnorm_spec_filename,fitpsf_filename_perframe_suffix,
                                           init_centroid=None,
                                             wv_sampling=None,
                                             mask_charge_transfer_radius=None,ra_dec_point_sources=None,
                                             IWA_init=0.0, OWA_init=1.0, frac_init_bandpass=0.005,
                                             IWA=0.0, OWA=0.5,
                                             mppool=None,
                                             overwrite=False,
                                             targetname=None,
                                             stis_spectrum=None,
                                             use_stpsf = True,
                                             use_breadspsf = None,
                                             poly_deg_coords = 2,
                                           plot_combined = True,
                                           wv_min = None, wv_max = None,):
    """

        cal_files:
        utils_dir:
        combined_contnorm_spec_filename:
        fitpsf_filename_perframe_suffix:
        init_centroid:
        wv_sampling:
        mask_charge_transfer_radius:
        ra_dec_point_sources:
        IWA_init:
        OWA_init:
        frac_init_bandpass:
        IWA:
        OWA:
        mppool:
        overwrite:
        targetname:
        stis_spectrum:
        use_stpsf:
        use_breadspsf:
        poly_deg_coords:
        plot_combined:
        wv_min : float
            Don't include wavelength less than wv_min in the fit. If None, default is 10% of the bandpass mask on the edges.
        wv_max : float
            Don't include wavelength greater than wv_max in the fit. If None, default is 10% of the bandpass mask on the edges.

    """

    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)

    if use_breadspsf is not None:
        use_stpsf = False

    if init_centroid is None:
        _tmp_centroid = np.array([0, 0])
    else:
        _tmp_centroid = np.array(init_centroid)

    grating = fits.getheader(cal_files[0])['GRATING'].strip()
    detector = fits.getheader(cal_files[0])['DETECTOR'].strip().lower()

    if wv_sampling is None:
        wv_sampling = default.wv_sampling_dict[grating][detector]

    # Define a series of processing tasks to be performed on each input file.
    # coords_filename = glob(fitpsf_filename.replace(".fits","_poly_centroid*.txt"))[0]
    preproc_task_list = [["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}],
                         ["compute_coordinates_arrays", {'targname': targetname}],
                         ["compute_advanced_badpix", {"threshold_badpix": 10, "mppool": None,
                                                      "combined_contnorm_filename": combined_contnorm_spec_filename}],
                         ["compute_interpdata_regwvs", {"wv_sampling": wv_sampling}]]

    N_files_to_be_processed = 0
    for filename in cal_files:
        fitpsf_filename_perframe = os.path.join(utils_dir, os.path.basename(filename).replace(".fits",fitpsf_filename_perframe_suffix + ".fits"))
        poly_centroid_filename = fitpsf_filename_perframe.replace(".fits", "_poly_centroid_IWA{0:.2f}_OWA{1:.2f}.txt".format(IWA,OWA))
        poly_fluxcal_filename = fitpsf_filename_perframe.replace(".fits", "_poly_fluxcal_IWA{0:.2f}_OWA{1:.2f}.txt".format(IWA,OWA))
        if overwrite or len(glob(poly_centroid_filename)) == 0:
            N_files_to_be_processed +=1
    if N_files_to_be_processed == 0:
        return

    if plot_combined:
        fig = plt.figure(figsize=(12, 10))
        fontsize = 12
        # gs = gridspec.GridSpec(8, 1, height_ratios=[1, 0.5, 0.3, 1, 0.5, 0.3, 1, 0.5], width_ratios=[1])
        gs = gridspec.GridSpec(5, 1, height_ratios=[1, 0.3, 1, 0.3, 1], width_ratios=[1])
        gs.update(left=0.1, right=0.95, bottom=0.07, top=0.95, wspace=0.0, hspace=0.0)
        ax1 = plt.subplot(gs[0, 0])
        if stis_spectrum is not None:
            stis_table = Table(fits.getdata(stis_spectrum, 1))
            stis_wvs = (np.array(stis_table["WAVELENGTH"]) * u.Angstrom).to(u.um).value  # angstroms -> mum
            stis_spec = np.array(
                stis_table["FLUX"]) * u.erg / u.s / u.cm ** 2 / u.Angstrom  # erg s-1 cm-2 A-1
            stis_spec = stis_spec.to(u.W * u.m ** -2 / u.um)
            stis_spec_Fnu = stis_spec * (stis_wvs * u.um) ** 2 / const.c  # from Flambda back to Fnu
            stis_spec_Fnu = stis_spec_Fnu.to(u.MJy).value
            plt.plot(stis_wvs, stis_spec_Fnu, linestyle=":", color="black", label="CALSPEC", linewidth=2)

    for filename in cal_files:
        fitpsf_filename_perframe = os.path.join(utils_dir, os.path.basename(filename).replace(".fits",fitpsf_filename_perframe_suffix + ".fits"))
        poly_centroid_filename = fitpsf_filename_perframe.replace(".fits", "_poly_centroid_IWA{0:.2f}_OWA{1:.2f}.txt".format(IWA,OWA))
        poly_fluxcal_filename = fitpsf_filename_perframe.replace(".fits", "_poly_fluxcal_IWA{0:.2f}_OWA{1:.2f}.txt".format(IWA,OWA))
        if overwrite or len(glob(poly_centroid_filename)) == 0:
            dataobj = JWSTNirspec_cal(filename, utils_dir=utils_dir)
            dataobj.run_preproc_list(save_utils=True, load_utils=True, preproc_task_list=preproc_task_list)

            if init_centroid is None:
                # Derive the first guess centrois from a narrow wavelength region
                bandpass = dataobj.wv_sampling[-1] - dataobj.wv_sampling[0]
                midwv = (dataobj.wv_sampling[-1] + dataobj.wv_sampling[0]) / 2.0
                debug_wv_range = [midwv - frac_init_bandpass * bandpass, midwv + frac_init_bandpass * bandpass]
                bestfit_paras, _, _, _ = fitpsf(dataobj,use_stpsf=use_stpsf, use_breadspsf=use_breadspsf,
                                                IWA=IWA_init, OWA=OWA_init, out_filename=None,
                                                overwrite=True, mppool=mppool, debug_wv_range=debug_wv_range,
                                                poly_deg_coords=0)
                _init_centroid = np.array([np.nanmedian(bestfit_paras[0, :, 2]), np.nanmedian(bestfit_paras[0, :, 3])])
            else:
                _init_centroid=  np.array(init_centroid)

            # Applying some rough coordinate correction before applying the masks
            if mask_charge_transfer_radius is not None or ra_dec_point_sources is not None:
                dataobj.apply_coords_offset(_init_centroid)

            # Do some masking
            if mask_charge_transfer_radius is not None:
                dataobj.compute_charge_bleeding_mask(threshold2mask=mask_charge_transfer_radius)
            # mask planets before computing the star spectrum
            if ra_dec_point_sources is not None:
                for ra_pl, dec_pl in ra_dec_point_sources:
                    where_pl = dataobj.where_point_source([ra_pl / 1000., dec_pl / 1000.], 0.16)
                    dataobj.bad_pixels[where_pl] = np.nan

            # undoing the rough coordinate correction because we don't want this to bias the centroid later
            if mask_charge_transfer_radius is not None or ra_dec_point_sources is not None:
                dataobj.apply_coords_offset(-_init_centroid)
            ######
            # Fit the whole wavelength range
            bestfit_paras, data, bestfit_model, residuals = fitpsf(dataobj, use_stpsf=use_stpsf,use_breadspsf=use_breadspsf,
                                                                   init_centroid=_init_centroid,
                                                                   IWA=IWA, OWA=OWA, out_filename=fitpsf_filename_perframe,
                                                                   overwrite=False, mppool=mppool, debug_wv_range=None,
                                                                   stis_spectrum=stis_spectrum,
                                                                   poly_deg_coords=poly_deg_coords,
                                                                   wv_min = wv_min, wv_max = wv_max)

            if plot_combined:
                _med_bestfit_paras = np.nanmean(bestfit_paras, axis=0)

                plt.figure(fig.number)
                ax1 = plt.subplot(gs[0, 0])
                plt.plot(dataobj.wv_sampling, _med_bestfit_paras[:, 1], linestyle="--",
                         label=os.path.basename(filename).replace(".fits", ""), linewidth=1)
                plt.xlim([dataobj.wv_sampling[0], dataobj.wv_sampling[-1]])
                plt.ylim([np.nanmin(_med_bestfit_paras[:, 1]) * 0.8, np.nanmax(_med_bestfit_paras[:, 1]) * 1.2])
                plt.legend(loc="upper right")
                plt.ylabel("Flux density (MJy)", fontsize=fontsize)
                plt.xlabel(r"Wavelength ($\mu$m)", fontsize=fontsize)
                plt.gca().tick_params(axis='x', labelsize=fontsize)
                plt.gca().tick_params(axis='y', labelsize=fontsize)
                plt.legend(loc="upper right")

                if "ifu" in dataobj.breads_header['COORDS']:
                    xcoord_label = 'IFU x (arcsec)'
                    ycoord_label = 'IFU y (arcsec)'
                elif "sky" in dataobj.breads_header['COORDS']:
                    xcoord_label = r'$\Delta$RA (arcsec)'
                    ycoord_label = r'$\Delta$Dec (arcsec)'

                ax1 = plt.subplot(gs[2, 0])
                plt.plot(dataobj.wv_sampling, _med_bestfit_paras[:, 2],
                         label=os.path.basename(filename).replace(".fits", ""))
                plt.legend(loc="upper right")
                plt.xlim([dataobj.wv_sampling[0], dataobj.wv_sampling[-1]])
                plt.xlabel(r"Wavelength ($\mu$m)", fontsize=fontsize)
                plt.ylabel(xcoord_label, fontsize=fontsize)
                plt.gca().tick_params(axis='x', labelsize=fontsize)
                plt.gca().tick_params(axis='y', labelsize=fontsize)

                ax1 = plt.subplot(gs[4, 0])
                plt.plot(dataobj.wv_sampling, _med_bestfit_paras[:, 3],
                         label=os.path.basename(filename).replace(".fits", ""))
                plt.xlim([dataobj.wv_sampling[0], dataobj.wv_sampling[-1]])
                plt.xlabel(r"Wavelength ($\mu$m)", fontsize=fontsize)
                plt.ylabel(ycoord_label, fontsize=fontsize)
                plt.gca().tick_params(axis='x', labelsize=fontsize)
                plt.gca().tick_params(axis='y', labelsize=fontsize)

    if plot_combined:
        splitbasename = os.path.basename(cal_files[0]).split("_")
        out_filename = os.path.join(utils_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[3] + fitpsf_filename_perframe_suffix + "_all_in_one.png")
        plt.savefig(out_filename, dpi=200)
    plt.close('all')


###########################################################################
# Host Star PSF Subtraction


def get_contnorm_spec(dataobj_list, out_filename=None, load_utils=False, spec_R_sampling=None,
                      masking_radius = None, ra_planets=None,dec_planets=None,interpolation=None,save_plots=True):
    """ Combine the continuum normalized stellar spectra of a list of data objects.

    Parameters
    ----------
    dataobj_list : list
        List of data objects
    out_filename: str
        If not None, the combined spectrum will be saved to this file.
        If the file already exists and load_utils is True, the function will try to load the combined spectrum from this file instead of recomputing it.
    load_utils : bool
        Whether to try to load the combined spectrum from out_filename if it exists, instead of recomputing it.
    spec_R_sampling : float or None (optional)
        Spectral resolution to sample the continuum-normalized star spectrum
        If None, the spectral resolution will be set to 4 times the instrumental spectral resolution of the IFU.
    masking_radius : float
        If not None, radius (in arcsec) of the region to mask around each planet defined by ra_planets and dec_planets.
    ra_planets : list
        List of delta right ascension positions relative to the host star (in arcsec).
    dec_planets : list
        List of delta declination positions relative to the host star (in arcsec).
    interpolation : str
        Either "linear" or "spline" interpolation.
    save_plots : bool
        Whether to save a html plot. Default is True. out_filename should be defined.

    Returns
    -------
    new_wavelengths : 1d array
        Wavelength array in micron of the continuum normalized star spectrum.
    combined_fluxes : 1d array
        Flux array of the continuum normalized star spectrum. (without unit)
    combined_errors : 1d array
        Flux errors array of the continuum normalized star spectrum. (without unit)
    combined_star_func : Interp1d
        Interpolation function of the combined continuum normalized star spectrum.

    """
    if interpolation is None:
        interpolation = "linear"

    # Reload the combined spectrum if it already exists and load_utils is True
    if load_utils and len(glob(out_filename)):
        print(len(glob(out_filename)), out_filename)
        with fits.open(out_filename) as hdulist:
            new_wavelengths = hdulist["WAVE"].data
            combined_fluxes = hdulist['COM_FLUXES'].data
            combined_errors = hdulist['COM_ERRORS'].data
    else:
        wvs_list = []
        normalized_im_list = []
        normalized_err_list = []

        x_nodes_to_compare = None
        for dataobj in dataobj_list:

            # Reload the individual continuum normalized spectra from the utils folder
            reload_outputs = dataobj.reload_starspectrum_contnorm()
            if reload_outputs is None:
                raise Exception("Need to run compute_starspectrum_contnorm first. Could not reload continuum normalized data.")
            new_wavelengths, combined_fluxes, combined_errors, spline_cont0, spline_paras0, wv_nodes = reload_outputs

            # Checking that all spline nodes are identical
            if x_nodes_to_compare is None:
                x_nodes_to_compare = wv_nodes
            else:
                if not np.allclose(wv_nodes, x_nodes_to_compare):
                    raise Exception("The wv_nodes of the spline continuum fit are different for different data objects. This should not happen. Please check the compute_starspectrum_contnorm outputs for each data object.")

            # Mask pixels we don't want to use: eg pixels too noisy or pixels below median flux
            spline_cont0[np.where(spline_cont0 / dataobj.noise < 5)] = np.nan
            spline_cont0 = copy(spline_cont0)
            spline_cont0[np.where(spline_cont0 < np.median(spline_cont0))] = np.nan
            spline_cont0[np.where(np.isnan(dataobj.bad_pixels))] = np.nan

            # mask planets if needed
            if ra_planets is not None and dec_planets is not None:
                dra_as_array, ddec_as_array = dataobj.get_sky_coords()
                for pl_ra, pl_dec in zip(ra_planets,dec_planets):
                    dist2pointsource_as = np.sqrt((dra_as_array - pl_ra/1000) ** 2 + (ddec_as_array - pl_dec/1000.) ** 2)
                    where_pl =  np.where(dist2pointsource_as < masking_radius)
                    spline_cont0[where_pl] = np.nan

            # normalize the data
            normalized_im = dataobj.data / spline_cont0
            normalized_err = dataobj.noise / spline_cont0

            wvs_list.extend(dataobj.wavelengths.flatten())
            normalized_im_list.extend(normalized_im.flatten())
            normalized_err_list.extend(normalized_err.flatten())

        if spec_R_sampling is None:
            spec_R_sampling = dataobj.breads_header['STCONTRS']

        if interpolation == "linear":
            new_wavelengths, combined_fluxes, combined_errors = combine_spectrum(np.array(wvs_list),
                                                                                 np.array(normalized_im_list),
                                                                                 np.array(normalized_err_list),
                                                                                 np.nanmedian(wvs_list) / spec_R_sampling)
        elif interpolation == "spline":
            new_wavelengths, combined_fluxes, combined_errors, spl = combine_spectrum_1dspline(np.array(wvs_list),
                                                                                               np.array(normalized_im_list),
                                                                                               np.array(normalized_err_list),
                                                                                               np.nanmedian(wvs_list) / spec_R_sampling,
                                                                                               oversampling=10)


        if out_filename is not None:
            hdulist = fits.HDUList()
            _breads_header = dataobj_list[0].breads_header
            _breads_header["STCONTFN"] = out_filename
            _breads_header['STCONTRS'] = spec_R_sampling
            hdulist.append(fits.PrimaryHDU(header=dataobj_list[0].priheader))
            hdulist.append(fits.ImageHDU(data=new_wavelengths, header=dataobj_list[0].extheader, name="WAVE"))
            hdulist.append(fits.ImageHDU(data=combined_fluxes, name='COM_FLUXES'))
            hdulist.append(fits.ImageHDU(data=combined_errors, name='COM_ERRORS'))
            hdulist.append(fits.ImageHDU(data=wv_nodes, name='wv_nodes'))
            hdulist.append(fits.ImageHDU(header=_breads_header, name='BREADS'))
            hdulist.writeto(out_filename, overwrite=True)
            hdulist.close()

        if save_plots:
            wl = np.asarray(new_wavelengths)
            fl = np.asarray(combined_fluxes)
            err = np.asarray(combined_errors)

            fig = go.Figure()

            # -- Spectrum envelope ------------------------------------------
            fig.add_trace(
                go.Scatter(
                    x=wl, y=fl,
                    mode="lines",
                    line=dict(color="royalblue", width=1.1),
                    name="Flux",
                    hovertemplate="? = %{x:.4f} µm<br>Flux = %{y:.4f}<extra></extra>",
                )
            )

            # Reference line at continuum = 1
            fig.add_hline(y=1.0, line=dict(color="gray", dash="dash", width=1))

            # -- Layout ------------------------------------------------------------
            fig.update_layout(
                title=dict(text="Continuum-Normalized Spectrum", font=dict(size=16)),
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.01,
                            xanchor="right", x=1),
                hovermode="x unified",
                height=500,
                margin=dict(l=70, r=30, t=70, b=60),
                xaxis=dict(title="Wavelength (µm)", showgrid=True),
                yaxis=dict(title="Normalized Flux", showgrid=True, zeroline=False),
            )
            fig.write_html(out_filename.replace(".fits",".html"))

    combined_star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False,fill_value=1)
    return new_wavelengths, combined_fluxes, combined_errors,combined_star_func

def compute_normalized_stellar_spectrum(cal_files, utils_dir, combined_contnorm_spec_filename,
                                        wv_nodes=None, suffix = None,
                                        coords_offset = None, coords_filename_filter=None,
                                        mask_charge_transfer_radius=None, mppool=None,
                                        ra_dec_point_sources=None,aper_rad=None,
                                        overwrite=False,targetname=None,
                                        spline3d_prior_filename=None):
    """

    Parameters
    ----------

    Returns
    -------

    """
    if not overwrite and len(glob(combined_contnorm_spec_filename)) >= 1:
        with fits.open(combined_contnorm_spec_filename) as hdulist:
            new_wavelengths = hdulist["WAVE"].data
            combined_fluxes = hdulist["COM_FLUXES"].data

        combined_star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False, fill_value=1)
        return combined_star_func
    else:
        if not os.path.exists(utils_dir):
            os.makedirs(utils_dir)

        grating = fits.getheader(cal_files[0])['GRATING'].strip()
        detector = fits.getheader(cal_files[0])['DETECTOR'].strip().lower()

        if wv_nodes is None:
            wv_nodes = default.nodes_dict[grating][detector]
        if coords_offset is None:
            coords_offset = (0,0)

        dataobj_list = []
        for filename in cal_files:
            dataobj = JWSTNirspec_cal(filename, utils_dir=utils_dir)

            if suffix is not None:
                dataobj.default_filenames["compute_starspectrum_contnorm"] = dataobj.default_filenames["compute_starspectrum_contnorm"].replace(".fits","_"+suffix+".fits")
                dataobj.default_filenames["compute_advanced_badpix"] = dataobj.default_filenames["compute_starspectrum_contnorm"].replace(".fits","_"+suffix+".fits")
                dataobj.default_filenames["compute_starsubtraction"] = dataobj.default_filenames["compute_starsubtraction"].replace(".fits","_"+suffix+".fits")

            if coords_filename_filter is not None:
                coords_filename = glob(os.path.join(utils_dir,os.path.basename(dataobj.filename).replace(".fits", coords_filename_filter)))[0]
                coords_offset = None
            else:
                coords_filename = None

            # Define a series of processing tasks to be performed on each input file.
            preproc_task_list = [
                ["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}, True, True],# True,True means "save data and load if you can"
                ["compute_coordinates_arrays", {'targname': targetname}, True, True],
                ["apply_coords_offset", {"coords_offset": coords_offset,"coords_filename":coords_filename}],
                ["compute_starspectrum_contnorm", {"wv_nodes": wv_nodes,"threshold_badpix": 100, "iterative": False,"spline3d_prior_filename":spline3d_prior_filename,
                                                   "mppool": mppool}, True, True],
                ["compute_advanced_badpix", {"threshold_badpix": 10, "mppool": mppool, "iterative": False}, False, True],# don't save, but load, temporary reduction only
                ["compute_starspectrum_contnorm", {"wv_nodes": wv_nodes,"threshold_badpix": 100, "iterative": False,"spline3d_prior_filename":spline3d_prior_filename,
                                                   "mppool": mppool}, True,False]]  # Save, but don't load, always overwrite the previous reduction
            dataobj.run_preproc_list(preproc_task_list=preproc_task_list)

            # Do some masking
            if mask_charge_transfer_radius is not None:
                dataobj.compute_charge_bleeding_mask(threshold2mask=mask_charge_transfer_radius)
            # mask planets before computing the star spectrum
            if ra_dec_point_sources is not None:
                if aper_rad is None:
                    aper_rad = 0.16
                for ra_pl, dec_pl in ra_dec_point_sources:
                    where_pl = dataobj.where_point_source([ra_pl / 1000., dec_pl / 1000.], aper_rad)
                    dataobj.bad_pixels[where_pl] = np.nan

            dataobj_list.append(dataobj)

        _out = get_contnorm_spec(dataobj_list, load_utils=True,
                                 out_filename=combined_contnorm_spec_filename,
                                 spec_R_sampling=2700 * 4, interpolation="linear")
        new_wavelengths, combined_fluxes, combined_errors, combined_star_func = _out

        return combined_star_func




def compute_3dsplines(cal_files, utils_dir, targetname,combined_contnorm_spec_filename,
                      wv_nodes=None,x_nodes=None,y_nodes=None,save_pickle=False, load_pickle=False,
                      coords_filename_filter = None,numthreads=1,spline3d_suffix = "",
                      centroid_per_frame=True, overwrite=False,mask_charge_transfer_radius=None,ra_dec_point_sources=None,aper_rad =None,
                      stamp_size = (0.2,0.2),
                      subtract_comp = None):

    grating = fits.getheader(cal_files[0])['GRATING'].strip()
    detector = fits.getheader(cal_files[0])['DETECTOR'].strip().lower()
    if wv_nodes is None:
        wv_nodes= default.wv_nodes_3D_dict[grating][detector]
    if x_nodes is None:
        x_nodes=default.x_nodes_3D_5x
    if y_nodes is None:
        y_nodes=default.y_nodes_3D_5x

    splitbasename = os.path.basename(cal_files[0]).split("_")
    pickle_filename = os.path.join(utils_dir,splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[3]+ "_sky"+spline3d_suffix+".pkl")
    if load_pickle and len(glob(pickle_filename)) >= 1:
        combdataobj = JWSTNirspec_multiple_cals.load(pickle_filename)
    else:
        dataobj_list = []
        # Define a series of processing tasks to be performed on each input file.
        preproc_task_list = [["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}],
                             ["compute_coordinates_arrays",{'targname':targetname}],
                             ["compute_advanced_badpix",{"threshold_badpix": 10, "mppool": None,
                                                         "combined_contnorm_filename": combined_contnorm_spec_filename}]]
        for filename in cal_files[:]:
            dataobj = JWSTNirspec_cal(filename, utils_dir=utils_dir)
            dataobj.run_preproc_list(save_utils=True, load_utils=True,preproc_task_list=preproc_task_list)

            if coords_filename_filter is not None and centroid_per_frame:
                coords_filename = glob(os.path.join(utils_dir,os.path.basename(filename).replace(".fits", coords_filename_filter)))[0]
                dataobj.apply_coords_offset(coords_filename = coords_filename)

            if mask_charge_transfer_radius is not None:
                dataobj.compute_charge_bleeding_mask(threshold2mask=mask_charge_transfer_radius)

            if ra_dec_point_sources is not None:
                if aper_rad is None:
                    aper_rad = 0.16
                for ra_pl, dec_pl in ra_dec_point_sources:
                    where_pl = dataobj.where_point_source([ra_pl / 1000., dec_pl / 1000.], aper_rad)
                    dataobj.bad_pixels[where_pl] = np.nan

            dataobj_list.append(dataobj)

        # combined data object
        combdataobj = JWSTNirspec_multiple_cals(dataobj_list)

        if coords_filename_filter is not None and not centroid_per_frame:
            coords_filename = glob(os.path.join(utils_dir,os.path.basename(combdataobj.filename).replace(".fits", coords_filename_filter)))[0]
            combdataobj.apply_coords_offset(coords_filename = coords_filename)

        combdataobj.default_filenames["compute_starspectrum_contnorm_3dspline"] = (
            combdataobj.default_filenames["compute_starspectrum_contnorm_3dspline"].replace(".fits",spline3d_suffix + ".fits"))
        combdataobj.default_filenames["compute_starsubtraction_3dspline"] = (
            combdataobj.default_filenames["compute_starsubtraction_3dspline"].replace(".fits",spline3d_suffix + ".fits"))

        if subtract_comp is not None:
            # dataobj, save_utils=False,centroid = None,OWA=None,spectrum_func=None,out_folder = "insert_psf",
            # mode=None,mppool=None,use_breadspsf=None,interpgrid=None
            projected_model = project_psf_model(combdataobj, save_utils=False, mode="breadspsf",
                                                centroid=subtract_comp["centroid"], OWA=subtract_comp["OWA"],
                                                spectrum_func=subtract_comp["spectrum_func"])

            combdataobj.data -= projected_model

        if save_pickle:
            combdataobj.save(filename=pickle_filename)


    if not overwrite and os.path.exists(combdataobj.default_filenames["compute_starsubtraction_3dspline"]):
        return combdataobj

    _out = combdataobj.reload_starspectrum_contnorm_3dspline()
    if overwrite or _out is None:
        _out = combdataobj.compute_starspectrum_contnorm_3dspline(save_utils=True,max_cores=numthreads,
                                                       wv_nodes=wv_nodes,
                                                       x_nodes=x_nodes,
                                                       y_nodes=y_nodes,
                                                       stamp_size = stamp_size)
    # new_wavelengths, combined_fluxes = _out[0],_out[1]
    new_wavelengths, combined_fluxes, combined_errors, spline_cont0, spline3d_paras,spline3d_paras_err,wv_nodes,x_nodes,y_nodes = _out
    # exit()

    _out = combdataobj.compute_starsubtraction_3dspline(save_utils=True,max_cores=numthreads,iterative=True,
                                                        threshold_badpix=10,only_identify_badpix=True)
    subtracted_im, spline_cont0, spline3d_paras, spline3d_paras_err, wv_nodes, x_nodes, y_nodes = _out

    # plt.imshow(spline3d_paras[2,:,:],origin="lower")
    # plt.colorbar()
    # plt.show()

    return combdataobj


def cube_extraction(cal_files, utils_dir, out_dir, combined_contnorm_spec_filename, coords_filename_filter,
                    mode="raw", suffix=None, contnorm_suffix=None,
                    x_vec=None, y_vec=None, wv_sampling=None,
                    mask_charge_transfer_radius=None, ra_dec_point_sources=None,
                    aper_radius=0.15,
                    RDI_IWA=None, RDI_OWA=None, RDI_ann_width=0.5, RDI_use_stpsf=False, RDI_use_breadspsf=True,
                    ASDI_contnorm_3dspline_filename=None, ASDI_starsub_3dspline_filename=None,
                    mppool=None,
                    overwrite=False,
                    targetname=None,
                    use_stpsf=False, use_breadspsf=True,
                    load_pickle=True,
                    save_pickle=True,
                    ifucoords=False):
    # - modes:
    # 	- raw
    # 	- 1dspline
    # 	- RDI
    # 	- ASDI

    if suffix is None:
        suffix = mode

    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)

    if use_breadspsf is not None:
        use_stpsf = False

    grating = fits.getheader(cal_files[0])['GRATING'].strip()
    detector = fits.getheader(cal_files[0])['DETECTOR'].strip().lower()

    if x_vec is None:
        x_vec = np.arange(-2, 2, 0.05)
    if y_vec is None:
        y_vec = np.arange(-2, 2, 0.05)

    if wv_sampling is None:
        wv_sampling = default.wv_sampling_dict[grating][detector]

    splitbasename = os.path.basename(cal_files[0]).split("_")

    build_cube_filename = os.path.join(out_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[
        3] + "_" + grating + "_" + suffix + "_cube.fits")
    if not overwrite and os.path.exists(build_cube_filename):
        with fits.open(build_cube_filename) as hdulist:
            flux_cube = hdulist['FLUX'].data
            fluxerr_cube = hdulist['FLUXERR'].data
            x_grid = hdulist['X'].data
            y_grid = hdulist['Y'].data
            wv_sampling = hdulist['WAVE'].data
        return flux_cube, fluxerr_cube, x_grid, y_grid, wv_sampling

    if ifucoords:
        pickle_suffix = "_" + suffix + "_ifu_regwvs"
    else:
        pickle_suffix = "_" + suffix + "_sky_regwvs"
    pickle_filename = os.path.join(utils_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[
        3] + pickle_suffix + ".pkl")

    if load_pickle and len(glob(pickle_filename)) >= 1:
        combdataobj = JWSTNirspec_multiple_cals.load(pickle_filename)
    else:
        dataobj_list = []
        for filename in cal_files:
            dataobj = JWSTNirspec_cal(filename, utils_dir=utils_dir)

            if contnorm_suffix is not None:
                dataobj.default_filenames["compute_starspectrum_contnorm"] = dataobj.default_filenames[
                    "compute_starspectrum_contnorm"].replace(".fits", "_" + contnorm_suffix + ".fits")
                dataobj.default_filenames["compute_advanced_badpix"] = dataobj.default_filenames[
                    "compute_starspectrum_contnorm"].replace(".fits", "_" + contnorm_suffix + ".fits")
                dataobj.default_filenames["compute_starsubtraction"] = dataobj.default_filenames[
                    "compute_starsubtraction"].replace(".fits", "_" + contnorm_suffix + ".fits")

            # Define a series of processing tasks to be performed on each input file.
            # coords_filename = glob(fitpsf_filename.replace(".fits","_poly_centroid*.txt"))[0]
            preproc_task_list = [["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}],
                                 ["compute_coordinates_arrays", {'targname': targetname}]]

            if mode == "1dspline":
                _task = ["compute_starsubtraction", {"threshold_badpix": 10, "mppool": mppool,
                                                     "combined_contnorm_filename": combined_contnorm_spec_filename,
                                                     "load_starspectrum_contnorm": None}]
            elif mode == "1dspline_with_prior":
                _task = ["compute_starsubtraction", {"threshold_badpix": 10, "mppool": mppool,
                                                     "combined_contnorm_filename": combined_contnorm_spec_filename,
                                                     "load_starspectrum_contnorm": dataobj.default_filenames[
                                                         "compute_starspectrum_contnorm"]}]
            else:
                _task = ["compute_advanced_badpix", {"threshold_badpix": 10, "mppool": mppool,
                                                     "combined_contnorm_filename": combined_contnorm_spec_filename}]
            preproc_task_list.append(_task)
            dataobj.run_preproc_list(save_utils=True, load_utils=True, preproc_task_list=preproc_task_list)

            coords_filename = \
            glob(os.path.join(utils_dir, os.path.basename(filename).replace(".fits", coords_filename_filter)))[0]
            dataobj.apply_coords_offset(coords_filename=coords_filename)

            if "RDI" in mode:
                dataobj.compute_interpdata_regwvs(wv_sampling=wv_sampling, save_utils=False)

            # Do some masking
            if mask_charge_transfer_radius is not None:
                dataobj.compute_charge_bleeding_mask(threshold2mask=mask_charge_transfer_radius)
            # mask point sources
            if ra_dec_point_sources is not None:
                _badpix_map_cp = copy(dataobj.bad_pixels)
                for ra_pl, dec_pl in ra_dec_point_sources:
                    if "sky" in dataobj.breads_header['COORDS']:
                        x_pl, y_pl = ra_pl, dec_pl
                    elif "ifu" in dataobj.breads_header['COORDS']:
                        _out = dataobj.get_ifu_coords(ras=ra_pl, decs=dec_pl)
                        x_pl, y_pl = float(_out[0]), float(_out[1])
                    where_pl = dataobj.where_point_source([x_pl / 1000., y_pl / 1000.], 0.16)
                    dataobj.bad_pixels[where_pl] = np.nan

            if ifucoords:
                dataobj.set_coords2ifu()

            if "RDI" in mode:
                rdi_filename_perframe = os.path.join(utils_dir, os.path.basename(filename).replace(".fits",
                                                                                                   "_fitpsf_" + suffix + ".fits"))
                if RDI_use_breadspsf:
                    RDI_use_stpsf = False
                if overwrite or len(glob(rdi_filename_perframe)) == 0:
                    bestfit_paras, data, bestfit_model, residuals = fitpsf(dataobj, use_stpsf=RDI_use_stpsf,
                                                                           use_breadspsf=RDI_use_breadspsf,
                                                                           IWA=RDI_IWA, OWA=RDI_OWA,
                                                                           ann_width=RDI_ann_width, padding=0.05,
                                                                           out_filename=rdi_filename_perframe,
                                                                           overwrite=overwrite, mppool=mppool,
                                                                           debug_wv_range=None,  # [4.5,4.51]
                                                                           poly_deg_coords=0, linear_interp=False)
                else:
                    with fits.open(rdi_filename_perframe) as hdulist:
                        bestfit_model = hdulist['BESTMODL'].data
                dataobj.data -= bestfit_model
                if ra_dec_point_sources is not None:
                    dataobj.bad_pixels = _badpix_map_cp

            dataobj_list.append(dataobj)

        combdataobj = JWSTNirspec_multiple_cals(dataobj_list)

        if mode != "RDI":
            combdataobj.compute_interpdata_regwvs(wv_sampling=wv_sampling)

        if save_pickle:
            combdataobj.save(suffix=pickle_suffix)

    if mode == "ASDI":
        hdulist = fits.open(ASDI_contnorm_3dspline_filename)
        new_wavelengths = hdulist["WAVE"].data
        combined_fluxes = hdulist["COM_FLUXES"].data
        hdulist.close()
        star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False, fill_value=1)
        # plt.figure()
        # plt.plot(wv_sampling, star_func(wv_sampling))
        # plt.show()

        # ASDI_contnorm_3dspline_filename,ASDI_starsub_3dspline_filename
        stellar_features = star_func(combdataobj.wavelengths)
        max_cores = mppool._processes
        _out = evaluate_3dspline_pointcloud(combdataobj, ASDI_starsub_3dspline_filename, max_cores=max_cores,
                                            stellar_features=stellar_features)
        ASDI_model, _ = _out
        # plt.plot(combdataobj.data[750,:])
        # plt.plot(ASDI_model[750,:])
        # plt.show()
        combdataobj.data -= ASDI_model

    out = build_cube(combdataobj, x_vec, y_vec,
                     use_breadspsf=use_breadspsf, use_stpsf=use_stpsf,
                     out_filename=build_cube_filename, overwrite=overwrite,
                     mppool=mppool, aper_radius=aper_radius,
                     debug_wv_range=None, N_pix_min=None,
                     linear_interp=False)

    flux_cube, fluxerr_cube, x_grid, y_grid, wv_sampling = out

    return flux_cube, fluxerr_cube, x_grid, y_grid, wv_sampling


def extract_spectrum(cube_filename, out_filename, xy_coords, labels=None):
    """
    Extract the spectrum of one or more point sources (planets) from an IFU cube.
    A separate FITS file is written for each planet. An interactive Plotly HTML
    file is written for the combined 1D spectra, and a matplotlib PNG is written
    for the combined 2D image.

    Parameters
    ----------
    cube_filename : str
        Path to the input FITS cube (must have FLUX, FLUXERR, X, Y, WAVE extensions).
    out_filename : str
        Base path for the output FITS file (and diagnostic plots). When multiple
        planets are given, each planet's label is inserted before the ".fits"
        extension, e.g. "out.fits" -> "out_b.fits", "out_c.fits", ...
    xy_coords : tuple(float, float) or list of tuple(float, float)
        Either a single (x, y) coordinate, or a list of (x, y) coordinates,
        one per planet/companion to extract.
    labels : list of str, optional
        Labels to use for each companion (e.g. "b", "c", ...) in filenames and
        plots. Defaults to "0", "1", "2", ... if not provided. Ignored (no
        filename suffix added) when only a single planet is given and labels
        is not provided.

    Returns
    -------
    results : list of dict
        One entry per planet, each with keys:
        'label', 'out_filename', 'wv_sampling', 'spectrum_Flambda', 'speckle_std_Flambda'
    """
    # --- Normalize input: allow either a single (x, y) tuple or a list of them ---
    if len(xy_coords) == 2 and np.isscalar(xy_coords[0]) and np.isscalar(xy_coords[1]):
        xy_coords_list = [tuple(xy_coords)]
    else:
        xy_coords_list = [tuple(xy) for xy in xy_coords]

    n_planets = len(xy_coords_list)
    single_planet_no_label = (n_planets == 1 and labels is None)

    if labels is None:
        labels = [str(i) for i in range(n_planets)]
    if len(labels) != n_planets:
        raise ValueError("labels must have the same length as xy_coords")

    with fits.open(cube_filename) as hdulist:
        flux_cube = hdulist['FLUX'].data
        fluxerr_cube = hdulist['FLUXERR'].data
        x_grid = hdulist['X'].data
        y_grid = hdulist['Y'].data
        wv_sampling = hdulist['WAVE'].data
        if 'wv_nodes' in hdulist:
            wv_nodes = hdulist['wv_nodes'].data
        else:
            wv_nodes = None

    r2star_grid = np.sqrt(x_grid ** 2 + y_grid ** 2)

    def to_Flambda(arr, wv=wv_sampling):
        return (arr * u.MJy * const.c / (wv * u.um) ** 2).to(u.W * u.m ** -2 / u.um).value

    # Qualitative color palette, one color per planet (cycles if > 10 planets)
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def hex_to_rgba(hex_color, alpha):
        hex_color = hex_color.lstrip('#')
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        return 'rgba({0},{1},{2},{3})'.format(r, g, b, alpha)

    results = []
    # interactive 1D spectra figure, built up across planets: Flambda on top, Jansky below
    fig_spec = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)

    # --- 2D image plot: built once (sum and std side by side), shared across all planets ---
    im_cube = flux_cube[100:(flux_cube.shape[0] - 100), :, :]
    all_nan_mask = np.all(np.isnan(im_cube), axis=0)
    im_sum = np.nansum(im_cube, axis=0)
    im_sum[all_nan_mask] = np.nan  # np.nansum returns 0, not nan, for all-nan slices
    im_std = np.nanstd(im_cube, axis=0)  # np.nanstd already returns nan for all-nan slices

    fig2, (ax_sum, ax_std) = plt.subplots(1, 2, num=2, figsize=(16, 6))
    fontsize = 12
    x_vec, y_vec = x_grid[0, :], y_grid[:, 0]
    dx, dy = x_vec[1] - x_vec[0], y_vec[1] - y_vec[0]
    extent = [x_vec[0] - dx / 2., x_vec[-1] + dx / 2., y_vec[0] - dy / 2., y_vec[-1] + dy / 2.]
    for ax, im, title in ((ax_sum, im_sum, "Sum"), (ax_std, im_std, "Std")):
        finite_im = im[np.isfinite(im)]
        med_im = np.nanmedian(im)
        mad_im = median_abs_deviation(finite_im) if finite_im.size > 0 else np.nan
        im_handle = ax.imshow(im, origin="lower", cmap="viridis", extent=extent,
                              vmin=med_im - 5 * mad_im, vmax=med_im + 5 * mad_im)
        cbar = fig2.colorbar(im_handle, ax=ax)
        cbar.set_label("{0} flux (MJy)".format(title), fontsize=fontsize)
        ax.set_xlim([-2, 2])
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_ylim([-2, 2])
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.invert_xaxis()
        ax.set_aspect('equal')
        ax.set_xlabel(r"$\Delta$RA (as)", fontsize=fontsize)
        ax.set_ylabel(r"$\Delta$Dec (as)", fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.set_title(title, fontsize=fontsize)

    for i_planet, ((x, y), label) in enumerate(zip(xy_coords_list, labels)):
        color = palette[i_planet % len(palette)]
        # Build a per-planet output filename
        if single_planet_no_label:
            planet_out_filename = out_filename
        else:
            planet_out_filename = out_filename.replace(".fits", "_{0}.fits".format(label))

        r2comp_grid = np.sqrt((x_grid - x) ** 2 + (y_grid - y) ** 2)
        kmax, lmax = np.unravel_index(np.nanargmin(r2comp_grid), r2comp_grid.shape)

        sep_comp = np.sqrt(x ** 2 + y ** 2)
        speckles_mask = (r2star_grid > (sep_comp - 0.05)) * (r2star_grid < (sep_comp + 0.05) * (r2comp_grid > 0.3))
        whereannulus = np.where(speckles_mask)

        speckles = flux_cube[:, whereannulus[0], whereannulus[1]]
        speckle_std = np.nanstd(speckles, axis=1)
        speckle_std[np.where(speckle_std == 0)] = np.nan

        spectrum = flux_cube[:, kmax, lmax]
        err = fluxerr_cube[:, kmax, lmax]

        spectrum_Flambda = to_Flambda(spectrum)
        err_Flambda = to_Flambda(err)
        speckle_std_Flambda = to_Flambda(speckle_std)
        speckles_Flambda = (speckles * u.MJy * const.c / (wv_sampling[:, None] * u.um) ** 2).to(u.W * u.m ** -2 / u.um).value

        spectrum_Jy = (spectrum * u.MJy).to(u.Jy).value
        speckle_std_Jy = (speckle_std * u.MJy).to(u.Jy).value
        speckles_Jy = (speckles * u.MJy).to(u.Jy).value

        # --- Write FITS output for this planet ---
        hdulist_out = fits.HDUList()
        hdulist_out.append(fits.ImageHDU(data=wv_sampling, name='WAVE'))
        hdulist_out.append(fits.ImageHDU(data=spectrum, name='FLUX_MJy'))
        hdulist_out.append(fits.ImageHDU(data=err, name='ERR_MJy'))
        hdulist_out.append(fits.ImageHDU(data=speckle_std, name='STD_MJy'))
        hdulist_out.append(fits.ImageHDU(data=speckles, name='SPECKLES'))
        hdulist_out.append(fits.ImageHDU(data=spectrum_Flambda, name='FLUX_FLAM'))
        hdulist_out.append(fits.ImageHDU(data=err_Flambda, name='ERR_FLAM'))
        hdulist_out.append(fits.ImageHDU(data=speckle_std_Flambda, name='STD_FLAM'))
        hdulist_out.append(fits.ImageHDU(data=speckles_Flambda, name='SPECKLES_FLAM'))
        if wv_nodes is not None:
            hdulist_out.append(fits.ImageHDU(data=wv_nodes, name='wv_nodes'))
        hdulist_out[0].header['LABEL'] = label
        hdulist_out[0].header['XCOORD'] = x
        hdulist_out[0].header['YCOORD'] = y
        try:
            hdulist_out.writeto(planet_out_filename, overwrite=True)
        except TypeError:
            hdulist_out.writeto(planet_out_filename, clobber=True)
        hdulist_out.close()

        # --- 1D spectrum: add traces for this planet to the shared Plotly figure ---
        upper = spectrum_Flambda + speckle_std_Flambda
        lower = spectrum_Flambda - speckle_std_Flambda
        # Shaded noise band via the 'tonexty' pattern: an invisible upper-bound
        # trace, followed by a lower-bound trace that fills up to the previous one.
        # This is more robust to NaN gaps than the 'toself' polygon-concatenation
        # trick, which can self-intersect and balloon out where speckle_std is NaN.
        N_speckles = speckles_Flambda.shape[1]
        speckle_subset = speckles_Flambda[:, ::N_speckles // 5]
        for i in range(speckle_subset.shape[1]):
            fig_spec.add_trace(go.Scatter(
                x=wv_sampling, y=speckle_subset[:, i], mode='lines',
                name="speckles " + label, legendgroup=label,
                opacity=0.5, showlegend=(i == 0),
                line=dict(color="grey", width=1),
            ), row=1, col=1)
        if wv_nodes is not None:
            marker = dict(symbol='line-ns', size=12, line=dict(color='black', width=2))
            fig_spec.add_trace(go.Scatter(
                x=wv_nodes, y=np.zeros(wv_nodes.shape)+np.nanmean(spectrum_Flambda + 5*speckle_std_Flambda), mode='markers',
                marker=marker,
                showlegend=False, hoverinfo='skip', legendgroup=label,
            ), row=1, col=1)
        fig_spec.add_trace(go.Scatter(
            x=wv_sampling, y=upper, mode='lines', line=dict(width=0),
            connectgaps=True,
            showlegend=False, hoverinfo='skip', legendgroup=label,
        ), row=1, col=1)
        fig_spec.add_trace(go.Scatter(
            x=wv_sampling, y=lower, mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor=hex_to_rgba(color, 0.15),
            connectgaps=True,
            showlegend=False, hoverinfo='skip', legendgroup=label,
        ), row=1, col=1)
        fig_spec.add_trace(go.Scatter(
            x=wv_sampling, y=spectrum_Flambda, mode='lines', name=label,
            legendgroup=label, line=dict(color=color, width=2),
        ), row=1, col=1)

        # --- Same panel, in Jansky, added below the Flambda panel ---
        upper_Jy = spectrum_Jy + speckle_std_Jy
        lower_Jy = spectrum_Jy - speckle_std_Jy
        speckle_subset_Jy = speckles_Jy[:, ::N_speckles // 5]
        for i in range(speckle_subset_Jy.shape[1]):
            fig_spec.add_trace(go.Scatter(
                x=wv_sampling, y=speckle_subset_Jy[:, i], mode='lines',
                name="speckles " + label, legendgroup=label,
                opacity=0.5, showlegend=False,
                line=dict(color="grey", width=1),
            ), row=2, col=1)
        if wv_nodes is not None:
            marker = dict(symbol='line-ns', size=12, line=dict(color='black', width=2))
            fig_spec.add_trace(go.Scatter(
                x=wv_nodes, y=np.zeros(wv_nodes.shape)+np.nanmean(spectrum_Jy + 5*speckle_std_Jy), mode='markers',
                marker=marker,
                showlegend=False, hoverinfo='skip', legendgroup=label,
            ), row=2, col=1)
        fig_spec.add_trace(go.Scatter(
            x=wv_sampling, y=upper_Jy, mode='lines', line=dict(width=0),
            connectgaps=True,
            showlegend=False, hoverinfo='skip', legendgroup=label,
        ), row=2, col=1)
        fig_spec.add_trace(go.Scatter(
            x=wv_sampling, y=lower_Jy, mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor=hex_to_rgba(color, 0.15),
            connectgaps=True,
            showlegend=False, hoverinfo='skip', legendgroup=label,
        ), row=2, col=1)
        fig_spec.add_trace(go.Scatter(
            x=wv_sampling, y=spectrum_Jy, mode='lines', name=label,
            legendgroup=label, showlegend=False, line=dict(color=color, width=2),
        ), row=2, col=1)

        # --- 2D image plot: per-planet annotations only (image itself plotted once, above) ---
        for ax in (ax_sum, ax_std):
            ax.plot(0, 0, "*", color="grey", markersize=10)
            txt = ax.text(0.2, -0.1, 'A', fontsize=fontsize, ha='center', va='top', color="grey")
            txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])

            circle = plt.Circle((x, y), 0.2, facecolor='#FFFFFF00', edgecolor='white')
            txt = ax.text(x - 0.2, y, label, fontsize=fontsize, ha='left', va='center', color="black")
            txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
            ax.add_patch(circle)

        results.append({
            'label': label,
            'out_filename': planet_out_filename,
            'wv_sampling': wv_sampling,
            'spectrum_Flambda': spectrum_Flambda,
            'speckle_std_Flambda': speckle_std_Flambda,
        })

    # --- Save the combined interactive 1D spectra as HTML ---
    fig_spec.update_layout(
        template="plotly_white",
        width=1000, height=650,
        margin=dict(l=60, r=20, t=20, b=50),
    )
    fig_spec.update_xaxes(title_text="Wavelength (um)", row=2, col=1)
    # Force standard scientific notation (1.2e-14) instead of Plotly's default
    # SI-prefix ticks (12f, 12a, ...), which are unfamiliar for these units.
    fig_spec.update_yaxes(title_text="Flux (W/m2/um)", exponentformat='e', tickformat='.2e', row=1, col=1)
    fig_spec.update_yaxes(title_text="Flux (Jy)", exponentformat='e', tickformat='.2e', row=2, col=1)
    fig_spec.write_html(out_filename.replace(".fits", "_1dspec.html"))

    # --- Save the combined 2D image as PNG ---
    fig2.savefig(out_filename.replace(".fits", "_im.png"), dpi=200)
    plt.close(fig2)

    return results


def compute_snr_grid(cal_files, utils_dir, out_dir,
                    combined_contnorm_spec_filename, coords_filename_filter,
                    model_grid_h5py,grid_paras,photfilter_filename,N_KL=3,
                    suffix=None,contnorm_suffix=None,
                    x_vec=None,y_vec=None,rv_vec=None,
                    mask_charge_transfer_radius=None,ra_dec_point_sources=None,fix_fitting_region_around_xy=None,
                    aper_radius=0.15,
                    mppool=None,
                    overwrite=False,
                    targetname=None,
                    use_stpsf=False, use_breadspsf=True,
                    ifucoords=False):

    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if use_breadspsf is not None and not (isinstance(use_breadspsf, bool) and not use_breadspsf):
        use_stpsf = False

    if suffix is None:
        suffix = "snr"

    grating = fits.getheader(cal_files[0])['GRATING'].strip()
    detector = fits.getheader(cal_files[0])['DETECTOR'].strip().lower()

    if x_vec is None:
        x_vec = np.arange(-2, 2, 0.05)
    if y_vec is None:
        y_vec = np.arange(-2, 2, 0.05)
    if rv_vec is None:
        rv_vec = np.array([0])

    flux_arr_list=[]
    fluxerr_arr_list=[]
    log_prob_list=[]
    for filename in cal_files:
        print(filename)

        grid_search_output_filename = os.path.join(out_dir, os.path.basename(filename).replace(".fits", "_"+ suffix + ".fits"))
        if not overwrite and os.path.exists(grid_search_output_filename):
            with fits.open(grid_search_output_filename) as hdulist:
                _priheader = hdulist[0].header
                _extheader = hdulist[1].header
                flux_arr = hdulist['flux_arr'].data
                fluxerr_arr = hdulist['fluxerr_arr'].data
                log_prob = hdulist['log_prob'].data
                x_vec = hdulist['x_vec'].data
                y_vec = hdulist['y_vec'].data
                rv_vec = hdulist['rv_vec'].data
                wv_nodes = hdulist['wv_nodes'].data
                _breads_header = hdulist['BREADS'].data

                # from scipy.ndimage import generic_filter
                # log_prob[0,:,:] = generic_filter(log_prob[0,:,:], np.nanmedian, size=3)
            flux_arr_list.append(flux_arr)
            fluxerr_arr_list.append(fluxerr_arr)
            log_prob_list.append(log_prob)

            continue

        dataobj = JWSTNirspec_cal(filename, utils_dir=utils_dir)

        if contnorm_suffix is not None:
            dataobj.default_filenames["compute_starspectrum_contnorm"] = dataobj.default_filenames["compute_starspectrum_contnorm"].replace(".fits", "_" + contnorm_suffix + ".fits")
            dataobj.default_filenames["compute_advanced_badpix"] = dataobj.default_filenames["compute_starspectrum_contnorm"].replace(".fits", "_" + contnorm_suffix + ".fits")
            dataobj.default_filenames["compute_starsubtraction"] = dataobj.default_filenames["compute_starsubtraction"].replace(".fits", "_" + contnorm_suffix + ".fits")

        hdulist = fits.open(dataobj.default_filenames["compute_starspectrum_contnorm"])
        spline_paras0 = hdulist['SPLINE_PARAS0'].data
        if 'STELLAR_FEATURES' in hdulist:
            stellar_features0 = hdulist['STELLAR_FEATURES'].data
            with_3dspline_prior = True
        else:
            stellar_features0 = None
            with_3dspline_prior = False
        wv_nodes = hdulist['wv_nodes'].data
        hdulist.close()



        # Define a series of processing tasks to be performed on each input file.
        # coords_filename = glob(fitpsf_filename.replace(".fits","_poly_centroid*.txt"))[0]
        preproc_task_list = [
                                 ["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}],
                                 ["compute_coordinates_arrays", {'targname': targetname}]
                            ]

        if with_3dspline_prior:
            _task = ["compute_advanced_badpix", {"threshold_badpix": 10, "mppool": mppool,
                                                 "combined_contnorm_filename": combined_contnorm_spec_filename,
                                                 "load_starspectrum_contnorm": None}]
        else:
            _task = ["compute_advanced_badpix", {"threshold_badpix": 10, "mppool": mppool,
                                                 "combined_contnorm_filename": combined_contnorm_spec_filename,
                                                 "load_starspectrum_contnorm": dataobj.default_filenames["compute_starspectrum_contnorm"]}]
        preproc_task_list.append(_task)

        dataobj.run_preproc_list(save_utils=True, load_utils=True, preproc_task_list=preproc_task_list)

        hdulist = fits.open(combined_contnorm_spec_filename)
        new_wavelengths = hdulist["WAVE"].data
        combined_fluxes = hdulist["COM_FLUXES"].data
        hdulist.close()
        dataobj.star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False, fill_value=1)
        # plt.plot(new_wavelengths, combined_fluxes)
        # plt.show()

        coords_filename = glob(os.path.join(utils_dir, os.path.basename(filename).replace(".fits", coords_filename_filter)))[0]
        dataobj.apply_coords_offset(coords_filename=coords_filename)

        # Do some masking
        if mask_charge_transfer_radius is not None:
            dataobj.compute_charge_bleeding_mask(threshold2mask=mask_charge_transfer_radius)

        if ifucoords:
            dataobj.set_coords2ifu()

        if use_stpsf:
            webbpsf_reload = dataobj.reload_quick_webbpsf_model()
            if webbpsf_reload is None:
                print("Did not find a quick STPSF, computing it now.")
                webbpsf_reload = dataobj.compute_quick_webbpsf_model(save_utils=True)
        elif use_breadspsf is not None and not (isinstance(use_breadspsf, bool) and not use_breadspsf):
            BREADS_DATA_ENV = os.getenv('BREADS_DATA')
            if isinstance(use_breadspsf, bool) and use_breadspsf:
                grating = dataobj.priheader['GRATING'].strip()
                detector = dataobj.priheader['DETECTOR'].strip().lower()
                if os.path.exists(
                        os.path.join(BREADS_DATA_ENV, "BreadsPSF", f"HD163466_J1757132_{grating}_{detector}.fits")):
                    use_breadspsf_str = f"HD163466_J1757132_{grating}_{detector}.fits"
                else:
                    use_breadspsf_str = f"J1757132_{grating}_{detector}.fits"
            elif isinstance(use_breadspsf, str):
                use_breadspsf_str = use_breadspsf
            breadsPSF_path = os.path.join(BREADS_DATA_ENV, "BreadsPSF", use_breadspsf_str)
            dataobj.reload_breadspsf_model(breadsPSF_path)


        if stellar_features0 is None:
            hdulist = fits.open(dataobj.default_filenames["compute_starsubtraction"])
            spline_paras0 = hdulist['SPLINE_PARAS0'].data
            hdulist.close()
            wherenan = np.where(np.isnan(spline_paras0))
            reg_mean_map = copy(spline_paras0)
            reg_mean_map[wherenan] = np.tile(np.nanmedian(spline_paras0, axis=1)[:, None], (1, spline_paras0.shape[1]))[wherenan]
            reg_std_map = np.abs(spline_paras0)
            reg_std_map[wherenan] = np.tile(np.nanmax(np.abs(spline_paras0), axis=1)[:, None], (1, spline_paras0.shape[1]))[wherenan]
            reg_std_map = reg_std_map
            reg_std_map = np.clip(reg_std_map, 1e-11/2.3504430539097893e-13, np.inf)
        else:
            reg_mean_map = None
            reg_std_map = None
        # reg_mean_map = None
        # reg_std_map = None

        # plt.imshow(reg_mean_map[1200:1300,:],origin="lower")
        # plt.clim(-50000,50000)
        # plt.show()
        # N_KL=0
        if N_KL is not None and N_KL != 0:
            tmp_badpixels = copy(dataobj.bad_pixels)
            # mask point sources
            if ra_dec_point_sources is not None:
                _badpix_map_cp = copy(dataobj.bad_pixels)
                for ra_pl, dec_pl in ra_dec_point_sources:
                    if "sky" in dataobj.breads_header['COORDS']:
                        x_pl, y_pl = ra_pl, dec_pl
                    elif "ifu" in dataobj.breads_header['COORDS']:
                        _out = dataobj.get_ifu_coords(ras=ra_pl, decs=dec_pl)
                        x_pl, y_pl = float(_out[0]), float(_out[1])
                    where_pl = dataobj.where_point_source([x_pl / 1000., y_pl / 1000.], 0.16)
                    # dataobj.bad_pixels[where_pl] = np.nan
                tmp_badpixels[where_pl] = np.nan

            hdulist = fits.open(dataobj.default_filenames["compute_starsubtraction"])
            subtracted_im = hdulist["IM_SUB"].data
            hdulist.close()

            # plt.figure()
            # plt.subplot(1,2,1)
            # plt.imshow(subtracted_im,origin="lower")
            # plt.clim([-500,500])
            # plt.subplot(1,2,2)
            # plt.imshow(dataobj.data,origin="lower")
            # plt.clim([-100000,100000])
            # # plt.show()

            # N_KL = 5
            first_half = np.where(dataobj.wavelengths < np.nanmedian(dataobj.wavelengths))
            second_half = np.where(dataobj.wavelengths > np.nanmedian(dataobj.wavelengths))
            wv4pca, im4pcs, n4pca, bp4pca = copy(dataobj.wavelengths), copy(subtracted_im), copy(dataobj.noise), copy(tmp_badpixels)
            bp4pca[second_half] = np.nan
            KLs_wvs_left, KLs_left = PCA_wvs_axis(wv4pca, im4pcs, n4pca, bp4pca,
                                                  np.nanmedian(dataobj.wavelengths) / (4 * dataobj.R),
                                                  N_KL=N_KL)
            wv4pca, im4pcs, n4pca, bp4pca = copy(dataobj.wavelengths), copy(subtracted_im), copy(dataobj.noise), copy(tmp_badpixels)
            bp4pca[first_half] = np.nan
            KLs_wvs_right, KLs_right = PCA_wvs_axis(wv4pca, im4pcs, n4pca, bp4pca,
                                                    np.nanmedian(dataobj.wavelengths) / (4 * dataobj.R),
                                                    N_KL=N_KL)
            # wv4pca, im4pcs, n4pca, bp4pca = copy(dataobj.wavelengths), copy(subtracted_im), copy(dataobj.noise), copy(tmp_badpixels)
            # KLs_wvs_all, KLs_all = PCA_wvs_axis(wv4pca, im4pcs, n4pca, bp4pca,
            #                                     np.nanmedian(dataobj.wavelengths) / (4 * dataobj.R), N_KL=N_KL)

            # plt.figure()
            wvs_KLs_f_list = []
            for k in range(KLs_left.shape[1]):
                # print(k,np.where(np.isnan(KLs_left[:, k]))[0])
                # plt.plot(KLs_wvs_left, KLs_left[:, k],label=f"{k}")
                KL_f = interp1d(KLs_wvs_left, KLs_left[:, k], bounds_error=False, fill_value=0.0, kind="cubic")
                wvs_KLs_f_list.append(KL_f)
                # plt.plot(KLs_wvs_left, KLs_left[:, k])
            for k in range(KLs_right.shape[1]):
                # print("l",k,np.where(np.isnan(KLs_right[:, k]))[0])
                # plt.plot(KLs_wvs_right, KLs_right[:, k],label=f"{k}")
                KL_f = interp1d(KLs_wvs_right, KLs_right[:, k], bounds_error=False, fill_value=0.0,
                                kind="cubic")
                wvs_KLs_f_list.append(KL_f)
                # plt.plot(KLs_wvs_right, KLs_right[:, k])
            # plt.legend()
            # plt.show()
        else:
            wvs_KLs_f_list = None

        # Read and normalize BT settl grid
        filter_arr = np.loadtxt(photfilter_filename)
        trans_wvs = filter_arr[:, 0] / 1e4
        trans = filter_arr[:, 1]
        photfilter_f = interp1d(trans_wvs, trans, bounds_error=False, fill_value=0)
        photfilter_wv0 = np.nansum(trans_wvs * photfilter_f(trans_wvs)) / np.nansum(photfilter_f(trans_wvs))
        bandpass = np.where(photfilter_f(trans_wvs) / np.nanmax(photfilter_f(trans_wvs)) > 0.01)
        photfilter_wvmin, photfilter_wvmax = trans_wvs[bandpass[0][0]], trans_wvs[bandpass[0][-1]]
        # print(photfilter_wvmin, photfilter_wvmax)

        # Define planet model grid from BTsettl
        minwv, maxwv = np.min(dataobj.wavelengths), np.max(dataobj.wavelengths)
        with h5py.File(model_grid_h5py,'r') as hf:
            grid_specs = np.array(hf.get("spec"))
            grid_temps = np.array(hf.get("temps"))
            grid_loggs = np.array(hf.get("loggs"))
            grid_wvs = np.array(hf.get("wvs"))
        grid_dwvs = grid_wvs[1::] - grid_wvs[0:np.size(grid_wvs) - 1]
        grid_dwvs = np.insert(grid_dwvs, 0, grid_dwvs[0])
        filter_norm = np.nansum((grid_dwvs * u.um) * photfilter_f(grid_wvs))
        Flambda = np.nansum((grid_dwvs * u.um)[None, None, :] * photfilter_f(grid_wvs)[None, None, :] * (
                    grid_specs * u.W * u.m ** -2 / u.um), axis=2) / filter_norm
        Fnu = Flambda * (photfilter_wv0 * u.um) ** 2 / const.c  # from Flambda back to Fnu
        grid_specs = grid_specs / Fnu[:, :, None].to(u.MJy).value

        myinterpgrid = RegularGridInterpolator((grid_temps, grid_loggs), grid_specs, method="linear",
                                               bounds_error=False, fill_value=np.nan)
        # teff, logg, vsini, rv, dra_comp, ddec_comp = 1500, 5.0, 0.0, None, None, None
        teff, logg, vsini = grid_paras
        rv, dra_comp, ddec_comp = None, None, None
        fix_parameters = [teff, logg, vsini, rv, dra_comp, ddec_comp]

        fm_paras = {"atm_grid": myinterpgrid, "atm_grid_wvs": grid_wvs, "star_func": dataobj.star_func,
                    "radius_as": aper_radius, "badpixfraction": 0.5, "nodes": wv_nodes,
                    "fix_parameters": fix_parameters,
                    "wvs_KLs_f": wvs_KLs_f_list,
                    "regularization": "user","reg_mean_map":reg_mean_map, "reg_std_map":reg_std_map,"stellar_features0":stellar_features0,#
                    "use_stpsf":use_stpsf,
                    "fix_fitting_region_around_xy":fix_fitting_region_around_xy}
        fm_func = hc_atmgrid_splinefm_jwst_ifu_cal

        if 0:
            print("ra_dec_point_sources",ra_dec_point_sources)
            # nonlin_paras = [16.84, ra_dec_point_sources[0][1]/1000., ra_dec_point_sources[0][0]/1000.]  # rv (km/s),y (pix), x (pix),
            # print("nonlin_paras",nonlin_paras)
            # nonlin_paras = [16.84, 0.1, 0.6]  # rv (km/s),y (pix), x (pix),
            nonlin_paras = (np.float64(16.84), np.float64(0.35000000000000003), np.float64(0.2500000000000011))
            print("nonlin_paras",nonlin_paras)
            # exit()
            # nonlin_paras = [0.0, ra_dec_point_sources[1][1]/1000., ra_dec_point_sources[1][0]/1000.]
            # d is the data vector a the specified location
            # M is the linear component of the model. M is a function of the non linear parameters x,y,rv
            # s is the vector of uncertainties corresponding to d
            d, M, s, extra_outputs = fm_func(nonlin_paras, dataobj, **fm_paras)
            where_finite = extra_outputs["where_trace_finite"]
            w = extra_outputs["wvs"]
            x = extra_outputs["ras"]
            y = extra_outputs["decs"]
            rows = extra_outputs["rows"]
            d_reg, s_reg = extra_outputs["regularization"]
            reg_wvs = extra_outputs["regularization_wvs"]
            reg_rows = extra_outputs["regularization_rows"]

            hdulist = fits.open(dataobj.default_filenames["compute_starsubtraction"])
            subtracted_im = hdulist["IM_SUB"].data
            hdulist.close()
            # plt.subplot(1,2,1)
            # plt.imshow(subtracted_im,origin="lower")
            # plt.clim([-500,500])
            # plt.ylim([1300,1500])
            # plt.xlim([800,1000])
            # plt.subplot(1,2,2)
            # canvas = np.zeros(subtracted_im.shape)
            # canvas[where_finite] = M[:,0]
            # plt.imshow(canvas,origin="lower")
            # # plt.clim([-500,500])
            # plt.ylim([1300,1500])
            # plt.xlim([800,1000])
            # plt.show()


            # M[:,0] = 0s

            # validpara = np.where(np.max(np.abs(M), axis=0) != 0)
            validpara = np.where(~np.isclose(np.nansum(np.abs(M/ s[:, None]), axis=0), 0, atol=1e-10))
            M = M[:, validpara[0]]
            print(M.shape)

            d = d / s
            M = M / s[:, None]

            from breads.fit import fitfm
            log_prob, rchi2, linparas, linparas_err = fitfm(nonlin_paras, dataobj, fm_func, fm_paras,scale_noise=False, bounds=None)
            print("best fit", linparas[0:5])
            print("best fit err", linparas_err[0:5])
            print("best fit snr", linparas[0:5] / linparas_err[0:5])
            print("rchi2", rchi2)
            print("log_prob", log_prob)
            nonlin_paras = [nonlin_paras[0],nonlin_paras[1]+1e-5,nonlin_paras[2]]
            log_prob, rchi2, linparas, linparas_err = fitfm(nonlin_paras, dataobj, fm_func, fm_paras,scale_noise=False, bounds=None)

            paras = linparas[validpara]
            print("best fit", linparas[0:5])
            print("best fit err", linparas_err[0:5])
            print("best fit snr", linparas[0:5] / linparas_err[0:5])
            print("rchi2", rchi2)
            print("log_prob", log_prob)
            plt.figure()
            plt.plot(np.nanmax(np.abs(M),axis=0))
            plt.figure()
            plt.plot(linparas[validpara], label="linparas")
            plt.plot(linparas_err[validpara], label="linparas_err")
            plt.legend()
            # plt.show()

            # logdet_Sigma = np.sum(2 * np.log(s))
            m = np.dot(M, paras)
            # r = d - m
            # chi2 = np.nansum(r ** 2)
            # N_data = np.size(d)
            # rchi2 = chi2 / N_data
            # res = r * s
            # plt.plot(d,label="d")
            # plt.plot(m,label="m")
            # plt.legend()
            # plt.show()


            canvas = np.zeros(subtracted_im.shape)
            canvas[where_finite] = m*s
            # row_id = 1368
            row_id = np.argmax(np.nansum(canvas,axis=1))
            print(row_id)
            # canvas[where_finite] = M[:,0]
            # plt.imshow(canvas,origin="lower")
            # plt.clim([0,2e-6])
            # plt.show()
            steradians_to_arcsec2 = 1 / (2. * np.pi / (360. * 3600.)) ** 2
            print(((0.1)**2/steradians_to_arcsec2))
            # scaling = ((0.1)**2/steradians_to_arcsec2)/0.3
            scaling = 1.
            plt.figure()
            plt.subplot(3,1,1)
            plt.plot(dataobj.wavelengths[row_id,:], subtracted_im[row_id,:]*scaling,label="subtracted_im")
            plt.plot(dataobj.wavelengths[row_id,:], dataobj.data[row_id,:]*scaling,label="data")
            plt.plot(dataobj.wavelengths[row_id,:], dataobj.noise[row_id,:]*rchi2*scaling,label="noise")
            plt.plot(dataobj.wavelengths[row_id,:], canvas[row_id,:]*scaling,label="model")
            plt.plot(dataobj.wavelengths[row_id,:], (dataobj.data[row_id,:]-canvas[row_id,:])*scaling,label="res",linestyle="--")
            where_reg_row = np.where(reg_rows==row_id)
            d_reg, s_reg = extra_outputs["regularization"]
            reg_wvs = extra_outputs["regularization_wvs"]
            reg_rows = extra_outputs["regularization_rows"]
            plt.plot(reg_wvs[where_reg_row],d_reg[where_reg_row]*scaling,label="reg")
            plt.errorbar(reg_wvs[where_reg_row],d_reg[where_reg_row]*scaling,yerr=s_reg[where_reg_row]*scaling,label="regerr")
            plt.legend()

            plt.subplot(3,1,2)
            canvas = np.zeros(subtracted_im.shape)
            canvas[where_finite] = M[:,0]*s
            plt.plot(dataobj.wavelengths[row_id,:], canvas[row_id,:])

            plt.subplot(3,1,3)
            # plt.plot(dataobj.bad_pixels[row_id,:],label="bad_pixels")
            for k in np.arange(1,M.shape[1]):
                # print(reg_rows[validpara[0]][k])
                canvas = np.zeros(subtracted_im.shape)
                canvas[where_finite] = M[:,k]*s
                plt.plot(canvas[row_id,:])


            plt.figure()
            plt.plot( d * s, label="data")
            plt.plot( m * s, label="Combined model")
            plt.plot( paras[0] * M[:, 0] * s, label="planet model")
            plt.plot( (m - paras[0] * M[:, 0]) * s, label="starlight model")
            plt.plot( d * s-m * s, label="residuals")
            plt.plot(s*rchi2, label="noise")
            # where_even_rows = np.where((reg_rows % 2) == 0)
            # plt.errorbar(reg_wvs[where_even_rows], d_reg[where_even_rows], yerr=s_reg[where_even_rows],
            #              label="even rows prior")
            # where_odd_rows = np.where((reg_rows % 2) == 1)
            # plt.errorbar(reg_wvs[where_odd_rows], d_reg[where_odd_rows], yerr=s_reg[where_odd_rows],
            #              label="odd rows prior")
            plt.ylabel("Flux (MJy)")
            plt.xlabel("Column pixels")
            plt.legend()

            plt.figure()
            plt.subplot(2,1,1)
            plt.plot((d-m)/rchi2)
            plt.subplot(2,1,2)
            plt.plot(np.cumsum(d-m))
            plt.show()

        if mppool is not None:
            numthreads = mppool._processes
        else:
            numthreads = None
        log_prob, rchi2, linparas, linparas_err = grid_search([rv_vec, y_vec, x_vec], dataobj, fm_func, fm_paras,
                                                                           numthreads=numthreads, scale_noise=True)
        N_linpara = linparas.shape[-1]

        _priheader = dataobj.priheader
        _extheader = dataobj.extheader
        _breads_header = dataobj.breads_header

        flux_arr = linparas[:, :, :, 0]
        fluxerr_arr = linparas_err[:, :, :, 0]
        snr_arr = flux_arr/fluxerr_arr
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(header=_priheader))
        hdulist.append(fits.ImageHDU(data=flux_arr,header=_extheader,name="flux_arr"))
        hdulist.append(fits.ImageHDU(data=fluxerr_arr, name='fluxerr_arr'))
        hdulist.append(fits.ImageHDU(data=log_prob, name='log_prob'))
        hdulist.append(fits.ImageHDU(data=rchi2, name='rchi2'))
        hdulist.append(fits.ImageHDU(data=rv_vec, name='rv_vec'))
        hdulist.append(fits.ImageHDU(data=x_vec, name='x_vec'))
        hdulist.append(fits.ImageHDU(data=y_vec, name='y_vec'))
        hdulist.append(fits.ImageHDU(data=wv_nodes, name='wv_nodes'))
        hdulist.append(fits.ImageHDU(header=_breads_header, name='BREADS'))
        hdulist.writeto(grid_search_output_filename, overwrite=True)
        hdulist.close()


        flux_arr_list.append(flux_arr)
        fluxerr_arr_list.append(fluxerr_arr)
        log_prob_list.append(log_prob)


        fig0 = plt.figure(figsize=(12,12))
        dx, dy = x_vec[1] - x_vec[0], y_vec[1] - y_vec[0]
        extent = [x_vec[0] - dx / 2., x_vec[-1] + dx / 2., y_vec[0] - dy / 2., y_vec[-1] + dy / 2.]
        rv0_id = len(rv_vec)//2
        im_list = [flux_arr[rv0_id,:,:],fluxerr_arr[rv0_id,:,:],snr_arr[rv0_id,:,:],log_prob[rv0_id,:,:]]
        im_names = ["flux_arr", "fluxerr_arr", "snr_arr", "log_prob"]
        fontsize=12
        for k,(im,title) in enumerate(zip(im_list,im_names)):
            plt.subplot(2,2,k+1)
            ax = plt.gca()
            finite_im = im[np.isfinite(im)]
            med_im = np.nanmedian(im)
            mad_im = median_abs_deviation(finite_im) if finite_im.size > 0 else np.nan
            vmin = med_im - 5 * mad_im
            vmax = med_im + 5 * mad_im
            im_handle = ax.imshow(im, origin="lower", cmap="viridis", extent=extent,
                                  vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(im_handle, ax=ax)
            cbar.set_label("{0}".format(title), fontsize=fontsize)
            ax.invert_xaxis()
            ax.set_aspect('equal')
            ax.set_xlabel(r"$\Delta$x (as)", fontsize=fontsize)
            ax.set_ylabel(r"$\Delta$y (as)", fontsize=fontsize)
            ax.tick_params(axis='x', labelsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize)
            ax.set_title(title, fontsize=fontsize)
        # --- Save the combined 2D image as PNG ---
        fig0.tight_layout(pad=3.0, w_pad=3.0, h_pad=3.0)
        fig0.savefig(grid_search_output_filename.replace(".fits", ".png"), dpi=200)
        # plt.show()
        plt.close(fig0)


    fluxmap_arr = np.array(flux_arr_list)
    fluxerrmap_arr = np.array(fluxerr_arr_list)
    fluxmap_combined = np.nansum(fluxmap_arr / fluxerrmap_arr ** 2, axis=0) / np.nansum(1 / fluxerrmap_arr ** 2, axis=0)
    fluxerrmap_combined = 1 / np.sqrt(np.nansum(1 / fluxerrmap_arr ** 2, axis=0))
    snrmap_combined=fluxmap_combined/fluxerrmap_combined

    log_prob_arr = np.array(log_prob_list)
    log_prob_combined = np.sum(log_prob_arr, axis=0)


    splitbasename = os.path.basename(cal_files[0]).split("_")
    grid_search_combined_filename = os.path.join(out_dir,splitbasename[0] + "_" + splitbasename[1]+ "_" + detector+ "_" + grating+"_"+suffix + "_combined.fits")

    hdulist = fits.HDUList()
    hdulist.append(fits.PrimaryHDU(header=_priheader))
    hdulist.append(fits.ImageHDU(data=fluxmap_combined, header=_extheader, name="flux_arr"))
    hdulist.append(fits.ImageHDU(data=fluxerrmap_combined, name='fluxerr_arr'))
    hdulist.append(fits.ImageHDU(data=log_prob_combined, name='log_prob'))
    hdulist.append(fits.ImageHDU(data=rv_vec, name='rv_vec'))
    hdulist.append(fits.ImageHDU(data=x_vec, name='x_vec'))
    hdulist.append(fits.ImageHDU(data=y_vec, name='y_vec'))
    hdulist.append(fits.ImageHDU(data=wv_nodes, name='wv_nodes'))
    hdulist.append(fits.ImageHDU(header=_breads_header, name='BREADS'))
    hdulist.writeto(grid_search_combined_filename, overwrite=True)
    hdulist.close()

    fig0 = plt.figure(figsize=(12,12))
    dx, dy = x_vec[1] - x_vec[0], y_vec[1] - y_vec[0]
    extent = [x_vec[0] - dx / 2., x_vec[-1] + dx / 2., y_vec[0] - dy / 2., y_vec[-1] + dy / 2.]
    rv0_id = len(rv_vec) // 2
    im_list = [fluxmap_combined[rv0_id, :, :], fluxerrmap_combined[rv0_id, :, :], snrmap_combined[rv0_id, :, :], log_prob_combined[rv0_id, :, :]]
    im_names = ["flux_arr", "fluxerr_arr", "snr_arr", "log_prob"]
    fontsize = 12
    for k, (im, title) in enumerate(zip(im_list, im_names)):
        plt.subplot(2, 2, k + 1)
        ax = plt.gca()
        finite_im = im[np.isfinite(im)]
        med_im = np.nanmedian(im)
        mad_im = median_abs_deviation(finite_im) if finite_im.size > 0 else np.nan
        vmin = med_im - 5 * mad_im
        vmax = med_im + 5 * mad_im
        im_handle = ax.imshow(im, origin="lower", cmap="viridis", extent=extent,
                              vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im_handle, ax=ax)
        cbar.set_label("{0}".format(title), fontsize=fontsize)
        ax.invert_xaxis()
        ax.set_aspect('equal')
        ax.set_xlabel(r"$\Delta$x (as)", fontsize=fontsize)
        ax.set_ylabel(r"$\Delta$y (as)", fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
    # --- Save the combined 2D image as PNG ---
    fig0.tight_layout(pad=3.0, w_pad=3.0, h_pad=3.0)
    fig0.savefig(grid_search_combined_filename.replace(".fits", ".png"), dpi=200)
    # plt.show()
    plt.close(fig0)

    coords = (rv_vec,y_vec,x_vec)
    return fluxmap_combined, fluxerrmap_combined,log_prob_combined,coords


############################################################################
#  Function to invoke all reduction steps in one go


def run_complete_stage1_2_nirspec(uncal_files, output_root_dir, utils_dir, overwrite=False,numthreads=1,
                          clean_1f_noise = True, model_charge_transfer=False,mppool=None,targetname=None,
                                  extend_sat=2):
    """
    Overarching top-level function to invoke stage1, stage2, 1/f noise cleaning code on rate maps, and stage2 again with cleaned rate maps.

    This will run the complete reduction from uncal files to cal files. It will take a while.

    If files already exist, repeat reductions are skipped, unless overwrite is set True

    Parameters
    ----------

    Returns
    -------

    """
    grating_list = []
    for filename in uncal_files:
        grating_list.append(fits.getheader(filename)['GRATING'].strip())
    grating_list = np.array(grating_list)
    unique_gratings = np.unique(grating_list)
    uncal_files_dict = {}
    for grating in unique_gratings:
        uncal_files_dict[grating] = np.array(uncal_files)[np.where(grating_list == grating)]

    cal_files = {}
    for grating in unique_gratings:
        stage1_outdir = os.path.join(output_root_dir, grating + "_stage1")
        stage2_outdir = os.path.join(output_root_dir, grating + "_stage2")
        if clean_1f_noise or model_charge_transfer:
            stage1_clean_outdir = os.path.join(output_root_dir, grating + "_stage1_cleaned")
            stage2_clean_outdir = os.path.join(output_root_dir, grating + "_stage2_cleaned")

        _rate_files = run_stage1_nirspec(uncal_files_dict[grating], stage1_outdir, overwrite=overwrite,maximum_cores=f"{numthreads}")
        cal_files[grating] = run_stage2_nirspec(_rate_files, stage2_outdir, skip_cubes=True, overwrite=overwrite, TA=False,
                                       cleanflicker_skip=True, save_plots=True)
        if model_charge_transfer or clean_1f_noise:
            # Subtract charge transfer
            _rate_files = clean_rate_nirspec(_rate_files, stage2_outdir, stage1_clean_outdir, N_nodes=40,utils_dir=utils_dir,
                                             overwrite=overwrite, save_plots=True,mppool=mppool,targetname=targetname,
                                             clean_1f_noise=clean_1f_noise,model_charge_transfer=model_charge_transfer,
                                             extend_sat=extend_sat)

            cal_files[grating] = run_stage2_nirspec(_rate_files, stage2_clean_outdir, skip_cubes=True,
                                                   overwrite=overwrite, TA=False, cleanflicker_skip=False, save_plots=True)

    return cal_files

###########################################################################
# Functions for invoking the MIRI/MRS pipeline

def mkdir_miri_files(path):
    """Short function to create directories for MIRI files"""
    if type(path) != str:
        raise TypeError("'path' must be a string")

    if not os.path.exists(path):
        os.makedirs(path)
    return path

def sort_by_target_name(input_dir, filetype='uncal.fits'):
    files = find_files_to_process(input_dir, filetype)
    targname_groups = defaultdict(list)

    for file in files:
        header = fits.getheader(file)
        targname = header.get('TARGNAME', 'UNKNOWN')
        targname_groups[targname].append(file)

    return dict(targname_groups)


def select_miri_output_directory(uncal_path, target_name, channel, band):
    """Short function to select the right MIRI output directory"""

    if band == 'SHORT':
        band_alias = 'A'
    elif band == 'MEDIUM':
        band_alias = 'B'
    elif band == 'LONG':
        band_alias = 'C'
    else:
        raise ValueError(f"Band {band} is not supported for stage 1 forward modeling")
    return os.path.join(uncal_path, target_name, channel + band_alias, 'stage1')


def run_stage1_miri(uncal_path, list_bands=None, overwrite=False, maximum_cores="1", skip_dark=False):
    """Run pipeline stage 1, with some customizations for reductions"""

    if list_bands is None:
        list_bands = ['12A', '12B', '12C', '34A', '34B', '34C']

    dict_files_by_target_names = sort_by_target_name(uncal_path)
    target_names = list(dict_files_by_target_names.keys())
    print("DEBUG target_names", target_names)

    time0 = time.perf_counter()
    print(time0)
    rate_files = []

    for target_name in target_names:
        print("DEBUG target_name", target_name)
        uncal_files = dict_files_by_target_names[target_name]

        for band in list_bands:
            mkdir_miri_files(os.path.join(uncal_path, target_name, band, 'stage1'))

        rate_files = []

        for i, file in enumerate(uncal_files):
            print(f"Processing file {i + 1} of {len(uncal_files)}.")
            hdu_uncal = fits.open(file)
            band_uncal = hdu_uncal[0].header['BAND']
            channel_uncal = hdu_uncal[0].header['CHANNEL']
            output_dir = select_miri_output_directory(uncal_path, target_name, channel_uncal, band_uncal)

            new_name = os.path.basename(file).replace('uncal.fits', 'rate.fits')
            out_name = os.path.join(output_dir, new_name)

            rate_files.append(out_name)

            if os.path.exists(out_name) and not overwrite:
                print(f"Output file {out_name} already exists in output dir;\n\tskipping {file}.")
            else:
                det1 = Detector1Pipeline()  # Instantiate the pipeline
                # defining used pipeline steps
                # This version only shows the step parameters which are changes from defaults.
                step_parameters = {
                    # group_scale - run with defaults
                    # dq_init - run with defaults
                    'saturation': {'n_pix_grow_sat': 0},
                    # check for saturated pixels, but do not expand to adjacent pixels
                    # ipc - run with defaults
                    # superbias - run with defaults
                    # linearity - run with defaults
                    'emicorr': {'skip': False},
                    'persistence': {'skip': True},
                    # This step does nothing; there are no nonzero parameters in the reference files yet
                    'dark_current': {'skip': skip_dark},
                    'jump': {'maximum_cores': maximum_cores},  # parallelize
                    'ramp_fit': {'maximum_cores': maximum_cores},  # parallelize
                    # gain_scale : run with defaults
                }

                det1.call(file, save_results=True, output_dir=output_dir,
                          steps=step_parameters)

    # Print out the time benchmark
    time1 = time.perf_counter()

    print(f"Total Runtime: {time1 - time0:0.4f} seconds")

    return rate_files, target_names

def run_bkg_subtraction(uncal_path, target_name, list_bands=None, overwrite=False):
    if list_bands is None:
        list_bands = ['12A', '12B', '12C', '34A', '34B', '34C']

    for band in list_bands:
        output_dir = os.path.join(uncal_path, target_name, band, 'stage1_sub_bkg')
        mkdir_miri_files(output_dir)
        background_outputdir = os.path.join(uncal_path, target_name, band, 'master_bkg')
        mkdir_miri_files(background_outputdir)
        rate_files_all = find_files_to_process(os.path.join(uncal_path, target_name, band, 'stage1'), filetype='rate.fits')

        bkg_files = [f for f in rate_files_all if 'BACKGROUND' in fits.getheader(f)['OBSLABEL']
                     or 'BKG' in fits.getheader(f)['OBSLABEL']]
        rate_files = [f for f in rate_files_all if f not in bkg_files]

        bkg_master = np.zeros((len(bkg_files), 1024, 1032))
        for i, bkg_file in enumerate(bkg_files):
            bkg_master[i, :, :] = fits.getdata(bkg_file)
        bkg_master = np.nanmedian(bkg_master, axis=0)
        plt.imshow(bkg_master, origin='lower')
        plt.show()
        fits.writeto(os.path.join(background_outputdir,f"background_master_{band}.fits"), bkg_master, overwrite=overwrite)

        for fid, rate_file in enumerate(rate_files):
            out_name = os.path.join(output_dir, os.path.basename(rate_file))
            rate = fits.getdata(rate_file)
            with fits.open(rate_file, mode="readonly") as hdu:
                hdu_copy = fits.HDUList([hd.copy() for hd in hdu])
                hdu_copy[1].data = rate - bkg_master
                hdu_copy[0].header['BKG_SUB'] = 'CUSTOM'
                hdu_copy.writeto(out_name, overwrite=overwrite)


def flat_fringing_stage1(uncal_path, target_name, list_bands=None, flat_path=None, flat_extended=False, bkg_sub=False,
                         overwrite=False):
    if list_bands is None:
        list_bands = ['12A', '12B', '12C', '34A', '34B', '34C']

    for band in list_bands:
        mkdir_miri_files(os.path.join(uncal_path, target_name, band, 'stage1_flat'))

    # Start a timer to keep track of runtime
    time0 = time.perf_counter()
    print(time0)

    for band in list_bands:
        if bkg_sub:
            rate_files = find_files_to_process(os.path.join(uncal_path, target_name, band, 'stage1_sub_bkg'), filetype='rate.fits')
        else:
            rate_files = find_files_to_process(os.path.join(uncal_path, target_name, band, 'stage1'), filetype='rate.fits')
        output_dir = os.path.join(uncal_path, target_name, band, 'stage1_flat')
        rate_filtered_files = []
        for fid, rate_file in enumerate(rate_files):
            print(fid, rate_file)

            out_name = os.path.join(output_dir, os.path.basename(rate_file))
            rate_filtered_files.append(out_name)

            if os.path.exists(out_name) and not overwrite:
                print(f"Output file {out_name} already exists;\n\tskipping {rate_file}.")
                continue

            hdr = fits.getheader(rate_file)
            detector = hdr['DETECTOR']
            band = hdr['BAND']

            if flat_path is None:
                flat_path_rate = os.getenv("FLAT_PATH")
                if output_dir is None:
                    raise ValueError("No FLAT_PATH specified to apply the fringe flat")
            else:
                flat_path_rate = flat_path

            print("Searching fringes flat files in:", flat_path_rate)

            if detector == 'MIRIFUSHORT':
                if band == 'SHORT':
                    flat_path_rate = os.path.join(flat_path_rate, '12A')
                elif band == 'MEDIUM':
                    flat_path_rate = os.path.join(flat_path_rate, '12B')
                elif band == 'LONG':
                    flat_path_rate = os.path.join(flat_path_rate, '12C')
                else:
                    raise ValueError(f'Unsupported band for file: {rate_file} must be either SHORT, MEDIUM or LONG')
                channel = 'CH2'

            else:
                if band == 'SHORT':
                    flat_path_rate = os.path.join(flat_path_rate, '34A')
                elif band == 'MEDIUM':
                    flat_path_rate = os.path.join(flat_path_rate, '34B')
                elif band == 'LONG':
                    flat_path_rate = os.path.join(flat_path_rate, '34C')
                else:
                    raise ValueError(f'Unsupported band for file: {rate_file} must be either SHORT, MEDIUM or LONG')
                channel = 'CH3'

            best_flat, flat_name, std_min = best_flat_selection(rate_file, flat_path_rate, channel,
                                                                flat_extended=flat_extended)
            best_flat[np.isnan(best_flat)] = 1
            rate_file_data = fits.getdata(rate_file)

            with fits.open(rate_file, mode="readonly") as hdu:
                hdu_copy = fits.HDUList([hd.copy() for hd in hdu])
                hdu_copy[1].data = rate_file_data / best_flat
                hdu_copy['ERR'].data /= best_flat
                hdu_copy[0].header['S_FLAT'] = flat_name
                hdu_copy[0].header['FLAT_STD_MIN'] = std_min

                hdu_copy.writeto(out_name, overwrite=overwrite)
                print(f"==> Wrote fringe-corrected file to {out_name}")

def run_stage2_miri(uncal_path, target_name, list_bands=None, custom_flatted=True, custom_bkg_sub=False, skip_cubes=True, skip_fringe=False,
                    skip_residual_fringes=False,
                    skip_flatfield=False, skip_straylight=True, overwrite=False):
    if list_bands is None:
        list_bands = ['12A', '12B', '12C', '34A', '34B', '34C']

    for band in list_bands:
        mkdir_miri_files(os.path.join(uncal_path, target_name, band, 'stage2'))

    time0 = time.perf_counter()
    print(time0)

    cal_files = []
    for band in list_bands:
        if custom_flatted:
            rate_file_band_path = os.path.join(uncal_path, target_name, band, 'stage1_flat')
            print(f"Processing the custom flatted rate files in {rate_file_band_path} for stage 2.")
        else:
            if custom_bkg_sub:
                rate_file_band_path = os.path.join(uncal_path, target_name, band, 'stage1_sub_bkg')
                print(f"Processing the background subtracted rate files in {rate_file_band_path} for stage 2.")
            else:
                rate_file_band_path = os.path.join(uncal_path, target_name, band, 'stage1')
                print(f"Processing the rate files in {rate_file_band_path} for stage 2.")

        rate_files = find_files_to_process(rate_file_band_path, filetype='rate.fits')

        for fid, rate_file in enumerate(rate_files):
            print(fid, rate_file)

            # Setting up steps and running the Spec2 portion of the pipeline.
            outputdir = os.path.join(uncal_path, target_name, band, 'stage2')
            out_name = os.path.join(outputdir, os.path.basename(rate_file).replace('rate.fits', 'cal.fits'))
            cal_files.append(out_name)
            if os.path.exists(out_name) and not overwrite:
                print(f"Output file {out_name} already exists;\n\tskipping {rate_file}.")
                continue

            spec2 = Spec2Pipeline()
            # spec2.output_dir = spec2_dir
            step_parameters = {
                # spec2.imprint_subtract.skip = False
                # spec2.msa_flagging.skip = False
                # # spec2.srctype.source_type = 'POINT'
                # spec2.flat_field.skip = False
                # spec2.pathloss.skip = False
                # spec2.photom.skip = False
                'straylight': {'skip': skip_straylight},
                'flat_field': {'skip': skip_flatfield},
                'fringe': {'skip': skip_fringe},
                'residual_fringe': {'skip': skip_residual_fringes},
                'cube_build': {'skip': skip_cubes},  # We do not want or need interpolated cubes
                'extract_1d': {'skip': True},
                # spec3.cube_build.coord_system = 'skyalign'
            }
            spec2.save_bsub = True

            spec2.call(rate_file, save_results=True, output_dir=outputdir,
                       steps=step_parameters)

        # Print out the time benchmark
        time1 = time.perf_counter()
        print(f"Runtime so far: {time1 - time0:0.4f} seconds")

    time1 = time.perf_counter()
    print(f"Total Runtime: {time1 - time0:0.4f} seconds")
    return cal_files


def run_stage3_miri(uncal_path, target_name, list_bands=None, overwrite=False):
    if list_bands is None:
        list_bands = ['12A', '12B', '12C', '34A', '34B', '34C']
    for band in list_bands:
        outputdir = mkdir_miri_files(os.path.join(uncal_path, target_name, band, 'stage3'))
        if os.path.exists(os.path.join(outputdir, f'Level3_ch{band[0]}-short_s3d.fits')) and overwrite is False:
            print(f"Output file Level3_ch{band[0]}-short_s3d.fits already exists;\n\tskipping.")
            continue

        inputdir = os.path.join(uncal_path, target_name, band, 'stage2')

        # Start a timer to keep track of runtime
        time0 = time.perf_counter()
        print(time0)
        calfiles = find_files_to_process(inputdir, filetype='mirifushort_cal.fits')
        sstring = calfiles  # cal_files_dir + '*cal.fits'
        print(sstring)
        calfiles = np.array(sorted(sstring))
        print(calfiles)
        sortfiles = sort_calfiles(calfiles)  # Split them up into bands
        print('Found ' + str(len(calfiles)) + ' input files to process for stage 3')

        asnlist = []
        bands = ['12A', '12B', '12C', '34A', '34B', '34C']
        for ii in range(0, len(sortfiles)):
            thesefiles = sortfiles[ii]
            ninband = len(thesefiles)
            if (ninband > 0):
                filename = 'l3asn-' + bands[ii] + '.json'
                asnlist.append(filename)
                writel3asn(thesefiles, filename, 'Level3')
        print("asnlist", asnlist)

        runspec3(asnlist[0], outputdir)


def run_full_miri_default_pipeline(uncal_path, target_name, list_bands=None, overwrite=False):
    run_stage1_miri(uncal_path, list_bands=list_bands, overwrite=overwrite, maximum_cores="1", skip_dark=False)
    run_stage2_miri(uncal_path, target_name, list_bands=list_bands, custom_flatted=False, skip_cubes=False,
                    skip_fringe=False, skip_residual_fringes=True, skip_flatfield=False, skip_straylight=False,
                    overwrite=overwrite)
    run_stage3_miri(uncal_path, target_name, list_bands=list_bands, overwrite=overwrite)

    return 1


# Define a useful function to write out a Lvl3 association file from an input list
def writel3asn(files, asnfile, prodname, **kwargs):
    # Define the basic association of science files
    asn = afl.asn_from_list(files, rule=DMS_Level3_Base, product_name=prodname)
    # Add any background files to the association
    if ('bg' in kwargs):
        print("bg in kwargs")
        for bgfile in kwargs['bg']:
            asn['products'][0]['members'].append({'expname': bgfile, 'exptype': 'background'})
    # Write the association to a json file
    _, serialized = asn.dump()
    with open(asnfile, 'w') as outfile:
        outfile.write(serialized)


def sort_calfiles(files):
    channel = []
    band = []

    for file in files:
        hdr = (fits.open(file))[0].header
        channel.append(hdr['CHANNEL'])
        band.append(hdr['BAND'])
    channel = np.array(channel)
    band = np.array(band)

    indx = np.where((channel == '12') & (band == 'SHORT'))
    files12A = files[indx]
    indx = np.where((channel == '12') & (band == 'MEDIUM'))
    files12B = files[indx]
    indx = np.where((channel == '12') & (band == 'LONG'))
    files12C = files[indx]
    indx = np.where((channel == '34') & (band == 'SHORT'))
    files34A = files[indx]
    indx = np.where((channel == '34') & (band == 'MEDIUM'))
    files34B = files[indx]
    indx = np.where((channel == '34') & (band == 'LONG'))
    files34C = files[indx]

    return files12A, files12B, files12C, files34A, files34B, files34C


def runspec3(filename, outputdir):
    # This initial setup is just to make sure that we get the latest parameter reference files
    # pulled in for our files.  This is a temporary workaround to get around an issue with
    # how this pipeline calling method works.
    crds_config = Spec3Pipeline.get_config_from_reference('l3asn-12A.json')  # The exact asn file used doesn't matter
    spec3 = Spec3Pipeline.from_config_section(crds_config)

    spec3.output_dir = outputdir
    spec3.save_results = True

    spec3.master_background.skip = True  # Computes and subtracts a master background signal
    spec3.outlier_detection.skip = False  # Identifies and flags any pixels with values that produce outliers in overlapping regions of cube space
    spec3.mrs_imatch.skip = False  # Ensure that there are no jumps in the background between individual exposures
    spec3.cube_build.skip = False  # Build the composite data cubes
    spec3.extract_1d.skip = False  # Extract 1d spectra from the composite data cubes

    spec3.process(filename)


def best_flat_selection(cal_file, flat_dir, channel, flat_extended=False, save_png=True, full_output=False):
    """ For MIRI, look at a selection of possible fringe flats and find the best match for a given science file

    Parameters
    ----------
    cal_file
    flat_dir
    channel
    flat_extended : bool
        use the FLAT_EXTENDED extension, instead of regular FLAT?
    save_png : bool
        Save a PNG showing the fringes used to determine the best match
    full_output : bool
    """
    hdu = fits.open(cal_file)
    data = hdu[1].data

    hdr = hdu[0].header
    pattern_type = hdr['PATTTYPE']
    dither_direction = hdr['DITHDIRC']
    dither_numero = hdr['PATT_NUM']
    band = hdr['BAND']

    print(f"Band: {band}")

    filenames = os.listdir(flat_dir)

    std = []
    file = []

    brightest_col = column_median_max_channel(data, channel=channel)
    print(f"Brightest column for Channel {channel}: {brightest_col}")
    if save_png:
        xlim = [450, 500]
        plt.title(f"Best fringes flat pattern selection\nFor channel {channel} {band}, using brighest column: {brightest_col} ")
        plt.xlabel("Row index")
        plt.ylabel("Fringes transmission")
        plt.xlim(*xlim)
        plt.ylim([0.5, 1.5])
        file_name = hdr['FILENAME']
        col_data = data[:, brightest_col]
        while np.any(np.isnan(col_data)):
            import astropy
            col_data = astropy.convolution.interpolate_replace_nans(col_data, kernel=[1,1,1])
        continuum = gaussian_filter(col_data, sigma=8)
        fringe_data = col_data / continuum
        plt.plot(fringe_data, label='data')

    for filename in filenames:
        if filename.endswith(".fits"):
            flat_hdu = fits.open(os.path.join(flat_dir, filename))
            hdr_flat = flat_hdu[0].header

            if flat_extended:
                flat = flat_hdu['FLAT_EXTENDED'].data
            else:
                flat = flat_hdu['FLAT'].data

            if hdr_flat['PATT_NUM'] == dither_numero and hdr_flat['PATTTYPE'] == pattern_type and hdr_flat[
                'DITHDIRC'] == dither_direction and hdr_flat['BAND'] == band:
                d_f = data[:, brightest_col] / flat[:, brightest_col]
                d_f_hf = d_f - gaussian_filter(d_f, sigma=8)
                fringe_residuals_std = np.nanstd(d_f_hf)
                std.append(fringe_residuals_std)
                file.append(filename)
                print(filename, fringe_residuals_std)

                if save_png:
                    if std[-1] < 40:
                        plt.plot(flat[:, brightest_col], label=filename)
            flat_hdu.close()

    idx = np.nanargmin(std)
    std_min = np.nanmin(std)
    flat_name = file[idx]
    print("Flat selected:", flat_name)

    if save_png:
        plt.text(0.05, 0.05, "Flat selected: "+flat_name, transform=plt.gca().transAxes)
        plt.legend()
        plt.tight_layout()
        out_name = f"./fig_fringes_{os.path.splitext(os.path.basename(file_name))[0]}.png"

        plt.savefig(out_name)
        print(f"==> Plot saved to {out_name}")
        plt.close()

    hdu_best_flat = fits.open(os.path.join(flat_dir, flat_name))
    if flat_extended:
        best_flat = hdu_best_flat['FLAT_EXTENDED'].data
    else:
        best_flat = hdu_best_flat['FLAT'].data
    if full_output:
        return best_flat, flat_name, std_min, file, std
    else:
        return best_flat, flat_name, std_min

def column_median_max(mat):
    medianes = np.nanmedian(mat, axis=0)
    col_index = np.nanargmax(medianes)
    return col_index


def column_median_max_channel(data, channel='CH1'):
    if channel == 'CH1' or channel == 'CH4':
        brightest_col = column_median_max(data[:, :500])
    elif channel == 'CH2' or channel == 'CH3':
        brightest_col = column_median_max(data[:, 500:]) + 500
    else:
        raise ValueError('Channel must be CH1, CH2, CH3 or CH4')

    return brightest_col

## Breads function functions for miri

def compute_coordinates_offset(path_cal_files, channel, utils_dir, target_name=None, IWA=None, OWA=None):
    from breads.instruments.jwstmiri_cal import JWSTMiri_cal
    from breads.instruments.jwstmiri_multiple_cals import JWSTMiri_multiple_cals

    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)

    def find_observation_numbers(cal_files):
        observation_numbers = []
        for filename in cal_files:
            base = os.path.basename(filename).split('_')[0]
            observation_number = base[-6:-3]
            if observation_number not in observation_numbers:
                observation_numbers.append(observation_number)
        return observation_numbers

    cal_files = find_files_to_process(path_cal_files, filetype="cal.fits")
    observation_numbers = find_observation_numbers(cal_files)

    print('Observation numbers:', observation_numbers)
    coords_offset = []
    for observation_number in observation_numbers:
        dataobj_list = []
        for filename in cal_files:
            base = os.path.basename(filename).split('_')[0]
            obs_number_file = base[-6:-3]
            print(filename, 'Observation number:', obs_number_file)
            if obs_number_file == observation_number:
                print('yes', filename)

                preproc_task_list = []
                preproc_task_list.append(["compute_med_filt_badpix", {"window_size": 10, "mad_threshold": 20}, True, True])
                if target_name is not None:
                    preproc_task_list.append(["compute_coordinates_arrays", {"targname": target_name}])
                else:
                    preproc_task_list.append(["compute_coordinates_arrays"])
                preproc_task_list.append(["convert_MJy_per_sr_to_MJy"])
                preproc_task_list.append(["compute_quick_webbpsf_model"])

                dataobj = JWSTMiri_cal(filename, channel_reduction=channel, utils_dir=utils_dir,
                                       save_utils=True, load_utils=True, preproc_task_list=preproc_task_list)

                dataobj_list.append(dataobj)
        dataobj_combined = JWSTMiri_multiple_cals(dataobj_list)
        ra_offset, dec_offset = dataobj_combined.compute_new_coords_from_webbPSFfit(IWA=IWA, OWA=OWA)
        print(observation_number, ra_offset, dec_offset)
        coords_offset.append([observation_number, ra_offset, dec_offset])

    return coords_offset


def compute_normalized_stellar_spectrum_miri(cal_files, channel, utils_dir, coords_offset=(0, 0),
                                             wv_nodes=None, target_name=None,
                                             star_hf_subtraction=True, mppool=None,
                                             ra_dec_point_sources=None, overwrite=False):
    from breads.instruments.jwstmiri_cal import JWSTMiri_cal

    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)

    hdulist_sc = fits.open(cal_files[0])
    detector = hdulist_sc[0].header["DETECTOR"].strip().lower()

    hdulist_sc.close()

    splitbasename = os.path.basename(cal_files[0]).split("_")
    combined_contnorm_spec_filename = os.path.join(utils_dir, splitbasename[0] + "_" + splitbasename[
        1] + "_" + detector + "_starspec_contnorm_combined_1dspline.fits")

    if not overwrite:
        if len(glob(combined_contnorm_spec_filename)):
            with fits.open(combined_contnorm_spec_filename) as hdulist:
                new_wavelengths = hdulist[0].data
                combined_fluxes = hdulist[1].data
                combined_errors = hdulist[2].data
                combined_star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False,
                                              fill_value=1)
            return combined_star_func

    dataobj_list = []
    for filename in cal_files:
        print(filename)

        preproc_task_list = []
        preproc_task_list.append(["compute_med_filt_badpix", {"window_size": 10, "mad_threshold": 20}, True, True])
        if target_name is not None:
            preproc_task_list.append(["compute_coordinates_arrays", {"targname": target_name}])
        else:
            preproc_task_list.append(["compute_coordinates_arrays"])
        preproc_task_list.append(["convert_MJy_per_sr_to_MJy"])
        preproc_task_list.append(["compute_quick_webbpsf_model"])
        preproc_task_list.append(["apply_coords_offset", {"coords_offset": coords_offset}])
        preproc_task_list.append(["compute_starspectrum_contnorm", {"x_nodes": wv_nodes,
                                                                    "threshold_badpix": 100,
                                                                    "mppool": mppool, "star_hf_subtraction":star_hf_subtraction}, True, True])
        preproc_task_list.append(["compute_starsubtraction", {"starsub_dir": "starsub1d",
                                                              "threshold_badpix": 10,
                                                              "mppool": mppool}, True, True])

        dataobj = JWSTMiri_cal(filename, channel_reduction=channel, utils_dir=utils_dir,
                               save_utils=True, load_utils=True, preproc_task_list=preproc_task_list)

        # mask planets before computing the star spectrum
        if ra_dec_point_sources is not None:
            for ra_pl, dec_pl in ra_dec_point_sources:
                where_pl = dataobj.where_point_source([ra_pl / 1000., dec_pl / 1000.], 0.16)
                dataobj.bad_pixels[where_pl] = np.nan

        dataobj_list.append(dataobj)

    new_wavelengths, combined_fluxes, combined_errors = get_contnorm_spec(dataobj_list, spline2d=False,
                                                                               load_utils=False,
                                                                               out_filename=combined_contnorm_spec_filename,
                                                                               spec_R_sampling=2700 * 4,
                                                                               interpolation="linear")
    if star_hf_subtraction:
        combined_star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False, fill_value=1)
    else:
        combined_star_func = interp1d(np.arange(0, 30, 100), np.ones_like(np.arange(0, 30, 100)), kind="linear",
                                  bounds_error=False, fill_value=1)

    return combined_star_func

def compute_starlight_subtraction_miri(cal_files, channel, utils_dir, wv_nodes=None, target_name=None,
                                       combined_star_func=None, star_hf_subtraction=True, coords_offset=(0, 0), mppool=None):
    from breads.instruments.jwstmiri_cal import JWSTMiri_cal

    dataobj_list = []
    for filename in cal_files[0::]:
        print(filename)

        preproc_task_list = []
        preproc_task_list.append(["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}, True, True])
        preproc_task_list.append(["compute_coordinates_arrays", {"targname": target_name}, True, True])
        preproc_task_list.append(["convert_MJy_per_sr_to_MJy"])
        preproc_task_list.append(["apply_coords_offset", {"coords_offset": coords_offset}])
        if combined_star_func is None:
            preproc_task_list.append(["compute_starspectrum_contnorm", {"x_nodes": wv_nodes,
                                                                        "threshold_badpix": 100,
                                                                        "mppool": mppool, "star_hf_subtraction":star_hf_subtraction}, True, True])

        dataobj = JWSTMiri_cal(filename, channel_reduction=channel, utils_dir=utils_dir,
                               save_utils=True, load_utils=True, preproc_task_list=preproc_task_list)

        if combined_star_func is not None:
            if star_hf_subtraction:
                dataobj.reload_starspectrum_contnorm()
            else:
                combined_star_func = interp1d(np.arange(0,30,100), np.ones_like(np.arange(0,30,100)), kind="linear", bounds_error=False, fill_value=1)
            dataobj.star_func = combined_star_func

        outputs = dataobj.reload_starsubtraction()

        if outputs is None:
            outputs = dataobj.compute_starsubtraction(save_utils=True, starsub_dir="starsub1d",
                                                      threshold_badpix=10, mppool=mppool)
        dataobj_list.append(dataobj)

    return dataobj_list

def get_combined_regwvs_miri(dataobj_list, channel, wv_sampling=None, use_starsub1d=False, reload=False):
    from breads.instruments.jwstmiri_cal import JWSTMiri_cal
    from breads.instruments.jwstmiri_multiple_cals import JWSTMiri_multiple_cals

    regwvs_dataobj_list = []
    for dataobj in dataobj_list:

        if use_starsub1d:
            starsub_filename = os.path.join(dataobj.utils_dir, "starsub1d", os.path.basename(dataobj.filename))
            print("starsub1d path for combined regwvs miri:", starsub_filename)
            starsub_dataobj = JWSTMiri_cal(starsub_filename, channel_reduction=channel, utils_dir=dataobj.utils_dir)
            if dataobj.data_unit == 'MJy':
                replace_data = dataobj.convert_MJy_per_sr_to_MJy(data_in_MJy_per_sr=starsub_dataobj.data)
            elif dataobj.data_unit == "MJy/sr":
                replace_data = starsub_dataobj.data
            regwvs_filename = dataobj.default_filenames["compute_interpdata_regwvs"].replace("_regwvs.fits",
                                                                                             "_starsub1d_regwvs.fits")
        else:
            replace_data = None
            regwvs_filename = dataobj.default_filenames["compute_interpdata_regwvs"]
        print("regwvs path for combined regwvs miri:", regwvs_filename)
        if reload == True:
            regwvs_dataobj = dataobj.reload_interpdata_regwvs(load_filename=regwvs_filename)
        else:
            regwvs_dataobj = None

        if regwvs_dataobj is None:
            print("[DEBUG] get combined regwvs miri checking wv_sampling", wv_sampling.shape)
            regwvs_dataobj = dataobj.compute_interpdata_regwvs(save_utils=regwvs_filename, wv_sampling=wv_sampling,
                                                               replace_data=replace_data)
        regwvs_dataobj_list.append(regwvs_dataobj)

    regwvs_combdataobj = JWSTMiri_multiple_cals(regwvs_dataobj_list)

    return regwvs_combdataobj
