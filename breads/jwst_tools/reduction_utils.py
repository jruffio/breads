import os
import time
from glob import glob
from copy import copy
import fnmatch

import numpy as np
from scipy.stats import median_abs_deviation
from astropy.io import fits

import multiprocessing as mp
import matplotlib.pyplot as plt
import datetime
from scipy.ndimage import generic_filter, gaussian_filter
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
import matplotlib.tri as tri

from jwst.pipeline import Detector1Pipeline, Spec2Pipeline, Spec3Pipeline
from jwst.associations import asn_from_list as afl  # Tools for creating association files
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base

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
from breads.utils import get_spline_model
import breads.jwst_tools.plotting
from breads.instruments.jwstmiri_cal import JWSTMiri_cal
from breads.instruments.jwstmiri_cal import fitpsf_miri
from breads.instruments.jwstmiri_cal import get_contnorm_spec_miri
from breads.instruments.jwstmiri_multiple_cal import JWSTMiri_multiple_cals
from breads.jwst_tools.flat_miri_utils import oddEvenCorrectionImage

from collections import defaultdict


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
        Optional list of exposure numers. The list of files will be filtered to contiain
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
    """ Run pipeline stage 1, with some customizations for reductions
    intended to be used with breads for IFU high contrast

    Currently only tested on NIRSpec IFU data

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
        Re-reduce and overwrite outputs, if these data were already reduced before?
        Default is to SKIP re-reducing anything already reduced.
    maximum_cores : string
        Passed to JWST pipeline functions that use multiprocessing, such as ramp fit
    """
    from jwst.pipeline import Detector1Pipeline

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    time0 = time.perf_counter()

    rate_files = []

    for i, file in enumerate(uncal_files):
        print(f"Stage 1 Processing file {i + 1} of {len(uncal_files)}.")

        outname = os.path.join(output_dir, os.path.basename(file).replace('uncal.fits', 'rate.fits'))
        rate_files.append(outname)

        if os.path.exists(outname) and not overwrite:
            print(f"\tStage 1 Output file {os.path.basename(outname)} already exists in output dir;\n\tskipping {os.path.basename(file)}.")
            continue

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
    if save_plots:
        breads.jwst_tools.plotting.plot_2d_image_set(rate_files,
                                                     output_dir = output_dir,
                                                     suptitle="Stage 1 pipeline reduction results",
                                                     plot_label = 'stage1')

        return rate_files

def run_stage2(rate_files, output_dir, skip_cubes=True, overwrite=False, TA=False, nsclean_skip=False, save_plots=True):
    """ Run pipeline stage 2, with some customizations for reductions
    intended to be used with breads for IFU high contrast

    Currently only tested on NIRSpec IFU data


    Parameters
    ----------
    rate_files
    output_dir
    skip_cubes
    overwrite
    TA

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

    for fid, rate_file in enumerate(rate_files):
        print(f"Stage 2 Processing file {fid + 1} of {len(rate_files)}.")

        # Setting up steps and running the Spec2 portion of the pipeline.

        outname = os.path.join(output_dir, os.path.basename(rate_file).replace('rate.fits', 'cal.fits'))
        cal_files.append(outname)
        if os.path.exists(outname) and not overwrite:
            print(f"\tStage 2 Output file {os.path.basename(outname)} already exists in output dir;\n\tskipping {os.path.basename(rate_file)}.")
            continue

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
            'pathloss':{'skip':pathloss_skip},
            'nsclean':{'skip':nsclean_skip},
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
    if save_plots:
        breads.jwst_tools.plotting.plot_2d_image_set(cal_files,
                                                     output_dir = output_dir,
                                                     suptitle="Stage 2 pipeline reduction results",
                                                     plot_label = 'stage2')

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

def run_coordinate_recenter(cal_files, utils_dir, crds_dir, init_centroid=(0, 0), wv_sampling=None, N_wvs_nodes=40,
                            mask_charge_transfer_radius=None,
                            IWA=0.3, OWA=1.0,
                            debug_init=None, debug_end=None,
                            numthreads=16,
                            save_plots=False,
                            filename_suffix="_webbpsf",
                            overwrite=False,
                            targetname=None):
    """


    Parameters
    ----------
    cal_files : list of strings
        Filenames of stage 2 reduced Cal files to process
    utils_dir
    crds_dir
    init_centroid : tuple of floats
        Starting guess for centroid location
    wv_sampling
        Wavelength sampling
    N_wvs_nodes
        Number of wavelength nodes
    mask_charge_transfer_radius
    IWA : float
        Inner working angle
    OWA : float
        Outer working angle
    debug_init
    debug_end
    numthreads
    save_plots
    filename_suffix
    overwrite
    targetname : str

    Returns
    -------

    Creates output files:
      [X]_fitpsf_[filename_suffix].fits
      [X]_poly2d_centroid_[filename_suffix].txt

    """
    mypool = mp.Pool(processes=numthreads)

    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)

    # Science data: List of stage 2 cal.fits files
    for filename in cal_files:
        print(filename)
    print("N files: {0}".format(len(cal_files)))

    # Define the filename of the output file saved by fitpsf
    splitbasename = os.path.basename(cal_files[0]).split("_")
    fitpsf_filename = os.path.join(utils_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[
        3] + "_fitpsf" + filename_suffix + ".fits")
    poly2d_centroid_filename = os.path.join(utils_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[
        3] + "_poly2d_centroid" + filename_suffix + ".txt")

    hdulist_sc = fits.open(cal_files[0])
    detector = hdulist_sc[0].header["DETECTOR"].strip().lower()
    if wv_sampling is None:
        wv_sampling = np.arange(np.nanmin(hdulist_sc["WAVELENGTH"].data),
                                np.nanmax(hdulist_sc["WAVELENGTH"].data),
                                np.nanmedian(hdulist_sc["WAVELENGTH"].data) / 300)
    hdulist_sc.close()

    # If the output centroid file already exists, from a prior run of this function, then
    # just reload those results and return them, without doing any additional calculation,
    # unless overwrite is set.
    if not overwrite:
        if len(glob(poly2d_centroid_filename)) == 1:
            print("Found centroid results from prior calculation. Loading and returning those.")
            output = np.loadtxt(poly2d_centroid_filename, delimiter=' ')
            poly_p_ra, poly_p_dec = output[0], output[1]
            print("RA correction " + detector, poly_p_ra)
            print("Dec correction " + detector, poly_p_dec)
            return poly_p_ra, poly_p_dec

    regwvs_dataobj_list = []
    for filename in cal_files[0::]:
        print(filename)
        if detector not in filename:
            raise Exception("The files in cal_files should all be for the same detector")

        # Define a series of processing tasks to be performed on each input file.
        preproc_task_list = []
        preproc_task_list.append(["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}, True, True])
        preproc_task_list.append(["compute_coordinates_arrays",{'targname':targetname}])
        preproc_task_list.append(["convert_MJy_per_sr_to_MJy"])
        preproc_task_list.append(["compute_starspectrum_contnorm", {"N_nodes": N_wvs_nodes,
                                                                    "threshold_badpix": 100,
                                                                    "mppool": mypool}, True, True])
        preproc_task_list.append(["compute_starsubtraction", {"starsub_dir": "starsub1d_tmp",
                                                              "threshold_badpix": 10,
                                                              "mppool": mypool}, True, True])
        preproc_task_list.append(["compute_interpdata_regwvs", {"wv_sampling": wv_sampling}, True, True])

        # Load that file. (TODO: Does this invoke the preproc_task_list?)
        dataobj = JWSTNirspec_cal(filename, crds_dir=crds_dir, utils_dir=utils_dir,
                                  save_utils=True, load_utils=True, preproc_task_list=preproc_task_list)
        regwvs_dataobj_list.append(dataobj.reload_interpdata_regwvs())

    # Combine all input files into a joint dataset
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
    wpsfs, wpsfs_header, wepsfs, webbpsf_wvs, webbpsf_x, webbpsf_y, wpsf_oversample, wpsf_pixelscale = webbpsf_reload
    webbpsf_x = np.tile(webbpsf_x[None, :, :], (wepsfs.shape[0], 1, 1))
    webbpsf_y = np.tile(webbpsf_y[None, :, :], (wepsfs.shape[0], 1, 1))

    # Fit a model PSF (WebbPSF) to the combined point cloud of dataobj_list
    # Save output as fitpsf_filename
    ann_width = None
    padding = 0.0
    sector_area = None
    where_center_disk = regwvs_combdataobj.where_point_source((0.0, 0.0), IWA)
    regwvs_combdataobj.bad_pixels[where_center_disk] = np.nan

    fitpsf(regwvs_combdataobj, wepsfs, webbpsf_x, webbpsf_y, out_filename=fitpsf_filename, IWA=0.0, OWA=OWA,
           mppool=mypool, init_centroid=init_centroid, ann_width=ann_width, padding=padding,
           sector_area=sector_area, RDI_folder_suffix=filename_suffix, rotate_psf=regwvs_combdataobj.east2V2_deg,
           flipx=True, psf_spaxel_area=(wpsf_pixelscale) ** 2, debug_init=debug_init, debug_end=debug_end)

    with fits.open(fitpsf_filename) as hdulist:
        bestfit_coords = hdulist[0].data

    x2fit = wv_sampling - np.nanmedian(wv_sampling)
    y2fit = bestfit_coords[0, :, 2]
    _wv_min = wv_sampling[0] + 0.1 * (wv_sampling[-1] - wv_sampling[0])
    _wv_max = wv_sampling[-1] - 0.1 * (wv_sampling[-1] - wv_sampling[0])
    print(_wv_min, _wv_max)
    wherefinite = np.where(np.isfinite(y2fit) * (wv_sampling > _wv_min) * (wv_sampling < _wv_max))
    poly_p_ra = np.polyfit(x2fit[wherefinite], y2fit[wherefinite], deg=2)
    print("RA correction " + detector, poly_p_ra)

    x2fit = wv_sampling - np.nanmedian(wv_sampling)
    y2fit = bestfit_coords[0, :, 3]
    wherefinite = np.where(np.isfinite(y2fit) * (wv_sampling > _wv_min) * (wv_sampling < _wv_max))
    poly_p_dec = np.polyfit(x2fit[wherefinite], y2fit[wherefinite], deg=2)
    print("Dec correction " + detector, poly_p_dec)

    # Save centroids to a text file
    np.savetxt(poly2d_centroid_filename, [poly_p_ra, poly_p_dec], delimiter=' ')

    if save_plots:
        color_list = ["#ff9900", "#006699", "#6600ff", "#006699", "#ff9900", "#6600ff"]
        print(bestfit_coords.shape)
        fontsize = 12
        plt.figure(figsize=(12, 10))
        plt.subplot(3, 1, 1)

        plt.plot(wv_sampling, bestfit_coords[0, :, 0] * 1e9, linestyle="-", color=color_list[0], label="Fixed centroid",
                 linewidth=1)
        plt.plot(wv_sampling, bestfit_coords[0, :, 1] * 1e9, linestyle="--", color=color_list[2], label="Free centroid",
                 linewidth=1)
        plt.xlim([wv_sampling[0], wv_sampling[-1]])
        plt.xlabel("Wavelength ($\\mu$m)", fontsize=fontsize)
        plt.ylabel("Flux density (mJy)", fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right")

        plt.subplot(3, 1, 2)
        plt.plot(wv_sampling, bestfit_coords[0, :, 2], label="bestfit centroid")
        poly_model = np.polyval(poly_p_ra, wv_sampling - np.nanmedian(wv_sampling))
        plt.plot(wv_sampling, poly_model, label="polyfit")
        plt.plot(wv_sampling, bestfit_coords[0, :, 2] - poly_model, label="residuals")
        plt.xlabel("Wavelength ($\\mu$m)", fontsize=fontsize)
        plt.ylabel("$\\Delta$RA (as)", fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right")

        plt.subplot(3, 1, 3)
        plt.plot(wv_sampling, bestfit_coords[0, :, 3], label="bestfit centroid")
        poly_model = np.polyval(poly_p_dec, wv_sampling - np.nanmedian(wv_sampling))
        plt.plot(wv_sampling, poly_model, label="polyfit")
        plt.plot(wv_sampling, bestfit_coords[0, :, 3] - poly_model, label="residuals")
        plt.xlabel("Wavelength ($\\mu$m)", fontsize=fontsize)
        plt.ylabel("$\\Delta$Dec (as)", fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        plt.tight_layout()

        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y%m%d_%H%M%S")

        out_filename = os.path.join(utils_dir, formatted_datetime + "_centroid_calibration.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)

    return poly_p_ra, poly_p_dec


###########################################################################
#  Functions for noise cleaning


def fm_column_background(nonlin_paras, cubeobj, nodes=20,
                         fix_parameters=None,
                         return_where_finite=False,
                         regularization=None,
                         badpixfraction=0.75,
                         M_spline=None,
                         spline_reg_std=1.0):
    """ Forward model column background, for use in forward_model_noise_clean

    Parameters
    ----------
    nonlin_paras
    cubeobj
    nodes
    fix_parameters
    return_where_finite
    regularization
    badpixfraction
    M_spline
    spline_reg_std

    Returns
    -------

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


def fm_charge_transfer(nonlin_paras, cubeobj, charge_transfer_mask=None, nodes=40, fix_parameters=None,
                       return_where_finite=False,
                       regularization=None, badpixfraction=0.75, M_spline=None, spline_reg_std=1.0):
    """ Forward model charge transfer ("bleeding") within the detector, particularly for bright/saturated sources

    Parameters
    ----------
    nonlin_paras
    cubeobj
    charge_transfer_mask
    nodes
    fix_parameters
    return_where_finite
    regularization
    badpixfraction
    M_spline
    spline_reg_std

    Returns
    -------

    """
    if fix_parameters is not None:
        _nonlin_paras = np.array(fix_parameters)
        _nonlin_paras[np.where(np.array(fix_parameters) is None)] = nonlin_paras
    else:
        _nonlin_paras = nonlin_paras

    if M_spline is None:
        if type(nodes) is int:
            n_nodes = nodes
            x_knots = np.linspace(np.nanmin(cubeobj.wavelengths), np.nanmax(cubeobj.wavelengths), n_nodes,
                                  endpoint=True).tolist()
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

    where_finite = np.where(np.isfinite(cubeobj.data) * np.isfinite(cubeobj.bad_pixels) * (cubeobj.noise != 0))
    d = cubeobj.data[where_finite]
    s = cubeobj.noise[where_finite]

    if np.size(where_finite[0]) <= (1 - badpixfraction) * np.size(cubeobj.data):
        # don't bother to do a fit if there are too many bad pixels
        return np.array([]), np.array([]).reshape(0, n_linpara), np.array([])
    else:
        where_finite_wvs = np.where(np.isfinite(cubeobj.wavelengths))
        if M_spline is None:
            m_tmp = get_spline_model(x_knots, cubeobj.wavelengths[where_finite_wvs], spline_degree=3)
        else:
            m_tmp = copy(M_spline)
        m_entire_image = np.zeros((cubeobj.data.shape[0], cubeobj.data.shape[1], m_tmp.shape[1]))
        m_entire_image[where_finite_wvs[0], where_finite_wvs[1], :] = m_tmp
        m_entire_image = m_entire_image * charge_transfer_mask[:, :, None]
        m_entire_image[np.where(np.isnan(m_entire_image))] = 0

        kernel_scale = _nonlin_paras[0]
        x = np.arange(-cubeobj.data.shape[0], cubeobj.data.shape[0] + 1)
        charge_transfer_kernel = 1 / (1 + x ** 2 / kernel_scale ** 2)  # Lorentzian Function

        # Convolve each column with the kernel
        m_entire_image_convolved = convolve1d(m_entire_image, weights=charge_transfer_kernel, axis=0, mode='constant')
        m_output = m_entire_image_convolved[where_finite[0], where_finite[1], :]

        extra_outputs = {}
        if regularization == "default":
            s_reg = np.zeros(n_nodes) + spline_reg_std
            d_reg = np.zeros(n_nodes)
            extra_outputs["regularization"] = (d_reg, s_reg)
        elif regularization == "user":
            raise Exception("user defined regularisation not yet implemented")
            extra_outputs["regularization"] = (d_reg, s_reg)

        if return_where_finite:
            extra_outputs["where_finite"] = where_finite

        if len(extra_outputs) >= 1:
            return d, m_output, s, extra_outputs
        else:
            return d, m_output, s


def forward_model_noise_clean(rate_file, cal_file_dir, clean_dir, crds_dir, N_nodes=40, model_charge_transfer=False,
                              utils_dir=None, coords_offset=(0, 0)):
    """ Clean 1/f stripe noise from NIRSpec IFU data. Inspired by NSClean but implemented independently.

    The way it works:
        subtraction done on rate.fits
        Use the cal.fits to retrieve the mask of the IFU slices
        Fit detector columns one at time. I just fit a smooth continuum (using my splines) to the masked detector
        column, and also masking the region around the star more aggressively I believe
        subtract the fitted continuum
        Save new rate.fits

    Parameters
    ----------
    rate_file
    cal_file_dir
    clean_dir
    crds_dir
    N_nodes
    model_charge_transfer
    utils_dir
    coords_offset

    Returns
    -------

    """

    basename = os.path.basename(rate_file)
    cal_filename = os.path.join(cal_file_dir, basename.replace("_rate.fits", "_cal.fits"))
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

    # Extent the slices masks to the edge of the detector
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
                "Could not find the corresponding RDI model webbpsf cal file. "
                "Please run run_coordinate_recenter(...) first.")
        with fits.open(cal_model_filename) as hdul_model:
            webbpsf_im = hdul_model["SCI"].data
        saturated_mask = np.full(cal_dataobj.data.shape, np.nan)
        saturated_mask[np.where(untangle_dq(dq, verbose=False)[1, :, :])] = 1
        saturated_mask[np.where((sep_im > 0.5))] = np.nan
        # Define the saturation threshold in Mjy/sr below. This is definitely not ideal, probably not accurate.
        # Will probably need to fix later.
        saturation_threshold = 1e5  # Mjy/sr

        charge_transfer_mask = (webbpsf_im - saturation_threshold) * saturated_mask
        charge_transfer_mask[np.where(np.isnan(charge_transfer_mask))] = 0.0
        charge_transfer_mask = np.clip(charge_transfer_mask, 0, np.inf)

        # Define the spline nodes for fitting the background in each detector column
        n_nodes_charge_transfer = 5  # number of nodes in the column
        x_knots_charge_transfer = np.linspace(np.nanmin(cal_dataobj.wavelengths), np.nanmax(cal_dataobj.wavelengths),
                                              n_nodes_charge_transfer,
                                              endpoint=True).tolist()
        where_finite_wvs = np.where(np.isfinite(cal_dataobj.wavelengths))
        m_spline_charge_transfer = get_spline_model(x_knots_charge_transfer, cal_dataobj.wavelengths[where_finite_wvs],
                                                    spline_degree=3)

        fix_parameters = [10]  # width of the lorentzian
        fm_paras = {"charge_transfer_mask": charge_transfer_mask, "fix_parameters": fix_parameters,
                    "regularization": None, "badpixfraction": 0.75, "M_spline": m_spline_charge_transfer,
                    "spline_reg_std": 1.0}
        nonlin_paras = []
        out_log_prob, _, rchi2, linparas, linparas_err = fitfm(nonlin_paras, data, fm_charge_transfer, fm_paras,
                                                               computeH0=False, scale_noise=False)
        d_masked, m, s, extra_outputs = fm_charge_transfer(nonlin_paras, data, return_where_finite=True, **fm_paras)
        where_finite = extra_outputs["where_finite"]
        d_masked_canvas = np.zeros(data.data.shape) + np.nan
        d_masked_canvas[where_finite] = d_masked
        data.bad_pixels = np.ones(data.data.shape)
        d, m, s, _ = fm_charge_transfer(nonlin_paras, data, return_where_finite=True, **fm_paras)
        model_canvas = np.dot(m, linparas)
        model_canvas = np.reshape(model_canvas, data.data.shape)

        ################################
        # remove lorentzian model
        im -= model_canvas
        ################################

    x = np.arange(2048)
    x_knots_column = np.linspace(0, 2048, N_nodes, endpoint=True).tolist()
    m_spline_column = get_spline_model(x_knots_column, x, spline_degree=3)

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
                    "regularization": "default", "M_spline": m_spline_column}
        if 1:  # optimize non linear parameter
            out_log_prob, _, rchi2, linparas, linparas_err = fitfm(nonlin_paras, data, fm_column_background, fm_paras,
                                                                   computeH0=False, scale_noise=False)
            if not np.isfinite(out_log_prob):
                continue
            d_masked, m, s, extra_outputs = fm_column_background(nonlin_paras, data, return_where_finite=True,
                                                                 **fm_paras)
            where_finite = extra_outputs["where_trace_finite"]
            data.bad_pixels = np.ones(data.data.shape)
            d, m, s, _ = fm_column_background(nonlin_paras, data, return_where_finite=True, **fm_paras)
            # print(data.data.shape,d.shape,np.size(where_finite[0]),np.size(d_masked))
            d_masked_canvas = np.zeros(d.shape) + np.nan
            d_masked_canvas[where_finite] = d_masked

            m = np.dot(m, linparas)
            mad = median_abs_deviation(((d_masked_canvas - m))[np.where(np.isfinite(d_masked_canvas))])

            data.bad_pixels = bad_pixels[:, colid]
            data.bad_pixels[np.where(np.abs(d_masked_canvas - m) > 5 * mad)] = np.nan

        if 1:  # optimize non linear parameter
            out_log_prob, _, rchi2, linparas, linparas_err = fitfm(nonlin_paras, data, fm_column_background, fm_paras,
                                                                   computeH0=False, scale_noise=False)
            if not np.isfinite(out_log_prob):
                continue
            d_masked, m, s, extra_outputs = fm_column_background(nonlin_paras, data, return_where_finite=True,
                                                                 **fm_paras)
            where_finite = extra_outputs["where_trace_finite"]
            data.bad_pixels = np.ones(data.data.shape)
            d, m, s, _ = fm_column_background(nonlin_paras, data, return_where_finite=True, **fm_paras)
            # print(data.data.shape,d.shape,np.size(where_finite[0]),np.size(d_masked))
            d_masked_canvas = np.zeros(d.shape) + np.nan
            d_masked_canvas[where_finite] = d_masked

            m = np.dot(m, linparas)

        new_im[:, colid] = im[:, colid] - m

    priheader['comment'] = 'Detector correlated noise removed by custom code'
    hdul[0].header = priheader
    hdul["SCI"].data = new_im
    new_rate_file = os.path.join(clean_dir, os.path.basename(rate_file))
    hdul.writeto(new_rate_file, overwrite=True)
    hdul.close()
    return new_rate_file


def run_noise_clean(rate_files, stage2_dir, output_dir, crds_dir, N_nodes=40, model_charge_transfer=False,
                    utils_dir=None, coords_offset=(0, 0), overwrite=False, save_plots=True):
    """Invoke forward model noise removal for a list of rate files

    Parameters
    ----------
    rate_files
    stage2_dir
    output_dir
    crds_dir
    N_nodes
    model_charge_transfer
    utils_dir
    coords_offset
    overwrite

    Returns
    -------

    """
    # We need to check that the desired output directories exist, and if not create them
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Start a timer to keep track of runtime
    time0 = time.perf_counter()

    cleaned_rate_files = []

    for fid, rate_file in enumerate(rate_files):

        print(f"Noise Clean: Processing file {fid + 1} of {len(rate_files)}: {os.path.basename(rate_file)}")
        outname = os.path.join(output_dir, os.path.basename(rate_file))
        cleaned_rate_files.append(outname)
        if os.path.exists(outname) and not overwrite:
            print(f"\tOutput file {os.path.basename(outname)} already exists in the cleaned output directory; skipping {os.path.basename(rate_file)}.")
            continue


        forward_model_noise_clean(rate_file, stage2_dir, output_dir, crds_dir,
                                  N_nodes=N_nodes,
                                  model_charge_transfer=model_charge_transfer, utils_dir=utils_dir,
                                  coords_offset=coords_offset)
        # Print out the time benchmark
        time1 = time.perf_counter()
        print(f"\tNoise Clean Runtime so far: {time1 - time0:0.4f} seconds\n")
    time1 = time.perf_counter()
    print(f"Noise Clean Total Runtime: {time1 - time0:0.4f} seconds")

    if save_plots:
        breads.jwst_tools.plotting.plot_2d_image_sets_side_by_side(rate_files, cleaned_rate_files,
                                                                   output_dir=output_dir,
                                                                   suptitle="Noise Cleaining results. Left = Before, Right = After.",
                                                                   plot_label='noiseclean')

    return cleaned_rate_files

###########################################################################
# Host Star PSF Subtraction 


def compute_normalized_stellar_spectrum(cal_files, utils_dir, crds_dir, coords_offset=(0, 0), wv_nodes=None,
                                        mask_charge_transfer_radius=None, mppool=None,
                                        ra_dec_point_sources=None, overwrite=False,targetname=None):
    """

    Parameters
    ----------
    cal_files
    utils_dir
    crds_dir
    coords_offset
    wv_nodes
    mask_charge_transfer_radius
    mppool
    ra_dec_point_sources
    overwrite

    Returns
    -------

    """
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
        preproc_task_list.append(["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}, True, True])
        preproc_task_list.append(["compute_coordinates_arrays",{'targname':targetname}])
        preproc_task_list.append(["convert_MJy_per_sr_to_MJy"])
        preproc_task_list.append(["apply_coords_offset", {"coords_offset": coords_offset}])
        preproc_task_list.append(["compute_starspectrum_contnorm", {"x_nodes": wv_nodes,
                                                                    "threshold_badpix": 100,
                                                                    "mppool": mppool}, True, True])
        preproc_task_list.append(["compute_starsubtraction", {"starsub_dir": "starsub1d",
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
            for ra_pl, dec_pl in ra_dec_point_sources:
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


def compute_starlight_subtraction(cal_files, utils_dir, crds_dir, wv_nodes=None, combined_star_func=None,
                                  coords_offset=(0, 0), mppool=None,targetname=None):
    """

    Parameters
    ----------
    cal_files
    utils_dir
    crds_dir
    wv_nodes
    combined_star_func
    coords_offset
    mppool

    Returns
    -------

    """
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
        preproc_task_list.append(["compute_coordinates_arrays",{'targname':targetname}])
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


###########################################################################
# Regular Wavelength Grids


def get_combined_regwvs(dataobj_list, wv_sampling=None, mask_charge_transfer_radius=None, use_starsub=False,recompute=False,starsub_dir='starsub1d'):
    """

    Parameters
    ----------
    dataobj_list
    wv_sampling
    mask_charge_transfer_radius
    use_starsub1d

    Returns
    -------

    """
    regwvs_dataobj_list = []
    for dataobj in dataobj_list:

        if use_starsub:
            starsub_filename = os.path.join(dataobj.utils_dir, starsub_dir, os.path.basename(dataobj.filename))
            starsub_dataobj = JWSTNirspec_cal(starsub_filename, crds_dir=dataobj.crds_dir, utils_dir=dataobj.utils_dir)
            if (dataobj.data_unit == 'MJy') and (starsub_dataobj.data_unit == 'MJy/sr'):
                replace_data = dataobj.convert_MJy_per_sr_to_MJy(data_in_MJy_per_sr=starsub_dataobj.data)
            elif (dataobj.data_unit == 'MJy/sr') and (starsub_dataobj.data_unit == 'MJy/sr'):
                replace_data = starsub_dataobj.data
            elif (dataobj.data_unit =='MJy') and (starsub_dataobj.data_unit == 'MJy'):
                replace_data = starsub_dataobj.data
            elif (dataobj.data_unit =='MJy/sr') and (starsub_dataobj.data_unit == 'MJy'):
                print('Exception: data obj in MJy/sr and starsub in MJy')
                raise Exception('conversion from MJy to MJy/sr not implemented yet.')
            regwvs_filename = dataobj.default_filenames["compute_interpdata_regwvs"].replace("_regwvs.fits",
                                                                                             "_starsub1d_regwvs.fits")
        else:
            replace_data = None
            regwvs_filename = dataobj.default_filenames["compute_interpdata_regwvs"]

        if not recompute:
            regwvs_dataobj = dataobj.reload_interpdata_regwvs(load_filename=regwvs_filename)
        else:
            print('RECOMPUTING GET_combined_REGWVS...')
            regwvs_dataobj = None
        if regwvs_dataobj is None:
            regwvs_dataobj = dataobj.compute_interpdata_regwvs(save_utils=regwvs_filename, wv_sampling=wv_sampling,
                                                               replace_data=replace_data)
        regwvs_dataobj_list.append(regwvs_dataobj)

    regwvs_combdataobj = JWSTNirspec_multiple_cals(regwvs_dataobj_list)
    if mask_charge_transfer_radius is not None:
        regwvs_combdataobj.compute_charge_bleeding_mask(threshold2mask=mask_charge_transfer_radius)

    return regwvs_combdataobj


def save_combined_regwvs(regwvs_combdataobj, out_filename):
    """

    Parameters
    ----------
    regwvs_combdataobj
    out_filename

    Returns
    -------

    """
    hdulist = fits.HDUList()
    hdulist.append(fits.ImageHDU(data=regwvs_combdataobj.data, name='DATA'))
    hdulist.append(fits.ImageHDU(data=regwvs_combdataobj.noise, name='ERR'))
    hdulist.append(fits.ImageHDU(data=regwvs_combdataobj.dra_as_array, name='RA'))
    hdulist.append(fits.ImageHDU(data=regwvs_combdataobj.ddec_as_array, name='DEC'))
    hdulist.append(fits.ImageHDU(data=regwvs_combdataobj.wavelengths, name='WAVE'))
    hdulist.append(fits.ImageHDU(data=regwvs_combdataobj.wv_sampling, name='WV_SAMPLING'))
    hdulist.append(fits.ImageHDU(data=regwvs_combdataobj.bad_pixels, name='BADPIX'))
    hdulist.writeto(out_filename, overwrite=True)
    hdulist.close()


def get_2D_point_cloud_interpolator(regwvs_combdataobj, wv0):
    """

    Parameters
    ----------
    regwvs_combdataobj
    wv0

    Returns
    -------

    """
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
        wv_sampling = regwvs_combdataobj.wv_sampling

    wv0_index = np.argmin(np.abs(wv_sampling - wv0))

    where_good = np.where(np.isfinite(bad_pixels[:, wv0_index]))
    x = dra_as_array[where_good[0], wv0_index]
    y = ddec_as_array[where_good[0], wv0_index]
    z = data[where_good[0], wv0_index]
    filtered_triangles = filter_big_triangles(x, y, 0.2)
    # Create filtered triangulation
    filtered_tri = tri.Triangulation(x, y, triangles=filtered_triangles)
    # Perform LinearTriInterpolator for filtered triangulation
    pointcloud_interp = tri.LinearTriInterpolator(filtered_tri, z)

    return pointcloud_interp


############################################################################
#  Function to invoke all reduction steps in one go


def run_complete_stage1_2_clean_reduction(input_dir, output_root_dir=None, overwrite=False):
    """Overarching top-level function to invoke stage1, stage2, and 1/f noise cleaning code

    This will run the complete reduction from uncal files to cal files. It will take a while.

    If files already exist, repeat reductions are skipped, unless overwrite is set True

    Parameters
    ----------
    input_dir
    output_root_dir
    overwrite

    Returns
    -------

    """

    if output_root_dir is None:
        output_root_dir = input_dir

    # Set up subdirectory paths
    det1_dir = os.path.join(output_root_dir, "stage1")  # Detector1 pipeline outputs will go here
    spec2_dir = os.path.join(output_root_dir, "stage2")  # Initial spec2 pipeline outputs will go here
    clean_det1_dir = os.path.join(output_root_dir,
                                  "stage1_clean")  # noise-cleaned Detector1 pipeline outputs will go here
    clean_spec2_dir = os.path.join(output_root_dir, "stage2_clean")  # noise-cleaned Spec2 pipeline outputs will go here

    # Find input rate files
    uncal_files = find_files_to_process(input_dir, 'uncal.fits')

    # Run all reduction steps
    rate_files = run_stage1(uncal_files, output_dir=det1_dir, overwrite=overwrite)
    cleaned_rate_files = run_noise_clean(rate_files, spec2_dir, clean_det1_dir, overwrite=overwrite)
    cleaned_cal_files = run_stage2(cleaned_rate_files, output_dir=clean_spec2_dir, overwrite=overwrite)

    return cleaned_cal_files

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


def run_stage1_miri(uncal_path, list_bands=None, overwrite=False, maximum_cores="1", skip_dark=False,
                    oddeven_correction=True):
    """Run pipeline stage 1, with some customizations for reductions"""

    crds_path = os.getenv('CUSTOM_CRDS_PATH')

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

            if oddeven_correction:
                new_name = os.path.basename(file).replace('uncal.fits', 'odd_even_corr_rate.fits')
                out_name_corr = os.path.join(output_dir, new_name)
                if os.path.exists(out_name_corr) and not overwrite:
                    print(f"Output file {out_name} already exists in output dir;\n\tskipping {file}.")
                else:
                    # Open the recently created stage 1 file
                    with fits.open(out_name) as hdul:
                        # Copy every hdu (Header/Data Units)
                        new_hdul = fits.HDUList([hdu.copy() for hdu in hdul])

                        # Modify the extension 'SCI' with odd/even flux correction
                        sci_hdu = new_hdul['SCI']
                        flux_corr = oddEvenCorrectionImage(hdul, crds_path)
                        sci_hdu.data = flux_corr

                        # Sauvegarder dans un nouveau fichier
                        new_hdul.writeto(out_name_corr, overwrite=overwrite)

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
            rate_files = find_files_to_process(os.path.join(uncal_path, target_name, band, 'stage1'), filetype='rate.fits')
        else:
            rate_files = find_files_to_process(os.path.join(uncal_path, target_name, band, 'stage1_sub_bkg'), filetype='rate.fits')
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
                flat_path_rate = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/miri_flat')
                print("checking", flat_path, os.path.dirname(__file__))
            else:
                flat_path_rate = flat_path

            if detector == 'MIRIFUSHORT':
                if band == 'SHORT':
                    flat_path_rate = os.path.join(flat_path_rate, '12A')
                elif band == 'MEDIUM':
                    flat_path_rate = os.path.join(flat_path_rate, '12B')
                elif band == 'LONG':
                    flat_path_rate = os.path.join(flat_path_rate, '12C')
                else:
                    raise ValueError(f'Unsupported band for file: {rate_file} must be either SHORT, MEDIUM or LONG')
                channel = 'CH1'

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

            if os.path.exists(flat_path_rate) == 0:
                raise ValueError(f"'flat_path' {flat_path_rate} does not exist")

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
    run_stage1_miri(uncal_path, list_bands=list_bands, overwrite=overwrite, maximum_cores="1", skip_dark=False,
                    oddeven_correction=False)
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

    brightest_col = colonne_median_max_channel(data, channel=channel)
    print("Brightest column:", brightest_col)
    if save_png:
        plt.title("Best fringes flat pattern selection")
        plt.xlabel("Row index")
        plt.ylabel("Fringes transmission")
        plt.xlim([450, 500])
        plt.ylim([0.5, 1.5])
        file_name = hdr['FILENAME']
        continuum = gaussian_filter(data[:, brightest_col], sigma=8)
        fringe_data = data[:, brightest_col] / continuum
        plt.plot(fringe_data, label='data')

    for filename in filenames:
        if filename.endswith(".fits"):
            flat = fits.open(os.path.join(flat_dir, filename))
            hdr_flat = flat[0].header

            if flat_extended:
                flat = flat['FLAT_EXTENDED'].data
            else:
                flat = flat['FLAT'].data

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

    if save_png:
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"../fig_fringes_{file_name}.png")
        plt.close()

    idx = np.nanargmin(std)
    std_min = np.nanmin(std)
    flat_name = file[idx]
    print("Flat selected:", flat_name)
    hdu_best_flat = fits.open(os.path.join(flat_dir, flat_name))
    if flat_extended:
        best_flat = hdu_best_flat['FLAT_EXTENDED'].data
    else:
        best_flat = hdu_best_flat['FLAT'].data
    if full_output:
        return best_flat, flat_name, std_min, file, std
    else:
        return best_flat, flat_name, std_min

def colonne_median_max(mat):
    medianes = np.nanmedian(mat, axis=0)  # Calculer la médiane de chaque colonne
    col_index = np.nanargmax(medianes)  # Trouver l'indice de la médiane max
    return col_index


def colonne_median_max_channel(data, channel='CH1'):
    if channel == 'CH1' or channel == 'CH3':
        brightest_col = colonne_median_max(data[:, :500])
    elif channel == 'CH2' or channel == 'CH4':
        brightest_col = colonne_median_max(data[:, 500:]) + 500
    else:
        raise ValueError('Channel must be CH1, CH2, CH3 or CH4')

    return brightest_col

## Breads function functions for miri

def run_coordinate_recenter_miri(cal_files, utils_dir, crds_dir, init_centroid=(0, 0), wv_sampling=None, N_wvs_nodes=40,
                                 mask_charge_transfer_radius=None,
                                 IWA=0.3, OWA=1.0,
                                 debug_init=None, debug_end=None,
                                 numthreads=16,
                                 save_plots=False,
                                 filename_suffix="_webbpsf",
                                 overwrite=False):
    mypool = mp.Pool(processes=numthreads)

    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)

    # Science data: List of stage 2 cal.fits files
    for filename in cal_files:
        print(filename)
    print("N files: {0}".format(len(cal_files)))

    # Definie the filename of the output file saved by fitpsf
    splitbasename = os.path.basename(cal_files[0]).split("_")
    fitpsf_filename = os.path.join(utils_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[
        3] + "_fitpsf" + filename_suffix + ".fits")
    poly2d_centroid_filename = os.path.join(utils_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[
        3] + "_poly2d_centroid" + filename_suffix + ".txt")

    hdulist_sc = fits.open(cal_files[0])
    detector = hdulist_sc[0].header["DETECTOR"].strip().lower()
    band = hdulist_sc[0].header["BAND"]

    if wv_sampling is None:
        if band == 'SHORT':
            wmin = 4.90
            wmax = 5.74
        elif band == 'MEDIUM':
            wmin = 5.66
            wmax = 6.63
        elif band == 'LONG':
            wmin = 6.53
            wmax = 7.65
        else:
            print(f"BAND {band} not supported")
        wv_sampling = np.arange(wmin, wmax, 0.5 * (wmin + wmax) / 300)
    hdulist_sc.close()

    regwvs_dataobj_list = []
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

        dataobj = JWSTMiri_cal(filename, crds_dir=crds_dir, utils_dir=utils_dir,
                               save_utils=True, load_utils=True, preproc_task_list=preproc_task_list)
        regwvs_dataobj_list.append(dataobj.reload_interpdata_regwvs())

    regwvs_combdataobj = JWSTMiri_multiple_cals(regwvs_dataobj_list)
    if mask_charge_transfer_radius is not None:
        regwvs_combdataobj.compute_charge_bleeding_mask(threshold2mask=mask_charge_transfer_radius)
    print(f"[DEBUG] wv_sampling after JWSTMiri_multiple_cal {regwvs_combdataobj.wv_sampling}")
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
    where_center_disk = regwvs_combdataobj.where_point_source((0.0, 0.0), IWA)
    regwvs_combdataobj.bad_pixels[where_center_disk] = np.nan

    fitpsf_miri(regwvs_combdataobj, wepsfs, webbpsf_X, webbpsf_Y, out_filename=fitpsf_filename, IWA=0.0, OWA=OWA,
                mppool=mypool, init_centroid=init_centroid, ann_width=ann_width, padding=padding,
                sector_area=sector_area, RDI_folder_suffix=filename_suffix, rotate_psf=regwvs_combdataobj.east2V2_deg,
                flipx=True, psf_spaxel_area=(wpsf_pixelscale) ** 2, debug_init=debug_init, debug_end=debug_end)

    with fits.open(fitpsf_filename) as hdulist:
        print(f"[DEBUG] path to fitpsf {fitpsf_filename}")
        bestfit_coords = hdulist[0].data
        wpsf_angle_offset = hdulist[0].header["INIT_ANG"]
        wpsf_ra_offset = hdulist[0].header["INIT_RA"]
        wpsf_dec_offset = hdulist[0].header["INIT_DEC"]

    x2fit = regwvs_combdataobj.wv_sampling - np.nanmedian(regwvs_combdataobj.wv_sampling)
    y2fit = bestfit_coords[0, :, 2]

    _wv_min = regwvs_combdataobj.wv_sampling[0] + 0.1 * (
                regwvs_combdataobj.wv_sampling[-1] - regwvs_combdataobj.wv_sampling[0])
    _wv_max = regwvs_combdataobj.wv_sampling[-1] - 0.1 * (
                regwvs_combdataobj.wv_sampling[-1] - regwvs_combdataobj.wv_sampling[0])
    print(_wv_min, _wv_max)

    wherefinite = np.where(
        np.isfinite(y2fit) * (regwvs_combdataobj.wv_sampling > _wv_min) * (regwvs_combdataobj.wv_sampling < _wv_max))
    poly_p_RA = np.polyfit(x2fit[wherefinite], y2fit[wherefinite], deg=2)
    print("[DEBUG] RA correction " + detector, poly_p_RA)

    y2fit = bestfit_coords[0, :, 3]
    wherefinite = np.where(
        np.isfinite(y2fit) * (regwvs_combdataobj.wv_sampling > _wv_min) * (regwvs_combdataobj.wv_sampling < _wv_max))
    poly_p_dec = np.polyfit(x2fit[wherefinite], y2fit[wherefinite], deg=2)
    print("Dec correction " + detector, poly_p_dec)

    np.savetxt(poly2d_centroid_filename, [poly_p_RA, poly_p_dec], delimiter=' ')

    if save_plots:
        color_list = ["#ff9900", "#006699", "#6600ff", "#006699", "#ff9900", "#6600ff"]
        print(bestfit_coords.shape)
        fontsize = 12
        plt.figure(figsize=(12, 10))
        plt.subplot(3, 1, 1)

        plt.plot(regwvs_combdataobj.wv_sampling, bestfit_coords[0, :, 0] * 1e9, linestyle="-", color=color_list[0],
                 label="Fixed centroid", linewidth=1)
        plt.plot(regwvs_combdataobj.wv_sampling, bestfit_coords[0, :, 1] * 1e9, linestyle="--", color=color_list[2],
                 label="Free centroid", linewidth=1)
        plt.xlim([regwvs_combdataobj.wv_sampling[0], regwvs_combdataobj.wv_sampling[-1]])
        plt.xlabel("Wavelength ($\\mu$m)", fontsize=fontsize)
        plt.ylabel("Flux density (mJy)", fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right")

        plt.subplot(3, 1, 2)
        plt.plot(regwvs_combdataobj.wv_sampling, bestfit_coords[0, :, 2], label="bestfit centroid")
        poly_model = np.polyval(poly_p_RA,
                                regwvs_combdataobj.wv_sampling - np.nanmedian(regwvs_combdataobj.wv_sampling))
        plt.plot(regwvs_combdataobj.wv_sampling, poly_model, label="polyfit")
        plt.plot(regwvs_combdataobj.wv_sampling, bestfit_coords[0, :, 2] - poly_model, label="residuals")
        plt.xlabel("Wavelength ($\\mu$m)", fontsize=fontsize)
        plt.ylabel("$\\Delta$RA (as)", fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right")

        plt.subplot(3, 1, 3)
        plt.plot(regwvs_combdataobj.wv_sampling, bestfit_coords[0, :, 3], label="bestfit centroid")
        poly_model = np.polyval(poly_p_dec,
                                regwvs_combdataobj.wv_sampling - np.nanmedian(regwvs_combdataobj.wv_sampling))
        plt.plot(regwvs_combdataobj.wv_sampling, poly_model, label="polyfit")
        plt.plot(regwvs_combdataobj.wv_sampling, bestfit_coords[0, :, 3] - poly_model, label="residuals")
        plt.xlabel("Wavelength ($\\mu$m)", fontsize=fontsize)
        plt.ylabel("$\\Delta$Dec (as)", fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        plt.tight_layout()

        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y%m%d_%H%M%S")

        out_filename = os.path.join(utils_dir, formatted_datetime + "_centroid_calibration.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)

    return regwvs_combdataobj.wv_sampling, poly_p_RA, poly_p_dec

def compute_normalized_stellar_spectrum_miri(cal_files, channel, utils_dir, crds_dir, wave_2d, coords_offset=(0, 0),
                                             wv_nodes=None, target_name=None,
                                             mask_charge_transfer_radius=None, mppool=None,
                                             ra_dec_point_sources=None, overwrite=False):
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)

    hdulist_sc = fits.open(cal_files[0])
    detector = hdulist_sc[0].header["DETECTOR"].strip().lower()
    if wv_nodes is None:
        wv_nodes = np.linspace(np.nanmin(wave_2d),
                               np.nanmax(wave_2d),
                               40, endpoint=True)
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
            preproc_task_list.append(["compute_coordinates_arrays", {"target_name": target_name}])
        else:
            preproc_task_list.append(["compute_coordinates_arrays"])
        preproc_task_list.append(["convert_MJy_per_sr_to_MJy"])
        preproc_task_list.append(["apply_coords_offset", {"coords_offset": coords_offset}])
        preproc_task_list.append(["compute_starspectrum_contnorm", {"x_nodes": wv_nodes,
                                                                    "threshold_badpix": 100,
                                                                    "mppool": mppool, "star_hf_subtraction":True}, True, True])
        preproc_task_list.append(["compute_starsubtraction", {"starsub_dir": "starsub1d",
                                                              "threshold_badpix": 10,
                                                              "mppool": mppool}, True, True])

        dataobj = JWSTMiri_cal(filename, channel_reduction=channel, utils_dir=utils_dir,
                               save_utils=True, load_utils=True, preproc_task_list=preproc_task_list)

        if mask_charge_transfer_radius is not None:
            dataobj.compute_charge_bleeding_mask(threshold2mask=mask_charge_transfer_radius)
        # mask planets before computing the star spectrum
        if ra_dec_point_sources is not None:
            for ra_pl, dec_pl in ra_dec_point_sources:
                where_pl = where_point_source(dataobj, [ra_pl / 1000., dec_pl / 1000.], 0.16)
                dataobj.bad_pixels[where_pl] = np.nan

        dataobj_list.append(dataobj)

    new_wavelengths, combined_fluxes, combined_errors = get_contnorm_spec_miri(dataobj_list, spline2d=False,
                                                                               load_utils=False,
                                                                               out_filename=combined_contnorm_spec_filename,
                                                                               spec_R_sampling=2700 * 4,
                                                                               interpolation="linear")

    combined_star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False, fill_value=1)
    return combined_star_func

def compute_starlight_subtraction_miri(cal_files, channel, utils_dir, crds_dir, wave_2d, wv_nodes=None,
                                       combined_star_func=None, coords_offset=(0, 0), mppool=None):
    hdulist_sc = fits.open(cal_files[0])
    detector = hdulist_sc[0].header["DETECTOR"].strip().lower()
    if wv_nodes is None:
        wv_nodes = np.linspace(np.nanmin(wave_2d),
                               np.nanmax(wave_2d),
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

        dataobj = JWSTMiri_cal(filename, channel_reduction=channel, utils_dir=utils_dir,
                               save_utils=True, load_utils=True, preproc_task_list=preproc_task_list)
        if combined_star_func is not None:
            dataobj.reload_starspectrum_contnorm()
            dataobj.star_func = combined_star_func

        outputs = dataobj.reload_starsubtraction()

        if outputs is None:
            outputs = dataobj.compute_starsubtraction(save_utils=True, starsub_dir="starsub1d",
                                                      threshold_badpix=10, mppool=mppool)
        subtracted_im, star_model, spline_paras0, _wv_nodes = outputs

        dataobj_list.append(dataobj)

    return dataobj_list

def get_combined_regwvs_miri(dataobj_list, channel, wv_sampling=None, mask_charge_transfer_radius=None,
                             use_starsub1d=False, reload=False):
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
    if mask_charge_transfer_radius is not None:
        regwvs_combdataobj.compute_charge_bleeding_mask(threshold2mask=mask_charge_transfer_radius)

    return regwvs_combdataobj
