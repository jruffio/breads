from copy import copy
import numpy as np
from scipy import sparse
import itertools
import os
import warnings

import astropy.io.fits as fits
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from scipy.optimize import lsq_linear

import multiprocessing as mp
import ctypes
from multiprocessing.sharedctypes import RawArray
import progressbar

from breads.utils import get_spline_model
from breads.fit import fitfm
from breads.jwst_tools.spline1d import Spline1D

def _task_normrows(paras):
    """ Worker function for normalize_rows(), for use in parallelized computations. Perform the task on a single row.

    Parameters
    ----------
    paras : tuple
        im_rows, im_wvs_rows, noise_rows, badpix_rows, wv_nodes, stellar_features, threshold,regularization,reg_mean_map,reg_std_map

    Returns
    -------
    new_im_rows, new_noise_rows, new_badpix_rows, res,paras_out

    """
    im_rows, im_wvs_rows, noise_rows, badpix_rows, wv_nodes, stellar_features, threshold,regularization,reg_mean_map,reg_std_map = paras
    spline = Spline1D.coerce(wv_nodes)

    new_im_rows = np.array(copy(im_rows), '<f4')  # .byteswap().newbyteorder()
    new_noise_rows = copy(noise_rows)
    new_badpix_rows = copy(badpix_rows)
    res = np.full(im_rows.shape, np.nan)
    paras_out = np.full((im_rows.shape[0],spline.n_params), np.nan)
    for k in range(im_rows.shape[0]):

        finite_mask = (
            np.isfinite(im_rows[k, :])
            & np.isfinite(badpix_rows[k, :])
            & np.isfinite(noise_rows[k, :])
            & np.isfinite(stellar_features[k, :])
        )

        valid_mask = (
            finite_mask
            & (noise_rows[k, :] != 0)
        )

        where_data_finite = np.where(valid_mask)

        if np.size(where_data_finite[0]) == 0:
            res[k, :] = np.nan
            continue

        M_spline = spline.get_spline_model(im_wvs_rows[k, :])
        # Remove the spline modes that should not be fitted (depends on the basis, see Spline1D.prune_modes):
        # cardinal, catmull_rom: redundant edge modes, evaluated on the full row (historical behavior).
        # bspline, bspline_nak: modes truncated by the sampling, evaluated on the pixels actually being fitted.
        if spline.is_bspline:
            spline.prune_modes(M_spline, M_eval=M_spline[where_data_finite[0], :])
        else:
            spline.prune_modes(M_spline)

        d = im_rows[k, where_data_finite[0]]
        d_err = noise_rows[k, where_data_finite[0]]

        M = M_spline[where_data_finite[0], :] * stellar_features[k, where_data_finite[0], None]

        # if regularization:
        #     validpara = np.where(np.nansum(M > np.nanmax(M) * 0.00001, axis=0) != 0)
        # else:
        #     validpara = np.where(np.nansum(M > np.nanmax(M) * 0.01, axis=0) != 0)
        validpara = np.where(np.nansum(M > 0, axis=0) != 0)
        M = M[:, validpara[0]]

        if len(validpara[0]) == 0:
            res[k, :] = np.nan
            continue

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
            d4fit, M4fit, s4fit = d, M, d_err

        bounds_min = [-np.inf, ] * M.shape[1]
        bounds_max = [np.inf, ] * M.shape[1]

        p = lsq_linear(M4fit / s4fit[:, None], d4fit / s4fit, bounds=(bounds_min, bounds_max)).x
        paras_out[k,validpara[0]] = p
        m = np.dot(M, p)
        res[k, where_data_finite[0]] = d - m
        new_im_rows[k, where_data_finite[0]] = m
        new_noise_rows[k, where_data_finite[0]] = d_err
        norm_res_row = np.full(im_rows.shape[1], np.nan)
        norm_res_row[where_data_finite] = (d - m) / d_err

        meddev = median_abs_deviation(norm_res_row[where_data_finite])
        where_bad = np.where((np.abs(norm_res_row) / meddev > threshold) | np.isnan(norm_res_row))
        new_badpix_rows[k, where_bad[0]] = np.nan

    return new_im_rows, new_noise_rows, new_badpix_rows, res, paras_out

def normalize_rows(image, im_wvs, noise=None, badpixs=None, stellar_features=None, N_nodes=40, mppool=None, threshold=10,
                   wv_nodes=None, regularization=True, reg_mean_map=None, reg_std_map=None):
    """
    Fit a spline model to each row of the detector image.
    Stellar spectral features can be included through the "stellar_features" parameter.

    Parameters
    ----------
    image : 2d array
        2D array of the image to be normalized (e.g. the star spectrum in each row)
    im_wvs : 2d array
        2D array of the wavelengths corresponding to each pixel in the image (same shape as image)
    noise : 2d array or None (optional)
        2D array of the noise corresponding to each pixel in the image (same shape as image). If None, all pixels are assumed to have noise of 1.
    badpixs : 2d array or None (optional)
        2D array of the bad pixel mask corresponding to each pixel in the image (same shape as image). If None, all pixels are assumed to be good (badpixs=1).
    stellar_features : 2d array or None
        This is to optionally include the stellar lines in the spline models, if not None, the stellar_features is multiplied to the spline model.
    N_nodes : int or None (optional, default is 40)
        If wv_nodes is None, Number of nodes to use for fitting splines for the continuum star spectrum estimation.
    mppool : multiprocessing.Pool or None (optional)
        If None, the computation is done without parallelization.
    threshold : float (optional)
        Threshold for flagging bad pixels based on the normalized residuals of the fit.
    wv_nodes : 1d array, Spline1D, or None (optional)
        If wv_nodes is specified, this wavelength spacing (in micron) will be used to do the splines fitting.
        A plain array means a cardinal spline basis. Use a breads.jwst_tools.spline1d.Spline1D to choose the basis
        (e.g. Spline1D(nodes, basis="bspline")).
        If None, N_nodes will set an evenly nodes spacing (cardinal basis).
    regularization : bool (optional)
        If True, the spline fitting is regularized by providing Gaussian priors on each linear spline parameter
        (cardinal and catmull_rom: value of the continuum at each node; B-spline bases: control points).
    reg_mean_map : 2d array or None (optional)
        If regularization is True, this 2d array (Nrows, n_params) provides the mean of the prior,
        with n_params = Spline1D.n_params (= number of nodes for the cardinal and catmull_rom bases).
    reg_std_map : 2d array or None (optional)
        If regularization is True, this 2d array (Nrows, n_params) provides the standard deviation of the prior.

    Returns
    -------
    new_image : 2d array
        Best fit model (e.g. spline fit of the continuum)
    new_noise :
        Same as input noise
    new_badpixs :
        Bad pixel map with additional bad pixels identified.
    new_res :
        Residuals; original data minus best fit model.
    new_spline_paras :
        Best fit parameters for the spline fits for each row.
    """
    if noise is None:
        noise = np.ones(image.shape)
    if badpixs is None:
        badpixs = np.ones(image.shape)
    if stellar_features is None:
        stellar_features = np.ones(image.shape)

    if wv_nodes is None:
        wv_nodes = np.linspace(np.nanmin(im_wvs), np.nanmax(im_wvs), N_nodes, endpoint=True)
    wv_nodes = Spline1D.coerce(wv_nodes)

    new_image = copy(image)
    new_noise = copy(noise)
    new_badpixs = copy(badpixs)
    new_res = np.full(image.shape, np.nan)
    new_spline_paras = np.zeros((image.shape[0], wv_nodes.n_params))

    #if chunk is too small, don't parallelize
    parallel_flag = True
    if mppool is not None:
        numthreads = mppool._processes
        chunk_size = image.shape[0] // (3 * numthreads)
        if chunk_size == 0:
            parallel_flag = False


    if (mppool is None) or (parallel_flag==False):
        paras = new_image, im_wvs, new_noise, new_badpixs, wv_nodes, stellar_features, threshold, regularization, reg_mean_map, reg_std_map
        outputs = _task_normrows(paras)
        new_image, new_noise, new_badpixs, new_res,new_spline_paras = outputs
    else:
        numthreads = mppool._processes
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
            _stellar_features = stellar_features[(k * chunk_size):((k + 1) * chunk_size), :]
            # regularization=None,reg_mean_map=None,reg_std_map=None
            if regularization:
                reg_mn_chunk= reg_mean_map[(k * chunk_size):((k + 1) * chunk_size), :]
                reg_std_chunk = reg_std_map[(k * chunk_size):((k + 1) * chunk_size), :]

            image_list.append(_new_image)
            wvs_list.append(_im_wvs)
            noise_list.append(_new_noise)
            badpixs_list.append(_new_badpixs)
            starmodel_list.append(_stellar_features)
            if regularization:
                reg_mean_map_list.append(reg_mn_chunk)
                reg_std_map_list.append(reg_std_chunk)

        _row_valid_pix = row_ids[((N_chunks - 1) * chunk_size):image.shape[0]]
        row_indices_list.append(_row_valid_pix)

        _new_image = new_image[((N_chunks - 1) * chunk_size):image.shape[0], :]
        _im_wvs = im_wvs[((N_chunks - 1) * chunk_size):image.shape[0], :]
        _new_noise = new_noise[((N_chunks - 1) * chunk_size):image.shape[0], :]
        _new_badpixs = new_badpixs[((N_chunks - 1) * chunk_size):image.shape[0], :]
        _stellar_features = stellar_features[((N_chunks - 1) * chunk_size):image.shape[0], :]
        if regularization:
            reg_mn_chunk = reg_mean_map[((N_chunks - 1) * chunk_size):image.shape[0], :]
            reg_std_chunk = reg_std_map[((N_chunks - 1) * chunk_size):image.shape[0], :]

        image_list.append(_new_image)
        wvs_list.append(_im_wvs)
        noise_list.append(_new_noise)
        badpixs_list.append(_new_badpixs)
        starmodel_list.append(_stellar_features)
        if regularization:
            reg_mean_map_list.append(reg_mn_chunk)
            reg_std_map_list.append(reg_std_chunk)

        if not regularization:
            outputs_list = mppool.map(_task_normrows, zip(image_list, wvs_list, noise_list, badpixs_list,
                                                          itertools.repeat(wv_nodes),
                                                          starmodel_list,
                                                          itertools.repeat(threshold),
                                                          itertools.repeat(False),
                                                          itertools.repeat(None),
                                                          itertools.repeat(None)))
        else:
            arguments = list(zip(image_list, wvs_list, noise_list, badpixs_list,
                                                          itertools.repeat(wv_nodes),
                                                          starmodel_list,
                                                          itertools.repeat(threshold),
                                                          itertools.repeat(regularization),
                                                          reg_mean_map_list,reg_std_map_list))

            outputs_list = mppool.map(_task_normrows, arguments)

        for row_indices, outputs in zip(row_indices_list, outputs_list):
            out_im_rows, out_noise_rows, out_badpixs_rows, out_res,spline_paras = outputs
            new_image[row_indices, :] = out_im_rows
            new_noise[row_indices, :] = out_noise_rows
            new_badpixs[row_indices, :] = out_badpixs_rows
            new_res[row_indices, :] = out_res
            new_spline_paras[row_indices, :] = spline_paras

    return new_image, new_noise, new_badpixs, new_res,new_spline_paras

def _tmp_fm(nonlin_paras, data_obj: "Instrument"):
    _d,M,_e,d_reg,s_reg = nonlin_paras
    if d_reg is not None and s_reg is not None:
        extra_outputs = {}
        extra_outputs["regularization"] = (d_reg, s_reg)
        return _d, M, _e, extra_outputs
    else :
        return _d, M, _e

def _model_error_paras(spline3d, stamp, linparas_cov):
    """
    Compute spline parameters that describe the 1 sigma error of the fitted model of one stamp, so that the model
    error can be evaluated at any sample like the model itself (design matrix times parameters, see
    _task_evaluate_3dspline). Used for the B-spline bases.

    For the cardinal and catmull_rom bases, the parameter errors are already the model errors at the nodes. For the B-spline bases,
    the parameters are anti-correlated control points, and interpolating their errors overestimates the model error
    by >10x. Instead:
    1. compute the exact model error at the nodes and midpoints between nodes of the extended stamp (spline only,
       without the stellar features): sqrt(diag(M C M^T)), with M the design matrix at these points and C the
       covariance of the parameters;
    2. fit it with the spline basis of the stamp, minimizing the relative difference so that the large errors at the
       edges do not drive the fit. The result smooths any variation of the error between nodes.

    Parameters
    ----------
    spline3d : Spline3D
    stamp : tuple
        Stamp (see Spline3D.stamps).
    linparas_cov : 2d array (N_stamp_params x N_stamp_params)
        Covariance of the parameters of the stamp (fitfm with return_covariance=True). NaN for the parameters that
        were not fitted.

    Returns
    -------
    merr_paras : 1d array (N_stamp_params)
        Spline parameters of the model error. NaN if it could not be computed.
    """
    n_params = linparas_cov.shape[0]

    # Keep the parameters that were fitted
    valid = np.isfinite(np.diag(linparas_cov))
    if np.sum(valid) == 0:
        return np.full(n_params, np.nan)
    C = linparas_cov[np.ix_(valid, valid)]

    # 1. Grid of points: nodes and midpoints between nodes, along each axis of the extended stamp.
    # The points are moved slightly inside the stamp, because the design matrix excludes its boundaries.
    k0, k1, k2, k3, l0, l1, l2, l3, m0, m1 = stamp
    wv_nodes = spline3d.wv_nodes
    y_nodes = spline3d.y_nodes[l0:l3 + 1]
    x_nodes = spline3d.x_nodes[k0:k3 + 1]
    grids = []
    for nodes in [wv_nodes, y_nodes, x_nodes]:
        midpoints = 0.5 * (nodes[1:] + nodes[:-1])
        points = np.sort(np.concatenate([nodes, midpoints]))
        eps = 1e-6 * (nodes[-1] - nodes[0])
        points = np.clip(points, nodes[0] + eps, nodes[-1] - eps)
        grids.append(points)
    w_grid, y_grid, x_grid = np.meshgrid(grids[0], grids[1], grids[2], indexing="ij")
    w_pts = w_grid.ravel()
    y_pts = y_grid.ravel()
    x_pts = x_grid.ravel()

    # Exact model error at these points: err^2 = diag(M C M^T), computed row by row as sum_j (M C)_ij M_ij
    M_pts = spline3d.design_matrix(stamp, x_pts, y_pts, w_pts)
    M_valid = M_pts[:, valid]
    variance_pts = np.sum(np.dot(M_valid, C) * M_valid, axis=1)
    variance_pts = np.clip(variance_pts, 0, None)   # tiny negative values from rounding errors
    err_pts = np.sqrt(variance_pts)

    good = np.isfinite(err_pts) & (err_pts > 0)
    if np.sum(good) == 0:
        return np.full(n_params, np.nan)

    # 2. Fit the errors with the spline basis: find p such that M_pts @ p ~ err_pts, in relative terms,
    # i.e. (M_pts / err_pts) @ p ~ 1
    A = M_pts[good] / err_pts[good, None]
    b = np.ones(np.sum(good))
    merr_paras = np.linalg.lstsq(A, b, rcond=None)[0]
    return merr_paras


def _task_fit_3dspline(paras):
    """
    Fit the 3D spline in one stamp (see Spline3D.stamps). Reads the data from the shared arrays and writes the best
    fit model, the parameters and the bad pixels back into them.
    """

    from breads.instruments import Instrument   # runtime import here to avoid circular import problem

    stamp_ids, spline3d, threshold, reg_mean_map, reg_std_map, types_tuple = paras
    mp_float_type, mp_bp_type = types_tuple


    data_np = _arraytonumpy(shared_data, shared_data_shape, dtype=mp_float_type)
    noise_np = _arraytonumpy(shared_noise, shared_data_shape, dtype=mp_float_type)
    wvs_np = _arraytonumpy(shared_wvs, shared_data_shape, dtype=mp_float_type)
    scaled_x_np = _arraytonumpy(shared_scaled_x, shared_data_shape, dtype=mp_float_type)
    scaled_y_np = _arraytonumpy(shared_scaled_y, shared_data_shape, dtype=mp_float_type)
    bp_np = _arraytonumpy(shared_bp, shared_data_shape, dtype=mp_float_type)
    stellar_features_np = _arraytonumpy(shared_stellar_features, shared_data_shape, dtype=mp_float_type)
    bestfit_model_np = _arraytonumpy(shared_bestfit_model, shared_data_shape, dtype=mp_float_type)

    spline3d_paras_np = _arraytonumpy(shared_spline3d_paras, shared_spline3d_paras_shape, dtype=mp_float_type)
    spline3d_paras_err_np = _arraytonumpy(shared_spline3d_paras_err, shared_spline3d_paras_shape, dtype=mp_float_type)
    spline3d_merr_paras_np = _arraytonumpy(shared_spline3d_merr_paras, shared_spline3d_paras_shape, dtype=mp_float_type)

    k0,k1,k2,k3,l0,l1,l2,l3,m0,m1 = stamp_ids
    sw, sy, sx = spline3d.stamp_param_slices(stamp_ids)

    extended_bool_map = spline3d.stamp_extended_mask(stamp_ids, scaled_x_np, scaled_y_np, wvs_np)
    bool_map_from_bp = (np.isfinite(bp_np) * np.isfinite(data_np) * np.isfinite(noise_np) * (noise_np != 0)* \
                        np.isfinite(wvs_np)* np.isfinite(stellar_features_np) * (data_np/noise_np > -10))
    where_data_finite = np.where(extended_bool_map*bool_map_from_bp)

    N_pix_threshold = 3
    if np.size(where_data_finite[0]) < N_pix_threshold:
        return None

    _d = data_np[where_data_finite]
    _x = scaled_x_np[where_data_finite]
    _y = scaled_y_np[where_data_finite]
    _w = wvs_np[where_data_finite]
    _e = noise_np[where_data_finite]

    inner_pixels = np.where(spline3d.stamp_inner_mask(stamp_ids, _x, _y, _w))

    if spline3d.is_bspline:
        # Sparse design matrix for the B-spline bases (local basis functions), solved by fitfm through the normal
        # equations without ever building the dense matrix (much less memory and time for large stamps).
        M_3dspline = spline3d.design_matrix_sparse(stamp_ids, _x, _y, _w)
        M = sparse.diags(stellar_features_np[where_data_finite].astype(M_3dspline.dtype)) @ M_3dspline
        # Deactivate the modes that are barely sampled (see Spline3D.prune_modes)
        M = M @ sparse.diags((~spline3d.prune_modes(stamp_ids, M_3dspline, M)).astype(M.dtype))
        M.eliminate_zeros()
        M = sparse.csr_matrix(M)
    else:
        M_3dspline = spline3d.design_matrix(stamp_ids, _x, _y, _w)
        M = M_3dspline * stellar_features_np[where_data_finite][:, None]

        # Deactivate the modes that are barely sampled (see Spline3D.prune_modes)
        M[:, spline3d.prune_modes(stamp_ids, M_3dspline, M)] = 0

    if reg_mean_map is not None and reg_std_map is not None:
        d_reg, s_reg = np.ravel(reg_mean_map[sw, sy, sx]), np.ravel(reg_std_map[sw, sy, sx])
    else:
        d_reg, s_reg = None, None

    # The full covariance of the parameters is only needed for the model error of the B-spline bases (see below)
    _results = fitfm(nonlin_paras=[_d,M,_e,d_reg,s_reg],dataobj=Instrument(),fm_func=_tmp_fm,fm_paras={},
                    marginalize_noise_scaling=False, scale_noise=False, return_covariance=spline3d.is_bspline)
    bestfit_log_prob, rchi2, linparas, linparas_err = _results[0:4]
    if np.all(np.isnan(linparas)):
        return None
    stamp_params_shape = spline3d.stamp_params_shape(stamp_ids)
    paras_canvas = np.reshape(linparas, stamp_params_shape)
    paras_err_canvas = np.reshape(linparas_err, stamp_params_shape)
    _linparas = copy(linparas)
    _linparas[np.where(~np.isfinite(_linparas))] = 0
    m = M @ _linparas if sparse.issparse(M) else np.dot(M, _linparas)

    spline3d_paras_np[m0,m1,sw,sy,sx] = paras_canvas
    spline3d_paras_err_np[m0,m1,sw,sy,sx] = paras_err_canvas
    if spline3d.is_bspline:
        spline3d_merr_paras_np[m0,m1,sw,sy,sx] = np.reshape(_model_error_paras(spline3d, stamp_ids, _results[4]),
                                                            stamp_params_shape)
    else:
        # Cardinal and Catmull-Rom bases: the parameters are the values of the model at the nodes, so their errors are
        # the errors of the model at the nodes, and the model error is interpolated between the nodes with the same
        # spline.
        spline3d_merr_paras_np[m0,m1,sw,sy,sx] = paras_err_canvas
    combined_idx = tuple(w[inner_pixels] for w in where_data_finite)
    bestfit_model_np[combined_idx] = m[inner_pixels]

    # Bad pixel identification from the normalized residuals
    norm_res = (_d[inner_pixels] - m[inner_pixels]) / _e[inner_pixels]
    meddev = median_abs_deviation(norm_res)
    where_bad = np.where((np.abs(norm_res) / meddev > threshold) | np.isnan(norm_res))

    combined_inner_pixels = tuple(w[where_bad] for w in inner_pixels)
    combined_idx = tuple(w[combined_inner_pixels] for w in where_data_finite)
    bp_np[combined_idx] = np.nan

    return


def radial_prior_3dspline(spline3d, x, y, wavelengths, data, noise, bad_pixels, annulus_width=0.1, radial_nsigma=5,
                          prior_nsigma=10, faint_fraction=0.1, end_plane_frac=0.1, min_points_annulus=10,
                          wv_window_frac=2e-3):
    """
    Compute a Gaussian prior on the parameters of a 3D spline from the radial profile of the data around the star
    (at each wavelength), to pass to fit_3dspline (reg_mean_map, reg_std_map).

    The prior keeps the parameters that the data do not constrain (edge of the field of view, gaps, masked regions)
    close to the azimuthal average of the data, without affecting the well sampled ones: it is wide (radial_nsigma
    times the azimuthal scatter).

    For each wavelength parameter (spline3d.param_positions):
    - the good pixels within +/- wv_window_frac x wavelength of it are binned in concentric annuli of width
      annulus_width around the star (the origin of the coordinates); for the first/last wavelength parameters, which
      are usually just outside the wavelength range of the data, each annulus uses its pixels within 2 x wv_window_frac
      x wavelength of its own shortest/longest wavelength with data (the nearest wavelength with data at that
      separation: the wavelength coverage depends on the position in the field);
    - in each annulus with at least min_points_annulus pixels: mean of the data, and radial_nsigma times their standard
      deviation;
    - the profile tapers to a read noise dominated prior, mean 0 and width prior_nsigma x noise floor, one annulus
      beyond the last annulus with data (and this prior is used beyond). The width is never smaller than this fallback
      width. The noise floor is the median noise of the faint_fraction faintest good pixels;
    - the prior of each parameter is the profile linearly interpolated at the separation of its (x, y) position;
    - first/last wavelength parameters: the width is increased (in quadrature) by end_plane_frac x |mean| to account
      for the change of the flux between the data and the parameter wavelength.

    The prior is in data units, i.e. for a 3D spline fitted without stellar spectrum (continuum normalization). For the
    B-spline bases, the parameters are control points, not values; the profile is smooth and wide, so a prior on the
    control points at their positions is a good approximation of the prior on the model.

    Parameters
    ----------
    spline3d : breads.jwst_tools.spline3d.Spline3D
        3D spline basis.
    x, y : arrays
        Coordinates of the pixels, in the coordinates of the nodes (IFU coordinates scaled to the reference wavelength,
        star at the origin), same shape as data.
    wavelengths, data, noise : arrays
        Wavelength (micron), data and noise of the pixels.
    bad_pixels : array
        NaN for bad pixels, 1 for good ones.
    annulus_width : float
        Width of the annuli (arcsec), e.g. the spaxel size (0.1 for NIRSpec).
    radial_nsigma : float
        Width of the prior in units of the standard deviation of the data in the annulus.
    prior_nsigma : float
        Width of the fallback prior in units of the noise floor.
    faint_fraction : float
        Fraction of the faintest good pixels used to compute the noise floor.
    end_plane_frac : float
        First/last wavelength parameters: fraction of |mean| added in quadrature to the width.
    min_points_annulus : int
        Annuli with fewer pixels are ignored.
    wv_window_frac : float
        Half width of the wavelength window around each wavelength parameter, relative to its wavelength.

    Returns
    -------
    reg_mean_map, reg_std_map : arrays (n_wv_params, n_y_params, n_x_params)
        Mean and width of the prior on the parameters (spline3d.param_positions).
    info : dict
        "noise_floor", "fallback_std", "wv_centers" (nominal center of the wavelength window of each wavelength
        parameter; for the first/last ones, the window actually depends on the annulus, see above) and
        "profiles" (for each wavelength parameter, (radius, mean, width) of the profile, including the taper point).
    """
    x, y, wavelengths = np.asarray(x), np.asarray(y), np.asarray(wavelengths)
    data, noise, bad_pixels = np.asarray(data), np.asarray(noise), np.asarray(bad_pixels)
    good = (np.isfinite(bad_pixels) & np.isfinite(data) & np.isfinite(noise) & (noise > 0) & np.isfinite(x) &
            np.isfinite(y) & np.isfinite(wavelengths))
    if not np.any(good):
        raise ValueError("No good pixels to compute the radial prior.")
    data_good = data[good]
    faint = data_good < np.nanpercentile(data_good, 100 * faint_fraction)
    noise_floor = float(np.nanmedian(noise[good][faint]))
    fallback_std = prior_nsigma * noise_floor

    wv_pos, y_pos, x_pos = spline3d.param_positions
    rr = np.sqrt(x_pos[None, :] ** 2 + y_pos[:, None] ** 2)  # (n_y_params, n_x_params)
    wv_min, wv_max = np.nanmin(wavelengths[good]), np.nanmax(wavelengths[good])

    reg_mean_map = np.zeros((np.size(wv_pos), np.size(y_pos), np.size(x_pos)))
    reg_std_map = np.full((np.size(wv_pos), np.size(y_pos), np.size(x_pos)), fallback_std)
    profiles, wv_centers = [], []
    # annulus of each good pixel, and shortest / longest wavelength of the good pixels of each annulus (the wavelength
    # coverage depends on the position in the field, e.g. the data of the core can start later than elsewhere)
    r_good = np.sqrt(x[good] ** 2 + y[good] ** 2)
    annulus_good = np.floor(r_good / annulus_width).astype(int)
    wvs_good = wavelengths[good]
    n_annuli = np.max(annulus_good) + 1
    wv_min_annulus = np.full(n_annuli, np.inf)
    wv_max_annulus = np.full(n_annuli, -np.inf)
    np.minimum.at(wv_min_annulus, annulus_good, wvs_good)
    np.maximum.at(wv_max_annulus, annulus_good, wvs_good)
    for iw, wv in enumerate(wv_pos):
        half_window = wv_window_frac * wv
        wv_center = np.clip(wv, wv_min + half_window, wv_max - half_window)
        wv_centers.append(wv_center)
        if wv < wv_min + half_window:
            # before the data: in each annulus, the window starts at the shortest wavelength with data of that annulus
            in_window = wvs_good < wv_min_annulus[annulus_good] + 2 * half_window
        elif wv > wv_max - half_window:
            # after the data: in each annulus, the window ends at the longest wavelength with data of that annulus
            in_window = wvs_good > wv_max_annulus[annulus_good] - 2 * half_window
        else:
            in_window = np.abs(wvs_good - wv) < half_window
        r = r_good[in_window]
        d = data_good[in_window]
        annulus_id = annulus_good[in_window]
        counts = np.bincount(annulus_id)
        sums = np.bincount(annulus_id, weights=d)
        sums2 = np.bincount(annulus_id, weights=d ** 2)
        populated = np.where(counts >= min_points_annulus)[0]
        if np.size(populated) == 0:
            profiles.append((np.array([0.0]), np.array([0.0]), np.array([fallback_std])))
            continue
        mean_c = sums[populated] / counts[populated]
        std_c = np.sqrt(np.clip(sums2[populated] / counts[populated] - mean_c ** 2, 0, None))
        r_c = (populated + 0.5) * annulus_width
        # taper to the fallback one annulus beyond the last annulus with data
        r_c = np.append(r_c, r_c[-1] + annulus_width)
        mean_c = np.append(mean_c, 0.0)
        std_c = np.maximum(np.append(radial_nsigma * std_c, fallback_std), fallback_std)
        profiles.append((r_c, mean_c, std_c))

        reg_mean_map[iw] = np.interp(rr, r_c, mean_c, right=0.0)
        reg_std_map[iw] = np.interp(rr, r_c, std_c, right=fallback_std)
        if iw == 0 or iw == np.size(wv_pos) - 1:
            reg_std_map[iw] = np.sqrt(reg_std_map[iw] ** 2 + (end_plane_frac * reg_mean_map[iw]) ** 2)

    info = {"noise_floor": noise_floor, "fallback_std": fallback_std, "wv_centers": np.array(wv_centers),
            "profiles": profiles}
    return reg_mean_map, reg_std_map, info


def fit_3dspline(dataobj, spline3d,
                 stellar_features=None,
                 threshold=10, reg_mean_map=None, reg_std_map=None,
                 max_cores = 1):
    """
    Fit a 3D spline (x, y, wavelength) to the point cloud of an IFU data object, stamp by stamp.

    Parameters
    ----------
    dataobj : JWST_IFUs
        Data object (data, noise, wavelengths, bad_pixels, IFU coordinates and breads_header["WV_REF"]).
    spline3d : breads.jwst_tools.spline3d.Spline3D
        3D spline basis: nodes, basis type, stamp size and number of overlapping nodes.
    stellar_features : 2d array or None
        Multiplicative factor applied to the spline model (e.g. the stellar spectrum). Default is ones.
    threshold : float
        Bad pixel threshold on the normalized residuals (in units of their median absolute deviation).
    reg_mean_map, reg_std_map : arrays (n_wv_params, n_y_params, n_x_params) or None
        Optional Gaussian priors on the spline parameters.
    max_cores : int
        Number of processes.

    Returns
    -------
    bestfit_model, noise, bad_pixels, residuals, spline3d_paras, spline3d_paras_err, spline3d_merr_paras
        spline3d_paras, spline3d_paras_err and spline3d_merr_paras have the shape spline3d.params_shape.
        spline3d_paras_err: marginal 1 sigma errors of the parameters (sqrt of the diagonal of their covariance).
        spline3d_merr_paras: spline parameters of the 1 sigma error of the model (see _model_error_paras); evaluated
        with the same basis as the model, it gives the model error at any sample (evaluate_3dspline). For the
        cardinal and catmull_rom bases, it is equal to spline3d_paras_err.
    """

    wv_ref = dataobj.breads_header["WV_REF"]

    ny, nx = dataobj.data.shape

    _ifux, _ifuy = dataobj.get_ifu_coords()

    mp_float_type = ctypes.c_float
    mp_bp_type = ctypes.c_uint8
    types_tuple = (mp_float_type, mp_bp_type)

    data_mp = RawArray(mp_float_type, nx * ny)
    data_shape = (ny, nx)
    data_np = _arraytonumpy(data_mp, data_shape, dtype=mp_float_type)
    data_np[:] = dataobj.data

    noise_mp = RawArray(mp_float_type, nx * ny)
    noise_np = _arraytonumpy(noise_mp, data_shape, dtype=mp_float_type)
    noise_np[:] = dataobj.noise

    wvs_mp = RawArray(mp_float_type, nx * ny)
    wvs_np = _arraytonumpy(wvs_mp, data_shape, dtype=mp_float_type)
    wvs_np[:] = dataobj.wavelengths

    scaled_x_mp = RawArray(mp_float_type, nx * ny)
    scaled_x_np = _arraytonumpy(scaled_x_mp, data_shape, dtype=mp_float_type)
    scaled_x_np[:] = _ifux * wv_ref / dataobj.wavelengths

    scaled_y_mp = RawArray(mp_float_type, nx * ny)
    scaled_y_np = _arraytonumpy(scaled_y_mp, data_shape, dtype=mp_float_type)
    scaled_y_np[:] = _ifuy * wv_ref / dataobj.wavelengths

    bp_mp = RawArray(mp_float_type, nx * ny)
    bp_np = _arraytonumpy(bp_mp, data_shape, dtype=mp_float_type)
    bp_np[:] = copy(dataobj.bad_pixels)

    stellar_features_mp = RawArray(mp_float_type, nx * ny)
    stellar_features_np = _arraytonumpy(stellar_features_mp, data_shape, dtype=mp_float_type)
    if stellar_features is None:
        # scalar fill: np.ones(...) would allocate a full-size float64 temporary (1.0 is exact in float32)
        stellar_features_np[:] = 1
    else:
        stellar_features_np[:] = stellar_features

    bestfit_model_mp = RawArray(mp_float_type, nx * ny)
    bestfit_model_np = _arraytonumpy(bestfit_model_mp, data_shape, dtype=mp_float_type)
    bestfit_model_np[:] = np.full(data_shape,np.nan, dtype=np.float32)

    spline3d_paras_shape = spline3d.params_shape
    spline3d_paras_mp = RawArray(mp_float_type, int(np.prod(spline3d_paras_shape)))
    spline3d_paras_np = _arraytonumpy(spline3d_paras_mp, spline3d_paras_shape, dtype=mp_float_type)
    spline3d_paras_np[:] = np.full(spline3d_paras_shape,np.nan, dtype=np.float32)

    spline3d_paras_err_mp = RawArray(mp_float_type, int(np.prod(spline3d_paras_shape)))
    spline3d_paras_err_np = _arraytonumpy(spline3d_paras_err_mp, spline3d_paras_shape, dtype=mp_float_type)
    spline3d_paras_err_np[:] = np.full(spline3d_paras_shape,np.nan, dtype=np.float32)

    spline3d_merr_paras_mp = RawArray(mp_float_type, int(np.prod(spline3d_paras_shape)))
    spline3d_merr_paras_np = _arraytonumpy(spline3d_merr_paras_mp, spline3d_paras_shape, dtype=mp_float_type)
    spline3d_merr_paras_np[:] = np.full(spline3d_paras_shape,np.nan, dtype=np.float32)

    # Small stamps fitted independently, because fitting the whole field at once would not be tractable.
    stamp_list = spline3d.stamps()

    _init_args = (
        data_mp, data_shape,
        noise_mp, wvs_mp, scaled_x_mp, scaled_y_mp, bp_mp,stellar_features_mp,bestfit_model_mp,
        spline3d_paras_mp, spline3d_paras_shape,spline3d_paras_err_mp, spline3d_merr_paras_mp
    )

    # The data, noise, stellar features etc. are in shared memory; only small objects are passed to each task.
    args_list = [(stamp_ids, spline3d, threshold, reg_mean_map, reg_std_map, types_tuple) for stamp_ids in stamp_list]

    if max_cores <= 1:
        _tpool_init_3dspline(*_init_args)

        print("\tPerforming serial fit_3dspline...")
        for args_tuple in args_list:
            _task_fit_3dspline(args_tuple)
    else:

        ctx = mp.get_context("spawn")  # avoids unsafe fork after OpenMP init
        tpool = ctx.Pool(
            processes=max_cores,
            initializer=_tpool_init_3dspline,
            initargs=_init_args,
            maxtasksperchild=50
        )

        try:
            tasks = [
                tpool.apply_async(_task_fit_3dspline, args=(args_tuple,))
                for args_tuple in args_list
            ]

            for t in progressbar.progressbar(tasks, prefix="Processing blocks ", min_poll_interval=1):
                t.wait()
        finally:
            tpool.close()
            tpool.join()

    residuals = dataobj.data - bestfit_model_np
    return bestfit_model_np, noise_np, bp_np, residuals, spline3d_paras_np,spline3d_paras_err_np,spline3d_merr_paras_np


def _task_evaluate_3dspline(paras):
    """
    Evaluate the 3D spline in one stamp (see Spline3D.stamps). Reads the coordinates and parameters from the shared
    arrays and writes the model and its error back into them.
    """
    stamp_ids, spline3d, types_tuple = paras
    mp_float_type, mp_bp_type = types_tuple

    noise_np = _arraytonumpy(shared_noise, shared_data_shape, dtype=mp_float_type)
    wvs_np = _arraytonumpy(shared_wvs, shared_data_shape, dtype=mp_float_type)
    scaled_x_np = _arraytonumpy(shared_scaled_x, shared_data_shape, dtype=mp_float_type)
    scaled_y_np = _arraytonumpy(shared_scaled_y, shared_data_shape, dtype=mp_float_type)
    bp_np = _arraytonumpy(shared_bp, shared_data_shape, dtype=mp_float_type)
    stellar_features_np = _arraytonumpy(shared_stellar_features, shared_data_shape, dtype=mp_float_type)
    bestfit_model_np = _arraytonumpy(shared_bestfit_model, shared_data_shape, dtype=mp_float_type)

    spline3d_paras_np = _arraytonumpy(shared_spline3d_paras, shared_spline3d_paras_shape, dtype=mp_float_type)
    spline3d_merr_paras_np = _arraytonumpy(shared_spline3d_merr_paras, shared_spline3d_paras_shape, dtype=mp_float_type)

    k0,k1,k2,k3,l0,l1,l2,l3,m0,m1 = stamp_ids
    sw, sy, sx = spline3d.stamp_param_slices(stamp_ids)

    extended_bool_map = spline3d.stamp_extended_mask(stamp_ids, scaled_x_np, scaled_y_np, wvs_np)
    bool_map_from_bp = (np.isfinite(bp_np) * np.isfinite(stellar_features_np))
    where_data_finite = np.where(extended_bool_map*bool_map_from_bp)

    N_pix_threshold = 3
    if np.size(where_data_finite[0]) < N_pix_threshold:
        return

    _x = scaled_x_np[where_data_finite]
    _y = scaled_y_np[where_data_finite]
    _w = wvs_np[where_data_finite]

    inner_pixels = np.where(spline3d.stamp_inner_mask(stamp_ids, _x, _y, _w))

    M_3dspline = spline3d.design_matrix(stamp_ids, _x, _y, _w)
    M = M_3dspline * stellar_features_np[where_data_finite][:, None]
    m = np.dot(M, np.ravel(spline3d_paras_np[m0,m1,sw,sy,sx]))
    # Model error: same spline evaluated on the parameters of the model error (see _model_error_paras; for the cardinal
    # and catmull_rom bases, the errors of the parameters, i.e. the model errors at the nodes, interpolated between
    # the nodes)
    merr = np.dot(M, np.ravel(spline3d_merr_paras_np[m0,m1,sw,sy,sx]))
    if spline3d.is_bspline:
        merr = np.clip(merr, 0, None)

    combined_idx = tuple(w[inner_pixels] for w in where_data_finite)
    bestfit_model_np[combined_idx] = m[inner_pixels]
    noise_np[combined_idx] = merr[inner_pixels]
    return


def evaluate_3dspline_grid(x_vec,y_vec,wv_sampling,spline3d_filename,max_cores = 1):
    """Evaluate a 3D spline file (see evaluate_3dspline) on a regular (wv, y, x) grid. Returns (model, error) cubes."""
    wv_grid,y_grid,x_grid = np.meshgrid(wv_sampling,y_vec,x_vec,indexing='ij' )
    return evaluate_3dspline(x_grid,y_grid,wv_grid,spline3d_filename,stellar_features=None,max_cores=max_cores)


def evaluate_3dspline_pointcloud(dataobj, spline3d_filename,stellar_features=None, max_cores=1):
    """Evaluate a 3D spline file (see evaluate_3dspline) at the IFU coordinates and wavelengths of a data object."""
    ifux,ifuy = dataobj.get_ifu_coords()
    return evaluate_3dspline(ifux,ifuy, dataobj.wavelengths, spline3d_filename,
                             stellar_features=stellar_features, max_cores=max_cores)


def evaluate_3dspline(ifux,ifuy,wvs,spline3d_filename,stellar_features=None,max_cores = 1):
    """
    Evaluate a 3D spline saved in a FITS file (3D spline products of JWST_IFUs, or BreadsPSF files) at the samples
    (ifux, ifuy, wvs). The basis, nodes, stamp layout and reference wavelength are read from the file.

    Parameters
    ----------
    ifux, ifuy, wvs : arrays (same shape)
        Coordinates (arcsec, not scaled) and wavelengths (micron) of the samples.
    spline3d_filename : str
        File with the SPLINE_PARAS0/SPLINE_PARAS0_ERR, wv_nodes/x_nodes/y_nodes HDUs and the BREADS header. The model
        error is computed from SPLINE_PARAS0_MERR if present (see fit_3dspline), otherwise from SPLINE_PARAS0_ERR
        (files made before SPLINE_PARAS0_MERR existed; only correct for the cardinal and catmull_rom bases).
    stellar_features : array or None
        Multiplicative factor applied to the model. Default is ones.
    max_cores : int
        Number of processes.

    Returns
    -------
    model, model_err : arrays with the shape of ifux. NaN outside of the spline nodes.
        Parameters that were not fitted (NaN) are set to 0.
    """
    from breads.jwst_tools.spline3d import Spline3D

    with fits.open(spline3d_filename) as hdulist:
        spline3d = Spline3D.from_hdulist(hdulist)
        spline3d_paras = hdulist["SPLINE_PARAS0"].data
        if "SPLINE_PARAS0_MERR" in hdulist:
            spline3d_merr_paras = hdulist["SPLINE_PARAS0_MERR"].data
        else:
            spline3d_merr_paras = hdulist["SPLINE_PARAS0_ERR"].data
            if spline3d.is_bspline:
                warnings.warn(f"No SPLINE_PARAS0_MERR in {spline3d_filename}: the model error is computed from "
                              f"SPLINE_PARAS0_ERR, which overestimates it for the {spline3d.basis} basis.")
        wv_ref = hdulist['BREADS'].header['WV_REF']

    mp_float_type = ctypes.c_float
    mp_bp_type = ctypes.c_uint8
    types_tuple = (mp_float_type, mp_bp_type)

    data_size = np.size(ifux)
    data_mp = None
    data_shape = ifux.shape

    noise_mp = RawArray(mp_float_type, data_size)
    noise_np = _arraytonumpy(noise_mp, data_shape, dtype=mp_float_type)
    noise_np[:] = np.full(data_shape,np.nan, dtype=np.float32)

    wvs_mp = RawArray(mp_float_type, data_size)
    wvs_np = _arraytonumpy(wvs_mp, data_shape, dtype=mp_float_type)
    wvs_np[:] = wvs

    scaled_x_mp = RawArray(mp_float_type, data_size)
    scaled_x_np = _arraytonumpy(scaled_x_mp, data_shape, dtype=mp_float_type)
    scaled_x_np[:] = ifux * wv_ref / wvs

    scaled_y_mp = RawArray(mp_float_type, data_size)
    scaled_y_np = _arraytonumpy(scaled_y_mp, data_shape, dtype=mp_float_type)
    scaled_y_np[:] = ifuy * wv_ref / wvs

    bp_mp = RawArray(mp_float_type, data_size)
    bp_np = _arraytonumpy(bp_mp, data_shape, dtype=mp_float_type)
    bp_np[:] = np.full(data_shape,1, dtype=np.float32)

    stellar_features_mp = RawArray(mp_float_type, data_size)
    stellar_features_np = _arraytonumpy(stellar_features_mp, data_shape, dtype=mp_float_type)
    if stellar_features is None:
        stellar_features_np[:] = 1  # scalar fill, see fit_3dspline
    else:
        stellar_features_np[:] = stellar_features

    bestfit_model_mp = RawArray(mp_float_type, data_size)
    bestfit_model_np = _arraytonumpy(bestfit_model_mp, data_shape, dtype=mp_float_type)
    bestfit_model_np[:] = np.full(data_shape,np.nan, dtype=np.float32)

    spline3d_paras_shape = spline3d.params_shape
    if tuple(np.shape(spline3d_paras)) != tuple(spline3d_paras_shape):
        raise ValueError(f"SPLINE_PARAS0 in {spline3d_filename} has shape {np.shape(spline3d_paras)}, "
                         f"but {spline3d} expects {spline3d_paras_shape}.")
    spline3d_paras_mp = RawArray(mp_float_type, int(np.prod(spline3d_paras_shape)))
    spline3d_paras_np = _arraytonumpy(spline3d_paras_mp, spline3d_paras_shape, dtype=mp_float_type)
    spline3d_paras_np[:] = copy(spline3d_paras)
    spline3d_paras_np[np.where(~np.isfinite(spline3d_paras_np))] = 0

    spline3d_merr_paras_mp = RawArray(mp_float_type, int(np.prod(spline3d_paras_shape)))
    spline3d_merr_paras_np = _arraytonumpy(spline3d_merr_paras_mp, spline3d_paras_shape, dtype=mp_float_type)
    spline3d_merr_paras_np[:] = copy(spline3d_merr_paras)
    spline3d_merr_paras_np[np.where(~np.isfinite(spline3d_merr_paras_np))] = 0

    # Same stamps as in the fit
    stamp_list = spline3d.stamps()

    _init_args = (
        data_mp, data_shape,
        noise_mp, wvs_mp, scaled_x_mp, scaled_y_mp, bp_mp,stellar_features_mp,bestfit_model_mp,
        spline3d_paras_mp, spline3d_paras_shape, None, spline3d_merr_paras_mp
    )
    args_list = [(stamp_ids, spline3d, types_tuple) for stamp_ids in stamp_list]

    if max_cores <= 1:
        _tpool_init_3dspline(*_init_args)

        print("\tPerforming serial evaluate_3dspline...")
        for args_tuple in args_list:
            _task_evaluate_3dspline(args_tuple)
    else:

        ctx = mp.get_context("spawn")  # avoids unsafe fork after OpenMP init
        tpool = ctx.Pool(
            processes=max_cores,
            initializer=_tpool_init_3dspline,
            initargs=_init_args,
            maxtasksperchild=50
        )

        try:
            tasks = [
                tpool.apply_async(_task_evaluate_3dspline, args=(args_tuple,))
                for args_tuple in args_list
            ]

            for t in progressbar.progressbar(tasks, prefix="Processing blocks ", min_poll_interval=1):
                t.wait()
        finally:
            tpool.close()
            tpool.join()

    return bestfit_model_np, noise_np


def save_3dspline_model_gif(spline3d_filename, gif_filename=None, N_frames=10, max_cores=1):
    """
    Save a gif of the 3D spline model saved in spline3d_filename (see evaluate_3dspline), evaluated on a regular grid
    of the scaled coordinates (x * WV_REF / wavelength, the coordinates of the spline nodes) with one frame per
    wavelength, and shown as log10(|model|).

    The model is plotted rather than the spline parameters because the parameters of the B-spline basis are not
    values of the model (control points), and neighbouring stamps store their parameters in separate copies.

    Parameters
    ----------
    spline3d_filename : str
        File with the 3D spline (as read by evaluate_3dspline).
    gif_filename : str or None
        Output gif. Default: spline3d_filename with a .gif extension.
    N_frames : int
        Number of wavelengths, evenly spaced strictly inside the range of the wavelength nodes.
    max_cores : int
        Number of processes for evaluate_3dspline.
    """
    from breads.jwst_tools.spline3d import Spline3D
    from breads.jwst_tools.plotting import save_cube_as_gif

    if gif_filename is None:
        gif_filename = spline3d_filename.replace(".fits", ".gif")
    with fits.open(spline3d_filename) as hdulist:
        spline3d = Spline3D.from_hdulist(hdulist)
        wv_ref = hdulist['BREADS'].header['WV_REF']

    # Regular grid (pixel centers) covering the spline nodes, in scaled coordinates
    x_nodes, y_nodes, wv_nodes = spline3d.x_nodes, spline3d.y_nodes, spline3d.wv_nodes
    dx = np.median(np.diff(x_nodes))
    dy = np.median(np.diff(y_nodes))
    x_vec = np.arange(x_nodes[0] + dx / 2, x_nodes[-1], dx)
    y_vec = np.arange(y_nodes[0] + dy / 2, y_nodes[-1], dy)
    # The model is only defined strictly between the first and last wavelength nodes
    wv_frames = np.linspace(np.min(wv_nodes), np.max(wv_nodes), N_frames + 2)[1:-1]

    wv_grid, y_grid, x_grid = np.meshgrid(wv_frames, y_vec, x_vec, indexing='ij')
    # evaluate_3dspline takes unscaled coordinates and scales them by WV_REF / wavelength
    model, _ = evaluate_3dspline(x_grid * wv_grid / wv_ref, y_grid * wv_grid / wv_ref, wv_grid, spline3d_filename,
                                 max_cores=max_cores)

    log_model = np.log10(np.abs(model))
    valid = np.isfinite(log_model)
    if not np.any(valid):
        print("save_3dspline_model_gif: the model is not finite anywhere, no gif saved.")
        return
    # Color scale set by the peak within 0.5 arcsec of the star (the coordinates are relative to the star): poorly
    # constrained stamps at the edge of the field can be orders of magnitude brighter than the PSF core.
    near_star = valid & (np.sqrt(x_grid ** 2 + y_grid ** 2) < 0.5)
    vmax = np.max(log_model[near_star]) if np.any(near_star) else np.percentile(log_model[valid], 99.9)
    extent = [x_vec[0] - dx / 2, x_vec[-1] + dx / 2, y_vec[0] - dy / 2, y_vec[-1] + dy / 2]
    save_cube_as_gif(log_model, filename=gif_filename, fps=3, vmin=vmax - 5, vmax=vmax, extent=extent,
                     wv_nodes=wv_frames)


def _tpool_init_3dspline(
    data : np.ndarray, data_shape : tuple,
    noise : np.ndarray,
    wvs : np.ndarray,
    scaled_x : np.ndarray,
    scaled_y : np.ndarray,
    bp : np.ndarray,
    stellar_features : np.ndarray,
    bestfit_model : np.ndarray,
    spline3d_paras : np.ndarray, spline3d_paras_shape : tuple,
    spline3d_paras_err : np.ndarray or None,
    spline3d_merr_paras : np.ndarray or None = None,
):
    """
    Initialize shared global variables for the multiprocessing pool.
    """
    global shared_data, shared_data_shape, shared_noise, \
        shared_wvs, shared_scaled_x, shared_scaled_y, shared_bp, shared_stellar_features,shared_bestfit_model, \
        shared_spline3d_paras, shared_spline3d_paras_shape,shared_spline3d_paras_err, shared_spline3d_merr_paras

    shared_data = data
    shared_data_shape = data_shape

    shared_noise = noise
    shared_wvs = wvs
    shared_scaled_x = scaled_x
    shared_scaled_y = scaled_y
    shared_bp = bp
    shared_stellar_features = stellar_features
    shared_bestfit_model = bestfit_model

    shared_spline3d_paras = spline3d_paras
    shared_spline3d_paras_shape = spline3d_paras_shape
    shared_spline3d_paras_err = spline3d_paras_err
    shared_spline3d_merr_paras = spline3d_merr_paras


_CTYPES_TO_NP = {
    ctypes.c_float:   np.float32,
    ctypes.c_double:  np.float64,
    ctypes.c_int32:   np.int32,
    ctypes.c_uint32:  np.uint32,
    ctypes.c_int16:   np.int16,
    ctypes.c_uint16:  np.uint16,
    ctypes.c_int8:    np.int8,
    ctypes.c_uint8:   np.uint8,
}

def _arraytonumpy(shared_array, shape=None, dtype=None):
    """
    Covert a shared array to a numpy array

    Originally from pyklip Wang, J. J., Ruffio, J.-B., De Rosa, R. J., et al. 2015, ASCL, ascl:1506.001
    https://bitbucket.org/pyKLIP/pyklip/src/main/pyklip/parallelized.py

    Args:
        shared_array: a multiprocessing.Array array
        shape: a shape for the numpy array. otherwise, will assume a 1d array
        dtype: data type of the arrays. Should be either ctypes.c_float(default) or ctypes.c_double

    Returns:
        numpy_array: numpy array for vectorized operation. still points to the same memory!
                     returns None is shared_array is None
    """
    if dtype is None:
        dtype = ctypes.c_float
    np_dtype = _CTYPES_TO_NP.get(dtype)

    # if you passed in nothing you get nothing
    if shared_array is None:
        return None

    buf = shared_array.get_obj() if hasattr(shared_array, "get_obj") else shared_array
    numpy_array = np.frombuffer(buf, dtype=np_dtype)
    if shape is not None:
        numpy_array.shape = shape

    return numpy_array




def plot_3dspline_residuals(combdataobj, overwrite = False):
    contnorm_spline3d_filename = combdataobj.default_filenames["compute_starspectrum_contnorm_3dspline"]
    # print(contnorm_spline3d_filename)
    starsub_spline3d_filename = combdataobj.default_filenames["compute_starsubtraction_3dspline"]
    # print(starsub_spline3d_filename)

    if not overwrite and os.path.exists(os.path.join(combdataobj.utils_dir, os.path.basename(starsub_spline3d_filename).replace(".fits", ""))):
        return

    combdataobj.set_coords2ifu()

    ny, nx = combdataobj.trace_id_map.shape
    n_files = ny // nx
    N_traces = int((np.nanmax(combdataobj.trace_id_map) + 1) // n_files)
    mod_trace_id_map = np.mod(combdataobj.trace_id_map, N_traces)

    _unique_ids = np.unique(mod_trace_id_map)
    _unique_ids = _unique_ids[np.where(np.isfinite(_unique_ids))].astype(int)


    hdulist = fits.open(contnorm_spline3d_filename)
    new_wavelengths = hdulist["WAVE"].data
    combined_fluxes = hdulist["COM_FLUXES"].data
    hdulist.close()
    star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False, fill_value=1)

    hdulist = fits.open(starsub_spline3d_filename)
    subtracted_im = hdulist["IM_SUB"].data
    star_model = hdulist["STARMODEL"].data
    fmderived_bad_pixels = hdulist['BADPIX'].data
    spline_paras0 = hdulist["SPLINE_PARAS0"].data
    wv_nodes = hdulist['wv_nodes'].data
    hdulist.close()

    for l in range(n_files):  # n_files
        for slice_id in _unique_ids:
            # if slice_id < 7 or slice_id >= 11:
            #     continue
            # combdataobj_r = np.sqrt(combdataobj.x ** 2 + combdataobj.y ** 2)
            where_finite = np.where(
                # (np.abs(combdataobj.x[(l*2048):((l+1)*2048),:])<1.00) &
                # (np.abs(combdataobj.y[(l*2048):((l+1)*2048),:])<1.00) &
                # (np.abs(combdataobj.x[(l*2048):((l+1)*2048),:])>0.26)&
                #                             (np.abs(combdataobj.wavelengths[(l * 2048):((l + 1) * 2048), :] - 4.5) < 1)
                np.isfinite((combdataobj.bad_pixels * fmderived_bad_pixels * np.isnan(subtracted_im))[
                                (l * 2048):((l + 1) * 2048), :]) &
                (mod_trace_id_map[(l * 2048):((l + 1) * 2048), :] == slice_id) )

            _x = combdataobj.x[(l * 2048):((l + 1) * 2048), :][where_finite]
            _y = combdataobj.y[(l * 2048):((l + 1) * 2048), :][where_finite]
            _w = combdataobj.wavelengths[(l * 2048):((l + 1) * 2048), :][where_finite]
            _d = combdataobj.data[(l * 2048):((l + 1) * 2048), :][where_finite]
            _e = combdataobj.noise[(l * 2048):((l + 1) * 2048), :][where_finite]
            _sm = star_model[(l * 2048):((l + 1) * 2048), :][where_finite]
            _si = subtracted_im[(l * 2048):((l + 1) * 2048), :][where_finite]

            mad_sm = median_abs_deviation(_si)

            fig = plt.figure(figsize=(20,12))
            plt.subplot(2,2, 1)
            plt.title(f"{slice_id}")
            plt.scatter(_x, _y, s=1, label="data")
            plt.gca().invert_xaxis()
            # plt.axis('equal')
            plt.xlabel("x ifu (arcsec)")
            plt.ylabel("y ifu (arcsec)")
            plt.xlim(-4, 4)
            plt.ylim(-4, 4)

            plt.subplot(2,2, 2)
            plt.scatter(_y,  np.abs(_si)/_sm, s=1,label="Residuals/Model")
            plt.scatter(_y,  np.abs(_e)/_sm, s=1,label="Noise floor (sigma/model)",alpha=0.2)
            plt.xlabel("y ifu (arcsec)")
            plt.ylabel("Rel. err.")
            plt.ylim([0.01,10])
            plt.yscale("log")
            plt.legend()

            plt.subplot(2,2, 3)
            plt.scatter(_y, _d, s=1, label="Data")
            # plt.errorbar(_x,_d,yerr=_e,label="data",fmt="none")
            plt.scatter(_y, _sm, s=1, label="Model")
            plt.scatter(_y,  np.abs(_si), s=1, label="Residuals",alpha=0.2)
            plt.xlabel("y ifu (arcsec)")
            plt.ylabel("MJy/sr")
            plt.ylim([np.abs(np.percentile(_d, 1))*0.01,np.abs(np.percentile(_d, 99))*10])
            plt.yscale("log")
            plt.legend()

            plt.subplot(2,2, 4)
            plt.scatter(_x, _d, s=1, label="Data")
            plt.scatter(_x, _sm, s=1, label="Model")
            plt.scatter(_x, np.abs(_si), s=1, label="Residuals",alpha=0.2)
            plt.xlabel("x ifu (arcsec)")
            plt.ylabel("MJy/sr")
            plt.ylim([np.abs(np.percentile(_d, 1))*0.01,np.abs(np.percentile(_d, 99))*10])
            plt.yscale("log")
            plt.legend()

            _plot_dir = os.path.join(combdataobj.utils_dir, os.path.basename(starsub_spline3d_filename).replace(".fits", ""),
                                     f"file_{l}")
            os.makedirs(_plot_dir, exist_ok=True)
            plt.savefig(os.path.join(_plot_dir, f"slice_{slice_id}.png"), dpi=200)
            plt.close(fig)