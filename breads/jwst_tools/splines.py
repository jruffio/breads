from copy import copy
import numpy as np
import itertools
import os

import astropy.io.fits as fits
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from scipy.optimize import lsq_linear

import multiprocessing as mp
import ctypes
from multiprocessing.sharedctypes import RawArray
from tqdm import tqdm

from breads.utils import get_spline_model
from breads.instruments import Instrument
from breads.fit import fitfm

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

    new_im_rows = np.array(copy(im_rows), '<f4')  # .byteswap().newbyteorder()
    new_noise_rows = copy(noise_rows)
    new_badpix_rows = copy(badpix_rows)
    res = np.full(im_rows.shape, np.nan)
    paras_out = np.full((im_rows.shape[0],np.size(wv_nodes)), np.nan)
    for k in range(im_rows.shape[0]):

        M_spline = get_spline_model(wv_nodes, im_wvs_rows[k, :], spline_degree=3)

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

        d = im_rows[k, where_data_finite[0]]
        d_err = noise_rows[k, where_data_finite[0]]

        M = M_spline[where_data_finite[0], :] * stellar_features[k, where_data_finite[0], None]

        if regularization:
            validpara = np.where(np.nansum(M > np.nanmax(M) * 0.00001, axis=0) != 0)
        else:
            validpara = np.where(np.nansum(M > np.nanmax(M) * 0.01, axis=0) != 0)
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
    wv_nodes : 1d array or None (optional)
        If wv_nodes is specified, this wavelength spacing (in micron) will be used to do the splines fitting.
        If None, N_nodes will set an evenly nodes spacing.
    regularization : bool (optional)
        If True, the spline fitting is regularized by providing priors on the value of the continuum at each spline node position.
        The mean and standard deviation of the continuum shape can be provided by reg_mean_map and reg_std_map, or if these are not provided, they will be computed from the data itself.
    reg_mean_map : 2d array or None (optional)
        If regularization is True, this 2d array (same shape as image) provides the mean of the prior.
    reg_std_map : 2d array or None (optional)
        If regularization is True, this 2d array (same shape as image) provides the standard deviation of the prior.

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

    new_image = copy(image)
    new_noise = copy(noise)
    new_badpixs = copy(badpixs)
    new_res = np.full(image.shape, np.nan)
    new_spline_paras = np.zeros((image.shape[0], np.size(wv_nodes)))

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

def _task_fit_3dspline(paras):
    """

    """
    stamp_ids, x_nodes, y_nodes,wv_nodes, wv_ref, stellar_features, threshold, reg_mean_map, reg_std_map, types_tuple = paras
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


    # k0 : left index of stamp in x direction including extended margins
    # k1 : left index of stamp in x direction (defines the first index that will be saved in the output)
    # k2 : right index of stamp in x direction  (defines the last index that will be saved in the output)
    # k3 : right index of stamp in x direction including extended margins
    # Same for l#, but in the y direction
    k0,k1,k2,k3,l0,l1,l2,l3,m0,m1 = stamp_ids
    # if not (k0 ==38 and k1 ==40 and k2 ==49 and k3 ==51 and l0 ==18 and l1 ==20 and l2 ==29 and l3 ==31):
    #     return None
    # else:
    #     print("I am here")
    # print("Current stamp indices",stamp_ids)
    # print("Current stamp values ",x_nodes[k1],x_nodes[k2],y_nodes[l1],y_nodes[l2])
    # print("Current extended stamp values ",x_nodes[k0],x_nodes[k3],y_nodes[l0],y_nodes[l3])

    _x_nodes = x_nodes[k0:k3+1]
    _y_nodes = y_nodes[l0:l3+1]

    extended_bool_map = (scaled_x_np>x_nodes[k0]) & (scaled_x_np<x_nodes[k3]) & \
               (scaled_y_np>y_nodes[l0]) & (scaled_y_np<y_nodes[l3]) & \
               (wvs_np>np.min(wv_nodes)) & (wvs_np<np.max(wv_nodes))
    bool_map_from_bp = (np.isfinite(bp_np) * np.isfinite(data_np) * np.isfinite(noise_np) * (noise_np != 0)* \
                        np.isfinite(wvs_np)* np.isfinite(stellar_features_np) * (data_np/noise_np > -10))
    where_data_finite = np.where(extended_bool_map*bool_map_from_bp)

    N_pix_threshold = 3#np.size(wv_nodes)* np.size(_y_nodes)* np.size(_x_nodes)
    if np.size(where_data_finite[0]) < N_pix_threshold:
        # print("exit 1")
        return None

    inner_bool_map = (scaled_x_np[where_data_finite]>x_nodes[np.max([k1-1,0])]) & (scaled_x_np[where_data_finite]<x_nodes[k2]) & \
               (scaled_y_np[where_data_finite]>y_nodes[np.max([l1-1,0])]) & (scaled_y_np[where_data_finite]<y_nodes[l2]) & \
               (wvs_np[where_data_finite]>np.min(wv_nodes)) & (wvs_np[where_data_finite]<np.max(wv_nodes))
    inner_pixels = np.where(inner_bool_map)
    # if np.size(inner_pixels[0]) < N_pix_threshold:
    #     # print("exit 2")
    #     return None

    _d = data_np[where_data_finite]
    _x = scaled_x_np[where_data_finite]
    _y = scaled_y_np[where_data_finite]
    _w = wvs_np[where_data_finite]
    _e = noise_np[where_data_finite]


    M_spline_x = get_spline_model(_x_nodes, _x, spline_degree=3)
    # plt.plot(np.linspace(_x_nodes[0],_x_nodes[-1],400),
    #          get_spline_model(_x_nodes, np.linspace(_x_nodes[0],_x_nodes[-1],400), spline_degree=3)[:,5])
    # plt.show()
    M_spline_y = get_spline_model(_y_nodes, _y, spline_degree=3)
    M_spline_wvs = get_spline_model(wv_nodes, _w, spline_degree=3)

    M_spline_x_tiled = np.tile(M_spline_x[:,None,None,:], (1,np.size(wv_nodes), np.size(_y_nodes), 1))
    M_spline_y_tiled = np.tile(M_spline_y[:,None,:,None], (1,np.size(wv_nodes), 1, np.size(_x_nodes)))
    M_spline_wvs_tiled = np.tile(M_spline_wvs[:,:,None,None], (1,1, np.size(_y_nodes), np.size(_x_nodes)))
    M_3dspline = M_spline_x_tiled * M_spline_y_tiled * M_spline_wvs_tiled
    M_3dspline = M_3dspline.reshape((M_3dspline.shape[0], -1)) # flatten the last 3 dimensions

    M = M_3dspline * stellar_features_np[where_data_finite][:, None]

    INvalidpara = np.where(~(np.nansum(M > np.nanmax(M) * 0.01, axis=0) != 0))
    M[:, INvalidpara[0]] = 0 # Deactivate those columns in the model matrix

    if reg_mean_map is not None and reg_std_map is not None:
        d_reg, s_reg = np.ravel(reg_mean_map[:, l0:l3 + 1, k0:k3 + 1]), np.ravel(reg_std_map[:, l0:l3 + 1, k0:k3 + 1])#/1e6
    else:
        d_reg, s_reg = None, None
    # d_reg, s_reg = None, None

    _results = fitfm(nonlin_paras=[_d,M,_e,d_reg,s_reg],dataobj=Instrument(),fm_func=_tmp_fm,fm_paras={},
                    marginalize_noise_scaling=False, scale_noise=False)
    bestfit_log_prob, rchi2, linparas, linparas_err = _results
    # plt.figure()
    # print(linparas)
    # # print(np.where(~np.isfinite(_d)))
    # # print(np.where(~np.isfinite(_e)))
    # # print(np.where(~np.isfinite(M)))
    # # plt.scatter(_y,_d,s=1,label="data")
    #
    # plt.figure()
    # plt.plot(_e)
    # plt.plot(_d)
    # # validpara = np.where((np.nansum(M > np.nanmax(M) * 0.001, axis=0)))
    # # plt.scatter(_y,M[:,validpara[0][0]],s=1,label="data")
    # plt.show()
    if np.all(np.isnan(linparas)):
        # print("exit 3")
        return None
    paras_canvas = np.reshape(linparas, (np.size(wv_nodes), np.size(_y_nodes), np.size(_x_nodes)))
    paras_err_canvas = np.reshape(linparas_err, (np.size(wv_nodes), np.size(_y_nodes), np.size(_x_nodes)))
    _linparas = copy(linparas)
    _linparas[np.where(~np.isfinite(_linparas))] = 0
    m = np.dot(M, _linparas)

    # spline3d_paras_np[m0,m1,:,l1:l2+1,k1:k2+1] = paras_canvas[:,(l1-l0):(l2-l0+1),(k1-k0):(k2-k0+1)]
    # spline3d_paras_err_np[m0,m1,:,l1:l2+1,k1:k2+1] = paras_err_canvas[:,(l1-l0):(l2-l0+1),(k1-k0):(k2-k0+1)]
    spline3d_paras_np[m0,m1,:,l0:l3+1,k0:k3+1] = paras_canvas
    spline3d_paras_err_np[m0,m1,:,l0:l3+1,k0:k3+1] = paras_err_canvas
    # concatenate "where" arrays
    combined_idx = tuple(w[inner_pixels] for w in where_data_finite)
    bestfit_model_np[combined_idx] = m[inner_pixels]

    norm_res = (_d[inner_pixels] - m[inner_pixels]) / _e[inner_pixels]
    meddev = median_abs_deviation(norm_res)
    where_bad = np.where((np.abs(norm_res) / meddev > threshold) | np.isnan(norm_res))

    # concatenate "where" arrays to get the indices of the bad pixels in the original data shape
    combined_inner_pixels = tuple(w[where_bad] for w in inner_pixels)
    combined_idx = tuple(w[combined_inner_pixels] for w in where_data_finite)
    bp_np[combined_idx] = np.nan

    # plt.figure()
    # plt.imshow(reg_mean_map[2, l0:l3 + 1, k0:k3 + 1],origin='lower')
    # plt.figure()
    # plt.imshow(reg_std_map[2, l0:l3 + 1, k0:k3 + 1],origin='lower')

    # plt.figure()
    # plt.scatter(_y,_d,s=1,label="data")
    # # plt.show()
    #
    # plt.figure()
    # plt.scatter(_y,_d,s=1,label="data")
    # plt.scatter(_y,m,s=1,label="model")
    # plt.scatter(_y,_d-m,s=1,label="res")
    # plt.legend()
    #
    # plt.figure()
    # plt.imshow(paras_canvas[2,:,:],origin='lower')
    # plt.show()

    return #paras_canvas[:,(l1-l0):(l2-l0+1),(k1-k0):(k2-k0+1)],paras_err_canvas[:,(l1-l0):(l2-l0+1),(k1-k0):(k2-k0+1)]


def fit_3dspline(dataobj,x_nodes,y_nodes,wv_nodes,
                 stamp_size = (0.2,0.2),N_overlap_nodes = 2,
                 stellar_features=None,
                 threshold=10, reg_mean_map=None, reg_std_map=None,
                 max_cores = 1):
    """
    threshold is disabled and not used currently.
    """

    wv_ref = dataobj.breads_header["WV_REF"]

    ss_x, ss_y = stamp_size

    ny, nx = dataobj.data.shape
    N_wv_nodes, N_y_nodes, N_x_nodes = len(wv_nodes), len(y_nodes), len(x_nodes)
    # wv_nodes_grid,y_nodes_grid,x_nodes_grid = np.meshgrid(wv_nodes,y_nodes,x_nodes,indexing='ij' )

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
        stellar_features_np[:] = np.ones(dataobj.data.shape)
    else:
        stellar_features_np[:] = stellar_features

    bestfit_model_mp = RawArray(mp_float_type, nx * ny)
    bestfit_model_np = _arraytonumpy(bestfit_model_mp, data_shape, dtype=mp_float_type)
    bestfit_model_np[:] = np.full(data_shape,np.nan, dtype=np.float32)

    spline3d_paras_mp = RawArray(mp_float_type, 2*2* N_wv_nodes * N_y_nodes * N_x_nodes)
    spline3d_paras_shape = (2,2,N_wv_nodes, N_y_nodes, N_x_nodes )
    spline3d_paras_np = _arraytonumpy(spline3d_paras_mp, spline3d_paras_shape, dtype=mp_float_type)
    spline3d_paras_np[:] = np.full(spline3d_paras_shape,np.nan, dtype=np.float32)

    spline3d_paras_err_mp = RawArray(mp_float_type, 2*2* N_wv_nodes * N_y_nodes * N_x_nodes)
    spline3d_paras_err_np = _arraytonumpy(spline3d_paras_err_mp, spline3d_paras_shape, dtype=mp_float_type)
    spline3d_paras_err_np[:] = np.full(spline3d_paras_shape,np.nan, dtype=np.float32)

    ###################
    # create the small stamps for calculating the 3D spline in.
    # This is because it would not be tractable to fit for the whole thing at once.import numpy as np

    x_chunk_starts_ids = np.searchsorted(x_nodes, np.arange(x_nodes[0], x_nodes[-1], ss_x))
    x_chunk_ends_ids   = np.append(x_chunk_starts_ids[1:] - 1, len(x_nodes) - 1)
    y_chunk_starts_ids = np.searchsorted(y_nodes, np.arange(y_nodes[0], y_nodes[-1], ss_y))
    y_chunk_ends_ids   = np.append(y_chunk_starts_ids[1:] - 1, len(y_nodes) - 1)

    x_bigchunk_starts_ids = np.clip(x_chunk_starts_ids-N_overlap_nodes, 0, N_x_nodes - 1)
    x_bigchunk_ends_ids   = np.clip(x_chunk_ends_ids+N_overlap_nodes, 0, N_x_nodes - 1)
    y_bigchunk_starts_ids = np.clip(y_chunk_starts_ids-N_overlap_nodes, 0, N_y_nodes - 1)
    y_bigchunk_ends_ids   = np.clip(y_chunk_ends_ids+N_overlap_nodes, 0, N_y_nodes - 1)

    # Define chunks
    stamp_list = []
    for k in range(len(x_chunk_starts_ids)):
        for l in range(len(y_chunk_starts_ids)):
            stamp_tuple = (x_bigchunk_starts_ids[k],x_chunk_starts_ids[k],x_chunk_ends_ids[k],x_bigchunk_ends_ids[k],
                           y_bigchunk_starts_ids[l],y_chunk_starts_ids[l],y_chunk_ends_ids[l],y_bigchunk_ends_ids[l],
                           l%2, k%2)
            stamp_list.append(stamp_tuple)
    N_stamps = len(stamp_list)

    if 0:
        stamp_list = stamp_list[(840-41*0):(840-41*0+1)]
        # for stamp_id,stamp_tuple in enumerate(stamp_list):
        #     k0,k1,k2,k3,l0,l1,l2,l3,m0,m1 = stamp_tuple
        #     print("stamp_id",stamp_id,stamp_id % len(x_chunk_starts_ids),(stamp_id-(stamp_id % len(x_chunk_starts_ids)))//len(x_chunk_starts_ids))
        #     print(k0,k1,k2,k3,l0,l1,l2,l3,m0,m1)
        #     print("x", x_nodes[k0],x_nodes[k1],x_nodes[k2],x_nodes[k3])
        #     print("y", y_nodes[l0],y_nodes[l1],y_nodes[l2],y_nodes[l3])
        # N_stamps = len(stamp_list)
        # print(N_stamps)
        # print(np.sqrt(N_stamps))
        # exit()

    _init_args = (
        data_mp, data_shape,
        noise_mp, wvs_mp, scaled_x_mp, scaled_y_mp, bp_mp,stellar_features_mp,bestfit_model_mp,
        spline3d_paras_mp, spline3d_paras_shape,spline3d_paras_err_mp
    )

    if 0 or max_cores <= 1:
        _tpool_init_3dspline(*_init_args)

        print("\tPerforming serial fit_3dspline...")
        for id,stamp_ids in enumerate(stamp_list):
            print(id, stamp_ids)
            paras = stamp_ids, x_nodes, y_nodes,wv_nodes,\
                    wv_ref, stellar_features, threshold, reg_mean_map, reg_std_map,types_tuple

            _task_fit_3dspline(paras)
        # print("coucou here")
        # exit()
    else:

        ctx = mp.get_context("spawn")  # avoids unsafe fork after OpenMP init
        tpool = ctx.Pool(
            processes=max_cores,
            initializer=_tpool_init_3dspline,
            initargs=_init_args,
            maxtasksperchild=50
        )

        args_list = []
        for id,stamp_ids in enumerate(stamp_list):

            paras = stamp_ids, x_nodes, y_nodes,wv_nodes,\
                    wv_ref, stellar_features, threshold, reg_mean_map, reg_std_map,types_tuple
            args_list.append(paras)

        try:
            tasks = [
                tpool.apply_async(_task_fit_3dspline, args=(args_tuple,))
                for args_tuple in args_list
            ]

            for t in tqdm(tasks, desc="Processing blocks"):
                t.wait()
        finally:
            tpool.close()
            tpool.join()

    residuals = dataobj.data - bestfit_model_np
    return bestfit_model_np, noise_np, bp_np, residuals, spline3d_paras_np,spline3d_paras_err_np


def _task_evaluate_3dspline(paras):
    """

    """
    stamp_ids, x_nodes, y_nodes,wv_nodes, wv_ref, stellar_features, types_tuple = paras
    mp_float_type, mp_bp_type = types_tuple

    # data_np = _arraytonumpy(shared_data, shared_data_shape, dtype=mp_float_type)
    noise_np = _arraytonumpy(shared_noise, shared_data_shape, dtype=mp_float_type)
    wvs_np = _arraytonumpy(shared_wvs, shared_data_shape, dtype=mp_float_type)
    scaled_x_np = _arraytonumpy(shared_scaled_x, shared_data_shape, dtype=mp_float_type)
    scaled_y_np = _arraytonumpy(shared_scaled_y, shared_data_shape, dtype=mp_float_type)
    bp_np = _arraytonumpy(shared_bp, shared_data_shape, dtype=mp_float_type)
    stellar_features_np = _arraytonumpy(shared_stellar_features, shared_data_shape, dtype=mp_float_type)
    bestfit_model_np = _arraytonumpy(shared_bestfit_model, shared_data_shape, dtype=mp_float_type)

    spline3d_paras_np = _arraytonumpy(shared_spline3d_paras, shared_spline3d_paras_shape, dtype=mp_float_type)
    spline3d_paras_err_np = _arraytonumpy(shared_spline3d_paras_err, shared_spline3d_paras_shape, dtype=mp_float_type)

    # _mins,_maxs = np.nanmin(scaled_x_np,axis=(1,2)),np.nanmax(scaled_x_np,axis=(1,2))

    # k0 : left index of stamp in x direction including extended margins
    # k1 : left index of stamp in x direction (defines the first index that will be saved in the output)
    # k2 : right index of stamp in x direction  (defines the last index that will be saved in the output)
    # k3 : right index of stamp in x direction including extended margins
    # Same for l#, but in the y direction
    k0,k1,k2,k3,l0,l1,l2,l3,m0,m1 = stamp_ids
    # print("Current stamp indices",stamp_ids)
    # print("Current stamp values ",x_nodes[k1],x_nodes[k2],y_nodes[l1],y_nodes[l2])
    # print("Current extended stamp values ",x_nodes[k0],x_nodes[k3],y_nodes[l0],y_nodes[l3])

    # print(stamp_ids)
    _x_nodes = x_nodes[k0:k3+1]
    _y_nodes = y_nodes[l0:l3+1]

    extended_bool_map = (scaled_x_np>x_nodes[k0]) & (scaled_x_np<x_nodes[k3]) & \
               (scaled_y_np>y_nodes[l0]) & (scaled_y_np<y_nodes[l3]) & \
               (wvs_np>np.min(wv_nodes)) & (wvs_np<np.max(wv_nodes))
    bool_map_from_bp = (np.isfinite(bp_np) * np.isfinite(stellar_features_np))
    where_data_finite = np.where(extended_bool_map*bool_map_from_bp)

    N_pix_threshold = 3
    if np.size(where_data_finite[0]) < N_pix_threshold:
        return

    inner_bool_map = (scaled_x_np[where_data_finite]>x_nodes[np.max([k1-1,0])]) & (scaled_x_np[where_data_finite]<x_nodes[k2]) & \
               (scaled_y_np[where_data_finite]>y_nodes[np.max([l1-1,0])]) & (scaled_y_np[where_data_finite]<y_nodes[l2]) & \
               (wvs_np[where_data_finite]>np.min(wv_nodes)) & (wvs_np[where_data_finite]<np.max(wv_nodes))
    inner_pixels = np.where(inner_bool_map)
    # if np.size(inner_pixels[0]) < N_pix_threshold:
    #     return

    # _d = data_np[where_data_finite]
    _x = scaled_x_np[where_data_finite]
    _y = scaled_y_np[where_data_finite]
    _w = wvs_np[where_data_finite]
    # _e = noise_np[where_data_finite]


    M_spline_x = get_spline_model(_x_nodes, _x, spline_degree=3)
    M_spline_y = get_spline_model(_y_nodes, _y, spline_degree=3)
    M_spline_wvs = get_spline_model(wv_nodes, _w, spline_degree=3)

    M_spline_x_tiled = np.tile(M_spline_x[:,None,None,:], (1,np.size(wv_nodes), np.size(_y_nodes), 1))
    M_spline_y_tiled = np.tile(M_spline_y[:,None,:,None], (1,np.size(wv_nodes), 1, np.size(_x_nodes)))
    M_spline_wvs_tiled = np.tile(M_spline_wvs[:,:,None,None], (1,1, np.size(_y_nodes), np.size(_x_nodes)))
    M_3dspline = M_spline_x_tiled * M_spline_y_tiled * M_spline_wvs_tiled
    M_3dspline = M_3dspline.reshape((M_3dspline.shape[0], -1)) # flatten the last 3 dimensions

    M = M_3dspline * stellar_features_np[where_data_finite][:, None]
    m = np.dot(M, np.ravel(spline3d_paras_np[m0,m1,:,l0:l3+1,k0:k3+1]))
    merr = np.dot(M, np.ravel(spline3d_paras_err_np[m0,m1,:,l0:l3+1,k0:k3+1]))

    combined_idx = tuple(w[inner_pixels] for w in where_data_finite)
    bestfit_model_np[combined_idx] = m[inner_pixels]
    noise_np[combined_idx] = merr[inner_pixels]

    # print(stamp_ids)
    # print("Current stamp indices",stamp_ids)
    # print("Current stamp values ",x_nodes[k1],x_nodes[k2],y_nodes[l1],y_nodes[l2])
    # print("Current extended stamp values ",x_nodes[k0],x_nodes[k3],y_nodes[l0],y_nodes[l3])
    # # plt.figure()
    # # _xxn,_yyn = np.meshgrid(_x_nodes, _y_nodes)
    # # print(_xxn.shape,spline3d_paras_np[m0,m1,2,l0:l3+1,k0:k3+1].shape)
    # # plt.subplot(1,2,1)
    # # plt.scatter(_xxn,spline3d_paras_np[2,l0:l3+1,k0:k3+1],s=1)
    # # plt.subplot(1,2,2)
    # # plt.scatter(_yyn,spline3d_paras_np[2,l0:l3+1,k0:k3+1],s=1)
    # # plt.figure()
    # # plt.imshow(spline3d_paras_np[0,0,2,l0:l3+1,k0:k3+1],origin="lower")
    #
    # plt.figure()
    # plt.imshow(bestfit_model_np,origin="lower")
    # plt.show()

    # plt.figure()
    # dx = x_nodes[1] - x_nodes[0]
    # dy = y_nodes[1] - y_nodes[0]
    # extent = [x_nodes[0] - dx / 2.0, x_nodes[-1] + dx / 2.0, y_nodes[0] - dy / 2.0, y_nodes[-1] + dy / 2.0]
    # # plt.imshow(np.log10(np.abs(spline3d_paras_np[m0,m1,2,:,:])),origin="lower",extent=extent)
    # plt.imshow(np.log10(np.abs(np.nanmean((spline3d_paras_np[:,:,2,:,:]),axis=(0,1)))),origin="lower",extent=extent)
    # plt.clim([0,10])
    #
    # plt.figure()
    # dx = x_nodes[1] - x_nodes[0]
    # dy = y_nodes[1] - y_nodes[0]
    # extent = [x_nodes[0] - dx / 2.0, x_nodes[-1] + dx / 2.0, y_nodes[0] - dy / 2.0, y_nodes[-1] + dy / 2.0]
    # plt.imshow(np.log10(np.abs(spline3d_paras_np[m0,m1,2,:,:])),origin="lower",extent=extent)
    # # plt.imshow(np.log10(np.abs(np.nanmean((spline3d_paras_np[:,:,2,:,:]),axis=(0,1)))),origin="lower",extent=extent)
    # plt.clim([0,10])
    # plt.show()
    #
    # plt.plot(_mins)
    # plt.figure()
    # plt.plot(_maxs)
    # print(_x_nodes)
    # print(_y_nodes)
    #
    # # plt.figure()
    # # plt.scatter(_x/4.0*_w,_y/4.0*_w,s=1)
    # #
    # # plt.figure()
    # # plt.scatter(_x/4.0*_w,m,s=1)
    # plt.show()
    # exit()
    return

def evaluate_3dspline_grid(x_vec,y_vec,wv_sampling,spline3d_filename,
                           N_overlap_nodes = 2,max_cores = 1,
                      stamp_size = None):
    wv_grid,y_grid,x_grid = np.meshgrid(wv_sampling,x_vec,y_vec,indexing='ij' )
    stellar_features = None
    return evaluate_3dspline(x_grid,y_grid,wv_grid,
                          spline3d_filename,
                     stellar_features=stellar_features,N_overlap_nodes = N_overlap_nodes,
                     max_cores = max_cores,stamp_size=stamp_size)

def evaluate_3dspline_pointcloud(dataobj, spline3d_filename,stellar_features=None,
                           N_overlap_nodes = 2, max_cores=1,
                      stamp_size = None):
    ifux,ifuy = dataobj.get_ifu_coords()
    # if hasattr(dataobj,'star_func'):
    #     stellar_features = dataobj.star_func(dataobj.wavelengths)
    return evaluate_3dspline(ifux,ifuy, dataobj.wavelengths,
                             spline3d_filename,
                             stellar_features=stellar_features, N_overlap_nodes=N_overlap_nodes,
                             max_cores=max_cores,stamp_size=stamp_size)

def evaluate_3dspline(ifux,ifuy,wvs,
                      spline3d_filename,
                 stellar_features=None,N_overlap_nodes = 2,
                 max_cores = 1,
                      stamp_size = None):
    """

    """

    hdulist = fits.open(spline3d_filename)
    wv_nodes = hdulist["wv_nodes"].data
    x_nodes = hdulist["x_nodes"].data
    y_nodes = hdulist["y_nodes"].data
    spline3d_paras = hdulist["SPLINE_PARAS0"].data
    spline3d_paras_err = hdulist["SPLINE_PARAS0_ERR"].data
    if stamp_size is None:
        ss_x,ss_y = (hdulist['BREADS'].header['3DSPLSSX'], hdulist['BREADS'].header['3DSPLSSY'])
    else:
        ss_x,ss_y = stamp_size
    wv_ref = hdulist['BREADS'].header['WV_REF']
    hdulist.close()

    N_wv_nodes, N_y_nodes, N_x_nodes = len(wv_nodes), len(y_nodes), len(x_nodes)
    # wv_nodes_grid,y_nodes_grid,x_nodes_grid = np.meshgrid(wv_nodes,y_nodes,x_nodes,indexing='ij' )

    mp_float_type = ctypes.c_float
    mp_bp_type = ctypes.c_uint8
    types_tuple = (mp_float_type, mp_bp_type)

    data_size = np.size(ifux)
    # data_mp = RawArray(mp_float_type, data_size)
    data_mp = None
    data_shape = ifux.shape
    # data_np = _arraytonumpy(data_mp, data_shape, dtype=mp_float_type)
    # data_np[:] = np.full(data_shape,np.nan, dtype=np.float32)

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
        stellar_features_np[:] = np.ones(ifux.shape)
    else:
        stellar_features_np[:] = stellar_features

    bestfit_model_mp = RawArray(mp_float_type, data_size)
    bestfit_model_np = _arraytonumpy(bestfit_model_mp, data_shape, dtype=mp_float_type)
    bestfit_model_np[:] = np.full(data_shape,np.nan, dtype=np.float32)

    spline3d_paras_mp = RawArray(mp_float_type, 2*2* N_wv_nodes * N_y_nodes * N_x_nodes)
    spline3d_paras_shape = (2,2,N_wv_nodes, N_y_nodes, N_x_nodes )
    spline3d_paras_np = _arraytonumpy(spline3d_paras_mp, spline3d_paras_shape, dtype=mp_float_type)
    spline3d_paras_np[:] = copy(spline3d_paras)
    spline3d_paras_np[np.where(~np.isfinite(spline3d_paras_np))] = 0

    spline3d_paras_err_mp = RawArray(mp_float_type, 2*2* N_wv_nodes * N_y_nodes * N_x_nodes)
    spline3d_paras_err_np = _arraytonumpy(spline3d_paras_err_mp, spline3d_paras_shape, dtype=mp_float_type)
    spline3d_paras_err_np[:] = copy(spline3d_paras_err)
    spline3d_paras_err_np[np.where(~np.isfinite(spline3d_paras_err_np))] = 0

    ###################
    # create the small stamps for calculating the 3D spline in.
    # This is because it would not be tractable to fit for the whole thing at once.import numpy as np

    x_chunk_starts_ids = np.searchsorted(x_nodes, np.arange(x_nodes[0], x_nodes[-1], ss_x))
    x_chunk_ends_ids   = np.append(x_chunk_starts_ids[1:] - 1, len(x_nodes) - 1)
    y_chunk_starts_ids = np.searchsorted(y_nodes, np.arange(y_nodes[0], y_nodes[-1], ss_y))
    y_chunk_ends_ids   = np.append(y_chunk_starts_ids[1:] - 1, len(y_nodes) - 1)

    x_bigchunk_starts_ids = np.clip(x_chunk_starts_ids-N_overlap_nodes, 0, N_x_nodes - 1)
    x_bigchunk_ends_ids   = np.clip(x_chunk_ends_ids+N_overlap_nodes, 0, N_x_nodes - 1)
    y_bigchunk_starts_ids = np.clip(y_chunk_starts_ids-N_overlap_nodes, 0, N_y_nodes - 1)
    y_bigchunk_ends_ids   = np.clip(y_chunk_ends_ids+N_overlap_nodes, 0, N_y_nodes - 1)

    # Define chunks
    stamp_list = []
    for k in range(len(x_chunk_starts_ids)):
        for l in range(len(y_chunk_starts_ids)):
            stamp_tuple = (x_bigchunk_starts_ids[k],x_chunk_starts_ids[k],x_chunk_ends_ids[k],x_bigchunk_ends_ids[k],
                           y_bigchunk_starts_ids[l],y_chunk_starts_ids[l],y_chunk_ends_ids[l],y_bigchunk_ends_ids[l],
                           l%2, k%2)
            stamp_list.append(stamp_tuple)
    N_stamps = len(stamp_list)

    if 0:
        print(spline3d_filename)
        stamp_list = stamp_list[200:221]
        # stamp_list = stamp_list[243:245]
        # stamp_list = stamp_list[(840-41*0)::41]
        # for stamp_id,stamp_tuple in enumerate(stamp_list):
        #     k0,k1,k2,k3,l0,l1,l2,l3,m0,m1 = stamp_tuple
        #     print("stamp_id",stamp_id)
        #     print("x", x_nodes[k0],x_nodes[k1],x_nodes[k2],x_nodes[k3])
        #     print("y", y_nodes[l0],y_nodes[l1],y_nodes[l2],y_nodes[l3])
        # N_stamps = len(stamp_list)
        # print(N_overlap_nodes)
        # exit()

    _init_args = (
        data_mp, data_shape,
        noise_mp, wvs_mp, scaled_x_mp, scaled_y_mp, bp_mp,stellar_features_mp,bestfit_model_mp,
        spline3d_paras_mp, spline3d_paras_shape,spline3d_paras_err_mp
    )

    if 0 or max_cores <= 1:
        _tpool_init_3dspline(*_init_args)

        print("\tPerforming serial evaluate_3dspline...")
        for id,stamp_ids in enumerate(stamp_list):
            print(id,len(stamp_list),stamp_ids)
            paras = stamp_ids, x_nodes, y_nodes,wv_nodes,\
                    wv_ref, stellar_features, types_tuple

            _task_evaluate_3dspline(paras)
        #     print("coucou")
        # exit()
    else:

        ctx = mp.get_context("spawn")  # avoids unsafe fork after OpenMP init
        tpool = ctx.Pool(
            processes=max_cores,
            initializer=_tpool_init_3dspline,
            initargs=_init_args,
            maxtasksperchild=50
        )

        args_list = []
        for id,stamp_ids in enumerate(stamp_list):

            paras = stamp_ids, x_nodes, y_nodes,wv_nodes,\
                    wv_ref, stellar_features, types_tuple
            args_list.append(paras)

        try:
            tasks = [
                tpool.apply_async(_task_evaluate_3dspline, args=(args_tuple,))
                for args_tuple in args_list
            ]

            for t in tqdm(tasks, desc="Processing blocks"):
                t.wait()
        finally:
            tpool.close()
            tpool.join()

    return bestfit_model_np, noise_np

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
):
    """
    Initialize shared global variables for the multiprocessing pool.
    """
    global shared_data, shared_data_shape, shared_noise, \
        shared_wvs, shared_scaled_x, shared_scaled_y, shared_bp, shared_stellar_features,shared_bestfit_model, \
        shared_spline3d_paras, shared_spline3d_paras_shape,shared_spline3d_paras_err

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
    print(contnorm_spline3d_filename)
    starsub_spline3d_filename = combdataobj.default_filenames["compute_starsubtraction_3dspline"]
    print(starsub_spline3d_filename)

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
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)

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
            plt.ylim([0.1*mad_sm,np.nanmax(_d)])
            plt.yscale("log")
            plt.legend()

            plt.subplot(2,2, 4)
            plt.scatter(_x, _d, s=1, label="Data")
            plt.scatter(_x, _sm, s=1, label="Model")
            plt.scatter(_x, np.abs(_si), s=1, label="Residuals",alpha=0.2)
            plt.xlabel("x ifu (arcsec)")
            plt.ylabel("MJy/sr")
            plt.ylim([0.1*mad_sm,np.nanmax(_d)])
            plt.yscale("log")
            plt.legend()

            _plot_dir = os.path.join(combdataobj.utils_dir, os.path.basename(starsub_spline3d_filename).replace(".fits", ""),
                                     f"file_{l}")
            os.makedirs(_plot_dir, exist_ok=True)
            plt.savefig(os.path.join(_plot_dir, f"slice_{slice_id}.png"), dpi=200)
            plt.close(fig)