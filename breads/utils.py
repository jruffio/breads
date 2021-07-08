import matplotlib.pyplot as plt
import numpy as np
import os
from copy import copy
import ctypes
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from scipy.interpolate import InterpolatedUnivariateSpline
import multiprocessing as mp
import pandas as pd
import itertools
from scipy import interpolate
from astropy import constants as const
from scipy.interpolate import interp1d
from scipy.special import loggamma
from py.path import local
from scipy.stats import median_absolute_deviation


def _task_findbadpix(paras):
    data_arr,noise_arr,badpix_arr,med_spec,M_spline = paras
    new_data_arr = np.array(copy(data_arr), '<f4')#.byteswap().newbyteorder()
    new_badpix_arr = copy(badpix_arr)
    res = np.zeros(data_arr.shape) + np.nan
    for k in range(data_arr.shape[1]):
        where_data_finite = np.where(np.isfinite(badpix_arr[:,k])*np.isfinite(data_arr[:,k])*np.isfinite(noise_arr[:,k])*(noise_arr[:,k]!=0))
        if np.size(where_data_finite[0]) == 0:
            res[:,k] = np.nan
            continue
        d = data_arr[where_data_finite[0],k]
        d_err = noise_arr[where_data_finite[0],k]

        M = M_spline[where_data_finite[0],:]*med_spec[where_data_finite[0],None]


        bounds_min = [0, ]* M.shape[1]
        bounds_max = [np.inf, ] * M.shape[1]
        p = lsq_linear(M/d_err[:,None],d/d_err,bounds=(bounds_min, bounds_max)).x
        # p,chi2,rank,s = np.linalg.lstsq(M/d_err[:,None],d/d_err,rcond=None)
        m = np.dot(M,p)
        res[where_data_finite[0],k] = d-m

        # where_bad = np.where((np.abs(res[:,k])>3*np.nanstd(res[:,k])) | np.isnan(res[:,k]))
        where_bad = np.where((np.abs(res[:,k])>3*median_absolute_deviation(res[where_data_finite[0],k])) | np.isnan(res[:,k]))
        new_badpix_arr[where_bad[0],k] = np.nan
        where_bad = np.where(np.isnan(np.correlate(new_badpix_arr[:,k] ,np.ones(2),mode="same")))
        new_badpix_arr[where_bad[0],k] = np.nan
        new_data_arr[where_bad[0],k] = np.nan

        new_data_arr[:,k] = np.array(pd.DataFrame(new_data_arr[:,k]).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]

    return new_data_arr,new_badpix_arr,res

def _remove_edges(paras):
    slices,nan_mask_boxsize = paras
    cp_slices = copy(slices)
    for slice in cp_slices:
        slice[np.where(slice==0)] = np.nan
        slice[np.where(np.isnan(correlate2d(slice,np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))] = np.nan
    cp_slices[:,0:nan_mask_boxsize//2,:] = np.nan
    cp_slices[:,-nan_mask_boxsize//2+1::,:] = np.nan
    cp_slices[:,:,0:nan_mask_boxsize//2] = np.nan
    cp_slices[:,:,-nan_mask_boxsize//2+1::] = np.nan

    return cp_slices


def findbadpix(cube, noisecube=None, badpixcube=None,chunks=20,mypool=None,med_spec=None):


    if noisecube is None:
        noisecube = np.ones(cube.shape)
    if badpixcube is None:
        badpixcube = np.ones(cube.shape)

    new_cube = copy(cube)
    new_badpixcube = copy(badpixcube)
    nz,ny,nx = cube.shape
    res = np.zeros(cube.shape) + np.nan

    x = np.arange(nz)
    x_knots = x[np.linspace(0,nz-1,chunks+1,endpoint=True).astype(np.int)]
    M_spline = get_spline_model(x_knots,x,spline_degree=3)

    N_valid_pix = ny*nx
    if med_spec is None:
        med_spec = np.nanmedian(cube,axis=(1,2))
    new_badpixcube[np.where(cube==0)] = np.nan

    # plt.plot(med_spec)
    # plt.show()

    nan_mask_boxsize = 3
    if mypool is None:
        new_badpixcube = _remove_edges(new_badpixcube,nan_mask_boxsize)
    else:
        numthreads = mypool._processes
        chunk_size = nz//(3*numthreads)
        N_chunks = nz//chunk_size
        wvs_indices_list = []
        slices_list = []
        for k in range(N_chunks-1):
            slices_list.append(new_badpixcube[(k*chunk_size):((k+1)*chunk_size)])
            wvs_indices_list.append(np.arange((k*chunk_size),((k+1)*chunk_size)))
        slices_list.append(new_badpixcube[((N_chunks-1)*chunk_size):nz])
        wvs_indices_list.append(np.arange(((N_chunks-1)*chunk_size),nz))

        outputs_list = mypool.map(_remove_edges, zip(slices_list,itertools.repeat(nan_mask_boxsize)))
        #save it to shared memory
        for indices, out in zip(wvs_indices_list,outputs_list):
            new_badpixcube[indices,:,:] = out


    if mypool is None:
        data_list = np.reshape(new_cube,(nz,nx*ny))
        noise_list = np.reshape(noisecube,(nz,nx*ny))
        badpix_list = np.reshape(new_badpixcube,(nz,nx*ny))
        out_data,out_badpix,out_res = _task_findbadpix((data_list,noise_list,badpix_list,med_spec,M_spline))
        new_cube = np.reshape(new_cube,(nz,ny,nx))
        new_badpixcube = np.reshape(new_badpixcube,(nz,ny,nx))
        res = np.reshape(out_res,(nz,ny,nx))
    else:
        numthreads = mypool._processes
        chunk_size = N_valid_pix//(3*numthreads)
        wherenotnans = np.where(np.nansum(np.isfinite(badpixcube),axis=0)!=0)
        row_valid_pix = wherenotnans[0]
        col_valid_pix = wherenotnans[1]
        N_chunks = N_valid_pix//chunk_size

        row_indices_list = []
        col_indices_list = []
        data_list = []
        noise_list = []
        badpix_list = []
        for k in range(N_chunks-1):
            _row_valid_pix = row_valid_pix[(k*chunk_size):((k+1)*chunk_size)]
            _col_valid_pix = col_valid_pix[(k*chunk_size):((k+1)*chunk_size)]

            row_indices_list.append(_row_valid_pix)
            col_indices_list.append(_col_valid_pix)

            data_list.append(cube[:,_row_valid_pix,_col_valid_pix])
            noise_list.append(noisecube[:,_row_valid_pix,_col_valid_pix])
            badpix_list.append(new_badpixcube[:,_row_valid_pix,_col_valid_pix])

        _row_valid_pix = row_valid_pix[((N_chunks-1)*chunk_size):N_valid_pix]
        _col_valid_pix = col_valid_pix[((N_chunks-1)*chunk_size):N_valid_pix]

        row_indices_list.append(_row_valid_pix)
        col_indices_list.append(_col_valid_pix)

        data_list.append(cube[:,_row_valid_pix,_col_valid_pix])
        noise_list.append(noisecube[:,_row_valid_pix,_col_valid_pix])
        badpix_list.append(new_badpixcube[:,_row_valid_pix,_col_valid_pix])

        outputs_list = mypool.map(_task_findbadpix, zip(data_list,noise_list,badpix_list,
                                                               itertools.repeat(med_spec),
                                                               itertools.repeat(M_spline)))
        for row_indices,col_indices,out in zip(row_indices_list,col_indices_list,outputs_list):
            out_data,out_badpix,out_res = out
            new_cube[:,row_indices,col_indices] = out_data
            new_badpixcube[:,row_indices,col_indices] = out_badpix
            res[:,row_indices,col_indices] = out_res

    return new_badpixcube, new_cube,res



def broaden(wvs,spectrum,R,mppool=None):
    """
    Broaden a spectrum to instrument resolution assuming a gaussian line spread function.

    Args:
        wvs: Wavelength vector (ndarray).
        spectrum: Spectrum vector (ndarray).
        R: Resolution of the instrument as lambda/(delta lambda) with delta lambda the FWHM of the line spread function.
            If scalar, the resolution is assumed to be independent of wavelength.
            Or the resolution can be specified at each wavelength if R is a vector of the same size as wvs.
        mypool: Multiprocessing pool to parallelize the code. If None (default), non parallelization is applied.
            E.g. mppool = mp.Pool(processes=10) # 10 is the number processes

    Returns
        Broadened spectrum
    """
    if mppool is None:
        # Each wavelength processed sequentially
        return _task_broaden((np.arange(np.size(spectrum)).astype(np.int),wvs,spectrum,R))
    else:
        conv_spectrum = np.zeros(spectrum.shape)

        # Divide the spectrum into 100 chunks to be parallelized
        chunk_size=100
        N_chunks = np.size(spectrum)//chunk_size
        indices_list = []
        for k in range(N_chunks-1):
            indices_list.append(np.arange(k*chunk_size,(k+1)*chunk_size).astype(np.int))
        indices_list.append(np.arange((N_chunks-1)*chunk_size,np.size(spectrum)).astype(np.int))
        # Start parallelization
        outputs_list = mppool.map(_task_broaden, zip(indices_list,
                                                               itertools.repeat(wvs),
                                                               itertools.repeat(spectrum),
                                                               itertools.repeat(R)))
        # Retrieve results
        for indices,out in zip(indices_list,outputs_list):
            conv_spectrum[indices] = out

        return conv_spectrum


def _task_broaden(paras):
    """
    Perform the spectrum broadening for broaden().
    """
    indices, wvs, spectrum, R = paras

    if type(R) is np.ndarray:
        Rvec = R
    else:
        Rvec = np.zeros(wvs.shape) + R # Resolution is assumed constant

    conv_spectrum = np.zeros(np.size(indices))
    dwvs = wvs[1::] - wvs[0:(np.size(wvs) - 1)]
    dwvs = np.append(dwvs,dwvs[-1])    # Size of each wavelength bin
    for l, k in enumerate(indices):
        FWHM = wvs[k] / Rvec[k] # Full width at half maximum of the LSF at current wavelength
        sig = FWHM / (2 * np.sqrt(2 * np.log(2))) # standard deviation of the LSF (1D gaussian)
        w = int(np.round(sig / dwvs[k] * 10.)) # Number of bins on each side defining the spec window

        # Extract a smaller a small window around the current wavelength
        stamp_spec = spectrum[np.max([0, k - w]):np.min([np.size(spectrum), k + w])]
        stamp_wvs = wvs[np.max([0, k - w]):np.min([np.size(wvs), k + w])]
        stamp_dwvs = dwvs[np.max([0, k - w]):np.min([np.size(wvs), k + w])]

        gausskernel = 1 / (np.sqrt(2 * np.pi) * sig) * np.exp(-0.5 * (stamp_wvs - wvs[k]) ** 2 / sig ** 2)
        conv_spectrum[l] = np.nansum(gausskernel*stamp_spec*stamp_dwvs) / np.nansum(gausskernel*stamp_dwvs)

    return conv_spectrum

def file_directory(file):
    return os.path.dirname(local(file))
