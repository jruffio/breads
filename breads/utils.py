import numpy as np
import os
from copy import copy
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from scipy.interpolate import InterpolatedUnivariateSpline
import multiprocessing as mp
import pandas as pd
import itertools
from py.path import local
from scipy.stats import median_absolute_deviation
from scipy.optimize import lsq_linear
from scipy.signal import correlate2d
from  scipy.interpolate import interp1d


def get_err_from_posterior(x,posterior):
    """
    Return the mode, and the left and right errors of a distribution. The errors are defined with a 68% confidence level.

    Args:
        x: Sampling of the 1D posterior
        posterior: Posterior array

    Returns:
        Mode, left error, right error

    """
    ind = np.argsort(posterior)
    cum_posterior = np.zeros(np.shape(posterior))
    cum_posterior[ind] = np.cumsum(posterior[ind])
    cum_posterior = cum_posterior/np.max(cum_posterior)
    argmax_post = np.argmax(cum_posterior)
    if len(x[0:np.min([argmax_post+1,len(x)])]) < 2:
        lx = x[0]
    else:
        tmp_cumpost = cum_posterior[0:np.min([argmax_post+1,len(x)])]
        tmp_x= x[0:np.min([argmax_post+1,len(x)])]
        deriv_tmp_cumpost = np.insert(tmp_cumpost[1::]-tmp_cumpost[0:np.size(tmp_cumpost)-1],np.size(tmp_cumpost)-1,0)
        try:
            whereinflection = np.where(deriv_tmp_cumpost<0)[0][0]
            where2keep = np.where((tmp_x<=tmp_x[whereinflection])+(tmp_cumpost>=tmp_cumpost[whereinflection]))
            tmp_cumpost = tmp_cumpost[where2keep]
            tmp_x = tmp_x[where2keep]
        except:
            pass
        lf = interp1d(tmp_cumpost,tmp_x,bounds_error=False,fill_value=x[0])
        lx = lf(1-0.6827)
    if len(x[argmax_post::]) < 2:
        rx=x[-1]
    else:
        tmp_cumpost = cum_posterior[argmax_post::]
        tmp_x= x[argmax_post::]
        deriv_tmp_cumpost = np.insert(tmp_cumpost[1::]-tmp_cumpost[0:np.size(tmp_cumpost)-1],np.size(tmp_cumpost)-1,0)
        try:
            whereinflection = np.where(deriv_tmp_cumpost>0)[0][0]
            where2keep = np.where((tmp_x>=tmp_x[whereinflection])+(tmp_cumpost>=tmp_cumpost[whereinflection]))
            tmp_cumpost = tmp_cumpost[where2keep]
            tmp_x = tmp_x[where2keep]
        except:
            pass
        rf = interp1d(tmp_cumpost,tmp_x,bounds_error=False,fill_value=x[-1])
        rx = rf(1-0.6827)
    return x[argmax_post],x[argmax_post]-lx,rx-x[argmax_post]

def _task_findbadpix(paras):
    data_arr,noise_arr,badpix_arr,med_spec,M_spline,threshold = paras
    new_data_arr = np.array(copy(data_arr), '<f4')#.byteswap().newbyteorder()
    new_badpix_arr = copy(badpix_arr)
    res = np.zeros(data_arr.shape) + np.nan
    for k in range(data_arr.shape[1]):
        where_data_finite = np.where(np.isfinite(med_spec)*np.isfinite(badpix_arr[:,k])*np.isfinite(data_arr[:,k])*np.isfinite(noise_arr[:,k])*(noise_arr[:,k]!=0))
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
        where_bad = np.where((np.abs(res[:,k])>threshold*median_absolute_deviation(res[where_data_finite[0],k])) | np.isnan(res[:,k]))
        new_badpix_arr[where_bad[0],k] = np.nan
        where_bad = np.where(np.isnan(np.correlate(new_badpix_arr[:,k] ,np.ones(2),mode="same")))
        new_badpix_arr[where_bad[0],k] = np.nan
        new_data_arr[where_bad[0],k] = np.nan

        new_data_arr[:,k] = np.array(pd.DataFrame(new_data_arr[:,k]).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]

    return new_data_arr,new_badpix_arr,res

def _remove_edges(paras):
    slices,nan_mask_boxsize = paras
    cp_slices = copy(slices)
    if nan_mask_boxsize != 0:
        for slice in cp_slices:
            slice[np.where(slice==0)] = np.nan
            slice[np.where(np.isnan(correlate2d(slice,np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))] = np.nan
        cp_slices[:,0:nan_mask_boxsize//2,:] = np.nan
        cp_slices[:,-nan_mask_boxsize//2+1::,:] = np.nan
        cp_slices[:,:,0:nan_mask_boxsize//2] = np.nan
        cp_slices[:,:,-nan_mask_boxsize//2+1::] = np.nan
    return cp_slices

def corrected_wavelengths(data, off0, off1, center_data):
    wavs = data.read_wavelengths.astype(float) * u.micron
    if center_data:
        wavs = wavs + (wavs - np.mean(wavs)) * off1 + off0 * u.angstrom
    else:
        wavs = wavs * (1 + off1) + off0 * u.angstrom
    return wavs

def findbadpix(cube, noisecube=None, badpixcube=None,chunks=20,mypool=None,med_spec=None,nan_mask_boxsize=3,threshold=3):
    if noisecube is None:
        noisecube = np.ones(cube.shape)
    if badpixcube is None:
        badpixcube = np.ones(cube.shape)

    new_cube = copy(cube)
    new_badpixcube = copy(badpixcube)
    new_badpixcube[np.where(np.isnan(cube)*np.isnan(noisecube))] = np.nan
    nz,ny,nx = cube.shape
    res = np.zeros(cube.shape) + np.nan

    x = np.arange(nz)
    x_knots = x[np.linspace(0,nz-1,chunks+1,endpoint=True).astype(np.int)]
    M_spline = get_spline_model(x_knots,x,spline_degree=3)

    N_valid_pix = ny*nx
    if med_spec is None:
        _cube = copy(cube)
        _cube[np.where(_cube<=0)] = np.nan
        med_spec = np.nanmedian(cube,axis=(1,2))
    new_badpixcube[np.where(cube==0)] = np.nan

    if mypool is None:
        new_badpixcube = _remove_edges((new_badpixcube,nan_mask_boxsize))
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
        out_data,out_badpix,out_res = _task_findbadpix((data_list,noise_list,badpix_list,med_spec,M_spline,threshold))
        new_cube = np.reshape(out_data,(nz,ny,nx))
        new_badpixcube = np.reshape(out_badpix,(nz,ny,nx))
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
                                                               itertools.repeat(M_spline),
                                                                itertools.repeat(threshold)))
        for row_indices,col_indices,out in zip(row_indices_list,col_indices_list,outputs_list):
            out_data,out_badpix,out_res = out
            new_cube[:,row_indices,col_indices] = out_data
            new_badpixcube[:,row_indices,col_indices] = out_badpix
            res[:,row_indices,col_indices] = out_res

    return new_badpixcube, new_cube, res



def broaden(wvs,spectrum,R,mppool=None,kernel=None):
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
        return _task_broaden((np.arange(np.size(spectrum)).astype(np.int),wvs,spectrum,R,kernel))
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
                                                               itertools.repeat(R),
                                                               itertools.repeat(kernel)))
        # Retrieve results
        for indices,out in zip(indices_list,outputs_list):
            conv_spectrum[indices] = out

        return conv_spectrum

def clean_nans(arr, set_to="median", allowed_range=None, continuum=None):
    if set_to == "continuum":
        cont = np.ravel(continuum)
        shape = arr.shape
        arr.flatten()
        for i in range(len(arr)):
            if np.isnan(arr[i]):
                arr[i] = cont[i]
        arr.reshape(shape)
        if allowed_range is not None:
            min_v, max_v = allowed_range
            arr[arr > max_v] = set_to
            arr[arr < min_v] = set_to
        return

    if set_to == "median":
        set_to = np.nanmedian(arr)
    np.nan_to_num(arr, copy = False, nan=set_to)
    if allowed_range is not None:
        min_v, max_v = allowed_range
        arr[arr > max_v] = set_to
        arr[arr < min_v] = set_to
    
        

def _task_broaden(paras):
    """
    Perform the spectrum broadening for broaden().
    """
    indices, wvs, spectrum, R,kernel = paras

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

        if kernel is None:
            gausskernel = 1 / (np.sqrt(2 * np.pi) * sig) * np.exp(-0.5 * (stamp_wvs - wvs[k]) ** 2 / sig ** 2)
        else:
            gausskernel = kernel((stamp_wvs - wvs[k])/wvs[k])
        gausskernel[np.where(np.isnan(stamp_spec))] = np.nan
        conv_spectrum[l] = np.nansum(gausskernel*stamp_spec*stamp_dwvs) / np.nansum(gausskernel*stamp_dwvs)

    return conv_spectrum

def file_directory(file):
    return os.path.dirname(local(file))

def LPFvsHPF(myvec,cutoff):
    """
    Ask JB to write documentation!
    """
    myvec_cp = copy(myvec)
    #handling nans:
    wherenans = np.where(np.isnan(myvec_cp))
    window = int(round(np.size(myvec_cp)/(cutoff/2.)/2.))#cutoff
    tmp = np.array(pd.DataFrame(np.concatenate([myvec_cp, myvec_cp[::-1]], axis=0)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))
    myvec_cp_lpf = np.array(pd.DataFrame(tmp).rolling(window=window, center=True).median().interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[0:np.size(myvec), 0]
    myvec_cp[wherenans] = myvec_cp_lpf[wherenans]


    fftmyvec = np.fft.fft(np.concatenate([myvec_cp, myvec_cp[::-1]], axis=0))
    LPF_fftmyvec = copy(fftmyvec)
    LPF_fftmyvec[cutoff:(2*np.size(myvec_cp)-cutoff+1)] = 0
    LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec_cp)]
    HPF_myvec = myvec_cp - LPF_myvec


    LPF_myvec[wherenans] = np.nan
    HPF_myvec[wherenans] = np.nan

    # plt.figure(10)
    # plt.plot(myvec_cp,label="fixed")
    # plt.plot(myvec,label="ori")
    # plt.plot(myvec_cp_lpf,label="lpf")
    # plt.plot(LPF_myvec,label="lpf fft")
    # plt.legend()
    # plt.show()
    return LPF_myvec,HPF_myvec

def gaussian2D(nx, ny, mu_x, mu_y, sig_x, sig_y, A):
    """
    Two Dimensional Gaussian for getting PSF for different wavelength slices
    """
    x_vals, y_vals = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    gauss = A * np.exp(-((x_vals - mu_x) ** 2) / (2 * sig_x * sig_x)) * \
        np.exp(-((y_vals - mu_y) ** 2) / (2 * sig_y * sig_y))
    return gauss

def get_spline_model(x_knots, x_samples, spline_degree=3):
    """
    Compute a spline based linear model.
    If Y=[y1,y2,..] are the values of the function at the location of the node [x1,x2,...].
    np.dot(M,Y) is the interpolated spline corresponding to the sampling of the x-axis (x_samples)


    Args:
        x_knots: List of nodes for the spline interpolation as np.ndarray in the same units as x_samples.
            x_knots can also be a list of ndarrays/list to model discontinous functions.
        x_samples: Vector of x values. ie, the sampling of the data.
        spline_degree: Degree of the spline interpolation (default: 3).
            if np.size(x_knots) <= spline_degree, then spline_degree = np.size(x_knots)-1

    Returns:
        M: Matrix of size (D,N) with D the size of x_samples and N the total number of nodes.
    """
    if type(x_knots[0]) is list or type(x_knots[0]) is np.ndarray:
        x_knots_list = x_knots
    else:
        x_knots_list = [x_knots]

    if np.size(x_knots_list) <= 1:
        return np.ones((np.size(x_samples),1))
    if np.size(x_knots_list) <= spline_degree:
        spline_degree = np.size(x_knots)-1

    M_list = []
    for nodes in x_knots_list:
        M = np.zeros((np.size(x_samples), np.size(nodes)))
        min,max = np.min(nodes),np.max(nodes)
        inbounds = np.where((min<x_samples)&(x_samples<max))
        _x = x_samples[inbounds]

        for chunk in range(np.size(nodes)):
            tmp_y_vec = np.zeros(np.size(nodes))
            tmp_y_vec[chunk] = 1
            spl = InterpolatedUnivariateSpline(nodes, tmp_y_vec, k=spline_degree, ext=0)
            M[inbounds[0], chunk] = spl(_x)
        M_list.append(M)
    return np.concatenate(M_list, axis=1)


def broaden_kernel(wvs,spectrum,kernel):
    """
    Broaden a spectrum to instrument resolution assuming a custom kernel.

    Args:
        wvs: Wavelength vector (ndarray).
        spectrum: Spectrum vector (ndarray).
        kernel: custom broadening kernel as a function kernel((wvs - wvs_curr) / wvs_curr)

    Returns
        Broadened spectrum
    """
    cp_spectrum = copy(spectrum)
    dwvs = wvs[1::] - wvs[0:(np.size(wvs) - 1)]
    dwvs = np.append(dwvs, dwvs[-1])  # Size of each wavelength bin
    where_nans = np.where(np.isnan(cp_spectrum))
    cp_spectrum[where_nans] = 0
    dwvs[where_nans] = 0

    wvs_mat = np.tile(wvs[None, :],(np.size(wvs),1))
    kernel_mat = kernel((wvs_mat - wvs[:,None]) / wvs[:,None])
    conv_spectrum = np.dot(kernel_mat,cp_spectrum * dwvs)#/np.dot(kernel_mat,dwvs)
    return conv_spectrum
