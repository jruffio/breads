import matplotlib.pyplot as plt
import numpy as np
from glob import glob
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
from scipy.optimize import lsq_linear
from scipy import interpolate
from astropy import constants as const
from scipy.interpolate import interpn
from scipy.interpolate import interp1d
from scipy.special import loggamma
from py.path import local


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
