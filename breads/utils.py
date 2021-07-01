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

def return_64x19(cube):
    # cube should be nz,ny,nx
    if np.size(cube.shape) == 3:
        _,ny,nx = cube.shape
    else:
        ny,nx = cube.shape
    onesmask = np.ones((64,19))
    if (ny != 64 or nx != 19):
        mask = copy(cube).astype(np.float)
        mask[np.where(mask==0)]=np.nan
        mask[np.where(np.isfinite(mask))]=1
        if np.size(cube.shape) == 3:
            im = np.nansum(mask,axis=0)
        else:
            im = mask
        ccmap =np.zeros((3,3))
        for dk in range(3):
            for dl in range(3):
                ccmap[dk,dl] = np.nansum(im[dk:np.min([dk+64,ny]),dl:np.min([dl+19,nx])]
                                         *onesmask[0:(np.min([dk+64,ny])-dk),0:(np.min([dl+19,nx])-dl)])
        dk,dl = np.unravel_index(np.nanargmax(ccmap),ccmap.shape)
        if np.size(cube.shape) == 3:
            return cube[:,dk:(dk+64),dl:(dl+19)]
        else:
            return cube[dk:(dk+64),dl:(dl+19)]
    else:
        return cube

def file_directory(file):
    return os.path.dirname(local(file))
