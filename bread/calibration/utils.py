import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
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

def read_osiris(filename,skip_baryrv=False):
    """
    Read OSIRIS spectral cube
    """
    with pyfits.open(filename) as hdulist:
        prihdr = hdulist[0].header
        curr_mjdobs = prihdr["MJD-OBS"]
        cube = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
        cube = return_64x19(cube)
        noisecube = np.rollaxis(np.rollaxis(hdulist[1].data,2),2,1)
        noisecube = return_64x19(noisecube)
        # cube = np.moveaxis(cube,0,2)
        badpixcube = np.rollaxis(np.rollaxis(hdulist[2].data,2),2,1)
        badpixcube = return_64x19(badpixcube)
        # badpixcube = np.moveaxis(badpixcube,0,2)
        badpixcube = badpixcube.astype(dtype=ctypes.c_double)
        badpixcube[np.where(badpixcube==0)] = np.nan
        badpixcube[np.where(badpixcube!=0)] = 1

    nz,ny,nx = cube.shape
    init_wv = prihdr["CRVAL1"]/1000. # wv for first slice in mum
    dwv = prihdr["CDELT1"]/1000. # wv interval between 2 slices in mum
    wvs=np.linspace(init_wv,init_wv+dwv*nz,nz,endpoint=False)

    if not skip_baryrv:
        keck = EarthLocation.from_geodetic(lat=19.8283 * u.deg, lon=-155.4783 * u.deg, height=4160 * u.m)
        sc = SkyCoord(float(prihdr["RA"]) * u.deg, float(prihdr["DEC"]) * u.deg)
        barycorr = sc.radial_velocity_correction(obstime=Time(float(prihdr["MJD-OBS"]), format="mjd", scale="utc"),
                                                 location=keck)
        baryrv = barycorr.to(u.km / u.s).value
    else:
        baryrv = None

    return wvs, cube, noisecube, badpixcube, baryrv


def return_64x19(cube):
    #cube should be nz,ny,nx
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
                ccmap[dk,dl] = np.nansum(im[dk:np.min([dk+64,ny]),dl:np.min([dl+19,nx])]*onesmask[0:(np.min([dk+64,ny])-dk),0:(np.min([dl+19,nx])-dl)])
        dk,dl = np.unravel_index(np.nanargmax(ccmap),ccmap.shape)
        if np.size(cube.shape) == 3:
            return cube[:,dk:(dk+64),dl:(dl+19)]
        else:
            return cube[dk:(dk+64),dl:(dl+19)]
    else:
        return cube
