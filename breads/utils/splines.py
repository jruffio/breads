import itertools
import os
from copy import copy

import astropy.coordinates
import astropy.io.fits as fits
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.simbad import Simbad
from py.path import local
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d
from scipy.optimize import lsq_linear
from scipy.signal import correlate2d
from scipy.stats import median_abs_deviation

def get_spline_model(x_knots, x_samples, spline_degree=3):
    """ Compute a spline based linear model.
    If Y = [y1, y2, ...] are the values of the function at the location of the node [x1,x2,...].
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

def filter_spec_with_spline(wvs, spec,specerr=None,x_nodes=None,M_spline=None):
    """

    Parameters
    ----------
    wvs
    spec
    specerr
    x_nodes
    m_spline

    Returns
    -------

    """
    if specerr is None:
        specerr = np.ones(spec.shape)

    if M_spline is None:
        M_spline = get_spline_model(x_nodes, wvs, spline_degree=3)

    M = M_spline/specerr[:,None]
    d = spec/specerr
    where_finite = np.where(np.isfinite(d))
    M = M[where_finite[0],:]
    d = d[where_finite]

    paras = lsq_linear(M,d).x
    m = np.dot(M, paras)
    r = d - m

    LPF_spec = np.zeros(spec.shape)+np.nan
    HPF_spec = np.zeros(spec.shape)+np.nan
    LPF_spec[where_finite] = m*specerr[where_finite]
    HPF_spec[where_finite] = r*specerr[where_finite]

    return HPF_spec,LPF_spec