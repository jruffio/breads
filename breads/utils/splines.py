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
from scipy.interpolate import BSpline
from scipy.interpolate import CubicHermiteSpline

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

def get_bspline_knots(nodes, spline_degree=3, not_a_knot=False, return_degree=False):
    """ Clamped knot vector used by get_bspline_model for one segment of nodes.

    The boundary nodes appear spline_degree+1 times in total (spline_degree extra copies on each side). These repeated
    end knots do not add any freedom: they only make the basis stop at the end nodes.

    - not_a_knot=False: every node is a knot. The number of B-spline basis functions is np.size(nodes)+spline_degree-1.
    - not_a_knot=True: same functions as the cardinal basis of get_spline_model (scipy InterpolatedUnivariateSpline,
      "not-a-knot" end conditions): the (spline_degree-1)/2 nodes next to each end node are not knots (for a cubic
      spline, the second and second-to-last nodes), so the number of basis functions is np.size(nodes). As in
      get_spline_model, the degree is reduced to np.size(nodes)-1 if there are too few nodes. Only for odd degrees.

    Args:
        nodes: 1d array of nodes (one segment).
        spline_degree: Degree of the B-splines.
        not_a_knot: See above.
        return_degree: If True, also return the degree actually used (only differs from spline_degree if not_a_knot).

    Returns:
        t, or (t, degree) if return_degree.
    """
    nodes = np.asarray(nodes)
    k = spline_degree
    if not_a_knot:
        if k % 2 == 0:
            raise ValueError("The not-a-knot B-spline basis is only defined for odd spline degrees.")
        if np.size(nodes) <= k:
            # Too few nodes: a single polynomial of degree n_nodes-1 (n_nodes functions), as in get_spline_model
            k = np.size(nodes) - 1
            interior = nodes[1:1]
        else:
            n_skip = (k + 1) // 2  # nodes that are not interior knots on each side (including the end node)
            interior = nodes[n_skip:np.size(nodes) - n_skip]
        t = np.concatenate((np.repeat(nodes[0], k + 1), interior, np.repeat(nodes[-1], k + 1)))
    else:
        t = np.concatenate((np.repeat(nodes[0], k), nodes, np.repeat(nodes[-1], k)))
    return (t, k) if return_degree else t

def get_bspline_model(x_knots, x_samples, spline_degree=3, not_a_knot=False):
    """ Compute a B-spline based linear model.
    Same interface/semantics as get_spline_model, but each column of M
    is now a local B-spline basis function instead of a cardinal
    (nodal-interpolating) basis function.
    The model coefficients are therefore B-spline control points, not the values of the function at the nodes.

    Args:
        x_knots: List of nodes (breakpoints) as np.ndarray in the same units as x_samples.
            x_knots can also be a list of ndarrays/list to model discontinous functions.
        x_samples: Vector of x values. ie, the sampling of the data.
        spline_degree: Degree of the B-splines (default: 3). Unlike get_spline_model, the degree is not reduced when
            there are few nodes: clamped B-splines of any degree are defined with as few as 2 nodes (e.g. 3 nodes and
            degree 3 give 5 basis functions). With not_a_knot=True, the degree is reduced as in get_spline_model.
        not_a_knot: If False (default), every node is a knot. If True, the knots are those of the cardinal basis of
            get_spline_model (not-a-knot end conditions, see get_bspline_knots): the B-splines then span exactly the
            same functions as get_spline_model, with one basis function per node.

    Returns:
        M: Matrix of size (D,N) with D the size of x_samples and N = sum over segments of (n_nodes + spline_degree - 1)
            (not_a_knot=False: N is larger than the total number of nodes), or N = total number of nodes
            (not_a_knot=True).
    """
    x_samples = np.asarray(x_samples)
    if type(x_knots[0]) is list or type(x_knots[0]) is np.ndarray:
        x_knots_list = x_knots
    else:
        x_knots_list = [x_knots]
    if np.size(x_knots_list) <= 1:
        return np.ones((np.size(x_samples), 1))

    M_list = []
    for nodes in x_knots_list:
        nodes = np.asarray(nodes)
        min_, max_ = np.min(nodes), np.max(nodes)
        inbounds = np.where((min_ < x_samples) & (x_samples < max_))
        _x = x_samples[inbounds]

        t, k = get_bspline_knots(nodes, spline_degree, not_a_knot=not_a_knot, return_degree=True)
        N_basis = np.size(t) - k - 1  # = n_nodes + k - 1, or n_nodes if not_a_knot

        M = np.zeros((np.size(x_samples), N_basis))
        if np.size(_x) != 0:
            # Design matrix: columns are the B-spline basis functions evaluated at _x.
            M[inbounds[0], :] = BSpline.design_matrix(_x, t, k).toarray()
        M_list.append(M)
    return np.concatenate(M_list, axis=1)


def get_catmull_rom_model(x_knots, x_samples):
    """ Compute a Catmull-Rom spline based linear model.
    Same interface as get_spline_model: if Y = [y1, y2, ...] are the values of the function at the nodes [x1, x2, ...],
    np.dot(M, Y) is the interpolated function at x_samples.

    The function is a cubic between consecutive nodes, going through the node values, with the slope at each node
    computed from the node values: the slope at that node of the parabola through the node and its two neighbors
    (np.gradient, edge_order=2; at the first and last nodes, the parabola through the first or last three nodes).
    For evenly spaced nodes, this is the usual Catmull-Rom slope (y[i+1] - y[i-1]) / (x[i+1] - x[i-1]) in the
    interior. Unlike get_spline_model, each column is local: it is zero beyond 2 nodes on each side of its node.
    With 2 nodes, the model is a straight line.

    Args:
        x_knots: List of nodes as np.ndarray in the same units as x_samples.
            x_knots can also be a list of ndarrays/list to model discontinous functions.
        x_samples: Vector of x values. ie, the sampling of the data.

    Returns:
        M: Matrix of size (D,N) with D the size of x_samples and N the total number of nodes.
    """
    x_samples = np.asarray(x_samples)
    if type(x_knots[0]) is list or type(x_knots[0]) is np.ndarray:
        x_knots_list = x_knots
    else:
        x_knots_list = [x_knots]
    if np.size(x_knots_list) <= 1:
        return np.ones((np.size(x_samples), 1))

    M_list = []
    for nodes in x_knots_list:
        nodes = np.asarray(nodes, dtype=float)
        n_nodes = np.size(nodes)
        min_, max_ = np.min(nodes), np.max(nodes)
        inbounds = np.where((min_ < x_samples) & (x_samples < max_))
        _x = x_samples[inbounds]
        edge_order = 2 if n_nodes >= 3 else 1

        M = np.zeros((np.size(x_samples), n_nodes))
        for chunk in range(n_nodes):
            values = np.zeros(n_nodes)
            values[chunk] = 1
            slopes = np.gradient(values, nodes, edge_order=edge_order)
            M[inbounds[0], chunk] = CubicHermiteSpline(nodes, values, slopes)(_x)
        M_list.append(M)
    return np.concatenate(M_list, axis=1)
