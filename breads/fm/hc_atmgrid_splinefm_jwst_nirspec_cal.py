import numpy as np
from copy import copy
import pandas as pd
from astropy import constants as const
from  scipy.interpolate import interp1d
from PyAstronomy import pyasl
import astropy.units as u

from breads.utils import broaden
# from breads.utils import LPFvsHPF

from breads.utils import get_spline_model


# pos: (x,y) or fiber, position of the companion
def hc_atmgrid_splinefm_jwst_nirspec_cal(nonlin_paras, cubeobj, atm_grid=None, atm_grid_wvs=None, star_func=None,radius_as=0.2, nodes=20,
             badpixfraction=0.75,fix_parameters=None,photfilter_f=None,Nrows_max=200,return_where_finite=False):
    """
    For high-contrast companions (planet + speckles).
    Description todo

    Args:
        nonlin_paras: Non-linear parameters of the model, which are first the parameters defining the atmopsheric grid
            (atm_grid). The following parameters are the spin (vsini), the radial velocity, and the position (if loc is
            not defined) of the planet in the FOV.
                [atm paras ....,vsini,rv,y,x] for 3d cubes (e.g. OSIRIS)
                [atm paras ....,vsini,rv,y] for 2d (e.g. KPIC, y being fiber)
                [atm paras ....,vsini,rv] for 1d spectra
        cubeobj: Data object.
            Must inherit breads.instruments.instrument.Instrument.
        atm_grid: Planet atmospheric model grid as a scipy.interpolate.RegularGridInterpolator object. Make sure the
            wavelength coverage of the grid is just right and not too big as it will slow down the spin broadening.
        atm_grid_wvs: Wavelength sampling on which atm_grid is defined. Wavelength needs to be uniformly sampled.
        star_func: interpolator object to model the stellar spectrum by continuum normalization to fit the speckle noise in each row of the detector.
        radius_as: Each pixel with coordinates that are within radius_as of the assumed companion location will be
            included in the fit. Must be expressed in arcsecond.
            Default is 0.2 arcsecond
        nodes: If int, number of nodes equally distributed. If list, custom locations of nodes [x1,x2,..].
            To model discontinous functions, use a list of list [[x1,...],[xn,...]]. Expressed in pixel coordinates.
        badpixfraction: Max fraction of bad pixels in data.
        fix_parameters: List. Use to fix the value of some non-linear parameters. The values equal to None are being
                    fitted for, other elements will be fixed to the value specified.

    Returns:
        d: Data as a 1d vector with bad pixels removed (no nans)
        M: Linear model as a matrix of shape (Nd,Np) with bad pixels removed (no nans). Nd is the size of the data
            vector and Np = N_nodes*boxw^2+1 is the number of linear parameters.
        s: Noise vector (standard deviation) as a 1d vector matching d.
    """
    # print(nonlin_paras)
    if fix_parameters is not None:
        _nonlin_paras = np.array(fix_parameters)
        _nonlin_paras[np.where(np.array(fix_parameters)==None)] = nonlin_paras
    else:
        _nonlin_paras = nonlin_paras

    Natmparas = len(atm_grid.values.shape)-1
    atm_paras = [p for p in _nonlin_paras[0:Natmparas]]
    other_nonlin_paras = _nonlin_paras[Natmparas::]

    data = cubeobj.data
    ny, nx = data.shape
    noise = cubeobj.noise
    bad_pixels = cubeobj.bad_pixels
    ra_array = cubeobj.dra_as_array
    dec_array = cubeobj.ddec_as_array
    wvs = cubeobj.wavelengths
    # dwvs = cubeobj.delta_wavelengths
    pixarea = cubeobj.area2d

    vsini,rv = other_nonlin_paras[0:2]
    # Defining the position of companion
    comp_dra_as,comp_ddec_as = other_nonlin_paras[2],other_nonlin_paras[3]

    #extract planet trace
    dist2comp_as = np.sqrt((ra_array-comp_dra_as)**2+(dec_array-comp_ddec_as)**2)
    if 0:
        kplc,lplc = np.unravel_index(np.nanargmin(dist2comp_as),dist2comp_as.shape)
        data[0:kplc,:] = np.nan
        data[kplc+1::,:] = np.nan

    mask_comp  = dist2comp_as<radius_as
    mask_vec = np.nansum(mask_comp,axis=1) != 0
    rows_ids = np.where(mask_vec)[0]
    new_mask = np.tile(mask_vec[:,None],(1,nx))
    # where_trace_finite = np.where(new_mask*np.isfinite(data)*np.isfinite(bad_pixels)*(noise!=0))
    where_trace_finite = np.where(new_mask*(dist2comp_as<0.5)*np.isfinite(data)*np.isfinite(bad_pixels)*(noise!=0))
    Nrows= np.size(rows_ids)
    Nd = np.size(where_trace_finite[0])
    if Nrows >Nrows_max:
        raise Exception("Too many rows")

    d = data[where_trace_finite]
    s = noise[where_trace_finite]
    w = wvs[where_trace_finite]
    # dw = dwvs[where_trace_finite]
    x = ra_array[where_trace_finite]
    y = dec_array[where_trace_finite]
    A = pixarea[where_trace_finite]

    # manage all the different cases to define the position of the spline nodes
    if type(nodes) is int:
        N_nodes = nodes
        min_wv, max_wv = np.nanmin(cubeobj.wavelengths),np.nanmax(cubeobj.wavelengths)
        x_knots = np.linspace(min_wv, max_wv, N_nodes, endpoint=True).tolist()
    elif type(nodes) is list  or type(nodes) is np.ndarray :
        x_knots = nodes
        if type(nodes[0]) is list or type(nodes[0]) is np.ndarray :
            N_nodes = np.sum([np.size(n) for n in nodes])
        else:
            N_nodes = np.size(nodes)
    else:
        raise ValueError("Unknown format for nodes.")

    # Number of linear parameters
    fitback = False
    if fitback:
        N_linpara = Nrows_max * N_nodes +1 + 3*Nrows_max
    else:
        N_linpara = Nrows_max * N_nodes +1

    if np.size(where_trace_finite[0]) <= (1-badpixfraction) * np.sum(new_mask) or vsini < 0:
        # don't bother to do a fit if there are too many bad pixels
        return np.array([]), np.array([]).reshape(0,N_linpara), np.array([])
    else:
        # Get the linear model (ie the matrix) for the spline
        M_speckles = np.zeros((Nd,Nrows_max, N_nodes))
        # M_spline = get_spline_model(x_knots, np.arange(nx), spline_degree=3)
        for _k in range(Nrows):
            M_spline = get_spline_model(x_knots, wvs[rows_ids[_k],:], spline_degree=3)
            where_finite_and_in_row = np.where(where_trace_finite[0]==rows_ids[_k])
            if np.size(where_finite_and_in_row[0]) == 0:
                continue
            selec_M_spline = M_spline[where_trace_finite[1][where_finite_and_in_row],:]
            where_del_col = np.where(np.nanmax(np.abs(selec_M_spline),axis=0)<0.5)
            if np.size(where_del_col[0]) == selec_M_spline.shape[1]:
                continue
            selec_M_spline[:,where_del_col[0]] = 0
            M_speckles[where_finite_and_in_row[0], _k, :] = selec_M_spline
        M_speckles = np.reshape(M_speckles, (Nd, Nrows_max * N_nodes))
        if fitback:
            M_background = M_speckles
        M_speckles = M_speckles*star_func(w)[:,None]

        planet_model = atm_grid(atm_paras)[0]

        if np.sum(np.isnan(planet_model)) >= 1 or np.sum(planet_model)==0 or np.size(atm_grid_wvs) != np.size(planet_model):
            return np.array([]), np.array([]).reshape(0,N_linpara), np.array([])
        else:
            if vsini != 0:
                spinbroad_model = pyasl.fastRotBroad(atm_grid_wvs, planet_model, 0.1, vsini)
            else:
                spinbroad_model = planet_model
            planet_f = interp1d(atm_grid_wvs,spinbroad_model, bounds_error=False, fill_value=0)

            comp_spec = planet_f(w* (1 - (rv - cubeobj.bary_RV) / const.c.to('km/s').value))*(u.W/u.m**2/u.um)
            comp_spec = comp_spec*cubeobj.aper_to_epsf_peak_f(w)*A/cubeobj.webbpsf_spaxel_area # normalized to peak flux
            comp_spec = comp_spec*(w*u.um)**2/const.c #from  Flambda to Fnu
            comp_spec = comp_spec.to(u.MJy).value

            comp_model = cubeobj.webbpsf_interp((x-comp_dra_as)*cubeobj.webbpsf_wv0/w, (y-comp_ddec_as)*cubeobj.webbpsf_wv0/w)*comp_spec

        # combine planet model with speckle model
        if fitback:
            M = np.concatenate([comp_model[:, None], M_speckles,M_background], axis=1)
        else:
            M = np.concatenate([comp_model[:, None], M_speckles], axis=1)
        if return_where_finite:
            return d, M, s,where_trace_finite
        else:
            return d, M, s