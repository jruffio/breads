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
def hc_atmgrid_2dsplinefm_jwst_nirspec_cal_FixedSlit(nonlin_paras, dataobj, ifuy_array=None,atm_grid=None, atm_grid_wvs=None,star_func=None,
                                                     wv_nodes=None,N_wvs_nodes=20,ifuy_nodes=None,delta_ifuy=0.05,
                                                     badpixfraction=0.75,fix_parameters=None,return_extra_outputs=False,
                                                     reg_mean_map=None,reg_std_map=None):

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
        dataobj: Data object.
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
    extra_outputs = {}
    # print(nonlin_paras)
    if fix_parameters is not None:
        _nonlin_paras = np.array(fix_parameters)
        _nonlin_paras[np.where(np.array(fix_parameters)==None)] = nonlin_paras
    else:
        _nonlin_paras = nonlin_paras

    Natmparas = len(atm_grid.values.shape)-1
    atm_paras = [p for p in _nonlin_paras[0:Natmparas]]
    other_nonlin_paras = _nonlin_paras[Natmparas::]

    data = dataobj.data
    ny, nx = data.shape
    noise = dataobj.noise
    bad_pixels = dataobj.bad_pixels
    ra_array = dataobj.dra_as_array
    dec_array = dataobj.ddec_as_array
    wvs = dataobj.wavelengths
    pixarea = dataobj.area2d

    if wv_nodes is None:
        wv_nodes = np.linspace(np.nanmin(dataobj.wavelengths), np.nanmax(dataobj.wavelengths), N_wvs_nodes, endpoint=True)

    if ifuy_nodes is None:
        ifuy_min, ifuy_max = np.nanmin(ifuy_array), np.nanmax(ifuy_array)
        ifuy_min, ifuy_max = np.floor(ifuy_min * 10) / 10, np.ceil(ifuy_max * 10) / 10
        ifuy_nodes = np.arange(ifuy_min, ifuy_max + 0.1, delta_ifuy)

    vsini,rv = other_nonlin_paras[0:2]
    # Defining the position of companion
    comp_dra_as,comp_ddec_as = other_nonlin_paras[2],other_nonlin_paras[3]

    # Number of linear parameters
    N_nodes =  np.size(ifuy_nodes) * np.size(wv_nodes)
    N_linpara = N_nodes + 1

    planet_model = atm_grid(atm_paras)[0]
    if np.sum(planet_model)==0 or np.size(atm_grid_wvs) != np.size(planet_model):
        print('##!!##')
        return np.array([]), np.array([]).reshape(0,N_linpara), np.array([])
    else:
        if vsini != 0:
            spinbroad_model = pyasl.fastRotBroad(atm_grid_wvs, planet_model, 0.1, vsini)
        else:
            spinbroad_model = planet_model
        planet_f = interp1d(atm_grid_wvs,spinbroad_model, bounds_error=False, fill_value=np.nan)

        comp_spec = planet_f(wvs* (1 - (rv - dataobj.bary_RV) / const.c.to('km/s').value))*(u.W/u.m**2/u.um)
        comp_spec = comp_spec*pixarea/dataobj.webbpsf_spaxel_area # normalized to peak flux
        comp_spec = comp_spec*(wvs*u.um)**2/const.c #from  Flambda to Fnu
        comp_spec = comp_spec.to(u.MJy).value

    where_finite = np.where(np.isfinite(data)*np.isfinite(bad_pixels)*(noise!=0)*np.isfinite(comp_spec))
    Nd = np.size(where_finite[0])

    d = data[where_finite]
    s = noise[where_finite]
    w = wvs[where_finite]
    x = ra_array[where_finite]
    y = dec_array[where_finite]
    ifuy = ifuy_array[where_finite]
    if return_extra_outputs:
        extra_outputs["wvs"] = w
        extra_outputs["ras"] = x
        extra_outputs["decs"] = y
        extra_outputs["ifuy"] = ifuy
        row_ids_im = np.tile(np.arange(ny)[:,None],(1,nx))
        extra_outputs["rows"] = row_ids_im[where_finite]
    comp_spec = comp_spec[where_finite]

    if np.size(where_finite[0]) <= (1-badpixfraction) * (nx*ny) or vsini < 0:
        # don't bother to do a fit if there are too many bad pixels
        print('______')
        return np.array([]), np.array([]).reshape(0,N_linpara), np.array([])
    else:

        M_spline_ifuy = get_spline_model(ifuy_nodes, ifuy, spline_degree=3)
        M_spline_wvs = get_spline_model(wv_nodes, w, spline_degree=3)
        M_spline_ifuy_repeated = np.repeat(M_spline_ifuy, np.size(wv_nodes), axis=1)
        M_spline_wvs_tiled = np.tile(M_spline_wvs, (1, np.size(ifuy_nodes)))
        M_speckles = M_spline_ifuy_repeated * M_spline_wvs_tiled

        M_speckles = M_speckles*star_func(w)[:,None]

        comp_model = dataobj.webbpsf_interp((x - comp_dra_as) * dataobj.webbpsf_wv0 / w,
                                            (y - comp_ddec_as) * dataobj.webbpsf_wv0 / w) * comp_spec

        # combine planet model with speckle model
        M = np.concatenate([comp_model[:, None], M_speckles], axis=1)

        if 1:
            s_reg_speckles = np.ravel(reg_std_map)
            d_reg_speckles = np.ravel(reg_mean_map)
            ifuy_nodes_grid,wv_nodes_grid = np.meshgrid(ifuy_nodes,wv_nodes,indexing="ij")
            wvs_reg_speckles = np.ravel(wv_nodes_grid)
            ifuy_reg_speckles = np.ravel(ifuy_nodes_grid)

            s_reg = np.array([np.nan])
            d_reg = np.array([np.nan])
            wvs_reg = np.array([np.nan])
            ifuy_reg = np.array([np.nan])

            s_reg = np.concatenate([s_reg, s_reg_speckles])
            d_reg = np.concatenate([d_reg, d_reg_speckles])
            wvs_reg = np.concatenate([wvs_reg, wvs_reg_speckles])
            ifuy_reg = np.concatenate([ifuy_reg, ifuy_reg_speckles])

            extra_outputs["regularization"] = (d_reg,s_reg)
            extra_outputs["regularization_wvs"] = wvs_reg
            extra_outputs["regularization_ifuy"] = ifuy_reg

        if return_extra_outputs:
            extra_outputs["where_finite"] = where_finite

        if len(extra_outputs) >= 1:
            return d, M, s,extra_outputs
        else:
            return d, M, s