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
             badpixfraction=0.75,fix_parameters=None,Nrows_max=200,return_extra_outputs=False,detec_KLs=None,wvs_KLs_f=None,
             regularization=None,reg_mean_map=None,reg_std_map=None):

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
    extra_outputs = {}
    # print(nonlin_paras)
    if fix_parameters is not None:
        _nonlin_paras = np.array(fix_parameters)
        _nonlin_paras[np.where(np.array(fix_parameters)==None)] = nonlin_paras
    else:
        _nonlin_paras = nonlin_paras

    if regularization is None:
        min_spline_ampl = 0.02
    else:
        min_spline_ampl = 0.0005


    Natmparas = len(atm_grid.values.shape)-1
    atm_paras = [p for p in _nonlin_paras[0:Natmparas]]
    other_nonlin_paras = _nonlin_paras[Natmparas::]

    # if regularization == "default":
    #     reg_mean_map = cubeobj.data
    #     tmp = np.array(pd.DataFrame(np.concatenate([data, data[::-1]], axis=0)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))

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


    planet_model = atm_grid(atm_paras)[0]
    if np.sum(planet_model)==0 or np.size(atm_grid_wvs) != np.size(planet_model):
        return np.array([]), np.array([]).reshape(0,N_linpara), np.array([])
    else:
        if vsini != 0:
            spinbroad_model = pyasl.fastRotBroad(atm_grid_wvs, planet_model, 0.1, vsini)
        else:
            spinbroad_model = planet_model
        planet_f = interp1d(atm_grid_wvs,spinbroad_model, bounds_error=False, fill_value=np.nan)

        comp_spec = planet_f(wvs* (1 - (rv - cubeobj.bary_RV) / const.c.to('km/s').value))*(u.W/u.m**2/u.um)
        comp_spec = comp_spec*pixarea/cubeobj.webbpsf_spaxel_area # normalized to peak flux
        # comp_spec = comp_spec*cubeobj.aper_to_epsf_peak_f(wvs)*pixarea/cubeobj.webbpsf_spaxel_area # normalized to peak flux
        comp_spec = comp_spec*(wvs*u.um)**2/const.c #from  Flambda to Fnu
        comp_spec = comp_spec.to(u.MJy).value

    mask_comp  = dist2comp_as<radius_as
    larger_mask_comp  = dist2comp_as<3*radius_as
    mask_vec = np.nansum(mask_comp,axis=1) != 0
    rows_ids = np.where(mask_vec)[0]
    if 0:
        print(rows_ids)
        for row_id in rows_ids:
            print(row_id)
            import matplotlib.pyplot as plt
            plt.plot(cubeobj.wavelengths[row_id,:],cubeobj.data[row_id,:])
            plt.scatter(nodes,reg_mean_map[row_id,:],s=50)
            plt.ylim([0,1e-9])
            plt.show()
    rows_ids_mask = np.zeros(np.size(rows_ids))+np.nan
    new_mask = np.tile(mask_vec[:,None],(1,nx))
    # where_trace_finite = np.where(new_mask*np.isfinite(data)*np.isfinite(bad_pixels)*(noise!=0))
    where_trace_finite = np.where(new_mask*larger_mask_comp*np.isfinite(data)*np.isfinite(bad_pixels)*(noise!=0)*np.isfinite(comp_spec))
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
    if return_extra_outputs:
        extra_outputs["wvs"] = w
        extra_outputs["ras"] = x
        extra_outputs["decs"] = y
        row_ids_im = np.tile(np.arange(ny)[:,None],(1,nx))
        extra_outputs["rows"] = row_ids_im[where_trace_finite]
    # A = pixarea[where_trace_finite]
    comp_spec = comp_spec[where_trace_finite]

    # manage all the different cases to define the position of the spline nodes
    if type(nodes) is int:
        N_nodes = nodes
        min_wv, max_wv = np.nanmin(cubeobj.wavelengths),np.nanmax(cubeobj.wavelengths)
        x_nodes = np.linspace(min_wv, max_wv, N_nodes, endpoint=True).tolist()
    elif type(nodes) is list  or type(nodes) is np.ndarray :
        x_nodes = nodes
        if type(nodes[0]) is list or type(nodes[0]) is np.ndarray :
            N_nodes = np.sum([np.size(n) for n in nodes])
        else:
            N_nodes = np.size(nodes)
    else:
        raise ValueError("Unknown format for nodes.")

    # Number of linear parameters
    # N_linpara = Nrows_max * N_nodes + 1
    fitback = False
    # if fitback:
    #     N_linpara += 3*Nrows_max
    # if wvs_KLs_f is not None:
    #     N_linpara += Nrows_max*len(wvs_KLs_f)
    # if detec_KLs is not None:
    #     N_linpara += Nrows_max*detec_KLs.shape[1]
    N_linpara = 1

    # print("coucou")
    # print(np.size(where_trace_finite[0]), (1-badpixfraction) * np.sum(new_mask), vsini < 0)
    if np.size(where_trace_finite[0]) <= (1-badpixfraction) * np.sum(new_mask*larger_mask_comp) or vsini < 0:
        # print("coucou")
        # don't bother to do a fit if there are too many bad pixels
        return np.array([]), np.array([]).reshape(0,N_linpara), np.array([])
    else:

        # Get the linear model (ie the matrix) for the spline
        # M_speckles = np.zeros((Nd,Nrows_max, N_nodes))
        # if regularization == "user":
        #     d_reg_speckles = np.zeros((Nrows_max, N_nodes))+np.nan
        #     s_reg_speckles = np.zeros((Nrows_max, N_nodes))+np.nan
        #     wvs_reg_speckles = np.zeros((Nrows_max, N_nodes))+np.nan
        #     rows_reg_speckles = np.tile((np.pad(rows_ids,(0,Nrows_max-np.size(rows_ids)),constant_values=0))[:,None],(1,N_nodes))
        # if wvs_KLs_f is not None:
        #     M_KLs = np.zeros((Nd,Nrows_max, len(wvs_KLs_f)))
        # if detec_KLs is not None:
        #     M_KLs_detec = np.zeros((Nd,Nrows_max, detec_KLs.shape[1]))
        M_speckles = np.zeros((Nd,Nrows, N_nodes))
        if regularization == "user":
            d_reg_speckles = np.zeros((Nrows, N_nodes))+np.nan
            s_reg_speckles = np.zeros((Nrows, N_nodes))+np.nan
            wvs_reg_speckles = np.zeros((Nrows, N_nodes))+np.nan
            rows_reg_speckles = np.tile((np.pad(rows_ids,(0,Nrows-np.size(rows_ids)),constant_values=0))[:,None],(1,N_nodes))
        if wvs_KLs_f is not None:
            M_KLs = np.zeros((Nd,Nrows, len(wvs_KLs_f)))
        if detec_KLs is not None:
            M_KLs_detec = np.zeros((Nd,Nrows, detec_KLs.shape[1]))
        # M_spline = get_spline_model(x_nodes, np.arange(nx), spline_degree=3)
        for _k in range(Nrows):
            where_finite_and_in_row = np.where(where_trace_finite[0]==rows_ids[_k])
            if np.size(where_finite_and_in_row[0]) == 0:
                continue
            M_spline = get_spline_model(x_nodes, wvs[rows_ids[_k], :], spline_degree=3)
            selec_M_spline = M_spline[where_trace_finite[1][where_finite_and_in_row],:]
            where_del_col = np.where(np.nanmax(np.abs(selec_M_spline),axis=0)<min_spline_ampl)#0.01#0.00001
            if np.size(where_del_col[0]) == selec_M_spline.shape[1]:
                continue
            selec_M_spline[:,where_del_col[0]] = 0
            M_speckles[where_finite_and_in_row[0], _k, :] = selec_M_spline
            rows_ids_mask[_k] = 1
            if regularization == "user":
                d_reg_speckles[_k,:] = reg_mean_map[rows_ids[_k],:]
                d_reg_speckles[_k,where_del_col[0]] = np.nan
                s_reg_speckles[_k,:] = reg_std_map[rows_ids[_k],:]
                s_reg_speckles[_k,where_del_col[0]] = np.nan
                wvs_reg_speckles[_k,:] = x_nodes
                wvs_reg_speckles[_k,where_del_col[0]] = np.nan


            if wvs_KLs_f is not None:
                for KLid,KL_f in enumerate(wvs_KLs_f):
                    KL_vec = KL_f(wvs[rows_ids[_k], :])
                    selec_KL_vec = KL_vec[where_trace_finite[1][where_finite_and_in_row]]
                    M_KLs[where_finite_and_in_row[0], _k, KLid] = selec_KL_vec
            if detec_KLs is not None:
                selec_KL_vec = detec_KLs[where_trace_finite[1][where_finite_and_in_row],:]
                M_KLs_detec[where_finite_and_in_row[0], _k, :] = selec_KL_vec

        # M_speckles = np.reshape(M_speckles, (Nd, Nrows_max * N_nodes))
        # if fitback:
        #     M_background = M_speckles
        # M_speckles = M_speckles*star_func(w)[:,None]
        # if wvs_KLs_f is not None:
        #     M_KLs = np.reshape(M_KLs, (Nd, Nrows_max * len(wvs_KLs_f)))
        # if detec_KLs is not None:
        #     M_KLs_detec = np.reshape(M_KLs_detec, (Nd, Nrows_max * detec_KLs.shape[1]))
        M_speckles = np.reshape(M_speckles, (Nd, Nrows * N_nodes))
        if fitback:
            M_background = M_speckles
        M_speckles = M_speckles*star_func(w)[:,None]
        if wvs_KLs_f is not None:
            M_KLs = np.reshape(M_KLs, (Nd, Nrows * len(wvs_KLs_f)))
        if detec_KLs is not None:
            M_KLs_detec = np.reshape(M_KLs_detec, (Nd, Nrows * detec_KLs.shape[1]))

        # planet_model = atm_grid(atm_paras)[0]
        #
        # if np.sum(np.isnan(planet_model)) >= 1 or np.sum(planet_model)==0 or np.size(atm_grid_wvs) != np.size(planet_model):
        #     return np.array([]), np.array([]).reshape(0,N_linpara), np.array([])
        # else:
        #     if vsini != 0:
        #         spinbroad_model = pyasl.fastRotBroad(atm_grid_wvs, planet_model, 0.1, vsini)
        #     else:
        #         spinbroad_model = planet_model
        #     planet_f = interp1d(atm_grid_wvs,spinbroad_model, bounds_error=False, fill_value=0)
        #
        #     comp_spec = planet_f(w* (1 - (rv - cubeobj.bary_RV) / const.c.to('km/s').value))*(u.W/u.m**2/u.um)
        #     comp_spec = comp_spec*cubeobj.aper_to_epsf_peak_f(w)*A/cubeobj.webbpsf_spaxel_area # normalized to peak flux
        #     comp_spec = comp_spec*(w*u.um)**2/const.c #from  Flambda to Fnu
        #     comp_spec = comp_spec.to(u.MJy).value
        #
        #     comp_model = cubeobj.webbpsf_interp((x-comp_dra_as)*cubeobj.webbpsf_wv0/w, (y-comp_ddec_as)*cubeobj.webbpsf_wv0/w)*comp_spec
        comp_model = cubeobj.webbpsf_interp((x - comp_dra_as) * cubeobj.webbpsf_wv0 / w,
                                            (y - comp_ddec_as) * cubeobj.webbpsf_wv0 / w) * comp_spec

        useless_paras = np.where(np.nansum(M_speckles > np.nanmax(M_speckles) * 0.005, axis=0) == 0)
        M_speckles[:, useless_paras[0]] = 0

        # combine planet model with speckle model
        M = np.concatenate([comp_model[:, None], M_speckles], axis=1)
        if fitback:
            M = np.concatenate([M,M_background], axis=1)
        if wvs_KLs_f is not None:
            M = np.concatenate([M,M_KLs], axis=1)
        if detec_KLs is not None:
            M = np.concatenate([M,M_KLs_detec], axis=1)


        if regularization == "default":
            N_speckles = M_speckles.shape[1]
            s_reg = np.array([np.nan])
            d_reg = np.array([np.nan])
            s_reg = np.concatenate([s_reg, np.nanmax(d)+np.zeros(N_speckles)])
            d_reg = np.concatenate([d_reg, 0+np.zeros(N_speckles)]) #
            if fitback:
                s_reg = np.concatenate([s_reg, np.nan+np.zeros(M_background.shape[1])])
                d_reg = np.concatenate([d_reg, np.nan+np.zeros(M_background.shape[1])])
            if wvs_KLs_f is not None:
                s_reg = np.concatenate([s_reg, np.nan+np.zeros(M_KLs.shape[1])])
                d_reg = np.concatenate([d_reg, np.nan+np.zeros(M_KLs.shape[1])])
            if detec_KLs is not None:
                s_reg = np.concatenate([s_reg, np.nan+np.zeros(M_KLs_detec.shape[1])])
                d_reg = np.concatenate([d_reg, np.nan+np.zeros(M_KLs_detec.shape[1])])
            extra_outputs["regularization"] = (d_reg,s_reg)
            if return_extra_outputs:
                raise Exception("default reg broken. need to add extra outputs")
                extra_outputs["regularization_wvs"] = wvs_reg
                extra_outputs["regularization_rows"] = rows_reg
        elif regularization == "user":
            # s_reg_speckles = np.ravel(reg_std_map[rows_ids,:]*rows_ids_mask[:,None]) #[np.where(np.isfinite(rows_ids_mask))]
            # s_reg_speckles = np.concatenate([s_reg_speckles,np.nan+np.zeros(Nrows_max*N_nodes-np.size(s_reg_speckles))])
            # d_reg_speckles = np.ravel(reg_mean_map[rows_ids,:]*rows_ids_mask[:,None])
            # d_reg_speckles = np.concatenate([d_reg_speckles,np.nan+np.zeros(Nrows_max*N_nodes-np.size(d_reg_speckles))])
            #
            # # wvs_nodes_map = np.tile(nodes[None,:],(np.size(rows_ids),1))
            # # plt.scatter()
            # import matplotlib.pyplot as plt
            # # plt.imshow(d_reg_speckles)
            # # plt.show()
            # #
            # plt.subplot(1,2,1)
            # print(rows_ids)
            # plt.imshow(d_reg_speckles)
            # plt.subplot(1,2,2)
            # plt.imshow(M_speckles)
            # plt.show()

            # for rowid in rows_ids:
            #     plt.scatter(reg_mean_map[rows_ids,:],s=100)
            #     plt.plot(w,d[])
            s_reg_speckles = np.ravel(s_reg_speckles)
            d_reg_speckles = np.ravel(d_reg_speckles)
            wvs_reg_speckles = np.ravel(wvs_reg_speckles)
            rows_reg_speckles = np.ravel(rows_reg_speckles)

            s_reg = np.array([np.nan])
            d_reg = np.array([np.nan])
            wvs_reg = np.array([np.nan])
            rows_reg = np.array([np.nan])
            # s_reg = np.array([1e-16])
            # d_reg = np.array([0])
            s_reg = np.concatenate([s_reg, s_reg_speckles])
            d_reg = np.concatenate([d_reg, d_reg_speckles]) #
            wvs_reg = np.concatenate([wvs_reg, wvs_reg_speckles]) #
            rows_reg = np.concatenate([rows_reg, rows_reg_speckles]) #
            if fitback:
                filler = np.nan+np.zeros(M_background.shape[1])
                s_reg = np.concatenate([s_reg,filler ])
                d_reg = np.concatenate([d_reg, filler])
                wvs_reg = np.concatenate([wvs_reg, filler])
                rows_reg = np.concatenate([rows_reg, filler]) #
            if wvs_KLs_f is not None:
                filler = np.nan+np.zeros(M_KLs.shape[1])
                s_reg = np.concatenate([s_reg, filler])
                d_reg = np.concatenate([d_reg, filler])
                wvs_reg = np.concatenate([wvs_reg, filler])
                rows_reg = np.concatenate([rows_reg, filler]) #
            if detec_KLs is not None:
                filler = np.nan+np.zeros(M_KLs_detec.shape[1])
                s_reg = np.concatenate([s_reg, filler])
                d_reg = np.concatenate([d_reg, filler])
                wvs_reg = np.concatenate([wvs_reg, filler])
                rows_reg = np.concatenate([rows_reg, filler]) #
            extra_outputs["regularization"] = (d_reg,s_reg)
            extra_outputs["regularization_wvs"] = wvs_reg
            extra_outputs["regularization_rows"] = rows_reg

        if return_extra_outputs:
            extra_outputs["where_trace_finite"] = where_trace_finite


        if len(extra_outputs) >= 1:
            return d, M, s,extra_outputs
        else:
            return d, M, s