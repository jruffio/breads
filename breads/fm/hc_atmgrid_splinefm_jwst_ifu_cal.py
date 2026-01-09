import astropy.units as u
import numpy as np
from PyAstronomy import pyasl
from astropy import constants as const
from scipy.interpolate import interp1d

from breads.utils import get_spline_model



# pos: (x,y) or fiber, position of the companion
def hc_atmgrid_splinefm_jwst_ifu_cal(nonlin_paras, cubeobj, atm_grid=None, atm_grid_wvs=None, star_func=None,radius_as=0.2, nodes=20,
             badpixfraction=0.75, fix_parameters=None, Nrows_max=200, detec_KLs=None, wvs_KLs_f=None,
             regularization=None, reg_mean_map=None, reg_std_map=None):

    """
    For high-contrast companions (planet + speckles).
    Function to create the model matrix M ([Planet spectrum, Star spectrum * M_spline]), the input data 1d vector d and the noise 1d vector.
    Regularization matrix can be added to the model M with the reg_mean_map and reg_std_map parameters.
    PCA can be added as well with the detec_KLs and wvs_KLs_f parameters.
    Args:
        nonlin_paras: list
            Non-linear parameters of the model, which are first the parameters defining the atmopsheric grid
            (atm_grid). The following parameters are the spin (vsini), the radial velocity, and the position (if loc is
            not defined) of the planet in the FOV.
                [atm paras ....,vsini,rv,y,x] for 3d cubes (e.g. OSIRIS)
                [atm paras ....,vsini,rv,y] for 2d (e.g. KPIC, y being fiber)
                [atm paras ....,vsini,rv] for 1d spectra
        cubeobj: Must inherit breads.instruments.jwst_IFUs
            Calibrated data object.
        atm_grid: Planet atmospheric model grid as a scipy.interpolate.RegularGridInterpolator object.
            Make sure the wavelength coverage of the grid is just right and not too big as it will slow down the spin broadening.
        atm_grid_wvs: 1d array
            Wavelength sampling on which atm_grid is defined. Wavelength needs to be uniformly sampled.
        star_func: scipy.interpolate._interpolate.interp1d object (only tested with this interpolator object so far)
            Interpolator object to model the stellar spectrum by continuum normalization to fit the speckle noise in each trace of the detector.
        radius_as: float (default is 0.2 arcsec)
            Each pixel with coordinates that are within radius_as of the assumed companion location will be
            included in the fit. Must be expressed in arcsecond.
        nodes: int, list of int or list of list
            If int, number of nodes equally distributed. If list, custom locations of nodes [x1,x2,..].
            To model discontinous functions, use a list of list [[x1,...],[xn,...]] TODO not sure I get this.
            Expressed in pixel coordinates.
        badpixfraction: float
            Max fraction of bad pixels in data. if above the threshold it returns all empty numpy arrays for M, d and s matrix.
        fix_parameters: List.
            Use to fix the value of some non-linear parameters. The values equal to None are being fitted for,
            other elements will be fixed to the value specified.
        Nrows_max: int (default is 200)
            Maximum number of rows to modelize simultaneously. Basically, if radius_as is too large compare to lambda/D, too many traces would be fitted simultaneously.
            This parameters assures that the matrix dimensions won't be too large.
        detec_KLs: None
        wvs_KLs_f: None
        regularization: str "user" or None.
            If set to "user", the values in reg_mean_map and reg_std_map are used to regularized the model fitting incorporating the regularization in the matrices d,M and s.
        reg_mean_map: 2d array (Nrows, Ncolumns)
            regularization 2d map of the signal. It helps regularize the sub-splines around those mean values.
        reg_std_map: 2d array (Nrows, Ncolumns)
            regularization 2d map of the noise. It helps regularize the sub-splines in those standard deviation intervals.

    Returns:
        d: Data as a 1d vector with bad pixels removed (no nans)
        M: Linear model as a matrix of shape (Nd,Np) with bad pixels removed (no nans). Nd is the size of the data
            vector and Np = N_nodes*boxw^2+1 is the number of linear parameters.
        s: Noise vector (standard deviation) as a 1d vector matching d.
    """
    extra_outputs = {}
    if fix_parameters is not None:
        _nonlin_paras = np.array(fix_parameters)
        _nonlin_paras[np.where(np.array(fix_parameters)==None)] = nonlin_paras
    else:
        _nonlin_paras = nonlin_paras

    if regularization is None:
        min_spline_ampl = 0.02
    else:
        min_spline_ampl = 0.005

    Natmparas = len(atm_grid.values.shape)-1
    atm_paras = [p for p in _nonlin_paras[0:Natmparas]]
    other_nonlin_paras = _nonlin_paras[Natmparas::]

    ifu_name = cubeobj.ifu_name
    data = cubeobj.data
    ny, nx = data.shape
    noise = cubeobj.noise
    bad_pixels = cubeobj.bad_pixels
    ra_array = cubeobj.dra_as_array
    dec_array = cubeobj.ddec_as_array
    wvs = cubeobj.wavelengths
    pixarea = cubeobj.area2d

    if ifu_name == 'miri':
        if cubeobj.channel_reduction == '1' or cubeobj.channel_reduction == '4':
            ra_array[:, 500:] = np.nan
            dec_array[:, 500:] = np.nan
            wvs[:, 500:] = np.nan
        else:
            ra_array[:, :500] = np.nan
            dec_array[:, :500] = np.nan
            wvs[:, :500] = np.nan

        #Transpose the arrays for miri
        data = data.transpose()
        ny, nx = data.shape
        bad_pixels = bad_pixels.transpose()
        noise = noise.transpose()
        wvs = wvs.transpose()
        dec_array = dec_array.transpose()
        ra_array = ra_array.transpose()
        pixarea = pixarea.transpose()

    vsini, rv = other_nonlin_paras[0:2]

    if vsini < 0:
        raise ValueError("vsini must be >= 0")

    # Defining the position of companion
    comp_dra_as, comp_ddec_as = other_nonlin_paras[2], other_nonlin_paras[3]

    comp_spec = _interpolate_companion_spectrum(cubeobj, atm_grid, atm_grid_wvs, atm_paras, vsini, rv, wvs, pixarea)

    where_trace_finite, Nd, new_mask, larger_mask_comp, Nrows, rows_ids = _extract_companion_traces(ra_array, dec_array, comp_dra_as, comp_ddec_as, radius_as, nx, data, bad_pixels, noise, comp_spec, star_func, Nrows_max, wvs)

    d = data[where_trace_finite]
    s = noise[where_trace_finite]
    w = wvs[where_trace_finite]
    x = ra_array[where_trace_finite]
    y = dec_array[where_trace_finite]

    extra_outputs["wvs"] = w
    extra_outputs["ras"] = x
    extra_outputs["decs"] = y
    row_ids_im = np.tile(np.arange(ny)[:, None], (1, nx))
    extra_outputs["rows"] = row_ids_im[where_trace_finite]
    comp_spec = comp_spec[where_trace_finite]

    x_nodes, N_nodes = _create_nodes(cubeobj, nodes)

    # Number of linear parameters
    N_linpara = 1

    if np.size(where_trace_finite[0]) <= (1 - badpixfraction) * np.sum(new_mask * larger_mask_comp):
        # don't bother to do a fit if there are too many bad pixels
        print(f"WARNING: Too many bad pixels in the tested window dra={comp_dra_as} arcsec, ddec={comp_ddec_as} arcsec")
        return np.array([]), np.array([]).reshape(0, N_linpara), np.array([]), extra_outputs

    M_speckles = np.zeros((Nd, Nrows, N_nodes))
    if regularization == "user":
        d_reg_speckles = np.full((Nrows, N_nodes), np.nan)
        s_reg_speckles = np.full((Nrows, N_nodes), np.nan)
        wvs_reg_speckles = np.full((Nrows, N_nodes), np.nan)
        rows_reg_speckles = np.tile(rows_ids[:, None],(1, N_nodes)) #Transform row vector into column vector and then duplicates N_nodes times the column to have a 2d matrix
    if wvs_KLs_f is not None:
        M_KLs = np.zeros((Nd, Nrows, len(wvs_KLs_f)))
    if detec_KLs is not None:
        M_KLs_detec = np.zeros((Nd, Nrows, detec_KLs.shape[1]))

    for _k in range(Nrows):
        where_finite_and_in_row = np.where(where_trace_finite[0] == rows_ids[_k])
        if np.size(where_finite_and_in_row[0]) == 0:
            continue
        M_spline = get_spline_model(x_nodes, wvs[rows_ids[_k], :], spline_degree=3)
        selec_M_spline = M_spline[where_trace_finite[1][where_finite_and_in_row], :]
        where_del_col = np.where(np.nanmax(np.abs(selec_M_spline), axis=0) < min_spline_ampl)  # 0.01#0.00001

        if np.size(where_del_col[0]) == selec_M_spline.shape[1]:
            #Nothing usable in this row (rows_ids[_k])
            continue

        selec_M_spline[:, where_del_col[0]] = 0
        M_speckles[where_finite_and_in_row[0], _k, :] = selec_M_spline
        if regularization == "user":
            d_reg_speckles[_k, :] = reg_mean_map[rows_ids[_k], :]
            d_reg_speckles[_k, where_del_col[0]] = np.nan
            s_reg_speckles[_k, :] = reg_std_map[rows_ids[_k], :]
            s_reg_speckles[_k, where_del_col[0]] = np.nan
            wvs_reg_speckles[_k, :] = x_nodes
            wvs_reg_speckles[_k, where_del_col[0]] = np.nan

        if wvs_KLs_f is not None:
            for KLid, KL_f in enumerate(wvs_KLs_f):
                KL_vec = KL_f(wvs[rows_ids[_k], :])
                selec_KL_vec = KL_vec[where_trace_finite[1][where_finite_and_in_row]]
                M_KLs[where_finite_and_in_row[0], _k, KLid] = selec_KL_vec

        if detec_KLs is not None:
            selec_KL_vec = detec_KLs[where_trace_finite[1][where_finite_and_in_row], :]
            M_KLs_detec[where_finite_and_in_row[0], _k, :] = selec_KL_vec

    M_speckles = np.reshape(M_speckles, (Nd, Nrows * N_nodes))
    M_speckles = M_speckles * star_func(w)[:, None]

    if wvs_KLs_f is not None:
        M_KLs = np.reshape(M_KLs, (Nd, Nrows * len(wvs_KLs_f)))
    else:
        M_KLs = None

    if detec_KLs is not None:
        M_KLs_detec = np.reshape(M_KLs_detec, (Nd, Nrows * detec_KLs.shape[1]))
    else:
        M_KLs_detec = None

    comp_model = cubeobj.webbpsf_interp((x - comp_dra_as) * cubeobj.webbpsf_wv0 / w,
                                        (y - comp_ddec_as) * cubeobj.webbpsf_wv0 / w) * comp_spec #multiply off axis PSF by the companion spectrum

    # combine planet model with speckle model
    M = _concatenate_matrices(M_speckles, comp_model, wvs_KLs_f, detec_KLs, M_KLs, M_KLs_detec)

    if regularization == "user":
        #Vectorize the 2d arrays for regularization
        s_reg_speckles = np.ravel(s_reg_speckles)
        d_reg_speckles = np.ravel(d_reg_speckles)
        wvs_reg_speckles = np.ravel(wvs_reg_speckles)
        rows_reg_speckles = np.ravel(rows_reg_speckles)

        s_reg = np.array([np.nan])
        d_reg = np.array([np.nan])
        wvs_reg = np.array([np.nan])
        rows_reg = np.array([np.nan])
        s_reg = np.concatenate([s_reg, s_reg_speckles])
        d_reg = np.concatenate([d_reg, d_reg_speckles])  #
        wvs_reg = np.concatenate([wvs_reg, wvs_reg_speckles])  #
        rows_reg = np.concatenate([rows_reg, rows_reg_speckles])  #
        if wvs_KLs_f is not None:
            filler = np.nan + np.zeros(M_KLs.shape[1])
            s_reg = np.concatenate([s_reg, filler])
            d_reg = np.concatenate([d_reg, filler])
            wvs_reg = np.concatenate([wvs_reg, filler])
            rows_reg = np.concatenate([rows_reg, filler])  #
        if detec_KLs is not None:
            filler = np.nan + np.zeros(M_KLs_detec.shape[1])
            s_reg = np.concatenate([s_reg, filler])
            d_reg = np.concatenate([d_reg, filler])
            wvs_reg = np.concatenate([wvs_reg, filler])
            rows_reg = np.concatenate([rows_reg, filler])  #
        extra_outputs["regularization"] = (d_reg, s_reg)
        extra_outputs["regularization_wvs"] = wvs_reg
        extra_outputs["regularization_rows"] = rows_reg


    extra_outputs["where_trace_finite"] = where_trace_finite

    return d, M, s, extra_outputs

def _interpolate_companion_spectrum(cubeobj, atm_grid, atm_grid_wvs, atm_paras, vsini, rv, wvs, pixarea):
    """Helper function to interpolate companion spectrum"""
    planet_model = atm_grid(atm_paras)[0]
    if cubeobj.data_unit != 'MJy':
        raise TypeError("cubeobj.data_unit must be 'MJy', try running the convert_MJy_per_sr_to_MJy preprocessing step before")

    if np.all(planet_model == 0):
        raise ValueError("Something wrong in planet spectrum, all values are 0")
    elif np.size(atm_grid_wvs) != np.size(planet_model):
        raise ValueError(
            f"Something wrong in planet spectrum, the wavelength grid size is {np.size(atm_grid_wvs)} while the planet model size is {np.size(planet_model)} ")
    else:
        if vsini != 0:
            spinbroad_model = pyasl.fastRotBroad(atm_grid_wvs, planet_model, 0.1, vsini)
        else:
            spinbroad_model = planet_model

        planet_f = interp1d(atm_grid_wvs, spinbroad_model, bounds_error=False, fill_value=np.nan)

        comp_spec = planet_f(wvs * (1 - (rv - cubeobj.bary_RV) / const.c.to('km/s').value)) * (u.W / u.m ** 2 / u.um)
        comp_spec = comp_spec * pixarea / cubeobj.webbpsf_spaxel_area  # normalized to peak flux
        comp_spec = comp_spec * (wvs * u.um) ** 2 / const.c  # from  Flambda to Fnu
        comp_spec = comp_spec.to(u.MJy).value

    return comp_spec

def _extract_companion_traces(ra_array, dec_array, comp_dra_as, comp_ddec_as, radius_as, nx, data, bad_pixels, noise, comp_spec, star_func, Nrows_max, wvs):
    # extract planet trace in the detector 2D space
    dist2comp_as = np.sqrt((ra_array - comp_dra_as) ** 2 + (dec_array - comp_ddec_as) ** 2)

    mask_comp = dist2comp_as < radius_as
    larger_mask_comp = dist2comp_as < 3 * radius_as
    mask_vec = np.nansum(mask_comp, axis=1) != 0
    rows_ids = np.where(mask_vec)[0]

    new_mask = np.tile(mask_vec[:, None], (1, nx))

    finite_mask = (
        np.isfinite(data)
        & np.isfinite(bad_pixels)
        & np.isfinite(comp_spec)
        & np.isfinite(star_func(wvs))
    )

    valid_mask = (
        finite_mask
        & new_mask
        & larger_mask_comp
        & (noise != 0)
    )

    where_trace_finite = np.where(valid_mask)
    Nrows = np.size(rows_ids)
    Nd = np.size(where_trace_finite[0])
    if Nrows > Nrows_max:
        raise Exception("Too many rows")

    return where_trace_finite, Nd, new_mask, larger_mask_comp, Nrows, rows_ids

def _create_nodes(cubeobj, nodes):
    """Manage all the different cases to define the position of the spline nodes"""
    if type(nodes) is int:
        N_nodes = nodes
        min_wv, max_wv = np.nanmin(cubeobj.wavelengths), np.nanmax(cubeobj.wavelengths)
        x_nodes = np.linspace(min_wv, max_wv, N_nodes, endpoint=True).tolist()
    elif type(nodes) is list or type(nodes) is np.ndarray:
        x_nodes = nodes
        if type(nodes[0]) is list or type(nodes[0]) is np.ndarray:
            N_nodes = np.sum([np.size(n) for n in nodes])
        else:
            N_nodes = np.size(nodes)
    else:
        raise ValueError("Unknown format for nodes.")

    return x_nodes, N_nodes

def _concatenate_matrices(M_speckles, comp_model, wvs_KLs_f, detec_KLs, M_KLs, M_KLs_detec):
    M = np.concatenate([comp_model[:, None], M_speckles], axis=1)

    if wvs_KLs_f is not None:
        M = np.concatenate([M, M_KLs], axis=1)
    if detec_KLs is not None:
        M = np.concatenate([M, M_KLs_detec], axis=1)

    return M