import numpy as np
import scipy.ndimage as ndi
from astropy import constants as const
from scipy.interpolate import interp1d

from breads.utils import get_spline_model, scale_psg, pixgauss2d



def hc_kpicrvfm(nonlin_paras, cubeobj, planet_f=None, transmission=None, star_spectrum=None,boxw=1, psfw=1.2,nodes=20,nodes_pl=None,
                badpixfraction=0.75,loc=None,fix_parameters=None,telluric_wvs=None,psg_tuple=None,fit_background=False):
    """
    Measuring RV of a companion for KPIC.
    For high-contrast companions (planet + speckles).
    Generate forward model fitting the continuum with a spline. No high pass filter or continuum normalization here.
    The spline are defined with a linear model.

    Args:
        nonlin_paras: Non-linear parameters of the model, which are the radial velocity and the position (if loc is not
            defined) of the planet in the FOV.
            [rv,y,x] for 3d cubes (e.g. OSIRIS)
            [rv,y] for 2d (e.g. KPIC, y being fiber)
            [rv] for 1d spectra
        cubeobj: Data object.
            Must inherit breads.instruments.instrument.Instrument.
        planet_f: Planet atmospheric model spectrum as an interp1d object. Wavelength in microns.
        transmission: Transmission spectrum (tellurics and instrumental).
            np.ndarray of size the number of wavelength bins.
        star_spectrum: Stellar spectrum to be continuum renormalized to fit the speckle noise at each location. It is
            (for now) assumed to be the same everywhere which is not compatible with a field dependent wavelength solution.
            np.ndarray of size the number of wavelength bins.
        boxw: size of the stamp to be extracted and modeled around the (x,y) location of the planet.
            Must be odd. Default is 1.
        psfw: Width (sigma) of the 2d gaussian used to model the planet PSF. This won't matter if boxw=1 however.
        nodes: If int, number of nodes equally distributed. If list, custom locations of nodes [x1,x2,..].
            To model discontinous functions, use a list of list [[x1,...],[xn,...]].
        badpixfraction: Max fraction of bad pixels in data.
        loc: Deprecated, Use fix_parameters.
            (x,y) position of the planet for spectral cubes, or fiber position (y position) for 2d data.
            When loc is not None, the x,y non-linear parameters should not be given.
        fix_parameters: List. Use to fix the value of some non-linear parameters. The values equal to None are being
                    fitted for, other elements will be fixed to the value specified.

    Returns:
        d: Data as a 1d vector with bad pixels removed (no nans)
        M: Linear model as a matrix of shape (Nd,Np) with bad pixels removed (no nans). Nd is the size of the data
            vector and Np = N_nodes*boxw^2+1 is the number of linear parameters.
        s: Noise vector (standard deviation) as a 1d vector matching d.
    """
    if fix_parameters is not None:
        _nonlin_paras = np.array(fix_parameters)
        _nonlin_paras[np.where(np.array(fix_parameters)==None)] = nonlin_paras
    else:
        _nonlin_paras = nonlin_paras

    # Handle the different data dimensions
    # Convert everything to 3D cubes (wv,y,x) for the followying
    if len(cubeobj.data.shape)==1:
        data = cubeobj.data[:,None,None]
        noise = cubeobj.noise[:,None,None]
        bad_pixels = cubeobj.bad_pixels[:,None,None]
    elif len(cubeobj.data.shape)==2:
        data = cubeobj.data[:,:,None]
        noise = cubeobj.noise[:,:,None]
        bad_pixels = cubeobj.bad_pixels[:,:,None]
    elif len(cubeobj.data.shape)==3:
        data = cubeobj.data
        noise = cubeobj.noise
        bad_pixels = cubeobj.bad_pixels
    if cubeobj.refpos is None:
        refpos = [0,0]
    else:
        refpos = cubeobj.refpos

    N_nonlinparas = 1 + (len(cubeobj.data.shape)-1)
    rv = _nonlin_paras[0]
    if psg_tuple is not None:
        N_nonlinparas += 2
        airmass, pwv = _nonlin_paras[1],_nonlin_paras[2]
    # Defining the position of companion
    # If loc is not defined, then the x,y position is assume to be a non linear parameter.
    if np.size(loc) ==2:
        x,y = loc
    elif np.size(loc) ==1 and loc is not None:
        x,y = 0,loc
    elif loc is None:
        if len(cubeobj.data.shape)==1:
            x,y = 0,0
        elif len(cubeobj.data.shape)==2:
            x,y = 0,_nonlin_paras[N_nonlinparas-1]
        elif len(cubeobj.data.shape)==3:
            x,y = _nonlin_paras[N_nonlinparas-1],_nonlin_paras[N_nonlinparas-2]

    nz, ny, nx = data.shape

    if len(cubeobj.wavelengths.shape)==1:
        wvs = cubeobj.wavelengths[:,None,None]
    elif len(cubeobj.wavelengths.shape)==2:
        wvs = cubeobj.wavelengths[:,:,None]
    elif len(cubeobj.wavelengths.shape)==3:
        wvs = cubeobj.wavelengths
    _, nywv, nxwv = wvs.shape

    if boxw % 2 == 0:
        raise ValueError("boxw, the width of stamp around the planet, must be odd in splinefm().")
    if boxw > ny or boxw > nx:
        raise ValueError("boxw cannot be bigger than the data in splinefm().")

    # remove pixels that are bad in the transmission or the star spectrum
    bad_pixels[np.where(np.isnan(star_spectrum*transmission))[0],:,:] = np.nan

    # Extract stamp data cube cropping at the edges
    w = int((boxw - 1) // 2)

    _paddata =np.pad(data,[(0,0),(w,w),(w,w)],mode="constant",constant_values = np.nan)
    _padnoise =np.pad(noise,[(0,0),(w,w),(w,w)],mode="constant",constant_values = np.nan)
    _padbad_pixels =np.pad(bad_pixels,[(0,0),(w,w),(w,w)],mode="constant",constant_values = np.nan)
    k, l = int(np.round(refpos[1] + y)), int(np.round(refpos[0] + x))
    dx,dy = x-l+refpos[0],y-k+refpos[1]
    padk,padl = k+w,l+w

    # high pass filter the data
    cube_stamp = _paddata[:, padk-w:padk+w+1, padl-w:padl+w+1]
    badpix_stamp = _padbad_pixels[:, padk-w:padk+w+1, padl-w:padl+w+1]
    badpixs = np.ravel(badpix_stamp)
    d = np.ravel(cube_stamp)
    s = np.ravel(_padnoise[:, padk-w:padk+w+1, padl-w:padl+w+1])
    badpixs[np.where(s==0)] = np.nan

    # manage all the different cases to define the position of the spline nodes
    if type(nodes) is int:
        N_nodes = nodes
        x_knots = np.linspace(np.min(wvs), np.max(wvs), N_nodes, endpoint=True).tolist()
    elif type(nodes) is list  or type(nodes) is np.ndarray :
        x_knots = nodes
        if type(nodes[0]) is list or type(nodes[0]) is np.ndarray :
            N_nodes = np.sum([np.size(n) for n in nodes])
        else:
            N_nodes = np.size(nodes)
    else:
        raise ValueError("Unknown format for nodes.")

    if nodes_pl is None:
        N_nodes_pl= 1
    elif type(nodes_pl) is int:
        N_nodes_pl = nodes_pl
        x_knots_pl = np.linspace(np.min(wvs), np.max(wvs), N_nodes_pl, endpoint=True).tolist()
    elif type(nodes_pl) is list  or type(nodes_pl) is np.ndarray :
        x_knots_pl = nodes_pl
        if type(nodes_pl[0]) is list or type(nodes_pl[0]) is np.ndarray :
            N_nodes_pl = np.sum([np.size(n) for n in nodes_pl])
        else:
            N_nodes_pl = np.size(nodes_pl)
    else:
        raise ValueError("Unknown format for nodes_pl.")

    if fit_background:
        N_linpara = boxw * boxw * N_nodes +1*N_nodes_pl + 3*boxw**2
    else:
        N_linpara = boxw * boxw * N_nodes +1*N_nodes_pl

    where_finite = np.where(np.isfinite(badpixs))
    if np.size(where_finite[0]) <= (1-badpixfraction) * np.size(badpixs) or \
            padk > ny+2*w-1 or padk < 0 or padl > nx+2*w-1 or padl < 0:
        # don't bother to do a fit if there are too many bad pixels
        return np.array([]), np.array([]).reshape(0,N_linpara), np.array([])
    else:

        if psg_tuple is not None:
            reduc_factor = 40
            _telluric_wvs = telluric_wvs[0:(np.size(telluric_wvs) // reduc_factor) * reduc_factor]
            _telluric_wvs = np.mean(np.reshape(_telluric_wvs, ((np.size(_telluric_wvs) // reduc_factor), reduc_factor)), axis=1)

            telluric_spec = scale_psg(psg_tuple, airmass, pwv)

            telluric_spec = telluric_spec[0:(np.size(telluric_spec) // reduc_factor) * reduc_factor]
            telluric_spec = np.mean(np.reshape(telluric_spec, ((np.size(telluric_spec) // reduc_factor), reduc_factor)), axis=1)
            where_order_wvs = np.where((np.nanmin(wvs)<_telluric_wvs)*(_telluric_wvs<np.nanmax(wvs)))
            _telluric_wvs = _telluric_wvs[where_order_wvs]
            telluric_spec = telluric_spec[where_order_wvs]
            star_model_r = np.median(_telluric_wvs) / np.median(_telluric_wvs - np.roll(_telluric_wvs, 1))
            star_model_downsample = star_model_r / 35000 / (2 * np.sqrt(2 * np.log(2)))
            conv_telluric_spec = ndi.gaussian_filter(telluric_spec, star_model_downsample)
            add_tel_func = interp1d(_telluric_wvs,conv_telluric_spec,bounds_error=False,fill_value=0)

            # import matplotlib.pyplot as plt
            # plt.plot(_telluric_wvs,conv_telluric_spec)
            # plt.show()

        # Get the linear model (ie the matrix) for the spline
        M_speckles = np.zeros((nz, boxw, boxw, boxw, boxw, N_nodes))
        for _k in range(boxw):
            for _l in range(boxw):
                lwvs = wvs[:,np.clip(k-w+_k,0,nywv-1),np.clip(l-w+_l,0,nxwv-1)]
                M_spline = get_spline_model(x_knots, lwvs, spline_degree=3)
                # if psg_tuple is not None:
                #     M_speckles[:, _k, _l, _k, _l, :] = M_spline * (add_tel_func(lwvs) *star_spectrum)[:, None]
                # else:
                #     M_speckles[:, _k, _l, _k, _l, :] = M_spline * star_spectrum[:, None]
                M_speckles[:, _k, _l, _k, _l, :] = M_spline * star_spectrum[:, None]
        M_speckles = np.reshape(M_speckles, (nz, boxw, boxw, boxw * boxw * N_nodes))

        if fit_background:
            M_background = np.zeros((nz, boxw, boxw, boxw, boxw,3))
            for _k in range(boxw):
                for _l in range(boxw):
                    lwvs = wvs[:,np.clip(k-w+_k,0,nywv-1),np.clip(l-w+_l,0,nxwv-1)]
                    M_spline = get_spline_model(np.linspace(np.min(wvs), np.max(wvs), 3, endpoint=True).tolist(), lwvs, spline_degree=3)
                    M_background[:, _k, _l, _k, _l, :] = M_spline
                    # M_background[:, _k, _l, _k, _l, 0] = 1
                    # M_background[:, _k, _l, _k, _l, 1] = lwvs
                    # M_background[:, _k, _l, _k, _l, 2] = lwvs**2
            M_background = np.reshape(M_background, (nz, boxw, boxw, 3*boxw**2))

        psfs = np.zeros((nz, boxw, boxw))
        # Technically allows super sampled PSF to account for a true 2d gaussian integration of the area of a pixel.
        # But this is disabled for now with hdfactor=1.
        hdfactor = 1#5
        xhdgrid, yhdgrid = np.meshgrid(np.arange(hdfactor * (boxw)).astype(np.float) / hdfactor,
                                       np.arange(hdfactor * (boxw)).astype(np.float) / hdfactor)
        psfs += pixgauss2d([1., w+dx, w+dy, psfw, 0.], (boxw, boxw), xhdgrid=xhdgrid, yhdgrid=yhdgrid)[None, :, :]
        psfs = psfs / np.nansum(psfs, axis=(1, 2))[:, None, None]

        if nodes_pl is None:
            pass
            scaled_psfs = np.zeros((nz,boxw,boxw))+np.nan
            for _k in range(boxw):
                for _l in range(boxw):
                    lwvs = wvs[:,np.clip(k-w+_k,0,nywv-1),np.clip(l-w+_l,0,nxwv-1)]
                    # The planet spectrum model is RV shifted and multiplied by the tranmission
                    if psg_tuple is not None:
                        planet_spec = add_tel_func(lwvs) * transmission * planet_f(lwvs * (1 - (rv - cubeobj.bary_RV) / const.c.to('km/s').value))
                    else:
                        planet_spec = transmission * planet_f(lwvs * (1 - (rv - cubeobj.bary_RV) / const.c.to('km/s').value))
                    scaled_psfs[:,_k,_l] = psfs[:, _k,_l] * planet_spec

            # flux ratio normalization
            star_flux = np.nanmean(star_spectrum) * np.size(star_spectrum)
            planet_flux = np.size(scaled_psfs) * np.nanmean(scaled_psfs)
            scaled_psfs = scaled_psfs / planet_flux * star_flux
            # print(np.nansum(scaled_psfs))

            if fit_background:
                M = np.concatenate([scaled_psfs[:, :, :, None], M_speckles,M_background], axis=3)
            else:
                M = np.concatenate([scaled_psfs[:, :, :, None], M_speckles], axis=3)
        else:
            scaled_psfs = np.zeros((nz,boxw,boxw,N_nodes_pl))+np.nan
            for _k in range(boxw):
                for _l in range(boxw):
                    lwvs = wvs[:,np.clip(k-w+_k,0,nywv-1),np.clip(l-w+_l,0,nxwv-1)]
                    M_spline = get_spline_model(x_knots_pl, lwvs, spline_degree=3)
                    # The planet spectrum model is RV shifted and multiplied by the tranmission
                    if psg_tuple is not None:
                        planet_spec = add_tel_func(lwvs) * transmission * planet_f(lwvs * (1 - (rv - cubeobj.bary_RV) / const.c.to('km/s').value))
                    else:
                        planet_spec = transmission * planet_f(lwvs * (1 - (rv - cubeobj.bary_RV) / const.c.to('km/s').value))
                    scaled_psfs[:,_k,_l,:] = M_spline*psfs[:, _k,_l][:,None] * planet_spec[:,None]

            # flux ratio normalization
            star_flux = np.nanmean(star_spectrum) * np.size(star_spectrum)
            planet_flux = np.size(scaled_psfs) * np.nanmean(scaled_psfs)
            scaled_psfs = scaled_psfs / planet_flux * star_flux
            # print(np.nansum(scaled_psfs))

            if fit_background:
                M = np.concatenate([scaled_psfs, M_speckles, M_background], axis=3)
            else:
                M = np.concatenate([scaled_psfs, M_speckles], axis=3)

        # combine planet model with speckle model
        # Ravel data dimension
        M = np.reshape(M, (nz * boxw * boxw, N_linpara))
        # Get rid of bad pixels
        sr = s[where_finite]
        dr = d[where_finite]
        Mr = M[where_finite[0], :]

        return dr, Mr, sr