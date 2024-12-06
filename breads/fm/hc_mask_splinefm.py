import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy import constants as const
from copy import deepcopy
from breads.utils import get_spline_model
from copy import copy
from scipy.optimize import lsq_linear

def pixgauss2d(p, shape, hdfactor=10, xhdgrid=None, yhdgrid=None):
    """
    2d gaussian model. Documentation to be completed. Also faint of t
    """
    A, xA, yA, sigx, sigy, bkg = p
    ny, nx = shape
    if xhdgrid is None or yhdgrid is None:
        xhdgrid, yhdgrid = np.meshgrid(np.arange(hdfactor * nx).astype(np.float) / hdfactor,
                                       np.arange(hdfactor * ny).astype(np.float) / hdfactor)
    else:
        hdfactor = xhdgrid.shape[0] // ny
    gaussA_hd = A / (2 * np.pi * sigx * sigy) * np.exp(
        -0.5 * ((xA - xhdgrid) ** 2 / (sigx ** 2) + (yA - yhdgrid) ** 2 / (sigy ** 2)))
    gaussA = np.nanmean(np.reshape(gaussA_hd, (ny, hdfactor, nx, hdfactor)), axis=(1, 3))
    return gaussA + bkg

def set_nodes(cont_stamp, noise_stamp, wavelengths, nodes, optimize_nodes, p, wid_mov=None, knot_margin=1e-4):
    data_f = np.divide(np.nanmedian(cont_stamp, axis=(1,2)), np.nanmedian(noise_stamp, axis=(1,2)))
    wvs = np.nanmedian(wavelengths, axis=(1,2))
    if wavelengths.size == 0:
        gmin, gmax = np.nan, np.nan
    else:
        gmin, gmax = np.nanmin(wavelengths), np.nanmax(wavelengths)
    if wid_mov is None:
        wid_mov = len(cont_stamp[0]) // 20
    if type(nodes) is int:
        if optimize_nodes:
            N_nodes = nodes
            grad = np.abs(np.gradient(data_f))
            ddata = np.nancumsum(data_f)
            minv, maxv = np.nanmin(ddata), np.nanmax(ddata)
            parts = minv + (maxv-minv) / (N_nodes-1) * np.arange(N_nodes)
            # x_knots = np.append(wvs[([np.searchsorted(ddata, part) for part in parts])], np.nanmax(wvs))
            args = [np.searchsorted(ddata, part)-1 for part in parts]
            args[-1], args[0] = len(ddata) - 1, 0
            x_pos = np.nanmedian(wavelengths[args], axis=(1, 2))
            x_pos[0], x_pos[-1] = gmin, gmax
            # print([np.searchsorted(ddata, part) for part in parts])
            # x_knots[0], x_knots[-1] = wvs[0], wvs[-1]
            lin_x = np.linspace(gmin, gmax, N_nodes, endpoint=True)
            # p = 1
            x_knots = p * x_pos + (1-p) * lin_x
            x_knots[0], x_knots[-1] = gmin - knot_margin, gmax + knot_margin
            if False:
                print(parts)
                print(args)
                print(x_knots)
                print(ddata[args])
                print(len(wvs), minv, maxv, wvs[0], wvs[-1], np.nanmin(wvs), np.nanmax(wvs), np.nanmin(wavelengths), np.nanmax(wavelengths))
                # plt.figure()
                # plt.plot(wvs, data_f)
                # plt.figure()
                # plt.plot(wvs, grad)
                plt.figure()
                plt.plot(wvs, ddata)
                plt.plot(x_pos, np.zeros_like(x_pos), "rX")
                plt.plot(lin_x, np.zeros_like(lin_x) + 0.5, "bX")
                plt.plot(x_knots, np.ones_like(x_knots), "gX")
                plt.figure()
                plt.subplot(2, 1, 1)
                for x_knot in x_knots:
                    plt.axvline(x_knot, linestyle=":", color="black")
                plt.grid()
                # plt.show()
                # exit()
        else:
            if wavelengths.size == 0:
                gmin, gmax = np.nan, np.nan
            else:
                gmin, gmax = np.nanmin(wavelengths), np.nanmax(wavelengths)
            N_nodes = nodes
            x_knots = np.linspace(gmin, gmax, N_nodes, endpoint=True).tolist()
    elif type(nodes) is list or type(nodes) is np.ndarray :
        x_knots = nodes
        if type(nodes[0]) is list or type(nodes[0]) is np.ndarray :
            N_nodes = np.sum([np.size(n) for n in nodes])
        else:
            N_nodes = np.size(nodes)
    else:
        raise ValueError("Unknown format for nodes.")
    return N_nodes, x_knots

def hc_mask_splinefm(nonlin_paras, cubeobj, stamp=None, planet_f=None, transmission=None, star_spectrum=None, boxw=1, psfw=1.2,nodes=20, star_flux=None,
                badpixfraction=0.75,loc=None, optimize_nodes=True, wid_mov=None, opt_p=0.7, knot_margin=1e-4, star_loc=None,
                     KLmodes=None,fit_background=False,recalc_noise=True,just_tellurics=False):
    """
    For high-contrast companions (planet + speckles).
    Generate forward model fitting the continuum with a spline. No high pass filter or continuum normalization here.
    The spline are defined with a linear model. Each spaxel (if applicable) is independently modeled which means the
    number of linear parameters increases as N_nodes*boxw^2+1.

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
        loc: (x,y) position of the planet for spectral cubes, or fiber position (y position) for 2d data.
            When loc is not None, the x,y non-linear parameters should not be given.

    Returns:
        d: Data as a 1d vector with bad pixels removed (no nans)
        M: Linear model as a matrix of shape (Nd,Np) with bad pixels removed (no nans). Nd is the size of the data
            vector and Np = N_nodes*boxw^2+1 is the number of linear parameters.
        s: Noise vector (standard deviation) as a 1d vector matching d.
    """

    try:
        sigx, sigy = psfw
    except:
        sigx, sigy = psfw, psfw

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

    rv = nonlin_paras[0]
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
            x,y = 0,nonlin_paras[1]
        elif len(cubeobj.data.shape)==3:
            x,y = nonlin_paras[2],nonlin_paras[1]

    nz, ny, nx = data.shape

    if len(cubeobj.wavelengths.shape)==1:
        wvs = cubeobj.wavelengths[:,None,None]
    elif len(cubeobj.wavelengths.shape)==2:
        wvs = cubeobj.wavelengths[:,:,None]
    elif len(cubeobj.wavelengths.shape)==3:
        wvs = cubeobj.wavelengths
    _, nywv, nxwv = wvs.shape

    k, l = int(np.round(refpos[1] + y)), int(np.round(refpos[0] + x))

    if boxw % 2 == 0:
        raise ValueError("boxw, the width of stamp around the planet, must be odd in splinefm().")
    if boxw > ny or boxw > nx:
        raise ValueError("boxw cannot be bigger than the data in splinefm().")

    # Extract stamp data cube cropping at the edges
    w = int((boxw - 1) // 2)

    dx,dy = x-l+refpos[0],y-k+refpos[1]
    padk,padl = k+w,l+w

    if star_spectrum is None:
        assert star_loc is not None, "both star_spectrum and star_loc cannot be None"
        sy, sx = star_loc
        # aper_s, mask_sx, mask_sy = 5.0, int(2.0*sigx)+1, int(2.0*sigy)+1
        aper_s, mask_sx, mask_sy = 5.0, w, w
        if star_flux is None:
            star_flux = np.nanmean(data) * data.size
        data_cp = copy(data)
        data_cp[:, k-mask_sx:k+mask_sx+1, l-mask_sy:l+mask_sy+1] = np.nan
        star = data_cp[:, int(np.round(sx-aper_s*sigx)):int(np.round(sx+aper_s*sigx)), int(np.round(sy-aper_s*sigy)):int(np.round(sy+aper_s*sigy))]
        star_spectrum = np.nanmean(star, axis=(1, 2))
    else:
        # flux ratio normalization
        star_flux = np.nanmean(star_spectrum) * np.size(star_spectrum)

    star_spectrum = star_spectrum / (np.nanmean(star_spectrum) * star_spectrum.size)

    if np.size(star_spectrum.shape) == 3:
        bad_pixels[np.where(np.isnan(transmission))[0],:,:] = np.nan
        bad_pixels[np.isnan(star_spectrum)] = np.nan
    elif np.size(star_spectrum.shape) == 1:
        # remove pixels that are bad in the transmission or the star spectrum
        bad_pixels[np.where(np.isnan(star_spectrum*transmission))[0],:,:] = np.nan

    _paddata =np.pad(data,[(0,0),(w,w),(w,w)],mode="constant",constant_values = np.nan)
    _padnoise =np.pad(noise,[(0,0),(w,w),(w,w)],mode="constant",constant_values = np.nan)
    _padbad_pixels =np.pad(bad_pixels,[(0,0),(w,w),(w,w)],mode="constant",constant_values = np.nan)

    # high pass filter the data
    cube_stamp = _paddata[:, padk-w:padk+w+1, padl-w:padl+w+1]
    badpix_stamp = _padbad_pixels[:, padk-w:padk+w+1, padl-w:padl+w+1]
    badpixs = np.ravel(badpix_stamp)
    d = np.ravel(cube_stamp)
    s = np.ravel(_padnoise[:, padk-w:padk+w+1, padl-w:padl+w+1])
    badpixs[np.where(s==0)] = np.nan
    if np.size(star_spectrum.shape) == 3:
        _padref =np.pad(star_spectrum,[(0,0),(w,w),(w,w)],mode="constant",constant_values = np.nan)
        ref_stamp = _padref[:, padk-w:padk+w+1, padl-w:padl+w+1]

    # manage all the different cases to define the position of the spline nodes
    # print(k, l, x, y)
    N_nodes, x_knots = set_nodes(cubeobj.continuum[:, padk-w:padk+w+1, padl-w:padl+w+1], _padnoise[:, padk-w:padk+w+1, padl-w:padl+w+1], \
        wvs[:, padk-w:padk+w+1, padl-w:padl+w+1], nodes, optimize_nodes, opt_p, wid_mov, knot_margin)

    if fit_background:
        N_background_linpara = 3*boxw**2
    else:
        N_background_linpara = 0
    if KLmodes is not None:
        N_KLmodes = KLmodes.shape[1]*boxw**2
    else:
        N_KLmodes = 0
    if just_tellurics:
        N_just_tell = 1
    else:
        N_just_tell = 0
    N_linpara = boxw * boxw * N_nodes +1 + N_background_linpara + N_KLmodes+N_just_tell

    # plt.plot(wvs[:, padk-w:padk+w+1, padl-w:padl+w+1][:, 0, 0], d/np.nanmax(d), label="data")
    # plt.legend()

    where_finite = np.where(np.isfinite(badpixs))
    if np.size(where_finite[0]) <= (1-badpixfraction) * np.size(badpixs) or \
            padk >= ny+2*w-1 or padk < 0 or padl >= nx+2*w-1 or padl < 0:
        # don't bother to do a fit if there are too many bad pixels
        return np.array([]), np.array([]).reshape(0,N_linpara), np.array([])
    else:
        if KLmodes is not None:
            # Get the linear model (ie the matrix) for the KL modes
            M_KLmodes = np.zeros((nz, boxw, boxw, boxw, boxw, KLmodes.shape[1]))
            for _k in range(boxw):
                for _l in range(boxw):
                    M_KLmodes[:, _k, _l, _k, _l, :] = KLmodes
            M_KLmodes = np.reshape(M_KLmodes, (nz, boxw, boxw, boxw * boxw * KLmodes.shape[1]))
        else:
            M_KLmodes = np.tile(np.array([])[None,None,None,:],(nz, boxw, boxw,0))

        # Get the linear model (ie the matrix) for the spline
        M_speckles = np.zeros((nz, boxw, boxw, boxw, boxw, N_nodes))
        for _k in range(boxw):
            for _l in range(boxw):
                lwvs = wvs[:,np.clip(k-w+_k,0,nywv-1),np.clip(l-w+_l,0,nxwv-1)]
                M_spline = get_spline_model(x_knots, lwvs, spline_degree=3)
                # print(M_spline.shape, "Mspline")
                # plt.subplot(2, 1, 2)
                # plt.plot(star_spectrum/np.nanmax(star_spectrum), label="star-spectrum")
                # for k in range(M_spline.shape[-1]):
                #     print(k)
                #     plt.plot(M_spline[:,k],label="spline {0}".format(k+1))
                # plt.legend()
                # plt.grid()
                # plt.xlabel("wavelength/index")
                # plt.savefig("./plots/TEMP3.png")
                if np.size(star_spectrum.shape) == 3:
                    M_speckles[:, _k, _l, _k, _l, :] = M_spline * ref_stamp[:,_k, _l, None]
                elif np.size(star_spectrum.shape) == 1:
                    M_speckles[:, _k, _l, _k, _l, :] = M_spline * star_spectrum[:, None]
        M_speckles = np.reshape(M_speckles, (nz, boxw, boxw, boxw * boxw * N_nodes))

        if fit_background:
            M_background = np.zeros((nz, boxw, boxw, boxw, boxw,3))
            for _k in range(boxw):
                for _l in range(boxw):
                    lwvs = wvs[:,np.clip(k-w+_k,0,nywv-1),np.clip(l-w+_l,0,nxwv-1)]
                    M_background[:, _k, _l, _k, _l, 0] = 1
                    M_background[:, _k, _l, _k, _l, 1] = lwvs
                    M_background[:, _k, _l, _k, _l, 2] = lwvs**2
            M_background = np.reshape(M_background, (nz, boxw, boxw, 3*boxw**2))
        else:
            M_background =  np.tile(np.array([])[None,None,None,:],(nz, boxw, boxw,0))

        if stamp is None:
            psfs = np.zeros((nz, boxw, boxw))
            # Technically allows super sampled PSF to account for a true 2d gaussian integration of the area of a pixel.
            # But this is disabled for now with hdfactor=1.
            hdfactor = 1#5
            xhdgrid, yhdgrid = np.meshgrid(np.arange(hdfactor * (boxw)).astype(np.float) / hdfactor,
                                        np.arange(hdfactor * (boxw)).astype(np.float) / hdfactor)
            psfs += pixgauss2d([1., w+dx, w+dy, sigx, sigy, 0.], (boxw, boxw), xhdgrid=xhdgrid, yhdgrid=yhdgrid)[None, :, :]
        else:
            _, sy, sx = stamp.shape
            assert (sy == sx == boxw), "stamp should be of shape (nz, boxw, boxw)"
            psfs = stamp

        psfs = psfs / np.nansum(psfs, axis=(1, 2))[:, None, None]

        scaled_psfs = np.zeros((nz,boxw,boxw))+np.nan
        if just_tellurics:
            just_tellurics_psfs = np.zeros((nz,boxw,boxw,1))+np.nan
        else:
            just_tellurics_psfs =  np.tile(np.array([])[None,None,None,:],(nz, boxw, boxw,0))
        for _k in range(boxw):
            for _l in range(boxw):
                lwvs = wvs[:,np.clip(k-w+_k,0,nywv-1),np.clip(l-w+_l,0,nxwv-1)]
                # The planet spectrum model is RV shifted and multiplied by the tranmission
                planet_spec = transmission * planet_f(lwvs * (1 - (rv - cubeobj.bary_RV) / const.c.to('km/s').value))
                scaled_psfs[:,_k,_l] = psfs[:, _k,_l] * planet_spec
                if just_tellurics:
                    just_tellurics_psfs[:,_k,_l,0] = psfs[:, _k,_l] * transmission

        planet_flux = np.size(scaled_psfs) * np.nanmean(scaled_psfs)
        scaled_psfs = scaled_psfs / planet_flux * star_flux
        # print(np.nansum(scaled_psfs))

        # combine planet model with speckle model
        M = np.concatenate([scaled_psfs[:, :, :, None], just_tellurics_psfs,M_speckles,M_KLmodes,M_background], axis=3)
        # Ravel data dimension
        M = np.reshape(M, (nz * boxw * boxw, N_linpara))
        # Get rid of bad pixels
        sr = s[where_finite]
        dr = d[where_finite]
        Mr = M[where_finite[0], :]

        if recalc_noise:
            d,M,s = dr, Mr, sr

            _bounds = ([-np.inf,]*N_linpara,[np.inf,]*N_linpara)

            validpara = np.where(np.nansum(M,axis=0)!=0)
            _bounds = (np.array(_bounds[0])[validpara[0]],np.array(_bounds[1])[validpara[0]])
            M = M[:,validpara[0]]

            d = d / s
            M = M / s[:, None]

            N_data = np.size(d)
            if N_data == 0 or 0 not in validpara[0]:
                pass
            else:
                paras = lsq_linear(M, d,bounds=_bounds).x
                m = np.dot(M, paras)
                r = d  - m
                chi2 = np.nansum(r**2)
                rchi2 = chi2 / N_data
                canvas_res = np.zeros(badpix_stamp.shape)+np.nan
                canvas_res[np.where(np.isfinite(badpix_stamp))] = r*s
                new_noise_stamp = np.tile(np.nanstd(canvas_res,axis=0)[None,:,:],(nz,1,1))
                sr = np.ravel(new_noise_stamp)[where_finite]

        return dr, Mr, sr
