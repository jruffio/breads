import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline
from astropy import constants as const

def get_spline_model(x_knots, x_samples, spline_degree=3):
    """
    Compute a spline based linear model.
    If Y=[y1,y2,..] are the values of the function at the location of the node [x1,x2,...].
    np.dot(M,Y) is the interpolated spline corresponding to the sampling of the x-axis (x_samples)


    Args:
        x_knots: List of nodes for the spline interpolation as np.ndarray in the same units as x_samples.
            x_knots can also be a list of ndarrays to model discontinous functions.
        x_samples: Vector of x values. ie, the sampling of the data.
        spline_degree: Degree of the spline interpolation (default: 3)

    Returns:
        M: Matrix of size (D,N) with D the size of x_samples and N the total number of nodes.
    """
    if type(x_knots[0]) is list:
        x_knots_list = x_knots
    else:
        x_knots_list = [x_knots]
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


def pixgauss2d(p, shape, hdfactor=10, xhdgrid=None, yhdgrid=None):
    """
    2d gaussian model. Documentation to be completed. Also faint of t
    """
    A, xA, yA, w, bkg = p
    ny, nx = shape
    if xhdgrid is None or yhdgrid is None:
        xhdgrid, yhdgrid = np.meshgrid(np.arange(hdfactor * nx).astype(np.float) / hdfactor,
                                       np.arange(hdfactor * ny).astype(np.float) / hdfactor)
    else:
        hdfactor = xhdgrid.shape[0] // ny
    gaussA_hd = A / (2 * np.pi * w ** 2) * np.exp(
        -0.5 * ((xA - xhdgrid + 0.5) ** 2 + (yA - yhdgrid + 0.5) ** 2) / w ** 2)
    gaussA = np.nanmean(np.reshape(gaussA_hd, (ny, hdfactor, nx, hdfactor)), axis=(1, 3))
    return gaussA + bkg


def splinefm(nonlin_paras, cubeobj, planet_f=None, transmission=None, star_spectrum=None,boxw=1, psfw=1.2,nodes=20,badpixfraction=0.75):
    """
    Generate forward model fitting the continuum with a spline. No high pass filter or continuum normalization here.
    The spline are defined with a linear model. Each spaxel (if applicable) is independently modeled which means the
    number of linear parameters increases as nodes*boxw^2+1.

    Args:
        nonlin_paras: [rv,y,x], Non-linear parameters of the model, which are the radial velocity and the position of
            the planet in the FOV.
        cubeobj: Data object.
            Must inherit breads.instruments.instrument.Instrument.
        planet_f: Planet atmospheric model spectrum as an interp1d object. Wavelength in microns.
        transmission: Transmission spectrum (tellurics and instrumental).
            np.ndarray of size the number of wavelength bins.
        star_spectrum: Stellar spectrum to be continuum renormalized to fit the speckle noise at each location.
            np.ndarray of size the number of wavelength bins.
        boxw: size of the stamp to be extracted and modeled around the (x,y) location of the planet.
            Must be odd. Default is 1.
        psfw: Width (sigma) of the 2d gaussian used to model the planet PSF.
        nodes: If int, number of nodes equally distributed. If list, custom locations of nodes [x1,x2,..].
            To model discontinous functions, use a list of list [[x1,...],[xn,...]].
        badpixfraction: Max fraction of bad pixels in data.

    Returns:
    """
    rv,y,x = nonlin_paras
    nz, ny, nx = cubeobj.data.shape
    wvs = cubeobj.wavelengths
    if boxw % 2 == 0:
        raise ValueError("boxw, the width of stamp around the planet, must be odd in splinefm().")
    if boxw > ny or boxw > nx:
        raise ValueError("boxw cannot be bigger than the data in splinefm().")

    # manage all the different cases to define the position of the spline nodes
    if type(nodes) is int:
        N_nodes = nodes
        x_knots = np.linspace(wvs[0], wvs[-1], N_nodes, endpoint=True).tolist()
    elif type(nodes) is list:
        x_knots = nodes
        if type(nodes[0]) is list:
            N_nodes = np.sum([np.size(n) for n in nodes])
        else:
            N_nodes = len(nodes)
    else:
        raise ValueError("Unknown format for nodes.")
    N_linpara = boxw * boxw * N_nodes +1


    # Extract stamp data cube cropping at the edges
    w = int((boxw - 1) // 2)
    # right, left  = np.min([l+w+1,nx]), np.max([l-w,0])
    # top, bottom = np.min([k+w+1,ny]), np.max([k-w,0])
    _paddata =np.pad(cubeobj.data,[(0,0),(w,w),(w,w)],mode="constant",constant_values = np.nan)
    _padnoise =np.pad(cubeobj.noise,[(0,0),(w,w),(w,w)],mode="constant",constant_values = np.nan)
    _padbad_pixels =np.pad(cubeobj.bad_pixels,[(0,0),(w,w),(w,w)],mode="constant",constant_values = np.nan)
    k, l = int(np.round(cubeobj.refpos[1] + y)), int(np.round(cubeobj.refpos[0] + x))
    dx,dy = x-l,y-k
    k,l = k+w,l+w
    d = np.ravel(_paddata[:, k-w:k+w+1, l-w:l+w+1])
    s = np.ravel(_padnoise[:, k-w:k+w+1, l-w:l+w+1])
    badpixs = np.ravel(_padbad_pixels[:, k-w:k+w+1, l-w:l+w+1])

    where_finite = np.where(np.isfinite(badpixs))
    if np.size(where_finite[0]) <= (1-badpixfraction) * np.size(d):
        # don't bother to do a fit if there are too many bad pixels
        return np.array([]), np.array([]).reshape(0,N_linpara), np.array([])
    else:
        # Get the linear model (ie the matrix) for the spline
        M_spline = get_spline_model(x_knots, wvs, spline_degree=3)
        M_speckles = np.zeros((nz, boxw, boxw, boxw, boxw, N_nodes))
        for m in range(boxw):
            for n in range(boxw):
                M_speckles[:, m, n, m, n, :] = M_spline * star_spectrum[:, None]
        M_speckles = np.reshape(M_speckles, (nz, boxw, boxw, N_linpara-1))


        psfs = np.zeros((nz, boxw, boxw))
        # Technically allows super sampled PSF to account for a true 2d gaussian integration of the area of a pixel.
        # But this is disabled for now with hdfactor=1.
        hdfactor = 1#5
        xhdgrid, yhdgrid = np.meshgrid(np.arange(hdfactor * (boxw)).astype(np.float) / hdfactor,
                                       np.arange(hdfactor * (boxw)).astype(np.float) / hdfactor)
        psfs += pixgauss2d([1., w+dx, w+dy, psfw, 0.], (boxw, boxw), xhdgrid=xhdgrid, yhdgrid=yhdgrid)[None, :, :]
        psfs = psfs / np.nansum(psfs, axis=(1, 2))[:, None, None]

        # The planet spectrum model is RV shifted and multiplied by the tranmission
        planet_spec = transmission * planet_f(wvs * (1 - (rv - cubeobj.bary_RV) / const.c.to('km/s').value))
        # Go from a 1d spectrum to the 3D scaled PSF
        scaled_psfs = psfs * planet_spec[:, None, None]

        # combine planet model with speckle model
        M = np.concatenate([scaled_psfs[:, :, :, None], M_speckles], axis=3)
        # Ravel data dimension
        M = np.reshape(M, (nz * boxw * boxw, N_linpara))
        # Get rid of bad pixels
        sr = s[where_finite]
        dr = d[where_finite]
        Mr = M[where_finite[0], :]

        return dr, Mr, sr