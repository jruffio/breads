import numpy as np
from copy import copy
import pandas as pd
from astropy import constants as const

from breads.utils import broaden
from breads.utils import LPFvsHPF



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


def iso_hpffm(nonlin_paras, cubeobj, planet_f=None, transmission=None,boxw=1, psfw=1.2,badpixfraction=0.75,
             hpf_mode=None,res_hpf=50,cutoff=5,fft_bounds=None,loc=None):
    """
    For isolated objects, so no speckle.
    Generate forward model removing the continuum with a fourier based high pass filter.

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
        boxw: size of the stamp to be extracted and modeled around the (x,y) location of the planet.
            Must be odd. Default is 1.
        psfw: Width (sigma) of the 2d gaussian used to model the planet PSF. This won't matter if boxw=1 however.
        badpixfraction: Max fraction of bad pixels in data.
        hpf_mode: choose type of high-pass filter to be used.
            "gauss": the data is broaden to the resolution specified by "res_hpf", which is then subtracted.
            "fft": a fft based high-pass filter is used using a cutoff frequency specified by "cutoff".
                This should not be used for (highly) non-uniform wavelength sampling or with gaps.
        res_hpf: float, if hpf_mode="gauss", resolution of the continuum to be subtracted.
        cutoff: int, if hpf_mode="fft", the higher the cutoff the more agressive the high pass filter.
            See breads.utils.LPFvsHPF().
        fft_bounds: [l1,l2,..ln] if hpf_mode is "fft", divide the spectrum into n chunks [l1,l2],..[..,ln] on which the
            fft high-pass filter is run separately.
        loc: (x,y) position of the planet for spectral cubes, or fiber position (y position) for 2d data.
            When loc is not None, the x,y non-linear parameters should not be given.

    Returns:
        d: Data as a 1d vector with bad pixels removed (no nans)
        M: Linear model as a matrix of shape (Nd,1) with bad pixels removed (no nans). Nd is the size of the data
            vector.
        s: Noise vector (standard deviation) as a 1d vector matching d.
    """
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
    if fft_bounds is None:
        fft_bounds = np.array([0,nz])

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


    # remove pixels that are bad in the transmission
    bad_pixels[np.where(np.isnan(transmission))[0],:,:] = np.nan

    # Extract stamp data cube cropping at the edges
    w = int((boxw - 1) // 2)
    # Number of linear parameters
    N_linpara = 1 # just the planet flux

    _paddata =np.pad(data,[(0,0),(w,w),(w,w)],mode="constant",constant_values = np.nan)
    _padnoise =np.pad(noise,[(0,0),(w,w),(w,w)],mode="constant",constant_values = np.nan)
    _padbad_pixels =np.pad(bad_pixels,[(0,0),(w,w),(w,w)],mode="constant",constant_values = np.nan)
    k, l = int(np.round(refpos[1] + y)), int(np.round(refpos[0] + x))
    dx,dy = x-l,y-k
    padk,padl = k+w,l+w

    # high pass filter the data
    cube_stamp = _paddata[:, padk-w:padk+w+1, padl-w:padl+w+1]
    badpix_stamp = _padbad_pixels[:, padk-w:padk+w+1, padl-w:padl+w+1]
    badpixs = np.ravel(_padbad_pixels[:, padk-w:padk+w+1, padl-w:padl+w+1])
    s = np.ravel(_padnoise[:, padk-w:padk+w+1, padl-w:padl+w+1])
    badpixs[np.where(s==0)] = np.nan

    where_finite = np.where(np.isfinite(badpixs))

    if np.size(where_finite[0]) <= (1-badpixfraction) * np.size(badpixs):
        # don't bother to do a fit if there are too many bad pixels
        return np.array([]), np.array([]).reshape(0,N_linpara), np.array([])
    else:
        psfs = np.zeros((nz, boxw, boxw))
        # Technically allows super sampled PSF to account for a true 2d gaussian integration of the area of a pixel.
        # But this is disabled for now with hdfactor=1.
        hdfactor = 1#5
        xhdgrid, yhdgrid = np.meshgrid(np.arange(hdfactor * (boxw)).astype(np.float) / hdfactor,
                                       np.arange(hdfactor * (boxw)).astype(np.float) / hdfactor)
        psfs += pixgauss2d([1., w+dx, w+dy, psfw, 0.], (boxw, boxw), xhdgrid=xhdgrid, yhdgrid=yhdgrid)[None, :, :]
        psfs = psfs / np.nansum(psfs, axis=(1, 2))[:, None, None]

        # Stamp cube that will contain the data
        data_hpf = np.zeros((nz,boxw,boxw))+np.nan
        data_lpf = np.zeros((nz,boxw,boxw))+np.nan
        # Stamp cube that will contain the planet model
        scaled_psfs_hpf = np.zeros((nz,boxw,boxw))+np.nan

        # Loop over each spaxel in the stamp cube (boxw,boxw)
        for _k in range(boxw):
            for _l in range(boxw):
                lwvs = wvs[:,np.clip(k-w+_k,0,nywv-1),np.clip(l-w+_l,0,nxwv-1)]

                # The planet spectrum model is RV shifted and multiplied by the tranmission
                # Go from a 1d spectrum to the 3D scaled PSF
                planet_spec = transmission * planet_f(lwvs * (1 - (rv - cubeobj.bary_RV) / const.c.to('km/s').value))
                scaled_vec = psfs[:, _k,_l] * planet_spec

                # High pass filter the data and the models
                if hpf_mode == "gauss":
                    data_lpf[:,_k,_l] = broaden(lwvs,cube_stamp[:,_k,_l]*badpix_stamp[:,_k,_l],res_hpf)
                    data_hpf[:,_k,_l] = cube_stamp[:,_k,_l]-data_lpf[:,_k,_l]

                    scaled_vec_lpf = broaden(lwvs,scaled_vec*badpix_stamp[:,_k,_l],res_hpf)
                    scaled_psfs_hpf[:,_k,_l] = scaled_vec-scaled_vec_lpf
                elif hpf_mode == "fft":
                    for lb,rb in zip(fft_bounds[0:-1],fft_bounds[1::]):
                        data_lpf[lb:rb, _k, _l],data_hpf[lb:rb,_k,_l] = LPFvsHPF(cube_stamp[lb:rb,_k,_l]*badpix_stamp[lb:rb,_k,_l],cutoff)

                        _,scaled_psfs_hpf[lb:rb,_k,_l] = LPFvsHPF(scaled_vec[lb:rb]*badpix_stamp[lb:rb,_k,_l],cutoff)


        d = np.ravel(data_hpf)

        # combine planet model with speckle model
        M = scaled_psfs_hpf[:, :, :, None]
        # Ravel data dimension
        M = np.reshape(M, (nz * boxw * boxw, N_linpara))
        # Get rid of bad pixels
        sr = s[where_finite]
        dr = d[where_finite]
        Mr = M[where_finite[0], :]

        return dr, Mr, sr