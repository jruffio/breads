import numpy as np
import astropy.io.fits as pyfits
from scipy.interpolate import interp1d
from breads.instruments.instrument import Instrument
from breads.calibration import TelluricCalibration
import breads.utils as utils
from photutils.aperture import EllipticalAperture, aperture_photometry

def read_planet_info(model, broaden, crop, margin, dataobj):
    print("reading planet file")
    if type(model) is str:
        planet_btsettl = "/scr3/jruffio/models/BT-Settl/BT-Settl_M-0.0_a+0.0/lte018-5.0-0.0a+0.0.BT-Settl.spec.7"
        arr = np.genfromtxt(planet_btsettl, delimiter=[12, 14], dtype=np.float64,
                        converters={1: lambda x: float(x.decode("utf-8").replace('D', 'e'))})
        model_wvs = arr[:, 0] / 1e4
        model_spec = 10 ** (arr[:, 1] - 8)
    else:
        model_wvs, model_spec = model

    print("setting planet model")
    if crop:
        minwv, maxwv= np.nanmin(dataobj.wavelengths), np.nanmax(dataobj.wavelengths)
        crop_btsettl = np.where((model_wvs > minwv - margin) * (model_wvs < maxwv + margin))
        model_wvs = model_wvs[crop_btsettl]
        model_spec = model_spec[crop_btsettl]
    if broaden:
        model_broadspec = dataobj.broaden(model_wvs,model_spec)
    
    planet_f = interp1d(model_wvs, model_broadspec, bounds_error=False, fill_value=np.nan)

    return planet_f

def read_transmission_info(transmission):
    if type(transmission) is str:
        with pyfits.open(transmission) as hdulist:
            transmission = hdulist[0].data
    return (transmission / np.nanmedian(transmission))

def read_star_info(star):
    print("reading star info")
    if type(star) is str:
        with pyfits.open(star) as hdulist:
            star_spectrum = hdulist[2].data
            star_x, star_y = hdulist[3].data, hdulist[4].data
            star_sigx, star_sigy = hdulist[5].data, hdulist[6].data
            star_flux = np.nanmean(star_spectrum) * np.size(star_spectrum)
            if "aperture_sigmas" in hdulist[2].header.keys():
                aperture_sigmas = hdulist[2].header["aperture_sigmas"]
            else:
                aperture_sigmas = 5
            return (star_x, star_y, star_sigx, star_sigy, star_flux, aperture_sigmas)
    elif isinstance(star, TelluricCalibration):
        star_spectrum = star.fluxs
        star_x, star_y = star.mu_xs, star.mu_ys
        star_sigx, star_sigy = star.sig_xs, star.sig_ys
        star_flux = np.nanmean(star_spectrum) * np.size(star_spectrum)
        aperture_sigmas = star.aperture_sigmas
        return (star_x, star_y, star_sigx, star_sigy, star_flux, aperture_sigmas)
    else:
        return star

def inject_planet(dataobj: Instrument, location, model, star, transmission, planet_star_ratio, \
    broaden=True, crop=True, margin=0.2):

    planet_f = read_planet_info(model, broaden, crop, margin, dataobj)
    star_x, star_y, sigx, sigy, star_flux, aperture_sigmas = read_star_info(star)
    transmission = read_transmission_info(transmission)
    x, y = location
    planet_x, planet_y = star_x + x, star_y + y 
    planet_data = np.zeros_like(dataobj.data)
    nz, ny, nx = dataobj.data.shape
    planet_f_vals = planet_f(dataobj.wavelengths)
    planet_flux = 0.0

    print("start injection")
    for ind, val in zip(range(nz), planet_f_vals):
        planet_data[ind] = val * utils.gaussian2D(ny, nx, planet_x[ind], planet_y[ind], \
            sigx[ind], sigy[ind], 1 / (2*np.pi*sigx[ind]*sigy[ind])) * transmission[ind]
        
        # aperture photometry
        aper_photo = aperture_photometry(planet_data[ind], \
            EllipticalAperture((planet_y[ind], planet_x[ind]), \
                aperture_sigmas*sigy[ind], aperture_sigmas*sigx[ind])) 
        # weirdly photutils uses order y, x
        planet_flux += aper_photo['aperture_sum'][0]
        
    print("normalizing and adding to data")
    const = planet_star_ratio * star_flux / planet_flux
    dataobj.data += planet_data * const
