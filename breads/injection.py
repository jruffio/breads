import numpy as np
import astropy.io.fits as pyfits
from scipy.interpolate import interp1d
from breads.instruments.instrument import Instrument
from breads.calibration import TelluricCalibration
import breads.utils as utils
from photutils.aperture import EllipticalAperture, aperture_photometry
from matplotlib import pyplot as plt

def read_planet_info(model, broaden, crop, margin, dataobj):
    if type(model) == interp1d:
        print("planet model is interp1d")
        return model
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
    return inject_planet_stamp(dataobj, location, model, star, transmission, planet_star_ratio, \
    broaden, crop, margin)

def inject_planet_stamp(dataobj: Instrument, location, model, star, transmission, planet_star_ratio, \
    broaden=True, crop=True, margin=0.2, stamp_size = 10):

    # plt.figure()
    # plt.imshow(np.nanmedian(dataobj.data, axis=0))

    planet_f = read_planet_info(model, broaden, crop, margin, dataobj)
    star_x, star_y, sigx, sigy, star_flux, aperture_sigmas = read_star_info(star)
    transmission = read_transmission_info(transmission)
    x, y = location
    planet_x, planet_y = star_x + x, star_y + y 
    planet_data = np.zeros_like(dataobj.data)
    nz, ny, nx = dataobj.data.shape
    planet_f_vals = planet_f(dataobj.wavelengths)

    star_med_x, star_med_y = int(np.nanmedian(star_x)), int(np.nanmedian(star_y))
    pl_med_x, pl_med_y = int(np.nanmedian(planet_x)), int(np.nanmedian(planet_y))

    stamp_cube = dataobj.data[:, star_med_x-stamp_size:star_med_x+stamp_size+1, star_med_y-stamp_size:star_med_y+stamp_size+1]
    total_flux = np.size(stamp_cube) * np.nanmean(stamp_cube)
    stamp_cube = stamp_cube/np.nansum(stamp_cube,axis=(1,2))[:,None,None]
    stamp_cube = stamp_cube * transmission[:, None, None] * planet_f_vals[:, pl_med_x-stamp_size:pl_med_x+stamp_size+1, pl_med_y-stamp_size:pl_med_y+stamp_size+1]
    
    planet_data[:, pl_med_x-stamp_size:pl_med_x+stamp_size+1, pl_med_y-stamp_size:pl_med_y+stamp_size+1] = stamp_cube
    planet_flux = np.size(stamp_cube) * np.nanmean(stamp_cube)
        
    print("normalizing and adding to data")
    const = planet_star_ratio * total_flux / planet_flux
    dataobj.data += planet_data * const

    # plt.figure()
    # plt.imshow(np.nanmedian(dataobj.data, axis=0))

    # plt.figure()
    # plt.plot(dataobj.read_wavelengths, np.nanmedian(stamp_cube, axis=(1,2)) / np.nanmedian(np.nanmedian(stamp_cube, axis=(1,2))))
    # plt.plot(dataobj.read_wavelengths, (planet_f(dataobj.read_wavelengths) * transmission) / np.nanmedian((planet_f(dataobj.read_wavelengths) * transmission)))

    # plt.show()
    # exit()


def inject_planet_real(dataobj: Instrument, location, model, star, transmission, planet_star_ratio, \
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
            sigy[ind], sigx[ind], 1 / (2*np.pi*sigx[ind]*sigy[ind])) * transmission[ind]
        
        # aperture photometry
        aper_photo = aperture_photometry(planet_data[ind], \
            EllipticalAperture((planet_y[ind], planet_x[ind]), \
                aperture_sigmas*sigy[ind], aperture_sigmas*sigx[ind])) 
        # weirdly photutils uses order y, x
        planet_flux += aper_photo['aperture_sum'][0]
        
    print("normalizing and adding to data")
    const = planet_star_ratio * star_flux / planet_flux
    dataobj.data += planet_data * const

def inject_planet_test(dataobj: Instrument, location, model, star, transmission, planet_star_ratio, \
    broaden=True, crop=True, margin=0.2):

    planet_f = read_planet_info(model, broaden, crop, margin, dataobj)
    star_x, star_y, sigx, sigy, star_flux, aperture_sigmas = read_star_info(star)
    transmission = read_transmission_info(transmission)
    x, y = location
    planet_x, planet_y = np.nanmedian(star_x) + x, np.nanmedian(star_y) + y 
    planet_data = np.zeros_like(dataobj.data)
    nz, ny, nx = dataobj.data.shape
    planet_f_vals = planet_f(dataobj.wavelengths)
    planet_flux = 0.0

    sigx, sigy = 1, 1

    print("start injection")
    for ind, val in zip(range(nz), planet_f_vals):
        planet_data[ind] = val * utils.gaussian2D(ny, nx, planet_x, planet_y, \
            sigy, sigx, 1 / (2*np.pi*sigx*sigy)) * transmission[ind]
        
        # aperture photometry
        aper_photo = aperture_photometry(planet_data[ind], \
            EllipticalAperture((planet_y, planet_x), \
                aperture_sigmas*sigy, aperture_sigmas*sigx)) 
        # weirdly photutils uses order y, x
        planet_flux += aper_photo['aperture_sum'][0]
         
    print("normalizing and adding to data")
    const = planet_star_ratio * star_flux / planet_flux
    dataobj.data = planet_data * const
