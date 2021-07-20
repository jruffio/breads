import numpy as np
import astropy.units as u
import breads.utils as utils
from breads.instruments.instrument import Instrument
from scipy.optimize import curve_fit, lsq_linear, minimize
from copy import deepcopy
import multiprocessing as mp
from itertools import repeat
import sys # for printing in mp, and error prints
from warnings import warn
import astropy.constants as const
from photutils.aperture import EllipticalAperture, aperture_photometry
import astropy.io.fits as pyfits

#############################
# OH line calibration code

def import_OH_line_data(filename = None):
    """
    Obtains wavelength-intensity data for OH lines using a given data file
    """
    # would be worthwhile to, in future, add this to __init__ of breads.calibration
    # and make a global (OH_wavelengths, OH_intensity)
    if filename is None:
        filename = str(utils.file_directory(__file__) + "/data/OH_line_data.dat")
    OH_lines_file = open(filename, 'r')
    OH_lines = [x for x in OH_lines_file.readlines() if x[0] != "#"]
    OH_wavelengths = np.array([]) * u.angstrom
    OH_intensity = np.array([])
    for line in OH_lines:
        walen, inten = line.split()
        OH_wavelengths = np.append(OH_wavelengths, float(walen) * u.angstrom)
        OH_intensity = np.append(OH_intensity, float(inten))
    return (OH_wavelengths, OH_intensity)

def gaussian1D(Xs, wavelength, fwhm):
    """
    one-dimensional Gaussian, given a mean wavelength and a FWHM, computed over given x values
    """
    sig = fwhm / (2 * np.sqrt(2 * np.log(2)))
    mu = wavelength
    gauss = np.exp(- (Xs - mu) ** 2 / (2 * sig * sig))
    return gauss / (sig * np.sqrt(2 * np.pi))

def sky_model_linear_parameters(wavs_val, sky_model, one_pixel, bad_pixel_threshold = 5):
    A = np.transpose(np.vstack([sky_model, wavs_val ** 4, wavs_val ** 3, wavs_val ** 2,  
                                wavs_val, np.ones_like(wavs_val)]))
    b = one_pixel
    best_x = lsq_linear(A, b)['x']
    best_model = np.dot(A, best_x)
    res = b - best_model
    good_pixels = np.where(np.abs(res) < bad_pixel_threshold * np.nanstd(res))[0] #find outliers
    return lsq_linear(A[good_pixels, :], b[good_pixels])['x']

def offset_fitter(wavs, off0, off1, R, one_pixel, relevant_OH,
                        verbose=True, bad_pixel_threshold = 5, center_data = False):
    """
    Fitter used for obtaining a constant offset correction for wavelength calibration
    """
    wavs = wavs.astype(float) * u.micron
    if center_data:
        wavs = wavs + (wavs - np.mean(wavs)) * off1 + off0 * u.angstrom
    else:
        wavs = wavs * (1 + off1) + off0 * u.angstrom
    
    sky_model = np.zeros_like(wavs.value)
    for i, wav in enumerate(relevant_OH[0]):
        fwhm = wav / R
        sky_model += relevant_OH[1][i] * \
                    (gaussian1D(wavs, wav, fwhm) * (wavs[1]-wavs[0])).to('').value 
    G, a, b, c, d, e = sky_model_linear_parameters(wavs.value, sky_model, one_pixel,
                                             bad_pixel_threshold = bad_pixel_threshold)
    if verbose:
        print(G, off0, off1, R, a, b, c, d, e)
    x = wavs.value
    return G * sky_model + (a * (x ** 4) + b * (x ** 3) + 
                            c * (x ** 2) + d * x + e)

def const_offset_initial_guess(wavs, one_pixel):
    roll_avg = np.zeros_like(one_pixel)
    w = len(one_pixel) // 20
    for i in range(len(one_pixel)):
        roll_avg[i] = np.mean(one_pixel[i: i+w])
    a, b, c = np.polyfit(wavs, roll_avg, deg=2)
    G = np.max(one_pixel) * 1e-3
    offset = 0
    return G, offset, a, b, c

def bounds_Rp0(R, zero_order, margin):
    if (R is None) and not (zero_order):
            bounds = (-np.inf, np.inf)
            Rp0 = 4000.0
    if (R is None) and (zero_order):
        bounds = ([-np.inf, -margin, -np.inf], [np.inf, margin, np.inf])
        Rp0 = 4000.0
    if not (R is None) and not (zero_order):
        bounds =  ([-np.inf, -np.inf, R-margin], [np.inf, np.inf, R+margin])
        Rp0 = R
    if not (R is None) and (zero_order):
        bounds = ([-np.inf, -margin, R-margin], [np.inf, margin, R+margin])
        Rp0 = R
    return (bounds, Rp0)

def wavelength_calibration_one_pixel(wavs, one_pixel, location, relevant_OH, R=4000.0, zero_order=False,
                                     verbose=True, frac_error=1e-3, bad_pixel_threshold=5, margin=1e-12, center_data=False):
    """
    returns needed calibration for one spatial pixel
    """
    row, col = location
    print(f"row: {row}, col: {col}")
    bounds, Rp0 = bounds_Rp0(R, zero_order, margin)

    good_pixels = np.where(~np.isclose(one_pixel, 0))[0] #find edge pixels
    
    if len(good_pixels) == 0:
        # if data is all zero
        warn(f"data at row: {row}, col: {col} is all 0")
        return ((np.nan, np.nan, np.nan), (u.angstrom, None, None), None)
        
    wavs, one_pixel = wavs[good_pixels], one_pixel[good_pixels]

    fit_wrapper = lambda *p : offset_fitter(*p, one_pixel, relevant_OH,
                                                verbose=verbose, bad_pixel_threshold = bad_pixel_threshold, 
                                                center_data = center_data)
    try:
        p0, pCov = curve_fit(fit_wrapper, wavs, one_pixel, p0=[0., 0., Rp0], xtol=frac_error, bounds=bounds)
        # bad_fit = ((np.isinf(pCov)).any()) \
        #             or (abs(p0[0]) < abs(np.sqrt(pCov[0, 0]))) \
        #                 or (abs(p0[1]) < abs(np.sqrt(pCov[1, 1]))) \
        #                     or (abs(p0[2]) < abs(np.sqrt(pCov[2, 2])))
        # if bad_fit: # fit is not great
        #     warn(f"data at row: {row}, col: {col} did not give a good fit")
        #     return ((np.nan, np.nan, np.nan), u.angstrom, (p0, pCov))
    except Exception as e:
        warn(f"data at row: {row}, col: {col} did not fit: \n" + str(e))
        raise e
        return ((np.nan, np.nan, np.nan), (u.angstrom, None, None), None)
    
    return (tuple(p0), (u.angstrom, None, None), pCov)

def relevant_OH_line_data(data: Instrument, OH_wavelengths, OH_intensity):
    """
    returns the relevant OH line data based on wavelength range of instrument
    """
    wavs = data.read_wavelengths * u.micron
    wav_low, wav_high = np.where(OH_wavelengths >= wavs[0])[0][0], np.where(OH_wavelengths <= wavs[-1])[0][-1]
    relevant_OH = OH_wavelengths[wav_low:wav_high], OH_intensity[wav_low:wav_high]
    return relevant_OH

def wavelength_calibration_one_pixel_wrapper(param):
    return wavelength_calibration_one_pixel(*param)

def sky_calibration(data: Instrument, num_threads = 16, R=4000, zero_order=False,
                                verbose=False, frac_error=1e-3, bad_pixel_threshold = 5, 
                                margin=1e-12, center_data=False, calib_filename=None):
    my_pool = mp.Pool(processes=num_threads)
    nz, nx, ny = data.data.shape
    OH_wavelengths, OH_intensity = import_OH_line_data()
    relevant_OH = relevant_OH_line_data(data, OH_wavelengths, OH_intensity)
    row_inputs = np.reshape(np.array(list(range(nx)) * ny), (nx, ny), order = 'F')
    col_inputs = np.reshape(np.array(list(range(ny)) * nx), (nx, ny), order = 'C')
    params = np.reshape(np.dstack((row_inputs, col_inputs)), (nx * ny, 2))
    args = zip(repeat(data.read_wavelengths), np.transpose(data.data.reshape((nz, nx * ny), order='C')), 
                params, repeat(relevant_OH), repeat(R), repeat(zero_order),
                repeat(verbose), repeat(frac_error), repeat(bad_pixel_threshold), 
                repeat(margin), repeat(center_data))
    p0s = my_pool.map(wavelength_calibration_one_pixel_wrapper, args)
    p0s_values = np.array(list(map(lambda x: x[0], p0s)))
    return SkyCalibration(data, np.reshape(p0s_values, (nx, ny, len(p0s[0][0]))), \
        p0s[0][1], calib_filename, center_data)

def corrected_wavelengths(data, off0, off1, center_data):
    wavs = data.read_wavelengths.astype(float) * u.micron
    if center_data:
        wavs = wavs + (wavs - np.mean(wavs)) * off1 + off0 * u.angstrom
    else:
        wavs = wavs * (1 + off1) + off0 * u.angstrom
    return wavs

class SkyCalibration:
    def __init__(self, data: Instrument, fit_values, unit, calib_filename, center_data):
        if calib_filename is None:
            calib_filename = "./sky_calib_file.fits"
        self.calib_filename = calib_filename
        self.unit = unit
        self.fit_values = fit_values
        corr_wavs = np.zeros_like(data.data)
        nz, nx, ny = corr_wavs.shape
        for i in range(nx):
            for j in range(ny):
                corr_wavs[:, i, j] = \
                    corrected_wavelengths(data, fit_values[i, j, 0], fit_values[i, j, 1], center_data)
        self.corrected_wavelengths = corr_wavs
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=corr_wavs,
                                        header=pyfits.Header(cards={"TYPE": "corrected_wavelengths"})))
        hdulist.append(pyfits.ImageHDU(data=fit_values[:, :, 0],
                                        header=pyfits.Header(cards={"TYPE": "const"})))
        hdulist.append(pyfits.ImageHDU(data=fit_values[:, :, 1],
                                        header=pyfits.Header(cards={"TYPE": "RV"})))                         
        hdulist.append(pyfits.ImageHDU(data=fit_values[:, :, 2],
                                        header=pyfits.Header(cards={"TYPE": "R"})))                 
        try:
            hdulist.writeto(calib_filename, overwrite=True)
        except TypeError:
            hdulist.writeto(calib_filename, clobber=True)
        hdulist.close()

#############################
# Telluric Calibration code

def mask_sky_remnant(slice, sigma=0.3, n_sigmas=2):
    """
    Given a slice of data cube at particular wavelength, 
    sets values below threshold to np.nan. 
    Deepcopys passed in argument.

    Useful for when the sky subtracted to get standard star image 
    was itself an image of the star at an offset.

    Default for threshold set to standard deviation of noise in 
    s161106_a007002_Kbb_020 far from star
    """
    slice = deepcopy(slice)
    slice[slice < (-sigma * n_sigmas)] = np.nan
    return slice

def gaussian2D(nx, ny, mu_x, mu_y, sig_x, sig_y, A):
    """
    Two Dimensional Gaussian for getting PSF for different wavelength slices
    """
    x_vals, y_vals = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    gauss = A * np.exp(-((x_vals - mu_x) ** 2) / (2 * sig_x * sig_x)) * \
        np.exp(-((y_vals - mu_y) ** 2) / (2 * sig_y * sig_y))
    return gauss

def psf_fitter(img_slice, psf_func=gaussian2D, x0=None, \
    residual=False, mask=False, minimize_method='nelder-mead', sigma=0.3, n_sigmas=2):
    """
    psf_func should be such that the first six arguments are nx, ny, mu_x, mu_y, sig_x, sig_y
    if you pass in a psf_func other than gaussian2D, you must pass in x0 initial parameters
    """
    if mask:
        img_slice = mask_sky_remnant(img_slice, sigma, n_sigmas)
    if psf_func == gaussian2D and x0 is None:
        x0 = [*np.unravel_index(np.nanargmax(img_slice), img_slice.shape), 2, 2, np.nanmax(img_slice)]
    else:
        assert (x0 is not None), \
            "if you pass in a psf_func other than gaussian2D, you must pass in x0 initial parameters"
    nx, ny = img_slice.shape
    wrapper = lambda params: np.nansum((psf_func(nx, ny, *params) - img_slice) ** 2)
    fit = minimize(wrapper, x0, method=minimize_method)
    if fit.success:
        fit_values = fit.x
    else:
        return(np.nan, np.nan, np.nan, np.nan, fit.x * np.nan, np.nan * np.zeros((nx, ny)))
    if residual:
        residuals = img_slice - psf_func(nx, ny, *fit_values)
    else:
        residuals = None
    return (fit_values[0], fit_values[1], fit_values[2], fit_values[3], fit_values, residuals)

def parse_star_spectrum(wavs, star_spectrum, R):
    # would be worthwhile to, in future, add this to __init__ of breads.calibration
    # and make a global scipy.interpolate.interp1d function for using always
    # (have it work over all filters for OSIRIS, etc.) 
    assert(len(star_spectrum) == 2), \
        "incorrect star_spectrum format, run \"help(telluric_calibration)\""

    if type(star_spectrum[0]) == str:
        assert(star_spectrum[0].endswith(".fits")), "file must be of .fits format"
        with pyfits.open(star_spectrum[0]) as hdulist:
            wavs_spec = np.array(hdulist[0].data / 1e4) # angstroms to microns
    else:
        wavs_spec = star_spectrum[0]

    if type(star_spectrum[1]) == str:
        assert(star_spectrum[0].endswith(".fits")), "file must be of .fits format"
        with pyfits.open(star_spectrum[1]) as hdulist:
            spec = np.array(hdulist[0].data)
    else:
        spec = star_spectrum[1]
    
    indices = np.logical_and(wavs[0] < wavs_spec, wavs_spec < wavs[-1])
    wavs_spec, spec = wavs_spec[indices], spec[indices]

    # reduce to ~1 OoM
    spec = spec / (10 ** (np.floor(np.log10(np.median(spec)))))

    return np.interp(wavs, wavs_spec, utils.broaden(wavs_spec, spec, R))

def telluric_calibration(data: Instrument, star_spectrum, calib_filename=None,
        psf_func=gaussian2D, x0=None, residual=False, mask=False, sigma=0.3, n_sigmas=2, verbose=False, 
        aperture_sigmas=5, R=4000):
    """
    star_spectrum needs to be a 2-tuple or array-like of length 2. 
    It can either be (wavelength data-array, flux data-array) or (wavelength datafile, flux datafile). 
    The file must be .fits, probably downloaded from Vizier. 
    Units of wavelengths must be angstroms for files (default for Vizier), and angstroms for array.
    star_spectrum need not be of the same resolution, length, or in any way related to data.

    aperture using for aperture photometry is of the size: 
    sig_x * aperture_sigmas, sig_y * aperture_sigmas

    psf_func should be such that the first six arguments are nx, ny, mu_x, mu_y, sig_x, sig_y
    if you pass in a psf_func other than gaussian2D, you must pass in x0 initial parameters
    """
    mu_xs, mu_ys, sig_xs, sig_ys, all_fit_values, residuals, fluxs = [], [], [], [], [], [], []
            
    if psf_func == gaussian2D and x0 is None:
        img_mean = np.nanmean(data.data, axis=0)
        x0 = [*np.unravel_index(np.nanargmax(img_mean), img_mean.shape), 2, 2, np.nanmax(img_mean)]
    else:
        assert (x0 is not None), \
            "if you pass in a psf_func other than gaussian2D, you must pass in x0 initial parameters"
    
    for i, img_slice in enumerate(data.data):
        if verbose and (i % 200 == 0):
            print(f'index {i} wavelength {data.read_wavelengths[i]}')
        params = psf_fitter(img_slice, psf_func=psf_func, x0=x0,\
            residual=residual, mask=mask, sigma=sigma, n_sigmas=n_sigmas)
        print(params)
        mu_x, mu_y, sig_x, sig_y, fit_vals, resid = params
        mu_xs += [mu_x]; mu_ys += [mu_y]
        sig_xs += [sig_x]; sig_ys += [sig_y]
        all_fit_values += [fit_vals]
        residuals += [resid]
        aper_photo = aperture_photometry(img_slice, \
            EllipticalAperture((mu_y, mu_x), aperture_sigmas*sig_y, aperture_sigmas*sig_x)) 
        # check if a, b order is correct, seems correct, weirdly photutils uses order y, x
        fluxs += [aper_photo['aperture_sum'][0]]

    transmission = fluxs / parse_star_spectrum(data.read_wavelengths, star_spectrum, R)
    return TelluricCalibration(data, *tuple(map(np.array, mu_xs, mu_ys, \
        (sig_xs, sig_ys, all_fit_values, residuals, fluxs, transmission))), calib_filename)

class TelluricCalibration:
    def __init__(self, data: Instrument, \
        mu_xs, mu_ys, sig_xs, sig_ys, fit_values, residuals, fluxs, transmission, calib_filename):
        if calib_filename is None:
            calib_filename = "./telluric_calib_file.fits"
        self.calib_filename = calib_filename
        self.mu_xs = mu_xs; self.mu_ys = mu_ys 
        self.sig_xs = sig_xs; self.sig_ys = sig_ys
        self.fit_values = fit_values
        self.residuals = residuals
        self.fluxs = fluxs
        self.transmission = transmission
        self.read_wavelengths = data.read_wavelengths

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=self.transmission,
                                        header=pyfits.Header(cards={"TYPE": "transmission"})))
        hdulist.append(pyfits.ImageHDU(data=self.read_wavelengths,
                                        header=pyfits.Header(cards={"TYPE": "wavelengths"})))
        hdulist.append(pyfits.ImageHDU(data=self.mu_xs,
                                        header=pyfits.Header(cards={"TYPE": "Mu X"})))
        hdulist.append(pyfits.ImageHDU(data=self.mu_ys,
                                        header=pyfits.Header(cards={"TYPE": "Mu Y"}))) 
        hdulist.append(pyfits.ImageHDU(data=self.sig_xs,
                                        header=pyfits.Header(cards={"TYPE": "Sigma X"})))
        hdulist.append(pyfits.ImageHDU(data=self.sig_ys,
                                        header=pyfits.Header(cards={"TYPE": "Sigma Y"})))                                   
        try:
            hdulist.writeto(calib_filename, overwrite=True)
        except TypeError:
            hdulist.writeto(calib_filename, clobber=True)
        hdulist.close()

def extract_star_spectrum(data: Instrument, star_spectrum, calib_filename="./star_spectrum_file.fits",
        psf_func=gaussian2D, x0=None, residual=False, mask=False, sigma=0.3, n_sigmas=2, verbose=False, 
        aperture_sigmas=5, R=4000):
    star_spectrum = (data.read_wavelengths, np.ones_like(data.read_wavelengths))
    return telluric_calibration(data, star_spectrum, calib_filename, \
        psf_func, x0, residual, mask, sigma, n_sigmas, verbose, aperture_sigmas, R)