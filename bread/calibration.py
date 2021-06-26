import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import bread.utils as utils
from bread.instruments.instrument import Instrument
from scipy.optimize import curve_fit, lsq_linear
from copy import copy
import multiprocessing as mp
from itertools import repeat
import sys # for printing in mp, and error prints
import dill # needed for mp on lambda functions
from warnings import warn
import astropy.constants as const

def import_OH_line_data(filename = None):
    """
    Obtains wavelength-intensity data for OH lines using a given data file
    """
    if filename is None:
        filename = str(utils.file_directory(__file__) + "/../data/OH_line_data.dat")
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

def wavelength_calibration_one_pixel(data: Instrument, location, relevant_OH, R=4000.0, zero_order=False,
                                     verbose=True, frac_error=1e-3, bad_pixel_threshold=5, margin=1e-12, center_data = False):
    """
    returns needed calibration for one spatial pixel
    """    
    row, col = location
    print(f"row: {row}, col: {col}")
    wavs = data.wavelengths * u.micron
    cube = data.spaxel_cube
    one_pixel = cube[:, row, col]

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
        return ((np.nan, np.nan, np.nan), (u.angstrom, None, None), None)
    
    return (tuple(p0), (u.angstrom, None, None), pCov)

def relevant_OH_line_data(data: Instrument, OH_wavelengths, OH_intensity):
    """
    returns the relevant OH line data based on wavelength range of instrument
    """
    wavs = data.wavelengths * u.micron
    wav_low, wav_high = np.where(OH_wavelengths >= wavs[0])[0][0], np.where(OH_wavelengths <= wavs[-1])[0][-1]
    relevant_OH = OH_wavelengths[wav_low:wav_high], OH_intensity[wav_low:wav_high]
    return relevant_OH

def wavelength_calibration_one_pixel_wrapper(param):
    return wavelength_calibration_one_pixel(*param)

def wavelength_calibration_cube(data: Instrument, num_threads = 16, R=4000, zero_order=False,
                                verbose=False, frac_error=1e-3, bad_pixel_threshold = 5, center_data=False):
    my_pool = mp.Pool(processes=num_threads)
    nz, nx, ny = data.spaxel_cube.shape
    OH_wavelengths, OH_intensity = import_OH_line_data()
    relevant_OH = relevant_OH_line_data(data, OH_wavelengths, OH_intensity)
    row_inputs = np.reshape(np.array(list(range(nx)) * ny), (nx, ny), order = 'F')
    col_inputs = np.reshape(np.array(list(range(ny)) * nx), (nx, ny), order = 'C')
    params = np.reshape(np.dstack((row_inputs, col_inputs)), (nx * ny, 2))
    args = zip(repeat(data), params, repeat(relevant_OH), repeat(R), repeat(zero_order),
               repeat(verbose), repeat(frac_error), repeat(bad_pixel_threshold), repeat(center_data))
    p0s = my_pool.map(wavelength_calibration_one_pixel_wrapper, args)
    p0s_values = np.array(list(map(lambda x: x[0], p0s)))
    return (np.reshape(p0s_values, (nx, ny, len(p0s[0][0]))), p0s[0][1])
