import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import bread.utils as utils
from bread.instruments.instrument import Instrument
from scipy.optimize import curve_fit, lsq_linear
from copy import copy
import multiprocessing as mp
from itertools import repeat
import sys # for printing in mp
import dill # needed for mp on lambda functions

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
    A = np.transpose(np.vstack([sky_model, wavs_val ** 2, wavs_val, np.ones_like(wavs_val)]))
    b = one_pixel
    best_x = lsq_linear(A, b)['x']
    best_model = np.dot(A, best_x)
    res = b - best_model
    good_pixels = np.where(np.abs(res) < bad_pixel_threshold * np.nanstd(res))[0] #find outliers
    return lsq_linear(A[good_pixels, :], b[good_pixels])['x']

def const_offset_fitter(wavs, offset, R, one_pixel, relevant_OH,
                        verbose=True, bad_pixel_threshold = 5):
    """
    Fitter used for obtaining a constant offset correction for wavelength calibration
    """
    wavs = wavs.astype(float) * u.micron
    sky_model = np.zeros_like(wavs.value)
    
    for i, wav in enumerate(relevant_OH[0]):
        fwhm = wav / R
        sky_model += relevant_OH[1][i] * \
                    (gaussian1D(wavs, wav+offset*u.nm, fwhm) * (wavs[1]-wavs[0])).to('').value 
    G, a, b, c = sky_model_linear_parameters(wavs.value, sky_model, one_pixel,
                                             bad_pixel_threshold = bad_pixel_threshold)
    if verbose:
        print(G, offset, R, a, b, c)
    return G * sky_model + (a * (wavs.value ** 2) + b * wavs.value + c)

def const_offset_initial_guess(wavs, one_pixel):
    roll_avg = np.zeros_like(one_pixel)
    w = len(one_pixel) // 20
    for i in range(len(one_pixel)):
        roll_avg[i] = np.mean(one_pixel[i: i+w])
    a, b, c = np.polyfit(wavs, roll_avg, deg=2)
    G = np.max(one_pixel) * 1e-3
    offset = 0
    return G, offset, a, b, c

def wavelength_calibration_one_pixel(data: Instrument, location, relevant_OH, R=4000, 
                                     verbose=True, frac_error=1e-3, bad_pixel_threshold=5):
    """
    returns needed calibration for one spatial pixel
    """    
    row, col = location
    print(f"row: {row}, col: {col}")
    sys.stdout.flush()
    wavs = data.wavelengths * u.micron
    sky_model = np.zeros_like(wavs.value)
    cube = data.spaxel_cube
    one_pixel = cube[:, row, col]
    if R is None:
        fit_wrapper = lambda *p : const_offset_fitter(*p, one_pixel, relevant_OH,
                                                      verbose=verbose, bad_pixel_threshold = bad_pixel_threshold)
        try:
            p0, _ = curve_fit(fit_wrapper, wavs, one_pixel, p0=[0, 4000], xtol=frac_error)
        except e:
            return ((np.nan, np.nan), u.nm)
    else:
        fit_wrapper = lambda *p : const_offset_fitter(*p, R, one_pixel, relevant_OH,
                                                      verbose=verbose, bad_pixel_threshold = bad_pixel_threshold)
        try:
            p0, _ = curve_fit(fit_wrapper, wavs, one_pixel, p0=[0], xtol=frac_error)
        except e:
            return ((np.nan, np.nan), u.nm)
    return (tuple(p0), u.nm)

def relevant_OH_line_data(data: Instrument, OH_wavelengths, OH_intensity):
    wavs = data.wavelengths * u.micron
    wav_low, wav_high = np.where(OH_wavelengths >= wavs[0])[0][0], np.where(OH_wavelengths <= wavs[-1])[0][-1]
    relevant_OH = OH_wavelengths[wav_low:wav_high], OH_intensity[wav_low:wav_high]
    return relevant_OH

def wavelength_calibration_one_pixel_wrapper(param):
    return wavelength_calibration_one_pixel(*param)

def wavelength_calibration_cube(data: Instrument, num_threads = 16, R=4000,
                                verbose=False, frac_error=1e-3, bad_pixel_threshold = 5):
    # mp code
    my_pool = mp.Pool(processes=num_threads)
    nz, nx, ny = data.spaxel_cube.shape
    OH_wavelengths, OH_intensity = import_OH_line_data()
    relevant_OH = relevant_OH_line_data(data, OH_wavelengths, OH_intensity)
    row_inputs = np.reshape(np.array(list(range(nx)) * ny), (nx, ny), order = 'F')
    col_inputs = np.reshape(np.array(list(range(ny)) * nx), (nx, ny), order = 'C')
    params = np.reshape(np.dstack((row_inputs, col_inputs)), (nx * ny, 2))
    args = zip(repeat(data), params, repeat(relevant_OH), repeat(R),
               repeat(verbose), repeat(frac_error), repeat(bad_pixel_threshold))
    p0s = my_pool.map(wavelength_calibration_one_pixel_wrapper, args)
    print(p0s)
    p0s_values = np.array(list(map(lambda x: x[0], p0s)))
    return (np.reshape(p0s_values, (nx, ny, len(p0s[0][0]))), p0s[0][1])
