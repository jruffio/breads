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

def sky_model_linear_parameters(wavs_val, sky_model, one_pixel):
    A = np.transpose(np.vstack([sky_model, wavs_val ** 2, wavs_val, np.ones_like(wavs_val)]))
    b = one_pixel
    return lsq_linear(A, b)['x']

def const_offset_fitter(wavs, offset, one_pixel, relevant_OH, verbose=True):
    """
    Fitter used for obtaining a constant offset correction for wavelength calibration
    """
    wavs = wavs.astype(float) * u.micron
    sky_model = np.zeros_like(wavs.value)
    for i, wav in enumerate(relevant_OH[0]):
        fwhm = wav / 4000
        sky_model += relevant_OH[1][i] * \
                    (gaussian1D(wavs, wav+offset*u.nm, fwhm) * (wavs[1]-wavs[0])).to('').value 
    G, a, b, c = sky_model_linear_parameters(wavs.value, sky_model, one_pixel)
    if verbose:
        print(G, offset, a, b, c)
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

def wavelength_calibration_one_pixel(data: Instrument, location, relevant_OH, verbose=True, frac_error=1e-3):
    """
    returns needed calibration for one spatial pixel
    """    
    row, col = location
    print(f"row: {row}, col: {col}")
    sys.stdout.flush()
    R = 4000
    wavs = data.wavelengths * u.micron
    sky_model = np.zeros_like(wavs.value)
    cube = data.spaxel_cube
    one_pixel = cube[:, row, col]
    fit_wrapper = lambda *p : const_offset_fitter(*p, one_pixel, relevant_OH, verbose=verbose)
    p0, _ = curve_fit(fit_wrapper, wavs, one_pixel, p0=[0], xtol=frac_error)
    return (p0[0] * u.nm, p0)

def relevant_OH_line_data(data: Instrument, OH_wavelengths, OH_intensity):
    wavs = data.wavelengths * u.micron
    wav_low, wav_high = np.where(OH_wavelengths >= wavs[0])[0][0], np.where(OH_wavelengths <= wavs[-1])[0][-1]
    relevant_OH = OH_wavelengths[wav_low:wav_high], OH_intensity[wav_low:wav_high]
    return relevant_OH

def wavelength_calibration_one_pixel_wrapper(param):
    return wavelength_calibration_one_pixel(*param)[0]

def wavelength_calibration_cube(data: Instrument, num_threads = 16, verbose=False, frac_error=1e-3):
    # mp code
    my_pool = mp.Pool(processes=num_threads)
    nz, nx, ny = data.spaxel_cube.shape
    OH_wavelengths, OH_intensity = import_OH_line_data()
    relevant_OH = relevant_OH_line_data(data, OH_wavelengths, OH_intensity)
    row_inputs = np.reshape(np.array(list(range(nx)) * ny), (nx, ny), order = 'F')
    col_inputs = np.reshape(np.array(list(range(ny)) * nx), (nx, ny), order = 'C')
    params = np.reshape(np.dstack((row_inputs, col_inputs)), (nx * ny, 2))
    args = zip(repeat(data), params, repeat(relevant_OH), repeat(verbose), repeat(frac_error))
    offsets = my_pool.map(wavelength_calibration_one_pixel_wrapper, args)
    # frac error of 1e-3 corresponds to 3 sig figs
    off_values = np.array(list(map(lambda x: x.value, offsets)))
    return (np.reshape(off_values, (nx, ny)), offsets[0].unit)
