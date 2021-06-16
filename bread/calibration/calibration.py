import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import utils
from scipy.optimize import curve_fit, lsq_linear
from copy import copy
import multiprocessing as mp
from itertools import repeat
import sys # for printing in mp
import dill # needed for mp on lambda functions

class Calibrator:
    def __init__(self, osiris_data, num_threads = 16, verbose=False, frac_error=1e-3):
        self.num_threads = num_threads
        self.my_pool = mp.Pool(processes=num_threads)
        self.OH_wavelengths, self.OH_intensity = self.import_OH_line_data()
        self.osiris_data = osiris_data
        self.verbose = verbose
        self.frac_error = frac_error
        self.relevant_OH = relevant_OH_line_data(self.osiris_data)
    
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['my_pool']
        return self_dict
    
    def import_OH_line_data(self, filename = "../../data/OH_line_data.dat"):
        """
        Obtains wavelength-intensity data for OH lines using a given data file
        """
        OH_lines_file = open(filename, 'r')
        OH_lines = [x for x in OH_lines_file.readlines() if x[0] != "#"]
        OH_wavelengths = np.array([]) * u.angstrom
        OH_intensity = np.array([])
        for line in OH_lines:
            walen, inten = line.split()
            OH_wavelengths = np.append(OH_wavelengths, float(walen) * u.angstrom)
            OH_intensity = np.append(OH_intensity, float(inten))
        return (OH_wavelengths, OH_intensity)

    def gaussian1D(self, Xs, wavelength, fwhm):
        """
        one-dimensional Gaussian, given a mean wavelength and a FWHM, computed over given x values
        """
        sig = fwhm / (2 * np.sqrt(2 * np.log(2)))
        mu = wavelength
        gauss = np.exp(- (Xs - mu) ** 2 / (2 * sig * sig))
        return gauss / (sig * np.sqrt(2 * np.pi))

    def sky_model_linear_parameters(self, wavs_val, sky_model, one_pixel):
        A = np.transpose(np.vstack([sky_model, wavs_val ** 2, wavs_val, np.ones_like(wavs_val)]))
        b = one_pixel
        return lsq_linear(A, b)['x']

    def const_offset_fitter(self, wavs, offset, one_pixel):
        """
        Fitter used for obtaining a constant offset correction for wavelength calibration
        """
        wavs = wavs.astype(float) * u.micron
        sky_model = np.zeros_like(wavs.value)
        for i, wav in enumerate(self.relevant_OH[0]):
            fwhm = wav / 4000
            sky_model += self.relevant_OH[1][i] * \
                        (self.gaussian1D(wavs, wav+offset*u.nm, fwhm) * (wavs[1]-wavs[0])).to('').value 
        G, a, b, c = self.sky_model_linear_parameters(wavs.value, sky_model, one_pixel)
        if self.verbose:
            print(G, offset, a, b, c)
        return G * sky_model + (a * (wavs.value ** 2) + b * wavs.value + c)

    def const_offset_initial_guess(self, wavs, one_pixel):
        roll_avg = np.zeros_like(one_pixel)
        w = len(one_pixel) // 20
        for i in range(len(one_pixel)):
            roll_avg[i] = np.mean(one_pixel[i: i+w])
        a, b, c = np.polyfit(wavs, roll_avg, deg=2)
        G = np.max(one_pixel) * 1e-3
        offset = 0
        return G, offset, a, b, c

    def wavelength_calibration_one_pixel(self, row, col):
        """
        returns needed calibration for one spatial pixel
        """
        print(f"row: {row}, col: {col}")
        sys.stdout.flush()
        osiris_data = self.osiris_data
        R = 4000
        wavs = osiris_data[0] * u.micron
        sky_model = np.zeros_like(wavs.value)
        cube = osiris_data[1]
        one_pixel = cube[:, row, col]
        fit_wrapper = lambda *p : self.const_offset_fitter(*p, one_pixel)
        p0, _ = curve_fit(fit_wrapper, wavs, one_pixel, p0=[0], xtol=self.frac_error)
        return (p0[0] * u.nm, p0)

    def relevant_OH_line_data():
        osiris_data = self.osiris_data
        OH_wavelengths, OH_intensity = self.OH_wavelengths, self.OH_intensity
        wavs = osiris_data[0] * u.micron
        wav_low, wav_high = np.where(OH_wavelengths >= wavs[0])[0][0], np.where(OH_wavelengths <= wavs[-1])[0][-1]
        relevant_OH = OH_wavelengths[wav_low:wav_high], OH_intensity[wav_low:wav_high]
        return relevant_OH

    def wavelength_calibration_one_pixel_wrapper(self, param):
        return self.wavelength_calibration_one_pixel(*param)[0]

    def wavelength_calibration_cube(self):
        # mp code
        osiris_data = self.osiris_data
        nz, nx, ny = osiris_data[1].shape
        row_inputs = np.reshape(np.array(list(range(nx)) * ny), (nx, ny), order = 'F')
        col_inputs = np.reshape(np.array(list(range(ny)) * nx), (nx, ny), order = 'C')
        params = np.reshape(np.dstack((row_inputs, col_inputs)), (nx * ny, 2))
        print(0)
        offsets = my_pool.map(self.wavelength_calibration_one_pixel_wrapper, params)
        # frac error of 1e-3 corresponds to 3 sig figs
        offsets = np.reshape(offset, (nx, ny))
        return offsets
    
# #         non mp code
#         nz, nx, ny = self.osiris_data[1].shape
#         offsets = np.zeros((nx, ny)) * u.nm
#         for row in range(nx):
#             for col in range(ny):
#                 offsets[row, col] = self.wavelength_calibration_one_pixel(row, col)[0]
#                 # frac error of 1e-3 corresponds to 3 sig figs
#         return offsets
