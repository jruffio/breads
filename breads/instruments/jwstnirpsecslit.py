from matplotlib.pyplot import axis
import matplotlib.pyplot as plt
from breads.instruments.instrument import Instrument
import breads.utils as utils
from warnings import warn
import astropy.io.fits as pyfits
import numpy as np
import ctypes
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.time import Time
from copy import copy
from breads.utils import broaden
from breads.utils import get_spline_model,_task_findbadpix
import multiprocessing as mp
from itertools import repeat
import pandas as pd
import astropy
import os

#NIRPSEC Wavelengths
def get_wavelen_values(header, wavelen_axis=3):
    """Get array of wavelength values in microns, via WCS.
    Works on JWST NIRSpec cubes but should be general across other instruments too
    Returns wavelengths in microns
    """
    wcs = astropy.wcs.WCS(header)
    pix_coords = np.zeros((header[f'NAXIS{wavelen_axis}'], 3))
    pix_coords[:, 2] = np.arange(header[f'NAXIS{wavelen_axis}'])
    wavelens = wcs.wcs_pix2world(pix_coords,0)[:, 2]*1e6
    return wavelens

class jwstnirpsecslit(Instrument):
    def __init__(self, filename=None,data_type="DRP"):
        super().__init__('jwstnirpsec')
        if filename is None:
            warning_text = "No data file provided. " + \
            "Please manually add data or use jwstnirpsec.read_data_file()"
            warn(warning_text)
        else:
            self.read_data_file(filename,data_type=data_type)

    def read_data_file(self, filename,data_type="DRP"):
        """
        Read OSIRIS spectral cube, also checks validity at the end
        """

        if data_type == "DRP":
            data = pyfits.getdata(filename)
            priheader = pyfits.getheader(filename, 0)
            wavelength = data['wavelength']
            flux  = data['flux']
            error  = data['FLUX_ERROR']
            badpix  = data["DQ"]
            wherebad = np.where((badpix % 2) == 1)
            badpix = badpix.astype(np.float)
            badpix[wherebad] = np.nan
            badpix[0:30] = np.nan
            badpix[3240::] = np.nan

            crop4um = np.argmin(np.abs(wavelength-4.0))

            self.wavelengths = wavelength[crop4um::]
            self.data = flux[crop4um::]
            self.noise = error[crop4um::]
            self.bad_pixels = badpix[crop4um::]
        elif data_type == "ETC":
            noise_filename = os.path.join(filename,"lineplot","lineplot_extracted_noise.fits")
            hdulist = pyfits.open(noise_filename)
            data = hdulist[1].data
            self.noise = np.array([fl for wv, fl in data])#*300
            flux_filename = os.path.join(filename,"lineplot","lineplot_extracted_flux.fits")
            hdulist = pyfits.open(flux_filename)
            data = hdulist[1].data
            self.wavelengths = np.array([wv for wv, fl in data])
            self.data = np.array([fl for wv, fl in data]) + self.noise*np.random.randn(np.size(self.noise))
            self.bad_pixels = np.ones(self.data.shape)
            priheader = hdulist[0].header
            self.bad_pixels[0:5] = np.nan
            self.bad_pixels[3000::] = np.nan


        self.bary_RV = 0
        self.R = 2700

        self.priheader = priheader

        self.valid_data_check()

    def trim_data(self, trim):
        if trim <= 0:
            return
        nz, nx, ny = self.data.shape
        self.bad_pixels[:trim] = np.nan
        self.bad_pixels[nz-trim:] = np.nan


    def remove_bad_pixels(self, chunks=20, mypool=None, med_spec=None, nan_mask_boxsize=3, w=5, \
                          num_threads = 16, wid_mov=None,threshold=3):

        nz = np.size(self.data)

        x = np.arange(nz)
        x_knots = x[np.linspace(0,nz-1,chunks+1,endpoint=True).astype(np.int)]
        M_spline = get_spline_model(x_knots,x,spline_degree=3)

        out_data,out_badpix,out_res = _task_findbadpix((self.data[:,None],self.noise[:,None],self.bad_pixels[:,None],med_spec,M_spline,threshold))
        out_data,out_badpix,out_res = out_data[:,0],out_badpix[:,0],out_res[:,0]

        continuum = set_continnuum((out_data,50))
        wherebad = np.where(np.isnan(out_badpix))
        self.data[wherebad] = continuum[wherebad]
        self.bad_pixels = out_badpix

        # plt.plot(out_data/np.nanmax(out_data))
        # plt.plot(out_res/np.nanmax(out_data))
        # plt.plot(out_badpix)
        # plt.show()

        return out_res

    def crop_image(self, x_range, y_range):
        self.data = self.data[:, x_range[0]:x_range[1], y_range[0]:y_range[1]]
        self.wavelengths = self.wavelengths[:, x_range[0]:x_range[1], y_range[0]:y_range[1]]
        self.noise = self.noise[:, x_range[0]:x_range[1], y_range[0]:y_range[1]]
        self.bad_pixels = self.data[:, x_range[0]:x_range[1], y_range[0]:y_range[1]]

    def broaden(self, wvs,spectrum, loc=None,mppool=None):
        """
        Broaden a spectrum to the resolution of this data object using the resolution attribute (self.R).
        LSF is assumed to be a 1D gaussian.
        The broadening is technically fiber dependent so you need to specify which fiber calibration to use.

        Args:
            wvs: Wavelength sampling of the spectrum to be broadened.
            spectrum: 1D spectrum to be broadened.
            loc: To be ignored. Could be used in the future to specify (x,y) position if field dependent resolution is
                available.
            mypool: Multiprocessing pool to parallelize the code. If None (default), non parallelization is applied.
                E.g. mppool = mp.Pool(processes=10) # 10 is the number processes

        Return:
            Broadened spectrum
        """
        return broaden(wvs, spectrum, self.R, mppool=mppool)

    def set_noise(self, method="sqrt_cont", num_threads = 16, wid_mov=None):
        nz, ny, nx = self.data.shape
        my_pool = mp.Pool(processes=num_threads)
        if wid_mov is None:
            wid_mov = nz // 10
        args = []
        for i in range(ny):
            for j in range(nx):
                args += [self.data[:, i, j]]
        output = my_pool.map(set_continnuum, zip(args, repeat(wid_mov)))
        self.continuum = np.zeros((nz, ny, nx))
        for i in range(ny):
            for j in range(nx):
                self.continuum[:, i, j] = output[(i*nx+j)]
        # self.continuum = np.reshape(self.continuum, (nz, ny, nx), order='F')
        if method == "sqrt_cont":
            self.noise = np.sqrt(np.abs(self.continuum))
        if method == "cont":
            self.noise = self.continuum

def set_continnuum(args):
    data, window = args
    tmp = np.array(pd.DataFrame(np.concatenate([data, data[::-1]], axis=0)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))
    myvec_cp_lpf = np.array(pd.DataFrame(tmp).rolling(window=window, center=True).median().interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[0:np.size(data), 0]
    return myvec_cp_lpf
