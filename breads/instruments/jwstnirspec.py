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
from breads.calibration import SkyCalibration
import multiprocessing as mp
from itertools import repeat
import pandas as pd
import astropy

#NIRSPEC Wavelengths
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

class JWSTNirspec(Instrument):
    def __init__(self, filename=None):
        super().__init__('jwstnirspec')
        if filename is None:
            warning_text = "No data file provided. " + \
            "Please manually add data or use JWSTNirspec.read_data_file()"
            warn(warning_text)
        else:
            self.read_data_file(filename)

    def read_data_file(self, filename):
        """
        Read OSIRIS spectral cube, also checks validity at the end
        """
        with pyfits.open(filename) as hdulist:
            priheader = hdulist[0].header
            cube = hdulist[1].data
            noisecube = hdulist[2].data
            badpixcube = hdulist[3].data.astype(np.float)
            # plt.plot(cube[:,17,25]/np.nanmedian(cube[:,17,25]))
            # plt.plot(badpixcube[:,17,25])
            # for k in range(200):
            #     plt.imshow(badpixcube[k*10,::])
            #     plt.clim([0,1])
            #     plt.show()
            badpixcube[np.where(badpixcube!=0)] = np.nan
            badpixcube[np.where(np.isfinite(badpixcube))] = 1
            badpixcube[np.where(np.abs(cube)>=1e-6)] = np.nan
            # wvs=get_wavelen_values(hdulist[1].header)

        from spectral_cube import SpectralCube
        tmpcube=SpectralCube.read(filename,hdu=1)
        wvs=tmpcube.spectral_axis
        wvs=np.array(wvs)


        nz, ny, nx = cube.shape
        self.wavelengths = np.tile(wvs[:,None,None],(1,ny,nx))
        self.data = cube
        self.noise = noisecube
        self.bad_pixels = badpixcube
        self.bary_RV = 0
        self.R = 2700

        self.priheader = priheader

        print(priheader["DETECTOR"].strip())
        if priheader["DETECTOR"].strip() == 'NRS1':
            crop_min, crop_max = 10, 140
        elif priheader["DETECTOR"].strip() == 'NRS2':
            crop_min, crop_max = 160, 150
        print(crop_min, crop_max)
        self.bad_pixels[0:crop_min, :, :] = np.nan
        self.bad_pixels[nz - crop_max::, :, :] = np.nan
        
        self.valid_data_check()

    def trim_data(self, trim):
        if trim <= 0:
            return
        nz, nx, ny = self.data.shape
        self.bad_pixels[:trim] = np.nan
        self.bad_pixels[nz-trim:] = np.nan

    # def remove_bad_pixels(self, chunks=20, mypool=None, med_spec=None, nan_mask_boxsize=3, w=5,threshold=3):
    #     if med_spec == "transmission" or med_spec == "pair subtraction":
    #         img_mean = np.nanmean(self.data, axis=0)
    #         x, y = np.unravel_index(np.nanargmax(img_mean), img_mean.shape)
    #         med_spec = np.nanmedian(self.data[:,x-w:x+w, y-w:y+w], axis=(1,2))
    #     elif med_spec == "default":
    #         med_spec = None
    #     new_badpixcube, new_cube, res = \
    #         utils.findbadpix(self.data, self.noise, self.bad_pixels, chunks, mypool, med_spec, nan_mask_boxsize,threshold)
    #     self.bad_pixels = new_badpixcube
    #     self.data = new_cube
    #     utils.clean_nans(self.data)
    #     return res


    def remove_bad_pixels(self, chunks=20, mypool=None, med_spec=None, nan_mask_boxsize=3, w=5, \
                          num_threads = 16, wid_mov=None,threshold=3):
        # if med_spec == "transmission" or med_spec == "pair subtraction":
        #     img_mean = np.nanmean(self.data, axis=0)
        #     x, y = np.unravel_index(np.nanargmax(img_mean), img_mean.shape)
        #     med_spec = np.nanmedian(self.data[:,x-w:x+w, y-w:y+w], axis=(1,2))
        # elif med_spec == "default":
        #     med_spec = None
        new_badpixcube, new_cube, res = \
            utils.findbadpix(self.data, self.noise, self.bad_pixels, chunks, mypool, med_spec, nan_mask_boxsize,threshold=threshold)
        self.bad_pixels = new_badpixcube
        self.data = new_cube
        try:
            temp = self.continuum
        except:
            nz, ny, nx = self.data.shape
            my_pool = mp.Pool(processes=num_threads)
            if wid_mov is None:
                wid_mov = 10#nz // 10
            args = []
            for i in range(ny):
                for j in range(nx):
                    args += [self.data[:, i, j]]
            output = my_pool.map(set_continnuum, zip(args, repeat(wid_mov)))
            self.continuum = np.zeros((nz, ny, nx))
            for i in range(ny):
                for j in range(nx):
                    self.continuum[:, i, j] = output[(i*nx+j)]

        # utils.mask_bleeding(self)
        # utils.clean_nans(self.data)
        return res

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
