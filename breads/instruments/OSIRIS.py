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
from copy import copy, deepcopy
from breads.utils import broaden
from breads.calibration import SkyCalibration
import multiprocessing as mp
from itertools import repeat
import pandas as pd

class OSIRIS(Instrument):
    def __init__(self, filename=None, skip_baryrv=False):
        super().__init__('OSIRIS')
        if filename is None:
            warning_text = "No data file provided. " + \
            "Please manually add data using OSIRIS.manual_data_entry() or add data using OSIRIS.read_data_file()"
            warn(warning_text)
        else:
            self.read_data_file(filename, skip_baryrv=skip_baryrv)
        self.calibrated = False
        self.refpos = None

    def read_data_file(self, filename, skip_baryrv=False):
        """
        Read OSIRIS spectral cube, also checks validity at the end
        """
        with pyfits.open(filename) as hdulist:
            prihdr = hdulist[0].header
            curr_mjdobs = prihdr["MJD-OBS"]
            cube = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
            noisecube = np.rollaxis(np.rollaxis(hdulist[1].data,2),2,1)
            # cube = np.moveaxis(cube,0,2)
            badpixcube = np.rollaxis(np.rollaxis(hdulist[2].data,2),2,1)
            if "bb" in hdulist[0].header["SFWNAME"]:
                cube = return_64x19(cube)
                noisecube = return_64x19(noisecube)
                badpixcube = return_64x19(badpixcube)
            # badpixcube = np.moveaxis(badpixcube,0,2)
            badpixcube = badpixcube.astype(dtype=ctypes.c_double)
            badpixcube[np.where(badpixcube!=0)] = 1
            badpixcube[np.where(badpixcube==0)] = np.nan

        nz, nx, ny = cube.shape
        init_wv = prihdr["CRVAL1"]/1000. # wv for first slice in mum
        dwv = prihdr["CDELT1"]/1000. # wv interval between 2 slices in mum
        wvs=np.linspace(init_wv,init_wv+dwv*nz,nz,endpoint=False)

        if not skip_baryrv:
            keck = EarthLocation.from_geodetic(lat=19.8283 * u.deg, lon=-155.4783 * u.deg, height=4160 * u.m)
            sc = SkyCoord(float(prihdr["RA"]) * u.deg, float(prihdr["DEC"]) * u.deg)
            barycorr = sc.radial_velocity_correction(obstime=Time(float(prihdr["MJD-OBS"]), format="mjd", scale="utc"),
                                                     location=keck)
            baryrv = barycorr.to(u.km / u.s).value
        else:
            baryrv = None

        self.wavelengths = np.zeros_like(cube)
        for i in range(nx):
            for j in range(ny):
                self.wavelengths[:, i, j] = wvs
        self.read_wavelengths = wvs
        self.data = cube
        self.noise = noisecube
        self.bad_pixels = badpixcube
        self.bary_RV = baryrv
        self.R = 4000
        
        self.valid_data_check()

    def trim_data(self, trim):
        if trim <= 0:
            return
        nz, nx, ny = self.data.shape
        self.bad_pixels[:trim] = np.nan
        self.bad_pixels[nz-trim:] = np.nan

    def remove_bad_pixels(self, chunks=20, mypool=None, med_spec=None, nan_mask_boxsize=3, w=5, \
        num_threads = 16, wid_mov=None):
        if med_spec == "transmission" or med_spec == "pair subtraction":
            img_mean = np.nanmean(self.data, axis=0)
            x, y = np.unravel_index(np.nanargmax(img_mean), img_mean.shape)
            med_spec = np.nanmedian(self.data[:,x-w:x+w, y-w:y+w], axis=(1,2))
        elif med_spec == "default":
            med_spec = None
        new_badpixcube, new_cube, res = \
            utils.findbadpix(self.data, self.noise, self.bad_pixels, chunks, mypool, med_spec, nan_mask_boxsize)
        self.bad_pixels = new_badpixcube
        self.data = new_cube
        try:
            temp = self.continuum
        except:
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
            
        utils.mask_bleeding(self)
        utils.clean_nans(self.data)
        return res

    def crop_image(self, x_range, y_range):
        self.data = self.data[:, x_range[0]:x_range[1], y_range[0]:y_range[1]]
        self.wavelengths = self.wavelengths[:, x_range[0]:x_range[1], y_range[0]:y_range[1]]
        self.noise = self.noise[:, x_range[0]:x_range[1], y_range[0]:y_range[1]]
        self.bad_pixels = self.data[:, x_range[0]:x_range[1], y_range[0]:y_range[1]]

    def set_reference_position(self, value):
        if type(value) is tuple:
            self.refpos = value
        if type(value) is str:
            with pyfits.open(value) as hdulist:
                mu_x = hdulist[3].data
                mu_y = hdulist[4].data
            self.refpos = (np.nanmedian(mu_x), np.nanmedian(mu_y))


    def calibrate(self, SkyCalibObj, allowed_range=(-1, 1)):
        """
        SkyCalibObj can be either an object of an SkyCalibration object, or
        the path+filename of the fits file that SkyCalibration generates.
        """
        if self.calibrated:
            warn("Overwriting previously done calibration")
        nz, nx, ny = self.data.shape
        if isinstance(SkyCalibObj, SkyCalibration):
            off0 = SkyCalibObj.fit_values[:, :, 0]
            off0 = SkyCalibObj.fit_values[:, :, 1]
        elif type(SkyCalibObj) is str:
            with pyfits.open(SkyCalibObj) as hdulist:
                off0 = hdulist[1].data
                off1 = hdulist[2].data
        else:
            warn("Invalid Input, run help(osiris.calibrate) for info.")
            return
        utils.clean_nans(off0, allowed_range=allowed_range)
        utils.clean_nans(off1)
        for i in range(nx):
            for j in range(ny):
                self.wavelengths[:, i, j] = \
                utils.corrected_wavelengths(self, off0[i, j], off1[i, j], False)

        utils.clean_nans(self.wavelengths)
        self.calibrated = True

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

    def set_noise(self, method="sqrt_cont", num_threads = 16, wid_mov=None, noise_floor=True):
        try:
            temp = self.continuum
        except:
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
            if noise_floor:
                noise_floor = get_noise_floor(self)
                where_below_thresh = np.where(self.noise < noise_floor)
                self.noise[where_below_thresh] = noise_floor[where_below_thresh]

def get_noise_floor(dataobj):
    noise_floor = np.zeros(dataobj.data.shape)
    for sliceid in range(dataobj.data.shape[0]):
        arr = (dataobj.data[sliceid,:,:]-dataobj.continuum[sliceid,:,:]) * dataobj.bad_pixels[sliceid,:,:]
        arr[dataobj.continuum[sliceid,:,:] > np.nanmedian(dataobj.continuum[sliceid,:,:]* dataobj.bad_pixels[sliceid,:,:])] = np.nan
        noise_floor[sliceid,:,:] = np.nanstd(arr)
    return noise_floor


def set_continnuum(args):
    data, window = args
    tmp = np.array(pd.DataFrame(np.concatenate([data, data[::-1]], axis=0)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))
    myvec_cp_lpf = np.array(pd.DataFrame(tmp).rolling(window=window, center=True).median().interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[0:np.size(data), 0]
    data = data[::-1]
    tmp = np.array(pd.DataFrame(np.concatenate([data, data[::-1]], axis=0)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))
    myvec_cp_lpf_r = np.array(pd.DataFrame(tmp).rolling(window=window, center=True).median().interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[0:np.size(data), 0]
    return (myvec_cp_lpf + myvec_cp_lpf_r[::-1]) / 2

def return_64x19(cube):
    """
    Hacky function. The dimensions of an OSIRIS cube are not always what it should be (64x19) in broadband. I think the
    problem is in Hbb. So this aligns everything by default.
    But only works for broadband filters, not the narrowbands ones like Kn3 or Kn5.
    """
    # cube should be nz,ny,nx
    if np.size(cube.shape) == 3:
        _, ny, nx = cube.shape
    else:
        ny, nx = cube.shape
    onesmask = np.ones((64, 19))
    if (ny != 64 or nx != 19):
        mask = copy(cube).astype(np.float)
        mask[np.where(mask == 0)] = np.nan
        mask[np.where(np.isfinite(mask))] = 1
        if np.size(cube.shape) == 3:
            im = np.nansum(mask, axis=0)
        else:
            im = mask
        ccmap = np.zeros((3, 3))
        for dk in range(3):
            for dl in range(3):
                ccmap[dk, dl] = np.nansum(im[dk:np.min([dk + 64, ny]), dl:np.min([dl + 19, nx])]
                                          * onesmask[0:(np.min([dk + 64, ny]) - dk),
                                            0:(np.min([dl + 19, nx]) - dl)])
        dk, dl = np.unravel_index(np.nanargmax(ccmap), ccmap.shape)
        if np.size(cube.shape) == 3:
            return cube[:, dk:(dk + 64), dl:(dl + 19)]
        else:
            return cube[dk:(dk + 64), dl:(dl + 19)]
    else:
        return cube
