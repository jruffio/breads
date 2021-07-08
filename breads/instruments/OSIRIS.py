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

class OSIRIS(Instrument):
    def __init__(self, filename=None, skip_baryrv=False):
        super().__init__('OSIRIS')
        if filename is None:
            warning_text = "No data file provided. " + \
            "Please manually add data using OSIRIS.manual_data_entry() or add data using OSIRIS.read_data_file()"
            warn(warning_text)
        else:
            self.read_data_file(filename, skip_baryrv=skip_baryrv)

    def read_data_file(self, filename, skip_baryrv=False):
        """
        Read OSIRIS spectral cube, also checks validity at the end
        """
        with pyfits.open(filename) as hdulist:
            prihdr = hdulist[0].header
            curr_mjdobs = prihdr["MJD-OBS"]
            cube = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
            cube = return_64x19(cube)
            noisecube = np.rollaxis(np.rollaxis(hdulist[1].data,2),2,1)
            noisecube = return_64x19(noisecube)
            # cube = np.moveaxis(cube,0,2)
            badpixcube = np.rollaxis(np.rollaxis(hdulist[2].data,2),2,1)
            badpixcube = return_64x19(badpixcube)
            # badpixcube = np.moveaxis(badpixcube,0,2)
            badpixcube = badpixcube.astype(dtype=ctypes.c_double)
            badpixcube[np.where(badpixcube!=0)] = 1
            badpixcube[np.where(badpixcube==0)] = np.nan

        nz,ny,nx = cube.shape
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

        self.wavelengths = wvs
        self.data = cube
        self.noise = noisecube
        self.bad_pixels = badpixcube
        self.bary_RV = baryrv
        
        self.valid_data_check()

def return_64x19(cube):
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
