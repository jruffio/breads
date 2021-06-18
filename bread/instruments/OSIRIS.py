from bread.instruments.instrument import Instrument
import bread.utils as utils
from warnings import warn
import astropy.io.fits as pyfits
import numpy as np
import ctypes
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.time import Time

class OSIRIS(Instrument):
    def __init__(self):
        super().__init__('OSIRIS')
        
    def manual_data_entry(self, wavelengths, spaxel_cube, noise_cube, bad_pixel_cube, bary_RV):
        warn("when feeding data manually, ensure correct units")
        self.wavelengths = wavelengths
        self.spaxel_cube = spaxel_cube
        self.noise_cube = noise_cube
        self.bad_pixel_cube = bad_pixel_cube
        self.bary_RV = bary_RV # in km/s
        self.valid_data_check()

    def read_data_file(self, filename, skip_baryrv=False):
        """
        Read OSIRIS spectral cube, also checks validity at the end
        """
        with pyfits.open(filename) as hdulist:
            prihdr = hdulist[0].header
            curr_mjdobs = prihdr["MJD-OBS"]
            cube = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
            cube = utils.return_64x19(cube)
            noisecube = np.rollaxis(np.rollaxis(hdulist[1].data,2),2,1)
            noisecube = utils.return_64x19(noisecube)
            # cube = np.moveaxis(cube,0,2)
            badpixcube = np.rollaxis(np.rollaxis(hdulist[2].data,2),2,1)
            badpixcube = utils.return_64x19(badpixcube)
            # badpixcube = np.moveaxis(badpixcube,0,2)
            badpixcube = badpixcube.astype(dtype=ctypes.c_double)
            badpixcube[np.where(badpixcube==0)] = np.nan
            badpixcube[np.where(badpixcube!=0)] = 1

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
        self.spaxel_cube = cube
        self.noise_cube = noisecube
        self.bad_pixel_cube = badpixcube
        self.bary_RV = baryrv
        
        self.valid_check_data()
        
    def valid_check_data(self):
        assert self.spaxel_cube.ndim == 3, "Spaxel Cube Data must be 3-dimensional"
        assert self.wavelengths.ndim == 1, "Wavelength Array must be 1-dimensional"
        assert self.spaxel_cube.shape[0] == self.wavelengths.shape[0], \
                        "Wavelength dimension of the spaxel data must be equal to size to wavelength array"
        assert self.noise_cube is None or self.noise_cube.shape == self.spaxel_cube.shape, \
                            "If present, noise cube must be of same size as spaxel data"
        assert self.bad_pixel_cube is None or self.bad_pixel_cube.shape == self.spaxel_cube.shape, \
                            "If present, bad pixel cube must be of same size as spaxel data"
        
    