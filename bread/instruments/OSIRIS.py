from instrument import Instrument
import bread.utils as utils

class OSIRIS(Instrument):
    def __init__(self, ins_type):
        super(ins_type)
        
    def __init__(self, filename):
        super(ins_type)
        self.read_data(filename)
        self.check_valid_data()
        
    def __init__(self, wavelengths, spaxel_cube, noise_cube, bad_pixel_cube, bary_RV):
        self.wavelengths = wavelengths
        self.spaxel_cube = spaxel_cube
        self.noise_cube = noise_cube
        self.bad_pixel_cube = bad_pixel_cube
        self.bary_RV = bary_RV
        self.valid_data_check()
        
    def valid_data_check(self):

    def read_data(self, filename):
        """
        Read OSIRIS spectral cube
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
        
    def valid_check_data(self):
        pass
        
    