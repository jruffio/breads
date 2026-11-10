import os
from warnings import warn

import breads.utils as utils


class Instrument:
    """
    Class representing the data from a specific instrument.
    This class is a template for instrument classes, and can be used to define custom data objects for instruments that might not be supported otherwise.
    """
    def __init__(self, ins_type="custom", verbose=True):
        """
        Create an empty instance of the Instrument class.

        Parameters
        ----------
        ins_type : str
            A string describing the type of instrument. This is just for bookkeeping and does not affect the functionality of the class. Default is "custom".
        verbose : bool
            If True, the class will print out information about the data and the processing steps. Default is True.

        """
        self.ins_type = ins_type

        self.wavelengths = None
        self.xcoords = None
        self.ycoords = None
        self.data = None
        self.noise = None
        self.bad_pixels = None
        self.bary_RV = None
        self.refpos = None

        self.verbose = verbose

    def manual_data_entry(self, wavelengths=None, xcoords=None, ycoords=None, data=None, noise=None, bad_pixels=None, bary_RV=None):
        """
        Manual entry of data into the Instrument class.

        Note: In most cases, the format of most arrays (data, wavelengths, xcoords, ycoords, etc.) is left for the user to decide, but it should be consistent with the forward model function used.
        This means that certain instrument classes might only work with certain forward models.
        Although, the data, noise, and bad pixel arrays should all be of the same shape.

        Parameters
        ----------
        wavelengths : array-like
            Wavelength of the data. Typically assumed to be in microns.
        xcoords : array-like
            X coordinates of the data typically in arcsec. This is optional.
        ycoords : array-like
            Y coordinates of the data  typically in arcsec. This is optional.
        data : array-like
            The data to be analyzed.
        noise : array-like
            The noise (sigma; standard deviation) of the data.
        bad_pixels : array-like
            A boolean array indicating the bad pixels in the data. True (or 1) for good pixels, False (or 0) for bad pixels.
        bary_RV : float
            The barycentric radial velocity of the observer at the time of observation, in km/s.
        """
        self.wavelengths = wavelengths
        self.xcoords = xcoords
        self.ycoords = ycoords
        self.data = data
        self.noise = noise
        self.bad_pixels = bad_pixels
        self.bary_RV = bary_RV # in km/s
        self.valid_data_check()

    def valid_data_check(self):
        assert self.noise is None or self.noise.shape == self.data.shape, \
                            "If present, noise must be of same shape as spaxel data"
        assert self.bad_pixels is None or self.bad_pixels.shape == self.data.shape, \
                            "If present, bad pixel must be of same shape as spaxel data"

    def broaden(self, wvs,spectrum):
        return None

    def remove_bad_pixels(self, chunks=20, mypool=None, med_spec=None, nan_mask_boxsize=3):
        return None
    