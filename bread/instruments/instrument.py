import os
import bread.utils as utils
from warnings import warn

class Instrument:
    def __init__(self, ins_type):
        self.ins_type = ins_type
        assert self.check_instrument_type(), "Instrument Not Implemented Yet"
        self.wavelengths = None
        self.spaxel_cube = None
        self.noise_cube = None
        self.bad_pixel_cube = None
        self.bary_RV = None
        
    def check_instrument_type(self):
        implemented = self.instruments_implemented()
        if self.ins_type in implemented:
            return True
        print("Instruments Implemented Yet:", implemented)
        return False
    
    def instruments_implemented(self):
        files = os.listdir(utils.file_directory(__file__))
        implemented = []
        for file in files:
            if ".py" in file and "instrument" not in file:
                implemented += [file[:file.index(".py")]]
        return implemented
    
    def manual_data_entry(self, wavelengths, spaxel_cube, noise_cube, bad_pixel_cube, bary_RV):
        warn("when feeding data manually, ensure correct units. wavelengths in microns, bary_RV in km/s")
        self.wavelengths = wavelengths
        self.spaxel_cube = spaxel_cube
        self.noise_cube = noise_cube
        self.bad_pixel_cube = bad_pixel_cube
        self.bary_RV = bary_RV # in km/s
        self.valid_data_check()
    
    def read_data(self, filename):
        print("Instruments Implemented Yet:", self.instruments_implemented())
        raise NotImplementedError(
            "Import derived class corresponding to your instrument. You are currently using the base class.")
        
    def valid_data_check(self):
        assert self.spaxel_cube.ndim == 3, "Spaxel Cube Data must be 3-dimensional"
        assert self.wavelengths.ndim == 1, "Wavelength Array must be 1-dimensional"
        assert self.spaxel_cube.shape[0] == self.wavelengths.shape[0], \
                        "Wavelength dimension of the spaxel data must be equal to size to wavelength array"
        assert self.noise_cube is None or self.noise_cube.shape == self.spaxel_cube.shape, \
                            "If present, noise cube must be of same size as spaxel data"
        assert self.bad_pixel_cube is None or self.bad_pixel_cube.shape == self.spaxel_cube.shape, \
                            "If present, bad pixel cube must be of same size as spaxel data"
     
    