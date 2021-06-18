import os
import bread.utils as utils

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
    
    def read_data(self, filename):
        print("Instruments Implemented Yet:", self.instruments_implemented())
        raise NotImplementedError(
            "Import derived class corresponding to your instrument. You are currently using the base class.")
     
    