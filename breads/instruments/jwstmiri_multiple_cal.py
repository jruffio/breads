from breads.instruments.jwstmiri_cal import JWSTMiri_cal
from warnings import warn
import numpy as np
from copy import copy

class JWSTMiri_multiple_cals(JWSTMiri_cal):
    def __init__(self, dataobj_list=None,verbose=True):
        """JWST NIRSpec 2D calibrated data.
        test


        Parameters
        ----------
        dataobj_list
        verbose
        """
        super().__init__(verbose=verbose)

        if len(dataobj_list) == 0:
            warning_text = "No data object provided provided. " + \
                           "Please manually add data or use JWSTMiri_multiple_cals.combine_dataobj_list()"
            warn(warning_text)
        else:
            self.combine_dataobj_list(dataobj_list)

    def combine_dataobj_list(self, dataobj_list):
        print("DEBUG intializing combine_dataobj_list")
        self.ins_type = dataobj_list[0].ins_type
        self.coords = dataobj_list[0].coords
        self.R = dataobj_list[0].R
        self.pixelscale = dataobj_list[0].pixelscale
        self.data_unit = dataobj_list[0].data_unit
        self.opmode = dataobj_list[0].opmode
        if hasattr(self, "wv_ref"):
            self.wv_ref = dataobj_list[0].wv_ref
        self.east2V2_deg = dataobj_list[0].east2V2_deg
        self.default_filenames = {}
        for key,val in zip(dataobj_list[0].default_filenames.keys(),dataobj_list[0].default_filenames.values()):
            if key == "compute_quick_webbpsf_model" or key == "compute_webbpsf_model":
                self.default_filenames[key] = val
            else:
                self.default_filenames[key] = val.replace(".fits","_combined.fits")
        self.utils_dir = dataobj_list[0].utils_dir
        self.crds_dir = dataobj_list[0].crds_dir
        self.bary_RV = dataobj_list[0].bary_RV
        self.refpos = dataobj_list[0].refpos
        if hasattr(dataobj_list[0], "wv_sampling"):
            self.wv_sampling = dataobj_list[0].wv_sampling
        if hasattr(dataobj_list[0], "wvs_ori"):
            self.wvs_ori = dataobj_list[0].wvs_ori

        self.filename = dataobj_list[0].filename
        self.priheader = dataobj_list[0].priheader
        self.extheader = dataobj_list[0].extheader

        self.filelist = []
        self.priheader_list = []
        self.extheader_list = []

        for dataobj in dataobj_list:
            self.filelist.append(dataobj.filename)
            self.priheader_list.append(dataobj.priheader)
            self.extheader_list.append(dataobj.extheader)

        self.data = np.concatenate([copy(dataobj.data) for dataobj in dataobj_list], axis=1)
        self.noise = np.concatenate([copy(dataobj.noise) for dataobj in dataobj_list], axis=1)
        self.bad_pixels = np.concatenate([copy(dataobj.bad_pixels) for dataobj in dataobj_list], axis=1)
        self.wavelengths = np.concatenate([copy(dataobj.wavelengths) for dataobj in dataobj_list], axis=1)
        self.dra_as_array = np.concatenate([copy(dataobj.dra_as_array) for dataobj in dataobj_list], axis=1)
        self.ddec_as_array = np.concatenate([copy(dataobj.ddec_as_array) for dataobj in dataobj_list], axis=1)
        self.area2d = np.concatenate([copy(dataobj.area2d) for dataobj in dataobj_list], axis=1)
