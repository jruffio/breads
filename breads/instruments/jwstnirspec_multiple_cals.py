from breads.instruments.jwstnirspec_cal import JWSTNirspec_cal
from warnings import warn
import numpy as np
from copy import copy

class JWSTNirspec_multiple_cals(JWSTNirspec_cal):
    def __init__(self, dataobj_list=None, verbose=True):
        """JWST NIRSpec 2D calibrated data, combined from multiple files

        This class is used to merge point cloud data from multiple images,
        typically from a series of spatially dithered exposures on a target.

        Parameters
        ----------
        dataobj_list : list of JWSTNirspec_cal objects
            Datasets to combine
        verbose : bool
            Be more verbose in text output?
        """
        super().__init__(verbose=verbose)

        if len(dataobj_list) == 0:
            warning_text = "No data object provided provided. " + \
                           "Please manually add data or use JWSTNirspec_multiple_cals.combine_dataobj_list()"
            warn(warning_text)
            # TODO consider making this an Exception error rather than just a warning?
            # Is there a compelling use case to allow manually adding data after initializing the class?
        else:
            self.combine_dataobj_list(dataobj_list)


    def combine_dataobj_list(self, dataobj_list):
        """ Combine the data from multiple data objects

        This concatenates the values from many attributes into a single overall combined dataset
        """
        self.ins_type = dataobj_list[0].ins_type
        self.coords = dataobj_list[0].coords
        self.R = dataobj_list[0].R
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

        self.data = np.concatenate([copy(dataobj.data) for dataobj in dataobj_list],axis=0)
        self.noise = np.concatenate([copy(dataobj.noise) for dataobj in dataobj_list],axis=0)
        self.bad_pixels = np.concatenate([copy(dataobj.bad_pixels) for dataobj in dataobj_list],axis=0)
        self.wavelengths = np.concatenate([copy(dataobj.wavelengths) for dataobj in dataobj_list],axis=0)
        self.dra_as_array = np.concatenate([copy(dataobj.dra_as_array) for dataobj in dataobj_list],axis=0)
        self.ddec_as_array = np.concatenate([copy(dataobj.ddec_as_array) for dataobj in dataobj_list],axis=0)
        self.area2d = np.concatenate([copy(dataobj.area2d) for dataobj in dataobj_list],axis=0)
        N_traces = np.size(np.unique(dataobj.trace_id_map[np.where(np.isfinite(dataobj.trace_id_map))]))
        self.trace_id_map = np.concatenate([dataobj.trace_id_map+dataobj_id*N_traces for dataobj_id,dataobj in enumerate(dataobj_list)],axis=0)
