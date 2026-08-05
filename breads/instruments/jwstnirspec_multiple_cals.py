from breads.instruments.jwstnirspec_cal import JWSTNirspec_cal
from warnings import warn
import numpy as np
from copy import copy
import os
import fnmatch
import matplotlib.pyplot as plt

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
        self.verbose = verbose

        if len(dataobj_list) == 0:
            warning_text = "No data object provided provided. " + \
                           "Please manually add data or use JWSTNirspec_multiple_cals.combine_dataobj_list()"
            warn(warning_text)
            # TODO consider making this an Exception error rather than just a warning?
            # Is there a compelling use case to allow manually adding data after initializing the class?
        else:
            self.ifu_name = 'nirspec'
            self.combine_dataobj_list(dataobj_list)
            self.bary_RV = 0
            self.refpos = None


    def combine_dataobj_list(self, dataobj_list):
        """ Combine the data from multiple data objects

        This concatenates the values from many attributes into a single overall combined dataset
        """
        self.breads_header = dataobj_list[0].breads_header
        self.R = dataobj_list[0].R
        self.opmode = dataobj_list[0].opmode
        # todo: delete commented lines?
        # if hasattr(self, "wv_ref"):
        #     self.wv_ref = dataobj_list[0].wv_ref
        self.east2V2_deg = dataobj_list[0].east2V2_deg

        splitbasename = os.path.basename(dataobj_list[0].filename).split("_")
        self.filename = os.path.join(os.path.dirname(dataobj_list[0].filename),splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[3] +".fits")

        self.priheader = dataobj_list[0].priheader
        self.extheader = dataobj_list[0].extheader

        self.utils_dir = dataobj_list[0].utils_dir
        self.crds_dir = dataobj_list[0].crds_dir
        if hasattr(dataobj_list[0], "wv_sampling"):
            self.wv_sampling = dataobj_list[0].wv_sampling
        if hasattr(dataobj_list[0], "wv_nodes"):
            self.wv_nodes = dataobj_list[0].wv_nodes

        self.default_filenames = {}
        basename = os.path.basename(self.filename)
        self.default_filenames["compute_med_filt_badpix"] = \
            os.path.join(self.utils_dir, basename.replace(".fits", "_roughbadpix_combined.fits"))
        self.default_filenames["compute_coordinates_arrays"] = \
            os.path.join(self.utils_dir, basename.replace(".fits", "_relcoords_combined.fits"))
        self.default_filenames["compute_webbpsf_model"] = \
            os.path.join(self.utils_dir,basename.replace(".fits", "_webbpsf.fits"))
        self.default_filenames["compute_quick_webbpsf_model"] = \
            os.path.join(self.utils_dir,basename.replace(".fits", "_quick_webbpsf.fits"))
        self.default_filenames["compute_new_coords_from_webbPSFfit"] = \
            os.path.join(self.utils_dir, basename.replace(".fits", "_newcen_wpsf_combined.fits"))
        self.default_filenames["compute_starspectrum_contnorm"] = \
            os.path.join(self.utils_dir, basename.replace(".fits", "_starspec_contnorm_combined.fits"))
        self.default_filenames["compute_starspectrum_contnorm_3dspline"] = \
            os.path.join(self.utils_dir, basename.replace(".fits", "_starspec_contnorm_3Dspline.fits"))
        self.default_filenames["compute_starsubtraction"] = \
            os.path.join(self.utils_dir, basename.replace(".fits", "_starsub_combined.fits"))
        self.default_filenames["compute_starsubtraction_3dspline"] = \
            os.path.join(self.utils_dir, basename.replace(".fits", "_starsub_combined_3Dspline.fits"))
        self.default_filenames["compute_advanced_badpix"] = \
            os.path.join(self.utils_dir, basename.replace(".fits", "_starsub_combined.fits"))
        self.default_filenames["compute_interpdata_regwvs"] = \
            os.path.join(self.utils_dir, basename.replace(".fits", "_regwvs_combined.fits"))



        self.filelist = []
        self.priheader_list = []
        self.extheader_list = []
        splitbasename = os.path.basename(dataobj_list[0].filename).split("_")
        prefix0 = splitbasename[0] + "_" + splitbasename[1] + "_*_" + splitbasename[3]
        for dataobj in dataobj_list:
            if not fnmatch.fnmatch(os.path.basename(dataobj.filename), prefix0+"*.fits"):
                raise Exception(f"All files given to jwstnirspec_multiple_cals should be from the same dither sequence, detector, and grating/filter pair. {dataobj.filename} does not match {prefix0}.")
            self.filelist.append(dataobj.filename)
            self.priheader_list.append(dataobj.priheader)
            self.extheader_list.append(dataobj.extheader)

        self.data = np.concatenate([copy(dataobj.data) for dataobj in dataobj_list], axis=0)
        self.noise = np.concatenate([copy(dataobj.noise) for dataobj in dataobj_list], axis=0)
        self.bad_pixels = np.concatenate([copy(dataobj.bad_pixels) for dataobj in dataobj_list], axis=0)
        self.wavelengths = np.concatenate([copy(dataobj.wavelengths) for dataobj in dataobj_list], axis=0)
        self.x = np.concatenate([copy(dataobj.x) for dataobj in dataobj_list], axis=0)
        self.y = np.concatenate([copy(dataobj.y) for dataobj in dataobj_list], axis=0)
        self.area2d = np.concatenate([copy(dataobj.area2d) for dataobj in dataobj_list], axis=0)
        N_traces = np.size(np.unique(dataobj_list[0].trace_id_map[np.where(np.isfinite(dataobj_list[0].trace_id_map))]))
        self.trace_id_map = np.concatenate([dataobj.trace_id_map+dataobj_id*N_traces for dataobj_id, dataobj in enumerate(dataobj_list)], axis=0)

