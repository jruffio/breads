from breads.instruments.jwstmiri_cal import JWSTMiri_cal, _fit_wpsf_task
from warnings import warn
import numpy as np
from copy import copy
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt


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
        self.ins_type = dataobj_list[0].ins_type
        self.coords = dataobj_list[0].coords
        self.R = dataobj_list[0].R
        self.pixelscale = dataobj_list[0].pixelscale
        self.data_unit = dataobj_list[0].data_unit
        self.opmode = dataobj_list[0].opmode
        if hasattr(self, "wv_ref"):
            self.wv_ref = dataobj_list[0].wv_ref
        if hasattr(dataobj_list[0], "webbpsf_wv0"):
            self.webbpsf_wv0 = dataobj_list[0].webbpsf_wv0
            self.webbpsf_im = dataobj_list[0].webbpsf_im
            self.webbpsf_X = dataobj_list[0].webbpsf_X
            self.webbpsf_Y = dataobj_list[0].webbpsf_Y
            self.east2V2_deg = dataobj_list[0].east2V2_deg
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

    def compute_new_coords_from_webbPSFfit(self, save_utils=False, IWA=None, OWA=None):
        """ Update coordinates after fitting a webbPSF at the median wavelength of the data.
               This is the wavelength at which the WebbPSF was saved in the class.

               It does not interpolate the data at that wavelength, only grabs the closest pixel.

               Parameters
               ----------
               save_utils : bool
                   Save in the utils directory

               Returns
               -------

               """

        if IWA is None:
            IWA = 0
        if OWA is None:
            OWA = 1.5

        # rough centroid fit
        fit_cen, fit_angle = True, False
        linear_interp = True
        init_paras = np.array([0, 0])

        mask = np.copy(self.bad_pixels).transpose()
        data = np.copy(self.data).transpose()
        noise = np.copy(self.noise).transpose()
        dra_as_array = np.copy(self.dra_as_array).transpose()
        ddec_as_array = np.copy(self.ddec_as_array).transpose()
        diff_wv_map = np.abs(self.wavelengths - self.webbpsf_wv0).transpose()

        mask[np.where(diff_wv_map > np.nanmedian(self.wavelengths) / self.R)] = np.nan
        allnans_rows = np.where(np.nansum(np.isfinite(diff_wv_map), axis=1) == 0)
        diff_wv_map[allnans_rows, :] = 0
        argmin_ids = np.nanargmin(diff_wv_map, axis=1)
        print(argmin_ids)

        paras = linear_interp, self.webbpsf_im, self.webbpsf_X, self.webbpsf_Y, self.east2V2_deg, True, \
            dra_as_array[:, argmin_ids], \
            ddec_as_array[:, argmin_ids], \
            data[:, argmin_ids], \
            noise[:, argmin_ids], \
            mask[:, argmin_ids], \
            IWA, OWA, fit_cen, fit_angle, init_paras
        out, _ = _fit_wpsf_task(paras)
        ra_offset, dec_offset, angle_offset = out[0, 2::]

        print("Estimated ra and dec offset from combined webbPSF fit:", ra_offset, dec_offset)

        return ra_offset, dec_offset