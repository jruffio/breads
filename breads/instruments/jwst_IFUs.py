from abc import ABC, abstractmethod
import itertools
import os.path
import sys
from copy import copy, deepcopy
from glob import glob
from warnings import warn
import numpy as np
import pickle
from types import SimpleNamespace

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.patheffects as PathEffects
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from astropy import constants as const
from astropy import units as u
from astropy.stats import sigma_clip
from astropy.table import Table
from matplotlib.pyplot import tight_layout
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
from scipy.interpolate import interp1d
from scipy.optimize import minimize, curve_fit, lsq_linear
from scipy.signal import convolve2d
from scipy.stats import median_abs_deviation
from tqdm import tqdm

import breads.utils as utils
from breads.utils import broaden, rotate_coordinates, find_closest_leftnright_elements
from breads.utils import get_spline_model
from breads.utils import get_breads_commit
from breads.jwst_tools.plotting import save_cube_as_gif,point_cloud_interpolator_2d
from breads.jwst_tools.splines import fit_3dspline,normalize_rows,evaluate_3dspline_pointcloud
from breads.jwst_tools.spectra import combine_spectrum
from breads.jwst_tools.build_cube import rprint

import subprocess
from datetime import datetime, timezone


class JWST_IFUs(ABC):
    def __init__(self, filename=None, utils_dir=None, verbose=True):
        """JWST IFU 2D calibrated data class.

        Parameters
        ----------
        filename : str
            Filename of a single JWST cal file to load
        utils_dir : str
            Path to a "utils" directory to write intermediate data products
        verbose : bool
            Be more verbose in output?

        About the "preproc_task_list" parameter. Each task should be a list containing:
            task[0] = the name of the class method
            task[1] = a dictionary with any relevant method arguments (but not including save_utils, see task[2])
                If not defined, it assumes no parameters are needed (task[1] = {}).
            task[2] = a boolean saying if the outputs should be saved in the utils folder.
                Default to class save_utils if not defined for the task.
                If it is a string instead, it will be saved with the string as the filename.
            task[3] = a boolean saying if we should attempt to load the data from the utils folder.
                Default to class load_utils if not defined for the task.

        """
        self.breads_header = pyfits.Header()
        # Pipeline identification
        self.breads_header['VERSION'] = ('1.0.0', 'BREADS pipeline version')
        breads_commit = get_breads_commit()
        self.breads_header['COMMITH'] = (breads_commit[:40], 'BREADS git commit hash')
        self.breads_header['REDDATE'] = (datetime.now(timezone.utc).isoformat(),'UTC date of reduction')
        self.breads_header['COORDS'] = "None"


        self.bad_pixels = None

        self.x = None
        self.y = None
        self.wavelengths = None
        self.area2d = None
        self.trace_id_map = None


        self.wv_sampling = None
        self.opmode = None
        self.webbpsf_im = None
        self.webbpsf_X = None
        self.webbpsf_Y = None
        self.webbpsf_interp = None

        self.leftnright_wavelengths = None

        if filename is not None:
            self.verbose = verbose
            if self.verbose:
                print(f"Reading data from {filename}")
            self.filename = filename

            if utils_dir is None:
                self.utils_dir = os.path.dirname(self.filename)
            else:
                self.utils_dir = utils_dir

            self.crds_dir = os.getenv('CRDS_PATH')
            self.bary_RV = 0
            self._init_read_fits()
            self._init_default_names()
            self._init_wave_wcs(filename)
        else:
            warning_text = "No data file provided. " + \
                           "Please manually add data or use JWSTNirspec.read_data_file()"
            warn(warning_text)

    def _init_read_fits(self):
        """
        Init the JWST_IFUs class with reading the fits file
        """
        ## Part 1: Loading information from the FITS header metadata
        hdulist_sc = pyfits.open(self.filename)
        self.priheader = hdulist_sc[0].header
        self.extheader = hdulist_sc[1].header
        self.breads_header["DATAUNIT"] = self.extheader["BUNIT"].strip()  # MJy/sr or MJy
        self.breads_header["DATA_HPF"] = False

        ## Part 2: Loading information from the FITS data
        self.data = hdulist_sc["SCI"].data
        self.readout_noise_var = hdulist_sc["VAR_RNOISE"].data
        self.photon_noise_var = hdulist_sc["VAR_POISSON"].data
        self.noise = np.sqrt(self.readout_noise_var + self.photon_noise_var)
        dq = hdulist_sc["DQ"].data
        hdulist_sc.close()

        ## Part 3: Creating bad pixels maps
        # Simplifying bad pixel map following convention in this package as: nan = bad, 1 = good
        self.bad_pixels = np.ones_like(self.data)
        # Pixels marked as "do not use" are marked as bad (nan = bad, 1 = good):
        self.bad_pixels[np.where(untangle_dq(dq, verbose=self.verbose)[0, :, :])] = np.nan
        self.bad_pixels[np.where(np.isnan(self.data))] = np.nan
        # Removing any data with zero noise
        where_zero_noise = np.where(self.noise == 0)
        self.noise[where_zero_noise] = np.nan
        self.bad_pixels[where_zero_noise] = np.nan

        self.east2V2_deg = -(float(self.extheader["ROLL_REF"]) + float(self.extheader["V3I_YANG"]))

        try:
            self.opmode = self.priheader["OPMODE"].strip()
        except Exception:
            print() #do nothing


    def _init_default_names(self):
        """
        Init the JWST_IFUs class with default names for saving the intermediate outputs.
        """
        ## Part 4: Defining the default filenames for each intermediate output
        self.default_filenames = {}
        basename = os.path.basename(self.filename)
        self.default_filenames["compute_med_filt_badpix"] = \
            os.path.join(self.utils_dir, basename.replace(".fits", "_roughbadpix.fits"))
        self.default_filenames["compute_coordinates_arrays"] = \
            os.path.join(self.utils_dir, basename.replace(".fits", "_relcoords.fits"))
        splitbasename = os.path.basename(self.filename).split("_")
        self.default_filenames["compute_webbpsf_model"] = \
            os.path.join(self.utils_dir,
                         splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[3] + "_webbpsf.fits")
        self.default_filenames["compute_quick_webbpsf_model"] = \
            os.path.join(self.utils_dir,
                         splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[3] + "_quick_webbpsf.fits")
        self.default_filenames["compute_new_coords_from_webbPSFfit"] = \
            os.path.join(self.utils_dir, basename.replace(".fits", "_newcen_wpsf.fits"))
        self.default_filenames["compute_starspectrum_contnorm"] = \
            os.path.join(self.utils_dir, basename.replace(".fits", "_starspec_contnorm.fits"))
        self.default_filenames["compute_starspectrum_contnorm_3dspline"] = \
            os.path.join(self.utils_dir, basename.replace(".fits", "_starspec_contnorm_3Dspline.fits"))
        self.default_filenames["compute_starsubtraction"] = \
            os.path.join(self.utils_dir, basename.replace(".fits", "_starsub.fits"))
        self.default_filenames["compute_starsubtraction_3dspline"] = \
            os.path.join(self.utils_dir, basename.replace(".fits", "_starsub_3Dspline.fits"))
        self.default_filenames["compute_advanced_badpix"] = \
            self.default_filenames["compute_starsubtraction"]
        self.default_filenames["compute_interpdata_regwvs"] = \
            os.path.join(self.utils_dir, basename.replace(".fits", "_regwvs.fits"))

    def run_preproc_list(self, save_utils=True, load_utils=True, preproc_task_list=None):
        """
        Init the JWST_IFUs class with pipeline tasks.

        Parameters
        ----------

        save_utils: bool
            if True, save the intermediate outputs for each step in the preproc_task_list as a FITS file
        load_utils: bool
            if True, load the intermediate outputs, if it exists, for each step in the preproc_task_list
        preproc_task_list:
           prepocessing list of task to execute.
            # Each task should be a list containing:
            # task[0] = the name of the class method
            # task[1] = a dictionary with any relevant method arguments (but not including save_utils, see task[2])
            # If not defined, it assumes no parameters are needed (task[1] = {}).
            # task[2] = a boolean saying if the outputs should be saved in the utils folder.
            # Default to class save_utils if not defined for the task.
            # If it is a string instead, it will be saved with the string as the filename.
            # task[3] = a boolean saying if we should attempt to load the data from the utils folder.
            # Default to class load_utils if not defined for the task.
        """

        # Mini "pipeline-like" sequence, running a list of task (ie, methods) specified in preproc_task_list
        if preproc_task_list is None:
            preproc_task_list = []
        for task in preproc_task_list:

            task_name = task[0]
            if len(task) > 1:
                dict_paras = task[1]
            else:
                dict_paras = {}
            if len(task) > 2:
                save_task = task[2]
            else:
                save_task = save_utils
            if len(task) > 3:
                load_task = task[3]
            else:
                load_task = load_utils

            # If save_task is not a string, the default filename is assumed
            if isinstance(save_task, str):
                task_out_filename = save_task
            else:
                if task_name in self.default_filenames.keys():
                    task_out_filename = self.default_filenames[task_name]
                else:
                    task_out_filename = None
            if "compute_" in task_name:
                _run_task = True

                if load_task and task_out_filename is not None:
                    # Loading data instead because this task has already been done and it is available in the utils folder.
                    func = getattr(self, task_name.replace("compute_", "reload_"))
                    out_reloading = func(load_filename=task_out_filename)
                    if out_reloading is not None:
                        if self.verbose:
                            print(f"Loaded data for {task_name} cached in {task_out_filename}")
                        _run_task = False

                if _run_task:
                    # Run task
                    if self.verbose:
                        print(f"Running {task_name} with parameters:")
                        print(f"\t save_utils: {save_task}")
                        for para_name in dict_paras.keys():
                            print(f"\t {para_name}: {dict_paras[para_name]}")

                    func = getattr(self, task_name)
                    func(save_utils=save_task, **dict_paras)
            else:
                # Run task
                if self.verbose:
                    print(f"Running {task_name} with parameters:")
                    print(f"\t save_utils: {save_task}")
                    for para_name in dict_paras.keys():
                        print(f"\t {para_name}: {dict_paras[para_name]}")

                func = getattr(self, task_name)
                func(**dict_paras)

    @abstractmethod
    def _init_wcs(self, filename):
        """Hook to be implemented by subclasses"""
        raise NotImplementedError

    @abstractmethod
    def _init_wave_wcs(self, filename):
        """Hook to be implemented by subclasses"""
        raise NotImplementedError

    @abstractmethod
    def compute_med_filt_badpix(self):
        """Hook to be implemented by subclasses"""
        raise NotImplementedError

    def _save_med_filt_badpix(self, save_utils, new_badpix):
        """Save the bad pixel map computed via median filtering.

        Format:
        Name        Ver    Type          Cards   Dimensions      Format
        --------    ---    ----          -----   ----------      ------
        PRIMARY       1    PrimaryHDU     268    ()
        BADPIXEL      1    ImageHDU        73    (2048, 2048)    float64
        BREADS        1    ImageHDU        13    ()

        Parameters
        ----------
        save_utils: str or None
            Path to save the bad pixel map computed via median filtering. if None, by default path and filename are used.
        new_badpix: 2d np.array
            New bad pixel map computed via median filtering.

        """
        if isinstance(save_utils, str):
            out_filename = save_utils
        else:
            out_filename = self.default_filenames["compute_med_filt_badpix"]

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(header=self.priheader))
        hdulist.append(pyfits.ImageHDU(data=new_badpix, header=self.extheader, name='BADPIXEL'))
        hdulist.append(pyfits.ImageHDU(header=self.breads_header, name='BREADS'))
        hdulist.writeto(out_filename, overwrite=True)
        hdulist.close()
        if self.verbose:
            print(f"  Saved the quick bad pixel map to {out_filename}")

    def reload_med_filt_badpix(self, load_filename=None):
        """ Reload and apply bad pixel map from med_filt_badpix.

        Parameters
        ----------
        load_filename : str or None
            Loading directory. If None, will use self.default_filenames["compute_med_filt_badpix"]

        Returns
        -------
        new_badpix : np.array
            nans = bad.

        """
        if load_filename is None:
            load_filename = self.default_filenames["compute_med_filt_badpix"]
        if len(glob(load_filename)) == 0:
            return None

        with pyfits.open(load_filename) as hdulist:
            new_badpix = hdulist['BADPIXEL'].data

        self.bad_pixels *= new_badpix

        return new_badpix

    def compute_coordinates_arrays(self, save_utils=False, center_with_targname=True, targname=None):
        """ Determine the relative coordinates in the focal plane relative to the target (sky coordinates RA/DEC).
        Compute the coordinates {wave, delta_ra, delta_dec, area2d} for each pixel in a 2D image

        Parameters
        ----------
        save_utils : bool
            Save the computed coordinates into the utils directory
        center_with_targname : bool
            if True, compute the star relative coordinates.
        targname : str or None (optional)
            The star name recognized by SIMBAD query.
            if None, the target name is set accordingly to the "TARGNAME" keyword in .fits header

        Returns
        -------
        wave_array: 2d array
            Wavelength coordinate in detector space (microns)
        x: 2d array
            Star relative RA coordinate in detector space (arcsec)
        y: 2d array
            Star relative DEC coordinate in detector space (arcsec)
        area2d: 2d array
            2D mapping of the pixel area (arcsec^2)

        """

        if self.verbose:
            print(f"Computing coordinates arrays.")

        hdulist = pyfits.open(self.filename) #open file
        self._init_wcs(self.filename)

        if center_with_targname:
            # Calculate the updated SkyCoord object for the desired date
            if targname is None:
                targname = hdulist[0].header["TARGNAME"]
            host_coord = utils.propagate_coordinates_at_epoch(targname, hdulist[0].header["DATE-OBS"])
            host_ra_deg = host_coord.ra.deg
            host_dec_deg = host_coord.dec.deg

            dra_as_array = (self.ra_array - host_ra_deg) * 3600 * np.cos(np.radians(self.dec_array))
            ddec_as_array = (self.dec_array - host_dec_deg) * 3600
        else:
            dra_as_array = self.ra_array
            ddec_as_array = self.dec_array

        self.x = dra_as_array
        self.y = ddec_as_array
        self.breads_header['COORDS'] = "sky"
        self.breads_header['COORUNIT'] = "arcsec"
        self.breads_header['AREAUNIT'] = "steradian"

        if save_utils:
            self._save_coordinates_arrays(save_utils)

        return self.wavelengths, self.x, self.y, self.area2d

    def _save_coordinates_arrays(self, save_utils):
        """ Save the computed sky coordinates into the save utils directory.
        This will save in a fits the relative coordinates in arcsec, the 2D pixelscale in steradian and the ID of the traces

        Format of the utils file:
        No.    Name      Ver    Type      Cards   Dimensions   Format
          0  PRIMARY       1 PrimaryHDU     264   ()
          1  WAVE          1 ImageHDU        73   (2048, 2048)   float32
          2  X             1 ImageHDU         9   (2048, 2048)   float64
          3  Y             1 ImageHDU         9   (2048, 2048)   float64
          4  AREA2D        1 ImageHDU         9   (2048, 2048)   float32
          5  TRACE_ID_MAP    1 ImageHDU         8   (2048, 2048)   float64
          6  BREADS        1 ImageHDU        11   ()

        Parameters
        ----------
        save_utils : str or None
            directory to save the computed coordinates. If None, the coordinates are saved in default utils directory.
        """
        if isinstance(save_utils, str):
            out_filename = save_utils
        else:
            out_filename = self.default_filenames["compute_coordinates_arrays"]

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(header=self.priheader))
        hdulist.append(pyfits.ImageHDU(data=self.wavelengths, header=self.extheader, name='WAVE'))
        hdulist.append(pyfits.ImageHDU(data=self.x,
                                       header=pyfits.Header({'BUNIT': self.breads_header['COORUNIT']}),
                                        name='X'))
        hdulist.append(pyfits.ImageHDU(data=self.y,
                                       header=pyfits.Header({'BUNIT': self.breads_header['COORUNIT']}),
                                        name='Y'))
        hdulist.append(pyfits.ImageHDU(data=self.area2d,
                                       header=pyfits.Header({'BUNIT': self.breads_header['AREAUNIT']}),
                                       name='AREA2D'))
        hdulist.append(pyfits.ImageHDU(data=self.trace_id_map, name='TRACE_ID_MAP'))
        hdulist.append(pyfits.ImageHDU(header=self.breads_header, name='BREADS'))
        hdulist.writeto(out_filename, overwrite=True)
        hdulist.close()
        if self.verbose:
            print(f"  Saved the computed coordinates arrays to {out_filename}")

    def reload_coordinates_arrays(self, load_filename=None):
        """ Reload coordinates arrays

        This updates the attributes self.x, self.y, self.area2d.

        Parameters
        ----------
        load_filename : str or None
            Filename to load coordinates from. If None, will use a default filename.
        """
        if load_filename is None:
            load_filename = self.default_filenames["compute_coordinates_arrays"]
        if len(glob(load_filename)) ==0:
            return None

        with pyfits.open(load_filename) as hdulist:
            wave_array = hdulist["WAVE"].data
            x = hdulist['X'].data
            y = hdulist['Y'].data
            area2d = hdulist['AREA2D'].data
            try:
                self.trace_id_map = hdulist['TRACE_ID_MAP'].data
            except KeyError:
                print("Old reduction of coordinates. Could not find hdulist['TRACE_ID_MAP'].data. Please reprocess.")
            try:
                hdr_area2d = hdulist['AREA2D'].header['BUNIT']
            except KeyError:
                print("Warning - Could not find data unit for hdulist['AREA2D']. \n *_relcoords.fits intermediate files seems to be computed from an old breads version")
                mean_area2d = np.nanmedian(area2d)
                if mean_area2d < 1e-12: #Unit surely in steradian
                    print("Unit seems to be in steradian")
                else:
                    print("Unit seems to be in arcsec^2, converting to steradian for compatibility")
                    arcsec2_to_steradian = (2. * np.pi / (360. * 3600.)) ** 2
                    area2d *= arcsec2_to_steradian
            self.breads_header['COORDS'] = hdulist['BREADS'].header['COORDS']
            self.breads_header['COORUNIT'] = hdulist['X'].header['BUNIT']
            self.breads_header['AREAUNIT'] = hdulist['AREA2D'].header['BUNIT']

        self.x, self.y, self.area2d = x, y, area2d
        return wave_array, x, y, area2d

    def set_coords2ifu(self):
        """ Set coordinate frame to IFU
        This is similar to the jwst pipeline 'ifualign' frame, with
        delta position in arcseconds in the IFU instrument axes frame

        Returns
        -------
        ifuX : 2d array
            X-Coordinate in IFU coordinates (arcsec)
        ifuY
            Y-Coordinate in IFU coordinates (arcsec)
        """
        if "ifu" in self.breads_header['COORDS']:
            print("Coordinates already ifu, not doing anything.")
            ifuX = self.x
            ifuY = self.y
        else:
            ifuX, ifuY = self.get_ifu_coords()
            self.x, self.y = ifuX, ifuY
            self.breads_header['COORDS'] = self.breads_header['COORDS'].replace("sky","ifu")
            if hasattr(self, 'webbpsf_interp') and self.webbpsf_interp is not None:
                # Need to recompute the quick webbpsf interpolator with the correct orientation
                wX, wY = rotate_coordinates(self.webbpsf_X.flatten(), self.webbpsf_Y.flatten(), -0.0, flipx=True)
                self.webbpsf_interp = CloughTocher2DInterpolator((wX, wY), self.webbpsf_im.flatten(),fill_value=0.0)
        return ifuX, ifuY

    def set_coords2sky(self):
        """ Set coordinate frame to sky
        This is similar to the jwst pipeline 'skyalign' frame, with
        delta position in arcseconds relative to the sky in ICRS RA, Dec coords.

        Returns
        -------
        dra_as_array : 2d array
            Star relative RA coordinates (arcsec)
        ddec_as_array : 2d array
            Star relative Dec coordinates (arcsec)
        """
        if "sky" in self.breads_header['COORDS']:
            print("Coordinates already sky, not doing anything.")
            dra_as_array = self.x
            ddec_as_array = self.y
        else:
            dra_as_array, ddec_as_array = self.get_sky_coords()
            self.x, self.y = dra_as_array, ddec_as_array
            self.breads_header['COORDS'] = self.breads_header['COORDS'].replace("ifu","sky")
            if hasattr(self, 'webbpsf_interp'):
                # Need to recompute the quick webbpsf interpolator with the correct orientation
                wX, wY = rotate_coordinates(self.webbpsf_X.flatten(), self.webbpsf_Y.flatten(), -self.east2V2_deg, flipx=True)
                self.webbpsf_interp = CloughTocher2DInterpolator((wX, wY), self.webbpsf_im.flatten(),fill_value=0.0)

        return dra_as_array, ddec_as_array

    def convert_MJy_per_sr_to_MJy(self):
        """
        Convert MJy/steradian to MJy.

        Parameters
        ----------

        Returns
        -------
        data : 2d array
            Data converted in MJy.
        noise: 2d array
            Flux noise converted in MJy.
        """
        if self.breads_header["DATAUNIT"] != "MJy/sr":
            raise Exception("Data should in MJy/sr to be converted from MJy/sr to MJy")

        self.data *= self.area2d
        self.noise *= self.area2d
        self.breads_header["DATAUNIT"] = "MJy"
        return self.data, self.noise

    def apply_coords_offset(self, coords_offset=None,coords_filename = None):
        """ Offset coordinates in the class:
        self.x -= coords_offset[0]
        self.y -= coords_offset[1]

        Can only call this method after compute_coordinates_arrays has been run for dra_as_array/ddec_as_array to be
        defined.
        Load/save feature not applicable here.

        Parameters
        ----------
        coords_offset: List
            (offset ra, offset dec) in arcsec


        Returns
        -------
        dra_as_array: in arcsec, new relative RA after offset
        ddec_as_array: in arcsec, new relative declination after offset

        """
        if coords_filename is not None and len(glob(coords_filename)) == 1:
            print(f"Found centroid filename {coords_filename}. Loading those.")
            coords_offset = np.loadtxt(coords_filename, delimiter=' ')
        if coords_offset is None:
            coords_offset = [0,0]

        if self.verbose:
            print(f"Applying relative coordinate offset {coords_offset}")
        if isinstance(coords_offset[0],list) or isinstance(coords_offset[0],np.ndarray):
            self.x -= np.polyval(coords_offset[0], self.wavelengths)
        else:
            if np.isfinite(coords_offset[0]):
                self.x -= coords_offset[0]
            else:
                raise ValueError("coords_offset must be finite")
        if isinstance(coords_offset[1],list) or isinstance(coords_offset[1],np.ndarray):
            self.y -= np.polyval(coords_offset[1], self.wavelengths)
        else:
            if np.isfinite(coords_offset[1]):
                self.y -= coords_offset[1]
            else:
                raise ValueError("coords_offset must be finite")

        return self.x, self.y


    def compute_webbpsf_model(self, image_mask=None, pixelscale=0.1, oversample=10, fov_arcsec=6, wv_sampling=None, save_utils=False, mppool=None):
        """ Compute WebbPSF simulated PSFs for JWST IFU

        Parameters
        ----------
        wv_sampling : np.array of floats
            Wavelength array. WebbPSF is computed at each wavelength.
            If None, it will use the self.wv_sampling attribute if it is available; eg if compute_interpdata_regwvs has been run before.
        image_mask : str or None
            image mask to use in webbpsf calculations. Default is None since we generally do not wish the edges of the
            IFU aperture in the simulated PSF.
        pixelscale : float
            Pixelscale to use for simulated PSF
        oversample : int
            Oversampling factor
        save_utils : bool
            Save in the utils directory
        mppool : multiprocessing.Pool
            If not None, Use multiprocessing to parallelize operations over wavelengths.
            Pool instance for use in parallelized computations.

        Returns
        -------
        wpsfs : np.array
            3D array of shape (nwavelengths, nY, nX) containing the simulated PSF at each wavelength
        wpsfs_header :
            FITS header of the simulated PSF, containing the relevant information about the PSF calculation
        wepsfs : np.array
            Effective PSF: 3D array of shape (nwavelengths, nY, nX) containing the simulated effective PSF at each wavelength,
            This means that each value is integrated over the spaxel area.
        wv_sampling : np.array
            Wavelength array at which the PSF was computed
        webbpsf_X : np.array
            2D array of X coordinates in arcsec for the simulated PSF
        webbpsf_Y : np.array
            2D array of Y coordinates in arcsec for the simulated PSF
        oversample : int
            Oversampling parameter used in calculation
        pixelscale : float
            Pixel scale used in calculation

        """

        if self.verbose:
            print("Computing PSFs. This has to iterate over many wavelengths, so is slow.")

        if wv_sampling is None:
            if not hasattr(self, "wv_sampling"):
                self.wv_sampling = self.get_regwvs_sampling()
            wv_sampling = self.wv_sampling
        else:
            if not hasattr(self, "wv_sampling"):
                if not np.allclose(self.wv_sampling, wv_sampling):
                    raise Exception(
                        "The wv_nodes of the spline continuum fit are different for different data objects. This should not happen. Please check the compute_starspectrum_contnorm outputs for each data object.")
            else:
                self.wv_sampling = wv_sampling

        nwavelen = np.size(wv_sampling)
        IFU = self._get_webbpsf_model_inputs(image_mask, pixelscale)

        if mppool is None:
            parallelize = False
            if self.verbose:
                print(f"\tPerforming serial calculation of PSF at {nwavelen} wavelengths.")

            outarr_not_created = True
            for wv_id, wv in tqdm(enumerate(wv_sampling), total=nwavelen, ncols=100):
                paras = IFU, wv, oversample, self.opmode, parallelize, fov_arcsec
                out = _get_wpsf_task(paras)
                if outarr_not_created:
                    wpsfs = np.zeros((nwavelen, out[0].shape[0], out[0].shape[1]))
                    wepsfs = np.zeros((nwavelen, out[0].shape[0], out[0].shape[1]))
                    outarr_not_created = False
                wpsfs[wv_id, :, :] = out[0]
                wepsfs[wv_id, :, :] = out[1]
                if wv_id == 0:
                    wpsfs_header = out[2] #save the webbpsf header of the first wavelength

        else: # Parallelized version
            parallelize = True
            if self.verbose:
                print(f"\tPerforming parallelized calculation of PSF at {nwavelen} wavelengths.")
            #we must prepare the IFU.pupilopd object to get pickled
            import tempfile
            fp = tempfile.NamedTemporaryFile()
            temp_filename = fp.name
            print('Writing pupilopd tempfile : {}'.format(temp_filename))
            IFU.pupilopd.writeto(temp_filename)  # save that FITS to disk
            IFU.pupilopd = temp_filename         # the object is now a pickle-able string

            print('preparing parameter list...')
            paras_list = []
            for wv_id, wv in enumerate(wv_sampling):
                rprint('{},{},{}'.format(wv_id, wv, nwavelen))
                paras = IFU, wv, oversample, self.opmode, parallelize, fov_arcsec
                paras_list.append(paras)
            print('')

            print('starting parallel _get_wpsf_task ...')
            # Iterate, and display progress bar
            pool_out = [ o for o in tqdm(mppool.imap(_get_wpsf_task, paras_list), total=nwavelen, ncols=100)]
            print('')

            print('collating pool outputs...')
            out = pool_out[0]
            wpsfs = np.zeros((nwavelen, *out[0].shape))
            wepsfs = np.zeros((nwavelen, *out[0].shape))
            wpsfs_header = out[2] #save the webbpsf header of the first wavelength
            for ind,out in enumerate(pool_out):
                rprint(ind)
                wpsfs[ind, :, :] = out[0]
                wepsfs[ind, :, :] = out[1]
            print('')
            print('done.')

        wepsfs *= oversample ** 2

        halffov_x = IFU.pixelscale / oversample * wpsfs.shape[2] / 2.0
        halffov_y = IFU.pixelscale / oversample * wpsfs.shape[1] / 2.0
        x = np.linspace(-halffov_x, halffov_x, wpsfs.shape[2], endpoint=True)
        y = np.linspace(-halffov_y, halffov_y, wpsfs.shape[1], endpoint=True)
        webbpsf_X, webbpsf_Y = np.meshgrid(x, y)

        wpsfs_additional_header = {'PIXELSCL': pixelscale, 'im_mask': image_mask,
                        'oversamp': oversample, 'DATE-BEG': self.priheader['DATE-BEG']}
        self.breads_header.update(wpsfs_additional_header)


        self.breads_header['WPSFAREA'] = pixelscale ** 2
        psf_wv0_id = np.argmin(np.abs(wv_sampling-np.nanmedian(self.wavelengths)))
        self.webbpsf_im = wepsfs[psf_wv0_id]
        self.webbpsf_X = webbpsf_X
        self.webbpsf_Y = webbpsf_Y
        self.breads_header['WBPSFWV0'] = wv_sampling[psf_wv0_id]
        if "sky" in self.breads_header['COORDS']:
            wX, wY = rotate_coordinates(self.webbpsf_X.flatten(), self.webbpsf_Y.flatten(), -self.east2V2_deg, flipx=True)
        elif "ifu" in self.breads_header['COORDS']:
            wX, wY = rotate_coordinates(self.webbpsf_X.flatten(), self.webbpsf_Y.flatten(), -0.0, flipx=True)
        self.webbpsf_interp = CloughTocher2DInterpolator((wX, wY), self.webbpsf_im.flatten(),fill_value=0.0)

        if save_utils:
            self._save_webbpsf_model(save_utils, wpsfs, wpsfs_header, wepsfs, webbpsf_X, webbpsf_Y,quick=False)
        return wpsfs, wpsfs_header, wepsfs, wv_sampling, webbpsf_X, webbpsf_Y, oversample, pixelscale

    def _save_webbpsf_model(self, save_utils, wpsfs, wpsfs_header, wepsfs, webbpsf_X, webbpsf_Y,
                            quick=False):
        """Save computed webbpsf model to save_utils.

        ===  ========  ===  ==========  =====  =================  =======
        No.  Name      Ver  Type        Cards  Dimensions         Format
        ===  ========  ===  ==========  =====  =================  =======
        0    OVERSAMP    1  PrimaryHDU    109  (600, 600, 2197)   float64
        1    PSFS        1  ImageHDU        9  (600, 600, 2197)   float64
        2    EPSFS       1  ImageHDU        9  (600, 600, 2197)   float64
        3    WAVE        1  ImageHDU        7  (2197,)            float64
        4    X           1  ImageHDU        8  (600, 600)         float64
        5    Y           1  ImageHDU        8  (600, 600)         float64
        6    BREADS      1  ImageHDU       20  ()
        ===  ========  ===  ==========  =====  =================  =======
        """

        if isinstance(save_utils, str):
            out_filename = save_utils
        else:
            if quick:
                out_filename = self.default_filenames["compute_quick_webbpsf_model"]
            else:
                out_filename = self.default_filenames["compute_webbpsf_model"]

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(header=pyfits.Header(cards=wpsfs_header)))
        hdulist.append(pyfits.ImageHDU(data=wpsfs, name='PSFS'))
        hdulist.append(pyfits.ImageHDU(data=wepsfs, name='EPSFS'))
        if not quick:
            hdulist.append(pyfits.ImageHDU(data=self.wv_sampling, name='WAVE'))
        hdulist.append(pyfits.ImageHDU(data=webbpsf_X, name='X'))
        hdulist.append(pyfits.ImageHDU(data=webbpsf_Y, name='Y'))
        hdulist.append(pyfits.ImageHDU(header=self.breads_header, name='BREADS'))
        hdulist.writeto(out_filename, overwrite=True)
        hdulist.close()
        if self.verbose:
            print(f"  Saved the computed PSFs to {out_filename}")

    def reload_webbpsf_model(self, load_filename=None):
        """ Reload a previously-computed WebbPSF model PSF from a FITS file

        Parameters
        ----------
        load_filename : str
            Optional filename to load. If not provided, a default filename will be used.

        Returns
        -------
        wpsfs, wpsfs_header, wepsfs, webbpsf_wvs, webbpsf_X, webbpsf_Y, wpsf_oversample, wpsf_pixelscale

        Also sets a whole bunch of object attributes.
        """
        if load_filename is None:
            load_filename = self.default_filenames["compute_webbpsf_model"]
        if len(glob(load_filename)) ==0:
            return None

        hdulist = pyfits.open(load_filename)

        wpsfs_header = hdulist[0].header
        wpsfs = hdulist['PSFS'].data
        wepsfs = hdulist['EPSFS'].data
        webbpsf_X = hdulist['X'].data
        webbpsf_Y = hdulist['Y'].data
        wpsf_pixelscale = wpsfs_header['PIXELSCL']
        wpsf_oversample = wpsfs_header['oversamp']
        self.breads_header['WBPSFWV0'] = hdulist["BREADS"].header['WBPSFWV0']
        self.breads_header['WPSFAREA'] = hdulist["BREADS"].header['WPSFAREA']

        if "WAVE" in hdulist:
            wv_sampling = hdulist['WAVE'].data
            if not hasattr(self, "wv_sampling"):
                if not np.allclose(self.wv_sampling, wv_sampling):
                    raise Exception(
                        "The wv_nodes of the spline continuum fit are different for different data objects. This should not happen. Please check the compute_starspectrum_contnorm outputs for each data object.")
            else:
                self.wv_sampling = wv_sampling

        hdulist.close()
        # Need to return a bunch of stuff here:

        psf_wv0_id = np.argmin(np.abs(wv_sampling-self.breads_header['WBPSFWV0']))
        self.webbpsf_im = wepsfs[psf_wv0_id]
        self.webbpsf_X = webbpsf_X
        self.webbpsf_Y = webbpsf_Y
        if "sky" in self.breads_header['COORDS']:
            wX, wY = rotate_coordinates(self.webbpsf_X.flatten(), self.webbpsf_Y.flatten(), -self.east2V2_deg, flipx=True)
        elif "ifu" in self.breads_header['COORDS']:
            wX, wY = rotate_coordinates(self.webbpsf_X.flatten(), self.webbpsf_Y.flatten(), -0.0, flipx=True)
        self.webbpsf_interp = CloughTocher2DInterpolator((wX, wY), self.webbpsf_im.flatten(),fill_value=0.0)

        return wpsfs, wpsfs_header, wepsfs, wv_sampling, webbpsf_X, webbpsf_Y, wpsf_oversample, wpsf_pixelscale


    def reload_breadspsf_model(self, load_filename):
        """ Reload a previously-computed BreadsPSF model PSF from a FITS file

        Parameters
        ----------
        load_filename : str
            filename to load. If not provided, a default filename will be used.

        Returns
        -------

        Also sets a whole bunch of object attributes.
        """
        if len(glob(load_filename)) ==0:
            return None

        hdulist = pyfits.open(load_filename)

        breads_header = hdulist[0].header
        epsfs = hdulist['EPSFS'].data
        psf_X = hdulist['X'].data
        psf_Y = hdulist['Y'].data
        self.breads_header['WPSFAREA'] = breads_header['BPSFAREA']
        self.breads_header['WBPSFWV0'] = breads_header['BPSFWV0']

        if "WAVE" in hdulist:
            wv_sampling = hdulist['WAVE'].data
            if not hasattr(self, "wv_sampling"):
                if not np.allclose(self.wv_sampling, wv_sampling):
                    raise Exception("WebbPSF wavelength sampling is different from the one known to the class.")
            else:
                self.wv_sampling = wv_sampling

        hdulist.close()
        # Need to return a bunch of stuff here:

        psf_wv0_id = np.argmin(np.abs(wv_sampling-self.breads_header['WBPSFWV0']))
        self.webbpsf_im = epsfs[psf_wv0_id]
        self.webbpsf_X = psf_X
        self.webbpsf_Y = psf_Y
        if "sky" in self.breads_header['COORDS']:
            wX, wY = rotate_coordinates(self.webbpsf_X.flatten(), self.webbpsf_Y.flatten(), -self.east2V2_deg, flipx=False)
        elif "ifu" in self.breads_header['COORDS']:
            wX, wY = rotate_coordinates(self.webbpsf_X.flatten(), self.webbpsf_Y.flatten(), -0.0, flipx=True)
        self.webbpsf_interp = CloughTocher2DInterpolator((wX, wY), self.webbpsf_im.flatten(),fill_value=0.0)

        return epsfs, wv_sampling, psf_X, psf_Y

    @abstractmethod
    def _get_webbpsf_model_inputs(self, image_mask, pixelscale):
        """Hook to be implemented by subclasses"""
        raise NotImplementedError

    def _get_webbpsf_fit_inputs(self):
        """
        Hook to override if necessary, returns input for webbPSF fit
        """
        return (
            np.copy(self.bad_pixels),
            np.copy(self.data),
            np.copy(self.noise),
            np.copy(self.x),
            np.copy(self.y),
            np.copy(np.abs(self.wavelengths - self.breads_header['WBPSFWV0'])),
        )

    def compute_quick_webbpsf_model(self, image_mask=None, pixelscale=0.1, oversample=10, fov_arcsec=6, save_utils=False):
        """ Compute WebbPSF simulated PSFs at the MEDIAN WAVELENGTH ONLY for the JWST IFU.

        Parameters
        ----------
        image_mask : str or None
            image mask to use in webbpsf calculations. Default is None since we generally do not wish the edges of the
            IFU aperture in the simulated PSF
        pixelscale : float
            Pixelscale to use for simulated PSF
        oversample : int (default is 10)
            Oversampling factor
        fov_arcsec : float (default is 6)
            Size of the FoV (arcsec)
        save_utils : bool
            Save in the utils directory

        Returns
        -------
        wpsfs : 2d numpy array
            OVERSAMP extension of the simulated webbPSF
        wepsfs_header : fits.header
            FITS header of the OVERSAMP webbPSF
        wepsfs : 2d numpy array
            OVERSAMP webbPSF smoothed by a gaussian kernel
        webbpsf_X : 2d numpy array
            X spatial coordinate of the webbPSF
        webbpsf_Y : 2d numpy array
            Y spatial coordinate of the webbPSF
        oversample : int
            Oversampling factor
        IFU.pixelscale : float
            Pixelscale used for simulated PSF

        """

        if self.verbose:
            print("Computing monochromatic PSF.")

        self.breads_header['WBPSFWV0'] = np.nanmedian(self.wavelengths)

        IFU = self._get_webbpsf_model_inputs(image_mask, pixelscale)

        paras = IFU, self.breads_header['WBPSFWV0'], oversample, self.opmode, None, fov_arcsec
        out = _get_wpsf_task(paras)
        wpsfs = out[0] #webbpsf oversampled
        wepsfs = out[1] #webbpsf oversampled + smoothing
        wepsfs_header = out[2] #webbpsf header

        wepsfs *= oversample ** 2

        halffov_x = IFU.pixelscale / oversample * wpsfs.shape[1] / 2.0
        halffov_y = IFU.pixelscale / oversample * wpsfs.shape[0] / 2.0
        x = np.linspace(-halffov_x, halffov_x, wpsfs.shape[1], endpoint=True)
        y = np.linspace(-halffov_y, halffov_y, wpsfs.shape[0], endpoint=True)
        webbpsf_X, webbpsf_Y = np.meshgrid(x, y)

        wepsfs_additional_header = {'PIXELSCL': IFU.pixelscale, 'im_mask': image_mask,
                        'oversamp': oversample, 'DATE-BEG': self.priheader['DATE-BEG']}
        self.breads_header.update(wepsfs_additional_header)

        if save_utils:
            self._save_webbpsf_model(save_utils, wpsfs, wepsfs_header, wepsfs, webbpsf_X, webbpsf_Y,
                            quick=True)

        self.breads_header['WPSFAREA'] = IFU.pixelscale ** 2
        self.webbpsf_im = wepsfs
        self.webbpsf_X = webbpsf_X
        self.webbpsf_Y = webbpsf_Y
        if "sky" in self.breads_header['COORDS']:
            wX, wY = rotate_coordinates(self.webbpsf_X.flatten(), self.webbpsf_Y.flatten(), -self.east2V2_deg, flipx=True)
        elif "ifu" in self.breads_header['COORDS']:
            wX, wY = rotate_coordinates(self.webbpsf_X.flatten(), self.webbpsf_Y.flatten(), -0.0, flipx=True)
        self.webbpsf_interp = CloughTocher2DInterpolator((wX, wY), self.webbpsf_im.flatten(),fill_value=0.0)

        return wpsfs, wepsfs_header, wepsfs, self.breads_header['WBPSFWV0'], webbpsf_X, webbpsf_Y, oversample, IFU.pixelscale

    def reload_quick_webbpsf_model(self, load_filename=None):
        """ Reload a previously-computed quick WebbPSF model PSF from a FITS file

        Parameters
        ----------
        load_filename : str or None
            FITS file name to reload the PSF from. If None, reloads the corresponding file in the default filenames.

        Returns
        -------
            wpsfs
            wpsfs_header
            wepsfs
            webbpsf_X
            webbpsf_Y
            wpsf_oversample
            wpsf_pixelscale
        """
        if load_filename is None:
            load_filename = self.default_filenames["compute_quick_webbpsf_model"]
        if len(glob(load_filename)) ==0:
            return None

        hdulist = pyfits.open(load_filename)

        wpsfs_header = hdulist[0].header
        wpsfs = hdulist['PSFS'].data
        wepsfs = hdulist['EPSFS'].data
        webbpsf_X = hdulist['X'].data
        webbpsf_Y = hdulist['Y'].data
        wpsf_pixelscale = hdulist["BREADS"].header['PIXELSCL']
        wpsf_oversample = hdulist["BREADS"].header['oversamp']
        self.breads_header['WBPSFWV0'] = hdulist["BREADS"].header['WBPSFWV0']
        self.breads_header['WPSFAREA'] = hdulist["BREADS"].header['WPSFAREA']

        hdulist.close()
        # Need to return a bunch of stuff here:

        self.webbpsf_im = wepsfs
        self.webbpsf_X = webbpsf_X
        self.webbpsf_Y = webbpsf_Y
        if "sky" in self.breads_header['COORDS']:
            wX, wY = rotate_coordinates(self.webbpsf_X.flatten(), self.webbpsf_Y.flatten(), -self.east2V2_deg, flipx=True)
        elif "ifu" in self.breads_header['COORDS']:
            wX, wY = rotate_coordinates(self.webbpsf_X.flatten(), self.webbpsf_Y.flatten(), -0.0, flipx=True)
        self.webbpsf_interp = CloughTocher2DInterpolator((wX, wY), self.webbpsf_im.flatten(),fill_value=0.0)

        return wpsfs, wpsfs_header, wepsfs, self.breads_header['WBPSFWV0'] , webbpsf_X, webbpsf_Y, wpsf_oversample, wpsf_pixelscale

    def insert_psf_model(self, save_utils=False,centroid = None,OWA=None,spectrum_func=None,out_folder = "insert_psf",
                         mode=None):
        """Inserts a PSF model
        TODO: add documentation
        Parameters
        ----------
        """

        if not hasattr(self,"webbpsf_interp"):
            raise Exception("WebbPSF not found. Please run compute_quick_webbpsf_model or compute_webbpsf_model first.")

        if mode is None:
            mode = "quick_webbpsf"

        if centroid is None:
            centroid = [0,0]

        if OWA is None:
            where_finite = np.where(np.isfinite(self.x))
        else:
            separation_arr = np.sqrt(self.x**2+self.y**2)
            where_finite = np.where(np.isfinite(self.x)*(separation_arr<OWA))

        _dra_as_array, _ddec_as_array = self.getskycoords()
        x = _dra_as_array[where_finite]
        y = _ddec_as_array[where_finite]
        w = self.wavelengths[where_finite]

        if mode == "quick_webbpsf":
            model_vec = self.webbpsf_interp((centroid[0] - x) * self.breads_header['WBPSFWV0'] / w,
                                        (centroid[1] - y) * self.breads_header['WBPSFWV0'] / w)
        else:
            raise Exception("Unknown mode {0} to inject PSF".format(mode))

        if spectrum_func is not None:
            model_vec *= spectrum_func(w)

        model_im = np.full(self.data.shape,np.nan)
        model_im[where_finite] = model_vec

        if save_utils:
            if isinstance(save_utils,str):
                out_filename = save_utils
            else:
                if not os.path.exists(os.path.join(self.utils_dir, out_folder)):
                    os.makedirs(os.path.join(self.utils_dir, out_folder))
                out_filename = os.path.join(self.utils_dir, out_folder,os.path.basename(self.filename))

            hdulist_sc = pyfits.open(self.filename)
            bu = self.extheader["BUNIT"].strip()
            if bu == 'MJy':
                hdulist_sc["SCI"].data = model_im
            if bu == 'MJy/sr':
                hdulist_sc["SCI"].data = model_im/ self.area2d
            hdulist_sc.writeto(out_filename, overwrite=True)
            hdulist_sc.close()

        if self.breads_header["DATAUNIT"] == 'MJy':
            return model_im
        elif self.breads_header["DATAUNIT"] == 'MJy/sr':
            return model_im/ self.area2d

    # def compute_new_coords_from_webbPSFfit(self, save_utils=False,IWA=None,OWA=None,apply_offset=True):
    #     """ Update coordinates after fitting a webbPSF at the median wavelength of the data.
    #     This is the wavelength at which the WebbPSF was saved in the class.
    #
    #     It does not interpolate the data at that wavelength, only grabs the closest pixel.
    #
    #     Parameters
    #     ----------
    #     save_utils : bool
    #         If True, save in the utils directory.
    #     apply_offset : bool
    #         If True, this applies the centroid offset estimated by the webbPSF fit to the RA and DEC coordinates.
    #     IWA : float
    #         Inner Working Angle, in arcsec. This boundary excludes the PSF core for the webbPSF fit. Useful if the PSF core is saturated.
    #     OWA : float
    #         Outer Working Angle, in arcsec. This boundary excludes the PSF wings for the webbPSF fit. Useful if the PSF wings are too noisy.
    #
    #
    #     Returns
    #     -------
    #     ra_offset : float
    #         returns the RA centroid offset (arcsec)
    #     dec_offset : float
    #         returns the DEC centroid offset (arcsec)
    #     """
    #     if IWA is None:
    #         IWA = 0
    #     if OWA is None:
    #         OWA = 1.5
    #
    #     # rough centroid fit
    #     fit_cen, fit_angle = True, False
    #     linear_interp=True
    #     init_paras = np.array([0,0])
    #
    #     # HOOK
    #     mask, data, noise, dra_as_array, ddec_as_array, diff_wv_map = self._get_webbpsf_fit_inputs()
    #
    #     mask[np.where(diff_wv_map > np.nanmedian(self.wavelengths) / self.R)] = np.nan
    #
    #     allnans_rows = np.where(np.nansum(np.isfinite(diff_wv_map), axis=1) == 0)
    #     diff_wv_map[allnans_rows, :] = 0
    #
    #     argmin_ids = np.nanargmin(diff_wv_map, axis=1)
    #
    #     paras = (
    #         linear_interp,
    #         self.webbpsf_im,
    #         self.webbpsf_X,
    #         self.webbpsf_Y,
    #         self.east2V2_deg,
    #         True,
    #         dra_as_array[:, argmin_ids],
    #         ddec_as_array[:, argmin_ids],
    #         data[:, argmin_ids],
    #         noise[:, argmin_ids],
    #         mask[:, argmin_ids],
    #         IWA,
    #         OWA,
    #         fit_cen,
    #         fit_angle,
    #         init_paras,
    #     )
    #
    #     out, _ = _fit_wpsf_task(paras)
    #     ra_offset, dec_offset, angle_offset = out[0, 2::]
    #
    #
    #     if save_utils:
    #         self._save_new_coords_from_webbPSFfit(save_utils, ra_offset, dec_offset, angle_offset)
    #
    #     if apply_offset:
    #         self.x -= ra_offset
    #         self.y -= dec_offset
    #     return ra_offset, dec_offset
    #
    # def _save_new_coords_from_webbPSFfit(self, save_utils, ra_offset, dec_offset, angle_offset):
    #     """Save the estimated centroid of the PSF from the webbPSF fit."""
    #
    #     if isinstance(save_utils, str):
    #         out_filename = save_utils
    #     else:
    #         out_filename = self.default_filenames["compute_new_coords_from_webbPSFfit"]
    #
    #     wpsfs_header = {"RA_CEN": ra_offset, "DEC_CEN": dec_offset, "ANGLE": angle_offset}
    #     hdulist = pyfits.HDUList()
    #     hdulist.append(pyfits.PrimaryHDU(header=pyfits.Header(cards=wpsfs_header)))
    #     hdulist.writeto(out_filename, overwrite=True)
    #     hdulist.close()
    #     if self.verbose:
    #         print(f"  Saved the computed PSFs to {out_filename}")
    #
    #
    # def reload_new_coords_from_webbPSFfit(self, load_filename=None,apply_offset=True):
    #     """ Reapply a previously-computed centroid shift based a WebbPSF fit.
    #
    #     Parameters
    #     ----------
    #     load_filename : str
    #         Filename of fits file to load
    #     apply_offset : Boolean
    #         If True, this applies the centroid offset to the RA and DEC coordinates.
    #
    #     Returns
    #     -------
    #     ra_offset : float
    #         returns the RA centroid offset (arcsec)
    #     dec_offset : float
    #         returns the DEC centroid offset (arcsec)
    #     """
    #     if load_filename is None:
    #         load_filename = self.default_filenames["compute_new_coords_from_webbPSFfit"]
    #     if len(glob(load_filename)) ==0:
    #         return None
    #
    #     hdulist = pyfits.open(load_filename)
    #     ra_offset = hdulist[0].header["RA_CEN"]
    #     dec_offset = hdulist[0].header["DEC_CEN"]
    #     hdulist.close()
    #
    #     if apply_offset:
    #         self.x -= ra_offset
    #         self.y -= dec_offset
    #     return ra_offset, dec_offset

    def compute_starspectrum_contnorm(self,  save_utils=False, mppool=None,spec_R_sampling=None, threshold_badpix=10,
                                      wv_nodes=None, N_nodes=40, iterative=True,spline3d_prior_filename=None):
        """ Compute star spectrum normalized by the continuum.
        See Figure 4 in Ruffio+2024 (https://ui.adsabs.harvard.edu/abs/2024AJ....168...73R/abstract).

        Parameters
        ----------
        save_utils : Boolean
            Save the intermediate star subtraction step products.
        mppool : multiprocessing.Pool or None (optional)
            If None, the computation is done without parallelization.
        spec_R_sampling : float or None (optional)
            Spectral resolution to sample the continuum-normalized star spectrum
            If None, the spectral resolution will be set to 4 times the instrumental spectral resolution of the IFU.
        threshold_badpix : float (optional)
            Hard threshold for bad pixel flagging. Thresholding is done by comparing the continuum normalized row with its median absolute deviation.
        wv_nodes : 1d array or None (optional)
            If wv_nodes is specified, this wavelength spacing (in micron) will be used to do the splines fitting.
            If None, N_nodes will set an evenly nodes spacing.
        N_nodes : int or None (optional, default is 40)
            If wv_nodes is None, Number of nodes to use for fitting splines for the continuum star spectrum estimation.
        iterative : Boolean (optional)
            If True, the fitting procedure is iteratively applied. It helps identifies potential additional bad pixels flagging and to have a better regularization.

        Returns
        -------
        new_wavelengths : 1d numpy array (N_wavelengths)
            New wavelengths axis of the combined high-frequency star spectrum (micron)
        combined_fluxes : 1d numpy array (N_wavelengths)
            Combined high-frequency star spectrum (MJy or MJy/sr)
        combined_errors : 1d numpy array (N_wavelengths)
            Combined flux errors (MJy or MJy/sr)
        spline_cont0 : 2d numpy array (N_detector_rows, N_detector_cols)
            Star continuum fitted by splines for each spectral trace of the detector.
        spline_paras0 : 2d numpy array (N_nodes, N_traces)
            Linear parameters returned by the continuum spline fitting routine for each spectral trace of the detector.
        wv_nodes : 1d numpy array (N_nodes)
            Nodes spacing in the wavelength dimension (in micron).

        """
        if self.breads_header["DATA_HPF"]:
            raise Exception("Data is already high-pass filtered, cannot compute star spectrum continuum normalization.")

        # _get_starspectrum_input() takes care of transposing the images for MIRI compared to NIRSpec
        im, im_wvs, err, bad_pixels, spec_R_sampling, wv_nodes = self._get_starspectrum_input(spec_R_sampling, wv_nodes, N_nodes)

        if self.verbose:
            print(f"Computing stellar spectrum (continuum normalized)")

        if spline3d_prior_filename is not None:
            if mppool is None:
                max_cores = 1
            else:
                max_cores = mppool._processes

            _out = evaluate_3dspline_pointcloud(self, spline3d_prior_filename, max_cores=max_cores)
            stellar_features, _ = _out

            reg_mean_map0 = None
            reg_std_map0 = None
        else:
            stellar_features = None

            # Define the regularization in the form of priors on the value of the flux at the position of the spline nodes
            reg_mean_map0 = np.zeros((im.shape[0], np.size(wv_nodes))) # The flux values at each position of the nodes
            reg_std_map0 = np.zeros((im.shape[0], np.size(wv_nodes))) # The corresponding width of the Gaussian prior at each nodes
            for rowid, row in enumerate(im):
                row_bp = bad_pixels[rowid, :]
                if np.nansum(np.isfinite(row * row_bp)) == 0:
                    continue
                # Set the prior to the median value of the row for each node
                median_row = np.nanmedian(row * row_bp)
                stddev_row = np.nanstd(row * row_bp)
                reg_mean_map0[rowid, :] = median_row
                # Set the width of the prior to its mean to have fairly unconstraining priors
                reg_std_map0[rowid, :] = np.max([np.abs(median_row),stddev_row])

        if reg_mean_map0 is None and reg_std_map0 is None:
            regularization = False
        else:
            regularization = True
        spline_cont0, _, new_badpixs, new_res, spline_paras0 = normalize_rows(im, im_wvs, noise=err,
                                                                              badpixs=bad_pixels,
                                                                              wv_nodes=wv_nodes, mppool=mppool,
                                                                              threshold=threshold_badpix,
                                                                              stellar_features = stellar_features,
                                                                              regularization=regularization,
                                                                              reg_mean_map=reg_mean_map0,
                                                                              reg_std_map=reg_std_map0)
        if iterative:
            if regularization:
                reg_mean_map1 = copy(spline_paras0)
                where_nan = np.where(np.isnan(reg_mean_map1))
                reg_mean_map1[where_nan] = reg_mean_map0[where_nan]
                reg_std_map1 = np.abs(reg_mean_map1)
            else:
                reg_mean_map1 = None
                reg_std_map1 = None
            spline_cont0, _, new_badpixs, new_res, spline_paras0 = normalize_rows(im, im_wvs, noise=err, badpixs=new_badpixs,
                                                                                  wv_nodes=wv_nodes, mppool=mppool,
                                                                                  threshold=threshold_badpix,
                                                                                  stellar_features = stellar_features,
                                                                                  regularization=regularization,
                                                                                  reg_mean_map=reg_mean_map1,
                                                                                  reg_std_map=reg_std_map1)

        # _get_masked_normalized_object does the continuum normalization and allows for different behavior between NIRSpec and MIRI
        continuum, normalized_im, normalized_err = self._get_masked_normalized_object(spline_cont0, im, err)

        # Bin the data to create a continuum normalized spectrum
        new_wavelengths, combined_fluxes, combined_errors = combine_spectrum(im_wvs.flatten(),
                                                                             normalized_im.flatten(),
                                                                             normalized_err.flatten(),
                                                                             np.nanmedian(im_wvs) / spec_R_sampling)


        self.wv_nodes = wv_nodes
        self.star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False, fill_value=1)
        self.breads_header['STCONTRS'] = spec_R_sampling
        self.breads_header['STCONTTH'] = threshold_badpix
        if spline3d_prior_filename is not None:
            self.breads_header['SP3DPRFI'] = spline3d_prior_filename
        self.breads_header['STCONTFN'] = ""

        if save_utils:
            self._save_starspectrum_contnorm(save_utils, new_wavelengths, combined_fluxes, combined_errors,
                                             spline_cont0, spline_paras0, wv_nodes, normalized_im,stellar_features)

        return new_wavelengths, combined_fluxes, combined_errors, spline_cont0, spline_paras0, wv_nodes


    def _save_starspectrum_contnorm(self, save_utils, new_wavelengths, combined_fluxes, combined_errors, spline_cont0,
                                    spline_paras0, wv_nodes, normalized_im,stellar_features):
        """Save the continuum normalized star spectrum in a fits file.

        No.    Name      Ver    Type      Cards   Dimensions   Format
          0  PRIMARY       1 PrimaryHDU     268   ()
          1  WAVE          1 ImageHDU        72   (2629,)   float64
          2  COM_FLUXES    1 ImageHDU         7   (2629,)   float64
          3  COM_ERRORS    1 ImageHDU         7   (2629,)   float64
          4  SPLINE_CONT0    1 ImageHDU         8   (2048, 2048)   float32
          5  SPLINE_PARAS0    1 ImageHDU         8   (40, 2048)   float64
          6  wv_nodes       1 ImageHDU         7   (40,)   float64
          7  CONT_NORM_IM    1 ImageHDU         8   (2048, 2048)   float32
          8  BREADS        1 ImageHDU        18   ()

        Parameters
        ----------
        save_utils : str or None
            Path to save the fits file. If None, the default directory will be used.
        new_wavelengths : 1d array
            Wavelength array in micron of the continuum normalized star spectrum.
        combined_fluxes : 1d array
            Flux array of the continuum normalized star spectrum. (without unit)
        combined_errors : 1d array
            Flux errors array of the continuum normalized star spectrum. (without unit)
        spline_cont0 : 2d array (Nrows x Ncols)
            Continuum fitted by splines for each trace.
        spline_paras0 : 2d array (Nrows x N_nodes)
            Splines best fit parameters for each trace.
        wv_nodes : 1d array
            Nodes spacing in the wavelength dimension (in micron).

        """
        if isinstance(save_utils, str):
            out_filename = save_utils
        else:
            out_filename = self.default_filenames["compute_starspectrum_contnorm"]

        self.breads_header["STCONTFN"] = out_filename
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(header=self.priheader))
        hdulist.append(pyfits.ImageHDU(data=new_wavelengths,header=self.extheader,name="WAVE"))
        hdulist.append(pyfits.ImageHDU(data=combined_fluxes, name='COM_FLUXES'))
        hdulist.append(pyfits.ImageHDU(data=combined_errors, name='COM_ERRORS'))
        hdulist.append(pyfits.ImageHDU(data=spline_cont0, name='SPLINE_CONT0'))
        hdulist.append(pyfits.ImageHDU(data=spline_paras0, name='SPLINE_PARAS0'))
        hdulist.append(pyfits.ImageHDU(data=wv_nodes, name='wv_nodes'))
        hdulist.append(pyfits.ImageHDU(data=normalized_im, name='CONT_NORM_IM'))
        if stellar_features is not None:
            hdulist.append(pyfits.ImageHDU(data=stellar_features, name='STELLAR_FEATURES'))
        hdulist.append(pyfits.ImageHDU(header=self.breads_header, name='BREADS'))
        hdulist.writeto(out_filename, overwrite=True)
        hdulist.close()

        if self.verbose:
            print(f"Saved the continuum normalized star spectrum to {out_filename}")

    def _get_starspectrum_input(self, spec_R_sampling, wv_nodes, N_nodes):
        """Get inputs to compute the continuum normalized star spectrum.
        Helper function for compute_starspectrum_contnorm().
        """

        im = np.copy(self.data)
        im_wvs = np.copy(self.wavelengths)
        err = np.copy(self.noise)
        if spec_R_sampling is None:
            spec_R_sampling = self.R*4
        if wv_nodes is None:
            wv_nodes = np.linspace(np.nanmin(im_wvs), np.nanmax(im_wvs), N_nodes, endpoint=True)
        bad_pixels = self.bad_pixels

        return im, im_wvs, err, bad_pixels, spec_R_sampling, wv_nodes

    def _get_masked_normalized_object(self, continuum, im, err):
        """Get the continuum normalized star spectrum.
        Helper function for compute_starspectrum_contnorm().

        Parameters
        ----------
        continuum : 2d array (Nrows x Ncols)
            Fitted continuum for each trace.
        im : 2d array (Nrows x Ncols)
            Flux in MJy or MJy/sr.
        err : 2d array (Nrows x Ncols)
            Flux errors in MJy or MJy/sr.
            Flux errors in MJy or MJy/sr

        Returns
        -------
        continuum : 2d array (Nrows x Ncols)
            Fitted continuum for each trace with bad pixels and SNR mask.
        normalized_spectrum : 2d array (Nrows x Ncols)
            Continuum normalized 2d flux image.
        normalized_errors : 2d array (Nrows x Ncols)
            Continuum normalized 2d flux error image.
         """

        continuum = copy(continuum)
        continuum[np.where(continuum / err < 5)] = np.nan
        continuum[np.where(continuum < np.median(continuum))] = np.nan
        continuum[np.where(np.isnan(self.bad_pixels))] = np.nan
        normalized_im = im / continuum
        normalized_err = err / continuum
        return continuum, normalized_im, normalized_err

    def reload_starspectrum_contnorm(self, load_filename=None):
        """ Reload star spectrum normalized by continuum

        Parameters
        ----------
        load_filename : str or None
            Filename to load spectrum data from, or leave None to use default filename

        Returns
        -------
        new_wavelengths : 1d numpy array (N_wavelengths)
            New wavelengths axis of the combined high-frequency star spectrum (micron)
        combined_fluxes : 1d numpy array (N_wavelengths)
            Combined continuum normalized spectrum of the star.
        combined_errors : 1d numpy array (N_wavelengths)
            error vector for combined_fluxes
        spline_cont0 : 2d numpy array (N_detector_rows, N_detector_cols)
            Star continuum fitted by splines for each spectral trace of the detector.
        spline_paras0 : 2d numpy array (N_nodes, N_traces)
            Linear parameters returned by the continuum spline fitting routine for each spectral trace of the detector.
        wv_nodes : 1d numpy array (N_nodes)
            Nodes spacing in the wavelength dimension (in micron).

        Also sets self.wv_nodes and self.star_func according to values in the reloaded file headers.

        """
        if load_filename is None:
            load_filename = self.default_filenames["compute_starspectrum_contnorm"]
        if len(glob(load_filename)) ==0:
            return None

        hdulist = pyfits.open(load_filename)
        new_wavelengths = hdulist["WAVE"].data
        combined_fluxes = hdulist["COM_FLUXES"].data
        combined_errors = hdulist["COM_ERRORS"].data
        spline_cont0 = hdulist["SPLINE_CONT0"].data
        spline_paras0 = hdulist["SPLINE_PARAS0"].data
        try:
            wv_nodes = hdulist['wv_nodes'].data
        except:
            wv_nodes = hdulist['x_nodes'].data
        self.breads_header['STCONTRS'] = hdulist['BREADS'].header['STCONTRS']
        self.breads_header['STCONTTH'] = hdulist['BREADS'].header['STCONTTH']
        self.breads_header["STCONTFN"] = load_filename
        hdulist.close()

        self.check_and_update_nodes(wv_nodes)
        self.star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False, fill_value=1)
        return new_wavelengths,combined_fluxes,combined_errors,spline_cont0,spline_paras0,wv_nodes


    def compute_starsubtraction(self,  save_utils=False, threshold_badpix=10,mppool=None,combined_contnorm_filename=None,
                                only_identify_badpix = False,starsub_dir=None, load_starspectrum_contnorm = None,iterative=True):
        """
        Fit the spline model row by row, but including the stellar features with self.star_func(), which is the continuum-normalized star spectrum.


        Parameters
        ----------
        save_utils : Boolean
            Save the intermediate star subtraction step products.
        threshold_badpix : float (optional)
            Hard threshold for bad pixel flagging. Thresholding is done by comparing the continuum normalized row with its median absolute deviation.
        mppool : multiprocessing.Pool or None (optional)
            If None, the computation is done without parallelization.
        combined_contnorm_filename : str (optional)
            This can be used if one wants to use a combined continuum-normalized starlight model from an entire sequence.
            One should include the filename of the data product from get_contnorm_spec(); or from compute_starspectrum_contnorm().
            This is because compute_starspectrum_contnorm() only applies to individual exposures.
        only_identify_badpix : Boolean
            For internal use only! Do not use as a user. Use compute_advanced_badpix() instead.
            If True, only update the bad pixel map. If False, replace self.data by the starlight subtracted data.
            The latter would be used when aiming to get high-pass filtered spectrum of a companion.
        starsub_dir : str or None (optional)
            Name of the subdirectory (eg, "starsub1d") to save a copy of the original fits file and replace the cal image with the star-subtracted image.
            If None (default), those files won't be saved. This is typically not needed in a normal workflow.
        load_starspectrum_contnorm : str or None (optional)
            This should not be used unless the default filenames were changed, but it is not recommended. This is only to define the regularization of the spline.
            It should be the filename of the utility file saved by compute_starspectrum_contnorm() (meaning _save_starspectrum_contnorm()).
            If None, the default filename is used.
        iterative : Boolean (optional)
            If true, perform fit twice. First time to identify bad pixels. If False, only do it once.


        Returns
        -------
        subtracted_im : ndarray
            Star subtracted image.
        star_model : ndarray
            Image of the best fit model of the star
        spline_paras0 : ndarray (N_nodes x N_traces)
            Linear parameters returned by the spline fitting routine for each spectral trace of the detector.
        self.wv_nodes : 1d numpy array (N_nodes)
            Splines nodes spacing in the wavelengths dimension (micron).
        """
        if self.verbose:
            print(f"Computing star subtraction.")
        self.breads_header['STSUBTH'] = threshold_badpix

        if combined_contnorm_filename is not None:
            hdulist = pyfits.open(combined_contnorm_filename)
            new_wavelengths = hdulist["WAVE"].data
            combined_fluxes = hdulist["COM_FLUXES"].data
            wv_nodes = hdulist['wv_nodes'].data
            self.breads_header['STCONTRS'] = hdulist['BREADS'].header['STCONTRS']
            self.breads_header['STCONTTH'] = hdulist['BREADS'].header['STCONTTH']
            self.breads_header["STCONTFN"] = combined_contnorm_filename
            hdulist.close()

            self.check_and_update_nodes(wv_nodes)
            self.star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False, fill_value=1)

        # _get_starsub_inputs() takes care of transposing the images for MIRI compared to NIRSpec
        im, im_wvs, err, bad_pixels, reg_mean_map, reg_std_map,stellar_features0 = self._get_starsub_inputs(load_starspectrum_contnorm)

        if stellar_features0 is None:
            stellar_features = self.star_func(im_wvs)
        else:
            stellar_features = stellar_features0 * self.star_func(im_wvs)

        # Fit the model twice, the first time is used to identify and mask outliers from sigma clipping with threshold_badpix.
        if iterative:
            Nit = 2
        else:
            Nit = 1

        if reg_mean_map is None and reg_std_map is None:
            regularization = False
        else:
            regularization = True

        for i in range(Nit):
            star_model, _, new_badpixs, subtracted_im, spline_paras0 = normalize_rows(im, im_wvs, noise=err,
                                                                                  badpixs=bad_pixels,
                                                                                  wv_nodes=self.wv_nodes,
                                                                                  stellar_features=stellar_features,
                                                                                  threshold=threshold_badpix,
                                                                                  mppool=mppool,
                                                                                  regularization=regularization,
                                                                                  reg_mean_map=reg_mean_map,
                                                                                  reg_std_map=reg_std_map)
            bad_pixels = bad_pixels * new_badpixs

        self._set_bad_pixels(bad_pixels)
        subtracted_im[np.where(np.isnan(subtracted_im))] = 0

        if save_utils:
            self._save_starsubtraction(save_utils, subtracted_im, im, star_model, spline_paras0, starsub_dir)

        if not only_identify_badpix:
            self.data = subtracted_im
            self.breads_header["DATA_HPF"] = True
            self.breads_header["HPF_TYPE"] = "spline1d"

        return subtracted_im, star_model, spline_paras0, self.wv_nodes

    def _save_starsubtraction(self, save_utils, subtracted_im, im, star_model, spline_paras0, starsub_dir):
        """Save the star subtraction product in a fits file.

        ===  =============  ===  ==========  =====  ============  =======
        No.  Name           Ver  Type        Cards  Dimensions    Format
        ===  =============  ===  ==========  =====  ============  =======
        0    PRIMARY          1  PrimaryHDU    268  ()
        1    IM_SUB           1  ImageHDU       73  (2048, 2048)  float64
        2    IM               1  ImageHDU        8  (2048, 2048)  float32
        3    STARMODEL        1  ImageHDU        8  (2048, 2048)  float32
        4    BADPIX           1  ImageHDU        8  (2048, 2048)  float32
        5    SPLINE_PARAS0    1  ImageHDU        8  (40, 2048)    float64
        6    wv_nodes          1  ImageHDU        7  (40,)         float64
        7    BREADS           1  ImageHDU       21  ()
        ===  =============  ===  ==========  =====  ============  =======
        """

        if isinstance(save_utils, str):
            out_filename = save_utils
        else:
            out_filename = self.default_filenames["compute_starsubtraction"]

        _breads_header = copy(self.breads_header)
        _breads_header["DATA_HPF"] = True
        _breads_header["HPF_TYPE"] = "spline1d"
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(header=self.priheader))
        hdulist.append(pyfits.ImageHDU(data=subtracted_im,header=self.extheader,name="IM_SUB"))
        hdulist.append(pyfits.ImageHDU(data=im, name='IM'))
        hdulist.append(pyfits.ImageHDU(data=star_model, name='STARMODEL'))
        hdulist.append(pyfits.ImageHDU(data=self.bad_pixels, name='BADPIX'))
        hdulist.append(pyfits.ImageHDU(data=spline_paras0, name='SPLINE_PARAS0'))
        hdulist.append(pyfits.ImageHDU(data=self.wv_nodes, name='wv_nodes'))
        hdulist.append(pyfits.ImageHDU(header=_breads_header, name='BREADS'))
        hdulist.writeto(out_filename, overwrite=True)

        if starsub_dir is not None:
            if not os.path.exists(os.path.join(self.utils_dir, starsub_dir)):
                os.makedirs(os.path.join(self.utils_dir, starsub_dir))
            hdulist_sc = pyfits.open(self.filename)
            du = self.breads_header["DATAUNIT"]
            bu = self.extheader["BUNIT"].strip()
            if du == 'MJy' and bu == 'MJy':
                hdulist_sc["SCI"].data = subtracted_im
            if du == 'MJy/sr' and bu == 'MJy/sr':
                hdulist_sc["SCI"].data = subtracted_im
            if du == 'MJy/sr' and bu == 'MJy':
                hdulist_sc["SCI"].data = subtracted_im * self.area2d
            if du == 'MJy' and bu == 'MJy/sr':
                hdulist_sc["SCI"].data = subtracted_im / self.area2d
            hdulist_sc["DQ"].data[np.where(np.isnan(self.bad_pixels))] = 1
            hdulist_sc.writeto(os.path.join(self.utils_dir, starsub_dir, os.path.basename(self.filename)),
                               overwrite=True)
            hdulist_sc.close()

    def _set_bad_pixels(self, bad_pixels):
        """Set bad pixels map"""
        self.bad_pixels = bad_pixels

    def _get_starsub_inputs(self, load_starspectrum_contnorm):
        """ Get the inputs for the star subtraction routine """
        if load_starspectrum_contnorm is None:
            load_starspectrum_contnorm = self.default_filenames["compute_starspectrum_contnorm"]

        hdulist = pyfits.open(load_starspectrum_contnorm)
        spline_paras0 = hdulist['SPLINE_PARAS0'].data
        if 'STELLAR_FEATURES' in hdulist:
            stellar_features0 = hdulist['STELLAR_FEATURES'].data
        else:
            stellar_features0 = None
        hdulist.close()

        if stellar_features0 is None:
            wherenan = np.where(np.isnan(spline_paras0))
            reg_mean_map = copy(spline_paras0)
            reg_mean_map[wherenan] = np.tile(np.nanmedian(spline_paras0, axis=1)[:, None], (1, spline_paras0.shape[1]))[wherenan]
            reg_std_map = np.abs(spline_paras0)
            reg_std_map[wherenan] = np.tile(np.nanmax(np.abs(spline_paras0), axis=1)[:, None], (1, spline_paras0.shape[1]))[wherenan]
            reg_std_map = reg_std_map
            reg_std_map = np.clip(reg_std_map, 1e-11, np.inf)
        else:
            reg_mean_map = None
            reg_std_map = None

        im = np.copy(self.data)
        im_wvs = np.copy(self.wavelengths)
        err = np.copy(self.noise)

        bad_pixels = np.copy(self.bad_pixels)

        return im, im_wvs, err, bad_pixels, reg_mean_map, reg_std_map,stellar_features0


    def reload_starsubtraction(self, load_filename=None):
        """ Reload Star Subtraction

        Parameters
        ----------
        load_filename : str or None
            Filename to load PSF subtracted data from, or leave None to use default filename

        Returns
        -------
        subtracted_im, star_model, spline_paras0, wv_nodes

        Also modifies self.bad_pixels

        """
        if load_filename is None:
            load_filename = self.default_filenames["compute_starsubtraction"]
        if len(glob(load_filename)) ==0:
            return None

        hdulist = pyfits.open(load_filename)
        subtracted_im = hdulist["IM_SUB"].data
        star_model = hdulist["STARMODEL"].data
        fmderived_bad_pixels = hdulist['BADPIX'].data
        spline_paras0 = hdulist["SPLINE_PARAS0"].data
        wv_nodes = hdulist['wv_nodes'].data
        self.breads_header["DATA_HPF"] = hdulist["BREADS"].header["DATA_HPF"]
        self.breads_header["HPF_TYPE"] = hdulist["BREADS"].header["HPF_TYPE"]
        self.breads_header['STSUBTH'] = hdulist["BREADS"].header["STSUBTH"]
        self.breads_header['STCONTRS'] = hdulist['BREADS'].header['STCONTRS']
        self.breads_header['STCONTTH'] = hdulist['BREADS'].header['STCONTTH']
        self.breads_header["STCONTFN"] = hdulist['BREADS'].header['STCONTFN']
        hdulist.close()

        self.check_and_update_nodes(wv_nodes)

        self.bad_pixels = self.bad_pixels * fmderived_bad_pixels
        self.data = subtracted_im
        return subtracted_im, star_model, spline_paras0, wv_nodes


    ## 3dspline

    def compute_starspectrum_contnorm_3dspline(self,  save_utils=False,max_cores=1,
                                               spec_R_sampling=None, threshold_badpix=100,
                                               wv_nodes=None,N_wv_nodes=5,
                                               x_nodes=None,delta_x_nodes=0.02,
                                               y_nodes=None,delta_y_nodes=0.02,
                                               stamp_size = (0.2,0.2),save_plots=True):
        """ Compute star spectrum continuum normalized by 3d spline

        Parameters
        ----------

        Returns
        -------

        """
        if spec_R_sampling is None:
            self.breads_header["3DSPL_R"] = self.R*4
        else:
            self.breads_header["3DSPL_R"] = spec_R_sampling
        self.breads_header["3DSPL_TH"] = threshold_badpix
        self.breads_header["3DSPLSSX"] = stamp_size[0]
        self.breads_header["3DSPLSSY"] = stamp_size[1]

        _ifux,_ifuy = self.get_ifu_coords()

        if wv_nodes is None:
            wv_nodes = np.linspace(np.nanmin(self.wavelengths), np.nanmax(self.wavelengths), N_wv_nodes, endpoint=True)
        if x_nodes is None:
            x_nodes = np.arange(-2, 2.0001, delta_x_nodes)
        if y_nodes is None:
            y_nodes = np.arange(-2, 2.0001, delta_y_nodes)

        self.breads_header["3DSPL_NW"] = N_wv_nodes
        self.breads_header["3DSPL_DX"] = delta_x_nodes
        self.breads_header["3DSPL_DY"] = delta_y_nodes
        self.wv_nodes = wv_nodes
        self.x_nodes = x_nodes
        self.y_nodes = y_nodes

        if self.verbose:
            print(f"Computing stellar spectrum with 3d spline (continuum normalized)")

        if 0: # initialize regularization
            reg_mean_map_init = np.full((len(wv_nodes), len(y_nodes), len(x_nodes) ),np.nan)
            reg_std_map_init = np.full((len(wv_nodes), len(y_nodes), len(x_nodes) ),np.nan)

            regwvs_tmpobj = SimpleNamespace()

            # once again a function to manage the difference between NIRSpec and MIRI (see redefinition in jwstmiri_cal.py)
            Ntraces, Nwv = self._get_interpdata_shapes(wv_nodes)
            self._init_regwvs_obj(regwvs_tmpobj, Ntraces, Nwv)
            for trace_id in range(Ntraces):
                wvs_finite, where_finite = self._get_where_finite(trace_id)
                if np.size(wvs_finite[0]) == 0 or np.size(where_finite[0]) == 0:
                    continue
                # interpolates everything row by row. Different behavior between NIRSpec and MIRI
                self._interpdata_regwvs_trace(regwvs_tmpobj, wv_nodes, wvs_finite, where_finite, trace_id)

            xx, yy = np.meshgrid(x_nodes, y_nodes)

            for wv_id, wv in enumerate(wv_nodes):
                pointcloud_interp = point_cloud_interpolator_2d(regwvs_tmpobj.x, regwvs_tmpobj.y, wv_nodes,
                                                                regwvs_tmpobj.data, regwvs_tmpobj.bad_pixels, wv)
                pointcloud_interp_noise = point_cloud_interpolator_2d(regwvs_tmpobj.x, regwvs_tmpobj.y, wv_nodes,
                                                                regwvs_tmpobj.noise, regwvs_tmpobj.bad_pixels, wv)
                if pointcloud_interp is not None:
                    reg_mean_map_init[wv_id, :, :] = pointcloud_interp(xx,yy)
                    reg_std_map_init[wv_id, :, :] = pointcloud_interp_noise(xx,yy)*10
        else:
            reg_mean_map_init = None
            reg_std_map_init = None

        spline_cont0, _, new_badpixs, residuals, spline3d_paras_np,spline3d_paras_err_np = fit_3dspline(self, x_nodes,y_nodes,wv_nodes,stamp_size = stamp_size,
                                                            reg_mean_map=reg_mean_map_init, reg_std_map=reg_std_map_init,
                                                            max_cores=max_cores,threshold=threshold_badpix)


        continuum = copy(spline_cont0)
        continuum[np.where(continuum / self.noise < 5)] = np.nan
        continuum[np.where(continuum < np.median(continuum))] = np.nan
        continuum[np.where(np.isnan(self.bad_pixels))] = np.nan
        normalized_im = self.data / continuum
        normalized_err = self.noise / continuum

        new_wavelengths, combined_fluxes, combined_errors = combine_spectrum(self.wavelengths.flatten(),
                                                                             normalized_im.flatten(),
                                                                             normalized_err.flatten(),
                                                                             np.nanmedian(self.wavelengths) / self.breads_header["3DSPL_R"])

        if save_utils:
            if isinstance(save_utils,str):
                out_filename = save_utils
            else:
                out_filename = self.default_filenames["compute_starspectrum_contnorm_3dspline"]

            if hasattr(self, "filelist"):
                for fid,filename in enumerate(self.filelist):
                    self.breads_header["FILE{0}".format(fid)] = os.path.basename(filename)

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(header=self.priheader))
            hdulist.append(pyfits.ImageHDU(data=new_wavelengths, header=self.extheader, name="WAVE"))
            hdulist.append(pyfits.ImageHDU(data=combined_fluxes, name='COM_FLUXES'))
            hdulist.append(pyfits.ImageHDU(data=combined_errors, name='COM_ERRORS'))
            hdulist.append(pyfits.ImageHDU(data=spline_cont0, name='SPLINE_CONT0'))
            hdulist.append(pyfits.ImageHDU(data=spline3d_paras_np, name='SPLINE_PARAS0'))
            hdulist.append(pyfits.ImageHDU(data=spline3d_paras_err_np, name='SPLINE_PARAS0_ERR'))
            hdulist.append(pyfits.ImageHDU(data=wv_nodes, name='wv_nodes'))
            hdulist.append(pyfits.ImageHDU(data=x_nodes, name='x_nodes'))
            hdulist.append(pyfits.ImageHDU(data=y_nodes, name='y_nodes'))
            hdulist.append(pyfits.ImageHDU(data=normalized_im, name='CONT_NORM_IM'))
            hdulist.append(pyfits.ImageHDU(header=self.breads_header, name='BREADS'))
            hdulist.writeto(out_filename, overwrite=True)
            hdulist.close()

            if save_plots:
                dx_nodes = x_nodes[1]-x_nodes[0]
                dy_nodes = y_nodes[1]-y_nodes[0]
                extent = [x_nodes[0]-dx_nodes/2.0,x_nodes[-1]+dx_nodes/2.0,y_nodes[0]-dy_nodes/2.0,y_nodes[-1]+dy_nodes/2.0]
                spline3d_paras_toplot = np.nanmedian(spline3d_paras_np,axis=(0,1))
                vmax = 4+np.log10(np.abs(median_abs_deviation(spline3d_paras_toplot[np.where(np.isfinite(spline3d_paras_toplot))])))
                save_cube_as_gif(np.log10(np.abs(spline3d_paras_toplot)),filename=out_filename.replace(".fits", ".gif"),
                                 fps=3,vmin=vmax-6,vmax=vmax,extent=extent,wv_nodes=wv_nodes)

                wl = np.asarray(new_wavelengths)
                fl = np.asarray(combined_fluxes)
                err = np.asarray(combined_errors)

                fig = go.Figure()

                # -- Spectrum + error envelope ------------------------------------------
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([wl, wl[::-1]]),
                        y=np.concatenate([fl + err, (fl - err)[::-1]]),
                        fill="toself",
                        fillcolor="rgba(99,110,250,0.18)",
                        line=dict(width=0),
                        hoverinfo="skip",
                        name="±1s",
                        showlegend=True,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=wl, y=fl,
                        mode="lines",
                        line=dict(color="royalblue", width=1.1),
                        name="Flux",
                        hovertemplate="? = %{x:.4f} µm<br>Flux = %{y:.4f}<extra></extra>",
                    )
                )

                # Reference line at continuum = 1
                fig.add_hline(y=1.0, line=dict(color="gray", dash="dash", width=1))

                # -- Layout ------------------------------------------------------------
                fig.update_layout(
                    title=dict(text="Continuum-Normalized Spectrum", font=dict(size=16)),
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.01,
                                xanchor="right", x=1),
                    hovermode="x unified",
                    height=500,
                    margin=dict(l=70, r=30, t=70, b=60),
                    xaxis=dict(title="Wavelength (µm)", showgrid=True),
                    yaxis=dict(title="Normalized Flux", showgrid=True, zeroline=False),
                )
                fig.write_html(out_filename.replace(".fits", "_starspec.html"))

        self.star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False, fill_value=1)
        return new_wavelengths,combined_fluxes,combined_errors,spline_cont0,spline3d_paras_np,spline3d_paras_err_np,wv_nodes,x_nodes,y_nodes


    def reload_starspectrum_contnorm_3dspline(self, load_filename=None):
        """ Reload star spectrum normalized by continuum computed with 3dspline

        Parameters
        ----------
        load_filename : str or None
            Filename to load spectrum data from, or leave None to use default filename

        Returns
        -------
        new_wavelengths : 1d numpy array (N_wavelengths)
            New wavelengths axis of the combined high-frequency star spectrum (micron)
        combined_fluxes : 1d numpy array (N_wavelengths)
            Combined continuum normalized spectrum of the star.
        combined_errors : 1d numpy array (N_wavelengths)
            error vector for combined_fluxes
        spline_cont0 : 2d numpy array (N_detector_rows, N_detector_cols)
            Star continuum fitted by splines for each spectral trace of the detector.
        spline_paras0 : 2d numpy array (N_nodes, N_traces)
            Linear parameters returned by the continuum spline fitting routine for each spectral trace of the detector.
        wv_nodes : 1d numpy array
            Nodes spacing in the wavelength dimension (in micron).
        x_nodes : 1d numpy array
            Nodes spacing in the x ifu dimension (in arcsec).
        y_nodes : 1d numpy array
            Nodes spacing in the y ifu dimension (in arcsec).

        Also sets self.wv_nodes and self.star_func according to values in the reloaded file headers.

        """
        if load_filename is None:
            load_filename = self.default_filenames["compute_starspectrum_contnorm_3dspline"]
        if len(glob(load_filename)) ==0:
            return None

        hdulist = pyfits.open(load_filename)
        new_wavelengths = hdulist["WAVE"].data
        combined_fluxes = hdulist["COM_FLUXES"].data
        combined_errors = hdulist["COM_ERRORS"].data
        spline_cont0 = hdulist["SPLINE_CONT0"].data
        spline_paras0 = hdulist["SPLINE_PARAS0"].data
        spline_paras0_err = hdulist["SPLINE_PARAS0_ERR"].data
        wv_nodes = hdulist['wv_nodes'].data
        x_nodes = hdulist['x_nodes'].data
        y_nodes = hdulist['y_nodes'].data
        self.breads_header['3DSPL_R'] = hdulist['BREADS'].header['3DSPL_R']
        self.breads_header['3DSPL_TH'] = hdulist['BREADS'].header['3DSPL_TH']
        self.breads_header['3DSPLSSX'] = hdulist['BREADS'].header['3DSPLSSX']
        self.breads_header['3DSPLSSY'] = hdulist['BREADS'].header['3DSPLSSY']
        self.breads_header['3DSPL_NW'] = hdulist['BREADS'].header['3DSPL_NW']
        self.breads_header['3DSPL_DX'] = hdulist['BREADS'].header['3DSPL_DX']
        self.breads_header['3DSPL_DY'] = hdulist['BREADS'].header['3DSPL_DY']
        self.breads_header["3DSPL_FN"] = load_filename
        hdulist.close()

        self.check_and_update_nodes(wv_nodes, x_nodes, y_nodes)

        self.star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False, fill_value=1)
        return new_wavelengths,combined_fluxes,combined_errors,spline_cont0,spline_paras0,spline_paras0_err,wv_nodes,x_nodes,y_nodes


    def compute_starsubtraction_3dspline(self,  save_utils=False,max_cores=1,
                                         threshold_badpix=10,save_plots=True,iterative=False,
                                         only_identify_badpix = False,
                                         combined_contnorm_filename = None):
        """
        Computing star subtraction with 3d splines

        Parameters
        ----------

        Returns
        -------

        """
        if combined_contnorm_filename is None:
            combined_contnorm_filename = self.default_filenames["compute_starspectrum_contnorm_3dspline"]

        if combined_contnorm_filename is not None:
            hdulist = pyfits.open(combined_contnorm_filename)
            new_wavelengths = hdulist["WAVE"].data
            combined_fluxes = hdulist["COM_FLUXES"].data
            wv_nodes = hdulist['wv_nodes'].data
            x_nodes = hdulist['x_nodes'].data
            y_nodes = hdulist['y_nodes'].data
            spline_paras0 = hdulist["SPLINE_PARAS0"].data
            spline_paras0_err = hdulist["SPLINE_PARAS0_ERR"].data
            stamp_size = (hdulist['BREADS'].header['3DSPLSSX'],hdulist['BREADS'].header['3DSPLSSY'])
            self.breads_header["3DSPL_FN"] = combined_contnorm_filename
            hdulist.close()

            self.check_and_update_nodes(wv_nodes, x_nodes, y_nodes)

            self.star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False, fill_value=1)

        if not hasattr(self, 'star_func'):
            raise Exception("self.star_func should be defined to continue.")


        _ifux,_ifuy = self.get_ifu_coords()

        if self.verbose:
            print(f"Computing star subtraction with 3d splines")

        if 0:
            reg_mean_map_init =  np.nanmedian(spline_paras0,axis=(0,1))
            reg_std_map_init =  np.nanmedian(spline_paras0_err,axis=(0,1))*10
            # where_low_snr_prior = np.where((spline_paras0/spline_paras0_err)<5)
            # reg_mean_map_init[where_low_snr_prior] = np.nan
            # reg_std_map_init[where_low_snr_prior] = np.nan
        else:
            reg_mean_map_init = None
            reg_std_map_init = None

        stellar_features = self.star_func(self.wavelengths)

        if iterative:
            N_iter = 2
        else:
            N_iter = 1

        for k in range(N_iter):
            _out = fit_3dspline(self, x_nodes,y_nodes,wv_nodes,stamp_size = stamp_size,stellar_features=stellar_features,
                                reg_mean_map=reg_mean_map_init, reg_std_map=reg_std_map_init,
                                max_cores=max_cores,threshold=threshold_badpix)
            spline_cont0, _, new_badpixs, subtracted_im, spline3d_paras_np, spline3d_paras_err_np = _out
            self.bad_pixels = self.bad_pixels * new_badpixs

        subtracted_im[np.where(np.isnan(subtracted_im))] = 0

        if save_utils:
            if isinstance(save_utils,str):
                out_filename = save_utils
            else:
                out_filename = self.default_filenames["compute_starsubtraction_3dspline"]

            if hasattr(self, "filelist"):
                for fid,filename in enumerate(self.filelist):
                    self.breads_header["FILE{0}".format(fid)] = os.path.basename(filename)

            _breads_header = copy(self.breads_header)
            _breads_header["DATA_HPF"] = True
            _breads_header["HPF_TYPE"] = "spline3d"
            _breads_header["BPSFAREA"] = np.nanmedian(self.area2d)
            _breads_header["BPSFWV0"] = self.breads_header['WV_REF']
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(header=self.priheader))
            hdulist.append(pyfits.ImageHDU(data=subtracted_im,header=self.extheader,name="IM_SUB"))
            hdulist.append(pyfits.ImageHDU(data=self.data, name='IM'))
            hdulist.append(pyfits.ImageHDU(data=spline_cont0, name='STARMODEL'))
            hdulist.append(pyfits.ImageHDU(data=self.bad_pixels, name='BADPIX'))
            hdulist.append(pyfits.ImageHDU(data=spline3d_paras_np, name='SPLINE_PARAS0'))
            hdulist.append(pyfits.ImageHDU(data=spline3d_paras_err_np, name='SPLINE_PARAS0_ERR'))
            hdulist.append(pyfits.ImageHDU(data=wv_nodes, name='wv_nodes'))
            hdulist.append(pyfits.ImageHDU(data=x_nodes, name='x_nodes'))
            hdulist.append(pyfits.ImageHDU(data=y_nodes, name='y_nodes'))
            hdulist.append(pyfits.ImageHDU(header=_breads_header, name='BREADS'))
            hdulist.writeto(out_filename, overwrite=True)
            hdulist.close()

            if save_plots:
                mad_res = median_abs_deviation(subtracted_im[np.where(np.isfinite(subtracted_im*self.bad_pixels))])
                dx_nodes = x_nodes[1]-x_nodes[0]
                dy_nodes = y_nodes[1]-y_nodes[0]
                extent = [x_nodes[0]-dx_nodes/2.0,x_nodes[-1]+dx_nodes/2.0,y_nodes[0]-dy_nodes/2.0,y_nodes[-1]+dy_nodes/2.0]
                spline3d_paras_toplot = np.nanmedian(spline3d_paras_np,axis=(0,1))
                vmax = 4+np.log10(np.abs(median_abs_deviation(spline3d_paras_toplot[np.where(np.isfinite(spline3d_paras_toplot))])))
                save_cube_as_gif(np.log10(np.abs(spline3d_paras_toplot)),filename=out_filename.replace(".fits", ".gif"),
                                 fps=3,vmin=vmax-6,vmax=vmax,extent=extent,wv_nodes=wv_nodes)

                plt.figure(figsize=(16, 8))
                plt.subplot(1, 2, 1)
                plt.title("Before")
                plt.imshow(self.data[0:2048, :], origin='lower', cmap='viridis')
                plt.clim([-mad_res * 20, mad_res * 20])
                plt.subplot(1, 2, 2)
                plt.title("After")
                plt.imshow(subtracted_im[0:2048, :], origin='lower', cmap='viridis')
                plt.clim([-mad_res * 20, mad_res * 20])
                plt.savefig(out_filename.replace(".fits", "_before_after.png"), bbox_inches='tight', dpi=300)

                _ny, _nx = subtracted_im.shape
                x = np.arange(_ny)
                mask = new_badpixs[:, _nx // 2]

                fig = make_subplots(rows=2, cols=1, subplot_titles=("vertical cut of dataset", "Residuals"))
                fig.add_trace(go.Scatter(x=x, y=self.data[:, _nx // 2] * mask, name="Data"), row=1, col=1)
                fig.add_trace(go.Scatter(x=x, y=spline_cont0[:, _nx // 2] * mask, name="Model"), row=1, col=1)
                fig.add_trace(go.Scatter(x=x, y=subtracted_im[:, _nx // 2] * mask, name="Residuals"), row=1, col=1)
                fig.add_trace(go.Scatter(x=x, y=subtracted_im[:, _nx // 2] * mask, name="Residuals", showlegend=False),
                              row=2, col=1)
                fig.update_yaxes(title_text=f"Flux {self.breads_header['DATAUNIT']}", row=1, col=1)
                fig.update_yaxes(title_text=f"Flux {self.breads_header['DATAUNIT']}",
                                 range=[-mad_res * 10, mad_res * 10], row=2, col=1)
                fig.update_xaxes(title_text=f"Row index", row=2, col=1)
                fig.update_layout(height=700)
                fig.write_html(out_filename.replace(".fits", "_cut.html"))


        if not only_identify_badpix:
            where_finite_data = np.where(np.isfinite(self.data))
            self.data[where_finite_data] = subtracted_im[where_finite_data]
            self.breads_header["DATA_HPF"] = True
            self.breads_header["HPF_TYPE"] = "spline3d"
        return subtracted_im, spline_cont0, spline3d_paras_np,spline3d_paras_err_np, self.wv_nodes,self.x_nodes,self.y_nodes


    def reload_starsubtraction_3dspline(self, load_filename=None):
        """ Reload star subtracted data using 3dspline

        Parameters
        ----------
        load_filename : str or None
            Filename to load spectrum data from, or leave None to use default filename

        Returns
        -------

        """
        if load_filename is None:
            load_filename = self.default_filenames["compute_starsubtraction_3dspline"]
        if len(glob(load_filename)) ==0:
            return None

        hdulist = pyfits.open(load_filename)
        subtracted_im = hdulist['IM_SUB'].data
        spline_cont0 = hdulist['STARMODEL'].data
        spline3d_paras_np = hdulist['SPLINE_PARAS0'].data
        spline3d_paras_err_np = hdulist['SPLINE_PARAS0_ERR'].data
        new_bad_pixels = hdulist['BADPIX'].data
        wv_nodes = hdulist['wv_nodes'].data
        x_nodes = hdulist['x_nodes'].data
        y_nodes = hdulist['y_nodes'].data
        self.breads_header['DATA_HPF'] = hdulist['BREADS'].header['DATA_HPF']
        self.breads_header['HPF_TYPE'] = hdulist['BREADS'].header['HPF_TYPE']
        hdulist.close()

        self.check_and_update_nodes(wv_nodes, x_nodes, y_nodes)

        self.bad_pixels = self.bad_pixels * new_bad_pixels
        self.data = subtracted_im
        return subtracted_im, spline_cont0, spline3d_paras_np,spline3d_paras_err_np, self.wv_nodes,self.x_nodes,self.y_nodes

    def check_and_update_nodes(self,wv_nodes,x_nodes=None,y_nodes=None):
        """
        Make sure that there is no already some spline nodes defined in the class. If not, then update them.

        """
        if hasattr(self, 'wv_nodes'):
            if not np.array_equal(self.wv_nodes, wv_nodes):
                raise ValueError(
                    f"wv_nodes already defined and does not match the new value. "
                    f"Existing: {self.wv_nodes}, New: {wv_nodes}"
                )
        else:
            self.wv_nodes = wv_nodes
        if x_nodes is not None:
            if hasattr(self, 'x_nodes'):
                if not np.array_equal(self.x_nodes, x_nodes):
                    raise ValueError(
                        f"x_nodes already defined and does not match the new value. "
                        f"Existing: {self.x_nodes}, New: {x_nodes}"
                    )
            else:
                self.x_nodes = x_nodes
        if y_nodes is not None:
            if hasattr(self, 'y_nodes'):
                if not np.array_equal(self.y_nodes, y_nodes):
                    raise ValueError(
                        f"y_nodes already defined and does not match the new value. "
                        f"Existing: {self.y_nodes}, New: {y_nodes}"
                    )
            else:
                self.y_nodes = y_nodes

    def compute_advanced_badpix(self,  save_utils=False, threshold_badpix=10,mppool=None,combined_contnorm_filename=None,
                                starsub_dir=None, load_starspectrum_contnorm = None,iterative=True):
        """
        Same as compute_starsubtraction() but simply enforcing only_identify_badpix = True.
        """
        only_identify_badpix = True
        return self.compute_starsubtraction(save_utils=save_utils, threshold_badpix=threshold_badpix,mppool=mppool,
                                            combined_contnorm_filename=combined_contnorm_filename,
                                            only_identify_badpix = only_identify_badpix,starsub_dir=starsub_dir,
                                            load_starspectrum_contnorm = load_starspectrum_contnorm,iterative=iterative)

    def reload_advanced_badpix(self, load_filename=None):
        """ Reload advanced bad pixel map computed by compute_starsubtraction().

        Parameters
        ----------
        load_filename : str or None
            Filename to load PSF subtracted data from, or leave None to use default filename

        Returns
        -------
        bad_pixels
            Also modifies self.bad_pixels

        """
        if load_filename is None:
            load_filename = self.default_filenames["compute_starsubtraction"]
        if len(glob(load_filename)) == 0:
            return None

        hdulist = pyfits.open(load_filename)
        fmderived_bad_pixels = hdulist['BADPIX'].data
        hdulist.close()

        self.bad_pixels = self.bad_pixels * fmderived_bad_pixels
        return self.bad_pixels

    def compute_interpdata_regwvs(self, save_utils=False, wv_sampling=None):
        """Interpolate onto a regular wavelength sampling.

        Parameters
        ----------
        save_utils : bool
            Save data to the utils directory, or not
        wv_sampling : np.array
            Wavelength sampling to interpolate onto. If None, the regular wavelength sampling will be estimated from the data.

        Returns
        -------
        self

        """
        if "regwvs" in self.breads_header['COORDS']:
            raise Exception("This data object is already interpolated. Won't interpolate again.")

        if wv_sampling is None:
            if (not hasattr(self, "wv_sampling")) or self.wv_sampling is None:
                self.wv_sampling = self.get_regwvs_sampling()
            wv_sampling = self.wv_sampling
        else:
            self.wv_sampling = wv_sampling

        regwvs_tmpobj = SimpleNamespace()

        # once again a function to manage the difference between NIRSpec and MIRI (see redefinition in jwstmiri_cal.py)
        Ntraces, Nwv = self._get_interpdata_shapes(wv_sampling)
        self._init_regwvs_obj(regwvs_tmpobj, Ntraces, Nwv)

        for trace_id in range(Ntraces):
            wvs_finite, where_finite = self._get_where_finite(trace_id)

            # interpolates everything row by row. Different behavior between NIRSpec and MIRI
            self._interpdata_regwvs_trace(regwvs_tmpobj, wv_sampling, wvs_finite, where_finite, trace_id)

        where_bad = np.where(regwvs_tmpobj.bad_pixels != 1.0)
        regwvs_tmpobj.data[where_bad] = np.nan
        regwvs_tmpobj.noise[where_bad] = np.nan
        regwvs_tmpobj.bad_pixels[where_bad] = np.nan

        self.breads_header['COORDS'] = self.breads_header['COORDS'] + " regwvs"
        if save_utils:
            self._save_interpdata_regwvs(save_utils, regwvs_tmpobj)

        # replace the attributes in self with the interpolated ones
        self.x = regwvs_tmpobj.x
        self.y = regwvs_tmpobj.y
        self.wavelengths = regwvs_tmpobj.wavelengths
        self.leftnright_wavelengths = regwvs_tmpobj.leftnright_wavelengths
        self.data = regwvs_tmpobj.data
        self.noise = regwvs_tmpobj.noise
        self.bad_pixels = regwvs_tmpobj.bad_pixels
        self.area2d = regwvs_tmpobj.area2d

        return regwvs_tmpobj

    def _get_interpdata_shapes(self, wv_sampling):
        """ Get the shape of the interpolated data"""
        Ntraces, Nwv = self.data.shape[0], np.size(wv_sampling)
        return Ntraces, Nwv

    def _get_where_finite(self, trace_id):
        """ Get the mask for the traces"""
        wvs_finite = np.where(np.isfinite(self.wavelengths[trace_id, :]))
        where_finite = np.where(np.isfinite(self.bad_pixels[trace_id, :]))
        return wvs_finite, where_finite

    def _interpdata_regwvs_trace(self, regwvs_dataobj, wv_sampling, wvs_finite, where_finite, trace_id):

        if np.size(wvs_finite[0]) > 0:
            regwvs_dataobj.x[trace_id, :] = np.interp(wv_sampling, self.wavelengths[trace_id, wvs_finite[0]],
                                                          self.x[trace_id, wvs_finite[0]], left=np.nan, right=np.nan)
            regwvs_dataobj.y[trace_id, :] = np.interp(wv_sampling, self.wavelengths[trace_id, wvs_finite[0]],
                                                           self.y[trace_id, wvs_finite[0]], left=np.nan, right=np.nan)
            regwvs_dataobj.wavelengths[trace_id, :] = wv_sampling
            regwvs_dataobj.area2d[trace_id, :] = np.interp(wv_sampling, self.wavelengths[trace_id, wvs_finite[0]],
                                                    self.area2d[trace_id, wvs_finite[0]], left=np.nan, right=np.nan)
            badpix_mask = np.isfinite(self.bad_pixels[trace_id, :]).astype(float)
            regwvs_dataobj.bad_pixels[trace_id, :] = np.interp(wv_sampling, self.wavelengths[trace_id, wvs_finite[0]],
                                                        badpix_mask[wvs_finite], left=0, right=0)

            # following little section written by chatgpt to find the left and right wavelengths in the original data
            v_left, v_right = find_closest_leftnright_elements(self.wavelengths[trace_id, wvs_finite[0]], wv_sampling)

            regwvs_dataobj.leftnright_wavelengths[0, trace_id, :] = v_left
            regwvs_dataobj.leftnright_wavelengths[1, trace_id, :] = v_right

        if np.size(where_finite[0]) > 0:
            regwvs_dataobj.data[trace_id, :] = np.interp(wv_sampling, self.wavelengths[trace_id, where_finite[0]],
                                                  self.data[trace_id, where_finite[0]], left=np.nan, right=np.nan)
            regwvs_dataobj.noise[trace_id, :] = np.interp(wv_sampling, self.wavelengths[trace_id, where_finite[0]],
                                                   self.noise[trace_id, where_finite[0]], left=np.nan, right=np.nan)

    def _init_regwvs_obj(self, regwvs_dataobj, Ntraces, Nwv):
        """ Initialize the arrays for the interpolation on a regular wavelength grid"""
        regwvs_dataobj.x = np.full((Ntraces, Nwv), np.nan)
        regwvs_dataobj.y = np.full((Ntraces, Nwv), np.nan)
        regwvs_dataobj.wavelengths = np.full((Ntraces, Nwv), np.nan)
        regwvs_dataobj.leftnright_wavelengths = np.full((2, Ntraces, Nwv), np.nan)
        regwvs_dataobj.data = np.full((Ntraces, Nwv), np.nan)
        regwvs_dataobj.noise = np.full((Ntraces, Nwv), np.nan)
        regwvs_dataobj.bad_pixels = np.full((Ntraces, Nwv), np.nan)
        regwvs_dataobj.area2d = np.full((Ntraces, Nwv), np.nan)


    def _save_interpdata_regwvs(self, save_utils, regwvs_dataobj):
        """ Save the interpolation data in a fits file.

        No.    Name      Ver    Type      Cards   Dimensions   Format
          0  PRIMARY       1 PrimaryHDU     268   ()
          1  INTERP_DATA    1 ImageHDU        73   (2197, 2048)   float64
          2  INTERP_ERR    1 ImageHDU         8   (2197, 2048)   float64
          3  INTERP_X      1 ImageHDU         8   (2197, 2048)   float64
          4  INTERP_Y      1 ImageHDU         8   (2197, 2048)   float64
          5  INTERP_WAVE    1 ImageHDU         8   (2197, 2048)   float64
          6  INTERP_BADPIX    1 ImageHDU         8   (2197, 2048)   float64
          7  INTERP_AREA2D    1 ImageHDU         8   (2197, 2048)   float64
          8  INTERP_LEFTNRIGHT    1 ImageHDU         9   (2197, 2048, 2)   float64
          9  BREADS        1 ImageHDU        21   ()
        """
        if isinstance(save_utils, str):
            out_filename = save_utils
        else:
            out_filename = self.default_filenames["compute_interpdata_regwvs"]
        if bool(self.breads_header["DATA_HPF"]) and not out_filename.endswith("_starsub.fits"):
            out_filename = out_filename.replace(".fits", "_starsub.fits")

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(header=self.priheader))
        hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.data,header=self.extheader, name='INTERP_DATA'))
        hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.noise, name='INTERP_ERR'))
        hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.x, name='INTERP_X'))
        hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.y, name='INTERP_Y'))
        hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.wavelengths, name='INTERP_WAVE'))
        hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.bad_pixels, name='INTERP_BADPIX'))
        hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.area2d, name='INTERP_AREA2D'))
        hdulist.append(pyfits.ImageHDU(data=regwvs_dataobj.leftnright_wavelengths, name='INTERP_LEFTNRIGHT'))
        hdulist.append(pyfits.ImageHDU(header=self.breads_header,name="BREADS"))
        hdulist.writeto(out_filename, overwrite=True)
        hdulist.close()

    def reload_interpdata_regwvs(self, load_filename=None):
        """ Reload interpolated data onto regular wavelengths

        Parameters
        ----------
        load_filename

        Returns
        -------

        """
        if "regwvs" in self.breads_header['COORDS']:
            raise Exception("This data object is already interpolated. Won't interpolate again.")

        if load_filename is None:
            load_filename = self.default_filenames["compute_interpdata_regwvs"]

        if bool(self.breads_header["DATA_HPF"]) and not load_filename.endswith("_starsub.fits"):
            load_filename = load_filename.replace(".fits", "_starsub.fits")
        if len(glob(load_filename)) ==0:
            return None

        with pyfits.open(load_filename) as hdulist:
            self.data = hdulist['INTERP_DATA'].data
            self.noise = hdulist['INTERP_ERR'].data
            self.x  = hdulist['INTERP_X'].data
            self.y = hdulist['INTERP_Y'].data
            self.wavelengths  = hdulist['INTERP_WAVE'].data
            self.bad_pixels  = hdulist['INTERP_BADPIX'].data
            self.area2d = hdulist['INTERP_AREA2D'].data
            try:
                self.leftnright_wavelengths = hdulist['INTERP_LEFTNRIGHT'].data
            except KeyError:
                pass
            self.breads_header['COORDS'] = hdulist['BREADS'].header['COORDS']
            self.breads_header['DATAUNIT'] = hdulist['BREADS'].header['DATAUNIT']

        self.wv_sampling = np.nanmedian(self.wavelengths, axis=0)

        return self

    def mask_interp_elements_too_far_from_bin_edges(self, dwv_threshold):
        """ Mask interpolated elements too far from the edge bins

        Parameters
        ----------
        dwv_threshold

        Returns
        -------
        mask : ndarray
            Mask of which pixels are masked

        Also modifies self.bad_pixels

        """
        if "regwvs" not in self.breads_header['COORDS']:
            raise Exception("'regwvs' in self.breads_header['COORDS']. This data object needs to be interpolated first.")
        dist_to_bin_edges = np.nanmin(np.abs(self.leftnright_wavelengths - self.wavelengths), axis=0)
        mask = dist_to_bin_edges>dwv_threshold
        self.bad_pixels[np.where(mask)] = np.nan
        return mask

    def get_ifu_coords(self, ras=None, decs=None):
        """ Get IFU coordinates

        If ras and dec are None, self.x and self.y are being converted and returned.

        Parameters
        ----------
        ras : np.array
            Array of right ascension coordinates to be converted to IFU coordinates.
        decs : np.array
            Array of declination coordinates to be converted to IFU coordinates.

        Returns
        -------
        ifuX, ifuY : arrays

        """

        ifuX = None
        ifuY = None

        if ras is not None and decs is not None:
            ifuX, ifuY = rotate_coordinates(ras, decs, self.east2V2_deg, flipx=False)
        else:
            if "ifu" in self.breads_header['COORDS']:
                ifuX, ifuY =  self.x, self.y
            elif "sky" in self.breads_header['COORDS']:
                ifuX, ifuY = rotate_coordinates(self.x, self.y, self.east2V2_deg, flipx=False)
            else:
                raise ValueError(f"coords type must be either 'ifu' or 'sky' not {self.breads_header['COORDS']}")

        if ifuX is None or ifuY is None:
            raise ValueError("Error trying to get IFU coordinates")

        return ifuX, ifuY

    def get_sky_coords(self, ifux=None, ifuy=None):
        """ Get sky coordinates

        If ifux and ifuy are None, self.x and self.y are being converted and returned.

        Parameters
        ----------
        ifux: np.array
            X-spatial coordinate in the detector (ifu coords)
        ifuy: np.array
            Y-spatial coordinate in the detector (ifu coords)

        Returns
        -------
        dra_as_array, ddec_as_array : arrays

        """
        dra_as_array = None
        ddec_as_array = None

        if ifux is not None and ifuy is not None:
            dra_as_array, ddec_as_array = rotate_coordinates(ifux, ifuy, -self.east2V2_deg, flipx=False)
        else:
            if "sky" in self.breads_header['COORDS']:
                dra_as_array, ddec_as_array =  self.x, self.y
            elif "ifu" in self.breads_header['COORDS']:
                dra_as_array, ddec_as_array = rotate_coordinates(self.x, self.y, -self.east2V2_deg, flipx=False)

        if dra_as_array is None or ddec_as_array is None:
            raise ValueError("Error trying to get sky coordinates")

        return dra_as_array, ddec_as_array

    def broaden(self, wvs, spectrum, loc=None, mppool=None):
        """ Broaden a spectrum to the resolution of this data object using the resolution attribute (self.R).

        LSF is assumed to be a 1D gaussian.
        The broadening is technically fiber dependent so you need to specify which fiber calibration to use.

        Parameters
        ----------
        wvs : ndarray
            Wavelength sampling of the spectrum to be broadened.
        spectrum : ndarray
            1D spectrum to be broadened.
        loc : None
            To be ignored. Could be used in the future to specify (x,y) position if field dependent resolution is
            available.
        mppool : multiprocessing Pool
            Multiprocessing pool to parallelize the code. If None (default), no parallelization is applied.
            E.g. mppool = mp.Pool(processes=10) # 10 is the number processes

        Returns
        -------
            Broadened spectrum

        """
        return broaden(wvs, spectrum, self.R, mppool=mppool)

    def get_regwvs_sampling(self):
        """ Get a regular wavelength sampling

        Determines the min and max values, and median step size, for the provided wavelengths array
        Computes a regular grid using that median sampling

        Returns
        -------
        wv_sampling : array
            Even regular wavelength values

        """
        wv_min, wv_max = np.nanmin(self.wavelengths), np.nanmax(self.wavelengths)
        sampling_dw = np.nanmedian(self.wavelengths[:, 1::] - self.wavelengths[:, 0:self.wavelengths.shape[1] - 1])
        wv_sampling = np.arange(wv_min, wv_max, sampling_dw)
        return wv_sampling

    def where_point_source(self, radec_as, rad_as):
        """Which pixels in a point cloud are within some given radius of a given location?

        Parameters
        ----------
        dataobj : data object
            Point cloud data object
        radec_as : tuple of floats
            RA, Dec coordinates of interest
        rad_as : float
            Radius in arcseconds

        Returns
        -------
        dist2pointsource_as : 2d Boolean mask
            Boolean mask of pixels within given radius of a given location
        """
        ra, dec = radec_as
        dist2pointsource_as = np.sqrt((self.x - ra) ** 2 + (self.y - dec) ** 2)
        return np.where(dist2pointsource_as < rad_as)


    def get_2D_point_cloud_interpolator(self, wv0=None, replace_data = None):
        """
        Generate a 2D point cloud interpolator at a given wavelength.

        Parameters
        ----------
        wv0 : float
            Wavelength slice at which to interpolate. Since the wavelength sampling is discrete, the function will just pick the closest wavelength sample.
        replace_data : np.array
            If not None, this array will be used instead of self.data for the interpolation. This can be useful if you want to interpolate something other than the original data (e.g. best fit model, residuals, etc.).

        Returns
        -------
        pointcloud_interp : scipy.interpolate.LinearTriInterpolator
            A 2D interpolator object that can be used to evaluate the interpolated data at any (x,y) position.

        """
        if "regwvs" not in self.breads_header['COORDS']:
            raise ValueError("Data needs to be interpolated on a regular wavelength grid. Please run compute_interpdata_regwvs().")

        if replace_data is None:
            pointcloud_interp = point_cloud_interpolator_2d(self.x, self.y, self.wv_sampling, self.data, self.bad_pixels, wv0)
        else:
            pointcloud_interp = point_cloud_interpolator_2d(self.x, self.y, self.wv_sampling, replace_data, self.bad_pixels, wv0)

        return pointcloud_interp

    def plot_2D_point_cloud(self, wv0=None, pointcloud_interp=None, x_vec=None, y_vec=None,save_plot=False,overlay_pointcloud=False):
        """
        Plot the 2D point cloud at a given wavelength.

        Parameters
        -------
        wv0 : float
            Wavelength slice at which to interpolate. Since the wavelength sampling is discrete, the function will just pick the closest wavelength sample.
            Is ignored if pointcloud_interp is provided, since that already corresponds to a specific wavelength slice.
        pointcloud_interp : scipy.interpolate.LinearTriInterpolator
            A 2D interpolator object that can be used to evaluate the interpolated data at any (x,y) position.
            see self.get_2D_point_cloud_interpolator()
        x_vec : np.array
            Array of x coordinates in arcsec.
            This is right ascension if sky coordinates (see self.breads_header['COORDS']).
        y_vec : np.array
            Array of y coordinates in arcsec.
            This is declination if sky coordinates (see self.breads_header['COORDS']).
        save_plot : Bool
            Whether to save the figure to disk or not.
        overlay_pointcloud: Bool
            Whether to overlay the original point cloud data points on top of the interpolated image.

        Returns
        -------

        """
        if "regwvs" not in self.breads_header['COORDS']:
            raise ValueError("Data needs to be interpolated on a regular wavelength grid. Please run compute_interpdata_regwvs().")

        if wv0 is None and pointcloud_interp is None:
            wv0 = np.nanmedian(self.wv_sampling)

        if pointcloud_interp is None:
            pointcloud_interp = self.get_2D_point_cloud_interpolator(wv0)

        if x_vec is None:
            x_vec = np.linspace(-3, 3, 60)
            x_vec += np.nanmedian(self.x)
        if y_vec is None:
            y_vec = np.linspace(-3, 3, 60)
            y_vec += np.nanmedian(self.y)

        dramin, dramax, ddecmin, ddecmax = np.min(x_vec), np.max(x_vec), np.min(y_vec), np.max(y_vec)
        dx_halfpix = (x_vec[1] - x_vec[0])/2.
        dy_halfpix = (y_vec[1] - y_vec[0])/2.
        extent = [dramin-dx_halfpix, dramax+dx_halfpix, ddecmin-dy_halfpix, ddecmax+dy_halfpix]

        inp = np.meshgrid(x_vec, y_vec)
        out = pointcloud_interp(inp[0], inp[1])

        fig = plt.figure(figsize=(6, 6))
        im = plt.imshow(np.log10(abs(out)), origin='lower', extent=extent, aspect='equal')

        if overlay_pointcloud:
            wv0_index = np.argmin(np.abs(self.wv_sampling - wv0))
            where_good = np.where(np.isfinite(self.bad_pixels[:, wv0_index]))
            x = self.x[where_good[0], wv0_index]
            y = self.y[where_good[0], wv0_index]
            plt.scatter(x,y,s=0.1,c="black")

        plt.xlim([extent[0], extent[1]])
        plt.ylim([extent[2], extent[3]])
        cbar = fig.colorbar(im, fraction=0.05, pad=0.04)
        unit = self.breads_header['DATAUNIT']
        label = r'log$_{10} \left(\frac{\mathrm{flux}}{\mathrm{' + unit + r'}}\right)$'
        cbar.set_label(label)
        if "ifu" in self.breads_header['COORDS']:
            plt.xlabel('IFU x (arcsec)')
            plt.ylabel('IFU y (arcsec)')
        elif "sky" in self.breads_header['COORDS']:
            plt.xlabel(r'$\Delta$RA (arcsec)')
            plt.ylabel(r'$\Delta$Dec (arcsec)')
        plt.title(os.path.basename(self.filename)+" Coords: "+self.breads_header['COORDS'])

        if save_plot:
            fig_filename = os.path.join(self.utils_dir, os.path.basename(self.filename).replace(".fits",f"_2D_point_cloud_wv{wv0:.4f}.png"))
            print(f"Saving plot in {fig_filename}")
            plt.savefig(fig_filename, bbox_inches='tight', dpi=200)

        return fig

    def plot_2D_point_cloud_html(self, wv0=None, pointcloud_interp=None, x_vec=None, y_vec=None, save_plot=False,overlay_pointcloud=False):
        """
        Plot the 2D point cloud at a given wavelength, saving as an interactive HTML file via Plotly.
        Parameters
        -------
        wv0 : float
            Wavelength slice at which to interpolate. Since the wavelength sampling is discrete, the function will just pick the closest wavelength sample.
            Is ignored if pointcloud_interp is provided, since that already corresponds to a specific wavelength slice.
        pointcloud_interp : scipy.interpolate.LinearTriInterpolator
            A 2D interpolator object that can be used to evaluate the interpolated data at any (x,y) position.
            see self.get_2D_point_cloud_interpolator()
        x_vec : np.array
            Array of x coordinates in arcsec.
            This is right ascension if sky coordinates (see self.breads_header['COORDS']).
        y_vec : np.array
            Array of y coordinates in arcsec.
            This is declination if sky coordinates (see self.breads_header['COORDS']).
        save_plot : Bool
            Whether to save the figure to disk or not.
        overlay_pointcloud: Bool
            Whether to overlay the original point cloud data points on top of the interpolated image.

        Returns
        -------
        fig : plotly.graph_objects.Figure
        """
        if "regwvs" not in self.breads_header['COORDS']:
            raise ValueError(
                "Data needs to be interpolated on a regular wavelength grid. Please run compute_interpdata_regwvs().")
        if wv0 is None and pointcloud_interp is None:
            wv0 = np.nanmedian(self.wv_sampling)
        if pointcloud_interp is None:
            pointcloud_interp = self.get_2D_point_cloud_interpolator(wv0)
        if x_vec is None:
            x_vec = np.linspace(-3, 3, 240)
            x_vec += np.nanmedian(self.x)
        if y_vec is None:
            y_vec = np.linspace(-3, 3, 240)
            y_vec += np.nanmedian(self.y)

        dramin, dramax, ddecmin, ddecmax = np.min(x_vec), np.max(x_vec), np.min(y_vec), np.max(y_vec)
        dx_halfpix = (x_vec[1] - x_vec[0]) / 2.
        dy_halfpix = (y_vec[1] - y_vec[0]) / 2.
        inp = np.meshgrid(x_vec, y_vec)
        out = pointcloud_interp(inp[0], inp[1])
        z = np.log10(np.abs(out))


        if "ifu" in self.breads_header['COORDS']:
            xlabel = 'IFU x (arcsec)'
            ylabel = 'IFU y (arcsec)'
        elif "sky" in self.breads_header['COORDS']:
            xlabel = '\u0394RA (arcsec)'
            ylabel = '\u0394Dec (arcsec)'
        else:
            xlabel = 'x (arcsec)'
            ylabel = 'y (arcsec)'

        unit = self.breads_header['DATAUNIT']
        colorbar_label = f'log\u2081\u2080 (flux / {unit})'
        title = os.path.basename(self.filename) + " Coords: " + self.breads_header['COORDS']

        fig = go.Figure()

        # Heatmap layer
        fig.add_trace(go.Heatmap(
            z=z,
            x=x_vec,
            y=y_vec,
            colorscale='Viridis',
            colorbar=dict(title=dict(text=colorbar_label, side='right')),
            hovertemplate=xlabel + ': %{x:.3f}<br>' + ylabel + ': %{y:.3f}<br>log\u2081\u2080(flux): %{z:.3f}<extra></extra>',
        ))
        if overlay_pointcloud:
            # Scatter overlay: good pixels at this wavelength
            wv0_index = np.argmin(np.abs(self.wv_sampling - wv0))
            where_good = np.where(np.isfinite(self.bad_pixels[:, wv0_index]))
            x_scatter = self.x[where_good[0], wv0_index]
            y_scatter = self.y[where_good[0], wv0_index]
            # Scatter overlay layer
            fig.add_trace(go.Scatter(
                x=x_scatter,
                y=y_scatter,
                mode='markers',
                marker=dict(size=2, color='black'),
                name='good pixels',
                hovertemplate=xlabel + ': %{x:.3f}<br>' + ylabel + ': %{y:.3f}<extra></extra>',
            ))

        fig.update_layout(
            title=title,
            xaxis=dict(
                title=xlabel,
                range=[dramax - 0.1, dramin + 0.2],  # reversed to match plt.xlim
                autorange=False,
            ),
            yaxis=dict(
                title=ylabel,
                range=[ddecmin + 0.1, ddecmax - 0.1],
                autorange=False,
                scaleanchor='x',  # enforce equal aspect ratio
                scaleratio=1,
            ),
            width=700,
            height=700,
        )

        if save_plot:
            fig_filename = os.path.join(
                self.utils_dir,
                os.path.basename(self.filename).replace(".fits", f"_2D_point_cloud_wv{wv0:.4f}.html")
            )
            print(f"Saving plot in {fig_filename}")
            fig.write_html(fig_filename)

        return fig

    def save(self, filename = None,suffix=None) -> None:
        """Save the object to a pickle file."""
        if filename is None:
            if suffix is None:
                suffix = ""
                if bool(self.breads_header["DATA_HPF"]):
                    suffix = suffix + "_HPF"+self.breads_header["HPF_TYPE"]
                suffix = suffix + "_"+self.breads_header['COORDS'].replace(" ","_")
            pickle_filename = os.path.join(self.utils_dir,os.path.basename(self.filename).replace(".fits",suffix+".pkl"))
        else:
            pickle_filename = filename

        with open(pickle_filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str) -> "MyClass":
        """Load an object from a pickle file."""
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj

#### Functions
def _get_wpsf_task(paras):
    """ Run WebbPSF for a single wavelength. Utility function for compute_webbpsf_model.

    Arguments
    ---------
    paras : tuple containing all arguments (TODO why is this set up this way?!?)
        Must contain IFU, center_wv, wpsf_oversample, opmode, parallelize

    Returns
    --------
    webbpsfim :  ndarray cube
        the OVERSAMP extension only from the computed PSF
    smoothed_im : ndarray cube
        The OVERSAMP extension, after smoothing by a Gaussian kernel based on the oversampling
    webbpsf_header : fits.header
        WebbPSF header corresponding to the OVERSAMP extension

    #TODO this should be updated to use OVERDIST probably!! Instead of the smoothing here. TBD.
    """
    IFU, center_wv, wpsf_oversample, opmode, parallelize, fov_arcsec = paras
    if opmode=='FIXEDSLIT':
        if not parallelize:
            print('FixedSlit webbpsf kernel...')
        else:
            rprint('FixedSlit webbpsf kernel...')
        kernel = np.ones((wpsf_oversample, wpsf_oversample*2))
    elif opmode=='IFU':
        kernel = np.ones((wpsf_oversample, wpsf_oversample))
    else:
        raise Exception('OPMODE unknown')

    ext = 'OVERSAMP'
    webbpsf_wv0_hdulist = IFU.calc_psf(monochromatic=center_wv * 1e-6,  # Wavelength, in **METERS**
                                fov_arcsec=fov_arcsec,  # angular size to simulate PSF over
                                oversample=wpsf_oversample,
                                # output pixel scale relative to the pixelscale set above
                                add_distortion=False)  # skip an extra computation step that's not relevant for IFU
    webbpsfim = webbpsf_wv0_hdulist[ext].data
    webbpsf_header = webbpsf_wv0_hdulist[ext].header
    smoothed_im = convolve2d(webbpsfim, kernel, mode='same') / wpsf_oversample ** 2
    return webbpsfim, smoothed_im, webbpsf_header


def untangle_dq(arr, verbose=True):
    """Reshape and unpack Data Quality array from ints using a bitmask to a datacube of individual bits

    Got help from ChatGPT.

    Parameters
    ----------
    verbose
    arr

    Returns
    -------

    """
    if verbose:
        print("\tUnpacking data quality bitmasks. DQ array is of type", arr.dtype)
    # Assume arr is your input numpy array of shape (ny, nx)
    ny, nx = arr.shape

    # Create a new numpy array of shape (32, ny, nx) to hold the cube
    cube = np.zeros((32, ny, nx), dtype=bool)

    # Create a mask array to extract the individual bits of each integer in the input array
    mask = np.array([1 << i for i in range(32)], dtype=arr.dtype)

    # Use NumPy's bitwise AND operator to extract the individual bits of each integer
    bits = (arr[..., np.newaxis] & mask[np.newaxis, np.newaxis, :]) > 0

    # Transpose the bits array and assign it to the first dimension of the cube array
    cube[:, :, :] = bits.transpose(2, 0, 1)
    return cube


# def fit_webbpsf(sc_im, sc_im_wvs, noise, bad_pixels, dra_as_array, ddec_as_array, interpolator, psf_wv0, fix_cen=None):
#     """Fit a webbpsf model to an image
#     #todo delete?
#
#     Parameters
#     ----------
#     sc_im
#     sc_im_wvs
#     noise
#     bad_pixels
#     dra_as_array
#     ddec_as_array
#     interpolator
#     psf_wv0
#     fix_cen
#
#     Returns
#     -------
#     bestfit_paras, psfsub_model_im, psfsub_sc_im
#
#     """
#     wv_min, wv_max = np.nanmin(sc_im_wvs), np.nanmax(sc_im_wvs)
#     wv_sampling = np.exp(np.arange(np.log(wv_min), np.log(wv_max), np.log(1 + 0.5 / 2700.)))
#
#     dist2host_as = np.sqrt(dra_as_array ** 2 + ddec_as_array ** 2)
#
#     psfsub_sc_im = np.full(sc_im.shape, np.nan)
#     psfsub_model_im = np.zeros_like(sc_im)
#     bestfit_paras = np.full((4, np.size(wv_sampling)), np.nan)
#     for wv_id, left_wv in enumerate(wv_sampling):
#         center_wv = left_wv * (1 + 0.25 / 2700) #TODO change 2700 hardcodeing, replace with spectral resolution?
#         right_wv = left_wv * (1 + 0.5 / 2700)
#
#         where_fit_mask = np.where(
#             np.isfinite(sc_im) * (noise != 0) * (np.isfinite(bad_pixels)) * (left_wv < sc_im_wvs) * (
#                         sc_im_wvs < right_wv) * (dist2host_as < 1.0))  # *(dist2host_as>0.5)
#         where_sc_mask = np.where(np.isfinite(sc_im) * (noise != 0) * (left_wv < sc_im_wvs) * (sc_im_wvs < right_wv))
#         Xfit = dra_as_array[where_fit_mask]
#         Yfit = ddec_as_array[where_fit_mask]
#         Zfit = sc_im[where_fit_mask]
#         Zerr2_fit = (noise[where_fit_mask]) ** 2
#
#         Xsc = dra_as_array[where_sc_mask]
#         Ysc = ddec_as_array[where_sc_mask]
#         Zsc = sc_im[where_sc_mask]
#
#         if (np.size(where_fit_mask[0]) < 377 / 4) or (np.size(where_sc_mask[0]) < 736 / 2):
#             print("Not enough points", wv_id, center_wv, np.size(where_fit_mask[0]), np.size(where_sc_mask[0]))
#             bestfit_paras[:, wv_id] = np.array([center_wv, np.nan, np.nan, np.nan])
#             psfsub_model_im[where_sc_mask] = np.nan
#             psfsub_sc_im[where_sc_mask] = np.nan
#             continue
#
#         if fix_cen is None:
#             m0 = interpolator(Xfit * psf_wv0 / center_wv, Yfit * psf_wv0 / center_wv)
#             a0 = np.nansum(Zfit * m0 / Zerr2_fit) / np.nansum(m0 ** 2 / Zerr2_fit)
#
#             # Define the function to fit
#             def myfunc(coords, xc, yc, A):
#                 _x, _y = coords[0], coords[1]
#                 znew = A * interpolator(_x - xc, _y - yc)
#                 return znew
#
#             # Define the initial parameter values for the fit
#             p0 = [0, 0, a0]
#             # Fit the data to the function
#             try:
#                 params, _ = curve_fit(myfunc, np.array([Xfit * psf_wv0 / center_wv, Yfit * psf_wv0 / center_wv]), Zfit,
#                                       p0=p0, method='lm', ftol=1e-6, xtol=1e-6)
#             except:
#                 print("curve_fit failed", wv_id, center_wv, np.size(where_fit_mask[0]), np.size(where_sc_mask[0]))
#                 bestfit_paras[:, wv_id] = np.array([center_wv, np.nan, np.nan, np.nan])
#                 psfsub_model_im[where_sc_mask] = np.nan
#                 psfsub_sc_im[where_sc_mask] = np.nan
#                 continue
#             # Extract the optimized parameter values
#             xc, yc, a = params
#
#         else:
#             m0 = interpolator((Xfit - fix_cen[0]) * psf_wv0 / center_wv, (Yfit - fix_cen[1]) * psf_wv0 / center_wv)
#             a0 = np.nansum(Zfit * m0 / Zerr2_fit) / np.nansum(m0 ** 2 / Zerr2_fit)
#             xc, yc, a = 0, 0, a0
#         psfmodel = a * interpolator(Xsc * psf_wv0 / center_wv - xc, Ysc * psf_wv0 / center_wv - yc)
#         psfsub_Zsc = Zsc - psfmodel
#
#         bestfit_paras[:, wv_id] = np.array(
#             [center_wv, xc * center_wv / psf_wv0, yc * center_wv / psf_wv0, a * interpolator(0, 0)])
#         psfsub_model_im[where_sc_mask] = psfmodel
#         psfsub_sc_im[where_sc_mask] = psfsub_Zsc
#
#     return bestfit_paras, psfsub_model_im, psfsub_sc_im











