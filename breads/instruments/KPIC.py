from breads.instruments.instrument import Instrument
import breads.utils as utils
from warnings import warn
import astropy.io.fits as pyfits
import numpy as np
import ctypes
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.time import Time
from copy import copy
from copy import deepcopy
from  scipy.interpolate import interp1d
from breads.utils import findbadpix
from breads.utils import broaden
import warnings

class KPIC(Instrument):
    def __init__(self, spec=None, trace=None, wvs=None, err=None, badpix=None,baryrv=None,orders=None,combine_mode="planet",fiber_goal_list = None):
        super().__init__('KPIC')
        if spec is None:
            warning_text = "No data file provided. " + \
            "Please manually add data using OSIRIS.manual_data_entry() or add data using OSIRIS.read_data_file()"
            warn(warning_text)
        else:
            self.read_data_file(spec, trace, wvs,err,badpix,baryrv,orders,combine_mode,fiber_goal_list)

    def read_data_file(self, spec, trace, wvs, err=None, badpix=None,baryrv=None,orders=None,combine_mode=None,fiber_goal_list = None):
        """
        Read OSIRIS spectral cube, also checks validity at the end

        Args:
            fiber_goal_list: List of the fibers that was being tracked. (indexed from 0)
                If None (default), will use GOALNM keyword.
                If "brightest", defines the fiber being tracked from the brightest trace in the array
                If nd.array, user-defined e.g. [0,0,1,1,2,2,3,3]
            combine_mode: if "star", for combining a sequence of on-axis star observations accounting for variable stellar intensity.
                    if "companion", just a weighted mean using the spectra errors.


        """

        if type(trace) is np.ndarray:
            line_width = trace
            self.res_fib_labels = ["s1","s2","s3","s4"]
        else:
            with pyfits.open(trace) as hdulist:
                arr = hdulist[0].data
                Nfib,nord,npix = arr.shape
                line_width = np.reshape(arr,(Nfib,nord*npix))
                self.res_fib_labels = get_fib_labels(hdulist[0].header)

        if type(wvs) is np.ndarray:
            self.wavelengths = wvs
            self.wvs_fib_labels = ["s1","s2","s3","s4"]
        else:
            with pyfits.open(wvs) as hdulist:
                arr = hdulist[0].data
                Nfib,nord,npix = arr.shape
                self.wavelengths = np.reshape(arr,(Nfib,nord*npix))
                self.wvs_fib_labels = get_fib_labels(hdulist[0].header)

        if self.wavelengths.shape[1] == 4 and line_width.shape[0] ==5:
            # hard coded for epoch 20210704 when the trace calibration included the gas cell, but not the wavcal
            self.wavelengths = np.concatenate([np.zeros((self.wavelengths.shape[0],1)),self.wavelengths],axis=1)

        if self.wavelengths.shape[0] != line_width.shape[0]:
            new_wvs = np.zeros(line_width.shape)
            new_wvs_fib_labels = []
            N_science_fibers = np.sum([1 if "s" in label else 0 for label in self.res_fib_labels])
            for k,fib_label in enumerate(self.res_fib_labels):
                fibnum = int(fib_label[1:2])
                # print(fib_label,fibnum,np.where("s{0}".format((fibnum%N_science_fibers)+1)==self.wvs_fib_labels)[0])
                new_wvs[k,:] = self.wavelengths[np.where("s{0}".format((fibnum%N_science_fibers)+1)==self.wvs_fib_labels)[0],:]
                new_wvs_fib_labels.append("s{0}".format(fibnum))
            self.wavelengths = new_wvs
            self.wvs_fib_labels = new_wvs_fib_labels
        dwvs = self.wavelengths[:, 1::] - self.wavelengths[:, 0:-1]
        dwvs = np.concatenate([dwvs, dwvs[:, -1][:, None]], axis=1)

        line_FWHM_wvunit = line_width* dwvs*2*np.sqrt(2*np.log(2))
        self.resolution = self.wavelengths/line_FWHM_wvunit

        # Make wavelength dimension first
        self.wavelengths = self.wavelengths.T
        self.resolution = self.resolution.T



        if type(spec) is np.ndarray:
            self.data = spec
            self.noise = err
            self.bad_pixels = badpix
            self.bary_RV = baryrv
            self.orders = orders
            self.fiber_goal_list = fiber_goal_list
        else:
            if type(spec) is not list:
                filelist = [spec]
            else:
                filelist = spec

            baryrv_list = []
            data_list = []
            noise_list = []
            fiber_list_from_hdr=[]
            GOALNM_exists = True
            for filename in filelist:
                print(filename)
                with pyfits.open(filename) as hdulist:
                    Nfib,Norder,Npix = hdulist[0].data.shape
                    data_list.append(hdulist[0].data)
                    header = hdulist[0].header
                    noise_list.append(hdulist[1].data)
                    baryrv_list.append(float(header["BARYRV"]))
                    try:
                        fiber_list_from_hdr.append(int(header["GOALNM"][-1])-1)
                    except:
                        GOALNM_exists = False

                sf_id_list = []
                sf_num_list = []
                for fibid in range(20):
                    try:
                        fiblabel = header["FIB{0}".format(fibid)]
                        if "s" in fiblabel:
                            sf_id_list.append(fibid)
                            sf_num_list.append(fiblabel[1:2])
                    except:
                        pass

            data_list = np.array(data_list)
            noise_list = np.array(noise_list)

            combined_spec = np.zeros((Nfib,Norder,Npix))
            combined_spec_sig = np.zeros((Nfib,Norder,Npix))
            #     If None (default), will use GOALNM keyword.
            #     If "brightest", defines the fiber being tracked from the brightest trace in the array
            #     If nd.array, user-defined e.g. [0,0,1,1,2,2,3,3]
            if fiber_goal_list is None:
                if GOALNM_exists:
                    self.fiber_goal_list = np.array(fiber_list_from_hdr)
                else:
                    raise Exception("GOALNM keyword not found in header. Please define fiber_goal_list.")
            elif fiber_goal_list == "brightest":
                self.fiber_goal_list = np.argmax(np.nansum(data_list, axis=(2, 3)), axis=1)
            else:
                self.fiber_goal_list = np.array(fiber_goal_list)

            if combine_mode is None:
                if len(filelist) > 1:
                    raise Exception("the input 'combine_mode' should not be None. Please choose 'star' or 'companion'")
                else:
                    combine_mode = "companion"

            for fib in range(Nfib):
                where_fib = np.where(self.fiber_goal_list == fib)[0]
                if len(where_fib) == 0:
                    continue
                for order in range(Norder):
                    badpix = findbadpix(data_list[where_fib,fib,order,:].T[:,:,None], noisecube=noise_list[where_fib,fib,order,:].T[:,:,None], badpixcube=None,
                                        chunks=5, mypool=None, med_spec=None,nan_mask_boxsize=0,threshold=5)[0][:,:,0].T
                    data_list[where_fib,fib,order,:] *= badpix

            for fib in range(Nfib):
                where_fib = np.where(self.fiber_goal_list == fib)[0]
                if len(where_fib) == 0:
                    where_fib = np.where(self.fiber_goal_list == self.fiber_goal_list[0])[0]
                    combined_spec[fib, :, :], combined_spec_sig[fib, :, :] = combine_science_spectra(
                        data_list[where_fib, fib, :, :], noise_list[where_fib, fib, :, :])
                elif len(where_fib) != 0 and combine_mode == "star":
                    combined_spec[fib, :, :], combined_spec_sig[fib, :, :] = combine_stellar_spectra(
                        data_list[where_fib, fib, :, :], noise_list[where_fib, fib, :, :])
                elif len(where_fib) != 0 and combine_mode == "companion":
                    combined_spec[fib, :, :], combined_spec_sig[fib, :, :] = combine_science_spectra(
                        data_list[where_fib, fib, :, :], noise_list[where_fib, fib, :, :])

            self.data = np.reshape(combined_spec,(Nfib,Norder*Npix)).T
            self.noise = np.reshape(combined_spec_sig,(Nfib,Norder*Npix)).T
            self.bad_pixels = np.reshape(edges2nans(combined_spec/ combined_spec), (Nfib, Norder * Npix)).T

            self.bary_RV = np.mean(baryrv_list)
            self.orders = np.arange(0,9)

        self.bad_pixels[np.where(self.noise==0)] = np.nan
        
        self.valid_data_check()

    def broaden(self, wvs,spectrum, loc=None,mppool=None):
        """
        Broaden a spectrum to the resolution of this data object using the line spread function (LSF) calibration
        available. LSF is assumed to be a 1D gaussian.
        The broadening is technically fiber dependent so you need to specify which fiber calibration to use.

        Args:
            wvs: Wavelength sampling of the spectrum to be broadened.
            spectrum: 1D spectrum to be broadened.
            loc: Fiber index to be used.
            mypool: Multiprocessing pool to parallelize the code. If None (default), non parallelization is applied.
                E.g. mppool = mp.Pool(processes=10) # 10 is the number processes

        Return:
            Broadened spectrum
        """
        fill_value = (self.resolution[:,loc][0],self.resolution[:,loc][-1])
        res_func = interp1d(self.wavelengths[:,loc], self.resolution[:,loc], bounds_error=False, fill_value=fill_value)
        return broaden(wvs, spectrum, res_func(wvs), mppool=mppool)

    def selec_order(self,orders):
        nz,nfib = self.data.shape
        ordersize = int(nz//np.size(self.orders))
        mask = []
        for order in orders:
            try:
                ind = np.where(np.array(self.orders)== order)[0]
                mask.extend(np.arange(ordersize)+ind*ordersize)
            except:
                raise ValueError("requested order {0} does not exist in this KPIC data obect".format(order))

        newself = deepcopy(self)
        newself.data = self.data[mask,:]
        newself.noise =  self.noise[mask,:]
        newself.bad_pixels = self.bad_pixels[mask,:]
        newself.wavelengths = self.wavelengths[mask,:]
        newself.resolution = self.resolution[mask,:]
        newself.orders = orders
        return newself



def edges2nans(spec):
    cp_spec = copy(spec)
    cp_spec[:,:,0:5] = np.nan
    cp_spec[:,:,2000::] = np.nan
    return cp_spec

def combine_stellar_spectra(spectra,errors,weights=None):
    if weights is None:
        _weights = np.ones(spectra.shape[0])/float(spectra.shape[0])
    else:
        _weights = weights
    cp_spectra = copy(spectra)*_weights[:,None,None]

    flux_per_spectra = np.nansum(cp_spectra, axis=2)[:,:,None]

    scaling4badpix = (np.nansum(flux_per_spectra,axis=0)/np.sum(np.isfinite(cp_spectra)*flux_per_spectra,axis=0))
    scaling4badpix[np.where(scaling4badpix>2)] = np.nan
    med_spec = np.nansum(cp_spectra, axis=0)*scaling4badpix
    errors = np.sqrt(np.nansum((errors*_weights[:,None,None])**2, axis=0))*scaling4badpix
    return med_spec,errors

def combine_science_spectra(spectra,errors):
    # out_spec = np.nansum(spectra, axis=0)
    out_spec = np.nanmean(spectra, axis=0)
    # out_spec = np.nanmedian(spectra, axis=0)
    mask = np.ones(spectra.shape)
    mask[np.where(np.isnan(errors))] = 0
    out_errors = np.sqrt(np.nansum(errors**2, axis=0))/np.sum(mask,axis=0)
    return out_spec,out_errors


def get_fib_labels(header):
    fiblabel_list = []
    for fibid in range(20):
        try:
            fiblabel_list.append(header["FIB{0}".format(fibid)].strip())
        except:
            pass
    if len(fiblabel_list) == 0:
        warnings.warn("FIB# keywords not found in file. Assuming [s1,s2,s3,s4] in this order.")
        return np.array(["s1","s2","s3","s4"])
    else:
        return np.array(fiblabel_list)

# def get_science_fibers(header):
#     sf_id_list = []
#     sf_num_list = []
#     for fibid in range(20):
#         try:
#             fiblabel = header["FIB{0}".format(fibid)]
#             if "s" in fiblabel:
#                 sf_id_list.append(fibid)
#                 sf_num_list.append(fiblabel[1:2])
#         except:
#             pass
#     if len(sf_num_list) == 0:
#         warnings.warn("FIB# keywords not found in fluxes file. Assuming [s1,s2,s3,s4] in this order.")
#         return np.array([0,1,2,3])
#     else:
#         return np.array(sf_id_list)[np.array(sf_num_list,dtype=np.float).argsort()]