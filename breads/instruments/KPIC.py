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

class KPIC(Instrument):
    def __init__(self, spec=None, trace=None, wvs=None, err=None, badpix=None,baryrv=None,orders=None,fiber_scan=False):
        super().__init__('KPIC')
        if spec is None:
            warning_text = "No data file provided. " + \
            "Please manually add data using OSIRIS.manual_data_entry() or add data using OSIRIS.read_data_file()"
            warn(warning_text)
        else:
            self.read_data_file(spec, trace, wvs,err,badpix,baryrv,orders,fiber_scan)

    def read_data_file(self, spec, trace, wvs, err=None, badpix=None,baryrv=None,orders=None,fiber_scan=False):
        """
        Read OSIRIS spectral cube, also checks validity at the end
        todo: replace fiber_bounce with a header fiber keyword once it exists
        """
        if type(wvs) is np.ndarray:
            self.wavelengths = wvs
        else:
            with pyfits.open(wvs) as hdulist:
                sfib_list = get_science_fibers(hdulist[0].header)
                arr = hdulist[0].data[sfib_list,:,:]
                nfib,nord,npix = arr.shape
                self.wavelengths = np.reshape(arr,(nfib,nord*npix))
        dwvs = self.wavelengths[:, 1::] - self.wavelengths[:, 0:-1]
        dwvs = np.concatenate([dwvs, dwvs[:, -1][:, None]], axis=1)

        self.wavelengths = self.wavelengths.T

        if type(trace) is np.ndarray:
            line_width = trace
        else:
            with pyfits.open(trace) as hdulist:
                sfib_list = get_science_fibers(hdulist[0].header)
                arr = hdulist[0].data[sfib_list,:,:]
                nfib,nord,npix = arr.shape
                line_width = np.reshape(arr,(nfib,nord*npix))
        if self.wavelengths.shape[1] == 4 and line_width.shape[0] ==5:
            # hard coded for epoch 20210704 when the trace calibration included the gas cell, but not the wavcal
            self.wavelengths = np.concatenate([np.zeros((self.wavelengths.shape[0],1)),self.wavelengths],axis=1)
            dwvs = np.concatenate([np.zeros((1,self.wavelengths.shape[0])),dwvs],axis=0)
        line_FWHM_wvunit = line_width* dwvs*2*np.sqrt(2*np.log(2))
        self.resolution = self.wavelengths/line_FWHM_wvunit.T

        if type(spec) is np.ndarray:
            self.data = spec
            self.noise = err
            self.bad_pixels = badpix
            self.bary_RV = baryrv
            self.orders = orders
        else:
            if type(spec) is not list:
                filelist = [spec]
            else:
                filelist = spec

            baryrv_list = []
            data_list = []
            noise_list = []
            for filename in filelist:
                with pyfits.open(filename) as hdulist:
                    sfib_list = get_science_fibers(hdulist[0].header)
                    Nfib,Norder,Npix = hdulist[0].data[sfib_list,:,:].shape
                    data_list.append(hdulist[0].data[sfib_list,:,:])
                    header = hdulist[0].header
                    noise_list.append(hdulist[1].data[sfib_list,:,:])
                    baryrv_list.append(float(header["BARYRV"]))

            data_list = np.array(data_list)
            noise_list = np.array(noise_list)

            combined_spec = np.zeros((Nfib,Norder,Npix))
            combined_spec_sig = np.zeros((Nfib,Norder,Npix))
            if fiber_scan:
                if type(fiber_scan) is np.ndarray:
                    fiber_list = fiber_scan
                else:
                    fiber_list = np.argmax(np.nansum(data_list, axis=(2,3)),axis=1)

                for fib in range(Nfib):
                    if fib != 1:
                        continue
                    where_fib = np.where(fiber_list == fib)[0]
                    if len(where_fib) == 0:
                        continue
                    # tmp_spec = data_list[where_fib,fib,:,:]
                    # med_spec = np.nanmean(tmp_spec/np.nanmean(tmp_spec,axis=2)[:,:,None],axis=0)
                    for order in range(Norder):
                        badpix = findbadpix(data_list[where_fib,fib,order,:].T[:,:,None], noisecube=noise_list[where_fib,fib,order,:].T[:,:,None], badpixcube=None,
                                            chunks=5, mypool=None, med_spec=None,nan_mask_boxsize=0)[0][:,:,0].T #med_spec[order,:]
                        data_list[where_fib,fib,order,:] *= badpix

                for fib in range(Nfib):
                    where_fib = np.where(fiber_list == fib)[0]
                    if len(where_fib) != 0:
                        combined_spec[fib, :, :], combined_spec_sig[fib, :, :] = combine_stellar_spectra(
                            data_list[where_fib, fib, :, :], noise_list[where_fib, fib, :, :])
            else:
                for fib in range(Nfib):
                    for order in range(Norder):
                        badpix = findbadpix(data_list[:,fib,order,:].T[:,:,None], noisecube=noise_list[:,fib,order,:].T[:,:,None], badpixcube=None,
                                            chunks=5, mypool=None, med_spec=None,nan_mask_boxsize=0)[0][:,:,0].T
                        data_list[:,fib,order,:] *= badpix

                for fib in range(Nfib):
                    combined_spec[fib, :, :], combined_spec_sig[fib, :, :] = combine_science_spectra(
                        data_list[:, fib, :, :], noise_list[:, fib, :, :])

            self.data = np.reshape(combined_spec,(nfib,Norder*Npix)).T
            self.noise = np.reshape(combined_spec_sig,(nfib,Norder*Npix)).T
            self.bad_pixels = np.reshape(edges2nans(combined_spec/ combined_spec), (nfib, Norder * Npix)).T

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
        # import matplotlib.pyplot as plt
        # plt.plot(mask)
        # plt.show()
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
    out_spec = np.nanmean(spectra, axis=0)
    mask = np.ones(spectra.shape)
    mask[np.where(np.isnan(errors))] = 0
    out_errors = np.sqrt(np.nansum(errors**2, axis=0))/np.sum(mask,axis=0)
    return out_spec,out_errors

def get_science_fibers(header):
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
    return np.array(sf_id_list)[np.array(sf_num_list,dtype=np.float).argsort()]
    # return np.array([0,1,2,3])