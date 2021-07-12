import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import astropy.io.fits as pyfits
from glob import glob
import h5py
from scipy.interpolate import RegularGridInterpolator

from breads.instruments.KPIC import KPIC
import multiprocessing as mp

import emcee
import corner

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    # 2020 RX_J0342.5_1216B
    spec_filelist = glob("/scr3/kpic/KPIC_Campaign/science/RX_J0342.5_1216B/20200928/fluxes/*_fluxes.fits")
    spec_filelist.sort()
    spec_filelist,sc_fib = spec_filelist,1 # Selecting fiber 2 because fiber bouncing
    host_filelist = glob("/scr3/kpic/KPIC_Campaign/science/RX_J0342.5_1216/20200928/fluxes/*_fluxes.fits")
    A0_filelist = glob("/scr3/kpic/KPIC_Campaign/science/HIP16322/20200928/fluxes/*_fluxes.fits")
    planet_btsettl = "/scr3/jruffio/models/BT-Settl/BT-Settl_M-0.0_a+0.0/lte018-5.0-0.0a+0.0.BT-Settl.spec.7"
    trace_filename = "/scr3/kpic/KPIC_Campaign/calibs/20200928/trace/nspec200928_0024_trace.fits"
    wvs_filename = "/scr3/kpic/KPIC_Campaign/calibs/20200928/wave/20200928_HIP_95771_psg_wvs.fits"


    wvs_phoenix = "/scr3/jruffio/data/kpic/models/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
    A0_phoenix = "/scr3/jruffio/models/phoenix/kap_And_lte11600-4.00-0.5.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"


    dataobj = KPIC(spec_filelist,trace_filename,wvs_filename,fiber_scan=False)
    A0obj = KPIC(A0_filelist,trace_filename,wvs_filename,fiber_scan=True)
    hostobj = KPIC(host_filelist,trace_filename,wvs_filename,fiber_scan=True)
    orders = [6]
    A0obj = A0obj.selec_order(orders)
    hostobj = hostobj.selec_order(orders)
    dataobj = dataobj.selec_order(orders)
    nz,nf = dataobj.data.shape

    mypool = mp.Pool(processes=32)

    # Define planet model from BTsettl
    minwv,maxwv= np.min(dataobj.wavelengths),np.max(dataobj.wavelengths)
    with h5py.File("/scr3/jruffio/code/OSIRIS/scripts/bt-settl_H-band_1600-2600K_KPIC.hdf5", 'r') as hf:
        grid_specs = np.array(hf.get("spec"))
        grid_temps = np.array(hf.get("temps"))
        grid_loggs = np.array(hf.get("loggs"))
        grid_wvs = np.array(hf.get("wvs"))
    crop_grid = np.where((grid_wvs > minwv - 0.02) * (grid_wvs < maxwv + 0.02))
    grid_wvs = grid_wvs[crop_grid]
    grid_specs = grid_specs[:,:,crop_grid[0]]
    myinterpgrid = RegularGridInterpolator((grid_temps,grid_loggs),grid_specs,method="linear",bounds_error=False,fill_value=np.nan)

    grid_Rs = dataobj.wavelengths[1::,1] / (dataobj.wavelengths[1::,1] - dataobj.wavelengths[0:-1,1])

    # Define transmission from standard star using Phoenix A0 stellar model
    with pyfits.open(wvs_phoenix) as hdulist:
        phoenix_wvs = hdulist[0].data / 1.e4
    crop_phoenix = np.where((phoenix_wvs > minwv - 0.02) * (phoenix_wvs < maxwv + 0.02))
    phoenix_wvs = phoenix_wvs[crop_phoenix]
    with pyfits.open(A0_phoenix) as hdulist:
        phoenix_A0 = hdulist[0].data[crop_phoenix]
    phoenix_A0_broad = dataobj.broaden(phoenix_wvs,phoenix_A0,loc=1,mppool=mypool)
    phoenix_A0_func = interp1d(phoenix_wvs, phoenix_A0_broad, bounds_error=False, fill_value=np.nan)

    transmission = A0obj.data[:,sc_fib]/phoenix_A0_func(dataobj.wavelengths[:,sc_fib])

    mypool.close()
    mypool.join()

    # Define star spectrum
    host_spectrum = hostobj.data[:,sc_fib]

    # Definition of the (extra) parameters for fm
    from breads.fm.hc_atmgrid_hpffm import hc_atmgrid_hpffm
    fm_paras = {"atm_grid":myinterpgrid,"atm_grid_wvs":grid_wvs,"transmission":transmission,"star_spectrum":host_spectrum,
                "loc":1,"boxw":1,"psfw":0.01,"badpixfraction":0.75,"hpf_mode":"fft","res_hpf":100,"cutoff":5}
    fm_func = hc_atmgrid_hpffm

    # if 0: # Example code to test the forward model
    #     nonlin_paras = [1800,4.0,0,20,sc_fib] # x (pix),y (pix), rv (km/s)
    #     # d is the data vector a the specified location
    #     # M is the linear component of the model. M is a function of the non linear parameters x,y,rv
    #     # s is the vector of uncertainties corresponding to d
    #     # fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":host_spectrum,
    #     #             "boxw":1,"nodes":nodes,"psfw":1.2,"badpixfraction":0.75}
    #     # d, M, s = hc_splinefm(nonlin_paras,dataobj,**fm_paras)
    #     d, M, s = fm_func(nonlin_paras,dataobj,**fm_paras)
    #     # print(M.shape)
    #     # exit()
    #
    #     validpara = np.where(np.sum(M,axis=0)!=0)
    #     M = M[:,validpara[0]]
    #     d = d / s
    #     M = M / s[:, None]
    #     from scipy.optimize import lsq_linear
    #     paras = lsq_linear(M, d).x
    #     m = np.dot(M,paras)
    #
    #     plt.subplot(2,1,1)
    #     plt.plot(d,label="data")
    #     plt.plot(m,label="model")
    #     plt.plot(paras[0]*M[:,0],label="planet model")
    #     plt.plot(m-paras[0]*M[:,0],label="starlight model")
    #     plt.legend()
    #     plt.subplot(2,1,2)
    #     plt.plot(M[:,0]/np.max(M[:,0]),label="planet model")
    #     for k in range(M.shape[-1]-1):
    #         plt.plot(M[:,k+1]/np.nanmax(M[:,k+1]),label="starlight model {0}".format(k+1))
    #     plt.legend()
    #     plt.show()

    if 1: # emcee
        ndim, nwalkers = 4, 100
        nonlin_paras_mins = np.array([1600, 3.5, 0, 0])
        nonlin_paras_maxs = np.array([2600, 5.5, 50, 50])
        p0 = np.random.rand(nwalkers, ndim) * (nonlin_paras_maxs-nonlin_paras_mins)[None,:] + nonlin_paras_mins[None,:]
        print(np.nanmedian(p0, axis=0))

        from breads.fit import log_prob
        os.environ["OMP_NUM_THREADS"] = "1"
        mypool = mp.Pool(processes=16)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[dataobj, fm_func, fm_paras],pool=mypool)
        # print(log_prob(p0[0],dataobj, fm_func, fm_paras))
        # exit()
        state = sampler.run_mcmc(p0, 100, progress=True)
        print("burnout over")
        sampler.reset()
        sampler.run_mcmc(state, 100, progress=True)
        samples = sampler.get_chain(flat=True)
        samples_gc = samples[:, 0]
        mypool.close()
        mypool.join()

        figure = corner.corner(samples, labels=["Teff", "logg", "spin", "RV","sc_fib"])
        plt.show()

