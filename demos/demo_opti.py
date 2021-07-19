import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import scipy.io as scio
import astropy.io.fits as pyfits
from glob import glob
import multiprocessing as mp
import h5py
from scipy.interpolate import RegularGridInterpolator
import time

from breads.instruments.KPIC import KPIC
from breads.fit import nlog_prob

import emcee
import corner

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    numthreads = 32
    orders = [6]

    filenums_fib,sc_fib = [116,118,120],2 # There is the gas cell here
    spec_filelist=[]
    datadir = "/scr3/kpic/KPIC_Campaign/science/HR7672B/20210704/fluxes/"
    for filenum in filenums_fib:
        spec_filelist.append(os.path.join(datadir,"nspec210704_{0:04d}_fluxes.fits".format(filenum)))

    filenums_fib = [111,112,113,114]
    host_filelist = []
    datadir = "/scr3/kpic/KPIC_Campaign/science/HR7672/20210704/fluxes/"
    for filenum in filenums_fib:
        host_filelist.append(os.path.join(datadir, "nspec210704_{0:04d}_fluxes.fits".format(filenum)))

    filenums_fib = [105,106,107,108,109,110]
    A0_filelist = []
    datadir = "/scr3/kpic/KPIC_Campaign/science/zetAql/20210704/fluxes_test/"
    for filenum in filenums_fib:
        A0_filelist.append(os.path.join(datadir, "nspec210704_{0:04d}_fluxes.fits".format(filenum)))

    planet_btsettl = "/scr3/jruffio/models/BT-Settl/BT-Settl_M-0.0_a+0.0/lte018-4.0-0.0a+0.0.BT-Settl.spec.7"
    trace_filename = "/scr3/kpic/KPIC_Campaign/calibs/20210704/trace/nspec210704_0030_trace.fits"
    wvs_filename = "/scr3/kpic/KPIC_Campaign/calibs/20210704/wave/20210704_HIP81497_psg_wvs.fits"

    wvs_phoenix = "/scr3/jruffio/data/kpic/models/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
    A0_phoenix = "/scr3/jruffio/models/phoenix/fitting/phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte09000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"

    dataobj = KPIC(spec_filelist,trace_filename,wvs_filename,fiber_scan=False)
    A0obj = KPIC(A0_filelist,trace_filename,wvs_filename,fiber_scan=True)
    hostobj = KPIC(host_filelist,trace_filename,wvs_filename,fiber_scan=True)
    A0obj = A0obj.selec_order(orders)
    hostobj = hostobj.selec_order(orders)
    dataobj = dataobj.selec_order(orders)
    nz,nf = dataobj.data.shape

    mypool = mp.Pool(processes=numthreads)

    print(1)
    # Define planet model grid from BTsettl
    minwv,maxwv= np.min(dataobj.wavelengths),np.max(dataobj.wavelengths)
    with h5py.File("/scr3/jruffio/code/OSIRIS/scripts/bt-settl_K-band_1000-3000K_KPIC.hdf5", 'r') as hf:
        grid_specs = np.array(hf.get("spec"))
        grid_temps = np.array(hf.get("temps"))
        grid_loggs = np.array(hf.get("loggs"))
        grid_wvs = np.array(hf.get("wvs"))
    crop_grid = np.where((grid_wvs > minwv - 0.02) * (grid_wvs < maxwv + 0.02))
    grid_wvs = grid_wvs[crop_grid]
    grid_specs = grid_specs[:,:,crop_grid[0]]
    myinterpgrid = RegularGridInterpolator((grid_temps,grid_loggs),grid_specs,method="linear",bounds_error=False,fill_value=np.nan)

    print(2)
    # Define transmission from standard star using Phoenix A0 stellar model
    with pyfits.open(wvs_phoenix) as hdulist:
        phoenix_wvs = hdulist[0].data / 1.e4
    crop_phoenix = np.where((phoenix_wvs > minwv - 0.02) * (phoenix_wvs < maxwv + 0.02))
    phoenix_wvs = phoenix_wvs[crop_phoenix]
    with pyfits.open(A0_phoenix) as hdulist:
        phoenix_A0 = hdulist[0].data[crop_phoenix]
    phoenix_A0_broad = dataobj.broaden(phoenix_wvs,phoenix_A0,loc=sc_fib,mppool=mypool)
    phoenix_A0_func = interp1d(phoenix_wvs, phoenix_A0_broad, bounds_error=False, fill_value=np.nan)

    print(3)
    transmission = A0obj.data[:,sc_fib]/phoenix_A0_func(dataobj.wavelengths[:,sc_fib])

    mypool.close()
    mypool.join()

    # Define star spectrum
    host_spectrum = hostobj.data[:,sc_fib]


    # Definition of the (extra) parameters for fm
    if 1: # model grid
        from breads.fm.hc_atmgrid_hpffm import hc_atmgrid_hpffm
        fm_paras = {"atm_grid":myinterpgrid,"atm_grid_wvs":grid_wvs,"transmission":transmission,"star_spectrum":host_spectrum,
                    "boxw":1,"psfw":0.01,"badpixfraction":0.75,"hpf_mode":"fft","cutoff":10,"loc":sc_fib,
                    "fft_bounds":np.arange(0,dataobj.data.shape[0]+1, 2048)}
        fm_func = hc_atmgrid_hpffm
        nonlin_labels = ["Teff", "logg", "spin", "RV"]
        nonlin_paras_mins = np.array([1000, 3.5, 0, -50])
        nonlin_paras_maxs = np.array([3000, 5.5, 50, 50])
    else: # single model
        planet_f = interp1d(grid_wvs, myinterpgrid([1800,4.0])[0], bounds_error=False, fill_value=np.nan)
        from breads.fm.hc_hpffm import hc_hpffm
        fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":host_spectrum,
                    "boxw":1,"psfw":0.01,"badpixfraction":0.75,"hpf_mode":"fft","cutoff":10,"loc":sc_fib,
                    "fft_bounds":np.arange(0,dataobj.data.shape[0]+1, 2048)}
        fm_func = hc_hpffm
        nonlin_labels = ["RV"]
        nonlin_paras_mins = np.array([-50])
        nonlin_paras_maxs = np.array([50])

    # /!\ Optional but recommended
    # Test the forward model for a fixed value of the non linear parameter.
    # Make sure it does not crash and look the way you want
    if 0:
        nonlin_paras = [1800,4.0,0,0] # x (pix),y (pix), rv (km/s)
        # d is the data vector a the specified location
        # M is the linear component of the model. M is a function of the non linear parameters x,y,rv
        # s is the vector of uncertainties corresponding to d
        d, M, s = fm_func(nonlin_paras,dataobj,**fm_paras)

        validpara = np.where(np.sum(M,axis=0)!=0)
        M = M[:,validpara[0]]
        d = d / s
        M = M / s[:, None]
        from scipy.optimize import lsq_linear
        paras = lsq_linear(M, d).x
        m = np.dot(M,paras)

        plt.subplot(2,1,1)
        plt.plot(d,label="data")
        plt.plot(m,label="model")
        plt.plot(paras[0]*M[:,0],label="planet model")
        plt.plot(m-paras[0]*M[:,0],label="starlight model")
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(M[:,0]/np.max(M[:,0]),label="planet model")
        for k in range(M.shape[-1]-1):
            plt.plot(M[:,k+1]/np.nanmax(M[:,k+1]),label="starlight model {0}".format(k+1))
        plt.legend()
        plt.show()

    simplex_init_steps = (nonlin_paras_maxs - nonlin_paras_mins)/2
    paras0 = (nonlin_paras_maxs+nonlin_paras_mins)/2
    initial_simplex = np.concatenate([paras0[None,:],paras0[None,:] + np.diag(simplex_init_steps)],axis=0)

    def nonlin_lnprior_func(nonlin_paras):
        for p, _min, _max in zip(nonlin_paras, nonlin_paras_mins, nonlin_paras_maxs):
            if p > _max or p < _min:
                return -np.inf
        return 0

    from scipy.optimize import minimize
    res = minimize(nlog_prob, paras0, args=(dataobj, fm_func, fm_paras,nonlin_lnprior_func), method="nelder-mead",
                           options={"xatol": 1e-2, "maxiter": 5e3,"initial_simplex":initial_simplex,"disp":True})
    print("Best fit values:")
    print(nonlin_labels)
    print(res.x)

    plt.show()

