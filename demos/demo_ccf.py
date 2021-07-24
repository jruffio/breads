import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import scipy.io as scio
import astropy.io.fits as pyfits
from glob import glob
import multiprocessing as mp

from breads.instruments.KPIC import KPIC
from breads.search_planet import search_planet

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

    planet_btsettl = "/scr3/jruffio/models/BT-Settl/BT-Settl_M-0.0_a+0.0/lte018-5.0-0.0a+0.0.BT-Settl.spec.7"
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

    # Define planet model from BTsettl
    arr = np.genfromtxt(planet_btsettl, delimiter=[12, 14], dtype=np.float,
                        converters={1: lambda x: float(x.decode("utf-8").replace('D', 'e'))})
    model_wvs = arr[:, 0] / 1e4
    model_spec = 10 ** (arr[:, 1] - 8)
    minwv,maxwv= np.min(dataobj.wavelengths),np.max(dataobj.wavelengths)
    crop_btsettl = np.where((model_wvs > minwv - 0.02) * (model_wvs < maxwv + 0.02))
    model_wvs = model_wvs[crop_btsettl]
    model_spec = model_spec[crop_btsettl]
    model_broadspec = dataobj.broaden(model_wvs,model_spec,loc=sc_fib,mppool=mypool)
    planet_f = interp1d(model_wvs, model_broadspec, bounds_error=False, fill_value=np.nan)

    # Define transmission from standard star using Phoenix A0 stellar model
    with pyfits.open(wvs_phoenix) as hdulist:
        phoenix_wvs = hdulist[0].data / 1.e4
    crop_phoenix = np.where((phoenix_wvs > minwv - 0.02) * (phoenix_wvs < maxwv + 0.02))
    phoenix_wvs = phoenix_wvs[crop_phoenix]
    with pyfits.open(A0_phoenix) as hdulist:
        phoenix_A0 = hdulist[0].data[crop_phoenix]
    phoenix_A0_broad = dataobj.broaden(phoenix_wvs,phoenix_A0,loc=sc_fib,mppool=mypool)
    phoenix_A0_func = interp1d(phoenix_wvs, phoenix_A0_broad, bounds_error=False, fill_value=np.nan)

    transmission = A0obj.data[:,sc_fib]/phoenix_A0_func(A0obj.wavelengths[:,sc_fib])

    mypool.close()
    mypool.join()

    # Define star spectrum
    host_spectrum = hostobj.data[:,sc_fib]

    # plt.plot(host_spectrum)
    # plt.show()

    # Definition of the forward model
    if 1: # Using a high pass filter
        from breads.fm.hc_hpffm import hc_hpffm
        fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":host_spectrum,
                    "boxw":1,"psfw":0.01,"badpixfraction":0.75,"hpf_mode":"fft","cutoff":10,"loc":sc_fib,
                    "fft_bounds":np.arange(0,dataobj.data.shape[0]+1, 2048)}
        fm_func = hc_hpffm
    else: # forward modelling the continuum with a spline
        from breads.fm.hc_splinefm import hc_splinefm
        N_nodes_per_order = 5
        nodes = []
        nz,nfib = dataobj.data.shape
        ordersize = int(nz//np.size(dataobj.orders)) # probably equal to 2048...
        for order_id in range(np.size(dataobj.orders)):
            minwvord = dataobj.wavelengths[order_id*ordersize,sc_fib]
            maxwvord = dataobj.wavelengths[(order_id+1)*ordersize-1,sc_fib]
            nodes.append(np.linspace(minwvord,maxwvord,N_nodes_per_order,endpoint=True))
        fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":host_spectrum,
                    "boxw":1,"psfw":0.01,"badpixfraction":0.75,"nodes":nodes,"loc":sc_fib}
        fm_func = hc_splinefm

    # /!\ Optional but recommended
    # Test the forward model for a fixed value of the non linear parameter.
    # Make sure it does not crash and look the way you want
    if 0:
        nonlin_paras = [-2] # rv (km/s)
        # d is the data vector a the specified location
        # M is the linear component of the model. M is a function of the non linear parameters x,y,rv
        # s is the vector of uncertainties corresponding to d
        d, M, s = fm_func(nonlin_paras,dataobj,**fm_paras)

        # plt.plot(s)
        # plt.show()

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

    # fit rv
    rvs = np.linspace(-400,400,401)
    out = search_planet([rvs],dataobj,fm_func,fm_paras,numthreads=numthreads)
    N_linpara = (out.shape[-1]-2)//2
    print(out.shape)

    plt.figure(1,figsize=(12,4))
    plt.subplot(1,3,1)
    snr = out[:,3]/out[:,3+N_linpara]
    plt.plot(rvs,snr)
    plt.plot(rvs,snr-np.nanmedian(snr),label="spline CCF")
    plt.ylabel("SNR")
    plt.xlabel("RV (km/s)")

    plt.subplot(1,3,2)
    plt.plot(rvs,out[:,0]-out[:,1])
    plt.ylabel("ln(Bayes factor)")
    plt.xlabel("RV (km/s)")

    plt.subplot(1,3,3)
    plt.plot(rvs,np.exp(out[:,0]-np.max(out[:,0])))
    plt.ylabel("RV posterior")
    plt.xlabel("RV (km/s)")
    plt.show()
