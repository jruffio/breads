import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import scipy.io as scio
import astropy.io.fits as pyfits
from glob import glob

from breads.instruments.KPIC import KPIC
from breads.search_planet import search_planet
from breads.fm.hc_splinefm import hc_splinefm
from breads.fm.iso_hpffm import iso_hpffm
import multiprocessing as mp

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
    planet_btsettl = "/scr3/jruffio/models/BT-Settl/BT-Settl_M-0.0_a+0.0/lte020-3.5-0.0a+0.0.BT-Settl.spec.7"
    trace_filename = "/scr3/kpic/KPIC_Campaign/calibs/20200928/trace/nspec200928_0024_trace.fits"
    wvs_filename = "/scr3/kpic/KPIC_Campaign/calibs/20200928/wave/20200928_HIP_95771_psg_wvs.fits"



    # 2021 RX_J0342.5_1216B
    # # spec_filelist = glob("/scr3/jruffio/data/kpic/20200928_RX_J0342.5_1216_B/*_fluxes.fits")
    # spec_filelist = glob("/scr3/kpic/KPIC_Campaign/science/RX_J0342.5_1216B/20210703/fluxes/*_fluxes.fits")
    # spec_filelist.sort()
    # spec_filelist,sc_fib = spec_filelist[1::2],0 # Selecting fiber 1 because fiber bouncing
    # # spec_filelist,sc_fib = spec_filelist[0::2],1 # Selecting fiber 2 because fiber bouncing
    # host_filelist = glob("/scr3/kpic/KPIC_Campaign/science/RX_J0342.5_1216/20210703/fluxes/*_fluxes.fits")
    # A0_filelist = glob("/scr3/kpic/KPIC_Campaign/science/HIP16322/20210703/fluxes/*_pairsub_fluxes.fits")
    # planet_btsettl = "/scr3/jruffio/models/BT-Settl/BT-Settl_M-0.0_a+0.0/lte018-4.5-0.0a+0.0.BT-Settl.spec.7"
    # trace_filename = "/scr3/kpic/KPIC_Campaign/calibs/20210703/trace/nspec210703_0647_trace.fits"
    # wvs_filename = "/scr3/kpic/KPIC_Campaign/calibs/20210703/wave/20210703_HIP81497_psg_wvs.fits"

    # # 1RXS2351+3127
    # spec_filelist = glob("/scr3/kpic/KPIC_Campaign/science/1RXS2351+3127B/20210703/fluxes/*_fluxes.fits")
    # spec_filelist.sort()
    # fnums,sc_fib = [689,691,693,696,698],1
    # # fnums,sc_fib = [690,692,697,699],0
    # spec_filelist = []
    # for fnum in fnums:
    #     spec_filelist.append("/scr3/kpic/KPIC_Campaign/science/1RXS2351+3127B/20210703/fluxes/nspec210703_{0:04d}_pairsub_fluxes.fits".format(fnum))
    # # spec_filelist,sc_fib = spec_filelist[0:6:2],0 # Selecting fiber 1 because fiber bouncing
    # # spec_filelist,sc_fib = spec_filelist[],1 # Selecting fiber 2 because fiber bouncing
    # host_filelist = glob("/scr3/kpic/KPIC_Campaign/science/1RXS2351+3127/20210703/fluxes/*_fluxes.fits")
    # A0_filelist = glob("/scr3/kpic/KPIC_Campaign/science/HIP5671/20210703/fluxes/*_pairsub_fluxes.fits")
    # planet_btsettl = "/scr3/jruffio/models/BT-Settl/BT-Settl_M-0.0_a+0.0/lte018-4.5-0.0a+0.0.BT-Settl.spec.7"
    # print(A0_filelist)
    # trace_filename = "/scr3/kpic/KPIC_Campaign/calibs/20210703/trace/nspec210703_0647_trace.fits"
    # wvs_filename = "/scr3/kpic/KPIC_Campaign/calibs/20210703/wave/20210703_HIP81497_psg_wvs.fits"



    wvs_phoenix = "/scr3/jruffio/data/kpic/models/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
    A0_phoenix = "/scr3/jruffio/models/phoenix/kap_And_lte11600-4.00-0.5.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"

    A0obj = KPIC(A0_filelist,trace_filename,wvs_filename,fiber_scan=True)
    dataobj = KPIC(spec_filelist,trace_filename,wvs_filename,fiber_scan=False)
    hostobj = KPIC(host_filelist,trace_filename,wvs_filename,fiber_scan=True)
    orders = [6]
    A0obj = A0obj.selec_order(orders)
    hostobj = hostobj.selec_order(orders)
    dataobj = dataobj.selec_order(orders)
    # plt.figure(1)
    # plt.subplot(4,1,1)
    # plt.plot(dataobj.wavelengths[:,0],dataobj.data[:,0])
    # plt.subplot(4,1,2)
    # plt.plot(dataobj.wavelengths[:,1],dataobj.data[:,1])
    # plt.subplot(4,1,3)
    # plt.plot(dataobj.wavelengths[:,2],dataobj.data[:,2])
    # plt.subplot(4,1,4)
    # plt.plot(dataobj.wavelengths[:,3],dataobj.data[:,3])
    # plt.figure(2)
    # plt.subplot(4,1,1)
    # plt.plot(hostobj.wavelengths[:,0],hostobj.data[:,0])
    # plt.subplot(4,1,2)
    # plt.plot(hostobj.wavelengths[:,1],hostobj.data[:,1])
    # plt.subplot(4,1,3)
    # plt.plot(hostobj.wavelengths[:,2],hostobj.data[:,2])
    # plt.subplot(4,1,4)
    # plt.plot(hostobj.wavelengths[:,3],hostobj.data[:,3])
    # plt.figure(3)
    # plt.subplot(4,1,1)
    # plt.plot(A0obj.wavelengths[:,0],A0obj.data[:,0])
    # plt.subplot(4,1,2)
    # plt.plot(A0obj.wavelengths[:,1],A0obj.data[:,1])
    # plt.subplot(4,1,3)
    # plt.plot(A0obj.wavelengths[:,2],A0obj.data[:,2])
    # plt.subplot(4,1,4)
    # plt.plot(A0obj.wavelengths[:,3],A0obj.data[:,3])
    # plt.show()
    nz,nf = dataobj.data.shape
    # print(np.where(np.isnan(dataobj.bad_pixels)))

    mypool = mp.Pool(processes=32)

    # Define planet model from BTsettl
    arr = np.genfromtxt(planet_btsettl, delimiter=[12, 14], dtype=np.float,
                        converters={1: lambda x: float(x.decode("utf-8").replace('D', 'e'))})
    model_wvs = arr[:, 0] / 1e4
    model_spec = 10 ** (arr[:, 1] - 8)
    minwv,maxwv= np.min(dataobj.wavelengths),np.max(dataobj.wavelengths)
    crop_btsettl = np.where((model_wvs > minwv - 0.02) * (model_wvs < maxwv + 0.02))
    model_wvs = model_wvs[crop_btsettl]
    model_spec = model_spec[crop_btsettl]
    model_broadspec = dataobj.broaden(model_wvs,model_spec,loc=1,mppool=mypool)
    planet_f = interp1d(model_wvs, model_broadspec, bounds_error=False, fill_value=np.nan)

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

    # plt.plot(transmission)
    # plt.show()

    mypool.close()
    mypool.join()

    # Define star spectrum
    host_spectrum = hostobj.data[:,sc_fib]

    # plt.plot(host_spectrum)
    # plt.show()

    # Definition of the (extra) parameters for splinefm()
    N_nodes_per_order = 5
    nodes = []
    nz,nfib = dataobj.data.shape
    ordersize = int(nz//np.size(dataobj.orders)) # probably equal to 2048...
    for order_id in range(np.size(dataobj.orders)):
        minwvord = dataobj.wavelengths[order_id*ordersize,sc_fib]
        maxwvord = dataobj.wavelengths[(order_id+1)*ordersize-1,sc_fib]
        nodes.append(np.linspace(minwvord,maxwvord,N_nodes_per_order,endpoint=True))
    fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":host_spectrum,
                "boxw":1,"nodes":nodes,"psfw":1.2,"badpixfraction":0.75}
    fm_func = hc_splinefm
    # fm_paras = {"planet_f":planet_f,"transmission":transmission,"boxw":1,"res_hpf":100,"psfw":1.2,"badpixfraction":0.75}
    # fm_func = iso_hpffm

    if 0: # Example code to test the forward model
        nonlin_paras = [20,sc_fib] # x (pix),y (pix), rv (km/s)
        # d is the data vector a the specified location
        # M is the linear component of the model. M is a function of the non linear parameters x,y,rv
        # s is the vector of uncertainties corresponding to d
        # fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":host_spectrum,
        #             "boxw":1,"nodes":nodes,"psfw":1.2,"badpixfraction":0.75}
        # d, M, s = hc_splinefm(nonlin_paras,dataobj,**fm_paras)
        d, M, s = fm_func(nonlin_paras,dataobj,**fm_paras)
        # print(M.shape)
        # exit()

        plt.subplot(2,1,1)
        plt.plot(d,label="data")
        plt.subplot(2,1,2)
        plt.plot(M[:,0]/np.nanmax(M[:,0]),label="planet model")
        for k in range(M.shape[-1]-1):
            plt.plot(M[:,k+1]/np.nanmax(M[:,k+1]),label="starlight model {0}".format(k+1))
        plt.legend()
        plt.show()

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
    # exit()
    # fit rv
    rvs = np.linspace(-400,400,401)
    fibs = np.array([sc_fib])
    out = search_planet([rvs,fibs],dataobj,fm_func,fm_paras,numthreads=32)
    N_linpara = (out.shape[-1]-2)//2
    print(out.shape)

    plt.figure(1)
    plt.plot(rvs,out[:,0,3]/out[:,0,3+N_linpara])
    # plt.plot(out[:,0,0]-out[:,0,1])
    plt.ylabel("SNR")
    plt.xlabel("RV (km/s)")
    # plt.clim([0,50])
    # plt.plot(out[:,0,0,2])
    plt.show()
