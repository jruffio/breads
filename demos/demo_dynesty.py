import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import astropy.io.fits as pyfits
from glob import glob
import multiprocessing as mp
import h5py
from scipy.interpolate import RegularGridInterpolator
from dynesty import NestedSampler
import dynesty
import pickle
import time

from breads.instruments.KPIC import KPIC
from breads.fit import log_prob

import emcee
import corner

def run_ns(loglike, ptform, logl_args, ptform_args, ndim, prefix,nlive=250,mppool = None, nproc=20):
    """
    Author: Jerry Xuan 07/16/2021
    Modified by JB
    """

    # initialize our nested sampler
    sampler = NestedSampler(loglike, ptform, ndim, logl_args=logl_args,ptform_args=ptform_args,
                            nlive=nlive, pool=mppool, queue_size=nproc,
                        use_pool={'prior_transform': False})

    # check that logfile doesn't already exist
    live_file = prefix + "_live_results.pkl"
    if os.path.isfile(live_file):
        print(live_file + " already exists, deleting first")
        os.remove(live_file)
    # run as generator to save progress intermittently
    start = time.time()
    for it, this_array in enumerate(sampler.sample(dlogz=0.5)): #
        # more useful to save the results dict from dynesty
        this_results = sampler.results
        with open(live_file, "wb") as f:
            pickle.dump(this_results, f)

        if it % 100 == 0:
            end = time.time()
            print("Time",end-start)
            print('Iter ' + str(this_results['niter']) )
            print('last logl, samples')
            print(this_results['logl'][-1])
            print(this_results['samples'][-1])
            # change in logz from last 2 iters
            try:
                this_dlogz = this_results['logz'][-1] - this_results['logz'][-2]
                print('last dlogz: ' + str(this_dlogz))
            except:
                print('first iter, no dlogz')
            start = time.time()

    f.close()
    res = sampler.results

    normalized_weights = np.exp(res.logwt - np.max(res.logwt))
    normalized_weights /= np.sum(normalized_weights)
    res.weights = normalized_weights
    equal_samples = dynesty.utils.resample_equal(res.samples, res.weights)

    return res, equal_samples


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


    # Define transmission from standard star using Phoenix A0 stellar model
    with pyfits.open(wvs_phoenix) as hdulist:
        phoenix_wvs = hdulist[0].data / 1.e4
    crop_phoenix = np.where((phoenix_wvs > minwv - 0.02) * (phoenix_wvs < maxwv + 0.02))
    phoenix_wvs = phoenix_wvs[crop_phoenix]
    with pyfits.open(A0_phoenix) as hdulist:
        phoenix_A0 = hdulist[0].data[crop_phoenix]
    phoenix_A0_broad = dataobj.broaden(phoenix_wvs,phoenix_A0,loc=sc_fib,mppool=mypool)
    phoenix_A0_func = interp1d(phoenix_wvs, phoenix_A0_broad, bounds_error=False, fill_value=np.nan)

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

        def ptform(u):
            x = np.array(u)
            nonlin_paras_mins = np.array([1000, 3.5, 0, -50])
            nonlin_paras_maxs = np.array([3000, 5.5, 50, 50])
            for i, _min, _max in zip(range(np.size(x)), nonlin_paras_mins, nonlin_paras_maxs):
                x[i] = x[i] * (_max - _min) + _min
            return x
    else: # single model
        planet_f = interp1d(grid_wvs, myinterpgrid([1800,4.0])[0], bounds_error=False, fill_value=np.nan)
        from breads.fm.hc_hpffm import hc_hpffm
        fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":host_spectrum,
                    "boxw":1,"psfw":0.01,"badpixfraction":0.75,"hpf_mode":"fft","cutoff":10,"loc":sc_fib,
                    "fft_bounds":np.arange(0,dataobj.data.shape[0]+1, 2048)}
        fm_func = hc_hpffm
        nonlin_labels = ["RV"]

        def ptform(u):
            x = np.array(u)
            nonlin_paras_mins = np.array([-50])
            nonlin_paras_maxs = np.array([50])
            for i, _min, _max in zip(range(np.size(x)), nonlin_paras_mins, nonlin_paras_maxs):
                x[i] = x[i] * (_max - _min) + _min
            return x

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


    nwalkers = 512
    nsteps = 100
    nlive = 50
    numthreads = 4
    output_base = "/scr3/jruffio/code/OSIRIS/scripts/dynesty_test"
    if 1: # Run dynesty, a temporary file will be saved every 100 iterations, which can be plotted with the else statement
        start = time.time()
        mppool = mp.Pool(processes=numthreads)
        res, equal_samples = run_ns(log_prob, ptform, logl_args=[dataobj, fm_func, fm_paras],
                                                           ptform_args=None, ndim=len(nonlin_labels),prefix=output_base, nlive=nlive,
                                                           mppool=mppool, nproc=numthreads)
        mypool.close()
        mypool.join()

        end = time.time()
        print("final time {0}s".format(end - start))  # seq 190.3493688106537s or 226s /// 30 or 50s (numthreads = 10)
    else:
        import pickle

        file = pickle.load(open(output_base + '_live_results.pkl', 'rb'))
        print(file.keys())
        max_samples = file["samples"][np.argmax(file["logl"])]
        normalized_weights = np.exp(file["logwt"] - np.max(file["logwt"]))
        normalized_weights /= np.sum(normalized_weights)
        equal_samples = dynesty.utils.resample_equal(file["samples"], normalized_weights)
        print(np.max(file["logl"]), max_samples)
        print(equal_samples.shape)

        # Look at convergence
        from dynesty import plotting as dyplot
        fig, axes = dyplot.traceplot(file, labels=nonlin_labels, )

    # corner plot
    figure = corner.corner(equal_samples, labels=nonlin_labels)
    plt.show()