import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import scipy.io as scio
import astropy.io.fits as pyfits

from breads.instruments.OSIRIS import OSIRIS
from breads.grid_search import grid_search
from breads.fm.hc_splinefm import hc_splinefm
from breads.fm.iso_hpffm import iso_hpffm
from breads.fm.hc_hpffm import hc_hpffm

import warnings
warnings.simplefilter('ignore')

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    numthreads = 16

    filename = "../../public_osiris_data/kap_And/20161106/science/s161106_a020002_Kbb_020.fits"
    dataobj = OSIRIS(filename)
    #dataobj.refpos = (10,-15)
    nz,ny,nx = dataobj.data.shape
    dataobj.noise = np.tile(np.nanstd(dataobj.data,axis=0),(nz,1,1))#np.ones((nz,ny,nx))
    # plt.imshow(dataobj.noise[1000,:,:])
    # plt.show()

    if 1:
        # Load the planet model, the transmission spectrum, and the star spectrum to be used in the forward model.
        # this is temporary and these files are specific to s161106_a020002_Kbb_020.fits
        filename = os.path.join("../../public_osiris_data/tmp/planet_spectrum.fits")
        with pyfits.open(filename) as hdulist:
            pl_wvs = hdulist[0].data
            pl_spec = hdulist[1].data
        planet_f = interp1d(pl_wvs, pl_spec, bounds_error=False, fill_value=np.nan)
        filename = os.path.join("../../public_osiris_data/tmp/transmission.fits")
        with pyfits.open(filename) as hdulist:
            transmission = hdulist[1].data
        filename = os.path.join("../../public_osiris_data/tmp/star_spectrum.fits")
        with pyfits.open(filename) as hdulist:
            star_spectrum = hdulist[1].data

    import multiprocessing as mp
    mypool = mp.Pool(processes=numthreads)
    dataobj.remove_bad_pixels(med_spec=star_spectrum,mypool=mypool)
    mypool.close()
    mypool.join()

    # Definition of the (extra) parameters for splinefm()
    fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":star_spectrum,
                "boxw":3,"nodes":20,"psfw":1.2,"badpixfraction":0.75}
    fm_func = hc_splinefm
    # fm_paras = {"planet_f":planet_f,"transmission":transmission,"boxw":1,"res_hpf":100,"psfw":1.2,"badpixfraction":0.75}
    # fm_func = iso_hpffm
    # fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":star_spectrum,
    #             "boxw":3,"psfw":1.2,"badpixfraction":0.75,"hpf_mode":"fft","cutoff":40}
    # fm_func = hc_hpffm

    if 0: # Example code to test the forward model
        nonlin_paras = [-15,37,10] # rv (km/s), y (pix), x (pix)
        # nonlin_paras = [-15,33+0,4-0] # rv (km/s), y (pix), x (pix)
        # nonlin_paras = [-15,46,14] # rv (km/s), y (pix), x (pix)
        # d is the data vector a the specified location
        # M is the linear component of the model. M is a function of the non linear parameters x,y,rv
        # s is the vector of uncertainties corresponding to d
        d, M, s = fm_func(nonlin_paras,dataobj,**fm_paras)

        validpara = np.where(np.sum(M,axis=0)!=0)
        M = M[:,validpara[0]]
        d = d / s
        M = M / s[:, None]
        from scipy.optimize import lsq_linear
        #print('how big:', np.size(M), np.size(d))
        #print('how many nans:', np.sum(np.isnan(M)), np.sum(np.isnan(d)))
        paras = lsq_linear(M, d).x
        m = np.dot(M,paras)
        paras_H0 = lsq_linear(M[:,1::], d).x
        m_H0 = np.dot(M[:,1::],paras_H0)

        plt.subplot(3,1,1)
        plt.plot(d,label="data")
        plt.plot(m,label="model")
        plt.plot(m_H0,label="model H0")
        plt.plot(paras[0]*M[:,0],label="planet model")
        plt.plot(m-paras[0]*M[:,0],label="starlight model")
        plt.legend()
        plt.subplot(3,1,2)
        plt.plot(d-m,label="residuals")
        plt.plot(d-m_H0,label="residuals H0")
        plt.legend()
        plt.subplot(3,1,3)
        plt.plot(M[:,0]/np.max(M[:,0]),label="planet model")
        for k in range(M.shape[-1]-1):
            plt.plot(M[:,k+1]/np.nanmax(M[:,k+1]),label="starlight model {0}".format(k+1))
        plt.legend()
        plt.show()

    # fit rv
    # rvs = np.linspace(-2000,2000,201)
    # ys = np.array([37])
    # xs = np.array([10])
    #seach for planets demo
    rvs = np.array([-15])
    ys = np.arange(ny)
    xs = np.arange(nx)
    log_prob,log_prob_H0,rchi2,linparas,linparas_err = grid_search([rvs,ys,xs],dataobj,fm_func,fm_paras,numthreads=numthreads, computeH0=True)
    N_linpara = linparas.shape[-1]

    # print('how big:',np.size(log_prob), np.size(log_prob_H0))
    # print('how many nans:',np.sum(np.isnan(log_prob)), np.sum(np.isnan(log_prob_H0)))
    k,l,m = np.unravel_index(np.nanargmax(log_prob-log_prob_H0),log_prob.shape)
    print("best fit parameters: rv={0},y={1},x={2}".format(rvs[k],ys[l],xs[m]) )
    print(np.nanmax(log_prob-log_prob_H0))
    best_log_prob,best_log_prob_H0,_,_,_ = grid_search([[rvs[k]], [ys[l]], [xs[m]]], dataobj, fm_func, fm_paras, numthreads=None)
    print(best_log_prob-best_log_prob_H0)

    plt.figure(1)
    plt.subplot(1,2,1)
    snr_map = linparas[k,:,:,0]/linparas_err[k,:,:,0]
    plt.imshow(snr_map,origin="lower")
    plt.clim([np.nanmin(snr_map),np.nanmax(snr_map)])
    cbar = plt.colorbar()
    cbar.set_label("SNR")
    # plt.plot(out[:,0,0,2])
    log_prob_diff = log_prob[k,:,:]-log_prob_H0[k,:,:]
    plt.subplot(1,2,2)
    plt.imshow(log_prob_diff,origin="lower")
    plt.clim([0, np.nanmax(np.where(np.isfinite(log_prob_diff), log_prob_diff, np.nan))])
    cbar = plt.colorbar()
    cbar.set_label("log_prob_H1 - log_prob_H0")
    plt.show()
