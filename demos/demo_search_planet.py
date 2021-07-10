import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import scipy.io as scio
import astropy.io.fits as pyfits

from breads.instruments.OSIRIS import OSIRIS
from breads.search_planet import search_planet
from breads.fm.hc_splinefm import hc_splinefm
from breads.fm.iso_hpffm import iso_hpffm

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass


    filename = "../../public_osiris_data/kap_And/20161106/science/s161106_a020002_Kbb_020.fits"
    dataobj = OSIRIS(filename)
    nz,ny,nx = dataobj.data.shape
    dataobj.noise = np.ones((nz,ny,nx))
    # plt.imshow(dataobj.data[1000,:,:])
    # plt.show()
    # print(np.where(np.isnan(dataobj.bad_pixels)))
    # exit()

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

    # Definition of the (extra) parameters for splinefm()
    fm_paras = {"planet_f":planet_f,"transmission":transmission,"star_spectrum":star_spectrum,
                "boxw":1,"nodes":20,"psfw":1.2,"nodes":20,"badpixfraction":0.75}
    fm_func = hc_splinefm
    # fm_paras = {"planet_f":planet_f,"transmission":transmission,"boxw":1,"res_hpf":100,"psfw":1.2,"badpixfraction":0.75}
    # fm_func = iso_hpffm

    if 0: # Example code to test the forward model
        nonlin_paras = [-15,30,9] # x (pix),y (pix), rv (km/s)
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
        plt.subplot(2,1,2)
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
    out = search_planet([rvs,ys,xs],dataobj,fm_func,fm_paras,numthreads=32)
    N_linpara = (out.shape[-1]-2)//2
    print(out.shape)

    plt.figure(1)
    plt.imshow(out[0,:,:,3]/out[0,:,:,3+N_linpara],origin="lower")
    # plt.imshow(out[0,:,:,0]-out[0,:,:,1],origin="lower")
    # plt.clim([0,20])
    cbar = plt.colorbar()
    cbar.set_label("SNR")
    # plt.plot(out[:,0,0,2])
    plt.show()
