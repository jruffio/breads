import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import scipy.io as scio
import astropy.io.fits as pyfits

from breads.instruments.OSIRIS import OSIRIS
from breads.search_planet import search_planet
from breads.search_planet import splinefm

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    filename = "../../public_osiris_data/kap_And/20161106/science/s161106_a020002_Kbb_020.fits"
    dataobj = OSIRIS(filename)
    nz,ny,nx = dataobj.spaxel_cube.shape
    dataobj.noise_cube = np.ones((nz,ny,nx))

    if 1:
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

    # Extra parameters for splinefm()
    fm_paras = {"planet_f":planet_f,"boxw":3,"N_nodes":20,"transmission":transmission,"star_spectrum":star_spectrum}
    # nonlin_paras = [-15,37,10] # x (pix),y (pix), rv (km/s)
    # d, M, s, bounds = splinefm(nonlin_paras,dataobj,**fm_paras)
    # plt.plot(d)
    # plt.plot(M[:,0])
    # plt.show()

    # fit rv
    # rvs = np.linspace(-2000,2000,201)
    # ys = np.array([37])
    # xs = np.array([10])
    #seach for planets demo
    rvs = np.array([-15])
    ys = np.arange(ny)
    xs = np.arange(nx)
    out = search_planet([rvs,ys,xs],dataobj,splinefm,fm_paras,numthreads=32)
    N_linpara = (out.shape[-1]-2)//2
    print(out.shape)
    print(out)

    plt.imshow(out[0,:,:,2]/out[0,:,:,2+N_linpara],origin="lower")
    cbar = plt.colorbar()
    cbar.set_label("SNR")
    # plt.plot(out[:,0,0,2])
    plt.show()
