from breads.instruments.OSIRIS import OSIRIS
import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d

from breads.instruments.OSIRIS import OSIRIS
from breads.search_planet import search_planet


def splinemodel(nonlin_paras,cubeobj,planet_f=None, boxw=0,transmission=None,star_spectrum=None):
    """
    doc
    """
    x, y, rv = nonlin_paras
    wvs = cubeobj.wavelengths

    return transmission[rv]*planet_f(wvs[rv])


if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    filename = "../../public_osiris_data/kap_And/20161106/science/s161106_a020002_Kbb_020.fits"
    dataobj = OSIRIS(filename)

    planet_f = interp1d(np.linspace(0,5,10000),np.sin(np.linspace(0,5,10000)*100))
    transmission = 0.5*np.ones(1600)
    star_spectrum = np.nansum(dataobj.spaxel_cube,axis=0)

    fm_paras = {"planet_f":planet_f,"boxw":3,"transmission":transmission,"star_spectrum":star_spectrum}
    nonlin_paras = [10,10,10] # x (pix),y (pix), rv (km/s)
    print(splinemodel(nonlin_paras,dataobj,**fm_paras))

    #seach for planets demo
    xs = np.zeros(1)
    ys = np.zeros(1)
    rvs = np.arange(1600)
    out = search_planet([xs,ys,rvs],dataobj,splinemodel,fm_paras,numthreads=32)
    print(out.shape)

    plt.plot(out[0,0,:])
    plt.show()
