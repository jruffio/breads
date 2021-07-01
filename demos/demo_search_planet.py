from breads.instruments.OSIRIS import OSIRIS
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    filename = "../../public_osiris_data/kap_And/20161106/science/s161106_a020002_Kbb_020.fits"
    datacube = OSIRIS(filename)

    #seach for planets demo

    plt.imshow(np.nansum(datacube.spaxel_cube,axis=0))
    plt.show()
