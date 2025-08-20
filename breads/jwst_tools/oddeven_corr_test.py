from temp_fit_to_sup import oddEvenCorrection_prop
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import os

import matplotlib

matplotlib.use("tkAgg")

def oddEvenCorrectionImage(hdu, channel=1):
    data = hdu["SCI"].data
    nbLine0, nbColumn0 = data.shape
    prim_hdr = hdu[0].header

    band = prim_hdr["BAND"]
    channel_data = prim_hdr["CHANNEL"]
    if channel_data == '12':
        if band == 'SHORT':
            hdu_coor = fits.open("C:/Users/abidot/Desktop/coor_miri/jwst_mirifushort_short_coor.fits")
        elif band == 'MEDIUM':
            hdu_coor = fits.open("C:/Users/abidot/Desktop/coor_miri/jwst_mirifushort_short_medium.fits")
        else:
            print(f"band {band} not supported")
    else:
        print(f"channel {channel_data} not supported")

    beta = hdu_coor['beta'].data

    if channel == 1:
        col_min = 5
        col_max = 500
    else:
        col_min = 500
        col_max = 1024

    beta = beta[:, col_min:col_max]
    data = data[:, col_min:col_max]
    beta_unique = np.unique(beta[500,:])
    nbLine, nbColumn = data.shape
    y = np.arange(0, nbLine)
    x = np.arange(0, nbColumn)
    xx, yy = np.meshgrid(x, y)

    print(yy.shape)
    print(xx.shape)
    flux_corr_2D = np.zeros((nbLine0, nbColumn0))


    for i, bet in enumerate(beta_unique):
        slice_data = data[beta==bet]
        print(i, len(beta_unique), slice_data.shape)
        if len(slice_data)>0:
            slice_yy = yy[beta==bet]
            slice_xx = xx[beta==bet]
            flux_corr = oddEvenCorrection_prop(slice_data, slice_xx, slice_yy, plot=False)
            for f, x, y in zip(flux_corr, slice_xx, slice_yy):
                flux_corr_2D[y, x + col_min] = f

    return flux_corr_2D

path = "C:/Users/abidot/Desktop/beta_pic/HR8799_CH1_stage1/"
files = os.listdir(path)

for file in files:
    if file.endswith("rate.fits"):
        print(file)
        filename = os.path.basename(file)
        filename_save = filename.replace(".fits", "_corr.fits")
        hdu = fits.open(os.path.join(path, file))
        flux_corr_2D = oddEvenCorrectionImage(hdu, channel=1)
        flux_corr_2D += oddEvenCorrectionImage(hdu, channel=2)

        # Ouvrir le fichier source
        with fits.open(os.path.join(path, file)) as hdul:
            # Copier tous les HDUs (Header/Data Units)
            new_hdul = fits.HDUList([hdu.copy() for hdu in hdul])

            # Modifier les données de l'extension 'SCI'
            sci_hdu = new_hdul['SCI']
            #matplotlib.use("tkAgg")
            plt.title("comp")
            plt.plot(sci_hdu.data[:, 372])
            plt.plot(flux_corr_2D[:, 372])
            plt.show()
            sci_hdu.data = flux_corr_2D

            # Sauvegarder dans un nouveau fichier
            new_hdul.writeto(os.path.join(path, filename_save), overwrite=True)

            plt.imshow(flux_corr_2D - hdu['SCI'].data, vmin=-0.1, vmax=0.1)
            plt.show()


#fits.writeto("C:/Users/abidot/Desktop/beta_pic/N_eta_ref/short/corr.fits", flux_corr_2D, overwrite=True)