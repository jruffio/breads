import os
import numpy as np

import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.ndimage import median_filter

import astropy.io.fits as pyfits
from astropy import constants as const
from astropy import units as u
from astropy.table import Table

from breads.jwst_tools.splines import evaluate_3dspline_grid
from breads.jwst_tools.plotting import save_cube_as_gif

def create_BreadsPSF(spline3d_filename, x_vec, y_vec, wv_sampling,basename=None,numthreads=1,stis_spectrum=None):
    BREADS_DATA_ENV = os.getenv('BREADS_DATA')
    breadsPSF_DIR = os.path.join(BREADS_DATA_ENV, "BreadsPSF")
    if not os.path.exists(breadsPSF_DIR):
        os.makedirs(breadsPSF_DIR)

    if basename is None:
        basename = "breadsPSF.fits"

    hdulist = pyfits.open(spline3d_filename)
    wv_nodes = hdulist["wv_nodes"].data
    x_nodes = hdulist["x_nodes"].data
    y_nodes = hdulist["y_nodes"].data
    spline3d_paras = hdulist["SPLINE_PARAS0"].data
    spline3d_paras_err = hdulist["SPLINE_PARAS0_ERR"].data
    breads_header = hdulist['BREADS'].header
    hdulist.close()

    _out = evaluate_3dspline_grid(x_vec, y_vec, wv_sampling, spline3d_filename, N_overlap_nodes=2, max_cores=numthreads)
    breadspsf, breadspsf_err = _out

    if stis_spectrum is not None:
        stis_table = Table(pyfits.getdata(stis_spectrum, 1))
        stis_wvs = (np.array(stis_table["WAVELENGTH"]) * u.Angstrom).to(u.um).value  # angstroms -> mum
        stis_spec = np.array(
            stis_table["FLUX"]) * u.erg / u.s / u.cm ** 2 / u.Angstrom  # erg s-1 cm-2 A-1
        stis_spec = stis_spec.to(u.W * u.m ** -2 / u.um)
        stis_spec_Fnu = stis_spec * (stis_wvs * u.um) ** 2 / const.c  # from Flambda back to Fnu
        stis_spec_Fnu = stis_spec_Fnu.to(u.MJy).value

        smoothed = median_filter(stis_spec_Fnu, size=200)  # adjust kernel size as needed
        stis_func = interp1d(stis_wvs,smoothed)

        # plt.plot(stis_wvs, stis_spec_Fnu, label="original", alpha=0.5)
        # plt.plot(stis_wvs, smoothed, label="median filtered")
        # plt.legend()
        # plt.show()

        sampled_stis = stis_func(wv_sampling)
        breadspsf /= sampled_stis[:,None,None]
        breadspsf_err /= sampled_stis[:,None,None]
        units = pyfits.Header({'BUNIT': "[MJy/sr]/[1MJy]"})
        vmin, vmax = 0.0, 1e11
    else:
        units = pyfits.Header({'BUNIT': "MJy"})
        vmin, vmax = 0.0, 500

    out_filename = os.path.join(breadsPSF_DIR, basename)
    x_grid, y_grid = np.meshgrid(x_vec, y_vec)
    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.PrimaryHDU(header=breads_header))
    hdulist.append(pyfits.ImageHDU(data=breadspsf, name='EPSFS', header=units))
    hdulist.append(pyfits.ImageHDU(data=breadspsf_err, name='EPSFS_ERR', header=units))
    hdulist.append(pyfits.ImageHDU(data=wv_sampling, name='WAVE'))
    hdulist.append(pyfits.ImageHDU(data=x_grid, name='X'))
    hdulist.append(pyfits.ImageHDU(data=y_grid, name='Y'))
    hdulist.append(pyfits.ImageHDU(data=x_nodes, name='x_nodes'))
    hdulist.append(pyfits.ImageHDU(data=y_nodes, name='y_nodes'))
    hdulist.append(pyfits.ImageHDU(data=wv_nodes, name='wv_nodes'))
    hdulist.append(pyfits.ImageHDU(data=spline3d_paras, name='SPLINE_PARAS'))
    hdulist.append(pyfits.ImageHDU(data=spline3d_paras_err, name='SPLINE_PARAS_ERR'))
    hdulist.writeto(out_filename, overwrite=True)
    hdulist.close()

    dx = x_vec[1] - x_vec[0]
    dy = y_vec[1] - y_vec[0]
    extent = [x_vec[0] - dx / 2.0, x_vec[-1] + dx / 2.0, y_vec[0] - dy / 2.0, y_vec[-1] + dy / 2.0]

    save_cube_as_gif(breadspsf[::50], filename=out_filename.replace(".fits", ".gif"),
                     fps=24, vmin=vmin, vmax=vmax, extent=extent, wv_nodes=wv_sampling[::50], dpi=100)

    return breadspsf, breadspsf_err
