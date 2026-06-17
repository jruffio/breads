import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.stats import median_abs_deviation

import astropy.io.fits as pyfits
from astropy import constants as const
from astropy import units as u
from astropy.table import Table
from astropy.io import fits

from breads.jwst_tools.splines import evaluate_3dspline_grid
from breads.jwst_tools.plotting import save_cube_as_gif

def create_BreadsPSF(spline3d_filename, x_vec, y_vec, wv_sampling,basename=None,numthreads=1,stis_spectrum=None,
                     overwrite=False,units_str = None,stamp_size=None):
    BREADS_DATA_ENV = os.getenv('BREADS_DATA')
    breadsPSF_DIR = os.path.join(BREADS_DATA_ENV, "BreadsPSF")
    if not os.path.exists(breadsPSF_DIR):
        os.makedirs(breadsPSF_DIR)

    if not overwrite and len(glob(os.path.join(breadsPSF_DIR,basename))) >= 1:
        with pyfits.open(os.path.join(breadsPSF_DIR,basename)) as hdul:
            breadspsf = hdul["EPSFS"].data
            breadspsf_err = hdul["EPSFS_ERR"].data
        return breadspsf, breadspsf_err

    if basename is None:
        basename = "breadsPSF.fits"

    _out = evaluate_3dspline_grid(x_vec, y_vec, wv_sampling, spline3d_filename, max_cores=numthreads,stamp_size=stamp_size)
    breadspsf, breadspsf_err = _out

    hdulist = pyfits.open(spline3d_filename)
    wv_nodes = hdulist["wv_nodes"].data
    x_nodes = hdulist["x_nodes"].data
    y_nodes = hdulist["y_nodes"].data
    spline3d_paras = hdulist["SPLINE_PARAS0"].data
    spline3d_paras_err = hdulist["SPLINE_PARAS0_ERR"].data
    breads_header = hdulist['BREADS'].header
    hdulist.close()

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
        nodes_stis = stis_func(wv_nodes)
        spline3d_paras /= nodes_stis[None,None,:,None,None]
        spline3d_paras_err /= nodes_stis[None,None,:,None,None]
        if units_str is None:
            units_str = "[MJy/sr]/[1MJy]"
    else:
        if units_str is None:
            units_str = "MJy"

    units = pyfits.Header({'BUNIT': units_str})

    if stamp_size is not None:
        breads_header['3DSPLSSX'] = stamp_size[0]
        breads_header['3DSPLSSY']= stamp_size[1]
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
    hdulist.append(pyfits.ImageHDU(data=spline3d_paras, name='SPLINE_PARAS', header=units))
    hdulist.append(pyfits.ImageHDU(data=spline3d_paras_err, name='SPLINE_PARAS_ERR', header=units))
    hdulist.writeto(out_filename, overwrite=True)
    hdulist.close()

    dx = x_vec[1] - x_vec[0]
    dy = y_vec[1] - y_vec[0]
    extent = [x_vec[0] - dx / 2.0, x_vec[-1] + dx / 2.0, y_vec[0] - dy / 2.0, y_vec[-1] + dy / 2.0]
    vmax = 4 + np.log10(np.abs(median_abs_deviation(breadspsf[np.where(np.isfinite(breadspsf))])))
    save_cube_as_gif(np.log10(np.abs(breadspsf[::50])), filename=out_filename.replace(".fits", ".gif"),
                     fps=24, vmin=vmax-6, vmax=vmax, extent=extent, wv_nodes=wv_sampling[::50], dpi=100)

    return breadspsf, breadspsf_err


def merge_breadspsfs(breadsPSF_basename_init, J1757132_breadsPSF_basename,breadsPSF_combined_init_basename, spline3d_filename_combined,
                     x_vec, y_vec, wv_sampling,
                     mask_charge_transfer_radius=0,dist_mask=None,numthreads=1,overwrite = False):
    BREADS_DATA_ENV = os.getenv('BREADS_DATA')
    breadsPSF_combined_init_path = os.path.join(BREADS_DATA_ENV, "BreadsPSF", breadsPSF_combined_init_basename)
    utils_dir = os.path.dirname(spline3d_filename_combined)
    if overwrite or not os.path.exists(breadsPSF_combined_init_path):  # combine faint and bright star breadsPSFs

        breadsPSF_path = os.path.join(BREADS_DATA_ENV, "BreadsPSF", J1757132_breadsPSF_basename)
        print(breadsPSF_path)
        hdulist = fits.open(breadsPSF_path)
        psf0_X = hdulist['X'].data
        spline3d_paras0 = hdulist['SPLINE_PARAS'].data
        spline3d_paras_err0 = hdulist['SPLINE_PARAS_ERR'].data
        x_nodes0 = hdulist['x_nodes'].data
        y_nodes0 = hdulist['y_nodes'].data
        wv_nodes0 = hdulist['wv_nodes'].data
        # J1757132_breadspsf_unit = hdulist['EPSFS'].header['BUNIT']
        hdulist.close()

        if dist_mask is None:
            dist_mask = np.nanmax(np.abs(psf0_X))

        breadsPSF_path = os.path.join(BREADS_DATA_ENV, "BreadsPSF", breadsPSF_basename_init)
        print(breadsPSF_path)
        hdulist = fits.open(breadsPSF_path)
        breadspsf_header = hdulist[0].header
        spline3d_paras1 = hdulist['SPLINE_PARAS'].data
        spline3d_paras_err1 = hdulist['SPLINE_PARAS_ERR'].data
        psf1_X = hdulist['X'].data
        psf1_Y = hdulist['Y'].data
        wv1_sampling = hdulist['WAVE'].data
        x_nodes1 = hdulist['x_nodes'].data
        y_nodes1 = hdulist['y_nodes'].data
        wv_nodes1 = hdulist['wv_nodes'].data
        # breadspsf_unit = hdulist['EPSFS'].header['BUNIT']
        hdulist.close()

        print(x_nodes0)
        print(x_nodes1)
        if not np.allclose(wv_nodes0, wv_nodes1):
            raise Exception("BreadsPSF wv_nodes are different.")
        if not np.allclose(x_nodes0, x_nodes1):
            raise Exception("BreadsPSF x_nodes are different.")
        if not np.allclose(y_nodes0, y_nodes1):
            raise Exception("BreadsPSF y_nodes are different.")

        xx, yy = np.meshgrid(x_nodes1, y_nodes1)
        rr = np.sqrt(xx ** 2 + yy ** 2)
        xx = np.tile(xx[None, :, :], (2,2,np.size(wv_nodes1), 1, 1))
        yy = np.tile(yy[None, :, :], (2,2,np.size(wv_nodes1), 1, 1))
        rr = np.tile(rr[None, :, :], (2,2,np.size(wv_nodes1), 1, 1))
        where_good = np.where(np.isfinite(spline3d_paras0) & np.isfinite(spline3d_paras1) & np.isfinite(spline3d_paras_err0) &
                              np.isfinite(spline3d_paras_err1) & (np.abs(xx) < dist_mask) & (np.abs(yy) < dist_mask) &
                              (np.abs(xx) >= mask_charge_transfer_radius) & np.abs(rr >= (2 * mask_charge_transfer_radius)))

        num = np.nansum(spline3d_paras0[where_good] * spline3d_paras1[where_good] / (
                spline3d_paras_err0[where_good] ** 2 + spline3d_paras_err1[where_good] ** 2))  #
        denum = np.nansum(spline3d_paras1[where_good] ** 2 / (
                spline3d_paras_err0[where_good] ** 2 + spline3d_paras_err1[where_good] ** 2))  #
        scaling = num / denum
        spline3d_paras1_scaled = scaling * spline3d_paras1
        spline3d_paras1_scaled_err = scaling * spline3d_paras_err1

        where_mask = np.where((np.abs(xx) <= mask_charge_transfer_radius) | (rr <= (2 * mask_charge_transfer_radius)) |
                              ~np.isfinite(spline3d_paras1_scaled) | ~np.isfinite(spline3d_paras1_scaled_err))
        spline3d_paras1_scaled[where_mask] = np.nan
        spline3d_paras1_scaled_err[where_mask] = np.nan
        where_mask = np.where((np.abs(xx) > dist_mask) | ((np.abs(yy) > dist_mask) & (np.abs(xx) >= mask_charge_transfer_radius)) |
                              ~np.isfinite(spline3d_paras0) | ~np.isfinite(spline3d_paras_err0))
        spline3d_paras0[where_mask] = np.nan
        spline3d_paras_err0[where_mask] = np.nan

        arr_tmp = np.concatenate([spline3d_paras0[None,:, :, :, :, :], spline3d_paras1_scaled[None,:, :, :, :, :]], axis=0)
        arr_err_tmp = np.concatenate([spline3d_paras_err0[None,:, :, :, :, :], spline3d_paras1_scaled_err[None,:, :, :, :, :]],axis=0)

        denum = np.nansum(1 / arr_err_tmp ** 2, axis=0)
        spline3d_paras_combined = np.nansum(arr_tmp / arr_err_tmp ** 2, axis=0) / denum
        spline3d_paras_combined_err = 1 / np.sqrt(denum)

        plt.figure()
        mid_id = spline3d_paras0.shape[1]//2
        plt.errorbar(x_nodes0, np.nanmedian(spline3d_paras0,axis=(0,1))[2, mid_id, :],
                     yerr=np.nanmedian(spline3d_paras_err0,axis=(0,1))[2, mid_id, :],
                     label='J1757132 (faint)', fmt="")
        plt.errorbar(x_nodes1, np.nanmedian(spline3d_paras1_scaled,axis=(0,1))[2, mid_id, :],
                     yerr=np.nanmedian(spline3d_paras1_scaled_err,axis=(0,1))[2, mid_id, :],
                     label='scaled HD163466', fmt="")
        plt.errorbar(x_nodes1, np.nanmedian(spline3d_paras_combined,axis=(0,1))[2, mid_id, :],
                     yerr=np.nanmedian(spline3d_paras_combined_err,axis=(0,1))[2, mid_id, :],
                     label='Combined', fmt="--")
        plt.yscale("log")
        plt.xlim([-1.4, 1.4])
        plt.xlabel("ifu x")
        plt.ylabel("Flux ([MJy/sr]/[1MJy])")
        plt.gca().invert_xaxis()
        plt.legend()

        plt.savefig(os.path.join(utils_dir, breadsPSF_combined_init_basename.replace(".fits", "_combination_cut_preview.png")),dpi=200)

        vmax = 4+np.log10(np.abs(median_abs_deviation(spline3d_paras0[np.where(np.isfinite(spline3d_paras0))])))
        dx = x_nodes1[1] - x_nodes1[0]
        dy = y_nodes1[1] - y_nodes1[0]
        extent = [x_nodes1[0] - dx / 2.0, x_nodes1[-1] + dx / 2.0, y_nodes1[0] - dy / 2.0, y_nodes1[-1] + dy / 2.0]
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.title("First PSF")
        plt.imshow(np.log10(np.abs(np.nanmedian(spline3d_paras0,axis=(0,1))[2, :, :])), origin='lower', extent=extent)
        plt.clim([vmax-6, vmax])
        plt.gca().invert_xaxis()

        plt.subplot(1, 3, 2)
        plt.title("Second PSF")
        plt.imshow(np.log10(np.abs(np.nanmedian(spline3d_paras1_scaled,axis=(0,1))[2, :, :])), origin='lower', extent=extent)
        plt.clim([vmax-6, vmax])
        plt.gca().invert_xaxis()

        plt.subplot(1, 3, 3)
        plt.title("Combined")
        plt.imshow(np.log10(np.abs(np.nanmedian(spline3d_paras_combined,axis=(0,1))[2, :, :])), origin='lower', extent=extent)
        plt.clim([vmax-6, vmax])
        plt.gca().invert_xaxis()
        # plt.show()

        plt.savefig(os.path.join(utils_dir, breadsPSF_combined_init_basename.replace(".fits", "_combination_im_preview.png")),dpi=200)

        # save combined:
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU())
        hdulist.append(fits.ImageHDU(data=spline3d_paras_combined, name='SPLINE_PARAS0'))
        hdulist.append(fits.ImageHDU(data=spline3d_paras_combined_err, name='SPLINE_PARAS0_ERR'))
        hdulist.append(fits.ImageHDU(data=wv_nodes1, name='wv_nodes'))
        hdulist.append(fits.ImageHDU(data=x_nodes1, name='x_nodes'))
        hdulist.append(fits.ImageHDU(data=y_nodes1, name='y_nodes'))
        hdulist.append(fits.ImageHDU(header=breadspsf_header, name='BREADS'))
        hdulist.writeto(spline3d_filename_combined, overwrite=True)
        hdulist.close()

        _out = evaluate_3dspline_grid(x_vec, y_vec, wv_sampling, spline3d_filename_combined,
                                      max_cores=numthreads)
        breadspsf, breadspsf_err = _out

        units = fits.Header({'BUNIT': "[MJy/sr]/[1MJy]"})
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(header=breadspsf_header))
        hdulist.append(fits.ImageHDU(data=breadspsf, name='EPSFS', header=units))
        hdulist.append(fits.ImageHDU(data=breadspsf_err, name='EPSFS_ERR', header=units))
        hdulist.append(fits.ImageHDU(data=wv_sampling, name='WAVE'))
        hdulist.append(fits.ImageHDU(data=psf1_X, name='X'))
        hdulist.append(fits.ImageHDU(data=psf1_Y, name='Y'))
        hdulist.append(fits.ImageHDU(data=x_nodes1, name='x_nodes'))
        hdulist.append(fits.ImageHDU(data=y_nodes1, name='y_nodes'))
        hdulist.append(fits.ImageHDU(data=wv_nodes1, name='wv_nodes'))
        hdulist.append(fits.ImageHDU(data=spline3d_paras_combined, name='SPLINE_PARAS', header=units))
        hdulist.append(fits.ImageHDU(data=spline3d_paras_combined_err, name='SPLINE_PARAS_ERR', header=units))
        hdulist.writeto(breadsPSF_combined_init_path, overwrite=True)
        hdulist.close()

        dx = x_vec[1] - x_vec[0]
        dy = y_vec[1] - y_vec[0]
        extent = [x_vec[0] - dx / 2.0, x_vec[-1] + dx / 2.0, y_vec[0] - dy / 2.0, y_vec[-1] + dy / 2.0]
        vmax = 4 + np.log10(np.abs(median_abs_deviation(breadspsf[np.where(np.isfinite(breadspsf))])))
        save_cube_as_gif(np.log10(np.abs(breadspsf[::50])),
                         filename=breadsPSF_combined_init_path.replace(".fits", ".gif"),
                         fps=24, vmin=vmax - 6, vmax=vmax, extent=extent, wv_nodes=wv_sampling[::50], dpi=100)
