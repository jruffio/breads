import numpy as np
from astropy.io import fits
import os

from astropy.stats import sigma_clip
from scipy.ndimage import gaussian_filter

from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

import math
import matplotlib
from scipy.signal import find_peaks

try:
    import jwst
    _HAS_OPTIONAL_DEPENDENCY_JWST = True
except ImportError:
    _HAS_OPTIONAL_DEPENDENCY_JWST = False

try:
    from BayesicFitting import Fitter, SplinesModel, LevenbergMarquardtFitter
except ImportError:
    # these are optional dependencies for BREADS JWST support; OK to ignore them for
    # non-JWST BREADS usage
    pass


def use_find_files_to_process(input_dir, filetype='uncal.fits'):
    from breads.jwst_tools.reduction_utils import find_files_to_process
    return find_files_to_process(input_dir, filetype=filetype)

def get_band_miri_header(prim_hdr):
    band = prim_hdr['BAND']
    channel = prim_hdr['CHANNEL']
    if band=='SHORT':
        band='A'
    elif band=='MEDIUM':
        band='B'
    elif band=='LONG':
        band='C'
    else:
        raise ValueError('band must be SHORT, MEDIUM, LONG')
    band = channel + band
    return band

def column_median_max(mat):
    medianes = np.nanmedian(mat, axis=0)
    col_index = np.nanargmax(medianes)
    return col_index

def find_brightest_cols_two_channels(data):
    col_id_1 = column_median_max(data[:, :500])
    col_id_2 = column_median_max(data[:, 500:]) + 500
    return col_id_1, col_id_2

def find_psf_peak_channel_2D(data, row_id, alpha, beta, band, detector_part='left'):
    from lmfit.models import VoigtModel

    if detector_part == 'left':
        brightest_col = column_median_max(data[:, :500])
    elif detector_part == 'right':
        brightest_col = column_median_max(data[:, 500:]) + 500
    else:
        raise ValueError(f"Unknown value {detector_part} for detector_part must be either 'left' or 'right'")

    print(f"Brightest column in {band} identified: {brightest_col}")

    model = VoigtModel()

    x_data = alpha[row_id, brightest_col - 10:brightest_col + 10]
    y = data[row_id, brightest_col - 10:brightest_col + 10]
    y = y[np.isfinite(x_data)]
    x_data = x_data[np.isfinite(x_data)]

    x_data = x_data[np.isfinite(y)]
    y=y[np.isfinite(y)]

    params = model.guess(y, x=x_data)
    result = model.fit(y, params, x=x_data)
    best_params = result.best_values

    beta_center = beta[row_id, brightest_col]

    return best_params['center'], beta_center


def replace_nan_with_median(image, dq, size=3):
    """
    Remplace les valeurs NaN et les valeurs inférieures à 0 d'une image par la médiane locale.

    :param image: np.ndarray, image avec des NaN ou des valeurs négatives
    :param size: int, taille du filtre médian
    :return: np.ndarray, image avec les NaN et les valeurs négatives remplacés
    """
    mask = np.isnan(image) | (dq > 4)
    filtered = median_filter(np.where(mask, 0, image), size=size)
    image[mask] = filtered[mask]
    return image

def miri_flat_running_mean(data_rate_path, output_dir, band, overwrite=False):

    filenames = os.listdir(data_rate_path)

    for filename in filenames:
        if filename.endswith("rate.fits"):
            print(f"Computing running mean flat for {filename}")

            hdu_f = fits.open(os.path.join(data_rate_path, filename))
            img = hdu_f[1].data
            DQ = hdu_f['DQ'].data
            prim_header = hdu_f[0].header
            hdu_f.close()
            if get_band_miri_header(prim_header) != band:
                print("wrong band", prim_header['BAND'], band)
                continue

            im_flat = np.zeros_like(img, dtype=np.float64) + np.nan
            col_min, col_max = 10, 1020  # 572, 1020

            img_no_nan = replace_nan_with_median(img, DQ)

            for j in range(col_min, col_max):
                if j<500:
                    sigma = 8
                else:
                    sigma = 12
                try:
                    lamb_micron = np.arange(0, 1024, 1)
                    where_finite_wave = np.where(np.isfinite(lamb_micron))
                    y_data = img_no_nan[:, j]

                    continuum = gaussian_filter(y_data, sigma=sigma)
                    flat = y_data/continuum

                    clip_data = sigma_clip(flat, sigma=3)
                    mask_clip = clip_data.mask

                    continuum[mask_clip] = np.nan

                    flat = y_data / continuum
                    flat[np.isnan(flat)] = 1

                    im_flat[where_finite_wave, j] = flat
                except Exception as e:
                    print(e)
                    im_flat[where_finite_wave, j] = np.nan

            im_flat_extended = np.copy(im_flat)

            im_flat[im_flat>1.3] = 1 #Hard thresholding
            im_flat[im_flat<0.5] = 1

            im_filt = img / im_flat

            hdu1 = fits.ImageHDU(data=im_flat, name='FLAT')
            hdu2 = fits.ImageHDU(data=im_filt, name='FLAT_IMAGE')
            hdu3 = fits.ImageHDU(data=im_flat_extended, name='FLAT_EXTENDED')

            # Create PrimaryHDU and HDUlist
            primary_hdu = fits.PrimaryHDU()
            primary_hdu.header = prim_header

            hdul = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3])

            flat_name = filename.replace("_rate.fits", "_flat.fits")
            output_name = os.path.join(output_dir, flat_name)
            hdul.writeto(output_name, overwrite=overwrite)
            print(f"==> Estimated fringe flat written to {output_name}")


def run_miri_flat_running_mean(flat_path, targetname, output_dir=None, list_bands=None, overwrite=True):
    if output_dir is None:
        output_dir = os.getenv("FLAT_PATH")
        if output_dir is None:
            raise ValueError("No FLAT_PATH specified to save the fringe flat")

    if list_bands is None:
        list_bands = ['12A', '12B', '12C', '34A', '34B', '34C']
    for band in list_bands:
        print(f"Computing flat for band {band}")
        data_rate_path_band = os.path.join(flat_path, targetname, band, 'stage1')
        output_dir_band = os.path.join(output_dir, band)
        print("[DEBUG] Writing flat image to {}".format(output_dir_band))
        if not os.path.exists(output_dir_band):
            os.makedirs(output_dir_band)

        miri_flat_running_mean(data_rate_path_band, output_dir_band, band, overwrite=overwrite)


def plot_flat(uncal_dir, target_name, list_bands=None):
    if list_bands is None:
        list_bands = ['12A', '12B', '12C', '34A', '34B', '34C']
    for band in list_bands:
        inputdir_stage1 = os.path.join(uncal_dir, target_name, band, 'stage1')
        print(inputdir_stage1)
        inputdir_stage1_flat = os.path.join(uncal_dir, target_name, band, 'stage1_flat')

        files_stage1 = use_find_files_to_process(inputdir_stage1, filetype='rate.fits')
        files_stage1_flat = use_find_files_to_process(inputdir_stage1_flat, filetype='rate.fits')

        for file_stage1, file_stage1_flat in zip(files_stage1, files_stage1_flat):
            data_stage1 = fits.getdata(file_stage1)
            data_stage1_flat = fits.getdata(file_stage1_flat)
            idx_max = column_median_max(data_stage1)

            plt.plot(data_stage1[:, idx_max], label='stage1')
            plt.plot(data_stage1_flat[:, idx_max], label='stage1_flat')
            plt.show()

    return 1

def plot_starsub_fit(uncal_dir, utils_dir, target_name, list_bands=None):
    if list_bands is None:
        list_bands = ['12A', '12B', '12C', '34A', '34B', '34C']

    for band in list_bands:
        inputdir_starsub = os.path.join(uncal_dir, target_name, utils_dir)
        print(inputdir_starsub)

        files_starsub = use_find_files_to_process(inputdir_starsub, filetype='starsub.fits')

        for file_starsub in files_starsub:
            data_original = fits.open(file_starsub)['IM'].data
            data_starmodel = fits.open(file_starsub)['STARMODEL'].data
            bp = fits.open(file_starsub)['BADPIX'].data
            res = fits.getdata(file_starsub)
            idx_max = column_median_max(data_original)

            for i in range(-2, 3, 1):
                plt.plot((data_original+bp)[:, idx_max+i], label='Star')
                plt.plot((data_starmodel+bp)[:, idx_max+i], label='Spline Model')
                plt.legend()
                plt.show()

                plt.plot((res+bp)[:, idx_max+i], label='res')
                plt.legend()
                plt.show()

    return 1

def beta_slice_ID(beta, channel):
    beta_c = np.copy(beta)

    if channel == 1 or channel == 4:
        col_min, col_max = 10, 500
        beta_c[:, 500:] = np.nan

    else:
        col_min, col_max = 500, 1020
        beta_c[:, :500] = np.nan

    row = beta[500, col_min:col_max]
    beta_values = np.unique(row)

    beta_slice_num = np.zeros_like(beta) + np.nan

    for i, value in enumerate(beta_values):
        beta_slice_num[beta_c == value] = channel * 100 + i

    beta_slice_num[:10, :] = np.nan
    beta_slice_num[-10:, :] = np.nan
    beta_slice_num[:, :10] = np.nan


    return beta_slice_num

def beta_masking_slice(beta, channel, liste_ID):
    beta_slice_num = beta_slice_ID(beta, channel)
    mask = np.ones_like(beta_slice_num)

    if len(liste_ID) > 0:
        for ID in liste_ID:
            mask[beta_slice_num == ID] = 0
    return mask

def beta_masking_slice_col(beta, channel, liste_col_ID):
    beta_slice_num = beta_slice_ID(beta, channel)
    mask = np.ones_like(beta_slice_num)

    if len(liste_col_ID) > 0:
        for col_ID in liste_col_ID:
            ID = np.nanmax(beta_slice_num[:, col_ID])
            print(f"Slice ID: {ID}")
            mask[beta_slice_num == ID] = 0

    return mask

def find_brightest_slices(data, channel, plot=False):
    data_copy = data.copy()
    if channel == 1 or channel == 4:
        data_copy[:, 500:] = np.nan
    else:
        data_copy[:, :500] = np.nan

    medianes = np.nanmedian(data_copy, axis=0)

    # Find the peaks to identify center of the slices
    peaks, _ = find_peaks(medianes)

    # Extract peaks height
    peak_heights = medianes[peaks]

    top_indices = np.argsort(peak_heights)
    top_peaks_values = peak_heights[top_indices]

    topN_indices = []

    for i in range(len(top_peaks_values)):
        topN_indices.append(np.where(medianes == top_peaks_values[i])[0])
    topN_indices = np.array(topN_indices)

    print(topN_indices, len(peaks))
    if plot:
        plt.plot(medianes)
        plt.scatter(topN_indices, medianes[topN_indices])
        plt.show()

    return topN_indices

def beta_masking_inverse_slice(data, beta, channel, N_slices=4):
    beta_slice_num = beta_slice_ID(beta, channel)
    list_col_ID = find_brightest_slices(data, channel)

    mask = np.zeros_like(beta_slice_num)
    if channel == 1 or channel == 4:
        mask[:, 500:] = 1
    elif channel == 2 or channel == 3:
        mask[:, :500] = 1

    list_ID =[]
    if len(list_col_ID) > 0:
        for col_ID in list_col_ID:
            list_ID.append(np.nanmax(beta_slice_num[:, col_ID]))

    list_ID = np.array(list_ID)[::-1]
    slice_id_masked = []
    for ID in list_ID:
        if len(slice_id_masked)>N_slices-1:
            continue
        if ID not in slice_id_masked:
            slice_id_masked.append(ID)
            print(f"Slice ID: {ID}")
            mask[beta_slice_num == ID] = 1

    return mask


def miri_flat_splines(data_rate_path, output_dir, band, overwrite=False):
    filenames = os.listdir(data_rate_path)

    for filename in filenames:
        if filename.endswith("rate.fits"):
            print(f"Computing running mean flat for {filename}")

            hdu_f = fits.open(os.path.join(data_rate_path, filename))
            img = hdu_f[1].data
            DQ = hdu_f['DQ'].data
            err = hdu_f['ERR'].data
            prim_header = hdu_f[0].header
            hdu_f.close()
            if get_band_miri_header(prim_header) != band:
                print("wrong band", prim_header['BAND'], band)
                continue

            col_min, col_max = 10, 1020  # 572, 1020

            flat = np.zeros_like(img, dtype=np.float64) + np.nan

            for j in range(col_min, col_max):
                try:
                    flux = img[:, j]
                    flux_0 = np.copy(flux)
                    y_idx = np.arange(flux.shape[0])
                    y_idx_0 = np.copy(y_idx)
                    y_idx = y_idx[np.isfinite(flux)]
                    flux = flux[np.isfinite(flux)]

                    N_nodes_continuum_array = [200, 70]

                    sigma = [5, 2]

                    for i in range(2):
                        N_nodes_continuum = N_nodes_continuum_array[i]
                        continuum = SplinesModel(nrknots=N_nodes_continuum, xrange=y_idx_0)
                        pars = [np.nanmean(flux)] + [0.] * (N_nodes_continuum + 1)  # pars for continuum splines
                        continuum.parameters = pars  # insert initial parameters

                        fitter = LevenbergMarquardtFitter(y_idx, continuum)
                        param = fitter.fit(flux, plot=False)

                        continuum_est = continuum.result(y_idx, param)

                        ratio = flux / continuum_est
                        clipped = sigma_clip(ratio, sigma=sigma[i])

                        where_good_pix = np.where(~clipped.mask)[0]

                        y_idx, flux = y_idx[where_good_pix], flux[where_good_pix]

                    pars = [np.nanmean(flux)] + [0.] * (N_nodes_continuum + 1)  # pars for continuum splines
                    continuum.parameters = pars  # insert initial parameters

                    fitter = LevenbergMarquardtFitter(y_idx, continuum)
                    param = fitter.fit(flux, plot=False)
                    continuum_est = continuum.result(y_idx_0, param)

                    flat[:, j] = flux_0 / continuum_est
                except Exception as e:
                    print(e)
                    flat[:, j] *= np.nan

            hdu1 = fits.ImageHDU(data=flat, name='FLAT')
            hdu2 = fits.ImageHDU(data=flat, name='FLAT_IMAGE')
            hdu3 = fits.ImageHDU(data=flat, name='FLAT_EXTENDED')

            # Create PrimaryHDU and HDUlist
            primary_hdu = fits.PrimaryHDU()
            primary_hdu.header = prim_header

            hdul = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3])

            flat_name = filename.replace("_rate.fits", "_flat_splines.fits")
            hdul.writeto(os.path.join(output_dir, flat_name), overwrite=overwrite)
