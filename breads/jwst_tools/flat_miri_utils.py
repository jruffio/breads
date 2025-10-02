import numpy as np
from astropy.io import fits
from lmfit.models import VoigtModel
import os

from astropy.stats import sigma_clip
from scipy.ndimage import gaussian_filter

from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

import math
from BayesicFitting import Fitter
from BayesicFitting import SplinesModel

import matplotlib
from scipy.signal import find_peaks

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

def select_band_coor(band, crds_path):
    if band == '12A':
        hdu = fits.open(os.path.join(crds_path, "references/jwst/miri/12A/jwst_mirifushort_short_coor.fits"))
    elif band == '12B':
        hdu = fits.open(os.path.join(crds_path, "references/jwst/miri/12B/jwst_mirifushort_medium_coor.fits"))
    elif band == '12C':
        hdu = fits.open(os.path.join(crds_path, "references/jwst/miri/12C/jwst_mirifushort_long_coor.fits"))
    elif band == '34A':
        hdu = fits.open(os.path.join(crds_path, "references/jwst/miri/34A/jwst_mirifulong_short_coor.fits"))
    elif band == '34B':
        hdu = fits.open(os.path.join(crds_path, "references/jwst/miri/34B/jwst_mirifulong_medium_coor.fits"))
    elif band == '34C':
        hdu = fits.open(os.path.join(crds_path, "references/jwst/miri/34C/jwst_mirifulong_long_coor.fits"))
    else:
        raise ValueError(f'band must be 12A, 12B, 12C, 34A, 34B, 34C, not {band}')
    return hdu

def colonne_median_max(mat):
    medianes = np.nanmedian(mat, axis=0)
    col_index = np.nanargmax(medianes)
    return col_index

def find_brightest_cols_two_channels(data):
    col_id_1 = colonne_median_max(data[:, :500])
    col_id_2 = colonne_median_max(data[:, 500:]) + 500
    return col_id_1, col_id_2

def find_psf_peak_channel_2D(data, row_id, crds_path, band, detector_part='left'):
    hdu = select_band_coor(band, crds_path)
    alpha = hdu["alpha"].data
    beta = hdu["beta"].data

    if detector_part == 'left':
        brightest_col = colonne_median_max(data[:, :500])
    elif detector_part == 'right':
        brightest_col = colonne_median_max(data[:, 500:]) + 500
    else:
        raise ValueError(f"Unknown value {detector_part} for detector_part must be either 'left' or 'right'")

    print(f"Brightest column identified: {brightest_col}")

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

def miri_flat_running_mean(flat_rate_path, output_dir, crds_path, band, overwrite=False):

    filenames = os.listdir(flat_rate_path)

    hdu = select_band_coor(band, crds_path)
    alpha = hdu["alpha"].data
    beta = hdu["beta"].data
    wave = hdu['LAMBDA'].data
    hdu.close()

    dbeta_left = np.abs(np.nanmin(np.diff(np.unique(beta[500, :500]))))
    dbeta_right = np.abs(np.nanmin(np.diff(np.unique(beta[500, 500:]))))

    mask = np.ones_like(alpha)
    mask[np.isnan(alpha)] = np.nan

    for filename in filenames:
        if filename.endswith("rate.fits"):
            print(f"Computing running mean flat for {filename}")

            hdu_f = fits.open(os.path.join(flat_rate_path, filename))
            img = hdu_f[1].data
            DQ = hdu_f['DQ'].data
            prim_header = hdu_f[0].header
            hdu_f.close()
            if get_band_miri_header(prim_header) != band:
                print("wrong band", prim_header['BAND'], band)
                continue

            im_flat = np.zeros_like(img, dtype=np.float64) + np.nan
            im_SNR = np.zeros_like(img, dtype=np.float64) + np.nan

            col_min, col_max = 10, 1020  # 572, 1020

            img_no_nan = replace_nan_with_median(img, DQ)

            for j in range(col_min, col_max):
                try:
                    lamb_micron = np.arange(0, 1024, 1)
                    where_finite_wave = np.where(np.isfinite(lamb_micron))
                    y_data = img_no_nan[:, j]

                    continuum = gaussian_filter(y_data, 8)
                    flat = y_data/continuum

                    clip_data = sigma_clip(flat, sigma=3)
                    mask_clip = clip_data.mask

                    continuum[mask_clip] = np.nan

                    flat = y_data / continuum
                    flat[np.isnan(flat)] = 1

                    im_flat[where_finite_wave, j] = flat
                    im_SNR[where_finite_wave, j] = y_data/np.sqrt(y_data)
                except Exception as e:
                    print(e)
                    im_flat[where_finite_wave, j] = np.nan

            im_flat_extended = np.copy(im_flat)*mask
            im_flat_extended[im_flat_extended > 1.3] = 1
            im_flat_extended[im_flat_extended < 0.5] = 1
            row_id = 500 #middle row of the detector

            alpha_peak, beta_center = find_psf_peak_channel_2D(img, row_id, crds_path, band, detector_part='left')
            beta_dist = beta[:, :500] - beta_center
            alpha_dist = alpha[:, :500] - alpha_peak
            where_too_far = np.where(np.abs(beta_dist)>2.2*dbeta_left)
            im_flat[where_too_far] = np.nan


            alpha_peak, beta_center = find_psf_peak_channel_2D(img, row_id, crds_path, band, detector_part='right')
            beta_dist = beta[:, 500:] - beta_center
            alpha_dist = alpha[:, 500:] - alpha_peak
            where_too_far = np.where(np.abs(beta_dist) > 2.2 * dbeta_right)
            where_too_far_off = where_too_far[1]+500
            im_flat[where_too_far[0], where_too_far_off] = np.nan


            im_flat[im_flat>1.3] = 1 #Hard thresholding
            im_flat[im_flat<0.5] = 1

            im_filt = img / im_flat

            im_flat *= mask

            hdu1 = fits.ImageHDU(data=im_flat, name='FLAT')
            hdu2 = fits.ImageHDU(data=wave, name='WAVELENGTH')
            hdu3 = fits.ImageHDU(data=im_filt, name='FLAT_IMAGE')
            hdu4 = fits.ImageHDU(data=im_flat_extended, name='FLAT_EXTENDED')

            # Create PrimaryHDU and HDUlist
            primary_hdu = fits.PrimaryHDU()
            primary_hdu.header = prim_header

            hdul = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3, hdu4])

            flat_name = filename.replace("_rate.fits", "_flat.fits")
            hdul.writeto(os.path.join(output_dir, flat_name), overwrite=overwrite)


def run_miri_flat_running_mean(flat_path, targetname, output_dir=None, list_bands=None, overwrite=True):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/miri_flat/")
    if list_bands is None:
        list_bands = ['12A', '12B', '12C', '34A', '34B', '34C']
    for band in list_bands:
        print(f"Computing flat for band {band}")
        flat_path_band = os.path.join(flat_path, targetname, band, 'stage1')
        output_dir_band = os.path.join(output_dir, band)
        crds_path = os.getenv('CUSTOM_CRDS_PATH')
        miri_flat_running_mean(flat_path_band, output_dir_band, crds_path, band, overwrite=overwrite)


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
            idx_max = colonne_median_max(data_stage1)

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
            idx_max = colonne_median_max(data_original)

            for i in range(-2, 3, 1):
                plt.plot((data_original+bp)[:, idx_max+i], label='Star')
                plt.plot((data_starmodel+bp)[:, idx_max+i], label='Spline Model')
                plt.legend()
                plt.show()

                plt.plot((res+bp)[:, idx_max+i], label='res')
                plt.legend()
                plt.show()

    return 1

def oddEvenCorrectionImage(hdu, crds_path, plot=False):
    data = hdu["SCI"].data
    nbLine0, nbColumn0 = data.shape
    prim_hdr = hdu[0].header

    channel_band =  get_band_miri_header(prim_hdr)
    hdu_coor = select_band_coor(channel_band, crds_path)

    beta = hdu_coor['beta'].data

    channels =['left', 'right']
    flux_corr_2D = np.zeros((nbLine0, nbColumn0))
    for channel in channels:
        print(channel)
        if channel == 'left':
            col_min = 5
            col_max = 500
        else:
            col_min = 500
            col_max = 1024

        beta_crop = beta[:, col_min:col_max]
        data_crop = data[:, col_min:col_max]
        beta_unique = np.unique(beta_crop[500, :]) #taking the middle-ish row of the detector
        nbLine, nbColumn = data_crop.shape
        y = np.arange(0, nbLine)
        x = np.arange(0, nbColumn)
        xx, yy = np.meshgrid(x, y)

        if channel == 'right':
            print(beta_unique)

        for i, bet in enumerate(beta_unique):
            slice_data = data_crop[beta_crop==bet]
            if channel == 'right':
                print(i, len(beta_unique), slice_data.shape)
            if len(slice_data)>0:
                slice_yy = yy[beta_crop==bet]
                slice_xx = xx[beta_crop==bet]
                flux_corr = oddEvenCorrection_prop(slice_data, slice_xx, slice_yy, plot=plot)
                for f, x, y in zip(flux_corr, slice_xx, slice_yy):
                    flux_corr_2D[y, x + col_min] = f

    return flux_corr_2D


def fitSplines(xs, y, weight=None, minknots=0, maxknots=10,
               min=None, max=None):
    """
    Try fit Splines with knots from 2..maxknots+2

    Parameters
    ----------
    xs : array
        xdata input
    y : array
        ydata to be fitted
    weights : array
        of the fit
    minknots : int

    """
    dmx = np.max(xs) if max is None else max
    dmn = np.min(xs) if min is None else min
    best = -math.inf
    for k in range(minknots, maxknots + 1):
        poly = SplinesModel(nrknots=k + 2, min=dmn, max=dmx)

        isfin = np.isfinite(y)
        xsfin = xs[isfin]
        yfin = y[isfin]
        weightfin = weight[isfin]

        ftr = Fitter(xsfin, poly)
        param = ftr.fit(yfin, weights=weightfin)
        yfit = poly.result(xsfin)

        try:
            ev = ftr.getEvidence(limits=[-100, 100], noiseLimits=[1e-6, 1e-3])
        except:
            ev = -math.inf
        if ev > best:
            best = ev
            kb = k
            pb = param
    # Use best model
    poly = SplinesModel(nrknots=kb + 2, min=dmn, max=dmx)
    poly.parameters = pb

    return poly



def oddEvenCorrection_prop(flux, xpix, ypix, plot=False):
    """
    Correct flux differences between odd and even lines

    Parameters
    ----------
    flux : array of float
        array with fluxes
    xpix : array of int
        indices of pixels in x direction
    ypix : array of int
        indices of pixels in y direction
    plot : boolean
        produce plot in figure "Odd-Even Correction"

    Returns
    -------
    array of flux : corrected fluxes

    """

    # get a unique list of the x values
    xlines = list(set(xpix))
    print(xpix)
    print(ypix)
    if plot:
        plt.figure("Odd-Even Correction")

    triangle = np.asarray([1, 2, 1], dtype=float) / 4.0

    cor = np.asarray([], dtype=float)
    ycr = np.asarray([], dtype=float)

    for k in xlines:
        q = np.where(xpix == k)
        print(len(q[0]))
        if len(q[0]) < 10: continue
        fq = flux[q]
        yq = ypix[q]
        cc = np.convolve(fq, triangle, "full")[1:-1]
        fc = fq/cc

        dfq = (fc[:-1] - fc[1:]) / 2
        dfq = np.where(yq[1:] % 2 == 0, -dfq, dfq)

        clip = sigma_clip(dfq, sigma=3, maxiters=5)
        where_valid = np.where(~clip.mask)
        dfq = dfq[where_valid]
        plt.plot(dfq)
        plt.show()

        yc = 0.5 * (yq[1:] + yq[:-1])
        yc = yc[where_valid]
        cor = np.append(cor, dfq)
        ycr = np.append(ycr, yc)


    dmx = np.max(ycr)
    dmn = np.min(ycr)

    median = np.nanmedian(cor)
    quart = np.nanmedian(abs(cor - median))

    wgt = np.where(abs(cor - median) > 8 * quart, 0.0, 1.0)

    poly = fitSplines(ycr, cor, weight=wgt, minknots=5, maxknots=5,
                      min=0, max=1024)

    yfit = poly.result(ycr)

    if plot:
        plt.plot(ycr, yfit, 'r.')
        plt.plot(ycr, cor, 'g.')
        plt.plot([dmn, dmx], [median, median], 'b-')
        plt.ylim(median - 4 * quart, median + 4 * quart)
        plt.xlim(0, 1030)
        plt.xlabel("y-pixel")
        plt.ylabel("odd-even correction")
        plt.title("Odd/even Correction")
        plt.show()

        #plt.plot(flux)
        flux_old = np.copy(flux)

    for k in xlines:
        q = np.where(xpix == k)
        yq = ypix[q]
        fc = poly.result(yq)
        b = np.where(yq % 2 == 0)
        fc[b] *= -1
        fc += 1
        flux[q] = flux[q]*fc

        if plot:
            plt.title("flat odd even coef")
            plt.plot(yq, flux[q]/flux_old[q])
            print(np.nanpercentile(flux[q]/flux_old[q], 75))
            plt.show()

    return flux

def beta_slice_ID(channel, band):
    crds_path = os.getenv('CUSTOM_CRDS_PATH')
    hdu = select_band_coor(band, crds_path)
    beta = hdu["beta"].data
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

def beta_masking_slice(channel, band, liste_ID):
    beta_slice_num = beta_slice_ID(channel, band)
    mask = np.ones_like(beta_slice_num)

    if len(liste_ID) > 0:
        for ID in liste_ID:
            mask[beta_slice_num == ID] = 0
    return mask

def beta_masking_slice_col(channel, band, liste_col_ID):
    beta_slice_num = beta_slice_ID(channel, band)
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

def beta_masking_inverse_slice(data, channel, band, N_slices=4):
    beta_slice_num = beta_slice_ID(channel, band)
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