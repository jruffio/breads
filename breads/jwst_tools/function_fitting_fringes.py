import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from certifi import where
from h5py.h5pl import append
from scipy.optimize import curve_fit
import os
from scipy.interpolate import interp1d
import traceback

from scipy.interpolate import LSQUnivariateSpline
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from astropy.stats import sigma_clip
import matplotlib


from scipy.ndimage import gaussian_filter, median_filter
import math

from BayesicFitting import PolynomialModel, UniformPrior, CircularUniformPrior
from BayesicFitting import EtalonModel, EtalonDriftPositive, PositiveEtalonModel
from BayesicFitting import SplinesModel
from BayesicFitting import SineModel
from BayesicFitting import LevenbergMarquardtFitter
from BayesicFitting import formatter as fmt

# Initialize the Formatter to give better looking results
from BayesicFitting import formatter_init as init
init( linelength=60, indent=13 )

def FPfunc_noPhaseShift(wavenumber, F, D, theta=0):
    return (1 + F * np.sin(2*np.pi*D*wavenumber*np.cos(theta))**2)**-1

def replace_nan_with_median(image, dq, size=3):
    """
    Remplace les valeurs NaN et les valeurs inférieures à 0 d'une image par la médiane locale.

    :param image: np.ndarray, image avec des NaN ou des valeurs négatives
    :param size: int, taille du filtre médian
    :return: np.ndarray, image avec les NaN remplacés
    """
    mask = np.isnan(image) | (dq > 4)
    filtered = median_filter(np.where(mask, 0, image), size=size)
    image[mask] = filtered[mask]
    return image

def micron_to_wavenumber(wave):
    return 1e4/wave

def get_wave(hdu):
    prim_header = hdu[0].header
    channel = prim_header['CHANNEL']
    band = prim_header['BAND']
    if channel == '12':
        if band=='SHORT':
            wave = fits.open("C:/Users/abidot/Desktop/coor_miri/jwst_mirifushort_short_coor.fits")['LAMBDA'].data
        elif band=='MEDIUM':
            wave = fits.open("C:/Users/abidot/Desktop/coor_miri/jwst_mirifushort_medium_coor.fits")['LAMBDA'].data
        elif band=='LONG':
            print("short long not supported yet")
        else:
            print(f"band {band} not supported")
    elif channel == '23':
        print("Longer channels not supported yet")
    else:
        print(f"channel/band {channel}/{band} not supported yet")

    return wave

def retrieve_data(hdu_data_flat):
    data = hdu_data_flat['SCI'].data
    DQ = hdu_data_flat['DQ'].data
    data_no_nan = replace_nan_with_median(np.copy(data), DQ)
    err = hdu_data_flat['ERR'].data

    return data, data_no_nan, DQ, err

def _get_fringes_transmission_column(data, err, DQ, wave, col_id, snr_threshold=10):
    y_data = data[:, col_id]
    dq = DQ[:, col_id]
    lamb = wave[:, col_id]
    y_data[dq > 5] = np.nan
    snr = y_data/err[:, col_id]
    where_finite = np.where(np.isfinite(y_data) & np.isfinite(lamb) & (snr>snr_threshold))[0]

    y_data = y_data[where_finite]
    #x_data = np.arange(0, y_data.shape[0])

    lamb = lamb[where_finite]

    peaks, _ = find_peaks(y_data, distance=6) #TODO depend on the channel
    troughs, _ = find_peaks(-y_data, distance=6)

    cs = CubicSpline(lamb[peaks], y_data[peaks])
    continuum_firstguess = cs(lamb)

    new_peaks, _ = find_peaks(y_data / continuum_firstguess, distance=6)
    cs2 = CubicSpline(lamb[new_peaks], y_data[new_peaks])
    continuum = cs2(lamb)

    fringes_transmission = y_data / continuum

    return fringes_transmission, lamb, new_peaks

def fit_FabryPerot_transmittance(fringes_transmission, lamb, peaks, plot=True):
    lamb_wavenumber = micron_to_wavenumber(lamb)
    nbPeaks = len(peaks)
    print(f"Number of peaks: {nbPeaks}")
    finesse = []
    D = []
    fringes_transmission_fitted = []
    lamb_for_plot = []
    for i in range(nbPeaks-1):
        lamb_crop = lamb_wavenumber[peaks[i]: peaks[i+1]]
        fringes_crop = fringes_transmission[peaks[i]: peaks[i+1]]

        p0 = [0.3, 1714 / 10000]
        best_params, covariance = curve_fit(FPfunc_noPhaseShift, lamb_crop, fringes_crop, p0=p0)

        finesse.append(best_params[0])
        D.append(best_params[1])

        lamb_for_plot.append(lamb_crop)
        fringes_transmission_fitted.append(FPfunc_noPhaseShift(lamb_crop, *best_params))

    lamb_for_plot = np.concatenate(lamb_for_plot)
    fringes_transmission_fitted = np.concatenate(fringes_transmission_fitted)

    if plot is True:
        plt.title("Fringes transmittance fit")
        plt.ylabel("Transmission")
        plt.xlabel("Wavenumber (cm-1)")
        plt.plot(lamb_wavenumber, fringes_transmission, label='empirical fringes transmittance')
        plt.plot(lamb_for_plot, fringes_transmission_fitted, label='fitted fringes transmittance')
        plt.show()



    return fringes_transmission_fitted, finesse, D, peaks

def fit_FabryPerot_science_transmittance(fringes_transmission, lamb, peaks, finesse_flat, D_flat, plot=True):
    lamb_wavenumber = micron_to_wavenumber(lamb)
    nbPeaks = len(peaks)
    print(f"Number of peaks: {nbPeaks}")
    finesse = []
    D = []
    fringes_transmission_fitted = []
    lamb_for_plot = []
    for i in range(nbPeaks - 1):
        lamb_crop = lamb_wavenumber[peaks[i]: peaks[i + 1]]
        fringes_crop = fringes_transmission[peaks[i]: peaks[i + 1]]

        p0 = [finesse_flat[i], D_flat[i]]
        bounds = ([-np.inf, 0.98*D_flat[i]], [np.inf, 1.02*D_flat[i]])
        best_params, covariance = curve_fit(FPfunc_noPhaseShift, lamb_crop, fringes_crop, p0=p0, bounds=bounds)

        finesse.append(best_params[0])
        D.append(best_params[1])

        lamb_for_plot.append(lamb_crop)
        fringes_transmission_fitted.append(FPfunc_noPhaseShift(lamb_crop, *best_params))

    lamb_for_plot = np.concatenate(lamb_for_plot)
    fringes_transmission_fitted = np.concatenate(fringes_transmission_fitted)

    if plot is True:
        plt.title("Fringes transmittance fit on science data")
        plt.ylabel("Transmission")
        plt.xlabel("Wavenumber (cm-1)")
        plt.plot(lamb_wavenumber, fringes_transmission, label='empirical fringes transmittance')
        plt.plot(lamb_for_plot, fringes_transmission_fitted, label='fitted fringes transmittance')
        plt.show()


    return fringes_transmission_fitted, finesse, D, peaks

def get_interpolator_fitnesse_etalon(finesse, D, lamb_wavenumber, peaks, new_lamb_axis):
    nbPeaks = len(peaks)
    lamb_crop = []
    finesse_crop = []
    D_crop = []
    for i in range(nbPeaks-1):
        lamb_crop.append(lamb_wavenumber[peaks[i]: peaks[i + 1]])
        finesse_crop.append(np.zeros_like(lamb_wavenumber[peaks[i]: peaks[i + 1]]) + finesse[i])
        D_crop.append(np.zeros_like(lamb_wavenumber[peaks[i]: peaks[i + 1]]) + D[i])

    lamb_concatenate = np.concatenate(lamb_crop)
    finesse_concatenate = np.concatenate(finesse_crop)
    D_concatenate = np.concatenate(D_crop)

    f_interp = interp1d(lamb_concatenate, finesse_concatenate, bounds_error=False, fill_value=0)
    D_interp = interp1d(lamb_concatenate, D_concatenate, bounds_error=False, fill_value=0)

    f_fitted = f_interp(new_lamb_axis)
    D_fitted = D_interp(new_lamb_axis)
    fringes_transmission_fitted = FPfunc_noPhaseShift(new_lamb_axis, f_fitted, D_fitted, theta=0)


    return fringes_transmission_fitted


def evaluate_fringes_residuals(data, wave, err, col_id, finesse, D, lamb, peaks, plot=True, title=None, show=True):
    wave = micron_to_wavenumber(wave)
    lamb_wavenumber = micron_to_wavenumber(lamb)
    new_lamb_axis = wave[:, col_id]
    where_finite = np.where(np.isfinite(new_lamb_axis) & (new_lamb_axis>0))[0]
    new_lamb_axis = new_lamb_axis[where_finite]

    fringes_transmission_fitted = get_interpolator_fitnesse_etalon(finesse, D, lamb_wavenumber, peaks, new_lamb_axis)

    y = data[where_finite, col_id]/fringes_transmission_fitted
    '''plt.plot(new_lamb_axis, y)
    plt.plot(new_lamb_axis, data[where_finite, col_id])
    plt.show()'''

    y -= gaussian_filter(y, sigma=8)
    std = np.nanstd(y)

    err_bar = err[where_finite, col_id]

    if plot:
        if title is None:
            plt.title("Residual fringes")
        else:
            plt.title(title)
        plt.xlabel("Wavenumber (cm-1)")
        plt.ylabel("Residual")
        plt.plot(new_lamb_axis, y, label='residual')
        plt.fill_between(new_lamb_axis, -err_bar, err_bar, color='gray', alpha=0.3, label='Erreur attendue')
        plt.legend()
        if show:
            plt.show()

    return std

def get_flat_old(hdu_science, hdu_data_flat, save_path, snr_thresh=20, spectrum=None, N_continuum=50, N_D=15, N_finesse=20, mask_star_lines=True):

    data, data_no_nan, DQ, err = retrieve_data(hdu_data_flat)
    data_science, data_science_no_nan, DQ_science, err_science = retrieve_data(hdu_science)

    alpha = fits.open("C:/Users/abidot/Desktop/coor_miri/jwst_mirifushort_short_coor.fits")['ALPHA'].data

    wave = get_wave(hdu_data_flat)  # Loading the wavelength map in micron

    flat = np.zeros_like(data_science) + 1
    D = np.zeros_like(data_science) + np.nan

    for col_id in range(5, 500):
        print(col_id)
        try:
            flat[:, col_id], D[:, col_id] = fit_FP_bayesian_drift(data_science, wave, err_science, DQ_science, col_id, alpha, snr_thresh=snr_thresh, spectrum=spectrum, N_continuum=N_continuum, N_D=N_D, N_finesse=N_finesse,
                                            mask_star_lines=mask_star_lines)
        except Exception as e:
            print(e)

    fits.writeto(save_path, flat, overwrite=True)

    '''

        # Estimate the fringes transmission by continuum spline fitting on the peaks
        fringes_transmission, lamb, peaks = _get_fringes_transmission_column(data_no_nan, err, DQ, wave, col_id,
                                                                             snr_threshold=10)
        # Fit the fringes transmission with Fabry Perot transmittance function
        fringes_transmission_fitted, finesse_flat, D_flat, peaks_flat = fit_FabryPerot_transmittance(fringes_transmission,
                                                                                                     lamb, peaks, plot=True)

        fringes_transmission_science, lamb_science, peaks_science = _get_fringes_transmission_column(data_science_no_nan,
                                                                                                     err_science,
                                                                                                     DQ_science, wave,
                                                                                                     col_id,
                                                                                                     snr_threshold=10)
        fringes_transmission_science_fitted, finesse_science, D_science, peaks_science = fit_FabryPerot_science_transmittance(
            fringes_transmission_science, lamb_science, peaks_science, finesse_flat, D_flat, plot=True)

        plt.plot(lamb_science, fringes_transmission_science)
        plt.plot(lamb, fringes_transmission)
        plt.show()

        print(evaluate_fringes_residuals(data_science, wave, err, col_id, finesse_science, D_science, lamb_science, peaks_science, plot=True, title="res with fit", show=False))
        print(evaluate_fringes_residuals(data_science, wave, err, col_id, finesse_flat, D_flat, lamb, peaks_flat, plot=True))'''


def get_flat_old_col(hdu_science, col_id, snr_thresh=20, spectrum=None, N_continuum=50, N_D=15, N_finesse=20, mask_star_lines=False):
    data_science, data_science_no_nan, DQ_science, err_science = retrieve_data(hdu_science)

    alpha = fits.open("C:/Users/abidot/Desktop/coor_miri/jwst_mirifushort_short_coor.fits")['ALPHA'].data

    wave = get_wave(hdu_science)  # Loading the wavelength map in micron

    flat, _ = fit_FP_bayesian_drift(data_science, wave, err_science, DQ_science, col_id, alpha, plot=False, snr_thresh=snr_thresh,
                          spectrum=spectrum, N_continuum=N_continuum, N_D=N_D, N_finesse=N_finesse, mask_star_lines=mask_star_lines)

    return flat



def fit_FP_bayesian(data, wave, col_id):
    rmin = 100
    rmax = 900
    lamb = wave[rmin:rmax, col_id]
    where_finite = np.where(np.isfinite(lamb))[0]
    lamb = lamb[where_finite]

    flux = data[rmin:rmax, col_id]
    flux = flux[where_finite]

    weight = np.sqrt(flux)

    wnum = 1000/lamb

    N_nodes_continuum = 50
    N_nodes_finesse = 20
    N_nodes_D = 20

    continuum = SplinesModel(nrknots=N_nodes_continuum, xrange=wnum)

    finesse = SplinesModel(nrknots=N_nodes_finesse, xrange=wnum)
    '''finesse.setLimits([0]*(N_nodes_finesse+2), [np.inf]*(N_nodes_finesse+2)
    finesse.setPrior(1, prior=UniformPrior(), limits=[0, 1])'''

    D = SplinesModel(nrknots=N_nodes_D, xrange=wnum)

    mdl = EtalonModel(fixed={0: continuum, 1: finesse, 2: D, 3: 0})
    #pars = [0.0]  # 1 remaining pars of Etalon
    pars = [np.nanmean(flux)] + [0.] * (N_nodes_continuum+1)  # 12 pars for continuum splines
    pars += [0.35] + [0.0] * (N_nodes_finesse+1)  # 12 pars for finesse
    pars += [3.40] + [0.0] * (N_nodes_D + 1)


    mdl.parameters = pars  # insert initial parameters

    fitter = LevenbergMarquardtFitter(wnum, mdl)

    mdl_flat = EtalonModel(fixed={0: 1.0, 1: finesse, 2: D, 3:0})

    print(mdl_flat)

    param = fitter.fit(flux, weights=weight, plot=True)
    print("ICI", len(param))
    print("Phi:", param[0])


    param_crop = []
    #param_crop.append(param[0])
    for i in range(N_nodes_continuum+3-1, N_nodes_continuum+N_nodes_finesse+N_nodes_D+7-1):
        param_crop.append(param[i])
    print("ICI 2", param_crop)
    plt.plot(wnum, mdl_flat.result(wnum, param_crop))
    plt.show()

    param_crop_continuum = param[1-1:N_nodes_continuum+3-1]
    plt.plot(wnum, continuum.result(wnum,param_crop_continuum))
    plt.show()

    param_crop_finesse = param[N_nodes_continuum+3-1:N_nodes_continuum+N_nodes_finesse+5-1]
    print("ICIIII", np.nanmin(param_crop_finesse))
    plt.title("Finesse")
    plt.plot(wnum, finesse.result(wnum,param_crop_finesse))
    plt.show()

    param_crop_D = param[N_nodes_continuum+N_nodes_finesse+5-1:]
    print(len(param_crop_D))
    plt.title("D")
    plt.plot(wnum, D.result(wnum, param_crop_D))
    plt.show()

    print("Parameters :", fmt(param, max=None))
    print("StDevs     :", fmt(fitter.stdevs, max=None))
    print("Scale      :", fmt(fitter.scale))
    print("Evidence   :", fmt(fitter.getEvidence(limits=[-10, 10], noiseLimits=[0.01, 10])))

    return

def fit_FP_bayesian_2_Etalon(data, wave, err, dq, col_id):
    rmin = 100
    rmax = 900
    lamb = wave[rmin:rmax, col_id]
    where_finite = np.where(np.isfinite(lamb))[0]
    lamb = lamb[where_finite]

    flux = data[rmin:rmax, col_id]
    flux = flux[where_finite]

    err = err[rmin:rmax, col_id]
    err = err[where_finite]

    #weight = np.sqrt(flux)
    dq = dq[where_finite, col_id]

    #where_good_pixels = np.where(dq<2)[0]
    #lamb = lamb[where_good_pixels]
    #flux = flux[where_good_pixels]

    wnum = 1000 / lamb

    plt.plot(wnum, flux)
    plt.show()

    plt.title("SNR")
    plt.plot(wnum, flux/np.sqrt(flux))
    plt.plot(wnum, flux/err)
    plt.show()

    N_nodes_continuum = 50
    N_nodes_finesse = 5
    N_nodes_D = 20

    continuum = SplinesModel(nrknots=N_nodes_continuum, xrange=wnum)
    finesse = SplinesModel(nrknots=N_nodes_finesse, xrange=wnum)

    D = SplinesModel(nrknots=N_nodes_D, xrange=wnum)

    mdl = EtalonModel(fixed={0: continuum, 1: finesse, 2:D,  3: 0})
    #mdl *= EtalonModel(fixed={0: 1, 3: 0})

    pars = [np.nanmean(flux)] + [0.] * (N_nodes_continuum + 1)  # 12 pars for continuum splines
    pars += [0.35] + [0.0] * (N_nodes_finesse + 1)  # 12 pars for finesse
    pars += [3.40] + [0.0] * (N_nodes_D + 1)#frequency1
    #pars += [0.01] #finesse2
    #pars += [3.18] #frequency2

    mdl.parameters = pars  # insert initial parameters

    fitter = LevenbergMarquardtFitter(wnum, mdl)
    param = fitter.fit(flux, plot=True)

    print('frequency found', param[-1])

    mdl_flat = EtalonModel(fixed={0: 1.0, 1: finesse, 2: D, 3: 0})#*EtalonModel(fixed={0: 1, 3: 0})
    print(mdl_flat)


    print("ICI", len(param))
    print("Phi:", param[0])

    param_crop = param[N_nodes_continuum + 2:]
    plt.plot(wnum, mdl_flat.result(wnum, param_crop))
    plt.show()

    '''#####SECOND ITERATION STEP######
    peaks, _ = find_peaks(mdl_flat.result(wnum, param_crop))
    #N_nodes_continuum = len(peaks)
    N_nodes_finesse = 5
    N_nodes_D = 20

    #continuum = SplinesModel(knots=wnum[peaks], xrange=wnum)
    #fitter = LevenbergMarquardtFitter(wnum, continuum)
    #param = fitter.fit(flux, plot=True)
    idx_sort = np.argsort(wnum[peaks])
    wnum_peaks = wnum[peaks][::-1]
    flux_peaks = flux[peaks][::-1]

    cs2 = CubicSpline(wnum_peaks, flux_peaks)
    continuum = cs2(wnum)
    plt.plot(wnum, flux)
    plt.plot(wnum, continuum)
    plt.show()

    finesse = SplinesModel(nrknots=N_nodes_finesse, xrange=wnum)

    D = SplinesModel(nrknots=N_nodes_D, xrange=wnum)

    mdl = EtalonModel(fixed={0: 1, 1: finesse, 2: D, 3: 0})
    #print("PRIOR?", mdl.getPrior(0))
    # mdl *= EtalonModel(fixed={0: 1, 3: 0})

    #pars = [np.nanmean(flux)] + [0.] * (N_nodes_continuum + 1)  # 12 pars for continuum splines
    pars = [0.35] + [0.0] * (N_nodes_finesse + 1)
    pars += [3.40] + [0.0] * (N_nodes_D + 1)  # frequency1
    # pars += [0.01] #finesse2
    # pars += [3.18] #frequency2

    mdl.parameters = pars  # insert initial parameters

    fitter = LevenbergMarquardtFitter(wnum, mdl)
    param = fitter.fit(flux, plot=True)

    print('frequency found', param[-1])

    mdl_flat = EtalonModel(fixed={0: 1.0, 1: finesse, 2: D, 3: 0})  # *EtalonModel(fixed={0: 1, 3: 0})
    print(mdl_flat)
    '''
    plt.title("Fringe flat fitted")
    plt.plot(wnum, mdl_flat.result(wnum, param_crop))
    plt.ylabel("Transmission")
    plt.xlabel("Wavenumber (mm-1)")
    plt.show()





    param_crop_continuum = param[0:N_nodes_continuum + 3 - 1]

    plt.plot(wnum, continuum.result(wnum, param_crop_continuum))
    plt.plot(wnum, mdl.result(wnum, param))
    plt.show()

    param_crop_finesse = param[N_nodes_continuum + 2:N_nodes_continuum + N_nodes_finesse + 4]
    plt.title("Finesse")
    plt.plot(wnum, finesse.result(wnum, param_crop_finesse))
    plt.xlabel("Wavenumber (mm-1)")
    plt.ylabel("Finesse")
    plt.show()

    param_crop_D = param[N_nodes_continuum + N_nodes_finesse + 5 - 1:]
    print(len(param_crop_D))
    plt.title("D cos(theta)")
    plt.xlabel("Wavenumber (mm-1)")
    plt.ylabel("D cos(theta) (mm)")
    plt.plot(wnum, D.result(wnum, param_crop_D))
    plt.show()

    print("Parameters :", fmt(param, max=None))
    print("StDevs     :", fmt(fitter.stdevs, max=None))
    print("Scale      :", fmt(fitter.scale))
    #print("Evidence   :", fmt(fitter.getEvidence(limits=[-10, 10], noiseLimits=[0.01, 10])))

    return


def fit_FP_bayesian_chunk(data, wave, err_matrix, dq, col_id):
    rmin = 100
    rmax = 900
    size_chunk = 100
    chunks = np.arange(rmin, rmax, size_chunk)

    for chunk in chunks:

        lamb = wave[chunk:chunk+size_chunk, col_id]
        where_finite = np.where(np.isfinite(lamb))[0]
        lamb = lamb[where_finite]

        flux = data[chunk:chunk+size_chunk, col_id]
        flux = flux[where_finite]

        err = err_matrix[chunk:chunk+size_chunk, col_id]
        err = err[where_finite]

        # weight = np.sqrt(flux)
        #dq = dq[where_finite, col_id]

        wnum = 1000 / lamb

        plt.plot(wnum, flux)
        plt.show()

        plt.title("SNR")
        plt.plot(wnum, flux / np.sqrt(flux))
        plt.plot(wnum, flux / err)
        plt.show()

        N_nodes_continuum = 10
        N_nodes_finesse = 5
        N_nodes_D = 5

        continuum = SplinesModel(nrknots=N_nodes_continuum, xrange=wnum)
        finesse = SplinesModel(nrknots=N_nodes_finesse, xrange=wnum)

        D = SplinesModel(nrknots=N_nodes_D, xrange=wnum)

        mdl = EtalonModel(fixed={0: continuum, 1: finesse, 2: D, 3: 0})

        pars = [np.nanmean(flux)] + [0.] * (N_nodes_continuum + 1)  # 12 pars for continuum splines
        pars += [0.35] + [0.0] * (N_nodes_finesse + 1)  # 12 pars for finesse
        pars += [3.40] + [0.0] * (N_nodes_D + 1)  # frequency1

        mdl.parameters = pars  # insert initial parameters

        fitter = LevenbergMarquardtFitter(wnum, mdl)
        param = fitter.fit(flux, plot=True)

        mdl_flat = EtalonModel(fixed={0: 1.0, 1: finesse, 2: D, 3: 0})
        print(mdl_flat)

        param_crop = param[N_nodes_continuum + 2:]
        plt.plot(wnum, mdl_flat.result(wnum, param_crop))
        plt.show()

        plt.plot(wnum, flux / mdl_flat.result(wnum, param_crop))
        plt.show()

        param_crop_finesse = param[N_nodes_continuum + 2:N_nodes_continuum + N_nodes_finesse + 4]
        plt.title("Finesse")
        plt.plot(wnum, finesse.result(wnum, param_crop_finesse))
        plt.show()

        param_crop_D = param[N_nodes_continuum + N_nodes_finesse + 5 - 1:]
        print(len(param_crop_D))
        plt.title("D")
        plt.plot(wnum, D.result(wnum, param_crop_D))
        plt.show()

        print("Parameters :", fmt(param, max=None))
        print("StDevs     :", fmt(fitter.stdevs, max=None))
        print("Scale      :", fmt(fitter.scale))
    # print("Evidence   :", fmt(fitter.getEvidence(limits=[-10, 10], noiseLimits=[0.01, 10])))

    return

def fit_FP_bayesian_simple(data, wave, err, dq, col_id, snr_thresh):
    rmin = 100
    rmax = 900
    flat = np.zeros_like(wave[:, col_id]) + 1
    lamb = wave[rmin:rmax, col_id]
    where_finite = np.where(np.isfinite(lamb))[0]
    lamb = lamb[where_finite]

    flux = data[rmin:rmax, col_id]
    flux = flux[where_finite]

    err = err[rmin:rmax, col_id]
    err = err[where_finite]

    SNR_mean = np.nanmean(flux)#/err)
    if SNR_mean > snr_thresh:

        #weight = np.sqrt(flux)
        dq = dq[where_finite, col_id]

        #where_good_pixels = np.where(dq<2)[0]
        #lamb = lamb[where_good_pixels]
        #flux = flux[where_good_pixels]

        wnum = 1000 / lamb

        plt.plot(wnum, flux)
        plt.show()

        N_nodes_continuum = 50
        N_nodes_D = 10

        continuum = SplinesModel(nrknots=N_nodes_continuum, xrange=wnum)
        finesse = 0.15 #SplinesModel(nrknots=N_nodes_finesse, xrange=wnum)

        D = SplinesModel(nrknots=N_nodes_D, xrange=wnum)

        mdl = EtalonModel(fixed={0: continuum, 1: finesse, 2:D, 3:0})
        #mdl.setPrior(1, CircularUniformPrior)

        pars = [np.nanmean(flux)] + [0.] * (N_nodes_continuum + 1)  # 12 pars for continuum splines
        pars += [3.40] + [0.0] * (N_nodes_D + 1)#frequency1
        mdl.parameters = pars  # insert initial parameters

        fitter = LevenbergMarquardtFitter(wnum, mdl)
        param = fitter.fit(flux, plot=True)

        param_crop_continuum = param[:N_nodes_continuum + 2]
        plt.title("Continuum")
        plt.plot(wnum, continuum.result(wnum, param_crop_continuum))
        plt.plot(wnum, flux)
        plt.show()

        param_crop_D = param[N_nodes_continuum + 2:N_nodes_continuum + N_nodes_D + 4]
        plt.title("D")
        plt.plot(wnum, D.result(wnum, param_crop_D))
        plt.show()

        param_etalon = param[N_nodes_continuum + 2:]
        mdl_flat = EtalonModel(fixed={0: 1.0, 1: finesse, 2: D, 3: 0})
        plt.title("Etalon transmission")
        plt.plot(wnum, mdl_flat.result(wnum, param_etalon))
        plt.show()
        #D = D.result(wnum, param_crop_D)

        N_nodes_finesse = 5
        finesse = SplinesModel(nrknots=N_nodes_finesse, xrange=wnum)

        mdl = EtalonModel(fixed={0: continuum, 1: finesse, 2: D, 3: 0})
        pars = param_crop_continuum
        pars_2 = np.array(([0.35] + [0.0] * (N_nodes_finesse + 1)))
        pars_3 = param_crop_D
        print(len(pars))
        print(len(pars_2))
        pars = np.concatenate((pars, pars_2, pars_3))


        mdl.parameters = pars  # insert initial parameters
        fitter = LevenbergMarquardtFitter(wnum, mdl)
        param = fitter.fit(flux, plot=True)

        mdl_flat = PositiveEtalonModel(fixed={0: 1.0, 1: finesse, 2: D, 3: 0})
        print(mdl_flat)

        param_finesse_crop = param[N_nodes_continuum + 2:]
        plt.plot(wnum, mdl_flat.result(wnum, param_finesse_crop))
        plt.show()

        lamb_axis = wave[rmin:rmax, col_id]
        wnum_axis = 1000 / lamb_axis
        flat[rmin:rmax] = mdl_flat.result(wnum_axis, param_finesse_crop)


    return flat

def fit_FP_bayesian_simple_poly(data, wave, err, dq, col_id, snr_thresh):
    rmin = 100
    rmax = 900
    flat = np.zeros_like(wave[:, col_id]) + 1
    lamb = wave[rmin:rmax, col_id]
    where_finite = np.where(np.isfinite(lamb))[0]
    lamb = lamb[where_finite]

    flux = data[rmin:rmax, col_id]
    flux = flux[where_finite]

    err = err[rmin:rmax, col_id]
    err = err[where_finite]

    SNR_mean = np.nanmean(flux)#/err)
    if SNR_mean > snr_thresh:

        #weight = np.sqrt(flux)
        dq = dq[where_finite, col_id]

        #where_good_pixels = np.where(dq<2)[0]
        #lamb = lamb[where_good_pixels]
        #flux = flux[where_good_pixels]

        wnum = 1000 / lamb

        plt.plot(wnum, flux)
        plt.show()

        N_nodes_continuum = 50
        N_nodes_D = 40

        continuum = SplinesModel(nrknots=N_nodes_continuum, xrange=wnum)
        finesse = 0.2 #SplinesModel(nrknots=N_nodes_finesse, xrange=wnum)

        D = PolynomialModel(degree=N_nodes_D)#SplinesModel(nrknots=N_nodes_D, xrange=wnum)

        mdl = EtalonModel(fixed={0: continuum, 1: finesse, 2:D,  3: 0})

        pars = [np.nanmean(flux)] + [0.] * (N_nodes_continuum+1)  # 12 pars for continuum splines
        pars += [3.40] + [0.0] * (N_nodes_D)#frequency1

        mdl.parameters = pars  # insert initial parameters

        fitter = LevenbergMarquardtFitter(wnum, mdl)
        param = fitter.fit(flux, plot=True)

        param_crop_continuum = param[:N_nodes_continuum + 2]
        plt.title("Continuum")
        plt.plot(wnum, continuum.result(wnum, param_crop_continuum))
        plt.plot(wnum, flux)
        plt.show()

        param_crop_D = param[N_nodes_continuum + 2:N_nodes_continuum + N_nodes_D + 4]
        plt.title("D")
        plt.plot(wnum, D.result(wnum, param_crop_D))
        plt.show()
        #D = D.result(wnum, param_crop_D)

        N_nodes_finesse = 10
        finesse = SplinesModel(nrknots=N_nodes_finesse, xrange=wnum)

        mdl = EtalonModel(fixed={0: continuum, 1: finesse, 2: D, 3: 0})
        pars = param_crop_continuum
        pars_2 = np.array(([0.35] + [0.0] * (N_nodes_finesse + 1)))
        pars_3 = param_crop_D
        print(len(pars))
        print(len(pars_2))
        pars = np.concatenate((pars, pars_2, pars_3))


        mdl.parameters = pars  # insert initial parameters
        fitter = LevenbergMarquardtFitter(wnum, mdl)
        param = fitter.fit(flux, plot=True)

        mdl_flat = EtalonModel(fixed={0: 1.0, 1: finesse, 2: D, 3: 0})
        print(mdl_flat)

        param_finesse_crop = param[N_nodes_continuum + 2:]
        plt.plot(wnum, mdl_flat.result(wnum, param_finesse_crop))
        plt.show()

        lamb_axis = wave[rmin:rmax, col_id]
        wnum_axis = 1000 / lamb_axis
        flat[rmin:rmax] = mdl_flat.result(wnum_axis, param_finesse_crop)


    return flat

def masking_infinite(wave, data, err, col_id, rmin, rmax):
    flat = np.zeros_like(wave[:, col_id]) + 1
    D_est = np.zeros_like(wave[:, col_id]) + np.nan
    lamb = wave[rmin:rmax, col_id]
    where_finite = np.where(np.isfinite(lamb))[0]
    lamb = lamb[where_finite]

    flux = data[rmin:rmax, col_id]
    flux = flux[where_finite]

    err_c = err[rmin:rmax, col_id]
    err_c = err_c[where_finite]

    return flat, D_est, lamb, flux, err_c


def fit_FP_bayesian_drift(data, wave, err, dq, col_id, alpha, snr_thresh, plot=False, spectrum=None, N_continuum=50, N_D=15, N_finesse=10, mask_star_lines=True):
    rmin = 2
    rmax = 1022

    flat, D_est, lamb, flux, err_c = masking_infinite(wave, data, err, col_id, rmin, rmax)

    SNR_mean = np.nanmedian(flux/err_c)

    if SNR_mean > snr_thresh:
        print(f"SNR good enough for flat: col id {col_id}, SNR mean {SNR_mean}")
        if plot:
            plt.title("SNR")
            plt.plot(flux/err_c)
            plt.show()

        wnum = 1000 / lamb
        wnum, flux = identify_badpix(wnum, flux)

        if spectrum is not None:
            lamb_star, spectrum_star = spectrum
            f_interp = interp1d(lamb_star, spectrum_star, bounds_error=False, fill_value=1)
            spectrum_star = f_interp(1000/wnum)
            if mask_star_lines:
                wnum, flux = identify_stellar_absorption(wnum, flux, spectrum_star)


        N_nodes_continuum = N_continuum
        N_nodes_D = N_D

        testing_frequencies = np.arange(3.35, 3.45, 0.01)
        chi2 = np.zeros_like(testing_frequencies)

        for k, freq in enumerate(testing_frequencies):
            continuum = SplinesModel(nrknots=N_nodes_continuum, xrange=wnum)
            D = SplinesModel(nrknots=N_nodes_D, xrange=wnum)
            finesse = np.sqrt(0.15)
            mdl = PositiveEtalonModel(fixed={0: continuum, 1: finesse, 2: D, 3: 0})

            pars = [np.nanmean(flux)] + [0.] * (N_nodes_continuum + 1)  # pars for continuum splines
            pars += [freq] + [0.0] * (N_nodes_D + 1)#frequency
            mdl.parameters = pars  # insert initial parameters

            fitter = LevenbergMarquardtFitter(wnum, mdl)
            param = fitter.fit(flux, plot=plot)

            plot_D(wnum, param, N_nodes_continuum, N_nodes_D, col_id, plot=plot)

            param_crop_continuum = param[:N_nodes_continuum + 2]
            param_crop_D = param[N_nodes_continuum + 2:N_nodes_continuum + N_nodes_D + 4]

            N_nodes_finesse = N_finesse
            finesse = SplinesModel(nrknots=N_nodes_finesse, xrange=wnum)

            mdl = PositiveEtalonModel(fixed={0: continuum, 1: finesse, 2: D, 3: 0})

            pars = param_crop_continuum
            pars_2 = np.array(([0.35] + [0.0] * (N_nodes_finesse + 1)))
            pars_3 = param_crop_D
            pars = np.concatenate((pars, pars_2, pars_3))

            mdl.parameters = pars  # insert initial parameters
            fitter = LevenbergMarquardtFitter(wnum, mdl)
            param = fitter.fit(flux, plot=plot)
            chi2[k] = np.sum((flux - mdl.result(wnum, param)) ** 2)


        plt.title("chi2")
        plt.plot(testing_frequencies, chi2)
        plt.show()

        idx = np.argmin(chi2)
        best_freq = testing_frequencies[idx]

        continuum = SplinesModel(nrknots=N_nodes_continuum, xrange=wnum)
        D = SplinesModel(nrknots=N_nodes_D, xrange=wnum)
        finesse = np.sqrt(0.15)
        mdl = PositiveEtalonModel(fixed={0: continuum, 1: finesse, 2: D, 3: 0})

        pars = [np.nanmean(flux)] + [0.] * (N_nodes_continuum + 1)  # pars for continuum splines
        pars += [best_freq] + [0.0] * (N_nodes_D + 1)  # frequency
        mdl.parameters = pars  # insert initial parameters

        fitter = LevenbergMarquardtFitter(wnum, mdl)
        param = fitter.fit(flux, plot=True)

        param_crop_continuum = param[:N_nodes_continuum + 2]
        param_crop_D = param[N_nodes_continuum + 2:N_nodes_continuum + N_nodes_D + 4]

        N_nodes_finesse = N_finesse
        finesse = SplinesModel(nrknots=N_nodes_finesse, xrange=wnum)

        mdl = PositiveEtalonModel(fixed={0: continuum, 1: finesse, 2: D, 3: 0})
        pars = param_crop_continuum
        pars_2 = np.array(([0.35] + [0.0] * (N_nodes_finesse + 1)))
        pars_3 = param_crop_D
        pars = np.concatenate((pars, pars_2, pars_3))

        mdl.parameters = pars  # insert initial parameters
        fitter = LevenbergMarquardtFitter(wnum, mdl)
        param = fitter.fit(flux, plot=True)

        mdl_flat = PositiveEtalonModel(fixed={0: 1.0, 1: finesse, 2: D, 3: 0})
        print(mdl_flat)

        param_finesse_crop = param[N_nodes_continuum + 2:]
        plt.plot(wnum, mdl_flat.result(wnum, param_finesse_crop))
        plt.show()

        lamb_axis = wave[:, col_id]
        where_good_snr = np.where(data[:, col_id]/err[:, col_id] > snr_thresh)[0]
        wnum_axis = 1000 / lamb_axis[where_good_snr]
        flat[where_good_snr] = mdl_flat.result(wnum_axis, param_finesse_crop)

        param_crop_D = param[N_nodes_continuum + N_nodes_finesse + 4:N_nodes_continuum + N_nodes_D + N_nodes_finesse + 6]
        #D_est[rmin:rmax] = D.result(wnum_axis, param_crop_D)
        plt.title("Best D")
        plt.plot(wnum, D.result(wnum, param_crop_D))
        plt.show()

        if spectrum is not None and plot:
            f = mdl_flat.result(wnum, param_finesse_crop)
            flux_flat = flux / f
            flux_continuum = gaussian_filter(flux_flat, sigma=8)
            spectrum_star_continuum = gaussian_filter(spectrum_star, sigma=8)
            plt.title("Spec comparison")
            plt.plot(wnum, flux_flat/flux_continuum)
            plt.plot(wnum, spectrum_star/spectrum_star_continuum)
            plt.show()
            flux_hf = flux_flat - gaussian_filter(flux_flat, sigma=8)
            star_hf = spectrum_star - gaussian_filter(spectrum_star, sigma=8)

            print("Correlation :", np.dot(flux_hf, star_hf)/(np.linalg.norm(flux_hf)*np.linalg.norm(star_hf)))

    else:
        print(f"SNR not good enough for flat: col id {col_id}, SNR: {SNR_mean}")

    flat[np.isnan(flat)] = 1
    flat[flat<0.1] = 1

    return flat, D_est

def identify_stellar_absorption(wnum, flux, spectrum_star):
    spectrum_star_continuum = gaussian_filter(spectrum_star, sigma=3)
    absorption_ratio = spectrum_star/spectrum_star_continuum

    plt.title("spectrum")
    plt.plot(wnum, spectrum_star)
    plt.show()

    plt.title("Spectrum continuum normalized")
    plt.plot(wnum, absorption_ratio)
    plt.show()

    plt.title("Star absorption rejection")
    plt.scatter(wnum, flux, label="full spectrum")
    wnum = wnum[(absorption_ratio<1.01) & (absorption_ratio>0.99)]
    flux = flux[(absorption_ratio<1.01) & (absorption_ratio>0.99)]
    plt.scatter(wnum, flux, label="spectrum with feature masked")
    plt.legend()
    plt.show()

    return wnum, flux

def plot_D(wnum, param, N_nodes_continuum, N_nodes_D, col_id, plot=True):
    D = SplinesModel(nrknots=N_nodes_D, xrange=wnum)
    param_crop_D = param[N_nodes_continuum + 2:N_nodes_continuum + N_nodes_D + 4]
    if plot:
        plt.title(f"D for {col_id}")
        plt.plot(wnum, D.result(wnum, param_crop_D))
        plt.show()

def identify_badpix(wnum, flux, plot=False):
    wnum_0, flux_0 = np.copy(wnum), np.copy(flux)
    N_nodes_continuum = 50
    continuum = SplinesModel(nrknots=N_nodes_continuum, xrange=wnum)
    sigma = [3, 5]

    wnum = wnum[np.isfinite(flux)]
    flux = flux[np.isfinite(flux)]

    for i in range(2):
        pars = [np.nanmean(flux)] + [0.] * (N_nodes_continuum + 1)  # 12 pars for continuum splines
        continuum.parameters = pars  # insert initial parameters

        fitter = LevenbergMarquardtFitter(wnum, continuum)
        param = fitter.fit(flux, plot=plot)

        continuum_est = continuum.result(wnum, param)

        ratio = flux/continuum_est
        clipped = sigma_clip(ratio, sigma=sigma[i])

        where_good_pix = np.where(~clipped.mask)[0]

        wnum, flux = wnum[where_good_pix], flux[where_good_pix]

    if plot:
        plt.plot(wnum_0, flux_0)
        plt.scatter(wnum, flux, color='red')
        plt.show()

    return wnum, flux




col_id = 369
#hdu_data_flat = fits.open("C:/Users/abidot/Desktop/beta_pic/N_eta_ref/short/jw01294004001_03102_00001_mirifushort_rate.fits")
#hdu_science = fits.open("C:/Users/abidot/Desktop/beta_pic/N_eta_ref/short/corr.fits")
#hdu_science = fits.open("C:/Users/abidot/Desktop/beta_pic/CH1_stage1/jw01294003001_03102_00001_mirifushort_rate.fits")

path = "C:/Users/abidot/Desktop/beta_pic/HR8799_CH1_stage1/" #"C:/Users/abidot/Desktop/beta_pic/linearization_comp/"  #"C:/Users/abidot/Desktop/beta_pic/CH1_stage1/" #"C:/Users/abidot/Desktop/beta_pic/N_eta_ref/short/"
files = os.listdir(path)#"C:/Users/abidot/Desktop/beta_pic/N_eta_ref/short/jw01294004001_03102_00001_mirifushort_rate.fits"


def traiter_colonne(col_id, data_science, wave, err_science, DQ_science, alpha, snr_thresh):
    try:
        flat_col, D_col = fit_FP_bayesian_drift(
            data_science, wave, err_science, DQ_science,
            col_id, alpha, snr_thresh=snr_thresh
        )
        return col_id, flat_col, D_col, None
    except Exception as e:
        return col_id, None, None, str(e)

def get_flat(hdu_science, hdu_data_flat, col_id_ignored, save_path, snr_thresh=20):
    data, _, DQ, _ = retrieve_data(hdu_data_flat)
    data_science, _, DQ_science, err_science = retrieve_data(hdu_science)
    alpha = fits.open("C:/Users/abidot/Desktop/coor_miri/jwst_mirifushort_short_coor.fits")['ALPHA'].data
    wave = get_wave(hdu_data_flat)

    shape = data_science.shape
    flat = np.ones(shape)
    D = np.full(shape, np.nan)

    # Préparer la fonction partiellement appliquée
    func = partial(
        traiter_colonne,
        data_science=data_science,
        wave=wave,
        err_science=err_science,
        DQ_science=DQ_science,
        alpha=alpha,
        snr_thresh=snr_thresh
    )

    colonnes = list(range(192, 195))

    with Pool(processes=cpu_count()) as pool:
        for col_id, flat_col, D_col, err in pool.imap_unordered(func, colonnes):
            if err is None:
                flat[:, col_id] = flat_col
                D[:, col_id] = D_col
            else:
                print(f"Erreur à la colonne {col_id}: {err}")

    fits.writeto(save_path, flat, overwrite=True)

if __name__ == "__main__":
    path_spec = "C:/Users/abidot/Downloads/Cubes_HR8799/"
    file_spec = "Level3_ch1-short_x1d.fits"
    spec_tab = fits.open(os.path.join(path_spec, file_spec))["EXTRACT1D"].data
    spectrum = np.array([spec_tab['WAVELENGTH'], spec_tab['FLUX']])
    #matplotlib.use("tkAgg")

    for file in files:
        if file.endswith("masked_rate_corr.fits"):
            print(file)
            filename = os.path.basename(file)
            filename_save = filename.replace(".fits", "flat.fits")
            save_path = os.path.join(path, filename_save)
            hdu_science = fits.open(os.path.join(path, file))

            get_flat_old(hdu_science, hdu_science, save_path, snr_thresh=20, spectrum=None, N_continuum=50, N_D=15, N_finesse=20, mask_star_lines=True)