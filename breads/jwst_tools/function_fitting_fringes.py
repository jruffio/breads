import os
import inspect
import signal
import traceback
import warnings
import contextlib
import multiprocessing as mp
from concurrent.futures import (ProcessPoolExecutor, Future, wait,
                                FIRST_COMPLETED)
from concurrent.futures.process import BrokenProcessPool

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import medfilt
from numpy.linalg import LinAlgError
from tqdm import tqdm

from breads.jwst_tools.PositiveEtalonModel import PositiveEtalonModel

try:
    from BayesicFitting import SplinesModel, LevenbergMarquardtFitter, ConvergenceError
except ImportError:
    # these are optional dependencies for BREADS JWST support; OK to ignore them for
    # non-JWST BREADS usage
    pass

# Heavy imports (jwst, matplotlib, reduction_utils) are done on demand:
# every worker imports this module and shouldn't pay for them.
class _LazyPyplot:
    def __getattr__(self, name):
        import matplotlib.pyplot
        return getattr(matplotlib.pyplot, name)


plt = _LazyPyplot()

_BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)

# Detector columns processed for each of the two channel bands of a rate file
# (left half / right half of the detector).
LEFT_COLUMNS = range(5, 500)
RIGHT_COLUMNS = range(500, 1010)


# ====================================================================
# 1) Basic utilities (I/O, conversions, masking)
# ====================================================================

def micron_to_wavenumber(wave):
    return 1e4 / wave


def _load_rate_file(filename):
    """Loads SCI/DQ/ERR + primary header and computes the WCS (alpha, beta,
    lambda) for every pixel. The file is closed on return (no memmap left
    open)."""
    from jwst.assign_wcs import AssignWcsStep

    with fits.open(filename, memmap=False) as hdul:
        data = np.array(hdul['SCI'].data)
        DQ = np.array(hdul['DQ'].data)
        err = np.array(hdul['ERR'].data)
        prim_header = hdul[0].header.copy()

    model = AssignWcsStep.call(filename)
    ny, nx = data.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    alpha, beta, wave = model.meta.wcs.transform('detector', 'alpha_beta', xx, yy)

    return data, DQ, err, alpha, beta, wave, prim_header


def load_data_and_wcs(filename):
    data, DQ, err, alpha, beta, wave, _ = _load_rate_file(filename)
    return data, DQ, err, alpha, beta, wave


def get_optimal_number_nodes(channel_band):
    if channel_band in ('1A', '1B', '1C'):
        return 50, 15, 20
    elif channel_band in ('2A', '2B', '2C'):
        return 80, 20, 20
    elif channel_band in ('3A', '3B', '3C'):
        return 30, 20, 20
    elif channel_band in ('4A', '4B', '4C'):
        return 30, 15, 20
    else:
        raise ValueError(f'band {channel_band} not recognized')


def get_first_guess(channel_band, fast=False):
    grids = {
        '1A': ((3.35, 3.55, 0.02), (3.35, 3.55, 0.005)),
        '1B': ((3.35, 3.55, 0.02), (3.35, 3.55, 0.005)),
        '1C': ((3.3, 3.55, 0.02), (3.2, 3.8, 0.005)),
        '2A': ((3.3, 3.55, 0.02), (3.2, 3.8, 0.005)),
        '2B': ((3.3, 3.55, 0.02), (3.2, 3.8, 0.005)),
        '2C': ((3.3, 3.55, 0.02), (3.2, 3.8, 0.005)),
        '3A': ((3.1, 3.2, 0.02), (2.9, 3.2, 0.001)),
        '3B': ((2.8, 3.1, 0.02), (2.9, 3.1, 0.005)),
        '3C': ((2.8, 3.0, 0.02), (2.8, 3.0, 0.005)),
        '4A': ((2.8, 2.9, 0.02), (2.8, 2.9, 0.005)),
        '4B': ((2.8, 2.9, 0.02), (2.8, 2.9, 0.005)),
        '4C': ((2.8, 2.9, 0.02), (2.8, 2.9, 0.005)),
    }
    fast_range, slow_range = grids[channel_band]
    lo, hi, step = fast_range if fast else slow_range
    return np.arange(lo, hi, step) / 10


def masking_infinite(wave, data, err, beta, col_id):
    lamb = wave[2:1022, col_id]

    where_finite = np.where(np.isfinite(lamb))[0]
    lamb = lamb[where_finite]

    flux = data[2:1022, col_id][where_finite]
    beta = beta[2:1022, col_id][where_finite]
    err_c = err[2:1022, col_id][where_finite]

    return lamb, flux, beta, err_c


def mask_DQ(dq):
    # JWST bit definitions (only those actually used are named here)
    DO_NOT_USE = 1     # bit 0
    DEAD = 1024        # bit 10

    bad_flags = DEAD | DO_NOT_USE
    return (dq & bad_flags) != 0


def fill_wave_with_poly2(wave_col):
    """
    Fills NaNs in a spectral axis via quadratic extrapolation.
    Existing values are preserved.
    """
    wave_filled = wave_col.copy()

    valid = np.isfinite(wave_col)
    pix = np.arange(len(wave_col))

    coeffs = np.polyfit(pix[valid], wave_col[valid], deg=2)
    poly = np.polyval(coeffs, pix)

    wave_filled[~valid] = poly[~valid]

    return wave_filled


def identify_badpix(wave, data, err, beta, col_id, DQ=None, plot=False):
    if plot:
        import matplotlib
        matplotlib.use('tkAgg')

    # ==================================================
    # Full column extraction
    # ==================================================
    lamb_full = wave[:, col_id]
    flux_full = data[:, col_id]
    beta_full = beta[:, col_id]
    err_full = err[:, col_id]
    if DQ is not None:
        DQ = DQ[:, col_id]

    lamb_full = fill_wave_with_poly2(lamb_full)

    # 1 = good pixel, nan = bad pixel
    bad_pixels_col = np.ones(len(lamb_full), dtype=float)

    where_finite = np.where(np.isfinite(lamb_full))[0]

    lamb = lamb_full[where_finite]
    flux = flux_full[where_finite]
    beta = beta_full[where_finite]
    err_c = err_full[where_finite]
    if DQ is not None:
        DQ = DQ[where_finite]

    wnum = micron_to_wavenumber(lamb)

    wnum_0 = np.copy(wnum)
    flux_0 = np.copy(flux)

    good_mask = np.ones(len(wnum), dtype=bool)

    # ==================================================
    # Masking NaN and the end/start of the column if it is shared with multiple slices
    # ==================================================
    finite_mask = np.isfinite(wnum) & np.isfinite(flux)

    beta_valid = beta[finite_mask]
    beta_unique, counts = np.unique(
        beta_valid[np.isfinite(beta_valid)], return_counts=True
    )

    beta_mask = np.ones(len(beta), dtype=bool)
    if len(beta_unique) > 1 and np.min(counts) < 10:
        beta_kept = beta_unique[np.argmax(counts)]
        beta_mask = (beta == beta_kept)

    good_mask &= finite_mask & beta_mask

    # ==================================================
    # DQ masking (Saturation + Dead pixels)
    # ==================================================
    if DQ is not None:
        good_mask &= ~mask_DQ(DQ)

    # ==================================================
    # MAD filtering
    # ==================================================
    median = medfilt(flux[good_mask].astype(np.float64), kernel_size=21)
    residuals = flux[good_mask] - median

    local_mad = median_filter(np.abs(residuals), size=31, mode='reflect')
    local_robust_std = 1.4826 * local_mad

    floor = np.nanmedian(local_robust_std) * 0.1
    local_robust_std = np.maximum(local_robust_std, floor)

    mad_mask_local = np.abs(residuals) < 1.5 * local_robust_std

    idx_good = np.where(good_mask)[0]
    mad_mask_global = np.zeros(len(wnum), dtype=bool)
    mad_mask_global[idx_good[mad_mask_local]] = True

    good_mask &= mad_mask_global

    # ==================================================
    # Reprojecting bookkeeping back to the original indexing
    # ==================================================
    bad_pixels_col[:] = np.nan
    idx_original_good = where_finite[good_mask]
    bad_pixels_col[idx_original_good] = 1

    wnum_clean = wnum[good_mask]
    flux_clean = flux[good_mask]
    err_clean = err_c[good_mask]

    if plot:
        plt.title('Bad pixels masking')
        plt.plot(wnum_0, flux_0)
        plt.scatter(wnum_clean, flux_clean, color='red')
        plt.show()

    return wnum_clean, flux_clean, err_clean, bad_pixels_col


def identify_stellar_absorption(wnum, flux, err, spectrum_star, plot=False):
    """
    Identify the strongest stellar features to mask them from the Fabry-Perot transmittance modeling
    :param wnum: wavenumber
    :param flux: column flux
    :param err: column error (masked consistently with wnum/flux)
    :param spectrum_star: spectrum star flux
    :param plot: plot if True
    :return: wnum_masked, flux_masked, err_masked
    """

    spectrum_star_continuum = gaussian_filter(spectrum_star, sigma=3)
    absorption_ratio = spectrum_star / spectrum_star_continuum

    if plot:
        plt.title("spectrum")
        plt.plot(wnum, spectrum_star)
        plt.show()

        plt.title("Spectrum continuum normalized")
        plt.plot(wnum, absorption_ratio)
        plt.show()

        plt.title("Star absorption rejection")
        plt.scatter(wnum, flux, label="full spectrum")

    keep = (absorption_ratio < 1.01) & (absorption_ratio > 0.99)
    wnum_masked = wnum[keep]
    flux_masked = flux[keep]
    err_masked = err[keep]

    if plot:
        plt.scatter(wnum_masked, flux_masked, label="spectrum with feature masked")
        plt.legend()
        plt.show()

    return wnum_masked, flux_masked, err_masked


# ====================================================================
# Function to construct the Fabry-Perot Model (simple, double or even triple
# for the dichroics) with the BayesicFitting framework
# ====================================================================

def _build_cavity_term(wnum, N_finesse, N_D, amplitude=1):
    """Builds a single Fabry-Perot etalon term:
    amplitude * PositiveEtalon(finesse_spline, D_spline).

    `amplitude` is either a numeric constant (1 or 1.0 for a secondary /
    dichroic cavity), or a SplinesModel (the continuum) for the main cavity
    during the fit.
    """
    finesse = SplinesModel(nrknots=N_finesse, xrange=wnum)
    D = SplinesModel(nrknots=N_D, xrange=wnum)
    return PositiveEtalonModel(fixed={0: amplitude, 1: finesse, 2: D, 3: 0})


def get_model_continuum_frequency(N_nodes_continuum, N_nodes_D, wnum, flux, freq0):
    continuum = SplinesModel(nrknots=N_nodes_continuum, xrange=wnum)
    D = SplinesModel(nrknots=N_nodes_D, xrange=wnum)
    finesse = np.sqrt(0.15)
    mdl = PositiveEtalonModel(fixed={0: continuum, 1: finesse, 2: D, 3: 0})

    pars = [np.nanmean(flux)] + [0.] * (N_nodes_continuum + 1)
    pars += [freq0] + [0.0] * (N_nodes_D + 1)
    mdl.parameters = pars

    return mdl


def get_model_continuum_finesse_frequency(N_nodes_continuum, N_nodes_finesse, N_nodes_D,
                                           wnum, param_crop_continuum, param_crop_D):
    """Main cavity alone: continuum(spline) * cavity(finesse, D)."""
    continuum = SplinesModel(nrknots=N_nodes_continuum, xrange=wnum)
    mdl = _build_cavity_term(wnum, N_nodes_finesse, N_nodes_D, amplitude=continuum)

    pars_2 = np.array(([0.35] + [0.0] * (N_nodes_finesse + 1)))
    mdl.parameters = np.concatenate((param_crop_continuum, pars_2, param_crop_D))

    return mdl


def get_model_continuum_finesse_frequency_double_cavity(N_nodes_continuum, N_nodes_finesse, N_nodes_D,
                                                          wnum, param_crop_continuum, param_crop_finesse,
                                                          param_crop_D, freq_second_cavity=0.6):
    """Main cavity + second cavity (same finesse/D nrknots as the main cavity)."""
    continuum = SplinesModel(nrknots=N_nodes_continuum, xrange=wnum)
    cavity_1 = _build_cavity_term(wnum, N_nodes_finesse, N_nodes_D, amplitude=continuum)
    cavity_2 = _build_cavity_term(wnum, N_nodes_finesse, N_nodes_D, amplitude=1)
    mdl_double_cavity = cavity_1 * cavity_2

    pars_4 = np.array(([0.15] + [0.0] * (N_nodes_finesse + 1)))
    pars_5 = np.array(([freq_second_cavity] + [0.0] * (N_nodes_D + 1)))

    mdl_double_cavity.parameters = np.concatenate(
        (param_crop_continuum, param_crop_finesse, param_crop_D, pars_4, pars_5)
    )

    return mdl_double_cavity


def get_model_continuum_finesse_frequency_double_cavity_dichroics(
        N_nodes_continuum, N_nodes_finesse, N_nodes_D, wnum,
        param_crop_continuum, param_crop_finesse, param_crop_D,
        param_crop_finesse_second_cavity, param_crop_freq_second_cavity, freq_dichro=2.8):
    """Main cavity + second cavity + dichroic cavity (nrknots fixed to 20). Used for bands 2C, 3A, 3B and 3C.
    """
    continuum = SplinesModel(nrknots=N_nodes_continuum, xrange=wnum)
    cavity_1 = _build_cavity_term(wnum, N_nodes_finesse, N_nodes_D, amplitude=continuum)
    cavity_2 = _build_cavity_term(wnum, N_nodes_finesse, N_nodes_D, amplitude=1)
    cavity_dichro = _build_cavity_term(wnum, 20, 20, amplitude=1)
    mdl_double_cavity_dichroics = cavity_1 * cavity_2 * cavity_dichro

    pars_6 = np.array(([0.15] + [0.0] * (20 + 1)))
    pars_7 = np.array(([freq_dichro] + [0.0] * (20 + 1)))

    mdl_double_cavity_dichroics.parameters = np.concatenate((
        param_crop_continuum, param_crop_finesse, param_crop_D,
        param_crop_finesse_second_cavity, param_crop_freq_second_cavity,
        pars_6, pars_7,
    ))

    return mdl_double_cavity_dichroics

# ====================================================================
# #Functions to retrieve the column flat after the fitting iteration
# ====================================================================

def get_flat_result(N_nodes_finesse, N_nodes_D, wnum):
    """Fringe pattern of the main cavity alone (amplitude=1, no continuum)."""
    return _build_cavity_term(wnum, N_nodes_finesse, N_nodes_D, amplitude=1.0)


def get_flat_result_second_cavity(N_nodes_finesse, N_nodes_D, wnum):
    """Fringe pattern: main cavity + second cavity, no dichroics."""
    cavity_1 = _build_cavity_term(wnum, N_nodes_finesse, N_nodes_D, amplitude=1.0)
    cavity_2 = _build_cavity_term(wnum, N_nodes_finesse, N_nodes_D, amplitude=1.0)
    return cavity_1 * cavity_2


def get_flat_result_second_cavity_dichoics(N_nodes_finesse, N_nodes_D, wnum):
    """Fringe pattern: main cavity + second cavity + dichroics."""
    cavity_1 = _build_cavity_term(wnum, N_nodes_finesse, N_nodes_D, amplitude=1.0)
    cavity_2 = _build_cavity_term(wnum, N_nodes_finesse, N_nodes_D, amplitude=1.0)
    cavity_dichro = _build_cavity_term(wnum, 20, 20, amplitude=1.0)
    return cavity_1 * cavity_2 * cavity_dichro


def get_dichroics_fringes_only(wnum):
    """Isolated dichroics fringe pattern (diagnostic utility)."""
    return _build_cavity_term(wnum, 20, 20, amplitude=1)


def _build_full_fit_model(kind, Nc, Nf, ND, wnum):
    """Rebuilds the (unparametrized) structure of the FULL model used during
    the fit — continuum included — for a given `kind` value.

    Useful for reloading a saved fit and re-evaluating it with its original
    parameters (see plot_fit_vs_data below). Unlike get_flat_result*, the
    main cavity's amplitude here is a continuum spline (as during the fit),
    not a 1.0 constant — we want to compare the model directly to the raw
    data, not to the normalized flat.
    """
    continuum = SplinesModel(nrknots=Nc, xrange=wnum)
    if kind == 'basic':
        return _build_cavity_term(wnum, Nf, ND, amplitude=continuum)
    elif kind == 'second_cavity':
        cavity_1 = _build_cavity_term(wnum, Nf, ND, amplitude=continuum)
        cavity_2 = _build_cavity_term(wnum, Nf, ND, amplitude=1)
        return cavity_1 * cavity_2
    elif kind == 'second_cavity_dichroics':
        cavity_1 = _build_cavity_term(wnum, Nf, ND, amplitude=continuum)
        cavity_2 = _build_cavity_term(wnum, Nf, ND, amplitude=1)
        cavity_dichro = _build_cavity_term(wnum, 20, 20, amplitude=1)
        return cavity_1 * cavity_2 * cavity_dichro
    else:
        raise ValueError(f"Cannot rebuild a model for kind={kind!r}")


# ====================================================================
# Helpers function to mask the data, fit FP models, and search for chi2 minimum
# ====================================================================

# Converged fits take a few hundred iterations at most (~270 measured on band
# 3B); BayesicFitting's default (1000 x number of parameters, i.e. ~1e5) makes
# a diverging fit run for minutes before raising ConvergenceError.
LM_MAXITER = 2000


def _fit_lm(wnum, mdl, flux, weights=None, plot=False):
    """Single entry point for all LM fits in this module."""
    fitter = LevenbergMarquardtFitter(wnum, mdl)
    return fitter.fit(flux, weights=weights, plot=plot, maxiter=LM_MAXITER)


def _chi2(flux, mdl, wnum, param):
    """Compute the chi square for the continuum + fringe fit"""
    return np.sum((flux - mdl.result(wnum, param)) ** 2)


def _grid_search_frequency(wnum, flux, weights, freq_grid, build_model, plot=False):
    """Searches, over freq_grid, for the frequency that minimizes chi2.

    build_model(freq) must return a model ready to be fitted for that frequency.
    Grid points whose fit fails (LinAlgError, or ConvergenceError when the
    model diverges to inf/nan) or gives a non-finite chi2 are ignored.
    Returns (best_freq, best_param, best_model, chi2_array).
    """
    chi2 = np.full(len(freq_grid), np.inf)
    best_idx, best_param, best_model = None, None, None
    for k, freq in enumerate(freq_grid):
        mdl = build_model(freq)
        try:
            param = _fit_lm(wnum, mdl, flux, weights=weights, plot=plot)
        except (LinAlgError, ConvergenceError):
            continue
        chi2_k = _chi2(flux, mdl, wnum, param)
        if not np.isfinite(chi2_k):
            continue
        chi2[k] = chi2_k
        # strict '<': on ties keep the first one, like np.argmin
        if best_idx is None or chi2[k] < chi2[best_idx]:
            best_idx, best_param, best_model = k, param, mdl

    if best_idx is None:
        raise LinAlgError("All fits of the frequency grid failed")

    return freq_grid[best_idx], best_param, best_model, chi2


def _prepare_column(data, wave, err, beta, col_id, channel_band, snr_thresh,
                     N_continuum, N_D, N_finesse, DQ=None):
    """Extraction + cleaning of one column's data, and computation of the
    number of spline nodes adjusted to the density of valid points.

    Returns None if the SNR is insufficient or if the column has no valid
    wavelength (the flat stays 1 everywhere).
    """
    lamb_axis = wave[:, col_id]
    if not np.any(np.isfinite(lamb_axis)):
        return None
    wnum_axis = micron_to_wavenumber(lamb_axis)

    SNR_mean = np.nanmedian(data[:, col_id] / err[:, col_id])
    if SNR_mean <= snr_thresh:
        return None

    wnum, flux, err_c, bad_pixel_col = identify_badpix(wave, data, err, beta, col_id, DQ=DQ)

    if N_continuum is None or N_D is None or N_finesse is None:
        N_continuum, N_D, N_finesse = get_optimal_number_nodes(channel_band)

    r = len(wnum) / 970
    N_continuum = max(5, int(N_continuum * r))
    N_D = max(3, int(N_D * r))
    N_finesse = max(3, int(N_finesse * r))

    return dict(
        wnum=wnum, flux=flux, err_c=err_c, bad_pixel_col=bad_pixel_col,
        N_continuum=N_continuum, N_D=N_D, N_finesse=N_finesse,
        wnum_axis=wnum_axis, lamb_axis=lamb_axis,
    )


def _mask_stellar_lines_if_needed(wnum, flux, err, spectrum, mask_star_lines, plot=False):
    """Masks the strongest stellar lines if a stellar spectrum is provided.
    Returns (wnum, flux, err) — err is masked consistently with wnum/flux,
    so that the weights computed afterward have the correct length.
    """
    if spectrum is None:
        return wnum, flux, err
    lamb_star, spectrum_star = spectrum
    f_interp = interp1d(lamb_star, spectrum_star, bounds_error=False, fill_value=1)
    spectrum_star_grid = f_interp(1000 / wnum)
    if mask_star_lines:
        wnum, flux, err = identify_stellar_absorption(wnum, flux, err, spectrum_star_grid, plot=plot)
    return wnum, flux, err


def _plot_debug_model(wnum, param, Nc, Nf, ND):
    """Plotting spline functions of the fitted etalon width and the finesse"""
    param_finesse = param[Nc + 2:Nc + Nf + 4]
    param_D = param[Nc + Nf + 4:Nc + ND + Nf + 6]

    plt.title("Best D")
    D = SplinesModel(nrknots=ND, xrange=wnum)
    plt.plot(wnum, D.result(wnum, param_D), label=f"# nodes: {Nf}")
    plt.legend()
    plt.show()

    plt.title("Best finesse")
    F = SplinesModel(nrknots=Nf, xrange=wnum)
    plt.plot(wnum, F.result(wnum, param_finesse), label=f"# nodes: {Nf}")
    plt.legend()
    plt.show()


# ====================================================================
# Fit the FP model to one single column with different model possibilities with respect to the band
# ====================================================================

def fit_FP_bayesian_core(data, wave, beta, err, channel_band, col_id, snr_thresh,
                          plot=False, plot_model=False, spectrum=None,
                          N_continuum=50, N_D=15, N_finesse=10,
                          mask_star_lines=True, fast=True, DQ=None,
                          use_weights=False,
                          second_cavity_grid=None, dichroics_grid=None):
    """Generic pipeline for fitting a Fabry-Perot etalon fringe pattern on one column.

    Valid combinations (the only physically sensible ones):
      - main cavity alone                         (second_cavity_grid=None, dichroics_grid=None)
      - main cavity + second cavity               (second_cavity_grid provided, dichroics_grid=None)
      - main cavity + second cavity + dichroics    (both grids provided)

    "main cavity + dichroics" WITHOUT a second cavity is NOT valid: if
    dichroics_grid is provided while second_cavity_grid is not, the grid is
    reinterpreted as a search for a SECOND cavity.

    In addition to the usual flat/continuum/bad-pixels, returns a 5th
    element `fit_info`: a dict describing the fit actually performed for
    this column (kind, number of nodes, data used, fitted parameters) —
    meant to be saved in the output file's FIT_PARAMS extension and
    reloaded later via plot_fit_vs_data.
    """
    flat = np.ones_like(wave[:, col_id])
    D_est = np.full_like(wave[:, col_id], np.nan)
    continuum = np.full_like(wave[:, col_id], np.nan)
    bad_pixel_col = None

    prep = _prepare_column(data, wave, err, beta, col_id, channel_band, snr_thresh,
                            N_continuum, N_D, N_finesse, DQ=DQ)
    if prep is None:
        flat[flat < 0.1] = 1
        fit_info = dict(
            kind='skipped', band=channel_band, Nc=None, ND=None, Nf=None,
            wnum=np.array([]), flux=np.array([]), err=np.array([]),
            param=np.array([]),
        )
        return flat, D_est, continuum, bad_pixel_col, fit_info

    wnum, flux = prep['wnum'], prep['flux']
    err_c = prep['err_c']
    bad_pixel_col = prep['bad_pixel_col']
    Nc, ND, Nf = prep['N_continuum'], prep['N_D'], prep['N_finesse']
    wnum_axis = prep['wnum_axis']

    wnum, flux, err_c = _mask_stellar_lines_if_needed(
        wnum, flux, err_c, spectrum, mask_star_lines, plot=plot
    )
    weights = 1 / err_c ** 2 if use_weights else None

    # ---- Stage 1: continuum + D, search for the main etalon's frequency ----
    freq_grid_1 = get_first_guess(channel_band, fast=fast)
    _, param, _, chi2 = _grid_search_frequency(
        wnum, flux, weights, freq_grid_1,
        build_model=lambda f: get_model_continuum_frequency(Nc, ND, wnum, flux, f),
        plot=plot,
    )
    if plot:
        plt.title("chi2")
        plt.plot(freq_grid_1, chi2)
        plt.show()

    param_continuum = param[:Nc + 2]
    param_D = param[Nc + 2:Nc + ND + 4]

    # ---- Stage 2: add finesse, full refit ----
    mdl = get_model_continuum_finesse_frequency(Nc, Nf, ND, wnum, param_continuum, param_D)
    param = _fit_lm(wnum, mdl, flux, weights=weights, plot=False)

    param_continuum = param[:Nc + 2]
    param_finesse = param[Nc + 2:Nc + Nf + 4]
    param_D = param[Nc + Nf + 4:Nc + ND + Nf + 6]

    if plot_model:
        _plot_debug_model(wnum, param, Nc, Nf, ND)

    result_param = param
    result_kind = 'basic'

    # ---- Stage 3 (optional): second cavity ----
    # If every fit of a grid fails, the column keeps the result of the previous stage.
    param_sc = None
    if second_cavity_grid is not None:
        try:
            _, param_sc, _, _ = _grid_search_frequency(
                wnum, flux, weights, second_cavity_grid,
                build_model=lambda f: get_model_continuum_finesse_frequency_double_cavity(
                    Nc, Nf, ND, wnum, param_continuum, param_finesse, param_D, freq_second_cavity=f),
                plot=plot,
            )
        except LinAlgError:
            warnings.warn(f"Column {col_id} (band {channel_band}): second cavity fit failed, "
                          f"keeping the main cavity fit.")
            dichroics_grid = None
    if param_sc is not None:
        param_continuum = param_sc[:Nc + 2]
        param_finesse = param_sc[Nc + 2:Nc + Nf + 4]
        param_D = param_sc[Nc + Nf + 4:Nc + ND + Nf + 6]
        param_finesse_sc = param_sc[Nc + ND + Nf + 6:Nc + ND + 2 * Nf + 8]
        param_freq_sc = param_sc[Nc + ND + 2 * Nf + 8:Nc + 2 * ND + 2 * Nf + 10]
        result_param = param_sc
        result_kind = 'second_cavity'

    # ---- Stage 4 (optional): dichroics ----
    if dichroics_grid is not None:
        if result_kind == 'second_cavity':
            # Correct case: main cavity + second cavity + dichroics.
            def build(f):
                return get_model_continuum_finesse_frequency_double_cavity_dichroics(
                    Nc, Nf, ND, wnum, param_continuum, param_finesse, param_D,
                    param_finesse_sc, param_freq_sc, freq_dichro=f)
            next_kind = 'second_cavity_dichroics'
        else:
            # dichroics_grid provided without second_cavity_grid: invalid
            # configuration. The grid is reinterpreted as a search for a
            # SECOND cavity (double_cavity model), never as a dichroics
            # model applied directly to the main cavity alone.
            def build(f):
                return get_model_continuum_finesse_frequency_double_cavity(
                    Nc, Nf, ND, wnum, param_continuum, param_finesse, param_D,
                    freq_second_cavity=f)
            next_kind = 'second_cavity'

        try:
            _, result_param, _, _ = _grid_search_frequency(
                wnum, flux, weights, dichroics_grid, build_model=build, plot=plot,
            )
            result_kind = next_kind
        except LinAlgError:
            warnings.warn(f"Column {col_id} (band {channel_band}): {next_kind} fit failed, "
                          f"keeping the {result_kind} fit.")

    # ---- Building the final flat according to the enabled stages ----
    param_continuum_final = result_param[:Nc + 2]
    param_rest = result_param[Nc + 2:]

    flat_builders = {
        'basic': lambda: get_flat_result(Nf, ND, wnum),
        'second_cavity': lambda: get_flat_result_second_cavity(Nf, ND, wnum),
        'second_cavity_dichroics': lambda: get_flat_result_second_cavity_dichoics(Nf, ND, wnum),
    }
    mdl_flat = flat_builders[result_kind]()
    flat = mdl_flat.result(wnum_axis, param_rest)

    mdl_continuum = SplinesModel(nrknots=Nc, xrange=wnum)
    continuum = mdl_continuum.result(wnum_axis, param_continuum_final)

    flat[flat < 0.1] = 1  # hard thresholding

    fit_info = dict(
        kind=result_kind, band=channel_band, Nc=Nc, ND=ND, Nf=Nf,
        wnum=wnum, flux=flux, err=err_c,
        param=np.asarray(result_param, dtype=np.float64),
    )

    return flat, D_est, continuum, bad_pixel_col, fit_info

# ====================================================================
# ---- Fit function for the different bands ----
# ====================================================================

def fit_FP_bayesian(data, wave, beta, err, channel_band, col_id, snr_thresh, **kwargs):
    """Channel 1 bands: no dichroics, no second cavity."""
    return fit_FP_bayesian_core(data, wave, beta, err, channel_band, col_id, snr_thresh,
                                 use_weights=True, **kwargs)


def fit_FP_bayesian_ch2(data, wave, beta, err, channel_band, col_id, snr_thresh, **kwargs):
    """Bands 2A / 2B: first cavity + second cavity."""
    return fit_FP_bayesian_core(
        data, wave, beta, err, channel_band, col_id, snr_thresh,
        use_weights=True,
        second_cavity_grid=np.arange(3.1, 3.3, 0.01) / 10,
        **kwargs,
    )


def fit_FP_bayesian_ch2c(data, wave, beta, err, channel_band, col_id, snr_thresh, **kwargs):
    """Band 2C: first cavity + second cavity + dichroics."""
    return fit_FP_bayesian_core(
        data, wave, beta, err, channel_band, col_id, snr_thresh,
        use_weights=True,
        second_cavity_grid=np.arange(3.1, 3.3, 0.03) / 10,
        dichroics_grid=np.arange(2.65, 2.85, 0.01),
        **kwargs,
    )



def fit_FP_bayesian_ch3(data, wave, beta, err, channel_band, col_id, snr_thresh, **kwargs):
    """Bands 3A, 3B, 3C: first cavity + second cavity + dichroics."""
    return fit_FP_bayesian_core(
        data, wave, beta, err, channel_band, col_id, snr_thresh,
        use_weights=True,
        second_cavity_grid=np.arange(0.26, 0.30, 0.001),
        dichroics_grid=np.arange(2.65, 2.85, 0.01),
        **kwargs,
    )


# ====================================================================
# Get the function for the required band
# ====================================================================

FIT_DISPATCH = {
    '1A': fit_FP_bayesian,
    '1B': fit_FP_bayesian,
    '1C': fit_FP_bayesian,
    '2A': fit_FP_bayesian_ch2,
    '2B': fit_FP_bayesian_ch2,
    '2C': fit_FP_bayesian_ch2c,
    '3A': fit_FP_bayesian_ch3,
    '3B': fit_FP_bayesian_ch3,
    '3C': fit_FP_bayesian_ch3,
}

# Human-readable label of the model used by each fit function, for the FITS
# header metadata (BAND1MDL / BAND2MDL).
FIT_MODEL_LABEL = {
    fit_FP_bayesian: 'first cavity only',
    fit_FP_bayesian_ch2: 'first cavity + second cavity',
    fit_FP_bayesian_ch2c: 'first cavity + second cavity + dichroics',
    fit_FP_bayesian_ch3: 'first cavity + second cavity + dichroics',
}


def get_fit_function(channel_band):
    """Return the fit function optimized for the given channel band."""
    return FIT_DISPATCH.get(channel_band, fit_FP_bayesian_ch3)


def get_fit_model_label(channel_band):
    """Human-readable label of the fit model used for this channel band
    ('first cavity only', 'first cavity + second cavity', or
    'first cavity + second cavity + dichroics') — used for the FITS header
    metadata.
    """
    return FIT_MODEL_LABEL.get(get_fit_function(channel_band), 'unknown')


def _call_fit(func, **kwargs):
    """Calls `func`, passing only the kwargs it actually accepts.

    Avoids having to write an if/elif block for each different signature
    (some functions don't have `spectrum`, `mask_star_lines`, or `DQ`).
    Functions taking **kwargs receive everything: the previous version
    filtered on the explicit parameters only, which silently dropped DQ,
    spectrum, N_*, fast and mask_star_lines for all fit_FP_bayesian* calls.
    """
    params = inspect.signature(func).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return func(**kwargs)
    return func(**{k: v for k, v in kwargs.items() if k in params})


def _empty_fit_info(kind, channel_band):
    return dict(
        kind=kind, band=channel_band, Nc=None, ND=None, Nf=None,
        wnum=np.array([]), flux=np.array([]), err=np.array([]),
        param=np.array([]),
    )


# ====================================================================
# 6) Worker (runs in the child processes)
# ====================================================================

# Options shared by all the tasks of a run, set once per worker by the
# initializer (avoids re-pickling `spectrum` for every task).
_WORKER_OPTIONS = None


def _limit_blas_threads():
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    threadpool_limits(1)


def _init_worker(options, limit_threads=True):
    global _WORKER_OPTIONS
    _WORKER_OPTIONS = options
    if limit_threads:
        # Ctrl-C is handled by the parent, which stops the pool cleanly.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        _limit_blas_threads()


def _fit_column_task(channel_band, col_id, columns):
    """Fits one column. `columns` = (data, wave, beta, err, DQ), each of shape
    (ny, 1): only the relevant column is sent to the worker.

    Never raises a "regular" exception: a fit error is returned as a
    fit_info of kind='error' together with the traceback.
    """
    data, wave, beta, err, DQ = columns
    opts = _WORKER_OPTIONS
    try:
        f, _, c, bp_col, fit_info = _call_fit(
            get_fit_function(channel_band),
            data=data, wave=wave, beta=beta, err=err, DQ=DQ,
            channel_band=channel_band, col_id=0, plot=False, **opts,
        )
        return col_id, f, c, bp_col, fit_info, None
    except Exception:
        return col_id, None, None, None, _empty_fit_info('error', channel_band), traceback.format_exc()


# ====================================================================
# 7) Multiprocessing orchestration (parent process)
# ====================================================================

def _available_cpus():
    if hasattr(os, 'process_cpu_count'):        # Python >= 3.13
        n = os.process_cpu_count()
    elif hasattr(os, 'sched_getaffinity'):      # Linux
        n = len(os.sched_getaffinity(0))
    else:
        n = os.cpu_count()
    return max(1, n or 1)


@contextlib.contextmanager
def _single_threaded_children_env():
    """The BLAS variables must be in the environment BEFORE the workers
    import numpy: they are set for the lifetime of the pool ('spawn' workers
    are started on demand) and restored afterwards."""
    saved = {k: os.environ.get(k) for k in _BLAS_THREAD_ENV_VARS}
    os.environ.update({k: "1" for k in _BLAS_THREAD_ENV_VARS})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _InlineExecutor:
    """Sequential executor in the current process (n_procs=1): same code path
    as the pool, handy for debugging / profiling."""

    def submit(self, fn, *args):
        fut = Future()
        try:
            fut.set_result(fn(*args))
        except BaseException as e:
            fut.set_exception(e)
        return fut

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def _split_band(combined_channel_band):
    """'12A' -> ('1A', '2A') ; '34A' -> ('4A', '3A'): channel band of columns
    5-499, then channel band of columns 500-1009."""
    if combined_channel_band[0] == '1':
        return combined_channel_band[0] + combined_channel_band[2], combined_channel_band[1] + combined_channel_band[2]
    return combined_channel_band[1] + combined_channel_band[2], combined_channel_band[0] + combined_channel_band[2]


class _FileJob:
    """State of a rate file being processed: accumulates the column results
    and writes the *_fit_flat.fits file once everything has been received."""

    def __init__(self, save_filename, prim_header, shape, channel_band_left, channel_band_right,
                 channel_band_left_fitted, channel_band_right_fitted=True):
        self.save_filename = save_filename
        self.prim_header = prim_header
        self.channel_band_left = channel_band_left
        self.channel_band_right = channel_band_right
        self.channel_band_left_fitted = channel_band_left_fitted
        self.channel_band_right_fitted = channel_band_right_fitted

        self.flat = np.ones(shape)
        self.continuum = np.full(shape, np.nan)
        self.bad_pixels = np.zeros(shape)
        self.fit_records = []
        self.n_errors = 0
        self.remaining = 0

    def tasks(self):
        if self.channel_band_left_fitted:
            for col_id in LEFT_COLUMNS:
                yield self.channel_band_left, col_id
        if self.channel_band_right_fitted:
            for col_id in RIGHT_COLUMNS:
                yield self.channel_band_right, col_id

    def add(self, col_id, f, c, bp_col, fit_info):
        if f is not None:
            self.flat[:, col_id] = f
            self.continuum[:, col_id] = c
            self.bad_pixels[:, col_id] = bp_col
        if fit_info is not None:
            record = dict(fit_info)
            record['col_id'] = col_id
            self.fit_records.append(record)
            if fit_info['kind'] == 'error':
                self.n_errors += 1
        self.remaining -= 1

    def write(self, overwrite):
        primary_hdu = fits.PrimaryHDU(header=self.prim_header)
        hdr = primary_hdu.header

        # ---- Metadata: which fit model was used for each band ----
        if self.channel_band_left_fitted:
            hdr['BAND1'] = (self.channel_band_left, 'Channel band fit for columns 5-499')
            hdr['BAND1MDL'] = (get_fit_model_label(self.channel_band_left), 'Fabry-Perot model used for BAND1')
        else:
            hdr['BAND1'] = (self.channel_band_left, 'Channel band for columns 5-499 (not fitted)')
            hdr['BAND1MDL'] = ('skipped', 'Fringe estimation skipped for BAND1')
        if self.channel_band_right_fitted:
            hdr['BAND2'] = (self.channel_band_right, 'Channel band fit for columns 500-1009')
            hdr['BAND2MDL'] = (get_fit_model_label(self.channel_band_right), 'Fabry-Perot model used for BAND2')
        else:
            hdr['BAND2'] = (self.channel_band_right, 'Channel band for columns 500-1009 (not fitted)')
            hdr['BAND2MDL'] = ('skipped', 'Fringe estimation skipped for BAND2')

        hdul_list = [
            primary_hdu,
            fits.ImageHDU(data=self.flat, name='FLAT'),
            fits.ImageHDU(data=self.flat, name='FLAT_EXTENDED'),
            fits.ImageHDU(data=self.continuum, name='CONTINUUM'),
            fits.ImageHDU(data=self.bad_pixels, name='BAD_PIX'),
        ]

        fit_params_hdu = _build_fit_params_hdu(self.fit_records)
        if fit_params_hdu is not None:
            hdul_list.append(fit_params_hdu)

        fits.HDUList(hdul_list).writeto(self.save_filename, overwrite=overwrite)


def _build_fit_params_hdu(fit_records):
    """Builds the FIT_PARAMS binary table extension from the fit metadata
    accumulated for all processed columns.

    One row per column: COL_ID, BAND, KIND ('basic' / 'second_cavity' /
    'second_cavity_dichroics' / 'skipped' / 'error'), NC/ND/NF (number of
    spline nodes used), and four VARIABLE-length columns (WNUM, FLUX, ERR,
    PARAM) since the number of valid points and spline nodes differs from
    one column to another.

    Returns None if `fit_records` is empty (nothing to save).
    """
    if not fit_records:
        return None

    fit_records = sorted(fit_records, key=lambda r: r['col_id'])

    col_ids = np.array([r['col_id'] for r in fit_records], dtype=np.int32)
    bands = np.array([r.get('band') or '' for r in fit_records], dtype='U4')
    kinds = np.array([r['kind'] for r in fit_records], dtype='U24')
    Ncs = np.array([r['Nc'] if r['Nc'] is not None else -1 for r in fit_records], dtype=np.int32)
    NDs = np.array([r['ND'] if r['ND'] is not None else -1 for r in fit_records], dtype=np.int32)
    Nfs = np.array([r['Nf'] if r['Nf'] is not None else -1 for r in fit_records], dtype=np.int32)

    def _obj_col(key):
        return np.array([np.asarray(r[key], dtype=np.float64) for r in fit_records], dtype=object)

    cols = [
        fits.Column(name='COL_ID', format='J', array=col_ids),
        fits.Column(name='BAND', format='4A', array=bands),
        fits.Column(name='KIND', format='24A', array=kinds),
        fits.Column(name='NC', format='J', array=Ncs),
        fits.Column(name='ND', format='J', array=NDs),
        fits.Column(name='NF', format='J', array=Nfs),
        # Variable-length columns: a 1D array whose size is specific to each
        # row (standard "heap" mechanism of FITS binary tables).
        fits.Column(name='WNUM', format='PD()', array=_obj_col('wnum')),
        fits.Column(name='FLUX', format='PD()', array=_obj_col('flux')),
        fits.Column(name='ERR', format='PD()', array=_obj_col('err')),
        fits.Column(name='PARAM', format='PD()', array=_obj_col('param')),
    ]
    return fits.BinTableHDU.from_columns(cols, name='FIT_PARAMS')


def _iter_files_to_fit(uncaldir, targetname, list_bands, bkg_sub, skip_4, overwrite):
    """Yields (combined_channel_band, rate_file, save_filename, channel_band_left_fitted) for each
    file to process, skipping those already done if overwrite=False."""
    from breads.jwst_reduction.reduction_utils import find_files_to_process

    save_root = os.getenv("FLAT_PATH")
    if save_root is None:
        raise ValueError("No FLAT_PATH specified to save the fringe flat")

    stage_dir = 'stage1_sub_bkg' if bkg_sub else 'stage1'
    for combined_channel_band in list_bands:
        input_path = os.path.join(uncaldir, targetname, combined_channel_band, stage_dir)
        save_path = os.path.join(save_root, combined_channel_band)
        os.makedirs(save_path, exist_ok=True)

        channel_band_left_fitted = not (combined_channel_band[1] == '4' and skip_4)
        if not channel_band_left_fitted:
            print(f'Band {combined_channel_band}: skipping channel 4 fringe estimation')

        for file in find_files_to_process(input_path, filetype='rate.fits'):
            filename = os.path.basename(file)
            save_filename = os.path.join(save_path, filename.replace('rate.fits', 'fit_flat.fits'))
            if os.path.exists(save_filename) and not overwrite:
                print(f"Flat for {filename} already created, skipping")
                continue
            yield combined_channel_band, os.path.join(input_path, file), save_filename, channel_band_left_fitted


def _column_slices(arrays, col_id):
    """Contiguous (ny, 1) copy of column col_id of each 2D array."""
    return tuple(
        None if a is None else np.ascontiguousarray(a[:, col_id:col_id + 1])
        for a in arrays
    )


def _collect_results(pending, active_jobs, pbar, overwrite, keep_job=None):
    """Gathers the results of finished tasks and writes the completed files.
    Returns as soon as no task of a job other than `keep_job` is still
    running (keep_job=None: wait for everything)."""
    def must_wait():
        return any(job.remaining > 0 for job in active_jobs if job is not keep_job)

    while pending and must_wait():
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for fut in done:
            job, channel_band = pending.pop(fut)
            try:
                col_id, f, c, bp_col, fit_info, tb = fut.result()
            except BrokenProcessPool as e:
                raise RuntimeError(
                    "A worker died abruptly (segfault or out of memory?). "
                    "Retry with fewer processes (n_procs), or n_procs=1 to debug."
                ) from e
            if tb is not None:
                tqdm.write(f"[{os.path.basename(job.save_filename)}] error on column {col_id} "
                           f"(band {channel_band}):\n{tb}")
            job.add(col_id, f, c, bp_col, fit_info)
            pbar.update(1)

        for job in [j for j in active_jobs if j.remaining == 0]:
            job.write(overwrite)
            active_jobs.remove(job)
            msg = f"==> Estimated fringe flat written to {job.save_filename}"
            if job.n_errors:
                msg += f" ({job.n_errors} columns failed)"
            tqdm.write(msg)


def get_flat_parallelized(uncaldir, targetname, list_bands=None, bkg_sub=False,
                           snr_thresh=20, spectrum=None, N_continuum=None,
                           N_D=None, N_finesse=None, mask_star_lines=True,
                           fast=False, skip_4=True, overwrite=True,
                           n_procs=None, chunksize=None, start_method='spawn', channel_bands=None):
    """Estimates the fringe flat of each *rate.fits file by fitting a
    Fabry-Perot model column by column, in parallel, and writes it to
    $FLAT_PATH/<combined_channel_band>/<...>_fit_flat.fits.

    Parameters
    ----------
    channel_bands : list of str or None
        If not None, only fit the columns of these channel bands (e.g. ['1A']); the flat is left to 1 in the
        columns of the other half of the detector. None (default) fits both halves (except channel 4 if skip_4).
    n_procs : int or None
        Number of processes. None = number of CPUs actually available to this
        process (affinity / cgroups honored). 1 = sequential execution without
        multiprocessing (debugging).
    chunksize : ignored
        Kept for backward compatibility: every column is now an individual
        task (better load balancing, the cost per column varies enormously
        with the band and the SNR).
    start_method : {'spawn', 'forkserver', 'fork'}
        Start method of the workers. 'spawn' (default) is the only one that is
        safe everywhere; it requires the calling script to be guarded by
        `if __name__ == "__main__":`. Independent of mp.set_start_method.
    """
    list_bands = list_bands or ['12A', '12B', '12C', '34A', '34B', '34C']

    options = dict(
        snr_thresh=snr_thresh, spectrum=spectrum,
        N_continuum=N_continuum, N_D=N_D, N_finesse=N_finesse,
        mask_star_lines=mask_star_lines, fast=fast,
    )

    n_procs = n_procs or _available_cpus()
    # Never more than ~1000 columns in flight (2 files at most).
    n_procs = max(1, min(n_procs, len(LEFT_COLUMNS) + len(RIGHT_COLUMNS)))

    with contextlib.ExitStack() as stack:
        if n_procs == 1:
            _init_worker(options, limit_threads=False)
            executor = _InlineExecutor()
        else:
            stack.enter_context(_single_threaded_children_env())
            executor = ProcessPoolExecutor(
                max_workers=n_procs,
                mp_context=mp.get_context(start_method),
                initializer=_init_worker,
                initargs=(options,),
            )
        pbar = stack.enter_context(tqdm(total=0, desc="Fitting columns", unit="col"))

        pending = {}       # future -> (job, channel_band)
        active_jobs = []
        try:
            for combined_channel_band, rate_file, save_filename, channel_band_left_fitted in _iter_files_to_fit(
                    uncaldir, targetname, list_bands, bkg_sub, skip_4, overwrite):
                pbar.set_description(f"Band {combined_channel_band} - {os.path.basename(rate_file)}")

                # Loading + WCS in the parent: happens while the workers are
                # still fitting the previous file.
                data, DQ, err, _, beta, wave, prim_header = _load_rate_file(rate_file)
                channel_band_left, channel_band_right = _split_band(combined_channel_band)
                job = _FileJob(save_filename, prim_header, data.shape,
                               channel_band_left, channel_band_right,
                               channel_band_left_fitted and (channel_bands is None or channel_band_left in channel_bands),
                               channel_bands is None or channel_band_right in channel_bands)
                active_jobs.append(job)

                tasks = list(job.tasks())
                job.remaining = len(tasks)
                pbar.total += len(tasks)
                pbar.refresh()

                arrays = (data, wave, beta, err, DQ)
                for channel_band, col_id in tasks:
                    fut = executor.submit(_fit_column_task, channel_band, col_id,
                                          _column_slices(arrays, col_id))
                    pending[fut] = (job, channel_band)
                del data, DQ, err, beta, wave, arrays

                # At most two files in flight: finish the previous one before
                # loading the next (bounds the parent's memory).
                _collect_results(pending, active_jobs, pbar, overwrite, keep_job=job)

            _collect_results(pending, active_jobs, pbar, overwrite)
        except BaseException:
            # Immediate stop (Ctrl-C, dead worker, write error...) without
            # waiting for the running fits to finish.
            if hasattr(executor, 'terminate_workers'):      # Python >= 3.14
                executor.terminate_workers()
            else:
                procs = list((getattr(executor, '_processes', None) or {}).values())
                executor.shutdown(wait=False, cancel_futures=True)
                for p in procs:
                    p.terminate()
            raise
        executor.shutdown(wait=True)


# ====================================================================
# Reloading saved fit parameters to plot the fitting accuracy afterwards
# ====================================================================

def load_fit_params(fits_path, col_id, ext='FIT_PARAMS'):
    """Reloads, from a *_fit_flat.fits file produced by get_flat_parallelized,
    the fit metadata saved for a given column.

    Returns a dict with the keys band, kind, Nc, ND, Nf, wnum, flux, err,
    param — or None if the column is not present in the file (for example
    if channel 4 was skipped for the corresponding band).
    """
    with fits.open(fits_path) as hdul:
        if ext not in hdul:
            raise KeyError(f"Missing extension '{ext}' in {fits_path}")
        table = hdul[ext].data
        rows = np.where(table['COL_ID'] == col_id)[0]
        if len(rows) == 0:
            return None
        row = table[rows[0]]

        return dict(
            col_id=int(row['COL_ID']),
            band=str(row['BAND']).strip(),
            kind=str(row['KIND']).strip(),
            Nc=int(row['NC']),
            ND=int(row['ND']),
            Nf=int(row['NF']),
            wnum=np.asarray(row['WNUM'], dtype=np.float64),
            flux=np.asarray(row['FLUX'], dtype=np.float64),
            err=np.asarray(row['ERR'], dtype=np.float64),
            param=np.asarray(row['PARAM'], dtype=np.float64),
        )


def plot_fit_vs_data(fits_path, col_id, ext='FIT_PARAMS', show=True):
    """Reloads the saved fit for `col_id` and plots the fitted Fabry-Perot
    model overlaid on the data (flux vs wavenumber), with residuals below.

    Returns the matplotlib figure (or None if the column wasn't fitted —
    SNR too low, channel skipped, or an error occurred during the fit).
    """
    info = load_fit_params(fits_path, col_id, ext=ext)
    if info is None:
        print(f"Column {col_id} not found in {fits_path}.")
        return None

    if info['kind'] in ('skipped', 'error') or len(info['wnum']) == 0:
        print(f"Column {col_id} (band {info['band']}): {info['kind']}, nothing to plot.")
        return None

    wnum, flux, param = info['wnum'], info['flux'], info['param']
    Nc, Nf, ND, kind = info['Nc'], info['Nf'], info['ND'], info['kind']

    mdl = _build_full_fit_model(kind, Nc, Nf, ND, wnum)
    mdl_continuum = SplinesModel(nrknots=Nc, xrange=wnum)

    order = np.argsort(wnum)
    wnum_sorted = wnum[order]
    wnum_fine = np.linspace(wnum_sorted.min(), wnum_sorted.max(), 5 * len(wnum_sorted))
    model_fine = mdl.result(wnum_fine, param)
    model_at_data = mdl.result(wnum, param)
    param_continuum = param[:Nc + 2]
    model_continuum = mdl_continuum.result(wnum_fine, param_continuum)

    knots = getattr(mdl_continuum, 'knots', None)
    knots_y = mdl_continuum.result(np.asarray(knots), param_continuum)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 6), gridspec_kw={'height_ratios': [3, 1]}
    )

    ax1.scatter(wnum, flux, s=8, color='k', label='Data')
    ax1.plot(wnum_fine, model_fine, color='C1', label=f"Model ({kind})")
    ax1.plot(wnum_fine, model_continuum, color='r', label="Column continuum")
    ax1.scatter(knots, knots_y, color='C2', marker='x', s=40, zorder=5, label='Nodes continuum')
    ax1.set_ylabel('Flux')
    ax1.set_title(f"Column #{col_id} — band {info['band']} — {kind}")
    ax1.legend()

    ax2.scatter(wnum, flux - model_at_data, s=8, color='k')
    ax2.axhline(0, color='C1', lw=1)
    ax2.set_xlabel("Wavenumber [cm-1]")
    ax2.set_ylabel('Residual')

    plt.tight_layout()
    if show:
        plt.show()

    return fig


def plot_two_cavity_patterns(fits_path, col_id, ext='FIT_PARAMS', show=True):
    """For a column fitted with at least two cavities (kind='second_cavity'
    or 'second_cavity_dichroics'), plots the main cavity's fringe pattern
    and the second cavity's fringe pattern separately — even if dichroics
    were subsequently fitted, they are not counted as one of the two
    cavities here.

    Returns the matplotlib figure, or None if the column has no second
    cavity (kind='basic', 'skipped', or 'error').
    """
    info = load_fit_params(fits_path, col_id, ext=ext)
    if info is None:
        print(f"Column {col_id} not found in {fits_path}.")
        return None

    kind = info['kind']
    if kind not in ('second_cavity', 'second_cavity_dichroics'):
        print(f"Column {col_id}: kind={kind!r}, no second cavity to plot.")
        return None

    wnum, param = info['wnum'], info['param']
    Nc, Nf, ND = info['Nc'], info['Nf'], info['ND']

    # Same slicing as in fit_FP_bayesian_core at the end of stage 3 (second
    # cavity): continuum, then finesse1/D1, then finesse2/D2 — valid whether
    # dichroics were subsequently fitted or not; only the blocks that follow
    # matter here.
    param_finesse_1 = param[Nc + 2:Nc + Nf + 4]
    param_D_1 = param[Nc + Nf + 4:Nc + ND + Nf + 6]
    param_finesse_2 = param[Nc + ND + Nf + 6:Nc + ND + 2 * Nf + 8]
    param_D_2 = param[Nc + ND + 2 * Nf + 8:Nc + 2 * ND + 2 * Nf + 10]

    # Same model structure (amplitude=1.0, Nf/ND nodes) for both cavities:
    # get_flat_result can be reused as-is.
    cavity_model = get_flat_result(Nf, ND, wnum)

    order = np.argsort(wnum)
    wnum_sorted = wnum[order]
    wnum_fine = np.linspace(wnum_sorted.min(), wnum_sorted.max(), 5 * len(wnum_sorted))

    pattern_1 = cavity_model.result(wnum_fine, np.concatenate((param_finesse_1, param_D_1)))
    pattern_2 = cavity_model.result(wnum_fine, np.concatenate((param_finesse_2, param_D_2)))

    fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(9, 6))

    ax1.plot(wnum_fine, pattern_1, color='C0')
    ax1.set_ylabel('Cavity 1 pattern')
    ax1.set_title(f"Column {col_id} — band {info['band']} — separated fringe patterns")

    ax1.plot(wnum_fine, pattern_2, color='C3')
    ax1.set_ylabel('Cavity 2 pattern')
    ax1.set_xlabel("Wavenumber [cm-1]")

    plt.tight_layout()
    if show:
        plt.show()

    return fig
