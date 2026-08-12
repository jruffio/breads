"""Unit tests for :mod:`breads.fit`.

These tests focus on the mathematical underpinnings of the fitting machinery:
parameter recovery, uncertainty calibration, chi-square behaviour, the analytic
log-probability expressions, regularization, and hypothesis testing.

Mock datasets are built from simple analytic models (a straight line, a Gaussian
on a constant background, a quadratic polynomial) with known true parameters, so
that assertions can compare the fitted values against the truth.  Reference
values are computed independently inside each test using :mod:`numpy.linalg`;
the functions under test are never mocked.

The forward-model conventions follow ``breads.fm.template.templatefm`` and the
tutorial in ``docs/source/framework/breads_simple_fit_tutorial.ipynb``: a
forward model returns ``(d, M, s)`` where ``d`` is the 1-D data vector, ``M`` is
the ``(N_data, N_linpara)`` design matrix, and ``s`` is the 1-D noise standard
deviation vector.  The *first* column of ``M`` is by convention the companion
(planet) flux, which is the column dropped when evaluating the H0 hypothesis.
"""

import warnings

import matplotlib
import numpy as np
import pytest
from scipy.special import loggamma

matplotlib.use("Agg")  # fit.py imports pyplot; keep tests headless

from breads.fit import fitfm, log_prob, combined_log_prob, nlog_prob, _get_lsq_fit
from breads.instruments import Instrument


# --------------------------------------------------------------------------
# Helpers and mock forward models
# --------------------------------------------------------------------------

def make_instrument(wavelengths, data, noise):
    """Build a minimal Instrument holding a 1-D mock dataset."""
    instrument = Instrument("test_instrument")
    with warnings.catch_warnings():
        # manual_data_entry emits a UserWarning reminding about units
        warnings.simplefilter("ignore")
        instrument.manual_data_entry(
            wavelengths=np.asarray(wavelengths, dtype=float),
            data=np.asarray(data, dtype=float),
            noise=np.asarray(noise, dtype=float),
            bad_pixels=None,
            bary_RV=0,
        )
    return instrument


def linear_fm_func(nonlin_paras, instrument, **fm_paras):
    """Straight line ``y = m*x + b``. Two linear parameters, no non-linear ones.

    Mirrors the forward model used in the breads simple fit tutorial.
    """
    x = instrument.wavelengths.flatten()
    d = instrument.data.flatten()
    s = instrument.noise.flatten()
    M = np.vstack([x, np.ones_like(x)]).T
    return d, M, s


def gaussian_fm_func(nonlin_paras, instrument, **fm_paras):
    """Gaussian of fixed width on a constant background.

    ``nonlin_paras = [mu]`` is the non-linear Gaussian centre; ``fm_paras``
    must contain ``sigma``.  The two linear parameters are the Gaussian
    amplitude (first column, i.e. the "companion") and the background level.
    """
    mu = nonlin_paras[0]
    sigma = fm_paras["sigma"]
    x = instrument.wavelengths.flatten()
    d = instrument.data.flatten()
    s = instrument.noise.flatten()
    gaussian = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    M = np.vstack([gaussian, np.ones_like(x)]).T
    return d, M, s


def polynomial_fm_func(nonlin_paras, instrument, **fm_paras):
    """Quadratic ``y = a*x^2 + b*x + c``. Three linear parameters."""
    x = instrument.wavelengths.flatten()
    d = instrument.data.flatten()
    s = instrument.noise.flatten()
    M = np.vstack([x ** 2, x, np.ones_like(x)]).T
    return d, M, s


def normalized_design_matrix(instrument, fm_func, nonlin_paras=(), fm_paras=None):
    """Return the noise-normalized ``(M, d)`` that fitfm solves internally."""
    d, M, s = fm_func(list(nonlin_paras), instrument, **(fm_paras or {}))
    return M / s[:, None], d / s


# Truth values shared by the Gaussian-based tests
GAUSS_MU_TRUE = 5.2
GAUSS_SIGMA = 0.8
GAUSS_AMPLITUDE_TRUE = 5.0
GAUSS_BACKGROUND_TRUE = 1.5


def gaussian_truth(x):
    """Noise-free Gaussian-on-background evaluated at ``x``."""
    return (
        GAUSS_AMPLITUDE_TRUE
        * np.exp(-0.5 * ((x - GAUSS_MU_TRUE) / GAUSS_SIGMA) ** 2)
        + GAUSS_BACKGROUND_TRUE
    )


@pytest.fixture
def noiseless_gaussian_instrument():
    """A perfectly noise-free Gaussian-on-background dataset."""
    x = np.linspace(0.0, 10.0, 101)
    return make_instrument(x, gaussian_truth(x), np.full_like(x, 0.2))


# --------------------------------------------------------------------------
# 1-5: Parameter recovery
# --------------------------------------------------------------------------

def test_linear_model_recovers_tutorial_values():
    """fitfm reproduces the weighted least-squares line of the tutorial dataset."""
    # Arrange: the exact dataset used in breads_simple_fit_tutorial.ipynb
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.1, 3.5, 6.1, 9.2, 10.2])
    s = np.full(5, 0.3)
    instrument = make_instrument(x, y, s)
    M_norm, d_norm = normalized_design_matrix(instrument, linear_fm_func)
    expected_paras = np.linalg.lstsq(M_norm, d_norm, rcond=None)[0]

    # Act
    log_prob, log_prob_H0, rchi2, linparas, linparas_err = fitfm(
        nonlin_paras=[], dataobj=instrument, fm_func=linear_fm_func, fm_paras={}
    )

    # Assert
    np.testing.assert_allclose(linparas, expected_paras, rtol=1e-10)
    np.testing.assert_allclose(linparas, [2.19, -0.35], atol=1e-10)
    assert np.all(np.isfinite(linparas_err))
    assert np.isfinite(log_prob) and np.isfinite(log_prob_H0)
    assert rchi2 > 0


def test_noiseless_gaussian_exact_recovery(noiseless_gaussian_instrument):
    """With no noise the true amplitude and background are recovered exactly."""
    # Arrange
    instrument = noiseless_gaussian_instrument

    # Act
    log_prob, log_prob_H0, rchi2, linparas, linparas_err = fitfm(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=instrument,
        fm_func=gaussian_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
    )

    # Assert: the model spans the data exactly, so the residuals vanish
    np.testing.assert_allclose(
        linparas, [GAUSS_AMPLITUDE_TRUE, GAUSS_BACKGROUND_TRUE], atol=1e-10
    )
    assert rchi2 == pytest.approx(0.0, abs=1e-20)
    assert log_prob > log_prob_H0


def test_polynomial_three_linear_parameters_exact_recovery():
    """A three-parameter design matrix is solved exactly on noise-free data."""
    # Arrange
    x = np.linspace(-3.0, 3.0, 41)
    a_true, b_true, c_true = 1.5, -2.0, 3.0
    y = a_true * x ** 2 + b_true * x + c_true
    instrument = make_instrument(x, y, np.full_like(x, 0.1))

    # Act
    _, _, rchi2, linparas, _ = fitfm(
        nonlin_paras=[],
        dataobj=instrument,
        fm_func=polynomial_fm_func,
        fm_paras={},
        computeH0=False,
    )

    # Assert
    np.testing.assert_allclose(linparas, [a_true, b_true, c_true], atol=1e-10)
    assert rchi2 == pytest.approx(0.0, abs=1e-20)


def test_linear_parameters_match_normal_equations():
    """fitfm's linear solution equals inv(M^T M) M^T d for the normalized system."""
    # Arrange
    rng = np.random.default_rng(11)
    x = np.linspace(0.0, 10.0, 61)
    noise = rng.uniform(0.1, 0.5, x.size)  # heteroscedastic to make weighting matter
    y = gaussian_truth(x) + rng.normal(0.0, noise)
    instrument = make_instrument(x, y, noise)

    M_norm, d_norm = normalized_design_matrix(
        instrument, gaussian_fm_func, [GAUSS_MU_TRUE], {"sigma": GAUSS_SIGMA}
    )
    expected = np.linalg.solve(M_norm.T @ M_norm, M_norm.T @ d_norm)

    # Act
    _, _, _, linparas, _ = fitfm(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=instrument,
        fm_func=gaussian_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
        computeH0=False,
    )

    # Assert
    np.testing.assert_allclose(linparas, expected, rtol=1e-8)


def test_noisy_gaussian_recovers_truth_within_uncertainty():
    """On noisy data the fitted parameters agree with truth to within 3 sigma."""
    # Arrange
    rng = np.random.default_rng(2024)
    x = np.linspace(0.0, 10.0, 101)
    sigma_noise = 0.2
    y = gaussian_truth(x) + rng.normal(0.0, sigma_noise, x.size)
    instrument = make_instrument(x, y, np.full_like(x, sigma_noise))

    # Act
    _, _, _, linparas, linparas_err = fitfm(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=instrument,
        fm_func=gaussian_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
        computeH0=False,
        scale_noise=False,
    )

    # Assert
    truth = np.array([GAUSS_AMPLITUDE_TRUE, GAUSS_BACKGROUND_TRUE])
    deviation_in_sigmas = np.abs(linparas - truth) / linparas_err
    assert np.all(deviation_in_sigmas < 3.0), (
        f"parameters {linparas} deviate from truth {truth} by "
        f"{deviation_in_sigmas} sigma"
    )


# --------------------------------------------------------------------------
# 6-10: Uncertainties
# --------------------------------------------------------------------------

def test_uncertainties_unscaled_match_inverse_normal_matrix():
    """scale_noise=False gives the textbook errors sqrt(diag(inv(M^T M)))."""
    # Arrange
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 10.0, 81)
    sigma_noise = 0.25
    y = gaussian_truth(x) + rng.normal(0.0, sigma_noise, x.size)
    instrument = make_instrument(x, y, np.full_like(x, sigma_noise))

    M_norm, _ = normalized_design_matrix(
        instrument, gaussian_fm_func, [GAUSS_MU_TRUE], {"sigma": GAUSS_SIGMA}
    )
    expected_err = np.sqrt(np.diag(np.linalg.inv(M_norm.T @ M_norm)))

    # Act
    _, _, rchi2, _, linparas_err = fitfm(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=instrument,
        fm_func=gaussian_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
        computeH0=False,
        scale_noise=False,
    )

    # Assert
    np.testing.assert_allclose(linparas_err, expected_err, rtol=1e-8)
    assert rchi2 == 1  # by construction in the scale_noise=False branch


def test_uncertainties_scaled_by_noise_scaling_factor():
    """scale_noise=True must inflate the errors by sqrt(reduced chi squared).

    KNOWN BUG (breads/fit.py, ``covphi = noise_scaling * iMTM``): the covariance
    matrix is multiplied by ``noise_scaling`` (= sqrt(rchi2)) rather than by
    ``noise_scaling ** 2`` (= rchi2).  The parameter covariance scales with the
    *variance* of the noise, so the correct rescaling of the covariance matrix is
    ``rchi2 * iMTM``, giving ``err_scaled = sqrt(rchi2) * err_unscaled``.  As
    implemented, breads returns ``err_scaled = rchi2 ** 0.25 * err_unscaled``,
    which under-inflates the error bars whenever rchi2 > 1.

    This test asserts the *conventionally correct* behaviour and is therefore
    expected to FAIL until ``covphi = noise_scaling ** 2 * iMTM`` is used.
    """
    # Arrange: quote a noise level three times smaller than the true scatter so
    # that rchi2 ~ 9 and the two scalings are clearly distinguishable.
    rng = np.random.default_rng(99)
    x = np.linspace(0.0, 10.0, 121)
    true_scatter = 0.6
    quoted_sigma = true_scatter / 3.0
    y = gaussian_truth(x) + rng.normal(0.0, true_scatter, x.size)
    instrument = make_instrument(x, y, np.full_like(x, quoted_sigma))

    common = dict(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=instrument,
        fm_func=gaussian_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
        computeH0=False,
    )

    # Act
    _, _, _, _, err_unscaled = fitfm(scale_noise=False, **common)
    _, _, rchi2, _, err_scaled = fitfm(scale_noise=True, **common)

    # Assert
    assert rchi2 > 4.0, "test setup should produce a badly underestimated noise"
    expected_scaled = np.sqrt(rchi2) * err_unscaled
    np.testing.assert_allclose(err_scaled, expected_scaled, rtol=1e-8)


def test_uncertainties_are_statistically_calibrated():
    """Monte-Carlo: scale_noise=True errors should match the actual scatter.

    The purpose of ``scale_noise=True`` is to repair error bars when the quoted
    noise vector is wrong.  Here the quoted sigma is three times too small, so a
    correctly implemented noise rescaling should still return uncertainties that
    match the empirical scatter of the fitted amplitude across realizations.

    KNOWN BUG: because ``covphi = noise_scaling * iMTM`` applies sqrt(rchi2)
    instead of rchi2 to the covariance (see
    ``test_uncertainties_scaled_by_noise_scaling_factor``), the returned errors
    come out too small by a factor of about sqrt(3) here.  This test asserts the
    correct calibration and is therefore expected to FAIL until that is fixed.
    """
    # Arrange
    rng = np.random.default_rng(1234)
    x = np.linspace(0.0, 10.0, 101)
    true_scatter = 0.6
    quoted_sigma = true_scatter / 3.0
    truth = gaussian_truth(x)
    n_realizations = 300

    # Act
    amplitudes = np.empty(n_realizations)
    reported_errors = np.empty(n_realizations)
    for i in range(n_realizations):
        instrument = make_instrument(
            x,
            truth + rng.normal(0.0, true_scatter, x.size),
            np.full_like(x, quoted_sigma),
        )
        _, _, _, linparas, linparas_err = fitfm(
            nonlin_paras=[GAUSS_MU_TRUE],
            dataobj=instrument,
            fm_func=gaussian_fm_func,
            fm_paras={"sigma": GAUSS_SIGMA},
            computeH0=False,
            scale_noise=True,
        )
        amplitudes[i] = linparas[0]
        reported_errors[i] = linparas_err[0]

    # Assert
    empirical_scatter = amplitudes.std(ddof=1)
    mean_reported_error = reported_errors.mean()
    assert mean_reported_error == pytest.approx(empirical_scatter, rel=0.15), (
        f"reported uncertainty {mean_reported_error:.4f} does not match the "
        f"empirical scatter {empirical_scatter:.4f}"
    )


def test_uncertainties_calibrated_when_noise_correctly_specified():
    """Positive control: with a correct noise vector the errors are calibrated.

    This exercises the same Monte-Carlo machinery as
    ``test_uncertainties_are_statistically_calibrated`` but with a correctly
    specified noise vector and ``scale_noise=False``, i.e. the code path that is
    free of the covariance-scaling bug.  It should pass.
    """
    # Arrange
    rng = np.random.default_rng(5678)
    x = np.linspace(0.0, 10.0, 101)
    sigma_noise = 0.2
    truth = gaussian_truth(x)
    n_realizations = 300

    # Act
    amplitudes = np.empty(n_realizations)
    reported_errors = np.empty(n_realizations)
    for i in range(n_realizations):
        instrument = make_instrument(
            x, truth + rng.normal(0.0, sigma_noise, x.size), np.full_like(x, sigma_noise)
        )
        _, _, _, linparas, linparas_err = fitfm(
            nonlin_paras=[GAUSS_MU_TRUE],
            dataobj=instrument,
            fm_func=gaussian_fm_func,
            fm_paras={"sigma": GAUSS_SIGMA},
            computeH0=False,
            scale_noise=False,
        )
        amplitudes[i] = linparas[0]
        reported_errors[i] = linparas_err[0]

    # Assert
    assert amplitudes.mean() == pytest.approx(GAUSS_AMPLITUDE_TRUE, abs=0.02)
    assert reported_errors.mean() == pytest.approx(
        amplitudes.std(ddof=1), rel=0.15
    )


def test_linear_scaling_invariance():
    """Scaling data and noise by a constant scales parameters and errors by it."""
    # Arrange
    scale = 7.0
    x = np.linspace(0.0, 10.0, 101)
    sigma_noise = 0.2
    y = gaussian_truth(x)
    instrument = make_instrument(x, y, np.full_like(x, sigma_noise))
    scaled_instrument = make_instrument(
        x, y * scale, np.full_like(x, sigma_noise * scale)
    )
    common = dict(
        nonlin_paras=[GAUSS_MU_TRUE],
        fm_func=gaussian_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
        computeH0=False,
        scale_noise=False,
    )

    # Act
    _, _, _, linparas, linparas_err = fitfm(dataobj=instrument, **common)
    _, _, _, scaled_paras, scaled_err = fitfm(dataobj=scaled_instrument, **common)

    # Assert
    np.testing.assert_allclose(scaled_paras, scale * linparas, rtol=1e-10)
    np.testing.assert_allclose(scaled_err, scale * linparas_err, rtol=1e-10)


def test_heteroscedastic_noise_is_correctly_weighted():
    """Per-point noise is used as inverse weights, not merely as an overall scale."""
    # Arrange: the first half of the spectrum is far more precise than the second
    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 10.0, 101)
    noise = np.where(x < 5.0, 0.05, 2.0)
    y = gaussian_truth(x) + rng.normal(0.0, noise)
    correct_instrument = make_instrument(x, y, noise)
    # Same data, but pretending the noise is uniform (and hence mis-weighted)
    flat_instrument = make_instrument(x, y, np.full_like(x, 1.0))
    common = dict(
        nonlin_paras=[GAUSS_MU_TRUE],
        fm_func=gaussian_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
        computeH0=False,
        scale_noise=False,
    )

    # Act
    _, _, _, weighted_paras, weighted_err = fitfm(dataobj=correct_instrument, **common)
    _, _, _, flat_paras, flat_err = fitfm(dataobj=flat_instrument, **common)

    # Assert: correct weighting is both tighter and closer to the truth
    truth = np.array([GAUSS_AMPLITUDE_TRUE, GAUSS_BACKGROUND_TRUE])
    assert np.all(weighted_err < flat_err)
    assert np.abs(weighted_paras[0] - truth[0]) < np.abs(flat_paras[0] - truth[0])
    np.testing.assert_allclose(weighted_paras, truth, atol=0.1)


# --------------------------------------------------------------------------
# 11-13: Chi-square behaviour
# --------------------------------------------------------------------------

def test_rchi2_near_unity_for_correct_noise_model():
    """A correctly specified noise vector yields a reduced chi squared near one."""
    # Arrange
    rng = np.random.default_rng(17)
    x = np.linspace(0.0, 10.0, 1001)
    sigma_noise = 0.2
    y = gaussian_truth(x) + rng.normal(0.0, sigma_noise, x.size)
    instrument = make_instrument(x, y, np.full_like(x, sigma_noise))

    # Act
    _, _, rchi2, _, _ = fitfm(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=instrument,
        fm_func=gaussian_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
        computeH0=False,
        scale_noise=True,
    )

    # Assert: expectation is (N_data - N_linpara) / N_data, i.e. ~1 for large N
    assert rchi2 == pytest.approx(1.0, abs=0.15)


def test_rchi2_scales_as_square_of_noise_underestimate():
    """Underestimating the noise by a factor k inflates rchi2 by k squared."""
    # Arrange: identical data, but the quoted noise is understated 3x
    rng = np.random.default_rng(21)
    x = np.linspace(0.0, 10.0, 1001)
    true_scatter = 0.6
    underestimate_factor = 3.0
    y = gaussian_truth(x) + rng.normal(0.0, true_scatter, x.size)
    correct_instrument = make_instrument(x, y, np.full_like(x, true_scatter))
    understated_instrument = make_instrument(
        x, y, np.full_like(x, true_scatter / underestimate_factor)
    )
    common = dict(
        nonlin_paras=[GAUSS_MU_TRUE],
        fm_func=gaussian_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
        computeH0=False,
        scale_noise=True,
    )

    # Act
    _, _, rchi2_correct, _, _ = fitfm(dataobj=correct_instrument, **common)
    _, _, rchi2_understated, _, _ = fitfm(dataobj=understated_instrument, **common)

    # Assert
    assert rchi2_correct == pytest.approx(1.0, abs=0.15)
    assert rchi2_understated == pytest.approx(
        rchi2_correct * underestimate_factor ** 2, rel=1e-10
    )


def test_rchi2_is_unity_when_scale_noise_false():
    """The scale_noise=False branch hard-sets rchi2 to exactly one.

    This is a bookkeeping flag rather than a measured goodness of fit: no noise
    rescaling is applied, so the reported value is 1 regardless of the data.
    """
    # Arrange: wildly inconsistent noise so that a *measured* rchi2 would be huge
    rng = np.random.default_rng(23)
    x = np.linspace(0.0, 10.0, 101)
    y = gaussian_truth(x) + rng.normal(0.0, 1.0, x.size)
    instrument = make_instrument(x, y, np.full_like(x, 0.01))

    # Act
    _, _, rchi2, _, _ = fitfm(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=instrument,
        fm_func=gaussian_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
        computeH0=False,
        scale_noise=False,
    )

    # Assert
    assert rchi2 == 1


def test_get_lsq_fit_with_explicit_n_data_returns_correct_chi2():
    """_get_lsq_fit(N_data=N) returns the textbook chi2, rchi2 and noise scaling.

    This is the code path used when regularization rows have been appended to
    the design matrix: the residuals beyond ``N_data`` belong to the priors and
    must be excluded from the goodness-of-fit statistics.
    """
    # Arrange: 6 data rows followed by 2 regularization rows
    rng = np.random.default_rng(31)
    n_data = 6
    M = rng.normal(size=(n_data + 2, 3))
    d = rng.normal(size=n_data + 2)
    unbounded = ([-np.inf] * 3, [np.inf] * 3)

    # Act
    paras, d_estimated, residuals, chi2, rchi2, noise_scaling = _get_lsq_fit(
        M, d, unbounded, N_data=n_data
    )

    # Assert
    expected_paras = np.linalg.lstsq(M, d, rcond=None)[0]
    np.testing.assert_allclose(paras, expected_paras, rtol=1e-8)
    np.testing.assert_allclose(d_estimated, M @ paras, rtol=1e-10)
    np.testing.assert_allclose(residuals, d - M @ paras, rtol=1e-10)

    expected_chi2 = np.sum(residuals[:n_data] ** 2)
    assert chi2 == pytest.approx(expected_chi2, rel=1e-10)
    assert rchi2 == pytest.approx(expected_chi2 / n_data, rel=1e-10)
    assert noise_scaling == pytest.approx(np.sqrt(expected_chi2 / n_data), rel=1e-10)
    # Only the first n_data residuals contribute
    assert chi2 < np.sum(residuals ** 2)


def test_get_lsq_fit_with_none_n_data_returns_textbook_chi2():
    """_get_lsq_fit(N_data=None) returns the full chi2 and its reduction by N.

    In the ``N_data is None`` branch the number of data points is taken to be the
    full length of the residual vector.  ``chi2`` is the plain sum of squared
    residuals, ``rchi2 = chi2 / N`` and ``noise_scaling = sqrt(rchi2)``.

    (Historical note: this branch previously divided the sum of squares by N an
    extra time, so that the returned ``chi2`` was already a reduced chi squared;
    that was corrected upstream so ``chi2`` is now the plain sum of squares.)

    Because there are no appended regularization rows here, every residual is a
    data residual, so this branch must agree with the explicitly supplied
    ``N_data`` branch exercised in
    ``test_get_lsq_fit_with_explicit_n_data_returns_correct_chi2``.
    """
    # Arrange
    rng = np.random.default_rng(31)
    n_rows = 8
    M = rng.normal(size=(n_rows, 3))
    d = rng.normal(size=n_rows)
    unbounded = ([-np.inf] * 3, [np.inf] * 3)

    # Act
    paras, _, residuals, chi2, rchi2, noise_scaling = _get_lsq_fit(
        M, d, unbounded, N_data=None
    )

    # Assert
    sum_squared_residuals = np.sum(residuals ** 2)
    assert chi2 == pytest.approx(sum_squared_residuals, rel=1e-10)
    assert rchi2 == pytest.approx(sum_squared_residuals / n_rows, rel=1e-10)
    assert noise_scaling == pytest.approx(
        np.sqrt(sum_squared_residuals / n_rows), rel=1e-10
    )
    # With no regularization rows this must match the explicit-N_data branch
    paras_explicit, _, _, chi2_explicit, _, _ = _get_lsq_fit(
        M, d, unbounded, N_data=n_rows
    )
    assert chi2 == pytest.approx(chi2_explicit, rel=1e-10)
    # The parameters themselves are the ordinary least-squares solution
    np.testing.assert_allclose(paras, np.linalg.lstsq(M, d, rcond=None)[0], rtol=1e-8)
    np.testing.assert_allclose(paras, paras_explicit, rtol=1e-10)


# --------------------------------------------------------------------------
# 14-19: Log probability and hypothesis testing
# --------------------------------------------------------------------------

def test_log_prob_matches_analytic_expression():
    """fitfm reproduces log(Eq. 36) of Ruffio+2019 as coded.

    The expression marginalized over the linear parameters is

        ln P = (Np - Nd)/2 ln(2 pi) - 1/2 ln|Sigma| - 1/2 ln|M^T M|
               - (Nd - Np)/2 ln(noise_scaling^2) - chi2 / (2 noise_scaling^2)

    Note that the ``chi2`` breads feeds into this expression comes from
    ``_get_lsq_fit(..., N_data=None)`` and is the plain sum of squared
    (noise-normalized) residuals, while ``noise_scaling`` is computed separately
    from the reduced chi squared ``chi2 / N_data``.  The reference below
    reproduces that split.
    """
    # Arrange
    rng = np.random.default_rng(41)
    x = np.linspace(0.0, 10.0, 51)
    sigma_noise = 0.3
    y = gaussian_truth(x) + rng.normal(0.0, sigma_noise, x.size)
    noise = np.full_like(x, sigma_noise)
    instrument = make_instrument(x, y, noise)

    M_norm, d_norm = normalized_design_matrix(
        instrument, gaussian_fm_func, [GAUSS_MU_TRUE], {"sigma": GAUSS_SIGMA}
    )
    n_data, n_linpara = M_norm.shape
    paras = np.linalg.lstsq(M_norm, d_norm, rcond=None)[0]
    residuals = d_norm - M_norm @ paras

    chi2 = np.sum(residuals ** 2)
    rchi2 = np.sum(residuals ** 2) / n_data
    noise_scaling = np.sqrt(rchi2)
    logdet_Sigma = np.sum(2 * np.log(noise))
    logdet_icovphi0 = np.linalg.slogdet(M_norm.T @ M_norm)[1]

    expected_log_prob = (
        ((n_linpara - n_data) / 2) * np.log(2 * np.pi)
        - 0.5 * logdet_Sigma
        - 0.5 * logdet_icovphi0
        - ((n_data - n_linpara) / 2) * np.log(noise_scaling ** 2)
        - 0.5 * chi2 / noise_scaling ** 2
    )

    # Act
    actual_log_prob, _, actual_rchi2, _, _ = fitfm(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=instrument,
        fm_func=gaussian_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
        computeH0=False,
        scale_noise=True,
    )

    # Assert
    assert actual_rchi2 == pytest.approx(rchi2, rel=1e-10)
    assert actual_log_prob == pytest.approx(expected_log_prob, rel=1e-10)


def test_log_prob_H0_matches_analytic_expression():
    """The H0 log probability is the same expression with the first column dropped.

    The H0 hypothesis removes the first (companion) column of the design matrix.
    Note that ``_compute_H0`` is always called with the default
    ``noise_scaling=1`` and uses the *full* model's column count in the
    prefactors; the reference value reproduces that convention.
    """
    # Arrange
    rng = np.random.default_rng(43)
    x = np.linspace(0.0, 10.0, 51)
    sigma_noise = 0.3
    y = gaussian_truth(x) + rng.normal(0.0, sigma_noise, x.size)
    noise = np.full_like(x, sigma_noise)
    instrument = make_instrument(x, y, noise)

    M_norm, d_norm = normalized_design_matrix(
        instrument, gaussian_fm_func, [GAUSS_MU_TRUE], {"sigma": GAUSS_SIGMA}
    )
    n_data, n_linpara = M_norm.shape
    M_H0 = M_norm[:, 1:]
    paras_H0 = np.linalg.lstsq(M_H0, d_norm, rcond=None)[0]
    residuals_H0 = d_norm - M_H0 @ paras_H0

    chi2_H0 = np.sum(residuals_H0 ** 2)
    logdet_Sigma = np.sum(2 * np.log(noise))
    logdet_icovphi0_H0 = np.linalg.slogdet(M_H0.T @ M_H0)[1]

    expected_log_prob_H0 = (
        ((n_linpara - n_data) / 2) * np.log(2 * np.pi)
        - 0.5 * logdet_Sigma
        - 0.5 * logdet_icovphi0_H0
        - ((n_data - n_linpara) / 2) * np.log(1.0)
        - 0.5 * chi2_H0 / 1.0
    )

    # Act
    _, actual_log_prob_H0, _, _, _ = fitfm(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=instrument,
        fm_func=gaussian_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
        computeH0=True,
    )

    # Assert
    assert actual_log_prob_H0 == pytest.approx(expected_log_prob_H0, rel=1e-10)


def test_log_prob_peaks_at_true_nonlinear_parameter():
    """Scanning the non-linear parameter, the log probability peaks at the truth."""
    # Arrange
    rng = np.random.default_rng(47)
    x = np.linspace(0.0, 10.0, 201)
    sigma_noise = 0.2
    mu_grid = np.linspace(4.0, 6.5, 251)  # step 0.01, contains GAUSS_MU_TRUE
    assert np.min(np.abs(mu_grid - GAUSS_MU_TRUE)) < 1e-12
    grid_step = mu_grid[1] - mu_grid[0]

    noiseless = make_instrument(x, gaussian_truth(x), np.full_like(x, sigma_noise))
    noisy = make_instrument(
        x,
        gaussian_truth(x) + rng.normal(0.0, sigma_noise, x.size),
        np.full_like(x, sigma_noise),
    )

    # Act
    noiseless_curve = np.array(
        [
            log_prob([mu], noiseless, gaussian_fm_func, {"sigma": GAUSS_SIGMA})
            for mu in mu_grid
        ]
    )
    noisy_curve = np.array(
        [
            log_prob([mu], noisy, gaussian_fm_func, {"sigma": GAUSS_SIGMA})
            for mu in mu_grid
        ]
    )

    # Assert
    assert mu_grid[np.argmax(noiseless_curve)] == pytest.approx(
        GAUSS_MU_TRUE, abs=grid_step
    )
    assert mu_grid[np.argmax(noisy_curve)] == pytest.approx(
        GAUSS_MU_TRUE, abs=10 * grid_step
    )
    # The curve must actually be peaked, not flat or monotonic
    assert noisy_curve.max() > noisy_curve[0] + 10
    assert noisy_curve.max() > noisy_curve[-1] + 10


def test_log_prob_peak_width_scales_with_noise_level():
    """The width of the log-probability peak is proportional to the noise level.

    For a Gaussian likelihood the curvature of ln P at its maximum gives the
    parameter uncertainty as ``sigma_mu = 1 / sqrt(-d2 lnP / dmu2)``, which for a
    linear-in-amplitude model scales linearly with the noise amplitude.  Doubling
    the noise must therefore double the recovered width.
    """
    # Arrange: noise-free data so the peak sits exactly at the truth and the
    # comparison isolates the noise dependence.
    x = np.linspace(0.0, 10.0, 201)
    y = gaussian_truth(x)
    delta = 0.02
    mu_samples = [GAUSS_MU_TRUE - delta, GAUSS_MU_TRUE, GAUSS_MU_TRUE + delta]

    def peak_width(noise_level):
        instrument = make_instrument(x, y, np.full_like(x, noise_level))
        curve = np.array(
            [
                log_prob(
                    [mu],
                    instrument,
                    gaussian_fm_func,
                    {"sigma": GAUSS_SIGMA},
                    scale_noise=False,
                )
                for mu in mu_samples
            ]
        )
        # second derivative via central differences on a uniform 3-point stencil
        second_derivative = (curve[0] - 2 * curve[1] + curve[2]) / delta ** 2
        assert second_derivative < 0, "log probability should be concave at the peak"
        return 1.0 / np.sqrt(-second_derivative)

    # Act
    width_low = peak_width(0.2)
    width_high = peak_width(0.4)

    # Assert
    assert width_high == pytest.approx(2.0 * width_low, rel=1e-3)


def test_bayes_factor_large_with_signal_and_small_without():
    """log_prob - log_prob_H0 discriminates a real companion from pure noise.

    ``scale_noise=False`` is used deliberately so that both hypotheses are
    evaluated with the same noise scaling.  (With ``scale_noise=True`` the H1
    branch uses the fitted ``noise_scaling`` while ``_compute_H0`` is always
    invoked with its default ``noise_scaling=1``, which makes the two terms not
    directly comparable.)

    With both hypotheses on the same footing the difference reduces to an
    analytic form: a signal-independent "Occam" term from the extra model column
    plus half the chi-square improvement.  With ``chi2`` now the plain sum of
    squared (noise-normalized) residuals, this is the textbook log likelihood
    ratio, so a strong companion produces a very large evidence gain.
    """
    # Arrange
    rng = np.random.default_rng(53)
    x = np.linspace(0.0, 10.0, 201)
    sigma_noise = 0.2
    noise = np.full_like(x, sigma_noise)
    with_signal = make_instrument(
        x, gaussian_truth(x) + rng.normal(0.0, sigma_noise, x.size), noise
    )
    without_signal = make_instrument(
        x,
        np.full_like(x, GAUSS_BACKGROUND_TRUE) + rng.normal(0.0, sigma_noise, x.size),
        noise,
    )
    common = dict(
        nonlin_paras=[GAUSS_MU_TRUE],
        fm_func=gaussian_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
        computeH0=True,
        scale_noise=False,
    )

    def analytic_evidence_gain(instrument):
        M_norm, d_norm = normalized_design_matrix(
            instrument, gaussian_fm_func, [GAUSS_MU_TRUE], {"sigma": GAUSS_SIGMA}
        )
        M_H0 = M_norm[:, 1:]
        chi2_H1 = np.sum(
            (d_norm - M_norm @ np.linalg.lstsq(M_norm, d_norm, rcond=None)[0]) ** 2
        )
        chi2_H0 = np.sum(
            (d_norm - M_H0 @ np.linalg.lstsq(M_H0, d_norm, rcond=None)[0]) ** 2
        )
        occam = -0.5 * (
            np.linalg.slogdet(M_norm.T @ M_norm)[1]
            - np.linalg.slogdet(M_H0.T @ M_H0)[1]
        )
        return occam + 0.5 * (chi2_H0 - chi2_H1)

    # Act
    lp_sig, lp_H0_sig, _, paras_sig, err_sig = fitfm(dataobj=with_signal, **common)
    lp_bg, lp_H0_bg, _, paras_bg, err_bg = fitfm(dataobj=without_signal, **common)
    evidence_gain_sig = lp_sig - lp_H0_sig
    evidence_gain_bg = lp_bg - lp_H0_bg

    # Assert: the evidence gain matches its analytic decomposition
    assert evidence_gain_sig == pytest.approx(
        analytic_evidence_gain(with_signal), rel=1e-8
    )
    assert evidence_gain_bg == pytest.approx(
        analytic_evidence_gain(without_signal), rel=1e-8
    )
    # A real companion is decisively favoured; pure background is not
    assert evidence_gain_sig > 1000
    assert evidence_gain_bg < 0
    assert evidence_gain_sig > 100 * abs(evidence_gain_bg)
    # Sanity check on the corresponding amplitudes
    assert paras_sig[0] / err_sig[0] > 20  # high signal to noise detection
    assert abs(paras_bg[0] / err_bg[0]) < 3  # consistent with zero


def test_marginalize_noise_scaling_matches_analytic_expression():
    """The marginalized-noise-scaling branch matches its analytic form.

    With ``marginalize_noise_scaling=True`` the noise scaling factor is
    analytically marginalized rather than fixed at its maximum-likelihood value,
    replacing the chi-square term with a Student-t style expression.  The best
    fit linear parameters are unaffected by this choice.
    """
    # Arrange
    rng = np.random.default_rng(59)
    x = np.linspace(0.0, 10.0, 51)
    sigma_noise = 0.3
    y = gaussian_truth(x) + rng.normal(0.0, sigma_noise, x.size)
    noise = np.full_like(x, sigma_noise)
    instrument = make_instrument(x, y, noise)

    M_norm, d_norm = normalized_design_matrix(
        instrument, gaussian_fm_func, [GAUSS_MU_TRUE], {"sigma": GAUSS_SIGMA}
    )
    n_data, n_linpara = M_norm.shape
    paras = np.linalg.lstsq(M_norm, d_norm, rcond=None)[0]
    residuals = d_norm - M_norm @ paras
    chi2 = np.sum(residuals ** 2)
    logdet_Sigma = np.sum(2 * np.log(noise))
    logdet_icovphi0 = np.linalg.slogdet(M_norm.T @ M_norm)[1]
    dof = n_data - n_linpara + 2 - 1

    expected_log_prob = (
        (n_linpara - n_data) / 2 * np.log(2 * np.pi)
        - 0.5 * logdet_Sigma
        - 0.5 * logdet_icovphi0
        - (dof / 2) * np.log(chi2)
        + loggamma(dof / 2)
    )

    common = dict(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=instrument,
        fm_func=gaussian_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
        computeH0=False,
    )

    # Act
    lp_marg, _, _, paras_marg, err_marg = fitfm(marginalize_noise_scaling=True, **common)
    lp_plain, _, _, paras_plain, err_plain = fitfm(
        marginalize_noise_scaling=False, **common
    )

    # Assert
    assert lp_marg == pytest.approx(expected_log_prob, rel=1e-10)
    assert np.isfinite(lp_marg)
    assert lp_marg != lp_plain  # different likelihood, same solution
    np.testing.assert_allclose(paras_marg, paras_plain, rtol=1e-12)
    np.testing.assert_allclose(err_marg, err_plain, rtol=1e-12)


def test_marginalize_noise_scaling_peaks_at_true_nonlinear_parameter():
    """The marginalized likelihood also peaks at the true non-linear parameter."""
    # Arrange
    rng = np.random.default_rng(61)
    x = np.linspace(0.0, 10.0, 201)
    sigma_noise = 0.2
    instrument = make_instrument(
        x,
        gaussian_truth(x) + rng.normal(0.0, sigma_noise, x.size),
        np.full_like(x, sigma_noise),
    )
    mu_grid = np.linspace(4.0, 6.5, 251)

    # Act
    curve = np.array(
        [
            fitfm(
                nonlin_paras=[mu],
                dataobj=instrument,
                fm_func=gaussian_fm_func,
                fm_paras={"sigma": GAUSS_SIGMA},
                computeH0=False,
                marginalize_noise_scaling=True,
            )[0]
            for mu in mu_grid
        ]
    )

    # Assert
    assert mu_grid[np.argmax(curve)] == pytest.approx(GAUSS_MU_TRUE, abs=0.1)


# --------------------------------------------------------------------------
# 20-25: Regularization (four-output forward models)
# --------------------------------------------------------------------------

def make_regularized_fm(prior_values, prior_sigmas):
    """Build a Gaussian forward model that also returns regularization priors.

    ``extra_outputs["regularization"]`` is a ``(d_reg, s_reg)`` pair with one
    entry per linear parameter, following the convention of
    ``breads.fm.hc_atmgrid_splinefm_jwst_ifu_cal``.  A NaN in ``s_reg`` means
    "no prior on this parameter"; a finite entry adds a row to the design matrix
    that pulls the parameter towards ``d_reg`` with weight ``1 / s_reg``.
    """
    d_reg = np.asarray(prior_values, dtype=float)
    s_reg = np.asarray(prior_sigmas, dtype=float)

    def regularized_fm_func(nonlin_paras, instrument, **fm_paras):
        d, M, s = gaussian_fm_func(nonlin_paras, instrument, **fm_paras)
        return d, M, s, {"regularization": (d_reg.copy(), s_reg.copy())}

    return regularized_fm_func


def four_output_fm_func(nonlin_paras, instrument, **fm_paras):
    """A four-output forward model whose extra dict has no regularization key."""
    d, M, s = gaussian_fm_func(nonlin_paras, instrument, **fm_paras)
    return d, M, s, {"wvs": instrument.wavelengths.flatten()}


@pytest.fixture
def noisy_gaussian_instrument():
    """A moderately noisy Gaussian-on-background dataset."""
    rng = np.random.default_rng(101)
    x = np.linspace(0.0, 10.0, 101)
    sigma_noise = 0.2
    return make_instrument(
        x,
        gaussian_truth(x) + rng.normal(0.0, sigma_noise, x.size),
        np.full_like(x, sigma_noise),
    )


def test_four_output_fm_without_regularization_matches_three_output(
    noisy_gaussian_instrument,
):
    """An extra_outputs dict lacking a 'regularization' key is inert."""
    # Arrange
    common = dict(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=noisy_gaussian_instrument,
        fm_paras={"sigma": GAUSS_SIGMA},
    )

    # Act
    three_output = fitfm(fm_func=gaussian_fm_func, **common)
    four_output = fitfm(fm_func=four_output_fm_func, **common)

    # Assert
    assert four_output[0] == pytest.approx(three_output[0], rel=1e-12)
    assert four_output[1] == pytest.approx(three_output[1], rel=1e-12)
    assert four_output[2] == pytest.approx(three_output[2], rel=1e-12)
    np.testing.assert_allclose(four_output[3], three_output[3], rtol=1e-12)
    np.testing.assert_allclose(four_output[4], three_output[4], rtol=1e-12)


def test_strong_regularization_pulls_parameter_toward_prior(
    noisy_gaussian_instrument,
):
    """A very tight prior forces the background towards its prior value."""
    # Arrange: no prior on the amplitude (NaN), a very tight prior on the
    # background pulling it to a value far from the true 1.5
    prior_background = 0.0
    fm_func = make_regularized_fm(
        prior_values=[np.nan, prior_background], prior_sigmas=[np.nan, 1e-6]
    )

    # Act
    _, _, _, linparas, _ = fitfm(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=noisy_gaussian_instrument,
        fm_func=fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
        computeH0=False,
    )

    # Assert
    assert linparas[1] == pytest.approx(prior_background, abs=1e-3)
    # The amplitude absorbs part of the background it was denied
    assert linparas[0] > GAUSS_AMPLITUDE_TRUE


def test_weak_regularization_recovers_unregularized_solution(
    noisy_gaussian_instrument,
):
    """A very loose prior leaves the unregularized solution unchanged."""
    # Arrange
    fm_func = make_regularized_fm(
        prior_values=[np.nan, 0.0], prior_sigmas=[np.nan, 1e8]
    )
    common = dict(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=noisy_gaussian_instrument,
        fm_paras={"sigma": GAUSS_SIGMA},
        computeH0=False,
    )

    # Act
    _, _, _, unregularized, _ = fitfm(fm_func=gaussian_fm_func, **common)
    _, _, _, weakly_regularized, _ = fitfm(fm_func=fm_func, **common)

    # Assert
    np.testing.assert_allclose(weakly_regularized, unregularized, rtol=1e-6)


def test_regularization_strength_interpolates_between_limits(
    noisy_gaussian_instrument,
):
    """Background estimate moves monotonically from the prior towards the data."""
    # Arrange
    prior_background = 0.0
    common = dict(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=noisy_gaussian_instrument,
        fm_paras={"sigma": GAUSS_SIGMA},
        computeH0=False,
    )
    _, _, _, unregularized, _ = fitfm(fm_func=gaussian_fm_func, **common)

    # Act
    backgrounds = [
        fitfm(
            fm_func=make_regularized_fm([np.nan, prior_background], [np.nan, s_reg]),
            **common,
        )[3][1]
        for s_reg in (1e-4, 1e-2, 1e-1, 1e0, 1e4)
    ]

    # Assert: monotonically increasing from the prior towards the free solution
    assert np.all(np.diff(backgrounds) > 0)
    assert backgrounds[0] == pytest.approx(prior_background, abs=1e-3)
    assert backgrounds[-1] == pytest.approx(unregularized[1], rel=1e-4)


def test_regularization_nan_entries_leave_parameter_unconstrained(
    noisy_gaussian_instrument,
):
    """NaN entries in s_reg add no row, so those parameters stay unconstrained.

    Here both entries are NaN, so the regularization block is empty and the
    result must be identical to the unregularized fit.
    """
    # Arrange
    fm_func = make_regularized_fm(
        prior_values=[np.nan, np.nan], prior_sigmas=[np.nan, np.nan]
    )
    common = dict(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=noisy_gaussian_instrument,
        fm_paras={"sigma": GAUSS_SIGMA},
        computeH0=False,
    )

    # Act
    _, _, _, unregularized, _ = fitfm(fm_func=gaussian_fm_func, **common)
    _, _, _, all_nan_priors, _ = fitfm(fm_func=fm_func, **common)

    # Assert
    np.testing.assert_allclose(all_nan_priors, unregularized, rtol=1e-10)


def test_regularization_does_not_constrain_the_companion_amplitude(
    noisy_gaussian_instrument,
):
    """A prior on the background alone must not bias the amplitude to its prior.

    The first linear parameter (the companion flux) carries a NaN prior, so it
    remains free even though a hard prior is applied to the background.
    """
    # Arrange: an absurd prior value for the background
    fm_func = make_regularized_fm(
        prior_values=[-999.0, GAUSS_BACKGROUND_TRUE], prior_sigmas=[np.nan, 1e-6]
    )

    # Act
    _, _, _, linparas, linparas_err = fitfm(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=noisy_gaussian_instrument,
        fm_func=fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
        computeH0=False,
    )

    # Assert: the background is pinned, the amplitude still recovers the truth
    assert linparas[1] == pytest.approx(GAUSS_BACKGROUND_TRUE, abs=1e-3)
    assert linparas[0] == pytest.approx(GAUSS_AMPLITUDE_TRUE, abs=0.15)
    assert np.all(np.isfinite(linparas_err))


def test_regularization_with_scale_noise_false_runs_and_differs(
    noisy_gaussian_instrument,
):
    """The scale_noise=False regularization path executes and gives finite results."""
    # Arrange
    fm_func = make_regularized_fm(prior_values=[np.nan, 0.0], prior_sigmas=[np.nan, 0.1])
    common = dict(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=noisy_gaussian_instrument,
        fm_func=fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
        computeH0=False,
    )

    # Act
    lp_scaled, _, rchi2_scaled, paras_scaled, err_scaled = fitfm(
        scale_noise=True, **common
    )
    lp_plain, _, rchi2_plain, paras_plain, err_plain = fitfm(scale_noise=False, **common)

    # Assert
    for value in (lp_scaled, lp_plain, rchi2_scaled, rchi2_plain):
        assert np.isfinite(value)
    assert np.all(np.isfinite(paras_scaled)) and np.all(np.isfinite(err_scaled))
    assert np.all(np.isfinite(paras_plain)) and np.all(np.isfinite(err_plain))
    # In the regularization branch rchi2 is measured, not forced to 1
    assert rchi2_plain != 1
    assert rchi2_scaled == pytest.approx(rchi2_plain, rel=1e-10)
    # Rescaling the noise changes the relative weight of the prior, so the
    # solutions must not be identical
    assert not np.allclose(paras_scaled, paras_plain)


def test_regularization_with_marginalize_noise_scaling_raises(
    noisy_gaussian_instrument,
):
    """Combining regularization with marginalized noise scaling is rejected."""
    # Arrange
    fm_func = make_regularized_fm(prior_values=[np.nan, 0.0], prior_sigmas=[np.nan, 0.1])

    # Act / Assert
    with pytest.raises(Exception, match="not compatible with the regularization"):
        fitfm(
            nonlin_paras=[GAUSS_MU_TRUE],
            dataobj=noisy_gaussian_instrument,
            fm_func=fm_func,
            fm_paras={"sigma": GAUSS_SIGMA},
            computeH0=False,
            marginalize_noise_scaling=True,
        )


# --------------------------------------------------------------------------
# 26-34: Degenerate, invalid and boundary inputs
# --------------------------------------------------------------------------

def assert_invalid_outputs(result, n_linpara):
    """Assert that fitfm returned its 'nothing could be fitted' sentinel values."""
    log_prob_value, log_prob_H0, rchi2, linparas, linparas_err = result
    assert log_prob_value == -np.inf
    assert log_prob_H0 == -np.inf
    assert rchi2 == np.inf
    assert linparas.shape == (n_linpara,)
    assert linparas_err.shape == (n_linpara,)
    assert np.all(np.isnan(linparas))
    assert np.all(np.isnan(linparas_err))


def test_wrong_number_of_fm_outputs_raises_value_error(noisy_gaussian_instrument):
    """A forward model returning other than 3 or 4 values is rejected."""
    # Arrange
    def two_output_fm_func(nonlin_paras, instrument, **fm_paras):
        d, M, _ = gaussian_fm_func(nonlin_paras, instrument, **fm_paras)
        return d, M

    # Act / Assert
    with pytest.raises(ValueError, match="Unrecognized number of matrices"):
        fitfm(
            nonlin_paras=[GAUSS_MU_TRUE],
            dataobj=noisy_gaussian_instrument,
            fm_func=two_output_fm_func,
            fm_paras={"sigma": GAUSS_SIGMA},
        )


def test_empty_data_returns_invalid_outputs(noisy_gaussian_instrument):
    """An empty data vector short-circuits to the invalid-output sentinels.

    Forward models are expected to return empty arrays (rather than raise) when
    the requested position falls entirely outside the valid data, e.g. off the
    edge of the field of view.
    """
    # Arrange
    n_linpara = 3

    def empty_fm_func(nonlin_paras, instrument, **fm_paras):
        return np.array([]), np.array([]).reshape(0, n_linpara), np.array([])

    # Act
    result = fitfm(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=noisy_gaussian_instrument,
        fm_func=empty_fm_func,
        fm_paras={},
    )

    # Assert
    assert_invalid_outputs(result, n_linpara)


def test_first_column_all_zero_returns_invalid_outputs(noisy_gaussian_instrument):
    """If the companion column is identically zero, nothing can be fitted.

    The first column of M is by convention the companion flux.  A column of
    zeros means the companion is unconstrained by the data, so fitfm bails out
    rather than returning a meaningless amplitude.
    """
    # Arrange
    def zero_companion_fm_func(nonlin_paras, instrument, **fm_paras):
        d, M, s = gaussian_fm_func(nonlin_paras, instrument, **fm_paras)
        M = M.copy()
        M[:, 0] = 0.0
        return d, M, s

    # Act
    result = fitfm(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=noisy_gaussian_instrument,
        fm_func=zero_companion_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
    )

    # Assert
    assert_invalid_outputs(result, 2)


def test_all_zero_columns_are_dropped_and_returned_as_nan(noisy_gaussian_instrument):
    """Zero columns are removed from the fit and reported as NaN in the outputs.

    The remaining parameters must be identical to a fit that never included the
    dead column in the first place.
    """
    # Arrange: a third, entirely empty basis vector appended to the model
    def padded_fm_func(nonlin_paras, instrument, **fm_paras):
        d, M, s = gaussian_fm_func(nonlin_paras, instrument, **fm_paras)
        M = np.hstack([M, np.zeros((M.shape[0], 1))])
        return d, M, s

    common = dict(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=noisy_gaussian_instrument,
        fm_paras={"sigma": GAUSS_SIGMA},
    )

    # Act
    reference = fitfm(fm_func=gaussian_fm_func, **common)
    padded = fitfm(fm_func=padded_fm_func, **common)

    # Assert
    assert padded[3].shape == (3,)
    assert np.isnan(padded[3][2]) and np.isnan(padded[4][2])
    np.testing.assert_allclose(padded[3][:2], reference[3], rtol=1e-12)
    np.testing.assert_allclose(padded[4][:2], reference[4], rtol=1e-12)
    assert padded[0] == pytest.approx(reference[0], rel=1e-12)
    assert padded[2] == pytest.approx(reference[2], rel=1e-12)


def test_singular_design_matrix_returns_invalid_outputs(noisy_gaussian_instrument):
    """A rank-deficient model matrix is caught and reported as invalid outputs.

    Two identical (non-zero) columns are not filtered by the zero-column check,
    so M^T M is singular and the covariance inversion raises; fitfm traps this
    and returns the sentinel values rather than propagating the exception.
    """
    # Arrange
    def degenerate_fm_func(nonlin_paras, instrument, **fm_paras):
        d, M, s = gaussian_fm_func(nonlin_paras, instrument, **fm_paras)
        duplicated = np.vstack([M[:, 0], M[:, 0]]).T
        return d, duplicated, s

    # Act
    result = fitfm(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=noisy_gaussian_instrument,
        fm_func=degenerate_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
        computeH0=False,
    )

    # Assert
    assert_invalid_outputs(result, 2)


def single_parameter_fm_func(nonlin_paras, instrument, **fm_paras):
    """Gaussian amplitude only: exactly one linear parameter (as in the tutorial)."""
    d, M, s = gaussian_fm_func(nonlin_paras, instrument, **fm_paras)
    return d, M[:, :1], s


def test_single_linear_parameter_with_computeH0_raises(noisy_gaussian_instrument):
    """A one-parameter model cannot support the H0 test and is rejected.

    Note that ``Warning`` is an exception class, so this ``raise Warning(...)``
    is a hard error rather than a soft warning.  Callers with a single linear
    parameter, such as the Gaussian example in the breads tutorial, must pass
    ``computeH0=False`` explicitly.
    """
    # Act / Assert
    with pytest.raises(Warning, match="cannot test H0 hypothesis"):
        fitfm(
            nonlin_paras=[GAUSS_MU_TRUE],
            dataobj=noisy_gaussian_instrument,
            fm_func=single_parameter_fm_func,
            fm_paras={"sigma": GAUSS_SIGMA},
            computeH0=True,
        )


def test_single_linear_parameter_with_computeH0_false_works(
    noiseless_gaussian_instrument,
):
    """With computeH0=False a single-parameter model fits and returns NaN for H0."""
    # Arrange: the data contain a constant background that this model cannot
    # represent, so the recovered amplitude is only approximately the truth.
    instrument = noiseless_gaussian_instrument

    # Act
    log_prob_value, log_prob_H0, rchi2, linparas, linparas_err = fitfm(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=instrument,
        fm_func=single_parameter_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
        computeH0=False,
    )

    # Assert
    assert linparas.shape == (1,)
    assert np.isnan(log_prob_H0)
    assert np.isfinite(log_prob_value)
    assert np.isfinite(linparas[0]) and np.isfinite(linparas_err[0])
    assert linparas[0] == pytest.approx(GAUSS_AMPLITUDE_TRUE, rel=0.5)
    assert rchi2 > 0


def test_finite_bounds_raise_warning(noisy_gaussian_instrument):
    """Any finite bound is rejected because it invalidates the marginalization.

    As with the H0 check, ``Warning`` is raised as an exception, so in practice
    the ``bounds`` argument can only be used with infinite limits.
    """
    # Arrange: a non-negativity constraint on the companion flux
    bounds = ([0.0, -np.inf], [np.inf, np.inf])

    # Act / Assert
    with pytest.raises(Warning, match="only theoretically accurate"):
        fitfm(
            nonlin_paras=[GAUSS_MU_TRUE],
            dataobj=noisy_gaussian_instrument,
            fm_func=gaussian_fm_func,
            fm_paras={"sigma": GAUSS_SIGMA},
            bounds=bounds,
        )


def test_infinite_bounds_equivalent_to_none(noisy_gaussian_instrument):
    """Explicit infinite bounds give exactly the same answer as bounds=None."""
    # Arrange
    common = dict(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=noisy_gaussian_instrument,
        fm_func=gaussian_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
    )
    infinite_bounds = ([-np.inf, -np.inf], [np.inf, np.inf])

    # Act
    default = fitfm(bounds=None, **common)
    explicit = fitfm(bounds=infinite_bounds, **common)

    # Assert
    assert explicit[0] == pytest.approx(default[0], rel=1e-12)
    assert explicit[1] == pytest.approx(default[1], rel=1e-12)
    assert explicit[2] == pytest.approx(default[2], rel=1e-12)
    np.testing.assert_allclose(explicit[3], default[3], rtol=1e-12)
    np.testing.assert_allclose(explicit[4], default[4], rtol=1e-12)


def test_bounds_argument_is_not_mutated(noisy_gaussian_instrument):
    """fitfm copies the bounds it is given rather than modifying them in place."""
    # Arrange
    lower = [-np.inf, -np.inf]
    upper = [np.inf, np.inf]

    # Act
    fitfm(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=noisy_gaussian_instrument,
        fm_func=gaussian_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
        bounds=(lower, upper),
    )

    # Assert
    assert lower == [-np.inf, -np.inf]
    assert upper == [np.inf, np.inf]


# --------------------------------------------------------------------------
# 35-42: Wrapper functions (log_prob, nlog_prob, combined_log_prob)
# --------------------------------------------------------------------------

def test_log_prob_matches_fitfm_first_output(noisy_gaussian_instrument):
    """log_prob returns exactly the first output of fitfm with computeH0=False."""
    # Arrange
    common = dict(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=noisy_gaussian_instrument,
        fm_func=gaussian_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
    )

    # Act
    wrapper_value = log_prob(**common)
    direct_value = fitfm(computeH0=False, **common)[0]

    # Assert
    assert wrapper_value == pytest.approx(direct_value, rel=1e-12)


def test_log_prob_respects_scale_noise_flag(noisy_gaussian_instrument):
    """The scale_noise flag is forwarded through to fitfm."""
    # Arrange
    common = dict(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=noisy_gaussian_instrument,
        fm_func=gaussian_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
    )

    # Act
    scaled = log_prob(scale_noise=True, **common)
    unscaled = log_prob(scale_noise=False, **common)

    # Assert
    assert scaled == pytest.approx(fitfm(computeH0=False, scale_noise=True, **common)[0])
    assert unscaled == pytest.approx(
        fitfm(computeH0=False, scale_noise=False, **common)[0]
    )
    assert scaled != unscaled


def test_log_prob_adds_nonlinear_prior(noisy_gaussian_instrument):
    """A non-linear prior is added to the marginalized log likelihood."""
    # Arrange
    prior_value = -2.5
    common = dict(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=noisy_gaussian_instrument,
        fm_func=gaussian_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
    )
    recorded_calls = []

    def flat_prior(nonlin_paras):
        recorded_calls.append(list(nonlin_paras))
        return prior_value

    # Act
    without_prior = log_prob(**common)
    with_prior = log_prob(nonlin_lnprior_func=flat_prior, **common)

    # Assert
    assert with_prior == pytest.approx(without_prior + prior_value, rel=1e-12)
    assert recorded_calls == [[GAUSS_MU_TRUE]]


def test_log_prob_prior_can_veto_a_parameter(noisy_gaussian_instrument):
    """A -inf prior drives the total log probability to -inf."""
    # Act
    result = log_prob(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=noisy_gaussian_instrument,
        fm_func=gaussian_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
        nonlin_lnprior_func=lambda paras: -np.inf,
    )

    # Assert
    assert result == -np.inf


def test_log_prob_returns_neg_inf_when_fm_raises(noisy_gaussian_instrument):
    """Any exception inside the fit is swallowed and reported as -inf.

    This keeps optimizers and samplers from crashing when they wander into
    parameter regions where the forward model cannot be evaluated.
    """
    # Arrange
    def exploding_fm_func(nonlin_paras, instrument, **fm_paras):
        raise RuntimeError("forward model failed")

    # Act
    result = log_prob(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=noisy_gaussian_instrument,
        fm_func=exploding_fm_func,
        fm_paras={},
    )

    # Assert
    assert result == -np.inf


def test_log_prob_handles_single_linear_parameter_without_raising(
    noisy_gaussian_instrument,
):
    """log_prob works with a one-parameter model because it disables the H0 test.

    log_prob calls fitfm with computeH0=False, so the single-parameter guard is
    not triggered and a finite value is returned; a genuinely failing model is
    what produces -inf.  This test pins down that distinction.
    """
    # Act
    single_param_value = log_prob(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=noisy_gaussian_instrument,
        fm_func=single_parameter_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
    )

    # Assert
    assert np.isfinite(single_param_value)


def test_nlog_prob_is_negative_of_log_prob(noisy_gaussian_instrument):
    """nlog_prob simply negates log_prob, including with a prior."""
    # Arrange
    common = dict(
        nonlin_paras=[GAUSS_MU_TRUE],
        dataobj=noisy_gaussian_instrument,
        fm_func=gaussian_fm_func,
        fm_paras={"sigma": GAUSS_SIGMA},
    )

    # Act / Assert
    assert nlog_prob(**common) == pytest.approx(-log_prob(**common), rel=1e-12)
    assert nlog_prob(nonlin_lnprior_func=lambda p: -1.25, **common) == pytest.approx(
        -log_prob(nonlin_lnprior_func=lambda p: -1.25, **common), rel=1e-12
    )


def test_nlog_prob_minimized_at_true_parameter():
    """Minimizing nlog_prob over the grid recovers the true Gaussian centre."""
    # Arrange
    rng = np.random.default_rng(71)
    x = np.linspace(0.0, 10.0, 201)
    sigma_noise = 0.2
    instrument = make_instrument(
        x,
        gaussian_truth(x) + rng.normal(0.0, sigma_noise, x.size),
        np.full_like(x, sigma_noise),
    )
    mu_grid = np.linspace(4.0, 6.5, 251)

    # Act
    curve = np.array(
        [
            nlog_prob([mu], instrument, gaussian_fm_func, {"sigma": GAUSS_SIGMA})
            for mu in mu_grid
        ]
    )

    # Assert
    assert mu_grid[np.argmin(curve)] == pytest.approx(GAUSS_MU_TRUE, abs=0.1)
    assert curve.min() < curve[0] and curve.min() < curve[-1]


def test_combined_log_prob_sums_individual_log_probs():
    """combined_log_prob is the sum of the per-dataset log probabilities."""
    # Arrange: two independent noise realizations of the same underlying source
    rng = np.random.default_rng(73)
    x_a = np.linspace(0.0, 10.0, 101)
    x_b = np.linspace(2.0, 8.0, 61)
    instrument_a = make_instrument(
        x_a, gaussian_truth(x_a) + rng.normal(0.0, 0.2, x_a.size), np.full_like(x_a, 0.2)
    )
    instrument_b = make_instrument(
        x_b, gaussian_truth(x_b) + rng.normal(0.0, 0.3, x_b.size), np.full_like(x_b, 0.3)
    )
    fm_paras = {"sigma": GAUSS_SIGMA}

    # Act
    combined = combined_log_prob(
        [GAUSS_MU_TRUE],
        [instrument_a, instrument_b],
        [gaussian_fm_func, gaussian_fm_func],
        [fm_paras, fm_paras],
    )
    individual_sum = log_prob(
        [GAUSS_MU_TRUE], instrument_a, gaussian_fm_func, fm_paras
    ) + log_prob([GAUSS_MU_TRUE], instrument_b, gaussian_fm_func, fm_paras)

    # Assert
    assert combined == pytest.approx(individual_sum, rel=1e-12)


def test_combined_log_prob_single_dataset_matches_log_prob(noisy_gaussian_instrument):
    """A one-element list degenerates to a plain log_prob call."""
    # Arrange
    fm_paras = {"sigma": GAUSS_SIGMA}

    # Act
    combined = combined_log_prob(
        [GAUSS_MU_TRUE], [noisy_gaussian_instrument], [gaussian_fm_func], [fm_paras]
    )
    single = log_prob(
        [GAUSS_MU_TRUE], noisy_gaussian_instrument, gaussian_fm_func, fm_paras
    )

    # Assert
    assert combined == pytest.approx(single, rel=1e-12)


def test_combined_log_prob_applies_prior_once_per_dataset(noisy_gaussian_instrument):
    """The non-linear prior is added once for every data object.

    combined_log_prob forwards the prior into each per-dataset log_prob call, so
    with N data objects the prior is counted N times rather than once.  This
    documents the current behaviour: callers combining many datasets should be
    aware that the prior is effectively raised to the Nth power.
    """
    # Arrange
    prior_value = -2.0
    fm_paras = {"sigma": GAUSS_SIGMA}
    data_objects = [noisy_gaussian_instrument] * 3

    # Act
    with_prior = combined_log_prob(
        [GAUSS_MU_TRUE],
        data_objects,
        [gaussian_fm_func] * 3,
        [fm_paras] * 3,
        nonlin_lnprior_func=lambda paras: prior_value,
    )
    without_prior = combined_log_prob(
        [GAUSS_MU_TRUE], data_objects, [gaussian_fm_func] * 3, [fm_paras] * 3
    )

    # Assert
    assert with_prior == pytest.approx(
        without_prior + len(data_objects) * prior_value, rel=1e-12
    )


def test_combined_log_prob_peaks_at_true_parameter():
    """Combining two datasets still peaks at the true non-linear parameter."""
    # Arrange
    rng = np.random.default_rng(79)
    x_a = np.linspace(0.0, 10.0, 101)
    x_b = np.linspace(0.0, 10.0, 101)
    instrument_a = make_instrument(
        x_a, gaussian_truth(x_a) + rng.normal(0.0, 0.2, x_a.size), np.full_like(x_a, 0.2)
    )
    instrument_b = make_instrument(
        x_b, gaussian_truth(x_b) + rng.normal(0.0, 0.2, x_b.size), np.full_like(x_b, 0.2)
    )
    fm_paras = {"sigma": GAUSS_SIGMA}
    mu_grid = np.linspace(4.0, 6.5, 126)

    # Act
    curve = np.array(
        [
            combined_log_prob(
                [mu],
                [instrument_a, instrument_b],
                [gaussian_fm_func, gaussian_fm_func],
                [fm_paras, fm_paras],
            )
            for mu in mu_grid
        ]
    )

    # Assert
    assert mu_grid[np.argmax(curve)] == pytest.approx(GAUSS_MU_TRUE, abs=0.1)


def test_combined_log_prob_ignores_its_bounds_argument(noisy_gaussian_instrument):
    """combined_log_prob accepts a bounds argument but never forwards it.

    The implementation hard-codes ``bounds=None`` in the per-dataset log_prob
    call, so a caller-supplied bounds tuple has no effect.  Were it forwarded,
    the finite lower bound below would raise inside fitfm.  This test documents
    the current behaviour.
    """
    # Arrange
    fm_paras = {"sigma": GAUSS_SIGMA}
    finite_bounds = ([0.0, -np.inf], [np.inf, np.inf])

    # Act
    with_bounds = combined_log_prob(
        [GAUSS_MU_TRUE],
        [noisy_gaussian_instrument],
        [gaussian_fm_func],
        [fm_paras],
        bounds=finite_bounds,
    )
    without_bounds = combined_log_prob(
        [GAUSS_MU_TRUE], [noisy_gaussian_instrument], [gaussian_fm_func], [fm_paras]
    )

    # Assert
    assert np.isfinite(with_bounds)
    assert with_bounds == pytest.approx(without_bounds, rel=1e-12)
