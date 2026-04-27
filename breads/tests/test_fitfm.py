from breads.fit import fitfm, log_prob
from breads.instruments import Instrument
from breads.grid_search import grid_search
import numpy as np
import matplotlib.pyplot as plt
import emcee
from breads.utils import get_err_from_posterior

def test_linear_model():
    data_obj = Instrument()  # Create a custom (empty for now) instrument instance to be filled namually later

    np.random.seed(5)  # For reproducibility

    slope = 2.0  # True slope of the linear trend, that we will try to estimate
    intercept = 5  # True intercept of the linear trend, that we will try to estimate
    noise_stddev = 1.0  # standard deviation of the Gaussian noise added to the data
    my_wvs = np.linspace(0, 10, 20)  # Wavelengths from 0 to 10. But here this is just the x-axis for our linear trend.

    # Fill the instrument instance with our synthetic data
    data_obj.manual_data_entry(
        wavelengths=my_wvs,  # Wavelengths from 0 to 10
        data=slope * my_wvs + intercept + np.random.normal(0, noise_stddev, my_wvs.size),
        noise=np.ones_like(my_wvs) * noise_stddev,
        bad_pixels=np.ones_like(my_wvs)
    )

    def linear_fm_func(nonlin_paras, data_obj: "Instrument", **fm_paras):
        """
        Build a linear forward model for a data object from an Instrument class.
        The linear parameters of the model are [slope, intercept].

        Parameters
        ----------
        nonlin_paras : array-like
            Non-linear parameters (not used in this simple linear model,but kept for consistency).
        data_obj : Instrument
            An instance of the Instrument class containing wavelengths and data.

        Returns
        -------
        d : np.ndarray
            The noisy data flattended.
        M : np.ndarray
            The design matrix for linear fitting (wavelengths and constant term).
        s : np.ndarray or None
            Uncertainties from instrument.noise (standard deviation).
        """

        # Flatten the data for 1D linear fit
        d = data_obj.data.flatten()
        x = data_obj.wavelengths.flatten()

        s = data_obj.noise.flatten()

        # Design matrix: linear term + constant
        M = np.vstack([x, np.ones_like(x)]).T

        return d, M, s

    results = fitfm(
        nonlin_paras=[],
        dataobj=data_obj,
        fm_func=linear_fm_func,
        fm_paras={},
        marginalize_noise_scaling=False, scale_noise=False
    )
    bestfit_log_prob, log_prob_H0, rchi2, linparas, linparas_err = results

    def log_likelihood_test(paras, data_obj):
        """
        Returns log-likelihood for a proposed mu using the Gaussian FM function.
        """
        a, b = paras  # extract scalar mu

        d, M, s = linear_fm_func([], data_obj)
        model = np.dot(M, [a, b])  # linear model prediction
        # The previous line is equivalent to:
        # model = a * data_obj.wavelengths + b  # linear model prediction

        # Compare model to observed data
        chi2 = np.sum(((d - model) / s) ** 2)

        # Log-likelihood
        return -0.5 * chi2

    # MCMC setup
    nwalkers = 50
    nsteps = 1000

    # Start walkers near the approximate mu
    paras_init = np.array([0, 0])
    ndim = np.size(paras_init)
    p0 = paras_init + 1e-2 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood_test, args=[data_obj])

    # Run MCMC
    sampler.run_mcmc(p0, nsteps, progress=True)
    samples = sampler.get_chain(flat=True)

    # Plot MCMC posterior as histogram
    # Discard burn-in and extract chains
    a_samples = samples[(nsteps // 4 * nwalkers)::, 0]
    b_samples = samples[(nsteps // 4 * nwalkers)::, 1]

    # print("Case 1: Fitting for slope and intercept (accurate noise model without noise scaling, no regularization):")
    # print("BREADS Slope (m):", linparas[0], "±", linparas_err[0])
    # print("BREADS Intercept (b):", linparas[1], "±", linparas_err[1])
    # print("Reduced chi2:", rchi2)
    #
    # print("EMCEE Slope (m):", np.median(a_samples), "±", np.std(a_samples))
    # print("EMCEE Intercept (b):", np.median(b_samples), "±", np.std(b_samples))

    assert np.abs(linparas[0] - slope) < 3 * linparas_err[0], "BREADS slope estimate is not within 3 sigma of true value"
    assert np.abs(linparas[1] - intercept) < 3 * linparas_err[1], "BREADS intercept estimate is not within 3 sigma of true value"
    assert np.abs(np.median(a_samples) - slope) < 3 * np.std(a_samples), "EMCEE slope estimate is not within 3 sigma of true value"
    assert np.abs(np.median(b_samples) - intercept) < 3 * np.std(b_samples), "EMCEE intercept estimate is not within 3 sigma of true value"

    assert np.abs(linparas_err[0] - np.std(a_samples)) < 0.1*np.std(a_samples), "BREADS slope uncertainty is not similar enough to EMCEE slope uncertainty"
    assert np.abs(linparas_err[1] - np.std(b_samples)) < 0.1*np.std(b_samples), "BREADS intercept uncertainty is not similar enough to EMCEE intercept uncertainty"

def test_nonlinear_para():
    # Case 2: introduction to non-linear parameters
    # Fitting for intercept, and position and amplitude of a Gaussian (accurate noise model without noise scaling, no regularization)
    # The gaussian can be thought of as a planet signal, and the intercept can be thought of as a linear parameter corresponding to the stellar flux.
    # This is to illustrate non-linear parameters.
    data_obj = Instrument()  # Create a custom instrument instance to be filled manually

    np.random.seed(5)  # For reproducibility

    intercept = 5
    noise_stddev = 1.0
    gauss_amplitude = 8.0
    gauss_center = 5.0
    gauss_width = 1.0
    # Wavelengths from 0 to 10. But here this is just the x-axis for our linear trend.
    my_wvs = np.linspace(0, 10, 20)
    gauss0 = gauss_amplitude * np.exp(-(my_wvs - gauss_center) ** 2 / (2 * gauss_width ** 2))

    data_obj.manual_data_entry(
        wavelengths=my_wvs,  # Wavelengths from 0 to 10
        data=np.full(my_wvs.shape, intercept) + gauss0 + np.random.normal(0, noise_stddev, my_wvs.size),
        noise=np.ones_like(my_wvs) * noise_stddev,
        bad_pixels=np.ones_like(my_wvs)
    )

    def intro_nonlin_fm(nonlin_paras, data_obj: "Instrument", gauss_width=None):
        """
        Build a linear forward model for case 2:
        Fitting for intercept, and position and amplitude of a Gaussian (accurate noise model without noise scaling, no regularization)
        The gaussian can be thought of as a planet signal, and the intercept can be thought of as a linear parameter corresponding to the stellar flux.

        The linear parameters of the model are [gauss_amplitude, slope].
        The non-linear parameter of the model are [gauss_center].

        Parameters
        ----------
        nonlin_paras : array-like
            Non-linear parameters: [gauss_center].
        data_obj : Instrument
            An instance of the Instrument class containing wavelengths and data.
        gauss_width : float
                The width of the Gaussian. This is fixed for simplicity, but it could also be a non-linear parameter.

        Returns
        -------
        d : np.ndarray
            The noisy data flattended.
        M : np.ndarray
            The design matrix for linear fitting (wavelengths and constant term).
        s : np.ndarray or None
            Uncertainties from instrument.noise (standard deviation).
        """

        # Flatten the data for 1D linear fit
        y = data_obj.data.flatten()
        x = data_obj.wavelengths.flatten()
        s = data_obj.noise.flatten()

        gauss_center = nonlin_paras[0]
        col0 = np.exp(-(x - gauss_center) ** 2 / (2 * gauss_width ** 2))

        # Design matrix: linear term + constant
        M = np.vstack([col0, np.ones_like(x)]).T

        return y, M, s

    # If we already know gauss_center, we can simply fit for the gaussian amplitude and intercept.
    results = fitfm(
        nonlin_paras=[gauss_center],
        dataobj=data_obj,
        fm_func=intro_nonlin_fm,
        fm_paras={'gauss_width': gauss_width},
        marginalize_noise_scaling=False, scale_noise=False
    )
    bestfit_log_prob, log_prob_H0, rchi2, linparas, linparas_err = results

    # print("BREADS gaussian amplitude (fixed gauss_center):", linparas[0], "±", linparas_err[0])
    # print("BREADS Intercept (fixed gauss_center):", linparas[1], "±", linparas_err[1])
    # print("Reduced chi2 (fixed gauss_center):", rchi2)

    assert np.abs(linparas[0] - gauss_amplitude) < 3 * linparas_err[0], "BREADS gaussian amplitude estimate is not within 3 sigma of true value (fixed gauss_center)"
    assert np.abs(linparas[1] - intercept) < 3 * linparas_err[1], "BREADS intercept estimate is not within 3 sigma of true value (fixed gauss_center)"

    # But here we also want to get the posterior for gauss_center
    # We can do this by running a grid search over gauss_center.

    # Let's define a grid of gauss_center values around the true value.
    # We will estimate the log probability of the model for each value of gauss_center.
    gauss_center_grid = np.linspace(gauss_center - 2, gauss_center + 2, 100)

    # grid_search effectively runs fitfm for each value of gauss_center in the grid, and returns the log probability and best-fit linear parameters for each value of gauss_center.
    bestfit_log_prob, _, rchi2, linparas, linparas_err = grid_search(
        para_vecs=[gauss_center_grid],
        dataobj=data_obj,
        fm_func=intro_nonlin_fm,
        fm_paras={"gauss_width": gauss_width},
        numthreads=None,
        bounds=None
    )

    # linparas is an array of shape (len(gauss_center_grid), N_linpara) containing the best-fit linear parameters for each value of gauss_center in the grid.
    # linparas_err is an array of shape (len(gauss_center_grid), N_linpara) containing the uncertainties of the best-fit linear parameters for each value of gauss_center in the grid.

    prob = np.exp(bestfit_log_prob - np.max(bestfit_log_prob))
    best_fit, left_err, right_err = get_err_from_posterior(gauss_center_grid, prob)
    # print("BREADS gauss_center (left/right uncertainties):", best_fit, "-", left_err, "+", right_err)

    assert np.abs(best_fit  - gauss_center)  < 3 * max(left_err, right_err), "BREADS gauss_center estimate is not within 3 sigma of true value (grid search)"

    # Now let's compare the BREADS posterior for the linear parameters to the MCMC posterior.
    # Note that the MCMC provides marginalized posteriors for all three parameters (gauss_center, gauss_amplitude, and intercept), while the grid search only really provides a marginalized posterior for gauss_center only.
    # The grid search does provide the best-fit linear parameters for each value of gauss_center, but these are not marginalized over gauss_center, they are only marginalized over the linear parameters.
    # So linparas_err is not directly comparable to the MCMC posteriors for the linear parameters.

    def log_likelihood_case2(paras, data_obj):
        """
        Returns log-likelihood for case 2.
        We are here fitting for 3 parameters: gauss_center, gauss_amplitude, and intercept.
        gauss_center is a non-linear parameters, while gauss_amplitude and intercept are linear parameters.
        Unlike in BREADS, they are all treated the same in the MCMC.
        """
        gauss_center, gauss_amplitude, intercept = paras

        d, M, s = intro_nonlin_fm([gauss_center], data_obj, gauss_width=gauss_width)
        model = np.dot(M, [gauss_amplitude, intercept])  # linear model prediction

        # Log-likelihood
        log_like = -0.5 * np.sum(((d - model) / s) ** 2)

        return log_like

    # MCMC setup
    nwalkers = 50
    nsteps = 1000

    # Start walkers
    paras_init = np.array([gauss_center, gauss_amplitude, intercept])
    ndim = np.size(paras_init)
    p0 = paras_init + 1e-2 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood_case2, args=[data_obj])

    # Run MCMC
    sampler.run_mcmc(p0, nsteps, progress=True)
    samples = sampler.get_chain(flat=True)

    # Plot MCMC posterior as histogram
    # Discard burn-in and extract chains
    gauss_center_samples = samples[(nsteps // 4 * nwalkers)::, 0]
    gauss_ampl_samples = samples[(nsteps // 4 * nwalkers)::, 1]
    intercept_samples = samples[(nsteps // 4 * nwalkers)::, 2]
    # print("EMCEE gaussian center:", np.median(gauss_center_samples), "±", np.std(gauss_center_samples))
    # print("EMCEE gaussian amplitude:", np.median(gauss_ampl_samples), "±", np.std(gauss_ampl_samples))
    # print("EMCEE Intercept:", np.median(intercept_samples), "±", np.std(intercept_samples))

    assert np.abs(np.median(gauss_center_samples) - gauss_center) < 3 * np.std(gauss_center_samples), "EMCEE gauss_center estimate is not within 3 sigma of true value (case 2)"
    assert np.abs(np.median(gauss_ampl_samples) - gauss_amplitude) < 3 * np.std(gauss_ampl_samples), "EMCEE gaussian amplitude estimate is not within 3 sigma of true value (case 2)"

    assert np.abs(np.std(gauss_center_samples) - (left_err+ right_err)/2.) < 0.25*(left_err+ right_err)/2., "EMCEE gauss_center uncertainty is not similar enough to BREADS gauss_center uncertainty (case 2)"

    # We can also use the breads.fit.log_prob() function in an MCMC sampler.
    # log_prob() simply wraps around fitfm(), but only return the log probability.
    # The major difference in using BREADS' log_prob() (instead of the MCMC likelihood defined previously) is that we are fitting for a single parameter (gauss_center), instead of 3 parameters in the previous MCMC (gauss_center, gauss_amplitude, and intercept).
    # BREADS takes care of marginalizing over the linear parameters (gauss_amplitude and intercept) on the fly.
    # Note that in this demonstration, it is numerically not really worth it to use BREADS, but the advantage of this method comes when fitting for 100s or 1000s of linear parameters, which cannot be fitted inside an MCMC.

    # Here is how it can be done:
    nwalkers = 50
    nsteps = 100

    # Start walkers
    fm_paras = {'gauss_width': gauss_width}
    paras_init = np.array([gauss_center])
    ndim = np.size(paras_init)
    p0 = paras_init + 1e-2 * np.random.randn(nwalkers, ndim)
    sampler_BREADS_like = emcee.EnsembleSampler(nwalkers, ndim, log_prob,
                                                args=[data_obj, intro_nonlin_fm, fm_paras, None, None, True, False])
    sampler_BREADS_like.run_mcmc(p0, nsteps, progress=True)
    samples_BREADS_like = sampler_BREADS_like.get_chain(flat=True)
    gauss_center_samples_BREADS_like = samples_BREADS_like[(nsteps // 4 * nwalkers)::, 0]

    # print("BREADS-like MCMC gaussian center:", np.median(gauss_center_samples_BREADS_like), "±", np.std(gauss_center_samples_BREADS_like))
    assert np.abs(np.median(gauss_center_samples_BREADS_like) - gauss_center) < 3 * np.std(gauss_center_samples_BREADS_like), "BREADS-like MCMC gauss_center estimate is not within 3 sigma of true value (case 2)"
    assert np.abs(np.std(gauss_center_samples_BREADS_like) - max(left_err, right_err)) < 0.25*max(left_err, right_err), "BREADS-like MCMC gauss_center uncertainty is not similar enough to BREADS grid search gauss_center uncertainty (case 2)"


def test_noise_scaling():
    # Case 3: Noise scaling
    # In this example, we assume that the noise is not well known.
    # For example, the noise might be underestimated, which is common in high-contrast imaging.
    # While the data will have noise_stddev = 1, we will "lie" to the data object and give it a noise vector with noise_stddev = 0.1.
    # We simply rescale the noise by a "noise scaling factor", which is the sqrt of the reduced chi2 of the fit.
    # First we will look at the case where we do not marginalize over the noise scaling factor, but simply rescale everything.
    # Then, we will show what happens when the noise is marginalized over as well.

    data_obj = Instrument()  # Create a custom instrument instance to be filled manually

    np.random.seed(5)  # For reproducibility

    intercept = 5
    noise_stddev = 1.0
    gauss_amplitude = 8.0
    gauss_center = 5.0
    gauss_width = 1.0
    # Wavelengths from 0 to 10. But here this is just the x-axis for our linear trend.
    my_wvs = np.linspace(0, 10, 20)
    gauss0 = gauss_amplitude * np.exp(-(my_wvs - gauss_center) ** 2 / (2 * gauss_width ** 2))

    data_obj.manual_data_entry(
        wavelengths=my_wvs,  # Wavelengths from 0 to 10
        data=np.full(my_wvs.shape, intercept) + gauss0 + np.random.normal(0, noise_stddev, my_wvs.size),
        noise=np.ones_like(my_wvs) * noise_stddev / 5.,
        # This models the case where the noise is underestimated by a factor of 5. The fit should find a noise scaling factor of ~5 to compensate for this.
        bad_pixels=np.ones_like(my_wvs)
    )

    # Here we can reuse the same forward model as in case 2, since the model is the same, we are just changing the noise.
    def intro_nonlin_fm(nonlin_paras, data_obj: "Instrument", gauss_width=gauss_width):
        """
        Build a linear forward model for case 2:
        Fitting for intercept, and position and amplitude of a Gaussian (accurate noise model without noise scaling, no regularization)
        The gaussian can be thought of as a planet signal, and the intercept can be thought of as a linear parameter corresponding to the stellar flux.

        The linear parameters of the model are [gauss_amplitude, slope].
        The non-linear parameter of the model are [gauss_center].

        Parameters
        ----------
        nonlin_paras : array-like
            Non-linear parameters: [gauss_center].
        data_obj : Instrument
            An instance of the Instrument class containing wavelengths and data.
        gauss_width : float
                The width of the Gaussian. This is fixed for simplicity, but it could also be a non-linear parameter.

        Returns
        -------
        d : np.ndarray
            The noisy data flattended.
        M : np.ndarray
            The design matrix for linear fitting (wavelengths and constant term).
        s : np.ndarray or None
            Uncertainties from instrument.noise (standard deviation).
        """

        # Flatten the data for 1D linear fit
        y = data_obj.data.flatten()
        x = data_obj.wavelengths.flatten()
        s = data_obj.noise.flatten()

        gauss_center = nonlin_paras[0]
        col0 = np.exp(-(x - gauss_center) ** 2 / (2 * gauss_width ** 2))

        # Design matrix: linear term + constant
        M = np.vstack([col0, np.ones_like(x)]).T

        return y, M, s

    # Fitting without noise rescaling scale_noise=False will lead to much smaller uncertainties on the linear parameters, and a much larger reduced chi2.
    results = fitfm(
        nonlin_paras=[gauss_center],
        dataobj=data_obj,
        fm_func=intro_nonlin_fm,
        fm_paras={'gauss_width': gauss_width},
        marginalize_noise_scaling=False, scale_noise=False
    )
    bestfit_log_prob, log_prob_H0, rchi2, linparas, linparas_err = results

    # print("BREADS gaussian amplitude (no noise scaling):", linparas[0], "±", linparas_err[0])
    # print("BREADS Intercept (no noise scaling):", linparas[1], "±", linparas_err[1])
    # print("Reduced chi2 (no noise scaling):", rchi2)

    assert np.abs(linparas[0] - gauss_amplitude) > 3 * linparas_err[0], "error bars should be wrong since the noise is understimated"
    assert np.abs(linparas[1] - intercept) > 3 * linparas_err[1], "error bars should be wrong since the noise is understimated"
    assert (rchi2-25)/25 < 0.1, "Reduced chi2 should be about 25 when noise is underestimated by a factor 5; no noise scaling is applied here"

    # We can correct for this by enabling scale_noise=True:
    results = fitfm(
        nonlin_paras=[gauss_center],
        dataobj=data_obj,
        fm_func=intro_nonlin_fm,
        fm_paras={'gauss_width': gauss_width},
        marginalize_noise_scaling=False, scale_noise=True
    )
    bestfit_log_prob, log_prob_H0, rchi2, linparas, linparas_err = results
    BREADS_noise_scaling = np.sqrt(rchi2)

    # print("BREADS gaussian amplitude (WITH noise scaling):", linparas[0], "±", linparas_err[0])
    # print("BREADS Intercept (WITH noise scaling):", linparas[1], "±", linparas_err[1])
    # print("Reduced chi2 (WITH noise scaling):", rchi2)

    assert np.abs(linparas[0] - gauss_amplitude) < 3 * linparas_err[0], "BREADS gaussian amplitude estimate is not within 3 sigma of true value (WITH noise scaling)"
    assert np.abs(linparas[1] - intercept) < 3 * linparas_err[1], "BREADS intercept estimate is not within 3 sigma of true value (WITH noise scaling)"
    # the rchi2 is always provided "before the correction" for the noise scaling, so it should still be about 25, even when scale_noise=True.
    assert (rchi2 - 25) / 25 < 0.1, "Reduced chi2 should be about 25 when noise is underestimated by a factor 5"

    # Now we want to get the posterior for gauss_center with the noise scaling.
    # Although note that this is not statistically accurate, since we are not actually marginalizing over the noise scaling factor here.
    gauss_center_grid = np.linspace(gauss_center - 2, gauss_center + 2, 100)

    bestfit_log_prob, _, rchi2, linparas, linparas_err = grid_search(
        para_vecs=[gauss_center_grid],
        dataobj=data_obj,
        fm_func=intro_nonlin_fm,
        fm_paras={"gauss_width": gauss_width},
        numthreads=None,
        bounds=None,
        marginalize_noise_scaling=False, scale_noise=True
    )

    prob_noise_scaling = np.exp(bestfit_log_prob - np.max(bestfit_log_prob))
    best_fit, left_err, right_err = get_err_from_posterior(gauss_center_grid, prob_noise_scaling)
    # print("BREADS gauss_center (left/right uncertainties):", best_fit, "-", left_err, "+", right_err)

    assert np.abs(best_fit - gauss_center) < 3 * max(left_err, right_err), "BREADS gauss_center estimate is not within 3 sigma of true value (noise scaling)"

    # Now, let's enable the noise marginalization in BREADS.
    # BREADS implements an analytical marginalization over the noise scaling factor, which is equivalent to assuming a Jeffreys prior on the noise scaling factor and integrating it out.
    # Note 1: this only works without regularization.
    # Note 2: linparas and linparas_err WON'T be marginalized over the noise scaling factor. This only works on the bestfit_log_prob output to derive the posteriors of non-linear parameters.
    gauss_center_grid = np.linspace(gauss_center - 2, gauss_center + 2, 100)

    bestfit_log_prob, _, rchi2, linparas, linparas_err = grid_search(
        para_vecs=[gauss_center_grid],
        dataobj=data_obj,
        fm_func=intro_nonlin_fm,
        fm_paras={"gauss_width": gauss_width},
        numthreads=None,
        bounds=None,
        marginalize_noise_scaling=True, scale_noise=False
    )

    where_max_prob = np.argmax(bestfit_log_prob)

    prob_marginalized_noise_scaling = np.exp(bestfit_log_prob - np.max(bestfit_log_prob))

    best_fit, left_err, right_err = get_err_from_posterior(gauss_center_grid, prob_marginalized_noise_scaling)
    # print("BREADS gauss_center (marginalize_noise_scaling; left/right uncertainties):", best_fit, "-", left_err, "+",
    #       right_err)

    assert np.abs(best_fit - gauss_center) < 3 * max(left_err, right_err), "BREADS gauss_center estimate is not within 3 sigma of true value (marginalize_noise_scaling is True)"

    # Now let's compare with a noise scaling factor marginalized in a MCMC.

    def log_likelihood_case3(paras, data_obj):
        """
        Returns log-likelihood for case 3.
        Testing marginalization over noise scaling factor.
        """
        gauss_center, gauss_amplitude, intercept, noise_scaling = paras
        if noise_scaling < 0:
            return -np.inf  # We don't want negative noise scaling factors.

        d, M, s = intro_nonlin_fm([gauss_center], data_obj, gauss_width=gauss_width)
        model = np.dot(M, [gauss_amplitude, intercept])  # linear model prediction

        s_scaled = s * noise_scaling

        # We need to include the penalty for the noise scaling factor in the likelihood, since it is a free parameter that can make the likelihood arbitrarily high by making the noise very large.
        log_like = -0.5 * np.sum(((d - model) / s_scaled) ** 2) - np.sum(np.log(
            s_scaled))  # The second term is the penalty for the noise scaling factor, which comes from the normalization of the Gaussian likelihood.

        # Log-likelihood
        return log_like

    # MCMC setup
    nwalkers = 50
    nsteps = 1000

    # Start walkers
    paras_init = np.array([gauss_center, gauss_amplitude, intercept, 5])
    ndim = np.size(paras_init)
    p0 = paras_init + 1e-2 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood_case3, args=[data_obj])

    # Run MCMC
    sampler.run_mcmc(p0, nsteps, progress=True)
    samples = sampler.get_chain(flat=True)

    # Plot MCMC posterior as histogram
    # Discard burn-in and extract chains
    gauss_center_samples = samples[(nsteps // 4 * nwalkers)::, 0]
    gauss_ampl_samples = samples[(nsteps // 4 * nwalkers)::, 1]
    intercept_samples = samples[(nsteps // 4 * nwalkers)::, 2]
    noise_scaling_samples = samples[(nsteps // 4 * nwalkers)::, 3]
    # print("EMCEE gaussian center:", np.median(gauss_center_samples), "±", np.std(gauss_center_samples))
    # print("EMCEE gaussian amplitude:", np.median(gauss_ampl_samples), "±", np.std(gauss_ampl_samples))
    # print("EMCEE Intercept:", np.median(intercept_samples), "±", np.std(intercept_samples))
    # print("EMCEE noise scaling factor:", np.median(noise_scaling_samples), "±", np.std(noise_scaling_samples))

    assert np.abs(np.median(gauss_center_samples) - gauss_center) < 3 * np.std(gauss_center_samples), "EMCEE gauss_center estimate is not within 3 sigma of true value (case 3)"
    assert np.abs(np.median(gauss_ampl_samples) - gauss_amplitude) < 3 * np.std(gauss_ampl_samples), "EMCEE gaussian amplitude estimate is not within 3 sigma of true value (case 3)"
    assert np.abs(np.median(intercept_samples) - intercept) < 3 * np.std(intercept_samples), "EMCEE intercept estimate is not within 3 sigma of true value (case 3)"
    assert np.abs(np.median(noise_scaling_samples) - 5) < 3 * np.std(noise_scaling_samples), "EMCEE noise scaling factor estimate is not within 3 sigma of BREADS noise scaling factor estimate (case 3)"

    assert np.abs(np.std(gauss_center_samples) - max(left_err, right_err)) < 0.25*max(left_err, right_err), "EMCEE gauss_center uncertainty is not similar enough to BREADS gauss_center uncertainty (case 3)"


def test_regularization():
    # Case 4: regularization of linear parameters
    # We will start from case 2, but now we will add a regularization on the linear parameters.
    # Regularization is a way to include prior information on the linear parameters. Regularization and prior are synonymous here.

    data_obj = Instrument()  # Create a custom instrument instance to be filled manually

    np.random.seed(5)  # For reproducibility

    intercept = 5
    noise_stddev = 3.0  # increasing the noise in this case to better illustrate the effect of regularization. With low noise, the regularization won't have much effect since the data is already very constraining.
    gauss_amplitude = 8.0
    gauss_center = 5.0
    gauss_width = 1.0
    # Wavelengths from 0 to 10. But here this is just the x-axis for our linear trend.
    my_wvs = np.linspace(0, 10, 20)
    gauss0 = gauss_amplitude * np.exp(-(my_wvs - gauss_center) ** 2 / (2 * gauss_width ** 2))

    data_obj.manual_data_entry(
        wavelengths=my_wvs,  # Wavelengths from 0 to 10
        data=np.full(my_wvs.shape, intercept) + gauss0 + np.random.normal(0, noise_stddev, my_wvs.size),
        noise=np.ones_like(my_wvs) * noise_stddev,
        bad_pixels=np.ones_like(my_wvs)
    )

    def intro_regularization_fm(nonlin_paras, data_obj: "Instrument", gauss_width=None, d_reg=None, s_reg=None):
        """
        Build a linear forward model for case 2:
        Fitting for intercept, and position and amplitude of a Gaussian (accurate noise model without noise scaling, no regularization)
        The gaussian can be thought of as a planet signal, and the intercept can be thought of as a linear parameter corresponding to the stellar flux.

        The linear parameters of the model are [gauss_amplitude, slope].
        The non-linear parameter of the model are [gauss_center].

        Parameters
        ----------
        nonlin_paras : array-like
            Non-linear parameters: [gauss_center].
        data_obj : Instrument
            An instance of the Instrument class containing wavelengths and data.
        gauss_width : float
                The width of the Gaussian. This is fixed for simplicity, but it could also be a non-linear parameter.

        Returns
        -------
        d : np.ndarray
            The noisy data flattended.
        M : np.ndarray
            The design matrix for linear fitting (wavelengths and constant term).
        s : np.ndarray or None
            Uncertainties from instrument.noise (standard deviation).
        """

        # Flatten the data for 1D linear fit
        y = data_obj.data.flatten()
        x = data_obj.wavelengths.flatten()
        s = data_obj.noise.flatten()

        gauss_center = nonlin_paras[0]
        col0 = np.exp(-(my_wvs - gauss_center) ** 2 / (2 * gauss_width ** 2))

        # Design matrix: linear term + constant
        M = np.vstack([col0, np.ones_like(x)]).T

        extra_outputs = {}
        # Use  extra_outputs["regularization"] to add a regularization term on the linear parameters:
        # Refer to Ruffio+2024 Eq. (A4) and relevant section for more details on the maths.
        # A Gaussian prior (other word for regularization) can be included by adding some extra terms to the data vector, the model matrix and the error vector.
        # d_reg are the mean of the Gaussian prior on the linear parameters.
        # s_reg are the standard deviation of the Gaussian prior on the linear parameters.
        # d_reg and s_reg have the same length as the number of linear parameters, but parameters without regularization are set to np.nan. fitfm() manages the bookkeeping accordinly.
        # (d_reg, s_reg) fully define the regularization of the linear parameter with Gaussian prior.
        # Note: the modification to the forward model matrix M_reg is done directly in fitfm, it is effectively some partial identity matrix.
        extra_outputs["regularization"] = (d_reg, s_reg)

        return y, M, s, extra_outputs

    # To set a prior on the intercept, which here takes the role of the starlight or background level. One can do the following:
    d_reg = np.array([gauss_amplitude, intercept])  # we set the prior to the true value for demonstration purposes.
    s_reg = np.array([0.001, 0.001])  # very tight prior here for demonstration purposes.

    results = fitfm(
        nonlin_paras=[gauss_center],
        dataobj=data_obj,
        fm_func=intro_regularization_fm,
        fm_paras={'gauss_width': gauss_width, 'd_reg': d_reg, 's_reg': s_reg},
        marginalize_noise_scaling=False, scale_noise=False
    )
    bestfit_log_prob, log_prob_H0, rchi2, linparas, linparas_err = results

    # # Here the uncertainty on the intercept matches the prior we set.
    # print("BREADS gaussian amplitude:", linparas[0], "±", linparas_err[0])
    # print("BREADS Intercept (regularized):", linparas[1], "±", linparas_err[1])
    # print("Reduced chi2:", rchi2)

    assert np.abs(linparas[0] -gauss_amplitude) < 3 * linparas_err[0], "BREADS gaussian amplitude estimate is not within 3 sigma of true value (regularization)"
    assert np.abs(linparas[1] - intercept) < 3 * linparas_err[1], "BREADS intercept estimate is not within 3 sigma of true value (regularization)"
    assert linparas_err[1] < 2*0.001, "BREADS intercept uncertainty should be similar to the prior uncertainty we set (regularization)"

    # First, let's get the posterior on gauss_center without any prior (setting them to np.nan):
    gauss_center_grid = np.linspace(gauss_center - 2, gauss_center + 2, 100)
    bestfit_log_prob, _, rchi2, linparas, linparas_err = grid_search(
        para_vecs=[gauss_center_grid],
        dataobj=data_obj,
        fm_func=intro_regularization_fm,
        fm_paras={"gauss_width": gauss_width, 'd_reg': [np.nan, np.nan], 's_reg': [np.nan, np.nan]},
        numthreads=None,
        bounds=None
    )

    prob_no_prior = np.exp(bestfit_log_prob - np.max(bestfit_log_prob))
    best_fit1, left_err1, right_err1 = get_err_from_posterior(gauss_center_grid, prob_no_prior)
    # print("BREADS gauss_center (no prior):", best_fit1, "-", left_err1, "+", right_err1)

    assert np.abs(best_fit1 - gauss_center) < 3 * max(left_err1, right_err1), "BREADS gauss_center estimate is not within 3 sigma of true value (no prior)"

    # Let's set the priors now:
    bestfit_log_prob, _, rchi2, linparas, linparas_err = grid_search(
        para_vecs=[gauss_center_grid],
        dataobj=data_obj,
        fm_func=intro_regularization_fm,
        fm_paras={"gauss_width": gauss_width, 'd_reg': d_reg, 's_reg': s_reg},
        numthreads=None,
        bounds=None
    )

    prob_with_prior = np.exp(bestfit_log_prob - np.max(bestfit_log_prob))
    best_fit2, left_err2, right_err2 = get_err_from_posterior(gauss_center_grid, prob_with_prior)
    # print("BREADS gauss_center (with prior):", best_fit2, "-", left_err2, "+", right_err2)

    assert np.abs(best_fit2 - gauss_center) < 3 * max(left_err2, right_err2), "BREADS gauss_center estimate is not within 3 sigma of true value (with prior)"

    assert left_err2 < left_err1, "The prior should make the uncertainty smaller, so left_err2 should be smaller than left_err1"
    assert right_err2 < right_err1, "The prior should make the uncertainty smaller, so right_err2 should be smaller than right_err1"

    # Now let's compare to the MCMC posterior.

    def log_likelihood_case4(paras, data_obj):
        """
        Returns log-likelihood for case 4.
        We are adding a prior on gauss_amplitude and intercept here.
        """
        _gauss_center, _gauss_amplitude, _intercept = paras

        d, M, s, _ = intro_regularization_fm([_gauss_center], data_obj, gauss_width=gauss_width)
        model = np.dot(M, [_gauss_amplitude, _intercept])  # linear model prediction

        # Log-likelihood
        log_like = -0.5 * np.sum(((d - model) / s) ** 2)
        # adding the prior on gauss_amplitude and intercept.
        log_like += -0.5 * np.sum(((_gauss_amplitude - gauss_amplitude) / 0.001) ** 2)
        log_like += -0.5 * np.sum(((_intercept - intercept) / 0.001) ** 2)

        return log_like

    # MCMC setup
    nwalkers = 50
    nsteps = 2000

    # Start walkers
    paras_init = np.array([gauss_center, gauss_amplitude, intercept])
    ndim = np.size(paras_init)
    p0 = paras_init + 1e-2 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood_case4, args=[data_obj])

    # Run MCMC
    sampler.run_mcmc(p0, nsteps, progress=True)
    samples = sampler.get_chain(flat=True)

    # Plot MCMC posterior as histogram
    # Discard burn-in and extract chains
    gauss_center_samples = samples[(nsteps // 4 * nwalkers)::, 0]
    gauss_ampl_samples = samples[(nsteps // 4 * nwalkers)::, 1]
    intercept_samples = samples[(nsteps // 4 * nwalkers)::, 2]
    # print("EMCEE gaussian center:", np.median(gauss_center_samples), "±", np.std(gauss_center_samples))
    # print("EMCEE gaussian amplitude:", np.median(gauss_ampl_samples), "±", np.std(gauss_ampl_samples))
    # print("EMCEE Intercept:", np.median(intercept_samples), "±", np.std(intercept_samples))

    assert np.abs(np.median(gauss_center_samples) - gauss_center) < 3 * np.std(gauss_center_samples), "EMCEE gauss_center estimate is not within 3 sigma of true value (case 4)"
    assert np.abs(np.median(gauss_ampl_samples) - gauss_amplitude) < 3 * np.std(gauss_ampl_samples), "EMCEE gaussian amplitude estimate is not within 3 sigma of true value (case 4)"

    assert np.abs(np.std(gauss_center_samples)- max(left_err2, right_err2)) < 0.25*max(left_err2, right_err2), "EMCEE gauss_center uncertainty is not similar enough to BREADS gauss_center uncertainty (case 4)"

    assert (max(left_err1, right_err1) - np.std(gauss_center_samples)) > 0.25*max(left_err1, right_err1), "The prior should make the uncertainty smaller, so EMCEE gauss_center uncertainty should be more similar to BREADS gauss_center uncertainty with prior (left_err2, right_err2) than without prior (left_err1, right_err1)"