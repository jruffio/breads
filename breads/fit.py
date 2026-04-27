from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear
from scipy.special import loggamma

__all__ =  ('fitfm', 'log_prob', 'combined_log_prob', 'nlog_prob')


def fitfm(nonlin_paras, dataobj, fm_func, fm_paras, computeH0=False, bounds=None, scale_noise=True, marginalize_noise_scaling=False):
    """
    Fit a forward model (FM) to a data object (defined by an instrument class) returning probabilities and best fit linear parameters.

    Parameters
    ----------
        nonlin_paras : list
            [p1,p2,...] List of non-linear parameters such as rv, y, x. The meaning and number of non-linear parameters depends on the forward model defined.
        dataobj : Instrument
            A data object following the template of breads.instruments.instrument.Instrument to be analyzed.
        fm_func : function
            A forward model function. See breads.fm.template.template() for an example.
        fm_paras : dict
            Additional parameters for fm_func (other than non-linear parameters and dataobj)
        computeH0 : bool, optional
            If true (default is False), compute the probability of the model removing the first element of the linear model; See second ouput log_prob_H0. This can be used to compute delta likelihoods.
        bounds : tuple of arrays, optional
            (Caution: the calculation of log prob is only theoretically accurate if no bounds are used.)
            Bounds on the linear parameters used in lsq_linear as a tuple of arrays (min_vals, maxvals).
            e.g. ([0,0,...], [np.inf,np.inf,...]). default no bounds.
            Each numpy array must have shape (N_linear_parameters,).
        scale_noise : bool, optional
            If true (default True), scale the noise by the reduced chi2 of the best fit. This is useful when the noise is not well estimated. Set to False if you want to use the original noise estimation.
        marginalize_noise_scaling : bool, optional
            If true (default False), marginalize the log probability with respect to the noise scaling factor with a Jeffreys prior. This is useful when the noise is not well estimated. Set to False if you want to use the original noise estimation.
            This only works when there is no regularization in the forward model.
            This also only affect the calculation of the log probability but not the best fit linear parameters and their uncertainties.

    Returns
    -------
        log_prob : float
            Probability of the model marginalized over linear parameters.
            Caution: if the size of the data vector (N_data) varies for different non-linear parameters (e.g., different positions with different edges and bad pixels), the log probability values are not directly comparable and should only be used to compute delta log probabilities.
            To properly use log_prob, one should ensure that the data vector does not change when exploring the non-linear parameters. This is needed to compute RVs or astrometry for example.
        log_prob_H0 : float
            Probability of the model without the planet component(s).
        s2 : float
            This is the reduced chi2 of the best fit; i.e., the square of the noise scaling factor.
            The noise scaling factor being the standard deviation of the normalized residuals (normalized by the data noise standard deviation).
            While this value is always returned, it is only applied to the computation of linparas_err and/or log_prob if scale_noise is True.
        linparas : np.ndarray
            Best fit linear parameters
        linparas_err : np.ndarray
            Uncertainties of best fit linear parameters
    """
    if computeH0:
        raise Exception('computeH0 not yet implemented here')

    fm_out = fm_func(nonlin_paras, dataobj, **fm_paras)

    #Check the forward model matrices
    # "no_reg" stands for matrices and vector "without the regularization part" as the forward model can optionally include regularization vectors and matrices as well.
    if len(fm_out) == 3:
        d_no_reg, M_no_reg, s_no_reg = fm_out
    elif len(fm_out) == 4:
        d_no_reg, M_no_reg, s_no_reg, extra_outputs = fm_out
        # extra_outputs can include a regularization prescription for example.
    else:
        raise ValueError(f"Unrecognized number of matrices for forward model, the number of outputs for {fm_func.__name__} is expected to be 3 or 4 but {len(fm_out)} were given.")

    N_linpara = M_no_reg.shape[1]
    N_data = np.size(d_no_reg)

    if N_data == 0:
        # Nothing can be fitted
        return _invalid_outputs(N_linpara)

    if N_linpara == 1 and computeH0:
        # Only one parameter to fit so cannot test both H0 and H1 hypothesis.
        computeH0 = False
        raise Warning("Only one parameter to fit so cannot test H0 hypothesis.")

    # Initializing the boundaries for least-square fit
    if bounds is None:
        _bounds = ([-np.inf]*N_linpara, [np.inf]*N_linpara)
    else:
        _bounds = (copy(bounds[0]), copy(bounds[1]))
        if any(np.any(np.isfinite(arr)) for arr in _bounds): #check if there is finite boundaries
            raise Warning("The calculation of log prob is only theoretically accurate if no finite bounds are used...")

    # Will reject the column(s) full of 0 of the model matrix M (without regularization)
    validpara = np.where(np.any(M_no_reg != 0, axis=0))

    if len(fm_out) == 4 and "N_planet_linparas" in extra_outputs.keys():
        N_planet_paras = extra_outputs["N_planet_linparas"]
    else:
        N_planet_paras = 1
    if np.min(validpara[0]) >= N_planet_paras:
        # the first linear parameters are invalid which means that the companion cannot be fitted
        # Hence, we return nan for the best fit linear parameters and -inf for their probability
        raise Warning("Companion cannot be fitted, returning nan arrays")
        return _invalid_outputs(N_linpara)

    M_no_reg = M_no_reg[:, validpara[0]]  # Filtering the column(s) full of 0
    _bounds = (np.array(_bounds[0])[validpara[0]], np.array(_bounds[1])[validpara[0]]) #Selecting the bounds for the valid parameters

    d_no_reg = d_no_reg / s_no_reg #Normalizing the data by the data standard deviation
    M_no_reg = M_no_reg / s_no_reg [:, None] #Normalizing the M_ij by the data standard deviation s_i

    # check if regularization is used in the forward model by checking the extra outputs of the forward model
    if len(fm_out) == 4 and "regularization" in extra_outputs.keys():
        d_reg, s_reg = extra_outputs["regularization"]
        if np.sum(~np.isnan(d_reg)) == 0 or np.sum(~np.isnan(s_reg)) == 0:
            # No regularization used in this case
            is_regularized = False
        else:
            is_regularized = True
            if marginalize_noise_scaling:
                raise Exception("The maths for the marginalization of the noise scaling factor is not compatible with the regularization. Set marginalize_noise_scaling = False")
    else:
        is_regularized = False

    if not is_regularized: # No regularization used in this case
        M = M_no_reg
        d = d_no_reg
        s = s_no_reg

        logdet_Sigma = np.sum(2 * np.log(s))

        paras, _, residuals, chi2, rchi2, noise_scaling = _get_lsq_fit(M, d, _bounds, N_data=None)
        if not scale_noise:
            noise_scaling = 1.0

        # Section to compute error bars of linear parameters
        MTM = np.dot(M.T, M)
        # error catching is because the matrix inversion can fail and we don't want this to crash the entire process when computing an SNR map for example.
        try:
            iMTM = np.linalg.inv(MTM)
            slogdet_icovphi0 = np.linalg.slogdet(MTM)
            logdet_icovphi0 = slogdet_icovphi0[1]
        except Exception as e:
            # only printing the error message, but will not stop because of it
            # Will simply return the outputs corresponding to invalid data.
            print("Exiting covariance section in fitfm() with error:")
            print(e)
            return _invalid_outputs(N_linpara)
    else: # Regularization used in this case
        # Prepares the vectors and matrics with the regularization part concatenated to the data vector, model matrix, and noise vector for the regularized fit.
        M, d, s, M_reg, d_reg, s_reg = _concatenate_model_regularization(d_no_reg, M_no_reg, s_no_reg, extra_outputs, validpara)
        # M, d, and s now include their regularization parts.

        # when the data is regularized, the noise scaling is a bit tricky because it will impact the relative weighting of the data and the regularization in the fit.
        # We therefore do this interatively with a first fit to get the noise scaling factor and then apply it to the data and the model before doing the final fit and computing the log probability.
        paras, _, residuals, chi2, rchi2, noise_scaling = _get_lsq_fit(M, d, _bounds, N_data=N_data)

        if scale_noise:
            M = np.concatenate([M_no_reg/noise_scaling, M_reg], axis=0)
            d = np.concatenate([d_no_reg/noise_scaling, d_reg/s_reg])
            s = np.concatenate([s_no_reg*noise_scaling, s_reg])
            paras, _, residuals, chi2, _, _ = _get_lsq_fit(M, d, _bounds, N_data=None)

            # noise scaling is done, set to unity for later as no more scaling is necessary
            noise_scaling = 1

        logdet_Sigma = np.sum(2 * np.log(s))


        # Section to compute error bars of linear parameters
        MTM = np.dot(M.T, M)
        MTM_no_reg = np.dot(M_no_reg.T, M_no_reg)
        try:
            iMTM = np.linalg.inv(MTM)
            covphi = np.dot(iMTM, np.dot(MTM_no_reg, iMTM.T))
            # The formula below assumes that we are using the determinant of the inverse covariance
            # That's why we are adding the minus sign
            logdet_icovphi0 = -np.sum(np.log(np.diag(covphi)))
        except Exception as e:
            # only printing the error message, but will not stop because of it
            # Will simply return the outputs corresponding to invalid data.
            print("Exiting covariance section in fitfm() with error:")
            print(e)
            return _invalid_outputs(N_linpara)

    covphi = noise_scaling ** 2 * iMTM
    diagcovphi = copy(np.diag(covphi))
    diagcovphi[np.where(diagcovphi < 0.0)] = np.nan  # uncertainties cannot be negative so replacing by nan here
    paras_err = np.sqrt(diagcovphi)  # get the uncertainties via the diagonal of the matrix

    # Caution: One need to be mindful of N_data:
    # N_data the size of d, but it can vary depending on the position  of the planet and the number of bad pixels for example, which can change the size of the data vector.
    # This means that the log_prob is not comparable between different planet positions.
    # To fix this issue, one needs to fix the data vector in the forward model function, such that it does not depend on the non-linear parameters.
    if marginalize_noise_scaling and not is_regularized:
        # Eq 41 in Ruffio+2019, https://ui.adsabs.harvard.edu/abs/2019AJ....158..200R/abstract
        log_prob = (M.shape[1] - N_data) / 2 * np.log(2 * np.pi) - 0.5 * logdet_Sigma - 0.5 * logdet_icovphi0 \
                   - ((N_data - M.shape[1] + 2 - 1) / 2) * np.log(chi2) + loggamma((N_data - M.shape[1] + 2 - 1) / 2)
    else:
        # log(Eq 36) in Ruffio+2019, https://ui.adsabs.harvard.edu/abs/2019AJ....158..200R/abstract
        log_prob = ((M.shape[1] - N_data) / 2) * np.log(2 * np.pi) - 0.5 * logdet_Sigma - 0.5 * logdet_icovphi0 \
                   - ((N_data - M.shape[1]) / 2) * np.log(noise_scaling ** 2) - 0.5 * chi2 / noise_scaling ** 2

    # Initialize the arrays of best-fit linear parameters and their uncertainties
    linparas = np.full(N_linpara, np.nan)
    linparas_err = np.full(N_linpara, np.nan)

    # Bookeeping the valid best fit linear parameters
    linparas[validpara] = paras
    linparas_err[validpara] = paras_err

    log_prob_H0 = None
    return log_prob, log_prob_H0, rchi2, linparas, linparas_err

def log_prob(nonlin_paras, dataobj, fm_func, fm_paras, nonlin_lnprior_func=None, bounds=None, scale_noise=True, marginalize_noise_scaling=False):
    """
    Wrapper to fit_fm() but only returns the log probability marginalized over the linear parameters.
    """
    if nonlin_lnprior_func is not None:
        prior = nonlin_lnprior_func(nonlin_paras)
    else:
        prior = 0
    try:
        lnprob = fitfm(nonlin_paras, dataobj, fm_func, fm_paras,computeH0=False,bounds=bounds,scale_noise=scale_noise, marginalize_noise_scaling=marginalize_noise_scaling)[0]+prior
    except:
        lnprob =  -np.inf
    return lnprob


def combined_log_prob(nonlin_paras, dataobjlist,fm_funclist, fm_paraslist, nonlin_lnprior_func=None, bounds=None):
    """
    For use when you have multiple data objects and want to combine the log-likelihoods for MCMC sampling

    Parameters
    ----------
    nonlin_paras : list of floats
        [p1,p2,...] List of non-linear parameters such as rv, y, x. The meaning and number of non-linear
        parameters depends on the forward model defined.
    dataobjlist: A list of data objects to combine
    fm_funclist: A list of fm_func for each data object
    fm_paraslist: A list of fm_paras to use as arguments for it's respective fm_func
    nonlin_lnprior_func: A function to compute priors, if None, defaults to zero priors

    Returns
    -------

    """
    combined_lnprob = 0
    for i, dataobj in enumerate(dataobjlist):
        lnprob=log_prob(nonlin_paras = nonlin_paras, dataobj=dataobj, fm_func=fm_funclist[i], fm_paras= fm_paraslist[i], nonlin_lnprior_func=nonlin_lnprior_func, bounds=None)
        combined_lnprob += lnprob
    return combined_lnprob

def nlog_prob(nonlin_paras, dataobj, fm_func, fm_paras,nonlin_lnprior_func=None,bounds=None,scale_noise=True):
    """ Returns the negative of the log_prob() for minimization routines.

    Args:
        nonlin_paras: [p1,p2,...] List of non-linear parameters such as rv, y, x. The meaning and number of non-linear
            parameters depends on the forward model defined.
        dataobj: A data object of type breads.instruments.instrument.Instrument to be analyzed.
        fm_func: A forward model function. See breads.fm.template.template() for an example.
        fm_paras: Additional parameters for fm_func (other than non-linear parameters and dataobj)
        computeH0: If true (default), compute the probability of the model removing the first element of the linear
            model; See second ouput log_prob_H0. This can be used to compute the Bayes factor for a fixed set of
            non-linear parameters
        bounds: (Caution: the calculation of log prob is only theoretically accurate if no bounds are used.)
            Bounds on the linear parameters used in lsq_linear as a tuple of arrays (min_vals, maxvals).
            e.g. ([0,0,...], [np.inf,np.inf,...]) default no bounds.
            Each numpy array must have shape (N_linear_parameters,).

    Returns:
        log_prob: Probability of the model marginalized over linear parameters.
    """
    nlogprob_val =  - log_prob(nonlin_paras, dataobj, fm_func, fm_paras,nonlin_lnprior_func,bounds,scale_noise)
    # print( nlogprob_val, nonlin_paras)
    return nlogprob_val

def _invalid_outputs(N_linear_parameters):
    """Helper to return invalid outputs"""
    linparas = np.full(N_linear_parameters, np.nan)
    linparas_err = np.full(N_linear_parameters, np.nan)
    log_prob = -np.inf
    log_prob_H0 = -np.inf
    s2 = np.inf

    return log_prob, log_prob_H0, s2, linparas, linparas_err

def _concatenate_model_regularization(d_no_reg, M_no_reg, s_no_reg, extra_outputs, validpara):
    """Helper function to concatenate regularization parameters with model matrix and data vectors"""
    #Retrieve the regularization priors
    d_reg, s_reg = extra_outputs["regularization"]

    #Filtering the bad columns
    s_reg = s_reg[validpara]
    d_reg = d_reg[validpara]

    #Filtering the finite values to match with the M_no_reg matrix
    where_finite_reg = np.where(np.isfinite(s_reg))
    s_reg = s_reg[where_finite_reg]
    d_reg = d_reg[where_finite_reg]

    #Creating the matrix for regularization
    M_reg = np.zeros((np.size(where_finite_reg[0]), M_no_reg.shape[1]))

    #Setting the diagonal of the matrix
    M_reg[np.arange(np.size(where_finite_reg[0])), where_finite_reg[0]] = 1 / s_reg

    #Concatenate each matrix/vector
    M = np.concatenate([M_no_reg, M_reg], axis=0)
    d = np.concatenate([d_no_reg, d_reg / s_reg])
    s = np.concatenate([s_no_reg, s_reg])

    return M, d, s, M_reg, d_reg, s_reg

def _get_lsq_fit(M_normalized, d_normalized, _bounds, N_data=None):
    """Helper to get the least squares fit of the model on the data. Model matrix and input data have to be normalized by the noise."""
    paras = lsq_linear(M_normalized, d_normalized, bounds=_bounds).x
    d_estimated = np.dot(M_normalized, paras)
    residuals = d_normalized - d_estimated
    if N_data is not None:
        #Truncating to N_data because we don't care about the residuals on the regularization priors
        chi2 = np.nansum(residuals[:N_data] ** 2)
        rchi2 = chi2 / N_data
    else:
        N_data = np.size(residuals)
        chi2 = np.nansum(residuals ** 2)
        rchi2 = chi2 / N_data
    noise_scaling = np.sqrt(rchi2)

    return paras, d_estimated, residuals, chi2, rchi2, noise_scaling

def _compute_H0(M_normalized, d_normalized, N_data, _bounds, logdet_Sigma, marginalize_noise_scaling=True, noise_scaling=1):
    """Helper to compute the H0 hypothesis (i.e. without off axis companion)"""

    M_H0 = M_normalized[:, 1:] #removing the first column i.e. the off axis companion model
    _crop_bounds = (np.array(_bounds[0])[1::], np.array(_bounds[1])[1::])

    paras_H0, d_estimated, residuals_H0, chi2_H0, rchi2, _ = _get_lsq_fit(M_H0, d_normalized, _crop_bounds, N_data=None)

    slogdet_icovphi0_H0 = np.linalg.slogdet(np.dot(M_H0.T, M_H0))
    # todo check the maths when N_linpara is different from M.shape[1]. E.g. at the edge of the FOV
    if marginalize_noise_scaling:
        log_prob_H0 = -0.5 * logdet_Sigma - 0.5 * slogdet_icovphi0_H0[1] - (
                    N_data + M_H0.shape[1] + 2 - 1) / 2 * np.log(chi2_H0) + \
                    loggamma((N_data - M_H0.shape[1] + 2 - 1) / 2) + (M_H0.shape[1] - N_data) / 2 * np.log(
                    2 * np.pi)
    else:
        # log(Eq 36) in Ruffio+2019:
        # TODO Seems more logical to me that it should be M_H0.shape[1] here, but I didn't redo the math. Need to check.
        log_prob_H0 = ((M_normalized.shape[1] - N_data) / 2) * np.log(2 * np.pi) - 0.5 * logdet_Sigma - 0.5 * slogdet_icovphi0_H0[
            1] \
                      - ((N_data - M_normalized.shape[1]) / 2) * np.log(noise_scaling ** 2) - 0.5 * chi2_H0 / noise_scaling ** 2

    return log_prob_H0