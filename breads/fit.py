from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear
from scipy.special import loggamma

__all__ =  ('fitfm', 'log_prob', 'combined_log_prob', 'nlog_prob')

def fitfm(nonlin_paras, dataobj, fm_func, fm_paras, computeH0=True, bounds=None, scale_noise=True, marginalize_noise_scaling=False,
          debug=False):
    """
    Fit a forward model to data returning probabilities and best fit linear parameters.

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
        log_prob_H0: Probability of the model without the planet marginalized over linear parameters.
        s2: noise scaling factor
        linparas: Best fit linear parameters
        linparas_err: Uncertainties of best fit linear parameters
    """
    fm_out = fm_func(nonlin_paras, dataobj, **fm_paras)

    #Check the forward model matrices
    if len(fm_out) == 3:
        d_no_reg, M_no_reg, s_no_reg = fm_out
    elif len(fm_out) == 4:
        d_no_reg, M_no_reg, s_no_reg, extra_outputs = fm_out
    else:
        raise ValueError(f"Unrecognized number of matrices for forward model, the number of outputs for {fm_func.__name__} is expected to be 3 or 4 but {len(fm_out)} were given.")

    N_linpara = M_no_reg.shape[1]
    N_data = np.size(d_no_reg)

    if N_data == 0:
        #Nothing can be fitted
        return _invalid_outputs(N_linpara)

    if N_linpara == 1 and computeH0:
        #Only one parameter to fit so cannot test both H0 and H1 hypothesis.
        computeH0 = False
        raise Warning("Only one parameter to fit so cannot test H0 hypothesis.")

    #Initializing the boundaries for least-square fit
    if bounds is None:
        _bounds = ([-np.inf]*N_linpara, [np.inf]*N_linpara)
    else:
        _bounds = (copy(bounds[0]), copy(bounds[1]))
        if any(np.any(np.isfinite(arr)) for arr in _bounds): #check if there is finite boundaries
            raise Warning("The calculation of log prob is only theoretically accurate if no finite bounds are used...")

    # Reject the column(s) full of 0 of the model matrix M (without regularization)
    validpara = np.where(np.any(M_no_reg != 0, axis=0))

    if 0 not in validpara[0]:
        #the first linear parameters is invalid which means that the companion cannot be fitted
        #Hence, we return nan for the best fit linear parameters and -inf for their probability
        print("Companion cannot be fitted, returning nan arrays")
        return _invalid_outputs(N_linpara)

    M_no_reg = M_no_reg[:, validpara[0]]  # Filtering the column(s) full of 0
    _bounds = (np.array(_bounds[0])[validpara[0]], np.array(_bounds[1])[validpara[0]]) #Selecting the bounds for the valid parameters

    d_no_reg = d_no_reg / s_no_reg #Normalizing the data by the data standard deviation
    M_no_reg = M_no_reg / s_no_reg [:, None] #Normalizing the M_ij by the data standard deviation s_i

    ##### Concatenating the regularization vectors with model matrix and data vector + computing the scale noise factor with a first lsq best fit
    if len(fm_out) == 4 and "regularization" in extra_outputs.keys():
        if marginalize_noise_scaling:
            raise Exception("The maths for the marginalization of the noise scaling factor is not compatible with the regularization. Set marginalize_noise_scaling = False")

        M, d, s, M_reg, d_reg, s_reg = _concatenate_model_regularization(d_no_reg, M_no_reg, s_no_reg, extra_outputs, validpara)

        if scale_noise:
            _, _, _, _, rchi2, noise_scaling = _get_lsq_fit(M, d, _bounds, N_data=N_data)
            M = np.concatenate([M_no_reg/noise_scaling, M_reg], axis=0)
            d = np.concatenate([d_no_reg/noise_scaling, d_reg/s_reg])
            s = np.concatenate([s_no_reg*noise_scaling, s_reg])

        # noise scaling is done, set to unity for later as no more scaling is necessary
        noise_scaling = 1

    else:
        # No regularization used in this case
        M = M_no_reg
        d = d_no_reg
        s = s_no_reg

    logdet_Sigma = np.sum(2 * np.log(s))

    paras, _, residuals, chi2, _, _ = _get_lsq_fit(M, d, _bounds, N_data=None)

    # Section to compute error bars of linear parameters
    MTM = np.dot(M.T, M)
    try:
        iMTM = np.linalg.inv(MTM)
        if len(fm_out) == 4 and "regularization" in extra_outputs.keys():
            if not scale_noise:
                rchi2 = np.nansum(residuals[:N_data] ** 2) / N_data #rchi2 computed only on the data residuals without the regularization.

            MTM_no_reg = np.dot(M_no_reg.T, M_no_reg)
            covphi = np.dot(iMTM, np.dot(MTM_no_reg,iMTM.T))
            # The formula below assumes that we are using the determinant of the inverse covariance
            # That's why we are adding the minus sign
            logdet_icovphi0 = -np.sum(np.log(np.diag(covphi)))

            if debug:
                pass
                slogdet_covphi0 = np.linalg.slogdet(covphi)
                logdet_icovphi01 = -slogdet_covphi0[1]

                logdet_MTM = np.linalg.slogdet(MTM)[1]
                logdet_MTM_no_reg = np.linalg.slogdet(MTM_no_reg)[1]
                logdet_icovphi02 = -(-2*logdet_MTM+logdet_MTM_no_reg)
                print("logdet_icovphi02, logdet_icovphi0",logdet_icovphi01,logdet_icovphi02,-np.sum(np.log(np.diag(covphi))))

        else:
            if scale_noise:
                rchi2 = np.nansum(residuals[:N_data] ** 2) / N_data
                noise_scaling = np.sqrt(rchi2)
            else:
                rchi2 = 1
                noise_scaling = 1

            covphi = noise_scaling * iMTM
            slogdet_icovphi0 = np.linalg.slogdet(MTM)
            logdet_icovphi0 = slogdet_icovphi0[1]
            if debug:
                plt.plot(np.diag(covphi))
                plt.show()
                print("logdet_icovphi02,logdet_icovphi0",logdet_icovphi0,-np.sum(np.log(np.diag(covphi))))
                exit()

    except Exception as e:
        print("Exiting covariance section in fitfm() with error:")
        print(e)
        return _invalid_outputs(N_linpara)


    diagcovphi = copy(np.diag(covphi))
    diagcovphi[np.where(diagcovphi<0.0)] = np.nan #uncertainties cannot be negative so replacing by nan here
    paras_err = np.sqrt(diagcovphi) #get the uncertainties via the diagonal of the matrix

    # JB: the N_data is kinda wrong I think because is N_data the size of d or the number of pixels on the detector?
    # But this is just a constant in the log probability, so as long as we don't use the absolute likelihood values,
    # we are fine
    if marginalize_noise_scaling:
        log_prob = (M.shape[1] - N_data) / 2 * np.log(2 * np.pi) -0.5 * logdet_Sigma - 0.5 * logdet_icovphi0 \
                   - ((N_data-M.shape[1]+2-1)/2) * np.log(chi2) + loggamma((N_data - M.shape[1] + 2 - 1) / 2)
    else:
        # log(Eq 36) in Ruffio+2019:
        log_prob = ((M.shape[1]-N_data)/2)*np.log(2*np.pi) -0.5 * logdet_Sigma - 0.5 * logdet_icovphi0 \
                   -((N_data-M.shape[1])/2) * np.log(noise_scaling**2) -0.5*chi2/noise_scaling**2
        # log_prob = -0.5*chi2/noise_scaling**2

    if computeH0:
        log_prob_H0 = _compute_H0(M, d, N_data, _bounds, logdet_Sigma, marginalize_noise_scaling)
    else:
        log_prob_H0 = np.nan

    # Initialize the arrays of best-fit linear parameters and their uncertainties
    linparas = np.full(N_linpara, np.nan)
    linparas_err = np.full(N_linpara, np.nan)

    #Bookeeping the valid best fit linear parameters
    linparas[validpara] = paras
    linparas_err[validpara] = paras_err

    return log_prob, log_prob_H0, rchi2, linparas, linparas_err

def log_prob(nonlin_paras, dataobj, fm_func, fm_paras, nonlin_lnprior_func=None, bounds=None, scale_noise=True):
    """
    Wrapper to fit_fm() but only returns the log probability marginalized over the linear parameters.

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
            e.g. ([0,0,...], [np.inf,np.inf,...]). default no bounds.
            Each numpy array must have shape (N_linear_parameters,).

    Returns:
        log_prob: Probability of the model marginalized over linear parameters.
    """
    if nonlin_lnprior_func is not None:
        prior = nonlin_lnprior_func(nonlin_paras)
    else:
        prior = 0
    try:
        lnprob = fitfm(nonlin_paras, dataobj, fm_func, fm_paras,computeH0=False,bounds=bounds,scale_noise=scale_noise)[0]+prior
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
        chi2 = np.nansum(residuals ** 2) / N_data
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