from copy import copy
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear
from scipy.special import loggamma
from scipy.linalg import cho_factor, cho_solve, get_lapack_funcs
from scipy import sparse

__all__ =  ('fitfm', 'log_prob', 'combined_log_prob', 'nlog_prob')


def fitfm(nonlin_paras, dataobj, fm_func, fm_paras, bounds=None, scale_noise=True, marginalize_noise_scaling=False,
          return_covariance=False, fast_unbounded_solve=True):
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
        return_covariance : bool, optional
            If true (default False), also return the full covariance matrix of the linear parameters (see below).
        fast_unbounded_solve : bool, optional
            If true (default True) and the linear parameters have no finite bounds, solve the least squares problem from
            the normal equations with a Cholesky factorization of M^T M (reused for the covariance and its log
            determinant) instead of lsq_linear, with or without regularization. This is mathematically equivalent and
            much faster for large model matrices; results differ at the round-off level. It falls back to lsq_linear
            if M^T M is not positive definite or is badly conditioned (see _cholesky_normal_equations). Bounded fits
            always use lsq_linear. The model matrix returned by fm_func can be a scipy sparse matrix (e.g. the 3D
            B-spline design matrices): it is then kept sparse in the fast path, and converted to a dense array if
            lsq_linear has to be used.

    Returns
    -------
        log_prob : float
            Probability of the model marginalized over linear parameters.
            Caution: if the size of the data vector (N_data) varies for different non-linear parameters (e.g., different positions with different edges and bad pixels), the log probability values are not directly comparable and should only be used to compute delta log probabilities.
            To properly use log_prob, one should ensure that the data vector does not change when exploring the non-linear parameters. This is needed to compute RVs or astrometry for example.
        s2 : float
            This is the reduced chi2 of the best fit; i.e., the square of the noise scaling factor.
            The noise scaling factor being the standard deviation of the normalized residuals (normalized by the data noise standard deviation).
            While this value is always returned, it is only applied to the computation of linparas_err and/or log_prob if scale_noise is True.
        linparas : np.ndarray
            Best fit linear parameters
        linparas_err : np.ndarray
            Uncertainties of best fit linear parameters
        linparas_cov : np.ndarray (N_linear_parameters x N_linear_parameters)
            Only if return_covariance is True. Covariance matrix of the linear parameters, whose diagonal is
            linparas_err**2. Rows and columns of the invalid parameters (NaN in linparas) are NaN.
    """
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
    is_sparse = sparse.issparse(M_no_reg)
    if is_sparse and not (fast_unbounded_solve and (bounds is None or _bounds_are_infinite(bounds))):
        M_no_reg, is_sparse = M_no_reg.toarray(), False

    if N_data == 0:
        # Nothing can be fitted
        return _invalid_outputs(N_linpara, return_covariance)

    # Initializing the boundaries for least-square fit
    if bounds is None:
        _bounds = ([-np.inf]*N_linpara, [np.inf]*N_linpara)
    else:
        _bounds = (copy(bounds[0]), copy(bounds[1]))
        if any(np.any(np.isfinite(arr)) for arr in _bounds): #check if there is finite boundaries
            warning_text = "The calculation of log prob is only theoretically accurate if no finite bounds are used..."
            warn(warning_text)

    # Will reject the column(s) full of 0 of the model matrix M (without regularization)
    if is_sparse:
        M_no_reg = sparse.csr_matrix(M_no_reg)
        abs_col_sums = np.asarray(abs(M_no_reg).T @ (1 / s_no_reg)).ravel()
    else:
        abs_col_sums = _abs_normalized_column_sums(M_no_reg, s_no_reg)
    validpara = np.where(~np.isclose(abs_col_sums, 0, atol=1e-10))

    # if len(fm_out) == 4 and "N_planet_linparas" in extra_outputs.keys():
    #     N_planet_paras = extra_outputs["N_planet_linparas"]
    # else:
    #     N_planet_paras = 1
    # if np.min(validpara[0]) >= N_planet_paras:
    #     # the first linear parameters are invalid which means that the companion cannot be fitted
    #     # Hence, we return nan for the best fit linear parameters and -inf for their probability
    #     warning_text = "Companion cannot be fitted, returning nan arrays"
    #     warn(warning_text)
    #     return _invalid_outputs(N_linpara, return_covariance)

    M_no_reg = M_no_reg[:, validpara[0]]  # Filtering the column(s) full of 0
    _bounds = (np.array(_bounds[0])[validpara[0]], np.array(_bounds[1])[validpara[0]]) #Selecting the bounds for the valid parameters

    d_no_reg = d_no_reg / s_no_reg #Normalizing the data by the data standard deviation
    # Normalizing the M_ij by the data standard deviation s_i. M_no_reg is a copy (fancy indexing above), so this is done
    # in place when the dtype allows it, to avoid another full-size copy of the model matrix.
    if is_sparse:
        M_no_reg = sparse.csr_matrix(M_no_reg, dtype=np.result_type(M_no_reg.dtype, s_no_reg.dtype))
        M_no_reg.data /= np.repeat(s_no_reg, np.diff(M_no_reg.indptr))  # each row divided by its noise
    elif np.result_type(M_no_reg.dtype, s_no_reg.dtype) == M_no_reg.dtype:
        M_no_reg /= s_no_reg[:, None]
    else:
        M_no_reg = M_no_reg / s_no_reg[:, None]

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

    # Rescaling of the linear parameters (purely numerical, the maths is unchanged).
    # The columns of the model matrix can have very different scales (different units for the planet flux, the
    # spline and the KL coefficients, or barely sampled basis functions), which makes the matrices below badly
    # conditioned. Each linear parameter phi_j is therefore replaced by phi'_j = c_j * phi_j, where c_j is the norm of
    # the j-th column of the full (whitened) least-squares matrix, i.e. data rows plus its regularization row if any:
    #     c_j^2 = ||M_j / s||^2 + 1 / s_reg_j^2
    # (c_j^-1 is the uncertainty of phi_j if it were fitted alone). The rescaled columns then all have unit norm.
    # Any positive c gives the same results; everything that depends on the linear parameters is transformed:
    #   - bounds and regularization priors (mean and width) on phi are multiplied by c,
    #   - the best fit values and uncertainties are divided by c at the end,
    #   - the log-determinants are corrected so that log_prob is the same as without rescaling:
    #     covphi' = C covphi C with C = diag(c), so log|covphi| = log|covphi'| - 2 sum(log c);
    #     and the regularization widths enter logdet_Sigma, so it is corrected by -2 sum(log c) over regularized paras.
    if is_sparse:
        _para_scale2 = np.asarray(M_no_reg.multiply(M_no_reg).sum(axis=0)).ravel()
    else:
        _para_scale2 = np.nansum(M_no_reg ** 2, axis=0)
    if is_regularized:
        _s_reg_valid = np.asarray(extra_outputs["regularization"][1], dtype=float)[validpara]
        _has_prior = np.isfinite(_s_reg_valid) & (_s_reg_valid != 0)
        _para_scale2[_has_prior] += 1.0 / _s_reg_valid[_has_prior] ** 2
    _para_scale = np.sqrt(_para_scale2)
    _para_scale[~(np.isfinite(_para_scale) & (_para_scale > 0))] = 1.0
    if is_sparse:
        M_no_reg.data /= _para_scale[M_no_reg.indices]
    elif np.result_type(M_no_reg.dtype, _para_scale.dtype) == M_no_reg.dtype:
        M_no_reg /= _para_scale[None, :]  # in place, M_no_reg is already a copy
    else:
        M_no_reg = M_no_reg / _para_scale[None, :]
    _bounds = (_bounds[0] * _para_scale, _bounds[1] * _para_scale)
    logdet_para_scale = 2 * np.sum(np.log(_para_scale))  # = log|C|^2, correction of the log-determinants
    N_fit_paras = M_no_reg.shape[1]  # number of fitted linear parameters (columns of M, with or without regularization)

    if not is_regularized: # No regularization used in this case
        M = M_no_reg
        d = d_no_reg
        s = s_no_reg

        logdet_Sigma = np.sum(2 * np.log(s))

        # Fast path: without bounds, the solution, its covariance and the log determinant all follow from a single
        # Cholesky factorization of MTM (None if MTM is not positive definite or badly conditioned).
        fast_fit = None
        if fast_unbounded_solve and _bounds_are_infinite(_bounds):
            MTM = _symmetric_gram(M)
            fast_fit = _cholesky_normal_equations(MTM, _matvec(M.T, d))
        if fast_fit is not None:
            c_and_lower, paras = fast_fit
            _, residuals, chi2, rchi2, noise_scaling = _residual_stats(M, d, paras, N_data=None)
            if not scale_noise:
                noise_scaling = 1.0
            covphi0 = cho_solve(c_and_lower, np.eye(MTM.shape[0]))
            # MTM = C^-1 MTM_orig C^-1, so log|MTM_orig| = log|MTM| + 2 sum(log c)
            logdet_icovphi0 = 2 * np.sum(np.log(np.diag(c_and_lower[0]))) + logdet_para_scale
        else:
            if is_sparse:
                M = M.toarray()
            paras, _, residuals, chi2, rchi2, noise_scaling = _get_lsq_fit(M, d, _bounds, N_data=None)
            if not scale_noise:
                noise_scaling = 1.0

            # Section to compute error bars of linear parameters
            MTM = np.dot(M.T, M)
            # error catching is because the matrix inversion can fail and we don't want this to crash the entire process when computing an SNR map for example.
            try:
                covphi0 = np.linalg.inv(MTM)
                slogdet_icovphi0 = np.linalg.slogdet(MTM)
                # MTM = C^-1 MTM_orig C^-1, so log|MTM_orig| = log|MTM| + 2 sum(log c)
                logdet_icovphi0 = slogdet_icovphi0[1] + logdet_para_scale
            except Exception as e:
                # only printing the error message, but will not stop because of it
                # Will simply return the outputs corresponding to invalid data.
                print("Exiting covariance section in fitfm() with error:")
                print(e)
                return _invalid_outputs(N_linpara, return_covariance)

        covphi = noise_scaling ** 2 * covphi0

    else: # Regularization used in this case
        # Prepares the vectors and matrics with the regularization part concatenated to the data vector, model matrix, and noise vector for the regularized fit.
        #Retrieve the regularization priors
        d_reg, s_reg = extra_outputs["regularization"]

        #Filtering the bad columns, and rescaling the priors to the rescaled parameters phi' = c * phi
        s_reg = s_reg[validpara] * _para_scale
        d_reg = d_reg[validpara] * _para_scale

        #Filtering the finite values to match with the M_no_reg matrix
        where_finite_reg = np.where(np.isfinite(s_reg))
        s_reg = s_reg[where_finite_reg]
        d_reg = d_reg[where_finite_reg]
        # The rescaled prior widths enter logdet_Sigma below: log(c * s_reg) = log(s_reg) + log(c)
        logdet_reg_scale = 2 * np.sum(np.log(_para_scale[where_finite_reg]))

        # when the data is regularized, the noise scaling is a bit tricky because it will impact the relative weighting of the data and the regularization in the fit.
        # We therefore do this interatively with a first fit to get the noise scaling factor and then apply it to the data and the model before doing the final fit and computing the log probability.
        # Fast path: the Gram matrix of the data is computed once and reused for both fits and for the covariance;
        # the diagonal regularization rows are added analytically, so the stacked matrix M is not built (None if the
        # normal matrix is not positive definite or badly conditioned, then lsq_linear is used as before).
        fast_fit = None
        if fast_unbounded_solve and _bounds_are_infinite(_bounds):
            fast_fit = _fit_regularized_normal_equations(M_no_reg, d_no_reg, d_reg, s_reg, where_finite_reg[0],
                                                         scale_noise)
        if fast_fit is not None:
            paras, chi2, rchi2, noise_scaling, MTM_no_reg_fast, MTM_fast, c_and_lower_fast = fast_fit
            if scale_noise:
                s = np.concatenate([s_no_reg * noise_scaling, s_reg])
                # noise scaling is done, set to unity for later as no more scaling is necessary
                noise_scaling = 1
            else:
                s = np.concatenate([s_no_reg, s_reg])
        else:
            if is_sparse:
                M_no_reg = M_no_reg.toarray()

            #Creating the matrix for regularization
            M_reg = np.zeros((np.size(where_finite_reg[0]), M_no_reg.shape[1]))

            #Setting the diagonal of the matrix
            M_reg[np.arange(np.size(where_finite_reg[0])), where_finite_reg[0]] = 1 / s_reg

            #Concatenate each matrix/vector
            M = np.concatenate([M_no_reg, M_reg], axis=0)
            d = np.concatenate([d_no_reg, d_reg / s_reg])
            s = np.concatenate([s_no_reg, s_reg])
            # M, d, and s now include their regularization parts.

            paras, _, residuals, chi2, rchi2, noise_scaling = _get_lsq_fit(M, d, _bounds, N_data=N_data)

            if scale_noise:
                M = np.concatenate([M_no_reg/noise_scaling, M_reg], axis=0)
                d = np.concatenate([d_no_reg/noise_scaling, d_reg/s_reg])
                s = np.concatenate([s_no_reg*noise_scaling, s_reg])
                paras, _, residuals, chi2, _, _ = _get_lsq_fit(M, d, _bounds, N_data=None)

                # noise scaling is done, set to unity for later as no more scaling is necessary
                noise_scaling = 1

        logdet_Sigma = np.sum(2 * np.log(s)) - logdet_reg_scale  # same value as without the rescaling


        # Section to compute error bars of linear parameters
        if fast_fit is not None:
            # already computed by the fast path
            MTM, MTM_no_reg = MTM_fast, MTM_no_reg_fast
        else:
            MTM = np.dot(M.T, M)
            MTM = 0.5 * (MTM + MTM.T)
            MTM_no_reg = np.dot(M_no_reg.T, M_no_reg)
            MTM_no_reg = 0.5 * (MTM_no_reg + MTM_no_reg.T)
        try:
            # iMTM = np.linalg.inv(MTM)
            # covphi = np.dot(iMTM, np.dot(MTM_no_reg, iMTM.T))
            c_and_lower = c_and_lower_fast if fast_fit is not None else cho_factor(MTM)  # Cholesky of MTM (fails loudly if not PD)
            X = cho_solve(c_and_lower, MTM_no_reg)  # X = MTM^{-1} @ MTM_no_reg
            covphi = cho_solve(c_and_lower, X.T).T  # covphi = MTM^{-1} @ MTM_no_reg @ MTM^{-1}

            # The condition number is computed on covphi in the original units of the linear parameters
            # (C^-1 covphi' C^-1), so that this warning is triggered exactly as before the rescaling.
            eigvals = np.linalg.eigvalsh(covphi / np.outer(_para_scale, _para_scale))  # covphi should be symmetric PSD
            cond = np.abs(eigvals.max() / eigvals.min())
            # print("condition number:", cond, nonlin_paras)
            # print("min/max eigenvalue:", eigvals.min(), eigvals.max())
            if cond > 1e10:
                # Only print the nonlinear parameters when they are small. Some callers (e.g., the 3D spline fit)
                # pass whole data arrays as nonlin_paras, which would fill the logs.
                if isinstance(nonlin_paras, (list, tuple)):
                    nonlin_size = sum(np.size(p) for p in nonlin_paras)
                else:
                    nonlin_size = np.size(nonlin_paras)
                if nonlin_size <= 20:
                    nonlin_str = str(nonlin_paras)
                else:
                    nonlin_str = f"<{nonlin_size} values, not printed>"
                warning_text = f"conditioning number too high (>1e10): cond={cond} for nonlin={nonlin_str}"
                warn(warning_text)
                # raise Exception(warning_text)
            # The formula below assumes that we are using the determinant of the inverse covariance
            # That's why we are adding the minus sign
            # log|covphi| = log|MTM_no_reg| - 2 log|MTM| (covphi = MTM^-1 MTM_no_reg MTM^-1). It is computed this way
            # rather than with slogdet(covphi), because covphi itself can be very badly conditioned (parameters
            # constrained almost only by their prior have a tiny variance), while MTM is well conditioned after the
            # rescaling, and MTM_no_reg is well conditioned once normalized by its own diagonal.
            logdet_MTM = 2 * np.sum(np.log(np.diag(c_and_lower[0])))  # from the Cholesky factor
            _diag_no_reg = np.sqrt(np.diag(MTM_no_reg))
            slogdet_MTM_no_reg = np.linalg.slogdet(MTM_no_reg / np.outer(_diag_no_reg, _diag_no_reg))
            logdet_MTM_no_reg = slogdet_MTM_no_reg[1] + 2 * np.sum(np.log(_diag_no_reg))
            # covphi = C covphi_orig C, so -log|covphi_orig| = -log|covphi| + 2 sum(log c)
            logdet_icovphi0 = 2 * logdet_MTM - logdet_MTM_no_reg + logdet_para_scale
            # print(logdet_icovphi0)
            # logdet_icovphi0 = -np.sum(np.log(np.diag(covphi)))
            # print(logdet_icovphi0,"3")
        except Exception as e:
            # only printing the error message, but will not stop because of it
            # Will simply return the outputs corresponding to invalid data.
            print("Exiting covariance section in fitfm() with error:")
            print(e)
            return _invalid_outputs(N_linpara, return_covariance)

    diagcovphi = copy(np.diag(covphi))
    diagcovphi[np.where(diagcovphi < 0.0)] = np.nan  # uncertainties cannot be negative so replacing by nan here
    paras_err = np.sqrt(diagcovphi)  # get the uncertainties via the diagonal of the matrix

    # Caution: One need to be mindful of N_data:
    # N_data the size of d, but it can vary depending on the position  of the planet and the number of bad pixels for example, which can change the size of the data vector.
    # This means that the log_prob is not comparable between different planet positions.
    # To fix this issue, one needs to fix the data vector in the forward model function, such that it does not depend on the non-linear parameters.
    if marginalize_noise_scaling and not is_regularized:
        # Eq 41 in Ruffio+2019, https://ui.adsabs.harvard.edu/abs/2019AJ....158..200R/abstract
        log_prob = (N_fit_paras - N_data) / 2 * np.log(2 * np.pi) - 0.5 * logdet_Sigma - 0.5 * logdet_icovphi0 \
                   - ((N_data - N_fit_paras + 2 - 1) / 2) * np.log(chi2) + loggamma((N_data - N_fit_paras + 2 - 1) / 2)
    else:
        # log(Eq 36) in Ruffio+2019, https://ui.adsabs.harvard.edu/abs/2019AJ....158..200R/abstract
        log_prob = ((N_fit_paras - N_data) / 2) * np.log(2 * np.pi) - 0.5 * logdet_Sigma - 0.5 * logdet_icovphi0 \
                   - ((N_data - N_fit_paras) / 2) * np.log(noise_scaling ** 2) - 0.5 * chi2 / noise_scaling ** 2

    # Initialize the arrays of best-fit linear parameters and their uncertainties
    linparas = np.full(N_linpara, np.nan)
    linparas_err = np.full(N_linpara, np.nan)

    # Bookeeping the valid best fit linear parameters, back to their original units (phi = phi' / c)
    linparas[validpara] = paras / _para_scale
    linparas_err[validpara] = paras_err / _para_scale

    if return_covariance:
        linparas_cov = np.full((N_linpara, N_linpara), np.nan)
        linparas_cov[np.ix_(validpara[0], validpara[0])] = covphi / np.outer(_para_scale, _para_scale)
        return log_prob, rchi2, linparas, linparas_err, linparas_cov
    return log_prob, rchi2, linparas, linparas_err

def log_prob(nonlin_paras, dataobj, fm_func, fm_paras, nonlin_lnprior_func=None, bounds=None, scale_noise=True, marginalize_noise_scaling=False):
    """
    Wrapper to fit_fm() but only returns the log probability marginalized over the linear parameters.
    """
    if nonlin_lnprior_func is not None:
        prior = nonlin_lnprior_func(nonlin_paras)
    else:
        prior = 0
    try:
        lnprob = fitfm(nonlin_paras, dataobj, fm_func, fm_paras,bounds=bounds,scale_noise=scale_noise, marginalize_noise_scaling=marginalize_noise_scaling)[0]+prior
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

def _invalid_outputs(N_linear_parameters, return_covariance=False):
    """Helper to return invalid outputs"""
    linparas = np.full(N_linear_parameters, np.nan)
    linparas_err = np.full(N_linear_parameters, np.nan)
    log_prob = -np.inf
    s2 = np.inf

    if return_covariance:
        return log_prob, s2, linparas, linparas_err, np.full((N_linear_parameters, N_linear_parameters), np.nan)
    return log_prob, s2, linparas, linparas_err


def _abs_normalized_column_sums(M, s, target_block_bytes=32 * 1024 ** 2):
    """Compute ``np.nansum(np.abs(M / s[:, None]), axis=0)`` with less memory, identical bit for bit.

    The direct expression allocates a chain of full-size temporaries (division, absolute value, and the NaN-replaced copy
    and mask made by np.nansum). Here a single full-size buffer is filled by blocks of rows, NaNs are replaced block by
    block, and one full-width reduction is taken at the end (numpy's pairwise summation along axis 0 depends on the array
    shape, so summing partial sums over blocks would not reproduce it exactly).

    Parameters
    ----------
        M : np.ndarray
            2D model matrix of shape (N_data, N_linpara).
        s : np.ndarray
            1D noise standard deviation vector of shape (N_data,).
        target_block_bytes : int, optional
            Approximate size of each block of rows; it only affects memory traffic, not the result.

    Returns
    -------
        np.ndarray
            1D array of shape (N_linpara,): sum of the absolute noise-normalized values of each column.
    """
    n_rows, n_cols = M.shape
    dtype = np.result_type(M.dtype, s.dtype)
    if n_cols == 0 or n_rows == 0:
        return np.nansum(np.abs(M / s[:, None]), axis=0)
    block_rows = int(np.clip(target_block_bytes // max(1, n_cols * dtype.itemsize), 1, n_rows))
    buf = np.empty((n_rows, n_cols), dtype=dtype)
    for i0 in range(0, n_rows, block_rows):
        i1 = min(i0 + block_rows, n_rows)
        block = buf[i0:i1]
        np.abs(M[i0:i1] / s[i0:i1, None], out=block)
        block[np.isnan(block)] = 0  # what np.nansum does
    return buf.sum(axis=0)


def _residual_stats(M_normalized, d_normalized, paras, N_data=None):
    """Model, residuals, chi2, reduced chi2 and noise scaling for given linear parameters (shared by the lsq_linear and
    Cholesky paths, so that both derive them identically). Rows beyond N_data (regularization) are not in chi2."""
    d_estimated = _matvec(M_normalized, paras)
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

    return d_estimated, residuals, chi2, rchi2, noise_scaling


def _get_lsq_fit(M_normalized, d_normalized, _bounds, N_data=None):
    """Helper to get the least squares fit of the model on the data. Model matrix and input data have to be normalized by the noise."""
    paras = lsq_linear(M_normalized, d_normalized, bounds=_bounds).x
    d_estimated, residuals, chi2, rchi2, noise_scaling = _residual_stats(M_normalized, d_normalized, paras, N_data=N_data)
    return paras, d_estimated, residuals, chi2, rchi2, noise_scaling


def _bounds_are_infinite(_bounds):
    """True if none of the linear parameters has a finite bound."""
    return bool(np.all(np.isneginf(_bounds[0])) and np.all(np.isposinf(_bounds[1])))


def _matvec(M, v):
    """M v for a dense array or a scipy sparse matrix."""
    return M @ v if sparse.issparse(M) else np.dot(M, v)


def _symmetric_gram(M):
    """M^T M (dense), symmetrized to remove the rounding asymmetries. M can be a scipy sparse matrix."""
    MTM = _sparse_gram(M) if sparse.issparse(M) else np.dot(M.T, M)
    return 0.5 * (MTM + MTM.T)


def _sparse_gram(M):
    """Dense M^T M for a sparse matrix whose rows share a small number of sparsity patterns.

    The rows are grouped by their set of non-zero columns (for a tensor product of B-splines, the knot interval of the
    sample along each axis), and each group contributes a small dense product V^T V, which is much faster than the
    sparse matrix product.
    """
    M = sparse.csr_matrix(M)
    M.sort_indices()
    N_col = M.shape[1]
    MTM = np.zeros((N_col, N_col))
    nnz_per_row = np.diff(M.indptr)
    for nnz in np.unique(nnz_per_row):
        if nnz == 0:
            continue
        rows = np.flatnonzero(nnz_per_row == nnz)
        positions = M.indptr[rows][:, None] + np.arange(nnz)[None, :]
        indices, values = M.indices[positions], M.data[positions]
        # group the rows by their column indices: 1D hash (much faster than np.unique(..., axis=0)), then check that
        # all the rows of each group really have the same columns
        weights = np.random.default_rng(0).integers(1, 2 ** 62, size=nnz, dtype=np.int64)
        keys = indices.astype(np.int64) @ weights
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        group_bounds = np.concatenate(([0], np.flatnonzero(np.diff(sorted_keys)) + 1, [sorted_keys.size]))
        for start, end in zip(group_bounds[:-1], group_bounds[1:]):
            group = order[start:end]
            cols = indices[group[0]]
            if not np.array_equal(indices[group], np.broadcast_to(cols, (group.size, nnz))):
                # hash collision: exact grouping for these rows
                patterns, inverse = np.unique(indices[group], axis=0, return_inverse=True)
                for g, cols_g in enumerate(patterns):
                    V = values[group[inverse.ravel() == g]]
                    MTM[np.ix_(cols_g, cols_g)] += V.T @ V
                continue
            V = values[group]
            MTM[np.ix_(cols, cols)] += V.T @ V
    return MTM


def _cholesky_normal_equations(MTM, MTd, max_cond=1e10):
    """Solve the normal equations MTM x = MTd of an unbounded least squares problem by a Cholesky factorization.

    Returns (c_and_lower, x), with c_and_lower the output of scipy.linalg.cho_factor, or None if the problem is not
    suited to this approach, in which case the caller falls back to lsq_linear:
    - MTM is not positive definite (e.g. rank deficient model matrix), so cho_factor fails;
    - MTM is badly conditioned: forming the normal equations squares the condition number of the model matrix, and
      the relative error of the solution is about cond(MTM) * 1e-16 (instead of sqrt(cond(MTM)) * 1e-16 for
      lsq_linear). The 1-norm condition number of MTM is estimated by LAPACK (pocon) from the Cholesky factor and
      compared to max_cond (1e10: relative error below ~1e-6). In fitfm the columns are rescaled to unit norm
      beforehand, which keeps cond(MTM) small for well posed problems.
    """
    if MTM.shape[0] == 0:
        return None
    try:
        c_and_lower = cho_factor(MTM)
    except (np.linalg.LinAlgError, ValueError):
        # LinAlgError: not positive definite, ValueError: non-finite entries
        return None
    if not np.all(np.diag(c_and_lower[0]) > 0):
        return None
    pocon, = get_lapack_funcs(("pocon",), (c_and_lower[0],))
    rcond, info = pocon(c_and_lower[0], np.linalg.norm(MTM, 1), uplo="L" if c_and_lower[1] else "U")
    if info != 0 or not rcond > 1 / max_cond:
        return None
    x = cho_solve(c_and_lower, MTd)
    if not np.all(np.isfinite(x)):
        return None
    return c_and_lower, x


def _fit_regularized_normal_equations(M_no_reg, d_no_reg, d_reg, s_reg, reg_ids, scale_noise):
    """Regularized unbounded least squares fit of fitfm through the normal equations.

    Equivalent to stacking the data (M_no_reg, d_no_reg) with the diagonal regularization rows (1/s_reg on the columns
    reg_ids, d_reg/s_reg) and calling _get_lsq_fit, a second time with the data rows divided by the noise scaling if
    scale_noise (see fitfm). The Gram matrix of the data is computed once and reused for both fits and for the
    covariance. M_no_reg and d_no_reg have to be normalized by the noise (and the parameters rescaled as in fitfm).

    Returns (paras, chi2, rchi2, noise_scaling, MTM_no_reg, MTM, c_and_lower) with the same meaning as in fitfm (MTM
    and c_and_lower of the final fit), or None if a normal matrix is not suited to the Cholesky approach.
    """
    N_data = np.size(d_no_reg)
    MTM_no_reg = _symmetric_gram(M_no_reg)
    MTd_no_reg = _matvec(M_no_reg.T, d_no_reg)
    w_reg = 1 / s_reg ** 2  # diagonal of M_reg^T M_reg, on the columns reg_ids
    MTd_reg = d_reg / s_reg ** 2  # M_reg^T (d_reg / s_reg)

    def _solve(data_weight):
        # Normal equations of [M_no_reg / sqrt(data_weight)^-1; M_reg] x = [d_no_reg * sqrt(data_weight); d_reg/s_reg]
        MTM = MTM_no_reg * data_weight
        MTM[reg_ids, reg_ids] += w_reg
        MTd = MTd_no_reg * data_weight
        MTd[reg_ids] += MTd_reg
        out = _cholesky_normal_equations(MTM, MTd)
        return None if out is None else (MTM, out[0], out[1])

    # first fit, chi2 on the data rows only (_get_lsq_fit with N_data)
    first = _solve(1.0)
    if first is None:
        return None
    MTM, c_and_lower, paras = first
    _, _, chi2, rchi2, noise_scaling = _residual_stats(M_no_reg, d_no_reg, paras, N_data=N_data)
    if scale_noise:
        # second fit with the data rows divided by the noise scaling, chi2 on all the rows (_get_lsq_fit without N_data)
        if not (np.isfinite(noise_scaling) and noise_scaling > 0):
            return None
        second = _solve(1 / noise_scaling ** 2)
        if second is None:
            return None
        MTM, c_and_lower, paras = second
        res_data = (d_no_reg - _matvec(M_no_reg, paras)) / noise_scaling
        res_reg = (d_reg - paras[reg_ids]) / s_reg
        chi2 = np.nansum(res_data ** 2) + np.nansum(res_reg ** 2)
    return paras, chi2, rchi2, noise_scaling, MTM_no_reg, MTM, c_and_lower
