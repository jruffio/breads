import numpy as np
from scipy.optimize import lsq_linear
from scipy.special import loggamma
import matplotlib.pyplot as plt
import warnings
from copy import copy

__all__ =  ('fitfm', 'log_prob', 'combined_log_prob', 'nlog_prob')

def fitfm(nonlin_paras, dataobj, fm_func, fm_paras,computeH0 = True,bounds = None,
          residuals=None,residuals_H0=None,noise4residuals=None,scale_noise=True,marginalize_noise_scaling=False,
          debug=False):
    """
    Fit a forard model to data returning probabilities and best fit linear parameters.

    Args:
        nonlin_paras: [p1,p2,...] List of non-linear parameters such as rv, y, x. The meaning and number of non-linear
            parameters depends on the forward model defined.
        dataobj: A data object of type breads.instruments.instrument.Instrument to be analyzed.
        fm_func: A forward model function. See breads.fm.template.template() for an example.
        fm_paras: Additional parameters for fm_func (other than non-linear parameters and dataobj)
        computeH0: If true (default), compute the probability of the model removing the first element of the linear
            model; See second ouput log_prob_H0. This can be used to compute the Bayes factor for a fixed set of
            non-linear parameters
        bounds: (/!\ Caution: the calculation of log prob is only theoretically accurate if no bounds are used.)
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
    fm_out = fm_func(nonlin_paras,dataobj,**fm_paras)

    if len(fm_out) == 3:
        d_no_reg, M_no_reg, s_no_reg = fm_out
    if len(fm_out) == 4:
        d_no_reg, M_no_reg, s_no_reg,extra_outputs = fm_out

    N_linpara = M_no_reg.shape[1]
    N_data = np.size(d_no_reg)
    linparas = np.ones(N_linpara)+np.nan
    linparas_err = np.ones(N_linpara)+np.nan
    if N_data == 0:
        log_prob = -np.inf
        log_prob_H0 = -np.inf
        s2 = np.inf
        return log_prob, log_prob_H0, s2, linparas, linparas_err

    if N_linpara == 1:
        computeH0 = False

    if bounds is None:
        _bounds = ([-np.inf,]*N_linpara,[np.inf,]*N_linpara)
    else:
        _bounds = (copy(bounds[0]),copy(bounds[1]))

    validpara = np.where(np.nanmax(np.abs(M_no_reg),axis=0)!=0)

    if 0 not in validpara[0]:
        log_prob = -np.inf
        log_prob_H0 = -np.inf
        s2 = np.inf
        return log_prob, log_prob_H0, s2, linparas, linparas_err

    _bounds = (np.array(_bounds[0])[validpara[0]],np.array(_bounds[1])[validpara[0]])
    M_no_reg = M_no_reg[:,validpara[0]]

    d_no_reg = d_no_reg / s_no_reg
    M_no_reg = M_no_reg / s_no_reg [:, None]

    if len(fm_out) == 4:
        if "regularization" in extra_outputs.keys() and marginalize_noise_scaling:
            raise Exception("The maths for the marginalization of the noise scaling factor is not compatible with the regularization. Set marginalize_noise_scaling = False")

        if "regularization" in extra_outputs.keys():
            d_reg,s_reg = extra_outputs["regularization"]
            s_reg = s_reg[validpara]
            d_reg = d_reg[validpara]
            where_reg = np.where(np.isfinite(s_reg))
            s_reg = s_reg[where_reg]
            d_reg = d_reg[where_reg]
            M_reg = np.zeros((np.size(where_reg[0]),M_no_reg.shape[1]))
            M_reg[np.arange(np.size(where_reg[0])),where_reg[0]] = 1/s_reg
            M = np.concatenate([M_no_reg,M_reg],axis=0)
            d = np.concatenate([d_no_reg,d_reg/s_reg])
            s = np.concatenate([s_no_reg,s_reg])

            if scale_noise:
                paras = lsq_linear(M, d, bounds=_bounds).x
                m = np.dot(M, paras)
                r = d - m
                rchi2 = np.nansum(r[0:N_data] ** 2) / N_data
                noise_scaling = np.sqrt(rchi2)

                M = np.concatenate([M_no_reg/noise_scaling,M_reg],axis=0)
                d = np.concatenate([d_no_reg/noise_scaling,d_reg/s_reg])
                s = np.concatenate([s_no_reg*noise_scaling,s_reg])

            # noise scaling is done, set to unity for later as no more scaling is necessary
            noise_scaling = 1
        else:
            M = M_no_reg
            d = d_no_reg
            s = s_no_reg
    else:
        M = M_no_reg
        d = d_no_reg
        s = s_no_reg

    logdet_Sigma = np.sum(2 * np.log(s))
    paras = lsq_linear(M, d,bounds=_bounds).x
    # paras = lsq_linear(M, d).x

    m = np.dot(M, paras)
    r = d  - m
    chi2 = np.nansum(r**2)
    # s2 = chi2 / np.size(r)

    if residuals is not None:
        residuals[0:np.size(s)] = r
    if noise4residuals is not None:
        noise4residuals[0:np.size(s)] = s
    # plt.figure()
    # for col in M.T:
    #     plt.plot(col / np.nanmean(col))
    # plt.show()

    # Section to compute error bars of linear parameters
    MTM = np.dot(M.T, M)
    try:
        iMTM = np.linalg.inv(MTM)
        if len(fm_out) == 4 and "regularization" in extra_outputs.keys():
            if not scale_noise:
                rchi2 = np.nansum(r[0:N_data] ** 2) / N_data
            MTM_noreg = np.dot(M_no_reg.T,M_no_reg)
            covphi = np.dot(iMTM,np.dot(MTM_noreg,iMTM.T))
            # The formula below assumes that we are using the determinant of the inverse covariance
            # That's why we are adding the minus sign
            logdet_icovphi0 = -np.sum(np.log(np.diag(covphi)))


            if debug:
                pass
                slogdet_covphi0 = np.linalg.slogdet(covphi)
                logdet_icovphi01 = -slogdet_covphi0[1]

                logdet_MTM = np.linalg.slogdet(MTM)[1]
                logdet_MTM_noreg = np.linalg.slogdet(MTM_noreg)[1]
                logdet_icovphi02 = -(-2*logdet_MTM+logdet_MTM_noreg)
                print("logdet_icovphi02,logdet_icovphi0",logdet_icovphi01,logdet_icovphi02,-np.sum(np.log(np.diag(covphi))))
                plt.plot(np.diag(covphi))
                plt.show()
                exit()
                # min_spline_ampl = 0.02 no PCs
                #66385.44310047594 66385.0385737088 min_spline_ampl = 0.02
                #67934.07623968158 67943.01648052462 min_spline_ampl = 0.0002
                # print(np.linalg.cond(MTM),np.linalg.cond(MTM_noreg))
                #193404.75080185774 3.059098264262221e+18  # min_spline_ampl = 0.02
                #193405.9507460225 1.1014045133611385e+19 # min_spline_ampl = 0.001
                #193405.9576081098 3.8402652368209265e+19 # min_spline_ampl = 0.00001
                # exit()
                # print(np.linalg.slogdet(MTM))
                # print(np.linalg.slogdet(MTM_noreg))
                # plt.figure(1)
                # eigenvalues,eigenvectors = np.linalg.eig(MTM)
                # plt.plot(eigenvalues)
                # plt.figure(2)
                # eigenvalues,eigenvectors = np.linalg.eig(MTM_noreg)
                # plt.plot(eigenvalues)
                # plt.figure(3)
                # plt.subplot(1,3,1)
                # plt.imshow(MTM)
                # plt.subplot(1,3,2)
                # plt.imshow(MTM_noreg)
                # plt.subplot(1,3,3)
                # plt.imshow(MTM-MTM_noreg)
                # plt.figure(4)
                # plt.subplot(1,3,1)
                # plt.imshow(M,aspect="auto")
                # plt.subplot(1,3,2)
                # plt.imshow(M_no_reg,aspect="auto")
                # plt.subplot(1,3,3)
                # plt.imshow(M_reg,aspect="auto")
                #
                # plt.figure(5)
                # plt.plot(np.log10(np.max(M,axis=1)),label="M")
                # plt.plot(np.log10(np.max(M_no_reg,axis=1)),label="M_no_reg")
                # plt.plot(np.log10(np.max(M_reg,axis=1)),label="M_reg")
                # plt.legend()
                #
                #
                # plt.figure(6)
                # plt.plot(np.log10(np.max(M,axis=0)),label="M")
                # plt.plot(np.log10(np.max(M_no_reg,axis=0)),label="M_no_reg")
                # plt.plot(np.log10(np.max(M_reg,axis=0)),label="M_reg")
                # plt.legend()
                # plt.show()
                # exit()

        else:
            rchi2 = np.nansum(r ** 2) / N_data
            if scale_noise:
                noise_scaling = np.sqrt(rchi2)
            else:
                noise_scaling = 1
            covphi = noise_scaling * iMTM
            # covphi = np.linalg.inv(MTM)
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
        log_prob = -np.inf
        log_prob_H0 = -np.inf
        rchi2 = np.inf
        return log_prob, log_prob_H0, rchi2, linparas, linparas_err
    diagcovphi = copy(np.diag(covphi))
    diagcovphi[np.where(diagcovphi<0.0)] = np.nan
    paras_err = np.sqrt(diagcovphi)

    # print("log_prob",logdet_Sigma,slogdet_icovphi0[1],(N_data - N_linpara + 2 - 1),chi2)
    # print("slogdet_icovphi0",slogdet_icovphi0)

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
        paras_H0 = lsq_linear(M[:,1::], d,bounds=(np.array(_bounds[0])[1::],np.array(_bounds[1])[1::])).x
        # paras_H0 = lsq_linear(M[:,1::], d).x
        m_H0 = np.dot(M[:,1::] , paras_H0)
        r_H0 = d  - m_H0
        chi2_H0 = np.nansum(r_H0**2)
        # rchi2_H0 = np.nansum(r_H0[0:N_data]**2) / N_data
        slogdet_icovphi0_H0 = np.linalg.slogdet(np.dot(M[:,1::].T, M[:,1::]))
        #todo check the maths when N_linpara is different from M.shape[1]. E.g. at the edge of the FOV
        # log_prob_H0 = -0.5*logdet_Sigma - 0.5*slogdet_icovphi0_H0[1] - (N_data-1+N_linpara-1-1)/2*np.log(chi2_H0)+ \
        #               loggamma((N_data-1+(N_linpara-1)-1)/2)+((N_linpara-1)-N_data)/2*np.log(2*np.pi)
        if marginalize_noise_scaling:
            log_prob_H0 = -0.5*logdet_Sigma - 0.5*slogdet_icovphi0_H0[1] - (N_data+(M.shape[1]-1)+2-1)/2*np.log(chi2_H0)+ \
                      loggamma((N_data-(M.shape[1]-1)+2-1)/2)+((M.shape[1]-1)-N_data)/2*np.log(2*np.pi)
        else:
            # log(Eq 36) in Ruffio+2019:
            log_prob_H0 = ((M.shape[1]-N_data)/2)*np.log(2*np.pi) -0.5 * logdet_Sigma - 0.5 * slogdet_icovphi0_H0[1] \
                       -((N_data-M.shape[1])/2) * np.log(noise_scaling**2) -0.5*chi2_H0/noise_scaling**2
        if residuals_H0 is not None:
            residuals_H0[0:np.size(s)] = r_H0
    else:
        log_prob_H0 = np.nan

    linparas[validpara] = paras
    linparas_err[validpara] = paras_err

    # import matplotlib.pyplot as plt
    # print(log_prob, log_prob_H0, rchi2)
    # print(linparas)
    # print(linparas_err)
    # plt.plot(d,label="d")
    # plt.plot(m,label="m")
    # plt.plot(r,label="r")
    # plt.legend()
    # plt.show()


    return log_prob, log_prob_H0, rchi2, linparas, linparas_err

def log_prob(nonlin_paras, dataobj, fm_func, fm_paras,nonlin_lnprior_func=None,bounds=None):
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
        bounds: (/!\ Caution: the calculation of log prob is only theoretically accurate if no bounds are used.)
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
        lnprob = fitfm(nonlin_paras, dataobj, fm_func, fm_paras,computeH0=False,bounds=bounds)[0]+prior
    except:
        lnprob =  -np.inf
    return lnprob


def combined_log_prob(nonlin_paras, dataobjlist,fm_funclist, fm_paraslist, nonlin_lnprior_func=None,bounds=None):
    '''
    For use when you have multiple data objects and want to combine the log-likelihoods for MCMC sampling

    Args:
        nonlin_paras: [p1,p2,...] List of non-linear parameters such as rv, y, x. The meaning and number of non-linear
            parameters depends on the forward model defined.
        dataobjlist: A list of data objects to combine
        fm_funclist: A list of fm_func for each data object
        fm_paraslist: A list of fm_paras to use as arguments for it's respective fm_func
        nonlin_lnprior_func: A function to compute priors, if None, defaults to zero priors
    '''
    combined_lnprob = 0
    for i, dataobj in enumerate(dataobjlist):
        lnprob=log_prob(nonlin_paras = nonlin_paras, dataobj=dataobj, fm_func=fm_funclist[i], fm_paras= fm_paraslist[i],nonlin_lnprior_func=None,bounds=None)
        combined_lnprob += lnprob
    return combined_lnprob

def nlog_prob(nonlin_paras, dataobj, fm_func, fm_paras,nonlin_lnprior_func=None,bounds=None):
    """
   Returns the negative of the log_prob() for minimization routines.

    Args:
        nonlin_paras: [p1,p2,...] List of non-linear parameters such as rv, y, x. The meaning and number of non-linear
            parameters depends on the forward model defined.
        dataobj: A data object of type breads.instruments.instrument.Instrument to be analyzed.
        fm_func: A forward model function. See breads.fm.template.template() for an example.
        fm_paras: Additional parameters for fm_func (other than non-linear parameters and dataobj)
        computeH0: If true (default), compute the probability of the model removing the first element of the linear
            model; See second ouput log_prob_H0. This can be used to compute the Bayes factor for a fixed set of
            non-linear parameters
        bounds: (/!\ Caution: the calculation of log prob is only theoretically accurate if no bounds are used.)
            Bounds on the linear parameters used in lsq_linear as a tuple of arrays (min_vals, maxvals).
            e.g. ([0,0,...], [np.inf,np.inf,...]) default no bounds.
            Each numpy array must have shape (N_linear_parameters,).

    Returns:
        log_prob: Probability of the model marginalized over linear parameters.
    """
    nlogprob_val =  - log_prob(nonlin_paras, dataobj, fm_func, fm_paras,nonlin_lnprior_func,bounds)
    # print( nlogprob_val, nonlin_paras)
    return nlogprob_val