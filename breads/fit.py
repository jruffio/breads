import numpy as np

from scipy.optimize import lsq_linear
from scipy.special import loggamma

def fitfm(nonlin_paras, dataobj, fm_func, fm_paras,computeH0 = True):
    print(nonlin_paras)
    d,M,s = fm_func(nonlin_paras,dataobj,**fm_paras)
    N_linpara = M.shape[1]

    validpara = np.where(np.nansum(M,axis=0)!=0)
    M = M[:,validpara[0]]

    d = d / s
    M = M / s[:, None]

    N_data = np.size(d)
    linparas = np.ones(N_linpara)+np.nan
    linparas_err = np.ones(N_linpara)+np.nan
    if N_data == 0:
        log_prob = np.nan
        log_prob_H0 = np.nan
        rchi2 = np.nan
    else:
        logdet_Sigma = np.sum(2 * np.log(s))
        paras = lsq_linear(M, d).x

        m = np.dot(M, paras)
        r = d  - m
        chi2 = np.nansum(r**2)
        rchi2 = chi2 / N_data

        covphi = rchi2 * np.linalg.inv(np.dot(M.T, M))
        slogdet_icovphi0 = np.linalg.slogdet(np.dot(M.T, M))

        log_prob = -0.5*logdet_Sigma - 0.5*slogdet_icovphi0[1] - (N_data-1+N_linpara-1)/2*np.log(chi2)+\
                   loggamma((N_data-1+N_linpara-1)/2)+(N_linpara-N_data)/2*np.log(2*np.pi)
        paras_err = np.sqrt(np.diag(covphi))

        if computeH0:
            paras_H0 = lsq_linear(M[:,1::], d).x
            m_H0 = np.dot(M[:,1::] , paras_H0)
            r_H0 = d  - m_H0
            chi2_H0 = np.nansum(r_H0**2)
            slogdet_icovphi0_H0 = np.linalg.slogdet(np.dot(M[:,1::].T, M[:,1::]))
            #todo check the maths when N_linpara is different from M.shape[1]. E.g. at the edge of the FOV
            log_prob_H0 = -0.5*logdet_Sigma - 0.5*slogdet_icovphi0_H0[1] - (N_data-1+N_linpara-1-1)/2*np.log(chi2_H0)+ \
                          loggamma((N_data-1+(N_linpara-1)-1)/2)+((N_linpara-1)-N_data)/2*np.log(2*np.pi)
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

def log_prob(nonlin_paras, dataobj, fm_func, fm_paras):
    return fitfm(nonlin_paras, dataobj, fm_func, fm_paras,computeH0=False)[0]