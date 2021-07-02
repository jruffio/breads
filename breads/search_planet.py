import multiprocessing as mp
import numpy as np
import itertools

from scipy.optimize import lsq_linear
from scipy.special import loggamma
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy import constants as const

try:
    import mkl
    mkl_exists = True
except ImportError:
    mkl_exists = False


def get_spline_model(x_knots, x_samples, spline_degree=3):
    M = np.zeros((np.size(x_samples), (np.size(x_knots))))
    for chunk in range(np.size(x_knots)):
        tmp_y_vec = np.zeros(np.size(x_knots))
        tmp_y_vec[chunk] = 1
        spl = InterpolatedUnivariateSpline(x_knots, tmp_y_vec, k=spline_degree, ext=0)
        M[:, chunk] = spl(x_samples)
    return M


def pixgauss2d(p, shape, hdfactor=10, xhdgrid=None, yhdgrid=None):
    A, xA, yA, w, bkg = p
    ny, nx = shape
    if xhdgrid is None or yhdgrid is None:
        xhdgrid, yhdgrid = np.meshgrid(np.arange(hdfactor * nx).astype(np.float) / hdfactor,
                                       np.arange(hdfactor * ny).astype(np.float) / hdfactor)
    else:
        hdfactor = xhdgrid.shape[0] // ny
    gaussA_hd = A / (2 * np.pi * w ** 2) * np.exp(
        -0.5 * ((xA - xhdgrid + 0.5) ** 2 + (yA - yhdgrid + 0.5) ** 2) / w ** 2)
    gaussA = np.nanmean(np.reshape(gaussA_hd, (ny, hdfactor, nx, hdfactor)), axis=(1, 3))
    return gaussA + bkg


def splinefm(nonlin_paras, cubeobj, planet_f=None, boxw=0, N_nodes=20, transmission=None, star_spectrum=None):
    """
    doc
    """
    rv,y,x = nonlin_paras
    nz, ny, nx = cubeobj.spaxel_cube.shape
    wvs = cubeobj.wavelengths

    k, l = cubeobj.refpos[1] + y, cubeobj.refpos[0] + x
    w = (boxw - 1) // 2
    d = np.ravel(cubeobj.spaxel_cube[:, k-w:k+w+1, l-w:l+w+1])
    s = np.ravel(cubeobj.noise_cube[:, k-w:k+w+1, l-w:l+w+1])
    badpixs = np.ravel(cubeobj.bad_pixel_cube[:, k-w:k+w+1, l-w:l+w+1])

    N_linpara = boxw * boxw * N_nodes+1
    # bounds_min = [-np.inf, ] +[0, ]* (N_linpara-1) # actually I don't think it works out with the maths TBD
    bounds_min = [-np.inf, ] * (N_linpara)
    bounds_max = [np.inf, ] * (N_linpara)
    where_finite = np.where(np.isfinite(badpixs))
    if np.size(where_finite[0]) <= 0.25 * (nz * boxw * boxw):
        return np.array([]), np.array([]), np.array([]), [bounds_min, bounds_max]
    else:
        x_knots = wvs[np.linspace(0, nz - 1, N_nodes, endpoint=True).astype(np.int)]
        M_spline = get_spline_model(x_knots, wvs, spline_degree=3)
        M_speckles = np.zeros((nz, boxw, boxw, boxw, boxw, N_nodes))
        for m in range(boxw):
            for n in range(boxw):
                M_speckles[:, m, n, m, n, :] = M_spline * star_spectrum[:, None]
        M_speckles = np.reshape(M_speckles, (nz, boxw, boxw, N_linpara-1))

        planet_spec = transmission * planet_f(wvs * (1 - (rv - cubeobj.bary_RV) / const.c.to('km/s').value))

        psfs = np.zeros((nz, boxw, boxw))
        hdfactor = 5
        psfwidth0 = 1.2
        xhdgrid, yhdgrid = np.meshgrid(np.arange(hdfactor * (boxw)).astype(np.float) / hdfactor,
                                       np.arange(hdfactor * (2 * w + 1)).astype(np.float) / hdfactor)
        psfs += pixgauss2d([1., w, w, psfwidth0, 0.], (boxw, boxw), xhdgrid=xhdgrid, yhdgrid=yhdgrid)[None, :, :]
        psfs = psfs / np.nansum(psfs, axis=(1, 2))[:, None, None]
        scaled_psfs = psfs * planet_spec[:, None, None]

        M = np.concatenate([scaled_psfs[:, :, :, None], M_speckles], axis=3)
        M = np.reshape(M, (nz * boxw * boxw, N_linpara))

        sr = s[where_finite]
        dr = d[where_finite]
        Mr = M[where_finite[0], :]

        return dr, Mr, sr, [bounds_min, bounds_max]

def fitfm(nonlin_paras, dataobj, fm_func, fm_paras):
    d,M,s,bounds = fm_func(nonlin_paras,dataobj,**fm_paras)
    d = d / s
    M = M / s[:, None]

    N_linpara = len(bounds[0])
    N_data = np.size(d)
    if N_data == 0:
        log_prob = np.nan
        log_prob_H0 = np.nan
        linparas = np.ones(N_linpara)+np.nan
        linparas_err = np.ones(N_linpara)+np.nan
    else:
        logdet_Sigma = np.sum(2 * np.log(s))

        linparas = lsq_linear(M, d,bounds=bounds).x

        m = np.dot(M, linparas)
        r = d  - m
        chi2 = np.nansum(r**2)

        covphi = chi2 / N_data * np.linalg.inv(np.dot(M.T, M))
        slogdet_icovphi0 = np.linalg.slogdet(np.dot(M.T, M))

        linparas_H0 = lsq_linear(M[:,1::], d,bounds=(bounds[0][1::], bounds[1][1::])).x
        m_H0 = np.dot(M[:,1::] , linparas_H0)
        r_H0 = d  - m_H0
        chi2_H0 = np.nansum(r_H0**2)
        slogdet_icovphi0_H0 = np.linalg.slogdet(np.dot(M[:,1::].T, M[:,1::]))

        log_prob = -0.5*logdet_Sigma - 0.5*slogdet_icovphi0[1] - (N_data-1+N_linpara-1)/2*np.log(chi2)+loggamma((N_data-1+N_linpara-1)/2)+(N_linpara-N_data)/2*np.log(2*np.pi)
        log_prob_H0 = -0.5*logdet_Sigma - 0.5*slogdet_icovphi0_H0[1] - (N_data-1+N_linpara-1-1)/2*np.log(chi2_H0)+loggamma((N_data-1+(N_linpara-1)-1)/2)+((N_linpara-1)-N_data)/2*np.log(2*np.pi)
        linparas_err = np.sqrt(np.diag(covphi))

        # import matplotlib.pyplot as plt
        # plt.plot(d,label="d")
        # plt.plot(m,label="m")
        # plt.plot(r,label="r")
        # plt.legend()
        # plt.show()

    return log_prob,log_prob_H0,linparas,linparas_err

# def log_prob(nonlin_paras, dataobj, fm_func, fm_paras):
#     return fitfm(nonlin_paras, dataobj, fm_func, fm_paras)[0]

def process_chunk(args):
    nonlin_paras_list, dataobj, fm_func, fm_paras,N_linpara = args

    out_chunk = np.zeros((np.size(nonlin_paras_list[0]),1+1+2*N_linpara))
    for k, nonlin_paras in enumerate(zip(*nonlin_paras_list)):
        log_prob,log_prob_H0,linparas,linparas_err = fitfm(nonlin_paras,dataobj,fm_func,fm_paras)
        out_chunk[k,0] = log_prob
        out_chunk[k,1] = log_prob_H0
        out_chunk[k,2:(N_linpara+2)] = linparas
        out_chunk[k,(N_linpara+2):(2*N_linpara+2)] = linparas_err
    return out_chunk


def search_planet(para_vecs,dataobj,fm_func,fm_paras,numthreads=None):
    para_grids = [np.ravel(pgrid) for pgrid in np.meshgrid(*para_vecs)]

    _,_,_,bounds = fm_func([v[0] for v in para_vecs], dataobj, **fm_paras)
    N_linpara = len(bounds[0])

    out_shape = [np.size(para_vecs[k]) for k in range(len(para_vecs))]+[1+1+2*N_linpara,]

    if numthreads is None:
        _out = process_chunk((para_grids,dataobj,fm_func,fm_paras,N_linpara))
        out = np.reshape(_out,out_shape)
    else:
        out = np.zeros(out_shape)
        mypool = mp.Pool(processes=numthreads)
        chunk_size = np.size(para_grids[0])//(3*numthreads)
        N_chunks = np.size(para_grids[0])//chunk_size
        nonlin_paras_lists = []
        indices_lists = []
        for k in range(N_chunks-1):
            nonlin_paras_lists.append([pgrid[(k*chunk_size):((k+1)*chunk_size)] for pgrid in para_grids])
            indices_lists.append(np.arange((k*chunk_size),((k+1)*chunk_size)))
        nonlin_paras_lists.append([pgrid[((N_chunks-1)*chunk_size):np.size(para_grids[0])] for pgrid in para_grids])
        indices_lists.append(np.arange(((N_chunks-1)*chunk_size),np.size(para_grids[0])))

        output_lists = mypool.map(process_chunk, zip(nonlin_paras_lists,
                                                     itertools.repeat(dataobj),
                                                     itertools.repeat(fm_func),
                                                     itertools.repeat(fm_paras),
                                                     itertools.repeat(N_linpara)))

        for indices, output_list in zip(indices_lists,output_lists):
            for k,outvec in zip(indices,output_list):
                out[np.unravel_index(k,out_shape[0:len(para_vecs)])] = outvec

    return out
