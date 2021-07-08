import multiprocessing as mp
import numpy as np
import itertools

from scipy.optimize import lsq_linear
from scipy.special import loggamma
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy import constants as const

from breads.fit import fitfm

try:
    import mkl
    mkl_exists = True
except ImportError:
    mkl_exists = False



def process_chunk(args):
    nonlin_paras_list, dataobj, fm_func, fm_paras = args

    for k, nonlin_paras in enumerate(zip(*nonlin_paras_list)):
        log_prob,log_prob_H0,rchi2,linparas,linparas_err = fitfm(nonlin_paras,dataobj,fm_func,fm_paras)
        N_linpara = np.size(linparas)
        if k ==0:
            out_chunk = np.zeros((np.size(nonlin_paras_list[0]),1+1+1+2*N_linpara))
        out_chunk[k,0] = log_prob
        out_chunk[k,1] = log_prob_H0
        out_chunk[k,2] = rchi2
        out_chunk[k,3:(N_linpara+3)] = linparas
        out_chunk[k,(N_linpara+3):(2*N_linpara+3)] = linparas_err
    return out_chunk


def search_planet(para_vecs,dataobj,fm_func,fm_paras,numthreads=None):
    para_grids = [np.ravel(pgrid) for pgrid in np.meshgrid(*para_vecs)]

    if numthreads is None:
        _out = process_chunk((para_grids,dataobj,fm_func,fm_paras))
        out_shape = [np.size(v) for v in para_vecs]+[_out.shape[-1],]
        out = np.reshape(_out,out_shape)
    else:
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
                                                     itertools.repeat(fm_paras)))

        for k,(indices, output_list) in enumerate(zip(indices_lists,output_lists)):
            if k ==0:
                out_shape = [np.size(v) for v in para_vecs]+[np.size(output_list[0]),]
                out = np.zeros(out_shape)
            for l,outvec in zip(indices,output_list):
                out[np.unravel_index(l,out_shape[0:len(para_vecs)])] = outvec

        mypool.close()
        mypool.join()
    return out
