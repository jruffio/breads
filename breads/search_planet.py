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
    """
    Process for search_planet()
    """
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
    """
    Planet detection, CCF, or grid search routine.
    It fits for the non linear parameters of a forward model over a user-specified grid of values while marginalizing
    over the linear parameters. For a planet detection or CCF routine, choose a forward model (fm_func) and provide a
    grid of x,y, and RV values.

    SNR (detection maps or CCFs) can be computed as:
    N_linpara = (out.shape[-1]-2)//2
    snr = out[...,3]/out[...,3+N_linpara]

    The natural logarithm of the Bayes factor can be computed as
    bayes_factor = out[...,0] - out[...,1]

    Args:
        para_vecs: [vec1,vec2,...] List of 1d arrays defining the sampling of the grid of non-linear parameters such as
            rv, y, x. The meaning and number of non-linear parameters depends on the forward model defined.
        dataobj: A data object of type breads.instruments.instrument.Instrument to be analyzed.
        fm_func: A forward model function. See breads.fm.template.template() for an example.
        fm_paras: Additional parameters for fm_func (other than non-linear parameters and dataobj)
        numthreads: Number of processes to be used in parallelization. Non parallization if defined as None (default).

    Returns:
        Out: An array of dimension (Nnl_1,Nnl_2,....,3+Nl*2) containing the marginalized probabilities, the noise
            scaling factor, and best fit linear parameters with associated uncertainties calculated over the grid of
            non-linear parameters. (Nnl_1,Nnl_2,..) is the shape of the non linear parameter grid with Nnl_1 the size of
            para_vecs[1] and so on. Nl is the number of linear parameters in the forward model. The last dimension
            is defined as follow:
                Out[:,...,0]: Probability of the model marginalized over linear parameters.
                Out[:,...,1]: Probability of the model without the planet marginalized over linear parameters.
                Out[:,...,2]: noise scaling factor
                Out[:,...,3:3+Nl]: Best fit linear parameters
                Out[:,...,3+Nl:3+2*Nl]: Uncertainties of best fit linear parameters

    """
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
