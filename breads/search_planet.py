import multiprocessing as mp
import numpy as np
import itertools

try:
    import mkl
    mkl_exists = True
except ImportError:
    mkl_exists = False

def process_chunk(args):
    nonlin_paras_list, dataobj, fm_func, fm_paras = args

    out_chunk = np.zeros(np.size(nonlin_paras_list[0]))
    for k, nonlin_paras in enumerate(zip(*nonlin_paras_list)):
        out_chunk[k] = fm_func(nonlin_paras,dataobj,**fm_paras)

    return out_chunk


def search_planet(para_vecs,dataobj,fm_func,fm_paras,numthreads=None):
    para_grids = [np.ravel(pgrid) for pgrid in np.meshgrid(*para_vecs)]

    out = np.zeros([len(para_vecs[k]) for k in range(len(para_vecs))])
    print(out.shape)

    if numthreads is None:
        for k, nonlin_paras in enumerate(zip(*para_grids)):
            out[np.unravel_index(k,out.shape)] = fm_func(nonlin_paras,dataobj,**fm_paras)
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

        output_lists = mypool.map(process_chunk, zip(nonlin_paras_lists,itertools.repeat(dataobj),itertools.repeat(fm_func),itertools.repeat(fm_paras)))

        for indices, output_list in zip(indices_lists,output_lists):
            for k,val in zip(indices,output_list):
                out[np.unravel_index(k,out.shape)] = val

    return out
