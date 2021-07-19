import numpy as np

def templatefm(nonlin_paras, cubeobj, para1=None,para2=None):
    """
    Unfinished: To be a reference for people to create their own forward model

    Number of linear parameters should remain constant for a given cubeobj and fixed extra parameters.
    be mindful of image edges. Use another fm function as model if needed.
    The output should not include any nans
    The first linear parameter is assumed to be the flux of the planet. It it is not, then the H1 vs H0 hypothesis is simply not true but it's fione
    If d is empty, return
        d, M, s = np.array([]), np.array([]).reshape(0, N_linpara), np.array([])

    Returns:
        d: Data as a 1d vector with bad pixels removed (no nans)
        M: Linear model as a matrix of shape (Nd,Np) with bad pixels removed (no nans). Nd is the size of the data
            vector and Np is the number of linear parameters.
        s: Noise vector (standard deviation) as a 1d vector matching d.
    """
    pass
    N_linpara = 0
    d, M, s = np.array([]), np.array([]).reshape(0, N_linpara), np.array([])
    return