import numpy as np

def templatefm(nonlin_paras, data_obj, para1=None,para2=None):
    """
    Template forward model (FM) to be used to create custom forward model functions.
    The goal of a forward model function is to build the data vector (d), linear model matrix (M), and noise vector (s) for a given set of non-linear parameters and a data object.
    It provides everything that is needed to run a linear least squares fit for the linear parameters of the model, which is typically done in the fitfm() function.
    d, M, and s should not include any nans, so it is the goal of the FM function to remove any bad pixels from the data and the model.
    The non-linear parameters are typically the parameters that define the shape of the planet signal (e.g. RV, Teff, logg, etc.) and the linear parameters are typically the parameters that define the flux of the planet signal and any other linear components (e.g. starlight, etc.).
    The forward model function should be defined such that the first linear parameter(s) correspond to the planet signal, and any additional linear parameters correspond to other components (e.g. starlight, etc.).

    Number of linear parameters (N_linpara) should remain constant for a given data object (see instrument class) and fixed extra parameters.


    Parameters
    ----------
        nonlin_paras : array-like
            A 1d array of non-linear parameters.
        data_obj : Instrument
            An instance of the Instrument class containing the data to be analyzed and any relevant information to build the forward model like x,y, lambda coordinates.
        para1 : any type
            Additional parameters for the forward model function. This can be used to pass any extra parameters that might be needed to build the forward model, such as a grid of planet spectra, a regularization term, etc. The meaning and type of these parameters is up to the user and should be defined in the context of the specific forward model being implemented.
        para2 : any type
            Ditto

    Returns
    -------
        d : np.ndarray
            Data as a 1d vector with bad pixels removed (no nans)
        M : np.ndarray
            Linear model as a matrix of shape (Nd,Np) with bad pixels removed (no nans). Nd is the size of the data vector and Np is the number of linear parameters.
        s : np.ndarray
            Noise vector (standard deviation) as a 1d vector matching d.
        extra_outputs : dict
            Optional output: A dictionary containing any additional outputs that might be useful for the analysis.
            This might include a regularization for the linear parameters.
            Or this can include any custom outputs specific to the forward model that can be used for further analysis or visualization.
    """
    N_linpara = 0
    d, M, s = np.array([]), np.array([]).reshape(0, N_linpara), np.array([])

    extra_outputs = {}
    extra_outputs["N_planet_linparas"] = None # Number of linear parameters corresponding to the planet signal.

    # Use  extra_outputs["regularization"] to add a regularization term on the linear parameters:
    # Refer to Ruffio+2024 Eq. (A4) and relevant section for more details on the maths.
    # A Gaussian prior (other word for regularization) can be included by adding some extra terms to the data vector, the model matrix and the error vector.
    # d_reg are the mean of the Gaussian prior on the linear parameters.
    # s_reg are the standard deviation of the Gaussian prior on the linear parameters.
    # d_reg and s_reg have the same length as the number of linear parameters, but parameters without regularization are set to np.nan. fitfm() manages the bookkeeping accordinly.
    # (d_reg, s_reg) fully define the regularization of the linear parameter with Gaussian prior.
    # Note: the modification to the forward model matrix M_reg is done directly in fitfm, it is effectively some partial identity matrix.
    d_reg, s_reg = np.array([np.nan]), np.array([np.nan])
    extra_outputs["regularization"] = (d_reg, s_reg)

    # return d, M, s
    # or:
    return d, M, s, extra_outputs