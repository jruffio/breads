import itertools
from glob import glob
import numpy as np
import os

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib.gridspec as gridspec

from astropy import constants as const
from astropy import units as u
from astropy.table import Table
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from tqdm import tqdm
from scipy.stats import median_abs_deviation


from breads.utils import rotate_coordinates

def fitpsf(dataobj, ref_dataobj = None,
           use_stpsf = False,use_breadspsf = None,
           stpsfs=None, stpsfX = None, stpsfY = None,
           out_filename=None, IWA=0, OWA=np.inf, mppool=None,
           init_centroid=None, fit_centroid=True, fit_angle = False,
           ann_width=None, padding=None, sector_area=None,
           linear_interp=True, rotate_psf=0.0, flipx=False,debug_wv_range=None,overwrite=False,
           stis_spectrum=None,poly_deg_coords=2,poly_deg_flux=1,
            wv_min = None,wv_max = None):
    """Fit a model PSF (psfs, psfX, psfY) to a combined dataset (dataobj_list).

    Parameters
    ----------
    dataobj : JWSTNirspec_multiple_cals object
        Combined dataset from multiple cal files
    ref_dataobj : JWSTNirspec_multiple_cals object
        Reference dataset to be fitted to dataobj.
    use_stpsf : bool
        Use STPSF for fitting. Use default BREADS features to define stpsfs, stpsfX, and stpsfY.
    use_breadspsf : str
        Basename of the BreadsPSF. eg "J1757132_G395H_nrs2.fits". BreadsPSF are assumed to be saved in os.path.join(BREADS_DATA_ENV, "BreadsPSF").
    stpsfs : np.array
        3D array of model PSFs. Needs to be defined on the same wv_sampling as dataobj.
    stpsfX : np.array
        3D array of X coordinates for the model PSFs.
    stpsfY : np.array
        3D array of Y coordinates for the model PSFs.
    out_filename : None
        If not None, save the best fit parameters and model to this filename as a fits file.
    IWA : float
        Inner Working Angle of the region to be fitted
    OWA : float
        Outer Working Angle of the region to be fitted
    mppool : multiprocessing.Pool
            If not None, use this multiprocessing pool to parallelize the PSF fitting across wavelengths. If None, do not parallelize.
    init_centroid : (float, float)
        Initial guess for the centroid of the PSF (x_init,y_init)
    fit_centroid : bool
        Whether to fit the centroid of the PSF. If False, the centroid is fixed at init_centroid.
    fit_angle : bool
        Whether to fit the rotation angle of the PSF. If False, the angle is fixed at rotate_psf.
    ann_width : float
        Width of the annuli in arcsec.
    padding : float
        Padding in arcsec to make the fitting region slightly bigger than the sectors.
    sector_area : float
        Target sector area to define the azimuthal divisions of each annulus. The smaller the sector_area, the more sectors in an annulus.
        With this definition, there are more sectors for larger annuli.
    linear_interp : bool
        Whether to use linear interpolation or CloughTocher2DInterpolator for interpolating the PSF.
    rotate_psf : None
        Typically should not need to be modified. This function performs the fit in BREADS ifu coordinate leverage dataobj.get_ifu_coords() method.
        Rotate the model by some angle.
    flipx : None
        Typically should not need to be modified, default behavior checks whether the input is STPSF or BREADS data object.
        Flip the x axis of the model being fitted. This is needed for because the convention is different for stpsf and the BREADS ifu coordinates.
    debug_wv_range : (min_wv, max_wv)
        Range of wavelengths in which to perform the fit. This is for debugging purposes, to only fit a subset of wavelengths.
    overwrite : bool
        Whether to overwrite the output file if it already exists. Default is False, which means that if the output file already exists, the function will load the results from the file instead of recomputing.
    stis_spectrum : str
        Filename of a calspec file. e.g. 1808347_stiswfc_006.fits from https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec
        Used for flux calibration derivation.
    wv_min : float
        Don't include wavelength less than wv_min in the fit. If None, default is 10% of the bandpass mask on the edges.
    wv_max : float
        Don't include wavelength greater than wv_max in the fit. If None, default is 10% of the bandpass mask on the edges.

    Returns
    -------
    bestfit_paras :
        Array of size (N_sectors, N_wavelengths, 5) with the best fit parameters. The last dimension is as follow:
        1/ Estimated flux BEFORE fitting centroid (using init_centroid)
        2/ Esitmated flux AFTER fitting centroid
    _d :
    bestfit_model :
    residuals :



        bestfit_paras = np.full((out[0].shape[0],np.size(dataobj.wv_sampling), 5), np.nan)  # flux_init, flux,ra,dec,angle

    """
    if 'MJy/sr' not in dataobj.breads_header["DATAUNIT"]:
        raise Exception("Input data should be MJy/sr, not "+dataobj.breads_header["DATAUNIT"])

    if "regwvs" not in dataobj.breads_header['COORDS']:
        raise ValueError("Data needs to be interpolated on a regular wavelength grid. Please run compute_interpdata_regwvs().")

    if out_filename is not None and not overwrite:
        if len(glob(out_filename)) >= 1:
            print("File found. Not recomputing. Instead loading "+out_filename)
            with pyfits.open(out_filename) as hdul:
                bestfit_paras = hdul["BESTPARA"].data
                bestfit_model = hdul["BESTMODL"].data
                residuals = hdul["RESIDUAL"].data
            return bestfit_paras, dataobj.data, bestfit_model, residuals

    if padding is None:
        padding = 0.0


    if use_stpsf:
        webbpsf_reload = dataobj.reload_webbpsf_model()
        if webbpsf_reload is None:
            print("Did not find a STPSF, computing it now, but it will take a while.")
            webbpsf_reload = dataobj.compute_webbpsf_model(save_utils=True, mppool= mppool)
        _, _, stpsfs, _, webbpsf_x, webbpsf_y, _, _ = webbpsf_reload
        stpsfX = np.tile(webbpsf_x[None, :, :], (stpsfs.shape[0], 1, 1))
        stpsfY = np.tile(webbpsf_y[None, :, :], (stpsfs.shape[0], 1, 1))
        flipx = True
    elif use_breadspsf is not None and not (isinstance(use_breadspsf, bool) and not use_breadspsf):
        BREADS_DATA_ENV = os.getenv('BREADS_DATA')
        if isinstance(use_breadspsf, bool) and use_breadspsf:
            grating = dataobj.priheader['GRATING'].strip()
            detector = dataobj.priheader['DETECTOR'].strip().lower()
            if os.path.exists(os.path.join(BREADS_DATA_ENV, "BreadsPSF", f"HD163466_J1757132_{grating}_{detector}.fits")):
                use_breadspsf_str = f"HD163466_J1757132_{grating}_{detector}.fits"
            else:
                use_breadspsf_str = f"J1757132_{grating}_{detector}.fits"
        elif isinstance(use_breadspsf, str):
            use_breadspsf_str = use_breadspsf
        breadsPSF_path = os.path.join(BREADS_DATA_ENV, "BreadsPSF",use_breadspsf_str)
        hdulist = pyfits.open(breadsPSF_path)
        stpsfs = hdulist['EPSFS'].data
        psf_X = hdulist['X'].data
        psf_Y = hdulist['Y'].data
        _wv_sampling = hdulist['WAVE'].data
        if not hasattr(dataobj, "wv_sampling"):
            if not np.allclose(dataobj.wv_sampling, _wv_sampling):
                raise Exception("BreadsPSF wavelength sampling is different from the one in the data object.")
        stpsfX = np.tile(psf_X[None, :, :], (stpsfs.shape[0], 1, 1))
        stpsfY = np.tile(psf_Y[None, :, :], (stpsfs.shape[0], 1, 1))
        hdulist.close()
        flipx = False

    if rotate_psf is None:
        rotate_psf = 0.0

    if init_centroid is None:
        init_centroid = np.array([0, 0])
        init_paras = init_centroid
    else:
        # convert init_centroid to ifu coordinates if coordinates are sky originally. Fitting is done in ifu coordinates.
        if "sky" in dataobj.breads_header['COORDS']:
            _out = dataobj.get_ifu_coords(ras=init_centroid[0], decs=init_centroid[1])
            init_paras = (float(_out[0]),float(_out[1]))
        elif "ifu" in dataobj.breads_header['COORDS']:
            init_paras = np.array(init_centroid)

    # only process frames with wavelength index between debug_init and debug_end
    if debug_wv_range is None:
        # debug_wv_range = (dataobj.wv_sampling[0],dataobj.wv_sampling[-1])
        debug_init = 0
        debug_end = np.size(dataobj.wv_sampling)
    else:
        debug_init = np.searchsorted(dataobj.wv_sampling, debug_wv_range[0], side='right')
        debug_end = np.searchsorted(dataobj.wv_sampling, debug_wv_range[1], side='left')
        print("Debugging mode. Only fitting wavelengths between:", debug_wv_range)


    if ref_dataobj is not None:
        if not np.allclose(dataobj.wv_sampling, ref_dataobj.wv_sampling):
            raise Exception("wv_sampling is not identical between dataobj and ref_dataobj. Please interpolate the data on a common wavelength grid before running fitpsf.")

    _x,_y = dataobj.get_ifu_coords()
    _d = dataobj.data
    _e = dataobj.noise
    _bp = dataobj.bad_pixels
    if ref_dataobj is not None:
        _psfX, _psfY = ref_dataobj.get_ifu_coords()
        _psfs = ref_dataobj.data
    else:
        _psfs, _psfX, _psfY = stpsfs, stpsfX, stpsfY

    # Define output model and residual arrays
    bestfit_model = np.full(dataobj.data.shape, np.nan)
    residuals = np.full(dataobj.data.shape, np.nan)

    bestfit_paras_defined = False
    if mppool is None:
        print(f"\tPerforming serial PSF fit at {debug_end - debug_init} wavelengths.")

        for wv_id, wv in tqdm(enumerate(dataobj.wv_sampling), total=len(dataobj.wv_sampling), ncols=100):
            if not (debug_init <= wv_id < debug_end):
                continue
            paras = linear_interp, _psfs[wv_id], _psfX[wv_id], _psfY[wv_id], rotate_psf,flipx, \
                _x[:, wv_id], _y[:, wv_id], _d[:, wv_id], _e[:,wv_id], _bp[:, wv_id], \
                IWA, OWA, fit_centroid, fit_angle, init_paras, ann_width, padding, sector_area
            out = _fit_wpsf_task(paras)
            if not bestfit_paras_defined:
                bestfit_paras = np.full((out[0].shape[0],np.size(dataobj.wv_sampling), 5), np.nan)  # flux_init, flux,ra,dec,angle
                bestfit_paras_defined = True
            bestfit_paras[:,wv_id, :] = out[0]
            bestfit_model[:, wv_id] = out[1]
            residuals[:, wv_id] = _d[:, wv_id] - out[1]

    else:
        print(f"\tPerforming parallelized PSF fit at {debug_end - debug_init} wavelengths.")

        output_lists = [o for o in tqdm(mppool.imap(_fit_wpsf_task,
                                  zip(itertools.repeat(linear_interp),
                                      _psfs[debug_init:debug_end],
                                      _psfX[debug_init:debug_end],
                                      _psfY[debug_init:debug_end],
                                      itertools.repeat(rotate_psf),
                                      itertools.repeat(flipx),
                                      _x.T[debug_init:debug_end],
                                      _y.T[debug_init:debug_end],
                                      _d.T[debug_init:debug_end],
                                      _e.T[debug_init:debug_end],
                                      _bp.T[debug_init:debug_end],
                                      itertools.repeat(IWA),
                                      itertools.repeat(OWA),
                                      itertools.repeat(fit_centroid),
                                      itertools.repeat(fit_angle),
                                      itertools.repeat(init_paras),
                                      itertools.repeat(ann_width),
                                      itertools.repeat(padding),
                                      itertools.repeat(sector_area))),
                                        total=debug_end-debug_init, ncols=100)]

        for out_id, out in tqdm(enumerate(output_lists), total=len(output_lists), ncols=100):
            if not bestfit_paras_defined:
                bestfit_paras = np.full((out[0].shape[0],np.size(dataobj.wv_sampling), 5), np.nan)  # flux_init, flux,ra,dec,angle
                bestfit_paras_defined = True
            bestfit_paras[:,debug_init+out_id, :] = out[0]
            bestfit_model[:, debug_init+out_id] = out[1]
            residuals[:, debug_init+out_id] = _d[:, debug_init+out_id] - out[1]

    if use_stpsf:
        # Get the median pixel area for the point source
        rescale_flux = np.nansum(dataobj.area2d*bestfit_model)/np.nansum(bestfit_model)
        bestfit_paras[:,:,0:2] *= rescale_flux # convert from MJy/sr to MJy

    # convert best fit coords from ifu coordinates to the original ones
    for k in range(bestfit_paras.shape[0]):
        if "sky" in dataobj.breads_header['COORDS']:
            bestfit_paras[:,:,2], bestfit_paras[:,:,3] = dataobj.get_sky_coords(ifux=bestfit_paras[:,:,2], ifuy=bestfit_paras[:,:,3])
        elif "ifu" in dataobj.breads_header['COORDS']:
            pass # the fitting was done in ifu coordinates, so nothing to do.

    if out_filename is not None:
        bestfit_paras_header = {"FLUXUNIT": "MJy", "COORDS": dataobj.breads_header['COORDS'],
                                "COORUNIT": dataobj.breads_header['COORUNIT']}
        wpsfsfit_header = {"INIT_ANG": rotate_psf, "INIT_X": init_paras[0], "INIT_Y": init_paras[1]}
        _breads_header = dataobj.breads_header
        _breads_header.update(wpsfsfit_header)
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(header=dataobj.priheader))
        hdulist.append(pyfits.ImageHDU(data=bestfit_paras, name = "BESTPARA",header=pyfits.Header(cards=bestfit_paras_header)))
        hdulist.append(pyfits.ImageHDU(data=bestfit_model, name = "BESTMODL"))
        hdulist.append(pyfits.ImageHDU(data=residuals, name = "RESIDUAL"))
        hdulist.append(pyfits.ImageHDU(header=_breads_header,name="BREADS"))
        hdulist.writeto(out_filename, overwrite=True)
        hdulist.close()

        poly_centroid_filename = out_filename.replace(".fits", "_poly_centroid_IWA{0:.2f}_OWA{1:.2f}.txt".format(IWA,OWA))
        poly_fluxcal_filename = out_filename.replace(".fits", "_poly_fluxcal_IWA{0:.2f}_OWA{1:.2f}.txt".format(IWA,OWA))
        plot_filename = out_filename.replace(".fits", "_results.png")
        analyze_fitpsf_results(dataobj,bestfit_paras,stis_spectrum=stis_spectrum,poly_deg_coords=poly_deg_coords,poly_deg_flux=poly_deg_flux,
                               poly_centroid_filename=poly_centroid_filename,poly_fluxcal_filename=poly_fluxcal_filename,plot_filename=plot_filename,
                           wv_min = wv_min,wv_max = wv_max)

        plot_filename = out_filename.replace(".fits", "2d_plot.png")
        if debug_wv_range is not None:
            wv0 = (debug_wv_range[0] + debug_wv_range[1]) / 2
        else:
            wv0 = (dataobj.wv_sampling[0] + dataobj.wv_sampling[-1]) / 2
        x_vec = np.arange(-OWA+init_centroid[0], OWA+init_centroid[0], 0.02)
        y_vec = np.arange(-OWA+init_centroid[1], OWA+init_centroid[1], 0.02)
        plot_fitpsf_2d_results(dataobj, bestfit_model, residuals, wv0=wv0,x_vec=x_vec,y_vec=y_vec,
                               overlay_pointcloud=False,plot_filename=plot_filename)


    return bestfit_paras, _d, bestfit_model, residuals


def _fit_wpsf_task(paras):
    """ Worker function for fitting a PSF model, for use in parallelized computations

    Parameters
    ----------
    paras : tuple containing many things
        paras = linear_interp, _psfs[wv_id], _psfX[wv_id], _psfY[wv_id], rotate_psf,flipx, \
            _x[:, wv_id], _y[:, wv_id], _d[:, wv_id], _e[:,wv_id], _bp[:, wv_id], \
            IWA, OWA, fit_centroid, fit_angle, init_paras, ann_width, padding, sector_area

    Returns
    -------

    """

    if len(paras) == 16:
        linear_interp, wepsf, wifuX, wifuY, east2V2_deg,flipx, _X, _Y, _Z, _Zerr, _Zbad, IWA, OWA, fit_cen, fit_angle, init_paras = paras
        ann_width, padding, sector_area = None, 0.0, None
    else:
        linear_interp, wepsf, wifuX, wifuY, east2V2_deg,flipx, _X, _Y, _Z, _Zerr, _Zbad, IWA, OWA, fit_cen, fit_angle, init_paras, ann_width, padding, sector_area = paras
    _R = np.sqrt((_X - init_paras[0]) ** 2 + (_Y - init_paras[1]) ** 2)
    _PA = np.arctan2(_X- init_paras[0], _Y- init_paras[1]) % (2 * np.pi)

    iterator_sectors = []
    if ann_width is None:
        rad_bounds = [(IWA, OWA)]
    else:
        rad_bounds = [(rmin, rmin + ann_width) for rmin in np.arange(IWA, OWA, ann_width)]
    for [r_min, r_max] in rad_bounds:
        # equivalent to using floor but casting as well
        if sector_area is None:
            curr_sep_N_subsections = 1
        else:
            curr_sep_N_subsections = np.max([int(np.pi * (r_max ** 2 - r_min ** 2) / sector_area), 1])
        # divide annuli into subsections : change method to defined the section. Now identical to parallelized
        dphi = 2 * np.pi / curr_sep_N_subsections
        phi_bounds_list = [[dphi * phi_i, dphi * (phi_i + 1)] for phi_i in
                           range(curr_sep_N_subsections)]
        phi_bounds_list[-1][1] = 2 * np.pi
        iterator_sectors.extend([((r_min, r_max), phi_bound) for phi_bound in phi_bounds_list])
    tot_sectors = len(iterator_sectors)

    out_paras = np.full((tot_sectors, 5), np.nan)
    out_model = np.full(_Z.shape, np.nan)
    for sector_id, sector in enumerate(iterator_sectors):
        rmin, rmax = sector[0]
        pamin, pamax = sector[1]
        if pamin < pamax:
            deltaphi = pamax - pamin + 2 * padding / np.mean([rmin, rmax])
        else:
            deltaphi = (2 * np.pi - (pamin - pamax)) + 2 * padding / np.mean([rmin, rmax])

        # If the length or the arc is higher than 2*pi, simply pick the entire circle.
        if deltaphi >= 2 * np.pi:
            pamin_pad = 0
            pamax_pad = 2 * np.pi
        else:
            pamin_pad = (pamin - padding / np.mean([rmin, rmax])) % (2.0 * np.pi)
            pamax_pad = (pamax + padding / np.mean([rmin, rmax])) % (2.0 * np.pi)

        rmin_pad = np.max([rmin - padding, 0.0])
        rmax_pad = rmax + padding
        if pamin_pad < pamax_pad:
            fit_sector = (rmin_pad <= _R) & (_R < rmax_pad) & (pamin_pad <= _PA) & (_PA < pamax_pad) & np.isfinite(_Zbad)
        else:
            fit_sector = (rmin_pad <= _R) & (_R < rmax_pad) & ((pamin_pad <= _PA) | (_PA < pamax_pad)) & np.isfinite(_Zbad)
        if pamin < pamax:
            sc_sector = (rmin <= _R) & (_R < rmax) & (pamin <= _PA) & (_PA < pamax) #& np.isfinite(_Zbad)
        else:
            sc_sector = (rmin <= _R) & (_R < rmax) & ((pamin <= _PA) | (_PA < pamax))# & np.isfinite(_Zbad)

        where_fit = np.where(fit_sector)
        if np.size(where_fit[0])<1:
            continue
        X, Y, Z, Zerr, Zbad = _X[where_fit], _Y[where_fit], _Z[where_fit], _Zerr[where_fit], _Zbad[where_fit]
        where_sc = np.where(sc_sector)
        if np.size(where_sc[0])<1:
            continue
        Xsc, Ysc = _X[where_sc], _Y[where_sc]

        where_wepsf_finite = np.where(np.isfinite(wepsf)*np.isfinite(wifuX)*np.isfinite(wifuY))
        if np.size(where_wepsf_finite[0])<3:
            continue
        wX, wY, wZ = wifuX[where_wepsf_finite], wifuY[where_wepsf_finite], wepsf[where_wepsf_finite]
        wX, wY = rotate_coordinates(wX, wY, -east2V2_deg, flipx=flipx)

        if linear_interp:
            webbpsf_interp = LinearNDInterpolator((wX, wY), wZ, fill_value=0.0)
        else:
            webbpsf_interp = CloughTocher2DInterpolator((wX, wY), wZ, fill_value=0.0)

        if fit_angle:
            p0 = np.array([0.0, 0.0, 0.0])
            simplex_init_steps = np.array([0.05, 0.05, 1 / 1000])
        else:
            p0 = np.array([0.0, 0.0])
            simplex_init_steps = np.array([0.05, 0.05])
        if init_paras is not None:
            p0 = np.array(init_paras)
        m0 = webbpsf_interp(X - p0[0], Y - p0[1])
        a0 = np.nansum(Z * m0 / Zerr ** 2) / np.nansum(m0 ** 2 / Zerr ** 2)
        initial_simplex = np.concatenate([p0[None, :], p0[None, :] + np.diag(simplex_init_steps)], axis=0)

        chi20 = _fitpsf_costfunc(p0, X, Y, Z, Zerr, webbpsf_interp)
        # Define the initial parameter values for the fit
        # Fit the data to the function
        try:
            if fit_cen:
                out = minimize(_fitpsf_costfunc, p0, args=(X, Y, Z, Zerr, webbpsf_interp), method="Nelder-Mead", bounds=None,
                               options={"xatol": np.inf, "fatol": chi20 * 1e-12, "maxiter": 5e3,
                                        "initial_simplex": initial_simplex, "disp": False})
                if fit_angle:
                    xc, yc, th = out.x
                    wX, wY = rotate_coordinates(wX, wY, th, flipx=False)
                    webbpsf_interp = CloughTocher2DInterpolator((wX, wY), wZ, fill_value=0.0)
                else:
                    xc, yc = out.x
                    th = 0.0
            else:
                xc, yc, th = p0[0], p0[1], 0
            m0 = webbpsf_interp(X - xc, Y - yc)
            a = np.nansum(Z * m0 / Zerr ** 2) / np.nansum(m0 ** 2 / Zerr ** 2)
        except:
            a, xc, yc, th = np.nan, np.nan, np.nan, np.nan

        out_paras[sector_id, :] = np.array([a0, a, xc, yc, th])
        out_model[where_sc] = a * webbpsf_interp(Xsc - xc, Ysc - yc)
    return out_paras, out_model


# Define the function to fit
def _fitpsf_costfunc(paras, _x, _y, data, error, _webbpsf_interp):
    """ Cost function used in PSF fitting

    Parameters
    ----------
    paras : tuple of floats
        Parameters for registering and aligning the PSF to the data.
        either (Xc, Yc) with 2 elements, or (Xc, Yc, Theta) with 3 elements.
        If only 2 elements, then Theta is set to 0
        Xc and Yc are the center location relative to the _x and _y parameters.
        Theta is a rotation angle for rotating the PSF to align.
    _x : ndarray
        X coordinates
    _y : ndarray
        Y coordinates
    data : ndarray
        Observed/measured PSF data to be fit
    error : ndarray
        Uncertainty in observed data to be fit
    _webbpsf_interp : interpolator object
        PSF interpolator object, used to obtain the shifted and aligned PSF

    Returns
    -------

    """
    if len(paras) == 2:
        xc, yc = paras
        th = 0
    else:
        xc, yc, th = paras
    _x_diff, _y_diff = rotate_coordinates(_x - xc, _y - yc, -th, flipx=False)
    znew = _webbpsf_interp(_x_diff, _y_diff)
    A = np.nansum(data * znew / error ** 2) / np.nansum(znew ** 2 / error ** 2)
    res = data - A * znew
    chi2 = np.nansum((res / error) ** 2)
    return chi2

def analyze_fitpsf_results(dataobj,bestfit_paras,poly_deg_coords = 4,poly_deg_flux=1, stis_spectrum=None,
                           poly_centroid_filename=None,poly_fluxcal_filename=None,plot_filename=None,
                           wv_min = None,wv_max = None):
    """
    Analyze the results of fitpsf, including fitting polynomials to the best fit centroids as a function of wavelength, and deriving a flux calibration if a stis_spectrum is provided.

    Parameters
    ----------
    dataobj : JWSTNirspec_multiple_cals object
        The data object that was fitted in fitpsf. Used for accessing the wavelength sampling and other metadata.
    bestfit_paras : ndarray
        The best fit parameters obtained from fitpsf, of shape (N_sectors, N_wavelengths, 5).
    stis_spectrum : str
        Filename of a calspec file. e.g. 1808347_stiswfc_006.fits from https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec
        Used for flux calibration derivation.
    poly_centroid_filename : str
        If not None, save the polynomial coefficients for the centroid fits to this filename as a text file.
    poly_fluxcal_filename : str
        If not None, save the polynomial coefficients for the flux calibration fits to this filename as a text file.
    plot_filename : str
        If not None, save a plot of the best fit fluxes and centroids as a function of wavelength.
    wv_min : float
        Don't include wavelength less than wv_min in the fit. If None, default is 10% of the bandpass mask on the edges.
    wv_max : float
        Don't include wavelength greater than wv_max in the fit. If None, default is 10% of the bandpass mask on the edges.

    Returns
    -------
    poly_p_x : ndarray
        Polynomial coefficients for the best fit X centroids as a function of wavelength.
    poly_p_y : ndarray
        Polynomial coefficients for the best fit Y centroids as a function of wavelength.
    poly_p_flux : ndarray or None
        Polynomial coefficients for the flux calibration as a function of wavelength, or None if stis_spectrum is not provided.

    """
    _med_bestfit_paras = np.nanmean(bestfit_paras, axis=0)

    if wv_min is None:
        _wv_min = dataobj.wv_sampling[0] + 0.1 * (dataobj.wv_sampling[-1] - dataobj.wv_sampling[0])
    else:
        _wv_min = wv_min
    if wv_max is None:
        _wv_max = dataobj.wv_sampling[-1] - 0.1 * (dataobj.wv_sampling[-1] - dataobj.wv_sampling[0])
    else:
        _wv_max = wv_max
    wherefinite = np.where(np.isfinite(_med_bestfit_paras[:, 2]) * (dataobj.wv_sampling > _wv_min) * (dataobj.wv_sampling < _wv_max))
    poly_p_x = np.polyfit(dataobj.wv_sampling[wherefinite], _med_bestfit_paras[:, 2][wherefinite], deg=poly_deg_coords)

    wherefinite = np.where(np.isfinite(_med_bestfit_paras[:, 3]) * (dataobj.wv_sampling > _wv_min) * (dataobj.wv_sampling < _wv_max))
    poly_p_y = np.polyfit(dataobj.wv_sampling[wherefinite], _med_bestfit_paras[:, 3][wherefinite], deg=poly_deg_coords)

    # Save centroids to a text file
    if poly_centroid_filename is not None:
        np.savetxt(poly_centroid_filename, [poly_p_x, poly_p_y], delimiter=' ')

    if stis_spectrum is not None:
        stis_table = Table(pyfits.getdata(stis_spectrum, 1))
        stis_wvs = (np.array(stis_table["WAVELENGTH"]) * u.Angstrom).to(u.um).value  # angstroms -> mum
        stis_spec = np.array(stis_table["FLUX"]) * u.erg / u.s / u.cm ** 2 / u.Angstrom  # erg s-1 cm-2 A-1
        stis_spec = stis_spec.to(u.W * u.m ** -2 / u.um)
        stis_spec_Fnu = stis_spec * (stis_wvs * u.um) ** 2 / const.c  # from Flambda back to Fnu
        stis_spec_Fnu = stis_spec_Fnu.to(u.MJy).value
        calspec_func = interp1d(stis_wvs, stis_spec_Fnu )

        flux_calib = calspec_func(dataobj.wv_sampling) / _med_bestfit_paras[:, 1]
        wherefinite = np.where(np.isfinite(flux_calib) * (dataobj.wv_sampling > _wv_min) * (dataobj.wv_sampling < _wv_max))
        flux_calib_wvs, flux_calib = dataobj.wv_sampling[wherefinite],flux_calib[wherefinite]
        # Deriving the parameters of the flux calibration
        poly_p_flux =  np.polyfit(flux_calib_wvs,flux_calib ,deg=poly_deg_flux)

        if poly_fluxcal_filename is not None:
            # Save centroids to a text file
            np.savetxt(poly_fluxcal_filename, [poly_p_flux], delimiter=' ')
    else:
        poly_p_flux = None

    if plot_filename is not None:

        color_list = ["#ff9900", "#006699", "#6600ff", "#006699", "#ff9900", "#6600ff"]
        fontsize = 12

        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(8, 1, height_ratios=[1, 0.5, 0.3, 1,0.5, 0.3, 1, 0.5], width_ratios=[1])
        gs.update(left=0.1, right=0.95, bottom=0.07, top=0.95, wspace=0.0, hspace=0.0)

        ax1 = plt.subplot(gs[0, 0])
        plt.plot(dataobj.wv_sampling, _med_bestfit_paras[:, 1], linestyle="--", color=color_list[2], label="Best fit",linewidth=1)
        if stis_spectrum is not None:
            plt.plot(stis_wvs, stis_spec_Fnu , linestyle=":", color="black", label="CALSPEC", linewidth=2)
            plt.plot(dataobj.wv_sampling, _med_bestfit_paras[:, 1] * np.polyval(poly_p_flux, dataobj.wv_sampling), linestyle="-.",
                     color=color_list[1], label="Corrected based on CALSPEC", linewidth=2)
        plt.xlim([dataobj.wv_sampling[0], dataobj.wv_sampling[-1]])
        plt.ylim([np.nanmin(_med_bestfit_paras[:, 1])*0.8,np.nanmax(_med_bestfit_paras[:, 1])*1.2])
        plt.ylabel("Flux density (MJy)", fontsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right")
        plt.text(0.01, 0.95, dataobj.priheader["TARGNAME"], fontsize=fontsize, ha='left', va='top', color="black",transform=plt.gca().transAxes)
        plt.xticks([])

        ax1 = plt.subplot(gs[1, 0])
        if stis_spectrum is not None:
            res2plot = _med_bestfit_paras[:, 1]  * np.polyval(poly_p_flux, dataobj.wv_sampling) - calspec_func(dataobj.wv_sampling)
            plt.plot(dataobj.wv_sampling,res2plot,
                     linestyle="-.", color=color_list[1], label="Corrected based on CALSPEC", linewidth=1)
            meddev = median_abs_deviation(res2plot[np.where(np.isfinite(res2plot))])
            plt.ylim([-10*meddev,10*meddev])
        plt.xlim([dataobj.wv_sampling[0], dataobj.wv_sampling[-1]])
        plt.xlabel("Wavelength ($\mu$m)", fontsize=fontsize)
        plt.ylabel("Diff. (MJy)", fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right")

        if "ifu" in dataobj.breads_header['COORDS']:
            xcoord_label = 'IFU x (arcsec)'
            ycoord_label = 'IFU y (arcsec)'
        elif "sky" in dataobj.breads_header['COORDS']:
            xcoord_label = '$\Delta$RA (arcsec)'
            ycoord_label = '$\Delta$Dec (arcsec)'

        ax1 = plt.subplot(gs[3, 0])
        plt.plot(dataobj.wv_sampling, _med_bestfit_paras[:, 2],label="Best fit")
        polyval_vec = np.polyval(poly_p_x, dataobj.wv_sampling)
        plt.plot(dataobj.wv_sampling, polyval_vec,label="Polyfit")
        plt.xlim([dataobj.wv_sampling[0], dataobj.wv_sampling[-1]])
        med,mad = np.nanmedian( _med_bestfit_paras[:, 2]),median_abs_deviation( _med_bestfit_paras[np.where(np.isfinite( _med_bestfit_paras[:, 2]))[0], 2])
        plt.ylim([med-10*mad,med+10*mad])
        plt.ylabel(xcoord_label, fontsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right")
        plt.text(0.01, 0.95, dataobj.priheader["TARGNAME"]+f"; poly p: {poly_p_x}",
                 fontsize=fontsize, ha='left', va='top', color="black",transform=plt.gca().transAxes)

        ax1 = plt.subplot(gs[4, 0])
        res2plot = _med_bestfit_paras[:, 2]-polyval_vec
        plt.plot(dataobj.wv_sampling, res2plot)
        meddev = median_abs_deviation(res2plot[np.where(np.isfinite(res2plot))])
        plt.ylim([-10*meddev,10*meddev])
        plt.xlim([dataobj.wv_sampling[0], dataobj.wv_sampling[-1]])
        plt.xlabel("Wavelength ($\mu$m)", fontsize=fontsize)
        plt.ylabel("Diff.", fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        ax1 = plt.subplot(gs[6, 0])
        plt.plot(dataobj.wv_sampling, _med_bestfit_paras[:, 3],label="Best fit")
        polyval_vec = np.polyval(poly_p_y, dataobj.wv_sampling)
        plt.plot(dataobj.wv_sampling, polyval_vec,label="Polyfit")
        plt.xlim([dataobj.wv_sampling[0], dataobj.wv_sampling[-1]])
        med,mad = np.nanmedian( _med_bestfit_paras[:, 3]),median_abs_deviation( _med_bestfit_paras[np.where(np.isfinite( _med_bestfit_paras[:, 3]))[0], 3])
        plt.ylim([med-10*mad,med+10*mad])
        plt.xlabel("Wavelength ($\mu$m)", fontsize=fontsize)
        plt.ylabel(ycoord_label, fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.text(0.01, 0.95, dataobj.priheader["TARGNAME"]+f"; poly p: {poly_p_y}",
                 fontsize=fontsize, ha='left', va='top', color="black",transform=plt.gca().transAxes)

        ax1 = plt.subplot(gs[7, 0])
        res2plot = _med_bestfit_paras[:, 3]-polyval_vec
        plt.plot(dataobj.wv_sampling, res2plot)
        meddev = median_abs_deviation(res2plot[np.where(np.isfinite(res2plot))])
        plt.ylim([-10*meddev,10*meddev])
        plt.xlim([dataobj.wv_sampling[0], dataobj.wv_sampling[-1]])
        plt.xlabel("Wavelength ($\mu$m)", fontsize=fontsize)
        plt.ylabel("Diff.", fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        plt.savefig(plot_filename, dpi=200)

    return poly_p_x, poly_p_y,poly_p_flux

def plot_fitpsf_2d_results(dataobj, bestfit_model, residuals, wv0=None,
                         x_vec=None, y_vec=None,
                         plot_filename=None, overlay_pointcloud=False,
                         label_model="Model"):
    """
    Plot the 2D results (interpolated point cloud) of fitpsf for a given wavelength slice, including the data, best fit model, and residuals.

    Parameters
    -------------
    dataobj : breads.instruments.jwstnirspec_cal
        The data object that was fitted in fitpsf. Used for accessing the wavelength sampling and other metadata.
    bestfit_model : ndarray
        The best fit model obtained from fitpsf, of shape (N_rows, N_wavelengths).
    residuals : ndarray
        The residuals obtained from fitpsf, of shape (N_rows, N_wavelengths).
    wv0 : float, optional
        The wavelength at which to plot the 2D results. If None, will use the median wavelength of the data.
    x_vec : ndarray, optional
        The x coordinates for the 2D grid to plot. If None, will use a default grid centered around the median x value of the data, spanning +/- 3 arcseconds with 60 points.
    y_vec : ndarray, optional
        The y coordinates for the 2D grid to plot. If None, will use a default grid centered around the median y value of the data, spanning +/- 3 arcseconds with 60 points.
    plot_filename : str, optional
        If not None, save the plot to this filename.
    overlay_pointcloud : bool, optional
        If True, overlay the original point cloud data points on top of the interpolated images.
    label_model : str, optional
        The label to use for the best fit model in the plot.

    Returns
    --------
    Fig : matplotlib.figure.Figure
        The figure object.

    """

    data_interp = dataobj.get_2D_point_cloud_interpolator(wv0=wv0)
    model_interp = dataobj.get_2D_point_cloud_interpolator(wv0=wv0,replace_data=bestfit_model)
    res_interp = dataobj.get_2D_point_cloud_interpolator(wv0=wv0, replace_data=residuals)

    if "regwvs" not in dataobj.breads_header['COORDS']:
        raise ValueError("Data needs to be interpolated on a regular wavelength grid. "
                         "Please run compute_interpdata_regwvs().")

    if wv0 is None:
        wv0 = np.nanmedian(dataobj.wv_sampling)

    if x_vec is None:
        x_vec = np.linspace(-3, 3, 60) + np.nanmedian(dataobj.x)
    if y_vec is None:
        y_vec = np.linspace(-3, 3, 60) + np.nanmedian(dataobj.y)

    dramin, dramax = np.min(x_vec), np.max(x_vec)
    ddecmin, ddecmax = np.min(y_vec), np.max(y_vec)
    dx_halfpix = (x_vec[1] - x_vec[0]) / 2.
    dy_halfpix = (y_vec[1] - y_vec[0]) / 2.
    extent = [dramin - dx_halfpix, dramax + dx_halfpix,
              ddecmin - dy_halfpix, ddecmax + dy_halfpix]
    inp = np.meshgrid(x_vec, y_vec)

    wv0_index = np.argmin(np.abs(dataobj.wv_sampling - wv0))
    where_good = np.where(np.isfinite(
        (dataobj.bad_pixels * bestfit_model)[:, wv0_index]))
    x = dataobj.x[where_good[0], wv0_index]
    y = dataobj.y[where_good[0], wv0_index]

    unit = dataobj.breads_header['DATAUNIT']

    # -- Layout: 3 touching image panels + 1 scatter panel with a gap ----------
    fig = plt.figure(figsize=(15, 4))

    # GridSpec: 4 columns; cols 0-2 are image panels (no space between them),
    # col 3 is the scatter plot with a wider gap on the left.
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 0.3, 1,0.3, 1],
                          wspace=0, hspace=0,  # no space between image panels
                          left=0.06, right=0.97, top=0.78, bottom=0.13)

    interp_list = [data_interp, model_interp, res_interp]
    label_list = ["Data", label_model, "Residuals"]

    # Compute colour limits per panel
    data_out = data_interp(inp[0], inp[1])
    model_out = model_interp(inp[0], inp[1])
    res_out = res_interp(inp[0], inp[1])
    # If it's masked, convert back to regular array
    # if isinstance(res_out, np.ma.MaskedArray):
    #     res_out = res_out.filled(np.nan)
    all_outs = [data_out, model_out, res_out]

    # Shared vmin/vmax for Data & Model; separate symmetric scale for Residuals
    vmin_dm = 0
    vmax_dm = 10*median_abs_deviation(data_out[np.where(np.isfinite(data_out))])
    vmax_r = 10*median_abs_deviation(res_out[np.where(np.isfinite(res_out))])

    vmins = [vmin_dm, vmin_dm, -vmax_r]
    vmaxs = [vmax_dm, vmax_dm, vmax_r]
    cmaps = ["viridis", "viridis", "RdBu_r"]  # diverging cmap for residuals  "RdBu_r"

    axes = []
    for k, (plot_label, interp_obj, out, vmin, vmax, cmap) in enumerate(
            zip(label_list, interp_list, all_outs, vmins, vmaxs, cmaps)):

        ax = fig.add_subplot(gs[0, k])
        axes.append(ax)

        im = ax.imshow(out, origin='lower', extent=extent,aspect='equal', vmin=vmin, vmax=vmax, cmap=cmap)

        if overlay_pointcloud:
            ax.scatter(x, y, s=0.5, c="gray", alpha=0.6)

        plt.xlim([extent[0], extent[1]])
        ax.invert_xaxis()
        plt.ylim([extent[2], extent[3]])

        # Colorbar on top
        cbar = fig.colorbar(im, ax=ax, location='top', fraction=0.046, pad=0.02)
        cbar.set_label(f"Flux ({unit})", labelpad=4, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        # Axis labels  only leftmost y-label, all x-labels
        if "ifu" in dataobj.breads_header['COORDS']:
            ax.set_xlabel('IFU x (as)', fontsize=9)
            if k == 0:
                ax.set_ylabel('IFU y (as)', fontsize=9)
        elif "sky" in dataobj.breads_header['COORDS']:
            ax.set_xlabel(r'$\Delta$RA (as)', fontsize=9)
            if k == 0:
                ax.set_ylabel(r'$\Delta$Dec (as)', fontsize=9)

        # Hide shared y-tick labels on middle and right panels
        if k > 0:
            ax.tick_params(labelleft=False)

        # Panel label
        txt = ax.text(0.03, 0.97, plot_label,transform=ax.transAxes, fontsize=11, va='top', ha='left',color='white')
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

    # -- Scatter panel ---------------------------------------------------------
    ax4 = fig.add_subplot(gs[0, 4])

    d_vals = dataobj.data[where_good[0], wv0_index]
    e_vals = dataobj.noise[where_good[0], wv0_index]
    m_vals = bestfit_model[where_good[0], wv0_index]
    r_vals = residuals[where_good[0], wv0_index]

    # ax4.errorbar(x, d_vals,yerr=e_vals, color="tab:blue",label="Data", zorder=3,alpha=0.5,fmt="none")
    ax4.scatter(x, d_vals, color="tab:blue", s=4, label="Data", zorder=3)
    ax4.scatter(x, m_vals, color="tab:orange", s=4, label=label_model, zorder=2)
    ax4.scatter(x, r_vals, color="gray", s=4, label="Residuals", zorder=4)
    ax4.invert_xaxis()

    if "ifu" in dataobj.breads_header['COORDS']:
        ax4.set_xlabel('IFU x (as)', fontsize=9)
    else:
        ax4.set_xlabel(r'$\Delta$RA (as)', fontsize=9)
    ax4.set_ylabel(f"Flux ({unit})", fontsize=9)
    ax4.legend(fontsize=8, markerscale=2)
    ax4.tick_params(labelsize=8)


    # -- Scatter panel ---------------------------------------------------------
    ax5 = fig.add_subplot(gs[0, 6])

    ax5.scatter(x, r_vals/e_vals, color="tab:blue", s=4)
    ax5.invert_xaxis()
    if "ifu" in dataobj.breads_header['COORDS']:
        ax5.set_xlabel('IFU x (as)', fontsize=9)
    else:
        ax5.set_xlabel(r'$\Delta$RA (as)', fontsize=9)
    ax5.set_ylabel(f"Rel. Err.", fontsize=9)
    ax5.tick_params(labelsize=8)

    if plot_filename is not None:
        plt.savefig(plot_filename, dpi=200, bbox_inches='tight')

    return fig