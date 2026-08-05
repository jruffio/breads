import itertools
import sys
import numpy as np
from glob import glob
import os

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt

from astropy import constants as const
from astropy import units as u
from scipy.stats import median_abs_deviation
from scipy.ndimage import generic_filter
from tqdm import tqdm
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator

from breads.utils import rotate_coordinates


def _build_cube_task(inputs):
    """ Worker function for creating a cube slice, for one single wavelength.
    Called from build_cube(); not intended to be called directly by users.

    Parameters
    ----------
    inputs : tuple containing many parameters
        X, Y, Z, Zerr, Zbp, wv_sampling, east2V2_deg, psf_interp_paras, wv_id, wv, ra_vec, dec_vec, aper_radius, N_pix_min

    Returns
    -------
    outs : list of lists
        Complex nested bunch of stuff... TODO figure out and document

    """
    X, Y, Z, Zerr, Zbp, wv_sampling, psf_interp_paras, wv_id, wv,  ifux_grid, ifuy_grid, aper_radius, N_pix_min, ifu_name = inputs


    ny,nx = ifux_grid.shape
    mfflux_arr = np.full_like(ifux_grid,np.nan)
    mffluxerr_arr = np.full_like(ifux_grid,np.nan)

    psf_interp = _interp_psf(psf_interp_paras)

    if psf_interp is None:
        return mfflux_arr,mffluxerr_arr

    for k in range(ny):
        for l in range(nx):
            ra, dec = ifux_grid[k,l],ifuy_grid[k,l]
            R = np.sqrt((X - ra) ** 2 + (Y - dec) ** 2)
            # if ifu_name == 'nirspec':
            #     Zerr_masking = Zerr / median_abs_deviation(Zerr[np.where(np.isfinite(Zerr))])
            #     where_finite = np.where(np.isfinite(Zbp) * (Zerr_masking < 5e1) * np.isfinite(X) * np.isfinite(Y) * (R < aper_radius))
            # elif ifu_name == 'miri':
            #     where_finite = np.where(np.isfinite(Zbp) * np.isfinite(X) * np.isfinite(Y) * (R < aper_radius))
            # else:
            #     raise ValueError('ifu_name must be either nirspec or miri')

            where_finite = np.where(np.isfinite(Zbp) * np.isfinite(X) * np.isfinite(Y) * (R < aper_radius))

            if np.size(where_finite[0]) < N_pix_min:
                mfflux_arr[k,l] = np.nan
                mffluxerr_arr[k,l] = np.nan
            else:
                X_fin = X[where_finite]
                Y_fin = Y[where_finite]
                Z_fin = Z[where_finite]

                Zerr_fin = Zerr[where_finite]
                M = psf_interp(X_fin - ra, Y_fin - dec)

                deno = np.nansum(M ** 2 / Zerr_fin ** 2)
                mfflux = np.nansum(M * Z_fin / Zerr_fin ** 2) / deno
                mffluxerr = 1 / np.sqrt(deno)

                res = Z_fin - mfflux * M
                noise_factor = np.nanstd(res / Zerr_fin)
                mfflux_arr[k,l] = mfflux
                mffluxerr_arr[k,l] = mffluxerr * noise_factor

    return mfflux_arr,mffluxerr_arr

def _interp_psf(paras):
    """ Interpolate PSF

    Parameters
    ----------
    paras : tuple
        Contains the following:
        linear_interp, wepsf, wifuX, wifuY, wv_id, flipx


    Returns
    -------
    webbpsf_interp : Interpolator object
        a scipy.interpolate Interpolator object for interpolating a PSF onto
        specified coordinates.

    """
    linear_interp, wepsf, wifuX, wifuY, wv_id, flipx = paras
    wX, wY, wZ = wifuX.ravel(), wifuY.ravel(), wepsf.flatten()
    wX, wY = rotate_coordinates(wX, wY, 0, flipx=flipx)

    wherepsffinite = np.where(np.isfinite(wZ))
    if np.size(wherepsffinite[0]) == 0:
        return None

    wX, wY, wZ = wX[wherepsffinite], wY[wherepsffinite], wZ[wherepsffinite]
    if linear_interp:
        webbpsf_interp = LinearNDInterpolator((wX, wY), wZ, fill_value=0.0)
    else:
        webbpsf_interp = CloughTocher2DInterpolator((wX, wY), wZ, fill_value=0.0)

    return webbpsf_interp

def build_cube(dataobj,
               x_vec, y_vec,
               use_stpsf = False, use_breadspsf = None,
               psfs=None, psfX = None, psfY = None,flipx=False,
               out_filename=None,overwrite=False,
               linear_interp=True, mppool=None, aper_radius=0.5,
               debug_wv_range=None, N_pix_min=None):
    """ Build a datacube by fitting a PSF at every wavelength and location

    Parameters
    ----------

    Returns
    -------
    flux_cube, fluxerr_cube, ra_grid, dec_grid

    """
    if 'MJy/sr' not in dataobj.breads_header["DATAUNIT"]:
        raise Exception("Input data should be MJy/sr, not "+dataobj.breads_header["DATAUNIT"])

    if "regwvs" not in dataobj.breads_header['COORDS']:
        raise ValueError("Data needs to be interpolated on a regular wavelength grid. Please run compute_interpdata_regwvs().")

    if out_filename is not None and not overwrite:
        if len(glob(out_filename)) >= 1:
            print("File found. Not recomputing. Instead loading "+out_filename)
            with pyfits.open(out_filename) as hdul:
                flux_cube = hdul["FLUX"].data
                fluxerr_cube = hdul["FLUXERR"].data
                ra_grid = hdul["X"].data
                dec_grid = hdul["Y"].data
                wv_sampling = hdul["WAVE"].data
            return flux_cube, fluxerr_cube, ra_grid, dec_grid,wv_sampling


    if use_stpsf:
        webbpsf_reload = dataobj.reload_webbpsf_model()
        if webbpsf_reload is None:
            print("Did not find a STPSF, computing it now, but it will take a while.")
            webbpsf_reload = dataobj.compute_webbpsf_model(save_utils=True, mppool= mppool)
        _, _, psfs, _, webbpsf_x, webbpsf_y, _, _ = webbpsf_reload
        psfX = np.tile(webbpsf_x[None, :, :], (psfs.shape[0], 1, 1))
        psfY = np.tile(webbpsf_y[None, :, :], (psfs.shape[0], 1, 1))
        flipx = True
        out_units = "MJy"
    elif use_breadspsf is not None and not (isinstance(use_breadspsf, bool) and not use_breadspsf):
        BREADS_DATA_ENV = os.getenv('BREADS_DATA')
        if isinstance(use_breadspsf, bool) and use_breadspsf:
            grating = dataobj.priheader['GRATING'].strip()
            detector = dataobj.priheader['DETECTOR'].strip().lower()
            if os.path.exists(os.path.join(BREADS_DATA_ENV, "BreadsPSF",f"HD163466_J1757132_{grating}_{detector}.fits")):
                use_breadspsf_str = f"HD163466_J1757132_{grating}_{detector}.fits"
            elif os.path.exists(os.path.join(BREADS_DATA_ENV, "BreadsPSF",f"J1757132_{grating}_{detector}.fits")):
                use_breadspsf_str = f"J1757132_{grating}_{detector}.fits"
            else:
                raise Exception(f"Adequate BreadsPSFs files not found in {BREADS_DATA_ENV}. Please download these files. ")
        elif isinstance(use_breadspsf, str):
            use_breadspsf_str = use_breadspsf
        breadsPSF_path = os.path.join(BREADS_DATA_ENV, "BreadsPSF",use_breadspsf_str)
        hdulist = pyfits.open(breadsPSF_path)
        psfs = hdulist['EPSFS'].data
        psf_X = hdulist['X'].data
        psf_Y = hdulist['Y'].data
        _wv_sampling = hdulist['WAVE'].data
        if not hasattr(dataobj, "wv_sampling"):
            if not np.allclose(dataobj.wv_sampling, _wv_sampling):
                raise Exception("BreadsPSF wavelength sampling is different from the one in the data object.")
        psfX = np.tile(psf_X[None, :, :], (psfs.shape[0], 1, 1))
        psfY = np.tile(psf_Y[None, :, :], (psfs.shape[0], 1, 1))
        units_str = hdulist['EPSFS'].header["BUNIT"].strip()
        hdulist.close()
        flipx = False
        if units_str == "[MJy/sr]/[1MJy]":
            out_units = "MJy"
        else:
            out_units = "Unknown"



    # only process frames with wavelength index between debug_init and debug_end
    if debug_wv_range is None:
        # debug_wv_range = (dataobj.wv_sampling[0],dataobj.wv_sampling[-1])
        debug_init = 0
        debug_end = np.size(dataobj.wv_sampling)
    else:
        debug_init = np.searchsorted(dataobj.wv_sampling, debug_wv_range[0], side='right')
        debug_end = np.searchsorted(dataobj.wv_sampling, debug_wv_range[1], side='left')
        print("Debugging mode. Only fitting wavelengths between:", debug_wv_range)


    _x,_y = dataobj.get_ifu_coords()
    _d = dataobj.data
    _e = dataobj.noise
    _bp = dataobj.bad_pixels

    ifu_name = dataobj.ifu_name
    if ifu_name == 'miri':
        raise Exception("Not yet implemented.")
        # all_interp_ra = all_interp_ra.transpose()
        # all_interp_dec = all_interp_dec.transpose()
        # all_interp_flux = all_interp_flux.transpose()
        # all_interp_err = all_interp_err.transpose()
        # all_interp_badpix = all_interp_badpix.transpose()

    if hasattr(dataobj, "filelist"):
        N_dithers = len(dataobj.filelist)
    else:
        N_dithers = 1

    # convert init_centroid to ifu coordinates if coordinates are sky originally. Fitting is done in ifu coordinates.
    if "sky" in dataobj.breads_header['COORDS']:
        ra_grid, dec_grid = np.meshgrid(x_vec, y_vec)
        ifux_grid, ifuy_grid = dataobj.get_ifu_coords(ras=ra_grid, decs=dec_grid)
    elif "ifu" in dataobj.breads_header['COORDS']:
        ifux_grid, ifuy_grid = np.meshgrid(x_vec, y_vec)

    flux_cube = np.full((np.size(dataobj.wv_sampling), ifux_grid.shape[0], ifux_grid.shape[1]), np.nan)
    fluxerr_cube = np.full((np.size(dataobj.wv_sampling), ifux_grid.shape[0], ifux_grid.shape[1]), np.nan)

    if N_pix_min is None:
        N_pix_min = int((np.pi * aper_radius ** 2 / (0.1**2) * N_dithers) / 2.)

    #step 1 prepare list of inputs
    inputs = []
    for wv_id, wv in enumerate(dataobj.wv_sampling):
        if not (debug_init <= wv_id < debug_end):
            continue
        rprint("prepping build_cube inputs... id: {} wave: {}".format(wv_id,wv))

        psf_interp_paras = linear_interp, psfs[wv_id, :, :], psfX[wv_id, :, :], psfY[wv_id, :, :], wv_id, flipx

        inputs.append([_x[:, wv_id], _y[:, wv_id], _d[:, wv_id], _e[:, wv_id], _bp[:, wv_id],
                       dataobj.wv_sampling,
                       psf_interp_paras,
                       wv_id, wv, ifux_grid, ifuy_grid, aper_radius, N_pix_min, ifu_name])

    #step 2 map _build_cube_task over input list
    if mppool is None:
        print(f"\tPerforming serial _build_cube_task at {debug_end - debug_init} wavelengths.")

        outputs = []
        # Iterate calculation serially, also showing a progress bar of percentage completion
        for j,inp in enumerate(tqdm(inputs, total=len(inputs), ncols=100)):
            mfflux_arr,mffluxerr_arr = _build_cube_task(inp)
            flux_cube[debug_init+j, :, :] = mfflux_arr
            fluxerr_cube[debug_init+j, :, :] = mffluxerr_arr
    else:
        print('starting parallel _build_cube_task...')
        # Iterate calculation in parallel, showing a progress bar of percentage completion
        outputs = list(tqdm(mppool.imap(_build_cube_task, inputs), total=len(inputs), ncols=100))

        #step 3 iterate over outputs and save values
        for j, inp in enumerate(inputs):
            rprint('cubing outputs... id: {} wave: {}'.format(debug_init+j,dataobj.wv_sampling[debug_init+j]))
            mfflux_arr,mffluxerr_arr = outputs[j]
            flux_cube[debug_init+j, :, :] = mfflux_arr
            fluxerr_cube[debug_init+j, :, :] = mffluxerr_arr

    if out_filename is not None:
        if debug_init != 0 or debug_end != np.size(dataobj.wv_sampling):
            out_filename = out_filename.replace(".fits","_from{0}_to_{1}um_{2}_{3}id.fits".format(debug_wv_range[0],debug_wv_range[1],debug_init,debug_end))
        print("saving",out_filename)

        _hdr_flux = pyfits.Header({'BUNIT': out_units})

        if "sky" in dataobj.breads_header['COORDS']:
            _x_out = ra_grid
            _y_out = dec_grid
        elif "ifu" in dataobj.breads_header['COORDS']:
            _x_out = ifux_grid
            _y_out = ifuy_grid
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(header=dataobj.breads_header))
        hdulist.append(pyfits.ImageHDU(data=flux_cube, name='FLUX',header=_hdr_flux))
        hdulist.append(pyfits.ImageHDU(data=fluxerr_cube, name='FLUXERR',header=_hdr_flux))
        hdulist.append(pyfits.ImageHDU(data=_x_out, name='X'))
        hdulist.append(pyfits.ImageHDU(data=_y_out, name='Y'))
        hdulist.append(pyfits.ImageHDU(data=dataobj.wv_sampling, name='WAVE'))
        if hasattr(dataobj, "wv_nodes"):
            hdulist.append(pyfits.ImageHDU(data=dataobj.wv_nodes, name='wv_nodes'))
        hdulist.writeto(out_filename, overwrite=True)
        hdulist.close()
    return flux_cube, fluxerr_cube, _x_out, _y_out,dataobj.wv_sampling


def cube_matchedfilter(flux_cube, fluxerr_cube, wv_sampling, ra_grid, dec_grid, planet_f, rv=0,
                       out_filename=None, outlier_threshold=None):
    """ Apply matched filter to a datacube

    Parameters
    ----------
    flux_cube
    fluxerr_cube
    wv_sampling
    ra_grid
    dec_grid
    planet_f
    rv
    out_filename
    outlier_threshold

    Returns
    -------
    snr_map, flux_map, fluxerr_map, ra_grid, dec_grid

    """
    comp_spec = planet_f(wv_sampling * (1 - rv / const.c.to('km/s').value)) * (u.W / u.m ** 2 / u.um)
    comp_spec = comp_spec * (wv_sampling * u.um) ** 2 / const.c  # from  Flambda to Fnu
    comp_spec = comp_spec.to(u.MJy).value

    ra_vec = ra_grid[0,:]
    dec_vec = dec_grid[:,0]

    flux_map = np.full((ra_grid.shape), np.nan)
    fluxerr_map = np.full((ra_grid.shape), np.nan)

    for ra_id, ra in enumerate(ra_vec):
        for dec_id, dec in enumerate(dec_vec):

            if outlier_threshold is not None:
                snr_vec = flux_cube[:, dec_id, ra_id] / fluxerr_cube[:, dec_id, ra_id]
                snr_vec = (snr_vec - generic_filter(snr_vec, np.nanmedian, size=50)) / median_abs_deviation(snr_vec[np.where(np.isfinite(snr_vec))])
                where_outliers = np.where(snr_vec > outlier_threshold)
                flux_cube[where_outliers[0], dec_id, ra_id] = np.nan
                fluxerr_cube[where_outliers[0], dec_id, ra_id] = np.nan

            deno = np.nansum(comp_spec** 2 / fluxerr_cube[:, dec_id, ra_id] ** 2)
            bbflux = np.nansum(comp_spec * flux_cube[:, dec_id, ra_id] / fluxerr_cube[:, dec_id, ra_id] ** 2) / deno
            bbfluxerr = 1 / np.sqrt(deno)

            res = flux_cube[:, dec_id, ra_id] - bbflux*comp_spec
            noise_factor = np.nanstd(res/fluxerr_cube[:, dec_id, ra_id])

            flux_map[dec_id, ra_id] = bbflux
            fluxerr_map[dec_id, ra_id] = bbfluxerr*noise_factor

    snr_map = flux_map / fluxerr_map
    if out_filename is not None:
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=flux_map))
        hdulist.append(pyfits.ImageHDU(data=fluxerr_map, name='FLUXERR'))
        hdulist.append(pyfits.ImageHDU(data=snr_map, name='SNR'))
        hdulist.append(pyfits.ImageHDU(data=ra_grid, name='RA'))
        hdulist.append(pyfits.ImageHDU(data=dec_grid, name='DEC'))
        hdulist.writeto(out_filename, overwrite=True)
        hdulist.close()
    return snr_map, flux_map, fluxerr_map, ra_grid, dec_grid


def matchedfilter_bb(fitpsf_filename, dataobj_list, psfs, psfX, psfY, ra_vec, dec_vec, planet_f, out_filename=None,
                     linear_interp=True, mppool=None, aper_radius=0.5, rv=0):
    """ Matched filter, baed on bb (black body?)
    %todo: to be deleted?

    Parameters
    ----------
    fitpsf_filename
    dataobj_list
    psfs
    psfX
    psfY
    ra_vec
    dec_vec
    planet_f
    out_filename
    linear_interp
    mppool
    aper_radius
    rv

    Returns
    -------
    snr_map, flux_map, fluxerr_map, ra_grid, dec_grid

    """
    print("Make sure interpdata_regwvs was already done ")
    dataobj0 = dataobj_list[0]
    wv_sampling = dataobj0.wv_sampling
    east2V2_deg = dataobj0.east2V2_deg

    comp_spec = planet_f(wv_sampling * (1 - rv / const.c.to('km/s').value)) * (u.W / u.m ** 2 / u.um)
    comp_spec = comp_spec * dataobj0.aper_to_epsf_peak_f(wv_sampling)  # normalized to peak flux
    comp_spec = comp_spec * (wv_sampling * u.um) ** 2 / const.c  # from  Flambda to Fnu
    comp_spec = comp_spec.to(u.MJy).value

    ra_grid, dec_grid = np.meshgrid(ra_vec, dec_vec)

    flux_map = np.zeros_like(ra_grid)
    fluxerr_map = np.zeros_like(ra_grid)

    all_interp_ra, all_interp_dec, all_interp_wvs, all_interp_flux, all_interp_err, all_interp_badpix, all_interp_area2d = \
        dataobj0.interpdata_regwvs(wv_sampling=None, modelfit=False, out_filename=dataobj0.interpdata_regwvs_filename,
                                   load_interpdata_regwvs=True)
    if len(dataobj_list) > 1:
        for dataobj in dataobj_list[1::]:
            interp_ra, interp_dec, interp_wvs, interp_flux, interp_err, interp_badpix, interp_area2d = \
                dataobj.interpdata_regwvs(wv_sampling=None, modelfit=False,
                                          out_filename=dataobj.interpdata_regwvs_filename, load_interpdata_regwvs=True)
            all_interp_ra = np.concatenate((all_interp_ra, interp_ra), axis=0)
            all_interp_dec = np.concatenate((all_interp_dec, interp_dec), axis=0)
            all_interp_wvs = np.concatenate((all_interp_wvs, interp_wvs), axis=0)
            all_interp_flux = np.concatenate((all_interp_flux, interp_flux), axis=0)
            all_interp_err = np.concatenate((all_interp_err, interp_err), axis=0)
            all_interp_badpix = np.concatenate((all_interp_badpix, interp_badpix), axis=0)
            all_interp_area2d = np.concatenate((all_interp_area2d, interp_area2d), axis=0)
    with pyfits.open(fitpsf_filename) as hdulist:
        all_interp_psfsub = hdulist[1].data
    psf_interp_list = []
    print("create psf model")
    debug_init = 0
    debug_end = np.size(wv_sampling)
    if 0 or mppool is None:
        for wv_id, wv in enumerate(wv_sampling):
            if not (debug_init < wv_id < debug_end):
                psf_interp_list.append(0)
                continue
            paras = linear_interp, psfs[wv_id, :, :], psfX[wv_id, :, :], psfY[wv_id, :, :], wv_id, east2V2_deg
            out = _interp_psf(paras)
            psf_interp_list.append(out)
    else:
        output_lists = mppool.map(_interp_psf, zip(itertools.repeat(linear_interp), psfs[debug_init:debug_end, :, :],
                                                   psfX[debug_init:debug_end, :, :], psfY[debug_init:debug_end, :, :],
                                                   np.arange(np.size(wv_sampling))[debug_init:debug_end],
                                                   itertools.repeat(east2V2_deg)))
        for k in range(debug_init):
            psf_interp_list.append(0)
        for wv_id, (wv, out) in enumerate(zip(wv_sampling, output_lists)):
            psf_interp_list.append(out)

    print("done creating psf model")

    for ra_id, ra in enumerate(ra_vec):
        for dec_id, dec in enumerate(dec_vec):
            print(ra, dec)
            sampled_psf = np.full(all_interp_flux.shape, np.nan)
            for wv_id, wv in enumerate(wv_sampling):
                if not (debug_init < wv_id < debug_end):
                    continue
                X = all_interp_ra[:, wv_id]
                Y = all_interp_dec[:, wv_id]
                R = np.sqrt((X - ra) ** 2 + (Y - dec) ** 2)
                where_finite = np.where(
                    np.isfinite(all_interp_badpix[:, wv_id]) * np.isfinite(X) * np.isfinite(Y) * (R < aper_radius))
                X = X[where_finite]
                Y = Y[where_finite]
                sampled_psf[where_finite[0], wv_id] = psf_interp_list[wv_id](X - ra, Y - dec)

            sampled_psf = (sampled_psf * comp_spec[None, :]) * all_interp_area2d / dataobj_list[0].breads_header["WPSFAREA"]

            deno = np.nansum(sampled_psf ** 2 / all_interp_err ** 2)
            mfflux = np.nansum(sampled_psf * all_interp_psfsub / all_interp_err ** 2) / deno
            mffluxerr = 1 / np.sqrt(deno)

            res = all_interp_psfsub - mfflux*sampled_psf
            noise_factor = np.nanstd(res/all_interp_err)

            flux_map[dec_id, ra_id] = mfflux
            fluxerr_map[dec_id, ra_id] = mffluxerr * noise_factor

    snr_map = flux_map / fluxerr_map
    if out_filename is not None:
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=flux_map))
        hdulist.append(pyfits.ImageHDU(data=fluxerr_map, name='FLUXERR'))
        hdulist.append(pyfits.ImageHDU(data=snr_map, name='SNR'))
        hdulist.append(pyfits.ImageHDU(data=ra_grid, name='RA'))
        hdulist.append(pyfits.ImageHDU(data=dec_grid, name='DEC'))
        hdulist.writeto(out_filename, overwrite=True)
        hdulist.close()
    return snr_map, flux_map, fluxerr_map, ra_grid, dec_grid


def rprint(string):
    """Print a line of text, using a carriage return to overprint the current line
    (rather than printing a new line)
    """
    sys.stdout.write('\r'+str(string))
    sys.stdout.flush()