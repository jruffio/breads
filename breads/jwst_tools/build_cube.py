import itertools
import sys
import numpy as np

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
    X, Y, Z, Zerr, Zbp, wv_sampling, east2V2_deg, psf_interp_paras, wv_id, wv, ra_vec, dec_vec, aper_radius, N_pix_min, ifu_name = inputs

    psf_interp = _interp_psf(psf_interp_paras)

    outs = []
    for ra_id, ra in enumerate(ra_vec):
        for dec_id, dec in enumerate(dec_vec):

            R = np.sqrt((X - ra) ** 2 + (Y - dec) ** 2)
            if ifu_name == 'nirspec':
                Zerr_masking = Zerr / median_abs_deviation(Zerr[np.where(np.isfinite(Zerr))])
                where_finite = np.where(np.isfinite(Zbp) * (Zerr_masking < 5e1) * np.isfinite(X) * np.isfinite(Y) * (R < aper_radius))
            elif ifu_name == 'miri':
                where_finite = np.where(np.isfinite(Zbp) * np.isfinite(X) * np.isfinite(Y) * (R < aper_radius))
            else:
                raise ValueError('ifu_name must be either nirspec or miri')

            if np.size(where_finite[0]) < N_pix_min:
                outs.append([ra_id, dec_id, np.nan, np.nan]) #changed from continue
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
                outs.append([ra_id, dec_id, mfflux, mffluxerr * noise_factor])
    return outs

def _interp_psf(paras):
    """ Interpolate PSF

    Parameters
    ----------
    paras : tuple
        Contains the following:
        linear_interp, wepsf, wifuX, wifuY, wv_id, east2V2_deg


    Returns
    -------
    webbpsf_interp : Interpolator object
        a scipy.interpolate Interpolator object for interpolating a PSF onto
        specified coordinates.

    """
    linear_interp, wepsf, wifuX, wifuY, wv_id, east2V2_deg = paras
    wX, wY, wZ = wifuX.ravel(), wifuY.ravel(), wepsf.flatten()
    wX, wY = rotate_coordinates(wX, wY, -east2V2_deg, flipx=True)

    wherepsffinite = np.where(np.isfinite(wZ))
    wX, wY, wZ = wX[wherepsffinite], wY[wherepsffinite], wZ[wherepsffinite]
    if linear_interp:
        webbpsf_interp = LinearNDInterpolator((wX, wY), wZ, fill_value=0.0)
    else:
        webbpsf_interp = CloughTocher2DInterpolator((wX, wY), wZ, fill_value=0.0)

    return webbpsf_interp

def build_cube(combdataobj, psfs, psfX, psfY, ra_vec, dec_vec, out_filename=None,
                    linear_interp=True, mppool=None, aper_radius=0.5,
                    debug_init=None, debug_end=None, N_pix_min=None):
    """ Build a datacube, based on the forward modeling processed results

    Parameters
    ----------
    combdataobj
    psfs
    psfX
    psfY
    ra_vec
    dec_vec
    out_filename
    linear_interp : bool
        Use linear interpolation (TODO document what is being interpolated ?)
    mppool : multiprocessing.Pool or None
        if a multiprocessing Pool is supplied, the calculation will use that pool to run in parallel.
        Otherwise it will run in serial on a single process.
    aper_radius : float
        Aperture radius
    debug_init : int or None
        Minimum wavelength image to limit the calculation. Optional, for debugging.
    debug_end : int or None
        Maximum wavelength image to limit the calculation. Optional, for debugging.
    N_pix_min

    Returns
    -------
    flux_cube, fluxerr_cube, ra_grid, dec_grid

    """
    if "regwvs" not in combdataobj.coords:
        raise Exception("This data object needs to be interpolated on regular wavelength grid. See dataobj.compute_interpdata_regwvs")

    if mppool is not None:
        print('Setting parallel_flag = True')
        parallel_flag = True
    else:
        print('Setting parallel_flag = False')
        parallel_flag = False

    ifu_name = combdataobj.ifu_name

    wv_sampling = combdataobj.wv_sampling
    east2V2_deg = combdataobj.east2V2_deg
    all_interp_ra = combdataobj.dra_as_array
    all_interp_dec = combdataobj.ddec_as_array
    all_interp_flux = combdataobj.data
    all_interp_err = combdataobj.noise
    all_interp_badpix = combdataobj.bad_pixels

    if ifu_name == 'miri':
        all_interp_ra = all_interp_ra.transpose()
        all_interp_dec = all_interp_dec.transpose()
        all_interp_flux = all_interp_flux.transpose()
        all_interp_err = all_interp_err.transpose()
        all_interp_badpix = all_interp_badpix.transpose()

    if hasattr(combdataobj, "filelist"):
        N_dithers = len(combdataobj.filelist)
    else:
        N_dithers = 1

    ra_grid, dec_grid = np.meshgrid(ra_vec, dec_vec)

    flux_cube = np.full((np.size(wv_sampling), ra_grid.shape[0], ra_grid.shape[1]), np.nan)
    fluxerr_cube = np.full((np.size(wv_sampling), ra_grid.shape[0], ra_grid.shape[1]), np.nan)

    # only process frames with wavelength index between debug_init and debug_end
    if debug_init is None:
        debug_init = 0
    if debug_end is None:
        debug_end = np.size(wv_sampling)
    print(f'Processing wavelength indices in range: {debug_init} to {debug_end}')

    if N_pix_min is None:
        N_pix_min = (np.pi * aper_radius ** 2 / 0.01 * N_dithers) / 4

    #step 1 prepare list of inputs
    inputs = []
    for wv_id, wv in enumerate(wv_sampling):
        if not (debug_init <= wv_id < debug_end):
            continue
        rprint("prepping build_cube inputs... id: {} wave: {}".format(wv_id,wv))

        psf_interp_paras = linear_interp, psfs[wv_id, :, :], psfX[wv_id, :, :], psfY[wv_id, :, :], wv_id, east2V2_deg

        X = all_interp_ra[:, wv_id]
        Y = all_interp_dec[:, wv_id]
        Z = all_interp_flux[:, wv_id]
        Zerr = all_interp_err[:, wv_id]
        Zbp = all_interp_badpix[:, wv_id]

        inputs.append([X, Y, Z, Zerr, Zbp, wv_sampling, east2V2_deg,
                       psf_interp_paras,
                       wv_id, wv, ra_vec, dec_vec, aper_radius, N_pix_min, ifu_name])

    #step 2 map _build_cube_task over input list
    if parallel_flag:
        print('starting parallel _build_cube_task...')
        # Iterate calculation in parallel, showing a progress bar of percentage completion
        outputs = list(tqdm(mppool.imap(_build_cube_task, inputs), total=len(inputs), ncols=100))
    else:
        print('starting serial _build_cube_task...')
        outputs = []
        # Iterate calculation serially, also showing a progress bar of percentage completion
        for inp in tqdm(inputs, total=len(inputs), ncols=100):
            outputs.append(_build_cube_task(inp))

    #step 3 iterate over outputs and save values
    for j, inp in enumerate(inputs):
        X, Y, Z, Zerr, Zbp, wv_sampling, east2V2_deg, psf_interp_paras, wv_id, wv, ra_vec, dec_vec, aper_radius, N_pix_min, ifu_name = inp
        rprint('cubing outputs... id: {} wave: {}'.format(wv_id,wv))
        outs = outputs[j]
        for o in outs:
            ra_id, dec_id, flux, err = o
            flux_cube[wv_id, dec_id, ra_id] = flux
            fluxerr_cube[wv_id, dec_id, ra_id] = err

    if out_filename is not None:
        if debug_init != 0 or debug_end != np.size(wv_sampling):
            out_filename = out_filename.replace(".fits","_from{0}to{1}.fits".format(debug_init,debug_end))
        print("saving",out_filename)
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=flux_cube))
        hdulist.append(pyfits.ImageHDU(data=fluxerr_cube, name='FLUXERR_CUBE'))
        hdulist.append(pyfits.ImageHDU(data=ra_grid, name='RA'))
        hdulist.append(pyfits.ImageHDU(data=dec_grid, name='DEC'))
        hdulist.append(pyfits.ImageHDU(data=wv_sampling, name='WAVE'))
        hdulist.writeto(out_filename, overwrite=True)
        hdulist.close()
    return flux_cube, fluxerr_cube, ra_grid, dec_grid


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