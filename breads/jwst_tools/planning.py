import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
import pysiaf
import stpsf
from poppy.utils import quantity_input

import os
import astropy.io.fits as fits
from scipy.signal import fftconvolve
from scipy.ndimage import map_coordinates
from breads.utils import rotate_coordinates

def visualize_nrs_fov(comp_name, comp_sep, comp_pa, v3pa, center_on = 'star',
                      show_inner_diff_spikes=True, diff_spike_len = 2,
                      psf_core_check_radius=0.15,
                      nirspec_aperture='ifu',
                      offset_star=None,
                      ax=None,
                      verbose=False):
    """ Visualize NIRSpec IFU or slit FOV for a companion

    Parameters
    ----------
    comp_name : str
        Name of a companion. Used only for plot label. Can be a list.
    comp_sep : astropy.units.Quantity
        Separation of companion, in arcseconds or equivalent unit. Can be a list.
    comp_pa : astropy.units.Quantity
        PA of companion, in degrees or equivalent unit. Can be a list.
    v3pa : float
        JWST observatory V3 PA value to use when creating the figure
    center_on : str
        Where to center the IFU FOV. May be "star", "companion", "midpoint" between them.
        Alternatively use offset_star
    offset_star : list or ndarray of floats
        Offset [∆V2, ∆V3] for where the star should be positioned.
        CAUTION THESE ARE NOT THE SAME VALUES AS WOULD BE NEEDED IN AN APT OFFSET REQUIREMENT.
        THAT OFFSET REQUIREMENT NEEDS VALUES IN THE INSTRUMENT LOCAL IDEAL COORDINATE SYSTEM.
    show_inner_diff_spikes : bool
        Also display lines to indicate the inner smaller set of diffraction spikes in JWST's complex PSF.
    diff_spike_len : float
        length to draw the spikes, in arcseconds
    nirspec_aperture: str
       e.g., "ifu" or "S200A1", etc.
    psf_core_check_radius : float
        Radius in arcseconds to use when checking for proximity to the stellar PSF core.
        The code will check and warn about potentially saturated pixels near the PSF core,
        and will flag those IFU slices in red. You can adjust this parameter based on
        expectations for how many spaxels may saturate for your target.


    Returns
    -------
    None
    """

    return _visualize_jwst_ifu_fov(comp_name, comp_sep, comp_pa, v3pa, center_on,
                          show_inner_diff_spikes, diff_spike_len,
                          psf_core_check_radius,
                          nirspec_aperture=nirspec_aperture,
                          instrument='NIRSpec',
                          offset_star=offset_star, ax=ax,
                          verbose=verbose)

def visualize_miri_mrs_fov(comp_name, comp_sep, comp_pa, v3pa, center_on = 'star',
                           show_inner_diff_spikes=True, diff_spike_len = 4,
                           psf_core_check_radius=None,
                           mrs_band='2A',
                           offset_star=None, ax=None,
                           verbose=False):
    """ Visualize MIRI MRS FOV for a companion

    Parameters
    ----------
    comp_name : str
        Name of a companion. Used only for plot label. Can be a list.
    comp_sep : astropy.units.Quantity
        Separation of companion, in arcseconds or equivalent unit. Can be a list.
    comp_pa : astropy.units.Quantity
        PA of companion, in degrees or equivalent unit. Can be a list.
    v3pa : float
        JWST observatory V3 PA value to use when creating the figure
    center_on : str
        Where to center the IFU FOV. May be "star", "companion", "midpoint" between them
    show_inner_diff_spikes : bool
        Also display lines to indicate the inner smaller set of diffraction spikes in JWST's complex PSF.
    diff_spike_len : float
        length to draw the spikes, in arcseconds
    mrs_band: str
       MRS channel, as a number followed by A/B/C. e.g., "2A", "3C, etc.
    psf_core_check_radius : float
        Radius in arcseconds to use when checking for proximity to the stellar PSF core.
        The code will check and warn about potentially saturated pixels near the PSF core,
        and will flag those IFU slices in red. You can adjust this parameter based on
        expectations for how many spaxels may saturate for your target.


    Returns
    -------
    None
    """
    mrs_slice_widths = {'1': 0.7, '2': 0.28, '3': 0.39, '4': 0.64}
    mrs_mid_wavelengths = {'1': 6, '2': 9, '3': 14, '4': 22}  # microns

    if psf_core_check_radius is None:
        psf_core_check_radius  =  1.2 * mrs_slice_widths[mrs_band[0]]
        print(f"Using MRS slice width is {mrs_slice_widths[mrs_band[0]]} for PSF core check")

    return _visualize_jwst_ifu_fov(comp_name, comp_sep, comp_pa, v3pa, center_on,
                          show_inner_diff_spikes, diff_spike_len,
                          psf_core_check_radius,
                          instrument='MIRI', mrs_band=mrs_band,
                          offset_star=offset_star, ax=ax,
                          verbose=verbose)


def _visualize_jwst_ifu_fov(comp_name, comp_sep, comp_pa, v3pa, center_on = 'star',
                          show_inner_diff_spikes=True, diff_spike_len = 2,
                          psf_core_check_radius=0.15,
                          instrument='NIRSpec',
                          nirspec_aperture='ifu',
                          mrs_band='2A',
                          offset_star = None, ax=None,
                          verbose=False):
    """ shared function to display the NIRSpec or MRS FOV relative to a companion

    See doc strings for visualize_nrs_fov or visualize_miri_mrs_fov

    This function is not intended to be called directly; use one of those instead.
    """
    # Disclaimer: Written in a bit of a rush; code could be cleaned up some still...
    if not isinstance(comp_name, list):
        comp_name_list = [comp_name]
    else:
        comp_name_list = comp_name
    if not isinstance(comp_sep, list):
        comp_sep_list = [comp_sep]
    else:
        comp_sep_list = comp_sep
    if not isinstance(comp_pa, list):
        comp_pa_list = [comp_pa]
    else:
        comp_pa_list = comp_pa

    comp_rel_pa = v3pa - comp_pa_list[0].to_value(u.deg)
    comp_rel_pa_rad = np.deg2rad(comp_rel_pa)
    comp_sep_as = comp_sep_list[0].to_value(u.arcsec)

    # Setup figure using siaf
    if ax is None:
        plt.figure()
        ax = plt.gca()

    slice_apernames = []
    if instrument.lower()=='nirspec':
        inst_siaf = pysiaf.Siaf('NIRSpec')
        if nirspec_aperture == 'ifu':
            for apname in inst_siaf.apernames:
                if 'IFU_SLICE' in apname:
                    inst_siaf.apertures[apname].plot(frame='tel', color='gray', alpha=0.5, ax=ax)
                    slice_apernames.append(apname)
            ref_aperture = inst_siaf.apertures['NRS_FULL_IFU']
        else:
            ref_aperture = inst_siaf.apertures['NRS_'+nirspec_aperture+'_SLIT']
            ref_aperture.plot(frame='tel', color='gray', alpha=0.5, ax=ax)
        aper_display_name = nirspec_aperture
    else:
        inst_siaf = pysiaf.Siaf('MIRI')
        for apname in inst_siaf.apernames:
            # print(apname, f'MIRIFU_{mrs_band}' in apname)
            if f'MIRIFU_{mrs_band}' in apname:
                inst_siaf.apertures[apname].plot(frame='tel', color='gray', alpha=0.5, ax=ax)
                slice_apernames.append(apname)
        ref_aperture = inst_siaf.apertures[f'MIRIFU_CHANNEL{mrs_band}']
        aper_display_name = f'MRS channel {mrs_band}'

    v2ref = ref_aperture.V2Ref
    v3ref = ref_aperture.V3Ref

    # How should the FOV be centered?
    if offset_star == None:
        if center_on.lower() == 'star':
            offset_star = [0,0]
        elif center_on.lower().startswith('comp'):
            offset_star = [ np.sin(comp_rel_pa_rad) * comp_sep_as,
                           -np.cos(comp_rel_pa_rad) * comp_sep_as]
        elif center_on.lower().startswith('midpoint'):
            offset_star = [ np.sin(comp_rel_pa_rad) * comp_sep_as * 0.5,
                           -np.cos(comp_rel_pa_rad) * comp_sep_as * 0.5]
        else:
            raise RuntimeError('unknown value for center_on parameter')
    v2star = v2ref + offset_star[0]
    v3star = v3ref + offset_star[1]

    ax.plot(v2star, v3star, color='black', marker='o')
    if verbose:
        dv2 = v2star-v2ref
        dv3 = v3star-v3ref
        print(f"Offset star by {v2star-v2ref:.3f}, {v3star-v3ref:.3f} arcsec in (V2, V3) frame.")
        print(f"Aperture {ref_aperture.AperName} has V3IdlYAngle =  {ref_aperture.V3IdlYAngle:.3f}")

        if instrument.lower() == 'nirspec':
            instrument_aper_pa = np.mod(v3pa + ref_aperture.V3IdlYAngle - 90, 360)  # extra 90 rot for NIRSpec IFU
            print(f"For V3PA = {v3pa:.1f}, NIRSpec {ref_aperture.AperName} aperture PA (including pipeline IFUAlign rotation) is {instrument_aper_pa:.1f}")
        else:
            instrument_aper_pa= np.mod(v3pa + ref_aperture.V3IdlYAngle, 360)  # no extra 90 rot for MIRI
            print(f"For V3PA = {v3pa:.1f}, MIRI {ref_aperture.AperName} aperture PA (including pipeline IFUAlign rotation) is {instrument_aper_pa:.1f}")

        comp_aper_pa =  np.mod(instrument_aper_pa - comp_pa.to_value(u.deg), 360)
        print(f"   Therefore companion PA ({comp_pa:.2f}) is oriented to local aperture angle {comp_aper_pa:.2f} from the star")
        star_aper_pa = np.mod(comp_aper_pa + 180, 360)
        print(f"   Therefore star is oriented to local aperture angle {star_aper_pa:.2f} from the companion")
        print(f"    (for stpsf sim use offset {comp_sep_as * np.sin(np.deg2rad(star_aper_pa)):.3f}, {comp_sep_as * np.cos(np.deg2rad(star_aper_pa)):.3f} arcsec) ")

    for angle in range(6):
        ang_rad = np.deg2rad(angle * 60)
        # Big outer diffraction spikes, from the individual hexagons
        outer_spikelen = diff_spike_len
        inner_spikelen = 0.5
        ax.plot(v2star - np.asarray([inner_spikelen, outer_spikelen]) * np.sin(ang_rad),
                 v3star + np.asarray([inner_spikelen, outer_spikelen]) * np.cos(ang_rad),
                 color='black', lw=1, marker='none')
        # smaller inner diffraction spikes, from overall primary
        if show_inner_diff_spikes:
            ang_rad = np.deg2rad(angle * 60 + 30)
            ax.plot([v2star, v2star - np.sin(ang_rad) * inner_spikelen], [v3star, v3star + np.cos(ang_rad) * inner_spikelen],
                     color='black', lw=2, marker='none')
    # Extra horizontal spikes from the +V3 SM strut
    for angle in range(2):
        ang_rad = np.deg2rad(angle * 180 + 90)
        spikelen = 1.5
        ax.plot([v2star, v2star - np.sin(ang_rad) * outer_spikelen], [v3star, v3star + np.cos(ang_rad) * outer_spikelen],
                 color='black', lw=1, marker='none')

    arrowlen = 2.3
    ax.arrow(v2star, v3star, -np.sin(np.deg2rad(v3pa)) * arrowlen, np.cos(np.deg2rad(v3pa)) * arrowlen,
              color='red', lw=3, head_width=0.1)
    ax.text(v2star - np.sin(np.deg2rad(v3pa)) * arrowlen, v3star + np.cos(np.deg2rad(v3pa)) * arrowlen,
             '   N', color='red')
    arrowlen = 1.3
    ax.arrow(v2star, v3star, -np.sin(np.deg2rad(v3pa - 90)) * arrowlen, np.cos(np.deg2rad(v3pa - 90)) * arrowlen,
              color='red', lw=2, head_width=0.1)
    ax.text(v2star - np.sin(np.deg2rad(v3pa - 90)) * arrowlen, v3star + np.cos(np.deg2rad(v3pa - 90)) * arrowlen,
             '   E', color='red')

    for _comp_pa,_comp_sep,_comp_name in zip(comp_pa_list,comp_sep_list,comp_name_list):
        comp_rel_pa = v3pa - _comp_pa.to_value(u.deg)
        comp_rel_pa_rad = np.deg2rad(comp_rel_pa)
        ax.plot([v2star - np.sin(comp_rel_pa_rad) * _comp_sep.to_value(u.arcsec)],
                 [v3star + np.cos(comp_rel_pa_rad) * _comp_sep.to_value(u.arcsec)],
                 color='blue', lw=1, marker='s')
        ax.text(v2star - np.sin(comp_rel_pa_rad) * (_comp_sep.to_value(u.arcsec) + 0.5),
                 v3star + np.cos(comp_rel_pa_rad) * (_comp_sep.to_value(u.arcsec) + 0.5),
                 _comp_name, color='blue')

    if instrument.lower() == 'nirspec':
        if nirspec_aperture == 'ifu':
            slice_V3IdlYAngle = inst_siaf.apertures['NRS_IFU_SLICE00'].V3IdlYAngle
            aplabel = 'IFUalign'
            ax.text(298.7, -499.3, f"{aplabel} X axis. Bottom of NRS detectors", rotation=slice_V3IdlYAngle - 90, fontsize=8, color='green',
                    horizontalalignment='center', verticalalignment='center')
            ax.text(301., -499.7, f"{aplabel} Y axis. ", rotation=slice_V3IdlYAngle - 180, fontsize=8, color='green',
                    horizontalalignment='center', verticalalignment='center')
            ax.text(301.2, -496.9, "Top of NRS detectors", rotation=slice_V3IdlYAngle - 90, fontsize=8, color='green',
                    horizontalalignment='center', verticalalignment='center')


        else:
            slice_V3IdlYAngle = ref_aperture.V3IdlYAngle
            xtel, ytel = ref_aperture.idl_to_tel(0,0)
            aplabel = nirspec_aperture
            ax.text(xtel+0.5, ytel+0.5, f"{aplabel}", rotation=slice_V3IdlYAngle-90, fontsize=8, color='green',
                    horizontalalignment='center', verticalalignment='center')

    # Let's check if the star falls within any IFU slices, and if so warn the user
    for apname in slice_apernames:
        ap = inst_siaf.apertures[apname]
        x, y = ap.corners('tel', rederive=False)
        x2, y2 = ap.closed_polygon_points('tel', rederive=False)
        vertices = np.asarray([x,y]).transpose()
        import matplotlib
        polygon = matplotlib.patches.Polygon(vertices, closed=True, facecolor='pink', edgecolor='red', alpha=0.5)
        # Note on polycon.contains_point:  from https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html
        #  "The proper use of this method depends on the transform of the patch. [...]
        #   The convention of checking against the transformed patch
        #   stems from the fact that this method is predominantly used
        #   to check if display coordinates (e.g. from mouse events) are
        #   within the patch. If you want to do the above check with
        #   data coordinates, you have to properly transform them first
        transformed_star_coord = polygon.get_data_transform().transform((v2star, v3star))

        if polygon.contains_point(transformed_star_coord, radius=psf_core_check_radius):
            if verbose:
                print(f"Caution: {apname} contains the PSF core, and may have saturated pixels and increased noise ")
            ax.add_patch(polygon)


    ax.set_title(f'{comp_name} at V3PA={v3pa} for {instrument} {aper_display_name.upper()}')


def azimuthal_slice_interp_pa(image, ra_vec, dec_vec,
                              r_arcsec, n_samples=720,
                              interp_order=1):
    """
    Interpolate image intensities along a circle of radius r_arcsec, ignoring NaNs.
    Angles follow astronomical PA convention:
        0° = North (+Dec), increasing toward East (+RA).

    image : 2D array (rows=Dec, cols=RA)
    ra_vec, dec_vec : 1D arrays of on-sky coords [arcsec] for unflipped cube
    r_arcsec : fixed projected separation (arcsec)
    interp_order : 1=bilinear, 3=bicubic
    """
    ny, nx = image.shape

    # Column coordinate vector for current image
    ra_cols =  ra_vec
    dec_rows = dec_vec

    # Define position angles (0 = North, CCW to East)
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    DEC = r_arcsec * np.cos(angles)  # cos -> Dec (North at angle=0)
    RA = r_arcsec * np.sin(angles)  # sin -> RA (East at angle=90°)

    # Convert sky coords -> pixel indices
    dra = ra_cols[1] - ra_cols[0]
    ddec = dec_rows[1] - dec_rows[0]
    cols = (RA - ra_cols[0]) / dra
    rows = (DEC - dec_rows[0]) / ddec

    in_bounds = (cols >= 0) & (cols <= nx - 1) & (rows >= 0) & (rows <= ny - 1)

    intensities = np.full(n_samples, np.nan, dtype=float)
    if np.any(in_bounds):
        coords = np.vstack([rows[in_bounds], cols[in_bounds]])

        # NaN-aware interpolation: values + weights
        img_filled = np.nan_to_num(image, nan=0.0)
        wmap = np.isfinite(image).astype(float)

        vals = map_coordinates(img_filled, coords, order=interp_order,
                               mode="constant", cval=0.0)
        wts = map_coordinates(wmap, coords, order=interp_order,
                              mode="constant", cval=0.0)

        good = wts > 0
        intensities[in_bounds] = np.where(good, vals, np.nan)

    # Convert radians to degrees [0,360)
    angles_deg = (np.degrees(angles)) % 360.0
    return angles_deg, intensities

def visualize_nrs_psf(V3PA, sep_planets, pa_planets, grating, detector, wv=None, convolve=False):
    """
    Visualize the JWST NIRSpec IFU PSF (or its autocorrelation) in the
    instrument's IFU frame, overlaid with the projected positions of companions,
    and plot the PSF/speckle intensity as a function of position angle at each
    companion's separation.

    This function can be used to determine the optimal V3PA to observe a planet to make it fall between speckles.

    Loads a precomputed empirical PSF cube (BreadsPSF) for the given grating/detector,
    selects a single wavelength slice, optionally auto-convolves the PSF with itself
    (e.g. to account for the planet PSF), converts each companion's
    sky-plane separation/PA into IFU x/y coordinates and IFU position angle given the
    telescope V3PA, and produces two figures:
      1. A 2D image of the (log-scaled, peak-normalized) PSF in IFU x/y, with each
         companion's position marked.
      2. A companion-by-companion azimuthal slice of the PSF intensity at each
         companion's projected separation, plotted vs. IFU PA (top panel) and vs.
         V3PA (bottom panel), with the companion's own PA/V3PA marked.

    Parameters
    ----------
    V3PA : float
        Telescope V3 position angle (degrees), used to convert between sky-frame
        (RA/Dec, North/East) and IFU-frame coordinates.
    sep_planets : dict
        Mapping of companion label (e.g. "b", "c", ...) to angular separation from
        the star, in milliarcseconds (mas).
    pa_planets : dict
        Mapping of companion label to position angle (degrees), measured North to
        East, matching the keys of `sep_planets`.
    grating : str
        NIRSpec grating name (e.g. "G395H"), used to locate the corresponding
        BreadsPSF FITS file.
    detector : str
        NIRSpec detector name (e.g. "nrs1"), used to locate the corresponding
        BreadsPSF FITS file.
    wv : float, optional
        Wavelength (in the same units as the PSF cube's WAVE extension) at which to
        select the PSF slice. If None (default), the middle wavelength slice of the
        cube is used. Raises an exception if `wv` falls outside the sampled range.
    convolve : bool, optional
        If True, autoconvolve the selected PSF slice with itself (NaN-resistant,
        using a validity-mask normalization) before plotting.
        Default is False, which plots the raw PSF slice.

    Returns
    -------
    None
        This function does not return a value; it creates matplotlib figures
        (call `plt.show()` separately to display them) and prints some diagnostic
        angle values to stdout.

    Notes
    -----
    - Requires the `BREADS_DATA` environment variable to point to the directory
      containing the `BreadsPSF` calibration files.
    - The PSF FITS file is expected to be either
      `HD163466_J1757132_{grating}_{detector}.fits` or, if that is not found,
      `J1757132_{grating}_{detector}.fits`.
    - IFU coordinates are computed via a fixed V3-to-V2 offset angle
      (`V3I_YANG = 138.97164917`) combined with `V3PA`.
    """
    BREADS_DATA_ENV = os.getenv('BREADS_DATA')
    if os.path.exists(os.path.join(BREADS_DATA_ENV, "BreadsPSF", f"HD163466_J1757132_{grating}_{detector}.fits")):
        use_breadspsf_str = f"HD163466_J1757132_{grating}_{detector}.fits"
    else:
        use_breadspsf_str = f"J1757132_{grating}_{detector}.fits"
    breadsPSF_path = os.path.join(BREADS_DATA_ENV, "BreadsPSF", use_breadspsf_str)
    hdulist = fits.open(breadsPSF_path)
    psfs = hdulist['EPSFS'].data
    psf_X = hdulist['X'].data
    psf_Y = hdulist['Y'].data
    wv_sampling = hdulist['WAVE'].data
    if wv is None:
        wv_id = np.size(wv_sampling)//2
    else:
        if wv < wv_sampling[0] or wv > wv_sampling[-1]:
            raise Exception("Wavelength (wv={0}) out of range ({1},{2}) for {3} detector.".format(wv, wv_sampling[0], wv_sampling[-1], detector))
        wv_id = np.argmin(np.abs(wv_sampling - wv))
    hdulist.close()

    x_vec = psf_X[0,:]
    y_vec = psf_Y[:,0]

    psf_im = psfs[wv_id, :, :]/np.nanmax(psfs[wv_id, :, :])

    if convolve:
        # nan-resistant convolution: zero out nans, then correct for
        # the missing coverage by convolving a validity mask too
        valid = np.isfinite(psf_im).astype(float)
        psf_filled = np.nan_to_num(psf_im, nan=0.0)

        num = fftconvolve(psf_filled, psf_filled, mode="same")
        norm = fftconvolve(valid, valid, mode="same")

        # avoid divide-by-zero where there's no overlap at all
        psf_autoconv = np.full_like(num, np.nan)
        mask = norm > 0
        psf_autoconv[mask] = num[mask] * (valid.sum() ** 2) / (norm[mask] * valid.sum())

        psf_im = psf_autoconv/np.nanmax(psf_autoconv)

    psf_im_log = np.log10(psf_im)

    # result in arcseconds
    ra_planets = {p: sep_planets[p]/1000. * np.sin(np.deg2rad(pa_planets[p])) for p in sep_planets}
    dec_planets = {p: sep_planets[p]/1000. * np.cos(np.deg2rad(pa_planets[p])) for p in sep_planets}

    #convert ra/dec to ifux/ifuy
    V3I_YANG = 138.97164917
    east2V2_deg = -(V3PA + V3I_YANG)
    # print(east2V2_deg % 360)
    # exit()

    ifux_planets = {}
    ifuy_planets = {}
    ifuPA_planets = {}
    for p in ra_planets:
        ifuX, ifuY = rotate_coordinates(ra_planets[p], dec_planets[p], east2V2_deg, flipx=False)
        ifux_planets[p] = ifuX
        ifuy_planets[p] = ifuY
        ifuPA_planets[p] = (pa_planets[p]+east2V2_deg) %360 #np.rad2deg(np.arctan2(ifux_planets[p],ifuy_planets[p]))%360
        # print()
        # print(pa_planets[p],np.rad2deg(np.arctan2(ra_planets[p],dec_planets[p])))
        # print((pa_planets[p]+east2V2_deg) %360, (pa_planets[p]-east2V2_deg) %360, np.rad2deg(np.arctan2(ifux_planets[p],ifuy_planets[p]))%360)

    color_list = ["#006699","#ff9900", "#6600ff", "pink","black"]

    fig0 = plt.figure(figsize=(6,6))
    dx, dy = x_vec[1] - x_vec[0], y_vec[1] - y_vec[0]
    extent = [x_vec[0] - dx / 2., x_vec[-1] + dx / 2., y_vec[0] - dy / 2., y_vec[-1] + dy / 2.]
    fontsize = 12
    ax = plt.gca()
    im_handle = ax.imshow(psf_im_log, origin="lower", cmap="viridis",extent=extent, vmin=-8, vmax=0)
    cbar = plt.colorbar(im_handle, ax=ax)
    cbar.set_label("S/N", fontsize=fontsize)
    ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.set_xlabel(r"$\Delta$ IFU x (as)", fontsize=fontsize)
    ax.set_ylabel(r"$\Delta$ IFU y (as)", fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    plt.plot(0, 0, "*", color="grey", markersize=10)
    txt = plt.text(0.1, -0.1, 'A', fontsize=fontsize, ha='center', va='top', color="grey")
    txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
    for pl,col in zip(ifux_planets.keys(),color_list):
        txt = plt.text(ifux_planets[pl] - 0.1, ifuy_planets[pl], pl, fontsize=fontsize, ha='center', va='top',color=col)
        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
        circle = plt.Circle((ifux_planets[pl] , ifuy_planets[pl]), 0.05, facecolor='#FFFFFF00',edgecolor=col)
        plt.gca().add_patch(circle)


    fig0 = plt.figure(figsize=(12, 6))
    for pl,col in zip(ifux_planets.keys(),color_list):
        angles_deg, intens = azimuthal_slice_interp_pa(
            psf_im,
            x_vec, y_vec,
            r_arcsec=np.sqrt(ifux_planets[pl]**2+ ifuy_planets[pl]**2),
            n_samples=720,
            interp_order=1
        )

        plt.subplot(2,1, 1)
        plt.plot(angles_deg, intens, color=col, label=pl, alpha=1.0)
        plt.axvline(ifuPA_planets[pl], color=col, linestyle="-", alpha=0.5)
        plt.subplot(2,1,2)
        plt.scatter((V3PA-(angles_deg-ifuPA_planets[pl])) % 360, intens, color=col, label=pl, alpha=1.0,s=1)

    plt.subplot(2,1, 1)
    plt.legend()
    plt.xlabel("IFU PA (deg)")
    plt.ylabel("Speckle-to-star flux ratio")
    plt.yscale("log")
    plt.xlim(0, 360)
    plt.grid(True, alpha=0.3)

    plt.subplot(2,1,2)
    plt.legend()
    plt.xlabel("V3PA (deg)")
    plt.ylabel("Speckle-to-star flux ratio")
    plt.yscale("log")
    plt.xlim(0, 360)
    plt.grid(True, alpha=0.3)
    plt.axvline(V3PA% 360, color="black", linestyle="-", alpha=0.5)
    txt = plt.text(V3PA% 360, np.nanmedian(intens), "{0}deg".format(V3PA% 360),fontsize=fontsize, ha='left', va='bottom',color="black")


#############
#  Functions for using STPSF to generate and plot a mock IFU slice for a star with companion(s)

import stpsf
import copy

def arrow_angle(ax, angle, pos=(0.15, 0.15), length=0.075, color='skyblue', labelN=False, **kwargs):
    # astronomy angle convention relative to +Y
    dx = -np.sin(np.deg2rad(angle))*length
    dy = np.cos(np.deg2rad(angle))*length
    ax.arrow(pos[0], pos[1], dx, dy, color=color, transform=ax.transAxes, **kwargs)
    if labelN:
        ax.text(pos[0]+dx*1.75, pos[1]+dy*1.75, "N", color=color,
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontweight='bold')


def compass(ax, angle, **kwargs):
    arrow_angle(ax, angle, labelN=True, **kwargs)
    arrow_angle(ax, angle+90, **kwargs)



@quantity_input(wavelength=u.meter)
def calc_mock_ifu_slice(instrument='NIRSpec',
                        v3pa=0, star_offset=(0, 0), wavelength=5*u.micron,
                        companion_info = dict(),
                        vmax=1e-2, vmin=1e-6,
                        verbose=True,
                        mark_companions=True, ax= None,
                        mrs_band='1A',
                        estimate_local_contrast=False, **kwargs):

    if instrument.lower()=='nirspec':
        inst = stpsf.NIRSpec()
        inst.mode = 'IFU'

        # NIRSpec aperture PA = V3PA + NIRSpec rotation. sign checked and confirmed in APT
        inst_pa = v3pa + inst._rotation - 90   # the -90 deg comes in from the IFUalign pipeline rotation, to put slices horizontal
        inst_label = instrument + " IFU"
        fov_pixels = 30

    elif instrument.lower()=='miri':
        inst = stpsf.MIRI()
        inst.mode = 'IFU'
        inst.band = mrs_band
        inst_pa = v3pa + inst._rotation        # no extra rotation for MRS
        inst_label = 'MIRI MRS band '+mrs_band

        # MRS FOV is slightly rectangular, more so than NIRSpec, so we account for that to sim a bit more accurately
        mrs_fov_sizes = {1: [3.2, 3.7], 2: [4.0, 4.8], 3: [5.2, 6.2], 4:[6.6, 7.7]}
        fov_pixels = np.asarray(np.round(np.asarray(mrs_fov_sizes[int(inst.band[0])]) / inst.pixelscale),int)[::-1]  # flip to get Y, X order correct

    print(f"Generating mock scene for {inst_label} at {wavelength*1e6:.1f} microns, using IFU FOV = {fov_pixels} pixels ...")

    inst.options['source_offset_x'] = star_offset[0]
    inst.options['source_offset_y'] = star_offset[1]

    if verbose:
        print(f"V3PA={v3pa}, therefore instrument aperture PA={inst_pa:.1f}")
        print(f"Calculating stellar PSF at offset = {star_offset}")

    psf = inst.calc_psf(monochromatic=wavelength, fov_pixels=fov_pixels)
    scene = copy.deepcopy(psf)

    psfs = {}
    comp_offsets_in_fov = {}
    for comp_label, comp_info  in companion_info.items():
        # Determine the position in the FOV for this companion
        dx = comp_info['r'].to_value(u.arcsec) * (-np.sin(np.deg2rad(comp_info['pa'].to_value(u.degree) - inst_pa)))
        dy = comp_info['r'].to_value(u.arcsec) * (np.cos(np.deg2rad(comp_info['pa'].to_value(u.degree) - inst_pa)))
        if verbose:
            print(f"Calculating comp {comp_label} PSF at offset = {dx:.2f}, {dy:.2f} from the star")

        inst.options['source_offset_x'] = star_offset[0] + dx
        inst.options['source_offset_y'] = star_offset[1] + dy
        comp_offsets_in_fov[comp_label] = (star_offset[0] + dx, star_offset[1] + dy)

        # Calculate PSF for this companion, at that location
        psfs[comp_label] = inst.calc_psf(monochromatic=wavelength, fov_pixels=fov_pixels)


        if estimate_local_contrast:
            raise NotImplementedError("Needs to be updated")
            local_cont = psf[3].data[14:17, 14:17].sum()  # TODO FIX THIS FOR OFFSETS IN PIXELS

            peak_norm = psfs['b'][3].data.max()
            print(f"  PSF halo local contrast = {local_cont/peak_norm/9:.1e} per spaxel in central 3x3 spaxels")

        # Determine contrast for this companion
        if 'contrast_function' in comp_info:
            comp_contrast = comp_info['contrast_function'](wavelength)
        else:
            comp_contrast = comp_info['contrast']

        # Scale the PSF for the companion and add to the stellar PSF
        for ext in range(4):

            scene[ext].data += psfs[comp_label][ext].data * comp_contrast

            if verbose and ext==3:
                print(f"  Comp {comp_label} modeled with contrast ~ {comp_contrast:.2e}")
                if estimate_local_contrast:
                    print(f"  Comp peak pixel is {comp_contrast / (local_cont/peak_norm/9):.2f}x relative to local PSF halo")
    # DISPLAY
    if ax is None:
        ax = plt.gca()
    stpsf.display_psf(scene,ext=3, title=f'{wavelength.to_value(u.micron)} µm, v3pa={v3pa}',
                      ax=ax, vmax=vmax, vmin=vmin, **kwargs)
    # annotate companion positions
    for comp_label,comp_info  in companion_info.items():
        cx, cy = comp_offsets_in_fov[comp_label]
        ax.plot([cx, cx], [cy + 0.2, cy + 0.5], color='cyan')
        ax.plot([cx + 0.2, cx + 0.5], [cy, cy], color='cyan')
        ax.text(cx+0.2, cy+0.2, comp_label, color='cyan')

    ax.text(0.97, 0.03, f"{inst_label} \nAperture PA = {v3pa + inst._rotation:.1f}\n IFUAlign cube PA = {inst_pa:.1f}º", color='cyan',
            transform=ax.transAxes,
            horizontalalignment='right')

    if estimate_local_contrast:
        ax.text(0.97, 0.97, f"Comp peak pixel is $\\sim${10.**(deltamag/-2.5) / (local_cont/peak_norm/9):.1f}x\nthe local PSF halo",
                color='yellow',
                transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='top')


    scene[1].header['APER_PA'] = inst_pa
    scene[1].header['V3PA'] = v3pa

    compass(plt.gca(), -inst_pa)
    return scene

