import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pysiaf

def visualize_nrs_fov(comp_name, comp_sep, comp_pa, v3pa, center_on = 'star',
                          show_inner_diff_spikes=True, diff_spike_len = 2,
                          psf_core_check_radius=0.15,
                          nirspec_aperture='ifu',
                          verbose=False):
    return visualize_jwst_ifu_fov(comp_name, comp_sep, comp_pa, v3pa, center_on,
                          show_inner_diff_spikes, diff_spike_len,
                          psf_core_check_radius,
                          nirspec_aperture=nirspec_aperture,
                          instrument='NIRSpec',
                          verbose=verbose)

def visualize_miri_mrs_fov(comp_name, comp_sep, comp_pa, v3pa, center_on = 'star',
                          show_inner_diff_spikes=True, diff_spike_len = 2,
                          psf_core_check_radius=0.15,
                           miri_channel='2A',
                          verbose=False):
    return visualize_jwst_ifu_fov(comp_name, comp_sep, comp_pa, v3pa, center_on,
                          show_inner_diff_spikes, diff_spike_len,
                          psf_core_check_radius,
                          instrument='MIRI', miri_channel=miri_channel,
                          verbose=verbose)


def visualize_jwst_ifu_fov(comp_name, comp_sep, comp_pa, v3pa, center_on = 'star',
                          show_inner_diff_spikes=True, diff_spike_len = None,
                          psf_core_check_radius=None,
                          instrument='NIRSpec',
                          nirspec_aperture='ifu',
                          miri_channel='2A',
                          verbose=False):
    """ Visualize NIRSpec IFU or slit FOV, or MIRI MRS FOV, for a companion

    Parameters
    ----------
    comp_name : str
        Name of a companion. Used only for plot label.
    comp_sep : astropy.units.Quantity
        Separation of companion, in arcseconds or equivalent unit
    comp_pa : astropy.units.Quantity
        PA of companion, in degrees or equivalent unit
    v3pa : float
        JWST observatory V3 PA value to use when creating the figure
    center_on : str
        Where to center the IFU FOV. May be "star", "companion", "midpoint" between them
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
    # Disclaimer: Written in a bit of a rush; code could be cleaned up some still...

    comp_rel_pa = v3pa - comp_pa.to_value(u.deg)
    comp_rel_pa_rad = np.deg2rad(comp_rel_pa)
    comp_sep_as = comp_sep.to_value(u.arcsec)

    # Setup figure using siaf
    plt.figure()

    if diff_spike_len is None:
        diff_spike_len = 2 if instrument.lower()=='nirspec' else 4
    if psf_core_check_radius is None:
        psf_core_check_radius  =  0.15 if instrument.lower()=='nirspec' else 0.7  # TODO: make this better per channel?

    slice_apernames = []
    if instrument.lower()=='nirspec':
        inst_siaf = pysiaf.Siaf('NIRSpec')
        if nirspec_aperture == 'ifu':
            for apname in inst_siaf.apernames:
                if 'IFU_SLICE' in apname:
                    inst_siaf.apertures[apname].plot(frame='tel', color='gray', alpha=0.5)
                    slice_apernames.append(apname)
            ref_aperture = inst_siaf.apertures['NRS_FULL_IFU']
        else:
            ref_aperture = inst_siaf.apertures['NRS_'+nirspec_aperture+'_SLIT']
            ref_aperture.plot(frame='tel', color='gray', alpha=0.5)
        aper_display_name = nirspec_aperture
    else:
        inst_siaf = pysiaf.Siaf('MIRI')
        for apname in inst_siaf.apernames:
            # print(apname, f'MIRIFU_{miri_channel}' in apname)
            if f'MIRIFU_{miri_channel}' in apname:
                inst_siaf.apertures[apname].plot(frame='tel', color='gray', alpha=0.5)
                slice_apernames.append(apname)
        ref_aperture = inst_siaf.apertures[f'MIRIFU_CHANNEL{miri_channel}']
        aper_display_name = f'MRS channel {miri_channel}'

    v2ref = ref_aperture.V2Ref
    v3ref = ref_aperture.V3Ref

    # How should the FOV be centered?
    if center_on.lower() == 'star':
        v2star = v2ref
        v3star = v3ref
    elif center_on.lower().startswith('comp'):
        v2star = v2ref + np.sin(comp_rel_pa_rad) * comp_sep_as
        v3star = v3ref - np.cos(comp_rel_pa_rad) * comp_sep_as
    elif center_on.lower().startswith('midpoint'):
        v2star = v2ref + np.sin(comp_rel_pa_rad) * comp_sep_as * 0.5
        v3star = v3ref - np.cos(comp_rel_pa_rad) * comp_sep_as * 0.5
    else:
        raise RuntimeError('unknown value for center_on parameter')
    plt.plot(v2star, v3star, color='black', marker='o')
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
        # Big diffraction spikes
        spikelen = diff_spike_len
        plt.plot([v2star, v2star - np.sin(ang_rad) * spikelen], [v3star, v3star + np.cos(ang_rad) * spikelen],
                 color='black', lw=1, marker='none')
        # smaller inner diffraction spikes
        if show_inner_diff_spikes:
            ang_rad = np.deg2rad(angle * 60 + 30)
            spikelen = 0.5
            plt.plot([v2star, v2star - np.sin(ang_rad) * spikelen], [v3star, v3star + np.cos(ang_rad) * spikelen],
                     color='black', lw=2, marker='none')
    # Extra horizontal spikes from the +V3 SM strut
    for angle in range(2):
        ang_rad = np.deg2rad(angle * 180 + 90)
        spikelen = 1.5
        plt.plot([v2star, v2star - np.sin(ang_rad) * spikelen], [v3star, v3star + np.cos(ang_rad) * spikelen],
                 color='black', lw=1, marker='none')

    arrowlen = 2.3
    plt.arrow(v2star, v3star, -np.sin(np.deg2rad(v3pa)) * arrowlen, np.cos(np.deg2rad(v3pa)) * arrowlen,
              color='red', lw=3, head_width=0.1)
    plt.text(v2star - np.sin(np.deg2rad(v3pa)) * arrowlen, v3star + np.cos(np.deg2rad(v3pa)) * arrowlen,
             '   N', color='red')
    arrowlen = 1.3
    plt.arrow(v2star, v3star, -np.sin(np.deg2rad(v3pa - 90)) * arrowlen, np.cos(np.deg2rad(v3pa - 90)) * arrowlen,
              color='red', lw=2, head_width=0.1)
    plt.text(v2star - np.sin(np.deg2rad(v3pa - 90)) * arrowlen, v3star + np.cos(np.deg2rad(v3pa - 90)) * arrowlen,
             '   E', color='red')

    comp_rel_pa = v3pa - comp_pa.to_value(u.deg)
    comp_rel_pa_rad = np.deg2rad(comp_rel_pa)
    plt.plot([v2star - np.sin(comp_rel_pa_rad) * comp_sep.to_value(u.arcsec)],
             [v3star + np.cos(comp_rel_pa_rad) * comp_sep.to_value(u.arcsec)],
             color='blue', lw=1, marker='s')
    plt.text(v2star - np.sin(comp_rel_pa_rad) * (comp_sep.to_value(u.arcsec) + 0.5),
             v3star + np.cos(comp_rel_pa_rad) * (comp_sep.to_value(u.arcsec) + 0.5),
             comp_name, color='blue')

    if instrument.lower() == 'nirspec':
        if nirspec_aperture == 'ifu':
            slice_V3IdlYAngle = inst_siaf.apertures['NRS_IFU_SLICE00'].V3IdlYAngle
            aplabel = 'IFUalign'
            plt.text(298.7, -499.3, f"{aplabel} X axis. Bottom of NRS detectors", rotation=slice_V3IdlYAngle - 90, fontsize=8, color='green',
                    horizontalalignment='center', verticalalignment='center')
            plt.text(301., -499.7, f"{aplabel} Y axis. ", rotation=slice_V3IdlYAngle - 180, fontsize=8, color='green',
                    horizontalalignment='center', verticalalignment='center')
            plt.text(301.2, -496.9, "Top of NRS detectors", rotation=slice_V3IdlYAngle - 90, fontsize=8, color='green',
                    horizontalalignment='center', verticalalignment='center')


        else:
            slice_V3IdlYAngle = ref_aperture.V3IdlYAngle
            xtel, ytel = ref_aperture.idl_to_tel(0,0)
            aplabel = nirspec_aperture
            plt.text(xtel+0.5, ytel+0.5, f"{aplabel}", rotation=slice_V3IdlYAngle-90, fontsize=8, color='green',
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
            print(f"Caution: {apname} contains the PSF core, and may have saturated pixels and increased noise ")
            plt.gca().add_patch(polygon)


    plt.title(f'{comp_name} at V3PA={v3pa} for {instrument} {aper_display_name.upper()}')
