import numpy as np
import matplotlib.pyplot as plt
import pysiaf
import astropy.units as u

def visualize_nrs_ifu_fov(comp_name, comp_sep, comp_pa, v3pa, center_on = 'star',
                          show_inner_diff_spikes=False):
    """ Visualize NIRSpec IFU FOV for a companion"""
	# Disclaimer: Written in a bit of a rush; code could be cleaned up some still...

    comp_rel_pa = v3pa - comp_pa.to_value(u.deg)
    comp_rel_pa_rad = np.deg2rad(comp_rel_pa)
    comp_sep_as = comp_sep.to_value(u.arcsec)

    # Setup figure draf siaf
    plt.figure()
    nrs_siaf = pysiaf.Siaf('NIRSpec')
    for apname in nrs_siaf.apernames:
        if 'IFU_SLICE' in apname:
            nrs_siaf.apertures[apname].plot(frame='tel', color='gray', alpha=0.5)


    v2ref = nrs_siaf.apertures['NRS_FULL_IFU'].V2Ref
    v3ref = nrs_siaf.apertures['NRS_FULL_IFU'].V3Ref

    # How should the FOV be centered?
    if center_on.lower() == 'star':
        v2star = v2ref
        v3star = v3ref
    elif center_on.lower().startswith('comp'):
        v2star = v2ref + np.sin(comp_rel_pa_rad)*comp_sep_as
        v3star = v3ref - np.cos(comp_rel_pa_rad)*comp_sep_as
    elif center_on.lower().startswith('midpoint'):
        v2star = v2ref + np.sin(comp_rel_pa_rad)*comp_sep_as* 0.5
        v3star = v3ref - np.cos(comp_rel_pa_rad)*comp_sep_as* 0.5
    plt.plot(v2star, v3star, color='black', marker='o')


    for angle in range(6):
        ang_rad = np.deg2rad(angle*60)
        # Big diffraction spikes
        spikelen = 2
        plt.plot([v2star, v2star-np.sin(ang_rad)*spikelen], [v3star, v3star+np.cos(ang_rad)*spikelen],
                 color='black', lw=1, marker='none')
        # smaller inner diffraction spikes
        if show_inner_diff_spikes:
            ang_rad = np.deg2rad(angle*60+30)
            spikelen = 0.5
            plt.plot([v2star, v2star-np.sin(ang_rad)*spikelen], [v3star, v3star+np.cos(ang_rad)*spikelen],
                     color='black', lw=2, marker='none')
    # Extra horizontal spikes from the +V3 SM strut
    for angle in range(2):
        ang_rad = np.deg2rad(angle*180 + 90)
        spikelen = 1.5
        plt.plot([v2star, v2star-np.sin(ang_rad)*spikelen], [v3star, v3star+np.cos(ang_rad)*spikelen],
                 color='black', lw=1, marker='none')


    arrowlen=2.3
    plt.arrow(v2star, v3star, -np.sin(np.deg2rad(v3pa))*arrowlen, np.cos(np.deg2rad(v3pa))*arrowlen,
                 color='red', lw=3, head_width=0.1)
    plt.text(v2star -np.sin(np.deg2rad(v3pa))*arrowlen, v3star + np.cos(np.deg2rad(v3pa))*arrowlen,
              '   N', color='red')
    arrowlen=1.3
    plt.arrow(v2star, v3star, -np.sin(np.deg2rad(v3pa-90))*arrowlen, np.cos(np.deg2rad(v3pa-90))*arrowlen,
                 color='red', lw=2, head_width=0.1)
    plt.text(v2star -np.sin(np.deg2rad(v3pa-90))*arrowlen, v3star + np.cos(np.deg2rad(v3pa-90))*arrowlen,
              '   E', color='red')


    comp_rel_pa = v3pa - comp_pa.to_value(u.deg)
    comp_rel_pa_rad = np.deg2rad(comp_rel_pa)
    #print(comp_pa, comp_rel_pa)
    plt.plot([v2star-np.sin(comp_rel_pa_rad)*comp_sep.to_value(u.arcsec)],
             [v3star+np.cos(comp_rel_pa_rad)*comp_sep.to_value(u.arcsec)],
                 color='blue', lw=1, marker='s')
    plt.text(v2star-np.sin(comp_rel_pa_rad)*(comp_sep.to_value(u.arcsec)+0.5),
             v3star+np.cos(comp_rel_pa_rad)*(comp_sep.to_value(u.arcsec)+0.5),
             comp_name,    color='blue')

    slice_V3IdlYAngle = nrs_siaf.apertures['NRS_IFU_SLICE00'].V3IdlYAngle

    plt.text(298.7, -499.3, "IFUalign X axis. Bottom of NRS detectors", rotation=slice_V3IdlYAngle-90, fontsize=8, color='green',
            horizontalalignment='center', verticalalignment='center')
    plt.text(301., -499.7, "IFUalign Y axis. ", rotation=slice_V3IdlYAngle-180, fontsize=8, color='green',
            horizontalalignment='center', verticalalignment='center')
    plt.text(301.2, -496.9, "Top of NRS detectosr", rotation=slice_V3IdlYAngle-90, fontsize=8, color='green',
            horizontalalignment='center', verticalalignment='center')

    plt.title(f'{comp_name} at V3PA={v3pa}')
