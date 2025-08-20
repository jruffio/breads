import os
import numpy as np
import os
from matplotlib.patches import Circle

import h5py
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib

from breads.jwst_tools.reduction_utils import find_files_to_process


#matplotlib.use('tkAgg')


def open_fm_outputs_miri(uncaldir, targetname, n_nodes, list_bands=None, coor_ptheta=None):
    if list_bands is None:
        list_bands = ['1A', '1B', '1C', '2A', '2B', '2C', '3A', '3B', '3C', '4A', '4B', '4C']

    for band in list_bands:
        fm_outputs_path = os.path.join(uncaldir, targetname, f'fm_outputs_{n_nodes}_nodes', band)
        list_fm_outputs = sort_by_dither_number(os.listdir(fm_outputs_path))
        flux_dithers_array = []
        noise_dithers_array = []
        rchi2_dithers_array = []
        for fm_output in list_fm_outputs:
            if fm_output.endswith('.hdf5'):
                flux_p, flux_error, rchi2, decs, ras, rv = read_fm_outputs(os.path.join(fm_outputs_path, fm_output))

                flux_dithers_array.append(flux_p[0])
                noise_dithers_array.append(flux_error[0])
                rchi2_dithers_array.append(rchi2[0])

                flux_p[np.isnan(flux_p)] = 0
                flux_error[np.isnan(flux_error)] = 0

        flux_dithers_array = np.array(flux_dithers_array)
        noise_dithers_array = np.array(noise_dithers_array)
        rchi2_dithers_array = np.array(rchi2_dithers_array)

        flux_combined, flux_error_combined = combined_outputs(flux_dithers_array, noise_dithers_array, weighted=True)

        plot_snr_maps(targetname, band, flux_dithers_array, noise_dithers_array, flux_combined, flux_error_combined, decs, ras, rv, coor_ptheta=coor_ptheta, savefig=True)
        plot_chi2_maps(targetname, band, rchi2_dithers_array, decs, ras, rv, coor_ptheta=coor_ptheta)


def sort_by_dither_number(file_list):
    """
    Sorts a list of filenames by the last number before '_mirifushort.hdf5' in each filename.

    Parameters:
        file_list (list): List of filenames as strings.

    Returns:
        list: Sorted list of filenames.
    """

    def extract_index(filename):
        # Split by underscores and find the 2nd-to-last element (e.g., '00003')
        return int(filename.split('_')[-4])
    # Keep only .hdf5 files
    hdf5_files = [f for f in file_list if f.endswith('.hdf5')]
    return sorted(hdf5_files, key=extract_index)

def read_fm_outputs(hdf5_file):
    with (h5py.File(hdf5_file, 'r') as file):
        flux_p = np.array(file.get("linparas"))[:, :, :, 0]  # taking the flux map for linear parameters #0 i.e. for companion flux only
        flux_error = np.array(file.get("linparas_err")[:, :, :, 0])
        rchi2 = np.array(file.get("rchi2")[:, :, :])
        decs = np.array(file.get("decs"))
        ras = np.array(file.get("ras"))
        rv = np.array(file.get("rvs"))

        return flux_p, flux_error, rchi2, decs, ras, rv

def combined_outputs(flux_dithers_array, noise_dithers_array, weighted=True):

    if weighted:
        flux_combined = np.nansum(flux_dithers_array / noise_dithers_array ** 2, axis=0) / np.nansum(1 / noise_dithers_array ** 2, axis=0)
        noise_combined = 1 / np.sqrt(np.nansum(1 / noise_dithers_array ** 2, axis=0))
    else:
        flux_combined = np.nanmean(flux_dithers_array, 0)
        noise_combined = np.nanmean(noise_dithers_array, 0)

    return flux_combined, noise_combined

def plot_snr_maps(targetname, band, flux_p, flux_error, flux_combined, flux_error_combined, decs, ras, rv, coor_ptheta=None, savefig=True):
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))  # optional figsize
    fig.suptitle(f'{targetname} SNR maps for {band}')
    # Plot the SNR maps
    for i in range(2):
        for j in range(2):
            index = 2 * i + j
            im = ax[i, j].pcolormesh(ras, decs, (flux_p[index] / flux_error[index]).T, cmap='viridis')
            ax[i, j].invert_xaxis()
            ax[i, j].set_title(f"dither #{index+1}")  # optional titles
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax[i, j], shrink=1)
            cbar.set_label('SNR', rotation=270, labelpad=10)
            # Optional: add coordinates if provided
            if coor_ptheta is not None:
                for coor in coor_ptheta:
                    dra, ddec = coor
                    ax[i, j].plot(dra, ddec, 'ro')
                    circle = Circle((dra, ddec), 0.15 / 2, color='red', fill=False, linewidth=1.5)
                    ax[i, j].add_patch(circle)

    # Set axis labels (you might want to set these only for outer plots)
    for a in ax[-1, :]:  # bottom row
        a.set_xlabel('Delta RA (")')
    for a in ax[:, 0]:  # left column
        a.set_ylabel('Delta DEC (")')

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    plt.xlabel('Delta RA (")')
    plt.ylabel('Delta DEC (")')
    plt.title(f"{targetname} combined SNR map on {band}")
    plt.gca().invert_xaxis()
    img = plt.pcolormesh(ras, decs, (flux_combined / flux_error_combined).transpose(), cmap='viridis')
    cbar = plt.colorbar()
    cbar.set_label('SNR', rotation=270)

    if coor_ptheta is not None:
        for coor in coor_ptheta:
            dra, ddec = coor
            ax.plot(dra, ddec, 'ro')
            circle = Circle((dra, ddec), 0.15 / 2, color='red', fill=False, linewidth=1.5)
            ax.add_patch(circle)

    plt.show()

def plot_chi2_maps(targetname, band, rchi2, decs, ras, rv, coor_ptheta=None):
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))  # optional figsize
    fig.suptitle(f'{targetname} Noise scaling factor maps for {band} \n fringe flat = extended flat ')
    # Plot the SNR maps
    for i in range(2):
        for j in range(2):
            index = 2 * i + j
            im = ax[i, j].pcolormesh(ras, decs, np.log10(rchi2[index].T), cmap='viridis', vmax=1.7)
            ax[i, j].invert_xaxis()
            ax[i, j].set_title(f"dither #{index+1}")  # optional titles
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax[i, j], shrink=1)
            cbar.set_label('log(scaling)', rotation=270, labelpad=10)
            # Optional: add coordinates if provided
            if coor_ptheta is not None:
                for coor in coor_ptheta:
                    dra, ddec = coor
                    ax[i, j].plot(dra, ddec, 'ro')
                    circle = Circle((dra, ddec), 0.15 / 2, color='red', fill=False, linewidth=1.5)
                    ax[i, j].add_patch(circle)

    # Set axis labels (you might want to set these only for outer plots)
    for a in ax[-1, :]:  # bottom row
        a.set_xlabel('Delta RA (")')
    for a in ax[:, 0]:  # left column
        a.set_ylabel('Delta DEC (")')

    plt.tight_layout()
    plt.show()

def plot_combined_channel(uncaldir, targetname, n_nodes, list_channels=None, coor_ptheta=None):
    if list_channels is None:
        list_channels = ['1', '2', '3', '4']
    list_bands = ['A', 'B', 'C']
    for channel in list_channels:
        flux_dithers_array = []
        noise_dithers_array = []
        rchi2_dithers_array = []
        for b in list_bands:
            band = channel + b
            fm_outputs_path = os.path.join(uncaldir, targetname, f'fm_outputs_{n_nodes}_nodes', band)
            list_fm_outputs = sort_by_dither_number(os.listdir(fm_outputs_path))
            for fm_output in list_fm_outputs:
                if fm_output.endswith('.hdf5'):
                    flux_p, flux_error, rchi2, decs, ras, rv = read_fm_outputs(os.path.join(fm_outputs_path, fm_output))

                    flux_dithers_array.append(flux_p[0])
                    noise_dithers_array.append(flux_error[0])
                    rchi2_dithers_array.append(rchi2[0])

                    flux_p[np.isnan(flux_p)] = 0
                    flux_error[np.isnan(flux_error)] = 0

        flux_dithers_array = np.array(flux_dithers_array)
        noise_dithers_array = np.array(noise_dithers_array)
        rchi2_dithers_array = np.array(rchi2_dithers_array)

        flux_combined, flux_error_combined = combined_outputs(flux_dithers_array, noise_dithers_array, weighted=True)

        fig, ax = plt.subplots()
        plt.xlabel('Delta RA (")')
        plt.ylabel('Delta DEC (")')
        plt.title(f"{targetname} combined SNR map on channel {channel}")
        plt.gca().invert_xaxis()
        img = plt.pcolormesh(ras, decs, (flux_combined / flux_error_combined).transpose(), cmap='viridis', vmax=10)
        cbar = plt.colorbar()
        cbar.set_label('SNR', rotation=270)

        if coor_ptheta is not None:
            for coor in coor_ptheta:
                dra, ddec = coor
                ax.plot(dra, ddec, 'ro')
                circle = Circle((dra, ddec), 0.15 / 2, color='red', fill=False, linewidth=1.5)
                ax.add_patch(circle)

        plt.show()

def plot_combined_hf_starspectrum(uncaldir, targetname, n_nodes, list_bands=None):
    if list_bands is None:
        list_bands = ['1A', '1B', '1C', '2A', '2B', '2C', '3A', '3B', '3C', '4A', '4B', '4C']

    for band in list_bands:
        utils_outputs_path = os.path.join(uncaldir, targetname, f'utils_fm_{n_nodes}_nodes', band)
        list_utils_outputs = find_files_to_process(utils_outputs_path, filetype='starspec_contnorm.fits')
        print(list_utils_outputs)
        plt.title(f"{targetname} combined HF star spectrum on band {band}")
        plt.xlabel('Wavelength (microns)')
        plt.ylabel('Continuum normalized flux')
        for utils in list_utils_outputs:
            com_flux = fits.open(utils)['COM_FLUXES'].data
            wave = fits.open(utils)[0].data
            plt.plot(wave, com_flux, label=os.path.basename(utils)[:25])
        plt.ylim([0.96, 1.04])
        plt.legend()
        plt.show()


coor_ptheta = [[1.643, 0.507], [-0.253, 0.922], [-0.626, -0.311], [-0.188, 0.356]] #H
#plot_combined_hf_starspectrum("/Users/abidot/Desktop/miri_data_4829_full/", 'HD 218396', 40, list_bands=['1A','1B','1C'])
#plot_combined_channel("/Users/abidot/Desktop/miri_data_4829_full/", 'HD 218396', 40, list_channels=['1'], coor_ptheta=coor_ptheta)
open_fm_outputs_miri("/Users/abidot/Desktop/miri_data_4829_full/", 'HD 218396', 40, list_bands=['1A'], coor_ptheta=coor_ptheta)