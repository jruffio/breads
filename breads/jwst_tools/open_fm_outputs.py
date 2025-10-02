import os
from matplotlib.patches import Circle
import h5py
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

from breads.jwst_tools.reduction_utils import find_files_to_process

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
                flux_p, flux_error, rchi2, decs, ras, rvs = read_fm_outputs(os.path.join(fm_outputs_path, fm_output))

                flux_dithers_array.append(flux_p)
                noise_dithers_array.append(flux_error)
                rchi2_dithers_array.append(rchi2)

                flux_p[np.isnan(flux_p)] = 0
                flux_error[np.isnan(flux_error)] = 0

        flux_dithers_array = np.array(flux_dithers_array)
        noise_dithers_array = np.array(noise_dithers_array)
        rchi2_dithers_array = np.array(rchi2_dithers_array)


        flux_combined, flux_error_combined = combined_outputs(flux_dithers_array, noise_dithers_array, weighted=True)

        plot_snr_maps(targetname, band, flux_dithers_array, noise_dithers_array, flux_combined, flux_error_combined, decs, ras, rvs, coor_ptheta=coor_ptheta, savefig=True)
        plot_chi2_maps(targetname, band, rchi2_dithers_array, decs, ras, rvs, coor_ptheta=coor_ptheta)


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

def plot_snr_maps(targetname, band, flux_p, flux_error, flux_combined, flux_error_combined, decs, ras, rvs, coor_ptheta=None, savefig=True):
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    for k, rv in enumerate(rvs):
        n_dithers = flux_p.shape[0]

        ncols = 2
        nrows = math.ceil(n_dithers / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 8))
        fig.suptitle(f"{targetname} SNR maps for {band}, \n rv = {rv} km/s")

        # Mettre axes en 1D pour itérer facilement
        axes = np.atleast_1d(axes).ravel()

        for index in range(n_dithers):
            ax = axes[index]
            snr_map = (flux_p[index, k] / flux_error[index, k]).T
            im = ax.pcolormesh(ras, decs, snr_map, cmap='viridis', vmin=-2, vmax=7)
            ax.invert_xaxis()
            ax.set_title(f"dither #{index+1}")  # optional titles
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, shrink=1)
            cbar.set_label('SNR', rotation=270, labelpad=10)

            # Custom format_coord to display x,y,z
            def format_coord(x_val, y_val, ras=ras, decs=decs, Z=snr_map):
                col = np.searchsorted(ras, x_val) - 1
                row = np.searchsorted(decs, y_val) - 1
                if 0 <= col < Z.shape[1] and 0 <= row < Z.shape[0]:
                    z = Z[row, col]
                    return f"RA={x_val:.2f}, DEC={y_val:.2f}, SNR={z:.2f}"
                else:
                    return f"RA={x_val:.2f}, DEC={y_val:.2f}"
            ax.format_coord = format_coord

            # Optional: add coordinates if provided
            if coor_ptheta is not None:
                for coor in coor_ptheta:
                    dra, ddec = coor
                    ax.plot(dra, ddec, 'ro')
                    circle = Circle((dra, ddec), 0.15 / 2, color='red', fill=False, linewidth=1.5)
                    ax.add_patch(circle)

        # Set axis labels
        axes = np.atleast_2d(axes)

        # X label
        for a in axes[-1, :]:
            a.set_xlabel('Delta RA (")')

        # Y label
        for a in axes[:, 0]:
            a.set_ylabel('Delta DEC (")')

        plt.tight_layout()
        plt.show()

        # === Combined map ===
        fig, ax = plt.subplots()
        plt.xlabel('Delta RA (")')
        plt.ylabel('Delta DEC (")')
        plt.title(f"{targetname} combined SNR map on {band}, \n rv = {rv} km/s")
        plt.gca().invert_xaxis()
        snr_combined = (flux_combined[k] / flux_error_combined[k]).T
        img = plt.pcolormesh(ras, decs, snr_combined, cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label('SNR', rotation=270)

        # Custom format_coord for combined
        def format_coord(x_val, y_val, ras=ras, decs=decs, Z=snr_combined):
            col = np.searchsorted(ras, x_val) - 1
            row = np.searchsorted(decs, y_val) - 1
            if 0 <= col < Z.shape[1] and 0 <= row < Z.shape[0]:
                z = Z[row, col]
                return f"RA={x_val:.2f}, DEC={y_val:.2f}, SNR={z:.2f}"
            else:
                return f"RA={x_val:.2f}, DEC={y_val:.2f}"
        ax.format_coord = format_coord

        if coor_ptheta is not None:
            for coor in coor_ptheta:
                dra, ddec = coor
                ax.plot(dra, ddec, 'ro')
                circle = Circle((dra, ddec), 0.15 / 2, color='red', fill=False, linewidth=1.5)
                ax.add_patch(circle)

        plt.show()


def plot_chi2_maps(targetname, band, rchi2, decs, ras, rvs, coor_ptheta=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    for k, rv in enumerate(rvs):
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))  # optional figsize
        fig.suptitle(f'{targetname} Noise scaling factor maps for {band} \n fringe flat = extended flat ')

        # Plot the scaling maps
        for i in range(2):
            for j in range(2):
                index = 2 * i + j
                scaling_map = np.sqrt(rchi2[index, k].T)
                im = ax[i, j].pcolormesh(ras, decs, scaling_map, cmap='viridis')
                ax[i, j].invert_xaxis()
                ax[i, j].set_title(f"dither #{index+1}")  # optional titles

                # Add colorbar
                cbar = fig.colorbar(im, ax=ax[i, j], shrink=1)
                cbar.set_label('Scaling', rotation=270, labelpad=10)

                # Custom format_coord for interactive value display
                def format_coord(x_val, y_val, ras=ras, decs=decs, Z=scaling_map):
                    col = np.searchsorted(ras, x_val) - 1
                    row = np.searchsorted(decs, y_val) - 1
                    if 0 <= col < Z.shape[1] and 0 <= row < Z.shape[0]:
                        z = Z[row, col]
                        return f"RA={x_val:.2f}, DEC={y_val:.2f}, Scaling={z:.3f}"
                    else:
                        return f"RA={x_val:.2f}, DEC={y_val:.2f}"
                ax[i, j].format_coord = format_coord

                # Optional: add coordinates if provided
                if coor_ptheta is not None:
                    for coor in coor_ptheta:
                        dra, ddec = coor
                        ax[i, j].plot(dra, ddec, 'ro')
                        circle = Circle((dra, ddec), 0.15 / 2, color='red', fill=False, linewidth=1.5)
                        ax[i, j].add_patch(circle)

        # Set axis labels
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
                    flux_p, flux_error, rchi2, decs, ras, rvs = read_fm_outputs(os.path.join(fm_outputs_path, fm_output))

                    flux_dithers_array.append(flux_p)
                    noise_dithers_array.append(flux_error)
                    rchi2_dithers_array.append(rchi2)

                    flux_p[np.isnan(flux_p)] = 0
                    flux_error[np.isnan(flux_error)] = 0

        flux_dithers_array = np.array(flux_dithers_array)
        noise_dithers_array = np.array(noise_dithers_array)

        print(flux_dithers_array.shape)

        flux_combined, flux_error_combined = combined_outputs(flux_dithers_array, noise_dithers_array, weighted=True)

        for k, rv in enumerate(rvs):

            fig, ax = plt.subplots()
            plt.xlabel('Delta RA (")')
            plt.ylabel('Delta DEC (")')
            plt.title(f"{targetname} combined SNR map on channel {channel}, rv={rv} km/s")
            plt.gca().invert_xaxis()
            img = plt.pcolormesh(ras, decs, (flux_combined[k] / flux_error_combined[k]).transpose(), cmap='viridis')
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

def plot_ccf(ra_0, dec_0, ras, decs, rvs, flux_combined, flux_error_combined):
    dras, ddec = ras - ra_0, decs - dec_0
    dras, ddec = np.meshgrid(dras, ddec)
    sep = np.sqrt(dras**2 + ddec**2)
    i, j = np.where(sep==np.nanmin(sep))[0], np.where(sep==np.nanmin(sep))[1]
    print(i, j)
    corr_value = []
    for k, rv in enumerate(rvs):
        snr = (flux_combined[k]/flux_error_combined[k]).transpose()
        corr_value.append(snr[i, j-1])
        plt.title(f"{rv}")
        plt.imshow(snr)
        plt.show()

    plt.plot(rvs, corr_value)
    plt.show()