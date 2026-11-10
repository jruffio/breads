import os
from matplotlib.patches import Circle
import h5py
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d
import astropy.units as u
import astropy.constants as const
import numpy as np
from astropy.io import fits
from astroquery.svo_fps import SvoFps
from breads.jwst_reduction.reduction_utils import find_files_to_process

# ===================================================
# Function to open the old hdf5 files saving the full forward modeling especially for MIRI.
# ===================================================

def open_fm_outputs_miri(uncaldir, targetname, n_nodes, list_bands=None, companion_offsets=None):
    if list_bands is None:
        list_bands = ['1A', '1B', '1C', '2A', '2B', '2C', '3A', '3B', '3C', '4A', '4B', '4C']

    for channel_band in list_bands:
        fm_outputs_path = os.path.join(uncaldir, targetname, f'fm_outputs_{n_nodes}_nodes', channel_band)
        list_fm_outputs = sort_by_dither_number(os.listdir(fm_outputs_path))
        flux_dithers_array = []
        noise_dithers_array = []
        rchi2_dithers_array = []
        for fm_output in list_fm_outputs:
            if h5py.is_hdf5(os.path.join(fm_outputs_path, fm_output)):
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

        plot_snr_maps(targetname, channel_band, flux_dithers_array, noise_dithers_array, flux_combined, flux_error_combined, decs, ras, rvs, companion_offsets=companion_offsets, savefig=True)
        plot_chi2_maps(targetname, channel_band, rchi2_dithers_array, decs, ras, rvs, companion_offsets=companion_offsets)

def open_fm_outputs_nirspec(uncaldir, targetname, n_nodes, companion_offsets=None):
    fm_outputs_path = os.path.join(uncaldir, targetname, f'fm_outputs_{n_nodes}_nodes')
    print(fm_outputs_path)
    list_fm_outputs = sort_by_dither_number(os.listdir(fm_outputs_path))
    print(list_fm_outputs)
    flux_dithers_array = []
    noise_dithers_array = []
    rchi2_dithers_array = []
    for fm_output in list_fm_outputs:
        if h5py.is_hdf5(os.path.join(fm_outputs_path, fm_output)):
            flux_p, flux_error, rchi2, decs, ras, rvs = read_fm_outputs(os.path.join(fm_outputs_path, fm_output))

            flux_dithers_array.append(flux_p)
            noise_dithers_array.append(flux_error)
            rchi2_dithers_array.append(rchi2)

            flux_p[np.isnan(flux_p)] = 0
            flux_error[np.isnan(flux_error)] = 0

            plt.imshow(flux_p[0] / flux_error[0])
            plt.show()
            print(flux_p[0])
            print(ras)

    flux_dithers_array = np.array(flux_dithers_array)
    noise_dithers_array = np.array(noise_dithers_array)
    rchi2_dithers_array = np.array(rchi2_dithers_array)



    flux_combined, flux_error_combined = combined_outputs(flux_dithers_array, noise_dithers_array, weighted=True)

    plot_snr_maps_nirspec(targetname, flux_dithers_array, noise_dithers_array, flux_combined, flux_error_combined, decs,
                  ras, rvs, companion_offsets=companion_offsets, savefig=True)
    #plot_chi2_maps(targetname, band, rchi2_dithers_array, decs, ras, rvs, companion_offsets=companion_offsets)


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
    """ Combine flux and uncertainties from multiple dithers

    Parameters
    ----------
    flux_dithers_array
        array of the fluxes for each dither
    noise_dithers_array
        array of the estimated uncertaintiesfor each dither
    weighted: bool
      If true, compute combined cubes weighted by the individual SNR in the input dithers.
      If false, just take the plain average of the SNRs per each dither

    Returns
    -------
    flux_combined, noise_combined
    """

    if weighted:
        # use numpy masked array here because it works better than nansum in this case
        # specifically it better handles the 1/noise part, which can yield np.Inf instead of np.Nan for some inputs
        masked_flux_dithers_array = np.ma.masked_array(flux_dithers_array, mask = np.isnan(flux_dithers_array))
        masked_noise_dithers_array = np.ma.masked_array(noise_dithers_array, mask = np.isnan(flux_dithers_array))

        flux_combined = np.sum(masked_flux_dithers_array / masked_noise_dithers_array ** 2, axis=0) / np.sum(1 / masked_noise_dithers_array ** 2, axis=0)
        noise_combined = 1 / np.sqrt(np.sum(1 / masked_noise_dithers_array ** 2, axis=0))
    else:
        flux_combined = np.nanmean(flux_dithers_array, 0)
        noise_combined = np.nanmean(noise_dithers_array, 0)

    return flux_combined, noise_combined


def annotate_plot_star_and_companions(ax, companion_offsets=None, mark_star=True):
    """ Utility function to draw markers for star and companions on a given plot

    Parameters
    -----------
    ax : matplotlib Axes instance
        Axes to plot into
    companion_offsets : 2D list or ndarray of floats, shape [N, 2]
        List or array of coordinates for offset planets or other substellar companions.
        Specified as [∆RA, ∆Dec] in arcseconds for each companion.
    mark_star : bool
        Draw a star symbol at coordinates (0,0) ?
    """
    if companion_offsets is not None:
        for coor in companion_offsets:
            dra, ddec = coor
            ax.plot(dra, ddec, 'ro')
            circle = Circle((dra, ddec), 0.15 / 2, color='red', fill=False, linewidth=1.5)
            ax.add_patch(circle)

    if mark_star:
        ax.scatter(0, 0, s=250, marker='*', facecolor='yellow', edgecolor='black')



def plot_snr_maps(targetname, channel_band, flux_p, flux_error, flux_combined, flux_error_combined, decs, ras, rvs,
                  vmin=-2, vmax = 7,
                  companion_offsets=None, savefig=True, mark_star=True):

    for k, rv in enumerate(rvs):
        n_dithers = flux_p.shape[0]

        ncols = 2
        nrows = math.ceil(n_dithers / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 8))
        fig.suptitle(f"{targetname} SNR maps for {channel_band}, \n rv = {rv} km/s")

        # Flatten axes to 1D for easy iteration
        axes = np.atleast_1d(axes).ravel()

        for index in range(n_dithers):
            ax = axes[index]
            snr_map = (flux_p[index, k] / flux_error[index, k]).T
            im = ax.pcolormesh(ras, decs, snr_map, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.invert_xaxis()
            ax.set_aspect('equal')
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

            annotate_plot_star_and_companions(ax, companion_offsets, mark_star)

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
        plt.title(f"{targetname} combined SNR map on {channel_band}, \n rv = {rv} km/s")
        plt.gca().invert_xaxis()
        snr_combined = (flux_combined[k] / flux_error_combined[k]).T
        img = plt.pcolormesh(ras, decs, snr_combined, cmap='viridis')
        ax.set_aspect('equal')
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

        annotate_plot_star_and_companions(ax, companion_offsets, mark_star)

        plt.show()


def plot_chi2_maps(targetname, channel_band, rchi2, decs, ras, rvs, companion_offsets=None, mark_star=True):

    for k, rv in enumerate(rvs):
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))  # optional figsize
        fig.suptitle(f'{targetname} Noise scaling factor maps for {channel_band} \n fringe flat = extended flat ')

        # Plot the scaling maps
        for i in range(2):
            for j in range(2):
                index = 2 * i + j
                scaling_map = np.sqrt(rchi2[index, k].T)
                im = ax[i, j].pcolormesh(ras, decs, scaling_map, cmap='viridis')
                ax[i, j].invert_xaxis()
                ax[i, j].set_aspect('equal')
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

                annotate_plot_star_and_companions(ax[i, j], companion_offsets, mark_star)

        # Set axis labels
        for a in ax[-1, :]:  # bottom row
            a.set_xlabel('Delta RA (")')
        for a in ax[:, 0]:  # left column
            a.set_ylabel('Delta DEC (")')

        plt.tight_layout()
        plt.show()


def plot_combined_channel(uncaldir, targetname, n_nodes, list_channels=None, companion_offsets=None, mark_star=True):
    if list_channels is None:
        list_channels = ['1', '2', '3', '4']
    list_bands = ['A', 'B', 'C']
    for channel in list_channels:
        flux_dithers_array = []
        noise_dithers_array = []
        rchi2_dithers_array = []
        for sub_band in list_bands:
            channel_band = channel + sub_band
            fm_outputs_path = os.path.join(uncaldir, targetname, f'fm_outputs_{n_nodes}_nodes', channel_band)
            list_fm_outputs = sort_by_dither_number(os.listdir(fm_outputs_path))
            for fm_output in list_fm_outputs:
                if h5py.is_hdf5(os.path.join(fm_outputs_path, fm_output)):
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
            ax.set_aspect('equal')
            cbar = plt.colorbar()
            cbar.set_label('SNR', rotation=270)

            annotate_plot_star_and_companions(ax, companion_offsets, mark_star)

            plt.show()

def plot_combined_hf_starspectrum(uncaldir, targetname, n_nodes, list_bands=None):
    if list_bands is None:
        list_bands = ['1A', '1B', '1C', '2A', '2B', '2C', '3A', '3B', '3C', '4A', '4B', '4C']

    for channel_band in list_bands:
        utils_outputs_path = os.path.join(uncaldir, targetname, f'utils_fm_{n_nodes}_nodes', channel_band)
        list_utils_outputs = find_files_to_process(utils_outputs_path, filetype='starspec_contnorm.fits')
        print(list_utils_outputs)
        plt.title(f"{targetname} combined HF star spectrum on band {channel_band}")
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


def plot_snr_maps_nirspec(targetname, flux_p, flux_error, flux_combined, flux_error_combined, decs,
                  ras, rvs, companion_offsets=None, vmin=-2, vmax=7, savefig=True, mark_star=True):
    for k, rv in enumerate(rvs):
        n_dithers = flux_p.shape[0]

        ncols = 2
        nrows = math.ceil(n_dithers / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 8))
        fig.suptitle(f"{targetname} SNR maps, \n rv = {rv} km/s")

        # Flatten axes to 1D for easy iteration
        axes = np.atleast_1d(axes).ravel()

        for index in range(n_dithers):
            ax = axes[index]
            snr_map = (flux_p[index, k] / flux_error[index, k]).T
            im = ax.pcolormesh(ras, decs, snr_map, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.invert_xaxis()
            ax.set_aspect('equal')
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

            annotate_plot_star_and_companions(ax, companion_offsets, mark_star)

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
        plt.title(f"{targetname} combined SNR map, \n rv = {rv} km/s")
        plt.gca().invert_xaxis()
        snr_combined = (flux_combined[k] / flux_error_combined[k]).T
        img = plt.pcolormesh(ras, decs, snr_combined, cmap='viridis')
        ax.set_aspect('equal')
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

        annotate_plot_star_and_companions(ax, companion_offsets, mark_star)

        plt.show()


# ===================================================
# Function to compute the contrast curves for MIRI normalized in the numerous photometric filter available for the MRS wavelength coverage.
# ===================================================

_filter_index_cache = {}

def resolve_svo_filter_id(band, facility="JWST", instrument="MIRI"):
    """
    Resolve a band name into an exact SVO filter ID. Accepts both a short
    name ('F1550C') and an already complete ID ('JWST/MIRI.F1550C').
    """
    # If band is already a complete ID, keep only the filter part
    band_short = band.split(".")[-1] if "." in band else band

    key = (facility, instrument)
    if key not in _filter_index_cache:
        table = SvoFps.get_filter_list(facility=facility, instrument=instrument)
        index = {fid.split(".")[-1].upper(): fid for fid in table["filterID"]}
        _filter_index_cache[key] = index

    index = _filter_index_cache[key]
    band_key = band_short.upper()

    if band_key not in index:
        raise ValueError(
            f"Filter '{band}' not found for {facility}/{instrument}. "
            f"Available filters: {sorted(index.keys())}"
        )

    return index[band_key]


def load_filter_svo(filter_id, retry_on_cache_miss=True):
    try:
        transmission = SvoFps.get_transmission_data(filter_id)
    except IndexError:
        if retry_on_cache_miss:
            # The local cache may have stored a transient network failure
            SvoFps.clear_cache()
            transmission = SvoFps.get_transmission_data(filter_id)
        else:
            raise

    ph_wvs = transmission["Wavelength"].data / 1e4  # Angstrom -> micron
    ph_trans = transmission["Transmission"].data
    return interp1d(ph_wvs, ph_trans, bounds_error=False, fill_value=0)


def synthetic_fnu(grid_wvs, grid_spec, band, facility="JWST", instrument="MIRI"):
    """
    Compute the synthetic flux Fnu (in MJy) of a spectrum through a JWST
    filter identified by its SVO name, in the "photon counter" convention.

    grid_wvs  : wavelength array in micron
    grid_spec : flux array in W/m^2/um, last axis = wavelength
    filter_id : SVO filter ID, e.g. "JWST/MIRI.F770W"
    """
    filter_id = resolve_svo_filter_id(band, facility, instrument)
    photfilter_f = load_filter_svo(filter_id)

    grid_dwvs = grid_wvs[1:] - grid_wvs[:-1]
    grid_dwvs = np.insert(grid_dwvs, 0, grid_dwvs[0])

    trans = photfilter_f(grid_wvs)

    photfilter_wv0 = np.sqrt(
        np.nansum(grid_dwvs * trans * grid_wvs)
        / np.nansum(grid_dwvs * trans / grid_wvs)
    )

    filter_norm = np.nansum(grid_dwvs * trans * grid_wvs)

    Flambda = np.nansum(
        grid_dwvs * trans * grid_wvs * grid_spec, axis=-1
    ) / filter_norm

    Flambda = Flambda * (u.W * u.m ** -2 / u.um)
    Fnu = Flambda * (photfilter_wv0 * u.um) ** 2 / const.c

    return Fnu.to(u.MJy).value


def to_plain_array(arr):
    """Convert a MaskedArray into a plain ndarray, with NaN for masked elements."""
    if np.ma.isMaskedArray(arr):
        return np.ma.filled(arr.astype(float), np.nan)
    return np.asarray(arr, dtype=float)

def save_contrast_curves_fits(results, targetname, output_path=None):
    """
    Save all median contrast curves (one per band/photfilter_name
    combination) in a single FITS file.

    Each HDU holds the 'sep' and 'contrast_med' columns, with BAND and
    FILTER header keywords identifying the curve.
    """
    if output_path is None:
        output_path = f"{targetname}_contrast_curves.fits"

    hdus = [fits.PrimaryHDU()]

    for channel_band, channel_band_results in results.items():
        for photfilter_name, curves in channel_band_results.items():
            col_sep = fits.Column(name="sep", format="D", array=curves["bin_centers"])
            col_contrast = fits.Column(name="contrast_med", format="D", array=curves["contrast_med"])
            hdu = fits.BinTableHDU.from_columns([col_sep, col_contrast])
            hdu.header["BAND"] = (channel_band, "MRS band used for the noise measurement")
            hdu.header["FILTER"] = (photfilter_name, "Photometric filter the contrast is renormalized to")
            hdu.header["EXTNAME"] = f"{channel_band}_{photfilter_name}"
            hdus.append(hdu)

    fits.HDUList(hdus).writeto(output_path, overwrite=True)
    print(f"Contrast curves saved to {output_path}")
    return output_path


def load_contrast_curve_fits(fits_path, channel_band, photfilter_name):
    """
    Load the median contrast curve for a given band/filter combination,
    from the file produced by save_contrast_curves_fits.

    Returns (sep, contrast_med).
    """
    with fits.open(fits_path) as hdul:
        for hdu in hdul[1:]:
            if hdu.header.get("BAND") == channel_band and hdu.header.get("FILTER") == photfilter_name:
                return hdu.data["sep"], hdu.data["contrast_med"]

    raise ValueError(
        f"No curve found for band='{channel_band}', filter='{photfilter_name}' in {fits_path}. "
        f"Available extensions: {[hdu.header.get('EXTNAME') for hdu in fits.open(fits_path)[1:]]}"
    )


def list_available_curves(fits_path):
    """List the (band, filter) pairs available in a contrast-curve file."""
    with fits.open(fits_path) as hdul:
        return [(hdu.header.get("BAND"), hdu.header.get("FILTER")) for hdu in hdul[1:]]


def plot_contrast_curves(uncaldir, targetname, list_nodes, flux_star, grid_wvs, grid_spec,
                          norm_band, list_bands=None, facility="JWST",
                          instrument="MIRI", save_contrast_curves=False, contrast_output_path=None, plot=True):
    """
    Parameters
    ----------
    flux_star : dict
        Host star flux in each output photometric filter, e.g.
        {"F770W": xxx, "F1550C": xxx}.
    grid_wvs, grid_spec : arrays
        Companion spectrum (spectral shape), in micron / W.m^-2.um^-1,
        normalized once and for all in the photometric filter `norm_band`
        (this normalization is what makes `flux_error_combined` meaningful
        in MJy).
    norm_band : str
        Name of the (SVO) photometric filter in which grid_spec was
        normalized, e.g. "F770W". Does NOT depend on list_bands (MRS channels).
    list_bands : list
        MIRI MRS spectral bands to process (e.g. "1A", "2B", ...), only used
        to locate the fm_outputs files, not for the photometric
        renormalization.
    """
    results = {}

    # Renormalization: identical for all MRS bands, computed only once.
    fnu_norm = synthetic_fnu(grid_wvs, grid_spec, norm_band, facility, instrument)
    renorm_factors = {
        photfilter_name: synthetic_fnu(grid_wvs, grid_spec, photfilter_name, facility, instrument) / fnu_norm
        for photfilter_name in flux_star.keys()
    }

    for n_nodes, channel_band in zip(list_nodes, list_bands):
        fm_outputs_path = os.path.join(uncaldir, targetname, f'fm_outputs_{n_nodes}_nodes', channel_band)
        list_fm_outputs = sort_by_dither_number(os.listdir(fm_outputs_path))

        flux_dithers_array = []
        noise_dithers_array = []
        rchi2_dithers_array = []
        ras, decs = None, None

        for fm_output in list_fm_outputs:
            if not h5py.is_hdf5(os.path.join(fm_outputs_path, fm_output)):
                continue

            flux_p, flux_error, rchi2, decs, ras, rvs = read_fm_outputs(
                os.path.join(fm_outputs_path, fm_output)
            )

            flux_p = np.nan_to_num(flux_p, nan=0.0)
            flux_error = np.nan_to_num(flux_error, nan=0.0)

            flux_dithers_array.append(flux_p)
            noise_dithers_array.append(flux_error)
            rchi2_dithers_array.append(rchi2)

        if ras is None:
            raise RuntimeError(f"No valid fm_output file found for band {channel_band}.")

        ras_grid, decs_grid = np.meshgrid(ras, decs)
        flux_dithers_array = np.array(flux_dithers_array)
        noise_dithers_array = np.array(noise_dithers_array)

        flux_combined, flux_error_combined = combined_outputs(
            flux_dithers_array, noise_dithers_array, weighted=True
        )

        sep_grid = np.sqrt(ras_grid ** 2 + decs_grid ** 2)
        results[channel_band] = {}

        for photfilter_name in flux_star.keys():
            renorm_factor = renorm_factors[photfilter_name]

            flux_error_out = flux_error_combined * renorm_factor
            contrast_5sig = 5 * flux_error_out / flux_star[photfilter_name]

            sep_1d = to_plain_array(np.ravel(sep_grid))
            contrast_1d = to_plain_array(np.ravel(contrast_5sig.transpose()))

            bins = np.linspace(sep_1d.min(), sep_1d.max(), 20)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])

            contrast_med = np.full(len(bins) - 1, np.nan)

            for i in range(len(bins) - 1):
                if i < len(bins) - 2:
                    mask = (sep_1d >= bins[i]) & (sep_1d < bins[i + 1])
                else:
                    mask = (sep_1d >= bins[i]) & (sep_1d <= bins[i + 1])  # dernier bin inclusif

                # Exclude masked/invalid pixels (flux_error set to 0 by nan_to_num)
                valid = mask & (contrast_1d > 0) & np.isfinite(contrast_1d)

                if np.any(valid):
                    contrast_med[i] = np.nanmedian(contrast_1d[valid])


            results[channel_band][photfilter_name] = {
                "sep_1d": sep_1d,
                "contrast_1d": contrast_1d,
                "bin_centers": bin_centers,
                "contrast_med": contrast_med,
            }

            if plot:
                plt.figure()
                plt.title(f"5-$\\sigma$ Sensitivity 1D ({channel_band} $\\to$ {photfilter_name})")
                plt.scatter(sep_1d, contrast_1d, s=5, alpha=0.5)
                plt.plot(bin_centers, contrast_med, color='red')
                plt.yscale("log")
                plt.xlabel("Separation (as)")
                plt.ylabel(f"5-$\\sigma$ Flux ratio ({photfilter_name})")
                plt.show()

    if save_contrast_curves:
        save_contrast_curves_fits(results, targetname, output_path=contrast_output_path)

    return results