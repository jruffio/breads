import numpy as np
from astropy.stats import sigma_clip

def combine_spectrum(wavelengths, fluxes, errors, bin_size):
    """ Combines the spectrum by combining the flux values in each bin using a weighted mean.

    Calculates and returns the new combined flux errors.

    :param wavelengths: 1D array of wavelengths.
    :param fluxes: 1D array of fluxes.
    :param errors: 1D array of flux errors.
    :param bin_size: scalar value specifying the wavelength bin size.
    :return: tuple containing three 1D arrays: the new wavelength array, the combined flux values, and the new combined flux errors.
    """
    # Remove NaN values from the input arrays
    nan_mask = np.logical_or(np.isnan(wavelengths), np.isnan(fluxes))
    where_mask = np.where(~nan_mask)
    wavelengths = wavelengths[where_mask]
    fluxes = fluxes[where_mask]
    errors = errors[where_mask]

    # Sort the arrays by wavelength
    sort_indices = np.argsort(wavelengths)
    wavelengths = wavelengths[sort_indices]
    fluxes = fluxes[sort_indices]
    errors = errors[sort_indices]

    # Determine the number of bins
    num_bins = int((wavelengths[-1] - wavelengths[0]) / bin_size) + 1

    # Initialize arrays to store the combined flux values and errors
    combined_fluxes = np.zeros(num_bins)
    combined_errors = np.zeros(num_bins)
    new_wavelengths = np.zeros(num_bins)

    # Loop through each bin
    for i in range(num_bins):
        # Determine the wavelength range for the bin
        bin_start = wavelengths[0] + i * bin_size
        bin_end = bin_start + bin_size

        # Find the flux values and errors that fall within the bin
        bin_mask = np.logical_and(wavelengths >= bin_start, wavelengths < bin_end)
        if np.sum(bin_mask.astype(int)) == 0:
            # Store the combined flux and error in the appropriate arrays
            combined_fluxes[i] = np.nan
            combined_errors[i] = np.nan
            new_wavelengths[i] = bin_start + bin_size / 2.0
            continue
        bin_fluxes = fluxes[bin_mask]
        bin_errors = errors[bin_mask]

        # Do sigma clipping
        tmp_snr = (bin_fluxes - np.nanmedian(bin_fluxes)) / bin_errors
        where_tmp_snr_finite = np.where(np.isfinite(tmp_snr))
        if np.size(where_tmp_snr_finite[0]) == 0:
            combined_fluxes[i] = np.nan
            combined_errors[i] = np.nan
            new_wavelengths[i] = bin_start + bin_size / 2.0
            continue
        mask = np.full(tmp_snr.shape, False, dtype=bool)
        mask[where_tmp_snr_finite] = sigma_clip(tmp_snr[where_tmp_snr_finite], 3, masked=True).mask
        where_valid = np.where(~mask)
        bin_fluxes = bin_fluxes[where_valid]
        bin_errors = bin_errors[where_valid]

        # Calculate the weighted mean of the flux values and errors in the bin
        weights = 1.0 / bin_errors ** 2
        if len(weights) > 0:
            weighted_flux = np.sum(weights * bin_fluxes) / np.sum(weights)
            weighted_error = 1.0 / np.sqrt(np.sum(weights))
        else:
            weighted_flux = np.nan
            weighted_error = np.nan

        # Store the combined flux and error in the appropriate arrays
        combined_fluxes[i] = weighted_flux
        combined_errors[i] = weighted_error
        new_wavelengths[i] = bin_start + bin_size / 2.0

    return new_wavelengths, combined_fluxes, combined_errors


def combine_spectrum_1dspline(wavelengths, fluxes, errors, bin_size, oversampling=10):
    """Combine a spectrum using a 1d epline

    Parameters
    ----------
    wavelengths
    fluxes
    errors
    bin_size
    oversampling

    Returns
    -------
    hd_wvs
    splev(hd_wvs, spl)
    err_func(hd_wvs)
    spl

    """
    new_wavelengths, combined_fluxes, combined_errors = combine_spectrum(wavelengths, fluxes, errors, bin_size)
    star_func = interp1d(new_wavelengths, combined_fluxes, kind="linear", bounds_error=False, fill_value=1)
    err_func = interp1d(new_wavelengths, combined_errors, kind="linear", bounds_error=False, fill_value=1)

    tmp = (fluxes - star_func(wavelengths)) / errors
    tmp_std = np.nanstd(tmp)
    where_outliers = np.where(np.abs(tmp) > (5 * tmp_std))
    fluxes[where_outliers] = np.nan

    # Remove NaN values from the input arrays
    nan_mask = np.logical_or(np.isnan(wavelengths), np.isnan(fluxes))
    where_mask = np.where(~nan_mask)
    wavelengths = wavelengths[where_mask]
    fluxes = fluxes[where_mask]
    errors = errors[where_mask]

    # Sort the arrays by wavelength
    sort_indices = np.argsort(wavelengths)
    wavelengths = wavelengths[sort_indices]
    fluxes = fluxes[sort_indices]
    errors = errors[sort_indices]

    spl = splrep(wavelengths, fluxes, k=3, t=new_wavelengths[1:(np.size(new_wavelengths)-1)], task=-1, s=None, w=1 / errors)

    hd_wvs = np.arange(new_wavelengths[0],new_wavelengths[-1], bin_size / oversampling)
    return hd_wvs, splev(hd_wvs, spl),err_func(hd_wvs),spl