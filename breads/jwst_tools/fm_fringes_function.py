import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
import os
from scipy.interpolate import interp1d

from astropy.stats import sigma_clip

from scipy.ndimage import gaussian_filter

from breads.jwst_tools.PositiveEtalonModel import PositiveEtalonModel
from breads.jwst_tools.PositiveEtalonCosModel import PositiveEtalonCosModel
from breads.jwst_tools.flat_miri_utils import find_brightest_cols_two_channels, select_band_coor
from breads.jwst_tools.flat_miri_utils import beta_masking_inverse_slice
from BayesicFitting import SplinesModel
from BayesicFitting import LevenbergMarquardtFitter

from breads.jwst_tools.reduction_utils import find_files_to_process
from breads.jwst_tools.flat_miri_utils import select_band_coor

from multiprocessing import Pool


def unravel_phase(wnum, D):
    dD = np.diff(D)
    plt.plot(wnum[:-1], dD)
    plt.plot(wnum, 2/wnum)
    plt.show()
    for i, diff in enumerate(dD):
        if np.abs(diff) > 2/wnum[i] :
            print("phase jump")
            if diff > 0:
                D[i] -= 2/wnum[i]
            elif diff < 0:
                D[i] += 2/wnum[i]
    return D


def micron_to_wavenumber(wave):
    return 1e4/wave


def retrieve_data(filename):
    hdu_data = fits.open(filename)

    data = hdu_data['SCI'].data
    DQ = hdu_data['DQ'].data
    err = hdu_data['ERR'].data

    return data, DQ, err

def get_optimal_number_nodes(band):
    if band == '1A' or band == '1B' or band == '1C':
        N_continuum = 50
        N_D = 15
        N_finesse = 20
    elif band == '2A' or band == '2B' or band == '2C':
        N_continuum = 80
        N_D = 20
        N_finesse = 20
    else:
        raise NotImplementedError

    return N_continuum, N_D, N_finesse

def get_first_guess(band, fast=False):
    if band == '1A':
        if fast:
            freq = np.arange(3.35, 3.55, 0.02)/10
        else:
            freq = np.arange(3.35, 3.55, 0.0005)/10
    elif band == '2A':
        if fast:
            freq = np.arange(3.3, 3.55, 0.02)/10
        else:
            freq = np.arange(3.2, 3.8, 0.005)/10
    elif band == '1B':
        if fast:
            freq = np.arange(3.35, 3.55, 0.02)/10
        else:
            freq = np.arange(3.35, 3.55, 0.0005)/10
    elif band == '2B':
        if fast:
            freq = np.arange(3.3, 3.55, 0.02)/10
        else:
            freq = np.arange(3.2, 3.8, 0.005)/10
    elif band == '1C':
        if fast:
            freq = np.arange(3.3, 3.55, 0.02)/10
        else:
            freq = np.arange(3.2, 3.8, 0.005)/10
    elif band == '2C':
        if fast:
            freq = np.arange(3.3, 3.55, 0.02) / 10
        else:
            freq = np.arange(3.2, 3.8, 0.005) / 10
    else:
        raise NotImplementedError

    return freq

def masking_infinite(wave, data, err, col_id, even_fit_only=False):
    flat = np.zeros_like(wave[:, col_id]) + 1
    D_est = np.zeros_like(wave[:, col_id]) + np.nan
    continuum = np.zeros_like(wave[:, col_id]) + np.nan

    y_rows = np.arange(2, 1022, 1)
    where_even = np.where(y_rows % 2 == 0)[0]

    lamb = wave[2:1022, col_id]
    if even_fit_only:
        lamb = lamb[where_even]
    where_finite = np.where(np.isfinite(lamb))[0]
    lamb = lamb[where_finite]

    flux = data[2:1022, col_id]
    if even_fit_only:
        flux = flux[where_even]
    flux = flux[where_finite]

    err_c = err[2:1022, col_id]
    if even_fit_only:
        err_c = err_c[where_even]
    err_c = err_c[where_finite]

    return flat, D_est, continuum, lamb, flux, err_c

def identify_badpix(wnum, flux, err, plot=False):
    wnum_0, flux_0 = np.copy(wnum), np.copy(flux)
    N_nodes_continuum = 50
    continuum = SplinesModel(nrknots=N_nodes_continuum, xrange=wnum)
    sigma = [3, 5]

    wnum = wnum[np.isfinite(flux)]
    flux = flux[np.isfinite(flux)]

    for i in range(2):
        pars = [np.nanmean(flux)] + [0.] * (N_nodes_continuum + 1)  # pars for continuum splines
        continuum.parameters = pars  # insert initial parameters

        fitter = LevenbergMarquardtFitter(wnum, continuum)
        param = fitter.fit(flux, plot=plot)

        continuum_est = continuum.result(wnum, param)

        ratio = flux/continuum_est
        clipped = sigma_clip(ratio, sigma=sigma[i])

        where_good_pix = np.where(~clipped.mask)[0]

        wnum, flux, err = wnum[where_good_pix], flux[where_good_pix], err[where_good_pix]

    if plot:
        plt.plot(wnum_0, flux_0)
        plt.scatter(wnum, flux, color='red')
        plt.show()

    return wnum, flux, err

def identify_stellar_absorption(wnum, flux, spectrum_star):
    spectrum_star_continuum = gaussian_filter(spectrum_star, sigma=3)
    absorption_ratio = spectrum_star/spectrum_star_continuum

    plt.title("spectrum")
    plt.plot(wnum, spectrum_star)
    plt.show()

    plt.title("Spectrum continuum normalized")
    plt.plot(wnum, absorption_ratio)
    plt.show()

    plt.title("Star absorption rejection")
    plt.scatter(wnum, flux, label="full spectrum")
    wnum = wnum[(absorption_ratio<1.01) & (absorption_ratio>0.99)]
    flux = flux[(absorption_ratio<1.01) & (absorption_ratio>0.99)]
    plt.scatter(wnum, flux, label="spectrum with feature masked")
    plt.legend()
    plt.show()

    return wnum, flux

def get_model_continuum_frequency(N_nodes_continuum, N_nodes_D, wnum, flux, freq0):
    continuum = SplinesModel(nrknots=N_nodes_continuum, xrange=wnum)
    D = SplinesModel(nrknots=N_nodes_D, xrange=wnum)
    finesse = np.sqrt(0.15)
    mdl = PositiveEtalonModel(fixed={0: continuum, 1: finesse, 2: D, 3: 0})

    pars = [np.nanmean(flux)] + [0.] * (N_nodes_continuum + 1)  # pars for continuum splines
    pars += [freq0] + [0.0] * (N_nodes_D + 1)  # frequency
    mdl.parameters = pars  # insert initial parameters

    return mdl

def get_model_continuum_finesse_frequency(N_nodes_continuum, N_nodes_finesse, N_nodes_D, wnum, param_crop_continuum, param_crop_D):
    continuum = SplinesModel(nrknots=N_nodes_continuum, xrange=wnum)
    finesse = SplinesModel(nrknots=N_nodes_finesse, xrange=wnum)
    D = SplinesModel(nrknots=N_nodes_D, xrange=wnum)

    mdl = PositiveEtalonModel(fixed={0: continuum, 1: finesse, 2: D, 3: 0})
    pars = param_crop_continuum
    pars_2 = np.array(([0.35] + [0.0] * (N_nodes_finesse + 1)))
    pars_3 = param_crop_D
    pars = np.concatenate((pars, pars_2, pars_3))

    mdl.parameters = pars  # insert initial parameters

    return mdl

def get_flat_result(N_nodes_finesse, N_nodes_D, wnum):
    finesse = SplinesModel(nrknots=N_nodes_finesse, xrange=wnum)
    D = SplinesModel(nrknots=N_nodes_D, xrange=wnum)
    mdl_flat = PositiveEtalonModel(fixed={0: 1.0, 1: finesse, 2: D, 3: 0})

    return mdl_flat

def fit_FP_bayesian(data, wave, err, band, col_id, snr_thresh, plot=False, plot_model=True, spectrum=None, N_continuum=50, N_D=15, N_finesse=10, mask_star_lines=True, fast=True):

    flat, D_est, continuum, lamb, flux, err_c = masking_infinite(wave, data, err, col_id)

    SNR_mean = np.nanmedian(flux/err_c)

    if SNR_mean > snr_thresh:
        print(f"SNR good enough for flat: col id {col_id}, SNR mean {SNR_mean}")
        if plot:
            plt.title("SNR")
            plt.plot(flux/err_c)
            plt.show()

        wnum = micron_to_wavenumber(lamb)
        wnum, flux, err_c = identify_badpix(wnum, flux, err_c)

        if spectrum is not None:
            lamb_star, spectrum_star = spectrum
            f_interp = interp1d(lamb_star, spectrum_star, bounds_error=False, fill_value=1)
            spectrum_star = f_interp(1000/wnum)
            if mask_star_lines:
                wnum, flux = identify_stellar_absorption(wnum, flux, spectrum_star)

        if N_continuum is None or N_D is None or N_finesse is None:
            N_nodes_continuum, N_nodes_D, N_nodes_finesse  = get_optimal_number_nodes(band)
        else:
            N_nodes_continuum, N_nodes_finesse, N_nodes_D = N_continuum, N_finesse, N_D

        testing_frequencies = get_first_guess(band, fast=True) #testing several Etalon width for first fit guess
        print(testing_frequencies)
        chi2 = np.zeros_like(testing_frequencies)

        for k, freq in enumerate(testing_frequencies):
            print(f"Testing frequency = {freq}")
            mdl = get_model_continuum_frequency(N_nodes_continuum, N_nodes_D, wnum, flux, freq)
            fitter = LevenbergMarquardtFitter(wnum, mdl)
            param = fitter.fit(flux, plot=plot)

            param_crop_continuum = param[:N_nodes_continuum + 2]
            param_crop_D = param[N_nodes_continuum + 2:N_nodes_continuum + N_nodes_D + 4]

            get_model_continuum_finesse_frequency(N_nodes_continuum, N_nodes_finesse, N_nodes_D, wnum,
                                                  param_crop_continuum, param_crop_D)

            fitter = LevenbergMarquardtFitter(wnum, mdl)
            param = fitter.fit(flux, plot=plot)
            chi2[k] = np.sum((flux - mdl.result(wnum, param)) ** 2)

        if plot:
            plt.title("chi2")
            plt.plot(testing_frequencies, chi2)
            plt.plot(testing_frequencies, np.nansum(err_c**2)*np.ones_like(testing_frequencies))
            plt.show()

        idx = np.argmin(chi2)
        best_freq = testing_frequencies[idx]

        mdl = get_model_continuum_frequency(N_nodes_continuum, N_nodes_D, wnum, flux, best_freq)

        fitter = LevenbergMarquardtFitter(wnum, mdl)
        param = fitter.fit(flux, plot=False)

        param_crop_continuum = param[:N_nodes_continuum + 2]
        param_crop_D = param[N_nodes_continuum + 2:N_nodes_continuum + N_nodes_D + 4]

        mdl = get_model_continuum_finesse_frequency(N_nodes_continuum, N_nodes_finesse, N_nodes_D, wnum, param_crop_continuum, param_crop_D)
        fitter = LevenbergMarquardtFitter(wnum, mdl)
        param = fitter.fit(flux, plot=False)

        mdl_flat = get_flat_result(N_nodes_finesse, N_nodes_D, wnum)

        param_crop_continuum = param[:N_nodes_continuum + 2]
        param_finesse_crop = param[N_nodes_continuum + 2:]
        param_crop_D = param[N_nodes_continuum + N_nodes_finesse + 4:N_nodes_continuum + N_nodes_D + N_nodes_finesse + 6]

        if plot_model:
            plt.title("Best D")
            D = SplinesModel(nrknots=N_nodes_D, xrange=wnum)
            plt.plot(wnum, D.result(wnum, param_crop_D), label=f"# nodes: {N_nodes_finesse}")
            plt.legend()
            plt.show()

            plt.title("Best D")
            plt.plot(wnum, unravel_phase(wnum, D.result(wnum, param_crop_D)))
            plt.show()

            plt.title("Best finesse")
            F = SplinesModel(nrknots=N_nodes_finesse, xrange=wnum)
            plt.plot(wnum, F.result(wnum, param_finesse_crop), label=f"# nodes: {N_nodes_finesse}")
            plt.legend()
            plt.show()

        if plot:
            plt.plot(wnum, mdl_flat.result(wnum, param_finesse_crop))
            plt.show()

        lamb_axis = wave[:, col_id]
        where_good_snr = np.where(data[:, col_id]/err[:, col_id] > snr_thresh)[0]
        wnum_axis = micron_to_wavenumber(lamb_axis[where_good_snr])
        flat[where_good_snr] = mdl_flat.result(wnum_axis, param_finesse_crop)


        if spectrum is not None and plot:
            f = mdl_flat.result(wnum, param_finesse_crop)
            flux_flat = flux / f
            flux_continuum = gaussian_filter(flux_flat, sigma=8)
            spectrum_star_continuum = gaussian_filter(spectrum_star, sigma=8)
            plt.title("Spec comparison")
            plt.plot(wnum, flux_flat/flux_continuum)
            plt.plot(wnum, spectrum_star/spectrum_star_continuum)
            plt.show()
            flux_hf = flux_flat - gaussian_filter(flux_flat, sigma=8)
            star_hf = spectrum_star - gaussian_filter(spectrum_star, sigma=8)

            print("Correlation :", np.dot(flux_hf, star_hf)/(np.linalg.norm(flux_hf)*np.linalg.norm(star_hf)))

    else:
        print(f"SNR not good enough for flat: col id {col_id}, SNR: {SNR_mean}")

    flat[flat<0.1] = 1 #hard thresholding

    continuum_mdl = SplinesModel(nrknots=N_nodes_continuum, xrange=wnum)
    continuum[where_good_snr] = continuum_mdl.result(wnum_axis, param_crop_continuum)
    return flat, D_est, continuum


def get_flat(uncaldir, targetname, list_bands=None, bkg_sub=False, snr_thresh=20, spectrum=None, N_continuum=None, N_D=None, N_finesse=None, mask_star_lines=True, fast=False):

    crds_path = os.getenv('CUSTOM_CRDS_PATH')

    if list_bands is None:
        list_bands = ['12A', '12B', '12C', '34A', '34B', '34C']

    for band in list_bands:
        if bkg_sub:
            input_path = os.path.join(uncaldir, targetname, band, 'stage1_sub_bkg')
        else:
            input_path = os.path.join(uncaldir, targetname, band, 'stage1')
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/miri_flat', band)
        save_path_continuum = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/miri_flat_continuum', band)
        fileslist = find_files_to_process(input_path, filetype='rate.fits')

        for file in fileslist:
            first_band = band[0]+band[2]
            second_band = band[1]+band[2]
            data_science, DQ_science, err_science = retrieve_data(file)
            coor_file = select_band_coor(band, crds_path)
            wave = coor_file['LAMBDA'].data  # Loading the wavelength map in micron
            filename = os.path.basename(file)
            save_filename = os.path.join(save_path, filename.replace('rate.fits', 'fit_flat.fits'))
            save_continuum_name = os.path.join(save_path_continuum, filename.replace('rate.fits', 'fit_continuum.fits'))
            flat = np.zeros_like(data_science) + 1
            D = np.zeros_like(data_science) + np.nan
            continuum = np.zeros_like(data_science) + np.nan

            for col_id in range(5, 500):
                print(col_id)
                try:
                    flat[:, col_id], D[:, col_id], continuum[:, col_id] = fit_FP_bayesian(data_science, wave, err_science, first_band, col_id, snr_thresh, plot=False, spectrum=spectrum, N_continuum=N_continuum, N_D=N_D, N_finesse=N_finesse, mask_star_lines=mask_star_lines, fast=fast)
                except Exception as e:
                    print("Exception", e)

            for col_id in range(500, 1020):
                print(col_id)
                try:
                    flat[:, col_id], D[:, col_id], continuum[:, col_id] = fit_FP_bayesian(data_science, wave,
                                                                                                    err_science,
                                                                                                    first_band, col_id,
                                                                                                    snr_thresh,
                                                                                                    plot=False,
                                                                                                    spectrum=spectrum,
                                                                                                    N_continuum=N_continuum,
                                                                                                    N_D=N_D,
                                                                                                    N_finesse=N_finesse,
                                                                                                    mask_star_lines=mask_star_lines, fast=fast)

                except Exception as e:
                    print(e)

            fits.writeto(save_filename, flat, overwrite=True)

def get_flat_brightest_slices(uncaldir, targetname, list_bands=None, snr_thresh=20, spectrum=None, N_continuum=None, N_D=None, N_finesse=None, mask_star_lines=True):

    crds_path = os.getenv('CUSTOM_CRDS_PATH')

    if list_bands is None:
        list_bands = ['12A', '12B', '12C', '34A', '34B', '34C']

    for band in list_bands:
        input_path = os.path.join(uncaldir, targetname, band, 'stage1')
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/miri_flat', band)
        save_path_continuum = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/miri_flat_continuum', band)
        fileslist = find_files_to_process(input_path, filetype='rate.fits')

        for file in fileslist:
            first_band = band[0]+band[2]
            second_band = band[1]+band[2]
            data_science, DQ_science, err_science = retrieve_data(file)
            prim_header = fits.open(file)[0].header

            coor_file = select_band_coor(band, crds_path)
            wave = coor_file['LAMBDA'].data  # Loading the wavelength map in micron
            filename = os.path.basename(file)
            save_filename = os.path.join(save_path, filename.replace('rate.fits', 'fit_flat.fits'))
            save_continuum_name = os.path.join(save_path_continuum, filename.replace('rate.fits', 'fit_continuum.fits'))
            flat = np.zeros_like(data_science) + 1
            D = np.zeros_like(data_science) + np.nan
            continuum = np.zeros_like(data_science) + np.nan

            data_science *= beta_masking_inverse_slice(data_science, int(band[0]), band, N_slices=4)
            data_science *= beta_masking_inverse_slice(data_science, int(band[1]), band, N_slices=4)

            for col_id in range(5, 500):
                print(col_id)
                try:
                    flat[:, col_id], D[:, col_id], continuum[:, col_id] = fit_FP_bayesian(data_science, wave, err_science, first_band, col_id, snr_thresh, plot=False, spectrum=spectrum, N_continuum=N_continuum, N_D=N_D, N_finesse=N_finesse, mask_star_lines=mask_star_lines)
                except Exception as e:
                    print(e)

            for col_id in range(500, 1020):
                print(col_id)
                try:
                    flat[:, col_id], D[:, col_id], continuum[:, col_id] = fit_FP_bayesian(data_science, wave, err_science, second_band, col_id, snr_thresh, plot=False, spectrum=spectrum, N_continuum=N_continuum, N_D=N_D, N_finesse=N_finesse, mask_star_lines=mask_star_lines)
                except Exception as e:
                    print(e)

            primary_hdu = fits.PrimaryHDU()
            primary_hdu.header = prim_header
            hdu1 = fits.ImageHDU(data=flat, name='FLAT_EXTENDED')
            hdu2 = fits.ImageHDU(data=flat, name='FLAT')

            # Combiner tous les HDU dans un HDUList
            hdul = fits.HDUList([primary_hdu, hdu1, hdu2])
            hdul.writeto(save_filename, overwrite=True)

def get_flat_multiprocess(uncaldir, targetname, bands=None, snr_thresh=20): #TODO
    crds_path = os.getenv('CUSTOM_CRDS_PATH')

    if bands is None:
        bands = ['12A', '12B', '12C', '34A', '34B', '34C']

    for band in bands:
        input_path = os.path.join(uncaldir, targetname, band, 'stage1')
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/miri_flat', band)
        save_path_continuum = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/miri_flat_continuum', band)
        fileslist = find_files_to_process(input_path, filetype='corr_rate.fits')

        for file in fileslist:
            first_band = band[0] + band[2]
            second_band = band[1] + band[2]
            data_science, DQ_science, err_science = retrieve_data(file)
            coor_file = select_band_coor(band, crds_path)
            wave = coor_file['LAMBDA'].data  # Loading the wavelength map in micron
            filename = os.path.basename(file)
            save_filename = os.path.join(save_path, filename.replace('rate.fits', 'fit_flat.fits'))
            save_continuum_name = os.path.join(save_path_continuum, filename.replace('rate.fits', 'fit_continuum.fits'))
            flat = np.zeros_like(data_science) + 1
            D = np.zeros_like(data_science) + np.nan
            continuum = np.zeros_like(data_science) + np.nan

            param_multiprocess = []
            for col_id in range(94, 95): #for col_id in range(5, 500):
                param_multiprocess.append((data_science, wave, err_science, first_band, col_id, snr_thresh, None, None, None, False))
            for col_id in range(700, 701): #for col_id in range(500, 1020):
                param_multiprocess.append((data_science, wave, err_science, second_band, col_id, snr_thresh, None, None, None, False))

            with Pool() as pool:
                results = pool.starmap(fit_FP_bayesian, param_multiprocess)
            return 1

def plot_psd_func(signal1, signal2, sampling_rate):

    N = len(signal1)

    signal1[np.isnan(signal1)] = 0
    signal2[np.isnan(signal2)] = 0
    # Calcul de la FFT et du module
    fft_vals_1 = np.fft.fft(signal1)
    fft_vals_2 = np.fft.fft(signal2)

    fft_magnitude_1 = np.abs(fft_vals_1) / N  # normalisation
    fft_magnitude_2 = np.abs(fft_vals_2) / N  # normalisation

    freqs = np.fft.fftfreq(N, d=1 / sampling_rate)

    idxs = np.where(freqs >= 0)
    freqs = freqs[idxs]
    fft_magnitude_1 = fft_magnitude_1[idxs]
    fft_magnitude_2 = fft_magnitude_2[idxs]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, fft_magnitude_1)
    plt.plot(freqs, fft_magnitude_2)
    plt.title("PSD")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
