import pytest
import astropy.io.fits as fits
import os
import astropy
import numpy as np
from glob import glob

##########################################
# Tests of JWST data reduction and analyses -- for MIRI
#
# This exercises end-to-end the reduction of JWST data in an automated way.
# This is a pretty slow test, and therefore is marked to be skipped by default.
# Run it by explicitly invoking tests marked slow:
#    > pytest -m slow
#
# It's also a bit of a disk space hog, and will take up > 1 GB of output files to run it.
# These are not cleaned up automatically (yet).
#
# Tests MIRI reductions.
# This is largely identical to test_jwst_nirspec.py, intentionally
#
#############################################

# Skip this entire file if 'jwst' is not installed
import pytest
jwst = pytest.importorskip("jwst")

from multiprocess import Pool

import shared_test_infrastructure
from breads.jwst_reduction.reduction_utils import run_stage1_miri, run_stage2_miri
from breads.jwst_reduction.reduction_utils import compute_normalized_stellar_spectrum_miri
from breads.jwst_reduction.reduction_utils import compute_snr_grid_miri, cube_extraction_miri, extract_spectrum
from breads.jwst_tools.default_miri import wv_sampling_dict

@pytest.fixture(scope="module")
def test_output_dir():
    return shared_test_infrastructure.get_test_output_dir(instrument="miri")

TEST_INPUTS_MIRI = ['jw04829001001_07101_00001_mirifushort_uncal.fits','jw04829001001_07101_00002_mirifushort_uncal.fits','jw04829001001_07101_00003_mirifushort_uncal.fits','jw04829001001_07101_00004_mirifushort_uncal.fits']
SIMBAD_NAME = "HR 8799"
TARGETNAME = "HD 218396"

# Planet astrometry (see http://whereistheplanet.com/)
RA_PLANET = -253.344  # mas
DEC_PLANET = 956.347  # mas
RA_DEC_POINT_SOURCES = [[RA_PLANET, DEC_PLANET], ]
# Grid of positions for the S/N map and the cube, centered on the planet (arcsec)
RA_VEC = np.arange(-0.4, 0.4, 0.1) + RA_PLANET / 1000.
DEC_VEC = np.arange(-0.4, 0.4, 0.1) + DEC_PLANET / 1000.
SNR_SUFFIX = "snr_map_quick"
CUBE_SUFFIX = "1dspline_quick"
CUBE_SUFFIX_STPSF = "1dspline_quick_stpsf"
MAX_FLUX_DIFFERENCE_PSF = 0.25  # relative difference allowed between the BreadsPSF and STPSF extractions
MIN_PEAK_SNR = 10
STAR_RV = -12.6  # km/s
# The analysis steps use channel 1A of the MIRIFUSHORT detector (combined channel band 12A)
CHANNEL, SUB_BAND = "1", "A"
CHANNEL_BAND = CHANNEL + SUB_BAND
COMBINED_CHANNEL_BAND = "12A"
PHOTFILTER_NAME = "JWST_MIRI.F560W.dat"
MODEL_GRID_NAME = "BT-Settlchris_0.5-6.0um_Teff500_1600_logg3.5_5.0_NIRSpec.hdf5"
TEFF, LOGG, VSINI = 1200, 4.0, 0.0


@pytest.fixture(scope="module")
def utils_dir(test_output_dir):
    _utils_dir = os.path.join(test_output_dir, "utils", CHANNEL_BAND)
    os.makedirs(_utils_dir, exist_ok=True)
    return _utils_dir

@pytest.fixture(scope="module")
def out_dir(test_output_dir):
    _out_dir = os.path.join(test_output_dir, "out", CHANNEL_BAND)
    os.makedirs(_out_dir, exist_ok=True)
    return _out_dir

@pytest.fixture(scope="module")
def breads_data():
    BREADS_DATA_ENV = os.getenv('BREADS_DATA')
    if BREADS_DATA_ENV is None:
        pytest.skip("BREADS_DATA must be defined to find the model grid and the photometric filter.")
    return BREADS_DATA_ENV

@pytest.fixture(scope="module")
def mypool():
    _pool = Pool(processes=min(8, os.cpu_count()))
    yield _pool
    _pool.close()
    _pool.join()

def get_cal_files(test_output_dir):
    cal_files = sorted(glob(os.path.join(test_output_dir, TARGETNAME, COMBINED_CHANNEL_BAND, "stage2",
                                         "*_mirifushort_cal.fits")))
    assert len(cal_files) == len(TEST_INPUTS_MIRI)
    return cal_files

def get_combined_contnorm_spec_filename(cal_files, utils_dir):
    splitbasename = os.path.basename(cal_files[0]).split("_")
    return os.path.join(utils_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + CHANNEL_BAND + "_starspec_contnorm_combined_1dspline.fits")


@pytest.mark.slow   # by default do not run this
def test_check_prior_test_outputs_not_present(test_output_dir):
    return shared_test_infrastructure.check_prior_test_outputs_not_present(test_output_dir)


@pytest.mark.slow   # by default do not run this
def test_download_from_mast(test_output_dir):
    """ Test we can download one file; this also obtains the input data for subsequent tests.
    """
    return shared_test_infrastructure.download_from_mast(TEST_INPUTS_MIRI, test_output_dir)


@pytest.mark.slow   # by default do not run this
def test_run_stage1_miri(test_output_dir):
    rate_files, target_names = run_stage1_miri(test_output_dir, output_dir=str(test_output_dir),
                                               overwrite=False, maximum_cores="1")

    assert len(rate_files) > 0
    with fits.open(rate_files[0]) as hdul:
        assert hdul[1].data.shape[0] > 0


@pytest.mark.slow   # by default do not run this
def test_compute_fringe_flat(test_output_dir, monkeypatch):
    """Estimate the fringe flat of the stage 1 rate files by fitting a Fabry-Perot model column by column, in
    channel band CHANNEL_BAND only (one half of the detector) to keep the test short."""
    pytest.importorskip("BayesicFitting")
    from breads.jwst_tools.function_fitting_fringes import get_flat_parallelized

    rate_files = glob(os.path.join(test_output_dir, "*", "*", "stage1", "*_rate.fits"))
    assert len(rate_files) > 0, "No stage 1 output rate files found"
    target_name, combined_channel_band = os.path.relpath(rate_files[0], test_output_dir).split(os.sep)[:2]

    # get_flat_parallelized writes the flats to $FLAT_PATH/<combined_channel_band>/
    flat_path = os.path.join(test_output_dir, "fringe_flat")
    monkeypatch.setenv("FLAT_PATH", flat_path)
    get_flat_parallelized(test_output_dir, target_name, list_bands=[combined_channel_band], snr_thresh=-1e5,
                          fast=True, skip_4=True, overwrite=True, n_procs=min(8, os.cpu_count()),
                          channel_bands=[CHANNEL_BAND])

    flat_files = glob(os.path.join(flat_path, combined_channel_band, "*_fit_flat.fits"))
    band_rate_files = glob(os.path.join(test_output_dir, target_name, combined_channel_band, "stage1", "*_rate.fits"))
    assert len(flat_files) == len(band_rate_files)
    with fits.open(flat_files[0]) as hdul:
        flat = hdul['FLAT'].data
        assert flat.shape == fits.getdata(band_rate_files[0], 'SCI').shape
        assert hdul[0].header['BAND1'] == CHANNEL_BAND and hdul[0].header['BAND1MDL'] != 'skipped'
        assert hdul[0].header['BAND2MDL'] == 'skipped'
        # channel 1 is on the left half of the detector (columns 5-499), the other half is not fitted
        assert np.all(flat[:, 500:1010] == 1)
        fitted = flat[:, 5:500]
        fitted = fitted[np.isfinite(fitted) & (fitted != 1)]
        assert fitted.size > 0, "No column of the fringe flat was fitted"
        assert 0.8 < np.median(fitted) < 1.2


@pytest.mark.slow   # by default do not run this
def test_flat_fringing_stage1(test_output_dir, monkeypatch):
    """Divide the stage 1 rate files by the best matching fringe flat computed by test_compute_fringe_flat."""
    from breads.jwst_reduction.reduction_utils import flat_fringing_stage1

    rate_files = glob(os.path.join(test_output_dir, "*", "*", "stage1", "*_rate.fits"))
    assert len(rate_files) > 0, "No stage 1 output rate files found"
    target_name, combined_channel_band = os.path.relpath(rate_files[0], test_output_dir).split(os.sep)[:2]
    flat_path = os.path.join(test_output_dir, "fringe_flat")
    flat_files = glob(os.path.join(flat_path, combined_channel_band, "*_fit_flat.fits"))
    assert len(flat_files) > 0, "No fringe flat found, run test_compute_fringe_flat first"

    # best_flat_selection saves its diagnostic plot in the current directory
    monkeypatch.chdir(flat_path)
    flat_fringing_stage1(test_output_dir, target_name, output_dir=str(test_output_dir),
                         list_bands=[combined_channel_band], flat_path=flat_path, overwrite=True)

    band_rate_files = glob(os.path.join(test_output_dir, target_name, combined_channel_band, "stage1", "*_rate.fits"))
    for rate_file in band_rate_files:
        out_name = os.path.join(test_output_dir, target_name, combined_channel_band, "stage1_flat",
                                os.path.basename(rate_file))
        assert os.path.exists(out_name)
        with fits.open(out_name) as hdul, fits.open(rate_file) as hdul_rate:
            flat_name = hdul[0].header['S_FLAT']
            assert flat_name in [os.path.basename(f) for f in flat_files]
            flat = fits.getdata(os.path.join(flat_path, combined_channel_band, flat_name), 'FLAT').copy()
            flat[np.isnan(flat)] = 1
            np.testing.assert_allclose(hdul[1].data, hdul_rate[1].data / flat, rtol=1e-6, equal_nan=True)


@pytest.mark.slow   # by default do not run this
def test_run_stage2_miri(test_output_dir):
    # Find the rate files written by the stage 1 test, under <output_dir>/<target>/<band>/stage1
    rate_files = glob(os.path.join(test_output_dir, "*", "*", "stage1", "*_rate.fits"))
    assert len(rate_files) > 0, "No stage 1 output rate files found"
    target_name = os.path.relpath(rate_files[0], test_output_dir).split(os.sep)[0]

    cal_files = run_stage2_miri(test_output_dir, target_name, output_dir=str(test_output_dir),
                                custom_flatted=False, overwrite=False)

    assert len(cal_files) > 0
    with fits.open(cal_files[0]) as hdul:
        assert hdul[1].data.shape[0] > 0


@pytest.mark.slow   # by default do not run this
def test_compute_normalized_stellar_spectrum(test_output_dir, utils_dir, breads_data, mypool):
    """ Test the computation of the continuum normalized stellar spectrum combining all the cal files."""
    cal_files = get_cal_files(test_output_dir)
    combined_contnorm_spec_filename = get_combined_contnorm_spec_filename(cal_files, utils_dir)

    combined_star_func = compute_normalized_stellar_spectrum_miri(cal_files, CHANNEL, utils_dir,
                                                                  combined_contnorm_spec_filename,
                                                                  mppool=mypool, targetname=SIMBAD_NAME,
                                                                  ra_dec_point_sources=RA_DEC_POINT_SOURCES)

    assert os.path.exists(combined_contnorm_spec_filename)
    assert np.any(np.isfinite(combined_star_func(wv_sampling_dict[CHANNEL_BAND])))

@pytest.mark.slow   # by default do not run this
def test_compute_snr_grid(test_output_dir, utils_dir, out_dir, breads_data, mypool):
    """ Test the S/N grid around the planet, and check that the planet is detected (peak S/N > MIN_PEAK_SNR)."""
    cal_files = get_cal_files(test_output_dir)
    combined_contnorm_spec_filename = get_combined_contnorm_spec_filename(cal_files, utils_dir)

    photfilter_filename = os.path.join(breads_data, "filters", PHOTFILTER_NAME)
    model_grid_h5py = os.path.join(breads_data, "model_grids", MODEL_GRID_NAME)

    compute_snr_grid_miri(cal_files, CHANNEL, utils_dir, os.path.join(out_dir, "snr_map"),
                          combined_contnorm_spec_filename,
                          None,  # no centroid recalibration
                          model_grid_h5py, (TEFF, LOGG, VSINI),
                          photfilter_filename=photfilter_filename,
                          suffix=SNR_SUFFIX, contnorm_suffix=None,
                          x_vec=RA_VEC, y_vec=DEC_VEC, rv_vec=np.array([STAR_RV]),
                          ra_dec_point_sources=RA_DEC_POINT_SOURCES,
                          mppool=mypool,
                          overwrite=False,
                          targetname=SIMBAD_NAME,
                          use_stpsf=True, use_breadspsf=False)

    peak_snr, x_peak, y_peak = get_snr_peak(cal_files, out_dir)
    print(f"Peak S/N = {peak_snr:.1f} at (x, y) = ({x_peak:.3f}, {y_peak:.3f}) arcsec")
    assert peak_snr > MIN_PEAK_SNR, f"Planet not detected: peak S/N = {peak_snr:.1f} <= {MIN_PEAK_SNR}"

def get_snr_peak(cal_files, out_dir):
    """ Read the combined S/N grid and return the peak S/N and its (x, y) location in arcsec."""
    splitbasename = os.path.basename(cal_files[0]).split("_")
    snr_grid_filename = os.path.join(out_dir, "snr_map", splitbasename[0] + "_" + splitbasename[1] + "_" + CHANNEL_BAND + "_" + SNR_SUFFIX + "_combined.fits")
    with fits.open(snr_grid_filename) as hdulist:
        flux_arr = hdulist['flux_arr'].data
        fluxerr_arr = hdulist['fluxerr_arr'].data
        x_vec = hdulist['x_vec'].data
        y_vec = hdulist['y_vec'].data
        rv_vec = hdulist['rv_vec'].data
    rv0_id = len(rv_vec) // 2
    snr_map = flux_arr[rv0_id, :, :] / fluxerr_arr[rv0_id, :, :]
    peak_y_id, peak_x_id = np.unravel_index(np.nanargmax(snr_map), snr_map.shape)
    return np.nanmax(snr_map), x_vec[peak_x_id], y_vec[peak_y_id]

@pytest.mark.slow   # by default do not run this
def test_cube_and_spectrum_extraction(test_output_dir, utils_dir, out_dir, breads_data, mypool):
    """ Test the cube extraction (1dspline mode, STPSF) on the same grid as the S/N map, and the extraction of the
    planet spectrum at the S/N peak."""
    cal_files = get_cal_files(test_output_dir)
    combined_contnorm_spec_filename = get_combined_contnorm_spec_filename(cal_files, utils_dir)

    flux_cube, fluxerr_cube, x_grid, y_grid, wv_sampling = \
        cube_extraction_miri(cal_files, CHANNEL, SUB_BAND, utils_dir, out_dir, combined_contnorm_spec_filename,
                             None,  # no centroid recalibration
                             mode="1dspline", suffix=CUBE_SUFFIX, contnorm_suffix=None,
                             x_vec=RA_VEC, y_vec=DEC_VEC, wv_sampling=None,
                             mppool=mypool,
                             overwrite=False,
                             targetname=SIMBAD_NAME,
                             use_stpsf=True,
                             use_breadspsf=False,
                             load_pickle=True,
                             save_pickle=True,
                             ifucoords=False)

    splitbasename = os.path.basename(cal_files[0]).split("_")
    cube_filename = os.path.join(out_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[3] + "_" + CHANNEL_BAND + "_" + CUBE_SUFFIX + "_cube.fits")
    assert os.path.exists(cube_filename)
    assert flux_cube.shape[1:] == (len(DEC_VEC), len(RA_VEC))
    assert np.any(np.isfinite(flux_cube))

    # Extract the planet spectrum at the S/N peak (the speckle annulus is not covered by this small cube)
    _, x_peak, y_peak = get_snr_peak(cal_files, out_dir)
    out_filename = os.path.join(out_dir, f"{TARGETNAME.replace(' ', '_')}_{CHANNEL_BAND}_{CUBE_SUFFIX}_1dspectrum.fits")
    extract_spectrum(cube_filename, out_filename, (x_peak, y_peak))

    with fits.open(out_filename) as hdulist_in:
        spectrum = hdulist_in['FLUX_MJy'].data
        err = hdulist_in['ERR_MJy'].data
    assert np.any(np.isfinite(spectrum))
    assert np.any(np.isfinite(err))
