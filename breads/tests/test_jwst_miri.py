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

import shared_test_infrastructure
from breads.jwst_reduction.reduction_utils import run_stage1_miri, run_stage2_miri

@pytest.fixture(scope="module")
def test_output_dir():
    return shared_test_infrastructure.get_test_output_dir(instrument="miri")

TEST_INPUTS_MIRI = ['jw04829001001_07101_00001_mirifushort_uncal.fits']


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
    """Estimate the fringe flat of the stage 1 rate files by fitting a Fabry-Perot model column by column."""
    pytest.importorskip("BayesicFitting")
    from breads.jwst_tools.function_fitting_fringes import get_flat_parallelized

    rate_files = glob(os.path.join(test_output_dir, "*", "*", "stage1", "*_rate.fits"))
    assert len(rate_files) > 0, "No stage 1 output rate files found"
    target_name, combined_channel_band = os.path.relpath(rate_files[0], test_output_dir).split(os.sep)[:2]

    # get_flat_parallelized writes the flats to $FLAT_PATH/<combined_channel_band>/
    flat_path = os.path.join(test_output_dir, "fringe_flat")
    monkeypatch.setenv("FLAT_PATH", flat_path)
    get_flat_parallelized(test_output_dir, target_name, list_bands=[combined_channel_band], snr_thresh=-1e5,
                          fast=True, skip_4=True, overwrite=True, n_procs=min(8, os.cpu_count()))

    flat_files = glob(os.path.join(flat_path, combined_channel_band, "*_fit_flat.fits"))
    band_rate_files = glob(os.path.join(test_output_dir, target_name, combined_channel_band, "stage1", "*_rate.fits"))
    assert len(flat_files) == len(band_rate_files)
    with fits.open(flat_files[0]) as hdul:
        flat = hdul['FLAT'].data
        assert flat.shape == fits.getdata(band_rate_files[0], 'SCI').shape
        assert hdul[0].header['BAND2MDL'] != 'skipped'
        # columns of the right half of the detector (500-1009) are always fitted
        fitted = flat[:, 500:1010]
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

