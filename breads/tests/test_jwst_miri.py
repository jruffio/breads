import pytest
import astropy.io.fits as fits
import os
import astropy
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
from breads.jwst_tools.reduction_utils import run_stage1_miri, run_stage2_miri

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

