import os, glob
import pytest

# Skip this entire file if 'jwst' is not installed
import pytest
jwst = pytest.importorskip("jwst")


import breads.jwst_tools

from astroquery.mast import Observations

#############################################
# Tests of JWST data reduction and analyses
#
# This exercises end-to-end the reduction of JWST data in an automated way.
# This is a pretty slow test, and therefore is marked to be skipped by default.
# Run it by explicitly invoking tests marked slow:
#    > pytest -m slow
#
# It's also a bit of a disk space hog, and will take up > 1 GB of output files to run it.
# These are not cleaned up automatically (yet).
#
#############################################

# Test setup
# todo: this could be a pytest fixture, I think

TEST_ROOT_NIRSPEC = './test_reduction_nirspec'
TEST_PATH_UNCAL = os.path.join(TEST_ROOT_NIRSPEC, 'uncal')

for path in [TEST_ROOT_NIRSPEC, TEST_PATH_UNCAL]:
    if not os.path.exists(path):
        os.mkdir(path)

TEST_INPUTS_NIRSPEC = ['jw01414014001_02101_00001_nrs2_uncal.fits',]

@pytest.mark.slow   # by default do not run this
def download_inputs_for_tests(filenames, output_dir='./'):
    for fn in filenames:
        data_uri = f"mast:JWST/product/{fn}"

        # Download the file to your current working directory
        status, message, url = Observations.download_file(data_uri, local_path=output_dir)
        if status =='COMPLETE':
            print("Download complete")
        else:
            raise RuntimeError(f"Error, download unsuccessful: {status} {message}")

download_inputs_for_tests(TEST_INPUTS_NIRSPEC, TEST_PATH_UNCAL)

# test full reduction (stage1, stage2, and charge transfer cleaning)

@pytest.mark.slow   # by default do not run this
def test_nirspec_run_complete_stage1_stage2():

    TEST_PATH_UNCAL = os.path.join(TEST_ROOT_NIRSPEC, 'uncal')
    uncal_files = glob.glob(os.path.join(TEST_PATH_UNCAL, '*uncal.fits'))


    clean_cal_files = breads.jwst_tools.reduction_utils.run_complete_stage1_2_clean_reduction(input_dir=TEST_PATH_UNCAL,
                                                                                        output_root_dir=TEST_ROOT_NIRSPEC)

    # check the outputs of each iterated stage of reduction

    for stage, file_label in (('stage1', 'rate'),
                              ('stage2', 'cal'),
                              ('stage1_clean', 'rate'),
                              ('stage2_clean', 'cal')):
        print(f"Testing {stage} outputs of {file_label}.fits files:")
        expected_files = [os.path.join(TEST_ROOT_NIRSPEC, stage, os.path.basename(fn).replace('uncal', file_label)) for fn in uncal_files]
        for fn in expected_files:
            assert os.path.exists(fn)                           # the expected file should exist
            assert isinstance(jwst.datamodels.open(fn),
                              jwst.datamodels.JwstDataModel)    # and all output files should be valid files that we can open

            print(fn + " OK!")




