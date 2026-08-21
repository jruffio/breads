import os, glob
import astropy.utils.data
import breads.jwst_tools.mast

# Functions for shared test infrastructure for JWST reduction tests
# These are used in both the test_jwst_miri.py and test_jwst_nirspec.py files, to avoid code duplication


def get_test_output_dir(instrument="nirspec"):
    """ Create a test output directory for JWST reduction tests.

    This uses the BREADS_DATA environment variable if set,
    or the default astropy download cache location otherwise.
    """
    BREADS_DATA_ENV = os.getenv('BREADS_DATA')
    data_root = BREADS_DATA_ENV if BREADS_DATA_ENV else str(astropy.utils.data._get_download_cache_loc())
    jwst_test_data_path = os.path.join(data_root, "jwst_test_data", instrument)
    os.makedirs(jwst_test_data_path, exist_ok=True)
    print(f"The JWST test data for {instrument} will be downloaded in: {jwst_test_data_path}")
    return jwst_test_data_path

def check_prior_test_outputs_not_present(test_output_dir):
    """ Check that prior test outputs are not present in the test output directory.

    If they are present, inform the user, because the test data reductions will not run
    if prior outputs are present.
    """
    if 'nirspec' in test_output_dir:
        pattern = os.path.join(test_output_dir, 'stage2', '*_cal.fits')
    else:
        pattern = os.path.join(test_output_dir, '*', '*', 'stage2', '*_cal.fits')
    output_cal_files = glob.glob(pattern)
    if len(output_cal_files) > 0:
        raise RuntimeError(f"Prior test output files exist in {test_output_dir}. Delete before running tests, or else tests will NOT re-reduce data: \\rm -r '{test_output_dir}' ")

def download_from_mast(filenames, test_output_dir):
    """ Test we can download one file; this also obtains the input data for subsequent tests.
    """
    for test_file in filenames:
        breads.jwst_tools.mast.download_one_file(test_file, output_dir=test_output_dir, verbose=True)
        assert os.path.exists(os.path.join(test_output_dir, test_file))

