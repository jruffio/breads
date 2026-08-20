import pytest
import astropy.io.fits as fits
import os
import astropy
from glob import glob


#####################
# test_jwst_miri
#
# Tests MIRI reductions.
# This is largely identical to test_jwst_nirspec.py, intentionally
#
#####################

# Skip this entire file if 'jwst' is not installed
import pytest
jwst = pytest.importorskip("jwst")


from breads.jwst_tools.reduction_utils import run_stage1_miri, run_stage2_miri

@pytest.fixture(scope="module")
def shared_output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("miri_outputs")

BREADS_DATA_ENV = os.getenv('BREADS_DATA')
if BREADS_DATA_ENV is None:
    jwst_test_data_path = os.path.join(str(astropy.utils.data._get_download_cache_loc()),'jwst_test_data', 'miri')
else:
    jwst_test_data_path = os.path.join(os.environ['BREADS_DATA'], "jwst_test_data", "miri")
os.makedirs(jwst_test_data_path, exist_ok=True)
print("The JWST test data will be downloaded in: {}".format(jwst_test_data_path))

test_file = 'jw04829001001_07101_00001_mirifushort_uncal.fits'


def test_download_from_mast():
    print('downloading {} -> {}'.format(test_file,jwst_test_data_path))
    mast_file_url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/{test_file}"
    hdul = fits.open(mast_file_url)
    hdul.writeto(os.path.join(jwst_test_data_path,test_file),overwrite=True)

    assert os.path.exists(os.path.join(jwst_test_data_path,test_file))


def test_run_stage1_miri(shared_output_dir):
    rate_files, target_names = run_stage1_miri(jwst_test_data_path, output_dir=str(shared_output_dir),
                                               overwrite=False, maximum_cores="1")

    assert len(rate_files) > 0
    with fits.open(rate_files[0]) as hdul:
        assert hdul[1].data.shape[0] > 0


def test_run_stage2_miri(shared_output_dir):
    # Find the rate files written by the stage 1 test, under <output_dir>/<target>/<band>/stage1
    rate_files = glob(os.path.join(shared_output_dir, "*", "*", "stage1", "*_rate.fits"))
    assert len(rate_files) > 0, "No stage 1 output rate files found"
    target_name = os.path.relpath(rate_files[0], shared_output_dir).split(os.sep)[0]

    cal_files = run_stage2_miri(jwst_test_data_path, target_name, output_dir=str(shared_output_dir),
                                custom_flatted=False, overwrite=False)

    assert len(cal_files) > 0
    with fits.open(cal_files[0]) as hdul:
        assert hdul[1].data.shape[0] > 0

