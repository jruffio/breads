import astropy.io.fits as fits
import os
import astropy

BREADS_DATA_ENV = os.getenv('BREADS_DATA')
if BREADS_DATA_ENV is None:
    jwst_test_data_path = os.path.join(str(astropy.utils.data._get_download_cache_loc()),'jwst_test_data')
else:
    jwst_test_data_path = os.path.join(os.environ['BREADS_DATA'], "jwst_test_data")
if not os.path.exists(jwst_test_data_path):
        os.mkdir(jwst_test_data_path)
print("The JWST test data will be downloaded in: {}".format(jwst_test_data_path))

test_file = 'jw03399002001_03102_00001_nrs2_uncal.fits'

def test_download_from_mast():
    print('downloading {} -> {}'.format(test_file,jwst_test_data_path))
    mast_file_url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/{test_file}"
    hdul = fits.open(mast_file_url)
    hdul.writeto(os.path.join(jwst_test_data_path,test_file),overwrite=True)

    assert os.path.exists(os.path.join(jwst_test_data_path,test_file))