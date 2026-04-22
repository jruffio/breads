import astropy.io.fits as fits
import os

os.environ['BREADS_DATA'] = '/stow/jruffio/data/breads_data'
if not os.path.exists(os.environ['BREADS_DATA']):
        os.mkdir(os.environ['BREADS_DATA'])
jwst_test_data_path = os.path.join(os.environ['BREADS_DATA'],"jwst_test_data/")
if not os.path.exists(jwst_test_data_path):
        os.mkdir(jwst_test_data_path)

def test_download_from_mast():
    file = 'jw03399002001_03102_00001_nrs2_uncal.fits'
    print('downloading {} -> {}'.format(file,jwst_test_data_path))
    mast_file_url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/{file}"
    hdul = fits.open(mast_file_url)
    hdul.writeto(os.path.join(jwst_test_data_path,file),overwrite=True)

    assert os.path.exists(os.path.join(jwst_test_data_path,file))