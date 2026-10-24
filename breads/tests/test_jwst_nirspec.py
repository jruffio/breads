import pytest
import astropy.io.fits as fits
import os
import numpy as np
from glob import glob

##########################################
# Tests of JWST data reduction and analyses -- for NIRSpec
#
# This exercises end-to-end the reduction and analysis of JWST NIRSpec IFU data in an automated way, using
# HIP 65426 b (program 8063, observation 5, G395H, NRS2, 4 dithers). It is a streamlined version of the HIP 65426 b
# tutorial (docs/source/tutorials/jwst/tutorial_HIP65426b.ipynb): no 1/f noise cleaning and no centroid
# recalibration, which is why the planet search region below is wide.
#
# This is a pretty slow test, and therefore is marked to be skipped by default.
# Run it by explicitly invoking tests marked slow:
#    > pytest -m slow breads/tests/test_jwst_nirspec.py
#
# The tests depend on each other and run in order: each step uses the outputs of the previous ones.
# It's also a bit of a disk space hog, and will take up > 1 GB of output files to run it.
# These are not cleaned up automatically (yet).
#
# The analysis steps (stellar spectrum, S/N grid, spectral extraction) also need the BREADS_DATA environment
# variable, to find the model grid, the photometric filter, and the default BreadsPSF.
#
#############################################

# Skip this entire file if 'jwst' is not installed
import pytest
jwst = pytest.importorskip("jwst")

from multiprocess import Pool

import shared_test_infrastructure
from breads.jwst_reduction.reduction_utils import run_stage1_nirspec, run_stage2_nirspec
from breads.jwst_reduction.reduction_utils import compute_normalized_stellar_spectrum
from breads.jwst_reduction.reduction_utils import compute_snr_grid, cube_extraction, extract_spectrum
import breads.jwst_tools.default_nirspec as default

TEST_INPUTS_NIRSPEC = ['jw08063005001_02101_00001_nrs2_uncal.fits',
                       'jw08063005001_02101_00002_nrs2_uncal.fits',
                       'jw08063005001_02101_00003_nrs2_uncal.fits',
                       'jw08063005001_02101_00004_nrs2_uncal.fits']

SIMBAD_NAME = "HIP 65426"
TARGETNAME = "HIP_65426"
GRATING = "G395H"
DETECTOR = "nrs2"
STAR_RV = 3.1  # km/s
# Planet astrometry (see http://whereistheplanet.com/)
RA_PLANET = 418.245  # mas
DEC_PLANET = -694.532  # mas
RA_DEC_POINT_SOURCES = [[RA_PLANET, DEC_PLANET], ]
# Grid of positions for the S/N map and the cube, centered on the planet (arcsec)
RA_VEC = np.arange(-0.2, 0.2, 0.1) + RA_PLANET / 1000.
DEC_VEC = np.arange(-0.2, 0.2, 0.1) + DEC_PLANET / 1000.
SNR_SUFFIX = "snr_map_quick"
CUBE_SUFFIX = "1dspline_quick"
MIN_PEAK_SNR = 10


@pytest.fixture(scope="module")
def test_output_dir():
    return shared_test_infrastructure.get_test_output_dir(instrument="nirspec")

@pytest.fixture(scope="module")
def uncal_dir(test_output_dir):
    _uncal_dir = os.path.join(test_output_dir, "uncal")
    os.makedirs(_uncal_dir, exist_ok=True)
    return _uncal_dir

@pytest.fixture(scope="module")
def utils_dir(test_output_dir):
    _utils_dir = os.path.join(test_output_dir, "utils")
    os.makedirs(_utils_dir, exist_ok=True)
    return _utils_dir

@pytest.fixture(scope="module")
def out_dir(test_output_dir):
    _out_dir = os.path.join(test_output_dir, "out")
    os.makedirs(_out_dir, exist_ok=True)
    return _out_dir

@pytest.fixture(scope="module")
def breads_data():
    BREADS_DATA_ENV = os.getenv('BREADS_DATA')
    if BREADS_DATA_ENV is None:
        pytest.skip("BREADS_DATA must be defined to find the model grid, photometric filter, and BreadsPSF files.")
    return BREADS_DATA_ENV

@pytest.fixture(scope="module")
def mypool():
    _pool = Pool(processes=min(8, os.cpu_count()))
    yield _pool
    _pool.close()
    _pool.join()

def get_cal_files(test_output_dir):
    cal_files = sorted(glob(os.path.join(test_output_dir, "stage2", "*_cal.fits")))
    assert len(cal_files) == len(TEST_INPUTS_NIRSPEC)
    return cal_files

def get_combined_contnorm_spec_filename(cal_files, utils_dir):
    splitbasename = os.path.basename(cal_files[0]).split("_")
    return os.path.join(utils_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + DETECTOR + "_starspec_contnorm_combined_1dspline.fits")


@pytest.mark.slow   # by default do not run this
def test_check_prior_test_outputs_not_present(test_output_dir):
    return  shared_test_infrastructure.check_prior_test_outputs_not_present(test_output_dir)


@pytest.mark.slow   # by default do not run this
def test_download_from_mast(uncal_dir):
    """ Test we can download the files; this also obtains the input data for subsequent tests.
    """
    return shared_test_infrastructure.download_from_mast(TEST_INPUTS_NIRSPEC, uncal_dir)

@pytest.mark.slow   # by default do not run this
def test_run_stage1(test_output_dir, uncal_dir):
    """ Test run_stage1_nirspec of the reduction pipeline on the uncal files."""
    uncal_files = [os.path.join(uncal_dir, filename) for filename in TEST_INPUTS_NIRSPEC]

    stage1_outdir = os.path.join(test_output_dir, "stage1")

    rate_files = run_stage1_nirspec(uncal_files, stage1_outdir, overwrite=False, maximum_cores="1")

    assert len(rate_files) == len(TEST_INPUTS_NIRSPEC)
    for rate_file in rate_files:
        # Verify the output rate file exists and is a valid JWST data model
        datamodel = jwst.datamodels.open(rate_file)
        assert isinstance(datamodel, jwst.datamodels.JwstDataModel)
        assert datamodel.data.shape[0] > 0
        print(rate_file + " OK!")

@pytest.mark.slow   # by default do not run this
def test_run_stage2(test_output_dir):
    """ Test run_stage2_nirspec of the reduction pipeline on the rate files (no 1/f noise cleaning)."""

    rate_files = sorted(glob(os.path.join(test_output_dir, "stage1", "*_rate.fits")))

    stage2_outdir = os.path.join(test_output_dir,"stage2")

    cal_files = run_stage2_nirspec(rate_files, stage2_outdir, skip_cubes=True, overwrite=False,
                                   cleanflicker_skip=True)

    assert len(cal_files) == len(TEST_INPUTS_NIRSPEC)
    for cal_file in cal_files:
        # Verify the output cal file exists and is a valid JWST data model
        datamodel = jwst.datamodels.open(cal_file)
        assert isinstance(datamodel, jwst.datamodels.JwstDataModel)
        assert datamodel.data.shape[0] > 0
        print(cal_file + " OK!")

@pytest.mark.slow   # by default do not run this
def test_compute_normalized_stellar_spectrum(test_output_dir, utils_dir, breads_data, mypool):
    """ Test the computation of the continuum normalized stellar spectrum combining all the cal files."""
    cal_files = get_cal_files(test_output_dir)
    combined_contnorm_spec_filename = get_combined_contnorm_spec_filename(cal_files, utils_dir)

    combined_star_func = compute_normalized_stellar_spectrum(cal_files, utils_dir, combined_contnorm_spec_filename,
                                                             mppool=mypool, targetname=SIMBAD_NAME,
                                                             mask_charge_transfer_radius=None,
                                                             ra_dec_point_sources=RA_DEC_POINT_SOURCES)

    assert os.path.exists(combined_contnorm_spec_filename)
    wv_sampling = default.wv_sampling_dict[GRATING][DETECTOR]
    assert np.any(np.isfinite(combined_star_func(wv_sampling)))

@pytest.mark.slow   # by default do not run this
def test_compute_snr_grid(test_output_dir, utils_dir, out_dir, breads_data, mypool):
    """ Test the S/N grid around the planet, and check that the planet is detected (peak S/N > MIN_PEAK_SNR)."""
    cal_files = get_cal_files(test_output_dir)
    combined_contnorm_spec_filename = get_combined_contnorm_spec_filename(cal_files, utils_dir)

    photfilter_name = default.photfilter_name_dict[GRATING][DETECTOR]
    photfilter_filename = os.path.join(breads_data, "filters", "JWST_NIRCam." + photfilter_name + ".dat")
    model_grid_h5py = os.path.join(breads_data, "model_grids", "BT-Settlchris_0.5-6.0um_Teff500_1600_logg3.5_5.0_NIRSpec.hdf5")
    teff, logg, vsini = 1500, 4.5, 0.0

    compute_snr_grid(cal_files, utils_dir, os.path.join(out_dir, "snr_map"),
                     combined_contnorm_spec_filename,
                     photfilter_filename,
                     model_grid_h5py=model_grid_h5py, grid_paras=(teff, logg, vsini), N_KL=3,
                     suffix=SNR_SUFFIX, contnorm_suffix=None,
                     x_vec=RA_VEC, y_vec=DEC_VEC, rv_vec=np.array([STAR_RV]),
                     mask_charge_transfer_radius=None, ra_dec_point_sources=RA_DEC_POINT_SOURCES,
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
    snr_grid_filename = os.path.join(out_dir, "snr_map", splitbasename[0] + "_" + splitbasename[1] + "_" + DETECTOR + "_" + GRATING + "_" + SNR_SUFFIX + "_combined.fits")
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
    """ Test the cube extraction (1dspline mode) on the same grid as the S/N map, and the extraction of the planet
    spectrum at the S/N peak."""
    cal_files = get_cal_files(test_output_dir)
    combined_contnorm_spec_filename = get_combined_contnorm_spec_filename(cal_files, utils_dir)

    flux_cube, fluxerr_cube, x_grid, y_grid, wv_sampling = \
        cube_extraction(cal_files, utils_dir, out_dir, combined_contnorm_spec_filename,
                        coords_filename_filter=None,  # no centroid recalibration
                        mode="1dspline", suffix=CUBE_SUFFIX, contnorm_suffix=None,
                        x_vec=RA_VEC, y_vec=DEC_VEC, wv_sampling=None,
                        mask_charge_transfer_radius=None, ra_dec_point_sources=None,
                        aper_radius=0.15,
                        mppool=mypool,
                        overwrite=False,
                        targetname=SIMBAD_NAME,
                        use_stpsf=False,
                        use_breadspsf=True,
                        load_pickle=True,
                        save_pickle=True,
                        ifucoords=False)

    splitbasename = os.path.basename(cal_files[0]).split("_")
    cube_filename = os.path.join(out_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[3] + "_" + GRATING + "_" + CUBE_SUFFIX + "_cube.fits")
    assert os.path.exists(cube_filename)
    assert flux_cube.shape[1:] == (len(DEC_VEC), len(RA_VEC))
    assert np.any(np.isfinite(flux_cube))

    # Extract the planet spectrum at the S/N peak (the speckle annulus is not covered by this small cube)
    _, x_peak, y_peak = get_snr_peak(cal_files, out_dir)
    out_filename = os.path.join(out_dir, f"{TARGETNAME}_{DETECTOR}_{CUBE_SUFFIX}_1dspectrum.fits")
    extract_spectrum(cube_filename, out_filename, (x_peak, y_peak))

    with fits.open(out_filename) as hdulist_in:
        spectrum = hdulist_in['FLUX_MJy'].data
        err = hdulist_in['ERR_MJy'].data
    assert np.any(np.isfinite(spectrum))
    assert np.any(np.isfinite(err))
