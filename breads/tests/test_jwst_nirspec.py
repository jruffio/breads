import pytest
import astropy.io.fits as fits
import os
import astropy
from glob import glob

##########################################
# Tests of JWST data reduction and analyses -- for NIRSpec
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

# Skip this entire file if 'jwst' is not installed
import pytest
jwst = pytest.importorskip("jwst")

import shared_test_infrastructure
from breads.jwst_tools.reduction_utils import run_stage1, run_stage2

@pytest.fixture(scope="module")
def test_output_dir():
    return shared_test_infrastructure.get_test_output_dir(instrument="nirspec")

# TEST_INPUTS_NIRSPEC = ['jw01414014001_02101_00001_nrs2_uncal.fits',]
TEST_INPUTS_NIRSPEC = ['jw03399002001_03102_00001_nrs2_uncal.fits']


@pytest.mark.slow   # by default do not run this
def test_check_prior_test_outputs_not_present(test_output_dir):
    return  shared_test_infrastructure.check_prior_test_outputs_not_present(test_output_dir)


@pytest.mark.slow   # by default do not run this
def test_download_from_mast(test_output_dir):
    """ Test we can download one file; this also obtains the input data for subsequent tests.
    """
    return shared_test_infrastructure.download_from_mast(TEST_INPUTS_NIRSPEC, test_output_dir)

@pytest.mark.slow   # by default do not run this
def test_run_stage1(test_output_dir):
    """ Test run_stage_1 of the reduction pipeline on one uncal file."""
    uncal_files = glob(os.path.join(test_output_dir,"*_uncal.fits"))

    stage1_outdir = os.path.join(test_output_dir, "stage1")

    rate_files = run_stage1(uncal_files, stage1_outdir, overwrite=False, maximum_cores="1")

    for rate_file in rate_files:
        # Verify the output rate file exists and is a valid JWST data model
        datamodel = jwst.datamodels.open(rate_file)
        assert isinstance(datamodel, jwst.datamodels.JwstDataModel)
        assert datamodel.data.shape[0] > 0
        print(rate_file + " OK!")

@pytest.mark.slow   # by default do not run this
def test_run_stage2(test_output_dir):
    """ Test run_stage_2 of the reduction pipeline on one rate file."""

    rate_files = glob(os.path.join(test_output_dir, "stage1", "*_rate.fits"))

    stage2_outdir = os.path.join(test_output_dir,"stage2")

    cal_files = run_stage2(rate_files, stage2_outdir, overwrite=False) #, maximum_cores="1")

    for cal_file in cal_files:
        # Verify the output cal file exists and is a valid JWST data model
        datamodel = jwst.datamodels.open(cal_file)
        assert isinstance(datamodel, jwst.datamodels.JwstDataModel)
        assert datamodel.data.shape[0] > 0
        print(cal_file + " OK!")


# def test_step1(test_output_dir):
#     # writes to test_output_dir/...
#     utils_dir = os.path.join(test_output_dir,"utils")
#
#     ###### Step 0 ######
#
#     wv_nodes = wv_nodes_dict[detector]
#
#     uncal_filename_filter = filename_filter + '_*_' + detector + '_uncal.fits'
#     cal_filename_filter = filename_filter + '_*_' + detector + '_cal.fits'
#
#     if model_charge_transfer:
#         MCT_path_append = '_MCT'
#     else:
#         MCT_path_append = ''
#
#     if JOINT:
#         joint_path_append = '_joint'
#     else:
#         joint_path_append = ''
#
#     uncal_files = find_files_to_process(raw_path, filetype=uncal_filename_filter)
#     stage1_outdir = os.path.join(data_path, grating + "_stage1")
#     stage2_outdir = os.path.join(data_path, grating + "_stage2")
#     utils_before_cleaning_dir = os.path.join(data_path, grating + "_utils_before_cleaning")
#     stage1_clean_outdir = os.path.join(data_path, grating + "_stage1_cleaned" + MCT_path_append)
#     stage2_clean_outdir = os.path.join(data_path, grating + "_stage2_cleaned" + MCT_path_append)
#     utils_dir = os.path.join(data_path, grating + "_utils" + MCT_path_append)
#
#     ###### Step 1 ######
#
#     cal_files = run_stage2(rate_files, stage2_outdir, skip_cubes=True, overwrite=False)
#
#     ###### Step 2 ######
#     mypool = Pool(processes=numthreads)
#
#     poly_p_RA, poly_p_dec = run_coordinate_recenter(cal_files, utils_before_cleaning_dir,
#                                                     init_centroid=(0, 0), wv_sampling=None, N_wvs_nodes=40,
#                                                     mask_charge_transfer_radius=mask_charge_transfer_radius,
#                                                     IWA=0.3, OWA=1.0,
#                                                     debug_init=None, debug_end=None,
#                                                     mppool=mypool,
#                                                     save_plots=True,
#                                                     overwrite=False,
#                                                     filename_suffix="_webbpsf_init",
#                                                     targetname=targetname)
#
#     poly_p_RA, poly_p_dec = run_coordinate_recenter(cal_files, utils_before_cleaning_dir,
#                                                     init_centroid=(poly_p_RA[-1], poly_p_dec[-1]), wv_sampling=None,
#                                                     N_wvs_nodes=40,
#                                                     mask_charge_transfer_radius=mask_charge_transfer_radius,
#                                                     IWA=0.3, OWA=1.0,
#                                                     debug_init=None, debug_end=None,
#                                                     mppool=mypool,
#                                                     save_plots=True,
#                                                     overwrite=False,
#                                                     filename_suffix="_webbpsf",
#                                                     targetname=targetname)
#
#     ###### Step 3 ######
#
#     if model_charge_transfer:
#         print('!NOISE CLEAN! starting in: {}'.format(stage1_clean_outdir))
#     new_rate_files = run_noise_clean(rate_files, stage2_outdir, stage1_clean_outdir,
#                                      N_nodes=40,
#                                      model_charge_transfer=model_charge_transfer, utils_dir=utils_before_cleaning_dir,
#                                      coords_offset=(poly_p_RA, poly_p_dec), overwrite=False)
#
#     ###### Step 4 ######
#
#     cleaned_cal_files = run_stage2(new_rate_files, stage2_clean_outdir, skip_cubes=True, overwrite=False)
#
#     ###### Step 5 ######
#
#     combined_star_func = compute_normalized_stellar_spectrum(cleaned_cal_files, utils_dir,
#                                                              coords_offset=(poly_p_RA, poly_p_dec),
#                                                              wv_nodes=wv_nodes,
#                                                              mask_charge_transfer_radius=mask_charge_transfer_radius,
#                                                              mppool=mypool,
#                                                              ra_dec_point_sources=None, overwrite=False,
#                                                              targetname=targetname)
#     ###### Step 6 ######
#
#     dataobj_list = compute_starlight_subtraction(cleaned_cal_files, utils_dir, combined_star_func=combined_star_func,
#                                                  coords_offset=(poly_p_RA, poly_p_dec), mppool=mypool,
#                                                  targetname=targetname, wv_nodes=wv_nodes)
#
#     ###### Step 7 ######
#
#     regwvs_combdataobj = get_combined_regwvs(dataobj_list,
#                                              mask_charge_transfer_radius=mask_charge_transfer_radius,
#                                              use_starsub=False)
#     if JOINT:
#         pass  # remain in sky coords
#         out_filename = os.path.join(PSF_path,
#                                     targetname + "_" + grating + "_" + detector + "_2d_point_cloud" + joint_path_append + ".fits")
#     else:
#         regwvs_combdataobj.set_coords2ifu()
#         out_filename = os.path.join(PSF_path, targetname + "_" + grating + "_" + detector + "_2d_point_cloud.fits")
#
#     # 2D interpolator object median wavelength
#     wv_sampling = regwvs_combdataobj.wv_sampling
#     wv0 = np.nanmedian(regwvs_combdataobj.wavelengths)
#     wv0_id = np.argmin(np.abs(wv_sampling - wv0))
#     pointcloud_interp = get_2D_point_cloud_interpolator(regwvs_combdataobj, wv0)
#     save_combined_regwvs(regwvs_combdataobj, out_filename)
#
#     ###### Step 8 ######
#
#     if JOINT:
#         dramin, dramax = -2, 2
#         ddecmin, ddecmax = -2, 2
#     else:
#         dramin, dramax = np.nanmin(regwvs_combdataobj.dra_as_array), np.nanmax(regwvs_combdataobj.dra_as_array)
#         ddecmin, ddecmax = np.nanmin(regwvs_combdataobj.ddec_as_array), np.nanmax(regwvs_combdataobj.ddec_as_array)
#     print(dramin, dramax, ddecmin, ddecmax)
#     ra_vec = np.linspace(dramin, dramax, 60)
#     dec_vec = np.linspace(ddecmin, ddecmax, 60)
#     print(ra_vec, dec_vec)
#
#     cleaned_cal_files = find_files_to_process(stage2_clean_outdir, filetype=cal_filename_filter)
#
#     splitbasename = os.path.basename(cleaned_cal_files[0]).split("_")
#     filename_suffix = "_webbpsf"
#     poly2d_centroid_filename = os.path.join(utils_before_cleaning_dir,
#                                             splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[
#                                                 3] + "_poly2d_centroid" + filename_suffix + ".txt")
#
#     if JOINT:
#         cube_filename = os.path.join(output_path, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[
#             3] + "_spectral_cube_ish" + joint_path_append + ".fits")
#     else:
#         cube_filename = os.path.join(output_path, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[
#             3] + "_spectral_cube_ish.fits")
#
#     # regwvs WITH starsub
#     regwvs_starsub_combdataobj = get_combined_regwvs(dataobj_list,
#                                                      mask_charge_transfer_radius=mask_charge_transfer_radius,
#                                                      use_starsub=True)
#     if JOINT:
#         pass  # remain in sky coords
#     else:
#         regwvs_starsub_combdataobj.set_coords2ifu()
#
#     # Fit a model PSF (WebbPSF) to the combined point cloud of dataobj_list
#     debug_init, debug_end = None, None
#     # debug_init,debug_end = 1000,1100 # min max wavelength indices for partial extraction
#     aper_radius = 0.15
#
#     webbpsf_reload = regwvs_starsub_combdataobj.reload_webbpsf_model()
#     if webbpsf_reload is None:
#         webbpsf_reload = regwvs_starsub_combdataobj.compute_webbpsf_model(
#             wv_sampling=regwvs_starsub_combdataobj.wv_sampling,
#             image_mask=None,
#             pixelscale=0.1, oversample=10,
#             parallelize=True, mppool=mypool,
#             save_utils=True)
#     wpsfs, wpsfs_header, wepsfs, webbpsf_wvs, webbpsf_X, webbpsf_Y, wpsf_oversample, wpsf_pixelscale = webbpsf_reload
#     webbpsf_X = np.tile(webbpsf_X[None, :, :], (wepsfs.shape[0], 1, 1))
#     webbpsf_Y = np.tile(webbpsf_Y[None, :, :], (wepsfs.shape[0], 1, 1))
#
#     ###### Step 9 ######
#
#     t0 = time.time()
#     flux_cube, fluxerr_cube, ra_grid, dec_grid = \
#         build_cube(regwvs_starsub_combdataobj,  # combined point cloud
#                    wepsfs, webbpsf_X, webbpsf_Y,  # webbPdSF model for flux extraction
#                    ra_vec, dec_vec,  # spatial sampling of final cube
#                    out_filename=cube_filename, linear_interp=True, mppool=mypool, aper_radius=aper_radius,
#                    debug_init=debug_init, debug_end=debug_end)
#     t1 = time.time()
#     print('build cube ran in {} seconds...'.format(np.round(t1 - t0)))
