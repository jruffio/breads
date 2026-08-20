import pytest
import astropy.io.fits as fits
import os
import astropy
from glob import glob


# Skip this entire file if 'jwst' is not installed
import pytest
jwst = pytest.importorskip("jwst")


from breads.jwst_tools.reduction_utils import run_stage1,run_stage2

@pytest.fixture(scope="module")
def shared_output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("nirspec_outputs")

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

def test_run_stage1(shared_output_dir):
    filename = os.path.join(jwst_test_data_path,test_file)
    uncal_files = [filename]

    stage1_outdir = os.path.join(shared_output_dir,"stage1")

    rate_files = run_stage1(uncal_files, stage1_outdir, overwrite=False, maximum_cores="1")

    with fits.open(rate_files[0]) as hdul:
        assert hdul[1].data.shape[0] > 0

def test_run_stage2(shared_output_dir):
    stage1_outdir = os.path.join(shared_output_dir,"stage1")
    filename = glob(os.path.join(stage1_outdir,"*_rate.fits"))[0]
    rate_files = [filename]

    stage2_outdir = os.path.join(shared_output_dir,"stage2")

    cal_files = run_stage2(rate_files, stage2_outdir, overwrite=False) #, maximum_cores="1")

    with fits.open(cal_files[0]) as hdul:
        assert hdul[1].data.shape[0] > 0

# def test_step1(shared_output_dir):
#     # writes to shared_output_dir/...
#     utils_dir = os.path.join(shared_output_dir,"utils")
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
