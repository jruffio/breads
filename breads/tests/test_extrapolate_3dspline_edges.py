"""Tests of the optional step breads.jwst_reduction.reduction_utils.extrapolate_3dspline_edges, on a small synthetic
3D spline file (needs the [jwst] optional dependencies to import reduction_utils)."""
import numpy as np
import pytest
from astropy.io import fits

pytest.importorskip("jwst")
from breads.jwst_reduction.reduction_utils import extrapolate_3dspline_edges
from breads.jwst_tools.spline3d import Spline3D


def _write_starsub_file(filename, n_wv=5, wv_range=(2.85, 4.12), x_step=0.02, basis="catmull_rom",
                        instrument="NIRSPEC", detector="NRS1"):
    """Small star subtraction file: PSF shape changing linearly with wavelength x power-law spectrum, both edge
    slices corrupted. The primary header has the instrument and detector, as in the files of compute_3dsplines."""
    wv_nodes = np.linspace(wv_range[0], wv_range[1], n_wv)
    spline3d = Spline3D(wv_nodes, np.arange(-0.1, 0.1001, x_step), np.arange(-0.1, 0.1001, 0.02),
                        basis=basis, stamp_size=(0.2, 0.2))
    rng = np.random.default_rng(0)
    shape_base = rng.uniform(0.5, 1.5, spline3d.params_shape[3:])
    shape_base = shape_base / np.sum(shape_base)
    gradient = rng.uniform(-1, 1, shape_base.shape) / shape_base.size
    gradient = gradient - np.mean(gradient)
    paras = np.zeros(spline3d.params_shape)
    for k in range(n_wv):
        paras[:, :, k] = (shape_base + 0.01 * gradient * (wv_nodes[k] - 2.85)) * 50 * wv_nodes[k] ** -2
    expected = paras.copy()
    paras[:, :, 0] = paras[:, :, 0] * rng.uniform(0.5, 1.5, shape_base.shape)
    paras[:, :, -1] = paras[:, :, -1] * rng.uniform(0.5, 1.5, shape_base.shape)

    header = fits.Header()
    header["WV_REF"] = 4.0
    spline3d.update_header(header)
    primary_header = fits.Header({"INSTRUME": instrument, "DETECTOR": detector})
    hdulist = fits.HDUList([fits.PrimaryHDU(header=primary_header)])
    hdulist.append(fits.ImageHDU(data=np.ones((4, 4)), name="IM_SUB"))
    hdulist.append(fits.ImageHDU(data=np.ones((4, 4)), name="IM"))
    hdulist.append(fits.ImageHDU(data=np.ones((4, 4)), name="STARMODEL"))
    hdulist.append(fits.ImageHDU(data=np.ones((4, 4)), name="BADPIX"))
    hdulist.append(fits.ImageHDU(data=paras.astype(np.float32), name="SPLINE_PARAS0"))
    hdulist.append(fits.ImageHDU(data=(0.01 * np.abs(paras)).astype(np.float32), name="SPLINE_PARAS0_ERR"))
    hdulist.append(fits.ImageHDU(data=(0.01 * np.abs(paras)).astype(np.float32), name="SPLINE_PARAS0_MERR"))
    for hdu in spline3d.to_hdus():
        hdulist.append(hdu)
    hdulist.append(fits.ImageHDU(header=header, name="BREADS"))
    hdulist.writeto(filename, overwrite=True)
    return paras, expected


@pytest.mark.parametrize("detector, edges, replaced",
                         [("NRS1", "inner", [-1]), ("NRS2", "inner", [0]), ("NRS1", "outer", [0]),
                          ("NRS2", ["inner", "outer"], [0, -1]), ("NRS2", "red", [-1])])
def test_extrapolate_3dspline_edges(tmp_path, detector, edges, replaced):
    filename = tmp_path / "starsub_combined_3Dspline.fits"
    paras, expected = _write_starsub_file(filename, detector=detector)
    out_filename = extrapolate_3dspline_edges(str(filename), edges=edges, edge_flux="powerlaw", save_plots=False)
    assert out_filename == str(filename).replace(".fits", "_extrapolated.fits")
    with fits.open(out_filename) as hdulist:
        new_paras = hdulist["SPLINE_PARAS0"].data.astype(float)
        assert "STARMODEL" not in hdulist and "IM_SUB" not in hdulist
        assert "IM" in hdulist and "BADPIX" in hdulist
        assert Spline3D.from_hdulist(hdulist).basis == "catmull_rom"
    n_wv = paras.shape[2]
    for k in range(n_wv):
        if k in [r % n_wv for r in replaced]:
            np.testing.assert_allclose(new_paras[:, :, k], expected[:, :, k], rtol=1e-5)
        else:
            np.testing.assert_array_equal(new_paras[:, :, k], paras[:, :, k].astype(np.float32))
    # The original file is not modified
    with fits.open(filename) as hdulist:
        assert "STARMODEL" in hdulist
        np.testing.assert_array_equal(hdulist["SPLINE_PARAS0"].data, paras.astype(np.float32))


def test_extrapolate_3dspline_edges_inner_outer_only_nirspec(tmp_path):
    filename = tmp_path / "starsub_combined_3Dspline.fits"
    _write_starsub_file(filename, instrument="MIRI", detector="MIRIFULONG")
    with pytest.raises(ValueError):
        extrapolate_3dspline_edges(str(filename), edges="inner", save_plots=False)
    out_filename = extrapolate_3dspline_edges(str(filename), edges="blue", save_plots=False)
    assert out_filename.endswith("_extrapolated.fits")


@pytest.mark.parametrize("basis", ["catmull_rom", "cardinal"])
def test_extrapolate_3dspline_edges_two_detectors(tmp_path, basis):
    filename1 = tmp_path / "nrs1_starsub_combined_3Dspline.fits"
    filename2 = tmp_path / "nrs2_starsub_combined_3Dspline.fits"
    # Same PSF shape (linear in wavelength) on both sides of the gap
    paras1, expected1 = _write_starsub_file(filename1, wv_range=(2.85, 4.12), basis=basis, detector="NRS1")
    _write_starsub_file(filename2, wv_range=(4.07, 5.30), basis=basis, detector="NRS2")
    with fits.open(filename2) as hdulist:
        original2 = hdulist["SPLINE_PARAS0"].data.copy()

    # File names as str or as pathlib.Path
    for filename, other in [(str(filename1), str(filename2)), (filename1, filename2)]:
        out_filename = extrapolate_3dspline_edges(filename, edges="inner", edge_flux="powerlaw",
                                                  other_detector_spline3d=other, joint_degree=1, save_plots=False,
                                                  overwrite=True)
        with fits.open(out_filename) as hdulist:
            new_paras = hdulist["SPLINE_PARAS0"].data.astype(float)
            assert hdulist["BREADS"].header["3DSPLEXO"] == filename2.name
            assert hdulist["BREADS"].header["3DSPLEXD"] == 1
            assert hdulist["BREADS"].header["3DSPLEXW"]  # weighted fit by default
        np.testing.assert_allclose(new_paras[:, :, -1], expected1[:, :, -1], rtol=1e-5)
        np.testing.assert_array_equal(new_paras[:, :, :-1], paras1[:, :, :-1].astype(np.float32))
    # The file of the other detector is not modified
    with fits.open(filename2) as hdulist:
        np.testing.assert_array_equal(hdulist["SPLINE_PARAS0"].data, original2)

    # Different spatial nodes are refused
    filename3 = tmp_path / "other_grid_starsub_combined_3Dspline.fits"
    _write_starsub_file(filename3, wv_range=(4.07, 5.30), x_step=0.025, basis=basis)
    with pytest.raises(ValueError):
        extrapolate_3dspline_edges(str(filename1), other_detector_spline3d=str(filename3), save_plots=False,
                                   overwrite=True)
