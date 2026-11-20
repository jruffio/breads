"""Tests of the 3D spline (x, y, wavelength) basis, Spline3D, and of fit_3dspline / evaluate_3dspline.
These tests do not require the [jwst] optional dependencies."""
import os

import numpy as np
import pytest
from astropy.io import fits

from breads.utils.splines import get_spline_model, get_bspline_model
from breads.jwst_tools.spline1d import Spline1D
from breads.jwst_tools.spline3d import Spline3D
from breads.jwst_tools.splines import fit_3dspline, evaluate_3dspline, radial_prior_3dspline

WV_NODES = np.linspace(4.07, 5.3, 5)
X_NODES = np.arange(-2, 2.0001, 0.02)
Y_NODES = np.arange(-2, 2.0001, 0.02)


def _old_stamp_list(x_nodes, y_nodes, stamp_size, N_overlap_nodes):
    """Return the stamps as defined by the original fit_3dspline/evaluate_3dspline code (before Spline3D)."""
    ss_x, ss_y = stamp_size
    N_x_nodes, N_y_nodes = len(x_nodes), len(y_nodes)
    x_chunk_starts_ids = np.searchsorted(x_nodes, np.arange(x_nodes[0], x_nodes[-1], ss_x))
    x_chunk_ends_ids = np.append(x_chunk_starts_ids[1:] - 1, len(x_nodes) - 1)
    y_chunk_starts_ids = np.searchsorted(y_nodes, np.arange(y_nodes[0], y_nodes[-1], ss_y))
    y_chunk_ends_ids = np.append(y_chunk_starts_ids[1:] - 1, len(y_nodes) - 1)
    x_bigchunk_starts_ids = np.clip(x_chunk_starts_ids - N_overlap_nodes, 0, N_x_nodes - 1)
    x_bigchunk_ends_ids = np.clip(x_chunk_ends_ids + N_overlap_nodes, 0, N_x_nodes - 1)
    y_bigchunk_starts_ids = np.clip(y_chunk_starts_ids - N_overlap_nodes, 0, N_y_nodes - 1)
    y_bigchunk_ends_ids = np.clip(y_chunk_ends_ids + N_overlap_nodes, 0, N_y_nodes - 1)
    stamp_list = []
    for k in range(len(x_chunk_starts_ids)):
        for l in range(len(y_chunk_starts_ids)):
            stamp_list.append((x_bigchunk_starts_ids[k], x_chunk_starts_ids[k], x_chunk_ends_ids[k], x_bigchunk_ends_ids[k],
                               y_bigchunk_starts_ids[l], y_chunk_starts_ids[l], y_chunk_ends_ids[l], y_bigchunk_ends_ids[l],
                               l % 2, k % 2))
    return stamp_list


def _old_design_matrix(x_nodes, y_nodes, wv_nodes, stamp, _x, _y, _w):
    """Compute the stamp design matrix as the original _task_fit_3dspline/_task_evaluate_3dspline code did."""
    k0, k1, k2, k3, l0, l1, l2, l3, m0, m1 = stamp
    _x_nodes = x_nodes[k0:k3 + 1]
    _y_nodes = y_nodes[l0:l3 + 1]
    M_spline_x = get_spline_model(_x_nodes, _x, spline_degree=3)
    M_spline_y = get_spline_model(_y_nodes, _y, spline_degree=3)
    M_spline_wvs = get_spline_model(wv_nodes, _w, spline_degree=3)
    M_spline_x_tiled = np.tile(M_spline_x[:, None, None, :], (1, np.size(wv_nodes), np.size(_y_nodes), 1))
    M_spline_y_tiled = np.tile(M_spline_y[:, None, :, None], (1, np.size(wv_nodes), 1, np.size(_x_nodes)))
    M_spline_wvs_tiled = np.tile(M_spline_wvs[:, :, None, None], (1, 1, np.size(_y_nodes), np.size(_x_nodes)))
    M_3dspline = M_spline_x_tiled * M_spline_y_tiled * M_spline_wvs_tiled
    return M_3dspline.reshape((M_3dspline.shape[0], -1))


def _samples_in_extended_stamp(sp, stamp, rng, n=500):
    """Draw n random samples (x, y, w) inside the extended region of a stamp."""
    k0, k1, k2, k3, l0, l1, l2, l3, m0, m1 = stamp
    x = rng.uniform(sp.x_nodes[k0], sp.x_nodes[k3], n)
    y = rng.uniform(sp.y_nodes[l0], sp.y_nodes[l3], n)
    w = rng.uniform(WV_NODES[0], WV_NODES[-1], n)
    keep = sp.stamp_extended_mask(stamp, x, y, w)
    return x[keep].astype(np.float32), y[keep].astype(np.float32), w[keep].astype(np.float32)


def test_cardinal_stamps_and_design_matrix_identical_to_old_code():
    """Check that, for the cardinal basis, Spline3D gives bit for bit the stamps and design matrix of the original
    fit_3dspline code, and the same parameter array shape (2, 2, Nwv, Ny, Nx)."""
    sp = Spline3D(WV_NODES, X_NODES, Y_NODES)
    assert sp.basis == "cardinal"
    assert sp.params_shape == (2, 2, 5, np.size(Y_NODES), np.size(X_NODES))
    stamps = sp.stamps()
    assert [tuple(int(v) for v in s) for s in stamps] == \
           [tuple(int(v) for v in s) for s in _old_stamp_list(X_NODES, Y_NODES, (0.2, 0.2), 2)]
    rng = np.random.default_rng(0)
    for stamp in [stamps[0], stamps[len(stamps) // 2], stamps[-1], stamps[123]]:
        x, y, w = _samples_in_extended_stamp(sp, stamp, rng)
        M_new = sp.design_matrix(stamp, x, y, w)
        M_old = _old_design_matrix(X_NODES, Y_NODES, WV_NODES, stamp, x, y, w)
        np.testing.assert_array_equal(M_new, M_old)
        k0, k1, k2, k3, l0, l1, l2, l3, m0, m1 = stamp
        assert sp.stamp_param_slices(stamp) == (slice(None), slice(l0, l3 + 1), slice(k0, k3 + 1))
        assert M_new.shape[1] == np.prod(sp.stamp_params_shape(stamp))


@pytest.mark.parametrize("basis", ["bspline", "bspline_nak"])
def test_bspline_local_stamp_basis(basis):
    """Check that, for the B-spline bases, the spatial basis of a stamp is built from the nodes of the stamp only, the
    wavelength basis from all the wavelength nodes, and that the stamp basis sums to 1."""
    sp = Spline3D(WV_NODES, X_NODES, Y_NODES, basis=basis)
    nak = basis == "bspline_nak"
    extra = 0 if nak else 2
    assert sp.params_shape == (2, 2, 5 + extra, np.size(Y_NODES) + extra, np.size(X_NODES) + extra)
    rng = np.random.default_rng(1)
    stamps = sp.stamps()
    for stamp in [stamps[0], stamps[len(stamps) // 2], stamps[-1], stamps[57]]:
        k0, k1, k2, k3, l0, l1, l2, l3, m0, m1 = stamp
        x, y, w = _samples_in_extended_stamp(sp, stamp, rng)
        M_x = get_bspline_model(X_NODES[k0:k3 + 1], x, not_a_knot=nak)
        M_y = get_bspline_model(Y_NODES[l0:l3 + 1], y, not_a_knot=nak)
        M_w = get_bspline_model(WV_NODES, w, not_a_knot=nak)
        M_expected = (M_w[:, :, None, None] * M_y[:, None, :, None] * M_x[:, None, None, :]).reshape(x.size, -1)
        M = sp.design_matrix(stamp, x, y, w)
        np.testing.assert_allclose(M, M_expected, rtol=1e-12, atol=1e-15)
        assert M.shape[1] == np.prod(sp.stamp_params_shape(stamp))
        assert sp.stamp_param_slices(stamp) == (slice(None), slice(l0, l0 + M_y.shape[1]), slice(k0, k0 + M_x.shape[1]))
        np.testing.assert_allclose(M.sum(axis=1), 1, rtol=1e-6)


def test_bspline_nak_stamp_fit_equals_cardinal():
    """Check that bspline_nak gives the same parameter array shape and the same fitted model of a stamp as cardinal."""
    sp_c = Spline3D(WV_NODES, X_NODES, Y_NODES)
    sp_n = Spline3D(WV_NODES, X_NODES, Y_NODES, basis="bspline_nak")
    assert sp_n.params_shape == sp_c.params_shape
    stamp = sp_c.stamps()[len(sp_c.stamps()) // 2]
    assert sp_n.stamp_param_slices(stamp) == sp_c.stamp_param_slices(stamp)
    rng = np.random.default_rng(2)
    x, y, w = _samples_in_extended_stamp(sp_c, stamp, rng, n=20000)
    x, y, w = x.astype(float), y.astype(float), w.astype(float)
    d = np.exp(-(x ** 2 + y ** 2) / (2 * 0.1 ** 2)) * (1 + 0.2 * (w - 4.5)) + 0.01 * rng.normal(size=x.size)
    models = []
    for sp in [sp_c, sp_n]:
        M = sp.design_matrix(stamp, x, y, w)
        models.append(M @ np.linalg.lstsq(M, d, rcond=None)[0])
    np.testing.assert_allclose(models[1], models[0], rtol=0, atol=1e-8)


def test_stamp_inner_regions_tile_and_are_centered():
    """Check that the inner regions of the stamps tile the field without gaps or overlaps, and extend half a node
    spacing beyond the first and last nodes of their stamp."""
    sp = Spline3D(WV_NODES, X_NODES, Y_NODES)
    stamps = sp.stamps()
    dx = X_NODES[1] - X_NODES[0]
    center = stamps[len(stamps) // 2]
    k0, k1, k2, k3, l0, l1, l2, l3, m0, m1 = center
    x_low, x_high, y_low, y_high = sp.stamp_inner_bounds(center)
    np.testing.assert_allclose([x_low, x_high], [X_NODES[k1] - dx / 2, X_NODES[k2] + dx / 2])
    np.testing.assert_allclose(x_low - X_NODES[k0], X_NODES[k3] - x_high)  # same margin on both sides
    np.testing.assert_allclose(y_low - Y_NODES[l0], Y_NODES[l3] - y_high)

    # Every sample inside the node range belongs to exactly one inner region
    rng = np.random.default_rng(3)
    x = rng.uniform(X_NODES[0], X_NODES[-1], 20000)
    y = rng.uniform(Y_NODES[0], Y_NODES[-1], 20000)
    w = np.full(x.shape, np.mean(WV_NODES))
    counts = np.sum([sp.stamp_inner_mask(stamp, x, y, w) for stamp in stamps], axis=0)
    assert np.all(counts == 1)


def test_prune_modes():
    """Check that prune_modes applies the right rule for each basis: decorrelation (default) or historical rule for
    cardinal, and for the B-splines a mode is switched off only when the data only see its tail."""
    rng = np.random.default_rng(4)
    sp_c = Spline3D(WV_NODES, X_NODES, Y_NODES)
    assert sp_c.cardinal_pruning == "decorrelation"
    stamp = sp_c.stamps()[len(sp_c.stamps()) // 2]
    x, y, w = _samples_in_extended_stamp(sp_c, stamp, rng)
    M_spline = sp_c.design_matrix(stamp, x, y, w)
    M = M_spline * rng.uniform(0.5, 2, x.size)[:, None]
    np.testing.assert_array_equal(sp_c.prune_modes(stamp, M_spline, M), Spline3D.prune_modes_decorrelation(M))
    sp_h = Spline3D(WV_NODES, X_NODES, Y_NODES, cardinal_pruning="historical")
    np.testing.assert_array_equal(sp_h.prune_modes(stamp, M_spline, M), np.nansum(M > np.nanmax(M) * 0.01, axis=0) == 0)
    assert sp_h != sp_c and sp_h.header_cards()["3DSPLPRN"][0] == "historical"
    with pytest.raises(ValueError):
        Spline3D(WV_NODES, X_NODES, Y_NODES, cardinal_pruning="unknown")

    sp_b = Spline3D(WV_NODES, X_NODES, Y_NODES, basis="bspline")
    x, y, w = _samples_in_extended_stamp(sp_b, stamp, rng)
    M_spline = sp_b.design_matrix(stamp, x, y, w)
    peaks = sp_b.stamp_mode_peaks(stamp)
    assert peaks.shape == (M_spline.shape[1],)
    np.testing.assert_array_less(np.nanmax(M_spline, axis=0), peaks * (1 + 1e-9))
    removed = sp_b.prune_modes(stamp, M_spline)
    np.testing.assert_array_equal(removed, np.nanmax(M_spline, axis=0) < 1e-3 * peaks)
    # Only edge modes of the (sparsely sampled) stamp can be switched off, never the interior ones
    removed_3d = removed.reshape(sp_b.stamp_params_shape(stamp))
    assert not np.any(removed_3d[:, 3:-3, 3:-3])
    # Without samples, everything is switched off; the threshold is stored in the header
    assert np.all(sp_b.prune_modes(stamp, M_spline[:0]))
    assert sp_b.header_cards()["3DSPLCRP"][0] == 1e-3
    assert Spline3D(WV_NODES, X_NODES, Y_NODES, basis="bspline", crop_threshold=1e-2) != sp_b


def test_prune_modes_decorrelation():
    """Check that prune_modes_decorrelation removes the redundant columns and keeps the independent ones."""
    rng = np.random.default_rng(5)
    sp_c = Spline3D(WV_NODES, X_NODES, Y_NODES)
    stamp = sp_c.stamps()[len(sp_c.stamps()) // 2]
    x, y, w = _samples_in_extended_stamp(sp_c, stamp, rng, n=30000)   # many more samples than the 980 parameters
    M = sp_c.design_matrix(stamp, x, y, w)

    # With dense samples everywhere in the extended stamp, the cardinal modes are well separated: nothing is removed
    assert not np.any(Spline3D.prune_modes_decorrelation(M))

    # Duplicated / proportional columns: only one of each set is kept
    M_dup = np.concatenate([M, 0.5 * M[:, :3]], axis=1)
    removed = Spline3D.prune_modes_decorrelation(M_dup)
    assert np.sum(removed) == 3
    assert np.all(removed[:3] | removed[-3:]) and not np.any(removed[3:-3])

    # Data only in the left half of the stamp (in x): the columns without samples are removed, and the kept columns
    # are well conditioned
    k0, k1, k2, k3 = stamp[:4]
    x_cut = sp_c.x_nodes[(k0 + k3) // 2]
    left = x < x_cut
    M_left = M[left]
    removed = Spline3D.prune_modes_decorrelation(M_left)
    assert np.all(removed[np.all(M_left == 0, axis=0)])
    kept_cols = M_left[:, ~removed] / np.linalg.norm(M_left[:, ~removed], axis=0)
    s = np.linalg.svd(kept_cols, compute_uv=False)
    assert s[0] / s[-1] < 1e3
    # Stricter tolerance removes at least as many columns
    assert np.sum(Spline3D.prune_modes_decorrelation(M_left, tol=1e-1)) >= np.sum(removed)

    # No samples: everything is switched off
    assert np.all(Spline3D.prune_modes_decorrelation(M[:0]))


def test_fits_roundtrip_and_legacy(tmp_path):
    """Check that a Spline3D saved to FITS is read back identical, that older files and BreadsPSF-like files are read
    correctly, and that a 3D file is never read as a 1D spline."""
    for sp in [Spline3D(WV_NODES, X_NODES, Y_NODES),
               Spline3D(WV_NODES, X_NODES, Y_NODES, cardinal_pruning="historical"),
               Spline3D(WV_NODES, X_NODES[::2], Y_NODES, basis="bspline", stamp_size=(0.3, 0.4), N_overlap_nodes=3),
               Spline3D(WV_NODES, X_NODES, Y_NODES, basis="bspline_nak")]:
        fn = os.path.join(tmp_path, "sp3d.fits")
        hdr = fits.Header({"WV_REF": 4.5})
        sp.update_header(hdr)
        fits.HDUList([fits.PrimaryHDU()] + sp.to_hdus() + [fits.ImageHDU(header=hdr, name="BREADS")]).writeto(fn, overwrite=True)
        with fits.open(fn) as hdul:
            assert Spline3D.from_hdulist(hdul) == sp
            assert Spline1D.from_hdulist(hdul, name="nonexistent") is None
        assert Spline3D.from_file(fn) == sp
    # legacy file: no 3DSPLBAS/3DSPLNOV/3DSPLPRN (made with the historical cardinal pruning)
    legacy_hdr = fits.Header({"3DSPLSSX": 0.2, "3DSPLSSY": 0.2, "3DSPL_NW": 5, "WV_REF": 4.5})
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(WV_NODES, name="wv_nodes"),
                         fits.ImageHDU(X_NODES, name="x_nodes"), fits.ImageHDU(Y_NODES, name="y_nodes"),
                         fits.ImageHDU(header=legacy_hdr, name="BREADS")])
    assert Spline3D.from_hdulist(hdul) == Spline3D(WV_NODES, X_NODES, Y_NODES, cardinal_pruning="historical")
    # BreadsPSF-like: keywords in the primary header
    hdul = fits.HDUList([fits.PrimaryHDU(header=legacy_hdr), fits.ImageHDU(X_NODES, name="x_nodes"),
                         fits.ImageHDU(Y_NODES, name="y_nodes"), fits.ImageHDU(WV_NODES, name="wv_nodes")])
    assert Spline3D.from_hdulist(hdul) == Spline3D(WV_NODES, X_NODES, Y_NODES, cardinal_pruning="historical")
    assert Spline3D(WV_NODES, X_NODES, Y_NODES) != Spline3D(WV_NODES, X_NODES, Y_NODES, basis="bspline")
    assert Spline3D(WV_NODES, X_NODES, Y_NODES) != Spline3D(WV_NODES, X_NODES, Y_NODES, basis="bspline_nak")
    assert Spline3D(WV_NODES, X_NODES, Y_NODES) != Spline3D(WV_NODES, X_NODES, Y_NODES, N_overlap_nodes=3)


class _FakeIFU:
    """Simulate a minimal data object for fit_3dspline: a point cloud of a smooth chromatic PSF."""

    def __init__(self, seed=0, ny=60, nx=300):
        rng = np.random.default_rng(seed)
        self.ifux = rng.uniform(-0.45, 0.45, (ny, nx))
        self.ifuy = rng.uniform(-0.45, 0.45, (ny, nx))
        self.wavelengths = rng.uniform(4.1, 4.9, (ny, nx))
        self.breads_header = {"WV_REF": 4.5}
        sx, sy = self.ifux * 4.5 / self.wavelengths, self.ifuy * 4.5 / self.wavelengths
        self.model = 100 * np.exp(-(sx ** 2 + sy ** 2) / (2 * 0.15 ** 2)) * (1 + 0.2 * (self.wavelengths - 4.5))
        self.noise = np.ones((ny, nx))
        self.data = self.model + 0.01 * rng.normal(size=(ny, nx))
        self.bad_pixels = np.ones((ny, nx))

    def get_ifu_coords(self):
        return self.ifux, self.ifuy


@pytest.mark.parametrize("basis", ["cardinal", "bspline", "bspline_nak"])
def test_fit_and_evaluate_3dspline(tmp_path, basis):
    """Check on a simulated PSF that fit_3dspline recovers the model for each basis, and that evaluate_3dspline on the
    saved file reproduces the best fit model at the same samples."""
    dataobj = _FakeIFU()
    # The data cover the whole node grid (scaled coordinates up to ~0.49 arcsec). With nodes outside of the data, the
    # cardinal parameters of the edge stamps can be huge and cancel each other (pre-existing behavior), and their float32
    # storage then limits the accuracy of the re-evaluation.
    sp = Spline3D(np.linspace(4.0, 5.0, 4), np.arange(-0.4, 0.4001, 0.05), np.arange(-0.4, 0.4001, 0.05),
                  basis=basis, stamp_size=(0.3, 0.3))
    model, _, bp, res, paras, paras_err, merr_paras = fit_3dspline(dataobj, sp, threshold=100, max_cores=1)
    assert paras.shape == sp.params_shape
    good = np.isfinite(model)
    sx, sy = dataobj.ifux * 4.5 / dataobj.wavelengths, dataobj.ifuy * 4.5 / dataobj.wavelengths
    inside_nodes = (np.abs(sx) < 0.4) & (np.abs(sy) < 0.4)
    assert np.mean(good[inside_nodes]) > 0.99  # every sample within the node grid is modeled
    assert not np.any(good[~inside_nodes])
    assert np.nanstd(res[good]) < 0.05

    fn = os.path.join(tmp_path, "fit3d.fits")
    hdr = fits.Header({"WV_REF": 4.5})
    sp.update_header(hdr)
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(paras, name="SPLINE_PARAS0"),
                  fits.ImageHDU(paras_err, name="SPLINE_PARAS0_ERR")] + sp.to_hdus() +
                 [fits.ImageHDU(header=hdr, name="BREADS")]).writeto(fn)
    model_eval, model_err = evaluate_3dspline(dataobj.ifux, dataobj.ifuy, dataobj.wavelengths, fn, max_cores=1)
    np.testing.assert_allclose(model_eval[good], model[good], rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("basis", ["cardinal", "bspline", "bspline_nak"])
def test_radial_prior_3dspline(basis):
    """Check that radial_prior_3dspline follows the azimuthal average of the data and falls back to the noise floor
    beyond the data, and that fitting with it leaves the model unchanged where the data constrain it."""
    dataobj = _FakeIFU(ny=120, nx=300)
    sx = dataobj.ifux * 4.5 / dataobj.wavelengths
    sy = dataobj.ifuy * 4.5 / dataobj.wavelengths
    dataobj.noise = 0.5 + 0.1 * np.random.default_rng(1).random(dataobj.data.shape)
    sp = Spline3D(np.linspace(4.0, 5.0, 4), np.arange(-0.6, 0.6001, 0.05), np.arange(-0.6, 0.6001, 0.05),
                  basis=basis, stamp_size=(0.3, 0.3))
    reg_mean, reg_std, info = radial_prior_3dspline(sp, sx, sy, dataobj.wavelengths, dataobj.data, dataobj.noise,
                                                    dataobj.bad_pixels, annulus_width=0.05)
    wv_pos, y_pos, x_pos = sp.param_positions
    assert reg_mean.shape == reg_std.shape == (np.size(wv_pos), np.size(y_pos), np.size(x_pos))
    assert np.all(np.isfinite(reg_mean)) and np.all(np.isfinite(reg_std))
    assert np.all(reg_std >= info["fallback_std"] * (1 - 1e-12))
    np.testing.assert_allclose(info["fallback_std"], 10 * info["noise_floor"])
    # first/last wavelength parameters are outside of the data: the window is moved inside the data
    assert info["wv_centers"][0] > 4.1 - 1e-9 and info["wv_centers"][-1] < 4.9 + 1e-9
    # profile ~ azimuthal average of the PSF at the center of the data
    iw = np.argmin(np.abs(wv_pos - 4.5))
    rr = np.sqrt(x_pos[None, :] ** 2 + y_pos[:, None] ** 2)
    core = rr < 0.03
    np.testing.assert_allclose(reg_mean[iw][core], 100 * np.exp(-rr[core] ** 2 / (2 * 0.15 ** 2)), rtol=0.15)
    # beyond the data (r > ~0.68 arcsec + one annulus): fallback prior
    far = rr > 0.8
    assert np.all(reg_mean[:, far] == 0) and np.allclose(reg_std[:, far], info["fallback_std"])
    # end planes are widened by 10% of the mean
    assert np.all(reg_std[0] >= np.abs(0.1 * reg_mean[0]) - 1e-9)

    # fit with and without the prior
    model0, _, _, res0, paras0, _, _ = fit_3dspline(dataobj, sp, threshold=100, max_cores=1)
    model1, _, _, res1, paras1, _, _ = fit_3dspline(dataobj, sp, threshold=100, max_cores=1,
                                                 reg_mean_map=reg_mean, reg_std_map=reg_std)
    good = np.isfinite(model0) & np.isfinite(model1)
    assert np.mean(good[np.isfinite(model0)]) > 0.99
    assert np.nanstd(res1[good]) < 0.05
    assert np.nanmax(np.abs(model1[good] - model0[good]) / dataobj.noise[good]) < 0.1
    assert np.nanmax(np.abs(paras1)) < 10 * np.nanmax(np.abs(dataobj.model))


@pytest.mark.parametrize("basis", ["cardinal", "bspline", "bspline_nak"])
def test_model_error_3dspline(tmp_path, basis):
    """Check that the model error given by evaluate_3dspline (SPLINE_PARAS0_MERR) matches the exact model error
    sqrt(diag(M C M^T)), with M the design matrix of the stamp and C the covariance of its parameters, and that for
    the cardinal basis it is still computed from the parameter errors."""
    from breads.fit import fitfm
    from breads.instruments import Instrument

    dataobj = _FakeIFU(ny=120, nx=300)
    sp = Spline3D(np.linspace(4.0, 5.0, 4), np.arange(-0.4, 0.4001, 0.05), np.arange(-0.4, 0.4001, 0.05),
                  basis=basis, stamp_size=(0.3, 0.3))
    model, _, bp, res, paras, paras_err, merr_paras = fit_3dspline(dataobj, sp, threshold=100, max_cores=1)
    assert merr_paras.shape == sp.params_shape
    if basis == "cardinal":
        np.testing.assert_array_equal(merr_paras, paras_err)

    fn = os.path.join(tmp_path, "fit3d.fits")
    hdr = fits.Header({"WV_REF": 4.5})
    sp.update_header(hdr)
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(paras, name="SPLINE_PARAS0"),
                  fits.ImageHDU(paras_err, name="SPLINE_PARAS0_ERR"),
                  fits.ImageHDU(merr_paras, name="SPLINE_PARAS0_MERR")] + sp.to_hdus() +
                 [fits.ImageHDU(header=hdr, name="BREADS")]).writeto(fn)
    _, model_err = evaluate_3dspline(dataobj.ifux, dataobj.ifuy, dataobj.wavelengths, fn, max_cores=1)

    # Exact model error in the inner region of the stamp containing (0, 0), same fit as _task_fit_3dspline
    sx = (dataobj.ifux * 4.5 / dataobj.wavelengths).ravel()
    sy = (dataobj.ifuy * 4.5 / dataobj.wavelengths).ravel()
    w = dataobj.wavelengths.ravel()
    stamp = [s for s in sp.stamps() if sp.stamp_inner_mask(s, np.array([0.0]), np.array([0.0]), np.array([4.5]))[0]][0]
    ext = sp.stamp_extended_mask(stamp, sx, sy, w)
    M = sp.design_matrix(stamp, sx[ext], sy[ext], w[ext])
    M[:, sp.prune_modes(stamp, M, M)] = 0
    fm_func = lambda nonlin_paras, dataobj: nonlin_paras
    _, _, linparas, linparas_err, cov = fitfm([dataobj.data.ravel()[ext], M, dataobj.noise.ravel()[ext]],
                                              Instrument(verbose=False), fm_func, {}, scale_noise=False,
                                              return_covariance=True)
    valid = np.isfinite(linparas_err)
    np.testing.assert_allclose(np.diag(cov)[valid], linparas_err[valid] ** 2, rtol=1e-8)
    assert np.all(np.isnan(cov[~valid]))
    inner = sp.stamp_inner_mask(stamp, sx[ext], sy[ext], w[ext])
    M_in = M[inner][:, valid]
    C = cov[np.ix_(valid, valid)]
    exact = np.sqrt(np.sum(np.dot(M_in, C) * M_in, axis=1))
    ratio = model_err.ravel()[ext][inner] / exact
    if basis == "cardinal":
        # unchanged behavior: model errors at the nodes interpolated with the cardinal spline (correlations ignored)
        assert 0.7 < np.median(ratio) < 2.0
    else:
        # the fitted error map follows the exact error on the scale of the nodes (it smooths the variations between
        # the nodes); the former formula (M @ paras_err) overestimated it by more than 10x for the B-spline bases
        assert 0.7 < np.median(ratio) < 1.3
        assert np.percentile(ratio, 1) > 0.25 and np.percentile(ratio, 99) < 2.5
