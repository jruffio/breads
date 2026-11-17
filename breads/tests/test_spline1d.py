"""Tests of the 1d spline basis switch (cardinal vs B-spline) used in the JWST BREADS functions.
These tests do not require the [jwst] optional dependencies."""
import os

import numpy as np
import pytest
from astropy.io import fits

from breads.utils.splines import get_spline_model, get_bspline_model
from breads.jwst_tools.spline1d import Spline1D
from breads.jwst_tools.splines import normalize_rows

NODES = np.array([4.07, 4.1, 4.2, 4.3, 4.45, 4.6, 4.8, 5.0, 5.2, 5.3])
X = np.linspace(4.0, 5.4, 1500)


def _old_cardinal_trim(M_spline):
    """Copy of the historical inline 'remove redundant spline modes' code (before Spline1D.prune_modes)."""
    dotprod_M = np.dot(M_spline.transpose(), M_spline)
    dotprod_M_diag = np.sqrt(np.diag(dotprod_M))
    dotprod_M = np.abs(dotprod_M / (dotprod_M_diag[np.newaxis, :] * dotprod_M_diag[:, np.newaxis]))
    d1 = np.diag(dotprod_M, k=1)
    is_one = d1 > (1 - 5e-3)
    not_one_idx = np.flatnonzero(~is_one)
    if not_one_idx.size == 0:
        last_lead, first_trail = len(d1) - 1, 0
    else:
        last_lead = not_one_idx[0] - 1 if is_one[0] else -1
        first_trail = not_one_idx[-1] + 1 if is_one[-1] else M_spline.shape[1] - 1
    M_spline[:, 0:last_lead + 1] = 0
    M_spline[:, first_trail + 1:M_spline.shape[1]] = 0
    return M_spline


def test_cardinal_identical_to_get_spline_model():
    """Backward compatibility: Spline1D with the default cardinal basis reproduces the legacy get_spline_model() bit for bit, and a plain node array is coerced to it."""
    sp = Spline1D(NODES)
    assert sp.basis == "cardinal"
    np.testing.assert_array_equal(sp.get_spline_model(X), get_spline_model(NODES, X, spline_degree=3))
    assert sp.n_params == np.size(NODES)
    np.testing.assert_array_equal(sp.param_positions, NODES)
    # plain arrays are coerced to the cardinal basis
    assert Spline1D.coerce(NODES) == sp
    assert Spline1D.coerce(sp) is sp


def test_cardinal_segmented_identical():
    """Backward compatibility for discontinuous (segmented) nodes: the cardinal Spline1D matches the legacy get_spline_model() for a list of node arrays."""
    segs = [NODES[:5], NODES[5:]]
    sp = Spline1D(segs)
    np.testing.assert_array_equal(sp.get_spline_model(X), get_spline_model(segs, X, spline_degree=3))
    assert sp.n_params == np.size(NODES)


def test_bspline_basis_properties():
    """Mathematical correctness of the B-spline basis: N_nodes+2 modes, partition of unity inside the nodes, zero outside, and exact reproduction of constants and linear functions via the Greville abscissae."""
    sp = Spline1D(NODES, basis="b-spline")
    assert sp.basis == "bspline"
    M = sp.get_spline_model(X)
    assert M.shape == (np.size(X), np.size(NODES) + 2)
    assert sp.n_params == M.shape[1]
    np.testing.assert_array_equal(M, get_bspline_model(NODES, X, spline_degree=3))
    inb = (X > NODES[0]) & (X < NODES[-1])
    np.testing.assert_allclose(M[inb].sum(axis=1), 1, atol=1e-12)  # partition of unity
    assert np.all(M[~inb] == 0)
    # A constant is fit exactly with constant control points
    np.testing.assert_allclose(M[inb] @ np.full(sp.n_params, 3.7), 3.7, rtol=1e-12)
    # Greville abscissae
    pos = sp.param_positions
    assert np.size(pos) == sp.n_params
    assert np.all(np.diff(pos) > 0)
    assert pos[0] == NODES[0] and pos[-1] == NODES[-1]
    # Linear functions are reproduced exactly with control points = f(Greville)
    np.testing.assert_allclose(M[inb] @ (2 * pos + 1), 2 * X[inb] + 1, rtol=1e-12)


def test_bspline_segmented_and_empty_sampling():
    """Parameter count of a segmented B-spline (N_nodes+2 per segment), and that sampling entirely outside the nodes returns an all-zero matrix instead of crashing."""
    segs = [NODES[:5], NODES[5:]]
    sp = Spline1D(segs, basis="bspline")
    assert sp.n_params == 5 + 2 + 5 + 2
    assert sp.get_spline_model(X).shape[1] == sp.n_params
    # no sample within the nodes
    M = get_bspline_model(NODES, np.array([1., 2.]), spline_degree=3)
    assert M.shape == (2, np.size(NODES) + 2) and np.all(M == 0)


def test_prune_modes_cardinal_matches_old_code():
    """Regression test: Spline1D.prune_modes() for the cardinal basis removes exactly the same redundant edge modes as the historical inline code, on random partial wavelength coverages."""
    rng = np.random.default_rng(1)
    sp = Spline1D(NODES)
    for _ in range(20):
        lo, hi = np.sort(rng.uniform(4.0, 5.4, 2))
        x = np.sort(rng.uniform(lo, hi, 300))
        M = sp.get_spline_model(x)
        np.testing.assert_array_equal(sp.prune_modes(M.copy()), _old_cardinal_trim(M.copy()))
        _, mask = sp.prune_modes(M.copy(), return_mask=True)
        assert np.all(_old_cardinal_trim(M.copy())[:, mask] == 0)


def test_prune_modes_bspline():
    """B-spline mode pruning: nothing is removed under full coverage, a barely sampled edge mode is removed when coverage stops just inside the last node interval, and crop_threshold controls this."""
    sp = Spline1D(NODES, basis="bspline", crop_threshold=1e-3)
    # Full coverage: no mode removed
    M = sp.get_spline_model(X)
    _, mask = sp.prune_modes(M.copy(), return_mask=True)
    assert not np.any(mask)
    # Coverage stops just inside the last node interval: the last mode is barely sampled (~1e-9) and removed,
    # while the second to last mode is still well sampled and kept.
    x = np.linspace(4.0, NODES[-2] + 0.001 * (NODES[-1] - NODES[-2]), 1000)
    M = sp.get_spline_model(x)
    M_pruned, mask = sp.prune_modes(M.copy(), return_mask=True)
    assert mask[-1]
    assert not np.any(mask[:-1])
    assert np.all(M_pruned[:, mask] == 0)
    # The crop threshold is a parameter
    sp0 = Spline1D(NODES, basis="bspline", crop_threshold=0)
    # With crop_threshold=0, the barely sampled last mode is kept as well
    assert not np.any(sp0.prune_modes(M.copy(), return_mask=True)[1])


def test_fits_roundtrip_and_legacy(tmp_path):
    """FITS I/O used by the utils_dir caching and consistency checks: to_hdu()/from_hdulist() round-trip nodes, basis and crop_threshold, and legacy files (no SPLBASIS, or an x_nodes HDU) are read as cardinal."""
    for sp in [Spline1D(NODES), Spline1D(NODES, basis="bspline", crop_threshold=2e-3),
               Spline1D([NODES[:5], NODES[5:]], basis="bspline")]:
        fn = os.path.join(tmp_path, "sp.fits")
        fits.HDUList([fits.PrimaryHDU(), sp.to_hdu()]).writeto(fn, overwrite=True)
        with fits.open(fn) as hdul:
            assert hdul["wv_nodes"].header["SPLBASIS"] == sp.basis
            assert hdul["wv_nodes"].header["SPLNPAR"] == sp.n_params
            sp2 = Spline1D.from_hdulist(hdul)
        assert sp2 == sp
    # legacy file without SPLBASIS -> cardinal
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(NODES, name="wv_nodes")])
    assert Spline1D.from_hdulist(hdul) == Spline1D(NODES)
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(NODES, name="x_nodes")])
    assert Spline1D.from_hdulist(hdul) == Spline1D(NODES)
    assert Spline1D.from_hdulist(fits.HDUList([fits.PrimaryHDU()])) is None
    assert Spline1D(NODES) != Spline1D(NODES, basis="bspline")


def _synthetic_image(nrows=6, ncols=400, seed=0):
    rng = np.random.default_rng(seed)
    wvs = np.tile(np.linspace(4.05, 5.32, ncols)[None, :], (nrows, 1)) + 0.002 * np.arange(nrows)[:, None]
    cont = 100 * (1 + 0.3 * np.sin(3 * wvs)) * (1 + 0.1 * np.arange(nrows)[:, None])
    noise = np.ones_like(cont)
    im = cont + rng.normal(size=cont.shape) * noise
    return im, wvs, noise, cont


@pytest.mark.parametrize("basis", ["cardinal", "bspline"])
def test_normalize_rows_shapes(basis):
    """End-to-end continuum fit with normalize_rows() for each basis: spline_paras has N_params columns and the fitted model recovers a smooth synthetic continuum within 2% (residuals at the noise level)."""
    im, wvs, noise, cont = _synthetic_image()
    sp = Spline1D(NODES, basis=basis)
    reg_mean = np.tile(np.nanmedian(im, axis=1)[:, None], (1, sp.n_params))
    reg_std = np.abs(reg_mean)
    model, _, badpix, res, paras = normalize_rows(im, wvs, noise=noise, wv_nodes=sp, threshold=10,
                                                  regularization=True, reg_mean_map=reg_mean, reg_std_map=reg_std)
    assert paras.shape == (im.shape[0], sp.n_params)
    # Only the pixels within the range of the nodes are modeled
    inb = np.isfinite(res) & (wvs > NODES[0]) & (wvs < NODES[-1])
    assert np.std(res[inb]) < 1.5
    np.testing.assert_allclose(model[inb], cont[inb], rtol=0.02)


def test_normalize_rows_cardinal_array_equals_spline1d():
    """Backward compatibility of normalize_rows(): passing a plain node array or the equivalent cardinal Spline1D gives identical outputs."""
    im, wvs, noise, _ = _synthetic_image()
    out_arr = normalize_rows(im, wvs, noise=noise, wv_nodes=NODES, regularization=False)
    out_sp = normalize_rows(im, wvs, noise=noise, wv_nodes=Spline1D(NODES), regularization=False)
    for a, b in zip(out_arr, out_sp):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("basis", ["cardinal", "bspline"])
def test_extract_spectrum_header_has_spline_basis(tmp_path, basis):
    """Provenance: the planet spectrum written by extract_spectrum() (shared with collaborators) states the spline basis (SPLBASIS, SPLNPAR, SPLNNODE) in its primary header. Skipped without the [jwst] extra."""
    try:
        from breads.jwst_reduction.reduction_utils import extract_spectrum
    except Exception:
        pytest.skip("extract_spectrum requires the [jwst] optional dependencies")
    sp = Spline1D(NODES, basis=basis)
    nwv, ny, nx = 50, 5, 5
    wv = np.linspace(4.1, 5.2, nwv)
    y_grid, x_grid = np.meshgrid(np.linspace(-0.2, 0.2, ny), np.linspace(-0.2, 0.2, nx), indexing="ij")
    hdr = fits.Header()
    sp.update_header(hdr)
    cube_fn = os.path.join(tmp_path, "cube.fits")
    fits.HDUList([fits.PrimaryHDU(header=hdr),
                  fits.ImageHDU(np.ones((nwv, ny, nx)), name="FLUX"),
                  fits.ImageHDU(0.1 * np.ones((nwv, ny, nx)), name="FLUXERR"),
                  fits.ImageHDU(x_grid, name="X"), fits.ImageHDU(y_grid, name="Y"),
                  fits.ImageHDU(wv, name="WAVE"), sp.to_hdu()]).writeto(cube_fn)
    out_fn = os.path.join(tmp_path, "spec.fits")
    extract_spectrum(cube_fn, out_fn, (0.0, 0.0))
    with fits.open(out_fn) as hdul:
        h0 = hdul[0].header
        assert h0["SPLBASIS"] == basis
        assert h0["SPLNPAR"] == sp.n_params
        assert h0["SPLNNODE"] == np.size(NODES)
        assert Spline1D.from_hdulist(hdul) == sp
