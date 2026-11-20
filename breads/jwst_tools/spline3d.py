"""
3D spline (x, y, wavelength) model used to fit the starlight / PSF in IFU point clouds (see
breads.jwst_tools.splines.fit_3dspline) and to build BreadsPSFs (breads.jwst_tools.breadspsf).

The 3D basis is the tensor product of three 1D splines (breads.jwst_tools.spline1d.Spline1D), one per axis, all with
the same basis ("cardinal", the historical default, "bspline" or "bspline_nak", see Spline1D). The spatial coordinates are in arcsec, scaled to a
reference wavelength (x * WV_REF / wavelength), so that the PSF is approximately achromatic in these coordinates.

The full field of view is not fitted at once: it is split in stamps of size `stamp_size` (arcsec), each fitted
independently with `N_overlap_nodes` extra nodes on each side. Adjacent stamps overlap, so the parameters of each stamp
are stored in a separate copy depending on the parity of the stamp indices: the parameter array has shape
(2, 2, n_wv_params, n_y_params, n_x_params).

As in the original implementation, the spatial basis of each stamp is built from its own subset of nodes (the nodes of
the extended stamp), for every basis; the wavelength basis is the same for all stamps (built on all wv_nodes). The
parameters of a stamp are stored in its parity copy starting at the index of its first node:
- cardinal: the parameters are the values of the model at the nodes (n_nodes per axis).
- bspline_nak: B-splines spanning the same functions as the cardinal basis (n_nodes parameters per axis); the
  parameters are control points.
- bspline: B-splines with a knot at every node (n_nodes + degree - 1 parameters per axis, i.e. one extra degree of
  freedom near each edge of the stamp compared to cardinal); the parameters are control points.

Modes (columns of the model matrix of a stamp) that are not constrained by the data are switched off in the fit (see
Spline3D.prune_modes):
- cardinal, cardinal_pruning="decorrelation" (default): redundant modes, found by a greedy decorrelation of the sampled
  columns (Spline3D.prune_modes_decorrelation, the 3D generalization of the cardinal rule of Spline1D.prune_modes);
- cardinal, cardinal_pruning="historical": no sample above 1% of the maximum of the stamp model matrix (the rule of the
  original implementation; products without the 3DSPLPRN keyword were made with it);
- B-spline bases: the same rule as Spline1D.prune_modes (maximum over the fitted samples below crop_threshold times the
  peak value of the mode).
"""
import numpy as np

from breads.jwst_tools.spline1d import Spline1D, DEFAULT_BSPLINE_CROP_THRESHOLD

__all__ = ["Spline3D"]

CARDINAL_PRUNING_RULES = ("decorrelation", "historical")
DEFAULT_CARDINAL_PRUNING = "decorrelation"
DEFAULT_DECORRELATION_TOL = 1e-2

# Keywords of the BREADS header describing the 3D spline (3DSPLSSX/3DSPLSSY predate this class).
_HDR_COMMENTS = {"3DSPLBAS": "3D spline basis (cardinal, bspline, bspline_nak)",
                 "3DSPLDEG": "3D spline degree",
                 "3DSPLSSX": "3D spline stamp size in x (arcsec)",
                 "3DSPLSSY": "3D spline stamp size in y (arcsec)",
                 "3DSPLNOV": "3D spline number of overlapping nodes",
                 "3DSPLCRP": "3D bsplines: crop threshold for truncated modes",
                 "3DSPLPRN": "3D cardinal spline: mode pruning (decorrelation, historical)",
                 "3DSPL_NW": "3D spline number of wavelength nodes",
                 "3DSPL_NX": "3D spline number of x nodes",
                 "3DSPL_NY": "3D spline number of y nodes",
                 "3DSPL_DX": "3D spline median x node spacing (arcsec)",
                 "3DSPL_DY": "3D spline median y node spacing (arcsec)"}


class Spline3D:
    """
    Define a 3D spline basis (nodes, basis type and stamp layout), and build its design matrices for the fits.

    Parameters
    ----------
    wv_nodes : 1d array
        Wavelength nodes (micron).
    x_nodes, y_nodes : 1d arrays
        Spatial nodes (arcsec, in coordinates scaled to the reference wavelength).
    basis : str (optional, default "cardinal")
        "cardinal", "bspline" or "bspline_nak" (see Spline1D), used for the three axes.
    degree : int (optional, default 3)
        Degree of the splines.
    stamp_size : tuple (optional, default (0.2, 0.2))
        (x, y) size of the stamps fitted independently (arcsec).
    N_overlap_nodes : int (optional, default 2)
        Number of extra nodes on each side of a stamp used in its fit.
    crop_threshold : float (optional, default DEFAULT_BSPLINE_CROP_THRESHOLD = 1e-3)
        Only used for the B-spline bases, in prune_modes(): a mode is switched off if its maximum over the fitted samples
        is less than crop_threshold times its peak value (same rule as Spline1D).
    cardinal_pruning : str (optional, default "decorrelation")
        Only used for the cardinal basis, in prune_modes(): "decorrelation" (prune_modes_decorrelation) or "historical"
        (no sample above 1% of the maximum of the stamp model matrix).
    """

    def __init__(self, wv_nodes, x_nodes, y_nodes, basis="cardinal", degree=3, stamp_size=(0.2, 0.2),
                 N_overlap_nodes=2, crop_threshold=DEFAULT_BSPLINE_CROP_THRESHOLD, cardinal_pruning=DEFAULT_CARDINAL_PRUNING):
        if cardinal_pruning not in CARDINAL_PRUNING_RULES:
            raise ValueError(f"Unknown cardinal_pruning '{cardinal_pruning}'. Use one of {CARDINAL_PRUNING_RULES}.")
        self.cardinal_pruning = cardinal_pruning
        self.crop_threshold = float(crop_threshold)
        self.wv = Spline1D(np.asarray(wv_nodes, dtype=float), basis=basis, degree=degree, crop_threshold=crop_threshold)
        self.x = Spline1D(np.asarray(x_nodes, dtype=float), basis=basis, degree=degree, crop_threshold=crop_threshold)
        self.y = Spline1D(np.asarray(y_nodes, dtype=float), basis=basis, degree=degree, crop_threshold=crop_threshold)
        self.basis = self.wv.basis
        self.degree = int(degree)
        self.stamp_size = (float(stamp_size[0]), float(stamp_size[1]))
        self.N_overlap_nodes = int(N_overlap_nodes)
        self._local_axes = {}  # cache of the local Spline1D of the stamps (see _local_axis)

    @classmethod
    def default(cls, wavelengths, N_wv_nodes=5, delta_x_nodes=0.02, delta_y_nodes=0.02, basis="cardinal",
                stamp_size=(0.2, 0.2), N_overlap_nodes=2, crop_threshold=DEFAULT_BSPLINE_CROP_THRESHOLD,
                cardinal_pruning=DEFAULT_CARDINAL_PRUNING):
        """Create a Spline3D with N_wv_nodes evenly spaced over the wavelengths, and x, y nodes from -2 to 2 arcsec."""
        wv_nodes = np.linspace(np.nanmin(wavelengths), np.nanmax(wavelengths), N_wv_nodes, endpoint=True)
        x_nodes = np.arange(-2, 2.0001, delta_x_nodes)
        y_nodes = np.arange(-2, 2.0001, delta_y_nodes)
        return cls(wv_nodes, x_nodes, y_nodes, basis=basis, stamp_size=stamp_size, N_overlap_nodes=N_overlap_nodes,
                   crop_threshold=crop_threshold, cardinal_pruning=cardinal_pruning)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    @property
    def wv_nodes(self):
        return self.wv.nodes

    @property
    def x_nodes(self):
        return self.x.nodes

    @property
    def y_nodes(self):
        return self.y.nodes

    @property
    def is_bspline(self):
        """True for both B-spline bases (bspline and bspline_nak)."""
        return self.wv.is_bspline

    @property
    def params_shape(self):
        """Shape of the parameter array: (2, 2, n_wv_params, n_y_params, n_x_params)."""
        return (2, 2, self.wv.n_params, self.y.n_params, self.x.n_params)

    @property
    def param_positions(self):
        """(wv, y, x) positions associated with the parameters of each axis (nodes, or Greville abscissae)."""
        return self.wv.param_positions, self.y.param_positions, self.x.param_positions

    # ------------------------------------------------------------------------------------------------------------------
    # Stamps
    def stamps(self):
        """
        Split the field into the stamps that are fitted independently, and return them as a list.

        Each stamp is a tuple of node indices (k0, k1, k2, k3, l0, l1, l2, l3, m0, m1): x nodes k1..k2 are the stamp
        itself, k0..k3 the stamp extended by N_overlap_nodes on each side (same for y with l), and (m0, m1) the parity
        copy where its parameters are stored.
        """
        x_nodes, y_nodes = self.x_nodes, self.y_nodes
        N_x_nodes, N_y_nodes = len(x_nodes), len(y_nodes)
        ss_x, ss_y = self.stamp_size
        N_ov = self.N_overlap_nodes

        x_chunk_starts_ids = np.searchsorted(x_nodes, np.arange(x_nodes[0], x_nodes[-1], ss_x))
        x_chunk_ends_ids = np.append(x_chunk_starts_ids[1:] - 1, len(x_nodes) - 1)
        y_chunk_starts_ids = np.searchsorted(y_nodes, np.arange(y_nodes[0], y_nodes[-1], ss_y))
        y_chunk_ends_ids = np.append(y_chunk_starts_ids[1:] - 1, len(y_nodes) - 1)

        x_bigchunk_starts_ids = np.clip(x_chunk_starts_ids - N_ov, 0, N_x_nodes - 1)
        x_bigchunk_ends_ids = np.clip(x_chunk_ends_ids + N_ov, 0, N_x_nodes - 1)
        y_bigchunk_starts_ids = np.clip(y_chunk_starts_ids - N_ov, 0, N_y_nodes - 1)
        y_bigchunk_ends_ids = np.clip(y_chunk_ends_ids + N_ov, 0, N_y_nodes - 1)

        stamp_list = []
        for k in range(len(x_chunk_starts_ids)):
            for l in range(len(y_chunk_starts_ids)):
                stamp_list.append((x_bigchunk_starts_ids[k], x_chunk_starts_ids[k], x_chunk_ends_ids[k],
                                   x_bigchunk_ends_ids[k],
                                   y_bigchunk_starts_ids[l], y_chunk_starts_ids[l], y_chunk_ends_ids[l],
                                   y_bigchunk_ends_ids[l],
                                   l % 2, k % 2))
        return stamp_list

    def stamp_extended_mask(self, stamp, x, y, w):
        """Return a boolean mask of the samples (x, y, w) that fall inside the extended stamp, used to fit it."""
        k0, k1, k2, k3, l0, l1, l2, l3, m0, m1 = stamp
        x_nodes, y_nodes, wv_nodes = self.x_nodes, self.y_nodes, self.wv_nodes
        return (x > x_nodes[k0]) & (x < x_nodes[k3]) & \
               (y > y_nodes[l0]) & (y < y_nodes[l3]) & \
               (w > np.min(wv_nodes)) & (w < np.max(wv_nodes))

    @staticmethod
    def _inner_bounds(nodes, i1, i2):
        """
        Return the (low, high) bounds of the inner region of a stamp along one axis: half a node spacing beyond its
        first (i1) and last (i2) nodes, so that consecutive stamps tile the axis. At the ends of the axis: the first /
        last node.
        """
        low = nodes[0] if i1 == 0 else 0.5 * (nodes[i1 - 1] + nodes[i1])
        high = nodes[-1] if i2 == len(nodes) - 1 else 0.5 * (nodes[i2] + nodes[i2 + 1])
        return low, high

    def stamp_inner_bounds(self, stamp):
        """Return the bounds (x_low, x_high, y_low, y_high) of the inner region of a stamp (see stamp_inner_mask)."""
        k0, k1, k2, k3, l0, l1, l2, l3, m0, m1 = stamp
        return self._inner_bounds(self.x_nodes, k1, k2) + self._inner_bounds(self.y_nodes, l1, l2)

    def stamp_inner_mask(self, stamp, x, y, w):
        """
        Return a boolean mask of the samples (x, y, w) that fall inside the inner region of the stamp, i.e. where the
        model of this stamp is used (see stamp_inner_bounds).
        """
        x_low, x_high, y_low, y_high = self.stamp_inner_bounds(stamp)
        wv_nodes = self.wv_nodes
        return (x > x_low) & (x <= x_high) & \
               (y > y_low) & (y <= y_high) & \
               (w > np.min(wv_nodes)) & (w < np.max(wv_nodes))

    def _local_axis(self, axis, i0, i3):
        """
        Return the 1D spline of a stamp along one spatial axis, built on the nodes i0..i3 of the extended stamp only
        (as in the original implementation). Cached.
        """
        key = ("x" if axis is self.x else "y" if axis is self.y else "wv", i0, i3)
        if key not in self._local_axes:
            self._local_axes[key] = Spline1D(axis.nodes[i0:i3 + 1], basis=self.basis, degree=self.degree)
        return self._local_axes[key]

    def _axis_param_slice(self, axis, i0, i3):
        """
        Return the slice of the parameter array where a stamp spanning nodes i0..i3 stores its parameters along one
        spatial axis: from i0, n_params long (i0..i3 for cardinal and bspline_nak, i0..i3+degree-1 for bspline).
        """
        return slice(i0, i0 + self._local_axis(axis, i0, i3).n_params)

    def stamp_param_slices(self, stamp):
        """Return the (wv, y, x) slices of the parameter array (after the parity indices) used by a stamp."""
        k0, k1, k2, k3, l0, l1, l2, l3, m0, m1 = stamp
        return slice(None), self._axis_param_slice(self.y, l0, l3), self._axis_param_slice(self.x, k0, k3)

    def _axis_model(self, axis, i0, i3, samples):
        """Evaluate the 1D basis of a stamp along one spatial axis at the samples (see _local_axis)."""
        return self._local_axis(axis, i0, i3).get_spline_model(samples)

    def design_matrix(self, stamp, x, y, w):
        """
        Compute the design matrix (N_samples x N_stamp_params) of a stamp at the samples (x, y, w): the model at these
        samples is the design matrix times the parameters of the stamp. The columns follow the flattened (C order)
        parameter sub-array [wv, y, x] of the stamp (see stamp_param_slices).
        """
        k0, k1, k2, k3, l0, l1, l2, l3, m0, m1 = stamp
        M_spline_x = self._axis_model(self.x, k0, k3, x)
        M_spline_y = self._axis_model(self.y, l0, l3, y)
        M_spline_wvs = self.wv.get_spline_model(w)
        n_w, n_y, n_x = M_spline_wvs.shape[1], M_spline_y.shape[1], M_spline_x.shape[1]

        M_spline_x_tiled = np.tile(M_spline_x[:, None, None, :], (1, n_w, n_y, 1))
        M_spline_y_tiled = np.tile(M_spline_y[:, None, :, None], (1, n_w, 1, n_x))
        M_spline_wvs_tiled = np.tile(M_spline_wvs[:, :, None, None], (1, 1, n_y, n_x))
        M_3dspline = M_spline_x_tiled * M_spline_y_tiled * M_spline_wvs_tiled
        return M_3dspline.reshape((M_3dspline.shape[0], -1))  # flatten the last 3 dimensions

    def stamp_mode_peaks(self, stamp):
        """
        Return the peak value of each mode (column of the design matrix) of a stamp, i.e. the product of the peaks of
        its 1D basis functions. Used to prune the modes of the B-spline bases (see prune_modes).
        """
        if not self.is_bspline:
            raise ValueError("stamp_mode_peaks is only defined for the B-spline bases")
        k0, k1, k2, k3, l0, l1, l2, l3, m0, m1 = stamp
        peaks = (self.wv.mode_peaks[:, None, None] * self._local_axis(self.y, l0, l3).mode_peaks[None, :, None] *
                 self._local_axis(self.x, k0, k3).mode_peaks[None, None, :])
        return peaks.ravel()

    def prune_modes(self, stamp, M_3dspline, M=None):
        """
        Decide which parameters of a stamp should not be fitted, because the data cannot constrain them.

        These parameters (modes, i.e. columns of the design matrix) are where there is little or no data (edge of the
        field, gaps, masked pixels), or are redundant with other modes. Fitting them would give arbitrary values that
        make the model blow up where there is no data. The fit (_task_fit_3dspline) sets their columns to zero, so
        their best fit value and error are NaN.

        Rule for each basis:
        - cardinal, cardinal_pruning="decorrelation" (default): modes mostly explained by the other modes (see
          prune_modes_decorrelation).
        - cardinal, cardinal_pruning="historical": modes that never exceed 1% of the maximum of M.
        - bspline, bspline_nak: modes cut off by the sampling, i.e. whose largest sampled value is less than
          crop_threshold times the peak of the mode (same rule as Spline1D.prune_modes).

        Parameters
        ----------
        stamp : tuple
            Stamp (see stamps).
        M_3dspline : 2d array (N_samples x N_stamp_params)
            Design matrix of the stamp (spline only) at the samples that are fitted. Used by the B-spline rule.
        M : 2d array or None
            Model matrix that is actually fitted (e.g. M_3dspline times the stellar spectrum). Used by the cardinal
            rules. Default is M_3dspline.

        Returns
        -------
        pruned : 1d boolean array (N_stamp_params)
            True for the parameters to switch off (not fitted), False for the ones to fit.
            All True if there are no samples.
        """
        if self.is_bspline:
            if M_3dspline.shape[0] == 0:
                return np.ones(M_3dspline.shape[1], dtype=bool)
            col_max = np.nanmax(np.abs(M_3dspline), axis=0)
            return col_max < self.crop_threshold * self.stamp_mode_peaks(stamp)
        if M is None:
            M = M_3dspline
        if self.cardinal_pruning == "decorrelation":
            return self.prune_modes_decorrelation(M)
        return np.nansum(M > np.nanmax(M) * 0.01, axis=0) == 0

    @staticmethod
    def prune_modes_decorrelation(M, tol=DEFAULT_DECORRELATION_TOL):
        """
        Find the modes (columns of M) that are redundant with the other modes over the samples, so that they are not
        fitted (see prune_modes).

        The columns are tested one by one, from the best sampled (largest norm) to the least sampled. Each column is
        approximated by a least squares combination of the columns already kept; it is kept only if the residual of
        that approximation is at least a fraction tol of the column (in squared norm). Columns without any sample are
        always removed.

        Example with a single column already kept: if c is the normalized dot product of the two columns (their dot
        product divided by the product of their norms), the fraction not explained is 1 - c^2. The new column is
        removed if 1 - c^2 < tol, i.e. |c| > sqrt(1 - tol). For tol = 1e-2, |c| > sqrt(0.99) ~ 0.995, the threshold of
        the cardinal rule of Spline1D.prune_modes.

        Parameters
        ----------
        M : 2d array (N_samples x N_stamp_params)
            Model matrix of the stamp at the fitted samples (e.g. design_matrix times the stellar spectrum).
        tol : float
            Minimum fraction of a column (in squared norm) that must not be explained by the columns already kept.

        Returns
        -------
        removed : 1d boolean array (N_stamp_params)
            True for the columns to remove.
        """
        from scipy.linalg import solve_triangular

        n_params = M.shape[1]
        removed = np.ones(n_params, dtype=bool)
        if M.shape[0] == 0:
            return removed
        gram = np.dot(M.T, M)
        norms = np.sqrt(np.clip(np.diag(gram), 0, None))
        sampled = np.where(norms > 0)[0]
        if sampled.size == 0:
            return removed
        gram_n = gram[np.ix_(sampled, sampled)] / np.outer(norms[sampled], norms[sampled])  # cross-correlations

        # greedy Gram-Schmidt in the order of decreasing norm, with an incremental Cholesky factor L of the kept columns
        order = np.argsort(-norms[sampled], kind="stable")
        kept = []
        L = np.zeros((sampled.size, sampled.size))
        for j in order:
            if len(kept) == 0:
                residual = 1.0
            else:
                v = solve_triangular(L[:len(kept), :len(kept)], gram_n[kept, j], lower=True)
                residual = 1.0 - np.dot(v, v)
            if residual >= tol:
                n = len(kept)
                if n > 0:
                    L[n, :n] = v
                L[n, n] = np.sqrt(residual)
                kept.append(j)
        removed[sampled[kept]] = False
        return removed

    def stamp_params_shape(self, stamp):
        """Return the shape (n_wv, n_y, n_x) of the parameters of a stamp."""
        sw, sy, sx = self.stamp_param_slices(stamp)
        return (self.wv.n_params, sy.stop - sy.start, sx.stop - sx.start)

    # ------------------------------------------------------------------------------------------------------------------
    # FITS I/O
    def header_cards(self):
        """Return the header keywords describing the 3D spline, as {keyword: (value, comment)}."""
        cards = {"3DSPLBAS": self.basis,
                 "3DSPLDEG": self.degree,
                 "3DSPLSSX": self.stamp_size[0],
                 "3DSPLSSY": self.stamp_size[1],
                 "3DSPLNOV": self.N_overlap_nodes,
                 "3DSPLCRP": self.crop_threshold,
                 "3DSPLPRN": self.cardinal_pruning,
                 "3DSPL_NW": int(np.size(self.wv_nodes)),
                 "3DSPL_NX": int(np.size(self.x_nodes)),
                 "3DSPL_NY": int(np.size(self.y_nodes)),
                 "3DSPL_DX": float(np.median(np.diff(self.x_nodes))) if np.size(self.x_nodes) > 1 else 0.0,
                 "3DSPL_DY": float(np.median(np.diff(self.y_nodes))) if np.size(self.y_nodes) > 1 else 0.0}
        return {key: (val, _HDR_COMMENTS[key]) for key, val in cards.items()}

    def update_header(self, header):
        """Write the 3D spline keywords into an astropy header (in place). Returns the header."""
        for key, (val, comment) in self.header_cards().items():
            header[key] = (val, comment)
        return header

    def to_hdus(self):
        """Return the nodes as FITS extensions [wv_nodes, x_nodes, y_nodes] (the rest is saved in the header)."""
        from astropy.io import fits
        return [fits.ImageHDU(data=np.asarray(self.wv_nodes), name="wv_nodes"),
                fits.ImageHDU(data=np.asarray(self.x_nodes), name="x_nodes"),
                fits.ImageHDU(data=np.asarray(self.y_nodes), name="y_nodes")]

    @classmethod
    def from_hdulist(cls, hdulist):
        """
        Create the Spline3D saved in an opened FITS file (3D spline products or BreadsPSF files): nodes from the
        wv_nodes/x_nodes/y_nodes extensions, the rest from the BREADS header (primary header for BreadsPSFs).
        Older files without the keywords: cardinal, degree 3, 2 overlapping nodes, historical pruning.
        """
        header = hdulist["BREADS"].header if "BREADS" in hdulist else hdulist[0].header
        return cls(np.array(hdulist["wv_nodes"].data), np.array(hdulist["x_nodes"].data),
                   np.array(hdulist["y_nodes"].data),
                   basis=header.get("3DSPLBAS", "cardinal"),
                   degree=header.get("3DSPLDEG", 3),
                   stamp_size=(header["3DSPLSSX"], header["3DSPLSSY"]),
                   N_overlap_nodes=header.get("3DSPLNOV", 2),
                   crop_threshold=header.get("3DSPLCRP", DEFAULT_BSPLINE_CROP_THRESHOLD),
                   cardinal_pruning=header.get("3DSPLPRN", "historical"))

    @classmethod
    def from_file(cls, filename):
        """Create the Spline3D saved in a FITS file (see from_hdulist)."""
        from astropy.io import fits
        with fits.open(filename) as hdulist:
            return cls.from_hdulist(hdulist)

    # ------------------------------------------------------------------------------------------------------------------
    def __eq__(self, other):
        """Check that two 3D splines are the same basis (nodes, basis type, stamp layout and pruning rule)."""
        if not isinstance(other, Spline3D):
            return NotImplemented
        return (self.basis == other.basis and self.degree == other.degree and
                self.stamp_size == other.stamp_size and self.N_overlap_nodes == other.N_overlap_nodes and
                np.array_equal(self.wv_nodes, other.wv_nodes) and np.array_equal(self.x_nodes, other.x_nodes) and
                np.array_equal(self.y_nodes, other.y_nodes) and
                (not self.is_bspline or self.crop_threshold == other.crop_threshold) and
                (self.is_bspline or self.cardinal_pruning == other.cardinal_pruning))

    def __ne__(self, other):
        out = self.__eq__(other)
        return out if out is NotImplemented else not out

    __hash__ = None

    def __repr__(self):
        return (f"Spline3D(basis='{self.basis}', degree={self.degree}, nodes (wv, y, x)=({np.size(self.wv_nodes)}, "
                f"{np.size(self.y_nodes)}, {np.size(self.x_nodes)}), params_shape={self.params_shape}, "
                f"stamp_size={self.stamp_size}, N_overlap_nodes={self.N_overlap_nodes}"
                + (f", crop_threshold={self.crop_threshold})" if self.is_bspline else f", cardinal_pruning='{self.cardinal_pruning}')"))
