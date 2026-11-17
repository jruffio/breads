"""
Spline1D: a small container describing the 1d (wavelength) spline basis used to model the starlight continuum in the
JWST BREADS functions (e.g. compute_starspectrum_contnorm, compute_starsubtraction, hc_atmgrid_splinefm_jwst_ifu_cal).

It lets the same code use either:
    - the "cardinal" basis (breads.utils.splines.get_spline_model; default and historical behavior): each linear
      parameter is the value of the continuum at the position of a node.
    - the "bspline" basis (breads.utils.splines.get_bspline_model): each linear parameter is a B-spline control point.
      Clamped knots are used, so there are n_nodes + degree - 1 parameters per segment of nodes (i.e., not one per node).

Code that needs the number of linear parameters must use Spline1D.n_params (not np.size(nodes)), and code that needs a
wavelength associated with each linear parameter must use Spline1D.param_positions (not the nodes).

Any function accepting spline nodes should call Spline1D.coerce() on it, so that plain arrays of nodes keep working
(interpreted as a cardinal basis).
"""
import numpy as np
import astropy.io.fits as fits

from breads.utils.splines import get_spline_model, get_bspline_model, get_bspline_knots

__all__ = ["Spline1D"]

# Default crop threshold of the bspline basis (see Spline1D.prune_modes): a B-spline mode whose maximum over the
# fitted samples is below 0.1% of its peak is removed from the fit.
DEFAULT_BSPLINE_CROP_THRESHOLD = 1e-3

_BASIS_ALIASES = {"cardinal": "cardinal",
                  "bspline": "bspline", "b-spline": "bspline", "b_spline": "bspline", "bsplines": "bspline",
                  "b-splines": "bspline"}

# FITS header keywords, used both in the wv_nodes HDU and in the primary headers of the data products.
_HDR_COMMENTS = {"SPLBASIS": "1D continuum spline basis (cardinal or bspline)",
                 "SPLDEG": "1D continuum spline degree",
                 "SPLCROP": "bspline: crop threshold for truncated modes",
                 "SPLNNODE": "1D continuum spline number of nodes",
                 "SPLNPAR": "1D continuum spline number of linear parameters",
                 "SPLSEGS": "1D spline segment lengths (discontinuous nodes)"}


def _normalize_basis(basis):
    key = str(basis).strip().lower()
    if key not in _BASIS_ALIASES:
        raise ValueError(f"Unknown spline basis '{basis}'. Use 'cardinal' or 'bspline'.")
    return _BASIS_ALIASES[key]


class Spline1D:
    """
    Description of a 1d spline basis (nodes + type of basis).

    Parameters
    ----------
    nodes : 1d array, or list of 1d arrays
        Position of the spline nodes (e.g. in microns). A list of arrays defines a discontinuous spline (one independent
        spline per segment), same convention as breads.utils.splines.get_spline_model.
    basis : str (optional, default "cardinal")
        "cardinal" (get_spline_model) or "bspline" (get_bspline_model).
    degree : int (optional, default 3)
        Degree of the splines.
    crop_threshold : float (optional, default DEFAULT_BSPLINE_CROP_THRESHOLD = 1e-3)
        Only used for the bspline basis, in prune_modes(). A B-spline mode is removed from a fit if its maximum value
        over the fitted samples is less than crop_threshold times the peak value of that mode.
    """

    def __init__(self, nodes, basis="cardinal", degree=3, crop_threshold=DEFAULT_BSPLINE_CROP_THRESHOLD):
        if isinstance(nodes, (list, tuple)) and len(nodes) > 0 and isinstance(nodes[0], (list, tuple, np.ndarray)):
            self.nodes = [np.asarray(n, dtype=float) for n in nodes]
        else:
            self.nodes = np.asarray(nodes, dtype=float)
        self.basis = _normalize_basis(basis)
        self.degree = int(degree)
        self.crop_threshold = float(crop_threshold)
        self._n_params = None
        self._mode_peaks = None

    # ------------------------------------------------------------------------------------------------------------------
    # Construction helpers
    @classmethod
    def coerce(cls, obj, wavelengths=None, basis=None):
        """
        Convert obj into a Spline1D.

        Parameters
        ----------
        obj : Spline1D, 1d array, list of 1d arrays, or int
            - Spline1D: returned as is (if basis is given, it must match).
            - array or list of arrays: nodes of a spline with the given basis (default "cardinal").
            - int: number of nodes, evenly spaced between the min and max of wavelengths.
        wavelengths : array (optional)
            Only used when obj is an int.
        basis : str or None (optional)
            Basis to use when obj is not already a Spline1D. Default is "cardinal".
        """
        if isinstance(obj, cls):
            if basis is not None and _normalize_basis(basis) != obj.basis:
                raise ValueError(f"Spline1D basis is '{obj.basis}' but '{basis}' was requested.")
            return obj
        if basis is None:
            basis = "cardinal"
        if obj is None:
            raise ValueError("Spline nodes cannot be None.")
        if isinstance(obj, (int, np.integer)):
            if wavelengths is None:
                raise ValueError("wavelengths must be given to define the spline nodes from a number of nodes.")
            obj = np.linspace(np.nanmin(wavelengths), np.nanmax(wavelengths), int(obj), endpoint=True)
        return cls(obj, basis=basis)

    def copy(self):
        return Spline1D(self.nodes if not self.is_segmented else [np.copy(n) for n in self.nodes],
                        basis=self.basis, degree=self.degree, crop_threshold=self.crop_threshold)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    @property
    def is_segmented(self):
        return isinstance(self.nodes, list)

    @property
    def nodes_list(self):
        """List of node arrays, one per segment."""
        return self.nodes if self.is_segmented else [self.nodes]

    @property
    def n_nodes(self):
        return int(np.sum([np.size(n) for n in self.nodes_list]))

    @property
    def is_bspline(self):
        return self.basis == "bspline"

    @property
    def n_params(self):
        """Number of linear parameters (= number of columns of get_spline_model())."""
        if self._n_params is None:
            mid = np.nanmedian(np.concatenate(self.nodes_list))
            self._n_params = int(self.get_spline_model(np.array([mid])).shape[1])
        return self._n_params

    @property
    def param_positions(self):
        """
        Position (e.g. wavelength) associated with each linear parameter.
        cardinal: the nodes themselves. bspline: the Greville abscissae of the clamped knot vector.
        """
        if self.n_params == 1:
            return np.array([np.nanmedian(np.concatenate(self.nodes_list))])
        if not self.is_bspline:
            return np.concatenate(self.nodes_list)
        k = self._effective_degree()
        pos_list = []
        for nodes in self.nodes_list:
            t = get_bspline_knots(nodes, k)
            n_basis = np.size(t) - k - 1
            if k == 0:
                pos_list.append(0.5 * (t[:-1] + t[1:])[:n_basis])
            else:
                pos_list.append(np.array([np.mean(t[i + 1:i + k + 1]) for i in range(n_basis)]))
        return np.concatenate(pos_list)

    @property
    def mode_peaks(self):
        """Peak value of each basis function (column of the model), evaluated on a dense grid. Cached."""
        if self._mode_peaks is None:
            x_dense = np.concatenate([np.linspace(np.min(n), np.max(n), 50 * np.size(n) + 1)[1:-1]
                                      for n in self.nodes_list])
            self._mode_peaks = np.nanmax(np.abs(self.get_spline_model(x_dense)), axis=0)
        return self._mode_peaks

    def _effective_degree(self):
        # Same rule as in get_spline_model and get_bspline_model
        if not self.is_segmented and np.size(self.nodes) <= self.degree:
            return np.size(self.nodes) - 1
        return self.degree

    def _model_nodes(self):
        return self.nodes

    # ------------------------------------------------------------------------------------------------------------------
    # Linear model
    def get_spline_model(self, x_samples):
        """
        Linear model matrix M (D x n_params) of the spline basis evaluated at x_samples.
        np.dot(M, paras) is the spline model.
        """
        x_samples = np.asarray(x_samples)
        if self.is_bspline:
            return get_bspline_model(self._model_nodes(), x_samples, spline_degree=self.degree)
        return get_spline_model(self._model_nodes(), x_samples, spline_degree=self.degree)

    def prune_modes(self, M_spline, M_eval=None, return_mask=False):
        """
        Set to zero the columns of M_spline that should not be fitted. M_spline is modified in place and returned.

        cardinal: remove redundant edge modes, i.e. leading/trailing columns that are nearly collinear with their
            neighbor (first off-diagonal of the normalized Gram matrix > 1-5e-3). This is the historical behavior.
        bspline: remove modes that have been truncated by the sampling (data coverage, bad pixels, edges), i.e.
            columns whose maximum is less than crop_threshold times the peak value of the full mode.

        Parameters
        ----------
        M_spline : 2d array (D x n_params)
            Spline model matrix, modified in place.
        M_eval : 2d array or None (optional)
            Matrix (? x n_params) used to decide which modes to remove (e.g. M_spline restricted to the valid pixels).
            Default is M_spline itself.
        return_mask : bool (optional)
            If True, also return a boolean array (n_params) flagging the removed modes.

        Returns
        -------
        M_spline, or (M_spline, removed_mask) if return_mask
        """
        if M_eval is None:
            M_eval = M_spline
        removed = np.zeros(M_spline.shape[1], dtype=bool)
        if self.is_bspline:
            col_max = np.nanmax(np.abs(M_eval), axis=0) if M_eval.shape[0] != 0 else np.zeros(M_eval.shape[1])
            removed[col_max < self.crop_threshold * self.mode_peaks] = True
            M_spline[:, removed] = 0
            return (M_spline, removed) if return_mask else M_spline

        # cardinal: remove redundant spline modes
        dotprod_M = np.dot(M_eval.transpose(), M_eval)
        dotprod_M_diag = np.sqrt(np.diag(dotprod_M))
        dotprod_M = np.abs(dotprod_M / (dotprod_M_diag[np.newaxis, :] * dotprod_M_diag[:, np.newaxis]))
        d1 = np.diag(dotprod_M, k=1)  # First off diagonal
        is_one = d1 > (1 - 5e-3)
        not_one_idx = np.flatnonzero(~is_one)  # indices where v is NOT ~1
        if not_one_idx.size == 0:
            # vector is all ones
            last_lead, first_trail = len(d1) - 1, 0
        else:
            last_lead = not_one_idx[0] - 1 if is_one[0] else -1  # -1 = no leading ones
            first_trail = not_one_idx[-1] + 1 if is_one[-1] else M_eval.shape[1] - 1  # no trailing ones
        M_spline[:, 0:last_lead + 1] = 0
        M_spline[:, first_trail + 1:M_spline.shape[1]] = 0
        removed[0:last_lead + 1] = True
        removed[first_trail + 1:] = True
        return (M_spline, removed) if return_mask else M_spline

    # ------------------------------------------------------------------------------------------------------------------
    # FITS I/O
    def header_cards(self):
        """Dict of FITS keywords -> (value, comment) describing the spline basis."""
        cards = {"SPLBASIS": self.basis,
                 "SPLDEG": self.degree,
                 "SPLNNODE": self.n_nodes,
                 "SPLNPAR": self.n_params}
        if self.is_bspline:
            cards["SPLCROP"] = self.crop_threshold
        if self.is_segmented:
            cards["SPLSEGS"] = ",".join(str(np.size(n)) for n in self.nodes_list)
        return {key: (val, _HDR_COMMENTS[key]) for key, val in cards.items()}

    def update_header(self, header):
        """Write the spline description keywords into an astropy header (in place). Returns the header."""
        for key in _HDR_COMMENTS:
            if key in header:
                del header[key]
        for key, (val, comment) in self.header_cards().items():
            header[key] = (val, comment)
        return header

    def to_hdu(self, name="wv_nodes"):
        """ImageHDU containing the nodes (concatenated if segmented) with the basis described in its header."""
        hdu = fits.ImageHDU(data=np.concatenate(self.nodes_list), name=name)
        self.update_header(hdu.header)
        return hdu

    @classmethod
    def from_header_and_nodes(cls, nodes, header=None):
        """Rebuild a Spline1D from an array of nodes and a header (absent keywords -> cardinal, degree 3)."""
        basis, degree, crop = "cardinal", 3, DEFAULT_BSPLINE_CROP_THRESHOLD
        if header is not None:
            basis = header.get("SPLBASIS", basis)
            degree = header.get("SPLDEG", degree)
            crop = header.get("SPLCROP", crop)
            segs = header.get("SPLSEGS", None)
            if segs:
                lengths = [int(s) for s in str(segs).split(",")]
                nodes = np.split(np.asarray(nodes), np.cumsum(lengths)[:-1])
                nodes = [np.asarray(n) for n in nodes]
        return cls(nodes, basis=basis, degree=degree, crop_threshold=crop)

    @classmethod
    def from_hdulist(cls, hdulist, name="wv_nodes"):
        """
        Read the spline description from an opened FITS file. Looks for the HDU `name` (or the legacy "x_nodes").
        Files without the SPLBASIS keyword (i.e. made before the B-spline option existed) are cardinal.
        Returns None if there are no nodes in the file.
        """
        for _name in [name, "x_nodes"]:
            if _name in hdulist:
                hdu = hdulist[_name]
                return cls.from_header_and_nodes(np.array(hdu.data), hdu.header)
        return None

    # ------------------------------------------------------------------------------------------------------------------
    def __eq__(self, other):
        if not isinstance(other, Spline1D):
            return NotImplemented
        if self.basis != other.basis or self.degree != other.degree or self.is_segmented != other.is_segmented:
            return False
        if self.is_bspline and self.crop_threshold != other.crop_threshold:
            return False
        if len(self.nodes_list) != len(other.nodes_list):
            return False
        return all(np.array_equal(a, b) for a, b in zip(self.nodes_list, other.nodes_list))

    def __ne__(self, other):
        out = self.__eq__(other)
        return out if out is NotImplemented else not out

    __hash__ = None

    def __repr__(self):
        return (f"Spline1D(basis='{self.basis}', degree={self.degree}, n_nodes={self.n_nodes}, "
                f"n_params={self.n_params}, range=[{np.min(np.concatenate(self.nodes_list)):.4f}, "
                f"{np.max(np.concatenate(self.nodes_list)):.4f}])")
