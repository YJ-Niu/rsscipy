"""cKDTree — fast kd-tree backed by the Rust ``_scicore_rs.spatial`` module.

This is the fast (C++-equivalent) implementation of a kd-tree.
``scicore.spatial.KDTree`` (``_kdtree.py``) subclasses :class:`cKDTree` and
delegates its query methods to it. The heavy computation (tree construction +
nearest-neighbor / ball queries) lives in Rust; this module handles
scipy-compatible argument normalization and output formatting.

.. note::
   Periodic (``boxsize``) topology is accepted but distances are computed
   with the plain (non-periodic) metric.
"""

from __future__ import annotations

import rsnumpy as np

from scicore._scicore_rs import spatial as _rust_spatial

KDTreeCore = _rust_spatial.KDTreeCore

__all__ = ["cKDTree", "cKDTreeNode", "cKDTree_query"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_klist(k):
    """Normalize ``k`` to a list of 1-based ranks for the Rust core.

    An integer ``k`` means "the ``k`` nearest neighbors" -> ranks ``1..k``.
    A sequence ``k`` is taken as the explicit (1-based) ranks.
    """
    if isinstance(k, (list, tuple, np.ndarray)):
        return [int(v) for v in np.asarray(k).ravel().tolist()]
    return list(range(1, int(k) + 1))


def _to_pow(d, p):
    """Raise a true distance ``d`` to ``d**p`` units (``p=inf`` -> ``d``)."""
    if p == float("inf"):
        return d
    return d ** p


# ---------------------------------------------------------------------------
# cKDTreeNode
# ---------------------------------------------------------------------------

class cKDTreeNode:
    """A node in a :class:`cKDTree`.

    Exposes the node structure used by ``KDTree``'s tree traversal
    (``split_dim``, ``split``, ``data``, ``indices``, ``children`` and the
    ``lesser``/``greater`` child nodes).
    """

    def __init__(self, core=None, node_index=0):
        self._core = core
        self._node_index = node_index
        if core is not None:
            (
                self.split_dim,
                self.split,
                self.children,
                self._is_leaf,
                self._less,
                self._greater,
                self.start,
                self.end,
            ) = core.node_info(node_index)
        else:  # empty placeholder node
            self.split_dim = -1
            self.split = 0.0
            self.children = 0
            self._is_leaf = True
            self._less = 0
            self._greater = 0
            self.start = 0
            self.end = 0

    @property
    def indices(self):
        return np.asarray(self._core.node_indices(self._node_index), dtype=np.intp)

    @property
    def data(self):
        m = self._core.m
        flat = self._core.node_data(self._node_index)
        return np.asarray(flat, dtype=float).reshape(-1, m)

    @property
    def lesser(self):
        return cKDTreeNode(self._core, self._less)

    @property
    def greater(self):
        return cKDTreeNode(self._core, self._greater)


# ---------------------------------------------------------------------------
# cKDTree
# ---------------------------------------------------------------------------

class cKDTree:
    """k-d tree for nearest-neighbor lookup (Rust-backed)."""

    def __init__(
        self,
        data,
        leafsize=10,
        compact_nodes=True,
        copy_data=True,
        balanced_tree=True,
        boxsize=None,
    ):
        data = np.asarray(data)
        if data.dtype.kind == "c":
            raise TypeError("cKDTree does not work with complex data")
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        self._data = data
        self._leafsize = leafsize
        self._compact_nodes = compact_nodes
        self._copy_data = copy_data
        self._balanced_tree = balanced_tree
        self._boxsize = boxsize
        self._core = KDTreeCore(
            data, leafsize, compact_nodes, copy_data, balanced_tree, boxsize
        )

    # -- properties ---------------------------------------------------------

    @property
    def data(self):
        return self._data

    @property
    def leafsize(self):
        return self._leafsize

    @property
    def compact_nodes(self):
        return self._compact_nodes

    @property
    def copy_data(self):
        return self._copy_data

    @property
    def balanced_tree(self):
        return self._balanced_tree

    @property
    def boxsize(self):
        return self._boxsize

    @property
    def m(self):
        return self._core.m

    @property
    def n(self):
        return self._core.n

    @property
    def ndim(self):
        return self._core.m

    @property
    def maxes(self):
        return np.asarray(self._core.maxes(), dtype=float)

    @property
    def mins(self):
        return np.asarray(self._core.mins(), dtype=float)

    @property
    def tree(self):
        return cKDTreeNode(self._core, 0)

    # -- queries ------------------------------------------------------------

    def query(self, x, k=1, eps=0.0, p=2.0, distance_upper_bound=np.inf, workers=1):
        """Find the ``k`` nearest neighbors of each point in ``x``."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            leading = ()
            x2 = x.reshape(1, -1)
        else:
            leading = x.shape[:-1]
            x2 = x.reshape(-1, x.shape[-1])
        klist = _as_klist(k)
        nk = len(klist)
        nq = x2.shape[0]
        d_flat, i_flat = self._core.query(x2, klist, eps, p, distance_upper_bound)
        d = np.asarray(d_flat, dtype=float).reshape(nq, nk)
        i = np.asarray(i_flat, dtype=np.intp).reshape(nq, nk)
        if nk == 1:
            d = d[:, 0]
            i = i[:, 0]
            if leading:
                d = d.reshape(leading)
                i = i.reshape(leading)
            else:
                # single query point + k=1 -> scalar (scipy returns np.float64/int)
                d = float(d[0])
                i = int(i[0])
        else:
            d = d.reshape(leading + (nk,))
            i = i.reshape(leading + (nk,))
        return d, i

    def query_ball_point(
        self, x, r, p=2.0, eps=0.0, workers=1, return_sorted=None, return_length=False
    ):
        """Find all points in the tree within radius ``r`` of each point in ``x``."""
        x = np.asarray(x, dtype=float)
        was_1d = x.ndim == 1
        if was_1d:
            x = x.reshape(1, -1)
        nq = x.shape[0]
        r_arr = np.asarray(r, dtype=float)
        if r_arr.ndim == 0:
            results = self._core.query_ball_point(x, float(r_arr), p, eps)
        else:
            results = [
                self._core.query_ball_point(x[q : q + 1], float(r_arr[q]), p, eps)[0]
                for q in range(nq)
            ]
        results = [np.asarray(res, dtype=np.intp) for res in results]
        if return_length:
            lens = [len(res) for res in results]
            if was_1d:
                return lens[0]
            return np.asarray(lens, dtype=np.intp)
        sort = (nq > 1) if return_sorted is None else return_sorted
        if sort:
            results = [np.sort(res) for res in results]
        if was_1d:
            return results[0]
        return results

    def query_ball_tree(self, other, r, p=2.0, eps=0.0):
        """Find all pairs of points (self, other) within radius ``r``."""
        r_arr = np.asarray(r, dtype=float)
        if r_arr.ndim == 0:
            results = self._core.query_ball_tree(other._core, float(r_arr), p, eps)
        else:
            # per-point radius: query each self point against other individually
            results = [
                other._core.query_ball_point(self._data[q : q + 1], float(r_arr[q]), p, eps)[0]
                for q in range(self._core.n)
            ]
        return [np.asarray(res, dtype=np.intp) for res in results]

    def query_pairs(self, r, p=2.0, eps=0.0, output_type="set"):
        """Find all pairs of points in the tree within radius ``r``."""
        pairs = self._core.query_pairs(r, p, eps)
        if output_type == "set":
            return set(pairs)
        if not pairs:
            return np.empty((0, 2), dtype=np.intp)
        return np.asarray(pairs, dtype=np.intp).reshape(-1, 2)

    def count_neighbors(self, other, r, p=2.0, weights=None, cumulative=True):
        """Count how many nearby pairs can be formed (binned by radius)."""
        r_arr = np.asarray(r, dtype=float)
        scalar_r = r_arr.ndim == 0
        if weights is None:
            sw = None
            ow = None
        elif isinstance(weights, (tuple, list)) and len(weights) == 2:
            sw = np.asarray(weights[0], dtype=float).ravel().tolist()
            ow = np.asarray(weights[1], dtype=float).ravel().tolist()
        else:
            sw = np.asarray(weights, dtype=float).ravel().tolist()
            ow = None
        if scalar_r:
            real_r = [_to_pow(float(r_arr), p)]
            res = self._core.count_neighbors(other._core, real_r, p, sw, ow, cumulative)
            val = float(res[0])
            return int(val) if weights is None else val
        uniq, inverse = np.unique(r_arr, return_inverse=True)
        real_r = [_to_pow(float(v), p) for v in np.asarray(uniq).ravel().tolist()]
        res = np.asarray(
            self._core.count_neighbors(other._core, real_r, p, sw, ow, cumulative),
            dtype=float,
        )
        out = res[np.asarray(inverse, dtype=np.intp)]
        return out.astype(np.intp) if weights is None else out

    def sparse_distance_matrix(self, other, max_distance, p=2.0, output_type="dok_array"):
        """Compute a sparse distance matrix (zeros beyond ``max_distance``)."""
        entries = self._core.sparse_distance_matrix(other._core, max_distance, p)
        n = self._core.n
        nother = other._core.n
        i = np.asarray([e[0] for e in entries], dtype=np.intp)
        j = np.asarray([e[1] for e in entries], dtype=np.intp)
        v = np.asarray([e[2] for e in entries], dtype=float)
        if output_type == "ndarray":
            rec = np.empty(len(v), dtype=[("i", np.intp), ("j", np.intp), ("v", float)])
            rec["i"] = i
            rec["j"] = j
            rec["v"] = v
            return rec
        if output_type == "dict":
            return {(int(a), int(b)): float(c) for a, b, c in zip(i, j, v)}
        from scicore import sparse

        if output_type in ("coo_array", "coo_matrix", "caught_coo_matrix"):
            return sparse.coo_array((v, (i, j)), shape=(n, nother))
        # dok (default)
        mat = sparse.dok_array((n, nother))
        for a, b, c in zip(i, j, v):
            mat[int(a), int(b)] = float(c)
        return mat


# ---------------------------------------------------------------------------
# module-level helper (kept for import compatibility)
# ---------------------------------------------------------------------------

def cKDTree_query(*args, **kwargs):
    """Legacy module-level helper: ``cKDTree_query(tree, x, ...)`` -> ``tree.query(...)``."""
    if not args:
        raise TypeError("cKDTree_query requires a cKDTree instance as first argument")
    tree = args[0]
    return tree.query(*args[1:], **kwargs)
