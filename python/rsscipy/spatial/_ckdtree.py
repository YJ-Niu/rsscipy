"""Stub replacement for the deleted Cython ``_ckdtree`` extension.

Provides the KD-tree routines used by ``scipy.spatial`` and
``scipy.cluster``, implemented on top of :mod:`rsnumpy`.  This is a stub:
the full KD-tree implementation should later be ported to Rust under
``rust/rsscipy_rs/``.
"""

from __future__ import annotations

import rsnumpy as np


class cKDTreeNode:
    """A node in a KD-tree (stub)."""

    def __init__(self):
        self.left = None
        self.right = None
        self.split_dim = None
        self.split = None
        self.data = None
        self.indices = None


class cKDTree:
    """k-d tree for nearest-neighbor lookup (stub)."""

    def __init__(self, data, leafsize=16):
        self.data = np.asarray(data)
        self.leafsize = leafsize

    def query(self, x, k=1, n_jobs=None):
        x = np.asarray(x)
        # brute-force nearest neighbor (stub)
        d = np.sqrt(np.sum((self.data[None, :, :] - x[:, None, :]) ** 2, axis=2))
        idx = np.argmin(d, axis=1)
        dist = d[np.arange(len(x)), idx]
        return dist, idx

    def query_ball_point(self, x, r, n_jobs=None):
        raise NotImplementedError("query_ball_point not yet ported")

    def query_ball_tree(self, other, r, n_jobs=None):
        raise NotImplementedError("query_ball_tree not yet ported")

    def query_pairs(self, r, n_jobs=None):
        raise NotImplementedError("query_pairs not yet ported")

    def count_neighbors(self, other, r, n_jobs=None):
        raise NotImplementedError("count_neighbors not yet ported")

    def sparse_distance_matrix(self, other, max_distance, n_jobs=None):
        raise NotImplementedError("sparse_distance_matrix not yet ported")


def cKDTree_query(*args, **kwargs):
    raise NotImplementedError("cKDTree.query not yet ported")

# Module-level attribute expected by spatial/__init__.py
__all__ = ['cKDTree', 'cKDTreeNode']
