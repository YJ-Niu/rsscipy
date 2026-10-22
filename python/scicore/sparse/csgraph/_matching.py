"""Stub replacement for the deleted Cython ``_matching`` extension."""

from __future__ import annotations

import rsnumpy as np

__all__ = ['maximum_bipartite_matching', 'min_weight_full_bipartite_matching']


def maximum_bipartite_matching(csgraph, *args, **kwargs):
    return np.zeros(csgraph.shape[0], dtype=int)


def min_weight_full_bipartite_matching(csgraph, *args, **kwargs):
    return np.zeros(csgraph.shape[0], dtype=int)
