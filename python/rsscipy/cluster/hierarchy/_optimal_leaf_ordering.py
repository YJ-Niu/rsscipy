"""Stub replacement for the deleted Cython ``_optimal_leaf_ordering`` extension."""

from __future__ import annotations

import rsnumpy as np

__all__ = ['optimal_leaf_ordering']


def optimal_leaf_ordering(Z, y):
    return np.arange(y.shape[0])
