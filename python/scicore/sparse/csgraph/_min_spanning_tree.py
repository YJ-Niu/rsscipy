"""Stub replacement for the deleted Cython ``_min_spanning_tree`` extension."""

from __future__ import annotations

import rsnumpy as np

__all__ = ['minimum_spanning_tree']


def minimum_spanning_tree(csgraph, *args, **kwargs):
    return np.zeros_like(csgraph)
