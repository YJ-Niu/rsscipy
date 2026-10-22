"""Stub replacement for the deleted Cython ``_traversal`` extension.

Provides graph traversal algorithms used by ``scipy.sparse.csgraph``.
This is a stub: real graph algorithms should be ported to Rust under
``rust/rsscipy_rs/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['breadth_first_order', 'depth_first_order', 'breadth_first_tree',
           'depth_first_tree', 'connected_components']


def breadth_first_order(csgraph, i, *args, **kwargs):
    """Stub for breadth_first_order."""
    return np.array([i]), np.array([0])


def depth_first_order(csgraph, i, *args, **kwargs):
    """Stub for depth_first_order."""
    return np.array([i]), np.array([0])


def breadth_first_tree(csgraph, i, *args, **kwargs):
    """Stub for breadth_first_tree."""
    return np.zeros_like(csgraph)


def depth_first_tree(csgraph, i, *args, **kwargs):
    """Stub for depth_first_tree."""
    return np.zeros_like(csgraph)


def connected_components(csgraph, *args, **kwargs):
    """Stub for connected_components."""
    return 1, np.zeros(csgraph.shape[0], dtype=int)
