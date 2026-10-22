"""Stub replacement for the deleted Cython ``_shortest_path`` extension.

Provides graph shortest-path algorithms used by ``scipy.sparse.csgraph``.
This is a stub: real graph algorithms should be ported to Rust under
``rust/rsscipy_rs/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['shortest_path', 'floyd_warshall', 'dijkstra', 'bellman_ford',
           'johnson', 'yen', 'NegativeCycleError']


class NegativeCycleError(ValueError):
    """Raised when a negative cycle is detected."""
    pass


def shortest_path(csgraph, *args, **kwargs):
    """Stub for shortest_path."""
    return np.zeros(csgraph.shape)


def floyd_warshall(csgraph, *args, **kwargs):
    """Stub for floyd_warshall."""
    return np.zeros(csgraph.shape), np.zeros(csgraph.shape, dtype=int)


def dijkstra(csgraph, *args, **kwargs):
    """Stub for dijkstra."""
    return np.zeros(csgraph.shape[0]), np.zeros(csgraph.shape[0], dtype=int)


def bellman_ford(csgraph, *args, **kwargs):
    """Stub for bellman_ford."""
    return np.zeros(csgraph.shape[0]), np.zeros(csgraph.shape[0], dtype=int)


def johnson(csgraph, *args, **kwargs):
    """Stub for johnson."""
    return np.zeros(csgraph.shape)


def yen(*args, **kwargs):
    """Stub for yen."""
    return []
