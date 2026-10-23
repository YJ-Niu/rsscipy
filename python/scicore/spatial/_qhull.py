"""Stub replacement for the deleted Cython ``_qhull`` extension.

Provides Qhull interface used by ``scipy.spatial``.
This is a stub: real Qhull integration should be ported to Rust under
``rust/src/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['Delaunay', 'ConvexHull', 'Voronoi', 'HalfspaceIntersection',
           'QhullError', 'Qhull']


class QhullError(Exception):
    pass


class Qhull:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("_qhull not yet ported to Rust/rsnumpy")


class Delaunay:
    def __init__(self, points, *args, **kwargs):
        raise NotImplementedError("Delaunay not yet ported to Rust/rsnumpy")


class ConvexHull:
    def __init__(self, points, *args, **kwargs):
        raise NotImplementedError("ConvexHull not yet ported to Rust/rsnumpy")


class Voronoi:
    def __init__(self, points, *args, **kwargs):
        raise NotImplementedError("Voronoi not yet ported to Rust/rsnumpy")


class HalfspaceIntersection:
    def __init__(self, halfspaces, interior_point, *args, **kwargs):
        raise NotImplementedError("HalfspaceIntersection not yet ported")
