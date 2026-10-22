"""Stub replacement for the deleted Cython ``_interpnd`` extension.

Provides multi-dimensional interpolation helpers used by ``scipy.interpolate``.
This is a stub: real interpolation routines should be ported to Rust under
``rust/scicore_rs/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['_ndim_coords_from_arrays', 'LinearNDInterpolator',
           'NDInterpolatorBase', 'CloughTocher2DInterpolator']


def _ndim_coords_from_arrays(*args, **kwargs):
    """Stub for _ndim_coords_from_arrays."""
    return args


class NDInterpolatorBase:
    """Stub ND interpolator base class."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("_interpnd not yet ported to Rust/rsnumpy")


class LinearNDInterpolator(NDInterpolatorBase):
    """Stub linear ND interpolator."""

    def __init__(self, points, values, *args, **kwargs):
        self.points = np.asarray(points)
        self.values = np.asarray(values)


class CloughTocher2DInterpolator(NDInterpolatorBase):
    """Stub Clough-Tocher 2D interpolator."""

    def __init__(self, points, values, *args, **kwargs):
        self.points = np.asarray(points)
        self.values = np.asarray(values)
