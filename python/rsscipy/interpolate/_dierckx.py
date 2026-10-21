"""Stub replacement for the deleted Cython ``_dierckx`` extension.

Provides FITPACK/Dierckx spline routines used by ``scipy.interpolate``.
This is a stub: real spline routines should be ported to Rust under
``rust/rsscipy_rs/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['qr_reduce', 'fpback', 'data_matrix']


def qr_reduce(*args, **kwargs):
    """Stub for Dierckx qr_reduce."""
    return None, None, None


def fpback(*args, **kwargs):
    """Stub for Dierckx fpback."""
    return None, None, None


def data_matrix(*args, **kwargs):
    """Stub for Dierckx data_matrix."""
    return None
