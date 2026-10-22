"""Stub replacement for the deleted C ``_direct`` extension.

Provides the DIRECT (Dividing Rectangles) optimizer used by
``scipy.optimize._direct_py``.
This is a stub: real DIRECT should be ported to Rust under ``rust/scicore_rs/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['direct']


def direct(f, bounds, args=(), tol=1e-5, maxiter=1000, **kwargs):
    """Stub for DIRECT optimizer."""
    x0 = np.mean(bounds, axis=1)
    return x0, np.inf, 0, 0, 0
