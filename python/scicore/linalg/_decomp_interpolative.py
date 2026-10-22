"""Stub replacement for the deleted Cython ``_decomp_interpolative`` extension.

Provides interpolative decomposition routines used by ``scipy.linalg``.
This is a stub: real interpolative decomposition should be ported to Rust
under ``rust/rsscipy_rs/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['interp_decomp']


def interp_decomp(A, k=None):
    """Stub for interpolative decomposition."""
    return None
