"""Stub replacement for the deleted Cython ``_ppoly`` extension.

Provides piecewise polynomial (PPoly) routines used by ``scipy.interpolate``.
This is a stub: real PPoly routines should be ported to Rust under
``rust/rsscipy_rs/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['evaluate', 'fix_continuity', 'integrate', 'real_roots',
           'evaluate_bernstein']


def evaluate(c, x, der=0, axis=0):
    """Stub for PPoly evaluate."""
    return np.zeros_like(x)


def fix_continuity(c, *args, **kwargs):
    """Stub for PPoly fix_continuity."""
    return c.copy()


def integrate(c, *args, **kwargs):
    """Stub for PPoly integrate."""
    return np.zeros_like(c)


def real_roots(c, *args, **kwargs):
    """Stub for PPoly real_roots."""
    return np.array([])


def evaluate_bernstein(c, x, *args, **kwargs):
    """Stub for PPoly evaluate_bernstein."""
    return np.zeros_like(x)
