"""Stub replacement for the deleted Cython ``_stats`` extension.

Provides basic statistical functions used by ``scipy.stats._stats_py``.
This is a stub: real statistical routines should be ported to Rust under
``rust/rsscipy_rs/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['_kendall_dis', '_toint64', '_weightedrankedtau',
           'gaussian_kernel_estimate', 'gaussian_kernel_estimate_log']


def _kendall_dis(x, y):
    """Stub for Kendall's tau distance."""
    return 0.0


def _toint64(x):
    """Stub for converting to int64."""
    return np.asarray(x, dtype=np.int64)


def _weightedrankedtau(x, y, sw=None):
    """Stub for weighted ranked tau."""
    return 0.0, 1.0


def gaussian_kernel_estimate(x, *args, **kwargs):
    """Stub for gaussian_kernel_estimate."""
    return np.zeros_like(x), np.zeros_like(x)


def gaussian_kernel_estimate_log(x, *args, **kwargs):
    """Stub for gaussian_kernel_estimate_log."""
    return np.zeros_like(x), np.zeros_like(x)


