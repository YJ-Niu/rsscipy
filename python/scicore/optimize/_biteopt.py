"""Stub replacement for the deleted C++ ``_biteopt`` extension.

Provides the BiteOpt derivative-free global optimizer used by
``scipy.optimize._biteopt_py``.
This is a stub: real BiteOpt should be ported to Rust under ``rust/src/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['minimize']


def minimize(func, x0, args=(), bounds=None, maxiter=1000, **kwargs):
    """Stub for BiteOpt minimize."""
    return x0.copy(), func(x0.copy(), *args), 0
