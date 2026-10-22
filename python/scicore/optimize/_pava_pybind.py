"""Stub replacement for the deleted C++ ``_pava_pybind`` extension.

Provides PAVA (pool adjacent violators algorithm) used by ``scipy.optimize``.
This is a stub: real PAVA should be ported to Rust under ``rust/src/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['pava']


def pava(x, y=None):
    """Stub for PAVA."""
    return x.copy()
