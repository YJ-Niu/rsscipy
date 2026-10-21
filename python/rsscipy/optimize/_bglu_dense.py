"""Stub replacement for the deleted Cython ``_bglu_dense`` extension.

Provides BGLU (blocked Gaussian elimination) dense factorization used by
``scipy.optimize._linprog_rs``.
This is a stub: real BGLU should be ported to Rust under ``rust/rsscipy_rs/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['LU', 'BGLU']


class LU:
    """Stub LU factorization."""

    def __init__(self, A):
        self.A = A

    def solve(self, b):
        return np.linalg.solve(self.A, b)


class BGLU:
    """Stub BGLU factorization."""

    def __init__(self, A):
        self.A = A

    def solve(self, b):
        return np.linalg.solve(self.A, b)
