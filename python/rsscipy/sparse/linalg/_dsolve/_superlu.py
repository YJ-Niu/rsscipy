"""Stub replacement for the deleted C ``_superlu`` extension.

Provides SuperLU sparse direct solver interface used by
``scipy.sparse.linalg._dsolve.linsolve``.
This is a stub: real sparse direct solve should be ported to Rust under
``rust/rsscipy_rs/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['gssv', 'gstrf', 'gstrs', 'SuperLU']


def gssv(N, nnz, a, ia, ja, nrhs, b, ldb, perm_c, perm_r, etree, info):
    """Stub for SuperLU gssv (solve sparse linear system)."""
    # Return dummy solution
    x = np.zeros(nrhs)
    info[0] = 0
    return x, info


def gstrf(M, N, nnz, a, ia, ja, perm_c, perm_r, etree, info, L, U, work):
    """Stub for SuperLU gstrf (factorize sparse matrix)."""
    info[0] = 0


def gstrs(trans, L, U, perm_c, perm_r, nrhs, b, ldb, info, work):
    """Stub for SuperLU gstrs (solve factored system)."""
    info[0] = 0


class SuperLU:
    """Stub SuperLU object."""

    def __init__(self, A):
        self.A = A
        self.L = None
        self.U = None
        self.perm_c = None
        self.perm_r = None

    def solve(self, b):
        """Stub solve."""
        return np.zeros_like(b)

    def solve_transpose(self, b):
        """Stub solve_transpose."""
        return np.zeros_like(b)
