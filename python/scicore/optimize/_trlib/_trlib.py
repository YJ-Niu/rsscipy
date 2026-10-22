"""Stub replacement for the deleted Cython ``_trlib`` extension.

Provides the TRLIB quadratic subproblem solver used by ``scipy.optimize._trlib``.
This is a stub: real trust-region subproblem solvers should be ported to Rust
under ``rust/src/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['TRLIBQuadraticSubproblem']


class TRLIBQuadraticSubproblem:
    """Stub trust-region quadratic subproblem solver."""

    def __init__(self, x, fun, jac, hess, hessp, tol_rel_i=-2.0,
                 tol_rel_b=-3.0, disp=False):
        self.x = x
        self.fun = fun
        self.jac = jac
        self.hess = hess
        self.hessp = hessp
        self.tol_rel_i = tol_rel_i
        self.tol_rel_b = tol_rel_b
        self.disp = disp

    def solve(self):
        """Stub solve - returns the initial point."""
        return self.x.copy(), 0.0, 0
