"""Stub replacement for the deleted C ``_slsqplib`` extension.

Provides the SLSQP solver interface used by ``scipy.optimize._slsqp``.
This is a stub: real SLSQP should be ported to Rust under ``rust/src/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['slsqp', 'nnls']


def slsqp(func, x0, args=(), jac=None, bounds=None, constraints=(),
          maxiter=100, ftol=1e-6, disp=False, **kwargs):
    """Stub for SLSQP."""
    raise NotImplementedError("_slsqplib not yet ported to Rust/rsnumpy")


def nnls(A, b, maxiter=None):
    """Stub for NNLS (non-negative least squares)."""
    return np.zeros(b.shape), 0
