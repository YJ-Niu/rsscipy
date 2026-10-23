"""Stub replacement for the deleted C ``_minpack`` extension.

Provides the MINPACK least-squares solver interface used by
``scipy.optimize._minpack_py``.
This is a stub: real MINPACK should be ported to Rust under ``rust/src/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['hybrj', 'hybrd']


def hybrj(fcn, x, fvec, xtol, maxfev, ml, mu, epsfcn, diag, mode, factor,
          nprint, info, nfev, njev, fjac, r, qtf, wa1, wa2, wa3, wa4):
    """Stub for MINPACK hybrj."""
    info[0] = 1
    return 0


def hybrd(fcn, x, fvec, xtol, maxfev, ml, mu, epsfcn, diag, mode, factor,
          nprint, info, nfev, njev, fjac, r, qtf, wa1, wa2, wa3, wa4):
    """Stub for MINPACK hybrd."""
    info[0] = 1
    return 0
