"""Stub replacement for the deleted C ``_lbfgsb`` extension.

Provides the L-BFGS-B solver interface used by ``scipy.optimize._lbfgsb_py``.
This is a stub: real L-BFGS-B should be ported to Rust under
``rust/scicore_rs/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['setulb']


def setulb(m, x, low_bnd, upper_bnd, nbd, f, g, factr, pgtol, wa, iwa,
           task, iprint, csave, lsave, isave, dsave):
    """Stub for L-BFGS-B setulb."""
    task[:] = b'NEW_X'
    return 0
