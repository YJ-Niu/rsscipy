"""Stub replacement for the deleted C ``_odepack`` extension.

Provides the ODEPACK solver interface used by ``scipy.integrate._odepack_py``.
This is a stub: real ODE solvers should be ported to Rust under
``rust/src/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['odeint', 'ODEintWarning', 'lsoda', 'zvode', 'dvode']


class ODEintWarning(Warning):
    pass


def odeint(func, y0, t, args=(), Dfun=None, col_deriv=0, ml=None, mu=None,
           rtol=None, atol=None, tcrit=None, h0=0.0, hmax=0.0, hmin=0.0,
           ixpr=0, mxstep=0, mxhnil=0, mxordn=12, mxords=5, printmessg=0,
           tfirst=False):
    """Stub for odeint. Returns y0 repeated for each time point."""
    y0 = np.asarray(y0, dtype=float)
    out = np.zeros((len(t),) + y0.shape)
    for i, _ in enumerate(t):
        out[i] = y0
    return out


def lsoda(*args, **kwargs):
    """Stub for lsoda."""
    raise NotImplementedError("lsoda not yet ported to Rust/rsnumpy")


def zvode(*args, **kwargs):
    """Stub for zvode."""
    raise NotImplementedError("zvode not yet ported to Rust/rsnumpy")


def dvode(*args, **kwargs):
    """Stub for dvode."""
    raise NotImplementedError("dvode not yet ported to Rust/rsnumpy")
