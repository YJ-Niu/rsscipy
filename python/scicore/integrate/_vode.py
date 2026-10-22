"""Stub replacement for the deleted C ``_vode`` extension.

Provides the VODE ODE solver interface used by ``scipy.integrate._ode``.
This is a stub: real ODE solvers should be ported to Rust under
``rust/src/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['dvode', 'zvode']


def _stub(*args, **kwargs):
    raise NotImplementedError("_vode not yet ported to Rust/rsnumpy")


for _name in __all__:
    globals()[_name] = _stub
