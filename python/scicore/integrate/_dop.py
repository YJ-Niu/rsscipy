"""Stub replacement for the deleted C ``_dop`` extension.

Provides the DOP853 ODE solver interface used by ``scipy.integrate._ivp``.
This is a stub: real ODE solvers should be ported to Rust under
``rust/scicore_rs/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['dop853', 'dopri5', 'dopri853']


def _stub(*args, **kwargs):
    raise NotImplementedError("_dop not yet ported to Rust/rsnumpy")


for _name in __all__:
    globals()[_name] = _stub
