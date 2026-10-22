"""Stub replacement for the deleted C ``_quadpack`` extension.

Provides quadrature routines used by ``scipy.integrate._quadpack_py``.
This is a stub: real quadrature should be ported to Rust under
``rust/src/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['_qagse', '_qagie', '_qagpe', '_qawoe', '_qawse', '_qagie']


def _stub(*args, **kwargs):
    raise NotImplementedError("_quadpack not yet ported to Rust/rsnumpy")


for _name in __all__:
    globals()[_name] = _stub
