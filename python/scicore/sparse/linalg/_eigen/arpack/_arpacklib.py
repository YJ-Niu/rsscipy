"""Stub replacement for the deleted C ``_arpacklib`` extension.

Provides ARPACK eigenvalue solver wrappers used by
``scipy.sparse.linalg._eigen.arpack``.
This is a stub: real ARPACK bindings should be ported to Rust under
``rust/src/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = []


def _stub(*args, **kwargs):
    raise NotImplementedError("_arpacklib not yet ported to Rust/rsnumpy")


# ARPACK wrapper function names used by arpack.py
_wrap_funcs = []
for prefix in ('s', 'd', 'c', 'z'):
    for func in ('saupd_wrap', 'seupd_wrap', 'naupd_wrap', 'neupd_wrap',
                 'saup2_wrap', 'seup2_wrap', 'naup2_wrap', 'neup2_wrap',
                 'sgets_wrap', 'ngets_wrap'):
        _wrap_funcs.append(prefix + func)

for _name in _wrap_funcs:
    globals()[_name] = _stub
