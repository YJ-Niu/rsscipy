"""Stub replacement for the deleted C ``_propack`` extension.

Provides PROPACK SVD solver wrappers used by ``scipy.sparse.linalg._svdp``.
This is a stub: real PROPACK bindings should be ported to Rust under
``rust/src/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['slansvd', 'dlansvd', 'clansvd', 'zlansvd',
           'slansvd_irl', 'dlansvd_irl', 'clansvd_irl', 'zlansvd_irl']


def _stub(*args, **kwargs):
    raise NotImplementedError("_propack not yet ported to Rust/rsnumpy")


for _name in __all__:
    globals()[_name] = _stub
