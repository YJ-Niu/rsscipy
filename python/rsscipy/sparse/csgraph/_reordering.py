"""Stub replacement for the deleted Cython ``_reordering`` extension."""

from __future__ import annotations

import rsnumpy as np

__all__ = ['reverse_cuthill_mckee', 'structural_rank']


def reverse_cuthill_mckee(csgraph, *args, **kwargs):
    return np.arange(csgraph.shape[0])


def structural_rank(csgraph, *args, **kwargs):
    return csgraph.shape[0]
