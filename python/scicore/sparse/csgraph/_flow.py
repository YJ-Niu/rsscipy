"""Stub replacement for the deleted Cython ``_flow`` extension."""

from __future__ import annotations

import rsnumpy as np

__all__ = ['maximum_flow']


def maximum_flow(csgraph, source, sink, *args, **kwargs):
    return 0.0, np.zeros_like(csgraph)
