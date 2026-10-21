"""Stub replacement for the deleted C ``_hausdorff`` extension."""

from __future__ import annotations

import rsnumpy as np

__all__ = ['hausdorff_distance']


def hausdorff_distance(u, v):
    """Stub: Hausdorff distance."""
    return float(np.max(np.abs(u - v)))
