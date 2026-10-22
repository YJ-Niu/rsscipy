"""Stub replacement for the deleted C ``_lsap`` extension.

Provides linear sum assignment (LSAP) solver used by ``scipy.optimize``.
This is a stub: real LSAP should be ported to Rust under ``rust/rsscipy_rs/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['linear_sum_assignment']


def linear_sum_assignment(cost_matrix, maximize=False):
    """Stub for linear sum assignment."""
    n, m = cost_matrix.shape
    return np.arange(min(n, m)), np.arange(min(n, m))
