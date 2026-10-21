"""Stub replacement for the deleted Cython ``_ansari_swilk_statistics`` extension.

Provides Ansari-Bradley statistic helpers used by ``scipy.stats``.
This is a stub: real implementations should be ported to Rust under
``rust/rsscipy_rs/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['ansari_swilk_statistic', 'gscale']


def ansari_swilk_statistic(x, y):
    """Stub for Ansari-Bradley statistic."""
    return 0.0, 1.0


def gscale(n, m):
    """Stub for gscale."""
    return n, m, 0

