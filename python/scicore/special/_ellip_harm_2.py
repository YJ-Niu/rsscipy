"""Stub replacement for the deleted Cython ``_ellip_harm_2`` extension.

Provides ellipsoidal harmonic helper functions used by ``scipy.special._ellip_harm``.
This is a stub.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = []


def _ellipsoid(h2, k2, n, p, s, signm=1, signn=1):
    return 0.0


def _ellipsoid_norm(h2, k2, n):
    return 1.0
