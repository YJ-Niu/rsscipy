"""Stub replacement for the deleted C ``_zeros`` extension.

Provides root-finding (zeros) routines used by ``scipy.optimize``.
This is a stub: real zero-finding should be ported to Rust under
``rust/scicore_rs/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['brentq', 'brenth', 'ridder', 'bisect']


def brentq(f, a, b, args=(), xtol=2e-12, rtol=8.88e-16, maxiter=100,
           full_output=False):
    """Stub for Brent's method."""
    return 0.0


def brenth(f, a, b, args=(), xtol=2e-12, rtol=8.88e-16, maxiter=100,
           full_output=False):
    """Stub for Brent's method (h variant)."""
    return 0.0


def ridder(f, a, b, args=(), xtol=2e-12, rtol=8.88e-16, maxiter=100,
           full_output=False):
    """Stub for Ridder's method."""
    return 0.0


def bisect(f, a, b, args=(), xtol=2e-12, rtol=8.88e-16, maxiter=100,
           full_output=False):
    """Stub for bisection method."""
    return 0.0
