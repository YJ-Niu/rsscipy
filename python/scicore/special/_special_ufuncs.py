"""Stub replacement for the deleted Cython ``_special_ufuncs`` extension.

Provides special functions used by ``scipy.special._multiufuncs`` and other
modules, implemented on top of :mod:`rsnumpy`.  This is a stub.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = []


def _stub(*args, **kwargs):
    return float('nan')


class _gufunc(np.ufunc):
    """A thin ufunc-like wrapper around a scalar function."""

    def __init__(self, name: str, func, nin: int = 1, nout: int = 1,
                 signature=None):
        self.name = name
        self._func = func
        self.nin = nin
        self.nout = nout
        self.signature = signature
        self.ntypes = 0
        self.identity = None
        self.nargs = nin
        self.types = []

    @property
    def __name__(self):
        return self.name

    def __repr__(self):
        return f"<rsnumpy ufunc '{self.name}'>"


# Functions imported by _multiufuncs.py
legendre_p = _gufunc('legendre_p', _stub)
assoc_legendre_p = _gufunc('assoc_legendre_p', _stub)
sph_legendre_p = _gufunc('sph_legendre_p', _stub)
sph_harm_y = _gufunc('sph_harm_y', _stub)

globals().update({
    'legendre_p': legendre_p,
    'assoc_legendre_p': assoc_legendre_p,
    'sph_legendre_p': sph_legendre_p,
    'sph_harm_y': sph_harm_y,
})
