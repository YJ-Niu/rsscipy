"""Stub replacement for the deleted Cython ``_specfun`` extension.

Provides Bessel/Zorn special-function helpers used by ``scipy.special._basic``,
implemented on top of :mod:`rsnumpy`.  This is a stub.
"""

from __future__ import annotations

import math
import rsnumpy as np

__all__ = []


def _stub(*args, **kwargs):
    return float('nan')


# Functions imported by _basic.py
_jdzo = _stub
_jyzo = _stub
_cyzo = _stub
_cerzo = _stub
_fcszo = _stub

globals().update({
    'jdzo': _jdzo,
    'jyzo': _jyzo,
    'cyzo': _cyzo,
    'cerzo': _cerzo,
    'fcszo': _fcszo,
})
