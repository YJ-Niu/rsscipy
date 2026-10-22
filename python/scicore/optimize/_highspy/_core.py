"""Stub replacement for the deleted C++ ``_core`` extension (HiGHS).

Provides the HiGHS LP solver interface used by ``scipy.optimize._highspy``.
This is a stub: real HiGHS bindings should be ported to Rust under
``rust/rsscipy_rs/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['highs', 'kHighsInf', 'kHighsMinusInf',
           'HighsDebugLevel', 'HighsBasisStatus', 'HighsModelStatus',
           'HighsBasisStatusEnum', 'HighsModelStatusEnum', 'ObjSense',
           'kMinimize', 'kMaximize', 'simplex_constants']

kHighsInf = float('inf')
kHighsMinusInf = float('-inf')

HighsDebugLevel = type('HighsDebugLevel', (), {'kNone': 0, 'kMinimal': 1, 'kFull': 2})()
HighsBasisStatus = type('HighsBasisStatus', (), {'kZero': 0, 'kBasic': 1, 'kNonBasic': 2, 'kSuperBasic': 3})()
HighsModelStatus = type('HighsModelStatus', (), {'kNotset': 0, 'kLoadError': 1, 'kModelError': 2, 'kOptimal': 3})()
HighsBasisStatusEnum = HighsBasisStatus
HighsModelStatusEnum = HighsModelStatus
ObjSense = type('ObjSense', (), {'kMinimize': 0, 'kMaximize': 1})()
kMinimize = 0
kMaximize = 1
simplex_constants = type('simplex_constants', (), {'kInf': float('inf'), 'kMinusInf': float('-inf')})()


def highs(*args, **kwargs):
    """Stub for HiGHS solver."""
    raise NotImplementedError("_core not yet ported to Rust/rsnumpy")
