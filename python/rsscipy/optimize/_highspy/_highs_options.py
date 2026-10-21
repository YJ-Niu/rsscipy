"""Stub replacement for the deleted C++ ``_highs_options`` extension.

Provides HiGHS options used by ``scipy.optimize._highspy``.
This is a stub.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = []

class Options:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
