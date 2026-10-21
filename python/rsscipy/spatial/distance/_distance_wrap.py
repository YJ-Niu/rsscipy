"""Stub replacement for the deleted C ``_distance_wrap`` extension.

Provides distance wrapper functions used by ``scipy.spatial.distance._distance``.
This is a stub.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = []

# Wrappers that call into _distance_pybind - not needed in stub since
# _distance_pybind stubs already raise NotImplementedError.
