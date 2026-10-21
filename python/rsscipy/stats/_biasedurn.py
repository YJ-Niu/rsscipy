"""Stub replacement for the deleted Cython ``_biasedurn`` extension.

Provides biased urn distribution samplers used by ``scipy.stats``.
This is a stub: real samplers should be ported to Rust under
``rust/rsscipy_rs/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['_PyFishersNCHypergeometric', '_PyWalleniusNCHypergeometric',
           '_PyStochasticLib3']


class _PyFishersNCHypergeometric:
    def __init__(self, *args, **kwargs):
        pass


class _PyWalleniusNCHypergeometric:
    def __init__(self, *args, **kwargs):
        pass


class _PyStochasticLib3:
    def __init__(self, *args, **kwargs):
        pass
