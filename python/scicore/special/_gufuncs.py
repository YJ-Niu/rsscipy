"""Stub replacement for the deleted Cython ``_gufuncs`` extension.

Provides generalized ufuncs used by ``scipy.special`` and other modules,
implemented on top of :mod:`rsnumpy`.  This is a stub: real special-function
implementations should later be ported to Rust under ``rust/src/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = []


class _gufunc(np.ufunc):
    """A thin generalized-ufunc wrapper around a scalar function."""

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

    def resolve_dtypes(self, dtypes=None, *args, **kwargs):
        raise NotImplementedError(
            "rsnumpy gufunc does not expose a NumPy ufunc C-API object")

    def __call__(self, *args, **kwargs):
        import rsnumpy as _np
        if len(args) == 1 and isinstance(args[0], _np.ndarray):
            arr = args[0]
            flat = list(arr.tolist())
            result = [self._func(float(x)) for x in flat]
            out = _np.array(result, dtype=float).reshape(arr.shape)
            return out
        vals = [float(a) for a in args]
        result = self._func(*vals)
        if self.nout == 1:
            return result
        return tuple(result)

    @property
    def __name__(self):
        return self.name

    def __repr__(self):
        return f"<rsnumpy gufunc '{self.name}'>"


def _stub(*args, **kwargs):
    return float('nan')


# Functions imported by other SciPy modules
_for_stub = [
    # from _spfun_stats.py
    '_poisson_binom_pmf', '_poisson_binom_cdf',
    # from _multiufuncs.py
    'legendre_p_all', 'assoc_legendre_p_all', 'sph_legendre_p_all',
    'sph_harm_y_all',
    # from _basic.py
    '_lqn', '_lqmn', '_rctj', '_rcty',
    # other commonly referenced gufuncs
    'log_ndtr', 'ndtri_exp', 'xlogy', 'xlog1py', 'kl_div', 'rel_entr',
    'entr', 'pseudo_huber', 'huber',
]
for _name in _for_stub:
    globals()[_name] = _gufunc(_name, _stub)
