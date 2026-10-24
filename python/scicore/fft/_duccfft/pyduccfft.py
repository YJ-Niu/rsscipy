"""Stub replacement for the deleted C ``pyduccfft`` extension.

Provides a minimal FFT backend interface used by ``scipy.fft._duccfft``.
This is a stub: real FFT should be ported to Rust under ``rust/src/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['dct', 'dst', 'idct', 'idst', 'dctn', 'dstn', 'idctn', 'idstn',
           'fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn', 'rfft', 'irfft',
           'rfft2', 'irfft2', 'rfftn', 'irfftn', 'hfft', 'ihfft',
           'hfft2', 'ihfft2', 'hfftn', 'ihfftn', 'good_size', 'next_fast_len',
           'prev_good_size', 'r2r_fftpack']


def _stub(*args, **kwargs):
    raise NotImplementedError("pyduccfft not yet ported to Rust/rsnumpy")


# Prime radices for which the reference FFT backend (pocketfft) has
# efficient algorithms. ``next_fast_len`` / ``prev_fast_len`` (exposed here as
# ``good_size`` / ``prev_good_size``) return the nearest length whose prime
# factors all lie in this set, matching ``scipy.fft.next_fast_len``: complex
# transforms use {2, 3, 5, 7, 11}; real transforms use {2, 3, 5}.
_COMPLEX_PRIMES = (2, 3, 5, 7, 11)
_REAL_PRIMES = (2, 3, 5)


def _is_fast_size(n, primes):
    """Return True if every prime factor of ``n`` is in ``primes``."""
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1


def good_size(n, real=False):
    """Smallest fast FFT length >= ``n`` (cf. ``scipy.fft.next_fast_len``)."""
    n = int(n)
    if n < 1:
        raise ValueError(f'next_fast_len requires positive input, got {n}')
    primes = _REAL_PRIMES if real else _COMPLEX_PRIMES
    m = n
    while not _is_fast_size(m, primes):
        m += 1
    return m


def prev_good_size(n, real=False):
    """Largest fast FFT length <= ``n`` (cf. ``scipy.fft.prev_fast_len``)."""
    n = int(n)
    if n < 1:
        raise ValueError(f'prev_fast_len requires positive input, got {n}')
    primes = _REAL_PRIMES if real else _COMPLEX_PRIMES
    m = n
    while m > 1 and not _is_fast_size(m, primes):
        m -= 1
    return m


def next_fast_len(n, real=False):
    return good_size(n, real)


def r2r_fftpack(*args, **kwargs):
    raise NotImplementedError("r2r_fftpack not yet ported")


for _name in __all__:
    if _name not in ('good_size', 'next_fast_len', 'prev_good_size', 'r2r_fftpack'):
        globals()[_name] = _stub
