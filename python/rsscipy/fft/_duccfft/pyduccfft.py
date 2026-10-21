"""Stub replacement for the deleted C ``pyduccfft`` extension.

Provides a minimal FFT backend interface used by ``scipy.fft._duccfft``.
This is a stub: real FFT should be ported to Rust under ``rust/rsscipy_rs/``.
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


def good_size(n):
    return int(n)


def next_fast_len(n):
    return int(n)


def prev_good_size(n):
    return int(n)


def r2r_fftpack(*args, **kwargs):
    raise NotImplementedError("r2r_fftpack not yet ported")


for _name in __all__:
    if _name not in ('good_size', 'next_fast_len', 'prev_good_size', 'r2r_fftpack'):
        globals()[_name] = _stub
