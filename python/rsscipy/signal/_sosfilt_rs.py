"""Python wrapper for the Rust `_sosfilt_rs` extension module.

Replaces the original `_sosfilt.pyx`. The Rust module operates on raw f64
buffers and modifies `x` and `zi` in place.
"""

import rsnumpy as np

from ._sosfilt_rs import _sosfilt_float

__all__ = ['_sosfilt']


def _sosfilt(sos, x, zi):
    """Second-order sections filter.

    Parameters
    ----------
    sos : ndarray, shape (n_sections, 6)
        Second-order sections coefficients.
    x : ndarray, shape (n_signals, n_samples)
        Input signal. Modified in place.
    zi : ndarray, shape (n_signals, n_sections, 2)
        Initial conditions. Modified in place.
    """
    _sosfilt_float(sos, x, zi)
