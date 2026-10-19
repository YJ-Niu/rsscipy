"""Python adapter for the Rust second-order-sections filtering kernel."""

import rsnumpy as np

from ._sosfilt_rs import _sosfilt_complex, _sosfilt_float

__all__ = ['_sosfilt']


def _sosfilt(sos, x, zi):
    if x.dtype.kind == 'c':
        x_real, x_imag, zi_real, zi_imag = _sosfilt_complex(
            np.reshape(sos, (-1,)).tolist(),
            np.reshape(x, (-1,)).tolist(),
            np.reshape(zi, (-1,)).tolist(),
            sos.shape[0],
            x.shape[0],
            x.shape[1],
        )
        x_result = (
            np.asarray(x_real).reshape(x.shape)
            + 1j * np.asarray(x_imag).reshape(x.shape)
        )
        zi_result = (
            np.asarray(zi_real).reshape(zi.shape)
            + 1j * np.asarray(zi_imag).reshape(zi.shape)
        )
    else:
        x_result, zi_result = _sosfilt_float(sos, x, zi)
    x[...] = np.asarray(x_result, dtype=x.dtype).reshape(x.shape)
    zi[...] = np.asarray(zi_result, dtype=zi.dtype).reshape(zi.shape)
