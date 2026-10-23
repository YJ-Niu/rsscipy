"""Python adapter for the Rust two-dimensional median filter."""

import rsnumpy as np

from scicore._scicore_rs import signal as _rust_signal
_medfilt2d_rust = _rust_signal._medfilt2d

__all__ = ['_medfilt2d']


def _medfilt2d(image, size):
    image = np.asarray(image)
    values = _medfilt2d_rust(
        np.reshape(image, (-1,)).tolist(),
        list(image.shape),
        np.asarray(size).reshape((-1,)).tolist(),
    )
    return np.asarray(values).astype(image.dtype).reshape(image.shape)
