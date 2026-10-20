"""Python adapter for the Rust two-dimensional convolution kernel."""

import rsnumpy as np

from rsscipy._rsscipy_rs import signal as _rust_signal

_convolve2d_rust = _rust_signal._convolve2d


def _convolve2d(in1, in2, flip, mode, boundary, fillvalue):
    dtype = np.result_type(in1, in2)
    in1 = np.asarray(in1).astype(dtype)
    in2 = np.asarray(in2).astype(dtype)

    if boundary == 0:
        fill = np.asarray(fillvalue)
        if fill.size == 0:
            raise ValueError("`fillvalue` cannot be an empty array.")
        if fill.size != 1:
            raise ValueError("`fillvalue` must be scalar or an array with one element.")
        if dtype.kind != 'c' and fill.dtype.kind == 'c':
            raise ValueError(
                "could not cast `fillvalue` directly to the output type"
            )
        fill = fill.reshape((-1,))[0]
    else:
        fill = 0

    real, imag, shape = _convolve2d_rust(
        np.reshape(in1, (-1,)).tolist(),
        np.reshape(in2, (-1,)).tolist(),
        (
            list(in1.shape),
            list(in2.shape),
            bool(flip),
            mode,
            boundary,
            fill,
        ),
    )
    result = np.asarray(real).astype(dtype)
    if dtype.kind == 'c':
        result = result + 1j * np.asarray(imag).astype(dtype)
    return result.astype(dtype).reshape(tuple(shape))
