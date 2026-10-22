"""Python adapter for the Rust one-dimensional rank filter."""

import rsnumpy as np

from rsscipy._rsscipy_rs import ndimage as _rust_ndimage


def rank_filter(input, rank, window, output, mode, cval, origin):
    input = np.asarray(input)
    output = np.asarray(output)
    if input.ndim != 1 or output.shape != input.shape:
        raise ValueError("rank filter input and output must be matching 1-D arrays")

    if input.dtype.kind == "f":
        values = _rust_ndimage._rank_filter_float(
            input.astype(np.float64).tolist(),
            rank,
            window,
            mode,
            float(cval),
            origin,
        )
        output[...] = np.asarray(values, dtype=output.dtype)
    elif input.dtype.kind in "iu":
        values = _rust_ndimage._rank_filter_int(
            input.astype(np.int64).tolist(),
            rank,
            window,
            mode,
            int(cval),
            origin,
        )
        output[...] = np.asarray(values, dtype=output.dtype)
    else:
        raise TypeError("Unsupported array type")
