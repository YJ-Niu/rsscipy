"""Python adapter for the Rust spline and IIR boundary kernels."""

import rsnumpy as np

from rsscipy._rsscipy_rs.signal import (
    _sepfir2d,
    _symiirorder1_ic,
    _symiirorder2_ic_bwd,
    _symiirorder2_ic_fwd,
)

__all__ = [
    'sepfir2d',
    'symiirorder1_ic',
    'symiirorder2_ic_fwd',
    'symiirorder2_ic_bwd',
]


def _kernel_dtype(array):
    if array.dtype.kind == 'c':
        return np.complex64 if array.dtype == np.complex64 else np.complex128
    if array.dtype.kind == 'f':
        return np.float32 if array.dtype == np.float32 else np.float64
    if array.dtype.kind == 'u' and array.dtype.itemsize == 1:
        return np.float32
    return np.float64


def _as_array(real, imag, dtype, shape):
    result = np.asarray(real).astype(dtype)
    if dtype.kind == 'c':
        result = result + 1j * np.asarray(imag).astype(dtype)
    return result.astype(dtype).reshape(shape)


def sepfir2d(input, hrow, hcol):
    input = np.asarray(input)
    if input.ndim < 2:
        raise ValueError("object of too small depth for desired array")
    if input.ndim > 2:
        raise ValueError("object too deep for desired array")

    dtype = _kernel_dtype(input)
    input = input.astype(dtype)
    dtype = input.dtype
    hrow = np.asarray(hrow)
    hcol = np.asarray(hcol)
    if hrow.ndim > 1 or hcol.ndim > 1:
        raise ValueError("object too deep for desired array")
    hrow = hrow.astype(dtype)
    hcol = hcol.astype(dtype)
    real, imag = _sepfir2d(
        np.reshape(input, (-1,)).tolist(),
        list(input.shape),
        hrow.tolist(),
        hcol.tolist(),
    )
    return _as_array(real, imag, dtype, input.shape)


def symiirorder1_ic(input, z1, precision=-1.0):
    input = np.asarray(input)
    dtype = _kernel_dtype(input)
    input = input.astype(dtype)
    dtype = input.dtype
    if input.ndim not in (1, 2):
        raise ValueError("input must be one- or two-dimensional")
    if input.ndim == 1:
        output_shape = (1, 1)
    else:
        output_shape = (input.shape[0], 1)
    real, imag = _symiirorder1_ic(
        np.reshape(input, (-1,)).tolist(),
        list(input.shape),
        z1,
        precision,
        dtype in (np.float32, np.complex64),
        dtype.kind == 'c',
    )
    return _as_array(real, imag, dtype, output_shape)


def _symiirorder2_args(input, precision):
    input = np.asarray(input)
    if input.dtype.kind == 'c':
        raise ValueError("Incorrect type.")
    dtype = _kernel_dtype(input)
    input = input.astype(dtype)
    if input.ndim not in (1, 2):
        raise ValueError("input must be one- or two-dimensional")
    rows = 1 if input.ndim == 1 else input.shape[0]
    return (
        input,
        np.reshape(input, (-1,)).tolist(),
        list(input.shape),
        rows,
        precision,
        dtype == np.float32,
    )


def symiirorder2_ic_fwd(input, r, omega, precision=-1.0):
    input, values, shape, rows, precision, single = _symiirorder2_args(
        input, precision
    )
    result = _symiirorder2_ic_fwd(values, shape, r, omega, precision, single)
    return np.asarray(result, dtype=input.dtype).reshape((rows, 2))


def symiirorder2_ic_bwd(input, r, omega, precision=-1.0):
    input, values, shape, rows, precision, single = _symiirorder2_args(
        input, precision
    )
    result = _symiirorder2_ic_bwd(values, shape, r, omega, precision, single)
    return np.asarray(result, dtype=input.dtype).reshape((rows, 2))
