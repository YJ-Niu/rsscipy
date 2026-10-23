"""Level-3 BLAS interface used by `scipy.linalg.blas.get_blas_funcs`.

Replaces the C++ extension built from ``scipy/linalg/src/blas/blas_l*.cpp``.
Each routine below mirrors the (keyword-based) calling convention of the wrapper
it replaces, e.g.

    gemm(alpha, a, b, beta=0, c=None, trans_a=0, trans_b=0, overwrite_c=0) -> c

with ``trans_a``/``trans_b`` meaning ``0`` = no transpose, ``1`` = transpose,
``2`` = conjugate transpose.

Argument conversion -- forcing Fortran order and the right dtype, which
``blas_helpers.hpp`` did with ``NPY_ARRAY_F_CONTIGUOUS`` -- is kept in Python
because it needs rsnumpy.  The arithmetic then runs in Rust; see
the unified Rust extension.

Only a subset of the 149 routines in ``cython_blas_signatures.txt`` is provided
so far.  Everything else raises the same ``ValueError`` that
``blas.get_blas_funcs`` raises for a missing symbol, so callers degrade
predictably rather than silently.
"""

import rsnumpy as np

from scicore._scicore_rs import linalg as _rust_linalg
gemm_f32 = _rust_linalg.gemm_f32
gemm_f64 = _rust_linalg.gemm_f64

# NOTE: rsnumpy stores *every* numeric array as float64 internally, so a
# `dtype=float32` array still reports a `<f8` buffer.  There is therefore no
# single-precision data to feed `sgemm` with, and there is likewise no way to
# reach a complex buffer (`__array_interface__` rejects complex arrays).  Until
# rsnumpy grows those storage types this module can offer only the
# double-precision real kernel.
__all__ = ['dgemm']

# dtype -> (kernel, element type, bytes per element)
_KERNELS = {
    'f4': (gemm_f32, np.float32, 4),
    'f8': (gemm_f64, np.float64, 8),
}


def _prepare(value, dtype, ndim, name):
    """Return (contiguous buffer bytes, shape) for an array argument."""
    arr = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    if arr.ndim != ndim:
        raise ValueError(f'expected {ndim}-D array for {name}')
    return arr.__array_interface__['data'], tuple(arr.shape)


def _as_result(buffer, shape, dtype):
    """Wrap raw little-endian bytes back into an rsnumpy array."""
    return np.ascontiguousarray(np.frombuffer(buffer, dtype=dtype).reshape(shape))


def _gemm(typestr, alpha, a, b, beta=None, c=None, trans_a=0, trans_b=0,
          overwrite_c=0):
    kernel, dtype, esize = _KERNELS[typestr]

    a_buf, a_shape = _prepare(a, dtype, 2, 'a')
    b_buf, b_shape = _prepare(b, dtype, 2, 'b')

    if not (0 <= trans_a <= 2):
        raise ValueError(f'invalid trans_a={trans_a}')
    if not (0 <= trans_b <= 2):
        raise ValueError(f'invalid trans_b={trans_b}')

    if c is None:
        c_buf, c_shape = b'', None
    else:
        c_buf, c_shape = _prepare(c, dtype, 2, 'c')

    out = kernel(a_buf, a_shape, trans_a, b_buf, b_shape, trans_b,
                 float(alpha), 0.0 if beta is None else float(beta), c_buf)

    # Shape of the result, derived exactly as `blas_l3.cpp` does.
    m = a_shape[1] if trans_a else a_shape[0]
    n = b_shape[0] if trans_b else b_shape[1]
    result = _as_result(out, (m, n), dtype)

    m = a_shape[1] if trans_a else a_shape[0]
    n = b_shape[0] if trans_b else b_shape[1]
    return _as_result(out, (m, n), dtype)


def dgemm(alpha, a, b, beta=None, c=None, trans_a=0, trans_b=0, overwrite_c=0):
    """dgemm(alpha, a, b, [beta, c, offz, trans_a, trans_b, overwrite_c])"""
    return _gemm('f8', alpha, a, b, beta, c, trans_a, trans_b, overwrite_c)
