"""Double-precision dense LAPACK interface used by `scipy.linalg.lapack`.

Replaces the C++ extension built from ``scipy/linalg/src/lapack/lapack_*.cpp``.
The LU family comes first because `lu`, `solve` and `inv` all sit on top of it.

Calling conventions mirror the wrappers they replace:

    dgetrf(a, overwrite_a=0)                       -> (lu, piv, info)
    dgetrs(lu, piv, b, trans=0, overwrite_b=0)     -> (x, info)
    dgetri(lu, piv, lwork=None, overwrite_lu=0)    -> (inv, info)
    dgetri_lwork(n)                                -> (work, info)

`piv` is **0-based** (SciPy's convention) even though LAPACK itself is 1-based;
the conversion happens in Rust.

Argument conversion lives in Python because it needs rsnumpy; the arithmetic
runs in Rust via the unified extension.

Only the double-precision real routines are offered.  rsnumpy stores every
array as float64 and rejects complex arrays outright, so ``s-``, ``c-`` and
``z-`` variants have no data they could operate on -- see the note in
``scipy.linalg._fblas``.
"""

import rsnumpy as np

from rsscipy._rsscipy_rs import linalg as _rust_linalg
getrf_f64 = _rust_linalg.getrf_f64
getri_f64 = _rust_linalg.getri_f64
getri_lwork_f64 = _rust_linalg.getri_lwork_f64
getrs_f64 = _rust_linalg.getrs_f64

__all__ = ['dgetrf', 'dgetrs', 'dgetri', 'dgetri_lwork']


def _matrix(a, name, square=False):
    """Return (raw float64 bytes, rows, cols) for a 2-D array argument."""
    arr = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    if arr.ndim != 2:
        raise ValueError(f'expected a 2-D array for {name!r}, got {arr.ndim}-D')
    rows, cols = arr.shape
    if square and rows != cols:
        raise ValueError(f'expected a square array for {name!r}, got {arr.shape}')
    return arr.__array_interface__['data'], rows, cols


def _pivots(piv, n, name='piv'):
    """Return pivots as a plain list of 0-based integers."""
    arr = np.ascontiguousarray(np.asarray(piv, dtype=np.int32)).reshape(-1)
    values = arr.tolist()
    if len(values) != n:
        raise ValueError(f'expected {n} pivots for {name!r}, got {len(values)}')
    return values


def _as_matrix(buffer, shape):
    return np.frombuffer(buffer, dtype=np.float64).reshape(shape)


def dgetrf(a, overwrite_a=0):
    """dgetrf(a, [overwrite_a]) -> (lu, piv, info): LU factorisation A = P L U."""
    data, m, n = _matrix(a, 'a')
    lu, piv, info = getrf_f64(data, m, n)
    return (
        _as_matrix(lu, (m, n)),
        np.ascontiguousarray(np.array(piv, dtype=np.int32)),
        info,
    )


def dgetrs(lu, piv, b, trans=0, overwrite_b=0):
    """dgetrs(lu, piv, b, [trans, overwrite_b]) -> (x, info): solve op(A) x = b."""
    if trans not in (0, 1, 2):
        raise ValueError(f'invalid trans={trans}, expected 0, 1 or 2')
    lu_data, n, n_cols = _matrix(lu, 'lu', square=True)
    if n_cols != n:
        raise ValueError('expected a square array for \'lu\'')
    pivots = _pivots(piv, n)

    arr = np.ascontiguousarray(np.asarray(b, dtype=np.float64))
    if arr.ndim != 2:
        raise ValueError(f'expected a 2-D array for \'b\', got {arr.ndim}-D')
    nrhs = arr.shape[1]
    if arr.shape[0] != n:
        raise ValueError(
            f'shapes of \'lu\' {(n, n)} and \'b\' {arr.shape} are incompatible'
        )

    x, info = getrs_f64(lu_data, n, pivots, arr.__array_interface__['data'],
                        nrhs, trans)
    return _as_matrix(x, (n, nrhs)), info


def dgetri(lu, piv, lwork=None, overwrite_lu=0):
    """dgetri(lu, piv, [lwork, overwrite_lu]) -> (inv, info): inverse from LU."""
    lu_data, n, _ = _matrix(lu, 'lu', square=True)
    pivots = _pivots(piv, n)

    if lwork is None:
        # SciPy's wrapper asks for 3*n unless the caller overrides it.
        lwork = max(3 * n, n)
    else:
        lwork = int(lwork)

    inv, info = getri_f64(lu_data, n, pivots, lwork)
    return _as_matrix(inv, (n, n)), info


def dgetri_lwork(n):
    """dgetri_lwork(n) -> (work, info): optimal workspace size query."""
    return getri_lwork_f64(int(n))
