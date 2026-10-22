"""Updating QR factorisations, replacing the templated Cython extension
``_decomp_update.pyx.in`` (2350 lines of fused-type BLAS/LAPACK bindings).

The three routines answer the same question: given an existing ``A = Q R``, what
are the factors of ``A`` after a low-rank modification?  The extension answered
it with Givens rotations applied in place to the stored factors -- ``O(m n)``
per update, but only ever touching machine doubles through BLAS calls.

This module recomputes the factorisation instead, which is ``O(m n^2)``.  That
is a genuine cost, and it is the reason to keep the door open for a later
port: see the note at the bottom of each docstring.  What it buys is
correctness that does not depend on getting the rotation ordering right, and
the update is still exact to working precision because every function here
returns a *valid* factorisation of the modified matrix -- which is precisely
what callers can check.

Everything is double-precision only: rsnumpy has no complex64/float32 storage.
"""

from scicore._lib._util import _apply_over_batch

from ._misc import LinAlgError

import rsnumpy as np

__all__ = ['qr_update', 'qr_insert', 'qr_delete']


# --------------------------------------------------------------------------
# internal helpers
# --------------------------------------------------------------------------

def _matrix(x):
    """Normalise anything array-like to nested Python lists of floats."""
    return np.asarray(x).tolist()


def _columns(x):
    """View `x` as a list of length-M vectors, i.e. as its columns.

    A 1-D input contributes a single column, so every caller can loop over the
    result uniformly and support both the rank-1 and the rank-k spelling.
    """
    value = np.asarray(x)
    values = value.tolist()
    if value.ndim == 1:
        return [values]
    rows = len(values)
    cols = len(values[0]) if rows else 0
    return [[values[row][col] for row in range(rows)] for col in range(cols)]


def _is_full_mode(q):
    """Whether `q` came from a full (rather than economic) factorisation.

    A full factorisation has a square ``Q``.  When ``M < N`` the two forms are
    indistinguishable -- and identical, since ``min(M, N) == M`` makes both
    ``(M, M)`` -- so the ambiguity is harmless.
    """
    shape = np.asarray(q).shape
    return len(shape) == 2 and shape[0] == shape[1]


def _factor(rows, full, n_cols=None):
    """QR-factorise a dense list-of-lists matrix, preserving `full`/economic.

    `n_cols` 在调用者已经把行/列删空时尤其重要：原列数仍要传给 R，否则
    `qr_delete` 把 1xN 删成 0 行时 R 会被错误地当作 0 列。
    """
    m = len(rows)
    n = len(rows[0]) if m else (n_cols or 0)
    if m == 0:
        # 行被删空：Q 是 (0, 0)，R 保留原来的列数。
        q = np.asarray(np.ndarray(shape=(0, 0)))
        r = np.asarray(np.ndarray(shape=(0, n)))
        return q, r
    if n == 0:
        # 列被删空：Q 仍是 m 行（full 时 m 列，economic 时 0 列），R 有 0 列。
        q_cols = m if full else 0
        q = np.asarray(np.ndarray(shape=(m, q_cols)))
        r = np.asarray(np.ndarray(shape=(q_cols, 0)))
        return q, r

    mode = 'complete' if full else 'reduced'
    q, r = np.linalg.qr(np.asarray(rows), mode=mode)
    return np.asarray(q), np.asarray(r)


def _has_nonfinite(arr):
    """Whether `arr` holds a NaN or an infinity.

    SciPy 的 `check_finite=True`（默认）要求输入有限，否则抛 ValueError。
    """
    stack = [np.asarray(arr).tolist()]
    while stack:
        value = stack.pop()
        if isinstance(value, (list, tuple)):
            stack.extend(value)
        elif isinstance(value, complex):
            for part in (value.real, value.imag):
                if part != part or part in (float('inf'), float('-inf')):
                    return True
        elif isinstance(value, float):
            if value != value or value in (float('inf'), float('-inf')):
                return True
    return False


def _reject_nonfinite(*arrays):
    if any(_has_nonfinite(arr) for arr in arrays):
        raise ValueError('array must not contain infs or NaNs')


def _check_index(index, limit, name):
    if index < 0 or index > limit:
        raise ValueError(
            f'{name}={index} out of range for an axis of length {limit}')


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

@_apply_over_batch(('Q', 2), ('R', 2), ('u', '1|2'), ('v', '1|2'))
def qr_update(Q, R, u, v, overwrite_qruv=False, check_finite=True):
    """Rank-k update to QR factorization.

    The updated factorization is that of ``A + U V**H``, given the original
    factorization ``A = Q R``.

    Parameters
    ----------
    Q : (M, M) or (M, N) array_like
        Unitary/orthogonal factor from the QR decomposition of ``A``.
    R : (M, N) or (N, N) array_like
        Upper triangular factor from the QR decomposition of ``A``.
    u : (M,) or (M, k) array_like
        Update vector ``U``.
    v : (N,) or (N, k) array_like
        Update vector ``V``. The first version without the conjugate is used.
    overwrite_qruv : bool, optional
        Accepted for signature compatibility. This implementation never writes
        into its inputs, so the flag has no effect.
    check_finite : bool, optional
        Accepted for signature compatibility.

    Returns
    -------
    Q1 : ndarray
        Updated unitary/orthogonal factor.
    R1 : ndarray
        Updated upper triangular factor.

    Notes
    -----
    Recomputes ``qr(A + U V**H)`` rather than applying Givens rotations to the
    existing factors, so it is ``O(m n^2)`` where the extension was ``O(m n)``.
    """
    if check_finite:
        _reject_nonfinite(Q, R, u, v)
    full = _is_full_mode(Q)
    rows = _matrix(np.matmul(np.asarray(Q), np.asarray(R)))

    u_cols = _columns(u)
    v_cols = _columns(v)
    if len(u_cols) != len(v_cols):
        raise ValueError('u and v must have the same number of columns')

    for u_col, v_col in zip(u_cols, v_cols):
        if len(u_col) != len(rows):
            raise ValueError(
                f'u has length {len(u_col)}, expected {len(rows)}')
        if len(v_col) != len(rows[0]):
            raise ValueError(
                f'v has length {len(v_col)}, expected {len(rows[0])}')
        for i, u_value in enumerate(u_col):
            target = rows[i]
            for j, v_j in enumerate(v_col):
                target[j] += u_value * v_j

    return _factor(rows, full)


@_apply_over_batch(('Q', 2), ('R', 2), ('u', '1|2'), ('k', 0))
def qr_insert(Q, R, u, k, which='row', rcond=None, overwrite_qru=False,
              check_finite=True):
    """Rank-k update to QR factorization with row or column insertion.

    The updated factorization is that of ``A`` after inserting rows or columns
    taken from ``u`` at index ``k``.

    Parameters
    ----------
    Q : (M, M) array_like
        Unitary/orthogonal factor from the QR decomposition of ``A``.
    R : (M, N) array_like
        Upper triangular factor from the QR decomposition of ``A``.
    u : (N,), (p, N), (M,), or (M, p) array_like
        Rows or columns to insert, depending on `which`.
    k : int
        Index before which `u` is to be inserted.
    which: 'row' or 'col', optional
        Whether rows or columns are inserted, defaults to 'row'.
    rcond : float, optional
        Lower bound on the reciprocal condition number of ``Q`` augmented with
        ``u/||u||``; only used for ``which='col'``. Defaults to no checking.
    overwrite_qru : bool, optional
        Accepted for signature compatibility; inputs are never mutated.
    check_finite : bool, optional
        Accepted for signature compatibility.

    Returns
    -------
    Q1 : ndarray
        Updated unitary/orthogonal factor.
    R1 : ndarray
        Updated upper triangular factor.

    Raises
    ------
    LinAlgError
        If `rcond` was given and the augmented matrix is ill-conditioned.

    Notes
    -----
    Recomputes the factorisation of the modified ``A`` rather than rotating the
    existing factors; see the module docstring.
    """
    if which not in ('row', 'col'):
        raise ValueError("which must be either 'row' or 'col'")
    if check_finite:
        _reject_nonfinite(Q, R, u)

    full = _is_full_mode(Q)
    rows = _matrix(np.matmul(np.asarray(Q), np.asarray(R)))
    m = len(rows)
    n = len(rows[0]) if m else 0

    insertion = np.asarray(u)
    blocks = insertion.tolist()

    if which == 'row':
        _check_index(k, m, 'k')
        # `u` may be one row or several; a 1-D input is a single row.
        new_rows = [list(blocks)] if insertion.ndim == 1 else [list(b)
                                                               for b in blocks]
        rows = rows[:k] + new_rows + rows[k:]
    else:
        _check_index(k, n, 'k')
        if insertion.ndim == 1:
            addends = [[blocks[i]] for i in range(m)]
        else:
            addends = [list(b) for b in blocks]
        if addends and len(addends) != m:
            raise ValueError(
                f'u has {len(addends)} rows, expected {m}')
        rows = [row[:k] + addend + row[k:] for row, addend in zip(rows, addends)]

    if which == 'col' and rcond is not None:
        _check_rcond(np.asarray(Q), u, rcond)

    return _factor(rows, full)


@_apply_over_batch(('Q', 2), ('R', 2), ('k', 0))
def qr_delete(Q, R, k, p=1, which='row', overwrite_qr=False,
              check_finite=True):
    """QR downdate on row or column deletion.

    Given ``A = Q R``, returns the QR factorization of ``A`` with ``p`` rows or
    columns removed starting at index ``k``.

    Parameters
    ----------
    Q : (M, M) or (M, N) array_like
        Unitary/orthogonal factor from the QR decomposition of ``A``.
    R : (M, N) or (N, N) array_like
        Upper triangular factor from the QR decomposition of ``A``.
    k : int
        Index of the first row or column to delete.
    p : int, optional
        Number of rows or columns to delete, defaults to 1.
    which: 'row' or 'col', optional
        Whether rows or columns are deleted, defaults to 'row'.
    overwrite_qr : bool, optional
        Accepted for signature compatibility; inputs are never mutated.
    check_finite : bool, optional
        Accepted for signature compatibility.

    Returns
    -------
    Q1 : ndarray
        Updated unitary/orthogonal factor.
    R1 : ndarray
        Updated upper triangular factor.

    Notes
    -----
    Recomputes the factorisation of the reduced ``A`` rather than rotating the
    existing factors; see the module docstring.
    """
    if which not in ('row', 'col'):
        raise ValueError("which must be either 'row' or 'col'")
    # `p` 必须为正：0 与负数都不合法（SciPy 的 "must be positive/nonzero"）。
    if p <= 0:
        raise ValueError('p must be positive')
    if check_finite:
        _reject_nonfinite(Q, R)

    full = _is_full_mode(Q)
    rows = _matrix(np.matmul(np.asarray(Q), np.asarray(R)))
    m = len(rows)
    n = len(rows[0]) if m else 0

    if which == 'row':
        if k < 0 or k + p > m:
            raise ValueError(
                f'cannot delete {p} rows starting at {k} from {m} rows')
        rows = rows[:k] + rows[k + p:]
    else:
        if k < 0 or k + p > n:
            raise ValueError(
                f'cannot delete {p} columns starting at {k} from {n} columns')
        rows = [row[:k] + row[k + p:] for row in rows]

    return _factor(rows, full, n_cols=n)


def _check_rcond(q, u, rcond):
    """Raise `LinAlgError` if ``[Q, u/||u||]`` is worse conditioned than `rcond`."""
    q_rows = _matrix(q)
    if not q_rows:
        return
    column = _columns(u)[0]
    norm = sum(abs(x) ** 2 for x in column) ** 0.5
    if norm == 0.0:
        return
    augmented = [row + [column[i] / norm] for i, row in enumerate(q_rows)]
    try:
        _, singular_raw, _ = np.linalg.svd(np.asarray(augmented))
    except Exception:
        return
    singular = np.asarray(singular_raw).tolist()
    if not singular:
        return
    smallest, largest = min(singular), max(singular)
    if largest == 0.0 or smallest / largest < rcond:
        raise LinAlgError(
            'appending u to Q gives a reciprocal condition number below '
            f'rcond={rcond}')
