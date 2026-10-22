"""Apply Givens rotations to eliminate a diagonal block."""

import rsnumpy as np

from rsscipy._rsscipy_rs import linalg as _rust_linalg


def givens_elimination(S, v, diag):
    """Zero a diagonal block using a sequence of Givens rotations in place."""
    n = diag.shape[0]
    if S.ndim != 2 or S.shape != (n, n):
        raise ValueError("S must be a square matrix matching diag")
    if v.ndim != 1 or v.shape[0] != n:
        raise ValueError("v must be a vector matching diag")

    matrix, vector = _rust_linalg._givens_elimination(
        S.reshape((-1,)).tolist(),
        v.tolist(),
        diag.tolist(),
        n,
    )
    S[...] = np.asarray(matrix, dtype=S.dtype).reshape(S.shape)
    v[...] = np.asarray(vector, dtype=v.dtype)
