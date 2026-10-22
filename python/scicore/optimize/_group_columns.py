"""Rust-backed column grouping for sparse finite-difference Jacobians."""

import rsnumpy as np

from scicore._scicore_rs import optimize as _rust_optimize


def group_dense(m, n, A):
    matrix = np.asarray(A)
    if matrix.shape != (m, n):
        raise ValueError("matrix shape does not match dimensions")
    positive = (matrix > 0).tolist()
    return np.asarray(_rust_optimize._group_dense(positive, m, n), dtype=np.intp)


def group_sparse(m, n, indices, indptr):
    groups = _rust_optimize._group_sparse(indices.tolist(), indptr.tolist(), m, n)
    return np.asarray(groups, dtype=np.intp)
