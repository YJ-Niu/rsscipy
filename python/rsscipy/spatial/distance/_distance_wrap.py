"""Pure-Python replacement for the deleted C ``_distance_wrap`` extension.

Provides the small set of wrappers used by
``rsscipy.spatial.distance._distance``: cosine double wraps (called with a
pre-allocated ``dm`` to fill) and squareform vector conversions.
"""

from __future__ import annotations

import rsnumpy as np

from . import _distance_pybind


def cdist_cosine_double_wrap(XA, XB, dm, **kwargs):
    dm[:] = _distance_pybind.cdist_cosine(XA, XB, **kwargs)


def pdist_cosine_double_wrap(X, dm, **kwargs):
    dm[:] = _distance_pybind.pdist_cosine(X, **kwargs)


# ---------------------------------------------------------------------------
# seuclidean / mahalanobis / jensenshannon pairwise kernels
# ---------------------------------------------------------------------------

def _condense(D):
    """Upper-triangular condensed vector of the square matrix ``D``."""
    m = D.shape[0]
    parts = [D[i, i + 1:] for i in range(m - 1)]
    if not parts:
        return np.zeros(0, dtype='float64')
    return np.concatenate(parts)


def _seuclidean(A, B, V):
    d = A[:, None, :] - B[None, :, :]
    return np.sqrt((d * d / V).sum(axis=-1))


def _mahalanobis(A, B, VI):
    VI = np.asarray(VI, dtype='float64')
    n = A.shape[1]
    if VI.ndim == 1:
        if VI.shape[0] != n * n:
            raise ValueError('VI must be an (n, n) inverse covariance matrix '
                             'or its flat (n * n,) form.')
        VI = VI.reshape(n, n)
    d = A[:, None, :] - B[None, :, :]
    acc = np.zeros(d.shape[:2], dtype='float64')
    for i in range(n):
        acc = acc + d[..., i] * (d * VI[i]).sum(axis=-1)
    return np.sqrt(acc)


def _jensenshannon(P, Q):
    # Row-wise L1 normalization, then the symmetric KL formula with the
    # natural logarithm (matching the deleted C kernel; ``base`` is not
    # supported by cdist/pdist, mirroring upstream SciPy).
    ps = P.sum(axis=-1)
    qs = Q.sum(axis=-1)
    p = P / np.where(ps > 0, ps, 1.0)[:, None]
    q = Q / np.where(qs > 0, qs, 1.0)[:, None]
    pb = p[:, None, :]
    qb = q[None, :, :]
    m = (pb + qb) / 2.0
    safe = np.where(m > 0, m, 1.0)
    left = np.where(pb > 0, pb * np.log(np.where(pb > 0, pb / safe, 1.0)), 0.0)
    right = np.where(qb > 0, qb * np.log(np.where(qb > 0, qb / safe, 1.0)), 0.0)
    D = np.sqrt((left + right).sum(axis=-1) / 2.0)
    bad_p = (ps <= 0) | (P.min(axis=-1) < 0)
    bad_q = (qs <= 0) | (Q.min(axis=-1) < 0)
    return np.where(bad_p[:, None] | bad_q[None, :], np.inf, D)


def cdist_seuclidean_double_wrap(XA, XB, dm, V):
    dm[:] = _seuclidean(XA, XB, V)


def pdist_seuclidean_double_wrap(X, dm, V):
    dm[:] = _condense(_seuclidean(X, X, V))


def cdist_mahalanobis_double_wrap(XA, XB, dm, VI):
    dm[:] = _mahalanobis(XA, XB, VI)


def pdist_mahalanobis_double_wrap(X, dm, VI):
    dm[:] = _condense(_mahalanobis(X, X, VI))


def cdist_jensenshannon_double_wrap(XA, XB, dm):
    dm[:] = _jensenshannon(XA, XB)


def pdist_jensenshannon_double_wrap(X, dm):
    dm[:] = _condense(_jensenshannon(X, X))


def to_squareform_from_vector_wrap(M, X):
    """Fill the square matrix ``M`` from the condensed vector ``X``."""
    m = M.shape[0]
    k = 0
    for i in range(m):
        for j in range(i + 1, m):
            M[i, j] = X[k]
            M[j, i] = X[k]
            k += 1


def to_vector_from_squareform_wrap(X, v):
    """Extract the condensed vector from the upper triangle of ``X`` into ``v``."""
    m = X.shape[0]
    k = 0
    for i in range(m):
        for j in range(i + 1, m):
            v[k] = X[i, j]
            k += 1
