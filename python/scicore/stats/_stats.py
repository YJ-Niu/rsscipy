"""Stub replacement for the deleted Cython ``_stats`` extension.

Provides basic statistical functions used by ``scipy.stats._stats_py``.
This is a stub: real statistical routines should be ported to Rust under
``rust/scicore_rs/``.
"""

from __future__ import annotations

import warnings

import rsnumpy as np

__all__ = ['_kendall_dis', '_toint64', '_weightedrankedtau',
           'gaussian_kernel_estimate', 'gaussian_kernel_estimate_log',
           '_local_correlations', '_rank_distance_matrix',
           '_center_distance_matrix', '_transform_distance_matrix',
           '_local_covariance']


def _kendall_dis(x, y):
    """Stub for Kendall's tau distance."""
    return 0.0


def _toint64(x):
    """Stub for converting to int64."""
    return np.asarray(x, dtype=np.int64)


def _weightedrankedtau(x, y, sw=None):
    """Stub for weighted ranked tau."""
    return 0.0, 1.0


def gaussian_kernel_estimate(x, *args, **kwargs):
    """Stub for gaussian_kernel_estimate."""
    return np.zeros_like(x), np.zeros_like(x)


def gaussian_kernel_estimate_log(x, *args, **kwargs):
    """Stub for gaussian_kernel_estimate_log."""
    return np.zeros_like(x), np.zeros_like(x)


# ---------------------------------------------------------------------------
# MGC local correlations (ported from the deleted Cython ``_stats`` extension)
# ---------------------------------------------------------------------------

def _dense_rank_data(x):
    """Columnwise dense ranking (1-based), matching ``np.unique`` inverse."""
    _, v = np.unique(np.asarray(x), return_inverse=True)
    return np.asarray(v) + 1


def _rank_distance_matrix(distx):
    """Rank every column of a distance matrix (1-based dense ranks)."""
    distx = np.asarray(distx)
    cols = [_dense_rank_data(distx[:, i]).reshape(-1, 1)
            for i in range(distx.shape[1])]
    return np.hstack(cols)


def _center_distance_matrix(distx, global_corr='mgc', is_ranked=True):
    distx = np.asarray(distx, dtype='float64')
    n = distx.shape[0]
    m = distx.shape[1]
    rank_distx = np.zeros((n, m))

    if is_ranked:
        rank_distx = _rank_distance_matrix(distx)

    if global_corr == "rank":
        distx = np.asarray(rank_distx, dtype='float64')

    # 'mgc' distance transform (col-wise mean) - default
    col_mean = np.mean(distx, axis=0)
    exp_distx = np.repeat(col_mean * n / (n - 1), n).reshape(-1, n).T

    cent_distx = distx - exp_distx

    if global_corr != "mantel" and global_corr != "biased":
        np.fill_diagonal(cent_distx, 0)

    return cent_distx, rank_distx


def _transform_distance_matrix(distx, disty, global_corr='mgc', is_ranked=True):
    if global_corr == "rank":
        is_ranked = True

    cent_distx, rank_distx = _center_distance_matrix(distx, global_corr, is_ranked)
    cent_disty, rank_disty = _center_distance_matrix(disty, global_corr, is_ranked)

    return {"cent_distx": cent_distx, "cent_disty": cent_disty,
            "rank_distx": rank_distx, "rank_disty": rank_disty}


def _local_covariance(distx, disty, rank_distx, rank_disty):
    # ranks become 0-based indices
    rank_distx = np.asarray(rank_distx, dtype='int64') - 1
    rank_disty = np.asarray(rank_disty, dtype='int64') - 1

    dx = np.asarray(distx).tolist()
    dy = np.asarray(disty).tolist()
    rx = np.asarray(rank_distx).tolist()
    ry = np.asarray(rank_disty).tolist()

    n = len(dx)
    nx = int(np.max(rank_distx)) + 1
    ny = int(np.max(rank_disty)) + 1

    cov = [[0.0] * ny for _ in range(nx)]
    ex = [0.0] * nx
    ey = [0.0] * ny

    # sum element-wise products of A and B grouped by their ranks
    for i in range(n):
        dxi, dyi, rxi, ryi = dx[i], dy[i], rx[i], ry[i]
        for j in range(n):
            a = dxi[j]
            b = dyi[j]
            k = rxi[j]
            l = ryi[j]
            cov[k][l] += a * b
            ex[k] += a
            ey[l] += b

    # cumulative sums along column 0 and row 0
    for k in range(1, nx):
        cov[k][0] += cov[k - 1][0]
        ex[k] += ex[k - 1]
    for l in range(1, ny):
        cov[0][l] += cov[0][l - 1]
        ey[l] += ey[l - 1]

    # 2-D prefix sum (the covariance map)
    for k in range(nx - 1):
        for l in range(ny - 1):
            cov[k + 1][l + 1] += cov[k + 1][l] + cov[k][l + 1] - cov[k][l]

    # center the covariances
    n2 = float(n * n)
    for k in range(nx):
        exk = ex[k]
        row = cov[k]
        for l in range(ny):
            row[l] -= exk * ey[l] / n2

    return np.array(cov)


def _local_correlations(distx, disty, global_corr='mgc'):
    transformed = _transform_distance_matrix(distx, disty, global_corr)

    cent_x = transformed["cent_distx"]
    cent_y = transformed["cent_disty"]
    rank_x = transformed["rank_distx"]
    rank_y = transformed["rank_disty"]

    cov_mat = _local_covariance(cent_x, cent_y.T, rank_x, rank_y.T)

    local_varx = np.diagonal(
        _local_covariance(cent_x, cent_x.T, rank_x, rank_x.T))
    local_vary = np.diagonal(
        _local_covariance(cent_y, cent_y.T, rank_y, rank_y.T))

    denom = np.sqrt(np.outer(local_varx, local_vary))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_mat = np.asarray(cov_mat / denom.real, dtype='float64')

    # clip values that are negligibly larger than 1
    corr_mat = np.where(corr_mat > 1, 1.0, corr_mat)

    # zero out rows/cols whose local variance is <= 0
    row_mask = np.asarray(local_varx <= 0).reshape(-1, 1)
    col_mask = np.asarray(local_vary <= 0).reshape(1, -1)
    corr_mat = np.where(row_mask | col_mask, 0.0, corr_mat)

    return corr_mat


