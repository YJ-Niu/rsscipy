"""Pure-Python replacement for the deleted C ``_distance_pybind`` extension.

Implements the optimized ``cdist_*`` / ``pdist_*`` pairwise distance kernels
used by ``rsscipy.spatial.distance._distance``.  All metrics are computed with
rsnumpy broadcasting; hot paths can later be moved to Rust under
``rust/rsscipy_rs/`` without changing this module's API.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = [
    'cdist_braycurtis', 'pdist_braycurtis',
    'cdist_canberra', 'pdist_canberra',
    'cdist_chebyshev', 'pdist_chebyshev',
    'cdist_cityblock', 'pdist_cityblock',
    'cdist_cosine', 'pdist_cosine',
    'cdist_dice', 'pdist_dice',
    'cdist_euclidean', 'pdist_euclidean',
    'cdist_hamming', 'pdist_hamming',
    'cdist_jaccard', 'pdist_jaccard',
    'cdist_minkowski', 'pdist_minkowski',
    'cdist_rogerstanimoto', 'pdist_rogerstanimoto',
    'cdist_russellrao', 'pdist_russellrao',
    'cdist_sokalsneath', 'pdist_sokalsneath',
    'cdist_sqeuclidean', 'pdist_sqeuclidean',
    'cdist_yule', 'pdist_yule',
]


def _f64(X):
    return np.asarray(X, dtype='float64')


def _wvec(w, n):
    if w is None:
        return None
    w = _f64(w).reshape(-1)
    if w.shape[0] != n:
        raise ValueError("The weight vector does not match the data.")
    return w


def _finish(D, out):
    if out is not None:
        out[:] = D
        return out
    return D


def _condense(D):
    """Extract the condensed (upper-triangular) vector from a square matrix."""
    m = D.shape[0]
    parts = [D[i, i + 1:] for i in range(m - 1)]
    if not parts:
        return np.zeros(0, dtype='float64')
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# metric kernels: each takes A (m,n), B (k,n) and returns the (m,k) matrix
# ---------------------------------------------------------------------------

def _sqeuclidean(A, B, w=None):
    d = A[:, None, :] - B[None, :, :]
    if w is None:
        return (d * d).sum(axis=-1)
    return (w * d * d).sum(axis=-1)


def _euclidean(A, B, w=None):
    return np.sqrt(_sqeuclidean(A, B, w))


def _cityblock(A, B, w=None):
    d = np.abs(A[:, None, :] - B[None, :, :])
    if w is None:
        return d.sum(axis=-1)
    return (w * d).sum(axis=-1)


def _chebyshev(A, B, w=None):
    d = np.abs(A[:, None, :] - B[None, :, :])
    if w is None:
        return d.max(axis=-1)
    return (w * d).max(axis=-1)


def _minkowski(A, B, p=2, w=None):
    d = np.abs(A[:, None, :] - B[None, :, :])
    if w is None:
        return (d ** p).sum(axis=-1) ** (1.0 / p)
    return (w * d ** p).sum(axis=-1) ** (1.0 / p)


def _canberra(A, B, w=None):
    aa = np.abs(A[:, None, :])
    bb = np.abs(B[None, :, :])
    num = np.abs(A[:, None, :] - B[None, :, :])
    den = aa + bb
    term = np.where(den > 0, num / np.where(den == 0, 1.0, den), 0.0)
    if w is None:
        return term.sum(axis=-1)
    return (w * term).sum(axis=-1)


def _braycurtis(A, B, w=None):
    num = np.abs(A[:, None, :] - B[None, :, :])
    den = np.abs(A[:, None, :] + B[None, :, :])
    if w is not None:
        num = w * num
        den = w * den
    s_num = num.sum(axis=-1)
    s_den = den.sum(axis=-1)
    return np.where(s_den > 0, s_num / np.where(s_den == 0, 1.0, s_den), 0.0)


def _cosine(A, B, w=None):
    if w is None:
        dot = (A[:, None, :] * B[None, :, :]).sum(axis=-1)
        na = (A * A).sum(axis=-1)
        nb = (B * B).sum(axis=-1)
    else:
        dot = (w * A[:, None, :] * B[None, :, :]).sum(axis=-1)
        na = (w * A * A).sum(axis=-1)
        nb = (w * B * B).sum(axis=-1)
    denom = np.sqrt(na[:, None] * nb[None, :])
    return np.where(denom > 0, 1.0 - dot / np.where(denom == 0, 1.0, denom),
                    1.0)


def _hamming(A, B, w=None):
    neq = (A[:, None, :] != B[None, :, :]).astype('float64')
    if w is None:
        return neq.sum(axis=-1) / A.shape[1]
    return (w * neq).sum(axis=-1) / w.sum()


# --- boolean metrics -------------------------------------------------------

def _bool_counts(A, B, w):
    a = A[:, None, :]
    b = B[None, :, :]
    if w is None:
        ntt = (a * b).sum(axis=-1)
        nft = ((1.0 - a) * b).sum(axis=-1)
        ntf = (a * (1.0 - b)).sum(axis=-1)
        nff = ((1.0 - a) * (1.0 - b)).sum(axis=-1)
    else:
        ntt = (w * a * b).sum(axis=-1)
        nft = (w * (1.0 - a) * b).sum(axis=-1)
        ntf = (w * a * (1.0 - b)).sum(axis=-1)
        nff = (w * (1.0 - a) * (1.0 - b)).sum(axis=-1)
    return nff, nft, ntf, ntt


def _dice(A, B, w=None):
    _, nft, ntf, ntt = _bool_counts(A, B, w)
    return (ntf + nft) / (2.0 * ntt + ntf + nft)


def _jaccard(A, B, w=None):
    _, nft, ntf, ntt = _bool_counts(A, B, w)
    num = ntf + nft
    den = ntt + num
    return np.where(den > 0, num / np.where(den == 0, 1.0, den), 0.0)


def _rogerstanimoto(A, B, w=None):
    nff, nft, ntf, ntt = _bool_counts(A, B, w)
    num = 2.0 * (ntf + nft)
    return num / (ntt + nff + num)


def _russellrao(A, B, w=None):
    n = A.shape[1] if w is None else float(w.sum())
    _, _, _, ntt = _bool_counts(A, B, w)
    return (n - ntt) / n


def _sokalsneath(A, B, w=None):
    _, nft, ntf, ntt = _bool_counts(A, B, w)
    R = 2.0 * (ntf + nft)
    den = ntt + R
    if np.all(den == 0):
        raise ValueError('Sokal-Sneath dissimilarity is not defined for '
                         'vectors that are entirely false.')
    return R / den


def _yule(A, B, w=None):
    nff, nft, ntf, ntt = _bool_counts(A, B, w)
    half_R = ntf * nft
    return np.where(half_R > 0, 2.0 * half_R / (ntt * nff + half_R), 0.0)


# ---------------------------------------------------------------------------
# public cdist/pdist entry points (called as fn(XA, XB, out=..., **kwargs))
# ---------------------------------------------------------------------------

_KERNELS = {
    'braycurtis': _braycurtis,
    'canberra': _canberra,
    'chebyshev': _chebyshev,
    'cityblock': _cityblock,
    'cosine': _cosine,
    'dice': _dice,
    'euclidean': _euclidean,
    'hamming': _hamming,
    'jaccard': _jaccard,
    'minkowski': _minkowski,
    'rogerstanimoto': _rogerstanimoto,
    'russellrao': _russellrao,
    'sokalsneath': _sokalsneath,
    'sqeuclidean': _sqeuclidean,
    'yule': _yule,
}


def _check_kwargs(name, kwargs):
    """Reject unexpected keyword arguments, mirroring the strict C signatures.

    Every pybind metric accepts the optional weight vector ``w``; ``minkowski``
    additionally accepts ``p``. Any other keyword is a programming error and
    must raise TypeError (as the deleted C extension did).
    """
    allowed = {'w'}
    if name == 'minkowski':
        allowed.add('p')
    for key in kwargs:
        if key not in allowed:
            raise TypeError(
                f'{name}() got an unexpected keyword argument \'{key}\'')


def _make_cdist(name):
    kernel = _KERNELS[name]

    def cdist_fn(XA, XB, out=None, **kwargs):
        _check_kwargs(name, kwargs)
        A = _f64(XA)
        B = _f64(XB)
        if A.ndim == 1:
            A = A.reshape(1, -1)
        if B.ndim == 1:
            B = B.reshape(1, -1)
        w = _wvec(kwargs.get('w'), A.shape[1])
        extra = {}
        if name == 'minkowski':
            extra['p'] = kwargs.get('p', 2)
        D = kernel(A, B, w=w, **extra) if extra else kernel(A, B, w)
        return _finish(D, out)

    cdist_fn.__name__ = 'cdist_' + name
    return cdist_fn


def _make_pdist(name):
    kernel = _KERNELS[name]

    def pdist_fn(X, out=None, **kwargs):
        _check_kwargs(name, kwargs)
        A = _f64(X)
        if A.ndim == 1:
            A = A.reshape(1, -1)
        w = _wvec(kwargs.get('w'), A.shape[1])
        extra = {}
        if name == 'minkowski':
            extra['p'] = kwargs.get('p', 2)
        D = kernel(A, A, w=w, **extra) if extra else kernel(A, A, w)
        v = _condense(D)
        return _finish(v, out)

    pdist_fn.__name__ = 'pdist_' + name
    return pdist_fn


for _name in _KERNELS:
    globals()['cdist_' + _name] = _make_cdist(_name)
    globals()['pdist_' + _name] = _make_pdist(_name)

__all__.extend('cdist_' + _n for _n in _KERNELS)
__all__.extend('pdist_' + _n for _n in _KERNELS)
