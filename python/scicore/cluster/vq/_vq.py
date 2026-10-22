"""Python compatibility adapter for the Rust vector-quantization kernels."""

import rsnumpy as np

from rsscipy._rsscipy_rs import cluster as _rust_cluster


def _single_precision(array):
    return array.dtype.itemsize == 4


def vq(obs, codes):
    obs = np.asarray(obs)
    codes = np.asarray(codes)
    if obs.dtype != codes.dtype:
        raise TypeError('observation and code should have same dtype')
    if obs.dtype not in (np.float32, np.float64):
        raise TypeError('type other than float or double not supported')
    if obs.ndim != codes.ndim:
        raise ValueError(
            'observation and code should have same number of dimensions')
    if obs.ndim == 1:
        nobs, ncodes, nfeat = obs.shape[0], codes.shape[0], 1
    elif obs.ndim == 2:
        nobs, nfeat = obs.shape
        ncodes = codes.shape[0]
        if nfeat != codes.shape[1]:
            raise ValueError(
                'obs and code should have same number of features (columns)')
    else:
        raise ValueError('ndim different than 1 or 2 are not supported')

    labels, distances = _rust_cluster._vq(
        obs.reshape((-1,)).tolist(),
        codes.reshape((-1,)).tolist(),
        nobs,
        ncodes,
        nfeat,
        _single_precision(obs),
    )
    return (
        np.asarray(labels, dtype=np.int32),
        np.asarray(distances).astype(obs.dtype),
    )


def update_cluster_means(obs, labels, nc):
    obs = np.asarray(obs)
    labels = np.asarray(labels)
    if obs.dtype not in (np.float32, np.float64):
        raise TypeError('type other than float or double not supported')
    labels = labels.astype(np.int32)
    if labels.ndim != 1:
        raise ValueError('labels must be a 1d array')
    if obs.ndim == 1:
        nobs, nfeat = obs.shape[0], 1
        output_shape = (nc,)
    elif obs.ndim == 2:
        nobs, nfeat = obs.shape
        output_shape = (nc, nfeat)
    else:
        raise ValueError('ndim different than 1 or 2 are not supported')

    means, has_members = _rust_cluster._update_cluster_means(
        obs.reshape((-1,)).tolist(),
        labels.reshape((-1,)).tolist(),
        nc,
        nobs,
        nfeat,
        _single_precision(obs),
    )
    return (
        np.asarray(means).astype(obs.dtype).reshape(output_shape),
        np.asarray(has_members, dtype=np.bool_),
    )
