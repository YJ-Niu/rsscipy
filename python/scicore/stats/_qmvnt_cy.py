import rsnumpy as np

from rsscipy._rsscipy_rs import stats as _rust_stats


def _as_bytes(array):
    return np.ascontiguousarray(array, dtype=np.float64).tobytes()


def _qmvn_inner(q, rndm, n_qmc_samples, n_batches, cho, lo, hi):
    from rsscipy.special import ndtr, ndtri

    return _rust_stats.qmvn_inner(
        _as_bytes(q),
        _as_bytes(rndm),
        int(n_qmc_samples),
        int(n_batches),
        _as_bytes(cho),
        _as_bytes(lo),
        _as_bytes(hi),
        ndtr,
        ndtri,
    )


def _qmvt_inner(q, rndm, n_qmc_samples, n_batches, cho, lo, hi, nu):
    from rsscipy.special import gammaincinv, ndtr, ndtri

    return _rust_stats.qmvt_inner(
        _as_bytes(q),
        _as_bytes(rndm),
        int(n_qmc_samples),
        int(n_batches),
        _as_bytes(cho),
        _as_bytes(lo),
        _as_bytes(hi),
        float(nu),
        ndtr,
        ndtri,
        gammaincinv,
    )
