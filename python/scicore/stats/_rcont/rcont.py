import rsnumpy as np

from scicore._scicore_rs import stats as _rust_stats


def _scalar_int(value):
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _prepare(row, col, ntot, size):
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    size = _scalar_int(size)
    return row, col, _scalar_int(ntot), size


def _rvs(method, row, col, ntot, size, random_state):
    row, col, ntot, size = _prepare(row, col, ntot, size)
    values = method(
        row.tolist(), col.tolist(), ntot, size, random_state
    )
    return np.asarray(values, dtype=np.int64).reshape(
        (size, row.size, col.size)
    )


def rvs_rcont1(row, col, ntot, size, random_state):
    row, col, ntot, size = _prepare(row, col, ntot, size)
    draws = max(ntot - 1, 0) * size
    uniforms = np.ascontiguousarray(
        random_state.random(size=draws),
        dtype=np.float64,
    ).tobytes()
    values = _rust_stats.rvs_rcont1(
        row.tolist(), col.tolist(), ntot, size, uniforms
    )
    return np.asarray(values, dtype=np.int64).reshape(
        (size, row.size, col.size)
    )


def rvs_rcont2(row, col, ntot, size, random_state):
    return _rvs(_rust_stats.rvs_rcont2, row, col, ntot, size, random_state)
