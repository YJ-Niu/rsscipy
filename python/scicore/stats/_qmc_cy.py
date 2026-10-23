import rsnumpy as np

from scicore._scicore_rs import stats as _rust_stats


def _as_bytes(array, dtype=np.float64):
    return np.ascontiguousarray(array, dtype=dtype).tobytes()


def _discrepancy(function, sample, iterative, workers):
    sample = np.ascontiguousarray(sample, dtype=np.float64)
    rows, columns = sample.shape
    return function(
        sample.tobytes(),
        rows,
        columns,
        bool(iterative),
        int(workers),
    )


def _cy_wrapper_centered_discrepancy(sample, iterative, workers):
    return _discrepancy(
        _rust_stats.centered_discrepancy_py, sample, iterative, workers
    )


def _cy_wrapper_wrap_around_discrepancy(sample, iterative, workers):
    return _discrepancy(
        _rust_stats.wrap_around_discrepancy_py, sample, iterative, workers
    )


def _cy_wrapper_mixture_discrepancy(sample, iterative, workers):
    return _discrepancy(
        _rust_stats.mixture_discrepancy_py, sample, iterative, workers
    )


def _cy_wrapper_l2_star_discrepancy(sample, iterative, workers):
    return _discrepancy(
        _rust_stats.l2_star_discrepancy_py, sample, iterative, workers
    )


def _cy_wrapper_update_discrepancy(x_new_view, sample_view, initial_disc):
    sample = np.ascontiguousarray(sample_view, dtype=np.float64)
    rows, columns = sample.shape
    return _rust_stats.update_discrepancy_py(
        _as_bytes(x_new_view),
        sample.tobytes(),
        rows,
        columns,
        float(initial_disc),
    )


def _cy_van_der_corput(n, base, start_index, workers):
    values = _rust_stats.van_der_corput(
        int(n), int(base), int(start_index), int(workers)
    )
    return np.asarray(values, dtype=np.float64)


def _cy_van_der_corput_scrambled(
    n, base, start_index, permutations, workers
):
    permutations = np.ascontiguousarray(permutations, dtype=np.int64)
    rows, columns = permutations.shape
    values = _rust_stats.van_der_corput_scrambled(
        int(n),
        int(base),
        int(start_index),
        permutations.tobytes(),
        rows,
        columns,
        int(workers),
    )
    return np.asarray(values, dtype=np.float64)
