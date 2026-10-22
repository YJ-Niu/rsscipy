"""Python compatibility layer for Rust regular-grid interpolation kernels."""

import rsnumpy as np

from scicore._scicore_rs import interpolate as _rust_interpolate


def find_indices(grid, xi):
    xi = np.asarray(xi)
    npoints = xi.shape[1]
    indices, norm_distances = _rust_interpolate._find_indices(
        [axis.tolist() for axis in grid],
        xi.reshape((-1,)).tolist(),
        npoints,
    )
    return (
        np.asarray(indices, dtype=np.intp).reshape(xi.shape),
        np.asarray(norm_distances, dtype=float).reshape(xi.shape),
    )


def evaluate_linear_2d(values, indices, norm_distances, grid, out):
    values = np.asarray(values)
    indices = np.asarray(indices)
    norm_distances = np.asarray(norm_distances)
    real_imag = _rust_interpolate._evaluate_linear_2d(
        values.reshape((-1,)).tolist(),
        indices.reshape((-1,)).tolist(),
        norm_distances.reshape((-1,)).tolist(),
        values.shape[0],
        values.shape[1],
        indices.shape[1],
    )
    if values.dtype == np.complex128:
        result = [complex(real, imag) for real, imag in real_imag]
    else:
        result = [real for real, _ in real_imag]
    return np.asarray(result, dtype=values.dtype).reshape(out.shape)
