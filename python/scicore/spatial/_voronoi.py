import rsnumpy as np

from scicore._scicore_rs import spatial as _rust_spatial

__all__ = ["sort_vertices_of_regions"]


def sort_vertices_of_regions(simplices, regions):
    simplices = np.ascontiguousarray(simplices, dtype=np.int32)
    if simplices.ndim != 2 or simplices.shape[1] != 3:
        raise ValueError("simplices must be a 2D array with three columns")
    _rust_spatial.sort_vertices_of_regions(
        simplices.tobytes(),
        int(simplices.shape[0]),
        int(simplices.shape[1]),
        regions,
    )
