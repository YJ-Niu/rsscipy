"""Tools and utilities for working with compressed sparse graphs."""

import rsnumpy as np

from rsscipy.sparse import csr_array, csr_matrix, spmatrix, issparse
from rsscipy.sparse._sputils import is_pydata_spmatrix
from rsscipy._rsscipy_rs import csgraph as _rust_csgraph

DTYPE = np.float64
ITYPE = np.int32
NULL_IDX = -9999


def _is_matrix_instance(value):
    matrix_type = getattr(np, "matrix", None)
    return isinstance(matrix_type, type) and isinstance(value, matrix_type)


def csgraph_from_masked(graph):
    """Construct a CSR-format graph from a masked array."""
    graph = np.ma.asarray(graph)
    if graph.ndim != 2:
        raise ValueError("graph should have two dimensions")
    n = graph.shape[0]
    if graph.shape[1] != n:
        raise ValueError("graph should be a square array")

    mask = ~graph.mask
    data = np.asarray(graph.data[mask], dtype=DTYPE, order="C")
    row_indices = np.arange(n, dtype=ITYPE)
    indices = np.asarray(
        np.broadcast_to(row_indices, (n, n))[mask], dtype=ITYPE, order="C"
    )
    indptr = np.zeros(n + 1, dtype=ITYPE)
    indptr[1:] = mask.sum(axis=1).cumsum()
    return csr_array((data, indices, indptr), (n, n))


def csgraph_masked_from_dense(
    graph, null_value=0, nan_null=True, infinity_null=True, copy=True
):
    """Construct a masked array graph representation from a dense matrix."""
    graph = np.array(graph, copy=copy)
    if graph.ndim != 2:
        raise ValueError("graph should have two dimensions")
    if graph.shape[1] != graph.shape[0]:
        raise ValueError("graph should be a square array")

    if null_value is None:
        mask = np.zeros(graph.shape, dtype=bool)
    else:
        null_value = float(null_value)
        if np.isnan(null_value):
            nan_null = True
            null_value = None
        elif np.isinf(null_value):
            infinity_null = True
            null_value = None
        mask = (
            np.zeros(graph.shape, dtype=bool)
            if null_value is None
            else np.abs(graph - null_value) <= 1e-8 + 1e-5 * np.abs(null_value)
        )

    if infinity_null:
        mask = mask | np.isinf(graph)
    if nan_null:
        mask = mask | np.isnan(graph)
    return np.ma.masked_array(graph, mask=mask, fill_value=null_value)


def csgraph_from_dense(
    graph, null_value=0, nan_null=True, infinity_null=True
):
    """Construct a CSR-format sparse graph from a dense matrix."""
    was_matrix = _is_matrix_instance(graph)
    masked = csgraph_masked_from_dense(
        graph,
        null_value=null_value,
        nan_null=nan_null,
        infinity_null=infinity_null,
    )
    result = csgraph_from_masked(masked)
    if was_matrix:
        return csr_matrix(result, copy=False)
    return result


def csgraph_to_dense(csgraph, null_value=0):
    """Convert a sparse graph representation to a dense representation."""
    if not issparse(csgraph):
        raise ValueError("csgraph must be sparse")
    if csgraph.format not in ("lil", "csc", "csr"):
        raise ValueError("csgraph must be lil, csr, or csc format")
    csgraph = csgraph.tocsr()
    n = csgraph.shape[0]
    if csgraph.shape[1] != n:
        raise ValueError("csgraph should be a square matrix")

    graph = _rust_csgraph._dense_from_csr(
        np.asarray(csgraph.data, dtype=DTYPE, order="C").tolist(),
        np.asarray(csgraph.indices, dtype=ITYPE, order="C").tolist(),
        np.asarray(csgraph.indptr, dtype=ITYPE, order="C").tolist(),
        n,
        float(null_value),
    )
    return np.asarray(graph, dtype=DTYPE).reshape((n, n))


def csgraph_to_masked(csgraph):
    """Convert a sparse graph representation to a masked array."""
    return np.ma.masked_invalid(csgraph_to_dense(csgraph, np.nan))


def reconstruct_path(csgraph, predecessors, directed=True):
    """Construct a sparse tree from a graph and a predecessor list."""
    from ._validation import validate_graph

    original_graph = csgraph
    csgraph = validate_graph(csgraph, directed, dense_output=False)
    predecessors = np.asarray(predecessors)
    if not np.isdtype(predecessors.dtype, "integral"):
        raise ValueError("`predecessors` must be of an integral dtype")

    n = csgraph.shape[0]
    nnull = int((predecessors < 0).sum())
    indices = np.argsort(predecessors)[nnull:].astype(ITYPE)
    parent_indices = predecessors[indices]
    indptr = parent_indices.searchsorted(np.arange(n + 1)).astype(ITYPE)

    data = csgraph[parent_indices, indices]
    data = data.toarray().ravel() if issparse(data) else np.asarray(data).ravel()

    if not directed:
        reverse_data = csgraph[indices, parent_indices]
        reverse_data = (
            reverse_data.toarray().ravel()
            if issparse(reverse_data)
            else np.asarray(reverse_data).ravel()
        )
        data[data == 0] = np.inf
        reverse_data[reverse_data == 0] = np.inf
        data = np.minimum(data, reverse_data)

    if isinstance(original_graph, spmatrix):
        return csr_matrix((data, indices, indptr), shape=(n, n))
    tree = csr_array((data, indices, indptr), shape=(n, n))
    if is_pydata_spmatrix(original_graph):
        sparse_cls = original_graph.__class__
        try:
            return sparse_cls.from_scipy_sparse(
                tree, fill_value=original_graph.fill_value
            )
        except TypeError:
            return sparse_cls.from_scipy_sparse(tree)
    return tree


def construct_dist_matrix(
    graph, predecessors, directed=True, null_value=np.inf
):
    """Construct a distance matrix from a predecessor matrix."""
    from ._validation import validate_graph

    graph = validate_graph(
        graph,
        directed,
        dtype=DTYPE,
        csr_output=False,
        copy_if_dense=not directed,
    )
    predecessors = np.asarray(predecessors)
    if predecessors.dtype != ITYPE:
        raise TypeError("Type of predecessors array should be np.int32")
    if predecessors.shape != graph.shape:
        raise ValueError("graph and predecessors must have the same shape")

    n = graph.shape[0]
    distances = _rust_csgraph._construct_dist_matrix(
        np.asarray(graph, dtype=DTYPE).reshape((-1,)).tolist(),
        predecessors.reshape((-1,)).tolist(),
        n,
        directed,
        float(null_value),
    )
    return np.asarray(distances, dtype=DTYPE).reshape(graph.shape)
