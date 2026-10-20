import rsnumpy as np
from warnings import warn

from rsscipy._rsscipy_rs import csgraph as _rust_csgraph
from rsscipy.sparse import csr_array, issparse, SparseEfficiencyWarning
from rsscipy.sparse._sputils import convert_pydata_sparse_to_scipy
from rsscipy.sparse.csgraph import maximum_bipartite_matching


def reverse_cuthill_mckee(graph, symmetric_mode=False):
    """reverse_cuthill_mckee(graph, symmetric_mode=False)

    Returns the permutation array that orders a sparse CSR or CSC matrix
    in Reverse-Cuthill McKee ordering.

    By default, the input is assumed to be nonsymmetric and the ordering is
    computed from the structure of ``A + A.T``. Set ``symmetric_mode=True``
    when the input is guaranteed to be symmetric in structure.

    Parameters
    ----------
    graph : sparse array or matrix
        Input graph in CSC or CSR sparse format.
    symmetric_mode : bool, optional
        Whether the input is guaranteed to be symmetric.

    Returns
    -------
    perm : ndarray
        Array of permuted row and column indices.

    Notes
    -----
    .. versionadded:: 0.15.0

    References
    ----------
    E. Cuthill and J. McKee, "Reducing the Bandwidth of Sparse Symmetric
    Matrices", ACM '69 Proceedings of the 1969 24th national conference (1969).

    Examples
    --------
    >>> from rsscipy.sparse import csr_array
    >>> from rsscipy.sparse.csgraph import reverse_cuthill_mckee
    >>> graph = csr_array([[0, 1, 2, 0], [0, 0, 0, 1],
    ...                    [2, 0, 0, 3], [0, 0, 0, 0]])
    >>> reverse_cuthill_mckee(graph)
    array([3, 2, 1, 0], dtype=int32)
    """
    graph = convert_pydata_sparse_to_scipy(graph)
    if not issparse(graph):
        raise TypeError("Input graph must be sparse")
    if graph.format not in ("csc", "csr"):
        raise TypeError("Input must be in CSC or CSR sparse format.")
    nrows = graph.shape[0]
    if not symmetric_mode:
        graph = graph + graph.transpose()
    permutation = _rust_csgraph._reverse_cuthill_mckee(
        graph.indices.tolist(),
        graph.indptr.tolist(),
        nrows,
    )
    return np.asarray(permutation, dtype=graph.indices.dtype)


def structural_rank(graph):
    """structural_rank(graph)

    Compute the structural rank of a graph (matrix) with a given sparsity
    pattern. It is the number of entries in a maximum transversal of the
    corresponding bipartite graph, and is an upper bound on numerical rank.

    Parameters
    ----------
    graph : sparse array or matrix
        Input sparse array.

    Returns
    -------
    rank : int
        Structural rank of the sparse graph.

    References
    ----------
    I. S. Duff, "Computing the Structural Index", SIAM J. Alg. Disc. Meth.,
    Vol. 7, 594 (1986).

    Examples
    --------
    >>> from rsscipy.sparse import csr_array
    >>> from rsscipy.sparse.csgraph import structural_rank
    >>> structural_rank(csr_array([[1, 1], [1, 0]]))
    2
    """
    graph = convert_pydata_sparse_to_scipy(graph)
    if not issparse(graph):
        raise TypeError("Input must be sparse")
    if graph.format != "csr":
        if graph.format not in ("csc", "coo"):
            warn(
                "Input matrix should be in CSC, CSR, or COO matrix format",
                SparseEfficiencyWarning,
            )
        graph = csr_array(graph)
    if graph.shape[0] > graph.shape[1]:
        graph = graph.T.tocsr()
    return np.sum(maximum_bipartite_matching(graph) >= 0)
