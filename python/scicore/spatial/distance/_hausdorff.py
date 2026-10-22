"""Pure-Python replacement for the deleted C ``_hausdorff`` extension.

Implements the early-break directed Hausdorff algorithm of Taha & Hanbury.
The hot inner loop lives in the Rust core
(``scicore._scicore_rs.spatial.directed_hausdorff_core``); this module only
handles validation, the (optional) random permutation of the input points and
the index mapping back to the original arrays.
"""

from __future__ import annotations

import rsnumpy as np

from scicore._scicore_rs import spatial as _rust_spatial

__all__ = ['directed_hausdorff', 'hausdorff_distance']


def directed_hausdorff(u, v, seed=0):
    """Directed Hausdorff distance between two 2-D arrays.

    Returns ``(distance, index_in_u, index_in_v)`` using Euclidean
    (L2) distances between the point sets.
    """
    u = np.asarray(u, dtype='float64')
    v = np.asarray(v, dtype='float64')
    if u.ndim != 2 or v.ndim != 2:
        raise ValueError('u and v need to be 2 dimensional')
    if u.shape[1] != v.shape[1]:
        raise ValueError('u and v need to have the same number of columns '
                         f'u.shape[1]={u.shape[1]} v.shape[1]={v.shape[1]}')

    if isinstance(seed, np.random.Generator):
        # Randomized order: shuffling generally helps the early-break
        # heuristic find the Hausdorff pair sooner.
        resort1 = np.arange(u.shape[0], dtype='int64')
        resort2 = np.arange(v.shape[0], dtype='int64')
        seed.shuffle(resort1)
        seed.shuffle(resort2)
    else:
        # Legacy path (int / None / RandomState): keep the original order.
        # The distance is exact either way; only the reported indices for
        # tied Hausdorff pairs are implementation-defined.  Validate the
        # seed like ``np.random.RandomState(seed)`` would.
        if not isinstance(seed, np.random.RandomState):
            np.random.RandomState(seed)
        resort1 = np.arange(u.shape[0], dtype='int64')
        resort2 = np.arange(v.shape[0], dtype='int64')

    us = np.ascontiguousarray(np.take(u, resort1, axis=0))
    vs = np.ascontiguousarray(np.take(v, resort2, axis=0))

    dist, i_ret, j_ret = _rust_spatial.directed_hausdorff_core(
        us.tobytes(), vs.tobytes(), us.shape[1])
    return dist, int(resort1[i_ret]), int(resort2[j_ret])


def hausdorff_distance(u, v):
    """Hausdorff distance: the maximum of the two directed distances."""
    return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
