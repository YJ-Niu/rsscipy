"""Stub replacement for the deleted Cython ``_hierarchy`` extension.

Provides hierarchical clustering routines used by ``scipy.cluster.hierarchy``.
This is a stub: real clustering routines should be ported to Rust under
``rust/src/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = ['linkage', 'fcluster', 'cophenet', 'inconsistent', 'mst_single_linkage',
           'nn_chain', 'fast_linkage', 'calculate_cluster_sizes',
           'cluster_in', 'cluster_dist', 'cluster_maxclust_dist',
           'cluster_maxclust_monocrit', 'cluster_monocrit']


def _stub(*args, **kwargs):
    raise NotImplementedError("_hierarchy not yet ported to Rust/rsnumpy")


for _name in __all__:
    globals()[_name] = _stub
