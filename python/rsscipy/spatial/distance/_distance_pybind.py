"""Stub replacement for the deleted C ``_distance_pybind`` extension.

Provides distance function bindings used by ``scipy.spatial.distance._distance``.
This is a stub: real distance metrics should be ported to Rust under
``rust/rsscipy_rs/``.
"""

from __future__ import annotations

import rsnumpy as np

__all__ = []

# Distance metric function stubs used by _distance.py
_distance_funcs = [
    'cdist_braycurtis', 'pdist_braycurtis',
    'cdist_canberra', 'pdist_canberra',
    'cdist_chebyshev', 'pdist_chebyshev',
    'cdist_cityblock', 'pdist_cityblock',
    'cdist_correlation', 'pdist_correlation',
    'cdist_cosine', 'pdist_cosine',
    'cdist_dice', 'pdist_dice',
    'cdist_euclidean', 'pdist_euclidean',
    'cdist_hamming', 'pdist_hamming',
    'cdist_jaccard', 'pdist_jaccard',
    'cdist_jensenshannon', 'pdist_jensenshannon',
    'cdist_kulsinski', 'pdist_kulsinski',
    'cdist_mahalanobis', 'pdist_mahalanobis',
    'cdist_matching', 'pdist_matching',
    'cdist_minkowski', 'pdist_minkowski',
    'cdist_rogerstanimoto', 'pdist_rogerstanimoto',
    'cdist_russellrao', 'pdist_russellrao',
    'cdist_seuclidean', 'pdist_seuclidean',
    'cdist_sokalmichener', 'pdist_sokalmichener',
    'cdist_sokalsneath', 'pdist_sokalsneath',
    'cdist_sqeuclidean', 'pdist_sqeuclidean',
    'cdist_wminkowski', 'pdist_wminkowski',
    'cdist_yule', 'pdist_yule',
]


def _make_stub(name):
    def stub(*args, **kwargs):
        raise NotImplementedError(f"{name} not yet ported to Rust/rsnumpy")
    stub.__name__ = name
    return stub


for _name in _distance_funcs:
    globals()[_name] = _make_stub(_name)
