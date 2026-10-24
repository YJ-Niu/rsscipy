"""scicore.spatial kd-tree — functional tests against brute-force references.

Covers the Rust-backed ``cKDTree``/``KDTree`` query methods: ``query``,
``query_ball_point``, ``query_ball_tree``, ``query_pairs``, ``count_neighbors``
and ``sparse_distance_matrix``. Deterministic (fixed seed) so it runs in CI.
"""
import rsnumpy as np

from scicore.spatial import KDTree, cKDTree

_N, _M, _NY = 200, 3, 80


def _data():
    rng = np.random.default_rng(42)
    X = rng.random((_N, _M)) * 10
    Y = rng.random((_NY, _M)) * 10
    return X, Y


def _d(A, b):
    """Row-wise Euclidean distance from the rows of ``A`` to point ``b``."""
    return np.sqrt(((A - b) ** 2).sum(axis=1))


def test_query_single():
    X, _ = _data()
    tree = cKDTree(X)
    q = np.array([1.0, 2.0, 3.0])
    d, i = tree.query(q)
    dist = _d(X, q)
    assert np.isclose(d, dist[int(i)], rtol=1e-9)
    assert int(i) == int(np.argmin(dist))


def test_query_k():
    X, _ = _data()
    tree = cKDTree(X)
    q = np.array([1.0, 2.0, 3.0])
    k = 5
    d, i = tree.query(q, k)
    expect = np.sort(_d(X, q))[:k]
    assert np.allclose(np.sort(d), expect, rtol=1e-9)


def test_query_ball_point():
    X, _ = _data()
    tree = cKDTree(X)
    q = np.array([1.0, 2.0, 3.0])
    r = 2.0
    got = tree.query_ball_point(q, r)
    expect = set(np.nonzero(_d(X, q) <= r)[0].tolist())
    assert set(got.tolist()) == expect


def test_query_pairs():
    X, _ = _data()
    tree = cKDTree(X)
    r = 1.5
    got = set(tree.query_pairs(r))
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    expect = set()
    for a in range(_N):
        for b in range(a + 1, _N):
            if D[a, b] <= r:
                expect.add((a, b))
    assert got == expect


def test_query_ball_tree():
    X, Y = _data()
    qbt = cKDTree(X).query_ball_tree(cKDTree(Y), 1.5)
    for qi in range(_N):
        expect = set(np.nonzero(_d(Y, X[qi]) <= 1.5)[0].tolist())
        assert set(qbt[qi].tolist()) == expect


def test_count_neighbors():
    X, Y = _data()
    cnt = cKDTree(X).count_neighbors(cKDTree(Y), 1.5)
    total = 0
    for qi in range(_N):
        total += int(np.nonzero(_d(Y, X[qi]) <= 1.5)[0].size)
    assert int(cnt) == total


def test_sparse_distance_matrix():
    X, Y = _data()
    sdm = cKDTree(X).sparse_distance_matrix(cKDTree(Y), 1.5, output_type="dict")
    expect = {}
    for a in range(_N):
        da = _d(Y, X[a])
        for b in range(_NY):
            if da[b] <= 1.5:
                expect[(a, b)] = float(da[b])
    assert len(sdm) == len(expect)
    for key, dd in expect.items():
        assert abs(sdm[key] - dd) < 1e-9


def test_kdtree_delegation():
    # KDTree subclasses cKDTree and delegates; results must agree.
    X, Y = _data()
    kdt = KDTree(X)
    q = np.array([1.0, 2.0, 3.0])
    d, i = kdt.query(q)
    assert np.isclose(d, _d(X, q)[int(i)], rtol=1e-9)
    assert set(kdt.query_pairs(1.5)) == set(cKDTree(X).query_pairs(1.5))
    assert int(kdt.count_neighbors(KDTree(Y), 1.5)) == int(
        cKDTree(X).count_neighbors(cKDTree(Y), 1.5)
    )
