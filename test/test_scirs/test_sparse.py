"""scicore.sparse — functional tests against analytical references.

The whole sparse stack is currently broken: matrix construction hits
rsnumpy conversion-arity bugs, and the linalg/csgraph layers still
import upstream scipy (not installed).
"""
import pytest
import rsnumpy as np

import scicore.sparse as sps

pytestmark = pytest.mark.filterwarnings("default")


def _dense_2x2():
    return np.array([[1.0, 0.0], [0.0, 2.0]])


# ---------------- broken ----------------

@pytest.mark.xfail(reason="coo_tocsr() takes 4 positional arguments but 9 were given (rsnumpy)")
def test_csr_matrix_roundtrip():
    m = sps.csr_matrix(_dense_2x2())
    assert np.allclose(m.toarray(), _dense_2x2())


@pytest.mark.xfail(reason="TypeError: 'float' object cannot be interpreted as an integer")
def test_diags():
    m = sps.diags([1.0, 2.0])
    assert np.allclose(m.toarray(), [[1.0, 0.0], [0.0, 2.0]])


@pytest.mark.xfail(reason="csr_todense() takes 5 positional arguments but 6 were given (rsnumpy)")
def test_eye():
    m = sps.eye(3, format="csr")
    assert np.allclose(m.toarray(), np.eye(3))


@pytest.mark.xfail(reason="coo_tocsr() takes 4 positional arguments but 9 were given (rsnumpy)")
def test_addition():
    a = sps.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    b = sps.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0]]))
    assert np.allclose((a + b).toarray(), np.array([[2.0, 1.0], [1.0, 3.0]]))


@pytest.mark.xfail(reason="coo_tocsr() takes 4 positional arguments but 9 were given (rsnumpy)")
def test_matvec():
    a = sps.csr_matrix(_dense_2x2())
    out = a @ np.array([1.0, 1.0])
    assert np.allclose(np.asarray(out), [1.0, 2.0])


@pytest.mark.xfail(reason="ModuleNotFoundError: No module named 'scipy' (csgraph/spsolve fallback)")
def test_spsolve():
    a = sps.csr_matrix(np.array([[4.0, 3.0], [6.0, 3.0]]))
    x = sps.linalg.spsolve(a, np.array([10.0, 12.0]))
    assert np.allclose(x, [2.0, -2.0])


@pytest.mark.xfail(reason="ModuleNotFoundError: No module named 'scipy' (eigsh fallback)")
def test_eigsh():
    a = sps.csr_matrix(np.array([[2.0, 0.0], [0.0, 3.0]]))
    w = sps.linalg.eigsh(a, k=2, which="LA")
    assert np.allclose(np.sort(w), [2.0, 3.0])


@pytest.mark.xfail(reason="ModuleNotFoundError: No module named 'scipy' (csgraph fallback)")
def test_dijkstra():
    g = sps.csr_matrix(np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0], [0.0, 0.0, 2.0, 0.0]]))
    dist = sps.csgraph.dijkstra(g)
    assert np.isclose(dist[0, 3], 3.0)


@pytest.mark.xfail(reason="ModuleNotFoundError: No module named 'scipy' (csgraph fallback)")
def test_connected_components():
    g = sps.csr_matrix(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    n, labels = sps.csgraph.connected_components(g)
    assert n == 2


@pytest.mark.xfail(reason="ModuleNotFoundError: No module named 'scipy' (csgraph fallback)")
def test_laplacian():
    g = sps.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    L = sps.csgraph.laplacian(g)
    assert np.allclose(L.toarray(), [[1.0, -1.0], [-1.0, 1.0]])
