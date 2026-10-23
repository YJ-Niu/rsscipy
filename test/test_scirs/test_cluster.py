"""scicore.cluster — functional tests against analytical references."""
import pytest
import rsnumpy as np

import scicore.cluster.hierarchy as shier
import scicore.cluster.vq as svq

pytestmark = pytest.mark.filterwarnings("default")


def _two_blobs():
    return np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])


# ---------------- working ----------------

def test_kmeans2_two_blobs():
    centroids, inertia = svq.kmeans2(_two_blobs(), 2, missing="raise")
    c = sorted(map(tuple, np.round(centroids, 1).tolist()))
    assert c == [(0.1, 0.0), (10.1, 10.0)] or c == [(0.0, 0.0), (10.0, 10.0)]


def test_vq_assignment():
    codebook = np.array([[0.0, 0.0], [10.0, 10.0]])
    data = np.array([[0.1, 0.0], [9.9, 10.0]])
    labels, dist = svq.vq(codebook, data)
    assert list(labels) == [0, 1]


# ---------------- not yet ported ----------------

@pytest.mark.xfail(reason="_hierarchy not yet ported to Rust/rsnumpy")
def test_linkage_ward():
    Z = shier.linkage(_two_blobs(), method="ward")
    assert Z.shape == (3, 4)
    # the last merge joins the two blobs
    assert Z[-1, 2] > 5.0


@pytest.mark.xfail(reason="_hierarchy not yet ported to Rust/rsnumpy")
def test_fcluster_two_clusters():
    Z = shier.linkage(_two_blobs(), method="ward")
    labels = shier.fcluster(Z, t=2, criterion="maxclust")
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]
