"""scicore.interpolate — functional tests against analytical references."""
import pytest
import rsnumpy as np

import scicore.interpolate as sinterp

pytestmark = pytest.mark.filterwarnings("default")


# ---------------- working ----------------

def test_interp1d_linear():
    f = sinterp.interp1d(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 4.0]))
    assert np.isclose(f(1.5), 2.5)
    assert np.allclose(f(np.array([0.0, 1.0, 2.0])), [0.0, 1.0, 4.0])


# ---------------- not yet ported / broken ----------------

@pytest.mark.xfail(reason="LAPACK driver getrs not yet ported")
def test_cubic_spline():
    cs = sinterp.CubicSpline(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 4.0]))
    # interpolates the data points
    assert np.allclose(cs(np.array([0.0, 1.0, 2.0])), [0.0, 1.0, 4.0])
    # natural cubic on 3 points [0,1,4]: cs(0.5) = 0.125
    assert np.isclose(cs(0.5), 0.125, atol=1e-10)


@pytest.mark.xfail(reason="ValueError: could not broadcast input array of size 0 into selection of size 1")
def test_pchip_interpolator():
    p = sinterp.PchipInterpolator(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 4.0]))
    assert np.allclose(p(np.array([0.0, 1.0, 2.0])), [0.0, 1.0, 4.0])


@pytest.mark.xfail(reason="LAPACK driver getrs not yet ported")
def test_rbf():
    rbf = sinterp.Rbf(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    # Rbf through (0,0),(1,1) gives ~0.5 at 0.5 (with default function)
    assert np.isclose(rbf(0.5), 0.5, atol=1e-6)


@pytest.mark.xfail(reason="TypeError: Unsupported data type")
def test_rect_bivariate_spline():
    rbs = sinterp.RectBivariateSpline(
        np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([[0.0, 1.0], [1.0, 4.0]])
    )
    assert np.isclose(rbs(0.5, 0.5), 1.5)


@pytest.mark.xfail(reason="TypeError: Unsupported data type")
def test_univariate_spline():
    us = sinterp.UnivariateSpline(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 4.0]))
    assert np.allclose(us(np.array([0.0, 1.0, 2.0])), [0.0, 1.0, 4.0], atol=1e-6)
