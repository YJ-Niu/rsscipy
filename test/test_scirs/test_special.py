"""scicore.special — special functions against known values.

Many ufuncs currently return NaN (ufunc wiring incomplete) or raise;
they are xfail-marked with the observed symptom as reason.
"""
import pytest
import rsnumpy as np

import scicore.special as sp

pytestmark = pytest.mark.filterwarnings("default")


# ---------------- working ----------------

def test_gamma_known():
    assert np.isclose(sp.gamma(5), 24.0)
    assert np.isclose(sp.gamma(0.5), np.sqrt(np.pi), rtol=1e-12)
    assert np.isclose(sp.gamma(1), 1.0)


def test_gammaln_loggamma():
    assert np.isclose(sp.gammaln(5), np.log(24.0))
    assert np.isclose(sp.loggamma(0.5), np.log(np.sqrt(np.pi)), rtol=1e-12)


def test_erf_erfc():
    assert np.isclose(sp.erf(0.0), 0.0)
    assert np.isclose(sp.erf(1.0), 0.8427007929497149, rtol=1e-12)
    assert np.isclose(sp.erf(1.0) + sp.erfc(1.0), 1.0)


def test_factorial():
    assert np.isclose(sp.factorial(5), 120.0)
    assert np.isclose(sp.factorial(0), 1.0)


def test_sinc():
    assert np.isclose(sp.sinc(0.0), 1.0)
    assert np.isclose(sp.sinc(1.0), 0.0)
    assert np.isclose(sp.sinc(0.5), np.sin(np.pi / 2) / (np.pi / 2))


# ---------------- not yet ported / broken ----------------

@pytest.mark.xfail(reason="returns nan")
def test_erfinv():
    assert np.isclose(sp.erfinv(sp.erf(0.7)), 0.7, rtol=1e-10)


@pytest.mark.xfail(reason="returns nan")
def test_beta_betainc():
    assert np.isclose(sp.beta(1.0, 1.0), 1.0)
    assert np.isclose(sp.beta(2.0, 3.0), 1.0 / 12.0)
    assert np.isclose(sp.betainc(2.0, 2.0, 0.5), 0.5)


@pytest.mark.xfail(reason="returns nan")
def test_digamma_psi():
    euler = 0.5772156649015329
    assert np.isclose(sp.digamma(1.0), -euler, rtol=1e-12)
    assert np.isclose(sp.psi(4.0), (1 + 1 / 2 + 1 / 3) - euler, rtol=1e-12)


@pytest.mark.xfail(reason="TypeError: float() argument is NoneType")
def test_zeta():
    assert np.isclose(sp.zeta(2.0), np.pi**2 / 6, rtol=1e-10)
    assert np.isclose(sp.zeta(4.0), np.pi**4 / 90, rtol=1e-10)


@pytest.mark.xfail(reason="returns nan")
def test_bessel_j0_j1_y0():
    assert np.isclose(sp.j0(0.0), 1.0)
    assert np.isclose(sp.j0(2.4048255577), 0.0, atol=1e-7)
    assert np.isclose(sp.j1(0.0), 0.0)
    assert np.isclose(sp.y0(1.0), 0.7812128213, rtol=1e-6)


@pytest.mark.xfail(reason="BLAS dsbevd not yet ported")
def test_roots_legendre():
    r = sp.roots_legendre(5)
    assert np.allclose(sorted(r), -np.array(sorted(r))[::-1], atol=1e-12)


@pytest.mark.xfail(reason="BLAS dsbevd not yet ported")
def test_roots_hermite():
    r = sp.roots_hermite(3)
    assert np.allclose(np.sort(r), np.sort([0.0, -np.sqrt(1.5), np.sqrt(1.5)]), atol=1e-12)


@pytest.mark.xfail(reason="returns nan")
def test_eval_legendre():
    x = 0.4
    assert np.isclose(sp.eval_legendre(2, x), (3 * x**2 - 1) / 2)


@pytest.mark.xfail(reason="returns nan")
def test_expit():
    assert np.isclose(sp.expit(0.0), 0.5)
    assert np.isclose(sp.expit(2.0), 1.0 / (1.0 + np.exp(-2.0)), rtol=1e-12)


@pytest.mark.xfail(reason="TypeError: 'tuple' object cannot be interpreted as an integer")
def test_logsumexp():
    x = np.array([1.0, 2.0, 3.0])
    expected = np.log(np.exp(1.0) + np.exp(2.0) + np.exp(3.0))
    assert np.isclose(sp.logsumexp(x), expected, rtol=1e-12)


@pytest.mark.xfail(reason="returns nan")
def test_rel_entr_xlogy():
    assert np.isclose(sp.rel_entr(2.0, 1.0), 2.0 * np.log(2.0))
    assert np.isclose(sp.xlogy(3.0, 2.0), 3.0 * np.log(2.0))
    assert np.isclose(sp.xlogy(0.0, 2.0), 0.0)


@pytest.mark.xfail(reason="returns nan")
def test_gammainc_pair():
    # gamma(2) * gammainc(2, 1) = lower_incomplete_gamma(2, 1) = 1 - e^-1
    assert np.isclose(sp.gammainc(2.0, 1.0), (1.0 - np.exp(-1.0)) / sp.gamma(2.0), rtol=1e-12)
    assert np.isclose(sp.gammainc(2.0, 1.0) + sp.gammaincc(2.0, 1.0), 1.0)


@pytest.mark.xfail(reason="returns nan")
def test_spherical_jn():
    assert np.isclose(sp.spherical_jn(0, 1.0), np.sin(1.0))


@pytest.mark.xfail(reason="returns nan")
def test_sici():
    si, ci = sp.sici(1.0)
    assert np.isclose(si, 0.9460830703671830, rtol=1e-9)
    assert np.isclose(ci, 0.3374039229009681, rtol=1e-9)


@pytest.mark.xfail(reason="returns nan")
def test_exprel():
    assert np.isclose(sp.exprel(1.0), np.expm1(1.0))


@pytest.mark.xfail(reason="returns poly1d with nan coefficients")
def test_chebyt():
    t = sp.chebyt(2)
    assert np.isclose(t(0.5), -0.5)  # T_2(0.5) = 2*(0.25) - 1 = -0.5
