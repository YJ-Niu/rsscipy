"""Verification suite for the ``numpy`` -> ``rsnumpy`` migration of SciPy.

Run it with::

    python -m pytest tools/test_rsnumpy_migration.py -v

It is deliberately self-contained: it loads
``python/rsscipy/_rsnumpy_compat.py`` by path so it can also be executed before
the compiled extension modules are built.
"""

import compileall
import importlib.util
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_DIRS = ('python/rsscipy', 'tools', 'benchmarks')

# `numpy` that is *not* part of a dotted name (jax.numpy), a path
# (github.com/numpy/numpy, <numpy/arrayobject.h>) or a URL (numpy.org).
RESIDUAL_NUMPY = re.compile(r'(?<![\w./])numpy(?![\w/])')


@pytest.fixture(scope='module')
def compat():
    """Import (and thereby install) the rsnumpy compatibility layer."""
    path = ROOT / 'python' / 'rsscipy' / '_rsnumpy_compat.py'
    spec = importlib.util.spec_from_file_location('_rsnumpy_compat', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['_rsnumpy_compat'] = module
    spec.loader.exec_module(module)
    return module


def _iter_python_files():
    """All ported sources.

    ``python/rsscipy/_rsnumpy_compat.py`` is excluded on purpose: it documents
    the NumPy functions it re-implements, so ``numpy`` legitimately appears in
    its docstrings.  This verification file is excluded for the same reason.
    """
    # `_rsnumpy_compat.py` and ports like `_cythonized_array_utils.py` describe
    # their NumPy-equivalent behaviour in docstrings, so `numpy` legitimately
    # appears there as prose (never as working code).
    excluded = {
        pathlib.Path(__file__).name,
        '_rsnumpy_compat.py',
        '_cythonized_array_utils.py',
    }
    for name in SOURCE_DIRS:
        base = ROOT / name
        if not base.is_dir():
            continue
        for pattern in ('*.py', '*.pyi'):
            for path in base.rglob(pattern):
                # Vendored sources were not ported as part of this migration
                # and may legitimately depend on NumPy.
                if {'.venv', 'subprojects', '_external'} & set(path.parts):
                    continue
                if path.name in excluded:
                    continue
                yield path


# --------------------------------------------------------------------------
# 1. the migration itself
# --------------------------------------------------------------------------

def test_no_numpy_references_left_in_python_sources():
    offenders = []
    for path in _iter_python_files():
        text = path.read_text(encoding='utf-8', errors='ignore')
        for i, line in enumerate(text.splitlines(), start=1):
            if not RESIDUAL_NUMPY.search(line):
                continue
            # `array_api_compat.numpy` is a submodule of a third-party package,
            # not the NumPy library itself, so its name must stay as is.
            if 'array_api_compat' in line or 'numpy as np_compat' in line:
                continue
            offenders.append(f'{path.relative_to(ROOT)}:{i}: {line.strip()}')
    assert not offenders, 'numpy still referenced:\n' + '\n'.join(offenders[:20])


def test_no_numpy_import_statements():
    # `[\w.]*numpy\b` must not match the `numpy` inside `rsnumpy`
    pattern = re.compile(r'^\s*(?:from|import|cimport)\s+([\w.]+)', re.M)
    offenders = []
    for path in _iter_python_files():
        text = path.read_text(encoding='utf-8', errors='ignore')
        for m in pattern.finditer(text):
            module = m.group(1)
            if RESIDUAL_NUMPY.search(module):
                line = text[:m.start()].count('\n') + 1
                offenders.append(f'{path.relative_to(ROOT)}:{line}: {m.group(0)}')
    assert not offenders, 'numpy imports still present:\n' + '\n'.join(offenders[:20])


def test_all_python_sources_compile():
    ok = compileall.compile_dir(
        str(ROOT / 'python' / 'rsscipy'), quiet=2, force=True,
    )
    assert ok, 'some SciPy sources do not byte-compile'


# --------------------------------------------------------------------------
# 2. the rsnumpy compatibility layer
# --------------------------------------------------------------------------

def test_rsnumpy_testing_module_registered(compat):
    import rsnumpy
    assert hasattr(rsnumpy, 'testing')
    from rsnumpy.testing import (  # noqa: F401
        assert_, assert_allclose, assert_almost_equal, assert_approx_equal,
        assert_array_almost_equal, assert_array_almost_equal_nulp,
        assert_array_equal, assert_array_less, assert_array_max_ulp,
        assert_equal, assert_no_warnings, assert_raises, assert_string_equal,
        assert_warns, TestCase, HAS_REFCOUNT,
    )


def test_assertion_helpers_behave_like_numpy(compat):
    import rsnumpy as np
    from rsnumpy.testing import (
        assert_allclose, assert_almost_equal, assert_approx_equal,
        assert_array_almost_equal, assert_array_equal, assert_array_less,
        assert_equal, assert_,
    )

    assert_allclose([1.0, 2.0, np.nan], [1.0, 2.0 + 1e-12, np.nan])
    assert_equal([1, 2, 3], [1, 2, 3])
    assert_equal({'a': [1, 2]}, {'a': [1, 2]})
    assert_equal(np.array([np.nan]), np.array([np.nan]))
    assert_(True)
    assert_array_equal(np.arange(5), [0, 1, 2, 3, 4])
    assert_array_almost_equal([1.0], [1.0 + 1e-9])
    assert_almost_equal(1.0, 1.0 + 1e-12)
    assert_approx_equal(1.0, 1.0 + 1e-12)
    assert_array_less([1, 2], [2, 3])


def test_assertion_helpers_raise_on_mismatch(compat):
    import rsnumpy as np
    from rsnumpy.testing import (
        assert_allclose, assert_array_equal, assert_array_less, assert_equal,
    )
    with pytest.raises(AssertionError):
        assert_allclose([1.0], [2.0])
    with pytest.raises(AssertionError):
        assert_array_equal([1, 2], [1, 3])
    with pytest.raises(AssertionError):
        assert_array_less([3], [2])
    with pytest.raises(AssertionError):
        assert_equal([1, 2], [1, 2, 3])
    with pytest.raises(AssertionError):
        assert_allclose(np.zeros(3), np.zeros(4))


def test_rsnumpy_bug_workarounds(compat):
    """rsnumpy 1.2.4 ships two bugs that break SciPy; both are patched here."""
    import rsnumpy as np

    a = np.array([1.0, 1.0, 1.0])
    # bug 1: allclose(array, scalar) returned False
    assert np.allclose(a, 1.0)
    assert np.allclose(a, np.array(1.0))
    assert np.allclose(np.ones((2, 3)), 1.0)
    assert not np.allclose(np.array([1.0, 2.0]), 1.0)
    assert np.allclose(np.array([np.nan]), np.array([np.nan]), equal_nan=True)

    # bug 2: where() rejected scalar branches
    cond = np.array([True, False, True])
    assert np.allclose(np.where(cond, np.array([1.0, 2.0, 3.0]), 0.0),
                       [1.0, 0.0, 3.0])
    assert np.allclose(np.where(cond, 1.0, 0.0), [1.0, 0.0, 1.0])
    assert tuple(np.where(cond)[0].tolist()) == (0, 2)


def test_missing_top_level_aliases(compat):
    import rsnumpy as np
    for name in ('ufunc', 'fromfile', 'matrix_power', 'float128', 'float96'):
        assert hasattr(np, name), f'rsnumpy.{name} is still missing'


def test_foreign_array_adoption(compat):
    """Arrays returned by NumPy-based C extensions must be usable."""
    import rsnumpy as np

    class FakeForeignArray:
        """Mimics a numpy.ndarray as seen from rsnumpy."""

        def __init__(self, data):
            self._data = data

        def tolist(self):
            return self._data

        @property
        def __array_interface__(self):
            return {}

    foreign = FakeForeignArray([[1.0, 2.0], [3.0, 4.0]])
    out = np.asarray(foreign)
    assert out.shape == (2, 2)
    assert float(out.sum()) == 10.0


# --------------------------------------------------------------------------
# 3. numerical smoke test of the array API SciPy depends on
# --------------------------------------------------------------------------

def test_rsnumpy_numerical_smoke(compat):
    import rsnumpy as np

    a = np.array([[3.0, 1.0], [1.0, 2.0]])
    # `rsnumpy.linalg` hands back a low level Rust array, not `rsnumpy.ndarray`
    inv_a = np.asarray(np.linalg.inv(a))
    assert np.allclose(inv_a @ a, np.eye(2))
    assert np.isclose(float(np.linalg.det(a)), 5.0)
    assert np.allclose(np.asarray(np.linalg.solve(a, np.array([9.0, 8.0]))),
                       [2.0, 3.0])
    assert np.allclose(np.sort(np.asarray(np.linalg.eigvals(a)).real),
                       [1.382, 3.618], atol=1e-3)

    x = np.linspace(0.0, 2 * np.pi, 64)
    assert np.allclose(np.sin(x) ** 2 + np.cos(x) ** 2, 1.0)
    assert np.allclose(np.fft.irfft(np.fft.rfft(np.sin(x))), np.sin(x))

    assert int(np.sum(np.arange(10))) == 45
    assert np.array_equal(np.sort(np.array([3, 1, 2])), [1, 2, 3])
    assert np.allclose(np.mean(np.array([[1.0, 2.0], [3.0, 4.0]])), 2.5)
    assert np.allclose(np.convolve([1, 2, 3], [1, 1]), [1, 3, 5, 3])
    assert np.allclose(np.polyval([1, 0, -1], 3.0), 8.0)

    rng = np.random.default_rng(0)
    r = rng.random(1000)
    assert 0.4 < float(np.mean(r)) < 0.6


def test_rsnumpy_dtype_and_error_semantics(compat):
    import rsnumpy as np

    assert np.asarray([1, 2, 3]).dtype == np.int64
    assert np.asarray([1.0]).dtype == np.float64
    assert np.result_type(np.int64, np.float64) == np.float64

    with np.errstate(divide='ignore'):
        assert np.isinf(np.array([1.0]) / 0.0)[0]

    with pytest.raises(Exception):
        np.asarray([1.0, 2.0]).reshape(3)
