import importlib.util
import sys
from pathlib import Path

import pytest
import rsnumpy as np
import rsscipy


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python" / "rsscipy"


def _load_adapter(name, relative_path):
    path = PYTHON_DIR / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_rsscipy_package_import():
    assert rsscipy.__version__


def test_column_grouping_adapters():
    adapter = _load_adapter(
        "_rsscipy_test_group_columns", "optimize/_group_columns.py"
    )
    dense = np.asarray([
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0],
    ])
    assert adapter.group_dense(7, 6, dense).tolist() == [0, 1, 2, 0, 1, 2]

    groups = adapter.group_sparse(
        3,
        4,
        np.asarray([0, 1, 2, 2]),
        np.asarray([0, 1, 2, 3, 4]),
    )
    assert groups.tolist() == [0, 0, 0, 1]


def test_givens_elimination_adapter_mutates_arrays():
    adapter = _load_adapter(
        "_rsscipy_test_givens_elimination",
        "optimize/_lsq/givens_elimination.py",
    )
    matrix = np.asarray([[3.0]])
    vector = np.asarray([5.0])
    assert adapter.givens_elimination(matrix, vector, np.asarray([4.0])) is None
    assert np.allclose(matrix, [[5.0]])
    assert np.allclose(vector, [3.0])


def test_regular_grid_interpolation_adapter():
    adapter = _load_adapter(
        "_rsscipy_test_rgi", "interpolate/_rgi_cython.py"
    )
    xi = np.asarray([[0.25], [0.75]])
    indices, distances = adapter.find_indices(
        [np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])],
        xi,
    )
    out = np.empty((1,), dtype=np.float64)
    result = adapter.evaluate_linear_2d(
        np.asarray([[0.0, 1.0], [2.0, 3.0]]),
        indices,
        distances,
        None,
        out,
    )
    assert np.allclose(result, [1.25])


def test_ndimage_label_adapter_and_output_overflow():
    adapter = _load_adapter("_rsscipy_test_ni_label", "ndimage/_ni_label.py")
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    source = np.asarray([[1, 0, 1, 0, 1]], dtype=np.uint8)
    output = np.zeros(source.shape, dtype=np.int32)
    assert adapter._label(source, structure, output) == 3
    assert output.tolist() == [[1, 0, 2, 0, 3]]

    source = np.zeros((1, 511), dtype=np.uint8)
    source[0, ::2] = 1
    narrow_output = np.zeros(source.shape, dtype=np.uint8)
    with pytest.raises(adapter.NeedMoreBits):
        adapter._label(source, structure, narrow_output)

    wide_output = np.zeros(source.shape, dtype=np.intp)
    assert adapter._label(source, structure, wide_output) == 256
    assert wide_output[0, -1] == 256
