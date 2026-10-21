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


def test_ndimage_rank_filter_adapter():
    adapter = _load_adapter(
        "_rsscipy_test_rank_filter_1d", "ndimage/_rank_filter_1d.py"
    )
    values = np.asarray([3, 2, 5, 1, 4], dtype=np.int64)
    output = np.empty(values.shape, dtype=values.dtype)
    adapter.rank_filter(values, 1, 3, output, 2, 0, 0)
    assert output.tolist() == [3, 3, 2, 4, 4]

    floats = np.asarray([1.5, -2.0, 3.0], dtype=np.float32)
    float_output = np.empty(floats.shape, dtype=floats.dtype)
    adapter.rank_filter(floats, 1, 3, float_output, 4, 0.0, 0)
    assert float_output.tolist() == [0.0, 1.5, 0.0]


def test_directed_hausdorff_adapter():
    adapter = _load_adapter(
        "_rsscipy_test_hausdorff",
        "spatial/distance/_hausdorff.py",
    )

    points_a = np.asarray([[0.0, 0.0], [3.0, 0.0]])
    points_b = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    distance, index_a, index_b = adapter.directed_hausdorff(
        points_a, points_b, seed=0
    )
    assert distance == 2.0
    assert points_a[index_a].tolist() == [3.0, 0.0]
    assert points_b[index_b].tolist() == [1.0, 0.0]

    distance, index_a, index_b = adapter.directed_hausdorff(
        points_a, points_b, seed=np.random.default_rng(0)
    )
    assert distance == 2.0
    assert points_a[index_a].tolist() == [3.0, 0.0]
    assert points_b[index_b].tolist() == [1.0, 0.0]


@pytest.mark.parametrize("method", ["rvs_rcont1", "rvs_rcont2"])
@pytest.mark.parametrize("rng_kind", ["generator", "random_state"])
def test_random_contingency_table_adapter(method, rng_kind):
    adapter = _load_adapter(
        "_rsscipy_test_rcont", "stats/_rcont/rcont.py"
    )
    row = np.asarray([1, 3], dtype=np.int64)
    col = np.asarray([2, 1, 1], dtype=np.int64)
    if rng_kind == "generator":
        rng = np.random.default_rng(123)
    else:
        rng = np.random.RandomState(123)
    result = getattr(adapter, method)(
        row,
        col,
        4,
        12,
        rng,
    )
    assert result.shape == (12, 2, 3)
    assert np.all(result.sum(axis=2) == row)
    assert np.all(result.sum(axis=1) == col)


def test_random_contingency_table_degenerate_marginals():
    adapter = _load_adapter(
        "_rsscipy_test_rcont_degenerate", "stats/_rcont/rcont.py"
    )
    row = np.asarray([0, 1, 0], dtype=np.int64)
    col = np.asarray([1, 0, 0, 0], dtype=np.int64)
    result = adapter.rvs_rcont2(
        row, col, 1, 2, np.random.default_rng(10)
    )
    assert result.tolist() == [
        [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
    ]


def test_levy_stable_nolan_adapter():
    from rsscipy._rsscipy_rs import stats as rust_stats

    alpha_not_one = rust_stats.Nolan(0.5, 1.0, 1.0)
    assert np.isclose(alpha_not_one.zeta, -1.0)
    assert np.isclose(alpha_not_one.c1, 0.0)
    assert alpha_not_one.g(-alpha_not_one.xi) == 0.0
    assert alpha_not_one.g(np.pi / 2) == float("inf")

    alpha_one = rust_stats.Nolan(1.0, 0.5, 2.0)
    assert alpha_one.xi == np.pi / 2
    assert alpha_one.c1 == 0.0
    assert alpha_one.g(-np.pi / 2) == 0.0
    assert alpha_one.g(np.pi / 2) == float("inf")


def test_qmvnt_rust_kernels_for_one_dimension(monkeypatch):
    import math
    import types

    special = types.ModuleType("rsscipy.special")

    def normal_cdf(values):
        return np.asarray([
            0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))
            for value in values
        ])

    special.ndtr = normal_cdf
    special.ndtri = normal_cdf
    special.gammaincinv = lambda shape, values: np.full(len(values), 0.5)
    monkeypatch.setitem(sys.modules, "rsscipy.special", special)
    adapter = _load_adapter("_rsscipy_test_qmvnt", "stats/_qmvnt_cy.py")

    shifts = np.zeros((2, 1), dtype=np.float64)
    cholesky = np.asarray([[2.0]], dtype=np.float64)
    low = np.asarray([-1.0], dtype=np.float64)
    high = np.asarray([1.0], dtype=np.float64)

    normal_result = adapter._qmvn_inner(
        np.empty((0,), dtype=np.float64),
        shifts,
        3,
        2,
        cholesky,
        low,
        high,
    )
    expected = math.erf(0.5 / math.sqrt(2.0))
    assert np.isclose(normal_result[0], expected)
    assert normal_result[2] == 6

    student_cholesky = np.asarray([[1.0]], dtype=np.float64)
    student_result = adapter._qmvt_inner(
        np.asarray([0.5], dtype=np.float64),
        shifts,
        3,
        2,
        student_cholesky,
        low,
        high,
        2.0,
    )
    assert np.isclose(student_result[0], math.erf(1.0 / math.sqrt(2.0)))
    assert student_result[2] == 6


def test_qmvnt_rust_kernels_for_multiple_dimensions(monkeypatch):
    import math
    import types

    special = types.ModuleType("rsscipy.special")

    def normal_cdf(values):
        return np.asarray([
            0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))
            for value in values
        ])

    special.ndtr = normal_cdf
    special.ndtri = lambda values: np.asarray(values)
    special.gammaincinv = lambda shape, values: np.full(len(values), 0.5)
    monkeypatch.setitem(sys.modules, "rsscipy.special", special)
    adapter = _load_adapter(
        "_rsscipy_test_qmvnt_multi", "stats/_qmvnt_cy.py"
    )

    shifts = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    cholesky = np.asarray([[1.0, 0.0], [0.25, 1.0]], dtype=np.float64)
    low = np.asarray([-1.0, -0.5], dtype=np.float64)
    high = np.asarray([1.0, 0.75], dtype=np.float64)
    q = np.asarray([0.25], dtype=np.float64)

    for result in (
        adapter._qmvn_inner(
            q, shifts, 4, 2, cholesky, low, high
        ),
        adapter._qmvt_inner(
            np.asarray([0.25, 0.5]),
            shifts,
            4,
            2,
            cholesky,
            low,
            high,
            4.0,
        ),
    ):
        assert result[2] == 8
        assert np.isfinite(result[0])
        assert np.isfinite(result[1])


def test_qmc_rust_kernels_preserve_python_adapter_behavior():
    adapter = _load_adapter("_rsscipy_test_qmc_cy", "stats/_qmc_cy.py")
    sample = np.asarray(
        [[0.25, 0.75], [0.5, 0.25], [0.75, 0.5]],
        dtype=np.float64,
    )
    functions = (
        adapter._cy_wrapper_centered_discrepancy,
        adapter._cy_wrapper_wrap_around_discrepancy,
        adapter._cy_wrapper_mixture_discrepancy,
        adapter._cy_wrapper_l2_star_discrepancy,
    )

    for function in functions:
        serial = function(sample, False, 1)
        parallel = function(sample, False, 3)
        assert np.isfinite(serial)
        assert np.isclose(serial, parallel)

    initial = adapter._cy_wrapper_centered_discrepancy(sample, True, 1)
    new_point = np.asarray([0.125, 0.875], dtype=np.float64)
    updated = adapter._cy_wrapper_update_discrepancy(
        new_point, sample, initial
    )
    extended = np.concatenate((sample, new_point[None, :]), axis=0)
    expected = adapter._cy_wrapper_centered_discrepancy(
        extended, False, 1
    )
    assert np.isclose(updated, expected)

    expected_sequence = np.asarray(
        [2 / 3, 1 / 9, 4 / 9, 7 / 9, 2 / 9],
        dtype=np.float64,
    )
    assert np.allclose(
        adapter._cy_van_der_corput(5, 3, 2, 1),
        expected_sequence,
    )
    assert np.allclose(
        adapter._cy_van_der_corput(5, 3, 2, 4),
        expected_sequence,
    )

    permutations = np.asarray(
        [[0, 1, 2], [1, 2, 0], [2, 0, 1]],
        dtype=np.int64,
    )
    scrambled = adapter._cy_van_der_corput_scrambled(
        5, 3, 2, permutations, 1
    )
    scrambled_parallel = adapter._cy_van_der_corput_scrambled(
        5, 3, 2, permutations, 3
    )
    assert np.allclose(scrambled, scrambled_parallel)
    assert np.all((scrambled >= 0.0) & (scrambled <= 1.0))


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
def test_sobol_rust_adapter_initialization_draw_and_fast_forward(dtype):
    adapter = _load_adapter("_rsscipy_test_sobol", "stats/_sobol.py")
    direction = np.zeros((2, 4), dtype=dtype)
    adapter._initialize_v(direction, 2, 4)
    assert direction.tolist() == [[8, 4, 2, 1], [8, 12, 10, 15]]
    poly = adapter.get_poly_vinit("poly", dtype)
    vinit = adapter.get_poly_vinit("vinit", dtype)
    assert poly[:4].tolist() == [1, 3, 7, 11]
    assert vinit.shape == (21201, 18)
    assert vinit[1, 0] == 1

    quasi = np.zeros((2,), dtype=dtype)
    sample = np.empty((3, 2), dtype=np.float64)
    adapter._draw(
        3, 0, 2, 1 / 16, direction, quasi, sample
    )
    assert np.allclose(
        sample,
        [[0.5, 0.5], [0.75, 0.25], [0.25, 0.75]],
    )

    forwarded = np.zeros((2,), dtype=dtype)
    adapter._fast_forward(3, 0, 2, direction, forwarded)
    assert forwarded.tolist() == quasi.tolist()


def test_sobol_rust_adapter_scramble_and_categorical_kernels():
    adapter = _load_adapter(
        "_rsscipy_test_sobol_aux", "stats/_sobol.py"
    )
    ltm = np.asarray(
        [[[0, 0, 0], [1, 0, 0], [1, 1, 0]]],
        dtype=np.uint32,
    )
    sv = np.asarray([[1, 2, 4]], dtype=np.uint32)
    original_ltm = ltm.copy()
    original_sv = sv.copy()
    adapter._cscramble(1, 3, ltm, sv)
    assert np.all(np.diagonal(ltm[0]) == 1)
    expected_sv = []
    for direction in original_sv[0]:
        transformed = 0
        for row in range(2, -1, -1):
            row_value = sum(
                int(
                    original_ltm[0, row, column]
                    or row == column
                )
                << (2 - column)
                for column in range(3)
            )
            parity = sum(
                ((row_value >> bit) & 1) * ((int(direction) >> bit) & 1)
                for bit in range(3)
            ) % 2
            transformed += parity * (1 << (2 - row))
        expected_sv.append(transformed)
    assert sv.tolist() == [expected_sv]

    probabilities = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
    cumulative = np.empty_like(probabilities)
    adapter._fill_p_cumulative(probabilities, cumulative)
    assert np.allclose(cumulative, [0.2, 0.5, 1.0])

    result = np.zeros((3,), dtype=np.intp)
    adapter._categorize(
        np.asarray([0.0, 0.2, 0.4999, 0.5, 1.0]),
        cumulative,
        result,
    )
    assert result.tolist() == [2, 2, 1]
    assert adapter._test_find_index(cumulative, 3, 0.5) == 1
    assert [adapter._test_low_0_bit(value) for value in (0, 1, 2, 7)] == [
        1,
        2,
        1,
        4,
    ]

    narrow_result = np.zeros((3,), dtype=np.int32)
    adapter._categorize(
        np.asarray([0.0, 0.5, 1.0]),
        cumulative,
        narrow_result,
    )
    assert narrow_result.tolist() == [1, 1, 1]


def test_spherical_voronoi_rust_region_sorting_mutates_regions():
    adapter = _load_adapter(
        "_rsscipy_test_spherical_voronoi", "spatial/_voronoi.py"
    )
    simplices = np.asarray(
        [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
        dtype=np.int32,
    )
    regions = [[2, 0, 1], [3, 2, 0], [1, 3, 0], [3, 1, 2]]
    original_regions = [region.copy() for region in regions]
    adapter.sort_vertices_of_regions(simplices, regions)
    for generator, (region, original) in enumerate(
        zip(regions, original_regions)
    ):
        assert region[0] == original[0]
        assert sorted(region) == sorted(original)
        for current, following in zip(region, region[1:] + region[:1]):
            shared = set(simplices[current].tolist()) & set(
                simplices[following].tolist()
            )
            assert generator in shared
            assert len(shared) == 2


def test_spherical_voronoi_rust_region_sorting_validates_shapes():
    adapter = _load_adapter(
        "_rsscipy_test_spherical_voronoi_invalid",
        "spatial/_voronoi.py",
    )
    with pytest.raises(ValueError, match="three columns"):
        adapter.sort_vertices_of_regions(np.zeros((2, 4)), [[0]])

    simplices = np.asarray([[0, 1, 2]], dtype=np.int32)
    with pytest.raises(IndexError, match="invalid simplex index"):
        adapter.sort_vertices_of_regions(simplices, [[1]])


def test_matlab_mio_utils_rust_char_conversion():
    import types

    adapter = _load_adapter(
        "_rsscipy_test_mio_utils", "io/matlab/_mio_utils.py"
    )
    characters = ["a", "β", "🙂", " ", "x", "\x00"]
    encoded = "".join(characters).encode("utf-32-le")
    source = types.SimpleNamespace(
        shape=(2, 3),
        dtype=np.asarray(["a"], dtype="<U1").dtype,
        tobytes=lambda: encoded,
    )
    converted = adapter.chars_to_strings(source)
    assert converted.shape == (2,)
    assert converted.tolist() == ["aβ🙂", " x\x00"]

    source.shape = (1, 2, 3)
    converted = adapter.chars_to_strings(source)
    assert converted.shape == (1, 2)
    assert converted.tolist() == ["aβ🙂", " x\x00"]


def test_matlab_mio_utils_rust_char_conversion_empty_and_invalid():
    import types

    adapter = _load_adapter(
        "_rsscipy_test_mio_utils_empty",
        "io/matlab/_mio_utils.py",
    )
    empty = types.SimpleNamespace(
        shape=(2, 0),
        dtype=np.asarray(["x"], dtype="<U1").dtype,
        tobytes=lambda: b"",
    )
    result = adapter.chars_to_strings(empty)
    assert result.shape == (0,)
    assert result.dtype == empty.dtype

    invalid = types.SimpleNamespace(
        shape=(1,),
        dtype=empty.dtype,
        tobytes=lambda: (0x110000).to_bytes(4, "little"),
    )
    with pytest.raises(ValueError, match="invalid Unicode code point"):
        adapter.chars_to_strings(invalid)


def test_matlab_mio_utils_squeeze_adapter():
    adapter = _load_adapter(
        "_rsscipy_test_mio_utils_squeeze",
        "io/matlab/_mio_utils.py",
    )
    values = np.zeros((1, 3))
    assert np.array_equal(
        adapter.squeeze_element(values),
        np.squeeze(values),
    )
    scalar = adapter.squeeze_element(np.zeros((1, 1)))
    assert isinstance(scalar, float)

    empty = adapter.squeeze_element(np.empty((0,), dtype=np.uint8))
    assert empty.shape == (0,)
    assert empty.dtype == np.uint8
