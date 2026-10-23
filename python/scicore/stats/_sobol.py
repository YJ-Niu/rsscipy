import ast
from array import array
from pathlib import Path
import sys
from zipfile import ZipFile

import rsnumpy as np

from scicore._scicore_rs import stats as _rust_stats


_MAXDIM = _rust_stats._MAXDIM
_MAXDEG = _rust_stats._MAXDEG
_poly_dict = {}
_vinit_dict = {}
_direction_number_bytes = None


def _as_bytes(array):
    return np.ascontiguousarray(array).tobytes()


def _write_buffer(array, data):
    dtype = array.dtype
    format_codes = {
        ("u", 4): "I",
        ("u", 8): "Q",
        ("i", 4): "i",
        ("i", 8): "q",
        ("f", 4): "f",
        ("f", 8): "d",
    }
    format_code = format_codes.get((dtype.kind, dtype.itemsize))
    if format_code is None:
        raise TypeError(f"unsupported Sobol array dtype: {dtype}")
    values = np.asarray(memoryview(data).cast(format_code), dtype=dtype)
    array[...] = values.reshape(array.shape)


def _parse_npy(contents, expected_shape, expected_fortran_order):
    if contents[:6] != b"\x93NUMPY":
        raise ValueError("invalid Sobol direction-number archive")
    major = contents[6]
    if major == 1:
        header_length = int.from_bytes(contents[8:10], "little")
        header_start = 10
    elif major in (2, 3):
        header_length = int.from_bytes(contents[8:12], "little")
        header_start = 12
    else:
        raise ValueError("unsupported Sobol direction-number format")
    data_start = header_start + header_length
    header = ast.literal_eval(contents[header_start:data_start].decode("latin1"))
    if (
        header["descr"] not in ("<i8", "=i8")
        or tuple(header["shape"]) != expected_shape
        or header["fortran_order"] is not expected_fortran_order
    ):
        raise ValueError("unexpected Sobol direction-number array metadata")
    return contents[data_start:]


def _load_direction_number_bytes():
    global _direction_number_bytes
    if _direction_number_bytes is None:
        path = Path(__file__).with_name("_sobol_direction_numbers.npz")
        with ZipFile(path) as archive:
            poly = _parse_npy(
                archive.read("poly.npy"), (_MAXDIM,), False
            )
            vinit = _parse_npy(
                archive.read("vinit.npy"), (_MAXDIM, _MAXDEG), True
            )
        _direction_number_bytes = poly, vinit
    return _direction_number_bytes


def _direction_numbers(dtype):
    poly = _poly_dict.get(dtype)
    vinit = _vinit_dict.get(dtype)
    if poly is None or vinit is None:
        poly_bytes, vinit_bytes = _load_direction_number_bytes()
        poly_values = array("q")
        poly_values.frombytes(poly_bytes)
        vinit_values = array("q")
        vinit_values.frombytes(vinit_bytes)
        if sys.byteorder != "little":
            poly_values.byteswap()
            vinit_values.byteswap()
        poly = np.asarray(poly_values, dtype=dtype)
        vinit_row_major = array("q")
        for row in range(_MAXDIM):
            for column in range(_MAXDEG):
                vinit_row_major.append(vinit_values[row + column * _MAXDIM])
        vinit = np.asarray(vinit_row_major, dtype=dtype).reshape(
            (_MAXDIM, _MAXDEG)
        )
        _poly_dict[dtype] = poly
        _vinit_dict[dtype] = vinit
    return poly, vinit


def get_poly_vinit(kind, dtype):
    poly, vinit = _direction_numbers(dtype)
    if kind == "poly":
        return poly
    if kind == "vinit":
        return vinit
    raise ValueError(f"unknown direction-number kind: {kind!r}")


def _initialize_direction_numbers(poly, vinit, dtype):
    source_poly, source_vinit = _direction_numbers(dtype)
    poly[...] = source_poly
    vinit[...] = source_vinit


def _initialize_v(v, dim, bits):
    dtype = v.dtype
    poly, vinit = _load_direction_number_bytes()
    result = _rust_stats.sobol_initialize_v(
        poly,
        vinit,
        int(dim),
        int(bits),
        int(dtype.itemsize),
    )
    _write_buffer(v, result)


def _cscramble(dim, bits, ltm, sv):
    ltm_target = ltm
    sv_target = sv
    ltm = np.ascontiguousarray(ltm)
    sv = np.ascontiguousarray(sv)
    ltm_result, sv_result = _rust_stats.sobol_cscramble(
        int(dim),
        int(bits),
        _as_bytes(ltm),
        _as_bytes(sv),
        int(sv.dtype.itemsize),
    )
    _write_buffer(ltm_target, ltm_result)
    _write_buffer(sv_target, sv_result)


def _fill_p_cumulative(p, p_cumulative):
    p = np.ascontiguousarray(p, dtype=np.float64)
    result = _rust_stats.sobol_fill_p_cumulative(_as_bytes(p), len(p))
    _write_buffer(p_cumulative, result)


def _categorize(draws, p_cumulative, result):
    result_target = result
    draws = np.ascontiguousarray(draws, dtype=np.float64)
    p_cumulative = np.ascontiguousarray(p_cumulative, dtype=np.float64)
    result = np.ascontiguousarray(result)
    if result.dtype.kind != "i" or result.dtype.itemsize not in (4, 8):
        raise TypeError("categorical result must use int32 or int64 values")
    result_bytes = _rust_stats.sobol_categorize(
        _as_bytes(draws),
        _as_bytes(p_cumulative),
        result.tobytes(),
        len(result),
        int(result.dtype.itemsize),
    )
    _write_buffer(result_target, result_bytes)


def _draw(n, num_gen, dim, scale, sv, quasi, sample):
    quasi_target = quasi
    sample_target = sample
    sv = np.ascontiguousarray(sv)
    quasi = np.ascontiguousarray(quasi)
    sample = np.ascontiguousarray(sample, dtype=np.float64)
    quasi_result, sample_result = _rust_stats.sobol_draw(
        int(n),
        int(num_gen),
        int(dim),
        float(scale),
        _as_bytes(sv),
        _as_bytes(quasi),
        int(sv.dtype.itemsize),
    )
    _write_buffer(quasi_target, quasi_result)
    _write_buffer(sample_target, sample_result)

def _fast_forward(n, num_gen, dim, sv, quasi):
    quasi_target = quasi
    sv = np.ascontiguousarray(sv)
    quasi = np.ascontiguousarray(quasi)
    result = _rust_stats.sobol_fast_forward(
        int(n),
        int(num_gen),
        int(dim),
        _as_bytes(sv),
        _as_bytes(quasi),
        int(sv.dtype.itemsize),
    )
    _write_buffer(quasi_target, result)


def _test_find_index(p_cumulative, size, value):
    p_cumulative = np.ascontiguousarray(p_cumulative, dtype=np.float64)
    return _rust_stats.sobol_test_find_index(
        _as_bytes(p_cumulative), int(size), float(value)
    )


def _test_low_0_bit(value):
    return _rust_stats.sobol_test_low_0_bit(int(value))
