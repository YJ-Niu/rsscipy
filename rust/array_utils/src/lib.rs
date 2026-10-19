//! Rust port of `scipy/linalg/_cythonized_array_utils.pyx`.
//!
//! Two data paths are offered because rsnumpy only exposes a **zero-copy
//! buffer for float64 arrays** (`BufferError` for every other dtype) and refuses
//! `__array_interface__` for complex data:
//!
//! * `*_f64` take `memoryview`s of float64 data and read the raw memory without
//!   copying (complex arrays are passed as two float64 views, real/imaginary
//!   parts, which is how `.real` / `.imag` come back from rsnumpy).
//! * `is_sym_her_bytes` takes the `bytes` blob from `__array_interface__` and
//!   decodes it element by element, preserving exact comparison semantics for
//!   integer dtypes that would survive a round trip through float64.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

// --------------------------------------------------------------------------
// Little-endian element readers used by the `bytes` fallback.  `from_le_bytes`
// makes no assumption about the alignment of the incoming buffer.
// --------------------------------------------------------------------------
#[inline]
fn rd_u8(b: &[u8], i: usize) -> u8 {
    b[i]
}
#[inline]
fn rd_i8(b: &[u8], i: usize) -> i8 {
    i8::from_le_bytes([b[i]])
}
#[inline]
fn rd_u16(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes(b[i * 2..i * 2 + 2].try_into().unwrap())
}
#[inline]
fn rd_i16(b: &[u8], i: usize) -> i16 {
    i16::from_le_bytes(b[i * 2..i * 2 + 2].try_into().unwrap())
}
#[inline]
fn rd_u32(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())
}
#[inline]
fn rd_i32(b: &[u8], i: usize) -> i32 {
    i32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())
}
#[inline]
fn rd_u64(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap())
}
#[inline]
fn rd_i64(b: &[u8], i: usize) -> i64 {
    i64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap())
}
#[inline]
fn rd_f32(b: &[u8], i: usize) -> f32 {
    f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())
}
#[inline]
fn rd_f64(b: &[u8], i: usize) -> f64 {
    f64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap())
}

/// Compare the lower triangle of `A` against its transpose: `A[r, c] == A[c, r]`.
///
/// Comparing with `!=` deliberately keeps the IEEE behaviour of the Cython
/// version: `nan` is never symmetric, while `inf` compares equal to itself.
fn check_symmetric<T, G>(get: G, n: usize) -> bool
where
    T: PartialEq,
    G: Fn(usize) -> T,
{
    for r in 0..n {
        let row = r * n;
        for c in 0..=r {
            if get(row + c) != get(c * n + r) {
                return false;
            }
        }
    }
    true
}

/// The same traversal against the conjugate transpose: `A[r, c] == conj(A[c, r])`.
fn check_hermitian<T, G>(get: G, n: usize) -> bool
where
    T: PartialEq + std::ops::Neg<Output = T>,
    G: Fn(usize) -> (T, T),
{
    for r in 0..n {
        let row = r * n;
        for c in 0..=r {
            let (ar, ai) = get(row + c);
            let (br, bi) = get(c * n + r);
            if ar != br || ai != -bi {
                return false;
            }
        }
    }
    true
}

/// Borrow a C-contiguous float64 buffer as a slice, without copying.
fn f64_slice(view: &PyBuffer<f64>, want_items: usize) -> PyResult<&[f64]> {
    if !view.is_c_contiguous() {
        return Err(PyValueError::new_err("expected a C-contiguous buffer"));
    }
    if view.item_count() < want_items {
        return Err(PyValueError::new_err(format!(
            "buffer holds {} elements, expected at least {}",
            view.item_count(),
            want_items
        )));
    }
    // SAFETY: `PyBuffer<f64>::get` validated the format/itemsize, the view is
    // alive for as long as it is borrowed here, and we never mutate through it.
    Ok(unsafe { std::slice::from_raw_parts(view.buf_ptr() as *const f64, want_items) })
}

/// is_sym_f64(view, n) -> bool
///
/// Exact symmetry test over a zero-copy float64 buffer.
#[pyfunction]
fn is_sym_f64(view: PyBuffer<f64>, n: usize) -> PyResult<bool> {
    let data = f64_slice(&view, n * n)?;
    Ok(check_symmetric(|i: usize| data[i], n))
}

/// is_complex_f64(re, im, n, hermitian) -> bool
///
/// Exact symmetry / Hermitian test for a complex array supplied as two
/// float64 buffers (real and imaginary parts).
#[pyfunction]
#[pyo3(signature = (re, im, n, hermitian))]
fn is_complex_f64(
    re: PyBuffer<f64>,
    im: PyBuffer<f64>,
    n: usize,
    hermitian: bool,
) -> PyResult<bool> {
    let rr = f64_slice(&re, n * n)?;
    let ii = f64_slice(&im, n * n)?;
    if hermitian {
        Ok(check_hermitian(|i: usize| (rr[i], ii[i]), n))
    } else {
        // tuples implement PartialEq, so no conjugation happens here
        Ok(check_symmetric(|i: usize| (rr[i], ii[i]), n))
    }
}

/// Expand to a `match` over the real dtype codes supported by `np_numeric_t`.
macro_rules! real_dispatch {
    ($buf:expr, $n:expr, $code:expr) => {
        match $code {
            "b1" | "u1" => check_symmetric(|i: usize| rd_u8($buf, i), $n),
            "i1" => check_symmetric(|i: usize| rd_i8($buf, i), $n),
            "u2" => check_symmetric(|i: usize| rd_u16($buf, i), $n),
            "i2" => check_symmetric(|i: usize| rd_i16($buf, i), $n),
            "u4" => check_symmetric(|i: usize| rd_u32($buf, i), $n),
            "i4" => check_symmetric(|i: usize| rd_i32($buf, i), $n),
            "u8" => check_symmetric(|i: usize| rd_u64($buf, i), $n),
            "i8" => check_symmetric(|i: usize| rd_i64($buf, i), $n),
            "f4" => check_symmetric(|i: usize| rd_f32($buf, i), $n),
            "f8" => check_symmetric(|i: usize| rd_f64($buf, i), $n),
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "Unsupported dtype {:?} for exact comparison",
                    $code
                )))
            }
        }
    };
}

/// is_sym_her_bytes(buffer, n, typestr, hermitian) -> bool
///
/// Exact symmetry / Hermitian test over the `bytes` blob returned by
/// `__array_interface__['data']`.  Only needed for dtypes that are neither
/// float64 nor complex; it keeps integer comparisons exact.
#[pyfunction]
#[pyo3(signature = (buffer, n, typestr, _hermitian))]
fn is_sym_her_bytes(
    buffer: &Bound<'_, PyBytes>,
    n: usize,
    typestr: &str,
    _hermitian: bool,  // conjugation is the identity for real data
) -> PyResult<bool> {
    // typestr looks like "<f8" / "=c16" / "|u1": byte-order + type code
    let order = typestr.chars().next().unwrap_or('<');
    if !matches!(order, '<' | '=' | '|') {
        return Err(PyTypeError::new_err(format!(
            "Unsupported byte order {:?} in typestr {:?}",
            order, typestr
        )));
    }
    let code = &typestr[order.len_utf8()..];
    if code.starts_with('c') {
        return Err(PyTypeError::new_err(
            "complex arrays cannot be decoded from __array_interface__ bytes; \
             pass float64 real/imaginary buffers instead",
        ));
    }

    if n == 0 {
        return Ok(true);
    }

    let buf = buffer.as_bytes();
    // conjugation is the identity for real data
    Ok(real_dispatch!(buf, n, code))
}

#[pymodule]
fn _array_utils_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_sym_f64, m)?)?;
    m.add_function(wrap_pyfunction!(is_complex_f64, m)?)?;
    m.add_function(wrap_pyfunction!(is_sym_her_bytes, m)?)?;
    Ok(())
}
