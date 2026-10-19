//! Level-3 BLAS kernels for `scipy.linalg._fblas`.
//!
//! SciPy's own wrappers force every input to Fortran order and then hand raw
//! pointers to BLAS.  rsnumpy is a row-major (C order) library, so the
//! conversion is done here instead, using the standard identity
//!
//! ```text
//! (A @ B).T == B.T @ A.T
//! ```
//!
//! A row-major buffer read back as column-major is exactly the transpose, so:
//!
//! * the row-major buffer of `op(A)` is fed as the second BLAS operand,
//! * the row-major buffer of `op(B)` is fed as the first BLAS operand,
//! * the output buffer is row-major `C` reinterpreted as column-major `C.T`,
//!
//! all with `'N', 'N'` transposes.  Operands that are *not* transposed by the
//! caller therefore need no copy at all.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

// Apple's Accelerate ships the reference BLAS/LAPACK.  Other platforms would
// `link(name = "openblas")` (or use the pure-Rust `matrixmultiply` crate for
// the `gemm` family only).
#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn sgemm_(
        transa: *const u8,
        transb: *const u8,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f32,
        a: *const f32,
        lda: *const i32,
        b: *const f32,
        ldb: *const i32,
        beta: *const f32,
        c: *mut f32,
        ldc: *const i32,
    );
    fn dgemm_(
        transa: *const u8,
        transb: *const u8,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        b: *const f64,
        ldb: *const i32,
        beta: *const f64,
        c: *mut f64,
        ldc: *const i32,
    );
}

/// `gemm` homogeneous type of level 3.
trait Blas: Sized + Copy {
    const ZERO: Self;
    /// Row-major buffer -> row-major buffer of the transpose.
    fn transpose_in_place_copy(src: &[Self], rows: usize, cols: usize) -> Vec<Self>;
    fn slice_from_bytes(buf: &[u8]) -> Vec<Self>;
    fn bytes_from_slice(v: &[Self]) -> Vec<u8>;
    /// SAFETY: the caller guarantees `c` has room for `m * n` elements and that
    /// every pointer is valid for its declared leading dimension.
    // BLAS signatures are long by nature; splitting them would only obscure
    // the one-to-one mapping onto the reference implementation.
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_nn(
        m: i32,
        n: i32,
        k: i32,
        alpha: Self,
        a: *const Self,
        lda: i32,
        b: *const Self,
        ldb: i32,
        beta: Self,
        c: *mut Self,
        ldc: i32,
    );
}

macro_rules! impl_blas {
    ($T:ty, $gemm:ident, $size:expr) => {
        impl Blas for $T {
            const ZERO: Self = 0.0 as _;

            fn transpose_in_place_copy(src: &[Self], rows: usize, cols: usize) -> Vec<Self> {
                let mut out = Vec::with_capacity(rows * cols);
                for col in 0..cols {
                    for row in 0..rows {
                        out.push(src[row * cols + col]);
                    }
                }
                out
            }

            fn slice_from_bytes(buf: &[u8]) -> Vec<Self> {
                buf.chunks_exact($size)
                    .map(|c| {
                        let mut arr = [0u8; $size];
                        arr.copy_from_slice(c);
                        Self::from_le_bytes(arr)
                    })
                    .collect()
            }

            fn bytes_from_slice(v: &[Self]) -> Vec<u8> {
                v.iter().flat_map(|x| x.to_le_bytes()).collect()
            }

            unsafe fn gemm_nn(
                m: i32,
                n: i32,
                k: i32,
                alpha: Self,
                a: *const Self,
                lda: i32,
                b: *const Self,
                ldb: i32,
                beta: Self,
                c: *mut Self,
                ldc: i32,
            ) {
                let notrans = b'N';
                unsafe {
                    $gemm(
                        &notrans, &notrans, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
                    )
                }
            }
        }
    };
}

#[cfg(target_os = "macos")]
impl_blas!(f32, sgemm_, 4);
#[cfg(target_os = "macos")]
impl_blas!(f64, dgemm_, 8);

#[cfg(not(target_os = "macos"))]
compile_error!("define a BLAS backend for this platform (see the `link` block above)");

/// gemm_core(a_bytes, a_shape, trans_a, b_bytes, b_shape, trans_b, alpha, beta,
///           c_bytes, c_shape) -> bytes
///
/// All buffers are contiguous little-endian row-major; `c_bytes` may be empty
/// when there is no output array yet, in which case `beta` is passed as 0.
// See `gemm_nn`: this mirrors the BLAS argument list on purpose.
#[allow(clippy::too_many_arguments)]
fn gemm_core<T: Blas>(
    a_bytes: &[u8],
    a_shape: (usize, usize),
    trans_a: i32,
    b_bytes: &[u8],
    b_shape: (usize, usize),
    trans_b: i32,
    alpha: T,
    beta: T,
    c_bytes: &[u8],
) -> PyResult<Vec<u8>> {
    let a = T::slice_from_bytes(a_bytes);
    let b = T::slice_from_bytes(b_bytes);

    let (a_rows, a_cols) = a_shape;
    let (b_rows, b_cols) = b_shape;

    // op(A) has shape (m, k); op(B) has shape (k, n)
    let (m, k) = if trans_a != 0 {
        (a_cols, a_rows)
    } else {
        (a_rows, a_cols)
    };
    let k_expect = if trans_b != 0 { b_cols } else { b_rows };
    if k_expect != k {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch in input arrays: op(A) has inner dimension {k} \
             but op(B) has {k_expect}"
        )));
    }
    let n = if trans_b != 0 { b_rows } else { b_cols };

    if a.len() != a_rows * a_cols || b.len() != b_rows * b_cols {
        return Err(PyValueError::new_err(
            "buffer size does not match the declared shape",
        ));
    }

    let have_c = !c_bytes.is_empty();
    if have_c && c_bytes.len() != m * n * std::mem::size_of::<T>() {
        return Err(PyValueError::new_err(format!(
            "output array has the wrong size: {} bytes, expected {}",
            c_bytes.len(),
            m * n * std::mem::size_of::<T>()
        )));
    }

    // Row-major buffer of op(A) / op(B).  An untransposed operand is reusable
    // as is; a transposed one needs the explicit copy.
    let opa: Vec<T> = match trans_a {
        0 => a.clone(),
        1 | 2 => T::transpose_in_place_copy(&a, a_rows, a_cols),
        other => {
            return Err(PyValueError::new_err(format!(
                "invalid trans_a value {other}"
            )))
        }
    };
    let opb: Vec<T> = match trans_b {
        0 => b.clone(),
        1 | 2 => T::transpose_in_place_copy(&b, b_rows, b_cols),
        other => {
            return Err(PyValueError::new_err(format!(
                "invalid trans_b value {other}"
            )))
        }
    };

    // `trans == 2` asks for a conjugate transpose; both dtypes here are real,
    // so conjugation is the identity and only the transpose above applies.

    let mut out: Vec<T> = if have_c {
        // c is stored row-major (m, n) but BLAS writes it as column-major
        // (n, m), which is the same buffer, so nothing has to move.
        T::slice_from_bytes(c_bytes)
    } else {
        vec![T::ZERO; m * n]
    };

    // Reinterpret row-major op(B) as column-major (n, k) and row-major op(A) as
    // column-major (k, m); the product lands in column-major (n, m) == row-major C.
    let bm: i32 = n as i32; // rows of the first BLAS operand
    let bn: i32 = m as i32; // columns of the result
    let bk: i32 = k as i32;
    let lda = if n == 0 { 1 } else { bm };
    let ldb = if k == 0 { 1 } else { bk };
    let ldc = if n == 0 { 1 } else { bm };
    let beta_eff = if have_c { beta } else { T::ZERO };

    unsafe {
        T::gemm_nn(
            bm,
            bn,
            bk,
            alpha,
            opb.as_ptr(),
            lda,
            opa.as_ptr(),
            ldb,
            beta_eff,
            out.as_mut_ptr(),
            ldc,
        )
    }

    Ok(T::bytes_from_slice(&out))
}

#[pyfunction]
#[pyo3(signature = (a_bytes, a_shape, trans_a, b_bytes, b_shape, trans_b, alpha, beta, c_bytes))]
#[allow(clippy::too_many_arguments)]
fn gemm_f32(
    a_bytes: &Bound<'_, PyBytes>,
    a_shape: (usize, usize),
    trans_a: i32,
    b_bytes: &Bound<'_, PyBytes>,
    b_shape: (usize, usize),
    trans_b: i32,
    alpha: f32,
    beta: f32,
    c_bytes: &Bound<'_, PyBytes>,
) -> PyResult<Vec<u8>> {
    gemm_core::<f32>(
        a_bytes.as_bytes(),
        a_shape,
        trans_a,
        b_bytes.as_bytes(),
        b_shape,
        trans_b,
        alpha,
        beta,
        c_bytes.as_bytes(),
    )
}

#[pyfunction]
#[pyo3(signature = (a_bytes, a_shape, trans_a, b_bytes, b_shape, trans_b, alpha, beta, c_bytes))]
#[allow(clippy::too_many_arguments)]
fn gemm_f64(
    a_bytes: &Bound<'_, PyBytes>,
    a_shape: (usize, usize),
    trans_a: i32,
    b_bytes: &Bound<'_, PyBytes>,
    b_shape: (usize, usize),
    trans_b: i32,
    alpha: f64,
    beta: f64,
    c_bytes: &Bound<'_, PyBytes>,
) -> PyResult<Vec<u8>> {
    gemm_core::<f64>(
        a_bytes.as_bytes(),
        a_shape,
        trans_a,
        b_bytes.as_bytes(),
        b_shape,
        trans_b,
        alpha,
        beta,
        c_bytes.as_bytes(),
    )
}

#[pymodule]
fn _fblas_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gemm_f32, m)?)?;
    m.add_function(wrap_pyfunction!(gemm_f64, m)?)?;
    Ok(())
}
