//! Dense double-precision LAPACK kernels for `scipy.linalg._flapack`.
//!
//! SciPy's wrappers force every input to Fortran order before handing it to
//! LAPACK.  rsnumpy is row-major, so each kernel transposes its input into
//! column-major on the way in and back on the way out.  That costs `O(n^2)`
//! against LAPACK's `O(n^3)`, which is negligible and keeps the index
//! arithmetic obvious.
//!
//! Porting order follows what the rest of `scipy.linalg` actually needs first:
//! the LU family (`getrf`, `getrs`, `getri`) underpins `lu`, `solve` and `inv`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn dgetrf_(
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *mut i32,
        info: *mut i32,
    );
    fn dgetrs_(
        trans: *const u8,
        n: *const i32,
        nrhs: *const i32,
        a: *const f64,
        lda: *const i32,
        ipiv: *const i32,
        b: *mut f64,
        ldb: *const i32,
        info: *mut i32,
    );
    fn dgetri_(
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *const i32,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
}

fn to_f64(buf: &[u8]) -> PyResult<Vec<f64>> {
    if !buf.len().is_multiple_of(8) {
        return Err(PyValueError::new_err(
            "float64 buffer length is not a multiple of 8",
        ));
    }
    Ok(buf
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(c);
            f64::from_le_bytes(arr)
        })
        .collect())
}

fn to_bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Row-major `(rows, cols)` -> column-major buffer of the same matrix, i.e. the
/// transposition both directions need.
fn transpose<T: Copy>(src: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            out.push(src[row * cols + col]);
        }
    }
    out
}

fn leading_dim(rows: usize) -> i32 {
    i32::try_from(rows.max(1)).unwrap_or(i32::MAX)
}

/// getrf(a_bytes, m, n) -> (lu_bytes, pivots, info)
///
/// LU factorisation `A = P L U` with partial pivoting.  `pivots` is returned
/// **0-based**, matching SciPy's convention; LAPACK itself is 1-based.
#[pyfunction]
fn getrf_f64<'py>(
    py: Python<'py>,
    a_bytes: &Bound<'_, PyBytes>,
    m: usize,
    n: usize,
) -> PyResult<(Bound<'py, PyBytes>, Vec<i32>, i32)> {
    let row_major = to_f64(a_bytes.as_bytes())?;
    if row_major.len() != m * n {
        return Err(PyValueError::new_err(
            "buffer size does not match the declared shape",
        ));
    }

    let mut a_cm = transpose(&row_major, m, n);
    let lda = leading_dim(m);
    let npiv = m.min(n);
    let mut pivots = vec![0i32; npiv];
    let mut info: i32 = 0;

    unsafe {
        dgetrf_(
            &(m as i32),
            &(n as i32),
            a_cm.as_mut_ptr(),
            &lda,
            pivots.as_mut_ptr(),
            &mut info,
        );
    }

    for p in pivots.iter_mut() {
        *p -= 1; // SciPy hands 0-based permutation indices back to Python
    }

    let lu = PyBytes::new(py, &to_bytes(&transpose(&a_cm, n, m)));
    Ok((lu, pivots, info))
}

/// getrs(lu_bytes, n, pivots, b_bytes, nrhs, trans) -> (x_bytes, info)
///
/// Solves `op(A) x = b` from the `getrf` factorisation; `trans` is SciPy's
/// 0 = N / 1 = T / 2 = C.
#[pyfunction]
fn getrs_f64<'py>(
    py: Python<'py>,
    lu_bytes: &Bound<'_, PyBytes>,
    n: usize,
    pivots: Vec<i32>,
    b_bytes: &Bound<'_, PyBytes>,
    nrhs: usize,
    trans: i32,
) -> PyResult<(Bound<'py, PyBytes>, i32)> {
    let b_row = to_f64(b_bytes.as_bytes())?;
    if b_row.len() != n * nrhs {
        return Err(PyValueError::new_err(
            "b shape is incompatible with the factorisation",
        ));
    }

    let trans_char: u8 = match trans {
        0 => b'N',
        1 => b'T',
        2 => b'C',
        other => {
            return Err(PyValueError::new_err(format!(
                "invalid trans value {other}"
            )))
        }
    };

    let lu_row = to_f64(lu_bytes.as_bytes())?;
    if lu_row.len() != n * n {
        return Err(PyValueError::new_err("lu must be square"));
    }
    if pivots.len() != n {
        return Err(PyValueError::new_err("pivots length must match n"));
    }

    // The factorisation is used as-is for `trans == 0`.  SciPy's caller always
    // supplies the LU of A itself, so the row-major -> column-major transpose is
    // all that differs from the Fortran layout LAPACK wants.
    let lu_cm = transpose(&lu_row, n, n);
    let mut b_cm = transpose(&b_row, n, nrhs);
    let shifted: Vec<i32> = pivots.iter().map(|p| p + 1).collect();

    let lda = leading_dim(n);
    let ldb = leading_dim(n);
    let mut info: i32 = 0;

    unsafe {
        dgetrs_(
            &trans_char,
            &(n as i32),
            &(nrhs as i32),
            lu_cm.as_ptr(),
            &lda,
            shifted.as_ptr(),
            b_cm.as_mut_ptr(),
            &ldb,
            &mut info,
        );
    }

    let x = PyBytes::new(py, &to_bytes(&transpose(&b_cm, nrhs, n)));
    Ok((x, info))
}

/// getri(lu_bytes, n, pivots, lwork) -> (inv_bytes, info)
#[pyfunction]
fn getri_f64<'py>(
    py: Python<'py>,
    lu_bytes: &Bound<'_, PyBytes>,
    n: usize,
    pivots: Vec<i32>,
    lwork: usize,
) -> PyResult<(Bound<'py, PyBytes>, i32)> {
    let lu_row = to_f64(lu_bytes.as_bytes())?;
    if lu_row.len() != n * n {
        return Err(PyValueError::new_err("lu must be square"));
    }
    if pivots.len() != n {
        return Err(PyValueError::new_err("pivots length must match n"));
    }
    let lwork_i = i32::try_from(lwork)
        .map_err(|_| PyValueError::new_err("lwork does not fit in a 32-bit BLAS integer"))?;
    if lwork_i < n as i32 {
        return Err(PyValueError::new_err(format!(
            "lwork={lwork_i} is too small, need at least {n}"
        )));
    }

    let mut lu_cm = transpose(&lu_row, n, n);
    let shifted: Vec<i32> = pivots.iter().map(|p| p + 1).collect();
    let mut work = vec![0.0f64; lwork];
    let lda = leading_dim(n);
    let mut info: i32 = 0;

    unsafe {
        dgetri_(
            &(n as i32),
            lu_cm.as_mut_ptr(),
            &lda,
            shifted.as_ptr(),
            work.as_mut_ptr(),
            &lwork_i,
            &mut info,
        );
    }

    let inv = PyBytes::new(py, &to_bytes(&transpose(&lu_cm, n, n)));
    Ok((inv, info))
}

/// getri_lwork(n) -> (optimal_lwork, info): the workspace size query.
#[pyfunction]
fn getri_lwork_f64(n: usize) -> PyResult<(f64, i32)> {
    let mut work = 0.0f64;
    let mut info: i32 = 0;
    let n_i = n as i32;
    let lda = leading_dim(n);
    unsafe {
        dgetri_(
            &n_i,
            std::ptr::null_mut(),
            &lda,
            std::ptr::null(),
            &mut work,
            &-1,
            &mut info,
        );
    }
    Ok((work, info))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(getrf_f64, m)?)?;
    m.add_function(wrap_pyfunction!(getrs_f64, m)?)?;
    m.add_function(wrap_pyfunction!(getri_f64, m)?)?;
    m.add_function(wrap_pyfunction!(getri_lwork_f64, m)?)?;
    Ok(())
}
