//! Rust port of `scipy/signal/_sosfilt.pyx`.
//!
//! Second-order sections (SOS) filter.  Modifies `x` and `zi` in place.
//!
//! For each signal `i` and sample `n`:
//!   x_cur = x[i, n]
//!   for each section `s`:
//!     x_new = sos[s, 0] * x_cur + zi[s, 0]
//!     zi[s, 0] = sos[s, 1] * x_cur - sos[s, 4] * x_new + zi[s, 1]
//!     zi[s, 1] = sos[s, 2] * x_cur - sos[s, 5] * x_new
//!     x_cur = x_new
//!   x[i, n] = x_cur

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Extract a 2-D f64 buffer (C-contiguous in the last dimension).
fn extract_f64_2d<'py>(obj: &Bound<'py, PyAny>) -> PyResult<(PyBuffer<f64>, usize, usize)> {
    if obj.is_none() {
        return Err(PyTypeError::new_err("'x' must not be None"));
    }
    match PyBuffer::<f64>::get(obj) {
        Ok(buf) => {
            if buf.dimensions() != 2 {
                return Err(PyValueError::new_err(
                    "expected 2-D buffer",
                ));
            }
            let shape = buf.shape().to_vec();
            if shape.len() != 2 {
                return Err(PyValueError::new_err(
                    "expected 2-D buffer",
                ));
            }
            Ok((buf, shape[0], shape[1]))
        }
        Err(_) => Err(PyTypeError::new_err(
            "expected 'const float64_t' 2-D buffer",
        )),
    }
}

/// Extract a 3-D f64 buffer.
fn extract_f64_3d<'py>(obj: &Bound<'py, PyAny>) -> PyResult<(PyBuffer<f64>, usize, usize, usize)> {
    if obj.is_none() {
        return Err(PyTypeError::new_err("'x' must not be None"));
    }
    match PyBuffer::<f64>::get(obj) {
        Ok(buf) => {
            if buf.dimensions() != 3 {
                return Err(PyValueError::new_err(
                    "expected 3-D buffer",
                ));
            }
            let shape = buf.shape().to_vec();
            if shape.len() != 3 {
                return Err(PyValueError::new_err(
                    "expected 3-D buffer",
                ));
            }
            Ok((buf, shape[0], shape[1], shape[2]))
        }
        Err(_) => Err(PyTypeError::new_err(
            "expected 'const float64_t' 3-D buffer",
        )),
    }
}

unsafe fn f64_slice(buf: &PyBuffer<f64>) -> &[f64] {
    std::slice::from_raw_parts(buf.buf_ptr() as *const f64, buf.item_count())
}

// ---------------------------------------------------------------------------
// _sosfilt_float (f64)
// ---------------------------------------------------------------------------

/// Second-order sections filter for float64.
///
/// `sos` has shape (n_sections, 6), `x` has shape (n_signals, n_samples),
/// `zi` has shape (n_signals, n_sections, 2).  `x` and `zi` are modified
/// in place.
#[pyfunction]
fn _sosfilt_float<'py>(
    py: Python<'py>,
    sos: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    zi: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let (sos_buf, n_sections, sos_cols) = extract_f64_2d(sos)?;
    if sos_cols != 6 {
        return Err(PyValueError::new_err(
            "sos must have shape (n_sections, 6)",
        ));
    }

    let (x_buf, n_signals, n_samples) = extract_f64_2d(x)?;

    let (zi_buf, zi_n_signals, zi_n_sections, zi_dim2) = extract_f64_3d(zi)?;
    if zi_n_signals != n_signals || zi_n_sections != n_sections || zi_dim2 != 2 {
        return Err(PyValueError::new_err(
            "zi must have shape (n_signals, n_sections, 2)",
        ));
    }

    let sos_data = unsafe { f64_slice(&sos_buf) };
    let x_data = unsafe { f64_slice(&x_buf) };
    let zi_data = unsafe { f64_slice(&zi_buf) };

    // Strides: sos is (n_sections, 6), x is (n_signals, n_samples),
    // zi is (n_signals, n_sections, 2).  We assume C-contiguous last dim.
    // sos stride: row stride = 6, element stride = 1
    // x stride: row stride = n_samples, element stride = 1
    // zi stride: section stride = 2, element stride = 1

    // We need mutable access to x and zi.  Since PyBuffer only gives us a
    // pointer, we cast to mutable (the caller guarantees the arrays are
    // writable).
    let x_data_mut: &mut [f64] =
        unsafe { std::slice::from_raw_parts_mut(x_buf.buf_ptr() as *mut f64, x_buf.item_count()) };
    let zi_data_mut: &mut [f64] =
        unsafe { std::slice::from_raw_parts_mut(zi_buf.buf_ptr() as *mut f64, zi_buf.item_count()) };

    for i in 0..n_signals {
        for n in 0..n_samples {
            let mut x_cur = x_data_mut[i * n_samples + n];

            for s in 0..n_sections {
                let sos_base = s * 6;
                let x_new = sos_data[sos_base + 0] * x_cur
                    + zi_data_mut[i * n_sections * 2 + s * 2 + 0];

                zi_data_mut[i * n_sections * 2 + s * 2 + 0] =
                    sos_data[sos_base + 1] * x_cur
                    - sos_data[sos_base + 4] * x_new
                    + zi_data_mut[i * n_sections * 2 + s * 2 + 1];
                zi_data_mut[i * n_sections * 2 + s * 2 + 1] =
                    sos_data[sos_base + 2] * x_cur
                    - sos_data[sos_base + 5] * x_new;

                x_cur = x_new;
            }

            x_data_mut[i * n_samples + n] = x_cur;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// module
// ---------------------------------------------------------------------------

#[pymodule]
fn _sosfilt_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_sosfilt_float, m)?)?;
    Ok(())
}
