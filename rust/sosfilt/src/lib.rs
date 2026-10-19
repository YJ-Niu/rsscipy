//! Rust port of `scipy/signal/_sosfilt.pyx`.
//!
//! Second-order sections (SOS) filter. The Python adapter writes the returned
//! arrays back to `x` and `zi`.
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
use pyo3::types::PyList;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Extract a C-contiguous 2-D f64 buffer.
fn extract_f64_2d<'py>(obj: &Bound<'py, PyAny>) -> PyResult<(PyBuffer<f64>, usize, usize)> {
    if obj.is_none() {
        return Err(PyTypeError::new_err("'x' must not be None"));
    }
    match PyBuffer::<f64>::get(obj) {
        Ok(buf) => {
            if buf.dimensions() != 2 || !buf.is_c_contiguous() {
                return Err(PyValueError::new_err("expected a C-contiguous 2-D buffer"));
            }
            let shape = buf.shape().to_vec();
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
            if buf.dimensions() != 3 || !buf.is_c_contiguous() {
                return Err(PyValueError::new_err("expected a C-contiguous 3-D buffer"));
            }
            let shape = buf.shape().to_vec();
            Ok((buf, shape[0], shape[1], shape[2]))
        }
        Err(_) => Err(PyTypeError::new_err(
            "expected 'const float64_t' 3-D buffer",
        )),
    }
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
    sos: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    zi: &Bound<'py, PyAny>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (sos_buf, n_sections, sos_cols) = extract_f64_2d(sos)?;
    if sos_cols != 6 {
        return Err(PyValueError::new_err("sos must have shape (n_sections, 6)"));
    }

    let (x_buf, n_signals, n_samples) = extract_f64_2d(x)?;

    let (zi_buf, zi_n_signals, zi_n_sections, zi_dim2) = extract_f64_3d(zi)?;
    if zi_n_signals != n_signals || zi_n_sections != n_sections || zi_dim2 != 2 {
        return Err(PyValueError::new_err(
            "zi must have shape (n_signals, n_sections, 2)",
        ));
    }

    let sos_data = unsafe {
        std::slice::from_raw_parts(sos_buf.buf_ptr() as *const f64, sos_buf.item_count())
    };
    let mut x_data = unsafe {
        std::slice::from_raw_parts(x_buf.buf_ptr() as *const f64, x_buf.item_count()).to_vec()
    };
    let mut zi_data = unsafe {
        std::slice::from_raw_parts(zi_buf.buf_ptr() as *const f64, zi_buf.item_count()).to_vec()
    };

    for i in 0..n_signals {
        for n in 0..n_samples {
            let mut x_cur = x_data[i * n_samples + n];

            for s in 0..n_sections {
                let sos_base = s * 6;
                let zi_base = i * n_sections * 2 + s * 2;
                let x_new = sos_data[sos_base] * x_cur + zi_data[zi_base];

                zi_data[zi_base] = sos_data[sos_base + 1] * x_cur - sos_data[sos_base + 4] * x_new
                    + zi_data[zi_base + 1];
                zi_data[zi_base + 1] =
                    sos_data[sos_base + 2] * x_cur - sos_data[sos_base + 5] * x_new;

                x_cur = x_new;
            }

            x_data[i * n_samples + n] = x_cur;
        }
    }

    Ok((x_data, zi_data))
}

#[derive(Clone, Copy)]
struct ComplexValue {
    re: f64,
    im: f64,
}

type ComplexFilterOutput = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

fn extract_complex_values(obj: &Bound<'_, PyAny>) -> PyResult<Vec<ComplexValue>> {
    let values = obj.cast::<PyList>()?;
    values
        .iter()
        .map(|value| {
            Ok(ComplexValue {
                re: value.getattr("real")?.extract()?,
                im: value.getattr("imag")?.extract()?,
            })
        })
        .collect()
}

fn multiply(a: ComplexValue, b: ComplexValue) -> ComplexValue {
    ComplexValue {
        re: a.re * b.re - a.im * b.im,
        im: a.re * b.im + a.im * b.re,
    }
}

fn add(a: ComplexValue, b: ComplexValue) -> ComplexValue {
    ComplexValue {
        re: a.re + b.re,
        im: a.im + b.im,
    }
}

fn subtract(a: ComplexValue, b: ComplexValue) -> ComplexValue {
    ComplexValue {
        re: a.re - b.re,
        im: a.im - b.im,
    }
}

#[pyfunction]
fn _sosfilt_complex<'py>(
    sos: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    zi: &Bound<'py, PyAny>,
    n_sections: usize,
    n_signals: usize,
    n_samples: usize,
) -> PyResult<ComplexFilterOutput> {
    let sos_data = extract_complex_values(sos)?;
    let mut x_data = extract_complex_values(x)?;
    let mut zi_data = extract_complex_values(zi)?;

    if sos_data.len() != n_sections * 6 {
        return Err(PyValueError::new_err("sos must have shape (n_sections, 6)"));
    }
    if x_data.len() != n_signals * n_samples {
        return Err(PyValueError::new_err(
            "x shape does not match n_signals and n_samples",
        ));
    }
    if zi_data.len() != n_signals * n_sections * 2 {
        return Err(PyValueError::new_err(
            "zi must have shape (n_signals, n_sections, 2)",
        ));
    }

    for signal in 0..n_signals {
        for sample in 0..n_samples {
            let mut x_cur = x_data[signal * n_samples + sample];
            for section in 0..n_sections {
                let sos_base = section * 6;
                let zi_base = signal * n_sections * 2 + section * 2;
                let x_new = add(multiply(sos_data[sos_base], x_cur), zi_data[zi_base]);
                zi_data[zi_base] = add(
                    subtract(
                        multiply(sos_data[sos_base + 1], x_cur),
                        multiply(sos_data[sos_base + 4], x_new),
                    ),
                    zi_data[zi_base + 1],
                );
                zi_data[zi_base + 1] = subtract(
                    multiply(sos_data[sos_base + 2], x_cur),
                    multiply(sos_data[sos_base + 5], x_new),
                );
                x_cur = x_new;
            }
            x_data[signal * n_samples + sample] = x_cur;
        }
    }

    let x_real = x_data.iter().map(|value| value.re).collect();
    let x_imag = x_data.iter().map(|value| value.im).collect();
    let zi_real = zi_data.iter().map(|value| value.re).collect();
    let zi_imag = zi_data.iter().map(|value| value.im).collect();
    Ok((x_real, x_imag, zi_real, zi_imag))
}

// ---------------------------------------------------------------------------
// module
// ---------------------------------------------------------------------------

#[pymodule]
fn _sosfilt_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_sosfilt_float, m)?)?;
    m.add_function(wrap_pyfunction!(_sosfilt_complex, m)?)?;
    Ok(())
}
