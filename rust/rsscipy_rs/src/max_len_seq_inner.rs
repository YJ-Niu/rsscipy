//! Rust port of `scipy/signal/_max_len_seq_inner.pyx`.
//!
//! Maximum Length Sequence (MLS) generator using a linear feedback shift
//! register.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

fn extract_integral_i64(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    if let Ok(value) = obj.extract::<i64>() {
        return Ok(value);
    }
    let value = obj.extract::<f64>()?;
    if !value.is_finite()
        || value.fract() != 0.0
        || value < i64::MIN as f64
        || value >= 9_223_372_036_854_775_808.0
    {
        return Err(PyValueError::new_err("expected an integer value"));
    }
    Ok(value as i64)
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Extract a 1-D i32 buffer (for taps).
fn extract_i32_slice<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Vec<i32>> {
    if obj.is_none() {
        return Err(PyTypeError::new_err("'taps' must not be None"));
    }
    // Fast path: buffer.
    if let Ok(buf) = PyBuffer::<i32>::get(obj) {
        if buf.dimensions() != 1 {
            return Err(PyValueError::new_err(
                "wrong number of dimensions, expected 1",
            ));
        }
        if buf.is_c_contiguous() {
            return Ok(unsafe {
                std::slice::from_raw_parts(buf.buf_ptr() as *const i32, buf.item_count())
            }
            .to_vec());
        }
    }
    // Fallback: list via tolist().
    let values: Bound<'py, pyo3::types::PyList> = if obj.is_instance_of::<pyo3::types::PyList>() {
        obj.cast::<pyo3::types::PyList>()?.clone()
    } else {
        let list = obj.call_method0("tolist")?;
        list.cast::<pyo3::types::PyList>()?.clone()
    };
    let mut result = Vec::with_capacity(values.len());
    for item in values.iter() {
        let v = i32::try_from(extract_integral_i64(&item)?)
            .map_err(|_| PyValueError::new_err("tap values must fit in int32"))?;
        result.push(v);
    }
    Ok(result)
}

/// Extract a 1-D i8 buffer (for state/seq).
fn extract_i8_slice<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Vec<i8>> {
    if obj.is_none() {
        return Err(PyTypeError::new_err("'state' must not be None"));
    }
    if let Ok(buf) = PyBuffer::<i8>::get(obj) {
        if buf.dimensions() != 1 {
            return Err(PyValueError::new_err(
                "wrong number of dimensions, expected 1",
            ));
        }
        if buf.is_c_contiguous() {
            return Ok(unsafe {
                std::slice::from_raw_parts(buf.buf_ptr() as *const i8, buf.item_count())
            }
            .to_vec());
        }
    }
    let values: Bound<'py, pyo3::types::PyList> = if obj.is_instance_of::<pyo3::types::PyList>() {
        obj.cast::<pyo3::types::PyList>()?.clone()
    } else {
        let list = obj.call_method0("tolist")?;
        list.cast::<pyo3::types::PyList>()?.clone()
    };
    let mut result = Vec::with_capacity(values.len());
    for item in values.iter() {
        let v = i8::try_from(extract_integral_i64(&item)?)
            .map_err(|_| PyValueError::new_err("state and seq values must fit in int8"))?;
        result.push(v);
    }
    Ok(result)
}

fn extract_nonnegative_usize(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<usize> {
    let index = obj
        .call_method0("__index__")
        .map_err(|_| PyTypeError::new_err(format!("{name} must be an integer")))?;
    let value = extract_integral_i64(&index)?;
    usize::try_from(value)
        .map_err(|_| PyValueError::new_err(format!("{name} must be non-negative")))
}

// ---------------------------------------------------------------------------
// _max_len_seq_inner
// ---------------------------------------------------------------------------

/// Compute MLS using a shift register with ring-buffer indexing.
///
/// Inputs:
///   taps   — 1-D array of tap positions (int32)
///   state  — 1-D array of length nbits (int8, values 0 or 1)
///   nbits  — number of bits (intp)
///   length — number of output samples to compute (intp)
///   seq    — 1-D output array of length at least `length` (int8)
///
/// Returns:
///   sequence and final state of the shift register
#[pyfunction]
fn _max_len_seq_inner<'py>(
    taps: &Bound<'py, PyAny>,
    state: &Bound<'py, PyAny>,
    nbits: &Bound<'py, PyAny>,
    length: &Bound<'py, PyAny>,
    seq: &Bound<'py, PyAny>,
) -> PyResult<(Vec<i8>, Vec<i8>)> {
    let taps_data = extract_i32_slice(taps)?;
    let mut state_data = extract_i8_slice(state)?;
    let mut seq_data = extract_i8_slice(seq)?;

    let nbits = extract_nonnegative_usize(nbits, "nbits")?;
    let length = extract_nonnegative_usize(length, "length")?;

    if nbits == 0 {
        return Err(PyValueError::new_err("nbits must be greater than zero"));
    }

    if state_data.len() != nbits {
        return Err(PyValueError::new_err("state must have length nbits"));
    }
    if seq_data.len() < length {
        return Err(PyValueError::new_err(
            "seq must have length at least `length`",
        ));
    }
    if taps_data.iter().any(|&tap| tap < 0 || tap as usize > nbits) {
        return Err(PyValueError::new_err(
            "tap positions must be between zero and nbits",
        ));
    }

    let mut idx = 0usize;
    for output in seq_data.iter_mut().take(length) {
        let mut feedback = state_data[idx];
        *output = feedback;
        for &tap_pos in &taps_data {
            let tap_pos = tap_pos as usize;
            feedback ^= state_data[(tap_pos + idx) % nbits];
        }
        state_data[idx] = feedback;
        idx = (idx + 1) % nbits;
    }
    // Roll state so that next run, when idx==0, it's in the right place.
    // Equivalent to np.roll(state, -idx).
    let mut rolled = vec![0i8; nbits];
    for i in 0..nbits {
        rolled[i] = state_data[(i + idx) % nbits];
    }

    Ok((seq_data[..length].to_vec(), rolled))
}

// ---------------------------------------------------------------------------
// module
// ---------------------------------------------------------------------------

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_max_len_seq_inner, m)?)?;
    Ok(())
}
