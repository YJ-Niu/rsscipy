use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

fn vector(value: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    value
        .cast::<PyList>()?
        .iter()
        .map(|item| item.extract::<f64>())
        .collect()
}

fn convolve_packed(input: &[f64], kernel: &[f64], swap_real_imag: bool) -> PyResult<Vec<f64>> {
    let n = input.len();
    if n == 0 {
        return Err(PyValueError::new_err("input arrays must not be empty"));
    }
    if kernel.len() != n {
        return Err(PyValueError::new_err(
            "inout and omega must be 1-dimensional arrays of the same length",
        ));
    }
    let mut output = input.to_vec();
    if swap_real_imag {
        output[0] *= kernel[0];
        for index in (1..n.saturating_sub(1)).step_by(2) {
            let real = output[index] * kernel[index];
            output[index] = output[index + 1] * kernel[index + 1];
            output[index + 1] = real;
        }
        if n.is_multiple_of(2) {
            output[n - 1] *= kernel[n - 1];
        }
    } else {
        for (value, coefficient) in output.iter_mut().zip(kernel) {
            *value *= coefficient;
        }
    }
    Ok(output)
}

fn convolve_complex_packed(
    input: &[f64],
    omega_real: &[f64],
    omega_imag: &[f64],
) -> PyResult<Vec<f64>> {
    let n = input.len();
    if n == 0 {
        return Err(PyValueError::new_err("input arrays must not be empty"));
    }
    if omega_real.len() != n || omega_imag.len() != n {
        return Err(PyValueError::new_err(
            "inout and omega must be 1-dimensional arrays of the same length",
        ));
    }
    let mut output = input.to_vec();
    output[0] *= omega_real[0] + omega_imag[0];
    if n.is_multiple_of(2) {
        output[n - 1] *= omega_real[n - 1] + omega_imag[n - 1];
    }
    for index in (1..n.saturating_sub(1)).step_by(2) {
        let real = output[index] * omega_real[index] + output[index + 1] * omega_imag[index + 1];
        let imag = output[index + 1] * omega_real[index + 1] + output[index] * omega_imag[index];
        output[index] = real;
        output[index + 1] = imag;
    }
    Ok(output)
}

fn evaluate_kernel(
    py: Python<'_>,
    kernel_func: &Bound<'_, PyAny>,
    frequency: usize,
    extra_args: &[Bound<'_, PyAny>],
) -> PyResult<f64> {
    let mut args = Vec::with_capacity(extra_args.len() + 1);
    args.push(frequency.into_pyobject(py)?.into_any().unbind());
    args.extend(extra_args.iter().map(|argument| argument.clone().unbind()));
    let args = PyTuple::new(py, args)?;
    kernel_func.call(&args, None)?.extract()
}

#[pyfunction]
fn _convolve_packed(
    input: &Bound<'_, PyAny>,
    kernel: &Bound<'_, PyAny>,
    swap_real_imag: bool,
) -> PyResult<Vec<f64>> {
    convolve_packed(&vector(input)?, &vector(kernel)?, swap_real_imag)
}

#[pyfunction]
fn _convolve_complex_packed(
    input: &Bound<'_, PyAny>,
    omega_real: &Bound<'_, PyAny>,
    omega_imag: &Bound<'_, PyAny>,
) -> PyResult<Vec<f64>> {
    convolve_complex_packed(&vector(input)?, &vector(omega_real)?, &vector(omega_imag)?)
}

#[pyfunction]
#[pyo3(signature = (n, kernel_func, d=0, zero_nyquist=None, kernel_func_extra_args=None))]
fn _init_convolution_kernel(
    py: Python<'_>,
    n: usize,
    kernel_func: &Bound<'_, PyAny>,
    d: isize,
    zero_nyquist: Option<bool>,
    kernel_func_extra_args: Option<&Bound<'_, PyTuple>>,
) -> PyResult<Vec<f64>> {
    if n == 0 {
        return Err(PyValueError::new_err("n must be positive"));
    }
    let zero_nyquist = zero_nyquist.unwrap_or(d % 2 != 0);
    let extra_args = kernel_func_extra_args.map_or_else(Vec::new, |args| args.iter().collect());
    let mut output = vec![0.0; n];
    output[0] = evaluate_kernel(py, kernel_func, 0, &extra_args)? / n as f64;

    let derivative = d % 4;
    let scale_real = if derivative <= 1 { 1.0 } else { -1.0 } / n as f64;
    let scale_imag = if derivative == 0 || derivative == 3 {
        1.0
    } else {
        -1.0
    } / n as f64;

    let last_odd_index = if n.is_multiple_of(2) { n - 1 } else { n };
    let mut frequency = 1;
    for index in (1..last_odd_index).step_by(2) {
        let value = evaluate_kernel(py, kernel_func, frequency, &extra_args)?;
        output[index] = scale_real * value;
        output[index + 1] = scale_imag * value;
        frequency += 1;
    }
    if n.is_multiple_of(2) {
        output[n - 1] = if zero_nyquist {
            0.0
        } else {
            let value = evaluate_kernel(py, kernel_func, frequency, &extra_args)?;
            scale_real * value
        };
    }
    Ok(output)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_convolve_packed, module)?)?;
    module.add_function(wrap_pyfunction!(_convolve_complex_packed, module)?)?;
    module.add_function(wrap_pyfunction!(_init_convolution_kernel, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{convolve_complex_packed, convolve_packed};

    #[test]
    fn multiplies_packed_real_and_complex_spectra() {
        assert_eq!(
            convolve_packed(&[2.0, 3.0, 4.0, 5.0], &[7.0, 11.0, 13.0, 17.0], false).unwrap(),
            [14.0, 33.0, 52.0, 85.0]
        );
        assert_eq!(
            convolve_packed(&[2.0, 3.0, 4.0, 5.0], &[7.0, 11.0, 13.0, 17.0], true).unwrap(),
            [14.0, 52.0, 33.0, 85.0]
        );
        assert_eq!(
            convolve_complex_packed(
                &[2.0, 3.0, 4.0, 5.0],
                &[7.0, 11.0, 13.0, 17.0],
                &[19.0, 23.0, 29.0, 31.0]
            )
            .unwrap(),
            [52.0, 149.0, 121.0, 240.0]
        );
    }
}
