use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyList};

fn decode(bytes: &[u8], name: &str) -> PyResult<Vec<f64>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(PyValueError::new_err(format!(
            "{name} buffer length is not a multiple of float64 size"
        )));
    }
    Ok(bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| f64::from_ne_bytes(*chunk))
        .collect())
}

fn apply_special<'py>(
    py: Python<'py>,
    func: &Bound<'py, PyAny>,
    values: &[f64],
) -> PyResult<Vec<f64>> {
    let input = PyList::new(py, values)?;
    let output = func.call1((input,))?;
    output.call_method0("tolist")?.extract::<Vec<f64>>()
}

fn apply_gammaincinv<'py>(
    py: Python<'py>,
    func: &Bound<'py, PyAny>,
    shape: f64,
    values: &[f64],
) -> PyResult<Vec<f64>> {
    let input = PyList::new(py, values)?;
    let output = func.call1((shape, input))?;
    output.call_method0("tolist")?.extract::<Vec<f64>>()
}

#[allow(clippy::too_many_arguments)]
fn validate_inputs(
    q: &[f64],
    rndm: &[f64],
    samples: usize,
    batches: usize,
    cho: &[f64],
    lo: &[f64],
    hi: &[f64],
    q_dims: usize,
) -> PyResult<usize> {
    let n = lo.len();
    let cho_len = n
        .checked_mul(n)
        .ok_or_else(|| PyValueError::new_err("Cholesky matrix is too large"))?;
    let rndm_len = batches
        .checked_mul(n)
        .ok_or_else(|| PyValueError::new_err("QMC shift matrix is too large"))?;
    if n == 0 || hi.len() != n || cho.len() != cho_len || rndm.len() != rndm_len {
        return Err(PyValueError::new_err(
            "inconsistent QMC kernel input shapes",
        ));
    }
    if q.len() != q_dims {
        return Err(PyValueError::new_err("QMC lattice has the wrong dimension"));
    }
    samples
        .checked_mul(batches)
        .ok_or_else(|| PyValueError::new_err("QMC sample count overflows"))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn qmvn_inner(
    py: Python<'_>,
    q_bytes: &Bound<'_, PyBytes>,
    rndm_bytes: &Bound<'_, PyBytes>,
    samples: usize,
    batches: usize,
    cho_bytes: &Bound<'_, PyBytes>,
    lo_bytes: &Bound<'_, PyBytes>,
    hi_bytes: &Bound<'_, PyBytes>,
    phi: &Bound<'_, PyAny>,
    phinv: &Bound<'_, PyAny>,
) -> PyResult<(f64, f64, usize)> {
    let q = decode(q_bytes.as_bytes(), "q")?;
    let rndm = decode(rndm_bytes.as_bytes(), "rndm")?;
    let cho = decode(cho_bytes.as_bytes(), "cho")?;
    let lo = decode(lo_bytes.as_bytes(), "lo")?;
    let hi = decode(hi_bytes.as_bytes(), "hi")?;
    let n = lo.len();
    let q_dims = n
        .checked_sub(1)
        .ok_or_else(|| PyValueError::new_err("QMC dimension must be positive"))?;
    let n_samples = validate_inputs(&q, &rndm, samples, batches, &cho, &lo, &hi, q_dims)?;

    let first_scale = cho[0];
    if samples == 0 || batches == 0 {
        return Ok((0.0, 0.0, n_samples));
    }
    let first_c = apply_special(py, phi, &[lo[0] / first_scale])?[0];
    let first_dc = apply_special(py, phi, &[hi[0] / first_scale, lo[0] / first_scale])?;
    let first_dc = first_dc[0] - first_dc[1];

    let mut probability = 0.0;
    let mut error_variance = 0.0;
    let mut y = vec![0.0; (n - 1) * samples];

    for batch in 0..batches {
        let mut c = vec![first_c; samples];
        let mut dc = vec![first_dc; samples];
        let mut pv = vec![first_dc; samples];

        for i in 1..n {
            let random_shift = rndm[batch * n + i];
            let q_i = q[i - 1];
            let mut targets = Vec::with_capacity(samples);
            for k in 0..samples {
                let z = q_i * (k + 1) as f64 + random_shift;
                let fraction = z - z.floor();
                let tent = (2.0 * fraction - 1.0).abs();
                targets.push(c[k] + tent * dc[k]);
            }
            let mut y_i = apply_special(py, phinv, &targets)?;
            for value in &mut y_i {
                if *value == f64::INFINITY {
                    *value = 1e100;
                } else if *value == f64::NEG_INFINITY {
                    *value = -1e100;
                }
            }
            y[(i - 1) * samples..i * samples].copy_from_slice(&y_i);

            let mut lower_inputs = Vec::with_capacity(samples);
            let mut upper_inputs = Vec::with_capacity(samples);
            for k in 0..samples {
                let mut sum = 0.0;
                for previous in 0..i {
                    sum += cho[i * n + previous] * y[previous * samples + k];
                }
                let scale = cho[i * n + i];
                lower_inputs.push((lo[i] - sum) / scale);
                upper_inputs.push((hi[i] - sum) / scale);
            }
            c = apply_special(py, phi, &lower_inputs)?;
            let d = apply_special(py, phi, &upper_inputs)?;
            for k in 0..samples {
                dc[k] = d[k] - c[k];
                pv[k] *= dc[k];
            }
        }
        let batch_mean = pv.iter().sum::<f64>() / samples as f64;
        let delta = (batch_mean - probability) / (batch + 1) as f64;
        probability += delta;
        error_variance = (batch as f64 - 1.0) * error_variance / (batch + 1) as f64 + delta * delta;
    }

    Ok((probability, 3.0 * error_variance.sqrt(), n_samples))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn qmvt_inner(
    py: Python<'_>,
    q_bytes: &Bound<'_, PyBytes>,
    rndm_bytes: &Bound<'_, PyBytes>,
    samples: usize,
    batches: usize,
    cho_bytes: &Bound<'_, PyBytes>,
    lo_bytes: &Bound<'_, PyBytes>,
    hi_bytes: &Bound<'_, PyBytes>,
    nu: f64,
    phi: &Bound<'_, PyAny>,
    phinv: &Bound<'_, PyAny>,
    gammaincinv: &Bound<'_, PyAny>,
) -> PyResult<(f64, f64, usize)> {
    let q = decode(q_bytes.as_bytes(), "q")?;
    let rndm = decode(rndm_bytes.as_bytes(), "rndm")?;
    let cho = decode(cho_bytes.as_bytes(), "cho")?;
    let lo = decode(lo_bytes.as_bytes(), "lo")?;
    let hi = decode(hi_bytes.as_bytes(), "hi")?;
    let n = lo.len();
    let n_samples = validate_inputs(&q, &rndm, samples, batches, &cho, &lo, &hi, n)?;
    if samples == 0 || batches == 0 {
        return Ok((0.0, 0.0, n_samples));
    }

    let mut probability = 0.0;
    let mut error_variance = 0.0;
    let mut s = vec![0.0; n * samples];

    for batch in 0..batches {
        let mut pv = vec![1.0; samples];
        s.fill(0.0);
        let initial_q = q[0];
        let initial_shift = rndm[batch * n];
        let mut gamma_inputs = Vec::with_capacity(samples);
        for k in 0..samples {
            let z = initial_q * (k + 1) as f64 + initial_shift;
            let tent = (2.0 * (z - z.floor()) - 1.0).abs();
            gamma_inputs.push(tent);
        }
        let scaling = if nu > 0.0 {
            let gamma = apply_gammaincinv(py, gammaincinv, nu / 2.0, &gamma_inputs)?;
            gamma.iter().map(|value| (2.0 * value).sqrt()).collect()
        } else {
            vec![1.0; samples]
        };
        if scaling.len() != samples {
            return Err(PyValueError::new_err(
                "gammaincinv returned an unexpected result shape",
            ));
        }
        let mut c = vec![1.0; samples];
        let mut dc = vec![0.0; samples];

        for i in 0..n {
            let q_i = q[i];
            let random_shift = rndm[batch * n + i];
            if i > 0 {
                let mut targets = Vec::with_capacity(samples);
                for k in 0..samples {
                    let z = q_i * (k + 1) as f64 + random_shift;
                    let tent = (2.0 * (z - z.floor()) - 1.0).abs();
                    targets.push(c[k] + tent * dc[k]);
                }
                let y = apply_special(py, phinv, &targets)?;
                for row in i..n {
                    let coefficient = cho[row * n + i - 1];
                    for k in 0..samples {
                        s[row * samples + k] += coefficient * y[k];
                    }
                }
            }

            let mut lower_inputs = Vec::with_capacity(samples);
            let mut upper_inputs = Vec::with_capacity(samples);
            for k in 0..samples {
                lower_inputs.push(lo[i] * scaling[k] - s[i * samples + k]);
                upper_inputs.push(hi[i] * scaling[k] - s[i * samples + k]);
            }
            c = clipped_phi(py, phi, &lower_inputs)?;
            let d = clipped_phi(py, phi, &upper_inputs)?;
            for k in 0..samples {
                dc[k] = d[k] - c[k];
                pv[k] *= dc[k];
            }
        }
        let batch_mean = pv.iter().sum::<f64>() / samples as f64;
        let delta = (batch_mean - probability) / (batch + 1) as f64;
        probability += delta;
        error_variance = (batch as f64 - 1.0) * error_variance / (batch + 1) as f64 + delta * delta;
    }

    Ok((probability, 3.0 * error_variance.sqrt(), n_samples))
}

fn clipped_phi(py: Python<'_>, phi: &Bound<'_, PyAny>, inputs: &[f64]) -> PyResult<Vec<f64>> {
    let mut clipped = inputs.to_vec();
    for value in &mut clipped {
        *value = value.clamp(-9.0, 9.0);
    }
    let mut values = apply_special(py, phi, &clipped)?;
    for (value, input) in values.iter_mut().zip(inputs) {
        if *input < -9.0 {
            *value = 0.0;
        } else if *input >= 9.0 {
            *value = 1.0;
        }
    }
    Ok(values)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(qmvn_inner, module)?)?;
    module.add_function(wrap_pyfunction!(qmvt_inner, module)?)?;
    Ok(())
}
