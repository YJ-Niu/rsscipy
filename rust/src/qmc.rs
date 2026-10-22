use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn decode_f64(bytes: &[u8], name: &str) -> PyResult<Vec<f64>> {
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

fn decode_i64(bytes: &[u8], name: &str) -> PyResult<Vec<i64>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(PyValueError::new_err(format!(
            "{name} buffer length is not a multiple of int64 size"
        )));
    }
    Ok(bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| i64::from_ne_bytes(*chunk))
        .collect())
}

fn validate_sample(sample: &[f64], rows: usize, columns: usize) -> PyResult<()> {
    if rows.checked_mul(columns) != Some(sample.len()) {
        return Err(PyValueError::new_err(
            "sample buffer length does not match its shape",
        ));
    }
    Ok(())
}

fn parallel_sum<F>(rows: usize, workers: usize, function: F) -> PyResult<f64>
where
    F: Fn(usize) -> f64 + Sync,
{
    if workers == 0 {
        return Err(PyValueError::new_err("'workers' must be at least 1"));
    }
    if rows == 0 {
        return Ok(0.0);
    }
    let thread_count = workers.min(rows);
    let chunk_size = rows.div_ceil(thread_count);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(thread_count);
        for worker in 0..thread_count {
            let start = worker * chunk_size;
            let end = (start + chunk_size).min(rows);
            if start < end {
                let function = &function;
                handles.push(scope.spawn(move || (start..end).map(function).sum::<f64>()));
            }
        }
        handles.into_iter().try_fold(0.0, |total, handle| {
            handle
                .join()
                .map(|partial| total + partial)
                .map_err(|_| PyValueError::new_err("QMC worker thread panicked"))
        })
    })
}

fn centered_discrepancy(
    sample: &[f64],
    rows: usize,
    columns: usize,
    iterative: bool,
    workers: usize,
) -> PyResult<f64> {
    validate_sample(sample, rows, columns)?;
    let mut disc1 = 0.0;
    for i in 0..rows {
        let mut product = 1.0;
        for j in 0..columns {
            let value = sample[i * columns + j];
            let distance = (value - 0.5).abs();
            product *= 1.0 + 0.5 * distance - 0.5 * distance * distance;
        }
        disc1 += product;
    }
    let disc2 = parallel_sum(rows, workers, |i| {
        let mut total = 0.0;
        for j in 0..rows {
            let mut product = 1.0;
            for k in 0..columns {
                let a = sample[i * columns + k];
                let b = sample[j * columns + k];
                product *=
                    1.0 + 0.5 * (a - 0.5).abs() + 0.5 * (b - 0.5).abs() - 0.5 * (a - b).abs();
            }
            total += product;
        }
        total
    })?;
    let n = rows + usize::from(iterative);
    Ok(
        (13.0_f64 / 12.0).powf(columns as f64) - 2.0 / n as f64 * disc1
            + disc2 / (n as f64).powi(2),
    )
}

fn wrap_around_discrepancy(
    sample: &[f64],
    rows: usize,
    columns: usize,
    iterative: bool,
    workers: usize,
) -> PyResult<f64> {
    validate_sample(sample, rows, columns)?;
    let disc = parallel_sum(rows, workers, |i| {
        let mut total = 0.0;
        for j in 0..rows {
            let mut product = 1.0;
            for k in 0..columns {
                let distance = (sample[i * columns + k] - sample[j * columns + k]).abs();
                product *= 1.5 - distance + distance * distance;
            }
            total += product;
        }
        total
    })?;
    let n = rows + usize::from(iterative);
    Ok(-(4.0_f64 / 3.0).powf(columns as f64) + disc / (n as f64).powi(2))
}

fn mixture_discrepancy(
    sample: &[f64],
    rows: usize,
    columns: usize,
    iterative: bool,
    workers: usize,
) -> PyResult<f64> {
    validate_sample(sample, rows, columns)?;
    let mut disc1 = 0.0;
    for i in 0..rows {
        let mut product = 1.0;
        for j in 0..columns {
            let distance = (sample[i * columns + j] - 0.5).abs();
            product *= 5.0 / 3.0 - 0.25 * distance - 0.25 * distance * distance;
        }
        disc1 += product;
    }
    let disc2 = parallel_sum(rows, workers, |i| {
        let mut total = 0.0;
        for j in 0..rows {
            let mut product = 1.0;
            for k in 0..columns {
                let a = sample[i * columns + k];
                let b = sample[j * columns + k];
                let distance = (a - b).abs();
                product *=
                    15.0 / 8.0 - 0.25 * (a - 0.5).abs() - 0.25 * (b - 0.5).abs() - 0.75 * distance
                        + 0.5 * distance * distance;
            }
            total += product;
        }
        total
    })?;
    let n = (rows + usize::from(iterative)) as f64;
    Ok((19.0_f64 / 12.0).powf(columns as f64) - 2.0 / n * disc1 + disc2 / n.powi(2))
}

fn l2_star_discrepancy(
    sample: &[f64],
    rows: usize,
    columns: usize,
    iterative: bool,
    workers: usize,
) -> PyResult<f64> {
    validate_sample(sample, rows, columns)?;
    let mut disc1 = 0.0;
    for i in 0..rows {
        let mut product = 1.0;
        for j in 0..columns {
            let value = sample[i * columns + j];
            product *= 1.0 - value * value;
        }
        disc1 += product;
    }
    let disc2 = parallel_sum(rows, workers, |i| {
        let mut total = 0.0;
        for j in 0..rows {
            let mut product = 1.0;
            for k in 0..columns {
                product *= 1.0 - sample[i * columns + k].max(sample[j * columns + k]);
            }
            total += product;
        }
        total
    })?;
    let n = (rows + usize::from(iterative)) as f64;
    Ok(
        (3.0_f64.powf(-(columns as f64)) - 2.0_f64.powf(1.0 - columns as f64) * disc1 / n
            + disc2 / n.powi(2))
        .sqrt(),
    )
}

#[pyfunction]
#[pyo3(signature = (sample, rows, columns, iterative, workers))]
fn centered_discrepancy_py(
    sample: &Bound<'_, PyBytes>,
    rows: usize,
    columns: usize,
    iterative: bool,
    workers: usize,
) -> PyResult<f64> {
    centered_discrepancy(
        &decode_f64(sample.as_bytes(), "sample")?,
        rows,
        columns,
        iterative,
        workers,
    )
}

#[pyfunction]
#[pyo3(signature = (sample, rows, columns, iterative, workers))]
fn wrap_around_discrepancy_py(
    sample: &Bound<'_, PyBytes>,
    rows: usize,
    columns: usize,
    iterative: bool,
    workers: usize,
) -> PyResult<f64> {
    wrap_around_discrepancy(
        &decode_f64(sample.as_bytes(), "sample")?,
        rows,
        columns,
        iterative,
        workers,
    )
}

#[pyfunction]
#[pyo3(signature = (sample, rows, columns, iterative, workers))]
fn mixture_discrepancy_py(
    sample: &Bound<'_, PyBytes>,
    rows: usize,
    columns: usize,
    iterative: bool,
    workers: usize,
) -> PyResult<f64> {
    mixture_discrepancy(
        &decode_f64(sample.as_bytes(), "sample")?,
        rows,
        columns,
        iterative,
        workers,
    )
}

#[pyfunction]
#[pyo3(signature = (sample, rows, columns, iterative, workers))]
fn l2_star_discrepancy_py(
    sample: &Bound<'_, PyBytes>,
    rows: usize,
    columns: usize,
    iterative: bool,
    workers: usize,
) -> PyResult<f64> {
    l2_star_discrepancy(
        &decode_f64(sample.as_bytes(), "sample")?,
        rows,
        columns,
        iterative,
        workers,
    )
}

#[pyfunction]
fn update_discrepancy_py(
    x_new: &Bound<'_, PyBytes>,
    sample: &Bound<'_, PyBytes>,
    rows: usize,
    columns: usize,
    initial_disc: f64,
) -> PyResult<f64> {
    let x_new = decode_f64(x_new.as_bytes(), "x_new")?;
    let sample = decode_f64(sample.as_bytes(), "sample")?;
    validate_sample(&sample, rows, columns)?;
    if x_new.len() != columns {
        return Err(PyValueError::new_err(
            "new point length does not match sample dimension",
        ));
    }
    let n = (rows + 1) as f64;
    let mut absolute = Vec::with_capacity(columns);
    let mut product = 1.0;
    for value in &x_new {
        let distance = (value - 0.5).abs();
        absolute.push(distance);
        product *= 1.0 + 0.5 * distance - 0.5 * distance * distance;
    }
    let disc1 = -2.0 / n * product;
    let mut disc2 = 0.0;
    for i in 0..rows {
        product = 1.0;
        for j in 0..columns {
            product *= 1.0 + 0.5 * absolute[j] + 0.5 * (sample[i * columns + j] - 0.5).abs()
                - 0.5 * (x_new[j] - sample[i * columns + j]).abs();
        }
        disc2 += product;
    }
    disc2 *= 2.0 / n.powi(2);
    product = absolute.iter().map(|value| 1.0 + value).product();
    let disc3 = product / n.powi(2);
    Ok(initial_disc + disc1 + disc2 + disc3)
}

#[pyfunction]
fn van_der_corput(n: usize, base: i64, start_index: i64, workers: usize) -> PyResult<Vec<f64>> {
    if base < 2 {
        return Err(PyValueError::new_err("'base' must be at least 2"));
    }
    if workers == 0 {
        return Err(PyValueError::new_err("'workers' must be at least 1"));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let thread_count = workers.min(n);
    let chunk_size = n.div_ceil(thread_count);
    let mut result = vec![0.0; n];
    std::thread::scope(|scope| {
        for (worker, output) in result.chunks_mut(chunk_size).enumerate() {
            let start = worker * chunk_size;
            scope.spawn(move || {
                for (offset, value) in output.iter_mut().enumerate() {
                    let mut quotient = start_index + (start + offset) as i64;
                    let mut place = 1.0 / base as f64;
                    while quotient > 0 {
                        let remainder = quotient % base;
                        *value += remainder as f64 * place;
                        place /= base as f64;
                        quotient /= base;
                    }
                }
            });
        }
    });
    Ok(result)
}

#[pyfunction]
fn van_der_corput_scrambled(
    n: usize,
    base: i64,
    start_index: i64,
    permutations: &Bound<'_, PyBytes>,
    permutation_rows: usize,
    permutation_columns: usize,
    workers: usize,
) -> PyResult<Vec<f64>> {
    if base < 2 {
        return Err(PyValueError::new_err("'base' must be at least 2"));
    }
    if workers == 0 {
        return Err(PyValueError::new_err("'workers' must be at least 1"));
    }
    let permutations = decode_i64(permutations.as_bytes(), "permutations")?;
    if permutation_rows.checked_mul(permutation_columns) != Some(permutations.len())
        || permutation_columns != base as usize
    {
        return Err(PyValueError::new_err(
            "permutation shape does not match the sequence base",
        ));
    }
    if start_index < 0 {
        return Err(PyValueError::new_err(
            "'start_index' must be non-negative when scrambling",
        ));
    }
    if permutations
        .iter()
        .any(|value| *value < 0 || *value >= base)
    {
        return Err(PyValueError::new_err(
            "permutation entries must be valid base digits",
        ));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let chunk_size = n.div_ceil(workers.min(n));
    let mut result = vec![0.0; n];
    std::thread::scope(|scope| {
        for (worker, output) in result.chunks_mut(chunk_size).enumerate() {
            let start = worker * chunk_size;
            let permutations = &permutations;
            scope.spawn(move || {
                for (offset, value) in output.iter_mut().enumerate() {
                    let mut quotient = start_index + (start + offset) as i64;
                    let mut place = 1.0 / base as f64;
                    for row in 0..permutation_rows {
                        let remainder = (quotient % base) as usize;
                        *value +=
                            permutations[row * permutation_columns + remainder] as f64 * place;
                        place /= base as f64;
                        quotient /= base;
                    }
                }
            });
        }
    });
    Ok(result)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(centered_discrepancy_py, module)?)?;
    module.add_function(wrap_pyfunction!(wrap_around_discrepancy_py, module)?)?;
    module.add_function(wrap_pyfunction!(mixture_discrepancy_py, module)?)?;
    module.add_function(wrap_pyfunction!(l2_star_discrepancy_py, module)?)?;
    module.add_function(wrap_pyfunction!(update_discrepancy_py, module)?)?;
    module.add_function(wrap_pyfunction!(van_der_corput, module)?)?;
    module.add_function(wrap_pyfunction!(van_der_corput_scrambled, module)?)?;
    Ok(())
}
