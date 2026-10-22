use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};

fn checked_total(values: &[i64]) -> PyResult<i64> {
    values.iter().try_fold(0i64, |total, &value| {
        if value < 0 {
            return Err(PyValueError::new_err("marginals must be non-negative"));
        }
        total
            .checked_add(value)
            .ok_or_else(|| PyValueError::new_err("marginal total exceeds int64 range"))
    })
}

fn checked_dimensions(
    row: &[i64],
    col: &[i64],
    size: usize,
) -> PyResult<(usize, usize, usize, usize)> {
    let row_total = checked_total(row)?;
    let col_total = checked_total(col)?;
    if row_total != col_total {
        return Err(PyValueError::new_err("row and column totals must be equal"));
    }
    let cells = row
        .len()
        .checked_mul(col.len())
        .and_then(|n| n.checked_mul(size))
        .ok_or_else(|| PyValueError::new_err("requested output is too large"))?;
    let total = usize::try_from(row_total)
        .map_err(|_| PyValueError::new_err("marginal total does not fit this platform"))?;
    Ok((row.len(), col.len(), cells, total))
}

fn zeroed_output(cells: usize) -> PyResult<Vec<i64>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(cells)
        .map_err(|_| PyValueError::new_err("requested output is too large"))?;
    result.resize(cells, 0);
    Ok(result)
}

fn decode_uniforms(bytes: &[u8]) -> PyResult<Vec<f64>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(PyValueError::new_err(
            "random buffer length is not a multiple of float64 size",
        ));
    }
    Ok(bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| f64::from_ne_bytes(*chunk))
        .collect())
}

fn uniform(rng: &Bound<'_, PyAny>) -> PyResult<f64> {
    let value = rng.call_method0("random")?;
    let value = match value.extract::<f64>() {
        Ok(value) => value,
        Err(_) => value.call_method0("item")?.extract::<f64>()?,
    };
    if !value.is_finite() || !(0.0..1.0).contains(&value) {
        return Err(PyValueError::new_err(
            "random_state.random() must return a value in [0, 1)",
        ));
    }
    Ok(value)
}

fn sample_boyett(row: &[i64], col: &[i64], size: usize, uniforms: &[f64]) -> PyResult<Vec<i64>> {
    let (nr, nc, cells, total) = checked_dimensions(row, col, size)?;
    let mut result = zeroed_output(cells)?;
    if nr == 0 || nc == 0 || total == 0 || size == 0 {
        return Ok(result);
    }

    let mut work = Vec::new();
    work.try_reserve_exact(total)
        .map_err(|_| PyValueError::new_err("marginal total is too large"))?;
    for (j, &count) in col.iter().enumerate() {
        let count = usize::try_from(count)
            .map_err(|_| PyValueError::new_err("marginal does not fit this platform"))?;
        work.extend(std::iter::repeat_n(j, count));
    }
    let draws_per_sample = total.saturating_sub(1);
    let required_draws = draws_per_sample
        .checked_mul(size)
        .ok_or_else(|| PyValueError::new_err("requested random stream is too large"))?;
    if uniforms.len() != required_draws
        || uniforms
            .iter()
            .any(|&value| !value.is_finite() || !(0.0..1.0).contains(&value))
    {
        return Err(PyValueError::new_err("invalid random stream"));
    }

    for sample in 0..size {
        for i in (1..total).rev() {
            let draw = sample * draws_per_sample + (total - 1 - i);
            let j = (uniforms[draw] * (i + 1) as f64) as usize;
            work.swap(i, j.min(i));
        }
        let base = sample * nr * nc;
        let mut point = 0;
        for (i, &count) in row.iter().enumerate() {
            for &j in &work[point..point + count as usize] {
                result[base + i * nc + j] += 1;
            }
            point += count as usize;
        }
    }
    Ok(result)
}

fn log_factorial(n: i64) -> f64 {
    const COEFFICIENTS: [f64; 8] = [
        676.5203681218851,
        -1259.1392167224028,
        771.3234287776531,
        -176.6150291621406,
        12.507343278686905,
        -0.13857109526572012,
        9.984369578019572e-6,
        1.5056327351493116e-7,
    ];
    if n <= 1 {
        return 0.0;
    }
    let x0 = n as f64;
    let mut sum = 0.9999999999998099;
    for (i, coefficient) in COEFFICIENTS.iter().enumerate() {
        sum += coefficient / (x0 + i as f64 + 1.0);
    }
    let t = x0 + 7.5;
    0.9189385332046727 + (x0 + 0.5) * t.ln() - t + sum.ln()
}

fn hypergeometric(
    ia: i64,
    ib: i64,
    ic: i64,
    id: i64,
    ie: i64,
    rng: &Bound<'_, PyAny>,
) -> PyResult<i64> {
    let ii = ib - id;
    if ie <= 0 {
        return Ok(0);
    }
    let lower_bound = 0.max(ia - (ie - id));
    let upper_bound = ia.min(id);
    let mode = ((ia as f64 * id as f64 / ie as f64) + 0.5)
        .floor()
        .clamp(lower_bound as f64, upper_bound as f64) as i64;
    let mode_probability =
        (log_factorial(ia) + log_factorial(ib) + log_factorial(ic) + log_factorial(id)
            - log_factorial(ie)
            - log_factorial(mode)
            - log_factorial(id - mode)
            - log_factorial(ia - mode)
            - log_factorial(ii + mode))
        .exp();

    loop {
        let target = uniform(rng)?;
        let mut cumulative = mode_probability;
        if target <= cumulative {
            return Ok(mode);
        }

        let mut lower = mode;
        let mut lower_probability = mode_probability;
        let mut upper = mode;
        let mut upper_probability = mode_probability;
        while lower > lower_bound || upper < upper_bound {
            if upper < upper_bound {
                upper_probability *= ((id - upper) as f64 * (ia - upper) as f64)
                    / ((upper + 1) as f64 * (ii + upper + 1) as f64);
                upper += 1;
                cumulative += upper_probability;
                if target <= cumulative {
                    return Ok(upper);
                }
            }
            if lower > lower_bound {
                lower_probability *= (lower as f64 * (ii + lower) as f64)
                    / ((id - lower + 1) as f64 * (ia - lower + 1) as f64);
                lower -= 1;
                cumulative += lower_probability;
                if target <= cumulative {
                    return Ok(lower);
                }
            }
        }
    }
}

fn sample_patefield(
    row: &[i64],
    col: &[i64],
    size: usize,
    rng: &Bound<'_, PyAny>,
) -> PyResult<Vec<i64>> {
    let (nr, nc, cells, total) = checked_dimensions(row, col, size)?;
    let total = i64::try_from(total)
        .map_err(|_| PyValueError::new_err("marginal total exceeds int64 range"))?;
    let mut result = zeroed_output(cells)?;
    if nr == 0 || nc == 0 || total == 0 || size == 0 {
        return Ok(result);
    }

    for sample in 0..size {
        let base = sample * nr * nc;
        if nr == 1 {
            result[base..base + nc].copy_from_slice(col);
            continue;
        }
        if nc == 1 {
            for (i, &count) in row.iter().enumerate() {
                result[base + i] = count;
            }
            continue;
        }

        let mut remaining_cols = col.to_vec();
        let mut remaining_total = total;
        for i in 0..nr - 1 {
            let mut row_remaining = row[i];
            if row_remaining == 0 {
                remaining_total -= row[i];
                continue;
            }
            let mut row_column_total = remaining_total;
            remaining_total -= row[i];
            for j in 0..nc - 1 {
                let col_remaining = remaining_cols[j];
                let cell_total = row_column_total;
                row_column_total -= col_remaining;
                if col_remaining == 0 {
                    continue;
                }
                let other_rows = cell_total - row_remaining;
                let other_columns = cell_total - col_remaining;
                let cell = if cell_total == 0 {
                    0
                } else {
                    hypergeometric(
                        row_remaining,
                        other_rows,
                        other_columns,
                        col_remaining,
                        cell_total,
                        rng,
                    )?
                };
                result[base + i * nc + j] = cell;
                row_remaining -= cell;
                remaining_cols[j] -= cell;
            }
            result[base + i * nc + nc - 1] = row_remaining;
        }
        for j in 0..nc - 1 {
            result[base + (nr - 1) * nc + j] = remaining_cols[j];
        }
        result[base + nr * nc - 1] = row[nr - 1] - remaining_cols[..nc - 1].iter().sum::<i64>();
    }
    Ok(result)
}

#[pyfunction]
fn rvs_rcont1(
    row: Vec<i64>,
    col: Vec<i64>,
    ntot: i64,
    size: usize,
    uniforms: &Bound<'_, PyBytes>,
) -> PyResult<Vec<i64>> {
    if checked_total(&row)? != ntot {
        return Err(PyValueError::new_err("ntot does not match row totals"));
    }
    let uniforms = decode_uniforms(uniforms.as_bytes())?;
    sample_boyett(&row, &col, size, &uniforms)
}

#[pyfunction]
fn rvs_rcont2(
    row: Vec<i64>,
    col: Vec<i64>,
    ntot: i64,
    size: usize,
    random_state: &Bound<'_, PyAny>,
) -> PyResult<Vec<i64>> {
    if checked_total(&row)? != ntot {
        return Err(PyValueError::new_err("ntot does not match row totals"));
    }
    sample_patefield(&row, &col, size, random_state)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(rvs_rcont1, module)?)?;
    module.add_function(wrap_pyfunction!(rvs_rcont2, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::log_factorial;

    #[test]
    fn log_factorial_matches_small_exact_values() {
        assert_eq!(log_factorial(0), 0.0);
        assert_eq!(log_factorial(1), 0.0);
        assert!((log_factorial(5) - 120.0f64.ln()).abs() < 1e-12);
    }
}
