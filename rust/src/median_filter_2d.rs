//! Two-dimensional median filter with zero-padded boundaries.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::cmp::Ordering;

fn numbers_from_python(value: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    value
        .cast::<PyList>()?
        .iter()
        .map(|item| item.extract::<f64>())
        .collect()
}

fn median(values: &mut [f64]) -> f64 {
    let middle = (values.len() - 1) / 2;
    let (_, value, _) = values.select_nth_unstable_by(middle, |left, right| {
        left.partial_cmp(right).unwrap_or(Ordering::Equal)
    });
    *value
}

fn filter_2d(
    input: &[f64],
    rows: usize,
    cols: usize,
    window_rows: usize,
    window_cols: usize,
) -> PyResult<Vec<f64>> {
    if rows.checked_mul(cols) != Some(input.len()) {
        return Err(PyValueError::new_err("input shape does not match data"));
    }
    if window_rows == 0 || window_cols == 0 {
        return Err(PyValueError::new_err("window dimensions must be positive"));
    }
    let window_len = window_rows
        .checked_mul(window_cols)
        .ok_or_else(|| PyValueError::new_err("kernel size is too large"))?;
    let before_row = window_rows / 2;
    let before_col = window_cols / 2;
    let mut neighborhood = vec![0.0; window_len];
    let mut output = vec![0.0; input.len()];

    for row in 0..rows {
        for col in 0..cols {
            let mut count = 0;
            for window_row in 0..window_rows {
                let source_row = row as isize + window_row as isize - before_row as isize;
                for window_col in 0..window_cols {
                    let source_col = col as isize + window_col as isize - before_col as isize;
                    if source_row >= 0
                        && source_row < rows as isize
                        && source_col >= 0
                        && source_col < cols as isize
                    {
                        neighborhood[count] =
                            input[source_row as usize * cols + source_col as usize];
                    } else {
                        neighborhood[count] = 0.0;
                    }
                    count += 1;
                }
            }
            output[row * cols + col] = median(&mut neighborhood);
        }
    }
    Ok(output)
}

#[pyfunction]
fn _medfilt2d<'py>(
    image: &Bound<'py, PyAny>,
    shape: Vec<usize>,
    size: &Bound<'py, PyAny>,
) -> PyResult<Vec<f64>> {
    if shape.len() != 2 {
        return Err(PyValueError::new_err("input must be two-dimensional"));
    }
    let image = numbers_from_python(image)?;
    let size = numbers_from_python(size)?;
    if size.len() < 2 {
        return Err(PyValueError::new_err("Size must be a length two sequence"));
    }
    if size[0] < 1.0
        || size[1] < 1.0
        || size[0].fract() != 0.0
        || size[1].fract() != 0.0
        || size[0] > usize::MAX as f64
        || size[1] > usize::MAX as f64
    {
        return Err(PyValueError::new_err(
            "kernel size must contain positive integers",
        ));
    }
    filter_2d(
        &image,
        shape[0],
        shape[1],
        size[0] as usize,
        size[1] as usize,
    )
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_medfilt2d, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::filter_2d;

    #[test]
    fn applies_zero_padded_rectangular_window() {
        let input = (0..25).map(f64::from).collect::<Vec<_>>();
        let output = filter_2d(&input, 5, 5, 3, 3).unwrap();
        assert_eq!(
            output,
            vec![
                0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 6.0, 7.0, 8.0, 4.0, 6.0, 11.0, 12.0, 13.0, 9.0, 11.0,
                16.0, 17.0, 18.0, 14.0, 0.0, 16.0, 17.0, 18.0, 0.0
            ]
        );
    }

    #[test]
    fn oversized_window_is_zero_padded_to_full_size() {
        let output = filter_2d(&[1.0, 2.0, 3.0, 4.0], 2, 2, 5, 3).unwrap();
        assert_eq!(output.len(), 4);
        assert_eq!(output[0], 0.0);
    }
}
