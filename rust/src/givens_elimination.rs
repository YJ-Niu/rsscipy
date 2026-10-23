use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

fn vector(value: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    value
        .cast::<PyList>()?
        .iter()
        .map(|item| item.extract::<f64>())
        .collect()
}

fn eliminate(
    mut matrix: Vec<f64>,
    mut vector: Vec<f64>,
    diagonal: &[f64],
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut diagonal_row = vec![0.0; n];
    for i in 0..n {
        if diagonal[i] == 0.0 {
            continue;
        }
        diagonal_row[i + 1..].fill(0.0);
        diagonal_row[i] = diagonal[i];
        let mut vector_tail = 0.0;

        for j in i..n {
            if diagonal_row[j] == 0.0 {
                continue;
            }
            let f = matrix[j * n + j];
            let g = diagonal_row[j];
            let (cosine, sine, radius) = givens_rotation(f, g);
            matrix[j * n + j] = radius;

            for column in j + 1..n {
                let upper = matrix[j * n + column];
                let lower = diagonal_row[column];
                matrix[j * n + column] = cosine * upper + sine * lower;
                diagonal_row[column] = -sine * upper + cosine * lower;
            }

            let first = vector[j];
            vector[j] = cosine * first + sine * vector_tail;
            vector_tail = -sine * first + cosine * vector_tail;
        }
    }
    (matrix, vector)
}

fn givens_rotation(f: f64, g: f64) -> (f64, f64, f64) {
    if g == 0.0 {
        return (if f < 0.0 { -1.0 } else { 1.0 }, 0.0, f);
    }
    if f == 0.0 {
        return (0.0, g.signum(), g.abs());
    }
    let radius = f.hypot(g).copysign(f);
    (f / radius, g / radius, radius)
}

#[pyfunction]
fn _givens_elimination(
    matrix: &Bound<'_, PyAny>,
    vector_arg: &Bound<'_, PyAny>,
    diagonal_arg: &Bound<'_, PyAny>,
    n: usize,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let matrix = vector(matrix)?;
    let vector_values = vector(vector_arg)?;
    let diagonal = vector(diagonal_arg)?;
    if matrix.len() != n.saturating_mul(n) || vector_values.len() != n || diagonal.len() != n {
        return Err(PyValueError::new_err("array dimensions do not match"));
    }
    Ok(eliminate(matrix, vector_values, &diagonal, n))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_givens_elimination, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{eliminate, givens_rotation};

    #[test]
    fn rotation_eliminates_diagonal_block_and_rotates_vector() {
        let (result, rotated_vector) = eliminate(vec![3.0], vec![5.0], &[4.0], 1);
        assert!((result[0] - 5.0).abs() < 1e-14);
        assert!((rotated_vector[0] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn handles_zero_and_signed_rotation_inputs() {
        assert_eq!(givens_rotation(0.0, 2.0), (0.0, 1.0, 2.0));
        let (cosine, sine, radius) = givens_rotation(-3.0, 4.0);
        assert_eq!(radius, -5.0);
        assert_eq!(cosine, 0.6);
        assert_eq!(sine, -0.8);
    }
}
