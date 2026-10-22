use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn group_dense(matrix: &[Vec<bool>], nrows: usize, ncols: usize) -> PyResult<Vec<usize>> {
    if matrix.len() != nrows || matrix.iter().any(|row| row.len() != ncols) {
        return Err(PyValueError::new_err(
            "matrix shape does not match dimensions",
        ));
    }

    let mut groups = vec![usize::MAX; ncols];
    let mut current_group = 0;
    let mut union = vec![false; nrows];

    for column in 0..ncols {
        if groups[column] != usize::MAX {
            continue;
        }
        groups[column] = current_group;
        for row in 0..nrows {
            union[row] = matrix[row][column];
        }

        for candidate in column + 1..ncols {
            if groups[candidate] != usize::MAX {
                continue;
            }
            let intersects = (0..nrows).any(|row| union[row] && matrix[row][candidate]);
            if !intersects {
                for row in 0..nrows {
                    union[row] |= matrix[row][candidate];
                }
                groups[candidate] = current_group;
            }
        }
        current_group += 1;
    }

    Ok(groups)
}

fn group_sparse(
    indices: &[usize],
    indptr: &[usize],
    nrows: usize,
    ncols: usize,
) -> PyResult<Vec<usize>> {
    if indptr.len() != ncols.saturating_add(1)
        || indptr.first() != Some(&0)
        || indptr.last() != Some(&indices.len())
        || indptr.windows(2).any(|pair| pair[0] > pair[1])
        || indices.iter().any(|&index| index >= nrows)
    {
        return Err(PyValueError::new_err(
            "invalid compressed sparse matrix structure",
        ));
    }

    let mut groups = vec![usize::MAX; ncols];
    let mut current_group = 0;
    let mut union = vec![false; nrows];

    for column in 0..ncols {
        if groups[column] != usize::MAX {
            continue;
        }
        groups[column] = current_group;
        union.fill(false);
        for &row in &indices[indptr[column]..indptr[column + 1]] {
            union[row] = true;
        }

        for candidate in column + 1..ncols {
            if groups[candidate] != usize::MAX {
                continue;
            }
            let candidate_rows = &indices[indptr[candidate]..indptr[candidate + 1]];
            if !candidate_rows.iter().any(|&row| union[row]) {
                for &row in candidate_rows {
                    union[row] = true;
                }
                groups[candidate] = current_group;
            }
        }
        current_group += 1;
    }

    Ok(groups)
}

#[pyfunction]
#[pyo3(signature = (matrix, nrows, ncols))]
fn _group_dense(matrix: Vec<Vec<bool>>, nrows: usize, ncols: usize) -> PyResult<Vec<usize>> {
    group_dense(&matrix, nrows, ncols)
}

#[pyfunction]
#[pyo3(signature = (indices, indptr, nrows, ncols))]
fn _group_sparse(
    indices: Vec<usize>,
    indptr: Vec<usize>,
    nrows: usize,
    ncols: usize,
) -> PyResult<Vec<usize>> {
    group_sparse(&indices, &indptr, nrows, ncols)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_group_dense, module)?)?;
    module.add_function(wrap_pyfunction!(_group_sparse, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{group_dense, group_sparse};

    #[test]
    fn groups_dense_columns_by_nonzero_pattern() {
        let matrix = [
            [true, true, false, false, false, false],
            [true, true, true, false, false, false],
            [false, true, true, true, false, false],
            [false, false, true, true, true, false],
            [false, false, false, true, true, true],
            [false, false, false, false, true, true],
            [false, false, false, false, false, false],
        ]
        .iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
        assert_eq!(group_dense(&matrix, 7, 6).unwrap(), [0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn groups_sparse_columns_and_rejects_invalid_indices() {
        let indices = [0, 1, 1, 2];
        let indptr = [0, 1, 3, 4];
        assert_eq!(group_sparse(&indices, &indptr, 3, 3).unwrap(), [0, 0, 0]);
        assert!(group_sparse(&[3], &[0, 1], 3, 1).is_err());
    }
}
