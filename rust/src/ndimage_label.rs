use pyo3::exceptions::{PyAssertionError, PyValueError};
use pyo3::prelude::*;
use std::collections::VecDeque;

fn connected_components(
    foreground: &[bool],
    structure: &[bool],
    shape: &[usize],
) -> PyResult<(Vec<usize>, usize)> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(PyAssertionError::new_err(
            "Cannot label scalars or empty arrays",
        ));
    }
    let size = shape
        .iter()
        .try_fold(1usize, |size, &dimension| size.checked_mul(dimension))
        .ok_or_else(|| PyValueError::new_err("input size exceeds platform limits"))?;
    let structure_size = 3usize
        .checked_pow(shape.len() as u32)
        .ok_or_else(|| PyValueError::new_err("structure size exceeds platform limits"))?;
    if foreground.len() != size || structure.len() != structure_size {
        return Err(PyAssertionError::new_err(
            "input and structure shapes do not match",
        ));
    }

    let mut neighbor_offsets = Vec::new();
    for (flat_index, &active) in structure.iter().enumerate() {
        if !active {
            continue;
        }
        let mut remainder = flat_index;
        let mut offset = vec![0isize; shape.len()];
        let mut center = true;
        for axis in (0..shape.len()).rev() {
            let coordinate = remainder % 3;
            remainder /= 3;
            offset[axis] = coordinate as isize - 1;
            center &= offset[axis] == 0;
        }
        if !center {
            neighbor_offsets.push(offset);
        }
    }

    for (flat_index, &active) in structure.iter().enumerate() {
        let mut coordinates = vec![0usize; shape.len()];
        let mut remainder = flat_index;
        for axis in (0..shape.len()).rev() {
            coordinates[axis] = remainder % 3;
            remainder /= 3;
        }
        let mut mirrored_index = 0usize;
        for coordinate in coordinates {
            mirrored_index = mirrored_index * 3 + (2 - coordinate);
        }
        if active != structure[mirrored_index] {
            return Err(PyAssertionError::new_err(
                "Structuring element is not symmetric",
            ));
        }
    }

    let mut strides = vec![1usize; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1]
            .checked_mul(shape[axis + 1])
            .ok_or_else(|| PyValueError::new_err("input strides exceed platform limits"))?;
    }

    let mut labels = vec![0usize; size];
    let mut queue = VecDeque::new();
    let mut next_label = 0usize;
    for (start, &is_foreground) in foreground.iter().enumerate() {
        if !is_foreground || labels[start] != 0 {
            continue;
        }
        next_label += 1;
        labels[start] = next_label;
        queue.push_back(start);

        while let Some(index) = queue.pop_front() {
            for offset in &neighbor_offsets {
                let mut neighbor = index;
                let mut in_bounds = true;
                for axis in 0..shape.len() {
                    let coordinate = (index / strides[axis]) % shape[axis];
                    let candidate = coordinate as isize + offset[axis];
                    if candidate < 0 || candidate >= shape[axis] as isize {
                        in_bounds = false;
                        break;
                    }
                    neighbor = (neighbor as isize + offset[axis] * strides[axis] as isize) as usize;
                }
                if in_bounds && foreground[neighbor] && labels[neighbor] == 0 {
                    labels[neighbor] = next_label;
                    queue.push_back(neighbor);
                }
            }
        }
    }

    Ok((labels, next_label))
}

#[pyfunction]
fn _label(
    foreground: Vec<bool>,
    structure: Vec<bool>,
    shape: Vec<usize>,
) -> PyResult<(Vec<usize>, usize)> {
    connected_components(&foreground, &structure, &shape)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_label, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::connected_components;

    #[test]
    fn labels_components_using_axis_connectivity() {
        let input = [
            [true, false, false, false, false, false],
            [false, false, true, true, false, false],
            [false, false, true, true, true, false],
            [true, true, false, false, false, false],
            [true, true, false, false, false, false],
            [false, false, false, true, true, false],
        ]
        .concat();
        let structure = [false, true, false, true, true, true, false, true, false];
        let (labels, count) = connected_components(&input, &structure, &[6, 6]).unwrap();
        assert_eq!(count, 4);
        assert_eq!(
            labels,
            [
                1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 2, 2, 0, 3, 3, 0, 0, 0, 0, 3, 3, 0, 0,
                0, 0, 0, 0, 0, 4, 4, 0
            ]
        );
    }

    #[test]
    fn diagonal_structure_connects_diagonal_pixels() {
        let input = [true, false, false, true];
        let structure = [true; 9];
        let (labels, count) = connected_components(&input, &structure, &[2, 2]).unwrap();
        assert_eq!(count, 1);
        assert_eq!(labels, [1, 0, 0, 1]);
    }

    #[test]
    fn labels_diagonal_neighbors_in_three_dimensions() {
        let mut input = [false; 8];
        input[0] = true;
        input[7] = true;
        let (labels, count) = connected_components(&input, &[true; 27], &[2, 2, 2]).unwrap();
        assert_eq!(count, 1);
        assert_eq!(labels, [1, 0, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn rejects_asymmetric_structure_and_invalid_shapes() {
        assert!(connected_components(&[true], &[false, false, true], &[1]).is_err());
        assert!(connected_components(&[true], &[true; 9], &[1]).is_err());
    }
}
