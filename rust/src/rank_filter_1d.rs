use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn sample_index(index: isize, length: usize, mode: usize) -> Option<usize> {
    if (0..length as isize).contains(&index) {
        return Some(index as usize);
    }
    if length == 0 {
        return None;
    }
    match mode {
        0 => Some(index.clamp(0, length as isize - 1) as usize),
        1 => Some(index.rem_euclid(length as isize) as usize),
        2 => {
            let period = 2 * length as isize;
            let reflected = index.rem_euclid(period);
            Some(if reflected < length as isize {
                reflected as usize
            } else {
                (period - reflected - 1) as usize
            })
        }
        3 => {
            if length == 1 {
                return Some(0);
            }
            let period = 2 * (length as isize - 1);
            let reflected = index.rem_euclid(period);
            Some(if reflected < length as isize {
                reflected as usize
            } else {
                (period - reflected) as usize
            })
        }
        4 => None,
        _ => None,
    }
}

fn rank_filter<T: Copy>(
    input: &[T],
    rank: usize,
    window: usize,
    mode: usize,
    origin: isize,
    cval: T,
    mut compare: impl FnMut(&T, &T) -> std::cmp::Ordering,
) -> PyResult<Vec<T>> {
    if window == 0 || rank >= window || input.is_empty() || mode > 4 {
        return Err(PyValueError::new_err("invalid rank filter parameters"));
    }
    let left = ((window - 1) / 2) as isize + origin + isize::from(window.is_multiple_of(2));
    let mut output = Vec::with_capacity(input.len());
    for position in 0..input.len() {
        let mut values = Vec::with_capacity(window);
        for offset in 0..window {
            let source = position as isize + offset as isize - left;
            values.push(sample_index(source, input.len(), mode).map_or(cval, |i| input[i]));
        }
        values.select_nth_unstable_by(rank, &mut compare);
        output.push(values[rank]);
    }
    Ok(output)
}

#[pyfunction]
#[pyo3(signature = (input, rank, window, mode, cval, origin))]
fn _rank_filter_float(
    input: Vec<f64>,
    rank: usize,
    window: usize,
    mode: usize,
    cval: f64,
    origin: isize,
) -> PyResult<Vec<f64>> {
    rank_filter(&input, rank, window, mode, origin, cval, f64::total_cmp)
}

#[pyfunction]
#[pyo3(signature = (input, rank, window, mode, cval, origin))]
fn _rank_filter_int(
    input: Vec<i64>,
    rank: usize,
    window: usize,
    mode: usize,
    cval: i64,
    origin: isize,
) -> PyResult<Vec<i64>> {
    rank_filter(&input, rank, window, mode, origin, cval, i64::cmp)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_rank_filter_float, module)?)?;
    module.add_function(wrap_pyfunction!(_rank_filter_int, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::rank_filter;

    #[test]
    fn filters_rank_with_each_boundary_rule() {
        let input = [3, 2, 5, 1, 4];
        let reflect = rank_filter(&input, 1, 3, 2, 0, 0, i64::cmp).unwrap();
        assert_eq!(reflect, [3, 3, 2, 4, 4]);
        let nearest = rank_filter(&input, 1, 3, 0, 0, 0, i64::cmp).unwrap();
        assert_eq!(nearest, [3, 3, 2, 4, 4]);
        let constant = rank_filter(&input, 1, 3, 4, 0, 0, i64::cmp).unwrap();
        assert_eq!(constant, [2, 3, 2, 4, 1]);
        let wrap = rank_filter(&input, 1, 3, 1, 0, 0, i64::cmp).unwrap();
        assert_eq!(wrap, [3, 3, 2, 4, 3]);
        let mirror = rank_filter(&input, 1, 3, 3, 0, 0, i64::cmp).unwrap();
        assert_eq!(mirror, [2, 3, 2, 4, 1]);
    }

    #[test]
    fn supports_even_windows_origins_and_float_values() {
        let input = [3.0, 2.0, 5.0, 1.0, 4.0];
        let result = rank_filter(&input, 1, 2, 2, 0, 0.0, f64::total_cmp).unwrap();
        assert_eq!(result, [3.0, 3.0, 5.0, 5.0, 4.0]);
        let shifted = rank_filter(&input, 0, 3, 4, 0, 7.0, f64::total_cmp).unwrap();
        assert_eq!(shifted, [2.0, 2.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn handles_single_element_lines_and_rejects_bad_parameters() {
        assert_eq!(rank_filter(&[7], 1, 3, 2, 0, 0, i64::cmp).unwrap(), [7]);
        assert!(rank_filter(&[1, 2], 2, 2, 0, 0, 0, i64::cmp).is_err());
    }
}
