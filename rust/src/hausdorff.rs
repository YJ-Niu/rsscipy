use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn decode_f64(bytes: &[u8]) -> PyResult<Vec<f64>> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<f64>()) {
        return Err(PyValueError::new_err(
            "input buffer length is not a multiple of float64 size",
        ));
    }
    Ok(bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| f64::from_ne_bytes(*chunk))
        .collect())
}

fn directed_hausdorff(a: &[f64], b: &[f64], dimensions: usize) -> (f64, usize, usize) {
    let n1 = a.len() / dimensions.max(1);
    let n2 = b.len() / dimensions.max(1);
    let mut cmax: f64 = 0.0;
    let mut i_ret = 0;
    let mut j_ret = 0;

    for i in 0..n1 {
        let mut cmin = f64::INFINITY;
        let mut j_store = 0;
        let mut d: f64 = 0.0;
        for j in 0..n2 {
            d = 0.0;
            for k in 0..dimensions {
                let delta = a[i * dimensions + k] - b[j * dimensions + k];
                d += delta * delta;
            }
            if d < cmax {
                break;
            }
            if d < cmin {
                cmin = d;
                j_store = j;
            }
        }
        if cmin >= cmax && d >= cmax {
            cmax = cmin;
            i_ret = i;
            j_ret = j_store;
        }
    }

    (cmax.sqrt(), i_ret, j_ret)
}

#[pyfunction]
fn directed_hausdorff_core(
    a: &Bound<'_, PyBytes>,
    b: &Bound<'_, PyBytes>,
    dimensions: usize,
) -> PyResult<(f64, usize, usize)> {
    let a = decode_f64(a.as_bytes())?;
    let b = decode_f64(b.as_bytes())?;
    if dimensions == 0 {
        if !a.is_empty() || !b.is_empty() {
            return Err(PyValueError::new_err(
                "non-empty point arrays require at least one dimension",
            ));
        }
        return Ok(directed_hausdorff(&a, &b, 0));
    }
    if !a.len().is_multiple_of(dimensions) || !b.len().is_multiple_of(dimensions) {
        return Err(PyValueError::new_err(
            "buffer size does not match the point dimensions",
        ));
    }
    Ok(directed_hausdorff(&a, &b, dimensions))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(directed_hausdorff_core, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::directed_hausdorff;

    #[test]
    fn computes_directed_distance_and_pair() {
        let a = [0.0, 0.0, 3.0, 0.0];
        let b = [0.0, 0.0, 1.0, 0.0];
        assert_eq!(directed_hausdorff(&a, &b, 2), (2.0, 1, 1));
    }

    #[test]
    fn early_break_preserves_previous_maximum() {
        let a = [5.0, 0.0, 0.0, 0.0];
        let b = [0.0, 0.0, 10.0, 0.0];
        assert_eq!(directed_hausdorff(&a, &b, 2), (5.0, 0, 0));
    }

    #[test]
    fn empty_source_has_zero_distance() {
        assert_eq!(directed_hausdorff(&[], &[0.0, 1.0], 1), (0.0, 0, 0));
    }
}
