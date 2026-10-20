use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

fn values(value: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    value
        .cast::<PyList>()?
        .iter()
        .map(|item| item.extract::<f64>())
        .collect()
}

fn labels(value: &Bound<'_, PyAny>) -> PyResult<Vec<i32>> {
    value
        .cast::<PyList>()?
        .iter()
        .map(|item| item.extract::<i32>())
        .collect()
}

fn typed(value: f64, single_precision: bool) -> f64 {
    if single_precision {
        value as f32 as f64
    } else {
        value
    }
}

fn vector_quantize(
    observations: &[f64],
    code_book: &[f64],
    nobs: usize,
    ncodes: usize,
    nfeat: usize,
    single_precision: bool,
) -> PyResult<(Vec<i32>, Vec<f64>)> {
    if nobs.checked_mul(nfeat) != Some(observations.len())
        || ncodes.checked_mul(nfeat) != Some(code_book.len())
    {
        return Err(PyValueError::new_err("array shape does not match data"));
    }
    if ncodes == 0 {
        return Err(PyValueError::new_err(
            "code book must contain at least one code",
        ));
    }
    let mut codes = vec![0; nobs];
    let mut distances = vec![f64::INFINITY; nobs];

    for observation in 0..nobs {
        let obs_start = observation * nfeat;
        for code in 0..ncodes {
            let code_start = code * nfeat;
            let mut squared_distance = 0.0;
            for feature in 0..nfeat {
                let difference = typed(
                    code_book[code_start + feature] - observations[obs_start + feature],
                    single_precision,
                );
                squared_distance = typed(
                    squared_distance + typed(difference * difference, single_precision),
                    single_precision,
                );
            }
            if squared_distance < distances[observation] {
                codes[observation] = code as i32;
                distances[observation] = squared_distance;
            }
        }
        distances[observation] = typed(distances[observation].max(0.0).sqrt(), single_precision);
    }
    Ok((codes, distances))
}

fn update_means(
    observations: &[f64],
    labels: &[i32],
    nobs: usize,
    ncodes: usize,
    nfeat: usize,
    single_precision: bool,
) -> PyResult<(Vec<f64>, Vec<bool>)> {
    if nobs.checked_mul(nfeat) != Some(observations.len()) || labels.len() != nobs {
        return Err(PyValueError::new_err("array shape does not match data"));
    }
    let mut code_book = vec![0.0; ncodes.saturating_mul(nfeat)];
    let mut counts = vec![0usize; ncodes];
    for (observation, &label) in labels.iter().enumerate() {
        if label < 0 || label as usize >= ncodes {
            return Err(PyValueError::new_err(
                "cluster label is outside the code book range",
            ));
        }
        let cluster = label as usize;
        counts[cluster] += 1;
        for feature in 0..nfeat {
            let index = cluster * nfeat + feature;
            code_book[index] = typed(
                code_book[index] + observations[observation * nfeat + feature],
                single_precision,
            );
        }
    }
    for (cluster, &count) in counts.iter().enumerate() {
        if count > 0 {
            let divisor = typed(count as f64, single_precision);
            for feature in 0..nfeat {
                let index = cluster * nfeat + feature;
                code_book[index] = typed(code_book[index] / divisor, single_precision);
            }
        }
    }
    let has_members = counts.iter().map(|&count| count > 0).collect();
    Ok((code_book, has_members))
}

#[pyfunction]
#[pyo3(signature = (observations, code_book, nobs, ncodes, nfeat, single_precision))]
fn _vq(
    observations: &Bound<'_, PyAny>,
    code_book: &Bound<'_, PyAny>,
    nobs: usize,
    ncodes: usize,
    nfeat: usize,
    single_precision: bool,
) -> PyResult<(Vec<i32>, Vec<f64>)> {
    vector_quantize(
        &values(observations)?,
        &values(code_book)?,
        nobs,
        ncodes,
        nfeat,
        single_precision,
    )
}

#[pyfunction]
#[pyo3(signature = (observations, cluster_labels, ncodes, nobs, nfeat, single_precision))]
fn _update_cluster_means(
    observations: &Bound<'_, PyAny>,
    cluster_labels: &Bound<'_, PyAny>,
    ncodes: usize,
    nobs: usize,
    nfeat: usize,
    single_precision: bool,
) -> PyResult<(Vec<f64>, Vec<bool>)> {
    update_means(
        &values(observations)?,
        &labels(cluster_labels)?,
        nobs,
        ncodes,
        nfeat,
        single_precision,
    )
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_vq, module)?)?;
    module.add_function(wrap_pyfunction!(_update_cluster_means, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{update_means, vector_quantize};

    #[test]
    fn assigns_nearest_codes_for_flat_and_vector_observations() {
        let (codes, distances) =
            vector_quantize(&[0.0, 0.0, 4.0, 4.0], &[0.0, 0.0, 5.0, 5.0], 2, 2, 2, false).unwrap();
        assert_eq!(codes, [0, 1]);
        assert_eq!(distances, [0.0, 2.0_f64.sqrt()]);

        let (codes, distances) =
            vector_quantize(&[1.0, 4.0, 8.0], &[0.0, 5.0], 3, 2, 1, false).unwrap();
        assert_eq!(codes, [0, 1, 1]);
        assert_eq!(distances, [1.0, 1.0, 3.0]);
    }

    #[test]
    fn updates_means_and_marks_empty_clusters() {
        let (means, has_members) =
            update_means(&[1.0, 2.0, 3.0, 4.0, 9.0, 10.0], &[0, 0, 2], 3, 3, 2, false).unwrap();
        assert_eq!(means, [2.0, 3.0, 0.0, 0.0, 9.0, 10.0]);
        assert_eq!(has_members, [true, false, true]);
    }

    #[test]
    fn rounds_single_precision_accumulation_and_distances() {
        let (codes, distances) = vector_quantize(&[0.0, 3.0], &[1.0, 4.0], 2, 2, 1, true).unwrap();
        assert_eq!(codes, [0, 1]);
        assert_eq!(distances, [1.0, 1.0]);
    }
}
