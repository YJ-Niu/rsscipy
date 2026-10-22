use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn validate_csr(data: &[f64], indices: &[usize], indptr: &[usize], n: usize) -> PyResult<()> {
    if data.len() != indices.len()
        || indptr.len() != n.saturating_add(1)
        || indptr.first() != Some(&0)
        || indptr.last() != Some(&data.len())
        || indptr.windows(2).any(|pair| pair[0] > pair[1])
        || indices.iter().any(|&index| index >= n)
    {
        return Err(PyValueError::new_err(
            "invalid compressed sparse graph structure",
        ));
    }
    Ok(())
}

#[pyfunction]
fn _dense_from_csr(
    data: Vec<f64>,
    indices: Vec<usize>,
    indptr: Vec<usize>,
    n: usize,
    null_value: f64,
) -> PyResult<Vec<f64>> {
    dense_from_csr(&data, &indices, &indptr, n, null_value)
}

fn dense_from_csr(
    data: &[f64],
    indices: &[usize],
    indptr: &[usize],
    n: usize,
    null_value: f64,
) -> PyResult<Vec<f64>> {
    validate_csr(data, indices, indptr, n)?;
    let mut graph = vec![f64::INFINITY; n.saturating_mul(n)];
    for row in 0..n {
        for position in indptr[row]..indptr[row + 1] {
            let offset = row * n + indices[position];
            if data[position] < graph[offset] {
                graph[offset] = data[position];
            }
        }
    }
    for value in &mut graph {
        if value.is_infinite() && value.is_sign_positive() {
            *value = null_value;
        }
    }
    Ok(graph)
}

#[pyfunction]
fn _construct_dist_matrix(
    graph: Vec<f64>,
    predecessors: Vec<i32>,
    n: usize,
    directed: bool,
    null_value: f64,
) -> PyResult<Vec<f64>> {
    construct_dist_matrix(&graph, &predecessors, n, directed, null_value)
}

fn construct_dist_matrix(
    graph: &[f64],
    predecessors: &[i32],
    n: usize,
    directed: bool,
    null_value: f64,
) -> PyResult<Vec<f64>> {
    let size = n
        .checked_mul(n)
        .ok_or_else(|| PyValueError::new_err("graph shape is too large"))?;
    if graph.len() != size || predecessors.len() != size {
        return Err(PyValueError::new_err(
            "graph and predecessors must have the same shape",
        ));
    }

    let mut graph = graph.to_vec();
    if !directed {
        for value in &mut graph {
            if *value == 0.0 {
                *value = f64::INFINITY;
            }
        }
        for row in 0..n {
            for column in row + 1..n {
                let upper = row * n + column;
                let lower = column * n + row;
                if graph[lower] <= graph[upper] {
                    graph[upper] = graph[lower];
                } else {
                    graph[lower] = graph[upper];
                }
            }
        }
    }

    let mut distances = vec![0.0; size];
    for source in 0..n {
        for target in 0..n {
            if source == target {
                continue;
            }
            let mut current = target;
            let mut has_path = false;
            let mut steps = 0;
            while current != source {
                let parent = predecessors[source * n + current];
                if parent == -9999 {
                    break;
                }
                if parent < 0 || parent as usize >= n {
                    return Err(PyValueError::new_err(
                        "predecessors contain an invalid vertex index",
                    ));
                }
                let parent = parent as usize;
                distances[source * n + target] += graph[parent * n + current];
                has_path = true;
                current = parent;
                steps += 1;
                if steps > n {
                    return Err(PyValueError::new_err("predecessors contain a cycle"));
                }
            }
            if !has_path {
                distances[source * n + target] = null_value;
            }
        }
    }
    Ok(distances)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_dense_from_csr, module)?)?;
    module.add_function(wrap_pyfunction!(_construct_dist_matrix, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{construct_dist_matrix, dense_from_csr};

    #[test]
    fn converts_duplicate_edges_using_minimum_weight() {
        let graph = dense_from_csr(&[2.0, 3.0, 0.0], &[1, 1, 0], &[0, 2, 3], 2, 0.0).unwrap();
        assert_eq!(graph, [0.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn constructs_distances_and_rejects_cycles() {
        let graph = [0.0, 1.0, 2.0, 0.0];
        let predecessors = [-9999, 0, -9999, -9999];
        let distances =
            construct_dist_matrix(&graph, &predecessors, 2, true, f64::INFINITY).unwrap();
        assert_eq!(distances, [0.0, 1.0, f64::INFINITY, 0.0]);

        let cycle = [-9999, 1, 0, -9999];
        assert!(construct_dist_matrix(&graph, &cycle, 2, true, f64::INFINITY).is_err());
    }
}
