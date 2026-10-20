use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn reverse_cuthill_mckee(
    indices: &[usize],
    indptr: &[usize],
    nrows: usize,
) -> PyResult<Vec<usize>> {
    if indptr.len() != nrows.saturating_add(1)
        || indptr.first() != Some(&0)
        || indptr.last() != Some(&indices.len())
        || indptr.windows(2).any(|pair| pair[0] > pair[1])
        || indices.iter().any(|&index| index >= nrows)
    {
        return Err(PyValueError::new_err(
            "invalid compressed sparse graph structure",
        ));
    }

    let mut degree = vec![0usize; nrows];
    for row in 0..nrows {
        degree[row] = indptr[row + 1] - indptr[row];
        if indices[indptr[row]..indptr[row + 1]].contains(&row) {
            degree[row] += 1;
        }
    }

    let mut seeds: Vec<usize> = (0..nrows).collect();
    seeds.sort_by_key(|&node| degree[node]);
    let mut visited = vec![false; nrows];
    let mut order = Vec::with_capacity(nrows);

    for seed in seeds {
        if visited[seed] {
            continue;
        }
        visited[seed] = true;
        order.push(seed);
        let mut level_start = order.len() - 1;
        let mut level_end = order.len();

        while level_start < level_end {
            for position in level_start..level_end {
                let node = order[position];
                let mut neighbors = Vec::new();
                for &neighbor in &indices[indptr[node]..indptr[node + 1]] {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        neighbors.push(neighbor);
                    }
                }
                neighbors.sort_by_key(|&neighbor| degree[neighbor]);
                order.extend(neighbors);
            }
            level_start = level_end;
            level_end = order.len();
        }
    }

    order.reverse();
    Ok(order)
}

#[pyfunction]
fn _reverse_cuthill_mckee(
    indices: Vec<usize>,
    indptr: Vec<usize>,
    nrows: usize,
) -> PyResult<Vec<usize>> {
    reverse_cuthill_mckee(&indices, &indptr, nrows)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_reverse_cuthill_mckee, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::reverse_cuthill_mckee;

    #[test]
    fn orders_symmetric_graph_and_keeps_disconnected_vertices() {
        let indices = [
            0, 4, 1, 2, 5, 7, 1, 2, 4, 0, 2, 3, 6, 1, 5, 7, 3, 6, 1, 5, 7,
        ];
        let indptr = [0, 2, 6, 9, 11, 13, 16, 18, 21];
        let result = reverse_cuthill_mckee(&indices, &indptr, 8).unwrap();
        assert_eq!(result.len(), 8);
        let mut sorted = result.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn matches_reference_cuthill_mckee_permutation() {
        let indices = [
            0, 4, 1, 2, 5, 7, 1, 2, 4, 3, 6, 0, 2, 4, 1, 5, 7, 3, 6, 1, 5, 7,
        ];
        let indptr = [0, 2, 6, 9, 11, 14, 17, 19, 22];
        let result = reverse_cuthill_mckee(&indices, &indptr, 8).unwrap();
        assert_eq!(result, [6, 3, 7, 5, 1, 2, 4, 0]);
    }

    #[test]
    fn rejects_invalid_compressed_graph() {
        assert!(reverse_cuthill_mckee(&[2], &[0, 1, 1], 2).is_err());
    }
}
