use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

fn decode_simplices(bytes: &[u8], rows: usize, columns: usize) -> PyResult<Vec<i32>> {
    if columns != 3 || rows.checked_mul(columns).and_then(|n| n.checked_mul(4)) != Some(bytes.len())
    {
        return Err(PyValueError::new_err(
            "simplices must be a contiguous int32 array with three columns",
        ));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| i32::from_ne_bytes(*chunk))
        .collect())
}

fn sort_region(
    region_index: usize,
    region: &[usize],
    simplices: &[i32],
    simplex_count: usize,
) -> PyResult<Vec<usize>> {
    if region.is_empty() {
        return Ok(Vec::new());
    }
    if region.iter().any(|&index| index >= simplex_count) {
        return Err(PyIndexError::new_err(format!(
            "region {region_index} contains an invalid simplex index"
        )));
    }

    let mut current_simplex = region[0];
    let first = &simplices[current_simplex * 3..current_simplex * 3 + 3];
    let mut current_vertex = first
        .iter()
        .copied()
        .find(|&vertex| vertex != region_index as i32)
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "region {region_index} has no adjacent vertex in its first simplex"
            ))
        })?;
    let mut sorted = Vec::with_capacity(region.len());

    for _ in 0..region.len() {
        let visited_simplex = current_simplex;
        sorted.push(visited_simplex);
        for &simplex_index in region {
            if simplex_index == visited_simplex {
                continue;
            }
            let simplex = &simplices[simplex_index * 3..simplex_index * 3 + 3];
            if simplex.contains(&current_vertex) {
                current_simplex = simplex_index;
            }
        }
        let simplex = &simplices[current_simplex * 3..current_simplex * 3 + 3];
        current_vertex = simplex
            .iter()
            .copied()
            .find(|&vertex| vertex != region_index as i32 && vertex != current_vertex)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "region {region_index} contains disconnected simplices"
                ))
            })?;
    }
    Ok(sorted)
}

#[pyfunction]
fn sort_vertices_of_regions(
    py: Python<'_>,
    simplices_bytes: &Bound<'_, PyBytes>,
    simplex_rows: usize,
    simplex_columns: usize,
    regions: &Bound<'_, PyList>,
) -> PyResult<()> {
    let simplices = decode_simplices(simplices_bytes.as_bytes(), simplex_rows, simplex_columns)?;
    let mut ordered_regions = Vec::with_capacity(regions.len());
    for (region_index, region) in regions.iter().enumerate() {
        let indices = region.extract::<Vec<usize>>().map_err(|_| {
            PyValueError::new_err(format!(
                "region {region_index} must contain integer indices"
            ))
        })?;
        ordered_regions.push(sort_region(
            region_index,
            &indices,
            &simplices,
            simplex_rows,
        )?);
    }

    for (index, region) in ordered_regions.into_iter().enumerate() {
        regions.set_item(index, PyList::new(py, region)?)?;
    }
    Ok(())
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(sort_vertices_of_regions, module)?)?;
    Ok(())
}
