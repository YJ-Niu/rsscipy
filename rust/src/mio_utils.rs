use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyfunction]
fn chars_to_strings(
    input: &Bound<'_, PyBytes>,
    shape: Vec<usize>,
    codepoint_size: usize,
) -> PyResult<Vec<String>> {
    let Some(&width) = shape.last() else {
        return Err(PyValueError::new_err(
            "character array must have at least one dimension",
        ));
    };
    if codepoint_size != 4 {
        return Err(PyValueError::new_err(
            "character array must contain 32-bit Unicode code points",
        ));
    }
    let elements = shape
        .iter()
        .try_fold(1usize, |product, &dimension| product.checked_mul(dimension));
    let expected_bytes = elements
        .and_then(|count| count.checked_mul(codepoint_size))
        .ok_or_else(|| PyValueError::new_err("character array size overflows"))?;
    if input.as_bytes().len() != expected_bytes {
        return Err(PyValueError::new_err(
            "character buffer length does not match its shape",
        ));
    }
    if width == 0 {
        return Ok(Vec::new());
    }

    let bytes = input.as_bytes();
    let mut strings = Vec::with_capacity(expected_bytes / (width * codepoint_size));
    for row in bytes.chunks_exact(width * codepoint_size) {
        let mut string = String::with_capacity(width);
        for codepoint in row.chunks_exact(codepoint_size) {
            let value = u32::from_ne_bytes(codepoint.try_into().expect("four-byte chunk"));
            let character = char::from_u32(value).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "character array contains invalid Unicode code point {value:#x}"
                ))
            })?;
            string.push(character);
        }
        strings.push(string);
    }
    Ok(strings)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(chars_to_strings, module)?)?;
    Ok(())
}
