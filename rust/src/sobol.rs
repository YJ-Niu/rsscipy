use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

const MAX_DIM: usize = 21201;
const MAX_DEG: usize = 18;

fn decode_unsigned(bytes: &[u8], item_size: usize, name: &str) -> PyResult<Vec<u64>> {
    match item_size {
        4 if bytes.len().is_multiple_of(4) => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_ne_bytes(*chunk) as u64)
            .collect()),
        8 if bytes.len().is_multiple_of(8) => Ok(bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| u64::from_ne_bytes(*chunk))
            .collect()),
        4 | 8 => Err(PyValueError::new_err(format!(
            "{name} buffer has an invalid length for uint{bits}",
            bits = item_size * 8
        ))),
        _ => Err(PyValueError::new_err(format!(
            "{name} must use uint32 or uint64 values"
        ))),
    }
}

fn decode_unsigned_le(bytes: &[u8], name: &str) -> PyResult<Vec<u64>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(PyValueError::new_err(format!(
            "{name} buffer is not a sequence of little-endian int64 values"
        )));
    }
    Ok(bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| u64::from_le_bytes(*chunk))
        .collect())
}

fn encode_unsigned(values: &[u64], item_size: usize) -> PyResult<Vec<u8>> {
    let capacity = values
        .len()
        .checked_mul(item_size)
        .ok_or_else(|| PyValueError::new_err("Sobol buffer size overflows"))?;
    let mut bytes = Vec::with_capacity(capacity);
    match item_size {
        4 => {
            if values.iter().any(|value| *value > u32::MAX as u64) {
                return Err(PyValueError::new_err("Sobol value does not fit in uint32"));
            }
            for value in values {
                bytes.extend_from_slice(&(*value as u32).to_ne_bytes());
            }
        }
        8 => {
            for value in values {
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
        }
        _ => {
            return Err(PyValueError::new_err(
                "Sobol arrays must use uint32 or uint64 values",
            ));
        }
    }
    Ok(bytes)
}

fn byte_length(rows: usize, columns: usize, item_size: usize) -> PyResult<usize> {
    rows.checked_mul(columns)
        .and_then(|length| length.checked_mul(item_size))
        .ok_or_else(|| PyValueError::new_err("Sobol array size overflows"))
}

#[pyfunction]
fn sobol_initialize_v(
    poly_bytes: &Bound<'_, PyBytes>,
    vinit_bytes: &Bound<'_, PyBytes>,
    dim: usize,
    bits: usize,
    item_size: usize,
) -> PyResult<Vec<u8>> {
    if dim > MAX_DIM || bits > item_size * 8 || bits > 64 {
        return Err(PyValueError::new_err(
            "invalid Sobol direction number shape",
        ));
    }
    let poly = decode_unsigned_le(poly_bytes.as_bytes(), "poly")?;
    let vinit = decode_unsigned_le(vinit_bytes.as_bytes(), "vinit")?;
    if poly.len() != MAX_DIM || vinit.len() != MAX_DIM * MAX_DEG {
        return Err(PyValueError::new_err(
            "Sobol direction number data has an unexpected shape",
        ));
    }
    let length = byte_length(dim, bits, item_size)?;
    let mut v = vec![0_u64; length / item_size];
    if dim > 0 {
        for value in &mut v[..bits] {
            *value = 1;
        }
        for (row, &polynomial) in poly.iter().enumerate().take(dim).skip(1) {
            let degree = (u64::BITS - polynomial.leading_zeros() - 1) as usize;
            if degree == 0 || degree > MAX_DEG || degree > bits {
                return Err(PyValueError::new_err(
                    "invalid Sobol direction number polynomial",
                ));
            }
            let row_start = row * bits;
            for column in 0..degree {
                v[row_start + column] = vinit[row + column * MAX_DIM];
            }
            for column in degree..bits {
                let mut new_value = v[row_start + column - degree];
                let mut power = 1_u64;
                for k in 0..degree {
                    power = power.wrapping_shl(1);
                    if (polynomial >> (degree - 1 - k)) & 1 != 0 {
                        new_value ^= power.wrapping_mul(v[row_start + column - k - 1]);
                    }
                }
                v[row_start + column] = new_value;
            }
        }
        let mut power = 1_u64;
        for column in (0..bits).rev() {
            for row in 0..dim {
                let index = row * bits + column;
                v[index] = v[index].wrapping_mul(power);
            }
            power = power.wrapping_mul(2);
        }
    }
    encode_unsigned(&v, item_size)
}

fn validate_scramble(
    dim: usize,
    bits: usize,
    ltm: &[u64],
    sv: &[u64],
    item_size: usize,
) -> PyResult<()> {
    if dim > MAX_DIM || bits > item_size * 8 || bits > 64 {
        return Err(PyValueError::new_err("invalid Sobol scramble shape"));
    }
    if ltm.len() != byte_length(dim, bits * bits, item_size)? / item_size
        || sv.len() != byte_length(dim, bits, item_size)? / item_size
    {
        return Err(PyValueError::new_err(
            "Sobol scramble arrays have inconsistent shapes",
        ));
    }
    Ok(())
}

#[pyfunction]
fn sobol_cscramble(
    dim: usize,
    bits: usize,
    ltm_bytes: &Bound<'_, PyBytes>,
    sv_bytes: &Bound<'_, PyBytes>,
    item_size: usize,
) -> PyResult<(Vec<u8>, Vec<u8>)> {
    let mut ltm = decode_unsigned(ltm_bytes.as_bytes(), item_size, "ltm")?;
    let mut sv = decode_unsigned(sv_bytes.as_bytes(), item_size, "sv")?;
    validate_scramble(dim, bits, &ltm, &sv, item_size)?;
    for row in 0..dim {
        for index in 0..bits {
            ltm[row * bits * bits + index * bits + index] = 1;
        }
    }
    for row in 0..dim {
        for column in 0..bits {
            let direction = sv[row * bits + column];
            let mut result = 0_u64;
            for matrix_row in 0..bits {
                let mut parity = 0_u32;
                for matrix_column in 0..bits {
                    let matrix_value =
                        ltm[row * bits * bits + matrix_row * bits + (bits - 1 - matrix_column)];
                    parity ^=
                        ((matrix_value & 1) as u32) & (((direction >> matrix_column) & 1) as u32);
                }
                result |= (parity as u64) << (bits - 1 - matrix_row);
            }
            sv[row * bits + column] = result;
        }
    }
    Ok((
        encode_unsigned(&ltm, item_size)?,
        encode_unsigned(&sv, item_size)?,
    ))
}

#[pyfunction]
fn sobol_fill_p_cumulative(
    p_bytes: &Bound<'_, PyBytes>,
    output_length: usize,
) -> PyResult<Vec<u8>> {
    let bytes = p_bytes.as_bytes();
    if output_length.checked_mul(8) != Some(bytes.len()) {
        return Err(PyValueError::new_err(
            "probability arrays have inconsistent lengths",
        ));
    }
    let p: Vec<f64> = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| f64::from_ne_bytes(*chunk))
        .collect();
    let mut total = 0.0;
    let mut cumulative = Vec::with_capacity(p.len());
    for probability in p {
        total += probability;
        cumulative.push(total);
    }
    Ok(cumulative.into_iter().flat_map(f64::to_ne_bytes).collect())
}

fn find_index(cumulative: &[f64], value: f64) -> usize {
    let mut left = 0;
    let mut right = cumulative.len() - 1;
    while right > left {
        let middle = (left + right) / 2;
        if value > cumulative[middle] {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    right
}

#[pyfunction]
fn sobol_test_find_index(
    cumulative_bytes: &Bound<'_, PyBytes>,
    size: usize,
    value: f64,
) -> PyResult<usize> {
    let bytes = cumulative_bytes.as_bytes();
    if size == 0 || size.checked_mul(8) != Some(bytes.len()) {
        return Err(PyValueError::new_err(
            "cumulative probability array has an invalid size",
        ));
    }
    let cumulative: Vec<f64> = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| f64::from_ne_bytes(*chunk))
        .collect();
    Ok(find_index(&cumulative, value))
}

#[pyfunction]
fn sobol_categorize(
    draws_bytes: &Bound<'_, PyBytes>,
    cumulative_bytes: &Bound<'_, PyBytes>,
    result_bytes: &Bound<'_, PyBytes>,
    result_length: usize,
    item_size: usize,
) -> PyResult<Vec<u8>> {
    let decode_f64 = |bytes: &[u8], name: &str| -> PyResult<Vec<f64>> {
        if !bytes.len().is_multiple_of(8) {
            return Err(PyValueError::new_err(format!(
                "{name} buffer is not float64"
            )));
        }
        Ok(bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| f64::from_ne_bytes(*chunk))
            .collect())
    };
    let draws = decode_f64(draws_bytes.as_bytes(), "draws")?;
    let cumulative = decode_f64(cumulative_bytes.as_bytes(), "cumulative")?;
    let result_raw = result_bytes.as_bytes();
    let expected_result_length = result_length
        .checked_mul(item_size)
        .ok_or_else(|| PyValueError::new_err("categorical result size overflows"))?;
    if cumulative.is_empty()
        || cumulative.len() != result_length
        || result_raw.len() != expected_result_length
    {
        return Err(PyValueError::new_err(
            "Sobol categorical arrays have inconsistent shapes",
        ));
    }
    let mut output = Vec::with_capacity(expected_result_length);
    match item_size {
        4 => {
            let mut result: Vec<i32> = result_raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| i32::from_ne_bytes(*chunk))
                .collect();
            for draw in draws {
                let index = find_index(&cumulative, draw);
                result[index] = result[index]
                    .checked_add(1)
                    .ok_or_else(|| PyValueError::new_err("categorical count overflows"))?;
            }
            for value in result {
                output.extend_from_slice(&value.to_ne_bytes());
            }
        }
        8 => {
            let mut result: Vec<i64> = result_raw
                .as_chunks::<8>()
                .0
                .iter()
                .map(|chunk| i64::from_ne_bytes(*chunk))
                .collect();
            for draw in draws {
                let index = find_index(&cumulative, draw);
                result[index] = result[index]
                    .checked_add(1)
                    .ok_or_else(|| PyValueError::new_err("categorical count overflows"))?;
            }
            for value in result {
                output.extend_from_slice(&value.to_ne_bytes());
            }
        }
        _ => {
            return Err(PyValueError::new_err(
                "categorical result must use int32 or int64 values",
            ));
        }
    }
    Ok(output)
}

fn decode_quasi(bytes: &[u8], dim: usize, item_size: usize) -> PyResult<Vec<u64>> {
    let quasi = decode_unsigned(bytes, item_size, "quasi")?;
    if quasi.len() != dim {
        return Err(PyValueError::new_err(
            "Sobol quasi-state has an invalid shape",
        ));
    }
    Ok(quasi)
}

fn low_zero_bit(value: u64) -> usize {
    value.trailing_ones() as usize + 1
}

#[pyfunction]
fn sobol_draw(
    n: usize,
    num_gen: u64,
    dim: usize,
    scale: f64,
    sv_bytes: &Bound<'_, PyBytes>,
    quasi_bytes: &Bound<'_, PyBytes>,
    item_size: usize,
) -> PyResult<(Vec<u8>, Vec<u8>)> {
    let sv = decode_unsigned(sv_bytes.as_bytes(), item_size, "sv")?;
    let mut quasi = decode_quasi(quasi_bytes.as_bytes(), dim, item_size)?;
    let bits = sv.len().checked_div(dim).unwrap_or(0);
    let sample_len = n
        .checked_mul(dim)
        .ok_or_else(|| PyValueError::new_err("Sobol sample size overflows"))?;
    if dim.checked_mul(bits) != Some(sv.len()) {
        return Err(PyValueError::new_err(
            "Sobol draw arrays have invalid shapes",
        ));
    }
    let mut output = Vec::with_capacity(sample_len);
    let mut generation = num_gen;
    for _ in 0..n {
        let bit = low_zero_bit(generation);
        if bit > bits {
            return Err(PyValueError::new_err(
                "Sobol sequence exhausted the configured bit depth",
            ));
        }
        for coordinate in 0..dim {
            let value = quasi[coordinate] ^ sv[coordinate * bits + bit - 1];
            quasi[coordinate] = value;
            output.push(value as f64 * scale);
        }
        generation = generation.wrapping_add(1);
    }
    Ok((
        encode_unsigned(&quasi, item_size)?,
        output.into_iter().flat_map(f64::to_ne_bytes).collect(),
    ))
}

#[pyfunction]
fn sobol_fast_forward(
    n: u64,
    num_gen: u64,
    dim: usize,
    sv_bytes: &Bound<'_, PyBytes>,
    quasi_bytes: &Bound<'_, PyBytes>,
    item_size: usize,
) -> PyResult<Vec<u8>> {
    let sv = decode_unsigned(sv_bytes.as_bytes(), item_size, "sv")?;
    let mut quasi = decode_quasi(quasi_bytes.as_bytes(), dim, item_size)?;
    let bits = sv.len().checked_div(dim).unwrap_or(0);
    if dim.checked_mul(bits) != Some(sv.len()) {
        return Err(PyValueError::new_err(
            "Sobol direction matrix has an invalid shape",
        ));
    }
    let mut generation = num_gen;
    for _ in 0..n {
        let bit = low_zero_bit(generation);
        if bit > bits {
            return Err(PyValueError::new_err(
                "Sobol sequence exhausted the configured bit depth",
            ));
        }
        for coordinate in 0..dim {
            quasi[coordinate] ^= sv[coordinate * bits + bit - 1];
        }
        generation = generation.wrapping_add(1);
    }
    encode_unsigned(&quasi, item_size)
}

#[pyfunction]
fn sobol_test_low_0_bit(value: u64) -> usize {
    low_zero_bit(value)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("_MAXDIM", MAX_DIM)?;
    module.add("_MAXDEG", MAX_DEG)?;
    module.add_function(wrap_pyfunction!(sobol_initialize_v, module)?)?;
    module.add_function(wrap_pyfunction!(sobol_cscramble, module)?)?;
    module.add_function(wrap_pyfunction!(sobol_fill_p_cumulative, module)?)?;
    module.add_function(wrap_pyfunction!(sobol_categorize, module)?)?;
    module.add_function(wrap_pyfunction!(sobol_draw, module)?)?;
    module.add_function(wrap_pyfunction!(sobol_fast_forward, module)?)?;
    module.add_function(wrap_pyfunction!(sobol_test_find_index, module)?)?;
    module.add_function(wrap_pyfunction!(sobol_test_low_0_bit, module)?)?;
    Ok(())
}
