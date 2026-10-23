//! Number formatting for Matrix Market output.
//!
//! Matches the semantics of the original C++ `fast_matrix_market` backend:
//!   * `precision < 0`  -> shortest round-trip representation (e.g. `1`, `2.5`, `6.25`).
//!   * `precision >= 1` -> exactly `precision` significant digits in scientific
//!     notation with a signed, zero-padded (>= 2 digit) exponent
//!     (e.g. precision 3 -> `1.00e+00`).
//!
//! Integer fields ignore `precision` and are written as plain integers.

/// Format a single floating-point value.
pub fn format_value(v: f64, precision: i32) -> String {
    if precision < 0 {
        format!("{}", v)
    } else {
        scientific(v, precision as usize)
    }
}

/// `p` significant digits in scientific notation, e.g. `1.00e+00`.
fn scientific(v: f64, p: usize) -> String {
    let digits_after = p.saturating_sub(1);
    let s = format!("{:.*e}", digits_after, v);
    fix_exponent(&s)
}

/// Rust's `e` format yields exponents like `0` / `5` / `-3`; Matrix Market wants
/// `+00` / `+05` / `-03` (explicit sign, minimum two digits).
fn fix_exponent(s: &str) -> String {
    match s.split_once('e') {
        Some((mantissa, exp)) => {
            let (sign, digits) = match exp.strip_prefix('-') {
                Some(rest) => ('-', rest),
                None => ('+', exp),
            };
            let digits = if digits.len() < 2 {
                format!("0{}", digits)
            } else {
                digits.to_string()
            };
            format!("{}e{}{}", mantissa, sign, digits)
        }
        None => s.to_string(),
    }
}

/// Format a complex value as `"<re> <im>"`.
pub fn format_complex(re: f64, im: f64, precision: i32) -> String {
    format!(
        "{} {}",
        format_value(re, precision),
        format_value(im, precision)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortest_matches_doctest() {
        assert_eq!(format_value(1.0, -1), "1");
        assert_eq!(format_value(0.0, -1), "0");
        assert_eq!(format_value(2.5, -1), "2.5");
        assert_eq!(format_value(6.25, -1), "6.25");
    }

    #[test]
    fn precision_matches_doctest() {
        assert_eq!(format_value(1.0, 3), "1.00e+00");
        assert_eq!(format_value(0.0, 3), "0.00e+00");
        assert_eq!(format_value(2.5, 3), "2.50e+00");
        assert_eq!(format_value(6.25, 3), "6.25e+00");
        assert_eq!(format_value(3.0, 2), "3.0e+00");
        assert_eq!(format_value(-2.0, 2), "-2.0e+00");
        assert_eq!(format_complex(3.0, 0.0, 2), "3.0e+00 0.0e+00");
        assert_eq!(format_complex(1.0, -2.0, 2), "1.0e+00 -2.0e+00");
    }
}
