//! Matrix Market header and body parsing.
//!
//! The on-disk format:
//! ```text
//! %%MatrixMarket matrix <format> <field> <symmetry>
//! <comment lines, each starting with '%'>
//! <nrows> <ncols> [<nnz>]
//! <body>
//! ```

/// A parsed Matrix Market header.
#[derive(Debug, Clone)]
pub struct ParsedHeader {
    pub object: String,
    pub format: String,
    pub field: String,
    pub symmetry: String,
    pub nrows: i64,
    pub ncols: i64,
    pub nnz: i64,
    pub comment: String,
}

/// Parse the header from the full file text.
///
/// Returns the header plus the body text (everything after the dimension line).
pub fn parse_header(text: &str) -> Result<(ParsedHeader, String), String> {
    let lines: Vec<&str> = text.split('\n').collect();
    if lines.is_empty() {
        return Err("Not a Matrix Market file. Missing banner.".to_string());
    }
    let banner = lines[0].trim_end_matches('\r');
    if !banner.contains("MatrixMarket") {
        return Err("Not a Matrix Market file. Missing banner.".to_string());
    }
    let tokens: Vec<&str> = banner.split_whitespace().collect();
    if tokens.len() < 5 {
        return Err("Invalid MatrixMarket header: malformed banner.".to_string());
    }
    let object = tokens[1].to_string();
    let format = tokens[2].to_string();
    let field = tokens[3].to_string();
    let symmetry = tokens[4].to_string();

    // Comment lines continue until the first non-comment line (the dimension
    // line). Blank lines are treated as comments, like the C++ backend.
    let mut comment_lines: Vec<String> = Vec::new();
    let mut dim_idx: Option<usize> = None;
    for (i, line) in lines.iter().enumerate().skip(1) {
        let line = line.trim_end_matches('\r');
        let trimmed = line.trim();
        if trimmed.starts_with('%') || trimmed.is_empty() {
            let stripped = line.trim_start();
            let body = stripped.strip_prefix('%').unwrap_or(stripped);
            comment_lines.push(body.to_string());
        } else {
            dim_idx = Some(i);
            break;
        }
    }
    let dim_idx = dim_idx
        .ok_or_else(|| "Invalid MatrixMarket header: Premature EOF".to_string())?;
    let comment = comment_lines.join("\n");

    let dim_tokens: Vec<&str> = lines[dim_idx].split_whitespace().collect();
    let mut header = ParsedHeader {
        object,
        format,
        field,
        symmetry,
        nrows: 0,
        ncols: 0,
        nnz: 0,
        comment,
    };

    if header.object == "vector" {
        if dim_tokens.len() < 1 {
            return Err("Header dimension line not of length 1".to_string());
        }
        let vlen: i64 = dim_tokens[0]
            .parse()
            .map_err(|_| "Invalid vector length.".to_string())?;
        if vlen < 0 {
            return Err("Vector length can't be negative.".to_string());
        }
        header.nrows = vlen;
        header.ncols = 1;
        header.nnz = if header.format == "coordinate" {
            dim_tokens
                .get(1)
                .and_then(|t| t.parse().ok())
                .unwrap_or(0)
        } else {
            vlen
        };
    } else {
        if dim_tokens.len() < 2 {
            return Err("Header dimension line not of length 2".to_string());
        }
        header.nrows = dim_tokens[0]
            .parse()
            .map_err(|_| "Invalid matrix dimensions.".to_string())?;
        header.ncols = dim_tokens[1]
            .parse()
            .map_err(|_| "Invalid matrix dimensions.".to_string())?;
        if header.nrows < 0 || header.ncols < 0 {
            return Err("Matrix dimensions can't be negative.".to_string());
        }
        header.nnz = if header.format == "coordinate" {
            dim_tokens
                .get(2)
                .and_then(|t| t.parse().ok())
                .unwrap_or(0)
        } else {
            header.nrows * header.ncols
        };
    }

    let body = lines[dim_idx + 1..].join("\n");
    Ok((header, body))
}

/// A single Matrix Market value, tagged by its field type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Val {
    Real(f64),
    Int(i64),
    Uint(u64),
    Complex { re: f64, im: f64 },
}

impl Val {
    /// Negate the value (used for skew-symmetric mirroring).
    pub fn neg(self) -> Val {
        match self {
            Val::Real(x) => Val::Real(-x),
            Val::Int(x) => Val::Int(-x),
            Val::Uint(x) => Val::Int(-(x as i128) as i64),
            Val::Complex { re, im } => Val::Complex { re: -re, im: -im },
        }
    }

    /// Complex conjugate (used for hermitian mirroring); identity for real fields.
    pub fn conj(self, field: &str) -> Val {
        if field == "complex" {
            match self {
                Val::Complex { re, im } => Val::Complex { re, im: -im },
                other => other,
            }
        } else {
            self
        }
    }
}

/// Read one value from a token slice, advancing `idx`.
/// Complex values consume two tokens (`<re> <im>`); other fields consume one.
fn read_element(tokens: &[&str], idx: &mut usize, field: &str) -> Result<Val, String> {
    match field {
        "real" | "double" => {
            let t = tokens.get(*idx).ok_or("unexpected end of body")?;
            *idx += 1;
            t.parse::<f64>()
                .map(Val::Real)
                .map_err(|_| format!("invalid real value: {t}"))
        }
        "integer" => {
            let t = tokens.get(*idx).ok_or("unexpected end of body")?;
            *idx += 1;
            // Accept "1.0" style tokens some writers emit for integer fields.
            if let Ok(i) = t.parse::<i64>() {
                Ok(Val::Int(i))
            } else if let Ok(f) = t.parse::<f64>() {
                Ok(Val::Int(f as i64))
            } else {
                Err(format!("invalid integer value: {t}"))
            }
        }
        "unsigned-integer" => {
            let t = tokens.get(*idx).ok_or("unexpected end of body")?;
            *idx += 1;
            if let Ok(u) = t.parse::<u64>() {
                Ok(Val::Uint(u))
            } else if let Ok(f) = t.parse::<f64>() {
                Ok(Val::Uint(f as u64))
            } else {
                Err(format!("invalid unsigned integer value: {t}"))
            }
        }
        "complex" => {
            let re_t = tokens.get(*idx).ok_or("unexpected end of body")?;
            let im_t = tokens.get(*idx + 1).ok_or("unexpected end of body")?;
            *idx += 2;
            let re = re_t
                .parse::<f64>()
                .map_err(|_| format!("invalid complex real part: {re_t}"))?;
            let im = im_t
                .parse::<f64>()
                .map_err(|_| format!("invalid complex imag part: {im_t}"))?;
            Ok(Val::Complex { re, im })
        }
        _ => Err(format!("unknown field type: {field}")),
    }
}

/// Whether element `(r, c)` is stored in the file for the given symmetry.
pub fn is_stored(r: i64, c: i64, symmetry: &str) -> bool {
    match symmetry {
        "general" => true,
        "symmetric" | "hermitian" => r >= c,
        "skew-symmetric" => r > c,
        _ => true,
    }
}

/// Parse an array (dense) body into a full row-major matrix, expanding the
/// stored lower triangle into the full matrix according to the symmetry.
pub fn parse_array_body(
    body: &str,
    field: &str,
    symmetry: &str,
    nrows: i64,
    ncols: i64,
) -> Result<Vec<Val>, String> {
    let tokens: Vec<&str> = body.split_whitespace().collect();
    let mut mat = vec![Val::Real(0.0); (nrows * ncols) as usize];
    if field == "pattern" || (nrows == 0 || ncols == 0) {
        return Ok(mat);
    }

    // Matrix Market array format stores values in column-major order.
    let mut idx = 0usize;
    for c in 0..ncols {
        for r in 0..nrows {
            if !is_stored(r, c, symmetry) {
                continue;
            }
            let v = read_element(&tokens, &mut idx, field)?;
            place_value(&mut mat, r, c, nrows, ncols, v, field, symmetry);
        }
    }
    Ok(mat)
}

fn place_value(
    mat: &mut Vec<Val>,
    r: i64,
    c: i64,
    _nrows: i64,
    ncols: i64,
    v: Val,
    field: &str,
    symmetry: &str,
) {
    let set = |mat: &mut Vec<Val>, rr: i64, cc: i64, val: Val| {
        mat[(rr * ncols + cc) as usize] = val;
    };
    match symmetry {
        "general" | "" => set(mat, r, c, v),
        "symmetric" => {
            set(mat, r, c, v);
            if r != c {
                set(mat, c, r, v);
            }
        }
        "hermitian" => {
            set(mat, r, c, v);
            if r != c {
                set(mat, c, r, v.conj(field));
            }
        }
        "skew-symmetric" => {
            // Only strictly-lower entries are stored; the mirror gets -v and the
            // diagonal stays zero.
            set(mat, r, c, v);
            set(mat, c, r, v.neg());
        }
        _ => set(mat, r, c, v),
    }
}

/// Parse a coordinate body into the *stored* (0-based) triplets.
///
/// Symmetry generalization is intentionally NOT applied here — the Python
/// layer performs it (matching the original `_read_body_coo` contract).
pub fn parse_coo_body(
    body: &str,
    field: &str,
    nnz: i64,
) -> Result<(Vec<i64>, Vec<i64>, Vec<Val>), String> {
    let tokens: Vec<&str> = body.split_whitespace().collect();
    let mut rows = Vec::with_capacity(nnz as usize);
    let mut cols = Vec::with_capacity(nnz as usize);
    let mut data = Vec::with_capacity(nnz as usize);
    let mut idx = 0usize;
    for _ in 0..nnz {
        let r_t = tokens.get(idx).ok_or("unexpected end of body")?;
        let c_t = tokens.get(idx + 1).ok_or("unexpected end of body")?;
        idx += 2;
        let r: i64 = r_t.parse().map_err(|_| format!("invalid row index: {r_t}"))?;
        let c: i64 = c_t.parse().map_err(|_| format!("invalid column index: {c_t}"))?;
        rows.push(r - 1);
        cols.push(c - 1);
        if field != "pattern" {
            data.push(read_element(&tokens, &mut idx, field)?);
        }
    }
    Ok((rows, cols, data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_general_coordinate_header() {
        let text = "%%MatrixMarket matrix coordinate real general\n5 5 7\n2 3 1.0\n";
        let (h, body) = parse_header(text).unwrap();
        assert_eq!(h.object, "matrix");
        assert_eq!(h.format, "coordinate");
        assert_eq!(h.field, "real");
        assert_eq!(h.symmetry, "general");
        assert_eq!((h.nrows, h.ncols, h.nnz), (5, 5, 7));
        assert!(body.contains("2 3 1.0"));
    }

    #[test]
    fn parses_comment_lines() {
        let text = "%%MatrixMarket matrix array real general\n%\n% hello\n2 4\n1 0 0 2.5 0 0 0 6.25\n";
        let (h, body) = parse_header(text).unwrap();
        assert_eq!(h.comment, "\n hello");
        assert_eq!(h.nrows, 2);
        assert_eq!(h.ncols, 4);
        assert_eq!(h.nnz, 8);
        assert!(body.starts_with("1 0 0 2.5"));
    }

    #[test]
    fn coo_body_is_zero_based_stored() {
        let body = "2 3 1.0\n3 4 2.0\n";
        let (i, j, data) = parse_coo_body(body, "real", 2).unwrap();
        assert_eq!(i, vec![1, 2]);
        assert_eq!(j, vec![2, 3]);
        assert_eq!(data, vec![Val::Real(1.0), Val::Real(2.0)]);
    }

    #[test]
    fn array_body_expands_symmetry() {
        // symmetric 2x2: stores (0,0)=1, (1,0)=2, (1,1)=3
        let body = "1.0\n2.0\n3.0\n";
        let mat = parse_array_body(body, "real", "symmetric", 2, 2).unwrap();
        assert_eq!(mat, vec![Val::Real(1.0), Val::Real(2.0), Val::Real(2.0), Val::Real(3.0)]);
    }

    #[test]
    fn array_body_expands_hermitian() {
        // hermitian 2x2: stores (0,0)=1, (1,0)=1+2i, (1,1)=3
        let body = "1.0 0.0\n1.0 2.0\n3.0 0.0\n";
        let mat = parse_array_body(body, "complex", "hermitian", 2, 2).unwrap();
        assert_eq!(
            mat,
            vec![
                Val::Complex { re: 1.0, im: 0.0 },
                Val::Complex { re: 1.0, im: -2.0 },
                Val::Complex { re: 1.0, im: 2.0 },
                Val::Complex { re: 3.0, im: 0.0 },
            ]
        );
    }

    #[test]
    fn array_body_expands_skew_symmetric() {
        // skew-symmetric 2x2: stores only (1,0)=2.0
        let body = "2.0\n";
        let mat = parse_array_body(body, "real", "skew-symmetric", 2, 2).unwrap();
        assert_eq!(
            mat,
            vec![Val::Real(0.0), Val::Real(-2.0), Val::Real(2.0), Val::Real(0.0)]
        );
    }
}
