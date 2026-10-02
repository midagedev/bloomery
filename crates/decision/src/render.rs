//! JSON text as Python writes it.
//!
//! [`render`] is the encoder's `render`: a string passes through, anything else is
//! `json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)`. [`dumps`] is the
//! same writer with the key order kept (the response body). Floats print as Python's `repr`
//! ([`float_repr`]), integers as their exact digits.

use std::fmt::Write as _;

use crate::json::Json;

/// The encoder's text for one request value.
#[must_use]
pub fn render(v: &Json) -> String {
    match v {
        Json::Str(s) => s.clone(),
        _ => dumps(v, true),
    }
}

/// Compact JSON with non-ASCII kept; object keys sorted by code point when `sort_keys`, else in
/// their order.
#[must_use]
pub fn dumps(v: &Json, sort_keys: bool) -> String {
    let mut out = String::new();
    write_value(&mut out, v, sort_keys);
    out
}

fn write_value(out: &mut String, v: &Json, sort_keys: bool) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(true) => out.push_str("true"),
        Json::Bool(false) => out.push_str("false"),
        Json::Int(s) => out.push_str(s),
        Json::Float(f) => out.push_str(&float_repr(*f)),
        Json::Str(s) => write_str(out, s),
        Json::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(out, item, sort_keys);
            }
            out.push(']');
        }
        Json::Object(pairs) => {
            let mut order: Vec<&(String, Json)> = pairs.iter().collect();
            if sort_keys {
                // UTF-8 byte order is code point order, which is Python's str order.
                order.sort_by(|a, b| a.0.cmp(&b.0));
            }
            out.push('{');
            for (i, (k, item)) in order.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_str(out, k);
                out.push(':');
                write_value(out, item, sort_keys);
            }
            out.push('}');
        }
    }
}

/// Python's `py_encode_basestring`: `"` and `\` escaped, the five named controls by name, every
/// other code point below U+0020 as `\u00xx` (lower-case hex), everything else as is.
fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if u32::from(c) < 0x20 => {
                write!(out, "\\u{:04x}", u32::from(c)).expect("writing to a String cannot fail");
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `float.__repr__` of a finite f64: the shortest digits that read back to `x`, in fixed
/// notation when the decimal point position `p` (value = 0.d₁d₂… × 10^p) is in −3..=16, with `.0`
/// on an integral value, else `d[.ddd]e±XX` (at least two exponent digits).
#[must_use]
pub fn float_repr(x: f64) -> String {
    // `{:e}` writes the shortest round-trip digits as `[-]d[.ddd]e<exp>`.
    let sci = format!("{x:e}");
    let (sign, body) = match sci.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", sci.as_str()),
    };
    let (mantissa, exp) = body
        .split_once('e')
        .expect("`{:e}` of a finite f64 has an exponent");
    let exp: i32 = exp.parse().expect("`{:e}` writes a decimal exponent");
    let digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    let point = exp + 1;
    let n = i32::try_from(digits.len()).expect("an f64 has at most 17 significant digits");
    let mut out = String::from(sign);
    if point <= -4 || point > 16 {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let e = point - 1;
        write!(
            out,
            "e{}{:02}",
            if e < 0 { '-' } else { '+' },
            e.unsigned_abs()
        )
        .expect("writing to a String cannot fail");
    } else if point <= 0 {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', point.unsigned_abs() as usize));
        out.push_str(&digits);
    } else if point >= n {
        out.push_str(&digits);
        out.extend(std::iter::repeat_n(
            '0',
            (point - n).unsigned_abs() as usize,
        ));
        out.push_str(".0");
    } else {
        let p = point.unsigned_abs() as usize;
        out.push_str(&digits[..p]);
        out.push('.');
        out.push_str(&digits[p..]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse;

    /// Each right side is CPython's `repr(float(left))`.
    #[test]
    fn float_repr_is_pythons() {
        for (x, want) in [
            (1250.0, "1250.0"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1.5e-7, "1.5e-07"),
            (-0.0, "-0.0"),
            (0.0, "0.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (0.25, "0.25"),
            (0.1, "0.1"),
            (123456789012345678.0, "1.2345678901234568e+17"),
            (1234567890123456.7, "1234567890123456.8"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (-2.5e-5, "-2.5e-05"),
            (1e100, "1e+100"),
            (12.5, "12.5"),
        ] {
            assert_eq!(float_repr(x), want, "{x:?}");
        }
    }

    /// The right sides are `json.dumps(json.loads(left), ensure_ascii=False, separators=(",", ":"),
    /// sort_keys=True)`.
    #[test]
    fn render_is_python_dumps() {
        for (text, want) in [
            (r#""plain text""#, "plain text"),
            (
                r#"{"b": 1, "a": [true, null, 1.0, 1e16]}"#,
                r#"{"a":[true,null,1.0,1e+16],"b":1}"#,
            ),
            (
                r#"{"Mid": 1, "_m": 2, "alpha": 3, "Z": 4}"#,
                r#"{"Mid":1,"Z":4,"_m":2,"alpha":3}"#,
            ),
            (
                r#"{"고객": "김민지 😡", "a": "x"}"#,
                r#"{"a":"x","고객":"김민지 😡"}"#,
            ),
            (
                r#"{"s": "a\u0001b\tc\nd\u001fe\u007f\"\\"}"#,
                "{\"s\":\"a\\u0001b\\tc\\nd\\u001fe\u{7f}\\\"\\\\\"}",
            ),
            (
                r#"{"big": 123456789012345678901234567890, "neg": -0, "e": 1.5e-07}"#,
                r#"{"big":123456789012345678901234567890,"e":1.5e-07,"neg":0}"#,
            ),
            (r#"[]"#, "[]"),
            (r#"{}"#, "{}"),
            (
                r#"{"o": {"z": 1, "a": {"y": 2, "b": 3}}}"#,
                r#"{"o":{"a":{"b":3,"y":2},"z":1}}"#,
            ),
        ] {
            assert_eq!(render(&parse(text).unwrap()), want, "{text}");
        }
    }

    #[test]
    fn dumps_keeps_order_unsorted() {
        let v = parse(r#"{"b": 1, "a": 0.5}"#).unwrap();
        assert_eq!(dumps(&v, false), r#"{"b":1,"a":0.5}"#);
    }
}
