//! A JSON value as Python's `json.loads` builds it, and the reader that builds it.
//!
//! The encoder renders the request back with `json.dumps`, so the value keeps what Python keeps:
//! an object's key order (a duplicate key keeps its first position and takes its last value, as a
//! dict does), an integer's exact digits at any length, and a float as the f64 Python's `float()`
//! gives (Rust's `str::parse::<f64>` rounds correctly, as `float()` does). What Python accepts and
//! this engine refuses by name: the `NaN`, `Infinity` and `-Infinity` literals, a float literal past
//! the f64 range (Python reads it as `inf`), and an escaped lone surrogate (a Rust string cannot hold
//! one).
//!
//! The reader is ours rather than serde_json's because exact integers would need serde_json's
//! `arbitrary_precision`, a feature that cargo unifies into every crate built beside this one.

use std::collections::HashMap;

use crate::Error;

/// Nesting deeper than this is refused by name (Python stops near its recursion limit).
const MAX_DEPTH: usize = 512;

/// One JSON value.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// An integer literal as Python's `int` holds it, in canonical decimal (`-0` is `0`).
    Int(String),
    /// A literal with a fraction or an exponent: always finite.
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    /// Keys in first-insertion order, each once.
    Object(Vec<(String, Json)>),
}

impl Json {
    /// The value under `key` when `self` is an object holding it.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The string when `self` is one.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Python's truth value: `None`, `False`, `0`, `0.0`, `""`, `[]` and `{}` are false.
    #[must_use]
    pub fn truthy(&self) -> bool {
        match self {
            Json::Null => false,
            Json::Bool(b) => *b,
            Json::Int(s) => s != "0",
            Json::Float(f) => *f != 0.0,
            Json::Str(s) => !s.is_empty(),
            Json::Array(a) => !a.is_empty(),
            Json::Object(o) => !o.is_empty(),
        }
    }

    /// The f64 of a number (an integer through its decimal text), else `None`.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Int(s) => s.parse().ok(),
            Json::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// The integer when `self` is one that fits u64.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Int(s) => s.parse().ok(),
            _ => None,
        }
    }

    /// A Python-style type name for error messages.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Json::Null => "null",
            Json::Bool(_) => "a boolean",
            Json::Int(_) => "an integer",
            Json::Float(_) => "a float",
            Json::Str(_) => "a string",
            Json::Array(_) => "an array",
            Json::Object(_) => "an object",
        }
    }
}

/// `text` as one JSON document: one value, whitespace around it allowed (space, tab, CR, LF, as
/// Python's reader allows), nothing after it.
pub fn parse(text: &str) -> Result<Json, Error> {
    let mut r = Reader {
        b: text.as_bytes(),
        at: 0,
    };
    r.ws();
    let v = r.value(0)?;
    r.ws();
    if r.at != r.b.len() {
        return Err(r.err("extra data after the value"));
    }
    Ok(v)
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn err(&self, what: &str) -> Error {
        Error::Json {
            at: self.at,
            what: what.to_string(),
        }
    }

    fn ws(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.b.get(self.at) {
            self.at += 1;
        }
    }

    fn eat(&mut self, lit: &str) -> bool {
        if self.b[self.at..].starts_with(lit.as_bytes()) {
            self.at += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, Error> {
        if depth > MAX_DEPTH {
            return Err(Error::JsonTooDeep(MAX_DEPTH));
        }
        match self.b.get(self.at) {
            None => Err(self.err("a value is due")),
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') if self.eat("true") => Ok(Json::Bool(true)),
            Some(b'f') if self.eat("false") => Ok(Json::Bool(false)),
            Some(b'n') if self.eat("null") => Ok(Json::Null),
            Some(b'N') if self.eat("NaN") => Err(Error::JsonNonFinite("NaN")),
            Some(b'I') if self.eat("Infinity") => Err(Error::JsonNonFinite("Infinity")),
            Some(b'-') if self.b[self.at..].starts_with(b"-Infinity") => {
                Err(Error::JsonNonFinite("-Infinity"))
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.err("a value is due")),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, Error> {
        self.at += 1;
        let mut pairs: Vec<(String, Json)> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::new();
        self.ws();
        if self.eat("}") {
            return Ok(Json::Object(pairs));
        }
        loop {
            self.ws();
            if self.b.get(self.at) != Some(&b'"') {
                return Err(self.err("an object key (a string) is due"));
            }
            let key = self.string()?;
            self.ws();
            if !self.eat(":") {
                return Err(self.err("':' is due after an object key"));
            }
            self.ws();
            let v = self.value(depth + 1)?;
            match index.get(&key) {
                Some(&i) => pairs[i].1 = v,
                None => {
                    index.insert(key.clone(), pairs.len());
                    pairs.push((key, v));
                }
            }
            self.ws();
            if self.eat(",") {
                continue;
            }
            if self.eat("}") {
                return Ok(Json::Object(pairs));
            }
            return Err(self.err("',' or '}' is due in an object"));
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, Error> {
        self.at += 1;
        let mut items = Vec::new();
        self.ws();
        if self.eat("]") {
            return Ok(Json::Array(items));
        }
        loop {
            self.ws();
            items.push(self.value(depth + 1)?);
            self.ws();
            if self.eat(",") {
                continue;
            }
            if self.eat("]") {
                return Ok(Json::Array(items));
            }
            return Err(self.err("',' or ']' is due in an array"));
        }
    }

    fn hex4(&mut self) -> Result<u32, Error> {
        let digits = self
            .b
            .get(self.at..self.at + 4)
            .and_then(|d| std::str::from_utf8(d).ok())
            .filter(|d| d.bytes().all(|c| c.is_ascii_hexdigit()))
            .ok_or_else(|| self.err("four hex digits are due after \\u"))?;
        let v = u32::from_str_radix(digits, 16).map_err(|_| self.err("bad \\u escape"))?;
        self.at += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String, Error> {
        self.at += 1;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let Some(&c) = self.b.get(self.at) else {
                return Err(self.err("unterminated string"));
            };
            match c {
                b'"' => {
                    self.at += 1;
                    // The input is a &str and every escape pushes whole UTF-8 sequences.
                    return String::from_utf8(out).map_err(|_| self.err("invalid UTF-8"));
                }
                b'\\' => {
                    self.at += 1;
                    let Some(&e) = self.b.get(self.at) else {
                        return Err(self.err("unterminated escape"));
                    };
                    self.at += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                if !self.b[self.at..].starts_with(b"\\u") {
                                    return Err(Error::JsonLoneSurrogate(hi));
                                }
                                self.at += 2;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err(Error::JsonLoneSurrogate(hi));
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else if (0xDC00..0xE000).contains(&hi) {
                                return Err(Error::JsonLoneSurrogate(hi));
                            } else {
                                hi
                            };
                            char::from_u32(cp).ok_or_else(|| self.err("bad \\u escape"))?
                        }
                        _ => return Err(self.err("invalid escape")),
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                0..=0x1f => return Err(self.err("a control character inside a string")),
                _ => {
                    out.push(c);
                    self.at += 1;
                }
            }
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.at;
        while let Some(b'0'..=b'9') = self.b.get(self.at) {
            self.at += 1;
        }
        self.at - start
    }

    /// Python's number rule: `-?(0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?`, an int without the last two parts.
    fn number(&mut self) -> Result<Json, Error> {
        let start = self.at;
        let negative = self.eat("-");
        match self.b.get(self.at) {
            Some(b'0') => self.at += 1,
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return Err(self.err("a digit is due")),
        }
        let int_end = self.at;
        let mut float = false;
        if self.b.get(self.at) == Some(&b'.')
            && matches!(self.b.get(self.at + 1), Some(b'0'..=b'9'))
        {
            self.at += 1;
            self.digits();
            float = true;
        }
        if let Some(b'e' | b'E') = self.b.get(self.at) {
            let mark = self.at;
            self.at += 1;
            if let Some(b'+' | b'-') = self.b.get(self.at) {
                self.at += 1;
            }
            if self.digits() == 0 {
                self.at = mark;
            } else {
                float = true;
            }
        }
        let text =
            std::str::from_utf8(&self.b[start..self.at]).map_err(|_| self.err("invalid UTF-8"))?;
        if float {
            let f: f64 = text.parse().map_err(|_| self.err("bad float literal"))?;
            if !f.is_finite() {
                return Err(Error::JsonFloatRange(text.to_string()));
            }
            return Ok(Json::Float(f));
        }
        let digits = &text[usize::from(negative)..int_end - start];
        Ok(Json::Int(if digits == "0" {
            "0".to_string()
        } else {
            text.to_string()
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_keys_keep_first_position_and_last_value() {
        let v = parse(r#"{"b": 1, "a": 2, "b": 3}"#).unwrap();
        assert_eq!(
            v,
            Json::Object(vec![
                ("b".into(), Json::Int("3".into())),
                ("a".into(), Json::Int("2".into())),
            ])
        );
    }

    #[test]
    fn integers_stay_exact_and_floats_are_python_floats() {
        let v =
            parse("[123456789012345678901234567890, -18446744073709551617, -0, 1.0, 1e2, 1E-400]")
                .unwrap();
        let Json::Array(a) = v else { panic!() };
        assert_eq!(a[0], Json::Int("123456789012345678901234567890".into()));
        assert_eq!(a[1], Json::Int("-18446744073709551617".into()));
        assert_eq!(a[2], Json::Int("0".into()));
        assert_eq!(a[3], Json::Float(1.0));
        assert_eq!(a[4], Json::Float(100.0));
        assert_eq!(a[5], Json::Float(0.0));
    }

    #[test]
    fn python_only_literals_are_refused_by_name() {
        for (text, want) in [
            ("NaN", "NaN"),
            ("[Infinity]", "Infinity"),
            (r#"{"a": -Infinity}"#, "-Infinity"),
        ] {
            match parse(text) {
                Err(Error::JsonNonFinite(w)) => assert_eq!(w, want),
                other => panic!("{text}: {other:?}"),
            }
        }
        assert!(matches!(parse("1e400"), Err(Error::JsonFloatRange(_))));
        assert!(matches!(
            parse(r#""\ud800""#),
            Err(Error::JsonLoneSurrogate(0xD800))
        ));
        assert!(matches!(
            parse(r#""\udc00x""#),
            Err(Error::JsonLoneSurrogate(0xDC00))
        ));
    }

    #[test]
    fn strings_decode_escapes_and_pairs() {
        assert_eq!(
            parse(r#""a\u0001\t\n\"\\\/😡é""#).unwrap(),
            Json::Str("a\u{1}\t\n\"\\/😡é".into())
        );
    }

    #[test]
    fn malformed_input_is_refused() {
        for text in [
            "",
            "01",
            "[1,]",
            "{\"a\" 1}",
            "\"a\u{1}\"",
            "1 2",
            "-",
            "1.",
            "\u{feff}1",
        ] {
            assert!(parse(text).is_err(), "{text:?} parsed");
        }
    }
}
