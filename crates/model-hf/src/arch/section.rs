//! One object of a checkpoint's `config.json` and the reads the maps make of it: a field that must
//! be there, a positive count, a positive finite number, an array. A refusal names the object's
//! path and the key, `text_config.num_experts must be a positive u32`, in the error type `E` the
//! caller reads its config in.

use std::marker::PhantomData;

use hf::schema::{self, Fields};
use serde_json::Value as Json;

/// The object `value` of `config.json`, named `path` in a refusal.
pub(super) struct Section<'a, E> {
    pub value: &'a Json,
    path: &'a str,
    error: PhantomData<fn() -> E>,
}

impl<'a, E: From<String>> Section<'a, E> {
    /// `value` as the object `path`; a value that is not an object is refused.
    pub fn new(value: &'a Json, path: &'a str) -> Result<Self, E> {
        if !value.is_object() {
            return Err(E::from(format!("{path} is not an object")));
        }
        Ok(Self {
            value,
            path,
            error: PhantomData,
        })
    }

    /// The refusal of `key` for `why`.
    pub fn bad(&self, key: &str, why: &str) -> E {
        E::from(format!("{}.{key} {why}", self.path))
    }

    /// The value of `key`, which must be there.
    pub fn field(&self, key: &str) -> Result<&'a Json, E> {
        self.value
            .get(key)
            .ok_or_else(|| self.bad(key, "is missing"))
    }

    /// `key` as a positive u32.
    pub fn count(&self, key: &str) -> Result<u32, E> {
        self.field(key)?
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .filter(|&n| n > 0)
            .ok_or_else(|| self.bad(key, "must be a positive u32"))
    }

    /// `key` as a finite positive f32.
    pub fn number(&self, key: &str) -> Result<f32, E> {
        self.field(key)?
            .as_f64()
            .map(|n| n as f32)
            .filter(|n| n.is_finite() && *n > 0.0)
            .ok_or_else(|| self.bad(key, "must be a finite positive f32"))
    }

    /// `key` as an array.
    pub fn array(&self, key: &str) -> Result<&'a [Json], E> {
        self.field(key)?
            .as_array()
            .map(Vec::as_slice)
            .ok_or_else(|| self.bad(key, "is not an array"))
    }

    /// The object checked against `tables` ([`schema::check`]): a key no table names, and a
    /// value its rule does not allow, are refused by name.
    pub fn check(&self, tables: &[&Fields]) -> Result<(), E> {
        schema::check(self.value, self.path, tables).map_err(E::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A refusal in the test's own error type: the message alone.
    #[derive(Debug, PartialEq)]
    struct Why(String);

    impl From<String> for Why {
        fn from(why: String) -> Why {
            Why(why)
        }
    }

    fn section(value: &Json) -> Section<'_, Why> {
        Section::new(value, "text").unwrap()
    }

    /// Each read takes the values it names and refuses every other by key.
    #[test]
    fn each_read_refuses_what_it_does_not_take() {
        let json: Json = serde_json::json!({
            "n": 7, "zero": 0, "big": 4294967296u64, "neg": -1, "x": 1.5,
            "neg_x": -0.5, "list": [1, 2], "text": "a"
        });
        let s = section(&json);
        assert_eq!(s.count("n"), Ok(7));
        assert_eq!(s.number("x"), Ok(1.5));
        assert_eq!(s.array("list").map(<[Json]>::len), Ok(2));
        for key in ["zero", "big", "neg", "x", "text"] {
            assert_eq!(
                s.count(key),
                Err(Why(format!("text.{key} must be a positive u32"))),
                "{key}"
            );
        }
        for key in ["zero", "neg_x", "text"] {
            assert_eq!(
                s.number(key),
                Err(Why(format!("text.{key} must be a finite positive f32"))),
                "{key}"
            );
        }
        assert_eq!(
            s.array("n").map(<[Json]>::len),
            Err(Why("text.n is not an array".into()))
        );
        assert_eq!(s.field("gone"), Err(Why("text.gone is missing".into())));
        assert_eq!(
            Section::<Why>::new(&Json::from(3), "config").err(),
            Some(Why("config is not an object".into()))
        );
    }

    /// A finite f64 that rounds to infinity as f32 is not a number the reader takes.
    #[test]
    fn a_number_that_is_infinite_as_f32_is_refused() {
        let json: Json = serde_json::json!({"huge": 1e300});
        assert_eq!(
            section(&json).number("huge"),
            Err(Why("text.huge must be a finite positive f32".into()))
        );
    }
}
