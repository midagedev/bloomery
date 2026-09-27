//! GLM tool calls: the markup the GLM-5.3-Flash chat template teaches (its
//! `tools` header and the assistant `tool_calls` branch), parsed out of the
//! text after the think span.
//!
//! ```text
//! <tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>…</tool_call>
//! ```
//!
//! The template writes a value `v | tojson(ensure_ascii=False) if v is not
//! string else v`: a string raw, anything else as JSON. The request's tool
//! schema says which a value is: an argument whose schema `type` is
//! `"string"` is the raw text, byte for byte; one whose `type` names other
//! types only must be JSON; for any other argument (no schema, no `type`, or
//! a `type` list with `"string"` among others) a value that parses as JSON is
//! that JSON and anything else is the raw text, the only thing the template
//! writes unquoted. Whitespace between tags is skipped (GLM-4.6's template
//! writes newlines there); a value is everything up to its `</arg_value>`,
//! a `<` included.
//!
//! A call is emitted when its `</tool_call>` arrives; the arguments become a
//! JSON object in argument order, serialized to a string. Text before, between
//! and after calls is content. A call that does not follow the markup (no
//! name, text where a tag belongs, `<arg_value>` without its `<arg_key>`, a
//! key given twice, a value its schema types as JSON that is not JSON) and a
//! call still open when the stream ends are a [`GlmXmlError`], never content
//! and never dropped. Every decision depends on the text only, never on how it
//! was cut into pieces.

use std::collections::HashMap;

use serde_json::Value;

use crate::dsml::{Scanned, ToolCall};
use crate::reasoning::partial_suffix;

/// Opens a call.
pub const CALL_OPEN: &str = "<tool_call>";
/// Closes a call.
pub const CALL_CLOSE: &str = "</tool_call>";
/// Opens an argument's name.
pub const KEY_OPEN: &str = "<arg_key>";
/// Closes an argument's name.
pub const KEY_CLOSE: &str = "</arg_key>";
/// Opens an argument's value.
pub const VALUE_OPEN: &str = "<arg_value>";
/// Closes an argument's value.
pub const VALUE_CLOSE: &str = "</arg_value>";

/// How much of the offending text an error quotes.
const QUOTE: usize = 40;

/// A generation whose tool-call markup does not parse. `index` counts calls
/// in the generation from 0.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GlmXmlError {
    /// The stream ended inside a call.
    #[error("tool call {index} is not closed by {CALL_CLOSE} when the generation ends: {text:?}")]
    Unterminated { index: usize, text: String },
    /// `<tool_call>` not followed by a function name.
    #[error("tool call {index} names no function before {found:?}")]
    NoName { index: usize, found: String },
    /// Something else where the markup has a tag.
    #[error("tool call {index} ({name}): expected {expected}, found {found:?}")]
    Unexpected {
        index: usize,
        name: String,
        expected: &'static str,
        found: String,
    },
    /// One argument name twice in a call.
    #[error("tool call {index} ({name}): argument {key} given twice")]
    DuplicateKey {
        index: usize,
        name: String,
        key: String,
    },
    /// A value whose schema type is not a string, and which is not JSON.
    #[error(
        "tool call {index} ({name}): argument {key} has schema type {ty}, and its value is not \
         JSON: {value:?}"
    )]
    NotJson {
        index: usize,
        name: String,
        key: String,
        ty: String,
        value: String,
    },
}

/// What a tool schema says of one argument.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Kind {
    /// `"type": "string"`: the value is raw text.
    String,
    /// A `type` without `"string"`: the value is JSON; the type as written.
    Json(String),
}

/// Argument kinds by function and argument name, from a request's `tools`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArgTypes {
    kinds: HashMap<String, HashMap<String, Kind>>,
}

impl ArgTypes {
    /// Reads OpenAI `tools` (each `{"type": "function", "function": {...}}` or
    /// the function object itself, as the template accepts both): every
    /// `parameters.properties.<key>.type`. Anything else in the list is left
    /// out, and its arguments fall to the JSON-else-text rule.
    #[must_use]
    pub fn of_tools(tools: Option<&Value>) -> ArgTypes {
        let mut kinds: HashMap<String, HashMap<String, Kind>> = HashMap::new();
        for tool in tools.and_then(Value::as_array).into_iter().flatten() {
            let f = tool.get("function").unwrap_or(tool);
            let Some(name) = f.get("name").and_then(Value::as_str) else {
                continue;
            };
            let props = f
                .get("parameters")
                .and_then(|p| p.get("properties"))
                .and_then(Value::as_object);
            let args = kinds.entry(name.to_owned()).or_default();
            for (key, schema) in props.into_iter().flatten() {
                if let Some(kind) = schema.get("type").and_then(kind_of) {
                    args.insert(key.clone(), kind);
                }
            }
        }
        ArgTypes { kinds }
    }

    fn kind(&self, name: &str, key: &str) -> Option<&Kind> {
        self.kinds.get(name)?.get(key)
    }
}

/// A schema `type`: `"string"` is text; a type, or a list of types, without
/// `"string"` is JSON; a list with `"string"` among others says nothing.
fn kind_of(ty: &Value) -> Option<Kind> {
    match ty {
        Value::String(s) if s == "string" => Some(Kind::String),
        Value::String(s) => Some(Kind::Json(s.clone())),
        Value::Array(a) => {
            let names: Vec<&str> = a.iter().filter_map(Value::as_str).collect();
            (names.len() == a.len() && !names.contains(&"string"))
                .then(|| Kind::Json(names.join("|")))
        }
        _ => None,
    }
}

/// Streaming scanner over the text after the think span.
#[derive(Debug)]
pub struct GlmScan {
    types: ArgTypes,
    /// Inside a call (its `<tool_call>` consumed).
    inside: bool,
    /// Text not yet decided.
    buf: String,
    calls: usize,
}

impl GlmScan {
    /// A scanner that types arguments by `types`.
    #[must_use]
    pub fn new(types: ArgTypes) -> Self {
        GlmScan {
            types,
            inside: false,
            buf: String::new(),
            calls: 0,
        }
    }

    /// Feeds text; returns the content that may be sent and the calls that
    /// closed.
    pub fn push(&mut self, piece: &str) -> Result<Scanned, GlmXmlError> {
        self.buf.push_str(piece);
        let mut out = Scanned::default();
        loop {
            if self.inside {
                let Some(end) = self.buf.find(CALL_CLOSE) else {
                    return Ok(out);
                };
                let (name, arguments) = parse_call(&self.buf[..end], self.calls, &self.types)?;
                out.calls.push(ToolCall {
                    index: self.calls,
                    name,
                    arguments,
                });
                self.calls += 1;
                self.buf.drain(..end + CALL_CLOSE.len());
                self.inside = false;
            } else if let Some(at) = self.buf.find(CALL_OPEN) {
                out.content.push_str(&self.buf[..at]);
                self.buf.drain(..at + CALL_OPEN.len());
                self.inside = true;
            } else {
                let upto = self.buf.len() - partial_suffix(&self.buf, CALL_OPEN);
                out.content.push_str(&self.buf[..upto]);
                self.buf.drain(..upto);
                return Ok(out);
            }
        }
    }

    /// Releases what is held at the end of the stream; a call still open is
    /// an error.
    pub fn finish(&mut self) -> Result<Scanned, GlmXmlError> {
        let rest = std::mem::take(&mut self.buf);
        if self.inside {
            return Err(GlmXmlError::Unterminated {
                index: self.calls,
                text: format!("{CALL_OPEN}{rest}"),
            });
        }
        Ok(Scanned {
            content: rest,
            calls: Vec::new(),
        })
    }
}

/// The first characters of `s`, for an error.
fn quote(s: &str) -> String {
    s.chars().take(QUOTE).collect()
}

/// One call's body, the text between `<tool_call>` and `</tool_call>`: its
/// function name and its arguments as a JSON object's text.
fn parse_call(body: &str, index: usize, types: &ArgTypes) -> Result<(String, String), GlmXmlError> {
    let name_end = body.find('<').unwrap_or(body.len());
    let name = body[..name_end].trim();
    if name.is_empty() {
        return Err(GlmXmlError::NoName {
            index,
            found: quote(body),
        });
    }
    let unexpected = |expected, found: &str| GlmXmlError::Unexpected {
        index,
        name: name.to_owned(),
        expected,
        found: quote(found),
    };
    let mut args: Vec<(String, Value)> = Vec::new();
    let mut rest = &body[name_end..];
    loop {
        rest = rest.trim_start();
        if rest.is_empty() {
            break;
        }
        let Some(k) = rest.strip_prefix(KEY_OPEN) else {
            return Err(unexpected("<arg_key> or </tool_call>", rest));
        };
        let Some(key_end) = k.find(KEY_CLOSE) else {
            return Err(unexpected("</arg_key>", k));
        };
        let key = &k[..key_end];
        let v = k[key_end + KEY_CLOSE.len()..].trim_start();
        let Some(v) = v.strip_prefix(VALUE_OPEN) else {
            return Err(unexpected("<arg_value>", v));
        };
        let Some(value_end) = v.find(VALUE_CLOSE) else {
            return Err(unexpected("</arg_value>", v));
        };
        if args.iter().any(|(a, _)| a == key) {
            return Err(GlmXmlError::DuplicateKey {
                index,
                name: name.to_owned(),
                key: key.to_owned(),
            });
        }
        let raw = &v[..value_end];
        let value = match types.kind(name, key) {
            Some(Kind::String) => Value::String(raw.to_owned()),
            Some(Kind::Json(ty)) => {
                serde_json::from_str(raw).map_err(|_| GlmXmlError::NotJson {
                    index,
                    name: name.to_owned(),
                    key: key.to_owned(),
                    ty: ty.clone(),
                    value: quote(raw),
                })?
            }
            None => serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned())),
        };
        args.push((key.to_owned(), value));
        rest = &v[value_end + VALUE_CLOSE.len()..];
    }
    let fields: Vec<String> = args
        .iter()
        .map(|(k, v)| format!("{}:{v}", Value::String(k.clone())))
        .collect();
    Ok((name.to_owned(), format!("{{{}}}", fields.join(","))))
}
