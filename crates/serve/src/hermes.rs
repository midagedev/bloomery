//! Hermes tool calls: the `<tool_call>` JSON markup the Qwen3 chat template
//! teaches (its `tools` header and the assistant `tool_calls` branch), parsed
//! out of the text after the think span.
//!
//! ```text
//! <tool_call>
//! {"name": "get_weather", "arguments": {"location": "Seoul"}}
//! </tool_call>
//! ```
//!
//! The template writes `arguments` as the request gave it: an object through
//! `tojson`, or a string's text verbatim — both spellings parse, the string's
//! body read as the object it holds. A call is emitted when its
//! `</tool_call>` arrives, its `arguments` the JSON object serialized
//! compactly in the model's key order. Text before, between and after calls
//! is content; llama-server's generated parser takes the text before the
//! first call as content too, and refuses a generation that goes on past its
//! calls, where this parser keeps that text. A call that does not follow the
//! markup — a body that is not one JSON object holding a `name` string and
//! an `arguments` object, or a `<tool_call>` still open when the generation
//! ends — is a [`HermesError`], never content and never dropped. Every
//! decision depends on the text only, never on how it was cut into pieces.

use serde_json::Value;

use crate::dsml::{Scanned, ToolCall};
use crate::reasoning::partial_suffix;

/// Opens a call.
pub const CALL_OPEN: &str = "<tool_call>";
/// Closes a call.
pub const CALL_CLOSE: &str = "</tool_call>";

/// How much of the offending text an error quotes.
const QUOTE: usize = 40;

/// A generation whose tool-call markup does not parse. `index` counts calls
/// in the generation from 0.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HermesError {
    /// The stream ended inside a call.
    #[error("tool call {index} is not closed by {CALL_CLOSE} when the generation ends: {text:?}")]
    Unterminated { index: usize, text: String },
    /// The body between the tags is not one JSON object.
    #[error("tool call {index} holds no JSON object: {found:?}")]
    NotJson { index: usize, found: String },
    /// No non-empty `name` string.
    #[error("tool call {index} names no function: {found:?}")]
    NoName { index: usize, found: String },
    /// `arguments` is neither a JSON object nor a string holding one.
    #[error("tool call {index} ({name}) carries no arguments object: {found:?}")]
    NoArguments {
        index: usize,
        name: String,
        found: String,
    },
    /// A key besides `name` and `arguments`.
    #[error("tool call {index} ({name}) carries the unknown key {key:?}")]
    UnknownKey {
        index: usize,
        name: String,
        key: String,
    },
}

/// Streaming scanner over the text after the think span.
#[derive(Debug, Default)]
pub struct HermesScan {
    /// Inside a call (its `<tool_call>` consumed).
    inside: bool,
    /// A call closed: only whitespace may follow before the next opens, and
    /// it is the markup's layout — consumed, never content, as the template
    /// writes its calls one newline apart.
    after_call: bool,
    /// Text not yet decided.
    buf: String,
    calls: usize,
}

impl HermesScan {
    /// Feeds text; returns the content that may be sent and the calls that
    /// closed.
    pub fn push(&mut self, piece: &str) -> Result<Scanned, HermesError> {
        self.buf.push_str(piece);
        let mut out = Scanned::default();
        loop {
            if self.inside {
                let Some(end) = self.buf.find(CALL_CLOSE) else {
                    return Ok(out);
                };
                let (name, arguments) = parse_call(&self.buf[..end], self.calls)?;
                out.calls.push(ToolCall {
                    index: self.calls,
                    name,
                    arguments,
                });
                self.calls += 1;
                self.buf.drain(..end + CALL_CLOSE.len());
                self.inside = false;
                self.after_call = true;
            } else if self.after_call {
                let ws = self.buf.len() - self.buf.trim_start().len();
                let rest = &self.buf[ws..];
                if rest.is_empty() || CALL_OPEN.starts_with(rest) {
                    return Ok(out);
                }
                if rest.starts_with(CALL_OPEN) {
                    self.buf.drain(..ws + CALL_OPEN.len());
                    self.inside = true;
                    self.after_call = false;
                } else {
                    // Text after the calls: it stays content, its leading
                    // whitespace with it.
                    self.after_call = false;
                }
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
    /// an error, an incomplete opening tag is text, and the whitespace a
    /// closed call's layout left is nothing.
    pub fn finish(&mut self) -> Result<Scanned, HermesError> {
        let rest = std::mem::take(&mut self.buf);
        if self.inside {
            return Err(HermesError::Unterminated {
                index: self.calls,
                text: format!("{CALL_OPEN}{rest}"),
            });
        }
        if self.after_call && rest.trim().is_empty() {
            return Ok(Scanned::default());
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
/// function name and its arguments as a JSON object's text. The body holds
/// `name` and `arguments` and nothing else, as the template writes a call;
/// anything else is named, never guessed into a call.
fn parse_call(body: &str, index: usize) -> Result<(String, String), HermesError> {
    let not_json = || HermesError::NotJson {
        index,
        found: quote(body.trim()),
    };
    let v: Value = serde_json::from_str(body.trim()).map_err(|_| not_json())?;
    let Value::Object(fields) = v else {
        return Err(not_json());
    };
    let name = fields
        .get("name")
        .and_then(Value::as_str)
        .filter(|n| !n.is_empty())
        .ok_or_else(|| HermesError::NoName {
            index,
            found: quote(body.trim()),
        })?
        .to_owned();
    let no_arguments = |found: String| HermesError::NoArguments {
        index,
        name: name.clone(),
        found,
    };
    let arguments: Value = match fields.get("arguments") {
        None => return Err(no_arguments("absent".to_owned())),
        Some(v @ Value::Object(_)) => v.clone(),
        Some(Value::String(s)) => {
            // The template writes a string the request gave verbatim: its
            // body is the object it holds.
            serde_json::from_str(s).map_err(|_| no_arguments(quote(s)))?
        }
        Some(other) => return Err(no_arguments(quote(&other.to_string()))),
    };
    if !arguments.is_object() {
        return Err(no_arguments(quote(&arguments.to_string())));
    }
    if let Some(key) = fields
        .keys()
        .find(|k| !matches!(k.as_str(), "name" | "arguments"))
    {
        return Err(HermesError::UnknownKey {
            index,
            name,
            key: key.clone(),
        });
    }
    Ok((name, arguments.to_string()))
}
