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
//! ends — is a [`HermesError`], never content and never dropped; the chat
//! parser reads a generation the token limit cut inside a call as an answer,
//! not as this error ([`crate::dsml::ChatParser::try_finish_at`]). The call a
//! scan has opened and not closed is reported as it is read
//! ([`Scanned::open`]): its name and `arguments` object once their start is
//! read, and the compact JSON of the object so far, a prefix of the arguments
//! the call closes with. Every decision depends on the text only, never on how
//! it was cut into pieces.

use serde_json::Value;

use crate::callstream::{self, Progress};
use crate::constrain::{Fun, Part, Refusal, call};
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

impl HermesError {
    /// The text of the call the generation ended inside, when that is the whole fault: markup that
    /// read as the template's so far and was not closed.
    pub(crate) fn unclosed(&self) -> Option<String> {
        match self {
            HermesError::Unterminated { text, .. } => Some(text.clone()),
            _ => None,
        }
    }
}

/// Streaming scanner over the text after the think span.
#[derive(Clone, Debug, Default)]
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
                    out.open = partial_call(&self.buf).map(|(name, args)| Progress {
                        index: self.calls,
                        name,
                        args,
                    });
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
                if rest.len() < CALL_OPEN.len() && CALL_OPEN.starts_with(rest) {
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
            ..Scanned::default()
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

/// The call at the start of `body`, the text after `<tool_call>` whose close
/// has not come, as far as `body` reads it: the name once it and the opening
/// brace of an `arguments` object that follows it are read (the template's
/// order; any other shape, an `arguments` string included, is read when the
/// call closes), and the arguments [`parse_call`] will close with, as a
/// prefix — [`compact_prefix`]. A tail that could still become `</tool_call>`
/// is not read.
fn partial_call(body: &str) -> Option<(String, String)> {
    let body = &body[..body.len() - partial_suffix(body, CALL_CLOSE)];
    let t = body.trim_start().strip_prefix('{')?.trim_start();
    let t = t
        .strip_prefix("\"name\"")?
        .trim_start()
        .strip_prefix(':')?
        .trim_start();
    let (name, t) = match string_prefix(t)? {
        Str::Whole(name, used) => (name, &t[used..]),
        Str::Open(_) => return None,
    };
    let t = t.trim_start().strip_prefix(',')?.trim_start();
    let t = t
        .strip_prefix("\"arguments\"")?
        .trim_start()
        .strip_prefix(':')?
        .trim_start();
    if name.is_empty() || !t.starts_with('{') {
        return None;
    }
    Some((name, compact_prefix(t)))
}

/// A JSON string at the start of `s` (its opening quote there), read as far as
/// `s` goes.
enum Str {
    /// The decoded text and the bytes of `s` it took, quotes included.
    Whole(String, usize),
    /// The text decoded so far; the closing quote has not come.
    Open(String),
}

/// The string at the start of `s`: `None` when `s` does not start one or the
/// string is not valid JSON (a raw control character, an escape JSON does not
/// have, a lone surrogate).
fn string_prefix(s: &str) -> Option<Str> {
    let rest = s.strip_prefix('"')?;
    let mut out = String::new();
    let mut at = 0;
    while let Some(c) = rest[at..].chars().next() {
        at += c.len_utf8();
        match c {
            '"' => return Some(Str::Whole(out, 1 + at)),
            '\\' => {
                let Some(e) = rest[at..].chars().next() else {
                    return Some(Str::Open(out));
                };
                at += e.len_utf8();
                out.push(match e {
                    '"' | '\\' | '/' => e,
                    'b' => '\u{8}',
                    'f' => '\u{c}',
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    'u' => match unit(&rest[at..])? {
                        None => return Some(Str::Open(out)),
                        Some(hi @ 0xD800..=0xDBFF) => {
                            let after = &rest[at + 4..];
                            let Some(low) = after.strip_prefix("\\u") else {
                                return if after.len() < 2 && "\\u".starts_with(after) {
                                    Some(Str::Open(out))
                                } else {
                                    None
                                };
                            };
                            match unit(low)? {
                                None => return Some(Str::Open(out)),
                                Some(lo @ 0xDC00..=0xDFFF) => {
                                    at += 4 + 2 + 4;
                                    char::from_u32(
                                        0x1_0000
                                            + ((u32::from(hi) - 0xD800) << 10)
                                            + (u32::from(lo) - 0xDC00),
                                    )?
                                }
                                Some(_) => return None,
                            }
                        }
                        Some(0xDC00..=0xDFFF) => return None,
                        Some(n) => {
                            at += 4;
                            char::from_u32(u32::from(n))?
                        }
                    },
                    _ => return None,
                });
            }
            c if u32::from(c) < 0x20 => return None,
            c => out.push(c),
        }
    }
    Some(Str::Open(out))
}

/// The UTF-16 unit that four hex digits at the start of `s` spell: `Ok(None)`
/// when fewer than four have come, `None` when they are not hex.
fn unit(s: &str) -> Option<Option<u16>> {
    if !s.bytes().take(4).all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    match s.get(..4) {
        None => Some(None),
        Some(d) => u16::from_str_radix(d, 16).ok().map(Some),
    }
}

/// The compact JSON of the value at the start of `s`, as far as `s` determines
/// it: structure as it comes, a string as it is decoded, in the escapes
/// `serde_json` writes, and a number or literal once a delimiter ends it. It
/// ends at the value's close, or where the text stops being JSON. What it
/// returns is the prefix of the text [`parse_call`] closes with (Mistral's
/// scan of the value after `[ARGS]` reads one too, [`crate::mistral`]).
pub(crate) fn compact_prefix(s: &str) -> String {
    let mut out = String::new();
    let mut depth = 0usize;
    let mut at = 0;
    while let Some(c) = s[at..].chars().next() {
        match c {
            c if c.is_ascii_whitespace() => at += 1,
            '{' | '[' => {
                out.push(c);
                depth += 1;
                at += 1;
            }
            '}' | ']' => {
                out.push(c);
                depth = depth.saturating_sub(1);
                at += 1;
                if depth == 0 {
                    break;
                }
            }
            ',' | ':' => {
                out.push(c);
                at += 1;
            }
            '"' => match string_prefix(&s[at..]) {
                Some(Str::Whole(text, used)) => {
                    out.push_str(&callstream::string(&text));
                    at += used;
                }
                Some(Str::Open(text)) => {
                    out.push('"');
                    out.push_str(&callstream::string_body(&text));
                    break;
                }
                None => break,
            },
            _ => {
                let Some(len) = s[at..]
                    .find(|d: char| d.is_ascii_whitespace() || matches!(d, ',' | ':' | '}' | ']'))
                else {
                    break;
                };
                match serde_json::from_str::<Value>(&s[at..at + len]) {
                    Ok(v @ (Value::Number(_) | Value::Bool(_) | Value::Null)) => {
                        out.push_str(&v.to_string());
                    }
                    _ => break,
                }
                at += len;
            }
        }
    }
    out
}

/// The call this markup spells, as the parts structured output forces: the
/// inverse of [`HermesScan`]. `{"name": NAME, "arguments": {…}}` between the
/// tags, the name a choice among `funs` when there are several, the arguments
/// one JSON object of the function's parameters.
pub(crate) fn call_parts(funs: &[Fun<'_>]) -> Result<Vec<Part>, Refusal> {
    call(
        funs,
        &format!("{CALL_OPEN}\n{{\"name\": \""),
        str::to_owned,
        |f| {
            let mut parts = vec![Part::Text("\", \"arguments\": ".to_owned())];
            parts.extend(f.object()?);
            parts.push(Part::Text(format!("}}\n{CALL_CLOSE}")));
            Ok(parts)
        },
    )
}
