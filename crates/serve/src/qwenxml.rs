//! Qwen3-Coder tool calls: the XML markup the Qwen3.6 and Qwen3.8 chat
//! templates teach (their `tools` header and the assistant `tool_calls`
//! branch), parsed out of the text after the think span.
//!
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=KEY>
//! VALUE
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! A value is the text between `<parameter=KEY>\n` and `\n</parameter>\n`,
//! byte for byte: a source file that starts indented or ends in a newline
//! keeps it. What the value is read as the request's tool schema says, the
//! way llama.cpp's parser reads it: a parameter whose schema can resolve to a
//! string ([`resolves_to_string`]) is the raw text; every other parameter
//! (another `type`, a `type` list without `string`, or no `type` at all) is
//! JSON, spelled in `arguments` as the model wrote it.
//!
//! A call is emitted when its `</tool_call>` arrives; the parameters become a
//! JSON object in the model's order, serialized compactly to a string. Text
//! before the first call, and after the calls once something other than the
//! layout follows, is content; the whitespace a closed call's layout left is
//! nothing, as the template writes its calls one newline apart. Markup that
//! does not parse — no function opener, no name, text where a tag belongs, a
//! function or a parameter the request's tools do not name, one parameter
//! twice, a value its schema types JSON that is not JSON — and a call still
//! open when the stream ends are a [`QwenXmlError`], never content and never
//! dropped. Every decision depends on the text only, never on how it was cut
//! into pieces.

use std::collections::HashMap;

use serde_json::Value;

use crate::dsml::{Scanned, ToolCall};
use crate::reasoning::partial_suffix;

/// Opens a call.
pub const CALL_OPEN: &str = "<tool_call>";
/// Closes a call.
pub const CALL_CLOSE: &str = "</tool_call>";
/// Opens the call's function, its name to the `>` (`<function=NAME>`).
pub const FUNCTION_OPEN: &str = "<function=";
/// Closes the function.
pub const FUNCTION_CLOSE: &str = "</function>";
/// Opens a parameter, its name to the `>` (`<parameter=KEY>`).
pub const PARAM_OPEN: &str = "<parameter=";
/// Ends a value: the newline before the tag and the one after it are the
/// markup's own.
const PARAM_VALUE_CLOSE: &str = "\n</parameter>\n";

/// How much of the offending text an error quotes.
const QUOTE: usize = 40;

/// A generation whose tool-call markup does not parse. `index` counts calls
/// in the generation from 0.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum QwenXmlError {
    /// The stream ended inside a call.
    #[error("tool call {index} is not closed by {CALL_CLOSE} when the generation ends: {text:?}")]
    Unterminated { index: usize, text: String },
    /// `<tool_call>` not followed by `<function=NAME>`.
    #[error("tool call {index} opens no function: {found:?}")]
    NoFunction { index: usize, found: String },
    /// `<function=` with no name before its `>`.
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
    /// A function the request's tools do not name.
    #[error("tool call {index}: the request's tools name no function {name}")]
    UnknownFunction { index: usize, name: String },
    /// A parameter the request's tools do not name.
    #[error("tool call {index} ({name}): the request's tools name no parameter {key}")]
    UnknownParameter {
        index: usize,
        name: String,
        key: String,
    },
    /// One parameter twice in a call.
    #[error("tool call {index} ({name}): parameter {key} given twice")]
    DuplicateParameter {
        index: usize,
        name: String,
        key: String,
    },
    /// A value that is not JSON where the schema typed it JSON.
    #[error(
        "tool call {index} ({name}): parameter {key} is typed {ty}, and its value is not JSON: {value:?}"
    )]
    NotJson {
        index: usize,
        name: String,
        key: String,
        /// The schema's own spelling of the type, `none` when it wrote none.
        ty: String,
        value: String,
    },
}

/// What a parameter's schema says of its value.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Kind {
    /// The schema can resolve to a string: the value is the raw text.
    String,
    /// Every other schema: the value must be JSON. The type as written, for
    /// the error.
    Json(String),
}

/// Parameter kinds by function and parameter name, from a request's `tools`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParamKinds {
    kinds: HashMap<String, HashMap<String, Kind>>,
}

impl ParamKinds {
    /// Reads OpenAI `tools` (each `{"type": "function", "function": {...}}` or
    /// the function object itself, as the template accepts both): every
    /// `parameters.properties.<key>`'s schema as a [`Kind`]. A function or a
    /// parameter the tools do not name is left out, and a call that writes it
    /// meets [`QwenXmlError::UnknownFunction`] or
    /// [`QwenXmlError::UnknownParameter`].
    #[must_use]
    pub fn of_tools(tools: Option<&Value>) -> ParamKinds {
        let mut kinds: HashMap<String, HashMap<String, Kind>> = HashMap::new();
        for tool in tools.and_then(Value::as_array).into_iter().flatten() {
            let f = tool.get("function").unwrap_or(tool);
            let Some(name) = f.get("name").and_then(Value::as_str) else {
                continue;
            };
            let root = f.get("parameters").unwrap_or(&Value::Null);
            let props = root.get("properties").and_then(Value::as_object);
            let args = kinds.entry(name.to_owned()).or_default();
            for (key, schema) in props.into_iter().flatten() {
                args.insert(key.clone(), kind_of(schema, root));
            }
        }
        ParamKinds { kinds }
    }

    fn names(&self, name: &str) -> bool {
        self.kinds.contains_key(name)
    }

    fn kind(&self, name: &str, key: &str) -> Option<&Kind> {
        self.kinds.get(name)?.get(key)
    }
}

/// Whether a parameter's schema lets its value be a string, read the way
/// llama.cpp's `common_schema_info::resolves_to_string` reads it: a `type`
/// that names `string` (alone or in a list), an `anyOf` or `oneOf` branch
/// that does, an `allOf` whose components all do, a string `const` or `enum`
/// member, a string keyword (`pattern`, `minLength`, `maxLength`) or a string
/// `format`. A `$ref` into the tool's `parameters` (`#/…`) is followed, each
/// once per parameter; any other `$ref`, or one met again, is not a string.
fn resolves_to_string(schema: &Value, root: &Value, seen: &mut Vec<String>) -> bool {
    let Some(s) = schema.as_object() else {
        return false;
    };
    if let Some(r) = s.get("$ref") {
        let Some(r) = r.as_str() else {
            return false;
        };
        if seen.iter().any(|x| x == r) {
            return false;
        }
        seen.push(r.to_owned());
        return r
            .strip_prefix('#')
            .and_then(|path| root.pointer(path))
            .is_some_and(|t| resolves_to_string(t, root, seen));
    }
    match s.get("type") {
        Some(Value::String(t)) if t == "string" => return true,
        Some(Value::Array(a)) if a.iter().any(|t| t.as_str() == Some("string")) => return true,
        _ => {}
    }
    for k in ["oneOf", "anyOf"] {
        if let Some(alts) = s.get(k).and_then(Value::as_array)
            && alts.iter().any(|alt| resolves_to_string(alt, root, seen))
        {
            return true;
        }
    }
    if let Some(all) = s.get("allOf").and_then(Value::as_array)
        && all.iter().all(|c| resolves_to_string(c, root, seen))
    {
        return true;
    }
    if s.get("const").is_some_and(Value::is_string)
        || s.get("enum")
            .and_then(Value::as_array)
            .is_some_and(|e| e.iter().any(Value::is_string))
        || ["pattern", "minLength", "maxLength"]
            .iter()
            .any(|k| s.contains_key(*k))
    {
        return true;
    }
    s.get("format")
        .and_then(Value::as_str)
        .is_some_and(|f| STRING_FORMATS.contains(&f) || f.starts_with("uuid"))
}

/// The `format`s llama.cpp reads as a string.
const STRING_FORMATS: [&str; 8] = [
    "date",
    "time",
    "date-time",
    "uri",
    "email",
    "hostname",
    "ipv4",
    "ipv6",
];

/// One parameter's schema as a [`Kind`]: the raw text when the schema can
/// resolve to a string, else JSON — the type as the schema wrote it, `none`
/// when it wrote none. `root` is the tool's `parameters`, which a `$ref`
/// points into.
fn kind_of(schema: &Value, root: &Value) -> Kind {
    if resolves_to_string(schema, root, &mut Vec::new()) {
        return Kind::String;
    }
    let ty = match schema.get("type") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join("|"),
        _ => "none".to_owned(),
    };
    Kind::Json(ty)
}

/// Streaming scanner over the text after the think span.
#[derive(Debug)]
pub struct QwenScan {
    kinds: ParamKinds,
    /// Inside a call (its `<tool_call>` consumed).
    inside: bool,
    /// A call closed: only the layout between calls may follow, and it is
    /// consumed, never content, as the template writes its calls one newline
    /// apart.
    after_call: bool,
    /// Text not yet decided.
    buf: String,
    calls: usize,
}

impl QwenScan {
    /// A scanner that types parameters by `kinds`.
    #[must_use]
    pub fn new(kinds: ParamKinds) -> Self {
        QwenScan {
            kinds,
            inside: false,
            after_call: false,
            buf: String::new(),
            calls: 0,
        }
    }

    /// Feeds text; returns the content that may be sent and the calls that
    /// closed.
    pub fn push(&mut self, piece: &str) -> Result<Scanned, QwenXmlError> {
        self.buf.push_str(piece);
        let mut out = Scanned::default();
        loop {
            if self.inside {
                let Some((name, arguments, used)) =
                    parse_call(&self.buf, self.calls, &self.kinds, false)?
                else {
                    return Ok(out);
                };
                out.calls.push(ToolCall {
                    index: self.calls,
                    name,
                    arguments,
                });
                self.calls += 1;
                self.buf.drain(..used);
                self.inside = false;
                self.after_call = true;
            } else if self.after_call {
                let ws = self.buf.len() - self.buf.trim_start().len();
                let rest = &self.buf[ws..];
                if held(CALL_OPEN, rest) {
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
    /// an error (the markup error its text already shows, else
    /// [`QwenXmlError::Unterminated`]), an incomplete opening tag is text,
    /// and the whitespace a closed call's layout left is nothing.
    pub fn finish(&mut self) -> Result<Scanned, QwenXmlError> {
        if self.inside {
            parse_call(&self.buf, self.calls, &self.kinds, true)?;
        }
        let rest = std::mem::take(&mut self.buf);
        if self.inside {
            return Err(QwenXmlError::Unterminated {
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

/// Whether `rest` is a proper prefix of `tag` (the empty text included): the
/// text so far can still grow into the tag, so nothing is decided yet.
fn held(tag: &str, rest: &str) -> bool {
    rest.len() < tag.len() && tag.starts_with(rest)
}

/// One call, the text after `<tool_call>`, at the start of `s`: its function
/// name, its parameters as a JSON object's text, and the bytes it took
/// through `</tool_call>`. `Ok(None)` when the text so far can only be the
/// start of a call. An error that quotes the text after the fault waits for
/// a full quote, or for the stream's `end`, so its message does not depend
/// on where the stream was cut. Whitespace between the tags is skipped (the template
/// writes a newline after each); the newline that opens a value is not
/// whitespace between tags but the value's own boundary.
fn parse_call(
    s: &str,
    index: usize,
    kinds: &ParamKinds,
    end: bool,
) -> Result<Option<(String, String, usize)>, QwenXmlError> {
    let settled = |found: &str| end || found.chars().nth(QUOTE - 1).is_some();
    let lead = s.len() - s.trim_start().len();
    let rest = &s[lead..];
    if held(FUNCTION_OPEN, rest) {
        return Ok(None);
    }
    let Some(body) = rest.strip_prefix(FUNCTION_OPEN) else {
        if !settled(rest) {
            return Ok(None);
        }
        return Err(QwenXmlError::NoFunction {
            index,
            found: quote(rest),
        });
    };
    let Some(name_end) = body.find('>') else {
        return Ok(None);
    };
    let name = &body[..name_end];
    if name.is_empty() || name.contains(['<', '\n', '"']) {
        return Err(QwenXmlError::NoName {
            index,
            found: quote(&rest[..FUNCTION_OPEN.len() + name_end + 1]),
        });
    }
    if !kinds.names(name) {
        return Err(QwenXmlError::UnknownFunction {
            index,
            name: name.to_owned(),
        });
    }
    let unexpected = |expected, found: &str| QwenXmlError::Unexpected {
        index,
        name: name.to_owned(),
        expected,
        found: quote(found),
    };
    let mut params: Vec<(&str, String)> = Vec::new();
    let mut at = lead + FUNCTION_OPEN.len() + name_end + 1;
    loop {
        let tail = &s[at..];
        let ws = tail.len() - tail.trim_start().len();
        let rest = &tail[ws..];
        if held(PARAM_OPEN, rest) || held(FUNCTION_CLOSE, rest) {
            return Ok(None);
        }
        if let Some(after) = rest.strip_prefix(FUNCTION_CLOSE) {
            let ws2 = after.len() - after.trim_start().len();
            let rest = &after[ws2..];
            if held(CALL_CLOSE, rest) {
                return Ok(None);
            }
            if !rest.starts_with(CALL_CLOSE) {
                if !settled(rest) {
                    return Ok(None);
                }
                return Err(unexpected(CALL_CLOSE, rest));
            }
            at += ws + FUNCTION_CLOSE.len() + ws2 + CALL_CLOSE.len();
            break;
        }
        let Some(p) = rest.strip_prefix(PARAM_OPEN) else {
            if !settled(rest) {
                return Ok(None);
            }
            return Err(unexpected("a parameter or the function's close", rest));
        };
        let Some(key_end) = p.find('>') else {
            return Ok(None);
        };
        let key = &p[..key_end];
        if key.is_empty() || key.contains(['<', '\n', '"']) {
            return Err(unexpected(
                "a parameter name after <parameter=",
                &rest[..PARAM_OPEN.len() + key_end + 1],
            ));
        }
        let open = &p[key_end + 1..];
        let Some(v) = open.strip_prefix('\n') else {
            if open.is_empty() || !settled(rest) {
                return Ok(None);
            }
            return Err(unexpected("a newline after <parameter=KEY>", rest));
        };
        let Some(val_end) = v.find(PARAM_VALUE_CLOSE) else {
            return Ok(None);
        };
        let raw = &v[..val_end];
        if params.iter().any(|(k, _)| *k == key) {
            return Err(QwenXmlError::DuplicateParameter {
                index,
                name: name.to_owned(),
                key: key.to_owned(),
            });
        }
        let value = match kinds.kind(name, key) {
            Some(Kind::String) => Value::String(raw.to_owned()).to_string(),
            Some(Kind::Json(ty)) => match serde_json::from_str::<Value>(raw) {
                // The value as the model wrote it, as llama-server writes a
                // typed value back; a re-serialization would respell it.
                Ok(_) => raw.to_owned(),
                Err(_) => {
                    return Err(QwenXmlError::NotJson {
                        index,
                        name: name.to_owned(),
                        key: key.to_owned(),
                        ty: ty.clone(),
                        value: quote(raw),
                    });
                }
            },
            None => {
                return Err(QwenXmlError::UnknownParameter {
                    index,
                    name: name.to_owned(),
                    key: key.to_owned(),
                });
            }
        };
        params.push((key, value));
        at += ws + PARAM_OPEN.len() + key_end + 1 + 1 + val_end + PARAM_VALUE_CLOSE.len();
    }
    let fields: Vec<String> = params
        .iter()
        .map(|(k, v)| format!("{}:{v}", Value::String((*k).to_owned())))
        .collect();
    Ok(Some((
        name.to_owned(),
        format!("{{{}}}", fields.join(",")),
        at,
    )))
}
