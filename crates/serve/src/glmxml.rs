//! Key/value tool calls: the markup the GLM-5.3-Flash and Solar-Open2 chat
//! templates teach (their `tools` header and the assistant `tool_calls`
//! branch), parsed out of the text after the think span. Both spell one
//! grammar, each under its own markers ([`KvSpelling`]):
//!
//! ```text
//! CALL_OPEN NAME (KEY_OPEN KEY KEY_CLOSE [VALUE_OPEN] VALUE VALUE_CLOSE)* CALL_CLOSE
//!
//! GLM    <tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>…</tool_call>
//! Solar  <|tool_call:start|>NAME\n<|tool_arg:start|>KEY<|tool_arg:value|>VALUE<|tool_arg:end|>\n…<|tool_call:end|>
//! ```
//!
//! Solar's key close is also what opens its value. The templates write a value
//! as JSON unless it is a string, which they write raw. The request's tool
//! schema says which a value is: an argument whose schema `type` is
//! `"string"` is the raw text, byte for byte; one whose `type` names other
//! types only must be JSON; for any other argument (no schema, no `type`, or
//! a `type` list with `"string"` among others) a value that parses as JSON is
//! that JSON and anything else is the raw text, the only thing the templates
//! write unquoted. Whitespace between tags is skipped (GLM-4.6's template
//! writes newlines there, Solar's one after the name and after each
//! argument); a value is everything up to its close, a `<` included.
//!
//! A call is emitted when its close arrives; the arguments become a JSON
//! object in argument order, serialized to a string. Text before, between and
//! after calls is content, except the line feed a spelling writes after a
//! call. A call that does not follow the markup (no name, text where a tag
//! belongs, a value without its key, a key given twice, a value its schema
//! types as JSON that is not JSON) and a call still open when the stream ends
//! are a [`KvError`], never content and never dropped; the chat parser reads a
//! generation the token limit cut inside a call as an answer, not as this error
//! ([`crate::dsml::ChatParser::try_finish_at`]), unless an argument it completed
//! is already wrong (a key given twice, a value not JSON). The call a scan has
//! opened and not closed is reported as it is read ([`Scanned::open`]): its
//! name once a tag follows it, and the arguments so far, a prefix of the
//! arguments the call closes with, the text of a `"string"` argument as it
//! arrives. The chat parser's scan reads only the text each push adds
//! (`KvScan::push_stream`) and reports only the arguments it adds. Every
//! decision is independent of how the text and special-token offsets were cut into pieces.

use std::collections::HashMap;

use serde_json::Value;

use crate::callstream::{Args, Reading, find_from, held, read_value};
use crate::constrain::{
    Arg, Call, CallMode, Fun, Part, Refusal, Style, call, is_text, object_parameters, outside,
    value,
};
use crate::dsml::{Scanned, ToolCall};
use crate::engine::Tokenizer;
use crate::reasoning::{Marked, SpecialTags, partial_suffix};

/// How a template spells the key/value markup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvSpelling {
    /// GLM's: `<tool_call>NAME<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`.
    Glm,
    /// Solar-Open's: `<|tool_call:start|>NAME` and a line feed, then
    /// `<|tool_arg:start|>KEY<|tool_arg:value|>VALUE<|tool_arg:end|>` and a line
    /// feed per argument, then `<|tool_call:end|>`.
    Solar,
}

/// The markers of one [`KvSpelling`].
#[derive(Debug)]
struct Markers {
    /// The family a refusal names.
    family: &'static str,
    /// Opens a call.
    call_open: &'static str,
    /// Closes a call.
    call_close: &'static str,
    /// Opens an argument's name.
    key_open: &'static str,
    /// Closes an argument's name.
    key_close: &'static str,
    /// Opens an argument's value after the name's close; empty when the
    /// name's close opens it.
    value_open: &'static str,
    /// Closes an argument's value.
    value_close: &'static str,
    /// Whether the template writes a line feed after the name, after each
    /// argument and after the call.
    newline: bool,
}

static GLM: Markers = Markers {
    family: "GLM",
    call_open: "<tool_call>",
    call_close: "</tool_call>",
    key_open: "<arg_key>",
    key_close: "</arg_key>",
    value_open: "<arg_value>",
    value_close: "</arg_value>",
    newline: false,
};

static SOLAR: Markers = Markers {
    family: "Solar",
    call_open: "<|tool_call:start|>",
    call_close: "<|tool_call:end|>",
    key_open: "<|tool_arg:start|>",
    key_close: "<|tool_arg:value|>",
    value_open: "",
    value_close: "<|tool_arg:end|>",
    newline: true,
};

impl Markers {
    /// The line break the template writes where [`Markers::newline`] says.
    fn line(&self) -> &'static str {
        if self.newline { "\n" } else { "" }
    }

    /// The tags a template spells to open a call, an argument's name and its
    /// value.
    fn openers(&self) -> [&'static str; 3] {
        let value = if self.value_open.is_empty() {
            self.key_close
        } else {
            self.value_open
        };
        [self.call_open, self.key_open, value]
    }

    /// An argument's name and the text after its close, from `k`, the text
    /// after the name's opening; `None` when the close is missing or a tag
    /// opens before it.
    fn split_key<'a>(&self, k: &'a str) -> Option<(&'a str, &'a str)> {
        let (key, after) = k.split_once(self.key_close)?;
        (!key.contains('<')).then_some((key, after))
    }

    /// The text from an argument's value on, `after` being the text after its
    /// name's close; `None` when the value's opening is missing.
    fn value_start<'a>(&self, after: &'a str) -> Option<&'a str> {
        if self.value_open.is_empty() {
            Some(after)
        } else {
            after.trim_start().strip_prefix(self.value_open)
        }
    }
}

impl KvSpelling {
    const ALL: [KvSpelling; 2] = [KvSpelling::Glm, KvSpelling::Solar];

    fn markers(self) -> &'static Markers {
        match self {
            KvSpelling::Glm => &GLM,
            KvSpelling::Solar => &SOLAR,
        }
    }

    /// The spelling a template `source` writes: the one whose call, name and
    /// value openers it all spells; `None` when it spells none.
    #[must_use]
    pub fn of_template(source: &str) -> Option<KvSpelling> {
        KvSpelling::ALL
            .into_iter()
            .find(|s| s.markers().openers().iter().all(|tag| source.contains(tag)))
    }
}

/// How much of the offending text an error quotes.
const QUOTE: usize = 40;

/// A generation whose tool-call markup does not parse. `index` counts calls
/// in the generation from 0.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum KvError {
    /// The stream ended inside a call.
    #[error("tool call {index} is not closed by {close} when the generation ends: {text:?}")]
    Unterminated {
        index: usize,
        close: &'static str,
        text: String,
    },
    /// A call's opening not followed by a function name.
    #[error("tool call {index} names no function before {found:?}")]
    NoName { index: usize, found: String },
    /// Something else where the markup has a tag.
    #[error("tool call {index} ({name}): expected {expected}, found {found:?}")]
    Unexpected {
        index: usize,
        name: String,
        expected: String,
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

impl KvError {
    /// The text of the call the generation ended inside, when that is the whole fault: markup that
    /// read as the template's so far and was not closed.
    pub(crate) fn unclosed(&self) -> Option<String> {
        match self {
            KvError::Unterminated { text, .. } => Some(text.clone()),
            _ => None,
        }
    }
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
    /// `parameters.properties.<key>.type`, and a `$ref` that resolves to a
    /// string as text. Anything else in the list is left out, and its
    /// arguments fall to the JSON-else-text rule.
    #[must_use]
    pub fn of_tools(tools: Option<&Value>) -> ArgTypes {
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
                let kind = if schema.get("$ref").is_some()
                    && crate::qwenxml::resolves_to_string(schema, root, &mut Vec::new())
                {
                    Some(Kind::String)
                } else {
                    schema.get("type").and_then(kind_of)
                };
                if let Some(kind) = kind {
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
#[derive(Clone, Debug)]
pub struct KvScan {
    markers: &'static Markers,
    types: ArgTypes,
    /// Inside a call (its opening consumed).
    inside: bool,
    /// A call has just closed and the line feed its spelling writes after it
    /// has not been settled.
    line_due: bool,
    /// Text not yet decided.
    buf: Marked,
    specials: SpecialTags,
    calls: usize,
    /// The call open at the start of `buf`, as far as it is read.
    read: CallRead,
}

impl KvScan {
    /// A scanner for `spelling`'s markup that types arguments by `types`.
    #[must_use]
    pub fn new(spelling: KvSpelling, types: ArgTypes) -> Self {
        KvScan {
            markers: spelling.markers(),
            types,
            inside: false,
            line_due: false,
            buf: Marked::default(),
            specials: SpecialTags::default(),
            calls: 0,
            read: CallRead::default(),
        }
    }

    /// Feeds text; returns the content that may be sent and the calls that
    /// closed.
    pub fn push(&mut self, piece: &str) -> Result<Scanned, KvError> {
        Ok(self
            .push_stream(piece, &[])?
            .whole(&self.read.call, self.calls))
    }

    /// Reads call tags only where their special tokens wrote them, for those
    /// `tok` holds as one special token.
    pub(crate) fn read_specials(&mut self, tok: &dyn Tokenizer) {
        self.specials = SpecialTags::of(tok, &[self.markers.call_open, self.markers.call_close]);
    }

    /// [`KvScan::push`] that reads only the text `piece` adds and reports the
    /// open call's new arguments as `advances` instead of its whole text;
    /// `specials` gives the byte offset and length of each special token's
    /// text in `piece`.
    pub(crate) fn push_stream(
        &mut self,
        piece: &str,
        specials: &[(usize, usize)],
    ) -> Result<Scanned, KvError> {
        let m = self.markers;
        self.buf.push(piece, specials);
        let mut out = Scanned::default();
        let advances = out.advances.insert(Vec::new());
        loop {
            if self.inside {
                let end = loop {
                    let Some(at) = find_from(&self.buf.text, &mut self.read.closing, m.call_close)
                    else {
                        break None;
                    };
                    if self.specials.at(&self.buf, at, m.call_close) {
                        break Some(at);
                    }
                    self.read.closing = at + m.call_close.len();
                };
                let Some(end) = end else {
                    self.read.advance(m, &self.buf.text, &self.types);
                    self.read.call.send(self.calls, advances);
                    return Ok(out);
                };
                let (name, arguments) =
                    parse_call(m, &self.buf.text[..end], self.calls, &self.types)?;
                let call = ToolCall {
                    index: self.calls,
                    name,
                    arguments,
                };
                std::mem::take(&mut self.read).call.close(&call, advances);
                out.calls.push(call);
                self.calls += 1;
                self.buf.take_front(end + m.call_close.len());
                self.inside = false;
                self.line_due = m.newline;
            } else if self.line_due && !self.settle_line() {
                return Ok(out);
            } else if let Some(at) = self.specials.find(&self.buf, m.call_open) {
                self.buf.take_text_into(at, &mut out.content);
                self.buf.take_front(m.call_open.len());
                self.inside = true;
            } else {
                let upto = self.buf.text.len() - partial_suffix(&self.buf.text, m.call_open);
                self.buf.take_text_into(upto, &mut out.content);
                return Ok(out);
            }
        }
    }

    /// Settles the line feed after a call: dropped when the text starts with
    /// it, content when it starts with anything else. `false` while nothing
    /// is buffered to tell.
    fn settle_line(&mut self) -> bool {
        match self.buf.text.chars().next() {
            None => return false,
            Some('\n') => {
                self.buf.take_front(1);
            }
            Some(_) => {}
        }
        self.line_due = false;
        true
    }

    /// Releases what is held at the end of the stream; a call still open is
    /// an error.
    pub fn finish(&mut self) -> Result<Scanned, KvError> {
        let rest = std::mem::take(&mut self.buf).text;
        if self.inside {
            // Complete arguments keep their errors when the call is cut.
            if let Err(e @ (KvError::NotJson { .. } | KvError::DuplicateKey { .. })) =
                parse_call(self.markers, &rest, self.calls, &self.types)
            {
                return Err(e);
            }
            return Err(KvError::Unterminated {
                index: self.calls,
                close: self.markers.call_close,
                text: format!("{}{rest}", self.markers.call_open),
            });
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

/// One call's body, the text between a call's opening and its close: its
/// function name and its arguments as a JSON object's text.
fn parse_call(
    m: &Markers,
    body: &str,
    index: usize,
    types: &ArgTypes,
) -> Result<(String, String), KvError> {
    let name_end = body.find('<').unwrap_or(body.len());
    let name = body[..name_end].trim();
    if name.is_empty() {
        return Err(KvError::NoName {
            index,
            found: quote(body),
        });
    }
    let unexpected = |expected: &str, found: &str| KvError::Unexpected {
        index,
        name: name.to_owned(),
        expected: expected.to_owned(),
        found: quote(found),
    };
    let mut args: Vec<(String, Value)> = Vec::new();
    let mut rest = &body[name_end..];
    loop {
        rest = rest.trim_start();
        if rest.is_empty() {
            break;
        }
        let Some(k) = rest.strip_prefix(m.key_open) else {
            return Err(unexpected(
                &format!("{} or {}", m.key_open, m.call_close),
                rest,
            ));
        };
        let Some((key, after)) = m.split_key(k) else {
            return Err(unexpected(m.key_close, k));
        };
        let Some(v) = m.value_start(after) else {
            return Err(unexpected(m.value_open, after.trim_start()));
        };
        let Some(value_end) = v.find(m.value_close) else {
            return Err(unexpected(m.value_close, v));
        };
        if args.iter().any(|(a, _)| a == key) {
            return Err(KvError::DuplicateKey {
                index,
                name: name.to_owned(),
                key: key.to_owned(),
            });
        }
        let raw = &v[..value_end];
        let value = match types.kind(name, key) {
            Some(Kind::String) => Value::String(raw.to_owned()),
            Some(Kind::Json(ty)) => serde_json::from_str(raw).map_err(|_| KvError::NotJson {
                index,
                name: name.to_owned(),
                key: key.to_owned(),
                ty: ty.clone(),
                value: quote(raw),
            })?,
            None => serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned())),
        };
        args.push((key.to_owned(), value));
        rest = &v[value_end + m.value_close.len()..];
    }
    let fields: Vec<String> = args
        .iter()
        .map(|(k, v)| format!("{}:{v}", Value::String(k.clone())))
        .collect();
    Ok((name.to_owned(), format!("{{{}}}", fields.join(","))))
}

/// The call a scan has opened and not closed, read from where the last push
/// stopped, the arguments [`parse_call`] will close with as a prefix: each
/// complete argument, and the text of an argument the schema types `"string"`
/// as it arrives, up to the tail that could still become a tag that ends it.
/// A fault stops the reading where it stands: [`parse_call`] names it when the
/// call closes.
#[derive(Clone, Debug, Default)]
struct CallRead {
    call: Reading,
    stage: Stage,
    /// Where the stage's text starts, in the text after the call's opening.
    at: usize,
    /// How far the tag that ends the stage has been searched for.
    searched: usize,
    /// How far the call's close has been searched for.
    closing: usize,
}

#[derive(Clone, Debug, Default)]
enum Stage {
    /// The function name, up to the first tag.
    #[default]
    Name,
    /// Between arguments.
    Next,
    /// An argument's name.
    Key,
    /// The opening of the argument `key`'s value.
    Open(String),
    /// The value of the argument `key`.
    Value(String),
    /// The text stopped being the template's markup: the close decides.
    Stopped,
}

impl CallRead {
    /// Reads `s`, the call's text after its opening so far (its close not in
    /// it), on from where the last read stopped.
    fn advance(&mut self, m: &Markers, s: &str, types: &ArgTypes) {
        loop {
            let next = match &self.stage {
                Stage::Name => {
                    let Some(end) = find_from(s, &mut self.searched, "<") else {
                        return;
                    };
                    let name = s[..end].trim();
                    if name.is_empty() {
                        Stage::Stopped
                    } else {
                        self.call.name = Some(name.to_owned());
                        self.call.args.push('{');
                        self.at = end;
                        Stage::Next
                    }
                }
                Stage::Next => {
                    let tail = &s[self.at..];
                    let rest = tail.trim_start();
                    self.at += tail.len() - rest.len();
                    match rest.strip_prefix(m.key_open) {
                        Some(k) => {
                            self.at = s.len() - k.len();
                            self.searched = self.at;
                            Stage::Key
                        }
                        None if held(m.key_open, rest) => return,
                        None => Stage::Stopped,
                    }
                }
                Stage::Key => {
                    let from = self.searched;
                    match find_from(s, &mut self.searched, m.key_close) {
                        Some(end) if !s[self.at..end].contains('<') => {
                            let key = s[self.at..end].to_owned();
                            self.at = end + m.key_close.len();
                            Stage::Open(key)
                        }
                        None if !s[from..self.searched].contains('<') => return,
                        _ => Stage::Stopped,
                    }
                }
                Stage::Open(key) => {
                    let tail = &s[self.at..];
                    // A value opened by the name's close starts right after it.
                    let rest = if m.value_open.is_empty() {
                        tail
                    } else {
                        tail.trim_start()
                    };
                    self.at += tail.len() - rest.len();
                    match rest.strip_prefix(m.value_open) {
                        Some(v) => {
                            self.at = s.len() - v.len();
                            self.searched = self.at;
                            let name = self.call.name.as_deref().unwrap_or_default();
                            if types.kind(name, key) == Some(&Kind::String) {
                                Args::key(&mut self.call.args, key);
                                self.call.args.push('"');
                            }
                            Stage::Value(key.clone())
                        }
                        None if held(m.value_open, rest) => return,
                        None => Stage::Stopped,
                    }
                }
                Stage::Value(key) => {
                    let kind = types.kind(self.call.name.as_deref().unwrap_or_default(), key);
                    let text = kind == Some(&Kind::String);
                    let args = text.then_some(&mut self.call.args);
                    let Some(end) =
                        read_value(s, &mut self.searched, m.value_close, m.call_close, args)
                    else {
                        return;
                    };
                    let raw = &s[self.at..end];
                    self.at = end + m.value_close.len();
                    if text {
                        self.call.args.push('"');
                        Stage::Next
                    } else {
                        let value = match kind {
                            Some(Kind::Json(_)) => serde_json::from_str::<Value>(raw).ok(),
                            _ => Some(
                                serde_json::from_str(raw)
                                    .unwrap_or_else(|_| Value::String(raw.to_owned())),
                            ),
                        };
                        match value {
                            Some(value) => {
                                Args::key(&mut self.call.args, key);
                                self.call.args.push_str(&value.to_string());
                                Stage::Next
                            }
                            None => Stage::Stopped,
                        }
                    }
                }
                Stage::Stopped => return,
            };
            self.stage = next;
        }
    }
}

/// The call `spelling` spells, as the parts structured output forces: the
/// inverse of [`KvScan`]. The name, then the parameters as the object rule
/// writes them: the required ones, in the schema's order, then the optional
/// ones, each of which may be left out. The scanner reads a value as text
/// only where the schema says `"type": "string"`, itself or through a
/// `$ref`, so a text-valued parameter that does not say it is refused
/// rather than written as something that could read back as JSON.
pub(crate) fn call_parts(
    spelling: KvSpelling,
    funs: &[Fun<'_>],
    mode: CallMode,
) -> Result<Call, Refusal> {
    let m = spelling.markers();
    let block = |a: &Arg<'_>| -> Result<Vec<Part>, Refusal> {
        if is_text(&a.schema) && a.schema.get("type").and_then(Value::as_str) != Some("string") {
            return Err(outside(
                &a.path,
                format!(
                    "{}'s tool markup reads a text value as text only under \"type\": \
                     \"string\"; add it",
                    m.family
                ),
            ));
        }
        let mut parts = vec![Part::Text(format!(
            "{}{}{}{}",
            m.key_open, a.key, m.key_close, m.value_open
        ))];
        parts.extend(value(&a.schema, Style::Raw, &a.path)?);
        parts.push(Part::Text(format!("{}{}", m.value_close, m.line())));
        Ok(parts)
    };
    let body = call(
        funs,
        "",
        |name| format!("{name}{}", m.line()),
        |f| {
            let mut parts = object_parameters(&f.args, &block)?;
            parts.push(Part::Text(m.call_close.to_owned()));
            Ok(parts)
        },
    )?;
    Ok(Call::opened(m.call_open, mode, body))
}
