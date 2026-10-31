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
//! keeps it. That is the [`Framing::Lines`] spelling, Qwen3.6's and Qwen3.8's;
//! a template that renders a call compactly (MiMo's,
//! `<parameter=KEY>VALUE</parameter>`) is [`Framing::Compact`]: the value is the
//! bytes between the two tags, a newline in it its own. What the value is read
//! as the request's tool schema says, the way llama.cpp's parser reads it: a
//! parameter whose schema can resolve to a
//! string ([`resolves_to_string`]) is the raw text; a `boolean` parameter, as
//! Qwen3-Coder's reference parser reads one, is `true` or `false` in either
//! case ([`boolean`]), spelled in lower case; every other parameter (another
//! `type`, a `type` list without `string`, or no `type` at all) is JSON,
//! spelled in `arguments` as the model wrote it, a control character inside
//! one of its strings escaped ([`json_value`]).
//!
//! A call is emitted when its `</tool_call>` arrives; the parameters become a
//! JSON object in the model's order, serialized compactly to a string. Text
//! before the first call, and after the calls once something other than the
//! layout follows, is content; the whitespace a closed call's layout left is
//! nothing, as the template writes its calls one newline apart. Under a
//! template that teaches no think span ([`Framing::Coder`], Qwen3-Coder's
//! own) the first call may also open bare, `<function=NAME>` with no
//! `<tool_call>` before it — the omission the model now and then makes, as
//! llama.cpp's `tool_call_first` reads it: the name must name one of the
//! request's tools, the call still closes with `</tool_call>`, and a second
//! call needs its own `<tool_call>`. Markup that
//! does not parse — no function opener, no name, text where a tag belongs, a
//! function or a parameter the request's tools do not name, one parameter
//! twice, a value its schema types JSON that is not JSON, a boolean that is
//! neither word — and a call still open when the stream ends are a
//! [`QwenXmlError`], never content and never dropped; the chat parser reads a
//! generation the token limit cut inside a call as an answer, not as this error
//! ([`crate::dsml::ChatParser::try_finish_at`]).
//! The call a scan has opened and not closed is reported as it is read
//! ([`Scanned::open`]): its name once `<function=NAME>` is whole, and the
//! arguments so far, a prefix of the arguments the call closes with, the text
//! of a string parameter as it arrives. Every decision is independent of how
//! the text and special-token offsets were cut into pieces. A `<tool_call>` or `</tool_call>`
//! the vocabulary holds as one special token is markup only where that token
//! wrote it ([`SpecialTags`]): spelled in ordinary tokens it is text, content
//! outside a call and no close inside one.

use std::borrow::Cow;
use std::collections::HashMap;

use serde_json::Value;

use crate::callstream::{self, Reading, held, read_value};
use crate::constrain::{
    Arg, Call, CallMode, Fun, Part, Refusal, Style, call, object_parameters, value,
};
use crate::dsml::{Scanned, ToolCall};
use crate::engine::Tokenizer;
use crate::reasoning::{Marked, SpecialTags, partial_suffix};

// The format's markup is its description's (the models crate's `function_xml`, not this crate's `models` module),
// the strings the template table names a file's format by.
use ::models::function_xml::PARAM_VALUE_CLOSE;
pub use ::models::function_xml::{
    CALL_CLOSE, CALL_OPEN, FUNCTION_CLOSE, FUNCTION_OPEN, PARAM_CLOSE, PARAM_OPEN,
};

/// How a template frames a parameter's value, a fact of the template that its
/// rendered call shows ([`Framing::of_rendered`]): one parser reads both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Framing {
    /// `<parameter=KEY>\nVALUE\n</parameter>`: the newline after the opener and
    /// the one before the closer (with the one after it) are the markup's own.
    Lines,
    /// The lines framing of a template that renders no think span
    /// (Qwen3-Coder's own): its model has no span to reason in, and its first
    /// call may open bare, `<function=NAME>` with no `<tool_call>` before it
    /// (llama.cpp's `tool_call_first`).
    Coder,
    /// `<parameter=KEY>VALUE</parameter>`: the value is the bytes between the
    /// tags, whatever whitespace they hold.
    Compact,
}

impl Framing {
    /// What follows `<parameter=KEY>` before the value's first byte.
    const fn value_open(self) -> &'static str {
        match self {
            Framing::Lines | Framing::Coder => "\n",
            Framing::Compact => "",
        }
    }

    /// The text that ends a value, and is consumed with it.
    const fn value_close(self) -> &'static str {
        match self {
            Framing::Lines | Framing::Coder => PARAM_VALUE_CLOSE,
            Framing::Compact => PARAM_CLOSE,
        }
    }

    /// The framing a rendered assistant call shows: the text a template wrote
    /// for `<parameter=key>` holding the string `value` in its last
    /// `<tool_call>` (the probe's call; a header's example sits before it),
    /// read with the scanner's own delimiters, so the framing found is one
    /// the scanner closes. `None` when that call renders no such parameter,
    /// or in a spelling that is neither.
    pub(crate) fn of_rendered(rendered: &str, key: &str, value: &str) -> Option<Framing> {
        let (_, call) = rendered.rsplit_once(CALL_OPEN)?;
        let (_, after) = call.split_once(&format!("{PARAM_OPEN}{key}>"))?;
        [Framing::Lines, Framing::Compact]
            .into_iter()
            .find(|framing| {
                after.starts_with(&format!(
                    "{}{value}{}",
                    framing.value_open(),
                    framing.value_close()
                ))
            })
    }
}

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
    /// A value that is neither `true` nor `false` where the schema typed it boolean.
    #[error(
        "tool call {index} ({name}): parameter {key} is typed boolean, and its value is neither true nor false: {value:?}"
    )]
    NotBoolean {
        index: usize,
        name: String,
        key: String,
        value: String,
    },
}

impl QwenXmlError {
    /// The text of the call the generation ended inside, when that is the whole fault: markup that
    /// read as the template's so far and was not closed.
    pub(crate) fn unclosed(&self) -> Option<String> {
        match self {
            QwenXmlError::Unterminated { text, .. } => Some(text.clone()),
            _ => None,
        }
    }
}

/// What a parameter's schema says of its value.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Kind {
    /// The schema can resolve to a string: the value is the raw text.
    String,
    /// `type: boolean`: `true` or `false` in either case ([`boolean`]).
    Bool,
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

    /// Whether a tool's name starts with `part`: a name still arriving may
    /// yet be one the tools hold.
    fn may_name(&self, part: &str) -> bool {
        self.kinds.keys().any(|name| name.starts_with(part))
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
pub(crate) fn resolves_to_string(schema: &Value, root: &Value, seen: &mut Vec<String>) -> bool {
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
/// resolve to a string, a boolean for `type: boolean`, else JSON — the type as
/// the schema wrote it, `none` when it wrote none. `root` is the tool's
/// `parameters`, which a `$ref` points into.
fn kind_of(schema: &Value, root: &Value) -> Kind {
    if resolves_to_string(schema, root, &mut Vec::new()) {
        return Kind::String;
    }
    if schema.get("type").and_then(Value::as_str) == Some("boolean") {
        return Kind::Bool;
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
#[derive(Clone, Debug)]
pub struct QwenScan {
    kinds: ParamKinds,
    framing: Framing,
    /// Inside a call (its opener, `<tool_call>` or a bare `<function=NAME>`
    /// under [`Framing::Coder`], consumed or held as the call's own text).
    inside: bool,
    /// The open call opened bare, `<function=NAME>` with no `<tool_call>`
    /// before it: its text carries no `<tool_call>` the error or the answer
    /// would show one the model never wrote.
    bare: bool,
    /// A call closed: only the layout between calls may follow, and it is
    /// consumed, never content, as the template writes its calls one newline
    /// apart.
    after_call: bool,
    /// Text not yet decided.
    buf: Marked,
    /// The tags read only where their special tokens wrote them.
    specials: SpecialTags,
    calls: usize,
    read: CallRead,
}

impl QwenScan {
    /// A scanner that types parameters by `kinds` and reads values framed as
    /// `framing` says.
    #[must_use]
    pub fn new(kinds: ParamKinds, framing: Framing) -> Self {
        QwenScan {
            kinds,
            framing,
            inside: false,
            bare: false,
            after_call: false,
            buf: Marked::default(),
            specials: SpecialTags::default(),
            calls: 0,
            read: CallRead::default(),
        }
    }

    /// Reads `<tool_call>` and `</tool_call>` only where their special
    /// tokens wrote them, for those `tok` holds as one special token.
    pub(crate) fn read_specials(&mut self, tok: &dyn Tokenizer) {
        self.specials = SpecialTags::of(tok, &[CALL_OPEN, CALL_CLOSE]);
    }

    /// Feeds text; returns the content that may be sent and the calls that
    /// closed.
    pub fn push(&mut self, piece: &str) -> Result<Scanned, QwenXmlError> {
        Ok(self
            .push_stream(piece, &[])?
            .whole(&self.read.call, self.calls))
    }

    /// [`QwenScan::push`] of text whose special tokens' texts are at
    /// `specials` (byte offset and length), reporting the call text as
    /// `advances`, each piece's new arguments, instead of the open call's
    /// whole text.
    pub(crate) fn push_stream(
        &mut self,
        piece: &str,
        specials: &[(usize, usize)],
    ) -> Result<Scanned, QwenXmlError> {
        self.buf.push(piece, specials);
        let mut out = Scanned::default();
        let advances = out.advances.insert(Vec::new());
        loop {
            if self.inside {
                let closed = self.read.advance(
                    &self.buf,
                    &self.specials,
                    self.calls,
                    &self.kinds,
                    self.framing,
                )?;
                let Some((name, arguments, used)) = closed else {
                    self.read.call.send(self.calls, advances);
                    return Ok(out);
                };
                let call = ToolCall {
                    index: self.calls,
                    name,
                    arguments,
                };
                std::mem::take(&mut self.read).call.close(&call, advances);
                out.calls.push(call);
                self.calls += 1;
                self.buf.take_front(used);
                self.inside = false;
                self.after_call = true;
            } else if self.after_call {
                let ws = self.buf.text.len() - self.buf.text.trim_start().len();
                if held(CALL_OPEN, &self.buf.text[ws..]) {
                    return Ok(out);
                }
                if self.specials.at(&self.buf, ws, CALL_OPEN) {
                    self.buf.take_front(ws + CALL_OPEN.len());
                    self.inside = true;
                    self.bare = false;
                    self.after_call = false;
                } else {
                    // Text after the calls: it stays content, its leading
                    // whitespace with it.
                    self.after_call = false;
                }
            } else {
                // Before the first call: `<tool_call>` opens one, and a
                // template that teaches no think span ([`Framing::Coder`])
                // also lets the call open bare, `<function=NAME>` naming a
                // tool of the request — whichever the text holds first.
                let bare_first = self.calls == 0 && matches!(self.framing, Framing::Coder);
                let call = self.specials.find(&self.buf, CALL_OPEN);
                let bare = bare_first
                    .then(|| bare_open(&self.buf.text, &self.kinds))
                    .flatten()
                    .filter(|b| call.is_none_or(|a| *b < a));
                if let Some(at) = bare {
                    self.buf.take_text_into(at, &mut out.content);
                    self.inside = true;
                    self.bare = true;
                } else if let Some(at) = call {
                    self.buf.take_text_into(at, &mut out.content);
                    self.buf.take_front(CALL_OPEN.len());
                    self.inside = true;
                    self.bare = false;
                } else {
                    let bare = if bare_first {
                        bare_suffix(&self.buf.text, &self.kinds)
                    } else {
                        0
                    };
                    let hold = partial_suffix(&self.buf.text, CALL_OPEN).max(bare);
                    let upto = self.buf.text.len() - hold;
                    self.buf.take_text_into(upto, &mut out.content);
                    return Ok(out);
                }
            }
        }
    }

    /// Releases what is held at the end of the stream; a call still open is
    /// an error (the markup error its text already shows, else
    /// [`QwenXmlError::Unterminated`]), an incomplete opening tag is text,
    /// and the whitespace a closed call's layout left is nothing.
    pub fn finish(&mut self) -> Result<Scanned, QwenXmlError> {
        if self.inside {
            parse_call(
                &self.buf,
                &self.specials,
                self.calls,
                &self.kinds,
                self.framing,
                true,
            )?;
        }
        let rest = std::mem::take(&mut self.buf).text;
        if self.inside {
            return Err(QwenXmlError::Unterminated {
                index: self.calls,
                // A call that opened bare carries no `<tool_call>` its text
                // would show one the model never wrote.
                text: if self.bare {
                    rest
                } else {
                    format!("{CALL_OPEN}{rest}")
                },
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

/// Where a first call opens bare in `text`: `<function=NAME>` whole, its name
/// one the request's tools hold — llama.cpp's trigger literals carry each
/// tool's name, so an opener naming nothing the tools do is no trigger.
/// `None` when no whole opener has come, the text left to hold
/// ([`bare_suffix`]).
fn bare_open(text: &str, kinds: &ParamKinds) -> Option<usize> {
    let mut from = 0;
    while let Some(n) = text[from..].find(FUNCTION_OPEN) {
        let at = from + n;
        let name = &text[at + FUNCTION_OPEN.len()..];
        // The name is still arriving: the opener may yet name a tool.
        let end = name.find('>')?;
        if kinds.names(&name[..end]) {
            return Some(at);
        }
        from = at + 1;
    }
    None
}

/// How many of `text`'s trailing bytes a bare first call may still need: a
/// partial `<function=`, or one whose name is still arriving and may yet
/// name a tool of the request.
fn bare_suffix(text: &str, kinds: &ParamKinds) -> usize {
    let mut hold = partial_suffix(text, FUNCTION_OPEN);
    if let Some(at) = text.rfind(FUNCTION_OPEN) {
        let name = &text[at + FUNCTION_OPEN.len()..];
        if !name.contains('>') && kinds.may_name(name) {
            hold = hold.max(text.len() - at);
        }
    }
    hold
}

/// The JSON spelling of a boolean parameter's text: `true` or `false` in either case, as the Qwen3-Coder reference
/// parser reads one (a model that writes Python's `True` is understood), with JSON's whitespace around it allowed.
/// Anything else is no boolean: the reference's `false` for it would be a guess.
fn boolean(raw: &str) -> Option<&'static str> {
    let word = raw.trim_matches([' ', '\t', '\n', '\r']);
    if word.eq_ignore_ascii_case("true") {
        Some("true")
    } else if word.eq_ignore_ascii_case("false") {
        Some("false")
    } else {
        None
    }
}

/// The text to write into `arguments` for a JSON-typed parameter: `raw` as the
/// model wrote it when it parses, else the same text with every control
/// character U+0000–U+001F inside a string literal escaped (`\n`, `\r`, `\t`
/// for those, `\u00XX` for the rest; the raw byte is not JSON, its meaning
/// still the character itself) when that parses, else none: no JSON, and a
/// value for it would be a guess.
fn json_value(raw: &str) -> Option<Cow<'_, str>> {
    if serde_json::from_str::<Value>(raw).is_ok() {
        return Some(Cow::Borrowed(raw));
    }
    let mut escaped = String::with_capacity(raw.len());
    let mut in_string = false;
    let mut after_backslash = false;
    for c in raw.chars() {
        if !in_string {
            in_string = c == '"';
        } else if after_backslash {
            after_backslash = false;
        } else {
            match c {
                '\\' => after_backslash = true,
                '"' => in_string = false,
                '\n' => {
                    escaped.push_str("\\n");
                    continue;
                }
                '\r' => {
                    escaped.push_str("\\r");
                    continue;
                }
                '\t' => {
                    escaped.push_str("\\t");
                    continue;
                }
                c @ '\0'..='\u{1f}' => {
                    escaped.push_str(&format!("\\u{:04x}", u32::from(c)));
                    continue;
                }
                _ => {}
            }
        }
        escaped.push(c);
    }
    serde_json::from_str::<Value>(&escaped)
        .is_ok()
        .then_some(Cow::Owned(escaped))
}

/// The first characters of `s`, for an error.
fn quote(s: &str) -> String {
    s.chars().take(QUOTE).collect()
}

/// One call, the text after `<tool_call>`, at the start of `m`: its function
/// name, its parameters as a JSON object's text, and the bytes it took
/// through `</tool_call>`. `Ok(None)` when the text so far can only be the
/// start of a call. An error that quotes the text after the fault waits for
/// a full quote, or for the stream's `end`, so its message does not depend
/// on where the stream was cut. Whitespace between the tags is skipped (the template
/// writes a newline after each); where a value starts and ends is the
/// `framing`'s: under [`Framing::Lines`] the newline that opens a value is not
/// whitespace between tags but the value's own boundary, under
/// [`Framing::Compact`] the value is the bytes between its tags. Its
/// `</tool_call>` is the one `specials` reads as the marker.
fn parse_call(
    m: &Marked,
    specials: &SpecialTags,
    index: usize,
    kinds: &ParamKinds,
    framing: Framing,
    end: bool,
) -> Result<Option<(String, String, usize)>, QwenXmlError> {
    let s = m.text.as_str();
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
            if !specials.at(m, s.len() - rest.len(), CALL_CLOSE) {
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
        let Some(v) = open.strip_prefix(framing.value_open()) else {
            if open.is_empty() || !settled(rest) {
                return Ok(None);
            }
            return Err(unexpected("a newline after <parameter=KEY>", rest));
        };
        let Some(val_end) = v.find(framing.value_close()) else {
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
            Some(Kind::String) => callstream::string(raw),
            Some(Kind::Bool) => {
                boolean(raw)
                    .map(str::to_owned)
                    .ok_or_else(|| QwenXmlError::NotBoolean {
                        index,
                        name: name.to_owned(),
                        key: key.to_owned(),
                        value: quote(raw),
                    })?
            }
            Some(Kind::Json(ty)) => match json_value(raw) {
                // The value as the model wrote it, as llama-server writes a
                // typed value back; a re-serialization would respell it.
                Some(value) => value.into_owned(),
                None => {
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
        at += ws
            + PARAM_OPEN.len()
            + key_end
            + 1
            + framing.value_open().len()
            + val_end
            + framing.value_close().len();
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

/// The cursor and argument text of the active call. Only a value's undecided
/// delimiter suffix is searched again when more text arrives.
#[derive(Clone, Debug, Default)]
struct CallRead {
    at: usize,
    call: Reading,
    value: Option<(String, usize, usize)>,
}

impl CallRead {
    fn advance(
        &mut self,
        m: &Marked,
        specials: &SpecialTags,
        index: usize,
        kinds: &ParamKinds,
        framing: Framing,
    ) -> Result<Option<(String, String, usize)>, QwenXmlError> {
        let s = m.text.as_str();
        if self.call.name.is_none() {
            let rest = s.trim_start();
            let Some(body) = rest.strip_prefix(FUNCTION_OPEN) else {
                return parse_call(m, specials, index, kinds, framing, false);
            };
            let Some(end) = body.find('>') else {
                return Ok(None);
            };
            let name = &body[..end];
            if name.is_empty() || name.contains(['<', '\n', '"']) || !kinds.names(name) {
                return parse_call(m, specials, index, kinds, framing, false);
            }
            self.call.name = Some(name.to_owned());
            self.call.args.push('{');
            self.at = s.len() - body.len() + end + 1;
        }
        loop {
            if let Some((key, start, searched)) = &mut self.value {
                let close = framing.value_close();
                let kind = kinds.kind(self.call.name.as_deref().unwrap_or_default(), key);
                let text = kind == Some(&Kind::String);
                let args = text.then_some(&mut self.call.args);
                let Some(end) = read_value(s, searched, close, CALL_CLOSE, args) else {
                    return Ok(None);
                };
                if text {
                    self.call.args.push('"');
                } else {
                    callstream::Args::key(&mut self.call.args, key);
                    let raw = &s[*start..end];
                    match kind {
                        Some(Kind::Bool) => match boolean(raw) {
                            Some(word) => self.call.args.push_str(word),
                            None => return parse_call(m, specials, index, kinds, framing, false),
                        },
                        Some(Kind::Json(_)) => match json_value(raw) {
                            Some(value) => self.call.args.push_str(&value),
                            None => return parse_call(m, specials, index, kinds, framing, false),
                        },
                        _ => return parse_call(m, specials, index, kinds, framing, false),
                    }
                }
                self.at = end + close.len();
                self.value = None;
            }
            let rest = s[self.at..].trim_start();
            let Some(p) = rest.strip_prefix(PARAM_OPEN) else {
                if held(PARAM_OPEN, rest)
                    || held(FUNCTION_CLOSE, rest)
                    || rest
                        .strip_prefix(FUNCTION_CLOSE)
                        .is_some_and(|tail| held(CALL_CLOSE, tail.trim_start()))
                {
                    return Ok(None);
                }
                return parse_call(m, specials, index, kinds, framing, false);
            };
            let Some(end) = p.find('>') else {
                return Ok(None);
            };
            let key = &p[..end];
            if key.is_empty()
                || key.contains(['<', '\n', '"'])
                || kinds
                    .kind(self.call.name.as_deref().unwrap_or_default(), key)
                    .is_none()
            {
                return parse_call(m, specials, index, kinds, framing, false);
            }
            let Some(v) = p[end + 1..].strip_prefix(framing.value_open()) else {
                return parse_call(m, specials, index, kinds, framing, false);
            };
            let text = kinds.kind(self.call.name.as_deref().unwrap_or_default(), key)
                == Some(&Kind::String);
            if text {
                callstream::Args::key(&mut self.call.args, key);
                self.call.args.push('"');
            }
            let start = s.len() - v.len();
            self.value = Some((key.to_owned(), start, start));
        }
    }
}

/// The call this markup spells, as the parts structured output forces: the
/// inverse of [`QwenScan`]. `<function=NAME>` with a `<parameter=KEY>` block
/// per parameter as the object rule writes them — the required ones, in the
/// schema's order, then the optional ones, each of which may be left out —
/// each value written as the schema reads it (raw text for a string, JSON
/// for the rest) and framed as `framing` says ([`Framing::Lines`] a line
/// each, [`Framing::Compact`] none between the tags, as the compact
/// template writes them). The opener is `<tool_call>`.
pub(crate) fn call_parts(
    framing: Framing,
    funs: &[Fun<'_>],
    mode: CallMode,
) -> Result<Call, Refusal> {
    let block = |a: &Arg<'_>| -> Result<Vec<Part>, Refusal> {
        let mut parts = vec![Part::Text(format!(
            "{PARAM_OPEN}{}>{}",
            a.key,
            framing.value_open()
        ))];
        parts.extend(value(&a.schema, Style::Raw, &a.path)?);
        parts.push(Part::Text(match framing {
            Framing::Lines | Framing::Coder => PARAM_VALUE_CLOSE.to_owned(),
            Framing::Compact => PARAM_CLOSE.to_owned(),
        }));
        Ok(parts)
    };
    let body = call(funs, &format!("\n{FUNCTION_OPEN}"), str::to_owned, |f| {
        let mut parts = vec![Part::Text(">\n".to_owned())];
        parts.extend(object_parameters(&f.args, &block)?);
        parts.push(Part::Text(format!("{FUNCTION_CLOSE}\n{CALL_CLOSE}")));
        Ok(parts)
    })?;
    Ok(Call::opened(CALL_OPEN, mode, body))
}
