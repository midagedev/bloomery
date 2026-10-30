//! Mistral tool calls: the `[TOOL_CALLS]` markup Mistral Large 4's chat
//! template teaches (its `tools` header and the assistant `tool_calls`
//! branch), parsed out of the text after the think span.
//!
//! ```text
//! [TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}
//! [TOOL_CALLS]get_time[ARGS]{}
//! ```
//!
//! The template writes one markup per call — no call id, nothing between
//! calls — its name before `[ARGS]` and the arguments after it as one JSON
//! value: the `tojson` of the request's object, or the string the request
//! gave, verbatim. A call is emitted when its arguments' value ends, which the
//! markup names no close tag of: the value's own brackets, quotes and words
//! end it ([`value_end`]), so a `[TOOL_CALLS]` inside a string is text, not a
//! call. The closed call's `arguments` are the value serialized compactly, in
//! the model's key order. Text before, between and after calls is content.
//! Markup that does not parse — no plain name before `[ARGS]`, arguments that
//! are no JSON value — and a call still open when the generation ends are a
//! [`MistralError`], never content and never dropped; the chat parser reads a
//! generation the token limit cut inside a call as an answer, not as this
//! error ([`crate::dsml::ChatParser::try_finish_at`]). The call a scan has
//! opened and not closed is reported as it is read ([`Scanned::open`]): its
//! name once `[ARGS]` is whole, and the arguments so far, a prefix of the
//! arguments the call closes with, once the value's first byte is read. Every
//! decision depends on the text only, never on how it was cut into pieces.

use serde_json::Value;

use crate::callstream::Progress;
use crate::constrain::{Call, CallMode, Fun, Part, Refusal, call};
use crate::dsml::{Scanned, ToolCall};
use crate::hermes::compact_prefix;
use crate::reasoning::partial_suffix;

/// Opens a call.
pub const CALL_OPEN: &str = "[TOOL_CALLS]";
/// Between a call's name and its arguments.
pub const ARGS: &str = "[ARGS]";

/// How much of the offending text an error quotes.
const QUOTE: usize = 40;

/// A generation whose tool-call markup does not parse. `index` counts calls
/// in the generation from 0.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MistralError {
    /// The stream ended inside a call: its name, or its arguments' value, was
    /// still growing.
    #[error("tool call {index} is not closed when the generation ends: {text:?}")]
    Unterminated { index: usize, text: String },
    /// `[TOOL_CALLS]` not followed by a plain name and `[ARGS]`.
    #[error("tool call {index} names no function before {ARGS}: {found:?}")]
    NoName { index: usize, found: String },
    /// `[ARGS]` not followed by one JSON value.
    #[error("tool call {index} ({name}) carries no JSON value after {ARGS}: {found:?}")]
    NotJson {
        index: usize,
        name: String,
        found: String,
    },
}

impl MistralError {
    /// The text of the call the generation ended inside, when that is the whole fault: markup that
    /// read as the template's so far and was not closed.
    pub(crate) fn unclosed(&self) -> Option<String> {
        match self {
            MistralError::Unterminated { text, .. } => Some(text.clone()),
            _ => None,
        }
    }
}

/// The call a `[TOOL_CALLS]` opened, and what of it is read.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Open {
    /// The name, `[TOOL_CALLS]` consumed, `[ARGS]` not come.
    Name,
    /// The arguments' value, `[ARGS]` consumed; the call's name.
    Args { name: String },
}

/// Streaming scanner over the text after the think span.
#[derive(Clone, Debug, Default)]
pub struct MistralScan {
    /// The call the last `[TOOL_CALLS]` opened; `None` between calls.
    open: Option<Open>,
    /// Text not yet decided: the open call's name or arguments so far, or the
    /// text between calls.
    buf: String,
    calls: usize,
}

impl MistralScan {
    /// Feeds text; returns the content that may be sent and the calls that
    /// closed.
    pub fn push(&mut self, piece: &str) -> Result<Scanned, MistralError> {
        self.buf.push_str(piece);
        let mut out = Scanned::default();
        loop {
            let Some(open) = self.open.take() else {
                let Some(at) = self.buf.find(CALL_OPEN) else {
                    let upto = self.buf.len() - partial_suffix(&self.buf, CALL_OPEN);
                    out.content.push_str(&self.buf[..upto]);
                    self.buf.drain(..upto);
                    return Ok(out);
                };
                out.content.push_str(&self.buf[..at]);
                self.buf.drain(..at + CALL_OPEN.len());
                self.open = Some(Open::Name);
                continue;
            };
            match open {
                Open::Name => {
                    let Some(at) = self.buf.find(ARGS) else {
                        // The name so far, less the tail that could still grow into `[ARGS]`,
                        // is the whole name there is: a `[` or a newline in it is no name.
                        let upto = self.buf.len() - partial_suffix(&self.buf, ARGS);
                        if self.buf[..upto].contains(['[', '\n']) {
                            return Err(MistralError::NoName {
                                index: self.calls,
                                found: name_quote(&self.buf[..upto]),
                            });
                        }
                        self.open = Some(Open::Name);
                        return Ok(out);
                    };
                    let name = self.buf[..at].to_owned();
                    if !is_name(&name) {
                        return Err(MistralError::NoName {
                            index: self.calls,
                            found: name_quote(&name),
                        });
                    }
                    self.buf.drain(..at + ARGS.len());
                    self.open = Some(Open::Args { name });
                }
                Open::Args { name } => match value_end(&self.buf, false) {
                    invalid @ ValueEnd::Invalid(_) => {
                        return Err(MistralError::NotJson {
                            index: self.calls,
                            name,
                            found: invalid.fault(&self.buf),
                        });
                    }
                    ValueEnd::Incomplete => {
                        let args = compact_prefix(&self.buf);
                        if !args.is_empty() {
                            out.open = Some(Progress {
                                index: self.calls,
                                name: name.clone(),
                                args,
                            });
                        }
                        self.open = Some(Open::Args { name });
                        return Ok(out);
                    }
                    ValueEnd::Complete(end) => {
                        let call = self.closed(&self.buf[..end], &name)?;
                        out.calls.push(call);
                        self.calls += 1;
                        self.buf.drain(..end);
                    }
                },
            }
        }
    }

    /// Releases what is held at the end of the stream: the arguments' value,
    /// complete at the stream's end, closes its call; one still growing, or a
    /// name with no `[ARGS]`, is the error; the text between calls is content.
    pub fn finish(&mut self) -> Result<Scanned, MistralError> {
        let buf = std::mem::take(&mut self.buf);
        match self.open.take() {
            None => Ok(Scanned {
                content: buf,
                ..Scanned::default()
            }),
            Some(Open::Name) => Err(MistralError::Unterminated {
                index: self.calls,
                text: format!("{CALL_OPEN}{buf}"),
            }),
            Some(Open::Args { name }) => match value_end(&buf, true) {
                invalid @ ValueEnd::Invalid(_) => Err(MistralError::NotJson {
                    index: self.calls,
                    name,
                    found: invalid.fault(&buf),
                }),
                ValueEnd::Incomplete => Err(MistralError::Unterminated {
                    index: self.calls,
                    text: format!("{CALL_OPEN}{name}{ARGS}{buf}"),
                }),
                ValueEnd::Complete(end) => {
                    let call = self.closed(&buf[..end], &name)?;
                    Ok(Scanned {
                        calls: vec![call],
                        ..Scanned::default()
                    })
                }
            },
        }
    }

    /// The call whose arguments' value is `raw`, proven JSON, compact in the
    /// model's key order.
    fn closed(&self, raw: &str, name: &str) -> Result<ToolCall, MistralError> {
        let v: Value = serde_json::from_str(raw).map_err(|_| MistralError::NotJson {
            index: self.calls,
            name: name.to_owned(),
            found: quote(raw.trim()),
        })?;
        Ok(ToolCall {
            index: self.calls,
            name: name.to_owned(),
            arguments: v.to_string(),
        })
    }
}

/// The first characters of `s`, for an error.
fn quote(s: &str) -> String {
    s.chars().take(QUOTE).collect()
}

/// Whether `name` may name a call: non-empty, and free of the markup's own
/// `[` and of newlines.
fn is_name(name: &str) -> bool {
    !name.is_empty() && !name.contains(['[', '\n'])
}

/// The name text an error quotes: the name cut just past its first `[` or
/// newline, so every cut of one text quotes the same fault.
fn name_quote(name: &str) -> String {
    let end = name.find(['[', '\n']).map_or(name.len(), |at| at + 1);
    quote(&name[..end])
}

/// Where the JSON value at the start of `s` ends.
#[derive(Debug, PartialEq, Eq)]
enum ValueEnd {
    /// `s` starts with one complete JSON value, ending here.
    Complete(usize),
    /// `s` is a prefix of some JSON value: nothing is decided yet.
    Incomplete,
    /// No JSON value starts here; the codepoint at this place is the first
    /// that no value holds. What the error quotes is the text to it, a
    /// property of the text alone, never of how it was cut.
    Invalid(usize),
}

impl ValueEnd {
    /// The same reading, its end moved `by` further into the text.
    fn map(self, by: usize) -> Self {
        match self {
            ValueEnd::Complete(end) => ValueEnd::Complete(end + by),
            ValueEnd::Invalid(at) => ValueEnd::Invalid(at + by),
            ValueEnd::Incomplete => ValueEnd::Incomplete,
        }
    }

    /// The text an error quotes: through the codepoint the reading stopped
    /// at, so every cut of one text quotes the same.
    fn fault(self, s: &str) -> String {
        match self {
            ValueEnd::Invalid(at) => {
                let end = at
                    + s[at..]
                        .chars()
                        .next()
                        .expect("a fault codepoint")
                        .len_utf8();
                quote(s[..end].trim())
            }
            _ => unreachable!("a fault text for a value that may still parse"),
        }
    }
}

/// The JSON value at the start of `s`, after the whitespace it may open with:
/// where it ends. The value is proven JSON by `serde_json` once its end is
/// named; the reading here brackets it alone. `at_end` is set at the end of
/// the stream, where a number can grow no further.
fn value_end(s: &str, at_end: bool) -> ValueEnd {
    let t = s.trim_start_matches([' ', '\t', '\r', '\n']);
    let lead = s.len() - t.len();
    let Some(c) = t.chars().next() else {
        return ValueEnd::Incomplete;
    };
    match c {
        '"' => string_end(t).map(lead),
        '{' | '[' => container_end(t).map(lead),
        't' => word_end(t, "true").map(lead),
        'f' => word_end(t, "false").map(lead),
        'n' => word_end(t, "null").map(lead),
        c if c == '-' || c.is_ascii_digit() => number_end(t, at_end).map(lead),
        _ => ValueEnd::Invalid(0).map(lead),
    }
}

/// The string at the start of `s` (its opening quote there): where it ends,
/// at its closing quote. An escape JSON does not have and a raw control
/// character stop it, naming the codepoint; the pairs a lone surrogate makes
/// are `serde_json`'s to refuse once the quote closes.
fn string_end(s: &str) -> ValueEnd {
    let b = s.as_bytes();
    let mut i = 1;
    while i < b.len() {
        match b[i] {
            b'"' => return ValueEnd::Complete(i + 1),
            b'\\' => {
                let Some(e) = b.get(i + 1) else {
                    return ValueEnd::Incomplete;
                };
                match e {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => i += 2,
                    b'u' => {
                        if b.len() < i + 6 {
                            return ValueEnd::Incomplete;
                        }
                        let hex = &b[i + 2..i + 6];
                        if let Some(at) = hex.iter().position(|d| !d.is_ascii_hexdigit()) {
                            return ValueEnd::Invalid(i + 2 + at);
                        }
                        i += 6;
                    }
                    _ => return ValueEnd::Invalid(i + 1),
                }
            }
            c if c < 0x20 => return ValueEnd::Invalid(i),
            _ => i += 1,
        }
    }
    ValueEnd::Incomplete
}

/// The bracketed container at the start of `s` (its `{` or `[` there): where
/// it ends, at the bracket that closes its nesting. Only the strings inside
/// are told apart — a markup marker inside one is text — and the tokens
/// between the brackets are left to `serde_json` once it ends.
fn container_end(s: &str) -> ValueEnd {
    let (mut depth, mut in_str, mut esc) = (0usize, false, false);
    for (i, c) in s.bytes().enumerate() {
        if in_str {
            if esc {
                esc = false;
            } else {
                match c {
                    b'\\' => esc = true,
                    b'"' => in_str = false,
                    _ => {}
                }
            }
        } else {
            match c {
                b'"' => in_str = true,
                b'{' | b'[' => depth += 1,
                b'}' | b']' => {
                    depth -= 1;
                    if depth == 0 {
                        return ValueEnd::Complete(i + 1);
                    }
                }
                _ => {}
            }
        }
    }
    ValueEnd::Incomplete
}

/// The word `true`, `false` or `null` at the start of `s` (its first letter
/// there): where it ends — the word's own length, whatever follows it — or the
/// first letter that is not the word's.
fn word_end(s: &str, word: &str) -> ValueEnd {
    match s.bytes().zip(word.bytes()).position(|(a, b)| a != b) {
        Some(at) => ValueEnd::Invalid(at),
        None if s.len() < word.len() => ValueEnd::Incomplete,
        None => ValueEnd::Complete(word.len()),
    }
}

/// The number at the start of `s` (a `-` or a digit there): where it ends. A
/// number's parts grow while a digit, `.` or `e` extends them, so while the
/// text goes on its end waits ([`ValueEnd::Incomplete`]); a part left short by
/// a codepoint that cannot extend it is [`ValueEnd::Invalid`] at it, and at
/// the end of the stream (`at_end`) a complete number ends there.
fn number_end(s: &str, at_end: bool) -> ValueEnd {
    let b = s.as_bytes();
    let mut i = usize::from(b.first() == Some(&b'-'));
    let digits = |i: &mut usize| {
        let from = *i;
        while b.get(*i).is_some_and(|c| c.is_ascii_digit()) {
            *i += 1;
        }
        *i - from
    };
    let short_of = |i: usize| {
        if i == b.len() {
            ValueEnd::Incomplete
        } else {
            ValueEnd::Invalid(i)
        }
    };
    if digits(&mut i) == 0 {
        return short_of(i);
    }
    loop {
        match b.get(i) {
            None if at_end => return ValueEnd::Complete(i),
            None => return ValueEnd::Incomplete,
            Some(b'.') => {
                i += 1;
                if digits(&mut i) == 0 {
                    return short_of(i);
                }
            }
            Some(b'e') | Some(b'E') => {
                i += 1;
                if matches!(b.get(i), Some(b'+') | Some(b'-')) {
                    i += 1;
                }
                if digits(&mut i) == 0 {
                    return short_of(i);
                }
            }
            Some(_) => return ValueEnd::Complete(i),
        }
    }
}

/// The call this markup spells, as the parts structured output forces: the
/// inverse of [`MistralScan`]. `[TOOL_CALLS]`, the name, `[ARGS]` and one
/// JSON object of the function's parameters — the template's `arguments|
/// tojson` of an object, which is what a call closes with. Watched, each
/// `[TOOL_CALLS]` the model writes opens one call (the template writes a
/// conversation's calls one opener each).
pub(crate) fn call_parts(funs: &[Fun<'_>], mode: CallMode) -> Result<Call, Refusal> {
    let body = call(funs, "", str::to_owned, |f| {
        let mut parts = vec![Part::Text(ARGS.to_owned())];
        parts.extend(f.object()?);
        Ok(parts)
    })?;
    Ok(Call::opened(CALL_OPEN, mode, body))
}
