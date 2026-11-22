//! Inkling 2's block protocol: the parser that turns one generation into `reasoning_content`,
//! `content` and `tool_calls` (`common/parsers/inkling.cpp` in the reference).
//!
//! The assistant turn is a list of blocks, each opened by `<|message_model|>` and closed by
//! `<|end_message|>`; the template ends the prompt at the last message, so the model writes its
//! own first header:
//!
//! ```text
//! <|message_model|><|content_thinking|>chain of thought<|end_message|>
//! <|message_model|><|content_text|>the answer<|end_message|>
//! <|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Seoul"}}<|end_message|>
//! ```
//!
//! A thinking block is the reasoning, a text block the content (blocks of one kind concatenate),
//! and a block headed by a function's name a call whose body is `{"name":NAME,"args":OBJECT}`:
//! the name there is the header's, and the arguments are one JSON object, kept compact in the
//! model's key order. With the reasoning not split out ([`InklingScan::new`] with `false`) a
//! thinking block stays in `content`, its tags with it, as the reference's parser leaves it.
//!
//! Every decision depends on the text only, never on how it was cut into pieces: an open
//! thinking or text block releases its text as it arrives (holding back the tail that could still
//! become `<|end_message|>`), a call is reported as it is read ([`Scanned::open`]: its name once
//! `<|content_invoke_tool_json|>` is whole, and the arguments so far once their first byte is read)
//! and closes when its `<|end_message|>` is whole. Markup that does not parse — no plain name
//! before `<|content_invoke_tool_json|>`, a body that is not the template's, arguments that are
//! no JSON object — and a call still open when the generation ends are an [`InklingError`],
//! never content and never dropped; the chat parser reads a generation the token limit cut inside
//! a call as an answer, not as this error ([`crate::dsml::ChatParser::try_finish_at`]). Text
//! outside any block is content.

use serde_json::Value;

use crate::callstream::Progress;
use crate::constrain::{Call, CallMode, Fun, Part, Refusal, call};
use crate::dsml::{Scanned, ToolCall};
use crate::hermes::{compact_prefix, quote};
use crate::mistral::{ValueEnd, value_end};
use crate::reasoning::partial_suffix;

/// Opens every block of the assistant's turn.
pub const HEADER: &str = "<|message_model|>";
/// Opens a thinking block.
pub const THINK: &str = "<|content_thinking|>";
/// Opens a text block.
pub const TEXT: &str = "<|content_text|>";
/// Between a call's name (the header's) and its body.
pub const TOOL_JSON: &str = "<|content_invoke_tool_json|>";
/// Closes every block.
pub const END: &str = "<|end_message|>";

/// A generation whose block markup does not parse. `index` counts calls in the generation from 0.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InklingError {
    /// The stream ended inside a block header or a call.
    #[error("tool call {index} is not closed when the generation ends: {text:?}")]
    Unterminated { index: usize, text: String },
    /// A block header followed by neither a content token nor a plain name and
    /// `<|content_invoke_tool_json|>`.
    #[error("tool call {index} names no function before {TOOL_JSON}: {found:?}")]
    NoName { index: usize, found: String },
    /// A call whose body is not `{"name":NAME,"args":OBJECT}` and its end.
    #[error("tool call {index} ({name}) is not the template's call body: {found:?}")]
    BadCall {
        index: usize,
        name: String,
        found: String,
    },
}

impl InklingError {
    /// The text of the call the generation ended inside, when that is the whole fault.
    pub(crate) fn unclosed(&self) -> Option<String> {
        match self {
            InklingError::Unterminated { text, .. } => Some(text.clone()),
            _ => None,
        }
    }
}

/// Where the scan is in the turn.
#[derive(Clone, Debug, PartialEq, Eq)]
enum State {
    /// Between blocks, or before the first.
    Idle,
    /// A header is read; what kind of block it opens is not.
    Kind,
    /// Inside a thinking block.
    Thinking,
    /// Inside a text block.
    Text,
    /// A call's name, from after the header.
    Name,
    /// The body's fixed opening, `{"name":NAME,"args":`.
    Prefix { name: String },
    /// The arguments' object.
    Args { name: String },
    /// The object read (`raw` as the model wrote it, `args` compact), the body's `}` and
    /// `<|end_message|>` to come.
    Close {
        name: String,
        args: String,
        raw: String,
    },
}

/// Streaming scanner over one generation.
#[derive(Clone, Debug)]
pub struct InklingScan {
    /// Whether the thinking blocks are the reasoning (else content, tags included).
    extract: bool,
    state: State,
    /// Text not yet decided.
    buf: String,
    calls: usize,
}

impl InklingScan {
    /// A scan; `extract` splits the thinking blocks out as the reasoning.
    #[must_use]
    pub fn new(extract: bool) -> Self {
        InklingScan {
            extract,
            state: State::Idle,
            buf: String::new(),
            calls: 0,
        }
    }

    /// Feeds text; returns what may be sent and the calls that closed.
    pub fn push(&mut self, piece: &str) -> Result<Scanned, InklingError> {
        self.buf.push_str(piece);
        let mut out = Scanned::default();
        while self.step(&mut out)? {}
        out.open = self.open_call();
        Ok(out)
    }

    /// Releases what is held at the end of the stream: a block's text goes to its channel, text
    /// between blocks is content, and a header or a call the stream ends inside is the error.
    pub fn finish(&mut self) -> Result<Scanned, InklingError> {
        let mut out = Scanned::default();
        let buf = std::mem::take(&mut self.buf);
        let unterminated = |text: String| InklingError::Unterminated {
            index: self.calls,
            text,
        };
        match std::mem::replace(&mut self.state, State::Idle) {
            State::Idle => out.content = buf,
            State::Thinking => self.thinking_text(&buf, &mut out),
            State::Text => out.content = buf,
            State::Kind => return Err(unterminated(format!("{HEADER}{buf}"))),
            State::Name => return Err(unterminated(format!("{HEADER}{buf}"))),
            State::Prefix { name } => {
                return Err(unterminated(format!("{HEADER}{name}{TOOL_JSON}{buf}")));
            }
            State::Args { name } => {
                return Err(unterminated(format!(
                    "{HEADER}{name}{TOOL_JSON}{}{buf}",
                    prefix(&name)
                )));
            }
            State::Close { name, raw, .. } => {
                return Err(unterminated(format!(
                    "{HEADER}{name}{TOOL_JSON}{}{raw}{buf}",
                    prefix(&name)
                )));
            }
        }
        Ok(out)
    }

    /// One step of the state machine; whether it consumed anything and may go on.
    fn step(&mut self, out: &mut Scanned) -> Result<bool, InklingError> {
        match self.state.clone() {
            State::Idle => {
                let Some(at) = self.buf.find(HEADER) else {
                    let upto = self.buf.len() - partial_suffix(&self.buf, HEADER);
                    out.content.push_str(&self.buf[..upto]);
                    self.buf.drain(..upto);
                    return Ok(false);
                };
                out.content.push_str(&self.buf[..at]);
                self.buf.drain(..at + HEADER.len());
                self.state = State::Kind;
                Ok(true)
            }
            State::Kind => {
                if let Some(rest) = self.buf.strip_prefix(THINK) {
                    self.buf = rest.to_owned();
                    if !self.extract {
                        out.content.push_str(THINK);
                    }
                    self.state = State::Thinking;
                } else if let Some(rest) = self.buf.strip_prefix(TEXT) {
                    self.buf = rest.to_owned();
                    self.state = State::Text;
                } else if THINK.starts_with(&self.buf) || TEXT.starts_with(&self.buf) {
                    return Ok(false);
                } else {
                    self.state = State::Name;
                }
                Ok(true)
            }
            State::Thinking => {
                let (text, ended) = self.block_text();
                self.thinking_text(&text, out);
                if ended && !self.extract {
                    out.content.push_str(END);
                }
                Ok(ended)
            }
            State::Text => {
                let (text, ended) = self.block_text();
                out.content.push_str(&text);
                Ok(ended)
            }
            State::Name => {
                let Some(at) = self.buf.find(TOOL_JSON) else {
                    let upto = self.buf.len() - partial_suffix(&self.buf, TOOL_JSON);
                    if self.buf[..upto].contains(['<', '\n']) {
                        return Err(self.no_name(&self.buf[..upto]));
                    }
                    return Ok(false);
                };
                let name = self.buf[..at].to_owned();
                if name.is_empty() || name.contains(['<', '\n']) {
                    return Err(self.no_name(&name));
                }
                self.buf.drain(..at + TOOL_JSON.len());
                self.state = State::Prefix { name };
                Ok(true)
            }
            State::Prefix { name } => {
                let want = prefix(&name);
                if let Some(rest) = self.buf.strip_prefix(&want) {
                    self.buf = rest.to_owned();
                    self.state = State::Args { name };
                    Ok(true)
                } else if want.starts_with(&self.buf) {
                    Ok(false)
                } else {
                    Err(self.bad_call(&name, through_mismatch(&want, &self.buf)))
                }
            }
            State::Args { name } => match value_end(&self.buf, false) {
                invalid @ ValueEnd::Invalid(_) => {
                    let found = invalid.fault(&self.buf);
                    Err(self.bad_call(&name, &found))
                }
                ValueEnd::Incomplete => Ok(false),
                ValueEnd::Complete(end) => {
                    let raw = self.buf[..end].to_owned();
                    let args = match serde_json::from_str::<Value>(&raw) {
                        Ok(v @ Value::Object(_)) => v.to_string(),
                        _ => return Err(self.bad_call(&name, raw.trim())),
                    };
                    self.buf.drain(..end);
                    self.state = State::Close { name, args, raw };
                    Ok(true)
                }
            },
            State::Close { name, args, .. } => {
                let want = format!("}}{END}");
                if self.buf.starts_with(&want) {
                    self.buf.drain(..want.len());
                    out.calls.push(ToolCall {
                        index: self.calls,
                        name,
                        arguments: args,
                    });
                    self.calls += 1;
                    self.state = State::Idle;
                    Ok(true)
                } else if want.starts_with(&self.buf) {
                    Ok(false)
                } else {
                    Err(self.bad_call(&name, through_mismatch(&want, &self.buf)))
                }
            }
        }
    }

    /// The text of an open block that may be sent: up to its end, which is consumed (`true`), or
    /// all but the tail that could still become it.
    fn block_text(&mut self) -> (String, bool) {
        if let Some(at) = self.buf.find(END) {
            let text = self.buf[..at].to_owned();
            self.buf.drain(..at + END.len());
            self.state = State::Idle;
            return (text, true);
        }
        let upto = self.buf.len() - partial_suffix(&self.buf, END);
        let text = self.buf[..upto].to_owned();
        self.buf.drain(..upto);
        (text, false)
    }

    /// Thinking text to its channel.
    fn thinking_text(&self, text: &str, out: &mut Scanned) {
        if self.extract {
            out.reasoning.push_str(text);
        } else {
            out.content.push_str(text);
        }
    }

    /// The call open as of now, once its arguments have begun.
    fn open_call(&self) -> Option<Progress> {
        let (name, args) = match &self.state {
            State::Args { name } => (name, compact_prefix(&self.buf)),
            State::Close { name, args, .. } => (name, args.clone()),
            _ => return None,
        };
        (!args.is_empty()).then(|| Progress {
            index: self.calls,
            name: name.clone(),
            args,
        })
    }

    fn no_name(&self, found: &str) -> InklingError {
        let end = found.find(['<', '\n']).map_or(found.len(), |at| at + 1);
        InklingError::NoName {
            index: self.calls,
            found: quote(&found[..end]),
        }
    }

    fn bad_call(&self, name: &str, found: &str) -> InklingError {
        InklingError::BadCall {
            index: self.calls,
            name: name.to_owned(),
            found: quote(found),
        }
    }
}

/// The start of `got` through the first codepoint that is not where `want` has the same: what an
/// error quotes, so every cut of one text quotes the same.
fn through_mismatch<'a>(want: &str, got: &'a str) -> &'a str {
    let at = got
        .char_indices()
        .find(|&(i, c)| !want.get(i..).is_some_and(|w| w.starts_with(c)))
        .map_or(got.len(), |(i, c)| i + c.len_utf8());
    &got[..at]
}

/// The body's fixed opening for a call of `name`.
fn prefix(name: &str) -> String {
    format!(
        "{{\"name\":{},\"args\":",
        serde_json::to_string(name).expect("a string is JSON")
    )
}

/// The call this markup spells, as the parts structured output forces: the inverse of
/// [`InklingScan`] — the header, the function's name, `<|content_invoke_tool_json|>`, the body
/// with the arguments as one JSON object of the function's parameters, and the block's end. The
/// header also opens thinking and text blocks, so the call has no opener to watch for and a
/// watched one builds nothing.
pub(crate) fn call_parts(funs: &[Fun<'_>], mode: CallMode) -> Result<Call, Refusal> {
    if mode == CallMode::Watched {
        return Ok(Call::unopened(Vec::new()));
    }
    let parts = call(funs, HEADER, str::to_owned, |f| {
        let mut parts = vec![Part::Text(format!("{TOOL_JSON}{}", prefix(f.name)))];
        parts.extend(f.object()?);
        parts.push(Part::Text(format!("}}{END}")));
        Ok(parts)
    })?;
    Ok(Call::unopened(parts))
}
