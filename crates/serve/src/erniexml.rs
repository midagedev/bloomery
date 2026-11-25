//! ERNIE 5.5's tool calls: the XML markup the checkpoint's chat protocol
//! teaches (its tool list and the assistant `tool_calls` branch), parsed out
//! of the text after the think span.
//!
//! ```text
//! <ernie:tool_call>
//! <invoke name="NAME"><parameter name="KEY">VALUE</parameter>…</invoke>
//! <invoke name="NAME">…</invoke>
//! </ernie:tool_call>
//! ```
//!
//! The protocol writes the calls of a message in one block, an `invoke` a
//! call, and whitespace may stand between the elements (it writes a line
//! feed after the block's opening and after each `invoke`). A value is
//! everything up to the first `</parameter>` after it. What the value is read
//! as the request's tool schema says, the way the key/value markup reads one
//! ([`crate::glmxml`]): an argument whose schema `type` is `"string"` is the
//! raw text, byte for byte; one whose `type` names other types only must be
//! JSON; for any other argument (no schema, no `type`, or a `type` list with
//! `"string"` among others) a value that parses as JSON is that JSON and
//! anything else is the raw text.
//!
//! A call is emitted when its `</invoke>` arrives; its parameters become a
//! JSON object in the model's order, serialized compactly to a string. Text
//! before the first block is content; after a block the whitespace its layout
//! left is nothing, and any other text is content. Markup that does not
//! parse — text where an `invoke`, a parameter, an `invoke`'s close or the
//! block's close belongs, an opening tag that is not `<invoke name="…">` or
//! `<parameter name="…">`, a block that holds no invoke, one parameter twice,
//! the block's close inside a value, a value its schema types JSON that is
//! not JSON — and a block still open when the stream ends are an
//! [`ErnieXmlError`], never content and never dropped; the chat parser reads
//! a generation the token limit cut inside a block as an answer, not as this
//! error ([`crate::dsml::ChatParser::try_finish_at`]). The call a scan has
//! opened and not closed is reported as it is read ([`Scanned::open`]): its
//! name once its opening tag's `">` is read, and the arguments so far, a
//! prefix of the arguments the call closes with, the text of a `"string"`
//! argument as it arrives. Every decision is independent of how the text and
//! special-token offsets were cut into pieces. A `<ernie:tool_call>` or
//! `</ernie:tool_call>` the vocabulary holds as one token is markup only
//! where that token wrote it ([`SpecialTags`]): spelled in ordinary tokens it
//! is text, content outside a block and no close inside one.

use serde_json::Value;

use crate::callstream::{self, Reading, find_from, held, read_value};
use crate::constrain::{
    Arg, Call, CallMode, Ends, Fun, Order, Part, Refusal, Style, call, is_text, object_parameters,
    outside, value,
};
use crate::dsml::{Scanned, ToolCall};
use crate::engine::Tokenizer;
use crate::ernie::{CALL_CLOSE, CALL_OPEN};
use crate::glmxml::{ArgTypes, Kind};
use crate::reasoning::{Marked, SpecialTags, partial_suffix};

/// Opens a call's `invoke`; its name follows as ` name="NAME">`.
const INVOKE: &str = "<invoke";

/// Closes an `invoke`.
const INVOKE_CLOSE: &str = "</invoke>";

/// Opens a parameter; its name follows as ` name="KEY">`.
const PARAM: &str = "<parameter";

/// Closes a parameter: the first one after its value ends the value.
const PARAM_CLOSE: &str = "</parameter>";

/// The name attribute of an opening tag, after the word and a whitespace.
const NAME_ATTR: &str = "name=\"";

/// How much of the offending text an error quotes.
const QUOTE: usize = 40;

/// A generation whose tool-call markup does not parse. `index` counts calls
/// in the generation from 0.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ErnieXmlError {
    /// The stream ended inside a block.
    #[error(
        "tool call block {index} is not closed by {CALL_CLOSE} when the generation ends: {text:?}"
    )]
    Unterminated { index: usize, text: String },
    /// The block's opening not followed by an `invoke` or the block's close.
    #[error("tool call block {index} opens no invoke: {found:?}")]
    NoInvoke { index: usize, found: String },
    /// Something else where the markup has a tag.
    #[error("tool call {index} ({name}): expected {expected}, found {found:?}")]
    Unexpected {
        index: usize,
        name: String,
        expected: &'static str,
        found: String,
    },
    /// A block that closed holding no invoke.
    #[error("tool call block {index} holds no invoke")]
    EmptyBlock { index: usize },
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
        ty: String,
        value: String,
    },
}

impl ErnieXmlError {
    /// The text of the block the generation ended inside, when that is the
    /// whole fault: markup that read as the protocol's so far and was not
    /// closed.
    pub(crate) fn unclosed(&self) -> Option<String> {
        match self {
            ErnieXmlError::Unterminated { text, .. } => Some(text.clone()),
            _ => None,
        }
    }
}

/// The first characters of `s`, for an error.
fn quote(s: &str) -> String {
    s.chars().take(QUOTE).collect()
}

/// How the opening tag that starts with `word` reads at `at` of `m`: the
/// word, a whitespace, `name="NAME"` and `>`, the name any text but a quote
/// and not empty.
enum Tag {
    /// The text so far is a start of it: nothing is decided yet.
    Wait,
    /// The whole tag, the name it carries and where the text after it starts.
    Open { name: String, after: usize },
    /// The text is not this tag.
    Not,
}

/// The opening tag `word` at `at` of `m`: `<word`, at least one whitespace,
/// `name="NAME">` ([`Tag`]). A `word` the vocabulary holds as one token is
/// the tag only where that token wrote it.
fn opening(sp: &SpecialTags, m: &Marked, at: usize, word: &str) -> Tag {
    let text = m.text.as_str();
    if held(word, &text[at..]) {
        return Tag::Wait;
    }
    if !sp.at(m, at, word) {
        return Tag::Not;
    }
    let mut at = at + word.len();
    let tail = &text[at..];
    let ws = tail.len() - tail.trim_start().len();
    if ws == 0 {
        return if tail.is_empty() { Tag::Wait } else { Tag::Not };
    }
    at += ws;
    let tail = &text[at..];
    if held(NAME_ATTR, tail) {
        return Tag::Wait;
    }
    let Some(after) = tail.strip_prefix(NAME_ATTR) else {
        return Tag::Not;
    };
    let Some(end) = after.find('"') else {
        return Tag::Wait;
    };
    let name = &after[..end];
    if name.is_empty() {
        return Tag::Not;
    }
    let close = &after[end + 1..];
    if close.is_empty() {
        return Tag::Wait;
    }
    if !close.starts_with('>') {
        return Tag::Not;
    }
    Tag::Open {
        name: name.to_owned(),
        after: at + NAME_ATTR.len() + end + 2,
    }
}

/// Where `tag` starts in `m` at or after `*searched`: the first one that is
/// the tag where its token wrote it, where it is one; text spelling it is
/// stepped over. `None` moves `*searched` past the text that can no longer
/// begin it.
fn find_tag(sp: &SpecialTags, m: &Marked, searched: &mut usize, tag: &str) -> Option<usize> {
    loop {
        let at = find_from(&m.text, searched, tag)?;
        if sp.at(m, at, tag) {
            return Some(at);
        }
        *searched = at + tag.len();
    }
}

/// What a step of the open block read.
enum Step {
    /// The text so far is a start of what comes: nothing is decided yet.
    Wait,
    /// An `invoke` closed: the call it spells.
    Call(ToolCall),
    /// The block's close: the block ended.
    Block,
}

/// Streaming scanner over the text after the think span.
#[derive(Clone, Debug)]
pub struct ErnieScan {
    types: ArgTypes,
    /// Inside a block (its opening consumed, the buffer its text since).
    inside: bool,
    /// A block closed: only the layout may follow before the next opens, and
    /// it is consumed, never content.
    after_block: bool,
    /// The calls the open block closed.
    block_calls: usize,
    /// Text not yet decided.
    buf: Marked,
    /// The block's tags read only where their special tokens wrote them.
    specials: SpecialTags,
    calls: usize,
    /// The block's text, as far as it is read.
    read: CallRead,
}

impl ErnieScan {
    /// A scanner that types arguments by `types`.
    #[must_use]
    pub fn new(types: ArgTypes) -> Self {
        ErnieScan {
            types,
            inside: false,
            after_block: false,
            block_calls: 0,
            buf: Marked::default(),
            specials: SpecialTags::default(),
            calls: 0,
            read: CallRead::default(),
        }
    }

    /// Reads the block's tags only where their special tokens wrote them, for
    /// those `tok` holds as one special token.
    pub(crate) fn read_specials(&mut self, tok: &dyn Tokenizer) {
        self.specials = SpecialTags::of(tok, &[CALL_OPEN, CALL_CLOSE]);
    }

    /// Feeds text; returns the content that may be sent and the calls that
    /// closed.
    pub fn push(&mut self, piece: &str) -> Result<Scanned, ErnieXmlError> {
        Ok(self
            .push_stream(piece, &[])?
            .whole(&self.read.call, self.calls))
    }

    /// [`ErnieScan::push`] of text whose special tokens' texts are at
    /// `specials` (byte offset and length), reporting the call text as
    /// `advances`, each piece's new arguments, instead of the open call's
    /// whole text.
    pub(crate) fn push_stream(
        &mut self,
        piece: &str,
        specials: &[(usize, usize)],
    ) -> Result<Scanned, ErnieXmlError> {
        self.buf.push(piece, specials);
        let mut out = Scanned::default();
        let advances = out.advances.insert(Vec::new());
        loop {
            if self.inside {
                match self.read.advance(
                    &self.buf,
                    &self.specials,
                    self.calls,
                    &self.types,
                    self.block_calls,
                    false,
                )? {
                    Step::Wait => {
                        self.read.call.send(self.calls, advances);
                        return Ok(out);
                    }
                    Step::Call(call) => {
                        let at = self.read.at;
                        std::mem::take(&mut self.read).call.close(&call, advances);
                        self.read.at = at;
                        out.calls.push(call);
                        self.calls += 1;
                        self.block_calls += 1;
                    }
                    Step::Block => {
                        let used = self.read.at + CALL_CLOSE.len();
                        self.buf.take_front(used);
                        self.inside = false;
                        self.after_block = true;
                    }
                }
            } else if self.after_block {
                let ws = self.buf.text.len() - self.buf.text.trim_start().len();
                let rest = &self.buf.text[ws..];
                if held(CALL_OPEN, rest) {
                    return Ok(out);
                }
                if self.specials.at(&self.buf, ws, CALL_OPEN) {
                    self.buf.take_front(ws + CALL_OPEN.len());
                    self.open_block();
                } else {
                    // Text after the block: it stays content, its leading
                    // whitespace with it.
                    self.after_block = false;
                }
            } else if let Some(at) = self.specials.find(&self.buf, CALL_OPEN) {
                self.buf.take_text_into(at, &mut out.content);
                self.buf.take_front(CALL_OPEN.len());
                self.open_block();
            } else {
                let upto = self.buf.text.len() - partial_suffix(&self.buf.text, CALL_OPEN);
                self.buf.take_text_into(upto, &mut out.content);
                return Ok(out);
            }
        }
    }

    /// The buffer's text is a block's since its opening.
    fn open_block(&mut self) {
        self.inside = true;
        self.after_block = false;
        self.block_calls = 0;
        self.read = CallRead::default();
    }

    /// Releases what is held at the end of the stream; a block still open is
    /// an error (the markup error its text already shows, else
    /// [`ErnieXmlError::Unterminated`]), an incomplete opening tag is text,
    /// and the whitespace a closed block's layout left is nothing.
    pub fn finish(&mut self) -> Result<Scanned, ErnieXmlError> {
        if self.inside {
            self.read.advance(
                &self.buf,
                &self.specials,
                self.calls,
                &self.types,
                self.block_calls,
                true,
            )?;
            let rest = std::mem::take(&mut self.buf).text;
            return Err(ErnieXmlError::Unterminated {
                index: self.calls,
                text: format!("{CALL_OPEN}{rest}"),
            });
        }
        let rest = std::mem::take(&mut self.buf).text;
        if self.after_block && rest.trim().is_empty() {
            return Ok(Scanned::default());
        }
        Ok(Scanned {
            content: rest,
            ..Scanned::default()
        })
    }
}

/// The open block's reading: the call being read, as far as it goes, and
/// where each stage's text starts and was searched to. Only the text each
/// push adds is searched again.
#[derive(Clone, Debug, Default)]
struct CallRead {
    call: Reading,
    /// The parameters the call read, to refuse one twice.
    keys: Vec<String>,
    /// Where the stage's text starts, in the block's text.
    at: usize,
    /// The value being read: its parameter, where the value starts, how far
    /// its close has been searched for, and how far the block's close has.
    value: Option<(String, usize, usize, usize)>,
}

impl CallRead {
    /// Reads the block's text `m` on from where the last read stopped.
    /// `index` is the call's place in the generation, `block_calls` the
    /// calls the block closed, and `end` that the stream ended: a fault the
    /// text shows is named then however short its quote.
    fn advance(
        &mut self,
        m: &Marked,
        sp: &SpecialTags,
        index: usize,
        types: &ArgTypes,
        block_calls: usize,
        end: bool,
    ) -> Result<Step, ErnieXmlError> {
        let s = m.text.as_str();
        let settled = |found: &str| end || found.chars().nth(QUOTE - 1).is_some();
        loop {
            if let Some(value) = self.value.take() {
                match self.read_parameter(s, m, sp, index, types, value)? {
                    None => return Ok(Step::Wait),
                    Some(after) => self.at = after,
                }
                continue;
            }
            let tail = &s[self.at..];
            self.at += tail.len() - tail.trim_start().len();
            let rest = &s[self.at..];
            if rest.is_empty() {
                return Ok(Step::Wait);
            }
            if self.call.name.is_none() {
                if sp.at(m, self.at, CALL_CLOSE) {
                    if block_calls == 0 {
                        return Err(ErnieXmlError::EmptyBlock { index });
                    }
                    return Ok(Step::Block);
                }
                if held(CALL_CLOSE, rest) {
                    return Ok(Step::Wait);
                }
                match opening(sp, m, self.at, INVOKE) {
                    Tag::Wait => return Ok(Step::Wait),
                    Tag::Open { name, after } => {
                        self.call.name = Some(name);
                        self.call.args.push('{');
                        self.at = after;
                    }
                    Tag::Not if settled(rest) => {
                        return Err(ErnieXmlError::NoInvoke {
                            index,
                            found: quote(rest),
                        });
                    }
                    Tag::Not => return Ok(Step::Wait),
                }
            } else {
                let name = self.call.name.clone().unwrap_or_default();
                let unexpected =
                    |expected: &'static str, found: String| ErnieXmlError::Unexpected {
                        index,
                        name: name.clone(),
                        expected,
                        found,
                    };
                if sp.at(m, self.at, INVOKE_CLOSE) {
                    self.at += INVOKE_CLOSE.len();
                    // The object closes here: what was streamed is its prefix.
                    self.call.args.push('}');
                    return Ok(Step::Call(ToolCall {
                        index,
                        name,
                        arguments: self.call.args.clone(),
                    }));
                }
                if sp.at(m, self.at, CALL_CLOSE) {
                    return Err(unexpected(INVOKE_CLOSE, CALL_CLOSE.to_owned()));
                }
                if held(INVOKE_CLOSE, rest) {
                    return Ok(Step::Wait);
                }
                match opening(sp, m, self.at, PARAM) {
                    Tag::Wait => return Ok(Step::Wait),
                    Tag::Open { name: key, after } => {
                        if self.keys.contains(&key) {
                            return Err(ErnieXmlError::DuplicateParameter { index, name, key });
                        }
                        self.keys.push(key.clone());
                        if types.kind(&name, &key) == Some(&Kind::String) {
                            callstream::Args::key(&mut self.call.args, &key);
                            self.call.args.push('"');
                        }
                        self.value = Some((key, after, after, after));
                        self.at = after;
                    }
                    Tag::Not if settled(rest) => {
                        return Err(unexpected(
                            "</invoke>, or <parameter name=\"KEY\">",
                            quote(rest),
                        ));
                    }
                    Tag::Not => return Ok(Step::Wait),
                }
            }
        }
    }

    /// Reads on from the parameter `value` names: `Ok(None)` while its close
    /// has not come (the reading put back where it stopped), `Ok(Some(at))`
    /// with the parameter complete, the text after its close starting at
    /// `at`.
    fn read_parameter(
        &mut self,
        s: &str,
        m: &Marked,
        sp: &SpecialTags,
        index: usize,
        types: &ArgTypes,
        value: (String, usize, usize, usize),
    ) -> Result<Option<usize>, ErnieXmlError> {
        let (key, start, mut searched, mut closing) = value;
        let name = self.call.name.as_deref().unwrap_or_default().to_owned();
        let kind = types.kind(&name, &key);
        let text = kind == Some(&Kind::String);
        let close = read_value(
            s,
            &mut searched,
            PARAM_CLOSE,
            CALL_CLOSE,
            text.then_some(&mut self.call.args),
        );
        if let Some(block) = find_tag(sp, m, &mut closing, CALL_CLOSE)
            && close.is_none_or(|p| block < p)
        {
            // The tag alone: what follows it in the same push is not the fault's.
            return Err(ErnieXmlError::Unexpected {
                index,
                name,
                expected: PARAM_CLOSE,
                found: CALL_CLOSE.to_owned(),
            });
        }
        let Some(close) = close else {
            self.value = Some((key, start, searched, closing));
            return Ok(None);
        };
        if text {
            self.call.args.push('"');
        } else {
            let raw = &s[start..close];
            let value = match kind {
                Some(Kind::Json(ty)) => {
                    serde_json::from_str::<Value>(raw).map_err(|_| ErnieXmlError::NotJson {
                        index,
                        name: name.clone(),
                        key: key.clone(),
                        ty: ty.clone(),
                        value: quote(raw),
                    })?
                }
                _ => serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned())),
            };
            callstream::Args::key(&mut self.call.args, &key);
            self.call.args.push_str(&value.to_string());
        }
        Ok(Some(close + PARAM_CLOSE.len()))
    }
}

/// The call this markup spells, as the parts structured output forces: the
/// inverse of [`ErnieScan`]. The block's opening, then an `invoke` a call —
/// `<invoke name="NAME">` and a `<parameter name="KEY">` block per parameter
/// in key order, as the protocol's renderer writes every call the model sees,
/// each optional one left out or written in its place — each value written
/// as the schema reads it (raw text for a string, ended by the first
/// `</parameter>` and holding no block close; JSON for the rest), then
/// `</invoke>` and the line feed after it. The scanner reads a value as text
/// only where the schema says `"type": "string"`, itself or through a `$ref`,
/// so a text-valued parameter that does not say it is refused rather than
/// written as something that could read back as JSON. Forced, one call and
/// the block's close; watched, the block holds one call after another, each
/// held to its schema, until the model closes it.
pub(crate) fn call_parts(funs: &[Fun<'_>], mode: CallMode) -> Result<Call, Refusal> {
    let block = |a: &Arg<'_>| -> Result<Vec<Part>, Refusal> {
        if is_text(&a.schema) && a.schema.get("type").and_then(Value::as_str) != Some("string") {
            return Err(outside(
                &a.path,
                "ERNIE 5.5's tool markup reads a text value as text only under \"type\": \
                 \"string\"; add it",
            ));
        }
        let mut parts = vec![Part::Text(format!("<parameter name=\"{}\">", a.key))];
        let ends = Ends {
            close: PARAM_CLOSE,
            never: &[CALL_CLOSE],
        };
        parts.extend(value(&a.schema, Style::Raw(ends), &a.path)?);
        parts.push(Part::Text(PARAM_CLOSE.to_owned()));
        Ok(parts)
    };
    let name = |name: &str| format!("{name}\">");
    // One call after its function's name: its parameters, `</invoke>` and the line feed the
    // protocol writes after it, which is all that stands between two calls.
    let one = |f: &Fun<'_>| {
        let mut parts = object_parameters(&f.args, Order::Sorted, &block)?;
        parts.push(Part::Text(format!("{INVOKE_CLOSE}\n")));
        Ok(parts)
    };
    let invoke = format!("{INVOKE} {NAME_ATTR}");
    // The first call follows the block's opening and the line feed after it.
    let first = format!("\n{invoke}");
    let close = Part::Text(CALL_CLOSE.to_owned());
    let body = match mode {
        CallMode::Forced => call(funs, &first, name, |f| {
            let mut parts = one(f)?;
            parts.push(close.clone());
            Ok(parts)
        })?,
        CallMode::Watched => {
            // The block's end after a call: the model closes it or opens
            // another invoke, which ends the same way.
            let again = Part::Rep {
                close: vec![close],
                more: call(funs, &invoke, name, one)?,
            };
            call(funs, &first, name, |f| {
                let mut parts = one(f)?;
                parts.push(again.clone());
                Ok(parts)
            })?
        }
    };
    Ok(Call::opened(CALL_OPEN, mode, body))
}
