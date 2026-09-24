//! DSML tool calls for the V4.1 chat template, and the parser that turns one
//! generation into `reasoning_content`, `content` and `tool_calls`.
//!
//! The markup is the template's own (its `tools_header` and the assistant
//! `tool_calls` branch):
//!
//! ```text
//! \n\n<｜DSML｜tool_calls>
//! <｜DSML｜invoke name="NAME">
//! <｜DSML｜parameter name="KEY" string="true">RAW TEXT</｜DSML｜parameter>
//! <｜DSML｜parameter name="KEY" string="false">JSON</｜DSML｜parameter>
//! </｜DSML｜invoke>
//! </｜DSML｜tool_calls>
//! ```
//!
//! Only text after the think span is scanned: DSML inside reasoning is reasoning.
//! A call is emitted when its `</｜DSML｜invoke>` arrives. Arguments become a JSON
//! object in parameter order, serialized to a string; a `string="false"` value that
//! is not valid JSON is kept as a JSON string. Markup that does not parse, and a
//! block still open when the stream ends, go to `content` as the raw text.
//!
//! Every decision depends on the text only, never on how it was cut into pieces,
//! so a stream and a whole response parse to the same message.

use crate::reasoning::{ReasoningFormat, Split, THINK_CLOSE, ThinkSplit, partial_suffix};

/// The template's `dsml_token`.
pub const DSML: &str = "｜DSML｜";
/// Opens a tool-call block.
pub const CALLS_OPEN: &str = "<｜DSML｜tool_calls>";
/// Closes a tool-call block.
pub const CALLS_CLOSE: &str = "</｜DSML｜tool_calls>";
const SEPARATED_OPEN: &str = "\n\n<｜DSML｜tool_calls>";
const SEPARATOR: &str = "\n\n";
const INVOKE_OPEN: &str = "<｜DSML｜invoke name=\"";
const INVOKE_CLOSE: &str = "</｜DSML｜invoke>";
const PARAM_OPEN: &str = "<｜DSML｜parameter name=\"";
const PARAM_CLOSE: &str = "</｜DSML｜parameter>";

/// One parsed call. `index` counts calls in the generation from 0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    pub index: usize,
    pub name: String,
    /// A JSON object, as text (OpenAI's `function.arguments`).
    pub arguments: String,
}

/// What a scan step produced.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Scanned {
    pub content: String,
    pub calls: Vec<ToolCall>,
}

/// Streaming DSML scanner over the text after the think span.
#[derive(Debug, Default)]
pub struct DsmlScan {
    /// Inside a block: the text consumed since it opened or since its last call,
    /// which becomes content if the rest does not parse.
    block: Option<String>,
    /// Text not yet decided.
    buf: String,
    calls: usize,
}

enum Invoke {
    /// A complete call and the bytes it took.
    Call(String, String, usize),
    /// Not enough text yet.
    Wait,
    /// Not DSML the template emits.
    Malformed,
}

impl DsmlScan {
    /// Feeds text.
    pub fn push(&mut self, piece: &str) -> Scanned {
        self.buf.push_str(piece);
        let mut out = Scanned::default();
        loop {
            let progressed = match self.block.take() {
                None => self.text_step(&mut out),
                Some(raw) => self.block_step(raw, &mut out),
            };
            if !progressed {
                return out;
            }
        }
    }

    /// Releases what is held at the end of the stream; an open block is content.
    pub fn finish(&mut self) -> Scanned {
        let mut content = self.block.take().unwrap_or_default();
        content.push_str(&std::mem::take(&mut self.buf));
        Scanned {
            content,
            calls: Vec::new(),
        }
    }

    /// Outside a block: sends text up to a block opening or a tail that could
    /// become one. Returns whether a block opened.
    fn text_step(&mut self, out: &mut Scanned) -> bool {
        if let Some(at) = self.buf.find(CALLS_OPEN) {
            let before = &self.buf[..at];
            let (text, sep) = match before.strip_suffix(SEPARATOR) {
                Some(t) => (t, SEPARATOR),
                None => (before, ""),
            };
            out.content.push_str(text);
            self.block = Some(format!("{sep}{CALLS_OPEN}"));
            self.buf.drain(..at + CALLS_OPEN.len());
            return true;
        }
        let keep =
            partial_suffix(&self.buf, SEPARATED_OPEN).max(partial_suffix(&self.buf, CALLS_OPEN));
        let upto = self.buf.len() - keep;
        out.content.push_str(&self.buf[..upto]);
        self.buf.drain(..upto);
        false
    }

    /// Inside a block: takes one invoke or the block's close. Returns whether it
    /// consumed anything.
    fn block_step(&mut self, mut raw: String, out: &mut Scanned) -> bool {
        let lead = self.buf.len() - self.buf.trim_start().len();
        let rest = &self.buf[lead..];
        if rest.starts_with(CALLS_CLOSE) {
            self.buf.drain(..lead + CALLS_CLOSE.len());
            return true;
        }
        if rest.starts_with(INVOKE_OPEN) {
            match parse_invoke(rest) {
                Invoke::Call(name, arguments, used) => {
                    out.calls.push(ToolCall {
                        index: self.calls,
                        name,
                        arguments,
                    });
                    self.calls += 1;
                    self.buf.drain(..lead + used);
                    self.block = Some(String::new());
                    return true;
                }
                Invoke::Wait => {
                    self.block = Some(raw);
                    return false;
                }
                Invoke::Malformed => {}
            }
        } else if rest.is_empty() || CALLS_CLOSE.starts_with(rest) || INVOKE_OPEN.starts_with(rest)
        {
            self.block = Some(raw);
            return false;
        }
        // Not the template's markup: what the block consumed is text, and the
        // rest is scanned as text again.
        raw.push_str(&self.buf[..lead]);
        self.buf.drain(..lead);
        out.content.push_str(&raw);
        true
    }
}

/// Parses `<｜DSML｜invoke name="…">…</｜DSML｜invoke>` at the start of `s`.
fn parse_invoke(s: &str) -> Invoke {
    let Some(end) = s.find(INVOKE_CLOSE) else {
        return Invoke::Wait;
    };
    let body = &s[INVOKE_OPEN.len()..end];
    let Some(name_end) = body.find("\">") else {
        return Invoke::Malformed;
    };
    let name = &body[..name_end];
    if name.is_empty() || name.contains(['"', '\n', '<']) {
        return Invoke::Malformed;
    }
    let mut params: Vec<(String, String)> = Vec::new();
    let mut rest = &body[name_end + 2..];
    loop {
        rest = rest.trim_start();
        if rest.is_empty() {
            break;
        }
        let Some(p) = rest.strip_prefix(PARAM_OPEN) else {
            return Invoke::Malformed;
        };
        let Some(key_end) = p.find('"') else {
            return Invoke::Malformed;
        };
        let key = &p[..key_end];
        let p = &p[key_end..];
        let (is_string, p) = if let Some(v) = p.strip_prefix("\" string=\"true\">") {
            (true, v)
        } else if let Some(v) = p.strip_prefix("\" string=\"false\">") {
            (false, v)
        } else {
            return Invoke::Malformed;
        };
        let Some(val_end) = p.find(PARAM_CLOSE) else {
            return Invoke::Malformed;
        };
        let value = json_value(&p[..val_end], is_string);
        match params.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = value,
            None => params.push((key.to_owned(), value)),
        }
        rest = &p[val_end + PARAM_CLOSE.len()..];
    }
    let fields: Vec<String> = params
        .iter()
        .map(|(k, v)| format!("{}:{v}", json_string(k)))
        .collect();
    Invoke::Call(
        name.to_owned(),
        format!("{{{}}}", fields.join(",")),
        end + INVOKE_CLOSE.len(),
    )
}

fn json_string(s: &str) -> String {
    serde_json::Value::String(s.to_owned()).to_string()
}

/// A parameter value as JSON text: `string="true"` is the raw text, `string="false"`
/// is JSON (kept as a string when it does not parse).
fn json_value(raw: &str, is_string: bool) -> String {
    if !is_string && let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) {
        return v.to_string();
    }
    json_string(raw)
}

/// One generation's message, or the part of it one push added.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Message {
    /// `reasoning_content` (empty with [`ReasoningFormat::None`]).
    pub reasoning: String,
    pub content: String,
    pub calls: Vec<ToolCall>,
}

impl Message {
    fn append(&mut self, d: &Message) {
        self.reasoning.push_str(&d.reasoning);
        self.content.push_str(&d.content);
        self.calls.extend(d.calls.iter().cloned());
    }

    /// Nothing to send.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.reasoning.is_empty() && self.content.is_empty() && self.calls.is_empty()
    }
}

/// The streaming parser for one chat generation.
///
/// With [`ReasoningFormat::None`] and no tools the text passes through unchanged,
/// piece for piece. With tools, the think span is tracked in either format so DSML
/// inside it is never parsed; with `None` the span's text, its `</think>` included,
/// stays in `content`.
#[derive(Debug)]
pub struct ChatParser {
    format: ReasoningFormat,
    /// `None` when neither a split nor a scan is asked for.
    think: Option<ThinkSplit>,
    dsml: Option<DsmlScan>,
    total: Message,
}

impl ChatParser {
    /// A parser for a generation that follows `prompt` (the rendered template).
    /// `tools` turns the DSML scan on.
    #[must_use]
    pub fn new(prompt: &str, format: ReasoningFormat, tools: bool) -> Self {
        let split = format == ReasoningFormat::Deepseek || tools;
        ChatParser {
            format,
            think: split.then(|| ThinkSplit::for_prompt(prompt)),
            dsml: tools.then(DsmlScan::default),
            total: Message::default(),
        }
    }

    /// Parses a whole generation.
    #[must_use]
    pub fn parse(prompt: &str, format: ReasoningFormat, tools: bool, text: &str) -> Message {
        let mut p = ChatParser::new(prompt, format, tools);
        let _ = p.push(text);
        let _ = p.finish();
        p.total
    }

    /// Feeds generated text; returns what may be sent now.
    pub fn push(&mut self, text: &str) -> Message {
        let split = match &mut self.think {
            Some(t) => t.push(text),
            None => Split {
                content: text.to_owned(),
                ..Split::default()
            },
        };
        let dsml = self.dsml.as_mut().map(|d| d.push(&split.content));
        self.emit(split, dsml)
    }

    /// Releases everything held at the end of the generation.
    pub fn finish(&mut self) -> Message {
        // An open span's tail is reasoning; the scanner has seen none of it.
        let split = self
            .think
            .as_mut()
            .map(ThinkSplit::finish)
            .unwrap_or_default();
        let dsml = self.dsml.as_mut().map(DsmlScan::finish);
        self.emit(split, dsml)
    }

    /// The message so far.
    #[must_use]
    pub fn message(&self) -> &Message {
        &self.total
    }

    fn emit(&mut self, split: Split, dsml: Option<Scanned>) -> Message {
        let mut d = Message::default();
        match self.format {
            ReasoningFormat::Deepseek => d.reasoning = split.reasoning,
            ReasoningFormat::None => {
                d.content = split.reasoning;
                if split.closed {
                    d.content.push_str(THINK_CLOSE);
                }
            }
        }
        match dsml {
            Some(s) => {
                d.content.push_str(&s.content);
                d.calls = s.calls;
            }
            None => d.content.push_str(&split.content),
        }
        self.total.append(&d);
        d
    }
}
