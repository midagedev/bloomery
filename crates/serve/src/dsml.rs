//! DSML tool calls for the V4.1 chat template, and the parser that turns one
//! generation into `reasoning_content`, `content` and `tool_calls` under the
//! tool-call markup its template teaches ([`ToolFormat`]: DSML here, GLM's and
//! Solar-Open's in [`crate::glmxml`], Hermes' in [`crate::hermes`], Qwen3-Coder's in
//! [`crate::qwenxml`]).
//!
//! The markup is the template's own (its `tools_header` and the assistant
//! `tool_calls` branch), in one of two spellings ([`Spelling`]):
//!
//! ```text
//! \n\n<｜DSML｜tool_calls>                     spaced: <｜DSML｜ calls>
//! <｜DSML｜invoke name="NAME">                 spaced: <｜DSML｜ invoke name="NAME">
//! <｜DSML｜parameter name="KEY" string="true">RAW TEXT</｜DSML｜parameter>
//! <｜DSML｜parameter name="KEY" string="false">JSON</｜DSML｜parameter>
//! </｜DSML｜invoke>
//! </｜DSML｜tool_calls>
//! ```
//!
//! The spaced tags (V4.1's) carry a space after the DSML token (`parameter` likewise),
//! and their block is `calls`, not `tool_calls`. A scan reads the one spelling it is
//! given, never both.
//!
//! Only text after the think span is scanned: DSML inside reasoning is reasoning.
//! A call is emitted when its `</｜DSML｜invoke>` arrives. Arguments become a JSON
//! object in parameter order, serialized to a string; a `string="false"` value that
//! is not valid JSON is kept as a JSON string. Markup that does not parse, and a
//! block still open when the stream ends, go to `content` as the raw text.
//!
//! The call a scan has opened and not closed is reported as it is read
//! ([`Scanned::open`]): its name once the opening tag is whole, and the arguments
//! read so far, a prefix of the arguments the call closes with
//! ([`crate::callstream`]). The chat parser's scan reads only the text each push
//! adds (`DsmlScan::push_stream`) and reports only the arguments it adds.
//!
//! Every decision depends on the text only, never on how it was cut into pieces,
//! so a stream and a whole response parse to the same message. A parser that
//! streams its calls ([`ChatParser::streaming_calls`]) differs in one thing:
//! DSML that stops being a call after its name went out is an error, not content.

use jinja::{ChatTemplate, TemplateError};
use serde_json::{Map, Value, json};

use crate::callstream::{
    self, Advance, Args, CallPiece, Progress, Reading, Streamer, find_from, held, read_value,
};
use crate::constrain::{Fun, Part, Refusal, Style, call, is_text, value};
use crate::engine::Tokenizer;
use crate::glmxml::{self, ArgTypes, KvError, KvScan, KvSpelling};
use crate::hermes::{self, HermesError, HermesScan};
use crate::qwenxml::{self, Framing, ParamKinds, QwenScan, QwenXmlError};
use crate::reasoning::{ReasoningFormat, Split, ThinkSplit, ThinkTags, partial_suffix};

/// The template's `dsml_token`.
pub const DSML: &str = "｜DSML｜";
const SEPARATOR: &str = "\n\n";

/// How a DSML template spells its tags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spelling {
    /// The word follows the token: `<｜DSML｜tool_calls>`, `<｜DSML｜invoke …>`,
    /// `<｜DSML｜parameter …>` (DeepSeek-V4's).
    Joined,
    /// A space follows the token, and the block is `calls`: `<｜DSML｜ calls>`,
    /// `<｜DSML｜ invoke …>`, `<｜DSML｜ parameter …>` (DeepSeek-V4.1's).
    Spaced,
}

/// The tags of one [`Spelling`].
#[derive(Debug)]
struct Tags {
    /// The block's opening with the blank line the template puts before it.
    separated_open: &'static str,
    calls_open: &'static str,
    calls_close: &'static str,
    invoke_open: &'static str,
    invoke_close: &'static str,
    param_open: &'static str,
    param_close: &'static str,
}

/// The tags of the block, call and parameter words `$calls`, `$invoke` and `$param`.
macro_rules! tags {
    ($calls:literal, $invoke:literal, $param:literal) => {
        Tags {
            separated_open: concat!("\n\n<｜DSML｜", $calls, ">"),
            calls_open: concat!("<｜DSML｜", $calls, ">"),
            calls_close: concat!("</｜DSML｜", $calls, ">"),
            invoke_open: concat!("<｜DSML｜", $invoke, " name=\""),
            invoke_close: concat!("</｜DSML｜", $invoke, ">"),
            param_open: concat!("<｜DSML｜", $param, " name=\""),
            param_close: concat!("</｜DSML｜", $param, ">"),
        }
    };
}

static JOINED_TAGS: Tags = tags!("tool_calls", "invoke", "parameter");
static SPACED_TAGS: Tags = tags!(" calls", " invoke", " parameter");

impl Spelling {
    fn tags(self) -> &'static Tags {
        match self {
            Spelling::Joined => &JOINED_TAGS,
            Spelling::Spaced => &SPACED_TAGS,
        }
    }

    /// The spelling a template `source` that spells the DSML token writes, as
    /// llama.cpp reads it: spaced when the block's opener is built as
    /// `… ' calls>` (single or double quotes), else joined.
    #[must_use]
    pub fn of_template(source: &str) -> Spelling {
        if source.contains("' calls>") || source.contains("\" calls>") {
            Spelling::Spaced
        } else {
            Spelling::Joined
        }
    }
}

/// What a template builds after `dsml_token + '`, joined and spaced.
const RESPELL: [[&str; 2]; 3] = [
    ["dsml_token + 'tool_calls", "dsml_token + ' calls"],
    ["dsml_token + 'invoke", "dsml_token + ' invoke"],
    ["dsml_token + 'parameter", "dsml_token + ' parameter"],
];

/// A chat template `source` with its DSML tags spelled as `to`, and nothing else
/// changed: the way a model's own encoder writes them, which its GGUF's template may
/// not (the V4.1 files carry V4's). `Ok(None)`: the template spells no DSML tag, or
/// spells `to` already. An error: it spells the other spelling in a form this does
/// not read.
pub fn respell(source: &str, to: Spelling) -> Result<Option<String>, TemplateError> {
    if !source.contains(DSML) || Spelling::of_template(source) == to {
        return Ok(None);
    }
    let (from, into) = match to {
        Spelling::Spaced => (0, 1),
        Spelling::Joined => (1, 0),
    };
    // Every use of the token must be one this rewrites: what is left once the token's
    // definition and the rewritten forms are taken out names it nowhere, so no tag is left
    // in the other spelling.
    let mut unread = source.replace(format!("set dsml_token = '{DSML}'").as_str(), "");
    let mut out = source.to_owned();
    for pair in RESPELL {
        unread = unread.replace(pair[from], "");
        out = out.replace(pair[from], pair[into]);
    }
    if unread.contains("dsml_token")
        || unread.contains(DSML)
        || out == source
        || Spelling::of_template(&out) != to
    {
        return Err(TemplateError(format!(
            "the chat template spells its DSML tags in a form this server cannot respell as {to:?}"
        )));
    }
    Ok(Some(out))
}

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
    /// The reasoning the scan itself split out (Muse Spark's `to=self`
    /// span; empty for the scans that leave the split to the think tags).
    pub reasoning: String,
    /// The call the scan has opened and not closed, once its name is read
    /// ([`Progress`]).
    pub open: Option<Progress>,
    /// For a scan that streams by its own cursor, the call text it read this
    /// step, in order ([`Advance`]). `None` for a scan whose `calls` and
    /// `open` are the whole state.
    pub(crate) advances: Option<Vec<Advance>>,
}

impl Scanned {
    /// A step of a scan that reads by its own cursor in the whole-state form:
    /// the call `read` holds, as call `index`, is `open`, and no advances.
    pub(crate) fn whole(mut self, read: &Reading, index: usize) -> Scanned {
        self.advances = None;
        self.open = read.progress(index);
        self
    }
}

/// Streaming DSML scanner over the text after the think span.
#[derive(Clone, Debug)]
pub struct DsmlScan {
    tags: &'static Tags,
    /// Inside a block: the text consumed since it opened or since its last call,
    /// which becomes content if the rest does not parse.
    block: Option<String>,
    /// Text not yet decided.
    buf: String,
    calls: usize,
    /// The invoke open at the start of `buf`, as far as it is read.
    read: InvokeRead,
}

/// The attributes that end a parameter's opening tag: a text value, or JSON.
const STRING_TRUE: &str = "\" string=\"true\">";
const STRING_FALSE: &str = "\" string=\"false\">";

impl DsmlScan {
    /// A scan for a block in `spelling`.
    #[must_use]
    pub fn new(spelling: Spelling) -> Self {
        DsmlScan {
            tags: spelling.tags(),
            block: None,
            buf: String::new(),
            calls: 0,
            read: InvokeRead::default(),
        }
    }

    /// Feeds text.
    pub fn push(&mut self, piece: &str) -> Scanned {
        self.push_stream(piece).whole(&self.read.call, self.calls)
    }

    /// [`DsmlScan::push`] that reads only the text `piece` adds and reports
    /// the open call's new arguments as `advances` instead of its whole text.
    pub(crate) fn push_stream(&mut self, piece: &str) -> Scanned {
        self.buf.push_str(piece);
        let mut out = Scanned {
            advances: Some(Vec::new()),
            ..Scanned::default()
        };
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
            ..Scanned::default()
        }
    }

    /// Outside a block: sends text up to a block opening or a tail that could
    /// become one. Returns whether a block opened.
    fn text_step(&mut self, out: &mut Scanned) -> bool {
        let tags = self.tags;
        if let Some(at) = self.buf.find(tags.calls_open) {
            let before = &self.buf[..at];
            let (text, sep) = match before.strip_suffix(SEPARATOR) {
                Some(t) => (t, SEPARATOR),
                None => (before, ""),
            };
            out.content.push_str(text);
            self.block = Some(format!("{sep}{}", tags.calls_open));
            self.buf.drain(..at + tags.calls_open.len());
            return true;
        }
        let keep = partial_suffix(&self.buf, tags.separated_open)
            .max(partial_suffix(&self.buf, tags.calls_open));
        let upto = self.buf.len() - keep;
        out.content.push_str(&self.buf[..upto]);
        self.buf.drain(..upto);
        false
    }

    /// Inside a block: takes one invoke or the block's close. Returns whether it
    /// consumed anything.
    fn block_step(&mut self, mut raw: String, out: &mut Scanned) -> bool {
        let tags = self.tags;
        let lead = self.buf.len() - self.buf.trim_start().len();
        let rest = &self.buf[lead..];
        if rest.starts_with(tags.calls_close) {
            self.buf.drain(..lead + tags.calls_close.len());
            return true;
        }
        if rest.starts_with(tags.invoke_open) {
            let advances = out.advances.get_or_insert_default();
            let Some(end) = find_from(rest, &mut self.read.closing, tags.invoke_close) else {
                self.read.advance(rest, tags);
                self.read.call.send(self.calls, advances);
                self.block = Some(raw);
                return false;
            };
            let read = std::mem::take(&mut self.read);
            if let Some((name, arguments)) = parse_invoke(&rest[..end], tags) {
                let call = ToolCall {
                    index: self.calls,
                    name,
                    arguments,
                };
                read.call.close(&call, advances);
                out.calls.push(call);
                self.calls += 1;
                self.buf.drain(..lead + end + tags.invoke_close.len());
                self.block = Some(String::new());
                return true;
            }
            read.call.give_back(advances);
        } else if rest.is_empty()
            || tags.calls_close.starts_with(rest)
            || tags.invoke_open.starts_with(rest)
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

/// Parses `s`, an `<｜DSML｜invoke name="…">…` up to its `</｜DSML｜invoke>`, in
/// `tags`' spelling: the call's name and arguments, `None` when it is not DSML
/// the template emits.
fn parse_invoke(s: &str, tags: &Tags) -> Option<(String, String)> {
    let body = &s[tags.invoke_open.len()..];
    let name_end = body.find("\">")?;
    let name = &body[..name_end];
    if !is_name(name) {
        return None;
    }
    let mut params: Vec<(String, String)> = Vec::new();
    let mut rest = &body[name_end + 2..];
    loop {
        rest = rest.trim_start();
        if rest.is_empty() {
            break;
        }
        let p = rest.strip_prefix(tags.param_open)?;
        let key_end = p.find('"')?;
        let key = &p[..key_end];
        let (is_string, p) = string_attr(&p[key_end..])?;
        let val_end = p.find(tags.param_close)?;
        let value = json_value(&p[..val_end], is_string);
        match params.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = value,
            None => params.push((key.to_owned(), value)),
        }
        rest = &p[val_end + tags.param_close.len()..];
    }
    let fields: Vec<String> = params
        .iter()
        .map(|(k, v)| format!("{}:{v}", callstream::string(k)))
        .collect();
    Some((name.to_owned(), format!("{{{}}}", fields.join(","))))
}

/// Whether `name` may name a call in an invoke's opening tag.
fn is_name(name: &str) -> bool {
    !name.is_empty() && !name.contains(['"', '\n', '<'])
}

/// The text after a parameter's name, `p`, read as its opening tag's end:
/// whether the value is text, and the text after the tag.
fn string_attr(p: &str) -> Option<(bool, &str)> {
    p.strip_prefix(STRING_TRUE)
        .map(|v| (true, v))
        .or_else(|| p.strip_prefix(STRING_FALSE).map(|v| (false, v)))
}

/// The invoke a scan has opened and not closed, read from where the last push
/// stopped, the arguments [`parse_invoke`] will close with as a prefix: each
/// complete parameter, and the text of a `string="true"` parameter as it
/// arrives, up to the tail that could still become a tag that ends it.
#[derive(Clone, Debug, Default)]
struct InvokeRead {
    call: Reading,
    stage: InvokeStage,
    /// Where the stage's text starts, in the invoke's text.
    at: usize,
    /// How far the tag that ends the stage has been searched for.
    searched: usize,
    /// How far the invoke's close has been searched for.
    closing: usize,
}

#[derive(Clone, Debug, Default)]
enum InvokeStage {
    /// The call's name, up to the opening tag's `">`.
    #[default]
    Name,
    /// Between parameters.
    Next,
    /// A parameter's name and the rest of its opening tag.
    Key,
    /// The value of the parameter `key`, text or JSON.
    Value { key: String, text: bool },
    /// The text stopped being the template's markup: the close decides.
    Stopped,
}

impl InvokeRead {
    /// Reads `s`, the invoke's text so far (its close not in it), on from
    /// where the last read stopped.
    fn advance(&mut self, s: &str, tags: &Tags) {
        loop {
            let next = match &self.stage {
                InvokeStage::Name => {
                    let start = tags.invoke_open.len();
                    self.searched = self.searched.max(start);
                    let from = self.searched;
                    match find_from(s, &mut self.searched, "\">") {
                        Some(end) if is_name(&s[start..end]) => {
                            self.call.name = Some(s[start..end].to_owned());
                            self.call.args.push('{');
                            self.at = end + 2;
                            InvokeStage::Next
                        }
                        None if !s[from..self.searched].contains(['"', '\n', '<']) => return,
                        _ => InvokeStage::Stopped,
                    }
                }
                InvokeStage::Next => {
                    let tail = &s[self.at..];
                    let rest = tail.trim_start();
                    self.at += tail.len() - rest.len();
                    match rest.strip_prefix(tags.param_open) {
                        Some(p) => {
                            self.at = s.len() - p.len();
                            self.searched = self.at;
                            InvokeStage::Key
                        }
                        None if held(tags.param_open, rest) => return,
                        None => InvokeStage::Stopped,
                    }
                }
                InvokeStage::Key => {
                    let Some(end) = find_from(s, &mut self.searched, "\"") else {
                        return;
                    };
                    let attr = &s[end..];
                    match string_attr(attr) {
                        Some((text, v)) => {
                            let key = s[self.at..end].to_owned();
                            if text {
                                Args::key(&mut self.call.args, &key);
                                self.call.args.push('"');
                            }
                            self.at = s.len() - v.len();
                            self.searched = self.at;
                            InvokeStage::Value { key, text }
                        }
                        None if held(STRING_TRUE, attr) || held(STRING_FALSE, attr) => return,
                        None => InvokeStage::Stopped,
                    }
                }
                InvokeStage::Value { key, text } => {
                    let args = text.then_some(&mut self.call.args);
                    let Some(end) = read_value(
                        s,
                        &mut self.searched,
                        tags.param_close,
                        tags.invoke_close,
                        args,
                    ) else {
                        return;
                    };
                    if *text {
                        self.call.args.push('"');
                    } else {
                        Args::key(&mut self.call.args, key);
                        self.call
                            .args
                            .push_str(&json_value(&s[self.at..end], false));
                    }
                    self.at = end + tags.param_close.len();
                    InvokeStage::Next
                }
                InvokeStage::Stopped => return,
            };
            self.stage = next;
        }
    }
}

/// A parameter value as JSON text: `string="true"` is the raw text, `string="false"`
/// is JSON (kept as a string when it does not parse).
fn json_value(raw: &str, is_string: bool) -> String {
    if !is_string && let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) {
        return v.to_string();
    }
    callstream::string(raw)
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

/// Whether a chat request's output is scanned for tool calls: non-empty
/// `tools` and a `tool_choice` other than `"none"` (the template still sees
/// the tools with `"none"`).
pub(crate) fn parses_tools(b: &Map<String, Value>) -> bool {
    b.get("tools")
        .and_then(Value::as_array)
        .is_some_and(|a| !a.is_empty())
        && b.get("tool_choice").and_then(Value::as_str) != Some("none")
}

/// The tool-call markup a chat template teaches its model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolFormat {
    /// DSML blocks (`<｜DSML｜tool_calls>`; V4.1's `<｜DSML｜ calls>`), in the template's
    /// [`Spelling`].
    Dsml(Spelling),
    /// Key/value calls: GLM's `<tool_call>NAME<arg_key>…</arg_key><arg_value>…</arg_value>
    /// </tool_call>`, Solar-Open's `<|tool_call:start|>NAME` and
    /// `<|tool_arg:start|>…<|tool_arg:value|>…<|tool_arg:end|>` lines, in the template's
    /// [`KvSpelling`].
    KvMarkup(KvSpelling),
    /// Hermes' `<tool_call>` around one JSON object naming `name` and
    /// `arguments` (Qwen3's markup).
    Hermes,
    /// Qwen3-Coder's `<tool_call>` around `<function=NAME>` and
    /// `<parameter=KEY>` blocks (Qwen3.6's and Qwen3.8's markup), its values
    /// framed as the template renders a call ([`Framing`]).
    QwenXml(Framing),
    /// Muse Spark's recipient protocol: `to=<recipient><|message|>` messages
    /// ended by `<|eom|>`/`<|eot|>`, a `to=<tool>` message's body one ATEM
    /// calls block ([`crate::museatem`]); the `to=self` span is the
    /// reasoning.
    MuseSpark,
    /// None the server parses: a chat request that asks for tool calls is
    /// refused.
    Unparsed,
}

impl ToolFormat {
    /// The markup the template `source` spells for a call: the key/value
    /// markup's when it spells a [`KvSpelling`]'s call, name and value
    /// openers (GLM's `<tool_call>`, `<arg_key>` and `<arg_value>`), Qwen3-Coder's
    /// when it spells `<tool_call>`, `<function=` and `<parameter=` (llama.cpp
    /// reads the three spellings), DSML's when it spells the DSML token (in the
    /// spelling [`Spelling::of_template`] reads), else
    /// nothing the source alone can name — a `<tool_call>` template may carry
    /// Hermes' JSON body, which only a rendered call tells apart
    /// ([`ToolFormat::of_chat_template`]). Qwen3-Coder's framing is read from
    /// that rendered call too: the source is parsed and probed here, and a
    /// source that does not render one in either framing is
    /// [`ToolFormat::Unparsed`] (a guessed framing would change a value's
    /// bytes).
    #[must_use]
    pub fn of_template(source: &str) -> ToolFormat {
        ToolFormat::spelled(source, || {
            ChatTemplate::parse(source)
                .ok()
                .and_then(|t| probe_framing(&t))
        })
    }

    /// The markup a chat template teaches, read the way llama.cpp's
    /// autoparser reads it: the source's own spellings first (the key/value tags,
    /// Qwen3-Coder's with the framing its rendered call shows, the DSML token),
    /// then a rendered assistant call for a
    /// `<tool_call>` template the source cannot type by spelling — Hermes'
    /// when some rendered `<tool_call>`…`</tool_call>` holds a JSON body
    /// naming `name` and `arguments`. A template whose probe render fails
    /// teaches nothing this server parses.
    #[must_use]
    pub fn of_chat_template(t: &ChatTemplate) -> ToolFormat {
        let spelled = ToolFormat::spelled(t.source(), || probe_framing(t));
        if spelled != ToolFormat::Unparsed {
            return spelled;
        }
        if probe_render(t).is_some_and(|p| hermes_markup(&p)) {
            ToolFormat::Hermes
        } else {
            spelled
        }
    }

    /// The source's own spelling, `framing` read (only when the source
    /// spells Qwen3-Coder's markup) from a rendered call.
    fn spelled(source: &str, framing: impl FnOnce() -> Option<Framing>) -> ToolFormat {
        // The recipient protocol's two markers, as llama.cpp's template
        // detection reads them (`to=commentary` and `<|eom|>`): Muse
        // Glimmer's template carries the end token but no commentary
        // recipient.
        if source.contains("to=commentary") && source.contains("<|eom|>") {
            ToolFormat::MuseSpark
        } else if let Some(spelling) = KvSpelling::of_template(source) {
            ToolFormat::KvMarkup(spelling)
        } else if ::models::ToolFormat::of_template(source) == Some(::models::ToolFormat::QwenXml) {
            framing().map_or(ToolFormat::Unparsed, ToolFormat::QwenXml)
        } else if source.contains(DSML) {
            ToolFormat::Dsml(Spelling::of_template(source))
        } else {
            ToolFormat::Unparsed
        }
    }
}

/// The parameter and string value of the assistant call the probe renders.
const PROBE_KEY: &str = "location";
const PROBE_VALUE: &str = "Seoul";

/// What `t` renders for a conversation whose last turn is an assistant call
/// of `get_weather` with one string argument; `None` when it does not render.
fn probe_render(t: &ChatTemplate) -> Option<String> {
    let mut vars = Map::new();
    vars.insert(
        "messages".into(),
        json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {
                    "name": "get_weather", "arguments": {PROBE_KEY: PROBE_VALUE},
                }},
            ]},
        ]),
    );
    vars.insert("add_generation_prompt".into(), json!(false));
    vars.insert(
        "tools".into(),
        json!([{"type": "function", "function": {
            "name": "get_weather", "parameters": {
                "type": "object",
                "properties": {PROBE_KEY: {"type": "string"}},
            },
        }}]),
    );
    vars.insert("enable_thinking".into(), json!(true));
    t.render(&vars).ok()
}

/// How a template frames a parameter value, read from the call it renders
/// ([`probe_render`]); `None` when it renders no call, or one in neither
/// framing. A lines framing whose template opens no think span (the span is
/// read from the source, as llama.cpp reads it: the probe renders no
/// generation prompt, where a template may open one) is
/// [`Framing::Coder`].
fn probe_framing(t: &ChatTemplate) -> Option<Framing> {
    Some(
        match Framing::of_rendered(&probe_render(t)?, PROBE_KEY, PROBE_VALUE)? {
            Framing::Lines if ::models::ReasoningFormat::of_template(t.source()).is_none() => {
                Framing::Coder
            }
            framing => framing,
        },
    )
}

/// Whether a rendered template spells the Hermes call markup: some
/// `<tool_call>`…`</tool_call>` whose body is a JSON object naming `name` and
/// `arguments` (Qwen3's header spells it with placeholders, its assistant
/// branch with the call itself; Qwen3.8's `<function>` body does not match).
fn hermes_markup(rendered: &str) -> bool {
    let mut at = 0;
    while let Some(open) = rendered[at..].find(hermes::CALL_OPEN) {
        let body = at + open + hermes::CALL_OPEN.len();
        match rendered[body..].find(hermes::CALL_CLOSE) {
            Some(close) => {
                let inner = rendered[body..body + close].trim();
                if inner.starts_with('{')
                    && inner.ends_with('}')
                    && inner.contains("\"name\"")
                    && inner.contains("\"arguments\"")
                {
                    return true;
                }
                at = body + close;
            }
            None => return false,
        }
    }
    false
}

/// The tool-call scan a parser runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tools {
    /// DSML's markup, in the given spelling.
    Dsml(Spelling),
    /// The key/value markup in the given spelling, typing each argument by the
    /// request's tool schemas.
    KvMarkup(KvSpelling, ArgTypes),
    /// Hermes' markup.
    Hermes,
    /// Qwen3-Coder's markup, its values framed as the template renders them,
    /// typing each parameter by the request's tool schemas.
    QwenXml(Framing, ParamKinds),
    /// Muse Spark's recipient protocol ([`crate::museatem`]): its values
    /// carry no schema (JSON when a value parses, else the raw text).
    MuseSpark,
}

#[derive(Clone, Debug)]
enum Scan {
    Dsml(DsmlScan),
    Kv(KvScan),
    Hermes(HermesScan),
    Qwen(QwenScan),
    Recipient(crate::museatem::MuseScan),
}

/// A generation whose tool-call markup does not parse: which scan refused,
/// and its refusal.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MarkupError {
    /// The key/value scan's refusal.
    #[error("{0}")]
    Kv(#[from] KvError),
    /// The Hermes scan's refusal.
    #[error("{0}")]
    Hermes(#[from] HermesError),
    /// The Qwen3-Coder scan's refusal.
    #[error("{0}")]
    Qwen(#[from] QwenXmlError),
    /// A call's arguments were streamed as they were read and the markup does
    /// not end them that way ([`crate::callstream`]).
    #[error(
        "tool call {index} ({name}): its arguments were streamed as read, and the markup does \
         not end them that way (a parameter given twice, or text that stops being the call)"
    )]
    Diverged { index: usize, name: String },
}

impl PartialEq<KvError> for MarkupError {
    /// A key/value refusal equals its own form, so a caller holding the scan's
    /// error and one key/value refusal compares them directly.
    fn eq(&self, other: &KvError) -> bool {
        matches!(self, MarkupError::Kv(e) if e == other)
    }
}

/// The streaming parser for one chat generation.
///
/// With [`ReasoningFormat::None`] and no tools the text passes through unchanged,
/// piece for piece. With tools, the think span is tracked in either format so
/// tool-call markup inside it is never parsed; with `None` the span's text, its
/// `</think>` included, stays in `content`.
#[derive(Clone, Debug)]
pub struct ChatParser {
    format: ReasoningFormat,
    /// The tags the span is split on, and that [`ReasoningFormat::None`]
    /// writes back around the reasoning.
    tags: ThinkTags,
    /// `None` when neither a split nor a scan is asked for.
    pub(crate) think: Option<ThinkSplit>,
    scan: Option<Scan>,
    /// `Some` for a stream that sends a call as it is read
    /// ([`ChatParser::streaming_calls`]).
    streamer: Option<Streamer>,
    /// The call the last push left open once its name was read.
    open: Option<Progress>,
    /// Whether a call the token limit cuts after its name stays a call
    /// ([`ChatParser::keeping_cut_call`]).
    keep_cut: bool,
    total: Message,
}

impl ChatParser {
    /// A parser for a generation that follows `prompt` (the rendered template).
    /// `tools` turns the DSML scan on, in its spelling.
    #[must_use]
    pub fn new(prompt: &str, format: ReasoningFormat, tools: Option<Spelling>) -> Self {
        ChatParser::with_tools(prompt, format, tools.map(Tools::Dsml))
    }

    /// A parser for a generation that follows `prompt`, scanning for `tools`'
    /// markup when it is `Some`, on `<think>` tags.
    #[must_use]
    pub fn with_tools(prompt: &str, format: ReasoningFormat, tools: Option<Tools>) -> Self {
        ChatParser::with_tags(prompt, format, tools, ThinkTags::THINK)
    }

    /// [`ChatParser::with_tools`] on the span `tags` name.
    #[must_use]
    pub fn with_tags(
        prompt: &str,
        format: ReasoningFormat,
        tools: Option<Tools>,
        tags: ThinkTags,
    ) -> Self {
        // Muse Spark's scan splits its own reasoning (the `to=self` span),
        // so it wants no think split; the others track the span so markup
        // inside it is never parsed.
        let split = (format == ReasoningFormat::Deepseek || tools.is_some())
            && !matches!(tools, Some(Tools::MuseSpark));
        // Qwen3-Coder's call ends an open span without `</think>`, as
        // llama.cpp's parser for the markup reads it.
        let call_ends_span = matches!(tools, Some(Tools::QwenXml(..)));
        ChatParser {
            format,
            tags,
            think: split.then(|| {
                let mut t = ThinkSplit::for_prompt_with(prompt, tags);
                if format == ReasoningFormat::Deepseek {
                    t = t.trimming_after_opened();
                }
                if call_ends_span {
                    t.ending_at(qwenxml::CALL_OPEN)
                } else {
                    t
                }
            }),
            scan: tools.map(|t| match t {
                Tools::Dsml(spelling) => Scan::Dsml(DsmlScan::new(spelling)),
                Tools::KvMarkup(spelling, types) => Scan::Kv(KvScan::new(spelling, types)),
                Tools::Hermes => Scan::Hermes(HermesScan::default()),
                Tools::QwenXml(framing, kinds) => Scan::Qwen(QwenScan::new(kinds, framing)),
                Tools::MuseSpark => Scan::Recipient(crate::museatem::MuseScan::default()),
            }),
            streamer: None,
            open: None,
            keep_cut: false,
            total: Message::default(),
        }
    }

    /// The same parser, reporting each call as it is read: a push also yields
    /// the [`CallPiece`]s of the calls it opened and extended
    /// ([`ChatParser::take_pieces`]), which concatenate, per call, to the
    /// `arguments` it closes with. A call is announced once its name is read, so
    /// markup that then stops being that call is a [`MarkupError::Diverged`],
    /// where a parser that does not stream gives the text back as content
    /// (DSML) — the fragments already sent cannot be taken back.
    #[must_use]
    pub fn streaming_calls(mut self) -> Self {
        self.streamer = Some(Streamer::default());
        self.keeping_cut_call()
    }

    /// The same parser, answering a generation the token limit cuts inside a call after its name
    /// is read ([`ChatParser::try_finish_at`]) with that call and the arguments read so far,
    /// as llama-server's chat route does. The default gives the call's text back as content,
    /// for an answer whose call shape holds only closed calls.
    #[must_use]
    pub fn keeping_cut_call(mut self) -> Self {
        self.keep_cut = true;
        self
    }

    /// The pieces the pushes since the last call reported; empty for a parser
    /// that does not stream its calls.
    pub fn take_pieces(&mut self) -> Vec<CallPiece> {
        self.streamer
            .as_mut()
            .map(Streamer::take)
            .unwrap_or_default()
    }

    /// The same parser for a template whose markup is `format`: Qwen3-Coder's
    /// drops the whitespace that opens the content when reasoning is split
    /// out, with or without the request's tools, as llama.cpp's parser for
    /// the markup reads it ([`ThinkSplit::trimming_content`]).
    #[must_use]
    pub fn for_template(mut self, format: ToolFormat) -> Self {
        if matches!(format, ToolFormat::QwenXml(_)) && self.format == ReasoningFormat::Deepseek {
            self.think = self.think.take().map(ThinkSplit::trimming_content);
        }
        self
    }

    /// The same parser for a forced tool call that follows the model's own
    /// reasoning ([`crate::constrain::Constraint::follows_reasoning`]): only
    /// the close tag ends the span, so a call opener inside it is reasoning,
    /// as the generation that holds the call until the close reads it.
    #[must_use]
    pub(crate) fn following_reasoning(mut self) -> Self {
        self.think = self.think.take().map(ThinkSplit::ending_only_at_close);
        self
    }

    /// The same parser, its think split and call scan reading their tags
    /// only where their special tokens wrote them, for those `tok` holds as
    /// one special token (`SpecialTags`). The text it is fed then carries
    /// where its special tokens start ([`ChatParser::try_push`]).
    #[must_use]
    pub(crate) fn reading_specials(mut self, tok: &dyn Tokenizer) -> Self {
        self.think = self.think.take().map(|t| t.reading_specials(tok));
        match &mut self.scan {
            Some(Scan::Qwen(q)) => q.read_specials(tok),
            Some(Scan::Hermes(h)) => h.read_specials(tok),
            Some(Scan::Kv(k)) => k.read_specials(tok),
            Some(Scan::Dsml(_) | Scan::Recipient(_)) | None => {}
        }
        self
    }

    /// Parses a whole generation.
    #[must_use]
    pub fn parse(
        prompt: &str,
        format: ReasoningFormat,
        tools: Option<Spelling>,
        text: &str,
    ) -> Message {
        let mut p = ChatParser::new(prompt, format, tools);
        let _ = p.push(text);
        let _ = p.finish();
        p.total
    }

    /// Feeds generated text; returns what may be sent now. For a parser with
    /// no scan or the DSML scan, which cannot fail unless it streams its calls
    /// ([`ChatParser::streaming_calls`]); a key/value, Hermes or Qwen parser's caller
    /// uses [`ChatParser::try_push`].
    ///
    /// # Panics
    ///
    /// On markup a key/value, Hermes or Qwen scan refuses, or a streamed call the
    /// markup ends otherwise, naming it.
    pub fn push(&mut self, text: &str) -> Message {
        self.try_push(text, &[])
            .unwrap_or_else(|e| panic!("ChatParser::push on a scan that failed: {e}"))
    }

    /// Releases everything held at the end of the generation; see
    /// [`ChatParser::push`] for which parsers may call it.
    ///
    /// # Panics
    ///
    /// On markup a key/value, Hermes or Qwen scan refuses, or a streamed call the
    /// markup ends otherwise, naming it.
    pub fn finish(&mut self) -> Message {
        self.try_finish()
            .unwrap_or_else(|e| panic!("ChatParser::finish on a scan that failed: {e}"))
    }

    /// Feeds generated text, its special tokens' texts at `specials` (byte
    /// offset and length, ascending); returns what may be sent now, or
    /// the tool-call markup that does not parse.
    pub fn try_push(
        &mut self,
        text: &str,
        specials: &[(usize, usize)],
    ) -> Result<Message, MarkupError> {
        let split = match &mut self.think {
            Some(t) => t.push(text, specials),
            None => Split {
                content: text.to_owned(),
                specials: specials.to_vec(),
                ..Split::default()
            },
        };
        let scanned = match &mut self.scan {
            None => None,
            Some(Scan::Dsml(d)) => Some(d.push_stream(&split.content)),
            Some(Scan::Kv(k)) => Some(k.push_stream(&split.content, &split.specials)?),
            Some(Scan::Hermes(h)) => Some(h.push(&split.content, &split.specials)?),
            Some(Scan::Qwen(q)) => Some(q.push_stream(&split.content, &split.specials)?),
            Some(Scan::Recipient(m)) => Some(m.push(&split.content)),
        };
        self.stream(scanned.as_ref())?;
        Ok(self.emit(split, scanned))
    }

    /// Releases everything held at the end of the generation, or names the
    /// tool call still open.
    pub fn try_finish(&mut self) -> Result<Message, MarkupError> {
        self.try_finish_at(false)
    }

    /// [`ChatParser::try_finish`] for a generation that the token limit cut (`cut`) or that
    /// ended on its own. llama-server parses the end of the text as the end of the input, so a
    /// cut inside a call is no fault there: its answer holds what was read, and its
    /// `finish_reason` `length` says the generation was cut. The same here, for every scan that
    /// refuses an open call: a call the markup has begun well and the limit cut is content (the
    /// text it consumed, as the DSML scan gives an open block back), or, once its name was read
    /// and the parser keeps it ([`ChatParser::keeping_cut_call`]), a call with the arguments read.
    /// A call that ended without the limit, and markup that already read wrong, stay errors.
    pub fn try_finish_at(&mut self, cut: bool) -> Result<Message, MarkupError> {
        // An open span's tail is reasoning; the scanner has seen none of it.
        let split = self
            .think
            .as_mut()
            .map(ThinkSplit::finish)
            .unwrap_or_default();
        let kept = self.open.as_ref().filter(|_| self.keep_cut && cut);
        // The call the limit cut, for a whole-state scan and for one that reads
        // by its own cursor.
        let (whole, sent) = (kept.map(cut_call), kept.map(cut_close));
        let scanned = match &mut self.scan {
            None => None,
            Some(Scan::Dsml(d)) => {
                let released = d.finish();
                Some(sent.unwrap_or(released))
            }
            Some(Scan::Kv(k)) => Some(ended(k.finish(), KvError::unclosed, cut, sent)?),
            Some(Scan::Hermes(h)) => Some(ended(h.finish(), HermesError::unclosed, cut, whole)?),
            Some(Scan::Qwen(q)) => Some(ended(q.finish(), QwenXmlError::unclosed, cut, sent)?),
            // The scan closes a message the stream ends inside as its call on its own.
            Some(Scan::Recipient(m)) => Some(m.finish()),
        };
        self.stream(scanned.as_ref())?;
        Ok(self.emit(split, scanned))
    }

    /// Hands a scan step to the open call and the stream's pieces.
    fn stream(&mut self, scanned: Option<&Scanned>) -> Result<(), MarkupError> {
        let Some(s) = scanned else {
            self.open = None;
            return Ok(());
        };
        let Some(advances) = &s.advances else {
            self.open.clone_from(&s.open);
            return match &mut self.streamer {
                Some(st) => st.step(&s.calls, s.open.as_ref()),
                None => Ok(()),
            };
        };
        for a in advances {
            if let Some(st) = &mut self.streamer {
                st.advance(a)?;
            }
            self.open = match (a, self.open.take()) {
                (Advance::Args(_, true) | Advance::Diverged, _) => None,
                (Advance::Args(p, false), Some(mut open)) if open.index == p.index => {
                    open.args.push_str(&p.args);
                    Some(open)
                }
                (Advance::Args(p, false), _) => Some(p.clone()),
            };
        }
        Ok(())
    }

    /// The message so far.
    #[must_use]
    pub fn message(&self) -> &Message {
        &self.total
    }

    fn emit(&mut self, split: Split, scanned: Option<Scanned>) -> Message {
        let mut d = Message::default();
        // The scan's own reasoning split (Muse Spark's) follows the same
        // route as the think span's, after it.
        let own = scanned.as_ref().map_or("", |s| s.reasoning.as_str());
        match self.format {
            ReasoningFormat::Deepseek => {
                d.reasoning = split.reasoning;
                d.reasoning.push_str(own);
            }
            ReasoningFormat::None => {
                d.content = split.reasoning;
                if split.closed {
                    d.content.push_str(self.tags.close);
                }
                d.content.push_str(own);
            }
        }
        match scanned {
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

/// A scan's end of text. `released` is its release; when the limit cut (`cut`) the generation
/// inside a call it reads as the template's so far (`unclosed`: the text of that call, from the
/// scan's own error), the answer holds what was read of that call instead of its error: the `kept`
/// call (its name read, as [`cut_call`] or [`cut_close`] gives it), else the text it consumed as
/// content.
fn ended<E: Into<MarkupError>>(
    released: Result<Scanned, E>,
    unclosed: fn(&E) -> Option<String>,
    cut: bool,
    kept: Option<Scanned>,
) -> Result<Scanned, MarkupError> {
    let e = match released {
        Err(e) if cut => e,
        released => return released.map_err(Into::into),
    };
    let Some(text) = unclosed(&e) else {
        return Err(e.into());
    };
    Ok(kept.unwrap_or(Scanned {
        content: text,
        ..Scanned::default()
    }))
}

/// The call a cut left open, as the arguments read so far spell it.
fn cut_call(p: &Progress) -> Scanned {
    Scanned {
        calls: vec![ToolCall {
            index: p.index,
            name: p.name.clone(),
            arguments: p.args.clone(),
        }],
        ..Scanned::default()
    }
}

/// [`cut_call`] for a scan that reads by its own cursor: it sent the arguments as it read them,
/// so the stream only closes the call.
fn cut_close(p: &Progress) -> Scanned {
    let close = Progress {
        args: String::new(),
        ..p.clone()
    };
    Scanned {
        advances: Some(vec![Advance::Args(close, true)]),
        ..cut_call(p)
    }
}

impl ToolFormat {
    /// The forced call in this format's markup: the inverse of the format's
    /// scan, or a 501 refusal for markup no scan here reads.
    pub(crate) fn call_parts(self, funs: &[Fun<'_>]) -> Result<Vec<Part>, Refusal> {
        match self {
            ToolFormat::Hermes => hermes::call_parts(funs),
            ToolFormat::QwenXml(framing) => qwenxml::call_parts(framing, funs),
            ToolFormat::KvMarkup(spelling) => glmxml::call_parts(spelling, funs),
            ToolFormat::Dsml(spelling) => call_parts(spelling, funs),
            ToolFormat::MuseSpark => crate::museatem::call_parts(funs),
            ToolFormat::Unparsed => Err(Refusal {
                message: "the chat template's tool-call markup has no inverse in this server: \
                          send the request without tool_choice \"required\" or a named function"
                    .to_owned(),
                unsupported: true,
                field: None,
            }),
        }
    }
}

/// The call this markup spells, as the parts structured output forces: the
/// inverse of [`DsmlScan`]. One `invoke` block with a `parameter` block per
/// parameter in the schema's order, `string="true"` around a text value (raw)
/// and `string="false"` around any other (JSON), in the template's spelling.
pub(crate) fn call_parts(spelling: Spelling, funs: &[Fun<'_>]) -> Result<Vec<Part>, Refusal> {
    let t = spelling.tags();
    call(
        funs,
        &format!("{}\n{}", t.calls_open, t.invoke_open),
        str::to_owned,
        |f| {
            let mut parts = vec![Part::Text("\">\n".to_owned())];
            for a in &f.args {
                parts.push(Part::Text(format!(
                    "{}{}\" string=\"{}\">",
                    t.param_open,
                    a.key,
                    is_text(&a.schema)
                )));
                parts.extend(value(&a.schema, Style::Raw, &a.path)?);
                parts.push(Part::Text(format!("{}\n", t.param_close)));
            }
            parts.push(Part::Text(format!("{}\n{}", t.invoke_close, t.calls_close)));
            Ok(parts)
        },
    )
}
