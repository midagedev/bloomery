//! Muse Spark's recipient protocol: the parser that turns one generation
//! into `reasoning_content`, `content` and `tool_calls`
//! (`common/parsers/muse-spark.cpp` in the reference).
//!
//! A turn is one or more messages, each
//! `to=<recipient><|message|>{content}` ended by `<|eom|>` (more messages
//! follow) or `<|eot|>` (the turn's end); the generation prompt ends at
//! `<|start|>assistant`, so the model emits its own ` to=…<|message|>` and
//! every later message carries the `<|start|>assistant` opener again:
//!
//! ```text
//!  to=self<|message|>chain of thought<|eom|>
//! <|start|>assistant to=commentary<|message|>user-facing update<|eom|>
//! <|start|>assistant to=user<|message|>the answer<|eot|>
//! ```
//!
//! The recipient names the channel: `self` is the reasoning span, `commentary`
//! a user-facing update that does not end the turn, `user` the answer that
//! does, and any other name a tool call — the message's body is ATEM markup:
//!
//! ```text
//! <atem:function_calls>
//! <atem:invoke name="NAME">
//! <atem:parameter name="KEY">VALUE</atem:parameter>
//! </atem:invoke>
//! </atem:function_calls>
//! ```
//!
//! A value is JSON when it parses (the reference reads a schema-typed value
//! as JSON and a string-typed one as text; the scan has no schema here, so
//! it takes the JSON reading first, like the DSML parameter's
//! `string="false"`), else the raw text as a JSON string.
//!
//! Every decision depends on the text only, never on how it was cut into
//! pieces: an open `self` or `commentary` message releases its text as it
//! arrives (holding back the tail that could still become an end token), a
//! tool message is held whole — its markup is parsed only when the message
//! closes — and a message the stream ends inside releases what it has to its
//! own channel. Markup that does not parse, and text that is no message
//! open, go to `content` as the raw text. The call a scan has opened and not
//! closed is reported as it is read ([`Scanned::open`]): its name is the
//! recipient (the template writes the function's name there), its arguments
//! the complete parameters read so far.

use crate::callstream::Args;
use crate::constrain::{
    Arg, Call, CallMode, Fun, Part, Refusal, Style, call, is_text, object_parameters, value,
};
use crate::dsml::{Scanned, ToolCall};
use crate::reasoning::partial_suffix;

/// The turn's opener, as the generation prompt ends it.
const START: &str = "<|start|>assistant";
/// `to=`, the recipient's marker.
const TO: &str = " to=";
/// The header's end.
const MESSAGE: &str = "<|message|>";
/// More messages follow.
const EOM: &str = "<|eom|>";
/// The turn's end.
const EOT: &str = "<|eot|>";
/// The calls block's opener.
const CALLS_OPEN: &str = "<atem:function_calls>";
const CALLS_CLOSE: &str = "</atem:function_calls>";
const INVOKE_OPEN: &str = "<atem:invoke name=\"";
const INVOKE_CLOSE: &str = "</atem:invoke>";
const PARAM_OPEN: &str = "<atem:parameter name=\"";
const PARAM_CLOSE: &str = "</atem:parameter>";

/// The streaming scan over one generation. The reasoning split is this
/// scan's own: the `to=self` span, no think tag.
#[derive(Clone, Debug, Default)]
pub struct MuseScan {
    /// The message the scan is inside, if any: its recipient and the body
    /// text held back (a tool message's whole body; nothing is held for the
    /// channel messages, whose text is released as it arrives).
    open: Option<(String, String)>,
    /// Text not yet decided.
    buf: String,
    calls: usize,
}

impl MuseScan {
    /// Feeds text.
    pub fn push(&mut self, piece: &str) -> Scanned {
        self.buf.push_str(piece);
        let mut out = Scanned::default();
        loop {
            let progressed = match self.open.take() {
                None => self.idle_step(&mut out),
                Some(open) => self.message_step(open, &mut out),
            };
            if !progressed {
                break;
            }
        }
        out.open = self.open_call();
        out
    }

    /// Releases everything held at the end of the stream: a message the
    /// stream ends inside goes to its own channel, and a tool message's
    /// whole calls block is its call — markup that never became one is
    /// content.
    pub fn finish(&mut self) -> Scanned {
        let mut out = Scanned::default();
        let mut text = std::mem::take(&mut self.buf);
        if let Some((recipient, held)) = self.open.take() {
            text = format!("{held}{text}");
            self.close(&recipient, &text, &mut out);
        } else {
            out.content = text;
        }
        out
    }

    /// Outside a message: looks for the next `to=…<|message|>` open (after a
    /// `<|start|>assistant`, or at the stream's start, where the generation
    /// prompt already wrote the opener). Returns whether anything was
    /// consumed.
    fn idle_step(&mut self, out: &mut Scanned) -> bool {
        let Some(at) = self.buf.find(MESSAGE) else {
            // Hold what could still become an open: the partial tail of any
            // opener token, everything from the last ` to=` or
            // `<|start|>assistant` (whose recipient may still be coming), or
            // both when one sits inside the other.
            let release = [
                self.buf.rfind(TO),
                self.buf.rfind(START),
                Some(
                    self.buf.len()
                        - partial_suffix(&self.buf, MESSAGE)
                            .max(partial_suffix(&self.buf, TO))
                            .max(partial_suffix(&self.buf, START)),
                ),
            ]
            .into_iter()
            .flatten()
            .min()
            .unwrap_or(self.buf.len());
            out.content.push_str(&self.buf[..release]);
            self.buf.drain(..release);
            return false;
        };
        let (open_start, name) = match self.buf[..at].rfind(START) {
            Some(p) => (
                p,
                self.buf[p + START.len()..at]
                    .strip_prefix(TO)
                    .map(str::to_owned),
            ),
            None => (0, self.buf[..at].strip_prefix(TO).map(str::to_owned)),
        };
        match name.as_deref().filter(|n| recipient_is(n)) {
            Some(name) => {
                // Text before the opener is content, as it is in any other
                // position.
                out.content.push_str(&self.buf[..open_start]);
                self.buf.drain(..at + MESSAGE.len());
                self.open = Some((name.to_owned(), String::new()));
            }
            None => {
                // No message opens here: the text is content, and the scan
                // takes the next message's opener as it comes.
                out.content.push_str(&self.buf[..at + MESSAGE.len()]);
                self.buf.drain(..at + MESSAGE.len());
            }
        }
        true
    }

    /// Inside a message: takes its body to the first end token, releasing a
    /// channel message's text as it arrives and holding a tool message's
    /// whole. Returns whether anything was consumed.
    fn message_step(&mut self, open: (String, String), out: &mut Scanned) -> bool {
        let (recipient, mut held) = open;
        let end = [EOM, EOT]
            .into_iter()
            .filter_map(|t| self.buf.find(t).map(|at| (at, t)))
            .min_by_key(|(at, _)| *at);
        let Some((at, token)) = end else {
            let keep = partial_suffix(&self.buf, EOM).max(partial_suffix(&self.buf, EOT));
            let upto = self.buf.len() - keep;
            let piece = self.buf[..upto].to_owned();
            self.buf.drain(..upto);
            if recipient_is_tool(&recipient) {
                held.push_str(&piece);
                self.open = Some((recipient, held));
            } else {
                if recipient == "self" {
                    out.reasoning.push_str(&piece);
                } else {
                    out.content.push_str(&piece);
                }
                self.open = Some((recipient, String::new()));
            }
            return upto > 0;
        };
        held.push_str(&self.buf[..at]);
        self.buf.drain(..at + token.len());
        self.close(&recipient, &held, out);
        true
    }

    /// A closed message's body, sent to its recipient's channel. A tool
    /// message's body is parsed as one ATEM block; markup that does not
    /// parse is the raw text.
    fn close(&mut self, recipient: &str, body: &str, out: &mut Scanned) {
        if recipient_is_tool(recipient) {
            match parse_atem(body) {
                Ok(calls) => {
                    out.calls.extend(calls.map(|(name, arguments)| {
                        let call = ToolCall {
                            index: self.calls,
                            name,
                            arguments,
                        };
                        self.calls += 1;
                        call
                    }));
                }
                Err(()) => out.content.push_str(body),
            }
        } else if recipient == "self" {
            out.reasoning.push_str(body);
        } else {
            out.content.push_str(body);
        }
    }

    /// The call a tool message has opened, as far as its body reads it.
    fn open_call(&self) -> Option<crate::callstream::Progress> {
        let (recipient, held) = self.open.as_ref()?;
        recipient_is_tool(recipient).then(|| crate::callstream::Progress {
            index: self.calls,
            name: recipient.clone(),
            args: partial_args(held),
        })
    }
}

/// Whether `name` can name a recipient: non-empty, and free of the bytes a
/// real recipient escapes away (the reference's tool names match `[^<]*`,
/// and a header recipient is one line).
fn recipient_is(name: &str) -> bool {
    !name.is_empty() && !name.contains(['<', '\n'])
}

/// Whether `recipient` names a tool rather than a channel.
fn recipient_is_tool(recipient: &str) -> bool {
    !matches!(recipient, "self" | "commentary" | "user")
}

/// The complete `<atem:parameter name="KEY">VALUE</atem:parameter>` blocks
/// of a tool message's body so far, as the arguments JSON the message closes
/// with — a prefix of it.
fn partial_args(body: &str) -> String {
    let mut args = Args::new();
    let Some(at) = body.find(INVOKE_OPEN) else {
        return args.so_far();
    };
    if let Ok((_, tail)) = plain_name(&body[at + INVOKE_OPEN.len()..])
        && let Some(after) = tail.strip_prefix("\">")
    {
        let (parameters, _) = parameters(after);
        for (key, value) in parameters {
            args.field(&key, &json_value(&value));
        }
    }
    args.so_far()
}

/// One ATEM calls block: every invoke in it, each its name and its arguments
/// as JSON text in parameter order. An error when the body is not the block
/// — an invoke whose parameters leave text behind is not the template's
/// markup.
fn parse_atem(body: &str) -> Result<impl Iterator<Item = (String, String)>, ()> {
    let rest = body.trim().strip_prefix(CALLS_OPEN).ok_or(())?;
    let rest = rest.strip_suffix(CALLS_CLOSE).ok_or(())?;
    let mut calls = Vec::new();
    let mut rest = rest.trim_start();
    while let Some(after) = rest.strip_prefix(INVOKE_OPEN) {
        let (name, end) = plain_name(after)?;
        let after = end.strip_prefix("\">").ok_or(())?;
        let end = after.find(INVOKE_CLOSE).ok_or(())?;
        let (parameters, used) = parameters(&after[..end]);
        if !after[..end][used..].trim().is_empty() {
            return Err(());
        }
        let mut fields = Vec::new();
        for (key, value) in parameters {
            fields.push(format!(
                "{}:{}",
                crate::callstream::string(&key),
                json_value(&value)
            ));
        }
        calls.push((name, format!("{{{}}}", fields.join(","))));
        rest = after[end + INVOKE_CLOSE.len()..].trim_start();
    }
    if rest.trim().is_empty() && !calls.is_empty() {
        Ok(calls.into_iter())
    } else {
        Err(())
    }
}

/// `NAME">` at the start of `s`: the name and the rest. An error when the
/// name is empty or carries a byte the markup cannot spell.
fn plain_name(s: &str) -> Result<(String, &str), ()> {
    let end = s.find("\">").ok_or(())?;
    let name = &s[..end];
    if name.is_empty() || name.contains(['"', '<', '\n']) {
        return Err(());
    }
    Ok((name.to_owned(), &s[end..]))
}

/// The complete `<atem:parameter name="KEY">VALUE</atem:parameter>` blocks
/// at the start of `body`, in order, with the bytes they take. An incomplete
/// trailing parameter is dropped (it is a prefix of the next push's
/// reading).
fn parameters(body: &str) -> (Vec<(String, String)>, usize) {
    let mut out = Vec::new();
    let mut rest = body.trim_start();
    while let Some(after) = rest.strip_prefix(PARAM_OPEN) {
        let Ok((key, tail)) = plain_name(after) else {
            break;
        };
        let Some(after_key) = tail.strip_prefix("\">") else {
            break;
        };
        let Some(end) = after_key.find(PARAM_CLOSE) else {
            break;
        };
        out.push((key, after_key[..end].to_owned()));
        rest = after_key[end + PARAM_CLOSE.len()..].trim_start();
    }
    (out, body.len() - rest.len())
}

/// A parameter's value as JSON text: the JSON reading when it parses, else
/// the raw text as a JSON string.
fn json_value(raw: &str) -> String {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) {
        return v.to_string();
    }
    crate::callstream::string(raw)
}

/// The call this markup spells, as the parts structured output forces: the
/// inverse of [`MuseScan`] — ` to=NAME<|message|>`, then the ATEM block with
/// the template's own framing (a text value raw, any other as JSON), its
/// parameters as the object rule writes them: the required ones, in the
/// schema's order, then the optional ones, each of which may be left out.
/// The call has no opener to watch for (the recipient slot names the
/// answer's channels as well as a tool), so a watched one builds nothing.
pub(crate) fn call_parts(funs: &[Fun<'_>], mode: CallMode) -> Result<Call, Refusal> {
    if mode == CallMode::Watched {
        return Ok(Call::unopened(Vec::new()));
    }
    let block = |a: &Arg<'_>| -> Result<Vec<Part>, Refusal> {
        let mut parts = vec![Part::Text(format!("{PARAM_OPEN}{key}\">", key = a.key))];
        parts.extend(value(
            &a.schema,
            if is_text(&a.schema) {
                Style::Raw
            } else {
                Style::JsonInMarkup
            },
            &a.path,
        )?);
        parts.push(Part::Text(format!("{PARAM_CLOSE}\n")));
        Ok(parts)
    };
    let parts = call(funs, TO, str::to_owned, |f| {
        let mut parts = vec![Part::Text(format!(
            "{MESSAGE}{CALLS_OPEN}\n{INVOKE_OPEN}{name}\">\n",
            name = f.name
        ))];
        parts.extend(object_parameters(&f.args, &block)?);
        parts.push(Part::Text(format!("{INVOKE_CLOSE}\n{CALLS_CLOSE}")));
        Ok(parts)
    })?;
    Ok(Call::unopened(parts))
}

#[cfg(test)]
mod tests {
    use crate::dsml::Message;
    use crate::reasoning::ReasoningFormat;

    use super::MuseScan;

    /// The call markup the reference's own test renders
    /// (tests/test-chat.cpp, the Muse Spark cases).
    const CALL_MARKUP: &str = "<atem:function_calls>\n<atem:invoke name=\"special_function\">\
\n<atem:parameter name=\"arg1\">1</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";

    fn scan(text: &str) -> Message {
        let mut s = MuseScan::default();
        let out = s.push(text);
        let mut m = Message {
            reasoning: out.reasoning,
            content: out.content,
            calls: out.calls,
        };
        let end = s.finish();
        m.reasoning.push_str(&end.reasoning);
        m.content.push_str(&end.content);
        m.calls.extend(end.calls);
        m
    }

    /// A plain answer behaves as it does for Muse Glimmer: the final
    /// message's text is the content, its `<|eot|>` not included.
    #[test]
    fn a_plain_answer_is_the_final_messages_text() {
        let m = scan(" to=user<|message|>Hello, world!\nWhat's up?<|eot|>");
        assert_eq!(
            (m.reasoning.as_str(), m.content.as_str(), m.calls.len()),
            ("", "Hello, world!\nWhat's up?", 0)
        );
    }

    /// A commentary message is user-facing content, not reasoning: it joins
    /// the final answer's text with no separator between them.
    #[test]
    fn a_commentary_message_is_content() {
        let m = scan(
            " to=commentary<|message|>Hello, world!\nWhat's up?<|eom|>\
             <|start|>assistant to=user<|message|>Done!<|eot|>",
        );
        assert_eq!(
            (m.reasoning.as_str(), m.content.as_str()),
            ("", "Hello, world!\nWhat's up?Done!")
        );
    }

    /// Commentary announces the work, then the tool call follows in the same
    /// generation: the call is the message after it, one call in it, its
    /// number value read as JSON.
    #[test]
    fn a_tool_message_is_the_call_after_the_commentary() {
        let m = scan(&format!(
            " to=commentary<|message|>Let me look that up.<|eom|>\
             <|start|>assistant to=special_function<|message|>{CALL_MARKUP}"
        ));
        assert_eq!(m.reasoning, "");
        assert_eq!(m.content, "Let me look that up.");
        assert_eq!(m.calls.len(), 1);
        assert_eq!(
            (
                m.calls[0].index,
                m.calls[0].name.as_str(),
                m.calls[0].arguments.as_str()
            ),
            (0, "special_function", "{\"arg1\":1}")
        );
    }

    /// Reasoning still reaches the reasoning channel with commentary in the
    /// picture: the `to=self` span is the reasoning and nothing else is.
    #[test]
    fn the_self_span_is_the_reasoning() {
        let m = scan(
            " to=self<|message|>Thinking.<|eom|>\
             <|start|>assistant to=commentary<|message|>Working on it.<|eom|>\
             <|start|>assistant to=user<|message|>Hello!<|eot|>",
        );
        assert_eq!(
            (m.reasoning.as_str(), m.content.as_str(), m.calls.len()),
            ("Thinking.", "Working on it.Hello!", 0)
        );
    }

    /// A value that is not JSON is the raw text as a JSON string; one that
    /// is stays JSON; a repeated key keeps its last value in place.
    #[test]
    fn a_value_is_json_when_it_parses_else_the_raw_text() {
        let m = scan(
            " to=get_weather<|message|><atem:function_calls>\
             \n<atem:invoke name=\"get_weather\">\
             \n<atem:parameter name=\"city\">Seoul</atem:parameter>\
             \n<atem:parameter name=\"days\">3</atem:parameter>\
             \n<atem:parameter name=\"opts\">{\"a\":1}</atem:parameter>\
             \n</atem:invoke>\n</atem:function_calls>",
        );
        assert_eq!(
            m.calls[0].arguments,
            "{\"city\":\"Seoul\",\"days\":3,\"opts\":{\"a\":1}}"
        );
    }

    /// Markup that does not parse, a text that is no message open, and a
    /// tool message the stream ends inside go to `content` as the raw text.
    #[test]
    fn what_does_not_parse_is_content() {
        let m = scan(" to=tool<|message|><atem:function_calls> never closed");
        assert_eq!(m.content, "<atem:function_calls> never closed");
        let m = scan("no message here<|message|>or a bare header");
        assert_eq!(m.content, "no message here<|message|>or a bare header");
        let m = scan(
            " to=f<|message|><atem:function_calls>\\n<atem:invoke name=\"f\">\
             \\n<atem:parameter name=\"x\">1<atem:parameter>\\n</atem:invoke>\\n\
             </atem:function_calls>",
        );
        assert!(
            m.content
                .contains("<atem:parameter name=\"x\">1<atem:parameter>"),
            "{}",
            m.content
        );
        assert!(m.calls.is_empty());
    }

    /// Every decision depends on the text only: cut anywhere, a stream and a
    /// whole response parse to the same message. FAIL-first: with the end
    /// token's partial tail released, the cut at `<|eo` splits it wrongly.
    #[test]
    fn a_stream_and_a_whole_response_parse_the_same() {
        let text = format!(
            " to=self<|message|>Think<|eom|>\
             <|start|>assistant to=commentary<|message|>Working.<|eom|>\
             <|start|>assistant to=special_function<|message|>{CALL_MARKUP}"
        );
        let whole = scan(&text);
        for cut in [1..13, 40..41, 60..95, 96..140].into_iter().flatten() {
            let mut s = MuseScan::default();
            let mut m = Message::default();
            for piece in [
                text[..cut].to_owned(),
                text[cut..].to_owned(),
                String::new(),
            ] {
                let out = s.push(&piece);
                m.reasoning.push_str(&out.reasoning);
                m.content.push_str(&out.content);
                m.calls.extend(out.calls);
            }
            let end = s.finish();
            m.reasoning.push_str(&end.reasoning);
            m.content.push_str(&end.content);
            m.calls.extend(end.calls);
            assert_eq!(
                format!("{m:?}"),
                format!("{whole:?}"),
                "the stream cut at {cut} differs"
            );
        }
    }

    /// The call a tool message has opened is reported as it is read: its
    /// name is the recipient and its arguments the complete parameters so
    /// far, a prefix of the arguments the message closes with. The call
    /// itself is emitted when the message ends.
    #[test]
    fn an_open_tool_message_reports_its_progress() {
        let mut s = MuseScan::default();
        let out = s.push(" to=special_function<|message|><atem:function_calls>");
        assert_eq!(
            out.open
                .as_ref()
                .map(|p| (p.index, p.name.as_str(), p.args.as_str())),
            Some((0, "special_function", "{"))
        );
        let out = s.push("\n<atem:invoke name=\"special_function\">\n");
        assert_eq!(out.open.as_ref().map(|p| p.args.as_str()), Some("{"));
        let out = s.push("<atem:parameter name=\"arg1\">");
        assert_eq!(out.open.as_ref().map(|p| p.args.as_str()), Some("{"));
        let out = s.push("1</atem:parameter>\n</atem:invoke>\n</atem:function_calls>");
        // The markup is whole but the message is not ended: the call waits,
        // its arguments so far an open object (a prefix of what it closes
        // with).
        assert_eq!(
            out.open.as_ref().map(|p| p.args.as_str()),
            Some("{\"arg1\":1")
        );
        assert!(out.calls.is_empty());
        let out = s.push("<|eom|>");
        assert!(out.open.is_none());
        assert_eq!(
            out.calls
                .iter()
                .map(|c| (c.index, c.name.as_str(), c.arguments.as_str()))
                .collect::<Vec<_>>(),
            vec![(0, "special_function", "{\"arg1\":1}")]
        );
    }

    /// The parser the chat route runs: the recipient protocol's reasoning
    /// reaches `reasoning_content` under [`ReasoningFormat::Deepseek`] and
    /// stays in `content` under [`ReasoningFormat::None`].
    #[test]
    fn the_chat_parser_routes_the_recipient_channels() {
        use crate::dsml::{ChatParser, Tools};
        let text = " to=self<|message|>Thinking.<|eom|><|start|>assistant \
to=commentary<|message|>Working on it.<|eom|><|start|>assistant \
to=user<|message|>Hello!<|eot|>";
        for (format, want_reasoning, want_content) in [
            (
                ReasoningFormat::Deepseek,
                "Thinking.",
                "Working on it.Hello!",
            ),
            (ReasoningFormat::None, "", "Thinking.Working on it.Hello!"),
        ] {
            let mut p =
                ChatParser::with_tools("<|start|>assistant", format, Some(Tools::MuseSpark));
            let _ = p.push(text);
            let _ = p.finish();
            let m = p.message().clone();
            assert_eq!(
                (m.reasoning.as_str(), m.content.as_str()),
                (want_reasoning, want_content),
                "{format:?}"
            );
        }
    }
}
