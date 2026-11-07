//! The think-span splitter for chat templates that tag their reasoning.
//!
//! The tags are the model's ([`ThinkTags`]): `<think>` and `</think>` for most
//! families, the seat's own pair for one that writes others
//! ([`crate::Server::with_think_tags`]). The text below says `<think>` for
//! either.
//!
//! What the prompt ends with decides the start state, not a request flag. A
//! template ends its generation prompt with `<think>` when it opens the span
//! itself (V4.1's, GLM's, Qwen3.8's `<think>\n`) and with the closed span
//! when thinking is off, so the model's output starts inside the span or
//! past it; one that teaches the tags but leaves the span to the model
//! (Qwen3's thinking-on prompt ends at `assistant`) lets the model open it
//! on a leading `<think>`, as llama-server's parser takes an optional
//! leading span and its budget sampler starts counting at the start tag.
//! Inside the span everything up to the first `</think>` is reasoning (or up
//! to a call opener the parser names, which ends the span and stays in the
//! output — [`ThinkSplit::ending_at`]); the span is not re-entered, so a later
//! `<think>` is content. A span still
//! open when the stream ends leaves all its text in reasoning and none in
//! content. A tag the vocabulary holds as one special token is a tag only
//! where that token wrote it ([`SpecialTags`]): the same characters spelled
//! by ordinary tokens are text.
//!
//! A template that writes the answer in a section of its own (Kimi K3's
//! `<|open|>response<|sep|>` … `<|close|>response<|sep|>`, [`ResponseTags`])
//! has its tags taken out of the content: the one that opens the section where
//! the span ends, and the one that closes it where it stands. They are markup,
//! never the answer, and so is what follows the close (the message's own
//! `<|close|>message<|sep|>`): the answer ends with its section.

use serde_json::Value;

use crate::callstream::held;
use crate::engine::Tokenizer;

/// The markers a vocabulary holds as one special token each
/// ([`Tokenizer::is_special`]). The output parsers read such a marker only
/// where that token wrote it, so a model that quotes `</think>` or copies a
/// `<tool_call>` example in ordinary tokens writes text, not markup. A marker
/// with no special token of its own is read by its text.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SpecialTags(Vec<&'static str>);

impl SpecialTags {
    /// Those of `tags` that `tok` encodes to one special id.
    pub(crate) fn of(tok: &dyn Tokenizer, tags: &[&'static str]) -> SpecialTags {
        SpecialTags(
            tags.iter()
                .copied()
                .filter(|tag| matches!(tok.encode(tag)[..], [id] if tok.is_special(id)))
                .collect(),
        )
    }

    fn special(&self, tag: &str) -> bool {
        self.0.contains(&tag)
    }

    /// Whether the marker `tag` starts at byte `at` of `m`: for a special
    /// tag, one whole special token wrote it there.
    pub(crate) fn at(&self, m: &Marked, at: usize, tag: &str) -> bool {
        m.text.get(at..).is_some_and(|t| t.starts_with(tag))
            && (!self.special(tag) || m.specials.binary_search(&(at, tag.len())).is_ok())
    }

    /// Where the first marker `tag` starts in `m`.
    pub(crate) fn find(&self, m: &Marked, tag: &str) -> Option<usize> {
        if self.special(tag) {
            m.specials
                .iter()
                .copied()
                .find(|&(at, len)| len == tag.len() && m.text[at..].starts_with(tag))
                .map(|(at, _)| at)
        } else {
            m.text.find(tag)
        }
    }
}

/// Generated text and the special tokens' texts in it, ascending: each as
/// its byte offset and byte length.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Marked {
    pub(crate) text: String,
    pub(crate) specials: Vec<(usize, usize)>,
}

impl Marked {
    /// Appends `text`, its special tokens' texts at `specials`.
    pub(crate) fn push(&mut self, text: &str, specials: &[(usize, usize)]) {
        let base = self.text.len();
        self.specials
            .extend(specials.iter().map(|&(at, len)| (base + at, len)));
        self.text.push_str(text);
    }

    /// Takes out the first `n` bytes.
    pub(crate) fn take_front(&mut self, n: usize) -> Marked {
        let k = self.specials.partition_point(|&(at, _)| at < n);
        let rest = Marked {
            text: self.text.split_off(n),
            specials: self
                .specials
                .split_off(k)
                .iter()
                .map(|&(at, len)| (at - n, len))
                .collect(),
        };
        std::mem::replace(self, rest)
    }

    /// Releases a text prefix, transferring its allocation into an empty
    /// output instead of copying it before the prefix is dropped.
    pub(crate) fn take_text_into(&mut self, n: usize, out: &mut String) {
        let front = self.take_front(n).text;
        if out.is_empty() {
            *out = front;
        } else {
            out.push_str(&front);
        }
    }
}

/// The two tags of a think span: the text the model writes to open its
/// reasoning and to end it. Both are in the model's vocabulary, and the chat
/// template writes the same pair around the history's reasoning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThinkTags {
    /// The template's `thinking_start_token`.
    pub open: &'static str,
    /// The template's `thinking_end_token`.
    pub close: &'static str,
    /// The answer's section after the span, for a template that writes one.
    pub response: Option<ResponseTags>,
}

/// The two tags of the section a template writes the answer in: markup the
/// content never carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResponseTags {
    /// What opens the answer, after the span's close (or in the generation
    /// prompt, when thinking is off).
    pub open: &'static str,
    /// What ends the answer.
    pub close: &'static str,
}

impl ThinkTags {
    /// `<think>` and `</think>`: what a server without a seat's tags
    /// ([`crate::Server::with_think_tags`]) splits on.
    pub const THINK: ThinkTags = ThinkTags {
        open: "<think>",
        close: "</think>",
        response: None,
    };
}

#[cfg(test)]
impl ThinkTags {
    /// A pair that is not `<think>`, Solar-Open2's, for the tests that give a
    /// seat its own tags.
    pub(crate) const SOLAR: ThinkTags = ThinkTags {
        open: "<|think:start|>",
        close: "<|think:end|>",
        response: None,
    };
}

/// [`ThinkTags::THINK`]'s end tag, for a caller that serves a model on the
/// default tags. Code that writes the close takes the server's tags
/// (`State.think`), never this.
pub const THINK_CLOSE: &str = ThinkTags::THINK.close;

/// The request's `reasoning_format` (llama-server's field and names).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningFormat {
    /// The raw text, think tags included, in `content`.
    None,
    /// The span in `reasoning_content`, the rest in `content`.
    Deepseek,
}

impl ReasoningFormat {
    /// Reads the field: absent, `null`, `auto` and `deepseek` split, `none` does
    /// not. Anything else (`deepseek-legacy` included) is an error naming the value.
    pub fn from_request(v: Option<&Value>) -> Result<Self, String> {
        match v {
            None | Some(Value::Null) => Ok(ReasoningFormat::Deepseek),
            Some(Value::String(s)) => match s.as_str() {
                "none" => Ok(ReasoningFormat::None),
                "auto" | "deepseek" => Ok(ReasoningFormat::Deepseek),
                other => Err(format!(
                    "reasoning_format '{other}' is not supported by this server (none, auto, deepseek)"
                )),
            },
            Some(other) => Err(format!("reasoning_format must be a string, got {other}")),
        }
    }
}

/// Where a rendered prompt leaves the think span when the model's output
/// starts — the template property that says who opens the span. A template
/// that ends its generation prompt with the start tag starts the model
/// inside it; one that ends with the closed span (thinking off) has no
/// reasoning to split; one that teaches the tags but ends with neither
/// leaves the span to the model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ThinkEntry {
    /// The prompt ends with the closed span (its trailing whitespace with
    /// it), or with the answer's section already open: no reasoning is split
    /// or capped.
    #[default]
    Closed,
    /// The prompt ends with the start tag (its trailing whitespace with it):
    /// the output starts inside the span.
    Open,
    /// The prompt ends with neither tag: the model may open the span on a
    /// leading `<think>`, leading whitespace allowed.
    MayOpen,
}

impl ThinkEntry {
    /// Where `prompt` leaves a `<think>` span ([`ThinkEntry::of_prompt_with`]
    /// with [`ThinkTags::THINK`]).
    #[must_use]
    pub fn of_prompt(prompt: &str) -> ThinkEntry {
        ThinkEntry::of_prompt_with(prompt, ThinkTags::THINK)
    }

    /// Where `prompt` leaves the span `tags` name. The tags are matched at
    /// the prompt's end with the whitespace around them, as llama-server's
    /// parser anchors its generated prompt at the start tag and reads the
    /// whitespace around the tags as neither reasoning nor content.
    #[must_use]
    pub fn of_prompt_with(prompt: &str, tags: ThinkTags) -> ThinkEntry {
        let end = prompt.trim_end();
        if end.ends_with(tags.open) {
            ThinkEntry::Open
        } else if end.ends_with(tags.close) || tags.response.is_some_and(|r| end.ends_with(r.open))
        {
            // The answer's section already open ends the span too.
            ThinkEntry::Closed
        } else {
            ThinkEntry::MayOpen
        }
    }
}

/// What one push produced. Reasoning always precedes content in the stream, so a
/// push that crosses `</think>` has both.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Split {
    pub reasoning: String,
    /// A leading `<think>` was consumed by this push: the model opened the
    /// span.
    pub opened: bool,
    /// `</think>` was consumed by this push.
    pub closed: bool,
    pub content: String,
    /// The special tokens' texts in `content`: byte offset and length.
    pub specials: Vec<(usize, usize)>,
}

impl Split {
    fn content(m: Marked) -> Split {
        Split {
            content: m.text,
            specials: m.specials,
            ..Split::default()
        }
    }
}

/// Streaming splitter. Inside the span it holds back a tail that could still grow
/// into `</think>`; outside it passes text through untouched — unless the model
/// may still open the span, where it holds everything until the first output
/// that is neither the tag nor leading whitespace.
#[derive(Clone, Debug)]
pub struct ThinkSplit {
    tags: ThinkTags,
    inside: bool,
    /// The model may still open the span on a leading `<think>`: true until
    /// the output names the tag, or names any other text first.
    may_open: bool,
    held: Marked,
    /// The tags read only where their special token wrote them.
    specials: SpecialTags,
    /// A tag that ends the span too and stays in the output: a call the
    /// markup lets follow the reasoning without `</think>`.
    ends_too: Option<&'static str>,
    /// Content past every span starts at its first non-whitespace character
    /// (Qwen3-Coder's markup: llama.cpp's parser reads `reasoning << content`).
    trim_content: bool,
    /// The whitespace that opens the content is still being dropped.
    trimming: bool,
    /// A span the model opens with its own `<think>` drops the whitespace
    /// after its `</think>` ([`ThinkSplit::trimming_after_opened`]).
    trim_after_opened: bool,
    /// Where the answer's section stands, for tags that name one.
    section: Section,
    /// A content tail that could still grow into the section's tag.
    tail: Marked,
}

/// How far the content is into the answer's section.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Section {
    /// Before the answer's text: a leading open tag is taken out (a prompt
    /// that opened the section itself leaves none).
    Opening,
    /// In the answer: up to its close tag is content.
    Answer,
    /// Past the answer's close: the rest is the message's own markup, dropped.
    Ended,
}

impl ThinkSplit {
    /// The start state for a rendered prompt, on `<think>` tags.
    #[must_use]
    pub fn for_prompt(prompt: &str) -> Self {
        Self::for_prompt_with(prompt, ThinkTags::THINK)
    }

    /// The start state for a rendered prompt, on `tags`.
    #[must_use]
    pub fn for_prompt_with(prompt: &str, tags: ThinkTags) -> Self {
        Self::for_entry_with(ThinkEntry::of_prompt_with(prompt, tags), tags)
    }

    /// The start state for where the prompt left the span `tags` name.
    #[must_use]
    pub fn for_entry_with(entry: ThinkEntry, tags: ThinkTags) -> Self {
        ThinkSplit {
            tags,
            inside: entry == ThinkEntry::Open,
            may_open: entry == ThinkEntry::MayOpen,
            held: Marked::default(),
            specials: SpecialTags::default(),
            ends_too: None,
            trim_content: false,
            trimming: false,
            trim_after_opened: false,
            section: Section::Opening,
            tail: Marked::default(),
        }
    }

    /// The same splitter, its span also ended by `tag`, which stays in the
    /// output as the first text past the span (Qwen3-Coder's `<tool_call>`,
    /// one of llama.cpp's `thinking_end_tags` for its markup).
    #[must_use]
    pub fn ending_at(mut self, tag: &'static str) -> Self {
        self.ends_too = Some(tag);
        self
    }

    /// The same splitter, its span ended by the close tag alone
    /// ([`ThinkSplit::ending_at`] undone).
    #[must_use]
    pub(crate) fn ending_only_at_close(mut self) -> Self {
        self.ends_too = None;
        self
    }

    /// The same splitter, dropping the whitespace that opens the content on
    /// every path past the span: the span closed by the prompt, closed by
    /// `</think>`, or never opened. Qwen3-Coder's markup reads so in
    /// llama.cpp (`reasoning << content`, a space between the two).
    #[must_use]
    pub fn trimming_content(mut self) -> Self {
        self.trim_content = true;
        self.trimming = !self.inside && !self.may_open;
        self
    }

    /// The same splitter, dropping the whitespace after the `</think>` of a
    /// span the model opened, as llama.cpp's optional leading span takes it
    /// when reasoning is split out.
    #[must_use]
    pub fn trimming_after_opened(mut self) -> Self {
        self.trim_after_opened = true;
        self
    }

    /// The same splitter, reading its tags and the tag that ends its span
    /// too ([`ThinkSplit::ending_at`], set before this) only where their
    /// special tokens wrote them, for those `tok` holds as one special token.
    #[must_use]
    pub(crate) fn reading_specials(mut self, tok: &dyn Tokenizer) -> Self {
        let tags = [self.tags.open, self.tags.close];
        let response = self.tags.response.map(|r| [r.open, r.close]);
        self.specials = SpecialTags::of(
            tok,
            &[
                &tags[..],
                self.ends_too.as_slice(),
                response.as_ref().map_or(&[][..], |r| &r[..]),
            ]
            .concat(),
        );
        self
    }

    /// Whether the output is inside the span: the reasoning budget counts
    /// only there.
    pub(crate) fn in_span(&self) -> bool {
        self.inside
    }

    /// Feeds generated text, its special tokens' texts at `specials` (byte
    /// offset and length).
    pub fn push(&mut self, piece: &str, specials: &[(usize, usize)]) -> Split {
        if self.may_open {
            return self.may_open_step(piece, specials);
        }
        if !self.inside {
            let mut m = Marked::default();
            m.push(piece, specials);
            return Split::content(self.past_span(m));
        }
        self.in_span_step(piece, specials)
    }

    /// Text past the span: as given, or with its opening whitespace dropped
    /// while the content is still being trimmed.
    fn past_span(&mut self, mut m: Marked) -> Marked {
        if self.trimming {
            let ws = m.text.len() - m.text.trim_start_matches(is_space).len();
            self.trimming = ws == m.text.len();
            m.take_front(ws);
        }
        self.in_section(m)
    }

    /// Text past the span with the answer's section tags taken out: the open
    /// tag where the answer starts, the close tag where it ends. A tail that
    /// could still grow into the tag it waits for is held to the next push or
    /// to [`ThinkSplit::finish`]. Tags that name no section leave the text as it is.
    fn in_section(&mut self, m: Marked) -> Marked {
        let Some(tags) = self.tags.response else {
            return m;
        };
        let mut rest = std::mem::take(&mut self.tail);
        rest.push(&m.text, &m.specials);
        let mut out = Marked::default();
        loop {
            match self.section {
                Section::Opening => {
                    if self.specials.at(&rest, 0, tags.open) {
                        rest.take_front(tags.open.len());
                    } else if held(tags.open, &rest.text) {
                        self.tail = rest;
                        return out;
                    }
                    self.section = Section::Answer;
                }
                Section::Answer => {
                    if let Some(at) = self.specials.find(&rest, tags.close) {
                        let front = rest.take_front(at);
                        out.push(&front.text, &front.specials);
                        rest.take_front(tags.close.len());
                        self.section = Section::Ended;
                    } else {
                        let keep = partial_suffix(&rest.text, tags.close);
                        let front = rest.take_front(rest.text.len() - keep);
                        out.push(&front.text, &front.specials);
                        self.tail = rest;
                        return out;
                    }
                }
                Section::Ended => return out,
            }
        }
    }

    /// The content tail held for a section tag, released: the stream ended
    /// before the tag was whole, so it was text.
    fn release_tail(&mut self) -> Marked {
        std::mem::take(&mut self.tail)
    }

    /// Outside a span the model may still open. Leading whitespace and a
    /// partial tag are held — they precede neither content nor the span
    /// until one follows — a leading `<think>` enters the span consuming the
    /// whitespace and the tag (as llama-server's optional leading span takes
    /// them, into neither field), and any other text ends the chance: from
    /// there the text passes through.
    fn may_open_step(&mut self, piece: &str, specials: &[(usize, usize)]) -> Split {
        self.held.push(piece, specials);
        let ws = self.held.text.len() - self.held.text.trim_start().len();
        if self.specials.at(&self.held, ws, self.tags.open) {
            self.may_open = false;
            self.inside = true;
            self.trim_content |= self.trim_after_opened;
            self.held.take_front(ws + self.tags.open.len());
            let mut s = self.in_span_step("", &[]);
            s.opened = true;
            return s;
        }
        let rest = &self.held.text[ws..];
        if rest.is_empty() || held(self.tags.open, rest) {
            return Split::default();
        }
        self.may_open = false;
        self.trimming = self.trim_content;
        let held = std::mem::take(&mut self.held);
        Split::content(self.past_span(held))
    }

    /// Inside the span: everything up to the first `</think>` (or the tag
    /// that ends it too) is reasoning, a tail that could still grow into
    /// either held back.
    fn in_span_step(&mut self, piece: &str, specials: &[(usize, usize)]) -> Split {
        self.held.push(piece, specials);
        let close = self.specials.find(&self.held, self.tags.close);
        let call = self
            .ends_too
            .and_then(|tag| self.specials.find(&self.held, tag));
        if let Some(at) = call.filter(|&c| close.is_none_or(|t| c < t)) {
            self.inside = false;
            let reasoning = self.held.take_front(at).text;
            return Split {
                reasoning,
                ..Split::content(std::mem::take(&mut self.held))
            };
        }
        if let Some(at) = close {
            self.inside = false;
            self.trimming = self.trim_content;
            let reasoning = self.held.take_front(at).text;
            let mut after = std::mem::take(&mut self.held);
            after.take_front(self.tags.close.len());
            return Split {
                reasoning,
                closed: true,
                ..Split::content(self.past_span(after))
            };
        }
        let keep = self
            .ends_too
            .map_or(0, |tag| partial_suffix(&self.held.text, tag))
            .max(partial_suffix(&self.held.text, self.tags.close));
        Split {
            reasoning: self.held.take_front(self.held.text.len() - keep).text,
            ..Split::default()
        }
    }

    /// Releases the held tail at the end of the stream (an open span keeps it as reasoning).
    pub fn finish(&mut self) -> Split {
        if self.may_open {
            self.may_open = false;
            self.trimming = self.trim_content;
            let held = std::mem::take(&mut self.held);
            let mut m = self.past_span(held);
            let tail = self.release_tail();
            m.push(&tail.text, &tail.specials);
            return Split::content(m);
        }
        let tail = self.release_tail();
        Split {
            reasoning: std::mem::take(&mut self.held).text,
            ..Split::content(tail)
        }
    }
}

/// The whitespace llama.cpp's parser skips between its parts: C's `isspace`
/// over bytes, the six ASCII spaces and no other.
pub(crate) fn is_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r')
}

/// Length of the longest suffix of `text` that is a proper prefix of `pat`, cut on
/// a char boundary.
pub(crate) fn partial_suffix(text: &str, pat: &str) -> usize {
    let t = text.as_bytes();
    let p = pat.as_bytes();
    (1..p.len().min(t.len() + 1))
        .rev()
        .find(|&k| t[t.len() - k..] == p[..k] && text.is_char_boundary(t.len() - k))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holds_a_partial_close_tag() {
        let mut s = ThinkSplit::for_prompt("<｜Assistant｜><think>");
        assert_eq!(s.push("abc</th", &[]).reasoning, "abc");
        let p = s.push("ink>x", &[]);
        assert_eq!(
            (p.reasoning.as_str(), p.closed, p.content.as_str()),
            ("", true, "x")
        );
        assert_eq!(s.push("<think>y", &[]).content, "<think>y");
    }

    #[test]
    fn a_closed_prompt_passes_through() {
        let mut s = ThinkSplit::for_prompt("<｜Assistant｜></think>");
        assert_eq!(s.push("a</think>b", &[]).content, "a</think>b");
    }

    #[test]
    fn partial_suffix_is_proper() {
        assert_eq!(partial_suffix("x</think>", THINK_CLOSE), 0);
        assert_eq!(partial_suffix("x</thin", THINK_CLOSE), 6);
        assert_eq!(partial_suffix("<", THINK_CLOSE), 1);
    }

    /// A prompt opens the span only when its text ends with `<think>` — the
    /// whitespace after the tag with it (Qwen3.8 writes `<think>\n`) — closes
    /// it when it ends with `</think>` (thinking off writes the closed empty
    /// span), and leaves it to the model otherwise: the one rule the
    /// think-span budget's span check and the split's start state share.
    #[test]
    fn a_prompt_names_who_opens_the_span() {
        assert_eq!(
            ThinkEntry::of_prompt("<｜Assistant｜><think>"),
            ThinkEntry::Open
        );
        assert_eq!(
            ThinkEntry::of_prompt("<|im_start|>assistant\n<think>\n"),
            ThinkEntry::Open
        );
        assert_eq!(
            ThinkEntry::of_prompt("<｜Assistant｜></think>"),
            ThinkEntry::Closed
        );
        assert_eq!(
            ThinkEntry::of_prompt("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            ThinkEntry::Closed
        );
        assert_eq!(
            ThinkEntry::of_prompt("<|im_start|>assistant\n"),
            ThinkEntry::MayOpen
        );
        assert_eq!(ThinkEntry::of_prompt("<think>x"), ThinkEntry::MayOpen);
    }

    /// A prompt that leaves the span to the model: the model's leading
    /// `<think>` opens it — the whitespace before the tag and the tag itself
    /// in neither field — and any other text first leaves the whole
    /// generation as content, a later tag included. Opted in, the whitespace
    /// after its `</think>` is in neither field too.
    #[test]
    fn a_model_opened_span_enters_on_a_leading_tag() {
        let mut s = ThinkSplit::for_prompt("<|im_start|>assistant\n");
        let p = s.push(" \n<think>\nwhy\n</think>\nHello", &[]);
        assert_eq!(
            (p.reasoning.as_str(), p.opened, p.closed, p.content.as_str()),
            ("\nwhy\n", true, true, "\nHello")
        );
        let mut s = ThinkSplit::for_prompt("<|im_start|>assistant\n").trimming_after_opened();
        let p = s.push(" \n<think>\nwhy\n</think>\n", &[]);
        assert_eq!((p.reasoning.as_str(), p.content.as_str()), ("\nwhy\n", ""));
        assert_eq!(s.push(" Hello", &[]).content, "Hello");
        // Whitespace alone decides nothing; at the end it is content.
        let mut s = ThinkSplit::for_prompt("<|im_start|>assistant\n");
        assert_eq!(s.push("  ", &[]), Split::default());
        assert_eq!(s.finish().content, "  ");
        // Text before the tag leaves everything content, the tag included.
        let mut s = ThinkSplit::for_prompt("<|im_start|>assistant\n");
        assert_eq!(s.push("hi <think>x", &[]).content, "hi <think>x");
        assert_eq!(s.push("</think>y", &[]).content, "</think>y");
        // A partial tag is held until the text names it.
        let mut s = ThinkSplit::for_prompt("<|im_start|>assistant\n");
        assert_eq!(s.push("<thi", &[]), Split::default());
        assert_eq!(s.push("nk>ok", &[]).reasoning, "ok");
    }

    /// The splitter reads the tags it is given and no others: with Solar's
    /// pair a prompt ends in the open span at `<|think:start|>` and in the
    /// closed one at `<|think:end|>`, the output splits at `<|think:end|>`
    /// (a tag cut across pushes held until it is whole), `<think>` is text in
    /// the reasoning, and a model that opens the span itself opens it with
    /// `<|think:start|>`. On `<think>` tags the same prompts leave the span to
    /// the model.
    #[test]
    fn a_splitter_reads_the_tags_it_is_given() {
        const SOLAR: ThinkTags = ThinkTags {
            open: "<|think:start|>",
            close: "<|think:end|>",
            response: None,
        };
        let open = "<|im:start|>assistant<|im:content|><|think:start|>";
        let closed = "<|im:start|>assistant<|im:content|><|think:start|><|think:end|>";
        assert_eq!(ThinkEntry::of_prompt_with(open, SOLAR), ThinkEntry::Open);
        assert_eq!(
            ThinkEntry::of_prompt_with(closed, SOLAR),
            ThinkEntry::Closed
        );
        assert_eq!(ThinkEntry::of_prompt(open), ThinkEntry::MayOpen);
        assert_eq!(ThinkEntry::of_prompt(closed), ThinkEntry::MayOpen);

        let mut s = ThinkSplit::for_prompt_with(open, SOLAR);
        assert_eq!(s.push("why <think>", &[]).reasoning, "why <think>");
        assert_eq!(s.push("x<|think:en", &[]).reasoning, "x");
        let p = s.push("d|>ans", &[]);
        assert_eq!(
            (p.reasoning.as_str(), p.closed, p.content.as_str()),
            ("", true, "ans")
        );
        assert_eq!(s.push("<|think:end|>", &[]).content, "<|think:end|>");

        let mut s = ThinkSplit::for_prompt_with("<|im:start|>assistant<|im:content|>", SOLAR);
        assert_eq!(s.push("<|think:sta", &[]), Split::default());
        let p = s.push("rt|>why<|think:end|>ok", &[]);
        assert_eq!(
            (p.reasoning.as_str(), p.opened, p.closed, p.content.as_str()),
            ("why", true, true, "ok")
        );

        let mut s = ThinkSplit::for_prompt(open);
        assert_eq!(s.push("a<|think:end|>b", &[]).content, "a<|think:end|>b");
    }

    /// Kimi K3's tags, as its template spells them (`otag`/`ctag`).
    const K3: ThinkTags = ThinkTags {
        open: "<|open|>think<|sep|>",
        close: "<|close|>think<|sep|>",
        response: Some(ResponseTags {
            open: "<|open|>response<|sep|>",
            close: "<|close|>response<|sep|>",
        }),
    };

    /// The split of `text` fed as `pieces` of at most `step` chars, then
    /// finished: (reasoning, content).
    fn split_k3(prompt: &str, text: &str, step: usize) -> (String, String) {
        let mut s = ThinkSplit::for_prompt_with(prompt, K3);
        let chars: Vec<char> = text.chars().collect();
        let (mut reasoning, mut content) = (String::new(), String::new());
        for piece in chars.chunks(step) {
            let p = s.push(&piece.iter().collect::<String>(), &[]);
            reasoning.push_str(&p.reasoning);
            content.push_str(&p.content);
        }
        let p = s.finish();
        reasoning.push_str(&p.reasoning);
        content.push_str(&p.content);
        (reasoning, content)
    }

    /// A template that writes the answer in a section: the output starts in
    /// the span, `<|close|>think<|sep|>` ends it, the response's open tag
    /// and close tag are markup taken out of the content, however the text is
    /// cut into pushes; a prompt that opened the section itself (thinking off)
    /// leaves no open tag to take out, a tag cut short by the end of the
    /// stream was text, and what follows the answer's close (the message's
    /// own close) is markup dropped. Mutant: the text past the close kept.
    #[test]
    fn a_response_section_is_markup_not_content() {
        let on = "<|open|>message assistant<|sep|><|open|>think<|sep|>";
        let off = "<|open|>message assistant<|sep|><|open|>response<|sep|>";
        assert_eq!(ThinkEntry::of_prompt_with(on, K3), ThinkEntry::Open);
        assert_eq!(ThinkEntry::of_prompt_with(off, K3), ThinkEntry::Closed);
        let reply = "why<|close|>think<|sep|><|open|>response<|sep|>서울<|close|>response<|sep|>";
        // The message's own close follows the answer's: markup, not content.
        let whole = "why<|close|>think<|sep|><|open|>response<|sep|>서울<|close|>response<|sep|>\
                     <|close|>message<|sep|>";
        for step in [1, 2, 3, 7, 1000] {
            assert_eq!(
                split_k3(on, whole, step),
                ("why".to_owned(), "서울".to_owned()),
                "whole message, step {step}"
            );
            assert_eq!(
                split_k3(on, reply, step),
                ("why".to_owned(), "서울".to_owned()),
                "step {step}"
            );
            assert_eq!(
                split_k3(off, "서울<|close|>response<|sep|>", step),
                (String::new(), "서울".to_owned()),
                "off, step {step}"
            );
        }
        assert_eq!(
            split_k3(
                on,
                "why<|close|>think<|sep|><|open|>response<|sep|>ab<|close|>resp",
                4
            ),
            ("why".to_owned(), "ab<|close|>resp".to_owned())
        );
        // The answer's own text reads on past a lookalike of the close.
        assert_eq!(
            split_k3(off, "a<|close|>x<|close|>response<|sep|>", 5),
            (String::new(), "a<|close|>x".to_owned())
        );
        // No response open after the span: the text is the answer all the same.
        assert_eq!(
            split_k3(on, "why<|close|>think<|sep|>서울", 3),
            ("why".to_owned(), "서울".to_owned())
        );
        // Tags with no section leave the content whole.
        let mut s = ThinkSplit::for_prompt("<think>");
        assert_eq!(
            s.push("a</think><|open|>response<|sep|>b", &[]).content,
            "<|open|>response<|sep|>b"
        );
    }

    /// The opening whitespace dropped is C's `isspace` over bytes, as
    /// llama.cpp's parser skips it: `\x0B` and `\x0C` go, U+3000 and U+00A0
    /// stay, and a piece of whitespace alone waits for the text after it.
    #[test]
    fn the_content_drops_the_six_ascii_spaces_and_no_other() {
        let mut s = ThinkSplit::for_prompt("<think></think>").trimming_content();
        assert_eq!(s.push(" \t\n", &[]).content, "");
        assert_eq!(s.push("\x0B\x0C\rHello \n", &[]).content, "Hello \n");
        let mut s = ThinkSplit::for_prompt("<think></think>").trimming_content();
        assert_eq!(s.push("\u{3000}Hello", &[]).content, "\u{3000}Hello");
        let mut s = ThinkSplit::for_prompt("<think></think>").trimming_content();
        assert_eq!(s.push("\u{a0}Hello", &[]).content, "\u{a0}Hello");
    }
}
