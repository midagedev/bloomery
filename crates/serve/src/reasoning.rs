//! The think-span splitter for chat templates that tag their reasoning.
//!
//! What the prompt ends with decides the start state, not a request flag. A
//! template ends its generation prompt with `<think>` when it opens the span
//! itself (V4.1's, GLM's, Qwen3.8's `<think>\n`) and with the closed span
//! when thinking is off, so the model's output starts inside the span or
//! past it; one that teaches the tags but leaves the span to the model
//! (Qwen3's thinking-on prompt ends at `assistant`) lets the model open it
//! on a leading `<think>`, as llama-server's parser takes an optional
//! leading span and its budget sampler starts counting at the start tag.
//! Inside the span everything up to the first `</think>` is reasoning; the
//! span is not re-entered, so a later `<think>` is content. A span still
//! open when the stream ends leaves all its text in reasoning and none in
//! content.

use serde_json::Value;

/// The template's `thinking_start_token`.
pub const THINK_OPEN: &str = "<think>";
/// The template's `thinking_end_token`.
pub const THINK_CLOSE: &str = "</think>";

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
    /// it): no reasoning is split or capped.
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
    /// Where `prompt` leaves the span. The tags are matched at the prompt's
    /// end with the whitespace around them, as llama-server's parser anchors
    /// its generated prompt at the start tag and reads the whitespace around
    /// the tags as neither reasoning nor content.
    #[must_use]
    pub fn of_prompt(prompt: &str) -> ThinkEntry {
        let end = prompt.trim_end();
        if end.ends_with(THINK_OPEN) {
            ThinkEntry::Open
        } else if end.ends_with(THINK_CLOSE) {
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
}

/// Streaming splitter. Inside the span it holds back a tail that could still grow
/// into `</think>`; outside it passes text through untouched — unless the model
/// may still open the span, where it holds everything until the first output
/// that is neither the tag nor leading whitespace.
#[derive(Debug)]
pub struct ThinkSplit {
    inside: bool,
    /// The model may still open the span on a leading `<think>`: true until
    /// the output names the tag, or names any other text first.
    may_open: bool,
    held: String,
}

impl ThinkSplit {
    /// The start state for a rendered prompt.
    #[must_use]
    pub fn for_prompt(prompt: &str) -> Self {
        Self::for_entry(ThinkEntry::of_prompt(prompt))
    }

    /// The start state for where the prompt left the span.
    #[must_use]
    pub fn for_entry(entry: ThinkEntry) -> Self {
        ThinkSplit {
            inside: entry == ThinkEntry::Open,
            may_open: entry == ThinkEntry::MayOpen,
            held: String::new(),
        }
    }

    /// Whether the output is inside the span: the reasoning budget counts
    /// only there.
    pub(crate) fn in_span(&self) -> bool {
        self.inside
    }

    /// Feeds generated text.
    pub fn push(&mut self, piece: &str) -> Split {
        if self.may_open {
            return self.may_open_step(piece);
        }
        if !self.inside {
            return Split {
                content: piece.to_owned(),
                ..Split::default()
            };
        }
        self.in_span_step(piece)
    }

    /// Outside a span the model may still open. Leading whitespace and a
    /// partial tag are held — they precede neither content nor the span
    /// until one follows — a leading `<think>` enters the span consuming the
    /// whitespace and the tag (as llama-server's optional leading span takes
    /// them, into neither field), and any other text ends the chance: from
    /// there the text passes through.
    fn may_open_step(&mut self, piece: &str) -> Split {
        self.held.push_str(piece);
        let ws = self.held.len() - self.held.trim_start().len();
        let rest = &self.held[ws..];
        if let Some(tail) = rest.strip_prefix(THINK_OPEN).map(str::to_owned) {
            self.may_open = false;
            self.inside = true;
            self.held.clear();
            let mut s = self.in_span_step(&tail);
            s.opened = true;
            return s;
        }
        if rest.is_empty() || THINK_OPEN.starts_with(rest) {
            return Split::default();
        }
        self.may_open = false;
        Split {
            content: std::mem::take(&mut self.held),
            ..Split::default()
        }
    }

    /// Inside the span: everything up to the first `</think>` is reasoning,
    /// a tail that could still grow into the close held back.
    fn in_span_step(&mut self, piece: &str) -> Split {
        self.held.push_str(piece);
        if let Some(at) = self.held.find(THINK_CLOSE) {
            self.inside = false;
            let content = self.held[at + THINK_CLOSE.len()..].to_owned();
            self.held.truncate(at);
            return Split {
                reasoning: std::mem::take(&mut self.held),
                closed: true,
                content,
                ..Split::default()
            };
        }
        let keep = partial_suffix(&self.held, THINK_CLOSE);
        let rest = self.held.split_off(self.held.len() - keep);
        Split {
            reasoning: std::mem::replace(&mut self.held, rest),
            ..Split::default()
        }
    }

    /// Releases the held tail at the end of the stream (an open span keeps it as reasoning).
    pub fn finish(&mut self) -> Split {
        if self.may_open {
            self.may_open = false;
            return Split {
                content: std::mem::take(&mut self.held),
                ..Split::default()
            };
        }
        Split {
            reasoning: std::mem::take(&mut self.held),
            ..Split::default()
        }
    }
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
        assert_eq!(s.push("abc</th").reasoning, "abc");
        let p = s.push("ink>x");
        assert_eq!(
            (p.reasoning.as_str(), p.closed, p.content.as_str()),
            ("", true, "x")
        );
        assert_eq!(s.push("<think>y").content, "<think>y");
    }

    #[test]
    fn a_closed_prompt_passes_through() {
        let mut s = ThinkSplit::for_prompt("<｜Assistant｜></think>");
        assert_eq!(s.push("a</think>b").content, "a</think>b");
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
    /// generation as content, a later tag included.
    #[test]
    fn a_model_opened_span_enters_on_a_leading_tag() {
        let mut s = ThinkSplit::for_prompt("<|im_start|>assistant\n");
        let p = s.push(" \n<think>\nwhy\n</think>\nHello");
        assert_eq!(
            (p.reasoning.as_str(), p.opened, p.closed, p.content.as_str()),
            ("\nwhy\n", true, true, "\nHello")
        );
        // Whitespace alone decides nothing; at the end it is content.
        let mut s = ThinkSplit::for_prompt("<|im_start|>assistant\n");
        assert_eq!(s.push("  "), Split::default());
        assert_eq!(s.finish().content, "  ");
        // Text before the tag leaves everything content, the tag included.
        let mut s = ThinkSplit::for_prompt("<|im_start|>assistant\n");
        assert_eq!(s.push("hi <think>x").content, "hi <think>x");
        assert_eq!(s.push("</think>y").content, "</think>y");
        // A partial tag is held until the text names it.
        let mut s = ThinkSplit::for_prompt("<|im_start|>assistant\n");
        assert_eq!(s.push("<thi"), Split::default());
        assert_eq!(s.push("nk>ok").reasoning, "ok");
    }
}
