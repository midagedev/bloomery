//! The think-span splitter for the V4.1 chat template.
//!
//! The template ends its generation prompt with `<think>` when thinking is on and
//! with `</think>` when it is off, so the model's output starts inside the span or
//! after it. What the prompt ends with decides the start state, not a request flag:
//! a prompt that already closed the span has no reasoning to split. Inside the
//! span everything up to the first `</think>` is reasoning; the span is not
//! re-entered, so a later `<think>` is content. A span still open when the stream
//! ends leaves all its text in reasoning and none in content.

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

/// What one push produced. Reasoning always precedes content in the stream, so a
/// push that crosses `</think>` has both.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Split {
    pub reasoning: String,
    /// `</think>` was consumed by this push.
    pub closed: bool,
    pub content: String,
}

/// Streaming splitter. Inside the span it holds back a tail that could still grow
/// into `</think>`; outside it passes text through untouched.
#[derive(Debug)]
pub struct ThinkSplit {
    inside: bool,
    held: String,
}

impl ThinkSplit {
    /// The start state for a rendered prompt.
    #[must_use]
    pub fn for_prompt(prompt: &str) -> Self {
        ThinkSplit {
            inside: prompt.ends_with(THINK_OPEN),
            held: String::new(),
        }
    }

    /// Feeds generated text.
    pub fn push(&mut self, piece: &str) -> Split {
        if !self.inside {
            return Split {
                content: piece.to_owned(),
                ..Split::default()
            };
        }
        self.held.push_str(piece);
        if let Some(at) = self.held.find(THINK_CLOSE) {
            self.inside = false;
            let content = self.held[at + THINK_CLOSE.len()..].to_owned();
            self.held.truncate(at);
            return Split {
                reasoning: std::mem::take(&mut self.held),
                closed: true,
                content,
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
}
