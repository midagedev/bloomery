//! The streamed form of a tool call: OpenAI's `tool_calls` deltas, whose first
//! one carries the call's `index`, `id`, `type` and `function.name` and whose
//! next ones carry `function.arguments` fragments, in order, until the call
//! closes. The fragments concatenate to the call's `arguments`.
//!
//! A scan ([`crate::dsml`], [`crate::glmxml`], [`crate::hermes`],
//! [`crate::qwenxml`]) reports the call it has opened and not closed as a
//! [`Progress`]: its name and the arguments text read so far, which is a
//! prefix of the arguments the call will close with. The common owner here
//! (`Streamer`, held by [`crate::dsml::ChatParser`]) turns each report into
//! the [`CallPiece`]s that extend what was sent, and holds the one rule every
//! format shares: what was sent is never revised. A call that closes with
//! arguments the sent text is not a prefix of or under another name than it
//! was announced with, and a call announced that the markup then ends
//! otherwise (its tag never comes, or the text stops being the template's
//! markup), are a [`MarkupError::Diverged`], never a call whose earlier
//! fragments are quietly wrong.
//!
//! Each format reads its arguments in the form its closed call writes them: a
//! text parameter's value is released as it arrives, JSON-escaped, up to the
//! longest tail that could still become a tag that ends the value or the
//! call; every other value is released whole once it is complete.

use serde_json::Value;

use crate::dsml::{MarkupError, ToolCall};
use crate::reasoning::partial_suffix;

/// What a scan knows of the call it has opened and not closed: `args` is a
/// prefix of the arguments the call closes with, and it begins with `{` as
/// soon as the name is known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Progress {
    /// The call's place in the generation, from 0.
    pub index: usize,
    pub name: String,
    pub args: String,
}

/// One step of a streamed call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallPiece {
    /// The call's name is known: the delta with `index`, `id`, `type` and
    /// `function.name`.
    Open { index: usize, name: String },
    /// The next fragment of the call's `arguments`.
    Args { index: usize, text: String },
}

/// A call announced and not closed: what has gone out of it.
#[derive(Debug)]
struct Told {
    index: usize,
    name: String,
    /// The arguments text sent so far.
    sent: String,
}

/// The pieces a parser's scans owe the stream: the call being read, and what
/// went out of it.
#[derive(Debug, Default)]
pub(crate) struct Streamer {
    told: Option<Told>,
    pieces: Vec<CallPiece>,
}

impl Streamer {
    /// Takes one scan step: the `calls` it closed, in order, and the call it
    /// left open. The pieces are for [`Streamer::take`].
    pub(crate) fn step(
        &mut self,
        calls: &[ToolCall],
        open: Option<&Progress>,
    ) -> Result<(), MarkupError> {
        let mut told = self.told.take();
        for c in calls {
            let sent = match told.take() {
                Some(t) if t.index == c.index && t.name == c.name => t.sent,
                Some(t) => return Err(diverged(&t.name, t.index)),
                None => {
                    self.pieces.push(CallPiece::Open {
                        index: c.index,
                        name: c.name.clone(),
                    });
                    String::new()
                }
            };
            self.extend(c.index, &c.name, &sent, &c.arguments)?;
        }
        match (open, told) {
            (Some(p), Some(t)) if t.index == p.index && t.name == p.name => {
                self.extend(p.index, &p.name, &t.sent, &p.args)?;
                self.told = Some(Told {
                    sent: p.args.clone(),
                    ..t
                });
            }
            (Some(p), None) => {
                self.pieces.push(CallPiece::Open {
                    index: p.index,
                    name: p.name.clone(),
                });
                self.extend(p.index, &p.name, "", &p.args)?;
                self.told = Some(Told {
                    index: p.index,
                    name: p.name.clone(),
                    sent: p.args.clone(),
                });
            }
            (_, Some(t)) => return Err(diverged(&t.name, t.index)),
            (None, None) => {}
        }
        Ok(())
    }

    /// The fragment of `now` past `sent`, which `now` must begin with.
    fn extend(
        &mut self,
        index: usize,
        name: &str,
        sent: &str,
        now: &str,
    ) -> Result<(), MarkupError> {
        let Some(rest) = now.strip_prefix(sent) else {
            return Err(diverged(name, index));
        };
        if !rest.is_empty() {
            self.pieces.push(CallPiece::Args {
                index,
                text: rest.to_owned(),
            });
        }
        Ok(())
    }

    /// The pieces since the last call.
    pub(crate) fn take(&mut self) -> Vec<CallPiece> {
        std::mem::take(&mut self.pieces)
    }
}

fn diverged(name: &str, index: usize) -> MarkupError {
    MarkupError::Diverged {
        index,
        name: name.to_owned(),
    }
}

/// `s` as a JSON string literal, quotes included.
pub(crate) fn string(s: &str) -> String {
    Value::String(s.to_owned()).to_string()
}

/// `s` as the inside of a JSON string literal: its escapes, no quotes.
pub(crate) fn string_body(s: &str) -> String {
    let q = string(s);
    q[1..q.len() - 1].to_owned()
}

/// The text of a value that a tag ends, as far as it may be released: `v`
/// without the tail that could still grow into any of `tags`.
pub(crate) fn released<'a>(v: &'a str, tags: &[&str]) -> &'a str {
    let keep = tags.iter().map(|t| partial_suffix(v, t)).max().unwrap_or(0);
    &v[..v.len() - keep]
}

/// A call's arguments object as a scan has read it: `{`, then a field per
/// complete parameter, then optionally the text of one value in progress.
#[derive(Debug)]
pub(crate) struct Args(String);

impl Args {
    pub(crate) fn new() -> Self {
        Args("{".to_owned())
    }

    fn key(&mut self, key: &str) {
        if self.0.len() > 1 {
            self.0.push(',');
        }
        self.0.push_str(&string(key));
        self.0.push(':');
    }

    /// A parameter whose value is `json`, complete JSON text.
    pub(crate) fn field(&mut self, key: &str, json: &str) {
        self.key(key);
        self.0.push_str(json);
    }

    /// The text parameter `key` with `released` of its value so far; the
    /// string stays open.
    pub(crate) fn text_so_far(mut self, key: &str, released: &str) -> String {
        self.key(key);
        self.0.push('"');
        self.0.push_str(&string_body(released));
        self.0
    }

    /// The fields so far; the object stays open.
    pub(crate) fn so_far(self) -> String {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(index: usize, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            index,
            name: name.to_owned(),
            arguments: arguments.to_owned(),
        }
    }

    fn progress(index: usize, args: &str) -> Progress {
        Progress {
            index,
            name: "f".to_owned(),
            args: args.to_owned(),
        }
    }

    fn open(index: usize) -> CallPiece {
        CallPiece::Open {
            index,
            name: "f".to_owned(),
        }
    }

    fn args(index: usize, text: &str) -> CallPiece {
        CallPiece::Args {
            index,
            text: text.to_owned(),
        }
    }

    /// A call is announced once, extended by the part each report adds, and
    /// closed by the part the closing text adds; a call that opens and closes
    /// in one step goes out whole.
    #[test]
    fn a_call_is_announced_once_and_extended_in_order() {
        let mut s = Streamer::default();
        s.step(&[], Some(&progress(0, "{"))).unwrap();
        s.step(&[], Some(&progress(0, r#"{"a":"x"#))).unwrap();
        s.step(&[], Some(&progress(0, r#"{"a":"xy"#))).unwrap();
        s.step(&[call(0, "f", r#"{"a":"xyz"}"#)], None).unwrap();
        assert_eq!(
            s.take(),
            [
                open(0),
                args(0, "{"),
                args(0, r#""a":"x"#),
                args(0, "y"),
                args(0, r#"z"}"#),
            ]
        );
        s.step(&[call(1, "f", "{}")], None).unwrap();
        assert_eq!(s.take(), [open(1), args(1, "{}")]);
    }

    /// A call that closes in the step after another opened goes out after it,
    /// in index order.
    #[test]
    fn a_step_that_closes_one_call_and_opens_the_next() {
        let mut s = Streamer::default();
        s.step(&[call(0, "f", "{}")], Some(&progress(1, "{")))
            .unwrap();
        assert_eq!(s.take(), [open(0), args(0, "{}"), open(1), args(1, "{")]);
    }

    /// Fragments are never revised: arguments the sent text is not a prefix
    /// of, an announced call the markup drops, and a call that closes or is
    /// reported open under another index or name are named errors.
    #[test]
    fn what_was_sent_is_never_revised() {
        let mut s = Streamer::default();
        s.step(&[], Some(&progress(0, r#"{"a":1,"a":2"#))).unwrap();
        let e = s.step(&[call(0, "f", r#"{"a":2}"#)], None).unwrap_err();
        assert_eq!(
            e,
            MarkupError::Diverged {
                index: 0,
                name: "f".to_owned()
            }
        );

        let mut s = Streamer::default();
        s.step(&[], Some(&progress(0, "{"))).unwrap();
        assert!(matches!(
            s.step(&[], None),
            Err(MarkupError::Diverged { index: 0, .. })
        ));

        let mut s = Streamer::default();
        s.step(&[], Some(&progress(0, "{"))).unwrap();
        assert!(matches!(
            s.step(&[call(1, "f", "{}")], None),
            Err(MarkupError::Diverged { index: 0, .. })
        ));

        let mut s = Streamer::default();
        s.step(&[], Some(&progress(0, "{"))).unwrap();
        assert_eq!(
            s.step(&[call(0, "g", "{}")], None).unwrap_err(),
            diverged("f", 0)
        );

        let mut s = Streamer::default();
        s.step(&[], Some(&progress(0, "{"))).unwrap();
        let renamed = Progress {
            name: "g".to_owned(),
            ..progress(0, "{")
        };
        assert_eq!(s.step(&[], Some(&renamed)).unwrap_err(), diverged("f", 0));
    }

    /// The escaped form is the one a closed call writes, and the held tail is
    /// the longest that could grow into a tag.
    #[test]
    fn text_is_escaped_as_a_closed_call_writes_it_and_a_tag_tail_is_held() {
        assert_eq!(string_body("a\"b\\c\nd\u{1}é"), r#"a\"b\\c\nd\u0001é"#);
        assert_eq!(released("abc</par", &["</parameter>"]), "abc");
        assert_eq!(released("abc</", &["</parameter>", "</call>"]), "abc");
        assert_eq!(released("abc<", &["</parameter>"]), "abc");
        assert_eq!(released("abc", &["</parameter>"]), "abc");
        let mut a = Args::new();
        a.field("n", "3");
        assert_eq!(a.text_so_far("k", "v\n"), r#"{"n":3,"k":"v\n"#);
    }
}
