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
//! The scans of DSML, the key/value markup and Qwen3-Coder's markup read only
//! the text each push adds: each holds its open call as a `Reading` and reports
//! the arguments a push adds as an `Advance`, so a long value streams with
//! work linear in its length.
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

/// One step of a call a scan reads by its own cursor, reporting only the text
/// each push adds ([`crate::dsml::Scanned`]'s `advances`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Advance {
    /// The open call's arguments read since its last step, and whether they
    /// close it.
    Args(Progress, bool),
    /// The open call, some of it sent, ended other than as sent: given back
    /// as content, or closed with arguments the sent text does not begin.
    Diverged,
}

/// A call a scan reads by its own cursor, as read so far: its name once read,
/// the arguments read, and how much of them went out as [`Advance`]s.
#[derive(Clone, Debug, Default)]
pub(crate) struct Reading {
    pub(crate) name: Option<String>,
    pub(crate) args: String,
    sent: usize,
}

impl Reading {
    /// The call `index` as read so far, once its name is read.
    pub(crate) fn progress(&self, index: usize) -> Option<Progress> {
        self.name.as_ref().map(|name| Progress {
            index,
            name: name.clone(),
            args: self.args.clone(),
        })
    }

    /// Sends the arguments read since the last step of the open call `index`.
    pub(crate) fn send(&mut self, index: usize, out: &mut Vec<Advance>) {
        if let Some(name) = &self.name
            && self.sent < self.args.len()
        {
            out.push(Advance::Args(
                Progress {
                    index,
                    name: name.clone(),
                    args: self.args[self.sent..].to_owned(),
                },
                false,
            ));
            self.sent = self.args.len();
        }
    }

    /// Ends the reading with the close of `call`: the arguments not yet sent,
    /// or [`Advance::Diverged`] when `call`'s do not begin with those sent.
    pub(crate) fn close(self, call: &ToolCall, out: &mut Vec<Advance>) {
        out.push(match call.arguments.strip_prefix(&self.args[..self.sent]) {
            Some(rest) => Advance::Args(
                Progress {
                    index: call.index,
                    name: call.name.clone(),
                    args: rest.to_owned(),
                },
                true,
            ),
            None => Advance::Diverged,
        });
    }

    /// Ends the reading of a call given back as content:
    /// [`Advance::Diverged`] once some of it was sent.
    pub(crate) fn give_back(self, out: &mut Vec<Advance>) {
        if self.sent > 0 {
            out.push(Advance::Diverged);
        }
    }
}

/// A call announced and not closed: what has gone out of it.
#[derive(Clone, Debug)]
struct Told {
    index: usize,
    name: String,
    /// The whole-state scan's sent prefix; append-only advances leave it unchanged.
    sent: String,
}

/// The pieces a parser's scans owe the stream: the call being read, and what
/// went out of it.
#[derive(Clone, Debug, Default)]
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

    /// A step of a scan that reads by its own cursor.
    pub(crate) fn advance(&mut self, a: &Advance) -> Result<(), MarkupError> {
        let (p, closed) = match a {
            Advance::Args(p, closed) => (p, *closed),
            Advance::Diverged => {
                return match self.told.take() {
                    Some(t) => Err(diverged(&t.name, t.index)),
                    None => Ok(()),
                };
            }
        };
        if let Some(t) = &self.told {
            if t.index != p.index || t.name != p.name {
                return Err(diverged(&t.name, t.index));
            }
            self.extend(p.index, &p.name, "", &p.args)?;
        } else {
            self.step(&[], Some(p))?;
        }
        if closed {
            self.told = None;
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
fn released<'a>(v: &'a str, tags: &[&str]) -> &'a str {
    let keep = tags.iter().map(|t| partial_suffix(v, t)).max().unwrap_or(0);
    &v[..v.len() - keep]
}

/// Whether `rest` is a proper prefix of `tag` (the empty text included): the
/// text so far can still grow into the tag, so nothing is decided yet.
pub(crate) fn held(tag: &str, rest: &str) -> bool {
    rest.len() < tag.len() && tag.starts_with(rest)
}

/// Where `tag` starts in `s` at or after `*searched`. While it has not come,
/// `*searched` moves past the text that can no longer begin it, so the next
/// search reads only what arrives.
pub(crate) fn find_from(s: &str, searched: &mut usize, tag: &str) -> Option<usize> {
    let from = *searched;
    let found = s[from..].find(tag).map(|n| from + n);
    if found.is_none() {
        *searched = s.len() - partial_suffix(&s[from..], tag);
    }
    found
}

/// Where `close` starts in `s` for a value read from `*searched` on, once it
/// has come. A text value (`text`, the arguments it is written into) gains
/// the value's bytes up to `close`, or up to the tail that could still grow
/// into `close` or `ends` (the tag that ends the call), JSON-escaped;
/// `*searched` moves past them.
pub(crate) fn read_value(
    s: &str,
    searched: &mut usize,
    close: &str,
    ends: &str,
    text: Option<&mut String>,
) -> Option<usize> {
    let Some(args) = text else {
        return find_from(s, searched, close);
    };
    let from = *searched;
    let tail = &s[from..];
    let end = tail.find(close).map(|n| from + n);
    *searched = end.unwrap_or_else(|| from + released(tail, &[close, ends]).len());
    args.push_str(&string_body(&s[from..*searched]));
    end
}

/// A call's arguments object as a scan has read it: `{`, then a field per
/// complete parameter, then optionally the text of one value in progress.
#[derive(Debug)]
pub(crate) struct Args(String);

impl Args {
    pub(crate) fn new() -> Self {
        Args("{".to_owned())
    }

    /// Appends the field prefix for `key` to the open object `out`: a comma
    /// after an earlier field, the key, the colon.
    pub(crate) fn key(out: &mut String, key: &str) {
        if out.len() > 1 {
            out.push(',');
        }
        out.push_str(&string(key));
        out.push(':');
    }

    /// A parameter whose value is `json`, complete JSON text.
    pub(crate) fn field(&mut self, key: &str, json: &str) {
        Self::key(&mut self.0, key);
        self.0.push_str(json);
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
        let (mut searched, mut text) = (0, String::new());
        let s = "v\n\"</par";
        let close = read_value(s, &mut searched, "</parameter>", "</call>", Some(&mut text));
        assert_eq!((close, searched, text.as_str()), (None, 3, r#"v\n\""#));
        let s = "v\n\"</parameter>";
        let close = read_value(s, &mut searched, "</parameter>", "</call>", Some(&mut text));
        assert_eq!((close, text.as_str()), (Some(3), r#"v\n\""#));
        let mut searched = 0;
        assert_eq!(find_from("1, 2</p", &mut searched, "</p>"), None);
        assert_eq!(
            searched, 4,
            "only the tail that can begin the tag is searched again"
        );
        assert_eq!(find_from("1, 2</p>", &mut searched, "</p>"), Some(4));
    }
    #[test]
    fn advances_do_not_accumulate_argument_history() {
        let step = |p, closed| Advance::Args(p, closed);
        let mut s = Streamer::default();
        s.advance(&step(progress(0, "{"), false)).unwrap();
        assert_eq!(s.take(), [open(0), args(0, "{")]);
        for fragment in ["", "\"text\":", "\"hello\""] {
            s.advance(&step(progress(0, fragment), false)).unwrap();
            assert_eq!(
                s.told.as_ref().unwrap().sent,
                "{",
                "advances retain only the opening prefix"
            );
            assert_eq!(
                s.take(),
                if fragment.is_empty() {
                    vec![]
                } else {
                    vec![args(0, fragment)]
                }
            );
        }
        let renamed = Progress {
            name: "g".to_owned(),
            ..progress(0, "")
        };
        assert_eq!(
            s.advance(&step(renamed, false)).unwrap_err(),
            diverged("f", 0)
        );
        assert_eq!(
            s.advance(&step(progress(1, ""), false)).unwrap_err(),
            diverged("f", 0)
        );
        s.advance(&step(progress(0, "}"), true)).unwrap();
        assert!(s.told.is_none(), "closed calls release their state");
        assert_eq!(s.take(), [args(0, "}")]);
        s.advance(&step(progress(1, "{}"), true)).unwrap();
        assert_eq!(s.take(), [open(1), args(1, "{}")]);
        s.advance(&step(progress(2, "{"), false)).unwrap();
        assert_eq!(s.advance(&Advance::Diverged).unwrap_err(), diverged("f", 2));
        assert_eq!(s.advance(&Advance::Diverged), Ok(()), "nothing was sent");
    }

    /// A reading sends only what it read since its last step, and closes
    /// with the rest of the call it closes with, or diverges from what it sent.
    #[test]
    fn a_reading_sends_its_new_arguments_and_closes_on_the_call() {
        let mut out = Vec::new();
        let mut r = Reading::default();
        r.send(0, &mut out);
        assert!(out.is_empty(), "nothing goes out before the name");
        r.name = Some("f".to_owned());
        r.args.push_str(r#"{"a":"x"#);
        r.send(0, &mut out);
        r.send(0, &mut out);
        r.args.push('y');
        r.close(&call(0, "f", r#"{"a":"xyz"}"#), &mut out);
        assert_eq!(
            out,
            [
                Advance::Args(progress(0, r#"{"a":"x"#), false),
                Advance::Args(progress(0, r#"yz"}"#), true),
            ],
            "read and unsent text closes with the call's own"
        );
        let mut out = Vec::new();
        let mut r = Reading {
            name: Some("f".to_owned()),
            args: r#"{"a":1"#.to_owned(),
            ..Reading::default()
        };
        r.send(0, &mut out);
        r.clone().close(&call(0, "f", r#"{"a":2}"#), &mut out);
        r.give_back(&mut out);
        Reading::default().give_back(&mut out);
        assert_eq!(
            out,
            [
                Advance::Args(progress(0, r#"{"a":1"#), false),
                Advance::Diverged,
                Advance::Diverged,
            ]
        );
    }
}
