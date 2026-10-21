//! Client text in a chat prompt stays text.
//!
//! The chat path renders the whole prompt through the model's template and tokenizes the string with
//! the vocabulary's special tokens parsed ([`Tokenizer::encode`]), because the template writes its own
//! turn markers, think tags and BOS as text. The text a client sends goes through the same string, so
//! a special token's spelling in a message, a tool result, a tool call or a tool description would
//! become the token: a turn boundary, a think close, a DSML tag the client wrote.
//!
//! A [`Guard`] closes that before the render and after it. Before: each client string has a short
//! mark spliced in beside every special-token spelling it carries, and beside the beginning or the
//! end of one that a template could finish with what it writes next to the string
//! ([`Tokenizer::guard`]); the spelling itself is left whole, so a template that trims, splits,
//! compares or prints a string, or asks whether it starts or ends with a spelling, still works on
//! it. After: [`Guard::finish`] cuts the marks out of the rendered string and the tokenizer reads
//! the marked string ([`Tokenizer::encode_marked`]), which matches no special token that holds a
//! mark. The tokens the template wrote are in no client string, so they match as they always did,
//! and the text between them is one text to the BPE.
//!
//! What this does not find is a spelling the template builds from its own pieces around a client
//! string, none of which is a spelling's beginning or end: a template that writes
//! `'<' ~ key ~ '>'` makes `<|im_end|>` of the key `|im_end|`. The string would have to be marked
//! whole, and a template reads whole strings (a lookup, a comparison) in ways a mark would change.
//!
//! What is a client string, and what is the model's own: every string of a message, a tool and a
//! `chat_template_kwargs` value is the client's, except an assistant message's `content` and
//! reasoning. Those are the model's past output, which the client sends back as it came: the special
//! ids it printed decode to their spellings, and the spellings tokenize back to those ids, so the
//! prompt holds what the model saw. An assistant message's tool calls are not: their names and
//! arguments are text the model copied from a tool result or a page.
//!
//! The mark is a few ASCII symbols the request and the template do not hold, so no string of the
//! render can be mistaken for it, and it passes through the filters a template applies to strings
//! (`tojson`, `lower`, `trim`, `string`) unchanged. A template that cuts a marked string apart (a
//! slice through a mark) or that reads it another way renders other text than the marked render
//! does without its marks, which [`Guard::finish`] finds and refuses by name: a prompt that does
//! not carry the text the client sent is never built.

use std::borrow::Cow;
use std::cell::Cell;

use jinja::ChatTemplate;
use serde_json::{Map, Value};

use crate::engine::Tokenizer;

/// The symbols a mark is made of: ASCII, so `tojson` and `repr` write them as they are, caseless, and
/// none a special token's spelling is made of.
const SYMBOLS: &[u8] = b"~^@#$%*+=;?!";

/// Why a chat prompt could not be guarded.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum GuardError {
    /// The request and the template together hold every symbol pair a mark could be made of.
    #[error(
        "the request carries every pair of the symbols {symbols:?} the guard of control strings \
         marks text with, so it cannot be guarded"
    )]
    NoMark { symbols: &'static str },
    /// The template cut a marked string apart, or read it another way than it reads the text
    /// without the mark: the text it renders is not the text the client sent.
    #[error(
        "the chat template cut a text that carries a special-token spelling apart (the text it \
         renders is not the text the client sent, and the guard's mark {mark:?} does not survive \
         it), so it cannot keep that text as text; send the text without the spelling"
    )]
    Severed { mark: String },
}

/// The symbol pairs the strings of a request and its template hold, as a bitmap.
struct Pairs(Vec<bool>);

impl Pairs {
    fn new() -> Pairs {
        Pairs(vec![false; 256 * 256])
    }

    fn add(&mut self, text: &str) {
        for w in text.as_bytes().windows(2) {
            self.0[usize::from(w[0]) << 8 | usize::from(w[1])] = true;
        }
    }

    fn add_value(&mut self, v: &Value) {
        match v {
            // A string that holds JSON is read back by the template as the object it writes, so
            // the mark must not be in that text either: the pairs of the decoded text are held
            // as well as the pairs of the string that carries it.
            Value::String(s) => {
                self.add(s);
                if let Ok(parsed) = serde_json::from_str::<Value>(s) {
                    self.add_value(&parsed);
                }
            }
            Value::Array(a) => a.iter().for_each(|x| self.add_value(x)),
            Value::Object(o) => {
                for (k, x) in o {
                    self.add(k);
                    self.add_value(x);
                }
            }
            Value::Null | Value::Bool(_) | Value::Number(_) => {}
        }
    }

    fn holds(&self, a: u8, b: u8) -> bool {
        self.0[usize::from(a) << 8 | usize::from(b)]
    }

    /// The first two pairs of [`SYMBOLS`] none of the strings holds, in the order of the symbols.
    fn free(&self) -> Option<[[u8; 2]; 2]> {
        let mut free = SYMBOLS
            .iter()
            .flat_map(|&a| SYMBOLS.iter().map(move |&b| [a, b]))
            .filter(|&[a, b]| !self.holds(a, b));
        Some([free.next()?, free.next()?])
    }
}

/// What a chat render needs to keep client text text: the mark and the vocabulary that guards by it.
pub(crate) struct Guard<'a> {
    tok: &'a dyn Tokenizer,
    mark: String,
    /// A string that carries one of these is left as it came ([`Guard::except`]).
    except: Vec<String>,
    /// How many strings came back with a mark in them.
    marked: Cell<usize>,
}

impl<'a> Guard<'a> {
    /// A guard whose mark no string of `body` (the request), `strings` (the template's source and
    /// the server's own texts) holds. The mark is two pairs of symbols, each held by none of them: the
    /// whole mark is then in none of them, and neither is a piece of it two symbols long.
    pub(crate) fn new(
        tok: &'a dyn Tokenizer,
        body: &Map<String, Value>,
        strings: &[&str],
    ) -> Result<Guard<'a>, GuardError> {
        let mut seen = Pairs::new();
        for s in strings {
            seen.add(s);
        }
        for (k, v) in body {
            seen.add(k);
            seen.add_value(v);
        }
        let [first, last] = seen.free().ok_or(GuardError::NoMark {
            symbols: std::str::from_utf8(SYMBOLS).expect("ASCII symbols"),
        })?;
        let pair = |p: [u8; 2]| p.iter().copied().map(char::from).collect::<String>();
        let (first, last) = (pair(first), pair(last));
        Ok(Guard {
            tok,
            mark: format!("{first}{last}"),
            except: Vec::new(),
            marked: Cell::new(0),
        })
    }

    /// The same guard leaving alone every string that carries one of `markers`: a model's image
    /// and video placeholders, which the media part refuses in client text by name, whatever the
    /// vocabulary does with it. The request is refused there, so no text of it reaches an
    /// engine.
    #[must_use]
    pub(crate) fn except(mut self, markers: &[&str]) -> Guard<'a> {
        self.except = markers
            .iter()
            .filter(|m| !m.is_empty())
            .map(|m| (*m).to_owned())
            .collect();
        self
    }

    /// `text` with the mark in each special-token spelling.
    pub(crate) fn text<'t>(&self, text: &'t str) -> Cow<'t, str> {
        if self.except.iter().any(|m| text.contains(m.as_str())) {
            return Cow::Borrowed(text);
        }
        let guarded = self.tok.guard(text, &self.mark);
        if matches!(guarded, Cow::Owned(_)) {
            self.marked.set(self.marked.get() + 1);
        }
        guarded
    }

    /// [`Guard::text`] of a client's text, and `text` as it is when `client` is not: the model's own.
    pub(crate) fn text_if<'t>(&self, client: bool, text: &'t str) -> Cow<'t, str> {
        if client {
            self.text(text)
        } else {
            Cow::Borrowed(text)
        }
    }

    /// `text` guarded, when it carries a spelling.
    fn changed(&self, text: &str) -> Option<String> {
        match self.text(text) {
            Cow::Owned(g) => Some(g),
            Cow::Borrowed(_) => None,
        }
    }

    /// Every string of `v`, and every key of an object in it, guarded in place.
    pub(crate) fn value(&self, v: &mut Value) {
        match v {
            Value::String(s) => {
                if let Some(g) = self.changed(s) {
                    *s = g;
                }
            }
            Value::Array(a) => a.iter_mut().for_each(|x| self.value(x)),
            Value::Object(o) => {
                if o.keys().any(|k| self.changed(k).is_some()) {
                    for (k, mut x) in std::mem::take(o) {
                        self.value(&mut x);
                        o.insert(self.changed(&k).unwrap_or(k), x);
                    }
                } else {
                    o.values_mut().for_each(|x| self.value(x));
                }
            }
            Value::Null | Value::Bool(_) | Value::Number(_) => {}
        }
    }

    /// A tool call's `arguments` string: the JSON it holds guarded value by value, so a spelling
    /// written with JSON escapes is found as the text it reads back to, and written again only when
    /// a value changed, so the text of every other call is the client's own. A string that is no
    /// JSON is guarded as text.
    fn arguments(&self, text: &mut String) {
        let Ok(mut parsed) = serde_json::from_str::<Value>(text) else {
            if let Some(g) = self.changed(text) {
                *text = g;
            }
            return;
        };
        let before = self.marked.get();
        self.value(&mut parsed);
        if self.marked.get() != before {
            *text = parsed.to_string();
        }
    }

    fn tool_calls(&self, calls: &mut Value) {
        let Value::Array(calls) = calls else {
            return self.value(calls);
        };
        for call in calls {
            let Value::Object(call) = call else {
                self.value(call);
                continue;
            };
            for (k, x) in call.iter_mut() {
                if k == "function"
                    && let Value::Object(function) = &mut *x
                {
                    for (fk, fx) in function.iter_mut() {
                        match fx {
                            Value::String(s) if fk == "arguments" => self.arguments(s),
                            fx => self.value(fx),
                        }
                    }
                } else {
                    self.value(x);
                }
            }
        }
    }

    /// The strings of message `m`, role `role`, other than its `content` (the caller's: an array of
    /// parts is flattened first), guarded in place; an assistant message's reasoning is the model's
    /// own and stays as it came.
    pub(crate) fn message(&self, role: &str, m: &mut Map<String, Value>) {
        for (k, v) in m.iter_mut() {
            match k.as_str() {
                "content" => {}
                "reasoning_content" | "reasoning" if role == "assistant" => {}
                "tool_calls" => self.tool_calls(v),
                _ => self.value(v),
            }
        }
    }

    /// Whether any string came back with a mark in it.
    pub(crate) fn any(&self) -> bool {
        self.marked.get() > 0
    }

    /// The rendered prompt `marked` as the text the client sent and the marked string to tokenize,
    /// `None` when no string was marked. A template that cut a marked text apart, or that read it
    /// another way than it reads the text without the mark, is [`GuardError::Severed`].
    pub(crate) fn finish(
        &self,
        marked: String,
        template: &ChatTemplate,
        vars: &Map<String, Value>,
    ) -> Result<(String, Option<Guarded>), GuardError> {
        if !self.any() {
            return Ok((marked, None));
        }
        let text = marked.replace(&self.mark, "");
        // What the template renders of the client's text without the mark: a template that cuts
        // a marked text apart, or that reads it another way, renders other text than the marked
        // render does without its marks.
        let mut original = vars.clone();
        let mut pending: Vec<&mut Value> = original.values_mut().collect();
        while let Some(value) = pending.pop() {
            match value {
                Value::String(s) => *s = s.replace(&self.mark, ""),
                Value::Array(a) => pending.extend(a.iter_mut()),
                Value::Object(o) => {
                    *o = std::mem::take(o)
                        .into_iter()
                        .map(|(k, v)| (k.replace(&self.mark, ""), v))
                        .collect();
                    pending.extend(o.values_mut());
                }
                Value::Null | Value::Bool(_) | Value::Number(_) => {}
            }
        }
        let expected = template
            .render(&original)
            .map_err(|_| GuardError::Severed {
                mark: self.mark.clone(),
            })?;
        if text != expected {
            return Err(GuardError::Severed {
                mark: self.mark.clone(),
            });
        }
        Ok((
            text,
            Some(Guarded {
                text: marked,
                mark: self.mark.clone(),
            }),
        ))
    }
}

/// A rendered prompt that carries marks, as [`Tokenizer::encode_marked`] reads it.
pub(crate) struct Guarded {
    text: String,
    mark: String,
}

impl Guarded {
    /// The ids of the prompt: the text the client sent, with no special token across a mark.
    pub(crate) fn encode(&self, tok: &dyn Tokenizer) -> Vec<u32> {
        tok.encode_marked(&self.text, &self.mark)
    }
}
