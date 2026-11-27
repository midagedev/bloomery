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
use std::cell::{Cell, OnceCell};

use jinja::ChatTemplate;
use serde_json::{Map, Value};

use crate::engine::Tokenizer;

/// The symbols a mark is made of: ASCII, so `tojson` and `repr` write them as they are, caseless, and
/// none a special token's spelling is made of.
pub(crate) const SYMBOLS: &[u8] = b"~^@#$%*+=;?!";

/// Why a chat prompt could not be guarded.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum GuardError {
    /// The template cut a marked string apart, or read it another way than it reads the text
    /// without the mark: the text it renders is not the text the client sent.
    #[error(
        "the chat template cut a text that carries a special-token spelling apart (the text it \
         renders is not the text the client sent, and the guard's mark {mark:?} does not survive \
         it), so it cannot keep that text as text; send the text without the spelling"
    )]
    Severed { mark: String },
}

/// Symbol pairs held by the request's strings.
struct Pairs {
    seen: Vec<bool>,
}

impl Pairs {
    fn new() -> Pairs {
        Pairs {
            seen: vec![false; 256 * 256],
        }
    }

    fn add(&mut self, text: &str) {
        for w in text.as_bytes().windows(2) {
            self.seen[usize::from(w[0]) << 8 | usize::from(w[1])] = true;
        }
    }

    fn holds(&self, a: u8, b: u8) -> bool {
        self.seen[usize::from(a) << 8 | usize::from(b)]
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

fn add_value(v: &Value, add: &mut dyn FnMut(&str)) {
    match v {
        // Templates can decode JSON strings, so their decoded strings count as well.
        Value::String(s) => {
            add(s);
            if let Ok(parsed) = serde_json::from_str::<Value>(s) {
                add_value(&parsed, add);
            }
        }
        Value::Array(a) => a.iter().for_each(|x| add_value(x, add)),
        Value::Object(o) => {
            for (k, x) in o {
                add(k);
                add_value(x, add);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

/// The least frequent extension of `prefix` across the same strings used for the pair search.
fn extension(prefix: &[u8], scan: &impl Fn(&mut dyn FnMut(&str))) -> (u8, usize) {
    let mut next = [0; SYMBOLS.len()];
    scan(&mut |text| {
        for w in text.as_bytes().windows(prefix.len() + 1) {
            if w.starts_with(prefix)
                && let Some(i) = SYMBOLS.iter().position(|&b| b == w[prefix.len()])
            {
                next[i] += 1;
            }
        }
    });
    let i = (0..SYMBOLS.len())
        .min_by_key(|&i| (next[i], i))
        .expect("the marker alphabet is non-empty");
    (SYMBOLS[i], next[i])
}

/// Two absent pieces, preferring pairs. Following the least frequent extension makes the
/// fallback's length logarithmic in the number of input bytes, even beside long symbol runs.
fn choose_mark(body: &Map<String, Value>, strings: &[&str]) -> String {
    let scan = |add: &mut dyn FnMut(&str)| {
        for s in strings {
            add(s);
        }
        for (k, v) in body {
            add(k);
            add_value(v, add);
        }
    };
    let mut seen = Pairs::new();
    scan(&mut |text| seen.add(text));
    if let Some(pairs) = seen.free() {
        return pairs.into_iter().flatten().map(char::from).collect();
    }
    drop(seen);
    let mut prefix = vec![SYMBOLS[0]];
    loop {
        let (next, count) = extension(&prefix, &scan);
        prefix.push(next);
        if count == 0 {
            let piece: String = prefix.into_iter().map(char::from).collect();
            // Distinct endings keep the mark from sliding across a neighbouring symbol run.
            return format!(
                "{piece}{}{piece}{}",
                char::from(SYMBOLS[0]),
                char::from(SYMBOLS[1])
            );
        }
    }
}

/// What a chat render needs to keep client text text: the mark and the vocabulary that guards by it.
pub(crate) struct Guard<'a> {
    tok: &'a dyn Tokenizer,
    mark: OnceCell<String>,
    body: &'a Map<String, Value>,
    strings: Vec<&'a str>,
    /// A string that carries one of these is left as it came ([`Guard::except`]).
    except: Vec<String>,
    /// How many strings came back with a mark in them.
    marked: Cell<usize>,
}

impl<'a> Guard<'a> {
    /// A guard whose mark no string of `body` (the request), `strings` (the template's source and
    /// the server's own texts) holds. The mark is chosen only when client text needs guarding;
    /// two absent symbol pairs are preferred, with a longer absent piece when the pairs run out.
    pub(crate) fn new(
        tok: &'a dyn Tokenizer,
        body: &'a Map<String, Value>,
        strings: &[&'a str],
    ) -> Guard<'a> {
        Guard {
            tok,
            mark: OnceCell::new(),
            body,
            strings: strings.to_vec(),
            except: Vec::new(),
            marked: Cell::new(0),
        }
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
        let mark = match self.mark.get() {
            Some(mark) => mark,
            None => {
                if matches!(self.tok.guard(text, "~"), Cow::Borrowed(_)) {
                    return Cow::Borrowed(text);
                }
                self.mark
                    .get_or_init(|| choose_mark(self.body, &self.strings))
            }
        };
        let guarded = self.tok.guard(text, mark);
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
        let mark = self.mark.get().expect("a marked string has a guard mark");
        let text = marked.replace(mark, "");
        // What the template renders of the client's text without the mark: a template that cuts
        // a marked text apart, or that reads it another way, renders other text than the marked
        // render does without its marks.
        let mut original = vars.clone();
        let mut pending: Vec<&mut Value> = original.values_mut().collect();
        while let Some(value) = pending.pop() {
            match value {
                Value::String(s) => *s = s.replace(mark, ""),
                Value::Array(a) => pending.extend(a.iter_mut()),
                Value::Object(o) => {
                    *o = std::mem::take(o)
                        .into_iter()
                        .map(|(k, v)| (k.replace(mark, ""), v))
                        .collect();
                    pending.extend(o.values_mut());
                }
                Value::Null | Value::Bool(_) | Value::Number(_) => {}
            }
        }
        let expected = template
            .render(&original)
            .map_err(|_| GuardError::Severed { mark: mark.clone() })?;
        if text != expected {
            return Err(GuardError::Severed { mark: mark.clone() });
        }
        Ok((
            text,
            Some(Guarded {
                text: marked,
                mark: mark.clone(),
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
    /// Appends text the server writes after the template's render (a generation prompt): it holds
    /// no mark and tokenizes as the template's own text does.
    pub(crate) fn push_template_text(&mut self, text: &str) {
        self.text.push_str(text);
    }

    /// The ids of the prompt: the text the client sent, with no special token across a mark.
    pub(crate) fn encode(&self, tok: &dyn Tokenizer) -> Vec<u32> {
        tok.encode_marked(&self.text, &self.mark)
    }
}
