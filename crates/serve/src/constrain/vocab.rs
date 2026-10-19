//! What the vocabulary says about each token, read once when the server binds.
//!
//! A free string may take any token whose text its quoting lets through, so
//! which tokens those are is a property of the vocabulary alone: one decode of
//! every id at bind, not a scan a request or a step repeats. The class of a
//! JSON string (no `"`, no `\`, no control character; and, where a tag could
//! open or end a value, no `<`) is built here; the class of a raw value, which
//! ends at a fixed text, depends on that text's first byte and is built on
//! first use for each byte (a handful of bytes ever occur).
//! A token's text is its single-id decode: a byte-level token that is half a
//! character decodes to the replacement character, which holds none of the
//! bytes a class forbids, so the class reads the same either way.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::engine::Tokenizer;

/// A set of ids.
#[derive(Debug)]
pub(crate) struct Bits(Vec<u64>);

impl Bits {
    fn new(n: usize) -> Bits {
        Bits(vec![0; n.div_ceil(64)])
    }

    fn set(&mut self, id: usize) {
        self.0[id / 64] |= 1 << (id % 64);
    }

    /// Whether `id` is in the set; an id past the vocabulary is not.
    pub(crate) fn has(&self, id: u32) -> bool {
        let id = id as usize;
        self.0.get(id / 64).is_some_and(|w| w >> (id % 64) & 1 == 1)
    }
}

/// The tokens a free string may take, with each one's length in characters.
#[derive(Clone, Debug)]
pub(crate) struct Class {
    ok: Arc<Bits>,
    chars: Arc<[u8]>,
}

/// Characters a count holds exactly; a longer token counts as this many and
/// fits no length limit.
const CHARS_CAP: u8 = u8::MAX;

impl Class {
    /// Whether `id` is in the class and fits `room` more characters
    /// (`usize::MAX` for a string with no limit).
    pub(crate) fn fits(&self, id: u32, room: usize) -> bool {
        if !self.ok.has(id) {
            return false;
        }
        let n = self.chars[id as usize];
        room == usize::MAX || (n < CHARS_CAP && usize::from(n) <= room)
    }

    /// The characters `id` adds to a string.
    pub(crate) fn len_of(&self, id: u32) -> usize {
        self.chars.get(id as usize).map_or(0, |&n| usize::from(n))
    }

    /// The vocabulary size the class was read over.
    pub(crate) fn n_vocab(&self) -> usize {
        self.chars.len()
    }
}

/// The vocabulary read once: every id's text, its length in characters, the
/// stop ids, and the two JSON-string classes.
pub(crate) struct Vocab {
    /// Every id's text, concatenated; id `i` is `text[at[i]..at[i + 1]]`.
    text: String,
    at: Vec<u32>,
    chars: Arc<[u8]>,
    /// The ids with a non-empty text, by the first byte of it.
    first: Vec<Vec<u32>>,
    stops: Arc<[u32]>,
    json: Class,
    /// `json` without the tokens that hold a `<`.
    json_in_markup: Class,
    /// The raw classes built so far, by the byte their value may not hold.
    raw: Mutex<HashMap<u8, Class>>,
}

impl Vocab {
    /// Reads `tok`: one single-id decode for each of its ids.
    pub(crate) fn read(tok: &dyn Tokenizer) -> Vocab {
        let n = tok.n_vocab();
        let stops: Arc<[u32]> = tok.stops().into();
        let mut text = String::new();
        let mut at = Vec::with_capacity(n + 1);
        let mut chars = Vec::with_capacity(n);
        let mut first: Vec<Vec<u32>> = vec![Vec::new(); 256];
        let mut json = Bits::new(n);
        let mut json_in_markup = Bits::new(n);
        for id in 0..n {
            let id32 = u32::try_from(id).expect("a vocabulary fits u32");
            at.push(u32::try_from(text.len()).expect("the vocabulary's text fits u32"));
            let piece = tok.decode(&[id32]);
            let n_chars = piece.chars().count();
            chars.push(u8::try_from(n_chars).unwrap_or(CHARS_CAP));
            if let Some(&b) = piece.as_bytes().first() {
                first[usize::from(b)].push(id32);
            }
            let stop = stops.contains(&id32);
            let safe = !piece.is_empty()
                && !stop
                && !piece.bytes().any(|b| matches!(b, b'"' | b'\\') || b < 0x20);
            if safe {
                json.set(id);
                if !piece.contains('<') {
                    json_in_markup.set(id);
                }
            }
            text.push_str(&piece);
        }
        at.push(u32::try_from(text.len()).expect("the vocabulary's text fits u32"));
        let chars: Arc<[u8]> = chars.into();
        Vocab {
            json: Class {
                ok: Arc::new(json),
                chars: Arc::clone(&chars),
            },
            json_in_markup: Class {
                ok: Arc::new(json_in_markup),
                chars: Arc::clone(&chars),
            },
            text,
            at,
            chars,
            first,
            stops,
            raw: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn n_vocab(&self) -> usize {
        self.chars.len()
    }

    pub(crate) fn stops(&self) -> &Arc<[u32]> {
        &self.stops
    }

    /// The tokens a JSON string's body may take.
    pub(crate) fn json_class(&self) -> Class {
        self.json.clone()
    }

    /// The tokens a JSON string's body may take inside a chat format's
    /// markup: [`Vocab::json_class`] without a token that holds a `<`.
    pub(crate) fn json_class_in_markup(&self) -> Class {
        self.json_in_markup.clone()
    }

    /// The tokens whose text holds no `stop` byte: the body of a raw value
    /// that ends at a text beginning with it.
    pub(crate) fn raw_class(&self, stop: u8) -> Class {
        let mut built = self.raw.lock().unwrap_or_else(|e| e.into_inner());
        built
            .entry(stop)
            .or_insert_with(|| {
                let mut ok = Bits::new(self.n_vocab());
                for id in 0..self.n_vocab() {
                    let piece = self.piece(id);
                    let id32 = u32::try_from(id).expect("a vocabulary fits u32");
                    if !piece.is_empty()
                        && !self.stops.contains(&id32)
                        && !piece.bytes().any(|b| b == stop)
                    {
                        ok.set(id);
                    }
                }
                Class {
                    ok: Arc::new(ok),
                    chars: Arc::clone(&self.chars),
                }
            })
            .clone()
    }

    fn piece(&self, id: usize) -> &str {
        &self.text[self.at[id] as usize..self.at[id + 1] as usize]
    }

    /// The ids whose text is a non-empty prefix of `tail`, in id order, with
    /// the length of each text in bytes. A stop id is none of them.
    pub(crate) fn prefixes_of(&self, tail: &str) -> Vec<(u32, usize)> {
        let Some(&b) = tail.as_bytes().first() else {
            return Vec::new();
        };
        self.first[usize::from(b)]
            .iter()
            .filter(|&&id| !self.stops.contains(&id))
            .filter_map(|&id| {
                let piece = self.piece(id as usize);
                tail.starts_with(piece).then_some((id, piece.len()))
            })
            .collect()
    }
}
