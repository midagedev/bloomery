//! Text to token ids and back for the byte-level BPE vocabularies this
//! engine runs, bit-identical to the reference's `llama_tokenize` and
//! `llama_token_to_piece`.
//!
//! The vocabulary comes from the GGUF header (`tokenizer.ggml.*`). Encoding
//! is the reference's pipeline: special tokens are cut out of the text as
//! whole pieces first (longest text first), each remaining fragment is split
//! by the pre-tokenizer ([`pretok`]), and each word is merged by BPE rank
//! ([`bpe`]). Decoding concatenates pieces; this vocabulary family adds no
//! space prefix and cleans no spaces, so decoding strips nothing.

mod bpe;
mod collapse_table;
mod decode;
mod pretok;
mod unicode;
mod vocab;

use std::path::Path;

use memchr::memmem::Finder;

pub use decode::Decoder;
pub use vocab::{Specials, attr};

use vocab::Vocab;

/// Why a vocabulary could not be loaded.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}: {1}")]
    Gguf(String, gguf::LoadError),
    #[error("missing metadata key {0}")]
    Missing(&'static str),
    #[error("metadata key {0} has an unexpected type")]
    WrongType(&'static str),
    #[error("tokenizer model {0:?} is not supported (only gpt2)")]
    UnsupportedModel(String),
    #[error("pre-tokenizer {0:?} is not supported (deepseek-v3, hunyuan-dense, joyai-llm)")]
    UnsupportedPre(String),
    #[error("tokenizer.ggml.token_type has {types} entries for {tokens} tokens")]
    TokenTypesShort { types: usize, tokens: usize },
    #[error("a vocabulary of {0} tokens does not fit u32 ids")]
    TooManyTokens(usize),
    #[error("token text {0:?} appears twice in the vocabulary")]
    DuplicateToken(String),
}

/// A loaded vocabulary with its encoder and decoder tables.
pub struct Tokenizer {
    vocab: Vocab,
    bpe: bpe::Bpe,
    /// Partition order: longest text first.
    specials: Vec<u32>,
    finders: Vec<Finder<'static>>,
    pieces: Vec<Box<[u8]>>,
}

/// One stretch of the input after the special-token partition.
#[derive(Clone, Copy)]
enum Frag {
    Text { start: usize, len: usize },
    Token(u32),
}

impl Tokenizer {
    /// Load the vocabulary from a GGUF file's header; for a split set, the
    /// first shard (the one that carries `tokenizer.ggml.*`).
    pub fn from_gguf(path: impl AsRef<Path>) -> Result<Tokenizer, Error> {
        let vocab = Vocab::load(path.as_ref())?;
        let bpe = bpe::Bpe::new(&vocab.texts, &vocab.merges);
        let specials = vocab.special_order();
        let finders = specials
            .iter()
            .map(|&id| Finder::new(vocab.texts[id as usize].as_bytes()).into_owned())
            .collect();
        let pieces = decode::pieces(&vocab.texts, &vocab.attrs);
        Ok(Tokenizer {
            vocab,
            bpe,
            specials,
            finders,
            pieces,
        })
    }

    /// Token ids of `text`. `add_special` adds BOS/EOS where the vocabulary
    /// asks for them (`tokenizer.ggml.add_bos_token` / `add_eos_token`);
    /// `parse_special` lets CONTROL tokens written in the text become their
    /// ids (USER_DEFINED tokens always do, as in the reference).
    pub fn encode(
        &self,
        text: impl AsRef<[u8]>,
        add_special: bool,
        parse_special: bool,
    ) -> Vec<u32> {
        let text = text.as_ref();
        let mut out = Vec::with_capacity(text.len() / 3 + 2);
        if add_special
            && self.vocab.add_bos
            && let Some(bos) = self.vocab.specials.bos
        {
            out.push(bos);
        }

        let mut cpts = Vec::new();
        let mut word = Vec::new();
        let mut pre = pretok::Scratch::default();
        let mut bpe = bpe::Scratch::default();
        for frag in self.partition(text, parse_special) {
            match frag {
                Frag::Token(id) => out.push(id),
                Frag::Text { start, len } => {
                    unicode::decode_lenient(&text[start..start + len], &mut cpts);
                    let mut at = 0;
                    for &n in pretok::split(&cpts, &mut pre) {
                        word.clear();
                        for &cpt in &cpts[at..at + n] {
                            unicode::encode_cpt(cpt, &mut word);
                        }
                        at += n;
                        self.bpe.word(&word, &mut bpe, &mut out);
                    }
                }
            }
        }

        if add_special
            && self.vocab.add_eos
            && let Some(eos) = self.vocab.specials.eos
        {
            out.push(eos);
        }
        out
    }

    /// `tokenizer_st_partition`: for each special token in order, cut every
    /// occurrence out of every text fragment still unsplit.
    fn partition(&self, text: &[u8], parse_special: bool) -> Vec<Frag> {
        let mut frags = Vec::new();
        if text.is_empty() {
            return frags;
        }
        frags.push(Frag::Text {
            start: 0,
            len: text.len(),
        });
        let mut next = Vec::new();
        for (&id, finder) in self.specials.iter().zip(&self.finders) {
            if !parse_special
                && self.vocab.attrs[id as usize] & (attr::CONTROL | attr::UNKNOWN) != 0
            {
                continue;
            }
            let tlen = finder.needle().len();
            next.clear();
            let mut changed = false;
            for &f in &frags {
                let Frag::Text { start, len } = f else {
                    next.push(f);
                    continue;
                };
                let end = start + len;
                let mut at = start;
                while let Some(m) = finder.find(&text[at..end]) {
                    changed = true;
                    if m > 0 {
                        next.push(Frag::Text { start: at, len: m });
                    }
                    next.push(Frag::Token(id));
                    at += m + tlen;
                }
                if at < end {
                    next.push(Frag::Text {
                        start: at,
                        len: end - at,
                    });
                }
            }
            if changed {
                std::mem::swap(&mut frags, &mut next);
            }
        }
        frags
    }

    /// The bytes token `id` stands for; `special` renders CONTROL and UNKNOWN
    /// tokens as their text instead of nothing. An id outside the vocabulary
    /// is empty.
    pub fn piece(&self, id: u32, special: bool) -> &[u8] {
        let Some(p) = self.pieces.get(id as usize) else {
            return &[];
        };
        if !special && self.vocab.attrs[id as usize] & (attr::CONTROL | attr::UNKNOWN) != 0 {
            return &[];
        }
        p
    }

    /// The concatenated pieces of `ids`, exact bytes.
    pub fn decode_bytes(&self, ids: &[u32], special: bool) -> Vec<u8> {
        ids.iter()
            .flat_map(|&id| self.piece(id, special))
            .copied()
            .collect()
    }

    /// The text of `ids` with special tokens rendered; bytes that do not form
    /// UTF-8 become U+FFFD.
    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids, true)).into_owned()
    }

    /// Vocabulary size.
    pub fn n_vocab(&self) -> usize {
        self.vocab.texts.len()
    }

    /// Token `id`'s text as the vocabulary writes it.
    pub fn text(&self, id: u32) -> Option<&str> {
        self.vocab.texts.get(id as usize).map(String::as_str)
    }

    /// Token `id`'s attribute bits ([`attr`]).
    pub fn attr(&self, id: u32) -> Option<u32> {
        self.vocab.attrs.get(id as usize).copied()
    }

    /// The special ids after the reference's defaults and text-based
    /// detection.
    pub fn specials(&self) -> &Specials {
        &self.vocab.specials
    }

    /// Whether the vocabulary asks for a BOS / an EOS around encoded text.
    pub fn add_bos(&self) -> bool {
        self.vocab.add_bos
    }

    /// See [`Tokenizer::add_bos`].
    pub fn add_eos(&self) -> bool {
        self.vocab.add_eos
    }

    /// End-of-generation ids, ascending.
    pub fn eog(&self) -> &[u32] {
        &self.vocab.eog
    }

    /// `tokenizer.ggml.pre`.
    pub fn pre(&self) -> &str {
        &self.vocab.pre
    }

    /// The special tokens in partition order.
    pub fn special_tokens(&self) -> &[u32] {
        &self.specials
    }

    /// Every metadata key the load read, with its value as shown text.
    pub fn metadata_read(&self) -> &[(String, String)] {
        &self.vocab.keys_read
    }

    /// Merges whose sides or result are not vocabulary tokens.
    pub fn merges_off_vocab(&self) -> usize {
        self.bpe.merges_off_vocab
    }

    /// Roles (eot, fim_pre, ...) filled by text where several candidate
    /// texts exist; the reference's choice among them is hash-map order.
    pub fn ambiguous_roles(&self) -> &[&'static str] {
        &self.vocab.ambiguous_roles
    }

    /// The reference sorts special tokens by length with an unstable sort, so
    /// the order among equal-length tokens is unspecified; it can change a
    /// partition only if two such tokens can overlap in text. Returns the
    /// first pair that can, if any.
    pub fn special_overlap(&self) -> Option<(u32, u32)> {
        let text = |id: u32| self.vocab.texts[id as usize].as_bytes();
        let s = &self.specials;
        let mut i = 0;
        while i < s.len() {
            let len = text(s[i]).len();
            let j = i + s[i..]
                .iter()
                .take_while(|&&id| text(id).len() == len)
                .count();
            for (x, &a) in s[i..j].iter().enumerate() {
                for &b in &s[i + x + 1..j] {
                    let (ta, tb) = (text(a), text(b));
                    let overlaps =
                        (1..len).any(|k| ta[len - k..] == tb[..k] || tb[len - k..] == ta[..k]);
                    if overlaps {
                        return Some((a, b));
                    }
                }
            }
            i = j;
        }
        None
    }
}
