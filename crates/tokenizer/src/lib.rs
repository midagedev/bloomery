//! Text to token ids and back for GPT-2 byte-level BPE, gemma4's metaspace
//! BPE and llama SentencePiece vocabularies, following `llama_tokenize` and
//! `llama_token_to_piece`.
//!
//! The GGUF header supplies the vocabulary, special tokens and encoding
//! tables. Special tokens are partitioned first. BPE fragments pass through
//! the vocabulary's pre-tokenizer; SentencePiece merges a whole fragment by
//! token score. Both metaspace encoders spell U+0020 as U+2581. Decoding
//! concatenates token pieces without stripping a SentencePiece space prefix.
//!
//! Explicit `tokenizer.ggml.eog_token_ids` replaces heuristic stops, preserving
//! GGUF encoding and token attributes. The private `tokenizer.bloomery.eog_token_ids`
//! also selects HF encoding; if both are present their sorted, deduplicated sets
//! must agree. Explicit lists must be nonempty integer arrays of in-range ids.
//! Flint uses the pinned mainline ordered split program (including its collapsed
//! letter categories), not an HF encoding alias; see the pre-tokenizer module.

mod bpe;
mod collapse_table;
mod decode;
mod encoder;
mod guard;
mod letter_case_table;
mod merge;
mod pretok;
mod spelling;
mod spm;
mod unicode;
mod vocab;

use std::borrow::Cow;
use std::path::Path;

use memchr::memmem::Finder;

pub use decode::{Decoder, SpaceCleaner};
pub use unicode::byte_to_cpt_table;
pub use vocab::{Specials, attr};

use vocab::{Model, Vocab};

/// Why a vocabulary could not be loaded.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}: {1}")]
    Gguf(String, gguf::LoadError),
    #[error("missing metadata key {0}")]
    Missing(&'static str),
    #[error("metadata key {0} has an unexpected type")]
    WrongType(&'static str),
    #[error("tokenizer model {0:?} is not supported ({known})", known = names(&vocab::Model::NAMES))]
    UnsupportedModel(String),
    #[error("pre-tokenizer {0:?} is not supported ({known})", known = names(&pretok::Pre::NAMES))]
    UnsupportedPre(String),
    #[error("tokenizer.ggml.normalizer {0:?} is not supported for this tokenizer model")]
    UnsupportedNormalizer(String),
    #[error("{0}")]
    InvalidStops(String),
    #[error("tokenizer.ggml.token_type has {types} entries for {tokens} tokens")]
    TokenTypesShort { types: usize, tokens: usize },
    #[error("tokenizer.ggml.scores has {scores} entries for {tokens} tokens")]
    TokenScoresShort { scores: usize, tokens: usize },
    #[error("tokenizer.ggml.scores[{id}] is not finite: {score}")]
    InvalidTokenScore { id: usize, score: f32 },
    #[error("a vocabulary of {0} tokens does not fit u32 ids")]
    TooManyTokens(usize),
    #[error("token text {0:?} appears twice in the vocabulary")]
    DuplicateToken(String),
    #[error("tokenizer.ggml.tokens[{id}] {text:?} cannot preserve HF ByteLevel decoding")]
    InvalidHfToken { id: usize, text: String },
    #[error(
        "tokenizer.ggml.merges[{rank}] {merge:?} names a string that is no token, which a Hugging Face BPE cannot"
    )]
    InvalidHfMerge { rank: usize, merge: String },
    #[error(
        "tokenizer.ggml.tokens[{id}] {text:?} is an added token spelling a byte, which a Hugging Face BPE reads only if its model vocabulary lists it, and the header does not say"
    )]
    AmbiguousHfByte { id: usize, text: String },
    #[error("the vocabulary has no byte token <0x{0:02X}>")]
    MissingByteToken(u8),
}

/// Whether this crate runs the spec's tokenizer name: a BPE pre-tokenizer,
/// or `llama` for SentencePiece, which has no regex pre-tokenizer.
#[must_use]
pub fn runs_pre_tokenizer(name: &str) -> bool {
    named(&pretok::Pre::NAMES, name).is_some()
        || matches!(named(&Model::NAMES, name), Some(Model::Spm))
}

/// The value `name` stands for in a table of names this crate runs; `None`
/// for a name it does not run.
fn named<T: Copy>(table: &[(&str, T)], name: &str) -> Option<T> {
    table.iter().find(|(n, _)| *n == name).map(|&(_, v)| v)
}

/// A name table's names, comma-separated.
fn names<T>(table: &[(&str, T)]) -> String {
    let names: Vec<&str> = table.iter().map(|&(n, _)| n).collect();
    names.join(", ")
}

/// Whether one of the ascending `cuts` lies in `start..=end`: a cut stands beside the spelling it
/// guards, and a match that holds the byte before it or the byte after it is that spelling or
/// overlaps it. A cut is never at the start or the end of the client text it is in, where a token
/// the template wrote would be beside it.
fn spans_cut(cuts: &[usize], start: usize, end: usize) -> bool {
    cuts.get(cuts.partition_point(|&c| c < start))
        .is_some_and(|&c| c <= end)
}

/// A loaded vocabulary with its encoder and decoder tables. A clone is the
/// same tables, copied: cheaper than building them again from the keys.
#[derive(Clone)]
pub struct Tokenizer {
    vocab: Vocab,
    encoder: encoder::Encoder,
    /// Partition order: longest text first.
    specials: Vec<u32>,
    finders: Vec<Finder<'static>>,
    /// Whether the partition cuts leftmost-longest, as Hugging Face's `tokenizers` splits on its
    /// added tokens (a vocabulary marked as one, [`Pre::hf_reading`]), where the reference's
    /// order, kept for every other vocabulary, cuts the longest token first.
    leftmost: bool,
    /// The special tokens' spellings, as [`Tokenizer::guard`] searches them.
    spellings: guard::Spellings,
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
        let path = path.as_ref();
        let inv =
            gguf::inventory_of(path).map_err(|e| Error::Gguf(path.display().to_string(), e))?;
        Tokenizer::from_keys(&inv.meta)
    }

    /// Load the vocabulary from a model header's metadata pairs (`gguf::Split::keys`): a
    /// checkpoint folder's header, which no file holds, reads as a GGUF file's.
    pub fn from_keys(keys: &[(String, gguf::Value)]) -> Result<Tokenizer, Error> {
        let (vocab, encoding, hf_texts) = Vocab::load(keys)?;
        let leftmost = hf_texts.is_some();
        let encoder = encoder::Encoder::new(encoding, &vocab)?;
        let spelling = encoder.spelling();
        let specials = vocab.special_order();
        let finders = specials
            .iter()
            .map(|&id| Finder::new(vocab.texts[id as usize].as_bytes()).into_owned())
            .collect();
        let spellings =
            guard::Spellings::new(specials.iter().map(|&id| vocab.texts[id as usize].as_str()));
        let pieces = decode::pieces(&vocab.texts, &vocab.attrs, spelling);
        if let Some(texts) = hf_texts {
            decode::check_hf(&texts, &vocab.attrs, &pieces)?;
        }
        Ok(Tokenizer {
            vocab,
            encoder,
            specials,
            finders,
            leftmost,
            spellings,
            pieces,
        })
    }

    /// Token ids of `text`. `add_special` adds BOS/EOS where the vocabulary
    /// asks for them ([`Tokenizer::add_bos`], [`Tokenizer::add_eos`]);
    /// `parse_special` lets CONTROL tokens written in the text become their
    /// ids (USER_DEFINED tokens always do, as in the reference).
    ///
    /// # Panics
    ///
    /// BPE panics if a four-byte UTF-8 sequence decodes above U+10FFFF.
    /// Other malformed bytes follow the reference's lenient decoding;
    /// SentencePiece preserves malformed bytes through byte fallback.
    pub fn encode(
        &self,
        text: impl AsRef<[u8]>,
        add_special: bool,
        parse_special: bool,
    ) -> Vec<u32> {
        self.encode_cut(text.as_ref(), &[], add_special, parse_special)
    }

    /// [`Tokenizer::encode`] of a text that carries `mark`s: each occurrence of the non-empty `mark`
    /// is removed from the text, and no special token matches that holds the place one stood or
    /// the byte on either side of it. The text between the marks is one text to the BPE, as if
    /// the marks were not there, so the ids of a text with no mark are `encode`'s exactly.
    /// [`Tokenizer::guard`] puts the marks beside the spellings of special tokens that are plain
    /// text, and a mark any other way is cut out all the same.
    ///
    /// # Panics
    ///
    /// On an empty `mark`, and as [`Tokenizer::encode`] does.
    pub fn encode_marked(
        &self,
        text: &str,
        mark: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Vec<u32> {
        assert!(!mark.is_empty(), "encode_marked: the mark is empty");
        let mut clean = String::with_capacity(text.len());
        let mut cuts = Vec::new();
        for (i, piece) in text.split(mark).enumerate() {
            if i > 0 {
                cuts.push(clean.len());
            }
            clean.push_str(piece);
        }
        self.encode_cut(clean.as_bytes(), &cuts, add_special, parse_special)
    }

    /// `text` with `mark` spliced in beside every spelling of a special token it carries, and beside
    /// the beginning or the end of one it ends or starts with, which what a template writes next to
    /// the text could finish: before the spelling, after it when it starts the text (so a template's
    /// `startswith` and `endswith` of a spelling see it where it was), and after the spelling's
    /// first character when no side of the spelling holds text that is not whitespace, for a
    /// template's `trim` would leave the mark at the edge of the text, beside a token the template
    /// wrote there. The text with the marks cut out is `text`, and no special token matches over a
    /// mark ([`Tokenizer::encode_marked`]). A spelling of one character is left alone, which no mark
    /// could be beside without changing what the text starts or ends with. What this finds is what
    /// `parse_special = true` would turn into a token, CONTROL, USER_DEFINED and UNKNOWN alike; the
    /// text comes back borrowed when it carries none. Spellings that overlap are each marked. What a
    /// template builds from its own pieces around the text (`'<' ~ name ~ '>'`) is not found: no part
    /// of the text is a spelling.
    ///
    /// # Panics
    ///
    /// On an empty `mark`.
    pub fn guard<'a>(&self, text: &'a str, mark: &str) -> Cow<'a, str> {
        assert!(!mark.is_empty(), "guard: the mark is empty");
        let at = self.spellings.cuts(text);
        if at.is_empty() {
            return Cow::Borrowed(text);
        }
        let mut out = String::with_capacity(text.len() + at.len() * mark.len());
        let mut last = 0;
        for cut in at {
            out.push_str(&text[last..cut]);
            out.push_str(mark);
            last = cut;
        }
        out.push_str(&text[last..]);
        Cow::Owned(out)
    }

    /// The body of [`Tokenizer::encode`]: `cuts` are ascending byte offsets in `text` that no
    /// special token's match may span.
    fn encode_cut(
        &self,
        text: &[u8],
        cuts: &[usize],
        add_special: bool,
        parse_special: bool,
    ) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 2);
        if add_special
            && self.vocab.add_bos
            && let Some(bos) = self.vocab.specials.bos
        {
            out.push(bos);
        }

        let mut scratch = encoder::Scratch::default();
        for frag in self.partition(text, cuts, parse_special) {
            match frag {
                Frag::Token(id) => out.push(id),
                Frag::Text { start, len } => {
                    self.encoder.text(
                        &text[start..start + len],
                        &self.vocab.token_to_id,
                        &mut scratch,
                        &mut out,
                    );
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
    /// occurrence out of every text fragment still unsplit. An occurrence that
    /// reaches one of the ascending `cuts` ([`spans_cut`]) is not one: the
    /// search goes on from the next byte.
    fn partition(&self, text: &[u8], cuts: &[usize], parse_special: bool) -> Vec<Frag> {
        let mut frags = Vec::new();
        if text.is_empty() {
            return frags;
        }
        if self.leftmost {
            return self.partition_leftmost(text, cuts, parse_special);
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
                // `at` starts the text not yet pushed, `from` the search.
                let mut at = start;
                let mut from = start;
                while let Some(m) = finder.find(&text[from..end]) {
                    let found = from + m;
                    if spans_cut(cuts, found, found + tlen) {
                        from = found + 1;
                        continue;
                    }
                    changed = true;
                    if found > at {
                        next.push(Frag::Text {
                            start: at,
                            len: found - at,
                        });
                    }
                    next.push(Frag::Token(id));
                    at = found + tlen;
                    from = at;
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

    /// The partition of Hugging Face's `tokenizers`: the leftmost occurrence of any special
    /// token is cut out, the longest where several start there, and the search goes on after
    /// it, so a token that starts inside a cut one is no match. Cuts and `parse_special` are
    /// [`Tokenizer::partition`]'s. Each token's next occurrence is kept and searched again only
    /// when a cut passes it, so no byte is scanned more than once a token.
    fn partition_leftmost(&self, text: &[u8], cuts: &[usize], parse_special: bool) -> Vec<Frag> {
        let active: Vec<usize> = (0..self.specials.len())
            .filter(|&k| {
                parse_special
                    || self.vocab.attrs[self.specials[k] as usize] & (attr::CONTROL | attr::UNKNOWN)
                        == 0
            })
            .collect();
        let find = |k: usize, mut from: usize| {
            let tlen = self.finders[k].needle().len();
            while let Some(m) = self.finders[k].find(&text[from..]) {
                let found = from + m;
                if !spans_cut(cuts, found, found + tlen) {
                    return Some(found);
                }
                from = found + 1;
            }
            None
        };
        let mut next: Vec<Option<usize>> = active.iter().map(|&k| find(k, 0)).collect();
        let (mut frags, mut at) = (Vec::new(), 0);
        // Leftmost, then longest: the order of `specials` is longest first, so the first of
        // equal starts wins.
        while let Some((i, start)) = next
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.map(|s| (i, s)))
            .min_by_key(|&(i, s)| (s, i))
        {
            let k = active[i];
            let end = start + self.finders[k].needle().len();
            if start > at {
                frags.push(Frag::Text {
                    start: at,
                    len: start - at,
                });
            }
            frags.push(Frag::Token(self.specials[k]));
            at = end;
            for (i, s) in next.iter_mut().enumerate() {
                if s.is_some_and(|s| s < at) {
                    *s = find(active[i], at);
                }
            }
        }
        if at < text.len() {
            frags.push(Frag::Text {
                start: at,
                len: text.len() - at,
            });
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
    /// UTF-8 become U+FFFD. A vocabulary that cleans spaces
    /// ([`Tokenizer::space_cleaner`]) cleans the text as the reference's
    /// `clean_spaces` does.
    pub fn decode(&self, ids: &[u32]) -> String {
        let text = String::from_utf8_lossy(&self.decode_bytes(ids, true)).into_owned();
        if self.vocab.clean_spaces {
            decode::clean_str(&text)
        } else {
            text
        }
    }

    /// A [`SpaceCleaner`] when this vocabulary's decoded text has its spaces cleaned (the
    /// reference's `clean_spaces` for its pre-tokenizer's name), for a reader that streams
    /// pieces one at a time; `None` when decode returns its text as is.
    #[must_use]
    pub fn space_cleaner(&self) -> Option<SpaceCleaner> {
        self.vocab.clean_spaces.then(SpaceCleaner::default)
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

    /// Whether the vocabulary asks for a BOS / an EOS around encoded text:
    /// `tokenizer.ggml.add_bos_token` / `add_eos_token`, and a BOS always for
    /// the `gemma4` pre-tokenizer, as the reference forces it. SentencePiece and the
    /// `llama-bpe` pre-tokenizer default to adding a BOS unless the header disables it, as
    /// the reference's SentencePiece and llama3 pre types do.
    pub fn add_bos(&self) -> bool {
        self.vocab.add_bos
    }

    /// See [`Tokenizer::add_bos`].
    pub fn add_eos(&self) -> bool {
        self.vocab.add_eos
    }

    /// End-of-generation ids, ascending and unique. An explicit GGUF stop
    /// array replaces inferred stops, without changing special-token parsing.
    pub fn eog(&self) -> &[u32] {
        &self.vocab.eog
    }

    /// The pre-tokenizer's name: `tokenizer.ggml.pre`, or `gemma4` for a
    /// gemma4 model, which names its own; SentencePiece uses `default`.
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
        self.encoder.merges_off_vocab()
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

#[cfg(test)]
mod tests {
    use gguf::Value;
    use gguf::write::{Layout, Writer};

    use super::{Decoder, Error, SpaceCleaner, Tokenizer, attr, byte_to_cpt_table};

    /// A gemma4 vocabulary of a few tokens: six CONTROL tokens, one
    /// USER_DEFINED token, the 256 `<0xXX>` byte tokens but `missing_byte`,
    /// then the NORMAL tokens below. Like the released file's, the header
    /// names no pre-tokenizer; unlike it, it sets `add_bos_token` false, so a
    /// BOS can come only from the gemma4 rule. `header` adds the special ids
    /// and `add_eos_token`.
    fn gemma4_header(header: &[(&str, Value)], missing_byte: Option<u8>) -> Vec<(String, Value)> {
        let control = ["<pad>", "<eos>", "<bos>", "<unk>", "<turn|>", "<|channel|>"];
        let normal = [
            "a",
            "b",
            "\u{2581}",
            "\u{2581}b",
            "\u{2581}\u{2581}",
            "\u{2581}\u{2581}\u{2581}\u{2581}",
            "\n",
            "\n\n\n",
            "x",
            "y",
            "한",
            "국",
            "한국",
            "é",
            "😀",
        ];
        let mut texts: Vec<String> = control.iter().map(|t| t.to_string()).collect();
        let mut types = vec![3; control.len()];
        texts.push("<mark>".to_string());
        types.push(4);
        let bytes = (0..=u8::MAX).filter(|&b| Some(b) != missing_byte);
        texts.extend(bytes.map(|b| format!("<0x{b:02X}>")));
        types.resize(texts.len(), 6);
        texts.extend(normal.iter().map(|t| t.to_string()));
        types.resize(texts.len(), 1);
        let merges = [
            "\u{2581} b",
            "x y",
            "한 국",
            "\u{2581} \u{2581}",
            "\u{2581}\u{2581} \u{2581}\u{2581}",
        ];
        [
            ("tokenizer.ggml.model", Value::String("gemma4".into())),
            (
                "tokenizer.ggml.tokens",
                Value::Array(texts.into_iter().map(Value::String).collect()),
            ),
            (
                "tokenizer.ggml.token_type",
                Value::Array(types.into_iter().map(Value::I32).collect()),
            ),
            (
                "tokenizer.ggml.merges",
                Value::Array(
                    merges
                        .iter()
                        .map(|m| Value::String(m.to_string()))
                        .collect(),
                ),
            ),
            ("tokenizer.ggml.add_bos_token", Value::Bool(false)),
        ]
        .into_iter()
        .chain(header.iter().cloned())
        .map(|(k, v)| (k.to_string(), v))
        .collect()
    }

    pub(super) fn from_header(kvs: &[(String, Value)]) -> Result<Tokenizer, Error> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "tokenizer-{}-{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        // The writer needs an element type; empty-array reader fixtures use a typed placeholder.
        let mut typed = kvs.to_vec();
        for (_, value) in &mut typed {
            if matches!(value, Value::Array(a) if a.is_empty()) {
                *value = Value::Array(vec![Value::U32(0)]);
            }
        }
        let layout = Layout::new(&typed, Vec::new()).unwrap();
        let mut bytes = Writer::new(Vec::new(), layout)
            .and_then(Writer::finish)
            .unwrap();
        for (key, value) in kvs.iter().rev() {
            if matches!(value, Value::Array(a) if a.is_empty()) {
                let at = bytes
                    .windows(key.len())
                    .position(|s| s == key.as_bytes())
                    .unwrap()
                    + key.len();
                bytes[at + 8..at + 16].copy_from_slice(&0u64.to_le_bytes());
                bytes.drain(at + 16..at + 20);
            }
        }
        std::fs::write(&path, bytes).unwrap();
        let tok = Tokenizer::from_gguf(&path);
        std::fs::remove_file(&path).unwrap();
        tok
    }

    fn open(header: &[(&str, Value)], missing_byte: Option<u8>) -> Result<Tokenizer, Error> {
        from_header(&gemma4_header(header, missing_byte))
    }

    /// The ids of the tokens whose texts are `texts`.
    pub(super) fn ids(tok: &Tokenizer, texts: &[&str]) -> Vec<u32> {
        let id = |t: &str| {
            (0..tok.n_vocab() as u32)
                .find(|&i| tok.text(i) == Some(t))
                .unwrap()
        };
        texts.iter().map(|t| id(t)).collect()
    }

    #[test]
    fn qwen35_nfc_matches_hf() {
        // Hugging Face tokenizers 0.22.2 and 0.21.0: NFC, ByteLevel, byte ids and an unused a+b merge.
        let cases: &[(&str, &[u32])] = &[
            (
                "\u{1112}\u{1161}\u{11ab}\u{1100}\u{116e}\u{11a8}",
                &[237, 149, 156, 234, 181, 173],
            ),
            ("e\u{301}", &[195, 169]),
            ("A\u{30a}\u{301}", &[199, 186]),
            (
                "한국 é Ǻ",
                &[237, 149, 156, 234, 181, 173, 32, 195, 169, 32, 199, 186],
            ),
            // HF's tables are Unicode 9.0's: no Dives Akuru composition (13.0), and U+0EBA
            // (12.0) is a starter, not a mark of class 9 to reorder before U+0305.
            (
                "\u{11935}\u{11930}",
                &[240, 145, 164, 181, 240, 145, 164, 176],
            ),
            ("a\u{305}\u{eba}", &[97, 204, 133, 224, 186, 186]),
        ];
        let mut failures = Vec::new();
        for pre in ["qwen35", "qwen2", "glm4", "pixtral", "deepseek-v3"] {
            let mut header = vec![
                ("tokenizer.ggml.model".into(), Value::String("gpt2".into())),
                ("tokenizer.ggml.pre".into(), Value::String(pre.into())),
                (
                    "tokenizer.ggml.tokens".into(),
                    Value::Array(
                        crate::unicode::byte_to_cpt_table()
                            .into_iter()
                            .map(|cp| Value::String(char::from_u32(cp).unwrap().to_string()))
                            .chain([Value::String("ab".into())])
                            .collect(),
                    ),
                ),
                (
                    "tokenizer.ggml.merges".into(),
                    Value::Array(vec![Value::String("a b".into())]),
                ),
            ];
            for normalizer in [None, Some("NFC")] {
                if let Some(name) = normalizer {
                    header.push((
                        "tokenizer.ggml.normalizer".into(),
                        Value::String(name.into()),
                    ));
                }
                let tok = from_header(&header).unwrap();
                for &(text, nfc) in cases {
                    let raw: Vec<u32> = text.bytes().map(u32::from).collect();
                    let want = if normalizer.is_some() { nfc } else { &raw };
                    let actual = tok.encode(text, false, false);
                    if actual != want {
                        failures.push(format!(
                            "{pre} {normalizer:?} {text:?}: {actual:?} != {want:?}"
                        ));
                    }
                }
            }
            for value in [Value::String("NFD".into()), Value::U32(1)] {
                header.last_mut().unwrap().1 = value.clone();
                match from_header(&header) {
                    Err(e)
                        if e.to_string().contains("tokenizer.ggml.normalizer")
                            && match &value {
                                Value::String(name) => e.to_string().contains(name),
                                _ => true,
                            } => {}
                    Err(e) => failures.push(format!("{pre} {value:?}: wrong error {e}")),
                    Ok(_) => failures.push(format!("{pre} {value:?}: accepted unknown normalizer")),
                }
            }
            header[0].1 = Value::String("llama".into());
            header.last_mut().unwrap().1 = Value::String("NFC".into());
            assert!(
                matches!(from_header(&header), Err(Error::UnsupportedNormalizer(name)) if name == "NFC")
            );
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// A byte vocabulary with Flint's stop roles, without a downloaded fixture.
    fn flint_header() -> Vec<(String, Value)> {
        let mut texts: Vec<Value> = byte_to_cpt_table()
            .into_iter()
            .map(|cp| Value::String(char::from_u32(cp).unwrap().to_string()))
            .collect();
        texts.extend(
            [
                "<|begin_of_text|>",
                "<|end_of_text|>",
                "<|user|>",
                "<|observation|>",
                "<|developer|>",
                "<|channel|>",
            ]
            .map(|s| Value::String(s.into())),
        );
        let mut types = vec![Value::I32(1); 256];
        // The two inferred stop roles must retain GGUF's CONTROL overrides.
        types.extend([3, 3, 1, 1, 3, 3].map(Value::I32));
        vec![
            ("tokenizer.ggml.model".into(), Value::String("gpt2".into())),
            ("tokenizer.ggml.pre".into(), Value::String("flint".into())),
            ("tokenizer.ggml.tokens".into(), Value::Array(texts)),
            ("tokenizer.ggml.token_type".into(), Value::Array(types)),
            ("tokenizer.ggml.merges".into(), Value::Array(vec![])),
            ("tokenizer.ggml.bos_token_id".into(), Value::U32(256)),
            ("tokenizer.ggml.eos_token_id".into(), Value::U32(257)),
            (
                "tokenizer.ggml.eog_token_ids".into(),
                Value::Array([260, 259, 257, 258, 260].map(Value::U32).to_vec()),
            ),
        ]
    }

    #[test]
    fn flint_stops_preserve_gguf_encoding_and_special_parsing() {
        let mut header = flint_header();
        let tok = from_header(&header).unwrap();
        assert_eq!(tok.eog(), &[257, 258, 259, 260]);
        assert!(!tok.add_bos());
        assert!(!tok.add_eos());
        // The byte ids are their byte values: no NFC, prefix space or automatic BOS/EOS.
        for text in ["", "e\u{301}", "HelloWorld", " 中", "1234567"] {
            let want: Vec<u32> = text.bytes().map(u32::from).collect();
            assert_eq!(tok.encode(text, true, true), want, "{text:?}");
        }
        for (id, text) in [
            (257, "<|end_of_text|>"),
            (258, "<|user|>"),
            (259, "<|observation|>"),
            (260, "<|developer|>"),
        ] {
            assert_eq!(tok.encode(text, false, true), [id]);
            assert_eq!(
                tok.encode(text, false, false),
                text.bytes().map(u32::from).collect::<Vec<_>>(),
            );
            let original = if matches!(id, 258 | 259) {
                attr::NORMAL
            } else {
                0
            };
            assert_eq!(tok.attr(id), Some(original | attr::CONTROL));
            assert!(tok.piece(id, false).is_empty());
            assert_eq!(tok.piece(id, true), text.as_bytes());
        }
        assert_eq!(tok.attr(261), Some(attr::USER_DEFINED));
        assert_eq!(tok.encode("<|channel|>", false, false), [261]);

        // An explicit list wins over scalar EOS and name-based guesses, not attributes.
        header.last_mut().unwrap().1 = Value::Array(vec![Value::U32(260)]);
        let tok = from_header(&header).unwrap();
        assert_eq!(tok.eog(), &[260]);
        assert_eq!(tok.specials().eos, Some(257));
        assert_eq!(tok.encode("<|user|>", false, true), [258]);
        header.pop();
        // Without the array, the pinned mainline guesses three, not developer.
        assert_eq!(from_header(&header).unwrap().eog(), &[257, 258, 259]);
    }

    #[test]
    fn explicit_stop_sources_validate_and_agree_as_sets() {
        let mut header = flint_header();
        for value in [
            Value::Array(vec![]),
            Value::Array(vec![Value::I32(-1)]),
            Value::Array(vec![Value::U32(262)]),
            Value::Array(vec![Value::U64(u64::MAX)]),
            Value::Array(vec![Value::F32(257.0)]),
            Value::Array(vec![Value::Bool(true)]),
            Value::Array(vec![Value::String("257".into())]),
            Value::U32(257),
        ] {
            header.last_mut().unwrap().1 = value;
            let error = from_header(&header).err().expect("invalid GGUF stop list");
            assert!(error.to_string().contains("tokenizer.ggml.eog_token_ids"));
        }
        // All integer widths follow the existing metadata integer contract.
        for id in [
            Value::I8(1),
            Value::U8(1),
            Value::I16(1),
            Value::U16(1),
            Value::I32(1),
            Value::U32(1),
            Value::I64(1),
            Value::U64(1),
        ] {
            header.last_mut().unwrap().1 = Value::Array(vec![id.clone(), id]);
            assert_eq!(from_header(&header).unwrap().eog(), &[1]);
        }

        header[1].1 = Value::String("qwen35".into());
        header.last_mut().unwrap().1 =
            Value::Array(vec![Value::U32(260), Value::U32(257), Value::U32(260)]);
        header.push((
            "tokenizer.bloomery.eog_token_ids".into(),
            Value::Array(vec![Value::I32(257), Value::I32(260)]),
        ));
        let tok = from_header(&header).unwrap();
        assert_eq!(tok.eog(), &[257, 260]);
        // Equal arrays retain the private key's HF attribute contract.
        assert_eq!(tok.attr(261), Some(attr::CONTROL));
        assert_eq!(tok.attr(258), Some(attr::NORMAL));
        header.last_mut().unwrap().1 = Value::Array(vec![Value::U32(260)]);
        let error = from_header(&header)
            .err()
            .expect("conflicting stop sources");
        assert!(matches!(error, Error::InvalidStops(_)));
        let error = error.to_string();
        assert!(error.contains("tokenizer.ggml.eog_token_ids"));
        assert!(error.contains("tokenizer.bloomery.eog_token_ids"));
    }

    #[test]
    fn flint_replaces_invalid_utf8_without_normalizing_valid_text() {
        let tok = from_header(&flint_header()).unwrap();
        for (bytes, text) in [
            (&b"\xc0\x80"[..], "\u{fffd}\u{fffd}"),
            (&b"\xf4\x90\x80\x80"[..], "\u{fffd}\u{fffd}\u{fffd}\u{fffd}"),
            (&b"\xe2\x82"[..], "\u{fffd}\u{fffd}"),
            ("e\u{301}".as_bytes(), "e\u{301}"),
        ] {
            assert_eq!(
                tok.encode(bytes, false, false),
                text.bytes().map(u32::from).collect::<Vec<_>>(),
                "{bytes:?}"
            );
        }
        // The pinned reference preserves surrogate encodings, unlike overlong forms.
        assert_eq!(tok.encode(b"\xed\xa0\x80", false, false), [237, 160, 128]);
    }

    /// The tree that reads `gpt-4o` (Muse Spark Open's) turns an overlong form and a truncated
    /// sequence into U+FFFD a byte at a time and passes a surrogate; the trees that read
    /// `minimax-m2` accept `c0 80` as U+0000. FAIL-first: with `gpt-4o` on the lenient reader its
    /// row yields a NUL byte for `c0 80`.
    #[test]
    fn gpt_4o_decodes_utf8_strictly_and_minimax_m2_leniently() {
        let bytes = b"\xc0\x80 \xed\xa0\x80 \xf0\x9f\x98";
        let replaced = |n: usize| "\u{fffd}".repeat(n).into_bytes();
        let want = |head: Vec<u8>| {
            [head, b" \xed\xa0\x80 ".to_vec(), replaced(3)]
                .concat()
                .into_iter()
                .map(u32::from)
                .collect::<Vec<u32>>()
        };
        for (pre, head) in [("gpt-4o", replaced(2)), ("minimax-m2", vec![0])] {
            let mut header = flint_header();
            header[1].1 = Value::String(pre.into());
            let tok = from_header(&header).unwrap();
            assert_eq!(tok.encode(&bytes[..], false, false), want(head), "{pre}");
        }
    }

    /// `<|eot|>` is one of the reference's end-of-generation names (Muse
    /// Spark's template ends its turns with it): a vocabulary carrying it
    /// ends a generation at it, beside its own eos. FAIL-first: with the name
    /// out of the EOG list the row below fails.
    #[test]
    fn eot_by_name_ends_a_generation() {
        let texts = ["a", "<|end_of_text|>", "<|eot|>"];
        let header = vec![
            ("tokenizer.ggml.model".into(), Value::String("gpt2".into())),
            ("tokenizer.ggml.pre".into(), Value::String("gpt-4o".into())),
            (
                "tokenizer.ggml.tokens".into(),
                Value::Array(texts.map(|s| Value::String(s.into())).to_vec()),
            ),
            (
                "tokenizer.ggml.token_type".into(),
                Value::Array([1, 3, 3].map(Value::I32).to_vec()),
            ),
            ("tokenizer.ggml.merges".into(), Value::Array(vec![])),
            ("tokenizer.ggml.eos_token_id".into(), Value::U32(1)),
        ];
        let tok = from_header(&header).unwrap();
        assert_eq!(tok.eog(), &[1, 2]);
        assert_eq!(tok.specials().eos, Some(1));
    }

    #[test]
    fn explicit_stops_preserve_hf_ids_and_attributes() {
        let texts = [
            "a",
            "<|endoftext|>",
            "<|im_end|>",
            "<|fim_pad|>",
            "<|repo_name|>",
            "<|file_sep|>",
            "<|channel|>",
            "<|message|>",
            "<|start|>",
            "<|constrain|>",
            "<|end|>",
            "<|call|>",
            "<|return|>",
            "<unused0>",
        ];
        let mut header = vec![
            ("tokenizer.ggml.model".into(), Value::String("gpt2".into())),
            ("tokenizer.ggml.pre".into(), Value::String("qwen35".into())),
            (
                "tokenizer.ggml.tokens".into(),
                Value::Array(texts.map(|s| Value::String(s.into())).to_vec()),
            ),
            (
                "tokenizer.ggml.token_type".into(),
                Value::Array(
                    [1, 4, 3, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 3]
                        .map(Value::I32)
                        .to_vec(),
                ),
            ),
            ("tokenizer.ggml.merges".into(), Value::Array(vec![])),
            ("tokenizer.ggml.eos_token_id".into(), Value::U32(1)),
        ];
        let legacy = from_header(&header).unwrap();
        assert_eq!(legacy.eog(), &[1, 2, 3, 4, 5, 11, 12]);
        assert_eq!(legacy.specials().bos, Some(11));
        assert_eq!(legacy.attr(6), Some(attr::USER_DEFINED));
        assert_eq!(legacy.attr(3), Some(attr::CONTROL | attr::USER_DEFINED));
        header.push((
            "tokenizer.bloomery.eog_token_ids".into(),
            Value::Array(vec![Value::U32(2)]),
        ));
        for ids in [vec![2], vec![12, 10, 2, 2]] {
            header.last_mut().unwrap().1 =
                Value::Array(ids.iter().copied().map(Value::U32).collect());
            let tok = from_header(&header).unwrap();
            let mut want = ids;
            want.sort_unstable();
            want.dedup();
            assert_eq!(tok.eog(), want);
            assert_eq!(tok.specials().bos, None);
            assert_eq!(tok.specials().eos, Some(1));
            assert_eq!(tok.specials().eot, None);
            assert_eq!(tok.specials().fim_pad, None);
            for (id, ty) in [1, 4, 3, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 3]
                .into_iter()
                .enumerate()
            {
                let expected = match ty {
                    1 => attr::NORMAL,
                    3 => attr::CONTROL,
                    _ => attr::USER_DEFINED,
                };
                assert_eq!(tok.attr(id as u32), Some(expected), "{}", texts[id]);
                assert_eq!(tok.piece(id as u32, false).is_empty(), ty == 3);
            }
        }
        let mut mismatches = Vec::new();
        for (text, ty, decoded) in [
            ("", 1, None),
            ("▁", 1, None),
            ("▁x", 1, None),
            ("Ġx", 4, None),
            ("Ġx", 3, None),
            ("é", 4, None),
            ("Ġx", 1, Some(b" x".as_slice())),
            ("é", 1, Some(b"\xe9".as_slice())),
            ("▁x", 4, Some("▁x".as_bytes())),
            ("Ġ▁x", 4, Some("Ġ▁x".as_bytes())),
        ] {
            let mut changed = header.clone();
            changed.last_mut().unwrap().1 = Value::Array(vec![Value::U32(2)]);
            let Value::Array(tokens) = &mut changed[2].1 else {
                panic!();
            };
            tokens[0] = Value::String(text.into());
            let Value::Array(types) = &mut changed[3].1 else {
                panic!();
            };
            types[0] = Value::I32(ty);
            match (from_header(&changed), decoded) {
                (Ok(tok), Some(bytes)) => assert_eq!(tok.piece(0, true), bytes, "{text:?}"),
                (Err(error), None) => {
                    let error = error.to_string();
                    assert!(error.contains("tokens[0]"), "{error}");
                    assert!(error.contains(&format!("{text:?}")), "{error}");
                }
                (Ok(tok), None) => mismatches.push(format!(
                    "{text:?} type {ty}: accepted, piece {:?}",
                    tok.piece(0, true)
                )),
                (Err(error), Some(_)) => {
                    mismatches.push(format!("{text:?} type {ty}: {error}"));
                }
            }
        }
        assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
        for value in [
            Value::Array(vec![]),
            Value::Array(vec![Value::I32(-1)]),
            Value::Array(vec![Value::U32(texts.len() as u32)]),
            Value::Array(vec![Value::U64(u64::MAX)]),
            Value::Array(vec![Value::String("2".into())]),
            Value::U32(2),
        ] {
            header.last_mut().unwrap().1 = value.clone();
            let error = from_header(&header)
                .err()
                .expect("invalid stop key accepted");
            assert!(
                error
                    .to_string()
                    .contains("tokenizer.bloomery.eog_token_ids"),
                "{value:?}: {error}"
            );
        }
    }

    /// A family's turn-end token ends generation: a vocabulary holding it (and no other end name)
    /// stops on it, as a control token. One row a family whose end token is not the file's
    /// `eos_token_id`: MiniMax-M3's `[e~[` and Solar-Open2's `<|im:end|>` (the template writes it
    /// after every assistant message; the file's `eos_token_id` is `<|endoftext|>`).
    ///
    /// FAIL-first: a vocabulary that lists the row's name as no end-of-generation name runs past
    /// the token, and that row fails.
    #[test]
    fn a_familys_turn_end_token_ends_generation() {
        for end in ["[e~[", "<|im:end|>"] {
            let header = vec![
                ("tokenizer.ggml.model".into(), Value::String("gpt2".into())),
                ("tokenizer.ggml.pre".into(), Value::String("qwen35".into())),
                (
                    "tokenizer.ggml.tokens".into(),
                    Value::Array(vec![Value::String("a".into()), Value::String(end.into())]),
                ),
                ("tokenizer.ggml.merges".into(), Value::Array(vec![])),
            ];
            let tok = from_header(&header).unwrap();
            assert_eq!(tok.eog(), &[1], "{end}");
            assert!(tok.attr(1).is_some_and(|a| a & attr::CONTROL != 0), "{end}");
        }
    }

    /// Raon's pre-tokenizer NFC-normalizes the text whatever the header's normalizer says, and the
    /// converter's boolean `tokenizer.ggml.normalizer.nfc` selects the same for another BPE; a
    /// SentencePiece vocabulary refuses it by name. `<|im_end|>` and `<|endoftext|>`, the two
    /// ends the model's generation config lists, end generation.
    ///
    /// FAIL-first: without the pre-tokenizer's own NFC the `raon` row keeps the combining mark's
    /// two bytes; without the boolean key's read the `qwen35` row does.
    #[test]
    fn raon_normalizes_to_nfc_and_ends_on_its_two_end_tokens() {
        let bytes: Vec<Value> = crate::unicode::byte_to_cpt_table()
            .into_iter()
            .map(|cp| Value::String(char::from_u32(cp).unwrap().to_string()))
            .collect();
        let header = |pre: &str, flag: Option<bool>| {
            let mut h = vec![
                ("tokenizer.ggml.model".into(), Value::String("gpt2".into())),
                ("tokenizer.ggml.pre".into(), Value::String(pre.into())),
                (
                    "tokenizer.ggml.tokens".into(),
                    Value::Array(
                        bytes
                            .iter()
                            .cloned()
                            .chain(
                                ["ab", "<|im_end|>", "<|endoftext|>"]
                                    .map(|t| Value::String(t.into())),
                            )
                            .collect(),
                    ),
                ),
                (
                    "tokenizer.ggml.merges".into(),
                    Value::Array(vec![Value::String("a b".into())]),
                ),
            ];
            h.push(("tokenizer.ggml.eos_token_id".into(), Value::U32(257)));
            if let Some(flag) = flag {
                h.push(("tokenizer.ggml.normalizer.nfc".into(), Value::Bool(flag)));
            }
            h
        };
        let composed = [195, 169];
        let raw: Vec<u32> = "e\u{301}".bytes().map(u32::from).collect();
        for (pre, flag, want) in [
            ("raon", None, &composed[..]),
            ("raon", Some(true), &composed[..]),
            ("qwen35", Some(true), &composed[..]),
            ("qwen35", Some(false), &raw[..]),
            ("qwen35", None, &raw[..]),
        ] {
            let tok = from_header(&header(pre, flag)).unwrap();
            assert_eq!(tok.encode("e\u{301}", false, false), want, "{pre} {flag:?}");
        }
        let tok = from_header(&header("raon", None)).unwrap();
        assert_eq!(tok.eog(), &[257, 258]);
        let mut spm = header("raon", Some(true));
        spm[0].1 = Value::String("llama".into());
        assert!(
            matches!(from_header(&spm), Err(Error::UnsupportedNormalizer(name)) if name == "nfc")
        );
    }

    /// The explicit stop list selects Hugging Face's encoding, and only for a pre-tokenizer that
    /// has an HF reading (`qwen35`, `qwen2`, `minimax-m2`; `glm4` has none and is refused): qwen35's
    /// regex folds `ſ` to `s` in a contraction ("'ſx" is `'ſ` and `x`, ids 257 and 120 under
    /// the merges below, where qwen35 merges the one word to 258), and its BPE skips a byte its
    /// vocabulary lacks and merges across it ("acb" is `ab`), and a pair listed twice ranks at its
    /// last place ("abc" is `ab c`). Native `tokenizers` 0.21.0 gives the HF ids. A merge naming no
    /// token, or an added token spelling a byte, has no HF reading.
    ///
    /// FAIL-first: without the fold, the skip or the last rank, the HF row of its case fails.
    #[test]
    fn the_hf_marker_encodes_as_hugging_face() {
        let stop = (
            "tokenizer.bloomery.eog_token_ids".to_string(),
            Value::Array(vec![Value::U32(0)]),
        );
        let header = |tokens: Vec<String>, merges: &[&str], types: Option<Vec<i32>>| {
            let mut kvs = vec![
                (
                    "tokenizer.ggml.model".to_string(),
                    Value::String("gpt2".into()),
                ),
                (
                    "tokenizer.ggml.pre".to_string(),
                    Value::String("qwen35".into()),
                ),
                (
                    "tokenizer.ggml.tokens".to_string(),
                    Value::Array(tokens.into_iter().map(Value::String).collect()),
                ),
                (
                    "tokenizer.ggml.merges".to_string(),
                    Value::Array(
                        merges
                            .iter()
                            .map(|m| Value::String(m.to_string()))
                            .collect(),
                    ),
                ),
            ];
            if let Some(types) = types {
                kvs.push((
                    "tokenizer.ggml.token_type".to_string(),
                    Value::Array(types.into_iter().map(Value::I32).collect()),
                ));
            }
            kvs
        };
        let bytes: Vec<String> = crate::unicode::byte_to_cpt_table()
            .into_iter()
            .map(|cp| char::from_u32(cp).unwrap().to_string())
            .collect();
        let contraction = header(
            bytes
                .iter()
                .cloned()
                .chain(["Å¿", "'Å¿", "'Å¿x"].map(String::from))
                .collect(),
            &["Å ¿", "' Å¿", "'Å¿ x"],
            None,
        );
        let sparse = header(["a", "b", "ab"].map(String::from).to_vec(), &["a b"], None);
        let twice = header(
            ["a", "b", "c", "ab", "bc"].map(String::from).to_vec(),
            &["b c", "a b", "b c"],
            None,
        );
        for (kvs, text, hf, legacy) in [
            (&contraction, "'ſx", &[257u32, 120][..], &[258][..]),
            (
                &contraction,
                "it's",
                &[105, 116, 39, 115],
                &[105, 116, 39, 115],
            ),
            (&sparse, "acb", &[2], &[0, 1]),
            (&sparse, "ab", &[2], &[2]),
            (&twice, "abc", &[3, 2], &[0, 4]),
        ] {
            let mut marked = kvs.clone();
            marked.push(stop.clone());
            let mut ggml = kvs.clone();
            ggml.push(("tokenizer.ggml.eog_token_ids".into(), stop.1.clone()));
            let mut both = marked.clone();
            both.push(("tokenizer.ggml.eog_token_ids".into(), stop.1.clone()));
            for (kvs, want) in [(kvs, legacy), (&ggml, legacy), (&marked, hf), (&both, hf)] {
                let tok = from_header(kvs).unwrap();
                assert_eq!(tok.encode(text, false, false), want, "{text:?}");
            }
        }
        // The BPE is Hugging Face's under the marker whatever the pre-tokenizer: minimax-m2 skips
        // the byte its vocabulary lacks and ranks a pair listed twice at its last place too, and
        // qwen2 folds `ſ` in a contraction as qwen35 does.
        for (pre, kvs, text, hf, legacy) in [
            ("minimax-m2", &sparse, "acb", &[2][..], &[0, 1][..]),
            ("minimax-m2", &twice, "abc", &[3, 2][..], &[0, 4][..]),
            ("qwen2", &sparse, "acb", &[2][..], &[0, 1][..]),
            ("qwen2", &twice, "abc", &[3, 2][..], &[0, 4][..]),
            ("qwen2", &contraction, "'ſx", &[257, 120][..], &[258][..]),
        ] {
            let mut named = kvs.clone();
            named[1].1 = Value::String(pre.into());
            let mut marked = named.clone();
            marked.push(stop.clone());
            for (kvs, want) in [(&named, legacy), (&marked, hf)] {
                let tok = from_header(kvs).unwrap();
                assert_eq!(tok.encode(text, false, false), want, "{pre} {text:?}");
            }
        }
        let mut off_vocab = header(["a", "b"].map(String::from).to_vec(), &["a b"], None);
        let mut added_byte = header(
            ["a", "b", "ab"].map(String::from).to_vec(),
            &["a b"],
            Some(vec![1, 4, 1]),
        );
        let mut glm4 = sparse.clone();
        glm4[1].1 = Value::String("glm4".into());
        let mut llama = sparse;
        llama[0].1 = Value::String("llama".into());
        for (kvs, want, unmarked) in [
            (&mut off_vocab, "tokenizer.ggml.merges[0] \"a b\"", true),
            (&mut added_byte, "tokenizer.ggml.tokens[1] \"b\"", true),
            (&mut glm4, "pre-tokenizer \"glm4\" has no reading", true),
            (&mut llama, "not SentencePiece", false),
        ] {
            // Without the marker the first three load; a SentencePiece one here has no byte tokens.
            assert!(!unmarked || from_header(kvs).is_ok(), "{want}: unmarked");
            kvs.push(stop.clone());
            let error = from_header(kvs).err().expect(want).to_string();
            assert!(error.contains(want), "{error}");
        }
    }

    #[test]
    fn gpt2_defaults_stay_in_vocab() {
        for n in [1, 11] {
            for (add_bos, add_eos, want) in [
                (
                    true,
                    false,
                    Some(Error::Missing("tokenizer.ggml.bos_token_id")),
                ),
                (
                    false,
                    true,
                    Some(Error::Missing("tokenizer.ggml.eos_token_id")),
                ),
                (false, false, None),
            ] {
                let kvs = [
                    ("tokenizer.ggml.model", Value::String("gpt2".into())),
                    ("tokenizer.ggml.pre", Value::String("qwen2".into())),
                    (
                        "tokenizer.ggml.tokens",
                        Value::Array(
                            (b'a'..)
                                .take(n)
                                .map(|b| Value::String(char::from(b).to_string()))
                                .collect(),
                        ),
                    ),
                    (
                        "tokenizer.ggml.merges",
                        Value::Array(vec![Value::String("a a".into())]),
                    ),
                    ("tokenizer.ggml.add_bos_token", Value::Bool(add_bos)),
                    ("tokenizer.ggml.add_eos_token", Value::Bool(add_eos)),
                ]
                .map(|(k, v)| (k.to_string(), v));
                let path = std::env::temp_dir().join(format!(
                    "tokenizer-gpt2-defaults-{}.gguf",
                    std::process::id()
                ));
                let layout = Layout::new(&kvs, Vec::new()).unwrap();
                let file = std::fs::File::create(&path).unwrap();
                Writer::new(file, layout).and_then(Writer::finish).unwrap();
                let got = Tokenizer::from_gguf(&path);
                std::fs::remove_file(&path).unwrap();
                if let Some(want) = want {
                    assert_eq!(got.err().map(|e| e.to_string()), Some(want.to_string()));
                } else {
                    let tok = got.unwrap();
                    assert_eq!(tok.specials().bos, None);
                    assert_eq!(tok.specials().eos, None);
                    assert!(tok.eog().is_empty());
                }
            }
        }
    }

    /// Every `<0xXX>` fallback must carry BYTE even when all 256 texts exist.
    #[test]
    fn gemma4_requires_byte_token_attributes() {
        for (token_type, byte) in [(None, 0), (Some(1), 0xBC), (Some(5), 0xBC)] {
            let texts: Vec<Value> = std::iter::once(Value::String("<bos>".into()))
                .chain((0..=u8::MAX).map(|b| Value::String(format!("<0x{b:02X}>"))))
                .collect();
            let mut types = vec![Value::I32(6); texts.len()];
            types[0] = Value::I32(3);
            let mut kvs: Vec<(String, Value)> = [
                ("tokenizer.ggml.model", Value::String("gemma4".into())),
                ("tokenizer.ggml.tokens", Value::Array(texts)),
                (
                    "tokenizer.ggml.merges",
                    Value::Array(vec![Value::String("<bos> <bos>".into())]),
                ),
                ("tokenizer.ggml.bos_token_id", Value::U32(0)),
            ]
            .map(|(k, v)| (k.to_string(), v))
            .into();
            if let Some(token_type) = token_type {
                types[usize::from(byte) + 1] = Value::I32(token_type);
                kvs.push(("tokenizer.ggml.token_type".into(), Value::Array(types)));
            }
            let path = std::env::temp_dir().join(format!(
                "tokenizer-gemma4-byte-attrs-{}.gguf",
                std::process::id()
            ));
            let layout = Layout::new(&kvs, Vec::new()).unwrap();
            let file = std::fs::File::create(&path).unwrap();
            Writer::new(file, layout).and_then(Writer::finish).unwrap();
            let got = Tokenizer::from_gguf(&path);
            std::fs::remove_file(&path).unwrap();
            assert_eq!(
                got.err().map(|e| e.to_string()),
                Some(Error::MissingByteToken(byte).to_string())
            );
        }
    }

    /// The gemma4 pipeline, one row per step the reference takes: a space is
    /// U+2581 and merges into the word after it, a word of newlines that is a
    /// token stays whole while one that is not merges as any word, symbols
    /// start as UTF-8 characters, a symbol that is no token falls back to
    /// `<0xXX>` tokens, USER_DEFINED tokens (and `<|channel|>`, which the
    /// reference makes one) are cut out without special parsing while
    /// CONTROL ones are not, and BOS is added though the file does not ask.
    /// Only U+0020 becomes U+2581, and before the text is decoded, so an
    /// overlong one (C0 A0) stays a space and falls back to `<0x20>`; a
    /// literal U+2581 reads as a space, so it does not round-trip; invalid
    /// UTF-8 is one U+FFFD per bad byte. A
    /// header that asks for a BOS or an EOS it cannot name is refused, as the
    /// reference asserts one, and so is a vocabulary missing a byte token,
    /// where the fallback would drop that byte.
    ///
    /// FAIL-first: without the U+2581 respelling "a b" is `a <0x20> b`; with
    /// the respelling after decoding, "a\xC0\xA0b" is `a ▁b`;
    /// without the newline-word lookup "\n\n\n" is three `\n`; with one symbol
    /// per byte "a b" falls back to the three bytes of U+2581; with the file's
    /// `add_bos_token` the BOS is missing; without the BOS-id, EOS-id or
    /// byte-token check that header loads.
    /// With a wrong two- or four-byte symbol length, "é" or "😀" falls back
    /// to bytes; with the file's `qwen2` pre-tokenizer, `pre()` is not gemma4.
    #[test]
    fn gemma4_encodes_as_the_reference() {
        let bos = ("tokenizer.ggml.bos_token_id", Value::U32(2));
        let eos = ("tokenizer.ggml.eos_token_id", Value::U32(1));
        let eos_out_of_range = ("tokenizer.ggml.eos_token_id", Value::U32(100_000));
        let add_eos = |on| ("tokenizer.ggml.add_eos_token", Value::Bool(on));
        let refused = [
            (
                vec![eos.clone(), add_eos(true)],
                None,
                Error::Missing("tokenizer.ggml.bos_token_id"),
            ),
            (
                vec![bos.clone(), add_eos(true)],
                None,
                Error::Missing("tokenizer.ggml.eos_token_id"),
            ),
            (
                vec![bos.clone(), eos_out_of_range, add_eos(true)],
                None,
                Error::Missing("tokenizer.ggml.eos_token_id"),
            ),
            (
                vec![bos.clone(), eos.clone(), add_eos(true)],
                Some(0xBC),
                Error::MissingByteToken(0xBC),
            ),
        ];
        for (header, missing_byte, want) in refused {
            let got = open(&header, missing_byte).err().map(|e| e.to_string());
            assert_eq!(got, Some(want.to_string()));
        }

        let tok = open(&[bos.clone(), add_eos(false)], None).unwrap();
        assert_eq!(tok.encode("a", true, true), ids(&tok, &["<bos>", "a"]));

        let tok = open(
            &[
                bos,
                eos,
                add_eos(true),
                ("tokenizer.ggml.pre", Value::String("qwen2".into())),
            ],
            None,
        )
        .unwrap();
        assert_eq!(tok.pre(), "gemma4");
        let fffd = ["<0xEF>", "<0xBF>", "<0xBD>"];
        let cases: &[(&[u8], bool, &[&str])] = &[
            (b"a b", true, &["a", "\u{2581}b"]),
            ("a\u{2581}b".as_bytes(), true, &["a", "\u{2581}b"]),
            (b"a\xC0\xA0b", true, &["a", "<0x20>", "b"]),
            (b"a\n\n\nb", true, &["a", "\n\n\n", "b"]),
            (b"\n\n", true, &["\n", "\n"]),
            (b"\n\n\n\n", true, &["\n", "\n", "\n", "\n"]),
            (b"a\r\n\n\nb", true, &["a", "<0x0D>", "\n\n\n", "b"]),
            (
                b"     ",
                true,
                &["\u{2581}\u{2581}\u{2581}\u{2581}", "\u{2581}"],
            ),
            ("한국".as_bytes(), true, &["한국"]),
            ("é".as_bytes(), true, &["é"]),
            ("😀".as_bytes(), true, &["😀"]),
            (b"xy", true, &["<0x78>", "<0x79>"]),
            ("ü".as_bytes(), true, &["<0xC3>", "<0xBC>"]),
            ("a\u{a0}b".as_bytes(), true, &["a", "<0xC2>", "<0xA0>", "b"]),
            (
                "a\u{3000}b".as_bytes(),
                true,
                &["a", "<0xE3>", "<0x80>", "<0x80>", "b"],
            ),
            (
                b"a\xF0\x9F b",
                true,
                &[
                    "a",
                    fffd[0],
                    fffd[1],
                    fffd[2],
                    fffd[0],
                    fffd[1],
                    fffd[2],
                    "\u{2581}b",
                ],
            ),
            (b"a<|channel|>b", false, &["a", "<|channel|>", "b"]),
            (b"<mark> b", false, &["<mark>", "\u{2581}b"]),
            (b"a <turn|>b", true, &["a", "\u{2581}", "<turn|>", "b"]),
            (
                b"a <turn|>b",
                false,
                &[
                    "a", "\u{2581}", "<0x3C>", "<0x74>", "<0x75>", "<0x72>", "<0x6E>", "<0x7C>",
                    "<0x3E>", "b",
                ],
            ),
        ];
        for &(text, parse_special, want) in cases {
            let got = tok.encode(text, false, parse_special);
            let text = String::from_utf8_lossy(text);
            assert_eq!(got, ids(&tok, want), "{text:?} {parse_special}");
        }
        assert_eq!(
            tok.encode("a", true, true),
            ids(&tok, &["<bos>", "a", "<eos>"])
        );

        let decode = |text: &[u8]| {
            let round = tok.decode_bytes(&tok.encode(text, true, true), false);
            String::from_utf8(round).unwrap()
        };
        assert_eq!(decode("a\u{2581}b".as_bytes()), "a b");
        assert_eq!(decode(b"a\xF0\x9F b"), "a\u{FFFD}\u{FFFD} b");
        let text = "a b\n\n\nü 한국";
        assert_eq!(decode(text.as_bytes()), text);
    }

    /// Hugging Face cuts its added tokens leftmost, then longest; the reference cuts the longest
    /// first. They differ where two special tokens share an edge and the later one is longer: in
    /// `]~b]!p~[`, bos `]~b]` and pad `]!p~[` share the `]` (HF cuts bos and leaves `!p~[`, the
    /// reference cuts pad and leaves `]~b`); a shorter later token (eos `[e~[` over pad's `[`) is
    /// cut alike. A vocabulary marked as HF's cuts as HF does, and one that is not keeps the
    /// reference's order. The ids are `tokenizers` 0.21.0's on the same tokens, no merges, bytes
    /// as ByteLevel spells them.
    ///
    /// FAIL-first: a marked vocabulary that cuts longest-first gives the reference's ids on the
    /// rows where they differ.
    #[test]
    fn the_hf_marker_cuts_added_tokens_leftmost_longest() {
        let header = |marked: bool| {
            let bytes = crate::unicode::byte_to_cpt_table()
                .into_iter()
                .map(|cp| char::from_u32(cp).unwrap().to_string());
            let tokens: Vec<Value> = bytes
                .chain(["]!p~[", "]~b]", "[e~["].map(String::from))
                .map(Value::String)
                .collect();
            let types: Vec<Value> = (0..tokens.len())
                .map(|i| Value::I32(if i < 256 { 1 } else { 3 }))
                .collect();
            let mut kvs = vec![
                (
                    "tokenizer.ggml.model".to_string(),
                    Value::String("gpt2".into()),
                ),
                (
                    "tokenizer.ggml.pre".to_string(),
                    Value::String("minimax-m2".into()),
                ),
                ("tokenizer.ggml.tokens".to_string(), Value::Array(tokens)),
                ("tokenizer.ggml.token_type".to_string(), Value::Array(types)),
                ("tokenizer.ggml.merges".to_string(), Value::Array(vec![])),
            ];
            if marked {
                kvs.push((
                    "tokenizer.bloomery.eog_token_ids".to_string(),
                    Value::Array(vec![Value::U32(258)]),
                ));
            }
            kvs
        };
        let (hf, reference) = (
            from_header(&header(true)).unwrap(),
            from_header(&header(false)).unwrap(),
        );
        // text, HF's ids, the reference's.
        let cases: &[(&str, &[u32], &[u32])] = &[
            ("]~b]!p~[", &[257, 33, 112, 126, 91], &[93, 126, 98, 256]),
            (
                "x]~b]!p~[y",
                &[120, 257, 33, 112, 126, 91, 121],
                &[120, 93, 126, 98, 256, 121],
            ),
            (
                "]~b]!p~[e~[",
                &[257, 33, 112, 126, 258],
                &[93, 126, 98, 256, 101, 126, 91],
            ),
            ("]!p~[e~[", &[256, 101, 126, 91], &[256, 101, 126, 91]),
            ("]~b]]!p~[", &[257, 256], &[257, 256]),
            ("[e~[]~b]", &[258, 257], &[258, 257]),
        ];
        for &(text, want_hf, want_reference) in cases {
            assert_eq!(hf.encode(text, false, true), want_hf, "hf {text:?}");
            assert_eq!(
                reference.encode(text, false, true),
                want_reference,
                "reference {text:?}"
            );
        }
        // Without `parse_special` a control token is no cut, in either order.
        assert_eq!(hf.encode("]~b]", false, false), [93, 126, 98, 93]);
        // A cut inside the leftmost token leaves it no match, and the next one is cut.
        assert_eq!(
            hf.encode_marked("]\u{1}~b]!p~[", "\u{1}", false, true),
            [93, 126, 98, 256]
        );
    }

    /// A byte-alphabet vocabulary under `pre`: every byte is a token, so a text decodes to its
    /// own bytes and encodes to one byte token a byte.
    fn byte_header(pre: &str) -> Vec<(String, Value)> {
        let tokens: Vec<Value> = byte_to_cpt_table()
            .into_iter()
            .map(|cp| Value::String(char::from_u32(cp).unwrap().to_string()))
            .collect();
        [
            ("tokenizer.ggml.model", Value::String("gpt2".into())),
            ("tokenizer.ggml.pre", Value::String(pre.into())),
            ("tokenizer.ggml.tokens", Value::Array(tokens)),
            ("tokenizer.ggml.merges", Value::Array(vec![])),
        ]
        .map(|(k, v)| (k.to_string(), v))
        .to_vec()
    }

    /// The `llama-bpe` row cleans decoded spaces as the reference's `clean_spaces` does (its
    /// BPE default, kept for the llama3 type): the space before `?`, `!`, `.` and `,` is
    /// dropped — one of two, the one before the punctuation — an apostrophe between spaces
    /// keeps neither, and the space before `'s`, `'m`, `'re` and `'ve` is dropped while `'t`,
    /// `'d` and `'ll` keep theirs. A row that does not clean (`qwen35`) returns the text as is.
    ///
    /// FAIL-first: with the clean dropped every changed row fails; with one pass alone, the
    /// rows the other passes change.
    #[test]
    fn the_llama_bpe_row_cleans_decoded_spaces() {
        let cases: &[(&str, &str)] = &[
            (" ?", "?"),
            ("What ?", "What?"),
            ("Hello , world", "Hello, world"),
            ("a  ?", "a ?"),
            ("Hello ,  world", "Hello,  world"),
            ("a ' b", "a'b"),
            ("it 's fine", "it's fine"),
            ("I 'm here", "I'm here"),
            ("they 're", "they're"),
            ("we 've", "we've"),
            ("don 't", "don 't"),
            ("we 'll go", "we 'll go"),
            ("you 'd", "you 'd"),
            ("a 'x b", "a 'x b"),
            ("x' y", "x' y"),
            ("a ' b ' c", "a'b'c"),
            (" ' s", "'s"),
            (" '? t", " '? t"),
            ("a '", "a '"),
            ("xx ' ?", "xx '?"),
            ("x  '  ", "x ' "),
            ("x  ' s", "x's"),
            ("한 , 글", "한, 글"),
            ("no change", "no change"),
            ("", ""),
        ];
        let cleaned = from_header(&byte_header("llama-bpe")).unwrap();
        let raw = from_header(&byte_header("qwen35")).unwrap();
        assert!(cleaned.space_cleaner().is_some());
        assert!(raw.space_cleaner().is_none());
        for &(text, want) in cases {
            let ids = cleaned.encode(text, false, false);
            assert_eq!(&cleaned.decode(&ids), want, "{text:?}");
            assert_eq!(
                &raw.decode(&raw.encode(text, false, false)),
                text,
                "{text:?}"
            );
        }
    }

    /// llama.cpp's three passes over one whole text (llama-vocab.cpp, `detokenize`'s
    /// `clean_spaces`), as it writes them: the space before `?`, `!`, `.` and `,` dropped; an
    /// apostrophe between two spaces keeping neither, its right space marked so a later
    /// apostrophe does not read it as its left; the space before `'s`, `'m`, `'re` and `'ve`
    /// dropped, where `'t`, `'d` and `'ll` keep theirs. The sweep's oracle, independent of
    /// the cleaner it judges.
    fn reference_clean(text: &str) -> String {
        let t = text.as_bytes();
        let first = {
            let mut out = Vec::with_capacity(t.len());
            if !t.is_empty() {
                out.push(t[0]);
                for i in 1..t.len() {
                    if t[i - 1] == b' ' && matches!(t[i], b'?' | b'!' | b'.' | b',') {
                        out.pop();
                    }
                    out.push(t[i]);
                }
            }
            out
        };
        let second = {
            let mut mark = first;
            let mut out = Vec::with_capacity(mark.len());
            if !mark.is_empty() {
                out.push(mark[0]);
                let mut i = 1;
                while i < mark.len() {
                    let x = mark[i];
                    if x == b'\''
                        && i + 1 < mark.len()
                        && mark[i - 1] == b' '
                        && mark[i + 1] == b' '
                    {
                        out.pop();
                        mark[i + 1] = 0;
                        out.push(x);
                        i += 2;
                        continue;
                    }
                    out.push(x);
                    i += 1;
                }
            }
            out
        };
        let t = &second;
        let mut out = Vec::with_capacity(t.len());
        if !t.is_empty() {
            out.push(t[0]);
            for i in 1..t.len() {
                if t[i - 1] == b' '
                    && t[i] == b'\''
                    && i + 1 < t.len()
                    && (matches!(t[i + 1], b's' | b'm')
                        || i + 2 < t.len()
                            && matches!((t[i + 1], t[i + 2]), (b'r', b'e') | (b'v', b'e')))
                {
                    out.pop();
                }
                out.push(t[i]);
            }
        }
        String::from_utf8(out).expect("the passes drop ASCII bytes only")
    }

    /// Cleaning a stream piece by piece gives the whole text's clean. [`SpaceCleaner`] is swept
    /// over every string of the passes' alphabet up to length 7 (`x`, space, `?`, `'`, `s`, and
    /// with `r` `e` for `'re`) at every two-piece cut, then over deterministic pseudo-random
    /// longer strings of the whole alphabet at random cuts, beside the hand-derived cases and
    /// [`Decoder`] fed one byte token at a time, whose pieces cut characters apart and the
    /// decoder holds their halves before the cleaner sees text.
    ///
    /// FAIL-first: a hold that can re-fire a pattern the whole clean already fired (holding a
    /// fragment that starts inside a cleaned pattern) fails the rows that clean twice, first of
    /// all `a  ?` cut at 0.
    #[test]
    fn space_cleaning_streams_as_a_whole() {
        let cases: &[(&str, &str)] = &[
            (" ?", "?"),
            ("Hello , world", "Hello, world"),
            ("a  ?", "a ?"),
            ("a ' b", "a'b"),
            ("it 's fine", "it's fine"),
            ("they 're", "they're"),
            ("we 'll go", "we 'll go"),
            ("a ' b ' c", "a'b'c"),
            ("xx ' ?", "xx '?"),
            ("x  '  ", "x ' "),
            ("x  ' s", "x's"),
            ("한 , 글", "한, 글"),
            ("I am a cat. I am a dog.", "I am a cat. I am a dog."),
            ("I am a cat . I am a dog .", "I am a cat. I am a dog."),
            ("<|start_header_id|>", "<|start_header_id|>"),
            ("한 's", "한's"),
            ("é 'm x", "é'm x"),
            ("글 ' ?", "글 '?"),
        ];
        let streams = |text: &str, cuts: &[usize]| -> String {
            let mut cleaner = SpaceCleaner::default();
            let mut out = String::new();
            let mut at = 0;
            for &cut in cuts {
                let end = text.len().min(cut);
                if text.is_char_boundary(end) && end >= at {
                    out.push_str(&cleaner.push(&text[at..end]));
                    at = end;
                }
            }
            out.push_str(&cleaner.push(&text[at..]));
            out.push_str(&cleaner.finish());
            out
        };
        for &(text, want) in cases {
            for cut in 0..=text.len() {
                if text.is_char_boundary(cut) {
                    assert_eq!(streams(text, &[cut]), want, "{text:?} cut at {cut}");
                }
            }
        }
        // Every short string of the passes' alphabet, every cut: one set with the contraction
        // letter the third pass drops (`'s`), one with the pair it drops (`'re`).
        let mut words = Vec::new();
        fn walk(prefix: &[u8], alphabet: &[u8], left: usize, out: &mut Vec<String>) {
            if !prefix.is_empty() {
                out.push(String::from_utf8(prefix.to_vec()).unwrap());
            }
            if left == 0 {
                return;
            }
            for &b in alphabet {
                let mut next = prefix.to_vec();
                next.push(b);
                walk(&next, alphabet, left - 1, out);
            }
        }
        for alphabet in [b"x ?'s".to_vec(), b"x ?'re".to_vec()] {
            walk(&[], &alphabet, 7, &mut words);
        }
        // A deterministic pseudo-random tail of longer strings.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..20_000 {
            let len = 8 + (next() % 32) as usize;
            let text: String = (0..len)
                .map(|_| b"x ?'sremvtdl,.!"[next() as usize % 14] as char)
                .collect();
            words.push(text);
        }
        let mut failures = Vec::new();
        for text in &words {
            let want = reference_clean(text);
            for cut in 0..=text.len() {
                if !text.is_char_boundary(cut) {
                    continue;
                }
                let got = streams(text, &[cut]);
                if got != want {
                    failures.push(format!("{text:?} cut at {cut}: {got:?} != {want:?}"));
                }
            }
            let second = text.len() / 3;
            let third = 2 * text.len() / 3;
            if text.is_char_boundary(second) && text.is_char_boundary(third) {
                let got = streams(text, &[second, third]);
                if got != want {
                    failures.push(format!("{text:?}: {got:?} != {want:?}"));
                }
            }
        }
        assert!(
            failures.is_empty(),
            "{} of {}:\n{}",
            failures.len(),
            words.len(),
            failures
                .iter()
                .take(20)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
        let tok = from_header(&byte_header("llama-bpe")).unwrap();
        for &(text, want) in cases {
            let ids = tok.encode(text, false, false);
            let mut decoder = Decoder::new(&tok, true);
            let mut out = String::new();
            for &id in &ids {
                if let Some(s) = decoder.push(id) {
                    out.push_str(s);
                }
            }
            if let Some(rest) = decoder.finish() {
                out.push_str(&rest);
            }
            assert_eq!(out, want, "{text:?}");
        }
    }

    /// A push releases every byte no later piece can change: only a proper prefix of a
    /// cleaning pattern (` `, ` '`, ` 'r`) is held for the piece that completes it, so a
    /// text that ends in no pattern's prefix is out whole in its push and `finish` returns
    /// nothing.
    ///
    /// FAIL-first: a cleaner that holds text behind a longer window releases none of these
    /// rows whole, and holds a trailing space's text past its own push.
    #[test]
    fn a_push_releases_what_no_pattern_can_still_start() {
        for text in [
            "I am a cat. I am a dog.",
            "<|start_header_id|>",
            "Hello , world",
            "a ' b",
            "한 , 글",
        ] {
            let mut cleaner = SpaceCleaner::default();
            let out = cleaner.push(text);
            assert_eq!(&out, &reference_clean(text), "{text:?}");
            assert_eq!(cleaner.finish(), "", "{text:?}");
        }
        // What a hold keeps is the pattern's own prefix, and the piece that completes the
        // pattern takes it.
        let mut cleaner = SpaceCleaner::default();
        assert_eq!(cleaner.push("it '"), "it");
        assert_eq!(cleaner.push("s here"), "'s here");
        assert_eq!(cleaner.finish(), "");
        let mut cleaner = SpaceCleaner::default();
        assert_eq!(cleaner.push("x "), "x");
        assert_eq!(cleaner.finish(), " ");
    }

    /// The `llama-bpe` row defaults an absent `tokenizer.ggml.add_bos_token` to true, as the
    /// reference's llama3 pre type does, and the file's key still overrides it; `qwen2`, a
    /// row of no such default, keeps adding nothing.
    ///
    /// FAIL-first: the false default omitted the BOS a llama-bpe gguf with no key adds.
    #[test]
    fn llama_bpe_defaults_to_adding_bos() {
        let tok = from_header(&byte_header("llama-bpe")).unwrap();
        assert!(tok.add_bos());
        let Some(bos) = tok.specials().bos else {
            panic!("no bos id");
        };
        assert_eq!(tok.encode("x", true, true).first(), Some(&bos));
        let mut header = byte_header("llama-bpe");
        header.push((
            "tokenizer.ggml.add_bos_token".to_string(),
            Value::Bool(false),
        ));
        assert!(!from_header(&header).unwrap().add_bos());
        let raw = from_header(&byte_header("qwen2")).unwrap();
        assert!(!raw.add_bos());
        assert_eq!(raw.encode("x", true, true), raw.encode("x", false, false));
    }

    /// llama.cpp's own tokenizer test data (`ggml-vocab-llama-bpe`): every text of the `.inp`
    /// file, split on its `__ggml_vocab_test__` separator, tokenizes to exactly the ids of its
    /// `.out` line, with no BOS added and no special token parsed (`add_special = false` in its
    /// test-tokenizer-0). The gguf names no `add_bos_token` key, so this loader defaults it
    /// true as llama.cpp's pre type does; the vectors add no special token either way.
    ///
    /// FAIL-first: a `llama-bpe` name off the pre-tokenizer table, or the vocabulary the gguf
    /// spells unrun, fails the load before the first row.
    #[test]
    #[ignore = "needs llama.cpp's ggml-vocab-llama-bpe.gguf (its .inp and .out beside it) at $BLOOMERY_LLAMA_BPE_GGUF"]
    fn hw_llama_bpe_encodes_as_llama_tokenizer() {
        let path = std::path::PathBuf::from(std::env::var_os("BLOOMERY_LLAMA_BPE_GGUF").unwrap());
        let tok = Tokenizer::from_gguf(&path).unwrap();
        assert_eq!(tok.pre(), "llama-bpe");
        let inp = std::fs::read(format!("{}.inp", path.display())).unwrap();
        let out = std::fs::read_to_string(format!("{}.out", path.display())).unwrap();
        const SEP: &[u8] = b"\n__ggml_vocab_test__\n";
        let mut texts: Vec<&[u8]> = Vec::new();
        let mut at = 0;
        while let Some(found) = inp[at..]
            .windows(SEP.len())
            .position(|w| w == SEP)
            .map(|p| at + p)
        {
            texts.push(&inp[at..found]);
            at = found + SEP.len();
        }
        if at < inp.len() {
            texts.push(&inp[at..]);
        }
        let outs: Vec<&str> = out.lines().collect();
        assert_eq!(texts.len(), outs.len(), "a test a line");
        let mut failures = Vec::new();
        for (text, ids) in texts.iter().zip(&outs) {
            let want: Vec<u32> = ids
                .split_whitespace()
                .map(|t| t.parse::<u32>().unwrap())
                .collect();
            let got = tok.encode(text, false, false);
            if got != want {
                failures.push(format!("{text:?}: {got:?} != {want:?}"));
            }
        }
        assert!(
            failures.is_empty(),
            "{} of {}:\n{}",
            failures.len(),
            texts.len(),
            failures
                .iter()
                .take(10)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}

#[cfg(test)]
mod guard_tests;
#[cfg(test)]
mod spm_tests;
