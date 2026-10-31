//! Ids back to text, `llama_token_to_piece` semantics.

use crate::spelling::Spelling;
use crate::unicode::{byte_to_cpt_table, decode_lenient, encode_cpt};
use crate::vocab::attr;
use crate::{Error, Tokenizer};

fn byte_decoder() -> [Option<u8>; 324] {
    let mut table = [None; 324];
    for (b, &cpt) in byte_to_cpt_table().iter().enumerate() {
        table[cpt as usize] = Some(b as u8);
    }
    table
}

/// HF ByteLevel maps the whole spelling through its byte alphabet, or preserves the whole
/// spelling when any character is outside it. Unused ids are fillers with no HF spelling.
pub(crate) fn check_hf(texts: &[String], attrs: &[u32], pieces: &[Box<[u8]>]) -> Result<(), Error> {
    let table = byte_decoder();
    let mut bytes = Vec::new();
    for (id, text) in texts.iter().enumerate() {
        if attrs[id] & attr::UNUSED != 0 {
            continue;
        }
        bytes.clear();
        let all_bytes = text.chars().all(|c| {
            if let Some(b) = table.get(c as usize).copied().flatten() {
                bytes.push(b);
                true
            } else {
                false
            }
        });
        let expected = if all_bytes { &bytes } else { text.as_bytes() };
        if pieces[id].as_ref() != expected {
            return Err(Error::InvalidHfToken {
                id,
                text: text.clone(),
            });
        }
    }
    Ok(())
}

/// Each token's piece with special tokens rendered (the reference's
/// `cache_token_to_piece`):
///
/// * CONTROL, USER_DEFINED, UNKNOWN: the token text as written;
/// * NORMAL, byte-level: the text mapped back through the GPT-2 byte
///   alphabet, where a character outside the alphabet becomes the
///   reference's `[UNK_BYTE_0x<hex><text>]` marker;
/// * NORMAL, metaspace: the text with each U+2581 written as a space;
/// * BYTE: the byte its `<0xXX>` text names;
/// * anything else: empty.
pub(crate) fn pieces(texts: &[String], attrs: &[u32], spelling: Spelling) -> Vec<Box<[u8]>> {
    let cpt_to_byte = byte_decoder();
    let mut cpts = Vec::new();
    let mut utf8 = Vec::new();
    texts
        .iter()
        .zip(attrs)
        .map(|(text, &a)| {
            if a & (attr::CONTROL | attr::USER_DEFINED | attr::UNKNOWN) != 0 {
                return text.as_bytes().into();
            }
            if a & attr::NORMAL != 0 && spelling == Spelling::Metaspace {
                return text.replace('\u{2581}', " ").into_bytes().into();
            }
            if a & attr::NORMAL != 0 {
                let mut out = Vec::with_capacity(text.len());
                decode_lenient(text.as_bytes(), &mut cpts);
                for &cpt in &cpts {
                    match cpt_to_byte.get(cpt as usize).copied().flatten() {
                        Some(b) => out.push(b),
                        None => {
                            utf8.clear();
                            encode_cpt(cpt, &mut utf8);
                            out.extend_from_slice(b"[UNK_BYTE_0x");
                            for b in &utf8 {
                                out.extend_from_slice(format!("{b:02x}").as_bytes());
                            }
                            out.extend_from_slice(text.as_bytes());
                            out.push(b']');
                        }
                    }
                }
                return out.into();
            }
            if a & attr::BYTE != 0 {
                return vec![byte_of(text)].into();
            }
            Box::default()
        })
        .collect()
}

/// `strtol(text.substr(3, 2), 16)` as a byte: the hex digits of `<0xXX>`.
fn byte_of(text: &str) -> u8 {
    let digits = text.as_bytes().get(3..).unwrap_or_default();
    let mut v: u32 = 0;
    for &d in digits.iter().take(2) {
        match (d as char).to_digit(16) {
            Some(x) => v = v * 16 + x,
            None => break,
        }
    }
    // Two hex digits fit a byte.
    v as u8
}

/// The reference's `clean_spaces` detokenization (llama-vocab.cpp, its BPE default kept for
/// the llama3 pre type), as its three passes: the space before `?`, `!`, `.` and `,` dropped;
/// an apostrophe between two spaces keeping neither; the space before `'s`, `'m`, `'re` and
/// `'ve` dropped, where `'t`, `'d` and `'ll` keep theirs. The passes only drop ASCII bytes, so
/// valid UTF-8 stays valid. Each pass maps `(bytes, src)` to the bytes it keeps with every kept
/// byte traced to its place in the text the pipeline started from.
fn pass_dropped_spaces(bytes: &[u8], src: &[u32]) -> (Vec<u8>, Vec<u32>) {
    let (mut out, mut osrc) = (
        Vec::with_capacity(bytes.len()),
        Vec::with_capacity(bytes.len()),
    );
    for (i, &x) in bytes.iter().enumerate() {
        if i > 0
            && bytes[i - 1] == b' '
            && matches!(x, b'?' | b'!' | b'.' | b',')
            && out.pop().is_some()
        {
            osrc.pop();
        }
        out.push(x);
        osrc.push(src[i]);
    }
    (out, osrc)
}

/// The second pass: ` ' ` becomes `'`. The right space is marked so a later apostrophe does not
/// read it as its left one, and skipped.
fn pass_apostrophe_between_spaces(bytes: &[u8], src: &[u32]) -> (Vec<u8>, Vec<u32>) {
    let mut mark = bytes.to_vec();
    let (mut out, mut osrc) = (
        Vec::with_capacity(bytes.len()),
        Vec::with_capacity(bytes.len()),
    );
    let mut i = 0;
    while i < mark.len() {
        let x = mark[i];
        if i > 0
            && x == b'\''
            && i + 1 < mark.len()
            && mark[i - 1] == b' '
            && mark[i + 1] == b' '
            && out.pop().is_some()
        {
            osrc.pop();
            mark[i + 1] = 0;
            out.push(x);
            osrc.push(src[i]);
            i += 2;
            continue;
        }
        out.push(x);
        osrc.push(src[i]);
        i += 1;
    }
    (out, osrc)
}

/// The third pass: ` 's`, ` 'm`, ` 're` and ` 've` lose their space; ` 't`, ` 'd` and ` 'll`
/// keep theirs.
fn pass_contraction_spaces(bytes: &[u8], src: &[u32]) -> (Vec<u8>, Vec<u32>) {
    let (mut out, mut osrc) = (
        Vec::with_capacity(bytes.len()),
        Vec::with_capacity(bytes.len()),
    );
    for (i, &x) in bytes.iter().enumerate() {
        if i > 0
            && bytes[i - 1] == b' '
            && x == b'\''
            && i + 1 < bytes.len()
            && (matches!(bytes[i + 1], b's' | b'm')
                || i + 2 < bytes.len()
                    && matches!((bytes[i + 1], bytes[i + 2]), (b'r', b'e') | (b'v', b'e')))
            && out.pop().is_some()
        {
            osrc.pop();
        }
        out.push(x);
        osrc.push(src[i]);
    }
    (out, osrc)
}

/// The three passes over `text`, every kept byte traced to its place in it.
pub(crate) fn clean_traced(text: &[u8]) -> (Vec<u8>, Vec<u32>) {
    let src: Vec<u32> = (0..text.len() as u32).collect();
    let (bytes, src) = pass_dropped_spaces(text, &src);
    let (bytes, src) = pass_apostrophe_between_spaces(&bytes, &src);
    pass_contraction_spaces(&bytes, &src)
}

/// The three passes over one whole text.
pub(crate) fn clean_str(text: &str) -> String {
    let (bytes, _) = clean_traced(text.as_bytes());
    String::from_utf8(bytes).expect("dropping ASCII bytes leaves valid UTF-8")
}

/// The three passes over a growing byte stream: each [`SpaceCleaner::push`] returns the text
/// that is final. The passes are not compositional — the first pass dropping the space before
/// a `?` the pieces after the cut carry is what keeps the second from seeing the ` ' ` that
/// space closed, and the second's skipped right space depends on an apostrophe left of any
/// cut through it — so a cut is made only through a quiet zone, five bytes no rule reads with
/// three more beyond for a cause the next pieces carry: no rule's window spans the cut,
/// whatever follows. Each push cleans the bytes it holds as one whole, from the raw, emits
/// what the cut's left side produced (traced by the passes) and keeps the rest raw, to be
/// cleaned with the next piece. The exhaustive sweep of `space_cleaning_streams_as_a_whole`
/// covers the cut.
#[derive(Default)]
pub struct SpaceCleaner {
    raw: Vec<u8>,
}

/// A byte no rule reads: not the space every rule drops, not the apostrophe the last two
/// center on, not the punctuation byte the first centers on.
fn quiet_byte(b: u8) -> bool {
    b != b' ' && b != b'\'' && !matches!(b, b'?' | b'!' | b'.' | b',')
}

impl SpaceCleaner {
    /// The text of `text` that no later piece can change.
    pub fn push(&mut self, text: &str) -> String {
        self.raw.extend_from_slice(text.as_bytes());
        // The last cut whose five bytes — the two before it, its own, the two after — are
        // quiet, at a character's start, with three bytes left beyond it.
        let raw = &self.raw;
        let quiet = |r: usize| {
            r >= 2
                && raw.len() >= r + 3
                && raw[r] & 0xC0 != 0x80
                && raw[r - 2..r + 3].iter().all(|&b| quiet_byte(b))
        };
        let Some(cut) = (2..raw.len().saturating_sub(2)).rev().find(|&r| quiet(r)) else {
            return String::new();
        };
        let (bytes, src) = clean_traced(&self.raw);
        let stable = src
            .iter()
            .rposition(|&s| (s as usize) < cut)
            .map_or(0, |at| at + 1);
        let out = String::from_utf8(bytes[..stable].to_vec())
            .expect("dropping ASCII bytes leaves valid UTF-8");
        self.raw.drain(..cut);
        out
    }

    /// The text still held, cleaned.
    pub fn finish(&mut self) -> String {
        let (bytes, _) = clean_traced(&self.raw);
        self.raw.clear();
        String::from_utf8(bytes).expect("dropping ASCII bytes leaves valid UTF-8")
    }
}

/// A streaming decoder: feed ids one at a time and take back the text that is
/// complete so far. An incomplete UTF-8 sequence at the end is held until the
/// next piece completes it; a sequence that can never complete becomes
/// U+FFFD. A vocabulary that cleans spaces ([`Tokenizer::space_cleaner`])
/// cleans its text as [`SpaceCleaner`] does.
pub struct Decoder<'t> {
    tok: &'t Tokenizer,
    special: bool,
    pending: Vec<u8>,
    out: String,
    cleaner: Option<SpaceCleaner>,
}

impl<'t> Decoder<'t> {
    /// A decoder over `tok`; `special` renders CONTROL and UNKNOWN tokens as
    /// their text instead of nothing.
    pub fn new(tok: &'t Tokenizer, special: bool) -> Decoder<'t> {
        Decoder {
            tok,
            special,
            pending: Vec::new(),
            out: String::new(),
            cleaner: tok.space_cleaner(),
        }
    }

    /// Add one token; returns the text it completed, if any.
    pub fn push(&mut self, id: u32) -> Option<&str> {
        self.pending
            .extend_from_slice(self.tok.piece(id, self.special));
        self.out.clear();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    self.out.push_str(s);
                    self.pending.clear();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    // The first `valid` bytes were just checked.
                    self.out
                        .push_str(std::str::from_utf8(&self.pending[..valid]).unwrap_or_default());
                    match e.error_len() {
                        None => {
                            self.pending.drain(..valid);
                            break;
                        }
                        Some(bad) => {
                            self.out.push(char::REPLACEMENT_CHARACTER);
                            self.pending.drain(..valid + bad);
                        }
                    }
                }
            }
        }
        if let Some(cleaner) = &mut self.cleaner {
            let out = std::mem::take(&mut self.out);
            self.out = cleaner.push(&out);
        }
        (!self.out.is_empty()).then_some(self.out.as_str())
    }

    /// End of stream: whatever is still held, lossily.
    pub fn finish(&mut self) -> Option<String> {
        let mut rest = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        if let Some(cleaner) = &mut self.cleaner {
            rest = cleaner.push(&rest);
            rest.push_str(&cleaner.finish());
        }
        (!rest.is_empty()).then_some(rest)
    }
}
