//! Ids back to text, `llama_token_to_piece` semantics.

use crate::Tokenizer;
use crate::unicode::{byte_to_cpt_table, decode_lenient, encode_cpt};
use crate::vocab::attr;

/// Each token's piece with special tokens rendered (the reference's
/// `cache_token_to_piece`):
///
/// * CONTROL, USER_DEFINED, UNKNOWN: the token text as written;
/// * NORMAL: the text mapped back through the GPT-2 byte alphabet, where a
///   character outside the alphabet becomes the reference's
///   `[UNK_BYTE_0x<hex><text>]` marker;
/// * BYTE: the byte its `<0xXX>` text names;
/// * anything else: empty.
pub(crate) fn pieces(texts: &[String], attrs: &[u32]) -> Vec<Box<[u8]>> {
    let mut cpt_to_byte = [None::<u8>; 324];
    for (b, &cpt) in byte_to_cpt_table().iter().enumerate() {
        cpt_to_byte[cpt as usize] = Some(b as u8);
    }
    let mut cpts = Vec::new();
    let mut utf8 = Vec::new();
    texts
        .iter()
        .zip(attrs)
        .map(|(text, &a)| {
            if a & (attr::CONTROL | attr::USER_DEFINED | attr::UNKNOWN) != 0 {
                return text.as_bytes().into();
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

/// A streaming decoder: feed ids one at a time and take back the text that is
/// complete so far. An incomplete UTF-8 sequence at the end is held until the
/// next piece completes it; a sequence that can never complete becomes
/// U+FFFD.
pub struct Decoder<'t> {
    tok: &'t Tokenizer,
    special: bool,
    pending: Vec<u8>,
    out: String,
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
        (!self.out.is_empty()).then_some(self.out.as_str())
    }

    /// End of stream: whatever is still held, lossily.
    pub fn finish(&mut self) -> Option<String> {
        let rest = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        (!rest.is_empty()).then_some(rest)
    }
}
