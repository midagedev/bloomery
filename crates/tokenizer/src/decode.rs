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
/// `'ve` dropped, where `'t`, `'d` and `'ll` keep theirs. Each pass is the left-to-right
/// replacement of its patterns below, the passes run in the reference's order, each over
/// what the last kept. Every pattern is ASCII and starts with a space, so no pass splits a
/// character: a held prefix is ASCII and the released text stays valid UTF-8.
const PATTERNS: [&[(&[u8], &[u8])]; 3] = [
    &[(b" ?", b"?"), (b" !", b"!"), (b" .", b"."), (b" ,", b",")],
    &[(b" ' ", b"'")],
    &[
        (b" 's", b"'s"),
        (b" 'm", b"'m"),
        (b" 're", b"'re"),
        (b" 've", b"'ve"),
    ],
];

/// The three passes over one whole text.
pub(crate) fn clean_str(text: &str) -> String {
    let mut cleaner = SpaceCleaner::default();
    let mut out = cleaner.push(text);
    out.push_str(&cleaner.finish());
    out
}

/// The three passes over a growing byte stream: each [`SpaceCleaner::push`] returns the text
/// that is final. Each pass holds only what a later piece could still complete into one of
/// its patterns — a proper prefix of one — and releases every other byte in the push that
/// carries it, so the output over any split of the input into pushes is the whole text's
/// clean: a byte is released only once no pattern of its pass can match at its place, which
/// no later piece changes. The exhaustive sweep of `space_cleaning_streams_as_a_whole`
/// covers the cut.
#[derive(Default)]
pub struct SpaceCleaner {
    /// Each pass's bytes not yet final: a proper prefix of one of its patterns.
    pending: [Vec<u8>; 3],
}

impl SpaceCleaner {
    /// The text of `text` that no later piece can change.
    pub fn push(&mut self, text: &str) -> String {
        self.feed(Some(text.as_bytes()))
    }

    /// The text still held, cleaned.
    pub fn finish(&mut self) -> String {
        self.feed(None)
    }

    /// One piece through the passes: each cleans its pending with the piece appended, keeps
    /// what a later piece could still change, and hands what it released to the next pass.
    /// `None` is the end of the stream: nothing is held.
    fn feed(&mut self, text: Option<&[u8]>) -> String {
        if let Some(t) = text {
            self.pending[0].extend_from_slice(t);
        }
        let last = PATTERNS.len() - 1;
        for i in 0..last {
            let (head, tail) = self.pending.split_at_mut(i + 1);
            let released = clean_pass(&mut head[i], PATTERNS[i], text.is_some());
            tail[0].extend_from_slice(&head[i][..released]);
            let kept = head[i].len() - released;
            head[i].copy_within(released.., 0);
            head[i].truncate(kept);
        }
        let released = clean_pass(&mut self.pending[last], PATTERNS[last], text.is_some());
        let out = self.pending[last].drain(..released).collect();
        String::from_utf8(out).expect("only ASCII pattern prefixes are held")
    }
}

/// One pass over `pending` in place: each match's replacement written over it, the
/// released bytes compacted to the front, the held tail behind them. Returns how many
/// bytes the pass released; `pending` keeps the rest. A replacement is never longer
/// than its pattern, so the write stays behind the read. `stream` is false only at the
/// end of the stream, where nothing is held.
fn clean_pass(pending: &mut Vec<u8>, patterns: &[(&[u8], &[u8])], stream: bool) -> usize {
    let (mut read, mut write) = (0, 0);
    while read < pending.len() {
        match patterns
            .iter()
            .find(|&&(from, _)| pending[read..].starts_with(from))
        {
            Some(&(from, to)) => {
                pending[write..write + to.len()].copy_from_slice(to);
                write += to.len();
                read += from.len();
            }
            None => {
                if stream
                    && patterns
                        .iter()
                        .any(|(from, _)| from.starts_with(&pending[read..]))
                {
                    break;
                }
                pending[write] = pending[read];
                write += 1;
                read += 1;
            }
        }
    }
    pending.copy_within(read.., write);
    pending.truncate(write + pending.len() - read);
    write
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
