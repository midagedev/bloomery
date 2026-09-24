//! The slot file: what `POST /slots/0?action=save` writes and `restore` reads.
//!
//! Little-endian throughout:
//!
//! | bytes | field |
//! |---|---|
//! | 8 | [`MAGIC`] |
//! | 4 | [`VERSION`] (u32) |
//! | 4 | `n_vocab` of the server that wrote it (u32) |
//! | 8 | `n_ids`, the slot's positions (u64) |
//! | 4 · `n_ids` | the slot's ids, one u32 per position |
//! | rest | the engine's state ([`crate::Engine::save_state`]), to the end of the file |
//!
//! llama.cpp's sequence file has the same shape (magic, version, the token
//! count, the tokens, then the context state to the end). A file whose magic,
//! version or vocabulary differs, whose count passes the context, or whose id
//! passes the vocabulary is refused by name before the engine reads a byte.

use std::io::{self, Read, Write};

use crate::engine::StateError;

/// The file's first eight bytes.
pub(crate) const MAGIC: [u8; 8] = *b"BLMSLOT\0";
/// The layout above. A file of another version is refused, never read.
pub(crate) const VERSION: u32 = 1;

pub(crate) fn write_u32(w: &mut dyn Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub(crate) fn write_u64(w: &mut dyn Write, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub(crate) fn read_u32(r: &mut dyn Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

pub(crate) fn read_u64(r: &mut dyn Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

/// `ids` as u32s, in order.
pub(crate) fn write_ids(w: &mut dyn Write, ids: &[u32]) -> io::Result<()> {
    ids.iter().try_for_each(|&id| write_u32(w, id))
}

/// `n` ids, each below `n_vocab`; an id at or past it is refused by position.
pub(crate) fn read_ids(r: &mut dyn Read, n: usize, n_vocab: usize) -> Result<Vec<u32>, StateError> {
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        let id = read_u32(r)?;
        if !usize::try_from(id).is_ok_and(|v| v < n_vocab) {
            return Err(StateError::Format(format!(
                "id {id} at position {i} is past the vocabulary of {n_vocab}"
            )));
        }
        ids.push(id);
    }
    Ok(ids)
}

/// Writes the header and the slot's ids.
pub(crate) fn write_header(
    w: &mut dyn Write,
    n_vocab: usize,
    ids: &[u32],
) -> Result<(), StateError> {
    let vocab = u32::try_from(n_vocab)
        .map_err(|_| StateError::Format(format!("a vocabulary of {n_vocab} passes u32")))?;
    let n = u64::try_from(ids.len()).expect("a slice length fits u64");
    w.write_all(&MAGIC)?;
    write_u32(w, VERSION)?;
    write_u32(w, vocab)?;
    write_u64(w, n)?;
    write_ids(w, ids)?;
    Ok(())
}

/// Reads the header and the ids a file carries: at most `ctx_max` of them, each
/// below `n_vocab`, written by a server of the same vocabulary.
pub(crate) fn read_header(
    r: &mut dyn Read,
    n_vocab: usize,
    ctx_max: usize,
) -> Result<Vec<u32>, StateError> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(StateError::Format(format!(
            "not a slot file: magic {magic:02x?}, a slot file starts with {MAGIC:02x?}"
        )));
    }
    let version = read_u32(r)?;
    if version != VERSION {
        return Err(StateError::Format(format!(
            "slot file version {version}; this server reads version {VERSION}"
        )));
    }
    let vocab = read_u32(r)?;
    if usize::try_from(vocab).ok() != Some(n_vocab) {
        return Err(StateError::Format(format!(
            "the file was saved with a vocabulary of {vocab}; this model's is {n_vocab}"
        )));
    }
    let n = read_u64(r)?;
    let n = usize::try_from(n)
        .ok()
        .filter(|&n| n <= ctx_max)
        .ok_or_else(|| {
            StateError::Format(format!(
                "the file holds {n} positions; the context holds {ctx_max}"
            ))
        })?;
    read_ids(r, n, n_vocab)
}

/// A writer that counts the bytes it passed on.
pub(crate) struct Counting<T> {
    pub inner: T,
    pub bytes: u64,
}

impl<T> Counting<T> {
    pub(crate) fn new(inner: T) -> Self {
        Counting { inner, bytes: 0 }
    }
}

impl<T: Write> Write for Counting<T> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.bytes += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

impl<T: Read> Read for Counting<T> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.bytes += n as u64;
        Ok(n)
    }
}

/// llama.cpp's `fs_validate_filename` without subdirectories: a non-empty base
/// name of at most 255 bytes, no path separator, no control character, none of
/// `: * ? " < > |` nor their look-alikes, no `..`, not `.`, and no leading or
/// trailing space or trailing dot.
pub(crate) fn valid_filename(name: &str) -> bool {
    if name.is_empty() || name.len() > 255 {
        return false;
    }
    let forbidden = |c: char| {
        let u = u32::from(c);
        u <= 0x1F
            || u == 0x7F
            || (0x80..=0x9F).contains(&u)
            || matches!(
                c,
                '\u{FF0E}'
                    | '\u{2215}'
                    | '\u{2216}'
                    | '\u{FFFD}'
                    | '\u{FEFF}'
                    | '/'
                    | '\\'
                    | ':'
                    | '*'
                    | '?'
                    | '"'
                    | '<'
                    | '>'
                    | '|'
            )
    };
    !(name.chars().any(forbidden)
        || name.starts_with(' ')
        || name.ends_with(' ')
        || name.ends_with('.')
        || name.contains(".."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filenames_are_base_names_only() {
        for ok in ["a", "slot-0.bin", "prompt v2.bin", "é.bin", ".hidden"] {
            assert!(valid_filename(ok), "{ok:?} refused");
        }
        let long = "x".repeat(256);
        for bad in [
            "",
            ".",
            "..",
            "a/b",
            "/abs",
            "a\\b",
            "a..b",
            "trail.",
            " lead",
            "trail ",
            "a:b",
            "a*b",
            "a?b",
            "a\"b",
            "a<b",
            "a>b",
            "a|b",
            "a\nb",
            "a\u{7F}b",
            "a\u{85}b",
            "a\u{2215}b",
            "a\u{FF0E}b",
            "a\u{FEFF}b",
            long.as_str(),
        ] {
            assert!(!valid_filename(bad), "{bad:?} accepted");
        }
        assert!(valid_filename(&"x".repeat(255)));
    }
}
