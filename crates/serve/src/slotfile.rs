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
//! | 8 | `n_media`, the image spans among the ids (u64) |
//! | 48 · `n_media` | each span in order: `at` (u64), `len` (u64), the image's sha256 |
//! | rest | the engine's state ([`crate::Engine::save_state`]), to the end of the file |
//!
//! A file of version 1 ([`V1`]) has no `n_media` and no spans: it reads as a
//! slot that holds no image. llama.cpp's sequence file has the same shape
//! (magic, version, the token count, the tokens, then the context state to the
//! end) and keeps an image as its key alone too. A file whose magic, version or
//! vocabulary differs, whose count passes the context, whose id passes the
//! vocabulary, or whose span is empty, overlaps the one before or passes the
//! ids is refused by name before the engine reads a byte.

use std::io::{self, Read, Write};

use crate::engine::StateError;
use crate::media::{Held, ImageKey, MediaSpan};

/// The file's first eight bytes.
pub(crate) const MAGIC: [u8; 8] = *b"BLMSLOT\0";
/// The layout above. A file of a version other than this and [`V1`] is
/// refused, never read.
pub(crate) const VERSION: u32 = 2;
/// The layout before images: the ids alone.
pub(crate) const V1: u32 = 1;

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

/// The spans `n_ids` ids can hold: each at least one position, after the one
/// before it, and inside the ids. The writer and the reader hold a table to the
/// same rule.
fn check_spans(spans: &[MediaSpan], n_ids: usize) -> Result<(), StateError> {
    let mut free = 0;
    for (i, s) in spans.iter().enumerate() {
        let end = s.at.checked_add(s.len);
        let why = if s.len == 0 {
            "holds no position"
        } else if s.at < free {
            "overlaps the span before it"
        } else if end.is_none_or(|e| e > n_ids) {
            "passes the ids"
        } else {
            free = s.at + s.len;
            continue;
        };
        return Err(StateError::Format(format!(
            "image span {i} (at {}, {} positions) {why}, of {n_ids} positions",
            s.at, s.len
        )));
    }
    Ok(())
}

/// Writes the header: the slot's ids and its image spans.
pub(crate) fn write_header(
    w: &mut dyn Write,
    n_vocab: usize,
    held: &Held,
) -> Result<(), StateError> {
    let vocab = u32::try_from(n_vocab)
        .map_err(|_| StateError::Format(format!("a vocabulary of {n_vocab} passes u32")))?;
    check_spans(&held.media, held.ids.len())?;
    let fits = |n: usize| u64::try_from(n).expect("a length fits u64");
    w.write_all(&MAGIC)?;
    write_u32(w, VERSION)?;
    write_u32(w, vocab)?;
    write_u64(w, fits(held.ids.len()))?;
    write_ids(w, &held.ids)?;
    write_u64(w, fits(held.media.len()))?;
    for s in &held.media {
        write_u64(w, fits(s.at))?;
        write_u64(w, fits(s.len))?;
        w.write_all(&s.key.0)?;
    }
    Ok(())
}

/// Reads the header a file carries: at most `ctx_max` ids, each below
/// `n_vocab`, written by a server of the same vocabulary, and the image spans
/// among them (none in a file of version 1).
pub(crate) fn read_header(
    r: &mut dyn Read,
    n_vocab: usize,
    ctx_max: usize,
) -> Result<Held, StateError> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(StateError::Format(format!(
            "not a slot file: magic {magic:02x?}, a slot file starts with {MAGIC:02x?}"
        )));
    }
    let version = read_u32(r)?;
    if version != VERSION && version != V1 {
        return Err(StateError::Format(format!(
            "slot file version {version}; this server reads versions {V1} and {VERSION}"
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
    let ids = read_ids(r, n, n_vocab)?;
    if version == V1 {
        return Ok(Held::from(ids));
    }
    // A span holds a position at least, so no more spans than ids.
    let m = read_u64(r)?;
    let m = usize::try_from(m).ok().filter(|&m| m <= n).ok_or_else(|| {
        StateError::Format(format!("the file holds {m} image spans in {n} positions"))
    })?;
    let mut media = Vec::with_capacity(m);
    for _ in 0..m {
        let at = read_u64(r)?;
        let len = read_u64(r)?;
        let mut key = [0u8; 32];
        r.read_exact(&mut key)?;
        // A value past usize passes the ids, which check_spans names.
        let fit = |v: u64| usize::try_from(v).unwrap_or(usize::MAX);
        media.push(MediaSpan {
            at: fit(at),
            len: fit(len),
            key: ImageKey(key),
        });
    }
    check_spans(&media, n)?;
    Ok(Held { ids, media })
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

    fn span(at: usize, len: usize, key: u8) -> MediaSpan {
        MediaSpan {
            at,
            len,
            key: ImageKey([key; 32]),
        }
    }

    /// Seven ids holding two images, the spans 1..3 and 3..6.
    fn imaged() -> Held {
        Held {
            ids: vec![0, 7, 7, 7, 7, 7, 1],
            media: vec![span(1, 2, 4), span(3, 3, 5)],
        }
    }

    fn header(held: &Held) -> Vec<u8> {
        let mut bytes = Vec::new();
        write_header(&mut bytes, 8, held).expect("a header");
        bytes
    }

    fn read(bytes: &[u8]) -> Result<Held, StateError> {
        read_header(&mut &bytes[..], 8, 64)
    }

    /// A header carries the ids and the span table whole, and nothing past
    /// it is read: magic, version, vocabulary and count (24 bytes), four bytes
    /// an id, the span count, 48 bytes a span.
    #[test]
    fn a_header_round_trips_the_ids_and_the_spans() {
        let mut bytes = header(&imaged());
        assert_eq!(bytes.len(), 24 + 4 * 7 + 8 + 48 * 2);
        bytes.extend_from_slice(b"engine");
        let mut r = &bytes[..];
        assert_eq!(read_header(&mut r, 8, 64).expect("a header"), imaged());
        assert_eq!(r, b"engine", "the engine's part is left unread");
        let text = Held::from(vec![1, 2, 3]);
        assert_eq!(header(&text).len(), 24 + 4 * 3 + 8);
        assert_eq!(read(&header(&text)).expect("a header"), text);
    }

    /// A version 1 file is the ids alone: it reads as a slot that holds no
    /// image, and the engine's part starts right after the ids.
    #[test]
    fn a_v1_file_reads_with_no_image() {
        let v2 = header(&imaged());
        let mut v1 = v2[..24 + 4 * 7].to_vec();
        v1[8..12].copy_from_slice(&V1.to_le_bytes());
        v1.extend_from_slice(b"engine");
        let mut r = &v1[..];
        assert_eq!(
            read_header(&mut r, 8, 64).expect("a v1 file"),
            Held::from(imaged().ids)
        );
        assert_eq!(r, b"engine");
        let mut v3 = v2.clone();
        v3[8..12].copy_from_slice(&3u32.to_le_bytes());
        let e = read(&v3).expect_err("version 3").to_string();
        assert!(e.contains("reads versions 1 and 2"), "{e}");
    }

    /// A span that holds no position, overlaps the one before it or passes
    /// the ids is refused by name on the way out and on the way in, as is a
    /// span count past the ids.
    #[test]
    fn a_broken_span_table_is_refused_by_name() {
        for (bad, why) in [
            (span(3, 0, 5), "holds no position"),
            (span(2, 3, 5), "overlaps the span before it"),
            (span(5, 3, 5), "passes the ids"),
        ] {
            let mut held = imaged();
            held.media[1] = bad;
            let e = write_header(&mut Vec::new(), 8, &held)
                .expect_err("the writer refuses the table")
                .to_string();
            assert!(e.contains(why), "{why}: {e}");
            let mut bytes = header(&imaged());
            let at = bytes.len() - 48;
            bytes[at..at + 8].copy_from_slice(&(bad.at as u64).to_le_bytes());
            bytes[at + 8..at + 16].copy_from_slice(&(bad.len as u64).to_le_bytes());
            let e = read(&bytes)
                .expect_err("the reader refuses the table")
                .to_string();
            assert!(e.contains(why), "{why}: {e}");
        }
        let mut bytes = header(&imaged());
        let count = 24 + 4 * 7;
        bytes[count..count + 8].copy_from_slice(&8u64.to_le_bytes());
        let e = read(&bytes).expect_err("8 spans in 7 ids").to_string();
        assert!(e.contains("8 image spans in 7 positions"), "{e}");
        let mut cut = header(&imaged());
        cut.truncate(cut.len() - 1);
        assert!(matches!(read(&cut), Err(StateError::Io(_))), "a cut table");
    }
}
