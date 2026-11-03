//! Shared SHA-256 and lowercase hex for files and explicitly framed data.
//!
//! [`sha256_file`] hashes a whole file; [`sha256_reader`] hashes only the bytes yielded by its
//! reader, from the current position to EOF. A caller can bound it with [`Read::take`]. Neither
//! adds framing or checks an expected length. [`Sha256`] feeds structured data without copying
//! it: vocabulary lengths and other format-specific framing remain the caller's job.

use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use sha2::Digest;

/// A file open or read failure, retaining both the named path and the original I/O error.
#[derive(Debug, thiserror::Error)]
#[error("{}: {source}", .path.display())]
pub struct DigestError {
    pub path: PathBuf,
    #[source]
    pub source: io::Error,
}

/// A SHA-256 feed. Updates concatenate bytes exactly; callers supply any required framing.
#[derive(Default)]
pub struct Sha256(sha2::Sha256);

impl Sha256 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    pub fn finish(self) -> [u8; 32] {
        self.0.finalize().into()
    }
}

/// `bytes` as lowercase hex, two digits per byte, including leading zeroes.
pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        text.push(char::from(DIGITS[usize::from(byte >> 4)]));
        text.push(char::from(DIGITS[usize::from(byte & 15)]));
    }
    text
}

/// The digest whose 64 hex digits are `s` ([`hex`]'s text of a SHA-256, either case), or `None`.
pub fn unhex(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 || !s.bytes().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let mut d = [0u8; 32];
    for (i, b) in d.iter_mut().enumerate() {
        *b = u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(d)
}

/// Lowercase hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    sha256_hex_of([bytes])
}

/// Lowercase hex SHA-256 of `parts` concatenated in order, without added lengths or separators.
pub fn sha256_hex_of<'a>(parts: impl IntoIterator<Item = &'a [u8]>) -> String {
    let mut sha = Sha256::new();
    for part in parts {
        sha.update(part);
    }
    hex(&sha.finish())
}

/// Hash from the reader's current position to EOF, without seeking or enforcing a length.
/// Read failures retain `path` and their original source; short reads are not EOF.
pub fn sha256_reader<R: Read + ?Sized>(path: &Path, reader: &mut R) -> Result<String, DigestError> {
    let mut sha = Sha256::new();
    let mut buf = [0u8; 64 << 10];
    loop {
        let n = reader.read(&mut buf).map_err(|source| DigestError {
            path: path.to_owned(),
            source,
        })?;
        if n == 0 {
            return Ok(hex(&sha.finish()));
        }
        sha.update(&buf[..n]);
    }
}

/// Hash a whole file with bounded storage; errors name the file and retain their I/O source.
pub fn sha256_file(path: &Path) -> Result<String, DigestError> {
    let mut file = File::open(path).map_err(|source| DigestError {
        path: path.to_owned(),
        source,
    })?;
    sha256_reader(path, &mut file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const EMPTY: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    const ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const BINARY: &str = "f742b965f156c10374bc23aea96e3a8aff8facd6fc079defeaa30219ad86f211";

    #[test]
    fn known_vectors_keep_binary_bytes_and_hex_width() {
        assert_eq!(sha256_hex(b""), EMPTY);
        assert_eq!(sha256_hex(b"abc"), ABC);
        assert_eq!(sha256_hex(&[0, 255, 128]), BINARY);
        assert_eq!(hex(&[0, 1, 15, 16, 128, 255]), "00010f1080ff");
    }

    #[test]
    fn unhex_reads_hex_back_and_refuses_another_length_or_a_non_digit() {
        let d = Sha256::new().finish();
        assert_eq!(unhex(EMPTY), Some(d));
        assert_eq!(unhex(&EMPTY.to_uppercase()), Some(d));
        for bad in [
            &EMPTY[..62],
            &format!("{EMPTY}00"),
            &format!("{}g", &EMPTY[..63]),
            &format!("+a{}", &EMPTY[2..]),
            "",
        ] {
            assert_eq!(unhex(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn reader_crosses_padding_and_buffer_boundaries() {
        // Independent hashlib vectors for repeated 0xa5 bytes, not another invocation of this feed.
        let vectors = [
            (
                55,
                "26ee0116778740a66fe2ba10ea063748b27306acc99188ec812746d4e8d70083",
            ),
            (
                56,
                "4cf71e2b0aa0fcc0c271f68353026a77b8e50153632a8e4a73833cd64080e92e",
            ),
            (
                63,
                "a1942663a5b8b93dffc9c4ff5f62c71a1c021d1fcc1e470dd46172abace1bca5",
            ),
            (
                64,
                "bb626e5577021df95ea17eb6339e75904855b80087e40660931c4a89b302f74a",
            ),
            (
                65,
                "667f84020d981fcedce2816e4e9969a02d5c317a0aef56a6c588175820f82a81",
            ),
            (
                65535,
                "1acaf51d16a9c1a2b05a091386f193dad790603c7803b5e55dd4681a5bb9dfd2",
            ),
            (
                65536,
                "77007cd74a06dc54e5114d01a41d2721679d5668a0c20022fe102c87ad4d65b8",
            ),
            (
                65537,
                "a3253bfa7d790dc6f5d41b75421bcc2c606e8043f9fad4331e8aae0a6c504cfa",
            ),
        ];
        for (len, expected) in vectors {
            let data = vec![0xa5; len];
            assert_eq!(
                sha256_reader(Path::new("boundary"), &mut &data[..]).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn short_reads_are_not_eof() {
        struct Short(Cursor<&'static [u8]>);
        impl Read for Short {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                let n = buf.len().min(1);
                self.0.read(&mut buf[..n])
            }
        }
        let mut reader = Short(Cursor::new(b"abc"));
        assert_eq!(sha256_reader(Path::new("short"), &mut reader).unwrap(), ABC);
    }

    #[test]
    fn reader_preserves_position_and_explicit_prefix_limit() {
        let path = Path::new("bounded");
        let mut reader = Cursor::new(b"xabcTAIL");
        reader.set_position(1);
        assert_eq!(
            sha256_reader(path, &mut (&mut reader).take(3)).unwrap(),
            ABC
        );
        assert_eq!(reader.position(), 4);
        let mut next = [0; 1];
        reader.read_exact(&mut next).unwrap();
        assert_eq!(&next, b"T");
        assert_eq!(
            sha256_reader(path, &mut (&mut reader).take(0)).unwrap(),
            EMPTY
        );
        assert_eq!(reader.position(), 5);
        assert_eq!(
            sha256_reader(path, &mut (&mut reader).take(100)).unwrap(),
            sha256_hex(b"AIL")
        );
    }

    #[test]
    fn parts_preserve_byte_concatenation_without_adding_framing() {
        assert_eq!(sha256_hex_of([&b""[..], b"a", b"", b"bc"]), ABC);
        assert_eq!(sha256_hex_of([&b"ab"[..], b"c"]), ABC);
        assert_eq!(sha256_hex_of([&[0, 255][..], &[128]]), BINARY);
    }

    #[test]
    fn a_read_error_after_data_keeps_the_path_and_original_source() {
        struct FailAfter(bool);
        impl Read for FailAfter {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                if std::mem::replace(&mut self.0, false) {
                    buf[..3].copy_from_slice(b"abc");
                    Ok(3)
                } else {
                    Err(io::Error::new(io::ErrorKind::PermissionDenied, "denied"))
                }
            }
        }
        let path = Path::new("declared-tensor.data");
        let error = sha256_reader(path, &mut FailAfter(true)).unwrap_err();
        assert_eq!(error.path, path);
        assert_eq!(error.source.kind(), io::ErrorKind::PermissionDenied);
        let source = std::error::Error::source(&error)
            .unwrap()
            .downcast_ref::<io::Error>()
            .unwrap();
        assert_eq!(source.kind(), io::ErrorKind::PermissionDenied);
    }
}
