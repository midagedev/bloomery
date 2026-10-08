//! MD5 (RFC 1321), the digest the tokenizer and dequant manifests state for
//! the texts and files of their sets (`md5sum` writes the same hex). It
//! identifies a file, it does not authenticate one.

use crate::RefError;
use std::fmt::Display;
use std::io::Read;
use std::path::Path;

/// A streaming MD5 state.
#[derive(Clone)]
pub(crate) struct Md5 {
    state: [u32; 4],
    buf: [u8; 64],
    fill: usize,
    len: u64,
}

impl Md5 {
    pub(crate) fn new() -> Md5 {
        Md5 {
            state: [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476],
            buf: [0; 64],
            fill: 0,
            len: 0,
        }
    }

    pub(crate) fn update(&mut self, mut data: &[u8]) {
        self.len = self.len.wrapping_add(data.len() as u64);
        if self.fill > 0 {
            let take = (64 - self.fill).min(data.len());
            self.buf[self.fill..self.fill + take].copy_from_slice(&data[..take]);
            self.fill += take;
            data = &data[take..];
            if self.fill < 64 {
                return;
            }
            let block = self.buf;
            self.block(&block);
            self.fill = 0;
        }
        let (blocks, rest) = data.as_chunks::<64>();
        for b in blocks {
            self.block(b);
        }
        self.buf[..rest.len()].copy_from_slice(rest);
        self.fill = rest.len();
    }

    /// The digest as the lowercase hex `md5sum` prints.
    pub(crate) fn finish_hex(mut self) -> String {
        let bits = self.len.wrapping_mul(8);
        self.update(&[0x80]);
        while self.fill != 56 {
            self.update(&[0]);
        }
        self.update(&bits.to_le_bytes());
        self.state
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    fn block(&mut self, b: &[u8; 64]) {
        const S: [u32; 64] = [
            7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20,
            5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
            6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
        ];
        const K: [u32; 64] = [
            0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
            0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
            0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
            0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
            0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
            0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
            0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
            0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
            0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
            0xeb86d391,
        ];
        let mut m = [0u32; 16];
        for (w, c) in m.iter_mut().zip(b.as_chunks::<4>().0) {
            *w = u32::from_le_bytes(*c);
        }
        let [mut a, mut bb, mut c, mut d] = self.state;
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((bb & c) | (!bb & d), i),
                1 => ((d & bb) | (!d & c), (5 * i + 1) % 16),
                2 => (bb ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (bb | !d), (7 * i) % 16),
            };
            let f = f.wrapping_add(a).wrapping_add(K[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = bb;
            bb = bb.wrapping_add(f.rotate_left(S[i]));
        }
        for (s, v) in self.state.iter_mut().zip([a, bb, c, d]) {
            *s = s.wrapping_add(v);
        }
    }
}

/// The md5 of `bytes`, as `md5sum` prints it.
pub(crate) fn hex_of(bytes: &[u8]) -> String {
    let mut h = Md5::new();
    h.update(bytes);
    h.finish_hex()
}

/// The md5 and the length of the file at `path`, read in 1 MiB steps. An
/// unreadable file is [`RefError::Missing`], naming it.
pub(crate) fn file_digest(path: &Path) -> Result<(String, u64), RefError> {
    let mut f = std::fs::File::open(path)
        .map_err(|e| RefError::missing(path, format!("{}: {e}", path.display())))?;
    let mut h = Md5::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut len = 0u64;
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| RefError::missing(path, format!("{}: {e}", path.display())))?;
        if n == 0 {
            return Ok((h.finish_hex(), len));
        }
        h.update(&buf[..n]);
        len += n as u64;
    }
}

/// A manifest's `<path>\t<md5>` pair of the line `fields` (the line split on
/// tabs, its key first): a program or a file the set was produced with.
/// Another width, or an md5 that is not 32 lowercase hex digits, is
/// `Malformed` at `at`.
pub(crate) fn path_and_md5(
    fields: &[&str],
    at: &dyn Display,
) -> Result<(String, String), RefError> {
    let &[key, path, md5] = fields else {
        return Err(RefError::malformed(
            at,
            format!(
                "{:?}: want <key>\t<path>\t<md5>, got {} fields",
                fields.first().copied().unwrap_or(""),
                fields.len()
            ),
        ));
    };
    check_hex(key, md5, at)?;
    Ok((path.to_string(), md5.to_string()))
}

/// [`RefError::Malformed`] at `at` unless `md5`, the value of `what`, is 32
/// lowercase hex digits.
pub(crate) fn check_hex(what: &str, md5: &str, at: &dyn Display) -> Result<(), RefError> {
    if md5.len() == 32 && md5.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return Ok(());
    }
    Err(RefError::malformed(
        at,
        format!("{what}: {md5:?} is not an md5 (32 lowercase hex digits)"),
    ))
}

/// The file at `path` against the length and md5 its manifest row names:
/// [`RefError::Malformed`] naming the file and both digests when it is
/// another file (a byte changed, appended or cut), [`RefError::Missing`]
/// when it is not there.
pub(crate) fn check_file(path: &Path, bytes: u64, md5: &str) -> Result<(), RefError> {
    let (got_md5, got_bytes) = file_digest(path)?;
    if (got_md5.as_str(), got_bytes) == (md5, bytes) {
        return Ok(());
    }
    Err(RefError::malformed(
        path.display(),
        format!(
            "{got_bytes} bytes md5 {got_md5}, the manifest's row names {bytes} bytes md5 {md5}: \
             the file is not the one the set was dumped with"
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::{Md5, file_digest, hex_of};

    /// The test suite of RFC 1321 appendix A.5, which every block length
    /// class of the padding rule appears in (empty, under one block, across
    /// the 56-byte length boundary, several blocks).
    #[test]
    fn the_rfc_1321_suite() {
        for (text, md5) in [
            ("", "d41d8cd98f00b204e9800998ecf8427e"),
            ("a", "0cc175b9c0f1b6a831c399e269772661"),
            ("abc", "900150983cd24fb0d6963f7d28e17f72"),
            ("message digest", "f96b697d7cb7938d525a2f31aaf161d0"),
            (
                "abcdefghijklmnopqrstuvwxyz",
                "c3fcd3d76192e4007dfb496cca67e13b",
            ),
            (
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
                "d174ab98d277d9f5a5611c2c9f419d9f",
            ),
            (
                "12345678901234567890123456789012345678901234567890123456789012345678901234567890",
                "57edf4a22be3c955ac49da2e2107b67a",
            ),
        ] {
            assert_eq!(hex_of(text.as_bytes()), md5, "md5 of {text:?}");
        }
    }

    /// A text fed in pieces of any size hashes like the whole: the carry of
    /// a partial block across `update` calls.
    #[test]
    fn pieces_hash_like_the_whole() {
        let text: Vec<u8> = (0..1000u32).map(|i| (i * 7 + 3) as u8).collect();
        let whole = hex_of(&text);
        for piece in [1, 3, 55, 56, 63, 64, 65, 129, 999] {
            let mut h = Md5::new();
            for c in text.chunks(piece) {
                h.update(c);
            }
            assert_eq!(h.finish_hex(), whole, "pieces of {piece}");
        }
    }

    /// A file hashes like its bytes, across the read step, and an absent one
    /// is named.
    #[test]
    fn a_file_hashes_like_its_bytes() {
        let path = std::env::temp_dir().join(format!("bloomery-md5-{}.bin", std::process::id()));
        let bytes: Vec<u8> = (0..(3 << 19) + 17u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &bytes).unwrap_or_else(|e| panic!("{e}"));
        let got = file_digest(&path);
        std::fs::remove_file(&path).unwrap_or_else(|e| panic!("{e}"));
        match got {
            Ok((md5, len)) => assert_eq!((md5, len), (hex_of(&bytes), bytes.len() as u64)),
            Err(e) => panic!("{e}"),
        }
        assert!(file_digest(&path).is_err_and(|e| e.to_string().contains("bloomery-md5-")));
    }
}
