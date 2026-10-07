//! The one `O_DIRECT` reader: a file opened so its reads bypass the page
//! cache, the aligned span every caller reads over, and the read loop they
//! share — the open, the alignment contract and the short-read advance that
//! [`crate::open_direct`] held, the probe's `read_direct` loop and odswap's
//! `Ring::read_direct` beside them, one body.
//!
//! A read through [`DirectFile`] takes a buffer, a file offset and a length
//! that are each a multiple of [`DIRECT_ALIGN`] (a drive's logical block is
//! 512 B or 4 KiB; this holds for both). [`DirectFile::open`] proves the
//! mount takes such reads with one aligned probe read at load, so a
//! filesystem without direct IO (a Windows mount under WSL2) is refused by
//! name there and not at the first fill; [`DirectFile::open_buffered`] is
//! the same handle over a plain descriptor for the reader a caller opts
//! into, its reads served by the page cache.

use std::fs::File;
use std::io::ErrorKind;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use memmap2::MmapMut;

use crate::EngramError;

/// The alignment `O_DIRECT` reads keep: the buffer's address, the file
/// offset and the length are each a multiple of it (a drive's logical block
/// is 512 B or 4 KiB; this holds for both).
pub const DIRECT_ALIGN: usize = 4096;

/// `path` opened for reads that bypass the page cache (`O_DIRECT`). Every
/// read through the handle takes a buffer, a file offset and a length that
/// are multiples of [`DIRECT_ALIGN`]; the kernel refuses any other with
/// `EINVAL`.
pub fn open_direct(path: &Path) -> Result<File, EngramError> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .map_err(|source| EngramError::OpenDirect {
            path: path.to_path_buf(),
            source,
        })
}

/// A file read without the page cache ([`DirectFile::open`]) or through it
/// ([`DirectFile::open_buffered`]): one handle, one read loop, the refusal
/// names carrying its path.
pub struct DirectFile {
    path: PathBuf,
    file: File,
    /// Whether the descriptor carries `O_DIRECT`, for the readers a caller
    /// counts by mode.
    direct: bool,
}

impl DirectFile {
    /// `path` opened for reads that bypass the page cache, then one aligned
    /// probe read of its first block to prove the mount takes them: an
    /// `EINVAL` there (a filesystem without direct IO) is a named refusal
    /// with the path and the errno, not a read that fails later at a fill.
    pub fn open(path: &Path) -> Result<DirectFile, EngramError> {
        let file = open_direct(path)?;
        let mut probe = vec![0u8; 2 * DIRECT_ALIGN];
        let off = probe.as_ptr().align_offset(DIRECT_ALIGN);
        let buf = &mut probe[off..off + DIRECT_ALIGN];
        read_all(&file, buf, 0).map_err(|source| EngramError::DirectProbe {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(DirectFile {
            path: path.to_path_buf(),
            file,
            direct: true,
        })
    }

    /// `path` opened for reads the page cache serves: the same
    /// [`DirectFile::read_span`], no probe (any read the cache can serve is
    /// aligned to nothing). The caller that opts into this handle counts
    /// its reads as buffered itself.
    pub fn open_buffered(path: &Path) -> Result<DirectFile, EngramError> {
        let file = File::open(path).map_err(|source| EngramError::OpenDirect {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(DirectFile {
            path: path.to_path_buf(),
            file,
            direct: false,
        })
    }

    /// The file's path, for the refusals a caller names.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Whether the descriptor carries `O_DIRECT` ([`DirectFile::open`]) or
    /// reads through the page cache ([`DirectFile::open_buffered`]).
    pub fn is_direct(&self) -> bool {
        self.direct
    }

    /// Read `dst.len()` bytes at `at` into `dst`, advancing past short reads
    /// and retrying `EINTR`, and return the bytes placed: `dst.len()`, or
    /// fewer only when the file ends first — the caller whose need stops
    /// inside the last block judges its own `need`. The buffer's address,
    /// `at` and the length are each a multiple of [`DIRECT_ALIGN`]; anything
    /// else is a named refusal, not a read the kernel silently drops.
    pub fn read_span(&self, dst: &mut [u8], at: u64) -> Result<usize, EngramError> {
        check_span(self.direct, dst.as_ptr() as usize, at, dst.len()).map_err(|_| {
            EngramError::DirectUnaligned {
                path: self.path.clone(),
                at,
                len: dst.len(),
            }
        })?;
        read_all(&self.file, dst, at).map_err(|source| EngramError::DirectRead {
            path: self.path.clone(),
            at,
            len: dst.len(),
            source,
        })
    }
}

/// The aligned read of `len` bytes at `at`: the start rounded down to the
/// alignment, the span from there through the end rounded up, and the bytes
/// from the start through the last byte the caller needs. An aligned read
/// may name bytes before and after the caller's own — never fewer.
pub fn aligned_span(at: u64, len: u64) -> (u64, usize, usize) {
    let a = DIRECT_ALIGN as u64;
    let start = at / a * a;
    let end = (at + len).div_ceil(a) * a;
    (start, (end - start) as usize, (at + len - start) as usize)
}

/// An anonymous read-write mapping, the arena a caller fills with direct
/// reads: page-aligned by construction (so [`DIRECT_ALIGN`]-aligned windows
/// carved in it stay aligned) and never moved. Its pages leave only over
/// the ranges its owner names ([`AnonMap::dontneed`]), which frees an
/// anonymous range's pages — the next touch reads zeroes, which is the
/// owner's to guarantee does not name a span still read.
pub struct AnonMap(MmapMut);

impl AnonMap {
    /// `bytes` zeroed bytes of anonymous memory.
    pub fn new(bytes: usize) -> Result<AnonMap, EngramError> {
        MmapMut::map_anon(bytes)
            .map_err(|source| EngramError::AnonMap { bytes, source })
            .map(AnonMap)
    }

    /// The mapping's bytes, read where a caller lends them.
    pub fn bytes(&self) -> &[u8] {
        &self.0
    }

    /// The mapping's first byte as a pointer, for the caller that carves
    /// windows and writes them itself.
    pub fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr()
    }

    /// Return the pages of `at .. at + len` — whole pages of the mapping —
    /// to the kernel: the next touch of the range reads zeroes.
    pub fn dontneed(&self, at: usize, len: usize) -> Result<(), EngramError> {
        // SAFETY: the mapping is this value's own anonymous `MmapMut`, so
        // DONTNEED frees the range's pages rather than dropping a file's,
        // and `at .. at + len` names whole pages inside it (the caller carves
        // the ranges it owns); nothing aliases the range through another
        // view — the bytes are reached only through this mapping.
        unsafe {
            self.0
                .unchecked_advise_range(memmap2::UncheckedAdvice::DontNeed, at, len)
        }
        .map_err(|source| EngramError::AnonMapAdvise { at, len, source })
    }
}

/// The `O_DIRECT` alignment contract: the buffer's address, the offset and
/// the length each a multiple of [`DIRECT_ALIGN`] (a buffered handle reads
/// any buffer, but one loop serves both, so the contract holds for both).
fn check_span(direct: bool, ptr: usize, at: u64, len: usize) -> Result<(), ()> {
    if !direct {
        return Ok(());
    }
    if ptr.is_multiple_of(DIRECT_ALIGN)
        && at.is_multiple_of(DIRECT_ALIGN as u64)
        && len.is_multiple_of(DIRECT_ALIGN)
    {
        Ok(())
    } else {
        Err(())
    }
}

/// `file`'s bytes `at .. at + dst.len()` into `dst`, advancing past short
/// reads and retrying `EINTR`; the bytes placed, fewer than asked only when
/// the file ends first. Every other failure names its errno to the caller.
fn read_all(file: &File, dst: &mut [u8], at: u64) -> Result<usize, std::io::Error> {
    let mut got = 0usize;
    while got < dst.len() {
        match file.read_at(&mut dst[got..], at + got as u64) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(got)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_span_rounds_out_around_its_own_bytes() {
        let a = DIRECT_ALIGN as u64;
        for (at, len) in [
            (0, 1),
            (a, a),
            (a + 7, a),
            (3 * a - 1, 2),
            (1_000_003, 921_600),
        ] {
            let (start, span, need) = aligned_span(at, len);
            assert_eq!(start % a, 0, "start of {at}+{len}");
            assert_eq!(span as u64 % a, 0, "span of {at}+{len}");
            assert!(start <= at, "{at}+{len}: the start is not past the part");
            assert!(
                at + len <= start + span as u64,
                "{at}+{len}: the span holds the part"
            );
            assert_eq!(need as u64, at + len - start, "{at}+{len}: the need");
            assert!(
                span - need < DIRECT_ALIGN,
                "{at}+{len}: no whole extra block is read"
            );
        }
    }

    #[test]
    fn a_direct_span_refuses_an_unaligned_buffer_offset_or_length() {
        let a = DIRECT_ALIGN;
        // The one aligned case; every other row is a refusal.
        assert!(check_span(true, 8 * a, 4 * a as u64, a).is_ok());
        for (ptr, at, len) in [
            (8 * a + 1, 4 * a as u64, a),
            (8 * a, 4 * a as u64 + 1, a),
            (8 * a, 4 * a as u64, a + 1),
            (0, 0, 1),
        ] {
            assert!(check_span(true, ptr, at, len).is_err(), "{ptr}/{at}/{len}");
        }
        // A buffered handle takes any buffer: one loop, no kernel contract.
        for (ptr, at, len) in [(8 * a + 1, 4 * a as u64 + 1, 1), (1, 1, 1)] {
            assert!(check_span(false, ptr, at, len).is_ok(), "{ptr}/{at}/{len}");
        }
    }
}
