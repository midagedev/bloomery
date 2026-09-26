//! File-system calls std does not wrap, and the digest text the file writers
//! record. Each returns the bare `io::Error`; the caller names the path and
//! the operation.

use std::ffi::CString;
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use sha2::{Digest, Sha256};

/// `p` as a C string; a path holding a NUL is `InvalidInput`.
fn c_path(p: &Path) -> io::Result<CString> {
    CString::new(p.as_os_str().as_bytes())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))
}

/// Bytes free to this process on the filesystem that holds `dir`.
pub fn free_bytes(dir: &Path) -> io::Result<u64> {
    let c = c_path(dir)?;
    let mut st = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `c` is a NUL-terminated path that lives across the call, and
    // `st` is storage for one `statvfs`, which the call fills on success.
    if unsafe { libc::statvfs(c.as_ptr(), st.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the call returned 0, so it wrote every field of `st`.
    let st = unsafe { st.assume_init() };
    Ok(st.f_bavail.saturating_mul(st.f_frsize))
}

/// Rename `from` to `to`, refused with `AlreadyExists` when `to` exists.
pub fn rename_noreplace(from: &Path, to: &Path) -> io::Result<()> {
    let (a, b) = (c_path(from)?, c_path(to)?);
    // SAFETY: both are NUL-terminated paths that live across the call.
    let rc = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            a.as_ptr(),
            libc::AT_FDCWD,
            b.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// `bytes` as lowercase hex, two digits a byte.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Lowercase hex sha256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}
