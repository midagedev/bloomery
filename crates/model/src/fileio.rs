//! File-system calls std does not wrap. Each returns the bare `io::Error`;
//! the caller names the path and the operation.

use std::io;
use std::path::Path;

/// Bytes free to this process on the filesystem that holds `dir`.
pub fn free_bytes(dir: &Path) -> io::Result<u64> {
    os::free_bytes(dir).map_err(io::Error::from)
}

/// Rename `from` to `to`, refused with `AlreadyExists` when `to` exists.
pub fn rename_noreplace(from: &Path, to: &Path) -> io::Result<()> {
    os::rename_noreplace(from, to).map_err(io::Error::from)
}
