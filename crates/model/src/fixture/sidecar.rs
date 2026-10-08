//! The r8 sidecar of a fixture ([`SidecarSpec`]): written from the written
//! files, checked against them. The sidecar's identity is its source's shard
//! names and lengths, header digests and the head and tail of every stack
//! (`model::r8file`), so it is made once the fixture's bytes are final, and a
//! fixture rewritten under another seed or map needs its own.

use std::path::{Path, PathBuf};
use std::time::Instant;

use gguf::{Split, Weights};

use super::spec::SidecarSpec;
use super::{FixtureError, io_err};
use crate::r8file::{self, Sidecar};

/// A sidecar written or checked.
#[derive(Clone, Debug)]
pub struct SidecarStat {
    pub path: PathBuf,
    pub tensors: usize,
    /// The sidecar file's length when written, the stacks' bytes when checked.
    pub bytes: u64,
    pub secs: f64,
}

fn refused(path: &Path, detail: impl std::fmt::Display) -> FixtureError {
    FixtureError::Sidecar {
        path: path.to_path_buf(),
        detail: detail.to_string(),
    }
}

/// Where the sidecar of the fixture whose first shard is `first` lives:
/// `r8file::sidecar_path`, the one owner of the name.
pub fn path_of(first: &Path) -> Result<PathBuf, FixtureError> {
    let abs = std::path::absolute(first).map_err(|e| io_err(first, "resolve", e))?;
    r8file::sidecar_path(&abs).map_err(|e| refused(&abs, e))
}

/// Write the sidecar of the fixture whose first shard is `first`.
pub fn write(sc: &SidecarSpec, first: &Path) -> Result<SidecarStat, FixtureError> {
    let t = Instant::now();
    let split = Split::open(first)?;
    let names = (sc.stacks)(&split)?;
    let path = path_of(first)?;
    let stats =
        r8file::convert(&split, &names, &path, &mut |_| {}).map_err(|e| refused(&path, e))?;
    Ok(SidecarStat {
        path,
        tensors: stats.tensors.len(),
        bytes: stats.file_bytes,
        secs: t.elapsed().as_secs_f64(),
    })
}

/// The sidecar beside `fixture` (an open split) holds exactly the stacks
/// `sc` names, matches the fixture's identity, and is every stack's repack
/// byte for byte.
pub fn check(sc: &SidecarSpec, fixture: &Split) -> Result<SidecarStat, FixtureError> {
    let t = Instant::now();
    let first = fixture
        .shard_path(0)
        .ok_or_else(|| refused(Path::new(""), "the fixture has no first shard"))?;
    let path = path_of(first)?;
    if !path.try_exists().map_err(|e| io_err(&path, "stat", e))? {
        return Err(refused(
            &path,
            "is absent; `fixture generate` writes it beside the fixture",
        ));
    }
    let names = (sc.stacks)(fixture)?;
    let sidecar = Sidecar::open(&path, fixture, Weights::Mapped { populate: false })
        .map_err(|e| refused(&path, e))?;
    let held: Vec<&str> = sidecar.names().collect();
    if held != names.iter().map(String::as_str).collect::<Vec<_>>() {
        return Err(refused(
            &path,
            format!(
                "holds {} stacks ({} .. {}), the fixture's routed layers have {} ({} .. {})",
                held.len(),
                held.first().copied().unwrap_or("-"),
                held.last().copied().unwrap_or("-"),
                names.len(),
                names.first().map_or("-", String::as_str),
                names.last().map_or("-", String::as_str),
            ),
        ));
    }
    let stats = r8file::verify(fixture, &sidecar, &mut |_| {}).map_err(|e| refused(&path, e))?;
    Ok(SidecarStat {
        path,
        tensors: stats.tensors.len(),
        bytes: stats.bytes,
        secs: t.elapsed().as_secs_f64(),
    })
}
