//! engram-lab — the IO lab over the engine's engram table.
//!
//! The engine reads its engram rows through `engram` alone: the tables, the
//! hash and the prefetch helper. What measures those reads lives here, and
//! nothing in the engine uses it. [`SeededRows`] stands in for the hash when
//! the point is the access pattern alone — a uniform draw is the zero-reuse
//! floor. A real stream re-asks for rows, and [`cache::RowCache`] keeps them
//! in DRAM so that only its misses reach the disk; [`reuse::Lru`] is the
//! simulator its hits are held to. [`Context`] walks one contiguous token
//! stream through the hash. The bins are the measurements: `engram-rate`
//! (what one token's rows cost, arm by arm, under the lease) and
//! `engram-reuse` (how often a real stream asks for a row it already had).

use std::path::PathBuf;

pub mod cache;
mod context;
pub mod reuse;

pub use context::Context;

#[derive(Debug, thiserror::Error)]
pub enum LabError {
    #[error("row cache: {0}")]
    Cache(&'static str),
    #[error("reuse simulator: {0}")]
    Reuse(&'static str),
    #[error("{path}: {source}")]
    IdsFile {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{path}:{line}: '{text}': {source}")]
    TokenId {
        path: PathBuf,
        line: usize,
        text: String,
        source: std::num::ParseIntError,
    },
}

/// Bytes this process actually fetched from the block layer (`/proc/self/io`
/// `read_bytes`), readahead it submitted through `madvise` included.
///
/// **Diagnostic, not a hot-path counter**: it opens and reads a file, which
/// costs about as much as a whole warm token. Sample it at the ends of a run.
///
/// This is the only counter that proves a prefetched arm did any IO — see
/// [`engram::Faults`].
pub fn read_bytes() -> Result<u64, std::io::Error> {
    let text = std::fs::read_to_string("/proc/self/io")?;
    text.lines()
        .find_map(|l| l.strip_prefix("read_bytes:"))
        .and_then(|v| v.trim().parse().ok())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "/proc/self/io has no read_bytes line",
            )
        })
}

/// Reproducible row ids, standing in for the model's derivation.
///
/// **The real derivation is [`engram::Hash`]**, and an arm that wants the rows
/// a real token stream asks for uses it. What this reproduces instead is the
/// **access pattern**: uniform over the table, the same sequence for the same
/// seed, with no allocation and no metadata.
///
/// What it does not reproduce is **reuse** — the real hash keys on n-grams, so
/// common bigrams re-hit rows that are still resident. A uniform draw over
/// 384 M rows is the zero-reuse floor; real decode sits between it and an
/// all-resident table, and `engram-reuse` is where that distance is measured.
pub struct SeededRows {
    state: u64,
}

impl SeededRows {
    pub fn new(seed: u64) -> SeededRows {
        SeededRows { state: seed }
    }

    /// `n` ids uniform in `0..rows`, appended to a cleared caller-owned vector.
    pub fn next_into(&mut self, rows: u64, n: usize, out: &mut Vec<u32>) {
        out.clear();
        for _ in 0..n {
            out.push((self.next_u64() % rows) as u32);
        }
    }

    /// splitmix64 — a fixed sequence per seed, so a run is repeatable and two
    /// arms can be handed the same rows.
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}
