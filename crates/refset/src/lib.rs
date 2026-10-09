//! Reference sets: the files the gates compare the engine against, read in
//! one place, each refused by name before any comparison when it was not
//! dumped from the file the tree runs.
//!
//! Host only: this crate depends on the GGUF reader and nothing of the
//! engine, so the gate binaries and the model crate's tests read the same
//! sets through the same parser.
//!
//! - [`family`]: a family's row — where its sets live, the recipe that
//!   writes them, which line of a set states the model file it was dumped
//!   from, the ik build it must name, and the gates that read it — and the
//!   checks its readers run.
//! - [`arch`]: each architecture's families, reached by the architecture's
//!   name, and the table of all of them.
//! - [`fixture`]: the fixture tier's identity — the path of each
//!   architecture's fixture file and the `# fixture` line a set dumped from it
//!   carries.
//! - [`ik`]: ik's node dumps (`tools/ref/dump_ref.cpp`), the v1 and v2
//!   `MANIFEST.tsv` and the files its rows name.
//! - [`dsref`]: ik's DSpark draft sets (`tools/ref/dump_draft.cpp`).
//! - [`mtpref`]: ik's MTP draft sets (`tools/ref/dump_mtp.cpp`).
//! - [`greedy`]: ik's greedy continuations (`tools/ref/argmax_ref.cpp`).
//! - [`kld`]: ik's KL-divergence base runs (`tools/ref/ik-ppl.sh`).
//! - [`vision`]: the vision encoder's oracle (`tools/ref/vision/dump_vision.py`).
//! - [`visref`]: the V4.1 vision fork comparison (`tools/ref/vision/visref.sh`).
//! - [`tokenizer`]: `llama-tokenize`'s ids per vocabulary
//!   (`crates/tokenizer/tools/oracle.sh`).
//! - [`dequant`]: ggml's dequantized rows (`tools/ref/dump-dequant.sh`).
//!
//! Every manifest row is read by the column names its header line gives
//! that kind of row, never by position.

use std::path::PathBuf;

pub mod arch;
pub mod clefvis;
mod columns;
pub mod dequant;
pub mod dsref;
pub mod family;
pub mod fixture;
pub mod greedy;
pub mod ik;
pub mod kld;
mod md5;
pub mod mtpref;
pub mod tokenizer;
pub mod vision;
pub mod visref;

/// Why a reference set cannot be used.
#[derive(Debug, thiserror::Error)]
pub enum RefError {
    /// The set's manifest or a file it names cannot be read, or the set holds
    /// no row or line a caller asks for.
    #[error("{what}")]
    Missing { path: PathBuf, what: String },
    /// The set was dumped from another model file than the one the tree runs.
    #[error("stale reference: {set} was dumped from {dumped_from}, the tree runs {runs}")]
    Stale {
        set: String,
        dumped_from: String,
        runs: String,
    },
    /// The set was written by another ik build, or for another architecture,
    /// than its family names.
    #[error("{set}: {field} {got:?}, the {family} family wants {want}")]
    Foreign {
        set: String,
        family: &'static str,
        field: &'static str,
        got: String,
        want: String,
    },
    /// The set has no completion trailer: the dump that wrote it did not
    /// finish, and the files beside it may come from two runs.
    #[error("{set} has no `# complete` trailer: the dump that wrote it did not finish")]
    Unfinished { set: String },
    /// A line or field that does not parse, a row of the wrong width, or a
    /// file that disagrees with the row that names it.
    #[error("{at}: {what}")]
    Malformed { at: String, what: String },
}

impl RefError {
    pub(crate) fn missing(path: impl Into<PathBuf>, what: impl Into<String>) -> RefError {
        RefError::Missing {
            path: path.into(),
            what: what.into(),
        }
    }

    pub(crate) fn malformed(at: impl std::fmt::Display, what: impl std::fmt::Display) -> RefError {
        RefError::Malformed {
            at: at.to_string(),
            what: what.to_string(),
        }
    }
}

/// The data directory on the box — every family's sets, and the oracle
/// binaries — `$BLOOMERY_DATA`, else the workstation default `tools/box.sh`
/// also sets.
pub fn data_dir() -> PathBuf {
    std::env::var("BLOOMERY_DATA")
        .map_or_else(|_| PathBuf::from("/root/bloomery-data"), PathBuf::from)
}
