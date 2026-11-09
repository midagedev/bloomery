//! The host side of `bloomery-quantize`: llama-quantize's job in one pass, byte for byte. The routed-expert stacks
//! become IQ1_S or IQ2_XXS on the card or the host (ggml's bytes, the importance llama-quantize computes), every
//! other tensor the type llama-quantize gives it by the host's K-quant rules, the metadata and the shard layout as
//! llama-quantize writes them.
//!
//! - [`rules`]: llama-quantize's `--tensor-type-file`, read the way it reads it.
//! - [`resolve`]: `llama_tensor_get_type`, tensor by tensor, with its counters.
//! - [`imatrix`]: the importance matrix file as llama-quantize reads it.
//! - [`meta`]: the metadata of every output shard, key by key.
//! - [`plan`]: every tensor's type and fate, the bytes, the room and the derived wall; writes nothing.
//! - [`scan`]: the finite check of every source row.
//! - [`source`]: where the input is read from: a GGUF split or a checkpoint folder, one [`source::Source`].
//! - [`check`]: a row's bytes against the host rule's, naming the first byte that differs.
//! - [`host`]: the host rules on all cores: the IQ search as a [`Backend`] that opens no card, the K-quant rules.
//! - [`run`]: the writer, over a [`Backend`] (the card's lives in the binary).
//! - [`cli`]: the arguments and the usage text.
//!
//! Every refusal is an [`Error`] that names what it refuses: tensor, expert, row.

pub mod check;
pub mod cli;
pub mod host;
pub mod imatrix;
pub mod meta;
pub mod plan;
pub mod resolve;
pub mod rules;
pub mod run;
pub mod scan;
pub mod source;

pub use run::{Backend, BackendError, run};

/// The commit this build names (`BLOOMERY_GIT_COMMIT`, which the release build sets), or `unknown`.
pub const COMMIT: &str = match option_env!("BLOOMERY_GIT_COMMIT") {
    Some(c) => c,
    None => "unknown",
};

/// What ends a run, named.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An argument the usage text does not take.
    #[error("{0}")]
    Usage(String),
    /// An input the quantizer refuses by name: a tensor, an imatrix entry, a rule, a value.
    #[error("{0}")]
    Refused(String),
    /// A row the card quantized differently from the host rule.
    #[error(
        "--check: {tensor} expert {expert} row {row}: super-block {block} byte {byte} is {got:#04x}, the host rule's {want:#04x}"
    )]
    Mismatch {
        tensor: String,
        expert: usize,
        row: usize,
        block: usize,
        byte: usize,
        got: u8,
        want: u8,
    },
    /// A file that cannot be read or written.
    #[error("{path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
}

impl Error {
    /// The process's exit status: 64 a usage error, 66 a missing input, 1 every other refusal.
    #[must_use]
    pub fn status(&self) -> u8 {
        match self {
            Error::Usage(_) => 64,
            Error::Io { source, .. } if source.kind() == std::io::ErrorKind::NotFound => 66,
            _ => 1,
        }
    }

    pub(crate) fn refused(s: impl Into<String>) -> Error {
        Error::Refused(s.into())
    }
}

/// The result of this crate's calls.
pub type Result<T> = std::result::Result<T, Error>;
