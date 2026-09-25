//! The CPU engine: the shared modules (`ops`, `ffn`, `moe`, `head`, `kv`, `profile`) and,
//! under `arch/`, one module per model architecture — `arch::deepseek2` holds
//! DeepSeek-V2-Lite's `attn`, `derived` and `forward`.
//!
//! The module boundaries are the parallel work boundaries: each op module is written and
//! gated against the oracle set (`$BLOOMERY_DATA/ref/`) independently, and an architecture's
//! `forward` only wires them. Nothing reads another module's output during development,
//! because the oracle already holds every intermediate (see `docs/oracle.md`).
//!
//! Three shapes are fixed here on purpose, because `docs/research/quant-decode-efficiency.md`
//! §Q6 marks them as decisions that cannot be retrofitted:
//!
//!   1. **Activations are `[ne0 = embd, ne1 = n_tokens]`, batch first class.** M = 1 is the
//!      special case, never the signature. A forward written for one token cannot be
//!      converted: router, dispatch, KV indexing and the weighted sum all harden around it,
//!      and k-token verification is the largest lever we have (`docs/roofline.md`).
//!   2. **KV is keyed by `(sequence, position)`, two dimensions.** A verification pass writes
//!      k candidate positions and throws most away. A position axis that assumes one
//!      append-only sequence cannot grow a branch/rollback later.
//!   3. **MoE dispatch is expert-bucketed.** Per-token dispatch loses from batch 2 upward and
//!      multiplies a verification pass by k. Bucket → per-bucket matmul → inverse permutation.

pub mod arch;
pub mod ffn;
pub mod head;
pub mod kv;
pub mod moe;
pub mod ops;
pub mod placement;
pub mod profile;
pub mod r8file;

pub use ops::{Tensor2, Tensor2View};

/// Which sequence and which position a token occupies. See decision 2 above — this pair is
/// the KV key, and it exists from the first line so a speculative branch has somewhere to go.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Slot {
    pub seq: u32,
    pub pos: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("tensor {0} is not in the file")]
    MissingTensor(String),
    #[error("unsupported architecture {0:?}")]
    UnknownArchitecture(String),
    #[error("{what}: expected [{want_ne0}, {want_ne1}], got [{got_ne0}, {got_ne1}]")]
    Shape {
        what: &'static str,
        want_ne0: usize,
        want_ne1: usize,
        got_ne0: usize,
        got_ne1: usize,
    },
    #[error(transparent)]
    Gguf(#[from] gguf::LoadError),
    #[error(transparent)]
    Quant(#[from] gguf::QuantError),
    #[error(transparent)]
    Qdot(#[from] qdot::QdotError),
    /// A metadata key the file lacks, holds in the wrong type, or sets to a
    /// value this engine does not run: the full key and why.
    #[error("metadata {key}: {detail}")]
    Metadata { key: String, detail: String },
    /// A refusal of the placement layer that is not a metadata key's.
    #[error(transparent)]
    Placement(placement::PlacementError),
    /// A refusal of the r8 sidecar format: writing, opening or comparing it.
    #[error(transparent)]
    R8(#[from] r8file::R8Error),
}

/// A hyperparameter reader's metadata refusal stays a metadata error; every
/// other placement refusal keeps its own variant and text.
impl From<placement::PlacementError> for ModelError {
    fn from(e: placement::PlacementError) -> ModelError {
        match e {
            placement::PlacementError::Metadata { key, detail } => {
                ModelError::Metadata { key, detail }
            }
            other => ModelError::Placement(other),
        }
    }
}
