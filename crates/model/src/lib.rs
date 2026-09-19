//! DeepSeek-V2-Lite forward pass — the stage-1 model crate.
//!
//! The module boundaries here are the parallel work boundaries: each of `attn`, `ffn`,
//! `moe` and `head` is written and gated against `$BLOOMERY_DATA/ref/` independently, and
//! `forward` in this file only wires them. Nothing reads another module's output during
//! development, because the oracle already holds every intermediate (see `docs/oracle.md`).
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

pub mod attn;
pub mod derived;
pub mod ffn;
pub mod forward;
pub mod head;
pub mod kv;
pub mod moe;
pub mod ops;
pub mod profile;

pub use ops::Tensor2;

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
}
