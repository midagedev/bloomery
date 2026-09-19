//! MoE FFN: `ffn_norm-N` in, `ffn_out-N` out, for N >= 1.
//!
//! Dispatch is expert-bucketed, not per-token — `docs/research/quant-decode-efficiency.md`
//! §Q6 decision 2, and it cannot be retrofitted. Build the bucket table first, then one
//! matmul per bucket over the tokens routed to it, then scatter back through the inverse
//! permutation and weight. A per-token loop that happens to pass the gate at n_tokens = 6
//! is still the wrong structure and will be rejected in review.
//!
//! Gate: `crates/model/tests/moe.rs`. Routing ids are compared EXACTLY (they are i32 in the
//! oracle); numerics come second at 1e-3. Owned by the moe round.
use crate::{ModelError, Tensor2};
use gguf::Gguf;

/// Tokens grouped by the expert they were routed to.
///
/// `offsets` has `n_experts + 1` entries; the tokens for expert `e` are
/// `order[offsets[e]..offsets[e + 1]]`, each an index into the input's `ne1`.
pub struct Buckets {
    pub offsets: Vec<u32>,
    pub order: Vec<u32>,
    /// The router weight for each entry of `order`, in the same order.
    pub weight: Vec<f32>,
}

pub fn route(_gguf: &Gguf, _block: usize, _x: &Tensor2) -> Result<Buckets, ModelError> {
    todo!("moe round")
}

pub fn moe_ffn(_gguf: &Gguf, _block: usize, _x: &Tensor2) -> Result<Tensor2, ModelError> {
    todo!("moe round")
}
