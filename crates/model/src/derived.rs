//! Weight-derived state: the `wk_b` Q8_0 requant, built once per file.
//!
//! The absorption in `q_nope2_absorbed` consumes these blocks, and computing
//! them is a pure function of the weights — nothing about the tokens enters —
//! so rebuilding them per block, per head, per decode step re-derived bytes the
//! file had already fixed. [`Derived`] is those bytes, built eagerly at load.
//!
//! Why this is its own struct and not a field of
//! [`KvCache`](crate::kv::KvCache): the two change for different reasons, and
//! the difference is the whole API. The cache is **sequence state** — it changes
//! when the tokens change, and "should this be cleared?" means "am I starting a
//! new sequence?". This is **weight state** — it changes only when the file
//! changes, and "should this be cleared?" means "did I open a different
//! model?". Fold them into one struct and that one question has two answers,
//! every `begin` becomes a potential destroyer of derived weights, and a
//! cache-clear bug stops being a loud wrong-shape error and becomes quiet
//! wrong numbers. Separate types keep the lifetimes honest: `&Derived` for the
//! whole run of a model, `&mut KvCache` per step.
//!
//! Eager, not lazy, on purpose. A lazy build puts a branch — and, once
//! `matmul_q` grows threads, synchronization — on the decode hot path, which is
//! the same class of cost this module exists to remove, and it makes "which
//! step paid for the build" a question the profile answers differently every
//! run. `new` fills every block and every head before returning; the path
//! afterwards only ever sees `&Derived`. `size_bytes` and the decode binary
//! print the real size, and the build is one pass, once.

use crate::ModelError;
use crate::attn::{MlaParams, Q8Block, quantize_q8_0};
use gguf::{Gguf, TensorInfo, dequant_row};

/// Every block's `wk_b` as Q8_0 blocks, head-major — the bytes the per-step
/// absorption used to rebuild.
///
/// Blocks whose tensors are absent from the file hold `None`, permanently (not
/// lazily): the accessors refuse those indices with [`ModelError`] instead of
/// panicking, because "this file has no such weight" is a caller-facing fact,
/// not an invariant the indexing could check cheaply.
pub struct Derived {
    per_block: Vec<Option<BlockDerived>>,
}

/// One block's derived weights: every head's blocks concatenated, laid out
/// exactly as `q_nope2_absorbed` indexes them (`head·span` up front).
struct BlockDerived {
    blocks: Vec<Q8Block>,
    n_head: usize,
    /// Blocks per head: `latent * (nope / 32)`.
    span: usize,
}

impl Derived {
    /// Fill every block, every head. One dequant pass over the k-up rows of
    /// `attn_kv_b`; after this returns, the struct is read-only for the life of
    /// the model.
    pub fn new(gguf: &Gguf) -> Result<Derived, ModelError> {
        let n_block = gguf
            .block_count()
            .ok_or_else(|| ModelError::MissingTensor("metadata key block_count".into()))?
            as usize;
        let mut per_block = Vec::with_capacity(n_block);
        for b in 0..n_block {
            let Some(wkb) = gguf.find(&format!("blk.{b}.attn_kv_b.weight")) else {
                per_block.push(None);
                continue;
            };
            let p = MlaParams::read(gguf, b)?;
            per_block.push(Some(build_block(gguf, wkb, &p)?));
        }
        Ok(Derived { per_block })
    }

    /// Head `head`'s blocks for block `block`: `latent * (nope/32)` of them,
    /// running along the q_nope axis — the exact bytes the in-place build made
    /// (`tests/derived.rs` asserts that as byte equality, not a tolerance).
    pub fn wk_b_blocks(&self, block: usize, head: usize) -> Result<&[Q8Block], ModelError> {
        let bd = self.block(block)?;
        if head >= bd.n_head {
            return Err(ModelError::Shape {
                what: "derived wk_b head index",
                want_ne0: bd.n_head,
                want_ne1: 0,
                got_ne0: head,
                got_ne1: 0,
            });
        }
        Ok(&bd.blocks[head * bd.span..(head + 1) * bd.span])
    }

    /// Every head's blocks for block `block`, concatenated head-major — the
    /// slice [`q_nope2_absorbed`](crate::attn::q_nope2_absorbed) consumes.
    pub fn wk_b_all_heads(&self, block: usize) -> Result<&[Q8Block], ModelError> {
        Ok(&self.block(block)?.blocks)
    }

    /// How many blocks have derived weights (the rest hold no `attn_kv_b`).
    pub fn filled_blocks(&self) -> usize {
        self.per_block.iter().flatten().count()
    }

    /// Total bytes of Q8_0 blocks held — the number the decode binary prints.
    pub fn size_bytes(&self) -> usize {
        self.per_block
            .iter()
            .flatten()
            .map(|bd| bd.blocks.len() * std::mem::size_of::<Q8Block>())
            .sum()
    }

    fn block(&self, block: usize) -> Result<&BlockDerived, ModelError> {
        self.per_block
            .get(block)
            .and_then(|b| b.as_ref())
            .ok_or_else(|| ModelError::MissingTensor(format!("blk.{block}.attn_kv_b.weight")))
    }
}

/// The two loops this module owns, moved verbatim out of `q_nope2_absorbed`:
/// dequantize the head's k-up rows, then requant column `j`'s 32-value spans
/// along the q_nope axis into Q8_0. The op order is the contract —
/// `tests/derived.rs` byte-compares against an independent copy of these loops,
/// so any "equivalent" restructuring here is a gate event, not a refactoring.
fn build_block(gguf: &Gguf, wkb: &TensorInfo, p: &MlaParams) -> Result<BlockDerived, ModelError> {
    let bytes = gguf.data(wkb)?;
    let row_bytes =
        wkb.ty.type_size().unwrap() as usize * (p.latent / wkb.ty.blck_size().unwrap() as usize);
    let nblocks = p.nope / 32;
    let span = p.latent * nblocks;
    let mut blocks = Vec::with_capacity(p.n_head * span);
    let mut wk_row = vec![0.0f32; p.latent];

    for h in 0..p.n_head {
        // The head's k-up rows, dequantized once: rows[d][j] = kv_b row h·span+d, elem j.
        let mut rows = vec![0.0f32; p.nope * p.latent];
        for d in 0..p.nope {
            let off = (h * (p.nope + p.v_head) + d) * row_bytes;
            dequant_row(wkb.ty, &bytes[off..off + row_bytes], wk_row.as_mut_slice())?;
            rows[d * p.latent..(d + 1) * p.latent].copy_from_slice(&wk_row);
        }
        // Q8_0 blocks run along the q_nope axis: block (j, b) = 32 d-values of column j.
        let mut vals = [0.0f32; 32];
        for j in 0..p.latent {
            for b in 0..nblocks {
                for (l, v) in vals.iter_mut().enumerate() {
                    *v = rows[(32 * b + l) * p.latent + j];
                }
                blocks.push(quantize_q8_0(&vals));
            }
        }
    }
    Ok(BlockDerived {
        blocks,
        n_head: p.n_head,
        span,
    })
}
