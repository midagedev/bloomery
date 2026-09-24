//! Weight-derived state: the `wk_b` Q8_0 requant and the step plan, built once
//! per file.
//!
//! The absorption in `q_nope2_absorbed` consumes these blocks, and computing
//! them is a pure function of the weights — nothing about the tokens enters —
//! so rebuilding them per block, per head, per decode step re-derived bytes the
//! file had already fixed. [`Derived`] is those bytes, built eagerly at load.
//! The same argument covers the rest of a step's token-independent work —
//! tensor-name formatting and table scans, metadata reads, norm-gain decodes,
//! per-head and per-expert view construction — which lives here as [`Plan`].
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

use super::attn::{MlaParams, Q8Block, quantize_q8_0};
use super::hparams::Hparams;
use super::names;
use crate::ModelError;
use crate::head::HeadPlan;
use crate::moe::MoeBlockPlan;
use crate::ops::f32_tensor;
use gguf::{Gguf, TensorInfo, dequant_row};

/// Every block's `wk_b` as Q8_0 blocks, head-major — the bytes the per-step
/// absorption used to rebuild — plus the step plan.
///
/// Blocks whose tensors are absent from the file hold `None`, permanently (not
/// lazily): the accessors refuse those indices with [`ModelError`] instead of
/// panicking, because "this file has no such weight" is a caller-facing fact,
/// not an invariant the indexing could check cheaply.
pub struct Derived {
    per_block: Vec<Option<BlockDerived>>,
    plan: Plan,
}

/// One block's derived weights: every head's blocks concatenated, laid out
/// exactly as `q_nope2_absorbed` indexes them (`head·span` up front).
struct BlockDerived {
    blocks: Vec<Q8Block>,
    n_head: usize,
    /// Blocks per head: `latent * (nope / 32)`.
    span: usize,
}

/// Every token-independent lookup a decode step makes, resolved once: per-block
/// weight views and decoded gains, the head, the embedding table. The step
/// reads this and never formats a tensor name, scans the tensor table or walks
/// the metadata keys.
///
/// Read-only after `Derived::new` by construction — plain fields, no interior
/// mutability, no lazy fill (see the module doc for why that is the contract).
/// The tensors are byte-identical clones of what `Gguf::find` returned, so
/// `gguf.data`'s per-read bounds check keeps guarding them.
pub struct Plan {
    /// The file's hyperparameters, read and refused before anything else.
    pub hparams: Hparams,
    pub blocks: Vec<BlockPlan>,
    pub head: HeadPlan,
    /// The architecture-wide rms epsilon (`forward::rms_eps`'s key).
    pub eps: f32,
    /// `token_embd.weight` — the embedding lookup's view.
    pub embed: TensorInfo,
    /// The file this plan was built from, as tensor count and data base: a
    /// `Gguf` that disagrees is not the file these offsets belong to.
    origin: (usize, u64),
}

/// One block's slice of the plan.
pub struct BlockPlan {
    pub attn: AttnBlockPlan,
    pub ffn: FfnPlan,
    /// `blk.{b}.attn_norm.weight`, decoded.
    pub attn_gain: Vec<f32>,
    /// `blk.{b}.ffn_norm.weight`, decoded.
    pub ffn_gain: Vec<f32>,
    /// Whether the block routes to experts — decided by the file, the same way
    /// `plan.rs` decides between the dense and the shared-expert trio: presence
    /// of `ffn_gate_inp`, never a block number.
    pub routed: bool,
}

impl BlockPlan {
    /// The MoE plan; `Err` on a dense block, which has no experts to name.
    pub fn moe(&self) -> Result<&MoeBlockPlan, ModelError> {
        match &self.ffn {
            FfnPlan::Moe(p) => Ok(p),
            FfnPlan::Dense(_) => Err(ModelError::MissingTensor(
                "dense block has no MoE plan".into(),
            )),
        }
    }

    /// The dense trio; panics on a routed block, which `routed` decides first.
    pub fn dense(&self) -> (&TensorInfo, &TensorInfo, &TensorInfo) {
        match &self.ffn {
            FfnPlan::Dense(t) => (&t.gate, &t.up, &t.down),
            FfnPlan::Moe(_) => panic!("dense() on a routed block"),
        }
    }
}

/// One block's attention in the plan: geometry, the four weight views, the
/// decoded `attn_kv_a_norm` gain and the per-head v_up views `wv_b_heads_with`
/// consumes.
pub struct AttnBlockPlan {
    /// `MlaParams::read` for this block, cross-checks included.
    pub params: MlaParams,
    pub wq: TensorInfo,
    pub wa: TensorInfo,
    /// `attn_kv_b` — the base the v_up views were cut from.
    pub wkb: TensorInfo,
    pub wo: TensorInfo,
    /// `blk.{b}.attn_kv_a_norm.weight`, decoded.
    pub kv_a_norm_gain: Vec<f32>,
    /// One v_up view per head (`attn::v_up_views`).
    pub v_up_views: Vec<TensorInfo>,
}

/// The FFN half of a block: the dense trio or the MoE plan.
pub enum FfnPlan {
    Dense(DenseTrio),
    Moe(MoeBlockPlan),
}

/// The (gate, up, down) trio [`plan::ffn_weights`](super::plan::ffn_weights)
/// selected, owned.
pub struct DenseTrio {
    pub gate: TensorInfo,
    pub up: TensorInfo,
    pub down: TensorInfo,
}

impl Plan {
    /// The plan's tensors are offsets into one file; stepping a `Gguf` they
    /// were not built from would read that file's bytes at this file's
    /// offsets. `Gguf::data` bounds-checks every read (a differently-sized
    /// file errors instead of panicking — the existing guard for stale views),
    /// and this closes the same-shaped-file case: the stepped file's tensor
    /// count and data base must be the ones the plan was built from.
    pub(crate) fn check_origin(&self, gguf: &Gguf) -> Result<(), ModelError> {
        if gguf.tensor_count() == self.origin.0 && gguf.data_base() == self.origin.1 {
            return Ok(());
        }
        Err(ModelError::MissingTensor(format!(
            "derived plan was built for a different file ({} tensors, data base {})",
            self.origin.0, self.origin.1
        )))
    }
}

impl Derived {
    /// Fill every block, every head, and resolve the whole step plan. One
    /// dequant pass over the k-up rows of `attn_kv_b` plus one pass of finds,
    /// metadata reads and view builds; after this returns, the struct is
    /// read-only for the life of the model.
    pub fn new(gguf: &Gguf) -> Result<Derived, ModelError> {
        // `ModelError` has no metadata variant; the refusal's own text names
        // the key and the value, as `MlaParams::read`'s rope refusal does.
        let hparams = Hparams::read(gguf).map_err(|e| ModelError::MissingTensor(e.to_string()))?;
        let n_block = gguf
            .block_count()
            .ok_or_else(|| ModelError::MissingTensor("metadata key block_count".into()))?
            as usize;
        let embed = gguf
            .find("token_embd.weight")
            .ok_or_else(|| ModelError::MissingTensor("token_embd.weight".into()))?
            .clone();
        let embd = embed.dims[0] as usize;
        let eps = super::forward::rms_eps(gguf);
        // deepseek2 carries one architecture-wide rms eps; the file has no separate
        // final-norm key. The 1e-4 gate on `result_norm` is the numeric proof that
        // this key is the one the final norm runs with.
        let head = HeadPlan::new(gguf, eps)?;
        let mut per_block = Vec::with_capacity(n_block);
        let mut blocks = Vec::with_capacity(n_block);
        for b in 0..n_block {
            // `MlaParams::read` is the authority on whether a block can run at
            // all: its finds and cross-checks fail here, at load, with the same
            // errors the first step used to return.
            let p = MlaParams::read(gguf, b)?;
            let kv_b = names::attn_kv_b(b);
            let wkb = gguf
                .find(&kv_b)
                .ok_or_else(|| ModelError::MissingTensor(kv_b.clone()))?;
            let find = |name: String| -> Result<TensorInfo, ModelError> {
                gguf.find(&name)
                    .cloned()
                    .ok_or(ModelError::MissingTensor(name))
            };
            let attn = AttnBlockPlan {
                params: p.clone(),
                wq: find(names::attn_q(b))?,
                wa: find(names::attn_kv_a_mqa(b))?,
                wkb: wkb.clone(),
                wo: find(names::attn_output(b))?,
                kv_a_norm_gain: f32_tensor(
                    gguf,
                    gguf.find(&names::attn_kv_a_norm(b))
                        .ok_or_else(|| ModelError::MissingTensor(names::attn_kv_a_norm(b)))?,
                )?,
                v_up_views: super::attn::v_up_views(wkb, &p)?,
            };
            let gain = |name: String| -> Result<Vec<f32>, ModelError> {
                f32_tensor(
                    gguf,
                    gguf.find(&name)
                        .ok_or_else(|| ModelError::MissingTensor(name.clone()))?,
                )
            };
            let routed = gguf.find(&names::ffn_gate_inp(b)).is_some();
            let ffn = if routed {
                FfnPlan::Moe(super::plan::moe_block_plan(gguf, b, embd)?)
            } else {
                let (gate, up, down) = super::plan::ffn_weights(gguf, b)?;
                FfnPlan::Dense(DenseTrio {
                    gate: gate.clone(),
                    up: up.clone(),
                    down: down.clone(),
                })
            };
            per_block.push(Some(build_block(gguf, wkb, &p)?));
            blocks.push(BlockPlan {
                attn,
                ffn,
                attn_gain: gain(names::attn_norm(b))?,
                ffn_gain: gain(names::ffn_norm(b))?,
                routed,
            });
        }
        let plan = Plan {
            hparams,
            blocks,
            head,
            eps,
            embed,
            origin: (gguf.tensor_count(), gguf.data_base()),
        };
        Ok(Derived { per_block, plan })
    }

    /// The step plan — everything a step reads that the tokens cannot change.
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Block `block`'s plan; out-of-range is a caller-facing fact.
    pub fn block_plan(&self, block: usize) -> Result<&BlockPlan, ModelError> {
        self.plan
            .blocks
            .get(block)
            .ok_or_else(|| ModelError::MissingTensor(format!("block {block}")))
    }

    /// Block `block`'s attention plan.
    pub fn attn_plan(&self, block: usize) -> Result<&AttnBlockPlan, ModelError> {
        Ok(&self.block_plan(block)?.attn)
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
    /// slice [`q_nope2_absorbed`](super::attn::q_nope2_absorbed) consumes.
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
            .ok_or_else(|| ModelError::MissingTensor(names::attn_kv_b(block)))
    }
}

/// The two loops this module owns, moved verbatim out of `q_nope2_absorbed`:
/// dequantize the head's k-up rows, then requant column `j`'s 32-value spans
/// along the q_nope axis into Q8_0. The op order is the contract —
/// `tests/derived.rs` byte-compares against an independent copy of these loops,
/// so any "equivalent" restructuring here is a gate event, not a refactoring.
fn build_block(gguf: &Gguf, wkb: &TensorInfo, p: &MlaParams) -> Result<BlockDerived, ModelError> {
    let bytes = gguf.data(wkb)?;
    let row_bytes = super::attn::wkb_row_bytes(wkb, p.latent)?;
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
