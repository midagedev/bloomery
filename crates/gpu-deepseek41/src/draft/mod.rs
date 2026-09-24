//! The DSpark draft on its own card: the draft file's tensors and the two
//! the target lends it ([`load`]), the feature-to-KV graph that turns the
//! target's committed hidden states into each draft layer's window ring
//! ([`kv`]), the block pass over `[id_last, mask × (w − 1)]` ([`block`]) and
//! its head with the Markov loop ([`head`]); [`DraftBody`] drives them.
//!
//! The draft is `model::arch::dspark`'s file: three window-only V4.1-shaped
//! blocks behind `fc`, which projects the target's hidden states at
//! `target_layers`. Everything here reuses kernels the target or the draft
//! kernel rounds already own; this module adds no `#[kernel]`.
//!
//! The feature of one committed target position is the mean of the four
//! hyper-connection streams leaving each target layer `target_layers[i] - 1`
//! (the input of layer `target_layers[i]`), the three means concatenated in
//! `target_layers` order: `hp.target_layers.len() · n_embd` f32 per position.
//!
//! # Positions
//!
//! The reference `model.py`'s rule (`DSparkAttention.forward`): the committed
//! row of target position `p` turns at rope position `p` and sits in ring
//! slot `p % window`; a block after the last committed position `p` turns
//! row `j` at `p + 1 + j`, and every block row sees the whole ring (its
//! first `min(p + 1, window)` rows) and every row of the block. ik's oracle
//! turns every committed row after the prompt at `p + 1` and its block one
//! further on; the gates feed an oracle set its own positions, the engine
//! never does.
//!
//! # Contract for the target loop
//!
//! [`DraftBody::append`] takes the features of the positions the target
//! committed, in order, at most a window at a time; [`DraftBody::propose`]
//! takes the last accepted token and a width `w` in `1..=block::MAX_WIDTH`
//! and returns `w` proposed ids, the first for the position after the
//! accepted token; [`DraftBody::reset`] starts a new sequence. A rejected
//! proposal leaves nothing behind: only appended features enter the rings.
//! The width is the caller's choice and a pinned value where it is
//! measured; the draft has no default.
//!
//! The target verifies a proposal as its two-row pass does today
//! (`bloomery_gpu::hybrid::Chain::Pair`), generalised to `w + 1` rows served
//! one after another per layer. A merged multi-row host expert kernel is
//! ruled out — two rows' routed experts overlap far below that kernel's
//! break-even (measured; rig-log holds the numbers) — so the target's
//! `moe.rs` keeps its one-row expert service (`ne1 == 1`) and its
//! `EXPERTS_INTO_MAX`.

pub mod block;
pub mod head;
pub mod kv;
pub mod load;

use std::sync::Arc;

use bloomery_gpu::{Gpu, GpuError};
use gguf::Split;

use block::{BlockInput, BlockPass, MAX_WIDTH, Rule, embedding_row};
use kv::{DraftRings, KvAppend};
use load::DraftWeights;

const WHAT: &str = "draft::DraftBody";

/// The draft for one sequence: its weights, its rings and the positions
/// committed so far. See the module comment's contract.
pub struct DraftBody {
    w: DraftWeights,
    rings: DraftRings,
    kv: KvAppend,
    pass: BlockPass,
    /// The target's file, for the embedding row of each block's first id.
    target: Arc<Split>,
    /// Positions appended so far: the next committed position.
    committed: u32,
}

impl DraftBody {
    /// Rings, append buffers and the block pass over `w`. Load-time only.
    pub fn new(gpu: &Gpu, w: DraftWeights, target: Arc<Split>) -> Result<DraftBody, GpuError> {
        let rings = DraftRings::new(gpu.stream(), w.hp())?;
        let kv = KvAppend::new(gpu, w.hp())?;
        let pass = BlockPass::new(gpu, &w)?;
        Ok(DraftBody {
            w,
            rings,
            kv,
            pass,
            target,
            committed: 0,
        })
    }

    /// Append the features of the next committed positions (`feats.len()`
    /// a multiple of the feature width, at most a window of rows): each at
    /// its own position and ring slot. Returns the rows appended. Runs to
    /// completion on the engine stream.
    pub fn append(&mut self, gpu: &Gpu, feats: &[f32]) -> Result<usize, GpuError> {
        let p = self.committed;
        let window = u32::try_from(self.w.hp().window).map_err(|_| GpuError::State {
            what: WHAT,
            missing: "a window that fits a u32",
        })?;
        let n = self.kv.stage(gpu.stream(), feats, p % window, p)?;
        let next = u32::try_from(n)
            .ok()
            .and_then(|n| p.checked_add(n))
            .ok_or(GpuError::State {
                what: WHAT,
                missing: "a next position that fits a u32",
            })?;
        self.kv.enqueue(gpu, &self.w, &mut self.rings)?;
        gpu.stream().synchronize()?;
        self.committed = next;
        Ok(n)
    }

    /// Propose `width` ids after `id_last`, the token at position
    /// [`DraftBody::committed`]. Runs to completion on the engine stream.
    pub fn propose(&mut self, gpu: &Gpu, id_last: u32, width: usize) -> Result<Vec<u32>, GpuError> {
        if self.committed == 0 {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a committed position before the first block",
            });
        }
        if !(1..=MAX_WIDTH).contains(&width) {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("width {width}; 1..={MAX_WIDTH}"),
            });
        }
        let hp = self.w.hp();
        let row = embedding_row(&self.target, hp.n_embd, id_last)?;
        let input = BlockInput {
            id_last,
            row,
            width,
            first_pos: self.committed,
            ring_rows: (self.committed as usize).min(hp.window),
            rule: Rule::Reference,
        };
        self.pass.stage(gpu.stream(), &input)?;
        self.pass.enqueue(gpu, &self.w, &self.rings)?;
        let ids = self.pass.tokens(gpu.stream())?;
        Ok(ids)
    }

    /// Forget the sequence: zeroed rings, no committed position.
    pub fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        self.rings.reset(gpu.stream())?;
        gpu.stream().synchronize()?;
        self.committed = 0;
        Ok(())
    }

    /// Positions committed so far.
    #[must_use]
    pub fn committed(&self) -> u32 {
        self.committed
    }

    /// The weights.
    #[must_use]
    pub fn weights(&self) -> &DraftWeights {
        &self.w
    }

    /// The block pass, as the last proposal left it.
    #[must_use]
    pub fn pass(&self) -> &BlockPass {
        &self.pass
    }
}
