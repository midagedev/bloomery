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
//! One sequence, in this order:
//!
//! 1. [`DraftBody::reset`] — zeroed rings, no committed position.
//! 2. [`DraftBody::append`] of the prompt's features in order, positions
//!    `max(0, P − window) .. P` — the rows the window-only layers keep; a
//!    batched prompt ([`crate::body::prefill_with`]) hands over only those,
//!    and [`DraftBody::skip`] moves the committed position past the rest (at
//!    most a window a call; more than [`kv::GROUP`] rows in one call run
//!    eager, once a sequence).
//! 3. [`DraftBody::propose`]`(id_last, w)`: `id_last` is the token at
//!    position [`DraftBody::committed`] — after the prompt, the token the
//!    target sampled from its last row, which the target has not run yet —
//!    and `w` in `1..=block::MAX_WIDTH` is the caller's pinned width (the
//!    draft has no default). Returns `w` ids, the first for the position
//!    after `id_last`.
//! 4. The target's verify pass over `[id_last, ids…]` at positions
//!    `committed ..= committed + w`: `w + 1` rows served one after another
//!    per layer, as its two-row pass (`bloomery_gpu::hybrid::Chain::Pair`)
//!    does today: the target's `moe.rs` serves each row's host experts on
//!    its own (`ne1 == 1`, `EXPERTS_INTO_MAX`). Consecutive rows share many
//!    of those experts, so a service that reads each distinct expert once
//!    per layer is an open design on the target side; the draft's contract
//!    does not depend on it.
//! 5. [`DraftBody::append`] of the features of the `k + 1` positions the
//!    verify pass accepted — `id_last` and the `k` leading ids that matched
//!    — then step 3 with the target's next token as `id_last`.
//!
//! A rejected proposal leaves nothing behind: only appended features enter
//! the rings. Every call enqueues on the engine stream; `append` returns
//! without waiting, and `propose` returns once its ids are on the host, so
//! a step costs the draft one host wait.
//!
//! # Submission
//!
//! [`DraftBody::new`] captures, once at load, one graph per width
//! `1..=block::MAX_WIDTH` of the whole block pass (widening, three layers,
//! head, Markov loop and its argmax) and one per row count
//! `1..=kv::GROUP` of the append. A call writes its inputs into a pinned
//! image, copies it to the card in one transfer right before the replay
//! (outside the graph, so the graphs hold kernels only and the copy's
//! source never enters one), and replays. The graphs are captured under
//! [`Rule::Reference`], the engine's rule: its clamp limits are launch
//! scalars, so an ik-rule pass runs eager ([`Submit::Eager`], the gates'
//! twin of every call).

pub mod block;
pub mod head;
pub mod kv;
pub mod load;
mod stage;

use std::sync::Arc;

use bloomery_gpu::{Gpu, GpuError, Graph};
use gguf::Split;

use block::{BlockInput, BlockPass, MAX_WIDTH, Rule, embedding_row};
use kv::{DraftRings, GROUP, KvAppend};
use load::DraftWeights;

const WHAT: &str = "draft::DraftBody";

/// How a [`DraftBody`] call reaches the card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Submit {
    /// Every launch enqueued one by one: the gates' twin of a replay.
    Eager,
    /// A replay of the graph captured at load for the call's shape.
    Graph,
}

/// The graphs [`DraftBody::new`] captures.
struct Graphs {
    /// The block pass of `m` rows at `m - 1`.
    pass: Vec<Graph>,
    /// The append of `n` rows at `n - 1`.
    append: Vec<Graph>,
}

/// The draft for one sequence: its weights, its rings, the captured passes
/// and the positions committed so far. See the module comment.
pub struct DraftBody {
    /// Declared first: dropped before the weights and buffers they name.
    graphs: Graphs,
    w: DraftWeights,
    rings: DraftRings,
    kv: KvAppend,
    pass: BlockPass,
    /// The target's file, for the embedding row of each block's first id.
    target: Arc<Split>,
    /// Positions appended so far: the next committed position.
    committed: u32,
    /// The latest positions [`DraftBody::skip`] passed without a row: a
    /// block that would read one of their slots is refused.
    skipped: Option<std::ops::Range<u32>>,
}

impl DraftBody {
    /// Rings, append buffers, the block pass over `w`, and every graph.
    /// Load-time only.
    pub fn new(gpu: &Gpu, w: DraftWeights, target: Arc<Split>) -> Result<DraftBody, GpuError> {
        let mut rings = DraftRings::new(gpu.stream(), w.hp())?;
        let mut kv = KvAppend::new(gpu, w.hp())?;
        let mut pass = BlockPass::new(gpu, &w)?;
        let counted = |g: Graph, launches: usize| {
            if g.node_count() == launches {
                Ok(g)
            } else {
                Err(GpuError::Shape {
                    what: WHAT,
                    detail: format!(
                        "a capture of {} nodes for {launches} launches",
                        g.node_count()
                    ),
                })
            }
        };
        let pass_graphs = (1..=MAX_WIDTH)
            .map(|m| counted(pass.capture(gpu, &w, &rings, m)?, pass.launches(m)))
            .collect::<Result<Vec<_>, _>>()?;
        let append = (1..=GROUP)
            .map(|n| counted(kv.capture(gpu, &w, &mut rings, n)?, kv.launches(n)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(DraftBody {
            graphs: Graphs {
                pass: pass_graphs,
                append,
            },
            w,
            rings,
            kv,
            pass,
            target,
            committed: 0,
            skipped: None,
        })
    }

    /// Commit positions `committed .. to` without their rows: a reader that
    /// keeps only a window's rows skips the older ones. Their ring slots
    /// hold nothing of theirs; a block that would read one before appended
    /// rows overwrite it is refused ([`DraftBody::propose`]). Refused for a
    /// `to` below the committed position.
    pub fn skip(&mut self, to: u32) -> Result<(), GpuError> {
        if to < self.committed {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "a skip to position {to} after {} committed positions",
                    self.committed
                ),
            });
        }
        if to > self.committed {
            self.skipped = Some(self.committed..to);
            self.committed = to;
        }
        Ok(())
    }

    /// Append the features of the next committed positions (`feats.len()`
    /// a multiple of the feature width, at most a window of rows): each at
    /// its own position and ring slot. Returns the rows appended. Enqueues
    /// on the engine stream and returns without waiting.
    pub fn append(&mut self, gpu: &Gpu, feats: &[f32]) -> Result<usize, GpuError> {
        self.append_as(gpu, feats, Submit::Graph)
    }

    /// [`DraftBody::append`], submitted as `submit` says. An append of more
    /// than [`GROUP`] rows has no graph and runs eager either way.
    pub fn append_as(
        &mut self,
        gpu: &Gpu,
        feats: &[f32],
        submit: Submit,
    ) -> Result<usize, GpuError> {
        let p = self.committed;
        let window = u32::try_from(self.w.hp().window).map_err(|_| GpuError::State {
            what: WHAT,
            missing: "a window that fits a u32",
        })?;
        let s = gpu.stream();
        let n = self.kv.stage(s, feats, p % window, p)?;
        let next = u32::try_from(n)
            .ok()
            .and_then(|n| p.checked_add(n))
            .ok_or(GpuError::State {
                what: WHAT,
                missing: "a next position that fits a u32",
            })?;
        match (submit, self.graphs.append.get(n - 1)) {
            (Submit::Graph, Some(g)) => g.launch(s)?,
            _ => self.kv.enqueue(gpu, &self.w, &mut self.rings)?,
        }
        self.committed = next;
        Ok(n)
    }

    /// Propose `width` ids after `id_last`, the token at position
    /// [`DraftBody::committed`], under the reference rule. Blocks until the
    /// ids are on the host.
    pub fn propose(&mut self, gpu: &Gpu, id_last: u32, width: usize) -> Result<Vec<u32>, GpuError> {
        self.propose_as(gpu, id_last, width, Submit::Graph)
    }

    /// [`DraftBody::propose`], submitted as `submit` says.
    pub fn propose_as(
        &mut self,
        gpu: &Gpu,
        id_last: u32,
        width: usize,
        submit: Submit,
    ) -> Result<Vec<u32>, GpuError> {
        if self.committed == 0 {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a committed position before the first block",
            });
        }
        let hp = self.w.hp();
        let seen = self.committed as usize - (self.committed as usize).min(hp.window);
        if let Some(gap) = self.skipped.as_ref().filter(|g| g.end as usize > seen) {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "a block after position {} reads the ring rows of positions {seen}.., and \
                     positions {gap:?} were skipped without their rows",
                    self.committed
                ),
            });
        }
        let row = embedding_row(&self.target, hp.n_embd, id_last)?;
        let input = BlockInput {
            id_last,
            row,
            width,
            first_pos: self.committed,
            ring_rows: (self.committed as usize).min(hp.window),
            rule: Rule::Reference,
        };
        run_block(
            gpu,
            &mut self.pass,
            &self.graphs,
            &self.w,
            &self.rings,
            &input,
            submit,
        )
    }

    /// One block pass over `input` as it is — its own positions, ring rows
    /// and rule — against the rings as they are: the gates' way to feed an
    /// oracle set's block. A graph replay takes [`Rule::Reference`] only.
    /// Blocks until the ids are on the host.
    pub fn block(
        &mut self,
        gpu: &Gpu,
        input: &BlockInput<'_>,
        submit: Submit,
    ) -> Result<Vec<u32>, GpuError> {
        run_block(
            gpu,
            &mut self.pass,
            &self.graphs,
            &self.w,
            &self.rings,
            input,
            submit,
        )
    }

    /// Forget the sequence: zeroed rings, no committed position.
    pub fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        self.rings.reset(gpu.stream())?;
        gpu.stream().synchronize()?;
        self.committed = 0;
        self.skipped = None;
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

    /// The rings.
    #[must_use]
    pub fn rings(&self) -> &DraftRings {
        &self.rings
    }

    /// The rings, writable: the gates' way to seat an oracle's rows. The
    /// graphs read them in place, so a replay sees what is seated.
    pub fn rings_mut(&mut self) -> &mut DraftRings {
        &mut self.rings
    }

    /// The captured pass of `m` rows.
    #[must_use]
    pub fn pass_graph(&self, m: usize) -> Option<&Graph> {
        m.checked_sub(1).and_then(|i| self.graphs.pass.get(i))
    }

    /// The captured append of `n` rows.
    #[must_use]
    pub fn append_graph(&self, n: usize) -> Option<&Graph> {
        n.checked_sub(1).and_then(|i| self.graphs.append.get(i))
    }

    /// Kernel launches a block pass of `m` rows makes.
    #[must_use]
    pub fn pass_launches(&self, m: usize) -> usize {
        self.pass.launches(m)
    }

    /// Kernel launches an append of `n` rows makes.
    #[must_use]
    pub fn append_launches(&self, n: usize) -> usize {
        self.kv.launches(n)
    }
}

/// Stage `input`, run its pass as `submit` says, read its ids.
fn run_block(
    gpu: &Gpu,
    pass: &mut BlockPass,
    graphs: &Graphs,
    w: &DraftWeights,
    rings: &DraftRings,
    input: &BlockInput<'_>,
    submit: Submit,
) -> Result<Vec<u32>, GpuError> {
    let s = gpu.stream();
    let graph = match submit {
        Submit::Eager => None,
        Submit::Graph if input.rule != Rule::Reference => {
            return Err(GpuError::State {
                what: WHAT,
                missing: "the reference rule for a graph pass (ik's rule runs eager)",
            });
        }
        Submit::Graph => Some(
            input
                .width
                .checked_sub(1)
                .and_then(|i| graphs.pass.get(i))
                .ok_or_else(|| GpuError::Shape {
                    what: WHAT,
                    detail: format!("width {}; 1..={MAX_WIDTH}", input.width),
                })?,
        ),
    };
    pass.stage(s, input)?;
    match graph {
        Some(g) => g.launch(s)?,
        None => pass.enqueue(gpu, w, rings)?,
    }
    pass.tokens(s)
}
