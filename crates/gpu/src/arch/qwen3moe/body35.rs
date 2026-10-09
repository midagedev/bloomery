//! The Qwen3.6-35B-A3B (`qwen35moe`) and Qwen3.5-9B (`qwen35`, the Clef
//! backbone) `ChainBody`: the family's one layer body (`dispatch::layer`)
//! over plans that interleave gated-delta-rule layers with gated GQA layers
//! at head 256 — every routed layer's experts carrying the sigmoid-gated
//! shared expert as one more slot of joined stacks, every dense layer's FFN a
//! stack of one expert on the arena's fixed one-slot route. The head counts
//! are the file's; their group selects the flash (eight a block, or pairs).
//!
//! Load reads the file's description once (`model::arch::qwen35moe`): each
//! layer's kind from its tensors, never from its number, checked against
//! the geometry the kernels take; each layer's shared expert joined into its
//! routed stacks and its gate into the router (`Weights::join_rows`: the
//! parts leave the map, nothing is resident twice); the stores — K/V planes
//! for an attention layer, a recurrent state and conv ring for a delta
//! layer; three arenas — the decode step's one row, a pass's
//! [`MAX_PASS_ROWS`] and the prompt call's ubatch (allocated last, after a
//! check that it fits the card: [`Open35::ubatch`]); and the input records
//! the captured chains read, and the prompt image the prompt call reads.
//!
//! A step's and a pass's record, and the prompt image, carry the lane word:
//! the state lane every delta launch reads and writes, a device word and
//! never a launch argument, so one capture serves every position. A load
//! holds one lane, and the word is [`LANE`] in every record. `reset` zeroes
//! every lane and every conv ring: the recurrence is written in place, so
//! the next sequence would read the previous one's state.
//!
//! The prompt call ([`GpuModel::prefill_with`]) cuts a prompt by
//! [`PrefillPlan`] and walks each unit through the layer program over the
//! ubatch arena, eager in either step mode; a unit of more than
//! [`GEMV_COLS`] rows takes each op's wide arm (`wide`), one of at most
//! that many the gemv arm, bit for bit the decode steps. The last unit ends
//! in its last row's head ([`Tail::Last`]). The hidden-state call
//! ([`GpuModel::prefill_hidden`]) walks the same units and ends each in the
//! final norm of its rows ([`Tail::Hidden`]).
//!
//! A delta layer's state is written in place, so a cut behind the fed
//! positions finds it only in a copy taken at the cut's position: a load
//! that takes checkpoints ([`Body35::set_checkpoints`]) walks a prompt as
//! the runs between its marks ([`Body35::marked`]), a checkpoint of every
//! delta layer's state and conv ring copied to a pinned host slot at each,
//! and a cut to one of them plans the copy back for the next call
//! ([`Body35::cut`], [`Rollback`]). The K/V planes are cut by position, as
//! [`Body`]'s are.
//!
//! A load holds several resident sequences ([`Slots`]): each one its
//! layers' stores, the positions its recurrent stores hold and its
//! checkpoints ([`Slot35`]), exchanged by pointer on a select. The body
//! records no capture of its own: the step's and the passes' captures are
//! the model's, kept a slot, and the prompt call runs eager. A placed load's
//! placed side holds no sequence's state ([`Slots::new_seq`]), so it serves
//! every slot as it is. A whole-card load also runs several slots' rows as
//! one pass (`SlotRows`, `body35_slots`): each slot's record — its first
//! position and its ids, beside one lane word every slot's rows read — and
//! each slot's own stores, bound by the slot; a placed load refuses it.

use super::body::{Kernels, KvQ8, TapRows, f32_site};
use super::dispatch::{self, PassCtx};
use super::head_argmax::HeadArgmaxState;
use super::image::{ImageWrite, PromptImage};
use super::placed::{BatchWalk, Placed, PlacedOpen, StepWalk, WalkParts, placed_bytes};
use super::plan::{
    DeltaPlan, FfnPlan, FfnRoute, Flash, Form, GqaKind, GqaPlan, Kind35, LayerPlan, MixerPlan,
    SharedPlan, SiteTy, kinds35, moe_fits, q35,
};
use super::prefill::{PrefillPath, PrefillPlan, PrefillStep, WIDE_FROM};
use super::program::{Program, Tail};
use super::scratch::{
    Arena, Dims, Forms, IN_IDS, IN_POS0, Inbox, Io, KvPlanes, LANE, LayerStore, RecStore, RopeRows,
    StepParams, Wants, param_view, put_input,
};
use super::slot_pass::SlotIn;
use super::ubatch::UBATCH;
use super::wide::{GEMV_COLS, arena_bytes};
use crate::checkpoint::Checkpoints;
use crate::flash_gqa::HEAD_256;
use crate::head::Head;
use crate::host::{BatchLeg, StepLeg};
use crate::hybrid::Chain;
use crate::linear::{self, LinearShape};
use crate::model::{
    ChainBody, GpuModel, Instrumented, MAX_PASS_ROWS, Rollback, Rows, SlotRange, Slots, block_count,
};
use crate::rope_table::{RopeSpec, RopeTable};
use crate::site::{self, Order, file_site};
use crate::tensor::window;
use crate::weights::Weights;
use crate::{Gpu, GpuError, launch_u32};
use bloomery_levers::HostCfg;
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::Arch;
use model::arch::models::shape::MoeShape;
use model::arch::models::{Ffn, LayerSpec, Mixer, ModelSpec};
use model::placement::{Plan, WholeLoad};
use runtime::seqstate::{HOST_BUDGET, Kept};
use std::mem::ManuallyDrop;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

const WHAT: &str = "qwen35moe::Body35::load";

/// Card bytes a load leaves free past the ubatch arena: what the load and
/// the first prompt still allocate after it — the output head and the
/// heads of a pass of up to [`MAX_PASS_ROWS`] rows (a vocabulary of logits
/// each, some 8 MB together), the captured step and passes, and the
/// allocator's rounding of the arena's some forty buffers to its 2 MiB
/// pages (at most 80 MiB) — with the rest as margin.
pub const FIT_RESERVE: usize = 256 << 20;

/// What [`GpuModel::<Body35>::open`](GpuModel::open) takes besides the card
/// and the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Open35 {
    /// KV rows the caches hold.
    pub ctx: usize,
    /// The decode flash: the tensor-core pass when `true`.
    pub mma: bool,
    /// Tokens per ubatch of the prompt call, `1..=` the router's
    /// [`ubatch`](super::router::RouterDims::ubatch) (at most [`UBATCH`]);
    /// the ubatch arena holds `max(ubatch, GEMV_COLS)` rows, at most `ctx`.
    /// A size outside that range, or an arena that does not fit the card's
    /// free bytes with [`FIT_RESERVE`] left, is refused by name at load.
    pub ubatch: usize,
    /// The attention layers' KV planes' format ([`KvQ8`]); the delta layers'
    /// recurrent stores are f32 either way.
    pub kv: KvQ8,
}

/// Rows of the ubatch arena for ubatches of `u` tokens on a `ctx`-row cache:
/// at least a pass's [`GEMV_COLS`] (the plan's passes and tail walk it too),
/// at most the cache.
fn prompt_rows(u: usize, ctx: usize) -> usize {
    u.max(GEMV_COLS).min(ctx)
}

/// `o` as a placed load opens it: its prompt runs as passes of up to
/// [`MAX_PASS_ROWS`] positions through the host tier's batch port, so its
/// ubatch arena holds that many rows, whatever `o.ubatch` asks.
fn placed_open(o: Open35) -> Open35 {
    Open35 {
        ubatch: MAX_PASS_ROWS,
        ..o
    }
}

/// `u` as a ubatch size of `d`, or a named refusal outside
/// `1..=d.ubatch_most()`.
fn ubatch_of(d: &Dims, u: usize) -> Result<NonZeroUsize, GpuError> {
    let most = d.ubatch_most();
    NonZeroUsize::new(u)
        .filter(|u| u.get() <= most)
        .ok_or_else(|| {
            GpuError::shape(
                "qwen35moe::ubatch",
                format!(
                    "a ubatch of {u} tokens (1..={most}: at most {UBATCH}, and {} slots a token \
                     in one route table)",
                    d.slots()
                ),
            )
        })
}

/// The spacing of a prompt call's inner checkpoints (`runtime::seqstate`'s
/// marks), when the load takes them ([`Body35::set_checkpoints`]): the
/// ubatch size the load was asked for, as [`ubatch_of`] validates it — the
/// ubatch walk's most positions, the sibling body's rule — so the marks
/// fall on the walk's multiples and a mark never cuts a ubatch walk the
/// call's own ubatches do not already cut: a cut re-feeds at most one
/// ubatch of rows, and a prompt call takes at most one walk more than its
/// own ubatches. A denser spacing takes a copy of every store inside each
/// smaller window of the prompt's own walk without saving a walk; a sparser
/// one trades the copies away for rows every cut repeats. A placed load's
/// prompt runs as passes of a few rows instead ([`placed_open`]), and the
/// asked size stays the spacing: its marks cut a pass at most once a mark,
/// a copy a window apart. The size [`ubatch_of`] returns is at most
/// [`UBATCH`], so the `u32` loses nothing.
fn checkpoint_every(asked: usize, d: &Dims) -> Result<u32, GpuError> {
    Ok(ubatch_of(d, asked)?.get() as u32)
}

/// The rows and the device bytes of the ubatch arena of `d` for ubatches of
/// `u` tokens ([`prompt_rows`], [`arena_bytes`]): what [`ubatch_arena`]
/// checks against the card's free bytes and allocates, and what
/// [`GpuModel::whole_load`] hands a whole-fit verdict.
fn ubatch_need(d: &Dims, forms: Forms, u: usize) -> (usize, usize) {
    let rows = prompt_rows(u, d.ctx);
    (rows, arena_bytes(d, rows, forms))
}

/// The ubatch arena of `d` for ubatches of `u` tokens, after the check that
/// it fits: its bytes ([`ubatch_need`]) and [`FIT_RESERVE`] within the
/// card's `free` bytes, else a named refusal with the free bytes, the need
/// and the largest ubatch that fits, before anything is allocated. The
/// arena it allocates holds the bytes the check counted, or the load fails
/// by name. Load-time allocation.
fn ubatch_arena(
    stream: &CudaStream,
    (d, forms): (Dims, Forms),
    u: usize,
    free: usize,
) -> Result<Arena, GpuError> {
    const WHAT_FIT: &str = "qwen35moe::ubatch_arena";
    let (rows, need) = ubatch_need(&d, forms, u);
    if need + FIT_RESERVE > free {
        let fits = (1..u)
            .rev()
            .find(|&v| ubatch_need(&d, forms, v).1 + FIT_RESERVE <= free)
            .map_or("no ubatch size fits".to_string(), |v| {
                format!("ubatches of {v} tokens fit")
            });
        return Err(GpuError::shape(
            WHAT_FIT,
            format!(
                "the ubatch arena for ubatches of {u} tokens ({rows} rows) needs {need} bytes and \
                 {FIT_RESERVE} more stay free for what the load allocates after it; the card has \
                 {free} free: {fits}"
            ),
        ));
    }
    let a = Arena::with(stream, d, rows, forms)?;
    if a.bytes() != need {
        return Err(GpuError::shape(
            WHAT_FIT,
            format!(
                "the ubatch arena holds {} bytes, its fit check counted {need}",
                a.bytes()
            ),
        ));
    }
    Ok(a)
}

/// The ubatch arena of `d` for ubatches of `u` tokens on a placed load,
/// after the check that it and the placed side's `side` bytes
/// ([`placed_bytes`]) fit the `counted` bytes the load's plan holds for them
/// in the card's scratch — not the card's free bytes, which a card budget
/// (`BLOOMERY_CARD_BUDGET`) does not bound — else a named refusal with both
/// terms, before anything is allocated. Load-time allocation.
fn counted_arena(
    stream: &CudaStream,
    (d, forms): (Dims, Forms),
    u: usize,
    counted: u64,
    side: u64,
) -> Result<Arena, GpuError> {
    const WHAT_FIT: &str = "qwen35moe::counted_arena";
    let rows = prompt_rows(u, d.ctx);
    let need = arena_bytes(&d, rows, forms) as u64;
    if need + side > counted {
        return Err(GpuError::shape(
            WHAT_FIT,
            format!(
                "the ubatch arena for ubatches of {u} tokens ({rows} rows) needs {need} bytes and \
                 the placed side {side}; the plan counts {counted}"
            ),
        ));
    }
    let a = Arena::with(stream, d, rows, forms)?;
    if a.bytes() as u64 != need {
        return Err(GpuError::shape(
            WHAT_FIT,
            format!(
                "the ubatch arena holds {} bytes, its fit check counted {need}",
                a.bytes()
            ),
        ));
    }
    Ok(a)
}

/// Lanes of every recurrent store: one, read and written in place.
pub(super) const LANES: usize = 1;

/// Qwen3.6's per-replay host values: the token and its position.
pub struct DecodeInput35 {
    token: u32,
    pos: u32,
}

/// Every Qwen3.6 layer's kind, as a gate lists them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind35 {
    /// Gated GQA at head 256 over the layer's K/V planes.
    Attention,
    /// The gated delta rule over the layer's recurrent store.
    Delta,
}

/// A pass's input record — its first position, up to [`MAX_PASS_ROWS`] ids
/// and the lane word — and the windows a captured pass of `m` rows reads.
pub(super) struct RowsParams {
    /// Ids `0..m` at `m − 1`.
    ids: Vec<ManuallyDrop<DeviceBuffer<u32>>>,
    pos0: ManuallyDrop<DeviceBuffer<u32>>,
    lane: ManuallyDrop<DeviceBuffer<u32>>,
    /// The rows the last write planned.
    planned: Option<usize>,
    inbox: Inbox,
}

/// The lane word's offset in a pass's record, after its ids.
const RP_LANE: usize = IN_IDS + MAX_PASS_ROWS;
/// Words of a pass's record.
const RP_WORDS: usize = RP_LANE + 1;

impl RowsParams {
    fn new(stream: &CudaStream) -> Result<RowsParams, GpuError> {
        let inbox = Inbox::new(stream, RP_WORDS)?;
        // SAFETY: every window lies inside the inbox's `RP_WORDS` device
        // words (`IN_IDS + m <= RP_LANE < RP_WORDS`, `IN_POS0 < RP_WORDS`),
        // and the inbox moves into the struct beside them (a move of the
        // handle, not of the allocation), where it outlives them.
        let (ids, pos0, lane) = unsafe {
            (
                (1..=MAX_PASS_ROWS)
                    .map(|m| param_view::<u32>(inbox.dev(), IN_IDS, m))
                    .collect(),
                param_view::<u32>(inbox.dev(), IN_POS0, 1),
                param_view::<u32>(inbox.dev(), RP_LANE, 1),
            )
        };
        Ok(RowsParams {
            ids,
            pos0,
            lane,
            planned: None,
            inbox,
        })
    }

    /// Write `tokens` from position `pos` and the lane word, and enqueue the
    /// record's copy. Asynchronous: the pass behind it reads it.
    pub(super) fn write(
        &mut self,
        stream: &CudaStream,
        tokens: &[u32],
        pos: u32,
    ) -> Result<(), GpuError> {
        if !(1..=MAX_PASS_ROWS).contains(&tokens.len()) {
            return Err(GpuError::shape(
                "qwen35moe::RowsParams::write",
                format!("{} rows; a pass takes 1..={MAX_PASS_ROWS}", tokens.len()),
            ));
        }
        let host = self.inbox.host_mut()?;
        put_input(host, tokens, pos)?;
        host[RP_LANE] = LANE;
        self.inbox.upload(stream, RP_WORDS)?;
        self.planned = Some(tokens.len());
        Ok(())
    }

    /// The input of a pass of `m` rows, as its first launch reads it.
    pub(super) fn io(&self, m: usize) -> Result<Io<'_>, GpuError> {
        let ids = m
            .checked_sub(1)
            .and_then(|i| self.ids.get(i))
            .ok_or_else(|| {
                GpuError::shape(
                    "qwen35moe::RowsParams::io",
                    format!("a pass of {m} rows (1..={MAX_PASS_ROWS})"),
                )
            })?;
        Ok(Io {
            ids,
            pos0: &self.pos0,
            first: 0,
            lane: Some(&self.lane),
        })
    }

    fn bytes(&self) -> usize {
        self.inbox.bytes()
    }
}

/// Everything Qwen3.6's chain owns on the device.
pub struct Body35 {
    pub(super) vocab: usize,
    pub(super) eps: f32,
    pub(super) plans: Vec<LayerPlan>,
    pub(super) stores: Vec<LayerStore>,
    /// The rope table every attention layer reads by position, 64 f32 a row.
    pub(super) rope: RopeRows,
    /// The decode step's one-row arena and its input record.
    pub(super) s: Arena,
    pub(super) sp: StepParams,
    /// A pass's arena of [`MAX_PASS_ROWS`] rows and its input record.
    pub(super) a: Arena,
    pub(super) rp: RowsParams,
    /// A pass of several slots' input records ([`SlotIn`]: each busy slot's
    /// first position and ids), the lane word every slot's rows read
    /// ([`LANE`], the one lane of each slot's own store), and the ranges the
    /// last plan of such a pass wrote them for, which an eager pass must
    /// match (`body35_slots`).
    pub(super) slot_in: SlotIn,
    pub(super) slot_lane: DeviceBuffer<u32>,
    pub(super) slots_planned: Option<Vec<SlotRange>>,
    /// The prompt call's ubatch arena, its image, and the ubatch size.
    pub(super) u: Arena,
    pub(super) img: PromptImage,
    pub(super) ubatch: NonZeroUsize,
    pub(super) k: Kernels,
    pub(super) head_state: HeadArgmaxState,
    /// The flash pass the chain runs: the tensor-core pass from load on.
    pub(super) mma: bool,
    pub(super) taps: Option<TapRows>,
    /// A placed load's placed side ([`Placed`]): the host tier and what the
    /// walks over its legs read; `None` on a whole-card load.
    pub(super) placed: Option<Placed>,
    /// Positions the recurrent stores hold: a call's end once its launches
    /// are enqueued (the step, a pass, a prompt unit). The next position is
    /// the model's (`GpuModel::pos`); the two differ only after a call
    /// failed past its launch, and a call at the model's position is then
    /// refused ([`Body35::stores_at`]).
    held: u32,
    /// The delta layers' stores on the host at chosen positions: the points
    /// a cut behind the fed positions restores. A prompt call takes them
    /// only while `marks` is on.
    ckpt: Checkpoints,
    /// The checkpoints' spacing the load chose ([`checkpoint_every`]): every
    /// sequence's checkpoints take it ([`Slots::new_seq`]).
    every: u32,
    /// Whether a prompt call takes checkpoints: the load's, not a
    /// sequence's — the seat arms it once, and every sequence's prompt calls
    /// then take their own.
    marks: bool,
    /// Walks the last prompt call ran ([`Body35::prompt_walks`]): one a
    /// unit, counted from the call's start, whichever sequence ran it.
    prompt_walks: usize,
}

/// One Qwen3.6 sequence's own state ([`Slots`], what a select exchanges):
/// every layer's store — an attention layer's K/V planes, a delta layer's
/// recurrent state and conv ring —, the positions its recurrent stores hold,
/// and its checkpoints of those stores on the host. No capture travels with
/// it: the body records none over the stores (the step's and the passes'
/// are the model's, kept a slot).
pub struct Slot35 {
    pub(super) stores: Vec<LayerStore>,
    held: u32,
    ckpt: Checkpoints,
}

/// `name` of layer `l`.
fn blk(l: usize, stem: &str) -> String {
    format!("blk.{l}.{stem}")
}

/// The delta layers' stores a checkpoint copies, in [`Checkpoints::new`]'s
/// order: each one's state, then its conv ring.
fn copied(stores: &mut [LayerStore]) -> Vec<&mut DeviceBuffer<f32>> {
    let mut out = Vec::new();
    for s in stores {
        if let LayerStore::Rec(r) = s {
            out.push(&mut r.state);
            out.push(&mut r.ring);
        }
    }
    out
}

/// The joined name of layer `l`'s stack `part` (`gate`, `up`, `down`).
fn joint(l: usize, part: &str) -> String {
    format!("derived.blk.{l}.ffn_{part}_exps_sh")
}

/// The types each site launches (`crate::site`): a projection of the normed
/// rows or an output projection, a K-quant or Q8_0; β and α also F32; a
/// routed layer's stacks, the Q4_K gate and up and a K-quant down the routed
/// launches take; the embedding Q3_K, Q4_K, Q5_K or Q6_K rows or Q8_0 planes; the head Q6_K,
/// Q4_K or Q8_0 (`Head`).
const PROJ: &[SiteTy] = &[
    SiteTy::Q3K,
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Q8_0,
];
const BETA_ALPHA: &[SiteTy] = &[
    SiteTy::Q3K,
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Q8_0,
    SiteTy::F32,
];
const ROUTED: &[SiteTy] = &[SiteTy::Q4K];
const ROUTED_DOWN: &[SiteTy] = &[SiteTy::Q4K, SiteTy::Q6K];
const EMBED: &[SiteTy] = &[
    SiteTy::Q3K,
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Q8_0,
];
const HEAD_TY: &[SiteTy] = &[SiteTy::Q6K, SiteTy::Q4K, SiteTy::Q8_0];

/// The type of site `name` (`rows` rows of `k`) from `file`'s header.
fn ty_of(
    file: &Split,
    name: &str,
    rows: usize,
    k: usize,
    allowed: &[SiteTy],
) -> Result<SiteTy, GpuError> {
    file_site(file, WHAT, name, (rows, k), allowed)
}

/// The one type of a routed stack `part` of layer `l` (its experts' and its
/// shared expert's, which the load joins into one stack), of `rows` rows an
/// expert of `k` values, from `file`'s header; parts of two types are
/// refused by name.
fn routed_ty(
    file: &Split,
    l: usize,
    part: &str,
    (rows, k, experts): (usize, usize, usize),
    allowed: &[SiteTy],
) -> Result<SiteTy, GpuError> {
    let exps = ty_of(
        file,
        &blk(l, &format!("ffn_{part}_exps.weight")),
        experts * rows,
        k,
        allowed,
    )?;
    let sh = ty_of(
        file,
        &blk(l, &format!("ffn_{part}_shexp.weight")),
        rows,
        k,
        allowed,
    )?;
    if exps != sh {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "layer {l}: ffn_{part} experts are {exps}, the shared expert {sh}; one stack holds one type"
            ),
        ));
    }
    Ok(exps)
}

/// Layer `l`'s FFN by its description `spec`, every site's type from
/// `file`'s header: a routed one with the shared expert folded in — the
/// three stacks joined with their shared expert as expert `n` (the routed
/// count, [`join_ffn`]), the router with the shared gate as row `n` — or a
/// dense one, its three matrices a stack of one expert on the arena's fixed
/// route.
fn plan_ffn(file: &Split, d: &Dims, spec: &LayerSpec, l: usize) -> Result<FfnPlan, GpuError> {
    let (h, ff) = (d.hidden, d.ff);
    let ffn_norm = blk(l, "post_attention_norm.weight");
    match &spec.ffn {
        Ffn::Moe(m) => {
            let n = m.experts as usize;
            Ok(FfnPlan {
                ffn_norm,
                route: FfnRoute::Router {
                    gate_inp: format!("derived.blk.{l}.ffn_gate_inp_sh"),
                    shared: Some(SharedPlan),
                },
                gate: joint(l, "gate"),
                up: joint(l, "up"),
                down: joint(l, "down"),
                gate_ty: routed_ty(file, l, "gate", (ff, h, n), ROUTED)?,
                up_ty: routed_ty(file, l, "up", (ff, h, n), ROUTED)?,
                down_ty: routed_ty(file, l, "down", (h, ff, n), ROUTED_DOWN)?,
            })
        }
        Ffn::Dense { .. } => {
            let (gate, up, down) = (
                blk(l, "ffn_gate.weight"),
                blk(l, "ffn_up.weight"),
                blk(l, "ffn_down.weight"),
            );
            Ok(FfnPlan {
                ffn_norm,
                route: FfnRoute::Dense,
                gate_ty: ty_of(file, &gate, ff, h, PROJ)?,
                up_ty: ty_of(file, &up, ff, h, PROJ)?,
                down_ty: ty_of(file, &down, h, ff, PROJ)?,
                gate,
                up,
                down,
            })
        }
    }
}

/// Layer `l`'s routed stacks joined with their shared expert, and its router
/// with the shared gate (`Weights::join_rows`: the parts leave the map); a
/// dense layer has nothing to join.
fn join_ffn(
    stream: &CudaStream,
    w: &mut Weights,
    spec: &LayerSpec,
    l: usize,
) -> Result<(), GpuError> {
    if !matches!(spec.ffn, Ffn::Moe(_)) {
        return Ok(());
    }
    for part in ["gate", "up", "down"] {
        w.join_rows(
            stream,
            &[
                blk(l, &format!("ffn_{part}_exps.weight")).as_str(),
                blk(l, &format!("ffn_{part}_shexp.weight")).as_str(),
            ],
            joint(l, part),
        )?;
    }
    w.join_rows(
        stream,
        &[
            blk(l, "ffn_gate_inp.weight").as_str(),
            blk(l, "ffn_gate_inp_shexp.weight").as_str(),
        ],
        format!("derived.blk.{l}.ffn_gate_inp_sh"),
    )
}

/// Layer `l`'s gated attention over the flash `flash` its group selects,
/// every site's type from `file`'s header.
fn plan_gqa(file: &Split, d: &Dims, l: usize, flash: Flash) -> Result<GqaPlan, GpuError> {
    let (h, kv, att) = (d.hidden, d.kv_len(), d.attn_len());
    let (q, k, v, o) = (
        blk(l, "attn_q.weight"),
        blk(l, "attn_k.weight"),
        blk(l, "attn_v.weight"),
        blk(l, "attn_output.weight"),
    );
    Ok(GqaPlan {
        kind: GqaKind::Gated256,
        flash,
        attn_norm: blk(l, "attn_norm.weight"),
        attn_q_norm: blk(l, "attn_q_norm.weight"),
        attn_k_norm: blk(l, "attn_k_norm.weight"),
        q_ty: ty_of(file, &q, d.q_rows, h, PROJ)?,
        k_ty: ty_of(file, &k, kv, h, PROJ)?,
        v_ty: ty_of(file, &v, kv, h, PROJ)?,
        o_ty: ty_of(file, &o, h, att, PROJ)?,
        attn_q: q,
        attn_k: k,
        attn_v: v,
        attn_output: o,
    })
}

/// Layer `l`'s delta rule, every site's type from `file`'s header.
fn plan_delta(file: &Split, d: &Dims, shape: LinearShape, l: usize) -> Result<DeltaPlan, GpuError> {
    let (h, c, nv) = (d.hidden, shape.channels(), shape.n_v);
    let (qkv, gate, beta, alpha, out) = (
        blk(l, "attn_qkv.weight"),
        blk(l, "attn_gate.weight"),
        blk(l, "ssm_beta.weight"),
        blk(l, "ssm_alpha.weight"),
        blk(l, "ssm_out.weight"),
    );
    Ok(DeltaPlan {
        attn_norm: blk(l, "attn_norm.weight"),
        conv: blk(l, "ssm_conv1d.weight"),
        dt_bias: blk(l, "ssm_dt.bias"),
        ssm_a: blk(l, "ssm_a"),
        ssm_norm: blk(l, "ssm_norm.weight"),
        shape,
        qkv_ty: ty_of(file, &qkv, c, h, PROJ)?,
        gate_ty: ty_of(file, &gate, nv * linear::HEAD, h, PROJ)?,
        beta_ty: ty_of(file, &beta, nv, h, BETA_ALPHA)?,
        alpha_ty: ty_of(file, &alpha, nv, h, BETA_ALPHA)?,
        out_ty: ty_of(file, &out, h, nv * linear::HEAD, PROJ)?,
        qkv,
        gate,
        beta,
        alpha,
        ssm_out: out,
    })
}

/// Every weight of layer plan `p` resident as the plan read it from the
/// file: each site in its type and shape (`site::site`), each norm and
/// parameter table an F32 plane of its shape — the FFN's stacks only when
/// `stacks` (a placed load's are its placed side's to check, `placed`).
fn check_resident(w: &Weights, d: &Dims, p: &LayerPlan, stacks: bool) -> Result<(), GpuError> {
    let (h, ff) = (d.hidden, d.ff);
    let on =
        |name: &str, rows: usize, k: usize, ty: SiteTy| site::site(w, WHAT, name, (rows, k), ty);
    match &p.mixer {
        MixerPlan::Gqa(g) => {
            let (kv, att) = (d.kv_len(), d.attn_len());
            f32_site(w, &g.attn_norm, 1, h)?;
            f32_site(w, &g.attn_q_norm, 1, d.head)?;
            f32_site(w, &g.attn_k_norm, 1, d.head)?;
            on(&g.attn_q, d.q_rows, h, g.q_ty)?;
            on(&g.attn_k, kv, h, g.k_ty)?;
            on(&g.attn_v, kv, h, g.v_ty)?;
            on(&g.attn_output, h, att, g.o_ty)?;
        }
        MixerPlan::Delta(dp) => {
            let (c, nv) = (dp.shape.channels(), dp.shape.n_v);
            f32_site(w, &dp.attn_norm, 1, h)?;
            f32_site(w, &dp.conv, c, linear::CONV_TAPS)?;
            f32_site(w, &dp.dt_bias, 1, nv)?;
            f32_site(w, &dp.ssm_a, 1, nv)?;
            f32_site(w, &dp.ssm_norm, 1, linear::HEAD)?;
            on(&dp.qkv, c, h, dp.qkv_ty)?;
            on(&dp.gate, nv * linear::HEAD, h, dp.gate_ty)?;
            on(&dp.beta, nv, h, dp.beta_ty)?;
            on(&dp.alpha, nv, h, dp.alpha_ty)?;
            on(&dp.ssm_out, h, nv * linear::HEAD, dp.out_ty)?;
        }
    }
    let f = &p.ffn;
    let e = match &f.route {
        FfnRoute::Router { gate_inp, .. } => {
            let e = d.routed(WHAT)?.logits();
            f32_site(w, gate_inp, e, h)?;
            e
        }
        FfnRoute::Dense => 1,
    };
    f32_site(w, &f.ffn_norm, 1, h)?;
    if !stacks {
        return Ok(());
    }
    on(&f.gate, e * ff, h, f.gate_ty)?;
    on(&f.up, e * ff, h, f.up_ty)?;
    on(&f.down, e * h, ff, f.down_ty)
}

/// Layer `l`'s router joined with its shared expert's gate, the routed
/// stacks left as the plan uploaded them: a placed load's join
/// ([`join_ffn`]'s router half); a dense layer has nothing to join.
fn join_router(
    stream: &CudaStream,
    w: &mut Weights,
    spec: &LayerSpec,
    l: usize,
) -> Result<(), GpuError> {
    if !matches!(spec.ffn, Ffn::Moe(_)) {
        return Ok(());
    }
    w.join_rows(
        stream,
        &[
            blk(l, "ffn_gate_inp.weight").as_str(),
            blk(l, "ffn_gate_inp_shexp.weight").as_str(),
        ],
        format!("derived.blk.{l}.ffn_gate_inp_sh"),
    )
}

impl Forms {
    /// What the arena of a chain of `plans` holds beyond a fused K-quant
    /// chain's buffers (`Forms`'s doc), `d` its dims: every form one of the
    /// sites reads, the gate and up rows of an unfused gate·up, and the
    /// longest row-major output of a K-quant site launched alone.
    fn of(plans: &[LayerPlan], d: &Dims) -> Forms {
        let mut hid: Vec<SiteTy> = Vec::new();
        let (mut attn, mut h, mut glu, mut cols) = (Vec::new(), Vec::new(), false, 0usize);
        let mut alone = |ty: SiteTy, rows: usize| {
            if ty.kgemv_order() == Some(Order::RowMajor) {
                cols = cols.max(rows);
            }
        };
        for p in plans {
            match &p.mixer {
                MixerPlan::Gqa(g) => {
                    hid.extend([g.q_ty, g.k_ty, g.v_ty]);
                    attn.push(g.o_ty);
                    if !g.qkv_fused() {
                        alone(g.q_ty, d.q_rows);
                        alone(g.k_ty, d.kv_len());
                        alone(g.v_ty, d.kv_len());
                    }
                    if !g.o_fused() {
                        alone(g.o_ty, d.hidden);
                    }
                }
                MixerPlan::Delta(dp) => {
                    hid.extend([dp.qkv_ty, dp.gate_ty, dp.beta_ty, dp.alpha_ty]);
                    attn.push(dp.out_ty);
                    if !dp.input_fused() {
                        alone(dp.qkv_ty, dp.shape.channels());
                        alone(dp.gate_ty, dp.shape.n_v * linear::HEAD);
                        alone(dp.beta_ty, dp.shape.n_v);
                        alone(dp.alpha_ty, dp.shape.n_v);
                    }
                    if !dp.out_fused() {
                        alone(dp.out_ty, d.hidden);
                    }
                }
            }
            let f = &p.ffn;
            hid.extend([f.gate_ty, f.up_ty]);
            h.push(f.down_ty);
            if !f.gate_up_fused() {
                glu = true;
                alone(f.gate_ty, d.slots() * d.ff);
                alone(f.up_ty, d.slots() * d.ff);
            }
            if !f.down_sel() {
                alone(f.down_ty, d.hidden);
            }
        }
        let wants = |tys: &[SiteTy]| Wants {
            q128: tys.iter().any(|t| t.reads() == Form::Q8x128),
            q32: tys.iter().any(|t| t.reads() == Form::Q8x32),
        };
        Forms {
            hid: wants(&hid),
            attn: wants(&attn),
            h: wants(&h),
            glu,
            cols,
        }
    }
}

/// What a load reads from the file before any weight is uploaded: the
/// description, each layer's kind and plan with every site's type from the
/// header, the dims, the ubatch and the rope base — so a site of a type no
/// launch reads is refused by name before the upload.
struct Pre35 {
    vocab: usize,
    eps: f32,
    layers: Vec<LayerSpec>,
    kinds: Vec<Kind35>,
    plans: Vec<LayerPlan>,
    d: Dims,
    ubatch: NonZeroUsize,
    base: f32,
}

impl Pre35 {
    fn read(file: &Split, o: Open35) -> Result<Pre35, GpuError> {
        let Open35 {
            ctx, mma, ubatch, ..
        } = o;
        let read = model::arch::qwen35moe::spec::read(file).map_err(|e| GpuError::plan(WHAT, e))?;
        let spec = read.spec;
        if ctx == 0 {
            return Err(GpuError::shape(
                WHAT,
                "a cache of at least one row; asked for 0".to_string(),
            ));
        }
        let kinds = kinds35(&spec.layers)?;
        let d = dims(&spec, &kinds, ctx)?;
        let ubatch = ubatch_of(&d, ubatch)?;
        let base = spec
            .layers
            .iter()
            .find_map(|s| match &s.mixer {
                Mixer::Gqa(g) => Some(g.rope.base),
                _ => None,
            })
            .ok_or(GpuError::shape(
                WHAT,
                "no attention layer to read the rope base from",
            ))?;
        if mma && kinds.contains(&Kind35::Gqa(Flash::Pairs)) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the tensor-core decode flash at {}/{} heads: the pairs' pass is scalar only \
                     (open with mma false)",
                    d.n_head, d.n_kv
                ),
            ));
        }
        let vocab = spec.vocab as usize;
        ty_of(file, "token_embd.weight", vocab, d.hidden, EMBED)?;
        ty_of(
            file,
            crate::weights::head_tensor(file),
            vocab,
            d.hidden,
            HEAD_TY,
        )?;
        let plans = kinds
            .iter()
            .zip(&spec.layers)
            .enumerate()
            .map(|(l, (kind, layer))| {
                let mixer = match *kind {
                    Kind35::Gqa(flash) => MixerPlan::Gqa(plan_gqa(file, &d, l, flash)?),
                    Kind35::Delta(shape) => MixerPlan::Delta(plan_delta(file, &d, shape, l)?),
                };
                Ok(LayerPlan {
                    mixer,
                    ffn: plan_ffn(file, &d, layer, l)?,
                })
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        Ok(Pre35 {
            vocab,
            eps: spec.rms_eps,
            layers: spec.layers,
            kinds,
            plans,
            d,
            ubatch,
            base,
        })
    }
}

/// The arena's dims of `spec`, whose kinds `kinds` the plan checked: the
/// attention layers' head of 256 with a gate beside each query (every
/// attention layer of a file shares one head count), the FFN's — the router
/// the routed shape selects (every layer of a file shares it), or a dense
/// width every layer shares, never both kinds in one file — and the delta
/// layers' one shape (every delta layer of a file shares it). A delta
/// layer's gated norm writes the attention rows' buffer, so the two widths
/// must agree.
fn dims(spec: &ModelSpec, kinds: &[Kind35], ctx: usize) -> Result<Dims, GpuError> {
    let mut shapes = kinds.iter().filter_map(|k| match k {
        Kind35::Delta(s) => Some(*s),
        Kind35::Gqa(_) => None,
    });
    let lin = shapes.next();
    if let Some(s) = lin
        && let Some(other) = shapes.find(|o| *o != s)
    {
        return Err(GpuError::shape(
            WHAT,
            format!("two delta shapes, {s:?} and {other:?}; one arena serves one"),
        ));
    }
    let mut heads = spec.layers.iter().filter_map(|l| match &l.mixer {
        Mixer::Gqa(g) => Some((g.heads as usize, g.kv_heads as usize)),
        _ => None,
    });
    let (n_head, n_kv) = heads.next().ok_or(GpuError::shape(
        WHAT,
        "no attention layer to size the attention rows for",
    ))?;
    if let Some(other) = heads.find(|o| *o != (n_head, n_kv)) {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "two attention head counts, {n_head}/{n_kv} and {other:?}; one arena serves one"
            ),
        ));
    }
    let (router, ff) = ffn_dims(spec)?;
    let d = Dims {
        hidden: spec.hidden as usize,
        n_head,
        n_kv,
        head: HEAD_256,
        q_rows: 2 * n_head * HEAD_256,
        ff,
        router,
        lin,
        ctx,
    };
    if let Some(s) = lin
        && s.n_v * linear::HEAD != d.attn_len()
    {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "a delta layer's gated norm writes {} values a token, the attention rows hold {}",
                s.n_v * linear::HEAD,
                d.attn_len()
            ),
        ));
    }
    if !d.hidden.is_multiple_of(256) {
        return Err(GpuError::shape(
            WHAT,
            format!("hidden {} (whole K-quant super-blocks)", d.hidden),
        ));
    }
    Ok(d)
}

/// The FFN's part of the dims of `spec`: the router of its routed layers
/// (one shape, [`moe_fits`]) and the expert width, or no router and the
/// dense width its dense layers share. A file with both kinds, or neither,
/// is refused by name.
fn ffn_dims(spec: &ModelSpec) -> Result<(Option<super::router::RouterDims>, usize), GpuError> {
    let mut routed = spec.layers.iter().filter_map(|l| l.moe());
    let mut dense = spec.layers.iter().filter_map(|l| match l.ffn {
        Ffn::Dense { ff, .. } => Some(ff as usize),
        Ffn::Moe(_) => None,
    });
    match (routed.next(), dense.next()) {
        (Some(first), None) => {
            if let Some(other) = routed.find(|m| MoeShape::of(m) != MoeShape::of(first)) {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "two routed shapes, {:?} and {:?}; one arena serves one",
                        MoeShape::of(first),
                        MoeShape::of(other)
                    ),
                ));
            }
            let router = moe_fits(first).map_err(|e| GpuError::shape(WHAT, e))?;
            Ok((Some(router), q35::EXPERT_FF as usize))
        }
        (None, Some(ff)) => match dense.find(|&o| o != ff) {
            Some(other) => Err(GpuError::shape(
                WHAT,
                format!("two dense widths, {ff} and {other}; one arena serves one"),
            )),
            None => Ok((None, ff)),
        },
        (Some(_), Some(_)) => Err(GpuError::shape(
            WHAT,
            "routed and dense layers in one file; one arena serves one FFN kind",
        )),
        (None, None) => Err(GpuError::shape(WHAT, "no layer to size the FFN for")),
    }
}

impl GpuModel<Body35> {
    /// The whole Qwen3.6 model of `file` resident on `gpu`, plus the output
    /// head, with caches of `o.ctx` rows, the decode flash `o.mma` (the
    /// tensor-core pass when `true`) and ubatches of `o.ubatch` tokens
    /// ([`Open35`]). Every site's type is read from the file's header and a
    /// type no launch reads refused by name before any upload. The model
    /// takes the file.
    pub fn open(gpu: Gpu, file: Split, o: Open35) -> Result<GpuModel<Body35>, GpuError> {
        let n_layers = block_count(&file, "qwen35moe GpuModel::open")?;
        let pre = Pre35::read(&file, o)?;
        if pre.layers.len() != n_layers {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the description holds {} layers, the file's block count {n_layers}",
                    pre.layers.len()
                ),
            ));
        }
        GpuModel::load_blocks(gpu, &file, o.ctx, 0..n_layers, true, |gpu, w| {
            Body35::load(gpu, w, pre, o)
        })
    }

    /// The card bytes a placed load of `file` opened with `o` holds past the
    /// m = 1 scratch: its ubatch arena ([`arena_bytes`] at the placed
    /// ubatch, [`placed_open`]) and its placed side ([`placed_bytes`]) —
    /// what the plan's machine counts in the card's scratch
    /// (`model::arch::qwen3moe::place::machine`), and what
    /// [`GpuModel::open_placed`] refuses to pass.
    pub fn placed_arena_bytes(file: &Split, o: Open35) -> Result<u64, GpuError> {
        let o = placed_open(o);
        let pre = Pre35::read(file, o)?;
        let forms = Forms::of(&pre.plans, &pre.d);
        let arena = arena_bytes(&pre.d, prompt_rows(o.ubatch, o.ctx), forms) as u64;
        Ok(arena + placed_bytes(&pre.d, pre.plans.len())?)
    }

    /// What a whole load of `file` opened with `o` ([`GpuModel::open`])
    /// holds past its weights, stores and step arenas, as its own fit check
    /// counts it: its ubatch arena at `o.ubatch` ([`ubatch_need`]), checked
    /// against the card's free bytes, and [`FIT_RESERVE`], which the check
    /// keeps free past it. Read through the load's own description, so a
    /// file or options the load refuses are refused here by the same name.
    pub fn whole_load(file: &Split, o: Open35) -> Result<WholeLoad, GpuError> {
        let pre = Pre35::read(file, o)?;
        let forms = Forms::of(&pre.plans, &pre.d);
        Ok(WholeLoad::Checked {
            arena: ubatch_need(&pre.d, forms, pre.ubatch.get()).1 as u64,
            reserve: FIT_RESERVE as u64,
        })
    }

    /// The Qwen3.6 model of `file` placed by `plan` (made by
    /// `model::arch::qwen35moe::place::PlanInputs::plan_rule` under
    /// `model::arch::qwen3moe::place::card_routed` on the machine
    /// `model::arch::qwen3moe::place::machine` lays out): the plan's card
    /// segments resident ([`GpuModel::load_placed`]: the trunk, the shared
    /// experts, each layer's card experts) with each router joined with its
    /// shared gate, the host set read in and locked as `host` asks, and the
    /// body over them with its placed side ([`Placed`]), caches of `o.ctx`
    /// rows — the plan's context — and the ubatch arena of the placed
    /// ubatch ([`placed_open`]) held to the bytes the plan counts
    /// ([`GpuModel::placed_arena_bytes`]); the checkpoints' spacing
    /// ([`checkpoint_every`]) is the one part that reads `o.ubatch`. Refused
    /// by name as [`GpuModel::open`] refuses, for a plan of another context,
    /// and as the placed side refuses ([`Placed::new`]).
    pub fn open_placed(
        file: Split,
        plan: &Plan<'_>,
        o: Open35,
        host: HostCfg,
    ) -> Result<GpuModel<Body35>, GpuError> {
        const WHAT_P: &str = "qwen35moe GpuModel::open_placed";
        let asked = o.ubatch;
        let o = placed_open(o);
        let n_layers = block_count(&file, WHAT_P)?;
        if u64::try_from(o.ctx).ok() != Some(plan.ctx_max) {
            return Err(GpuError::shape(
                WHAT_P,
                format!(
                    "caches of {} rows on a plan of {} positions",
                    o.ctx, plan.ctx_max
                ),
            ));
        }
        let pre = Pre35::read(&file, o)?;
        if pre.layers.len() != n_layers {
            return Err(GpuError::shape(
                WHAT_P,
                format!(
                    "the description holds {} layers, the file's block count {n_layers}",
                    pre.layers.len()
                ),
            ));
        }
        let card = plan
            .machine
            .cards
            .first()
            .ok_or(GpuError::shape(WHAT_P, "a plan of no card"))?;
        let counted = model::arch::qwen3moe::place::counted_arena_bytes(card);
        let specs = pre.layers.clone();
        GpuModel::load_placed(
            file,
            plan,
            0,
            host,
            |stream, _, layers, w| {
                for l in layers {
                    let spec = specs.get(l).ok_or_else(|| {
                        GpuError::shape(WHAT_P, format!("layer {l} past the description"))
                    })?;
                    join_router(stream, w, spec, l)?;
                }
                Ok(())
            },
            |gpu, file, w, set| {
                let open = PlacedOpen {
                    plan,
                    file: Arc::new(file),
                    host,
                    set,
                    arch: "qwen35moe",
                };
                Body35::build(gpu, w, pre, o, asked, Some((open, counted)))
            },
        )
    }
}

impl Body35 {
    /// The whole-card load: each routed layer's stacks joined with its
    /// shared expert and its router with the shared gate ([`join_ffn`]),
    /// then the body ([`Body35::build`]).
    fn load(gpu: &Gpu, w: &mut Weights, pre: Pre35, o: Open35) -> Result<Body35, GpuError> {
        for (l, layer) in pre.layers.iter().enumerate() {
            join_ffn(gpu.stream(), w, layer, l)?;
        }
        Body35::build(gpu, w, pre, o, o.ubatch, None)
    }

    /// The body over the resident weights `w` of `pre`'s description: the
    /// stores, the arenas, the kernels and the prompt image; on a placed
    /// load (`placed`: its placed side's inputs and the arena bytes its plan
    /// counts) the ubatch arena held to what the plan counts, not to the
    /// card's free bytes, and the placed side ([`Placed::new`]). The
    /// checkpoints' spacing is the ubatch size the load was asked for
    /// (`asked`: [`checkpoint_every`]), whether or not a placed load walks
    /// its prompt at it.
    fn build(
        gpu: &Gpu,
        w: &Weights,
        pre: Pre35,
        o: Open35,
        asked: usize,
        placed: Option<(PlacedOpen<'_>, u64)>,
    ) -> Result<Body35, GpuError> {
        let Pre35 {
            vocab,
            eps,
            layers: _,
            kinds,
            plans,
            d,
            ubatch,
            base,
        } = pre;
        let ctx = o.ctx;
        let stream = gpu.stream();
        let mut stores = Vec::with_capacity(plans.len());
        for (kind, plan) in kinds.iter().zip(&plans) {
            check_resident(w, &d, plan, placed.is_none())?;
            stores.push(match *kind {
                Kind35::Gqa(_) => LayerStore::Kv(KvPlanes::new(stream, &d, o.kv)?),
                Kind35::Delta(shape) => LayerStore::Rec(RecStore::new(stream, shape, LANES)?),
            });
        }
        let forms = Forms::of(&plans, &d);
        let rope = RopeTable::new(&RopeSpec::window(base, q35::ROPE_DIMS as usize))?;
        let rope = RopeRows::new(stream, &rope, q35::ROPE_DIMS as usize, ctx)?;
        // The stores a checkpoint copies, always in this order: each delta
        // layer's state, then its conv ring.
        let lens: Vec<usize> = stores
            .iter()
            .filter_map(|s| match s {
                LayerStore::Rec(r) => Some([r.state.len(), r.ring.len()]),
                LayerStore::Kv(_) => None,
            })
            .flatten()
            .collect();
        let every = checkpoint_every(asked, &d)?;
        let ckpt = Checkpoints::new(gpu.context(), lens, HOST_BUDGET, every)?;
        let s = Arena::with(stream, d, 1, forms)?;
        let sp = StepParams::new(stream, true)?;
        let a = Arena::with(stream, d, MAX_PASS_ROWS, forms)?;
        let rp = RowsParams::new(stream)?;
        let slot_in = SlotIn::new(stream)?;
        let mut slot_lane = DeviceBuffer::<u32>::zeroed(stream, 1)?;
        slot_lane.copy_from_host(stream, &[LANE])?;
        let k = Kernels::load(gpu, true)?;
        let head_state = HeadArgmaxState::new(stream)?;
        let img = PromptImage::new(stream, ctx, true)?;
        let (u, placed) = match placed {
            None => {
                let (free, _) = gpu.mem_info()?;
                (ubatch_arena(stream, (d, forms), ubatch.get(), free)?, None)
            }
            Some((open, counted)) => {
                let side = placed_bytes(&d, plans.len())?;
                let u = counted_arena(stream, (d, forms), ubatch.get(), counted, side)?;
                (u, Some(Placed::new(gpu, w, &plans, &d, open)?))
            }
        };
        Ok(Body35 {
            vocab,
            eps,
            plans,
            stores,
            rope,
            s,
            sp,
            a,
            rp,
            slot_in,
            slot_lane,
            slots_planned: None,
            u,
            img,
            ubatch,
            k,
            head_state,
            mma: o.mma,
            taps: None,
            placed,
            held: 0,
            ckpt,
            every,
            marks: false,
            prompt_walks: 0,
        })
    }

    /// Every layer's kind, in order.
    #[must_use]
    pub fn kinds(&self) -> Vec<LayerKind35> {
        self.plans
            .iter()
            .map(|p| match p.mixer {
                MixerPlan::Gqa(_) => LayerKind35::Attention,
                MixerPlan::Delta(_) => LayerKind35::Delta,
            })
            .collect()
    }

    /// Launches of a pass of `m` rows through every layer, without the
    /// heads: the embedding, then each layer's mixer and FFN half
    /// (`dispatch::pass_launches`). The captured decode step adds the one
    /// head's three launches; a captured pass of `m >= 2` rows adds, per
    /// row, the copy of its residual row into its head and the head's three.
    #[must_use]
    pub fn pass_launches(&self, m: usize) -> usize {
        dispatch::pass_launches(&self.plans, m)
    }

    /// The flash pass this body runs: `true` for the tensor-core pass.
    #[must_use]
    pub fn flash_mma(&self) -> bool {
        self.mma
    }

    /// The placed side of a placed load ([`Placed`]); `None` on a
    /// whole-card load.
    #[must_use]
    pub fn placed(&self) -> Option<&Placed> {
        self.placed.as_ref()
    }

    /// Device bytes of the layer stores: the attention layers' K/V planes
    /// and the delta layers' states and conv rings.
    #[must_use]
    pub fn store_bytes(&self) -> usize {
        self.stores.iter().map(LayerStore::bytes).sum()
    }

    /// What a cut to at most `n` positions of a model standing at `pos`
    /// keeps, and why: every position, the empty model at 0, or the nearest
    /// checkpoint at or below `n`, nothing (`Missed`) when none lies there —
    /// a delta layer keeps no earlier state but a copy. After a call failed
    /// past its launch the stores hold more than `pos`, which is no longer
    /// kept. Never past `pos`. The rule [`Body35::cut`] takes.
    #[must_use]
    pub fn kept(&self, n: u32, pos: u32) -> Kept {
        if self.held == pos {
            self.ckpt.kept(n, pos)
        } else {
            self.ckpt.kept(n.min(pos), self.held)
        }
    }

    /// The checkpoints: their positions, their slots, what they have done.
    #[must_use]
    pub fn checkpoints(&self) -> &Checkpoints {
        &self.ckpt
    }

    /// Whether a prompt call takes checkpoints at its marks ([`Body35::marked`]),
    /// on every resident sequence, each into its own: off at load, so a
    /// binary that never cuts behind the fed positions copies nothing; the
    /// serve seat turns it on. Off drops none taken.
    pub fn set_checkpoints(&mut self, on: bool) {
        self.marks = on;
    }

    /// Whether a prompt call takes checkpoints ([`Body35::set_checkpoints`]).
    #[must_use]
    pub fn takes_checkpoints(&self) -> bool {
        self.marks
    }

    /// Walks the last prompt call ran ([`GpuModel::prefill_with`],
    /// [`GpuModel::prefill_hidden`]): one a unit of the call's plans, the
    /// count taken from the call's start. A call that fails past a walk
    /// leaves the walks it took.
    #[must_use]
    pub fn prompt_walks(&self) -> usize {
        self.prompt_walks
    }

    /// The marks a prompt call from `from` to `to` takes a checkpoint at:
    /// [`Checkpoints::marks`] less every inner mark whose neighbouring run
    /// would hold fewer than [`WIDE_FROM`] rows — a run that short runs the
    /// gemv arm, whose bits are not a longer run's, so the rows a kept
    /// prefix leaves behind would not be a whole fresh call's; the mark's
    /// position keeps no checkpoint, and a cut there keeps the mark below
    /// it. The call's own ends always stand.
    fn marked(&self, from: u32, to: u32) -> Vec<u32> {
        let min = WIDE_FROM as u32;
        let mut out = Vec::new();
        let mut last = from;
        for m in self.ckpt.marks(from, to) {
            let inner = m != from && m != to;
            if inner && (m - last < min || to - m < min) {
                continue;
            }
            out.push(m);
            last = m;
        }
        out
    }

    /// Refused by name unless the recurrent stores hold `pos` positions: a
    /// call at `pos` that failed after its launches were enqueued has
    /// already run the recurrence over it, and running it again would apply
    /// the position twice.
    pub(super) fn stores_at(&self, pos: u32) -> Result<(), GpuError> {
        const WHAT_S: &str = "qwen35moe::Body35::stores";
        match self.held {
            h if h == pos => Ok(()),
            h => Err(GpuError::shape(
                WHAT_S,
                format!(
                    "position {pos}, where the recurrent stores hold {h}{}: reset, or cut to a \
                     checkpoint (Body35::kept)",
                    if h == pos.wrapping_add(1) {
                        " (the call there failed after its launches)"
                    } else {
                        ""
                    }
                ),
            )),
        }
    }

    /// The waiting cut carried out ([`Checkpoints::apply`]): every delta
    /// layer's state and conv ring from the point's slot, or each zeroed.
    /// Nothing when no cut waits. Waits for the copies; never inside a
    /// capture (a cut has no stream of its own).
    fn settle(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        if !self.ckpt.pending() {
            return Ok(());
        }
        let Body35 { ckpt, stores, .. } = self;
        ckpt.apply(stream, &mut copied(stores))
    }

    /// The stores counted at the end of a call of `m` positions from `pos`,
    /// after any waiting cut and a check that they stand at `pos`: what
    /// follows launches, and a call that fails past its launch leaves them
    /// ahead of the model's position, which the next call refuses by name.
    pub(super) fn stand_held(
        &mut self,
        stream: &CudaStream,
        pos: u32,
        m: usize,
    ) -> Result<(), GpuError> {
        const WHAT_M: &str = "qwen35moe::Body35::stand_held";
        self.settle(stream)?;
        self.stores_at(pos)?;
        self.held = pos
            + u32::try_from(m).map_err(|_| {
                GpuError::shape(WHAT_M, format!("a call of {m} positions past u32"))
            })?;
        Ok(())
    }

    /// A checkpoint at `pos`, where the stores stand, after any waiting cut:
    /// the delta layers' stores copied to a host slot, or nothing where one
    /// stands or at 0. Refused by name when the stores hold another
    /// position. Waits for the copies.
    fn take_mark(&mut self, stream: &CudaStream, pos: u32) -> Result<(), GpuError> {
        self.stores_at(pos)?;
        self.settle(stream)?;
        let Body35 { ckpt, stores, .. } = self;
        ckpt.take(stream, pos, &mut copied(stores))?;
        Ok(())
    }

    /// A cut to `pos`, behind the stores' position: the point at `pos`
    /// planned for the next call ([`Body35::settle`]), every later point
    /// dropped as another branch's. Refused by name, nothing moved, at 0 (a
    /// reset), past what the stores hold, and where no point stands; the
    /// model's position is the caller's ([`GpuModel::rollback`] stands it).
    fn cut(&mut self, pos: u32) -> Result<(), GpuError> {
        const WHAT_C: &str = "qwen35moe::Body35::cut";
        if pos == 0 {
            return Err(GpuError::shape(
                WHAT_C,
                format!(
                    "back to position 0 from {}: the empty model is a reset",
                    self.held
                ),
            ));
        }
        self.ckpt.cut(pos, self.held)?;
        self.held = pos;
        Ok(())
    }

    /// Exchange the live sequence with `seq`: its stores, the positions they
    /// hold and its checkpoints, pointer moves only ([`Slots::swap_seq`],
    /// and a pass of several slots' per-slot work, `body35_slots`).
    pub(super) fn exchange(&mut self, seq: &mut Slot35) {
        std::mem::swap(&mut self.stores, &mut seq.stores);
        std::mem::swap(&mut self.held, &mut seq.held);
        std::mem::swap(&mut self.ckpt, &mut seq.ckpt);
    }

    /// Keep a copy of every layer's output residual after each layer of the
    /// decode chain (`on`), or stop. Load-time allocation.
    pub(super) fn set_taps(&mut self, stream: &CudaStream, on: bool) -> Result<(), GpuError> {
        self.taps = None;
        if on {
            let hidden = self.s.dims.hidden;
            let buf = DeviceBuffer::<f32>::zeroed(stream, self.plans.len() * hidden)?;
            let rows = (0..self.plans.len())
                .map(|l| {
                    let ptr = buf.cu_deviceptr() + (l * hidden * size_of::<f32>()) as u64;
                    // SAFETY: row `l` spans elements `l·hidden ..
                    // (l+1)·hidden` of `buf`, which moves into the same
                    // `TapRows` beside its windows and outlives them.
                    unsafe { window(ptr, hidden, buf.context()) }
                })
                .collect();
            self.taps = Some(TapRows { buf, rows });
        }
        Ok(())
    }
}

/// Enqueue the decode chain at one row: layer 0 with the step's embedding,
/// every later layer reading the previous layer's output, the last writing
/// the head's input, then the head — the walk `(1, 1, Step)`.
fn enqueue_chain(gpu: &Gpu, w: &Weights, b: &mut Body35, head: &mut Head) -> Result<(), GpuError> {
    let Body35 {
        eps,
        plans,
        stores,
        rope,
        s,
        sp,
        k,
        head_state,
        mma,
        taps,
        placed,
        ..
    } = b;
    let c = PassCtx {
        gpu,
        w,
        plans,
        k,
        mma: *mma,
        eps: *eps,
        table: &rope.table,
    };
    if let Some(Placed {
        hybrid, side, step, ..
    }) = placed
    {
        let mut leg = StepLeg::new(gpu.stream(), hybrid);
        return StepWalk {
            p: WalkParts {
                c: &c,
                stores: stores.as_mut_slice(),
                s,
                io: &sp.io(),
                m: 1,
                side,
                rows: step,
            },
            tail: Tail::Step {
                head,
                state: head_state,
                taps: taps.as_mut(),
            },
        }
        .walk(&mut leg);
    }
    Program {
        c: &c,
        stores: stores.as_mut_slice(),
        s,
        io: &sp.io(),
        m: 1,
        tail: Tail::Step {
            head,
            state: head_state,
            taps: taps.as_mut(),
        },
    }
    .walk()
}

/// Enqueue a pass of `heads.len()` rows from the pass record: every layer
/// at that many rows over the pass arena, then row `r`'s residual into
/// `heads[r]`'s input and that head, row by row — the walk `(1, m, Step)`.
fn enqueue_rows(
    gpu: &Gpu,
    w: &Weights,
    b: &mut Body35,
    heads: &mut [Head],
) -> Result<(), GpuError> {
    const WHAT_ROWS: &str = "qwen35moe::enqueue_rows";
    if b.placed.is_some() {
        return Err(GpuError::shape(
            WHAT_ROWS,
            "a pass of several rows on a placed load: its prompt runs through the prompt call \
             (prefill_with), each unit through the host tier's batch port",
        ));
    }
    let m = heads.len();
    // A capture records the launches only; the replay reads the record the
    // plan before it wrote, so only an eager pass needs one planned now.
    if b.rp.planned != Some(m) && !crate::capturing(gpu.stream())? {
        return Err(GpuError::state(
            WHAT_ROWS,
            "a record planned for this pass's rows (plan_rows before the pass)",
        ));
    }
    let Body35 {
        eps,
        plans,
        stores,
        rope,
        a,
        rp,
        k,
        head_state,
        mma,
        ..
    } = b;
    let c = PassCtx {
        gpu,
        w,
        plans,
        k,
        mma: *mma,
        eps: *eps,
        table: &rope.table,
    };
    Program {
        c: &c,
        stores: stores.as_mut_slice(),
        s: a,
        io: &rp.io(m)?,
        m,
        tail: Tail::Rows {
            heads,
            state: head_state,
        },
    }
    .walk()
}

/// What a prompt unit's walk ends in.
enum UnitEnd<'a> {
    /// Nothing: the next unit reads the arena's `x`.
    Pass,
    /// The head of its last row.
    Head(&'a mut Head),
    /// The final norm of every row into the arena's `normed`.
    Hidden,
}

impl Body35 {
    /// Enqueue the unit of the image's `tokens`, standing at position `pos`
    /// (the image's position for its first token, else refused): the walk
    /// `(1, t, Step)` over the ubatch arena, ending as `end` says.
    fn walk_unit(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        tokens: Range<usize>,
        pos: u32,
        end: UnitEnd<'_>,
    ) -> Result<(), GpuError> {
        const WHAT_UNIT: &str = "qwen35moe::walk_unit";
        let want = u32::try_from(tokens.start)
            .ok()
            .and_then(|s| self.img.pos0.checked_add(s));
        if want != Some(pos) {
            return Err(GpuError::state(
                WHAT_UNIT,
                "the unit's first position is the image's position for its first token",
            ));
        }
        let m = tokens.len();
        self.stand_held(gpu.stream(), pos, m)?;
        let win = self.img.windows(tokens)?;
        self.prompt_walks += 1;
        let Body35 {
            eps,
            plans,
            stores,
            rope,
            u,
            k,
            head_state,
            mma,
            placed,
            ..
        } = self;
        let c = PassCtx {
            gpu,
            w,
            plans,
            k,
            mma: *mma,
            eps: *eps,
            table: &rope.table,
        };
        let tail = match end {
            UnitEnd::Head(head) => Tail::Last {
                head,
                state: head_state,
            },
            UnitEnd::Pass => Tail::Pass,
            UnitEnd::Hidden => Tail::Hidden,
        };
        if let Some(Placed {
            hybrid,
            side,
            pass,
            pass_hsum,
            ..
        }) = placed
        {
            let mut leg = BatchLeg::new(gpu.stream(), hybrid, pass_hsum, MAX_PASS_ROWS);
            return BatchWalk {
                p: WalkParts {
                    c: &c,
                    stores: stores.as_mut_slice(),
                    s: u,
                    io: &win.io(),
                    m,
                    side,
                    rows: pass,
                },
            }
            .walk(&mut leg, tail);
        }
        Program {
            c: &c,
            stores: stores.as_mut_slice(),
            s: u,
            io: &win.io(),
            m,
            tail,
        }
        .walk()
    }
}

impl GpuModel<Body35> {
    /// The prompt call's ubatch size: the most tokens one ubatch takes.
    pub fn ubatch(&self) -> Result<usize, GpuError> {
        Ok(self.body("qwen35moe::ubatch")?.ubatch.get())
    }

    /// Run the prompt call in ubatches of up to `u` tokens from here on: the
    /// ubatch arena reallocated for them, after [`Open35::ubatch`]'s checks
    /// (the new arena is allocated before the old one is freed, so a refusal
    /// or a failed allocation leaves the old size in place). Load-time
    /// allocation; a token's values do not depend on the ubatch it lands in,
    /// so a prompt leaves the same bits at every size.
    pub fn set_ubatch(&mut self, u: usize) -> Result<(), GpuError> {
        let (gpu, _, body) = self.body_parts("qwen35moe::set_ubatch")?;
        if body.placed.is_some() {
            return Err(GpuError::shape(
                "qwen35moe::set_ubatch",
                format!(
                    "ubatches of {u} tokens on a placed load: its arena is the plan's count, \
                     passes of up to {MAX_PASS_ROWS}"
                ),
            ));
        }
        let d = body.u.dims;
        let size = ubatch_of(&d, u)?;
        let forms = Forms::of(&body.plans, &d);
        let (free, _) = gpu.mem_info()?;
        body.u = ubatch_arena(gpu.stream(), (d, forms), u, free)?;
        body.ubatch = size;
        Ok(())
    }

    /// The units [`GpuModel::prefill_with`] runs a prompt of `tokens` ids as
    /// by `path`, at this model's ubatch size.
    pub fn prefill_plan(&self, tokens: usize, path: PrefillPath) -> Result<PrefillPlan, GpuError> {
        let body = self.body("qwen35moe::prefill_plan")?;
        // A placed load walks units of at most GEMV_COLS rows through the
        // host tier's batch port: its prompt runs as passes.
        let path = match (body.placed.is_some(), path) {
            (true, PrefillPath::Gemm) => {
                return Err(GpuError::shape(
                    "qwen35moe::prefill_plan",
                    "a GEMM ubatch on a placed load: its prompt runs as passes (auto or pass)",
                ));
            }
            (true, _) => PrefillPath::Pass,
            (false, p) => p,
        };
        PrefillPlan::new(tokens, path, body.ubatch)
    }

    /// The last prompt image the prompt call wrote; `None` before one.
    pub fn prompt_image(&self) -> Result<Option<ImageWrite>, GpuError> {
        Ok(self.body("qwen35moe::prompt_image")?.img.last)
    }

    /// Feed `tokens` through the chain by the plan of `path` and return the
    /// greedy next token after the last one; positions continue from wherever
    /// the model stands. While the load takes checkpoints
    /// ([`Body35::set_checkpoints`]) the call runs as the runs between its
    /// marks ([`Body35::marked`]), a checkpoint of the delta layers' stores
    /// taken at each; else it is one run. The prompt's image — its first
    /// position, its ids and the lane word — is written and copied once per
    /// run, then each unit of the run's plan is one walk over the ubatch
    /// arena, eager in either step mode: a unit of at most [`GEMV_COLS`]
    /// rows (a pass, or a ubatch that short) leaves the cache rows, the
    /// recurrent state and the logits of one step per token bit for bit, a
    /// longer one agrees with them to its band (`wide`). The last unit's
    /// last row runs the head. The layer taps must be off, and every id must
    /// be below the vocabulary; a prompt past the cache is refused before any
    /// launch.
    pub fn prefill_with(&mut self, tokens: &[u32], path: PrefillPath) -> Result<u32, GpuError> {
        const WHAT_P: &str = "qwen35moe::prefill";
        if tokens.is_empty() {
            return Err(GpuError::shape(WHAT_P, "empty token slice"));
        }
        let pos0 = self.pos();
        let n = launch_u32(WHAT_P, "tokens", tokens.len())?;
        self.check_pos(pos0 + n - 1, WHAT_P)?;
        let marked = {
            let (_, _, body) = self.body_parts(WHAT_P)?;
            if body.taps.is_some() {
                return Err(GpuError::state(
                    WHAT_P,
                    "layer taps off (a prompt unit writes none)",
                ));
            }
            super::refuse_past_vocab(WHAT_P, tokens, body.vocab)?;
            body.prompt_walks = 0;
            body.takes_checkpoints()
        };
        let marks = if marked {
            self.body(WHAT_P)?.marked(pos0, pos0 + n)
        } else {
            vec![pos0 + n]
        };
        let (mut next, mut at) = (None, pos0);
        for &mark in &marks {
            if mark > at {
                let run = &tokens[(at - pos0) as usize..(mark - pos0) as usize];
                next = Some(self.feed35(run, path)?);
                at = mark;
            }
            if marked {
                let pos = self.pos();
                let (gpu, _, body) = self.body_parts(WHAT_P)?;
                body.take_mark(gpu.stream(), pos)?;
            }
        }
        next.ok_or(GpuError::state(WHAT_P, "a token read after the last unit"))
    }

    /// One run of [`GpuModel::prefill_with`]'s call, from where the model
    /// stands: the image written for the run's ids, then each unit of the
    /// run's plan one walk, the last ending in its last row's head. The
    /// argmax after the last id.
    fn feed35(&mut self, run: &[u32], path: PrefillPath) -> Result<u32, GpuError> {
        const WHAT_F: &str = "qwen35moe::prefill::run";
        let plan = self.prefill_plan(run.len(), path)?;
        {
            let pos = self.pos();
            let (gpu, _, body) = self.body_parts(WHAT_F)?;
            body.img.write(gpu.stream(), run, pos)?;
        }
        let n_steps = plan.steps.len();
        let (mut at, mut next) = (0usize, None);
        for (i, &step) in plan.steps.iter().enumerate() {
            let t = match step {
                PrefillStep::Ubatch(t) | PrefillStep::Pass(t) => t,
            };
            let last = i + 1 == n_steps;
            let unit = at..at + t;
            at += t;
            next = self.run_rows(t, WHAT_F, |gpu, w, body, head, pos| {
                let end = if last {
                    UnitEnd::Head(head)
                } else {
                    UnitEnd::Pass
                };
                body.walk_unit(gpu, w, unit, pos, end)?;
                Ok(last)
            })?;
        }
        next.ok_or(GpuError::state(WHAT_F, "a token read after the last unit"))
    }

    /// Feed `tokens` through the chain by the plan of `path`, as
    /// [`GpuModel::prefill_with`] does, and return every position's
    /// final-norm hidden state — the last layer's output RMS-normed by
    /// `output_norm`, llama.cpp's `result_norm` and HF's `last_hidden_state`
    /// — `tokens.len()` rows of the model's width, row `t` at `t · hidden`.
    /// No head runs: no output projection, no logits. Each unit ends in the
    /// norm of its rows ([`Tail::Hidden`]), copied to the host before the next
    /// unit overwrites them, and the fault word is read after each unit: a
    /// raised fault is [`GpuError::Fault`] and leaves the model poisoned.
    /// The layer taps must be off, and every id must be below the
    /// vocabulary; a prompt past the cache is refused before any launch.
    pub fn prefill_hidden(
        &mut self,
        tokens: &[u32],
        path: PrefillPath,
    ) -> Result<Vec<f32>, GpuError> {
        const WHAT_H: &str = "qwen35moe::prefill_hidden";
        if tokens.is_empty() {
            return Err(GpuError::shape(WHAT_H, "empty token slice"));
        }
        let pos0 = self.pos();
        self.check_pos(
            pos0 + launch_u32(WHAT_H, "tokens", tokens.len())? - 1,
            WHAT_H,
        )?;
        let plan = self.prefill_plan(tokens.len(), path)?;
        let hidden = {
            let (gpu, _, body) = self.body_parts(WHAT_H)?;
            if body.taps.is_some() {
                return Err(GpuError::state(
                    WHAT_H,
                    "layer taps off (a prompt unit writes none)",
                ));
            }
            super::refuse_past_vocab(WHAT_H, tokens, body.vocab)?;
            body.prompt_walks = 0;
            body.img.write(gpu.stream(), tokens, pos0)?;
            body.u.dims.hidden
        };
        let mut out = Vec::with_capacity(tokens.len() * hidden);
        let mut at = 0usize;
        for &step in &plan.steps {
            let t = match step {
                PrefillStep::Ubatch(t) | PrefillStep::Pass(t) => t,
            };
            let unit = at..at + t;
            at += t;
            let out = &mut out;
            self.run_rows(t, WHAT_H, |gpu, w, body, _, pos| {
                body.walk_unit(gpu, w, unit, pos, UnitEnd::Hidden)?;
                if let Some(f) = gpu.fault()? {
                    return Err(GpuError::fault(WHAT_H, f));
                }
                // SAFETY: rows `0..t` of `normed` span `t · hidden` values
                // inside it (`rows · hidden`, `t <= rows` by the walk's
                // refusal), and the arena stays in place for the copy.
                let rows = unsafe { super::scratch::f32_view(&body.u.normed, 0, t * hidden) };
                out.extend_from_slice(&rows.to_host_vec(gpu.stream())?);
                Ok(false)
            })?;
        }
        Ok(out)
    }
}

impl ChainBody for Body35 {
    type Input = DecodeInput35;
    type Host = Placed;

    fn arch() -> Arch {
        Arch::Qwen35moe
    }

    fn decode_input(&mut self, token: u32, pos: u32) -> Result<DecodeInput35, GpuError> {
        Ok(DecodeInput35 { token, pos })
    }

    /// The step's input record — its position, its token and the lane word
    /// — in one asynchronous copy ahead of the step's launches, after any
    /// waiting cut is carried out and the stores are checked to stand at
    /// `pos`: the launches advance them to `pos + 1`, which a failure past
    /// the enqueue leaves held for the next call to refuse.
    fn refresh(&mut self, stream: &CudaStream, input: &DecodeInput35) -> Result<(), GpuError> {
        let DecodeInput35 { token, pos } = *input;
        self.settle(stream)?;
        self.stores_at(pos)?;
        self.held = pos + 1;
        self.sp.write(stream, token, pos)
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        enqueue_chain(gpu, w, self, head)
    }

    /// Every delta layer's state lanes and conv ring back to zero, the
    /// checkpoints dropped and the stores counted at 0 — the live
    /// sequence's, the selected slot's; a parked one keeps its own — and
    /// the records' lane word to [`LANE`], after a placed load's host tier
    /// reset — the load's, whichever slot resets: a refused input's poison
    /// lifted, the words checked at rest. The K/V planes need nothing: the
    /// flash never loads a key row at or past the live count, and every row
    /// below it is written by its own step first. Synchronizes.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let stream = gpu.stream();
        if let Some(p) = self.placed.as_mut() {
            p.reset(stream)?;
        }
        let most = self
            .stores
            .iter()
            .map(|s| match s {
                LayerStore::Rec(r) => r.state.len().max(r.ring.len()),
                LayerStore::Kv(_) => 0,
            })
            .max()
            .unwrap_or(0);
        let zeros = vec![0.0f32; most];
        for s in &mut self.stores {
            if let LayerStore::Rec(r) = s {
                r.clear(stream, &zeros)?;
            }
        }
        self.ckpt.clear();
        self.held = 0;
        self.sp.write(stream, 0, 0)?;
        stream.synchronize()?;
        Ok(())
    }

    fn head_eps(&self) -> f32 {
        self.eps
    }

    fn resident_bytes(&self) -> usize {
        self.store_bytes()
            + self.rope.table.num_bytes()
            + self.s.bytes()
            + self.sp.bytes()
            + self.a.bytes()
            + self.rp.bytes()
            + self.slot_in.bytes()
            + self.slot_lane.num_bytes()
            + self.u.bytes()
            + self.img.bytes()
            + self.head_state.bytes()
            + self.taps.as_ref().map_or(0, |t| t.buf.num_bytes())
            + self.placed.as_ref().map_or(0, Placed::device_bytes)
    }

    fn layers(&self) -> Range<usize> {
        0..self.plans.len()
    }

    fn host(&mut self) -> Option<&mut Placed> {
        self.placed.as_mut()
    }
}

impl Slots for Body35 {
    type Seq = Slot35;

    /// A sequence of the load's shape in the state the load leaves: each
    /// layer's store zeroed — K/V planes in the cache format the load chose
    /// (read back from the live planes), or a recurrent state and conv ring
    /// of the layer's delta shape —, no position held, and checkpoints of
    /// the same stores at the load's spacing, no point taken and no host slot
    /// made. A placed load's sequences share its placed side whole: the card
    /// experts and the slot map are weights, the host tier's experts
    /// compute each token apart, its handoff words and health are the
    /// load's protocol (a go advances them whichever sequence runs), and its
    /// rows and host sums are a call's scratch. Load-time allocation.
    fn new_seq(&mut self, gpu: &Gpu) -> Result<Slot35, GpuError> {
        const WHAT_N: &str = "qwen35moe::Body35::new_seq";
        let stream = gpu.stream();
        let d = self.s.dims;
        let stores = self
            .plans
            .iter()
            .zip(&self.stores)
            .map(|(p, live)| match (&p.mixer, live) {
                (MixerPlan::Gqa(_), LayerStore::Kv(planes)) => {
                    let kv = match planes {
                        KvPlanes::F16 { .. } => KvQ8::F16,
                        KvPlanes::Q8 { .. } => KvQ8::Q8,
                    };
                    Ok(LayerStore::Kv(KvPlanes::new(stream, &d, kv)?))
                }
                (MixerPlan::Delta(dp), LayerStore::Rec(_)) => {
                    Ok(LayerStore::Rec(RecStore::new(stream, dp.shape, LANES)?))
                }
                _ => Err(GpuError::state(
                    WHAT_N,
                    "a live store of each layer's own kind",
                )),
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        let ckpt = Checkpoints::new(
            gpu.context(),
            self.ckpt.lens().to_vec(),
            HOST_BUDGET,
            self.every,
        )?;
        Ok(Slot35 {
            stores,
            held: 0,
            ckpt,
        })
    }

    /// Exchange the live sequence with `seq`: pointer moves only — the
    /// stores are buffer handles, the held count and the checkpoints host
    /// values the body holds by value — so nothing is copied, captured or
    /// synchronized. Nothing refuses: a prompt call waits for its
    /// checkpoints' copies before it returns, and a cut waits in its own
    /// sequence's checkpoints for that sequence's next call
    /// ([`Body35::settle`]), so it travels with it.
    fn swap_seq(&mut self, _gpu: &Gpu, seq: &mut Slot35) -> Result<(), GpuError> {
        self.exchange(seq);
        Ok(())
    }

    /// Device bytes one sequence holds: its layers' stores
    /// ([`Body35::store_bytes`]). Its checkpoints are host bytes, left out:
    /// pinned slots its prompt calls make as they take points, up to
    /// [`HOST_BUDGET`] a sequence and none at [`Slots::new_seq`].
    fn seq_bytes(&self) -> usize {
        self.store_bytes()
    }
}

impl Rollback for Body35 {
    /// A cut to `pos` ([`Body35::cut`]): the checkpoint there planned for
    /// the next call to carry out ([`Body35::settle`]), any other position
    /// behind the stores refused by name. No device work, so the default
    /// `rollback_on` serves.
    fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        self.cut(pos)
    }
}

impl Rows for Body35 {
    const MAX_ROWS: usize = MAX_PASS_ROWS;
    /// No host tier serves a Qwen3.6 replay; the pass is the chain's
    /// one-token layout at `m` rows.
    const CHAIN: Chain = Chain::Step;

    fn plan_rows(&mut self, stream: &CudaStream, tokens: &[u32], pos: u32) -> Result<(), GpuError> {
        self.stand_held(stream, pos, tokens.len())?;
        self.rp.write(stream, tokens, pos)
    }

    fn enqueue_rows(&mut self, gpu: &Gpu, w: &Weights, heads: &mut [Head]) -> Result<(), GpuError> {
        enqueue_rows(gpu, w, self, heads)
    }
}

impl Instrumented for Body35 {
    /// Rows `0..rows` of every attention layer's K and V planes filled with
    /// a deterministic pattern of finite, nonzero values that differ from
    /// row to row — f16 bits on an f16 cache, the q8_0 form of the same
    /// values on a q8_0 one; the delta layers' stores stay as they stand (a
    /// step's cost does not depend on the state's values) — a step shape,
    /// not a model state. The held counter follows the position the
    /// instrument stands the model at, so the step after it runs.
    fn seed_depth(&mut self, gpu: &Gpu, rows: usize) -> Result<(), GpuError> {
        let d = self.s.dims;
        let mut vals = vec![0f32; d.n_kv * d.ctx * d.head];
        let mut state = 0x9e37_79b9u32 ^ rows as u32;
        for h in 0..d.n_kv {
            for r in 0..rows.min(d.ctx) {
                for c in 0..d.head {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let v = ((state >> 9) as f32 / (1u32 << 23) as f32 - 0.5) + 1.0 / 64.0;
                    vals[(h * d.ctx + r) * d.head + c] = v;
                }
            }
        }
        let stream = gpu.stream();
        for s in &mut self.stores {
            if let LayerStore::Kv(p) = s {
                p.fill(stream, &vals, &d)?;
            }
        }
        stream.synchronize()?;
        self.held = u32::try_from(rows)
            .map_err(|_| GpuError::shape(WHAT, format!("a depth of {rows} positions past u32")))?;
        Ok(())
    }
}
