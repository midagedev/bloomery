//! The Qwen3.6-35B-A3B (`qwen35moe`) and Qwen3.5-9B (`qwen35`, the Clef
//! backbone) `ChainBody`: the family's one layer program (the chain's walk
//! over the ops of `dispatch` and `delta`, `program`) over plans that
//! interleave gated-delta-rule layers with gated GQA layers
//! at head 256 — every routed layer's experts carrying the sigmoid-gated
//! shared expert as one more slot of joined stacks, every dense layer's FFN a
//! stack of one expert on the arena's fixed one-slot route. The head counts
//! are the file's; their group selects the flash (eight a block, or pairs).
//! An attention layer's kind comes from the description (`plan::gqa_fits`):
//! the gated head-256 layer, or a head-128 layer turned by NORM pairs with no
//! head norm (a dense Llama's), whose flash is the group's of eight or of
//! seven.
//!
//! Load reads the file's description once (`model::arch::spec`): each
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
//! never a launch argument, so one capture serves every position. A plain
//! load holds one lane, and the word is [`LANE`] in every record. `reset`
//! zeroes every lane and every conv ring: the recurrence is written in
//! place, so the next sequence would read the previous one's state.
//!
//! A load that verifies drafted rows ([`GpuModel::open_verifying`]) holds
//! [`Body35::VERIFY_LANES`] lanes a delta layer, each stamped with the position its
//! state stands at, and runs `gdn_delta_lanes` over the committed lane the
//! body's one lane word names ([`LaneWord`]; the records' words are unread):
//! a step and a prompt unit write the state after their last row back in
//! place, and a verify of `m` rows writes row `j`'s state into lane `(c + j)
//! mod lanes`. Its commit of `accepted` rows moves the word to the lane row
//! `accepted − 1` wrote ([`LaneWord::keep`]) and copies nothing; a step, a
//! pass or a prompt unit is refused by name while a verify waits for its
//! commit, and a cut behind the verify's first position is a cut to a
//! checkpoint of the committed lane, as on a plain load.
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
//!
//! A load opened with [`Open35::media`] takes image rows ([`MediaBody`]): an
//! image's M-RoPE positions shift the rows of its sequence, so each sequence
//! holds a rotation table of its own ([`SeqRope`]), made at load before any
//! graph is captured and exchanged with the sequence; every rope launch reads
//! the table of the sequence it runs. A media call is eager and runs every
//! unit on [`PrefillPath::Wide`]; each unit's first layer copies the rows of
//! the spans inside it over the embeddings of their positions
//! (`image::MediaRows`, the unit's front, `dispatch::unit_front`).

use super::body::{Kernels, KvQ8, TapRows, f32_site};
use super::dispatch::{self, PassCtx};
use super::draft_taps::DraftTaps;
use super::head_argmax::HeadArgmaxState;
use super::image::{
    ImageWrite, MediaCarrier, MediaRows, PromptImage, imrope_sections, media_call, section_order,
};
use super::mtp35::Mtp35;
use super::placed::{
    BatchWalk, CommonForms, PLACED_ROWS, Placed, PlacedOpen, StepWalk, WalkParts, WideRoute,
    apart_plans, common_forms as placed_forms, counted_arena, placed_bytes, placed_scratch,
    prompt_rows, tally_shape,
};
use super::plan::{
    FfnPlan, FfnRoute, Flash, GqaKind, Kind35, LayerPlan, MixerPlan, ResidualPlan, SharedPlan,
    SiteTy, down_sel, gate_up_fused, input_fused, kind35, kinds35, moe_fits, o_fused, out_fused,
    q35, qkv_fused,
};
use super::prefill::{PrefillPath, PrefillPlan, PrefillStep, WIDE_FROM};
use super::program::{Program, Tail};
use super::scratch::{
    ApartForms, Arena, Dims, Forms, KvPlanes, LANE, LaneWord, LayerStore, PassRecord, RecStore,
    RopeRows, StepParams,
};
use super::slot_pass::SlotIn;
use super::swap35;
use super::ubatch::UBATCH;
use super::wide::{GEMV_COLS, arena_bytes};
use super::wide38::STREAM_FLOOR;
use crate::checkpoint::{self, Checkpoints};
use crate::draft::WalkMode;
use crate::folder::{Model, Origin};
use crate::head::Head;
use crate::host::prompt_stream::PromptStream;
use crate::host::swap::{CallPick, CallReport, Residency};
use crate::host::swap_source::{DEADLINE, LIVE_DELAY, ResidencySpec};
use crate::host::xstream::{Costs, XLayer, XMode, XReport, XStreamBody};
use crate::host::{BatchLeg, StepLeg};
use crate::hybrid::{Chain, SlotMap};
use crate::linear;
use crate::media::{ImageSpan, MediaBody, MediaSpan, RopeBase, SeqRope, kept_rule, rope_table};
use crate::model::fit::{FIT_RESERVE, largest_fit};
use crate::model::{
    ChainBody, GpuModel, HostServed, Instrumented, MAX_PASS_ROWS, Rollback, Rows, SlotRange, Slots,
    refuse_stage_rows_past,
};
use crate::rope_table::{RopeSpec, RopeTable};
use crate::route_softmax::RouterDims;
use crate::site::{self, Order, file_site};
use crate::tensor::{WindowMut, host_filled};
use crate::weights::Weights;
use crate::{Gpu, GpuError, launch_u32};
use bloomery_levers::{ChurnHosting, HostCfg};
use chain::arch::qwen3moe::read::{
    PlanDims, Read35, Sites35, joint, joint_gate_inp, layer_plans, plan_ffn, plan_gqa,
};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use gguf::{Split, TensorInfo};
use model::arch::Arch;
use model::arch::models::shape::MoeShape;
use model::arch::models::{Ffn, LayerSpec, Mixer, ModelSpec, MtpDraft, OutGate};
pub(super) use model::arch::names::blk;
use model::placement::{CardFormat, Plan, WholeLoad};
use runtime::prompt::CallEnd;
use runtime::seqstate::{Kept, Price};
use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;
use vision::mrope::MropeSeq;

const WHAT: &str = "qwen35moe::Body35::load";

// The family's no-probe floor ([`Body35::STREAM_FLOOR`]) is past every
// unit the gemv arm walks — a pass's [`MAX_PASS_ROWS`] and the step's one
// row among them — so no such unit picks, and the gemv arm, which reads the
// card copy's row where its front cut its places, never meets a moved map.
const _: () = assert!(GEMV_COLS < STREAM_FLOOR as usize && MAX_PASS_ROWS <= GEMV_COLS);

/// What [`GpuModel::<Body35>::open`](GpuModel::open) takes besides the card
/// and the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Open35 {
    /// KV rows the caches hold.
    pub ctx: usize,
    /// The decode flash: the tensor-core pass when `true`.
    pub mma: bool,
    /// Tokens per ubatch of the prompt call, `1..=` the router's
    /// [`ubatch`](crate::route_softmax::RouterDims::ubatch) (at most [`UBATCH`]);
    /// the ubatch arena holds `max(ubatch, GEMV_COLS)` rows, at most `ctx`.
    /// A size outside that range, or an arena that does not fit the card's
    /// free bytes with [`FIT_RESERVE`] left, is refused by name at load.
    pub ubatch: usize,
    /// The attention layers' KV planes' format ([`KvQ8`]); the delta layers'
    /// recurrent stores are f32 either way.
    pub kv: KvQ8,
    /// Whether the sequences take image rows ([`MediaBody`](crate::media::MediaBody)):
    /// each one holds a rotation table of its own, made here before any
    /// graph is captured (a captured graph bakes its buffer's address), that
    /// an image's positions rewrite ([`SeqRope`]). A text load holds none and
    /// every rope launch reads the load's one table.
    pub media: bool,
}

/// `o` as a placed load opens it: its ubatch is [`PLACED_ROWS`], whatever
/// `o.ubatch` asks.
fn placed_open(o: Open35) -> Open35 {
    Open35 {
        ubatch: PLACED_ROWS,
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
/// prompt runs in units of [`PLACED_ROWS`] instead ([`placed_open`]), and the
/// asked size stays the spacing: its marks cut a unit at most once a mark,
/// a copy a window apart. The size [`ubatch_of`] returns is at most
/// [`UBATCH`], so the `u32` loses nothing.
fn checkpoint_every(asked: usize, d: &Dims) -> Result<u32, GpuError> {
    Ok(ubatch_of(d, asked)?.get() as u32)
}

/// The f32s of each store one checkpoint copies, in [`Copied35`]'s order:
/// each delta layer's committed state lane, then its conv ring — one lane's
/// state whatever the load's lane count. The one derivation of what a
/// checkpoint holds, read by the sequences' `Checkpoints` and by
/// [`checkpoint_price35`].
fn checkpoint_lens(kinds: &[Kind35]) -> Vec<usize> {
    kinds
        .iter()
        .filter_map(|k| match k {
            Kind35::Delta(shape) => Some(shape.lens(PLAIN_LANES)),
            Kind35::Gqa(..) => None,
            Kind35::ShortConv(_) => {
                panic!("qwen35moe::checkpoint_lens: a short-conv layer, which `kinds35` refuses")
            }
        })
        .flatten()
        .collect()
}

/// One sequence's checkpoints of stores of `lens` f32s: the ledger's
/// ([`Checkpoints::sequence`]), or — when the description holds no recurrent
/// layer, so `lens` is empty — none to take ([`Checkpoints::stateless`]): the
/// K/V planes are cut by position and a cut keeps every position it asks for.
fn sequence_checkpoints(
    cuda: &Arc<CudaContext>,
    lens: Vec<usize>,
    ctx: usize,
    every: u32,
) -> Result<Checkpoints, GpuError> {
    if lens.is_empty() {
        Ok(Checkpoints::stateless(cuda, every))
    } else {
        Checkpoints::sequence(cuda, lens, ctx, every)
    }
}

/// What one checkpoint of a load of `spec`'s model costs on the host and how
/// a prompt call spaces them, the load asked for ubatches of `asked` tokens
/// ([`checkpoint_every`]): the figure every sequence's `Checkpoints` is made
/// with ([`Checkpoints::sequence`]) and a plan's host reserve prices
/// (`residency38::reserve_checkpoints`), so the two cannot differ. The
/// spacing reads no cache row, so the dims are read at one position.
pub fn checkpoint_price35(spec: &ModelSpec, asked: usize) -> Result<Price, GpuError> {
    let kinds = kinds35(&spec.layers)?;
    let every = checkpoint_every(asked, &dims(spec, &kinds, 1)?)?;
    Ok(checkpoint::price(&checkpoint_lens(&kinds), every))
}

/// The host bytes one sequence's checkpoints pin at most in a load of `ctx`
/// positions (a slot's share) of `spec`'s model asked for ubatches of
/// `asked` tokens ([`Price::budget`] of [`checkpoint_price35`]).
pub fn checkpoint_budget35(spec: &ModelSpec, asked: usize, ctx: u64) -> Result<u64, GpuError> {
    let price = checkpoint_price35(spec, asked)?;
    if price.point_bytes == 0 {
        // No layer holds a recurrent store: nothing is copied, nothing pinned.
        return Ok(0);
    }
    checkpoint::budget(price, ctx)
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
/// card's `free` bytes ([`Gpu::alloc_room`]), else a named refusal with the
/// free bytes, the need and the largest ubatch that fits, before anything is
/// allocated. The arena it allocates holds the bytes the check counted, or
/// the load fails by name. Load-time allocation.
fn ubatch_arena(
    stream: &CudaStream,
    (d, forms): (Dims, Forms),
    u: usize,
    free: usize,
) -> Result<Arena, GpuError> {
    const WHAT_FIT: &str = "qwen35moe::ubatch_arena";
    let (rows, need) = ubatch_need(&d, forms, u);
    if need + FIT_RESERVE > free {
        let fits = largest_fit(
            (1..u).rev(),
            |v| ubatch_need(&d, forms, v).1,
            free,
            FIT_RESERVE,
        )
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

/// Lanes of every recurrent store on a plain load: one, read and written in
/// place.
const PLAIN_LANES: usize = 1;

/// A verify of drafted rows still waiting for its commit: the position of its
/// first row and its row count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Pending35 {
    pos0: u32,
    rows: usize,
}

/// Qwen3.6's per-replay host values: the token and its position.
pub struct DecodeInput35 {
    token: u32,
    pos: u32,
}

/// Every Qwen3.6 layer's kind, as a gate lists them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind35 {
    /// Grouped-query attention over the layer's K/V planes.
    Attention,
    /// The gated delta rule over the layer's recurrent store.
    Delta,
}

/// Everything Qwen3.6's chain owns on the device.
pub struct Body35 {
    pub(super) vocab: usize,
    pub(super) eps: f32,
    pub(super) plans: Vec<LayerPlan>,
    pub(super) stores: Vec<LayerStore>,
    /// The rope table every attention layer reads by position, 64 f32 a row.
    pub(super) rope: RopeRows,
    /// The live sequence's rotation table on a media load: the rope launches
    /// of the sequence read it in place of `rope` ([`rope_table`]). `None`
    /// on a text load.
    pub(super) seq_rope: Option<SeqRope>,
    /// What a media load makes each further sequence's table from
    /// ([`Slots::new_seq`]); `None` on a text load.
    rope_base: Option<RopeBase>,
    /// The decode step's one-row arena and its input record.
    pub(super) s: Arena,
    pub(super) sp: StepParams,
    /// A pass's arena of [`MAX_PASS_ROWS`] rows and its input record.
    pub(super) a: Arena,
    pub(super) rp: PassRecord<MAX_PASS_ROWS>,
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
    /// The rows entering the layers a block draft reads, which every walk of
    /// a whole-card load lays in ([`GpuModel::set_draft_taps`]).
    pub(super) draft: Option<DraftTaps>,
    /// A placed load's placed side ([`Placed`]): the host tier and what the
    /// walks over its legs read; `None` on a whole-card load.
    pub(super) placed: Option<Placed>,
    /// The file's own next-token layer ([`GpuModel::open_mtp`]); `None` on a
    /// load that did not ask for it.
    pub(super) mtp: Option<Mtp35>,
    /// Lanes of every delta store: 1 on a plain load, [`Body35::VERIFY_LANES`]
    /// on a load that verifies drafted rows ([`GpuModel::open_verifying`],
    /// [`GpuModel::open_mtp`]).
    lanes: usize,
    /// The word naming the committed lane of every delta launch's walk; `Some`
    /// iff `lanes > 1`. The arenas hold views of it ([`Arena::run_lanes`]).
    lane: Option<LaneWord>,
    /// The verify waiting for its commit (`lanes > 1` only).
    pending: Option<Pending35>,
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
    /// The prompt call's host streaming (`BLOOMERY_XSTREAM`): off until a
    /// caller sets it ([`Body35::set_xstream`]).
    stream: PromptStream,
}

/// One Qwen3.6 sequence's own state ([`Slots`], what a select exchanges):
/// every layer's store — an attention layer's K/V planes, a delta layer's
/// recurrent state and conv ring —, the positions its recurrent stores hold,
/// its checkpoints of those stores on the host, and on a media load its
/// rotation table. No capture travels with it: the body records none over the stores (the step's and the passes'
/// are the model's, kept a slot).
pub struct Slot35 {
    pub(super) stores: Vec<LayerStore>,
    held: u32,
    ckpt: Checkpoints,
    /// The sequence's rotation table on a media load ([`Body35::seq_rope`]).
    pub(super) rope: Option<SeqRope>,
}

/// The delta layers' stores a checkpoint copies, lent for one call: each
/// layer's state lane `lane` (the committed one) as a window of its lanes,
/// and its conv ring.
struct Copied35<'a> {
    lanes: Vec<WindowMut<'a, f32>>,
    rings: Vec<&'a mut DeviceBuffer<f32>>,
}

impl<'a> Copied35<'a> {
    /// The stores of `stores` a checkpoint copies. Refused by name for a lane
    /// the stores do not hold.
    fn of(stores: &'a mut [LayerStore], lane: u32) -> Result<Copied35<'a>, GpuError> {
        let mut c = Copied35 {
            lanes: Vec::new(),
            rings: Vec::new(),
        };
        for s in stores {
            if let LayerStore::Rec(r) = s {
                let RecStore {
                    state, ring, lanes, ..
                } = r;
                if lane as usize >= *lanes {
                    return Err(GpuError::shape(
                        "qwen35moe::Body35::copied",
                        format!("lane {lane} of {lanes} lanes"),
                    ));
                }
                let len = state.len() / *lanes;
                let byte = lane as usize * len * size_of::<f32>();
                c.lanes.push(WindowMut::of_mut(state, byte, len)?);
                c.rings.push(ring);
            }
        }
        Ok(c)
    }

    /// The stores in [`Checkpoints::sequence`]'s order: each delta layer's
    /// lane, then its ring.
    fn list(&mut self) -> Vec<&mut DeviceBuffer<f32>> {
        let mut out: Vec<&mut DeviceBuffer<f32>> = Vec::new();
        for (l, r) in self.lanes.iter_mut().zip(self.rings.iter_mut()) {
            out.push(&mut **l);
            out.push(&mut **r);
        }
        out
    }
}

/// The types the embedding and the head read (`crate::site`): the embedding
/// Q3_K, Q4_K, Q5_K or Q6_K rows, Q8_0 planes or BF16 words (a checkpoint folder's table as
/// stored); the head Q6_K, Q4_K, Q8_0 or NVFP4 (`Head`).
const EMBED: &[SiteTy] = &[
    SiteTy::Q3K,
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Q8_0,
    SiteTy::Bf16,
];
const HEAD_TY: &[SiteTy] = &[SiteTy::Q6K, SiteTy::Q4K, SiteTy::Q8_0, SiteTy::Nvfp4];

/// The forms the layers the common card leg runs ([`model::placement::accepts::LegSplit::Common`]) hold
/// buffers of, from `file`'s header: [`CommonForms::NONE`] unless the read is a
/// placed one, which is the only one a card leg runs a layer of by it.
fn common_forms(
    file: &Split,
    d: &Dims,
    layers: &[LayerSpec],
    read: Read35,
) -> Result<CommonForms, GpuError> {
    if read != Read35::Placed {
        return Ok(CommonForms::NONE);
    }
    let moe = layers
        .iter()
        .enumerate()
        .filter_map(|(l, layer)| match &layer.ffn {
            Ffn::Moe(m) => Some((l, m.experts as usize)),
            Ffn::Dense { .. } => None,
        });
    placed_forms(file, WHAT, moe, (d.ff, d.hidden))
}

/// Layer `l`'s router joined with its shared gate, and, when its plan `f`
/// folds the shared expert in ([`SharedPlan::Joined`]), its routed stacks
/// joined with the shared expert (`Weights::join_rows`: the parts leave the
/// map); a shared expert kept apart stays its own three sites, and a dense
/// layer has nothing to join.
fn join_ffn(stream: &CudaStream, w: &mut Weights, f: &FfnPlan, l: usize) -> Result<(), GpuError> {
    let FfnRoute::Router {
        shared: Some(shared),
        ..
    } = &f.route
    else {
        return Ok(());
    };
    if let SharedPlan::Joined = shared {
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
    }
    join_gate_inp(stream, w, l)
}

/// Layer `l`'s router weight with its shared gate as the last row.
fn join_gate_inp(stream: &CudaStream, w: &mut Weights, l: usize) -> Result<(), GpuError> {
    w.join_rows(
        stream,
        &[
            blk(l, "ffn_gate_inp.weight").as_str(),
            blk(l, "ffn_gate_inp_shexp.weight").as_str(),
        ],
        joint_gate_inp(l),
    )
}

/// What a layer's plan reads of the arena's dims: the rows its sites are read at.
fn plan_dims(d: &Dims) -> PlanDims {
    PlanDims {
        hidden: d.hidden,
        ff: d.ff,
        q_rows: d.q_rows,
        kv_len: d.kv_len(),
        attn_len: d.attn_len(),
    }
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
            if g.kind.head_norm() {
                f32_site(w, &g.attn_q_norm, 1, d.head)?;
                f32_site(w, &g.attn_k_norm, 1, d.head)?;
            }
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
        MixerPlan::ShortConv(_) => {
            return Err(GpuError::shape(
                WHAT,
                "a short-conv layer, which this body runs no kernel for",
            ));
        }
    }
    let f = &p.ffn;
    let e = match &f.route {
        FfnRoute::Router { gate_inp, shared } => {
            let r = d.routed(WHAT)?;
            f32_site(w, gate_inp, r.logits(), h)?;
            match shared {
                Some(SharedPlan::Joined) => r.logits(),
                Some(SharedPlan::Apart(_)) | None => r.experts(),
            }
        }
        FfnRoute::Dense => 1,
    };
    f32_site(w, &f.ffn_norm, 1, h)?;
    if !stacks {
        return Ok(());
    }
    on(&f.gate, e * ff, h, f.gate_ty)?;
    on(&f.up, e * ff, h, f.up_ty)?;
    on(&f.down, e * h, ff, f.down_ty)?;
    if let Some(sh) = f.apart() {
        on(&sh.gate, ff, h, sh.gate_ty)?;
        on(&sh.up, ff, h, sh.up_ty)?;
        on(&sh.down, h, ff, sh.down_ty)?;
    }
    Ok(())
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
    join_gate_inp(stream, w, l)
}

impl Forms {
    /// What the arena of a chain of `plans` holds beyond a fused K-quant
    /// chain's buffers (`Forms`'s doc), `d` its dims: every form one of the
    /// sites reads, the gate and up rows of an unfused gate·up, and the
    /// longest row-major output of a K-quant site launched alone.
    pub(super) fn of(plans: &[LayerPlan], d: &Dims) -> Forms {
        let mut hid: Vec<SiteTy> = Vec::new();
        let (mut attn, mut h, mut glu, mut cols) = (Vec::new(), Vec::new(), false, 0usize);
        let mut apart: Vec<SiteTy> = Vec::new();
        let mut out_gate = false;
        let mut alone = |ty: SiteTy, rows: usize| {
            if site::kgemv_order(ty) == Some(Order::RowMajor) {
                cols = cols.max(rows);
            }
        };
        for p in plans {
            match &p.mixer {
                MixerPlan::Gqa(g) => {
                    hid.extend([g.q_ty, g.k_ty, g.v_ty]);
                    attn.push(g.o_ty);
                    out_gate |= g.kind == GqaKind::Gated256 && !g.o_ty.kquant();
                    if !qkv_fused(g) {
                        alone(g.q_ty, d.q_rows);
                        alone(g.k_ty, d.kv_len());
                        alone(g.v_ty, d.kv_len());
                    }
                    if !o_fused(g) {
                        alone(g.o_ty, d.hidden);
                    }
                }
                MixerPlan::Delta(dp) => {
                    hid.extend([dp.qkv_ty, dp.gate_ty, dp.beta_ty, dp.alpha_ty]);
                    attn.push(dp.out_ty);
                    if !input_fused(dp) {
                        alone(dp.qkv_ty, dp.shape.channels());
                        alone(dp.gate_ty, dp.shape.n_v * linear::HEAD);
                        alone(dp.beta_ty, dp.shape.n_v);
                        alone(dp.alpha_ty, dp.shape.n_v);
                    }
                    if !out_fused(dp) {
                        alone(dp.out_ty, d.hidden);
                    }
                }
                MixerPlan::ShortConv(_) => {
                    panic!("qwen35moe::Forms::of: a short-conv layer, which `kinds35` refuses")
                }
            }
            let f = &p.ffn;
            hid.extend([f.gate_ty, f.up_ty]);
            h.push(f.down_ty);
            // A shared expert kept apart: its gate and up read the normed
            // rows, its down its own SwiGLU rows, each site launched alone.
            if let Some(sh) = f.apart() {
                hid.extend([sh.gate_ty, sh.up_ty]);
                apart.push(sh.down_ty);
                alone(sh.gate_ty, d.ff);
                alone(sh.up_ty, d.ff);
                alone(sh.down_ty, d.hidden);
            }
            if !gate_up_fused(f) {
                glu = true;
                alone(f.gate_ty, d.slots() * d.ff);
                alone(f.up_ty, d.slots() * d.ff);
            }
            if !down_sel(f) {
                alone(f.down_ty, d.hidden);
            }
        }
        let wants = crate::site::WideActs::wants_of;
        Forms {
            // An input's f32 rows are the arena's own normed rows.
            hid: wants(&hid).without_f32(),
            attn: wants(&attn).without_f32(),
            h: wants(&h),
            glu,
            cols,
            // The shared expert's f32 rows are held whatever its down reads: the
            // wide part counts them (`wide::arena_bytes`), and the plan's card
            // bytes with it.
            apart: (!apart.is_empty()).then(|| ApartForms {
                h: crate::site::WideForms {
                    f32: true,
                    ..wants(&apart)
                },
            }),
            out_gate,
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
    /// Values of a head the rope turns, the description's.
    rope_dims: usize,
    /// The file's IMROPE sections and `rope.section_order`, read when the
    /// load takes media.
    sections: Option<([u32; 4], vision::mrope::SectionOrder)>,
    /// The forms the layers the common card leg runs hold buffers of
    /// ([`CommonForms::NONE`] under a whole read).
    common: CommonForms,
    /// The card format the description gives each tensor
    /// (`ModelTensors::card_format`, the one the whole-fit verdict sizes), which
    /// a whole load uploads it in.
    formats: BTreeMap<String, CardFormat>,
}

impl Pre35 {
    /// [`Pre35::read`] for a placed load ([`Read35::Placed`]): a layer whose
    /// stacks the card leg does not run is read as the host tier's, and the
    /// plans keep every routed layer's shared expert apart from the stacks
    /// ([`apart_plans`]).
    fn read_placed(file: &Split, o: Open35) -> Result<Pre35, GpuError> {
        let pre = Pre35::read(file, o, Read35::Placed)?;
        Ok(Pre35 {
            plans: apart_plans(pre.plans),
            ..pre
        })
    }

    /// The load's description from `file`'s header, its routed stacks held as
    /// `mode` says ([`Read35`]).
    fn read(file: &Split, o: Open35, mode: Read35) -> Result<Pre35, GpuError> {
        let Open35 {
            ctx,
            mma,
            ubatch,
            media,
            ..
        } = o;
        let read = model::arch::spec(file).map_err(|e| GpuError::plan(WHAT, e))?;
        let formats = read
            .tensors
            .tensors
            .iter()
            .filter_map(|t| Some((t.name.clone(), read.tensors.card_format(t)?)))
            .collect();
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
        let (base, rope_dims) = spec
            .layers
            .iter()
            .find_map(|s| match &s.mixer {
                Mixer::Gqa(g) => Some((g.rope.base, g.rope.dims as usize)),
                _ => None,
            })
            .ok_or(GpuError::shape(
                WHAT,
                "no attention layer to read the rope base from",
            ))?;
        let sections = media
            .then(|| {
                let sections = imrope_sections(&spec).ok_or(GpuError::shape(
                    WHAT,
                    "a media load of a file with no IMROPE attention layer: an image's positions \
                     have no sections to follow",
                ))?;
                let order = model::arch::qwen35moe::hparams::Hparams::read(file)
                    .map_err(|e| GpuError::plan(WHAT, e))?
                    .rope_order;
                Ok::<_, GpuError>((sections, section_order(order)))
            })
            .transpose()?;
        let has = |flash: Flash| {
            kinds
                .iter()
                .any(|k| matches!(k, Kind35::Gqa(_, f) if *f == flash))
        };
        if mma && has(Flash::Pairs) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the tensor-core decode flash at {}/{} heads: the pairs' pass is scalar only \
                     (open with mma false)",
                    d.n_head, d.n_kv
                ),
            ));
        }
        if !mma && has(Flash::Seven) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the scalar decode flash at {}/{} heads of {} (a group of seven): the shape \
                     table's pack-7 row has the tensor-core entry gqa_flash_seg_mma_g7 only and \
                     no scalar one (open with mma true)",
                    d.n_head, d.n_kv, d.head
                ),
            ));
        }
        let vocab = spec.vocab as usize;
        file_site(file, WHAT, "token_embd.weight", (vocab, d.hidden), EMBED)?;
        file_site(
            file,
            WHAT,
            crate::weights::head_tensor(file),
            (vocab, d.hidden),
            HEAD_TY,
        )?;
        let common = common_forms(file, &d, &spec.layers, mode)?;
        let pd = plan_dims(&d);
        let plans = layer_plans(file, WHAT, &pd, &kinds, &spec.layers, &Sites35::CARD, mode)?;
        Ok(Pre35 {
            vocab,
            eps: spec.rms_eps,
            layers: spec.layers,
            kinds,
            plans,
            d,
            ubatch,
            base,
            rope_dims,
            sections,
            common,
            formats,
        })
    }
}

/// The arena's dims of `spec`, whose kinds `kinds` the plan checked: the
/// attention layers' head from the description, with a gate beside each query
/// when the layers carry one (every attention layer of a file shares one
/// head count, head and gate), the FFN's — the router
/// the routed shape selects (every layer of a file shares it), or a dense
/// width every layer shares, never both kinds in one file — and the delta
/// layers' one shape (every delta layer of a file shares it). A delta
/// layer's gated norm writes the attention rows' buffer, so the two widths
/// must agree.
fn dims(spec: &ModelSpec, kinds: &[Kind35], ctx: usize) -> Result<Dims, GpuError> {
    let mut shapes = kinds.iter().filter_map(|k| match k {
        Kind35::Delta(s) => Some(*s),
        Kind35::Gqa(..) | Kind35::ShortConv(_) => None,
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
    let mut attn = spec.layers.iter().filter_map(|l| match &l.mixer {
        Mixer::Gqa(g) => Some((
            g.heads as usize,
            g.kv_heads as usize,
            g.head_dim as usize,
            g.out_gate == OutGate::Packed,
        )),
        _ => None,
    });
    let (n_head, n_kv, head, packed) = attn.next().ok_or(GpuError::shape(
        WHAT,
        "no attention layer to size the attention rows for",
    ))?;
    if let Some(other) = attn.find(|o| *o != (n_head, n_kv, head, packed)) {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "two attention shapes, {n_head}/{n_kv} heads of {head} (gate {packed}) and \
                 {other:?}; one arena serves one"
            ),
        ));
    }
    let (router, ff) = ffn_dims(spec)?;
    let d = Dims {
        hidden: spec.hidden as usize,
        n_head,
        n_kv,
        head,
        q_rows: n_head * head * if packed { 2 } else { 1 },
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
fn ffn_dims(spec: &ModelSpec) -> Result<(Option<RouterDims>, usize), GpuError> {
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
            let router = moe_fits(first, WHAT)?;
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

/// The layers a load of `file` runs: its trunk, the blocks before the
/// next-token layers its header declares (read as unused), as the file's
/// role table counts them.
fn trunk_count(file: &Split, what: &'static str) -> Result<usize, GpuError> {
    let read = model::arch::spec(file).map_err(|e| GpuError::plan(what, e))?;
    Ok(read.tensors.layers)
}

/// The next-token layer `file` carries, or why it has none: the family's
/// rule ([`model::arch::draft::draft_usable`]) states every refusal, before
/// anything is read from the card.
fn draft_of(file: &Split, o: Open35) -> Result<Box<MtpDraft>, GpuError> {
    const WHAT_D: &str = "qwen35moe GpuModel::open_mtp";
    if o.media {
        return Err(GpuError::shape(
            WHAT_D,
            "a media load: its sequences' rotation tables are their own, and the draft reads the \
             plain table",
        ));
    }
    model::arch::draft::draft_usable(file, None).map_err(|e| GpuError::plan(WHAT_D, e))?;
    match model::arch::qwen35moe::mtp::in_file(file).map_err(|e| GpuError::plan(WHAT_D, e))? {
        model::arch::qwen35moe::mtp::InFile::Draft(d) => Ok(d),
        other => Err(GpuError::shape(
            WHAT_D,
            format!("the file's next-token layer reads as {other:?} after the rule admitted it"),
        )),
    }
}

/// The tail block of `file` as the layer `draft` describes it: its plan over
/// the dims `d` — the trunk's attention, so the one the trunk's layers take —
/// and its tensors (`blk.{index}.*`, the layer's and the join's and head's
/// own stems) on the card, the router joined with the shared expert's gate
/// and the stacks with the shared expert where the plan folds it in, every
/// site held to the plan ([`check_resident`]).
fn draft_layer(
    gpu: &Gpu,
    file: &Split,
    d: &Dims,
    draft: &MtpDraft,
) -> Result<(Weights, LayerPlan), GpuError> {
    let l = draft.index as usize;
    let Kind35::Gqa(kind, flash) = kind35(&draft.layer, l)? else {
        return Err(GpuError::shape(
            WHAT,
            format!("layer {l}: the next-token layer is full attention, not a delta rule"),
        ));
    };
    let pd = plan_dims(d);
    let plan = LayerPlan {
        mixer: MixerPlan::Gqa(plan_gqa(file, WHAT, &pd, l, (kind, flash), &Sites35::CARD)?),
        ffn: plan_ffn(
            file,
            WHAT,
            &pd,
            &draft.layer,
            l,
            &Sites35::CARD,
            Read35::Whole,
        )?,
        residual: ResidualPlan::Plain,
    };
    let prefix = blk(l, "");
    let mut w = Weights::load_where(gpu.stream(), file, |name| name.starts_with(&prefix))?;
    join_ffn(gpu.stream(), &mut w, &plan.ffn, l)?;
    check_resident(&w, d, &plan, true)?;
    Ok((w, plan))
}

/// The header a whole-card load of `file` plans by and where its card tensors come from
/// ([`Model::into_whole`]); the next-token layer (`mtp`) of a checkpoint folder is refused by
/// name, since the folder's map leaves its `mtp.*` tensors out.
fn whole_origin(file: Model, mtp: bool) -> Result<(Split, Origin), GpuError> {
    let (header, origin) = file.into_whole();
    if mtp && matches!(origin, Origin::Folder { .. }) {
        return Err(GpuError::shape(
            "qwen35moe GpuModel::open_mtp",
            "a checkpoint folder's next-token layer is not read: its mtp.* tensors are left out \
             of the load",
        ));
    }
    Ok((header, origin))
}

/// The split a placed load reads: the plan's card segments and the host tier's experts come from
/// the file's own mapped tensors, and a checkpoint folder's tensors come through a part source
/// that only a whole-card load reads, so a folder is refused by name.
fn placed_split(file: Model, what: &'static str) -> Result<Split, GpuError> {
    match file {
        Model::Split(file) => Ok(file),
        Model::Folder(_) => Err(GpuError::shape(
            what,
            "a checkpoint folder loads whole on the card; a placed load reads a GGUF file",
        )),
    }
}

impl GpuModel<Body35> {
    /// The whole Qwen3.6 model of `file`, a GGUF split or a checkpoint folder
    /// ([`Model`]: a folder's card tensors are read through its part source,
    /// every layer and the head), resident on `gpu`, plus the output head,
    /// with caches of `o.ctx` rows, the decode flash `o.mma` (the
    /// tensor-core pass when `true`) and ubatches of `o.ubatch` tokens
    /// ([`Open35`]). Every site's type is read from the file's header and a
    /// type no launch reads refused by name before any upload. The model
    /// takes the file.
    pub fn open(gpu: Gpu, file: impl Into<Model>, o: Open35) -> Result<GpuModel<Body35>, GpuError> {
        GpuModel::open_lanes(gpu, file.into(), o, PLAIN_LANES, false)
    }

    /// [`GpuModel::open`] with the file's own next-token layer resident
    /// beside the trunk ([`Mtp35`]): the tail block's tensors on the card with
    /// their own K/V planes and arena, the target's `token_embd` and `output`
    /// read in place ([`GpuModel::mtp_draft`]). The file is asked first
    /// whether it carries a draft — the reason a file with none is refused is
    /// `model::arch::draft::draft_usable`'s, before any upload — and a media
    /// load, whose rotation tables are per sequence, is refused by name.
    ///
    /// The load holds [`Body35::VERIFY_LANES`] lanes a delta store, as
    /// [`GpuModel::open_verifying`]'s does, with its refusals: the layer's
    /// every proposal is verified by a pass of several rows, whose commit
    /// keeps any `1..=m` of them, and one lane holds only the last row's
    /// state. A checkpoint folder is refused by name: its map leaves the
    /// next-token layer's tensors out ([`whole_origin`]).
    pub fn open_mtp(
        gpu: Gpu,
        file: impl Into<Model>,
        o: Open35,
    ) -> Result<GpuModel<Body35>, GpuError> {
        GpuModel::open_lanes(gpu, file.into(), o, Body35::VERIFY_LANES, true)
    }

    /// [`GpuModel::open`] for a load that will verify drafted rows
    /// ([`Rows`]): each delta layer keeps [`Body35::VERIFY_LANES`] stamped
    /// lanes of state, so a verify of up to that many rows keeps any `1..=m`
    /// of them at its commit without a re-feed or a copy (module doc). The
    /// plan prices the lanes through `KvLayout::of_spec_lanes` at the same count
    /// (`model::arch::qwen35moe::place::PlanInputs::with_lanes`); the load's
    /// delta layers' stores, with their stamps, are held to those bytes by
    /// name ([`Body35::store_bytes`]). A plain [`GpuModel::open`] keeps its one lane
    /// and its `gdn_delta` launch, byte for byte. Refused by name past
    /// [`GpuModel::open`]'s refusals: resident slots beyond the live
    /// sequence ([`Slots::new_seq`]) and a pass of several slots.
    pub fn open_verifying(
        gpu: Gpu,
        file: impl Into<Model>,
        o: Open35,
    ) -> Result<GpuModel<Body35>, GpuError> {
        GpuModel::open_lanes(gpu, file.into(), o, Body35::VERIFY_LANES, false)
    }

    /// The whole-card load with `lanes` lanes a delta store, and the file's
    /// next-token layer when `mtp`.
    fn open_lanes(
        gpu: Gpu,
        file: Model,
        o: Open35,
        lanes: usize,
        mtp: bool,
    ) -> Result<GpuModel<Body35>, GpuError> {
        let (file, origin) = whole_origin(file, mtp)?;
        let n_layers = trunk_count(&file, "qwen35moe GpuModel::open")?;
        let draft = mtp.then(|| draft_of(&file, o)).transpose()?;
        let mut pre = Pre35::read(&file, o, Read35::Whole)?;
        if pre.layers.len() != n_layers {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the description holds {} layers, the file's trunk {n_layers}",
                    pre.layers.len()
                ),
            ));
        }
        let formats = std::mem::take(&mut pre.formats);
        let card = |t: &TensorInfo| formats.get(&t.name).copied();
        GpuModel::load_blocks_from(
            gpu,
            &file,
            (origin.bytes(), &card),
            o.ctx,
            0..n_layers,
            true,
            |gpu, w| {
                // Before the body, so its ubatch arena is sized from what the
                // draft leaves free.
                let d = pre.d;
                let mtp = draft
                    .map(|draft| {
                        let layer = draft_layer(gpu, &file, &d, &draft)?;
                        Mtp35::open(gpu, w, &draft, layer, (d, o.kv))
                    })
                    .transpose()?;
                let mut body = Body35::load(gpu, w, pre, o, lanes)?;
                body.mtp = mtp;
                Ok(body)
            },
        )
    }

    /// The card bytes a placed load of `file` opened with `o` holds past the
    /// m = 1 scratch: its ubatch arena ([`arena_bytes`] at the placed
    /// ubatch, [`placed_open`]) and its placed side ([`placed_bytes`]) —
    /// what the plan's machine counts in the card's scratch
    /// (`model::placement::workstation::placed_with_arena`), and what
    /// [`GpuModel::open_placed`] refuses to pass.
    ///
    /// Read as a placed load reads the file ([`Read35::Placed`]): a layer
    /// whose routed stacks the card leg does not run holds no card expert,
    /// so its stacks size no scratch, and a stack of a type no tier runs is
    /// refused by name here, before any plan.
    pub fn placed_arena_bytes(file: &Split, o: Open35) -> Result<u64, GpuError> {
        let pre = Pre35::read_placed(file, placed_open(o))?;
        placed_scratch(
            &pre.d,
            pre.plans.len(),
            (Forms::of(&pre.plans, &pre.d), pre.common),
        )
    }

    /// What a whole load of `file` opened with `o` ([`GpuModel::open`])
    /// holds past its weights, stores and step arenas, as its own fit check
    /// counts it: its ubatch arena at `o.ubatch` ([`ubatch_need`]), checked
    /// against the card's free bytes, and [`FIT_RESERVE`], which the check
    /// keeps free past it. Read through the load's own description, so a
    /// file or options the load refuses are refused here by the same name.
    pub fn whole_load(file: &Split, o: Open35) -> Result<WholeLoad, GpuError> {
        let pre = Pre35::read(file, o, Read35::Whole)?;
        let forms = Forms::of(&pre.plans, &pre.d);
        Ok(WholeLoad::Checked {
            arena: ubatch_need(&pre.d, forms, pre.ubatch.get()).1 as u64,
            reserve: FIT_RESERVE as u64,
        })
    }

    /// The Qwen3.6 model of `file` placed by `plan` (made by
    /// `model::arch::qwen35moe::place::PlanInputs::plan_rule` under
    /// `placement::accepts::routed_rule` of the qwen35moe body, on the machine
    /// `model::placement::workstation::placed_with_arena` lays out): the plan's card
    /// segments resident ([`GpuModel::load_placed_with`]: the trunk, the
    /// shared experts, each layer's card experts) with each router joined
    /// with its shared gate, the host set read in and locked as `host` asks,
    /// and the body over them with its placed side ([`Placed`]), caches of
    /// `o.ctx` rows — the plan's context — and the ubatch arena of the placed
    /// ubatch ([`placed_open`]) held to the bytes the plan counts
    /// ([`GpuModel::placed_arena_bytes`]); the checkpoints' spacing
    /// ([`checkpoint_every`]) is the one part that reads `o.ubatch`. The load
    /// runs no residency machine ([`Residency::Off`]): the plan's slot map is
    /// the model's for its life. Refused by name as [`GpuModel::open`]
    /// refuses, except that a layer whose routed stacks the card leg does not
    /// run is the host tier's ([`Read35::Placed`]) and a routed stack of a
    /// type no tier runs is refused instead; for a plan of another context,
    /// as the placed side refuses ([`Placed::new`]), and as the family's
    /// stack table refuses a layer's types ([`swap35::stacks`]), and as a
    /// checkpoint folder is refused ([`placed_split`]).
    pub fn open_placed(
        file: impl Into<Model>,
        plan: &Plan<'_>,
        o: Open35,
        host: HostCfg,
    ) -> Result<GpuModel<Body35>, GpuError> {
        GpuModel::open_placed_residency(file, plan, o, host, Residency::Off)
    }

    /// [`GpuModel::open_placed`] under `residency`: `mid-p<P>-s<S>` runs the
    /// residency machine over the card's routed stacks ([`crate::host::swap`],
    /// [`swap35::stacks`]) — the load's host set also holds each layer's
    /// churn pool (the card's experts past the first `P`), refused by name
    /// when the plan's host headroom cannot take it — while `off` is
    /// [`GpuModel::open_placed`] itself. The lever's word a load runs by is
    /// the registry's, read by the caller
    /// (`bloomery_levers::Levers::residency_at`). Under a machine the card
    /// copy of the slot map moves between passes: a prompt call is one
    /// residency pass its caller takes (`prompt_pass`; the app's `Prompt`
    /// impl), the wide units' maps follow the card copy ([`Placed`]'s
    /// `refresh_maps`), and a step's boundary is the step loop's. The body has
    /// no steps-feed path to refuse.
    pub fn open_placed_residency(
        file: impl Into<Model>,
        plan: &Plan<'_>,
        o: Open35,
        host: HostCfg,
        residency: Residency,
    ) -> Result<GpuModel<Body35>, GpuError> {
        const WHAT_P: &str = "qwen35moe GpuModel::open_placed";
        let file = placed_split(file.into(), WHAT_P)?;
        let asked = o.ubatch;
        let o = placed_open(o);
        let n_layers = trunk_count(&file, WHAT_P)?;
        if u64::try_from(o.ctx).ok() != Some(plan.ctx_max) {
            return Err(GpuError::shape(
                WHAT_P,
                format!(
                    "caches of {} rows on a plan of {} positions",
                    o.ctx, plan.ctx_max
                ),
            ));
        }
        let pre = Pre35::read_placed(&file, o)?;
        if pre.layers.len() != n_layers {
            return Err(GpuError::shape(
                WHAT_P,
                format!(
                    "the description holds {} layers, the file's trunk {n_layers}",
                    pre.layers.len()
                ),
            ));
        }
        let card = plan
            .machine
            .cards
            .first()
            .ok_or(GpuError::shape(WHAT_P, "a plan of no card"))?;
        let counted = model::placement::workstation::counted_arena_bytes(card);
        let specs = pre.layers.clone();
        let (top_k, max_rows) = tally_shape(&pre.d)?;
        // The stack table checks only the layers the plan gives the stage card
        // experts. The tier cards are named so a routed segment on one is the
        // tier's, not a card the map refuses.
        let n_expert = usize::try_from(plan.model.experts).map_err(|_| {
            GpuError::shape(WHAT_P, "a plan of more experts than the host addresses")
        })?;
        let tier_cards: Vec<usize> = (0..plan.machine.expert_tiers().count())
            .map(|i| plan.machine.cards.len() + i)
            .collect();
        let carded = SlotMap::of_plan_tiers(plan, 0, &tier_cards, 0..n_layers, n_expert)?;
        let residency = ResidencySpec {
            lever: residency,
            pool: ChurnHosting::Hosted,
            delay: LIVE_DELAY,
            deadline: DEADLINE,
            top_k,
            max_rows,
            stacks: Arc::new(swap35::stacks(plan.model, |l| Ok(carded.on_card(l)? > 0))?),
        };
        GpuModel::load_placed_with(
            file,
            plan,
            0,
            host,
            residency,
            |stream, _, layers, w| {
                for l in layers {
                    let spec = specs.get(l).ok_or_else(|| {
                        GpuError::shape(WHAT_P, format!("layer {l} past the description"))
                    })?;
                    join_router(stream, w, spec, l)?;
                }
                Ok(())
            },
            |gpu, file, w, set, glue| {
                let open = PlacedOpen {
                    plan,
                    file: Arc::clone(file),
                    host,
                    set,
                    glue,
                    arch: "qwen35moe",
                    family: &super::tier_leg::QWEN35_FAMILY,
                    rows: PLACED_ROWS,
                };
                let mut body =
                    Body35::build(gpu, w, pre, (o, asked, PLAIN_LANES), Some((open, counted)))?;
                // Last: the machine empties each layer's spare slots, and
                // every piece above sized itself from the load's map.
                if let Some(placed) = body.placed.as_mut() {
                    placed.start_residency(gpu)?;
                }
                Ok(body)
            },
        )
    }
}

impl Body35 {
    /// The whole-card load: each routed layer's router joined with its
    /// shared gate, and its stacks with its shared expert where the plan
    /// folds it in ([`join_ffn`]), then the body ([`Body35::build`]).
    fn load(
        gpu: &Gpu,
        w: &mut Weights,
        pre: Pre35,
        o: Open35,
        lanes: usize,
    ) -> Result<Body35, GpuError> {
        for (l, plan) in pre.plans.iter().enumerate() {
            join_ffn(gpu.stream(), w, &plan.ffn, l)?;
        }
        Body35::build(gpu, w, pre, (o, o.ubatch, lanes), None)
    }

    /// The body over the resident weights `w` of `pre`'s description: the
    /// stores, the arenas, the kernels and the prompt image; on a placed
    /// load (`placed`: its placed side's inputs and the arena bytes its plan
    /// counts) the ubatch arena held to what the plan counts, not to the
    /// card's free bytes, and the placed side ([`Placed::new`]). The
    /// checkpoints' spacing is the ubatch size the load was asked for
    /// (`asked`: [`checkpoint_every`]), whether or not a placed load walks
    /// its prompt at it. `lanes` is the delta stores' lane count: one, or
    /// [`Body35::VERIFY_LANES`] on a load that verifies drafted rows (whole
    /// card only; a placed load runs no verify).
    fn build(
        gpu: &Gpu,
        w: &Weights,
        pre: Pre35,
        (o, asked, lanes): (Open35, usize, usize),
        placed: Option<(PlacedOpen<'_>, u64)>,
    ) -> Result<Body35, GpuError> {
        if lanes != PLAIN_LANES && (lanes != Body35::VERIFY_LANES || placed.is_some()) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{lanes} lanes a delta store: a load keeps 1, or {} on a whole-card load that \
                     verifies drafted rows",
                    Body35::VERIFY_LANES
                ),
            ));
        }
        let Pre35 {
            vocab,
            eps,
            layers: _,
            kinds,
            plans,
            d,
            ubatch,
            base,
            rope_dims,
            sections,
            common,
            formats: _,
        } = pre;
        let ctx = o.ctx;
        let stream = gpu.stream();
        let mut stores = Vec::with_capacity(plans.len());
        for (kind, plan) in kinds.iter().zip(&plans) {
            check_resident(w, &d, plan, placed.is_none())?;
            stores.push(match *kind {
                Kind35::Gqa(..) => LayerStore::Kv(KvPlanes::new(stream, &d, o.kv)?),
                Kind35::Delta(shape) if lanes > PLAIN_LANES => {
                    LayerStore::Rec(RecStore::stamped(stream, shape.lens(lanes), lanes)?)
                }
                Kind35::Delta(shape) => {
                    LayerStore::Rec(RecStore::new(stream, shape.lens(lanes), lanes)?)
                }
                Kind35::ShortConv(_) => {
                    return Err(GpuError::shape(
                        WHAT,
                        "a short-conv layer, which this body runs no kernel for",
                    ));
                }
            });
        }
        let forms = Forms::of(&plans, &d);
        let table = RopeTable::new(&RopeSpec::window(base, rope_dims))?;
        let rope = RopeRows::new(stream, &table, rope_dims, ctx)?;
        let (seq_rope, rope_base) = match sections {
            Some((sections, order)) => {
                let base: Arc<[f32]> = RopeRows::host_rows(&table, rope_dims, ctx)?.into();
                let base = RopeBase::new(base, sections, order);
                (Some(base.seq(stream, ctx)?), Some(base))
            }
            None => (None, None),
        };
        let every = checkpoint_every(asked, &d)?;
        let ckpt = sequence_checkpoints(gpu.context(), checkpoint_lens(&kinds), ctx, every)?;
        let (lane, views) = if lanes > PLAIN_LANES {
            let (word, views) = LaneWord::with_views(stream, lanes, 3)?;
            (Some(word), views)
        } else {
            (None, Vec::new())
        };
        let mut s = Arena::with(stream, d, 1, forms)?;
        let sp = StepParams::new(stream, true)?;
        let mut a = Arena::with(stream, d, MAX_PASS_ROWS, forms)?;
        let rp = PassRecord::new(stream)?;
        let slot_in = SlotIn::new(stream)?;
        let mut slot_lane = DeviceBuffer::<u32>::zeroed(stream, 1)?;
        slot_lane.copy_from_host(stream, &[LANE])?;
        let k = Kernels::load(gpu, true)?;
        let head_state = HeadArgmaxState::new(stream)?;
        let img = PromptImage::new(stream, ctx, true)?;
        let (mut u, placed) = match placed {
            None => {
                let free = gpu.alloc_room()?;
                (ubatch_arena(stream, (d, forms), ubatch.get(), free)?, None)
            }
            Some((open, counted)) => {
                refuse_stage_rows_past(WHAT, open.plan, 0, rope.table.num_bytes() as u64)?;
                let side = placed_bytes(&d, plans.len(), open.rows, common)?;
                let u = counted_arena(stream, (d, forms), ubatch.get(), counted, side)?;
                let built = Placed::new(gpu, w, &plans, &d, open)?;
                built.within(side)?;
                (u, Some(built))
            }
        };
        u.media = o
            .media
            .then(|| MediaRows::new(stream, d.hidden))
            .transpose()?;
        if let Ok([step, prompt, pass]) = <[_; 3]>::try_from(views) {
            // The step and the prompt units write the state back in place;
            // the pass is the verify, a lane a row.
            s.run_lanes(step, false)?;
            u.run_lanes(prompt, false)?;
            a.run_lanes(pass, true)?;
        }
        Ok(Body35 {
            vocab,
            eps,
            plans,
            stores,
            rope,
            seq_rope,
            rope_base,
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
            draft: None,
            placed,
            mtp: None,
            lanes,
            lane,
            pending: None,
            held: 0,
            ckpt,
            every,
            marks: false,
            prompt_walks: 0,
            stream: PromptStream::default(),
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
                MixerPlan::ShortConv(_) => {
                    panic!("qwen35moe::Body35::kinds: a short-conv layer, which `kinds35` refuses")
                }
            })
            .collect()
    }

    /// Each layer's shared expert as the plan runs it: `true` when kept
    /// apart from the routed stacks at its own types, `false` when folded
    /// into them (or a layer with none).
    pub fn shared_apart(&self) -> Vec<bool> {
        self.plans.iter().map(|p| p.ffn.apart().is_some()).collect()
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

    /// The placed load's wide card route as its last prompt unit left it
    /// ([`Placed::wide_route`]): the ubatch arena's routed ids beside every
    /// layer's map and table and the last walked layer's places, a blocking
    /// read on `stream` for gates and diagnostics. `None` on a load that is
    /// not a placed one.
    pub fn wide_route(&self, stream: &CudaStream) -> Result<Option<WideRoute>, GpuError> {
        let Some(placed) = &self.placed else {
            return Ok(None);
        };
        Ok(Some(placed.wide_route(stream, self.u.route.ids())?))
    }

    /// Device bytes of the layer stores: the attention layers' K/V planes
    /// and the delta layers' states and conv rings.
    #[must_use]
    pub fn store_bytes(&self) -> usize {
        self.stores.iter().map(LayerStore::bytes).sum()
    }

    /// Device bytes of the live sequence's rotation table; 0 on a text load.
    #[must_use]
    pub fn rope_bytes(&self) -> usize {
        self.seq_rope.as_ref().map_or(0, |r| r.table().num_bytes())
    }

    /// Lanes of a delta store on a load that verifies drafted rows
    /// ([`GpuModel::open_verifying`], [`GpuModel::open_mtp`]): a lane for
    /// each of a pass's [`MAX_PASS_ROWS`] rows, which is the count the plan
    /// prices (`KvLayout::of_spec_lanes`).
    pub const VERIFY_LANES: usize = MAX_PASS_ROWS;

    /// Lanes of every delta store: 1, or [`Body35::VERIFY_LANES`] on a load
    /// that verifies drafted rows.
    #[must_use]
    pub fn lanes(&self) -> usize {
        self.lanes
    }

    /// The lane that holds the committed state on a load of several lanes;
    /// `None` on a plain load (its one lane is lane 0).
    #[must_use]
    pub fn committed_lane(&self) -> Option<u32> {
        self.lane.as_ref().map(LaneWord::lane)
    }

    /// What a cut to at most `n` positions of a model standing at `pos`
    /// keeps, and why: with a verify waiting, `n` when it lies past the
    /// verify's first position (`Rule`: its commit moves the lane word and
    /// copies nothing) and every position when `n` reaches them (`Current`);
    /// else every position, the empty model at 0, or the nearest checkpoint
    /// at or below `n`, nothing (`Missed`) when none lies there — a delta
    /// layer keeps no earlier state but a copy. After a call failed past its
    /// launch the stores hold more than `pos`, which is no longer kept. On a
    /// media load a point inside an image grants nothing: the cut is asked
    /// again at the image's start. Never past `pos`. The rule
    /// [`Body35::commit`] takes on a load that verifies drafted rows,
    /// [`Body35::cut`] otherwise.
    #[must_use]
    pub fn kept(&self, n: u32, pos: u32) -> Kept {
        let verify = self.pending.map(|p| p.pos0);
        kept_rule(self.seq_rope.as_ref(), verify, self.held, n, |n| {
            if self.held == pos {
                self.ckpt.kept(n, pos)
            } else {
                self.ckpt.kept(n.min(pos), self.held)
            }
        })
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

    /// The least count an `admit` pick admits: the family's no-probe floor
    /// (`wide38`'s `STREAM_FLOOR`); a prompt call of fewer ids opens no call.
    pub const STREAM_FLOOR: u32 = STREAM_FLOOR;

    /// What the next prompt calls move (`BLOOMERY_XSTREAM`,
    /// [`PromptStream`]): off until set here, so one load can run every
    /// arm. Under `Admit` a prompt call of the wide units' path moves each
    /// card layer's residency pool toward the experts its units route most
    /// ([`super::placed`]'s streaming walk); `Split` starts the expert stream
    /// on `gpu` at its first setting ([`Placed::start_xstream`]: the ring
    /// sized from the card's free bytes, the lane's probe, the costs this
    /// device and host price for the rule) and arms the walk's pass rows
    /// for its ring, whose pass runs the streamed experts on the card beside
    /// its stacks; the stream then stays for the load's life, idle under
    /// `Admit` and `Off`. Refused by name: `Admit` or `Split` on a load that
    /// runs no residency machine, `Split` on a load the ring cannot serve
    /// (no wide unit) or when the stream cannot start (the card's room among
    /// the refusals), and any change while a call streams.
    pub fn set_xstream(&mut self, gpu: &Gpu, mode: XMode) -> Result<(), GpuError> {
        if self.stream.streaming() {
            return Err(GpuError::state(
                "Body35::set_xstream",
                "no prompt call streaming",
            ));
        }
        let placed = self.placed.as_mut().filter(|p| p.hybrid.swap().is_some());
        let Some(p) = placed else {
            if mode == XMode::Off {
                self.stream.set_mode(mode);
                return Ok(());
            }
            return Err(GpuError::state(
                "BLOOMERY_XSTREAM",
                "a residency machine (BLOOMERY_RESIDENCY=mid-p<P>-s<S>) for admit or split",
            ));
        };
        if mode == XMode::Split {
            p.start_xstream(gpu)?;
        }
        self.stream.set_mode(mode);
        Ok(())
    }

    /// The load line's `xstream=` word ([`PromptStream::word`]); `off` on a
    /// load with no host tier.
    #[must_use]
    pub fn xstream_word(&self) -> String {
        match self.placed.as_ref() {
            Some(p) => self.stream.word(&p.hybrid),
            None => self.stream.mode().word().to_string(),
        }
    }

    /// The last streaming prompt call's picks, each with its walk, and its
    /// end; empty and `None` after a call that did not stream. Taken: a
    /// second read is empty ([`PromptStream::take_pick_records`]).
    pub fn take_stream_records(&mut self) -> (Vec<(usize, CallPick)>, Option<CallReport>) {
        self.stream.take_pick_records()
    }

    /// The expert stream's rule costs in place of the priced ones
    /// ([`PromptStream::set_costs`]). Refused by name on a load with no host
    /// tier, before `split` started the stream and while a call streams.
    pub fn set_xstream_costs(&mut self, costs: Costs) -> Result<(), GpuError> {
        let p = self.placed.as_mut().ok_or(GpuError::state(
            "Body35::set_xstream_costs",
            "a placed load's host tier",
        ))?;
        self.stream.set_costs(&mut p.hybrid, costs)
    }

    /// The last streaming prompt call's streamed layers, each with its walk,
    /// and the stream's report; empty and `None` after a call that did not
    /// stream under `split`. Taken: a second read is empty
    /// ([`PromptStream::take_xstream_records`]).
    pub fn take_xstream_records(&mut self) -> (Vec<(usize, XLayer)>, Option<XReport>) {
        self.stream.take_xstream_records()
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

    /// A call at `pos` is refused by name while a verify waits for its
    /// commit: its rows' states sit in lanes the committed lane word does
    /// not name yet.
    fn refuse_pending(&self, what: &'static str, pos: u32) -> Result<(), GpuError> {
        match &self.pending {
            None => Ok(()),
            Some(p) => Err(GpuError::shape(
                what,
                format!(
                    "a call at position {pos} while the verify of {} rows at {} waits for its \
                     commit (GpuModel::rollback keeps 1..={})",
                    p.rows, p.pos0, p.rows
                ),
            )),
        }
    }

    /// The committed lane: the lane word's on a load of several lanes, lane
    /// [`LANE`] otherwise.
    fn committed(&self) -> u32 {
        self.lane.as_ref().map_or(LANE, LaneWord::lane)
    }

    /// The waiting cut carried out ([`Checkpoints::apply`]): every delta
    /// layer's committed state lane and conv ring from the point's slot, or
    /// each zeroed — and, on a load of several lanes, the committed lane
    /// stamped at the position the cut left (`held`) and every other lane
    /// never written. Nothing when no cut waits. Waits for the copies; never
    /// inside a capture (a cut has no stream of its own).
    fn settle(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        if !self.ckpt.pending() {
            return Ok(());
        }
        let lane = self.committed();
        {
            let Body35 { ckpt, stores, .. } = &mut *self;
            let mut c = Copied35::of(stores, lane)?;
            ckpt.apply(stream, &mut c.list())?;
        }
        self.restamp(stream, lane)
    }

    /// Every stamped delta store's stamps: lane `lane` at the held position,
    /// every other never written. Nothing on a plain load. Never inside a
    /// capture.
    fn restamp(&mut self, stream: &CudaStream, lane: u32) -> Result<(), GpuError> {
        let at = self.held;
        for s in &mut self.stores {
            if let LayerStore::Rec(r) = s
                && r.stamp.is_some()
            {
                r.restamp_at(stream, lane, at)?;
            }
        }
        Ok(())
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
        self.refuse_pending(WHAT_M, pos)?;
        self.settle(stream)?;
        self.stores_at(pos)?;
        self.held = pos
            + u32::try_from(m).map_err(|_| {
                GpuError::shape(WHAT_M, format!("a call of {m} positions past u32"))
            })?;
        Ok(())
    }

    /// A checkpoint at `pos`, where the stores stand, after any waiting cut:
    /// the delta layers' committed lane and conv ring copied to a host slot,
    /// or nothing where one stands or at 0. Refused by name when the stores
    /// hold another position or a verify waits for its commit. Waits for the
    /// copies.
    fn take_mark(&mut self, stream: &CudaStream, pos: u32) -> Result<(), GpuError> {
        self.stores_at(pos)?;
        self.refuse_pending("qwen35moe::Body35::take_mark", pos)?;
        self.settle(stream)?;
        let lane = self.committed();
        let Body35 { ckpt, stores, .. } = self;
        let mut c = Copied35::of(stores, lane)?;
        ckpt.take(stream, pos, &mut c.list())?;
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
        self.pending = None;
        Ok(())
    }

    /// Keep the first `pos − pos0` rows of the verify waiting for its commit
    /// and take the rest back, on a load of several lanes: the lane word to
    /// the last kept row's lane ([`LaneWord::keep`]), nothing copied, the
    /// stores held at `pos`. A position at or below the verify's first (or
    /// with no verify waiting, behind the stores) is a cut to the checkpoint
    /// there ([`Body35::cut`]); `pos` at the stores' position with no verify
    /// waiting takes nothing. A `pos` past the verify's rows is refused by
    /// name, the verify left waiting. What is taken is [`Body35::kept`]'s
    /// rule.
    fn commit(&mut self, stream: &CudaStream, pos: u32) -> Result<(), GpuError> {
        const WHAT_K: &str = "qwen35moe::Body35::commit";
        let Some(p) = self.pending.filter(|p| pos > p.pos0) else {
            if pos == self.held && self.pending.is_none() {
                return Ok(());
            }
            return self.cut(pos);
        };
        if pos as usize > p.pos0 as usize + p.rows {
            return Err(GpuError::shape(
                WHAT_K,
                format!(
                    "back to position {pos}: the verify of {} rows at {} keeps 1..={} (row 0 \
                     always)",
                    p.rows, p.pos0, p.rows
                ),
            ));
        }
        let kept = (pos - p.pos0) as usize;
        self.lane
            .as_mut()
            .ok_or(GpuError::state(
                WHAT_K,
                "a lane word (a load of several lanes)",
            ))?
            .keep(stream, kept)?;
        self.pending = None;
        self.held = pos;
        Ok(())
    }

    /// Exchange the live sequence with `seq`: its stores, the positions they
    /// hold, its checkpoints and its rotation table, pointer moves only
    /// ([`Slots::swap_seq`], and a pass of several slots' per-slot work,
    /// `body35_slots`).
    pub(super) fn exchange(&mut self, seq: &mut Slot35) {
        std::mem::swap(&mut self.stores, &mut seq.stores);
        std::mem::swap(&mut self.held, &mut seq.held);
        std::mem::swap(&mut self.ckpt, &mut seq.ckpt);
        std::mem::swap(&mut self.seq_rope, &mut seq.rope);
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
        seq_rope,
        s,
        sp,
        k,
        head_state,
        mma,
        taps,
        draft,
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
        table: rope_table(rope, seq_rope.as_ref()),
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
    .walk_with(draft.as_mut())
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
    if b.rp.planned() != Some(m) && !crate::capturing(gpu.stream())? {
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
        seq_rope,
        a,
        rp,
        k,
        head_state,
        mma,
        draft,
        ..
    } = b;
    let c = PassCtx {
        gpu,
        w,
        plans,
        k,
        mma: *mma,
        eps: *eps,
        table: rope_table(rope, seq_rope.as_ref()),
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
    .walk_with(draft.as_mut())
}

/// What a prompt call runs after each unit: the unit's first position and
/// its rows.
pub(super) type UnitHook<'h> =
    dyn FnMut(&mut GpuModel<Body35>, u32, usize) -> Result<(), GpuError> + 'h;

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
        if let Some(rows) = self.u.media.as_mut() {
            rows.plan(pos as usize..pos as usize + m);
        }
        if let Some(d) = self.draft.as_mut() {
            d.plan(m);
        }
        self.prompt_walks += 1;
        let Body35 {
            eps,
            plans,
            stores,
            rope,
            seq_rope,
            u,
            k,
            head_state,
            mma,
            draft,
            placed,
            stream,
            ..
        } = self;
        let c = PassCtx {
            gpu,
            w,
            plans,
            k,
            mma: *mma,
            eps: *eps,
            table: rope_table(rope, seq_rope.as_ref()),
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
            side.refresh_maps(gpu.stream(), m)?;
            let mut leg = BatchLeg::new(gpu.stream(), hybrid, pass_hsum, side.cap);
            let walk = BatchWalk {
                p: WalkParts {
                    c: &c,
                    stores: stores.as_mut_slice(),
                    s: u,
                    io: &win.io(),
                    m,
                    side,
                    rows: pass,
                },
            };
            if stream.streaming() {
                return walk.walk_streamed(&mut leg, tail, stream, pos as usize + m);
            }
            return walk.walk(&mut leg, tail);
        }
        Program {
            c: &c,
            stores: stores.as_mut_slice(),
            s: u,
            io: &win.io(),
            m,
            tail,
        }
        .walk_with(draft.as_mut())
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
                     units of up to {PLACED_ROWS}"
                ),
            ));
        }
        let d = body.u.dims;
        let size = ubatch_of(&d, u)?;
        let forms = Forms::of(&body.plans, &d);
        let free = gpu.alloc_room()?;
        let mut arena = ubatch_arena(gpu.stream(), (d, forms), u, free)?;
        arena.media = body.u.media.take();
        body.u = arena;
        body.ubatch = size;
        Ok(())
    }

    /// The units [`GpuModel::prefill_with`] runs a prompt of `tokens` ids as
    /// by `path`, at this model's ubatch size.
    pub fn prefill_plan(&self, tokens: usize, path: PrefillPath) -> Result<PrefillPlan, GpuError> {
        let body = self.body("qwen35moe::prefill_plan")?;
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
        let next = self.prefill_call(tokens, path, CallEnd::Argmax)?;
        next.ok_or(GpuError::state(
            "qwen35moe::prefill",
            "a token read after the last unit",
        ))
    }

    /// [`GpuModel::prefill_with`] ending as `end` asks ([`CallEnd`]): the
    /// argmax after the last id, or for [`CallEnd::Unread`] no head at all,
    /// the fault word read at each run's end in its place.
    pub fn prefill_call(
        &mut self,
        tokens: &[u32],
        path: PrefillPath,
        end: CallEnd,
    ) -> Result<Option<u32>, GpuError> {
        self.prefill_hooked(tokens, path, None, end)
    }

    /// [`GpuModel::prefill_call`], and with `unit` called after each unit
    /// with the unit's first position and rows (the unit's walk enqueued, its
    /// head read when it is the call's last): where
    /// [`GpuModel::prefill_tapped`] reads its taps back. A closing call
    /// ([`CallEnd::Close`]) is refused by name before anything runs: a delta
    /// layer keeps no state inside a unit, where the end mark before the
    /// last id would stand.
    pub(super) fn prefill_hooked(
        &mut self,
        tokens: &[u32],
        path: PrefillPath,
        unit: Option<&mut UnitHook<'_>>,
        end: CallEnd,
    ) -> Result<Option<u32>, GpuError> {
        const WHAT_P: &str = "qwen35moe::prefill";
        if tokens.is_empty() {
            return Err(GpuError::shape(WHAT_P, "empty token slice"));
        }
        if end == CallEnd::Close {
            return Err(GpuError::shape(
                WHAT_P,
                "a closing call: a delta layer takes no checkpoint inside a unit, before the \
                 call's last id",
            ));
        }
        let pos0 = self.pos();
        self.body(WHAT_P)?.refuse_inside_image(WHAT_P, pos0)?;
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
        // A call before a decode history ends with each layer back at the
        // set it started with; without one (a fresh sequence) the call's
        // placement stays. Read once, here: the call keeps no row.
        let history = self
            .body(WHAT_P)?
            .placed
            .as_ref()
            .is_some_and(|p| p.hybrid.call_history());
        let r = self
            .stream_begin(path, tokens.len(), history)
            .and_then(|()| self.feed_marks(tokens, path, &marks, marked, unit, end.head()));
        // The call's own error first: an end refused after a failed call is
        // its echo. A failed call's streaming ends with each layer back at
        // the set it started with too.
        let ended = self.stream_end(r.is_ok() && !history);
        let next = r?;
        ended?;
        Ok(next)
    }

    /// [`GpuModel::prefill_call`]'s runs between its `marks` from where the
    /// model stands, a checkpoint taken at each when `marked`: with `head`
    /// the argmax after the last id, the head over the call's last row alone
    /// (a run before a mark reads only the fault word at its end).
    fn feed_marks(
        &mut self,
        tokens: &[u32],
        path: PrefillPath,
        marks: &[u32],
        marked: bool,
        mut unit: Option<&mut UnitHook<'_>>,
        head: bool,
    ) -> Result<Option<u32>, GpuError> {
        const WHAT_M: &str = "qwen35moe::prefill";
        let pos0 = self.pos();
        let end = pos0 + launch_u32(WHAT_M, "tokens", tokens.len())?;
        let (mut next, mut at) = (None, pos0);
        for &mark in marks {
            if mark > at {
                let run = &tokens[(at - pos0) as usize..(mark - pos0) as usize];
                let headed = head && mark == end;
                next = self.feed35(run, path, unit.as_deref_mut(), headed)?;
                at = mark;
            }
            if marked {
                let pos = self.pos();
                let (gpu, _, body) = self.body_parts(WHAT_M)?;
                body.take_mark(gpu.stream(), pos)?;
            }
        }
        if head && next.is_none() {
            return Err(GpuError::state(WHAT_M, "a token read after the last unit"));
        }
        Ok(next)
    }

    /// Open the prompt call's streaming ([`PromptStream::begin`]) with the
    /// body's own facts: whether the path runs wide units (a pass-fed call
    /// opens none: its units are the gemv arm's), the call's `n` ids and its
    /// end, whether it `returns` to its start, the floor
    /// ([`Body35::STREAM_FLOOR`]), no card tail rate and the map's layers.
    /// Nothing on a load with no host tier.
    fn stream_begin(&mut self, path: PrefillPath, n: usize, returns: bool) -> Result<(), GpuError> {
        const WHAT_B: &str = "qwen35moe prompt streaming";
        let end = self.pos() as usize + n;
        let (gpu, _, body) = self.body_parts(WHAT_B)?;
        let layers = body.plans.len();
        let Body35 { placed, stream, .. } = body;
        let Some(p) = placed.as_mut() else {
            return Ok(());
        };
        // No record prices this family's card tail: a walk balances without it.
        let tail = 0.0;
        stream.begin(
            &mut p.hybrid,
            gpu.stream(),
            path != PrefillPath::Pass,
            n,
            end,
            returns,
            STREAM_FLOOR,
            tail,
            layers,
        )
    }

    /// End the prompt call's streaming ([`PromptStream::end`]): the end's
    /// report kept for [`Body35::take_stream_records`]. The call passes
    /// `kept` false before a decode history: the decode's experts go back on
    /// the card.
    fn stream_end(&mut self, kept: bool) -> Result<(), GpuError> {
        let (gpu, _, body) = self.body_parts("qwen35moe prompt streaming")?;
        let Body35 { placed, stream, .. } = body;
        match placed.as_mut() {
            Some(p) => stream.end(&mut p.hybrid, gpu.stream(), kept),
            None => Ok(()),
        }
    }

    /// One run of [`GpuModel::prefill_call`]'s call, from where the model
    /// stands: the image written for the run's ids, then each unit of the
    /// run's plan one walk, the last ending in its last row's head when
    /// `head` asks for it (the argmax after the last id), else in a read of
    /// the fault word, so a fault ends the call before a checkpoint copies a
    /// store.
    fn feed35(
        &mut self,
        run: &[u32],
        path: PrefillPath,
        mut hook: Option<&mut UnitHook<'_>>,
        head: bool,
    ) -> Result<Option<u32>, GpuError> {
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
            let first = self.pos();
            let headed = last && head;
            next = self.run_rows(t, WHAT_F, |gpu, w, body, out, pos| {
                let end = if headed {
                    UnitEnd::Head(out)
                } else {
                    UnitEnd::Pass
                };
                body.walk_unit(gpu, w, unit, pos, end)?;
                body.stream.walk_done();
                if last
                    && !headed
                    && let Some(f) = gpu.fault()?
                {
                    return Err(GpuError::fault(WHAT_F, f));
                }
                Ok(headed)
            })?;
            if let Some(h) = hook.as_deref_mut() {
                h(self, first, t)?;
            }
        }
        if head && next.is_none() {
            return Err(GpuError::state(WHAT_F, "a token read after the last unit"));
        }
        Ok(next)
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
        self.body(WHAT_H)?.refuse_inside_image(WHAT_H, pos0)?;
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
        let mut out = host_filled::<f32>(tokens.len() * hidden);
        let mut at = 0usize;
        for &step in &plan.steps {
            let t = match step {
                PrefillStep::Ubatch(t) | PrefillStep::Pass(t) => t,
            };
            let unit = at..at + t;
            let out = out.get_mut(at * hidden..(at + t) * hidden).ok_or_else(|| {
                GpuError::shape(
                    WHAT_H,
                    format!("a unit of {t} positions at {at} of {}", tokens.len()),
                )
            })?;
            at += t;
            self.run_rows(t, WHAT_H, |gpu, w, body, _, pos| {
                body.walk_unit(gpu, w, unit, pos, UnitEnd::Hidden)?;
                if let Some(f) = gpu.fault()? {
                    return Err(GpuError::fault(WHAT_H, f));
                }
                // SAFETY: rows `0..t` of `normed` span `t · hidden` values
                // inside it (`rows · hidden`, `t <= rows` by the walk's
                // refusal), and the arena stays in place for the copy.
                let rows = unsafe { super::scratch::cells_view(&body.u.normed, 0, t * hidden) };
                rows.copy_to_host(gpu.stream(), out)?;
                Ok(false)
            })?;
        }
        Ok(out)
    }
}

impl Body35 {
    /// Whether the load takes media ([`Open35::media`]): every sequence holds
    /// a rotation table of its own.
    #[must_use]
    pub fn takes_media(&self) -> bool {
        self.seq_rope.is_some()
    }
}

impl GpuModel<Body35> {
    /// The rotation table the live sequence's rope launches read, `ctx` rows
    /// of 64 values read back: a media load's own, else the load's. An
    /// instrument; synchronizes.
    pub fn rope_rows(&self) -> Result<Vec<f32>, GpuError> {
        let body = self.body("qwen35moe::rope_rows")?;
        Ok(rope_table(&body.rope, body.seq_rope.as_ref()).to_host_vec(self.gpu().stream())?)
    }
}

impl XStreamBody for Body35 {
    /// [`Body35::set_xstream`].
    fn set_xstream(&mut self, gpu: &Gpu, mode: XMode) -> Result<(), GpuError> {
        Body35::set_xstream(self, gpu, mode)
    }

    /// [`Body35::xstream_word`].
    fn xstream_word(&self) -> String {
        Body35::xstream_word(self)
    }
}

/// The check of a media call's plan: `tokens` ids run as wide units
/// ([`GpuModel::prefill_plan`]).
fn plan_wide(tokens: usize) -> impl FnOnce(&GpuModel<Body35>) -> Result<(), GpuError> {
    move |m| m.prefill_plan(tokens, PrefillPath::Wide).map(drop)
}

impl MediaCarrier for Body35 {
    const VIA: &'static str = "Open35::media";

    fn width(&self) -> usize {
        self.hidden()
    }

    fn seq_rope(&self) -> Option<&SeqRope> {
        self.seq_rope.as_ref()
    }

    fn media_parts(&mut self) -> Option<(&mut SeqRope, &mut MediaRows)> {
        self.seq_rope.as_mut().zip(self.u.media.as_mut())
    }
}

impl MediaBody for Body35 {
    type Span<'a> = ImageSpan<'a>;

    fn media_span<'s>(s: &'s ImageSpan<'_>) -> &'s MediaSpan<'s> {
        &s.span
    }

    /// Image spans on a text load.
    fn media_refusal(m: &GpuModel<Body35>, spans: usize) -> Option<String> {
        let takes = m.body(WHAT).is_ok_and(Body35::takes_media);
        (spans > 0 && !takes).then(|| {
            format!(
                "a prompt with {spans} image spans on a load opened without media \
                 (Open35::media)"
            )
        })
    }

    /// [`GpuModel::prefill_with`] on [`PrefillPath::Wide`] with the spans' rows in
    /// place of their positions' embeddings and the sequence's rotation table
    /// shifted by them. The argmax after the last id.
    fn prefill_media(
        m: &mut GpuModel<Body35>,
        ids: &[u32],
        spans: &[ImageSpan<'_>],
    ) -> Result<u32, GpuError> {
        media_call(
            m,
            "qwen35moe::prefill_media",
            (ids, spans),
            plan_wide(ids.len()),
            |m| m.prefill_with(ids, PrefillPath::Wide),
        )
    }

    /// [`GpuModel::prefill_hidden`] on [`PrefillPath::Wide`], as
    /// [`MediaBody::prefill_media`] feeds: every position's final-norm hidden
    /// row.
    fn prefill_hidden_media(
        m: &mut GpuModel<Body35>,
        ids: &[u32],
        spans: &[ImageSpan<'_>],
    ) -> Result<Vec<f32>, GpuError> {
        media_call(
            m,
            "qwen35moe::prefill_hidden_media",
            (ids, spans),
            plan_wide(ids.len()),
            |m| m.prefill_hidden(ids, PrefillPath::Wide),
        )
    }
}

impl ChainBody for Body35 {
    type Input = DecodeInput35;
    type Host = Placed;
    const UNHOSTED_POOL: bloomery_levers::UnhostedPool = bloomery_levers::UnhostedPool::NotCarried;

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
        self.refuse_inside_image("qwen35moe::step", pos)?;
        self.refuse_pending("qwen35moe::step", pos)?;
        self.settle(stream)?;
        self.stores_at(pos)?;
        self.held = pos + 1;
        if let Some(d) = self.draft.as_mut() {
            d.plan(1);
        }
        self.sp.write(stream, token, pos)
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        enqueue_chain(gpu, w, self, head)
    }

    /// Every delta layer's state lanes and conv ring back to zero, the
    /// checkpoints dropped, the rotation table of a media load back to its
    /// text rows and the stores counted at 0 — the live
    /// sequence's, the selected slot's; a parked one keeps its own — and
    /// the records' lane word to [`LANE`] (on a load of several lanes the
    /// committed lane to lane 0, the stamps fresh and no verify waiting),
    /// after a placed load's host tier
    /// reset — the load's, whichever slot resets: a refused input's poison
    /// lifted, the words checked at rest. The K/V planes need nothing: the
    /// flash never loads a key row at or past the live count, and every row
    /// below it is written by its own step first. Synchronizes.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let stream = gpu.stream();
        if let Some(p) = self.placed.as_mut() {
            p.reset(stream)?;
        }
        for s in &mut self.stores {
            if let LayerStore::Rec(r) = s {
                r.clear(stream)?;
            }
        }
        self.ckpt.clear();
        self.held = 0;
        self.pending = None;
        if let Some(word) = self.lane.as_mut() {
            word.set(stream, LANE)?;
        }
        for s in &mut self.stores {
            if let LayerStore::Rec(r) = s
                && r.stamp.is_some()
            {
                r.restamp(stream)?;
            }
        }
        if let Some(rope) = self.seq_rope.as_mut() {
            rope.restore(stream, MropeSeq::new())?;
        }
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
            + self.rope_bytes()
            + self.s.bytes()
            + self.sp.bytes()
            + self.a.bytes()
            + self.rp.bytes()
            + self.slot_in.bytes()
            + self.slot_lane.num_bytes()
            + self.lane.as_ref().map_or(0, LaneWord::bytes)
            + self.u.bytes()
            + self.img.bytes()
            + self.head_state.bytes()
            + self.taps.as_ref().map_or(0, |t| t.buf.num_bytes())
            + self.draft.as_ref().map_or(0, DraftTaps::bytes)
            + self
                .mtp
                .as_ref()
                .map_or(0, |d| d.resident_bytes() + d.scratch_bytes())
            + self.placed.as_ref().map_or(0, Placed::device_bytes)
            + self.k.q35.as_ref().map_or(0, |k| k.kgemv.device_bytes())
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
    /// made, and on a media load a rotation table of text rows. A placed load's sequences share its placed side whole: the card
    /// experts and the slot map are weights, the host tier's experts
    /// compute each token apart, its handoff words and health are the
    /// load's protocol (a go advances them whichever sequence runs), and its
    /// rows and host sums are a call's scratch. Load-time allocation.
    fn new_seq(&mut self, gpu: &Gpu) -> Result<Slot35, GpuError> {
        const WHAT_N: &str = "qwen35moe::Body35::new_seq";
        // The draft K/V is one cache on the body, so a second sequence
        // would attend to the first sequence's rows.
        if self.mtp.is_some() {
            return Err(GpuError::shape(
                WHAT_N,
                "an in-file MTP load supports one resident sequence",
            ));
        }
        if self.lanes > PLAIN_LANES {
            return Err(GpuError::shape(
                WHAT_N,
                format!(
                    "a second resident sequence on a load that verifies drafted rows ({} lanes a \
                     delta layer, one lane word): its slots keep one lane each",
                    self.lanes
                ),
            ));
        }
        let stream = gpu.stream();
        let d = self.s.dims;
        let stores =
            self.plans
                .iter()
                .zip(&self.stores)
                .map(|(p, live)| match (&p.mixer, live) {
                    (MixerPlan::Gqa(_), LayerStore::Kv(planes)) => {
                        Ok(LayerStore::Kv(KvPlanes::new(stream, &d, planes.format())?))
                    }
                    (MixerPlan::Delta(dp), LayerStore::Rec(_)) => Ok(LayerStore::Rec(
                        RecStore::new(stream, dp.shape.lens(PLAIN_LANES), PLAIN_LANES)?,
                    )),
                    _ => Err(GpuError::state(
                        WHAT_N,
                        "a live store of each layer's own kind",
                    )),
                })
                .collect::<Result<Vec<_>, GpuError>>()?;
        let ckpt =
            sequence_checkpoints(gpu.context(), self.ckpt.lens().to_vec(), d.ctx, self.every)?;
        let rope = self
            .rope_base
            .as_ref()
            .map(|b| b.seq(stream, d.ctx))
            .transpose()?;
        Ok(Slot35 {
            stores,
            held: 0,
            ckpt,
            rope,
        })
    }

    /// Exchange the live sequence with `seq`: pointer moves only — the
    /// stores and the rotation table are buffer handles, the held count and
    /// the checkpoints host values the body holds by value — so nothing is
    /// copied, captured or synchronized. Nothing refuses: a prompt call waits for its
    /// checkpoints' copies before it returns, and a cut waits in its own
    /// sequence's checkpoints for that sequence's next call
    /// ([`Body35::settle`]), so it travels with it.
    fn swap_seq(&mut self, _gpu: &Gpu, seq: &mut Slot35) -> Result<(), GpuError> {
        self.exchange(seq);
        Ok(())
    }

    /// Device bytes one sequence holds: its layers' stores
    /// ([`Body35::store_bytes`]) and its rotation table
    /// ([`Body35::rope_bytes`]). Its checkpoints are host bytes, left out:
    /// pinned slots its prompt calls make as they take points, up to
    /// [`checkpoint_budget35`] a sequence and none at [`Slots::new_seq`].
    fn seq_bytes(&self) -> usize {
        self.store_bytes() + self.rope_bytes()
    }
}

impl Body35 {
    /// The file's next-token layer, resident when the load was opened with
    /// [`GpuModel::open_mtp`]; `None` on any other load.
    #[must_use]
    pub fn mtp(&self) -> Option<&Mtp35> {
        self.mtp.as_ref()
    }
}

/// The next-token layer's call; refused by name on a load without it.
impl GpuModel<Body35> {
    /// One step of the file's next-token layer ([`Mtp35`]): token `t` at
    /// position `pos` over `h`, the target's post-`output_norm` row of the
    /// position before it (`hidden` f32 on the card), through the layer's own
    /// K/V rows `0..=pos`; the draft's argmax token, first index on a tie. The
    /// draft's K/V row `pos` is written, so rows `0..pos` hold the earlier
    /// steps' of the caller's choosing. A poisoned model is refused; a fault
    /// the step raises is its error and poisons the model as the target's
    /// would.
    pub fn mtp_draft(
        &mut self,
        t: u32,
        pos: usize,
        h: &DeviceBuffer<f32>,
    ) -> Result<u32, GpuError> {
        const WHAT_D: &str = "qwen35moe mtp_draft";
        self.refuse_if_poisoned(WHAT_D)?;
        let (gpu, w, body) = self.body_parts(WHAT_D)?;
        let r = body.mtp_run(gpu, w, (t, pos), h);
        self.note_fault(WHAT_D, r)
    }
}

impl Body35 {
    /// [`GpuModel::mtp_draft`]'s step over the body's kernels and rope table.
    fn mtp_run(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        (t, pos): (u32, usize),
        h: &DeviceBuffer<f32>,
    ) -> Result<u32, GpuError> {
        let Body35 {
            mtp, k, rope, mma, ..
        } = self;
        let mtp = mtp.as_mut().ok_or(GpuError::state(
            "qwen35moe mtp_draft",
            "the load carries no next-token layer (open it with GpuModel::open_mtp)",
        ))?;
        mtp.enqueue(
            gpu,
            w,
            (&*k, &rope.table, *mma),
            (t, pos, WalkMode::Eager),
            h,
        )?;
        mtp.token(gpu)
    }
}

impl Rollback for Body35 {
    /// A cut to `pos` ([`Body35::cut`]): the checkpoint there planned for
    /// the next call to carry out ([`Body35::settle`]), any other position
    /// behind the stores refused by name. A text load has no device work in
    /// it; a media load's table is cut on the card
    /// ([`Rollback::rollback_on`]), and a load of several lanes commits a
    /// verify on the card ([`Body35::commit`]), so both are refused here by
    /// name.
    fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        if self.seq_rope.is_some() {
            return Err(GpuError::shape(
                "qwen35moe::Body35::rollback",
                format!(
                    "back to position {pos} without the card: GpuModel::rollback cuts a media \
                     load's rotation table"
                ),
            ));
        }
        if self.lanes > PLAIN_LANES {
            return Err(GpuError::shape(
                "qwen35moe::Body35::rollback",
                format!(
                    "back to position {pos} without the card: a load of {} lanes commits a \
                     verify by a copy of its lane word (GpuModel::rollback)",
                    self.lanes
                ),
            ));
        }
        self.cut(pos)
    }

    /// [`Rollback::rollback`] with the card at hand: the cut of the stores
    /// ([`Body35::cut`]) — on a load of several lanes the commit of the
    /// verify waiting ([`Body35::commit`]), any `1..=m` rows of it kept —,
    /// and on a media load the rotation table's: the
    /// images from `pos` on go and the rows after `pos` take the positions
    /// the kept images leave ([`SeqRope::cut`]). A cut inside an image is
    /// refused by name before anything moves.
    fn rollback_on(&mut self, gpu: &Gpu, pos: u32) -> Result<(), GpuError> {
        const WHAT_R: &str = "qwen35moe::Body35::rollback";
        let row = pos as usize;
        if let Some(rope) = &self.seq_rope {
            rope.rows()
                .seq()
                .clone()
                .cut(row)
                .map_err(|e| GpuError::shape(WHAT_R, e.to_string()))?;
        }
        if self.lanes > PLAIN_LANES {
            self.commit(gpu.stream(), pos)?;
        } else {
            self.cut(pos)?;
        }
        if let Some(rope) = self.seq_rope.as_mut() {
            rope.cut(gpu.stream(), row)?;
        }
        Ok(())
    }
}

impl Rows for Body35 {
    const MAX_ROWS: usize = MAX_PASS_ROWS;
    /// No host tier serves a Qwen3.6 replay; the pass is the chain's
    /// one-token layout at `m` rows.
    const CHAIN: Chain = Chain::Step;

    fn plan_rows(&mut self, stream: &CudaStream, tokens: &[u32], pos: u32) -> Result<(), GpuError> {
        self.refuse_inside_image("qwen35moe::rows", pos)?;
        self.stand_held(stream, pos, tokens.len())?;
        self.rp.write(stream, tokens, pos)?;
        if let Some(d) = self.draft.as_mut() {
            d.plan(tokens.len());
        }
        if self.lanes > PLAIN_LANES {
            self.pending = Some(Pending35 {
                pos0: pos,
                rows: tokens.len(),
            });
        }
        Ok(())
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
        if self.lanes > PLAIN_LANES {
            let lane = self.committed();
            self.restamp(stream, lane)?;
            stream.synchronize()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Flash, GqaKind, Kind35, checkpoint_lens, placed_split, whole_origin};
    use crate::checkpoint::{point_bytes, price};
    use crate::folder::{Folder, Model};
    use crate::linear::{KHeadMap, LinearShape};
    use crate::model::CardBytes;
    use gguf::Split;
    use gguf::Value;
    use gguf::write::{Layout, TensorDecl};
    use model::part::{Part, Planned, Source, Take};

    /// A part source that holds nothing: a load that read it would be refused by name.
    struct NoParts;

    impl Source for NoParts {
        type Error = std::fmt::Error;

        fn plan<E: std::fmt::Display>(
            &self,
            _: impl Fn(&str, gguf::GgmlType, &[u64]) -> Result<Take, E>,
        ) -> Result<Vec<Planned>, Self::Error> {
            Err(std::fmt::Error)
        }

        fn read_bytes(&self, _: &Planned) -> Result<u64, Self::Error> {
            Err(std::fmt::Error)
        }

        fn read(&self, _: &Planned) -> Result<Part<'_>, Self::Error> {
            Err(std::fmt::Error)
        }
    }

    fn header() -> Split {
        let kvs = [("toy.layers".to_string(), Value::U32(1))];
        let tensors = vec![TensorDecl {
            name: "w.weight".into(),
            dims: vec![32],
            type_id: gguf::GgmlType::F32.as_u32(),
            nbytes: 128,
        }];
        let layout = Layout::new(&kvs, tensors).unwrap();
        Split::header_only(std::env::temp_dir().join("body35-folder-toy.gguf"), &layout).unwrap()
    }

    fn folder() -> Model {
        Folder::new(header(), NoParts).into()
    }

    /// A placed load reads a GGUF file's mapped tensors, so a checkpoint folder handed to it is
    /// refused by name, and a split passes.
    #[test]
    fn a_placed_load_refuses_a_checkpoint_folder() {
        let e = placed_split(folder(), "toy").err().unwrap().to_string();
        assert!(
            e.contains("a checkpoint folder loads whole on the card"),
            "{e}"
        );
        assert!(placed_split(Model::from(header()), "toy").is_ok());
    }

    /// A whole-card load reads a folder's card tensors through its part source and a split's own,
    /// and refuses a folder's next-token layer by name, whose tensors the folder's map leaves out;
    /// a split's next-token layer is the draft rule's to refuse or take.
    #[test]
    fn a_whole_load_reads_a_folder_through_its_parts_and_refuses_its_draft() {
        let (_, origin) = whole_origin(folder(), false).unwrap();
        assert!(matches!(origin.bytes(), CardBytes::Parts(_)));
        let e = whole_origin(folder(), true).err().unwrap().to_string();
        assert!(e.contains("a checkpoint folder's next-token layer"), "{e}");
        for mtp in [false, true] {
            let (_, origin) = whole_origin(Model::from(header()), mtp).unwrap();
            assert!(matches!(origin.bytes(), CardBytes::File));
        }
    }

    /// A checkpoint of Qwen3.6-35B-A3B's trunk — 30 delta layers of 16 key
    /// and 32 value heads among 10 attention layers — copies each delta
    /// layer's state (2,097,152 B) and conv ring (360,448 B): 73,728,000 B,
    /// spaced at the 4,096-token default ubatch, so a slot of 2,048 positions
    /// pins one point and one of 10,240 three. The lengths are the stores'
    /// own ([`LinearShape::lens`]), in [`Copied35`]'s order.
    #[test]
    fn a_checkpoint_copies_each_delta_layers_state_and_ring() {
        let delta = Kind35::Delta(LinearShape {
            n_k: 16,
            n_v: 32,
            map: KHeadMap::Tiled,
        });
        let kinds: Vec<Kind35> = (0..40)
            .map(|l| {
                if l % 4 == 3 {
                    Kind35::Gqa(GqaKind::Gated256, Flash::Group)
                } else {
                    delta
                }
            })
            .collect();
        let lens = checkpoint_lens(&kinds);
        assert_eq!(lens.len(), 60);
        assert!(lens.chunks(2).all(|c| c == [32 * 128 * 128, 11 * 8192]));
        assert_eq!(point_bytes(&lens), 73_728_000);
        let price = price(&lens, 4_096);
        assert_eq!(price.budget(2_048), Ok(2 * 73_728_000));
        assert_eq!(price.budget(10_240), Ok(4 * 73_728_000));
    }
}
