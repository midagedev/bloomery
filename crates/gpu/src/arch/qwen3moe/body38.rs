//! The Qwen3.8-Flash-Next (`qwen4exp`) `ChainBody`: 48 layers in four
//! gated-residual streams — 36 sigmoid-gated delta-rule layers and 12
//! attention layers whose positions a mean-pool selector picks — each
//! routing to 512 experts — on the host tier, and those its plan puts on
//! the card there — beside a sigmoid-gated shared expert on the card, a PLE
//! site on layer 1, and a head whose mix is the final norm
//! ([`HeadNorm::Mixed`]).
//!
//! Load ([`Body38::open_placed`]) reads the description once
//! (`model::arch::qwen35moe::place::PlanInputs`), checks the chain's shape
//! against the kernels' geometry (`plan38`), refuses by name every feature
//! of the file the program does not run past the two it allows
//! ([`ALLOWED`]), joins each delta layer's β and α into one stack and each
//! router with the shared expert's gate as its last row, and builds the
//! stores, two arenas (the step's one row, an eager pass's
//! [`PASS_ROWS`](super::scratch38::PASS_ROWS)), the input records, the slot
//! map of the plan (`SlotMap::of_plan`: each layer's routed experts the plan
//! puts on the card, the rest the host's), the card leg over the card's
//! stacks (`card38`), and the host tier over every layer's routed stacks,
//! which serves the map's host slots. A plan with an expert tier card (plan
//! (b′), `place::machine_bp`) hangs that card under the host tier (`tier38`:
//! its routed segments, its set the map's tier rows, its computation, the
//! tiers' page), the slot map then the plan's of both cards
//! (`SlotMap::of_plan_tiers`); the step and the verify serve its layers, the
//! pass and the ubatch walk refuse its map by name, so such a load feeds its
//! prompt by steps. Under an explicit `BLOOMERY_RESIDENCY`
//! word the load also runs the residency machine over the card's stacks
//! (`swap38`: the common machine, Qwen3.8's parts and live delay), which
//! moves the map between passes; unset keeps the load's map (the tier's
//! experts are no part of the machine: held away, never moved). Every walk
//! runs its card side — the step, the verify and the pass the card leg, the
//! ubatch walk its card route (`wide38`) — and a walk with no tier leg (the
//! pass and the ubatch walk, and every walk of a load that hung no tier)
//! refuses by name a map with an expert on a tier card.
//! [`Body38::open_placed_mtp`] also opens the MTP draft layer on the same
//! card ([`Mtp38`]: its weights, its store, the reduced head's rows) and
//! arms its program's arena; its walks ([`GpuModel::mtp_draft`]) read the
//! target's weights and streams and write only the draft's store and
//! arena, so no target walk reads what they leave.
//!
//! The body serves resident sequence slots ([`Slots`],
//! [`GpuModel::add_slots`]): one sequence's stores, PLE ring and hash
//! history, lane word, checkpoints with their points, the positions it
//! holds and where its calls left their rows — and, on a load with the
//! draft, the draft's store, its captures and its own positions — travel
//! as [`Slot38`], exchanged by pointer moves at a select. The step's and
//! the pass's final-stream rows travel with it too: a draft walk reads the
//! rows its sequence's last target walk left in the shared arenas, so each
//! sequence holds its own and the arenas address whichever is live — the
//! same rule the slots' own captures follow. Everything a walk of another
//! sequence cannot read (the weights, the kernels, the arenas' scratch,
//! the host tier and residency machine, the slot map, the fault word)
//! stays one. A load made for one sequence launches and allocates exactly
//! as a load that never serves slots does; a sequence past the count the
//! load's plan was made for is refused by name.
//!
//! The walk is `program38`'s. The decode step is captured; a prompt runs
//! as captured steps, one a position ([`Prompt38::Step`]), as eager passes
//! of up to eight positions through the batch port ([`Prompt38::Pass`]) —
//! each row bit for bit its step — or as ubatches of up to the load's size
//! (`BLOOMERY_QWEN3_UBATCH`, at most 4,096 positions) through the same port
//! at their width (`wide38`, [`Prompt38::Gemm`]), each position's bits a
//! function of its own inputs and not bit for bit the pass's: its Q8_0
//! projections read q8 activations.
//!
//! A verify ([`Rows`]) runs 2 to [`VERIFY_ROWS`] consecutive positions as
//! one captured pass through the step port's `Cols` chain into one head of
//! its rows, each row bit for bit its step, and stands the model that many
//! positions on until its commit ([`Rollback`], `GpuModel::rollback` to the
//! first position not kept) keeps the first `k`: every delta layer records
//! its rows beside its one state lane and leaves the state as it stood, so
//! the commit replays the `k` kept records into the lane
//! ([`Body38::commit`]); the PLE hash's history is replayed over the kept
//! rows. Every other store is by
//! position, and nothing reads a row past the count: a pooled key a
//! rejected row completed is read only at a count that completes it again,
//! and the row at that count writes it before its select reads it. A step,
//! a pass or a verify is refused by name while a verify waits for its
//! commit.
//!
//! A load made a media load ([`GpuModel::enable_media`], once after the open)
//! takes image rows ([`MediaBody`]): an image's M-RoPE positions shift the
//! rows of its sequence, so each sequence holds a rotation table of its own
//! ([`SeqRope`]), made before any graph is captured and exchanged with the
//! sequence; every rope launch of the step, the passes, the verify and the
//! ubatch walk — the selecting layers' keys, selection and q/k — reads the
//! table of the sequence it runs, a pass of several slots each slot's own. A
//! media call is eager and runs every unit as a ubatch ([`Prompt38::Gemm`]):
//! each unit's first layer copies the rows of the spans inside it over the
//! embeddings of their positions (`image::MediaRows`), and the PLE hash takes
//! the image placeholder inside a span only. The MTP draft's walks read the
//! table of the sequence they walk, as the target's do: its store walks over
//! a media call's units take a span's rows in place of their positions'
//! embeddings (`Mtp38::store_rows`), its keys turn by the image's positions.
//!
//! The PLE rows are the host's: each position's n-gram hash
//! (`engram::Hash::ple_rows_into`) names 16 rows of the table, which stays in
//! the file; they are decoded (IQ4_NL rows by `gguf::quant::dequant_row`,
//! bloomery's F8 and NVFP4 rows by `gguf::quant::E4m3Rows` and
//! `gguf::quant::Nvfp4Rows` at the table's scale) into
//! the position's 2,560 values and copied to the arena ahead of the step. The
//! hash's history is the sequence's: a step at any position other than the
//! next is refused by name, and `reset` starts a new one.
//!
//! The model's position (`GpuModel::pos`) owns the next position. The body
//! keeps what its stores hold ([`Body38::kept`] reads it), which equals the
//! position except after a call failed past its launch. A call's plan (its
//! PLE rows, its record, its copies) moves nothing it would have to undo:
//! the history past its rows waits beside it, and the stores' count, the
//! history and a verify's commit move together at the launch — a step's at
//! the end of its refresh, a pass's right before its walk, a verify's at
//! the end of its plan. A call that fails before that runs again at the
//! same position with the same bits. One that fails after it leaves the
//! recurrent stores one call on, so a call at that position is refused by
//! name until `reset`: a delta layer keeps no earlier state to cut back to.

use super::body::KvQ8;
use super::card38::{Card38, MapCheck, Walk38};
use super::image::{
    MediaCarrier, MediaRows, imrope_sections, media_call, section_order, window_spans,
};
use super::mtp38::{
    CHAIN_PAIRS, ChainBlock, DraftRows38, DraftSeq38, MTP_ROWS, Mtp38, MtpCtx, MtpDraft, MtpTaps,
    Target38, TargetRows, chain_spans, check_target_rows,
};
use super::plan38::{self, GDN, Kind38, Layer38, Shape38, beta_alpha, geo, router};
use super::program38::{
    Ctx38, Kernels38, PLE_ARRIVAL_NODES, Parts38, Pass38, RowsCopy, STEP_MEMOPS, Seqs38, SlotSeq38,
    Step38, Verify38, seq_launches, step_launches, verify_launches,
};
use super::scratch::{
    Io, KvPlanes, LANE, LaneWord, PassRecord, RecStore, RopeRows, StepParams, stamps_at,
};
use super::scratch38::{
    Arena38, LANES, PASS_ROWS, Ple38, Store38, Taps38, VERIFY_ROWS, WideRecord, dims, final_res,
    no_carry, store_rule_bytes,
};
use super::slot_pass::SlotIn;
use super::swap38::{self, DEADLINE, LIVE_DELAY};
use super::tier38::{TierSide38, open_tier};
use super::ubatch::UBATCH as UBATCH_MOST;
use super::wide38::{
    Gemm38, PleRowsIn, STREAM_FLOOR, SplitWalk, Wide38, WideForce, WideForms, WideParts, WideTaps,
    dense_rows, route_taps_host,
};
use crate::checkpoint::bytes::{self, Fail, Put, refused};
use crate::checkpoint::saved::{
    HostBytes, Identity, Lent, Pending, SaveLane, SeqState, SeqView, Spans, Streams,
};
use crate::checkpoint::staging::NoRoom;
use crate::checkpoint::{self, Checkpoints};
use crate::draft::{DraftFeed, DraftHead, Feed, Held, Hidden, WalkMode};
use crate::folder::{Folder, Model};
use crate::head::{Head, HeadLaunches, HeadNorm};
use crate::host::nvtier::NvTier;
use crate::host::prompt_stream::PromptStream;
use crate::host::run::{HostRun, HostWidths};
use crate::host::served::{ResidencyParts, TierBody};
use crate::host::swap::{CallPick, CallReport, PassReport, Residency};
use crate::host::swap_source::{FileSwap, ResidencyGlue, ResidencySpec};
use crate::host::tier::TierOpen;
use crate::host::xstream::{Costs, XCfg, XLayer, XMode, XReport, XSTREAM_FORM};
use crate::host::{BatchLeg, PassKind, StepLeg, refuse_tier_count};
use crate::hybrid::{Boundary, BoundaryShape, Chain, HostResidency, Hybrid, SlotMap};
use crate::linear::delta::{RecoveryStore, RecoveryTable, carry_fold_host};
use crate::media::{ImageSpan, MediaBody, MediaSpan, RopeBase, SeqRope, kept_rule, rope_table};
use crate::model::slots::slot_twice;
use crate::model::{
    CardBytes, ChainBody, GpuModel, HostServed, MAX_PASS_ROWS, Rollback, RowHeads, Rows, SlotRange,
    SlotRowOrder, SlotRows, Slots,
};
use crate::ple::PleKernels;
use crate::prompt_timing::{PromptStats, PromptTiming, nanos};
use crate::rope_table::{RopeSpec, RopeTable};
use crate::site::{BY_NAME_COLS, ByName};
use crate::tensor::{WindowMut, to_host};
use crate::weights::{Weights, tensor_scales};
use crate::{Branch, DeviceTensor, Gpu, GpuError, SharedPages, StagedRows, launch_u32, ple, qsa};
use bloomery_levers::{ChurnHosting, HostCfg};
use cuda_core::{CudaEvent, CudaStream, DeviceBuffer, sys};
use engram::hash::{History, Window};
use engram::prefetch::{FillMode, HelperOptions, Prefetcher, caller_sibling};
use engram::rows::{PageAdvice, RunTable, SplitTable};
use engram::{Hash, RowTables};
use gguf::quant::{E4m3Rows, Nvfp4Rows, dequant_row};
use gguf::{GgmlType, Split, TensorInfo};
use model::arch::Arch;
use model::part::StoredRows;
use model::placement::{Device, Plan};
use runtime::seqstate::{Kept, Price, Take};
use runtime::swaprule::KeptRows;
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;
use vision::mrope::MropeSeq;

const WHAT: &str = "qwen4exp Body38";
/// The label of a share's refusals ([`GpuModel::seq_share_prefix`]).
pub(super) const WHAT_SHARE: &str = "qwen4exp share";

/// The coverage items the program opens a file with: the tokenizer's
/// pre-tokenizer and the template's tool-call parser are the chat surface's,
/// which the program does not use. Every other item the check lists is
/// refused by name.
pub const ALLOWED: &[&str] = &[
    "pre-tokenizer qwen35",
    "a tool-call parser for this template",
];

/// The Qwen3.8 model: one card, the skeleton over this body.
pub type Qwen38Model = GpuModel<Body38>;

/// A Qwen3.8 sequence state on the host ([`GpuModel::seq_save`]): the
/// shared positional rows and recurrent stores with their PLE histories
/// ([`SeqState<History>`]), on a load with the draft its store rows among
/// the positional rows, and the layer's side beside them ([`DraftRows38`]:
/// the positions its store holds and the two arenas' rows), so a sequence
/// put back after another ran drafts as if none had.
pub struct Seq38 {
    state: Arc<SeqState<History>>,
    draft: Option<DraftRows38>,
}

impl Seq38 {
    /// The positions it holds.
    #[must_use]
    pub fn positions(&self) -> u32 {
        self.state.positions()
    }

    /// The host bytes it holds: the shared state's and the draft side's.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.state.bytes() + self.draft.as_ref().map_or(0, DraftRows38::bytes)
    }

    /// The longest prefix of at most `n` positions a model holds right after
    /// the state is put back ([`SeqState::keep_point`]).
    #[must_use]
    pub fn keep_point(&self, n: u32) -> u32 {
        self.state.keep_point(n)
    }

    /// Refused by name unless a draft walk from `pos0` that reads rows
    /// `first .. first + m` of the `walk` arena reads the target's final
    /// hidden rows the state carries for the positions before it, as the
    /// walk itself checks them ([`check_target_rows`]): a state without the
    /// draft's side, or the ubatch arena, which a state does not carry, is
    /// refused.
    pub fn check_draft_rows(
        &self,
        walk: TargetRows,
        (first, m): (usize, usize),
        pos0: u32,
    ) -> Result<(), Fail> {
        let d = self
            .draft
            .as_ref()
            .ok_or_else(|| refused("a parked MTP draft on a state without the draft's side"))?;
        let records = match walk {
            TargetRows::Step => (d.step.0, d.final_rows[0], 1),
            TargetRows::Pass => (d.pass.0, d.final_rows[1], PASS_ROWS),
            TargetRows::Ubatch => return Err(refused("a parked MTP draft's ubatch rows")),
        };
        check_target_rows("qwen4exp slot restore", walk, records, (first, m), pos0)
            .map_err(Fail::Refused)
    }
}

/// The first eight bytes of a Qwen3.8 state's byte form ([`Seq38::put`]).
const STATE_TAG: [u8; 8] = *b"BLMQ38SQ";

/// The byte form's layout ([`Seq38::put`]): raised whenever a part, its
/// order or its encoding changes.
const STATE_FORMAT: u32 = 1;

/// A PLE history in a state's byte form: its tokens (a count, then each,
/// u32), then the position it stands before (u64).
impl HostBytes for History {
    fn put(&self, w: &mut Put<'_>) -> Result<(), Fail> {
        w.array(self.tokens())?;
        w.one(self.next_pos())
    }

    fn take(r: &mut bytes::Take<'_>) -> Result<History, Fail> {
        let tokens = r.array("a PLE history")?;
        Ok(History::of(tokens, r.one("a PLE history")?))
    }
}

/// A saved text history has this load's length and only text ids: snapshots
/// carry no image rotation state.
fn check_ple_history(h: &History, len: usize, window: &Window) -> Result<(), Fail> {
    if h.tokens().len() != len {
        return Err(refused(format!(
            "a PLE history of {} tokens; this load's holds {len}",
            h.tokens().len()
        )));
    }
    let Window::Eos(window) = window else {
        return Err(refused("a PLE history read for a hash of mapped ids"));
    };
    for &t in h.tokens() {
        window
            .check(t)
            .map_err(|e| refused(format!("a saved PLE history: {e}")))?;
    }
    Ok(())
}

impl Seq38 {
    /// The state in its byte form (`checkpoint::bytes`), what a slot file
    /// carries: [`STATE_TAG`], [`STATE_FORMAT`] (u32), the shared state
    /// ([`SeqState::put`]), then the draft's side (u8: 0 none, 1 its
    /// [`DraftRows38`] after it). [`GpuModel::seq_take`] reads it back.
    pub fn put(&self, w: &mut Put<'_>) -> Result<(), Fail> {
        w.vals(&STATE_TAG)?;
        w.one(STATE_FORMAT)?;
        self.state.put(w)?;
        match &self.draft {
            None => w.one(0u8),
            Some(d) => {
                w.one(1u8)?;
                d.put(w)
            }
        }
    }
}

impl SeqView for Seq38 {
    fn positions(&self) -> u32 {
        Seq38::positions(self)
    }

    fn bytes(&self) -> usize {
        Seq38::bytes(self)
    }

    fn keep_point(&self, n: u32) -> u32 {
        Seq38::keep_point(self, n)
    }
}

/// One Qwen3.8 sequence's resident state ([`Slots`], what a select
/// exchanges): the per-layer stores, the PLE ring with the hash's history
/// and its points, the committed lane word, the verify waiting for its
/// commit (a pass of several slots leaves one in each of its slots), the
/// checkpoints, the positions the stores hold and where the last call left
/// its rows — and, on a load with the draft ([`DraftSeq38`]), the draft's
/// store with the captures that address it and its own record of the
/// positions it holds. The step's and the pass's final-stream rows travel
/// with it too: a draft walk reads the rows its sequence's last target walk
/// left in the shared arenas, so each sequence holds its own and the arenas
/// address whichever is live — the same rule a slot's own captures follow,
/// each slot's graphs addressing the buffers that were live at their
/// capture. A pass of several slots leaves a parked slot's rows in its own
/// buffer by a copy ([`RowsCopy`]), or, for the first slot of a pass that
/// runs no row of slot 0, by walking into that buffer ([`stands_in`]).
pub struct Slot38 {
    stores: Vec<Store38>,
    /// The commit's replay table over `stores` ([`Body38::recovery`]).
    recovery: RecoveryTable,
    ple_ring: DeviceBuffer<f32>,
    lane: LaneWord,
    pending: Option<Pending38>,
    ckpt: Checkpoints,
    ple_points: Vec<(u32, History)>,
    point_rows: Vec<(u32, Vec<f32>)>,
    ple_hist: History,
    held: u32,
    wrote: Wrote38,
    last_walk: TargetRows,
    /// The step arena's final-stream row and the pass arena's
    /// [`PASS_ROWS`] rows, the buffers the shared arenas address while this
    /// sequence is live.
    rows_step: DeviceBuffer<f32>,
    rows_pass: DeviceBuffer<f32>,
    draft: Option<DraftSeq38>,
    /// The sequence's rotation table on a media load ([`Body38::seq_rope`]).
    rope: Option<SeqRope>,
}

impl Slot38 {
    /// The most positions of the first `n` another sequence can take from
    /// this one ([`Body38::share_state`]): the nearest checkpoint at or below
    /// `n` (and the held position) whose PLE history stands and, on a load
    /// with the draft, where the draft rejoins (the point carries the
    /// target's hidden row before it, and the draft's store holds every
    /// position below it); 0 when none stands, a verify waits for its
    /// commit, or on a media load the sequence holds an image (a share does
    /// not carry the rotation table its later rows' positions come from).
    fn share_point(&self, n: u32) -> u32 {
        let images = (self.rope.as_ref()).is_some_and(|r| !r.rows().seq().is_empty());
        if self.pending.is_some() || images {
            return 0;
        }
        let rejoins = |p: u32| {
            self.draft
                .as_ref()
                .is_none_or(|d| d.held() >= p as usize && point_at(&self.point_rows, p).is_some())
        };
        (self.ckpt.positions().into_iter().rev())
            .find(|&p| {
                p <= n.min(self.held) && point_at(&self.ple_points, p).is_some() && rejoins(p)
            })
            .unwrap_or(0)
    }
}

/// The index of the entry at position `p` among a sequence's points' host
/// sides (its PLE histories, its hidden rows).
fn point_at<T>(points: &[(u32, T)], p: u32) -> Option<usize> {
    points.iter().position(|&(q, _)| q == p)
}

/// The most host bytes a sequence state of `positions` positions holds on
/// a load of `gdn` delta and `qsa` selecting layers, its K/V rows in cache
/// format `kv`, with the draft's side when loaded: the positional rows and
/// current recurrent copy plus saved points bounded by the context and
/// checkpoint budget, the draft's store in the same format, the step and
/// pass arena rows, and one hidden row per saved inner point.
#[must_use]
pub fn seq38_bytes(gdn: usize, qsa: usize, positions: usize, drafts: bool, kv: KvQ8) -> u64 {
    let points = checkpoint_price38(gdn, qsa).saved_points(positions as u64);
    let shared =
        seq_positional_bytes(qsa, positions, kv) + (1 + points) * checkpoint_bytes(gdn, qsa);
    if !drafts {
        return shared;
    }
    let store = kv_rows_bytes(positions, kv);
    let rows = ((1 + PASS_ROWS) * geo::STREAMS * geo::HIDDEN * size_of::<f32>()) as u64;
    shared + store + rows + points * POINT_ROW_BYTES
}

/// The host bytes of the target's final hidden row a checkpoint keeps for
/// the MTP draft ([`Body38::point_rows`], [`DraftRows38::point_rows`]): one
/// position's four streams of [`geo::HIDDEN`] f32.
const POINT_ROW_BYTES: u64 = (geo::STREAMS * geo::HIDDEN * size_of::<f32>()) as u64;

/// The host bytes one sequence's checkpoint hidden rows hold at most on a
/// load with the MTP draft, in a load of `ctx` positions (a slot's share) of
/// `gdn` delta layers and `qsa` selecting layers: one [`POINT_ROW_BYTES`] row
/// for each point the sequence's checkpoint budget holds
/// ([`checkpoint_budget38`]).
pub fn point_row_budget38(gdn: usize, qsa: usize, ctx: u64) -> Result<u64, GpuError> {
    Ok(checkpoint_budget38(gdn, qsa, ctx)? / checkpoint_bytes(gdn, qsa) * POINT_ROW_BYTES)
}

/// The spacing of a prompt call's inner checkpoints (`runtime::seqstate`'s
/// marks), when the load takes them ([`Body38::set_checkpoints`]): the
/// ubatch walk's most positions, so a mark cuts a call's ubatches at most
/// once more than the call's own cut, and a context of 60k positions takes
/// 15 inner marks against the host budget's 29 slots of one copy
/// ([`checkpoint_bytes`]).
pub const CHECKPOINT_EVERY: u32 = UBATCH_MOST as u32;

// The plan prices a Qwen3.8 load's saved states at this spacing
// (`KvLayout::checkpoint_price`).
const _: () = assert!(CHECKPOINT_EVERY as u64 == model::arch::qwen35moe::place::UBATCH_PLANNED);

/// The f32s of each store one checkpoint of a load of `gdn` delta layers
/// and `qsa` selecting layers copies, in the layers' order: each delta
/// layer's committed lane, conv ring and carry, then the PLE ring, then each
/// selecting layer's raw key ring (its f16 pairs as f32 words, a copy that
/// reads no value). The one derivation of what a checkpoint holds, read by
/// the sequences' `Checkpoints`, [`checkpoint_bytes`] and
/// [`checkpoint_price38`].
fn checkpoint_lens(gdn: usize, qsa: usize) -> Vec<usize> {
    (0..gdn)
        .flat_map(|_| [GDN.state_len(), GDN.ring_len(), GDN.carry_len()])
        .chain([ple::ring_len(geo::STREAMS)])
        .chain((0..qsa).map(|_| KEY_RING_WORDS))
        .collect()
}

/// f32 words of a selecting layer's raw key ring (`[qsa::RING_ROWS][IDX_DIM]`
/// f16), as a checkpoint copies it.
const KEY_RING_WORDS: usize = qsa::RING_ROWS * geo::IDX_DIM / 2;
const _: () = assert!((qsa::RING_ROWS * geo::IDX_DIM).is_multiple_of(2));

/// [`checkpoint_lens`] of the delta and the selecting layers among `stores`.
fn checkpoint_lens_of(stores: &[Store38]) -> Vec<usize> {
    let gdn = stores
        .iter()
        .filter(|s| matches!(s, Store38::Rec { .. }))
        .count();
    checkpoint_lens(gdn, stores.len() - gdn)
}

/// The layer whose walk writes each store of [`checkpoint_lens`]'s order,
/// over layers that are delta (`true`) or selecting (`false`) in `delta`:
/// each delta layer's three stores its own, the PLE ring the PLE `site`'s,
/// then each selecting layer's key ring its own.
fn checkpoint_owners(delta: impl Iterator<Item = bool> + Clone, site: usize) -> Vec<usize> {
    let layers = delta.enumerate();
    let rec = layers.clone().filter(|&(_, d)| d).flat_map(|(l, _)| [l; 3]);
    let keys = layers.filter(|&(_, d)| !d).map(|(l, _)| l);
    rec.chain([site]).chain(keys).collect()
}

/// The commit's replay table over the delta layers among `stores`, each
/// under its own layer's fault.
fn recovery_table(stream: &CudaStream, stores: &[Store38]) -> Result<RecoveryTable, GpuError> {
    let rows: Vec<RecoveryStore<'_>> = stores
        .iter()
        .enumerate()
        .filter_map(|(layer, s)| match s {
            Store38::Rec { rec, stamp, record } => Some(RecoveryStore {
                layer,
                state: &rec.state,
                stamp,
                carry: rec.carry.as_ref(),
                record,
            }),
            Store38::Qsa { .. } => None,
        })
        .collect();
    // SAFETY: the stores live in the body beside the table and are never
    // replaced or reallocated after the load, and the commit's replay is
    // the only launch on them while it runs (the stream orders the rest).
    unsafe { RecoveryTable::new(stream, GDN, (LANES, VERIFY_ROWS), &rows) }
}

/// The bytes one checkpoint of a load of `gdn` delta layers and `qsa`
/// selecting layers copies: each delta layer's committed lane, conv ring and
/// carry, the PLE ring, and each selecting layer's raw key ring.
#[must_use]
pub fn checkpoint_bytes(gdn: usize, qsa: usize) -> u64 {
    checkpoint::point_bytes(&checkpoint_lens(gdn, qsa))
}

/// What one checkpoint of a load of `gdn` delta layers and `qsa` selecting
/// layers costs on the host and how a prompt call spaces them
/// ([`CHECKPOINT_EVERY`]): the figure every sequence's `Checkpoints` is made
/// with ([`Checkpoints::sequence`]) and a plan's host reserve prices
/// (`residency38::reserve_checkpoints`), so the two cannot differ.
#[must_use]
pub fn checkpoint_price38(gdn: usize, qsa: usize) -> Price {
    checkpoint::price(&checkpoint_lens(gdn, qsa), CHECKPOINT_EVERY)
}

/// The host bytes one sequence's checkpoints pin at most in a load of `ctx`
/// positions (a slot's share) of `gdn` delta layers and `qsa` selecting
/// layers ([`Price::budget`] of [`checkpoint_price38`]).
pub fn checkpoint_budget38(gdn: usize, qsa: usize, ctx: u64) -> Result<u64, GpuError> {
    checkpoint::budget(checkpoint_price38(gdn, qsa), ctx)
}

/// The positional bytes a sequence state of `positions` positions carries
/// for `qsa` selecting layers: each one's positional rows, its K and V in
/// the cache format `kv` ([`runtime::stores::selecting_rows_bytes`]; its raw
/// key ring is a checkpoint store, [`checkpoint_bytes`]).
#[must_use]
pub const fn seq_positional_bytes(qsa: usize, positions: usize, kv: KvQ8) -> u64 {
    qsa as u64
        * runtime::stores::selecting_rows_bytes(
            geo::N_KV,
            geo::HEAD,
            matches!(kv, KvQ8::Q8),
            geo::IDX_DIM,
            geo::POOL,
            positions,
        )
}

/// One attention store's K and V rows of `positions` positions in the cache
/// format `kv` (`runtime::stores::kv_plane_bytes`).
const fn kv_rows_bytes(positions: usize, kv: KvQ8) -> u64 {
    let q8 = matches!(kv, KvQ8::Q8);
    runtime::stores::kv_plane_bytes(geo::N_KV, geo::HEAD, q8, positions as u64)
}

/// How a prompt runs: captured steps, one a position — the decode step's
/// graph — eager passes of up to eight positions through the host tier's
/// batch port, both leaving every position's state and logits bit for bit;
/// or ubatches of up to the load's size through the same port at their
/// width, the Q8_0 projections on q8 activations (`wide38`). `Auto` is the
/// ubatches from [`Prompt38::GEMM_FROM`] positions on and the passes below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prompt38 {
    Step,
    Pass,
    Gemm,
    Auto,
}

impl Prompt38 {
    /// Positions an eager pass takes at most.
    pub const PASS_ROWS: usize = PASS_ROWS;
    /// The fewest positions `Auto` runs as ubatches: one past a pass.
    pub const GEMM_FROM: usize = PASS_ROWS + 1;

    /// The path a command line names: `step`, `pass`, `gemm` or `auto`;
    /// anything else is refused by name.
    pub fn parse(s: &str) -> Result<Prompt38, GpuError> {
        match s {
            "step" => Ok(Prompt38::Step),
            "pass" => Ok(Prompt38::Pass),
            "gemm" => Ok(Prompt38::Gemm),
            "auto" => Ok(Prompt38::Auto),
            other => Err(GpuError::shape(
                WHAT,
                format!("prompt path {other:?}: `step`, `pass`, `gemm` or `auto`"),
            )),
        }
    }

    /// The path a prompt of `n` positions runs: `Auto` resolved, any other
    /// itself.
    #[must_use]
    pub fn resolve(self, n: usize) -> Prompt38 {
        match self {
            Prompt38::Auto if n >= Prompt38::GEMM_FROM => Prompt38::Gemm,
            Prompt38::Auto => Prompt38::Pass,
            p => p,
        }
    }

    /// The name a record prints.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Prompt38::Step => "step",
            Prompt38::Pass => "pass",
            Prompt38::Gemm => "gemm",
            Prompt38::Auto => "auto",
        }
    }
}

/// A layer's mixer, as a gate lists them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind38 {
    /// The sigmoid-gated delta rule over its recurrent store.
    Gdn,
    /// Selecting gated attention over its K/V planes.
    Qsa,
}

/// The PLE site's host side: the hash and the sequence's history, the table
/// in the file, the rows of the positions last filled, decoded, and — when
/// the plan left the table on the NVMe tier — its reader. One mapping of the
/// table, two residencies: on the host the load's host set holds its pages
/// and a fill decodes them from the split's mapping as they are; on the NVMe
/// tier a fill's rows are advised and copied off the step thread
/// ([`PleRead`]) and decoded from the copy.
struct PleHost {
    hash: Hash,
    hist: History,
    /// [`PleHost::prompt_start`]'s walk of the history ahead of the fills,
    /// its storage kept from call to call.
    ahead_hist: History,
    file: Arc<Split>,
    shard: usize,
    info: TensorInfo,
    /// Table rows, values a row, bytes a row, and how a row decodes.
    rows: usize,
    width: usize,
    row_bytes: usize,
    decode: PleRows,
    /// Row ids of up to a call's positions (a pass's or a ubatch's), the
    /// count the last fill or keep named, and the fill's decoded values
    /// (`HIDDEN` a position).
    ids: Vec<u32>,
    named: usize,
    e: Vec<f32>,
    /// The NVMe tier's reader; `None` on the host.
    nvme: Option<PleRead>,
}

/// The PLE table's reads from the NVMe tier: engram's prefetcher over the
/// table in the split's own mapping ([`SplitTable`]), whose helper advises
/// a fill's rows and copies them, so the step thread takes neither the
/// advice nor a fault; a ubatch's rows (16 a position, at hashed places in
/// the table) in parts side by side over the worker pool's threads
/// ([`HelperOptions::spread`]). A prompt call's later
/// ubatches are read one ahead while the ubatch before them computes
/// ([`PleHost::prompt_start`]). A fill that waits for its rows at once and
/// finds none read ahead (a verify's, a step's with no read ahead) reads
/// them with [`Prefetcher::read_now`]: on the step thread when the helper
/// would read them alone, so no thread wakes for them.
struct PleRead {
    rows: Prefetcher,
    planes: usize,
    /// A prompt call's ubatches not yet filled, in order; while `flight`
    /// holds, the front's rows are on the helper.
    ahead: VecDeque<PleBatch>,
    /// The row ids of the batch last taken off `ahead`, the storage the next
    /// [`PleHost::prompt_start`] fills first: a decode step's read ahead
    /// ([`Body38::prefetch_input`]) allocates nothing.
    spare_ids: Vec<u32>,
    flight: bool,
    /// The started fill's rows already copied ([`PleRead::read_now`]).
    read: bool,
}

/// One ubatch of a prompt call as [`PleHost::prompt_start`] reads it ahead:
/// its first position, its positions and its row ids.
struct PleBatch {
    pos: u32,
    n: usize,
    ids: Vec<u32>,
}

impl PleRead {
    /// The reader of `file`'s table `name`, or of its rows where `stored`
    /// holds them (a checkpoint folder's, [`RunTable`]), for fills of up to
    /// `max_rows` rows; its helper on the SMT sibling of the loading thread's
    /// core when that thread is pinned (the step thread loads), else
    /// floating.
    fn open(
        file: &Arc<Split>,
        stored: Option<&StoredRows>,
        name: &str,
        max_rows: usize,
        step_rows: usize,
        expected_rows: usize,
    ) -> Result<PleRead, GpuError> {
        let table: Arc<dyn RowTables> = match stored {
            None => Arc::new(
                SplitTable::open(Arc::clone(file), name).map_err(|e| GpuError::plan(WHAT, e))?,
            ),
            Some(s) => {
                let open = |p: &model::part::StoredPlane| {
                    let table = RunTable::open(name, p.row_bytes, &p.runs, PageAdvice::Random)
                        .map_err(|e| GpuError::plan(WHAT, e))?;
                    if table.rows() != expected_rows as u64 {
                        return Err(GpuError::shape(
                            WHAT,
                            format!(
                                "{name}: source plane has {} rows, the table {expected_rows}",
                                table.rows()
                            ),
                        ));
                    }
                    Ok(table)
                };
                let values = open(&s.values)?;
                match &s.block_scales {
                    None => Arc::new(values),
                    Some(scales) => Arc::new(vec![values, open(scales)?]),
                }
            }
        };
        let planes = table.count();
        let rows = Prefetcher::over(
            table,
            &[max_rows; 2][..planes],
            HelperOptions::spread(FillMode::Touch, caller_sibling(), false),
        )
        .map_err(|e| GpuError::plan(WHAT, e))?;
        Ok(PleRead {
            rows,
            planes,
            ahead: VecDeque::with_capacity(1),
            spare_ids: Vec::with_capacity(step_rows),
            flight: false,
            read: false,
        })
    }

    /// The batch on the helper taken back unread: a call that stopped
    /// before its fills took it, or a new prompt call.
    fn drain(&mut self) -> Result<(), GpuError> {
        if std::mem::take(&mut self.flight) {
            ple_wait(|| self.rows.wait()).map_err(|e| GpuError::plan(WHAT, e))?;
        }
        Ok(())
    }

    /// The front batch off `ahead`, its row-id storage kept for the next
    /// read ahead.
    fn recycle_front(&mut self) {
        if let Some(batch) = self.ahead.pop_front() {
            self.spare_ids = batch.ids;
        }
    }

    /// The front batch's rows on the helper, when there is one.
    fn start_front(&mut self) -> Result<(), GpuError> {
        if let Some(next) = self.ahead.front() {
            self.rows
                .submit(std::iter::repeat_n(next.ids.as_slice(), self.planes))
                .map_err(|e| GpuError::plan(WHAT, e))?;
            self.flight = true;
        }
        Ok(())
    }

    /// The fill of `n` positions at `pos` naming `ids` taken in: the batch
    /// read ahead when it is this fill's, else every batch read ahead
    /// dropped. Whether the fill's rows are then on the helper.
    fn take_in(&mut self, pos: u32, n: usize, ids: &[u32]) -> Result<bool, GpuError> {
        self.read = false;
        let own = self
            .ahead
            .front()
            .is_some_and(|b| b.pos == pos && b.n == n && b.ids == ids);
        if own {
            self.recycle_front();
        } else {
            self.drain()?;
            self.recycle_front();
            self.ahead.clear();
        }
        Ok(self.flight)
    }

    /// `ids`, the rows of the fill of `n` positions at `pos`, on the helper:
    /// the batch read ahead when it is this fill's, else submitted now.
    /// [`PleRead::finish`] takes them.
    fn start(&mut self, pos: u32, n: usize, ids: &[u32]) -> Result<(), GpuError> {
        if !self.take_in(pos, n, ids)? {
            self.rows
                .submit(std::iter::repeat_n(ids, self.planes))
                .map_err(|e| GpuError::plan(WHAT, e))?;
            self.flight = true;
        }
        Ok(())
    }

    /// [`PleRead::start`] for a fill that waits for its rows at once: the
    /// batch read ahead when it is this fill's, else the rows read now
    /// ([`Prefetcher::read_now`]). [`PleRead::finish`] takes them.
    fn read_now(&mut self, pos: u32, n: usize, ids: &[u32]) -> Result<(), GpuError> {
        if !self.take_in(pos, n, ids)? {
            ple_wait(|| self.rows.read_now(std::iter::repeat_n(ids, self.planes)))
                .map_err(|e| GpuError::plan(WHAT, e))?;
            self.read = true;
        }
        Ok(())
    }

    /// The started fill's rows ([`PleRead::start`], [`PleRead::read_now`])
    /// copied to [`Prefetcher::filled`]; then the next ubatch read ahead
    /// goes to the helper, to run while this one computes.
    fn finish(&mut self) -> Result<(), GpuError> {
        let (helper, read) = (
            std::mem::take(&mut self.flight),
            std::mem::take(&mut self.read),
        );
        if helper {
            ple_wait(|| self.rows.wait()).map_err(|e| GpuError::plan(WHAT, e))?;
        } else if !read {
            return Err(GpuError::state(
                WHAT,
                "a PLE fill finished before it started",
            ));
        }
        self.start_front()
    }
}

/// `read`, the step thread waiting for PLE rows off the NVMe tier, timed as
/// the engine's PLE wait ([`crate::waits`]).
fn ple_wait<R>(read: impl FnOnce() -> R) -> R {
    let t = Instant::now();
    let r = read();
    crate::waits::ple(t.elapsed());
    r
}

/// How the PLE table's rows decode: IQ4_NL blocks, or bloomery's F8 or NVFP4
/// rows at the table's scale.
enum PleRows {
    Iq4Nl,
    F8(Box<E4m3Rows>),
    Nvfp4(Box<Nvfp4Rows>),
}

/// `rows`, rows of `width` values each, decoded into `e` in order.
fn ple_decode<'r>(
    e: &mut [f32],
    width: usize,
    decode: &PleRows,
    rows: impl Iterator<Item = &'r [u8]>,
) -> Result<(), GpuError> {
    for (src, out) in rows.zip(e.chunks_exact_mut(width)) {
        match decode {
            PleRows::Iq4Nl => dequant_row(GgmlType::IQ4_NL, src, out),
            PleRows::F8(f8) => f8.decode(src, out),
            PleRows::Nvfp4(fp4) => fp4.decode(src, out),
        }
        .map_err(model::ModelError::from)?;
    }
    Ok(())
}

impl PleHost {
    /// The table `per_layer_token_embd` of `file`, IQ4_NL rows or bloomery's
    /// F8 or NVFP4 rows with the table's scale (`per_layer_token_embd.scale`, read by
    /// [`tensor_scales`]) of the hash's width, and the hash from the first
    /// shard's keys, for calls of up to
    /// `cap` positions, read from `tier` — the plan's
    /// ([`Plan::row_tier`]): the host, or the NVMe tier through its reader
    /// ([`PleRead`]); any other tier is refused by name. A checkpoint
    /// folder's table (`file` its header) is read from its stored value and
    /// optional block-scale planes, from the NVMe tier alone. Load-time only.
    fn new(
        file: Arc<Split>,
        stored: Option<&StoredRows>,
        n_vocab: usize,
        cap: usize,
        tier: Device,
    ) -> Result<PleHost, GpuError> {
        let name = model::arch::qwen35moe::names::per_layer_token_embd();
        let refuse = |need: &'static str| GpuError::Tensor {
            what: WHAT,
            name: name.clone(),
            need,
        };
        let vocab = u32::try_from(n_vocab)
            .map_err(|_| GpuError::shape(WHAT, format!("a vocabulary of {n_vocab}")))?;
        let hash = Hash::ple_from_keys(file.keys(), "qwen4exp", vocab)
            .map_err(|e| GpuError::plan(WHAT, e))?;
        let hist = hash
            .new_history()
            .map_err(|e| GpuError::shape(WHAT, e.to_string()))?;
        let (shard, info) = file
            .find(&name)
            .map(|(s, t)| (s, t.clone()))
            .ok_or_else(|| refuse("in the file"))?;
        let width = info.dims.first().copied().unwrap_or(0) as usize;
        let rows = info.dims.get(1).copied().unwrap_or(0) as usize;
        let per_token = (geo::HIDDEN).checked_div(width).unwrap_or(0);
        let scale = || match stored {
            Some(s) => Ok(s.scale),
            None => tensor_scales(WHAT, &info, |n| {
                let (s, t) = file.find(n)?;
                Some((file.shard(s)?, t))
            })?
            .and_then(|s| s.first().copied())
            .ok_or_else(|| refuse("a scale beside its rows")),
        };
        let decode = match (info.ty, stored) {
            (GgmlType::IQ4_NL, None) => PleRows::Iq4Nl,
            (GgmlType::F8E4M3_ROWS, _) => PleRows::F8(Box::new(E4m3Rows::new(scale()?))),
            (GgmlType::NVFP4_ROWS, _) => PleRows::Nvfp4(Box::new(Nvfp4Rows::new(scale()?))),
            (_, None) => return Err(refuse("IQ4_NL, F8 or NVFP4 rows")),
            (_, Some(_)) => {
                return Err(refuse(
                    "F8 or NVFP4 rows where a checkpoint folder stores them",
                ));
            }
        };
        let (Some(blck), Some(size)) = (info.ty.blck_size(), info.ty.type_size()) else {
            return Err(refuse("a sized type"));
        };
        let (blck, size) = (blck as usize, size as usize);
        if width == 0 || !width.is_multiple_of(blck) || per_token * width != geo::HIDDEN {
            return Err(refuse("rows of a whole part of the model width"));
        }
        let row_bytes = width / blck * size;
        if rows == 0 || rows.checked_mul(row_bytes).map(|n| n as u64) != Some(info.nbytes) {
            return Err(refuse("a whole number of rows"));
        }
        if let Some(s) = stored {
            let fits = match (info.ty, &s.block_scales) {
                (GgmlType::F8E4M3_ROWS, None) => s.values.row_bytes == width as u64,
                (GgmlType::NVFP4_ROWS, Some(scales)) => {
                    s.values.row_bytes == (width / 2) as u64
                        && scales.row_bytes == (width / 16) as u64
                }
                _ => false,
            };
            if !fits {
                return Err(refuse(
                    "value and block-scale planes of the table's row type",
                ));
            }
        }
        let nvme = match tier {
            Device::Host if stored.is_some() => {
                return Err(refuse(
                    "on the NVMe tier, where a checkpoint folder's rows are read",
                ));
            }
            Device::Host => None,
            Device::Nvme => Some(PleRead::open(
                &file,
                stored,
                &name,
                cap * per_token,
                per_token,
                rows,
            )?),
            Device::Card(_) | Device::Unused => {
                return Err(refuse(
                    "on the host or the NVMe tier, where the plan reads it",
                ));
            }
        };
        Ok(PleHost {
            hash,
            ahead_hist: hist.clone(),
            hist,
            file,
            shard,
            info,
            rows,
            width,
            row_bytes,
            decode,
            ids: vec![0; cap * per_token],
            named: 0,
            e: vec![0.0; cap * geo::HIDDEN],
            nvme,
        })
    }

    /// The rows of `tokens` at positions `pos ..`, decoded into the first
    /// `tokens.len()` positions of `e`, and the history past them, returned:
    /// the history stays where it stands until the call's launch takes the
    /// returned one ([`Body38::launch`]). On the NVMe tier the rows come
    /// from the reader's copy ([`PleRead::finish`]), on the host from the
    /// mapping. A token the hash refuses (the image placeholder, an id past
    /// the vocabulary), a position other than the history's next, or a row
    /// past the table is refused by name.
    fn fill(&mut self, pos: u32, tokens: &[u32]) -> Result<History, GpuError> {
        self.fill_spanned(pos, tokens, &[])
    }

    /// [`PleHost::fill`] for a unit that holds image positions: `spans` are
    /// the unit's image spans as row ranges of `tokens`
    /// ([`Hash::ple_rows_media`]), inside which the image placeholder
    /// hashes like any id and the history takes it; outside them it is
    /// refused as in `fill`. The rows named ([`PleHost::name_rows`]), then
    /// read ([`PleHost::read_named`]).
    fn fill_spanned(
        &mut self,
        pos: u32,
        tokens: &[u32],
        spans: &[Range<usize>],
    ) -> Result<History, GpuError> {
        let hist = self.name_rows(pos, tokens, spans)?;
        self.read_named(pos, tokens.len())?;
        Ok(hist)
    }

    /// The rows [`PleHost::name_rows`] named for the fill of `n` positions
    /// at `pos`: on the NVMe tier read now ([`PleRead::read_now`]), then
    /// decoded ([`PleHost::decode`]).
    fn read_named(&mut self, pos: u32, n: usize) -> Result<(), GpuError> {
        if let Some(read) = &mut self.nvme {
            read.read_now(pos, n, &self.ids[..self.named])?;
        }
        self.decode()
    }

    /// [`PleHost::fill_spanned`]'s rows named, for a walk that takes them
    /// later: on the NVMe tier their read started on the helper
    /// ([`PleRead::start`]). [`PleHost::decode`] takes them.
    fn name_spanned(
        &mut self,
        pos: u32,
        tokens: &[u32],
        spans: &[Range<usize>],
    ) -> Result<History, GpuError> {
        let hist = self.name_rows(pos, tokens, spans)?;
        if let Some(read) = &mut self.nvme {
            read.start(pos, tokens.len(), &self.ids[..self.named])?;
        }
        Ok(hist)
    }

    /// The rows of `tokens` at `pos` named into `ids` and the history past
    /// them returned, every refusal made.
    fn name_rows(
        &mut self,
        pos: u32,
        tokens: &[u32],
        spans: &[Range<usize>],
    ) -> Result<History, GpuError> {
        let n = tokens.len();
        let per_token = geo::HIDDEN / self.width;
        let ids = self
            .ids
            .get_mut(..n * per_token)
            .ok_or_else(|| GpuError::shape(WHAT, format!("{n} PLE positions at once")))?;
        let mut hist = self.hist.clone();
        self.named = 0;
        self.hash
            .ple_rows_media(&mut hist, u64::from(pos), tokens, spans, ids)
            .map_err(|e| GpuError::shape(WHAT, format!("PLE rows at position {pos}: {e}")))?;
        self.named = ids.len();
        if let Some(&r) = ids.iter().find(|&&r| r as usize >= self.rows) {
            return Err(GpuError::shape(
                WHAT,
                format!("PLE row {r} past the table's {} rows", self.rows),
            ));
        }
        Ok(hist)
    }

    /// The rows [`PleHost::name_spanned`] named, decoded into the first
    /// positions of `e`.
    fn decode(&mut self) -> Result<(), GpuError> {
        let (ids, rb) = (&self.ids[..self.named], self.row_bytes);
        match &mut self.nvme {
            None => {
                let data = self
                    .file
                    .shard(self.shard)
                    .ok_or(GpuError::state(WHAT, "the PLE table's shard"))?
                    .data(&self.info)?;
                let rows = ids.iter().map(|&r| &data[r as usize * rb..][..rb]);
                ple_decode(&mut self.e, self.width, &self.decode, rows)?;
            }
            Some(read) => {
                read.finish()?;
                let filled = read.rows.filled();
                if filled.len() != ids.len() * rb {
                    return Err(GpuError::shape(
                        WHAT,
                        format!(
                            "the PLE reader copied {} B for {} rows of {rb} B",
                            filled.len(),
                            ids.len()
                        ),
                    ));
                }
                match (&self.decode, read.planes) {
                    (PleRows::Nvfp4(fp4), 2) => {
                        let (codes, scales) = filled.split_at(filled.len() / 9 * 8);
                        fp4.decode_planes(codes, scales, &mut self.e[..ids.len() * self.width])
                            .map_err(model::ModelError::from)?;
                    }
                    (_, 1) => ple_decode(
                        &mut self.e,
                        self.width,
                        &self.decode,
                        filled.chunks_exact(rb),
                    )?,
                    _ => return Err(GpuError::state(WHAT, "the PLE reader's storage planes")),
                }
            }
        }
        Ok(())
    }

    /// The named rows of the first `n` positions in an arena's `e`: decoded
    /// here and copied ([`PleHost::decode`], [`PleHost::upload`]), or, for
    /// NVFP4 planes read from the NVMe tier, refused here and copied as read
    /// to the arena's planes, which the card decodes into `e`
    /// ([`PleKernels::enqueue_rows_nvfp4`], the host decode's values).
    fn land(
        &mut self,
        stream: &CudaStream,
        k: &PleKernels,
        arena: &mut Ple38,
        n: usize,
    ) -> Result<(), GpuError> {
        let planes = match (&mut self.nvme, &self.decode) {
            (Some(read), PleRows::Nvfp4(fp4)) if read.planes == 2 => Some((read, fp4)),
            _ => None,
        };
        let Some((read, fp4)) = planes else {
            self.decode()?;
            return self.upload(stream, &mut arena.e, n);
        };
        read.finish()?;
        let filled = read.rows.filled();
        if filled.len() != self.named * self.row_bytes || self.named * self.width != n * geo::HIDDEN
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the PLE reader copied {} B for {} rows of {} B, {n} positions",
                    filled.len(),
                    self.named,
                    self.row_bytes
                ),
            ));
        }
        let groups = filled.len() / 9;
        k.enqueue_rows_nvfp4(
            stream,
            (fp4, filled),
            &mut arena.planes,
            groups,
            &mut arena.e,
        )
    }

    /// The decoded rows of the first `n` positions copied to an arena's
    /// `e`, its first `n · HIDDEN` values: one synchronous copy.
    fn upload(
        &self,
        stream: &CudaStream,
        e: &mut DeviceBuffer<f32>,
        n: usize,
    ) -> Result<(), GpuError> {
        let len = n * geo::HIDDEN;
        let mut w = WindowMut::<f32>::of_mut(e, 0, len)?;
        let src = self
            .e
            .get(..len)
            .ok_or_else(|| GpuError::shape(WHAT, format!("{n} PLE positions at once")))?;
        w.copy_from_host(stream, src)?;
        Ok(())
    }

    /// Read a Gemm call's ubatches ahead: `tokens` at `pos`, cut into the
    /// ubatches `cut` ([`call_units`], the cut the call runs across its
    /// checkpoints), each one's row ids computed now from a clone of the
    /// history walked as the fills will walk it. The first ubatch's fill
    /// reads its own rows, the call's one read no walk hides but its own
    /// layers before the PLE site on a walk that serves no host leg
    /// ([`Body38::stage_gemm`]); each fill then hands the helper the next
    /// ubatch's ([`PleRead::finish`]). On the host nothing is read ahead: the
    /// fills decode pages the load populated. A call whose rows
    /// the hash refuses reads nothing ahead, and its first fill refuses it
    /// by name. `spans` are the call's image spans as position ranges
    /// ([`MediaRows::positions`]); each ubatch hashes its own part of them
    /// ([`window_spans`]) as its fill will.
    fn prompt_start(
        &mut self,
        pos: u32,
        tokens: &[u32],
        cut: impl IntoIterator<Item = usize>,
        spans: &[Range<usize>],
    ) -> Result<(), GpuError> {
        let Some(read) = &mut self.nvme else {
            return Ok(());
        };
        read.drain()?;
        read.recycle_front();
        read.ahead.clear();
        let per_token = geo::HIDDEN / self.width;
        self.ahead_hist.clone_from(&self.hist);
        let mut at = 0usize;
        for m in cut {
            let Some(p) = u32::try_from(at).ok().and_then(|a| pos.checked_add(a)) else {
                break;
            };
            let mut ids = std::mem::take(&mut read.spare_ids);
            ids.resize(m * per_token, 0);
            let unit = window_spans(spans.iter().cloned(), &(p as usize..p as usize + m));
            if self
                .hash
                .ple_rows_media(
                    &mut self.ahead_hist,
                    u64::from(p),
                    &tokens[at..at + m],
                    &unit,
                    &mut ids,
                )
                .is_err()
            {
                read.ahead.clear();
                break;
            }
            read.ahead.push_back(PleBatch { pos: p, n: m, ids });
            at += m;
        }
        Ok(())
    }

    /// [`Body38::prefetch_input`]: `token` at `pos` read ahead as a prompt
    /// call of one position whose read starts at once.
    fn prefetch(&mut self, pos: u32, token: u32) -> Result<(), GpuError> {
        self.prompt_start(pos, &[token], [1], &[])?;
        if let Some(read) = &mut self.nvme {
            read.start_front()?;
        }
        Ok(())
    }

    /// The history a verify's `fill` moved past its rows, moved instead
    /// from `before` (the history as it stood before that fill) past the
    /// `kept` rows from position `pos`: the history a fill of those rows
    /// alone leaves.
    fn keep(&mut self, before: History, pos: u32, kept: &[u32]) -> Result<(), GpuError> {
        let per_token = geo::HIDDEN / self.width;
        let ids = self.ids.get_mut(..kept.len() * per_token).ok_or_else(|| {
            GpuError::shape(WHAT, format!("{} PLE positions at once", kept.len()))
        })?;
        let mut hist = before;
        self.named = 0;
        self.hash
            .ple_rows_into(&mut hist, u64::from(pos), kept, ids)
            .map_err(|e| GpuError::shape(WHAT, format!("PLE rows at position {pos}: {e}")))?;
        self.named = ids.len();
        self.hist = hist;
        Ok(())
    }

    /// A new sequence's history; on the NVMe tier, nothing read ahead.
    fn restart(&mut self) -> Result<(), GpuError> {
        if let Some(read) = &mut self.nvme {
            read.drain()?;
            read.ahead.clear();
        }
        self.hist = self
            .hash
            .new_history()
            .map_err(|e| GpuError::shape(WHAT, e.to_string()))?;
        Ok(())
    }
}

/// Refused by name, `what` the caller: the first of `tokens` the raw-id PLE
/// hash of `window` does not take (the image placeholder outside `spans`,
/// the call's image spans as row ranges of `tokens`, an id past the
/// vocabulary), with its index in the call, or a hash that is not raw-id.
/// The hash's own test ([`engram::hash::EosWindow::check_at`]), so a call's
/// check before its first launch and each fill refuse the same ids.
fn ple_takes(
    what: &'static str,
    window: &Window,
    tokens: &[u32],
    spans: &[Range<usize>],
) -> Result<(), GpuError> {
    let Window::Eos(w) = window else {
        return Err(GpuError::state(what, "a raw-id PLE hash"));
    };
    for (i, &t) in tokens.iter().enumerate() {
        w.check_at(t, spans.iter().any(|s| s.contains(&i)))
            .map_err(|e| GpuError::shape(what, format!("token {i}: {e}")))?;
    }
    Ok(())
}

/// A layer's store read back ([`Body38::stores_host`]): a delta layer's
/// state, conv ring and carry, or a selecting layer's K/V planes, its raw
/// indexer key ring and its pooled indexer keys. A side's plane is its f16
/// bits, or on a q8_0 cache its codes plane's 16-bit halves (each word's low
/// half first) then its scales plane: bit equality reads either, a row's
/// values only the f16.
#[derive(Clone, Debug, PartialEq)]
pub enum Store38Host {
    Rec {
        state: Vec<f32>,
        ring: Vec<f32>,
        carry: Vec<f32>,
    },
    Qsa {
        k: Vec<u16>,
        v: Vec<u16>,
        ring: Vec<u16>,
        pooled: Vec<u16>,
    },
}

impl Store38Host {
    /// Bit equality: the f32 values by their bits, so a NaN equals itself
    /// and `-0.0` differs from `0.0`.
    #[must_use]
    pub fn same_bits(&self, other: &Store38Host) -> bool {
        let bits = |a: &[f32], b: &[f32]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        };
        match (self, other) {
            (
                Store38Host::Rec {
                    state: s,
                    ring: r,
                    carry: c,
                },
                Store38Host::Rec {
                    state: s2,
                    ring: r2,
                    carry: c2,
                },
            ) => bits(s, s2) && bits(r, r2) && bits(c, c2),
            (Store38Host::Qsa { .. }, Store38Host::Qsa { .. }) => self == other,
            _ => false,
        }
    }

    /// A delta store's committed lane with the rows its carry holds for
    /// count `pos` taken in ([`carry_fold_host`]): the state the next step
    /// takes its token into, wherever the calls before it were cut. `None`
    /// for a selecting store.
    #[must_use]
    pub fn live_lane(&self, pos: u32) -> Option<Vec<f32>> {
        match self {
            Store38Host::Rec { state, carry, .. } => Some(carry_fold_host(carry, state, pos, GDN)),
            Store38Host::Qsa { .. } => None,
        }
    }
}

/// A layer's router at the step's row, read back
/// ([`Body38::route_taps`]).
#[derive(Clone, Debug)]
pub struct RouteTap {
    /// The routed experts' logits, the shared expert's gate left out.
    pub logits: Vec<f32>,
    /// The routed slots' expert ids, in slot order.
    pub ids: Vec<u32>,
}

/// A verify that waits for its commit: its first position, its tokens, and
/// the PLE history as it stood before them; `ple_due` while its PLE rows'
/// arrival is still to be delivered ([`Body38::deliver_ple`]).
struct Pending38 {
    pos0: u32,
    tokens: Vec<u32>,
    hist: History,
    ple_due: bool,
}

/// A call planned and not yet launched: its first position, its rows, the
/// PLE history past them, and a verify's tokens. [`Body38::launch`] takes
/// it; a call that fails first leaves it to be planned over.
struct Staged38 {
    pos: u32,
    rows: u32,
    hist: History,
    verify: Option<Vec<u32>>,
}

/// Where a gate plants a call's failure ([`Body38::plant_before_launch`],
/// [`Body38::plant_after_launch`]): once its plan and copies are made and
/// before its launch, or once its chain has run and its host legs been
/// served.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Plant {
    BeforeLaunch,
    AfterLaunch,
}

/// The planted failure at `at`, taken, as the call's error; nothing when
/// none is planted there.
fn planted(plant: &mut Option<Plant>, at: Plant) -> Result<(), GpuError> {
    if *plant == Some(at) {
        *plant = None;
        return Err(GpuError::state(
            WHAT,
            match at {
                Plant::BeforeLaunch => "the planted failure before the launch",
                Plant::AfterLaunch => "the planted failure after the launch",
            },
        ));
    }
    Ok(())
}

/// [`planted`] after an eager walk, which a capture also records: a capture
/// launches nothing, so an armed plant waits for a walk that does. The
/// stream is asked only while a plant is armed.
fn planted_eager(plant: &mut Option<Plant>, gpu: &Gpu) -> Result<(), GpuError> {
    if plant.is_some() && !crate::capturing(gpu.stream())? {
        planted(plant, Plant::AfterLaunch)?;
    }
    Ok(())
}

/// One step's host values: its token and position.
#[derive(Clone, Copy, Debug)]
pub struct DecodeInput38 {
    token: u32,
    pos: u32,
}

/// The stream rule's costs of one Qwen3.8 expert — the architecture fact the
/// expert stream reads beside the lane's priced rate: the host union's
/// tile-path cost a column for an expert of the file's common layer kind and
/// its weight-listing floor, and the card route's fixed and per-column cost
/// of one streamed expert. The union's costs are the constrained fit of the
/// Q4 prompt-union records (the per-layer `stat prompt lb` walls against
/// their slots and listed experts, the pickrule round's fit: the records
/// identify the per-column slope alone, the floor comes from the
/// weight-listing anchor, one expert's bytes at the host's read bandwidth);
/// the card route's are the q38seed card's decomposition
/// (`docs/cards/q38seed-ab.card`), which the same records' shadow tail
/// corroborates at the card's expert count. One row for every file of the
/// family: the IQ file's union takes no tile path in this tree, so its own
/// row waits on that path (the per-file override
/// [`Body38::set_xstream_costs`]); no Q4-derived column cost is applied to
/// it silently.
pub const XSTREAM_COSTS: Costs = Costs {
    host_us_per_col: 3.67,
    host_us_fixed: 19.95,
    card_us_fixed: 13.0,
    card_us_per_col: 0.16,
};

/// The card route's tail past a layer's last copy, in µs per card pick (a
/// routed pick the card serves before the unit's pick: the unit's picks less
/// the inherited host experts' columns) — the family cost the pick's walk
/// balances against the union, for the Q4 kind. The tail is a level per unit
/// width, not a constant, so the walk prices each unit with it
/// ([`runtime::xsplit::unit_constants`]); the walk alone reads it, the ring's
/// rule never does. The rate sits at or under the level the Q4 prompt-union
/// records give at the largest prompts' unit.
///
/// Sensitivity [derived]: the walk's balance at 4096 columns has a knife edge
/// in this rate. Its low end loses 6 % of the admits 2 % over the last flat
/// value and half of them 4 % further; on the count wired here, before the
/// pick, the edge sits 12 % above the row.
pub const XSTREAM_TAIL_US_PER_CARD_COL: f64 = 0.125;

/// The stream rule's costs for a file whose common routed kind no record has
/// fitted — every kind but the Q4 one — until a record of that kind's host
/// union exists (the per-file override [`Body38::set_xstream_costs`] takes a
/// calibrated row then). The IQ kinds of the `UD-Q3_K_XL` file (IQ3_XXS, IQ4_XS) run a
/// multi-column tile on the host union (`qdot::has_tile`), so this row's
/// per-column cost does not describe their union; the record that closes it
/// is a prompt-union record of that file at 512 and 4096 columns (the
/// per-layer `stat prompt lb` walls against their slots and listed experts).
/// No Q4-derived tail is wired beside it: a walk on this row balances
/// without one.
pub const XSTREAM_COSTS_UNFITTED: Costs = Costs {
    host_us_per_col: 7.0,
    host_us_fixed: 23.36,
    card_us_fixed: 13.0,
    card_us_per_col: 0.16,
};

/// The file's common routed kind: the gate·up type most of its layers carry —
/// the kind whose host union the expert stream's costs row prices. Ties sit
/// with the Q4 kind, the family's own.
fn common_routed(stacks: &swap38::Qwen38Stacks) -> GgmlType {
    let kinds = [
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::IQ3_XXS,
        GgmlType::IQ4_XS,
    ];
    let count = |want: GgmlType| {
        (0..)
            .map_while(|l| stacks.pair(l))
            .filter(|&(gate, _)| gate == want)
            .count()
    };
    let mut best = (GgmlType::Q4_K, 0usize);
    for ty in kinds {
        let n = count(ty);
        if n > best.1 {
            best = (ty, n);
        }
    }
    best.0
}

/// Everything Qwen3.8's chain owns. Field order is drop order: the host tier
/// (its lock over the host set, its page windows) before the buffers.
pub struct Body38 {
    hybrid: Hybrid<HostRun>,
    plans: Vec<Layer38>,
    stores: Vec<Store38>,
    /// Every delta store's state, stamps and carry, for the commit's one
    /// replay launch ([`Body38::commit`]); it moves with `stores` between
    /// sequences ([`Body38::exchange`]).
    recovery: RecoveryTable,
    ple_ring: DeviceBuffer<f32>,
    ple: PleHost,
    rope: RopeRows,
    /// The live sequence's rotation table on a media load: the rope launches
    /// of the sequence read it in place of `rope` ([`rope_table`]). `None`
    /// on a text load ([`GpuModel::enable_media`]).
    seq_rope: Option<SeqRope>,
    /// What a media load makes each further sequence's table from
    /// ([`Slots::new_seq`]); `None` on a text load.
    rope_base: Option<RopeBase>,
    /// The media call's rows on their way into the ubatch walk's first layer;
    /// `Some` exactly when `seq_rope` is.
    media: Option<MediaRows>,
    /// The file's IMROPE sections and `rope.section_order`, read at load:
    /// `None` when no attention layer rotates by sections, which a media
    /// load refuses by name.
    imrope: Option<([u32; 4], vision::mrope::SectionOrder)>,
    /// The step's one-row arena and record; an eager pass's arena, record
    /// and host sums.
    s: Arena38,
    sp: StepParams,
    a: Arena38,
    rp: PassRecord<PASS_ROWS>,
    pass_hsum: DeviceBuffer<f32>,
    k: Kernels38,
    /// The stream every walk's shared expert runs on beside its card leg.
    branch: Branch,
    /// The slot map's card copy, and the card leg over its card experts. The
    /// host tier holds the host copy, and its residency machine, when it runs
    /// one, this `Arc`: the buffer lives until both let go.
    slots: Arc<DeviceTensor<u32>>,
    card: Card38,
    /// The stage card's side of the expert tier, on a load that hung one.
    tier: Option<TierSide38>,
    /// A ubatch walk's arena, record, host sums and own buffers, for
    /// ubatches of up to `wide.rows` positions.
    wa: Arena38,
    wr: WideRecord,
    wide_hsum: DeviceBuffer<f32>,
    wide: Wide38,
    /// The ubatch walk's timing, when a caller armed it
    /// ([`GpuModel::set_prompt38_stats`]); unarmed, the walk records and
    /// waits for nothing.
    wide_timing: Option<PromptTiming>,
    /// The prompt call's host streaming (`BLOOMERY_XSTREAM`): off until a
    /// caller sets it ([`Body38::set_xstream`]).
    stream: PromptStream,
    /// The map check every walk reads ([`MapCheck::refuse`]): the tier's
    /// slot map's, or the one a gate planted in its place until it is taken
    /// back or the model reset.
    card_leg: MapCheck,
    /// The lane word every delta launch reads, and the verify waiting for
    /// its commit.
    lane: LaneWord,
    pending: Option<Pending38>,
    /// How a verify's PLE rows reach its walk after its launch: one staged
    /// row of [`VERIFY_ROWS`] positions' values.
    ple_arrival: StagedRows<f32>,
    /// Each layer's streams after it and its route, when a gate armed them.
    taps: Option<Taps38>,
    /// The launches the loaded `output`'s head records past its norm: the
    /// captured walks' launch count's head term ([`step_launches`]).
    head: HeadLaunches,
    eps: f32,
    vocab: usize,
    ctx: usize,
    /// Positions the stores hold: a call's count once it is launched
    /// ([`Body38::launch`]). The next position is the model's
    /// (`GpuModel::pos`); the two differ only after a call failed past its
    /// launch, and a call at the model's position is then refused.
    held: u32,
    /// The call planned and not yet launched.
    staged: Option<Staged38>,
    /// The staged ubatch's PLE rows are named and on their read, not yet
    /// decoded into its arena: its walk copies them in at the PLE site
    /// ([`Body38::stage_gemm`]).
    ple_late: bool,
    /// The sequence slots the load's plan was made for ([`Slots`]), and how
    /// many stand: the load's own sequence plus every [`Slots::new_seq`]. A
    /// sequence past the plan is refused by name.
    slots_planned: usize,
    slots_made: usize,
    /// A pass of several slots' input records, on a load planned for more
    /// than one sequence.
    slot_in: Option<SlotIn>,
    /// The arena the last planned call walked (a step's, a pass's or a
    /// verify's the step arena or the pass arena, a ubatch's its own): where
    /// its rows' final streams sit for a caller that reads them back
    /// ([`Body38::target_streams`], [`Tapped`]).
    last_walk: TargetRows,
    /// Each target arena's [`Held`]: where the step, the pass (a verify's
    /// rows too) and the ubatch walk left their rows, which a draft's walk
    /// reads by position ([`mtp_target`]) and a sequence state carries.
    wrote: Wrote38,
    /// A failure a gate planted for the next call.
    plant: Option<Plant>,
    /// The MTP draft layer, when the load opened one ([`Body38::open_placed_mtp`]).
    mtp: Option<Mtp38>,
    /// The load's residency side ([`ResidencyGlue`]): the machine, when the
    /// lever runs one, over `hybrid`'s slot map through `slots`; every call
    /// nothing without one.
    residency_glue: ResidencyGlue,
    /// The NVMe expert tier's arena, when the plan paged routed experts
    /// ([`Body38::nvme_tier`]).
    nvme_tier: Option<Arc<NvTier>>,
    /// The file's common routed kind ([`common_routed`]): which expert
    /// stream's costs row and card tail its walks take.
    routed_gate: GgmlType,
    /// The recurrent stores on the host at chosen positions, and the PLE
    /// hash's history at each, ascending: the points a cut behind the fed
    /// positions restores. A prompt call takes them only while `marks` is on.
    ckpt: Checkpoints,
    ple_points: Vec<(u32, History)>,
    /// On a load with the draft, the target's final hidden row at the
    /// position before each point whose row an arena still held when the
    /// point was taken, ascending: what the draft's first walk after a cut
    /// to the point reads ([`Body38::draft_rejoins`]).
    point_rows: Vec<(u32, Vec<f32>)>,
    marks: bool,
    /// Whether a prompt call closes its prompt ([`Body38::set_close`]).
    close: bool,
    /// Whether the ubatch walk being planned is a closing call's last, whose
    /// delta layers keep their last chunk out of the state
    /// (`DeltaKernels::enqueue_delta_prompt`'s `keep`): the end mark one row
    /// short of the walk's end is then the lane and the carry less that row.
    keep_last: bool,
    /// A prompt call's checkpoint at the end of a ubatch, opened before the
    /// ubatch walks ([`Body38::open_end_mark`]) and sealed at the mark
    /// ([`Body38::take_mark`]).
    end_mark: Option<checkpoint::Pending>,
    /// Where an open end mark's copies run, made at its first use.
    marker: Option<EndMarker>,
    /// What a sequence state of this load belongs to.
    who: Identity,
}

/// The model a Qwen3.8 load reads ([`Body38::open_placed`] and its siblings): a GGUF split, or a
/// checkpoint folder ([`Folder38`]). A folder carries the PLE table's rows beside its parts.
pub type Model38 = Model<Option<StoredRows>>;

/// A checkpoint folder as a Qwen3.8 load reads it: its header (`gguf::Split::located`, the
/// header its plan was made from), its card tensors through its part source, and the PLE
/// table's rows where the folder stores them, which the table's NVMe reader maps
/// ([`StoredRows`]). The host set and the residency machine read the routed stacks the header
/// locates where the folder stores them ([`crate::parts::OnCard`]); a folder whose header
/// locates none holds every tensor on its one card and the PLE table on the NVMe tier, and
/// runs no residency machine. The part source also reads the folder's MTP tensors when a draft
/// is planned. Each part's pages leave the cache once it is
/// staged under the host config's `card_dontneed`, as a split's card segments do.
pub type Folder38 = Folder<Option<StoredRows>>;

impl Body38 {
    /// Card `card` of `plan`, which `inputs` made, resident: the plan's
    /// segments, the joins ([`Body38::derive`]) and the body over them with
    /// the host tier over every layer's routed experts, holding the load's
    /// host set as `host` asks, its ubatches of up to `ubatch` positions —
    /// the value the plan's machine was built with (`place::machine`) — and
    /// the plan's expert tier card, when it names one, hung under the host
    /// tier. The load's slot map for the model's life ([`Residency::Off`]);
    /// refused by name: a coverage item past [`ALLOWED`], more expert tier
    /// cards than the host tier serves, a tier layer with no card expert, a
    /// tier whose prompt-batch bytes are not the plan's reserves
    /// (`HostTier::check_tier_reserves`), a plan of more than one stage card
    /// or not every layer, a
    /// layer or a width the kernels do not take, a `ubatch` outside
    /// `1..=min(UBATCH, ctx)`, a ubatch arena past what the plan's
    /// scratch and reserves count, and a stage card reserve for the card
    /// route's run or the activations' MMA forms that the ubatch does not
    /// read (`place::card_route_run`, `place::act_forms_bytes`). The card
    /// route runs the whole ubatch at once on a card that reserves that run
    /// (`place::widen_card_route`), else in runs of `ROUTE_ROWS` tokens. On a
    /// card that reserves the activations' MMA forms
    /// (`place::machine_for_experts`) the model-width rows' BF16 sites read
    /// the rows rounded once and the MXFP8 sites their rows' MXFP8 form
    /// ([`WideForms`]), else each tile converts the f32 rows.
    /// `file` is a GGUF split or a checkpoint folder ([`Model38`]); a
    /// folder whose header locates no file is refused by name under a
    /// residency machine ([`Model::into_origin`]).
    pub fn open_placed(
        file: impl Into<Model38>,
        plan: &Plan<'_>,
        inputs: &model::arch::qwen35moe::place::PlanInputs,
        card: usize,
        host: HostCfg,
        ubatch: usize,
    ) -> Result<Qwen38Model, GpuError> {
        Body38::open_with(
            file,
            plan,
            inputs,
            card,
            host,
            ubatch,
            Residency::Off,
            ChurnHosting::Hosted,
            None,
            1,
        )
    }

    /// [`Body38::open_placed`] for a load that serves `slots` resident
    /// sequences ([`Slots`]): the plan must have counted them
    /// ([`PlanInputs::plan_with_slots`](model::arch::qwen35moe::place::PlanInputs)),
    /// and a sequence past the count is refused by name. Zero slots is
    /// refused by name.
    pub fn open_placed_slots(
        file: impl Into<Model38>,
        plan: &Plan<'_>,
        inputs: &model::arch::qwen35moe::place::PlanInputs,
        card: usize,
        host: HostCfg,
        ubatch: usize,
        slots: usize,
    ) -> Result<Qwen38Model, GpuError> {
        Body38::open_with(
            file,
            plan,
            inputs,
            card,
            host,
            ubatch,
            Residency::Off,
            ChurnHosting::Hosted,
            None,
            slots,
        )
    }

    /// [`Body38::open_placed`] under `residency`: `mid-p<P>-s<S>` runs the
    /// residency machine over the card's routed stacks
    /// ([`crate::host::swap`], [`swap38::Qwen38Stacks`]) — under
    /// [`ChurnHosting::Hosted`] the load's host set also holds each layer's
    /// churn pool (the card's experts past the first `P`), refused by name
    /// when the plan's host headroom cannot take it; under
    /// [`ChurnHosting::Unhosted`] it only knows the pool, whose victims are
    /// read back from the mapping — while `off` is [`Body38::open_placed`]
    /// itself. The lever's word a load runs by and its pool's hosting are the
    /// caller's, from the registry and the word's resolution
    /// (`bloomery_levers::Levers::residency_at`, `residency38_at_plan`).
    /// `slots` is [`Body38::open_placed_slots`]'s: the plan must have
    /// counted them.
    #[allow(
        clippy::too_many_arguments,
        reason = "open_placed's six, the residency and its pool's hosting, and the planned slots (rust-quality R8)"
    )]
    pub fn open_placed_residency(
        file: impl Into<Model38>,
        plan: &Plan<'_>,
        inputs: &model::arch::qwen35moe::place::PlanInputs,
        card: usize,
        host: HostCfg,
        ubatch: usize,
        residency: Residency,
        pool: ChurnHosting,
        slots: usize,
    ) -> Result<Qwen38Model, GpuError> {
        Body38::open_with(
            file, plan, inputs, card, host, ubatch, residency, pool, None, slots,
        )
    }

    /// [`Body38::open_placed`] of `plan`'s target plan with the MTP draft
    /// of `draft` (the draft file `mtp` was read from) opened on the same
    /// card beside it ([`Mtp38::open`]); refused as either refuses. The
    /// load's slot map for the model's life ([`Residency::Off`]).
    #[allow(
        clippy::too_many_arguments,
        reason = "open_placed's six and the draft's file and inputs (rust-quality R8)"
    )]
    pub fn open_placed_mtp(
        file: impl Into<Model38>,
        plan: &model::arch::qwen35moe::place::MtpPlan<'_>,
        inputs: &model::arch::qwen35moe::place::PlanInputs,
        card: usize,
        host: HostCfg,
        ubatch: usize,
        draft: &Split,
        mtp: &model::arch::qwen35moe::place::MtpInputs,
    ) -> Result<Qwen38Model, GpuError> {
        Body38::open_with(
            file,
            &plan.plan,
            inputs,
            card,
            host,
            ubatch,
            Residency::Off,
            ChurnHosting::Hosted,
            Some((draft, mtp, plan)),
            1,
        )
    }

    /// [`Body38::open_placed_mtp`] for a load that serves `slots` resident
    /// sequences ([`Body38::open_placed_slots`]): the plan must have counted
    /// them
    /// ([`PlanInputs::plan_mtp_with_slots`](model::arch::qwen35moe::place::PlanInputs)).
    #[allow(
        clippy::too_many_arguments,
        reason = "open_placed_mtp's eight and the slot count (rust-quality R8)"
    )]
    pub fn open_placed_mtp_slots(
        file: impl Into<Model38>,
        plan: &model::arch::qwen35moe::place::MtpPlan<'_>,
        inputs: &model::arch::qwen35moe::place::PlanInputs,
        card: usize,
        host: HostCfg,
        ubatch: usize,
        draft: &Split,
        mtp: &model::arch::qwen35moe::place::MtpInputs,
        slots: usize,
    ) -> Result<Qwen38Model, GpuError> {
        Body38::open_with(
            file,
            &plan.plan,
            inputs,
            card,
            host,
            ubatch,
            Residency::Off,
            ChurnHosting::Hosted,
            Some((draft, mtp, plan)),
            slots,
        )
    }

    /// [`Body38::open_placed_mtp`] under `residency`, as
    /// [`Body38::open_placed_residency`] is [`Body38::open_placed`], for a
    /// load that serves `slots` resident sequences
    /// ([`Body38::open_placed_mtp_slots`]): the plan must have counted them.
    #[allow(
        clippy::too_many_arguments,
        reason = "open_placed_mtp's eight, the residency and its pool's hosting, and the planned slots (rust-quality R8)"
    )]
    pub fn open_placed_mtp_residency(
        file: impl Into<Model38>,
        plan: &model::arch::qwen35moe::place::MtpPlan<'_>,
        inputs: &model::arch::qwen35moe::place::PlanInputs,
        card: usize,
        host: HostCfg,
        ubatch: usize,
        draft: &Split,
        mtp: &model::arch::qwen35moe::place::MtpInputs,
        residency: Residency,
        pool: ChurnHosting,
        slots: usize,
    ) -> Result<Qwen38Model, GpuError> {
        Body38::open_with(
            file,
            &plan.plan,
            inputs,
            card,
            host,
            ubatch,
            residency,
            pool,
            Some((draft, mtp, plan)),
            slots,
        )
    }

    /// The load every constructor shares, under `residency` with its churn
    /// pool hosted or only known as `pool` says (the word's resolution's
    /// decision, [`ResidencySpec::pool`]); `mtp` the draft's file, inputs and
    /// plan, when there is one, and `slots` the sequence slots the plan was
    /// made for ([`Slots`]).
    #[allow(
        clippy::too_many_arguments,
        reason = "open_placed's six, the residency and its pool's hosting, the draft's file, inputs and plan, and the planned slots (rust-quality R8)"
    )]
    fn open_with(
        file: impl Into<Model38>,
        plan: &Plan<'_>,
        inputs: &model::arch::qwen35moe::place::PlanInputs,
        card: usize,
        host: HostCfg,
        ubatch: usize,
        residency: Residency,
        pool: ChurnHosting,
        mtp: Option<(
            &Split,
            &model::arch::qwen35moe::place::MtpInputs,
            &model::arch::qwen35moe::place::MtpPlan<'_>,
        )>,
        slots: usize,
    ) -> Result<Qwen38Model, GpuError> {
        if slots == 0 {
            return Err(GpuError::shape(WHAT, "a load of 0 resident sequence slots"));
        }
        let (file, origin) = file.into().into_origin(residency)?;
        // A draft in the folder's own MTP layer reads its tensors through the folder's parts.
        let draft_parts = match origin.bytes() {
            CardBytes::Parts(parts) => mtp
                .filter(|(draft, _, _)| draft.shard_path(0) == file.shard_path(0))
                .map(|_| parts),
            CardBytes::File => None,
        };
        let bytes = origin.bytes();
        let stored = origin.extra().and_then(Option::as_ref);
        refuse_tier_count(WHAT, plan.machine.expert_tiers().count())?;
        let refused: Vec<String> = inputs
            .unimplemented()
            .into_iter()
            .map(|u| match u.layer {
                Some(l) => format!("{} (layer {l})", u.feature),
                None => u.feature,
            })
            .filter(|f| !ALLOWED.contains(&f.as_str()))
            .collect();
        if !refused.is_empty() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the file needs what the program does not run: {}",
                    refused.join("; ")
                ),
            ));
        }
        let n = inputs.spec.layers.len();
        let spec = plan
            .machine
            .cards
            .get(card)
            .ok_or_else(|| GpuError::shape(WHAT, format!("the plan has no card {card}")))?;
        let layers = spec.layers.clone();
        // The card route's run and the arena bytes the plan counts with it,
        // in the form this build holds the file's route in.
        let (route_run, counted) = model::arch::qwen35moe::place::card_route_run(
            spec,
            ubatch as u64,
            !plan.machine.tiers.is_empty(),
            inputs.route_form(crate::NATIVE_FP4),
        )
        .map_err(|e| GpuError::shape(WHAT, e.to_string()))?;
        let route_run = usize::try_from(route_run)
            .map_err(|_| GpuError::shape(WHAT, "the card route's run passes usize"))?;
        if plan.machine.cards.len() != 1 || layers != (0..n) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "card {card} runs layers {layers:?} of a plan over {} cards; the program runs \
                     every layer (0..{n}) on one card",
                    plan.machine.cards.len()
                ),
            ));
        }
        let shape = plan38::read(&inputs.spec)?;
        let spec = ResidencySpec {
            lever: residency,
            pool,
            delay: LIVE_DELAY,
            deadline: DEADLINE,
            top_k: geo::N_USED,
            max_rows: PASS_ROWS,
            stacks: Arc::new(swap38::stacks(&inputs.model, plan)?),
        };
        GpuModel::load_placed_from(
            file,
            bytes,
            plan,
            card,
            host,
            spec,
            (&[], 0),
            |stream, _, layers, w| Body38::derive(stream, &shape.kinds, layers, w),
            |gpu, file, w, set, glue| {
                let draft = mtp
                    .map(|(d, m, p)| {
                        Mtp38::open(gpu, w, (d, draft_parts), m, p, host.card_dontneed, slots)
                    })
                    .transpose()?;
                let mut body = Body38::load_placed(
                    gpu,
                    (file, stored),
                    w,
                    plan,
                    inputs,
                    &shape,
                    host,
                    set,
                    glue,
                    (ubatch, route_run, counted, slots),
                )?;
                if let Some(mut d) = draft {
                    let rows = body.rope.table.len() / body.rope.width;
                    d.arm(
                        gpu.stream(),
                        body.s.route.dims(),
                        body.vocab,
                        rows,
                        mtp.map(|(_, _, p)| p.arena_bytes).unwrap_or_default(),
                    )?;
                    body.mtp = Some(d);
                }
                Ok(body)
            },
        )
    }

    /// The joins: each delta layer's β and α into one F32 stack (β's rows
    /// first), each router with the shared expert's gate as row 512. Each
    /// row keeps its file bits.
    fn derive(
        stream: &CudaStream,
        kinds: &[Kind38],
        layers: Range<usize>,
        w: &mut Weights,
    ) -> Result<(), GpuError> {
        for l in layers {
            let kind = kinds
                .get(l)
                .ok_or_else(|| GpuError::shape(WHAT, format!("layer {l} past the description")))?;
            if *kind == Kind38::Gdn {
                let (b, a) = (
                    model::arch::qwen35moe::names::ssm_beta(l),
                    model::arch::qwen35moe::names::ssm_alpha(l),
                );
                w.join_rows(stream, &[&b, &a], beta_alpha(l))?;
            }
            let (r, sh) = (
                model::arch::qwen35moe::names::ffn_gate_inp(l),
                model::arch::qwen35moe::names::ffn_gate_inp_shexp(l),
            );
            w.join_rows(stream, &[&r, &sh], router(l))?;
        }
        Ok(())
    }

    /// The body of the plan's one card over the chain's `shape` (module doc).
    /// `set` is the load's host set and `residency_glue` its residency side:
    /// the machine, when the lever runs one, is started once the pieces are
    /// sized ([`HostServed::start_residency`], the load's last step).
    #[allow(
        clippy::too_many_arguments,
        reason = "the load's card handle, file with its stored PLE rows, weights, plan, inputs and chain shape, the host tier's residency and levers, the residency glue, and the ubatch with the card route's run, the arena bytes the plan counts and the planned slots (rust-quality R8)"
    )]
    fn load_placed(
        gpu: &Gpu,
        (file, stored): (&Arc<Split>, Option<&StoredRows>),
        w: &Weights,
        plan: &Plan<'_>,
        inputs: &model::arch::qwen35moe::place::PlanInputs,
        shape: &Shape38,
        host: HostCfg,
        set: HostResidency,
        residency_glue: ResidencyGlue,
        (ub, route_run, counted, planned): (usize, usize, u64, usize),
    ) -> Result<Body38, GpuError> {
        let spec = &inputs.spec;
        let n = spec.layers.len();
        let ctx = usize::try_from(plan.ctx_max)
            .ok()
            .filter(|&c| c > 0)
            .ok_or_else(|| GpuError::shape(WHAT, format!("ctx_max {}", plan.ctx_max)))?;
        if ub == 0 || ub > UBATCH_MOST.min(ctx) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a ubatch of {ub} positions: the load takes 1..={} at a cache of {ctx}",
                    UBATCH_MOST.min(ctx)
                ),
            ));
        }
        let Shape38 {
            kinds,
            router: router_dims,
            base,
            ple: ple_layer,
        } = shape;
        let (router_dims, base) = (*router_dims, *base);
        let plans = plan38::plans(w, kinds, *ple_layer)?;
        gpu.context().bind_to_thread()?;
        let stream = gpu.stream();
        let d = dims(router_dims, ctx);
        // The plan's cache format is the one the stores take.
        let kv = KvQ8::of(inputs.kv.is_q8());
        let handle = gpu.stream_handle();
        let stores = kinds
            .iter()
            .map(|k| match k {
                Kind38::Gdn => Store38::rec(stream),
                Kind38::Qsa => Store38::qsa(&handle, &d, kv),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (rec, sel) = store_rule_bytes(ctx, kv);
        for (l, (st, k)) in stores.iter().zip(kinds).enumerate() {
            let want = match k {
                Kind38::Gdn => rec,
                Kind38::Qsa => sel,
            };
            if st.bytes() as u64 != want {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {l}'s store holds {} bytes, runtime::stores counts {want}",
                        st.bytes()
                    ),
                ));
            }
        }
        let ple_ring = DeviceBuffer::zeroed(stream, ple::ring_len(geo::STREAMS))?;
        let rope = RopeTable::new(&RopeSpec::window(base, geo::ROPE))?;
        let rope = RopeRows::new(stream, &rope, geo::ROPE, ctx)?;
        // The sites that read q8_1 rows (the file's K-quants) project by name
        // through forms of their own: one pair of by-name states a walk's
        // arena, the branch's beside the main stream's.
        let by_sites = plan38::q8_1_sites(w, &plans)?;
        let by_any = !by_sites.is_empty();
        let by_state = |cols: usize| ByName::for_sites(gpu, w, by_sites.iter().copied(), cols);
        let by_none = || ByName::for_sites(gpu, w, std::iter::empty::<&str>(), 1);
        let s = Arena38::new(stream, router_dims, 1, ctx, (by_state(1)?, by_state(1)?))?;
        let sp = StepParams::new(stream, true)?;
        let a = Arena38::new(
            stream,
            router_dims,
            PASS_ROWS,
            ctx,
            (by_state(PASS_ROWS)?, by_state(PASS_ROWS)?),
        )?;
        let rp = PassRecord::new(stream)?;
        let pass_hsum = DeviceBuffer::zeroed(stream, PASS_ROWS * geo::HIDDEN)?;
        // The ubatch walk projects a K-quant site by chunks of the by-name
        // unit's columns on the main stream alone.
        let wa = Arena38::wide(
            stream,
            router_dims,
            (ub, UBATCH_MOST, final_res(&plans)),
            ctx,
            (by_state(BY_NAME_COLS.min(ub))?, by_none()?),
        )?;
        let wr = WideRecord::new(stream, ub)?;
        let wide_hsum = DeviceBuffer::zeroed(stream, ub * geo::HIDDEN)?;
        let host_cols = usize::try_from(runtime::stores::reader_columns(ub as u64))
            .map_err(|_| GpuError::shape(WHAT, "the reader's columns pass usize"))?;
        let run = 0..n;
        let tiers = TierOpen::of_machine(plan.machine)?;
        let tier_cards: Vec<usize> = tiers.iter().map(|t| t.card).collect();
        let map = SlotMap::of_plan_tiers(plan, 0, &tier_cards, run.clone(), geo::EXPERTS)?;
        let card_leg = MapCheck::of(&map, !tiers.is_empty())?;
        let slots = Arc::new(DeviceTensor::upload(
            stream,
            &map.stage_view(),
            n,
            geo::EXPERTS,
        )?);
        let stacks = swap38::stacks(&inputs.model, plan)?;
        // A residency machine moves experts to the host after the load, and
        // the host service writes a route trace: either keeps every host leg.
        let machine = residency_glue.source().is_some();
        let card = Card38::new(gpu, w, &stacks, &map, n, machine || host.route_trace)?;
        // A tier layer's join sums the card's slots and the tier's in one
        // launch, which a layer with no card expert does not run.
        if let Some(l) = run
            .clone()
            .find(|&l| map.on_tier(l).is_ok_and(|k| k > 0) && !card.has(l))
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l} holds {} routed experts on the tier card and none on the stage \
                     card: a tier layer's join sums both cards' slots",
                    map.on_tier(l)?
                ),
            ));
        }
        // A load planned for several sequences runs a pass of their rows
        // through the step port's columns, up to the pass arena's rows; a
        // load for one sequence the verify's. The expert tier's page, its
        // places and its card leg are cut to the same columns.
        let cols = if planned > 1 { PASS_ROWS } else { VERIFY_ROWS };
        let tier = (!tiers.is_empty())
            .then(|| TierSide38::new(gpu, &map, cols))
            .transpose()?;
        // The activations' MMA forms take the stage card's reserve for them.
        let forms = plan.machine.cards.first().map_or(Ok(0), |c| {
            model::arch::qwen35moe::place::act_forms_bytes(c, ub as u64)
                .map_err(|e| GpuError::shape(WHAT, e.to_string()))
        })?;
        let wide_forms = WideForms::of(gpu, w, &plans, forms > 0)?;
        let mut wide = Wide38::new(stream, ub, route_run, &card, tier.is_some(), &wide_forms)?;
        let wide_bytes = (wa.bytes() + wr.bytes() + wide_hsum.num_bytes() + wide.bytes()) as u64;
        let counted = counted + forms;
        if wide_bytes > counted {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a ubatch of {ub} holds {wide_bytes} card bytes, the plan counts {counted}"
                ),
            ));
        }
        let boundary = Boundary::with_cols(
            gpu.context(),
            stream,
            BoundaryShape {
                hidden: geo::HIDDEN,
                n_used: geo::N_USED,
            },
            1,
            cols,
        )?;
        let widths = HostWidths {
            embd: geo::HIDDEN,
            ff: geo::FF,
            n_used: geo::N_USED,
        };
        let hosted: Vec<usize> =
            HostRun::viewed(run, machine, (|l| card.n_card(l), geo::EXPERTS)).collect();
        let mut experts = crate::model::host_run(
            gpu.context(),
            plan,
            file,
            hosted.iter().copied(),
            host.r8,
            set.arena(),
            widths,
            |src| model::arch::qwen35moe::host::layers_of_spec(src, spec, hosted.iter().copied()),
        )?;
        residency_glue.attach_tier(experts.tier())?;
        let nvme_tier = experts.tier().cloned();
        if nvme_tier.is_some() {
            wide.paged();
        }
        experts.prepare_union(host_cols)?;
        // The plan reserves the tier's block scratch for blocks of the batch
        // port's columns: the block route's bytes there.
        let block = usize::try_from(model::arch::qwen35moe::place::tier_route_scratch_bytes(
            host_cols as u64,
        ))
        .map(|b| (host_cols, b))
        .map_err(|_| GpuError::shape(WHAT, "the tier's block scratch passes usize"))?;
        let tier_cards = tiers
            .iter()
            .enumerate()
            .map(|(i, t)| {
                open_tier(
                    gpu,
                    file,
                    plan,
                    (t, i),
                    &map,
                    n,
                    cols,
                    block,
                    host.card_dontneed,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let port_cols = boundary.layout().cols();
        if let Some(t) = tier_cards.iter().find(|t| t.cols() != port_cols) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the expert tier {} opened for {} columns a row, the step port's {port_cols}: \
                     a pass hands the tier one row of the port's columns",
                    t.name(),
                    t.cols()
                ),
            ));
        }
        let mut hybrid = Hybrid::new(boundary, map, experts, n)?;
        // The host service writes a route trace, so the tier refuses one on a
        // load that runs a layer without it.
        if let Some(l) = (0..n).find(|&l| card.hostless(l)) {
            hybrid.mark_unserved(l);
        }
        hybrid.watch_fault(gpu.fault_word())?;
        hybrid.keep_residency(set);
        let tiered = !tier_cards.is_empty();
        if tiered {
            hybrid.attach_tiers(tier_cards, gpu)?;
        }
        // With a tier the batch port's tier legs are made here too, which the
        // plan reserves on the tier card and the host.
        hybrid.prepare_batch(gpu.context(), host_cols)?;
        if tiered {
            let cards: Vec<usize> = tiers.iter().map(|t| t.card).collect();
            hybrid.check_tier_reserves(plan, &cards)?;
        }
        // The PLE table's tier is the plan's: the host set built above holds
        // it on the host; on the NVMe tier its reader reads it.
        let ple_tier = plan
            .row_tier()
            .map_err(|e| GpuError::plan(WHAT, e))?
            .ok_or(GpuError::state(WHAT, "a plan that places the PLE table"))?;
        let ple = PleHost::new(
            Arc::clone(file),
            stored,
            spec.vocab as usize,
            host_cols,
            ple_tier,
        )?;
        let ckpt = Checkpoints::sequence(
            gpu.context(),
            checkpoint_lens_of(&stores),
            ctx,
            CHECKPOINT_EVERY,
        )?;
        let who = Identity::of_load("qwen4exp", file, gpu, ctx)?;
        // An IMROPE file's image rows read their sections in the file's
        // `rope.section_order`, a key the description does not carry.
        let imrope = imrope_sections(spec)
            .map(|s| {
                let order = model::arch::qwen35moe::hparams::Hparams::read(file)
                    .map_err(|e| GpuError::plan(WHAT, e))?
                    .rope_order;
                Ok::<_, GpuError>((s, section_order(order)))
            })
            .transpose()?;
        // The head's launches in a captured walk, from the head's own rule
        // for the `output` it reads; a type no head entry reads is refused by
        // name.
        let head = HeadLaunches::of(w)?;
        let recovery = recovery_table(stream, &stores)?;
        let mut body = Body38 {
            hybrid,
            plans,
            stores,
            recovery,
            ple_ring,
            ple,
            rope,
            seq_rope: None,
            rope_base: None,
            media: None,
            imrope,
            s,
            sp,
            a,
            rp,
            pass_hsum,
            k: Kernels38::load(gpu, by_any)?,
            branch: Branch::new(gpu.stream_handle())?,
            slots,
            card,
            tier,
            wa,
            wr,
            wide_hsum,
            wide,
            wide_timing: None,
            stream: PromptStream::default(),
            card_leg,
            lane: LaneWord::new(stream, LANES)?,
            pending: None,
            ple_arrival: StagedRows::new(
                gpu.context(),
                1,
                VERIFY_ROWS * geo::HIDDEN,
                "cuMemAllocHost (a verify's PLE rows staging)",
            )?,
            taps: None,
            eps: spec.rms_eps,
            vocab: spec.vocab as usize,
            head,
            ctx,
            held: 0,
            staged: None,
            ple_late: false,
            slots_planned: planned,
            slots_made: 1,
            slot_in: (planned > 1).then(|| SlotIn::new(stream)).transpose()?,
            last_walk: TargetRows::Step,
            wrote: Wrote38::default(),
            plant: None,
            mtp: None,
            residency_glue,
            nvme_tier,
            routed_gate: common_routed(&stacks),
            ckpt,
            ple_points: Vec::new(),
            point_rows: Vec::new(),
            marks: false,
            close: false,
            keep_last: false,
            end_mark: None,
            marker: None,
            who,
        };
        body.sp.write(stream, 0, 0)?;
        // Last: the machine frees each layer's spare slots, and every piece
        // above sized itself from the load's map, capacity = live.
        body.start_residency(gpu)?;
        stream.synchronize()?;
        Ok(body)
    }

    /// Every layer's mixer, in order.
    #[must_use]
    pub fn kinds(&self) -> Vec<LayerKind38> {
        self.plans
            .iter()
            .map(|p| match p.mixer {
                super::plan38::Mixer38::Gdn(_) => LayerKind38::Gdn,
                super::plan38::Mixer38::Qsa(_) => LayerKind38::Qsa,
            })
            .collect()
    }

    /// The cache format the selecting stores hold: the load's one choice,
    /// read back from the live planes (f16 on a load of none).
    fn kv_format(&self) -> KvQ8 {
        self.stores
            .iter()
            .find_map(|s| match s {
                Store38::Qsa { kv, .. } => Some(kv.format()),
                Store38::Rec { .. } => None,
            })
            .unwrap_or_default()
    }

    /// The captured step's launches ([`super::program38`]'s count), and how
    /// many of them are stream memory-operation batches: the go and the wait
    /// of each layer with a host leg ([`Card38::hostless`]).
    #[must_use]
    pub fn step_launches(&self) -> (usize, usize) {
        let served = (0..self.plans.len())
            .filter(|&l| !self.card.hostless(l))
            .count();
        (
            step_launches(&self.plans, &self.card, self.tier.as_ref(), self.head),
            STEP_MEMOPS * served,
        )
    }

    /// Positions every store holds.
    #[must_use]
    pub fn ctx(&self) -> usize {
        self.ctx
    }

    /// The MTP draft layer; `None` on a load without one.
    #[must_use]
    pub fn mtp(&self) -> Option<&Mtp38> {
        self.mtp.as_ref()
    }

    /// The streams' buffer that holds the target's hidden rows after a walk
    /// of `walk`'s arena left them final ([`TargetRows`]): the walk's current
    /// buffer flips once at each PLE layer past layer 0.
    fn final_streams(&self, walk: TargetRows) -> &DeviceBuffer<f32> {
        final_streams(&self.plans, [&self.s, &self.a, &self.wa], walk)
    }

    /// The target's hidden rows `0..rows` after a walk of `walk`'s arena
    /// that left them final ([`TargetRows`]), four streams of
    /// [`geo::HIDDEN`] a row. Blocking; gate use.
    pub fn target_streams(
        &self,
        gpu: &Gpu,
        walk: TargetRows,
        rows: usize,
    ) -> Result<Vec<f32>, GpuError> {
        let buf = self.final_streams(walk);
        let wide = geo::STREAMS * geo::HIDDEN;
        if rows == 0 || rows * wide > buf.len() {
            return Err(GpuError::shape(
                WHAT,
                format!("{rows} rows of an arena of {}", buf.len() / wide),
            ));
        }
        let mut v = buf.to_host_vec(gpu.stream())?;
        v.truncate(rows * wide);
        Ok(v)
    }

    /// The arena the last planned call walked: where its rows' final streams
    /// sit ([`Body38::target_streams`]) — a MTP draft's taps read them there.
    #[must_use]
    pub fn last_walk(&self) -> TargetRows {
        self.last_walk
    }

    /// The values of one tapped row: four streams of [`geo::HIDDEN`], the
    /// final hidden row a position's — an MTP draft's tap need.
    #[must_use]
    pub fn mtp_tap_width(&self) -> usize {
        geo::STREAMS * geo::HIDDEN
    }

    /// The values of one embedding row, [`geo::HIDDEN`]: the width of a media
    /// span's rows, which stand in for a position's embedding in the target
    /// and in the MTP draft's store walks.
    #[must_use]
    pub fn embed_width(&self) -> usize {
        geo::HIDDEN
    }

    fn draft_mut(&mut self) -> Result<&mut Mtp38, GpuError> {
        self.mtp.as_mut().ok_or(GpuError::state(
            WHAT,
            "an MTP draft (Body38::open_placed_mtp)",
        ))
    }

    fn draft_ref(&self) -> Result<&Mtp38, GpuError> {
        self.mtp.as_ref().ok_or(GpuError::state(
            WHAT,
            "an MTP draft (Body38::open_placed_mtp)",
        ))
    }

    /// One walk of the draft's program ([`Mtp38::run`]) beside this body.
    fn mtp_run(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        feed: DraftFeed<'_, TargetRows>,
        head: DraftHead,
        mode: WalkMode,
    ) -> Result<MtpDraft, GpuError> {
        let Body38 {
            mtp,
            k,
            rope,
            seq_rope,
            eps,
            plans,
            s,
            a,
            wa,
            wrote,
            ..
        } = self;
        let d = mtp.as_mut().ok_or(GpuError::state(
            WHAT,
            "an MTP draft (Body38::open_placed_mtp)",
        ))?;
        let target = mtp_target(plans, [s, a, wa], wrote, feed);
        d.run(
            &mtp_ctx(gpu, w, k, *eps, rope_table(rope, seq_rope.as_ref())),
            target,
            feed,
            head,
            mode,
        )
    }

    /// One walk of the draft's program with no readback
    /// ([`Mtp38::run_walk`]) beside this body.
    fn mtp_run_walk(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        feed: DraftFeed<'_, TargetRows>,
        head: DraftHead,
        mode: WalkMode,
    ) -> Result<usize, GpuError> {
        let Body38 {
            mtp,
            k,
            rope,
            seq_rope,
            eps,
            plans,
            s,
            a,
            wa,
            wrote,
            ..
        } = self;
        let d = mtp.as_mut().ok_or(GpuError::state(
            WHAT,
            "an MTP draft (Body38::open_placed_mtp)",
        ))?;
        let target = mtp_target(plans, [s, a, wa], wrote, feed);
        d.run_walk(
            &mtp_ctx(gpu, w, k, *eps, rope_table(rope, seq_rope.as_ref())),
            target,
            feed,
            head,
            mode,
        )
    }

    /// The draft's program and its context over this body's own pieces, for
    /// a store walk of several sequences' rows ([`Mtp38::store_rows`],
    /// [`Mtp38::append_store`]).
    fn mtp_with<T>(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        f: impl FnOnce(&mut Mtp38, &MtpCtx<'_>) -> Result<T, GpuError>,
    ) -> Result<T, GpuError> {
        let Body38 {
            mtp,
            k,
            rope,
            seq_rope,
            eps,
            ..
        } = self;
        let d = mtp.as_mut().ok_or(GpuError::state(
            WHAT,
            "an MTP draft (Body38::open_placed_mtp)",
        ))?;
        f(
            d,
            &mtp_ctx(gpu, w, k, *eps, rope_table(rope, seq_rope.as_ref())),
        )
    }

    /// One window's chain of the draft's program ([`Mtp38::run_chain`])
    /// beside this body: `refresh`'s walk then `own` own walks, one readback.
    fn mtp_run_chain(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        refresh: DraftFeed<'_, TargetRows>,
        own: usize,
        head: DraftHead,
        mode: WalkMode,
    ) -> Result<MtpDraft, GpuError> {
        let Body38 {
            mtp,
            k,
            rope,
            seq_rope,
            eps,
            plans,
            s,
            a,
            wa,
            wrote,
            ..
        } = self;
        let d = mtp.as_mut().ok_or(GpuError::state(
            WHAT,
            "an MTP draft (Body38::open_placed_mtp)",
        ))?;
        let target = mtp_target(plans, [s, a, wa], wrote, refresh);
        d.run_chain(
            &mtp_ctx(gpu, w, k, *eps, rope_table(rope, seq_rope.as_ref())),
            target,
            refresh,
            own,
            head,
            mode,
        )
    }

    /// One round of several slots' chains ([`GpuModel::mtp_chain_slots`])
    /// beside this body, the homes canonical: `chains`' `(slot, refresh,
    /// own)`, each sequence the live one (`None`) or `parked`'s entry its
    /// `seqs` index names. Each chain is checked and staged with its
    /// sequence live ([`Mtp38::stage_chain`]), then each walk runs once over
    /// every chain it reaches, each chain's attention with its sequence live
    /// ([`on_seq`]), and the round is read back once ([`Mtp38::read_chains`]).
    fn mtp_chain_round(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        parked: &mut [&mut Slot38],
        (chains, seqs): (&[SlotChain38<'_>], &[Option<usize>]),
        head: DraftHead,
        mode: WalkMode,
    ) -> Result<Vec<MtpDraft>, GpuError> {
        let stream = gpu.stream();
        let mut blocks: Vec<ChainBlock> = Vec::with_capacity(chains.len());
        let (mut at, mut pair) = (0, 0);
        for (block, (&(_, feed, own), &seq)) in chains.iter().zip(seqs).enumerate() {
            let seq = seq.map(|i| &mut *parked[i]);
            let b = on_seq(self, seq, |body| {
                let Body38 {
                    mtp,
                    plans,
                    s,
                    a,
                    wa,
                    wrote,
                    ..
                } = body;
                let d = mtp.as_mut().ok_or(GpuError::state(
                    WHAT,
                    "an MTP draft (Body38::open_placed_mtp)",
                ))?;
                let target = mtp_target(plans, [s, a, wa], wrote, DraftFeed::Rows(feed));
                d.stage_chain(stream, target, feed, (own, head, mode), (block, at, pair))
            })?;
            at += b.m();
            pair += b.pairs();
            blocks.push(b);
        }
        self.draft_mut()?.upload_chains(stream)?;
        let walks = blocks.iter().map(|b| b.pairs()).max().unwrap_or(0);
        for j in 0..walks {
            self.mtp_with(gpu, w, |d, c| d.chain_in(c, &blocks, j))?;
            for (block, at, m) in chain_spans(&blocks, j) {
                let seq = seqs[block].map(|i| &mut *parked[i]);
                on_seq(self, seq, |body| {
                    body.mtp_with(gpu, w, |d, c| d.attend(c, at, m))
                })?;
            }
            self.mtp_with(gpu, w, |d, c| d.chain_out(c, &blocks, j, head))?;
        }
        for (b, &seq) in blocks.iter().zip(seqs) {
            let seq = seq.map(|i| &mut *parked[i]);
            on_seq(self, seq, |body| {
                body.draft_mut()?.end_chain(b);
                Ok(())
            })?;
        }
        self.draft_ref()?.read_chains(stream, &blocks)
    }

    /// Tokens of the vocabulary.
    #[must_use]
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Device bytes of the layers' stores and the PLE ring.
    #[must_use]
    pub fn store_bytes(&self) -> usize {
        self.stores.iter().map(Store38::bytes).sum::<usize>() + self.ple_ring.num_bytes()
    }

    /// The live sequence's stores by position on the card, the selecting
    /// stores' and the draft store's: the bytes their mapped pages hold
    /// (the rows the sequence reached, by pages) and the bytes their layout
    /// takes at the load's context, which the plan counts.
    #[must_use]
    pub fn store_room(&self) -> (usize, usize) {
        let draft = self.mtp.as_ref().map_or((0, 0), Mtp38::store_room);
        let (mapped, layout) = self
            .stores
            .iter()
            .filter(|s| matches!(s, Store38::Qsa { .. }))
            .fold(draft, |(m, l), s| (m + s.mapped(), l + s.bytes()));
        (mapped, layout)
    }

    /// The bytes of the live sequence's stores by position (the selecting
    /// stores' and the draft store's) whose pages another sequence's stores
    /// map too ([`GpuModel::seq_share_prefix`]): counted in
    /// [`Body38::store_room`]'s mapped bytes of both.
    #[must_use]
    pub fn store_shared(&self) -> usize {
        let draft = self.mtp.as_ref().map_or(0, Mtp38::store_shared);
        self.stores.iter().map(Store38::shared).sum::<usize>() + draft
    }

    /// The device bytes the planned slots' stores by position may still
    /// back as their sequences grow: every slot's pages at the load's
    /// context, less the pages the live sequence already backs. A slot not
    /// live is counted as backing none, so the count errs high.
    fn store_growth(&self) -> usize {
        let whole = self.mtp.as_ref().map_or(0, Mtp38::store_whole)
            + self.stores.iter().map(Store38::whole).sum::<usize>();
        (self.slots_planned * whole).saturating_sub(self.store_room().0)
    }

    /// The host tier.
    #[must_use]
    pub fn hybrid(&self) -> &Hybrid<HostRun> {
        &self.hybrid
    }

    /// The NVMe expert tier's arena, on a load whose plan paged routed
    /// experts: the counters a run reports ([`NvTier::stats`]). `None` on a
    /// load with no tier.
    #[must_use]
    pub fn nvme_tier(&self) -> Option<&Arc<NvTier>> {
        self.nvme_tier.as_ref()
    }

    /// The host tier, for a caller that attaches or marks its route trace.
    pub fn hybrid_mut(&mut self) -> &mut Hybrid<HostRun> {
        &mut self.hybrid
    }

    /// Keep every residency boundary's report from now on, for
    /// [`Body38::take_residency_passes`]: a binary that prints the `residency
    /// pass` records asks once, with the boundaries it takes at most between
    /// two takes. Nothing is kept until then.
    pub fn log_residency(&mut self, passes: usize) {
        self.residency_glue.log_passes(passes);
    }

    /// The least count an `admit` pick admits ([`super::wide38`]'s
    /// `STREAM_FLOOR`): a prompt call of fewer ids opens no call.
    pub const STREAM_FLOOR: u32 = STREAM_FLOOR;

    /// What the next prompt calls move (`BLOOMERY_XSTREAM`,
    /// [`super::wide38`]'s module doc): off until set here, so one load can
    /// run every arm. `Split` starts the expert stream on `gpu` at its first
    /// setting ([`crate::host::HostTier::xstream_start`]: the ring sized from
    /// the card's free bytes, the lane's probe, [`XSTREAM_COSTS`] for the
    /// rule) and arms the walk's card route for its ring; the stream then
    /// stays for the load's life, idle under `Admit` and `Off`. Refused by
    /// name: `Admit` or `Split` on a load that runs no residency machine,
    /// `Split` on a load that hangs an expert tier, on one whose card route
    /// reads NVFP4 planes ([`XSTREAM_FORM`]) or when the stream cannot start
    /// (the card's room among the refusals), and any change while a call
    /// streams.
    pub fn set_xstream(&mut self, gpu: &Gpu, mode: XMode) -> Result<(), GpuError> {
        const WHAT_S: &str = "Body38::set_xstream";
        if self.stream.streaming() {
            return Err(GpuError::state(WHAT_S, "no prompt call streaming"));
        }
        if mode != XMode::Off && self.hybrid.swap().is_none() {
            return Err(GpuError::state(
                "BLOOMERY_XSTREAM",
                "a residency machine (BLOOMERY_RESIDENCY=mid-p<P>-s<S>) for admit or split",
            ));
        }
        if mode == XMode::Split && self.tier.is_some() {
            return Err(GpuError::shape(
                "BLOOMERY_XSTREAM",
                "split on a load with an expert tier: the stream's ring serves the stage card \
                 of a load with none",
            ));
        }
        if mode == XMode::Split
            && self
                .card
                .routed_types()
                .any(|(gate_up, down)| gate_up == GgmlType::NVFP4 || down == GgmlType::NVFP4)
        {
            return Err(GpuError::shape(
                XSTREAM_FORM,
                "split on a load whose card route reads NVFP4 planes: the ring pass reads a \
                 matrix the ring holds as words",
            ));
        }
        if mode == XMode::Split && self.hybrid.xstream().is_none() {
            // The file's own row: the Q4 kind's recalibrated one, any other
            // kind's the unfitted row until its record exists.
            let costs = if self.routed_gate == GgmlType::Q4_K {
                XSTREAM_COSTS
            } else {
                XSTREAM_COSTS_UNFITTED
            };
            // The plan reserves no ring: it takes what the card's margin leaves past the keep-free
            // and the pages the slots' stores may still back.
            let mut cfg = XCfg::new(costs, geo::EXPERTS, geo::N_USED, DEADLINE, None);
            cfg.keep_free += self.store_growth() as u64;
            let half = self
                .hybrid
                .xstream_start(
                    gpu.context(),
                    gpu.stream(),
                    cfg,
                    crate::host::xstream::PROBE,
                )?
                .half_slots();
            if let Some(r) = self.wide.card.as_mut() {
                r.arm_ring(gpu.stream(), half)?;
            }
        }
        self.stream.set_mode(mode);
        Ok(())
    }

    /// How the ubatch walks cut their rows (`BLOOMERY_SPLIT_WALK`,
    /// [`SplitWalk`]) from the next walk on. The load sets
    /// [`SplitWalk::Rule`] on a load whose routed experts sit on the cards
    /// and in host memory alone and [`SplitWalk::Off`] on one with an expert
    /// tier or an NVMe-paged plan; a split there is refused by name, the cut
    /// left as it was.
    pub fn set_split_walk(&mut self, w: SplitWalk) -> Result<(), GpuError> {
        self.wide.set_split_walk(w)
    }

    /// How the ubatch walks cut their rows ([`Body38::set_split_walk`]).
    #[must_use]
    pub fn split_walk(&self) -> SplitWalk {
        self.wide.split_walk()
    }

    /// The units the last ubatch walk ran (two for a split walk); 0 before
    /// a walk since the load. Gate use.
    #[must_use]
    pub fn ubatch_units(&self) -> usize {
        self.wide.walked_units()
    }

    /// The rows the last ubatch walk's last layer ran past its mixer's store
    /// writes: the walk's rows something reads after it on a last layer
    /// without a host leg, every row otherwise; 0 before a walk since the
    /// load. Gate use.
    #[must_use]
    pub fn ubatch_last_rows(&self) -> usize {
        self.wide.last_rows()
    }

    /// The tokens one run of the ubatch walk's card route takes
    /// (`place::card_route_run`); `None` on a load whose slot map puts no
    /// routed expert on the card. Gate use.
    #[must_use]
    pub fn card_route_run(&self) -> Option<usize> {
        self.wide.card.as_ref().map(|c| c.run)
    }

    /// Whether a returning prompt call ([`super::wide38`]'s module doc) runs
    /// its whole return at its end instead of in its last unit's walk: the
    /// twin a gate holds the walk's return to, the same end state slot for
    /// slot. Off at the load. Gate use. Refused by name while a call streams
    /// ([`PromptStream::set_return_at_end`]).
    pub fn set_return_at_end(&mut self, on: bool) -> Result<(), GpuError> {
        self.stream.set_return_at_end(on)
    }

    /// The expert stream's rule costs in place of [`XSTREAM_COSTS`]
    /// ([`PromptStream::set_costs`]). Refused by name before `split` started
    /// the stream and while a call streams.
    pub fn set_xstream_costs(&mut self, costs: Costs) -> Result<(), GpuError> {
        self.stream.set_costs(&mut self.hybrid, costs)
    }

    /// The load line's `xstream=` word: the mode, and under a started
    /// stream its ring's slots a half, its pinned staging slots (0:
    /// pageable), the lane's probed and priced rates in GB/s and the least
    /// unit width that streams ([`PromptStream::word`]).
    #[must_use]
    pub fn xstream_word(&self) -> String {
        self.stream.word(&self.hybrid)
    }

    /// The last streaming prompt call's streamed layers, each with its
    /// ubatch, and the stream's end; empty and `None` after a call that did
    /// not stream. Taken: a second read is empty
    /// ([`PromptStream::take_xstream_records`]).
    pub fn take_xstream_records(&mut self) -> (Vec<(usize, XLayer)>, Option<XReport>) {
        self.stream.take_xstream_records()
    }

    /// The last streaming prompt call's picks, each with its ubatch, and its
    /// end; empty and `None` after a call that did not stream. Taken: a
    /// second read is empty ([`PromptStream::take_pick_records`]).
    pub fn take_stream_records(&mut self) -> (Vec<(usize, CallPick)>, Option<CallReport>) {
        self.stream.take_pick_records()
    }

    /// The residency machine's source, when the load runs one.
    #[must_use]
    pub fn residency_source(&self) -> Option<&FileSwap> {
        self.residency_glue.source()
    }

    /// The residency boundaries' reports since the last take, in order, each
    /// with the kind of the pass it ended; the log keeps its capacity.
    pub fn take_residency_passes(&mut self) -> Vec<(PassKind, PassReport)> {
        self.residency_glue.take_passes()
    }

    /// The residency machine's late-flip debug ring, taken
    /// ([`HostTier::take_late_flips`]): one entry a flip whose copy the
    /// boundary that lands it waited for, with the terms of its wait that
    /// name the mechanism. Empty without a machine.
    pub fn take_late_flips(&mut self) -> crate::host::swap::LateLog {
        self.hybrid.take_late_flips()
    }

    /// Arm (or disarm) the per-layer taps ([`GpuModel::set_layer_taps`], the
    /// one caller, drops the captures first). Load-time allocation.
    fn set_taps(&mut self, gpu: &Gpu, on: bool) -> Result<(), GpuError> {
        self.taps = if on {
            Some(Taps38::new(gpu.stream(), self.plans.len(), &self.s.route)?)
        } else {
            None
        };
        Ok(())
    }

    /// Every layer's streams after the last step, `4 · 2560` a layer, stream
    /// `s` of layer `l` at `l · 10240 + s · 2560`. Blocking; refused when the
    /// taps are not armed.
    pub fn taps(&self, gpu: &Gpu) -> Result<Vec<f32>, GpuError> {
        let taps = self
            .taps
            .as_ref()
            .ok_or(GpuError::state(WHAT, "armed taps (set_layer_taps)"))?;
        let mut out = Vec::with_capacity(taps.out.len() * geo::STREAMS * geo::HIDDEN);
        for t in &taps.out {
            out.extend(t.to_host_vec(gpu.stream())?);
        }
        Ok(out)
    }

    /// Every layer's router after the last step: its [`geo::EXPERTS`]
    /// logits and its [`geo::N_USED`] routed ids, in slot order. Blocking;
    /// refused when the taps are not armed.
    pub fn route_taps(&self, gpu: &Gpu) -> Result<Vec<RouteTap>, GpuError> {
        let taps = self
            .taps
            .as_ref()
            .ok_or(GpuError::state(WHAT, "armed taps (set_layer_taps)"))?;
        taps.logits
            .iter()
            .zip(&taps.ids)
            .map(|(lg, id)| {
                let (mut lg, mut id) =
                    (lg.to_host_vec(gpu.stream())?, id.to_host_vec(gpu.stream())?);
                if lg.len() < geo::EXPERTS || id.len() < geo::N_USED {
                    return Err(GpuError::shape(
                        WHAT,
                        format!("route taps of {} logits and {} ids", lg.len(), id.len()),
                    ));
                }
                lg.truncate(geo::EXPERTS);
                id.truncate(geo::N_USED);
                Ok(RouteTap {
                    logits: lg,
                    ids: id,
                })
            })
            .collect()
    }

    /// Every layer's store and the PLE ring, read back: what a run leaves
    /// behind, for a gate to compare bit for bit — a delta store's state as
    /// the committed lane holds it, a selecting store's whole planes (every
    /// row backed first, [`Store38::fill`]). Blocking.
    pub fn stores_host(&mut self, gpu: &Gpu) -> Result<(Vec<Store38Host>, Vec<f32>), GpuError> {
        let stream = gpu.stream();
        for s in &mut self.stores {
            s.fill()?;
        }
        let stores = self
            .stores
            .iter()
            .map(|s| {
                Ok(match s {
                    Store38::Rec { rec, .. } => Store38Host::Rec {
                        state: lane_of(rec.state.to_host_vec(stream)?, self.lane.lane())?,
                        ring: rec.ring.to_host_vec(stream)?,
                        carry: rec
                            .carry
                            .as_ref()
                            .ok_or_else(no_carry)?
                            .to_host_vec(stream)?,
                    },
                    Store38::Qsa { kv, ring, pooled } => {
                        let q8 = |codes: &DeviceBuffer<u32>, scales: &DeviceBuffer<u16>| {
                            let mut halves: Vec<u16> = codes
                                .to_host_vec(stream)?
                                .into_iter()
                                .flat_map(|w| [w as u16, (w >> 16) as u16])
                                .collect();
                            halves.extend(scales.to_host_vec(stream)?);
                            Ok::<_, GpuError>(halves)
                        };
                        let (k, v) = match &**kv {
                            KvPlanes::F16 { k, v } => {
                                (k.to_host_vec(stream)?, v.to_host_vec(stream)?)
                            }
                            KvPlanes::Q8 { kq, kd, vq, vd } => (q8(kq, kd)?, q8(vq, vd)?),
                        };
                        Store38Host::Qsa {
                            k,
                            v,
                            ring: ring.to_host_vec(stream)?,
                            pooled: pooled.to_host_vec(stream)?,
                        }
                    }
                })
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        Ok((stores, self.ple_ring.to_host_vec(stream)?))
    }

    /// Every selecting store's K/V planes, raw key ring and pooled keys set to
    /// `bits`, an f16 pattern: a gate's stand-in for the rows `reset` leaves
    /// as they were, so a path that reads a row it did not write reads the
    /// pattern, not a previous run's value. On a q8_0 cache the scales hold
    /// `bits` and each codes word two copies of it. Synchronizes; gate use.
    pub fn fill_planes(&mut self, gpu: &Gpu, bits: u16) -> Result<(), GpuError> {
        let stream = gpu.stream();
        let fill = |buf: &mut DeviceBuffer<u16>| buf.copy_from_host(stream, &vec![bits; buf.len()]);
        let word = u32::from(bits) * 0x1_0001;
        for s in &mut self.stores {
            s.fill()?;
            s.own_from(0)?;
            if let Store38::Qsa { kv, ring, pooled } = s {
                match &mut **kv {
                    KvPlanes::F16 { k, v } => {
                        fill(k)?;
                        fill(v)?;
                    }
                    KvPlanes::Q8 { kq, kd, vq, vd } => {
                        for codes in [kq, vq] {
                            codes.copy_from_host(stream, &vec![word; codes.len()])?;
                        }
                        fill(kd)?;
                        fill(vd)?;
                    }
                }
                fill(ring)?;
                fill(pooled)?;
            }
        }
        Ok(())
    }

    /// Every selecting store's pooled keys from pool `from` on set to
    /// `bits`, an f16 pattern: a gate's stand-in for pools a rejected row
    /// left past the count, which no select may read before the row that
    /// completes them writes them again. Synchronizes; gate use.
    pub fn poison_pools_from(&mut self, gpu: &Gpu, from: usize, bits: u16) -> Result<(), GpuError> {
        let stream = gpu.stream();
        for s in &mut self.stores {
            if let Store38::Qsa { pooled, .. } = s {
                pooled.fill()?;
                pooled.own_from(0)?;
                let mut rows = pooled.to_host_vec(stream)?;
                let at = (from * crate::qsa::DIM).min(rows.len());
                rows[at..].fill(bits);
                pooled.copy_from_host(stream, &rows)?;
            }
        }
        Ok(())
    }

    /// The lane the lane word holds: the committed delta state's.
    #[must_use]
    pub fn lane(&self) -> u32 {
        self.lane.lane()
    }

    /// The captured verify's nodes at `m` rows: its walk's launches
    /// ([`super::program38`]'s count) and its PLE rows' arrival
    /// ([`PLE_ARRIVAL_NODES`]) at each PLE site.
    #[must_use]
    pub fn verify_launches(&self, m: usize) -> usize {
        let sites = self.plans.iter().filter(|p| p.ple.is_some()).count();
        verify_launches(&self.plans, &self.card, self.tier.as_ref(), m, self.head)
            + sites * PLE_ARRIVAL_NODES
    }

    /// Refused by name unless the stores hold `pos` positions: a call from
    /// `pos` that failed after its launch has already run the delta layers'
    /// recurrence and the conv ring over its rows, and running them again
    /// would apply them twice.
    fn stores_at(&self, pos: u32) -> Result<(), GpuError> {
        match self.held {
            h if h == pos => Ok(()),
            h if h > pos => Err(GpuError::shape(
                WHAT,
                format!(
                    "position {pos}: the call there failed after its chain was launched, so the \
                     recurrent stores hold it already (they stand at {h}); reset — a delta layer \
                     keeps no earlier state to cut back to"
                ),
            )),
            h => Err(GpuError::shape(
                WHAT,
                format!("position {pos}, where the stores hold {h} positions"),
            )),
        }
    }

    /// What a cut to at most `n` positions of a model standing at `pos`
    /// keeps, and why: with a verify waiting, `n` when it lies past the
    /// verify's first position (`Rule`: its commit replays the accepted
    /// recovery records into the state lane) and every position when
    /// `n` reaches them (`Current`);
    /// else the checkpoints' rule ([`Checkpoints::kept`]): every position, the
    /// empty model at 0, or the nearest point at or below `n`, nothing
    /// (`Missed`) when none lies there — a delta layer keeps no earlier state
    /// but a copy. After a call failed past its launch the stores hold more
    /// than `pos`, which is no longer kept. On a media load a point inside an
    /// image grants nothing: the cut is asked again at the image's start.
    /// Never past `pos`. The rule
    /// [`Body38::commit`] takes.
    #[must_use]
    pub fn kept(&self, n: u32, pos: u32) -> Kept {
        let verify = self.pending.as_ref().map(|p| p.pos0);
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

    /// Whether a prompt call takes checkpoints at its marks: off at load, so
    /// a binary that never cuts behind the fed positions copies nothing; the
    /// serve seat turns it on. Off drops none taken.
    pub fn set_checkpoints(&mut self, on: bool) {
        self.marks = on;
    }

    /// Whether a prompt call takes checkpoints ([`Body38::set_checkpoints`]).
    #[must_use]
    pub fn takes_checkpoints(&self) -> bool {
        self.marks
    }

    /// Whether a prompt call closes its prompt: its last id is the one a
    /// server would step after the call, run as the call's last row, whose
    /// head gives the argmax after it, and the call's checkpoints are those of
    /// the call less its last id: the end mark stands before the last id
    /// ([`GpuModel::feed38`]). A call of passes or steps leaves the bits of
    /// the call less its last id and that id's step; a call by ubatches
    /// leaves the ubatch walk's last row, as one call of the whole prompt
    /// does. Off at load; the serve seat turns it on around a prompt's last
    /// call.
    pub fn set_close(&mut self, on: bool) {
        self.close = on;
    }

    /// The PLE rows of `token` at `pos`, the input the next step is to take,
    /// read ahead on the NVMe tier's helper while the caller hands the step's
    /// answer on. The history and every device buffer stay as they are: the
    /// next fill takes the rows when its position and row ids are these
    /// ([`PleRead::start`]), and any other call drains them first. Nothing
    /// on the host, at a position other than the history's next, or for a
    /// token the hash refuses (its own fill refuses it by name).
    pub fn prefetch_input(&mut self, token: u32, pos: u32) -> Result<(), GpuError> {
        self.ple.prefetch(pos, token)
    }

    /// The waiting cut carried out ([`Checkpoints::apply`]): the committed
    /// lane, the conv rings, the PLE ring and the raw key rings from the
    /// point's slot, each
    /// delta layer's committed lane stamped at the position the cut left
    /// (`held`) and every other never written. Nothing when no cut waits.
    /// Waits for the copies.
    fn settle(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        if !self.ckpt.pending() {
            return Ok(());
        }
        let lane = self.lane.lane();
        let mut c = Copied::of(&mut self.stores, &mut self.ple_ring, lane, None, None)?;
        self.ckpt.apply(stream, &mut c.list())?;
        drop(c);
        self.restamp(stream, lane, self.held)
    }

    /// Every delta layer's stamps: lane `lane` at position `at`, every other
    /// never written. Synchronizes; never inside a capture.
    fn restamp(&mut self, stream: &CudaStream, lane: u32, at: u32) -> Result<(), GpuError> {
        let st = stamps_at(LANES, lane, at)?;
        for s in &mut self.stores {
            if let Store38::Rec { stamp, .. } = s {
                stamp.copy_from_host(stream, &st)?;
            }
        }
        Ok(())
    }

    /// [`Body38::take_mark`]'s refusals and its waiting cut: the stores and
    /// the PLE history at `pos`, and no verify waiting.
    fn mark_checks(&mut self, stream: &CudaStream, pos: u32) -> Result<(), GpuError> {
        self.stores_at(pos)?;
        if let Some(p) = &self.pending {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a checkpoint at {pos} while the verify of {} rows at {} waits for its commit",
                    p.tokens.len(),
                    p.pos0
                ),
            ));
        }
        self.settle(stream)?;
        if self.ple.hist.next_pos() != u64::from(pos) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a checkpoint at {pos} with the PLE history at {}",
                    self.ple.hist.next_pos()
                ),
            ));
        }
        Ok(())
    }

    /// A checkpoint at `pos`, where the stores stand, after any waiting cut:
    /// the recurrent stores copied to a host slot with the PLE history beside
    /// them and, on a load with the draft, the target's final hidden row at
    /// `pos - 1` when an arena still holds it ([`Body38::point_rows`]), or
    /// nothing where one stands. Refused by name while a verify waits or the
    /// stores hold another position. Waits for the copies.
    fn take_mark(&mut self, stream: &CudaStream, pos: u32) -> Result<(), GpuError> {
        if let Err(e) = self.mark_checks(stream, pos) {
            let _ = self.drop_end_mark(stream);
            return Err(e);
        }
        if let Some(open) = self.end_mark.take() {
            let hist = self.ple.hist.clone();
            return self.seal_end_mark(stream, pos, open, hist, |_| Ok(()));
        }
        let lane = self.lane.lane();
        let mut c = Copied::of(&mut self.stores, &mut self.ple_ring, lane, None, None)?;
        let take = self.ckpt.take(stream, pos, &mut c.list());
        drop(c);
        match take {
            Ok(Take::Copy { evicted, .. }) => {
                if let Some(e) = evicted {
                    self.ple_points.retain(|&(p, _)| p != e);
                    self.point_rows.retain(|&(p, _)| p != e);
                }
                let hist = self.ple.hist.clone();
                self.point_taken(stream, pos, hist)
            }
            Ok(Take::Held | Take::Empty) => Ok(()),
            Err(e) => {
                self.ple_points.clear();
                self.point_rows.clear();
                Err(e)
            }
        }
    }

    /// A point's host side once its stores' copy holds `pos`: the PLE
    /// history `hist` beside it and, on a load with the draft, the target's final
    /// hidden row at `pos - 1` when an arena still holds it. Blocking.
    fn point_taken(
        &mut self,
        stream: &CudaStream,
        pos: u32,
        hist: History,
    ) -> Result<(), GpuError> {
        let i = self.ple_points.partition_point(|&(p, _)| p < pos);
        self.ple_points.insert(i, (pos, hist));
        let point_row = match (&self.mtp, pos.checked_sub(1)) {
            (Some(_), Some(before)) => self.hidden_row(stream, before)?,
            _ => None,
        };
        if let Some(row) = point_row {
            let i = self.point_rows.partition_point(|&(p, _)| p < pos);
            self.point_rows.insert(i, (pos, row));
        }
        Ok(())
    }

    /// The checkpoint at `to`, where a ubatch walking from `pos` ends, opened
    /// before it walks ([`Checkpoints::open_take`], after any waiting cut):
    /// the walk records each layer's done event ([`WideParts::done`]) and
    /// [`Body38::copy_end_mark`] puts each store's copy behind the layer
    /// that writes it, so it runs under the walk's later layers. Nothing opens
    /// where no copy is due or a verify waits ([`Body38::take_mark`] takes
    /// that mark as before).
    fn open_end_mark(&mut self, stream: &CudaStream, pos: u32, to: u32) -> Result<(), GpuError> {
        self.stores_at(pos)?;
        if self.pending.is_some() || self.end_mark.is_some() {
            return Ok(());
        }
        if self.stores.len() != self.plans.len() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} stores for {} layers",
                    self.stores.len(),
                    self.plans.len()
                ),
            ));
        }
        self.settle(stream)?;
        let lane = self.lane.lane();
        let mut c = Copied::of(&mut self.stores, &mut self.ple_ring, lane, None, None)?;
        let open = self.ckpt.open_take(stream, to, &mut c.list());
        drop(c);
        let Some(open) = open? else {
            return Ok(());
        };
        if self.marker.is_none() {
            match EndMarker::new(stream, self.plans.len()) {
                Ok(m) => self.marker = Some(m),
                Err(e) => {
                    self.ckpt.abandon(stream, open)?;
                    return Err(e);
                }
            }
        }
        self.end_mark = Some(open);
        Ok(())
    }

    /// An open end mark's copies, enqueued once its ubatch's walk is: each
    /// store on the marker's stream behind the done event of the layer that
    /// writes it ([`checkpoint_owners`]): each delta layer's lane, conv ring
    /// and carry behind the layer's, the PLE ring behind its site's, each
    /// selecting layer's key ring behind the layer's. Nothing without an
    /// open mark.
    fn copy_end_mark(&mut self) -> Result<(), GpuError> {
        let (Some(open), Some(m)) = (self.end_mark.as_mut(), self.marker.as_ref()) else {
            return Ok(());
        };
        let site = (self.plans.iter().position(|p| p.ple.is_some())).ok_or(GpuError::state(
            WHAT,
            "the PLE site the ring's copy waits for",
        ))?;
        let delta = (self.stores.iter()).map(|s| matches!(s, Store38::Rec { .. }));
        let owners = checkpoint_owners(delta, site);
        let lane = self.lane.lane();
        let mut c = Copied::of(&mut self.stores, &mut self.ple_ring, lane, None, None)?;
        for (i, store) in c.list().into_iter().enumerate() {
            let l = *(owners.get(i))
                .ok_or(GpuError::state(WHAT, "the layer a store's copy waits for"))?;
            if i == 0 || owners[i - 1] != l {
                let done = m.done.get(l).ok_or(GpuError::state(
                    WHAT,
                    "the done event of the layer a store's copy waits for",
                ))?;
                m.copy.wait(done)?;
            }
            self.ckpt.copy(&m.copy, open, i, store)?;
        }
        Ok(())
    }

    /// The end mark `open` at `pos` sealed ([`Checkpoints::seal`]) once its
    /// walk is waited for and found sound, and its point's host side
    /// ([`Body38::point_taken`]) with the PLE history `hist` and `edit` run
    /// over its stores ([`Checkpoints::seal_with`]); points a later take evicted lose theirs.
    /// A mark of another position is given back and refused by name.
    fn seal_end_mark(
        &mut self,
        stream: &CudaStream,
        pos: u32,
        open: checkpoint::Pending,
        hist: History,
        edit: impl FnOnce(&mut [&mut [f32]]) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        let copy = self.marker.as_ref().map_or(stream, |m| &*m.copy);
        if open.at() != pos {
            let at = open.at();
            self.ckpt.abandon(copy, open)?;
            return Err(GpuError::shape(
                WHAT,
                format!("the end mark opened at {at} sealed at {pos}"),
            ));
        }
        if let Err(e) = self.ckpt.seal_with(copy, open, edit) {
            self.ple_points.clear();
            self.point_rows.clear();
            return Err(e);
        }
        let kept = self.ckpt.positions();
        self.ple_points.retain(|(p, _)| kept.contains(p));
        self.point_rows.retain(|(p, _)| kept.contains(p));
        if kept.contains(&pos) {
            self.point_taken(stream, pos, hist)?;
        }
        Ok(())
    }

    /// The closing call's end mark at `at`, the position before its last id,
    /// sealed once its fold has walked a row past it ([`fold_units`]): every
    /// store but the delta lanes holds a position's state a row past it (the
    /// rings keep a pass past their reach, the rest are rows by position),
    /// and the walk kept its last chunk out of the lanes ([`Body38::keep_last`]),
    /// in the point's copy each delta layer's carry gives that row back (its
    /// count one less, standing at `at`), and the PLE history `hist` (at
    /// `at`) goes beside it ([`Body38::seal_end_mark`]). Refused by
    /// name, the take forgotten, when a carry does not stand a row past `at`
    /// with a row in it.
    fn seal_close_mark(
        &mut self,
        stream: &CudaStream,
        at: u32,
        hist: History,
    ) -> Result<(), GpuError> {
        let Some(open) = self.end_mark.take() else {
            return Ok(());
        };
        let delta = (self.stores.iter())
            .filter(|s| matches!(s, Store38::Rec { .. }))
            .count();
        self.seal_end_mark(stream, at, open, hist, |stores| {
            for i in (0..delta).map(|j| 3 * j + 2) {
                let carry = stores
                    .get_mut(i)
                    .filter(|c| c.len() >= 2)
                    .ok_or(GpuError::state(
                        WHAT,
                        "each delta layer's carry in the point",
                    ))?;
                let (rows, past) = (carry[0].to_bits(), carry[1].to_bits());
                if rows == 0 || past != at + 1 {
                    return Err(GpuError::shape(
                        WHAT,
                        format!(
                            "the closing mark at {at}: store {i}'s carry holds {rows} rows \
                             standing at {past}, not a row standing at {}",
                            at + 1
                        ),
                    ));
                }
                carry[0] = f32::from_bits(rows - 1);
                carry[1] = f32::from_bits(at);
            }
            Ok(())
        })
    }

    /// An open end mark given back ([`Checkpoints::abandon`]): its ubatch
    /// failed before the mark, or the mark was refused.
    fn drop_end_mark(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        let Some(open) = self.end_mark.take() else {
            return Ok(());
        };
        let copy = self.marker.as_ref().map_or(stream, |m| &*m.copy);
        self.ckpt.abandon(copy, open)
    }

    /// The target's final hidden row at `pos` ([`Hidden::Target`]'s
    /// rows), read from the arena whose record holds it ([`Wrote38`]);
    /// `None` when none does. Blocking.
    fn hidden_row(&self, stream: &CudaStream, pos: u32) -> Result<Option<Vec<f32>>, GpuError> {
        let wide = geo::STREAMS * geo::HIDDEN;
        for walk in [TargetRows::Step, TargetRows::Pass, TargetRows::Ubatch] {
            let (first, rows) = self.wrote.of(walk).parts();
            let Some(row) = pos.checked_sub(first).filter(|&r| r < rows) else {
                continue;
            };
            let complete = self.wrote.final_of(walk);
            if !pos
                .checked_sub(complete.parts().0)
                .is_some_and(|r| complete.holds(r as usize, 1, pos))
            {
                continue;
            }
            let buf = self.final_streams(walk);
            let at = row as usize * wide;
            if at + wide > buf.len() {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "the {walk:?} arena's row {row} of {} rows",
                        buf.len() / wide
                    ),
                ));
            }
            let v = crate::tensor::Window::<f32>::of(buf, at * size_of::<f32>(), wide)?;
            return Ok(Some(to_host(stream, &v)?));
        }
        Ok(None)
    }

    /// Whether the MTP draft rejoins the sequence at `pos` after a cut there
    /// ([`Body38::cut`]) or a state put back as its first `pos` positions:
    /// the point at `pos` carries the target's hidden row at `pos - 1`
    /// ([`Body38::point_rows`]) and the draft's store holds every position
    /// below `pos`. False on a load without the draft.
    #[must_use]
    pub fn draft_rejoins(&self, pos: u32) -> bool {
        self.rejoin_row(pos).is_some()
    }

    /// The hidden row the point at `pos` carries ([`Body38::point_rows`]) where
    /// the draft rejoins there ([`Body38::draft_rejoins`]).
    fn rejoin_row(&self, pos: u32) -> Option<&[f32]> {
        let d = self.mtp.as_ref()?;
        let (_, row) = self.point_rows.iter().find(|&&(p, _)| p == pos)?;
        (d.held() >= pos as usize).then_some(&row[..])
    }

    /// The draft's store cut to `pos` and `row`, the target's hidden row at
    /// `pos - 1`, written to the step arena's row 0, its record holding that
    /// one position: what a plain step at `pos - 1` leaves for the draft's
    /// next walk, the row a window that rejoins the sequence walks first.
    /// Refused by name, nothing moved, for a row of another width, at 0 and
    /// on a load without the draft; a copy that fails leaves the store
    /// forgotten. Blocking.
    fn rejoin_draft(&mut self, stream: &CudaStream, pos: u32, row: &[f32]) -> Result<(), GpuError> {
        let wide = geo::STREAMS * geo::HIDDEN;
        if row.len() != wide || pos == 0 || self.final_streams(TargetRows::Step).len() < wide {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a rejoin at {pos} with a hidden row of {} values into a step arena of {}",
                    row.len(),
                    self.final_streams(TargetRows::Step).len()
                ),
            ));
        }
        let Body38 {
            plans,
            s,
            a,
            wa,
            mtp,
            wrote,
            ..
        } = self;
        let d = mtp.as_mut().ok_or(GpuError::state(
            WHAT,
            "an MTP draft (Body38::open_placed_mtp)",
        ))?;
        d.cut(pos);
        let step = final_streams_mut(plans, [s, a, wa], TargetRows::Step);
        let mut host = vec![0.0f32; step.len()];
        host[..wide].copy_from_slice(row);
        if let Err(e) = step.copy_from_host(stream, &host) {
            d.forget();
            return Err(e.into());
        }
        wrote.set(TargetRows::Step, pos - 1, 1, 1);
        Ok(())
    }

    /// A cut to `pos`, behind the stores' position: the point at `pos`
    /// planned for the next call ([`Body38::settle`]), the PLE history set
    /// back to the point's, every later point dropped, the MTP draft
    /// rejoining at `pos` where it can ([`Body38::draft_rejoins`]: its store
    /// cut there, the point's hidden row in the step arena,
    /// [`Body38::rejoin_draft`]) and its store forgotten where it cannot (its
    /// rows past `pos` belong to the branch the cut dropped), and a verify
    /// waiting dropped. Refused by name, nothing moved, at 0 (a reset) and
    /// where no point stands.
    fn cut(&mut self, stream: &CudaStream, pos: u32) -> Result<(), GpuError> {
        const WHAT_C: &str = "qwen4exp cut";
        if pos == 0 {
            return Err(GpuError::shape(
                WHAT_C,
                format!(
                    "back to position 0 from {}: the empty model is a reset",
                    self.held
                ),
            ));
        }
        let hist = self
            .ple_points
            .iter()
            .find(|&&(p, _)| p == pos)
            .map(|(_, h)| h.clone());
        let why = match &self.pending {
            None => "with no verify waiting".to_owned(),
            Some(p) => format!("below the verify of {} rows at {}", p.tokens.len(), p.pos0),
        };
        let refuse = |e: String| {
            GpuError::shape(
                WHAT_C,
                format!("back to position {pos} from {} {why}: {e}", self.held),
            )
        };
        let Some(hist) = hist else {
            return Err(refuse(format!(
                "no checkpoint stands there ({})",
                self.kept(pos, pos)
            )));
        };
        if hist.next_pos() != u64::from(pos) {
            return Err(refuse(format!(
                "the point's PLE history stands at {}",
                hist.next_pos()
            )));
        }
        self.ckpt
            .cut(pos, self.held)
            .map_err(|e| refuse(e.to_string()))?;
        let rejoin = self.rejoin_row(pos).map(<[f32]>::to_vec);
        self.ple.hist = hist;
        self.ple_points.retain(|&(p, _)| p <= pos);
        self.point_rows.retain(|&(p, _)| p <= pos);
        self.pending = None;
        self.staged = None;
        self.held = pos;
        self.wrote.cut(pos);
        let rejoined = rejoin.map(|row| self.rejoin_draft(stream, pos, &row));
        if !matches!(rejoined, Some(Ok(())))
            && let Some(d) = self.mtp.as_mut()
        {
            d.forget();
        }
        rejoined.unwrap_or(Ok(()))
    }

    /// What every save takes before its copy: the sequence state at `pos`,
    /// where the stores stand (a waiting cut carried out first), the draft's
    /// side read (its arenas' rows, blocking), the PLE history and the
    /// checkpoints' pick for `pos` ([`Checkpoints::save_points`]) as the
    /// state's host values, the stores lent, and `take` (the copy, on the
    /// engine stream or on a lane) run on them. The K/V planes' rows and the
    /// pooled keys' of positions below `pos`, the draft's
    /// store rows of the same positions, the recurrent stores at `pos` and at
    /// each picked point, each with its PLE history. Refused by name while a
    /// verify waits, after a call failed past its launch, when the history
    /// disagrees, and for a picked point with no PLE history.
    fn take_state<R>(
        &mut self,
        stream: &CudaStream,
        pos: u32,
        take: impl FnOnce(
            Identity,
            &Lent<'_, '_, '_>,
            History,
            Vec<(u32, History)>,
            &Checkpoints,
        ) -> Result<R, GpuError>,
    ) -> Result<(R, Option<DraftRows38>), GpuError> {
        const WHAT_S: &str = "qwen4exp snapshot";
        self.stores_at(pos)?;
        if let Some(p) = &self.pending {
            return Err(GpuError::shape(
                WHAT_S,
                format!(
                    "a snapshot at {pos} while the verify of {} rows at {} waits for its commit",
                    p.tokens.len(),
                    p.pos0
                ),
            ));
        }
        self.settle(stream)?;
        let current = self.ple.hist.clone();
        if current.next_pos() != u64::from(pos) {
            return Err(GpuError::shape(
                WHAT_S,
                format!(
                    "a snapshot at {pos} with the PLE history at {}",
                    current.next_pos()
                ),
            ));
        }
        let points = self
            .ckpt
            .save_points(pos)
            .into_iter()
            .map(|p| {
                self.ple_points
                    .iter()
                    .find(|&&(q, _)| q == p)
                    .map(|(_, h)| (p, h.clone()))
                    .ok_or_else(|| {
                        GpuError::shape(
                            WHAT_S,
                            format!("a snapshot at {pos} picks the point at {p}, which has no PLE history"),
                        )
                    })
            })
            .collect::<Result<Vec<(u32, History)>, GpuError>>()?;
        let lane = self.lane.lane();
        // The draft's side reads the arenas, none of the stores lent below,
        // and `Copied`'s drop holds its borrow to the end.
        let draft = self.draft_rows(stream, pos)?;
        if let Some(d) = self.mtp.as_mut() {
            d.fit_store(pos as usize)?;
        }
        let mut c = Copied::of(
            &mut self.stores,
            &mut self.ple_ring,
            lane,
            Some((self.ctx, pos as usize)),
            self.mtp.as_mut(),
        )?;
        let (positional, mut recurrent) = c.list_split();
        let lent = Lent {
            positional,
            recurrent: &mut recurrent,
        };
        let taken = take(self.who.clone(), &lent, current, points, &self.ckpt)?;
        Ok((taken, draft))
    }

    /// The sequence state at `pos` ([`Body38::take_state`]), copied on the
    /// engine stream.
    fn save_state(&mut self, stream: &CudaStream, pos: u32) -> Result<Seq38, GpuError> {
        let (state, draft) = self.take_state(stream, pos, |who, lent, current, points, ckpt| {
            SeqState::save(stream, who, pos, lent, current, points, ckpt)
        })?;
        Ok(Seq38 {
            state: Arc::new(state),
            draft,
        })
    }

    /// [`Body38::save_state`] with the copy on `lane`'s stream into its
    /// pinned staging ([`SeqState::save_async`]), the draft's store rows in
    /// it: the state and its copy in flight, or [`NoRoom`], nothing begun.
    /// The draft's arenas' rows are read here, blocking.
    fn save_state_async(
        &mut self,
        stream: &CudaStream,
        lane: &SaveLane,
        pos: u32,
    ) -> Result<Result<(Seq38, Pending<History>), NoRoom>, GpuError> {
        let (taken, draft) = self.take_state(stream, pos, |who, lent, current, points, ckpt| {
            let streams = Streams {
                lane,
                engine: stream,
            };
            SeqState::save_async(streams, who, pos, lent, current, points, ckpt)
        })?;
        Ok(taken.map(|pending| {
            let state = pending.shared();
            (Seq38 { state, draft }, pending)
        }))
    }

    /// The draft's side of a sequence state at `pos` ([`DraftRows38`]);
    /// `None` on a load without the draft: the positions the store holds and
    /// the step's and the pass's arena rows, each below `pos` (the ones a
    /// walk can read), and the hidden rows of the points the state carries
    /// ([`Checkpoints::save_points`], [`Body38::point_rows`]), which a put-back
    /// as a point's prefix rejoins the draft with. Refused by name when the
    /// store holds positions past `pos` or an arena's rows reach past it: a
    /// state's rows are the positions the model stands at, and a record past
    /// them is a verify's rejected tail or an anchor row the commit or the
    /// snapshot's caller left standing; neither is the state's to carry.
    /// Blocking.
    fn draft_rows(&self, stream: &CudaStream, pos: u32) -> Result<Option<DraftRows38>, GpuError> {
        const WHAT_D: &str = "qwen4exp snapshot";
        let Some(d) = self.mtp.as_ref() else {
            return Ok(None);
        };
        let held = d.held();
        let held = u32::try_from(held)
            .ok()
            .filter(|&held| held <= pos)
            .ok_or_else(|| {
                GpuError::shape(
                    WHAT_D,
                    format!("a state at {pos} of a draft store that holds {held} positions"),
                )
            })?;
        let wide = geo::STREAMS * geo::HIDDEN;
        let rows = |walk: TargetRows, most: usize| -> Result<(Held, Vec<f32>), GpuError> {
            let held = self.wrote.of(walk);
            let (first, count) = held.parts();
            let n = count as usize;
            if n > most || n > 0 && u64::from(first) + u64::from(count) > u64::from(pos) {
                return Err(GpuError::shape(
                    WHAT_D,
                    format!(
                        "the {walk:?} arena's {} at a state of {pos} positions",
                        held.shown()
                    ),
                ));
            }
            let v = match n * wide {
                0 => Vec::new(),
                len => {
                    let window =
                        crate::tensor::Window::<f32>::of(self.final_streams(walk), 0, len)?;
                    to_host(stream, &window)?
                }
            };
            Ok((held, v))
        };
        let step = rows(TargetRows::Step, 1)?;
        let pass = rows(TargetRows::Pass, PASS_ROWS)?;
        let picked = self.ckpt.save_points(pos);
        let point_rows = self
            .point_rows
            .iter()
            .filter(|(p, _)| picked.contains(p))
            .cloned()
            .collect();
        Ok(Some(DraftRows38 {
            held,
            step,
            pass,
            final_rows: [
                self.wrote.final_of(TargetRows::Step),
                self.wrote.final_of(TargetRows::Pass),
            ],
            point_rows,
        }))
    }

    /// Refused by name unless `d` fits this load standing at `pos`, nothing
    /// copied: the draft loaded, its store's positions and each arena's at
    /// or below `pos`, each arena's rows within its buffer (the step's one,
    /// the pass's [`PASS_ROWS`]) and of the streams' width, and each point's
    /// hidden row one row of that width at a point in `1..pos`.
    fn draft_fits(&self, d: &DraftRows38, pos: u32) -> Result<(), GpuError> {
        const WHAT_F: &str = "qwen4exp resume";
        if self.mtp.is_none() {
            return Err(GpuError::shape(
                WHAT_F,
                "the draft's side of a state put back on a load without the draft".to_owned(),
            ));
        }
        if d.held > pos {
            return Err(GpuError::shape(
                WHAT_F,
                format!("a draft store of {} positions put back at {pos}", d.held),
            ));
        }
        let wide = geo::STREAMS * geo::HIDDEN;
        for (what, (held, v), rows) in [("step", &d.step, 1usize), ("pass", &d.pass, PASS_ROWS)] {
            let (first, count) = held.parts();
            let n = count as usize;
            if n > rows
                || v.len() != n * wide
                || n > 0 && u64::from(first) + u64::from(count) > u64::from(pos)
            {
                return Err(GpuError::shape(
                    WHAT_F,
                    format!(
                        "the {what} arena's {} ({} values) put back into {rows} rows of {wide} \
                         at position {pos}",
                        held.shown(),
                        v.len()
                    ),
                ));
            }
        }
        if let Some((p, row)) = d
            .point_rows
            .iter()
            .find(|(p, row)| *p == 0 || *p >= pos || row.len() != wide)
        {
            return Err(GpuError::shape(
                WHAT_F,
                format!(
                    "the point at {p}'s hidden row of {} values put back at position {pos}",
                    row.len()
                ),
            ));
        }
        Ok(())
    }

    /// `d` put back on a model whose state of `n` positions is put back
    /// as its first `at` ([`Body38::draft_rows`]' rows): the step's and the
    /// pass's arenas their rows, the ubatch's none, and the draft's store,
    /// whose rows the state's positional rows put back, holding `d`'s
    /// positions when `at` is `n`. A cut to `at < n` is what a whole put-back
    /// and [`Body38::cut`] leave: the arenas' records cut to `at`, and the
    /// draft rejoining at `at` where `d`'s store holds every position below
    /// it and the point there carries its hidden row (the store counted at
    /// `at`, [`Mtp38::adopt`], and [`Body38::rejoin_draft`]), else the
    /// draft's store forgotten. Refused by name, as [`Body38::draft_fits`]
    /// refuses, before any copy. The arenas are zeroed on the card and only
    /// `d`'s rows copied in; those copies block.
    fn put_draft_rows(
        &mut self,
        stream: &CudaStream,
        d: &DraftRows38,
        n: u32,
        at: u32,
    ) -> Result<(), GpuError> {
        const WHAT_P: &str = "qwen4exp resume";
        self.draft_fits(d, n)?;
        let rejoin = d
            .point_rows
            .iter()
            .find(|&&(p, _)| p == at)
            .filter(|_| at < n && d.held >= at);
        let Body38 {
            plans,
            s,
            a,
            wa,
            mtp,
            wrote,
            ..
        } = &mut *self;
        for (walk, (_, v)) in [(TargetRows::Step, &d.step), (TargetRows::Pass, &d.pass)] {
            let arena = final_streams_mut(plans, [s, a, wa], walk);
            arena.zero_async(stream)?;
            if !v.is_empty() {
                WindowMut::<f32>::of_mut(arena, 0, v.len())?.copy_from_host(stream, v)?;
            }
        }
        let draft = mtp.as_mut().ok_or(GpuError::state(
            WHAT_P,
            "an MTP draft (Body38::open_placed_mtp)",
        ))?;
        if at == n {
            draft.adopt(d.held as usize)?;
        } else if rejoin.is_some() {
            draft.adopt(at as usize)?;
        } else {
            draft.forget();
        }
        *wrote = Wrote38 {
            step: d.step.0,
            pass: d.pass.0,
            ubatch: Held::NONE,
            final_rows: [d.final_rows[0], d.final_rows[1], Held::NONE],
        };
        wrote.cut(at);
        if let Some((_, row)) = rejoin {
            self.rejoin_draft(stream, at, row)?;
        }
        Ok(())
    }

    /// The first `ask` positions of `s` (or the nearest point below, `at`:
    /// [`SeqState::keep_point`]) put back on the empty model
    /// ([`SeqState::load_prefix`]): the state a whole put-back and a cut to
    /// `at` ([`Body38::cut`]) leave. The stores stand at `at`, the committed
    /// lane stamped there, the PLE history the point's, the points up to `at`
    /// the model's own with their hidden rows, and on a load with the draft
    /// its side cut to `at` ([`Body38::put_draft_rows`]). Only the rows of
    /// the first `at` positions go to the card. Returns `at`. Refused by
    /// name, nothing copied, for another model's state, a draft side the load
    /// does not take (a state with it onto a load without the draft, or the
    /// other way: the store layouts differ) or onto a model that is not
    /// empty.
    fn load_state(&mut self, stream: &CudaStream, s: &Seq38, ask: u32) -> Result<u32, GpuError> {
        const WHAT_L: &str = "qwen4exp resume";
        if self.held != 0 || self.pending.is_some() || self.staged.is_some() {
            return Err(GpuError::shape(
                WHAT_L,
                format!("a resume onto stores at {}: reset first", self.held),
            ));
        }
        let n = s.positions();
        match &s.draft {
            Some(d) => self.draft_fits(d, n)?,
            None if self.mtp.is_some() => {
                return Err(GpuError::shape(
                    WHAT_L,
                    "a state without the draft's side put back on a load with the draft (the \
                     store layouts differ)"
                        .to_owned(),
                ));
            }
            None => {}
        }
        let lane = self.lane.lane();
        // The reset preceding a put-back gave shared pages back; fit backs
        // the prefix on memory no other sequence maps.
        let kept = s.keep_point(ask) as usize;
        for st in &mut self.stores {
            st.fit(kept)?;
        }
        if let Some(d) = self.mtp.as_mut() {
            d.fit_store(kept)?;
        }
        let (at, cur, pts) = {
            let mut c = Copied::of(
                &mut self.stores,
                &mut self.ple_ring,
                lane,
                Some((self.ctx, n as usize)),
                self.mtp.as_mut(),
            )?;
            let (positional, mut recurrent) = c.list_split();
            let mut lent = Lent {
                positional,
                recurrent: &mut recurrent,
            };
            let r = s
                .state
                .load_prefix(stream, &self.who, &mut lent, &mut self.ckpt, ask)?;
            (r.at, r.current.clone(), r.points.to_vec())
        };
        let off = std::iter::once((at, &cur))
            .chain(pts.iter().map(|(p, h)| (*p, h)))
            .find(|&(p, h)| h.next_pos() != u64::from(p));
        if let Some((p, h)) = off {
            self.ckpt.clear();
            return Err(GpuError::shape(
                WHAT_L,
                format!(
                    "a state whose PLE history at {p} stands at {}",
                    h.next_pos()
                ),
            ));
        }
        if let Some(d) = &s.draft {
            self.put_draft_rows(stream, d, n, at)?;
            self.point_rows = d
                .point_rows
                .iter()
                .filter(|&&(p, _)| p <= at)
                .cloned()
                .collect();
        }
        self.restamp(stream, lane, at)?;
        self.ple.hist = cur;
        self.ple_points = pts;
        self.held = at;
        Ok(at)
    }

    /// The first positions of `src`, another slot's sequence, up to its
    /// share point for `ask` ([`Slot38::share_point`], `at`) put on the empty
    /// model: the state a resume of a save `src` took at `at` leaves, with
    /// the selecting stores' rows below `at` held on `src`'s pages
    /// ([`Store38::share`]: every page of whole rows below `at` mapped, the
    /// page where they end copied) and the recurrent stores, the PLE ring and
    /// the raw key rings put back from `src`'s point there
    /// ([`Checkpoints::load_point`]). The stores stand at `at`, the committed
    /// lane stamped there, the PLE history the point's; on a load with the
    /// draft its store shares `src`'s rows the same way and the draft
    /// rejoins at `at` from the point's hidden row
    /// ([`Body38::rejoin_draft`]). The model's own point at `at` is taken
    /// last ([`Body38::take_mark`]), so a cut back to `at` stands. Returns
    /// `at` and the bytes the stores map onto `src`'s pages and copied.
    /// Refused by name, nothing moved, onto a model that is not empty and
    /// where `src` has no share point at or below `ask`.
    fn share_state(
        &mut self,
        stream: &CudaStream,
        src: &Slot38,
        ask: u32,
    ) -> Result<(u32, SharedPages), GpuError> {
        if self.held != 0 || self.pending.is_some() || self.staged.is_some() {
            return Err(GpuError::shape(
                WHAT_SHARE,
                format!("a share onto stores at {}: reset first", self.held),
            ));
        }
        let at = src.share_point(ask);
        let (Some(h), row) = (point_at(&src.ple_points, at), point_at(&src.point_rows, at)) else {
            return Err(GpuError::shape(
                WHAT_SHARE,
                format!(
                    "no point of the other sequence at or below {ask} to take (points {:?}, \
                     {} held)",
                    src.ckpt.positions(),
                    src.held
                ),
            ));
        };
        let mut moved = SharedPages::default();
        for (st, from) in self.stores.iter_mut().zip(&src.stores) {
            moved = moved + st.share(from, at as usize)?;
        }
        if let (Some(d), Some(from)) = (self.mtp.as_mut(), src.draft.as_ref()) {
            moved = moved + d.share_store(from, at as usize)?;
        }
        let lane = self.lane.lane();
        let mut c = Copied::of(&mut self.stores, &mut self.ple_ring, lane, None, None)?;
        src.ckpt.load_point(stream, at, &mut c.list())?;
        drop(c);
        self.restamp(stream, lane, at)?;
        self.ple.hist = src.ple_points[h].1.clone();
        self.held = at;
        if self.mtp.is_some() {
            let row = row.ok_or(GpuError::state(WHAT_SHARE, "the point's hidden row"))?;
            self.rejoin_draft(stream, at, &src.point_rows[row].1)?;
        }
        self.take_mark(stream, at)?;
        Ok((at, moved))
    }

    /// Plant a failure for the next call once its plan and copies are made,
    /// before its launch: the call fails and moves nothing. Gate use.
    pub fn plant_before_launch(&mut self) {
        self.plant = Some(Plant::BeforeLaunch);
    }

    /// Plant a failure for the next call once its chain has run and its
    /// host legs been served: the call fails with the stores past it. Gate
    /// use.
    pub fn plant_after_launch(&mut self) {
        self.plant = Some(Plant::AfterLaunch);
    }

    /// The PLE row ids the last fill (or a commit's replay) named, a
    /// position's ids after the one before. Gate use.
    #[must_use]
    pub fn ple_rows(&self) -> &[u32] {
        &self.ple.ids[..self.ple.named]
    }

    /// Stage a call of `rows` from `pos` whose PLE rows `fill` returned the
    /// history past: [`Body38::launch`] takes it. The stores' rows through
    /// the call's last position are backed first ([`Store38::fit`]), so every
    /// call that writes a row, a long reply's steps and a verify's rows
    /// among them, finds its pages mapped, and the pages it writes from
    /// `pos` on are this sequence's alone ([`Store38::own_from`]).
    fn stage(
        &mut self,
        pos: u32,
        rows: usize,
        hist: History,
        verify: Option<Vec<u32>>,
    ) -> Result<(), GpuError> {
        for s in &mut self.stores {
            s.fit(pos as usize + rows)?;
            s.own_from(pos as usize)?;
        }
        self.staged = Some(Staged38 {
            pos,
            rows: launch_u32(WHAT, "positions", rows)?,
            hist,
            verify,
        });
        Ok(())
    }

    /// The staged call of `rows` from `pos` launched: the stores' count past
    /// its rows, the PLE history past them, and a verify's commit waiting
    /// with the history as it stood before. Only the last `read` rows are
    /// recorded as final hidden rows. Refused by name, moving nothing,
    /// when no call of those rows from there is staged. A launch that fails
    /// after this is counted as run: the refusal that follows is named,
    /// where a count left behind would run the rows twice.
    fn launch(&mut self, pos: u32, rows: usize, read: usize) -> Result<(), GpuError> {
        if read > rows {
            return Err(GpuError::shape(
                WHAT,
                format!("{read} final hidden rows of a call of {rows}"),
            ));
        }
        let read = launch_u32(WHAT, "final hidden rows", read)?;
        let s = self
            .staged
            .take_if(|s| s.pos == pos && s.pos == self.held && s.rows as usize == rows);
        let Some(s) = s else {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a launch of {rows} rows from {pos} with the stores at {} and no such call \
                     planned",
                    self.held
                ),
            ));
        };
        let before = std::mem::replace(&mut self.ple.hist, s.hist);
        self.held = s.pos + s.rows;
        self.wrote.set(self.last_walk, s.pos, s.rows, read);
        if let Some(tokens) = s.verify {
            self.pending = Some(Pending38 {
                pos0: s.pos,
                tokens,
                hist: before,
                ple_due: false,
            });
        }
        Ok(())
    }

    /// The waiting verify's PLE rows, when their arrival is due
    /// ([`StagedRows`]): read on this thread and decoded
    /// ([`PleHost::read_named`]), staged, and the arrival's flag raised on
    /// every path, an error included.
    fn deliver_ple(&mut self) -> Result<(), GpuError> {
        let Some(p) = self.pending.as_mut().filter(|p| p.ple_due) else {
            return Ok(());
        };
        p.ple_due = false;
        let (pos, n) = (p.pos0, p.tokens.len());
        let rows = self.ple.read_named(pos, n).and_then(|()| {
            self.ple
                .e
                .get(..n * geo::HIDDEN)
                .ok_or_else(|| GpuError::shape(WHAT, format!("{n} PLE positions at once")))
        });
        self.ple_arrival.deliver(0, |staging| {
            let rows = rows?;
            let held = staging.len();
            staging
                .get_mut(..rows.len())
                .ok_or_else(|| {
                    GpuError::shape(
                        WHAT,
                        format!("{} PLE values into a staging of {held}", rows.len()),
                    )
                })?
                .copy_from_slice(rows);
            Ok(())
        })
    }

    /// The call's position `pos` is where the stores stand and `n` more fit
    /// them, and no verify waits for its commit, else refused by name.
    fn check_next(&self, pos: u32, n: usize) -> Result<(), GpuError> {
        self.stores_at(pos)?;
        if let Some(p) = &self.pending {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{n} positions from {pos} while the verify of {} rows at {} waits for its \
                     commit (GpuModel::rollback to the first position not kept)",
                    p.tokens.len(),
                    p.pos0
                ),
            ));
        }
        if pos as usize + n > self.ctx {
            return Err(GpuError::shape(
                WHAT,
                format!("{n} positions from {pos}, in stores of {}", self.ctx),
            ));
        }
        Ok(())
    }

    /// Plan an eager pass of `tokens` (1..=8) at `pos`, where the stores
    /// stand: the map check ([`MapCheck::refuse`]), its PLE rows, its
    /// record and its rows' copy to the pass arena, staged for
    /// [`Body38::walk_pass`]'s launch.
    fn plan_pass(&mut self, stream: &CudaStream, tokens: &[u32], pos: u32) -> Result<(), GpuError> {
        self.card_leg.refuse(Walk38::Pass)?;
        let n = tokens.len();
        if self.taps.is_some() {
            return Err(GpuError::state(WHAT, "layer taps off (a pass writes none)"));
        }
        self.check_next(pos, n)?;
        self.settle(stream)?;
        let hist = self.ple.fill(pos, tokens)?;
        self.last_walk = TargetRows::Pass;
        self.rp.write(stream, tokens, pos)?;
        self.ple.upload(stream, &mut self.a.ple.e, n)?;
        self.stage(pos, n, hist, None)
    }

    /// Launch the planned pass of `m` rows from `pos` through the batch
    /// port, then the head of its last row into `head` when given; with no
    /// head and `finals` (a reader of the pass's final rows: the call's tap,
    /// or the checkpoint after it, [`Body38::take_mark`]) the last layer's
    /// combine alone over them ([`Pass38::finals`]).
    fn walk_pass(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        m: usize,
        pos: u32,
        head: Option<&mut Head>,
        finals: bool,
    ) -> Result<(), GpuError> {
        if !(1..=self.a.rows).contains(&m) {
            return Err(GpuError::shape(
                WHAT,
                format!("a pass of {m} rows on an arena of {}", self.a.rows),
            ));
        }
        planted(&mut self.plant, Plant::BeforeLaunch)?;
        let read = if head.is_some() || finals { m } else { 0 };
        self.launch(pos, m, read)?;
        let io = Io {
            lane: Some(self.lane.word()),
            ..self.rp.io(m)?
        };
        let Body38 {
            hybrid,
            plans,
            stores,
            ple_ring,
            rope,
            seq_rope,
            a,
            pass_hsum,
            k,
            branch,
            slots,
            card,
            eps,
            ctx,
            plant,
            ..
        } = self;
        let mut prog = Pass38 {
            branch,
            p: Parts38 {
                c: Ctx38 {
                    gpu,
                    w,
                    k,
                    eps: *eps,
                    table: rope_table(rope, seq_rope.as_ref()),
                    ctx: *ctx,
                },
                plans,
                seqs: Seqs38::One {
                    stores,
                    ple_ring,
                    io: &io,
                },
                s: a,
                m,
                cur: 0,
                taps: None,
                slots,
                card,
                tier: None,
                each: false,
            },
        };
        let mut leg = BatchLeg::new(gpu.stream(), hybrid, pass_hsum, PASS_ROWS);
        prog.walk(&mut leg)?;
        planted(plant, Plant::AfterLaunch)?;
        match head {
            Some(h) => prog.head(h),
            None if finals => prog.finals(),
            None => Ok(()),
        }
    }
}

impl Body38 {
    /// Plan a ubatch of `tokens` (1..=the load's size) at `pos`, where the
    /// stores stand: the map check ([`MapCheck::refuse`]), the armed
    /// route taps' rows and the planted routes' positions, its PLE rows (on
    /// a walk that serves no host leg named and on their read, the walk
    /// copying them in at its PLE site), its record and its rows' copy to
    /// the ubatch arena, staged for [`Body38::walk_gemm`]'s launch. Each
    /// refusal comes before anything moves. With the timing armed, the
    /// plan's host wall is the prompt's prologue.
    fn plan_gemm(&mut self, stream: &CudaStream, tokens: &[u32], pos: u32) -> Result<(), GpuError> {
        let t0 = self.wide_timing.as_ref().map(|_| Instant::now());
        let r = self.stage_gemm(stream, tokens, pos);
        if let (Some(t0), Some(w)) = (t0, self.wide_timing.as_mut()) {
            w.add_prologue(nanos(t0.elapsed()));
        }
        r
    }

    /// [`Body38::plan_gemm`]'s staging, untimed.
    fn stage_gemm(
        &mut self,
        stream: &CudaStream,
        tokens: &[u32],
        pos: u32,
    ) -> Result<(), GpuError> {
        self.card_leg.refuse(Walk38::Ubatch)?;
        let n = tokens.len();
        if self.taps.is_some() {
            return Err(GpuError::state(
                WHAT,
                "layer taps off (a ubatch writes none)",
            ));
        }
        if let Some(t) = &self.wide.taps
            && n > t.rows
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a ubatch of {n} rows past the {} armed route tap rows \
                     (set_ubatch_route_taps)",
                    t.rows
                ),
            ));
        }
        if let Some(f) = &self.wide.force {
            f.covers(pos as usize, n)?;
        }
        self.check_next(pos, n)?;
        self.settle(stream)?;
        let unit = pos as usize..pos as usize + n;
        let spans = window_spans(self.media.iter().flat_map(MediaRows::positions), &unit);
        // A walk that serves no host leg (every layer's routed experts on the
        // card, `Card38::hostless`, and no expert tier) enqueues nothing that
        // waits for this thread, so it takes its rows at its PLE site and
        // their read runs under the layers before it; any other walk copies
        // them in now.
        self.ple_late = self.tier.is_none() && self.card.all_hostless();
        let hist = match self.ple_late {
            true => self.ple.name_spanned(pos, tokens, &spans)?,
            false => self.ple.fill_spanned(pos, tokens, &spans)?,
        };
        self.last_walk = TargetRows::Ubatch;
        self.wr.write(stream, tokens, pos)?;
        if !self.ple_late {
            self.ple.upload(stream, &mut self.wa.ple.e, n)?;
        }
        self.stage(pos, n, hist, None)
    }

    /// Launch the planned ubatch of `m` rows from `pos` through the batch
    /// port at its width, its last `read` rows final after it (the head's
    /// row, the point row the checkpoint after it keeps, [`Body38::take_mark`],
    /// or every row the call's tap reads; the last layer runs the others only
    /// as far as the stores need, `WideParts::kept`), then the head of its
    /// last row into `head` when given; with a row past the head's read, the
    /// last layer's combine alone first, over every unit the head's mix
    /// leaves ([`Gemm38::finals`]). Refused by name past the arena's rows, and
    /// with the head and no row read.
    fn walk_gemm(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        m: usize,
        pos: u32,
        head: Option<&mut Head>,
        read: usize,
    ) -> Result<(), GpuError> {
        if !(1..=self.wide.rows).contains(&m) {
            return Err(GpuError::shape(
                WHAT,
                format!("a ubatch of {m} rows on an arena of {}", self.wide.rows),
            ));
        }
        let heads = usize::from(head.is_some());
        if read < heads || read > m {
            return Err(GpuError::shape(
                WHAT,
                format!("a ubatch of {m} rows read at its last {read}, the head at {heads}"),
            ));
        }
        let dense = dense_rows(pos as usize, m, self.ctx)?;
        planted(&mut self.plant, Plant::BeforeLaunch)?;
        self.launch(pos, m, read)?;
        let late = std::mem::take(&mut self.ple_late);
        let io = Io {
            lane: Some(self.lane.word()),
            ..self.wr.io()
        };
        let Body38 {
            hybrid,
            plans,
            stores,
            ple_ring,
            ple,
            rope,
            seq_rope,
            wa,
            wide_hsum,
            wide,
            wide_timing,
            stream,
            media,
            k,
            slots,
            card,
            tier,
            eps,
            ctx,
            plant,
            end_mark,
            marker,
            keep_last,
            ..
        } = self;
        let done = marker
            .as_ref()
            .filter(|_| end_mark.is_some())
            .map(|m| m.done.as_slice());
        let mut rows_in = |arena: &mut Ple38| -> Result<(), GpuError> {
            ple.land(gpu.stream(), &k.ple, arena, m)
        };
        let mut prog = Gemm38 {
            p: WideParts {
                c: Ctx38 {
                    gpu,
                    w,
                    k,
                    eps: *eps,
                    table: rope_table(rope, seq_rope.as_ref()),
                    ctx: *ctx,
                },
                plans,
                stores,
                ple_ring,
                s: wa,
                x: wide,
                io: &io,
                slots,
                card,
                tier: tier.as_ref(),
                m,
                pos0: pos as usize,
                dense,
                read,
                stream,
                media: media.as_mut(),
                ple_rows: late.then_some(&mut rows_in as &mut PleRowsIn<'_>),
                done,
                keep: std::mem::take(keep_last),
            },
        };
        let cap = prog.p.x.rows;
        let mut leg = BatchLeg::new(gpu.stream(), hybrid, wide_hsum, cap);
        leg.set_timer(wide_timing.as_mut());
        prog.walk(&mut leg)?;
        if prog.p.ple_rows.is_some() {
            return Err(GpuError::state(
                WHAT,
                "a ubatch walk past its PLE rows' site with the rows not copied in",
            ));
        }
        planted(plant, Plant::AfterLaunch)?;
        if read > heads {
            prog.finals(head.is_some())?;
        }
        match head {
            Some(h) => prog.head(h),
            None => Ok(()),
        }
    }

    /// The path a prompt of `n` ids runs on this load: `path` resolved
    /// ([`Prompt38::resolve`]), but `Auto` runs ubatches at every length on a
    /// load with an expert tier, whose pass walk has no tier leg (`card38`'s
    /// `MapCheck` refuses it by name).
    #[must_use]
    pub fn resolve_prompt(&self, path: Prompt38, n: usize) -> Prompt38 {
        match path {
            Prompt38::Auto if self.tier.is_some() => Prompt38::Gemm,
            p => p.resolve(n),
        }
    }

    /// Positions one ubatch takes at most: the load's size
    /// (`BLOOMERY_QWEN3_UBATCH`, at most the stores' positions).
    #[must_use]
    pub fn ubatch_rows(&self) -> usize {
        self.wide.rows
    }

    /// The ubatches a prompt of `n` positions runs as ([`Prompt38::Gemm`]):
    /// their sizes in order, each the load's size but the last.
    pub fn ubatch_cut(&self, n: usize) -> impl Iterator<Item = usize> + use<> {
        ubatch_cut(n, self.wide.rows)
    }

    /// Every walk's map check reads `map` instead of the tier's own until a
    /// call with `None` takes it back or the model is reset: a gate's
    /// stand-in for a placement the walks refuse by name — an expert on a
    /// tier card at every walk with no tier leg. The check alone reads it;
    /// the walks run the loaded map. Gate use.
    pub fn plant_slot_map(&mut self, map: Option<SlotMap>) -> Result<(), GpuError> {
        self.card_leg = MapCheck::of(
            map.as_ref().unwrap_or(self.hybrid.slots()),
            self.tier.is_some(),
        )?;
        Ok(())
    }

    /// Layers the expert tier holds routed experts of; 0 on a load without
    /// a tier.
    #[must_use]
    pub fn tier_layers(&self) -> usize {
        self.tier.as_ref().map_or(0, TierSide38::layers)
    }

    /// Layers whose routed experts the card leg runs: the loaded slot map's
    /// layers with an expert on the card.
    #[must_use]
    pub fn card_layers(&self) -> usize {
        self.card.card_layers()
    }

    /// The card layers' routed stack types, each with its layers (the card
    /// leg's own summary, `card38`'s `card_stacks`): what the load line
    /// names beside the card layers.
    #[must_use]
    pub fn card_stacks(&self) -> String {
        self.card.card_stacks()
    }

    /// The last ubatch walk's rows of each selecting layer: those that
    /// select nothing (the prefill flash's) and those that select;
    /// `None` before a walk reached a selecting layer since the load or the
    /// last reset. Gate use.
    #[must_use]
    pub fn ubatch_split(&self) -> Option<(usize, usize)> {
        self.wide.split
    }

    /// Arm the ubatch walk's route taps for units of up to `rows` tokens
    /// (`0` disarms them): each layer's router logits and routed ids, which
    /// [`Body38::ubatch_route_taps`] reads back; a ubatch of more rows is
    /// refused by name before it moves anything. Load-time allocation; gate
    /// use.
    pub fn set_ubatch_route_taps(&mut self, gpu: &Gpu, rows: usize) -> Result<(), GpuError> {
        self.wide.taps = if rows == 0 {
            None
        } else {
            Some(WideTaps::new(
                gpu.stream(),
                self.plans.len(),
                rows,
                self.wa.route.dims().logits(),
            )?)
        };
        Ok(())
    }

    /// The last ubatch walk's route taps for the positions `pos0 .. pos0 +
    /// m`: position `pos0 + t`'s route at layer `l` at `[t][l]` — the
    /// router's own logits, and the ids the walk ran (a planted route's,
    /// [`Body38::plant_ubatch_routes`]). Blocking; refused by name when the
    /// taps are not armed, or when the last walk since they were armed did
    /// not run exactly those positions (a prompt of several ubatches leaves
    /// its last one's).
    pub fn ubatch_route_taps(
        &self,
        gpu: &Gpu,
        pos0: usize,
        m: usize,
    ) -> Result<Vec<Vec<RouteTap>>, GpuError> {
        let taps = self.wide.taps.as_ref().ok_or(GpuError::state(
            WHAT,
            "armed ubatch route taps (set_ubatch_route_taps)",
        ))?;
        route_taps_host(taps, gpu.stream(), pos0, m)
    }

    /// Plant the routes the ubatch walks take for the positions `pos0 ..
    /// pos0 + routes.len()` (`None` takes them back; a reset does too):
    /// position `pos0 + t`'s router logits at layer `l`, `routes[t][l]`'s
    /// [`geo::EXPERTS`] values, stand in for the router's own, and the
    /// routing launch runs again over them — so the walk takes their ids and
    /// weights bit for bit where a step routes from those logits, and no
    /// token's experts can differ from theirs. The shared expert's gate
    /// stays the router's own. A ubatch with a position outside them is
    /// refused by name before it moves anything. Load-time allocation; gate
    /// use.
    pub fn plant_ubatch_routes(
        &mut self,
        gpu: &Gpu,
        plant: Option<(usize, &[Vec<RouteTap>])>,
    ) -> Result<(), GpuError> {
        self.wide.force = match plant {
            None => None,
            Some((pos0, routes)) => Some(WideForce::new(
                gpu.stream(),
                self.plans.len(),
                pos0,
                routes,
            )?),
        };
        Ok(())
    }
}

/// The ubatch sizes of a prompt of `n` positions in ubatches of at most
/// `rows`: [`Body38::ubatch_cut`], and the cut [`GpuModel::prompt38`] runs.
fn ubatch_cut(n: usize, rows: usize) -> impl Iterator<Item = usize> {
    (0..n.div_ceil(rows)).map(move |i| rows.min(n - i * rows))
}

/// The units of a prompt call of `n` ids from position `from` on a path of
/// units of at most `width` rows: each run up to the next of the call's
/// checkpoint marks past `from` (`marks`, ascending, the call's end the last
/// of them; none on a load that takes none), else up to the call's end, cut
/// as [`ubatch_cut`] cuts it. A unit's rows, and whether a checkpoint is
/// taken after it. The one cut of a call: its walks run it
/// ([`GpuModel::feed38`]) and its PLE rows are read ahead by it
/// ([`PleHost::prompt_start`]).
fn call_units(from: u32, n: usize, width: usize, marks: &[u32]) -> Vec<(usize, bool)> {
    let (from, end) = (from as usize, from as usize + n);
    let mut stops: Vec<(usize, bool)> = marks
        .iter()
        .map(|&p| (p as usize, true))
        .filter(|&(p, _)| p > from)
        .collect();
    if stops.last().is_none_or(|&(p, _)| p < end) {
        stops.push((end, false));
    }
    let (mut units, mut at) = (Vec::new(), from);
    for (stop, marked) in stops {
        let k = (stop - at).div_ceil(width);
        units.extend(
            ubatch_cut(stop - at, width)
                .enumerate()
                .map(|(i, m)| (m, marked && i + 1 == k)),
        );
        at = stop;
    }
    units
}

/// The units of a closing call of `n` ids from `from` whose marks (as
/// [`call_units`] takes them) end at its last id's position, that id folded
/// into the call's last ubatch: [`call_units`] without that end mark, when
/// the last ubatch then holds a row before the id. `None` otherwise: the
/// mark cuts the call there as any mark does.
fn fold_units(from: u32, n: usize, width: usize, marks: &[u32]) -> Option<Vec<(usize, bool)>> {
    let (&end, inner) = marks.split_last()?;
    if end as usize + 1 != from as usize + n {
        return None;
    }
    let units = call_units(from, n, width, inner);
    units.last().is_some_and(|&(m, _)| m >= 2).then_some(units)
}

impl GpuModel<Body38> {
    /// Arm (or disarm) the per-layer taps: after each layer the step copies
    /// its streams into the layer's tap, which [`Body38::taps`] reads back,
    /// and after each router its logits and ids ([`Body38::route_taps`]).
    /// Every captured chain is dropped first, the mode kept: a capture
    /// records the copies into the taps armed at its time, so a replay after
    /// a change would write freed memory or leave the new taps unwritten.
    /// Load-time allocation.
    pub fn set_layer_taps(&mut self, on: bool) -> Result<(), GpuError> {
        self.drop_captures();
        let (gpu, _, body) = self.body_parts("qwen4exp set_layer_taps")?;
        body.set_taps(gpu, on)
    }

    /// Arm (or disarm) the ubatch walk's timing: one event a part boundary a
    /// layer, the serve's host times and the prompt's walls, which
    /// [`GpuModel::take_prompt38_stats`] reads. Load-time allocation; unarmed,
    /// a walk records no event, keeps no time and waits for nothing beyond
    /// its own serves.
    pub fn set_prompt38_stats(&mut self, on: bool) -> Result<(), GpuError> {
        let (gpu, _, body) = self.body_parts("qwen4exp set_prompt38_stats")?;
        body.wide_timing = if on {
            Some(PromptTiming::new(gpu.context(), body.plans.len(), 2)?)
        } else {
            None
        };
        Ok(())
    }

    /// The prompt's ubatch-walk stats, taken — the next prompt starts clean.
    /// `Ok(None)` with the timing unarmed or no ubatch walked (a prompt by
    /// steps or passes keeps none).
    pub fn take_prompt38_stats(&mut self) -> Result<Option<PromptStats>, GpuError> {
        let (_, _, body) = self.body_parts("qwen4exp take_prompt38_stats")?;
        Ok(body.wide_timing.as_mut().and_then(PromptTiming::take))
    }

    /// Whether a closing call ([`Body38::set_close`]) of `n` ids on `path`
    /// from where the model stands runs its last id in the ubatch walk: the
    /// call is ubatches, and its last ubatch holds a row before that id
    /// ([`fold_units`]) or the call takes no checkpoint. Otherwise the end
    /// mark before the last id would leave it a walk of its own, which a
    /// step runs faster; a serve seat then feeds the call less that id and
    /// steps it.
    pub fn closes_by_ubatch(&self, n: usize, path: Prompt38) -> Result<bool, GpuError> {
        const WHAT_C: &str = "qwen4exp closing call";
        let pos = self.pos();
        let body = self.body(WHAT_C)?;
        if body.resolve_prompt(path, n) != Prompt38::Gemm {
            return Ok(false);
        }
        if !body.marks {
            return Ok(true);
        }
        let end = launch_u32(WHAT_C, "positions", pos as usize + n - 1)?;
        let marks = body.ckpt.marks(pos, end);
        Ok(fold_units(pos, n, body.wide.rows, &marks).is_some())
    }

    /// Feed `tokens` from where the model stands by `path` and return the
    /// greedy next token after the last one: captured steps, one a position,
    /// eager passes of up to eight positions, or ubatches of up to
    /// [`Body38::ubatch_rows`], the last pass or ubatch ending in its last
    /// row's head. Steps and passes leave every position's state and logits
    /// bit for bit; ubatches leave a position's bits a function of its own
    /// inputs, the same whatever the ubatches' cut. A prompt past the stores, or with an id the embedding or the
    /// PLE hash does not take (one past the vocabulary, the image
    /// placeholder), is refused before any launch, so either path leaves the
    /// model where it stood; the layer taps must be off for passes and
    /// ubatches.
    pub fn prompt38(&mut self, tokens: &[u32], path: Prompt38) -> Result<u32, GpuError> {
        self.prompt38_with(tokens, path, None)
    }

    /// [`GpuModel::prompt38`] with a tap on the prompt's units: `sink` is
    /// called after each unit the path runs — every token's step, every
    /// pass, every ubatch — with the arena its rows' final streams sit in,
    /// the unit's first position and its rows, before the next unit
    /// overwrites them. A row the head's mix does not cover (a pass or a
    /// ubatch before the last, a split walk's first unit) gets the last
    /// layer's combine alone before the sink reads it; only a tap reads such
    /// a row, so a call without one runs no such combine. A `None` sink is
    /// [`GpuModel::prompt38`] itself: the steps run as one call of the whole
    /// prompt; a sink's per-token steps leave the same bits, a head readback
    /// a token.
    ///
    /// The call is one residency pass ([`crate::host::swap`]): its boundary
    /// before it and none inside — the passes and the ubatches run through
    /// `run_rows`, which opens none — and 0 rows kept after it, whatever it
    /// returned: the rule counts decode rows only, and the batch service
    /// notes no id. A step-fed prompt is refused by name under a running
    /// machine: each prompt id would end a decode pass the rule counts. With
    /// no machine the steps feed opens no pass around its steps: each id is
    /// a step with its own boundary and kept row.
    /// Under host streaming ([`Body38::set_xstream`]) a ubatch-fed call
    /// is also the machine's call: its walks move the pool
    /// ([`super::wide38`]'s module doc), its placement stays for the decode
    /// after it, and a call that fails leaves each layer at the set it
    /// started with. A pass-fed call, and one of fewer ids than the pick's
    /// floor, opens none: its rows give an expert fewer counts than the
    /// floor, so its pick would admit nothing.
    pub fn prompt38_with(
        &mut self,
        tokens: &[u32],
        path: Prompt38,
        sink: Option<&mut Prompt38Sink<'_>>,
    ) -> Result<u32, GpuError> {
        self.prompt38_read(tokens, path, sink, true)
    }

    /// Feed prompt positions and hidden-row taps without producing a token.
    pub fn prefill38_with(
        &mut self,
        tokens: &[u32],
        path: Prompt38,
        sink: Option<&mut Prompt38Sink<'_>>,
    ) -> Result<(), GpuError> {
        self.prompt38_read(tokens, path, sink, false).map(|_| ())
    }

    fn prompt38_read(
        &mut self,
        tokens: &[u32],
        path: Prompt38,
        sink: Option<&mut Prompt38Sink<'_>>,
        head: bool,
    ) -> Result<u32, GpuError> {
        let resolved = self.prompt38_checked(tokens, path)?;
        // The steps feed is no pass of its own: each id is a step with its
        // boundary and its kept row, so none is opened around them.
        let one_pass = resolved != Prompt38::Step;
        if one_pass {
            self.pass_boundary()?;
        }
        let fed = self.prompt38_fed(tokens, resolved, sink, head);
        let kept = match one_pass {
            true => self.keep_rows(KeptRows::prefix(0), PassKind::Prompt),
            false => Ok(()),
        };
        // The call's own error first: a keep refused after a failed call is
        // its echo.
        let next = fed?;
        kept?;
        Ok(next)
    }

    /// Prepare the media call's rotation table and span staging before `run`, and finish
    /// both after it. The draft joins inside this frame, so its first key uses the image's
    /// rotations even when a continuing call starts with an image.
    pub fn media38_frame<T>(
        &mut self,
        ids: &[u32],
        spans: &[ImageSpan<'_>],
        run: impl FnOnce(&mut Self) -> Result<T, GpuError>,
    ) -> Result<T, GpuError> {
        media_call(self, "qwen4exp prompt_media", (ids, spans), |_| Ok(()), run)
    }

    /// [`GpuModel::prompt38_with`] on [`Prompt38::Gemm`] inside the
    /// residency pass its caller opened and keeps (`prompt_pass`: a media
    /// call, which the session wraps as it wraps every body's prompt call):
    /// the same checks and the same walk, with no boundary before it and no
    /// kept count after it.
    fn prompt38_in_pass(&mut self, tokens: &[u32]) -> Result<u32, GpuError> {
        let resolved = self.prompt38_checked(tokens, Prompt38::Gemm)?;
        self.prompt38_fed(tokens, resolved, None, true)
    }

    /// [`GpuModel::prompt38_with`]'s checks, all before any launch: the
    /// model stays where it stood on a refusal. The path the call runs as.
    fn prompt38_checked(&mut self, tokens: &[u32], path: Prompt38) -> Result<Prompt38, GpuError> {
        const WHAT_P: &str = "qwen4exp prompt";
        if tokens.is_empty() {
            return Err(GpuError::shape(WHAT_P, "empty token slice"));
        }
        self.refuse_if_poisoned(WHAT_P)?;
        let body = self.body(WHAT_P)?;
        let (pos, ctx) = (self.pos() as usize, body.ctx());
        body.refuse_inside_image(WHAT_P, self.pos())?;
        super::refuse_past_vocab(WHAT_P, tokens, body.vocab)?;
        let spans = window_spans(
            body.media.iter().flat_map(MediaRows::positions),
            &(pos..pos + tokens.len()),
        );
        ple_takes(WHAT_P, body.ple.hash.window(), tokens, &spans)?;
        if pos + tokens.len() > ctx {
            return Err(GpuError::shape(
                WHAT_P,
                format!(
                    "a prompt of {} ids from position {pos} passes the stores' {ctx} positions",
                    tokens.len()
                ),
            ));
        }
        let resolved = body.resolve_prompt(path, tokens.len());
        if resolved == Prompt38::Step && body.hybrid().swap().is_some() {
            return Err(GpuError::shape(
                WHAT_P,
                "a step-fed prompt beside the residency machine: each prompt id \
                 would end a decode pass the rule counts, where the prompt call keeps \
                 0; feed the prompt by passes or ubatches (a `pass`, `gemm` or `auto` \
                 path) or load with BLOOMERY_RESIDENCY=off",
            ));
        }
        Ok(resolved)
    }

    /// The call's streaming and its units on the resolved `path`, after the
    /// checks ([`GpuModel::prompt38_checked`]) and, for a path that is a
    /// pass, the residency boundary: the argmax after the last id.
    fn prompt38_fed(
        &mut self,
        tokens: &[u32],
        resolved: Prompt38,
        sink: Option<&mut Prompt38Sink<'_>>,
        head: bool,
    ) -> Result<u32, GpuError> {
        const WHAT_P: &str = "qwen4exp prompt";
        // A successful call before a decode history ends with each layer
        // back at the set it started with: the decode has been routing
        // experts of its own, and the call's picks — the prompt's hottest —
        // put the pool back where they found it rather than leave the decode
        // to churn back at the flip cap. Without a history (a fresh server)
        // the call's placement stays, the gain it was run for. The history
        // is read once, here: the call keeps no row, so it does not move
        // until the end, and the walk returns each layer in the call's last
        // unit on it.
        let history = self.body(WHAT_P)?.hybrid().call_history();
        let r = self
            .stream_begin(resolved, tokens.len(), history)
            .and_then(|()| self.feed_marked(tokens, resolved, sink, head));
        // The call's own error first: an end refused after a failed call is
        // its echo. A failed call's streaming ends with each layer back at
        // the set it started with too.
        let ended = self.stream_end(r.is_ok() && !history);
        if r.is_err() {
            // A unit that failed past its launch recorded rows its walk did
            // not finish: no record from the model's position on stands.
            let pos = self.pos();
            self.body_parts(WHAT_P)?.2.wrote.cut(pos);
        }
        let next = r?;
        ended?;
        Ok(next)
    }

    /// Open the prompt call's streaming ([`PromptStream::begin`]) with the
    /// body's own facts: the call's ids and end position, whether the path
    /// is the ubatch walk's, the pick's no-probe floor ([`STREAM_FLOOR`]),
    /// the Q4 kind's card tail rate ([`XSTREAM_TAIL_US_PER_CARD_COL`]; none
    /// on a kind no record has priced — the walk then balances without it)
    /// and the map's layers. A pass runs at most [`PASS_ROWS`] rows, under
    /// the floor, so no pass-fed call opens one.
    fn stream_begin(&mut self, path: Prompt38, n: usize, returns: bool) -> Result<(), GpuError> {
        const _: () = assert!(PASS_ROWS < STREAM_FLOOR as usize);
        const WHAT_B: &str = "qwen4exp prompt streaming";
        let end = self.pos() as usize + n;
        let (gpu, _, body) = self.body_parts(WHAT_B)?;
        // The file's own tail rate: the Q4 kind's measured one, none on a
        // kind no record has priced — the walk then balances without it.
        let tail = if body.routed_gate == GgmlType::Q4_K {
            XSTREAM_TAIL_US_PER_CARD_COL
        } else {
            0.0
        };
        body.stream.begin(
            &mut body.hybrid,
            gpu.stream(),
            path == Prompt38::Gemm,
            n,
            end,
            returns,
            STREAM_FLOOR,
            tail,
            body.plans.len(),
        )
    }

    /// End the prompt call's streaming ([`PromptStream::end`]): the ends'
    /// reports kept for [`Body38::take_stream_records`] and
    /// [`Body38::take_xstream_records`]. [`Body38::prompt38_with`]'s call
    /// passes `kept` false before a decode history
    /// ([`crate::host::HostTier::call_history`]): the decode's experts go
    /// back on the card.
    fn stream_end(&mut self, kept: bool) -> Result<(), GpuError> {
        let (gpu, _, body) = self.body_parts("qwen4exp prompt streaming")?;
        body.stream.end(&mut body.hybrid, gpu.stream(), kept)
    }

    /// [`GpuModel::prompt38_with`]'s units ([`GpuModel::feed38`]); while the
    /// load takes checkpoints ([`Body38::set_checkpoints`]) a checkpoint is
    /// taken at each of the call's marks (`Checkpoints::marks`: its start
    /// when it continues a sequence, every [`CHECKPOINT_EVERY`] positions
    /// inside it, its end), the units cut there, each on the path the whole
    /// call resolved to: a ubatch walk leaves the same bits cut there or not,
    /// as a pass and a step do. The argmax after the last id.
    fn feed_marked(
        &mut self,
        tokens: &[u32],
        path: Prompt38,
        mut sink: Option<&mut Prompt38Sink<'_>>,
        head: bool,
    ) -> Result<u32, GpuError> {
        const WHAT_M: &str = "qwen4exp prompt checkpoints";
        let from = self.pos();
        let to = launch_u32(WHAT_M, "positions", tokens.len())
            .ok()
            .and_then(|n| from.checked_add(n))
            .ok_or_else(|| GpuError::shape(WHAT_M, format!("{} ids from {from}", tokens.len())))?;
        let body = self.body(WHAT_M)?;
        // A closing call's marks are the call's less its last id.
        let end = to - u32::from(body.close);
        let marks = match body.marks {
            true => body.ckpt.marks(from, end),
            false => Vec::new(),
        };
        if path != Prompt38::Step {
            return self.feed38(tokens, path, &marks, sink, head);
        }
        if marks.is_empty() {
            return self.feed_steps(tokens, sink);
        }
        let (mut next, mut at) = (None, from);
        for mark in marks {
            if mark > at {
                let run = &tokens[(at - from) as usize..(mark - from) as usize];
                next = Some(self.feed_steps(run, sink.as_deref_mut())?);
                at = mark;
            }
            self.mark_here(WHAT_M)?;
        }
        if at < to {
            next = Some(self.feed_steps(&tokens[(at - from) as usize..], sink)?);
        }
        next.ok_or(GpuError::state(
            WHAT_M,
            "a token after the call's last mark",
        ))
    }

    /// The checkpoint at the model's position ([`Body38::take_mark`]).
    fn mark_here(&mut self, what: &'static str) -> Result<(), GpuError> {
        let pos = self.pos();
        let (gpu, _, body) = self.body_parts(what)?;
        body.take_mark(gpu.stream(), pos)
    }

    /// [`GpuModel::feed_marked`]'s steps: every token's body, one readback
    /// at the end; under a tap a readback a token, the tap reading each
    /// token's row where the step left it.
    fn feed_steps(
        &mut self,
        tokens: &[u32],
        sink: Option<&mut Prompt38Sink<'_>>,
    ) -> Result<u32, GpuError> {
        let Some(f) = sink else {
            return self.step(tokens);
        };
        let mut next = None;
        for &token in tokens {
            next = Some(self.step(&[token])?);
            let first = self.pos() - 1;
            f(self, TargetRows::Step, first, 1)?;
        }
        next.ok_or(GpuError::state(
            "qwen4exp prompt",
            "a token read after the last step",
        ))
    }

    /// [`GpuModel::prompt38_with`]'s passes or ubatches once its checks have
    /// passed, its path resolved and its residency pass open: the call's
    /// units ([`call_units`]), a checkpoint taken at each of `marks`
    /// ([`GpuModel::feed_marked`]), and the head after the last unit alone
    /// when `head` asks for it.
    /// A unit a checkpoint follows inside the call ends without the head: the
    /// last layer's combine over its last row when the checkpoint keeps it
    /// (the draft's point row, [`Body38::take_mark`]), over every row when the
    /// tap reads them, and the fault word read before the checkpoint copies a
    /// store. Each unit reads the rows those readers name and no other
    /// ([`Body38::walk_gemm`]'s `read`).
    ///
    /// A closing call ([`Body38::set_close`]) by ubatches whose marks end
    /// before its last id runs that id as the last row of its last ubatch
    /// (the fold) when the ubatch holds a row before it ([`fold_units`]): the
    /// ubatch keeps its last delta chunk out of the lanes
    /// ([`Body38::keep_last`]), and the end mark opens on it and is sealed a
    /// row before where the walk stands ([`Body38::seal_close_mark`]), the
    /// point's row read with the head's.
    /// Otherwise the mark cuts the call there as any mark does, and the last
    /// id is a ubatch of its own: the same bits, each row's a function of its
    /// own inputs whatever ubatch holds it.
    fn feed38(
        &mut self,
        tokens: &[u32],
        path: Prompt38,
        marks: &[u32],
        mut sink: Option<&mut Prompt38Sink<'_>>,
        head: bool,
    ) -> Result<u32, GpuError> {
        const WHAT_P: &str = "qwen4exp prompt";
        let tapped = sink.is_some();
        let pos0 = self.pos();
        let body = self.body(WHAT_P)?;
        let (width, walk) = match path {
            Prompt38::Gemm => (body.wide.rows, TargetRows::Ubatch),
            Prompt38::Pass => (PASS_ROWS, TargetRows::Pass),
            Prompt38::Step | Prompt38::Auto => {
                return Err(GpuError::state(
                    WHAT_P,
                    "a path of passes or ubatches `Prompt38::resolve` settles",
                ));
            }
        };
        let point_row = body.mtp.is_some();
        let n = tokens.len();
        let mut units = call_units(pos0, n, width, marks);
        let mut fold = None;
        let folded = fold_units(pos0, n, width, marks).filter(|_| body.close);
        if let (Some(folded), Prompt38::Gemm) = (folded, path) {
            units = folded;
            let mut hist = body.ple.hist.clone();
            for &t in &tokens[..n - 1] {
                hist.push(t);
            }
            fold = Some((pos0 + launch_u32(WHAT_P, "positions", n - 1)?, hist));
        }
        if path == Prompt38::Gemm {
            // The call's PLE rows are read ahead: the first ubatch's fill
            // reads its own, each later one's runs under the ubatch before
            // it, across the call's checkpoints ([`PleHost::prompt_start`]).
            let (_, _, body) = self.body_parts(WHAT_P)?;
            let spans: Vec<_> = body.media.iter().flat_map(MediaRows::positions).collect();
            body.ple
                .prompt_start(pos0, tokens, units.iter().map(|&(m, _)| m), &spans)?;
        }
        if marks.first() == Some(&pos0) {
            self.mark_here(WHAT_P)?;
        }
        let (mut next, mut at) = (None, 0);
        for (i, &(m, marked)) in units.iter().enumerate() {
            let chunk = &tokens[at..at + m];
            at += m;
            let last = i + 1 == units.len();
            let point = marked && !last;
            let headed = last && head;
            let folded = fold.as_ref().filter(|_| last).map(|(e, _)| *e);
            let read = match tapped {
                true => m,
                false => {
                    usize::from(headed || marked && point_row)
                        + usize::from(folded.is_some() && point_row)
                }
            };
            let pos = self.pos();
            {
                let (gpu, _, body) = self.body_parts(WHAT_P)?;
                match path {
                    Prompt38::Gemm => {
                        body.plan_gemm(gpu.stream(), chunk, pos)?;
                        body.keep_last = folded.is_some();
                        if marked {
                            let to = launch_u32(WHAT_P, "positions", pos as usize + m)?;
                            body.open_end_mark(gpu.stream(), pos, to)?;
                        }
                        if let Some(e) = folded {
                            body.open_end_mark(gpu.stream(), pos, e)?;
                        }
                    }
                    _ => body.plan_pass(gpu.stream(), chunk, pos)?,
                }
            }
            let walked = self.run_rows(m, WHAT_P, |gpu, w, body, head, pos| {
                let out = headed.then_some(head);
                match path {
                    Prompt38::Gemm => {
                        body.walk_gemm(gpu, w, m, pos, out, read)?;
                        body.stream.walk_done();
                        body.copy_end_mark()?;
                    }
                    _ => body.walk_pass(gpu, w, m, pos, out, read > usize::from(headed))?,
                }
                if (point || last && !headed)
                    && let Some(f) = gpu.fault()?
                {
                    return Err(GpuError::fault(WHAT_P, f));
                }
                Ok(headed)
            });
            let r = walked.and_then(|n| {
                next = n;
                match sink.as_deref_mut() {
                    Some(f) => f(self, walk, pos, m),
                    None => Ok(()),
                }
            });
            if let Err(e) = r {
                // The unit failed before its mark: an end mark it opened is
                // given back, the unit's error reported.
                if let Ok((gpu, _, body)) = self.body_parts(WHAT_P) {
                    let _ = body.drop_end_mark(gpu.stream());
                }
                return Err(e);
            }
            if marked {
                self.mark_here(WHAT_P)?;
            }
        }
        if let Some((e, hist)) = fold {
            let (gpu, _, body) = self.body_parts(WHAT_P)?;
            body.seal_close_mark(gpu.stream(), e, hist)?;
        }
        if !head {
            return Ok(0);
        }
        next.ok_or(GpuError::state(
            WHAT_P,
            "a token read after the call's last unit",
        ))
    }
}

impl crate::host::xstream::XStreamBody for Body38 {
    /// [`Body38::set_xstream`]: the Qwen3.8 row's guards, its costs row and
    /// its card's arming, over the shared start.
    fn set_xstream(&mut self, gpu: &Gpu, mode: XMode) -> Result<(), GpuError> {
        Body38::set_xstream(self, gpu, mode)
    }

    /// [`Body38::xstream_word`].
    fn xstream_word(&self) -> String {
        Body38::xstream_word(self)
    }
}

/// Each target arena's [`Held`]: the body's record of where the step, the
/// pass (a verify's rows too) and the ubatch walk left their rows — what a
/// draft's walk checks its read positions and completion against, and a
/// sequence state carries of the two arenas a parked refresh reads.
#[derive(Clone, Copy, Debug, Default)]
struct Wrote38 {
    step: Held,
    pass: Held,
    ubatch: Held,
    /// Completed positions, separate from the arena's row-to-position mapping.
    final_rows: [Held; 3],
}

impl Wrote38 {
    /// The arena `walk`'s rows hold `rows` rows at `pos ..`, as the call that
    /// last wrote it left them; only its last `read` rows are final streams.
    fn set(&mut self, walk: TargetRows, pos: u32, rows: u32, read: u32) {
        let (held, complete) = match walk {
            TargetRows::Step => (&mut self.step, &mut self.final_rows[0]),
            TargetRows::Pass => (&mut self.pass, &mut self.final_rows[1]),
            TargetRows::Ubatch => (&mut self.ubatch, &mut self.final_rows[2]),
        };
        *held = Held::at(pos, rows);
        *complete = Held::at(pos + rows - read, read);
    }

    /// Every arena's positions from `pos` on taken back.
    fn cut(&mut self, pos: u32) {
        for h in [&mut self.step, &mut self.pass, &mut self.ubatch] {
            h.cut(pos);
        }
        for h in &mut self.final_rows {
            h.cut(pos);
        }
    }

    fn of(&self, walk: TargetRows) -> Held {
        match walk {
            TargetRows::Step => self.step,
            TargetRows::Pass => self.pass,
            TargetRows::Ubatch => self.ubatch,
        }
    }

    fn final_of(&self, walk: TargetRows) -> Held {
        self.final_rows[match walk {
            TargetRows::Step => 0,
            TargetRows::Pass => 1,
            TargetRows::Ubatch => 2,
        }]
    }
}

/// The streams' buffer of `walk`'s arena (the step's, the pass's, the
/// ubatch walk's) that holds the target's hidden rows once the walk left
/// them final ([`TargetRows`]): buffer [`final_res`].
fn final_streams<'b>(
    plans: &[Layer38],
    [s, a, wa]: [&'b Arena38; 3],
    walk: TargetRows,
) -> &'b DeviceBuffer<f32> {
    let cur = final_res(plans);
    let arena = match walk {
        TargetRows::Step => s,
        TargetRows::Pass => a,
        TargetRows::Ubatch => wa,
    };
    &arena.res[cur]
}

/// [`final_streams`] mutable, for a state's arena rows put back
/// ([`Body38::put_draft_rows`]).
fn final_streams_mut<'b>(
    plans: &[Layer38],
    [s, a, wa]: [&'b mut Arena38; 3],
    walk: TargetRows,
) -> &'b mut DeviceBuffer<f32> {
    let cur = final_res(plans);
    let arena = match walk {
        TargetRows::Step => s,
        TargetRows::Pass => a,
        TargetRows::Ubatch => wa,
    };
    &mut arena.res[cur]
}

/// The target's streams a [`Hidden::Target`] feed of a walk or a chain's
/// refresh names, over the body's arenas, with the positions the arena's
/// rows hold ([`Wrote38`]): the walk refuses rows that hold other
/// positions, or none.
fn mtp_target<'b>(
    plans: &[Layer38],
    arenas: [&'b Arena38; 3],
    wrote: &Wrote38,
    feed: DraftFeed<'_, TargetRows>,
) -> Option<Target38<'b>> {
    match feed {
        DraftFeed::Rows(Feed {
            hidden: Hidden::Target { walk, .. },
            ..
        }) => Some(Target38 {
            buf: final_streams(plans, arenas, walk),
            held: wrote.of(walk),
            final_rows: wrote.final_of(walk),
        }),
        _ => None,
    }
}

/// The draft's context over the body's own pieces, `table` the rotation
/// table the live sequence's rope launches read ([`rope_table`]): a media
/// load's own, so the draft's keys turn by an image's positions as the
/// target's do.
fn mtp_ctx<'a>(
    gpu: &'a Gpu,
    w: &'a Weights,
    k: &'a Kernels38,
    eps: f32,
    table: &'a DeviceBuffer<f32>,
) -> MtpCtx<'a> {
    MtpCtx {
        gpu,
        tw: w,
        k,
        eps,
        table,
    }
}

/// A prompt call's tap ([`GpuModel::prompt38_with`]): called after each unit
/// the path runs, with the arena its rows' final streams sit in, the unit's
/// first position and its rows.
pub type Prompt38Sink<'a> =
    dyn FnMut(&mut GpuModel<Body38>, TargetRows, u32, usize) -> Result<(), GpuError> + 'a;

/// One slot's chain of a round of several slots' chains
/// ([`GpuModel::mtp_chain_slots`]): the slot, the refresh, the own walks.
pub type SlotChain38<'a> = (usize, Feed<'a, TargetRows>, usize);

/// The MTP draft's calls ([`Mtp38::run`]); each is refused by name on a load
/// without a draft.
impl GpuModel<Body38> {
    /// One walk of the draft (module doc of `mtp38`): `feed`'s rows into
    /// `head`, eager or captured, and each row's token and probability. A
    /// poisoned model is refused; a fault the walk raises is its error and
    /// stays on the card's word, which the target's next readback names.
    pub fn mtp_draft(
        &mut self,
        feed: DraftFeed<'_, TargetRows>,
        head: DraftHead,
        mode: WalkMode,
    ) -> Result<MtpDraft, GpuError> {
        const WHAT_D: &str = "qwen4exp mtp_draft";
        self.refuse_if_poisoned(WHAT_D)?;
        let (gpu, w, body) = self.body_parts(WHAT_D)?;
        let r = body.mtp_run(gpu, w, feed, head, mode);
        // A fault the draft raised poisons the model as the target's would.
        self.note_fault(WHAT_D, r)
    }

    /// One walk of the draft with no readback ([`Mtp38::run`]'s checks and
    /// launches): the prompt's warmup rows, whose tokens are not asked for.
    /// A fault the walk raises stays on the card's fault word, which the
    /// next readback — a chain's, the target's — names.
    pub fn mtp_walk(
        &mut self,
        feed: DraftFeed<'_, TargetRows>,
        head: DraftHead,
        mode: WalkMode,
    ) -> Result<(), GpuError> {
        const WHAT_D: &str = "qwen4exp mtp_walk";
        self.refuse_if_poisoned(WHAT_D)?;
        let (gpu, w, body) = self.body_parts(WHAT_D)?;
        let r = body.mtp_run_walk(gpu, w, feed, head, mode).map(|_| ());
        self.note_fault(WHAT_D, r)
    }

    /// One store walk of several resident sequences' rows (module doc of
    /// `mtp38`): each `(slot, feed)`'s rows checked and staged on its slot as
    /// [`GpuModel::mtp_walk`] checks a store walk there, then the walk's
    /// launches once over every slot's rows, the embedding and the append a
    /// slot, each slot's into its own store. Each slot's store and the
    /// positions it holds come out bit for bit as its own store walk leaves
    /// them, and the selected slot stays selected. Refused by name: a slot
    /// named twice, rows past the store walk's
    /// [`MTP_STORE_ROWS`](super::mtp38::MTP_STORE_ROWS), a poisoned
    /// model, and as `mtp_walk` refuses each slot's walk, before anything but
    /// the walk's scratch moves. A fault a launch raises stays on the card's
    /// fault word, which the next readback names.
    pub fn mtp_store_slots(
        &mut self,
        walks: &[(usize, Feed<'_, TargetRows>)],
        head: DraftHead,
    ) -> Result<(), GpuError> {
        const WHAT_D: &str = "qwen4exp mtp_store_slots";
        self.refuse_if_poisoned(WHAT_D)?;
        if let Some((i, &(slot, _))) = walks
            .iter()
            .enumerate()
            .find(|&(i, (slot, _))| walks[..i].iter().any(|(s, _)| s == slot))
        {
            return Err(GpuError::shape(
                WHAT_D,
                format!("slot {slot} named again at walk {i}: one walk a slot"),
            ));
        }
        let home = self.selected();
        let r = self.store_slots_at(walks, head);
        let back = self.select_slot(home);
        self.note_fault(WHAT_D, r.and(back))
    }

    /// [`GpuModel::mtp_store_slots`]'s walk, its checks done: every slot's
    /// rows staged with that slot selected, the launches over all of them,
    /// then each slot's append with that slot selected.
    fn store_slots_at(
        &mut self,
        walks: &[(usize, Feed<'_, TargetRows>)],
        head: DraftHead,
    ) -> Result<(), GpuError> {
        const WHAT_D: &str = "qwen4exp mtp_store_slots";
        let mut blocks = Vec::with_capacity(walks.len());
        let mut ids = Vec::new();
        for &(slot, feed) in walks {
            self.select_slot(slot)?;
            let (gpu, _, body) = self.body_parts(WHAT_D)?;
            let Body38 {
                mtp,
                plans,
                s,
                a,
                wa,
                wrote,
                ..
            } = body;
            let d = mtp.as_mut().ok_or(GpuError::state(
                WHAT_D,
                "an MTP draft (Body38::open_placed_mtp)",
            ))?;
            let target = mtp_target(plans, [s, a, wa], wrote, DraftFeed::Rows(feed));
            blocks.push(d.stage_store(gpu.stream(), target, feed, head, ids.len())?);
            ids.extend_from_slice(feed.tokens);
        }
        if blocks.is_empty() {
            return Ok(());
        }
        let (gpu, w, body) = self.body_parts(WHAT_D)?;
        body.mtp_with(gpu, w, |d, c| d.store_rows(c, &ids, &blocks, false))?;
        for (&(slot, _), &b) in walks.iter().zip(&blocks) {
            self.select_slot(slot)?;
            let (gpu, w, body) = self.body_parts(WHAT_D)?;
            body.mtp_with(gpu, w, |d, c| d.append_store(c, b))?;
        }
        Ok(())
    }

    /// Fill the draft walk's scratch with NaNs without changing its inbox or stored state.
    pub fn poison_mtp_scratch(&mut self) -> Result<(), GpuError> {
        let (gpu, _, body) = self.body_parts("qwen4exp poison_mtp_scratch")?;
        body.draft_mut()?.poison_scratch(gpu.stream())
    }

    /// Arm (or disarm) the draft's taps ([`Mtp38::set_taps`]); its captured
    /// walks are dropped first. Load-time allocation.
    pub fn set_mtp_taps(&mut self, on: bool) -> Result<(), GpuError> {
        let (gpu, _, body) = self.body_parts("qwen4exp set_mtp_taps")?;
        body.draft_mut()?.set_taps(gpu.stream(), on)
    }

    /// One window's chain of the draft (module doc of `mtp38`): `refresh`'s
    /// rows — the target's kept rows with the target's hidden rows, a
    /// [`DraftFeed::Rows`] feed — walked, then `own` own walks, each reading
    /// the walk before it on the card, into `head`, eager or captured, and
    /// the proposal's ids (1 + `own` of them) and their probabilities read
    /// back once. A poisoned model is refused; a fault any walk of the chain
    /// raised is its error and poisons the model as the target's would.
    pub fn mtp_chain(
        &mut self,
        refresh: DraftFeed<'_, TargetRows>,
        own: usize,
        head: DraftHead,
        mode: WalkMode,
    ) -> Result<MtpDraft, GpuError> {
        const WHAT_D: &str = "qwen4exp mtp_chain";
        self.refuse_if_poisoned(WHAT_D)?;
        let (gpu, w, body) = self.body_parts(WHAT_D)?;
        let r = body.mtp_run_chain(gpu, w, refresh, own, head, mode);
        // A fault a walk of the chain raised poisons the model as the
        // target's would.
        self.note_fault(WHAT_D, r)
    }

    /// Several slots' window chains of the draft at once, one readback a
    /// round: each `(slot, refresh, own)` the chain [`GpuModel::mtp_chain`]
    /// runs on that slot, the chains' walks of each depth run as one walk
    /// over every slot's rows — its attention each slot's over its own
    /// store — and each slot's proposal, probabilities and store come out
    /// as its own chain leaves them, each launch computing a row from that
    /// row's inputs alone. Chains past one walk's rows or readback run in
    /// rounds of them, in order, and more chains than one call binds
    /// ([`MAX_PASS_ROWS`]) in calls of that many. Every walk runs launch by
    /// launch: one walk's launches address every slot's store. The model
    /// stands with slot 0 selected (`GpuModel::on_slots`). Refused by name:
    /// a slot named twice, a poisoned model, and as `mtp_chain` refuses each
    /// chain, before its round's first walk. Earlier chains in the round may
    /// already be staged, and rounds before it have run. A fault any walk
    /// raised poisons every slot its call named.
    pub fn mtp_chain_slots(
        &mut self,
        chains: &[SlotChain38<'_>],
        head: DraftHead,
        mode: WalkMode,
    ) -> Result<Vec<MtpDraft>, GpuError> {
        const WHAT_D: &str = "qwen4exp mtp_chain_slots";
        self.refuse_if_poisoned(WHAT_D)?;
        for (i, &(slot, ..)) in chains.iter().enumerate() {
            if chains[..i].iter().any(|&(seen, ..)| seen == slot) {
                return Err(slot_twice(WHAT_D, slot));
            }
        }
        if chains.len() > MAX_PASS_ROWS {
            let mut out = Vec::with_capacity(chains.len());
            for part in chains.chunks(MAX_PASS_ROWS) {
                out.extend(self.mtp_chain_slots(part, head, mode)?);
            }
            return Ok(out);
        }
        let slots: Vec<usize> = chains.iter().map(|&(slot, ..)| slot).collect();
        self.on_slots(&slots, WHAT_D, |gpu, w, body, parked| {
            let mut seqs = Vec::with_capacity(chains.len());
            let mut next = 0..parked.len();
            for &(slot, ..) in chains {
                seqs.push(match slot {
                    0 => None,
                    _ => Some(next.next().ok_or(GpuError::state(
                        WHAT_D,
                        "a parked sequence for every slot but slot 0",
                    ))?),
                });
            }
            let mut out = Vec::with_capacity(chains.len());
            let mut start = 0;
            while start < chains.len() {
                let (mut rows, mut pairs, mut end) = (0, 0, start);
                while let Some((_, feed, own)) = chains.get(end) {
                    let (r, p) = (feed.tokens.len(), 1 + own);
                    if end > start && (rows + r > MTP_ROWS || pairs + p > CHAIN_PAIRS) {
                        break;
                    }
                    (rows, pairs, end) = (rows + r, pairs + p, end + 1);
                }
                out.extend(body.mtp_chain_round(
                    gpu,
                    w,
                    parked,
                    (&chains[start..end], &seqs[start..end]),
                    head,
                    mode,
                )?);
                start = end;
            }
            Ok(out)
        })
    }

    /// The last draft walk's taps ([`Mtp38::taps`]). Blocking.
    pub fn mtp_taps(&self) -> Result<MtpTaps, GpuError> {
        let body = self.body("qwen4exp mtp_taps")?;
        body.draft_ref()?.taps(self.gpu().stream())
    }

    /// The last draft walk's `l_out` rows ([`Mtp38::l_out`]). Blocking.
    pub fn mtp_l_out(&self) -> Result<Vec<f32>, GpuError> {
        let body = self.body("qwen4exp mtp_l_out")?;
        body.draft_ref()?.l_out(self.gpu().stream())
    }

    /// The last draft walk's head logits and the head's rows
    /// ([`Mtp38::logits`]). Blocking.
    pub fn mtp_logits(&self) -> Result<(Vec<f32>, usize), GpuError> {
        let body = self.body("qwen4exp mtp_logits")?;
        body.draft_ref()?.logits(self.gpu().stream())
    }

    /// The target's hidden rows after a walk of `walk`'s arena
    /// ([`Body38::target_streams`]). Blocking.
    pub fn target_streams(&self, walk: TargetRows, rows: usize) -> Result<Vec<f32>, GpuError> {
        let body = self.body("qwen4exp target_streams")?;
        body.target_streams(self.gpu(), walk, rows)
    }
}

impl GpuModel<Body38> {
    /// The sequence state on the host ([`Seq38`], `checkpoint::saved`): the
    /// positional rows below the model's position (on a load with the draft,
    /// its store's rows among them), the recurrent stores there and at the
    /// selected checkpoints below it, each with its PLE history.
    /// A waiting cut is carried out first. Refused by name on a poisoned
    /// model, while a verify waits for its commit, and after a call failed
    /// past its launch.
    pub fn seq_save(&mut self) -> Result<Seq38, GpuError> {
        const WHAT_S: &str = "qwen4exp snapshot";
        self.refuse_if_poisoned(WHAT_S)?;
        let pos = self.pos();
        let (gpu, _, body) = self.body_parts(WHAT_S)?;
        body.save_state(gpu.stream(), pos)
    }

    /// The state [`Seq38::put`] wrote, read for this load, the model
    /// untouched: refused by name for another tag or format, a state of
    /// another model ([`SeqState::take`]), and a PLE history of another
    /// length than the load's or with an id the load's hash refuses. The put back ([`GpuModel::seq_resume`])
    /// holds the rest to the stores.
    pub fn seq_take(&self, r: &mut bytes::Take<'_>) -> Result<Seq38, Fail> {
        let body = self.body("qwen4exp state read").map_err(Fail::Refused)?;
        let tag = r.vals::<u8>(STATE_TAG.len(), "the tag")?;
        let format: u32 = r.one("the format")?;
        if tag != STATE_TAG || format != STATE_FORMAT {
            return Err(refused(format!(
                "a stream of tag {:?} and format {format}, not a Qwen3.8 state of format \
                 {STATE_FORMAT}",
                String::from_utf8_lossy(&tag)
            )));
        }
        let state = SeqState::<History>::take(r, &body.who)?;
        let n_gram = body.ple.hist.tokens().len();
        for h in state.values() {
            check_ple_history(h, n_gram, body.ple.hash.window())?;
        }
        let draft = match r.one::<u8>("the draft's side")? {
            0 => None,
            1 => Some(DraftRows38::take(r)?),
            c => return Err(refused(format!("a draft's side of code {c}"))),
        };
        Ok(Seq38 {
            state: Arc::new(state),
            draft,
        })
    }

    /// The host arrays a sequence save would hold, without reading the card.
    pub fn seq_save_bytes(&mut self) -> Result<u64, GpuError> {
        const WHAT_S: &str = "qwen4exp snapshot size";
        let pos = self.pos();
        let (_, _, b) = self.body_parts(WHAT_S)?;
        let points = b.ckpt.save_points(pos);
        let draft = if b.mtp.is_some() {
            (b.wrote.step.parts().1 as usize)
                .checked_add(b.wrote.pass.parts().1 as usize)
                .and_then(|n| n.checked_mul(geo::STREAMS))
                .and_then(|n| n.checked_mul(geo::HIDDEN))
                .and_then(|n| {
                    b.point_rows
                        .iter()
                        .filter(|(p, _)| points.contains(p))
                        .try_fold(n, |n, (_, r)| n.checked_add(r.len()))
                })
                .and_then(|n| n.checked_mul(size_of::<f32>()))
                .ok_or_else(|| GpuError::shape(WHAT_S, "draft host bytes pass the address space"))?
        } else {
            0
        };
        let mut stores = Copied::of(
            &mut b.stores,
            &mut b.ple_ring,
            b.lane.lane(),
            Some((b.ctx, pos as usize)),
            b.mtp.as_mut(),
        )?;
        let (positional, mut recurrent) = stores.list_split();
        Lent {
            positional,
            recurrent: &mut recurrent,
        }
        .bytes(points.len())?
        .checked_add(draft as u64)
        .ok_or_else(|| GpuError::shape(WHAT_S, "host bytes pass u64"))
    }

    /// [`GpuModel::seq_save`] whose copy runs on `lane`'s stream into its
    /// pinned staging and whose call returns once it is enqueued
    /// ([`SeqState::save_async`]): the state and its copy in flight, which
    /// the model's rows are not cut, reset or replaced before; or [`NoRoom`],
    /// nothing begun, when the staging holds no free extent for it. The
    /// draft's store rows are in the copy; its arenas' rows are read here,
    /// blocking. Refused by name as [`GpuModel::seq_save`] refuses.
    pub fn seq_save_async(
        &mut self,
        lane: &SaveLane,
    ) -> Result<Result<(Seq38, Pending<History>), NoRoom>, GpuError> {
        const WHAT_S: &str = "qwen4exp snapshot";
        self.refuse_if_poisoned(WHAT_S)?;
        let pos = self.pos();
        let (gpu, _, body) = self.body_parts(WHAT_S)?;
        body.save_state_async(gpu.stream(), lane, pos)
    }

    /// A save lane on the model's card sized to `limit` pinned bytes of
    /// states ([`SaveLane::new`]): load-time only.
    pub fn seq_lane(&mut self, limit: u64) -> Result<SaveLane, GpuError> {
        let (gpu, _, _) = self.body_parts("qwen4exp save lane")?;
        SaveLane::new(gpu.context(), limit)
    }

    /// Replace the sequence with `s`, which [`GpuModel::seq_save`] took of
    /// this load: a reset unless the model is empty already (a reset just
    /// before, or the load), then — as a pass of `s`'s positions whose work
    /// is the copies — the state put back; the model stands at its
    /// positions, its checkpoints the points `s` carried and no other, and
    /// the calls after it are bit for bit those after the state was taken.
    /// Refused by name, nothing copied, for another model's state (its
    /// file, card or context), and while a pass of several slots waits for
    /// its commit.
    pub fn seq_resume(&mut self, s: &Seq38) -> Result<(), GpuError> {
        if s.positions() == 0 {
            return self.seq_reset_for("qwen4exp resume");
        }
        self.seq_resume_prefix(s, s.positions()).map(drop)
    }

    /// The model reset unless it is empty already: what every resume and
    /// share starts from. Refused by name, nothing reset, while a pass of
    /// several slots waits for its commit: the reset would drop the pass the
    /// commit keeps rows of.
    fn seq_reset_for(&mut self, what: &'static str) -> Result<(), GpuError> {
        self.refuse_if_slots_wait(what)?;
        let b = self.body(what)?;
        let empty = self.pos() == 0
            && b.held == 0
            && b.pending.is_none()
            && b.staged.is_none()
            && b.ckpt.positions().is_empty()
            && !b.ckpt.pending();
        if !empty {
            self.reset()?;
        }
        Ok(())
    }

    /// [`GpuModel::seq_resume`] of the first `n` positions of `s` the state
    /// can be cut to ([`Seq38::keep_point`]): the model stands at that many,
    /// which it returns, as a whole resume and a rollback to it leave it
    /// (the draft rejoined there where the point carries its hidden row and
    /// the state's store reaches it, else its store forgotten; its arenas'
    /// records cut), with only those positions' rows moved to the card
    /// ([`SeqState::load_prefix`]). Refused by name as
    /// [`GpuModel::seq_resume`] refuses, and for a prefix of no positions
    /// (reset the model instead). `s`'s copy must have landed.
    pub fn seq_resume_prefix(&mut self, s: &Seq38, n: u32) -> Result<u32, GpuError> {
        const WHAT_R: &str = "qwen4exp resume";
        self.seq_reset_for(WHAT_R)?;
        let at = s.keep_point(n);
        if at == 0 {
            return Err(GpuError::shape(
                WHAT_R,
                format!(
                    "a prefix of 0 positions of a state of {} (asked {n}): reset the model instead",
                    s.positions()
                ),
            ));
        }
        self.run_rows(at as usize, WHAT_R, |gpu, _, body, _, _| {
            let put = body.load_state(gpu.stream(), s, n)?;
            if put != at {
                return Err(GpuError::shape(
                    WHAT_R,
                    format!("a prefix of {at} positions put back as {put}"),
                ));
            }
            Ok(false)
        })?;
        Ok(at)
    }
}

impl GpuModel<Body38> {
    /// The most positions of the first `n` the selected slot can take from
    /// slot `from` ([`GpuModel::seq_share_prefix`]): the nearest point of
    /// `from`'s sequence at or below `n` where the selected slot can stand
    /// (`Slot38::share_point`); 0 when none stands, for the selected slot
    /// itself and for a slot out of range.
    #[must_use]
    pub fn seq_share_point(&self, from: usize, n: u32) -> u32 {
        self.parked_ref(from).map_or(0, |s| s.share_point(n))
    }

    /// The selected slot's sequence replaced by the first positions of slot
    /// `from`'s, up to its share point for `n` ([`GpuModel::seq_share_point`]):
    /// a reset unless the model is empty already, then the stores rebuilt
    /// beside `from`'s (`Body38::share_state`): the selecting stores' rows,
    /// and the draft's store rows on a load with the draft, held on `from`'s
    /// pages wherever a page holds the prefix's whole rows alone, the
    /// recurrent stores put back from `from`'s point. The selected slot
    /// stands at that point, which it returns with the bytes its stores map
    /// onto `from`'s pages and copied, as a resume of a save `from` took
    /// there and a cut to it leave it; the calls after it are bit for bit
    /// those. A later write of either sequence below the point first gives
    /// the pages it reaches memory of their own ([`Store38::own_from`]), and
    /// a reset gives the shared pages back. Refused by name, nothing moved,
    /// where no point stands (a share point of 0), on a poisoned model, and
    /// while a pass of several slots waits for its commit.
    pub fn seq_share_prefix(
        &mut self,
        from: usize,
        n: u32,
    ) -> Result<(u32, SharedPages), GpuError> {
        self.refuse_if_poisoned(WHAT_SHARE)?;
        if self.seq_share_point(from, n) == 0 {
            return Err(GpuError::shape(
                WHAT_SHARE,
                format!(
                    "slot {from} holds no point at or below {n} that slot {} can take",
                    self.selected()
                ),
            ));
        }
        self.seq_reset_for(WHAT_SHARE)?;
        self.beside(from, WHAT_SHARE, |gpu, body, src| {
            body.share_state(gpu.stream(), src, n)
        })
    }
}

impl Body38 {
    /// Whether the load takes media ([`GpuModel::enable_media`]): every
    /// sequence holds a rotation table of its own.
    #[must_use]
    pub fn takes_media(&self) -> bool {
        self.seq_rope.is_some()
    }

    /// Device bytes of the live sequence's rotation table; 0 on a text load.
    #[must_use]
    pub fn rope_bytes(&self) -> usize {
        self.seq_rope.as_ref().map_or(0, |r| r.table().num_bytes())
    }
}

impl MediaCarrier for Body38 {
    const VIA: &'static str = "GpuModel::enable_media";

    fn width(&self) -> usize {
        self.embed_width()
    }

    fn seq_rope(&self) -> Option<&SeqRope> {
        self.seq_rope.as_ref()
    }

    fn media_parts(&mut self) -> Option<(&mut SeqRope, &mut MediaRows)> {
        self.seq_rope.as_mut().zip(self.media.as_mut())
    }
}

impl GpuModel<Body38> {
    /// Make the load a media load, once, after the open and before
    /// [`GpuModel::add_slots`] and any capture: the live sequence and every
    /// later slot hold a rotation table of their own, which an image's
    /// positions rewrite ([`SeqRope`]), and the ubatch walk takes the rows
    /// of a media call's spans in place of their positions' embeddings
    /// ([`MediaBody::prefill_media`]). The tables start as the load's plain
    /// one; a captured chain bakes the table's address, so the table is
    /// made before the first capture. The MTP draft's walks read the same
    /// table as the target's (`mtp_ctx`), each sequence's walks
    /// captured over its own. Refused by name: a second call, a draft that
    /// has captured a walk (before or after a reset), a file with no
    /// attention layer rotating by IMROPE sections, more than one slot, a
    /// captured chain, and a model that has run a call.
    pub fn enable_media(&mut self) -> Result<(), GpuError> {
        const WHAT_E: &str = "qwen4exp enable_media";
        let (slots, captured, pos) = (self.slots(), self.has_capture(), self.pos());
        let (gpu, _, body) = self.body_parts(WHAT_E)?;
        if body.seq_rope.is_some() {
            return Err(GpuError::state(WHAT_E, "a load that takes no media yet"));
        }
        if body
            .mtp
            .as_ref()
            .is_some_and(|d| !d.graph_nodes().is_empty())
        {
            return Err(GpuError::state(
                WHAT_E,
                "a load whose MTP draft has no captured walks: a captured walk bakes the \
                 rotation table's address, and the captures stay with the load through a reset",
            ));
        }
        let Some((sections, order)) = body.imrope else {
            return Err(GpuError::shape(
                WHAT_E,
                "a media load of a file with no IMROPE attention layer: an image's positions \
                 have no sections to follow",
            ));
        };
        if slots != 1 {
            return Err(GpuError::shape(
                WHAT_E,
                format!(
                    "{slots} slots: every sequence's rotation table is made with its slot, so \
                     media is enabled before GpuModel::add_slots"
                ),
            ));
        }
        if captured || pos != 0 || body.held != 0 {
            return Err(GpuError::state(
                WHAT_E,
                "a model that has run no call and captured no chain: a captured chain bakes the \
                 rotation table's address",
            ));
        }
        let stream = gpu.stream();
        let plain: Arc<[f32]> = body.rope.table.to_host_vec(stream)?.into();
        let base = RopeBase::new(plain, sections, order);
        body.seq_rope = Some(base.seq(stream, body.ctx)?);
        body.rope_base = Some(base);
        body.media = Some(MediaRows::new(stream, geo::HIDDEN)?);
        Ok(())
    }

    /// The rotation table the live sequence's rope launches read, `ctx` rows
    /// of 64 values read back: a media load's own, else the load's. An
    /// instrument; synchronizes.
    pub fn rope_rows(&self) -> Result<Vec<f32>, GpuError> {
        let body = self.body("qwen4exp rope_rows")?;
        Ok(rope_table(&body.rope, body.seq_rope.as_ref()).to_host_vec(self.gpu().stream())?)
    }
}

impl MediaBody for Body38 {
    type Span<'a> = ImageSpan<'a>;

    fn media_span<'s>(s: &'s ImageSpan<'_>) -> &'s MediaSpan<'s> {
        &s.span
    }

    /// Image spans on a text load.
    fn media_refusal(m: &GpuModel<Body38>, spans: usize) -> Option<String> {
        let takes = m.body(WHAT).is_ok_and(Body38::takes_media);
        (spans > 0 && !takes).then(|| {
            format!(
                "a prompt with {spans} image spans on a text load (GpuModel::enable_media makes \
                 a media load)"
            )
        })
    }

    /// The ubatch walk of [`GpuModel::prompt38_with`] with the spans' rows in
    /// place of their positions' embeddings and the sequence's rotation
    /// table shifted by them, inside the residency pass the session opens
    /// around a prompt call (`prompt_pass`). The argmax after the last id.
    /// The hidden-tail call has no Body38 form and is refused by name.
    fn prefill_media(
        m: &mut GpuModel<Body38>,
        ids: &[u32],
        spans: &[ImageSpan<'_>],
    ) -> Result<u32, GpuError> {
        media_call(
            m,
            "qwen4exp prefill_media",
            (ids, spans),
            |_| Ok(()),
            |m| m.prompt38_in_pass(ids),
        )
    }
}

impl ChainBody for Body38 {
    type Input = DecodeInput38;
    type Host = Body38;
    const UNHOSTED_POOL: bloomery_levers::UnhostedPool = bloomery_levers::UnhostedPool::Carried;

    fn arch() -> Arch {
        Arch::Qwen35moe
    }

    /// The step at `pos`, where the stores stand: the map check
    /// ([`MapCheck::refuse`], at every step: a planted map may change between
    /// two replays of one capture), its PLE rows read from the file, staged
    /// for the refresh's launch.
    fn decode_input(&mut self, token: u32, pos: u32) -> Result<DecodeInput38, GpuError> {
        self.refuse_inside_image("qwen4exp step", pos)?;
        self.card_leg.refuse(Walk38::Step)?;
        self.check_next(pos, 1)?;
        let hist = self.ple.fill(pos, &[token])?;
        self.stage(pos, 1, hist, None)?;
        Ok(DecodeInput38 { token, pos })
    }

    /// The step's input record (its position, its token, the lane word) and
    /// its PLE rows: two copies ahead of the step's launches. Once they are
    /// sent the step is launched ([`Body38::launch`]): what follows enqueues
    /// or replays its chain.
    fn refresh(&mut self, stream: &CudaStream, input: &DecodeInput38) -> Result<(), GpuError> {
        planted(&mut self.plant, Plant::BeforeLaunch)?;
        self.settle(stream)?;
        self.last_walk = TargetRows::Step;
        self.s
            .ple
            .e
            .copy_from_host(stream, &self.ple.e[..geo::HIDDEN])?;
        self.sp.write(stream, input.token, input.pos)?;
        self.launch(input.pos, 1, 1)
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        let io = Io {
            lane: Some(self.lane.word()),
            ..self.sp.io()
        };
        let Body38 {
            hybrid,
            plans,
            stores,
            ple_ring,
            rope,
            seq_rope,
            s,
            k,
            branch,
            slots,
            card,
            tier,
            taps,
            eps,
            ctx,
            plant,
            ..
        } = self;
        let prog = Step38 {
            branch,
            p: Parts38 {
                c: Ctx38 {
                    gpu,
                    w,
                    k,
                    eps: *eps,
                    table: rope_table(rope, seq_rope.as_ref()),
                    ctx: *ctx,
                },
                plans,
                seqs: Seqs38::One {
                    stores,
                    ple_ring,
                    io: &io,
                },
                s,
                m: 1,
                cur: 0,
                taps: taps.as_mut(),
                slots,
                card,
                tier: tier.as_mut(),
                each: false,
            },
            head,
        };
        let mut leg = StepLeg::new(gpu.stream(), hybrid);
        prog.walk(&mut leg)?;
        planted_eager(plant, gpu)
    }

    /// Every delta store's lanes and the PLE ring back to zero, every lane's
    /// stamp to a fresh store's, the lane word to 0, a new PLE history, no
    /// verify waiting or call staged, no gate plant (a failure, a slot map,
    /// ubatch routes), no ubatch walk's split or tapped positions, after the
    /// host tier's reset. The K/V planes, the
    /// raw key rings and the pooled keys need nothing: nothing reads a row at
    /// or past a live count, and every row below it is written by its own
    /// step first; the pages of the planes and the pooled keys past the rows
    /// the ended sequence reached give their memory back, as do the draft
    /// store's ([`Store38::settle`]).
    /// Synchronizes: the engine stream waited for within
    /// [`crate::host::ENGINE_BOUND`] before zeroing the stores and at
    /// the end ([`crate::host::await_done`]).
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        const WHAT: &str = "Body38::reset";
        let stream = gpu.stream();
        self.hybrid.settle(stream)?;
        crate::host::await_done(WHAT, || stream.query()).map_err(|e| self.hybrid.noted(e))?;
        for s in &mut self.stores {
            if let Store38::Rec { rec, .. } = s {
                rec.clear(stream)?;
            }
            s.restamp(stream)?;
            s.settle()?;
        }
        self.lane.set(stream, 0)?;
        self.pending = None;
        self.k
            .ple
            .reset_ring(stream, &mut self.ple_ring, geo::STREAMS)?;
        self.ple.restart()?;
        self.ckpt.clear();
        self.ple_points.clear();
        self.point_rows.clear();
        if let Some(d) = self.mtp.as_mut() {
            d.forget();
            d.settle()?;
        }
        if let Some(rope) = self.seq_rope.as_mut() {
            rope.restore(stream, MropeSeq::new())?;
        }
        self.wrote = Wrote38::default();
        self.held = 0;
        self.staged = None;
        self.plant = None;
        self.card_leg = MapCheck::of(self.hybrid.slots(), self.tier.is_some())?;
        self.wide.split = None;
        self.wide.force = None;
        if let Some(t) = self.wide.taps.as_mut() {
            t.walked = None;
        }
        self.sp.write(stream, 0, 0)?;
        debug_assert_eq!(LANE, 0);
        crate::host::await_done(WHAT, || stream.query()).map_err(|e| self.hybrid.noted(e))
    }

    fn head_eps(&self) -> f32 {
        self.eps
    }

    /// The head's mix is the final norm.
    fn head_norm(&self) -> HeadNorm {
        HeadNorm::Mixed
    }

    fn resident_bytes(&self) -> usize {
        self.store_bytes()
            + self.recovery.bytes()
            + self.rope.table.num_bytes()
            + self.rope_bytes()
            + self.s.bytes()
            + self.s.by_bytes()
            + self.sp.bytes()
            + self.a.bytes()
            + self.a.by_bytes()
            + self.rp.bytes()
            + self.pass_hsum.num_bytes()
            + self.wa.bytes()
            + self.wa.by_bytes()
            + self.wr.bytes()
            + self.wide_hsum.num_bytes()
            + self.wide.bytes()
            + self.wide.taps.as_ref().map_or(0, WideTaps::bytes)
            + self.slots.buf().num_bytes()
            + self.card.bytes()
            + self.tier.as_ref().map_or(0, TierSide38::bytes)
            + self.lane.bytes()
            + self.slot_in.as_ref().map_or(0, SlotIn::bytes)
            + self.hybrid.boundary().device_bytes()
            + self.taps.as_ref().map_or(0, Taps38::bytes)
            + self
                .mtp
                .as_ref()
                .map_or(0, |d| d.resident_bytes() + d.arena_bytes())
    }

    fn layers(&self) -> Range<usize> {
        0..self.plans.len()
    }

    fn host(&mut self) -> Option<&mut Body38> {
        Some(self)
    }
}

impl Slots for Body38 {
    type Seq = Slot38;

    /// A sequence of the load's shape in the state the load leaves: zeroed
    /// stores with a fresh delta store's stamps, the PLE ring zeroed and a
    /// new hash history, lane 0, no checkpoint and no position held, its
    /// arena rows zeroed, and — on a load with the draft — a zeroed draft
    /// store no walk has captured over. Refused by name past the count the
    /// load's plan was made for, naming the plan's count and the sequence
    /// asked for. Load-time allocation.
    fn new_seq(&mut self, gpu: &Gpu) -> Result<Slot38, GpuError> {
        const WHAT_N: &str = "qwen4exp Body38::new_seq";
        if self.slots_made >= self.slots_planned {
            return Err(GpuError::shape(
                WHAT_N,
                format!(
                    "sequence {} of a plan that counts {} slots; plan the load for as many \
                     resident sequences as it must serve",
                    self.slots_made + 1,
                    self.slots_planned
                ),
            ));
        }
        let stream = gpu.stream();
        let handle = gpu.stream_handle();
        let d = dims(self.s.route.dims(), self.ctx);
        // The load's one cache format, read back from the live planes.
        let kv = self.kv_format();
        let stores = self
            .kinds()
            .into_iter()
            .map(|k| match k {
                LayerKind38::Gdn => Store38::rec(stream),
                LayerKind38::Qsa => Store38::qsa(&handle, &d, kv),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ckpt = Checkpoints::sequence(
            gpu.context(),
            checkpoint_lens_of(&stores),
            self.ctx,
            CHECKPOINT_EVERY,
        )?;
        let ple_hist = self
            .ple
            .hash
            .new_history()
            .map_err(|e| GpuError::shape(WHAT_N, e.to_string()))?;
        let arenas = [&self.s, &self.a, &self.wa];
        let rows_len = |walk: TargetRows| final_streams(&self.plans, arenas, walk).len();
        let (rows_step, rows_pass) = (
            DeviceBuffer::zeroed(stream, rows_len(TargetRows::Step))?,
            DeviceBuffer::zeroed(stream, rows_len(TargetRows::Pass))?,
        );
        let draft = match self.mtp.as_ref() {
            None => None,
            Some(d) => Some(d.new_seq(&handle)?),
        };
        let rope = self
            .rope_base
            .as_ref()
            .map(|b| b.seq(stream, self.ctx))
            .transpose()?;
        let recovery = recovery_table(stream, &stores)?;
        let seq = Slot38 {
            stores,
            recovery,
            ple_ring: DeviceBuffer::zeroed(stream, ple::ring_len(geo::STREAMS))?,
            lane: LaneWord::new(stream, LANES)?,
            pending: None,
            ckpt,
            ple_points: Vec::new(),
            point_rows: Vec::new(),
            ple_hist,
            held: 0,
            wrote: Wrote38::default(),
            last_walk: TargetRows::Step,
            rows_step,
            rows_pass,
            draft,
            rope,
        };
        // Last, so a sequence whose allocation failed is not counted.
        self.slots_made += 1;
        Ok(seq)
    }

    /// Exchange the live sequence with `seq` ([`Body38::exchange`]):
    /// pointer moves only, so nothing is copied, captured or synchronized,
    /// and each slot's captures keep addressing its own buffers. Refused by
    /// name, moving nothing, while a call is staged or the live sequence's
    /// verify waits for its commit: the staged call's launches are the live
    /// sequence's alone, and the commit is the live sequence's call.
    fn swap_seq(&mut self, _gpu: &Gpu, seq: &mut Slot38) -> Result<(), GpuError> {
        const WHAT_W: &str = "qwen4exp Body38::swap_seq";
        if let Some(s) = &self.staged {
            return Err(GpuError::shape(
                WHAT_W,
                format!(
                    "a select while a call of {} rows from {} is staged and not launched",
                    s.rows, s.pos
                ),
            ));
        }
        if let Some(p) = &self.pending {
            return Err(GpuError::shape(
                WHAT_W,
                format!(
                    "a select while the verify of {} rows at {} waits for its commit \
                     (GpuModel::rollback to the first position not kept)",
                    p.tokens.len(),
                    p.pos0
                ),
            ));
        }
        if self.mtp.is_some() != seq.draft.is_some() {
            return Err(GpuError::state(
                WHAT_W,
                "the draft's side of every sequence (a load that drafts serves sequences that \
                 each carry one)",
            ));
        }
        self.exchange(seq);
        Ok(())
    }

    /// Device bytes one sequence holds: its stores and PLE ring, its replay
    /// table, its lane word, its two arena-row buffers, the draft's store on
    /// a load with one and its rotation table on a media load — the same
    /// terms [`seq38_bytes`] counts a host park's read of them by, the two
    /// tables aside (they are no part of a saved state).
    fn seq_bytes(&self) -> usize {
        let arenas = [&self.s, &self.a, &self.wa];
        self.stores.iter().map(Store38::bytes).sum::<usize>()
            + self.recovery.bytes()
            + self.rope_bytes()
            + self.ple_ring.num_bytes()
            + self.lane.bytes()
            + final_streams(&self.plans, arenas, TargetRows::Step).num_bytes()
            + final_streams(&self.plans, arenas, TargetRows::Pass).num_bytes()
            + self.mtp.as_ref().map_or(0, Mtp38::seq_bytes)
    }
}

impl Body38 {
    /// Exchange every per-sequence piece of the live sequence with `seq`'s:
    /// pointer moves only — each is a buffer handle or a host value the body
    /// holds by value. [`Slots::swap_seq`]'s exchange after its refusals,
    /// and the one a pass of several slots plans and commits a parked slot
    /// through (in and back out, the call on it between).
    fn exchange(&mut self, seq: &mut Slot38) {
        std::mem::swap(&mut self.stores, &mut seq.stores);
        std::mem::swap(&mut self.recovery, &mut seq.recovery);
        std::mem::swap(&mut self.ple_ring, &mut seq.ple_ring);
        std::mem::swap(&mut self.lane, &mut seq.lane);
        std::mem::swap(&mut self.pending, &mut seq.pending);
        std::mem::swap(&mut self.ckpt, &mut seq.ckpt);
        std::mem::swap(&mut self.ple_points, &mut seq.ple_points);
        std::mem::swap(&mut self.point_rows, &mut seq.point_rows);
        std::mem::swap(&mut self.ple.hist, &mut seq.ple_hist);
        std::mem::swap(&mut self.held, &mut seq.held);
        std::mem::swap(&mut self.wrote, &mut seq.wrote);
        std::mem::swap(&mut self.seq_rope, &mut seq.rope);
        // The ubatch arena is the load's, not the sequence's: another
        // sequence's prompt call may have written it since this one's, so
        // the sequence brought live holds no position there. A draft parked
        // with rows waiting in that arena skips them (`Policy::park`).
        self.wrote.set(TargetRows::Ubatch, 0, 0, 0);
        std::mem::swap(&mut self.last_walk, &mut seq.last_walk);
        // The arena rows a draft walk reads by position (the walk's graph
        // replays the shared arena's address, so the sequence's rows move as
        // the buffer the arena addresses, not as a copy).
        let plans = &self.plans;
        let step = [&mut self.s, &mut self.a, &mut self.wa];
        std::mem::swap(
            final_streams_mut(plans, step, TargetRows::Step),
            &mut seq.rows_step,
        );
        let pass = [&mut self.s, &mut self.a, &mut self.wa];
        std::mem::swap(
            final_streams_mut(plans, pass, TargetRows::Pass),
            &mut seq.rows_pass,
        );
        // Both sides carry a draft or neither does: `new_seq` makes a
        // sequence's draft side by the load's, and `swap_seq` refuses a
        // mix.
        if let (Some(d), Some(s)) = (&mut self.mtp, &mut seq.draft) {
            d.swap_slot(s);
        }
    }
}

impl TierBody for Body38 {
    type Experts = HostRun;

    fn hybrid(&self) -> &Hybrid<HostRun> {
        &self.hybrid
    }

    fn hybrid_mut(&mut self) -> &mut Hybrid<HostRun> {
        &mut self.hybrid
    }

    fn residency_parts(&mut self) -> Option<ResidencyParts<'_, HostRun>> {
        Some(ResidencyParts {
            glue: &mut self.residency_glue,
            hybrid: &mut self.hybrid,
            slots: &self.slots,
        })
    }

    /// A launched verify's PLE rows delivered first ([`Body38::deliver_ple`]):
    /// the host legs its walk serves come after its PLE site.
    fn serve_chain(&mut self, chain: Chain) -> Result<(), GpuError> {
        let delivered = self.deliver_ple();
        let served = self.hybrid.serve_captured_of(chain);
        delivered.and(served)?;
        planted(&mut self.plant, Plant::AfterLaunch)
    }
}

impl Body38 {
    /// Plan a verify of `tokens` (2..=[`VERIFY_ROWS`]) at `pos`, where the
    /// stores stand: the ids checked as a prompt's are, the PLE rows named,
    /// the arrival's flag lowered, the record copied to the pass arena, then
    /// the verify launched ([`Body38::launch`]) and left waiting for its
    /// commit and for its PLE rows, which arrive at its PLE site
    /// ([`StagedRows`]): the host reads and delivers them once the walk is
    /// enqueued ([`Body38::deliver_ple`]), so their read runs under the
    /// layers before the site. Its graph's launch follows this plan with no
    /// body call between. Refused by name
    /// before anything moves: a slot map with a routed expert on a tier card
    /// ([`MapCheck::refuse`]), another row count, the taps armed, a verify
    /// already waiting, a position other than the stores' or past them, an
    /// id the embedding or the PLE hash does not take; a failure before the
    /// launch leaves no verify waiting and the lane word where it stood.
    fn plan_verify(
        &mut self,
        stream: &CudaStream,
        tokens: &[u32],
        pos: u32,
    ) -> Result<(), GpuError> {
        const WHAT_V: &str = "qwen4exp verify";
        self.refuse_inside_image(WHAT_V, pos)?;
        self.card_leg.refuse(Walk38::Verify)?;
        let n = tokens.len();
        if !(2..=VERIFY_ROWS).contains(&n) {
            return Err(GpuError::shape(
                WHAT_V,
                format!("a verify of {n} rows; the records hold 2..={VERIFY_ROWS}"),
            ));
        }
        if self.taps.is_some() {
            return Err(GpuError::state(
                WHAT_V,
                "layer taps off (a verify writes none)",
            ));
        }
        self.check_next(pos, n)?;
        super::refuse_past_vocab(WHAT_V, tokens, self.vocab)?;
        ple_takes(WHAT_V, self.ple.hash.window(), tokens, &[])?;
        self.settle(stream)?;
        let hist = self.ple.name_rows(pos, tokens, &[])?;
        self.last_walk = TargetRows::Pass;
        self.rp.write(stream, tokens, pos)?;
        self.ple_arrival.clear(0)?;
        self.stage(pos, n, hist, Some(tokens.to_vec()))?;
        planted(&mut self.plant, Plant::BeforeLaunch)?;
        self.launch(pos, n, n)?;
        if let Some(p) = self.pending.as_mut() {
            p.ple_due = true;
        }
        Ok(())
    }

    /// Enqueue the planned verify of `head.m()` rows through the step port
    /// into `head`, one head of the verify's rows. An eager verify needs its
    /// plan; a capture records the launches only.
    fn walk_verify(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        const WHAT_V: &str = "qwen4exp verify";
        let m = head.m();
        let planned = self.pending.as_ref().map(|p| p.tokens.len());
        if !crate::capturing(gpu.stream())? {
            if planned != Some(m) {
                return Err(GpuError::state(
                    WHAT_V,
                    "a verify planned for this head's rows (plan_rows before the pass)",
                ));
            }
            // An eager walk serves its host legs as it enqueues them: its
            // rows arrive before it.
            self.deliver_ple()?;
        }

        let io = Io {
            lane: Some(self.lane.word()),
            ..self.rp.io(m)?
        };
        let Body38 {
            hybrid,
            plans,
            stores,
            ple_ring,
            ple_arrival,
            rope,
            seq_rope,
            a,
            k,
            branch,
            slots,
            card,
            tier,
            eps,
            ctx,
            plant,
            ..
        } = self;
        let prog = Verify38 {
            branch,
            p: Parts38 {
                c: Ctx38 {
                    gpu,
                    w,
                    k,
                    eps: *eps,
                    table: rope_table(rope, seq_rope.as_ref()),
                    ctx: *ctx,
                },
                plans,
                seqs: Seqs38::One {
                    stores,
                    ple_ring,
                    io: &io,
                },
                s: a,
                m,
                cur: 0,
                taps: None,
                slots,
                card,
                tier: tier.as_mut(),
                each: true,
            },
            head,
            copies: Vec::new(),
            arrival: Some(&*ple_arrival),
        };
        let mut leg = StepLeg::new(gpu.stream(), hybrid);
        prog.walk(&mut leg)?;
        planted_eager(plant, gpu)
    }

    /// Keep the first `pos − pos0` rows of the verify waiting for its commit
    /// and take the rest back: replay the accepted recovery records into
    /// the one state lane, where the lane word stays, the PLE history over
    /// the kept rows, and the MTP
    /// draft's records — the arenas' held rows and the store's held
    /// positions — cut to `pos`, the rejected rows no longer any walk's to
    /// read. With no verify
    /// waiting nothing is taken at the position the stores stand at, and a
    /// position behind it is a cut to the checkpoint there ([`Body38::cut`]),
    /// as it is at or below a waiting verify's first position. What is taken
    /// is [`Body38::kept`]'s rule; anything else is refused by name, the
    /// verify left waiting.
    fn commit(&mut self, gpu: &Gpu, pos: u32) -> Result<(), GpuError> {
        let Some(p) = self.pending.as_ref().filter(|p| pos > p.pos0) else {
            if pos == self.held && self.pending.is_none() {
                return Ok(());
            }
            return self.cut(gpu.stream(), pos);
        };
        let rows = p.tokens.len();
        if pos as usize > p.pos0 as usize + rows {
            return Err(GpuError::shape(
                "qwen4exp commit",
                format!(
                    "back to position {pos}: the verify of {rows} rows at {} keeps 1..={rows} \
                     (row 0 always)",
                    p.pos0
                ),
            ));
        }
        let Some(p) = self.pending.take() else {
            return Ok(());
        };
        let kept = (pos - p.pos0) as usize;
        let stream = gpu.stream();
        self.k.linear.delta.enqueue_delta_recovery_layers(
            stream,
            (self.lane.word(), 0),
            kept,
            &self.recovery,
            gpu.unlabelled_sink(),
        )?;
        // One lane: the replay's lane, the last kept row's
        // (`linear::delta::kept_lane`), is the one it read, so the word stays.
        const _: () = assert!(LANES == 1);
        if kept < rows {
            self.ple.keep(p.hist, p.pos0, &p.tokens[..kept])?;
        }
        self.held = pos;
        self.wrote.cut(pos);
        if let Some(d) = self.mtp.as_mut() {
            d.cut(pos);
        }
        Ok(())
    }
}

/// Where an end mark's copies run ([`Body38::copy_end_mark`]): a stream of
/// their own and an event a layer the walk records ([`WideParts::done`]).
struct EndMarker {
    copy: Arc<CudaStream>,
    done: Vec<CudaEvent>,
}

impl EndMarker {
    fn new(stream: &CudaStream, layers: usize) -> Result<EndMarker, GpuError> {
        let ctx = stream.context();
        let flags = Some(sys::CUevent_flags_enum_CU_EVENT_DISABLE_TIMING);
        Ok(EndMarker {
            copy: ctx.new_stream()?,
            done: (0..layers)
                .map(|_| ctx.new_event(flags))
                .collect::<Result<_, _>>()?,
        })
    }
}

/// The stores a checkpoint copies, lent for one call ([`Copied::of`]), and
/// a sequence state's positional rows beside them.
struct Copied<'a> {
    /// Each selecting layer's rows of the positions the call carries, then
    /// the draft's store rows of the same positions.
    spans: Vec<Spans<'a>>,
    /// Each delta layer's committed state lane, conv ring and carry, in
    /// layer order.
    lanes: Vec<WindowMut<'a, f32>>,
    rings: Vec<&'a mut DeviceBuffer<f32>>,
    carries: Vec<&'a mut DeviceBuffer<f32>>,
    ple: &'a mut DeviceBuffer<f32>,
    /// Each selecting layer's raw key ring as f32 words, in layer order.
    keys: Vec<WindowMut<'a, f32>>,
}

impl<'a> Copied<'a> {
    /// The stores of `stores` and `ple_ring` a checkpoint copies — each
    /// delta layer's state lane `lane`, its conv ring and carry, then the PLE
    /// ring, then each selecting layer's raw key ring ([`checkpoint_lens`])
    /// — and, with `rows` (`(ctx, n)`), each selecting layer's rows of
    /// positions below `n`: its K/V planes' ([`KvPlanes::spans`](super::scratch::KvPlanes::spans), in
    /// the cache's format)
    /// and its pooled keys' of the pools those positions reach; then
    /// `draft`'s store rows of the same positions ([`Mtp38::store_spans`]).
    fn of(
        stores: &'a mut [Store38],
        ple: &'a mut DeviceBuffer<f32>,
        lane: u32,
        rows: Option<(usize, usize)>,
        draft: Option<&'a mut Mtp38>,
    ) -> Result<Copied<'a>, GpuError> {
        // A draft's rows are the positions' rows: without the positions
        // there is nothing of the draft to copy, and taking it would drop it.
        if rows.is_none() && draft.is_some() {
            return Err(GpuError::state(
                "qwen38::copied",
                "the positions a draft store's rows are copied for",
            ));
        }
        let len = GDN.state_len();
        let mut c = Copied {
            spans: Vec::new(),
            lanes: Vec::new(),
            rings: Vec::new(),
            carries: Vec::new(),
            ple,
            keys: Vec::new(),
        };
        for s in stores {
            match s {
                Store38::Rec { rec, .. } => {
                    let RecStore {
                        state, ring, carry, ..
                    } = rec;
                    let byte = lane as usize * len * size_of::<f32>();
                    c.lanes.push(WindowMut::of_mut(state, byte, len)?);
                    c.rings.push(ring);
                    c.carries.push(carry.as_mut().ok_or_else(no_carry)?);
                }
                Store38::Qsa { kv, ring, pooled } => {
                    c.keys.push(WindowMut::of_mut(ring, 0, KEY_RING_WORDS)?);
                    if let Some((ctx, n)) = rows {
                        let d = geo::IDX_DIM;
                        c.spans.extend(kv.spans((geo::N_KV, ctx, geo::HEAD), n)?);
                        c.spans
                            .push(Spans::by_pool(&mut **pooled, 1, 0, n, (geo::POOL, d))?);
                    }
                }
            }
        }
        if let (Some((_, n)), Some(d)) = (rows, draft) {
            c.spans.extend(d.store_spans(n)?);
        }
        Ok(c)
    }

    /// The stores in the checkpoints' order: each delta layer's lane, ring
    /// and carry, then the PLE ring, then each selecting layer's key ring.
    fn list(&mut self) -> Vec<&mut DeviceBuffer<f32>> {
        self.list_split().1
    }

    /// The positional rows, and the stores in the checkpoints' order
    /// ([`Copied::list`]).
    fn list_split(&mut self) -> (&mut [Spans<'a>], Vec<&mut DeviceBuffer<f32>>) {
        let Copied {
            spans,
            lanes,
            rings,
            carries,
            ple,
            keys,
        } = self;
        let mut out: Vec<&mut DeviceBuffer<f32>> = Vec::new();
        for ((l, r), c) in lanes
            .iter_mut()
            .zip(rings.iter_mut())
            .zip(carries.iter_mut())
        {
            out.push(&mut **l);
            out.push(&mut **r);
            out.push(&mut **c);
        }
        out.push(&mut **ple);
        out.extend(keys.iter_mut().map(|k| &mut **k));
        (spans, out)
    }
}

/// Lane `lane` of a delta state read back whole (`LANES` lanes of
/// `GDN.state_len()`).
fn lane_of(mut state: Vec<f32>, lane: u32) -> Result<Vec<f32>, GpuError> {
    let len = GDN.state_len();
    let at = lane as usize * len;
    if state.len() != LANES * len || at + len > state.len() {
        return Err(GpuError::shape(
            WHAT,
            format!("lane {lane} of a state of {} values", state.len()),
        ));
    }
    state.truncate(at + len);
    Ok(state.split_off(at))
}

impl Rows for Body38 {
    const MAX_ROWS: usize = VERIFY_ROWS;
    /// Unread: the verify's chain is its rows' ([`Rows::chain_of`]).
    const CHAIN: Chain = Chain::Step;
    const HEADS: RowHeads = RowHeads::One;

    /// One row of `m` columns through the step port.
    fn chain_of(m: usize) -> Chain {
        Chain::Cols(m)
    }

    fn plan_rows(&mut self, stream: &CudaStream, tokens: &[u32], pos: u32) -> Result<(), GpuError> {
        self.plan_verify(stream, tokens, pos)
    }

    /// The verify into `heads[0]`, a head of its rows; any other heads are
    /// refused by name.
    fn enqueue_rows(&mut self, gpu: &Gpu, w: &Weights, heads: &mut [Head]) -> Result<(), GpuError> {
        match heads {
            [head] => self.walk_verify(gpu, w, head),
            _ => Err(GpuError::shape(
                "qwen4exp verify",
                format!(
                    "{} heads; a verify runs into one head of its rows",
                    heads.len()
                ),
            )),
        }
    }
}

impl Rollback for Body38 {
    /// Refused by name: the commit enqueues copies, so it runs with the
    /// card at hand ([`Rollback::rollback_on`], what `GpuModel::rollback`
    /// calls).
    fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        Err(GpuError::shape(
            "qwen4exp commit",
            format!("back to position {pos} without the card: GpuModel::rollback commits"),
        ))
    }

    /// The commit or the cut ([`Body38::commit`]) and, on a media load, the
    /// rotation table's: the images from `pos` on go and the rows after
    /// `pos` take the positions the kept images leave ([`SeqRope::cut`]). A
    /// cut inside an image is refused by name before anything moves.
    fn rollback_on(&mut self, gpu: &Gpu, pos: u32) -> Result<(), GpuError> {
        let row = pos as usize;
        if let Some(rope) = &self.seq_rope {
            rope.rows()
                .seq()
                .clone()
                .cut(row)
                .map_err(|e| GpuError::shape("qwen4exp commit", e.to_string()))?;
        }
        self.commit(gpu, pos)?;
        if let Some(rope) = self.seq_rope.as_mut() {
            rope.cut(gpu.stream(), row)?;
        }
        Ok(())
    }
}

/// A pass of several resident slots' rows (`GpuModel::step_slots`): every
/// busy slot's rows walked as one verify of all of them (`Verify38` over
/// [`Seqs38::Slots`]) — the row-wise launches once over every row, the
/// sequence-bound launches over each slot's own rows and stores, most once
/// over every slot (`crate::seq_rows`), each row's bits the launch that
/// slot's rows would run alone — into one head of every row.
/// Each slot is left as a verify of its rows leaves the live sequence:
/// its stores past its rows and its verify waiting for its commit
/// ([`Body38::keep_slot`]).
impl Body38 {
    /// Plan the pass of `rows` (the busy slots' ranges in pass order, slot
    /// 0 first when it is busy) over `ids` (every row's token), `parked`
    /// every busy slot's sequence but slot 0's — the live one — in `rows`
    /// order. Refused by name before anything moves: ranges that do not
    /// tile the pass from row 0, slot 0 busy and not first, a parked count
    /// other than the ranges past slot 0, a load planned for one sequence,
    /// the layer taps armed, a route trace attached, more rows than the step port's columns, a slot map with a
    /// routed expert on a tier card ([`MapCheck::refuse`]), and for each
    /// slot on its own sequence a row count outside `1..=VERIFY_ROWS`, a position
    /// other than its stores' or past them, a verify waiting, an id the
    /// embedding or the PLE hash does not take. Then each slot's call, on
    /// its sequence: its waiting cut settled, its PLE rows read into its
    /// rows of the pass arena, the call launched as a verify of its rows
    /// ([`Body38::launch`]) — and the slots' input records written.
    pub(super) fn plan_slots(
        &mut self,
        stream: &CudaStream,
        parked: &mut [&mut Slot38],
        rows: &[SlotRange],
        ids: &[u32],
    ) -> Result<(), GpuError> {
        let total = SlotRowOrder::RetainedPrefix.admit(WHAT_SLOTS, rows, parked.len())?;
        if self.slot_in.is_none() {
            return Err(GpuError::shape(
                WHAT_SLOTS,
                "a pass of several slots on a load planned for one sequence",
            ));
        }
        if self.taps.is_some() {
            return Err(GpuError::state(
                WHAT_SLOTS,
                "layer taps off (a pass of several slots writes none)",
            ));
        }
        if self.hybrid.route_traced() {
            return Err(GpuError::shape(
                WHAT_SLOTS,
                "a pass of several slots with a route trace attached: the trace records one \
                 sequence's positions",
            ));
        }
        let cols = self.hybrid.boundary().layout().cols();
        if total > cols {
            return Err(GpuError::shape(
                WHAT_SLOTS,
                format!("{total} rows in one pass; the load's step port takes {cols} columns"),
            ));
        }
        self.card_leg.refuse(Walk38::Verify)?;
        for_slots(self, parked, rows, |b, r| {
            b.check_slot(r, r.ids(WHAT_SLOTS, ids)?)
        })?;
        planted(&mut self.plant, Plant::BeforeLaunch)?;
        for_slots(self, parked, rows, |b, r| {
            b.launch_slot(stream, r, r.ids(WHAT_SLOTS, ids)?)
        })?;
        self.slot_in
            .as_mut()
            .ok_or(GpuError::state(WHAT_SLOTS, "the slots' input records"))?
            .write(stream, rows, ids)
    }

    /// Refused by name unless the live sequence takes a verify of slot
    /// `r.slot`'s `tokens` from `r.pos0` as one slot of a pass:
    /// `1..=VERIFY_ROWS` rows (the rows a delta store records), from where its
    /// stores stand, no verify waiting, ids the embedding and the PLE hash
    /// take.
    fn check_slot(&self, r: &SlotRange, tokens: &[u32]) -> Result<(), GpuError> {
        let named = |e: GpuError| GpuError::shape(WHAT_SLOTS, format!("slot {}: {e}", r.slot));
        let m = tokens.len();
        if !(1..=VERIFY_ROWS).contains(&m) {
            return Err(GpuError::shape(
                WHAT_SLOTS,
                format!(
                    "slot {}'s {m} rows; a delta store records 1..={VERIFY_ROWS} rows of a verify",
                    r.slot
                ),
            ));
        }
        self.refuse_inside_image(WHAT_SLOTS, r.pos0)
            .map_err(named)?;
        self.check_next(r.pos0, m).map_err(named)?;
        super::refuse_past_vocab(WHAT_SLOTS, tokens, self.vocab).map_err(named)?;
        ple_takes(WHAT_SLOTS, self.ple.hash.window(), tokens, &[]).map_err(named)
    }

    /// The live sequence's call as slot `r.slot` of a pass: its waiting cut
    /// settled, the PLE rows of `tokens` read and copied into rows `r.rows`
    /// of the pass arena, and the call launched as a verify of its rows,
    /// waiting for its commit.
    fn launch_slot(
        &mut self,
        stream: &CudaStream,
        r: &SlotRange,
        tokens: &[u32],
    ) -> Result<(), GpuError> {
        let (m, h) = (tokens.len(), geo::HIDDEN);
        self.settle(stream)?;
        let hist = self.ple.fill(r.pos0, tokens)?;
        self.last_walk = TargetRows::Pass;
        let byte = r.rows.start * h * size_of::<f32>();
        WindowMut::<f32>::of_mut(&mut self.a.ple.e, byte, m * h)?
            .copy_from_host(stream, &self.ple.e[..m * h])?;
        self.stage(r.pos0, m, hist, Some(tokens.to_vec()))?;
        self.launch(r.pos0, m, m)
    }

    /// Enqueue the pass of `rows` [`Body38::plan_slots`] planned into
    /// `head`, one head of every row, `parked` as there. An eager pass
    /// needs each slot's plan; a capture records the launches only. A pass
    /// that runs no row of slot 0 leaves slot 0's final-stream rows as they
    /// stood ([`stands_in`]).
    pub(super) fn enqueue_slots(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        head: &mut Head,
        parked: &mut [&mut Slot38],
        rows: &[SlotRange],
    ) -> Result<(), GpuError> {
        let total = SlotRowOrder::RetainedPrefix.admit(WHAT_SLOTS, rows, parked.len())?;
        if !crate::capturing(gpu.stream())? {
            for_slots(self, parked, rows, |b, r| {
                let planned = b
                    .pending
                    .as_ref()
                    .is_some_and(|p| p.pos0 == r.pos0 && p.tokens.len() == r.rows.len());
                if planned {
                    Ok(())
                } else {
                    Err(GpuError::state(
                        WHAT_SLOTS,
                        "each slot's rows planned (plan_slots before the pass)",
                    ))
                }
            })?;
        }
        // The homes are canonical, so the pass arena's final-stream rows are
        // slot 0's own, where its draft's waiting rows sit. A pass that runs
        // no row of slot 0 walks into its first slot's own rows instead: slot
        // 0's stay as its last call left them, and the first slot's land from
        // row 0, where its copy would put them.
        let stand_in = rows.first().is_some_and(|r| stands_in(r.slot));
        let swap = |b: &mut Body38, parked: &mut [&mut Slot38]| -> Result<(), GpuError> {
            if !stand_in {
                return Ok(());
            }
            let seq = parked
                .first_mut()
                .ok_or_else(|| rows[0].no_parked(WHAT_SLOTS))?;
            let arena = [&mut b.s, &mut b.a, &mut b.wa];
            std::mem::swap(
                final_streams_mut(&b.plans, arena, TargetRows::Pass),
                &mut seq.rows_pass,
            );
            Ok(())
        };
        swap(self, parked)?;
        let walked = self.walk_slots(gpu, w, head, parked, rows, total);
        let back = swap(self, parked);
        walked.and(back)
    }

    /// [`Body38::enqueue_slots`]'s walk of the pass's `total` rows, each
    /// parked slot past the first copied to its own final-stream rows.
    /// The first slot's rows are written in place.
    fn walk_slots(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        head: &mut Head,
        parked: &mut [&mut Slot38],
        rows: &[SlotRange],
        total: usize,
    ) -> Result<(), GpuError> {
        let Body38 {
            hybrid,
            plans,
            stores,
            ple_ring,
            lane,
            rope,
            seq_rope,
            a,
            k,
            branch,
            slots,
            card,
            tier,
            eps,
            ctx,
            plant,
            slot_in,
            ..
        } = self;
        let ins = slot_in
            .as_ref()
            .ok_or(GpuError::state(WHAT_SLOTS, "the slots' input records"))?;
        let recs = rows
            .iter()
            .enumerate()
            .map(|(j, r)| ins.io(j, r))
            .collect::<Result<Vec<_>, GpuError>>()?;
        let mut live = Some((
            stores.as_mut_slice(),
            ple_ring,
            &*lane,
            rope_table(rope, seq_rope.as_ref()),
        ));
        let mut others = parked.iter_mut();
        let mut seqs = Vec::with_capacity(rows.len());
        let mut copies = Vec::new();
        for (i, (r, rec)) in rows.iter().zip(&recs).enumerate() {
            let (stores, ple_ring, lane, table) = if r.slot == 0 {
                live.take().ok_or_else(|| slot_twice(WHAT_SLOTS, r.slot))?
            } else {
                let Slot38 {
                    stores,
                    ple_ring,
                    lane,
                    rows_pass,
                    rope: own,
                    ..
                } = &mut **others.next().ok_or_else(|| r.no_parked(WHAT_SLOTS))?;
                if i != 0 {
                    copies.push(RowsCopy {
                        rows: r.rows.clone(),
                        to: rows_pass,
                    });
                }
                (
                    stores.as_mut_slice(),
                    ple_ring,
                    &*lane,
                    rope_table(rope, own.as_ref()),
                )
            };
            seqs.push(SlotSeq38 {
                stores,
                ple_ring,
                io: Io {
                    lane: Some(lane.word()),
                    ..rec.io()
                },
                rows: r.rows.clone(),
                table,
            });
        }
        let prog = Verify38 {
            branch,
            p: Parts38 {
                c: Ctx38 {
                    gpu,
                    w,
                    k,
                    eps: *eps,
                    table: rope_table(rope, seq_rope.as_ref()),
                    ctx: *ctx,
                },
                plans,
                seqs: Seqs38::Slots(seqs),
                s: a,
                m: total,
                cur: 0,
                taps: None,
                slots,
                card,
                tier: tier.as_mut(),
                each: true,
            },
            head,
            copies,
            arrival: None,
        };
        let mut leg = StepLeg::new(gpu.stream(), hybrid);
        prog.walk(&mut leg)?;
        planted_eager(plant, gpu)
    }

    /// The pass's slot keeps its rows below position `at`: the commit of
    /// its waiting verify ([`Body38::commit`]: the kept rows replayed into
    /// the state lane, the PLE history replayed over the kept rows, the
    /// held count and the draft's records cut), on `seq` — a parked slot's
    /// sequence, exchanged in and back out — or on the live sequence.
    pub(super) fn keep_slot(
        &mut self,
        gpu: &Gpu,
        seq: Option<&mut Slot38>,
        at: u32,
    ) -> Result<(), GpuError> {
        on_seq(self, seq, |b| b.commit(gpu, at))
    }

    /// The captured pass of several slots' launches for the slots and row
    /// counts of `key` (each `(slot, rows)`, in pass order): the verify
    /// walk's at every row of the pass (no PLE arrival: the pass's rows are
    /// in its arena before its launch), each slot past the first's
    /// sequence-bound launches ([`super::program38`]'s `seq_launches`), and
    /// the copy of each parked slot's rows.
    #[must_use]
    pub fn slots_launches(&self, key: &[(usize, usize)]) -> usize {
        let rows = key.iter().map(|&(_, m)| m).sum();
        let stand_in = key.first().is_some_and(|&(slot, _)| stands_in(slot));
        verify_launches(&self.plans, &self.card, self.tier.as_ref(), rows, self.head)
            + key.len().saturating_sub(1) * seq_launches(&self.plans, self.kv_format())
            + key.iter().filter(|&&(slot, _)| slot != 0).count()
            - usize::from(stand_in)
    }
}

impl SlotRows for Body38 {
    /// The pass arena's rows, which a load planned for several sequences
    /// widens the step port to ([`Body38::plan_slots`] refuses past the
    /// port's columns).
    const MAX_ROWS: usize = PASS_ROWS;
    const HEADS: RowHeads = RowHeads::One;
    /// The commit settles a partial keep ([`Body38::keep_slot`]: the state
    /// lane, the PLE history and the held count cut to the kept rows).
    const SETTLES_PARTIAL_KEEP: bool = true;

    /// One row of the pass's columns through the step port: the step's
    /// chain at one row.
    fn chain_of(rows: &[SlotRange]) -> Chain {
        match rows.last().map_or(0, |r| r.rows.end) {
            1 => Chain::Step,
            m => Chain::Cols(m),
        }
    }

    fn plan_slots(
        &mut self,
        stream: &CudaStream,
        parked: &mut [&mut Slot38],
        rows: &[SlotRange],
        ids: &[u32],
    ) -> Result<(), GpuError> {
        Body38::plan_slots(self, stream, parked, rows, ids)
    }

    /// The pass into `heads[0]`, a head of its rows; any other heads are
    /// refused by name.
    fn enqueue_slots(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        heads: &mut [Head],
        parked: &mut [&mut Slot38],
        rows: &[SlotRange],
    ) -> Result<(), GpuError> {
        match heads {
            [head] => Body38::enqueue_slots(self, gpu, w, head, parked, rows),
            _ => Err(GpuError::shape(
                WHAT_SLOTS,
                format!(
                    "{} heads; a pass of several slots runs into one head of its rows",
                    heads.len()
                ),
            )),
        }
    }

    /// The commit of the slot's waiting verify at its first position past
    /// the kept rows ([`Body38::keep_slot`]).
    fn keep_slot(
        &mut self,
        gpu: &Gpu,
        seq: Option<&mut Slot38>,
        r: &SlotRange,
        kept: usize,
    ) -> Result<(), GpuError> {
        let at = r.pos0 + launch_u32(WHAT_SLOTS, "kept rows", kept)?;
        Body38::keep_slot(self, gpu, seq, at)
    }
}

/// What a pass of several slots' refusals name.
const WHAT_SLOTS: &str = "qwen4exp slots";

/// Whether a pass whose first rows are slot `first`'s runs no row of slot 0
/// (slot 0, when busy, lays its rows first, [`SlotRowOrder::RetainedPrefix`]): its walk writes
/// that slot's own final-stream rows in place of the arena's
/// ([`Body38::enqueue_slots`]), and the slot's copy is not launched.
fn stands_in(first: usize) -> bool {
    first != 0
}

/// `f` on `b` with `seq` live — exchanged in and back out
/// ([`Body38::exchange`]) — or with the live sequence when `None`.
fn on_seq<T>(
    b: &mut Body38,
    seq: Option<&mut Slot38>,
    f: impl FnOnce(&mut Body38) -> Result<T, GpuError>,
) -> Result<T, GpuError> {
    match seq {
        None => f(b),
        Some(seq) => {
            b.exchange(seq);
            let r = f(b);
            b.exchange(seq);
            r
        }
    }
}

/// `f` on `b` for each range of `rows` in order, with that slot's sequence
/// live ([`on_seq`]): slot 0's the live one, every other slot's the next
/// of `parked`.
fn for_slots(
    b: &mut Body38,
    parked: &mut [&mut Slot38],
    rows: &[SlotRange],
    mut f: impl FnMut(&mut Body38, &SlotRange) -> Result<(), GpuError>,
) -> Result<(), GpuError> {
    let mut others = parked.iter_mut();
    for r in rows {
        let seq = match r.slot {
            0 => None,
            _ => Some(&mut **others.next().ok_or_else(|| r.no_parked(WHAT_SLOTS))?),
        };
        on_seq(b, seq, |b| f(b, r))?;
    }
    Ok(())
}

const _: () = assert!(GDN.n_v == geo::V_HEADS);
const _: () = assert!(VERIFY_ROWS <= PASS_ROWS);

#[cfg(test)]
mod tests {
    use super::{
        Arc, Device, E4m3Rows, GgmlType, GpuError, Nvfp4Rows, PleHost, Split, checkpoint_budget38,
        checkpoint_bytes, checkpoint_lens, checkpoint_owners, checkpoint_price38, dequant_row, geo,
        ple_takes,
    };
    use engram::hash::{EosWindow, Window};

    /// Every position of a saved text history rejects image and out-of-vocab
    /// ids; text ids, including EOS and the last vocabulary id, pass.
    #[test]
    fn saved_ple_histories_take_only_text_ids() {
        use super::{History, check_ple_history};

        let window = Window::Eos(EosWindow {
            eos: 1,
            image: Some(9),
            n_vocab: 16,
        });
        let good = [0, 1, 15];
        check_ple_history(&History::of(good.to_vec(), 7), 3, &window).unwrap();
        for (id, says) in [(9, "image"), (16, "vocabulary"), (u32::MAX, "vocabulary")] {
            for i in 0..good.len() {
                let mut tokens = good;
                tokens[i] = id;
                let why = check_ple_history(&History::of(tokens.to_vec(), 7), 3, &window)
                    .unwrap_err()
                    .to_string();
                assert!(
                    why.contains("a saved PLE history") && why.contains(says),
                    "{why}"
                );
            }
        }
        let why = check_ple_history(&History::of(vec![0, 1], 7), 3, &window)
            .unwrap_err()
            .to_string();
        assert!(why.contains("a PLE history of 2 tokens"), "{why}");
        let mapped = Window::Mapped {
            token_map: vec![0; 16],
            pad: 0,
        };
        let why = check_ple_history(&History::of(good.to_vec(), 7), 3, &mapped)
            .unwrap_err()
            .to_string();
        assert!(why.contains("a hash of mapped ids"), "{why}");
    }

    /// A checkpoint of Qwen3.8-Flash-Next's 36 delta layers and 12
    /// selecting layers (48 layers at interval 4) copies each delta layer's
    /// committed lane (786,432 f32), conv ring (112,640 f32) and carry
    /// (132,612 f32), the PLE ring (174,080 f32), and each selecting layer's
    /// raw key ring (eleven rows of 128 f16, 704 f32 words): 149,292,608 B,
    /// one derivation for the sequences' `Checkpoints` and the plan's price,
    /// spaced at the 4,096-token ubatch, so a slot of 2,048 positions pins two
    /// points (a whole call's one and the previous call's end), one of 10,240
    /// four, and the host budget fits 28.
    #[test]
    fn a_checkpoint_copies_the_delta_layers_lane_ring_the_ple_ring_and_the_key_rings() {
        let lens = checkpoint_lens(36, 12);
        assert_eq!(lens.len(), 3 * 36 + 1 + 12);
        assert!(
            lens[..108]
                .chunks(3)
                .all(|c| c == [786_432, 112_640, 132_612])
        );
        assert_eq!(lens[108], 174_080);
        assert!(lens[109..].iter().all(|&n| n == 704));
        assert_eq!(checkpoint_bytes(36, 12), 149_292_608);
        assert_eq!(checkpoint_price38(36, 12).point_bytes, 149_292_608);
        assert_eq!(checkpoint_price38(36, 12).every, 4_096);
        for (ctx, points) in [(2_048u64, 2u64), (10_240, 4), (1 << 20, 28)] {
            assert_eq!(
                checkpoint_budget38(36, 12, ctx).expect("a budget"),
                points * 149_292_608,
                "ctx {ctx}"
            );
        }
    }

    /// An end mark's copy of each store waits for the layer that writes it:
    /// a delta layer's three stores for that layer, the PLE ring for its
    /// site, each selecting layer's key ring for that selecting layer (not
    /// the site, which an earlier selecting layer's ring would pass while a
    /// later layer still appends its keys).
    #[test]
    fn an_end_marks_store_waits_for_the_layer_that_writes_it() {
        let delta = [true, true, true, false, true, true, true, false];
        let owners = checkpoint_owners(delta.iter().copied(), 1);
        assert_eq!(owners.len(), checkpoint_lens(6, 2).len());
        let mut want: Vec<usize> = [0, 1, 2, 4, 5, 6].iter().flat_map(|&l| [l; 3]).collect();
        want.extend([1, 3, 7]);
        assert_eq!(owners, want);
    }

    /// The call's check is the hash's: a text id passes; the image
    /// placeholder outside the call's image spans and an id past the
    /// vocabulary are refused by name with their index; the placeholder
    /// passes inside a span, by its row in the call; a hash that is not
    /// raw-id is refused.
    #[test]
    fn a_call_is_checked_by_the_hashs_own_test() {
        let w = Window::Eos(EosWindow {
            eos: 1,
            image: Some(9),
            n_vocab: 16,
        });
        assert!(ple_takes("t", &w, &[0, 1, 15], &[]).is_ok());
        for (ids, at) in [([2, 3, 9], "token 2:"), ([15, 16, 2], "token 1:")] {
            let e = ple_takes("t", &w, &ids, &[]).expect_err("the id is refused");
            assert!(e.to_string().contains(at), "{ids:?}: {e}");
        }
        let mapped = Window::Mapped {
            token_map: vec![0; 16],
            pad: 0,
        };
        let e = ple_takes("t", &mapped, &[0], &[]).expect_err("a mapped hash is refused");
        assert!(e.to_string().contains("a raw-id PLE hash"), "{e}");
        let (both, first) = (1..3, 1..2);
        let ids = [2, 9, 9, 3];
        assert!(ple_takes("t", &w, &ids, std::slice::from_ref(&both)).is_ok());
        let e = ple_takes("t", &w, &ids, std::slice::from_ref(&first)).expect_err("past the end");
        assert!(e.to_string().contains("token 2:"), "{e}");
        let e = ple_takes("t", &w, &[9, 2], std::slice::from_ref(&first)).expect_err("before it");
        assert!(e.to_string().contains("token 0:"), "{e}");
    }

    /// The header of a one-file split of the PLE keys (16 heads of 5 rows, 160 values a row)
    /// and the table of type id `ty` in `nbytes`, with its `.scale` when `scaled`.
    fn ple_layout(ty: u32, nbytes: usize, scaled: bool) -> gguf::write::Layout {
        use gguf::Value;
        use gguf::write::{Layout, TensorDecl};
        let k = |s: &str| format!("qwen4exp.ple.{s}");
        let u64s = |v: Vec<u64>| Value::Array(v.into_iter().map(Value::U64).collect());
        let kvs = [
            (k("layers"), Value::Array(vec![Value::I32(1)])),
            (k("ngram_size"), Value::U32(3)),
            (k("heads_per_ngram"), Value::U32(8)),
            (k("eos_token_id"), Value::U32(1)),
            (
                "qwen4exp.embedding_length_per_layer_input".to_string(),
                Value::U32(160),
            ),
            (k("layer_multipliers"), u64s(vec![3, 5, 7])),
            (k("head_vocab_sizes"), u64s(vec![5; 16])),
            (k("head_offsets"), u64s((0..16).map(|h| 5 * h).collect())),
        ];
        let decl = |name: &str, dims: Vec<u64>, type_id, nbytes: usize| TensorDecl {
            name: name.to_string(),
            dims,
            type_id,
            nbytes: nbytes as u64,
        };
        let mut tensors = vec![decl(
            "per_layer_token_embd.weight",
            vec![160, 80],
            ty,
            nbytes,
        )];
        if scaled {
            tensors.push(decl("per_layer_token_embd.scale", vec![1], 0, 4));
        }
        Layout::new(&kvs, tensors).unwrap()
    }

    /// The split of [`ple_layout`] holding the table `rows` and `scale` as its `.scale` when
    /// given, opened as the body opens its file, and the PLE host over it. The file, its own
    /// whichever test runs beside it, is removed once the host is open.
    fn ple_host(ty: u32, rows: &[u8], scale: Option<f32>, cap: usize) -> Result<PleHost, GpuError> {
        use gguf::write::Writer;
        use std::sync::atomic::{AtomicUsize, Ordering};
        static FILES: AtomicUsize = AtomicUsize::new(0);
        let layout = ple_layout(ty, rows.len(), scale.is_some());
        let path = std::env::temp_dir().join(format!(
            "body38-ple-{ty}-{}-{}.gguf",
            std::process::id(),
            FILES.fetch_add(1, Ordering::Relaxed)
        ));
        let mut w = Writer::new(std::fs::File::create(&path).unwrap(), layout).unwrap();
        w.tensor("per_layer_token_embd.weight", rows).unwrap();
        if let Some(scale) = scale {
            w.tensor("per_layer_token_embd.scale", &scale.to_le_bytes())
                .unwrap();
        }
        w.finish().unwrap();
        let host = PleHost::new(
            Arc::new(Split::open(&path).unwrap()),
            None,
            16,
            cap,
            Device::Host,
        );
        std::fs::remove_file(&path).unwrap();
        host
    }

    /// The PLE host decodes the rows the hash names by the table's type: an
    /// IQ4_NL table as `dequant_row` does, tables in bloomery's F8 and NVFP4
    /// rows as `E4m3Rows` and `Nvfp4Rows` at the scale beside them, each row
    /// at its place in the position's values. An F8 or NVFP4 table without
    /// its scale is refused by name.
    #[test]
    fn ple_rows_decode_by_the_tables_type() {
        let f8: Vec<u8> = (0..80 * 160u32)
            .map(|i| (i * 37 % 256) as u8)
            .map(|b| if b & 0x7F == 0x7F { 0x38 } else { b })
            .collect();
        let iq4: Vec<u8> = (0..80 * 5u32)
            .flat_map(|blk| {
                let qs = (0..16u32).map(move |j| ((blk * 31 + j * 7) % 256) as u8);
                [0x00, 0x3c].into_iter().chain(qs)
            })
            .collect();
        let fp4: Vec<u8> = (0..80 * 10u32)
            .flat_map(|g| {
                let qs = (0..8u32).map(move |j| ((g * 29 + j * 11) % 256) as u8);
                [0x30 + (g % 16) as u8].into_iter().chain(qs)
            })
            .collect();
        let tokens = [7u32, 3, 9, 2];
        for (ty, table, row_bytes) in [
            (GgmlType::F8E4M3_ROWS.as_u32(), &f8, 160),
            (GgmlType::NVFP4_ROWS.as_u32(), &fp4, 90),
            (GgmlType::IQ4_NL.as_u32(), &iq4, 90),
        ] {
            let mut host = ple_host(ty, table, Some(0.375), tokens.len()).unwrap();
            host.fill(0, &tokens).unwrap();
            assert_eq!(host.named, tokens.len() * 16);
            for (i, &row) in host.ids[..host.named].iter().enumerate() {
                let src = &table[row as usize * row_bytes..][..row_bytes];
                let mut want = [0.0f32; 160];
                if ty == GgmlType::IQ4_NL.as_u32() {
                    dequant_row(GgmlType::IQ4_NL, src, &mut want).unwrap();
                } else if ty == GgmlType::NVFP4_ROWS.as_u32() {
                    Nvfp4Rows::new(0.375).decode(src, &mut want).unwrap();
                } else {
                    E4m3Rows::new(0.375).decode(src, &mut want).unwrap();
                }
                let got = &host.e[i * 160..][..160];
                assert!(
                    got.iter()
                        .zip(&want)
                        .all(|(g, w)| g.to_bits() == w.to_bits()),
                    "type {ty} value row {i} (table row {row})"
                );
            }
        }
        for (ty, table) in [
            (GgmlType::F8E4M3_ROWS.as_u32(), &f8),
            (GgmlType::NVFP4_ROWS.as_u32(), &fp4),
        ] {
            let e = ple_host(ty, table, None, 1)
                .err()
                .expect("a scaled table without its scale");
            assert!(
                e.to_string()
                    .contains("per_layer_token_embd.scale is not in the file"),
                "{e}"
            );
        }
    }

    /// A checkpoint folder's F8 table, its rows in runs of two files where the folder stores
    /// them and its header alone in the split, decodes from the NVMe tier's reader the values
    /// the split holding the same rows decodes on the host; on the host, and as rows of another
    /// type, the folder's table is refused by name. The NVMe tier's fill counts its wait for the
    /// rows as the engine's PLE wait.
    #[test]
    fn a_folder_s_stored_rows_decode_as_the_split_s() {
        let f8: Vec<u8> = (0..80 * 160u32)
            .map(|i| (i * 53 % 256) as u8)
            .map(|b| if b & 0x7F == 0x7F { 0x38 } else { b })
            .collect();
        let tokens = [7u32, 3, 9, 2, 1, 5];
        let mut want = ple_host(GgmlType::F8E4M3_ROWS.as_u32(), &f8, Some(0.375), 8).unwrap();
        want.fill(0, &tokens).unwrap();
        let dir = std::env::temp_dir().join(format!("body38-ple-folder-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (a, b) = (dir.join("a.safetensors"), dir.join("b.safetensors"));
        let cut = 30 * 160;
        std::fs::write(&a, [&[9u8; 7][..], &f8[..cut]].concat()).unwrap();
        std::fs::write(&b, &f8[cut..]).unwrap();
        let stored = model::part::StoredRows {
            values: model::part::StoredPlane {
                row_bytes: 160,
                runs: vec![(a, 7..7 + cut as u64), (b, 0..(f8.len() - cut) as u64)],
            },
            block_scales: None,
            scale: 0.375,
        };
        let open = |ty: GgmlType, tier| {
            let row = if ty == GgmlType::F8E4M3_ROWS { 160 } else { 90 };
            let layout = ple_layout(ty.as_u32(), 80 * row, true);
            let header = Split::header_only(dir.clone(), &layout).unwrap();
            PleHost::new(Arc::new(header), Some(&stored), 16, 8, tier)
        };
        let mut got = open(GgmlType::F8E4M3_ROWS, Device::Nvme).unwrap();
        let waited = crate::waits::ple_waited();
        got.fill(0, &tokens).unwrap();
        assert!(
            crate::waits::ple_waited() > waited,
            "a fill off the NVMe tier waited for its rows"
        );
        assert_eq!(got.ids[..got.named], want.ids[..want.named]);
        assert!(
            got.e.len() == want.e.len()
                && got
                    .e
                    .iter()
                    .zip(&want.e)
                    .all(|(g, w)| g.to_bits() == w.to_bits())
        );
        for (ty, tier, need) in [
            (GgmlType::F8E4M3_ROWS, Device::Host, "on the NVMe tier"),
            (
                GgmlType::IQ4_NL,
                Device::Nvme,
                "F8 or NVFP4 rows where a checkpoint",
            ),
            (
                GgmlType::NVFP4_ROWS,
                Device::Nvme,
                "value and block-scale planes",
            ),
        ] {
            let e = open(ty, tier).err().expect("refused");
            assert!(e.to_string().contains(need), "{ty}: {e}");
        }
        let codes = &f8[..80 * 80];
        let scales: Vec<u8> = (0..80 * 10).map(|i| 0x30 + (i % 48) as u8).collect();
        let packed: Vec<u8> = codes
            .as_chunks::<8>()
            .0
            .iter()
            .zip(&scales)
            .flat_map(|(c, &s)| std::iter::once(s).chain(c.iter().copied()))
            .collect();
        let mut want = ple_host(GgmlType::NVFP4_ROWS.as_u32(), &packed, Some(0.375), 8).unwrap();
        let (a, b) = (dir.join("fp4-a.safetensors"), dir.join("fp4-b.safetensors"));
        let (cc, sc) = (30 * 80, 30 * 10);
        std::fs::write(&a, [&[9u8; 7][..], &codes[..cc], &scales[..sc]].concat()).unwrap();
        std::fs::write(&b, [&codes[cc..], &scales[sc..]].concat()).unwrap();
        let stored = model::part::StoredRows {
            values: model::part::StoredPlane {
                row_bytes: 80,
                runs: vec![
                    (a.clone(), 7..7 + cc as u64),
                    (b.clone(), 0..(codes.len() - cc) as u64),
                ],
            },
            block_scales: Some(model::part::StoredPlane {
                row_bytes: 10,
                runs: vec![
                    (a, 7 + cc as u64..7 + cc as u64 + sc as u64),
                    (
                        b,
                        (codes.len() - cc) as u64..(codes.len() - cc + scales.len() - sc) as u64,
                    ),
                ],
            }),
            scale: 0.375,
        };
        let layout = ple_layout(GgmlType::NVFP4_ROWS.as_u32(), packed.len(), true);
        let header = Arc::new(Split::header_only(dir.clone(), &layout).unwrap());
        let mut got =
            PleHost::new(Arc::clone(&header), Some(&stored), 16, 8, Device::Nvme).unwrap();
        // A call with a checkpoint mark inside it (position 4) reads every
        // unit after its first ahead, across the mark.
        let units = super::call_units(0, tokens.len(), 2, &[4, 6]);
        got.prompt_start(0, &tokens, units.iter().map(|&(m, _)| m), &[])
            .unwrap();
        let mut at = 0;
        for (i, &(m, _)) in units.iter().enumerate() {
            let (pos, unit) = (at as u32, &tokens[at..at + m]);
            at += m;
            want.hist = want.fill(pos, unit).unwrap();
            let waited = crate::waits::ple_waited();
            got.hist = got.fill(pos, unit).unwrap();
            assert!(
                crate::waits::ple_waited() > waited,
                "unit {i}: the fill's wait for its rows on the helper is the engine's PLE wait"
            );
            assert_eq!(got.ids[..got.named], want.ids[..want.named]);
            assert_eq!(got.e.len(), want.e.len());
            assert!(
                got.e
                    .iter()
                    .zip(&want.e)
                    .all(|(g, w)| g.to_bits() == w.to_bits())
            );
            let ahead = got.nvme.as_ref().is_some_and(|r| r.flight);
            assert_eq!(
                ahead,
                i + 1 < units.len(),
                "unit {i}: the next unit read ahead"
            );
        }
        // A step's rows read ahead ([`PleHost::prefetch`]) leave the history
        // where it stands and reuse their storage: the next fill takes them
        // when they are its own, drains a guess of another id, and a second
        // guess replaces the first; each fill decodes the rows its own id names.
        for (guess, actual, replace) in [(7, 7, false), (3, 9, false), (2, 1, true)] {
            let pos = got.hist.next_pos() as u32;
            let before = got.hist.clone();
            let history_storage = got.ahead_hist.tokens().as_ptr();
            got.prefetch(pos, guess).unwrap();
            assert_eq!(got.hist, before, "{guess}: the history stands");
            assert_eq!(got.ahead_hist.tokens().as_ptr(), history_storage);
            let read = got.nvme.as_ref().unwrap();
            assert!(read.flight, "{guess}: its read started");
            let ids_storage = read.ahead[0].ids.as_ptr();
            if replace {
                got.prefetch(pos, actual).unwrap();
            }
            want.hist = want.fill(pos, &[actual]).unwrap();
            got.hist = got.fill(pos, &[actual]).unwrap();
            assert_eq!(got.hist, want.hist);
            assert_eq!(got.ids[..got.named], want.ids[..want.named]);
            assert!(
                got.e[..geo::HIDDEN]
                    .iter()
                    .zip(&want.e)
                    .all(|(g, w)| g.to_bits() == w.to_bits()),
                "{guess} then {actual}: the rows of {actual}"
            );
            let read = got.nvme.as_ref().unwrap();
            assert!(!read.flight && read.ahead.is_empty());
            assert_eq!(read.spare_ids.as_ptr(), ids_storage, "{guess}: ids reused");
        }
        let mut wrong_width = stored.clone();
        wrong_width.block_scales.as_mut().unwrap().row_bytes = 9;
        let e = PleHost::new(Arc::clone(&header), Some(&wrong_width), 16, 8, Device::Nvme)
            .err()
            .expect("refused");
        assert!(
            e.to_string().contains("value and block-scale planes"),
            "{e}"
        );
        let mut short = stored;
        short.block_scales.as_mut().unwrap().runs[1].1.end -= 10;
        let e = PleHost::new(header, Some(&short), 16, 8, Device::Nvme)
            .err()
            .expect("refused");
        assert!(e.to_string().contains("source plane has 79 rows"), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A prompt call's units: its runs cut at the checkpoint marks past its
    /// start, each as the ubatch cut, the unit before each mark marked; with
    /// no marks one run of the whole call.
    #[test]
    fn a_call_cuts_its_units_at_its_marks() {
        let (t, f) = (true, false);
        for (from, n, width, marks, want) in [
            (
                0,
                8192,
                4096,
                &[4096, 8192][..],
                &[(4096, t), (4096, t)][..],
            ),
            (0, 8192, 4096, &[], &[(4096, f), (4096, f)]),
            (100, 5000, 4096, &[100, 4096, 5100], &[(3996, t), (1004, t)]),
            (
                0,
                9000,
                4096,
                &[4096, 8192, 9000],
                &[(4096, t), (4096, t), (808, t)],
            ),
            (4090, 10, 8, &[4090, 4096, 4100], &[(6, t), (4, t)]),
            (0, 10, 8, &[10], &[(8, f), (2, t)]),
        ] {
            assert_eq!(
                super::call_units(from, n, width, marks),
                want,
                "{n} ids from {from}, {width} wide, marks {marks:?}"
            );
        }
    }
}
