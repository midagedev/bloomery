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
//! first position not kept) keeps the first `k`: every delta layer's state
//! after each row is in a lane of its own, so the commit moves the lane
//! word to row `k − 1`'s ([`kept_lane`]) and copies nothing; the PLE hash's
//! history is replayed over the kept rows. Every other store is by
//! position, and nothing reads a row past the count: a pooled key a
//! rejected row completed is read only at a count that completes it again,
//! and the row at that count writes it before its select reads it. A step,
//! a pass or a verify is refused by name while a verify waits for its
//! commit.
//!
//! The PLE rows are the host's: each position's n-gram hash
//! (`engram::Hash::ple_rows_into`) names 16 rows of the IQ4_NL table, which
//! stays in the file; they are decoded (`gguf::quant::dequant_row`) into the
//! position's 2,560 values and copied to the arena ahead of the step. The
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

use super::card38::{Card38, MapCheck, Walk38};
use super::mtp38::{
    DraftRows38, DraftSeq38, Held38, Mtp38, MtpCtx, MtpDraft, MtpFeed, MtpHead, MtpHidden, MtpMode,
    MtpTaps, Target38, TargetRows,
};
use super::plan38::{self, GDN, Kind38, Layer38, Shape38, beta_alpha, geo, router};
use super::program38::{
    Ctx38, Kernels38, Parts38, Pass38, RowsCopy, STEP_MEMOPS, Seqs38, SlotSeq38, Step38, Verify38,
    seq_launches, step_launches, verify_launches,
};
use super::scratch::{Io, LANE, RecStore, RopeRows, StepParams, f32_view};
use super::scratch38::{
    Arena38, LANES, LaneWord, PASS_ROWS, PassRecord, Store38, Taps38, VERIFY_ROWS, WideRecord,
    dims, kept_lane, store_rule_bytes,
};
use super::slot_pass::SlotIn;
use super::swap38::{DEADLINE, LIVE_DELAY, Qwen38Stacks};
use super::tier38::{TierSide38, open_tier};
use super::ubatch::UBATCH as UBATCH_MOST;
use super::wide38::{
    Gemm38, STREAM_FLOOR, Stream38, Wide38, WideForce, WideParts, WideTaps, dense_rows, pick_floor,
    route_taps_host,
};
use crate::checkpoint::Checkpoints;
use crate::checkpoint::saved::{Identity, Lent, SeqState, Spans};
use crate::head::{Head, HeadNorm};
use crate::host::run::{HostRun, HostWidths};
use crate::host::swap::{
    BoundaryAt, CallCfg, CallPick, CallReport, PassReport, ResetReport, Residency,
};
use crate::host::swap_source::{FileSwap, ResidencyGlue, ResidencySpec};
use crate::host::tier::TierOpen;
use crate::host::xstream::{Costs, XCfg, XLayer, XMode, XReport};
use crate::host::{BatchLeg, PassKind, StepLeg, refuse_tier_count};
use crate::hybrid::{Boundary, BoundaryShape, Chain, HostResidency, Hybrid, Refusal, SlotMap};
use crate::model::{
    ChainBody, GpuModel, HostServed, Rollback, RowHeads, Rows, SlotRange, SlotRows, Slots,
};
use crate::prompt_timing::{PromptStats, PromptTiming, nanos};
use crate::rope_table::{RopeSpec, RopeTable};
use crate::tensor::WindowMut;
use crate::weights::Weights;
use crate::{DeviceTensor, Gpu, GpuError, launch_u32, ple};
use bloomery_levers::HostCfg;
use cuda_core::{CudaStream, DeviceBuffer};
use engram::Hash;
use engram::hash::{History, Window};
use gguf::quant::dequant_row;
use gguf::{GgmlType, Split, TensorInfo};
use model::arch::Arch;
use model::placement::Plan;
use runtime::seqstate::{HOST_BUDGET, Kept, Take, Why};
use runtime::swaprule::KeptRows;
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

const WHAT: &str = "qwen4exp Body38";

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
/// ([`SeqState<History>`]), and on a load with the draft the layer's side
/// beside them ([`DraftRows38`]) — its store rows and the two arenas' rows,
/// so a sequence put back after another ran drafts as if none had.
pub struct Seq38 {
    state: SeqState<History>,
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
/// buffer by a copy ([`RowsCopy`]).
pub struct Slot38 {
    stores: Vec<Store38>,
    ple_ring: DeviceBuffer<f32>,
    lane: LaneWord,
    pending: Option<Pending38>,
    ckpt: Checkpoints,
    ple_points: Vec<(u32, History)>,
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
}

/// The host bytes a sequence state of `positions` positions holds on a load
/// of `gdn` delta and `qsa` selecting layers, with the draft's side when the
/// load carries it ([`Seq38::bytes`]'s terms at plan time, for a seat's park
/// budget): the positional rows and two recurrent copies
/// ([`seq_positional_bytes`], [`checkpoint_bytes`]), the draft's store rows
/// below the same positions, and the step's and the pass's arena rows — the
/// most a parked refresh reads.
#[must_use]
pub fn seq38_bytes(gdn: usize, qsa: usize, positions: usize, drafts: bool) -> u64 {
    let shared = seq_positional_bytes(qsa, positions) + 2 * checkpoint_bytes(gdn);
    if !drafts {
        return shared;
    }
    let store = 2 * (geo::N_KV * positions * geo::HEAD * size_of::<u16>()) as u64;
    let rows = ((1 + PASS_ROWS) * geo::STREAMS * geo::HIDDEN * size_of::<f32>()) as u64;
    shared + store + rows
}

/// The spacing of a prompt call's inner checkpoints (`runtime::seqstate`'s
/// marks), when the load takes them ([`Body38::set_checkpoints`]): the
/// ubatch walk's most positions, so a mark cuts a call's ubatches at most
/// once more than the call's own cut, and a context of 60k positions takes
/// 15 inner marks against the host budget's 32 slots of one copy
/// ([`checkpoint_bytes`]).
pub const CHECKPOINT_EVERY: u32 = UBATCH_MOST as u32;

/// The bytes one checkpoint of a load of `gdn` delta layers copies: each
/// delta layer's committed lane and conv ring, and the PLE ring.
#[must_use]
pub fn checkpoint_bytes(gdn: usize) -> u64 {
    ((gdn * (GDN.state_len() + GDN.ring_len()) + ple::ring_len(geo::STREAMS)) * size_of::<f32>())
        as u64
}

/// The positional bytes a sequence state of `positions` positions carries
/// for `qsa` selecting layers: each one's K and V rows, raw indexer keys and
/// the pooled keys of the pools the positions reach.
#[must_use]
pub const fn seq_positional_bytes(qsa: usize, positions: usize) -> u64 {
    let f16 = size_of::<u16>();
    let layer = 2 * geo::N_KV * positions * geo::HEAD * f16
        + positions * geo::IDX_DIM * f16
        + positions.div_ceil(geo::POOL) * geo::IDX_DIM * f16;
    (qsa * layer) as u64
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
/// in the file, and the rows of the positions last filled, decoded.
struct PleHost {
    hash: Hash,
    hist: History,
    file: Arc<Split>,
    shard: usize,
    info: TensorInfo,
    /// Table rows, values a row, bytes a row.
    rows: usize,
    width: usize,
    row_bytes: usize,
    /// Row ids of up to a call's positions (a pass's or a ubatch's), the
    /// count the last fill or keep named, and the fill's decoded values
    /// (`HIDDEN` a position).
    ids: Vec<u32>,
    named: usize,
    e: Vec<f32>,
}

impl PleHost {
    /// The table `per_layer_token_embd` of `file`, IQ4_NL rows of the hash's
    /// width, and the hash from the first shard's keys, for calls of up to
    /// `cap` positions. Load-time only.
    fn new(file: Arc<Split>, n_vocab: usize, cap: usize) -> Result<PleHost, GpuError> {
        let name = model::arch::qwen35moe::names::per_layer_token_embd();
        let refuse = |need: &'static str| GpuError::Tensor {
            what: WHAT,
            name: name.clone(),
            need,
        };
        let path = file
            .shard_path(0)
            .ok_or(GpuError::state(WHAT, "the file's first shard"))?;
        let meta = gguf::inventory_of(path).map_err(|e| GpuError::plan(WHAT, e))?;
        let vocab = u32::try_from(n_vocab)
            .map_err(|_| GpuError::shape(WHAT, format!("a vocabulary of {n_vocab}")))?;
        let hash =
            Hash::ple_from_gguf(&meta, "qwen4exp", vocab).map_err(|e| GpuError::plan(WHAT, e))?;
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
        if info.ty != GgmlType::IQ4_NL
            || width == 0
            || !width.is_multiple_of(32)
            || per_token * width != geo::HIDDEN
        {
            return Err(refuse("IQ4_NL rows of a whole part of the model width"));
        }
        let row_bytes = width / 32 * 18;
        if rows == 0 || info.nbytes != (rows * row_bytes) as u64 {
            return Err(refuse("a whole number of IQ4_NL rows"));
        }
        Ok(PleHost {
            hash,
            hist,
            file,
            shard,
            info,
            rows,
            width,
            row_bytes,
            ids: vec![0; cap * per_token],
            named: 0,
            e: vec![0.0; cap * geo::HIDDEN],
        })
    }

    /// The rows of `tokens` at positions `pos ..`, decoded into the first
    /// `tokens.len()` positions of `e`, and the history past them, returned:
    /// the history stays where it stands until the call's launch takes the
    /// returned one ([`Body38::launch`]). A token the hash refuses (the
    /// image placeholder, an id past the vocabulary), a position other than
    /// the history's next, or a row past the table is refused by name.
    fn fill(&mut self, pos: u32, tokens: &[u32]) -> Result<History, GpuError> {
        let n = tokens.len();
        let per_token = geo::HIDDEN / self.width;
        let ids = self
            .ids
            .get_mut(..n * per_token)
            .ok_or_else(|| GpuError::shape(WHAT, format!("{n} PLE positions at once")))?;
        let mut hist = self.hist.clone();
        self.named = 0;
        self.hash
            .ple_rows_into(&mut hist, u64::from(pos), tokens, ids)
            .map_err(|e| GpuError::shape(WHAT, format!("PLE rows at position {pos}: {e}")))?;
        self.named = ids.len();
        if let Some(&r) = ids.iter().find(|&&r| r as usize >= self.rows) {
            return Err(GpuError::shape(
                WHAT,
                format!("PLE row {r} past the table's {} rows", self.rows),
            ));
        }
        let data = self
            .file
            .shard(self.shard)
            .ok_or(GpuError::state(WHAT, "the PLE table's shard"))?
            .data(&self.info)?;
        for (j, &r) in ids.iter().enumerate() {
            let src = &data[r as usize * self.row_bytes..][..self.row_bytes];
            let out = &mut self.e[j * self.width..][..self.width];
            dequant_row(GgmlType::IQ4_NL, src, out).map_err(model::ModelError::from)?;
        }
        Ok(hist)
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

    /// A new sequence's history.
    fn restart(&mut self) -> Result<(), GpuError> {
        self.hist = self
            .hash
            .new_history()
            .map_err(|e| GpuError::shape(WHAT, e.to_string()))?;
        Ok(())
    }
}

/// Refused by name, `what` the caller: the first of `tokens` the raw-id PLE
/// hash of `window` does not take (the image placeholder, an id past the
/// vocabulary), with its index in the call, or a hash that is not raw-id.
/// The hash's own test ([`engram::hash::EosWindow::check`]), so a call's
/// check before its first launch and each fill refuse the same ids.
fn ple_takes(what: &'static str, window: &Window, tokens: &[u32]) -> Result<(), GpuError> {
    let Window::Eos(w) = window else {
        return Err(GpuError::state(what, "a raw-id PLE hash"));
    };
    for (i, &t) in tokens.iter().enumerate() {
        w.check(t)
            .map_err(|e| GpuError::shape(what, format!("token {i}: {e}")))?;
    }
    Ok(())
}

/// A layer's store read back ([`Body38::stores_host`]): a delta layer's
/// state and conv ring, or a selecting layer's K/V planes (f16 bits) and its
/// raw and pooled indexer keys.
#[derive(Clone, Debug, PartialEq)]
pub enum Store38Host {
    Rec {
        state: Vec<f32>,
        ring: Vec<f32>,
    },
    Qsa {
        k: Vec<u16>,
        v: Vec<u16>,
        raw: Vec<u16>,
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
                Store38Host::Rec { state: s, ring: r },
                Store38Host::Rec {
                    state: s2,
                    ring: r2,
                },
            ) => bits(s, s2) && bits(r, r2),
            (Store38Host::Qsa { .. }, Store38Host::Qsa { .. }) => self == other,
            _ => false,
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
/// the PLE history as it stood before them.
struct Pending38 {
    pos0: u32,
    tokens: Vec<u32>,
    hist: History,
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
/// expert stream reads beside the lane's measured rate: the host union's
/// no-tile cost a column and its floor for an expert of the file's common
/// layer kind on the pool's threads, and the card route's fixed and
/// per-column cost of one streamed expert (the measured union bench and the
/// q38seed card's decomposition, `docs/cards/q38seed-ab.card`).
pub const XSTREAM_COSTS: Costs = Costs {
    host_us_per_col: 7.0,
    host_us_fixed: 23.36,
    card_us_fixed: 13.0,
    card_us_per_col: 0.16,
};

/// Everything Qwen3.8's chain owns. Field order is drop order: the host tier
/// (its lock over the host set, its page windows) before the buffers.
pub struct Body38 {
    hybrid: Hybrid<HostRun>,
    plans: Vec<Layer38>,
    stores: Vec<Store38>,
    ple_ring: DeviceBuffer<f32>,
    ple: PleHost,
    rope: RopeRows,
    /// The step's one-row arena and record; an eager pass's arena, record
    /// and host sums.
    s: Arena38,
    sp: StepParams,
    a: Arena38,
    rp: PassRecord,
    pass_hsum: DeviceBuffer<f32>,
    k: Kernels38,
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
    stream: Stream38,
    /// The map check every walk reads ([`MapCheck::refuse`]): the tier's
    /// slot map's, or the one a gate planted in its place until it is taken
    /// back or the model reset.
    card_leg: MapCheck,
    /// The lane word every delta launch reads, and the verify waiting for
    /// its commit.
    lane: LaneWord,
    pending: Option<Pending38>,
    /// Each layer's streams after it and its route, when a gate armed them.
    taps: Option<Taps38>,
    /// The loaded `output`'s form (`mtp::output_form`): the step's launch
    /// count's head term ([`step_launches`]).
    head_form: model::arch::qwen35moe::mtp::BorrowedHead,
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
    /// Each target arena's [`Held38`]: where the step, the pass (a verify's
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
    /// The recurrent stores on the host at chosen positions, and the PLE
    /// hash's history at each, ascending: the points a cut behind the fed
    /// positions restores. A prompt call takes them only while `marks` is on.
    ckpt: Checkpoints,
    ple_points: Vec<(u32, History)>,
    marks: bool,
    /// What a sequence state of this load belongs to.
    who: Identity,
}

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
    /// `1..=min(UBATCH, ctx)`, and a ubatch arena past what the plan's
    /// scratch counts.
    pub fn open_placed(
        file: Split,
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
        file: Split,
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
            None,
            slots,
        )
    }

    /// [`Body38::open_placed`] under `residency`: `mid-p<P>-s<S>` runs the
    /// residency machine over the card's routed stacks
    /// ([`crate::host::swap`], [`Qwen38Stacks`]) — the load's host set also
    /// holds each layer's churn pool (the card's experts past the first `P`),
    /// refused by name when the plan's host headroom cannot take it — while
    /// `off` is [`Body38::open_placed`] itself. The lever's word a load runs
    /// by is the registry's, read by the caller
    /// (`bloomery_levers::Levers::residency_at`). `slots` is
    /// [`Body38::open_placed_slots`]'s: the plan must have counted them.
    #[allow(
        clippy::too_many_arguments,
        reason = "open_placed's six, the residency and the planned slots (rust-quality R8)"
    )]
    pub fn open_placed_residency(
        file: Split,
        plan: &Plan<'_>,
        inputs: &model::arch::qwen35moe::place::PlanInputs,
        card: usize,
        host: HostCfg,
        ubatch: usize,
        residency: Residency,
        slots: usize,
    ) -> Result<Qwen38Model, GpuError> {
        Body38::open_with(
            file, plan, inputs, card, host, ubatch, residency, None, slots,
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
        file: Split,
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
        file: Split,
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
        reason = "open_placed_mtp's eight, the residency and the planned slots (rust-quality R8)"
    )]
    pub fn open_placed_mtp_residency(
        file: Split,
        plan: &model::arch::qwen35moe::place::MtpPlan<'_>,
        inputs: &model::arch::qwen35moe::place::PlanInputs,
        card: usize,
        host: HostCfg,
        ubatch: usize,
        draft: &Split,
        mtp: &model::arch::qwen35moe::place::MtpInputs,
        residency: Residency,
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
            Some((draft, mtp, plan)),
            slots,
        )
    }

    /// The load every constructor shares, under `residency`; `mtp` the
    /// draft's file, inputs and plan, when there is one, and `slots` the
    /// sequence slots the plan was made for ([`Slots`]).
    #[allow(
        clippy::too_many_arguments,
        reason = "open_placed's six, the residency, the draft's file, inputs and plan, and the planned slots (rust-quality R8)"
    )]
    fn open_with(
        file: Split,
        plan: &Plan<'_>,
        inputs: &model::arch::qwen35moe::place::PlanInputs,
        card: usize,
        host: HostCfg,
        ubatch: usize,
        residency: Residency,
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
        refuse_tier_count(WHAT, plan.machine.tiers.len())?;
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
        let counted = model::arch::qwen35moe::place::counted_ubatch_bytes(spec);
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
            delay: LIVE_DELAY,
            deadline: DEADLINE,
            top_k: geo::N_USED,
            max_rows: PASS_ROWS,
            stacks: Arc::new(Qwen38Stacks::of(&inputs.model)?),
        };
        GpuModel::load_placed_with(
            file,
            plan,
            card,
            host,
            spec,
            |stream, _, layers, w| Body38::derive(stream, &shape.kinds, layers, w),
            |gpu, file, w, set, glue| {
                let draft = mtp
                    .map(|(d, m, p)| Mtp38::open(gpu, w, d, m, p, host.card_dontneed, slots))
                    .transpose()?;
                let mut body = Body38::load_placed(
                    gpu,
                    file,
                    w,
                    plan,
                    inputs,
                    &shape,
                    host,
                    set,
                    glue,
                    (ubatch, counted, slots),
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
        reason = "the load's card handle, file, weights, plan, inputs and chain shape, the host tier's residency and levers, the residency glue, and the ubatch with the arena bytes the plan counts and the planned slots (rust-quality R8)"
    )]
    fn load_placed(
        gpu: &Gpu,
        file: &Arc<Split>,
        w: &Weights,
        plan: &Plan<'_>,
        inputs: &model::arch::qwen35moe::place::PlanInputs,
        shape: &Shape38,
        host: HostCfg,
        set: HostResidency,
        residency_glue: ResidencyGlue,
        (ub, counted, planned): (usize, u64, usize),
    ) -> Result<Body38, GpuError> {
        let (spec, hp) = (&inputs.spec, &inputs.hp);
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
        let stores = kinds
            .iter()
            .map(|k| match k {
                Kind38::Gdn => Store38::rec(stream),
                Kind38::Qsa => Store38::qsa(stream, &d),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (rec, sel) = store_rule_bytes(ctx);
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
        let s = Arena38::new(stream, router_dims, 1, ctx)?;
        let sp = StepParams::new(stream, true)?;
        let a = Arena38::new(stream, router_dims, PASS_ROWS, ctx)?;
        let rp = PassRecord::new(stream)?;
        let pass_hsum = DeviceBuffer::zeroed(stream, PASS_ROWS * geo::HIDDEN)?;
        let wa = Arena38::wide(stream, router_dims, ub, UBATCH_MOST, ctx)?;
        let wr = WideRecord::new(stream, ub)?;
        let wide_hsum = DeviceBuffer::zeroed(stream, ub * geo::HIDDEN)?;
        let host_cols = ub.max(PASS_ROWS);
        let run = 0..n;
        let tiers = TierOpen::of_machine(plan.machine);
        let tier_cards: Vec<usize> = tiers.iter().map(|t| t.card).collect();
        let map = SlotMap::of_plan_tiers(plan, 0, &tier_cards, run.clone(), geo::EXPERTS)?;
        let card_leg = MapCheck::of(&map, !tiers.is_empty())?;
        let slots = Arc::new(DeviceTensor::upload(
            stream,
            &map.stage_view(),
            n,
            geo::EXPERTS,
        )?);
        let stacks = Qwen38Stacks::of(&inputs.model)?;
        let card = Card38::new(gpu, w, &stacks, &map, n)?;
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
        let tier = (!tiers.is_empty())
            .then(|| TierSide38::new(gpu, &map))
            .transpose()?;
        let wide = Wide38::new(stream, ub, &card, tier.is_some())?;
        let wide_bytes = (wa.bytes() + wr.bytes() + wide_hsum.num_bytes() + wide.bytes()) as u64;
        if wide_bytes > counted {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a ubatch of {ub} holds {wide_bytes} card bytes, the plan counts {counted}"
                ),
            ));
        }
        // A load planned for several sequences runs a pass of their rows
        // through the step port's columns, up to the pass arena's rows; a
        // load for one sequence the verify's.
        let cols = if planned > 1 { PASS_ROWS } else { VERIFY_ROWS };
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
        let mut experts = HostRun::build(Arc::clone(file), 0, host.r8, widths, |src| {
            model::arch::qwen35moe::host::layers(src, hp, run.clone())
        })?;
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
                    &stacks,
                    n,
                    block,
                    host.card_dontneed,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut hybrid = Hybrid::new(boundary, map, experts, n)?;
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
            hybrid.check_tier_reserves(plan.machine, &cards)?;
        }
        let ple = PleHost::new(Arc::clone(file), spec.vocab as usize, host_cols)?;
        let lens = stores
            .iter()
            .filter(|s| matches!(s, Store38::Rec { .. }))
            .flat_map(|_| [GDN.state_len(), GDN.ring_len()])
            .chain([ple_ring.len()])
            .collect();
        let ckpt = Checkpoints::new(gpu.context(), lens, HOST_BUDGET, CHECKPOINT_EVERY)?;
        let who = Identity {
            arch: "qwen4exp",
            file: file
                .shard_path(0)
                .ok_or(GpuError::state(WHAT, "the file's first shard"))?
                .to_path_buf(),
            card: gpu.device_name()?,
            ctx,
        };
        // The loaded `output`'s form, the step's head-count term: the one
        // owner of the list is `mtp::output_form` (the borrow check asks it
        // too); a type it does not take (a Q4_K head, no file today) counts
        // as the Q8_0 form's, which reads the f32 rows with no quantize.
        let head_form = match w.get(&model::arch::qwen35moe::names::output()) {
            Some(crate::weights::DevWeight::KQuant { ty, .. }) => {
                model::arch::qwen35moe::mtp::output_form(*ty)
                    .unwrap_or(model::arch::qwen35moe::mtp::BorrowedHead::Q8_0)
            }
            _ => model::arch::qwen35moe::mtp::BorrowedHead::Q8_0,
        };
        let mut body = Body38 {
            hybrid,
            plans,
            stores,
            ple_ring,
            ple,
            rope,
            s,
            sp,
            a,
            rp,
            pass_hsum,
            k: Kernels38::load(gpu)?,
            slots,
            card,
            tier,
            wa,
            wr,
            wide_hsum,
            wide,
            wide_timing: None,
            stream: Stream38::default(),
            card_leg,
            lane: LaneWord::new(stream)?,
            pending: None,
            taps: None,
            eps: spec.rms_eps,
            vocab: spec.vocab as usize,
            head_form,
            ctx,
            held: 0,
            staged: None,
            slots_planned: planned,
            slots_made: 1,
            slot_in: (planned > 1).then(|| SlotIn::new(stream)).transpose()?,
            last_walk: TargetRows::Step,
            wrote: Wrote38::default(),
            plant: None,
            mtp: None,
            residency_glue,
            ckpt,
            ple_points: Vec::new(),
            marks: false,
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

    /// The captured step's launches ([`super::program38`]'s count), and how
    /// many of them are stream memory-operation batches.
    #[must_use]
    pub fn step_launches(&self) -> (usize, usize) {
        (
            step_launches(&self.plans, &self.card, self.tier.as_ref(), self.head_form),
            STEP_MEMOPS * self.plans.len(),
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
    /// of `walk`'s arena ran its head: the walk's current buffer flips once
    /// at each PLE layer past layer 0.
    fn final_streams(&self, walk: TargetRows) -> &DeviceBuffer<f32> {
        final_streams(&self.plans, [&self.s, &self.a, &self.wa], walk)
    }

    /// The target's hidden rows `0..rows` after a walk of `walk`'s arena
    /// that ran its head, four streams of [`geo::HIDDEN`] a row. Blocking;
    /// gate use.
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
        feed: MtpFeed<'_>,
        head: MtpHead,
        mode: MtpMode,
    ) -> Result<MtpDraft, GpuError> {
        let Body38 {
            mtp,
            k,
            rope,
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
            &mtp_ctx(gpu, w, k, *eps, &rope.table),
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
        feed: MtpFeed<'_>,
        head: MtpHead,
        mode: MtpMode,
    ) -> Result<usize, GpuError> {
        let Body38 {
            mtp,
            k,
            rope,
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
            &mtp_ctx(gpu, w, k, *eps, &rope.table),
            target,
            feed,
            head,
            mode,
        )
    }

    /// One window's chain of the draft's program ([`Mtp38::run_chain`])
    /// beside this body: `refresh`'s walk then `own` own walks, one readback.
    fn mtp_run_chain(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        refresh: MtpFeed<'_>,
        own: usize,
        head: MtpHead,
        mode: MtpMode,
    ) -> Result<MtpDraft, GpuError> {
        let Body38 {
            mtp,
            k,
            rope,
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
            &mtp_ctx(gpu, w, k, *eps, &rope.table),
            target,
            refresh,
            own,
            head,
            mode,
        )
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

    /// The host tier.
    #[must_use]
    pub fn hybrid(&self) -> &Hybrid<HostRun> {
        &self.hybrid
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
    /// `Split` on a load that hangs an expert tier or when the stream cannot
    /// start (the card's room among the refusals), and any change while a
    /// call streams.
    pub fn set_xstream(&mut self, gpu: &Gpu, mode: XMode) -> Result<(), GpuError> {
        const WHAT_S: &str = "Body38::set_xstream";
        if self.stream.on {
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
        if mode == XMode::Split && self.hybrid.xstream().is_none() {
            let cfg = XCfg {
                costs: XSTREAM_COSTS,
                experts: geo::EXPERTS,
                top_k: geo::N_USED,
                max_half: 0,
                keep_free: crate::host::xstream::KEEP_FREE,
                min_half: crate::host::xstream::MIN_HALF,
                fill_threads: crate::host::xstream::FILL_THREADS,
                staging_slots: crate::host::xstream::STAGING_SLOTS,
                deadline: DEADLINE,
                probe: Vec::new(),
            };
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
        self.stream.mode = Some(mode);
        Ok(())
    }

    /// The expert stream's rule costs in place of [`XSTREAM_COSTS`]
    /// ([`crate::host::xstream::XStream::set_costs`]). Refused by name before
    /// `split` started the stream and while a call streams.
    pub fn set_xstream_costs(&mut self, costs: Costs) -> Result<(), GpuError> {
        const WHAT_C: &str = "Body38::set_xstream_costs";
        if self.stream.on {
            return Err(GpuError::state(WHAT_C, "no prompt call streaming"));
        }
        self.hybrid
            .xstream_mut()
            .ok_or(GpuError::state(
                WHAT_C,
                "an expert stream (Body38::set_xstream's split starts it)",
            ))?
            .set_costs(costs)
    }

    /// What the next prompt calls move ([`Body38::set_xstream`]).
    #[must_use]
    pub fn xstream_mode(&self) -> XMode {
        self.stream.mode.unwrap_or(XMode::Off)
    }

    /// The load line's `xstream=` word: the mode, and under a started
    /// stream its ring's slots a half, its pinned staging slots (0:
    /// pageable), the lane's measured rate in GB/s and the least unit width
    /// that streams.
    #[must_use]
    pub fn xstream_word(&self) -> String {
        let mode = self.xstream_mode().word();
        match self.hybrid.xstream() {
            Some(x) => format!(
                "{mode} xstream_half={} xstream_staging={} xstream_lane_gbs={:.2} \
                 xstream_m_min={}",
                x.half_slots(),
                x.staging_slots(),
                x.lane_b_per_us() / 1000.0,
                x.m_min().map_or("none".to_string(), |m| m.to_string())
            ),
            None => mode.to_string(),
        }
    }

    /// The last streaming prompt call's streamed layers, each with its
    /// ubatch, and the stream's end; empty and `None` after a call that did
    /// not stream. Taken: a second read is empty.
    pub fn take_xstream_records(&mut self) -> (Vec<(usize, XLayer)>, Option<XReport>) {
        (
            std::mem::take(&mut self.stream.layers),
            self.stream.xend.take(),
        )
    }

    /// The last streaming prompt call's picks, each with its ubatch, and its
    /// end; empty and `None` after a call that did not stream. Taken: a
    /// second read is empty.
    pub fn take_stream_records(&mut self) -> (Vec<(usize, CallPick)>, Option<CallReport>) {
        (
            std::mem::take(&mut self.stream.picks),
            self.stream.end.take(),
        )
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
    /// the committed lane holds it. Blocking.
    pub fn stores_host(&self, gpu: &Gpu) -> Result<(Vec<Store38Host>, Vec<f32>), GpuError> {
        let stream = gpu.stream();
        let stores = self
            .stores
            .iter()
            .map(|s| {
                Ok(match s {
                    Store38::Rec { rec, .. } => Store38Host::Rec {
                        state: lane_of(rec.state.to_host_vec(stream)?, self.lane.lane())?,
                        ring: rec.ring.to_host_vec(stream)?,
                    },
                    Store38::Qsa { kv, raw, pooled } => {
                        let (k, v) = kv.f16("qwen38::stores_host")?;
                        Store38Host::Qsa {
                            k: k.to_host_vec(stream)?,
                            v: v.to_host_vec(stream)?,
                            raw: raw.to_host_vec(stream)?,
                            pooled: pooled.to_host_vec(stream)?,
                        }
                    }
                })
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        Ok((stores, self.ple_ring.to_host_vec(stream)?))
    }

    /// Every selecting store's K/V planes and raw and pooled keys set to
    /// `bits`, an f16 pattern: a gate's stand-in for the rows `reset` leaves
    /// as they were, so a path that reads a row it did not write reads the
    /// pattern, not a previous run's value. Synchronizes; gate use.
    pub fn fill_planes(&mut self, gpu: &Gpu, bits: u16) -> Result<(), GpuError> {
        const WHAT: &str = "qwen38::fill_planes";
        let stream = gpu.stream();
        for s in &mut self.stores {
            if let Store38::Qsa { kv, raw, pooled } = s {
                let (k, v) = kv.f16_mut(WHAT)?;
                for buf in [k, v, raw, pooled] {
                    buf.copy_from_host(stream, &vec![bits; buf.len()])?;
                }
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

    /// The lane word set to `lane` with no verify behind it — a gate's
    /// stand-in for a commit that moved the word to a lane no call wrote:
    /// the next launch that reads it raises the stamp's fault. Gate use.
    pub fn plant_lane(&mut self, gpu: &Gpu, lane: u32) -> Result<(), GpuError> {
        self.lane.set(gpu.stream(), lane)
    }

    /// The captured verify's launches at `m` rows ([`super::program38`]'s
    /// count).
    #[must_use]
    pub fn verify_launches(&self, m: usize) -> usize {
        verify_launches(
            &self.plans,
            &self.card,
            self.tier.as_ref(),
            m,
            self.head_form,
        )
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
    /// verify's first position (`Rule`: its commit moves the lane word and
    /// copies nothing) and every position when `n` reaches them (`Current`);
    /// else the checkpoints' rule ([`Checkpoints::kept`]): every position, the
    /// empty model at 0, or the nearest point at or below `n`, nothing
    /// (`Missed`) when none lies there — a delta layer keeps no earlier state
    /// but a copy. After a call failed past its launch the stores hold more
    /// than `pos`, which is no longer kept. Never past `pos`. The rule
    /// [`Body38::commit`] takes.
    #[must_use]
    pub fn kept(&self, n: u32, pos: u32) -> Kept {
        if let Some(p) = &self.pending
            && n > p.pos0
        {
            let at = n.min(self.held);
            let why = if at == self.held {
                Why::Current
            } else {
                Why::Rule
            };
            return Kept {
                asked: n,
                held: self.held,
                at,
                why,
            };
        }
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

    /// The waiting cut carried out ([`Checkpoints::apply`]): the committed
    /// lane, the conv rings and the PLE ring from the point's slot, each
    /// delta layer's committed lane stamped at the position the cut left
    /// (`held`) and every other never written. Nothing when no cut waits.
    /// Waits for the copies.
    fn settle(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        if !self.ckpt.pending() {
            return Ok(());
        }
        let lane = self.lane.lane();
        let mut c = Copied::of(&mut self.stores, &mut self.ple_ring, lane, None)?;
        self.ckpt.apply(stream, &mut c.list())?;
        drop(c);
        self.restamp(stream, lane, self.held)
    }

    /// Every delta layer's stamps: lane `lane` at position `at`, every other
    /// never written. Synchronizes; never inside a capture.
    fn restamp(&mut self, stream: &CudaStream, lane: u32, at: u32) -> Result<(), GpuError> {
        let mut st = [crate::linear::delta::NEVER; LANES];
        *st.get_mut(lane as usize)
            .ok_or_else(|| GpuError::shape(WHAT, format!("lane {lane} of {LANES}")))? = at;
        for s in &mut self.stores {
            if let Store38::Rec { stamp, .. } = s {
                stamp.copy_from_host(stream, &st)?;
            }
        }
        Ok(())
    }

    /// A checkpoint at `pos`, where the stores stand, after any waiting cut:
    /// the recurrent stores copied to a host slot with the PLE history beside
    /// them, or nothing where one stands. Refused by name while a verify
    /// waits or the stores hold another position. Waits for the copies.
    fn take_mark(&mut self, stream: &CudaStream, pos: u32) -> Result<(), GpuError> {
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
        let lane = self.lane.lane();
        let mut c = Copied::of(&mut self.stores, &mut self.ple_ring, lane, None)?;
        let take = self.ckpt.take(stream, pos, &mut c.list());
        drop(c);
        match take {
            Ok(Take::Copy { evicted, .. }) => {
                if let Some(e) = evicted {
                    self.ple_points.retain(|&(p, _)| p != e);
                }
                let i = self.ple_points.partition_point(|&(p, _)| p < pos);
                self.ple_points.insert(i, (pos, self.ple.hist.clone()));
                Ok(())
            }
            Ok(Take::Held | Take::Empty) => Ok(()),
            Err(e) => {
                self.ple_points.clear();
                Err(e)
            }
        }
    }

    /// A cut to `pos`, behind the stores' position: the point at `pos`
    /// planned for the next call ([`Body38::settle`]), the PLE history set
    /// back to the point's, every later point dropped, the MTP draft's store
    /// forgotten (its rows past `pos` belong to the branch the cut dropped),
    /// and a verify waiting dropped. Refused by name, nothing moved, at 0 (a
    /// reset) and where no point stands.
    fn cut(&mut self, pos: u32) -> Result<(), GpuError> {
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
        self.ple.hist = hist;
        self.ple_points.retain(|&(p, _)| p <= pos);
        self.pending = None;
        self.staged = None;
        self.held = pos;
        self.wrote.cut(pos);
        if let Some(d) = self.mtp.as_mut() {
            d.forget();
        }
        Ok(())
    }

    /// The sequence state at `pos`, where the stores stand (a waiting cut
    /// carried out first): the K/V planes' rows, the raw keys' and the pooled
    /// keys' of positions below `pos`, the recurrent stores at `pos` and at
    /// the last checkpoint below it (the last prompt call's end), each with
    /// its PLE history. Refused by name while a verify waits, after a call
    /// failed past its launch, and when the history disagrees.
    fn save_state(&mut self, stream: &CudaStream, pos: u32) -> Result<Seq38, GpuError> {
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
        let points: Vec<(u32, History)> = self
            .ple_points
            .iter()
            .rev()
            .find(|&&(p, _)| p < pos)
            .cloned()
            .into_iter()
            .collect();
        let lane = self.lane.lane();
        // The draft's rows read the draft's store and the arenas, none of the
        // copies below, and `Copied`'s drop holds its borrow to the end.
        let draft = self.draft_rows(stream, pos)?;
        let mut c = Copied::of(
            &mut self.stores,
            &mut self.ple_ring,
            lane,
            Some((self.ctx, pos as usize)),
        )?;
        let (positional, mut recurrent) = c.list_split();
        let lent = Lent {
            positional,
            recurrent: &mut recurrent,
        };
        let state = SeqState::save(
            stream,
            self.who.clone(),
            pos,
            &lent,
            current,
            points,
            &self.ckpt,
        )?;
        Ok(Seq38 { state, draft })
    }

    /// The draft's side of a sequence state at `pos` ([`DraftRows38`]);
    /// `None` on a load without the draft: the store's rows and the step's
    /// and the pass's arena rows, each the positions below `pos` — the ones
    /// a walk can read. Refused by name when the store holds positions past
    /// `pos` or an arena's rows reach past it: a state's rows are the
    /// positions the model stands at, and a record past them is a verify's
    /// rejected tail or an anchor row the commit or the snapshot's caller
    /// left standing — neither is the state's to carry. Blocking.
    fn draft_rows(&self, stream: &CudaStream, pos: u32) -> Result<Option<DraftRows38>, GpuError> {
        const WHAT_D: &str = "qwen4exp snapshot";
        let Some(d) = self.mtp.as_ref() else {
            return Ok(None);
        };
        let held = d.held();
        if held > pos as usize {
            return Err(GpuError::shape(
                WHAT_D,
                format!("a state at {pos} of a draft store that holds {held} positions"),
            ));
        }
        let wide = geo::STREAMS * geo::HIDDEN;
        let rows = |walk: TargetRows, most: usize| -> Result<(Held38, Vec<f32>), GpuError> {
            let held = self.wrote.of(walk);
            let (first, n) = held.parts();
            let n = n as usize;
            if n > most || n > 0 && first as usize + n > pos as usize {
                return Err(GpuError::shape(
                    WHAT_D,
                    format!(
                        "the {walk:?} arena's {} at a state of {pos} positions",
                        held_shown(first, n)
                    ),
                ));
            }
            let mut v = self.final_streams(walk).to_host_vec(stream)?;
            v.truncate(n * wide);
            Ok((Held38::at(first, u32::try_from(n).unwrap_or(u32::MAX)), v))
        };
        let step = rows(TargetRows::Step, 1)?;
        let pass = rows(TargetRows::Pass, PASS_ROWS)?;
        Ok(Some(DraftRows38 {
            held: u32::try_from(held).unwrap_or(u32::MAX),
            store: d.store_host(stream, held)?,
            step,
            pass,
        }))
    }

    /// Refused by name unless `d` fits this load standing at `pos`, nothing
    /// copied: the draft loaded, its store's positions and each arena's at
    /// or below `pos`, each arena's rows within its buffer (the step's one,
    /// the pass's [`PASS_ROWS`]) and of the streams' width.
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
            let (first, n) = held.parts();
            let n = n as usize;
            if n > rows || v.len() != n * wide || n > 0 && first as usize + n > pos as usize {
                return Err(GpuError::shape(
                    WHAT_F,
                    format!(
                        "the {what} arena's {} ({} values) put back into {rows} rows of {wide} \
                         at position {pos}",
                        held_shown(first, n),
                        v.len()
                    ),
                ));
            }
        }
        Ok(())
    }

    /// `d` put back on a model that stands at `pos` ([`Body38::draft_rows`]'
    /// rows): the draft's store holding `d`'s positions, the step's and the
    /// pass's arenas their rows, the ubatch's none. Refused by name, as
    /// [`Body38::draft_fits`] refuses, before any copy. Blocking.
    fn put_draft_rows(
        &mut self,
        stream: &CudaStream,
        d: &DraftRows38,
        pos: u32,
    ) -> Result<(), GpuError> {
        const WHAT_P: &str = "qwen4exp resume";
        self.draft_fits(d, pos)?;
        let Body38 {
            plans,
            s,
            a,
            wa,
            mtp,
            wrote,
            ..
        } = self;
        for (walk, (_, v)) in [(TargetRows::Step, &d.step), (TargetRows::Pass, &d.pass)] {
            let len = final_streams(plans, [s, a, wa], walk).len();
            let mut host = vec![0.0f32; len];
            host[..v.len()].copy_from_slice(v);
            final_streams_mut(plans, [s, a, wa], walk).copy_from_host(stream, &host)?;
        }
        mtp.as_mut()
            .ok_or(GpuError::state(
                WHAT_P,
                "an MTP draft (Body38::open_placed_mtp)",
            ))?
            .adopt_store(stream, d.held as usize, &d.store)?;
        *wrote = Wrote38 {
            step: d.step.0,
            pass: d.pass.0,
            ubatch: Held38::NONE,
        };
        Ok(())
    }

    /// `s` put back on the empty model ([`SeqState::load`]): the stores
    /// stand at its positions, the committed lane stamped there, the PLE
    /// history and the points it carried the model's own. Refused by name,
    /// nothing copied, for another model's state, a draft side the load does
    /// not take (a state with it onto a load without the draft, or the other
    /// way: the store layouts differ) or onto a model that is not empty.
    fn load_state(&mut self, stream: &CudaStream, s: &Seq38) -> Result<(), GpuError> {
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
        let (cur, pts) = {
            let mut c = Copied::of(
                &mut self.stores,
                &mut self.ple_ring,
                lane,
                Some((self.ctx, n as usize)),
            )?;
            let (positional, mut recurrent) = c.list_split();
            let mut lent = Lent {
                positional,
                recurrent: &mut recurrent,
            };
            let (cur, pts) = s.state.load(stream, &self.who, &mut lent, &mut self.ckpt)?;
            (cur.clone(), pts.to_vec())
        };
        let off = std::iter::once((n, &cur))
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
            self.put_draft_rows(stream, d, n)?;
        }
        self.restamp(stream, lane, n)?;
        self.ple.hist = cur;
        self.ple_points = pts;
        self.held = n;
        Ok(())
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
    /// history past: [`Body38::launch`] takes it.
    fn stage(
        &mut self,
        pos: u32,
        rows: usize,
        hist: History,
        verify: Option<Vec<u32>>,
    ) -> Result<(), GpuError> {
        self.staged = Some(Staged38 {
            pos,
            rows: launch_u32(WHAT, "positions", rows)?,
            hist,
            verify,
        });
        Ok(())
    }

    /// The staged call of `rows` from `pos` launched: the stores' count past
    /// its rows, the PLE history past them, and a verify's commit waiting,
    /// with the history as it stood before. Refused by name, moving nothing,
    /// when no call of those rows from there is staged. A launch that fails
    /// after this is counted as run: the refusal that follows is named,
    /// where a count left behind would run the rows twice.
    fn launch(&mut self, pos: u32, rows: usize) -> Result<(), GpuError> {
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
        self.wrote.set(self.last_walk, s.pos, s.rows as usize);
        if let Some(tokens) = s.verify {
            self.pending = Some(Pending38 {
                pos0: s.pos,
                tokens,
                hist: before,
            });
        }
        Ok(())
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
        // SAFETY: the pass arena holds `PASS_ROWS · HIDDEN` rows of `e` and
        // `n <= PASS_ROWS` (the record's write refused more), and `e` stays in
        // place while the window lives (one synchronous copy).
        let mut e = unsafe { f32_view(&self.a.ple.e, 0, n * geo::HIDDEN) };
        e.copy_from_host(stream, &self.ple.e[..n * geo::HIDDEN])?;
        self.stage(pos, n, hist, None)
    }

    /// Launch the planned pass of `m` rows from `pos` through the batch
    /// port, then the head of its last row into `head` when given.
    fn walk_pass(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        m: usize,
        pos: u32,
        head: Option<&mut Head>,
    ) -> Result<(), GpuError> {
        if !(1..=self.a.rows).contains(&m) {
            return Err(GpuError::shape(
                WHAT,
                format!("a pass of {m} rows on an arena of {}", self.a.rows),
            ));
        }
        planted(&mut self.plant, Plant::BeforeLaunch)?;
        self.launch(pos, m)?;
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
            a,
            pass_hsum,
            k,
            slots,
            card,
            eps,
            ctx,
            plant,
            ..
        } = self;
        let mut prog = Pass38 {
            p: Parts38 {
                c: Ctx38 {
                    gpu,
                    w,
                    k,
                    eps: *eps,
                    table: &rope.table,
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
            None => Ok(()),
        }
    }
}

impl Body38 {
    /// Plan a ubatch of `tokens` (1..=the load's size) at `pos`, where the
    /// stores stand: the map check ([`MapCheck::refuse`]), the armed
    /// route taps' rows and the planted routes' positions, its PLE rows, its
    /// record and its rows' copy to the ubatch arena, staged for
    /// [`Body38::walk_gemm`]'s launch. Each refusal comes before anything
    /// moves. With the timing armed, the plan's host wall is the prompt's
    /// prologue.
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
        let hist = self.ple.fill(pos, tokens)?;
        self.last_walk = TargetRows::Ubatch;
        self.wr.write(stream, tokens, pos)?;
        // SAFETY: the ubatch arena holds `wide.rows · HIDDEN` values of `e`
        // and `n <= wide.rows` (the record's write refused more), and `e`
        // stays in place while the window lives (one synchronous copy).
        let mut e = unsafe { f32_view(&self.wa.ple.e, 0, n * geo::HIDDEN) };
        e.copy_from_host(stream, &self.ple.e[..n * geo::HIDDEN])?;
        self.stage(pos, n, hist, None)
    }

    /// Launch the planned ubatch of `m` rows from `pos` through the batch
    /// port at its width, then the head of its last row into `head` when
    /// given.
    fn walk_gemm(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        m: usize,
        pos: u32,
        head: Option<&mut Head>,
    ) -> Result<(), GpuError> {
        if !(1..=self.wide.rows).contains(&m) {
            return Err(GpuError::shape(
                WHAT,
                format!("a ubatch of {m} rows on an arena of {}", self.wide.rows),
            ));
        }
        let dense = dense_rows(pos as usize, m, self.ctx)?;
        planted(&mut self.plant, Plant::BeforeLaunch)?;
        self.launch(pos, m)?;
        let io = Io {
            lane: Some(self.lane.word()),
            ..self.wr.io()
        };
        let Body38 {
            hybrid,
            plans,
            stores,
            ple_ring,
            rope,
            wa,
            wide_hsum,
            wide,
            wide_timing,
            stream,
            k,
            slots,
            card,
            tier,
            eps,
            ctx,
            plant,
            ..
        } = self;
        let mut prog = Gemm38 {
            p: WideParts {
                c: Ctx38 {
                    gpu,
                    w,
                    k,
                    eps: *eps,
                    table: &rope.table,
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
                cur: 0,
                stream,
            },
        };
        let cap = prog.p.x.rows;
        let mut leg = BatchLeg::new(gpu.stream(), hybrid, wide_hsum, cap);
        leg.set_timer(wide_timing.as_mut());
        prog.walk(&mut leg)?;
        planted(plant, Plant::AfterLaunch)?;
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

    /// The last ubatch walk's rows of each selecting layer: those the
    /// prefill flash ran (selecting nothing) and those the selection did;
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
            Some(PromptTiming::new(gpu.context(), body.plans.len(), 1)?)
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
    /// overwrites them. A `None` sink is [`GpuModel::prompt38`] itself: the
    /// steps run as one call of the whole prompt; a sink's per-token steps
    /// leave the same bits, a head readback a token.
    ///
    /// The call is one residency pass ([`crate::host::swap`]): its boundary
    /// before it and none inside — the passes and the ubatches run through
    /// `run_rows`, which opens none — and 0 rows kept after it, whatever it
    /// returned: the rule counts decode rows only, and the batch service
    /// notes no id. A step-fed prompt is refused by name under a running
    /// machine: each prompt id would end a decode pass the rule counts.
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
        const WHAT_P: &str = "qwen4exp prompt";
        if tokens.is_empty() {
            return Err(GpuError::shape(WHAT_P, "empty token slice"));
        }
        self.refuse_if_poisoned(WHAT_P)?;
        let body = self.body(WHAT_P)?;
        super::refuse_past_vocab(WHAT_P, tokens, body.vocab)?;
        ple_takes(WHAT_P, body.ple.hash.window(), tokens)?;
        let (pos, ctx) = (self.pos() as usize, body.ctx());
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
        self.pass_boundary()?;
        let r = self
            .stream_begin(resolved, tokens.len())
            .and_then(|()| self.feed_marked(tokens, resolved, sink));
        // The call's own error first: an end or a keep refused after a
        // failed call is its echo. A failed call's streaming ends with each
        // layer back at the set it started with.
        let ended = self.stream_end(r.is_ok());
        let kept = self.keep_rows(KeptRows::prefix(0), PassKind::Prompt);
        let next = r?;
        ended?;
        kept?;
        Ok(next)
    }

    /// Open the prompt call's streaming when the lever is `admit` or
    /// `split`, the path is the ubatch walk's ([`super::wide38`]'s module
    /// doc) and the prompt's `n` ids reach the least floor a pick admits
    /// from ([`STREAM_FLOOR`] under `admit`, the stream rule's under
    /// `split`): the residency machine's call
    /// ([`crate::host::HostTier::call_begin`]) inside the pass the boundary
    /// just opened, and under `split` the expert stream's
    /// ([`crate::host::HostTier::xstream_begin`]). Nothing otherwise: a row
    /// routes an expert once at most, so a prompt of fewer ids than the
    /// floor — and every pass, of at most [`PASS_ROWS`] rows — gives no
    /// expert a count its pick would admit, and its walks would only wait
    /// on each layer's download for nothing. Refused by name: a call
    /// streaming already, and a mode with no machine.
    fn stream_begin(&mut self, path: Prompt38, n: usize) -> Result<(), GpuError> {
        const _: () = assert!(PASS_ROWS < STREAM_FLOOR as usize);
        const WHAT_B: &str = "qwen4exp prompt streaming";
        let (gpu, _, body) = self.body_parts(WHAT_B)?;
        if body.stream.on {
            return Err(GpuError::state(WHAT_B, "no call streaming (stream_end)"));
        }
        body.stream.picks.clear();
        body.stream.end = None;
        body.stream.layers.clear();
        body.stream.xend = None;
        body.stream.ubatch = 0;
        let mode = body.xstream_mode();
        let floor = match (mode, body.hybrid.xstream()) {
            (XMode::Split, Some(x)) => (0..body.plans.len())
                .map(|l| pick_floor(x.constants(l)))
                .chain([STREAM_FLOOR])
                .min()
                .unwrap_or(STREAM_FLOOR),
            _ => STREAM_FLOOR,
        };
        if mode == XMode::Off || path != Prompt38::Gemm || n < floor as usize {
            return Ok(());
        }
        let cfg = CallCfg { floor };
        if !body.hybrid.call_begin(gpu.stream(), cfg)? {
            return Err(GpuError::state(
                "BLOOMERY_XSTREAM",
                "a residency machine (BLOOMERY_RESIDENCY=mid-p<P>-s<S>)",
            ));
        }
        body.stream.on = true;
        body.stream.split = mode == XMode::Split && body.hybrid.xstream_begin();
        if mode == XMode::Split && !body.stream.split {
            body.stream.on = false;
            let _ = body.hybrid.call_end(gpu.stream(), false);
            return Err(GpuError::state(
                "BLOOMERY_XSTREAM=split",
                "an expert stream (Body38::set_xstream starts it)",
            ));
        }
        Ok(())
    }

    /// End the prompt call's streaming, if it streams: the expert stream's
    /// end under `split` ([`crate::host::HostTier::xstream_end`]), then the
    /// residency machine's ([`crate::host::HostTier::call_end`]) whatever the
    /// first returned, its placement `kept` for the decode after it, else
    /// each layer back at the set the call started with; the ends' reports
    /// kept for [`Body38::take_stream_records`] and
    /// [`Body38::take_xstream_records`]. The first error is returned.
    fn stream_end(&mut self, kept: bool) -> Result<(), GpuError> {
        let (gpu, _, body) = self.body_parts("qwen4exp prompt streaming")?;
        if !std::mem::take(&mut body.stream.on) {
            return Ok(());
        }
        let x = match std::mem::take(&mut body.stream.split) {
            true => body.hybrid.xstream_end(),
            false => Ok(None),
        };
        let ended = body.hybrid.call_end(gpu.stream(), kept);
        body.stream.xend = x?;
        body.stream.end = ended?;
        Ok(())
    }

    /// [`GpuModel::prompt38_with`]'s units ([`GpuModel::feed38`]); while the
    /// load takes checkpoints ([`Body38::set_checkpoints`]) the call runs as
    /// the runs between its marks (`Checkpoints::marks`: its start when it
    /// continues a sequence, every [`CHECKPOINT_EVERY`] positions inside it,
    /// its end), each on the path the whole call resolved to, a checkpoint
    /// taken at every mark: a ubatch walk leaves the same bits cut there or
    /// not, as a pass and a step do. The argmax after the last id.
    fn feed_marked(
        &mut self,
        tokens: &[u32],
        path: Prompt38,
        mut sink: Option<&mut Prompt38Sink<'_>>,
    ) -> Result<u32, GpuError> {
        const WHAT_M: &str = "qwen4exp prompt checkpoints";
        let body = self.body(WHAT_M)?;
        if !body.marks {
            return self.feed38(tokens, path, sink);
        }
        let from = self.pos();
        let to = launch_u32(WHAT_M, "positions", tokens.len())
            .ok()
            .and_then(|n| from.checked_add(n))
            .ok_or_else(|| GpuError::shape(WHAT_M, format!("{} ids from {from}", tokens.len())))?;
        let marks = body.ckpt.marks(from, to);
        let (mut next, mut at) = (None, from);
        for mark in marks {
            if mark > at {
                let run = &tokens[(at - from) as usize..(mark - from) as usize];
                next = Some(self.feed38(run, path, sink.as_deref_mut())?);
                at = mark;
            }
            let pos = self.pos();
            let (gpu, _, body) = self.body_parts(WHAT_M)?;
            body.take_mark(gpu.stream(), pos)?;
        }
        next.ok_or(GpuError::state(WHAT_M, "a mark at the call's end"))
    }

    /// [`GpuModel::prompt38_with`]'s paths once its checks have passed, its
    /// path resolved and its residency pass open: the resolved `path`'s
    /// units.
    fn feed38(
        &mut self,
        tokens: &[u32],
        path: Prompt38,
        mut sink: Option<&mut Prompt38Sink<'_>>,
    ) -> Result<u32, GpuError> {
        const WHAT_P: &str = "qwen4exp prompt";
        let tapped = sink.is_some();
        match path {
            Prompt38::Step => {
                if !tapped {
                    // The plain call: every token's body, one readback at the
                    // end.
                    return self.step(tokens);
                }
                // A tap reads each token's row where the step left it.
                let mut next = None;
                for &token in tokens {
                    next = Some(self.step(&[token])?);
                    let (first, rows) = (self.pos() - 1, 1usize);
                    if let Some(f) = sink.as_deref_mut() {
                        f(self, TargetRows::Step, first, rows)?;
                    }
                }
                next.ok_or(GpuError::state(WHAT_P, "a token read after the last step"))
            }
            Prompt38::Gemm => {
                let rows = self.body(WHAT_P)?.wide.rows;
                let (mut next, mut at) = (None, 0);
                for m in ubatch_cut(tokens.len(), rows) {
                    let chunk = &tokens[at..at + m];
                    at += m;
                    let last = at == tokens.len();
                    let pos = self.pos();
                    {
                        let (gpu, _, body) = self.body_parts(WHAT_P)?;
                        body.plan_gemm(gpu.stream(), chunk, pos)?;
                    }
                    next = self.run_rows(m, WHAT_P, |gpu, w, body, head, pos| {
                        body.walk_gemm(gpu, w, m, pos, last.then_some(head))?;
                        body.stream.ubatch += 1;
                        Ok(last)
                    })?;
                    if let Some(f) = sink.as_deref_mut() {
                        f(self, TargetRows::Ubatch, pos, m)?;
                    }
                }
                next.ok_or(GpuError::state(
                    WHAT_P,
                    "a token read after the last ubatch",
                ))
            }
            Prompt38::Pass => {
                let n = tokens.len().div_ceil(PASS_ROWS);
                let mut next = None;
                for (i, chunk) in tokens.chunks(PASS_ROWS).enumerate() {
                    let last = i + 1 == n;
                    let pos = self.pos();
                    {
                        let (gpu, _, body) = self.body_parts(WHAT_P)?;
                        body.plan_pass(gpu.stream(), chunk, pos)?;
                    }
                    let m = chunk.len();
                    next = self.run_rows(m, WHAT_P, |gpu, w, body, head, pos| {
                        body.walk_pass(gpu, w, m, pos, last.then_some(head))?;
                        Ok(last)
                    })?;
                    if let Some(f) = sink.as_deref_mut() {
                        f(self, TargetRows::Pass, pos, m)?;
                    }
                }
                next.ok_or(GpuError::state(WHAT_P, "a token read after the last pass"))
            }
            Prompt38::Auto => Err(GpuError::state(
                WHAT_P,
                "a path `Prompt38::resolve` settles (it returned `auto`)",
            )),
        }
    }
}

/// The positions `first .. first + n`, for a refusal's words of a state's
/// arena rows.
fn held_shown(first: u32, n: usize) -> String {
    match n {
        0 => "no position".to_string(),
        n => format!("positions {}..{}", first, first as usize + n),
    }
}

/// Each target arena's [`Held38`]: the body's record of where the step, the
/// pass (a verify's rows too) and the ubatch walk left their rows — what a
/// draft's walk checks its read positions against, and a sequence state
/// carries of the two arenas a parked refresh reads.
#[derive(Clone, Copy, Debug, Default)]
struct Wrote38 {
    step: Held38,
    pass: Held38,
    ubatch: Held38,
}

impl Wrote38 {
    /// The arena `walk`'s rows hold `rows` rows at `pos ..`, as the call that
    /// last wrote it left them.
    fn set(&mut self, walk: TargetRows, pos: u32, rows: usize) {
        let held = Held38::at(pos, u32::try_from(rows).unwrap_or(u32::MAX));
        match walk {
            TargetRows::Step => self.step = held,
            TargetRows::Pass => self.pass = held,
            TargetRows::Ubatch => self.ubatch = held,
        }
    }

    /// Every arena's positions from `pos` on taken back.
    fn cut(&mut self, pos: u32) {
        for h in [&mut self.step, &mut self.pass, &mut self.ubatch] {
            h.cut(pos);
        }
    }

    fn of(&self, walk: TargetRows) -> Held38 {
        match walk {
            TargetRows::Step => self.step,
            TargetRows::Pass => self.pass,
            TargetRows::Ubatch => self.ubatch,
        }
    }
}

/// The streams' buffer of `walk`'s arena (the step's, the pass's, the
/// ubatch walk's) that holds the target's hidden rows once the walk ran its
/// head: a walk's current buffer flips once at each PLE layer past layer 0.
fn final_streams<'b>(
    plans: &[Layer38],
    [s, a, wa]: [&'b Arena38; 3],
    walk: TargetRows,
) -> &'b DeviceBuffer<f32> {
    let cur = plans.iter().skip(1).filter(|p| p.ple.is_some()).count() % 2;
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
    let cur = plans.iter().skip(1).filter(|p| p.ple.is_some()).count() % 2;
    let arena = match walk {
        TargetRows::Step => s,
        TargetRows::Pass => a,
        TargetRows::Ubatch => wa,
    };
    &mut arena.res[cur]
}

/// The target's streams a [`MtpHidden::Target`] feed of a walk or a chain's
/// refresh names, over the body's arenas, with the positions the arena's
/// rows hold ([`Wrote38`]): the walk refuses rows that hold other
/// positions, or none.
fn mtp_target<'b>(
    plans: &[Layer38],
    arenas: [&'b Arena38; 3],
    wrote: &Wrote38,
    feed: MtpFeed<'_>,
) -> Option<Target38<'b>> {
    match feed {
        MtpFeed::Rows {
            hidden: MtpHidden::Target { walk, .. },
            ..
        } => Some(Target38 {
            buf: final_streams(plans, arenas, walk),
            held: wrote.of(walk),
        }),
        _ => None,
    }
}

/// The draft's context over the body's own pieces.
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

/// The MTP draft's calls ([`Mtp38::run`]); each is refused by name on a load
/// without a draft.
impl GpuModel<Body38> {
    /// One walk of the draft (module doc of `mtp38`): `feed`'s rows into
    /// `head`, eager or captured, and each row's token and probability. A
    /// poisoned model is refused; a fault the walk raises is its error and
    /// stays on the card's word, which the target's next readback names.
    pub fn mtp_draft(
        &mut self,
        feed: MtpFeed<'_>,
        head: MtpHead,
        mode: MtpMode,
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
        feed: MtpFeed<'_>,
        head: MtpHead,
        mode: MtpMode,
    ) -> Result<(), GpuError> {
        const WHAT_D: &str = "qwen4exp mtp_walk";
        self.refuse_if_poisoned(WHAT_D)?;
        let (gpu, w, body) = self.body_parts(WHAT_D)?;
        let r = body.mtp_run_walk(gpu, w, feed, head, mode).map(|_| ());
        self.note_fault(WHAT_D, r)
    }

    /// Arm (or disarm) the draft's taps ([`Mtp38::set_taps`]); its captured
    /// walks are dropped first. Load-time allocation.
    pub fn set_mtp_taps(&mut self, on: bool) -> Result<(), GpuError> {
        let (gpu, _, body) = self.body_parts("qwen4exp set_mtp_taps")?;
        body.draft_mut()?.set_taps(gpu.stream(), on)
    }

    /// One window's chain of the draft (module doc of `mtp38`): `refresh`'s
    /// rows — the target's kept rows with the target's hidden rows, a
    /// [`MtpFeed::Rows`] feed — walked, then `own` own walks, each reading
    /// the walk before it on the card, into `head`, eager or captured, and
    /// the proposal's ids (1 + `own` of them) and their probabilities read
    /// back once. A poisoned model is refused; a fault any walk of the chain
    /// raised is its error and poisons the model as the target's would.
    pub fn mtp_chain(
        &mut self,
        refresh: MtpFeed<'_>,
        own: usize,
        head: MtpHead,
        mode: MtpMode,
    ) -> Result<MtpDraft, GpuError> {
        const WHAT_D: &str = "qwen4exp mtp_chain";
        self.refuse_if_poisoned(WHAT_D)?;
        let (gpu, w, body) = self.body_parts(WHAT_D)?;
        let r = body.mtp_run_chain(gpu, w, refresh, own, head, mode);
        // A fault a walk of the chain raised poisons the model as the
        // target's would.
        self.note_fault(WHAT_D, r)
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
    /// positional rows below the model's position, the recurrent stores
    /// there and at the last checkpoint below it, each with its PLE history.
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

    /// Replace the sequence with `s`, which [`GpuModel::seq_save`] took of
    /// this load: a reset unless the model is empty already (a reset just
    /// before, or the load), then — as a pass of `s`'s positions whose work
    /// is the copies — the state put back; the model stands at its
    /// positions, its checkpoints the points `s` carried and no other, and
    /// the calls after it are bit for bit those after the state was taken.
    /// Refused by name, nothing copied, for another model's state (its
    /// file, card or context).
    pub fn seq_resume(&mut self, s: &Seq38) -> Result<(), GpuError> {
        const WHAT_R: &str = "qwen4exp resume";
        let b = self.body(WHAT_R)?;
        let empty = self.pos() == 0
            && b.held == 0
            && b.pending.is_none()
            && b.staged.is_none()
            && b.ckpt.positions().is_empty()
            && !b.ckpt.pending();
        if !empty {
            self.reset()?;
        }
        let n = s.positions() as usize;
        if n == 0 {
            return Ok(());
        }
        self.run_rows(n, WHAT_R, |gpu, _, body, _, _| {
            body.load_state(gpu.stream(), s)?;
            Ok(false)
        })?;
        Ok(())
    }
}

impl ChainBody for Body38 {
    type Input = DecodeInput38;
    type Host = Body38;

    fn arch() -> Arch {
        Arch::Qwen35moe
    }

    /// The step at `pos`, where the stores stand: the map check
    /// ([`MapCheck::refuse`], at every step: a planted map may change between
    /// two replays of one capture), its PLE rows read from the file, staged
    /// for the refresh's launch.
    fn decode_input(&mut self, token: u32, pos: u32) -> Result<DecodeInput38, GpuError> {
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
        self.launch(input.pos, 1)
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
            s,
            k,
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
            p: Parts38 {
                c: Ctx38 {
                    gpu,
                    w,
                    k,
                    eps: *eps,
                    table: &rope.table,
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
    /// host tier's reset. The K/V planes and the
    /// raw and pooled keys need nothing: nothing reads a row at or past a
    /// live count, and every row below it is written by its own step first.
    /// Synchronizes: the engine stream waited for within
    /// [`crate::host::ENGINE_BOUND`] before the first blocking copy and at
    /// the end ([`crate::host::await_done`]).
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        const WHAT: &str = "Body38::reset";
        let stream = gpu.stream();
        self.hybrid.settle(stream)?;
        crate::host::await_done(WHAT, || stream.query()).map_err(|e| self.hybrid.noted(e))?;
        let most = self
            .stores
            .iter()
            .map(|s| match s {
                Store38::Rec { rec, .. } => rec.state.len().max(rec.ring.len()),
                Store38::Qsa { .. } => 0,
            })
            .max()
            .unwrap_or(0);
        let zeros = vec![0.0f32; most];
        for s in &mut self.stores {
            if let Store38::Rec { rec, .. } = s {
                rec.clear(stream, &zeros)?;
            }
            s.restamp(stream)?;
        }
        self.lane.set(stream, 0)?;
        self.pending = None;
        self.k
            .ple
            .reset_ring(stream, &mut self.ple_ring, geo::STREAMS)?;
        self.ple.restart()?;
        self.ckpt.clear();
        self.ple_points.clear();
        if let Some(d) = self.mtp.as_mut() {
            d.forget();
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
            + self.rope.table.num_bytes()
            + self.s.bytes()
            + self.sp.bytes()
            + self.a.bytes()
            + self.rp.bytes()
            + self.pass_hsum.num_bytes()
            + self.wa.bytes()
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
        let d = dims(self.s.route.dims(), self.ctx);
        let stores = self
            .kinds()
            .into_iter()
            .map(|k| match k {
                LayerKind38::Gdn => Store38::rec(stream),
                LayerKind38::Qsa => Store38::qsa(stream, &d),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let lens = stores
            .iter()
            .filter(|s| matches!(s, Store38::Rec { .. }))
            .flat_map(|_| [GDN.state_len(), GDN.ring_len()])
            .chain([ple::ring_len(geo::STREAMS)])
            .collect();
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
            Some(d) => Some(d.new_seq(stream)?),
        };
        let seq = Slot38 {
            stores,
            ple_ring: DeviceBuffer::zeroed(stream, ple::ring_len(geo::STREAMS))?,
            lane: LaneWord::new(stream)?,
            pending: None,
            ckpt: Checkpoints::new(gpu.context(), lens, HOST_BUDGET, CHECKPOINT_EVERY)?,
            ple_points: Vec::new(),
            ple_hist,
            held: 0,
            wrote: Wrote38::default(),
            last_walk: TargetRows::Step,
            rows_step,
            rows_pass,
            draft,
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

    /// Device bytes one sequence holds: its stores and PLE ring, its lane
    /// word, its two arena-row buffers and the draft's store on a load with
    /// one — the same terms [`seq38_bytes`] counts a host park's read of
    /// them by.
    fn seq_bytes(&self) -> usize {
        let arenas = [&self.s, &self.a, &self.wa];
        self.stores.iter().map(Store38::bytes).sum::<usize>()
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
        std::mem::swap(&mut self.ple_ring, &mut seq.ple_ring);
        std::mem::swap(&mut self.lane, &mut seq.lane);
        std::mem::swap(&mut self.pending, &mut seq.pending);
        std::mem::swap(&mut self.ckpt, &mut seq.ckpt);
        std::mem::swap(&mut self.ple_points, &mut seq.ple_points);
        std::mem::swap(&mut self.ple.hist, &mut seq.ple_hist);
        std::mem::swap(&mut self.held, &mut seq.held);
        std::mem::swap(&mut self.wrote, &mut seq.wrote);
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

impl HostServed for Body38 {
    fn serve_captured(&mut self, chain: Chain) -> Result<(), GpuError> {
        self.hybrid.serve_captured_of(chain)?;
        planted(&mut self.plant, Plant::AfterLaunch)
    }

    fn noted(&self, e: GpuError) -> GpuError {
        self.hybrid.noted(e)
    }

    fn take_host_refusal(&mut self) -> Option<Refusal> {
        self.hybrid.take_step_refusal()
    }

    /// The host tier's refusal poison lifted
    /// ([`Hybrid::lift_refusal`]) — the settling a reset runs is
    /// [`Hybrid::settle`]'s ([`Body38::reset`]).
    fn lift_refusal(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.hybrid.lift_refusal(stream)
    }

    /// The refusal the host tier is poisoned by now
    /// ([`Hybrid::refusal_poison`]).
    fn refusal_poison(&self) -> Option<Refusal> {
        self.hybrid.refusal_poison()
    }

    fn host_residency(&self) -> Option<&HostResidency> {
        self.hybrid.residency()
    }

    /// The host tier's residency boundary at `at`
    /// ([`ResidencyGlue::at_boundary`]), its report logged when a binary
    /// asked ([`Body38::log_residency`]).
    fn at_boundary(&mut self, stream: &CudaStream, at: BoundaryAt) -> Result<(), GpuError> {
        self.residency_glue
            .at_boundary(&mut self.hybrid, stream, at)
    }

    fn keep_rows(&mut self, kept: KeptRows, kind: PassKind) -> Result<(), GpuError> {
        self.residency_glue.keep_rows(&mut self.hybrid, kept, kind)
    }

    fn residency_reset(&mut self, stream: &CudaStream) -> Result<Option<ResetReport>, GpuError> {
        self.residency_glue.reset(&mut self.hybrid, stream)
    }

    fn stop_residency(&mut self) {
        self.residency_glue.stop(&mut self.hybrid);
    }

    fn start_residency(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        self.residency_glue.start(
            &mut self.hybrid,
            gpu.context(),
            gpu.stream(),
            Arc::clone(&self.slots),
        )
    }
}

impl Body38 {
    /// Plan a verify of `tokens` (2..=[`VERIFY_ROWS`]) at `pos`, where the
    /// stores stand: the ids checked as a prompt's are, the PLE rows, the
    /// record and the rows' copy to the pass arena, then the verify launched
    /// ([`Body38::launch`]) and left waiting for its commit — its graph's
    /// launch follows this plan with no body call between. Refused by name
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
        self.card_leg.refuse(Walk38::Verify)?;
        let n = tokens.len();
        if !(2..=VERIFY_ROWS).contains(&n) {
            return Err(GpuError::shape(
                WHAT_V,
                format!("a verify of {n} rows; the lanes hold 2..={VERIFY_ROWS}"),
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
        ple_takes(WHAT_V, self.ple.hash.window(), tokens)?;
        self.settle(stream)?;
        let hist = self.ple.fill(pos, tokens)?;
        self.last_walk = TargetRows::Pass;
        self.rp.write(stream, tokens, pos)?;
        // SAFETY: the pass arena holds `PASS_ROWS · HIDDEN` rows of `e` and
        // `n <= VERIFY_ROWS <= PASS_ROWS`, and `e` stays in place while the
        // window lives (one synchronous copy).
        let mut e = unsafe { f32_view(&self.a.ple.e, 0, n * geo::HIDDEN) };
        e.copy_from_host(stream, &self.ple.e[..n * geo::HIDDEN])?;
        self.stage(pos, n, hist, Some(tokens.to_vec()))?;
        planted(&mut self.plant, Plant::BeforeLaunch)?;
        self.launch(pos, n)
    }

    /// Enqueue the planned verify of `head.m()` rows through the step port
    /// into `head`, one head of the verify's rows. An eager verify needs its
    /// plan; a capture records the launches only.
    fn walk_verify(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        const WHAT_V: &str = "qwen4exp verify";
        let m = head.m();
        let planned = self.pending.as_ref().map(|p| p.tokens.len());
        if planned != Some(m) && !crate::capturing(gpu.stream())? {
            return Err(GpuError::state(
                WHAT_V,
                "a verify planned for this head's rows (plan_rows before the pass)",
            ));
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
            rope,
            a,
            k,
            slots,
            card,
            tier,
            eps,
            ctx,
            plant,
            ..
        } = self;
        let prog = Verify38 {
            p: Parts38 {
                c: Ctx38 {
                    gpu,
                    w,
                    k,
                    eps: *eps,
                    table: &rope.table,
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
        };
        let mut leg = StepLeg::new(gpu.stream(), hybrid);
        prog.walk(&mut leg)?;
        planted_eager(plant, gpu)
    }

    /// Keep the first `pos − pos0` rows of the verify waiting for its commit
    /// and take the rest back: the lane word to the last kept row's lane
    /// ([`kept_lane`]), the PLE history over the kept rows, and the MTP
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
            return self.cut(pos);
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
        self.lane.set(stream, kept_lane(self.lane.lane(), kept))?;
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

/// The stores a checkpoint copies, lent for one call ([`Copied::of`]), and
/// a sequence state's positional rows beside them.
struct Copied<'a> {
    /// Each selecting layer's rows of the positions the call carries.
    spans: Vec<Spans<'a>>,
    /// Each delta layer's committed state lane and conv ring, in layer
    /// order.
    lanes: Vec<WindowMut<'a, f32>>,
    rings: Vec<&'a mut DeviceBuffer<f32>>,
    ple: &'a mut DeviceBuffer<f32>,
}

impl<'a> Copied<'a> {
    /// The stores of `stores` and `ple_ring` a checkpoint copies — each
    /// delta layer's state lane `lane` and its conv ring, then the PLE ring
    /// — and, with `rows` (`(ctx, n)`), each selecting layer's rows of
    /// positions below `n`: its K and V planes' (`[N_KV][ctx][HEAD]` each),
    /// its raw keys' (`[ctx][IDX_DIM]`) and its pooled keys' of the pools
    /// those positions reach.
    fn of(
        stores: &'a mut [Store38],
        ple: &'a mut DeviceBuffer<f32>,
        lane: u32,
        rows: Option<(usize, usize)>,
    ) -> Result<Copied<'a>, GpuError> {
        let len = GDN.state_len();
        let mut c = Copied {
            spans: Vec::new(),
            lanes: Vec::new(),
            rings: Vec::new(),
            ple,
        };
        for s in stores {
            match s {
                Store38::Rec { rec, .. } => {
                    let RecStore { state, ring, .. } = rec;
                    let byte = lane as usize * len * size_of::<f32>();
                    c.lanes.push(WindowMut::of_mut(state, byte, len)?);
                    c.rings.push(ring);
                }
                Store38::Qsa { kv, raw, pooled } => {
                    if let Some((ctx, n)) = rows {
                        let (h, d) = (geo::HEAD, geo::IDX_DIM);
                        let (k, v) = kv.f16_mut("qwen38::copied")?;
                        c.spans.push(Spans::planes(k, geo::N_KV, ctx * h, n * h)?);
                        c.spans.push(Spans::planes(v, geo::N_KV, ctx * h, n * h)?);
                        c.spans.push(Spans::planes(raw, 1, 0, n * d)?);
                        let pools = crate::qsa::pools_for(n) * d;
                        c.spans.push(Spans::planes(pooled, 1, 0, pools)?);
                    }
                }
            }
        }
        Ok(c)
    }

    /// The stores in the checkpoints' order: each delta layer's lane, then
    /// its ring, then the PLE ring.
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
            ple,
        } = self;
        let mut out: Vec<&mut DeviceBuffer<f32>> = Vec::new();
        for (l, r) in lanes.iter_mut().zip(rings.iter_mut()) {
            out.push(&mut **l);
            out.push(&mut **r);
        }
        out.push(&mut **ple);
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

    fn rollback_on(&mut self, gpu: &Gpu, pos: u32) -> Result<(), GpuError> {
        self.commit(gpu, pos)
    }
}

/// A pass of several resident slots' rows (`GpuModel::step_slots`): every
/// busy slot's rows walked as one verify of all of them (`Verify38` over
/// [`Seqs38::Slots`]) — the row-wise launches once over every row, each
/// slot's sequence-bound launches over its own rows and stores, each the
/// launch that slot's rows would run alone — into one head of every row.
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
    /// a load with an expert tier card, the layer taps armed, a route trace
    /// attached, more rows than the step port's columns, a slot map with a
    /// routed expert on a tier card ([`MapCheck::refuse`]), and for each
    /// slot on its own sequence a row count outside `1..=LANES`, a position
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
        let total = slot_rows(rows, parked.len())?;
        if self.slot_in.is_none() {
            return Err(GpuError::shape(
                WHAT_SLOTS,
                "a pass of several slots on a load planned for one sequence",
            ));
        }
        if self.tier.is_some() {
            return Err(GpuError::shape(
                WHAT_SLOTS,
                "a pass of several slots on a load with an expert tier card",
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
            b.check_slot(r, slot_ids(ids, r)?)
        })?;
        planted(&mut self.plant, Plant::BeforeLaunch)?;
        for_slots(self, parked, rows, |b, r| {
            b.launch_slot(stream, r, slot_ids(ids, r)?)
        })?;
        self.slot_in
            .as_mut()
            .ok_or(GpuError::state(WHAT_SLOTS, "the slots' input records"))?
            .write(stream, rows, ids)
    }

    /// Refused by name unless the live sequence takes a verify of slot
    /// `r.slot`'s `tokens` from `r.pos0` as one slot of a pass: `1..=LANES`
    /// rows (each keeps its state in a lane of its own), from where its
    /// stores stand, no verify waiting, ids the embedding and the PLE hash
    /// take.
    fn check_slot(&self, r: &SlotRange, tokens: &[u32]) -> Result<(), GpuError> {
        let named = |e: GpuError| GpuError::shape(WHAT_SLOTS, format!("slot {}: {e}", r.slot));
        let m = tokens.len();
        if !(1..=LANES).contains(&m) {
            return Err(GpuError::shape(
                WHAT_SLOTS,
                format!(
                    "slot {}'s {m} rows; a slot's rows each keep their state in a lane of their \
                     own: 1..={LANES}",
                    r.slot
                ),
            ));
        }
        self.check_next(r.pos0, m).map_err(named)?;
        super::refuse_past_vocab(WHAT_SLOTS, tokens, self.vocab).map_err(named)?;
        ple_takes(WHAT_SLOTS, self.ple.hash.window(), tokens).map_err(named)
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
        self.launch(r.pos0, m)
    }

    /// Enqueue the pass of `rows` [`Body38::plan_slots`] planned into
    /// `head`, one head of every row, `parked` as there. An eager pass
    /// needs each slot's plan; a capture records the launches only.
    pub(super) fn enqueue_slots(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        head: &mut Head,
        parked: &mut [&mut Slot38],
        rows: &[SlotRange],
    ) -> Result<(), GpuError> {
        let total = slot_rows(rows, parked.len())?;
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
        let Body38 {
            hybrid,
            plans,
            stores,
            ple_ring,
            lane,
            rope,
            a,
            k,
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
        let mut live = Some((stores.as_mut_slice(), ple_ring, &*lane));
        let mut others = parked.iter_mut();
        let mut seqs = Vec::with_capacity(rows.len());
        let mut copies = Vec::new();
        for (r, rec) in rows.iter().zip(&recs) {
            let (stores, ple_ring, lane) = if r.slot == 0 {
                live.take().ok_or_else(|| slot_twice(r))?
            } else {
                let Slot38 {
                    stores,
                    ple_ring,
                    lane,
                    rows_pass,
                    ..
                } = &mut **others.next().ok_or_else(|| no_parked(r))?;
                copies.push(RowsCopy {
                    rows: r.rows.clone(),
                    to: rows_pass,
                });
                (stores.as_mut_slice(), ple_ring, &*lane)
            };
            seqs.push(SlotSeq38 {
                stores,
                ple_ring,
                io: Io {
                    lane: Some(lane.word()),
                    ..rec.io()
                },
                rows: r.rows.clone(),
            });
        }
        let prog = Verify38 {
            p: Parts38 {
                c: Ctx38 {
                    gpu,
                    w,
                    k,
                    eps: *eps,
                    table: &rope.table,
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
        };
        let mut leg = StepLeg::new(gpu.stream(), hybrid);
        prog.walk(&mut leg)?;
        planted_eager(plant, gpu)
    }

    /// The pass's slot keeps its rows below position `at`: the commit of
    /// its waiting verify ([`Body38::commit`]: the lane word to the last
    /// kept row's lane, the PLE history replayed over the kept rows, the
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
    /// counts of `key` (each `(slot, rows)`, in pass order): the verify's
    /// at every row of the pass, each slot past the first's
    /// sequence-bound launches ([`super::program38`]'s `seq_launches`), and
    /// the copy of each parked slot's rows.
    #[must_use]
    pub fn slots_launches(&self, key: &[(usize, usize)]) -> usize {
        let rows = key.iter().map(|&(_, m)| m).sum();
        self.verify_launches(rows)
            + key.len().saturating_sub(1) * seq_launches(&self.plans)
            + key.iter().filter(|&&(slot, _)| slot != 0).count()
    }
}

impl SlotRows for Body38 {
    /// The pass arena's rows, which a load planned for several sequences
    /// widens the step port to ([`Body38::plan_slots`] refuses past the
    /// port's columns).
    const MAX_ROWS: usize = PASS_ROWS;
    const HEADS: RowHeads = RowHeads::One;
    /// The commit settles a partial keep ([`Body38::keep_slot`]: the lane
    /// word, the PLE history and the held count cut to the kept rows).
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

/// The pass's rows `R`, once `rows` tile `0..R` in order from row 0 with
/// no empty range, slot 0 — when busy — first, and `parked` counts the
/// ranges past slot 0; else refused by name.
fn slot_rows(rows: &[SlotRange], parked: usize) -> Result<usize, GpuError> {
    let mut at = 0;
    for (i, r) in rows.iter().enumerate() {
        if r.rows.start != at || r.rows.is_empty() {
            return Err(GpuError::shape(
                WHAT_SLOTS,
                format!(
                    "slot {}'s rows {:?}; the ranges tile the pass in order from row 0",
                    r.slot, r.rows
                ),
            ));
        }
        if r.slot == 0 && i != 0 {
            return Err(GpuError::shape(
                WHAT_SLOTS,
                "slot 0's rows past another slot's: slot 0's own rows hold its rows from row 0, \
                 so a pass that runs slot 0 lays its rows first",
            ));
        }
        at = r.rows.end;
    }
    let others = rows.iter().filter(|r| r.slot != 0).count();
    if at == 0 || parked != others {
        return Err(GpuError::shape(
            WHAT_SLOTS,
            format!(
                "a pass of {at} rows with {parked} parked sequences for {others} slots past slot 0"
            ),
        ));
    }
    Ok(at)
}

/// Slot `r.slot`'s ids of the pass's `ids`.
fn slot_ids<'i>(ids: &'i [u32], r: &SlotRange) -> Result<&'i [u32], GpuError> {
    ids.get(r.rows.clone()).ok_or_else(|| {
        GpuError::shape(
            WHAT_SLOTS,
            format!("slot {}'s rows {:?} of {} ids", r.slot, r.rows, ids.len()),
        )
    })
}

/// Slot 0 named twice in one pass, refused.
fn slot_twice(r: &SlotRange) -> GpuError {
    GpuError::shape(WHAT_SLOTS, format!("slot {} twice in one pass", r.slot))
}

/// A busy slot past slot 0 with no parked sequence handed for it, refused.
fn no_parked(r: &SlotRange) -> GpuError {
    GpuError::shape(WHAT_SLOTS, format!("slot {}'s parked sequence", r.slot))
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
            _ => Some(&mut **others.next().ok_or_else(|| no_parked(r))?),
        };
        on_seq(b, seq, |b| f(b, r))?;
    }
    Ok(())
}

const _: () = assert!(GDN.n_v == geo::V_HEADS);
const _: () = assert!(VERIFY_ROWS <= PASS_ROWS);

#[cfg(test)]
mod tests {
    use super::ple_takes;
    use engram::hash::{EosWindow, Window};

    /// The call's check is the hash's: a text id passes; the image
    /// placeholder and an id past the vocabulary are refused by name with
    /// their index wherever they stand; a hash that is not raw-id is
    /// refused.
    #[test]
    fn a_call_is_checked_by_the_hashs_own_test() {
        let w = Window::Eos(EosWindow {
            eos: 1,
            image: Some(9),
            n_vocab: 16,
        });
        assert!(ple_takes("t", &w, &[0, 1, 15]).is_ok());
        for (ids, at) in [([2, 3, 9], "token 2:"), ([15, 16, 2], "token 1:")] {
            let e = ple_takes("t", &w, &ids).expect_err("the id is refused");
            assert!(e.to_string().contains(at), "{ids:?}: {e}");
        }
        let mapped = Window::Mapped {
            token_map: vec![0; 16],
            pad: 0,
        };
        let e = ple_takes("t", &mapped, &[0]).expect_err("a mapped hash is refused");
        assert!(e.to_string().contains("a raw-id PLE hash"), "{e}");
    }
}
