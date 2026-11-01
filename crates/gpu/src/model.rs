//! `GpuModel<B>` — the GPU engine; an architecture's CPU `forward::step` is its reference
//! (docs/gpu-design.md decisions 1 and 7). One card holds the model: its
//! weights, the body's resident buffers and the output head live on the device
//! from load, and `step` enqueues one decode step and synchronizes once for
//! the argmax.
//!
//! This file is the part that is the same for every architecture
//! (docs/arch-split.md): the drop order of the captured chains, the graph
//! cache keyed by the rows a chain runs, the rule that a mode change discards
//! a capture, the position and its advance and the single argmax readback.
//! Which weights a model derives at load, what a layer chain enqueues, which
//! rows it appends to which cache and what one step's host input is belong to
//! a [`ChainBody`] under [`crate::arch`], and each model's constructor lives
//! beside its body; `GpuModel` is generic over it and monomorphic at every
//! call site, so nothing on the token path is a virtual call.
//! [`GpuModel::step`] runs the body's whole chain — layer 0 embedding its
//! token, every later layer reading the previous layer's output residual, the
//! `head.rs` output head on the last — and returns the argmax of the last
//! token it was given. The chain submits in one of two modes ([`StepMode`]):
//! eager, which enqueues the body per token, or graph, which captures the
//! body once and replays it. Only the argmax readback synchronizes.
//!
//! What only some bodies can do is a capability trait, and the method that
//! drives it exists only where the body implements it: [`HostServed`] (a host
//! tier inside the step), [`Rows`] (a pass of several rows), [`Rollback`],
//! [`Instrumented`] (a synthetic depth) and [`Probed`] (the node-price probe).

mod errors;
pub(crate) mod fit;
pub(crate) mod kernels;
pub(crate) mod lookup;
pub(crate) mod probe;
pub(crate) mod slots;

pub use kernels::{Q8_0GemvHeadsArgs, StepKernels};
pub(crate) use lookup::f32_gain;
pub use probe::{OpTime, StepProbe, weight_bytes};
pub use slots::{SlotRange, SlotRowOrder, SlotRows, Slots, SlotsOut, slots_a_pass};

use slots::{ParkedSlot, PoisonWhy, SlotFault, SlotGraphs, SlotSet};

use crate::best::{self, Best};
use crate::fault::Fault;
use crate::graph::MappedHost;
use crate::head::{Head, HeadNorm};
use crate::host::leg_tier;
use crate::host::nvtier::{ArenaKind, ArenaPages, NvTier, PageSource};
use crate::host::refuse_expert_tiers;
use crate::host::run::{HostRun, HostWidths};
use crate::host::swap::{BoundaryAt, MachineCfg, Residency};
use crate::host::swap_source::{FileSwap, ResidencyGlue, ResidencySpec};
use crate::hybrid::{Boundary, BoundaryShape, Chain, HostResidency, Hybrid, Refusal, SlotMap};
use crate::watchdog;
use crate::weights::{CardOf, HEAD_TENSOR, Weights, head_tensor};
use crate::{Gpu, GpuError, Graph, NodeInfo};
use bloomery_levers::{ChurnHosting, HostCfg, NvTierMem};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use engram::direct::DIRECT_ALIGN;
use gguf::Split;
use model::arch::Arch;
use model::moe::HostLayer;
use model::placement::churn::ChurnPool;
use model::placement::need_hook;
use model::placement::workstation::{self, ArenaGrant, HostNeed};
use model::placement::{ArenaFloor, ExpertList, Plan};
use model::r8file::R8Source;
use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};

// -------------------------------------------------------------- chain body

/// One architecture's layer chain: what the shared skeleton drives at capture
/// time, and everything the step touches that is shaped by the model rather
/// than by the runtime — the derived weights, the KV row store, the load-time
/// arena, the weight names, the step's own kernels (docs/arch-split.md,
/// 「트레이트 둘」).
///
/// Monomorphic by construction: `GpuModel<B>` names one body, and a binary
/// that can open two architectures branches once, on the enum around the
/// model, never inside a layer. How a body loads is its own constructor's
/// (`open`, `open_hybrid`, `open_placed` beside the body), which returns the
/// resident [`GpuModel`].
pub trait ChainBody: Sized {
    /// The per-replay host values of one step. deepseek2 is the token and the
    /// position; deepseek41 adds the engram rows and the compressed-index
    /// plan, which is why this is a type and not two arguments.
    type Input;

    /// The host service a replay of this body's chain needs ([`HostServed`]):
    /// the body itself when its chain can hold host work, [`NoHost`] when it
    /// runs on the card alone.
    type Host: HostServed;

    /// Whether this body's residency load takes a churn pool it does not
    /// host ([`bloomery_levers::ChurnHosting::Unhosted`] reaching its
    /// [`ResidencySpec`]): an unset residency word's resolution reads it
    /// ([`bloomery_levers::room_for`]), so a body that does not carry the
    /// pool keeps the rule that raises P until the pool fits.
    const UNHOSTED_POOL: bloomery_levers::UnhostedPool;

    /// The architecture this body is the chain of.
    fn arch() -> Arch;

    /// The input of one decode step at `(token, pos)`. A body whose next step
    /// needs work issued ahead of it does that here, not in `refresh`.
    fn decode_input(&mut self, token: u32, pos: u32) -> Result<Self::Input, GpuError>;

    /// Write `input` into the device buffers the captured graph reads, so one
    /// capture serves every position. Runs before an eager enqueue or a replay
    /// and never inside a capture.
    fn refresh(&mut self, stream: &CudaStream, input: &Self::Input) -> Result<(), GpuError>;

    /// [`ChainBody::decode_input`] and [`ChainBody::refresh`] of a step at
    /// `pos` whose token is not yet known on the host: the step before's
    /// argmax, word 0 of `token` on the card, copied into the step's input
    /// on the stream behind that step. Whatever the step needs of its token
    /// on the host waits for [`ChainBody::chained_token`], after the replay's
    /// launch and before its host service. `Ok(false)` stages nothing: a
    /// body that cannot take its input from the card (the default).
    fn refresh_chained(
        &mut self,
        stream: &CudaStream,
        pos: u32,
        token: &DeviceBuffer<u32>,
    ) -> Result<bool, GpuError> {
        let _ = (stream, pos, token);
        Ok(false)
    }

    /// The token of the step [`ChainBody::refresh_chained`] launched, once
    /// the readback of the step before named it: what the step needs of it
    /// on the host, done before its replay's host service. `None` when that
    /// readback failed: the body lets the launched step end all the same.
    fn chained_token(&mut self, token: Option<u32>) -> Result<(), GpuError> {
        let _ = token;
        Ok(())
    }

    /// Enqueue the whole chain — every resident layer and then `head`.
    /// Asynchronous throughout: no allocation, no synchronization, no host
    /// round trip, so this is both the eager body and what the capture
    /// records.
    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError>;

    /// Rewind every cache to empty. The weights, the arena and any captured
    /// chain stay: none of them depends on the cache contents. The contract
    /// is the load's: every state a call writes and a later call reads before
    /// writing it is back where the load left it, so the next prompt writes
    /// the bits it writes in a fresh process ([`GpuModel::reset`]).
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError>;

    /// The rms epsilon the output head normalizes with, read from the file at
    /// load by this body's plan.
    fn head_eps(&self) -> f32;

    /// What stands between the output head's input and its projection: the
    /// final RMS norm unless the body's own last launch is the final norm.
    fn head_norm(&self) -> HeadNorm {
        HeadNorm::Rms
    }

    /// Device bytes this body holds resident: caches, arena, step module.
    fn resident_bytes(&self) -> usize;

    /// The model layers this load carries, in order: its cache slot `i` is
    /// layer `layers().start + i`.
    fn layers(&self) -> Range<usize>;

    /// The host service of this load, `None` when the chain holds no host
    /// work: what a replay is served by and a failed step's refusal is named
    /// from.
    fn host(&mut self) -> Option<&mut Self::Host>;
}

/// A body whose chain holds host work: a host tier computes part of the
/// captured step, and a replay of the chain is served on the calling thread
/// right after its launch ([`GpuModel::step`], [`GpuModel::step_rows`],
/// [`GpuModel::step_slots`]).
pub trait HostServed {
    /// Serve the host's share of the chain `chain` a graph replay just
    /// submitted, in chain order, before anything else waits on the stream:
    /// [`Chain::Step`] for the one-token step, [`Rows::CHAIN`] for a
    /// multi-row pass, [`SlotRows::chain_of`] for a pass of several slots.
    fn serve_captured(&mut self, chain: Chain) -> Result<(), GpuError>;

    /// The refusal that failed the step's host service, once
    /// ([`crate::hybrid::Hybrid::take_step_refusal`]): what the step's
    /// failure is named from after the stream has drained.
    fn take_host_refusal(&mut self) -> Option<Refusal>;

    /// Lift the host tier's refusal poison
    /// ([`crate::host::HostTier::lift_refusal`]) on `stream`, which
    /// [`GpuModel::reset`] calls once every slot a refused call ran on has
    /// been reset. A body whose tier lifts a refusal in its own
    /// [`ChainBody::reset`] ([`crate::host::HostTier::reset`], the whole
    /// of it) needs nothing here; a body that only settles its tier on a
    /// reset forwards to its tier.
    fn lift_refusal(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        let _ = stream;
        Ok(())
    }

    /// The refusal the host tier is poisoned by now
    /// ([`crate::host::HostTier::refusal_poison`]): what
    /// [`GpuModel::reset`] asks of a tier refusal the model never recorded,
    /// to poison every slot by. The default is none: a body whose tier
    /// names its refusals in its own [`ChainBody::reset`]
    /// ([`crate::host::HostTier::reset`]) records none for the model.
    fn refusal_poison(&self) -> Option<Refusal> {
        None
    }

    /// What a placed load did to the plan's host set — populated, locked —
    /// held by the host tier for its lifetime ([`GpuModel::load_placed`]);
    /// `None` on any other load.
    fn host_residency(&self) -> Option<&HostResidency>;

    /// `e` with what the host tier's residency machine holds now when it is
    /// a [`GpuError::Stalled`] ([`crate::host::HostTier::noted`]): a body
    /// with a machine forwards to its tier; the default leaves `e` as it is.
    fn noted(&self, e: GpuError) -> GpuError {
        e
    }

    /// The residency boundary at `at` on the engine stream `stream`
    /// ([`crate::host::HostTier::swap_at`]): the flips live there land and
    /// the next ones are issued. Before a launch ([`BoundaryAt::Launch`]) it
    /// takes the boundary made ahead of that pass when there is one, else
    /// makes it; ahead ([`BoundaryAt::Ahead`]) it makes the next pass's
    /// boundary under the rest of the pass before it
    /// ([`crate::host::swap::SwapMachine::boundary_ahead`]). A body with no
    /// residency machine does nothing, the load's slot map for the model's
    /// life.
    ///
    /// The contract a body with a machine relies on: every launch that
    /// reads the slot map has one boundary before it, taken or made at
    /// `Launch` after the chain it launches is known to exist; and every
    /// boundary runs after the last launched pass's host service has
    /// returned, so every host word the engine stream's enqueued work waits
    /// on is written, and after that pass's kept rows
    /// ([`HostServed::keep_rows`]). The callers: `launch_served` (every
    /// replay), and through [`GpuModel::pass_boundary`] an eager step or
    /// pass (`GpuModel::run_tokens`, `GpuModel::run_pass`, a pass of
    /// several slots) and a prompt call, whose caller makes one before it
    /// and none inside it — all at `Launch`; and at `Ahead` the passes whose
    /// kept rows are known before their readback, after those kept rows:
    /// each step of `GpuModel::run_tokens` and a pass of several slots
    /// ([`GpuModel::step_slots`]).
    fn at_boundary(&mut self, stream: &CudaStream, at: BoundaryAt) -> Result<(), GpuError> {
        let _ = (stream, at);
        Ok(())
    }

    /// The pass the last boundary opened, a `kind`, keeps `kept` rows: a
    /// step its one row, a verify its accepted rows, a drafted slots pass
    /// each slot's accepted rows, a prompt call none. The
    /// caller that knows the pass's outcome says so before the next
    /// boundary, which refuses a pass with no kept rows by name; rows kept
    /// with no pass open are refused here, the write side of that refusal. A
    /// body with no machine ignores it.
    fn keep_rows(
        &mut self,
        kept: runtime::swaprule::KeptRows,
        kind: crate::host::PassKind,
    ) -> Result<(), GpuError> {
        let _ = (kept, kind);
        Ok(())
    }

    /// The residency back to its seed at a quiet boundary, on `stream`
    /// ([`crate::host::swap::SwapMachine::reset`]); `None` for a body with no
    /// machine.
    fn residency_reset(
        &mut self,
        stream: &CudaStream,
    ) -> Result<Option<crate::host::swap::ResetReport>, GpuError> {
        let _ = stream;
        Ok(None)
    }

    /// The residency to a new seed at a quiet boundary, on `stream`
    /// ([`crate::host::swap::SwapMachine::reseed`]); `None` for a body with
    /// no machine.
    fn residency_reseed(
        &mut self,
        stream: &CudaStream,
        order: &[&[u32]],
    ) -> Result<Option<crate::host::swap::ResetReport>, GpuError> {
        let _ = (stream, order);
        Ok(None)
    }

    /// The residency machine's rule; `None` for a body with no machine.
    fn residency_rule(&self) -> Option<&runtime::swaprule::SwapRule> {
        None
    }

    /// Stop the residency machine before anything of the model is freed
    /// ([`crate::host::HostTier::stop_swap`]): the model's drop calls it
    /// first. A body with no machine does nothing.
    fn stop_residency(&mut self) {}

    /// End the load on every expert tier's node
    /// ([`crate::host::HostTier::close_nodes`]), freeing nothing in this
    /// process. A body whose tier runs on no node does nothing.
    fn close_nodes(&mut self) -> Result<(), GpuError> {
        Ok(())
    }

    /// Start the residency machine the load prepared
    /// ([`crate::host::HostTier::start_swap`]), once the body's pieces are
    /// sized: the machine empties each layer's spare slots, so every piece
    /// that sizes itself from the map's capacity is made first. Load-time
    /// only. A body with no machine does nothing.
    fn start_residency(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let _ = gpu;
        Ok(())
    }
}

/// The host service of a body whose chain runs on the card alone: no value
/// of it exists, so its [`ChainBody::host`] is always `None`.
pub enum NoHost {}

impl HostServed for NoHost {
    fn serve_captured(&mut self, _chain: Chain) -> Result<(), GpuError> {
        match *self {}
    }

    fn take_host_refusal(&mut self) -> Option<Refusal> {
        match *self {}
    }

    fn host_residency(&self) -> Option<&HostResidency> {
        match *self {}
    }
}

/// The most rows one captured chain runs: the size of the graph cache, and
/// the bound on every body's [`Rows::MAX_ROWS`].
pub const MAX_PASS_ROWS: usize = 8;

/// How a pass of [`Rows`] lays its rows over output heads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowHeads {
    /// A head of one row a row: row `r` into head `r`, row 0's the step's.
    PerRow,
    /// One head of `m` rows for the whole pass: the lm_head read once for
    /// every row, each row's logits and token bit for bit its one-row
    /// head's (`Head`'s rows are independent).
    One,
}

/// A body that runs several consecutive positions as one pass of `m` rows,
/// bit for bit `m` one-token steps in turn — what a verify of drafted tokens
/// runs.
pub trait Rows: ChainBody {
    /// The most rows one pass takes, 2 to [`MAX_PASS_ROWS`]. A pass of more
    /// rows does not compile ([`GpuModel::step_rows`]).
    const MAX_ROWS: usize;

    /// How the pass lays its rows over the chain, as a host tier serves a
    /// replay of it ([`HostServed::serve_captured`]).
    const CHAIN: Chain;

    /// How the pass lays its rows over output heads.
    const HEADS: RowHeads = RowHeads::PerRow;

    /// The chain a replay of the pass of `m` rows is served as: [`Rows::CHAIN`]
    /// unless the body's chain depends on the rows.
    fn chain_of(m: usize) -> Chain {
        let _ = m;
        Self::CHAIN
    }

    /// The pass's host half: the steps of `tokens[r]` at `pos + r`, planned
    /// in turn and written into the buffers the captured pass reads.
    /// `tokens` holds 2 to [`Rows::MAX_ROWS`] ids. Never inside a capture.
    fn plan_rows(&mut self, stream: &CudaStream, tokens: &[u32], pos: u32) -> Result<(), GpuError>;

    /// Enqueue the pass the last [`Rows::plan_rows`] planned, through every
    /// resident layer and row `r` into `heads[r]`, one head a row — or, for
    /// [`RowHeads::One`], every row into `heads[0]`, a head of the pass's
    /// rows. Asynchronous as [`ChainBody::enqueue_chain`] is.
    fn enqueue_rows(&mut self, gpu: &Gpu, w: &Weights, heads: &mut [Head]) -> Result<(), GpuError>;
}

/// A body that can take positions back.
pub trait Rollback: ChainBody {
    /// Take back the positions from `pos` on, so the next step runs at `pos`.
    fn rollback(&mut self, pos: u32) -> Result<(), GpuError>;

    /// [`Rollback::rollback`] with the card at hand, for a body whose take
    /// back enqueues work on the engine stream; what [`GpuModel::rollback`]
    /// calls.
    fn rollback_on(&mut self, gpu: &Gpu, pos: u32) -> Result<(), GpuError> {
        let _ = gpu;
        self.rollback(pos)
    }
}

/// A body that can stand at a synthetic depth.
pub trait Instrumented: ChainBody {
    /// Fill the first `rows` cache rows of every resident layer with the
    /// body's own synthetic pattern — the state a prompt of `rows` tokens
    /// leaves behind, without decoding one. An instrument: the rows are not
    /// what the model would have written.
    fn seed_depth(&mut self, gpu: &Gpu, rows: usize) -> Result<(), GpuError>;
}

/// A body whose chain carries the node-price probe's arms.
pub trait Probed: ChainBody {
    /// Arm (or disarm) the node-price probe on this body's arena, and drop
    /// whatever the body captured itself. The skeleton drops its own
    /// captures.
    fn set_probe(&mut self, probe: StepProbe) -> Result<(), GpuError>;
}

/// What a binary or a gate drives, once per token — the surface that does not
/// name an architecture. Static dispatch only: it exists so `generate` and the
/// gate harnesses can be written once over `impl Engine`, not for virtual
/// calls.
pub trait Engine {
    fn step(&mut self, tokens: &[u32]) -> Result<u32, GpuError>;
    fn reset(&mut self) -> Result<(), GpuError>;
    fn seed_depth(&mut self, rows: usize) -> Result<(), GpuError>;
    fn pos(&self) -> u32;
    fn resident_bytes(&self) -> usize;
    fn arch(&self) -> Arch;
}

/// How [`GpuModel::step`] submits the chain. Both modes run the same body
/// over the same resident buffers; the graph mode records it once and
/// replays, so it pays the host submit of ~700 launches only at capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepMode {
    /// Enqueue the whole body per token.
    Eager,
    /// Capture the body on first use, then replay it per token.
    Graph,
}

/// What a [`GpuModel::step_ahead`] did besides answering.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepAhead {
    /// The token fed was the answer whose step the call before launched:
    /// that step was read back and nothing was fed.
    pub took: bool,
    /// The answer's step was launched before the answer was read back: the
    /// model stands past the answer.
    pub launched: bool,
}

// ------------------------------------------------------------ graph cache

/// The captured chains, one per key: the rows the chain runs — 1 for the
/// one-token step, `m` for a pass of [`Rows`]. Nothing else keys a capture:
/// a chain reads every per-call value (position, live key count, a row's
/// lane) from device buffers refreshed before the replay, so one capture
/// serves every call of its shape.
struct Graphs([Option<Graph>; MAX_PASS_ROWS]);

impl Graphs {
    fn new() -> Graphs {
        Graphs([const { None }; MAX_PASS_ROWS])
    }

    /// The capture of `rows` rows, if one is held.
    fn get(&self, rows: usize) -> Option<&Graph> {
        self.0.get(rows.wrapping_sub(1)).and_then(Option::as_ref)
    }

    /// The slot of `rows` rows: 1 to [`MAX_PASS_ROWS`], else refused.
    fn slot(&mut self, rows: usize, what: &'static str) -> Result<&mut Option<Graph>, GpuError> {
        self.0.get_mut(rows.wrapping_sub(1)).ok_or_else(|| {
            GpuError::shape(
                what,
                format!("a chain of {rows} rows; the cache keys 1 to {MAX_PASS_ROWS}"),
            )
        })
    }

    fn any(&self) -> bool {
        self.0.iter().any(Option::is_some)
    }

    fn clear(&mut self) {
        self.0 = [const { None }; MAX_PASS_ROWS];
    }
}

// ------------------------------------------------------------------ model

/// Everything one card holds for a model, as the model's constructor hands it
/// to [`GpuModel::new`]: every piece is resident already.
pub struct Resident<B> {
    pub gpu: Gpu,
    pub weights: Weights,
    pub body: B,
    /// The output head, when the load carries it: a load of some blocks has
    /// no logits to take.
    pub head: Option<Head>,
    /// KV rows the resident caches were sized for.
    pub ctx_max: usize,
}

/// The placed load's pieces between its prelude and its body
/// ([`GpuModel::placed_pre`]).
struct PlacedPre {
    gpu: Gpu,
    weights: Weights,
    layers: Range<usize>,
    head: bool,
    ctx_max: usize,
    /// The prelude's phases; the body and the head are timed after it.
    times: LoadTimes,
}

/// The wall of each phase of a placed load ([`GpuModel::load_placed`],
/// [`GpuModel::load_placed_with`]): what it spent on the card's context and
/// device modules, the weights upload, the derived weights, the plan's host
/// set, the body's build and the output head. Each is the wall of the
/// host-side call, so work a phase only enqueues (a cache's zeroing memset)
/// completes inside whichever later phase first waits on the stream. The
/// placement plan's wall is the body's loader's to stamp
/// ([`GpuModel::note_load_plan`]).
#[derive(Debug, Clone, Copy, Default)]
pub struct LoadTimes {
    /// The inputs read and the placement plan, when the loader stamped it.
    pub plan: Option<Duration>,
    /// The card's context, its engine stream and this crate's device modules.
    pub context: Duration,
    /// The card segments' upload: the staging and the host-to-device copies.
    pub upload: Duration,
    /// The architecture's derived weights.
    pub derive: Duration,
    /// The plan's host set: its r8 source, and the populate walk and the
    /// lock when the load ran them. The walk runs on a helper beside the
    /// context, the upload and the derived weights, so this is the wall the
    /// load waited for it after them, not the walk's own (the populate's
    /// record has that).
    pub host_set: Duration,
    /// The body over the weights: the residency glue when the load runs a
    /// machine, the caches, the ring shadows, the chain pieces, the host
    /// tier, and a tier card's load when the plan hangs one under it.
    pub body: Duration,
    /// The output head, when the card carries it.
    pub head: Duration,
}

/// One resident model on one card. Everything `step` touches is allocated at
/// load, never per step. A second card is another `GpuModel` (a draft's).
///
/// Fields drop in declaration order, after the drop has stopped the engine
/// watchdog and then the body's residency machine, and a graph must be
/// destroyed while every buffer it addresses is still alive: the captured
/// chains first — the live slot's, the passes of several slots', then each
/// parked slot's ([`ParkedSlot`]: its graph cache before its sequence) —
/// then the heads, the body (which drops its own captures first, and its
/// host tier's lock over the host set before the mappings under it), the
/// weights they all address, and the card last.
pub struct GpuModel<B: ChainBody> {
    /// The captured chains, keyed by rows: the live slot's cache, exchanged
    /// with the parked slots' on a [`GpuModel::select_slot`].
    graphs: Graphs,
    /// The captured passes of several slots ([`GpuModel::step_slots`]),
    /// model-wide. Declared before `parked`, the heads and the body: a
    /// capture addresses every busy slot's stores, the heads and the
    /// body's arena ([`SlotGraphs`]).
    slot_graphs: SlotGraphs,
    /// The parked slots ([`GpuModel::add_slots`]), one entry a slot past the
    /// first. Declared after `graphs` and before the heads, the body, the
    /// weights and the card: a parked capture addresses those buffers (and
    /// its own sequence's stores, which its entry drops after the capture).
    parked: Vec<ParkedSlot<B>>,
    /// Row `r`'s output head: row 0's is the one-token step's, made at load
    /// when the load carries the head; rows 1 on are made by the first pass
    /// of [`Rows`] that needs them. Empty on a load without the head.
    heads: Vec<Head>,
    /// A [`RowHeads::One`] body's head of `m` rows at `m − 1`, made by the
    /// first pass of `m` rows.
    pass_heads: Vec<Option<Head>>,
    /// The rows of the last call when it was a [`RowHeads::One`] pass: its
    /// logits are that head's, and the step head's readbacks refuse.
    one_pass: Option<usize>,
    /// The rows of the last call when it was a pass of several slots: what
    /// [`GpuModel::slots_logits`] reads.
    slot_rows: Option<usize>,
    /// The one head's logits as [`GpuModel::slots_logits_into`] fetched them
    /// to take each row's from; its capacity is grown once and reused.
    logits_stage: Vec<f32>,
    /// The launches that read a pass's rows as their best logits, made by
    /// the first such read ([`GpuModel::pass_best_into`]).
    best: Option<Best>,
    /// The ranges of a pass of several slots' verify rows waiting for
    /// [`GpuModel::commit_slots`] ([`GpuModel::verify_slots`]).
    slots_waiting: Option<Vec<SlotRange>>,
    /// On the heap, so moving the model never moves the body's own state.
    body: Box<B>,
    weights: Weights,
    gpu: Gpu,
    /// KV rows the resident caches were sized for; `step` refuses to grow them.
    ctx_max: usize,
    mode: StepMode,
    /// The slot every later call acts on ([`GpuModel::select_slot`]): 0
    /// until a select moves it, and the live fields hold this slot's
    /// position, captures and sequence.
    selected: usize,
    /// The cache row the next `step` token lands in. Written only through
    /// [`GpuModel::stand_at`].
    pos: u32,
    /// The answer whose step the last [`GpuModel::step_ahead`] launched
    /// before reading it back, the model standing past it. Dropped by any
    /// move of the position or of the slot; a call that makes one without a
    /// readback of its own, or hands the sequence on, reads the step's fault
    /// first ([`GpuModel::settle_ahead`]).
    ahead: Option<u32>,
    /// The fault a call read back (crate::fault) and the slots it ran on:
    /// their caches and rings hold what it condemned, so every later call
    /// refuses until a `reset` of each of those slots.
    poisoned: Option<SlotFault>,
    /// The load's phases, when its load timed them ([`LoadTimes`]); `None`
    /// on a load that timed none.
    load_times: Option<LoadTimes>,
    /// The step and pass readbacks begun ([`GpuModel::reads`]): the stamp
    /// every residency boundary carries ([`BoundaryAt`]).
    reads: u64,
    /// The engine thread's progress words ([`crate::watchdog`]), written by
    /// the guarded calls below with relaxed stores only.
    watch: Arc<watchdog::Watch>,
    /// The watchdog thread over `watch`, from `new` to `drop`: a reset
    /// leaves it in place, and the drop stops and joins it before anything
    /// of the model is freed.
    watchdog: watchdog::Watchdog,
}

impl<B: ChainBody> Drop for GpuModel<B> {
    /// The engine watchdog stops and joins first, then the residency machine
    /// ([`HostServed::stop_residency`]): the chains, the heads and the body
    /// free card memory as they drop, and a free waits for a copy the
    /// machine left queued behind a staging word — a wait the watchdog
    /// would otherwise name.
    fn drop(&mut self) {
        self.watchdog.stop();
        if let Some(host) = self.body.host() {
            host.stop_residency();
        }
    }
}

impl<B: ChainBody> GpuModel<B> {
    /// The model over `r`, standing at position 0 in graph mode with nothing
    /// captured, and its engine watchdog started ([`crate::watchdog`]).
    pub fn new(r: Resident<B>) -> GpuModel<B> {
        let Resident {
            gpu,
            weights,
            body,
            head,
            ctx_max,
        } = r;
        let watch = Arc::new(watchdog::Watch::new());
        let watchdog = watchdog::Watchdog::start(Arc::clone(&watch));
        GpuModel {
            graphs: Graphs::new(),
            slot_graphs: SlotGraphs::new(),
            parked: Vec::new(),
            heads: head.into_iter().collect(),
            pass_heads: (0..MAX_PASS_ROWS).map(|_| None).collect(),
            one_pass: None,
            slot_rows: None,
            logits_stage: Vec::new(),
            best: None,
            slots_waiting: None,
            body: Box::new(body),
            weights,
            gpu,
            ctx_max,
            mode: StepMode::Graph,
            selected: 0,
            pos: 0,
            ahead: None,
            poisoned: None,
            load_times: None,
            reads: 0,
            watch,
            watchdog,
        }
    }

    /// Blocks `layers` of the one-shard `file` resident on `gpu`: every
    /// tensor of that range plus the globals in its kernels' device format
    /// ([`Weights::load`]), then `body` over them — the constructor's own
    /// derived weights filed into the set, and the body with its caches, its
    /// m = 1 arena and its step module — and, when `head`, the output head
    /// over those same weights, normalizing with the epsilon the body read.
    /// The geometry checks run before anything is uploaded: a cache of 0
    /// rows, a range outside the file, a file of several shards (a model of
    /// several shards loads by its placement plan, [`GpuModel::load_placed`]).
    pub(crate) fn load_blocks(
        gpu: Gpu,
        file: &Split,
        ctx_max: usize,
        layers: Range<usize>,
        head: bool,
        body: impl FnOnce(&Gpu, &mut Weights) -> Result<B, GpuError>,
    ) -> Result<GpuModel<B>, GpuError> {
        GpuModel::load_blocks_from(
            gpu,
            file,
            (CardBytes::File, &|_| None),
            ctx_max,
            layers,
            head,
            body,
        )
    }

    /// [`GpuModel::load_blocks`] with the card's tensors read from `bytes`: the file's own, or a
    /// part source's when `file` is a checkpoint folder's header ([`CardBytes::Parts`]), each in
    /// the card format `card` gives it (the body's description, `ModelTensors::card_format`, the
    /// format the whole-fit verdict sizes) and in its type's own where `card` gives none. A folder
    /// loads the whole model, every layer and the head: a part of its layers, and a head tied to
    /// the token embedding (which its part load does not alias), are refused by name.
    pub(crate) fn load_blocks_from(
        gpu: Gpu,
        file: &Split,
        (bytes, card): (CardBytes<'_>, CardOf<'_>),
        ctx_max: usize,
        layers: Range<usize>,
        head: bool,
        body: impl FnOnce(&Gpu, &mut Weights) -> Result<B, GpuError>,
    ) -> Result<GpuModel<B>, GpuError> {
        let what = "GpuModel::load_blocks";
        if ctx_max == 0 {
            return Err(GpuError::shape(what, "ctx_max must be >= 1"));
        }
        let n_layers = block_count(file, what)?;
        if layers.start >= layers.end || layers.end > n_layers {
            return Err(GpuError::shape(
                what,
                format!("layer range {layers:?} outside 0..{n_layers}"),
            ));
        }
        one_shard(file, what)?;
        let mut weights = match bytes {
            CardBytes::File => Weights::load_as(gpu.stream(), file, layers, true, card)?,
            CardBytes::Parts(parts) => {
                refuse_partial_folder(file, &layers, n_layers, head)?;
                parts.load_whole(gpu.stream(), card)?
            }
        };
        let body = body(&gpu, &mut weights)?;
        let head = if head {
            Some(Head::of(&gpu, &weights, body.head_eps(), body.head_norm())?)
        } else {
            None
        };
        Ok(GpuModel::new(Resident {
            gpu,
            weights,
            body,
            head,
            ctx_max,
        }))
    }

    /// Card `card` of `plan` resident: the segments the plan puts on the
    /// card ([`Weights::load_placed`] — whole tensors, and each routed
    /// stack's card `ExpertList` of its layer), the weights `derive` files for the card's layers, and the
    /// body `body` builds over them, which keeps `file` for what the plan
    /// leaves on the host and its host tier the host set's residency; plus
    /// the output head when the card carries it. Between the uploads and the
    /// body the plan's host set is read in and locked as `host` asks
    /// ([`HostResidency::at_load`]), so that no step takes the first touch of
    /// a host expert page; `host` also says whether
    /// the card segments' file pages are released once uploaded. The caches hold the plan's `ctx_max` rows, the
    /// context its budget was made for. The card opens on the device the plan
    /// resolved it to, or by its name on a census-free plan
    /// ([`Gpu::open_card`]), never by an ordinal from outside the process. A load under adaptive
    /// residency goes through [`GpuModel::load_placed_with`], which hands the
    /// body the file as one `Arc` the machine's source shares.
    pub fn load_placed(
        file: Split,
        plan: &Plan<'_>,
        card: usize,
        host: HostCfg,
        derive: impl FnOnce(&CudaStream, &Split, Range<usize>, &mut Weights) -> Result<(), GpuError>,
        body: impl FnOnce(&Gpu, Split, &Weights, HostResidency) -> Result<B, GpuError>,
    ) -> Result<GpuModel<B>, GpuError> {
        GpuModel::load_placed_bytes(file, CardBytes::File, plan, card, host, derive, body)
    }

    /// [`GpuModel::load_placed`] with the card's tensors read from `bytes`: the file's own, or a
    /// part source's when `file` is a checkpoint folder's header ([`CardBytes::Parts`]).
    pub fn load_placed_bytes(
        file: Split,
        bytes: CardBytes<'_>,
        plan: &Plan<'_>,
        card: usize,
        host: HostCfg,
        derive: impl FnOnce(&CudaStream, &Split, Range<usize>, &mut Weights) -> Result<(), GpuError>,
        body: impl FnOnce(&Gpu, Split, &Weights, HostResidency) -> Result<B, GpuError>,
    ) -> Result<GpuModel<B>, GpuError> {
        let (p, residency) = placed_pre(&file, bytes, plan, card, host, (&[], 0), &[], derive)?;
        let t_body = Instant::now();
        let body = body(&p.gpu, file, &p.weights, residency)?;
        placed_done(p, body, t_body.elapsed())
    }

    /// [`GpuModel::load_placed`] under `residency` ([`ResidencySpec`]): the
    /// plan's host set also holds each layer's churn pool — the card's
    /// experts past the pinned ones ([`ChurnPool`]), checked before anything
    /// uploads — and the body is handed the file as one `Arc` the machine's
    /// source shares, the host set, and the [`ResidencyGlue`] that starts the
    /// machine over its slot map ([`HostServed::start_residency`], called
    /// once the body's pieces are sized) and carries its boundary calls.
    #[allow(
        clippy::too_many_arguments,
        reason = "load_placed's arguments and the residency spec (rust-quality R8)"
    )]
    pub fn load_placed_with(
        file: Split,
        plan: &Plan<'_>,
        card: usize,
        host: HostCfg,
        residency: ResidencySpec,
        derive: impl FnOnce(&CudaStream, &Split, Range<usize>, &mut Weights) -> Result<(), GpuError>,
        body: impl FnOnce(
            &Gpu,
            &Arc<Split>,
            &Weights,
            HostResidency,
            ResidencyGlue,
        ) -> Result<B, GpuError>,
    ) -> Result<GpuModel<B>, GpuError> {
        GpuModel::load_placed_hosting(file, plan, card, host, residency, (&[], 0), derive, body)
    }

    /// [`GpuModel::load_placed_with`] whose host set also holds `hosted` (its
    /// runs and their file bytes, which the host need counts, and which the
    /// churn pool's check takes out of the plan's host headroom first):
    /// per entry, the experts of the plan's model tensor at that index the
    /// host tier serves though the plan places the tensor nowhere (a draft
    /// layer the body loads beside the plan's own), read in and locked with
    /// the plan's host segments as `host` asks.
    #[allow(
        clippy::too_many_arguments,
        reason = "load_placed_with's arguments and the extra host runs (rust-quality R8)"
    )]
    pub fn load_placed_hosting(
        file: Split,
        plan: &Plan<'_>,
        card: usize,
        host: HostCfg,
        residency: ResidencySpec,
        hosted: (&[(usize, ExpertList)], u64),
        derive: impl FnOnce(&CudaStream, &Split, Range<usize>, &mut Weights) -> Result<(), GpuError>,
        body: impl FnOnce(
            &Gpu,
            &Arc<Split>,
            &Weights,
            HostResidency,
            ResidencyGlue,
        ) -> Result<B, GpuError>,
    ) -> Result<GpuModel<B>, GpuError> {
        GpuModel::load_placed_from(
            file,
            CardBytes::File,
            plan,
            card,
            host,
            residency,
            hosted,
            derive,
            body,
        )
    }

    /// [`GpuModel::load_placed_hosting`] with the card's tensors read from `bytes`: the file's
    /// own, or a part source's when `file` is a checkpoint folder's header
    /// ([`CardBytes::Parts`]).
    #[allow(
        clippy::too_many_arguments,
        reason = "load_placed_hosting's arguments and where the card's bytes come from (rust-quality R8)"
    )]
    pub fn load_placed_from(
        file: Split,
        bytes: CardBytes<'_>,
        plan: &Plan<'_>,
        card: usize,
        host: HostCfg,
        residency: ResidencySpec,
        hosted: (&[(usize, ExpertList)], u64),
        derive: impl FnOnce(&CudaStream, &Split, Range<usize>, &mut Weights) -> Result<(), GpuError>,
        body: impl FnOnce(
            &Gpu,
            &Arc<Split>,
            &Weights,
            HostResidency,
            ResidencyGlue,
        ) -> Result<B, GpuError>,
    ) -> Result<GpuModel<B>, GpuError> {
        const WHAT: &str = "GpuModel::load_placed_with";
        let churn = match residency.lever {
            Residency::Off => None,
            Residency::Mid { pinned, .. } => {
                Some(ChurnPool::of(plan, card, pinned).map_err(|e| GpuError::plan(WHAT, e))?)
            }
        };
        // The host set holds `hosted` beside the plan's own segments, out of
        // the same headroom a hosted pool must fit; a pool the set only
        // knows takes none of it.
        if let Some(p) = &churn {
            match residency.pool {
                ChurnHosting::Hosted => p.check_beside(plan, hosted.1),
                ChurnHosting::Unhosted => p.check_known_beside(plan, hosted.1),
            }
            .map_err(|e| GpuError::plan(WHAT, e))?;
        }
        let runs = churn.as_ref().map_or(&[][..], |p| p.runs.as_slice());
        let (pool, known) = match residency.pool {
            ChurnHosting::Hosted => ((runs, churn.as_ref().map_or(0, |p| p.bytes)), &[][..]),
            ChurnHosting::Unhosted => ((&[][..], 0), runs),
        };
        let joined: Vec<(usize, ExpertList)>;
        let extra = if hosted.0.is_empty() {
            pool
        } else {
            joined = pool.0.iter().chain(hosted.0).cloned().collect();
            (&joined[..], pool.1 + hosted.1)
        };
        let file = Arc::new(file);
        let (p, set) = placed_pre(&file, bytes, plan, card, host, extra, known, derive)?;
        let t_body = Instant::now();
        let glue = match residency.lever {
            Residency::Off => ResidencyGlue::off(),
            lever @ Residency::Mid { .. } => {
                let (params, pinned) = lever
                    .params(residency.delay)
                    .ok_or_else(|| GpuError::shape(WHAT, "a residency machine with no rule"))?;
                let source = FileSwap::new(
                    plan,
                    p.layers.clone(),
                    Arc::clone(&file),
                    host.r8,
                    set.set().clone(),
                    set.populated().is_some(),
                    &p.weights,
                    residency.stacks.as_ref(),
                    p.gpu.context(),
                )?;
                ResidencyGlue::new(
                    source,
                    MachineCfg {
                        params,
                        pinned: vec![
                            pinned;
                            residency.stacks.map_layers().unwrap_or(p.layers.len())
                        ],
                        top_k: residency.top_k,
                        max_rows: residency.max_rows,
                        unrouted: residency.stacks.unrouted(),
                        deadline: residency.deadline,
                    },
                )
            }
        };
        let body = body(&p.gpu, &file, &p.weights, set, glue)?;
        placed_done(p, body, t_body.elapsed())
    }

    /// What the load did to the plan's host set — populated, locked — on a
    /// placed load ([`GpuModel::load_placed`]), as the body's host tier holds
    /// it ([`HostServed::host_residency`]); `None` on any other.
    pub fn host_residency(&mut self) -> Option<&HostResidency> {
        self.body
            .host()
            .map(|h| &*h)
            .and_then(HostServed::host_residency)
    }

    /// The load's phases, when its load timed them ([`LoadTimes`]); `None`
    /// on a load that timed none.
    #[must_use]
    pub fn load_times(&self) -> Option<&LoadTimes> {
        self.load_times.as_ref()
    }

    /// Stamp the placement plan's wall `plan` into the load's phases,
    /// overwriting one already stamped. A load that timed no phases drops
    /// it: a blocks load has no phases to add it to.
    pub fn note_load_plan(&mut self, plan: Duration) {
        if let Some(times) = &mut self.load_times {
            times.plan = Some(plan);
        }
    }

    /// The card this model runs on.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The model's resident weights.
    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    /// The model layers this load carries ([`ChainBody::layers`]).
    pub fn layers(&self) -> Range<usize> {
        self.body.layers()
    }

    /// The positions the resident caches were sized for: the most a
    /// sequence holds.
    pub fn ctx_max(&self) -> usize {
        self.ctx_max
    }

    /// Device bytes of everything this model holds resident: the weights and
    /// the body, plus the heads' scratch — a multi-row pass's heads once one
    /// has run — and every parked slot's sequence
    /// ([`GpuModel::add_slots`]).
    pub fn resident_bytes(&self) -> usize {
        self.weights.resident_bytes()
            + self.body.resident_bytes()
            + self.heads.iter().map(Head::resident_bytes).sum::<usize>()
            + self
                .pass_heads
                .iter()
                .flatten()
                .map(Head::resident_bytes)
                .sum::<usize>()
            + self.parked.iter().map(|p| p.bytes).sum::<usize>()
    }

    /// The cache row the next [`GpuModel::step`] token lands in.
    pub fn pos(&self) -> u32 {
        self.pos
    }

    /// Mark the engine thread busy in a prompt call until the returned
    /// guard drops, for the engine watchdog: a
    /// family's prompt walks the body through [`GpuModel::body_parts`], past
    /// the guarded calls, so the session that drives the prompt holds this
    /// mark. `None` while a guarded call of this model runs on this thread.
    #[must_use]
    pub fn prompt_busy(&self) -> Option<crate::EngineBusy> {
        watchdog::busy(&self.watch, watchdog::PROMPT, self.reads)
    }

    /// The engine watchdog's progress word ([`crate::watchdog`]): it moves
    /// whenever the engine thread ends a wait on the card, lands a residency
    /// boundary, finishes a token's pass inside a guarded call or ends a
    /// guarded call. A gate's instrument.
    #[must_use]
    pub fn engine_progress(&self) -> u64 {
        self.watch.progress()
    }

    /// The step and pass readbacks begun since the load — one a
    /// [`GpuModel::step`] call, one a [`GpuModel::step_rows`] pass: a
    /// residency boundary's report carries the count it ran after
    /// ([`crate::host::swap::PassReport::reads`]), so a boundary made ahead
    /// of a step's readback carries the count before it.
    #[must_use]
    pub fn reads(&self) -> u64 {
        self.reads
    }

    /// Stand at `pos`: the one write of the position.
    fn stand_at(&mut self, pos: u32) {
        self.pos = pos;
        self.ahead = None;
    }

    /// How the chain submits ([`GpuModel::set_mode`]).
    #[must_use]
    pub fn mode(&self) -> StepMode {
        self.mode
    }

    /// Choose how the chain submits. Changing the mode drops every captured
    /// chain — the parked slots' and the passes of several slots' with the
    /// live one's: a graph is a recording of this body over these buffers,
    /// and a later `Graph` run recaptures rather than replay a stale one.
    pub fn set_mode(&mut self, mode: StepMode) {
        if mode != self.mode {
            self.drop_captures();
        }
        self.mode = mode;
    }

    /// Drop every captured chain, the mode kept — the parked slots' and the
    /// passes of several slots' with the live one's: for a body change that
    /// moves what a capture recorded (the buffers its launches address), so
    /// the next `Graph` run recaptures rather than replay a stale one.
    pub(crate) fn drop_captures(&mut self) {
        self.graphs.clear();
        self.slot_graphs.clear();
        for p in &mut self.parked {
            p.graphs.clear();
        }
    }

    /// Whether any chain is captured.
    #[must_use]
    pub fn has_capture(&self) -> bool {
        self.graphs.any()
    }

    /// Rewind to position 0 with empty caches — the fresh-context state for
    /// the next prompt: a prompt after it gives the tokens and logits it
    /// gives in a fresh process right after the load, bit for bit. The
    /// weights, the arena and any captured chain stay (they do not depend on
    /// the cache contents). It rewinds the selected slot's sequence
    /// ([`ChainBody::reset`] sees the live one alone); the parked slots keep
    /// their positions and stores. A poisoning fault names the slots it ran
    /// on (one for a step, every busy slot of a pass of several,
    /// [`GpuModel::step_slots`]): the reset of each takes it off that set,
    /// and the fault word and the poison clear once none is left, or at
    /// once when no other slot is parked — a one-slot model's reset clears a
    /// word whatever raised it; a reset of a slot outside the set rewinds it
    /// and leaves the fault standing, a raised word no call recorded
    /// included. A host service's refusal poisons the slots of its call the
    /// same way, and the host tier's own refusal poison is lifted with it —
    /// a reset of one slot of the set settles the tier and leaves the
    /// refusal standing, so the slots the refused call partly wrote cannot
    /// step again until each has been reset ([`HostServed::lift_refusal`]).
    /// A tier refusal the model never recorded — a service no
    /// recording owner saw, so neither its slots nor its call are known —
    /// poisons every slot at the first reset that meets it, and lifts only
    /// then. A caller that must not continue past one checks
    /// [`GpuModel::poisoned`] first.
    ///
    /// A step launched ahead and not read back yet is read first
    /// ([`GpuModel::settle_ahead`]): its fault is this reset's error, the
    /// slot poisoned by it and not rewound, and the next reset clears it.
    ///
    /// A residency machine's map stays where use has taken it
    /// ([`GpuModel::residency_reset`] is the explicit call).
    pub fn reset(&mut self) -> Result<(), GpuError> {
        let _busy = watchdog::busy(&self.watch, watchdog::RESET, self.reads);
        self.settle_ahead("GpuModel::reset")?;
        // The body owns its row store, so it owns what "empty" means there.
        self.body.reset(&self.gpu)?;
        // Empty caches hold nothing a fault condemned — once every slot the
        // fault ran on is reset ([`SlotFault`]), and on a model with no
        // parked slot, whose word no other sequence can have raised.
        let selected = self.selected;
        let lifted = self
            .poisoned
            .as_mut()
            .is_some_and(|p| p.slots.lift(selected));
        if self.parked.is_empty() || lifted {
            self.gpu.clear_fault()?;
            self.poisoned = None;
            self.lift_host_refusal()?;
        } else if self.poisoned.is_none()
            && let Some(refusal) = self.host_refusal()?
        {
            // A tier refusal this model never recorded (a service that
            // reached no recording owner): the slots it ran on are unknown,
            // so every slot is poisoned until each has been reset — this
            // reset's slot has just been rewound — and no one slot's reset
            // lifts the tier.
            let mut poison = SlotFault {
                slots: SlotSet::every(self.parked.len() + 1, "GpuModel::reset")?,
                why: PoisonWhy::Refused(refusal),
            };
            poison.slots.lift(selected);
            self.poisoned = Some(poison);
        }
        self.one_pass = None;
        self.slot_rows = None;
        // The pass's other slots keep their waiting rows: each is refused by
        // its body until its own commit or reset.
        self.slots_waiting = None;
        self.stand_at(0);
        Ok(())
    }

    /// Lift the host tier's refusal poison
    /// ([`HostServed::lift_refusal`]): what [`GpuModel::reset`] does once
    /// the slots a refused call ran on have all been reset. Nothing for a
    /// body whose chain holds no host work.
    fn lift_host_refusal(&mut self) -> Result<(), GpuError> {
        let GpuModel { body, gpu, .. } = self;
        match body.host() {
            Some(host) => host.lift_refusal(gpu.stream()),
            None => Ok(()),
        }
    }

    /// The refusal the host tier is poisoned by now
    /// ([`HostServed::refusal_poison`]): an unrecorded tier refusal, what a
    /// reset that meets one poisons every slot by. Nothing for a body whose
    /// chain holds no host work.
    fn host_refusal(&mut self) -> Result<Option<Refusal>, GpuError> {
        Ok(match self.body.host() {
            Some(host) => host.refusal_poison(),
            None => None,
        })
    }

    /// Capture the whole chain — every resident layer plus the head — into
    /// one graph over the resident buffers, key 1, and return its node count.
    /// One graph, not a chain of per-layer graphs: the layers differ only in
    /// which weights and which cache they address, all of them frozen at
    /// load, and a single `cuGraphLaunch` is the whole point of the capture
    /// (a chain of 27 launches would pay 27 host submits per token).
    pub fn capture_step(&mut self) -> Result<usize, GpuError> {
        const WHAT: &str = "GpuModel::capture_step";
        let GpuModel {
            graphs,
            heads,
            body,
            weights,
            gpu,
            ..
        } = self;
        let head = heads.first_mut().ok_or(no_head(WHAT))?;
        let graph = gpu.capture(|_| body.enqueue_chain(gpu, weights, head))?;
        let nodes = graph.node_count();
        *graphs.slot(1, WHAT)? = Some(graph);
        Ok(nodes)
    }

    /// Every node of the captured whole chain, as the driver lists them —
    /// what a structure gate counts the kinds of.
    pub fn step_graph_nodes(&self) -> Result<Vec<NodeInfo>, GpuError> {
        self.graphs
            .get(1)
            .ok_or(GpuError::state(
                "GpuModel::step_graph_nodes",
                "no captured chain",
            ))?
            .nodes()
    }

    /// The most nodes on one dependency path of the captured whole chain
    /// ([`Graph::span`](crate::Graph::span)) — what a structure gate holds a
    /// body's forks to.
    pub fn step_graph_span(&self) -> Result<usize, GpuError> {
        self.graphs
            .get(1)
            .ok_or(GpuError::state(
                "GpuModel::step_graph_span",
                "no captured chain",
            ))?
            .span()
    }

    /// Enqueue the whole chain eagerly on the engine stream. Pure enqueues —
    /// the same body [`GpuModel::capture_step`] records.
    fn enqueue_chain_step(&mut self) -> Result<(), GpuError> {
        let GpuModel {
            heads,
            body,
            weights,
            gpu,
            ..
        } = self;
        let head = heads.first_mut().ok_or(no_head("GpuModel::step"))?;
        body.enqueue_chain(gpu, weights, head)
    }

    /// Feed `tokens` through the chain one position at a time and return the
    /// argmax of the LAST one — the greedy next token. Each token gets its
    /// own parameter refresh (outside any capture) and one body; only the
    /// argmax readback at the end synchronizes the stream, so a prompt of P
    /// tokens is P bodies and one sync, unless the body's refresh waits for
    /// the walk before (a step whose rows arrive after its launch: one sync
    /// a token).
    ///
    /// Positions continue from wherever the model stands: a prompt then its
    /// continuation is `step(&prompt)` followed by one `step(&[tok])` per
    /// generated token. [`GpuModel::reset`] rewinds. A fault any token
    /// raised is the call's error, also when a later token then fails for
    /// another reason ([`GpuModel::note_fault`]). A model that carries the
    /// head carries every layer: each constructor that makes the head loads
    /// the whole chain.
    pub fn step(&mut self, tokens: &[u32]) -> Result<u32, GpuError> {
        let _busy = watchdog::busy(&self.watch, watchdog::STEP, self.reads);
        if tokens.is_empty() {
            return Err(GpuError::shape("GpuModel::step", "empty token slice"));
        }
        self.ready_step("GpuModel::step")?;
        let token = self.run_tokens(tokens);
        self.note_fault("GpuModel::step", token)
    }

    /// The checks of a one-token step, [`GpuModel::step`]'s and
    /// [`GpuModel::step_ahead`]'s: refused by name on a poisoned model,
    /// while a pass of several slots waits for its commit and with no head;
    /// in graph mode the step captured on its first use.
    fn ready_step(&mut self, what: &'static str) -> Result<(), GpuError> {
        self.refuse_if_poisoned(what)?;
        self.refuse_if_slots_wait(what)?;
        if self.heads.is_empty() {
            return Err(no_head(what));
        }
        if self.mode == StepMode::Graph && self.graphs.get(1).is_none() {
            self.capture_step()?;
        }
        Ok(())
    }

    /// [`GpuModel::step`]'s tokens once its checks have passed: each
    /// position's refresh and chain, its kept row and then the next pass's
    /// residency boundary made ahead ([`HostServed::at_boundary`] at
    /// [`BoundaryAt::Ahead`]: its host time runs under the rest of the
    /// step, the head's read included), then the head's readback. This and
    /// a pass of several slots ([`GpuModel::step_slots`]) are where a
    /// boundary runs ahead of its pass. An error here can
    /// follow launches, so the caller passes it through
    /// [`GpuModel::note_fault`]; a boundary that fails leaves the model at
    /// the step's position, as a refused kept row does.
    fn run_tokens(&mut self, tokens: &[u32]) -> Result<u32, GpuError> {
        self.one_pass = None;
        self.slot_rows = None;
        for &token in tokens {
            self.launch_token(token)?;
        }
        self.reads += 1;
        self.heads
            .first()
            .ok_or(no_head("GpuModel::step"))?
            .token(&self.gpu)
    }

    /// One position of [`GpuModel::run_tokens`]: `token`'s refresh and
    /// chain at the model's position, its kept row, the next pass's boundary
    /// made ahead, and the model one position on.
    fn launch_token(&mut self, token: u32) -> Result<(), GpuError> {
        let pos = self.pos;
        self.check_pos(pos, "GpuModel::step")?;
        self.refresh_params(token, pos)?;
        let r = match self.mode {
            StepMode::Eager => self
                .pass_boundary()
                .and_then(|()| self.enqueue_chain_step()),
            StepMode::Graph => self.replay(1, Chain::Step),
        };
        self.name_host_refusal(r)?;
        self.keep_rows(
            runtime::swaprule::KeptRows::prefix(1),
            crate::host::PassKind::Step,
        )?;
        self.boundary_at(BoundaryAt::Ahead { reads: self.reads })?;
        self.stand_at(pos + 1);
        watchdog::pass_done();
        Ok(())
    }

    /// [`GpuModel::step`] of one token for a caller that feeds the answer
    /// back unless its generation stops there: the step of `token` (taken as
    /// it runs when it is the answer whose step the last call launched),
    /// then, when the answer's position is below `within` (the positions the
    /// caller holds; 0 for none) and the model can (graph mode, the position
    /// inside the cache, a body that takes its input from the card,
    /// [`ChainBody::refresh_chained`]), the step of the answer launched
    /// before the answer is read back ([`GpuModel::launch_chained`]). When
    /// it launched, the model stands past the answer: the next `step_ahead`
    /// of it takes that step, and any other call keeps the answer fed and
    /// follows it on the stream, a call that hands the sequence on or reads
    /// nothing back reading the step's fault first
    /// ([`GpuModel::settle_ahead`]).
    pub fn step_ahead(&mut self, token: u32, within: u32) -> Result<(u32, StepAhead), GpuError> {
        const WHAT: &str = "GpuModel::step_ahead";
        let _busy = watchdog::busy(&self.watch, watchdog::STEP, self.reads);
        self.ready_step(WHAT)?;
        let r = self.run_ahead(token, within);
        self.note_fault(WHAT, r)
    }

    /// [`GpuModel::step_ahead`] once its checks have passed.
    fn run_ahead(&mut self, token: u32, within: u32) -> Result<(u32, StepAhead), GpuError> {
        const WHAT: &str = "GpuModel::step_ahead";
        let took = self.ahead.take() == Some(token);
        if !took {
            self.one_pass = None;
            self.slot_rows = None;
            self.launch_token(token)?;
            self.reads += 1;
        }
        let pos = self.pos;
        if pos < within
            && self.mode == StepMode::Graph
            && self.check_pos(pos, WHAT).is_ok()
            && let Some(answer) = self.launch_chained(pos)?
        {
            let launched = true;
            return Ok((answer, StepAhead { took, launched }));
        }
        let head = self.heads.first().ok_or(no_head(WHAT))?;
        let launched = false;
        Ok((head.token(&self.gpu)?, StepAhead { took, launched }))
    }

    /// The step at `pos` chained behind the one just launched, in graph
    /// mode: its input from the head's argmax on the card
    /// ([`ChainBody::refresh_chained`]), the head's readback copied behind
    /// the step before ([`Head::copy_tokens`]), its replay launched, then the
    /// answer read back and handed to the body ([`ChainBody::chained_token`])
    /// before the replay's host service, its kept row and the next pass's
    /// boundary; the model stands past it, marked as launched ahead. Returns
    /// the answer, or `None` when the body stages nothing. Once the replay
    /// is launched the body hears of the answer on every path, a failed
    /// readback included, so the replay ends.
    fn launch_chained(&mut self, pos: u32) -> Result<Option<u32>, GpuError> {
        const WHAT: &str = "GpuModel::step_ahead";
        let GpuModel {
            graphs,
            heads,
            body,
            gpu,
            reads,
            ..
        } = self;
        let head = heads.first_mut().ok_or(no_head(WHAT))?;
        if !body.refresh_chained(gpu.stream(), pos, head.token_words())? {
            return Ok(None);
        }
        head.copy_tokens(gpu)?;
        let graph = graphs
            .get(1)
            .ok_or(GpuError::state(WHAT, "no captured chain"))?;
        if let Some(host) = body.host() {
            host.at_boundary(gpu.stream(), BoundaryAt::Launch { reads: *reads })?;
        }
        graph.launch(gpu.stream())?;
        let answer = head.copied_token();
        let named = body.chained_token(answer.as_ref().ok().copied());
        let served = match body.host() {
            Some(host) => host.serve_captured(Chain::Step),
            None => Ok(()),
        };
        let answer = answer?;
        named?;
        self.name_host_refusal(served)?;
        self.keep_rows(
            runtime::swaprule::KeptRows::prefix(1),
            crate::host::PassKind::Step,
        )?;
        self.boundary_at(BoundaryAt::Ahead { reads: self.reads })?;
        self.stand_at(pos + 1);
        self.reads += 1;
        self.ahead = Some(answer);
        watchdog::pass_done();
        Ok(Some(answer))
    }

    /// The fault of the step [`GpuModel::step_ahead`] launched ahead while no
    /// step has read it back: the engine stream waited for and the fault
    /// word read ([`Gpu::fault`]). A call that hands the sequence on (a
    /// save, [`GpuModel::reset`]) or moves it without a readback of its own
    /// ([`GpuModel::rollback`], [`GpuModel::select_slot`]) runs this first,
    /// so the step's fault is that call's error and poisons the slot it ran
    /// on ([`GpuModel::note_fault`]), the step no longer pending. Nothing
    /// when no step is launched ahead, so decode steps wait for nothing more;
    /// a clean word leaves the step to the next `step_ahead` of its answer.
    pub(crate) fn settle_ahead(&mut self, what: &'static str) -> Result<(), GpuError> {
        if self.ahead.is_none() {
            return Ok(());
        }
        let read = match self.gpu.fault() {
            Ok(Some(fault)) => Err(GpuError::fault(what, fault)),
            r => r.map(drop),
        };
        if read.is_err() {
            self.ahead = None;
        }
        self.note_fault(what, read)
    }

    /// Launch the captured chain of `rows` rows and serve the host's share
    /// of the replay as `chain` ([`launch_served`]).
    fn replay(&mut self, rows: usize, chain: Chain) -> Result<(), GpuError> {
        let GpuModel {
            graphs,
            body,
            gpu,
            reads,
            ..
        } = self;
        let graph = graphs
            .get(rows)
            .ok_or(GpuError::state("GpuModel::replay", "no captured chain"))?;
        launch_served(graph, body.as_mut(), gpu, *reads, chain)
    }

    /// The residency boundary before a pass the caller enqueues itself (a
    /// prompt call, before its first group and none inside it), and before
    /// every eager step or pass here ([`HostServed::at_boundary`] at
    /// [`BoundaryAt::Launch`]: the boundary a step made ahead of this pass
    /// taken, else made now). Nothing for a body with no host service.
    /// Refused by name first while a pass of several slots waits for its
    /// commit ([`GpuModel::refuse_if_slots_wait`]): no pass opens inside it.
    pub fn pass_boundary(&mut self) -> Result<(), GpuError> {
        self.refuse_if_slots_wait("GpuModel::pass_boundary")?;
        let _busy = watchdog::busy(&self.watch, watchdog::PASS_BOUNDARY, self.reads);
        self.boundary_at(BoundaryAt::Launch { reads: self.reads })
    }

    /// The host service's residency boundary at `at`; nothing for a body
    /// with no host service.
    fn boundary_at(&mut self, at: BoundaryAt) -> Result<(), GpuError> {
        let GpuModel { body, gpu, .. } = self;
        match body.host() {
            Some(host) => host.at_boundary(gpu.stream(), at),
            None => Ok(()),
        }
    }

    /// The pass the last boundary opened keeps `kept` rows
    /// ([`HostServed::keep_rows`]): what its caller, which knows the pass's
    /// outcome, says before the next pass.
    pub fn keep_rows(
        &mut self,
        kept: runtime::swaprule::KeptRows,
        kind: crate::host::PassKind,
    ) -> Result<(), GpuError> {
        match self.body.host() {
            Some(host) => host.keep_rows(kept, kind),
            None => Ok(()),
        }
    }

    /// The residency back to its seed ([`HostServed::residency_reset`]);
    /// `None` for a body with no machine. Only an explicit call does this:
    /// [`GpuModel::reset`] leaves the residency where use has taken it.
    pub fn residency_reset(&mut self) -> Result<Option<crate::host::swap::ResetReport>, GpuError> {
        let _busy = watchdog::busy(&self.watch, watchdog::RESIDENCY_RESET, self.reads);
        let GpuModel { body, gpu, .. } = self;
        match body.host() {
            Some(host) => host.residency_reset(gpu.stream()),
            None => Ok(None),
        }
    }

    /// End the load on every expert tier's node ([`HostServed::close_nodes`])
    /// and free nothing: what a server that leaves the model to the process
    /// exit calls instead of its Drop, which would close them and then free
    /// the model through the device driver. No call on the model follows.
    pub fn close_nodes(&mut self) -> Result<(), GpuError> {
        match self.body.host() {
            Some(host) => host.close_nodes(),
            None => Ok(()),
        }
    }

    /// The residency to a new seed ([`HostServed::residency_reseed`]),
    /// `order[i]` the ranking of the machine's `i`-th layer, what every later
    /// [`GpuModel::residency_reset`] returns to; `None` for a body with no
    /// machine.
    pub fn residency_reseed(
        &mut self,
        order: &[&[u32]],
    ) -> Result<Option<crate::host::swap::ResetReport>, GpuError> {
        let _busy = watchdog::busy(&self.watch, watchdog::RESIDENCY_RESET, self.reads);
        let GpuModel { body, gpu, .. } = self;
        match body.host() {
            Some(host) => host.residency_reseed(gpu.stream(), order),
            None => Ok(None),
        }
    }

    /// The residency machine's rule ([`HostServed::residency_rule`]): its
    /// layers, seed and usage counts; `None` for a body with no machine.
    pub fn residency_rule(&mut self) -> Option<&runtime::swaprule::SwapRule> {
        self.body.host().and_then(|h| h.residency_rule())
    }

    /// A step's, pass's or prompt call's result, with a host refusal named:
    /// when the body's host service refused input the card should already
    /// have refused, it released every wait of the step, so the call drains,
    /// and the fault word read on the engine stream behind it names the
    /// refusal ([`crate::hybrid::name_refusal`]): the card's fault when the
    /// card raised one at or before that layer, else the host's error. When the word holds a
    /// later layer's fault, the caller's [`GpuModel::note_fault`] returns
    /// that fault with the host's error behind it, so the call ends poisoned
    /// and still names the refusal's layer, which came first. A failed read
    /// names the refusal and the call's error beside its own. Any other
    /// result passes through. An unsettled release keeps its error: the
    /// refusal is taken and poisons the call's slots, and the stream whose
    /// waits still stand is not read.
    ///
    /// On a model that parks another slot the refusal also poisons the slots
    /// the call ran on (`slots`, [`PoisonWhy::Refused`]): the layers before
    /// the refused one had already written their in-place state, so every
    /// later call is refused by name until each of those slots has been
    /// reset, and the tier's own refusal poison is lifted only then
    /// ([`GpuModel::reset`]). A model with no parked slot keeps the load's
    /// own refusal: the tier's, lifted by the load's reset.
    pub(crate) fn name_host_refusal<T>(&mut self, r: Result<T, GpuError>) -> Result<T, GpuError> {
        let slots = SlotSet::one(self.selected);
        self.name_host_refusal_by(r, slots, |_| {})
    }

    /// [`GpuModel::name_host_refusal`] of a call that ran on `slots`, with
    /// the refusal's detail rewritten by `name` first: what a pass whose
    /// rows are not one sequence's positions adds to say whose row it was.
    fn name_host_refusal_by<T>(
        &mut self,
        r: Result<T, GpuError>,
        slots: SlotSet,
        name: impl FnOnce(&mut Refusal),
    ) -> Result<T, GpuError> {
        let (body, parked, poisoned) = (&mut self.body, &self.parked, &mut self.poisoned);
        errors::name_host_refusal(
            r,
            || {
                let mut refusal = body.host().and_then(HostServed::take_host_refusal)?;
                name(&mut refusal);
                if !parked.is_empty() && poisoned.is_none() {
                    *poisoned = Some(SlotFault {
                        slots,
                        why: PoisonWhy::Refused(refusal.clone()),
                    });
                }
                Some(refusal)
            },
            || self.gpu.fault(),
        )
    }

    /// The step head's readback after a [`RowHeads::One`] pass, whose rows
    /// that head never saw, is refused by name.
    fn refuse_one_pass(&self, what: &'static str) -> Result<(), GpuError> {
        match self.one_pass {
            Some(m) => Err(GpuError::shape(
                what,
                format!(
                    "the last call was a pass of {m} rows into one head of {m} rows; \
                     `rows_logits` (a verify) or `slots_logits` (a pass of several slots) \
                     reads its rows"
                ),
            )),
            None => Ok(()),
        }
    }

    /// Row `r` of the pass `read` describes into `row` (`n_vocab` f32): the
    /// one read of a pass's row that [`GpuModel::pass_row_into`] and
    /// [`GpuModel::slots_logits_into`] take by their body's layout.
    /// [`RowHeads::PerRow`] copies head `r` alone; [`RowHeads::One`] copies
    /// the pass's one head into the stage unless this pass's head is there
    /// already, and takes row `r` from it (`[v·m + r]`). Blocking read.
    /// Refused by name, as `what`, for a row past the pass's rows, a `row`
    /// of another length, and under [`RowHeads::One`] when the last call
    /// was not a pass of those rows.
    fn pass_row(
        &self,
        what: &'static str,
        read: &mut PassRead<'_>,
        r: usize,
        row: &mut [f32],
    ) -> Result<(), GpuError> {
        let m = read.rows;
        if r >= m {
            return Err(GpuError::shape(what, format!("row {r} of a pass of {m}")));
        }
        match read.layout {
            RowHeads::PerRow => self
                .heads
                .get(r)
                .ok_or(GpuError::state(what, "no pass of these rows has run"))?
                .logits_into_host(&self.gpu, row),
            RowHeads::One => {
                let head = one_head(&self.pass_heads, self.one_pass, m, what)?;
                let n_vocab = head.n_vocab();
                if row.len() != n_vocab {
                    return Err(GpuError::shape(
                        what,
                        format!("a row of {} f32 for {n_vocab} logits", row.len()),
                    ));
                }
                if !read.fetched {
                    read.stage.resize(m * n_vocab, 0.0);
                    head.logits_into_host(&self.gpu, read.stage)?;
                    read.fetched = true;
                } else if read.stage.len() != m * n_vocab {
                    return Err(GpuError::shape(
                        what,
                        format!(
                            "a fetched pass of {} f32 for {m} rows of {n_vocab}",
                            read.stage.len()
                        ),
                    ));
                }
                for (o, &v) in row.iter_mut().zip(read.stage.iter().skip(r).step_by(m)) {
                    *o = v;
                }
                Ok(())
            }
        }
    }

    /// Row `r` of the pass `read` describes read as its best logits into
    /// `row` ([`best::ROW_WORDS`] words, [`crate::best`]'s row): the read a
    /// pick that needs only a row's best takes, in place of
    /// [`GpuModel::pass_row`]'s whole row. [`RowHeads::PerRow`] reads head `r`
    /// alone; [`RowHeads::One`] reads every row of the pass's one head into
    /// the stage unless this pass's rows are there already, and takes row
    /// `r`'s words from it. Blocking read. Refused by name, as `what`, as
    /// [`GpuModel::pass_row`] refuses, and for a `row` of another length.
    fn pass_best(
        &mut self,
        what: &'static str,
        read: &mut PassRead<'_, u32>,
        r: usize,
        row: &mut [u32],
    ) -> Result<(), GpuError> {
        let m = read.rows;
        if r >= m || row.len() != best::ROW_WORDS {
            return Err(GpuError::shape(
                what,
                format!(
                    "row {r} of a pass of {m} into {} words of {}",
                    row.len(),
                    best::ROW_WORDS
                ),
            ));
        }
        let n_vocab = self.n_vocab()?;
        let launches = match &mut self.best {
            Some(launches) => launches,
            none => none.insert(Best::new(&self.gpu, n_vocab)?),
        };
        read.stage.resize(best::MAX_ROWS * best::ROW_WORDS, 0);
        let at = match read.layout {
            RowHeads::PerRow => {
                let head = self
                    .heads
                    .get(r)
                    .ok_or(GpuError::state(what, "no pass of these rows has run"))?;
                launches.enqueue(&self.gpu, head.logits(), 1)?;
                launches.read(&self.gpu, read.stage)?;
                0
            }
            RowHeads::One => {
                let head = one_head(&self.pass_heads, self.one_pass, m, what)?;
                if !read.fetched {
                    launches.enqueue(&self.gpu, head.logits(), m)?;
                    launches.read(&self.gpu, read.stage)?;
                    read.fetched = true;
                }
                r * best::ROW_WORDS
            }
        };
        row.copy_from_slice(&read.stage[at..at + best::ROW_WORDS]);
        Ok(())
    }

    /// The heads a pass of `m` rows laid as `layout` writes: for
    /// [`RowHeads::PerRow`] row `r`'s head for every `r < m` (row 0's is the
    /// load's, the rest made here), for [`RowHeads::One`] one head of `m`
    /// rows; each over the resident weights.
    fn make_heads(
        &mut self,
        m: usize,
        layout: RowHeads,
        what: &'static str,
    ) -> Result<(), GpuError> {
        if self.heads.is_empty() {
            return Err(no_head(what));
        }
        match layout {
            RowHeads::PerRow => {
                while self.heads.len() < m {
                    let head = Head::with_norm(
                        &self.gpu,
                        &self.weights,
                        self.body.head_eps(),
                        1,
                        self.body.head_norm(),
                    )?;
                    self.heads.push(head);
                }
            }
            RowHeads::One => {
                let slot = self.pass_heads.get_mut(m.wrapping_sub(1)).ok_or_else(|| {
                    GpuError::shape(what, format!("a head of {m} rows (1..={MAX_PASS_ROWS})"))
                })?;
                if slot.is_none() {
                    *slot = Some(Head::with_norm(
                        &self.gpu,
                        &self.weights,
                        self.body.head_eps(),
                        m,
                        self.body.head_norm(),
                    )?);
                }
            }
        }
        Ok(())
    }

    /// Refused by name while a pass of several slots' verify rows waits for
    /// its commit ([`GpuModel::verify_slots`]): a select would move a
    /// waiting slot live, and any other call would run on a sequence whose
    /// rows are not yet kept.
    pub(crate) fn refuse_if_slots_wait(&self, what: &'static str) -> Result<(), GpuError> {
        match &self.slots_waiting {
            None => Ok(()),
            Some(ranges) => Err(GpuError::shape(
                what,
                format!(
                    "a call while the pass of slots {:?} waits for its commit \
                     (GpuModel::commit_slots, or reset)",
                    ranges.iter().map(|r| r.slot).collect::<Vec<_>>()
                ),
            )),
        }
    }

    /// Plant `fault` in the card's fault word, ordered on the engine stream:
    /// the next readback carries it ([`GpuError::Fault`]), as a raised word
    /// does — a test seam for the slots harness's poison clauses
    /// (`crates/gpu-gates/src/slots_gate.rs` H5), on any load, whether or
    /// not its chain holds host work. Gate use; never on a serving path.
    pub fn plant_fault(&self, fault: Fault) -> Result<(), GpuError> {
        let word = self.gpu.fault_word();
        let mut words = vec![0u32; word.len()];
        words[0] = fault.word();
        words[crate::fault::mask_index(fault.layer)] = fault.sites;
        // SAFETY: `words` spans the fault allocation's own length, so both
        // the source and the destination ranges lie inside live host and
        // device memory of this context, and the copy is ordered on the
        // engine stream like every launch that touches the allocation
        // ([`Gpu::clear_fault`]).
        let rc = unsafe {
            cuda_core::sys::cuMemcpyHtoDAsync_v2(
                word.cu_deviceptr(),
                words.as_ptr().cast(),
                std::mem::size_of_val(&words[..]),
                self.gpu.stream().cu_stream(),
            )
        };
        crate::graph::cu(rc, "cuMemcpyHtoDAsync_v2 (fault word)")?;
        Ok(())
    }

    /// A step, or a gate/debug instrument's call, on a poisoned model is
    /// refused, naming the fault — and, once the model serves more than one
    /// slot, the slots it ran on that are not reset yet. A host service's
    /// refusal of the call's input and a commit of several slots' rows that
    /// failed partway poison the same way, naming the refusal's layer and
    /// the slot its keep failed at.
    pub(crate) fn refuse_if_poisoned(&self, what: &'static str) -> Result<(), GpuError> {
        let (slots, why) = match &self.poisoned {
            None => return Ok(()),
            Some(p) => (&p.slots, &p.why),
        };
        let (is, which) = if slots.as_slice().len() == 1 {
            ("is", "that slot")
        } else {
            ("are", "each of them")
        };
        match why {
            // The one-sequence refusal stays the load's own: no other slot
            // exists for the fault to name.
            PoisonWhy::Fault(fault) if self.parked.is_empty() => Err(GpuError::Poisoned {
                what,
                fault: *fault,
            }),
            PoisonWhy::Fault(fault) => Err(GpuError::shape(
                what,
                format!(
                    "{} {is} poisoned by an earlier device fault at {fault:#}; select {which} and \
                     reset() to clear it",
                    slots
                ),
            )),
            PoisonWhy::Refused(r) => Err(GpuError::shape(
                what,
                format!(
                    "{} {is} poisoned by the host service's refusal of layer {} ({}); select \
                     {which} and reset() to clear it",
                    slots, r.layer, r.detail
                ),
            )),
            PoisonWhy::SlotKeep { slot, error } => Err(GpuError::shape(
                what,
                format!(
                    "{} {is} poisoned by a commit of several slots' rows that failed at slot \
                     {slot}, after the keeps before it had run ({error}); select {which} and \
                     reset() to clear it",
                    slots
                ),
            )),
        }
    }

    /// Pass through the result of `what`, a call that may have launched,
    /// remembering a fault it carries: the state the faulted call wrote
    /// poisons every later call until [`GpuModel::reset`]. A fault is the
    /// call's error whatever else failed: another error is read behind
    /// ([`GpuModel::fault_behind`]) unless its stream is stalled or its
    /// release unsettled. The fault prints its site mask in this
    /// body's step order. A body crate passes its own launches' results
    /// through it — a draft layer's walk poisons the model as a step does.
    pub fn note_fault<T>(
        &mut self,
        what: &'static str,
        r: Result<T, GpuError>,
    ) -> Result<T, GpuError> {
        self.note_fault_in(what, r, SlotSet::one(self.selected))
    }

    /// [`GpuModel::note_fault`] of a call that ran on `slots`: a fault it
    /// carries poisons every one of them.
    fn note_fault_in<T>(
        &mut self,
        what: &'static str,
        r: Result<T, GpuError>,
        slots: SlotSet,
    ) -> Result<T, GpuError> {
        let mut e = match errors::note_fault(r, |e| match e {
            // The stream did not drain within the bound: the fault word's
            // read would wait it out again.
            e @ GpuError::Stalled { .. } => match self.body.host() {
                Some(h) => h.noted(e),
                None => e,
            },
            e => self.fault_behind(what, e),
        }) {
            Ok(v) => return Ok(v),
            Err(e) => e,
        };
        if let GpuError::Fault { fault, .. } = &mut e {
            *fault = fault.in_order(crate::arch::step_order::of(B::arch()));
            self.poisoned = Some(SlotFault {
                slots,
                why: PoisonWhy::Fault(*fault),
            });
        }
        Err(e)
    }

    /// `e`, the error of `what` after launches that may have raised the
    /// fault word, named by the word: the engine stream waited for, then the
    /// word read. A raised word is the error whatever `e` was, with `e`
    /// behind it, so the fault never outlives the call that raised it — the
    /// next call's readback would name it. A clean word leaves `e`. A failed
    /// wait or read names `e` beside its own error.
    fn fault_behind(&self, what: &'static str, e: GpuError) -> GpuError {
        let gpu = &self.gpu;
        let read = gpu.fault();
        match read {
            Ok(Some(fault)) => GpuError::Fault {
                what,
                fault,
                behind: Some(Box::new(e)),
            },
            Ok(None) => e,
            Err(s) => GpuError::shape(
                what,
                format!(
                    "the call failed ({e}), and waiting for its launches or reading the fault \
                     word failed too ({s})"
                ),
            ),
        }
    }

    /// The fault that poisons this model, if a step has read one back. The
    /// slot it was live on is the refusal's to name
    /// ([`GpuModel::reset`] lifts the fault on that slot alone). `None`
    /// also when the poison is a commit's half-settled keep or a host
    /// service's refusal, which name no device fault: a reader that must
    /// not continue past any poison goes through the model's own refusal
    /// instead.
    #[must_use]
    pub fn poisoned(&self) -> Option<Fault> {
        match &self.poisoned {
            Some(p) => match &p.why {
                PoisonWhy::Fault(fault) => Some(*fault),
                // A half-settled commit or a host refusal names no device
                // fault; a reader that must not continue past any poison goes
                // through the model's own refusal instead.
                PoisonWhy::Refused(_) | PoisonWhy::SlotKeep { .. } => None,
            },
            None => None,
        }
    }

    /// The head's logits of the last `step` (`n_vocab` f32). Blocking read;
    /// gate/debug use. After [`GpuModel::step_rows`] these are row 0's: the
    /// pass writes row 0 through this same head and row `r` through head `r`
    /// ([`GpuModel::rows_logits`]).
    pub fn logits(&self) -> Result<Vec<f32>, GpuError> {
        self.refuse_one_pass("GpuModel::logits")?;
        self.heads
            .first()
            .ok_or(no_head("GpuModel::logits"))?
            .logits_to_host(&self.gpu)
    }

    /// The vocabulary size: the logits a row holds.
    pub fn n_vocab(&self) -> Result<usize, GpuError> {
        Ok(self
            .heads
            .first()
            .ok_or(no_head("GpuModel::n_vocab"))?
            .n_vocab())
    }

    /// [`GpuModel::logits`] into `out` (`n_vocab` f32), for a caller that reads
    /// them every token into one buffer. Blocking read.
    pub fn logits_into(&self, out: &mut [f32]) -> Result<(), GpuError> {
        self.refuse_one_pass("GpuModel::logits_into")?;
        self.heads
            .first()
            .ok_or(no_head("GpuModel::logits_into"))?
            .logits_into_host(&self.gpu, out)
    }

    /// The head's normed rows of the last `step` or prompt call (`result_norm`, `hidden`
    /// f32 a row): the call's last position's row. Blocking read; gate/debug use.
    pub fn head_normed(&self) -> Result<Vec<f32>, GpuError> {
        self.refuse_one_pass("GpuModel::head_normed")?;
        self.heads
            .first()
            .ok_or(no_head("GpuModel::head_normed"))?
            .normed_to_host(&self.gpu)
    }

    // --------------------------------------------- what the body's own impl
    // --------------------------------------------- blocks are handed

    /// The card, the weights and the body — for the body's own instruments,
    /// in whichever crate the body lives. Never fails: a model is resident
    /// from its constructor on. The `Result` and `what` are its callers'
    /// shape.
    pub fn body_parts(
        &mut self,
        _what: &'static str,
    ) -> Result<(&Gpu, &Weights, &mut B), GpuError> {
        Ok((&self.gpu, &self.weights, &mut *self.body))
    }

    /// [`GpuModel::body_parts`] beside the head's normed rows of the last
    /// `step` or prompt call on the card ([`GpuModel::head_normed`]'s rows,
    /// not read back), for a body's launch on the engine stream that reads
    /// them where they sit. Refused as `head_normed` is.
    pub(crate) fn body_parts_normed(
        &mut self,
        what: &'static str,
    ) -> Result<(&Gpu, &Weights, &mut B, &DeviceBuffer<f32>), GpuError> {
        self.refuse_one_pass(what)?;
        let head = self.heads.first().ok_or(no_head(what))?;
        Ok((&self.gpu, &self.weights, &mut *self.body, head.normed()))
    }

    /// The body, for an instrument that only reads it. Never fails, as
    /// [`GpuModel::body_parts`].
    pub fn body(&self, _what: &'static str) -> Result<&B, GpuError> {
        Ok(&*self.body)
    }

    /// Build the body's input for `(token, pos)` and write it to the device.
    /// Runs before an eager enqueue or a graph replay; a captured graph reads
    /// those buffers at run time, which is what lets one graph serve every
    /// position.
    pub(crate) fn refresh_params(&mut self, token: u32, pos: u32) -> Result<(), GpuError> {
        let input = self.body.decode_input(token, pos)?;
        self.body.refresh(self.gpu.stream(), &input)
    }

    /// The slot of layer `l` inside the resident range — the index of its
    /// cache and of its names in the body.
    pub(crate) fn layer_slot(&self, l: usize, what: &'static str) -> Result<usize, GpuError> {
        let layers = self.body.layers();
        if !layers.contains(&l) {
            return Err(GpuError::shape(
                what,
                format!("layer {l} is outside the resident range {layers:?}"),
            ));
        }
        Ok(l - layers.start)
    }

    /// The engine stream — what an instrument synchronizes on after driving
    /// a replay. Never fails, as [`GpuModel::body_parts`].
    pub(crate) fn stage_stream(&self) -> Result<&CudaStream, GpuError> {
        Ok(self.gpu.stream())
    }

    /// Err unless `pos` leaves room for one more row in the resident cache.
    pub(crate) fn check_pos(&self, pos: u32, what: &'static str) -> Result<(), GpuError> {
        if pos as usize + 1 > self.ctx_max {
            return Err(GpuError::shape(
                what,
                format!(
                    "pos {pos} + 1 exceeds the resident cache's {} rows",
                    self.ctx_max
                ),
            ));
        }
        Ok(())
    }

    /// Run `n` positions as one eager pass the body enqueues itself — `pass`
    /// gets the card, the weights, the body, the output head and the first
    /// position — and stand `n` positions on. When `pass` reports that it
    /// enqueued the head, return the head's token (a blocking read); a pass
    /// that does not is not read back. A fault the pass returns, the head's
    /// readback carries, or the word holds behind any other error of the pass
    /// ([`GpuModel::note_fault`]) poisons the model. So does the host
    /// service's refusal of the pass's input, named with the selected
    /// slot — the slot every prompt walk's pass runs on
    /// ([`GpuModel::name_host_refusal`]).
    ///
    /// The prompt call's fault-read contract has two named modes, and each
    /// body runs one:
    /// - **call-end read** (qwen3moe's prefill): only the call's last pass
    ///   enqueues the head, so the word is read once, behind the head at the
    ///   call's end (a call that runs no head, `CallEnd::Unread`, reads the
    ///   word itself at the end of its last pass); a fault an earlier pass
    ///   raised is named there, and the passes between it and the end ran on
    ///   the condemned state.
    /// - **group-end read** (deepseek41's prompt call): every group's pass
    ///   returns `true` only for the group that holds the call's last
    ///   position, and a group that fails is read behind at once
    ///   ([`GpuModel::fault_behind`]), so a fault does not cross the failed
    ///   call; the call takes its positions back before it returns the error.
    pub fn run_rows(
        &mut self,
        n: usize,
        what: &'static str,
        pass: impl FnOnce(&Gpu, &Weights, &mut B, &mut Head, u32) -> Result<bool, GpuError>,
    ) -> Result<Option<u32>, GpuError> {
        let _busy = watchdog::busy(&self.watch, watchdog::RUN_ROWS, self.reads);
        self.refuse_if_poisoned(what)?;
        self.refuse_if_slots_wait(what)?;
        if n == 0 {
            return Err(GpuError::shape(what, "a pass needs positions"));
        }
        let (pos, n) = (self.pos, crate::launch_u32(what, "positions", n)?);
        self.check_pos(pos + n - 1, what)?;
        self.one_pass = None;
        self.slot_rows = None;
        let GpuModel {
            heads,
            body,
            weights,
            gpu,
            ..
        } = self;
        let head = heads.first_mut().ok_or(no_head(what))?;
        let token = match pass(gpu, weights, &mut **body, head, pos) {
            Ok(true) => head.token(gpu).map(Some),
            Ok(false) => Ok(None),
            Err(e) => Err(e),
        };
        let token = self.name_host_refusal(token);
        let token = self.note_fault(what, token)?;
        self.stand_at(pos + n);
        Ok(token)
    }
}

/// Launch `graph`, a capture of `body`'s, on the engine stream and serve the
/// host's share of the replay, as `chain`, when the body has a host service
/// (`GpuModel::replay`, a pass of several slots). `GpuModel::launch_chained`
/// replays the step in these steps by hand, reading the answer between the
/// launch and the serve: a change to the order here is one there. The host
/// service's residency boundary runs first, stamped with `reads`
/// ([`HostServed::at_boundary`] at [`BoundaryAt::Launch`]): the caller has
/// looked `graph` up, so the boundary runs once the chain is known to exist.
fn launch_served<B: ChainBody>(
    graph: &Graph,
    body: &mut B,
    gpu: &Gpu,
    reads: u64,
    chain: Chain,
) -> Result<(), GpuError> {
    if let Some(host) = body.host() {
        host.at_boundary(gpu.stream(), BoundaryAt::Launch { reads })?;
    }
    graph.launch(gpu.stream())?;
    match body.host() {
        Some(host) => host.serve_captured(chain),
        None => Ok(()),
    }
}

/// The pinned host memory `ctx`'s card maps ([`MappedHost::read_in_place`]:
/// write-combined on an integrated device alone) as the NVMe tier's arena
/// pages, which a pin gives the card's address of.
pub(crate) struct PinnedSource<'a>(pub(crate) &'a Arc<CudaContext>);

impl PageSource for PinnedSource<'_> {
    /// `len` bytes from the allocation's first [`DIRECT_ALIGN`]-aligned
    /// byte, which a direct fill's window starts on. `cuMemHostAlloc`
    /// promises no alignment, and a driver can hand a small allocation out of
    /// a chunk it shares with others, off a page boundary, so the allocation
    /// takes [`DIRECT_ALIGN`] bytes more.
    fn alloc(&self, len: usize) -> Result<Box<dyn ArenaPages>, GpuError> {
        let page = MappedHost::read_in_place(
            self.0,
            len + DIRECT_ALIGN,
            "cuMemHostAlloc (the NVMe tier's arena; `BLOOMERY_NVTIER_MEM=anon` takes it \
             from anonymous memory instead)",
        )?;
        let first = page.host_at(0) as usize;
        let skip = first.next_multiple_of(DIRECT_ALIGN) - first;
        Ok(Box::new(PinnedPages { page, skip, len }))
    }
}

/// One allocation of mapped host memory, its arena starting `skip` bytes in.
struct PinnedPages {
    page: MappedHost,
    skip: usize,
    len: usize,
}

impl ArenaPages for PinnedPages {
    fn base(&self) -> *mut u8 {
        self.page.host_at(self.skip)
    }

    fn bytes(&self) -> usize {
        self.len
    }

    fn device_base(&self) -> Option<u64> {
        Some(self.page.dev_at(self.skip))
    }
}

/// The arena `lever` asks for on `source`'s device: pinned pages when the
/// lever pins on a device of that integrated bit ([`NvTierMem::pinned`]), the
/// bit write-combining reads too, else anonymous.
pub(crate) fn arena_kind_of<'a>(
    source: &'a PinnedSource<'_>,
    lever: NvTierMem,
) -> Result<ArenaKind<'a>, GpuError> {
    let integrated = crate::raw_device_integrated(source.0.cu_device())?;
    Ok(if lever.pinned(integrated) {
        ArenaKind::Provided(source)
    } else {
        ArenaKind::Anon
    })
}

/// The NVMe expert tier of a paged load, the common chain's one build
/// every family's host run attaches
/// ([`HostRun::attach_tier`](crate::host::run::HostRun::attach_tier)):
/// the arena `arena` grants (the split dial's `HostTotals::nvme_arena_bytes`,
/// or what the host's reading left of it at load, [`HostResidency::arena`])
/// over `file`'s shards, read `O_DIRECT` or — the lever's fallback —
/// through the page cache, in pinned host memory `ctx`'s card maps or in
/// anonymous memory, as the lever asks of `ctx`'s device: pinned on an
/// integrated device by default ([`bloomery_levers::NvTierMem`]), its
/// victims taken as the lever's rule asks ([`bloomery_levers::NvTierEvict`]). `None`
/// on a plan that pages no routed expert or reserved no arena: the host
/// leg reads the file mapping as it stands. Refused by name: a load
/// whose r8 pair reads a sidecar while the plan pages routed experts —
/// the slot's bytes follow the sidecar resolution, which the tier does
/// not carry yet.
pub fn nvme_tier(
    ctx: &Arc<CudaContext>,
    plan: &Plan<'_>,
    file: &Arc<Split>,
    r8: bool,
    arena: ArenaGrant,
) -> Result<Option<Arc<NvTier>>, GpuError> {
    const WHAT: &str = "GpuModel::nvme_tier";
    if plan.host.nvme_expert_bytes == 0 {
        return Ok(None);
    }
    let levers = bloomery_levers::nvtier_levers().map_err(|e| GpuError::plan(WHAT, e))?;
    let pair = model::r8file::R8Pair::at_load(Arc::clone(file), r8)
        .map_err(|e| GpuError::plan(WHAT, e))?;
    if pair.source().sidecar().is_some() {
        return Err(GpuError::shape(
            WHAT,
            "a paged plan under an r8 pair that reads a sidecar: the NVMe tier's slots follow \
             the sidecar's bytes, which the tier does not fill from yet (`BLOOMERY_R8=off` \
             reads the source: same bits)",
        ));
    }
    let source = PinnedSource(ctx);
    let mem = arena_kind_of(&source, levers.mem)?;
    NvTier::of_paged(plan, file, levers.direct, arena, mem, levers.evict).map(|t| t.map(Arc::new))
}

/// The host leg of a load on `plan`: the routed layers of `set` (the layers
/// that carry a host share, ascending, contiguous or not) built by `layers`,
/// given the NVMe tier's arena `arena` grants when the plan pages routed
/// experts and reserved one, on `ctx`'s card ([`nvme_tier`], then
/// [`HostRun::build_at`], then [`HostRun::attach_tier`]). The one owner of a
/// body's host-run build.
#[allow(
    clippy::too_many_arguments,
    reason = "the run's plan, file, layer set, widths and layers, and its tier's card, r8 lever and arena, each named"
)]
pub fn host_run<E>(
    ctx: &Arc<CudaContext>,
    plan: &Plan<'_>,
    file: &Arc<Split>,
    set: impl IntoIterator<Item = usize>,
    r8: bool,
    arena: ArenaGrant,
    widths: HostWidths,
    layers: impl FnOnce(R8Source<'_>) -> Result<Vec<HostLayer>, E>,
) -> Result<HostRun, GpuError>
where
    E: std::error::Error + Send + Sync + 'static,
{
    let arena = nvme_tier(ctx, plan, file, r8, arena)?;
    let mut run = HostRun::build_at(Arc::clone(file), set, r8, widths, layers)?;
    run.attach_tier(arena)?;
    Ok(run)
}

/// The layers and the context a body that runs every layer on one card with no expert tier loads at: card `card`'s
/// layers of `plan`, which must be `0..n_layers`, and `ctx_max` as a positive `u32` count of positions. Refused by
/// name as `what`'s: a plan of other cards or an expert tier card ([`refuse_expert_tiers`]: a reserve-only member
/// is none), another layer range, a context of no position or past the `u32` the position words count.
pub fn one_card_run(
    what: &'static str,
    plan: &Plan<'_>,
    card: usize,
    n_layers: usize,
) -> Result<(Range<usize>, usize), GpuError> {
    refuse_expert_tiers(what, plan.machine)?;
    stage_run(what, plan, card, n_layers)
}

/// [`one_card_run`] for a body that hangs the plan's expert tier cards under its host tier: the stage card's layers
/// and context, the tier cards the load's own to count ([`crate::host::refuse_tier_count`]). Refused by name as
/// `what`'s: a plan of other than one stage card, another layer range, a context of no position or past the `u32`
/// the position words count.
pub fn stage_run(
    what: &'static str,
    plan: &Plan<'_>,
    card: usize,
    n_layers: usize,
) -> Result<(Range<usize>, usize), GpuError> {
    if plan.machine.cards.len() != 1 {
        return Err(GpuError::shape(
            what,
            format!(
                "a plan of {} cards; the program runs every layer on one stage card",
                plan.machine.cards.len(),
            ),
        ));
    }
    let layers = plan
        .machine
        .cards
        .get(card)
        .map(|c| c.layers.clone())
        .ok_or_else(|| GpuError::shape(what, format!("the plan has no card {card}")))?;
    if layers != (0..n_layers) {
        return Err(GpuError::shape(
            what,
            format!(
                "card {card} runs layers {layers:?}; the program runs every layer (0..{n_layers})"
            ),
        ));
    }
    let ctx = usize::try_from(plan.ctx_max)
        .ok()
        .filter(|&c| c > 0 && u32::try_from(c).is_ok())
        .ok_or_else(|| {
            GpuError::shape(
                what,
                format!(
                    "ctx_max {}: at least 1 and at most the positions a u32 counts",
                    plan.ctx_max
                ),
            )
        })?;
    Ok((layers, ctx))
}

/// What the layers of a description agree on: the first layer's `value` is kept in `first`, and every other layer's
/// must equal it, or the layer is refused by name as `what`'s `owner`.
pub fn agreed<T: PartialEq + std::fmt::Debug>(
    owner: &'static str,
    what: &str,
    l: usize,
    first: &mut Option<T>,
    value: T,
) -> Result<(), GpuError> {
    match first {
        Some(f) if *f != value => Err(GpuError::shape(
            owner,
            format!("layer {l}: {what} is {value:?}, an earlier layer's {f:?}"),
        )),
        _ => {
            first.get_or_insert(value);
            Ok(())
        }
    }
}

/// The host tier of a hybrid body's load: the boundary of `rows` rows, the host run over `run` built by `layers`
/// ([`host_run`]) and attached to the residency `glue`, and the [`Hybrid`] over them and the slot `map`, watching the
/// card's fault word and keeping the host set `set` ([`HostResidency`]); the arena leg is attached when the load
/// starts a residency machine. The one owner of a body's hybrid assembly after its stores and card experts.
#[allow(
    clippy::too_many_arguments,
    reason = "the card, file, plan, run and map, the host levers and set, the glue, widths and rows, and the layers' builder, each named"
)]
pub fn hosted_hybrid<E>(
    (gpu, file, plan): (&Gpu, &Arc<Split>, &Plan<'_>),
    (run, map): (Range<usize>, SlotMap),
    (host, set, glue): (HostCfg, HostResidency, &ResidencyGlue),
    (widths, rows): (HostWidths, usize),
    layers: impl FnOnce(R8Source<'_>) -> Result<Vec<HostLayer>, E>,
) -> Result<Hybrid<HostRun>, GpuError>
where
    E: std::error::Error + Send + Sync + 'static,
{
    hosted_hybrid_tiered(
        (gpu, file, plan),
        (run, map),
        (host, set, glue),
        (widths, rows),
        None,
        layers,
    )
}

/// [`hosted_hybrid`] for a body that hangs the plan's expert tier cards under the host tier (`hang`,
/// [`leg_tier::hang_tiers`]): they are attached after the [`Hybrid`] is made and before the arena leg, which a tier
/// keeps off. `None` is [`hosted_hybrid`].
#[allow(
    clippy::too_many_arguments,
    reason = "the card, file, plan, run and map, the host levers and set, the glue, widths and rows, the tier hang and the layers' builder, each named"
)]
pub fn hosted_hybrid_tiered<E>(
    (gpu, file, plan): (&Gpu, &Arc<Split>, &Plan<'_>),
    (run, map): (Range<usize>, SlotMap),
    (host, set, glue): (HostCfg, HostResidency, &ResidencyGlue),
    (widths, rows): (HostWidths, usize),
    hang: Option<&leg_tier::Hang<'_>>,
    layers: impl FnOnce(R8Source<'_>) -> Result<Vec<HostLayer>, E>,
) -> Result<Hybrid<HostRun>, GpuError>
where
    E: std::error::Error + Send + Sync + 'static,
{
    let hosted: Vec<usize> = run.clone().collect();
    hosted_hybrid_at(
        (gpu, file, plan),
        (&hosted, run.len(), map),
        (host, set, glue),
        (widths, rows),
        hang,
        layers,
    )
}

/// [`hosted_hybrid_tiered`] over the layers of `hosted` — the model layers that carry a host share, strictly ascending,
/// not necessarily contiguous (a layer whose experts the card holds whole is none) — of a chain of `chain` layers: the
/// host run's views are `layers`' one per hosted layer in order, and the [`Hybrid`] counts the whole chain. A body whose
/// every layer is hosted names `0..chain`, which is [`hosted_hybrid_tiered`]. The one owner of a body's hybrid
/// assembly after its stores and card experts.
#[allow(
    clippy::too_many_arguments,
    reason = "the card, file, plan, hosted layers, chain and map, the host levers and set, the glue, widths and rows, the tier hang and the layers' builder, each named"
)]
pub fn hosted_hybrid_at<E>(
    (gpu, file, plan): (&Gpu, &Arc<Split>, &Plan<'_>),
    (hosted, chain, map): (&[usize], usize, SlotMap),
    (host, set, glue): (HostCfg, HostResidency, &ResidencyGlue),
    (widths, rows): (HostWidths, usize),
    hang: Option<&leg_tier::Hang<'_>>,
    layers: impl FnOnce(R8Source<'_>) -> Result<Vec<HostLayer>, E>,
) -> Result<Hybrid<HostRun>, GpuError>
where
    E: std::error::Error + Send + Sync + 'static,
{
    let boundary = Boundary::with_rows(
        gpu.context(),
        gpu.stream(),
        BoundaryShape {
            hidden: widths.embd,
            n_used: widths.n_used,
        },
        rows,
    )?;
    let experts = host_run(
        gpu.context(),
        plan,
        file,
        hosted.iter().copied(),
        host.r8,
        set.arena(),
        widths,
        layers,
    )?;
    glue.attach_tier(experts.tier())?;
    let mut hybrid = Hybrid::new(boundary, map, experts, chain)?;
    hybrid.watch_fault(gpu.fault_word())?;
    hybrid.keep_residency(set);
    if let Some(hang) = hang {
        leg_tier::hang_tiers((gpu, file, plan), hang, &mut hybrid)?;
    }
    // The residency machine frees arena slots between passes, which a card hold keeps: a load that starts one keeps
    // the host leg.
    hybrid.attach_arena_leg(gpu, glue.source().is_some())?;
    Ok(hybrid)
}

/// Where a placed load's card tensors come from ([`GpuModel::load_placed_from`]).
pub enum CardBytes<'a> {
    /// The file's own mapped tensors ([`Weights::load_placed`]).
    File,
    /// A part source's, for a file that is a checkpoint folder's header
    /// (`gguf::Split::header_only`): card `card` of `plan` loaded on the stream, each part's file
    /// pages released when the host config's `card_dontneed` asks (`crate::parts::PlacedSource`).
    Parts(&'a dyn PartsLoad),
}

/// A part source's host reservation and load of card `card` of `plan`.
pub trait PartsLoad: Send {
    /// Plan within offered host room, including pinned upload staging.
    fn host_bytes(&self, plan: &Plan<'_>, card: usize, room: u64) -> Result<u64, GpuError>;
    /// Load within the checked peak returned by host_bytes.
    fn load(
        &self,
        stream: &CudaStream,
        plan: &Plan<'_>,
        card: usize,
        release: bool,
        room: u64,
    ) -> Result<Weights, GpuError>;

    /// Every tensor the source makes on the card, each whole in the card format `card` gives it
    /// and in its type's own where `card` gives none: the load of a model that sits wholly on
    /// one card, with no plan ([`GpuModel::load_blocks_from`]).
    fn load_whole(&self, stream: &CudaStream, card: CardOf<'_>) -> Result<Weights, GpuError>;

    /// The host admission and exact read peak to pass to load: the reads planned within the room
    /// the need's other terms leave them, `extra` out ([`HostNeed::read_room`]), in place of the
    /// least reads the plan reserved (`workstation::PART_READS`).
    fn host_need(
        &self,
        plan: &Plan<'_>,
        card: usize,
        extra: u64,
        available: u64,
    ) -> Result<(HostNeed, u64), GpuError> {
        let mut need = HostNeed::of(plan, extra);
        let peak = self.host_bytes(plan, card, need.read_room(available))?;
        need.read_ahead = 0;
        if peak > 0 {
            let ring = crate::upload::UploadRing::HOST_MAX;
            need.read_ahead = peak.checked_sub(ring).ok_or_else(|| {
                GpuError::shape("PartsLoad", "read peak is under its staging ring")
            })?;
            need.shadows = need
                .shadows
                .checked_add(ring)
                .ok_or_else(|| GpuError::shape("PartsLoad", "pinned ring bytes pass u64"))?;
        }
        Ok((need, peak))
    }
}

/// The steps of a placed load before its body: the plan's card and context
/// budget, the card, the host's memory against the plan's host need, the
/// card's weights (`bytes`': the file's, or a part source's) with `derive`'s
/// filings, and the plan's host set read in and locked as `host` asks, with
/// `extra`'s runs (of `extra`'s bytes) beside it and `known`'s runs known,
/// neither read in nor locked nor counted ([`HostResidency::at_load_with`]).
/// The pieces, and the host set.
///
/// What waits for what. The refusals that need no read come first and alone —
/// the need hook's (`BLOOMERY_NEED_HOOK`), a plan's card that is not one
/// visible device ([`Gpu::resolve_card`]) and a host whose `MemAvailable` is
/// under the plan's need ([`HostNeed`]) — and the host's reading is taken
/// before anything starts reading, so the host set's pages do not move it.
/// Then the host set's walk (its r8 source, the populate, the lock) starts on
/// a helper, beside the card's context and module load, the weights' upload
/// and `derive`: the drive and the driver's JIT do not share a resource, and
/// the upload's reads of the card segments and the walk's reads of the host
/// segments are different bytes of one drive. The upload releases a card
/// segment's pages as it stages it (whole pages inside the segment only), so
/// it never drops a page of the host set. The body needs the host set, so the
/// helper is joined before it, and the time that join waits is the load's
/// `host_set` phase. A failure of the card side ends the load only once the
/// walk has ended too: the walk has no cancel. A plan whose host set also
/// holds experts of card segments (`extra`, the residency machine's churn
/// pool) reads them in and locks them where the upload releases the card
/// segments' pages, so that load keeps the old order: the walk after the
/// upload and derive. The host need counts on that order: a part source's
/// reads end with the upload, so the need holds them and `extra` never at
/// once ([`HostNeed::read_room`]).
#[allow(
    clippy::too_many_arguments,
    reason = "the load's file and where its card bytes come from, its plan and card, the host's levers, the extra and known host runs, and the derive (rust-quality R8)"
)]
fn placed_pre(
    file: &Split,
    bytes: CardBytes<'_>,
    plan: &Plan<'_>,
    card: usize,
    host: HostCfg,
    extra: (&[(usize, ExpertList)], u64),
    known: &[(usize, ExpertList)],
    derive: impl FnOnce(&CudaStream, &Split, Range<usize>, &mut Weights) -> Result<(), GpuError>,
) -> Result<(PlacedPre, HostResidency), GpuError> {
    let what = "GpuModel::load_placed";
    let spec = plan
        .machine
        .cards
        .get(card)
        .ok_or_else(|| GpuError::shape(what, format!("the plan has no card {card}")))?;
    let ctx_max = usize::try_from(plan.ctx_max)
        .ok()
        .filter(|&c| c > 0)
        .ok_or_else(|| GpuError::shape(what, format!("the plan's ctx_max {}", plan.ctx_max)))?;
    let layers = spec.layers.clone();
    let (extra, extra_bytes) = extra;
    // The hook answers before the load opens its card or reads a byte in: a
    // broker that refuses the load holds nothing of the load's.
    need_hook::consult().map_err(|e| GpuError::plan(what, e))?;
    let ordinal = Gpu::resolve_card(&spec.name, spec.device)?;
    let reading = workstation::host_available_read().map_err(|e| GpuError::plan(what, e))?;
    // A plan with an arena may find the host short of its need by less than
    // the arena's room above its floor: the tier is then built over what
    // the reading leaves, never silently — the `nvtier` record names both.
    let least = if plan.host.nvme_arena_bytes > 0 {
        ArenaFloor::of(plan)
            .map_err(|e| GpuError::plan(what, e))?
            .bytes
    } else {
        0
    };
    let (need, room) = match &bytes {
        CardBytes::File => (HostNeed::of(plan, extra_bytes), 0),
        CardBytes::Parts(parts) => parts.host_need(plan, card, extra_bytes, reading.0)?,
    };
    let grant = need
        .grant(&reading, least)
        .map_err(|e| GpuError::plan(what, format!("{e}{}", plan.host.room_clause())))?;
    let walk = || HostResidency::at_load_with(file, plan, |_| true, host, extra, known);
    let card_side = || -> Result<(Gpu, Weights, [Duration; 3]), GpuError> {
        let t = Instant::now();
        let gpu = Gpu::with_device(ordinal)?;
        let context = t.elapsed();
        let t = Instant::now();
        let mut weights = match bytes {
            CardBytes::File => {
                Weights::load_placed(gpu.stream(), file, plan, card, host.card_dontneed)?
            }
            CardBytes::Parts(parts) => {
                parts.load(gpu.stream(), plan, card, host.card_dontneed, room)?
            }
        };
        let upload = t.elapsed();
        let t = Instant::now();
        derive(gpu.stream(), file, layers.clone(), &mut weights)?;
        Ok((gpu, weights, [context, upload, t.elapsed()]))
    };
    // `extra` is experts of card segments the host set also reads in and may
    // lock: their pages lie inside the segments the upload releases, so a walk
    // beside the upload would lock a page the release is dropping. With none,
    // the host set and the card segments share no whole page.
    let (card_side, host_side, host_set) = if extra.is_empty() {
        let (card, host, waited) = beside(what, walk, card_side)?;
        (card, Some(host), waited)
    } else {
        let card = card_side();
        let t = Instant::now();
        let host = card.is_ok().then(walk);
        (card, host, t.elapsed())
    };
    let (gpu, weights, [context, upload, derived]) = card_side?;
    let residency = host_side
        .ok_or_else(|| GpuError::shape(what, "the host set's walk did not run"))??
        .with_arena(grant);
    Ok((
        PlacedPre {
            gpu,
            weights,
            layers,
            head: spec.head,
            ctx_max,
            times: LoadTimes {
                plan: None,
                context,
                upload,
                derive: derived,
                host_set,
                body: Duration::ZERO,
                head: Duration::ZERO,
            },
        },
        residency,
    ))
}

/// Run `background` on a helper thread beside `foreground` on this one, and
/// join the helper whichever way `foreground` went: both results, and how long
/// the join waited after `foreground` returned. The helper floats off the
/// caller's cpu mask (`threads::helper::float_caller`), as a binary pins its
/// load thread to one cpu. A panic on the helper is the caller's; a helper that
/// cannot start is `what`'s error and `foreground` does not run.
fn beside<A, B: Send>(
    what: &'static str,
    background: impl FnOnce() -> Result<B, GpuError> + Send,
    foreground: impl FnOnce() -> A,
) -> Result<(A, Result<B, GpuError>, Duration), GpuError> {
    std::thread::scope(|scope| {
        let helper = std::thread::Builder::new()
            .name("load-beside".to_owned())
            .spawn_scoped(scope, move || {
                threads::helper::float_caller().map_err(|e| GpuError::plan(what, e))?;
                background()
            })
            .map_err(|e| GpuError::plan(what, format!("a load helper did not spawn: {e}")))?;
        let done = foreground();
        let t = Instant::now();
        let background = match helper.join() {
            Ok(background) => background,
            Err(panic) => std::panic::resume_unwind(panic),
        };
        Ok((done, background, t.elapsed()))
    })
}

/// Refused by name, as `what`'s, when the load's scratch, `made` device
/// bytes, passes what card `card` of `plan` sets aside for it: the card's
/// `scratch_bytes` term and the stage's position rows the plan counts on that
/// card (`CardTotals::stage_rows_bytes`, the rope tables at the context). The
/// plan counts those bytes by those terms alone, no formula of the body's, so
/// the load holds them to it. `parts` names what the body's scratch holds, for
/// the message.
pub fn refuse_scratch_past(
    what: &'static str,
    plan: &Plan<'_>,
    card: usize,
    made: usize,
    parts: &str,
) -> Result<(), GpuError> {
    let term = plan
        .machine
        .cards
        .get(card)
        .map(|c| c.scratch_bytes)
        .ok_or_else(|| GpuError::shape(what, format!("the plan has no card {card}")))?;
    let rows = stage_rows(what, plan, card)?;
    let term = term + rows;
    if made as u64 > term {
        return Err(GpuError::shape(
            what,
            format!(
                "the load's scratch holds {made} device bytes ({parts}); card {card}'s plan \
                 sets aside {term} for it ({rows} of them the stage's position rows)"
            ),
        ));
    }
    Ok(())
}

/// Refused by name, as `what`'s, when buffers the load made beside the plan's
/// terms, `made` device bytes of `name` (`parts` says what they hold), pass the
/// `reserve` the plan kept for them out of the expert budget
/// (`PlanInputs::prompt_reserve`): the plan counts those bytes by the family's
/// one formula, no sum of the body's, so the load holds them to it.
pub fn refuse_reserve_past(
    what: &'static str,
    name: &str,
    reserve: u64,
    made: u64,
    parts: &str,
) -> Result<(), GpuError> {
    if made > reserve {
        return Err(GpuError::shape(
            what,
            format!(
                "the load's {name} hold {made} device bytes ({parts}); the plan reserves \
                 {reserve} for them"
            ),
        ));
    }
    Ok(())
}

/// Refused by name, as `what`'s, when the stage's position rows the load
/// made, `made` device bytes, pass what card `card` of `plan` counts for them
/// (`CardTotals::stage_rows_bytes`, the rope tables at the context): the check
/// of a body whose scratch the plan holds by a term of its own rather than
/// through [`refuse_scratch_past`].
pub(crate) fn refuse_stage_rows_past(
    what: &'static str,
    plan: &Plan<'_>,
    card: usize,
    made: u64,
) -> Result<(), GpuError> {
    let rows = stage_rows(what, plan, card)?;
    if made > rows {
        return Err(GpuError::shape(
            what,
            format!(
                "the stage's position rows hold {made} device bytes; card {card}'s plan counts \
                 {rows} for them"
            ),
        ));
    }
    Ok(())
}

/// The stage's position rows card `card` of `plan` counts.
fn stage_rows(what: &'static str, plan: &Plan<'_>, card: usize) -> Result<u64, GpuError> {
    plan.cards
        .get(card)
        .map(|c| c.stage_rows_bytes)
        .ok_or_else(|| GpuError::shape(what, format!("the plan totals no card {card}")))
}

/// The placed load after its body, which took `built`: the output head the
/// card carries, over the body's weights, and the model over all of it, with
/// the load's phases.
fn placed_done<B: ChainBody>(
    p: PlacedPre,
    body: B,
    built: Duration,
) -> Result<GpuModel<B>, GpuError> {
    let PlacedPre {
        gpu,
        weights,
        layers: _,
        head,
        ctx_max,
        mut times,
    } = p;
    let t = Instant::now();
    let head = head
        .then(|| Head::of(&gpu, &weights, body.head_eps(), body.head_norm()))
        .transpose()?;
    times.body = built;
    times.head = t.elapsed();
    let mut model = GpuModel::new(Resident {
        gpu,
        weights,
        body,
        head,
        ctx_max,
    });
    model.load_times = Some(times);
    Ok(model)
}

// -------------------------------------------------------- the capabilities

impl<B: Rows> GpuModel<B> {
    /// Run `tokens[r]` at position `pos + r`, for the model's position `pos`,
    /// as one pass of `M` rows ([`Rows::enqueue_rows`]) and return each row's
    /// greedy next token. Bit for bit `M` calls of `step(&[tokens[r]])`; the
    /// model stands `M` positions on. `M` is 2 to the body's
    /// [`Rows::MAX_ROWS`], or the call does not compile. The first pass of
    /// `M` rows makes the heads it needs ([`Rows::HEADS`]), and in graph mode
    /// captures the pass (key `M`). A caller that does not keep the later
    /// rows takes them back with [`GpuModel::rollback`]. A fault the pass
    /// raised is its error, as [`GpuModel::step`]'s is.
    ///
    /// Every caller says how many rows it keeps ([`GpuModel::keep_rows`])
    /// before the next pass: the host tier refuses the next boundary
    /// without it, with or without a residency machine.
    pub fn step_rows<const M: usize>(&mut self, tokens: [u32; M]) -> Result<[u32; M], GpuError> {
        const WHAT: &str = "GpuModel::step_rows";
        const { rows_fit::<B>(M) };
        let _busy = watchdog::busy(&self.watch, watchdog::STEP_ROWS, self.reads);
        self.refuse_if_poisoned(WHAT)?;
        self.refuse_if_slots_wait(WHAT)?;
        let pos = self.pos;
        self.check_pos(pos + crate::launch_u32(WHAT, "rows", M - 1)?, WHAT)?;
        self.make_heads(M, B::HEADS, WHAT)?;
        if self.mode == StepMode::Graph && self.graphs.get(M).is_none() {
            self.capture_rows::<M>()?;
        }
        self.body.plan_rows(self.gpu.stream(), &tokens, pos)?;
        let tokens = self.run_pass::<M>(pos);
        self.note_fault(WHAT, tokens)
    }

    /// [`GpuModel::step_rows`]'s pass at `pos` once its rows are planned:
    /// the enqueue or replay, then every row's readback, in row order. An
    /// error here can follow launches, so the caller passes it through
    /// [`GpuModel::note_fault`].
    fn run_pass<const M: usize>(&mut self, pos: u32) -> Result<[u32; M], GpuError> {
        const WHAT: &str = "GpuModel::step_rows";
        self.one_pass = (B::HEADS == RowHeads::One).then_some(M);
        self.slot_rows = None;
        let r = match self.mode {
            StepMode::Eager => self.pass_boundary().and_then(|()| {
                let GpuModel {
                    heads,
                    pass_heads,
                    body,
                    weights,
                    gpu,
                    ..
                } = self;
                let heads = pass_slice(heads, pass_heads, M, B::HEADS, WHAT)?;
                body.enqueue_rows(gpu, weights, heads)
            }),
            StepMode::Graph => self.replay(M, B::chain_of(M)),
        };
        self.name_host_refusal(r)?;
        self.stand_at(pos + crate::launch_u32(WHAT, "rows", M)?);
        self.reads += 1;
        let mut out = [0u32; M];
        match B::HEADS {
            RowHeads::PerRow => {
                let heads = self.heads.get(..M).ok_or(no_head(WHAT))?;
                for (o, head) in out.iter_mut().zip(heads) {
                    *o = head.token(&self.gpu)?;
                }
            }
            RowHeads::One => {
                let head = one_head(&self.pass_heads, self.one_pass, M, WHAT)?;
                let tokens = head.tokens(&self.gpu)?;
                if tokens.len() != M {
                    return Err(GpuError::shape(
                        WHAT,
                        format!("{} tokens from a head of {M} rows", tokens.len()),
                    ));
                }
                out.copy_from_slice(&tokens);
            }
        }
        Ok(out)
    }

    /// Capture the pass of `M` rows into its own graph over the resident
    /// buffers (key `M`), making the heads it needs first, and return its
    /// node count. The one-token step's capture stays.
    pub fn capture_rows<const M: usize>(&mut self) -> Result<usize, GpuError> {
        const WHAT: &str = "GpuModel::capture_rows";
        const { rows_fit::<B>(M) };
        self.make_heads(M, B::HEADS, WHAT)?;
        let GpuModel {
            graphs,
            heads,
            pass_heads,
            body,
            weights,
            gpu,
            ..
        } = self;
        let heads = pass_slice(heads, pass_heads, M, B::HEADS, WHAT)?;
        let graph = gpu.capture(|_| body.enqueue_rows(gpu, weights, heads))?;
        let nodes = graph.node_count();
        *graphs.slot(M, WHAT)? = Some(graph);
        Ok(nodes)
    }

    /// Every node of the captured pass of `M` rows, as the driver lists them.
    pub fn rows_graph_nodes<const M: usize>(&self) -> Result<Vec<NodeInfo>, GpuError> {
        const { rows_fit::<B>(M) };
        self.graphs
            .get(M)
            .ok_or(GpuError::state(
                "GpuModel::rows_graph_nodes",
                "no captured pass of these rows",
            ))?
            .nodes()
    }

    /// Each row's logits of the last [`GpuModel::step_rows`] of `M` rows
    /// (`n_vocab` f32 each), each through [`GpuModel::pass_row_into`], the
    /// read a sampled pass takes its rows by. Blocking read; gate/debug use.
    pub fn rows_logits<const M: usize>(&self) -> Result<[Vec<f32>; M], GpuError> {
        const { rows_fit::<B>(M) };
        let n_vocab = self.n_vocab()?;
        let mut pass = Vec::new();
        let mut out: [Vec<f32>; M] = std::array::from_fn(|_| vec![0.0; n_vocab]);
        for (r, o) in out.iter_mut().enumerate() {
            self.pass_row_into(M, r, &mut pass, r > 0, o)?;
        }
        Ok(out)
    }

    /// Row `r`'s logits of the last [`GpuModel::step_rows`] of `m` rows into
    /// `row` (`n_vocab` f32), the read a sampled pass takes each row by:
    /// [`RowHeads::PerRow`] copies head `r` alone; [`RowHeads::One`] copies
    /// the pass's one head into `pass` (grown to `m · n_vocab` f32 once, then
    /// reused) unless `fetched` says this pass's head is there already, and
    /// takes row `r` from it (`[v·m + r]`). Blocking read. Refused by name
    /// for a row past `m`, a `row` of another length, and under
    /// [`RowHeads::One`] when the last call was not a pass of `m` rows.
    pub fn pass_row_into(
        &self,
        m: usize,
        r: usize,
        pass: &mut Vec<f32>,
        fetched: bool,
        row: &mut [f32],
    ) -> Result<(), GpuError> {
        let mut read = PassRead {
            layout: B::HEADS,
            rows: m,
            stage: pass,
            fetched,
        };
        self.pass_row("GpuModel::pass_row_into", &mut read, r, row)
    }

    /// Row `r` of the last [`GpuModel::step_rows`] of `m` rows read as its
    /// best logits into `row` ([`best::ROW_WORDS`] words): as
    /// [`GpuModel::pass_row_into`], the pass's rows read once into `stage`
    /// unless `fetched` says they are there ([`GpuModel::pass_best`]).
    pub fn pass_best_into(
        &mut self,
        m: usize,
        r: usize,
        stage: &mut Vec<u32>,
        fetched: bool,
        row: &mut [u32],
    ) -> Result<(), GpuError> {
        let mut read = PassRead {
            layout: B::HEADS,
            rows: m,
            stage,
            fetched,
        };
        self.pass_best("GpuModel::pass_best_into", &mut read, r, row)
    }
}

/// The read of one pass's rows ([`GpuModel::pass_row`], and as their best
/// logits [`GpuModel::pass_best`]): the pass's `rows`, the `layout` its heads
/// lay them in, and under [`RowHeads::One`] the host copy of its one head
/// (f32), or of its rows' best words (u32) — `stage`, which the first row
/// read fetches (`fetched` from then on) and the rows after it take theirs
/// from.
struct PassRead<'a, T = f32> {
    layout: RowHeads,
    rows: usize,
    stage: &'a mut Vec<T>,
    fetched: bool,
}

/// The [`RowHeads::One`] head of the pass of `m` rows the last call ran:
/// refused by name, as `what`, when the last call (`one_pass`) was another,
/// or no pass of these rows has run.
fn one_head<'a>(
    pass_heads: &'a [Option<Head>],
    one_pass: Option<usize>,
    m: usize,
    what: &'static str,
) -> Result<&'a Head, GpuError> {
    if one_pass != Some(m) {
        return Err(GpuError::state(
            what,
            "the last call was not a pass of these rows",
        ));
    }
    pass_heads
        .get(m.wrapping_sub(1))
        .and_then(Option::as_ref)
        .ok_or(GpuError::state(what, "no pass of these rows has run"))
}

/// The heads a pass of `m` rows laid as `layout` writes: `heads[..m]`, or
/// the one head of `m` rows in `pass` for [`RowHeads::One`]; refused by
/// name before they are made.
fn pass_slice<'a>(
    heads: &'a mut [Head],
    pass: &'a mut [Option<Head>],
    m: usize,
    layout: RowHeads,
    what: &'static str,
) -> Result<&'a mut [Head], GpuError> {
    match layout {
        RowHeads::PerRow => heads.get_mut(..m).ok_or(no_head(what)),
        RowHeads::One => pass
            .get_mut(m.wrapping_sub(1))
            .and_then(Option::as_mut)
            .map(std::slice::from_mut)
            .ok_or(no_head(what)),
    }
}

/// Holds when a pass of `m` rows is one `B` takes: 2 to [`Rows::MAX_ROWS`],
/// which is at most [`MAX_PASS_ROWS`]. Evaluated at compile time by every
/// method of [`Rows`]'s skeleton.
const fn rows_fit<B: Rows>(m: usize) {
    assert!(
        B::MAX_ROWS <= MAX_PASS_ROWS,
        "a body's Rows::MAX_ROWS is at most MAX_PASS_ROWS"
    );
    assert!(
        m >= 2 && m <= B::MAX_ROWS,
        "a pass of this many rows is not one this body takes (Rows::MAX_ROWS)"
    );
}

impl<B: Rollback> GpuModel<B> {
    /// Take back the positions from `pos` on ([`Rollback::rollback`]): the
    /// next step runs at `pos`. A step launched ahead is read first
    /// ([`GpuModel::settle_ahead`]): its fault is the call's error.
    pub fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        let _busy = watchdog::busy(&self.watch, watchdog::ROLLBACK, self.reads);
        self.refuse_if_slots_wait("GpuModel::rollback")?;
        self.settle_ahead("GpuModel::rollback")?;
        if pos > self.pos {
            return Err(GpuError::shape(
                "GpuModel::rollback",
                format!("back to position {pos} from {}", self.pos),
            ));
        }
        self.body.rollback_on(&self.gpu, pos)?;
        self.stand_at(pos);
        Ok(())
    }
}

impl<B: Instrumented> GpuModel<B> {
    /// Fill the first `rows` cache rows of every resident layer and stand at
    /// position `rows` — the state a prompt of `rows` tokens leaves behind,
    /// without decoding one. A prepared cache buys the step shape of a deep
    /// prompt for a copy. This is an instrument: the rows are not what the
    /// model would have written, so the tokens that come out are meaningless.
    ///
    /// It is a step-shape equivalence, not a value one. The step's cost does
    /// not depend on the key values — no kernel branches on them, and the
    /// one value-dependent guard drops keys past the causal limit rather
    /// than reading their size.
    ///
    /// Synchronizes; never inside a capture.
    pub fn seed_depth(&mut self, rows: usize) -> Result<(), GpuError> {
        let _busy = watchdog::busy(&self.watch, watchdog::SEED_DEPTH, self.reads);
        if rows == 0 {
            return Err(GpuError::shape(
                "GpuModel::seed_depth",
                "rows must be at least 1",
            ));
        }
        if rows >= self.ctx_max {
            return Err(GpuError::shape(
                "GpuModel::seed_depth",
                format!(
                    "{rows} seeded rows leave no room for a step in the \
                 resident cache's {} rows",
                    self.ctx_max
                ),
            ));
        }
        let pos = crate::launch_u32("GpuModel::seed_depth", "rows", rows)?;
        self.settle_ahead("GpuModel::seed_depth")?;
        self.body.seed_depth(&self.gpu, rows)?;
        self.stand_at(pos);
        Ok(())
    }
}

impl<B: Probed> GpuModel<B> {
    /// Arm (or disarm) the node-price probe. Like [`GpuModel::set_mode`] this
    /// drops every captured chain — the parked slots' with the live one's —
    /// since the probe changes which launches the
    /// body issues. A probe with either lever set makes the chain a timing
    /// instrument: `skip_quant` leaves activation buffers unwritten, so the
    /// tokens that come out are not the model's answer.
    pub fn set_probe(&mut self, probe: StepProbe) -> Result<(), GpuError> {
        self.body.set_probe(probe)?;
        self.drop_captures();
        Ok(())
    }
}

/// The refusal of a call that needs the output head on a load without it.
fn no_head(what: &'static str) -> GpuError {
    GpuError::state(what, "no output head: this load carries none")
}

/// `block_count` of `file`: the layer count.
pub(crate) fn block_count(file: &Split, what: &'static str) -> Result<usize, GpuError> {
    let n = file
        .arch_get_u64("block_count")
        .ok_or(GpuError::metadata(what, "block_count"))?;
    usize::try_from(n).map_err(|_| GpuError::shape(what, format!("block_count {n}")))
}

/// Refused by name unless the load of the checkpoint folder whose header is `file` is whole: every
/// layer of `0..n_layers` with the head, which the folder holds as a tensor of its own (a head tied
/// to the token embedding is a file's: the part load does not alias it).
fn refuse_partial_folder(
    file: &Split,
    layers: &Range<usize>,
    n_layers: usize,
    head: bool,
) -> Result<(), GpuError> {
    let what = "GpuModel::load_blocks";
    if *layers != (0..n_layers) || !head {
        return Err(GpuError::shape(
            what,
            format!(
                "a checkpoint folder loads the whole model: layers {layers:?} of 0..{n_layers} \
                 with head {head} are a file's"
            ),
        ));
    }
    if head_tensor(file) != HEAD_TENSOR {
        return Err(GpuError::shape(
            what,
            format!(
                "a checkpoint folder with no {HEAD_TENSOR}: a head tied to the token embedding \
                 is a file's"
            ),
        ));
    }
    Ok(())
}

/// The reader of `file`'s one shard, for a caller that reads a single file;
/// a split set of several shards is refused.
pub(crate) fn one_shard<'a>(
    file: &'a Split,
    what: &'static str,
) -> Result<&'a gguf::Gguf, GpuError> {
    match (file.shard_count(), file.shard(0)) {
        (1, Some(g)) => Ok(g),
        (n, _) => Err(GpuError::shape(
            what,
            format!("the file is {n} shards, and this reads one"),
        )),
    }
}

impl<B: Instrumented> Engine for GpuModel<B> {
    fn step(&mut self, tokens: &[u32]) -> Result<u32, GpuError> {
        GpuModel::step(self, tokens)
    }

    fn reset(&mut self) -> Result<(), GpuError> {
        GpuModel::reset(self)
    }

    fn seed_depth(&mut self, rows: usize) -> Result<(), GpuError> {
        GpuModel::seed_depth(self, rows)
    }

    fn pos(&self) -> u32 {
        GpuModel::pos(self)
    }

    fn resident_bytes(&self) -> usize {
        GpuModel::resident_bytes(self)
    }

    fn arch(&self) -> Arch {
        B::arch()
    }
}

#[cfg(test)]
mod tests {
    use super::{beside, refuse_partial_folder, refuse_reserve_past};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    /// Buffers at the plan's reserve pass; one byte past it is refused by name with both figures
    /// and what the buffers are. Mutant: the comparison read `>=`, or reversed.
    #[test]
    fn made_bytes_past_the_reserve_are_refused_by_name() {
        refuse_reserve_past("w", "pass buffers", 100, 100, "the rows").expect("at the reserve");
        let err = refuse_reserve_past("w", "pass buffers", 100, 101, "the rows")
            .expect_err("one byte past the reserve")
            .to_string();
        assert_eq!(
            err,
            "w: the load's pass buffers hold 101 device bytes (the rows); the plan reserves 100 \
             for them"
        );
    }

    /// A reserve-only member after the stage card (the card a draft sits on) is no expert tier: a
    /// one-card run passes with it and keeps the stage card's layers and the context. Mutant: the
    /// check read `machine.tiers.is_empty()`.
    #[test]
    fn a_reserve_only_member_passes_the_one_card_run() {
        use model::placement::workstation::{
            A6000, NO_TIER_BATCH, RTX_3090, draft_reserve, plan_holding,
        };
        use model::placement::{Device, KvBytes, ModelTensors, plan_with};
        struct NoKv;
        impl KvBytes for NoKv {
            fn layer_bytes(&self, _: usize, _: u64) -> u64 {
                0
            }
        }
        let model = ModelTensors {
            tensors: Vec::new(),
            layers: 3,
            experts: 8,
            experts_used: 2,
            bf16_words: |_| false,
        };
        let held = plan_holding(
            3,
            A6000,
            &[],
            &[RTX_3090],
            Some(draft_reserve(Device::Card(1), 7)),
            NO_TIER_BATCH,
        )
        .expect("a holder after the stage card");
        let plan = plan_with(&model, &held, 16, &NoKv, None).expect("the holder's plan");
        let run = super::one_card_run("t", &plan, 0, 3).expect("a holder is no expert tier");
        assert_eq!(run, (0..3, 16));
    }

    /// A folder's part load reads every layer and a head of its own: a part of the layers, no
    /// head, and a head tied to the token embedding are each refused by name, and the whole model
    /// with `output.weight` passes. Mutant: the tied-head comparison dropped, or the layer range
    /// compared with `<`.
    #[test]
    fn a_folder_loads_every_layer_and_a_head_of_its_own() {
        use gguf::Split;
        use gguf::write::{Layout, TensorDecl};
        let header = |names: &[&str]| {
            let tensors = names
                .iter()
                .map(|name| TensorDecl {
                    name: (*name).into(),
                    dims: vec![32],
                    type_id: gguf::GgmlType::F32.as_u32(),
                    nbytes: 128,
                })
                .collect();
            let layout = Layout::new(&[], tensors).expect("a layout");
            Split::header_only(std::env::temp_dir().join("model-folder-toy.gguf"), &layout)
                .expect("a header")
        };
        let own = header(&["token_embd.weight", "output.weight"]);
        let tied = header(&["token_embd.weight"]);
        refuse_partial_folder(&own, &(0..4), 4, true).expect("every layer and its own head");
        for (file, layers, head, want) in [
            (&own, 0..3, true, "layers 0..3 of 0..4"),
            (&own, 0..4, false, "with head false"),
            (&tied, 0..4, true, "a head tied to the token embedding"),
        ] {
            let e = refuse_partial_folder(file, &layers, 4, head)
                .expect_err(want)
                .to_string();
            assert!(e.contains(want), "{e}");
        }
    }
    /// The background of a load runs while its foreground does: the foreground waits for a
    /// flag only the background sets, so a load that ran them one after the other would time
    /// the wait out. Both results come back, the foreground's first. Mutant: `beside` runs
    /// the foreground before it starts the helper.
    #[test]
    fn the_host_set_walk_runs_beside_the_card_side() {
        let started = AtomicBool::new(false);
        let (seen, background, waited) = beside(
            "beside",
            || {
                started.store(true, Ordering::SeqCst);
                Ok(7u32)
            },
            || {
                let until = Instant::now() + Duration::from_secs(10);
                while !started.load(Ordering::SeqCst) {
                    if Instant::now() >= until {
                        return false;
                    }
                    std::thread::yield_now();
                }
                true
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert!(
            seen,
            "the background had not started while the foreground ran"
        );
        assert_eq!(background.unwrap_or_else(|e| panic!("{e}")), 7);
        assert!(waited < Duration::from_secs(10), "{waited:?}");
    }
}
