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

pub(crate) mod kernels;
pub(crate) mod lookup;
pub(crate) mod probe;
pub(crate) mod slots;

pub use kernels::{Q8_0GemvHeadsArgs, StepKernels};
pub(crate) use lookup::f32_gain;
pub use probe::{OpTime, StepProbe};
pub use slots::{SlotRange, SlotRows, Slots, SlotsOut};

use slots::{ParkedSlot, PoisonWhy, SlotFault, SlotGraphs, SlotSet};

use crate::fault::Fault;
use crate::head::{Head, HeadNorm};
use crate::host::swap::{BoundaryAt, MachineCfg, Residency};
use crate::host::swap_source::{FileSwap, ResidencyGlue, ResidencySpec};
use crate::hybrid::{Chain, HostResidency, Refusal, name_refusal};
use crate::weights::Weights;
use crate::{Gpu, GpuError, Graph, NodeInfo};
use bloomery_levers::HostCfg;
use cuda_core::CudaStream;
use gguf::Split;
use model::arch::Arch;
use model::placement::churn::ChurnPool;
use model::placement::workstation::{self, HostNeed};
use model::placement::{ExpertList, Plan};
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

    /// The architecture this body is the chain of.
    fn arch() -> Arch;

    /// The input of one decode step at `(token, pos)`. A body whose next step
    /// needs work issued ahead of it does that here, not in `refresh`.
    fn decode_input(&mut self, token: u32, pos: u32) -> Result<Self::Input, GpuError>;

    /// Write `input` into the device buffers the captured graph reads, so one
    /// capture serves every position. Runs before an eager enqueue or a replay
    /// and never inside a capture.
    fn refresh(&mut self, stream: &CudaStream, input: &Self::Input) -> Result<(), GpuError>;

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

    /// Stop the residency machine before anything of the model is freed
    /// ([`crate::host::HostTier::stop_swap`]): the model's drop calls it
    /// first. A body with no machine does nothing.
    fn stop_residency(&mut self) {}

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
    /// lock when the load ran them.
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
/// Fields drop in declaration order, after the drop has stopped the body's
/// residency machine, and a graph must be destroyed while
/// every buffer it addresses is still alive: the captured chains first — the
/// live slot's, the passes of several slots', then each parked slot's
/// ([`ParkedSlot`]: its graph cache before its sequence) — then the heads,
/// the body (which drops its own captures first, and its host tier's lock
/// over the host set before the mappings under it), the weights they all
/// address, and the card last.
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
}

impl<B: ChainBody> Drop for GpuModel<B> {
    /// The residency machine stops first ([`HostServed::stop_residency`]):
    /// the chains, the heads and the body free card memory as they drop, and
    /// a free waits for a copy the machine left queued behind a staging word.
    fn drop(&mut self) {
        if let Some(host) = self.body.host() {
            host.stop_residency();
        }
    }
}

impl<B: ChainBody> GpuModel<B> {
    /// The model over `r`, standing at position 0 in graph mode with nothing
    /// captured.
    pub fn new(r: Resident<B>) -> GpuModel<B> {
        let Resident {
            gpu,
            weights,
            body,
            head,
            ctx_max,
        } = r;
        GpuModel {
            graphs: Graphs::new(),
            slot_graphs: SlotGraphs::new(),
            parked: Vec::new(),
            heads: head.into_iter().collect(),
            pass_heads: (0..MAX_PASS_ROWS).map(|_| None).collect(),
            one_pass: None,
            slot_rows: None,
            slots_waiting: None,
            body: Box::new(body),
            weights,
            gpu,
            ctx_max,
            mode: StepMode::Graph,
            selected: 0,
            pos: 0,
            poisoned: None,
            load_times: None,
            reads: 0,
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
        let mut weights = Weights::load(gpu.stream(), file, layers, true)?;
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
        let (p, residency) = placed_pre(&file, plan, card, host, (&[], 0), derive)?;
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
        const WHAT: &str = "GpuModel::load_placed_with";
        let churn = match residency.lever {
            Residency::Off => None,
            Residency::Mid { pinned, .. } => {
                Some(ChurnPool::of(plan, card, pinned).map_err(|e| GpuError::plan(WHAT, e))?)
            }
        };
        // The host set holds `hosted` beside the plan's own segments, out of
        // the same headroom the pool must fit.
        if let Some(p) = &churn {
            p.check_beside(plan, hosted.1)
                .map_err(|e| GpuError::plan(WHAT, e))?;
        }
        let pool = churn
            .as_ref()
            .map_or((&[][..], 0), |p| (p.runs.as_slice(), p.bytes));
        let joined: Vec<(usize, ExpertList)>;
        let extra = if hosted.0.is_empty() {
            pool
        } else {
            joined = pool.0.iter().chain(hosted.0).cloned().collect();
            (&joined[..], pool.1 + hosted.1)
        };
        let file = Arc::new(file);
        let (p, set) = placed_pre(&file, plan, card, host, extra, derive)?;
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
    /// A residency machine's map stays where use has taken it
    /// ([`GpuModel::residency_reset`] is the explicit call).
    pub fn reset(&mut self) -> Result<(), GpuError> {
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
    /// tokens is P bodies and one sync.
    ///
    /// Positions continue from wherever the model stands: a prompt then its
    /// continuation is `step(&prompt)` followed by one `step(&[tok])` per
    /// generated token. [`GpuModel::reset`] rewinds. A fault any token
    /// raised is the call's error, also when a later token then fails for
    /// another reason ([`GpuModel::note_fault`]). A model that carries the
    /// head carries every layer: each constructor that makes the head loads
    /// the whole chain.
    pub fn step(&mut self, tokens: &[u32]) -> Result<u32, GpuError> {
        self.refuse_if_poisoned("GpuModel::step")?;
        self.refuse_if_slots_wait("GpuModel::step")?;
        if tokens.is_empty() {
            return Err(GpuError::shape("GpuModel::step", "empty token slice"));
        }
        if self.heads.is_empty() {
            return Err(no_head("GpuModel::step"));
        }
        if self.mode == StepMode::Graph && self.graphs.get(1).is_none() {
            self.capture_step()?;
        }
        let token = self.run_tokens(tokens);
        self.note_fault("GpuModel::step", token)
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
        }
        self.reads += 1;
        self.heads
            .first()
            .ok_or(no_head("GpuModel::step"))?
            .token(&self.gpu)
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
    pub fn pass_boundary(&mut self) -> Result<(), GpuError> {
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
        let GpuModel { body, gpu, .. } = self;
        match body.host() {
            Some(host) => host.residency_reset(gpu.stream()),
            None => Ok(None),
        }
    }

    /// A step's, pass's or prompt call's result, with a host refusal named:
    /// when the body's host service refused input the card should already
    /// have refused, it released every wait of the step, so the call drains,
    /// and the fault word read on the engine stream behind it names the
    /// refusal ([`name_refusal`]) — the card's fault when the card raised one
    /// at or before that layer, else the host's error. When the word holds a
    /// later layer's fault, the caller's [`GpuModel::note_fault`] returns
    /// that fault with the host's error behind it, so the call ends poisoned
    /// and still names the refusal's layer, which came first. A failed read
    /// names the refusal and the call's error beside its own. Any other
    /// result passes through.
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
        const WHAT: &str = "GpuModel::name_host_refusal";
        let e = match r {
            Err(e) => e,
            Ok(v) => return Ok(v),
        };
        let Some(mut refusal) = self.body.host().and_then(HostServed::take_host_refusal) else {
            return Err(e);
        };
        name(&mut refusal);
        if !self.parked.is_empty() && self.poisoned.is_none() {
            self.poisoned = Some(SlotFault {
                slots,
                why: PoisonWhy::Refused(refusal.clone()),
            });
        }
        match self.gpu.fault() {
            Ok(word) => Err(name_refusal(&refusal, word)),
            Err(s) => Err(GpuError::shape(
                WHAT,
                format!(
                    "{} refused the input of layer {} ({}) and the call failed ({e}), and reading \
                     the fault word behind it failed too ({s})",
                    refusal.what, refusal.layer, refusal.detail
                ),
            )),
        }
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

    /// The [`RowHeads::One`] head of `m` rows, once a pass made it.
    fn pass_head(&self, m: usize, what: &'static str) -> Result<&Head, GpuError> {
        self.pass_heads
            .get(m.wrapping_sub(1))
            .and_then(Option::as_ref)
            .ok_or(GpuError::state(what, "no pass of these rows has run"))
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
    /// call's error whatever else failed: any other error is read behind
    /// ([`GpuModel::fault_behind`]). The fault prints its site mask in this
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
        let mut e = match r {
            Ok(v) => return Ok(v),
            Err(e @ GpuError::Fault { .. }) => e,
            // The stream did not drain within the bound: the fault word's
            // read would wait it out again.
            Err(e @ GpuError::Stalled { .. }) => match self.body.host() {
                Some(h) => h.noted(e),
                None => e,
            },
            Err(e) => self.fault_behind(what, e),
        };
        if let GpuError::Fault { fault, .. } = &mut e {
            *fault = fault.in_arch(B::arch());
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
    ///   call's end; a fault an earlier pass raised is named there, and the
    ///   passes between it and the end ran on the condemned state.
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
/// host's share of the replay, as `chain`, when the body has a host service:
/// the only place a captured chain or pass is replayed (`GpuModel::replay`,
/// a pass of several slots). The host service's residency boundary runs
/// first, stamped with `reads` ([`HostServed::at_boundary`] at
/// [`BoundaryAt::Launch`]): the caller has looked `graph` up, so the
/// boundary runs once the chain is known to exist.
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

/// The steps of a placed load before its body: the plan's card and context
/// budget, the card, the host's memory against the plan's host need, the
/// card's weights with `derive`'s filings, and — after the uploads, so the
/// card's file bytes have left the page cache first — the plan's host set
/// read in and locked as `host` asks, with `extra`'s runs (of `extra`'s
/// bytes) beside it. The pieces, and the host set. Every placed load passes
/// here, so the two refusals before any upload are made once: a plan's card
/// that is not one visible device ([`Gpu::open_card`]), and a
/// host whose `MemAvailable` is under the plan's need ([`HostNeed`]).
fn placed_pre(
    file: &Split,
    plan: &Plan<'_>,
    card: usize,
    host: HostCfg,
    extra: (&[(usize, ExpertList)], u64),
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
    let t = Instant::now();
    let gpu = Gpu::open_card(&spec.name, spec.device)?;
    let context = t.elapsed();
    let available = workstation::host_available().map_err(|e| GpuError::plan(what, e))?;
    HostNeed::of(plan, extra_bytes)
        .check(available)
        .map_err(|e| GpuError::plan(what, e))?;
    let t = Instant::now();
    let mut weights = Weights::load_placed(gpu.stream(), file, plan, card, host.card_dontneed)?;
    let upload = t.elapsed();
    let t = Instant::now();
    derive(gpu.stream(), file, layers.clone(), &mut weights)?;
    let derived = t.elapsed();
    let t = Instant::now();
    let residency = HostResidency::at_load_with(file, plan, |_| true, host, extra)?;
    let host_set = t.elapsed();
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
    /// before the next pass: a body with a residency machine refuses the
    /// next boundary without it.
    pub fn step_rows<const M: usize>(&mut self, tokens: [u32; M]) -> Result<[u32; M], GpuError> {
        const WHAT: &str = "GpuModel::step_rows";
        const { rows_fit::<B>(M) };
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
                let head = self.pass_head(M, WHAT)?;
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
        const WHAT: &str = "GpuModel::pass_row_into";
        if r >= m {
            return Err(GpuError::shape(WHAT, format!("row {r} of a pass of {m}")));
        }
        match B::HEADS {
            RowHeads::PerRow => self
                .heads
                .get(r)
                .ok_or(GpuError::state(WHAT, "no pass of these rows has run"))?
                .logits_into_host(&self.gpu, row),
            RowHeads::One => {
                if self.one_pass != Some(m) {
                    return Err(GpuError::state(
                        WHAT,
                        "the last call was not a pass of these rows",
                    ));
                }
                let head = self.pass_head(m, WHAT)?;
                let n_vocab = head.n_vocab();
                if row.len() != n_vocab {
                    return Err(GpuError::shape(
                        WHAT,
                        format!("a row of {} f32 for {n_vocab} logits", row.len()),
                    ));
                }
                if !fetched {
                    pass.resize(m * n_vocab, 0.0);
                    head.logits_into_host(&self.gpu, pass)?;
                } else if pass.len() != m * n_vocab {
                    return Err(GpuError::shape(
                        WHAT,
                        format!(
                            "a fetched pass of {} f32 for {m} rows of {n_vocab}",
                            pass.len()
                        ),
                    ));
                }
                for (o, &v) in row.iter_mut().zip(pass.iter().skip(r).step_by(m)) {
                    *o = v;
                }
                Ok(())
            }
        }
    }
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
    /// next step runs at `pos`.
    pub fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        self.refuse_if_slots_wait("GpuModel::rollback")?;
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
