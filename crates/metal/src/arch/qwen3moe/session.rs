//! A qwen35moe session on Metal: the file opened into its plan, its arena and
//! its pipelines, then one token a step.
//!
//! A host-fed step ([`StepMode::HostFed`]) writes the token id into the host
//! id slot, encodes the token on one command buffer, commits it, waits for
//! it, and reads the token's fault word, then the head's pick. While it
//! waits it encodes the next position's host-fed token and holds it off the
//! queue ([`Parked`]): the id is the host's to pick from this token's logits,
//! and the token reads it from the slot at run time, so the next host-fed
//! step writes the slot and commits that buffer instead of encoding one.
//! The encode is the same whenever it runs: a host-fed token's commands
//! depend on its position and the stream's boundary alone. A parked token
//! that another signalled buffer has passed on the stream, or whose position
//! is not the step's, is dropped unrun. A chain step
//! ([`StepMode::Chain`]) also commits the next token before it waits, its
//! embed reading the pick on the device: the token runs ahead of the call
//! that names it. The next call keeps it when its id is that pick; any other
//! id breaks the promise, and the token is waited out and dropped, its fault
//! word cleared if it raised, and replaced by a host-fed token. Nothing is
//! restored: the dropped token wrote only what its replacement writes again
//! (its record slot, its cache row, its conv ring row and the delta lane it
//! writes, not the one it reads). A chain step at the cache's last row
//! commits nothing ahead.
//!
//! An ahead step ([`StepMode::Ahead`]) is a chain step whose ids are known:
//! the ids of a prompt. The host writes the next id into this token's record
//! slot — the slot whose pick a chain's head would write — and the token
//! ahead, committed before this one is waited, embeds it from the device; no
//! token the mode commits carries a head, for no pick of the run's is read
//! but the prompt's last row's, and a headless step returns the id it was
//! fed, reading no pick back. A call keeps the token ahead only when the two
//! agree on the head: a pick-reading call cannot land a headless token, and
//! the head of a token a chain committed would overwrite the slot the host
//! writes, so each drops the other's token and recomputes its own.
//!
//! A raised fault poisons the session: the step waits out the token ahead,
//! returns the fault by name, decoded to its layer and sites, and every later
//! step is refused by name until [`Session::reset`].
//!
//! A checkpoint copies each delta layer's state lane at the position's parity
//! and its whole conv ring, and each short conv layer's whole ring.
//! [`Session::checkpoint`] takes one at the position the session stands at,
//! with no token ahead and the stream idle; [`Session::checkpoint_copy`] takes
//! one at a position inside the last pass, from the copies that pass left
//! ([`StateOut::after_rows`]).
//! [`Session::cut`] back to a point copies those lanes and rings back and
//! stands there; a cut to 0 is a reset, which drops the points too. A cut the
//! ledger refuses leaves the position and every store as they were. A file
//! with a recurrent store of either kind has checkpoints; only a file with
//! none (its caches hold every position) has none.
//!
//! [`Session::pass`] runs more than a step's window of ids as one pass: every
//! layer once over all the rows, its GEMMs and row kernels over the rows
//! arena, which the open makes beside the token's. The pass is the token at
//! its last position as far as the record goes: its faults land in, and its
//! head writes its pick to, the step record slot of its last row, and the
//! delta state it leaves is the lane a step at the position after it reads. A
//! pass is a command buffer a layer, each opened at the token boundary and
//! only the last one waited: there is no host wait inside the pass, and none
//! past the last layer's. A prompt call cuts its ids into passes
//! at its checkpoint marks ([`Session::prefill`]). A pass may also leave
//! copies of the delta layers' lanes and rings after rows inside it: each is
//! the state the pass held after that row, and the pass's own bits do not
//! move. A file with a short conv layer takes no such copies: a pass that asks
//! for one is refused by name.
//!
//! Every buffer the kernels bind is registered into the stream's residency
//! set when the session allocates it, and the open waits for the set to be
//! requested resident ([`crate::residency`]): the first token and the first
//! pass find the weights and the rows arena wired, and a token after a pass
//! or an idle gap finds them still wired. [`Session::residency`] reports the
//! set, and [`Session::open_split`] how long the open's parts took.
//!
//! [`Session::step_traced`] is the instrument the gates read nodes through:
//! the same ops of a host-fed step, each on a command buffer of its own,
//! waited, with a hook after each. It encodes what [`Session::step`]
//! encodes, op for op. [`Session::step_profile`] is the finer instrument:
//! each group on a command buffer of its own, its GPU seconds read, one
//! sample a group. The step clock ([`Session::start_step_clock`]) is the
//! coarser one: each settled token's encode, commit-to-start, GPU and wake
//! seconds, a row a mode, for a profiling run to average and print.

use core::fmt;
use core::ops::Range;
use core::sync::atomic::Ordering;
use core::time::Duration;
use std::path::Path;
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use bloomery_device::fault::Fault;
use bloomery_device::{Buffer, DeviceError, Elem, Elems, Stream};
use chain::ChainError;
use chain::arch::qwen3moe::ops::{ChainOps, LayerCtx};
use chain::arch::qwen3moe::plan::{
    DeltaPlan, FfnPlan, FfnRoute, GqaPlan, LayerPlan, ResidualPlan, ShortConvPlan, SiteTy,
};
use chain::arch::qwen3moe::read::Sites35;
use chain::arch::qwen3moe::walk::Walk;
use model_hf::ModelFile;
use runtime::prompt::PrefillMode;
use runtime::seqstate::Kept;

use super::arena::{Arena, LANES, LayerBufs, MixerBufs, lane_of, slot_of};
use super::check;
use super::count::{GroupBytes, GroupKind, Layer};
use super::ops::{Feed, Half, MetalOps, RowsCtx};
use super::rows::{RowsArena, SnapBufs};
use crate::buffer::MetalBuffer;
use crate::checkpoint::{self, Checkpoints, Restore};
use crate::device::MetalDevice;
use crate::encode::EncodeCount;
use crate::geom::Geometry;
use crate::handle::Handle;
use crate::instance::Instance;
use crate::kernels::{Kernels, PASS_ROWS, SNAPS};
use crate::library::Library;
use crate::residency::{Alloc, ResidencyStats};
use crate::stream::{
    GpuInterval, GpuStamps, MetalStream, Parked, Settled, Signalled, TokenRecording, host_secs,
};
use objc2_metal::MTLDevice;

pub(super) const WHAT: &str = "qwen35moe session";

/// The most rows of one pass, and the spacing of a prompt call's inner
/// checkpoints ([`runtime::seqstate::marks`]): a mark falls between two
/// passes, so a pass never holds a position a checkpoint needs. The family's
/// ubatch rule at an instance's slots a row over [`max_pass_slots`] slots
/// ([`pass_rows_max`]).
///
/// A point copies each delta layer's state lane and its conv ring.
/// Qwen3.6-35B-A3B has 30 delta layers of 16 key heads and 32 value heads:
/// a lane is 32·128·128 f32s (2_097_152 bytes) and a ring is
/// 11·(2·16 + 32)·128 f32s (360_448 bytes), so a point is
/// 30·(2_097_152 + 360_448) = 73_728_000 bytes. One copy every 2048
/// positions is 36_000 bytes beside each position of the prompt, and a cut
/// between points re-feeds at most 2047 positions.
pub const CHECKPOINT_EVERY: u32 = 2048;

/// The slots of one pass of `g`'s instance the rows arena holds: its experts'
/// buffers are a row a slot.
fn max_pass_slots(g: &Geometry) -> usize {
    g.experts.slots * CHECKPOINT_EVERY as usize
}

/// The most rows of one pass of `g`'s instance, by the family's ubatch rule.
#[must_use]
pub fn pass_rows_max(g: &Geometry) -> usize {
    chain::arch::qwen3moe::router::ubatch_at(g.experts.slots, max_pass_slots(g))
}

/// How a session of `g`'s instance feeds a prompt call when nothing sets it:
/// in passes when the rows arm's launches take the instance's slots, a step
/// each otherwise. A dense FFN's one slot has its own rows arm (the dense
/// GEMMs), so an instance of one always takes passes. A routed instance's
/// slots are the router's picks and the shared expert's slot exactly when the
/// router has a shared row, and its route table has a group for each routed
/// expert and one for the shared expert when there is one; the instances the
/// arm runs are those that agree on both, a layer with a shared expert and one
/// without alike.
fn prefill_at_open(g: &Geometry) -> PrefillMode {
    let (router, experts) = (&g.router, &g.experts);
    let shared = router.shared == 1;
    if experts.dense
        || (experts.has_shared == shared
            && experts.slots == experts.routed_ids + router.shared
            && router.rows == router.experts + router.shared)
    {
        PrefillMode::Batch
    } else {
        PrefillMode::Steps
    }
}

/// The site types Metal's mat-vecs launch: the kernels' quant set
/// ([`Quant::of_ggml`](crate::kernels::Quant::of_ggml) read over
/// [`Quant::dense`](crate::kernels::Quant::dense), held equal to this list by
/// `arena`'s test) as families — Q4_K, Q6_K, Q8_0, Q5_K and MLX_A4, the A4
/// kernels' affine 4-bit groups. The F32 tensors a session reads as
/// values (the norms, the conv, the dt bias, `ssm_a`) and the router's
/// MLX_A8, which its loader dequantizes exactly at open, are no typed site; a
/// type outside the list is refused by name at the site (`file_site`).
pub(crate) const LAUNCHED: &[SiteTy] = &[
    SiteTy::Q4K,
    SiteTy::Q6K,
    SiteTy::MlxA4,
    SiteTy::Q8_0,
    SiteTy::Q5K,
];

/// The types a layer's projections may be: [`LAUNCHED`] and F32, which the
/// segmented mat-vec reads (the delta net's `ssm_alpha` and `ssm_beta`). Only
/// a P1 site runs F32; the arena's checks refuse it at every other site by
/// name ([`Quant::dense`](crate::kernels::Quant::dense)).
pub(crate) const PROJECTED: &[SiteTy] = &[
    SiteTy::Q4K,
    SiteTy::Q6K,
    SiteTy::MlxA4,
    SiteTy::Q8_0,
    SiteTy::Q5K,
    SiteTy::F32,
];

/// The site lists the plan reads: [`PROJECTED`] for the projections and for
/// the stacks list, which also holds β and α (the routed stacks and the
/// shared parts it names are held to the dispatch lists below as well), every
/// other list [`LAUNCHED`].
pub const SITES: Sites35 = Sites35 {
    proj: PROJECTED,
    dense: PROJECTED,
    stacks: PROJECTED,
    gate_up: LAUNCHED,
    down: LAUNCHED,
    shared: LAUNCHED,
};

/// How long a profiled group's stamps wait for the completed status its
/// settle already implied: the buffer's signal fires before the driver
/// reports it complete, so the stamps give the report a moment.
const STAMP_WAIT: Duration = Duration::from_secs(1);

/// A session's refusal or failure.
#[derive(Debug)]
pub enum SessionError {
    /// The device, a buffer or a launch refused, or a command buffer failed.
    Device(DeviceError),
    /// The plan refused the file.
    Plan(ChainError),
    /// A shape, a type or a request this session does not run, refused by
    /// name.
    Shape { what: &'static str, detail: String },
    /// The step raised the fault word. The session is poisoned until
    /// [`Session::reset`].
    Fault(Fault),
    /// A step asked of a session a fault poisoned.
    Poisoned(Fault),
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SessionError::Device(e) => write!(f, "{WHAT}: {e}"),
            SessionError::Plan(e) => write!(f, "{WHAT}: the plan: {e}"),
            SessionError::Shape { what, detail } => write!(f, "{what}: {detail}"),
            SessionError::Fault(fault) => write!(
                f,
                "{WHAT}: the step raised the fault word: {fault}; the session is poisoned until \
                 reset"
            ),
            SessionError::Poisoned(fault) => write!(
                f,
                "{WHAT}: a step refused: the session is poisoned by {fault}; reset it first"
            ),
        }
    }
}

impl std::error::Error for SessionError {}

impl From<DeviceError> for SessionError {
    fn from(e: DeviceError) -> Self {
        SessionError::Device(e)
    }
}

impl From<ChainError> for SessionError {
    fn from(e: ChainError) -> Self {
        SessionError::Plan(e)
    }
}

impl From<model_hf::model_file::Error> for SessionError {
    fn from(e: model_hf::model_file::Error) -> Self {
        refuse(e.to_string())
    }
}

pub(super) fn refuse(detail: impl Into<String>) -> SessionError {
    SessionError::Shape {
        what: WHAT,
        detail: detail.into(),
    }
}

/// What a step returns: the head's pick and what the token encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stepped {
    pub id: u32,
    pub count: EncodeCount,
}

/// How a step's token reaches the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepMode {
    /// The id is written into the host id slot; the step commits its token
    /// and waits for it, and nothing runs ahead. The next position's
    /// host-fed token is encoded during the wait and committed by the next
    /// host-fed call, its id written first.
    HostFed,
    /// As host-fed, unless a token runs ahead with this id; either way the
    /// next token is committed before this one is waited, embedding this
    /// token's pick on the device.
    Chain,
    /// The ids are known ahead of the calls: a prompt's. The token this call
    /// names lands with no head — its pick is nobody's input, the call after
    /// names its id — and with `next` named, that token is committed before
    /// this one is waited, embedding `next` from this token's record slot,
    /// which the host writes for it. A chain's token ahead reads the pick
    /// this token's head would have written there; this token writes no
    /// head, so the slot is the host's. `next` of `None` commits nothing
    /// ahead: the last row of a span a checkpoint ends, or the row before
    /// one whose pick is read.
    Ahead { next: Option<u32> },
}

/// What the session's steps did with the token ahead, the host's waits, the
/// command buffers committed and the passes run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepStats {
    /// The host's waits on the session's stream: every settle and sync.
    pub waits: u64,
    /// The command buffers committed on the session's stream.
    pub submits: u64,
    /// Tokens committed ahead of the call that names them.
    pub ahead: u64,
    /// Tokens ahead a call kept: its id was their embedded pick.
    pub kept: u64,
    /// Tokens ahead waited out and dropped: a broken promise, a fault, a
    /// reset, a traced step, a checkpoint or a cut.
    pub discarded: u64,
    /// Steps at the cache's last row that would commit a token ahead, and
    /// commit none.
    pub last_row: u64,
    /// Host-fed tokens committed from the encode the step before made.
    pub primed: u64,
    /// Passes that ran to their settle.
    pub passes: u64,
}

/// The token committed ahead of its call.
struct Ahead {
    signalled: Signalled,
    count: EncodeCount,
    pos: u32,
    /// The id its embed reads: the pick of the token before it (a chain's,
    /// learned at that token's land) or the id its committing call named (an
    /// ahead's, which knew it).
    embeds: u32,
    /// Whether the token's encode carried the head: a chain's does, its pick
    /// the next promise; an ahead's does not, the call after names its id.
    head: bool,
}

/// The next host-fed token, encoded during a host-fed step's wait and held
/// off the queue until the call that names its id.
struct Primed {
    parked: Parked,
    count: EncodeCount,
    pos: u32,
    /// The encode's own seconds, when the step clock ran at the encode.
    encode_s: Option<f64>,
}

/// The host's encode of a committed token, as the clock counts it.
#[derive(Clone, Copy)]
enum Encoded {
    /// Encoded at the commit, from this host second on.
    Since(f64),
    /// Encoded and parked a step before, in this many seconds.
    Took(f64),
}

/// One committed token the clock holds until its settle: its identity on the
/// stream, its stamps (a command buffer's handle, which only a lock may
/// carry), and the host's seconds around its encode and commit.
struct PendingToken {
    value: u64,
    stamps: GpuStamps,
    encode_s: f64,
    submitted: f64,
}

/// How many step modes a clock holds a row for.
const MODES: usize = 3;

/// A mode's place in a step clock's rows.
fn row_of(mode: StepMode) -> usize {
    match mode {
        StepMode::HostFed => 0,
        StepMode::Chain => 1,
        StepMode::Ahead { .. } => 2,
    }
}

/// One mode's phases, summed; the averages are [`StepPhase`].
#[derive(Clone, Copy, Default)]
struct Phases {
    encode_s: f64,
    start_s: f64,
    gpu_s: f64,
    wake_s: f64,
}

/// One mode's averaged step phases, as a run's steps spent them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepPhase {
    /// The settled steps the averages hold.
    pub steps: u64,
    /// The host's encode of a token, µs: the encoder's begin to the commit,
    /// or to the park for a token a host-fed step encoded during its wait,
    /// which no step waits on.
    pub encode_us: f64,
    /// A token's GPU start against the host clock at its commit, µs.
    pub start_us: f64,
    /// The GPU's own µs on a token.
    pub gpu_us: f64,
    /// The host's µs from a token's GPU end to the settle's return.
    pub wake_us: f64,
}

/// The step clock: what a run's steps spent, by the mode of the call that
/// settled each token. [`Session::start_step_clock`] starts one; a session
/// whose clock is off times nothing, and a step of it costs no read.
pub struct StepClock {
    /// The committed tokens whose settle has not come, the run-ahead one
    /// deep: two stand at once, the token a call settles and the one it
    /// committed ahead of it. The stamps' buffer handle is not `Sync`, so
    /// the lock carries it and it never leaves the clock.
    pending: Mutex<Vec<PendingToken>>,
    /// One row a mode, [`StepMode`]'s order.
    rows: [Phases; MODES],
    steps: [u64; MODES],
}

impl StepClock {
    fn new() -> Self {
        Self {
            // The run-ahead holds two; a step's commits never grow it.
            pending: Mutex::new(Vec::with_capacity(2)),
            rows: [Phases::default(); MODES],
            steps: [0; MODES],
        }
    }

    /// The registry, taken as it stands: every change to it is one push or
    /// one removal, so a panic while it was held left it whole.
    fn registry(&self) -> std::sync::MutexGuard<'_, Vec<PendingToken>> {
        self.pending.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The token `commit` clocked, held until its settle.
    fn committed(&self, token: PendingToken) {
        self.registry().push(token);
    }

    /// The token of signal `value`, taken from the registry: `None` when the
    /// clock holds none (it was committed before the clock started).
    fn take(&self, value: u64) -> Option<PendingToken> {
        let mut pending = self.registry();
        let at = pending.iter().position(|t| t.value == value)?;
        Some(pending.swap_remove(at))
    }

    /// The token of signal `value` whose settle will not come — dropped,
    /// reset — its clock dropped with it.
    fn dropped(&self, value: u64) {
        let mut pending = self.registry();
        if let Some(at) = pending.iter().position(|t| t.value == value) {
            pending.swap_remove(at);
        }
    }

    /// The settled token's phases under `mode`'s row: its stamps read (the
    /// settle already implied them complete) and `wake` the host's clock at
    /// the record.
    fn settled(
        &mut self,
        mode: StepMode,
        token: PendingToken,
        wake: f64,
    ) -> Result<(), SessionError> {
        let stamps = token.stamps.times(STAMP_WAIT)?;
        let row = &mut self.rows[row_of(mode)];
        row.encode_s += token.encode_s;
        row.start_s += stamps.start - token.submitted;
        row.gpu_s += stamps.end - stamps.start;
        row.wake_s += wake - stamps.end;
        self.steps[row_of(mode)] += 1;
        Ok(())
    }

    /// `mode`'s averaged phases; zeros when no step of the mode settled.
    #[must_use]
    pub fn phase(&self, mode: StepMode) -> StepPhase {
        let n = self.steps[row_of(mode)];
        let p = &self.rows[row_of(mode)];
        let avg = |s: f64| if n == 0 { 0.0 } else { s / n as f64 * 1e6 };
        StepPhase {
            steps: n,
            encode_us: avg(p.encode_s),
            start_us: avg(p.start_s),
            gpu_us: avg(p.gpu_s),
            wake_us: avg(p.wake_s),
        }
    }
}

/// One group of a profiled token: what the group runs, the layer it runs in
/// (none for the embed and the head), and the GPU seconds the group's own
/// command buffer ran, `GPUEndTime` − `GPUStartTime`.
#[derive(Clone, Copy, Debug)]
pub struct ProfileGroup {
    pub kind: GroupKind,
    pub layer: Option<usize>,
    pub gpu_s: f64,
}

/// One op of the walk, as a traced step names it to its hook.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Embed,
    AttnIn(usize),
    AttnCore(usize),
    AttnOut(usize),
    DeltaIn(usize),
    DeltaCore(usize),
    DeltaOut(usize),
    ConvIn(usize),
    ConvCore(usize),
    ConvOut(usize),
    Route(usize),
    Experts(usize),
    Head,
}

/// A buffer a traced step's hook reads, as of the op it is called after.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Node {
    /// The residual stream.
    Residual,
    /// Layer `l`'s P1 outputs, valid from its P1 to the next layer's.
    DeltaQkv(usize),
    DeltaZ(usize),
    DeltaBeta(usize),
    DeltaAlpha(usize),
    AttnQ(usize),
    AttnK(usize),
    AttnV(usize),
    /// The delta step's gated numerator (no per-head scale) and its partial
    /// sums of the squares of the values.
    DeltaG,
    DeltaPartials,
    /// Layer `l`'s recurrent state.
    DeltaState(usize),
    /// Layer `l`'s conv ring: every slot of the conv input rows.
    DeltaRing(usize),
    /// Layer `l`'s short conv ring: every slot of the conv's `B ⊙ x` rows.
    ConvRing(usize),
    /// The attention output after its gate.
    AttnOut,
    /// Layer `l`'s cache row `pos`, every key (or value) head, as f32.
    KeyRow(usize, u32),
    ValueRow(usize, u32),
    /// Layer `l`'s router logits (the routed experts, then the shared gate)
    /// and its weights (the picks', then the shared expert's).
    RouterLogits(usize),
    RouterWeights(usize),
    /// The experts' activations, one row a slot.
    ExpertRows,
    /// The head's logits.
    Logits,
}

/// What a traced step's hook reads: the arena after an op's work is done.
pub struct Probe<'a> {
    arena: &'a Arena,
    /// The delta lane the next token reads.
    lane: usize,
}

/// Every value of `b` in `r`. No device work is in flight when a probe or a
/// session reads.
fn read<T: Elem + Default>(b: &MetalBuffer<T>, r: Range<usize>) -> Result<Vec<T>, SessionError> {
    let mut out = vec![T::default(); r.len()];
    b.pages()
        .read_at(bloomery_device::Elems(r.start), &mut out)?;
    Ok(out)
}

impl Probe<'_> {
    /// The values of `node`. A node of a layer the arena does not hold, or
    /// of the other mixer kind, is refused by name.
    pub fn read(&self, node: Node) -> Result<Vec<f32>, SessionError> {
        let a = self.arena;
        let delta = |l: usize| match a.layers.get(l).map(|x| &x.mixer) {
            Some(MixerBufs::Delta(d)) => Ok(d),
            _ => Err(refuse(format!("{node:?}: layer {l} is no delta layer"))),
        };
        let attn = |l: usize| match a.layers.get(l).map(|x| &x.mixer) {
            Some(MixerBufs::Attention(at)) => Ok(at),
            _ => Err(refuse(format!("{node:?}: layer {l} is no attention layer"))),
        };
        let conv = |l: usize| match a.layers.get(l).map(|x| &x.mixer) {
            Some(MixerBufs::ShortConv(c)) => Ok(c),
            _ => Err(refuse(format!(
                "{node:?}: layer {l} is no short conv layer"
            ))),
        };
        let ffn = |l: usize| {
            a.layers
                .get(l)
                .map(|x| &x.ffn)
                .ok_or_else(|| refuse(format!("{node:?}: no layer {l}")))
        };
        let whole = |b: &MetalBuffer<f32>| read(b, 0..b.pages().len().get());
        match node {
            Node::Residual => whole(&a.x),
            Node::DeltaQkv(l) => read(&a.proj, delta(l)?.qkv.clone()),
            Node::DeltaZ(l) => read(&a.proj, delta(l)?.z.clone()),
            Node::DeltaBeta(l) => read(&a.proj, delta(l)?.beta.clone()),
            Node::DeltaAlpha(l) => read(&a.proj, delta(l)?.alpha.clone()),
            Node::AttnQ(l) => read(&a.proj, attn(l)?.q.clone()),
            Node::AttnK(l) => read(&a.proj, attn(l)?.k.clone()),
            Node::AttnV(l) => read(&a.proj, attn(l)?.v.clone()),
            Node::DeltaG => whole(&a.g),
            Node::DeltaPartials => whole(&a.delta_partials),
            Node::DeltaState(l) => {
                let lane = a
                    .geom
                    .delta
                    .state_elems(1)
                    .ok_or_else(|| refuse("a state lane does not fit"))?;
                let at = self.lane * lane;
                read(&delta(l)?.state, at..at + lane)
            }
            Node::DeltaRing(l) => whole(&delta(l)?.ring),
            Node::ConvRing(l) => whole(&conv(l)?.ring),
            Node::AttnOut => whole(&a.attn),
            Node::KeyRow(l, pos) | Node::ValueRow(l, pos) => {
                let at = attn(l)?;
                let cache = if matches!(node, Node::KeyRow(..)) {
                    &at.k_cache
                } else {
                    &at.v_cache
                };
                let pos = pos as usize;
                if pos >= a.ctx {
                    return Err(refuse(format!("{node:?}: the cache holds {} rows", a.ctx)));
                }
                let head = a.geom.attn.head;
                let heads = cache.pages().len().get() / (a.ctx * head);
                let mut out = Vec::with_capacity(heads * head);
                for h in 0..heads {
                    let at = (h * a.ctx + pos) * head;
                    let bits = read(cache, at..at + head)?;
                    out.extend(bits.into_iter().map(gguf::quant::half_to_f32));
                }
                Ok(out)
            }
            Node::RouterLogits(l) => match &ffn(l)?.router {
                Some(router) => whole(&router.logits),
                None => Err(refuse(format!(
                    "{node:?}: layer {l} has a dense FFN and no router"
                ))),
            },
            Node::RouterWeights(l) => whole(&ffn(l)?.weights),
            Node::ExpertRows => whole(&a.h),
            Node::Logits => whole(&a.logits),
        }
    }

    /// Layer `l`'s router picks: the routed ids, then the shared expert's.
    pub fn router_ids(&self, l: usize) -> Result<Vec<u32>, SessionError> {
        let f = &self
            .arena
            .layers
            .get(l)
            .ok_or_else(|| refuse(format!("router ids: no layer {l}")))?
            .ffn;
        read(&f.ids, 0..f.ids.pages().len().get())
    }
}

/// A qwen35moe session: the plan, the arena, the pipelines and the stream a
/// step runs on.
pub struct Session {
    stream: MetalStream,
    kernels: Kernels,
    plans: Vec<LayerPlan>,
    arena: Arena,
    layers: Vec<Layer>,
    pos: u32,
    poisoned: Option<Fault>,
    /// The token at `pos`, committed by the last step that ran one ahead.
    ahead: Option<Ahead>,
    /// The next host-fed token, encoded and parked.
    primed: Option<Primed>,
    stats: StepStats,
    /// The step clock, when a caller started one; `None` steps and times
    /// nothing.
    clock: Option<StepClock>,
    /// Each delta layer's state lane and conv ring and each short conv
    /// layer's ring, at the positions a prompt call takes; none for a file of
    /// no recurrent store, whose caches hold every position below the one it
    /// stands at.
    checkpoints: Option<Checkpoints>,
    /// The device the mark copies are opened on, by the first pass that
    /// takes one when the open did not make them.
    device: Handle<dyn MTLDevice>,
    /// The buffers of a pass.
    rows: RowsArena,
    /// How a prompt call feeds its ids.
    prefill: PrefillMode,
    /// How long the open's parts took.
    open_split: OpenSplit,
}

/// When a session opens the copies a pass leaves after rows inside it
/// ([`StateOut`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkCopies {
    /// With the session, beside the rows arena: a server's session, whose
    /// chat calls mark their messages.
    AtOpen,
    /// On the first pass that takes one: a session that never marks never
    /// holds them.
    OnFirstMark,
}

/// How long each part of a session's open took, on the host's clock. The
/// pipelines are built beside the other three, which run one after another:
/// the open's wall is the longer of the two, not the sum.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpenSplit {
    /// The plan read from the file and the weights copied into the arena.
    pub load: Duration,
    /// The rows arena's buffers allocated, and the mark copies' when the
    /// open makes them.
    pub rows: Duration,
    /// The residency set's commit: the driver wiring every buffer the open
    /// made.
    pub residency: Duration,
    /// The kernels' pipelines built, the prefill attention's among them.
    pub pipelines: Duration,
}

/// What [`Session::open_load`] made: the checked plan, the stream with its
/// residency set committed, both arenas, and how long each part took.
struct Loaded {
    checked: check::Checked,
    stream: MetalStream,
    arena: Arena,
    rows: RowsArena,
    load: Duration,
    rows_t: Duration,
    residency_t: Duration,
}

/// What a pass returns of its head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassOut {
    /// No head: the rows' picks are nobody's. The pass returns the id it was
    /// last fed.
    None,
    /// The head over the last row: its pick, and its logits row in
    /// [`Session::logits_row`]. (The verify arm's every-row head is its own
    /// variant.)
    Last,
}

/// The copies of the recurrent state a pass leaves, named by the rows they
/// follow: each delta layer's state lane and conv ring after that row, the
/// bits a pass ending there leaves in its stores, which
/// [`Session::checkpoint_copy`] takes as the point at the position after the
/// row. A prompt call names the message marks inside a pass here.
#[derive(Clone, Copy, Debug, Default)]
pub struct StateOut<'a> {
    /// The rows after which a copy is taken, ascending, each a row of the
    /// pass; at most [`SNAPS`]. None takes none.
    pub after_rows: &'a [u32],
}

/// One op of a profiled pass: the op, and the GPU seconds its own command
/// buffer ran, waited before the next op's was committed.
#[derive(Clone, Copy, Debug)]
pub struct PassSample {
    pub op: Op,
    pub gpu_s: f64,
}

/// What a pass returns: the head's pick, or the id it was last fed, and what
/// its ops encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Passed {
    pub id: u32,
    pub count: EncodeCount,
}

/// A recurrent store's pages, the range a checkpoint copies and its name.
type Store<'a> = (&'a MetalBuffer<f32>, Range<usize>, &'static str);

impl Session {
    /// The instance the file at `path` selects by its shape: the library a
    /// session of it with a cache of `ctx` rows opens on. The open's own
    /// check decides it, so a geometry no kernel takes, and every other
    /// refusal the open would meet first, is refused by name before a device
    /// opens.
    pub fn instance_of(path: &Path, ctx: usize) -> Result<&'static Instance, SessionError> {
        Self::instance_of_model(&ModelFile::open(path)?, ctx)
    }

    /// [`Session::instance_of`] for a model file already open.
    pub fn instance_of_model(
        file: &ModelFile,
        ctx: usize,
    ) -> Result<&'static Instance, SessionError> {
        Ok(check::check(file.header(), ctx)
            .checked_or_first()?
            .instance)
    }

    /// The session of the file at `path` on `device`, its pipelines from
    /// `library`, with a cache of `ctx` rows: the plan read from the file's
    /// header, every weight copied into the arena, every store at rest, and
    /// the rows arena of [`Session::pass_rows`] rows, all resident before
    /// the open returns. Checkpoints are taken every [`CHECKPOINT_EVERY`]
    /// positions.
    pub fn open(
        device: &MetalDevice,
        library: &Library,
        path: &Path,
        ctx: usize,
    ) -> Result<Session, SessionError> {
        Self::open_with(device, library, path, ctx, CHECKPOINT_EVERY)
    }

    /// [`Session::open`] with the inner checkpoints spaced `every` positions.
    pub fn open_with(
        device: &MetalDevice,
        library: &Library,
        path: &Path,
        ctx: usize,
        every: u32,
    ) -> Result<Session, SessionError> {
        Self::open_as(device, library, path, ctx, every, MarkCopies::OnFirstMark)
    }

    /// [`Session::open_with`] with the mark copies opened as `copies` says,
    /// over the file at `path` as [`ModelFile::open`] takes it.
    pub fn open_as(
        device: &MetalDevice,
        library: &Library,
        path: &Path,
        ctx: usize,
        every: u32,
        copies: MarkCopies,
    ) -> Result<Session, SessionError> {
        Self::open_model_as(device, library, &ModelFile::open(path)?, ctx, every, copies)
    }

    /// [`Session::open`] over a model file already open: a GGUF split or a
    /// checkpoint folder.
    pub fn open_model(
        device: &MetalDevice,
        library: &Library,
        file: &ModelFile,
        ctx: usize,
    ) -> Result<Session, SessionError> {
        Self::open_model_as(
            device,
            library,
            file,
            ctx,
            CHECKPOINT_EVERY,
            MarkCopies::OnFirstMark,
        )
    }

    /// [`Session::open_as`] over a model file already open. Every buffer is
    /// made and the residency set committed here, once: a refusal is a named
    /// error and leaves the set as it was, and no pass allocates or builds a
    /// pipeline but the first marked pass of a session opened
    /// [`MarkCopies::OnFirstMark`]. A file whose geometry some kernel cannot
    /// take, or whose instance is another than `library`'s, is refused by
    /// name.
    pub fn open_model_as(
        device: &MetalDevice,
        library: &Library,
        file: &ModelFile,
        ctx: usize,
        every: u32,
        copies: MarkCopies,
    ) -> Result<Session, SessionError> {
        // The pipelines are built on a thread of their own while this one
        // reads the plan and loads the weights: the two share the device and
        // nothing else. A refused build is still the open's first refusal;
        // the stream the load made, and the set it committed, go with it.
        let (built, loaded) = std::thread::scope(|s| {
            let pipelines = s.spawn(|| {
                let started = Instant::now();
                let kernels = Kernels::from_library(library)?;
                // Built here so a refused build fails the open, never a pass.
                kernels.prepare_prefill()?;
                Ok::<_, SessionError>((kernels, started.elapsed()))
            });
            let loaded = Self::open_load(device, library, file, ctx, copies);
            let built = pipelines
                .join()
                .unwrap_or_else(|p| std::panic::resume_unwind(p));
            (built, loaded)
        });
        let (kernels, pipelines) = built?;
        let Loaded {
            checked,
            stream,
            arena,
            rows,
            load,
            rows_t,
            residency_t,
        } = loaded?;
        let plans = checked.plans;
        let ctx_positions = u32::try_from(arena.ctx).map_err(|_| {
            refuse(format!(
                "a cache of {} rows: more than a u32 position names",
                arena.ctx
            ))
        })?;
        let lens = store_lens(&arena)?;
        let checkpoints = (!lens.is_empty())
            .then(|| {
                Checkpoints::open(
                    device.handle().clone(),
                    lens,
                    u64::from(ctx_positions),
                    every,
                )
            })
            .transpose()
            .map_err(checkpoint_error)?;
        let prefill = prefill_at_open(&checked.geom);
        let layers = arena.token_layers();
        Ok(Session {
            stream,
            kernels,
            plans,
            arena,
            layers,
            pos: 0,
            poisoned: None,
            ahead: None,
            primed: None,
            stats: StepStats::default(),
            clock: None,
            checkpoints,
            device: device.handle().clone(),
            rows,
            prefill,
            open_split: OpenSplit {
                load,
                rows: rows_t,
                residency: residency_t,
                pipelines,
            },
        })
    }

    /// The load half of [`Session::open_model_as`], run beside the pipeline
    /// build: the plan's checks, the stream and its residency set, the arena,
    /// the rows arena (and the mark copies' when `copies` asks at open), and
    /// the set's commit. The arena is sized at the geometry the file was
    /// judged at and the launches at the library's: one instance, or the
    /// open is refused before any buffer is made.
    fn open_load(
        device: &MetalDevice,
        library: &Library,
        file: &ModelFile,
        ctx: usize,
        copies: MarkCopies,
    ) -> Result<Loaded, SessionError> {
        let started = Instant::now();
        // Every refusal the header and tensor table hold, in the order the
        // checks meet them; the open stops at the first.
        let checked = check::check(file.header(), ctx).checked_or_first()?;
        if checked.geom != Geometry::of(library.instance()) {
            return Err(refuse(format!(
                "{}: the file is {}, the library is built for {}",
                file.header()
                    .shard_path(0)
                    .map_or_else(|| "the model file".into(), |p| p.display().to_string()),
                checked.instance,
                library.instance()
            )));
        }
        let mut stream = device.new_stream()?;
        stream.enable_residency(device.metal())?;
        let residency = stream
            .residency()
            .ok_or_else(|| refuse("the stream made no residency set"))?;
        let staged = residency.stage();
        let alloc = Alloc::new(device.handle(), &staged);
        let weights = file.weights()?;
        let arena = Arena::open(alloc, &weights, &checked.geom, &checked.layout)?;
        let load = started.elapsed();
        let started = Instant::now();
        let mut rows = RowsArena::open(alloc, &arena, pass_rows_max(&checked.geom).min(arena.ctx))?;
        // A file with no delta layer has nothing a mark copies: a short conv
        // ring is not copied inside a pass (the pass is refused by name).
        let stores = arena
            .layers
            .iter()
            .any(|l| matches!(l.mixer, MixerBufs::Delta(_)));
        if copies == MarkCopies::AtOpen && stores {
            rows.snaps = Some(SnapBufs::open(alloc, &arena)?);
        }
        let rows_t = started.elapsed();
        // The open waits for the driver to wire the arenas, so the first
        // token and the first pass find them resident.
        let started = Instant::now();
        staged.commit();
        let residency_t = started.elapsed();
        Ok(Loaded {
            checked,
            stream,
            arena,
            rows,
            load,
            rows_t,
            residency_t,
        })
    }

    /// How long the open's parts took.
    #[must_use]
    pub fn open_split(&self) -> OpenSplit {
        self.open_split
    }

    /// The launch geometry the session runs at: its instance's.
    #[must_use]
    pub fn geom(&self) -> &Geometry {
        &self.arena.geom
    }

    /// The token as the count models it.
    #[must_use]
    pub fn token_layers(&self) -> &[Layer] {
        &self.layers
    }

    /// The next step's position.
    #[must_use]
    pub fn pos(&self) -> u32 {
        self.pos
    }

    /// The rows the cache holds.
    #[must_use]
    pub fn ctx(&self) -> usize {
        self.arena.ctx
    }

    /// The head's rows.
    #[must_use]
    pub fn vocab(&self) -> usize {
        self.arena.vocab
    }

    /// The RMS norms' epsilon.
    #[must_use]
    pub fn eps(&self) -> f32 {
        self.arena.eps
    }

    /// The model's layers.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.arena.layers.len()
    }

    /// The fault a step raised, while the session is poisoned.
    #[must_use]
    pub fn poisoned(&self) -> Option<Fault> {
        self.poisoned
    }

    /// A step may run: the session is not poisoned and the cache has a row.
    fn ready(&self) -> Result<(), SessionError> {
        if let Some(fault) = self.poisoned {
            return Err(SessionError::Poisoned(fault));
        }
        if self.pos as usize >= self.arena.ctx {
            return Err(refuse(format!(
                "a step at position {}: the cache holds {} rows",
                self.pos, self.arena.ctx
            )));
        }
        Ok(())
    }

    /// What the steps did with the token ahead, the host's waits and the
    /// command buffers committed so far.
    #[must_use]
    pub fn stats(&self) -> StepStats {
        StepStats {
            waits: self.stream.waits(),
            submits: self.stream.submits(),
            ..self.stats
        }
    }

    /// Start the step clock: every step from here on has its phases
    /// recorded, a row a mode, until the next start. The clock is a
    /// profiling run's instrument; it holds nothing back and clears nothing,
    /// not even at [`Session::reset`], which a caller that wants fresh rows
    /// starts again past. The GPU's stamps and the host's reads share one
    /// clock ([`host_secs`]), so the phases subtract them directly.
    pub fn start_step_clock(&mut self) {
        self.clock = Some(StepClock::new());
    }

    /// The step clock's phases, when one runs.
    #[must_use]
    pub fn step_clock(&self) -> Option<&StepClock> {
        self.clock.as_ref()
    }

    /// One token at the next position, `id` its input; the head's pick is
    /// returned, or for an ahead step the id it was fed, which computed no
    /// pick to read back. A token running ahead with this id is kept when
    /// the call's mode and the token agree on the head — a pick-reading call
    /// ([`StepMode::Chain`], [`StepMode::HostFed`]) lands only a token that
    /// carries one, an ahead call only one that does not — and any other is
    /// waited out and dropped first. In [`StepMode::Chain`] and an ahead
    /// step that names a next id, the next token is committed before this
    /// one is waited, unless this is the cache's last row. A call waits for
    /// this token alone, and on a dropped token before it.
    pub fn step(&mut self, id: u32, mode: StepMode) -> Result<Stepped, SessionError> {
        self.ready()?;
        let head = !matches!(mode, StepMode::Ahead { .. });
        let (current, count) = match self.ahead.take() {
            Some(a) if a.embeds == id && a.head == head => {
                self.stats.kept += 1;
                (a.signalled, a.count)
            }
            ahead => {
                if let Some(a) = ahead {
                    self.drop_ahead(a)?;
                }
                self.commit(id, self.pos, Feed::Host, head)?
            }
        };
        // On an error past this point the token at `pos` is waited for and
        // not landed: its fault word, if it raised, poisons the next step at
        // `pos`. The first error is the step's; a later one on the same path
        // is not returned.
        let next = self.pos + 1;
        // The token ahead, as the mode commits it: a chain's embeds this
        // token's pick (its head writes the record slot the embed reads) and
        // carries a head of its own; an ahead's embeds the id the host writes
        // into that slot and carries none.
        let ahead = match mode {
            StepMode::HostFed | StepMode::Ahead { next: None } => None,
            StepMode::Chain => Some((next, 0, true, None)),
            StepMode::Ahead {
                next: Some(host_written),
            } => Some((next, host_written, false, Some(host_written))),
        };
        if let Some((at, embeds, carries_head, host_written)) = ahead {
            if (at as usize) < self.arena.ctx {
                if let Err(e) = self.commit_ahead(at, embeds, carries_head, host_written) {
                    let _ = self.stream.settle(current);
                    return Err(e);
                }
            } else {
                self.stats.last_row += 1;
            }
        }
        // A host-fed step encodes the next host-fed token while this one
        // runs; its id is the host's pick of this token's logits.
        if mode == StepMode::HostFed
            && (next as usize) < self.arena.ctx
            && let Err(e) = self.prime(next)
        {
            let _ = self.stream.settle(current);
            return Err(e);
        }
        let value = current.value();
        let settled = match self.stream.settle(current) {
            Ok(settled) => settled,
            Err(e) => {
                if let Some(a) = self.ahead.take() {
                    let _ = self.drop_ahead(a);
                }
                return Err(e.into());
            }
        };
        if let Some(clock) = self.clock.as_mut()
            && let Some(token) = clock.take(value)
        {
            let wake = host_secs();
            clock.settled(mode, token, wake)?;
        }
        let stepped = self.land(&settled, count, head, id);
        match (&stepped, self.ahead.as_mut()) {
            // A chain's token ahead learns its promise here, the pick of the
            // token it embeds; an ahead's named its at the commit.
            (Ok(_), Some(a)) if !a.head => {}
            (Ok(s), Some(a)) => a.embeds = s.id,
            (Err(_), Some(_)) => self.wait_out_ahead(),
            _ => {}
        }
        stepped
    }

    /// How a prompt call feeds its ids: in passes, or a decode step each.
    #[must_use]
    pub fn prefill(&self) -> PrefillMode {
        self.prefill
    }

    /// Feed prompt calls in passes ([`PrefillMode::Batch`], the default) or a
    /// step each ([`PrefillMode::Steps`], the same-binary arm and the
    /// oracle's).
    pub fn set_prefill(&mut self, mode: PrefillMode) {
        self.prefill = mode;
    }

    /// The most rows a pass of this session holds: the family's ubatch rule,
    /// and no more than the cache's rows.
    #[must_use]
    pub fn pass_rows(&self) -> usize {
        self.rows.cap
    }

    /// What the residency set of the session's kernel buffers holds and how
    /// often it was requested.
    #[must_use]
    pub fn residency(&self) -> ResidencyStats {
        self.stream
            .residency()
            .expect("a session opens its stream with a residency set")
            .stats()
    }

    /// The bytes the rows arena holds.
    #[must_use]
    pub fn rows_arena_bytes(&self) -> u64 {
        self.rows.bytes
    }

    /// `ids` as one pass at the next positions: every layer once over all the
    /// rows, the residual rows added into by GEMMs and row kernels. More than a
    /// step's window ([`PASS_ROWS`]) and at most [`Session::pass_rows`] ids;
    /// any other count is refused by name, with the position unmoved. A token
    /// running ahead is waited out and dropped first. The pass is one command
    /// buffer a layer, its ops in the walk's order, each closed by the barrier
    /// its last dispatch is followed by; the buffers are committed unwaited,
    /// each after the one before it, and the pass waits on the last at its
    /// end: a failed buffer is named there, its fault word is read once, and
    /// a raise poisons the session as a step's does. [`PassOut::Last`] runs the head
    /// over the last row and returns its pick; [`PassOut::None`] returns the
    /// last id fed.
    ///
    /// A refusal inside the first layer commits nothing, so the stores stay as
    /// they were; one in a later layer waits out the layers committed, and a
    /// failure at the settle leaves the stores as the buffers left them: the
    /// caller resets.
    ///
    /// `snaps` names the rows after which the delta layers' lanes and rings
    /// are copied ([`StateOut`]); rows out of order, past the pass or more
    /// than [`SNAPS`] are refused by name before anything runs. The copies
    /// hold their positions once the pass settles clean, and until the next
    /// pass, cut or reset.
    pub fn pass(
        &mut self,
        ids: &[u32],
        out: PassOut,
        snaps: StateOut<'_>,
    ) -> Result<Passed, SessionError> {
        self.pass_with(ids, out, snaps, None)
    }

    /// [`Session::pass`] with every op on a command buffer of its own,
    /// waited before the next is committed, its GPU seconds read: one sample
    /// an op, in the pass's order, appended to `samples` (cleared first). Each
    /// buffer runs alone, so a sample is the op's own time and not a queued
    /// buffer's overlap with the one before it. The ops are the unprofiled
    /// pass's, in its order, so it leaves the same bits. A refusal or a
    /// failure inside the walk leaves the stores as the ops that ran left
    /// them; the caller resets.
    pub fn pass_profile(
        &mut self,
        ids: &[u32],
        out: PassOut,
        samples: &mut Vec<PassSample>,
    ) -> Result<Passed, SessionError> {
        samples.clear();
        self.pass_with(ids, out, StateOut::default(), Some(samples))
    }

    fn pass_with(
        &mut self,
        ids: &[u32],
        out: PassOut,
        snaps: StateOut<'_>,
        profile: Option<&mut Vec<PassSample>>,
    ) -> Result<Passed, SessionError> {
        self.ready()?;
        let m = ids.len();
        let after = snaps.after_rows;
        if m <= PASS_ROWS {
            return Err(refuse(format!(
                "a pass of {m} ids; a step's window is {PASS_ROWS} and a pass is more"
            )));
        }
        let cap = self.pass_rows();
        if m > cap {
            return Err(refuse(format!("a pass of {m} ids; a pass holds {cap}")));
        }
        if after.len() > SNAPS
            || !after.windows(2).all(|w| w[0] < w[1])
            || after.iter().any(|&r| r as usize >= m)
        {
            return Err(refuse(format!(
                "a pass of {m} ids with copies after rows {after:?}: at most {SNAPS}, ascending, \
                 each a row of the pass"
            )));
        }
        if !after.is_empty() && !self.copies_in_pass() {
            return Err(refuse(format!(
                "a pass of {m} ids with copies after rows {after:?}: a file with a short conv layer \
                 takes none, the ring's copy is not run"
            )));
        }
        let from = self.pos;
        let rows_n = u32::try_from(m).map_err(|_| refuse(format!("{m} ids do not fit u32")))?;
        match from.checked_add(rows_n) {
            Some(to) if to as usize <= self.arena.ctx => {}
            _ => {
                return Err(refuse(format!(
                    "a pass of {m} ids at position {from}: the cache holds {} rows",
                    self.arena.ctx
                )));
            }
        }
        if let Some(a) = self.ahead.take() {
            self.drop_ahead(a)?;
        }
        let rows = &mut self.rows;
        if !after.is_empty() && rows.snaps.is_none() {
            let residency = self
                .stream
                .residency()
                .ok_or_else(|| refuse("the stream has no residency set"))?;
            let staged = residency.stage();
            let copies = SnapBufs::open(Alloc::new(&self.device, &staged), &self.arena)?;
            staged.commit();
            rows.snaps = Some(copies);
        }
        if let Some(copies) = rows.snaps.as_mut() {
            copies.holds = [None; SNAPS];
        }
        rows.ids.write_at(Elems(0), ids)?;
        let slot = slot_of(from + rows_n - 1);
        let head = out == PassOut::Last;
        let cap_rows = rows.cap;
        let (count, settled) = match profile {
            None => {
                let mut ops = PassLayers {
                    buffers: LayerBuffers::new(&self.stream),
                    kernels: &self.kernels,
                    arena: &mut self.arena,
                    rows,
                    after,
                    from,
                    head,
                };
                let walked = Walk::new(&mut ops, &self.plans, m)
                    .run(cap_rows)
                    .and_then(|()| ops.buffers.finish());
                let LayerBuffers { count, last, .. } = ops.buffers;
                // A refusal after a layer was committed: that layer ran and
                // wrote, so the stores are as it left them and the caller
                // resets. Before the first commit nothing ran.
                if let Err(e) = walked {
                    if last.is_some() {
                        let _ = self.stream.sync();
                    }
                    return Err(e);
                }
                let signalled = last.ok_or_else(|| refuse("a pass committed no buffer"))?;
                (count, self.stream.settle(signalled)?)
            }
            Some(samples) => {
                let mut ops = PassProfiler {
                    stream: &self.stream,
                    kernels: &self.kernels,
                    arena: &mut self.arena,
                    rows,
                    after,
                    from,
                    head,
                    count: EncodeCount::default(),
                    settled: None,
                    samples,
                };
                let walked = Walk::new(&mut ops, &self.plans, m).run(cap_rows);
                let (count, settled) = (ops.count, ops.settled.take());
                if let Err(e) = walked {
                    if settled.is_some() {
                        let _ = self.stream.sync();
                    }
                    return Err(e);
                }
                (
                    count,
                    settled.ok_or_else(|| refuse("a pass settled no op"))?,
                )
            }
        };
        self.stats.passes += 1;
        if let Some(fault) = self.arena.faults[slot].read_settled(&settled)? {
            self.poisoned = Some(fault);
            return Err(SessionError::Fault(fault));
        }
        let id = if head {
            self.arena.step_id.pages().load(slot, Ordering::Relaxed)?
        } else {
            ids[m - 1]
        };
        if let Some(copies) = self.rows.snaps.as_mut() {
            for (held, &r) in copies.holds.iter_mut().zip(after) {
                *held = Some(from + r + 1);
            }
        }
        self.pos = from + rows_n;
        Ok(Passed { id, count })
    }

    /// Commit the token ahead at `at` and register the promise `embeds`: the
    /// id the host writes into this token's record slot first when
    /// `host_written` names one, then the token encoded on the device's feed
    /// — the record slot of the token before it — carrying a head only when
    /// `head`. The stream's token boundary orders it after this token; on
    /// `Err` nothing is registered and the caller settles this token.
    fn commit_ahead(
        &mut self,
        at: u32,
        embeds: u32,
        head: bool,
        host_written: Option<u32>,
    ) -> Result<(), SessionError> {
        if let Some(id) = host_written {
            // The slot the token ahead embeds: this token's, which its
            // skipped head leaves for the host. Its last reader, the embed
            // of the token two positions back, settled with the call before.
            self.arena
                .step_id
                .pages()
                .store(slot_of(self.pos), id, Ordering::Relaxed)?;
        }
        let (signalled, count) = self.commit(embeds, at, Feed::Device, head)?;
        self.stats.ahead += 1;
        self.ahead = Some(Ahead {
            signalled,
            count,
            pos: at,
            embeds,
            head,
        });
        Ok(())
    }

    /// Encode and commit the token at `pos`, signalled: its embed reads the
    /// host id slot, which `id` is written into first, or the record slot of
    /// the token before it, and it carries the head only when `head` — a
    /// step whose pick nobody reads skips it.
    fn commit(
        &mut self,
        id: u32,
        pos: u32,
        feed: Feed,
        head: bool,
    ) -> Result<(Signalled, EncodeCount), SessionError> {
        if feed == Feed::Host {
            self.arena.id_in.pages().store(0, id, Ordering::Relaxed)?;
        }
        // The parked token of this position, when it is still fresh: a
        // host-fed token with a head is what the step before encoded.
        let primed = self
            .primed
            .take()
            .filter(|p| p.pos == pos && feed == Feed::Host && head)
            .and_then(|p| {
                let recording = self.stream.unpark(p.parked)?;
                Some((recording, p.count, p.encode_s.map(Encoded::Took)))
            });
        let clock = self.clock.as_ref();
        let (recording, count, encode_s) = match primed {
            Some(primed) => {
                self.stats.primed += 1;
                primed
            }
            None => {
                let encode_from = clock.is_some().then(host_secs);
                let mut recording = self.stream.begin()?;
                let mut enc = recording.token_encoder()?;
                Walk::new(
                    &mut MetalOps::new(&mut enc, &mut self.arena, &self.kernels, pos, feed, head),
                    &self.plans,
                    1,
                )
                .run(1)?;
                let count = enc.count();
                enc.end();
                (recording, count, encode_from.map(Encoded::Since))
            }
        };
        let stamps = clock.is_some().then(|| recording.stamps());
        let signalled = self.stream.submit_signalled(recording)?;
        if let (Some(clock), Some(stamps), Some(encoded)) = (clock, stamps, encode_s) {
            let at = host_secs();
            clock.committed(PendingToken {
                value: signalled.value(),
                stamps,
                encode_s: match encoded {
                    Encoded::Since(from) => at - from,
                    Encoded::Took(s) => s,
                },
                submitted: at,
            });
        }
        Ok((signalled, count))
    }

    /// Encode the host-fed token at `pos`, with its head, and park it for
    /// the call that names its id ([`Session::commit`] takes it). Its embed
    /// reads the host id slot, which that call writes before the commit; the
    /// token this step waits for has read the slot by then.
    fn prime(&mut self, pos: u32) -> Result<(), SessionError> {
        let encode_from = self.clock.is_some().then(host_secs);
        let mut recording = self.stream.begin()?;
        let mut enc = recording.token_encoder()?;
        Walk::new(
            &mut MetalOps::new(
                &mut enc,
                &mut self.arena,
                &self.kernels,
                pos,
                Feed::Host,
                true,
            ),
            &self.plans,
            1,
        )
        .run(1)?;
        let count = enc.count();
        enc.end();
        let parked = self.stream.park(recording)?;
        self.primed = Some(Primed {
            parked,
            count,
            pos,
            encode_s: encode_from.map(|from| host_secs() - from),
        });
        Ok(())
    }

    /// The token ahead waited out and dropped, its fault word cleared if it
    /// raised: the token at its position is computed again, from the id the
    /// call names, and starts clean.
    fn drop_ahead(&mut self, a: Ahead) -> Result<(), SessionError> {
        self.stats.discarded += 1;
        if let Some(clock) = &self.clock {
            clock.dropped(a.signalled.value());
        }
        let settled = self.stream.settle(a.signalled)?;
        let fault = &self.arena.faults[slot_of(a.pos)];
        if fault.read_settled(&settled)?.is_some() {
            fault.clear_settled(&settled)?;
        }
        Ok(())
    }

    /// The token ahead waited out and dropped when the token before it
    /// faulted: the step returns that fault, and [`Session::reset`] clears
    /// whatever the token ahead raised. Its own failure is not returned.
    fn wait_out_ahead(&mut self) {
        if let Some(a) = self.ahead.take() {
            self.stats.discarded += 1;
            if let Some(clock) = &self.clock {
                clock.dropped(a.signalled.value());
            }
            let _ = self.stream.settle(a.signalled);
        }
    }

    /// [`Session::step`] host-fed, with every op on a command buffer of its
    /// own, waited, and `hook` called after each with what the op left. The
    /// count is the sum of the ops' counts: what one encoder of the whole
    /// token encodes. A token running ahead is dropped first. A hook's error
    /// ends the step and is returned; the session keeps the work done so far,
    /// and its position does not move.
    pub fn step_traced<H>(&mut self, id: u32, hook: H) -> Result<Stepped, SessionError>
    where
        H: FnMut(Op, &Probe<'_>) -> Result<(), SessionError>,
    {
        self.ready()?;
        if let Some(a) = self.ahead.take() {
            self.drop_ahead(a)?;
        }
        self.arena.id_in.pages().store(0, id, Ordering::Relaxed)?;
        let mut tracer = Tracer {
            stream: &self.stream,
            kernels: &self.kernels,
            arena: &mut self.arena,
            pos: self.pos,
            count: EncodeCount::default(),
            settled: None,
            hook,
        };
        Walk::new(&mut tracer, &self.plans, 1).run(1)?;
        let count = tracer.count;
        let settled = tracer
            .settled
            .ok_or_else(|| refuse("a traced step settled no op"))?;
        self.land(&settled, count, true, id)
    }

    /// [`Session::step`] host-fed, with every group on a command buffer of
    /// its own, waited, its GPU seconds read and one sample a group appended
    /// to `out`, in the token's order. The count is the sum of the groups'
    /// counts: what one encoder of the whole token encodes. A token running
    /// ahead is dropped first. `out` is cleared and keeps its capacity, so a
    /// steady call allocates nothing.
    pub fn step_profile(
        &mut self,
        id: u32,
        out: &mut Vec<ProfileGroup>,
    ) -> Result<Stepped, SessionError> {
        self.ready()?;
        if let Some(a) = self.ahead.take() {
            self.drop_ahead(a)?;
        }
        self.arena.id_in.pages().store(0, id, Ordering::Relaxed)?;
        out.clear();
        let mut profiler = Profiler {
            stream: &self.stream,
            kernels: &self.kernels,
            arena: &mut self.arena,
            pos: self.pos,
            count: EncodeCount::default(),
            settled: None,
            out,
        };
        Walk::new(&mut profiler, &self.plans, 1).run(1)?;
        let count = profiler.count;
        let settled = profiler
            .settled
            .ok_or_else(|| refuse("a profiled step settled no group"))?;
        self.land(&settled, count, true, id)
    }

    /// The bytes the profiled group `g` of the token at `pos` moved, by what
    /// moved them: the byte rule's one owner is the arena's. `ids` is that
    /// group's layer's router ids as the step left them (the routed picks,
    /// then the shared expert's); the expert groups take them, every other
    /// group none.
    pub fn group_bytes(
        &self,
        g: &ProfileGroup,
        pos: u32,
        ids: &[u32],
    ) -> Result<GroupBytes, SessionError> {
        let layer = g.layer.unwrap_or(0);
        self.arena.group_bytes(g.kind, layer, pos, ids)
    }

    /// A settled token's end: its fault word (a fault poisons the session),
    /// then its pick — the head's when the token carries one, else the id
    /// `fed` the call, for a headless token computed no pick to read back;
    /// the position moves.
    fn land(
        &mut self,
        settled: &Settled,
        count: EncodeCount,
        head: bool,
        fed: u32,
    ) -> Result<Stepped, SessionError> {
        let slot = slot_of(self.pos);
        if let Some(fault) = self.arena.faults[slot].read_settled(settled)? {
            self.poisoned = Some(fault);
            return Err(SessionError::Fault(fault));
        }
        let id = if head {
            self.arena.step_id.pages().load(slot, Ordering::Relaxed)?
        } else {
            fed
        };
        self.pos += 1;
        Ok(Stepped { id, count })
    }

    /// What the last step left, read as a traced step's hook reads it. A
    /// token running ahead is still writing the arena, so a probe is refused
    /// by name while one runs: a host-fed step drops it.
    pub fn probe(&self) -> Result<Probe<'_>, SessionError> {
        if self.ahead.is_some() {
            return Err(refuse(format!(
                "a probe at position {}: the token at it runs ahead; a host-fed step drops it",
                self.pos
            )));
        }
        self.stream.sync()?;
        Ok(Probe {
            arena: &self.arena,
            lane: lane_of(self.pos),
        })
    }

    /// The head's logits row of the last stepped token: the shared pages the
    /// head wrote, zero copy. Refused while a token runs ahead, as a probe
    /// is: its head would overwrite the row while the caller reads it.
    pub fn logits_row(&self) -> Result<&[f32], SessionError> {
        if self.ahead.is_some() {
            return Err(refuse(format!(
                "the logits row at position {}: the token at it runs ahead; a host-fed step \
                 drops it",
                self.pos
            )));
        }
        self.stream.sync()?;
        // SAFETY: no token runs ahead (refused above) and the last step
        // settled before it returned, so no device work writes these pages;
        // the row borrows the session, whose steps and stores all go through
        // `&mut self`, so none writes them for as long as the row lives.
        Ok(unsafe { self.arena.logits.pages().as_slice() })
    }

    /// Every store back at rest: position 0, the caches, the recurrent
    /// states and rings, the lane words, the tickets, both record slots and
    /// both fault words cleared, the poison lifted. A token running ahead is
    /// waited out first.
    pub fn reset(&mut self) -> Result<(), SessionError> {
        if let Some(a) = self.ahead.take() {
            self.stats.discarded += 1;
            if let Some(clock) = &self.clock {
                clock.dropped(a.signalled.value());
            }
            self.stream.settle(a.signalled)?;
        }
        self.stream.sync()?;
        self.arena.clear_stores()?;
        for fault in &self.arena.faults {
            fault.clear_waited()?;
        }
        self.pos = 0;
        self.poisoned = None;
        if let Some(c) = self.checkpoints.as_mut() {
            c.clear();
        }
        self.drop_copies();
        Ok(())
    }

    /// The last pass's copies hold no position any more.
    fn drop_copies(&mut self) {
        if let Some(copies) = self.rows.snaps.as_mut() {
            copies.holds = [None; SNAPS];
        }
    }

    /// The checkpoints' positions, ascending.
    #[must_use]
    pub fn checkpoint_positions(&self) -> Vec<u32> {
        self.checkpoints
            .as_ref()
            .map_or_else(Vec::new, Checkpoints::positions)
    }

    /// The bytes one checkpoint copies: none with no recurrent store.
    #[must_use]
    pub fn checkpoint_bytes(&self) -> u64 {
        self.checkpoints
            .as_ref()
            .map_or(0, Checkpoints::point_bytes)
    }

    /// The checkpoints the budget holds.
    #[must_use]
    pub fn checkpoint_capacity(&self) -> usize {
        self.checkpoints.as_ref().map_or(0, Checkpoints::capacity)
    }

    /// The spacing of a prompt call's inner checkpoints; none with no
    /// recurrent store.
    #[must_use]
    pub fn checkpoint_every(&self) -> Option<u32> {
        self.checkpoints.as_ref().map(Checkpoints::every)
    }

    /// The positions a prompt call from `from` to `to` takes a checkpoint at:
    /// none with no recurrent store.
    #[must_use]
    pub fn checkpoint_marks(&self, from: u32, to: u32) -> Vec<u32> {
        self.checkpoints
            .as_ref()
            .map_or_else(Vec::new, |c| c.marks(from, to))
    }

    /// What a cut to at most `n` keeps of the position the session stands at.
    /// With no recurrent store a cut keeps any position up to it.
    #[must_use]
    pub fn kept(&self, n: u32) -> Kept {
        match &self.checkpoints {
            Some(c) => c.kept(n, self.pos),
            None => Kept::rule(n, self.pos, n.min(self.pos)),
        }
    }

    /// Whether a pass can copy the recurrent stores after one of its rows
    /// ([`StateOut::after_rows`]): not on a file with a short conv layer, whose
    /// ring has no pass copy; a caller marks such a file between passes.
    pub fn copies_in_pass(&self) -> bool {
        !self
            .arena
            .layers
            .iter()
            .any(|l| matches!(l.mixer, MixerBufs::ShortConv(_)))
    }

    /// A checkpoint at the position, the stream idle: the token ahead is
    /// waited out and dropped first, then each delta layer's state lane at
    /// the position's parity and its ring, and each short conv layer's ring,
    /// are copied once, from the shared pages straight into the point's slot.
    /// A file of no recurrent store has nothing to copy: its caches hold the
    /// position. Refused while the session is poisoned.
    pub fn checkpoint(&mut self) -> Result<(), SessionError> {
        self.take_point(self.pos, None)
    }

    /// A checkpoint at `at`, a position inside the last pass that pass left
    /// a copy of ([`StateOut::after_rows`]): each delta layer's lane and ring
    /// copied once from that copy into the point's slot, the stream idle. A
    /// position no copy holds (another pass, a cut or a reset came since) is
    /// refused by name, as is one past the position. Points go in ascending
    /// order: a copy's point before the point at the pass's end.
    pub fn checkpoint_copy(&mut self, at: u32) -> Result<(), SessionError> {
        let held = self
            .rows
            .snaps
            .as_ref()
            .and_then(|c| c.holds.iter().position(|&h| h == Some(at)));
        match held {
            Some(k) if at <= self.pos => self.take_point(at, Some(k)),
            _ => Err(refuse(format!(
                "a checkpoint at {at} from a pass's copy: no copy the last pass left holds it \
                 (the session stands at {})",
                self.pos
            ))),
        }
    }

    /// The point at `at` from each recurrent layer's live stores (`copy` none:
    /// a delta layer's lane at `at`'s parity, `at` the position, and each
    /// ring) or from the last pass's copy `copy`, after the stream is idle. A
    /// short conv layer has no pass copy. Refused while poisoned.
    fn take_point(&mut self, at: u32, copy: Option<usize>) -> Result<(), SessionError> {
        if let Some(fault) = self.poisoned {
            return Err(SessionError::Poisoned(fault));
        }
        self.quiesce()?;
        let lane_len = held_lane_len(&self.arena)?;
        let live = lane_of(at)
            .checked_mul(lane_len)
            .ok_or_else(|| refuse("a delta state lane's offset overflowed"))?;
        let copies = match copy {
            None => None,
            Some(k) => Some((
                k,
                self.rows
                    .snaps
                    .as_ref()
                    .ok_or_else(|| refuse("a pass's copy with no copies open"))?,
            )),
        };
        let mut views = Vec::new();
        for (l, layer) in self.arena.layers.iter().enumerate() {
            // The layer's stores in the order `store_lens` lists them: each
            // one's pages, the range its copy covers and its name.
            let stores: [Option<Store<'_>>; 2] = match &layer.mixer {
                MixerBufs::Attention(_) => [None, None],
                MixerBufs::ShortConv(c) => {
                    if copy.is_some() {
                        return Err(refuse(format!("short conv layer {l} has no pass copies")));
                    }
                    let ring = 0..c.ring.pages().len().get();
                    [Some((&c.ring, ring, "a short conv ring")), None]
                }
                MixerBufs::Delta(d) => {
                    let (pages, lane_at, ring_at) = match copies {
                        None => ((&d.state, &d.ring), live, 0),
                        Some((k, c)) => {
                            let at =
                                c.at.get(l).copied().flatten().ok_or_else(|| {
                                    refuse(format!("delta layer {l} has no copies"))
                                })?;
                            (
                                (&c.lane, &c.ring),
                                at.lane + k * lane_len,
                                at.ring + k * at.ring_len,
                            )
                        }
                    };
                    let ring_len = d.ring.len().get();
                    [
                        Some((pages.0, lane_at..lane_at + lane_len, "a delta state lane")),
                        Some((pages.1, ring_at..ring_at + ring_len, "a delta conv ring")),
                    ]
                }
            };
            for (buf, range, what) in stores.into_iter().flatten() {
                // SAFETY: `quiesce` above left the stream idle (no token
                // ahead, every committed buffer done), and the views borrow
                // the arena and the rows arena, whose device work and host
                // writes all go through `&mut self`, until the take below has
                // copied them: nothing writes the pages while they are read.
                let pages = unsafe { buf.pages().as_slice() };
                views.push(
                    pages
                        .get(range)
                        .ok_or_else(|| refuse(format!("{what} lies past its buffer")))?,
                );
            }
        }
        match self.checkpoints.as_mut() {
            Some(c) => c.take(at, &views).map_err(checkpoint_error),
            // No recurrent store: the caches hold the position as they are.
            None if has_recurrent_store(&self.arena) => Err(no_checkpoints()),
            None => Ok(()),
        }
    }

    /// Cut to position `n`. A cut to 0 resets, which drops the checkpoints. A
    /// cut to the position does nothing. A cut to a checkpoint waits out the
    /// token ahead, copies that position's state lanes and conv rings back
    /// once, from the point's slot straight into the stores, and stands at
    /// `n`. Any other `n` is the ledger's refusal, by name, after the token
    /// ahead is dropped: the position and every store stay. A copy back that
    /// fails resets the session before the error returns. Refused while the
    /// session is poisoned.
    pub fn cut(&mut self, n: u32) -> Result<(), SessionError> {
        if let Some(fault) = self.poisoned {
            return Err(SessionError::Poisoned(fault));
        }
        if n == 0 {
            return self.reset();
        }
        if n == self.pos {
            return Ok(());
        }
        self.quiesce()?;
        let lane_len = held_lane_len(&self.arena)?;
        let start = lane_of(n)
            .checked_mul(lane_len)
            .ok_or_else(|| refuse("a delta state lane's offset overflowed"))?;
        let recurrent = has_recurrent_store(&self.arena);
        // The lane `lane_of(n)` of each delta state and the whole ring of
        // each delta and short conv layer. The other lane is the one the next
        // token writes, so it is left as it is.
        let mut dst = Vec::new();
        for layer in &mut self.arena.layers {
            match &mut layer.mixer {
                MixerBufs::Attention(_) => {}
                MixerBufs::Delta(d) => {
                    dst.push(Restore {
                        buf: &mut d.state,
                        at: Elems(start),
                    });
                    dst.push(Restore {
                        buf: &mut d.ring,
                        at: Elems(0),
                    });
                }
                MixerBufs::ShortConv(c) => dst.push(Restore {
                    buf: &mut c.ring,
                    at: Elems(0),
                }),
            }
        }
        let cut = match self.checkpoints.as_mut() {
            Some(c) => c.cut(n, self.pos, &mut dst),
            // A recurrent store with no checkpoints would keep a position
            // whose store no point holds.
            None if recurrent => return Err(no_checkpoints()),
            // No recurrent store: the caches hold every position below `n`,
            // and the next step's append overwrites the rows past it.
            None if n < self.pos => Ok(()),
            None => {
                return Err(refuse(format!(
                    "a cut to {n}, past the position {}",
                    self.pos
                )));
            }
        };
        match cut {
            Ok(()) => {}
            // A copy that stopped midway leaves some layers at `n` and the
            // rest where they were: reset, so no step runs on mixed state.
            Err(e @ checkpoint::Error::Device(_)) => {
                let e = checkpoint_error(e);
                return Err(match self.reset() {
                    Ok(()) => e,
                    Err(r) => refuse(format!("{e}; and the reset after it failed: {r}")),
                });
            }
            Err(e) => return Err(checkpoint_error(e)),
        }
        self.pos = n;
        self.drop_copies();
        Ok(())
    }

    /// The token ahead waited out and dropped, then the stream settled: a
    /// take or a restore copies shared pages that no device work writes.
    fn quiesce(&mut self) -> Result<(), SessionError> {
        if let Some(a) = self.ahead.take() {
            self.drop_ahead(a)?;
        }
        self.stream.sync()?;
        Ok(())
    }
}

fn checkpoint_error(e: checkpoint::Error) -> SessionError {
    match e {
        checkpoint::Error::Device(e) => SessionError::Device(e),
        other => refuse(other.to_string()),
    }
}

/// f32s of one delta state lane at `g`.
fn lane_len(g: &Geometry) -> Result<usize, SessionError> {
    g.delta
        .state_elems(1)
        .ok_or_else(|| refuse("a delta state lane's length overflowed"))
}

/// What one layer holds that a checkpoint copies, by the lengths of its
/// buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Held {
    /// An attention layer: its caches hold every position below the one the
    /// session stands at.
    Caches,
    /// A delta layer: its state (both lanes) and its conv ring.
    Delta { state: usize, ring: usize },
    /// A short conv layer: its ring.
    Conv { ring: usize },
}

impl Held {
    fn of(layer: &LayerBufs) -> Held {
        match &layer.mixer {
            MixerBufs::Attention(_) => Held::Caches,
            MixerBufs::Delta(d) => Held::Delta {
                state: d.state.pages().len().get(),
                ring: d.ring.pages().len().get(),
            },
            MixerBufs::ShortConv(c) => Held::Conv {
                ring: c.ring.pages().len().get(),
            },
        }
    }
}

/// Whether any layer of `arena` holds a recurrent store: the file has
/// checkpoints exactly then.
fn has_recurrent_store(arena: &Arena) -> bool {
    arena
        .layers
        .iter()
        .any(|layer| Held::of(layer) != Held::Caches)
}

/// The refusal of a file with a recurrent store and no checkpoints.
fn no_checkpoints() -> SessionError {
    refuse("a file with a recurrent store holds no checkpoints")
}

/// f32s of one delta state lane of the file's delta layers: none for a file
/// with no delta layer, whose geometry holds no lane.
fn held_lane_len(arena: &Arena) -> Result<usize, SessionError> {
    if arena
        .layers
        .iter()
        .any(|layer| matches!(layer.mixer, MixerBufs::Delta(_)))
    {
        lane_len(&arena.geom)
    } else {
        Ok(0)
    }
}

/// The f32s of each store a checkpoint copies, in the order every layer lists
/// them: a delta layer's state lane then its ring, a short conv layer's ring.
/// None for a file with no recurrent store. The state buffer is both lanes; a
/// checkpoint copies one.
fn store_lens_of(
    layers: impl IntoIterator<Item = Held>,
    lane_len: usize,
) -> Result<Vec<usize>, SessionError> {
    let both = lane_len
        .checked_mul(LANES)
        .ok_or_else(|| refuse("the delta state's length overflowed"))?;
    let mut lens = Vec::new();
    for held in layers {
        match held {
            Held::Caches => {}
            Held::Delta { state, ring } => {
                if state != both {
                    return Err(refuse(format!(
                        "a delta state of {state} f32s; {LANES} lanes of {lane_len} are {both}"
                    )));
                }
                if ring == 0 {
                    return Err(refuse("a conv ring of no f32s"));
                }
                lens.push(lane_len);
                lens.push(ring);
            }
            Held::Conv { ring } => {
                if ring == 0 {
                    return Err(refuse("a short conv ring of no f32s"));
                }
                lens.push(ring);
            }
        }
    }
    Ok(lens)
}

/// [`store_lens_of`] over the arena's layers.
fn store_lens(arena: &Arena) -> Result<Vec<usize>, SessionError> {
    store_lens_of(arena.layers.iter().map(Held::of), held_lane_len(arena)?)
}

/// The traced step's ops: each op of [`MetalOps`] on its own command buffer,
/// waited, then the hook.
struct Tracer<'s, H> {
    stream: &'s MetalStream,
    kernels: &'s Kernels,
    arena: &'s mut Arena,
    pos: u32,
    count: EncodeCount,
    /// The last op's settle: the head's, once the walk is done.
    settled: Option<Settled>,
    hook: H,
}

impl<H> Tracer<'_, H>
where
    H: FnMut(Op, &Probe<'_>) -> Result<(), SessionError>,
{
    fn one(
        &mut self,
        op: Op,
        f: impl FnOnce(&mut MetalOps<'_, '_>) -> Result<(), SessionError>,
    ) -> Result<(), SessionError> {
        let mut recording = self.stream.begin()?;
        let mut enc = recording.token_encoder()?;
        f(&mut MetalOps::new(
            &mut enc,
            self.arena,
            self.kernels,
            self.pos,
            Feed::Host,
            true,
        ))?;
        let c = enc.count();
        enc.end();
        let signalled = self.stream.submit_signalled(recording)?;
        self.settled = Some(self.stream.settle(signalled)?);
        self.count += c;
        let probe = Probe {
            arena: self.arena,
            lane: lane_of(self.pos + 1),
        };
        (self.hook)(op, &probe)
    }
}

impl<H> ChainOps for Tracer<'_, H>
where
    H: FnMut(Op, &Probe<'_>) -> Result<(), SessionError>,
{
    type Error = SessionError;

    fn embed(&mut self, m: usize) -> Result<(), SessionError> {
        self.one(Op::Embed, |o| o.embed(m))
    }

    fn attn_in(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), SessionError> {
        self.one(Op::AttnIn(l.layer), |o| o.attn_in(l, g))
    }

    fn attn_core(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), SessionError> {
        self.one(Op::AttnCore(l.layer), |o| o.attn_core(l, g))
    }

    fn attn_out(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), SessionError> {
        self.one(Op::AttnOut(l.layer), |o| o.attn_out(l, g))
    }

    fn delta_in(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), SessionError> {
        self.one(Op::DeltaIn(l.layer), |o| o.delta_in(l, d))
    }

    fn delta_core(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), SessionError> {
        self.one(Op::DeltaCore(l.layer), |o| o.delta_core(l, d))
    }

    fn delta_out(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), SessionError> {
        self.one(Op::DeltaOut(l.layer), |o| o.delta_out(l, d))
    }

    fn conv_in(&mut self, l: LayerCtx<'_>, c: &ShortConvPlan) -> Result<(), SessionError> {
        self.one(Op::ConvIn(l.layer), |o| o.conv_in(l, c))
    }

    fn conv_core(&mut self, l: LayerCtx<'_>, c: &ShortConvPlan) -> Result<(), SessionError> {
        self.one(Op::ConvCore(l.layer), |o| o.conv_core(l, c))
    }

    fn conv_out(&mut self, l: LayerCtx<'_>, c: &ShortConvPlan) -> Result<(), SessionError> {
        self.one(Op::ConvOut(l.layer), |o| o.conv_out(l, c))
    }

    fn route(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), SessionError> {
        if matches!(f.route, FfnRoute::Dense) {
            // No router, so no group of its own.
            return Ok(());
        }
        self.one(Op::Route(l.layer), |o| o.route(l, f))
    }

    fn experts(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), SessionError> {
        self.one(Op::Experts(l.layer), |o| o.experts(l, f))
    }

    fn head(&mut self, m: usize) -> Result<(), SessionError> {
        self.one(Op::Head, |o| o.head(m))
    }
}

/// The pass's ops: [`MetalOps`]'s rows, encoded a layer to a command buffer.
/// The embed rides with layer 0 and the head with the last layer, so the
/// pass commits exactly its layer count of buffers and the longest is one
/// layer's work.
struct PassLayers<'s> {
    buffers: LayerBuffers<'s>,
    kernels: &'s Kernels,
    arena: &'s mut Arena,
    rows: &'s mut RowsArena,
    /// The rows the delta layers copy their stores after.
    after: &'s [u32],
    from: u32,
    head: bool,
}

impl PassLayers<'_> {
    /// `f` over the ops of the buffer `layer` encodes on. The embed and the
    /// head name no layer: they encode on the buffer open.
    fn one(
        &mut self,
        layer: Option<usize>,
        f: impl FnOnce(&mut MetalOps<'_, '_>) -> Result<(), SessionError>,
    ) -> Result<(), SessionError> {
        let open = self.buffers.at(layer)?;
        open.encoder(|encoder| {
            f(&mut MetalOps::rows(
                encoder,
                self.arena,
                RowsCtx {
                    arena: self.rows,
                    after: self.after,
                },
                self.kernels,
                self.from,
                self.head,
            ))
        })
    }
}

impl ChainOps for PassLayers<'_> {
    type Error = SessionError;

    fn embed(&mut self, m: usize) -> Result<(), SessionError> {
        self.one(None, |o| o.embed(m))
    }

    fn attn_in(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), SessionError> {
        self.one(Some(l.layer), |o| o.attn_in(l, g))
    }

    fn attn_core(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), SessionError> {
        self.one(Some(l.layer), |o| o.attn_core(l, g))
    }

    fn attn_out(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), SessionError> {
        self.one(Some(l.layer), |o| o.attn_out(l, g))
    }

    fn delta_in(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), SessionError> {
        self.one(Some(l.layer), |o| o.delta_in(l, d))
    }

    fn delta_core(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), SessionError> {
        self.one(Some(l.layer), |o| o.delta_core(l, d))
    }

    fn delta_out(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), SessionError> {
        self.one(Some(l.layer), |o| o.delta_out(l, d))
    }

    fn conv_in(&mut self, l: LayerCtx<'_>, c: &ShortConvPlan) -> Result<(), SessionError> {
        self.one(Some(l.layer), |o| o.conv_in(l, c))
    }

    fn conv_core(&mut self, l: LayerCtx<'_>, c: &ShortConvPlan) -> Result<(), SessionError> {
        self.one(Some(l.layer), |o| o.conv_core(l, c))
    }

    fn conv_out(&mut self, l: LayerCtx<'_>, c: &ShortConvPlan) -> Result<(), SessionError> {
        self.one(Some(l.layer), |o| o.conv_out(l, c))
    }

    fn route(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), SessionError> {
        if matches!(f.route, FfnRoute::Dense) {
            // No router, so no group of its own.
            return Ok(());
        }
        self.one(Some(l.layer), |o| o.route(l, f))
    }

    fn experts(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), SessionError> {
        self.one(Some(l.layer), |o| o.experts(l, f))
    }

    fn head(&mut self, m: usize) -> Result<(), SessionError> {
        self.one(None, |o| o.head(m))
    }
}

/// The command buffers of a pass, one a layer: the buffer being encoded, and
/// the signal of the last one committed. A buffer is committed with a signal
/// and not waited when the walk reaches the next layer, so the host encodes a
/// layer while the card runs the one before it; each buffer opens at the
/// stream's token boundary, which the buffer ahead signals at its end, so it
/// sees what that one wrote. The pass settles the last signal once.
struct LayerBuffers<'s> {
    stream: &'s MetalStream,
    /// The buffer being encoded, not yet committed.
    open: Option<OpenLayer<'s>>,
    /// The last committed buffer's signal.
    last: Option<Signalled>,
    /// What the committed buffers encoded.
    count: EncodeCount,
}

/// A layer's buffer being encoded.
struct OpenLayer<'s> {
    /// The layer its ops name; the embed opens a buffer before one does.
    layer: Option<usize>,
    recording: TokenRecording<'s>,
}

impl<'s> LayerBuffers<'s> {
    fn new(stream: &'s MetalStream) -> Self {
        Self {
            stream,
            open: None,
            last: None,
            count: EncodeCount::default(),
        }
    }

    /// The buffer an op of `layer` encodes on. The open one when it is
    /// `layer`'s, or when either names no layer; otherwise the open buffer is
    /// committed and the next one opened.
    fn at(&mut self, layer: Option<usize>) -> Result<&mut TokenRecording<'s>, SessionError> {
        let open = match self.open.take() {
            Some(mut open) if layer.is_none() || open.layer.is_none() || open.layer == layer => {
                open.layer = open.layer.or(layer);
                open
            }
            behind => {
                if let Some(behind) = behind {
                    self.commit(behind)?;
                }
                OpenLayer {
                    layer,
                    recording: self.stream.begin()?.open_token_encoder()?,
                }
            }
        };
        Ok(&mut self.open.insert(open).recording)
    }

    /// Commit `open` with a signal, unwaited.
    fn commit(&mut self, open: OpenLayer<'s>) -> Result<(), SessionError> {
        let (recording, count) = open.recording.finish();
        // A buffer's own failure is named at the last signal's settle: the
        // driver reports a failed buffer there, whichever buffer it was.
        self.last = Some(self.stream.submit_signalled(recording)?);
        self.count += count;
        Ok(())
    }

    /// Commit the buffer still open, the walk done.
    fn finish(&mut self) -> Result<(), SessionError> {
        match self.open.take() {
            Some(open) => self.commit(open),
            None => Ok(()),
        }
    }
}

/// The profiled pass's ops: each op of [`MetalOps`]'s rows arms on a
/// command buffer of its own, waited, its GPU seconds read and one sample
/// appended. A buffer opens at the stream's token boundary, so each op sees
/// what the one before it wrote.
struct PassProfiler<'s> {
    stream: &'s MetalStream,
    kernels: &'s Kernels,
    arena: &'s mut Arena,
    rows: &'s mut RowsArena,
    /// The rows the delta layers copy their stores after.
    after: &'s [u32],
    from: u32,
    head: bool,
    count: EncodeCount,
    /// The last op's settle: the head's, once the walk is done.
    settled: Option<Settled>,
    samples: &'s mut Vec<PassSample>,
}

impl PassProfiler<'_> {
    fn one(
        &mut self,
        op: Op,
        f: impl FnOnce(&mut MetalOps<'_, '_>) -> Result<(), SessionError>,
    ) -> Result<(), SessionError> {
        let mut recording = self.stream.begin()?;
        let mut enc = recording.token_encoder()?;
        f(&mut MetalOps::rows(
            &mut enc,
            self.arena,
            RowsCtx {
                arena: self.rows,
                after: self.after,
            },
            self.kernels,
            self.from,
            self.head,
        ))?;
        let c = enc.count();
        enc.end();
        let stamps = recording.stamps();
        let signalled = self.stream.submit_signalled(recording)?;
        self.settled = Some(self.stream.settle(signalled)?);
        let gpu_s = match stamps.interval(STAMP_WAIT)? {
            GpuInterval::Ran(s) => s,
            GpuInterval::StillRunning => {
                return Err(refuse(format!(
                    "{op:?}: its buffer still runs after its settle"
                )));
            }
        };
        self.samples.push(PassSample { op, gpu_s });
        self.count += c;
        Ok(())
    }
}

impl ChainOps for PassProfiler<'_> {
    type Error = SessionError;

    fn embed(&mut self, m: usize) -> Result<(), SessionError> {
        self.one(Op::Embed, |o| o.embed(m))
    }

    fn attn_in(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), SessionError> {
        self.one(Op::AttnIn(l.layer), |o| o.attn_in(l, g))
    }

    fn attn_core(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), SessionError> {
        self.one(Op::AttnCore(l.layer), |o| o.attn_core(l, g))
    }

    fn attn_out(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), SessionError> {
        self.one(Op::AttnOut(l.layer), |o| o.attn_out(l, g))
    }

    fn delta_in(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), SessionError> {
        self.one(Op::DeltaIn(l.layer), |o| o.delta_in(l, d))
    }

    fn delta_core(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), SessionError> {
        self.one(Op::DeltaCore(l.layer), |o| o.delta_core(l, d))
    }

    fn delta_out(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), SessionError> {
        self.one(Op::DeltaOut(l.layer), |o| o.delta_out(l, d))
    }

    fn conv_in(&mut self, l: LayerCtx<'_>, c: &ShortConvPlan) -> Result<(), SessionError> {
        self.one(Op::ConvIn(l.layer), |o| o.conv_in(l, c))
    }

    fn conv_core(&mut self, l: LayerCtx<'_>, c: &ShortConvPlan) -> Result<(), SessionError> {
        self.one(Op::ConvCore(l.layer), |o| o.conv_core(l, c))
    }

    fn conv_out(&mut self, l: LayerCtx<'_>, c: &ShortConvPlan) -> Result<(), SessionError> {
        self.one(Op::ConvOut(l.layer), |o| o.conv_out(l, c))
    }

    fn route(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), SessionError> {
        if matches!(f.route, FfnRoute::Dense) {
            // No router, so no group of its own.
            return Ok(());
        }
        self.one(Op::Route(l.layer), |o| o.route(l, f))
    }

    fn experts(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), SessionError> {
        self.one(Op::Experts(l.layer), |o| o.experts(l, f))
    }

    fn head(&mut self, m: usize) -> Result<(), SessionError> {
        self.one(Op::Head, |o| o.head(m))
    }
}

/// The profiled step's groups: each group of [`MetalOps`] on its own command
/// buffer, waited, its GPU interval read, one sample appended. The groups
/// are the token's as the encoder counts them, so the walk's `experts` is two
/// groups (the gate-up and the down), the head is two (the logits and the
/// pick), and on a sandwich every half's out and the down are each followed by
/// a join group.
struct Profiler<'s> {
    stream: &'s MetalStream,
    kernels: &'s Kernels,
    arena: &'s mut Arena,
    pos: u32,
    count: EncodeCount,
    /// The last group's settle: the head's, once the walk is done.
    settled: Option<Settled>,
    out: &'s mut Vec<ProfileGroup>,
}

impl Profiler<'_> {
    /// The join group of `half` on a sandwich layer, the hand-on group on a
    /// plain residual that hands on; a plain residual that does not has
    /// neither, and no group is sampled.
    fn join(&mut self, l: LayerCtx<'_>, half: Half) -> Result<(), SessionError> {
        let kind = match l.plan.residual {
            ResidualPlan::Sandwich(_) => GroupKind::Join,
            ResidualPlan::Plain if self.arena.hands_on(l.layer) => GroupKind::HandOn,
            ResidualPlan::Plain => return Ok(()),
        };
        self.one(kind, Some(l.layer), |o| o.join(l, half))
    }

    fn one(
        &mut self,
        kind: GroupKind,
        layer: Option<usize>,
        f: impl FnOnce(&mut MetalOps<'_, '_>) -> Result<(), SessionError>,
    ) -> Result<(), SessionError> {
        let mut recording = self.stream.begin()?;
        let mut enc = recording.token_encoder()?;
        f(&mut MetalOps::new(
            &mut enc,
            self.arena,
            self.kernels,
            self.pos,
            Feed::Host,
            true,
        ))?;
        let c = enc.count();
        enc.end();
        let stamps = recording.stamps();
        let signalled = self.stream.submit_signalled(recording)?;
        let settled = self.stream.settle(signalled)?;
        self.settled = Some(settled);
        self.count += c;
        // The settle above waited the buffer out, so its stamps are the
        // driver's record of a completed run; anything else is a refusal.
        let gpu = match stamps.interval(STAMP_WAIT)? {
            GpuInterval::Ran(s) => s,
            GpuInterval::StillRunning => {
                return Err(refuse(format!(
                    "{kind:?} layer {layer:?}: its buffer still runs after its settle"
                )));
            }
        };
        self.out.push(ProfileGroup {
            kind,
            layer,
            gpu_s: gpu,
        });
        Ok(())
    }
}

impl ChainOps for Profiler<'_> {
    type Error = SessionError;

    fn embed(&mut self, m: usize) -> Result<(), SessionError> {
        self.one(GroupKind::Embed, None, |o| o.embed(m))
    }

    fn attn_in(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), SessionError> {
        self.one(GroupKind::AttnP1, Some(l.layer), |o| o.attn_in(l, g))
    }

    fn attn_core(&mut self, l: LayerCtx<'_>, _: &GqaPlan) -> Result<(), SessionError> {
        self.one(GroupKind::AttnFlash, Some(l.layer), |o| {
            o.attn_flash(l.layer)
        })
    }

    fn attn_out(&mut self, l: LayerCtx<'_>, _: &GqaPlan) -> Result<(), SessionError> {
        self.one(GroupKind::AttnOut, Some(l.layer), |o| o.attn_project(l))?;
        self.join(l, Half::Mixer)
    }

    fn delta_in(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), SessionError> {
        self.one(GroupKind::DeltaP1, Some(l.layer), |o| o.delta_in(l, d))
    }

    fn delta_core(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), SessionError> {
        self.one(GroupKind::DeltaCore, Some(l.layer), |o| o.delta_core(l, d))
    }

    fn delta_out(&mut self, l: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), SessionError> {
        self.one(GroupKind::DeltaOut, Some(l.layer), |o| o.delta_project(l))?;
        self.join(l, Half::Mixer)
    }

    fn conv_in(&mut self, l: LayerCtx<'_>, c: &ShortConvPlan) -> Result<(), SessionError> {
        self.one(GroupKind::ConvP1, Some(l.layer), |o| o.conv_in(l, c))
    }

    fn conv_core(&mut self, l: LayerCtx<'_>, c: &ShortConvPlan) -> Result<(), SessionError> {
        self.one(GroupKind::ConvCore, Some(l.layer), |o| o.conv_core(l, c))
    }

    fn conv_out(&mut self, l: LayerCtx<'_>, _: &ShortConvPlan) -> Result<(), SessionError> {
        self.one(GroupKind::ConvOut, Some(l.layer), |o| o.conv_project(l))?;
        self.join(l, Half::Mixer)
    }

    fn route(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), SessionError> {
        if matches!(f.route, FfnRoute::Dense) {
            // No router, so no group of its own.
            return Ok(());
        }
        self.one(GroupKind::Router, Some(l.layer), |o| o.route(l, f))
    }

    fn experts(&mut self, l: LayerCtx<'_>, _: &FfnPlan) -> Result<(), SessionError> {
        self.one(GroupKind::GateUp, Some(l.layer), |o| {
            o.experts_pair(l.layer)
        })?;
        self.one(GroupKind::Down, Some(l.layer), |o| o.experts_down(l))?;
        self.join(l, Half::Ffn)
    }

    fn head(&mut self, m: usize) -> Result<(), SessionError> {
        self.one(GroupKind::Head, None, |o| o.head_logits(m))?;
        self.one(GroupKind::HeadPick, None, |o| o.head_pick(m))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::REGISTERED;
    use chain::linear::Ring;

    /// A session feeds its prompt calls in passes at every registered
    /// instance: the rows arm runs a layer that has a shared expert and one
    /// that has none.
    #[test]
    fn every_registered_instance_opens_in_passes() {
        for inst in REGISTERED {
            assert_eq!(
                prefill_at_open(&Geometry::of(inst)),
                PrefillMode::Batch,
                "{}",
                inst.name
            );
        }
    }

    /// A dense FFN's one slot has its own rows arm: its instance feeds a
    /// prompt call in passes, whatever the router the program holds.
    #[test]
    fn a_dense_instance_opens_in_passes() {
        let mut inst = crate::instance::QWEN35MOE_A3B;
        inst.key.ffn = crate::instance::FfnAxes::Dense { ff: 512 };
        let g = Geometry::of(&inst);
        assert!(g.experts.dense);
        assert_eq!(prefill_at_open(&g), PrefillMode::Batch);
        // A pass of the dense instance holds the family's most rows: its one
        // slot a row.
        assert_eq!(pass_rows_max(&g), CHECKPOINT_EVERY as usize);
    }

    /// A short conv layer's ring is a checkpointed store: a file of conv and
    /// attention layers lists one store per conv layer, in layer order beside
    /// a delta layer's lane and ring, so the file has checkpoints and a cut
    /// restores its rings.
    ///
    /// Mutant: `store_lens_of` does nothing for `Held::Conv`: the list of the
    /// conv layout is empty, the file has no checkpoints, and a cut keeps a
    /// position whose rings nobody restored (`q_mark_cut`'s conv row).
    #[test]
    fn a_short_conv_ring_is_a_checkpointed_store() {
        let ring = Ring::new(7).values(2048);
        let lfm3 = "ccFcScFcScFcScFcScFcScFc".chars().map(|c| {
            if c == 'c' {
                Held::Conv { ring }
            } else {
                Held::Caches
            }
        });
        assert_eq!(store_lens_of(lfm3, 0).unwrap(), vec![ring; 13]);
        let lane = 4096;
        let mixed = [
            Held::Delta {
                state: LANES * lane,
                ring: 99,
            },
            Held::Caches,
            Held::Conv { ring },
        ];
        assert_eq!(store_lens_of(mixed, lane).unwrap(), vec![lane, 99, ring]);
    }

    /// A file whose layers all hold caches lists no store and so has no
    /// checkpoints: its caches hold every position below the one it stands
    /// at.
    ///
    /// Mutant: `store_lens_of` pushes a store for `Held::Caches`.
    #[test]
    fn a_file_of_caches_lists_no_store() {
        assert!(store_lens_of([Held::Caches; 4], 0).unwrap().is_empty());
    }

    /// A ring of no values, and a delta state that is not both lanes, are
    /// refused by name: a store the checkpoint could not copy is no store.
    ///
    /// Mutant: delete the `ring == 0` refusal of `Held::Conv`, or the
    /// `state != both` refusal of `Held::Delta`.
    #[test]
    fn a_store_that_cannot_be_copied_is_refused_by_name() {
        let refusal = |held| store_lens_of([held], 8).unwrap_err().to_string();
        assert!(refusal(Held::Conv { ring: 0 }).contains("short conv ring of no f32s"));
        assert!(refusal(Held::Delta { state: 8, ring: 4 }).contains("2 lanes of 8 are 16"));
    }

    /// A geometry whose slots are not the picks and the shared slot the
    /// router's row says feeds a step each.
    #[test]
    fn a_geometry_the_rows_arm_does_not_take_opens_in_steps() {
        for inst in REGISTERED {
            let mut g = Geometry::of(inst);
            g.experts.slots += 1;
            assert_eq!(prefill_at_open(&g), PrefillMode::Steps, "{}", inst.name);
        }
    }

    /// A prompt call's inner checkpoints fall between passes: their spacing
    /// is the most rows of one pass.
    #[test]
    fn checkpoint_spacing_is_the_pass_rows() {
        for inst in REGISTERED {
            assert_eq!(
                pass_rows_max(&Geometry::of(inst)),
                CHECKPOINT_EVERY as usize,
                "{}",
                inst.name
            );
        }
    }
}
