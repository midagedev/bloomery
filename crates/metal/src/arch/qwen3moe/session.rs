//! A qwen35moe session on Metal: the file opened into its plan, its arena and
//! its pipelines, then one token a step.
//!
//! A host-fed step ([`StepMode::HostFed`]) writes the token id into the host
//! id slot, encodes the token on one command buffer, commits it, waits for
//! it, and reads the token's fault word, then the head's pick. A chain step
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
//! and its whole conv ring. [`Session::checkpoint`] takes one at the position
//! the session stands at, with no token ahead and the stream idle.
//! [`Session::cut`] back to a point copies that lane and ring back and stands
//! there; a cut to 0 is a reset, which drops the points too. A cut the ledger
//! refuses leaves the position and every store as they were.
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

use bloomery_device::fault::Fault;
use bloomery_device::{DeviceError, Elem, Elems, Stream};
use chain::ChainError;
use chain::arch::qwen3moe::ops::{ChainOps, LayerCtx};
use chain::arch::qwen3moe::plan::{DeltaPlan, FfnPlan, GqaPlan, LayerPlan, SiteTy, kinds35};
use chain::arch::qwen3moe::read::{PlanDims, Read35, Sites35, layer_plans};
use chain::arch::qwen3moe::walk::Walk;
use gguf::Split;
use runtime::seqstate::{self, Kept};

use super::arena::{Arena, LANES, MixerBufs, lane_of, slot_of};
use super::count::{GroupBytes, GroupKind, Layer};
use super::ops::{Feed, MetalOps};
use crate::buffer::MetalBuffer;
use crate::checkpoint::{self, Checkpoints};
use crate::device::MetalDevice;
use crate::encode::EncodeCount;
use crate::kernels::{HEAD_DIM, Kernels, segments_for_keys, state_elems};
use crate::library::Library;
use crate::stream::{GpuInterval, GpuStamps, MetalStream, Settled, Signalled, host_secs};

const WHAT: &str = "qwen35moe session";

/// Positions between the inner checkpoints a prompt call takes
/// ([`runtime::seqstate::marks`]).
///
/// A point copies each delta layer's state lane and its conv ring.
/// Qwen3.6-35B-A3B has 30 delta layers of 16 key heads and 32 value heads:
/// a lane is 32·128·128 f32s (2_097_152 bytes) and a ring is
/// 11·(2·16 + 32)·128 f32s (360_448 bytes), so a point is
/// 30·(2_097_152 + 360_448) = 73_728_000 bytes. One copy every 256
/// positions is 288_000 bytes beside each position of the prompt, and a
/// cut between points re-feeds at most 255 positions.
pub const CHECKPOINT_EVERY: u32 = 256;

/// The site types Metal's mat-vecs launch: the kernels' quant set
/// ([`Quant::of_ggml`](crate::kernels::Quant::of_ggml), held equal to this
/// list by `arena`'s test) as families — Q4_K, Q6_K
/// and MLX_A4, the A4 kernels' affine 4-bit groups. The F32 tensors a session
/// reads as values (the norms, the conv, the dt bias, `ssm_a`) and the
/// router's MLX_A8, which its loader dequantizes exactly at open, are no
/// typed site; a type outside the list is refused by name at the site
/// (`file_site`).
pub(crate) const LAUNCHED: &[SiteTy] = &[SiteTy::Q4K, SiteTy::Q6K, SiteTy::MlxA4];

/// The site lists the plan reads: every list is [`LAUNCHED`].
pub const SITES: Sites35 = Sites35 {
    proj: LAUNCHED,
    stacks: LAUNCHED,
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

fn refuse(detail: impl Into<String>) -> SessionError {
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
    /// and waits for it, and nothing runs ahead.
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

/// What the session's steps did with the token ahead, and the host's waits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepStats {
    /// The host's waits on the session's stream: every settle and sync.
    pub waits: u64,
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
    /// The host's encode of a token, µs: the encoder's begin to the commit.
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
    /// The GPU clock's seconds minus the host clock's, as the start probe
    /// read it: the stamps' epoch is the GPU's own, and no system call
    /// names its difference from the host's.
    offset: f64,
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
    fn new(offset: f64) -> Self {
        Self {
            offset,
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
        row.start_s += (stamps.start - self.offset) - token.submitted;
        row.gpu_s += stamps.end - stamps.start;
        row.wake_s += wake - (stamps.end - self.offset);
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
                let lane = state_elems(1).ok_or_else(|| refuse("a state lane does not fit"))?;
                let at = self.lane * lane;
                read(&delta(l)?.state, at..at + lane)
            }
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
                let heads = cache.pages().len().get() / (a.ctx * HEAD_DIM);
                let mut out = Vec::with_capacity(heads * HEAD_DIM);
                for h in 0..heads {
                    let at = (h * a.ctx + pos) * HEAD_DIM;
                    let bits = read(cache, at..at + HEAD_DIM)?;
                    out.extend(bits.into_iter().map(gguf::quant::half_to_f32));
                }
                Ok(out)
            }
            Node::RouterLogits(l) => whole(&ffn(l)?.logits),
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
    stats: StepStats,
    /// The step clock, when a caller started one; `None` steps and times
    /// nothing.
    clock: Option<StepClock>,
    /// Each delta layer's state lane and conv ring, at the positions a
    /// prompt call takes.
    checkpoints: Checkpoints,
}

impl Session {
    /// The session of the file at `path` on `device`, its pipelines from
    /// `library`, with a cache of `ctx` rows: the plan read from the file's
    /// header, every weight copied into the arena, every store at rest.
    /// Checkpoints are taken every [`CHECKPOINT_EVERY`] positions.
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
        let split = Split::open(path)
            .map_err(|e| refuse(format!("{} did not open: {e}", path.display())))?;
        let read = model::arch::qwen35moe::spec::read(&split)
            .map_err(|e| refuse(format!("{}: {e}", path.display())))?;
        let spec = read.spec;
        let kinds = kinds35(&spec.layers)?;
        let pd = PlanDims::q35(&spec, WHAT)?;
        let plans = layer_plans(
            &split,
            WHAT,
            &pd,
            &kinds,
            &spec.layers,
            &SITES,
            Read35::Whole,
        )?;
        let kernels = Kernels::from_library(library)?;
        let stream = device.new_stream()?;
        let arena = Arena::open(device.metal(), &split, &spec, &plans, ctx)?;
        let ctx_positions = u32::try_from(arena.ctx).map_err(|_| {
            refuse(format!(
                "a cache of {} rows: more than a u32 position names",
                arena.ctx
            ))
        })?;
        let lens = store_lens(&arena)?;
        let checkpoints = Checkpoints::open(
            device.handle().clone(),
            lens,
            u64::from(ctx_positions),
            every,
        )
        .map_err(checkpoint_error)?;
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
            stats: StepStats::default(),
            clock: None,
            checkpoints,
        })
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

    /// What the steps did with the token ahead, and the host's waits so far.
    #[must_use]
    pub fn stats(&self) -> StepStats {
        StepStats {
            waits: self.stream.waits(),
            ..self.stats
        }
    }

    /// Start the step clock: every step from here on has its phases
    /// recorded, a row a mode, until the next start. The clock is a
    /// profiling run's instrument; it holds nothing back and clears nothing,
    /// not even at [`Session::reset`], which a caller that wants fresh rows
    /// starts again past. Its first act is the offset probe: an empty
    /// command buffer, settled, whose GPU stamps against the host clock at
    /// its settle fix the difference between the two clocks.
    pub fn start_step_clock(&mut self) -> Result<(), SessionError> {
        let mut recording = self.stream.begin()?;
        drop(recording.compute_encoder()?);
        let stamps = recording.stamps();
        let signalled = self.stream.submit_signalled(recording)?;
        self.stream.settle(signalled)?;
        let probe = stamps.times(STAMP_WAIT)?;
        let offset = probe.end - host_secs();
        self.clock = Some(StepClock::new(offset));
        Ok(())
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
        let clock = self.clock.as_ref();
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
        let stamps = clock.is_some().then(|| recording.stamps());
        let signalled = self.stream.submit_signalled(recording)?;
        if let (Some(clock), Some(stamps), Some(from)) = (clock, stamps, encode_from) {
            let at = host_secs();
            clock.committed(PendingToken {
                value: signalled.value(),
                stamps,
                encode_s: at - from,
                submitted: at,
            });
        }
        Ok((signalled, count))
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
        self.checkpoints.clear();
        Ok(())
    }

    /// The ledger's counts.
    #[must_use]
    pub fn checkpoint_stats(&self) -> seqstate::Stats {
        self.checkpoints.stats()
    }

    /// The checkpoints' positions, ascending.
    #[must_use]
    pub fn checkpoint_positions(&self) -> Vec<u32> {
        self.checkpoints.positions()
    }

    /// The bytes one checkpoint copies.
    #[must_use]
    pub fn checkpoint_bytes(&self) -> u64 {
        self.checkpoints.point_bytes()
    }

    /// The checkpoints the budget holds.
    #[must_use]
    pub fn checkpoint_capacity(&self) -> usize {
        self.checkpoints.capacity()
    }

    /// The spacing of a prompt call's inner checkpoints.
    #[must_use]
    pub fn checkpoint_every(&self) -> u32 {
        self.checkpoints.every()
    }

    /// The positions a prompt call from `from` to `to` takes a checkpoint at.
    #[must_use]
    pub fn checkpoint_marks(&self, from: u32, to: u32) -> Vec<u32> {
        self.checkpoints.marks(from, to)
    }

    /// What a cut to at most `n` keeps of the position the session stands at.
    #[must_use]
    pub fn kept(&self, n: u32) -> Kept {
        self.checkpoints.kept(n, self.pos)
    }

    /// A checkpoint at the position, the stream idle: the token ahead is
    /// waited out and dropped first. Refused while the session is poisoned.
    pub fn checkpoint(&mut self) -> Result<(), SessionError> {
        if let Some(fault) = self.poisoned {
            return Err(SessionError::Poisoned(fault));
        }
        self.quiesce()?;
        let owned = self.snapshot_stores()?;
        let refs: Vec<&[f32]> = owned.iter().map(Vec::as_slice).collect();
        self.checkpoints
            .take(self.pos, &refs)
            .map_err(checkpoint_error)
    }

    /// Cut to position `n`. A cut to 0 resets, which drops the checkpoints. A
    /// cut to the position does nothing. A cut to a checkpoint waits out the
    /// token ahead, copies that position's state lane and conv ring back, and
    /// stands at `n`. Any other `n` is the ledger's refusal, by name, after
    /// the token ahead is dropped: the position and every store stay. A copy
    /// back that fails resets the session before the error returns.
    /// Refused while the session is poisoned.
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
        let mut copied = vec_of(self.checkpoints.lens());
        {
            let mut refs: Vec<&mut [f32]> = copied.iter_mut().map(Vec::as_mut_slice).collect();
            self.checkpoints
                .cut(n, self.pos, &mut refs)
                .map_err(checkpoint_error)?;
        }
        if let Err(e) = self.install_stores(n, &copied) {
            // A copy that stopped midway leaves some layers at `n` and the
            // rest where they were: reset, so no step runs on mixed state.
            return Err(match self.reset() {
                Ok(()) => e,
                Err(r) => refuse(format!("{e}; and the reset after it failed: {r}")),
            });
        }
        self.pos = n;
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

    /// Each delta layer's state lane at the position's parity, then its ring.
    fn snapshot_stores(&self) -> Result<Vec<Vec<f32>>, SessionError> {
        let lane_len = lane_len()?;
        let start = lane_of(self.pos)
            .checked_mul(lane_len)
            .ok_or_else(|| refuse("a delta state lane's offset overflowed"))?;
        let mut out = Vec::new();
        for layer in &self.arena.layers {
            let MixerBufs::Delta(d) = &layer.mixer else {
                continue;
            };
            let mut lane = vec![0.0; lane_len];
            d.state.pages().read_at(Elems(start), &mut lane)?;
            out.push(lane);
            out.push(d.ring.pages().to_vec());
        }
        Ok(out)
    }

    /// `copied` back into lane `lane_of(n)` of each delta state, and into the
    /// whole ring. The other lane is the one the next token writes, so it is
    /// left as it is.
    fn install_stores(&mut self, n: u32, copied: &[Vec<f32>]) -> Result<(), SessionError> {
        let lane_len = lane_len()?;
        let start = lane_of(n)
            .checked_mul(lane_len)
            .ok_or_else(|| refuse("a delta state lane's offset overflowed"))?;
        let mut i = 0;
        for layer in &mut self.arena.layers {
            let MixerBufs::Delta(d) = &mut layer.mixer else {
                continue;
            };
            let lane_src = copied
                .get(i)
                .ok_or_else(|| refuse("a restore is short a state lane"))?;
            let ring_src = copied
                .get(i + 1)
                .ok_or_else(|| refuse("a restore is short a conv ring"))?;
            i += 2;
            if lane_src.len() != lane_len {
                return Err(refuse(format!(
                    "a restored state lane is {} f32s; a lane is {lane_len}",
                    lane_src.len()
                )));
            }
            d.state.write_at(Elems(start), lane_src)?;
            d.ring.write(ring_src)?;
        }
        if i != copied.len() {
            return Err(refuse(format!(
                "a restore of {} stores; the delta layers take {i}",
                copied.len()
            )));
        }
        Ok(())
    }
}

fn checkpoint_error(e: checkpoint::Error) -> SessionError {
    match e {
        checkpoint::Error::Device(e) => SessionError::Device(e),
        other => refuse(other.to_string()),
    }
}

/// f32s of one delta state lane.
fn lane_len() -> Result<usize, SessionError> {
    state_elems(1).ok_or_else(|| refuse("a delta state lane's length overflowed"))
}

/// One zeroed buffer per copied store.
fn vec_of(lens: &[usize]) -> Vec<Vec<f32>> {
    lens.iter().map(|&n| vec![0.0; n]).collect()
}

/// Each delta layer's state lane, then its conv ring. The state buffer is
/// both lanes; a checkpoint copies one.
fn store_lens(arena: &Arena) -> Result<Vec<usize>, SessionError> {
    let lane_len = lane_len()?;
    let both = lane_len
        .checked_mul(LANES)
        .ok_or_else(|| refuse("the delta state's length overflowed"))?;
    let mut lens = Vec::new();
    for layer in &arena.layers {
        let MixerBufs::Delta(d) = &layer.mixer else {
            continue;
        };
        let state_len = d.state.pages().len().get();
        if state_len != both {
            return Err(refuse(format!(
                "a delta state of {state_len} f32s; {LANES} lanes of {lane_len} are {both}"
            )));
        }
        let ring_len = d.ring.pages().len().get();
        if ring_len == 0 {
            return Err(refuse("a conv ring of no f32s"));
        }
        lens.push(lane_len);
        lens.push(ring_len);
    }
    if lens.is_empty() {
        return Err(refuse("a checkpoint copies no delta layer"));
    }
    Ok(lens)
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
        self.count.groups += c.groups;
        self.count.dispatches += c.dispatches;
        self.count.barriers += c.barriers;
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

    fn route(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), SessionError> {
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
/// are the token's as the encoder counts them, so the walk's `attn_core`
/// past one segment of keys and its `experts` are two groups each.
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
        self.count.groups += c.groups;
        self.count.dispatches += c.dispatches;
        self.count.barriers += c.barriers;
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
        let n_keys = self.pos + 1;
        let segments = segments_for_keys(n_keys);
        self.one(GroupKind::AttnFlash, Some(l.layer), |o| {
            o.attn_flash(l.layer, n_keys, segments)
        })
    }

    fn attn_out(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), SessionError> {
        self.one(GroupKind::AttnOut, Some(l.layer), |o| o.attn_out(l, g))
    }

    fn delta_in(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), SessionError> {
        self.one(GroupKind::DeltaP1, Some(l.layer), |o| o.delta_in(l, d))
    }

    fn delta_core(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), SessionError> {
        self.one(GroupKind::DeltaCore, Some(l.layer), |o| o.delta_core(l, d))
    }

    fn delta_out(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), SessionError> {
        self.one(GroupKind::DeltaOut, Some(l.layer), |o| o.delta_out(l, d))
    }

    fn route(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), SessionError> {
        self.one(GroupKind::Router, Some(l.layer), |o| o.route(l, f))
    }

    fn experts(&mut self, l: LayerCtx<'_>, _: &FfnPlan) -> Result<(), SessionError> {
        self.one(GroupKind::GateUp, Some(l.layer), |o| {
            o.experts_pair(l.layer)
        })?;
        self.one(GroupKind::Down, Some(l.layer), |o| o.experts_fold(l.layer))
    }

    fn head(&mut self, m: usize) -> Result<(), SessionError> {
        self.one(GroupKind::Head, None, |o| o.head(m))
    }
}
