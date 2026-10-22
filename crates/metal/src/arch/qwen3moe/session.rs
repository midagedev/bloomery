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
//! A raised fault poisons the session: the step waits out the token ahead,
//! returns the fault by name, decoded to its layer and sites, and every later
//! step is refused by name until [`Session::reset`].
//!
//! [`Session::step_traced`] is the instrument the gates read nodes through:
//! the same ops of a host-fed step, each on a command buffer of its own,
//! waited, with a hook after each. It encodes what [`Session::step`]
//! encodes, op for op.

use core::fmt;
use core::ops::Range;
use core::sync::atomic::Ordering;
use std::path::Path;

use bloomery_device::fault::Fault;
use bloomery_device::{DeviceError, Elem, Stream};
use chain::ChainError;
use chain::arch::qwen3moe::ops::{ChainOps, LayerCtx};
use chain::arch::qwen3moe::plan::{DeltaPlan, FfnPlan, GqaPlan, LayerPlan, kinds35};
use chain::arch::qwen3moe::read::{PlanDims, Read35, layer_plans};
use chain::arch::qwen3moe::walk::Walk;
use gguf::Split;

use super::arena::{Arena, MixerBufs, lane_of, slot_of};
use super::count::Layer;
use super::ops::{Feed, MetalOps};
use crate::buffer::MetalBuffer;
use crate::device::MetalDevice;
use crate::encode::EncodeCount;
use crate::kernels::{HEAD_DIM, Kernels, state_elems};
use crate::library::Library;
use crate::stream::{MetalStream, Settled, Signalled};

const WHAT: &str = "qwen35moe session";

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
    /// reset, a traced step.
    pub discarded: u64,
    /// Chain steps at the cache's last row, which commit nothing ahead.
    pub last_row: u64,
}

/// The token committed ahead of its call.
struct Ahead {
    signalled: Signalled,
    count: EncodeCount,
    pos: u32,
    /// The id its embed reads: the pick of the token before it.
    embeds: u32,
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
    /// The token at `pos`, committed by the last chain step.
    ahead: Option<Ahead>,
    stats: StepStats,
}

impl Session {
    /// The session of the file at `path` on `device`, its pipelines from
    /// `library`, with a cache of `ctx` rows: the plan read from the file's
    /// header, every weight copied into the arena, every store at rest.
    pub fn open(
        device: &MetalDevice,
        library: &Library,
        path: &Path,
        ctx: usize,
    ) -> Result<Session, SessionError> {
        let split = Split::open(path)
            .map_err(|e| refuse(format!("{} did not open: {e}", path.display())))?;
        let read = model::arch::qwen35moe::spec::read(&split)
            .map_err(|e| refuse(format!("{}: {e}", path.display())))?;
        let spec = read.spec;
        let kinds = kinds35(&spec.layers)?;
        let pd = PlanDims::q35(&spec, WHAT)?;
        let plans = layer_plans(&split, WHAT, &pd, &kinds, &spec.layers, Read35::Whole)?;
        let kernels = Kernels::from_library(library)?;
        let stream = device.new_stream()?;
        let arena = Arena::open(device.metal(), &split, &spec, &plans, ctx)?;
        if u32::try_from(arena.ctx).is_err() {
            return Err(refuse(format!(
                "a cache of {} rows: more than a u32 position names",
                arena.ctx
            )));
        }
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

    /// One token at the next position, `id` its input; the head's pick is
    /// returned. A token running ahead with this id is kept; any other is
    /// waited out and dropped first. In [`StepMode::Chain`] the next token is
    /// committed before this one is waited, unless this is the cache's last
    /// row. A call waits for this token alone, and on a broken promise for
    /// the dropped token before it.
    pub fn step(&mut self, id: u32, mode: StepMode) -> Result<Stepped, SessionError> {
        self.ready()?;
        let (current, count) = match self.ahead.take() {
            Some(a) if a.embeds == id => {
                self.stats.kept += 1;
                (a.signalled, a.count)
            }
            ahead => {
                if let Some(a) = ahead {
                    self.drop_ahead(a)?;
                }
                self.commit(id, self.pos, Feed::Host)?
            }
        };
        // On an error past this point the token at `pos` is waited for and
        // not landed: its fault word, if it raised, poisons the next step at
        // `pos`. The first error is the step's; a later one on the same path
        // is not returned.
        if mode == StepMode::Chain {
            let next = self.pos + 1;
            if (next as usize) < self.arena.ctx {
                match self.commit(id, next, Feed::Device) {
                    Ok((signalled, count)) => {
                        self.stats.ahead += 1;
                        self.ahead = Some(Ahead {
                            signalled,
                            count,
                            pos: next,
                            embeds: 0,
                        });
                    }
                    Err(e) => {
                        let _ = self.stream.settle(current);
                        return Err(e);
                    }
                }
            } else {
                self.stats.last_row += 1;
            }
        }
        let settled = match self.stream.settle(current) {
            Ok(settled) => settled,
            Err(e) => {
                if let Some(a) = self.ahead.take() {
                    let _ = self.drop_ahead(a);
                }
                return Err(e.into());
            }
        };
        let stepped = self.land(&settled, count);
        match (&stepped, self.ahead.as_mut()) {
            (Ok(s), Some(a)) => a.embeds = s.id,
            (Err(_), Some(_)) => self.wait_out_ahead(),
            _ => {}
        }
        stepped
    }

    /// Encode and commit the token at `pos`, signalled: its embed reads the
    /// host id slot, which `id` is written into first, or the record slot of
    /// the token before it.
    fn commit(
        &mut self,
        id: u32,
        pos: u32,
        feed: Feed,
    ) -> Result<(Signalled, EncodeCount), SessionError> {
        if feed == Feed::Host {
            self.arena.id_in.pages().store(0, id, Ordering::Relaxed)?;
        }
        let mut recording = self.stream.begin()?;
        let mut enc = recording.token_encoder()?;
        Walk::new(
            &mut MetalOps::new(&mut enc, &mut self.arena, &self.kernels, pos, feed),
            &self.plans,
            1,
        )
        .run(1)?;
        let count = enc.count();
        enc.end();
        Ok((self.stream.submit_signalled(recording)?, count))
    }

    /// The token ahead waited out and dropped, its fault word cleared if it
    /// raised: the token at its position is computed again, from the id the
    /// call names, and starts clean.
    fn drop_ahead(&mut self, a: Ahead) -> Result<(), SessionError> {
        self.stats.discarded += 1;
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
        self.land(&settled, count)
    }

    /// A settled token's end: its fault word (a fault poisons the session),
    /// then its pick; the position moves.
    fn land(&mut self, settled: &Settled, count: EncodeCount) -> Result<Stepped, SessionError> {
        let slot = slot_of(self.pos);
        if let Some(fault) = self.arena.faults[slot].read_settled(settled)? {
            self.poisoned = Some(fault);
            return Err(SessionError::Fault(fault));
        }
        let id = self.arena.step_id.pages().load(slot, Ordering::Relaxed)?;
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
            self.stream.settle(a.signalled)?;
        }
        self.stream.sync()?;
        self.arena.clear_stores()?;
        for fault in &self.arena.faults {
            fault.clear_waited()?;
        }
        self.pos = 0;
        self.poisoned = None;
        Ok(())
    }
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
