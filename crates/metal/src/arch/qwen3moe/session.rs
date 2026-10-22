//! A qwen35moe session on Metal: the file opened into its plan, its arena and
//! its pipelines, then one token a step.
//!
//! A step is host-fed and waited: the host writes the token id into the id
//! slot, the walk encodes the token on one command buffer, and the step
//! commits it, waits for it, reads the fault word, then the head's pick. A
//! raised fault poisons the session: the step returns it by name, decoded to
//! its layer and sites, and every later step is refused by name until
//! [`Session::reset`].
//!
//! [`Session::step_traced`] is the instrument the gates read nodes through:
//! the same ops, each on a command buffer of its own, waited, with a hook
//! after each. It encodes what [`Session::step`] encodes, op for op.

use core::fmt;
use core::ops::Range;
use core::sync::atomic::Ordering;
use std::path::Path;

use bloomery_device::fault::{Fault, FaultWord};
use bloomery_device::{DeviceError, Elem, Stream};
use chain::ChainError;
use chain::arch::qwen3moe::ops::{ChainOps, LayerCtx};
use chain::arch::qwen3moe::plan::{
    DeltaPlan, FfnPlan, GqaPlan, Kind35, LayerPlan, MixerPlan, kinds35, q35,
};
use chain::arch::qwen3moe::read::{PlanDims, Read35, plan_delta, plan_ffn, plan_gqa};
use chain::arch::qwen3moe::walk::Walk;
use gguf::Split;
use model::arch::models::Mixer;

use super::arena::{Arena, MixerBufs};
use super::count::Layer;
use super::ops::MetalOps;
use crate::buffer::MetalBuffer;
use crate::device::MetalDevice;
use crate::encode::EncodeCount;
use crate::kernels::{HEAD_DIM, Kernels};
use crate::library::Library;
use crate::stream::MetalStream;

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
            Node::DeltaState(l) => whole(&delta(l)?.state),
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
        let (n_head, n_kv) = spec
            .layers
            .iter()
            .find_map(|l| match &l.mixer {
                Mixer::Gqa(g) => Some((g.heads as usize, g.kv_heads as usize)),
                _ => None,
            })
            .ok_or_else(|| refuse("no attention layer to size the attention rows for"))?;
        let head = q35::HEAD_DIM as usize;
        let pd = PlanDims {
            hidden: spec.hidden as usize,
            ff: q35::EXPERT_FF as usize,
            q_rows: 2 * n_head * head,
            kv_len: n_kv * head,
            attn_len: n_head * head,
        };
        let plans = kinds
            .iter()
            .zip(&spec.layers)
            .enumerate()
            .map(|(l, (kind, layer))| {
                let mixer = match *kind {
                    Kind35::Gqa(flash) => MixerPlan::Gqa(plan_gqa(&split, WHAT, &pd, l, flash)?),
                    Kind35::Delta(shape) => {
                        MixerPlan::Delta(plan_delta(&split, WHAT, &pd, shape, l)?)
                    }
                };
                Ok(LayerPlan {
                    mixer,
                    ffn: plan_ffn(&split, WHAT, &pd, layer, l, Read35::Whole)?,
                })
            })
            .collect::<Result<Vec<_>, ChainError>>()?;
        let kernels = Kernels::from_library(library)?;
        let stream = device.new_stream()?;
        let arena = Arena::open(device.metal(), &split, &spec, &plans, ctx)?;
        let layers = arena.token_layers();
        Ok(Session {
            stream,
            kernels,
            plans,
            arena,
            layers,
            pos: 0,
            poisoned: None,
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

    /// One token: `id` written into the id slot, the walk encoded on one
    /// command buffer, committed and waited; then the fault word and the
    /// head's pick.
    pub fn step(&mut self, id: u32) -> Result<Stepped, SessionError> {
        self.ready()?;
        self.arena.id_in.pages().store(0, id, Ordering::Relaxed)?;
        let mut recording = self.stream.begin()?;
        let mut enc = recording.compute_encoder_concurrent()?;
        Walk::new(
            &mut MetalOps::new(&mut enc, &mut self.arena, &self.kernels, self.pos),
            &self.plans,
            1,
        )
        .run(1)?;
        let count = enc.count();
        enc.end();
        self.stream.submit(recording)?;
        self.finish(count)
    }

    /// [`Session::step`] with every op on a command buffer of its own,
    /// waited, and `hook` called after each with what the op left. The count
    /// is the sum of the ops' counts: what one encoder of the whole token
    /// encodes. A hook's error ends the step and is returned; the session
    /// keeps the work done so far, and its position does not move.
    pub fn step_traced<H>(&mut self, id: u32, hook: H) -> Result<Stepped, SessionError>
    where
        H: FnMut(Op, &Probe<'_>) -> Result<(), SessionError>,
    {
        self.ready()?;
        self.arena.id_in.pages().store(0, id, Ordering::Relaxed)?;
        let mut tracer = Tracer {
            stream: &self.stream,
            kernels: &self.kernels,
            arena: &mut self.arena,
            pos: self.pos,
            count: EncodeCount::default(),
            hook,
        };
        Walk::new(&mut tracer, &self.plans, 1).run(1)?;
        let count = tracer.count;
        self.finish(count)
    }

    /// The step's end: wait, read the fault word (a fault poisons the
    /// session), read the pick, move the position.
    fn finish(&mut self, count: EncodeCount) -> Result<Stepped, SessionError> {
        self.stream.sync()?;
        if let Some(fault) = self.arena.fault.read(&self.stream)? {
            self.poisoned = Some(fault);
            return Err(SessionError::Fault(fault));
        }
        let id = self.arena.step_id.pages().load(0, Ordering::Relaxed)?;
        self.pos += 1;
        Ok(Stepped { id, count })
    }

    /// What the last step left, read as a traced step's hook reads it.
    pub fn probe(&self) -> Result<Probe<'_>, SessionError> {
        self.stream.sync()?;
        Ok(Probe { arena: &self.arena })
    }

    /// Every store back at rest: position 0, the caches, the recurrent
    /// states and rings, the tickets and the fault word cleared, the poison
    /// lifted.
    pub fn reset(&mut self) -> Result<(), SessionError> {
        self.stream.sync()?;
        self.arena.clear_stores();
        self.arena.fault.clear(&self.stream)?;
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
        let mut enc = recording.compute_encoder_concurrent()?;
        f(&mut MetalOps::new(
            &mut enc,
            self.arena,
            self.kernels,
            self.pos,
        ))?;
        let c = enc.count();
        enc.end();
        self.stream.submit(recording)?;
        self.stream.sync()?;
        self.count.groups += c.groups;
        self.count.dispatches += c.dispatches;
        self.count.barriers += c.barriers;
        (self.hook)(op, &Probe { arena: self.arena })
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
