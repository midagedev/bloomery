//! The prompt batch: a prompt of `P` ids fed in batches of at most
//! [`T_MAX`] positions ([`prefill`]), which leaves the model where `P`
//! decode steps over the same ids leave it, bit for bit, in everything a
//! later step reads: every KDA layer's state and conv ring, every latent
//! layer's latent and index rows and pool plane, the checkpoints, and the
//! last position's logits.
//!
//! A call is cut at its checkpoint marks
//! ([`bloomery_gpu::checkpoint::Checkpoints::marks`]: its start, every
//! multiple of [`CHECKPOINT_EVERY`](super::CHECKPOINT_EVERY) inside it, its
//! end), each run
//! between two marks into batches of at most [`T_MAX`] positions
//! ([`call_batches`]), and each mark's checkpoint is taken where the steps
//! take it, so a cut keeps the same points after either feed. A batch is one
//! walk of the runtime's layer schedule at the point `(1, T, Batch)`
//! ([`runtime::sched::walk`]) through the host tier's batch port
//! ([`BatchLeg`]); the walk serves only the layers with a host leg
//! ([`LayerProgram::host_leg`]: the dense lead downloads nothing). Per layer:
//! - the front: the mixer sub-layer whole and the feed-forward sub-layer's
//!   input, then the dense block and its `hc_post`, or the routed block's
//!   norm, its router over the batch's rows and the download of the rows,
//!   ids and weights to the host;
//! - the shadow (a routed layer): the card experts, where the slot map puts
//!   any, and the shared expert, while the host serves every token of the
//!   batch in one union call ([`GlmHost`]'s) and uploads the sums;
//! - the back: the host's sums plus the shadow's, and `hc_post`.
//!
//! The head runs after the last layer of the batch that holds the call's last
//! position, for that position alone.
//!
//! Every launch writes, per token, what the step's one-token launch writes,
//! and what carries state from a position to the next (the conv ring, the
//! delta rule's state, the latent rows an attention reads) runs in position
//! order:
//! - one launch over the batch's `T` rows where the kernel takes any count:
//!   the RMS norms, `hc_pre` (in its token groups), the fold and `hc_post`,
//!   the KDA conv and prep, the delta step, the gated norm, the latent and
//!   index appends, the pool keys the batch's tokens complete, the router's
//!   two launches, the card places, the sums;
//! - chunks of up to [`CHUNK`] tokens where it takes at most that many: the
//!   q8_0 gemvs (`q8_0_gemv_mcol`, `q8_0_gemv_heads_mcol`), the
//!   gate·up·SwiGLU (`ds41_shexp_gate_up_q8_0_mcol`), the card experts'
//!   launches, the k-pool selector, and the attention, each token over the
//!   positions at and before its own — the batch's own rows and pools
//!   written before the first chunk attends.
//!
//! A latent layer's chunk attends as the step does at its tokens'
//! positions: while its last token sees at most the positions the indexer
//! keeps whole (`place::dense_positions`), every position at and before
//! each token's, which is the list the selector gives there; past them the
//! selector runs over the chunk's tokens (`crate::mla::select`) and the
//! attention reads the positions each token's list names, a token of the
//! chunk still within them listing every one of its positions.
//!
//! A latent layer's joined projection runs as its two row ranges — the
//! query's low rank, then the latent, the index key and the pool gate — so
//! the query's norm reads whole rows; each row's dot is the joined launch's.
//!
//! A call that fails is taken back to where it found the model, through the
//! checkpoint of its start, unless a fault poisoned it. The fault word is
//! read at the end of every batch, before a mark's checkpoint can copy a
//! state the fault condemned.

use std::mem::ManuallyDrop;
use std::ops::Range;

use bloomery_gpu::head::Head;
use bloomery_gpu::host::BatchLeg;
use bloomery_gpu::kpool;
use bloomery_gpu::latent::{IndexKeyArgs, LATENT, LatentAppendArgs, Rows, pools_for};
use bloomery_gpu::linear::conv::KdaConvArgs;
use bloomery_gpu::linear::delta::{DeltaArgs, DeltaLanesArgs, KdaLanesArgs};
use bloomery_gpu::linear::norm_gate::NormGateArgs;
use bloomery_gpu::q8f32::{GemvOut, Q8_0GemvHeadsMcolArgs, Q8_0GemvMcolArgs};
use bloomery_gpu::qsa::list_width;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{COL_GROUP, DeviceTensor, Gpu, GpuError, Q8Act};
use bloomery_gpu_deepseek41::attn::{self, AttnArgs, SelectedRows};
use bloomery_gpu_deepseek41::hc::{
    HC_MAX_TOKENS, HC_MIX, HC_STREAMS, HcPostArgs, HcPreScratch, HcQ8Params, HcQ8PreArgs,
};
use bloomery_gpu_deepseek41::router::glm5next::{N_EXPERT, N_USED};
use bloomery_gpu_deepseek41::span::{span, span_mut};
use cuda_core::DeviceBuffer;
use gguf::GgmlType;
use gguf::quant::dequant_row;
use model::arch::glm5next::names::Sub;
use model::moe::UNION_MAX_COLS;
use runtime::layer::{FfnKind, MixerKind};
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};

use super::{
    Body, Dims, Embedding, Glm5nextModel, LANES, Parts, Store, f32t, f32v, prompt, q8, shape,
    weight,
};
use crate::ffn::{self, CardRows};
use crate::host::GlmHost;
use crate::mla::{self, Select};
use crate::tensors::{FfnNames, MixerNames, other_kind};

/// What the prompt batch's errors name.
const WHAT: &str = "glm5next prefill";

/// Positions one batch runs at most: the host union's columns.
pub const T_MAX: usize = UNION_MAX_COLS;

/// Tokens one chunk runs at most: the m-column kernels', and `hc_pre`'s
/// token group.
pub const CHUNK: usize = COL_GROUP;
const _: () = assert!(CHUNK == HC_MAX_TOKENS);

/// How a prompt is fed: in batches ([`prefill`]) or one decode step per id
/// ([`prompt`]) — the same-binary arm, which is the decode step and not a
/// second implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillMode {
    Batch,
    Steps,
}

impl PrefillMode {
    /// The mode [`PrefillMode::name`] names; `None` for any other word.
    #[must_use]
    pub fn from_name(name: &str) -> Option<PrefillMode> {
        [PrefillMode::Batch, PrefillMode::Steps]
            .into_iter()
            .find(|m| m.name() == name)
    }

    /// The name a `load` line prints and `--prefill` takes.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            PrefillMode::Batch => "batch",
            PrefillMode::Steps => "steps",
        }
    }
}

/// The body's prompt feed: its mode, and the batch's buffers once a batch
/// feed made them.
pub(crate) struct PromptState {
    mode: PrefillMode,
    batch: Option<Box<Batch>>,
}

impl PromptState {
    /// The steps feed with no buffers: what a load starts as.
    pub(crate) fn new() -> PromptState {
        PromptState {
            mode: PrefillMode::Steps,
            batch: None,
        }
    }

    /// Device bytes of the batch's buffers, 0 before they are made.
    pub(crate) fn bytes(&self) -> usize {
        self.batch.as_ref().map_or(0, |b| b.bytes())
    }
}

/// The batch's buffers: the walk's, and the host sums the batch port uploads
/// (lent to the port apart from them).
struct Batch {
    bufs: Bufs,
    hsum: DeviceBuffer<f32>,
}

impl Batch {
    fn bytes(&self) -> usize {
        self.bufs.bytes() + self.hsum.num_bytes()
    }
}

/// Every buffer a batch writes or reads besides the stores, for up to `cap`
/// tokens where a launch takes the batch whole and [`CHUNK`] where it takes
/// a chunk.
struct Bufs {
    cap: usize,
    /// The embedding rows on the host, four copies a token, before their
    /// upload.
    rows: Vec<f32>,
    /// Each token's position, its live count (the position plus one), and
    /// the attention's visible counts (no window row, then every position at
    /// and before its own), on the host.
    pos_host: Vec<u32>,
    cnt_host: Vec<u32>,
    vis_host: Vec<u32>,
    /// The four streams per token, ping-ponged as the step's.
    streams: [DeviceBuffer<f32>; 2],
    x: DeviceBuffer<f32>,
    xn: DeviceBuffer<f32>,
    out: DeviceBuffer<f32>,
    /// The fold `hc_post` writes beside the streams: never read.
    fold: DeviceBuffer<f32>,
    mixes: DeviceBuffer<f32>,
    hc: DeviceBuffer<f32>,
    hc_scratch: HcPreScratch,
    pos: DeviceBuffer<u32>,
    cnt: DeviceBuffer<u32>,
    vis: DeviceBuffer<u32>,
    // A KDA mixer's, per token.
    qkv: DeviceBuffer<f32>,
    conv: DeviceBuffer<f32>,
    fa: DeviceBuffer<f32>,
    ga: DeviceBuffer<f32>,
    beta_raw: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    f: DeviceBuffer<f32>,
    z: DeviceBuffer<f32>,
    decay: DeviceBuffer<f32>,
    o: DeviceBuffer<f32>,
    gated: DeviceBuffer<f32>,
    // A latent mixer's: the projections per token, the heads per chunk.
    qa: DeviceBuffer<f32>,
    kv: DeviceBuffer<f32>,
    qr: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    qabs: DeviceBuffer<f32>,
    att: DeviceBuffer<f32>,
    av: DeviceBuffer<f32>,
    part_v: DeviceBuffer<f32>,
    part_ms: DeviceBuffer<f32>,
    // The k-pool selector's, per chunk: the indexer query and head weights
    // (a token a column), the pools' scores and the lists.
    qi: DeviceBuffer<f32>,
    wi: DeviceBuffer<f32>,
    scores: DeviceBuffer<f32>,
    list: DeviceBuffer<u32>,
    // The feed-forward blocks': the SwiGLU rows per chunk, the rest per
    // token.
    h: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    sh_y: DeviceBuffer<f32>,
    acc: DeviceBuffer<f32>,
    pre: DeviceBuffer<f32>,
    probs: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
    sel: DeviceBuffer<u32>,
    // The card experts', per chunk: the rows' q8_1 form, the gate·up rows a
    // slot, per chunk width the q8_1 of its slots' columns, the downs.
    act_x: Q8Act,
    card_h: DeviceBuffer<f32>,
    act_h: Vec<Q8Act>,
    card_down: DeviceBuffer<f32>,
}

impl Bufs {
    /// The buffers for batches of up to `cap` tokens of `d`'s widths, `ff`
    /// the widest dense or shared-expert width, `expert_ff` a routed
    /// expert's, over stores of `ctx` positions. The attention's partials
    /// cover `ctx` keys: a chunk selects only on stores past the dense
    /// positions, which are a list's width, so they cover a list too.
    /// Load-time or first-prompt only.
    fn new(
        gpu: &Gpu,
        d: &Dims,
        ff: usize,
        expert_ff: usize,
        ctx: usize,
        cap: usize,
    ) -> Result<Bufs, GpuError> {
        let stream = gpu.stream();
        let z = |len: usize| DeviceBuffer::<f32>::zeroed(stream, len);
        let zu = |len: usize| DeviceBuffer::<u32>::zeroed(stream, len);
        let (n, ch, v, nv) = (
            d.embd,
            d.kda.channels(),
            d.kda.n_v * bloomery_gpu::linear::HEAD,
            d.kda.n_v,
        );
        let head = bloomery_gpu::linear::HEAD;
        let rows = CHUNK * d.heads;
        let segs = attn::segments(0, ctx);
        let groups = cap.div_ceil(CHUNK);
        Ok(Bufs {
            cap,
            rows: vec![0.0; cap * HC_STREAMS * n],
            pos_host: vec![0; cap],
            cnt_host: vec![0; cap],
            vis_host: vec![0; 2 * cap],
            streams: [z(cap * HC_STREAMS * n)?, z(cap * HC_STREAMS * n)?],
            x: z(cap * n)?,
            xn: z(cap * n)?,
            out: z(cap * n)?,
            fold: z(cap * n)?,
            mixes: z(cap * HC_MIX)?,
            hc: z(cap * HC_MIX)?,
            hc_scratch: HcPreScratch::with_groups(stream, HC_STREAMS * n, groups)?,
            pos: zu(cap)?,
            cnt: zu(cap)?,
            vis: zu(2 * cap)?,
            qkv: z(cap * ch)?,
            conv: z(cap * ch)?,
            fa: z(cap * head)?,
            ga: z(cap * head)?,
            beta_raw: z(cap * nv)?,
            beta: z(cap * nv)?,
            f: z(cap * v)?,
            z: z(cap * v)?,
            decay: z(cap * v)?,
            o: z(cap * v)?,
            gated: z(cap * v)?,
            qa: z(cap * d.q_lora)?,
            kv: z(cap * kv_width(d))?,
            qr: z(cap * d.q_lora)?,
            q: z(CHUNK * d.heads * d.head_k)?,
            qabs: z(rows * LATENT)?,
            att: z(rows * LATENT)?,
            av: z(CHUNK * d.heads * d.head_v)?,
            part_v: z(attn::partials_v_len(rows, segs))?,
            part_ms: z(attn::partials_ms_len(rows, segs))?,
            qi: z(CHUNK * kpool::HEADS * kpool::DIM)?,
            wi: z(CHUNK * kpool::HEADS)?,
            scores: z(CHUNK * pools_for(ctx))?,
            list: zu(CHUNK * list_width(d.kept))?,
            h: z(CHUNK * ff)?,
            normed: z(cap * n)?,
            sh_y: z(cap * n)?,
            acc: z(cap * n)?,
            pre: z(cap * n)?,
            probs: z(cap * N_EXPERT)?,
            ids: zu(cap * N_USED)?,
            weights: z(cap * N_USED)?,
            sel: zu(cap * N_USED)?,
            act_x: Q8Act::with_k(stream, CHUNK, n)?,
            card_h: z(CHUNK * N_USED * expert_ff)?,
            act_h: (1..=CHUNK)
                .map(|c| Q8Act::with_slots(stream, c * N_USED, expert_ff))
                .collect::<Result<_, _>>()?,
            card_down: z(CHUNK * N_USED * n)?,
        })
    }

    fn bytes(&self) -> usize {
        let f = [
            &self.streams[0],
            &self.streams[1],
            &self.x,
            &self.xn,
            &self.out,
            &self.fold,
            &self.mixes,
            &self.hc,
            &self.qkv,
            &self.conv,
            &self.fa,
            &self.ga,
            &self.beta_raw,
            &self.beta,
            &self.f,
            &self.z,
            &self.decay,
            &self.o,
            &self.gated,
            &self.qa,
            &self.kv,
            &self.qr,
            &self.q,
            &self.qabs,
            &self.att,
            &self.av,
            &self.part_v,
            &self.part_ms,
            &self.qi,
            &self.wi,
            &self.scores,
            &self.h,
            &self.normed,
            &self.sh_y,
            &self.acc,
            &self.pre,
            &self.probs,
            &self.weights,
            &self.card_h,
            &self.card_down,
        ];
        f.iter().map(|b| b.num_bytes()).sum::<usize>()
            + [
                &self.pos, &self.cnt, &self.vis, &self.ids, &self.sel, &self.list,
            ]
            .iter()
            .map(|b| b.num_bytes())
            .sum::<usize>()
            + self.hc_scratch.device_bytes()
            + self.act_x.device_bytes()
            + self.act_h.iter().map(Q8Act::device_bytes).sum::<usize>()
    }
}

/// The joined projection's rows after the query's low rank: the latent, the
/// index key and the pool gate.
fn kv_width(d: &Dims) -> usize {
    LATENT + 2 * d.index_d
}

/// The chunks of a batch of `t` tokens: runs of [`CHUNK`] from its first,
/// the last one shorter; `(first token, tokens)` each.
fn chunks(t: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..t).step_by(CHUNK).map(move |c0| (c0, CHUNK.min(t - c0)))
}

/// The batches of a call from `from` to `to` whose checkpoint marks are
/// `marks` ([`bloomery_gpu::checkpoint::Checkpoints::marks`]): each run between two marks cut into
/// `⌈len / T_MAX⌉` batches of near-equal size, the first ones a position
/// longer. Each layer reads every host expert its batch's tokens route to
/// once, so a short last batch would pay that read for few tokens.
#[must_use]
pub fn call_batches(from: u32, to: u32, marks: &[u32]) -> Vec<Range<u32>> {
    let mut out = Vec::new();
    let mut at = from;
    for &mark in marks.iter().filter(|&&k| k > from && k <= to) {
        let len = (mark - at) as usize;
        let k = len.div_ceil(T_MAX);
        let mut p = at;
        for j in 0..k {
            let n = (len / k + usize::from(j < len % k)) as u32;
            out.push(p..p + n);
            p += n;
        }
        at = mark;
    }
    out
}

/// The batches a call of `n` ids from the model's position runs
/// ([`call_batches`] over the body's marks): what a `time prompt` record
/// counts as its passes.
pub fn batches_of(m: &Glm5nextModel, n: usize) -> Result<Vec<Range<u32>>, GpuError> {
    let from = m.pos();
    let to = end_of(from, n)?;
    let marks = m.body(WHAT)?.ckpt.marks(from, to);
    Ok(call_batches(from, to, &marks))
}

/// The call's end, `from + n`, refused by name for no id or past `u32`.
fn end_of(from: u32, n: usize) -> Result<u32, GpuError> {
    u32::try_from(n)
        .ok()
        .and_then(|n| from.checked_add(n))
        .filter(|&to| to > from)
        .ok_or_else(|| shape(format!("a prompt of {n} ids from {from}")))
}

/// The call of `ids` from the model's position, refused by name before
/// anything runs: on a poisoned model, for no id, and past the positions the
/// stores hold — the model then stands where it stood, every store as it
/// was. Returns the call's end.
fn check_call(m: &Glm5nextModel, ids: &[u32]) -> Result<u32, GpuError> {
    if let Some(fault) = m.poisoned() {
        return Err(GpuError::Poisoned { what: WHAT, fault });
    }
    let from = m.pos();
    let to = end_of(from, ids.len())?;
    let ctx = m.body(WHAT)?.ctx;
    if to as usize > ctx {
        return Err(shape(format!(
            "a prompt of {} ids from position {from} ends at {to}, past the {ctx} positions the \
             stores hold (the load's ctx)",
            ids.len()
        )));
    }
    Ok(to)
}

/// Feed `ids` from where `m` stands by the body's mode ([`set_prefill`]) and
/// return the argmax after the last: [`prefill`] or the steps ([`prompt`]),
/// each call refused by name before anything runs when it would pass the
/// stores' positions, so neither feed stops part of the way there. The batch
/// call is one residency pass ([`call`]); the steps feed is refused by name
/// while a residency machine runs ([`refuse_steps_under_residency`]).
pub fn feed(m: &mut Glm5nextModel, ids: &[u32]) -> Result<u32, GpuError> {
    check_call(m, ids)?;
    match m.body(WHAT)?.prompt.mode {
        PrefillMode::Batch => call(m, |m| prefill(m, ids)),
        PrefillMode::Steps => {
            super::refuse_steps_under_residency(m)?;
            prompt(m, ids)
        }
    }
}

/// A prompt call `run` of `m` as one residency pass: its boundary before it
/// and none inside ([`GpuModel::pass_boundary`]), so the slot map is one map
/// for the whole call, and 0 rows kept after it, whatever it returned — the
/// residency rule counts decode rows only, and the batch service notes no
/// id. Nothing more on a load with no residency machine.
fn call<T>(
    m: &mut Glm5nextModel,
    run: impl FnOnce(&mut Glm5nextModel) -> Result<T, GpuError>,
) -> Result<T, GpuError> {
    m.pass_boundary()?;
    let r = run(m);
    let kept = m.keep_rows(0, bloomery_gpu::host::PassKind::Prompt);
    // The call's own error first: a keep refused after a failed call is its echo.
    let v = r?;
    kept?;
    Ok(v)
}

/// The body's feed mode.
pub fn prefill_mode(m: &Glm5nextModel) -> Result<PrefillMode, GpuError> {
    Ok(m.body(WHAT)?.prompt.mode)
}

/// Set `m`'s feed to `mode`; the batch feed's buffers, the host tier's batch
/// sets and the host union's slabs are made here, once, so that a timed
/// prompt allocates nothing. Returns whether it made any.
pub fn set_prefill(m: &mut Glm5nextModel, mode: PrefillMode) -> Result<bool, GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    body.prompt.mode = mode;
    if mode == PrefillMode::Steps || body.prompt.batch.is_some() {
        return Ok(false);
    }
    body.make_batch(gpu)?;
    Ok(true)
}

/// Feed `ids` from where `m` stands in batches (module doc) and return the
/// argmax after the last. Refused by name before anything runs: on a
/// poisoned model, past the stores' positions, with the taps armed (a tap
/// holds one token's streams, and a batch runs many). The batch's buffers
/// are made by the first call when [`set_prefill`] has not made them.
pub fn prefill(m: &mut Glm5nextModel, ids: &[u32]) -> Result<u32, GpuError> {
    let to = check_call(m, ids)?;
    let from = m.pos();
    {
        let (gpu, _, body) = m.body_parts(WHAT)?;
        if body.taps.is_some() {
            return Err(shape(
                "a prompt batch with the taps armed: a tap holds one step's streams".to_string(),
            ));
        }
        if body.prompt.batch.is_none() {
            body.make_batch(gpu)?;
        }
    }
    let marks = m.body(WHAT)?.ckpt.marks(from, to);
    let mut argmax = None;
    let mut at = from;
    for mark in marks {
        for run in call_batches(at, mark, &[mark]) {
            let seg = &ids[(run.start - from) as usize..(run.end - from) as usize];
            let last = run.end == to;
            let ran = m.run_rows(seg.len(), WHAT, |gpu, w, body, head, pos| {
                body.enqueue_batch(gpu, w, head, seg, pos, last)
            });
            match ran {
                Ok(t) => argmax = t.or(argmax),
                Err(e) => return Err(take_back(m, from, e)),
            }
        }
        at = at.max(mark);
        let pos = m.pos();
        let (gpu, _, body) = m.body_parts(WHAT)?;
        body.checkpoint(gpu, pos)?;
    }
    argmax.ok_or(GpuError::State {
        what: WHAT,
        missing: "the head of the batch that holds the call's last position",
    })
}

/// A call from `from` that failed with `e`: its positions taken back through
/// the checkpoint of its start, so the model stands where the call found it.
/// A fault poisons the model and is returned as it came: nothing runs on it
/// until a reset.
fn take_back(m: &mut Glm5nextModel, from: u32, e: GpuError) -> GpuError {
    if matches!(e, GpuError::Fault { .. }) || m.poisoned().is_some() {
        return e;
    }
    let kept = match m.body(WHAT) {
        Ok(body) => body.keep_point(from, m.pos()),
        Err(b) => return b,
    };
    match m.rollback(kept) {
        Ok(()) if kept == from => e,
        Ok(()) => shape(format!(
            "a prompt call from position {from} failed ({e}); its positions are taken back and \
             the model stands at {kept}, the cut the checkpoints grant"
        )),
        Err(r) => shape(format!(
            "a prompt call from position {from} failed ({e}), and taking it back failed too ({r})"
        )),
    }
}

/// One store's bits as a digest ([`store_digests`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoreDigest {
    pub layer: usize,
    /// `state`, `ring`, `latent`, `index` or `pooled`.
    pub what: &'static str,
    /// FNV-1a over the store's words, in order.
    pub fnv: u64,
}

/// Every layer's stores read back and digested, in layer order: a KDA
/// layer's committed lane of its state and its conv ring, a latent layer's
/// latent and index rows and its pool plane (all the rows the stores hold,
/// fed or not). Blocking.
pub fn store_digests(m: &mut Glm5nextModel) -> Result<Vec<StoreDigest>, GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    let stream = gpu.stream();
    let lane = body.s.lanes.committed() as usize;
    let mut out = Vec::new();
    for (layer, s) in body.stores.iter().enumerate() {
        match s {
            Store::Kda { state, ring, .. } => {
                for (what, b) in [("state", state.part(lane)), ("ring", ring)] {
                    let words = b.to_host_vec(stream)?;
                    out.push(StoreDigest {
                        layer,
                        what,
                        fnv: fnv(words.iter().map(|v| u64::from(v.to_bits()))),
                    });
                }
            }
            Store::Latent {
                latent,
                index,
                pooled,
            } => {
                for (what, b) in [("latent", latent), ("index", index), ("pooled", pooled)] {
                    let words = b.buf().to_host_vec(stream)?;
                    out.push(StoreDigest {
                        layer,
                        what,
                        fnv: fnv(words.iter().map(|&v| u64::from(v))),
                    });
                }
            }
        }
    }
    Ok(out)
}

/// FNV-1a over `words`, each word's eight little-endian bytes.
fn fnv(words: impl Iterator<Item = u64>) -> u64 {
    words.fold(0xcbf2_9ce4_8422_2325_u64, |h, w| {
        w.to_le_bytes().iter().fold(h, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    })
}

/// Which cached positions a chunk's tokens attend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Keys {
    /// Every one at and before each token's position.
    Dense,
    /// The ones each token's selector list names.
    Selected,
}

/// The keys the chunk at `positions` attends over index rows `rows`: every
/// cached position while the chunk ends within `dense`, the positions the
/// indexer keeps whole (the step's list there is every position, in order);
/// the selector's lists past them. A chunk past the rows is refused by name.
fn prompt_keys(rows: usize, positions: Range<usize>, dense: usize) -> Result<Keys, GpuError> {
    if positions.end > rows {
        return Err(shape(format!(
            "a prompt chunk at positions {positions:?} past the {rows} index rows"
        )));
    }
    Ok(if positions.end <= dense {
        Keys::Dense
    } else {
        Keys::Selected
    })
}

impl Body {
    /// The batch feed's buffers, the host tier's batch sets for as many
    /// tokens and the host union's slabs, made once.
    fn make_batch(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        if self.prompt.batch.is_some() {
            return Ok(());
        }
        let ff = self.cfg.iter().map(|c| c.ff).max().unwrap_or(0);
        let expert_ff = self.card.ff();
        let cap = T_MAX.min(self.ctx);
        let bufs = Bufs::new(gpu, &self.dims, ff, expert_ff, self.ctx, cap)?;
        let hsum = DeviceBuffer::zeroed(gpu.stream(), cap * self.dims.embd)?;
        self.hybrid.prepare_batch(gpu.context(), cap)?;
        self.hybrid.host_mut().prepare_union(cap)?;
        gpu.stream().synchronize()?;
        self.prompt.batch = Some(Box::new(Batch { bufs, hsum }));
        Ok(())
    }

    /// Enqueue one batch of `ids` from position `pos`: its rows and
    /// positions on the card, the walk, and the head after it when `last`
    /// — the pass [`bloomery_gpu::GpuModel::run_rows`] runs, returning whether the head
    /// was enqueued. An inner batch waits for its launches and reads the
    /// fault word, so a fault ends the call before the mark after it.
    fn enqueue_batch(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        head: &mut Head,
        ids: &[u32],
        pos: u32,
        last: bool,
    ) -> Result<bool, GpuError> {
        self.stores_at(pos)?;
        let t = ids.len();
        let n = self.dims.embd;
        let layers = self.cfg.len();
        let Body {
            hybrid,
            cfg,
            names,
            dims,
            k,
            s,
            stores,
            slots,
            card,
            embd,
            held,
            dense,
            prompt,
            ..
        } = self;
        let Batch { bufs, hsum } = prompt.batch.as_deref_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "the batch's buffers (set_prefill)",
        })?;
        if t == 0 || t > bufs.cap {
            return Err(shape(format!(
                "a batch of {t} positions: 1 to {} a batch",
                bufs.cap
            )));
        }
        let rows = &mut bufs.rows[..t * HC_STREAMS * n];
        for (&token, r) in ids.iter().zip(rows.chunks_exact_mut(HC_STREAMS * n)) {
            fill_row(embd, token, r)?;
        }
        for i in 0..t {
            let p = pos + i as u32;
            bufs.pos_host[i] = p;
            bufs.cnt_host[i] = p + 1;
            bufs.vis_host[2 * i] = 0;
            bufs.vis_host[2 * i + 1] = p + 1;
        }
        let stream = gpu.stream();
        span_mut(WHAT, &mut bufs.streams[0], 0, t * HC_STREAMS * n)?
            .copy_from_host(stream, &bufs.rows[..t * HC_STREAMS * n])?;
        span_mut(WHAT, &mut bufs.pos, 0, t)?.copy_from_host(stream, &bufs.pos_host[..t])?;
        span_mut(WHAT, &mut bufs.cnt, 0, t)?.copy_from_host(stream, &bufs.cnt_host[..t])?;
        span_mut(WHAT, &mut bufs.vis, 0, 2 * t)?.copy_from_host(stream, &bufs.vis_host[..2 * t])?;
        *held = pos + t as u32;
        let cap = bufs.cap;
        let mut prog = PromptProgram {
            gpu,
            w,
            p: Parts::of(k, dims, cfg, names, s, stores, slots, card, None),
            b: bufs,
            t,
            cur: 0,
            dense: *dense,
        };
        let o = Overlap {
            units: 1,
            cols: t,
            port: PortKind::Batch,
        };
        let mut leg = BatchLeg::new(stream, hybrid, hsum, cap);
        sched::walk(o, layers, &mut leg, &mut prog)?;
        if last {
            prog.head(head)?;
            return Ok(true);
        }
        match gpu.fault()? {
            Some(fault) => Err(GpuError::fault(WHAT, fault)),
            None => Ok(false),
        }
    }
}

/// Token `token`'s embedding row into `out`, four copies, one a stream; a
/// token past the vocabulary is refused by name.
fn fill_row(e: &Embedding, token: u32, out: &mut [f32]) -> Result<(), GpuError> {
    let t = token as usize;
    if t >= e.n_vocab {
        return Err(shape(format!(
            "token {token} is past the {} embedding rows",
            e.n_vocab
        )));
    }
    let data = e
        .file
        .shard(e.shard)
        .ok_or(GpuError::State {
            what: WHAT,
            missing: "the embedding's shard",
        })?
        .data(&e.info)?;
    let src = &data[t * e.row_bytes..][..e.row_bytes];
    let (first, rest) = out.split_at_mut(e.row.len());
    dequant_row(GgmlType::Q8_0, src, first).map_err(model::ModelError::from)?;
    for s in rest.chunks_exact_mut(first.len()) {
        s.copy_from_slice(first);
    }
    Ok(())
}

/// `y = W · x` for the q8_0 weight `name` over `c` token columns of `x`,
/// token-major into `y` (`q8_0_gemv_mcol`: each column the one-column
/// gemv's bits).
fn mcol(
    gpu: &Gpu,
    w: &Weights,
    name: &str,
    x: &DeviceBuffer<f32>,
    c: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let (qs, d) = q8(w, name)?;
    gpu.q8f32().enqueue_q8_0_gemv_mcol(
        gpu.stream(),
        Q8_0GemvMcolArgs {
            qs,
            d,
            x,
            m: c,
            out: GemvOut::TokenMajor,
            y,
        },
    )
}

/// Rows `rows` of a resident q8_0 weight as a weight of their own: both
/// planes' windows, given back when it drops.
struct RowWindow {
    qs: ManuallyDrop<DeviceTensor<u32>>,
    d: ManuallyDrop<DeviceTensor<u16>>,
}

impl RowWindow {
    /// Rows `rows` of the planes `qs` and `d`, refused by name unless they
    /// are a non-empty run inside both.
    fn of(
        qs: &DeviceTensor<u32>,
        d: &DeviceTensor<u16>,
        rows: Range<usize>,
    ) -> Result<RowWindow, GpuError> {
        if rows.is_empty() || rows.end > qs.rows() || qs.rows() != d.rows() {
            return Err(shape(format!(
                "rows {rows:?} of a q8_0 weight of {} (scales {})",
                qs.rows(),
                d.rows()
            )));
        }
        let (n, r0) = (rows.len(), rows.start);
        let at_q = (r0 * qs.cols() * size_of::<u32>()) as u64;
        let at_d = (r0 * d.cols() * size_of::<u16>()) as u64;
        // SAFETY: rows r0 .. r0 + n lie inside both planes (checked above),
        // each a whole row from its first element, so aligned for its type;
        // the planes are resident weights, in place and alive for the load,
        // and the windows are given back when this drops, within the walk
        // that made them.
        let (qs, d) = unsafe {
            (
                DeviceTensor::window(
                    qs.buf().cu_deviceptr() + at_q,
                    n,
                    qs.cols(),
                    qs.buf().context(),
                ),
                DeviceTensor::window(
                    d.buf().cu_deviceptr() + at_d,
                    n,
                    d.cols(),
                    d.buf().context(),
                ),
            )
        };
        Ok(RowWindow { qs, d })
    }
}

impl Drop for RowWindow {
    fn drop(&mut self) {
        // SAFETY: each window is taken once, here, and handed straight back
        // wrapped, so nothing drops the memory it does not own.
        let (qs, d) = unsafe {
            (
                ManuallyDrop::take(&mut self.qs),
                ManuallyDrop::take(&mut self.d),
            )
        };
        DeviceTensor::release(ManuallyDrop::new(qs));
        DeviceTensor::release(ManuallyDrop::new(d));
    }
}

/// One batch's walk: the body's parts, the batch's buffers, its tokens, the
/// stream buffer the next sub-layer reads, and the positions the latent
/// layers attend whole.
struct PromptProgram<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    p: Parts<'a>,
    b: &'a mut Bufs,
    t: usize,
    cur: usize,
    dense: usize,
}

impl PromptProgram<'_> {
    /// Sub-layer `sub` of layer `l`'s input into `x`: its own mix of the
    /// streams over the batch's tokens (`hc_pre_q8_0` in its token groups)
    /// and their fold by it.
    fn hc_in(&mut self, l: usize, sub: Sub) -> Result<(), GpuError> {
        let (gpu, w, t) = (self.gpu, self.w, self.t);
        let d = *self.p.d;
        let n = self
            .p
            .names
            .get(l)
            .ok_or_else(|| shape(format!("layer {l} past the {} named", self.p.names.len())))?
            .hc(sub);
        let (qs, dd) = q8(w, &n.fn_)?;
        let params = HcQ8Params {
            qs,
            d: dd,
            scale: f32v(w, &n.scale)?,
            base: f32v(w, &n.base)?,
            eps: d.hc_eps,
            iters: d.hc_iters,
        };
        let b = &mut *self.b;
        let streams = &b.streams[self.cur];
        let hc = &self.p.k.hc;
        hc.enqueue_pre_q8_0(
            gpu.stream(),
            &HcQ8PreArgs {
                params: &params,
                x: streams,
                tokens: t,
                rms_eps: d.rms_eps,
            },
            t.min(CHUNK),
            &mut b.hc_scratch,
            &mut b.mixes,
            &mut b.hc,
        )?;
        hc.enqueue_fold(gpu.stream(), streams, &b.hc, d.embd, t, &mut b.x)
    }

    /// The sub-layer's output `out` into the other stream buffer by its mix
    /// (`hc_post`), residual from the current one.
    fn hc_out(&mut self) -> Result<(), GpuError> {
        let b = &mut *self.b;
        let [s0, s1] = &mut b.streams;
        let (res, next) = if self.cur == 0 {
            (&*s0, s1)
        } else {
            (&*s1, s0)
        };
        self.p.k.hc.enqueue_post(
            self.gpu.stream(),
            &HcPostArgs {
                x: &b.out,
                res,
                hc: &b.hc,
                n_embd: self.p.d.embd,
                tokens: self.t,
            },
            next,
            &mut b.fold,
        )?;
        self.cur ^= 1;
        Ok(())
    }

    /// Layer `l`'s KDA mixer over the batch (`kda::kda`'s launches, the
    /// projections by chunk and the conv, the delta step and the gated norm
    /// over every token in position order).
    fn kda(&mut self, l: usize) -> Result<(), GpuError> {
        const W: &str = "glm5next prefill kda";
        let (gpu, w, t) = (self.gpu, self.w, self.t);
        let stream = gpu.stream();
        let d = *self.p.d;
        let fault = gpu.layer_sink(l)?;
        let Some(MixerNames::Kda(nm)) = self.p.names.get(l).map(|n| &n.mixer) else {
            return Err(other_kind(W, l));
        };
        let Some(Store::Kda { state, stamp, ring }) = self.p.stores.get_mut(l) else {
            return Err(GpuError::State {
                what: W,
                missing: "the layer's KDA store",
            });
        };
        let b = &mut *self.b;
        let head = bloomery_gpu::linear::HEAD;
        let (n, ch, nv) = (d.embd, d.kda.channels(), d.kda.n_v);
        let v = nv * head;
        gpu.elem().enqueue_rms_norm(
            stream,
            &b.x,
            f32v(w, &nm.norm)?,
            d.rms_eps,
            n,
            t,
            &mut b.xn,
        )?;
        for (c0, c) in chunks(t) {
            let xs = span(W, &b.xn, c0 * n, c * n)?;
            mcol(
                gpu,
                w,
                &nm.qkv,
                &xs,
                c,
                &mut *span_mut(W, &mut b.qkv, c0 * ch, c * ch)?,
            )?;
            mcol(
                gpu,
                w,
                &nm.f_a,
                &xs,
                c,
                &mut *span_mut(W, &mut b.fa, c0 * head, c * head)?,
            )?;
            mcol(
                gpu,
                w,
                &nm.g_a,
                &xs,
                c,
                &mut *span_mut(W, &mut b.ga, c0 * head, c * head)?,
            )?;
            mcol(
                gpu,
                w,
                &nm.beta,
                &xs,
                c,
                &mut *span_mut(W, &mut b.beta_raw, c0 * nv, c * nv)?,
            )?;
            let fa = span(W, &b.fa, c0 * head, c * head)?;
            mcol(
                gpu,
                w,
                &nm.f_b,
                &fa,
                c,
                &mut *span_mut(W, &mut b.f, c0 * v, c * v)?,
            )?;
            let ga = span(W, &b.ga, c0 * head, c * head)?;
            mcol(
                gpu,
                w,
                &nm.g_b,
                &ga,
                c,
                &mut *span_mut(W, &mut b.z, c0 * v, c * v)?,
            )?;
        }
        let lin = &self.p.k.linear;
        lin.conv.enqueue_kda_conv_prep(
            stream,
            KdaConvArgs {
                x: &b.qkv,
                b_raw: &b.beta_raw,
                f: &b.f,
                w: f32v(w, &nm.conv)?,
                dt_bias: f32v(w, &nm.dt_bias)?,
                ssm_a: f32v(w, &nm.a)?,
                pos: &b.pos,
                shape: d.kda,
                lb: d.lb,
                eps: d.rms_eps,
                m: t,
                fault,
                y: &mut b.conv,
                beta: &mut b.beta,
                decay: &mut b.decay,
                ring,
            },
        )?;
        lin.delta.enqueue_kda_delta_lanes(
            stream,
            KdaLanesArgs {
                lanes: DeltaLanesArgs {
                    delta: DeltaArgs {
                        qkv: &b.conv,
                        beta: &b.beta,
                        decay: &b.decay,
                        lane: self.p.lane,
                        lane_at: 0,
                        lanes: LANES,
                        shape: d.kda,
                        m: t,
                        fault,
                        o: &mut b.o,
                        state: state.whole_mut(),
                    },
                    each: false,
                    pos: &b.pos,
                    stamp,
                },
                row: 0,
            },
        )?;
        lin.norm_gate.enqueue_norm_gate_sigmoid(
            stream,
            NormGateArgs {
                o: &b.o,
                z: &b.z,
                w: f32v(w, &nm.gate_norm)?,
                eps: d.rms_eps,
                n_v: nv,
                m: t,
                fault,
                y: &mut b.gated,
            },
        )?;
        for (c0, c) in chunks(t) {
            let gs = span(W, &b.gated, c0 * v, c * v)?;
            mcol(
                gpu,
                w,
                &nm.out,
                &gs,
                c,
                &mut *span_mut(W, &mut b.out, c0 * n, c * n)?,
            )?;
        }
        Ok(())
    }

    /// Layer `l`'s latent mixer over the batch (`mla::mla`'s launches): the
    /// norm, the joined projection by its two row ranges and chunk, the
    /// query's norm, every token's latent and index rows appended at its
    /// position and the pools the tokens complete, then chunk by chunk the
    /// heads, the selector past the dense positions ([`prompt_keys`]), the
    /// attention and the output projection.
    fn mla(&mut self, l: usize) -> Result<(), GpuError> {
        const W: &str = "glm5next prefill mla";
        let (gpu, w, t) = (self.gpu, self.w, self.t);
        let stream = gpu.stream();
        let d = *self.p.d;
        let fault = gpu.layer_sink(l)?;
        let Some(MixerNames::Latent(nm)) = self.p.names.get(l).map(|n| &n.mixer) else {
            return Err(other_kind(W, l));
        };
        let Some(Store::Latent {
            latent,
            index,
            pooled,
        }) = self.p.stores.get_mut(l)
        else {
            return Err(GpuError::State {
                what: W,
                missing: "the layer's latent store",
            });
        };
        let b = &mut *self.b;
        let s = &*self.p.s;
        let (n, ql, kvw) = (d.embd, d.q_lora, kv_width(&d));
        gpu.elem().enqueue_rms_norm(
            stream,
            &b.x,
            f32v(w, &nm.norm)?,
            d.rms_eps,
            n,
            t,
            &mut b.xn,
        )?;
        {
            let (qs, dd) = q8(w, &nm.stack)?;
            let qa_rows = RowWindow::of(qs, dd, 0..ql)?;
            let kv_rows = RowWindow::of(qs, dd, ql..ql + kvw)?;
            for (c0, c) in chunks(t) {
                let xs = span(W, &b.xn, c0 * n, c * n)?;
                for (win, y, width) in [(&qa_rows, &mut b.qa, ql), (&kv_rows, &mut b.kv, kvw)] {
                    gpu.q8f32().enqueue_q8_0_gemv_mcol(
                        stream,
                        Q8_0GemvMcolArgs {
                            qs: &win.qs,
                            d: &win.d,
                            x: &xs,
                            m: c,
                            out: GemvOut::TokenMajor,
                            y: &mut *span_mut(W, y, c0 * width, c * width)?,
                        },
                    )?;
                }
            }
        }
        gpu.elem().enqueue_rms_norm(
            stream,
            &b.qa,
            f32v(w, &nm.q_a_norm)?,
            d.rms_eps,
            ql,
            t,
            &mut b.qr,
        )?;
        let lat = &self.p.k.latent;
        lat.enqueue_latent_append(
            stream,
            LatentAppendArgs {
                rows: Rows {
                    x: &b.kv,
                    stride: kvw,
                    m: t,
                },
                off: 0,
                gain: f32v(w, &nm.kv_a_norm)?,
                pos: &b.pos,
                eps: d.rms_eps,
                fault,
                cache: latent,
            },
        )?;
        lat.enqueue_index_key_append(
            stream,
            IndexKeyArgs {
                rows: Rows {
                    x: &b.kv,
                    stride: kvw,
                    m: t,
                },
                k_off: LATENT,
                g_off: LATENT + d.index_d,
                w: f32v(w, &nm.index_norm)?,
                b: f32v(w, &nm.index_norm_bias)?,
                pos: &b.pos,
                eps: d.norm_eps,
                fault,
                cache: index,
            },
        )?;
        mla::pool(
            stream, w, self.p.k, &nm.sel, index, &b.cnt, t, fault, pooled,
        )?;
        let first = b.pos_host[0] as usize;
        let rows = index.rows();
        let kept = d.kept;
        let (qs_kb, d_kb) = q8(w, &nm.k_b)?;
        let (qs_vb, d_vb) = q8(w, &nm.v_b)?;
        // SAFETY: a view of no rows at the address of the layer's own latent
        // cache, a live allocation aligned for u16 that outlives the view; the
        // attention reads no window row through it, and the view is released
        // below before the cache can drop.
        let window = unsafe {
            DeviceTensor::<u16>::window(latent.buf().cu_deviceptr(), 0, LATENT, gpu.context())
        };
        let r = (|| -> Result<(), GpuError> {
            for (c0, c) in chunks(t) {
                let qr = span(W, &b.qr, c0 * ql, c * ql)?;
                mcol(gpu, w, &nm.q_b, &qr, c, &mut b.q)?;
                gpu.q8f32().enqueue_q8_0_gemv_heads_mcol(
                    stream,
                    Q8_0GemvHeadsMcolArgs {
                        qs: qs_kb,
                        d: d_kb,
                        x: &b.q,
                        rows_per_head: LATENT,
                        x_head_stride: d.head_k,
                        y_head_stride: LATENT,
                        y_off: 0,
                        m: c,
                        x_col_stride: d.heads * d.head_k,
                        y_col_stride: d.heads * LATENT,
                        y: &mut b.qabs,
                    },
                )?;
                let keys = prompt_keys(rows, first + c0..first + c0 + c, self.dense)?;
                if keys == Keys::Selected {
                    let xn = span(W, &b.xn, c0 * n, c * n)?;
                    let cnt = span(W, &b.cnt, c0, c)?;
                    let mut vis = span_mut(W, &mut b.vis, 2 * c0, 2 * c)?;
                    mla::select(
                        gpu,
                        w,
                        self.p.k,
                        &nm.sel,
                        Select {
                            stream,
                            m: c,
                            xn: &xn,
                            qr: &qr,
                            cnt: &cnt,
                            index,
                            pooled,
                            qi: &mut b.qi,
                            wi: &mut b.wi,
                            scores: &mut b.scores,
                            list: &mut b.list,
                            vis: &mut vis,
                            kept,
                            fault,
                        },
                    )?;
                }
                let vis = span(W, &b.vis, 2 * c0, 2 * c)?;
                self.p.k.attn.enqueue(
                    stream,
                    AttnArgs {
                        q: &b.qabs,
                        window: &window,
                        compressed: Some(&*latent),
                        selected: (keys == Keys::Selected).then_some(SelectedRows {
                            rows: &b.list,
                            stride: list_width(kept),
                        }),
                        vis: &vis,
                        sinks: &s.sinks,
                        scale: 1.0 / (d.head_k as f32).sqrt(),
                        tokens: c,
                        heads: d.heads,
                        part_v: &mut b.part_v,
                        part_ms: &mut b.part_ms,
                        y: &mut b.att,
                        fault,
                    },
                )?;
                gpu.q8f32().enqueue_q8_0_gemv_heads_mcol(
                    stream,
                    Q8_0GemvHeadsMcolArgs {
                        qs: qs_vb,
                        d: d_vb,
                        x: &b.att,
                        rows_per_head: d.head_v,
                        x_head_stride: LATENT,
                        y_head_stride: d.head_v,
                        y_off: 0,
                        m: c,
                        x_col_stride: d.heads * LATENT,
                        y_col_stride: d.heads * d.head_v,
                        y: &mut b.av,
                    },
                )?;
                mcol(
                    gpu,
                    w,
                    &nm.out,
                    &b.av,
                    c,
                    &mut *span_mut(W, &mut b.out, c0 * n, c * n)?,
                )?;
            }
            Ok(())
        })();
        DeviceTensor::release(window);
        r
    }

    /// Layer `l`'s dense block over the batch: the norm, then by chunk the
    /// gate·up·SwiGLU and the down projection.
    fn dense(&mut self, l: usize) -> Result<(), GpuError> {
        const W: &str = "glm5next prefill dense";
        let (gpu, w, t) = (self.gpu, self.w, self.t);
        let stream = gpu.stream();
        let (d, c) = (*self.p.d, self.p.cfg[l]);
        let Some(FfnNames::Dense {
            norm,
            gate,
            up,
            down,
        }) = self.p.names.get(l).map(|n| &n.ffn)
        else {
            return Err(other_kind(W, l));
        };
        let b = &mut *self.b;
        let n = d.embd;
        gpu.elem()
            .enqueue_rms_norm(stream, &b.x, f32v(w, norm)?, d.rms_eps, n, t, &mut b.xn)?;
        let (g, u) = (weight(w, gate)?, weight(w, up)?);
        for (c0, cn) in chunks(t) {
            let xs = span(W, &b.xn, c0 * n, cn * n)?;
            self.p
                .k
                .experts
                .enqueue_shexp_gate_up_mcol(stream, g, u, &xs, cn, c.limit, &mut b.h)?;
            mcol(
                gpu,
                w,
                down,
                &b.h,
                cn,
                &mut *span_mut(W, &mut b.out, c0 * n, cn * n)?,
            )?;
        }
        Ok(())
    }

    /// Layer `l`'s routed block up to its host leg: the norm into `normed`,
    /// the router over every token (the scores, then each token's picks), and
    /// the download of the rows, weights and ids to the host.
    fn route(&mut self, port: &mut BatchLeg<'_, GlmHost>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let (gpu, w, t) = (self.gpu, self.w, self.t);
        let stream = gpu.stream();
        let (d, c) = (*self.p.d, self.p.cfg[l]);
        let fault = gpu.layer_sink(l)?;
        let nm = ffn::moe_names(&self.p, l)?;
        let b = &mut *self.b;
        gpu.elem().enqueue_rms_norm(
            stream,
            &b.x,
            f32v(self.w, nm.norm)?,
            d.rms_eps,
            d.embd,
            t,
            &mut b.normed,
        )?;
        let bias = if c.bias {
            f32v(w, nm.bias)?
        } else {
            &self.p.s.no_bias
        };
        self.p.k.router.enqueue_router_rows(
            stream,
            f32t(w, nm.router)?,
            &b.normed,
            bias,
            d.scale,
            t,
            &mut b.probs,
            &mut b.ids,
            &mut b.weights,
            fault,
        )?;
        let key = port.key(at);
        port.hybrid()
            .enqueue_download(stream, [&b.normed, &b.weights], &b.ids, key)
    }

    /// Layer `l`'s card experts, where the slot map puts any, and its shared
    /// expert by chunk, under its host leg; the card sum plus the shared
    /// expert's output after them.
    fn shadow_rows(&mut self, l: usize) -> Result<(), GpuError> {
        const W: &str = "glm5next prefill shadow";
        let (gpu, w, t) = (self.gpu, self.w, self.t);
        let stream = gpu.stream();
        let c = self.p.cfg[l];
        let n = self.p.d.embd;
        let card = self.p.card.has(l);
        let b = &mut *self.b;
        if card {
            ffn::card_rows(
                gpu,
                w,
                &mut self.p,
                l,
                CardRows {
                    normed: &b.normed,
                    ids: &b.ids,
                    weights: &b.weights,
                    t,
                    sel: &mut b.sel,
                    act_x: &mut b.act_x,
                    h: &mut b.card_h,
                    act_h: &mut b.act_h,
                    down: &mut b.card_down,
                    acc: &mut b.acc,
                },
            )?;
        }
        let nm = ffn::moe_names(&self.p, l)?;
        let (g, u) = (weight(w, nm.sh_gate)?, weight(w, nm.sh_up)?);
        for (c0, cn) in chunks(t) {
            let xs = span(W, &b.normed, c0 * n, cn * n)?;
            self.p
                .k
                .experts
                .enqueue_shexp_gate_up_mcol(stream, g, u, &xs, cn, c.limit, &mut b.h)?;
            mcol(
                gpu,
                w,
                nm.sh_down,
                &b.h,
                cn,
                &mut *span_mut(W, &mut b.sh_y, c0 * n, cn * n)?,
            )?;
        }
        if card {
            gpu.elem()
                .enqueue_add(stream, &b.acc, &b.sh_y, t * n, &mut b.pre)?;
        }
        Ok(())
    }

    /// After the last layer: the last token's streams' mean into the head's
    /// input, and the head.
    fn head(&mut self, head: &mut Head) -> Result<(), GpuError> {
        let n = self.p.d.embd;
        let last = span(
            WHAT,
            &self.b.streams[self.cur],
            (self.t - 1) * HC_STREAMS * n,
            HC_STREAMS * n,
        )?;
        self.p
            .k
            .hc
            .enqueue_mean(self.gpu.stream(), &last, n, 0, head.input_mut())?;
        head.enqueue(self.gpu, self.w)
    }
}

impl<'a> LayerProgram for PromptProgram<'a> {
    type Port = BatchLeg<'a, GlmHost>;

    /// A routed layer's; the dense lead has none.
    fn host_leg(&self, at: At) -> bool {
        self.p.cfg.get(at.layer).is_some_and(|c| c.kind.host_leg())
    }

    /// The mixer sub-layer, then the block's input and the dense block with
    /// its `hc_post`, or the routed block up to its download.
    fn front(&mut self, port: &mut BatchLeg<'a, GlmHost>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let kind = self.p.cfg[l].kind;
        self.hc_in(l, Sub::Attn)?;
        match kind.mixer {
            MixerKind::DeltaRule => self.kda(l)?,
            MixerKind::Latent => self.mla(l)?,
            MixerKind::Gqa => {
                return Err(shape(format!(
                    "layer {l}: a GQA mixer, which glm5next has none of"
                )));
            }
        }
        self.hc_out()?;
        self.hc_in(l, Sub::Ffn)?;
        match kind.ffn {
            FfnKind::Dense => {
                self.dense(l)?;
                self.hc_out()
            }
            FfnKind::Moe => self.route(port, at),
        }
    }

    /// A routed layer's card experts and shared expert under its host leg.
    fn shadow(&mut self, _: &mut BatchLeg<'a, GlmHost>, at: At) -> Result<(), GpuError> {
        if !self.p.cfg[at.layer].kind.host_leg() {
            return Ok(());
        }
        self.shadow_rows(at.layer)
    }

    /// A routed layer's host sums (the walk's serve uploaded them) plus the
    /// shadow's, and `hc_post`.
    fn back(&mut self, port: &mut BatchLeg<'a, GlmHost>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        if !self.p.cfg[l].kind.host_leg() {
            return Ok(());
        }
        let n = self.p.d.embd;
        let b = &mut *self.b;
        let shadowed = if self.p.card.has(l) { &b.pre } else { &b.sh_y };
        self.gpu.elem().enqueue_add(
            self.gpu.stream(),
            port.hsum(),
            shadowed,
            self.t * n,
            &mut b.out,
        )?;
        self.hc_out()
    }
}
