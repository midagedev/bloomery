//! The Qwen3.6-35B-A3B (`qwen35moe`) `ChainBody`: the family's one layer
//! body (`dispatch::layer`) over plans that interleave gated-delta-rule
//! layers with gated GQA layers at head 256, every layer's experts carrying
//! the sigmoid-gated shared expert as one more slot of joined stacks.
//!
//! Load reads the file's description once (`model::arch::qwen35moe`): each
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
//! never a launch argument, so one capture serves every position. A load
//! holds one lane, and the word is [`LANE`] in every record. `reset` zeroes
//! every lane and every conv ring: the recurrence is written in place, so
//! the next sequence would read the previous one's state.
//!
//! The prompt call ([`GpuModel::prefill_with`]) cuts a prompt by
//! [`PrefillPlan`] and walks each unit through the layer program over the
//! ubatch arena, eager in either step mode; a unit of more than
//! [`GEMV_COLS`] rows takes each op's wide arm (`wide`), one of at most
//! that many the gemv arm, bit for bit the decode steps. The last unit ends
//! in its last row's head ([`Tail::Last`]).

use super::body::{Kernels, TapRows, f32_site, kq_site};
use super::dispatch::{self, PassCtx};
use super::head_argmax::HeadArgmaxState;
use super::image::{ImageWrite, PromptImage};
use super::plan::{
    DeltaPlan, GqaKind, GqaPlan, Kind35, Kq, LayerPlan, MixerPlan, MoePlan, SharedPlan, kinds35,
    moe_fits, q35,
};
use super::prefill::{PrefillPath, PrefillPlan, PrefillStep};
use super::program::{Program, Tail};
use super::scratch::{
    Arena, Dims, IN_IDS, IN_POS0, Inbox, Io, KvPlanes, LANE, LayerStore, RecStore, RopeRows,
    StepParams, param_view, put_input,
};
use super::ubatch::UBATCH;
use super::wide::{GEMV_COLS, arena_bytes};
use crate::flash_gqa::{GROUP, HEAD_256};
use crate::head::Head;
use crate::hybrid::Chain;
use crate::linear::{self, LinearShape};
use crate::model::{ChainBody, GpuModel, Instrumented, MAX_PASS_ROWS, NoHost, Rows, block_count};
use crate::rope_table::{RopeSpec, RopeTable};
use crate::tensor::window;
use crate::weights::Weights;
use crate::{Gpu, GpuError, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::Arch;
use model::arch::models::shape::MoeShape;
use model::arch::models::{Mixer, ModelSpec};
use std::mem::ManuallyDrop;
use std::num::NonZeroUsize;
use std::ops::Range;

const WHAT: &str = "qwen35moe::Body35::load";

/// Card bytes a load leaves free past the ubatch arena: what the load and
/// the first prompt still allocate after it — the output head and the
/// heads of a pass of up to [`MAX_PASS_ROWS`] rows (a vocabulary of logits
/// each, some 8 MB together), the captured step and passes, and the
/// allocator's rounding of the arena's some forty buffers to its 2 MiB
/// pages (at most 80 MiB) — with the rest as margin.
pub const FIT_RESERVE: usize = 256 << 20;

/// What [`GpuModel::<Body35>::open`](GpuModel::open) takes besides the card
/// and the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Open35 {
    /// KV rows the caches hold.
    pub ctx: usize,
    /// The decode flash: the tensor-core pass when `true`.
    pub mma: bool,
    /// Tokens per ubatch of the prompt call, `1..=` the router's
    /// [`ubatch`](super::router::RouterDims::ubatch) (at most [`UBATCH`]);
    /// the ubatch arena holds `max(ubatch, GEMV_COLS)` rows, at most `ctx`.
    /// A size outside that range, or an arena that does not fit the card's
    /// free bytes with [`FIT_RESERVE`] left, is refused by name at load.
    pub ubatch: usize,
}

/// Rows of the ubatch arena for ubatches of `u` tokens on a `ctx`-row cache:
/// at least a pass's [`GEMV_COLS`] (the plan's passes and tail walk it too),
/// at most the cache.
fn prompt_rows(u: usize, ctx: usize) -> usize {
    u.max(GEMV_COLS).min(ctx)
}

/// `u` as a ubatch size of `d`, or a named refusal outside
/// `1..=d.router.ubatch()`.
fn ubatch_of(d: &Dims, u: usize) -> Result<NonZeroUsize, GpuError> {
    let most = d.router.ubatch();
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

/// The ubatch arena of `d` for ubatches of `u` tokens, after the check that
/// it fits: its bytes ([`arena_bytes`]) and [`FIT_RESERVE`] within the
/// card's `free` bytes, else a named refusal with the free bytes, the need
/// and the largest ubatch that fits, before anything is allocated. The
/// arena it allocates holds the bytes the check counted, or the load fails
/// by name. Load-time allocation.
fn ubatch_arena(stream: &CudaStream, d: Dims, u: usize, free: usize) -> Result<Arena, GpuError> {
    const WHAT_FIT: &str = "qwen35moe::ubatch_arena";
    let rows = prompt_rows(u, d.ctx);
    let need = arena_bytes(&d, rows);
    if need + FIT_RESERVE > free {
        let fits = (1..u)
            .rev()
            .find(|&v| arena_bytes(&d, prompt_rows(v, d.ctx)) + FIT_RESERVE <= free)
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
    let a = Arena::new(stream, d, rows)?;
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

/// Lanes of every recurrent store: one, read and written in place.
pub(super) const LANES: usize = 1;

/// Qwen3.6's per-replay host values: the token and its position.
pub struct DecodeInput35 {
    token: u32,
    pos: u32,
}

/// Every Qwen3.6 layer's kind, as a gate lists them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind35 {
    /// Gated GQA at head 256 over the layer's K/V planes.
    Attention,
    /// The gated delta rule over the layer's recurrent store.
    Delta,
}

/// A pass's input record — its first position, up to [`MAX_PASS_ROWS`] ids
/// and the lane word — and the windows a captured pass of `m` rows reads.
pub(super) struct RowsParams {
    /// Ids `0..m` at `m − 1`.
    ids: Vec<ManuallyDrop<DeviceBuffer<u32>>>,
    pos0: ManuallyDrop<DeviceBuffer<u32>>,
    lane: ManuallyDrop<DeviceBuffer<u32>>,
    /// The rows the last write planned.
    planned: Option<usize>,
    inbox: Inbox,
}

/// The lane word's offset in a pass's record, after its ids.
const RP_LANE: usize = IN_IDS + MAX_PASS_ROWS;
/// Words of a pass's record.
const RP_WORDS: usize = RP_LANE + 1;

impl RowsParams {
    fn new(stream: &CudaStream) -> Result<RowsParams, GpuError> {
        let inbox = Inbox::new(stream, RP_WORDS)?;
        // SAFETY: every window lies inside the inbox's `RP_WORDS` device
        // words (`IN_IDS + m <= RP_LANE < RP_WORDS`, `IN_POS0 < RP_WORDS`),
        // and the inbox moves into the struct beside them (a move of the
        // handle, not of the allocation), where it outlives them.
        let (ids, pos0, lane) = unsafe {
            (
                (1..=MAX_PASS_ROWS)
                    .map(|m| param_view::<u32>(inbox.dev(), IN_IDS, m))
                    .collect(),
                param_view::<u32>(inbox.dev(), IN_POS0, 1),
                param_view::<u32>(inbox.dev(), RP_LANE, 1),
            )
        };
        Ok(RowsParams {
            ids,
            pos0,
            lane,
            planned: None,
            inbox,
        })
    }

    /// Write `tokens` from position `pos` and the lane word, and enqueue the
    /// record's copy. Asynchronous: the pass behind it reads it.
    pub(super) fn write(
        &mut self,
        stream: &CudaStream,
        tokens: &[u32],
        pos: u32,
    ) -> Result<(), GpuError> {
        if !(1..=MAX_PASS_ROWS).contains(&tokens.len()) {
            return Err(GpuError::shape(
                "qwen35moe::RowsParams::write",
                format!("{} rows; a pass takes 1..={MAX_PASS_ROWS}", tokens.len()),
            ));
        }
        let host = self.inbox.host_mut()?;
        put_input(host, tokens, pos)?;
        host[RP_LANE] = LANE;
        self.inbox.upload(stream, RP_WORDS)?;
        self.planned = Some(tokens.len());
        Ok(())
    }

    /// The input of a pass of `m` rows, as its first launch reads it.
    pub(super) fn io(&self, m: usize) -> Result<Io<'_>, GpuError> {
        let ids = m
            .checked_sub(1)
            .and_then(|i| self.ids.get(i))
            .ok_or_else(|| {
                GpuError::shape(
                    "qwen35moe::RowsParams::io",
                    format!("a pass of {m} rows (1..={MAX_PASS_ROWS})"),
                )
            })?;
        Ok(Io {
            ids,
            pos0: &self.pos0,
            first: 0,
            lane: Some(&self.lane),
        })
    }

    fn bytes(&self) -> usize {
        self.inbox.bytes()
    }
}

/// Everything Qwen3.6's chain owns on the device.
pub struct Body35 {
    pub(super) vocab: usize,
    pub(super) eps: f32,
    pub(super) plans: Vec<LayerPlan>,
    pub(super) stores: Vec<LayerStore>,
    /// The rope table every attention layer reads by position, 64 f32 a row.
    pub(super) rope: RopeRows,
    /// The decode step's one-row arena and its input record.
    pub(super) s: Arena,
    pub(super) sp: StepParams,
    /// A pass's arena of [`MAX_PASS_ROWS`] rows and its input record.
    pub(super) a: Arena,
    pub(super) rp: RowsParams,
    /// The prompt call's ubatch arena, its image, and the ubatch size.
    pub(super) u: Arena,
    pub(super) img: PromptImage,
    pub(super) ubatch: NonZeroUsize,
    pub(super) k: Kernels,
    pub(super) head_state: HeadArgmaxState,
    /// The flash pass the chain runs: the tensor-core pass from load on.
    pub(super) mma: bool,
    pub(super) taps: Option<TapRows>,
}

/// `name` of layer `l`.
fn blk(l: usize, stem: &str) -> String {
    format!("blk.{l}.{stem}")
}

/// The joined name of layer `l`'s stack `part` (`gate`, `up`, `down`).
fn joint(l: usize, part: &str) -> String {
    format!("derived.blk.{l}.ffn_{part}_exps_sh")
}

/// Layer `l`'s routed FFN with the shared expert folded in: the three
/// stacks joined with their shared expert as expert `n` (the routed count),
/// the router with the shared gate as row `n`, each checked against the
/// shape and type its launch takes.
fn resolve_ffn(
    stream: &CudaStream,
    w: &mut Weights,
    d: &Dims,
    l: usize,
) -> Result<MoePlan, GpuError> {
    let (h, ff) = (d.hidden, d.ff);
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
    let router = format!("derived.blk.{l}.ffn_gate_inp_sh");
    w.join_rows(
        stream,
        &[
            blk(l, "ffn_gate_inp.weight").as_str(),
            blk(l, "ffn_gate_inp_shexp.weight").as_str(),
        ],
        router.clone(),
    )?;
    let e = d.router.logits();
    let f = MoePlan {
        ffn_norm: blk(l, "post_attention_norm.weight"),
        ffn_gate_inp: router,
        ffn_gate_exps: joint(l, "gate"),
        ffn_up_exps: joint(l, "up"),
        ffn_down_exps: joint(l, "down"),
        down_ty: Kq::Q4K,
        shared: Some(SharedPlan),
    };
    f32_site(w, &f.ffn_norm, 1, h)?;
    f32_site(w, &f.ffn_gate_inp, e, h)?;
    kq_site(w, &f.ffn_gate_exps, e * ff, h, &[Kq::Q4K])?;
    kq_site(w, &f.ffn_up_exps, e * ff, h, &[Kq::Q4K])?;
    let down_ty = kq_site(w, &f.ffn_down_exps, e * h, ff, &[Kq::Q4K, Kq::Q6K])?;
    Ok(MoePlan { down_ty, ..f })
}

/// Layer `l`'s gated attention, every weight checked.
fn resolve_gqa(w: &Weights, d: &Dims, l: usize) -> Result<GqaPlan, GpuError> {
    let (h, kv, att) = (d.hidden, d.kv_len(), d.attn_len());
    let g = GqaPlan {
        kind: GqaKind::Gated256,
        attn_norm: blk(l, "attn_norm.weight"),
        attn_q: blk(l, "attn_q.weight"),
        attn_k: blk(l, "attn_k.weight"),
        attn_v: blk(l, "attn_v.weight"),
        attn_q_norm: blk(l, "attn_q_norm.weight"),
        attn_k_norm: blk(l, "attn_k_norm.weight"),
        attn_output: blk(l, "attn_output.weight"),
        v_ty: Kq::Q4K,
    };
    f32_site(w, &g.attn_norm, 1, h)?;
    f32_site(w, &g.attn_q_norm, 1, d.head)?;
    f32_site(w, &g.attn_k_norm, 1, d.head)?;
    kq_site(w, &g.attn_q, d.q_rows, h, &[Kq::Q4K])?;
    kq_site(w, &g.attn_k, kv, h, &[Kq::Q4K])?;
    let v_ty = kq_site(w, &g.attn_v, kv, h, &[Kq::Q4K, Kq::Q6K])?;
    kq_site(w, &g.attn_output, h, att, &[Kq::Q4K])?;
    Ok(GqaPlan { v_ty, ..g })
}

/// Layer `l`'s delta rule, every weight checked.
fn resolve_delta(
    w: &Weights,
    d: &Dims,
    shape: LinearShape,
    l: usize,
) -> Result<DeltaPlan, GpuError> {
    let (h, c, nv) = (d.hidden, shape.channels(), shape.n_v);
    let p = DeltaPlan {
        attn_norm: blk(l, "attn_norm.weight"),
        qkv: blk(l, "attn_qkv.weight"),
        qkv_ty: Kq::Q4K,
        gate: blk(l, "attn_gate.weight"),
        beta: blk(l, "ssm_beta.weight"),
        alpha: blk(l, "ssm_alpha.weight"),
        conv: blk(l, "ssm_conv1d.weight"),
        dt_bias: blk(l, "ssm_dt.bias"),
        ssm_a: blk(l, "ssm_a"),
        ssm_norm: blk(l, "ssm_norm.weight"),
        ssm_out: blk(l, "ssm_out.weight"),
        shape,
    };
    f32_site(w, &p.attn_norm, 1, h)?;
    f32_site(w, &p.conv, c, linear::CONV_TAPS)?;
    f32_site(w, &p.dt_bias, 1, nv)?;
    f32_site(w, &p.ssm_a, 1, nv)?;
    f32_site(w, &p.ssm_norm, 1, linear::HEAD)?;
    let qkv_ty = kq_site(w, &p.qkv, c, h, &[Kq::Q4K, Kq::Q6K])?;
    kq_site(w, &p.gate, nv * linear::HEAD, h, &[Kq::Q4K])?;
    kq_site(w, &p.beta, nv, h, &[Kq::Q4K])?;
    kq_site(w, &p.alpha, nv, h, &[Kq::Q4K])?;
    kq_site(w, &p.ssm_out, h, nv * linear::HEAD, &[Kq::Q4K])?;
    Ok(DeltaPlan { qkv_ty, ..p })
}

/// The arena's dims of `spec`, whose kinds `kinds` the plan checked: the
/// attention layers' head of 256 with a gate beside each query, the router
/// the routed shape selects (every layer of a file shares it), and the delta
/// layers' one shape (every delta layer of a file shares it). A delta layer's gated norm writes the attention rows' buffer,
/// so the two widths must agree.
fn dims(spec: &ModelSpec, kinds: &[Kind35], ctx: usize) -> Result<Dims, GpuError> {
    let mut shapes = kinds.iter().filter_map(|k| match k {
        Kind35::Delta(s) => Some(*s),
        Kind35::Gqa => None,
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
    let mut routed = spec.layers.iter().filter_map(|l| l.moe());
    let first = routed.next().ok_or(GpuError::shape(
        WHAT,
        "no routed layer to size the router for",
    ))?;
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
    let router = moe_fits(first).map_err(|e| GpuError::shape(WHAT, e))?;
    let (n_head, n_kv) = (q35::HEADS as usize, q35::KV_HEADS as usize);
    let d = Dims {
        hidden: spec.hidden as usize,
        n_head,
        n_kv,
        head: HEAD_256,
        q_rows: 2 * n_head * HEAD_256,
        ff: q35::EXPERT_FF as usize,
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
    if n_head / n_kv != GROUP || !d.hidden.is_multiple_of(256) {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "{n_head}/{n_kv} heads (the flash's group is {GROUP}), hidden {} (whole K-quant \
                 super-blocks)",
                d.hidden
            ),
        ));
    }
    Ok(d)
}

impl GpuModel<Body35> {
    /// The whole Qwen3.6 model of `file` resident on `gpu`, plus the output
    /// head, with caches of `o.ctx` rows, the decode flash `o.mma` (the
    /// tensor-core pass when `true`) and ubatches of `o.ubatch` tokens
    /// ([`Open35`]). The model takes the file.
    pub fn open(gpu: Gpu, file: Split, o: Open35) -> Result<GpuModel<Body35>, GpuError> {
        let n_layers = block_count(&file, "qwen35moe GpuModel::open")?;
        GpuModel::load_blocks(gpu, &file, o.ctx, 0..n_layers, true, |gpu, w| {
            Body35::load(gpu, &file, w, 0..n_layers, o)
        })
    }
}

impl Body35 {
    fn load(
        gpu: &Gpu,
        file: &Split,
        w: &mut Weights,
        layers: Range<usize>,
        o: Open35,
    ) -> Result<Body35, GpuError> {
        let Open35 { ctx, mma, ubatch } = o;
        let read = model::arch::qwen35moe::spec::read(file).map_err(|e| GpuError::plan(WHAT, e))?;
        let spec = read.spec;
        if layers != (0..spec.layers.len()) || ctx == 0 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the chain runs the whole model, layers 0..{} into a cache of at least one \
                     row; asked for {layers:?} and {ctx} rows",
                    spec.layers.len()
                ),
            ));
        }
        let kinds = kinds35(&spec.layers)?;
        let d = dims(&spec, &kinds, ctx)?;
        let ubatch = ubatch_of(&d, ubatch)?;
        let base = spec
            .layers
            .iter()
            .find_map(|s| match &s.mixer {
                Mixer::Gqa(g) => Some(g.rope.base),
                _ => None,
            })
            .ok_or(GpuError::shape(
                WHAT,
                "no attention layer to read the rope base from",
            ))?;
        let stream = gpu.stream();
        let (mut plans, mut stores) = (Vec::new(), Vec::new());
        for (l, kind) in kinds.iter().enumerate() {
            let ffn = resolve_ffn(stream, w, &d, l)?;
            let (mixer, store) = match *kind {
                Kind35::Gqa => (
                    MixerPlan::Gqa(resolve_gqa(w, &d, l)?),
                    LayerStore::Kv(KvPlanes::new(stream, &d)?),
                ),
                Kind35::Delta(shape) => (
                    MixerPlan::Delta(resolve_delta(w, &d, shape, l)?),
                    LayerStore::Rec(RecStore::new(stream, shape, LANES)?),
                ),
            };
            plans.push(LayerPlan { mixer, ffn });
            stores.push(store);
        }
        let rope = RopeTable::new(&RopeSpec::window(base, q35::ROPE_DIMS as usize))?;
        let rope = RopeRows::new(stream, &rope, q35::ROPE_DIMS as usize, ctx)?;
        let s = Arena::new(stream, d, 1)?;
        let sp = StepParams::new(stream, true)?;
        let a = Arena::new(stream, d, MAX_PASS_ROWS)?;
        let rp = RowsParams::new(stream)?;
        let k = Kernels::load(gpu, true)?;
        let head_state = HeadArgmaxState::new(stream)?;
        let img = PromptImage::new(stream, ctx, true)?;
        let (free, _) = gpu.mem_info()?;
        let u = ubatch_arena(stream, d, ubatch.get(), free)?;
        Ok(Body35 {
            vocab: spec.vocab as usize,
            eps: spec.rms_eps,
            plans,
            stores,
            rope,
            s,
            sp,
            a,
            rp,
            u,
            img,
            ubatch,
            k,
            head_state,
            mma,
            taps: None,
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
            })
            .collect()
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

    /// Device bytes of the layer stores: the attention layers' K/V planes
    /// and the delta layers' states and conv rings.
    #[must_use]
    pub fn store_bytes(&self) -> usize {
        self.stores.iter().map(LayerStore::bytes).sum()
    }

    /// Keep a copy of every layer's output residual after each layer of the
    /// decode chain (`on`), or stop. Load-time allocation.
    pub(super) fn set_taps(&mut self, stream: &CudaStream, on: bool) -> Result<(), GpuError> {
        self.taps = None;
        if on {
            let hidden = self.s.dims.hidden;
            let buf = DeviceBuffer::<f32>::zeroed(stream, self.plans.len() * hidden)?;
            let rows = (0..self.plans.len())
                .map(|l| {
                    let ptr = buf.cu_deviceptr() + (l * hidden * size_of::<f32>()) as u64;
                    // SAFETY: row `l` spans elements `l·hidden ..
                    // (l+1)·hidden` of `buf`, which moves into the same
                    // `TapRows` beside its windows and outlives them.
                    unsafe { window(ptr, hidden, buf.context()) }
                })
                .collect();
            self.taps = Some(TapRows { buf, rows });
        }
        Ok(())
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
        s,
        sp,
        k,
        head_state,
        mma,
        taps,
        ..
    } = b;
    let c = PassCtx {
        gpu,
        w,
        plans,
        k,
        mma: *mma,
        eps: *eps,
        table: &rope.table,
    };
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
    .walk()
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
    let m = heads.len();
    // A capture records the launches only; the replay reads the record the
    // plan before it wrote, so only an eager pass needs one planned now.
    if b.rp.planned != Some(m) && !crate::capturing(gpu.stream())? {
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
        a,
        rp,
        k,
        head_state,
        mma,
        ..
    } = b;
    let c = PassCtx {
        gpu,
        w,
        plans,
        k,
        mma: *mma,
        eps: *eps,
        table: &rope.table,
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
    .walk()
}

impl Body35 {
    /// Enqueue the unit of the image's `tokens`, standing at position `pos`
    /// (the image's position for its first token, else refused): the walk
    /// `(1, t, Step)` over the ubatch arena, ending in the head of its last
    /// row when `head` is given.
    fn walk_unit(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        tokens: Range<usize>,
        pos: u32,
        head: Option<&mut Head>,
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
        let win = self.img.windows(tokens)?;
        let Body35 {
            eps,
            plans,
            stores,
            rope,
            u,
            k,
            head_state,
            mma,
            ..
        } = self;
        let c = PassCtx {
            gpu,
            w,
            plans,
            k,
            mma: *mma,
            eps: *eps,
            table: &rope.table,
        };
        let tail = match head {
            Some(head) => Tail::Last {
                head,
                state: head_state,
            },
            None => Tail::Pass,
        };
        Program {
            c: &c,
            stores: stores.as_mut_slice(),
            s: u,
            io: &win.io(),
            m,
            tail,
        }
        .walk()
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
        let d = body.u.dims;
        let size = ubatch_of(&d, u)?;
        let (free, _) = gpu.mem_info()?;
        body.u = ubatch_arena(gpu.stream(), d, u, free)?;
        body.ubatch = size;
        Ok(())
    }

    /// The units [`GpuModel::prefill_with`] runs a prompt of `tokens` ids as
    /// by `path`, at this model's ubatch size.
    pub fn prefill_plan(&self, tokens: usize, path: PrefillPath) -> Result<PrefillPlan, GpuError> {
        let ub = self.body("qwen35moe::prefill_plan")?.ubatch;
        Ok(PrefillPlan::new(tokens, path, ub))
    }

    /// The last prompt image the prompt call wrote; `None` before one.
    pub fn prompt_image(&self) -> Result<Option<ImageWrite>, GpuError> {
        Ok(self.body("qwen35moe::prompt_image")?.img.last)
    }

    /// Feed `tokens` through the chain by the plan of `path` and return the
    /// greedy next token after the last one; positions continue from wherever
    /// the model stands. The prompt's image — its first position, its ids
    /// and the lane word — is written and copied once, then each unit of the
    /// plan is one walk over the ubatch arena, eager in either step mode: a
    /// unit of at most [`GEMV_COLS`] rows (a pass, or a ubatch that short)
    /// leaves the cache rows, the recurrent state and the logits of one step
    /// per token bit for bit, a longer one agrees with them to its band
    /// (`wide`). The last unit's last row runs the head. The layer taps must
    /// be off, and every id must be below the vocabulary; a prompt past the
    /// cache is refused before any launch.
    pub fn prefill_with(&mut self, tokens: &[u32], path: PrefillPath) -> Result<u32, GpuError> {
        const WHAT_P: &str = "qwen35moe::prefill";
        if tokens.is_empty() {
            return Err(GpuError::shape(WHAT_P, "empty token slice"));
        }
        let pos0 = self.pos();
        self.check_pos(
            pos0 + launch_u32(WHAT_P, "tokens", tokens.len())? - 1,
            WHAT_P,
        )?;
        let plan = self.prefill_plan(tokens.len(), path)?;
        {
            let (gpu, _, body) = self.body_parts(WHAT_P)?;
            if body.taps.is_some() {
                return Err(GpuError::state(
                    WHAT_P,
                    "layer taps off (a prompt unit writes none)",
                ));
            }
            super::refuse_past_vocab(WHAT_P, tokens, body.vocab)?;
            body.img.write(gpu.stream(), tokens, pos0)?;
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
            next = self.run_rows(t, WHAT_P, |gpu, w, body, head, pos| {
                body.walk_unit(gpu, w, unit, pos, last.then_some(head))?;
                Ok(last)
            })?;
        }
        next.ok_or(GpuError::state(WHAT_P, "a token read after the last unit"))
    }
}

impl ChainBody for Body35 {
    type Input = DecodeInput35;
    type Host = NoHost;

    fn arch() -> Arch {
        Arch::Qwen35moe
    }

    fn decode_input(&mut self, token: u32, pos: u32) -> Result<DecodeInput35, GpuError> {
        Ok(DecodeInput35 { token, pos })
    }

    /// The step's input record — its position, its token and the lane word
    /// — in one asynchronous copy ahead of the step's launches.
    fn refresh(&mut self, stream: &CudaStream, input: &DecodeInput35) -> Result<(), GpuError> {
        let DecodeInput35 { token, pos } = *input;
        self.sp.write(stream, token, pos)
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        enqueue_chain(gpu, w, self, head)
    }

    /// Every delta layer's state lanes and conv ring back to zero, and the
    /// records' lane word to [`LANE`]. The K/V planes need nothing: the
    /// flash never loads a key row at or past the live count, and every row
    /// below it is written by its own step first. Synchronizes.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let stream = gpu.stream();
        let most = self
            .stores
            .iter()
            .map(|s| match s {
                LayerStore::Rec(r) => r.state.len().max(r.ring.len()),
                LayerStore::Kv(_) => 0,
            })
            .max()
            .unwrap_or(0);
        let zeros = vec![0.0f32; most];
        for s in &mut self.stores {
            if let LayerStore::Rec(r) = s {
                r.clear(stream, &zeros)?;
            }
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
            + self.s.bytes()
            + self.sp.bytes()
            + self.a.bytes()
            + self.rp.bytes()
            + self.u.bytes()
            + self.img.bytes()
            + self.head_state.bytes()
            + self.taps.as_ref().map_or(0, |t| t.buf.num_bytes())
    }

    fn layers(&self) -> Range<usize> {
        0..self.plans.len()
    }

    fn host(&mut self) -> Option<&mut NoHost> {
        None
    }
}

impl Rows for Body35 {
    const MAX_ROWS: usize = MAX_PASS_ROWS;
    /// No host tier serves a Qwen3.6 replay; the pass is the chain's
    /// one-token layout at `m` rows.
    const CHAIN: Chain = Chain::Step;

    fn plan_rows(&mut self, stream: &CudaStream, tokens: &[u32], pos: u32) -> Result<(), GpuError> {
        self.rp.write(stream, tokens, pos)
    }

    fn enqueue_rows(&mut self, gpu: &Gpu, w: &Weights, heads: &mut [Head]) -> Result<(), GpuError> {
        enqueue_rows(gpu, w, self, heads)
    }
}

impl Instrumented for Body35 {
    /// Rows `0..rows` of every attention layer's K and V planes filled with
    /// a deterministic pattern of finite, nonzero f16 values that differ
    /// from row to row; the delta layers' stores stay as they stand (a
    /// step's cost does not depend on the state's values) — a step shape,
    /// not a model state.
    fn seed_depth(&mut self, gpu: &Gpu, rows: usize) -> Result<(), GpuError> {
        let d = self.s.dims;
        let mut plane = vec![0u16; d.n_kv * d.ctx * d.head];
        let mut state = 0x9e37_79b9u32 ^ rows as u32;
        for h in 0..d.n_kv {
            for r in 0..rows.min(d.ctx) {
                for c in 0..d.head {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let v = ((state >> 9) as f32 / (1u32 << 23) as f32 - 0.5) + 1.0 / 64.0;
                    plane[(h * d.ctx + r) * d.head + c] = gguf::quant::f32_to_f16_bits(v);
                }
            }
        }
        let stream = gpu.stream();
        for s in &mut self.stores {
            if let LayerStore::Kv(p) = s {
                p.k.copy_from_host(stream, &plane)?;
                p.v.copy_from_host(stream, &plane)?;
            }
        }
        stream.synchronize()?;
        Ok(())
    }
}
