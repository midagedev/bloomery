//! The Qwen3.6-35B-A3B (`qwen35moe`) and Qwen3.5-27B (`qwen35`, the Clef
//! backbone) `ChainBody`: the family's one layer body (`dispatch::layer`)
//! over plans that interleave gated-delta-rule layers with gated GQA layers
//! at head 256 — every routed layer's experts carrying the sigmoid-gated
//! shared expert as one more slot of joined stacks, every dense layer's FFN a
//! stack of one expert on the arena's fixed one-slot route. The head counts
//! are the file's; their group selects the flash (eight a block, or pairs).
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
//! in its last row's head ([`Tail::Last`]). The hidden-state call
//! ([`GpuModel::prefill_hidden`]) walks the same units and ends each in the
//! final norm of its rows ([`Tail::Hidden`]).

use super::body::{Kernels, TapRows, f32_site};
use super::dispatch::{self, PassCtx};
use super::head_argmax::HeadArgmaxState;
use super::image::{ImageWrite, PromptImage};
use super::plan::{
    DeltaPlan, FfnPlan, FfnRoute, Flash, Form, GqaKind, GqaPlan, Kind35, LayerPlan, MixerPlan,
    SharedPlan, SiteTy, kinds35, moe_fits, q35,
};
use super::prefill::{PrefillPath, PrefillPlan, PrefillStep};
use super::program::{Program, Tail};
use super::scratch::{
    Arena, Dims, Forms, IN_IDS, IN_POS0, Inbox, Io, KvPlanes, LANE, LayerStore, RecStore, RopeRows,
    StepParams, Wants, param_view, put_input,
};
use super::ubatch::UBATCH;
use super::wide::{GEMV_COLS, arena_bytes};
use crate::flash_gqa::HEAD_256;
use crate::head::Head;
use crate::hybrid::Chain;
use crate::linear::{self, LinearShape};
use crate::model::{ChainBody, GpuModel, Instrumented, MAX_PASS_ROWS, NoHost, Rows, block_count};
use crate::rope_table::{RopeSpec, RopeTable};
use crate::site::{self, Order, file_site};
use crate::tensor::window;
use crate::weights::Weights;
use crate::{Gpu, GpuError, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::Arch;
use model::arch::models::shape::MoeShape;
use model::arch::models::{Ffn, LayerSpec, Mixer, ModelSpec};
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
/// `1..=d.ubatch_most()`.
fn ubatch_of(d: &Dims, u: usize) -> Result<NonZeroUsize, GpuError> {
    let most = d.ubatch_most();
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
fn ubatch_arena(
    stream: &CudaStream,
    (d, forms): (Dims, Forms),
    u: usize,
    free: usize,
) -> Result<Arena, GpuError> {
    const WHAT_FIT: &str = "qwen35moe::ubatch_arena";
    let rows = prompt_rows(u, d.ctx);
    let need = arena_bytes(&d, rows, forms);
    if need + FIT_RESERVE > free {
        let fits = (1..u)
            .rev()
            .find(|&v| arena_bytes(&d, prompt_rows(v, d.ctx), forms) + FIT_RESERVE <= free)
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
    let a = Arena::with(stream, d, rows, forms)?;
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

/// The types each site launches (`crate::site`): a projection of the normed
/// rows or an output projection, a K-quant or Q8_0; β and α also F32; a
/// routed layer's stacks, the Q4_K gate and up and a K-quant down the routed
/// launches take; the embedding Q4_K, Q5_K or Q6_K rows or Q8_0 planes; the head Q6_K,
/// Q4_K or Q8_0 (`Head`).
const PROJ: &[SiteTy] = &[
    SiteTy::Q3K,
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Q8_0,
];
const BETA_ALPHA: &[SiteTy] = &[
    SiteTy::Q3K,
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Q8_0,
    SiteTy::F32,
];
const ROUTED: &[SiteTy] = &[SiteTy::Q4K];
const ROUTED_DOWN: &[SiteTy] = &[SiteTy::Q4K, SiteTy::Q6K];
const EMBED: &[SiteTy] = &[SiteTy::Q4K, SiteTy::Q5K, SiteTy::Q6K, SiteTy::Q8_0];
const HEAD_TY: &[SiteTy] = &[SiteTy::Q6K, SiteTy::Q4K, SiteTy::Q8_0];

/// The type of site `name` (`rows` rows of `k`) from `file`'s header.
fn ty_of(
    file: &Split,
    name: &str,
    rows: usize,
    k: usize,
    allowed: &[SiteTy],
) -> Result<SiteTy, GpuError> {
    file_site(file, WHAT, name, (rows, k), allowed)
}

/// The one type of a routed stack `part` of layer `l` (its experts' and its
/// shared expert's, which the load joins into one stack), of `rows` rows an
/// expert of `k` values, from `file`'s header; parts of two types are
/// refused by name.
fn routed_ty(
    file: &Split,
    l: usize,
    part: &str,
    (rows, k, experts): (usize, usize, usize),
    allowed: &[SiteTy],
) -> Result<SiteTy, GpuError> {
    let exps = ty_of(
        file,
        &blk(l, &format!("ffn_{part}_exps.weight")),
        experts * rows,
        k,
        allowed,
    )?;
    let sh = ty_of(
        file,
        &blk(l, &format!("ffn_{part}_shexp.weight")),
        rows,
        k,
        allowed,
    )?;
    if exps != sh {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "layer {l}: ffn_{part} experts are {exps}, the shared expert {sh}; one stack holds one type"
            ),
        ));
    }
    Ok(exps)
}

/// Layer `l`'s FFN by its description `spec`, every site's type from
/// `file`'s header: a routed one with the shared expert folded in — the
/// three stacks joined with their shared expert as expert `n` (the routed
/// count, [`join_ffn`]), the router with the shared gate as row `n` — or a
/// dense one, its three matrices a stack of one expert on the arena's fixed
/// route.
fn plan_ffn(file: &Split, d: &Dims, spec: &LayerSpec, l: usize) -> Result<FfnPlan, GpuError> {
    let (h, ff) = (d.hidden, d.ff);
    let ffn_norm = blk(l, "post_attention_norm.weight");
    match &spec.ffn {
        Ffn::Moe(m) => {
            let n = m.experts as usize;
            Ok(FfnPlan {
                ffn_norm,
                route: FfnRoute::Router {
                    gate_inp: format!("derived.blk.{l}.ffn_gate_inp_sh"),
                    shared: Some(SharedPlan),
                },
                gate: joint(l, "gate"),
                up: joint(l, "up"),
                down: joint(l, "down"),
                gate_ty: routed_ty(file, l, "gate", (ff, h, n), ROUTED)?,
                up_ty: routed_ty(file, l, "up", (ff, h, n), ROUTED)?,
                down_ty: routed_ty(file, l, "down", (h, ff, n), ROUTED_DOWN)?,
            })
        }
        Ffn::Dense { .. } => {
            let (gate, up, down) = (
                blk(l, "ffn_gate.weight"),
                blk(l, "ffn_up.weight"),
                blk(l, "ffn_down.weight"),
            );
            Ok(FfnPlan {
                ffn_norm,
                route: FfnRoute::Dense,
                gate_ty: ty_of(file, &gate, ff, h, PROJ)?,
                up_ty: ty_of(file, &up, ff, h, PROJ)?,
                down_ty: ty_of(file, &down, h, ff, PROJ)?,
                gate,
                up,
                down,
            })
        }
    }
}

/// Layer `l`'s routed stacks joined with their shared expert, and its router
/// with the shared gate (`Weights::join_rows`: the parts leave the map); a
/// dense layer has nothing to join.
fn join_ffn(
    stream: &CudaStream,
    w: &mut Weights,
    spec: &LayerSpec,
    l: usize,
) -> Result<(), GpuError> {
    if !matches!(spec.ffn, Ffn::Moe(_)) {
        return Ok(());
    }
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
    w.join_rows(
        stream,
        &[
            blk(l, "ffn_gate_inp.weight").as_str(),
            blk(l, "ffn_gate_inp_shexp.weight").as_str(),
        ],
        format!("derived.blk.{l}.ffn_gate_inp_sh"),
    )
}

/// Layer `l`'s gated attention over the flash `flash` its group selects,
/// every site's type from `file`'s header.
fn plan_gqa(file: &Split, d: &Dims, l: usize, flash: Flash) -> Result<GqaPlan, GpuError> {
    let (h, kv, att) = (d.hidden, d.kv_len(), d.attn_len());
    let (q, k, v, o) = (
        blk(l, "attn_q.weight"),
        blk(l, "attn_k.weight"),
        blk(l, "attn_v.weight"),
        blk(l, "attn_output.weight"),
    );
    Ok(GqaPlan {
        kind: GqaKind::Gated256,
        flash,
        attn_norm: blk(l, "attn_norm.weight"),
        attn_q_norm: blk(l, "attn_q_norm.weight"),
        attn_k_norm: blk(l, "attn_k_norm.weight"),
        q_ty: ty_of(file, &q, d.q_rows, h, PROJ)?,
        k_ty: ty_of(file, &k, kv, h, PROJ)?,
        v_ty: ty_of(file, &v, kv, h, PROJ)?,
        o_ty: ty_of(file, &o, h, att, PROJ)?,
        attn_q: q,
        attn_k: k,
        attn_v: v,
        attn_output: o,
    })
}

/// Layer `l`'s delta rule, every site's type from `file`'s header.
fn plan_delta(file: &Split, d: &Dims, shape: LinearShape, l: usize) -> Result<DeltaPlan, GpuError> {
    let (h, c, nv) = (d.hidden, shape.channels(), shape.n_v);
    let (qkv, gate, beta, alpha, out) = (
        blk(l, "attn_qkv.weight"),
        blk(l, "attn_gate.weight"),
        blk(l, "ssm_beta.weight"),
        blk(l, "ssm_alpha.weight"),
        blk(l, "ssm_out.weight"),
    );
    Ok(DeltaPlan {
        attn_norm: blk(l, "attn_norm.weight"),
        conv: blk(l, "ssm_conv1d.weight"),
        dt_bias: blk(l, "ssm_dt.bias"),
        ssm_a: blk(l, "ssm_a"),
        ssm_norm: blk(l, "ssm_norm.weight"),
        shape,
        qkv_ty: ty_of(file, &qkv, c, h, PROJ)?,
        gate_ty: ty_of(file, &gate, nv * linear::HEAD, h, PROJ)?,
        beta_ty: ty_of(file, &beta, nv, h, BETA_ALPHA)?,
        alpha_ty: ty_of(file, &alpha, nv, h, BETA_ALPHA)?,
        out_ty: ty_of(file, &out, h, nv * linear::HEAD, PROJ)?,
        qkv,
        gate,
        beta,
        alpha,
        ssm_out: out,
    })
}

/// Every weight of layer plan `p` resident as the plan read it from the
/// file: each site in its type and shape (`site::site`), each norm and
/// parameter table an F32 plane of its shape.
fn check_resident(w: &Weights, d: &Dims, p: &LayerPlan) -> Result<(), GpuError> {
    let (h, ff) = (d.hidden, d.ff);
    let on =
        |name: &str, rows: usize, k: usize, ty: SiteTy| site::site(w, WHAT, name, (rows, k), ty);
    match &p.mixer {
        MixerPlan::Gqa(g) => {
            let (kv, att) = (d.kv_len(), d.attn_len());
            f32_site(w, &g.attn_norm, 1, h)?;
            f32_site(w, &g.attn_q_norm, 1, d.head)?;
            f32_site(w, &g.attn_k_norm, 1, d.head)?;
            on(&g.attn_q, d.q_rows, h, g.q_ty)?;
            on(&g.attn_k, kv, h, g.k_ty)?;
            on(&g.attn_v, kv, h, g.v_ty)?;
            on(&g.attn_output, h, att, g.o_ty)?;
        }
        MixerPlan::Delta(dp) => {
            let (c, nv) = (dp.shape.channels(), dp.shape.n_v);
            f32_site(w, &dp.attn_norm, 1, h)?;
            f32_site(w, &dp.conv, c, linear::CONV_TAPS)?;
            f32_site(w, &dp.dt_bias, 1, nv)?;
            f32_site(w, &dp.ssm_a, 1, nv)?;
            f32_site(w, &dp.ssm_norm, 1, linear::HEAD)?;
            on(&dp.qkv, c, h, dp.qkv_ty)?;
            on(&dp.gate, nv * linear::HEAD, h, dp.gate_ty)?;
            on(&dp.beta, nv, h, dp.beta_ty)?;
            on(&dp.alpha, nv, h, dp.alpha_ty)?;
            on(&dp.ssm_out, h, nv * linear::HEAD, dp.out_ty)?;
        }
    }
    let f = &p.ffn;
    let e = match &f.route {
        FfnRoute::Router { gate_inp, .. } => {
            let e = d.routed(WHAT)?.logits();
            f32_site(w, gate_inp, e, h)?;
            e
        }
        FfnRoute::Dense => 1,
    };
    f32_site(w, &f.ffn_norm, 1, h)?;
    on(&f.gate, e * ff, h, f.gate_ty)?;
    on(&f.up, e * ff, h, f.up_ty)?;
    on(&f.down, e * h, ff, f.down_ty)
}

impl Forms {
    /// What the arena of a chain of `plans` holds beyond a fused K-quant
    /// chain's buffers (`Forms`'s doc), `d` its dims: every form one of the
    /// sites reads, the gate and up rows of an unfused gate·up, and the
    /// longest row-major output of a K-quant site launched alone.
    fn of(plans: &[LayerPlan], d: &Dims) -> Forms {
        let mut hid: Vec<SiteTy> = Vec::new();
        let (mut attn, mut h, mut glu, mut cols) = (Vec::new(), Vec::new(), false, 0usize);
        let mut alone = |ty: SiteTy, rows: usize| {
            if ty.kgemv_order() == Some(Order::RowMajor) {
                cols = cols.max(rows);
            }
        };
        for p in plans {
            match &p.mixer {
                MixerPlan::Gqa(g) => {
                    hid.extend([g.q_ty, g.k_ty, g.v_ty]);
                    attn.push(g.o_ty);
                    if !g.qkv_fused() {
                        alone(g.q_ty, d.q_rows);
                        alone(g.k_ty, d.kv_len());
                        alone(g.v_ty, d.kv_len());
                    }
                    if !g.o_fused() {
                        alone(g.o_ty, d.hidden);
                    }
                }
                MixerPlan::Delta(dp) => {
                    hid.extend([dp.qkv_ty, dp.gate_ty, dp.beta_ty, dp.alpha_ty]);
                    attn.push(dp.out_ty);
                    if !dp.input_fused() {
                        alone(dp.qkv_ty, dp.shape.channels());
                        alone(dp.gate_ty, dp.shape.n_v * linear::HEAD);
                        alone(dp.beta_ty, dp.shape.n_v);
                        alone(dp.alpha_ty, dp.shape.n_v);
                    }
                    if !dp.out_fused() {
                        alone(dp.out_ty, d.hidden);
                    }
                }
            }
            let f = &p.ffn;
            hid.extend([f.gate_ty, f.up_ty]);
            h.push(f.down_ty);
            if !f.gate_up_fused() {
                glu = true;
                alone(f.gate_ty, d.slots() * d.ff);
                alone(f.up_ty, d.slots() * d.ff);
            }
            if !f.down_sel() {
                alone(f.down_ty, d.hidden);
            }
        }
        let wants = |tys: &[SiteTy]| Wants {
            q128: tys.iter().any(|t| t.reads() == Form::Q8x128),
            q32: tys.iter().any(|t| t.reads() == Form::Q8x32),
        };
        Forms {
            hid: wants(&hid),
            attn: wants(&attn),
            h: wants(&h),
            glu,
            cols,
        }
    }
}

/// What a load reads from the file before any weight is uploaded: the
/// description, each layer's kind and plan with every site's type from the
/// header, the dims, the ubatch and the rope base — so a site of a type no
/// launch reads is refused by name before the upload.
struct Pre35 {
    vocab: usize,
    eps: f32,
    layers: Vec<LayerSpec>,
    kinds: Vec<Kind35>,
    plans: Vec<LayerPlan>,
    d: Dims,
    ubatch: NonZeroUsize,
    base: f32,
}

impl Pre35 {
    fn read(file: &Split, o: Open35) -> Result<Pre35, GpuError> {
        let Open35 { ctx, mma, ubatch } = o;
        let read = model::arch::qwen35moe::spec::read(file).map_err(|e| GpuError::plan(WHAT, e))?;
        let spec = read.spec;
        if ctx == 0 {
            return Err(GpuError::shape(
                WHAT,
                "a cache of at least one row; asked for 0".to_string(),
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
        if mma && kinds.contains(&Kind35::Gqa(Flash::Pairs)) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the tensor-core decode flash at {}/{} heads: the pairs' pass is scalar only \
                     (open with mma false)",
                    d.n_head, d.n_kv
                ),
            ));
        }
        let vocab = spec.vocab as usize;
        ty_of(file, "token_embd.weight", vocab, d.hidden, EMBED)?;
        ty_of(file, "output.weight", vocab, d.hidden, HEAD_TY)?;
        let plans = kinds
            .iter()
            .zip(&spec.layers)
            .enumerate()
            .map(|(l, (kind, layer))| {
                let mixer = match *kind {
                    Kind35::Gqa(flash) => MixerPlan::Gqa(plan_gqa(file, &d, l, flash)?),
                    Kind35::Delta(shape) => MixerPlan::Delta(plan_delta(file, &d, shape, l)?),
                };
                Ok(LayerPlan {
                    mixer,
                    ffn: plan_ffn(file, &d, layer, l)?,
                })
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        Ok(Pre35 {
            vocab,
            eps: spec.rms_eps,
            layers: spec.layers,
            kinds,
            plans,
            d,
            ubatch,
            base,
        })
    }
}

/// The arena's dims of `spec`, whose kinds `kinds` the plan checked: the
/// attention layers' head of 256 with a gate beside each query (every
/// attention layer of a file shares one head count), the FFN's — the router
/// the routed shape selects (every layer of a file shares it), or a dense
/// width every layer shares, never both kinds in one file — and the delta
/// layers' one shape (every delta layer of a file shares it). A delta
/// layer's gated norm writes the attention rows' buffer, so the two widths
/// must agree.
fn dims(spec: &ModelSpec, kinds: &[Kind35], ctx: usize) -> Result<Dims, GpuError> {
    let mut shapes = kinds.iter().filter_map(|k| match k {
        Kind35::Delta(s) => Some(*s),
        Kind35::Gqa(_) => None,
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
    let mut heads = spec.layers.iter().filter_map(|l| match &l.mixer {
        Mixer::Gqa(g) => Some((g.heads as usize, g.kv_heads as usize)),
        _ => None,
    });
    let (n_head, n_kv) = heads.next().ok_or(GpuError::shape(
        WHAT,
        "no attention layer to size the attention rows for",
    ))?;
    if let Some(other) = heads.find(|o| *o != (n_head, n_kv)) {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "two attention head counts, {n_head}/{n_kv} and {other:?}; one arena serves one"
            ),
        ));
    }
    let (router, ff) = ffn_dims(spec)?;
    let d = Dims {
        hidden: spec.hidden as usize,
        n_head,
        n_kv,
        head: HEAD_256,
        q_rows: 2 * n_head * HEAD_256,
        ff,
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
    if !d.hidden.is_multiple_of(256) {
        return Err(GpuError::shape(
            WHAT,
            format!("hidden {} (whole K-quant super-blocks)", d.hidden),
        ));
    }
    Ok(d)
}

/// The FFN's part of the dims of `spec`: the router of its routed layers
/// (one shape, [`moe_fits`]) and the expert width, or no router and the
/// dense width its dense layers share. A file with both kinds, or neither,
/// is refused by name.
fn ffn_dims(spec: &ModelSpec) -> Result<(Option<super::router::RouterDims>, usize), GpuError> {
    let mut routed = spec.layers.iter().filter_map(|l| l.moe());
    let mut dense = spec.layers.iter().filter_map(|l| match l.ffn {
        Ffn::Dense { ff, .. } => Some(ff as usize),
        Ffn::Moe(_) => None,
    });
    match (routed.next(), dense.next()) {
        (Some(first), None) => {
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
            Ok((Some(router), q35::EXPERT_FF as usize))
        }
        (None, Some(ff)) => match dense.find(|&o| o != ff) {
            Some(other) => Err(GpuError::shape(
                WHAT,
                format!("two dense widths, {ff} and {other}; one arena serves one"),
            )),
            None => Ok((None, ff)),
        },
        (Some(_), Some(_)) => Err(GpuError::shape(
            WHAT,
            "routed and dense layers in one file; one arena serves one FFN kind",
        )),
        (None, None) => Err(GpuError::shape(WHAT, "no layer to size the FFN for")),
    }
}

impl GpuModel<Body35> {
    /// The whole Qwen3.6 model of `file` resident on `gpu`, plus the output
    /// head, with caches of `o.ctx` rows, the decode flash `o.mma` (the
    /// tensor-core pass when `true`) and ubatches of `o.ubatch` tokens
    /// ([`Open35`]). Every site's type is read from the file's header and a
    /// type no launch reads refused by name before any upload. The model
    /// takes the file.
    pub fn open(gpu: Gpu, file: Split, o: Open35) -> Result<GpuModel<Body35>, GpuError> {
        let n_layers = block_count(&file, "qwen35moe GpuModel::open")?;
        let pre = Pre35::read(&file, o)?;
        if pre.layers.len() != n_layers {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the description holds {} layers, the file's block count {n_layers}",
                    pre.layers.len()
                ),
            ));
        }
        GpuModel::load_blocks(gpu, &file, o.ctx, 0..n_layers, true, |gpu, w| {
            Body35::load(gpu, w, pre, o)
        })
    }
}

impl Body35 {
    fn load(gpu: &Gpu, w: &mut Weights, pre: Pre35, o: Open35) -> Result<Body35, GpuError> {
        let Pre35 {
            vocab,
            eps,
            layers,
            kinds,
            plans,
            d,
            ubatch,
            base,
        } = pre;
        let ctx = o.ctx;
        let stream = gpu.stream();
        let mut stores = Vec::with_capacity(plans.len());
        for (l, ((kind, layer), plan)) in kinds.iter().zip(&layers).zip(&plans).enumerate() {
            join_ffn(stream, w, layer, l)?;
            check_resident(w, &d, plan)?;
            stores.push(match *kind {
                Kind35::Gqa(_) => LayerStore::Kv(KvPlanes::new(stream, &d)?),
                Kind35::Delta(shape) => LayerStore::Rec(RecStore::new(stream, shape, LANES)?),
            });
        }
        let forms = Forms::of(&plans, &d);
        let rope = RopeTable::new(&RopeSpec::window(base, q35::ROPE_DIMS as usize))?;
        let rope = RopeRows::new(stream, &rope, q35::ROPE_DIMS as usize, ctx)?;
        let s = Arena::with(stream, d, 1, forms)?;
        let sp = StepParams::new(stream, true)?;
        let a = Arena::with(stream, d, MAX_PASS_ROWS, forms)?;
        let rp = RowsParams::new(stream)?;
        let k = Kernels::load(gpu, true)?;
        let head_state = HeadArgmaxState::new(stream)?;
        let img = PromptImage::new(stream, ctx, true)?;
        let (free, _) = gpu.mem_info()?;
        let u = ubatch_arena(stream, (d, forms), ubatch.get(), free)?;
        Ok(Body35 {
            vocab,
            eps,
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
            mma: o.mma,
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

/// What a prompt unit's walk ends in.
enum UnitEnd<'a> {
    /// Nothing: the next unit reads the arena's `x`.
    Pass,
    /// The head of its last row.
    Head(&'a mut Head),
    /// The final norm of every row into the arena's `normed`.
    Hidden,
}

impl Body35 {
    /// Enqueue the unit of the image's `tokens`, standing at position `pos`
    /// (the image's position for its first token, else refused): the walk
    /// `(1, t, Step)` over the ubatch arena, ending as `end` says.
    fn walk_unit(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        tokens: Range<usize>,
        pos: u32,
        end: UnitEnd<'_>,
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
        let tail = match end {
            UnitEnd::Head(head) => Tail::Last {
                head,
                state: head_state,
            },
            UnitEnd::Pass => Tail::Pass,
            UnitEnd::Hidden => Tail::Hidden,
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
        let forms = Forms::of(&body.plans, &d);
        let (free, _) = gpu.mem_info()?;
        body.u = ubatch_arena(gpu.stream(), (d, forms), u, free)?;
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
                let end = if last {
                    UnitEnd::Head(head)
                } else {
                    UnitEnd::Pass
                };
                body.walk_unit(gpu, w, unit, pos, end)?;
                Ok(last)
            })?;
        }
        next.ok_or(GpuError::state(WHAT_P, "a token read after the last unit"))
    }

    /// Feed `tokens` through the chain by the plan of `path`, as
    /// [`GpuModel::prefill_with`] does, and return every position's
    /// final-norm hidden state — the last layer's output RMS-normed by
    /// `output_norm`, llama.cpp's `result_norm` and HF's `last_hidden_state`
    /// — `tokens.len()` rows of the model's width, row `t` at `t · hidden`.
    /// No head runs: no output projection, no logits. Each unit ends in the
    /// norm of its rows ([`Tail::Hidden`]), copied to the host before the next
    /// unit overwrites them, and the fault word is read after each unit: a
    /// raised fault is [`GpuError::Fault`] and leaves the model poisoned.
    /// The layer taps must be off, and every id must be below the
    /// vocabulary; a prompt past the cache is refused before any launch.
    pub fn prefill_hidden(
        &mut self,
        tokens: &[u32],
        path: PrefillPath,
    ) -> Result<Vec<f32>, GpuError> {
        const WHAT_H: &str = "qwen35moe::prefill_hidden";
        if tokens.is_empty() {
            return Err(GpuError::shape(WHAT_H, "empty token slice"));
        }
        let pos0 = self.pos();
        self.check_pos(
            pos0 + launch_u32(WHAT_H, "tokens", tokens.len())? - 1,
            WHAT_H,
        )?;
        let plan = self.prefill_plan(tokens.len(), path)?;
        let hidden = {
            let (gpu, _, body) = self.body_parts(WHAT_H)?;
            if body.taps.is_some() {
                return Err(GpuError::state(
                    WHAT_H,
                    "layer taps off (a prompt unit writes none)",
                ));
            }
            super::refuse_past_vocab(WHAT_H, tokens, body.vocab)?;
            body.img.write(gpu.stream(), tokens, pos0)?;
            body.u.dims.hidden
        };
        let mut out = Vec::with_capacity(tokens.len() * hidden);
        let mut at = 0usize;
        for &step in &plan.steps {
            let t = match step {
                PrefillStep::Ubatch(t) | PrefillStep::Pass(t) => t,
            };
            let unit = at..at + t;
            at += t;
            let out = &mut out;
            self.run_rows(t, WHAT_H, |gpu, w, body, _, pos| {
                body.walk_unit(gpu, w, unit, pos, UnitEnd::Hidden)?;
                if let Some(f) = gpu.fault()? {
                    return Err(GpuError::fault(WHAT_H, f));
                }
                // SAFETY: rows `0..t` of `normed` span `t · hidden` values
                // inside it (`rows · hidden`, `t <= rows` by the walk's
                // refusal), and the arena stays in place for the copy.
                let rows = unsafe { super::scratch::f32_view(&body.u.normed, 0, t * hidden) };
                out.extend_from_slice(&rows.to_host_vec(gpu.stream())?);
                Ok(false)
            })?;
        }
        Ok(out)
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
