//! Launch order: the chain and one layer, enqueued asynchronously against a
//! resident arena. No allocation, no synchronization, no host round trip —
//! so this is both the eager body and what the capture records. One layer
//! body serves every pass and both bodies of the family: the decode step is
//! its one-row instance, a pass its `m`-row instance, and a layer's
//! [`LayerPlan`] picks its mixer — attention over K/V planes, or the delta
//! rule over a recurrent store ([`super::delta`]) — and whether its experts
//! carry a folded shared expert. Which kernel an op launches follows from
//! `m` and the plan inside the op: past [`GEMV_COLS`] rows its first line
//! hands the unit to its wide arm (`wide`: the grouped GEMM, the prefill
//! flash, the router's two launches), and a unit's embedding is followed by
//! its dense route table. At one row the FFN norm is folded into
//! the router's launch, which writes the bytes of the pair it replaces, so a
//! one-row pass and a multi-row one leave the same state. The q8_1 of the
//! attention rows and of the SwiGLU rows stay launches of their own: folded
//! into the projection beside them, the work lands in every block of a grid
//! that already streams its weights at the card's bandwidth.

use super::body::{ATTN_SCALE, ATTN_SCALE_256, Body, Kernels};
use super::delta;
use super::experts::{CombineArgs, GateUpArgs};
use super::head_argmax::HeadArgmaxState;
use super::plan::{FfnPlan, FfnRoute, Form, GqaKind, GqaPlan, LayerPlan, MixerPlan, SiteTy};
use super::program::{Program, Tail};
use super::proj::{OResidArgs, QkvArgs};
use super::scratch::{Append128, Append256, Arena, FlashPass, Io, KvPlanes, StoreMut};
use super::wide::{self, GEMV_COLS};
use crate::elem::EmbedRowsArgs;
use crate::gated_quant::GateLayout;
use crate::head::Head;
use crate::model::lookup::{f32_gain, f32_tensor, kq_weight};
use crate::q38::{EmbedQ8Args, OutGateArgs};
use crate::site::{self, Order};
use crate::tensor::Q8Act;
use crate::weights::{DevWeight, Weights};
use crate::{FaultSink, Gpu, GpuError};
use cuda_core::DeviceBuffer;
use gguf::quant::GgmlType;
use model::arch::qwen3moe::names::token_embd;

/// What every launch of one layer reads besides the arena and the store:
/// the engine, the weights, the layer's plan, the kernels, the flash pass,
/// the norm epsilon, the rope table, and the layer's index with its fault
/// sink — every launch of the layer that can refuse its input raises with
/// that index.
pub(super) struct Ctx<'a> {
    pub(super) gpu: &'a Gpu,
    pub(super) w: &'a Weights,
    pub(super) p: &'a LayerPlan,
    pub(super) k: &'a Kernels,
    pub(super) mma: bool,
    pub(super) eps: f32,
    pub(super) table: &'a DeviceBuffer<f32>,
    pub(super) layer: usize,
    pub(super) sink: FaultSink,
}

impl<'a> Ctx<'a> {
    /// Layer `layer`'s context, its sink made from its index.
    pub(super) fn new(
        gpu: &'a Gpu,
        w: &'a Weights,
        (p, layer): (&'a LayerPlan, usize),
        k: &'a Kernels,
        mma: bool,
        eps: f32,
        table: &'a DeviceBuffer<f32>,
    ) -> Result<Ctx<'a>, GpuError> {
        Ok(Ctx {
            gpu,
            w,
            p,
            k,
            mma,
            eps,
            table,
            layer,
            sink: gpu.layer_sink(layer)?,
        })
    }
}

/// Enqueue the whole decode chain at m = 1: layer 0 with its embedding,
/// every later layer reading the previous layer's output, then the head —
/// the walk `(1, 1, Step)` ([`Program`]). The combine writes a layer's
/// output where the next reader takes it — the arena's `x` for the next
/// layer, the head's input after the last — so no copy crosses a layer
/// boundary.
pub(super) fn enqueue_chain(
    gpu: &Gpu,
    w: &Weights,
    b: &mut Body,
    head: &mut Head,
) -> Result<(), GpuError> {
    let Body {
        hp,
        plans,
        kv,
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
        eps: hp.rms_eps,
        table: &rope.table,
    };
    Program {
        c: &c,
        stores: kv.as_mut_slice(),
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

/// Enqueue the one-row head: its norm and quantization, then the Q6_K
/// projection with the argmax folded in (`head_argmax`) in place of the
/// shared head's gemv and `argmax_fault` — the same logits and the same
/// (token, fault word) readback. A head of another type (the type the load
/// read from the file) runs the shared head's own arm: a Q8_0 one the norm,
/// the q8f32 gemv and the argmax, three launches as the fused head's; a Q4_K
/// one the norm, the quantizer, the Q4_K gemv and the argmax.
pub(super) fn enqueue_head(
    gpu: &Gpu,
    w: &Weights,
    k: &Kernels,
    state: &mut HeadArgmaxState,
    head: &mut Head,
) -> Result<(), GpuError> {
    if !matches!(
        w.get("output.weight"),
        Some(DevWeight::KQuant {
            ty: GgmlType::Q6_K,
            ..
        })
    ) {
        return head.enqueue(gpu, w);
    }
    let stream = gpu.stream();
    head.enqueue_with_tail(gpu, w, |act, out_w, fault, logits, out| {
        k.head
            .enqueue(stream, out_w, act, fault, logits, state, out)
    })
}

/// Enqueue layer `slot`'s step, the step's front in front of it when `embed`
/// — the token's embedding row into `x` and its position and live key count
/// into the arena, from the step's input record; `x` in, the output residual
/// into `out` (`None`: back into `x`), the step's K/V row appended to the
/// layer's planes at the position the front wrote.
pub(super) fn enqueue_layer(
    gpu: &Gpu,
    w: &Weights,
    b: &mut Body,
    slot: usize,
    embed: bool,
    out: Option<&mut DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    let Body {
        hp,
        plans,
        kv,
        rope,
        s,
        sp,
        k,
        mma,
        ..
    } = b;
    let c = Ctx::new(
        gpu,
        w,
        (&plans[slot], slot),
        k,
        *mma,
        hp.rms_eps,
        &rope.table,
    )?;
    layer(&c, StoreMut::Kv(&mut kv[slot]), s, &sp.io(), 1, embed, out)
}

/// Enqueue the step's front alone: the embedding row of the step's token
/// into the one-row arena's `x`, and its position and live key count. A
/// layer run on its own from another input ([`enqueue_layer`] without
/// `embed`) needs the position the front writes; the caller then replaces
/// `x`.
pub(super) fn enqueue_front(gpu: &Gpu, w: &Weights, b: &mut Body) -> Result<(), GpuError> {
    let Body { s, sp, k, .. } = b;
    embed_rows(gpu, w, k, &sp.io(), s)
}

/// What every launch of a prefill pass reads besides its arena, its inputs
/// and the cache: the engine, the weights, every layer's plan, the kernels,
/// the flash pass, the norm epsilon and the rope table.
pub(super) struct PassCtx<'a> {
    pub(super) gpu: &'a Gpu,
    pub(super) w: &'a Weights,
    pub(super) plans: &'a [LayerPlan],
    pub(super) k: &'a Kernels,
    pub(super) mma: bool,
    pub(super) eps: f32,
    pub(super) table: &'a DeviceBuffer<f32>,
}

/// Enqueue one prefill pass of `m` tokens over arena `s` from the input
/// record windows in `io`: the embedding rows with the rows' positions and
/// live key counts, then every layer at `m` rows, the combine leaving each
/// layer's output in `x` — the walk `(1, m, Step)` ([`Program`]). The same
/// enqueues run eager and under a capture.
pub(super) fn enqueue_pass(
    c: &PassCtx<'_>,
    kv: &mut [KvPlanes],
    s: &mut Arena,
    io: &Io<'_>,
    m: usize,
) -> Result<(), GpuError> {
    Program {
        c,
        stores: kv,
        s,
        io,
        m,
        tail: Tail::Pass,
    }
    .walk()
}

/// The launches a pass of `m` rows makes over layers `plans`, counted from
/// [`layer`]'s enqueues: the embedding, then per layer the mixer and the FFN
/// half.
/// - Attention: norm+quant, q·k·v, QK-norm+rope+append, the flash's segment
///   pass and merge, the attention rows' quantizer (with the output gate at
///   head 256) and the output projection — 7 — plus a Q6_K value
///   projection's gemv, and at more than one row its token-major copy. With
///   q·k·v of other types each projection launches alone
///   ([`site_launches`]); an output projection not of Q4_K is its
///   quantizer (a K-quant) or the f32 out gate (any other), the projection
///   and the residual add.
/// - Delta rule ([`delta::launches`]): norm+quant, the two projection
///   launches, conv, delta, gated norm, quantizer and output projection —
///   8 — plus a Q6_K q·k·v projection's token-major copy at more than one
///   row.
/// - FFN: the norm with the router (one launch at one row, two at more),
///   gate·up, one quantizer over every token's slots, the down `_sel` over
///   every token's slots, and the combine — 5 or 6.
///
/// - A dense FFN: `norm_quant` and the four launches after the router — 5;
///   an unfused gate·up is the gate, the up and the SwiGLU; a Q8_0 down is
///   one launch with no quantizer, a Q3_K or Q5_K one the quantizer and its
///   projection ([`site_launches`]).
///
/// So a qwen3moe layer is 12 or 13 launches at one row and 13 or 15 at
/// every `m` from two to [`GEMV_COLS`]. Past it (the wide arm, `wide`) the
/// embedding is followed by the unit's dense route table, and a layer is
/// its mixer — the gated attention's norm, quantizer, three GEMMs, rope,
/// prefill flash, gated quantizer, output GEMM and residual add, 10; the
/// delta rule's 12 ([`delta::launches`]) — and the FFN's norm, quantizer,
/// router logits and routing, route table, gate, up, SwiGLU quantizer, down
/// and combine, 10; a dense FFN's norm, quantizer, gate, up, SwiGLU
/// quantizer, down and combine, 7. A wide input is quantized once for each
/// form its sites read ([`wide_quants`]: none for F32 sites alone, two for
/// a K-quant beside a Q8_0), and a gated output projection not of a K-quant
/// takes the f32 out gate and its own quantizer in place of the gated one.
pub(super) fn pass_launches(plans: &[LayerPlan], m: usize) -> usize {
    let wide = m > GEMV_COLS;
    let many = usize::from(m > 1);
    let site = |t: SiteTy| site_launches(t, m);
    let per_layer = |p: &LayerPlan| {
        let mixer = match &p.mixer {
            MixerPlan::Gqa(g) if wide => {
                let out = if g.o_ty.kquant() {
                    1
                } else {
                    1 + wide_quants(&[g.o_ty])
                };
                1 + wide_quants(&[g.q_ty, g.k_ty, g.v_ty]) + 3 + 2 + out + 2
            }
            MixerPlan::Gqa(g) => {
                let qkv = if g.qkv_fused() {
                    1 + usize::from(g.v_ty == SiteTy::Q6K) * (1 + many)
                } else {
                    site(g.q_ty) + site(g.k_ty) + site(g.v_ty)
                };
                let out = if g.o_fused() {
                    2
                } else {
                    1 + out_resid_launches(g.o_ty, m)
                };
                1 + qkv + 3 + out
            }
            MixerPlan::Delta(d) => delta::launches(d, m),
        };
        let f = &p.ffn;
        let ffn = match (&f.route, wide) {
            (FfnRoute::Router { .. }, true) => 10,
            (FfnRoute::Router { .. }, false) => 5 + many,
            (FfnRoute::Dense, true) => 1 + wide_quants(&[f.gate_ty, f.up_ty]) + 5,
            (FfnRoute::Dense, false) => {
                let gate_up = if f.gate_up_fused() {
                    1
                } else {
                    site(f.gate_ty) + site(f.up_ty) + 1
                };
                let down = if f.down_sel() {
                    2
                } else {
                    usize::from(f.down_ty.kquant()) + site(f.down_ty)
                };
                2 + gate_up + down
            }
        };
        mixer + ffn
    };
    1 + usize::from(wide) + plans.iter().map(per_layer).sum::<usize>()
}

/// The quantizer launches a wide input takes for sites of types `tys`: one
/// for each of the q8_1-of-128 and q8-of-32 forms one of them reads.
pub(super) fn wide_quants(tys: &[SiteTy]) -> usize {
    [Form::Q8x128, Form::Q8x32]
        .iter()
        .filter(|f| tys.iter().any(|t| t.reads() == **f))
        .count()
}

/// Enqueue layer `slot`'s FFN half alone: `ffn_inp` in, the output residual
/// into `x`.
pub(super) fn enqueue_ffn(
    gpu: &Gpu,
    w: &Weights,
    b: &mut Body,
    slot: usize,
) -> Result<(), GpuError> {
    let Body {
        hp,
        plans,
        rope,
        s,
        k,
        mma,
        ..
    } = b;
    let c = Ctx::new(
        gpu,
        w,
        (&plans[slot], slot),
        k,
        *mma,
        hp.rms_eps,
        &rope.table,
    )?;
    ffn(&c, &plans[slot].ffn, s, 1, None)
}

/// One layer at `m` rows: the embedding rows with their positions and live
/// key counts in front when `embed` ([`embed_rows`]), the mixer the plan
/// names over the layer's store, then the FFN half into `out` (`None`: back
/// into `x`).
pub(super) fn layer(
    c: &Ctx<'_>,
    st: StoreMut<'_>,
    s: &mut Arena,
    io: &Io<'_>,
    m: usize,
    embed: bool,
    out: Option<&mut DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    mixer(c, st, s, io, m, embed)?;
    ffn(c, &c.p.ffn, s, m, out)
}

/// One layer at `m` rows up to its FFN half ([`layer`]'s first part): the
/// embedding rows in front when `embed`, then the mixer the plan names over
/// the layer's store, which leaves the FFN's input residual in `ffn_inp`.
pub(super) fn mixer(
    c: &Ctx<'_>,
    st: StoreMut<'_>,
    s: &mut Arena,
    io: &Io<'_>,
    m: usize,
    embed: bool,
) -> Result<(), GpuError> {
    if embed {
        embed_rows(c.gpu, c.w, c.k, io, s)?;
        if m > GEMV_COLS {
            wide::route_dense(c, s, m)?;
        }
    }
    match (&c.p.mixer, st) {
        (MixerPlan::Gqa(g), StoreMut::Kv(kv)) => attention(c, g, kv, s, m)?,
        (MixerPlan::Delta(d), StoreMut::Rec(r)) => delta::delta(c, d, r, s, io, m)?,
        _ => {
            return Err(GpuError::shape(
                "qwen3moe::layer",
                format!(
                    "layer {}'s plan and store are of two kinds (attention over K/V planes, the \
                     delta rule over a recurrent store)",
                    c.layer
                ),
            ));
        }
    }
    Ok(())
}

/// A unit's front, one launch: the embedding rows of `io`'s ids into `s.x`
/// and each row's position and live key count into `s.pos` and `s.n_keys` —
/// the rows every later launch of the unit reads. The table's resident type
/// picks the lookup: Q4_K, Q5_K or Q6_K rows, or Q8_0 planes (the load admits
/// only these).
pub(super) fn embed_rows(
    gpu: &Gpu,
    w: &Weights,
    k: &Kernels,
    io: &Io<'_>,
    s: &mut Arena,
) -> Result<(), GpuError> {
    const WHAT: &str = "qwen3moe::embed_rows";
    let name = token_embd();
    if let Some(DevWeight::Q8_0 { qs, d, .. }) = w.get(&name) {
        return k.q35(WHAT)?.q38.enqueue_embed_rows(
            gpu.stream(),
            EmbedQ8Args {
                qs,
                d,
                ids: io.ids,
                pos0: io.pos0,
                first: io.first,
                fault: gpu.unlabelled_sink(),
                y: &mut s.x,
                pos: &mut s.pos,
                n_keys: &mut s.n_keys,
            },
        );
    }
    let ty = match w.get(&name) {
        Some(DevWeight::KQuant { ty, .. }) => *ty,
        _ => return Err(GpuError::tensor(WHAT, &name, "K-quant rows or Q8_0 planes")),
    };
    gpu.elem().enqueue_embed_rows_kquant(
        gpu.stream(),
        ty,
        EmbedRowsArgs {
            w: kq_weight(w, &name)?,
            ids: io.ids,
            pos0: io.pos0,
            first: io.first,
            y: &mut s.x,
            pos: &mut s.pos,
            n_keys: &mut s.n_keys,
        },
    )
}

/// The launches [`site_gemv`] makes for a site of type `ty` at `m` rows: a
/// K-quant's gemv, and past one row the token-major copy of a row-major one;
/// any other type's one launch.
pub(super) fn site_launches(ty: SiteTy, m: usize) -> usize {
    1 + usize::from(ty.kgemv_order() == Some(Order::RowMajor) && m > 1)
}

/// `y = W · x` for site `name` of type `ty`, `rows` rows, at `m <=
/// GEMV_COLS` rows, token-major: a K-quant's `site::kgemv` on the q8_1 rows
/// `act` (a row-major one past one row into `cols`, then copied
/// token-major), any other type's `site::gemv` on the f32 rows `x`.
#[allow(
    clippy::too_many_arguments,
    reason = "one site's context, weight, rows, two input forms, width, scratch and output (rust-quality R8)"
)]
pub(super) fn site_gemv(
    c: &Ctx<'_>,
    (ty, name): (SiteTy, &str),
    rows: usize,
    (act, x): (&Q8Act, &DeviceBuffer<f32>),
    m: usize,
    cols: Option<&mut DeviceBuffer<f32>>,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    const WHAT: &str = "qwen3moe::site_gemv";
    let gpu = c.gpu;
    let q35 = c.k.q35(WHAT)?;
    let Some(order) = ty.kgemv_order() else {
        return site::gemv(gpu, &q35.g32, (ty, c.w, name), x, m, y);
    };
    let gemv = |out: &mut DeviceBuffer<f32>| {
        site::kgemv(gpu, &q35.kgemv, (ty, c.w, name), act, c.sink, out)
    };
    if m == 1 || order == Order::TokenMajor {
        return gemv(y);
    }
    let cols = cols.ok_or(GpuError::state(
        WHAT,
        "the arena's row-major scratch (a K-quant site launched alone past one row)",
    ))?;
    gemv(cols)?;
    c.k.proj.enqueue_token_major(gpu.stream(), cols, rows, m, y)
}

/// The attention half: `x` in, `ffn_inp = x + attn_output(attn(x))` out —
/// q, k and v in one launch (a Q6_K v in its own) when Q4_K, else each by
/// its type ([`site_gemv`]), the head norm and the rope by the rows'
/// positions with the cache append, the flash over the rows' live key
/// counts, the output projection — a Q4_K one with the residual add in its
/// store, any other into `normed` and an add. At head 256 the query
/// projection writes each head's `[q | gate]`, the rope reads the queries
/// out of those rows and turns the first 64 values, and the output
/// projection's input is the flash output multiplied by `σ(gate)`: in its
/// quantizer for a K-quant, as f32 rows for any other type
/// ([`GqaKind::Gated256`]).
fn attention(
    c: &Ctx<'_>,
    n: &GqaPlan,
    kv: &mut KvPlanes,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    if m > GEMV_COLS {
        return wide::attention(c, n, kv, s, m);
    }
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let stream = gpu.stream();
    let d = s.dims;
    let i = s.col(m)?;
    gpu.fused().enqueue_norm_quant(
        stream,
        &s.x,
        f32_gain(w, &n.attn_norm)?,
        c.eps,
        &mut s.act_x[i],
        &mut s.normed,
        c.sink,
    )?;
    if n.qkv_fused() {
        let wv = kq_weight(w, &n.attn_v)?;
        let (off_k, off_v) = s.qkv_offsets();
        k.proj.enqueue_qkv(
            stream,
            QkvArgs {
                wq: kq_weight(w, &n.attn_q)?,
                wk: kq_weight(w, &n.attn_k)?,
                wv: (n.v_ty == SiteTy::Q4K).then_some(wv),
                act: &s.act_x[i],
                y: &mut s.qkv,
                off_k,
                off_v,
            },
        )?;
        if n.v_ty == SiteTy::Q6K {
            match s.v_cols.as_mut().filter(|_| m > 1) {
                Some(cols) => {
                    gpu.enqueue_gemv_q6k(wv, &s.act_x[i], cols)?;
                    k.proj
                        .enqueue_token_major(stream, cols, d.kv_len(), m, &mut s.v)?;
                }
                None => gpu.enqueue_gemv_q6k(wv, &s.act_x[i], &mut s.v)?,
            }
        }
    } else {
        let Arena {
            normed,
            act_x,
            q,
            k: kr,
            v,
            cols,
            ..
        } = s;
        let x = (&act_x[i], &*normed);
        site_gemv(c, (n.q_ty, &n.attn_q), d.q_rows, x, m, cols.as_mut(), q)?;
        site_gemv(c, (n.k_ty, &n.attn_k), d.kv_len(), x, m, cols.as_mut(), kr)?;
        site_gemv(c, (n.v_ty, &n.attn_v), d.kv_len(), x, m, cols.as_mut(), v)?;
    }
    match n.kind {
        GqaKind::Neox128 => {
            kv.append_128(
                &k.neox,
                stream,
                Append128 {
                    q: &mut s.q,
                    k: &mut s.k,
                    v: &s.v,
                    gq: f32_gain(w, &n.attn_q_norm)?,
                    gk: f32_gain(w, &n.attn_k_norm)?,
                    table: c.table,
                    pos: &s.pos,
                    eps: c.eps,
                    n_head: d.n_head,
                    n_kv: d.n_kv,
                    ctx: d.ctx,
                    m,
                    fault: c.sink,
                },
            )?;
            kv.flash_128(
                &k.flash,
                stream,
                FlashPass {
                    q: &s.q,
                    n_keys: &s.n_keys,
                    scale: ATTN_SCALE,
                    n_kv: d.n_kv,
                    ctx: d.ctx,
                    m,
                    part_v: &mut s.part_v,
                    part_ms: &mut s.part_ms,
                    fault: c.sink,
                    y: &mut s.attn,
                },
                c.mma,
            )?;
            gpu.enqueue_quantize_q8_1_layer(&s.attn, &mut s.act_attn[i], c.layer)?;
        }
        GqaKind::Gated256 => gated_256(c, n, kv, s, m)?,
    }
    if n.o_fused() {
        let wo = kq_weight(w, &n.attn_output)?;
        return k.proj.enqueue_o_resid(
            stream,
            OResidArgs {
                w: wo,
                act: &s.act_attn[i],
                x: &s.x,
                y: &mut s.ffn_inp,
            },
        );
    }
    let Arena {
        x,
        normed,
        q_out,
        act_attn,
        ffn_inp,
        cols,
        ..
    } = s;
    // A gated output projection of another type reads the f32 rows
    // `gated_256` wrote into the free query buffer.
    let rows = q_out.as_ref().ok_or(GpuError::state(
        "qwen3moe::attention",
        "the arena's gated rows",
    ))?;
    out_resid(
        c,
        (n.o_ty, &n.attn_output),
        d.hidden,
        (&act_attn[i], rows),
        m,
        (x, normed, cols.as_mut()),
        ffn_inp,
    )
}

/// The launches [`out_resid`] makes for an output projection of type `ty`
/// at `m` rows: the projection into `normed`, then the residual add.
pub(super) fn out_resid_launches(ty: SiteTy, m: usize) -> usize {
    site_launches(ty, m) + 1
}

/// `y = x + W · rows` for an output projection `name` of type `ty` not
/// folded into its residual add: the projection into `normed` (free once
/// the layer's input projections ran), then the add.
pub(super) fn out_resid(
    c: &Ctx<'_>,
    (ty, name): (SiteTy, &str),
    hidden: usize,
    input: (&Q8Act, &DeviceBuffer<f32>),
    m: usize,
    (x, normed, cols): (
        &DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        Option<&mut DeviceBuffer<f32>>,
    ),
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    site_gemv(c, (ty, name), hidden, input, m, cols, normed)?;
    c.gpu
        .elem()
        .enqueue_add(c.gpu.stream(), x, normed, m * hidden, y)
}

/// The attention's head-256 middle ([`GqaKind::Gated256`]): the q/k norm
/// and the turn of the first 64 values with the cache append, the queries
/// read out of the `[q | gate]` rows into their own buffer; the flash over
/// the rows' live key counts; the output projection's input — for a K-quant
/// `attn_output` quantized with each value multiplied by the sigmoid of its
/// gate, for any other type those products as f32 rows into the query
/// buffer the flash has read.
fn gated_256(
    c: &Ctx<'_>,
    n: &GqaPlan,
    kv: &mut KvPlanes,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    const WHAT: &str = "qwen3moe::gated_256";
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let q35 = k.q35(WHAT)?;
    let stream = gpu.stream();
    let d = s.dims;
    let i = s.col(m)?;
    let q_out = s
        .q_out
        .as_mut()
        .ok_or(GpuError::state(WHAT, "the arena's buffer of gated queries"))?;
    kv.append_256(
        &k.neox,
        stream,
        Append256 {
            qg: &s.q,
            q: q_out,
            k: &mut s.k,
            v: &s.v,
            gq: f32_gain(w, &n.attn_q_norm)?,
            gk: f32_gain(w, &n.attn_k_norm)?,
            table: c.table,
            pos: &s.pos,
            eps: c.eps,
            n_head: d.n_head,
            n_kv: d.n_kv,
            ctx: d.ctx,
            m,
            fault: c.sink,
        },
    )?;
    kv.flash_256(
        &k.flash,
        stream,
        FlashPass {
            q: q_out,
            n_keys: &s.n_keys,
            scale: ATTN_SCALE_256,
            n_kv: d.n_kv,
            ctx: d.ctx,
            m,
            part_v: &mut s.part_v,
            part_ms: &mut s.part_ms,
            fault: c.sink,
            y: &mut s.attn,
        },
        d.n_head,
        n.flash,
        c.mma,
    )?;
    if !n.o_ty.kquant() {
        return q35.q38.enqueue_out_gate(
            stream,
            OutGateArgs {
                attn: &s.attn,
                qg: &s.q,
                n_head: d.n_head,
                m,
                fault: c.sink,
                y: q_out,
            },
        );
    }
    q35.gated.enqueue_q8act(
        stream,
        (&s.attn, &s.q),
        GateLayout {
            head: d.head,
            head_stride: 2 * d.head,
            offset: d.head,
            col_stride: d.q_rows,
        },
        &mut s.act_attn[i],
        m,
        c.sink,
    )
}

/// The FFN half: `ffn_inp` in, `ffn_inp + Σ_s w_s ·
/// down_s(swiglu(gate_s, up_s))` over each token's slots out, into `out`
/// (`None`: into `x`). A plan without a shared expert routes `k` slots a
/// token (the file's `top_k`); with one ([`FfnRoute::Router`]'s `shared`)
/// the gated router adds one more, the shared expert's id in the joined
/// stacks weighted by the sigmoid of the router's last row, and every launch
/// after it runs `k + 1` slots a token. A dense FFN ([`FfnRoute::Dense`])
/// launches no router: its norm is `norm_quant` at every `m`, and the launches
/// after it run its one slot a token from the arena's fixed route.
/// The down `_sel` runs every token's slots in one launch: slot `t · slots
/// + j` is token `t`'s slot `j`, its id, its q8_1 column and its down rows
/// all at that index, so each slot's row is the row a one-token launch
/// writes. Its input is one quantizer launch over every token's slots (each
/// 128-value block is quantized on its own, so a column's bytes do not
/// depend on the columns beside it). At one token the norm runs inside the
/// router's launch (`enqueue_norm_fused`, `norm_quant`'s bytes); at more,
/// `norm_quant` and the `m`-token router.
pub(super) fn ffn(
    c: &Ctx<'_>,
    n: &FfnPlan,
    s: &mut Arena,
    m: usize,
    out: Option<&mut DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    const WHAT: &str = "qwen3moe::ffn";
    if m > GEMV_COLS {
        return wide::ffn(c, n, s, m, out);
    }
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let stream = gpu.stream();
    let i = s.col(m)?;
    let gain = f32_gain(w, &n.ffn_norm)?;
    let (gate_inp, shared) = match &n.route {
        FfnRoute::Router { gate_inp, shared } => (gate_inp, shared),
        FfnRoute::Dense => {
            gpu.fused().enqueue_norm_quant(
                stream,
                &s.ffn_inp,
                gain,
                c.eps,
                &mut s.act_ffn[i],
                &mut s.normed,
                c.sink,
            )?;
            return slots_down(c, n, s, m, out);
        }
    };
    let router = f32_tensor(w, gate_inp)?;
    match shared {
        None if m == 1 => k.router.enqueue_norm_fused(
            stream,
            router,
            &s.ffn_inp,
            gain,
            c.eps,
            &mut s.act_ffn[i],
            c.sink,
            s.route.plain(WHAT)?,
        )?,
        None => {
            gpu.fused().enqueue_norm_quant(
                stream,
                &s.ffn_inp,
                gain,
                c.eps,
                &mut s.act_ffn[i],
                &mut s.normed,
                c.sink,
            )?;
            k.router
                .enqueue_fused(stream, router, &s.normed, m, c.sink, s.route.plain(WHAT)?)?;
        }
        Some(_) if m == 1 => k.q35(WHAT)?.router.enqueue_norm_fused(
            stream,
            router,
            &s.ffn_inp,
            gain,
            c.eps,
            &mut s.act_ffn[i],
            c.sink,
            s.route.gated(WHAT)?,
        )?,
        Some(_) => {
            gpu.fused().enqueue_norm_quant(
                stream,
                &s.ffn_inp,
                gain,
                c.eps,
                &mut s.act_ffn[i],
                &mut s.normed,
                c.sink,
            )?;
            k.q35(WHAT)?.router.enqueue_fused(
                stream,
                router,
                &s.normed,
                m,
                c.sink,
                s.route.gated(WHAT)?,
            )?;
        }
    }
    slots_down(c, n, s, m, out)
}

/// The FFN half after its slots are picked: the gate·up·SwiGLU of every
/// token's slots, one quantizer over them, the down `_sel` and the combine
/// ([`ffn`]'s doc).
fn slots_down(
    c: &Ctx<'_>,
    n: &FfnPlan,
    s: &mut Arena,
    m: usize,
    out: Option<&mut DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    const WHAT: &str = "qwen3moe::ffn";
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let stream = gpu.stream();
    let d = s.dims;
    let i = s.col(m)?;
    let slots = d.slots();
    let used = d.router.map_or(0, |r| r.used());
    if slots != c.p.slots(used) {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "layer {}: the arena is cut for {slots} slots a token, the plan routes {}",
                c.layer,
                c.p.slots(used)
            ),
        ));
    }
    if n.gate_up_fused() {
        k.experts.enqueue_gate_up(
            stream,
            GateUpArgs {
                wg: kq_weight(w, &n.gate)?,
                wu: kq_weight(w, &n.up)?,
                act: &s.act_ffn[i],
                sel: s.route.ids(),
                n_slots: m * slots,
                rows_per_expert: d.ff,
                fault: c.sink,
                h: &mut s.h,
            },
        )?;
    } else {
        // A dense FFN's one slot a token (the load refuses an unfused
        // routed gate·up): the token's gate and up rows apart, then the
        // SwiGLU into `h`.
        let Arena {
            normed,
            act_ffn,
            glu,
            cols,
            h,
            ..
        } = s;
        let g = glu.as_mut().ok_or(GpuError::state(
            WHAT,
            "the arena's gate and up rows (an unfused gate·up)",
        ))?;
        let x = (&act_ffn[i], &*normed);
        site_gemv(c, (n.gate_ty, &n.gate), d.ff, x, m, cols.as_mut(), &mut g.g)?;
        site_gemv(c, (n.up_ty, &n.up), d.ff, x, m, cols.as_mut(), &mut g.u)?;
        gpu.elem()
            .enqueue_swiglu(stream, &g.g, &g.u, m * slots * d.ff, h)?;
    }
    if n.down_sel() {
        let wd = kq_weight(w, &n.down)?;
        gpu.enqueue_quantize_q8_1_layer(&s.h, &mut s.act_h[i], c.layer)?;
        let act_h = &s.act_h[i];
        if n.down_ty == SiteTy::Q4K {
            gpu.q4k_sel().enqueue_gemv_q4k_sel(
                stream,
                wd,
                act_h,
                s.route.ids(),
                m * slots,
                d.hidden,
                &mut s.down,
            )?;
        } else {
            k.q6_sel.enqueue_gemv_q6k_sel(
                stream,
                wd,
                act_h,
                s.route.ids(),
                m * slots,
                d.hidden,
                &mut s.down,
            )?;
        }
    } else {
        // A dense FFN's one slot a token: the down over the token's SwiGLU
        // row is the plain projection, token-major.
        if n.down_ty.kquant() {
            gpu.enqueue_quantize_q8_1_layer(&s.h, &mut s.act_h[i], c.layer)?;
        }
        let Arena {
            h,
            act_h,
            cols,
            down,
            ..
        } = s;
        site_gemv(
            c,
            (n.down_ty, &n.down),
            d.hidden,
            (&act_h[i], h),
            m * slots,
            cols.as_mut(),
            down,
        )?;
    }
    let y = match out {
        Some(y) => y,
        None => &mut s.x,
    };
    k.experts.enqueue_combine_tokens(
        stream,
        CombineArgs {
            down: &s.down,
            w: s.route.weights(),
            resid: &s.ffn_inp,
            rows: d.hidden,
            n_slots: slots,
            m,
            y,
        },
    )
}
