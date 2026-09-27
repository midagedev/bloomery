//! Launch order: the chain and one layer, enqueued asynchronously against a
//! resident arena. No allocation, no synchronization, no host round trip —
//! so this is both the eager body and what the capture records. One layer
//! body serves every pass and both bodies of the family: the decode step is
//! its one-row instance, a pass its `m`-row instance, and a layer's
//! [`LayerPlan`] picks its mixer — attention over K/V planes, or the delta
//! rule over a recurrent store ([`super::delta`]) — and whether its experts
//! carry a folded shared expert. Which kernel an op launches follows from
//! `m` and the plan inside the op. At one row the FFN norm is folded into
//! the router's launch, which writes the bytes of the pair it replaces, so a
//! one-row pass and a multi-row one leave the same state. The q8_1 of the
//! attention rows and of the SwiGLU rows stay launches of their own: folded
//! into the projection beside them, the work lands in every block of a grid
//! that already streams its weights at the card's bandwidth.

use super::body::{ATTN_SCALE, ATTN_SCALE_256, Body, Kernels};
use super::delta;
use super::experts::{CombineArgs, GateUpArgs};
use super::head_argmax::HeadArgmaxState;
use super::plan::{GqaKind, GqaPlan, Kq, LayerPlan, MixerPlan, MoePlan};
use super::program::{Program, Tail};
use super::proj::{OResidArgs, QkvArgs};
use super::scratch::{Arena, Io, KvPlanes, StoreMut};
use crate::elem::EmbedRowsArgs;
use crate::flash_gqa::GqaArgs;
use crate::gated_quant::GateLayout;
use crate::head::Head;
use crate::model::lookup::{f32_gain, f32_tensor, kq_weight};
use crate::rope_neox::{NeoxArgs, PartialNeoxArgs};
use crate::weights::Weights;
use crate::{FaultSink, Gpu, GpuError};
use cuda_core::DeviceBuffer;
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
/// (token, fault word) readback.
pub(super) fn enqueue_head(
    gpu: &Gpu,
    w: &Weights,
    k: &Kernels,
    state: &mut HeadArgmaxState,
    head: &mut Head,
) -> Result<(), GpuError> {
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
    let Body { s, sp, .. } = b;
    embed_rows(gpu, w, &sp.io(), s)
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
///   projection's gemv, and at more than one row its token-major copy.
/// - Delta rule ([`delta::launches`]): norm+quant, the two projection
///   launches, conv, delta, gated norm, quantizer and output projection —
///   8 — plus a Q6_K q·k·v projection's token-major copy at more than one
///   row.
/// - FFN: the norm with the router (one launch at one row, two at more),
///   gate·up, one quantizer over every token's slots, the down `_sel` over
///   every token's slots, and the combine — 5 or 6.
///
/// So a qwen3moe layer is 12 or 13 launches at one row and 13 or 15 at
/// every `m` from two on.
pub(super) fn pass_launches(plans: &[LayerPlan], m: usize) -> usize {
    let many = usize::from(m > 1);
    let per_layer = |p: &LayerPlan| {
        let mixer = match &p.mixer {
            MixerPlan::Gqa(g) => 7 + usize::from(g.v_ty == Kq::Q6K) * (1 + many),
            MixerPlan::Delta(d) => delta::launches(d, m),
        };
        mixer + 5 + many
    };
    1 + plans.iter().map(per_layer).sum::<usize>()
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
    if embed {
        embed_rows(c.gpu, c.w, io, s)?;
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
    ffn(c, &c.p.ffn, s, m, out)
}

/// A unit's front, one launch: the embedding rows of `io`'s ids into `s.x`
/// and each row's position and live key count into `s.pos` and `s.n_keys` —
/// the rows every later launch of the unit reads.
pub(super) fn embed_rows(
    gpu: &Gpu,
    w: &Weights,
    io: &Io<'_>,
    s: &mut Arena,
) -> Result<(), GpuError> {
    gpu.elem().enqueue_embed_rows_q4k(
        gpu.stream(),
        EmbedRowsArgs {
            w: kq_weight(w, &token_embd())?,
            ids: io.ids,
            pos0: io.pos0,
            first: io.first,
            y: &mut s.x,
            pos: &mut s.pos,
            n_keys: &mut s.n_keys,
        },
    )
}

/// The attention half: `x` in, `ffn_inp = x + attn_output(attn(x))` out —
/// q, k and v in one launch (a Q6_K v in its own), the head norm and the
/// rope by the rows' positions with the cache append, the flash over the
/// rows' live key counts, the output projection with the residual add in its
/// store. At head 256 the query projection writes each head's `[q | gate]`,
/// the rope reads the queries out of those rows and turns the first 64
/// values, and the output projection's quantizer multiplies the flash
/// output by `σ(gate)` ([`GqaKind::Gated256`]).
fn attention(
    c: &Ctx<'_>,
    n: &GqaPlan,
    kv: &mut KvPlanes,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let stream = gpu.stream();
    let d = s.dims;
    let i = m - 1;
    gpu.fused().enqueue_norm_quant(
        stream,
        &s.x,
        f32_gain(w, &n.attn_norm)?,
        c.eps,
        &mut s.act_x[i],
        &mut s.normed,
        c.sink,
    )?;
    let wv = kq_weight(w, &n.attn_v)?;
    let (off_k, off_v) = s.qkv_offsets();
    k.proj.enqueue_qkv(
        stream,
        QkvArgs {
            wq: kq_weight(w, &n.attn_q)?,
            wk: kq_weight(w, &n.attn_k)?,
            wv: (n.v_ty == Kq::Q4K).then_some(wv),
            act: &s.act_x[i],
            y: &mut s.qkv,
            off_k,
            off_v,
        },
    )?;
    if n.v_ty == Kq::Q6K {
        match s.v_cols.as_mut().filter(|_| m > 1) {
            Some(cols) => {
                gpu.enqueue_gemv_q6k(wv, &s.act_x[i], cols)?;
                k.proj
                    .enqueue_token_major(stream, cols, d.kv_len(), m, &mut s.v)?;
            }
            None => gpu.enqueue_gemv_q6k(wv, &s.act_x[i], &mut s.v)?,
        }
    }
    match n.kind {
        GqaKind::Neox128 => {
            k.neox.enqueue_head_norm_neox_append(
                stream,
                NeoxArgs {
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
                    cache_k: &mut kv.k,
                    cache_v: &mut kv.v,
                },
            )?;
            k.flash.enqueue_pass(
                stream,
                GqaArgs {
                    q: &s.q,
                    kc: &kv.k,
                    vc: &kv.v,
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
    let wo = kq_weight(w, &n.attn_output)?;
    k.proj.enqueue_o_resid(
        stream,
        OResidArgs {
            w: wo,
            act: &s.act_attn[i],
            x: &s.x,
            y: &mut s.ffn_inp,
        },
    )
}

/// The attention's head-256 middle ([`GqaKind::Gated256`]): the q/k norm
/// and the turn of the first 64 values with the cache append, the queries
/// read out of the `[q | gate]` rows into their own buffer; the flash over
/// the rows' live key counts; the output projection's input quantized with
/// each value multiplied by the sigmoid of its gate.
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
    let q_out = s
        .q_out
        .as_mut()
        .ok_or(GpuError::state(WHAT, "the arena's buffer of gated queries"))?;
    k.neox.enqueue_head_norm_neox_append_256(
        stream,
        PartialNeoxArgs {
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
            cache_k: &mut kv.k,
            cache_v: &mut kv.v,
        },
    )?;
    k.flash.enqueue_pass_256(
        stream,
        GqaArgs {
            q: q_out,
            kc: &kv.k,
            vc: &kv.v,
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
        c.mma,
    )?;
    q35.gated.enqueue_q8act(
        stream,
        (&s.attn, &s.q),
        GateLayout {
            head: d.head,
            head_stride: 2 * d.head,
            offset: d.head,
            col_stride: d.q_rows,
        },
        &mut s.act_attn[m - 1],
        m,
        c.sink,
    )
}

/// The routed FFN half: `ffn_inp` in, `ffn_inp + Σ_s w_s ·
/// down_s(swiglu(gate_s, up_s))` over each token's slots out, into `out`
/// (`None`: into `x`). A plan without a shared expert routes `k` slots a
/// token (the file's `top_k`); with one ([`MoePlan::shared`]) the gated
/// router adds one more, the shared expert's id in the joined stacks
/// weighted by the sigmoid of the router's last row, and every launch after
/// it runs `k + 1` slots a token.
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
    n: &MoePlan,
    s: &mut Arena,
    m: usize,
    out: Option<&mut DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    const WHAT: &str = "qwen3moe::ffn";
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let stream = gpu.stream();
    let d = s.dims;
    let i = m - 1;
    let router = f32_tensor(w, &n.ffn_gate_inp)?;
    let gain = f32_gain(w, &n.ffn_norm)?;
    match n.shared {
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
    let slots = d.slots();
    if slots != c.p.slots(d.router.used()) {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "layer {}: the arena is cut for {slots} slots a token, the plan routes {}",
                c.layer,
                c.p.slots(d.router.used())
            ),
        ));
    }
    let gate_up = GateUpArgs {
        wg: kq_weight(w, &n.ffn_gate_exps)?,
        wu: kq_weight(w, &n.ffn_up_exps)?,
        act: &s.act_ffn[i],
        sel: s.route.ids(),
        n_slots: m * slots,
        rows_per_expert: d.ff,
        fault: c.sink,
        h: &mut s.h,
    };
    k.experts.enqueue_gate_up(stream, gate_up)?;
    let wd = kq_weight(w, &n.ffn_down_exps)?;
    gpu.enqueue_quantize_q8_1_layer(&s.h, &mut s.act_h[i], c.layer)?;
    let act_h = &s.act_h[i];
    match n.down_ty {
        Kq::Q4K => gpu.q4k_sel().enqueue_gemv_q4k_sel(
            stream,
            wd,
            act_h,
            s.route.ids(),
            m * slots,
            d.hidden,
            &mut s.down,
        )?,
        Kq::Q6K => k.q6_sel.enqueue_gemv_q6k_sel(
            stream,
            wd,
            act_h,
            s.route.ids(),
            m * slots,
            d.hidden,
            &mut s.down,
        )?,
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
