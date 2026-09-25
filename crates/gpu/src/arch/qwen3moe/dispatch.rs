//! Launch order: the chain and one layer, enqueued asynchronously against a
//! resident arena. No allocation, no synchronization, no host round trip —
//! so this is both the eager body and what the capture records. One layer
//! body serves both passes: the decode step is its one-row instance, the
//! prompt prefill its `m`-row instance. At one row the FFN norm is folded
//! into the router's launch, which writes the bytes of the pair it replaces,
//! so a one-row prefill pass and a multi-row one leave the same state. The
//! q8_1 of the attention rows and of the SwiGLU rows stay launches of their
//! own: folded into the projection beside them, the work lands in every
//! block of a grid that already streams its weights at the card's bandwidth.

use super::body::{Body, Kernels, Kq, LayerNames};
use super::experts::{CombineArgs, GateUpArgs};
use super::head_argmax::HeadArgmaxState;
use super::proj::{OResidArgs, QkvArgs};
use super::router::N_USED;
use super::scratch::{Arena, Io, KvPlanes};
use crate::flash_gqa::{GqaArgs, HEAD};
use crate::head::Head;
use crate::model::lookup::{f32_gain, f32_tensor, kq_weight};
use crate::rope_neox::NeoxArgs;
use crate::weights::Weights;
use crate::{Gpu, GpuError};
use cuda_core::DeviceBuffer;
use model::arch::qwen3moe::names::token_embd;

/// What every launch of one layer reads besides the arena and the cache:
/// the engine, the weights, the layer's names, the kernels, the flash pass
/// and the norm epsilon.
struct Ctx<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    n: &'a LayerNames,
    k: &'a Kernels,
    mma: bool,
    eps: f32,
}

/// Enqueue the whole decode chain at m = 1: layer 0 with its embedding,
/// every later layer reading the previous layer's output, then the head.
/// The combine writes a layer's output where the next reader takes it — the
/// arena's `x` for the next layer, the head's input after the last — so no
/// copy crosses a layer boundary.
pub(super) fn enqueue_chain(
    gpu: &Gpu,
    w: &Weights,
    b: &mut Body,
    head: &mut Head,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let last = b.names.len() - 1;
    for slot in 0..b.names.len() {
        let out = (slot == last).then(|| head.input_mut());
        enqueue_layer(gpu, w, b, slot, slot == 0, out)?;
        if let Some(t) = b.taps.as_mut() {
            let src: &DeviceBuffer<f32> = if slot == last {
                head.input_mut()
            } else {
                &b.s.x
            };
            t.rows[slot].copy_from_device_async(src, stream)?;
        }
    }
    enqueue_head(gpu, w, &b.k, &mut b.head_state, head)
}

/// Enqueue the one-row head: its norm and quantization, then the Q6_K
/// projection with the argmax folded in (`head_argmax`) in place of the
/// shared head's gemv and `argmax_fault` — the same logits and the same
/// (token, fault word) readback.
fn enqueue_head(
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

/// Enqueue layer `slot`'s step, embedding the token into `x` in front when
/// `embed`; `x` in, the output residual into `out` (`None`: back into `x`),
/// the step's K/V row appended to the layer's planes.
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
        names,
        kv,
        s,
        sp,
        k,
        mma,
        ..
    } = b;
    let c = Ctx {
        gpu,
        w,
        n: &names[slot],
        k,
        mma: *mma,
        eps: hp.rms_eps,
    };
    layer(&c, &mut kv[slot], s, &sp.io(), 1, embed, out)
}

/// What every launch of a prefill pass reads besides its arena, its inputs
/// and the cache: the engine, the weights, every layer's names, the kernels,
/// the flash pass and the norm epsilon.
pub(super) struct PassCtx<'a> {
    pub(super) gpu: &'a Gpu,
    pub(super) w: &'a Weights,
    pub(super) names: &'a [LayerNames],
    pub(super) k: &'a Kernels,
    pub(super) mma: bool,
    pub(super) eps: f32,
}

/// Enqueue one prefill pass of `m` tokens over arena `s`, reading the
/// tokens, positions, live key counts and rope rows in `io`: every layer at
/// `m` rows, the combine leaving each layer's output in `x`. The same
/// enqueues run eager and under a capture.
pub(super) fn enqueue_pass(
    c: &PassCtx<'_>,
    kv: &mut [KvPlanes],
    s: &mut Arena,
    io: &Io<'_>,
    m: usize,
) -> Result<(), GpuError> {
    for (slot, (n, kv)) in c.names.iter().zip(kv.iter_mut()).enumerate() {
        let lc = Ctx {
            gpu: c.gpu,
            w: c.w,
            n,
            k: c.k,
            mma: c.mma,
            eps: c.eps,
        };
        layer(&lc, kv, s, io, m, slot == 0, None)?;
    }
    Ok(())
}

/// Enqueue the head after the last prefill pass of `m` tokens: that pass's
/// last row of `x` into the head's input, then the one-row head.
pub(super) fn enqueue_pass_head(
    gpu: &Gpu,
    w: &Weights,
    k: &Kernels,
    state: &mut HeadArgmaxState,
    s: &Arena,
    m: usize,
    head: &mut Head,
) -> Result<(), GpuError> {
    let row = m
        .checked_sub(1)
        .and_then(|i| s.x_rows.get(i))
        .ok_or_else(|| {
            GpuError::shape(
                "qwen3moe::enqueue_pass_head",
                format!("a pass of {m} rows on a {}-row arena", s.rows),
            )
        })?;
    head.input_mut().copy_from_device_async(row, gpu.stream())?;
    enqueue_head(gpu, w, k, state, head)
}

/// The launches [`enqueue_pass`] makes at `m` rows over layers `names`,
/// counted from [`layer`]'s enqueues: the embedding, then per layer the
/// attention half — norm+quant, q·k·v, QK-norm+rope+append, the flash's
/// segment pass and merge, the attention rows' quantizer and the output
/// projection, plus a Q6_K value projection's gemv (and at more than one row
/// its token-major copy) — and the FFN half: the norm with the router (one
/// launch at one row, two at more), gate·up, one quantizer over every
/// token's slots, the down `_sel` over every token's slots, and the combine.
/// So 12 or 13 per layer at one row and 13 or 15 at every `m` from two on.
pub(super) fn pass_launches(names: &[LayerNames], m: usize) -> usize {
    let per_layer = |n: &LayerNames| {
        let q6v = usize::from(n.v_ty == Kq::Q6K);
        if m == 1 { 12 + q6v } else { 13 + 2 * q6v }
    };
    1 + names.iter().map(per_layer).sum::<usize>()
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
        names,
        s,
        k,
        mma,
        ..
    } = b;
    let c = Ctx {
        gpu,
        w,
        n: &names[slot],
        k,
        mma: *mma,
        eps: hp.rms_eps,
    };
    ffn(&c, s, 1, None)
}

/// One layer at `m` rows: the embedding rows into `x` in front when
/// `embed`, the attention half, then the FFN half into `out` (`None`: back
/// into `x`).
fn layer(
    c: &Ctx<'_>,
    kv: &mut KvPlanes,
    s: &mut Arena,
    io: &Io<'_>,
    m: usize,
    embed: bool,
    out: Option<&mut DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    if embed {
        c.gpu.elem().enqueue_embed_rows_q4k(
            c.gpu.stream(),
            kq_weight(c.w, &token_embd())?,
            io.tokens,
            &mut s.x,
        )?;
    }
    attention(c, kv, s, io, m)?;
    ffn(c, s, m, out)
}

/// The attention half: `x` in, `ffn_inp = x + attn_output(attn(x))` out —
/// q, k and v in one launch (a Q6_K v in its own), the output projection
/// with the residual add in its store.
fn attention(
    c: &Ctx<'_>,
    kv: &mut KvPlanes,
    s: &mut Arena,
    io: &Io<'_>,
    m: usize,
) -> Result<(), GpuError> {
    let (gpu, w, n, k) = (c.gpu, c.w, c.n, c.k);
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
        gpu.unlabelled_sink(),
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
                    .enqueue_token_major(stream, cols, d.n_kv * HEAD, m, &mut s.v)?;
            }
            None => gpu.enqueue_gemv_q6k(wv, &s.act_x[i], &mut s.v)?,
        }
    }
    k.neox.enqueue_head_norm_neox_append(
        stream,
        NeoxArgs {
            q: &mut s.q,
            k: &mut s.k,
            v: &s.v,
            gq: f32_gain(w, &n.attn_q_norm)?,
            gk: f32_gain(w, &n.attn_k_norm)?,
            cs: io.cs,
            pos: io.pos,
            eps: c.eps,
            n_head: d.n_head,
            n_kv: d.n_kv,
            ctx: d.ctx,
            m,
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
            n_keys: io.n_keys,
            scale: 1.0 / (HEAD as f32).sqrt(),
            n_kv: d.n_kv,
            ctx: d.ctx,
            m,
            part_v: &mut s.part_v,
            part_ms: &mut s.part_ms,
            y: &mut s.attn,
        },
        c.mma,
    )?;
    let wo = kq_weight(w, &n.attn_output)?;
    gpu.enqueue_quantize_q8_1(&s.attn, &mut s.act_attn[i])?;
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

/// The routed FFN half: `ffn_inp` in, `ffn_inp + Σ_s w_s ·
/// down_s(swiglu(gate_s, up_s))` over each token's eight experts out, into
/// `out` (`None`: into `x`). The down `_sel` runs every token's slots in one
/// launch: slot `t · N_USED + j` is token `t`'s slot `j`, its id, its q8_1
/// column and its down rows all at that index, so each slot's row is the
/// row a one-token launch writes. Its input is one quantizer launch over
/// every token's slots (each 128-value block is quantized on its own, so a
/// column's bytes do not depend on the columns beside it). At one token the
/// norm runs inside the router's launch (`enqueue_norm_fused`,
/// `norm_quant`'s bytes); at more, `norm_quant` and the `m`-token router.
fn ffn(
    c: &Ctx<'_>,
    s: &mut Arena,
    m: usize,
    out: Option<&mut DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    let (gpu, w, n, k) = (c.gpu, c.w, c.n, c.k);
    let stream = gpu.stream();
    let d = s.dims;
    let i = m - 1;
    if m == 1 {
        k.router.enqueue_norm_fused(
            stream,
            f32_tensor(w, &n.ffn_gate_inp)?,
            &s.ffn_inp,
            f32_gain(w, &n.ffn_norm)?,
            c.eps,
            &mut s.act_ffn[i],
            gpu.unlabelled_sink(),
            &mut s.route,
        )?;
    } else {
        gpu.fused().enqueue_norm_quant(
            stream,
            &s.ffn_inp,
            f32_gain(w, &n.ffn_norm)?,
            c.eps,
            &mut s.act_ffn[i],
            &mut s.normed,
            gpu.unlabelled_sink(),
        )?;
        k.router.enqueue_fused(
            stream,
            f32_tensor(w, &n.ffn_gate_inp)?,
            &s.normed,
            m,
            &mut s.route,
        )?;
    }
    let gate_up = GateUpArgs {
        wg: kq_weight(w, &n.ffn_gate_exps)?,
        wu: kq_weight(w, &n.ffn_up_exps)?,
        act: &s.act_ffn[i],
        sel: &s.route.ids,
        n_slots: m * N_USED,
        rows_per_expert: d.ff,
        h: &mut s.h,
    };
    k.experts.enqueue_gate_up(stream, gate_up)?;
    let wd = kq_weight(w, &n.ffn_down_exps)?;
    gpu.enqueue_quantize_q8_1(&s.h, &mut s.act_h[i])?;
    let act_h = &s.act_h[i];
    match n.down_ty {
        Kq::Q4K => gpu.q4k_sel().enqueue_gemv_q4k_sel(
            stream,
            wd,
            act_h,
            &s.route.ids,
            m * N_USED,
            d.hidden,
            &mut s.down,
        )?,
        Kq::Q6K => k.q6_sel.enqueue_gemv_q6k_sel(
            stream,
            wd,
            act_h,
            &s.route.ids,
            m * N_USED,
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
            w: &s.route.weights,
            resid: &s.ffn_inp,
            rows: d.hidden,
            n_slots: N_USED,
            m,
            y,
        },
    )
}
