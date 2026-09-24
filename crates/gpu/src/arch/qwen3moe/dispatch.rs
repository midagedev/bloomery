//! Launch order: the chain and one layer, enqueued asynchronously against
//! the resident arena. No allocation, no synchronization, no host round
//! trip — so this is both the eager body and what the capture records.

use super::body::{Body, Kq, LayerNames};
use super::experts::GateUpArgs;
use super::router::N_USED;
use super::scratch::{Arena, KvPlanes};
use crate::flash_gqa::{GqaArgs, HEAD};
use crate::head::Head;
use crate::model::lookup::{f32_gain, f32_tensor, kq_weight};
use crate::rope_neox::NeoxArgs;
use crate::weights::Weights;
use crate::{Gpu, GpuError};
use model::arch::qwen3moe::names::token_embd;

/// Enqueue the whole decode chain at m = 1: layer 0 with its embedding,
/// every later layer reading the previous layer's output, then the head.
/// The residual crosses a layer boundary as one device copy (`l_out` into
/// `x`, a memcpy node in the capture); the last layer copies into the
/// head's input instead.
pub(super) fn enqueue_chain(
    gpu: &Gpu,
    w: &Weights,
    b: &mut Body,
    head: &mut Head,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let last = b.names.len() - 1;
    for slot in 0..b.names.len() {
        enqueue_layer(gpu, w, b, slot, slot == 0)?;
        if let Some(t) = b.taps.as_mut() {
            t.rows[slot].copy_from_device_async(&b.s.l_out, stream)?;
        }
        if slot == last {
            head.input_mut()
                .copy_from_device_async(&b.s.l_out, stream)?;
        } else {
            b.s.x.copy_from_device_async(&b.s.l_out, stream)?;
        }
    }
    head.enqueue(gpu, w)
}

/// Enqueue layer `slot`'s step, embedding the token into `x` in front when
/// `embed`; `x` in, `l_out` out, the step's K/V row appended to the layer's
/// planes.
pub(super) fn enqueue_layer(
    gpu: &Gpu,
    w: &Weights,
    b: &mut Body,
    slot: usize,
    embed: bool,
) -> Result<(), GpuError> {
    let Body {
        hp,
        names,
        kv,
        s,
        k,
        mma,
        ..
    } = b;
    let n = &names[slot];
    if embed {
        gpu.elem().enqueue_embed_rows_q4k(
            gpu.stream(),
            kq_weight(w, &token_embd())?,
            &s.token_buf,
            &mut s.x,
        )?;
    }
    attention(gpu, w, n, &mut kv[slot], s, k, *mma, hp.rms_eps)?;
    ffn(gpu, w, n, s, k, hp.rms_eps)
}

/// The attention half: `x` in, `ffn_inp = x + attn_output(attn(x))` out.
#[allow(
    clippy::too_many_arguments,
    reason = "a stage of `enqueue_layer`, taking its caller's arguments (rust-quality R8)"
)]
fn attention(
    gpu: &Gpu,
    w: &Weights,
    n: &LayerNames,
    kv: &mut KvPlanes,
    s: &mut Arena,
    k: &super::body::Kernels,
    mma: bool,
    eps: f32,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let d = s.dims;
    gpu.fused().enqueue_norm_quant(
        stream,
        &s.x,
        f32_gain(w, &n.attn_norm)?,
        eps,
        &mut s.act_x,
        &mut s.normed,
    )?;
    gpu.enqueue_gemv_q4k(kq_weight(w, &n.attn_q)?, &s.act_x, &mut s.q)?;
    gpu.enqueue_gemv_q4k(kq_weight(w, &n.attn_k)?, &s.act_x, &mut s.k)?;
    let wv = kq_weight(w, &n.attn_v)?;
    match n.v_ty {
        Kq::Q4K => gpu.enqueue_gemv_q4k(wv, &s.act_x, &mut s.v)?,
        Kq::Q6K => gpu.enqueue_gemv_q6k(wv, &s.act_x, &mut s.v)?,
    }
    k.neox.enqueue_head_norm_neox_append(
        stream,
        NeoxArgs {
            q: &mut s.q,
            k: &mut s.k,
            v: &s.v,
            gq: f32_gain(w, &n.attn_q_norm)?,
            gk: f32_gain(w, &n.attn_k_norm)?,
            cs: &s.cs_buf,
            pos: &s.pos_buf,
            eps,
            n_head: d.n_head,
            n_kv: d.n_kv,
            ctx: d.ctx,
            m: 1,
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
            n_keys: &s.n_keys_buf,
            scale: 1.0 / (HEAD as f32).sqrt(),
            n_kv: d.n_kv,
            ctx: d.ctx,
            part_v: &mut s.part_v,
            part_ms: &mut s.part_ms,
            y: &mut s.attn,
        },
        mma,
    )?;
    gpu.enqueue_quantize_q8_1(&s.attn, &mut s.act_attn)?;
    gpu.enqueue_gemv_q4k(kq_weight(w, &n.attn_output)?, &s.act_attn, &mut s.attn_out)?;
    gpu.elem()
        .enqueue_add(stream, &s.attn_out, &s.x, d.hidden, &mut s.ffn_inp)
}

/// Enqueue layer `slot`'s FFN half alone: `ffn_inp` in, `l_out` out.
pub(super) fn enqueue_ffn(
    gpu: &Gpu,
    w: &Weights,
    b: &mut Body,
    slot: usize,
) -> Result<(), GpuError> {
    let Body {
        hp, names, s, k, ..
    } = b;
    ffn(gpu, w, &names[slot], s, k, hp.rms_eps)
}

/// The routed FFN half: `ffn_inp` in, `l_out = ffn_inp + Σ_s w_s ·
/// down_s(swiglu(gate_s, up_s))` out over the router's eight experts.
fn ffn(
    gpu: &Gpu,
    w: &Weights,
    n: &LayerNames,
    s: &mut Arena,
    k: &super::body::Kernels,
    eps: f32,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let d = s.dims;
    gpu.fused().enqueue_norm_quant(
        stream,
        &s.ffn_inp,
        f32_gain(w, &n.ffn_norm)?,
        eps,
        &mut s.act_ffn,
        &mut s.normed,
    )?;
    gpu.q8f32().enqueue_f32_gemv(
        stream,
        f32_tensor(w, &n.ffn_gate_inp)?,
        &s.normed,
        1,
        &mut s.logits,
    )?;
    k.router.enqueue(stream, &s.logits, &mut s.route)?;
    k.experts.enqueue_gate_up(
        stream,
        GateUpArgs {
            wg: kq_weight(w, &n.ffn_gate_exps)?,
            wu: kq_weight(w, &n.ffn_up_exps)?,
            act: &s.act_ffn,
            sel: &s.route.ids,
            n_slots: N_USED,
            rows_per_expert: d.ff,
            h: &mut s.h,
        },
    )?;
    gpu.enqueue_quantize_q8_1(&s.h, &mut s.act_h)?;
    let wd = kq_weight(w, &n.ffn_down_exps)?;
    match n.down_ty {
        Kq::Q4K => gpu.q4k_sel().enqueue_gemv_q4k_sel(
            stream,
            wd,
            &s.act_h,
            &s.route.ids,
            N_USED,
            d.hidden,
            &mut s.down,
        )?,
        Kq::Q6K => k.q6_sel.enqueue_gemv_q6k_sel(
            stream,
            wd,
            &s.act_h,
            &s.route.ids,
            N_USED,
            d.hidden,
            &mut s.down,
        )?,
    }
    k.experts.enqueue_combine(
        stream,
        &s.down,
        &s.route.weights,
        &s.ffn_inp,
        N_USED,
        &mut s.l_out,
    )
}
