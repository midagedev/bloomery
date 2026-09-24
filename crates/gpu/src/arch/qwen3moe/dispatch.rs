//! Launch order: the chain and one layer, enqueued asynchronously against a
//! resident arena. No allocation, no synchronization, no host round trip —
//! so this is both the eager body and what the capture records. One layer
//! body serves both passes: the decode step is its one-row instance, the
//! prompt prefill the same launches at `m` rows.

use super::body::{Body, Kernels, Kq, LayerNames};
use super::experts::{CombineArgs, GateUpArgs};
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
    head.enqueue(gpu, w)
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

/// Enqueue prefill pass `chunk` of `m` tokens (the prompt's tokens `chunk ·
/// MAX_TOKENS ..`) over the prefill arena: every layer at `m` rows, the
/// combine leaving each layer's output in `x`, and with `head`, the last
/// token's row copied into the head's input and the head after it.
pub(super) fn enqueue_prefill_pass(
    gpu: &Gpu,
    w: &Weights,
    b: &mut Body,
    chunk: usize,
    m: usize,
    head: Option<&mut Head>,
) -> Result<(), GpuError> {
    let Body {
        hp,
        names,
        kv,
        k,
        mma,
        prefill,
        ..
    } = b;
    let pf = prefill.as_mut().ok_or(GpuError::state(
        "qwen3moe::enqueue_prefill_pass",
        "no prefill arena",
    ))?;
    let win = pf.windows(chunk, m)?;
    let io = win.io();
    let s = &mut pf.a;
    for (slot, (n, kv)) in names.iter().zip(kv.iter_mut()).enumerate() {
        let c = Ctx {
            gpu,
            w,
            n,
            k,
            mma: *mma,
            eps: hp.rms_eps,
        };
        layer(&c, kv, s, &io, m, slot == 0, None)?;
    }
    match head {
        Some(head) => {
            head.input_mut()
                .copy_from_device_async(&s.x_rows[m - 1], gpu.stream())?;
            head.enqueue(gpu, w)
        }
        None => Ok(()),
    }
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
    gpu.enqueue_quantize_q8_1(&s.attn, &mut s.act_attn[i])?;
    k.proj.enqueue_o_resid(
        stream,
        OResidArgs {
            w: kq_weight(w, &n.attn_output)?,
            act: &s.act_attn[i],
            x: &s.x,
            y: &mut s.ffn_inp,
        },
    )
}

/// The routed FFN half: `ffn_inp` in, `ffn_inp + Σ_s w_s ·
/// down_s(swiglu(gate_s, up_s))` over each token's eight experts out, into
/// `out` (`None`: into `x`). The down `_sel` takes one token's slots per
/// launch.
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
    k.experts.enqueue_gate_up(
        stream,
        GateUpArgs {
            wg: kq_weight(w, &n.ffn_gate_exps)?,
            wu: kq_weight(w, &n.ffn_up_exps)?,
            act: &s.act_ffn[i],
            sel: &s.route.ids,
            n_slots: m * N_USED,
            rows_per_expert: d.ff,
            h: &mut s.h,
        },
    )?;
    let wd = kq_weight(w, &n.ffn_down_exps)?;
    for t in 0..m {
        gpu.enqueue_quantize_q8_1_at(&s.h, t * N_USED * d.ff, &mut s.act_h)?;
        match n.down_ty {
            Kq::Q4K => gpu.q4k_sel().enqueue_gemv_q4k_sel(
                stream,
                wd,
                &s.act_h,
                &s.ids[t],
                N_USED,
                d.hidden,
                &mut s.down_rows[t],
            )?,
            Kq::Q6K => k.q6_sel.enqueue_gemv_q6k_sel(
                stream,
                wd,
                &s.act_h,
                &s.ids[t],
                N_USED,
                d.hidden,
                &mut s.down_rows[t],
            )?,
        }
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
