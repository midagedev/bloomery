//! The latent attention mixer of one layer at one token, absorbed and
//! rope-free (ik `src/graphs/build_glm5next.cpp`): `x` in, the update into
//! `out`, the token's latent row and index row appended to the layer's
//! store at its position.
//!
//! The launches, in order:
//! 1. `attn_norm` RMS;
//! 2. the joined `[q_a; latent; index key; pool gate]` projection;
//! 3. `attn_q_a_norm` RMS of the q_a part;
//! 4. the latent's RMS and its f16 row at the position (`latent_rms_append`);
//! 5. the index key's biased LayerNorm and the gate beside it, one f16 row
//!    at the position (`index_key_ln_append`);
//! 6. `attn_q_b`: the heads' queries;
//! 7. `attn_k_b` per head: each query absorbed into the latent width;
//! 8. the attention over every cached position (`ds41_attn_seg` and
//!    `ds41_attn_merge`) with no window rows, a −∞ sink per head (a fold of
//!    nothing) and the scale `1/√key_length_mla`: the latent row is both key
//!    and value;
//! 9. `attn_v_b` per head: each head's latent output to its values;
//! 10. the output projection.
//!
//! Eleven launches. Every cached position is attended: the plan refuses a
//! context past the positions the token-pool indexer keeps whole, so the
//! index rows are written for the selector a longer context will need and
//! read by nothing yet.

use bloomery_gpu::DeviceTensor;
use bloomery_gpu::latent::{IndexKeyArgs, LATENT, LatentAppendArgs, Rows};
use bloomery_gpu::model::Q8_0GemvHeadsArgs;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_deepseek41::attn::AttnArgs;
use model::arch::glm5next::names;

use crate::body::{Parts, Store, f32v, gemv, q8};

/// The launches [`mla`] makes.
pub(crate) const LAUNCHES: usize = 11;

/// Enqueue layer `l`'s latent mixer on `p`'s buffers (module doc).
pub(crate) fn mla(gpu: &Gpu, w: &Weights, p: &mut Parts<'_>, l: usize) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let d = *p.d;
    let fault = gpu.layer_sink(l)?;
    let s = &mut *p.s;
    let Some(Store::Latent { latent, index }) = p.stores.get_mut(l) else {
        return Err(GpuError::State {
            what: "glm5next mla",
            missing: "the layer's latent store",
        });
    };
    let stride = d.stack();
    gpu.elem().enqueue_rms_norm(
        stream,
        &s.x,
        f32v(w, &names::attn_norm(l))?,
        d.rms_eps,
        d.embd,
        1,
        &mut s.xn,
    )?;
    gemv(gpu, w, &names::attn_a_stack(l), &s.xn, &mut s.stack)?;
    gpu.elem().enqueue_rms_norm(
        stream,
        &s.stack,
        f32v(w, &names::attn_q_a_norm(l))?,
        d.rms_eps,
        d.q_lora,
        1,
        &mut s.qr,
    )?;
    let lat = &p.k.latent;
    lat.enqueue_latent_append(
        stream,
        LatentAppendArgs {
            rows: Rows {
                x: &s.stack,
                stride,
                m: 1,
            },
            off: d.q_lora,
            gain: f32v(w, &names::attn_kv_a_norm(l))?,
            pos: &s.pos,
            eps: d.rms_eps,
            fault,
            cache: latent,
        },
    )?;
    lat.enqueue_index_key_append(
        stream,
        IndexKeyArgs {
            rows: Rows {
                x: &s.stack,
                stride,
                m: 1,
            },
            k_off: d.q_lora + LATENT,
            g_off: d.q_lora + LATENT + d.index_d,
            w: f32v(w, &names::indexer_k_norm(l))?,
            b: f32v(w, &names::indexer_k_norm_bias(l))?,
            pos: &s.pos,
            eps: d.norm_eps,
            fault,
            cache: index,
        },
    )?;
    gemv(gpu, w, &names::attn_q_b(l), &s.qr, &mut s.q)?;
    let (qs, dd) = q8(w, &names::attn_k_b(l))?;
    p.k.step.enqueue_q8_0_gemv_heads(
        stream,
        Q8_0GemvHeadsArgs {
            qs,
            d: dd,
            x: &s.q,
            rows_per_head: LATENT,
            x_head_stride: d.head_k,
            y_head_stride: LATENT,
            y_off: 0,
            y: &mut s.qabs,
        },
    )?;
    // SAFETY: a view of no rows at the address of the layer's own latent
    // cache, a live allocation aligned for u16 that outlives the view; the
    // attention reads no window row through it, and the view is released
    // below before the cache can drop.
    let window = unsafe {
        DeviceTensor::<u16>::window(latent.buf().cu_deviceptr(), 0, LATENT, gpu.context())
    };
    let r = p.k.attn.enqueue(
        stream,
        AttnArgs {
            q: &s.qabs,
            window: &window,
            compressed: Some(&*latent),
            selected: None,
            vis: &s.vis,
            sinks: &s.sinks,
            scale: 1.0 / (d.head_k as f32).sqrt(),
            tokens: 1,
            heads: d.heads,
            part_v: &mut s.part_v,
            part_ms: &mut s.part_ms,
            y: &mut s.att,
            fault,
        },
    );
    DeviceTensor::release(window);
    r?;
    let (qs, dd) = q8(w, &names::attn_v_b(l))?;
    p.k.step.enqueue_q8_0_gemv_heads(
        stream,
        Q8_0GemvHeadsArgs {
            qs,
            d: dd,
            x: &s.att,
            rows_per_head: d.head_v,
            x_head_stride: LATENT,
            y_head_stride: d.head_v,
            y_off: 0,
            y: &mut s.av,
        },
    )?;
    gemv(gpu, w, &names::attn_output(l), &s.av, &mut s.out)
}
