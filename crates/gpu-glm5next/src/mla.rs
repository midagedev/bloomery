//! The latent attention mixer of one layer at one token, absorbed and
//! rope-free (ik `src/graphs/build_glm5next.cpp`): `x` in, the update into
//! `out`, the token's latent row and index row appended to the layer's
//! store at its position, and the key of the pool it completes.
//!
//! The launches, in order:
//! 1. `attn_norm` RMS;
//! 2. the joined `[q_a; latent; index key; pool gate]` projection;
//! 3. `attn_q_a_norm` RMS of the q_a part;
//! 4. the latent's RMS and its f16 row at the position (`latent_rms_append`);
//! 5. the index key's biased LayerNorm and the gate beside it, one f16 row
//!    at the position (`index_key_ln_append`);
//!
//! then the k-pool selector forks onto the body's branch stream —
//! 6. the key of the pool the token completes ([`pool`], `index_pool`);
//!
//! and [`select`]:
//! 7. `indexer.proj`: the head weights, an f32 gemv of the normed input;
//! 8. `indexer.attn_q_b`: the indexer query, a q8_0 gemv of the q_a norm;
//! 9. the scores of the pools the token sees (`kpool_score`);
//! 10. the list and its length as the attention's visible count
//!     (`qsa_topk_high`) —
//!
//! while the main stream runs
//! 11. `attn_q_b`: the heads' queries;
//! 12. `attn_k_b` per head: each query absorbed into the latent width;
//!
//! and after the branch joins,
//! 13. the attention over the listed positions (`ds41_attn_seg_sel` and
//!     `ds41_attn_merge`) with no window rows, a −∞ sink per head (a fold of
//!     nothing) and the scale `1/√key_length_mla`: the latent row is both key
//!     and value;
//! 14. `attn_v_b` per head: each head's latent output to its values;
//! 15. the output projection.
//!
//! Sixteen launches, the attention's two counted. The list is whole pools
//! and the tail: a token that sees at most `top_k / kpool` complete pools
//! lists every position up to its own, in order — the positions and the
//! order the dense attention reads — and past that the `top_k / kpool` pools
//! of the highest scores, the higher pool of a tie (ik's CPU `top_k`), then
//! the positions of its own incomplete pool. That is the model's cut and
//! mainline's (llama.cpp #27752); ik lists position 0 in a short tail's
//! empty slots besides, which no list here holds.

use bloomery_gpu::kpool::{self, ScoreArgs};
use bloomery_gpu::latent::{IndexKeyArgs, IndexPoolArgs, LATENT, LatentAppendArgs, Rows};
use bloomery_gpu::model::Q8_0GemvHeadsArgs;
use bloomery_gpu::qsa::{TopkHighArgs, list_width};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{DeviceTensor, FaultSink, Gpu, GpuError};
use bloomery_gpu_deepseek41::attn::{AttnArgs, SelectedRows};
use cuda_core::{CudaStream, DeviceBuffer};

use crate::body::{Kernels, Parts, Store, f32t, f32v, gemv, q8};
use crate::tensors::{MixerNames, SelNames, other_kind};

/// The launches [`mla`] makes.
pub(crate) const LAUNCHES: usize = 16;

/// The pools `m` tokens complete, on `stream` (module doc, 6): each token of
/// live count `c` (its `cnt` word, its position plus one) writes the key of
/// pool `c/4 − 1` into `pooled` when `c` is a multiple of four, from rows it
/// and the three before it appended to `index`. Those rows must be appended
/// before it on a stream it is ordered after. A pool whose first rows an
/// earlier call appended is completed by the token that ends it, in whichever
/// call that token runs, so a call that ends mid-pool leaves it to the next.
#[allow(
    clippy::too_many_arguments,
    reason = "the launch's stream, kernels, weights and names, and the counts, rows and plane it reads and writes (rust-quality R8)"
)]
pub(crate) fn pool(
    stream: &CudaStream,
    w: &Weights,
    k: &Kernels,
    n: &SelNames,
    index: &DeviceTensor<u16>,
    cnt: &DeviceBuffer<u32>,
    m: usize,
    fault: FaultSink,
    pooled: &mut DeviceTensor<u16>,
) -> Result<(), GpuError> {
    k.latent.enqueue_index_pool(
        stream,
        IndexPoolArgs {
            cache: index,
            ape: f32v(w, &n.ape)?,
            n_keys: cnt,
            m,
            fault,
            pooled,
        },
    )
}

/// What [`select`] reads and writes for `m` tokens of one latent layer: the
/// stream it launches on, the tokens' normed input (`[m][embd]`) and q_a
/// norm (`[m][q_lora]`), their live counts, the layer's index cache and pool
/// plane, and where the indexer query (`[HEADS·DIM][m]`), the head weights
/// (`[HEADS][m]`), the scores (`[m][pools]`), the lists (`[m][width]`) and
/// the visible counts (`[m][2]`, the second word of each) land.
pub(crate) struct Select<'a> {
    pub stream: &'a CudaStream,
    pub m: usize,
    pub xn: &'a DeviceBuffer<f32>,
    pub qr: &'a DeviceBuffer<f32>,
    pub cnt: &'a DeviceBuffer<u32>,
    pub index: &'a DeviceTensor<u16>,
    pub pooled: &'a DeviceTensor<u16>,
    pub qi: &'a mut DeviceBuffer<f32>,
    pub wi: &'a mut DeviceBuffer<f32>,
    pub scores: &'a mut DeviceBuffer<f32>,
    pub list: &'a mut DeviceBuffer<u32>,
    pub vis: &'a mut DeviceBuffer<u32>,
    /// Pools a token keeps once it sees more.
    pub kept: usize,
    pub fault: FaultSink,
}

/// The k-pool selector of one latent layer for `m` tokens, on `a.stream`:
/// the head weights, the indexer query, the scores and the lists, four
/// launches (module doc, 7–10). Every pool the tokens see must be written
/// before it ([`pool`]) on a stream it is ordered after. The gemvs take at
/// most [`bloomery_gpu::COL_GROUP`] tokens: the step calls it with one, a
/// prompt batch with each chunk's.
pub(crate) fn select(
    gpu: &Gpu,
    w: &Weights,
    k: &Kernels,
    n: &SelNames,
    a: Select<'_>,
) -> Result<(), GpuError> {
    let ctx = a.index.rows();
    gpu.q8f32()
        .enqueue_f32_gemv(a.stream, f32t(w, &n.proj)?, a.xn, a.m, a.wi)?;
    let (qs, dd) = q8(w, &n.q_b)?;
    gpu.q8f32()
        .enqueue_q8_0_gemv(a.stream, qs, dd, a.qr, a.m, a.qi)?;
    k.kpool.enqueue_score(
        a.stream,
        ScoreArgs {
            q: a.qi,
            w: a.wi,
            n_keys: a.cnt,
            pooled: a.pooled,
            tokens: a.m,
            ctx,
            kept: a.kept,
            scale: kpool::weights_scale(kpool::HEADS, kpool::DIM),
            fault: a.fault,
            scores: a.scores,
        },
    )?;
    k.qsa.enqueue_topk_high(
        a.stream,
        TopkHighArgs {
            n_keys: a.cnt,
            scores: a.scores,
            ctx,
            kept: a.kept,
            m: a.m,
            fault: a.fault,
            list: a.list,
            vis: a.vis,
        },
    )
}

/// Enqueue layer `l`'s latent mixer on `p`'s buffers (module doc).
pub(crate) fn mla(gpu: &Gpu, w: &Weights, p: &mut Parts<'_>, l: usize) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let d = *p.d;
    let fault = gpu.layer_sink(l)?;
    let Some(MixerNames::Latent(n)) = p.names.get(l).map(|n| &n.mixer) else {
        return Err(other_kind("glm5next mla", l));
    };
    let s = &mut *p.s;
    let Some(Store::Latent {
        latent,
        index,
        pooled,
    }) = p.stores.get_mut(l)
    else {
        return Err(GpuError::State {
            what: "glm5next mla",
            missing: "the layer's latent store",
        });
    };
    let stride = d.stack();
    gpu.elem().enqueue_rms_norm(
        stream,
        &s.x,
        f32v(w, &n.norm)?,
        d.rms_eps,
        d.embd,
        1,
        &mut s.xn,
    )?;
    gemv(gpu, w, &n.stack, &s.xn, &mut s.stack)?;
    gpu.elem().enqueue_rms_norm(
        stream,
        &s.stack,
        f32v(w, &n.q_a_norm)?,
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
            gain: f32v(w, &n.kv_a_norm)?,
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
            w: f32v(w, &n.index_norm)?,
            b: f32v(w, &n.index_norm_bias)?,
            pos: &s.pos,
            eps: d.norm_eps,
            fault,
            cache: index,
        },
    )?;
    let fork = p.k.branch.fork(stream)?;
    pool(
        fork.stream(),
        w,
        p.k,
        &n.sel,
        index,
        &s.cnt,
        1,
        fault,
        pooled,
    )?;
    select(
        gpu,
        w,
        p.k,
        &n.sel,
        Select {
            stream: fork.stream(),
            m: 1,
            xn: &s.xn,
            qr: &s.qr,
            cnt: &s.cnt,
            index,
            pooled,
            qi: &mut s.qi,
            wi: &mut s.wi,
            scores: &mut s.scores,
            list: &mut s.list,
            vis: &mut s.vis,
            kept: d.kept,
            fault,
        },
    )?;
    gemv(gpu, w, &n.q_b, &s.qr, &mut s.q)?;
    let (qs, dd) = q8(w, &n.k_b)?;
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
    fork.join()?;
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
            selected: Some(SelectedRows {
                rows: &s.list,
                stride: list_width(d.kept),
            }),
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
    let (qs, dd) = q8(w, &n.v_b)?;
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
    gemv(gpu, w, &n.out, &s.av, &mut s.out)
}
