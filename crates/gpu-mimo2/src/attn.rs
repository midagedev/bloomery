//! The attention sub-layer of one token: `x` (the layer's input) in, the
//! residual after the attention `x1` out. Seven launches, the same on every
//! layer. A store shorter than the context is the ring
//! ([`enqueue_append`], [`enqueue_pass`]); a store that fills it is the plane
//! entry. The launches are `attn_norm` RMS, the fused q·k·v projection, the
//! rope-and-append (the key heads turned in place, the value rows times the
//! layer's multiplier, both into the f16 planes), the K192 flash's segment
//! pass and its merge, the output projection, and the add of `x`.
//!
//! MiMo facts that fix this file: the score head is 192 values and the value
//! head 128 (the planes' rows differ in width, [`flash_gqa::HEAD_K192`] and
//! [`flash_gqa::HEAD`]); the nine full layers have 4 key/value heads over 64
//! query heads (group 16, every key) and the 39 window layers 8 (group 8,
//! 128 positions and the per-head sinks); the rope turns 64 of the 192 score
//! values at the layer's own base; every layer multiplies its value rows by
//! one constant before the f16 cache; the scores' scale is `1/√192`. The
//! layer's window, sinks, base and multiplier come from its description
//! ([`facts::attn_args`]), never from a literal here.

use bloomery_gpu::flash_gqa::{self, FlashGqaKernels, GqaK192Args};
use bloomery_gpu::rope_neox::{K192Args, RopeNeoxKernels};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use cuda_core::CudaStream;
use model::arch::mimo2::names;
use model::arch::mimo2::program::{self as facts, AttnArgs};

use crate::body::{Parts, WHAT, shape};

/// Layer `l`'s sinks as the host holds them, refused by name unless they are
/// one per query head ([`facts::sinks_row`]) and every one finite: a device
/// buffer is not read at launch, so the load is the only place a non-finite
/// sink of the file is caught ([`flash_gqa::check_sinks`]).
pub(crate) fn checked_sinks(l: usize, sinks: &[f32], heads: usize) -> Result<(), GpuError> {
    facts::sinks_row(l, sinks.len(), heads).map_err(|r| shape(r.to_string()))?;
    flash_gqa::check_sinks(sinks).map_err(|e| match e {
        GpuError::Shape { detail, .. } => shape(format!("layer {l}: {detail}")),
        other => other,
    })
}

/// A layer's attention tensors, named once at load.
pub(crate) struct AttnNames {
    pub norm: String,
    pub qkv: String,
    pub o: String,
    /// The per-head sinks, on a window layer.
    pub sinks: Option<String>,
}

impl AttnNames {
    pub(crate) fn of(l: usize, cfg: &AttnArgs) -> AttnNames {
        AttnNames {
            norm: names::attn_norm(l),
            qkv: names::attn_qkv(l),
            o: names::attn_output(l),
            sinks: cfg.sinks.then(|| names::attn_sinks(l)),
        }
    }
}

/// The rope-and-append into `rows` rows a key head. A cache shorter than
/// `args.ctx` is the ring entry; a cache that fills the context is the plane
/// entry (a full layer, a window layer whose ring is the whole plane, and
/// the oracle's plane sequence).
pub(crate) fn enqueue_append(
    rope: &RopeNeoxKernels,
    stream: &CudaStream,
    args: K192Args<'_>,
    rows: usize,
) -> Result<(), GpuError> {
    if rows < args.ctx {
        rope.enqueue_neox_append_k192_ring(stream, args, rows)
    } else {
        rope.enqueue_neox_append_k192(stream, args)
    }
}

/// The K192 flash over the same rows as [`enqueue_append`].
pub(crate) fn enqueue_pass(
    flash: &FlashGqaKernels,
    stream: &CudaStream,
    args: GqaK192Args<'_>,
    n_head: usize,
    rows: usize,
) -> Result<(), GpuError> {
    if rows < args.ctx {
        flash.enqueue_pass_k192_ring(stream, args, n_head, rows)
    } else {
        flash.enqueue_pass_k192(stream, args, n_head)
    }
}

/// Enqueue layer `l`'s attention (module doc): `x` in, `x1` out.
pub(crate) fn attention(
    gpu: &Gpu,
    w: &Weights,
    p: &mut Parts<'_>,
    l: usize,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let (d, c) = (*p.d, p.cfg[l]);
    let a = c.attn;
    let n = &p.names[l].attn;
    let fault = gpu.layer_sink(l)?;
    let s = &mut *p.s;
    let store = &mut p.stores[l];
    let rows = store.rows;
    gpu.elem().enqueue_rms_norm(
        stream,
        &s.x,
        w.f32_buf(WHAT, &n.norm)?,
        d.rms_eps,
        d.embd,
        1,
        &mut s.xn,
    )?;
    w.q8_gemv(gpu, WHAT, &n.qkv, &s.xn, &mut s.qkv)?;
    enqueue_append(
        &p.k.rope,
        stream,
        K192Args {
            qkv: &mut s.qkv,
            q: &mut s.q,
            table: &p.ropes[c.rope].rows.table,
            pos: &s.pos,
            v_scale: a.v_scale,
            n_head: d.heads,
            n_kv: a.kv_heads,
            ctx: p.ctx,
            m: 1,
            fault,
            cache_k: &mut store.k,
            cache_v: &mut store.v,
        },
        rows,
    )?;
    let sinks = n
        .sinks
        .as_deref()
        .map(|name| w.f32_buf(WHAT, name))
        .transpose()?;
    enqueue_pass(
        &p.k.flash,
        stream,
        GqaK192Args {
            q: &s.q,
            kc: &store.k,
            vc: &store.v,
            n_keys: &s.n_keys,
            scale: d.score_scale,
            n_kv: a.kv_heads,
            ctx: p.ctx,
            m: 1,
            window: a.window,
            sinks,
            part_v: &mut s.part_v,
            part_ms: &mut s.part_ms,
            fault,
            y: &mut s.y,
        },
        d.heads,
        rows,
    )?;
    w.q8_gemv(gpu, WHAT, &n.o, &s.y, &mut s.out)?;
    gpu.elem()
        .enqueue_add(stream, &s.x, &s.out, d.embd, &mut s.x1)
}
