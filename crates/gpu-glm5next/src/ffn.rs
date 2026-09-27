//! The feed-forward blocks of one layer at one token: `x` (the fold of the
//! streams by the block's own mix) in, the update into `out`.
//!
//! A dense block ([`dense`]): `ffn_norm` RMS, the gate·up·SwiGLU with the
//! shared limit (`ds41_shexp_gate_up` over the q8_0 planes), the down
//! projection. Three launches, all in the layer's front.
//!
//! A routed block runs around its host leg; every routed expert is the host
//! tier's:
//! - [`front`]: `ffn_norm` RMS into the boundary's handoff region, the
//!   sigmoid router over 288 experts with the selection bias (top 8, the
//!   weights from the unbiased scores, normalized and scaled), the handoff
//!   into the host-mapped image and the go. Three kernels and the go.
//! - [`shadow`]: the shared expert, gate·up·SwiGLU and down, while the host
//!   computes. Two kernels.
//! - [`back`]: the wait, then the routed sum plus the shared expert's
//!   output, `moe + shexp` as ik adds them. One kernel after the wait.
//!
//! The handoff carries the boundary's eight slots through
//! `FfnKernels::enqueue_handoff`, which picks its entry by the slot count and
//! refuses by name a count it has no entry for. The routed sum is the host
//! tier's alone, so neither post entry of the V4.1 piece is called: the sum
//! is one add, and `hc_post` is the sub-layer's own.
//!
//! ik divides GLM's eight weights by their bare sum; the router divides by
//! the sum plus `1e-20` (`route_core::renorm_divisor`), under half an ulp of
//! every sum from 2^-42 up. The two agree bit for bit unless all eight
//! sigmoid scores sit below that: a named difference.

use bloomery_gpu::hybrid::{Boundary, Hybrid};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_deepseek41::chain::ffn::Handoff;
use bloomery_gpu_deepseek41::router::glm5next::N_EXPERT;
use model::arch::glm5next::names;

use crate::body::{Parts, f32t, f32v, gemv, weight};
use crate::host::GlmHost;

/// The dense block's launches.
pub(crate) const DENSE_LAUNCHES: usize = 3;

/// A routed block's launches: the front's three kernels and the go, the
/// shadow's two, the wait and the sum.
pub(crate) const MOE_LAUNCHES: usize = 3 + 1 + 2 + 1 + 1;

/// Enqueue layer `l`'s dense block (module doc).
pub(crate) fn dense(gpu: &Gpu, w: &Weights, p: &mut Parts<'_>, l: usize) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let (d, c) = (*p.d, p.cfg[l]);
    let s = &mut *p.s;
    gpu.elem().enqueue_rms_norm(
        stream,
        &s.x,
        f32v(w, &names::ffn_norm(l))?,
        d.rms_eps,
        d.embd,
        1,
        &mut s.xn,
    )?;
    p.k.experts.enqueue_shexp_gate_up(
        stream,
        weight(w, &names::ffn_gate(l))?,
        weight(w, &names::ffn_up(l))?,
        &s.xn,
        c.limit,
        &mut s.h,
    )?;
    gemv(gpu, w, &names::ffn_down(l), &s.h, &mut s.out)
}

/// Enqueue layer `l`'s routed block up to its go (module doc).
pub(crate) fn front(
    gpu: &Gpu,
    w: &Weights,
    p: &mut Parts<'_>,
    hybrid: &mut Hybrid<GlmHost>,
    l: usize,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let (d, c) = (*p.d, p.cfg[l]);
    let fault = gpu.layer_sink(l)?;
    let s = &mut *p.s;
    gpu.elem().enqueue_rms_norm(
        stream,
        &s.x,
        f32v(w, &names::ffn_norm(l))?,
        d.rms_eps,
        d.embd,
        1,
        hybrid.boundary_mut().normed_mut(),
    )?;
    let bias = if c.bias {
        f32v(w, &names::exp_probs_b(l))?
    } else {
        &s.no_bias
    };
    p.k.router.enqueue_router(
        stream,
        f32t(w, &names::ffn_gate_inp(l))?,
        hybrid.boundary().normed(),
        bias,
        d.scale,
        &mut s.rout,
        fault,
    )?;
    let h = Handoff {
        ids: &s.rout.ids,
        weights: &s.rout.weights,
        map: p.slots.buf(),
        row_off: c.row_off,
        n_expert: N_EXPERT,
    };
    let target = hybrid.boundary_mut().handoff_target_of(0)?;
    p.k.ffn
        .enqueue_handoff(stream, &h, target, fault, &mut s.sel)?;
    hybrid.boundary().enqueue_go_of(stream, l, 0)
}

/// Enqueue layer `l`'s shared expert in its host leg's shadow, after its
/// [`front`].
pub(crate) fn shadow(
    gpu: &Gpu,
    w: &Weights,
    p: &mut Parts<'_>,
    boundary: &Boundary,
    l: usize,
) -> Result<(), GpuError> {
    let c = p.cfg[l];
    let s = &mut *p.s;
    p.k.experts.enqueue_shexp_gate_up(
        gpu.stream(),
        weight(w, &names::ffn_gate_shexp(l))?,
        weight(w, &names::ffn_up_shexp(l))?,
        boundary.normed(),
        c.limit,
        &mut s.h,
    )?;
    gemv(gpu, w, &names::ffn_down_shexp(l), &s.h, &mut s.sh_y)
}

/// Enqueue layer `l`'s wait and its routed sum plus the shared expert's
/// output into `out`, after its [`shadow`].
pub(crate) fn back(gpu: &Gpu, p: &mut Parts<'_>, boundary: &Boundary) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let n = p.d.embd;
    let s = &mut *p.s;
    boundary.enqueue_back_of(stream, 0)?;
    gpu.elem()
        .enqueue_add(stream, boundary.hsum_of(0)?, &s.sh_y, n, &mut s.out)
}
