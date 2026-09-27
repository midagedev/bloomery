//! The KDA mixer of one layer at one token: `x` (the fold of the streams by
//! the sub-layer's own mix) in, the update into `out`, the layer's state and
//! conv ring read and written in place. One lane: the store keeps one state
//! and no history of it, so a position is taken back only by a reset.
//!
//! The launches, in order (ik `src/llama-kda.cpp`):
//! 1. `attn_norm` RMS;
//! 2. the joined q·k·v projection;
//! 3. the forget and gate low-rank halves (`ssm_f_a`, `ssm_g_a`) and β's raw
//!    projection, three q8_0 gemvs of the normed row, then `ssm_f_b` and
//!    `ssm_g_b` on the halves' outputs;
//! 4. the conv and prep with one decay per key channel
//!    (`kda_conv_prep`: the conv over the ring by the token's position, SiLU,
//!    the q and k L2 norms, σ(β), `exp(lb · σ(−a · (f + dt)))`);
//! 5. the delta step (`kda_delta`) over the state's one lane;
//! 6. the gated per-head RMS norm with the sigmoid gate;
//! 7. the output projection.
//!
//! Eleven launches. ik clamps the state to ±1e6 after every token; the
//! delta step raises its fault site instead when the state stops being
//! finite and never clamps: a named difference, reached only past that
//! magnitude.

use bloomery_gpu::linear::conv::KdaConvArgs;
use bloomery_gpu::linear::delta::DeltaArgs;
use bloomery_gpu::linear::norm_gate::NormGateArgs;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use model::arch::glm5next::names;

use crate::body::{Parts, Store, f32v, gemv};

/// The launches [`kda`] makes.
pub(crate) const LAUNCHES: usize = 11;

/// Enqueue layer `l`'s KDA mixer on `p`'s buffers (module doc).
pub(crate) fn kda(gpu: &Gpu, w: &Weights, p: &mut Parts<'_>, l: usize) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let d = *p.d;
    let fault = gpu.layer_sink(l)?;
    let s = &mut *p.s;
    let Some(Store::Kda { state, ring }) = p.stores.get_mut(l) else {
        return Err(GpuError::State {
            what: "glm5next kda",
            missing: "the layer's KDA store",
        });
    };
    gpu.elem().enqueue_rms_norm(
        stream,
        &s.x,
        f32v(w, &names::attn_norm(l))?,
        d.rms_eps,
        d.embd,
        1,
        &mut s.xn,
    )?;
    gemv(gpu, w, &names::attn_qkv(l), &s.xn, &mut s.qkv)?;
    gemv(gpu, w, &names::ssm_f_a(l), &s.xn, &mut s.fa)?;
    gemv(gpu, w, &names::ssm_g_a(l), &s.xn, &mut s.ga)?;
    gemv(gpu, w, &names::ssm_beta(l), &s.xn, &mut s.beta_raw)?;
    gemv(gpu, w, &names::ssm_f_b(l), &s.fa, &mut s.f)?;
    gemv(gpu, w, &names::ssm_g_b(l), &s.ga, &mut s.z)?;
    let lin = &p.k.linear;
    lin.conv.enqueue_kda_conv_prep(
        stream,
        KdaConvArgs {
            x: &s.qkv,
            b_raw: &s.beta_raw,
            f: &s.f,
            w: f32v(w, &names::ssm_conv1d_qkv(l))?,
            dt_bias: f32v(w, &names::ssm_dt_bias(l))?,
            ssm_a: f32v(w, &names::ssm_a(l))?,
            pos: &s.pos,
            shape: d.kda,
            lb: d.lb,
            eps: d.rms_eps,
            m: 1,
            fault,
            y: &mut s.conv,
            beta: &mut s.beta,
            decay: &mut s.decay,
            ring,
        },
    )?;
    lin.delta.enqueue_kda_delta(
        stream,
        DeltaArgs {
            qkv: &s.conv,
            beta: &s.beta,
            decay: &s.decay,
            lane: &s.lane,
            lane_at: 0,
            lanes: 1,
            shape: d.kda,
            m: 1,
            fault,
            o: &mut s.o,
            state,
        },
    )?;
    lin.norm_gate.enqueue_norm_gate_sigmoid(
        stream,
        NormGateArgs {
            o: &s.o,
            z: &s.z,
            w: f32v(w, &names::ssm_norm(l))?,
            eps: d.rms_eps,
            n_v: d.kda.n_v,
            m: 1,
            fault,
            y: &mut s.gated,
        },
    )?;
    gemv(gpu, w, &names::attn_output(l), &s.gated, &mut s.out)
}
