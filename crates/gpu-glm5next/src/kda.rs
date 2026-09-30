//! The KDA mixer of one layer at one token: `x` (the fold of the streams by
//! the sub-layer's own mix) in, the update into `out`, the layer's conv ring
//! written at the token's position and its state read from the committed
//! lane: the step (row 0) writes it back in place, a verify's row 1 into the
//! next lane, on a load of two (`place::KdaLanes`). The store keeps no
//! history past its lanes: an earlier position comes back only through a
//! checkpoint.
//!
//! The launches, in order (ik `src/llama-kda.cpp`):
//! 1. `attn_norm` RMS;
//! 2. the joined q·k·v projection;
//! 3. the forget and gate low-rank halves (`ssm_f_a`, `ssm_g_a`) and β's raw
//!    projection, three q8_0 gemvs of the normed row, then `ssm_f_b` and
//!    `ssm_g_b` on the halves' outputs;
//! 4. the conv and prep with one decay per key channel
//!    (`kda_conv_prep`: the conv over the ring by the token's position, SiLU,
//!    the q and k L2 norms, σ(β), `exp(lb · σ(−ssm_a · (f + dt)))`);
//! 5. the delta step (`kda_delta_lanes`) over the state's committed lane, its
//!    stamp checked against the position;
//! 6. the gated per-head RMS norm with the sigmoid gate;
//! 7. the output projection.
//!
//! Eleven launches. ik clamps the state to ±1e6 after every token; the
//! delta step raises its fault site instead when the state stops being
//! finite and never clamps: a named difference, reached only past that
//! magnitude.

use bloomery_gpu::linear::conv::KdaConvArgs;
use bloomery_gpu::linear::delta::{DeltaArgs, DeltaLanesArgs, KdaLanesArgs};
use bloomery_gpu::linear::norm_gate::NormGateArgs;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};

use crate::body::{Parts, Store, f32v, gemv};
use crate::tensors::{MixerNames, other_kind};

/// The launches [`kda`] makes.
pub(crate) const LAUNCHES: usize = 11;

/// Enqueue layer `l`'s KDA mixer on `p`'s buffers (module doc).
pub(crate) fn kda(gpu: &Gpu, w: &Weights, p: &mut Parts<'_>, l: usize) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let d = *p.d;
    let fault = gpu.layer_sink(l)?;
    let Some(MixerNames::Kda(n)) = p.names.get(l).map(|n| &n.mixer) else {
        return Err(other_kind("glm5next kda", l));
    };
    let s = &mut *p.s;
    let Some(Store::Kda { state, stamp, ring }) = p.stores.get_mut(l) else {
        return Err(GpuError::State {
            what: "glm5next kda",
            missing: "the layer's KDA store",
        });
    };
    gpu.elem().enqueue_rms_norm(
        stream,
        &s.x,
        f32v(w, &n.norm)?,
        d.rms_eps,
        d.embd,
        1,
        &mut s.xn,
    )?;
    gemv(gpu, w, &n.qkv, &s.xn, &mut s.qkv)?;
    gemv(gpu, w, &n.f_a, &s.xn, &mut s.fa)?;
    gemv(gpu, w, &n.g_a, &s.xn, &mut s.ga)?;
    gemv(gpu, w, &n.beta, &s.xn, &mut s.beta_raw)?;
    gemv(gpu, w, &n.f_b, &s.fa, &mut s.f)?;
    gemv(gpu, w, &n.g_b, &s.ga, &mut s.z)?;
    let lin = &p.k.linear;
    lin.conv.enqueue_kda_conv_prep(
        stream,
        KdaConvArgs {
            x: &s.qkv,
            b_raw: &s.beta_raw,
            f: &s.f,
            w: f32v(w, &n.conv)?,
            dt_bias: f32v(w, &n.dt_bias)?,
            ssm_a: f32v(w, &n.a)?,
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
    lin.delta.enqueue_kda_delta_lanes(
        stream,
        KdaLanesArgs {
            lanes: DeltaLanesArgs {
                delta: DeltaArgs {
                    qkv: &s.conv,
                    beta: &s.beta,
                    decay: &s.decay,
                    lane: p.lane,
                    lane_at: 0,
                    lanes: state.lanes().count(),
                    shape: d.kda,
                    m: 1,
                    fault,
                    o: &mut s.o,
                    state: state.whole_mut(),
                },
                each: p.row > 0,
                pos: &s.pos,
                stamp,
            },
            row: p.row,
        },
    )?;
    lin.norm_gate.enqueue_norm_gate_sigmoid(
        stream,
        NormGateArgs {
            o: &s.o,
            z: &s.z,
            w: f32v(w, &n.gate_norm)?,
            eps: d.rms_eps,
            n_v: d.kda.n_v,
            m: 1,
            fault,
            y: &mut s.gated,
        },
    )?;
    gemv(gpu, w, &n.out, &s.gated, &mut s.out)
}
