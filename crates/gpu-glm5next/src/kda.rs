//! The KDA mixer of one layer at one token: `x` (the fold of the streams by
//! the sub-layer's own mix) in, the update into `out`, the layer's conv ring
//! written at the token's position and its state read from the committed
//! lane: the step (row 0) and each row of a plain pass of slots write it
//! back in place, a verify's row 1 — a verify's own, or a slot's in a NextN
//! load's pass — into the next lane, on a load of two (`place::KdaLanes`;
//! the row base, `Parts::base`). The store keeps no history past its
//! lanes: an earlier position comes back only through a checkpoint.
//!
//! The launches are `mixer_kda::enqueue`'s (ik `src/llama-kda.cpp`; the
//! module's doc lists them): this file is the model's part of them. [`Step`]
//! is the q8_0 projections at one token, one gemv a weight; the prompt batch's
//! (the chunked gemvs or the GEMM front) is `prefill`'s. [`dims`], [`vecs`],
//! [`names`] and [`state`] are what both hand the mixer.
//!
//! Eleven launches at a step. ik clamps the state to ±1e6 after every token;
//! the delta step raises its fault site instead when the state stops being
//! finite and never clamps: a named difference, reached only past that
//! magnitude.

use bloomery_gpu::linear::mixer_kda::{
    self, CORE_LAUNCHES, KdaBufs, KdaDecay, KdaDims, KdaIn, KdaProj, KdaRule, KdaState, KdaVecs,
};
use bloomery_gpu::site::{ByName, Out};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use cuda_core::DeviceBuffer;

use crate::body::{Dims, Kernels, Parts, Store, f32v, project, project_joined};
use crate::tensors::{KdaNames, LayerNames, MixerNames, other_kind};

/// The projections the step makes: the joined q·k·v, `f_a`, `g_a`, β, `f_b`,
/// `g_b` and the output.
const PROJ_LAUNCHES: usize = 7;

/// The launches [`kda`] makes.
pub(crate) const LAUNCHES: usize = CORE_LAUNCHES + PROJ_LAUNCHES;

/// The mixer's widths, constants and rule: GLM-5.3-Flash's bounded sigmoid
/// decay and plain `σ(β)`.
pub(crate) fn dims(d: &Dims) -> KdaDims {
    KdaDims {
        embd: d.embd,
        shape: d.kda,
        rms_eps: d.rms_eps,
        rule: KdaRule {
            decay: KdaDecay::Sigmoid { lower_bound: d.lb },
            beta_scale: 1.0,
            write_gate: false,
        },
    }
}

/// A layer's resident f32 vectors.
pub(crate) fn vecs<'w>(w: &'w Weights, n: &KdaNames) -> Result<KdaVecs<'w>, GpuError> {
    Ok(KdaVecs {
        norm: f32v(w, &n.norm)?,
        conv: f32v(w, &n.conv)?,
        dt_bias: f32v(w, &n.dt_bias)?,
        a: f32v(w, &n.a)?,
        gate_norm: f32v(w, &n.gate_norm)?,
    })
}

/// Layer `l`'s KDA tensor names, refused by name for another kind's.
pub(crate) fn names<'n>(
    what: &'static str,
    layers: &'n [LayerNames],
    l: usize,
) -> Result<&'n KdaNames, GpuError> {
    match layers.get(l).map(|n| &n.mixer) {
        Some(MixerNames::Kda(n)) => Ok(n),
        _ => Err(other_kind(what, l)),
    }
}

/// Layer `l`'s recurrent store as the mixer takes it: the committed `lane`
/// word, and whether the rows write their own lane from the row base `row`.
pub(crate) fn state<'s>(
    what: &'static str,
    stores: &'s mut [Store],
    l: usize,
    lane: &'s DeviceBuffer<u32>,
    (each, row): (bool, usize),
) -> Result<KdaState<'s>, GpuError> {
    let Some(Store::Kda { state, stamp, ring }) = stores.get_mut(l) else {
        return Err(GpuError::State {
            what,
            missing: "the layer's KDA store",
        });
    };
    Ok(KdaState {
        lanes: state.lanes().count(),
        state: state.whole_mut(),
        stamp,
        ring,
        lane,
        each,
        row,
    })
}

/// The step's projections of layer `l`, one column, each by the site type its
/// weight is resident in.
struct Step<'a> {
    w: &'a Weights,
    n: &'a KdaNames,
    k: &'a Kernels,
    bn: &'a mut ByName,
}

/// The step runs one row; the mixer asked for `m`.
fn one_row(m: usize) -> Result<(), GpuError> {
    if m == 1 {
        return Ok(());
    }
    Err(GpuError::shape(
        "glm5next kda",
        format!("the step's projections are one column, asked for {m}"),
    ))
}

impl KdaProj for Step<'_> {
    fn input(
        &mut self,
        gpu: &Gpu,
        l: usize,
        m: usize,
        xn: &DeviceBuffer<f32>,
        y: KdaIn<'_>,
    ) -> Result<(), GpuError> {
        one_row(m)?;
        let (w, n, k) = (self.w, self.n, self.k);
        let bn = &mut *self.bn;
        project_joined(gpu, k, bn, (w, &n.qkv, l), xn, 1, y.qkv)?;
        project(gpu, k, bn, (w, &n.f_a, l), xn, 1, Out::whole(y.fa))?;
        project(gpu, k, bn, (w, &n.g_a, l), xn, 1, Out::whole(y.ga))?;
        project(gpu, k, bn, (w, &n.beta, l), xn, 1, Out::whole(y.beta_raw))?;
        project(gpu, k, bn, (w, &n.f_b, l), y.fa, 1, Out::whole(y.f))?;
        project(gpu, k, bn, (w, &n.g_b, l), y.ga, 1, Out::whole(y.z))
    }

    fn output(
        &mut self,
        gpu: &Gpu,
        l: usize,
        m: usize,
        gated: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        one_row(m)?;
        project(
            gpu,
            self.k,
            self.bn,
            (self.w, &self.n.out, l),
            gated,
            1,
            Out::whole(out),
        )
    }
}

/// Enqueue layer `l`'s KDA mixer on `p`'s buffers (module doc).
pub(crate) fn kda(gpu: &Gpu, w: &Weights, p: &mut Parts<'_>, l: usize) -> Result<(), GpuError> {
    const WHAT: &str = "glm5next kda";
    let d = *p.d;
    let n = names(WHAT, p.names, l)?;
    let base = p.base();
    let st = state(WHAT, p.stores, l, p.lane, (base > 0, base))?;
    let s = &mut *p.s;
    mixer_kda::enqueue(
        gpu,
        gpu.stream(),
        &p.k.linear,
        &mut Step {
            w,
            n,
            k: p.k,
            bn: &mut s.by_name,
        },
        (l, 1, &dims(&d)),
        &vecs(w, n)?,
        st,
        KdaBufs {
            x: &s.x,
            pos: &s.pos,
            xn: &mut s.xn,
            qkv: &mut s.qkv,
            fa: &mut s.fa,
            ga: &mut s.ga,
            beta_raw: &mut s.beta_raw,
            write_raw: None,
            f: &mut s.f,
            z: &mut s.z,
            conv: &mut s.conv,
            beta: &mut s.beta,
            decay: &mut s.decay,
            o: &mut s.o,
            gated: &mut s.gated,
            out: &mut s.out,
        },
    )
}
