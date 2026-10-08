//! The feed-forward blocks of one layer at one token: the residual after the
//! attention `x1` in, the layer's output out (`x`, or the head's input after
//! the last layer).
//!
//! The dense block (layer 0, [`dense`]): `ffn_norm` RMS, the gate·up·SwiGLU
//! (`ds41_shexp_gate_up` over the q8_0 planes, plain at the layer's limit 0),
//! the down projection and the add of `x1`. Four launches.
//!
//! A routed block runs around its host leg; every routed expert is on the
//! host tier (the plan keeps none on a card):
//! - [`front`]: `ffn_norm` RMS into the boundary's handoff region, the
//!   sigmoid router over 256 experts with the selection bias (top 8, the
//!   weights from the unbiased scores, normalized and scaled), the handoff
//!   into the host-mapped image and the go. Three kernels and the go.
//! - [`back`]: the wait, then the add of `x1` and the host's routed sum. One
//!   kernel after the wait.
//!
//! MiMo facts that fix this file: the routed block has no shared expert, so
//! the card has nothing to run under the host leg and the walk has no shadow;
//! the routed sum is the whole of the block's output; the routed experts'
//! SwiGLU is plain (no clamp); the dense block's width is 16384.

use std::ops::Range;

use bloomery_gpu::host::handoff::Handoff;
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::hybrid::Hybrid;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_deepseek41::router::mimo2::N_EXPERT;
use cuda_core::DeviceBuffer;
use model::ModelError;
use model::arch::mimo2::host;
use model::arch::mimo2::hparams::Hparams;
use model::arch::mimo2::names;
use model::arch::mimo2::program::BlockArgs;
use model::moe::HostLayer;
use model::r8file::R8Source;
use runtime::layer::FfnKind;

use crate::body::{Parts, WHAT, shape};

/// The block's tensors, named once at load.
pub(crate) enum FfnNames {
    Dense {
        norm: String,
        gate: String,
        up: String,
        down: String,
    },
    Moe {
        norm: String,
        router: String,
        /// The selection bias, when the file has it.
        bias: Option<String>,
    },
}

impl FfnNames {
    pub(crate) fn of(l: usize, kind: FfnKind, cfg: &BlockArgs) -> FfnNames {
        match kind {
            FfnKind::Dense => FfnNames::Dense {
                norm: names::ffn_norm(l),
                gate: names::ffn_gate(l),
                up: names::ffn_up(l),
                down: names::ffn_down(l),
            },
            FfnKind::Moe => FfnNames::Moe {
                norm: names::ffn_norm(l),
                router: names::ffn_gate_inp(l),
                bias: cfg.bias.then(|| names::exp_probs_b(l)),
            },
        }
    }
}

/// `l`'s names as the other kind's, by name.
fn other_kind(what: &str, l: usize) -> GpuError {
    shape(format!(
        "layer {l}: the names of another block kind than the {what}"
    ))
}

/// Enqueue layer `l`'s dense block (module doc) into `dest`, the head's input
/// after the last layer and `x` otherwise.
pub(crate) fn dense(
    gpu: &Gpu,
    w: &Weights,
    p: &mut Parts<'_>,
    l: usize,
    dest: Option<&mut DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let (d, c) = (*p.d, p.cfg[l]);
    let FfnNames::Dense {
        norm,
        gate,
        up,
        down,
    } = &p.names[l].ffn
    else {
        return Err(other_kind("dense block", l));
    };
    let s = &mut *p.s;
    gpu.elem().enqueue_rms_norm(
        stream,
        &s.x1,
        w.f32_buf(WHAT, norm)?,
        d.rms_eps,
        d.embd,
        1,
        &mut s.xn,
    )?;
    p.k.experts.enqueue_shexp_gate_up(
        stream,
        w.resident(WHAT, gate)?,
        w.resident(WHAT, up)?,
        &s.xn,
        c.ffn.limit,
        &mut s.h,
    )?;
    w.q8_gemv(gpu, WHAT, down, &s.h, &mut s.out)?;
    let dest = dest.unwrap_or(&mut s.x);
    gpu.elem().enqueue_add(stream, &s.x1, &s.out, d.embd, dest)
}

/// Enqueue layer `l`'s routed block up to its go (module doc).
pub(crate) fn front(
    gpu: &Gpu,
    w: &Weights,
    p: &mut Parts<'_>,
    hybrid: &mut Hybrid<HostRun>,
    l: usize,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let (d, c) = (*p.d, p.cfg[l]);
    let fault = gpu.layer_sink(l)?;
    let FfnNames::Moe { norm, router, bias } = &p.names[l].ffn else {
        return Err(other_kind("routed block", l));
    };
    let s = &mut *p.s;
    gpu.elem().enqueue_rms_norm(
        stream,
        &s.x1,
        w.f32_buf(WHAT, norm)?,
        d.rms_eps,
        d.embd,
        1,
        hybrid.boundary_mut().normed_mut(),
    )?;
    let bias = match bias {
        Some(name) => w.f32_buf(WHAT, name)?,
        None => &s.no_bias,
    };
    p.k.router.enqueue_router(
        stream,
        w.f32_tensor(WHAT, router)?,
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
    p.k.handoff
        .enqueue_handoff(stream, &h, target, fault, &mut s.sel)?;
    hybrid.boundary().enqueue_go_of(stream, l, 0)
}

/// Enqueue layer `l`'s wait and the add of the host's routed sum to `x1`
/// into `dest`, after its [`front`]: the head's input after the last layer
/// and `x` otherwise.
pub(crate) fn back(
    gpu: &Gpu,
    p: &mut Parts<'_>,
    hybrid: &Hybrid<HostRun>,
    dest: Option<&mut DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let boundary = hybrid.boundary();
    let s = &mut *p.s;
    boundary.enqueue_back_of(stream, 0)?;
    let dest = dest.unwrap_or(&mut s.x);
    gpu.elem()
        .enqueue_add(stream, boundary.hsum_of(0)?, &s.x1, p.d.embd, dest)
}

/// The host views of the routed layers `run`, each from [`host::layer`]; a
/// layer of the run past the trunk is refused by name. Load-time only: the
/// stacks are checked here.
pub(crate) fn routed_layers(
    src: R8Source<'_>,
    hp: &Hparams,
    run: Range<usize>,
) -> Result<Vec<HostLayer>, ModelError> {
    run.map(|l| host::layer(src, hp, l)).collect()
}
