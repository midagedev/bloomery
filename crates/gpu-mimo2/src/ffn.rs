//! The feed-forward blocks of one layer at one token: the residual after the
//! attention `x1` in, the layer's output out (`x`, or the head's input after
//! the last layer).
//!
//! The dense block (layer 0, [`dense`]): `ffn_norm` RMS, the gated block
//! ([`bloomery_gpu_deepseek41::block`]: gate·up·SwiGLU at the layer's limit,
//! plain at 0, and the down projection, launched by the file's weight types),
//! and the add of `x1`. Four launches over q8_0 planes, and
//! [`bloomery_gpu_deepseek41::block::BlockPlan::extra_launches`] more over
//! K-quants.
//!
//! A routed block runs around its host leg; the card computes the experts the
//! slot map puts on it ([`bloomery_gpu::card_experts`]), the host tier the rest.
//! The three parts are the common routed block's ([`bloomery_gpu::routed_block`])
//! over MiMo's router instance and names:
//! - [`front`]: `ffn_norm` RMS into the boundary's handoff region, the
//!   sigmoid router over 256 experts with the selection bias (top 8, the
//!   weights from the unbiased scores, normalized and scaled), the handoff
//!   into the host-mapped image — each slot's place in the card's stacks, or
//!   the host mark, from the layer's slot-map row — and the go. Three kernels
//!   and the go.
//! - [`shadow`]: while the host computes, the card's slots, six launches
//!   ([`bloomery_gpu::card_experts`]) ending in the card sum plus `x1`; none on a layer with
//!   no card experts.
//! - [`back`]: the wait, then the add of the host's routed sum and the
//!   shadow's `acc + x1` (`x1` alone on a layer with no card experts). One
//!   kernel after the wait.
//!
//! MiMo facts that fix this file: the routed block has no shared expert, so
//! the shadow is the card experts alone; the routed sum plus the residual is
//! the whole of the block's output; the routed experts' SwiGLU is plain (no
//! clamp); the dense block's width is 16384.

use std::ops::Range;

use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::hybrid::{Boundary, Hybrid};
use bloomery_gpu::kquant::Act;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_deepseek41::block::{Block, BlockIo, BlockKernels, enqueue_block};
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
    let plan = c
        .block
        .ok_or_else(|| shape(format!("layer {l}: a dense block without its plan")))?;
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
    let kernels = BlockKernels {
        experts: &p.k.experts,
        kquant: p.k.block.as_ref(),
    };
    let blk = Block {
        plan,
        names: [gate, up, down],
        ff: c.ffn.ff,
        limit: c.ffn.limit,
        layer: l,
    };
    let io = BlockIo {
        x: &s.xn,
        h: &mut s.h,
        y: &mut s.out,
    };
    enqueue_block(gpu, w, &kernels, &mut s.block, &blk, io)?;
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
    let s = &mut *p.s;
    p.routed.front((gpu, w), &mut s.route, &s.x1, (hybrid, l))
}

/// Enqueue layer `l`'s card experts, when it has any, in its host leg's
/// shadow, after its [`front`]: the card sum plus `x1` into the card's `pre`.
pub(crate) fn shadow(
    gpu: &Gpu,
    w: &Weights,
    p: &mut Parts<'_>,
    boundary: &Boundary,
    l: usize,
) -> Result<(), GpuError> {
    let rule = Act::SwigluClamp {
        limit: p.cfg[l].ffn.limit,
    };
    let s = &*p.s;
    p.routed
        .shadow((gpu, w), p.card, &s.route, (boundary, l), (&s.x1, rule))
}

/// Enqueue layer `l`'s wait and the add of the host's routed sum to the
/// shadow's `acc + x1` (`x1` on a layer with no card experts) into `dest`,
/// after its [`shadow`]: the head's input after the last layer and `x`
/// otherwise.
pub(crate) fn back(
    gpu: &Gpu,
    p: &mut Parts<'_>,
    hybrid: &Hybrid<HostRun>,
    l: usize,
    dest: Option<&mut DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    let s = &mut *p.s;
    let dest = dest.unwrap_or(&mut s.x);
    p.routed.back(gpu, hybrid, p.card, l, (&s.x1, dest))
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
