//! The decode step's chain: every layer's two sub-layers around their mHC mixes, the head's fold,
//! the head and its MultiMax epilogue. A layer the plan hosts no expert of runs whole here; a
//! layer with a host share runs as [`crate::placed`]'s walk.

use bloomery_gpu::head::Head;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, Window, capturing, copy_row};
use model::arch::ernie55::names::Sub;

use crate::body::{Body, Cx, Run, WHAT, tap_l_out, tap_picks};

/// The step's position: the one before the position the stores stand at, which
/// [`bloomery_gpu::model::ChainBody::refresh`] moved past it. A capture may come before any
/// refresh (the open captures the step before the first token) and reads no position: its
/// launches read the words the refresh before each replay writes. Under one, before any step,
/// this is 0.
pub(crate) fn pos(gpu: &Gpu, body: &Body) -> Result<u32, GpuError> {
    match body.held.checked_sub(1) {
        Some(pos) => Ok(pos),
        None if capturing(gpu.stream())? => Ok(0),
        None => Err(GpuError::State {
            what: WHAT,
            missing: "a refreshed step's position",
        }),
    }
}

/// The whole chain of one step: the layers (through the host tier's step port on a placed
/// load), the head's fold into the head's input and the head with MultiMax.
pub(crate) fn enqueue(
    gpu: &Gpu,
    w: &Weights,
    body: &mut Body,
    head: &mut Head,
) -> Result<(), GpuError> {
    let pos = pos(gpu, body)?;
    match body.placed.take() {
        Some(mut placed) => {
            let walked = crate::placed::walk((gpu, w), body, &mut placed, pos);
            body.placed = Some(placed);
            walked?;
        }
        None => {
            for l in 0..body.sh.n_layer {
                attn(gpu, w, body, l, Run::Step(pos))?;
                ffn_whole(gpu, w, body, l, 1)?;
                close(gpu, w, body, l, 1)?;
            }
        }
    }
    head_of(gpu, w, body, head, 1)
}

/// Layer `l`'s attention sub-layer at `at`: the pre, the mixer of its kind and the post.
pub(crate) fn attn(
    gpu: &Gpu,
    w: &Weights,
    body: &mut Body,
    l: usize,
    at: Run,
) -> Result<(), GpuError> {
    let cx = Cx { gpu, w };
    let Body { sh, parts, y, .. } = body;
    parts.trunk.enqueue_pre(cx, l, Sub::Attn)?;
    parts.mix(cx, (l, sh.latent[l]), at, y)?;
    parts.trunk.enqueue_post(cx, l, Sub::Attn, y)
}

/// Layer `l`'s feed-forward block whole over the walk's `m` rows into the body's output: the
/// dense block on the leading layer, the routed block with every expert on the card on the rest
/// (its picks tapped when armed).
pub(crate) fn ffn_whole(
    gpu: &Gpu,
    w: &Weights,
    body: &mut Body,
    l: usize,
    m: usize,
) -> Result<(), GpuError> {
    let cx = Cx { gpu, w };
    {
        let Body { sh, parts, y, .. } = &mut *body;
        parts.trunk.enqueue_pre(cx, l, Sub::Ffn)?;
        if sh.dense[l] {
            return parts.trunk.enqueue_dense(cx, l, y);
        }
        let x = parts.trunk.normed();
        if m == 1 {
            parts.moe.enqueue_step(cx, l, x, y)?;
        } else {
            parts.moe.enqueue_prompt(cx, l, m, x, y)?;
        }
    }
    tap_picks(gpu, body, l, m)
}

/// The feed-forward sub-layer's post of the body's output into the streams, then layer `l`'s tap
/// of the walk's last of `m` rows.
pub(crate) fn close(
    gpu: &Gpu,
    w: &Weights,
    body: &mut Body,
    l: usize,
    m: usize,
) -> Result<(), GpuError> {
    let cx = Cx { gpu, w };
    {
        let Body { parts, y, .. } = &mut *body;
        parts.trunk.enqueue_post(cx, l, Sub::Ffn, y)?;
    }
    tap_l_out(gpu, body, l, m)
}

/// The head's input from the head's fold of the last of the walk's `m` rows of streams, then the
/// head with MultiMax.
pub(crate) fn head_of(
    gpu: &Gpu,
    w: &Weights,
    body: &mut Body,
    head: &mut Head,
    m: usize,
) -> Result<(), GpuError> {
    let cx = Cx { gpu, w };
    {
        let width = body.sh.streams(1);
        let crate::body::Parts {
            trunk, head: mix, ..
        } = &mut body.parts;
        let last = Window::<f32>::of(trunk.streams(), (m - 1) * width * size_of::<f32>(), width)?;
        mix.enqueue_fold(cx, &last, head.input_mut())?;
    }
    head_epilogue(gpu, w, body, head)
}

/// The head over its input with MultiMax between its projection and its argmax; the raw logits
/// tapped when armed.
pub(crate) fn head_epilogue(
    gpu: &Gpu,
    w: &Weights,
    body: &mut Body,
    head: &mut Head,
) -> Result<(), GpuError> {
    let cx = Cx { gpu, w };
    let Body {
        sh, parts, taps, ..
    } = body;
    let vocab = sh.n_vocab;
    let mix = &mut parts.head;
    head.enqueue_with_epilogue(gpu, w, |logits| {
        if let Some(t) = taps.as_mut() {
            copy_row(gpu.stream(), logits, &mut t.raw, vocab)?;
        }
        mix.enqueue_multimax(cx, logits, vocab)
    })
}
