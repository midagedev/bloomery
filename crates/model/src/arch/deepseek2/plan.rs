//! Tensor resolution: this architecture's names in, `&TensorInfo`s out.
//!
//! The shared FFN and MoE modules hold the math and know shapes only; which
//! tensor a block's gate, up, down, router or expert stack *is* comes from
//! [`names`](super::names), and that question is this architecture's. Every
//! block-taking entry point lives here — `(gguf, block)` in, a resolved plan or
//! the shared kernel's output out — so the shared modules never see a block
//! number or a name.
//!
//! A decode step resolves nothing: [`Derived`](super::derived::Derived) calls
//! the resolvers here once per file and the step path takes the tensors from
//! its plan.

use super::names;
use crate::ModelError;
use crate::ffn;
use crate::moe::{self, Buckets, MoeBlockPlan, MoeTensors};
use crate::ops::Tensor2;
use gguf::{Gguf, TensorInfo};

/// `name`'s tensor, or the error that names what the file is missing.
fn tensor<'a>(gguf: &'a Gguf, name: &str) -> Result<&'a TensorInfo, ModelError> {
    gguf.find(name)
        .ok_or_else(|| ModelError::MissingTensor(name.to_string()))
}

/// The (gate, up, down) weight trio for `block`'s dense-shaped FFN.
///
/// MoE blocks carry `ffn_*_shexp` tensors, the dense block carries `ffn_*`. If
/// this block has any shared-expert tensor the shexp trio is used, otherwise the
/// dense trio. In this file the two are mutually exclusive per block, so the
/// preference never triggers — it exists so a file with both stays deterministic
/// instead of silently ambiguous. A missing tensor reports its exact name.
pub(crate) fn ffn_weights(
    gguf: &Gguf,
    block: usize,
) -> Result<(&TensorInfo, &TensorInfo, &TensorInfo), ModelError> {
    // Profiler hook (crate::profile): resolving the trio costs up to six
    // `find`s, each a linear scan of the tensor table. Level-1 only, typeless:
    // nothing is read, only found.
    let lvl = crate::profile::level();
    let t_call = if lvl > 0 {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let gate = names::ffn_gate(block);
    let up = names::ffn_up(block);
    let down = names::ffn_down(block);
    let gate_shexp = names::ffn_gate_shexp(block);
    let up_shexp = names::ffn_up_shexp(block);
    let down_shexp = names::ffn_down_shexp(block);

    let is_shexp = gguf.find(&gate_shexp).is_some()
        || gguf.find(&up_shexp).is_some()
        || gguf.find(&down_shexp).is_some();
    let (gate, up, down) = if is_shexp {
        (gate_shexp, up_shexp, down_shexp)
    } else {
        (gate, up, down)
    };
    let gate_w = tensor(gguf, &gate)?;
    let up_w = tensor(gguf, &up)?;
    let down_w = tensor(gguf, &down)?;
    if let Some(t_call) = t_call {
        crate::profile::record_time("ffn_weights", t_call.elapsed().as_nanos() as u64);
    }
    Ok((gate_w, up_w, down_w))
}

/// The FUSED_UP_GATE stage on its own: `ffn_norm-N` in, `ffn_up_gate-N` out.
///
/// Exposed so the gate can pin the intermediate separately from the down
/// projection — a red `ffn_out` with a green intermediate is the down stage.
pub fn dense_ffn_up_gate(gguf: &Gguf, block: usize, x: &Tensor2) -> Result<Tensor2, ModelError> {
    let (gate_w, up_w, _) = ffn_weights(gguf, block)?;
    ffn::dense_ffn_up_gate_with(gguf, gate_w, up_w, x)
}

/// One dense-shaped FFN: `x` is the block's `ffn_norm-N`, the result is its
/// `ffn_out-N` (block 0) or its shared-expert output `ffn_shexp-N` (MoE blocks).
///
/// Resolves the trio per call — the direct-call path; a decode step hands the
/// trio from [`Derived`](super::derived::Derived) to
/// [`ffn::dense_ffn_with`] instead.
pub fn dense_ffn(gguf: &Gguf, block: usize, x: &Tensor2) -> Result<Tensor2, ModelError> {
    let (gate_w, up_w, down_w) = ffn_weights(gguf, block)?;
    ffn::dense_ffn_with(gguf, gate_w, up_w, down_w, x)
}

/// `block`'s router weight — its presence in the file is what makes the block
/// routed, so a caller that only asks "is this block MoE" uses `find` directly.
pub(crate) fn router(gguf: &Gguf, block: usize) -> Result<&TensorInfo, ModelError> {
    tensor(gguf, &names::ffn_gate_inp(block))
}

/// One MoE block's load-time plan: resolve the seven tensors, then hand them to
/// the shared builder, which owns every shape check.
pub(crate) fn moe_block_plan(
    gguf: &Gguf,
    block: usize,
    embd: usize,
) -> Result<MoeBlockPlan, ModelError> {
    let tensors = MoeTensors {
        gate_inp: router(gguf, block)?,
        gate_exps: tensor(gguf, &names::ffn_gate_exps(block))?,
        up_exps: tensor(gguf, &names::ffn_up_exps(block))?,
        down_exps: tensor(gguf, &names::ffn_down_exps(block))?,
        shexp_gate: tensor(gguf, &names::ffn_gate_shexp(block))?,
        shexp_up: tensor(gguf, &names::ffn_up_shexp(block))?,
        shexp_down: tensor(gguf, &names::ffn_down_shexp(block))?,
    };
    MoeBlockPlan::build(gguf, block, embd, tensors)
}

/// Route every token of `block` to its `n_used` experts (see
/// `moe::route_with` for what routing computes).
pub fn route(gguf: &Gguf, block: usize, x: &Tensor2) -> Result<Buckets, ModelError> {
    moe::route_with(gguf, router(gguf, block)?, x)
}

/// The MoE FFN of `block`: `ffn_norm-N` in, `ffn_out-N` out.
///
/// Builds the block plan per call — the direct-call path. A decode step hands
/// the plan from [`Derived`](super::derived::Derived) to
/// [`moe_ffn_with`](moe::moe_ffn_with) instead, so it pays this once per model, not once per
/// block per step.
pub fn moe_ffn(gguf: &Gguf, block: usize, x: &Tensor2) -> Result<Tensor2, ModelError> {
    let plan = moe_block_plan(gguf, block, x.ne0)?;
    moe::moe_ffn_with(gguf, &plan, x)
}
