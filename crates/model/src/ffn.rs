//! Dense FFN: `ffn_norm-N` in, `ffn_out-N` out.
//!
//! Block 0 in this model, and the shared expert of every MoE block: the math is
//! identical (a SwiGLU gate/up pair, then the down projection) and only the
//! tensor names differ (`ffn_*` vs `ffn_*_shexp`). Which trio a block carries is
//! decided by presence in the file, so no block number is special-cased. The
//! MoE module runs its own copy of the three ops (`moe.rs::shexp_ffn`) so its
//! tensor finds land under its own profiler sites; [`swiglu`] below is the one
//! owner of the combine itself.
//!
//! Gate: `crates/model/tests/ffn.rs` against the oracle.
use crate::ModelError;
use crate::ops::{Tensor2, matmul_q, matmul_q_group};
use crate::profile;
use gguf::{Gguf, TensorInfo};
use std::time::Instant;

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
    let lvl = profile::level();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    let gate = format!("blk.{block}.ffn_gate.weight");
    let up = format!("blk.{block}.ffn_up.weight");
    let down = format!("blk.{block}.ffn_down.weight");
    let gate_shexp = format!("blk.{block}.ffn_gate_shexp.weight");
    let up_shexp = format!("blk.{block}.ffn_up_shexp.weight");
    let down_shexp = format!("blk.{block}.ffn_down_shexp.weight");

    let is_shexp = gguf.find(&gate_shexp).is_some()
        || gguf.find(&up_shexp).is_some()
        || gguf.find(&down_shexp).is_some();
    let (gate, up, down) = if is_shexp {
        (gate_shexp, up_shexp, down_shexp)
    } else {
        (gate, up, down)
    };
    let gate_w = gguf
        .find(&gate)
        .ok_or_else(|| ModelError::MissingTensor(gate.clone()))?;
    let up_w = gguf
        .find(&up)
        .ok_or_else(|| ModelError::MissingTensor(up.clone()))?;
    let down_w = gguf
        .find(&down)
        .ok_or_else(|| ModelError::MissingTensor(down.clone()))?;
    if let Some(t_call) = t_call {
        profile::record_time("ffn_weights", t_call.elapsed().as_nanos() as u64);
    }
    Ok((gate_w, up_w, down_w))
}

/// SwiGLU combine: `silu(gate) * up`, elementwise over the whole
/// `[ne0, n_tokens]` block. The order is the contract — `silu(up) * gate` also
/// produces numbers and only the oracle tells them apart. ggml fuses this with
/// the two matmuls as FUSED_UP_GATE; the gate pins the product, not the fusion.
pub(crate) fn swiglu(gate: &Tensor2, up: &Tensor2) -> Tensor2 {
    // Profiler hook (crate::profile): the SiLU·up combine, level-1 typeless —
    // activation work, no weight read. `moe.rs` calls this per routed expert
    // and for the shared expert, so one site answers "what does SwiGLU cost"
    // across dense, shexp and routed experts.
    let lvl = profile::level();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    assert_eq!(
        (gate.ne0, gate.ne1),
        (up.ne0, up.ne1),
        "swiglu needs gate and up at the same shape"
    );
    // Written into a block, not `map().collect()`: the collect form is
    // slower per element and this runs once per routed expert per step.
    let mut out = Tensor2::scratch(gate.ne0, gate.ne1);
    qdot::swiglu(&gate.data, &up.data, &mut out.data);
    if let Some(t_call) = t_call {
        profile::record_time("swiglu", t_call.elapsed().as_nanos() as u64);
    }
    out
}

/// The FUSED_UP_GATE stage over an already-resolved trio: one group dispatch
/// for the gate/up pair (same `x`, one row split) and the SwiGLU combine.
/// Single owner of that math — both public entries below call it.
fn up_gate_with(
    gguf: &Gguf,
    gate_w: &TensorInfo,
    up_w: &TensorInfo,
    x: &Tensor2,
) -> Result<Tensor2, ModelError> {
    let mut gu = matmul_q_group(gguf, &[gate_w, up_w], &[x, x])?;
    let up = gu.pop().expect("one output per pair");
    let gate = gu.pop().expect("one output per pair");
    Ok(swiglu(&gate, &up))
}

/// The FUSED_UP_GATE stage on its own: `ffn_norm-N` in, `ffn_up_gate-N` out.
///
/// Exposed so the gate can pin the intermediate separately from the down
/// projection — a red `ffn_out` with a green intermediate is the down stage.
pub fn dense_ffn_up_gate(gguf: &Gguf, block: usize, x: &Tensor2) -> Result<Tensor2, ModelError> {
    let (gate_w, up_w, _) = ffn_weights(gguf, block)?;
    up_gate_with(gguf, gate_w, up_w, x)
}

/// One dense-shaped FFN: `x` is the block's `ffn_norm-N`, the result is its
/// `ffn_out-N` (block 0) or its shared-expert output `ffn_shexp-N` (MoE blocks).
///
/// Activations stay `[ne0, n_tokens]` throughout — batch first class, M = 1 the
/// special case. Resolves the trio per call — the direct-call path; a decode
/// step hands the trio from [`Derived`](crate::derived::Derived) to
/// [`dense_ffn_with`] instead.
pub fn dense_ffn(gguf: &Gguf, block: usize, x: &Tensor2) -> Result<Tensor2, ModelError> {
    let (gate_w, up_w, down_w) = ffn_weights(gguf, block)?;
    dense_ffn_with(gguf, gate_w, up_w, down_w, x)
}

/// One dense-shaped FFN over an already-resolved trio — the step path, which
/// takes the tensors from [`Derived`](crate::derived::Derived).
pub fn dense_ffn_with(
    gguf: &Gguf,
    gate_w: &TensorInfo,
    up_w: &TensorInfo,
    down_w: &TensorInfo,
    x: &Tensor2,
) -> Result<Tensor2, ModelError> {
    let h = up_gate_with(gguf, gate_w, up_w, x)?;
    matmul_q(gguf, down_w, &h)
}
