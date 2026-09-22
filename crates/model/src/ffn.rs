//! Dense FFN: `ffn_norm-N` in, `ffn_out-N` out.
//!
//! Block 0 in this model, and the shared expert of every MoE block: the math is
//! identical (a SwiGLU gate/up pair, then the down projection), so this module
//! runs the same two entry points for both and never asks which trio it was
//! handed — the caller resolves the tensors, which is the architecture's
//! `plan` module. The MoE module runs its own group dispatches;
//! [`swiglu`] below is the one owner of the combine itself.
//!
//! Gate: `crates/model/tests/ffn.rs` against the oracle.
use crate::ModelError;
use crate::ops::{Tensor2, matmul_q, matmul_q_group};
use crate::profile;
use gguf::{Gguf, TensorInfo};
use std::time::Instant;

/// SwiGLU combine: `silu(gate) * up`, elementwise over the whole
/// `[ne0, n_tokens]` block. The order is the contract — `silu(up) * gate` also
/// produces numbers and only the oracle tells them apart. ggml fuses this with
/// the two matmuls as FUSED_UP_GATE; the gate pins the product, not the fusion.
#[must_use]
pub(crate) fn swiglu(gate: &Tensor2, up: &Tensor2) -> Tensor2 {
    swiglu_timed(gate, up).0
}

/// [`swiglu`] with the combine's own wall time handed back, so a caller that
/// wraps already-hooked work can keep its own site's wall free of the
/// child's nanoseconds (the profiler's no-double-count rule). The record
/// here is the same one [`swiglu`] always made.
pub(crate) fn swiglu_timed(gate: &Tensor2, up: &Tensor2) -> (Tensor2, u64) {
    // Profiler hook (crate::profile): the SiLU·up combine, level-1 typeless —
    // activation work, no weight read. The MoE down group rides it per
    // routed expert and for the shared expert (caller-side for multi-column
    // inputs, inside the dispatch for decode), so one site answers "what
    // does SwiGLU cost" across dense, shexp and routed experts.
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
    let ns = match t_call {
        Some(t_call) => {
            let ns = t_call.elapsed().as_nanos() as u64;
            profile::record_time("swiglu", ns);
            ns
        }
        None => 0,
    };
    (out, ns)
}

/// The FUSED_UP_GATE stage over an already-resolved trio: one group dispatch
/// for the gate/up pair (same `x`, one row split) and the SwiGLU combine.
/// `ffn_norm-N` in, `ffn_up_gate-N` out.
///
/// Single owner of that math — [`dense_ffn_with`] calls it too. Exposed so the
/// gate can pin the intermediate separately from the down projection: a red
/// `ffn_out` with a green intermediate is the down stage.
pub(crate) fn dense_ffn_up_gate_with(
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

/// One dense-shaped FFN over an already-resolved trio: `x` is the block's
/// `ffn_norm-N`, the result is its `ffn_out-N` (a dense block) or its
/// shared-expert output `ffn_shexp-N` (a MoE block).
///
/// Activations stay `[ne0, n_tokens]` throughout — batch first class, M = 1 the
/// special case. The step path, which takes the tensors from the load-time
/// plan.
pub fn dense_ffn_with(
    gguf: &Gguf,
    gate_w: &TensorInfo,
    up_w: &TensorInfo,
    down_w: &TensorInfo,
    x: &Tensor2,
) -> Result<Tensor2, ModelError> {
    let h = dense_ffn_up_gate_with(gguf, gate_w, up_w, x)?;
    matmul_q(gguf, down_w, &h)
}
