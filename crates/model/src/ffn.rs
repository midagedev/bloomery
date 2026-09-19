//! Dense FFN: `ffn_norm-N` in, `ffn_out-N` out.
//!
//! Block 0 in this model, and the shared expert of every MoE block: the math is
//! identical (a SwiGLU gate/up pair, then the down projection) and only the
//! tensor names differ (`ffn_*` vs `ffn_*_shexp`). Which trio a block carries is
//! decided by presence in the file, so no block number is special-cased and the
//! moe round calls this same function for the shared experts.
//!
//! Gate: `crates/model/tests/ffn.rs` against the oracle. Owned by the ffn round.
use crate::ModelError;
use crate::ops::{Tensor2, matmul_q};
use gguf::{Gguf, TensorInfo};

/// The (gate, up, down) weight trio for `block`'s dense-shaped FFN.
///
/// MoE blocks carry `ffn_*_shexp` tensors, the dense block carries `ffn_*`. If
/// this block has any shared-expert tensor the shexp trio is used, otherwise the
/// dense trio. In this file the two are mutually exclusive per block, so the
/// preference never triggers — it exists so a file with both stays deterministic
/// instead of silently ambiguous. A missing tensor reports its exact name.
fn ffn_weights(
    gguf: &Gguf,
    block: usize,
) -> Result<(&TensorInfo, &TensorInfo, &TensorInfo), ModelError> {
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
    Ok((gate_w, up_w, down_w))
}

/// SwiGLU combine: `silu(gate) * up`, elementwise over the whole
/// `[ne0, n_tokens]` block. The order is the contract — `silu(up) * gate` also
/// produces numbers and only the oracle tells them apart. ggml fuses this with
/// the two matmuls as FUSED_UP_GATE; the gate pins the product, not the fusion.
fn swiglu(gate: &Tensor2, up: &Tensor2) -> Tensor2 {
    assert_eq!(
        (gate.ne0, gate.ne1),
        (up.ne0, up.ne1),
        "swiglu needs gate and up at the same shape"
    );
    let data = gate
        .data
        .iter()
        .zip(&up.data)
        .map(|(&g, &u)| {
            let s = g / (1.0 + (-g).exp());
            s * u
        })
        .collect();
    Tensor2::from_vec(gate.ne0, gate.ne1, data)
}

/// The FUSED_UP_GATE stage over an already-resolved trio: two matmuls and the
/// SwiGLU combine. Single owner of that math — both public entries below call it.
fn up_gate_with(
    gguf: &Gguf,
    gate_w: &TensorInfo,
    up_w: &TensorInfo,
    x: &Tensor2,
) -> Result<Tensor2, ModelError> {
    let gate = matmul_q(gguf, gate_w, x)?;
    let up = matmul_q(gguf, up_w, x)?;
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
/// special case.
pub fn dense_ffn(gguf: &Gguf, block: usize, x: &Tensor2) -> Result<Tensor2, ModelError> {
    let (gate_w, up_w, down_w) = ffn_weights(gguf, block)?;
    let h = up_gate_with(gguf, gate_w, up_w, x)?;
    matmul_q(gguf, down_w, &h)
}
