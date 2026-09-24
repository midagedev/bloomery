//! The qwen3moe (Qwen3-30B-A3B) oracle: ik's dumps of this model
//! (`tools/ref/models/qwen3moe.sh`) — the 5-token prefill on the CPU and on
//! CUDA, and a decode step on the CPU.

use super::Oracle;
use model::arch::Arch;

/// Step 4 after a four-token prefill run node by node under the dumped
/// schedule, `-c 512`.
pub const STEP4: &str = "ref_qwen3moe_step4_every_node";

/// The decode-step sets (the model profile's `ref_step_variant`), by position.
pub const STEP_SETS: &[&str] = &[STEP4];

pub static ORACLE: Oracle = Oracle {
    arch: Arch::Qwen3moe,
    cuda_set: Some("ref_cuda_qwen3moe"),
    cpu_set: "ref_qwen3moe",
    legacy_cuda_set: None,
    step_sets: STEP_SETS,
    taps: &[],
};
