//! The qwen3moe (Qwen3-30B-A3B) oracle: ik's dumps of this model
//! (`tools/ref/models/qwen3moe.sh`) — the 5-token prefill on the CPU and on
//! CUDA, and decode steps on the CPU at three depths.

use super::Oracle;
use model::arch::Arch;

/// Step 4 after a four-token prefill run node by node under the dumped
/// schedule, `-c 512`.
pub const STEP4: &str = "ref_qwen3moe_step4_every_node";

/// Step 1,024 after a fused prefill of 1,024 ids of prose, `-c 2048`.
pub const D1K: &str = "ref_qwen3moe_d1k";

/// Step 4,096 after a fused prefill of 4,096 ids of prose, `-c 4608`.
pub const D4K: &str = "ref_qwen3moe_d4k";

/// The decode-step sets (the model profile's `ref_step_variant`), by position.
pub const STEP_SETS: &[&str] = &[STEP4, D1K, D4K];

pub static ORACLE: Oracle = Oracle {
    arch: Arch::Qwen3moe,
    cuda_set: Some("ref_cuda_qwen3moe"),
    cpu_set: "ref_qwen3moe",
    legacy_cuda_set: None,
    step_sets: STEP_SETS,
    taps: &[],
};
