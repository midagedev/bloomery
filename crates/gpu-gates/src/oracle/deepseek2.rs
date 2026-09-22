//! The deepseek2 (DeepSeek-V2-Lite) oracle: ik's dumps of this model and the
//! tap names several gates share.

use super::Oracle;
use model::arch::Arch;

/// Block 0's output residual (the dense block's closing ADD).
pub const L_OUT_0: &str = "l_out-0";

/// Block 1's output residual (the first routed block's closing ADD).
pub const L_OUT_1: &str = "l_out-1";

/// The last block's output residual — the output head's input.
pub const L_OUT_26: &str = "l_out-26";

pub static ORACLE: Oracle = Oracle {
    arch: Arch::Deepseek2,
    cuda_set: "ref_cuda_v2",
    cpu_set: "ref",
    legacy_cuda_set: "ref_cuda",
    taps: &[L_OUT_0, L_OUT_1, L_OUT_26],
};
