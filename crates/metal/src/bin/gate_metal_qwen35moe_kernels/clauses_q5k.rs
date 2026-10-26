//! The Q5_K clauses: GGUF's Q5_K decode (`quant.metal`'s `Q5K` owner), the
//! type of most routed downs in Qwen3.6's UD files. Every entry the owner
//! instantiates runs on synthetic rows at the family's shapes against the f64
//! oracle (`clauses_synth` holds the bodies): the five mat-vec modes, the
//! pair, the fold, the head over the whole vocabulary, the embedding of a
//! whole table, and the prompt GEMMs (dense, segmented and grouped) on the
//! gate's kernels and on the tensor-op kernels. Clauses on fixture tensors
//! come with a fixture that holds Q5_K tensors.

use bloomery_metal::kernels::Quant;

use super::{ClauseRun, Harness, clauses_synth};

/// The clauses on the gate's own kernels.
pub fn table<'a>(h: &'a Harness<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    clauses_synth::table(
        h,
        Quant::Q5K,
        [
            "q5k_matvec",
            "q5k_row_tail",
            "q5k_norm",
            "q5k_seg",
            "q5k_headnorm",
            "q5k_pair",
            "q5k_fold",
            "q5k_head",
            "q5k_embed",
            "q5k_gemm",
            "q5k_gemm_seg",
            "q5k_grouped",
        ],
    )
}

/// The clauses on the tensor-op kernels (`clauses_tensor`'s table).
pub fn tensor_table<'a>(
    tensor: &'a Result<Harness<'a>, String>,
) -> Vec<(&'static str, ClauseRun<'a>)> {
    clauses_synth::tensor_table(
        tensor,
        Quant::Q5K,
        [
            "tensor_q5k_gemm",
            "tensor_q5k_gemm_seg",
            "tensor_q5k_grouped",
        ],
    )
}
