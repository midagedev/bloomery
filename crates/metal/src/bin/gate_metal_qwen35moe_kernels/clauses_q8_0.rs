//! The Q8_0 clauses: GGUF's Q8_0 decode (`quant.metal`'s `Q8_0` owner), the
//! type of the attention, the shared expert, the embedding and the head in
//! Qwen3.6's UD files. Every entry the owner instantiates runs on synthetic
//! rows at the family's shapes against the f64 oracle (`clauses_synth` holds
//! the bodies): the five mat-vec modes, the pair, the fold, the head over the
//! whole vocabulary, the embedding of a whole table, and the prompt GEMMs
//! (dense, segmented and grouped) on the gate's kernels and on the tensor-op
//! kernels. Clauses on fixture tensors come with a fixture that holds Q8_0
//! tensors.

use bloomery_metal::kernels::Quant;

use super::{ClauseRun, Harness, clauses_synth};

/// The clauses on the gate's own kernels.
pub fn table<'a>(h: &'a Harness<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    clauses_synth::table(
        h,
        Quant::Q8_0,
        [
            "q8_0_matvec",
            "q8_0_row_tail",
            "q8_0_norm",
            "q8_0_seg",
            "q8_0_headnorm",
            "q8_0_pair",
            "q8_0_fold",
            "q8_0_head",
            "q8_0_embed",
            "q8_0_gemm",
            "q8_0_gemm_seg",
            "q8_0_grouped",
        ],
    )
}

/// The clauses on the tensor-op kernels (`clauses_tensor`'s table).
pub fn tensor_table<'a>(
    tensor: &'a Result<Harness<'a>, String>,
) -> Vec<(&'static str, ClauseRun<'a>)> {
    clauses_synth::tensor_table(
        tensor,
        Quant::Q8_0,
        [
            "tensor_q8_0_gemm",
            "tensor_q8_0_gemm_seg",
            "tensor_q8_0_grouped",
        ],
    )
}
