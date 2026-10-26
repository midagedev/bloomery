// The decode owners of the 32-value block types (Q5_0, Q5_1): the block
// layouts and the value rule each mat-vec, embedding and GEMM instantiates.
// No owner is written yet; the file is in build.rs's list so the owners that
// land here edit this file alone.
//
// This file comes after quant.metal (the owner shape it follows) and before
// gemm_shared.metal, so the tensor-op GEMMs' source holds it too.

#include <metal_stdlib>
