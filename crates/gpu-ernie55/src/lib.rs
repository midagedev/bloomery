//! bloomery-gpu-ernie55: the ERNIE 5.5 Flash Preview program over `bloomery-gpu`'s kernels, host code only. The model
//! description is `model::arch::ernie55`.

pub mod attn_hca;
pub mod attn_mla;
pub mod head;
pub mod moe;
pub mod trunk;
