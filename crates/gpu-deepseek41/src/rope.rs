//! The tail rope and the latent K/V row's norm-rope-append are `bloomery_gpu`'s
//! ([`bloomery_gpu::rope_tail`]): DeepSeek V4.1's and ERNIE 5.5's HCA layers run
//! the same entries, over the rope table every ggml-rope architecture shares
//! ([`bloomery_gpu::rope_table`]). This module re-exports them under the path
//! this crate's users read them at.

pub(crate) use bloomery_gpu::rope_table::rope_pair_rn;
pub use bloomery_gpu::rope_tail::{
    Direction, KvAppendArgs, RopeKernels, RopeSpec, RopeTable, TailShape, ggml_rope_cache,
};
