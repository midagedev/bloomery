//! The pooled compressed KV — the pooling kernel, its step layout
//! ([`CompGeom`]) and its launchers — is `bloomery_gpu`'s
//! ([`bloomery_gpu::compress`]): DeepSeek V4.1's and ERNIE 5.5's HCA layers
//! run the same entries. This module re-exports them under the path this
//! crate's users read them at.

pub use bloomery_gpu::compress::*;
