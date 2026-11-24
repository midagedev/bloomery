//! The attention over window rows ⧺ compressed rows is `bloomery_gpu`'s
//! ([`bloomery_gpu::hca`]): DeepSeek V4.1's and ERNIE 5.5's HCA layers run the
//! same entries. This module re-exports them under the path this crate's users
//! read them at.

pub use bloomery_gpu::hca::*;
