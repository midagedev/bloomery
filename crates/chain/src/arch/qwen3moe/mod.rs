//! The qwen3moe family's chain plan: Qwen3-30B-A3B, Qwen3.6-35B-A3B, Qwen3.5-9B and LFM3.
//!
//! [`plan`] is what one layer mixes with and how its experts run; [`read`] makes it from the file's
//! header. [`ops`] names what a backend enqueues for each part of a layer, and [`walk`] orders them.

pub mod ops;
pub mod plan;
pub mod read;
pub mod router;
pub mod walk;
