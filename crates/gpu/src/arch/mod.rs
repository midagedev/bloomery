//! The architecture-specific half of the GPU engine (docs/arch-split.md).
//!
//! Three layers of host code know a model, and all three live under here: the
//! load-time plan (which tensors, which keys, which derived weights), the
//! capture-time chain (which launches in which order) and the taps a gate
//! reads back. The kernels do not: a kernel is named for a shape and takes
//! that shape as launch arguments, so it stays in the crate root beside the
//! runtime (docs/gpu-design.md decision 6).
//!
//! Module names are the `general.architecture` strings themselves, so one
//! `grep deepseek41` finds the chain, the metadata keys, the tool profile and
//! the gates at once.

pub mod deepseek2;
pub mod qwen3moe;
