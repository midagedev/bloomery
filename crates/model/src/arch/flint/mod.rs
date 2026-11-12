//! The validated description, placement inputs and routed host adapter this reader serves its
//! architectures with (`flint`, `exaone-moe`): full and sliding attention independently select NoPE
//! or NeoX rotation, every layer has head-local q/k norms, and attention sinks are per file (every
//! layer carries them or none does). Routed blocks carry one ungated shared MLP whose width is
//! explicit, not a count of physical modules; a next-token block is described in
//! `ModelSpec.mtp`, run by no program. The facts no GGUF key states come from the architecture's
//! description row (`crate::arch::rules`). Executable admission belongs to the common
//! program/site registry.

pub mod host;
mod hparams;
pub mod place;
mod roles;
pub mod spec;

#[cfg(test)]
mod exaone_tests;
#[cfg(test)]
mod tests;
