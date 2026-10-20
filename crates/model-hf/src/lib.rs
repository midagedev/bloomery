//! What the CPU model description does with the Hugging Face hub and its checkpoints, kept out of
//! `model` so that crate, and every gate that links it, does not carry the hf crate: [`remote`]
//! reads a fixture's source headers from a repo by HTTP Range (`fixture fetch`, the `fixture` bin of
//! this crate), [`arch::glm5next`] maps GLM-5.3-Flash's checkpoint onto the glm5next program's names
//! and layouts, [`arch::qwen35moe`] maps Qwen3.8-Flash-Next's onto qwen4exp's, and the `prepare` bin
//! writes a GGUF split from such a checkpoint folder.

pub mod arch;
pub mod remote;
