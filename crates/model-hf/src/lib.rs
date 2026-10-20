//! The CPU model description's two uses of the Hugging Face hub and its checkpoints, kept out of
//! `model` so that crate, and every gate that links it, does not carry the hf crate: [`remote`]
//! reads a fixture's source headers from a repo by HTTP Range (`fixture fetch`, the `fixture` bin of
//! this crate), and [`arch::glm5next`] maps GLM-5.3-Flash's checkpoint onto the glm5next program's names
//! and layouts.

pub mod arch;
pub mod remote;
