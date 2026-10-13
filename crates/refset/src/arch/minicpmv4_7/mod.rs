//! The MiniCPM-V 4.7 35B-A3B image sets: llama.cpp mainline's mtmd `minicpmv4_7` projector ([`vis`], the tower per tile)
//! and the checkpoint's text model on image prompts ([`prompt`], the hidden states and the chat ids), dumped by
//! `tools/ref/clefvis/dump_mtmd.cpp` (`just dump-ref-mcvis`). The architecture is the projector's
//! `clip.projector_type`, as the DeepSeek-V4.1 vision family's is; the text model of the checkpoint is a qwen35moe file
//! with its own vocabulary, and its prompt sets state that architecture.

use crate::family::Family;

pub mod prompt;
pub mod vis;

/// The vision tower's architecture: its mmproj file's `clip.projector_type`.
pub const ARCH: &str = "minicpmv4_7";

/// The architecture's families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[&vis::PREPROC, &vis::TAPS, &prompt::HIDDEN, &prompt::CHATIDS];

#[cfg(test)]
mod prompt_tests;
#[cfg(test)]
mod vis_tests;
