//! MiniCPM-V 4.7's vision side as its mmproj file declares it (`clip.projector_type =
//! minicpmv4_7`): a 27-block SigLIP-style ViT of 14-pixel patches with a 2×2 merger after block 6,
//! a final 2×2 merger into the text model's width, and a table of four separator rows
//! (`v.tok_embd_sep`).
//!
//! The host side of an image is llama.cpp's mtmd rule for this projector: [`plan`] slices the
//! image (the llava-uhd plan of `mtmd_image_preprocessor_minicpmv`), [`group`] resizes and
//! normalizes the overview and the slices and lays the group out as tokens, and
//! [`crate::mrope::Canvas`] positions it.

pub mod card;
pub mod group;
pub mod hparams;
pub mod media;
pub mod names;
pub mod plan;
pub mod tensors;

pub use group::{Group, MarkError, Marks, Sep};
pub use hparams::Hparams;
pub use media::{Media, preprocess};
pub use plan::Plan;

/// `projector_type` of the files this module reads.
pub const PROJECTOR_TYPE: &str = "minicpmv4_7";
