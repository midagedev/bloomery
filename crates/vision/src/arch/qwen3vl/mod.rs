//! Clef Flash's vision encoder as its mmproj file declares it (`clip.projector_type =
//! qwen3vl_merger`): Qwen3-VL's ViT without DeepStack — 27 blocks, LayerNorm with bias, a GELU-tanh
//! MLP, a learned 48×48 position table, a Conv3d patch embedding stored as two 16×16 kernels —
//! and the merger (a 2×2 merge into a 4608-wide MLP).
//!
//! The host side of an image is llama.cpp's mtmd rule for this projector: [`size`] plans the
//! pixel size, [`media`] pads the image onto a black canvas, normalizes it and cuts it in merge
//! order with each patch repeated over the two temporal frames.

pub mod card;
pub mod hparams;
pub mod media;
pub mod names;
pub mod size;
pub mod tensors;

pub use hparams::Hparams;
pub use media::{Media, preprocess};
pub use size::{SizeRule, TokenLimits};

/// `projector_type` of the files this module reads.
pub const PROJECTOR_TYPE: &str = "qwen3vl_merger";

/// Temporal frames of the patch embedding: the two kernels (`v.patch_embd.weight` and `.1`) a
/// still image fills with itself.
pub const TEMPORAL_FRAMES: usize = 2;
