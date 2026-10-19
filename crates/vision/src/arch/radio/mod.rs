//! Nemotron 3.5 Bolt's vision encoder as its mmproj file declares it (`clip.projector_type =
//! nemotron_omni`): NVIDIA's RADIO ViT — 32 blocks of LayerNorm with bias, separate query, key and
//! value projections with bias, a GELU MLP, no rotary embedding, a learned 128×128 position table
//! resized to the image's grid, and ten prefix rows (three class tokens and seven registers) that
//! take part in attention and carry no position — and the projector: a 2×2 pixel shuffle, an RMS
//! norm and a ReLU² MLP to the text model's width.
//!
//! The host side of an image is llama.cpp's mtmd rule for this projector: [`crate::size`] plans
//! the pixel size (one dynamic tile from 1,024 to 13,312 patches, the aspect kept), [`media`]
//! resizes the image to it directly (no pad), normalizes it with the file's channel statistics and
//! cuts it in merge order, one frame a patch.

pub mod card;
pub mod hparams;
pub mod media;
pub mod names;
pub mod tensors;

pub use crate::size;
pub use crate::size::{SizeRule, TokenLimits};
pub use hparams::Hparams;
pub use media::{Media, preprocess};

/// `projector_type` of the files this module reads.
pub const PROJECTOR_TYPE: &str = "nemotron_omni";

/// Rows in front of the patches that attend and are dropped before the projector: the class
/// tokens and the registers (`v.class_embd` has this many rows).
pub const PREFIX_ROWS: usize = 10;

/// The side of the learned position table: `v.position_embd.weight` has `POS_SIDE²` rows (the
/// config's `cpe_max_size` 2,048 pixels over the patch).
pub const POS_SIDE: usize = 128;

/// The projector's hidden width: `mm.model.mlp.1.weight` maps the merged rows to it
/// (`projector_hidden_size`).
pub const PROJECTOR_HIDDEN: usize = 20_480;
