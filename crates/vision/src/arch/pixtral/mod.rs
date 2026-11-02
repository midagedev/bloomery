//! Mistral Large 4's vision encoder as its mmproj file declares it (`clip.projector_type =
//! pixtral`): Mistral's Pixtral ViT — 30 blocks of RMS norm without bias, separate query, key and
//! value projections without bias, a 2D rotary embedding over each head of 128 (the first half of
//! a head turned by the patch's row, the second by its column, adjacent pairs), a SiLU-gated MLP,
//! and one RMS norm before the first block — and the projector: an RMS norm, a 2×2 patch merger
//! back to the ViT width, and a GELU MLP to the text model's width, with the text model's
//! `[IMG_BREAK]` row after every row of merged tokens but the last.
//!
//! The host side of an image is llama.cpp's mtmd rule for this projector (its `dyn_size`
//! preprocessor, the file having no `clip.vision.preproc_image_size`): [`crate::size`] plans the
//! pixel size (8 to 1,024 merged tokens of 28² pixels, the aspect kept), [`media`] resizes the
//! image onto a black canvas of that size (`PAD_CEIL`), normalizes it with the file's channel
//! statistics and cuts it in merge order, one frame a patch.

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
pub const PROJECTOR_TYPE: &str = "pixtral";

/// The rotary base of both halves of a head: mtmd fixes it for this projector
/// (`hparams.rope_theta`), and the file's `clip.rope.freq_base`, where it has one, must agree.
pub const ROPE_THETA: f32 = 10_000.0;

/// The rows an image of `nx`×`ny` merged tokens hands the text model: its tokens row by row, an
/// `[IMG_BREAK]` row after each row but the last (mtmd's `n_patches_y * n_patches_x +
/// n_patches_y - 1`). `ny` is at least one: a plan has a row.
#[must_use]
pub const fn rows_of(nx: usize, ny: usize) -> usize {
    nx * ny + ny - 1
}
