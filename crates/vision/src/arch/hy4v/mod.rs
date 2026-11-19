//! Hy4-flash's vision encoder as its mmproj file declares it (`clip.projector_type =
//! hy_v4_flash`): a SigLIP-shaped ViT — 27 blocks of LayerNorm with bias, separate query, key and
//! value projections with bias, a tanh-GELU MLP with bias, no rotary embedding, and a learned
//! 128×128 position table resampled to the image's grid by HunyuanVL's rule (a source scale of
//! `(n + 0.1) / 128`) — and the projector (`rmsnorm_mlp_adapter`): an RMS norm, a 2×2 strided
//! convolution to `4·dim`, an erf GELU, a 1×1 convolution to the text model's width, and the
//! file's learned newline row after every row of merged tokens.
//!
//! The host side of an image is llama.cpp's mtmd rule for this projector (its `dyn_size`
//! preprocessor): [`crate::size`] plans the pixel size (4 to 4,096 merged tokens of 32² pixels,
//! the file's `image_min_pixels` and `image_max_pixels`, the aspect kept), [`media`] resizes the
//! image to it directly with the Lanczos filter (`PAD_NONE`), normalizes it with the file's channel
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
pub const PROJECTOR_TYPE: &str = "hy_v4_flash";

/// The side of the learned position table: `v.position_embd.weight` has `POS_SIDE²` rows (the
/// config's `max_image_size` 2,048 pixels over the patch; the converter drops the table's unused
/// class row).
pub const POS_SIDE: usize = 128;

/// The rows an image of `nx`×`ny` merged tokens hands the text model: each row of tokens followed
/// by the file's newline row (mtmd's `(n_patches_x / merge + 1) · n_patches_y / merge`).
#[must_use]
pub const fn rows_of(nx: usize, ny: usize) -> usize {
    (nx + 1) * ny
}
