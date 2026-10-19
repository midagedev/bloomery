//! MiMo-V2.6-Flash's vision encoder as its mmproj file declares it (`clip.vision.projector_type =
//! mimovl`): a Qwen2.5-VL-shaped ViT of 28 blocks — RMS norm, a fused query/key/value projection with
//! grouped-query attention (32 query heads and 8 key/value heads of 64 values, the heads' width 2048
//! against a model width of 1280), a SwiGLU MLP with biases, no position table, a Conv3d patch embedding
//! stored as two 16×16 kernels — whose attention is full on four blocks and, on the other 24, a band of
//! ±64 patches in the block's patch order with one softmax sink a head, and the merger (a 2×2 merge, a
//! LayerNorm and a two-layer GELU MLP to the text model's width).
//!
//! The file also carries the audio tower's tensors (`a.*`, `mm.a.*`); [`tensors::check`] leaves them to
//! the audio tower.
//!
//! The host side of an image is llama.cpp's mtmd rule for this projector (`dyn_size`, the rule
//! `qwen3vl` runs): [`crate::size`] plans the pixel size inside the file's own budget of 8,192 to
//! 8,388,608 pixels, [`media`] resizes and pads the image onto a black canvas, normalizes it with
//! ImageNet's statistics and cuts it in merge order with each patch repeated over the two temporal
//! frames.

pub mod card;
pub mod hparams;
pub mod media;
pub mod names;
pub mod tensors;

pub use crate::size;
pub use crate::size::{SizeRule, TokenLimits};
pub use hparams::{Hparams, Mode};
pub use media::{Media, preprocess};

use crate::preprocess::ChannelStats;

/// `projector_type` of the files this module reads.
pub const PROJECTOR_TYPE: &str = "mimovl";

/// The width of an attention head. The model width is not `n_head · HEAD_DIM`: the head width is
/// the fused projection's rows over `n_head + 2·n_kv_head`, which [`tensors::check`] holds the file
/// to.
pub const HEAD_DIM: usize = 64;

/// The base of the tower's 2D RoPE: a literal of llama.cpp's graph (`ggml_rope_multi(…, 10000, …)`),
/// not a key of the file.
pub const ROPE_THETA: f32 = 10_000.0;

/// The epsilon of the merger's LayerNorm: a literal of llama.cpp's graph (`mimovl.cpp`), not a key
/// of the file; the tower's RMS norms read [`Hparams::eps`].
pub const POST_LN_EPS: f32 = 1e-6;

/// Temporal frames of the patch embedding: the two kernels (`v.patch_embd.weight` and `.1`) a still
/// image fills with itself.
pub const TEMPORAL_FRAMES: usize = 2;

/// ImageNet's per-channel statistics, the pair the file's `clip.vision.image_mean` and `image_std`
/// hold and the preprocessing normalizes with.
pub const IMAGENET: ChannelStats = ChannelStats {
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
};
