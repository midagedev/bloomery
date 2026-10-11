//! GLM-5.3-Flash's vision encoder as its mmproj file declares it (`clip.projector_type = glm5v`):
//! llama.cpp's `glm4v` tower with a clamped SwiGLU — 24 blocks of RMS norm, a per-head RMS norm on
//! query and key, a SwiGLU MLP with bias, no position table, a Conv3d patch embedding stored as
//! two 14×14 kernels — and the merger (a 2×2 convolution, a linear, LayerNorm, GELU and a SwiGLU
//! MLP to the text model's width).
//!
//! The host side of an image is llama.cpp's mtmd rule for this projector: [`size`] plans the
//! canvas inside a pixel budget, [`media`] resizes the image onto the canvas's top-left corner over
//! black, normalizes it with CLIP's statistics and cuts it in merge order with each patch repeated
//! over the two temporal frames, and [`Media`] is the server's description of the image input.
//! The pixel budget is the file's own (16 to 8000 merged tokens, [`Hparams::default_limits`])
//! unless the token limits replace it ([`Hparams::size_rule_at`]).

pub mod card;
pub mod hparams;
pub mod media;
pub mod names;
pub mod rules;
pub mod size;
pub mod tensors;

pub use hparams::Hparams;
pub use media::{Media, preprocess};
pub use size::{Canvas, SizeRule};

use crate::preprocess::ChannelStats;

/// `projector_type` of the files this module reads.
pub const PROJECTOR_TYPE: &str = "glm5v";

/// The base of the tower's 2D RoPE: a constant of llama.cpp's loader for this projector
/// (`hparams.rope_theta`), not a key of the file.
pub const ROPE_THETA: f32 = 10_000.0;

/// The epsilon of the merger's LayerNorm: a literal of llama.cpp's graph (`glm4v.cpp`), not a key
/// of the file; the tower's RMS norms read [`Hparams::eps`].
pub const MERGER_NORM_EPS: f32 = 1e-5;

/// Temporal frames of the patch embedding: the two kernels (`v.patch_embd.weight` and `.1`) a
/// still image fills with itself.
pub const TEMPORAL_FRAMES: usize = 2;

/// The hidden width of the merger's SwiGLU MLP. No header key carries it: it is the shape of
/// `mm.gate.weight`, `mm.up.weight` and `mm.down.weight`, which [`tensors::check`] holds the file
/// to.
pub const MERGER_FF: usize = 10_240;

/// CLIP's per-channel statistics, the pair the file's `clip.vision.image_mean` and `image_std`
/// hold and the preprocessing normalizes with.
pub const CLIP: ChannelStats = ChannelStats {
    mean: [
        0.481_454_670_429_229_74_f64 as f32,
        0.457_827_508_449_554_44_f64 as f32,
        0.408_210_724_592_208_86_f64 as f32,
    ],
    std: [
        0.268_629_550_933_837_9_f64 as f32,
        0.261_302_590_370_178_2_f64 as f32,
        0.275_777_101_516_723_63_f64 as f32,
    ],
};
