//! DeepSeek-V4.1-Flash's vision encoder as an mmproj file declares it (`clip.projector_type =
//! deepseek41v`, the layout of smalinin's `mmproj-DeepSeek-V4.1-Flash-BF16.gguf`): a 32-block ViT
//! with a linear patch embedding and 2D RoPE, and the aligner (a 3×3 unfold, then two linear layers
//! with an erf GELU between), plus the three learned span delimiters.

pub mod hparams;
pub mod names;
pub mod tensors;

pub use hparams::Hparams;

/// `projector_type` of the files this module reads.
pub const PROJECTOR_TYPE: &str = "deepseek41v";

/// The input id every position of an image span carries: `<｜deepseek_image｜>`, config.json's
/// `image_token_id`. It belongs to the text model's tokenizer, not to the encoder file.
pub const IMAGE_TOKEN_ID: u32 = 129_264;
