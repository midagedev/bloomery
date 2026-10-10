//! vision — the host side of an image input: from file bytes to the ViT's patch tensor and the
//! token span the text model sees, plus the tensor names and hyperparameters of the encoder file.
//!
//! No device code and no encoder arithmetic live here. What does:
//!
//! * [`image`] — decode a PNG or a JPEG into 8-bit RGB, the reference's `Image.convert("RGB")`.
//! * [`resample`] — Pillow's bicubic resize and `ImageOps.pad`, ported to the integer: the
//!   reference resizes every image whose size is not already its grid through them; and the same
//!   resize under llama.cpp's `PAD_CEIL` pad. Both pads take the projector's fill as an argument.
//! * [`mrope`] — the M-RoPE positions of a sequence that holds images ([`MropeSeq`]) and the rows of
//!   a rope table they select: what a text model that reads `table[row]` needs to place an image.
//! * [`preprocess`] — the plan of one image ([`GridPlan`]), and the normalize to bf16 and the cut
//!   into patches that every projector type shares: row-major in V4.1's order, or in merge groups
//!   with the patch repeated over temporal frames.
//! * [`media`] — what a model tells the server about its image input ([`MediaModel`]): data plus
//!   one prepare.
//! * [`arch`] — per projector type, the names and hyperparameters of the encoder file (the mmproj
//!   GGUF), each read once and refused by name when this crate does not run it, and what the
//!   projector's own image rule needs: V4.1's resize plan and span layout (`deepseek41v::grid`,
//!   `deepseek41v::span`), its reference `load_image` and its bytes on its card; Clef's size rule
//!   and pad colour (`qwen3vl::size`, `qwen3vl::media`).
//!
//! The reference of V4.1 is deepseek-ai/DeepSeek-V4.1-Flash `inference/image_processor.py` and
//! `inference/vision.py`, and Pillow's `libImaging/Resample.c` and `ImageOps.py` of the version
//! the oracle ran; `tools/ref/vision/` dumps what they produce and the gates compare against it.
//! The reference of `qwen3vl_merger` (Clef Flash) is llama.cpp's mtmd preprocessing.

pub mod arch;
pub mod image;
pub mod media;
pub mod mrope;
pub mod preprocess;
pub mod resample;

pub use image::{FileKind, Rgb8};
pub use media::{MediaModel, Prepared};
pub use mrope::{MropeError, MropeSeq};
pub use preprocess::{GridPlan, PatchLayout, Patches, patchify};

/// Everything this crate refuses, each naming what it refused.
#[derive(Debug, thiserror::Error)]
pub enum VisionError {
    /// The image bytes did not decode.
    #[error("image decode: {0}")]
    Decode(String),
    /// A decoded image this crate does not convert to RGB.
    #[error("image format: {0}")]
    Format(String),
    /// An image or a plan with no pixels in one axis.
    #[error("image size {width}x{height}: {detail}")]
    Size {
        width: usize,
        height: usize,
        detail: String,
    },
    /// A token limit, patch or merge side, or a frame count, that this crate's size rule or patch
    /// layout does not take.
    #[error("image size limits: {detail}")]
    Limits { detail: String },
    /// A metadata key of the encoder file that is absent, of another type, or holds a value this
    /// crate does not run.
    #[error("metadata {key}: {detail}")]
    Metadata { key: String, detail: String },
    /// A tensor of the encoder file that is unknown, duplicated, missing or of another shape or type.
    #[error("tensor {name}: {detail}")]
    Tensor { name: String, detail: String },
    /// The encoder file itself did not open.
    #[error(transparent)]
    Load(#[from] gguf::LoadError),
}
