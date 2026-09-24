//! vision — the host side of an image input: from file bytes to the ViT's patch tensor and the
//! token span the text model sees, plus the tensor names and hyperparameters of the encoder file.
//!
//! No device code and no encoder arithmetic live here. What does:
//!
//! * [`image`] — decode a PNG into 8-bit RGB, the reference's `Image.convert("RGB")`.
//! * [`grid`] — the resize plan (`plan_image_grid`, `safe_resize` of the reference
//!   `image_processor.py`), integer-identical to it.
//! * [`resample`] — Pillow's bicubic resize and `ImageOps.pad`, ported to the integer: the
//!   reference resizes every image whose size is not already its grid through them.
//! * [`preprocess`] — normalize the padded image to bf16 and cut it into patches in the reference's
//!   order.
//! * [`span`] — the token layout of one image in the prompt.
//! * [`arch`] — per projector type, the names and hyperparameters of the encoder file (the mmproj
//!   GGUF), each read once and refused by name when this crate does not run it.
//!
//! The reference is deepseek-ai/DeepSeek-V4.1-Flash `inference/image_processor.py` and
//! `inference/vision.py`, and Pillow's `libImaging/Resample.c` and `ImageOps.py` of the version
//! the oracle ran; `tools/ref/vision/` dumps what they produce and the gates compare against it.

pub mod arch;
pub mod grid;
pub mod image;
pub mod preprocess;
pub mod resample;
pub mod span;

pub use grid::{GridParams, GridPlan, plan_image_grid};
pub use image::Rgb8;
pub use preprocess::{Patches, preprocess};
pub use span::{ImageSpan, SpanType, image_span};

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
