//! Each model's session pieces: its [`crate::Open`], [`crate::Prompt`] and
//! [`crate::Keep`], and the drafts that run beside it. A binary turns on the
//! feature of the model it opens.

#[cfg(feature = "deepseek41")]
pub mod deepseek41;
#[cfg(feature = "glm5next")]
pub mod glm5next;
/// The qwen3moe family's Qwen3.8 session pieces: the body lives in the gpu
/// crate's own qwen3moe arch, so they need no feature of their own.
pub mod qwen3moe;
