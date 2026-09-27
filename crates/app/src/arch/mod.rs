//! Each model's session pieces: its [`crate::Open`], [`crate::Prompt`] and
//! [`crate::Keep`], and the drafts that run beside it. A binary turns on the
//! feature of the model it opens.

#[cfg(feature = "deepseek41")]
pub mod deepseek41;
