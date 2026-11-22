//! The qwen3moe family on this backend, as the chain plan names it: the
//! qwen35moe token, as this backend counts it and runs it, the session as
//! the generation loop's target (`target`), and the greedy generation a
//! caller runs over it (`generate`). The family-neutral parts of the session
//! live in [`crate::session`] and are re-exported here under the paths the
//! bins and gates read them at.

pub use crate::session::lanes;

#[cfg(target_os = "macos")]
mod arena;
#[cfg(target_os = "macos")]
pub mod check;
pub mod count;
#[cfg(target_os = "macos")]
pub mod generate;
#[cfg(target_os = "macos")]
pub mod mtp_check;
#[cfg(target_os = "macos")]
mod ops;
#[cfg(target_os = "macos")]
pub mod plan;
#[cfg(target_os = "macos")]
mod rows;
#[cfg(target_os = "macos")]
pub mod session;
#[cfg(target_os = "macos")]
pub mod target;
