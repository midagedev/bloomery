//! The qwen3moe family on this backend, as the chain plan names it: the
//! qwen35moe token, as this backend counts it and runs it, and the session
//! as the generation loop's target (`target`).

#[cfg(target_os = "macos")]
mod arena;
pub mod count;
#[cfg(target_os = "macos")]
mod ops;
#[cfg(target_os = "macos")]
pub mod session;
#[cfg(target_os = "macos")]
pub mod target;
