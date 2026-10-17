//! Device audio tower for decoded Gemma4a mel chunks.

pub mod arch;
mod ops;

pub use arch::gemma4a::{Encoded, Encoder};
