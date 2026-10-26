//! The `llama` architecture's families: the plain dense Llama layout (every layer full attention with an
//! SwiGLU feed-forward, rope over adjacent pairs). Its one family today is [`dense::LOGITS`], llama.cpp mainline's
//! logits rows of Nanbeige4.3-33B, a model file of its own.

use crate::family::Family;

pub mod dense;

/// The architecture every `llama` manifest names in its `# arch` line.
pub const ARCH: &str = "llama";

/// The mainline commit every dense-logits set names in its `# build` line: the one tree
/// `tools/ref/dense-logits.py` links against whatever the model.
pub const LCPP_BUILD: &str = super::qwen35::LCPP_BUILD;

/// The architecture's families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[&dense::LOGITS];
