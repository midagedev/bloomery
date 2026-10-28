//! Every tensor name an `lfm3` file spells. The model-level names are the common ones, and the
//! rope's factors; a layer's names are `blk.{l}.{stem}`.

pub(crate) use crate::arch::classify::ROPE_FREQS;
pub use crate::arch::names::{blk, output, output_norm, token_embd};

/// The mixer's pre-norm, on a short-conv and an attention layer alike (the reference's
/// `operator_norm`).
pub const ATTN_NORM: &str = "attn_norm.weight";
/// The norm on the mixer's output before it joins the stream.
pub const POST_ATTN_NORM: &str = "post_attention_norm.weight";
/// The feed-forward block's pre-norm.
pub const FFN_NORM: &str = "ffn_norm.weight";
/// The norm on the feed-forward block's output before it joins the stream.
pub const POST_FFN_NORM: &str = "post_ffw_norm.weight";
/// The SwiGLU block's gate projection.
pub const FFN_GATE: &str = "ffn_gate.weight";
/// The SwiGLU block's up projection.
pub const FFN_UP: &str = "ffn_up.weight";
/// The SwiGLU block's down projection.
pub const FFN_DOWN: &str = "ffn_down.weight";

/// The short conv's input projection, `B | C | x` in that order. Its presence marks a short-conv
/// layer.
pub const CONV_IN: &str = "shortconv.in_proj.weight";
/// The short conv's filters: `taps` values for each channel, the oldest tap first.
pub const CONV_TAPS: &str = "shortconv.conv.weight";
/// The short conv's output projection.
pub const CONV_OUT: &str = "shortconv.out_proj.weight";

/// The query projection. Its presence marks an attention layer.
pub const ATTN_Q: &str = "attn_q.weight";
/// The key projection.
pub const ATTN_K: &str = "attn_k.weight";
/// The value projection.
pub const ATTN_V: &str = "attn_v.weight";
/// The attention output projection.
pub const ATTN_OUTPUT: &str = "attn_output.weight";
/// The per-head output gate's projection: one value per query head.
pub const ATTN_GATE: &str = "attn_gate.weight";
/// The RMS gain every query head is normed with.
pub const ATTN_Q_NORM: &str = "attn_q_norm.weight";
/// The RMS gain every key head is normed with.
pub const ATTN_K_NORM: &str = "attn_k_norm.weight";
