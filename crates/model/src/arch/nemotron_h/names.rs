//! Every tensor name a `nemotron_h` file spells. The model-level names are
//! the common ones; a layer's names are `blk.{l}.{stem}`.

use crate::arch::names::blk;

pub use crate::arch::names::{output, output_norm, token_embd};

/// The block RMS gain. Every layer carries it, whatever the layer mixes.
pub const ATTN_NORM: &str = "attn_norm.weight";

/// Mamba-2 `A`, stored without a `.weight` suffix.
pub const SSM_A: &str = "ssm_a";
/// Mamba-2 `D`.
pub const SSM_D: &str = "ssm_d";
/// The conv's bias. The loader marks it not required.
pub const SSM_CONV_BIAS: &str = "ssm_conv1d.bias";
/// The conv's weight.
pub const SSM_CONV: &str = "ssm_conv1d.weight";
/// The time-step bias.
pub const SSM_DT_BIAS: &str = "ssm_dt.bias";
/// The input projection. Its presence marks a Mamba-2 layer.
pub const SSM_IN: &str = "ssm_in.weight";
/// The grouped norm.
pub const SSM_NORM: &str = "ssm_norm.weight";
/// The output projection.
pub const SSM_OUT: &str = "ssm_out.weight";

/// The query projection. Its presence marks an attention layer.
pub const ATTN_Q: &str = "attn_q.weight";
/// The key projection.
pub const ATTN_K: &str = "attn_k.weight";
/// The value projection.
pub const ATTN_V: &str = "attn_v.weight";
/// The attention output projection.
pub const ATTN_OUTPUT: &str = "attn_output.weight";

/// The up projection, with no gate. Its presence marks a dense MLP layer.
pub const FFN_UP: &str = "ffn_up.weight";
/// The down projection.
pub const FFN_DOWN: &str = "ffn_down.weight";

/// `blk.{l}.attn_norm.weight`.
#[must_use]
pub fn attn_norm(l: usize) -> String {
    blk(l, ATTN_NORM)
}

/// `blk.{l}.ssm_in.weight`.
#[must_use]
pub fn ssm_in(l: usize) -> String {
    blk(l, SSM_IN)
}

/// `blk.{l}.attn_q.weight`.
#[must_use]
pub fn attn_q(l: usize) -> String {
    blk(l, ATTN_Q)
}

/// `blk.{l}.ffn_up.weight`.
#[must_use]
pub fn ffn_up(l: usize) -> String {
    blk(l, FFN_UP)
}
