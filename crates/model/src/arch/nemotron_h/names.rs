//! Every tensor name a `nemotron_h` file spells. The model-level names are
//! the common ones; a layer's names are `blk.{l}.{stem}`.

use crate::arch::names::blk;

pub use crate::arch::names::{output, output_norm, token_embd};

pub use crate::arch::hybrid::{
    ATTN_K, ATTN_NORM, ATTN_OUTPUT, ATTN_Q, ATTN_V, SSM_A, SSM_CONV, SSM_CONV_BIAS, SSM_D,
    SSM_DT_BIAS, SSM_IN, SSM_NORM, SSM_OUT,
};

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
