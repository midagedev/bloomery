//! Every tensor name an Inkling 2 step reads: the spellings every family takes from
//! [`crate::arch::names`] (the attention projections' and the router's among them), and the
//! names of this family's own — the relative-position bias table and its clips, the sink
//! projection, the short convolutions, the windowless layers' separate input norms and
//! conv taps, the block attention residuals' queries, the token embedding's norm, the fused
//! expert stack and the loop's message queries and controllers (both dialects' spellings,
//! [`hparams::Dialect`]).

use crate::arch::names::blk;

pub use crate::arch::names::{
    attn_gate, attn_k, attn_norm, attn_output, attn_q, attn_v, exp_probs_b, ffn_down_exps,
    ffn_gate_inp, ffn_norm, output, output_norm, token_embd,
};

/// `token_embd_norm.weight`: the gain of the norm on the token embeddings (raw image embeddings
/// skip it).
pub const TOKEN_EMBD_NORM: &str = "token_embd_norm.weight";

/// `output_res_query.weight`: the final mix's one query row.
pub const OUTPUT_RES_QUERY: &str = "output_res_query.weight";

/// `loop_res_query.weight`: the loop's message queries for the attention and ffn controllers,
/// two rows a core layer.
pub const LOOP_RES_QUERY: &str = "loop_res_query.weight";

/// `loop_kv_res_query.weight`: the loop's message queries for the k and v controllers, two rows
/// a core layer.
pub const LOOP_KV_RES_QUERY: &str = "loop_kv_res_query.weight";

/// `blk.{l}.attn_q_clip.weight` — the per-head channel scale on q.
#[must_use]
pub fn attn_q_clip(l: usize) -> String {
    blk(l, "attn_q_clip.weight")
}

/// `blk.{l}.attn_sink_proj.weight` — the per-(head, query) sink logit projection.
#[must_use]
pub fn attn_sink_proj(l: usize) -> String {
    blk(l, "attn_sink_proj.weight")
}

/// `blk.{l}.attn_rel.weight` — the r projection, `d_rel` values a head.
#[must_use]
pub fn attn_rel(l: usize) -> String {
    blk(l, "attn_rel.weight")
}

/// `blk.{l}.attn_rel_bias.weight` — the bias-vs-distance bank, `extent` values a head.
#[must_use]
pub fn attn_rel_bias(l: usize) -> String {
    blk(l, "attn_rel_bias.weight")
}

/// `blk.{l}.attn_rel_clip.weight` — the per-head channel scale on r.
#[must_use]
pub fn attn_rel_clip(l: usize) -> String {
    blk(l, "attn_rel_clip.weight")
}

/// `blk.{l}.attn_k_conv.weight` — the k path's static conv taps.
#[must_use]
pub fn attn_k_conv(l: usize) -> String {
    blk(l, "attn_k_conv.weight")
}

/// `blk.{l}.attn_v_conv.weight` — the v path's static conv taps.
#[must_use]
pub fn attn_v_conv(l: usize) -> String {
    blk(l, "attn_v_conv.weight")
}

/// `blk.{l}.attn_post_conv.weight` — the attention output's conv taps.
#[must_use]
pub fn attn_post_conv(l: usize) -> String {
    blk(l, "attn_post_conv.weight")
}

/// `blk.{l}.ffn_post_conv.weight` — the feed-forward block's output's conv taps.
#[must_use]
pub fn ffn_post_conv(l: usize) -> String {
    blk(l, "ffn_post_conv.weight")
}

/// `blk.{l}.attn_res_query.weight` — the attention mix's query rows (one a windowed layer,
/// three a windowless one); layer 0 carries none.
#[must_use]
pub fn attn_res_query(l: usize) -> String {
    blk(l, "attn_res_query.weight")
}

/// `blk.{l}.ffn_res_query.weight` — the feed-forward mix's one query row.
#[must_use]
pub fn ffn_res_query(l: usize) -> String {
    blk(l, "ffn_res_query.weight")
}

/// `blk.{l}.ffn_gate_up_exps.weight` — the routed experts' fused gate and up stack, its rows
/// interleaved (gate i, up i).
#[must_use]
pub fn ffn_gate_up_exps(l: usize) -> String {
    blk(l, "ffn_gate_up_exps.weight")
}

/// The windowless layers' own input norms and conv-tap projections, by part.
#[must_use]
pub fn inp(l: usize, part: Inp) -> String {
    blk(
        l,
        match part {
            Inp::NormQ => "attn_q_inp_norm.weight",
            Inp::NormK => "attn_k_inp_norm.weight",
            Inp::NormV => "attn_v_inp_norm.weight",
            Inp::ConvK => "attn_k_conv_proj.weight",
            Inp::ConvV => "attn_v_conv_proj.weight",
        },
    )
}

/// A windowless layer's input part an own tensor names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Inp {
    NormQ,
    NormK,
    NormV,
    ConvK,
    ConvV,
}

/// `blk.{l}.loop_attn_down.weight` and its eleven fellows — the message controllers of the
/// unsloth dialect's loop, `{module}_{down,gain,forget}` for `module` one of `attn`, `k`, `v`,
/// `mlp`.
#[must_use]
pub fn loop_ctrl_pr(l: usize, module: Ctrl, part: CtrlPart) -> String {
    blk(l, &format!("loop_{}_{part}.weight", module.pr()))
}

/// `blk.{l}.loop_msg_query.weight` — llama.cpp master's dialect: the four message queries of a
/// core layer in one tensor.
#[must_use]
pub fn loop_msg_query(l: usize) -> String {
    blk(l, "loop_msg_query.weight")
}

/// `blk.{l}.loop_q_ctrl_down.weight` and its eleven fellows — the message controllers of
/// llama.cpp master's dialect, `{module}_ctrl_{down,gain,forget}` for `module` one of `q`, `k`,
/// `v`, `ffn`.
#[must_use]
pub fn loop_ctrl_master(l: usize, module: Ctrl, part: CtrlPart) -> String {
    blk(l, &format!("loop_{}_ctrl_{part}.weight", module.master()))
}

/// One of a core layer's four message controllers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ctrl {
    /// The attention input's controller.
    Attn,
    /// The k path's.
    K,
    /// The v path's.
    V,
    /// The feed-forward input's.
    Ffn,
}

/// One controller's three projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CtrlPart {
    Down,
    Gain,
    Forget,
}

impl Ctrl {
    /// The unsloth dialect's module word.
    #[must_use]
    pub const fn pr(self) -> &'static str {
        match self {
            Ctrl::Attn => "attn",
            Ctrl::K => "k",
            Ctrl::V => "v",
            Ctrl::Ffn => "mlp",
        }
    }

    /// llama.cpp master's dialect's module word.
    #[must_use]
    pub const fn master(self) -> &'static str {
        match self {
            Ctrl::Attn => "q",
            Ctrl::K => "k",
            Ctrl::V => "v",
            Ctrl::Ffn => "ffn",
        }
    }
}

impl CtrlPart {
    /// The projection's word, both dialects' own.
    #[must_use]
    pub const fn word(self) -> &'static str {
        match self {
            CtrlPart::Down => "down",
            CtrlPart::Gain => "gain",
            CtrlPart::Forget => "forget",
        }
    }
}

impl std::fmt::Display for CtrlPart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.word())
    }
}
