//! Every tensor name a qwen4exp step reads, in one place: the model-level
//! names, then a layer's `blk.{N}.*` names by the sub-layer that reads them.
//! Which layers carry which is the layer's [`LayerSpec`](models::LayerSpec),
//! read by [`super::spec`]; the roles table ([`super::roles`]) holds the same
//! stems for the classifier, and the plan's gate holds every name to the file
//! (`tests/qwen4exp_meta.rs`).

/// `blk.{l}.{stem}`.
fn blk(l: usize, stem: &str) -> String {
    format!("blk.{l}.{stem}")
}

pub fn token_embd() -> String {
    "token_embd.weight".to_string()
}

pub fn output() -> String {
    "output.weight".to_string()
}

/// The head's hyper-connection mix: the streams' grouped RMS gain, then the
/// low-rank gate's down and up — the output norm of this architecture.
pub fn output_hc_norm() -> String {
    "output_hc_norm.weight".to_string()
}

pub fn output_hc_down() -> String {
    "output_hc_down.weight".to_string()
}

pub fn output_hc_up() -> String {
    "output_hc_up.weight".to_string()
}

/// The PLE table: rows of `embedding_length_per_layer_input` values that the
/// host gathers by the n-gram hash; it has no layer in its name.
pub fn per_layer_token_embd() -> String {
    "per_layer_token_embd.weight".to_string()
}

// ---------------------------------------------------------------- GDN

pub fn attn_qkv(l: usize) -> String {
    blk(l, "attn_qkv.weight")
}

/// The output gate's projection `z` (a sigmoid on qwen4exp).
pub fn attn_gate(l: usize) -> String {
    blk(l, "attn_gate.weight")
}

pub fn ssm_conv1d(l: usize) -> String {
    blk(l, "ssm_conv1d.weight")
}

pub fn ssm_dt_bias(l: usize) -> String {
    blk(l, "ssm_dt.bias")
}

pub fn ssm_a(l: usize) -> String {
    blk(l, "ssm_a")
}

pub fn ssm_alpha(l: usize) -> String {
    blk(l, "ssm_alpha.weight")
}

pub fn ssm_beta(l: usize) -> String {
    blk(l, "ssm_beta.weight")
}

pub fn ssm_norm(l: usize) -> String {
    blk(l, "ssm_norm.weight")
}

pub fn ssm_out(l: usize) -> String {
    blk(l, "ssm_out.weight")
}

// ---------------------------------------------------------------- GQA

/// The query with its per-head output gate interleaved.
pub fn attn_q(l: usize) -> String {
    blk(l, "attn_q.weight")
}

pub fn attn_k(l: usize) -> String {
    blk(l, "attn_k.weight")
}

pub fn attn_v(l: usize) -> String {
    blk(l, "attn_v.weight")
}

pub fn attn_q_norm(l: usize) -> String {
    blk(l, "attn_q_norm.weight")
}

pub fn attn_k_norm(l: usize) -> String {
    blk(l, "attn_k_norm.weight")
}

pub fn attn_output(l: usize) -> String {
    blk(l, "attn_output.weight")
}

/// The mean-pool indexer's query projection (bf16 in the file).
pub fn indexer_q_proj(l: usize) -> String {
    blk(l, "indexer.q_proj.weight")
}

/// The indexer's raw key projection: its output is cached unnormed and
/// unturned, a position's row per position (bf16 in the file).
pub fn indexer_k_proj(l: usize) -> String {
    blk(l, "indexer.k_proj.weight")
}

pub fn indexer_q_norm(l: usize) -> String {
    blk(l, "indexer.q_norm.weight")
}

pub fn indexer_k_norm(l: usize) -> String {
    blk(l, "indexer.k_norm.weight")
}

// -------------------------------------------------------- feed-forward

/// The router's f32 logits matrix; its input is the block's mix, no norm.
pub fn ffn_gate_inp(l: usize) -> String {
    blk(l, "ffn_gate_inp.weight")
}

/// The shared expert's scalar gate row: `σ(row · x)` scales its output.
pub fn ffn_gate_inp_shexp(l: usize) -> String {
    blk(l, "ffn_gate_inp_shexp.weight")
}

pub fn ffn_gate_shexp(l: usize) -> String {
    blk(l, "ffn_gate_shexp.weight")
}

pub fn ffn_up_shexp(l: usize) -> String {
    blk(l, "ffn_up_shexp.weight")
}

pub fn ffn_down_shexp(l: usize) -> String {
    blk(l, "ffn_down_shexp.weight")
}

pub fn ffn_gate_exps(l: usize) -> String {
    blk(l, "ffn_gate_exps.weight")
}

pub fn ffn_up_exps(l: usize) -> String {
    blk(l, "ffn_up_exps.weight")
}

pub fn ffn_down_exps(l: usize) -> String {
    blk(l, "ffn_down_exps.weight")
}

// ---------------------------------------------------- hyper-connections

/// The sub-layer a gated-residual mix feeds: the mixer or the feed-forward
/// block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sub {
    Attn,
    Ffn,
}

impl Sub {
    fn stem(self) -> &'static str {
        match self {
            Sub::Attn => "attn",
            Sub::Ffn => "ffn",
        }
    }
}

/// The streams' grouped RMS gain, `[n_embd · streams]`.
pub fn hc_norm(l: usize, sub: Sub) -> String {
    blk(l, &format!("hc_{}_norm.weight", sub.stem()))
}

/// The low-rank gate's down projection, `streams · n_embd → rank` (q8_0).
pub fn hc_down(l: usize, sub: Sub) -> String {
    blk(l, &format!("hc_{}_down.weight", sub.stem()))
}

/// The low-rank gate's up projection, `rank → streams · n_embd` (q8_0).
pub fn hc_up(l: usize, sub: Sub) -> String {
    blk(l, &format!("hc_{}_up.weight", sub.stem()))
}

/// The inject projection, `streams · n_embd → streams` (f32).
pub fn hc_inject(l: usize, sub: Sub) -> String {
    blk(l, &format!("hc_{}_inject.weight", sub.stem()))
}

// ---------------------------------------------------------------- PLE

pub fn ple_key(l: usize) -> String {
    blk(l, "ple_key.weight")
}

pub fn ple_value(l: usize) -> String {
    blk(l, "ple_value.weight")
}

pub fn ple_norm_key(l: usize) -> String {
    blk(l, "ple_norm_key.weight")
}

pub fn ple_norm_query(l: usize) -> String {
    blk(l, "ple_norm_query.weight")
}

pub fn ple_norm_conv(l: usize) -> String {
    blk(l, "ple_norm_conv.weight")
}

pub fn ple_conv1d(l: usize) -> String {
    blk(l, "ple_conv1d.weight")
}
