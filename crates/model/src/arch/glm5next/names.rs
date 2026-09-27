//! Every tensor name the glm5next step reads, in one place: the model-level
//! names, then a layer's `blk.{N}.*` names by the sub-layer that reads them.
//! Which layers carry which is the layer's [`LayerSpec`](models::LayerSpec),
//! read by [`super::spec`]; the roles table ([`super::roles`]) holds the same
//! stems for the classifier.

/// `blk.{l}.{stem}`.
fn blk(l: usize, stem: &str) -> String {
    format!("blk.{l}.{stem}")
}

pub fn token_embd() -> String {
    "token_embd.weight".to_string()
}

pub fn output_norm() -> String {
    "output_norm.weight".to_string()
}

pub fn output() -> String {
    "output.weight".to_string()
}

/// The mixer's RMS gain, both kinds.
pub fn attn_norm(l: usize) -> String {
    blk(l, "attn_norm.weight")
}

/// The mixer's output projection, both kinds.
pub fn attn_output(l: usize) -> String {
    blk(l, "attn_output.weight")
}

// ---------------------------------------------------------------- KDA

pub fn attn_q(l: usize) -> String {
    blk(l, "attn_q.weight")
}

pub fn attn_k(l: usize) -> String {
    blk(l, "attn_k.weight")
}

pub fn attn_v(l: usize) -> String {
    blk(l, "attn_v.weight")
}

/// The q, k and v projections joined at load into one row stream, the conv's
/// channel order.
pub fn attn_qkv(l: usize) -> String {
    format!("derived.blk.{l}.attn_qkv")
}

/// The conv taps of the q, k or v channels, `conv` per channel.
pub fn ssm_conv1d(l: usize, part: char) -> String {
    blk(l, &format!("ssm_conv1d_{part}.weight"))
}

/// The three conv tap sets joined at load in the q, k, v channel order.
pub fn ssm_conv1d_qkv(l: usize) -> String {
    format!("derived.blk.{l}.ssm_conv1d_qkv")
}

pub fn ssm_f_a(l: usize) -> String {
    blk(l, "ssm_f_a.weight")
}

pub fn ssm_f_b(l: usize) -> String {
    blk(l, "ssm_f_b.weight")
}

pub fn ssm_g_a(l: usize) -> String {
    blk(l, "ssm_g_a.weight")
}

pub fn ssm_g_b(l: usize) -> String {
    blk(l, "ssm_g_b.weight")
}

pub fn ssm_beta(l: usize) -> String {
    blk(l, "ssm_beta.weight")
}

pub fn ssm_a(l: usize) -> String {
    blk(l, "ssm_a")
}

pub fn ssm_dt_bias(l: usize) -> String {
    blk(l, "ssm_dt.bias")
}

pub fn ssm_norm(l: usize) -> String {
    blk(l, "ssm_norm.weight")
}

// ------------------------------------------------------------- latent

pub fn attn_q_a(l: usize) -> String {
    blk(l, "attn_q_a.weight")
}

pub fn attn_q_a_norm(l: usize) -> String {
    blk(l, "attn_q_a_norm.weight")
}

pub fn attn_q_b(l: usize) -> String {
    blk(l, "attn_q_b.weight")
}

pub fn attn_kv_a_mqa(l: usize) -> String {
    blk(l, "attn_kv_a_mqa.weight")
}

pub fn attn_kv_a_norm(l: usize) -> String {
    blk(l, "attn_kv_a_norm.weight")
}

pub fn attn_k_b(l: usize) -> String {
    blk(l, "attn_k_b.weight")
}

pub fn attn_v_b(l: usize) -> String {
    blk(l, "attn_v_b.weight")
}

pub fn indexer_attn_k(l: usize) -> String {
    blk(l, "indexer.attn_k.weight")
}

pub fn indexer_k_norm(l: usize) -> String {
    blk(l, "indexer.k_norm.weight")
}

pub fn indexer_k_norm_bias(l: usize) -> String {
    blk(l, "indexer.k_norm.bias")
}

pub fn indexer_compressor_gate(l: usize) -> String {
    blk(l, "indexer_compressor_gate.weight")
}

/// The four projections of the normed input joined at load into one row
/// stream: `[q_a; latent; index key; pool gate]` per token.
pub fn attn_a_stack(l: usize) -> String {
    format!("derived.blk.{l}.attn_a_stack")
}

// -------------------------------------------------------- feed-forward

pub fn ffn_norm(l: usize) -> String {
    blk(l, "ffn_norm.weight")
}

pub fn ffn_gate(l: usize) -> String {
    blk(l, "ffn_gate.weight")
}

pub fn ffn_up(l: usize) -> String {
    blk(l, "ffn_up.weight")
}

pub fn ffn_down(l: usize) -> String {
    blk(l, "ffn_down.weight")
}

pub fn ffn_gate_inp(l: usize) -> String {
    blk(l, "ffn_gate_inp.weight")
}

pub fn exp_probs_b(l: usize) -> String {
    blk(l, "exp_probs_b.bias")
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

pub fn ffn_gate_shexp(l: usize) -> String {
    blk(l, "ffn_gate_shexp.weight")
}

pub fn ffn_up_shexp(l: usize) -> String {
    blk(l, "ffn_up_shexp.weight")
}

pub fn ffn_down_shexp(l: usize) -> String {
    blk(l, "ffn_down_shexp.weight")
}

// ---------------------------------------------------- hyper-connections

/// The sub-layer a hyper-connection mix feeds: the mixer or the
/// feed-forward block.
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

pub fn hc_fn(l: usize, sub: Sub) -> String {
    blk(l, &format!("hc_{}_fn.weight", sub.stem()))
}

pub fn hc_scale(l: usize, sub: Sub) -> String {
    blk(l, &format!("hc_{}_scale.weight", sub.stem()))
}

pub fn hc_base(l: usize, sub: Sub) -> String {
    blk(l, &format!("hc_{}_base.weight", sub.stem()))
}
