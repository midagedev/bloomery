//! Every `blk.{N}.<tensor>.weight` string this architecture reads, in one place.
//!
//! The tensor names are a value that differs per architecture, not a type: the
//! shared modules (`ffn`, `moe`) hold the math and ask this table for the name
//! of the weight they need. Anything that formats a `blk.` literal elsewhere in
//! `crates/model/src` is a name this table does not yet own.

/// `blk.{block}.attn_q.weight` — the query projection.
pub fn attn_q(block: usize) -> String {
    format!("blk.{block}.attn_q.weight")
}

/// `blk.{block}.attn_kv_a_mqa.weight` — the KV down projection.
pub fn attn_kv_a_mqa(block: usize) -> String {
    format!("blk.{block}.attn_kv_a_mqa.weight")
}

/// `blk.{block}.attn_kv_b.weight` — the KV up projection (k-up ; v-up).
pub fn attn_kv_b(block: usize) -> String {
    format!("blk.{block}.attn_kv_b.weight")
}

/// `blk.{block}.attn_kv_a_norm.weight` — the latent RMS gain.
pub fn attn_kv_a_norm(block: usize) -> String {
    format!("blk.{block}.attn_kv_a_norm.weight")
}

/// `blk.{block}.attn_output.weight` — the attention output projection.
pub fn attn_output(block: usize) -> String {
    format!("blk.{block}.attn_output.weight")
}

/// `blk.{block}.attn_norm.weight` — the pre-attention RMS gain.
pub fn attn_norm(block: usize) -> String {
    format!("blk.{block}.attn_norm.weight")
}

/// `blk.{block}.ffn_norm.weight` — the pre-FFN RMS gain.
pub fn ffn_norm(block: usize) -> String {
    format!("blk.{block}.ffn_norm.weight")
}

/// `blk.{block}.ffn_gate_inp.weight` — the MoE router. Its presence in the file
/// is what makes a block routed; no block number is special-cased.
pub fn ffn_gate_inp(block: usize) -> String {
    format!("blk.{block}.ffn_gate_inp.weight")
}

/// `blk.{block}.ffn_gate.weight` — the dense SwiGLU gate.
pub fn ffn_gate(block: usize) -> String {
    format!("blk.{block}.ffn_gate.weight")
}

/// `blk.{block}.ffn_up.weight` — the dense SwiGLU up projection.
pub fn ffn_up(block: usize) -> String {
    format!("blk.{block}.ffn_up.weight")
}

/// `blk.{block}.ffn_down.weight` — the dense down projection.
pub fn ffn_down(block: usize) -> String {
    format!("blk.{block}.ffn_down.weight")
}

/// `blk.{block}.ffn_gate_shexp.weight` — the shared expert's SwiGLU gate.
pub fn ffn_gate_shexp(block: usize) -> String {
    format!("blk.{block}.ffn_gate_shexp.weight")
}

/// `blk.{block}.ffn_up_shexp.weight` — the shared expert's up projection.
pub fn ffn_up_shexp(block: usize) -> String {
    format!("blk.{block}.ffn_up_shexp.weight")
}

/// `blk.{block}.ffn_down_shexp.weight` — the shared expert's down projection.
pub fn ffn_down_shexp(block: usize) -> String {
    format!("blk.{block}.ffn_down_shexp.weight")
}

/// `blk.{block}.ffn_gate_exps.weight` — the routed experts' gate stack.
pub fn ffn_gate_exps(block: usize) -> String {
    format!("blk.{block}.ffn_gate_exps.weight")
}

/// `blk.{block}.ffn_up_exps.weight` — the routed experts' up stack.
pub fn ffn_up_exps(block: usize) -> String {
    format!("blk.{block}.ffn_up_exps.weight")
}

/// `blk.{block}.ffn_down_exps.weight` — the routed experts' down stack.
pub fn ffn_down_exps(block: usize) -> String {
    format!("blk.{block}.ffn_down_exps.weight")
}
