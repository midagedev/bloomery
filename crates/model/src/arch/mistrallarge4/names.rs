//! Every tensor name a `mistral-large4` file spells. The model-level names
//! are the common ones beside the rope's frequency factors, and every layer's
//! stems are `blk.{l}.{stem}` (llama.cpp creates each with no
//! `TENSOR_NOT_REQUIRED`), the indexer's and the SSMax pairs among them.

pub use crate::arch::names::{output, output_norm, token_embd};

/// The pre-attention norm's gain.
pub const ATTN_NORM: &str = "attn_norm.weight";
/// The query projection: the hidden row to every head's `key_length_mla`
/// values, no query latent on the path.
pub const ATTN_Q: &str = "attn_q.weight";
/// The latent's down-projection, the rope part in its tail (`kv_lora_rank +
/// rope` values); a layer that reads another's cache carries none.
pub const ATTN_KV_A_MQA: &str = "attn_kv_a_mqa.weight";
/// The RMS gain on the latent.
pub const ATTN_KV_A_NORM: &str = "attn_kv_a_norm.weight";
/// The per-head up-projection of the latent to the head's unrotated key
/// values, the query absorbed against it.
pub const ATTN_K_B: &str = "attn_k_b.weight";
/// The per-head up-projection of the latent to the head's values.
pub const ATTN_V_B: &str = "attn_v_b.weight";
/// The attention output projection.
pub const ATTN_OUTPUT: &str = "attn_output.weight";
/// The SSMax scale `s_h`, one value per head.
pub const ATTN_SSMAX_W: &str = "attn_ssmax.weight";
/// The SSMax bias `b_h`, one value per head.
pub const ATTN_SSMAX_B: &str = "attn_ssmax.bias";
/// The indexer's LayerNorm gain on the layer input, before the query
/// projection; a full indexer layer's own.
pub const INDEXER_Q_NORM_W: &str = "indexer.q_norm.weight";
/// The indexer's LayerNorm bias on the layer input.
pub const INDEXER_Q_NORM_B: &str = "indexer.q_norm.bias";
/// The indexer's query projection: the normed input to every index head.
pub const INDEXER_ATTN_Q_B: &str = "indexer.attn_q_b.weight";
/// The index key projection: the layer input to one key row.
pub const INDEXER_ATTN_K: &str = "indexer.attn_k.weight";
/// The LayerNorm gain on the projected key.
pub const INDEXER_K_NORM_W: &str = "indexer.k_norm.weight";
/// The LayerNorm bias on the projected key.
pub const INDEXER_K_NORM_B: &str = "indexer.k_norm.bias";
/// The index heads' weights, one per head, on the rectified scores.
pub const INDEXER_PROJ: &str = "indexer.proj.weight";
/// The indexer's SSMax scale `s_h`, one value per index head.
pub const INDEXER_SSMAX_W: &str = "indexer.ssmax.weight";
/// The indexer's SSMax bias `b_h`, one value per index head.
pub const INDEXER_SSMAX_B: &str = "indexer.ssmax.bias";
/// The dense lead layer's FFN gate projection.
pub const FFN_GATE: &str = "ffn_gate.weight";
/// The dense lead layer's FFN up projection.
pub const FFN_UP: &str = "ffn_up.weight";
/// The dense lead layer's FFN down projection.
pub const FFN_DOWN: &str = "ffn_down.weight";
/// The pre-FFN norm's gain.
pub const FFN_NORM: &str = "ffn_norm.weight";
/// The router: one softmax score per expert.
pub const FFN_GATE_INP: &str = "ffn_gate_inp.weight";
/// The routed experts' gate stack.
pub const FFN_GATE_EXPS: &str = "ffn_gate_exps.weight";
/// The routed experts' up stack.
pub const FFN_UP_EXPS: &str = "ffn_up_exps.weight";
/// The routed experts' down stack.
pub const FFN_DOWN_EXPS: &str = "ffn_down_exps.weight";
/// The shared expert's gate projection.
pub const FFN_GATE_SHEXP: &str = "ffn_gate_shexp.weight";
/// The shared expert's up projection.
pub const FFN_UP_SHEXP: &str = "ffn_up_shexp.weight";
/// The shared expert's down projection.
pub const FFN_DOWN_SHEXP: &str = "ffn_down_shexp.weight";
