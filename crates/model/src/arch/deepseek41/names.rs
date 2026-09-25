//! Every tensor name the DeepSeek-V4.1 and V4 decode steps read, in one place:
//! the model-level names, then the `blk.{N}.*` names every layer carries, then
//! the ones only some layers carry. Which layers carry those is
//! [`LayerKind`](super::hparams::LayerKind)'s answer, decided by probing these
//! names. Anything that formats a `blk.` literal elsewhere in `crates/model/src`
//! is a name this table does not yet own.

/// `token_embd.weight` — the token embedding, one row per token.
pub fn token_embd() -> String {
    "token_embd.weight".to_string()
}

/// `output_norm.weight` — the RMS gain before the head.
pub fn output_norm() -> String {
    "output_norm.weight".to_string()
}

/// `output.weight` — the head, one row per vocabulary entry.
pub fn output() -> String {
    "output.weight".to_string()
}

/// `output_hc_fn.weight` — the hyper-connection head's mix of the streams
/// into one before the output norm; a V4 file carries it, a V4.1 file
/// collapses with the last FFN's lagged mix instead.
pub fn output_hc_fn() -> String {
    "output_hc_fn.weight".to_string()
}

/// `output_hc_base.weight` — the bias of the head's mix.
pub fn output_hc_base() -> String {
    "output_hc_base.weight".to_string()
}

/// `output_hc_scale.weight` — the scale of the head's mix.
pub fn output_hc_scale() -> String {
    "output_hc_scale.weight".to_string()
}

/// `blk.{block}.attn_norm.weight` — the pre-attention RMS gain.
pub fn attn_norm(block: usize) -> String {
    format!("blk.{block}.attn_norm.weight")
}

/// `blk.{block}.attn_q_a.weight` — the query down projection.
pub fn attn_q_a(block: usize) -> String {
    format!("blk.{block}.attn_q_a.weight")
}

/// `blk.{block}.attn_q_a_norm.weight` — the RMS gain on the query latent.
pub fn attn_q_a_norm(block: usize) -> String {
    format!("blk.{block}.attn_q_a_norm.weight")
}

/// `blk.{block}.attn_q_b.weight` — the query up projection, every head.
pub fn attn_q_b(block: usize) -> String {
    format!("blk.{block}.attn_q_b.weight")
}

/// `blk.{block}.attn_kv.weight` — the projection to the latent that is both
/// key and value.
pub fn attn_kv(block: usize) -> String {
    format!("blk.{block}.attn_kv.weight")
}

/// `blk.{block}.attn_kv_a_norm.weight` — the RMS gain on the KV latent.
pub fn attn_kv_a_norm(block: usize) -> String {
    format!("blk.{block}.attn_kv_a_norm.weight")
}

/// `blk.{block}.attn_sinks.weight` — one sink logit per head.
pub fn attn_sinks(block: usize) -> String {
    format!("blk.{block}.attn_sinks.weight")
}

/// `blk.{block}.attn_output_a.weight` — the grouped output projection, one
/// block of the diagonal per output group.
pub fn attn_output_a(block: usize) -> String {
    format!("blk.{block}.attn_output_a.weight")
}

/// `blk.{block}.attn_output_b.weight` — the projection from the groups back
/// to the stream width.
pub fn attn_output_b(block: usize) -> String {
    format!("blk.{block}.attn_output_b.weight")
}

/// `blk.{block}.hc_attn_fn.weight` — the hyper-connection mixes of the
/// attention sublayer, from the flattened streams.
pub fn hc_attn_fn(block: usize) -> String {
    format!("blk.{block}.hc_attn_fn.weight")
}

/// `blk.{block}.hc_attn_base.weight` — the bias of those mixes.
pub fn hc_attn_base(block: usize) -> String {
    format!("blk.{block}.hc_attn_base.weight")
}

/// `blk.{block}.hc_attn_scale.weight` — the scales of the pre, post and
/// combine mixes.
pub fn hc_attn_scale(block: usize) -> String {
    format!("blk.{block}.hc_attn_scale.weight")
}

/// `blk.{block}.hc_ffn_fn.weight` — the hyper-connection mixes of the FFN
/// sublayer.
pub fn hc_ffn_fn(block: usize) -> String {
    format!("blk.{block}.hc_ffn_fn.weight")
}

/// `blk.{block}.hc_ffn_base.weight` — the bias of those mixes.
pub fn hc_ffn_base(block: usize) -> String {
    format!("blk.{block}.hc_ffn_base.weight")
}

/// `blk.{block}.hc_ffn_scale.weight` — the scales of the pre, post and
/// combine mixes.
pub fn hc_ffn_scale(block: usize) -> String {
    format!("blk.{block}.hc_ffn_scale.weight")
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

/// `blk.{block}.ffn_gate_tid2eid.weight` — the hash router's table: the
/// experts a token id runs, one row of `expert_used_count` ids per token. Its
/// presence is what makes a block hash-routed.
pub fn ffn_gate_tid2eid(block: usize) -> String {
    format!("blk.{block}.ffn_gate_tid2eid.weight")
}

/// `blk.{block}.exp_probs_b.bias` — the router's selection bias.
pub fn exp_probs_b(block: usize) -> String {
    format!("blk.{block}.exp_probs_b.bias")
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

/// `blk.{block}.attn_compressor_kv.weight` — the compressor's latent
/// projection. Its presence is what makes a block a compressed-stream source.
pub fn attn_compressor_kv(block: usize) -> String {
    format!("blk.{block}.attn_compressor_kv.weight")
}

/// `blk.{block}.attn_compressor_gate.weight` — the compressor's pooling
/// scores; a source that pools one row per token has none.
pub fn attn_compressor_gate(block: usize) -> String {
    format!("blk.{block}.attn_compressor_gate.weight")
}

/// `blk.{block}.attn_compressor_norm.weight` — the RMS gain on a pooled row.
pub fn attn_compressor_norm(block: usize) -> String {
    format!("blk.{block}.attn_compressor_norm.weight")
}

/// `blk.{block}.attn_compressor_ape.weight` — the compressor's position table:
/// one score bias row per slot of a pooled group, added before the pooling
/// softmax.
pub fn attn_compressor_ape(block: usize) -> String {
    format!("blk.{block}.attn_compressor_ape.weight")
}

/// `blk.{block}.indexer_compressor_kv.weight` — the indexer's own compressor,
/// which pools the index keys from the layer input. Its presence is what
/// makes a block an index-key owner without `indexer.attn_k`.
pub fn indexer_compressor_kv(block: usize) -> String {
    format!("blk.{block}.indexer_compressor_kv.weight")
}

/// `blk.{block}.indexer_compressor_gate.weight` — its pooling scores.
pub fn indexer_compressor_gate(block: usize) -> String {
    format!("blk.{block}.indexer_compressor_gate.weight")
}

/// `blk.{block}.indexer_compressor_norm.weight` — the RMS gain on a pooled
/// index key.
pub fn indexer_compressor_norm(block: usize) -> String {
    format!("blk.{block}.indexer_compressor_norm.weight")
}

/// `blk.{block}.indexer_compressor_ape.weight` — its position table.
pub fn indexer_compressor_ape(block: usize) -> String {
    format!("blk.{block}.indexer_compressor_ape.weight")
}

/// `blk.{block}.indexer.attn_k.weight` — the index key projection of a pooled
/// row. Its presence is what makes a block an index-key owner.
pub fn indexer_attn_k(block: usize) -> String {
    format!("blk.{block}.indexer.attn_k.weight")
}

/// `blk.{block}.indexer.k_norm.weight` — the RMS gain on an index key.
pub fn indexer_k_norm(block: usize) -> String {
    format!("blk.{block}.indexer.k_norm.weight")
}

/// `blk.{block}.indexer.attn_q_b.weight` — the indexer's query projection. Its
/// presence is what makes a block run the top-k selection.
pub fn indexer_attn_q_b(block: usize) -> String {
    format!("blk.{block}.indexer.attn_q_b.weight")
}

/// `blk.{block}.indexer.proj.weight` — the indexer's per-head score weights.
pub fn indexer_proj(block: usize) -> String {
    format!("blk.{block}.indexer.proj.weight")
}

/// `blk.{block}.engram_embd.weight` — the engram table, read by gathered
/// rows. Its presence is what makes a block an engram site.
pub fn engram_embd(block: usize) -> String {
    format!("blk.{block}.engram_embd.weight")
}

/// `blk.{block}.engram_k.weight` — the RMS gain on the engram keys, one row
/// per stream.
pub fn engram_k(block: usize) -> String {
    format!("blk.{block}.engram_k.weight")
}

/// `blk.{block}.engram_q.weight` — the RMS gain on the streams the keys are
/// scored against, one row per stream.
pub fn engram_q(block: usize) -> String {
    format!("blk.{block}.engram_q.weight")
}

/// `blk.{block}.engram_wkv.weight` — the projection of the gathered rows into
/// one key per stream and the value they share.
pub fn engram_wkv(block: usize) -> String {
    format!("blk.{block}.engram_wkv.weight")
}
