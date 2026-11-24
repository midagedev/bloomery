//! What a model folder's `config.json` says and where each of its keys goes: into a header key of
//! the file ([`MAP`], by suffix of `ernie55.`), into nothing because the architecture fixes it
//! ([`FIXED`], the value the description assumes) or into nothing because the engine never runs it
//! ([`IGNORED`]). Every key of the released config is in exactly one of the three; the converter
//! that writes the header and this reader hold to the same table.

/// A config key and the header key it becomes, by suffix. Two config keys that name one header
/// key hold one value, and the converter refuses a config where they differ. A transformed value
/// says how in its entry.
pub const MAP: &[(&str, &str)] = &[
    ("num_hidden_layers", "block_count"),
    ("max_position_embeddings", "context_length"),
    ("hidden_size", "embedding_length"),
    ("vocab_size", "vocab_size"),
    ("intermediate_size", "feed_forward_length"),
    ("first_k_dense_replace", "leading_dense_block_count"),
    ("num_attention_heads", "attention.head_count"),
    ("hybrid_mla_num_attention_heads", "attention.head_count"),
    ("rms_norm_eps", "attention.layer_norm_rms_epsilon"),
    // A latent layer's entry (-2 in the config) is written 0.
    ("compress_ratios", "attention.compress_ratios"),
    ("sliding_window", "attention.sliding_window"),
    ("q_lora_rank", "attention.q_lora_rank"),
    ("hybrid_mla_q_lora_rank", "attention.q_lora_rank"),
    ("kv_lora_rank", "attention.kv_lora_rank"),
    ("hybrid_mla_kv_lora_rank", "attention.kv_lora_rank"),
    // `attention.key_length_mla` is the no-position values plus the rotated ones; the cached
    // row `attention.key_length` is `kv_lora_rank` plus the rotated ones.
    ("hybrid_mla_qk_nope_head_dim", "attention.key_length_mla"),
    ("hybrid_mla_v_head_dim", "attention.value_length_mla"),
    ("hybrid_mla_qk_rope_head_dim", "rope.dimension_count"),
    ("qk_rope_head_dim", "rope.dimension_count"),
    ("qk_pos_emb_head_dim", "rope.dimension_count"),
    ("head_dim", "attention.hca.head_length"),
    ("v_head_dim", "attention.hca.head_length"),
    ("o_groups", "attention.output_group_count"),
    ("o_lora_rank", "attention.output_lora_rank"),
    ("index_n_heads", "attention.indexer.head_count"),
    ("index_head_dim", "attention.indexer.key_length"),
    ("index_topk", "attention.indexer.top_k"),
    ("rope_theta", "rope.freq_base"),
    ("rotary_base", "rope.freq_base"),
    ("compress_rope_theta", "attention.compress_rope_freq_base"),
    // The nested object becomes `rope.scaling.type`, `.factor`, `.original_context_length`,
    // `.yarn_beta_fast` and `.yarn_beta_slow`.
    ("rope_scaling", "rope.scaling"),
    ("n_routed_experts", "expert_count"),
    ("num_experts_per_tok", "expert_used_count"),
    ("moe_intermediate_size", "expert_feed_forward_length"),
    ("n_shared_experts", "expert_shared_count"),
    ("moe_latent_size", "expert_latent_length"),
    // The initial value of the learned per-expert factor; the factors themselves are
    // `ffn_exp_scale`.
    ("routed_scaling_factor", "expert_weights_scale"),
    ("norm_topk_prob", "expert_weights_norm"),
    ("scoring_func", "expert_gating_func"),
    ("n_group", "expert_group_count"),
    ("topk_group", "expert_group_used_count"),
    ("activation_situ_beta", "activation.situ_beta"),
    ("activation_situ_linear_beta", "activation.situ_linear_beta"),
    ("hc_mult", "hyper_connection.count"),
    ("hc_eps", "hyper_connection.epsilon"),
    ("hc_sinkhorn_iters", "hyper_connection.sinkhorn_iterations"),
];

/// A config key the architecture fixes, and the value the description assumes of it, as JSON.
/// A model folder whose config differs is another architecture.
pub const FIXED: &[(&str, &str)] = &[
    ("hidden_act", "\"situ\""),
    ("gated_linear_unit", "true"),
    ("use_mla", "false"),
    ("use_gated_attn", "true"),
    ("gated_attn_use_q_lora", "true"),
    ("gated_attn_act", "\"sigmoid\""),
    ("use_vha_attention", "true"),
    ("use_qk_norm", "true"),
    ("add_full_attention_sink_bias", "true"),
    ("enable_hyper_connections", "true"),
    ("latent_moe_use_norm", "true"),
    ("moe_split_feature_routing", "true"),
    ("moe_shared_expert_gate", "false"),
    ("moe_layer_freq", "1"),
    ("topk_method", "\"noaux_tc\""),
    ("routed_scaling_factor_learnable", "true"),
    ("multimax", "\"lm_head\""),
    ("tie_word_embeddings", "false"),
    ("num_key_value_heads", "1"),
    ("experimental_attention_variant", "\"dsv4_hybrid\""),
    ("hybrid_mla_attention", "\"mqa_dsa\""),
    ("mqa_split_kv_b_proj", "true"),
    ("hca_rope_type", "\"yarn\""),
    ("rope_type", "\"rope\""),
    ("rope_interleave", "true"),
    ("rotary_percent", "1.0"),
    ("attention_bias", "false"),
    ("use_bias", "false"),
    ("num_nextn_predict_layers", "0"),
];

/// A config key the engine does not run, by name: the training-side flags the reference
/// implementations never read, the framework's serving knobs, the draft head's (`dspark_*`, a
/// later round's), the tokenizer's ids (the tokenizer's own tables carry them) and a width the
/// others derive.
pub const IGNORED: &[&str] = &[
    // Training-side flags.
    "use_attn_merger",
    "qk_norm_type",
    "csa_dense_mode",
    "moe_group_experts",
    "mtp_shared_last_layer",
    "use_dense_mtp",
    // The framework's: expert dispatch, a hadamard kernel choice, the cache flag, the checkpoint's
    // own labels and quantization (the file's tensor types carry it).
    "moe_token_dispatcher_type",
    "use_fast_hadamard",
    "use_cache",
    "architectures",
    "model_type",
    "torch_dtype",
    "transformers_version",
    "quantization_config",
    // The draft head.
    "dspark_block_size",
    "dspark_markov_rank",
    "dspark_noise_token_id",
    "dspark_target_layer_ids",
    // The tokenizer's ids.
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    // Widths another key gives: the heads of a latent layer's expanded key-value (the cached row
    // is shared by every head) and an HCA head's no-position values (`head_dim` less the rotated).
    "hybrid_mla_num_key_value_heads",
    "qk_nope_head_dim",
];

#[cfg(test)]
mod tests {
    use super::{FIXED, IGNORED, MAP};

    /// A key is in one table only, and no table lists it twice.
    #[test]
    fn a_config_key_has_one_home() {
        let mut keys: Vec<&str> = MAP.iter().map(|(k, _)| *k).collect();
        keys.extend(FIXED.iter().map(|(k, _)| *k));
        keys.extend(IGNORED);
        let n = keys.len();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), n);
    }
}
