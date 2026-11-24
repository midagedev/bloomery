//! The ERNIE 5.5 map's gates, host only: the real config read into its header keys, a toy folder of
//! three layers (a compressed-attention layer with a dense block, one with a routed block and a
//! latent-attention layer with a routed block) planned from its headers alone, every refusal by
//! name, and the real folder planned from its 128 shards' headers.

use std::path::Path;

use gguf::GgmlType::{BF16, F32, FP8_B128, MXFP4};
use gguf::arch::ernie55::{ARCH, STEM_TYPES, allowed_types};
use hf::quant_config::{Config, Format};
use safetensors::Dtype;
use serde_json::json;

use super::*;
use crate::folder::{self, Keep, Make};
use crate::test_support::Entries;

/// The released `config.json`, as the folder holds it.
const REAL_CONFIG: &str = r#"{
 "activation_situ_beta": 4.0,
 "activation_situ_linear_beta": 25.0,
 "add_full_attention_sink_bias": true,
 "architectures": [
  "Ernie5_5_MoeForCausalLM"
 ],
 "attention_bias": false,
 "bos_token_id": 1,
 "compress_ratios": [
  128,
  128,
  128,
  128,
  128,
  128,
  128,
  -2,
  128,
  128,
  128,
  128,
  128,
  128,
  -2,
  128,
  128,
  128,
  128,
  128,
  128,
  -2,
  128,
  128,
  128,
  128,
  128,
  128,
  -2,
  128,
  128,
  128,
  128,
  128,
  128,
  -2,
  128,
  128,
  128,
  128,
  128,
  128,
  -2
 ],
 "compress_rope_theta": 160000.0,
 "csa_dense_mode": true,
 "dspark_block_size": 7,
 "dspark_markov_rank": 256,
 "dspark_noise_token_id": 1024,
 "dspark_target_layer_ids": [
  40,
  41,
  42
 ],
 "enable_hyper_connections": true,
 "eos_token_id": 2,
 "experimental_attention_variant": "dsv4_hybrid",
 "first_k_dense_replace": 1,
 "gated_attn_act": "sigmoid",
 "gated_attn_use_q_lora": true,
 "gated_linear_unit": true,
 "hc_eps": 1e-06,
 "hc_mult": 4,
 "hc_sinkhorn_iters": 20,
 "hca_rope_type": "yarn",
 "head_dim": 512,
 "hidden_act": "situ",
 "hidden_size": 4096,
 "hybrid_mla_attention": "mqa_dsa",
 "hybrid_mla_kv_lora_rank": 512,
 "hybrid_mla_num_attention_heads": 64,
 "hybrid_mla_num_key_value_heads": 64,
 "hybrid_mla_q_lora_rank": 1024,
 "hybrid_mla_qk_nope_head_dim": 192,
 "hybrid_mla_qk_rope_head_dim": 64,
 "hybrid_mla_v_head_dim": 256,
 "index_head_dim": 128,
 "index_n_heads": 64,
 "index_topk": 2048,
 "intermediate_size": 14336,
 "kv_lora_rank": 512,
 "latent_moe_use_norm": true,
 "max_position_embeddings": 1048576,
 "model_type": "ernie5_5_moe",
 "moe_group_experts": false,
 "moe_intermediate_size": 2048,
 "moe_latent_size": 2048,
 "moe_layer_freq": 1,
 "moe_shared_expert_gate": false,
 "moe_split_feature_routing": true,
 "moe_token_dispatcher_type": "deepep",
 "mqa_split_kv_b_proj": true,
 "mtp_shared_last_layer": true,
 "multimax": "lm_head",
 "n_group": 1,
 "n_routed_experts": 512,
 "n_shared_experts": 1,
 "norm_topk_prob": true,
 "num_attention_heads": 64,
 "num_experts_per_tok": 10,
 "num_hidden_layers": 43,
 "num_key_value_heads": 1,
 "num_nextn_predict_layers": 0,
 "o_groups": 8,
 "o_lora_rank": 1024,
 "pad_token_id": 0,
 "q_lora_rank": 1024,
 "qk_nope_head_dim": 448,
 "qk_norm_type": "per_head",
 "qk_pos_emb_head_dim": 64,
 "qk_rope_head_dim": 64,
 "quantization_config": {
  "activation_scheme": "dynamic",
  "expert_dtype": "mxfp4",
  "fmt": "e4m3",
  "quant_method": "fp8",
  "scale_fmt": "ue8m0",
  "weight_block_size": [
   128,
   128
  ]
 },
 "rms_norm_eps": 1e-06,
 "rope_interleave": true,
 "rope_scaling": {
  "beta_fast": 32,
  "beta_slow": 1,
  "factor": 16,
  "original_max_position_embeddings": 65536,
  "type": "yarn"
 },
 "rope_theta": 8000000,
 "rope_type": "rope",
 "rotary_base": 8000000,
 "rotary_percent": 1.0,
 "routed_scaling_factor": 3.0,
 "routed_scaling_factor_learnable": true,
 "scoring_func": "sigmoid",
 "sliding_window": 128,
 "tie_word_embeddings": false,
 "topk_group": 1,
 "topk_method": "noaux_tc",
 "torch_dtype": "bfloat16",
 "transformers_version": "4.46.1",
 "use_attn_merger": true,
 "use_bias": false,
 "use_cache": true,
 "use_dense_mtp": false,
 "use_fast_hadamard": true,
 "use_gated_attn": true,
 "use_mla": false,
 "use_qk_norm": true,
 "use_vha_attention": true,
 "v_head_dim": 512,
 "vocab_size": 201216
}"#;

/// The toy's widths. Every first dim of an F8 weight is whole 128-blocks along its columns, and
/// the widths differ where the map could mix two tensors up.
const E: usize = 256;
const FF: usize = 512;
const LAYERS: usize = 3;
const DENSE_LAYERS: usize = 1;
const HEADS: usize = 2;
const D: usize = 128;
const ROPE: usize = 64;
const Q_LORA: usize = 384;
const KV_LORA: usize = 512;
const O_GROUPS: usize = 2;
const O_LORA: usize = 128;
const NOPE: usize = 192;
const V: usize = 256;
const INDEX_HEADS: usize = 2;
const INDEX_DIM: usize = 128;
const EFF: usize = 384;
const LATENT: usize = 128;
const N_EXP: usize = 4;
const VOCAB: usize = 8;
const RANK: usize = 16;
const RATIO: usize = 128;
/// A hyper-connection mixer's rows: the four streams of the hidden width, and its columns.
const MIX_IN: usize = 4 * E;
const MIX_OUT: usize = 24;

/// Layer 2 is the latent-attention one.
fn dsa(l: usize) -> bool {
    l == 2
}

/// The real config at the toy's widths.
fn toy_config() -> Json {
    let mut c: Json = serde_json::from_str(REAL_CONFIG).unwrap();
    for (key, v) in [
        ("hidden_size", E),
        ("intermediate_size", FF),
        ("num_hidden_layers", LAYERS),
        ("first_k_dense_replace", DENSE_LAYERS),
        ("num_attention_heads", HEADS),
        ("hybrid_mla_num_attention_heads", HEADS),
        ("hybrid_mla_num_key_value_heads", HEADS),
        ("head_dim", D),
        ("v_head_dim", D),
        ("qk_nope_head_dim", D - ROPE),
        ("q_lora_rank", Q_LORA),
        ("hybrid_mla_q_lora_rank", Q_LORA),
        ("kv_lora_rank", KV_LORA),
        ("hybrid_mla_kv_lora_rank", KV_LORA),
        ("o_groups", O_GROUPS),
        ("o_lora_rank", O_LORA),
        ("hybrid_mla_qk_nope_head_dim", NOPE),
        ("hybrid_mla_v_head_dim", V),
        ("index_n_heads", INDEX_HEADS),
        ("index_head_dim", INDEX_DIM),
        ("index_topk", 16),
        ("moe_intermediate_size", EFF),
        ("moe_latent_size", LATENT),
        ("n_routed_experts", N_EXP),
        ("num_experts_per_tok", 2),
        ("vocab_size", VOCAB),
    ] {
        c[key] = json!(v);
    }
    c["compress_ratios"] = json!([RATIO, RATIO, -2]);
    c["dspark_target_layer_ids"] = json!([0, 1, 2]);
    c
}

/// The stored scale grid of an F8 weight of `rows` by `k`: 128 by 128 blocks, rounded up.
fn grid(rows: usize, k: usize) -> Vec<usize> {
    vec![rows.div_ceil(128), k.div_ceil(128)]
}

/// A folder's tensors as the checkpoint stores them.
struct Toy(Entries);

impl Toy {
    fn add(&mut self, name: &str, dtype: Dtype, shape: &[usize]) {
        self.0.push((name.to_string(), dtype, shape.to_vec()));
    }

    /// An F8_E4M3 weight of `rows` by `k` and its f32 block scale, `<name>_scale_inv` (a weight
    /// stored with no `.weight` is named the same way).
    fn f8(&mut self, name: &str, rows: usize, k: usize) {
        self.add(name, Dtype::F8E4M3, &[rows, k]);
        self.add(&format!("{name}_scale_inv"), Dtype::F32, &grid(rows, k));
    }

    /// A routed expert projection of `rows` by `k`: I8 pairs of codes and an E8M0 scale a 32 values.
    fn mx(&mut self, name: &str, rows: usize, k: usize) {
        self.add(name, Dtype::I8, &[rows, k / 2]);
        self.add(&format!("{name}_scale_inv"), Dtype::F8E8M0, &[rows, k / 32]);
    }
}

/// The folder's tensors at the toy widths, as the released checkpoint names and shapes them, with
/// two of the draft's blocks and the FlashIndexer's gate and bias beside.
fn toy_entries() -> Entries {
    use Dtype::{Bf16, F32};
    let mut t = Toy(vec![]);
    t.add("model.embed_tokens.weight", Bf16, &[VOCAB, E]);
    t.add("lm_head.weight", Bf16, &[VOCAB, E]);
    t.add("model.norm.weight", Bf16, &[E]);
    t.add("model.lm_head.multimax_ranges", Bf16, &[4]);
    t.add("model.lm_head.multimax_ts", Bf16, &[4]);
    t.add("model.mhc_contract.hc_head_fn", F32, &[MIX_IN, 4]);
    t.add("model.mhc_contract.hc_head_base", F32, &[4]);
    t.add("model.mhc_contract.hc_head_scale", F32, &[1]);
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}.");
        let n = |rest: &str| format!("{p}{rest}");
        t.add(&n("input_layernorm.weight"), Bf16, &[E]);
        t.add(&n("post_attention_layernorm.weight"), Bf16, &[E]);
        t.add(&n("self_attn.vha_postmix_U"), Bf16, &[HEADS, RANK]);
        t.add(&n("self_attn.vha_postmix_V"), Bf16, &[HEADS, RANK]);
        for block in ["self_attention_hyper_connection", "mlp_hyper_connection"] {
            t.add(
                &n(&format!("{block}.mapping_proj.weight")),
                F32,
                &[MIX_IN, MIX_OUT],
            );
            t.add(&n(&format!("{block}.bias")), F32, &[MIX_OUT]);
            for alpha in ["alpha_pre", "alpha_post", "alpha_res"] {
                t.add(&n(&format!("{block}.{alpha}")), F32, &[1]);
            }
        }
        if dsa(l) {
            t.f8(&n("self_attn.gate_proj.weight"), HEADS * V, Q_LORA);
            t.f8(&n("self_attn.q_a_proj.weight"), Q_LORA, E);
            t.add(&n("self_attn.q_a_layernorm.weight"), Bf16, &[Q_LORA]);
            t.f8(
                &n("self_attn.q_b_proj.weight"),
                HEADS * (NOPE + ROPE),
                Q_LORA,
            );
            t.f8(&n("self_attn.kv_a_proj_with_mqa.weight"), KV_LORA + ROPE, E);
            t.add(&n("self_attn.kv_a_layernorm.weight"), Bf16, &[KV_LORA]);
            t.f8(
                &n("self_attn.kv_b_proj.weight"),
                HEADS * (NOPE + V),
                KV_LORA,
            );
            t.add(
                &n("self_attn.core_attention.softmax_offset"),
                Bf16,
                &[HEADS],
            );
            t.f8(&n("self_attn.o_proj.weight"), E, HEADS * V);
            let ix = "self_attn.core_attention.indexer.";
            t.f8(
                &n(&format!("{ix}wq_b.weight")),
                INDEX_HEADS * INDEX_DIM,
                Q_LORA,
            );
            t.add(&n(&format!("{ix}wk.weight")), Bf16, &[INDEX_DIM, E]);
            t.add(
                &n(&format!("{ix}weights_proj.weight")),
                Bf16,
                &[INDEX_HEADS, E],
            );
            t.add(&n(&format!("{ix}k_norm.weight")), Bf16, &[INDEX_DIM]);
            t.add(&n(&format!("{ix}k_norm.bias")), Bf16, &[INDEX_DIM]);
            t.add(&n(&format!("{ix}flashindexer_bias")), Bf16, &[INDEX_HEADS]);
            t.add(
                &n(&format!("{ix}linear_gate.weight")),
                Bf16,
                &[INDEX_HEADS, E],
            );
        } else {
            t.f8(&n("self_attn.gate_proj.weight"), HEADS * D, Q_LORA);
            t.f8(&n("self_attn.linear_q_down_proj.weight"), Q_LORA, E);
            t.add(&n("self_attn.q_layernorm.weight"), Bf16, &[Q_LORA]);
            t.f8(&n("self_attn.linear_q_up_proj.weight"), HEADS * D, Q_LORA);
            t.add(&n("self_attn.kv_layernorm.weight"), Bf16, &[D]);
            t.f8(&n("self_attn.linear_kv_proj.weight"), D, E);
            t.add(&n("self_attn.core_attention.attn_sink"), F32, &[HEADS]);
            t.f8(
                &n("self_attn.linear_o_group_proj"),
                O_GROUPS * O_LORA,
                HEADS * D / O_GROUPS,
            );
            t.f8(&n("self_attn.o_proj.weight"), E, O_GROUPS * O_LORA);
            let c = "self_attn.core_attention.compressor.";
            t.add(&n(&format!("{c}linear_wkv.weight")), Bf16, &[D, E]);
            t.add(&n(&format!("{c}linear_wgate.weight")), Bf16, &[D, E]);
            t.add(&n(&format!("{c}ape")), F32, &[RATIO, D]);
            t.add(&n(&format!("{c}norm.weight")), Bf16, &[D]);
        }
        if l < DENSE_LAYERS {
            t.f8(&n("mlp.gate_proj.weight"), FF, E);
            t.f8(&n("mlp.up_proj.weight"), FF, E);
            t.f8(&n("mlp.down_proj.weight"), E, FF);
        } else {
            let m = "block_sparse_moe.";
            t.add(&n(&format!("{m}gate.weight")), Bf16, &[N_EXP, E]);
            t.add(&n(&format!("{m}gate.weight_1")), Bf16, &[N_EXP, E]);
            t.add(&n(&format!("{m}e_score_correction_bias")), F32, &[N_EXP]);
            t.add(
                &n(&format!("{m}gate.routed_scaling_factor_param")),
                F32,
                &[N_EXP],
            );
            t.f8(&n(&format!("{m}fc1_latent_proj.weight")), LATENT, E);
            t.f8(&n(&format!("{m}fc2_latent_proj.weight")), E, LATENT);
            t.add(&n(&format!("{m}latent_norm.weight")), Bf16, &[LATENT]);
            t.f8(&n(&format!("{m}shared_experts.w1.weight")), EFF, E);
            t.f8(&n(&format!("{m}shared_experts.w3.weight")), EFF, E);
            t.f8(&n(&format!("{m}shared_experts.w2.weight")), E, EFF);
            for e in 0..N_EXP {
                t.mx(&n(&format!("{m}experts.{e}.w1.weight")), EFF, LATENT);
                t.mx(&n(&format!("{m}experts.{e}.w3.weight")), EFF, LATENT);
                t.mx(&n(&format!("{m}experts.{e}.w2.weight")), LATENT, EFF);
            }
        }
    }
    t.add("model.mtp.0.input_layernorm.weight", Bf16, &[E]);
    t.f8("model.mtp.0.self_attn.linear_q_down_proj.weight", Q_LORA, E);
    t.f8("model.mtp.feature_projection.weight", E, 2 * E);
    t.add("model.mtp.markov_head.markov_w1.weight", Bf16, &[8, 8]);
    t.0
}

/// A stored tensor's program stem, the stored name past `model.layers.L.` (or the whole name of a
/// model-level one), and the type the map declares it in.
type Row = (&'static str, &'static str, GgmlType);

/// Every tensor stored outside the layers.
const MODEL: &[Row] = &[
    ("model.embed_tokens.weight", "token_embd.weight", BF16),
    ("lm_head.weight", "output.weight", BF16),
    ("model.norm.weight", "output_norm.weight", F32),
    (
        "model.lm_head.multimax_ranges",
        "output_multimax_ranges.weight",
        F32,
    ),
    (
        "model.lm_head.multimax_ts",
        "output_multimax_ts.weight",
        F32,
    ),
    ("model.mhc_contract.hc_head_fn", "output_hc_fn.weight", F32),
    (
        "model.mhc_contract.hc_head_base",
        "output_hc_base.weight",
        F32,
    ),
    (
        "model.mhc_contract.hc_head_scale",
        "output_hc_scale.weight",
        F32,
    ),
];

/// What every layer holds, either attention kind.
const COMMON: &[Row] = &[
    ("input_layernorm.weight", "attn_norm.weight", F32),
    ("post_attention_layernorm.weight", "ffn_norm.weight", F32),
    ("self_attn.gate_proj.weight", "attn_gate.weight", FP8_B128),
    ("self_attn.vha_postmix_U", "attn_vha_u.weight", F32),
    ("self_attn.vha_postmix_V", "attn_vha_v.weight", F32),
    (
        "self_attention_hyper_connection.mapping_proj.weight",
        "hc_attn_fn.weight",
        F32,
    ),
    (
        "self_attention_hyper_connection.bias",
        "hc_attn_base.weight",
        F32,
    ),
    (
        "mlp_hyper_connection.mapping_proj.weight",
        "hc_ffn_fn.weight",
        F32,
    ),
    ("mlp_hyper_connection.bias", "hc_ffn_base.weight", F32),
];

/// What a compressed-attention layer holds beside [`COMMON`].
const HCA: &[Row] = &[
    (
        "self_attn.linear_q_down_proj.weight",
        "attn_q_a.weight",
        FP8_B128,
    ),
    ("self_attn.q_layernorm.weight", "attn_q_a_norm.weight", F32),
    (
        "self_attn.linear_q_up_proj.weight",
        "attn_q_b.weight",
        FP8_B128,
    ),
    (
        "self_attn.kv_layernorm.weight",
        "attn_kv_a_norm.weight",
        F32,
    ),
    (
        "self_attn.core_attention.attn_sink",
        "attn_sinks.weight",
        F32,
    ),
    (
        "self_attn.linear_kv_proj.weight",
        "attn_kv.weight",
        FP8_B128,
    ),
    (
        "self_attn.linear_o_group_proj",
        "attn_output_a.weight",
        FP8_B128,
    ),
    ("self_attn.o_proj.weight", "attn_output_b.weight", FP8_B128),
    (
        "self_attn.core_attention.compressor.linear_wkv.weight",
        "attn_compressor_kv.weight",
        BF16,
    ),
    (
        "self_attn.core_attention.compressor.linear_wgate.weight",
        "attn_compressor_gate.weight",
        BF16,
    ),
    (
        "self_attn.core_attention.compressor.ape",
        "attn_compressor_ape.weight",
        F32,
    ),
    (
        "self_attn.core_attention.compressor.norm.weight",
        "attn_compressor_norm.weight",
        F32,
    ),
];

/// What a latent-attention layer holds beside [`COMMON`].
const DSA: &[Row] = &[
    ("self_attn.q_a_proj.weight", "attn_q_a.weight", FP8_B128),
    (
        "self_attn.q_a_layernorm.weight",
        "attn_q_a_norm.weight",
        F32,
    ),
    ("self_attn.q_b_proj.weight", "attn_q_b.weight", FP8_B128),
    (
        "self_attn.kv_a_layernorm.weight",
        "attn_kv_a_norm.weight",
        F32,
    ),
    (
        "self_attn.core_attention.softmax_offset",
        "attn_sinks.weight",
        F32,
    ),
    (
        "self_attn.kv_a_proj_with_mqa.weight",
        "attn_kv_a_mqa.weight",
        FP8_B128,
    ),
    ("self_attn.kv_b_proj.weight", "attn_kv_b.weight", FP8_B128),
    ("self_attn.o_proj.weight", "attn_output.weight", FP8_B128),
    (
        "self_attn.core_attention.indexer.wq_b.weight",
        "indexer.attn_q_b.weight",
        FP8_B128,
    ),
    (
        "self_attn.core_attention.indexer.wk.weight",
        "indexer.attn_k.weight",
        BF16,
    ),
    (
        "self_attn.core_attention.indexer.weights_proj.weight",
        "indexer.proj.weight",
        BF16,
    ),
    (
        "self_attn.core_attention.indexer.k_norm.weight",
        "indexer.k_norm.weight",
        F32,
    ),
    (
        "self_attn.core_attention.indexer.k_norm.bias",
        "indexer.k_norm.bias",
        F32,
    ),
];

/// The leading dense layer's feed-forward block.
const DENSE_BLOCK: &[Row] = &[
    ("mlp.gate_proj.weight", "ffn_gate.weight", FP8_B128),
    ("mlp.up_proj.weight", "ffn_up.weight", FP8_B128),
    ("mlp.down_proj.weight", "ffn_down.weight", FP8_B128),
];

/// A routed layer's block, but its experts.
const ROUTED_BLOCK: &[Row] = &[
    ("block_sparse_moe.gate.weight", "ffn_gate_inp.weight", BF16),
    (
        "block_sparse_moe.gate.weight_1",
        "ffn_gate_inp_1.weight",
        BF16,
    ),
    (
        "block_sparse_moe.e_score_correction_bias",
        "exp_probs_b.bias",
        F32,
    ),
    (
        "block_sparse_moe.gate.routed_scaling_factor_param",
        "ffn_exp_scale.weight",
        F32,
    ),
    (
        "block_sparse_moe.fc1_latent_proj.weight",
        "ffn_routed_down.weight",
        FP8_B128,
    ),
    (
        "block_sparse_moe.fc2_latent_proj.weight",
        "ffn_routed_up.weight",
        FP8_B128,
    ),
    (
        "block_sparse_moe.latent_norm.weight",
        "ffn_routed_norm.weight",
        F32,
    ),
    (
        "block_sparse_moe.shared_experts.w1.weight",
        "ffn_gate_shexp.weight",
        FP8_B128,
    ),
    (
        "block_sparse_moe.shared_experts.w3.weight",
        "ffn_up_shexp.weight",
        FP8_B128,
    ),
    (
        "block_sparse_moe.shared_experts.w2.weight",
        "ffn_down_shexp.weight",
        FP8_B128,
    ),
];

/// The rows of layer `l`.
fn layer_rows(l: usize) -> Vec<Row> {
    let mut rows = COMMON.to_vec();
    rows.extend(if dsa(l) { DSA } else { HCA });
    rows.extend(if l < DENSE_LAYERS {
        DENSE_BLOCK
    } else {
        ROUTED_BLOCK
    });
    rows
}

fn plan(
    config: &Json,
    entries: &Entries,
    q8: bool,
) -> Result<folder::Planning<Step>, folder::Error> {
    let map = Checkpoint::parse(config)?;
    folder::plan_tensors(
        &map,
        &Config::parse(config, None).unwrap(),
        entries,
        q8,
        Keep::InPlace,
    )
}

fn refusal(config: &Json, entries: &Entries) -> String {
    match plan(config, entries, false) {
        Ok(_) => panic!("planned, where a refusal was expected"),
        Err(e) => e.to_string(),
    }
}

/// Each planned tensor's ggml dims and type, by name.
fn table(p: &folder::Planning<Step>) -> BTreeMap<String, (Vec<u64>, GgmlType)> {
    p.planned
        .iter()
        .map(|t| {
            (
                t.decl.name.clone(),
                (t.decl.dims.clone(), GgmlType::from_u32(t.decl.type_id)),
            )
        })
        .collect()
}

/// Each planned tensor's mapping by name.
fn mapped(p: &folder::Planning<Step>) -> BTreeMap<&str, &Mapped<Step>> {
    p.planned
        .iter()
        .filter_map(|t| match &t.make {
            Make::Tensor(m) => Some((t.decl.name.as_str(), m)),
            _ => None,
        })
        .collect()
}

/// The program stem of a planned name: the name past `blk.N.`.
fn stem_of(name: &str) -> &str {
    name.strip_prefix("blk.")
        .and_then(|r| r.split_once('.'))
        .map_or(name, |(_, stem)| stem)
}

/// The stored shape of the tensor `name` in `entries`.
fn shape_in(entries: &Entries, name: &str) -> Vec<usize> {
    entries
        .iter()
        .find(|(n, _, _)| n == name)
        .unwrap_or_else(|| panic!("{name} is not in the toy folder"))
        .2
        .clone()
}

/// ggml's dims of a row-major `shape`, but a mixer's: it is stored `[in, out]` and written
/// `[out, in]`, so its ggml dims are the stored shape as it is.
fn written_dims(stem: &str, shape: &[usize]) -> Vec<u64> {
    if stem.ends_with("_fn.weight") {
        shape.iter().map(|&d| d as u64).collect()
    } else {
        shape.iter().rev().map(|&d| d as u64).collect()
    }
}

/// The real config reads into the header keys the model description's reader holds to: 43 layers
/// whose latent-attention ones (every 7th from the 8th) are written ratio 0 beside the pooling
/// ratio 128 of the rest, the first layer dense, 512 experts of 2048 in a latent of 2048 with 10
/// used, the head widths of both attention kinds, YaRN's keys and the hyper-connections'.
/// Mutants: `compress_ratios` written with `-2` as a ratio, a key of the table renamed.
#[test]
fn the_real_config_reads_into_the_header_keys() {
    let config: Json = serde_json::from_str(REAL_CONFIG).unwrap();
    Config::parse(&config, None).unwrap();
    let map = Checkpoint::parse(&config).unwrap_or_else(|e| panic!("{e}"));
    let ratios = (0..43)
        .map(|l| Value::U32(if l % 7 == 0 && l > 0 { 0 } else { 128 }))
        .collect();
    let u = |k: &str, v: u32| (k.to_string(), Value::U32(v));
    let f = |k: &str, v: f64| (k.to_string(), Value::F32(v as f32));
    let s = |k: &str, v: &str| (k.to_string(), Value::String(v.into()));
    let want = [
        u("block_count", 43),
        u("context_length", 1_048_576),
        u("embedding_length", 4096),
        u("vocab_size", 201_216),
        u("feed_forward_length", 14_336),
        u("leading_dense_block_count", 1),
        u("attention.head_count", 64),
        f("attention.layer_norm_rms_epsilon", 1e-6),
        (
            "attention.compress_ratios".to_string(),
            Value::Array(ratios),
        ),
        u("attention.sliding_window", 128),
        u("attention.q_lora_rank", 1024),
        u("attention.kv_lora_rank", 512),
        u("attention.key_length", 576),
        u("attention.key_length_mla", 256),
        u("attention.value_length_mla", 256),
        u("rope.dimension_count", 64),
        u("attention.hca.head_length", 512),
        u("attention.output_group_count", 8),
        u("attention.output_lora_rank", 1024),
        u("attention.indexer.head_count", 64),
        u("attention.indexer.key_length", 128),
        u("attention.indexer.top_k", 2048),
        f("rope.freq_base", 8_000_000.0),
        f("attention.compress_rope_freq_base", 160_000.0),
        s("rope.scaling.type", "yarn"),
        f("rope.scaling.factor", 16.0),
        u("rope.scaling.original_context_length", 65_536),
        f("rope.scaling.yarn_beta_fast", 32.0),
        f("rope.scaling.yarn_beta_slow", 1.0),
        u("expert_count", 512),
        u("expert_used_count", 10),
        u("expert_feed_forward_length", 2048),
        u("expert_shared_count", 1),
        u("expert_latent_length", 2048),
        f("expert_weights_scale", 3.0),
        ("expert_weights_norm".to_string(), Value::Bool(true)),
        u("expert_gating_func", 2),
        u("expert_group_count", 1),
        u("expert_group_used_count", 1),
        f("activation.situ_beta", 4.0),
        f("activation.situ_linear_beta", 25.0),
        u("hyper_connection.count", 4),
        u("hyper_connection.sinkhorn_iterations", 20),
        f("hyper_connection.epsilon", 1e-6),
    ];
    for (key, value) in &want {
        let got = map.keys.iter().find(|(k, _)| k == key);
        assert_eq!(got.map(|(_, v)| v), Some(value), "{key}");
    }
    assert_eq!(map.keys.len(), want.len(), "a key the table does not name");
    assert_eq!(ARCH, "ernie5_5");
    assert_eq!(map.architecture(), ARCH);
}

/// A toy folder plans from its headers alone: every tensor at the name the table gives it, at
/// ggml dims and in the type the placed format and role give (the FP8 weights 128-blocks, a bare
/// `linear_o_group_proj` among them; the routers, compressor and indexer matrices BF16; every
/// norm, vector and mixer F32; the routed stacks MXFP4 `[k, rows, experts]`), one source each but
/// a mixer's scalars and a stack's experts, and nothing left unplaced. A layer holds the tensors
/// of its kind: 26, 36 and 37 beside the head's 8. Mutants: a stem of the table mapped from the
/// wrong stored name (gate and up swapped, the HCA output pair crossed), a kind's tensor
/// dropped.
#[test]
fn a_toy_folder_plans_every_tensor_at_the_released_names_dims_and_types() {
    let entries = toy_entries();
    let p = plan(&toy_config(), &entries, false).unwrap_or_else(|e| panic!("{e}"));
    assert!(p.plan.refused.is_empty(), "{:?}", p.plan.refused);
    let (got, sources) = (table(&p), mapped(&p));
    let mut want: BTreeMap<String, (Vec<u64>, GgmlType)> = BTreeMap::new();
    let mut from: BTreeMap<String, String> = BTreeMap::new();
    for &(stored, stem, ty) in MODEL {
        want.insert(
            stem.to_string(),
            (written_dims(stem, &shape_in(&entries, stored)), ty),
        );
        from.insert(stem.to_string(), stored.to_string());
    }
    for l in 0..LAYERS {
        for (tail, stem, ty) in layer_rows(l) {
            let stored = format!("model.layers.{l}.{tail}");
            want.insert(
                format!("blk.{l}.{stem}"),
                (written_dims(stem, &shape_in(&entries, &stored)), ty),
            );
            from.insert(format!("blk.{l}.{stem}"), stored);
        }
        for stem in ["hc_attn_scale.weight", "hc_ffn_scale.weight"] {
            want.insert(format!("blk.{l}.{stem}"), (vec![3], F32));
        }
        if l >= DENSE_LAYERS {
            let stack = |stem: &str, k: usize, rows: usize| {
                (
                    format!("blk.{l}.{stem}"),
                    (vec![k as u64, rows as u64, N_EXP as u64], MXFP4),
                )
            };
            want.extend([
                stack("ffn_gate_exps.weight", LATENT, EFF),
                stack("ffn_up_exps.weight", LATENT, EFF),
                stack("ffn_down_exps.weight", EFF, LATENT),
            ]);
        }
    }
    assert_eq!(want.len(), 8 + 26 + 36 + 37, "the table's own count");
    for (name, w) in &want {
        assert_eq!(got.get(name), Some(w), "{name}");
    }
    assert_eq!(got.len(), want.len(), "a tensor the table does not name");
    for (name, stored) in &from {
        assert_eq!(
            sources[name.as_str()].sources,
            std::slice::from_ref(stored),
            "{name}"
        );
    }
}

/// The map and the header table name the same stems: every planned stem has a list in
/// `gguf::arch::ernie55`, the planned type is in it with the Q8_0 rewrite asked or not, and the
/// table holds no stem the map never makes. Mutant: a stem renamed on one side.
#[test]
fn the_map_and_the_header_table_name_the_same_stems() {
    let entries = toy_entries();
    let mut stems = BTreeSet::new();
    for q8 in [false, true] {
        let p = plan(&toy_config(), &entries, q8).unwrap_or_else(|e| panic!("{e}"));
        for (name, (_, ty)) in table(&p) {
            let stem = stem_of(&name);
            let allowed =
                allowed_types(stem).unwrap_or_else(|| panic!("{name}: no list for {stem}"));
            assert!(
                allowed.contains(&ty),
                "{name} (q8 {q8}): {ty:?} not in {allowed:?}"
            );
            stems.insert(stem.to_string());
        }
    }
    let listed: BTreeSet<String> = STEM_TYPES.iter().map(|(s, _)| s.to_string()).collect();
    assert_eq!(stems, listed);
}

/// A mixer is stored `[in, out]` and written `[out, in]`: its planned tensor carries the transpose
/// and the stored shape as its ggml dims, the model-level head mixer the same, and the three
/// scalars of a block are the sources of its one scale vector, in the order `alpha_pre`,
/// `alpha_post`, `alpha_res`. The transpose writes the `[out, in]` rows of an `[in, out]` matrix
/// and refuses what is not an F32 matrix of its bytes. Mutants: the step left off, the scalars
/// in another order.
#[test]
fn a_mixer_is_written_out_in_and_its_scalars_make_one_scale() {
    let p = plan(&toy_config(), &toy_entries(), false).unwrap_or_else(|e| panic!("{e}"));
    let m = mapped(&p);
    for name in [
        "blk.0.hc_attn_fn.weight",
        "blk.2.hc_ffn_fn.weight",
        "output_hc_fn.weight",
    ] {
        assert_eq!(m[name].steps, [Step::Transpose], "{name}");
    }
    assert_eq!(
        m["blk.1.hc_attn_fn.weight"].dims,
        [MIX_IN as u64, MIX_OUT as u64]
    );
    assert_eq!(m["output_hc_fn.weight"].dims, [MIX_IN as u64, 4]);
    assert!(m["blk.1.hc_attn_base.weight"].steps.is_empty());
    for (stem, block) in [
        ("hc_attn_scale.weight", "self_attention_hyper_connection"),
        ("hc_ffn_scale.weight", "mlp_hyper_connection"),
    ] {
        let scale = m[format!("blk.2.{stem}").as_str()];
        let alphas: Vec<String> = ["alpha_pre", "alpha_post", "alpha_res"]
            .iter()
            .map(|a| format!("model.layers.2.{block}.{a}"))
            .collect();
        assert_eq!(scale.sources, alphas, "{stem}");
        assert_eq!(scale.dims, [3]);
    }
    let mut rows = Normal {
        type_id: GgmlType::F32.as_u32(),
        dims: vec![3, 2],
        bytes: [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        scale: vec![],
    };
    Step::Transpose.apply(&mut rows).unwrap();
    let values: Vec<f32> = rows
        .bytes
        .chunks(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(
        (rows.dims.as_slice(), values),
        (&[2u64, 3][..], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    );
    let mut not_f32 = Normal {
        type_id: GgmlType::BF16.as_u32(),
        dims: vec![3, 2],
        bytes: vec![0; 12],
        scale: vec![],
    };
    let e = Step::Transpose.apply(&mut not_f32).unwrap_err();
    assert!(e.contains("not type"), "{e}");
    let mut vector = Normal {
        type_id: GgmlType::F32.as_u32(),
        dims: vec![6],
        bytes: vec![0; 24],
        scale: vec![],
    };
    let e = Step::Transpose.apply(&mut vector).unwrap_err();
    assert!(e.contains("reads a matrix"), "{e}");
}

/// A layer's routed experts stack by expert id into one tensor a projection: `w1` is the gate,
/// `w3` the up and `w2` the down projection, the sources in expert order, `[k, rows, experts]`.
/// Mutants: `w1` and `w3` swapped, the sources sorted by name.
#[test]
fn the_routed_stacks_stack_by_expert_id() {
    let p = plan(&toy_config(), &toy_entries(), false).unwrap_or_else(|e| panic!("{e}"));
    let m = mapped(&p);
    for l in [1, 2] {
        for (stem, w, dims) in [
            ("ffn_gate_exps.weight", "w1", [LATENT, EFF, N_EXP]),
            ("ffn_up_exps.weight", "w3", [LATENT, EFF, N_EXP]),
            ("ffn_down_exps.weight", "w2", [EFF, LATENT, N_EXP]),
        ] {
            let stack = m[format!("blk.{l}.{stem}").as_str()];
            let sources: Vec<String> = (0..N_EXP)
                .map(|e| format!("model.layers.{l}.block_sparse_moe.experts.{e}.{w}.weight"))
                .collect();
            assert_eq!(stack.sources, sources, "{stem}");
            assert_eq!(stack.dims, dims.map(|d| d as u64), "{stem}");
        }
    }
    assert!(!m.contains_key("blk.0.ffn_gate_exps.weight"));
}

/// The draft's tensors (`model.mtp.*`) and FlashIndexer's gate and bias are skipped with the
/// reason no reader takes them yet, and the trunk's tensors are not: the plan skips exactly the
/// toy's six names. Mutant: the draft's prefix left out of the skip.
#[test]
fn the_draft_and_flashindexer_are_skipped_by_reason() {
    let p = plan(&toy_config(), &toy_entries(), false).unwrap_or_else(|e| panic!("{e}"));
    let draft = [
        "model.mtp.0.input_layernorm.weight",
        "model.mtp.0.self_attn.linear_q_down_proj.weight",
        "model.mtp.feature_projection.weight",
        "model.mtp.markov_head.markov_w1.weight",
    ];
    let flash = [
        "model.layers.2.self_attn.core_attention.indexer.flashindexer_bias",
        "model.layers.2.self_attn.core_attention.indexer.linear_gate.weight",
    ];
    let skipped: Vec<&str> = p.skipped.keys().map(String::as_str).collect();
    let mut want: Vec<&str> = draft.iter().chain(&flash).copied().collect();
    want.sort_unstable();
    assert_eq!(skipped, want);
    for name in draft {
        assert!(p.skipped[name].contains("DSpark draft"), "{name}");
    }
    for name in flash {
        assert!(p.skipped[name].contains("FlashIndexer"), "{name}");
    }
}

/// The head mix is the pair `U` and `V` of `[heads, rank]` rows in every layer, and the header
/// states its one rank; a layer of another rank or another head count is refused by name. Mutant:
/// the rank read from one layer alone.
#[test]
fn the_head_mix_rank_is_the_stored_one() {
    let map = Checkpoint::parse(&toy_config()).unwrap();
    let p = plan(&toy_config(), &toy_entries(), false).unwrap();
    let planned = |edit: &dyn Fn(&str, &mut Vec<u64>)| -> Vec<(String, Vec<u64>)> {
        p.planned
            .iter()
            .map(|t| {
                let mut dims = t.decl.dims.clone();
                edit(&t.decl.name, &mut dims);
                (t.decl.name.clone(), dims)
            })
            .collect()
    };
    let rank = |list: &[(String, Vec<u64>)]| {
        map.vha_rank(
            &list
                .iter()
                .map(|(n, d)| (n.as_str(), d.as_slice()))
                .collect::<Vec<_>>(),
        )
    };
    assert_eq!(rank(&planned(&|_, _| {})).unwrap(), RANK as u32);
    let e = rank(&planned(&|n, d| {
        if n == "blk.2.attn_vha_u.weight" {
            d[0] += 1;
        }
    }))
    .unwrap_err()
    .to_string();
    assert!(
        e.contains("blk.2.attn_vha_u.weight: dims [17, 2], not [16, 2]"),
        "{e}"
    );
    let e = rank(&planned(&|n, d| {
        if n == "blk.1.attn_vha_v.weight" {
            d[1] += 1;
        }
    }))
    .unwrap_err()
    .to_string();
    assert!(
        e.contains("blk.1.attn_vha_v.weight: dims [16, 3], not [16, 2]"),
        "{e}"
    );
    let e = rank(&[]).unwrap_err().to_string();
    assert!(e.contains("no layer holds the head mix"), "{e}");
}

/// A config of another shape than the program runs is refused by key: another architecture or
/// `model_type`, a value the program fixes, a key the map does not know, a value of the wrong
/// type for an ignored key, widths that disagree, and a `compress_ratios` list that is not a
/// ratio or `-2` per layer. Mutant: a `Need` rule relaxed to `Read`.
#[test]
fn a_config_of_another_shape_is_refused_by_key() {
    type Case = (&'static str, Box<dyn Fn(&mut Json)>, &'static str);
    let cases: Vec<Case> = vec![
        (
            "another architecture",
            Box::new(|c| c["architectures"] = json!(["Ernie4_5ForCausalLM"])),
            "architectures",
        ),
        (
            "another model_type",
            Box::new(|c| c["model_type"] = json!("ernie5_6_moe")),
            "model_type",
        ),
        (
            "another activation",
            Box::new(|c| c["hidden_act"] = json!("silu")),
            "hidden_act",
        ),
        (
            "a tied head",
            Box::new(|c| c["tie_word_embeddings"] = json!(true)),
            "tie_word_embeddings",
        ),
        (
            "two key-value heads",
            Box::new(|c| c["num_key_value_heads"] = json!(2)),
            "num_key_value_heads",
        ),
        (
            "an attention bias",
            Box::new(|c| c["attention_bias"] = json!(true)),
            "attention_bias",
        ),
        (
            "an ungated attention",
            Box::new(|c| c["use_gated_attn"] = json!(false)),
            "use_gated_attn",
        ),
        (
            "a softmax router",
            Box::new(|c| c["scoring_func"] = json!("softmax")),
            "scoring_func",
        ),
        (
            "a fixed routed scale",
            Box::new(|c| c["routed_scaling_factor_learnable"] = json!(false)),
            "routed_scaling_factor_learnable",
        ),
        (
            "a next-token block",
            Box::new(|c| c["num_nextn_predict_layers"] = json!(1)),
            "num_nextn_predict_layers",
        ),
        (
            "another rope",
            Box::new(|c| c["rope_scaling"]["type"] = json!("linear")),
            "rope_scaling",
        ),
        (
            "a rope scaling with no factor",
            Box::new(|c| {
                c["rope_scaling"].as_object_mut().unwrap().remove("factor");
            }),
            "factor",
        ),
        (
            "a key no map reads",
            Box::new(|c| c["moe_layer_pattern"] = json!([1])),
            "moe_layer_pattern",
        ),
        (
            "an ignored key of another type",
            Box::new(|c| c["use_fast_hadamard"] = json!("yes")),
            "use_fast_hadamard",
        ),
        (
            "a rotary base off the rope's",
            Box::new(|c| c["rotary_base"] = json!(10_000)),
            "rotary_base",
        ),
        (
            "a value width off the head's",
            Box::new(|c| c["v_head_dim"] = json!(D + 1)),
            "v_head_dim",
        ),
        (
            "a nope width off the head's",
            Box::new(|c| c["qk_nope_head_dim"] = json!(D)),
            "qk_nope_head_dim",
        ),
        (
            "latent-attention heads off the heads",
            Box::new(|c| c["hybrid_mla_num_attention_heads"] = json!(HEADS + 1)),
            "hybrid_mla_num_attention_heads",
        ),
        (
            "a latent rank off the other kind's",
            Box::new(|c| c["hybrid_mla_kv_lora_rank"] = json!(KV_LORA + 128)),
            "hybrid_mla_kv_lora_rank",
        ),
        (
            "output groups that do not divide the heads",
            Box::new(|c| c["o_groups"] = json!(3)),
            "o_groups",
        ),
        (
            "more experts used than there are",
            Box::new(|c| c["num_experts_per_tok"] = json!(N_EXP + 1)),
            "num_experts_per_tok",
        ),
        (
            "more dense layers than layers",
            Box::new(|c| c["first_k_dense_replace"] = json!(LAYERS + 1)),
            "first_k_dense_replace",
        ),
        (
            "a ratio list of another length",
            Box::new(|c| c["compress_ratios"] = json!([RATIO, -2])),
            "compress_ratios",
        ),
        (
            "a ratio of zero",
            Box::new(|c| c["compress_ratios"] = json!([RATIO, 0, -2])),
            "compress_ratios",
        ),
        (
            "a ratio of -1",
            Box::new(|c| c["compress_ratios"] = json!([RATIO, -1, -2])),
            "compress_ratios",
        ),
        (
            "a draft layer past the trunk",
            Box::new(|c| c["dspark_target_layer_ids"] = json!([0, LAYERS])),
            "dspark_target_layer_ids",
        ),
        (
            "no draft layer",
            Box::new(|c| c["dspark_target_layer_ids"] = json!([])),
            "dspark_target_layer_ids",
        ),
        (
            "a layer count that is not a count",
            Box::new(|c| c["num_hidden_layers"] = json!(0)),
            "num_hidden_layers",
        ),
    ];
    for (what, edit, key) in cases {
        let mut c = toy_config();
        edit(&mut c);
        let e = Checkpoint::parse(&c).unwrap_err().to_string();
        assert!(e.contains(key), "{what}: {e}");
    }
}

/// An F8 weight `name` of `rows` by `k` and its scale, as entries.
fn f8_entries(name: &str, rows: usize, k: usize) -> Entries {
    let mut t = Toy(vec![]);
    t.f8(name, rows, k);
    t.0
}

/// `entries` with the tensors `edit` leaves.
fn entries_with(edit: impl FnOnce(&mut Entries)) -> Entries {
    let mut e = toy_entries();
    edit(&mut e);
    e
}

/// A folder of another shape than the config's is refused by the tensor's name: a tensor the
/// layer's kind does not hold (or lacks), a head or embedding of another width, a router or a
/// scalar of a mixer missing or misshapen, a short or misshapen expert stack, a name no ERNIE 5.5
/// folder has, and F8 weights whose scales are missing or of another grid. Mutant: a kind's
/// tensor accepted on the other kind's layer.
#[test]
fn a_folder_of_another_shape_is_refused_by_tensor() {
    use Dtype::{Bf16, F32, I8};
    let layer = |l: usize, rest: &str| format!("model.layers.{l}.{rest}");
    let without = |names: &[String]| entries_with(|e| e.retain(|(n, _, _)| !names.contains(n)));
    let reshaped = |name: String, shape: Vec<usize>| {
        entries_with(|e| e.iter_mut().find(|(n, _, _)| *n == name).unwrap().2 = shape)
    };
    let with = |tensors: Vec<(String, Dtype, Vec<usize>)>| entries_with(|e| e.extend(tensors));
    let weight_and_scale = |name: String| [name.clone(), format!("{name}_scale_inv")];
    let cases: Vec<(&str, Entries, &str)> = vec![
        (
            "a compressed-attention layer without its compressor's key-value matrix",
            without(&[layer(
                1,
                "self_attn.core_attention.compressor.linear_wkv.weight",
            )]),
            "blk.1.attn_compressor_kv.weight: is missing from the folder",
        ),
        (
            "a latent-attention layer without its kv_b",
            without(&weight_and_scale(layer(2, "self_attn.kv_b_proj.weight"))),
            "blk.2.attn_kv_b.weight: is missing from the folder",
        ),
        (
            "a latent-attention layer without its indexer's key",
            without(&[layer(2, "self_attn.core_attention.indexer.wk.weight")]),
            "blk.2.indexer.attn_k.weight: is missing from the folder",
        ),
        (
            "a routed layer without a router",
            without(&[layer(1, "block_sparse_moe.gate.weight_1")]),
            "blk.1.ffn_gate_inp_1.weight: is missing from the folder",
        ),
        (
            "a routed layer without its shared expert's down projection",
            without(&weight_and_scale(layer(
                2,
                "block_sparse_moe.shared_experts.w2.weight",
            ))),
            "blk.2.ffn_down_shexp.weight: is missing from the folder",
        ),
        (
            "a compressed-attention tensor on the latent-attention layer",
            with(f8_entries(
                &layer(2, "self_attn.linear_kv_proj.weight"),
                D,
                E,
            )),
            "layers.2.self_attn.linear_kv_proj.weight: not an ERNIE 5.5 tensor of a Dsa layer",
        ),
        (
            "a latent-attention tensor on a compressed-attention layer",
            with(f8_entries(
                &layer(1, "self_attn.kv_b_proj.weight"),
                HEADS * (NOPE + V),
                KV_LORA,
            )),
            "layers.1.self_attn.kv_b_proj.weight: not an ERNIE 5.5 tensor of a Hca layer",
        ),
        (
            "a dense block on a routed layer",
            with(f8_entries(&layer(1, "mlp.gate_proj.weight"), FF, E)),
            "blk.1.ffn_gate.weight: is a tensor of no Some(Hca) layer 1 of 3",
        ),
        (
            "an embedding of another width",
            reshaped("model.embed_tokens.weight".into(), vec![VOCAB, E / 2]),
            "model.embed_tokens.weight: dims [128, 8], not hidden 256 by vocabulary 8",
        ),
        (
            "a head of another vocabulary",
            reshaped("lm_head.weight".into(), vec![VOCAB + 1, E]),
            "lm_head.weight: dims [256, 9], not hidden 256 by vocabulary 8",
        ),
        (
            "a mixer scalar missing",
            without(&[layer(1, "mlp_hyper_connection.alpha_res")]),
            "model.layers.1.hc_ffn_scale.weight alpha_res: is missing from the folder",
        ),
        (
            "a mixer scalar that is a vector",
            reshaped(
                layer(1, "self_attention_hyper_connection.alpha_pre"),
                vec![2],
            ),
            "alpha_pre: shape [2], not one scalar",
        ),
        (
            "a routed stack short of an expert",
            without(&weight_and_scale(layer(
                1,
                "block_sparse_moe.experts.3.w2.weight",
            ))),
            "the stack's 3 expert ids are not 0..4",
        ),
        (
            "a routed projection of another shape",
            entries_with(|e| {
                for (n, _, s) in e.iter_mut() {
                    if n == &layer(2, "block_sparse_moe.experts.2.w1.weight") {
                        *s = vec![EFF, LATENT / 2 + 16];
                    }
                    if n == &layer(2, "block_sparse_moe.experts.2.w1.weight_scale_inv") {
                        *s = vec![EFF, LATENT / 32 + 1];
                    }
                }
            }),
            "shape differs from",
        ),
        (
            "an expert projection that is not w1, w2 or w3",
            with(vec![
                (
                    layer(1, "block_sparse_moe.experts.0.w4.weight"),
                    I8,
                    vec![EFF, LATENT / 2],
                ),
                (
                    layer(1, "block_sparse_moe.experts.0.w4.weight_scale_inv"),
                    Dtype::F8E8M0,
                    vec![EFF, LATENT / 32],
                ),
            ]),
            "not an ERNIE 5.5 expert projection",
        ),
        (
            "a layer past the config's",
            with(vec![(
                layer(LAYERS, "input_layernorm.weight"),
                Bf16,
                vec![E],
            )]),
            "layer 3 of 3",
        ),
        (
            "a name no ERNIE 5.5 folder has",
            with(vec![("model.audio_tower.weight".into(), Bf16, vec![8, 8])]),
            "model.audio_tower.weight: not an ERNIE 5.5 tensor",
        ),
        (
            "an F8 weight with no scale",
            without(&[layer(0, "self_attn.linear_q_down_proj.weight_scale_inv")]),
            "linear_q_down_proj.weight: F8_E4M3 [384, 256] with [] fits none of",
        ),
        (
            "an F8 weight whose scale is another grid",
            reshaped(
                layer(0, "self_attn.linear_q_down_proj.weight_scale_inv"),
                vec![1, 1],
            ),
            "linear_q_down_proj.weight",
        ),
        (
            "a bare F8 weight with no scale",
            without(&[layer(1, "self_attn.linear_o_group_proj_scale_inv")]),
            "F8_E4M3 outside a quantized weight",
        ),
        (
            "a scale beside a weight that is no F8",
            entries_with(|e| {
                let at = e
                    .iter()
                    .position(|(n, _, _)| *n == layer(0, "mlp.up_proj.weight"))
                    .unwrap();
                e[at].1 = F32;
            }),
            "mlp.up_proj.weight_scale_inv",
        ),
    ];
    for (what, entries, want) in cases {
        let e = refusal(&toy_config(), &entries);
        assert!(e.contains(want), "{what}: {e}");
    }
}

/// A quantization config that is not ERNIE's is refused by name: expert codes of another name, and
/// a block size the folder's scales do not cover. Mutant: `expert_dtype` read as `fp4` alone.
#[test]
fn a_quantization_that_is_not_ernies_is_refused() {
    let mut nv = toy_config();
    nv["quantization_config"]["expert_dtype"] = json!("nvfp4");
    let e = Config::parse(&nv, None)
        .map(|_| ())
        .unwrap_err()
        .to_string();
    assert!(e.contains("expert_dtype Some(\"nvfp4\")"), "{e}");
    let mut b64 = toy_config();
    b64["quantization_config"]["weight_block_size"] = json!([64, 64]);
    let e = plan(&b64, &toy_entries(), false)
        .err()
        .expect("scales of 128 blocks over 64")
        .to_string();
    assert!(e.contains("weight_scale_inv") || e.contains("scale"), "{e}");
}

/// The real folder, its 128 shards' headers read and no tensor byte, plans with no refusal: every
/// planned tensor of the 55 stems in the count and bytes the headers derive by the checkpoint's
/// own formats (1,552 tensors, 155,069,470,588 bytes), the draft's 4,716 tensors skipped, each of
/// the 477 FP8 weights beside its f32 scale a 128 by 128 block (`<name>.weight_scale_inv`, or
/// `<name>_scale_inv` for the bare grouped output projection), each of the 126 routed stacks
/// MXFP4 `[2048, 2048, 512]` of 512 experts' I8 codes and E8M0 scales a 32 values, and the header
/// states the head mix's rank. Prints each role's line.
#[test]
#[ignore = "needs the ERNIE-5.5-Flash-Preview folder at /models/ERNIE-5.5-Flash-Preview"]
fn hw_the_ernie55_folder_plans_as_the_map_reads_it() {
    let dir = Path::new("/models/ERNIE-5.5-Flash-Preview");
    let config: Json =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let folder = Folder::open(dir).unwrap_or_else(|e| panic!("{e}"));
    let entries: Entries = folder
        .names()
        .map(|n| {
            let raw = folder.raw(n).unwrap_or_else(|e| panic!("{n}: {e}"));
            (n.to_string(), raw.dtype, raw.shape.to_vec())
        })
        .collect();
    assert_eq!(entries.len(), 140_457);
    let map = Checkpoint::parse(&config).unwrap_or_else(|e| panic!("{e}"));
    let p = folder::plan_tensors(
        &map,
        &Config::read(dir).unwrap(),
        &entries,
        false,
        Keep::InPlace,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    assert!(p.plan.refused.is_empty(), "{:?}", p.plan.refused);
    // The index's tensors, less the 64,989 trunk and 4,642 draft companions.
    assert_eq!(p.plan.placed.len(), 140_457 - 64_989 - 4_642);
    let draft: Vec<&String> = p.skipped.keys().collect();
    assert_eq!(draft.len(), 4_716);
    assert!(draft.iter().all(|n| n.starts_with("model.mtp.")));
    let mut roles: BTreeMap<&str, (usize, u64)> = BTreeMap::new();
    for t in &p.planned {
        let r = roles.entry(stem_of(&t.decl.name)).or_default();
        r.0 += 1;
        r.1 += t.decl.nbytes;
    }
    for (stem, count, bytes) in ROLES {
        assert_eq!(roles.get(stem), Some(&(count, bytes)), "{stem}");
        println!("ernie55 role {stem}: {count} tensors, {bytes} bytes");
    }
    assert_eq!(roles.len(), ROLES.len());
    assert_eq!(p.planned.len(), 1_552);
    assert_eq!(
        p.planned.iter().map(|t| t.decl.nbytes).sum::<u64>(),
        155_069_470_588
    );
    let stored: BTreeMap<&str, (Dtype, &[usize])> = entries
        .iter()
        .map(|(n, d, s)| (n.as_str(), (*d, s.as_slice())))
        .collect();
    let (mut blocks, mut experts, mut bare) = (0, 0, 0);
    for (name, placed) in p
        .plan
        .placed
        .iter()
        .filter(|(n, _)| !n.starts_with("model.mtp."))
    {
        let (_, shape) = stored[name.as_str()];
        match placed.format {
            Format::Fp8Block {
                rows: 128,
                cols: 128,
            } => {
                blocks += 1;
                bare += usize::from(!name.ends_with(".weight"));
                assert_eq!(placed.companions, [format!("{name}_scale_inv")], "{name}");
                let (dtype, grid_shape) = stored[placed.companions[0].as_str()];
                assert_eq!(dtype, Dtype::F32, "{name}");
                assert_eq!(grid_shape, grid(shape[0], shape[1]), "{name}");
            }
            Format::Mxfp4 => {
                experts += 1;
                assert_eq!(placed.companions, [format!("{name}_scale_inv")], "{name}");
                let (dtype, scale_shape) = stored[placed.companions[0].as_str()];
                assert_eq!(dtype, Dtype::F8E8M0, "{name}");
                assert_eq!(scale_shape, [shape[0], shape[1] / 16], "{name}");
            }
            _ => {}
        }
    }
    assert_eq!((blocks, bare, experts), (477, 37, 512 * 3 * 42));
    println!(
        "ernie55 dense: {blocks} FP8 128-block weights, {bare} with the bare <name>_scale_inv; experts: {experts} MXFP4 projections"
    );
    let table = table(&p);
    for stem in [
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
    ] {
        for l in 1..43 {
            assert_eq!(
                table[&format!("blk.{l}.{stem}")],
                (vec![2048, 2048, 512], MXFP4),
                "{stem} {l}"
            );
        }
    }
    println!(
        "ernie55 expert stack: blk.1.ffn_gate_exps.weight {:?}",
        table["blk.1.ffn_gate_exps.weight"]
    );
    let header = map
        .header(
            &p.planned
                .iter()
                .map(|t| (t.decl.name.as_str(), t.decl.dims.as_slice()))
                .collect::<Vec<_>>(),
            &folder,
        )
        .unwrap_or_else(|e| panic!("{e}"));
    let rank = header
        .iter()
        .find(|(k, _)| k == "ernie5_5.attention.vha.rank");
    assert_eq!(rank.map(|(_, v)| v), Some(&Value::U32(16)));
    println!(
        "ernie55 header: {} keys, vha rank 16, {} tensors planned, {} skipped",
        header.len(),
        p.planned.len(),
        p.skipped.len()
    );
}

const ROLES: [(&str, usize, u64); 55] = [
    ("attn_compressor_ape.weight", 37, 9699328),
    ("attn_compressor_gate.weight", 37, 155189248),
    ("attn_compressor_kv.weight", 37, 155189248),
    ("attn_compressor_norm.weight", 37, 75776),
    ("attn_gate.weight", 43, 423886848),
    ("attn_kv.weight", 37, 80019456),
    ("attn_kv_a_mqa.weight", 6, 14598144),
    ("attn_kv_a_norm.weight", 43, 88064),
    ("attn_kv_b.weight", 6, 90832896),
    ("attn_norm.weight", 43, 704512),
    ("attn_output.weight", 6, 415236096),
    ("attn_output_a.weight", 37, 1280311296),
    ("attn_output_b.weight", 37, 1280311296),
    ("attn_q_a.weight", 43, 185991168),
    ("attn_q_a_norm.weight", 43, 176128),
    ("attn_q_b.weight", 43, 1384120320),
    ("attn_sinks.weight", 43, 11008),
    ("attn_vha_u.weight", 43, 176128),
    ("attn_vha_v.weight", 43, 176128),
    ("exp_probs_b.bias", 42, 86016),
    ("ffn_down.weight", 1, 60555264),
    ("ffn_down_exps.weight", 42, 47915728896),
    ("ffn_down_shexp.weight", 42, 363331584),
    ("ffn_exp_scale.weight", 42, 86016),
    ("ffn_gate.weight", 1, 60555264),
    ("ffn_gate_exps.weight", 42, 47915728896),
    ("ffn_gate_inp.weight", 42, 176160768),
    ("ffn_gate_inp_1.weight", 42, 176160768),
    ("ffn_gate_shexp.weight", 42, 363331584),
    ("ffn_norm.weight", 43, 704512),
    ("ffn_routed_down.weight", 42, 363331584),
    ("ffn_routed_norm.weight", 42, 344064),
    ("ffn_routed_up.weight", 42, 363331584),
    ("ffn_up.weight", 1, 60555264),
    ("ffn_up_exps.weight", 42, 47915728896),
    ("ffn_up_shexp.weight", 42, 363331584),
    ("hc_attn_base.weight", 43, 4128),
    ("hc_attn_fn.weight", 43, 67633152),
    ("hc_attn_scale.weight", 43, 516),
    ("hc_ffn_base.weight", 43, 4128),
    ("hc_ffn_fn.weight", 43, 67633152),
    ("hc_ffn_scale.weight", 43, 516),
    ("indexer.attn_k.weight", 6, 6291456),
    ("indexer.attn_q_b.weight", 6, 51904512),
    ("indexer.k_norm.bias", 6, 3072),
    ("indexer.k_norm.weight", 6, 3072),
    ("indexer.proj.weight", 6, 3145728),
    ("output.weight", 1, 1648361472),
    ("output_hc_base.weight", 1, 16),
    ("output_hc_fn.weight", 1, 262144),
    ("output_hc_scale.weight", 1, 4),
    ("output_multimax_ranges.weight", 1, 16),
    ("output_multimax_ts.weight", 1, 16),
    ("output_norm.weight", 1, 16384),
    ("token_embd.weight", 1, 1648361472),
];
