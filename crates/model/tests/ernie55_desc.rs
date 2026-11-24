//! ernie55 description gate: what the header reader and the placement inputs resolve from an ERNIE
//! 5.5 Flash Preview file whose header is built here, host only. The checkpoint's own facts are
//! the oracle: the model folder's `config.json` (embedded as [`CONFIG`]; every key is homed in
//! `config::{MAP, FIXED, IGNORED}`) and the shapes and dtypes of its safetensors headers
//! ([`LAYER_COMMON`] and the tables beside it, in the file's ggml order, each entry naming its
//! source tensor). The file's tensor types are the loader's (`gguf::arch::ernie55::STEM_TYPES`, the
//! first of each stem's list): FP8 weights as 128-blocks, the experts as the checkpoint's MXFP4
//! blocks. No tensor's bytes are read: the header lays out as a header-only `Split`
//! (`common/header_tables.rs`).
//!
//! Four contracts, one test each: the config's keys ([`config_keys_have_one_home`]), the header
//! read as the config ([`the_header_reads_as_the_config`]), the layers' stores
//! ([`kv_bytes_of_both_kinds`]) and the plan on one 32 GB card ([`plan_on_a_5090`]), which prints
//! its figures and holds them to the derivation written before the run.

use gguf::Split;
use gguf::arch::ernie55::{ARCH, PRE, allowed_types};
use model::arch::coverage;
use model::arch::ernie55::config::{FIXED, IGNORED, MAP};
use model::arch::ernie55::hparams::{Hparams, unread_keys};
use model::arch::ernie55::kv::kv_layout;
use model::arch::ernie55::place::PlanInputs;
use model::arch::ernie55::roles;
use model::arch::ernie55::spec;
use model::placement::accepts::{Program, routed_rule};
use model::placement::workstation::{self, CardSpec, MIB};
use model::placement::{self, KvBytes, PlanLevers, Role};
use models::{
    Arch, Collapse, Ffn, HcKind, HcMix, IndexKeys, LatentOut, LatentUp, Mixer, OutGate, Score,
    Selector, Source,
};
use serde_json::{Value as J, json};

#[path = "common/header_tables.rs"]
mod header_tables;

/// The model folder's `config.json`, whole.
const CONFIG: &str = r#"{"activation_situ_beta":4.0,"activation_situ_linear_beta":25.0,"add_full_attention_sink_bias":true,"architectures":["Ernie5_5_MoeForCausalLM"],"attention_bias":false,"bos_token_id":1,"compress_ratios":[128,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2],"compress_rope_theta":160000.0,"csa_dense_mode":true,"dspark_block_size":7,"dspark_markov_rank":256,"dspark_noise_token_id":1024,"dspark_target_layer_ids":[40,41,42],"enable_hyper_connections":true,"eos_token_id":2,"experimental_attention_variant":"dsv4_hybrid","first_k_dense_replace":1,"gated_attn_act":"sigmoid","gated_attn_use_q_lora":true,"gated_linear_unit":true,"hc_eps":1e-06,"hc_mult":4,"hc_sinkhorn_iters":20,"hca_rope_type":"yarn","head_dim":512,"hidden_act":"situ","hidden_size":4096,"hybrid_mla_attention":"mqa_dsa","hybrid_mla_kv_lora_rank":512,"hybrid_mla_num_attention_heads":64,"hybrid_mla_num_key_value_heads":64,"hybrid_mla_q_lora_rank":1024,"hybrid_mla_qk_nope_head_dim":192,"hybrid_mla_qk_rope_head_dim":64,"hybrid_mla_v_head_dim":256,"index_head_dim":128,"index_n_heads":64,"index_topk":2048,"intermediate_size":14336,"kv_lora_rank":512,"latent_moe_use_norm":true,"max_position_embeddings":1048576,"model_type":"ernie5_5_moe","moe_group_experts":false,"moe_intermediate_size":2048,"moe_latent_size":2048,"moe_layer_freq":1,"moe_shared_expert_gate":false,"moe_split_feature_routing":true,"moe_token_dispatcher_type":"deepep","mqa_split_kv_b_proj":true,"mtp_shared_last_layer":true,"multimax":"lm_head","n_group":1,"n_routed_experts":512,"n_shared_experts":1,"norm_topk_prob":true,"num_attention_heads":64,"num_experts_per_tok":10,"num_hidden_layers":43,"num_key_value_heads":1,"num_nextn_predict_layers":0,"o_groups":8,"o_lora_rank":1024,"pad_token_id":0,"q_lora_rank":1024,"qk_nope_head_dim":448,"qk_norm_type":"per_head","qk_pos_emb_head_dim":64,"qk_rope_head_dim":64,"rms_norm_eps":1e-06,"rope_interleave":true,"rope_scaling":{"beta_fast":32,"beta_slow":1,"factor":16,"original_max_position_embeddings":65536,"type":"yarn"},"rope_theta":8000000,"rope_type":"rope","rotary_base":8000000,"rotary_percent":1.0,"routed_scaling_factor":3.0,"routed_scaling_factor_learnable":true,"scoring_func":"sigmoid","sliding_window":128,"tie_word_embeddings":false,"topk_group":1,"topk_method":"noaux_tc","torch_dtype":"bfloat16","transformers_version":"4.46.1","use_attn_merger":true,"use_bias":false,"use_cache":true,"use_dense_mtp":false,"use_fast_hadamard":true,"use_gated_attn":true,"use_mla":false,"use_qk_norm":true,"use_vha_attention":true,"v_head_dim":512,"vocab_size":201216,"quantization_config":{"activation_scheme":"dynamic","fmt":"e4m3","quant_method":"fp8","expert_dtype":"mxfp4","scale_fmt":"ue8m0","weight_block_size":[128,128]}}"#;

fn config() -> J {
    serde_json::from_str(CONFIG).expect("the config parses")
}

fn int(cfg: &J, key: &str) -> u64 {
    cfg[key]
        .as_u64()
        .unwrap_or_else(|| panic!("{key}: not an unsigned integer"))
}

fn float(cfg: &J, key: &str) -> J {
    json!(
        cfg[key]
            .as_f64()
            .unwrap_or_else(|| panic!("{key}: not a number"))
    )
}

/// The rank of the head mix's `U` and `V`: `self_attn.vha_postmix_U` is `[64, 16]` in the
/// checkpoint, and no config key holds it.
const VHA_RANK: u64 = 16;

/// The header keys the config gives, as the converter writes them ([`MAP`]): the latent layers'
/// entry of `compress_ratios` (-2) is 0, the nested `rope_scaling` is five keys, the key lengths
/// are the sums of the config's parts, and the sigmoid router is id 2.
fn keys(cfg: &J) -> serde_json::Map<String, J> {
    let rope = int(cfg, "hybrid_mla_qk_rope_head_dim");
    let nope = int(cfg, "hybrid_mla_qk_nope_head_dim");
    let kv_lora = int(cfg, "hybrid_mla_kv_lora_rank");
    let ratios: Vec<J> = cfg["compress_ratios"]
        .as_array()
        .expect("an array")
        .iter()
        .map(|r| json!(r.as_i64().expect("an integer").max(0)))
        .collect();
    let scaling = &cfg["rope_scaling"];
    assert_eq!(cfg["scoring_func"], "sigmoid");
    let table: Vec<(&str, J)> = vec![
        ("general.architecture", json!(ARCH)),
        ("tokenizer.ggml.pre", json!(PRE)),
        ("ernie5_5.block_count", json!(cfg["num_hidden_layers"])),
        (
            "ernie5_5.context_length",
            json!(cfg["max_position_embeddings"]),
        ),
        ("ernie5_5.embedding_length", json!(cfg["hidden_size"])),
        ("ernie5_5.vocab_size", json!(cfg["vocab_size"])),
        (
            "ernie5_5.feed_forward_length",
            json!(cfg["intermediate_size"]),
        ),
        (
            "ernie5_5.leading_dense_block_count",
            json!(cfg["first_k_dense_replace"]),
        ),
        (
            "ernie5_5.attention.head_count",
            json!(cfg["num_attention_heads"]),
        ),
        (
            "ernie5_5.attention.layer_norm_rms_epsilon",
            json!(float(cfg, "rms_norm_eps")),
        ),
        ("ernie5_5.attention.compress_ratios", json!(ratios)),
        (
            "ernie5_5.attention.sliding_window",
            json!(cfg["sliding_window"]),
        ),
        (
            "ernie5_5.attention.q_lora_rank",
            json!(cfg["hybrid_mla_q_lora_rank"]),
        ),
        (
            "ernie5_5.attention.kv_lora_rank",
            json!(cfg["hybrid_mla_kv_lora_rank"]),
        ),
        ("ernie5_5.attention.key_length", json!(kv_lora + rope)),
        ("ernie5_5.attention.key_length_mla", json!(nope + rope)),
        (
            "ernie5_5.attention.value_length_mla",
            json!(cfg["hybrid_mla_v_head_dim"]),
        ),
        ("ernie5_5.rope.dimension_count", json!(rope)),
        ("ernie5_5.attention.hca.head_length", json!(cfg["head_dim"])),
        (
            "ernie5_5.attention.output_group_count",
            json!(cfg["o_groups"]),
        ),
        (
            "ernie5_5.attention.output_lora_rank",
            json!(cfg["o_lora_rank"]),
        ),
        (
            "ernie5_5.attention.indexer.head_count",
            json!(cfg["index_n_heads"]),
        ),
        (
            "ernie5_5.attention.indexer.key_length",
            json!(cfg["index_head_dim"]),
        ),
        ("ernie5_5.attention.indexer.top_k", json!(cfg["index_topk"])),
        ("ernie5_5.attention.vha.rank", json!(VHA_RANK)),
        ("ernie5_5.rope.freq_base", json!(float(cfg, "rope_theta"))),
        (
            "ernie5_5.attention.compress_rope_freq_base",
            json!(float(cfg, "compress_rope_theta")),
        ),
        ("ernie5_5.rope.scaling.type", json!(scaling["type"])),
        (
            "ernie5_5.rope.scaling.factor",
            json!(float(scaling, "factor")),
        ),
        (
            "ernie5_5.rope.scaling.original_context_length",
            json!(scaling["original_max_position_embeddings"]),
        ),
        (
            "ernie5_5.rope.scaling.yarn_beta_fast",
            json!(float(scaling, "beta_fast")),
        ),
        (
            "ernie5_5.rope.scaling.yarn_beta_slow",
            json!(float(scaling, "beta_slow")),
        ),
        ("ernie5_5.expert_count", json!(cfg["n_routed_experts"])),
        (
            "ernie5_5.expert_used_count",
            json!(cfg["num_experts_per_tok"]),
        ),
        (
            "ernie5_5.expert_feed_forward_length",
            json!(cfg["moe_intermediate_size"]),
        ),
        (
            "ernie5_5.expert_shared_count",
            json!(cfg["n_shared_experts"]),
        ),
        ("ernie5_5.expert_gating_func", json!(2)),
        (
            "ernie5_5.expert_weights_scale",
            json!(float(cfg, "routed_scaling_factor")),
        ),
        ("ernie5_5.expert_weights_norm", json!(cfg["norm_topk_prob"])),
        ("ernie5_5.expert_group_count", json!(cfg["n_group"])),
        ("ernie5_5.expert_group_used_count", json!(cfg["topk_group"])),
        (
            "ernie5_5.expert_latent_length",
            json!(cfg["moe_latent_size"]),
        ),
        (
            "ernie5_5.activation.situ_beta",
            json!(float(cfg, "activation_situ_beta")),
        ),
        (
            "ernie5_5.activation.situ_linear_beta",
            json!(float(cfg, "activation_situ_linear_beta")),
        ),
        ("ernie5_5.hyper_connection.count", json!(cfg["hc_mult"])),
        (
            "ernie5_5.hyper_connection.epsilon",
            json!(float(cfg, "hc_eps")),
        ),
        (
            "ernie5_5.hyper_connection.sinkhorn_iterations",
            json!(cfg["hc_sinkhorn_iters"]),
        ),
    ];
    table.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// A tensor of the file: stem and dims in ggml order (`ne0` first). Its type is the loader's:
/// the first of the stem's list in `gguf::arch::ernie55::STEM_TYPES`.
type Row = (&'static str, &'static [u64]);

// The tables below are the checkpoint's: each tensor's safetensors shape is the reverse of its
// ggml dims unless its entry says otherwise. The model's tensors:
// `model.embed_tokens.weight` BF16 [201216, 4096] (also `lm_head.weight`), `model.norm.weight`
// BF16 [4096], `model.mhc_contract.hc_head_fn` F32 [16384, 4] stored as [in, out] (the converter
// transposes: the engine's `fn` is `ne0` = the flattened streams), `hc_head_base` F32 [4],
// `hc_head_scale` F32 [1], `model.lm_head.multimax_ranges` and `multimax_ts` BF16 [4].
const MODEL: &[Row] = &[
    ("token_embd.weight", &[4096, 201_216]),
    ("output.weight", &[4096, 201_216]),
    ("output_norm.weight", &[4096]),
    ("output_hc_fn.weight", &[16_384, 4]),
    ("output_hc_base.weight", &[4]),
    ("output_hc_scale.weight", &[1]),
    ("output_multimax_ranges.weight", &[4]),
    ("output_multimax_ts.weight", &[4]),
];

/// Every layer: `input_layernorm`, `linear_q_down_proj`/`q_a_proj` (F8 [1024, 4096]),
/// `q_layernorm`/`q_a_layernorm` [1024], the sink (`core_attention.attn_sink` F32 [64] on an HCA
/// layer, `softmax_offset` BF16 [64] on a latent one: written F32), `vha_postmix_U`/`V` BF16
/// [64, 16], and each sublayer's hyper-connection: `mapping_proj.weight` F32 [16384, 24] stored
/// [in, out] (transposed by the converter), `bias` [24], and the three scalars `alpha_pre`,
/// `alpha_post`, `alpha_res` joined as `scale` [3].
const LAYER_COMMON: &[Row] = &[
    ("attn_norm.weight", &[4096]),
    ("attn_q_a.weight", &[4096, 1024]),
    ("attn_q_a_norm.weight", &[1024]),
    ("attn_sinks.weight", &[64]),
    ("attn_vha_u.weight", &[16, 64]),
    ("attn_vha_v.weight", &[16, 64]),
    ("hc_attn_fn.weight", &[16_384, 24]),
    ("hc_attn_base.weight", &[24]),
    ("hc_attn_scale.weight", &[3]),
    ("hc_ffn_fn.weight", &[16_384, 24]),
    ("hc_ffn_base.weight", &[24]),
    ("hc_ffn_scale.weight", &[3]),
];

/// An HCA layer: `linear_q_up_proj` F8 [32768, 1024], `linear_kv_proj` F8 [512, 4096],
/// `kv_layernorm` [512], `linear_o_group_proj` F8 [8192, 4096], `o_proj` F8 [4096, 8192],
/// `core_attention.compressor.{linear_wkv, linear_wgate}.weight` BF16 [512, 4096], `.norm` [512],
/// `.ape` F32 [128, 512], `gate_proj` F8 [8192, 1024].
const HCA: &[Row] = &[
    ("attn_q_b.weight", &[1024, 32_768]),
    ("attn_kv.weight", &[4096, 512]),
    ("attn_kv_a_norm.weight", &[512]),
    ("attn_output_a.weight", &[4096, 8192]),
    ("attn_output_b.weight", &[8192, 4096]),
    ("attn_compressor_kv.weight", &[4096, 512]),
    ("attn_compressor_gate.weight", &[4096, 512]),
    ("attn_compressor_norm.weight", &[512]),
    ("attn_compressor_ape.weight", &[512, 128]),
    ("attn_gate.weight", &[1024, 8192]),
];

/// A latent layer: `q_b_proj` F8 [16384, 1024], `kv_a_proj_with_mqa` F8 [576, 4096],
/// `kv_a_layernorm` [512], `kv_b_proj` F8 [28672, 512] kept whole (64 heads of 192 key rows then
/// 256 value rows; the card's halves are cut from it by a later move), `gate_proj` F8 [16384, 1024],
/// `o_proj` F8 [4096, 16384], and the indexer: `wq_b` F8 [8192, 1024], `wk` BF16 [128, 4096],
/// `weights_proj` BF16 [64, 4096], `k_norm.weight` and `k_norm.bias` BF16 [128].
const LATENT: &[Row] = &[
    ("attn_q_b.weight", &[1024, 16_384]),
    ("attn_kv_a_mqa.weight", &[4096, 576]),
    ("attn_kv_a_norm.weight", &[512]),
    ("attn_kv_b.weight", &[512, 28_672]),
    ("attn_gate.weight", &[1024, 16_384]),
    ("attn_output.weight", &[16_384, 4096]),
    ("indexer.attn_q_b.weight", &[1024, 8192]),
    ("indexer.attn_k.weight", &[4096, 128]),
    ("indexer.proj.weight", &[4096, 64]),
    ("indexer.k_norm.weight", &[128]),
    ("indexer.k_norm.bias", &[128]),
];

/// Layer 0's feed-forward block: `post_attention_layernorm` [4096], `mlp.{gate,up}_proj` F8
/// [14336, 4096], `mlp.down_proj` F8 [4096, 14336].
const DENSE: &[Row] = &[
    ("ffn_norm.weight", &[4096]),
    ("ffn_gate.weight", &[4096, 14_336]),
    ("ffn_up.weight", &[4096, 14_336]),
    ("ffn_down.weight", &[14_336, 4096]),
];

/// A routed layer's feed-forward block: `block_sparse_moe.gate.weight` and `.weight_1` BF16
/// [512, 4096], `e_score_correction_bias` F32 [512], `gate.routed_scaling_factor_param` F32 [512],
/// `experts.N.w{1,2,3}.weight` I8 [2048, 1024] with E8M0 scales [2048, 64] (the 512 experts
/// stacked by the converter as MXFP4 [2048, 2048, 512]), `shared_experts.w{1,3}` F8 [2048, 4096]
/// and `w2` F8 [4096, 2048], `fc1_latent_proj` F8 [2048, 4096], `fc2_latent_proj` F8 [4096, 2048],
/// `latent_norm.weight` BF16 [2048].
const ROUTED: &[Row] = &[
    ("ffn_norm.weight", &[4096]),
    ("ffn_gate_inp.weight", &[4096, 512]),
    ("ffn_gate_inp_1.weight", &[4096, 512]),
    ("exp_probs_b.bias", &[512]),
    ("ffn_exp_scale.weight", &[512]),
    ("ffn_gate_exps.weight", &[2048, 2048, 512]),
    ("ffn_up_exps.weight", &[2048, 2048, 512]),
    ("ffn_down_exps.weight", &[2048, 2048, 512]),
    ("ffn_gate_shexp.weight", &[4096, 2048]),
    ("ffn_up_shexp.weight", &[4096, 2048]),
    ("ffn_down_shexp.weight", &[2048, 4096]),
    ("ffn_routed_down.weight", &[4096, 2048]),
    ("ffn_routed_up.weight", &[2048, 4096]),
    ("ffn_routed_norm.weight", &[2048]),
];

/// The layers 0-based that run latent attention: every seventh from the eighth.
fn latent(l: usize) -> bool {
    l > 0 && l.is_multiple_of(7)
}

/// The tensor table of the whole file, one line a tensor, with `edit` applied to each line (its
/// name, the line; `None` drops it).
fn tensors(edit: impl Fn(&str, String) -> Option<String>) -> String {
    let line = |name: String, dims: &[u64]| {
        let stem = name
            .strip_prefix("blk.")
            .and_then(|rest| rest.split_once('.'))
            .map_or(name.as_str(), |(_, stem)| stem);
        let ty = allowed_types(stem)
            .unwrap_or_else(|| panic!("{name}: the loader's table has no stem {stem}"))[0]
            .name()
            .expect("a named type");
        let dims: Vec<String> = dims.iter().map(u64::to_string).collect();
        edit(&name, format!("{name}\t{ty}\t{}\t0", dims.join("x")))
    };
    let mut out = Vec::new();
    out.extend(MODEL.iter().filter_map(|&(n, d)| line(n.to_string(), d)));
    for l in 0..43 {
        let groups: [&[Row]; 4] = [
            LAYER_COMMON,
            if latent(l) { LATENT } else { HCA },
            if l == 0 { DENSE } else { ROUTED },
            &[],
        ];
        for &(stem, dims) in groups.into_iter().flatten() {
            out.extend(line(format!("blk.{l}.{stem}"), dims));
        }
    }
    format!("{}\n", out.join("\n"))
}

/// The header of the file as `cfg` and `edit` give it, `tag` its name.
fn header(
    tag: &str,
    cfg: &J,
    edit_kv: impl FnOnce(&mut serde_json::Map<String, J>),
    edit: impl Fn(&str, String) -> Option<String>,
) -> Split {
    let mut kv = keys(cfg);
    edit_kv(&mut kv);
    header_tables::header_of(tag, &J::Object(kv).to_string(), &tensors(edit))
}

/// The refusal text of reading `split`, or the empty string for a file that reads.
fn refusal(split: &Split) -> String {
    match Hparams::read(split).and_then(|hp| roles::classify(split, &hp).map(|_| ())) {
        Ok(()) => String::new(),
        Err(e) => e.to_string(),
    }
}

/// Every key of the released config is homed: in [`MAP`] (a header key), in [`FIXED`] (a value the
/// description assumes, which the config holds) or in [`IGNORED`] (never run), in exactly one; two
/// config keys that name one header key hold one value. FAIL-first: a key taken out of `IGNORED`
/// has no home, and the clause names it.
#[test]
fn config_keys_have_one_home() {
    let cfg = config();
    let object = cfg.as_object().expect("an object");
    let mut bad = Vec::new();
    for key in object.keys() {
        let homes = [
            MAP.iter().any(|(k, _)| k == key),
            FIXED.iter().any(|(k, _)| k == key),
            IGNORED.contains(&key.as_str()),
        ];
        if homes.iter().filter(|&&h| h).count() != 1 {
            bad.push(format!(
                "config key {key} is in {homes:?} of MAP, FIXED, IGNORED"
            ));
        }
    }
    for (key, want) in FIXED {
        let want: J = serde_json::from_str(want).expect("a JSON value");
        if object.get(*key) != Some(&want) {
            bad.push(format!(
                "config key {key} is {:?}, FIXED holds {want}",
                object.get(*key)
            ));
        }
    }
    for (key, suffix) in MAP {
        if !object.contains_key(*key) {
            bad.push(format!("MAP names {key}, which the config does not hold"));
        }
        for (other, s) in MAP {
            if s == suffix && cfg[*key] != cfg[*other] {
                bad.push(format!("{key} and {other} both give {suffix} and differ"));
            }
        }
    }
    for key in IGNORED {
        if !object.contains_key(*key) {
            bad.push(format!(
                "IGNORED names {key}, which the config does not hold"
            ));
        }
    }
    println!(
        "ernie55 config: {} keys; {} mapped, {} fixed, {} ignored",
        object.len(),
        MAP.len(),
        FIXED.len(),
        IGNORED.len()
    );
    assert!(
        bad.is_empty(),
        "{} config pins differ:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

/// The reader reads the `ernie5` pre-tokenizer's files alone: a header that names another
/// `tokenizer.ggml.pre` (this family's old made-up name among them) or none is refused with the key's
/// name and the one it reads. FAIL-first: the check dropped from `Hparams::read` reads both.
#[test]
fn another_pre_tokenizer_is_refused_by_name() {
    let cfg = config();
    for (tag, edit) in [
        (
            "ernie55-pre-llama",
            Box::new(|kv: &mut serde_json::Map<String, J>| {
                kv.insert("tokenizer.ggml.pre".to_string(), json!("llama-bpe"));
            }) as Box<dyn Fn(&mut serde_json::Map<String, J>)>,
        ),
        (
            "ernie55-pre-old",
            Box::new(|kv: &mut serde_json::Map<String, J>| {
                kv.insert("tokenizer.ggml.pre".to_string(), json!("ernie55"));
            }),
        ),
        (
            "ernie55-pre-none",
            Box::new(|kv: &mut serde_json::Map<String, J>| {
                kv.remove("tokenizer.ggml.pre");
            }),
        ),
    ] {
        let got = refusal(&header(tag, &cfg, |kv| edit(kv), |_, l| Some(l)));
        assert!(
            got.contains("tokenizer.ggml.pre") && got.contains(PRE),
            "{tag}: the refusal is {got:?}, it should name tokenizer.ggml.pre and {PRE}"
        );
    }
}

/// The header reads as the config says: every hparam is a config value (not a literal of this
/// reader), the 6 latent layers are 7, 14, … 42, layer 0 is dense, no key of the header is left
/// unread, and the reader refuses by name a tensor its layer's kind does not carry, a required
/// tensor that is absent and a dim the keys do not give. FAIL-first: a stem moved between kinds
/// in `roles`, or a dims row of `layer_dims` changed, reds the clause.
#[test]
fn the_header_reads_as_the_config() {
    let cfg = config();
    let split = header("ernie55-read", &cfg, |_| {}, |_, l| Some(l));
    let hp = Hparams::read(&split).unwrap_or_else(|e| panic!("read: {e}"));
    let mut bad = Vec::new();
    let mut pin = |what: &str, got: u64, want: u64| {
        if got != want {
            bad.push(format!("{what}: {got}, the config gives {want}"));
        }
    };
    pin("n_layer", hp.n_layer as u64, int(&cfg, "num_hidden_layers"));
    pin("n_embd", hp.n_embd as u64, int(&cfg, "hidden_size"));
    pin("n_vocab", hp.n_vocab as u64, int(&cfg, "vocab_size"));
    pin(
        "n_ctx_train",
        hp.n_ctx_train as u64,
        int(&cfg, "max_position_embeddings"),
    );
    pin(
        "dense_lead",
        hp.dense_lead as u64,
        int(&cfg, "first_k_dense_replace"),
    );
    pin(
        "dense_ff",
        hp.dense_ff as u64,
        int(&cfg, "intermediate_size"),
    );
    pin("heads", hp.heads as u64, int(&cfg, "num_attention_heads"));
    pin(
        "mla.q_lora",
        hp.mla.q_lora as u64,
        int(&cfg, "hybrid_mla_q_lora_rank"),
    );
    pin(
        "mla.kv_lora",
        hp.mla.kv_lora as u64,
        int(&cfg, "hybrid_mla_kv_lora_rank"),
    );
    pin(
        "mla.nope",
        hp.mla.nope() as u64,
        int(&cfg, "hybrid_mla_qk_nope_head_dim"),
    );
    pin(
        "mla.head_v",
        hp.mla.head_v as u64,
        int(&cfg, "hybrid_mla_v_head_dim"),
    );
    pin(
        "mla.pe",
        hp.mla.pe as u64,
        int(&cfg, "hybrid_mla_qk_rope_head_dim"),
    );
    pin("hca.head", hp.hca.head as u64, int(&cfg, "head_dim"));
    pin(
        "hca.window",
        hp.hca.window as u64,
        int(&cfg, "sliding_window"),
    );
    pin(
        "hca.o_groups",
        hp.hca.o_groups as u64,
        int(&cfg, "o_groups"),
    );
    pin("hca.o_lora", hp.hca.o_lora as u64, int(&cfg, "o_lora_rank"));
    pin("hca.ratio", hp.hca.ratio as u64, 128);
    pin(
        "indexer.heads",
        hp.indexer.heads as u64,
        int(&cfg, "index_n_heads"),
    );
    pin(
        "indexer.d",
        hp.indexer.d as u64,
        int(&cfg, "index_head_dim"),
    );
    pin(
        "indexer.top_k",
        hp.indexer.top_k as u64,
        int(&cfg, "index_topk"),
    );
    pin("vha_rank", hp.vha_rank as u64, VHA_RANK);
    pin(
        "experts.n_expert",
        hp.experts.n_expert as u64,
        int(&cfg, "n_routed_experts"),
    );
    pin(
        "experts.n_used",
        hp.experts.n_used as u64,
        int(&cfg, "num_experts_per_tok"),
    );
    pin(
        "experts.ff",
        hp.experts.ff as u64,
        int(&cfg, "moe_intermediate_size"),
    );
    pin(
        "experts.n_shared",
        hp.experts.n_shared as u64,
        int(&cfg, "n_shared_experts"),
    );
    pin("latent", hp.latent as u64, int(&cfg, "moe_latent_size"));
    pin("hc.streams", hp.hc.streams as u64, int(&cfg, "hc_mult"));
    pin(
        "hc.sinkhorn",
        hp.hc.sinkhorn_iters as u64,
        int(&cfg, "hc_sinkhorn_iters"),
    );
    pin("hc.mix", hp.hc.mix() as u64, 24);
    pin("rope.original_ctx", hp.rope.original_ctx as u64, 65_536);
    let f = |key: &str| cfg[key].as_f64().expect("a number") as f32;
    for (what, got, want) in [
        ("rope.base", hp.rope.base, f("rope_theta")),
        (
            "rope.compress_base",
            hp.rope.compress_base,
            f("compress_rope_theta"),
        ),
        (
            "rope.factor",
            hp.rope.factor,
            cfg["rope_scaling"]["factor"].as_f64().unwrap() as f32,
        ),
        (
            "weights_scale",
            hp.experts.weights_scale,
            f("routed_scaling_factor"),
        ),
        (
            "situ.gate_beta",
            hp.situ.gate_beta,
            f("activation_situ_beta"),
        ),
        (
            "situ.value_beta",
            hp.situ.value_beta,
            f("activation_situ_linear_beta"),
        ),
        ("rms_eps", hp.rms_eps, f("rms_norm_eps")),
        ("hc.eps", hp.hc.eps, f("hc_eps")),
    ] {
        if got != want {
            bad.push(format!("{what}: {got}, the config gives {want}"));
        }
    }
    let latents: Vec<usize> = (0..hp.n_layer).filter(|&l| hp.latent_layer(l)).collect();
    if latents != [7, 14, 21, 28, 35, 42] {
        bad.push(format!(
            "latent layers {latents:?}, the config's -2 entries are 7, 14, … 42"
        ));
    }
    if !hp.dense(0) || hp.dense(1) {
        bad.push("layer 0 is the one dense layer".to_string());
    }
    let unread = unread_keys(&split);
    if !unread.is_empty() {
        bad.push(format!("keys left unread: {unread:?}"));
    }
    println!(
        "ernie55 hparams: {} layers ({} HCA ratio {}, {} latent), {} experts top-{}, latent {}",
        hp.n_layer,
        hp.n_layer - latents.len(),
        hp.hca.ratio,
        latents.len(),
        hp.experts.n_expert,
        hp.experts.n_used,
        hp.latent
    );

    // The roles: every tensor claimed, the routed stacks of 42 layers, 6,684,672 bytes an expert.
    let model = roles::classify(&split, &hp).unwrap_or_else(|e| panic!("classify: {e}"));
    let stacks: Vec<_> = model
        .tensors
        .iter()
        .filter(|t| t.role == Role::RoutedExperts)
        .collect();
    if stacks.len() != 42 * 3 {
        bad.push(format!("{} routed stacks, 42 layers of 3", stacks.len()));
    }
    let expert: u64 = stacks.iter().take(3).map(|t| t.file_bytes / 512).sum();
    if expert != 6_684_672 {
        bad.push(format!(
            "an expert is {expert} bytes, the header says 6,684,672"
        ));
    }
    println!(
        "ernie55 roles: {} tensors, {} routed stacks, {expert} bytes an expert",
        model.tensors.len(),
        stacks.len()
    );

    // The refusals, each by name.
    let mut refuses = |tag: &str, edit: &dyn Fn(&str, String) -> Option<String>, want: &str| {
        let got = refusal(&header(tag, &cfg, |_| {}, edit));
        if !got.contains(want) {
            bad.push(format!(
                "{tag}: the refusal is {got:?}, it should name {want}"
            ));
        }
    };
    refuses(
        "ernie55-indexer-on-hca",
        &|n, l| {
            Some(if n == "blk.8.attn_norm.weight" {
                format!("{l}\nblk.8.indexer.proj.weight\tBF16\t4096x64\t0")
            } else {
                l
            })
        },
        "blk.8.indexer.proj.weight",
    );
    refuses(
        "ernie55-no-bias",
        &|n, l| (n != "blk.14.indexer.k_norm.bias").then_some(l),
        "blk.14.indexer.k_norm.bias",
    );
    refuses(
        "ernie55-no-scale",
        &|n, l| (n != "blk.3.ffn_exp_scale.weight").then_some(l),
        "blk.3.ffn_exp_scale.weight",
    );
    refuses(
        "ernie55-split-kv-b",
        &|n, l| {
            Some(if n == "blk.7.attn_kv_b.weight" {
                format!("{l}\nblk.7.attn_k_b.weight\tMXFP8\t192x512x64\t0")
            } else {
                l
            })
        },
        "blk.7.attn_k_b.weight",
    );
    refuses(
        "ernie55-gate-on-hidden",
        &|n, l| {
            Some(if n == "blk.7.attn_gate.weight" {
                l.replace("1024x16384", "4096x16384")
            } else {
                l
            })
        },
        "blk.7.attn_gate.weight",
    );
    refuses(
        "ernie55-no-multimax",
        &|n, l| (n != "output_multimax_ts.weight").then_some(l),
        "output_multimax_ts.weight",
    );
    assert!(
        bad.is_empty(),
        "{} pins differ:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

/// The stores of the two kinds at 32768 positions, against the figures of the checkpoint's
/// dimensions: a latent layer holds 1,408 bytes a position (the 576-value row and the 128-value
/// indexer key, f16); an HCA layer a 128-row ring and a pooled row every 128 positions (both 1,024
/// bytes) and the pooling state of 524,288 bytes. FAIL-first: the state left out, or the indexer
/// key left out of a latent layer, reds the clause.
#[test]
fn kv_bytes_of_both_kinds() {
    let cfg = config();
    let hp = Hparams::read(&header("ernie55-kv", &cfg, |_| {}, |_, l| Some(l))).expect("reads");
    let kv = kv_layout(&hp);
    let ctx = 32_768;
    let (mut mla, mut pooled, mut ring_state) = (0u64, 0u64, 0u64);
    for l in 0..hp.n_layer {
        if hp.latent_layer(l) {
            mla += kv.layer_bytes(l, ctx);
        } else {
            pooled += kv.by_position(l, ctx);
            ring_state += kv.fixed(l, ctx);
        }
    }
    println!(
        "ernie55 kv @{ctx}: latent {mla} B (6 layers), HCA pooled rows {pooled} B, ring + state {ring_state} B"
    );
    let want = (
        6 * 32_768 * 1408,
        37 * 256 * 1024,
        37 * (128 * 1024 + 524_288),
    );
    assert_eq!(
        (mla, pooled, ring_state),
        want,
        "latent layers, HCA pooled rows, HCA ring + state"
    );
    // The ring is whole rows to the window and the rows grow with the positions.
    assert_eq!(kv.fixed(0, 64), 64 * 1024 + 524_288);
    assert_eq!(kv.by_position(0, 262_144), 2048 * 1024);
    assert_eq!(kv.layer_bytes(7, 262_144), 262_144 * 1408);
}

/// The card the plan is made on: an RTX 5090 as `nvidia-smi` reports it, its driver reserve the
/// figure the named cards carry [assumed: no 5090 row in `workstation`].
const RTX_5090: CardSpec = CardSpec {
    name: "5090",
    total_bytes: 32_607 * MIB,
    driver_reserve_bytes: 550 * MIB,
    device: None,
    free_bytes: None,
    held_by: None,
    integrated: false,
    margin_bytes: None,
    granule_bytes: workstation::GRANULE,
};

/// The plan of the file on one 5090 at `ctx`: per routed layer the experts the card holds, and the
/// host's bytes. The routed stacks take the rule of [`Program::Ernie55Body`], the common card leg;
/// no draft, no prompt pass and no stream ring are counted, so the figures are the decode plan's.
struct Planned {
    n_l: Vec<u64>,
    card_experts: u64,
    dense: u64,
    kv: u64,
    rounding: u64,
    host_experts: u64,
    host_table: u64,
}

fn planned(inputs: &PlanInputs, ctx: u64) -> Planned {
    inputs.check_ctx(ctx).expect("a trained context");
    let machine = workstation::plan_on(RTX_5090, inputs.model.layers);
    let plan = placement::plan_routed_reserving(
        &inputs.model,
        &machine,
        ctx,
        &inputs.kv,
        &PlanLevers::default(),
        routed_rule(Program::Ernie55Body),
        0,
    )
    .unwrap_or_else(|e| panic!("plan @{ctx}: {e}"));
    let bad = plan.violations();
    assert!(
        bad.is_empty(),
        "plan @{ctx} breaks its invariants: {}",
        placement::joined(&bad)
    );
    let card = &plan.cards[0];
    Planned {
        n_l: plan.n_l.clone(),
        card_experts: card.experts,
        dense: card.dense_bytes,
        kv: card.kv_bytes,
        rounding: card.rounding_bytes,
        host_experts: plan.host.expert_bytes,
        host_table: plan.host.table_bytes,
    }
}

/// What `--plan` would place on a 5090 at the contexts 32768 and 262144: printed, and held to the
/// band the derivation gave before it ran (`predictions.txt`: 3,396 card experts, 81 a routed
/// layer, host routed 121.05 GB at 32768; 3,096, 74 and 123.05 GB at 262144; the report's §2.1
/// gives 3,236 and 77 a layer, the 166-expert gap being the head in MXFP8 planes, no draft term,
/// the card's size and reserves, and the pooling state). Every expert is on the card or the host:
/// the card's bytes and the host's make the file's 42 x 512 x 6,684,672. FAIL-first: a KV store
/// dropped from `kv_layout` moves the card's experts out of the band.
#[test]
fn plan_on_a_5090() {
    let cfg = config();
    let split = header("ernie55-plan", &cfg, |_| {}, |_, l| Some(l));
    let inputs = PlanInputs::describe(&split).expect("describes");
    let all = 42 * 512 * 6_684_672u64;
    let (a, b) = (planned(&inputs, 32_768), planned(&inputs, 262_144));
    for (ctx, p) in [(32_768, &a), (262_144, &b)] {
        let routed: Vec<u64> = p.n_l[1..].to_vec();
        let (lo, hi) = (routed.iter().min().unwrap(), routed.iter().max().unwrap());
        println!(
            "ernie55 plan @{ctx}, 5090 32 GB: card experts {} ({lo}..{hi} a routed layer, layer 0 {}), dense {} B, kv {} B, rounding {} B; host experts {} B, host table {} B",
            p.card_experts, p.n_l[0], p.dense, p.kv, p.rounding, p.host_experts, p.host_table
        );
        println!("  card experts per layer: {:?}", p.n_l);
        assert_eq!(p.n_l[0], 0, "layer 0 is dense");
        assert_eq!(p.n_l.iter().sum::<u64>(), p.card_experts);
        assert_eq!(
            p.card_experts * 6_684_672 + p.host_experts,
            all,
            "every expert is on the card or the host"
        );
    }
    let per_layer = |p: &Planned| p.card_experts as f64 / 42.0;
    println!(
        "ernie55 plan vs report 2.1 (3,236 experts, 77.0 a layer): @32768 {} ({:.1}), @262144 {} ({:.1})",
        a.card_experts,
        per_layer(&a),
        b.card_experts,
        per_layer(&b)
    );
    assert!(
        (76.0..=86.0).contains(&per_layer(&a)),
        "@32768: {} a layer, derived 80.9 (band 76 to 86)",
        per_layer(&a)
    );
    assert!(
        (69.0..=79.0).contains(&per_layer(&b)),
        "@262144: {} a layer, derived 73.7 (band 69 to 79)",
        per_layer(&b)
    );
    assert!(
        b.card_experts < a.card_experts,
        "a longer context leaves the card fewer experts"
    );
    assert!(b.kv > a.kv);
}

/// The description of the file: every layer a latent attention (an HCA layer with the pooled stream
/// and the grouped output, or the absorbed MLA row with the DSA selector), each gated from the
/// query latent and holding a sink; the first layer a dense `situ` block and the rest the
/// sigmoid-routed experts in a latent beside a shared one; the residual four mHC streams with the
/// root RMS factor, each sublayer its own mix and the head folding them with its own. FAIL-first:
/// `OutGate::QLatent` set to `Proj`, `sum_norm` dropped or `MhcRoot` swapped for `Mhc` is a line
/// of this clause.
#[test]
fn the_description_is_the_architecture() {
    let cfg = config();
    let split = header("ernie55-spec", &cfg, |_| {}, |_, l| Some(l));
    let read = spec::read(&split).expect("describes");
    let s = &read.spec;
    assert_eq!(s.arch, Arch::Ernie55);
    assert_eq!((s.hidden, s.vocab, s.layers.len()), (4096, 201_216, 43));
    for (l, layer) in s.layers.iter().enumerate() {
        let a = layer
            .latent()
            .unwrap_or_else(|| panic!("layer {l} is not latent attention"));
        assert_eq!((a.heads, a.q_lora), (64, 1024), "layer {l}");
        assert!(a.sinks && a.out_gate == OutGate::QLatent, "layer {l}");
        assert_eq!(layer.residual, models::Residual::Hc, "layer {l}");
        if latent(l) {
            assert_eq!(a.latent, 512, "layer {l}");
            assert_eq!(a.up, LatentUp::Absorbed { qk: 256, v: 256 }, "layer {l}");
            assert_eq!(
                (a.pe, a.rope.map(|r| (r.dims, r.base.to_bits(), r.yarn))),
                (0, Some((64, 8_000_000f32.to_bits(), None))),
                "layer {l}"
            );
            assert_eq!(a.out, LatentOut::Plain, "layer {l}");
            assert!(a.compress.is_none() && a.window.is_none(), "layer {l}");
            let Some(Selector::StreamTopK {
                heads,
                d,
                k,
                keys,
                list,
                candidates,
            }) = &a.select
            else {
                panic!("layer {l}: no stream top-k");
            };
            assert_eq!((*heads, *d, *k), (64, 128, 2048), "layer {l}");
            assert!(
                matches!(keys, Source::Own(IndexKeys::FromInput { .. })),
                "layer {l}"
            );
            assert!(
                matches!(list, Source::Own(())) && candidates.is_none(),
                "layer {l}"
            );
        } else {
            assert_eq!(a.latent, 512, "layer {l}");
            assert_eq!(a.up, LatentUp::KeqV, "layer {l}");
            let rope = a.rope.expect("rope");
            assert_eq!(
                (rope.dims, rope.base.to_bits()),
                (64, 160_000f32.to_bits()),
                "layer {l}"
            );
            assert!(rope.yarn.is_some() && a.q_head_norm, "layer {l}");
            assert_eq!(
                a.out,
                LatentOut::Grouped {
                    groups: 8,
                    rank: 1024
                },
                "layer {l}"
            );
            assert_eq!(a.window, Some(128), "layer {l}");
            let c = a.compress.as_ref().expect("a pooled stream");
            assert_eq!(c.ratio, 128, "layer {l}");
            assert!(
                matches!(c.rows, Source::Own(r) if r.gated && r.ape && !r.overlap),
                "layer {l}"
            );
            assert!(a.select.is_none(), "layer {l}");
        }
        match (&layer.ffn, l) {
            (Ffn::Dense { ff: 14_336, .. }, 0) => {}
            (Ffn::Moe(m), l) if l > 0 => {
                assert_eq!(
                    (m.experts, m.top_k, m.expert_ff, m.latent),
                    (512, 10, 2048, Some(2048)),
                    "layer {l}"
                );
                assert!(
                    m.sum_norm && !m.pre_exp_norm && m.post_exp_norm.is_none(),
                    "layer {l}"
                );
                let r = m.router;
                assert!(r.score == Score::Sigmoid && r.bias && !r.hash, "layer {l}");
                assert_eq!(
                    r.scale.to_bits(),
                    1.0f32.to_bits(),
                    "the learned scale is the factor, layer {l}"
                );
                assert_eq!(m.shared.map(|x| x.ff), Some(2048), "layer {l}");
            }
            (other, l) => panic!("layer {l}: {other:?}"),
        }
        assert!(matches!(layer.mixer, Mixer::Latent(_)));
    }
    let hc = s.hc.expect("hyper-connections");
    assert_eq!(hc.streams, 4);
    assert_eq!(
        hc.kind,
        HcKind::MhcRoot {
            sinkhorn: 20,
            eps: 1e-6,
            mix: HcMix::Own,
            collapse: Collapse::Head
        }
    );
    let facts = read.defaults.join("\n");
    for part in [
        "router norm = selected-sum",
        "ffn_exp_scale",
        "1 / (sqrt(mean) + eps)",
        "mHC post multiplier = 2.0",
    ] {
        assert!(facts.contains(part), "the defaults lack {part:?}:\n{facts}");
    }
}

/// No program runs the family, so the description is read and refused by name: the coverage check
/// lists the needs no program runs (the mHC root factor, the query-latent gate on both output
/// rows, the pooled stream, the selector, the latent experts) and [`PlanInputs::read`] adds the
/// ops the description cannot state. FAIL-first: an `Ops::unrun` that lists nothing leaves the
/// refusal without the four ops.
#[test]
fn the_description_is_not_runnable() {
    let cfg = config();
    let split = header("ernie55-refuse", &cfg, |_| {}, |_, l| Some(l));
    let inputs = PlanInputs::describe(&split).expect("describes");
    let needs: Vec<String> = coverage::check(&inputs.spec, &inputs.model)
        .iter()
        .map(|i| i.feature.clone())
        .collect();
    for part in [
        "hyper-connections of 4 streams with the RMS factor 1 / (sqrt(mean) + eps)",
        "of a projection of the query latent, on the grouped output projection's rows",
        "of a projection of the query latent, on the heads' outputs",
    ] {
        assert!(
            needs.iter().any(|n| n.contains(part)),
            "no item {part:?} in {needs:#?}"
        );
    }
    let refused = PlanInputs::read(&split)
        .expect_err("no program runs it")
        .to_string();
    println!("ernie55 refusal: {refused}");
    for part in [
        "the head mix of rank 16",
        "2 summed sigmoid routers",
        "the head's MultiMax: 4 ranges",
        "the indexer's NeoX rope over its leading 64 values",
    ] {
        assert!(
            refused.contains(part),
            "the refusal lacks {part:?}: {refused}"
        );
    }
}
