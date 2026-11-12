use std::collections::BTreeMap;
use std::convert::Infallible;

use gguf::GgmlType;
use hf::quant_config::Config;
use safetensors::{Dtype, Safetensors};
use serde_json::{Value as Json, json};

use super::*;
use crate::folder::{self, Keep};
use crate::test_support::{Entries, temp_dir, tokenizer, write_folder};

/// The toy widths: four blocks — the first dense and full (NoPE), the second routed through the
/// wide window, the third routed through the narrow one and clamped, the fourth routed and full
/// and clamped — and the MTP block, over eight experts of two hundred fifty-six values in a
/// hidden width of two hundred fifty-six. Every first dim is a whole FP8 block and a whole Q8_0
/// and Q4_K row.
const E: usize = 256;
const HEADS: usize = 2;
const KV: usize = 1;
const HEAD_DIM: usize = 128;
const Q: usize = HEADS * HEAD_DIM;
const K: usize = KV * HEAD_DIM;
const LAYERS: usize = 4;
const MTP: usize = LAYERS;
const FF: usize = 768;
const EFF: usize = 512;
const EXPERTS: usize = 8;
const SHARED: usize = SHARED_COUNT * EFF;
const SHARED_COUNT: usize = 1;
const VOCAB: usize = 8;

/// `base`'s object with `more`'s keys over it: two `json!` halves of one config, kept apart so
/// neither outgrows the macro's recursion.
fn merged(mut base: Json, more: Json) -> Json {
    base.as_object_mut()
        .unwrap()
        .extend(more.as_object().unwrap().clone());
    base
}

/// The config of the toy blocks, in the checkpoint's own spelling.
fn toy_config() -> Json {
    merged(
        json!({
            "architectures": ["ExaoneMoeForCausalLM"], "model_type": "exaone_moe",
            "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3",
                "ignored_layers": ["lm_head", "mtp"], "quant_method": "fp8",
                "weight_block_size": [128, 128]},
            "attention_dropout": 0.0, "bos_token_id": 1, "dtype": "bfloat16",
            "eos_token_id": 2, "head_dim": HEAD_DIM, "hidden_act": "silu", "hidden_size": E,
            "initializer_range": 0.02, "intermediate_size": FF,
        }),
        json!({
            "layer_types": ["full_attention", "sliding_attention", "sliding_attention",
                            "full_attention"],
            "max_position_embeddings": 4096,
            "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
            "moe_intermediate_size": EFF, "mtp_layer_types": ["sliding_attention"],
            "mtp_loss_scaling_factor": 0.05, "mtp_num_speculative_steps": 4,
            "mtp_sliding_windows": [128], "n_group": 1, "norm_topk_prob": true,
            "num_attention_heads": HEADS, "num_experts": EXPERTS, "num_experts_per_tok": 2,
            "num_hidden_layers": LAYERS, "num_key_value_heads": KV,
            "num_nextn_predict_layers": 1, "num_shared_experts": SHARED_COUNT,
            "pad_token_id": 0, "rms_norm_eps": 1e-5,
            "rope_parameters": {"rope_theta": 1e6, "rope_type": "default"},
            "routed_scaling_factor": 2.5, "scoring_func": "sigmoid",
            "sliding_windows": [0, 4096, 128, 0], "swiglu_limits": [0.0, 0.0, 7.0, 7.0],
            "tie_word_embeddings": false, "tokenizer_class": "GPT2Tokenizer",
            "topk_group": 1, "topk_method": "noaux_tc", "transformers_version": "5.9.0",
            "use_cache": true, "vocab_size": VOCAB,
        }),
    )
}

/// One tensor of an entries list.
fn push(t: &mut Entries, name: String, dtype: Dtype, shape: &[usize]) {
    t.push((name, dtype, shape.to_vec()));
}

/// An FP8 weight's `weight_scale_inv` grid: one scale a 128-by-128 square.
fn grid(rows: usize, cols: usize) -> Vec<usize> {
    vec![rows.div_ceil(128), cols.div_ceil(128)]
}

/// A block's attention projections, each beside its scale grid.
fn attention(t: &mut Entries, at: &str) {
    use Dtype::{F8E4M3, F32};
    for (proj, rows, cols) in [
        ("q_proj", Q, E),
        ("k_proj", K, E),
        ("v_proj", K, E),
        ("o_proj", E, Q),
    ] {
        push(
            t,
            format!("{at}self_attn.{proj}.weight"),
            F8E4M3,
            &[rows, cols],
        );
        push(
            t,
            format!("{at}self_attn.{proj}.weight_scale_inv"),
            F32,
            &grid(rows, cols),
        );
    }
}

/// The toy folder's tensors at those widths, as the checkpoint names and shapes them: name,
/// dtype and shape.
fn toy_entries() -> Entries {
    use Dtype::{Bf16, F8E4M3, F32};
    let mut t: Entries = vec![];
    push(
        &mut t,
        "model.embed_tokens.weight".into(),
        Bf16,
        &[VOCAB, E],
    );
    push(&mut t, "lm_head.weight".into(), Bf16, &[VOCAB, E]);
    push(&mut t, "model.norm.weight".into(), Bf16, &[E]);
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}.");
        push(&mut t, format!("{p}input_layernorm.weight"), Bf16, &[E]);
        push(
            &mut t,
            format!("{p}post_attention_layernorm.weight"),
            Bf16,
            &[E],
        );
        push(
            &mut t,
            format!("{p}self_attn.q_norm.weight"),
            Bf16,
            &[HEAD_DIM],
        );
        push(
            &mut t,
            format!("{p}self_attn.k_norm.weight"),
            Bf16,
            &[HEAD_DIM],
        );
        attention(&mut t, &p);
        if l == 0 {
            for (proj, rows, cols) in [
                ("gate_proj", FF, E),
                ("up_proj", FF, E),
                ("down_proj", E, FF),
            ] {
                push(
                    &mut t,
                    format!("{p}mlp.{proj}.weight"),
                    F8E4M3,
                    &[rows, cols],
                );
                push(
                    &mut t,
                    format!("{p}mlp.{proj}.weight_scale_inv"),
                    F32,
                    &grid(rows, cols),
                );
            }
            continue;
        }
        push(&mut t, format!("{p}mlp.gate.weight"), F32, &[EXPERTS, E]);
        push(
            &mut t,
            format!("{p}mlp.e_score_correction_bias"),
            F32,
            &[EXPERTS],
        );
        for (proj, rows, cols) in [
            ("gate_proj", SHARED, E),
            ("up_proj", SHARED, E),
            ("down_proj", E, SHARED),
        ] {
            push(
                &mut t,
                format!("{p}mlp.shared_experts.{proj}.weight"),
                F8E4M3,
                &[rows, cols],
            );
            push(
                &mut t,
                format!("{p}mlp.shared_experts.{proj}.weight_scale_inv"),
                F32,
                &grid(rows, cols),
            );
        }
        for id in 0..EXPERTS {
            for (proj, rows, cols) in [
                ("gate_proj", EFF, E),
                ("up_proj", EFF, E),
                ("down_proj", E, EFF),
            ] {
                push(
                    &mut t,
                    format!("{p}mlp.experts.{id}.{proj}.weight"),
                    F8E4M3,
                    &[rows, cols],
                );
                push(
                    &mut t,
                    format!("{p}mlp.experts.{id}.{proj}.weight_scale_inv"),
                    F32,
                    &grid(rows, cols),
                );
            }
        }
    }
    // The MTP block, as the checkpoint names it: BF16, no scale beside any weight.
    push(&mut t, "mtp.fc.weight".into(), Bf16, &[E, 2 * E]);
    push(
        &mut t,
        "mtp.pre_fc_norm_embedding.weight".into(),
        Bf16,
        &[E],
    );
    push(&mut t, "mtp.pre_fc_norm_hidden.weight".into(), Bf16, &[E]);
    push(&mut t, "mtp.norm.weight".into(), Bf16, &[E]);
    push(
        &mut t,
        "mtp.layers.0.input_layernorm.weight".into(),
        Bf16,
        &[E],
    );
    push(
        &mut t,
        "mtp.layers.0.post_attention_layernorm.weight".into(),
        Bf16,
        &[E],
    );
    push(
        &mut t,
        "mtp.layers.0.self_attn.q_norm.weight".into(),
        Bf16,
        &[HEAD_DIM],
    );
    push(
        &mut t,
        "mtp.layers.0.self_attn.k_norm.weight".into(),
        Bf16,
        &[HEAD_DIM],
    );
    for (proj, rows, cols) in [
        ("q_proj", Q, E),
        ("k_proj", K, E),
        ("v_proj", K, E),
        ("o_proj", E, Q),
    ] {
        push(
            &mut t,
            format!("mtp.layers.0.self_attn.{proj}.weight"),
            Bf16,
            &[rows, cols],
        );
    }
    for (proj, rows, cols) in [
        ("gate_proj", FF, E),
        ("up_proj", FF, E),
        ("down_proj", E, FF),
    ] {
        push(
            &mut t,
            format!("mtp.layers.0.mlp.{proj}.weight"),
            Bf16,
            &[rows, cols],
        );
    }
    t
}

/// A plan of `entries` under `config`'s checkpoint map, in the load's `Keep::InPlace`.
fn plan(config: &Json, entries: &Entries) -> Result<folder::Planning<Infallible>, folder::Error> {
    let map = Checkpoint::parse(config)?;
    folder::plan_tensors(
        &map,
        &Config::parse(config, None).unwrap(),
        entries,
        false,
        Keep::InPlace,
    )
}

fn refusal(config: &Json, entries: &Entries) -> String {
    match plan(config, entries) {
        Ok(_) => panic!("planned, where a refusal was expected"),
        Err(e) => e.to_string(),
    }
}

/// The real model's config — seventy-eight blocks of the published window, density and clamp
/// pattern and one MTP block — as its `config.json` spells it, at the published counts.
fn real_config() -> Json {
    let full_at = |l: usize| l == 0 || (l >= 5 && (l - 5).is_multiple_of(4));
    let layer_types: Vec<&str> = (0..78)
        .map(|l| {
            if full_at(l) {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();
    let windows: Vec<u32> = (0..78)
        .map(|l| {
            if full_at(l) {
                0
            } else if l == 1 {
                4096
            } else {
                128
            }
        })
        .collect();
    let mlp: Vec<&str> = (0..78)
        .map(|l| if l < 2 { "dense" } else { "sparse" })
        .collect();
    let limits: Vec<f32> = (0..78).map(|l| if l >= 62 { 7.0 } else { 0.0 }).collect();
    merged(
        json!({
            "architectures": ["ExaoneMoeForCausalLM"], "model_type": "exaone_moe",
            "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3",
                "ignored_layers": ["lm_head", "mtp"], "quant_method": "fp8",
                "weight_block_size": [128, 128]},
            "bos_token_id": 1, "eos_token_id": 53, "head_dim": 128, "hidden_act": "silu",
            "hidden_size": 6144, "intermediate_size": 18432, "layer_types": layer_types,
            "max_position_embeddings": 262144, "mlp_layer_types": mlp,
        }),
        json!({
            "moe_intermediate_size": 2048, "mtp_layer_types": ["sliding_attention"],
            "mtp_loss_scaling_factor": 0.05, "mtp_num_speculative_steps": 4,
            "mtp_sliding_windows": [128], "n_group": 1, "norm_topk_prob": true,
            "num_attention_heads": 64, "num_experts": 256, "num_experts_per_tok": 8,
            "num_hidden_layers": 78, "num_key_value_heads": 8,
            "num_nextn_predict_layers": 1, "num_shared_experts": 1, "pad_token_id": 0,
            "rms_norm_eps": 1e-5,
            "rope_parameters": {"rope_theta": 1e6, "rope_type": "default"},
            "routed_scaling_factor": 2.5, "scoring_func": "sigmoid",
            "sliding_windows": windows, "swiglu_limits": limits,
            "tie_word_embeddings": false, "tokenizer_class": "GPT2Tokenizer",
            "topk_group": 1, "topk_method": "noaux_tc", "vocab_size": 153600,
        }),
    )
}

/// The real config reads into llama.cpp's keys: every scalar at its published value and every
/// per-layer key an array of seventy-nine entries — the seventy-eight blocks and the MTP one —
/// whose window, pattern and clamp entries are the published pattern's, the rope and the window
/// pattern agreeing (the rope turns on where a window is). Mutants: the MTP entry dropped from
/// `block_count`, the patterns read from the attention kinds alone, the clamp MTP entry dropped.
#[test]
fn the_real_config_reads_into_llama_cpp_s_keys() {
    let c = Checkpoint::parse(&real_config()).unwrap();
    let key = |suffix: &str| {
        c.keys
            .iter()
            .find(|(k, _)| k == suffix)
            .map(|(_, v)| v.clone())
    };
    for (suffix, want) in [
        ("vocab_size", Value::U32(153_600)),
        ("context_length", Value::U32(262_144)),
        ("embedding_length", Value::U32(6144)),
        ("block_count", Value::U32(79)),
        ("feed_forward_length", Value::U32(18_432)),
        ("leading_dense_block_count", Value::U32(2)),
        ("nextn_predict_layers", Value::U32(1)),
        ("attention.head_count", Value::U32(64)),
        ("attention.head_count_kv", Value::U32(8)),
        ("attention.key_length", Value::U32(128)),
        ("attention.value_length", Value::U32(128)),
        ("attention.layer_norm_rms_epsilon", Value::F32(1e-5)),
        ("rope.freq_base", Value::F32(1e6)),
        ("rope.dimension_count", Value::U32(128)),
        ("expert_count", Value::U32(256)),
        ("expert_used_count", Value::U32(8)),
        ("expert_shared_count", Value::U32(1)),
        ("expert_feed_forward_length", Value::U32(2048)),
        ("expert_shared_feed_forward_length", Value::U32(2048)),
        ("expert_gating_func", Value::U32(2)),
        ("expert_weights_norm", Value::Bool(true)),
        ("expert_weights_scale", Value::F32(2.5)),
        ("expert_group_count", Value::U32(1)),
        ("expert_group_used_count", Value::U32(1)),
    ] {
        assert_eq!(key(suffix), Some(want), "{suffix}");
    }
    let Some(Value::Array(windows)) = key("attention.sliding_window") else {
        panic!("sliding_window is an array");
    };
    assert_eq!(windows.len(), 79);
    for (l, want) in [
        (0usize, 0u32),
        (1, 4096),
        (2, 128),
        (5, 0),
        (61, 0),
        (62, 128),
        (77, 0),
        (78, 128),
    ] {
        assert_eq!(windows[l], Value::U32(want), "sliding_window entry {l}");
    }
    for suffix in ["attention.sliding_window_pattern", "attention.rope_pattern"] {
        let Some(Value::Array(pattern)) = key(suffix) else {
            panic!("{suffix} is an array");
        };
        assert_eq!(pattern.len(), 79);
        for (l, want) in [(0, false), (1, true), (5, false), (77, false), (78, true)] {
            assert_eq!(pattern[l], Value::Bool(want), "{suffix} entry {l}");
        }
    }
    for suffix in ["swiglu_clamp_exp", "swiglu_clamp_shexp"] {
        let Some(Value::Array(clamp)) = key(suffix) else {
            panic!("{suffix} is an array");
        };
        assert_eq!(clamp.len(), 79);
        for (l, want) in [(0, 0.0f32), (61, 0.0), (62, 7.0), (77, 7.0), (78, 0.0)] {
            assert_eq!(clamp[l], Value::F32(want), "{suffix} entry {l}");
        }
    }
}

/// The planned tensors by name: ggml dims and declared type.
fn table(p: &folder::Planning<Infallible>) -> BTreeMap<String, (Vec<u64>, GgmlType)> {
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

/// The toy folder plans from its headers alone, with nothing refused and nothing skipped: every
/// tensor in the table below, at the ggml dims the config gives and in the type its role reads
/// it in — the FP8 routed stacks kept as the checkpoint's FP8_B128, an FP8 trunk projection
/// decoded and quantized to Q8_0, BF16 where `prepare` keeps the stored form, F32 norms, router
/// and bias. Mutants: a trunk projection read `Matrix` (kept FP8), the head read `Bf16`.
#[test]
fn a_toy_folder_plans_its_tensors() {
    let p = plan(&toy_config(), &toy_entries()).unwrap();
    assert!(p.plan.refused.is_empty(), "{:?}", p.plan.refused);
    assert!(p.skipped.is_empty(), "{:?}", p.skipped);
    let got = table(&p);
    let n = |v: usize| v as u64;
    let (e, q, k, ff, eff, sh, experts) = (n(E), n(Q), n(K), n(FF), n(EFF), n(SHARED), n(EXPERTS));
    let want: BTreeMap<String, (Vec<u64>, GgmlType)> = [
        ("token_embd.weight", (vec![e, n(VOCAB)], GgmlType::BF16)),
        ("output.weight", (vec![e, n(VOCAB)], GgmlType::BF16)),
        ("output_norm.weight", (vec![e], GgmlType::F32)),
    ]
    .into_iter()
    .map(|(s, v)| (s.to_string(), v))
    .chain((0..=MTP).flat_map(|b| {
        let mtp = b == MTP;
        let attn = if mtp { GgmlType::BF16 } else { GgmlType::Q8_0 };
        let mut layer: Vec<(&str, (Vec<u64>, GgmlType))> = vec![
            ("attn_norm.weight", (vec![e], GgmlType::F32)),
            ("ffn_norm.weight", (vec![e], GgmlType::F32)),
            ("attn_q_norm.weight", (vec![n(HEAD_DIM)], GgmlType::F32)),
            ("attn_k_norm.weight", (vec![n(HEAD_DIM)], GgmlType::F32)),
            ("attn_q.weight", (vec![e, q], attn)),
            ("attn_k.weight", (vec![e, k], attn)),
            ("attn_v.weight", (vec![e, k], attn)),
            ("attn_output.weight", (vec![q, e], attn)),
        ];
        let dense = b == 0 || b == MTP;
        if dense {
            let ty = if mtp { GgmlType::BF16 } else { GgmlType::Q8_0 };
            layer.extend([
                ("ffn_gate.weight", (vec![e, ff], ty)),
                ("ffn_up.weight", (vec![e, ff], ty)),
                ("ffn_down.weight", (vec![ff, e], ty)),
            ]);
        } else {
            layer.extend([
                ("ffn_gate_inp.weight", (vec![e, experts], GgmlType::F32)),
                ("exp_probs_b.bias", (vec![experts], GgmlType::F32)),
                (
                    "ffn_gate_exps.weight",
                    (vec![e, eff, experts], GgmlType::FP8_B128),
                ),
                (
                    "ffn_up_exps.weight",
                    (vec![e, eff, experts], GgmlType::FP8_B128),
                ),
                (
                    "ffn_down_exps.weight",
                    (vec![eff, e, experts], GgmlType::FP8_B128),
                ),
                ("ffn_gate_shexp.weight", (vec![e, sh], GgmlType::Q8_0)),
                ("ffn_up_shexp.weight", (vec![e, sh], GgmlType::Q8_0)),
                ("ffn_down_shexp.weight", (vec![sh, e], GgmlType::Q8_0)),
            ]);
        }
        if b == MTP {
            layer.extend([
                ("nextn.eh_proj.weight", (vec![2 * e, e], GgmlType::BF16)),
                ("nextn.enorm.weight", (vec![e], GgmlType::F32)),
                ("nextn.hnorm.weight", (vec![e], GgmlType::F32)),
                ("nextn.shared_head_norm.weight", (vec![e], GgmlType::F32)),
            ]);
        }
        layer
            .into_iter()
            .map(move |(s, v)| (format!("blk.{b}.{s}"), v))
    }))
    .collect();
    assert_eq!(
        got.keys().collect::<Vec<_>>(),
        want.keys().collect::<Vec<_>>()
    );
    for (name, want) in &want {
        assert_eq!(&got[name], want, "{name}");
    }
}

/// The MTP block's own tensors take the names llama.cpp gives the next-token layer — the input
/// projection `eh_proj` of `[enorm(e); hnorm(h)]`, the two input norms, the shared head's norm —
/// and the block's attention and dense feed-forward tensors the same names as every layer's.
/// Mutants: `mtp.norm` mapped to `hnorm`, `fc` read as a block tensor of its own stem.
#[test]
fn the_mtp_block_takes_the_nextn_names() {
    let p = plan(&toy_config(), &toy_entries()).unwrap();
    let got = table(&p);
    let n = |v: usize| v as u64;
    assert_eq!(
        got["blk.4.nextn.eh_proj.weight"],
        (vec![2 * n(E), n(E)], GgmlType::BF16)
    );
    for stem in ["enorm", "hnorm", "shared_head_norm"] {
        assert_eq!(
            got[&format!("blk.4.nextn.{stem}.weight")],
            (vec![n(E)], GgmlType::F32),
            "{stem}"
        );
    }
    // The MTP block's attention and dense block are a layer's own, at its dims.
    assert_eq!(
        got["blk.4.attn_q.weight"],
        (vec![n(E), n(Q)], GgmlType::BF16)
    );
    assert_eq!(
        got["blk.4.ffn_down.weight"],
        (vec![n(FF), n(E)], GgmlType::BF16)
    );
    assert!(!got.contains_key("blk.4.ffn_gate_inp.weight"));
    assert!(!got.contains_key("nextn.eh_proj.weight"));
    assert!(!got.contains_key("blk.4.nextn.embed_tokens.weight"));
}

/// A config the map refuses, by the name of the key: another model's, an attention kind that
/// disagrees with its window, a window list of another length, a dense entry after a sparse one,
/// every block dense, a negative clamp, MTP lists of another length or kind, a softmax router,
/// another rope type, a tied head, an unknown key and a count of another shape.
#[test]
fn a_config_of_another_shape_is_refused_by_key() {
    type Case = (&'static str, Box<dyn Fn(&mut Json)>, &'static str);
    let cases: Vec<Case> = vec![
        (
            "another model_type",
            Box::new(|c| c["model_type"] = json!("exaone_moe_v2")),
            "model_type",
        ),
        (
            "another architecture",
            Box::new(|c| c["architectures"] = json!(["ExaoneMoeForSequenceClassification"])),
            "architectures",
        ),
        (
            "a kind that disagrees with its window",
            Box::new(|c| c["layer_types"][0] = json!("sliding_attention")),
            "entry 0",
        ),
        (
            "a window list of another length",
            Box::new(|c| {
                c["sliding_windows"].as_array_mut().unwrap().pop();
            }),
            "sliding_windows hold",
        ),
        (
            "an attention kind no model has",
            Box::new(|c| c["layer_types"][0] = json!("linear_attention")),
            "entry 0",
        ),
        (
            "a dense entry after a sparse one",
            Box::new(|c| c["mlp_layer_types"][2] = json!("dense")),
            "entry 2",
        ),
        (
            "every block dense",
            Box::new(|c| {
                c["mlp_layer_types"] = json!(["dense", "dense", "dense", "dense"]);
            }),
            "holds no routed block",
        ),
        (
            "a negative clamp",
            Box::new(|c| c["swiglu_limits"][2] = json!(-7.0)),
            "entry 2",
        ),
        (
            "a clamp list of another length",
            Box::new(|c| {
                c["swiglu_limits"].as_array_mut().unwrap().pop();
            }),
            "swiglu_limits hold",
        ),
        (
            "an MTP kind list of another length",
            Box::new(|c| c["mtp_layer_types"] = json!([])),
            "mtp_layer_types",
        ),
        (
            "an MTP block that attends fully",
            Box::new(|c| c["mtp_layer_types"] = json!(["full_attention"])),
            "mtp_layer_types",
        ),
        (
            "an MTP window of zero",
            Box::new(|c| c["mtp_sliding_windows"] = json!([0])),
            "mtp_sliding_windows",
        ),
        (
            "a softmax router",
            Box::new(|c| c["scoring_func"] = json!("softmax")),
            "scoring_func",
        ),
        (
            "an aux-loss router",
            Box::new(|c| c["topk_method"] = json!("aux_loss")),
            "topk_method",
        ),
        (
            "a scaled rope",
            Box::new(|c| c["rope_parameters"]["rope_type"] = json!("yarn")),
            "rope_type",
        ),
        (
            "a head tied to the embedding",
            Box::new(|c| c["tie_word_embeddings"] = json!(true)),
            "tie_word_embeddings",
        ),
        ("an unknown key", Box::new(|c| c["zzz"] = json!(1)), "zzz"),
        (
            "heads that do not group",
            Box::new(|c| {
                c["num_attention_heads"] = json!(3);
                c["num_key_value_heads"] = json!(2);
            }),
            "must group",
        ),
        (
            "more experts kept than routed over",
            Box::new(|c| c["num_experts_per_tok"] = json!(9)),
            "exceeds num_experts",
        ),
    ];
    for (what, edit, key) in cases {
        let mut c = toy_config();
        edit(&mut c);
        let e = Checkpoint::parse(&c).unwrap_err().to_string();
        assert!(e.contains(key), "{what}: {e}");
    }
}

/// A folder of another shape than the config's is refused by the tensor's name: a tensor no
/// block of this config holds (a router on the dense block, a shared expert there, a dense
/// projection on a routed one, a vision tower), a tensor a block lacks, a layer past the count,
/// a shape off the config's, a stack short of an expert, a duplicate expert id and a name no
/// K-EXAONE folder has.
#[test]
fn a_folder_of_another_shape_is_refused_by_tensor() {
    use Dtype::{Bf16, F8E4M3, F32};
    let layer = |l: usize, rest: &str| format!("model.layers.{l}.{rest}");
    let without = |name: String| {
        toy_entries()
            .into_iter()
            .filter(|(n, _, _)| *n != name)
            .collect()
    };
    let with = |name: String, dtype: Dtype, shape: Vec<usize>| {
        let mut e = toy_entries();
        e.push((name, dtype, shape));
        e
    };
    let reshaped = |name: String, shape: Vec<usize>| {
        let mut e = toy_entries();
        e.iter_mut().find(|(n, _, _)| *n == name).unwrap().2 = shape;
        e
    };
    let cases: Vec<(&str, Json, Entries, &str)> = vec![
        (
            "a router on the dense block",
            toy_config(),
            with(layer(0, "mlp.gate.weight"), F32, vec![EXPERTS, E]),
            "is a tensor no block of this config holds",
        ),
        (
            "a shared expert on the dense block",
            toy_config(),
            with(
                layer(0, "mlp.shared_experts.gate_proj.weight"),
                F8E4M3,
                vec![SHARED, E],
            ),
            "a routed block's shared experts",
        ),
        (
            "a dense projection on a routed block",
            toy_config(),
            with(layer(1, "mlp.gate_proj.weight"), F8E4M3, vec![FF, E]),
            "a dense block's projection",
        ),
        (
            "a vision tower",
            toy_config(),
            with("vision_tower.proj.weight".into(), Bf16, vec![8, 8]),
            "not a K-EXAONE tensor",
        ),
        (
            "a block without its query projection",
            toy_config(),
            toy_entries()
                .into_iter()
                .filter(|(n, _, _)| {
                    *n != layer(2, "self_attn.q_proj.weight")
                        && *n != layer(2, "self_attn.q_proj.weight_scale_inv")
                })
                .collect(),
            "blk.2.attn_q.weight: is missing from the folder",
        ),
        (
            "a scale grid whose weight is gone",
            toy_config(),
            without(layer(2, "self_attn.q_proj.weight")),
            "q_proj.weight_scale_inv: not a K-EXAONE tensor",
        ),
        (
            "the MTP block without its input projection",
            toy_config(),
            without("mtp.fc.weight".into()),
            "blk.4.nextn.eh_proj.weight: is missing from the folder",
        ),
        (
            "a routed block without its bias",
            toy_config(),
            without(layer(3, "mlp.e_score_correction_bias")),
            "blk.3.exp_probs_b.bias: is missing from the folder",
        ),
        (
            "a layer past the config's",
            toy_config(),
            with(layer(LAYERS, "input_layernorm.weight"), Bf16, vec![E]),
            "layer 4 of 4",
        ),
        (
            "an embedding of another width",
            toy_config(),
            reshaped("model.embed_tokens.weight".into(), vec![VOCAB, E / 2]),
            "holds dims [128, 8] where the config gives [256, 8]",
        ),
        (
            "a stack short of an expert",
            toy_config(),
            toy_entries()
                .into_iter()
                .filter(|(n, _, _)| !n.contains("layers.2.mlp.experts.7."))
                .collect(),
            "the stack's 7 expert ids are not 0..8",
        ),
        (
            "a duplicate expert id",
            toy_config(),
            with(
                layer(2, "mlp.experts.0.gate_proj.weight"),
                F8E4M3,
                vec![EFF, E],
            ),
            "duplicate expert id 0",
        ),
        (
            "a name no K-EXAONE folder has",
            toy_config(),
            with(
                layer(1, "mlp.switch_mlp.gate_proj.weight"),
                Bf16,
                vec![8, 8],
            ),
            "not a K-EXAONE tensor",
        ),
    ];
    for (what, config, entries, want) in cases {
        let e = refusal(&config, &entries);
        assert!(e.contains(want), "{what}: {e}");
    }
}

/// A safetensors file of `values` (name, dtype, shape, bytes).
fn stored(values: &[(String, Dtype, Vec<usize>, Vec<u8>)]) -> Safetensors {
    let list: Vec<(&str, &str, &[usize], &[u8])> = values
        .iter()
        .map(|(n, d, s, b)| (n.as_str(), d.name(), s.as_slice(), b.as_slice()))
        .collect();
    Safetensors::parse(safetensors::write(&list)).unwrap()
}

/// The planned tensor `name` made of the stored `st`, as `prepare` makes it: its sources
/// normalized in the plan's place, then the map's moves.
fn made(
    p: &folder::Planning<Infallible>,
    st: &Safetensors,
    name: &str,
) -> Result<hf::modelopt::Normal, String> {
    let folder::Make::Tensor(m) = &p
        .planned
        .iter()
        .find(|t| t.decl.name == name)
        .unwrap_or_else(|| panic!("{name} is not planned"))
        .make
    else {
        panic!("{name} is a tensor");
    };
    let placed = m.placed(&p.plan).map_err(|e| e.to_string())?;
    hf::modelopt::normalize(|s| st.raw(s), &placed, m.role).map_err(|e| e.to_string())
}

/// A routed stack's FP8_B128 blocks are the checkpoint's: each row's 128-value block carries the
/// f32 scale of the 128-by-128 square it lies in — row `r`'s block `j` the grid's
/// `[r / 128, j]` — before its 128 e4m3fn codes, the experts in id order. Mutants: the scale
/// read from the row alone, the codes written before the scale, the experts stacked by name.
#[test]
fn an_fp8_stack_makes_its_checkpoint_blocks() {
    let config = toy_config();
    let p = plan(&config, &toy_entries()).unwrap();
    // Expert `e`'s gate projection: codes `e + 1` a value, the grid's scale of square [r, c]
    // distinct and exact in f32.
    let code = |e: usize, _r: usize, c: usize| (e + 1 + c % 100) as u8;
    let scale = |r: usize, c: usize| 0.5 * (r + 1) as f32 + 0.25 * c as f32;
    let mut values = vec![];
    for e in 0..EXPERTS {
        let mut codes = Vec::with_capacity(EFF * E);
        for r in 0..EFF {
            for c in 0..E {
                codes.push(code(e, r, c));
            }
        }
        let mut scales = Vec::with_capacity((EFF / 128) * (E / 128) * 4);
        for r in 0..EFF / 128 {
            for c in 0..E / 128 {
                scales.extend(scale(r, c).to_le_bytes());
            }
        }
        values.push((
            format!("model.layers.2.mlp.experts.{e}.gate_proj.weight"),
            Dtype::F8E4M3,
            vec![EFF, E],
            codes,
        ));
        values.push((
            format!("model.layers.2.mlp.experts.{e}.gate_proj.weight_scale_inv"),
            Dtype::F32,
            vec![EFF / 128, E / 128],
            scales,
        ));
    }
    let n = made(&p, &stored(&values), "blk.2.ffn_gate_exps.weight").unwrap();
    assert_eq!(n.type_id, GgmlType::FP8_B128.as_u32());
    assert_eq!(n.dims, vec![E as u64, EFF as u64, EXPERTS as u64]);
    let row = EFF as u64 * (E / 128) as u64 * 132;
    assert_eq!(n.bytes.len() as u64, row * EXPERTS as u64);
    let block = |e: usize, r: usize, j: usize| {
        let at = (e as u64 * row + r as u64 * 264 + j as u64 * 132) as usize;
        (&n.bytes[at..at + 4], &n.bytes[at + 4..at + 132])
    };
    for (e, r, j) in [(0usize, 0, 0), (3, 300, 1), (7, 511, 0)] {
        let (bytes, codes) = block(e, r, j);
        assert_eq!(
            f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            scale(r / 128, j),
            "expert {e} row {r} block {j}"
        );
        let want: Vec<u8> = (j * 128..(j + 1) * 128).map(|c| code(e, r, c)).collect();
        assert_eq!(codes, &want[..], "expert {e} row {r} block {j}");
    }
}

/// The table the config derives of what a folder stores ([`FolderMap::stored`]) holds the real
/// model's shape: one hundred eighteen thousand three hundred ten tensors — the published
/// per-expert count among them — with no name the map skips. The whole shape plans with no
/// refusal into its one thousand two hundred fifty-six tensors, of the bytes the config's widths
/// and the map's roles give. Mutants: an expert dropped from the stored table, a stack's role
/// changed to `Q8_0`.
#[test]
fn the_real_shape_derives_its_stored_table_and_plans() {
    let config = real_config();
    let map = Checkpoint::parse(&config).unwrap();
    let stored = map.stored().unwrap();
    assert!(stored.tensors.len() == 118_310, "{}", stored.tensors.len());
    let experts = stored
        .tensors
        .iter()
        .filter(|(n, _, _)| n.contains(".mlp.experts."))
        .count();
    assert_eq!(experts, 76 * 256 * 6);
    let p = plan(&config, &stored.tensors).unwrap();
    assert!(p.plan.refused.is_empty(), "{:?}", p.plan.refused);
    assert!(p.skipped.is_empty(), "{:?}", p.skipped);
    assert_eq!(p.planned.len(), 1_256);
    // The split's bytes [derived] from the config's widths and the map's roles: an FP8 stack a
    // block of 132 bytes, a Q8_0 row 34 a 32 values, F32 four bytes a value and BF16 two.
    let (e, q, k, ff, eff, sh, experts) = (6144usize, 8192, 1024, 18432, 2048, 2048, 256usize);
    let f32n = |n: usize| (n * 4) as u64;
    let bf16 = |n: usize| (n * 2) as u64;
    let q8 = |rows: usize, ne0: usize| (rows * (ne0 / 32) * 34) as u64;
    let fp8 = |rows: usize, cols: usize, n: usize| (rows * (cols / 128) * 132 * n) as u64;
    let attention = || q8(q, e) + 2 * q8(k, e) + q8(e, q);
    let layer = |routed: bool| {
        2 * f32n(e)
            + 2 * f32n(128)
            + attention()
            + if routed {
                f32n(e * experts)
                    + f32n(experts)
                    + 3 * fp8(eff, e, experts)
                    + 2 * q8(sh, e)
                    + q8(e, sh)
            } else {
                2 * q8(ff, e) + q8(e, ff)
            }
    };
    let want = 2 * bf16(153_600 * e)
        + f32n(e)
        + 2 * layer(false)
        + 76 * layer(true)
        + (2 * f32n(e)
            + 2 * f32n(128)
            + bf16(q * e)
            + 2 * bf16(k * e)
            + bf16(e * q)
            + 2 * bf16(ff * e)
            + bf16(e * ff)
            + bf16(e * 2 * e)
            + 3 * f32n(e));
    let bytes: u64 = p.planned.iter().map(|t| t.decl.nbytes).sum();
    assert_eq!(bytes, want);
}

/// The toy folder opens as `bloomery-quantize`'s `--hf` source: the same planned tensors, in the
/// same types, with the tokenizer checked against Hugging Face's. Mutants: a trunk role changed
/// to `Matrix` (the source then carries FP8_B128 at a dense site), the head kept BF16 by the
/// bootstrap rules' absence.
#[test]
fn the_toy_folder_opens_as_the_quantizer_s_source() {
    let dir = temp_dir("exaone-source");
    write_folder(&dir, &toy_bytes(), &toy_config(), &tokenizer("exaone-moe"));
    let input = quantize::cli::HfInput {
        dir: dir.clone(),
        arch: None,
        drop: vec![],
        shard_gib: 48,
    };
    let source = crate::quantize_source::open(&input).unwrap();
    assert_eq!(source.architecture(), Some(ARCHITECTURE));
    let tensors = source.tensors();
    assert_eq!(tensors.len(), 77);
    let mut by_type: BTreeMap<GgmlType, usize> = BTreeMap::new();
    for (_, t) in &tensors {
        *by_type.entry(t.ty).or_default() += 1;
    }
    assert_eq!(
        by_type,
        BTreeMap::from([
            (GgmlType::F32, 30),
            (GgmlType::Q8_0, 28),
            (GgmlType::BF16, 10),
            (GgmlType::FP8_B128, 9),
        ])
    );
    let find = |name: &str| tensors.iter().find(|(_, t)| t.name == name).unwrap();
    assert_eq!(
        find("blk.2.ffn_gate_exps.weight").1.dims,
        vec![E as u64, EFF as u64, EXPERTS as u64]
    );
    assert_eq!(find("blk.4.nextn.eh_proj.weight").1.ty, GgmlType::BF16);
    assert_eq!(find("token_embd.weight").1.ty, GgmlType::BF16);
    assert_eq!(find("blk.1.attn_q.weight").1.ty, GgmlType::Q8_0);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The toy folder's tensors with real FP8 bytes and scales: the codes distinct and below the
/// e4m3fn NaN ones, the scales of the checkpoint's grid, the BF16 tensors any bytes.
fn toy_bytes() -> Vec<(String, &'static str, Vec<usize>, Vec<u8>)> {
    let entries = toy_entries();
    entries
        .into_iter()
        .map(|(name, dtype, shape)| {
            let n = shape.iter().product::<usize>();
            let bytes: Vec<u8> = match dtype {
                Dtype::F8E4M3 => (0..n).map(|i| (i % 100 + 1) as u8).collect(),
                Dtype::F32 if name.ends_with("weight_scale_inv") => (0..n)
                    .flat_map(|i| (0.5 + (i % 7) as f32).to_le_bytes())
                    .collect(),
                Dtype::F32 => (0..n)
                    .flat_map(|i| ((i % 5) as f32 - 2.0).to_le_bytes())
                    .collect(),
                Dtype::Bf16 => (0..n * 2).map(|i| (i % 251) as u8).collect(),
                other => panic!("{name}: {other:?}"),
            };
            (name, dtype.name(), shape, bytes)
        })
        .collect()
}

/// The join: the header the map writes for the toy folder is read by the registry's reader (the
/// row `exaone-moe` names, the flint one) into the per-layer description the reader's own tests
/// pin — the windows and the rope exactly where windowed, the dense lead, the clamp act on the
/// routed and the shared experts, the biased sigmoid router, the MTP block in `spec.mtp` with
/// its nextn tensors — beside the tokenizer keys the folder's files give it: `pre` the pinned
/// `exaone-moe` row and the chat template the folder's jinja file spells. Mutant (red at the
/// layer-1 window): the map's window array entry of the wide window narrowed.
#[test]
fn the_map_s_header_reads_through_the_registry() {
    let dir = temp_dir("exaone-join");
    write_folder(&dir, &toy_bytes(), &toy_config(), &tokenizer("exaone-moe"));
    let template = "<|assistant|>\n<tool_call>\n<function=name>\n</function>\n</tool_call>\n";
    std::fs::write(dir.join("chat_template.jinja"), template).unwrap();
    let opened = crate::folder::open(&dir).unwrap();
    let read = model::arch::registry::row_of(ARCHITECTURE)
        .and_then(|row| row.read)
        .expect("the exaone-moe row holds a reader")(&opened.header)
    .unwrap();
    let s = &read.spec;
    assert_eq!(s.arch, model::arch::models::Arch::ExaoneMoe);
    assert_eq!(
        (s.hidden, s.vocab, s.ctx_train),
        (E as u32, VOCAB as u32, 4096)
    );
    assert_eq!(s.layers.len(), LAYERS);
    assert_eq!(s.mtp.len(), 1);
    assert_eq!(s.chat.pre, "exaone-moe");
    assert_eq!(
        opened.header.value("tokenizer.chat_template"),
        Some(&Value::String(template.into()))
    );
    let want_window = [None, Some(4096), Some(128), None, Some(128)];
    for (l, layer) in s.layers.iter().chain(&s.mtp).enumerate() {
        let model::arch::models::Mixer::Gqa(g) = &layer.mixer else {
            panic!("layer {l} not GQA");
        };
        assert_eq!(
            (g.heads, g.kv_heads, g.head_dim, g.value_dim),
            (HEADS as u32, KV as u32, HEAD_DIM as u32, HEAD_DIM as u32),
            "layer {l}"
        );
        assert_eq!(g.window, want_window[l], "layer {l}");
        let roped = want_window[l].is_some();
        assert_eq!(
            (
                g.rope.mode,
                g.rope.dims,
                if roped { g.rope.base.to_bits() } else { 0 }
            ),
            if roped {
                (
                    model::arch::models::RopeMode::Neox,
                    HEAD_DIM as u32,
                    1e6_f32.to_bits(),
                )
            } else {
                (model::arch::models::RopeMode::None, 0, 0)
            },
            "layer {l}"
        );
        assert!(g.qk_norm && !g.sinks, "layer {l}");
        let want_act = if [2usize, 3].contains(&l) {
            model::arch::models::Act::SwiGluGateFirst { limit: 7.0 }
        } else {
            model::arch::models::Act::SwiGlu { limit: None }
        };
        match &layer.ffn {
            model::arch::models::Ffn::Dense { ff, act } => {
                assert!(l == 0 || l == MTP, "layer {l} dense");
                assert_eq!(*ff, FF as u32);
                assert_eq!(
                    *act,
                    model::arch::models::Act::SwiGlu { limit: None },
                    "layer {l}"
                );
            }
            model::arch::models::Ffn::Moe(m) => {
                assert!((1..LAYERS).contains(&l), "layer {l} routed");
                assert_eq!(
                    (m.experts, m.top_k, m.expert_ff),
                    (EXPERTS as u32, 2, EFF as u32)
                );
                assert_eq!(m.act, want_act, "layer {l}");
                assert_eq!(
                    m.router,
                    model::arch::models::Router {
                        score: model::arch::models::Score::Sigmoid,
                        bias: true,
                        norm: model::arch::models::RouterNorm::SelectedSum,
                        scale: 2.5,
                        hash: false,
                    },
                    "layer {l}"
                );
                let shared = m.shared.as_ref().unwrap();
                assert_eq!(shared.ff, SHARED as u32);
                assert_eq!(shared.act, want_act, "layer {l}");
            }
        }
    }
    for name in ["eh_proj", "enorm", "hnorm", "shared_head_norm"] {
        assert!(
            opened
                .header
                .find(&format!("blk.{MTP}.nextn.{name}.weight"))
                .is_some(),
            "{name}"
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// `prepare --plan` of the toy folder names the architecture and its tensors by place and type
/// — the FP8 stacks made at load, the trunk made at load, nothing read in place — with the
/// tokenizer checked against Hugging Face's. Mutants: a role changed to `File` (a tensor then
/// read in place).
#[test]
fn the_toy_folder_prints_a_prepare_plan() {
    let dir = temp_dir("exaone-prepare-plan");
    write_folder(&dir, &toy_bytes(), &toy_config(), &tokenizer("exaone-moe"));
    let opened = crate::folder::open(&dir).unwrap();
    assert_eq!(opened.header.architecture(), Some(ARCHITECTURE));
    let lines = folder::plan_lines(&opened.header, &opened.skipped);
    assert_eq!(
        lines[0],
        format!(
            "exaone-moe: 77 tensors, {} bytes; 0 folder tensors skipped",
            opened
                .header
                .iter_tensors()
                .map(|(_, t)| t.nbytes)
                .sum::<u64>()
        )
    );
    assert!(
        lines.iter().any(|l| l.contains("made at load")),
        "{lines:?}"
    );
    assert!(lines.iter().any(|l| l.contains("fp8_b128")), "{lines:?}");
    assert_eq!(opened.tokenizer.n_vocab(), VOCAB);
    std::fs::remove_dir_all(&dir).unwrap();
}
