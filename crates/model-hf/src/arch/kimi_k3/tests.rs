use std::collections::BTreeMap;
use std::path::PathBuf;

use gguf::write::Layout;
use hf::modelopt;
use hf::quant_config::Config;
use model::arch::kimi_k3::hparams::Hparams;
use safetensors::{Dtype, Safetensors};
use serde_json::json;

use super::*;
use crate::folder::{self, Keep};
use crate::test_support::{Entries, json};

/// The widths the K3 description's own tests use: 4 layers (KDA at 0, 1 and 3, a latent one at 2,
/// the first dense, a bank every 2), 2 heads, 4 experts of 256 in a latent of 512, and a
/// vocabulary of 2. Every first dim is a whole block of the type the released set holds its stem in.
const E: usize = 512;
const HEADS: usize = 2;
const D: usize = 128;
const CONV: usize = 4;
const Q_LORA: usize = 256;
const KV_LORA: usize = 512;
const NOPE: usize = 128;
const PE: usize = 64;
const HEAD_V: usize = 128;
const LAYERS: usize = 4;
const FF: usize = 768;
const N_EXP: usize = 4;
const EFF: usize = 256;
const SHARED: usize = 2;
const LATENT: usize = 512;
const VOCAB: usize = 2;
/// The indexer's geometry on a K3.1 folder: the heads and key width the description's kernels run
/// (the reader refuses any other) and the kept positions.
const INDEX_HEADS: usize = 32;
const INDEX_DIM: usize = 128;
/// `A_log` is stored past the heads.
const A_LOG_LEN: usize = 4;
const INNER: usize = HEADS * D;
const LATENT_LAYER: usize = 2;

/// A committed real config, `kimi_k3` or `kimi_k31`.
fn real(k31: bool) -> Json {
    json(if k31 {
        "quant-configs/moonshotai__Kimi-K3.1-Pro__config.json.gz"
    } else {
        "quant-configs/moonshotai__Kimi-K3__config.json.gz"
    })
}

/// The real config at the toy widths of this file.
fn toy_config(k31: bool) -> Json {
    let mut c = real(k31);
    let t = &mut c["text_config"];
    for (key, v) in [
        ("hidden_size", E),
        ("intermediate_size", FF),
        ("num_hidden_layers", LAYERS),
        ("num_attention_heads", HEADS),
        ("num_key_value_heads", HEADS),
        ("q_lora_rank", Q_LORA),
        ("kv_lora_rank", KV_LORA),
        ("qk_nope_head_dim", NOPE),
        ("qk_rope_head_dim", PE),
        ("v_head_dim", HEAD_V),
        ("first_k_dense_replace", 1),
        ("moe_intermediate_size", EFF),
        ("num_experts", N_EXP),
        ("num_experts_per_token", 2),
        ("num_shared_experts", SHARED),
        ("routed_expert_hidden_size", LATENT),
        ("attn_res_block_size", 2),
        ("vocab_size", VOCAB),
    ] {
        t[key] = json!(v);
    }
    t["linear_attn_config"]["num_heads"] = json!(HEADS);
    t["linear_attn_config"]["head_dim"] = json!(D);
    t["linear_attn_config"]["full_attn_layers"] = json!([LATENT_LAYER + 1]);
    t["linear_attn_config"]["kda_layers"] = json!([1, 2, 4]);
    if k31 {
        t["dsa_config"]["dsa_layers"] = json!([LATENT_LAYER + 1]);
        t["dsa_config"]["index_num_heads"] = json!(INDEX_HEADS);
        t["dsa_config"]["index_head_dim"] = json!(INDEX_DIM);
        t["dsa_config"]["index_topk"] = json!(8);
    }
    for (key, id) in [
        ("bos_token_id", 0),
        ("eos_token_id", 1),
        ("pad_token_id", 1),
    ] {
        c[key] = json!(id);
        c["text_config"][key] = json!(id);
    }
    c
}

const LM: &str = "language_model.";

/// The folder's tensors at the toy widths, as the real checkpoint names and shapes them: name,
/// dtype and shape.
fn toy_entries(k31: bool) -> Entries {
    use Dtype::{Bf16, F32, U8};
    let mut t: Entries = vec![];
    let mut add =
        |name: String, dtype: Dtype, shape: &[usize]| t.push((name, dtype, shape.to_vec()));
    add(format!("{LM}model.embed_tokens.weight"), Bf16, &[VOCAB, E]);
    add(format!("{LM}lm_head.weight"), Bf16, &[VOCAB, E]);
    add(format!("{LM}model.norm.weight"), Bf16, &[E]);
    add(format!("{LM}model.output_attn_res_norm.weight"), Bf16, &[E]);
    add(
        format!("{LM}model.output_attn_res_proj.weight"),
        Bf16,
        &[1, E],
    );
    add("vision_tower.patch_embed.proj.weight".into(), Bf16, &[8, 8]);
    add("mm_projector.proj.0.weight".into(), Bf16, &[8, 8]);
    for l in 0..LAYERS {
        let p = format!("{LM}model.layers.{l}.");
        let mut layer = |rest: &str, dtype: Dtype, shape: &[usize]| {
            add(format!("{p}{rest}"), dtype, shape);
        };
        layer("input_layernorm.weight", Bf16, &[E]);
        layer("post_attention_layernorm.weight", Bf16, &[E]);
        layer("self_attention_res_norm.weight", Bf16, &[E]);
        layer("self_attention_res_proj.weight", Bf16, &[1, E]);
        layer("mlp_res_norm.weight", Bf16, &[E]);
        layer("mlp_res_proj.weight", Bf16, &[1, E]);
        if l == LATENT_LAYER {
            layer("self_attn.q_a_proj.weight", Bf16, &[Q_LORA, E]);
            layer("self_attn.q_a_layernorm.weight", Bf16, &[Q_LORA]);
            layer(
                "self_attn.q_b_proj.weight",
                Bf16,
                &[HEADS * (NOPE + PE), Q_LORA],
            );
            layer(
                "self_attn.kv_a_proj_with_mqa.weight",
                Bf16,
                &[KV_LORA + PE, E],
            );
            layer("self_attn.kv_a_layernorm.weight", Bf16, &[KV_LORA]);
            layer(
                "self_attn.kv_b_proj.weight",
                Bf16,
                &[HEADS * (NOPE + HEAD_V), KV_LORA],
            );
            layer("self_attn.g_proj.weight", Bf16, &[HEADS * HEAD_V, E]);
            layer("self_attn.o_proj.weight", Bf16, &[E, HEADS * HEAD_V]);
            if k31 {
                layer(
                    "self_attn.indexer.wq_b.weight",
                    Bf16,
                    &[INDEX_HEADS * INDEX_DIM, Q_LORA],
                );
                layer("self_attn.indexer.wk.weight", Bf16, &[INDEX_DIM, E]);
                layer(
                    "self_attn.indexer.weights_proj.weight",
                    Bf16,
                    &[INDEX_HEADS, E],
                );
                layer("self_attn.indexer.k_norm.weight", Bf16, &[INDEX_DIM]);
                layer("self_attn.indexer.k_norm.bias", Bf16, &[INDEX_DIM]);
            }
        } else {
            for proj in ["q_proj", "k_proj", "v_proj", "g_proj"] {
                layer(&format!("self_attn.{proj}.weight"), Bf16, &[INNER, E]);
            }
            for conv in ["q_conv1d", "k_conv1d", "v_conv1d"] {
                layer(&format!("self_attn.{conv}.weight"), F32, &[INNER, 1, CONV]);
            }
            layer("self_attn.f_a_proj.weight", Bf16, &[D, E]);
            layer("self_attn.f_b_proj.weight", Bf16, &[INNER, D]);
            layer("self_attn.b_proj.weight", Bf16, &[HEADS, E]);
            layer("self_attn.A_log", F32, &[A_LOG_LEN]);
            layer("self_attn.dt_bias", F32, &[INNER]);
            layer("self_attn.o_norm.weight", F32, &[D]);
            layer("self_attn.o_proj.weight", Bf16, &[E, INNER]);
        }
        if l == 0 {
            layer("mlp.gate_proj.weight", Bf16, &[FF, E]);
            layer("mlp.up_proj.weight", Bf16, &[FF, E]);
            layer("mlp.down_proj.weight", Bf16, &[E, FF]);
        } else {
            let m = "block_sparse_moe.";
            layer(&format!("{m}gate.weight"), Bf16, &[N_EXP, E]);
            layer(&format!("{m}gate.e_score_correction_bias"), F32, &[N_EXP]);
            layer(
                &format!("{m}routed_expert_down_proj.weight"),
                Bf16,
                &[LATENT, E],
            );
            layer(
                &format!("{m}routed_expert_up_proj.weight"),
                Bf16,
                &[E, LATENT],
            );
            layer(&format!("{m}routed_expert_norm.weight"), Bf16, &[LATENT]);
            layer(
                &format!("{m}shared_experts.gate_proj.weight"),
                Bf16,
                &[SHARED * EFF, E],
            );
            layer(
                &format!("{m}shared_experts.up_proj.weight"),
                Bf16,
                &[SHARED * EFF, E],
            );
            layer(
                &format!("{m}shared_experts.down_proj.weight"),
                Bf16,
                &[E, SHARED * EFF],
            );
            for e in 0..N_EXP {
                for (w, rows, k) in [
                    ("w1", EFF, LATENT),
                    ("w3", EFF, LATENT),
                    ("w2", LATENT, EFF),
                ] {
                    let x = format!("{m}experts.{e}.{w}.");
                    layer(&format!("{x}weight_packed"), U8, &[rows, k / 2]);
                    layer(&format!("{x}weight_scale"), U8, &[rows, k / 32]);
                }
            }
        }
    }
    t
}

/// The released set's table at the toy widths, from the description's own: each tensor of a K3
/// folder's header as ggml dims and the type this map declares it in. A K3.1 folder adds the
/// indexer's five on the latent layer.
fn header_table(k31: bool) -> BTreeMap<String, (Vec<u64>, GgmlType)> {
    use GgmlType::{BF16, F32, MXFP4};
    let n = |v: usize| v as u64;
    let (e, inner, d) = (n(E), n(INNER), n(D));
    let mut t: Vec<(String, Vec<u64>, GgmlType)> = vec![
        ("token_embd.weight".into(), vec![e, n(VOCAB)], BF16),
        ("output.weight".into(), vec![e, n(VOCAB)], BF16),
        ("output_norm.weight".into(), vec![e], F32),
        ("output_res_score.weight".into(), vec![e], F32),
    ];
    for l in 0..LAYERS {
        let mut layer: Vec<(&str, Vec<u64>, GgmlType)> = vec![
            ("attn_norm.weight", vec![e], F32),
            ("ffn_norm.weight", vec![e], F32),
            ("attn_res_score.weight", vec![e], F32),
            ("ffn_res_score.weight", vec![e], F32),
        ];
        if l == LATENT_LAYER {
            layer.extend([
                ("attn_q_a.weight", vec![e, n(Q_LORA)], BF16),
                ("attn_q_a_norm.weight", vec![n(Q_LORA)], F32),
                (
                    "attn_q_b.weight",
                    vec![n(Q_LORA), n(HEADS * (NOPE + PE))],
                    BF16,
                ),
                ("attn_kv_a_mqa.weight", vec![e, n(KV_LORA + PE)], BF16),
                ("attn_kv_a_norm.weight", vec![n(KV_LORA)], F32),
                ("attn_k_b.weight", vec![n(NOPE), n(KV_LORA), n(HEADS)], BF16),
                (
                    "attn_v_b.weight",
                    vec![n(KV_LORA), n(HEAD_V), n(HEADS)],
                    BF16,
                ),
                ("attn_gate.weight", vec![e, n(HEADS * HEAD_V)], BF16),
                ("attn_output.weight", vec![n(HEADS * HEAD_V), e], BF16),
            ]);
            if k31 {
                layer.extend([
                    (
                        "indexer.attn_q_b.weight",
                        vec![n(Q_LORA), n(INDEX_HEADS * INDEX_DIM)],
                        BF16,
                    ),
                    ("indexer.attn_k.weight", vec![e, n(INDEX_DIM)], BF16),
                    ("indexer.proj.weight", vec![e, n(INDEX_HEADS)], BF16),
                    ("indexer.k_norm.weight", vec![n(INDEX_DIM)], F32),
                    ("indexer.k_norm.bias", vec![n(INDEX_DIM)], F32),
                ]);
            }
        } else {
            layer.extend([
                ("attn_q.weight", vec![e, inner], BF16),
                ("attn_k.weight", vec![e, inner], BF16),
                ("attn_v.weight", vec![e, inner], BF16),
                ("ssm_conv1d_q.weight", vec![n(CONV), 1, inner], F32),
                ("ssm_conv1d_k.weight", vec![n(CONV), 1, inner], F32),
                ("ssm_conv1d_v.weight", vec![n(CONV), 1, inner], F32),
                ("ssm_f_a.weight", vec![e, d], BF16),
                ("ssm_f_b.weight", vec![d, inner], BF16),
                ("ssm_g.weight", vec![e, inner], BF16),
                ("ssm_beta.weight", vec![e, n(HEADS)], F32),
                ("ssm_a", vec![n(HEADS)], F32),
                ("ssm_dt.bias", vec![inner], F32),
                ("ssm_norm.weight", vec![d], F32),
                ("attn_output.weight", vec![inner, e], BF16),
            ]);
        }
        if l == 0 {
            layer.extend([
                ("ffn_gate.weight", vec![e, n(FF)], BF16),
                ("ffn_up.weight", vec![e, n(FF)], BF16),
                ("ffn_down.weight", vec![n(FF), e], BF16),
            ]);
        } else {
            let (latent, eff, shared) = (n(LATENT), n(EFF), n(SHARED * EFF));
            layer.extend([
                ("ffn_gate_inp.weight", vec![e, n(N_EXP)], F32),
                ("exp_probs_b.bias", vec![n(N_EXP)], F32),
                ("ffn_gate_exps.weight", vec![latent, eff, n(N_EXP)], MXFP4),
                ("ffn_up_exps.weight", vec![latent, eff, n(N_EXP)], MXFP4),
                ("ffn_down_exps.weight", vec![eff, latent, n(N_EXP)], MXFP4),
                ("ffn_routed_down.weight", vec![e, latent], BF16),
                ("ffn_routed_up.weight", vec![latent, e], BF16),
                ("ffn_routed_norm.weight", vec![latent], F32),
                ("ffn_gate_shexp.weight", vec![e, shared], BF16),
                ("ffn_up_shexp.weight", vec![e, shared], BF16),
                ("ffn_down_shexp.weight", vec![shared, e], BF16),
            ]);
        }
        t.extend(
            layer
                .into_iter()
                .map(|(s, d, ty)| (format!("blk.{l}.{s}"), d, ty)),
        );
    }
    t.into_iter().map(|(n, d, ty)| (n, (d, ty))).collect()
}

/// A plan of `entries` under `config`'s checkpoint map, in the load's `Keep::InPlace` and `q8`
/// off.
fn plan(config: &Json, entries: &Entries) -> Result<folder::Planning<Step>, folder::Error> {
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

/// The real configs of both checkpoints read, each into the header keys llama.cpp's converter
/// writes: 93 layers of 69 KDA and 24 latent (every 4th, and the last two), the first dense, 896
/// experts of 3072 in a latent of 3584, 16 used, and K3.1's indexer keys alone on its folder.
#[test]
fn the_real_configs_read_into_the_converters_keys() {
    for k31 in [false, true] {
        let c = Checkpoint::parse(&real(k31)).unwrap();
        assert_eq!(c.layers, 93);
        assert_eq!(c.latent_layer.iter().filter(|&&l| l).count(), 24);
        assert!(c.latent_layer[3] && c.latent_layer[7] && c.latent_layer[91] && c.latent_layer[92]);
        assert!(!c.latent_layer[0] && !c.latent_layer[1] && !c.latent_layer[90]);
        assert_eq!((c.indexer, c.dense_lead, c.experts), (k31, 1, 896));
        let key = |suffix: &str| {
            c.keys
                .iter()
                .find(|(k, _)| k == suffix)
                .map(|(_, v)| v.clone())
        };
        for (suffix, want) in [
            ("block_count", Value::U32(93)),
            ("embedding_length", Value::U32(7168)),
            ("vocab_size", Value::U32(163_840)),
            ("context_length", Value::U32(1_048_576)),
            ("feed_forward_length", Value::U32(33_792)),
            ("leading_dense_block_count", Value::U32(1)),
            ("attention.head_count", Value::U32(96)),
            ("attention.key_length", Value::U32(576)),
            ("attention.value_length", Value::U32(512)),
            ("attention.key_length_mla", Value::U32(192)),
            ("attention.value_length_mla", Value::U32(128)),
            ("attention.q_lora_rank", Value::U32(1536)),
            ("attention.kv_lora_rank", Value::U32(512)),
            ("attention.layer_norm_rms_epsilon", Value::F32(1e-5)),
            ("rope.dimension_count", Value::U32(64)),
            ("ssm.conv_kernel", Value::U32(4)),
            ("kda.head_dim", Value::U32(128)),
            ("kda.gate_lower_bound", Value::F32(-5.0)),
            ("expert_count", Value::U32(896)),
            ("expert_used_count", Value::U32(16)),
            ("expert_feed_forward_length", Value::U32(3072)),
            ("expert_shared_count", Value::U32(2)),
            ("expert_gating_func", Value::U32(2)),
            ("expert_weights_scale", Value::F32(1.0)),
            ("expert_weights_norm", Value::Bool(true)),
            ("expert_latent_length", Value::U32(3584)),
            ("attn_res.block_size", Value::U32(12)),
            ("activation.situ_beta", Value::F32(4.0)),
            ("activation.situ_linear_beta", Value::F32(25.0)),
        ] {
            assert_eq!(key(suffix), Some(want), "{suffix} (k31 {k31})");
        }
        let Some(Value::Array(kv)) = key("attention.head_count_kv") else {
            panic!("head_count_kv is an array");
        };
        let latent: Vec<usize> = (0..93).filter(|&l| kv[l] == Value::I32(1)).collect();
        assert_eq!(kv.len(), 93);
        assert_eq!(latent.len(), 24);
        assert!(kv.iter().all(|v| matches!(v, Value::I32(0 | 1))));
        assert_eq!(
            (latent[..3].to_vec(), latent[22..].to_vec()),
            (vec![3, 7, 11], vec![91, 92])
        );
        let indexer = ["head_count", "key_length", "top_k", "rope_dimension_count"]
            .map(|s| key(&format!("attention.indexer.{s}")));
        let types = key("attention.indexer.types");
        let eps = key("attention.layer_norm_epsilon");
        if k31 {
            assert_eq!(
                indexer,
                [32, 128, 2048, 0].map(|v| Some(Value::U32(v))),
                "the indexer's geometry and its no rope values"
            );
            let Some(Value::Array(layers)) = types else {
                panic!("attention.indexer.types is an array");
            };
            let on: Vec<bool> = layers.iter().map(|v| *v == Value::Bool(true)).collect();
            assert!(layers.iter().all(|v| matches!(v, Value::Bool(_))));
            assert_eq!(on, c.latent_layer, "the indexer is on the latent layers");
            assert_eq!(eps, Some(Value::F32(1e-5)), "the key LayerNorm's epsilon");
        } else {
            assert_eq!(indexer, [None, None, None, None]);
            assert_eq!((types, eps), (None, None));
        }
    }
}

/// The planned tensors by name: ggml dims and declared type.
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

/// A toy folder of either checkpoint plans from its headers alone: every tensor of the released
/// set's table (107 at four layers, K3.1's indexer adding five) in the kimi-k3 names at ggml dims
/// and the types this map declares (BF16 matrices as stored, F32 vectors, the router and the conv
/// taps, the routed stacks MXFP4), the vision tower and its projector skipped by reason, and
/// nothing left unplaced. `g_proj` is the KDA layers' `ssm_g` and the latent layer's `attn_gate`.
#[test]
fn a_toy_folder_plans_every_tensor_at_the_released_names_and_dims() {
    for k31 in [false, true] {
        let p = plan(&toy_config(k31), &toy_entries(k31)).unwrap();
        assert!(p.plan.refused.is_empty(), "{:?}", p.plan.refused);
        let (got, want) = (table(&p), header_table(k31));
        assert_eq!(got.len(), if k31 { 112 } else { 107 });
        assert_eq!(
            got.keys().collect::<Vec<_>>(),
            want.keys().collect::<Vec<_>>(),
            "k31 {k31}"
        );
        for (name, want) in &want {
            assert_eq!(&got[name], want, "{name} (k31 {k31})");
        }
        assert!(got["blk.1.ssm_g.weight"].0 == [E as u64, INNER as u64]);
        assert!(got["blk.2.attn_gate.weight"].0 == [E as u64, (HEADS * HEAD_V) as u64]);
        assert!(
            !got.contains_key("blk.2.ssm_g.weight") && !got.contains_key("blk.1.attn_gate.weight")
        );
        assert_eq!(
            p.skipped.keys().map(String::as_str).collect::<Vec<_>>(),
            [
                "mm_projector.proj.0.weight",
                "vision_tower.patch_embed.proj.weight"
            ]
        );
        assert!(p.skipped.values().all(|why| why.contains("vision tower")));
    }
}

/// Each cross-layer residual pair is one score vector made of its own norm and projection, in
/// that order: a layer's `self_attention_res_*` pair is its attention score, its `mlp_res_*` pair
/// its feed-forward score, and the final pair is the head's. Mutants: the attention and
/// feed-forward pairs swapped, the final pair read from a layer's.
#[test]
fn each_residual_pair_makes_its_own_score_vector() {
    let p = plan(&toy_config(false), &toy_entries(false)).unwrap();
    let made = |name: &str| match &p.planned.iter().find(|t| t.decl.name == name).unwrap().make {
        folder::Make::Tensor(m) => (m.sources.clone(), m.steps.clone(), m.dims.clone()),
        _ => panic!("{name} is a tensor"),
    };
    let pair = |stem: &str| {
        vec![
            format!("{LM}model.{stem}_norm.weight"),
            format!("{LM}model.{stem}_proj.weight"),
        ]
    };
    for l in [0usize, 1, 3] {
        let layer = |stem: &str| {
            vec![
                format!("{LM}model.layers.{l}.{stem}_norm.weight"),
                format!("{LM}model.layers.{l}.{stem}_proj.weight"),
            ]
        };
        let steps = vec![Step::ResProduct];
        assert_eq!(
            made(&format!("blk.{l}.attn_res_score.weight")),
            (layer("self_attention_res"), steps.clone(), vec![E as u64])
        );
        assert_eq!(
            made(&format!("blk.{l}.ffn_res_score.weight")),
            (layer("mlp_res"), steps, vec![E as u64])
        );
    }
    assert_eq!(
        made("output_res_score.weight"),
        (
            pair("output_attn_res"),
            vec![Step::ResProduct],
            vec![E as u64]
        )
    );
}

/// The K3.1 folder's indexer tensors are the names the K3 description spells (the ones
/// Deepseek-V4.1's and Mistral Large 4's give theirs), one tensor each on the latent layer alone,
/// and its header carries the indexer's keys in the spelling the description reads.
#[test]
fn the_k31_indexer_takes_the_names_the_k3_description_spells() {
    use model::arch::kimi_k3::names as k3;
    let p = plan(&toy_config(true), &toy_entries(true)).unwrap();
    let got = table(&p);
    let l = LATENT_LAYER;
    let (index_q, index_k, index_proj, index_norm, bias) = (
        k3::indexer_attn_q_b(l),
        k3::indexer_attn_k(l),
        k3::indexer_proj(l),
        k3::indexer_k_norm(l),
        k3::indexer_k_norm_bias(l),
    );
    assert_eq!(
        [&index_q, &index_k, &index_proj, &index_norm, &bias].map(|n| got.get(n).cloned()),
        [
            Some((
                vec![Q_LORA as u64, (INDEX_HEADS * INDEX_DIM) as u64],
                GgmlType::BF16
            )),
            Some((vec![E as u64, INDEX_DIM as u64], GgmlType::BF16)),
            Some((vec![E as u64, INDEX_HEADS as u64], GgmlType::BF16)),
            Some((vec![INDEX_DIM as u64], GgmlType::F32)),
            Some((vec![INDEX_DIM as u64], GgmlType::F32)),
        ]
    );
    let indexed: Vec<&String> = got.keys().filter(|n| n.contains("indexer")).collect();
    assert_eq!(indexed.len(), 5);
    assert!(indexed.iter().all(|n| n.starts_with(&format!("blk.{l}."))));
    let c = Checkpoint::parse(&toy_config(true)).unwrap();
    let key = |s: &str| c.keys.iter().find(|(k, _)| k == s).map(|(_, v)| v.clone());
    assert_eq!(
        ["head_count", "key_length", "top_k"].map(|s| key(&format!("attention.indexer.{s}"))),
        [INDEX_HEADS as u32, INDEX_DIM as u32, 8].map(|v| Some(Value::U32(v)))
    );
    let mut types = vec![Value::Bool(false); LAYERS];
    types[l] = Value::Bool(true);
    assert_eq!(key("attention.indexer.types"), Some(Value::Array(types)));
}

/// The header of `planned` under `map`, each tensor in the type this map declares it in, written
/// under the architecture word the map's folder has.
fn header(map: &Checkpoint, planned: &[folder::Planned<Step>]) -> gguf::Split {
    let kvs = super::super::trunk::header(map.word.word(), &map.keys);
    let decls = planned.iter().map(|t| t.decl.clone()).collect();
    let layout = Layout::new(&kvs, decls).unwrap();
    gguf::Split::header_only(PathBuf::from("kimi-k3.gguf"), &layout).unwrap()
}

/// The K3 toy's header reads as the K3 description's hyperparameters: four layers of the kinds
/// the config's lists name, the keys, every layer's tensors of its kind and their dims, each
/// tensor in the type the map declares (BF16 matrices as stored, MXFP4 stacks, F32 vectors),
/// which the description's table holds. A K3 header has no indexer.
#[test]
fn the_k3_toy_header_reads_as_the_kimi_k3_description() {
    use model::arch::kimi_k3::hparams::Kind;
    let config = toy_config(false);
    let p = plan(&config, &toy_entries(false)).unwrap();
    let split = header(&Checkpoint::parse(&config).unwrap(), &p.planned);
    let hp = Hparams::read(&split).unwrap();
    assert_eq!(
        (
            hp.n_layer,
            hp.n_embd,
            hp.n_head,
            hp.latent,
            hp.dense_lead,
            hp.attn_res_block
        ),
        (LAYERS, E, HEADS, LATENT, 1, 2)
    );
    assert_eq!(hp.kinds, [Kind::Kda, Kind::Kda, Kind::Mla, Kind::Kda]);
    assert_eq!((hp.experts.n_expert, hp.experts.n_used), (N_EXP, 2));
    assert_eq!((hp.kda.head_dim, hp.kda.conv), (D, CONV));
    assert!(hp.indexer.is_none());
}

/// The K3.1 toy's header reads as the K3.1 description: the K3's hyperparameters under the
/// `kimi-k31` word, the indexer's geometry and layers the keys give, and the indexer's five
/// tensors on the latent layer in the types the map declares. (`prepare`'s gate opens the whole
/// folder through `model::arch::spec`; this header holds no tokenizer.)
#[test]
fn the_k31_toy_header_reads_as_the_kimi_k31_description() {
    let config = toy_config(true);
    let p = plan(&config, &toy_entries(true)).unwrap();
    let split = header(&Checkpoint::parse(&config).unwrap(), &p.planned);
    assert_eq!(split.architecture(), Some("kimi-k31"));
    let hp = Hparams::read(&split).unwrap();
    let ix = hp.indexer.as_ref().expect("a K3.1 header has the indexer");
    assert_eq!(
        (ix.heads, ix.d, ix.top_k, ix.norm_eps),
        (INDEX_HEADS, INDEX_DIM, 8, 1e-5)
    );
    assert_eq!(ix.layers, [false, false, true, false]);
    for l in 0..LAYERS {
        let indexed = hp
            .layer_dims(l)
            .iter()
            .filter(|(stem, _)| stem.starts_with("indexer."))
            .count();
        assert_eq!(indexed, if l == LATENT_LAYER { 5 } else { 0 }, "layer {l}");
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
    p: &folder::Planning<Step>,
    map: &Checkpoint,
    st: &Safetensors,
    name: &str,
) -> Result<Normal, String> {
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
    let mut n = modelopt::normalize(|s| st.raw(s), &placed, m.role).map_err(|e| e.to_string())?;
    if let Some(moves) = map.mover(name, &m.steps) {
        moves(&mut n)?;
    }
    Ok(n)
}

/// A toy folder's tensors made of its stored bytes agree with the plan's declarations: every
/// tensor but the residual pairs (whose one-dimensional norms the common stack reader does not
/// take yet) comes out in the declared type, dims and byte count, the KDA conv taps, the router,
/// the routed stacks and the latent layer's halves among them.
#[test]
fn a_toy_folder_makes_every_tensor_as_planned() {
    let config = toy_config(true);
    let entries = toy_entries(true);
    let map = Checkpoint::parse(&config).unwrap();
    let p = plan(&config, &entries).unwrap();
    let zeros: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = entries
        .iter()
        .map(|(n, d, s)| {
            (
                n.clone(),
                *d,
                s.clone(),
                vec![0u8; s.iter().product::<usize>() * d.size()],
            )
        })
        .collect();
    let st = stored(&zeros);
    let mut checked = 0;
    for t in &p.planned {
        let folder::Make::Tensor(m) = &t.make else {
            panic!("{} is a tensor", t.decl.name);
        };
        if m.steps == [Step::ResProduct] {
            continue;
        }
        let n =
            made(&p, &map, &st, &t.decl.name).unwrap_or_else(|e| panic!("{}: {e}", t.decl.name));
        assert_eq!(
            (n.type_id, &n.dims, n.bytes.len() as u64),
            (t.decl.type_id, &t.decl.dims, t.decl.nbytes),
            "{}",
            t.decl.name
        );
        checked += 1;
    }
    assert_eq!(checked, 112 - 9);
}

/// `A_log` is `-exp` of its first `num_attention_heads` values, each rounded once to F32, and the
/// tail the checkpoint pads it with is dropped: dims `[heads]`. Mutants: the padding kept, `exp`
/// without its sign, the values taken in f32.
#[test]
fn a_log_is_minus_exp_of_the_heads_values() {
    let config = toy_config(false);
    let map = Checkpoint::parse(&config).unwrap();
    let p = plan(&config, &toy_entries(false)).unwrap();
    let a_log = [0.0f32, 1.0, 7.0, -3.0];
    let bytes: Vec<u8> = a_log.iter().flat_map(|v| v.to_le_bytes()).collect();
    let name = format!("{LM}model.layers.1.self_attn.A_log");
    let st = stored(&[(name, Dtype::F32, vec![A_LOG_LEN], bytes)]);
    let n = made(&p, &map, &st, "blk.1.ssm_a").unwrap();
    let want: Vec<u8> = [-1.0f32, -std::f32::consts::E]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(
        (n.type_id, n.dims, n.bytes),
        (GgmlType::F32.as_u32(), vec![HEADS as u64], want)
    );
    let mut bad = Normal {
        type_id: GgmlType::F32.as_u32(),
        dims: vec![2],
        bytes: [f32::NAN, 0.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        scale: vec![],
    };
    let e = Step::NegExp { heads: 2 }
        .apply("blk.1.ssm_a", &mut bad)
        .unwrap_err();
    assert!(
        e.contains("blk.1.ssm_a") && e.contains("non-finite input NaN"),
        "{e}"
    );
}

/// `kv_b_proj` `[heads (nope + v), rank]` splits per head: the key half is each head's first
/// `nope` rows transposed to `[rank, nope]` (ggml dims `[nope, rank, heads]`) and the value half
/// its last `v` rows as they are (ggml dims `[rank, v, heads]`), as the converter's
/// `view(heads, v + nope, rank)`, `split`, `transpose(1, 2)` does. Mutants: the halves swapped,
/// the key half not transposed, the heads' rows read in one run.
#[test]
fn kv_b_splits_per_head_into_a_transposed_key_half_and_a_value_half() {
    let (heads, nope, v, rank) = (2usize, 3usize, 2usize, 4usize);
    let rows = heads * (nope + v);
    // Element (row, col) holds `row * 10 + col`, a distinct u16 a BF16 move never reads.
    let at = |row: usize, col: usize| (row * 10 + col) as u16;
    let source = || Normal {
        type_id: GgmlType::BF16.as_u32(),
        dims: vec![rank as u64, rows as u64],
        bytes: (0..rows)
            .flat_map(|r| (0..rank).flat_map(move |c| at(r, c).to_le_bytes()))
            .collect(),
        scale: vec![],
    };
    let u16s = |b: &[u8]| {
        b.as_chunks::<2>()
            .0
            .iter()
            .map(|x| u16::from_le_bytes(*x))
            .collect::<Vec<_>>()
    };
    let mut k = source();
    Step::KeyHalf { heads, nope, v }
        .apply("k_b", &mut k)
        .unwrap();
    let want_k: Vec<u16> = (0..heads)
        .flat_map(|h| {
            (0..rank).flat_map(move |c| (0..nope).map(move |r| at(h * (nope + v) + r, c)))
        })
        .collect();
    assert_eq!(
        (k.dims.clone(), u16s(&k.bytes)),
        (vec![nope as u64, rank as u64, heads as u64], want_k)
    );
    let mut x = source();
    Step::ValueHalf { heads, nope, v }
        .apply("v_b", &mut x)
        .unwrap();
    let want_v: Vec<u16> = (0..heads)
        .flat_map(|h| {
            (0..v).flat_map(move |r| (0..rank).map(move |c| at(h * (nope + v) + nope + r, c)))
        })
        .collect();
    assert_eq!(
        (x.dims.clone(), u16s(&x.bytes)),
        (vec![rank as u64, v as u64, heads as u64], want_v)
    );
    let mut short = source();
    short.bytes.pop();
    let e = Step::KeyHalf { heads, nope, v }
        .apply("k_b", &mut short)
        .unwrap_err();
    assert!(e.contains("k_b") && e.contains("BF16 rows"), "{e}");
}

/// A residual pair's score is the elementwise product of its norm and its projection, in F32:
/// the stacked pair `[k, 1, 2]` (norm, then projection) becomes `[k]`. Mutants: the norm alone,
/// the sum, the projection taken as the first half. A product past F32 and a stack of another
/// shape are refused by name.
#[test]
fn a_residual_score_is_the_product_of_its_norm_and_projection() {
    let norm = [1.0f32, -2.0, 0.5, 4.0];
    let proj = [10.0f32, 3.0, -8.0, 0.25];
    let stack = |a: &[f32], b: &[f32]| Normal {
        type_id: GgmlType::F32.as_u32(),
        dims: vec![a.len() as u64, 1, 2],
        bytes: a.iter().chain(b).flat_map(|v| v.to_le_bytes()).collect(),
        scale: vec![],
    };
    let mut n = stack(&norm, &proj);
    Step::ResProduct
        .apply("blk.0.attn_res_score.weight", &mut n)
        .unwrap();
    let got: Vec<f32> = n
        .bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    assert_eq!((n.dims, got), (vec![4], vec![10.0, -6.0, -4.0, 1.0]));
    let mut big = stack(&[f32::MAX, 1.0], &[2.0, 1.0]);
    let e = Step::ResProduct.apply("s", &mut big).unwrap_err();
    assert!(e.contains("s: ResProduct at 0: the product is inf"), "{e}");
    let mut flat = stack(&norm, &proj);
    flat.dims = vec![4, 2];
    let e = Step::ResProduct.apply("s", &mut flat).unwrap_err();
    assert!(e.contains("reads a stacked pair"), "{e}");
}

/// The moves carry the config's widths: the heads of `A_log`, the heads, nope and value widths of
/// the key and value halves.
#[test]
fn the_moves_carry_the_configs_widths() {
    let p = plan(&toy_config(false), &toy_entries(false)).unwrap();
    let steps = |name: &str| match &p.planned.iter().find(|t| t.decl.name == name).unwrap().make {
        folder::Make::Tensor(m) => m.steps.clone(),
        _ => panic!("{name} is a tensor"),
    };
    let (heads, nope, v) = (HEADS, NOPE, HEAD_V);
    assert_eq!(steps("blk.1.ssm_a"), [Step::NegExp { heads }]);
    assert_eq!(
        steps("blk.2.attn_k_b.weight"),
        [Step::KeyHalf { heads, nope, v }]
    );
    assert_eq!(
        steps("blk.2.attn_v_b.weight"),
        [Step::ValueHalf { heads, nope, v }]
    );
    assert!(steps("blk.1.ssm_dt.bias").is_empty() && steps("blk.1.ssm_conv1d_q.weight").is_empty());
}

/// compressed-tensors' E2M1 magnitudes, indexed by the code's low three bits; the code's fourth
/// bit is the sign.
const E2M1: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// How a toy encoder may deviate from compressed-tensors' `mxfp4-pack-quantized`.
#[derive(Clone, Copy, PartialEq)]
enum Wrong {
    /// compressed-tensors' own: `packed = code(2i) | code(2i + 1) << 4`, scale byte `127 + e`.
    No,
    /// The two nibbles of a byte the other way round.
    Nibbles,
    /// The scale byte one above (a bias of 128).
    BiasUp,
    /// The scale byte one below (a bias of 126).
    BiasDown,
}

/// The value of element `col` of row `row` of expert `expert` of the projection `seed`: a sign,
/// an E2M1 magnitude and the block's exponent, all exactly representable in F32. Adjacent
/// elements differ, so a swap of the nibbles of a byte moves every value.
fn toy_value(seed: usize, expert: usize, row: usize, col: usize) -> f64 {
    let code = (seed * 5 + expert * 3 + row * 7 + col * 3) % 8;
    // A zero has no sign: the code of -0 is not one the engine need keep.
    let sign = if code != 0 && (seed + expert + row + col).is_multiple_of(5) {
        -1.0
    } else {
        1.0
    };
    sign * E2M1[code] * 2f64.powi(toy_exponent(seed, expert, row, col / 32))
}

/// The exponent of block `block` of row `row`: 2^-7 to 2^7.
fn toy_exponent(seed: usize, expert: usize, row: usize, block: usize) -> i32 {
    ((seed + expert * 2 + row + block * 5) % 15) as i32 - 7
}

/// `weight_packed` and `weight_scale` of one expert's `[rows, k]` values, the way compressed-tensors
/// writes them (`wrong` deviating), the codes from the table [`E2M1`] and not from the engine.
fn encode(seed: usize, expert: usize, rows: usize, k: usize, wrong: Wrong) -> (Vec<u8>, Vec<u8>) {
    let code = |row: usize, col: usize| -> u8 {
        let v = toy_value(seed, expert, row, col);
        let magnitude = v.abs() / 2f64.powi(toy_exponent(seed, expert, row, col / 32));
        let index = E2M1
            .iter()
            .position(|&m| m == magnitude)
            .expect("an E2M1 magnitude") as u8;
        index | u8::from(v.is_sign_negative()) << 3
    };
    let mut packed = Vec::with_capacity(rows * k / 2);
    let mut scales = Vec::with_capacity(rows * k / 32);
    for row in 0..rows {
        for col in (0..k).step_by(2) {
            let (low, high) = (code(row, col), code(row, col + 1));
            packed.push(if wrong == Wrong::Nibbles {
                high | low << 4
            } else {
                low | high << 4
            });
        }
        for block in 0..k / 32 {
            let e = (127 + toy_exponent(seed, expert, row, block)) as u8;
            scales.push(match wrong {
                Wrong::BiasUp => e + 1,
                Wrong::BiasDown => e - 1,
                _ => e,
            });
        }
    }
    (packed, scales)
}

/// The routed stacks of layer 1 of the toy folder, built by `wrong`'s encoder, as the stack the
/// engine's reader makes of them: GGUF's MXFP4 blocks.
fn expert_stacks(wrong: Wrong) -> [(&'static str, &'static str, Vec<f32>); 3] {
    let config = toy_config(true);
    let map = Checkpoint::parse(&config).unwrap();
    let p = plan(&config, &toy_entries(true)).unwrap();
    // w1 is the gate, w3 the up and w2 the down projection of an expert.
    [
        ("w1", "ffn_gate_exps", 1),
        ("w3", "ffn_up_exps", 3),
        ("w2", "ffn_down_exps", 2),
    ]
    .map(|(w, stem, seed)| {
        let (rows, k) = if w == "w2" {
            (LATENT, EFF)
        } else {
            (EFF, LATENT)
        };
        let mut values = vec![];
        for e in 0..N_EXP {
            let (packed, scales) = encode(seed, e, rows, k, wrong);
            let x = format!("{LM}model.layers.1.block_sparse_moe.experts.{e}.{w}.");
            values.push((
                format!("{x}weight_packed"),
                Dtype::U8,
                vec![rows, k / 2],
                packed,
            ));
            values.push((
                format!("{x}weight_scale"),
                Dtype::U8,
                vec![rows, k / 32],
                scales,
            ));
        }
        let n = made(&p, &map, &stored(&values), &format!("blk.1.{stem}.weight")).unwrap();
        assert_eq!(n.type_id, GgmlType::MXFP4.as_u32());
        assert_eq!(n.dims, [k as u64, rows as u64, N_EXP as u64]);
        let mut out = vec![0f32; N_EXP * rows * k];
        gguf::quant::dequant_row(GgmlType::MXFP4, &n.bytes, &mut out).unwrap();
        (w, stem, out)
    })
}

/// The routed experts' MXFP4 packs dequantize to the values compressed-tensors encoded, bit for
/// bit: the pack's even value is the low nibble of its byte, the scale byte is `127 + e` for the
/// block's `2^e`, and `w1`, `w3` and `w2` are the gate, up and down stacks, expert by expert. The
/// same check on an encoder with the nibbles swapped, or the bias one above or below, finds
/// every one of the three stacks' values moved, so the toy values pin both facts.
#[test]
fn the_mxfp4_packs_dequantize_to_the_encoded_values() {
    let seeds = [("w1", 1), ("w3", 3), ("w2", 2)];
    let exact = |stacks: &[(&str, &str, Vec<f32>); 3]| {
        stacks.iter().all(|(w, _, got)| {
            let seed = seeds.iter().find(|(x, _)| x == w).unwrap().1;
            let (rows, k) = if *w == "w2" {
                (LATENT, EFF)
            } else {
                (EFF, LATENT)
            };
            got.iter().enumerate().all(|(i, g)| {
                let (e, r, c) = (i / (rows * k), i / k % rows, i % k);
                g.to_bits() == (toy_value(seed, e, r, c) as f32).to_bits()
            })
        })
    };
    assert!(exact(&expert_stacks(Wrong::No)));
    for wrong in [Wrong::Nibbles, Wrong::BiasUp, Wrong::BiasDown] {
        assert!(!exact(&expert_stacks(wrong)));
    }
}

/// A config the map refuses, by the name of the key: another model's, one that breaks the layer
/// split, a field the program does not run, and a K3.1 key on a K3 config or the reverse.
#[test]
fn a_config_of_another_shape_is_refused_by_key() {
    type Case = (&'static str, bool, Box<dyn Fn(&mut Json)>, &'static str);
    let cases: Vec<Case> = vec![
        (
            "an architecture that is the other variant's",
            false,
            Box::new(|c| c["architectures"] = json!(["KimiK31ForConditionalGeneration"])),
            "architectures",
        ),
        (
            "another model_type",
            false,
            Box::new(|c| c["model_type"] = json!("kimi_k4")),
            "model_type",
        ),
        (
            "a dsa_config on a K3 config",
            false,
            Box::new(|c| c["text_config"]["dsa_config"] = json!({})),
            "dsa_config",
        ),
        (
            "a K3.1 config with no dsa_config",
            true,
            Box::new(|c| {
                c["text_config"]
                    .as_object_mut()
                    .unwrap()
                    .remove("dsa_config");
            }),
            "dsa_config",
        ),
        (
            "an indexer on layers that are not the latent ones",
            true,
            Box::new(|c| c["text_config"]["dsa_config"]["dsa_layers"] = json!([1])),
            "dsa_layers",
        ),
        (
            "an indexer with a rope",
            true,
            Box::new(|c| c["text_config"]["dsa_config"]["index_rope_dim"] = json!(64)),
            "index_rope_dim",
        ),
        (
            "an indexer that skips positions",
            true,
            Box::new(|c| c["text_config"]["dsa_config"]["index_topk_freq"] = json!(2)),
            "index_topk_freq",
        ),
        (
            "a layer that is neither KDA nor latent",
            false,
            Box::new(|c| c["text_config"]["linear_attn_config"]["kda_layers"] = json!([1, 2])),
            "kda_layers",
        ),
        (
            "a layer that is both",
            false,
            Box::new(|c| {
                c["text_config"]["linear_attn_config"]["kda_layers"] = json!([1, 2, 3, 4])
            }),
            "kda_layers",
        ),
        (
            "a layer outside the model",
            false,
            Box::new(|c| c["text_config"]["linear_attn_config"]["full_attn_layers"] = json!([5])),
            "full_attn_layers",
        ),
        (
            "key heads that are not the query heads",
            false,
            Box::new(|c| c["text_config"]["num_key_value_heads"] = json!(1)),
            "num_key_value_heads",
        ),
        (
            "a low-rank gate",
            false,
            Box::new(|c| {
                c["text_config"]["linear_attn_config"]["use_full_rank_gate"] = json!(false)
            }),
            "use_full_rank_gate",
        ),
        (
            "a decay bound that is not negative",
            false,
            Box::new(|c| c["text_config"]["linear_attn_config"]["gate_lower_bound"] = json!(1.0)),
            "gate_lower_bound",
        ),
        (
            "a softmax router",
            false,
            Box::new(|c| c["text_config"]["moe_router_activation_func"] = json!("softmax")),
            "moe_router_activation_func",
        ),
        (
            "a next-token block",
            false,
            Box::new(|c| c["text_config"]["num_nextn_predict_layers"] = json!(1)),
            "num_nextn_predict_layers",
        ),
        (
            "a head tied to the embedding",
            false,
            Box::new(|c| c["text_config"]["tie_word_embeddings"] = json!(true)),
            "tie_word_embeddings",
        ),
        (
            "a rope scaling the map does not read",
            false,
            Box::new(|c| c["text_config"]["rope_scaling"] = json!({"type": "yarn"})),
            "rope_scaling",
        ),
        (
            "a layer count that is not a count",
            false,
            Box::new(|c| c["text_config"]["num_hidden_layers"] = json!(0)),
            "num_hidden_layers",
        ),
        (
            "every layer dense",
            false,
            Box::new(|c| c["text_config"]["first_k_dense_replace"] = json!(LAYERS)),
            "first_k_dense_replace",
        ),
    ];
    for (what, k31, edit, key) in cases {
        let mut c = toy_config(k31);
        edit(&mut c);
        let e = Checkpoint::parse(&c).unwrap_err().to_string();
        assert!(e.contains(key), "{what}: {e}");
    }
}

/// `entries` with the tensors `edit` leaves, named by their folder names.
fn entries_with(k31: bool, edit: impl FnOnce(&mut Entries)) -> Entries {
    let mut e = toy_entries(k31);
    edit(&mut e);
    e
}

/// A folder of another shape than the config's is refused by the tensor's name: a tensor the
/// config's layer kind does not hold (or lacks), a shape off the config's widths, a short or
/// misshapen expert stack, half a residual pair, a name no Kimi K3 folder has, and a quantization
/// that is not the family's.
#[test]
fn a_folder_of_another_shape_is_refused_by_tensor() {
    use Dtype::{Bf16, U8};
    let layer = |l: usize, rest: &str| format!("{LM}model.layers.{l}.{rest}");
    let without = |k31: bool, name: String| entries_with(k31, |e| e.retain(|(n, _, _)| *n != name));
    let with = |k31: bool, name: String, dtype: Dtype, shape: Vec<usize>| {
        entries_with(k31, |e| e.push((name, dtype, shape)))
    };
    let reshaped = |name: String, shape: Vec<usize>| {
        entries_with(false, |e| {
            e.iter_mut().find(|(n, _, _)| *n == name).unwrap().2 = shape;
        })
    };
    let cases: Vec<(&str, Json, Entries, &str)> = vec![
        (
            "a KDA layer without its q projection",
            toy_config(false),
            without(false, layer(1, "self_attn.q_proj.weight")),
            "blk.1.attn_q.weight: is missing from the folder",
        ),
        (
            "a latent layer without its kv_b",
            toy_config(false),
            without(false, layer(2, "self_attn.kv_b_proj.weight")),
            "blk.2.attn_k_b.weight: is missing from the folder",
        ),
        (
            "a K3.1 folder without its indexer's key",
            toy_config(true),
            without(true, layer(2, "self_attn.indexer.wk.weight")),
            "indexer.attn_k.weight: is missing from the folder",
        ),
        (
            "a router bias missing",
            toy_config(false),
            without(
                false,
                layer(3, "block_sparse_moe.gate.e_score_correction_bias"),
            ),
            "blk.3.exp_probs_b.bias: is missing from the folder",
        ),
        (
            "the final mix's pair missing",
            toy_config(false),
            without(false, format!("{LM}model.output_attn_res_proj.weight")),
            "half of a norm and projection pair",
        ),
        (
            "a layer's pair of one tensor",
            toy_config(false),
            without(false, layer(0, "mlp_res_norm.weight")),
            "half of a norm and projection pair",
        ),
        (
            "a q projection on a latent layer",
            toy_config(false),
            with(
                false,
                layer(LATENT_LAYER, "self_attn.q_proj.weight"),
                Bf16,
                vec![INNER, E],
            ),
            "is a tensor a latent-attention layer 2 of this config does not hold",
        ),
        (
            "a kv_b on a KDA layer",
            toy_config(false),
            with(
                false,
                layer(1, "self_attn.kv_b_proj.weight"),
                Bf16,
                vec![HEADS * (NOPE + HEAD_V), KV_LORA],
            ),
            "is a tensor a KDA layer 1 of this config does not hold",
        ),
        (
            "an indexer on a K3 folder",
            toy_config(false),
            with(
                false,
                layer(LATENT_LAYER, "self_attn.indexer.wk.weight"),
                Bf16,
                vec![INDEX_DIM, E],
            ),
            "is a tensor a latent-attention layer 2 of this config does not hold",
        ),
        (
            "an indexer on a KDA layer",
            toy_config(true),
            with(
                true,
                layer(1, "self_attn.indexer.wk.weight"),
                Bf16,
                vec![INDEX_DIM, E],
            ),
            "is a tensor a KDA layer 1 of this config does not hold",
        ),
        (
            "a conv of another width",
            toy_config(false),
            reshaped(
                layer(1, "self_attn.q_conv1d.weight"),
                vec![INNER, 1, CONV + 1],
            ),
            "q_conv1d.weight: shape [256, 1, 5], not [256, 1, 4]",
        ),
        (
            "A_log with fewer values than heads",
            toy_config(false),
            reshaped(layer(1, "self_attn.A_log"), vec![1]),
            "1 values, fewer than the 2 heads",
        ),
        (
            "a kv_b of another height",
            toy_config(false),
            reshaped(
                layer(2, "self_attn.kv_b_proj.weight"),
                vec![HEADS * NOPE, KV_LORA],
            ),
            "not 2 heads of 128 + 128 rows by the latent rank 512",
        ),
        (
            "an embedding of another width",
            toy_config(false),
            reshaped(format!("{LM}model.embed_tokens.weight"), vec![VOCAB, E / 2]),
            "embed_tokens.weight: shape [2, 256], not [2, 512]",
        ),
        (
            "a residual norm of another width",
            toy_config(false),
            reshaped(layer(3, "self_attention_res_norm.weight"), vec![E / 2]),
            "shape [256], not [512]",
        ),
        (
            "a residual projection that is a matrix",
            toy_config(false),
            reshaped(layer(3, "mlp_res_proj.weight"), vec![2, E]),
            "shape [2, 512], not [1, 512]",
        ),
        (
            "a routed stack short of an expert",
            toy_config(false),
            entries_with(false, |e| {
                e.retain(|(n, _, _)| !n.contains("layers.1.block_sparse_moe.experts.3.w2."))
            }),
            "the stack's 3 expert ids are not 0..4",
        ),
        (
            "a routed projection of another shape",
            toy_config(false),
            entries_with(false, |e| {
                for (n, _, s) in e.iter_mut() {
                    if n.ends_with("layers.3.block_sparse_moe.experts.2.w1.weight_packed") {
                        *s = vec![EFF, LATENT / 2 + 16];
                    }
                    if n.ends_with("layers.3.block_sparse_moe.experts.2.w1.weight_scale") {
                        *s = vec![EFF, LATENT / 32 + 1];
                    }
                }
            }),
            "shape differs from",
        ),
        (
            "routed stacks whose rows and columns are swapped",
            toy_config(false),
            entries_with(false, |e| {
                for (n, _, s) in e.iter_mut() {
                    if n.contains("layers.3.block_sparse_moe.experts.") && n.contains(".w2.") {
                        *s = if n.ends_with("weight_packed") {
                            vec![EFF, LATENT / 2]
                        } else {
                            vec![EFF, LATENT / 32]
                        };
                    }
                }
            }),
            "not the latent 512 by the expert width 256",
        ),
        (
            "a name no Kimi K3 folder has",
            toy_config(false),
            with(false, layer(1, "self_attn.q_norm.weight"), Bf16, vec![D]),
            "layers.1.self_attn.q_norm.weight: not a Kimi K3 tensor",
        ),
        (
            "an expert projection that is not w1, w2 or w3",
            toy_config(false),
            with(
                false,
                layer(1, "block_sparse_moe.experts.0.w4.weight_packed"),
                U8,
                vec![EFF, LATENT / 2],
            ),
            "not a Kimi K3 expert projection",
        ),
        (
            "a layer past the config's",
            toy_config(false),
            with(
                false,
                layer(LAYERS, "input_layernorm.weight"),
                Bf16,
                vec![E],
            ),
            "layer 4 of 4",
        ),
        (
            "a tensor outside the language model",
            toy_config(false),
            with(false, "audio_tower.proj.weight".into(), Bf16, vec![8, 8]),
            "audio_tower.proj.weight: not a Kimi K3 tensor",
        ),
    ];
    for (what, config, entries, want) in cases {
        let e = refusal(&config, &entries);
        assert!(e.contains(want), "{what}: {e}");
    }
}

/// A quantization that is not the family's is refused by name: a group of 16, NVFP4's packs and
/// a block-scaled FP8 config over the folder's MXFP4 packs. The folder's own is read, so the
/// plan places every routed pack as MXFP4 and nothing else.
#[test]
fn a_quantization_that_is_not_the_familys_is_refused() {
    let config = toy_config(false);
    fn group(c: &mut Json) -> &mut Json {
        &mut c["text_config"]["quantization_config"]["config_groups"]["group_0"]
    }
    let mut g16 = config.clone();
    group(&mut g16)["weights"]["group_size"] = json!(16);
    let e = Config::parse(&g16, None).unwrap_err().to_string();
    assert!(e.contains("mxfp4-pack-quantized weights"), "{e}");
    let mut nv = config.clone();
    group(&mut nv)["format"] = json!("nvfp4-pack-quantized");
    let e = Config::parse(&nv, None)
        .map(|_| ())
        .unwrap_err()
        .to_string();
    assert!(e.contains("nvfp4-pack-quantized"), "{e}");
    let mut fp8 = config.clone();
    fp8["text_config"]["quantization_config"] = json!({"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
               "weight_block_size": [128, 128]});
    let parsed = Config::parse(&fp8, None).unwrap();
    let entries = toy_entries(false);
    let planned = folder::plan_tensors(
        &Checkpoint::parse(&fp8).unwrap(),
        &parsed,
        &entries,
        false,
        Keep::InPlace,
    );
    let e = planned
        .err()
        .expect("an FP8 config over MXFP4 packs")
        .to_string();
    assert!(
        e.contains("experts.0.w1.weight_scale: not a Kimi K3 expert projection"),
        "{e}"
    );
    let ok = Config::parse(&config, None).unwrap();
    let p = folder::plan_tensors(
        &Checkpoint::parse(&config).unwrap(),
        &ok,
        &entries,
        false,
        Keep::InPlace,
    )
    .unwrap();
    let packs = entries
        .iter()
        .filter(|(n, _, _)| n.ends_with(".weight_packed"))
        .count();
    let placed = p
        .plan
        .placed
        .values()
        .filter(|x| x.format == hf::quant_config::Format::Mxfp4)
        .count();
    assert_eq!((packs, placed), (3 * 3 * N_EXP, 3 * 3 * N_EXP));
}

/// A residual pair stored as the checkpoint stores it, a BF16 norm `[512]` and a BF16 projection
/// `[1, 512]`, is made into one F32 score vector of the products of their BF16 values, each pair
/// into its own score tensor. Mutants: the projection dropped, the pairs' layers swapped.
#[test]
fn a_stored_residual_pair_makes_its_score_vector() {
    let config = toy_config(false);
    let map = Checkpoint::parse(&config).unwrap();
    let p = plan(&config, &toy_entries(false)).unwrap();
    let bf16 = |v: f32| ((v.to_bits() >> 16) as u16).to_le_bytes();
    let norm = |l: usize, i: usize| ((l * 3 + i % 7) as f32 + 1.0) * 0.5;
    let proj = |l: usize, i: usize| ((i % 5) as f32 - 2.0) * (l as f32 + 1.0);
    let tensor = |name: String, dtype: Dtype, shape: Vec<usize>, f: &dyn Fn(usize) -> f32| {
        let bytes = (0..E).flat_map(|i| bf16(f(i))).collect();
        (name, dtype, shape, bytes)
    };
    let mut values = vec![];
    for l in [1usize, 3] {
        for (stem, shift) in [("self_attention_res", 0), ("mlp_res", 10)] {
            let at = |part: &str| format!("{LM}model.layers.{l}.{stem}_{part}.weight");
            values.push(tensor(at("norm"), Dtype::Bf16, vec![E], &|i| {
                norm(l + shift, i)
            }));
            values.push(tensor(at("proj"), Dtype::Bf16, vec![1, E], &|i| {
                proj(l + shift, i)
            }));
        }
    }
    let st = stored(&values);
    for (l, name, shift) in [
        (1usize, "blk.1.attn_res_score.weight", 0),
        (1, "blk.1.ffn_res_score.weight", 10),
        (3, "blk.3.attn_res_score.weight", 0),
        (3, "blk.3.ffn_res_score.weight", 10),
    ] {
        let n = made(&p, &map, &st, name).unwrap();
        let want: Vec<u8> = (0..E)
            .flat_map(|i| {
                let bf = |v: f32| f32::from_bits(u32::from(u16::from_le_bytes(bf16(v))) << 16);
                (bf(norm(l + shift, i)) * bf(proj(l + shift, i))).to_le_bytes()
            })
            .collect();
        assert_eq!(
            (n.type_id, n.dims, n.bytes),
            (GgmlType::F32.as_u32(), vec![E as u64], want),
            "{name}"
        );
    }
}

/// The real K3.1 folder's shards 1, 2, 4 and 94 (layers 0, 1 and 3 and the head), each expert
/// past expert 0 expanded from it, as the committed header file holds them.
fn real_headers() -> Entries {
    let experts = real(true)["text_config"]["num_experts"].as_u64().unwrap();
    let mut out = vec![];
    for (name, dtype, shape) in crate::test_support::entries(&json(
        "checkpoint-headers/moonshotai-kimi-k3.1-pro-layers-0-1-3.tensors.json.gz",
    )) {
        match name.find(".experts.0.") {
            Some(at) => {
                for e in 0..experts {
                    let named = format!("{}.experts.{e}.{}", &name[..at], &name[at + 11..]);
                    out.push((named, dtype, shape.clone()));
                }
            }
            None => out.push((name, dtype, shape)),
        }
    }
    out
}

/// The table the config derives of what a folder stores ([`FolderMap::stored`]) is the real K3.1
/// folder's: its layers 0, 1 and 3 and its head, name by name with dtype and shape.
#[test]
fn the_stored_table_is_the_real_folders_in_its_sampled_shards() {
    let map = Checkpoint::parse(&real(true)).unwrap();
    let stored = map.stored().expect("a Kimi map derives its table").tensors;
    let sampled = |name: &str| match name.split_once("model.layers.") {
        Some((_, rest)) => matches!(rest.split('.').next(), Some("0" | "1" | "3")),
        None => true,
    };
    let derived: BTreeMap<_, _> = stored
        .into_iter()
        .filter(|t| sampled(&t.0))
        .map(|t| (t.0, (t.1, t.2)))
        .collect();
    let real: BTreeMap<_, _> = real_headers()
        .into_iter()
        .map(|t| (t.0, (t.1, t.2)))
        .collect();
    // The head's 5 tensors and the three headers' 23, 5,404 and 5,403.
    assert_eq!(real.len(), 5 + 23 + 5_404 + 5_403);
    assert_eq!(derived, real);
}

/// The whole real K3.1 folder, its 497,172 tensors as the config derives them, plans with no
/// refusal into 2,693 tensors of 1,561,602,808,192 bytes: 93 layers, 24 of them latent with the
/// indexer's 5 tensors (120 in all), and 304 steps, the moves the converter's layouts need.
#[test]
fn the_whole_real_k31_folder_plans_with_no_refusal() {
    let config = real(true);
    let entries = Checkpoint::parse(&config)
        .unwrap()
        .stored()
        .unwrap()
        .tensors;
    assert_eq!(entries.len(), 497_172);
    let p = plan(&config, &entries).unwrap();
    assert!(p.plan.refused.is_empty(), "{:?}", p.plan.refused);
    assert!(p.skipped.is_empty(), "{:?}", p.skipped);
    assert_eq!(p.planned.len(), 2_693);
    let bytes: u64 = p.planned.iter().map(|t| t.decl.nbytes).sum();
    assert_eq!(bytes, 1_561_602_808_192);
    let indexer = p
        .planned
        .iter()
        .filter(|t| t.decl.name.contains(".indexer."))
        .count();
    assert_eq!(indexer, 120);
    // The write's header, at the real widths, is a file the K3.1 description reads: every type
    // and shape the plan declares is the table's, the indexer on the 24 latent layers alone.
    let split = header(&Checkpoint::parse(&config).unwrap(), &p.planned);
    let hp = Hparams::read(&split).expect("the planned K3.1 header reads");
    let dsa: Vec<usize> = (0..hp.n_layer).filter(|&l| hp.dsa(l)).collect();
    assert_eq!(dsa.len(), 24);
    assert_eq!((dsa[0], dsa[1], dsa[23]), (3, 7, 92));
    model::arch::kimi_k3::roles::classify(&split, &hp).expect("every tensor has its role");
    let steps: usize = p
        .planned
        .iter()
        .map(|t| match &t.make {
            folder::Make::Tensor(m) => m.steps.len(),
            _ => 0,
        })
        .sum();
    assert_eq!(steps, 304);
}
