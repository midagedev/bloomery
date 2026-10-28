//! The HF-native dense family's gates ([`Family::Qwen35Hf`]): the real
//! `nvidia/Qwen3.8-27B-NVFP4` config and shard headers read as the GGUF file of the same model
//! (names, dims and the rule's types), the config's own table row by row, the tensors the map
//! leaves out and refuses by name, and a small folder of the export's layout opened as a `qwen35`
//! file.

use std::collections::BTreeMap;
use std::path::Path;

use gguf::GgmlType;
use gguf::quant::Q8Block;
use hf::quant_config::Config;
use safetensors::Dtype;
use serde_json::json;

use super::dense_tests::{Row, config_rows};
use super::*;
use crate::folder;
use crate::test_support::{
    Entries, Stored, bf16_tensor, entries, gguf_tensors, json, key_json, temp_dir, tokenizer,
    write_folder,
};

const CONFIG: &str = "checkpoint-headers/nvidia__Qwen3.8-27B-NVFP4__config.json.gz";
const HF_QUANT: &str = "checkpoint-headers/nvidia__Qwen3.8-27B-NVFP4__hf_quant_config.json.gz";
const TENSORS: &str = "checkpoint-headers/qwen3.8-27b-nvfp4-layers-0-3.tensors.json.gz";
const MLX_CONFIG: &str = "quant-configs/mlx-community__Qwen3.8-27B-4bit__config.json.gz";
const GGUF: &str = "checkpoint-headers/qwen3.8-27b-q4km-gguf.header.json.gz";
const TEXT: &str = "model.language_model.layers";

/// The tensors of the checkpoint's next-token layer, which the dense map leaves out by name.
const MTP_NAMES: [&str; 15] = [
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
];

/// Whether each layer of `config` attends, by `layer_types`.
fn kinds(config: &Json) -> Vec<bool> {
    config["text_config"]["layer_types"]
        .as_array()
        .unwrap()
        .iter()
        .map(|kind| match kind.as_str().unwrap() {
            "full_attention" => true,
            "linear_attention" => false,
            other => panic!("{other}"),
        })
        .collect()
}

/// The 2,194 tensors of the export (the index's `weight_map` count): the committed layers 0 (a
/// delta-rule layer) and 3 (an attention layer) stand for every layer of their kind, whose
/// names, dtypes and shapes were compared with them in all three shard headers, and every
/// unlayered tensor is committed as it is.
fn real_entries(config: &Json) -> Entries {
    let committed = entries(&json(TENSORS));
    let mut out: Entries = committed
        .iter()
        .filter(|(name, ..)| !name.starts_with(TEXT))
        .cloned()
        .collect();
    for (l, full) in kinds(config).into_iter().enumerate() {
        let prefix = format!("{TEXT}.{}.", if full { 3 } else { 0 });
        out.extend(committed.iter().filter_map(|(name, dtype, shape)| {
            let rest = name.strip_prefix(&prefix)?;
            Some((format!("{TEXT}.{l}.{rest}"), *dtype, shape.clone()))
        }));
    }
    out
}

/// `entries` planned under `config`, its quantization read from `hf_quant` (ModelOpt's own file)
/// or, with none, from the copy in `config.json`.
fn plan_with(
    checkpoint: &Checkpoint,
    config: &Json,
    hf_quant: Option<&Json>,
    entries: &Entries,
    q8: bool,
) -> Result<folder::Planning<Step>, folder::Error> {
    let quant = Config::parse(config, hf_quant).unwrap();
    folder::plan_tensors(checkpoint, &quant, entries, q8)
}

/// [`plan_with`] over ModelOpt's `hf_quant_config.json`.
fn planning(
    checkpoint: &Checkpoint,
    config: &Json,
    entries: &Entries,
    q8: bool,
) -> Result<folder::Planning<Step>, folder::Error> {
    plan_with(checkpoint, config, Some(&json(HF_QUANT)), entries, q8)
}

/// The mapped tensor `name` of a planning's plain tensors.
fn tensor<'a>(got: &BTreeMap<&str, &'a folder::Planned<Step>>, name: &str) -> &'a Mapped {
    let folder::Make::Tensor(m) = &got[name].make else {
        panic!("{name}: not a plain tensor");
    };
    m
}

/// The type the rule gives the file's tensor `name`: NVFP4 for the feed-forward blocks and the
/// head, Q8_0 for the FP8 attention and delta-rule projections, the embedding as stored (BF16, or
/// Q8_0 where `q8` asks) and F32 for the rest.
fn rule_type(name: &str, q8: bool) -> GgmlType {
    let stem = name.strip_suffix(".weight").unwrap_or(name);
    match stem.rsplit('.').next().unwrap() {
        "token_embd" if q8 => GgmlType::Q8_0,
        "token_embd" => GgmlType::BF16,
        "output" | "ffn_gate" | "ffn_up" | "ffn_down" => GgmlType::NVFP4,
        "attn_qkv" | "attn_gate" | "ssm_out" | "attn_q" | "attn_k" | "attn_v" | "attn_output" => {
            GgmlType::Q8_0
        }
        _ => GgmlType::F32,
    }
}

/// The export's config picks the HF dense family, its own `quantization_config` and ModelOpt's
/// `hf_quant_config.json` place every tensor alike, the MLX folder of the same class stays the
/// MLX family's, and the header keys are the GGUF file's `qwen35.*` keys but for the next-token
/// layer (the file counts it in `block_count` and names it in `nextn_predict_layers`).
#[test]
fn the_real_nvfp4_config_picks_the_hf_family_and_reads_as_the_gguf_files_keys() {
    let config = json(CONFIG);
    let gguf = json(GGUF);
    let checkpoint = Checkpoint::parse(&config).unwrap();
    assert_eq!(checkpoint.geometry.family, Family::Qwen35Hf);
    let mlx = Checkpoint::parse(&json(MLX_CONFIG)).unwrap();
    assert_eq!(mlx.geometry.family, Family::Qwen35);

    let entries = real_entries(&config);
    assert_eq!(entries.len(), 2194);
    let planned = |p: folder::Planning<Step>| -> Vec<(String, Vec<u64>, u32)> {
        p.planned
            .iter()
            .map(|p| (p.decl.name.clone(), p.decl.dims.clone(), p.decl.type_id))
            .collect()
    };
    assert_eq!(
        planned(plan_with(&checkpoint, &config, None, &entries, false).unwrap()),
        planned(planning(&checkpoint, &config, &entries, false).unwrap()),
        "config.json's copy of the quantization and hf_quant_config.json place alike"
    );

    let keys = checkpoint.keys(&[], |_| unreachable!("no PLE")).unwrap();
    let got: serde_json::Map<String, Json> =
        keys.iter().map(|(k, v)| (k.clone(), key_json(v))).collect();
    let mut want: serde_json::Map<String, Json> = gguf["keys"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(k, _)| k.starts_with("qwen35."))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    assert_eq!(
        want.remove("qwen35.nextn_predict_layers"),
        Some(json!(["U32", 1]))
    );
    assert_eq!(
        want.insert("qwen35.block_count".into(), json!(["U32", 64])),
        Some(json!(["U32", 65]))
    );
    assert_eq!(Json::Object(got), Json::Object(want));
}

/// The export's tensors, planned under its own quantization config, are the tensors of the GGUF
/// file of the same model that its trunk holds: the same 851 names at the same ggml dims, each
/// of the type the rule gives (NVFP4 feed-forward blocks and head, Q8_0 from the FP8
/// projections, F32 vectors, the embedding as stored or Q8_0 where asked), each made of the
/// checkpoint tensor of its own name by the moves the layout needs (the `+ 1` of every gamma but
/// the delta rule's own, the V heads tiled, the taps as stored). The 193 NVFP4 tensors read a
/// `.scale` each, the plan reads 401 input scales and refuses nothing, and the vision tower's 333
/// tensors and the next-token layer's 15 are left out by name.
#[test]
fn the_real_nvfp4_headers_plan_the_gguf_files_trunk() {
    let config = json(CONFIG);
    let gguf = json(GGUF);
    let full = kinds(&config);
    let entries = real_entries(&config);
    let checkpoint = Checkpoint::parse(&config).unwrap();
    let want: BTreeMap<String, (Vec<u64>, GgmlType)> = gguf_tensors(&gguf)
        .into_iter()
        .filter(|(name, _)| !name.starts_with("blk.64."))
        .collect();
    assert_eq!(want.len(), 851);
    for q8 in [false, true] {
        let planning = planning(&checkpoint, &config, &entries, q8).unwrap();
        assert!(
            planning.plan.refused.is_empty(),
            "{:?}",
            planning.plan.refused
        );
        assert_eq!(planning.plan.input_scales.len(), 401);
        let got: BTreeMap<&str, &folder::Planned<Step>> = planning
            .planned
            .iter()
            .filter(|p| matches!(p.make, folder::Make::Tensor(_)))
            .map(|p| (p.decl.name.as_str(), p))
            .collect();
        assert_eq!(
            got.keys().copied().collect::<Vec<_>>(),
            want.keys().map(String::as_str).collect::<Vec<_>>()
        );
        for (name, (dims, _)) in &want {
            let decl = &got[name.as_str()].decl;
            assert_eq!(decl.dims, *dims, "{name}");
            assert_eq!(
                GgmlType::from_u32(decl.type_id),
                rule_type(name, q8),
                "{name} (q8 {q8})"
            );
        }

        let scales: Vec<(&folder::Planned<Step>, &str)> = planning
            .planned
            .iter()
            .filter_map(|p| match &p.make {
                folder::Make::Scale { of } => Some((p, of.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(scales.len(), 64 * 3 + 1);
        for (scale, of) in &scales {
            let stem = of.strip_suffix(".weight").unwrap();
            assert_eq!(scale.decl.name, format!("{stem}.scale"));
            assert_eq!(scale.decl.dims, [1], "{of}");
            assert_eq!(
                GgmlType::from_u32(got[of].decl.type_id),
                GgmlType::NVFP4,
                "{of}"
            );
        }

        let mut left_out: BTreeMap<&str, &str> = entries
            .iter()
            .map(|(name, ..)| name.as_str())
            .filter(|name| name.starts_with("model.visual."))
            .map(|name| (name, VISION))
            .collect();
        assert_eq!(left_out.len(), 333);
        left_out.extend(MTP_NAMES.map(|name| (name, MTP)));
        let skipped: BTreeMap<&str, &str> = planning
            .skipped
            .iter()
            .map(|(name, why)| (name.as_str(), why.as_str()))
            .collect();
        assert_eq!(skipped, left_out);
    }

    // Planned as Q8_0 is asked, the embedding keeps the role the map gave it (as stored, the plan
    // reads a BF16 matrix of that role as BF16).
    let planning = planning(&checkpoint, &config, &entries, true).unwrap();
    let got: BTreeMap<&str, &folder::Planned<Step>> = planning
        .planned
        .iter()
        .filter(|p| matches!(p.make, folder::Make::Tensor(_)))
        .map(|p| (p.decl.name.as_str(), p))
        .collect();
    let rows = |from: usize, head: usize| vec![Step::VRows { from, head }];
    let (matrix, projection) = (Use::Matrix(Plain::Q8_0), Use::Q8_0);
    for (l, &attends) in full.iter().enumerate() {
        let blk = |stem: &str| format!("blk.{l}.{stem}");
        let source = |rest: &str| vec![format!("{TEXT}.{l}.{rest}")];
        // Each tensor of the layer: its checkpoint name, role and moves.
        let mut expect: Vec<(&str, String, Use, Vec<Step>)> = vec![
            (
                "ffn_gate.weight",
                "mlp.gate_proj.weight".into(),
                matrix,
                vec![],
            ),
            ("ffn_up.weight", "mlp.up_proj.weight".into(), matrix, vec![]),
            (
                "ffn_down.weight",
                "mlp.down_proj.weight".into(),
                matrix,
                vec![],
            ),
            (
                "attn_norm.weight",
                "input_layernorm.weight".into(),
                Use::Matrix(Plain::F32),
                vec![Step::PlusOne],
            ),
            (
                "post_attention_norm.weight",
                "post_attention_layernorm.weight".into(),
                Use::Matrix(Plain::F32),
                vec![Step::PlusOne],
            ),
        ];
        if attends {
            for (stem, rest) in [
                ("attn_q.weight", "self_attn.q_proj.weight"),
                ("attn_k.weight", "self_attn.k_proj.weight"),
                ("attn_v.weight", "self_attn.v_proj.weight"),
                ("attn_output.weight", "self_attn.o_proj.weight"),
            ] {
                expect.push((stem, rest.into(), projection, vec![]));
            }
            for (stem, rest) in [
                ("attn_q_norm.weight", "self_attn.q_norm.weight"),
                ("attn_k_norm.weight", "self_attn.k_norm.weight"),
            ] {
                expect.push((
                    stem,
                    rest.into(),
                    Use::Matrix(Plain::F32),
                    vec![Step::PlusOne],
                ));
            }
        } else {
            let f32s = Use::Matrix(Plain::F32);
            expect.extend([
                (
                    "attn_qkv.weight",
                    "linear_attn.in_proj_qkv.weight".into(),
                    projection,
                    rows(4096, 128),
                ),
                (
                    "attn_gate.weight",
                    "linear_attn.in_proj_z.weight".into(),
                    projection,
                    rows(0, 128),
                ),
                (
                    "ssm_out.weight",
                    "linear_attn.out_proj.weight".into(),
                    projection,
                    vec![Step::VCols { head: 128 }],
                ),
                (
                    "ssm_conv1d.weight",
                    "linear_attn.conv1d.weight".into(),
                    f32s,
                    rows(4096, 128),
                ),
                (
                    "ssm_alpha.weight",
                    "linear_attn.in_proj_a.weight".into(),
                    f32s,
                    rows(0, 1),
                ),
                (
                    "ssm_beta.weight",
                    "linear_attn.in_proj_b.weight".into(),
                    f32s,
                    rows(0, 1),
                ),
                (
                    "ssm_a",
                    "linear_attn.A_log".into(),
                    f32s,
                    vec![Step::VCols { head: 1 }, Step::NegExp],
                ),
                (
                    "ssm_dt.bias",
                    "linear_attn.dt_bias".into(),
                    f32s,
                    vec![Step::VCols { head: 1 }],
                ),
                (
                    "ssm_norm.weight",
                    "linear_attn.norm.weight".into(),
                    f32s,
                    vec![],
                ),
            ]);
        }
        assert_eq!(
            expect.len(),
            if attends { 11 } else { 14 },
            "layer {l}: the GGUF file's tensors of its kind"
        );
        for (stem, rest, role, steps) in expect {
            let name = blk(stem);
            let m = tensor(&got, &name);
            assert_eq!(m.sources, source(&rest), "{name}");
            assert_eq!(m.role, role, "{name}");
            assert_eq!(m.steps, steps, "{name}");
        }
    }
    for (name, source, role, steps) in [
        (
            "token_embd.weight",
            "model.language_model.embed_tokens.weight",
            matrix,
            vec![],
        ),
        (
            "output_norm.weight",
            "model.language_model.norm.weight",
            Use::Matrix(Plain::F32),
            vec![Step::PlusOne],
        ),
        ("output.weight", "lm_head.weight", matrix, vec![]),
    ] {
        let m = tensor(&got, name);
        assert_eq!(m.sources, [source], "{name}");
        assert_eq!(m.role, role, "{name}");
        assert_eq!(m.steps, steps, "{name}");
    }
}

/// What the HF dense map refuses by name: a next-token tensor the layer does not have (the
/// layer's names are left out one by one, not by their prefix), a tensor in MLX's names, a router,
/// conv taps in MLX's order, a projection missing from a layer, a block whose width the config
/// does not give, and an FP8 weight whose scale is not the one tensor scale the export writes.
#[test]
fn the_hf_dense_map_refuses_what_it_does_not_read_by_name() {
    let config = json(CONFIG);
    let base = real_entries(&config);
    let refuse = |config: &Json, edit: &dyn Fn(&mut Entries)| {
        let mut e = base.clone();
        edit(&mut e);
        let checkpoint = Checkpoint::parse(config).unwrap();
        planning(&checkpoint, config, &e, false)
            .err()
            .expect("refused")
            .to_string()
    };
    let push = |name: &str, shape: Vec<usize>| {
        let name = name.to_string();
        move |e: &mut Entries| e.push((name.clone(), Dtype::Bf16, shape.clone()))
    };
    for name in [
        "mtp.zzz.weight",
        "mtp.layers.0.mlp.gate.weight",
        "mtp.layers.x.mlp.up_proj.weight",
        "mtp.fc.bias",
    ] {
        let e = refuse(&config, &push(name, vec![4, 4]));
        assert!(
            e.contains(name) && e.contains("next-token layer this map names"),
            "{e}"
        );
    }
    for (name, shape, why) in [
        (
            "language_model.model.layers.0.mlp.gate_proj.weight",
            vec![17408, 5120],
            "not a Qwen3.5 tensor",
        ),
        (
            "model.language_model.layers.3.mlp.gate.weight",
            vec![128, 5120],
            "not a Qwen3.5 tensor",
        ),
        (
            "model.language_model.layers.3.mlp.shared_expert.up_proj.weight",
            vec![512, 5120],
            "not a Qwen3.5 tensor",
        ),
        (
            "model.language_model.layers.0.linear_attn.zzz.weight",
            vec![4, 4],
            "not a Qwen3.5 tensor",
        ),
        (
            "model.language_model.layers.64.mlp.up_proj.bias",
            vec![4],
            "layer 64 of 64",
        ),
    ] {
        let e = refuse(&config, &push(name, shape));
        assert!(e.contains(name) && e.contains(why), "{e}");
    }

    let taps = format!("{TEXT}.0.linear_attn.conv1d.weight");
    let e = refuse(&config, &|e| {
        e.iter_mut().find(|t| t.0 == taps).unwrap().2 = vec![10240, 4, 1];
    });
    assert!(
        e.contains(&taps) && e.contains("[10240, 4, 1] are not [channels, 1, kernel]"),
        "{e}"
    );

    let e = refuse(&config, &|e| {
        e.retain(|(n, _, _)| !n.starts_with(&format!("{TEXT}.5.mlp.down_proj.")));
    });
    assert!(
        e.contains("blk.5.ffn_down.weight") && e.contains("is missing"),
        "{e}"
    );
    let mut wider = config.clone();
    wider["text_config"]["intermediate_size"] = json!(17409);
    let e = refuse(&wider, &|_| {});
    assert!(
        e.contains(&format!("{TEXT}.0.mlp.gate_proj.weight"))
            && e.contains("intermediate_size 17409"),
        "{e}"
    );

    // A block-wise scale grid, which no tensor-scale FP8 weight has: the plan refuses the weight
    // ("fits none of" the config's formats), and the map names what it left without a place.
    let q = format!("{TEXT}.3.self_attn.q_proj");
    let grid = |e: &mut Entries| {
        e.iter_mut()
            .find(|t| t.0 == format!("{q}.weight_scale"))
            .unwrap()
            .2 = vec![96, 40];
    };
    let e = refuse(&config, &grid);
    assert!(e.contains(&q), "{e}");
    let mut e = base.clone();
    grid(&mut e);
    let plan = hf::quant_config::plan(
        &Config::parse(&config, Some(&json(HF_QUANT))).unwrap(),
        e.iter().map(|(n, d, s)| (n.as_str(), *d, s.as_slice())),
        crate::arch::module_class,
    );
    assert!(
        plan.refused[&format!("{q}.weight")].contains("fits none of"),
        "{:?}",
        plan.refused
    );
}

/// One row of the HF dense family's root table: the object, the key, a value accepted, a value
/// refused, and what the refusal names ([`config_rows`]'s form).
fn root_rows() -> Vec<Row> {
    serde_json::from_str(
        r#"[
        ["", "model_type", "qwen3_5", "qwen3_5_moe", "model_type"],
        ["", "image_token_id", 248056, "x", "image_token_id"],
        ["", "language_model_only", false, "x", "language_model_only"],
        ["", "dtype", "bfloat16", 7, "dtype"],
        ["", "tie_word_embeddings", false, true, "tie_word_embeddings"]
    ]"#,
    )
    .unwrap()
}

/// The HF dense family's own root table, row by row: a key accepts its value and refuses another
/// by name; a key no table names is refused, so are the MLX family's root keys (`quantization`, a
/// root `eos_token_id`, the sampling defaults), which this export does not have; the text model's
/// keys are the MLX dense family's, so the rows of its `text_config` hold here too. A config that
/// names the dense class without `modelopt` is the MLX family's, and refused for the HF keys.
#[test]
fn the_hf_dense_config_accepts_its_values_and_refuses_others_by_name() {
    let base = json(CONFIG);
    let rows: Vec<Row> = root_rows()
        .into_iter()
        .chain(config_rows().into_iter().filter(|r| r.0 == "/text_config"))
        .collect();
    let run = |c: &Json| -> Result<(), String> {
        Checkpoint::parse(c).map_err(|e| e.to_string())?;
        Config::parse(c, None).map(drop).map_err(|e| e.to_string())
    };
    let with = |at: &str, key: &str, value: &Json| {
        let mut c = base.clone();
        let object = c.pointer_mut(at).unwrap().as_object_mut().unwrap();
        match value {
            Json::String(s) if s == "$" => {}
            Json::Null => {
                object.remove(key);
            }
            v => {
                object.insert(key.to_string(), v.clone());
            }
        }
        c
    };
    let mut missed = vec![];
    run(&base).unwrap();
    for (at, key, allowed, refused, what) in &rows {
        if let Err(e) = run(&with(at, key, allowed)) {
            missed.push(format!("{at} {key}={allowed}: refused: {e}"));
        }
        match run(&with(at, key, refused)) {
            Err(e) if e.contains(what.as_str()) => {}
            other => missed.push(format!("{at} {key}={refused}: {other:?}")),
        }
    }
    for (key, value) in [
        ("zzz", json!(1)),
        ("quantization", json!({"bits": 4})),
        ("eos_token_id", json!([248046, 248044])),
        ("generation_config", json!({})),
        ("do_sample", json!(true)),
        ("temperature", json!(0.6)),
        ("top_k", json!(40)),
        ("top_p", json!(0.8)),
    ] {
        match run(&with("", key, &value)) {
            Err(e) if e.contains(key) => {}
            other => missed.push(format!("root {key}: {other:?}")),
        }
    }
    assert!(
        missed.is_empty(),
        "accepted or unnamed:\n{}",
        missed.join("\n")
    );

    // Without ModelOpt's producer the class is MLX's, whose root has no `dtype`.
    for method in [json!("fp8"), Json::Null] {
        let mut c = base.clone();
        match method {
            Json::Null => {
                c.as_object_mut().unwrap().remove("quantization_config");
            }
            m => c["quantization_config"]["quant_method"] = m,
        }
        let Err(e) = Checkpoint::parse(&c) else {
            panic!("a dense config whose quant_method is not modelopt was not refused");
        };
        let e = e.to_string();
        assert!(e.contains("dtype"), "{e}");
    }
}

/// Every key of the HF dense family's own root table has a row, so a key added to it is accepted
/// and refused by a gate before it is read.
#[test]
fn every_key_of_the_hf_dense_root_table_has_a_row() {
    let rows = root_rows();
    let missing: Vec<&str> = ROOT_HF_DENSE
        .iter()
        .map(|(key, _)| *key)
        .filter(|key| !rows.iter().any(|r| r.0.is_empty() && r.1 == *key))
        .collect();
    assert!(missing.is_empty(), "no row: {missing:?}");
}

/// The e4m3 value of the byte `b`.
fn e4m3(b: u8) -> f32 {
    let (exp, man) = (i32::from((b >> 3) & 0xF), f64::from(b & 7));
    let mag = if exp == 0 {
        man / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + man / 8.0) * 2f64.powi(exp - 7)
    };
    (if b & 0x80 == 0 { mag } else { -mag }) as f32
}

/// The F8_E4M3 bytes of a `rows` by `k` weight: a counting pattern of codes that are numbers,
/// shifted by `seed`.
fn fp8_codes(rows: usize, k: usize, seed: usize) -> Vec<u8> {
    (0..rows * k)
        .map(|i| ((i * 37 + seed * 11) % 256) as u8)
        .map(|b| if b & 0x7F == 0x7F { b ^ 1 } else { b })
        .collect()
}

/// The one F32 tensor scale of the `seed`th FP8 projection.
fn fp8_scale(seed: usize) -> f32 {
    (seed + 1) as f32 / 64.0
}

/// An F32 tensor of no dimension, as the export writes each scale.
fn scalar(name: String, value: f32) -> Stored {
    (name, "F32", vec![], value.to_le_bytes().to_vec())
}

/// An FP8 linear `stem` as the export stores it: its `rows` by `k` weight, one F32 `weight_scale`
/// and one F32 `input_scale`, neither of a dimension.
fn fp8_linear(stem: &str, rows: usize, k: usize, seed: usize) -> Vec<Stored> {
    vec![
        (
            format!("{stem}.weight"),
            "F8_E4M3",
            vec![rows, k],
            fp8_codes(rows, k, seed),
        ),
        scalar(format!("{stem}.weight_scale"), fp8_scale(seed)),
        scalar(format!("{stem}.input_scale"), 1.0),
    ]
}

/// An NVFP4 linear `stem` as the export stores it: its codes (`k / 2` bytes a row), a block scale
/// of 1.0 per 16 values, and the F32 tensor scale and input scale.
fn nvfp4_linear(stem: &str, rows: usize, k: usize) -> Vec<Stored> {
    let codes: Vec<u8> = (0..rows * k / 2).map(|i| (i * 29 % 251) as u8).collect();
    vec![
        (format!("{stem}.weight"), "U8", vec![rows, k / 2], codes),
        (
            format!("{stem}.weight_scale"),
            "F8_E4M3",
            vec![rows, k / 16],
            vec![0x38; rows * k / 16],
        ),
        scalar(format!("{stem}.weight_scale_2"), 0.5),
        scalar(format!("{stem}.input_scale"), 1.0),
    ]
}

const HIDDEN: usize = 128;
const FF: usize = 128;

/// A list of modules: name, rows and k.
type Modules = Vec<(String, usize, usize)>;

/// The small folder's NVFP4 modules and FP8 modules. The head, and the three projections of each
/// layer's block; a delta-rule layer's three, then an attention layer's four.
fn small_modules() -> (Modules, Modules) {
    let p = "model.language_model";
    let nvfp4: Modules = std::iter::once(("lm_head".to_string(), 8, HIDDEN))
        .chain((0..2).flat_map(|l| {
            [
                ("gate_proj", FF, HIDDEN),
                ("up_proj", FF, HIDDEN),
                ("down_proj", HIDDEN, FF),
            ]
            .map(|(s, r, k)| (format!("{p}.layers.{l}.mlp.{s}"), r, k))
        }))
        .collect();
    let fp8: Modules = [
        ("layers.0.linear_attn.in_proj_qkv", 640, HIDDEN),
        ("layers.0.linear_attn.in_proj_z", 384, HIDDEN),
        ("layers.0.linear_attn.out_proj", HIDDEN, 384),
        ("layers.1.self_attn.q_proj", 256, HIDDEN),
        ("layers.1.self_attn.k_proj", 64, HIDDEN),
        ("layers.1.self_attn.v_proj", 64, HIDDEN),
        ("layers.1.self_attn.o_proj", HIDDEN, 128),
    ]
    .map(|(s, r, k)| (format!("{p}.{s}"), r, k))
    .to_vec();
    (nvfp4, fp8)
}

/// A folder `dir` of two layers (gated delta rule, then attention) of the dense Qwen3.5-class
/// model as small as the blocks allow, in ModelOpt's layout: transformers' names, the FP8
/// projections of one tensor scale, the NVFP4 blocks and head, BF16 gammas without the `+ 1`, the
/// taps as `[channels, 1, kernel]`, a vision tensor and the next-token layer's, with `extra`
/// beside them.
fn write_small_nvfp4_folder(dir: &Path, extra: Vec<Stored>) {
    let p = "model.language_model";
    let (nvfp4, fp8) = small_modules();
    let quantized: serde_json::Map<String, Json> = nvfp4
        .iter()
        .map(|(m, ..)| (m.clone(), json!({"quant_algo": "NVFP4", "group_size": 16})))
        .chain(
            fp8.iter()
                .map(|(m, ..)| (m.clone(), json!({"quant_algo": "FP8"}))),
        )
        .collect();
    let config = json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "dtype": "bfloat16", "image_token_id": 3, "language_model_only": false,
        "tie_word_embeddings": false,
        "quantization_config": {
            "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
            "ignore": ["mtp*"], "quantized_layers": quantized,
        },
        "text_config": {
            "model_type": "qwen3_5_text", "attn_output_gate": true, "dtype": "bfloat16",
            "num_hidden_layers": 2, "max_position_embeddings": 4096, "hidden_size": HIDDEN,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
            "intermediate_size": FF, "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 64, "linear_num_key_heads": 2,
            "linear_num_value_heads": 6, "linear_value_head_dim": 64,
            "layer_types": ["linear_attention", "full_attention"],
            "full_attention_interval": 4, "rms_norm_eps": 1e-6, "vocab_size": 8,
            "bos_token_id": 2, "eos_token_id": 2, "pad_token_id": null,
            "partial_rotary_factor": 0.5, "hidden_act": "silu",
            "output_gate_type": "swish", "tie_word_embeddings": false,
            "mamba_ssm_dtype": "float32",
            "rope_parameters": {"mrope_interleaved": true, "mrope_section": [4, 2, 2],
                "rope_theta": 1e6, "rope_type": "default", "partial_rotary_factor": 0.5},
        },
    });
    let mut t: Vec<Stored> = vec![
        bf16_tensor(&format!("{p}.embed_tokens.weight"), &[8, HIDDEN], |_| {
            0x3f80
        }),
        bf16_tensor(&format!("{p}.norm.weight"), &[HIDDEN], |_| 0x3f00),
    ];
    for (module, rows, k) in &nvfp4 {
        t.extend(nvfp4_linear(module, *rows, *k));
    }
    for (seed, (module, rows, k)) in fp8.iter().enumerate() {
        t.extend(fp8_linear(module, *rows, *k, seed));
    }
    for l in 0..2 {
        for (norm, bf16) in [
            ("input_layernorm", 0x3f00),
            ("post_attention_layernorm", 0x3e80),
        ] {
            t.push(bf16_tensor(
                &format!("{p}.layers.{l}.{norm}.weight"),
                &[HIDDEN],
                |_| bf16,
            ));
        }
    }
    let m = format!("{p}.layers.0.linear_attn");
    for (stem, rows) in [("in_proj_a", 6), ("in_proj_b", 6)] {
        t.push(bf16_tensor(
            &format!("{m}.{stem}.weight"),
            &[rows, HIDDEN],
            |_| 0x3f80,
        ));
    }
    t.push(bf16_tensor(
        &format!("{m}.conv1d.weight"),
        &[640, 1, 4],
        |_| 0x3f80,
    ));
    t.push(bf16_tensor(&format!("{m}.A_log"), &[6], |_| 0x3f80));
    t.push(bf16_tensor(&format!("{m}.dt_bias"), &[6], |_| 0x3f80));
    t.push(bf16_tensor(&format!("{m}.norm.weight"), &[64], |_| 0x3f00));
    for norm in ["q_norm", "k_norm"] {
        t.push(bf16_tensor(
            &format!("{p}.layers.1.self_attn.{norm}.weight"),
            &[32],
            |_| 0x3e80,
        ));
    }
    t.push(bf16_tensor(
        "model.visual.patch_embed.proj.weight",
        &[2, 3],
        |_| 0x3f80,
    ));
    t.push(bf16_tensor("mtp.fc.weight", &[HIDDEN, 2 * HIDDEN], |_| {
        0x3f80
    }));
    t.extend(extra);
    write_folder(dir, &t, &config, &tokenizer("qwen35"));
}

/// A small folder in the export's layout opens as a `qwen35` file: the engine's own header reader
/// takes it as the dense variant, the NVFP4 blocks and head keep their codes and read a `.scale`
/// each, every FP8 projection is Q8_0 (the part source makes each block of 32 values of the
/// exact e4m3 value times the weight's one F32 scale, rounded by ggml's reference rule), a gamma
/// is read as stored plus one but the delta rule's own, the vision tensor and the next-token
/// layer's are left out by name and the draft is refused by name, and an `mtp.` tensor the
/// layer does not have refuses the folder.
#[test]
fn a_small_nvfp4_folder_opens_as_a_qwen35_file() {
    use model::arch::qwen35moe::hparams::{FfnKind, Hparams, Kind, Variant};
    use model::part::{Source, Take};

    let dir = temp_dir("qwen35-nvfp4-open");
    write_small_nvfp4_folder(&dir, vec![]);
    let opened = folder::open(&dir).unwrap_or_else(|e| panic!("{e}"));

    let hp = Hparams::read(&opened.header).unwrap();
    assert_eq!(hp.variant, Variant::Qwen35);
    assert_eq!(hp.kinds, [Kind::DeltaRule, Kind::Attention]);
    assert_eq!(hp.ffns, [FfnKind::Dense; 2]);
    assert_eq!(hp.ff, Some(128));
    let ty = |name: &str| opened.header.find(name).unwrap().1.ty;
    for name in [
        "output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.1.ffn_down.weight",
    ] {
        assert_eq!(ty(name), GgmlType::NVFP4, "{name}");
        let scale = name.replace(".weight", ".scale");
        assert_eq!(ty(&scale), GgmlType::F32, "{scale}");
    }
    for name in [
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_out.weight",
        "blk.1.attn_q.weight",
        "blk.1.attn_k.weight",
        "blk.1.attn_v.weight",
        "blk.1.attn_output.weight",
    ] {
        assert_eq!(ty(name), GgmlType::Q8_0, "{name}");
        let scale = name.replace(".weight", ".scale");
        assert!(opened.header.find(&scale).is_none(), "{scale}");
    }
    assert_eq!(ty("token_embd.weight"), GgmlType::BF16);
    for name in [
        "output_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.1.post_attention_norm.weight",
        "blk.0.ssm_norm.weight",
        "blk.1.attn_q_norm.weight",
        "blk.0.ssm_conv1d.weight",
        "blk.0.ssm_alpha.weight",
    ] {
        assert_eq!(ty(name), GgmlType::F32, "{name}");
    }
    assert_eq!(
        opened
            .header
            .find("blk.0.ssm_conv1d.weight")
            .unwrap()
            .1
            .dims,
        [4, 640]
    );

    let want = [
        "blk.0.attn_norm.weight",
        "blk.1.post_attention_norm.weight",
        "output_norm.weight",
        "blk.0.ssm_norm.weight",
        "blk.1.attn_q.weight",
        "blk.1.attn_k.weight",
        "blk.1.attn_v.weight",
        "blk.1.attn_output.weight",
    ];
    let planned = opened
        .source
        .plan(|name, _, _| {
            Ok::<_, String>(if want.contains(&name) {
                Take::Whole
            } else {
                Take::Skip
            })
        })
        .unwrap();
    assert_eq!(planned.len(), want.len());
    let (_, fp8) = small_modules();
    for p in &planned {
        let name = p.info.name.as_str();
        let bytes = opened.source.read(p).unwrap().spans().concat();
        let f32s = |bytes: &[u8]| -> Vec<f32> {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        };
        match name {
            // BF16 0.5 stored; the program reads 1.5 (and 0.25 as 1.25) but of the delta rule's
            // own gamma.
            "blk.0.attn_norm.weight" | "output_norm.weight" => {
                assert_eq!(f32s(&bytes), vec![1.5; HIDDEN], "{name}");
            }
            "blk.1.post_attention_norm.weight" => {
                assert_eq!(f32s(&bytes), vec![1.25; HIDDEN], "{name}");
            }
            "blk.0.ssm_norm.weight" => assert_eq!(f32s(&bytes), vec![0.5; 64], "{name}"),
            _ => {
                let module = format!(
                    "model.language_model.layers.1.self_attn.{}",
                    match name {
                        "blk.1.attn_q.weight" => "q_proj",
                        "blk.1.attn_k.weight" => "k_proj",
                        "blk.1.attn_v.weight" => "v_proj",
                        _ => "o_proj",
                    }
                );
                let seed = fp8.iter().position(|(m, ..)| *m == module).unwrap();
                let (rows, k) = (fp8[seed].1, fp8[seed].2);
                assert_eq!(
                    (p.info.ty, p.info.dims.clone()),
                    (GgmlType::Q8_0, vec![k as u64, rows as u64]),
                    "{name}"
                );
                let codes = fp8_codes(rows, k, seed);
                let want: Vec<u8> = codes
                    .as_chunks::<32>()
                    .0
                    .iter()
                    .flat_map(|chunk| {
                        let x: [f32; 32] =
                            std::array::from_fn(|j| e4m3(chunk[j]) * fp8_scale(seed));
                        Q8Block::quantize_ref(&x).unwrap().to_bytes()
                    })
                    .collect();
                assert_eq!(bytes.len(), rows * k / 32 * 34, "{name}");
                assert!(bytes == want, "{name}: Q8_0 of the exact values");
            }
        }
    }

    let mut vision: Vec<&str> = opened
        .skipped
        .iter()
        .filter(|(_, why)| why.as_str() == VISION)
        .map(|(name, _)| name.as_str())
        .collect();
    vision.sort_unstable();
    assert_eq!(vision, ["model.visual.patch_embed.proj.weight"]);
    assert_eq!(
        opened.skipped.get("mtp.fc.weight").map(String::as_str),
        Some(MTP)
    );
    assert_eq!(opened.skipped.len(), 2);
    let refused = opened.draft.as_ref().err().expect("the draft is refused");
    let refused = refused.to_string();
    assert!(
        refused.contains("mtp.fc.weight") && refused.contains("does not read"),
        "{refused}"
    );
    std::fs::remove_dir_all(&dir).unwrap();

    let dir = temp_dir("qwen35-nvfp4-mtp");
    write_small_nvfp4_folder(
        &dir,
        vec![bf16_tensor("mtp.layers.0.zzz.weight", &[4, 4], |_| 0x3f80)],
    );
    let refused = folder::open(&dir).err().expect("the folder is refused");
    let refused = refused.to_string();
    assert!(
        refused.contains("mtp.layers.0.zzz.weight")
            && refused.contains("next-token layer this map names"),
        "{refused}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
