//! The dense family's gates ([`Family::Qwen35`], [`Family::Qwen35Text`]): the real 27B folder's
//! shard headers and config read as the GGUF file of the same model, the config's tables row by
//! row, the tensors the map refuses by name, and a small folder of real bytes opened as a `qwen35`
//! file.

use std::collections::BTreeMap;
use std::path::PathBuf;

use gguf::GgmlType;
use gguf::write::{Layout, TensorDecl};
use hf::quant_config::Config;
use safetensors::Dtype;
use serde_json::json;

use super::*;
use crate::folder;
use crate::test_support::{
    Entries, Stored, bf16_tensor, entries, gguf_tensors, json, key_json, mlx_linear, temp_dir,
    tokenizer, write_folder,
};

const CONFIG_38: &str = "quant-configs/mlx-community__Qwen3.8-27B-4bit__config.json.gz";
const CONFIG_36: &str = "quant-configs/mlx-community__Qwen3.6-27B-4bit__config.json.gz";
const GGUF: &str = "checkpoint-headers/qwen3.8-27b-q4km-gguf.header.json.gz";
const TEXT: &str = "language_model.model.layers";

/// The tensors of the 27B folder's three committed shard headers: 2,180 of them, the vision
/// tower's 333 among them.
fn real_entries() -> Entries {
    (1..=3)
        .flat_map(|n| {
            let name = format!(
                "checkpoint-headers/qwen3.8-27b-mlx-4bit-model-0000{n}-of-00003.header.json.gz"
            );
            let mut header = json(&name);
            header.as_object_mut().unwrap().remove("__metadata__");
            entries(&header)
        })
        .collect()
}

fn planning(
    checkpoint: &Checkpoint,
    config: &Json,
    entries: &Entries,
) -> Result<folder::Planning<Step>, folder::Error> {
    folder::plan_tensors(
        checkpoint,
        &Config::parse(config, None).unwrap(),
        entries,
        false,
    )
}

/// The fixture value `[type, value]` as the header key it is.
fn key_value(v: &Json) -> Value {
    let [ty, x] = v.as_array().unwrap().as_slice() else {
        panic!("{v}: not [type, value]");
    };
    match ty.as_str().unwrap() {
        "U32" => Value::U32(x.as_u64().unwrap() as u32),
        "I32" => Value::I32(x.as_i64().unwrap() as i32),
        "F32" => Value::F32(x.as_f64().unwrap() as f32),
        "BOOL" => Value::Bool(x.as_bool().unwrap()),
        "STR" => Value::String(x.as_str().unwrap().into()),
        "ARR" => {
            let [item, values] = x.as_array().unwrap().as_slice() else {
                panic!("{x}: not [element type, values]");
            };
            assert_eq!(item, "I32", "{x}");
            Value::Array(
                values
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|i| Value::I32(i.as_i64().unwrap() as i32))
                    .collect(),
            )
        }
        other => panic!("{other}"),
    }
}

/// The header-only file of `keys` and `tensors`.
fn split_of(keys: &[(String, Value)], tensors: Vec<TensorDecl>) -> gguf::Split {
    let layout = Layout::new(keys, tensors).unwrap();
    gguf::Split::header_only(PathBuf::from("qwen35-header"), &layout).unwrap()
}

/// What a folder's tokenizer files make of the header: the model, the pre-tokenizer row and an
/// empty token for each id of the vocabulary. A header fixture carries no tokenizer, so both
/// sides are given the same.
fn tokenizer_keys(vocab: u32) -> Vec<(String, Value)> {
    vec![
        ("tokenizer.ggml.model".into(), Value::String("gpt2".into())),
        ("tokenizer.ggml.pre".into(), Value::String("qwen35".into())),
        (
            "tokenizer.ggml.tokens".into(),
            Value::Array(vec![Value::String(String::new()); vocab as usize]),
        ),
    ]
}

/// The real 27B folder's tensors plan as the tensors of the GGUF file of the same model that its
/// trunk holds: the same names at the same ggml dims, every matrix MLX's 4-bit blocks and every
/// other tensor F32, each feed-forward projection made of the checkpoint projection of its own
/// name, and the vision tower left out. Its next-token layer (`blk.64`) is the one part the
/// folder has no tensors of.
#[test]
fn the_real_27b_headers_plan_the_gguf_files_trunk() {
    let config = json(CONFIG_38);
    let gguf = json(GGUF);
    let entries = real_entries();
    assert_eq!(entries.len(), 2180);
    let checkpoint = Checkpoint::parse(&config).unwrap();
    let planning = planning(&checkpoint, &config, &entries).unwrap();
    assert!(
        planning.plan.refused.is_empty(),
        "{:?}",
        planning.plan.refused
    );
    let want: BTreeMap<String, (Vec<u64>, GgmlType)> = gguf_tensors(&gguf)
        .into_iter()
        .filter(|(name, _)| !name.starts_with("blk.64."))
        .collect();
    let got: BTreeMap<&str, &folder::Planned<Step>> = planning
        .planned
        .iter()
        .map(|p| (p.decl.name.as_str(), p))
        .collect();
    assert_eq!(got.len(), 851);
    assert_eq!(
        got.keys().copied().collect::<Vec<_>>(),
        want.keys().map(String::as_str).collect::<Vec<_>>()
    );
    for (name, (dims, ty)) in &want {
        let mlx = GgmlType::from_u32(got[name.as_str()].decl.type_id);
        assert_eq!(got[name.as_str()].decl.dims, *dims, "{name}");
        // The GGUF file's K-quants are MLX's 4-bit blocks here; its F32 tensors are the folder's
        // BF16 vectors.
        assert_eq!(*ty == GgmlType::F32, mlx == GgmlType::F32, "{name}");
        assert!(
            matches!(mlx, GgmlType::F32 | GgmlType::MLX_A4),
            "{name}: {mlx}"
        );
    }
    for l in 0..64 {
        for (stem, projection) in [
            ("ffn_gate", "gate_proj"),
            ("ffn_up", "up_proj"),
            ("ffn_down", "down_proj"),
        ] {
            let folder::Make::Tensor(m) = &got[format!("blk.{l}.{stem}.weight").as_str()].make
            else {
                panic!("blk.{l}.{stem}: a scale beside a type that reads none");
            };
            assert_eq!(
                m.sources,
                [format!("{TEXT}.{l}.mlp.{projection}.weight")],
                "blk.{l}.{stem}"
            );
            assert_eq!(m.role, Use::Matrix(Plain::Q8_0), "blk.{l}.{stem}");
            assert!(m.steps.is_empty(), "blk.{l}.{stem}: {:?}", m.steps);
        }
    }
    let mut vision: Vec<&str> = entries
        .iter()
        .map(|(n, _, _)| n.as_str())
        .filter(|n| n.starts_with("vision_tower."))
        .collect();
    vision.sort_unstable();
    assert_eq!(vision.len(), 333);
    assert_eq!(
        planning
            .skipped
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vision
    );
    assert!(planning.skipped.values().all(|why| why == VISION));
}

/// The 27B header of the folder and the header of the GGUF file read as one model: the folder's
/// header keys are the file's `qwen35.*` keys but for the next-token layer (the file counts it in
/// `block_count` and names it in `nextn_predict_layers`), and the file reader's [`ModelSpec`] of
/// the two headers, tensors and keys alone, is the same: the layers, the full-attention interval
/// of 4, the 24, 4 and 256 heads, the delta rule's 16 key and 48 value heads of 128, the width
/// 17408 of every dense block, the vocabulary and the rope.
#[test]
fn the_real_27b_header_reads_as_the_gguf_files_model_spec() {
    let config = json(CONFIG_38);
    let gguf = json(GGUF);
    let checkpoint = Checkpoint::parse(&config).unwrap();
    let planning = planning(&checkpoint, &config, &real_entries()).unwrap();
    let folder_keys = checkpoint.keys(&[], |_| unreachable!("no PLE")).unwrap();
    let got: serde_json::Map<String, Json> = folder_keys
        .iter()
        .map(|(k, v)| (k.clone(), key_json(v)))
        .collect();
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

    let vocab = checkpoint.vocab.size();
    assert_eq!(u64::from(vocab), gguf["token_count"].as_u64().unwrap());
    let mut folder_header = vec![(
        gguf::GENERAL_ARCHITECTURE.to_string(),
        Value::String("qwen35".into()),
    )];
    folder_header.extend(folder_keys);
    folder_header.extend(tokenizer_keys(vocab));
    let folder_split = split_of(
        &folder_header,
        planning.planned.iter().map(|p| p.decl.clone()).collect(),
    );

    let mut file_header: Vec<(String, Value)> = gguf["keys"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(k, _)| !k.starts_with("tokenizer."))
        .map(|(k, v)| (k.clone(), key_value(v)))
        .collect();
    file_header.extend(tokenizer_keys(vocab));
    let file_split = split_of(
        &file_header,
        gguf["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| {
                let dims: Vec<u64> = t[1]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_u64().unwrap())
                    .collect();
                let type_id = t[2].as_u64().unwrap() as u32;
                TensorDecl {
                    name: t[0].as_str().unwrap().to_string(),
                    type_id,
                    nbytes: folder::size_of(type_id, &dims).unwrap(),
                    dims,
                }
            })
            .collect(),
    );

    let folder_spec = model::arch::spec(&folder_split).unwrap().spec;
    let file_spec = model::arch::spec(&file_split).unwrap().spec;
    assert_eq!(folder_spec.arch, model::arch::models::Arch::Qwen35);
    assert_eq!(folder_spec.layers.len(), 64);
    assert_eq!(folder_spec, file_spec);
}

/// The folder of Qwen3.6's 27B, the same layout under another release, and the folder
/// lmstudio-community publishes of Qwen3.8's (the config of mlx-community's, byte for byte)
/// read as the 27B config does; the text-only class's config, the text model's own object under
/// the class `Qwen3_5ForCausalLM`, makes the same header and the same tensors.
#[test]
fn the_other_27b_folders_and_the_text_only_class_read_as_the_same_model() {
    let config = json(CONFIG_38);
    let entries = real_entries();
    let checkpoint = Checkpoint::parse(&config).unwrap();
    let keys = |c: &Checkpoint| c.keys(&[], |_| unreachable!("no PLE")).unwrap();
    let names = |p: &folder::Planning<Step>| {
        p.planned
            .iter()
            .map(|p| (p.decl.name.clone(), p.decl.dims.clone(), p.decl.type_id))
            .collect::<Vec<_>>()
    };
    let base = names(&planning(&checkpoint, &config, &entries).unwrap());

    let older = json(CONFIG_36);
    let older_checkpoint = Checkpoint::parse(&older).unwrap();
    assert_eq!(keys(&older_checkpoint), keys(&checkpoint));
    assert_eq!(
        names(&planning(&older_checkpoint, &older, &entries).unwrap()),
        base
    );

    let mut text_only = config["text_config"].clone();
    for key in [
        "quantization",
        "quantization_config",
        "transformers_version",
    ] {
        text_only[key] = config[key].clone();
    }
    text_only["architectures"] = json!(["Qwen3_5ForCausalLM"]);
    let from_text = Checkpoint::parse(&text_only).unwrap();
    assert_eq!(keys(&from_text), keys(&checkpoint));
    assert_eq!(
        names(&planning(&from_text, &text_only, &entries).unwrap()),
        base
    );
}

/// One row of the dense config's tables: the object, the key, a value accepted ("$" keeps the
/// checkpoint's, `null` removes the key), a value refused, and what the refusal names.
type Row = (String, String, Json, Json, String);

fn config_rows() -> Vec<Row> {
    serde_json::from_str(
        r#"[
        ["", "model_type", "qwen3_5", "qwen3_5_moe", "model_type"],
        ["", "quantization", "$", {"bits": 4}, "quantization"],
        ["", "eos_token_id", [248046, 248044], [], "eos_token_id"],
        ["", "image_token_id", 248056, "x", "image_token_id"],
        ["", "language_model_only", true, "x", "language_model_only"],
        ["", "generation_config", {}, 7, "generation_config"],
        ["", "do_sample", false, "x", "do_sample"],
        ["", "temperature", 0.6, "x", "temperature"],
        ["", "top_k", 40, "x", "top_k"],
        ["", "top_p", 0.8, "x", "top_p"],
        ["", "tie_word_embeddings", false, true, "tie_word_embeddings"],
        ["/text_config", "model_type", "qwen3_5_text", "qwen3_5_moe_text", "model_type"],
        ["/text_config", "full_attention_interval", 4, "x", "full_attention_interval"],
        ["/text_config", "attn_output_gate", true, false, "attn_output_gate"],
        ["/text_config", "output_gate_type", "swish", "sigmoid", "output_gate_type"],
        ["/text_config", "mlp_only_layers", [], [3], "mlp_only_layers"],
        ["/text_config", "intermediate_size", "$", 0, "intermediate_size"],
        ["/text_config", "hidden_act", "silu", "gelu", "hidden_act"],
        ["/text_config", "tie_word_embeddings", false, true, "tie_word_embeddings"],
        ["/text_config", "mtp_num_hidden_layers", 1, "x", "mtp_num_hidden_layers"],
        ["/text_config", "mtp_use_dedicated_embeddings", false, "x", "mtp_use_dedicated_embeddings"],
        ["/text_config", "num_experts", null, 8, "num_experts"],
        ["/text_config", "num_experts_per_tok", null, 2, "num_experts_per_tok"],
        ["/text_config", "moe_intermediate_size", null, 512, "moe_intermediate_size"],
        ["/text_config", "shared_expert_intermediate_size", null, 512, "shared_expert_intermediate_size"]
    ]"#,
    )
    .unwrap()
}

/// The dense config's rows: each key accepts its value and refuses another by name, a key no
/// table names is refused in each object, and the routed families' keys (`num_experts`, the
/// shared expert's width) are refused as foreign. A text-only config that holds a `text_config`
/// is refused, so is a multimodal class with none, so is a dense config under the routed class,
/// and so are `layer_types` that no interval makes.
#[test]
fn the_dense_config_accepts_its_values_and_refuses_others_by_name() {
    let base = json(CONFIG_38);
    let rows = config_rows();
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
    for at in ["", "/text_config"] {
        let mut c = base.clone();
        c.pointer_mut(at)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("zzz".into(), json!(1));
        match run(&c) {
            Err(e) if e.contains("zzz") => {}
            other => missed.push(format!("{at} zzz: {other:?}")),
        }
    }
    for (at, key, allowed, refused, what) in &rows {
        if let Err(e) = run(&with(at, key, allowed)) {
            missed.push(format!("{at} {key}={allowed}: refused: {e}"));
        }
        match run(&with(at, key, refused)) {
            Err(e) if e.contains(what.as_str()) => {}
            other => missed.push(format!("{at} {key}={refused}: {other:?}")),
        }
    }
    let mut c = base.clone();
    c["text_config"]["layer_types"] = json!(
        (0..64)
            .map(|l| if l % 4 == 3 || l == 5 {
                "full_attention"
            } else {
                "linear_attention"
            })
            .collect::<Vec<_>>()
    );
    match run(&c) {
        Err(e) if e.contains("layer_types") => {}
        other => missed.push(format!("layer_types of no interval: {other:?}")),
    }
    let mut c = base.clone();
    c["architectures"] = json!(["Qwen3_5ForCausalLM"]);
    match run(&c) {
        Err(e) if e.contains("text_config") => {}
        other => missed.push(format!("text-only with a text_config: {other:?}")),
    }
    let mut c = base.clone();
    c.as_object_mut().unwrap().remove("text_config");
    match run(&c) {
        Err(e) if e.contains("text_config") => {}
        other => missed.push(format!("no text_config: {other:?}")),
    }
    let mut c = base.clone();
    c["architectures"] = json!(["Qwen3_5MoeForConditionalGeneration"]);
    match run(&c) {
        Err(e) if e.contains("do_sample") || e.contains("model_type") => {}
        other => missed.push(format!("a dense config as the MoE class: {other:?}")),
    }
    assert!(
        missed.is_empty(),
        "accepted or unnamed:\n{}",
        missed.join("\n")
    );
}

/// Every key of the dense family's own tables has a row in [`config_rows`], so a key added to a
/// table is accepted and refused by a gate before it is read.
#[test]
fn every_key_of_the_dense_tables_has_a_row() {
    let tables = [
        ("", [ROOT_MLX, ROOT_35].concat()),
        ("/text_config", {
            let mut keys = TEXT_35.to_vec();
            keys.extend_from_slice(trunk::Ffn::Dense.keys());
            keys
        }),
    ];
    let rows = config_rows();
    let missing: Vec<String> = tables
        .iter()
        .flat_map(|(at, table)| table.iter().map(move |(key, _)| (*at, *key)))
        .filter(|&(at, key)| {
            !rows
                .iter()
                .any(|r| (r.0.as_str(), r.1.as_str()) == (at, key))
        })
        .map(|(at, key)| format!("{at} {key}"))
        .collect();
    assert!(missing.is_empty(), "no row: {missing:?}");
}

/// What the dense map refuses by name: a router, a shared expert or a stack of experts (the
/// routed families' tensors), a block whose width the config does not give, a projection missing
/// from a layer, and an MLX folder's next-token layer under the text prefix.
#[test]
fn the_dense_map_refuses_what_it_does_not_read_by_name() {
    let config = json(CONFIG_38);
    let base = real_entries();
    let refuse = |config: &Json, edit: &dyn Fn(&mut Entries)| {
        let mut e = base.clone();
        edit(&mut e);
        let checkpoint = Checkpoint::parse(config).unwrap();
        planning(&checkpoint, config, &e)
            .err()
            .expect("refused")
            .to_string()
    };
    for (stem, shape) in [
        ("mlp.gate.weight", vec![128, 5120]),
        ("mlp.shared_expert.gate_proj.weight", vec![512, 5120]),
        ("mlp.shared_expert_gate.weight", vec![1, 5120]),
        ("mlp.switch_mlp.gate_proj.weight", vec![8, 512, 5120]),
    ] {
        let e = refuse(&config, &|e| {
            e.push((format!("{TEXT}.3.{stem}"), Dtype::Bf16, shape.clone()));
        });
        assert!(
            e.contains(&format!("{TEXT}.3.{stem}")) && e.contains("not a Qwen3.5 tensor"),
            "{e}"
        );
    }
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
    let e = refuse(&config, &|e| {
        e.push((
            "language_model.mtp.fc.weight".into(),
            Dtype::Bf16,
            vec![5120, 10240],
        ));
    });
    assert!(
        e.contains("language_model.mtp.fc.weight") && e.contains("next-token layer"),
        "{e}"
    );
}

/// The distinguishable weights of an MLX linear: `mlx_linear`'s words, each byte xor `seed`.
fn seeded_linear(stem: &str, rows: usize, k: usize, seed: u8) -> Vec<Stored> {
    let mut t = mlx_linear(stem, 4, &[rows], k);
    t[0].3.iter_mut().for_each(|b| *b ^= seed);
    t
}

/// A folder `dir` of two layers (gated delta rule, then attention) of a dense Qwen3.5-class model
/// as small as the groups allow, in MLX's names: the multimodal class with a vision tensor, or
/// the text-only class. A tensor of the next-token layer sits beside the trunk's.
fn write_small_dense_folder(dir: &std::path::Path, text_only: bool) {
    let p = "language_model.model";
    let quantization = json!({"group_size": 64, "bits": 4, "mode": "affine"});
    let text = json!({
        "model_type": "qwen3_5_text", "attn_output_gate": true,
        "num_hidden_layers": 2, "max_position_embeddings": 4096, "hidden_size": 64,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
        "intermediate_size": 128, "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 64, "linear_num_key_heads": 2,
        "linear_num_value_heads": 6, "linear_value_head_dim": 64,
        "layer_types": ["linear_attention", "full_attention"],
        "full_attention_interval": 4, "rms_norm_eps": 1e-6, "vocab_size": 8,
        "bos_token_id": 2, "eos_token_id": 2, "pad_token_id": null,
        "partial_rotary_factor": 0.5, "hidden_act": "silu",
        "output_gate_type": "swish", "tie_word_embeddings": false,
        "rope_parameters": {"mrope_interleaved": true, "mrope_section": [4, 2, 2],
            "rope_theta": 1e6, "rope_type": "default", "partial_rotary_factor": 0.5},
    });
    let config = if text_only {
        let mut config = text;
        config["architectures"] = json!(["Qwen3_5ForCausalLM"]);
        config["quantization"] = quantization.clone();
        config["quantization_config"] = quantization;
        config
    } else {
        json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "eos_token_id": [2],
            "image_token_id": 3,
            "tie_word_embeddings": false,
            "quantization": quantization,
            "quantization_config": quantization,
            "text_config": text,
        })
    };
    let mut t: Vec<Stored> = vec![];
    for stem in [format!("{p}.embed_tokens"), "language_model.lm_head".into()] {
        t.extend(mlx_linear(&stem, 4, &[8], 64));
    }
    t.push(bf16_tensor(&format!("{p}.norm.weight"), &[64], |_| 0x3fc0));
    for l in 0..2 {
        let m = format!("{p}.layers.{l}");
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            t.push(bf16_tensor(&format!("{m}.{norm}.weight"), &[64], |_| {
                0x3f80
            }));
        }
        t.extend(seeded_linear(&format!("{m}.mlp.gate_proj"), 128, 64, 0x00));
        t.extend(seeded_linear(&format!("{m}.mlp.up_proj"), 128, 64, 0x5a));
        t.extend(seeded_linear(&format!("{m}.mlp.down_proj"), 64, 128, 0xa5));
    }
    let m = format!("{p}.layers.0.linear_attn");
    for (stem, rows, k) in [
        ("in_proj_qkv", 640, 64),
        ("in_proj_z", 384, 64),
        ("in_proj_a", 6, 64),
        ("in_proj_b", 6, 64),
        ("out_proj", 64, 384),
    ] {
        t.extend(mlx_linear(&format!("{m}.{stem}"), 4, &[rows], k));
    }
    t.push(bf16_tensor(
        &format!("{m}.conv1d.weight"),
        &[640, 4, 1],
        |_| 0x3f80,
    ));
    t.push(bf16_tensor(&format!("{m}.A_log"), &[6], |_| 0x3f80));
    t.push(bf16_tensor(&format!("{m}.dt_bias"), &[6], |_| 0x3f80));
    t.push(bf16_tensor(&format!("{m}.norm.weight"), &[64], |_| 0x3f80));
    let m = format!("{p}.layers.1.self_attn");
    for (stem, rows, k) in [
        ("q_proj", 128, 64),
        ("k_proj", 32, 64),
        ("v_proj", 32, 64),
        ("o_proj", 64, 64),
    ] {
        t.extend(mlx_linear(&format!("{m}.{stem}"), 4, &[rows], k));
    }
    for norm in ["q_norm", "k_norm"] {
        t.push(bf16_tensor(&format!("{m}.{norm}.weight"), &[32], |_| {
            0x3e80
        }));
    }
    if !text_only {
        t.push(bf16_tensor(
            "vision_tower.patch_embed.proj.weight",
            &[2, 3],
            |_| 0x3f80,
        ));
    }
    t.push(bf16_tensor("mtp.fc.weight", &[64, 128], |_| 0x3f80));
    write_folder(dir, &t, &config, &tokenizer("qwen35"));
}

/// A folder of the dense family written as the multimodal class and as the text-only class opens
/// as a `qwen35` file: the engine's own header reader takes it as the dense variant with every
/// layer's block dense at the config's width, the three projections of a layer read the bytes
/// stored under their own names (the gate, the up and the down are three different tensors
/// here), a next-token layer's tensor is listed as skipped and refuses the draft by name, and the
/// vision tower is left out. Both classes make the same header keys and the same tensors.
#[test]
fn a_small_dense_folder_opens_as_a_qwen35_file() {
    use model::arch::qwen35moe::hparams::{FfnKind, Hparams, Kind, Variant};
    use model::part::{Source, Take};

    let multimodal = temp_dir("qwen35-open");
    write_small_dense_folder(&multimodal, false);
    let opened = folder::open(&multimodal).unwrap();

    let hp = Hparams::read(&opened.header).unwrap();
    assert_eq!(hp.variant, Variant::Qwen35);
    assert_eq!(hp.kinds, [Kind::DeltaRule, Kind::Attention]);
    assert_eq!(hp.ffns, [FfnKind::Dense; 2]);
    assert_eq!(hp.ff, Some(128));
    assert_eq!((hp.n_expert, hp.interval), (0, 2));
    let ty = |name: &str| opened.header.find(name).unwrap().1.ty;
    for name in [
        "blk.0.ffn_gate.weight",
        "blk.1.ffn_up.weight",
        "blk.1.ffn_down.weight",
    ] {
        assert_eq!(ty(name), GgmlType::MLX_A4, "{name}");
    }
    assert_eq!(ty("blk.0.post_attention_norm.weight"), GgmlType::F32);
    assert!(
        opened
            .header
            .find("vision_tower.patch_embed.proj.weight")
            .is_none()
    );
    assert!(opened.header.find("blk.0.ffn_gate_inp.weight").is_none());

    // Each projection's first group of 32 bytes is the stored row's first words, xor its seed.
    let planned = opened
        .source
        .plan(|_, _, _| Ok::<_, String>(Take::Whole))
        .unwrap();
    let read = |name: &str| -> Vec<u8> {
        let p = planned.iter().find(|p| p.info.name == name).unwrap();
        let part = opened.source.read(p).unwrap();
        part.spans().concat()
    };
    let words = |seed: u8| -> Vec<u8> {
        (0..8u32)
            .flat_map(|i| i.wrapping_mul(0x9e37_79b9).to_le_bytes())
            .map(|b| b ^ seed)
            .collect()
    };
    for (name, rows, groups, seed) in [
        ("ffn_gate", 128, 1, 0x00),
        ("ffn_up", 128, 1, 0x5a),
        ("ffn_down", 64, 2, 0xa5),
    ] {
        let bytes = read(&format!("blk.1.{name}.weight"));
        assert_eq!(bytes.len(), rows * groups * 36, "{name}");
        assert_eq!(bytes[..32], words(seed)[..], "{name}");
    }

    assert_eq!(
        opened.skipped.get("mtp.fc.weight").map(String::as_str),
        Some(MTP)
    );
    assert_eq!(
        opened
            .skipped
            .get("vision_tower.patch_embed.proj.weight")
            .map(String::as_str),
        Some(VISION)
    );
    let refused = opened.draft.as_ref().err().expect("the draft is refused");
    let refused = refused.to_string();
    assert!(
        refused.contains("mtp.fc.weight") && refused.contains("does not read"),
        "{refused}"
    );

    let text_only = temp_dir("qwen35-text-open");
    write_small_dense_folder(&text_only, true);
    let from_text = folder::open(&text_only).unwrap();
    assert_eq!(from_text.header.keys(), opened.header.keys());
    let names = |o: &folder::Opened| {
        o.source
            .plan(|_, _, _| Ok::<_, String>(Take::Whole))
            .unwrap()
            .iter()
            .map(|p| p.info.name.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(names(&from_text), names(&opened));
    assert!(from_text.draft.is_err());
    std::fs::remove_dir_all(&multimodal).unwrap();
    std::fs::remove_dir_all(&text_only).unwrap();
}
