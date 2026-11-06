//! A toy Kimi K3 or K3.1 checkpoint folder on disk: the real configs' fields at toy widths, the
//! tensors the map's own table says the folder stores ([`FolderMap::stored`]) with
//! deterministic values, one shard a layer, the head's and the vision tower's, and the tiktoken
//! files of `hf/tests/fixtures/tiktoken-toy`.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use model_hf::arch::kimi_k3::Checkpoint;
use model_hf::folder::FolderMap;
use safetensors::Dtype;
use serde_json::{Value, json};

pub const E: usize = 512;
pub const HEADS: usize = 2;
pub const LAYERS: usize = 4;
pub const N_EXP: usize = 4;
/// The toy tiktoken vocabulary's size and the three special ids its config names.
pub const VOCAB: usize = 525;
/// The 0-based latent layer: `full_attn_layers` and `dsa_layers` list it 1-based.
pub const LATENT_LAYER: usize = 2;

fn fixtures() -> String {
    format!("{}/../hf/tests/fixtures", env!("CARGO_MANIFEST_DIR"))
}

fn gunzip(path: &str) -> Value {
    let gz = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut text = Vec::new();
    flate2::read::GzDecoder::new(gz.as_slice())
        .read_to_end(&mut text)
        .unwrap();
    serde_json::from_slice(&text).unwrap()
}

/// The real K3 or K3.1 config at the toy widths.
pub fn config(k31: bool) -> Value {
    let name = if k31 { "Kimi-K3.1-Pro" } else { "Kimi-K3" };
    let mut c = gunzip(&format!(
        "{}/quant-configs/moonshotai__{name}__config.json.gz",
        fixtures()
    ));
    let t = &mut c["text_config"];
    for (key, v) in [
        ("hidden_size", E),
        ("intermediate_size", 768),
        ("num_hidden_layers", LAYERS),
        ("num_attention_heads", HEADS),
        ("num_key_value_heads", HEADS),
        ("q_lora_rank", 256),
        ("kv_lora_rank", 512),
        ("qk_nope_head_dim", 128),
        ("qk_rope_head_dim", 64),
        ("v_head_dim", 128),
        ("first_k_dense_replace", 1),
        ("moe_intermediate_size", 256),
        ("num_experts", N_EXP),
        ("num_experts_per_token", 2),
        ("num_shared_experts", 2),
        ("routed_expert_hidden_size", 512),
        ("attn_res_block_size", 2),
        ("vocab_size", VOCAB),
    ] {
        t[key] = json!(v);
    }
    t["linear_attn_config"]["num_heads"] = json!(HEADS);
    t["linear_attn_config"]["head_dim"] = json!(128);
    t["linear_attn_config"]["full_attn_layers"] = json!([LATENT_LAYER + 1]);
    t["linear_attn_config"]["kda_layers"] = json!([1, 2, 4]);
    if k31 {
        t["dsa_config"]["dsa_layers"] = json!([LATENT_LAYER + 1]);
        t["dsa_config"]["index_num_heads"] = json!(4);
        t["dsa_config"]["index_head_dim"] = json!(128);
        t["dsa_config"]["index_topk"] = json!(8);
    }
    for (key, id) in [
        ("bos_token_id", 267),
        ("eos_token_id", 269),
        ("pad_token_id", 522),
    ] {
        c[key] = json!(id);
        c["text_config"][key] = json!(id);
    }
    c
}

/// A deterministic finite value from the tensor's name and its place in it.
fn value(name: &str, i: usize) -> f32 {
    let h = name
        .bytes()
        .fold(17usize, |h, b| (h * 31 + usize::from(b)) % 1009);
    ((h + i * 7) % 31) as f32 / 16.0 - 0.9
}

/// The stored bytes of tensor `name`: BF16 and F32 values finite, `A_log` positive, a pack's
/// bytes any, a pack's scale around 2^0.
fn bytes(name: &str, dtype: Dtype, shape: &[usize]) -> Vec<u8> {
    let n: usize = shape.iter().product();
    match dtype {
        Dtype::Bf16 => (0..n)
            .flat_map(|i| ((value(name, i).to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
        Dtype::F32 if name.ends_with("A_log") => (0..n)
            .flat_map(|i| (value(name, i).abs() + 0.1).to_le_bytes())
            .collect(),
        Dtype::F32 => (0..n).flat_map(|i| value(name, i).to_le_bytes()).collect(),
        Dtype::U8 if name.ends_with("weight_scale") => {
            (0..n).map(|i| 124 + (i + name.len()) as u8 % 6).collect()
        }
        Dtype::U8 => (0..n)
            .map(|i| (value(name, i).to_bits() >> 11) as u8 ^ i as u8)
            .collect(),
        other => panic!("{other:?}"),
    }
}

/// The shard file of `name`: one a layer (the first after the head's), the head's, the vision
/// tower's last.
fn shard_of(name: &str) -> String {
    let n = match name.strip_prefix("language_model.model.layers.") {
        Some(rest) => rest.split('.').next().unwrap().parse::<usize>().unwrap() + 1,
        None if name.starts_with("language_model.") => LAYERS + 1,
        None => LAYERS + 2,
    };
    format!("model-{n:05}-of-{:05}.safetensors", LAYERS + 2)
}

/// One input shard's tensors: name, dtype, shape and bytes.
pub type Rows = Vec<(String, Dtype, Vec<usize>, Vec<u8>)>;
/// The tensors of a toy folder, in a map of shard file to its tensors.
pub type Shards = BTreeMap<String, Rows>;

/// What the folder of `config` stores, by shard: the map's table and the vision tower's two
/// tensors, which a K3 or K3.1 folder holds and the map skips.
pub fn shards(config: &Value) -> Shards {
    let map = Checkpoint::parse(config).unwrap();
    let mut entries = map.stored().unwrap().tensors;
    entries.push((
        "vision_tower.patch_embed.proj.weight".into(),
        Dtype::Bf16,
        vec![8, 8],
    ));
    entries.push(("mm_projector.proj.0.weight".into(), Dtype::Bf16, vec![8, 8]));
    let mut out = Shards::new();
    for (name, dtype, shape) in entries {
        let data = bytes(&name, dtype, &shape);
        out.entry(shard_of(&name))
            .or_default()
            .push((name, dtype, shape, data));
    }
    out
}

/// The index json of `shards`.
pub fn index(shards: &Shards) -> String {
    let weight_map: BTreeMap<&str, &str> = shards
        .iter()
        .flat_map(|(file, tensors)| tensors.iter().map(move |t| (t.0.as_str(), file.as_str())))
        .collect();
    json!({"metadata": {}, "weight_map": weight_map}).to_string()
}

/// The safetensors file of one shard's tensors.
pub fn shard_bytes(tensors: &[(String, Dtype, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
    let views: Vec<(&str, &str, &[usize], &[u8])> = tensors
        .iter()
        .map(|(n, d, s, b)| (n.as_str(), d.name(), s.as_slice(), b.as_slice()))
        .collect();
    safetensors::write(&views)
}

/// The config, the tokenizer files and the index of a toy folder at `dir`, and the shards in
/// `which` (all of them when `None`).
pub fn write_folder(dir: &Path, k31: bool, which: Option<&[&str]>) -> Shards {
    std::fs::create_dir_all(dir).unwrap();
    let config = config(k31);
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    for f in [
        "generation_config.json",
        "tokenizer_config.json",
        "tiktoken.model",
        "tokenization_kimi.py",
    ] {
        std::fs::copy(format!("{}/tiktoken-toy/{f}", fixtures()), dir.join(f)).unwrap();
    }
    let shards = shards(&config);
    std::fs::write(dir.join("model.safetensors.index.json"), index(&shards)).unwrap();
    for (file, tensors) in &shards {
        if which.is_none_or(|w| w.contains(&file.as_str())) {
            std::fs::write(dir.join(file), shard_bytes(tensors)).unwrap();
        }
    }
    shards
}
