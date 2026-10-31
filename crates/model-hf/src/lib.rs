//! What the CPU model description does with the Hugging Face hub and its checkpoints, kept out of
//! `model` so that crate, and every gate that links it, does not carry the hf crate: [`remote`]
//! reads a fixture's source headers from a repo by HTTP Range (`fixture fetch`, the `fixture` bin of
//! this crate), [`arch::glm5next`] maps GLM-5.3-Flash's checkpoint onto the glm5next program's names
//! and layouts, [`arch::qwen35moe`] maps Qwen3.8-Flash-Next's onto qwen4exp's and Qwen3.6's MLX
//! folder onto qwen35moe's, [`folder`] is the one reader of a checkpoint folder that every such
//! map plugs into (its plan, header, located split and tokenizer check), the `prepare` bin writes
//! a GGUF split from such a checkpoint folder, [`part`] reads a load's tensors, whole or a
//! rank's part, from a checkpoint folder as `model::part` reads them from a GGUF file, and
//! [`model_file`] is the one owner of the choice between a GGUF split and a checkpoint folder:
//! a path opens as either, and its header, tokenizer, name and weights are asked of one value.

pub mod arch;
pub mod folder;
pub mod model_file;
pub mod part;
pub mod remote;

pub use model_file::{ModelFile, Weights};

#[cfg(test)]
mod test_support {
    use std::collections::BTreeMap;
    use std::io::Read;
    use std::path::Path;

    use gguf::GgmlType;
    use safetensors::Dtype;
    use serde_json::{Value as Json, json};

    /// The committed fixture `name` of the hf crate (`tests/fixtures`), gunzipped when it ends in `.gz`.
    pub(crate) fn fixture(name: &str) -> Vec<u8> {
        let path = format!("{}/../hf/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
        let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        if !name.ends_with(".gz") {
            return raw;
        }
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(raw.as_slice())
            .read_to_end(&mut out)
            .unwrap();
        out
    }

    /// [`fixture`] as JSON.
    pub(crate) fn json(name: &str) -> Json {
        serde_json::from_slice(&fixture(name)).unwrap()
    }

    /// A header key's value as the committed key files write it: `[type, value]`, an array
    /// `["ARR", [element type, values]]`.
    pub(crate) fn key_json(v: &gguf::Value) -> Json {
        use gguf::Value;
        match v {
            Value::U32(x) => json!(["U32", x]),
            Value::F32(x) => json!(["F32", f64::from(*x)]),
            Value::String(s) => json!(["STR", s]),
            Value::Bool(b) => json!(["BOOL", b]),
            Value::Array(a) => {
                let (tag, items): (&str, Vec<Json>) = match a.first() {
                    Some(Value::I32(_)) => (
                        "I32",
                        a.iter()
                            .map(|x| match x {
                                Value::I32(i) => json!(i),
                                _ => unreachable!(),
                            })
                            .collect(),
                    ),
                    Some(Value::Bool(_)) => (
                        "BOOL",
                        a.iter()
                            .map(|x| match x {
                                Value::Bool(b) => json!(b),
                                _ => unreachable!(),
                            })
                            .collect(),
                    ),
                    _ => (
                        "U64",
                        a.iter()
                            .map(|x| match x {
                                Value::U64(i) => json!(i),
                                _ => unreachable!(),
                            })
                            .collect(),
                    ),
                };
                json!(["ARR", [tag, items]])
            }
            other => panic!("{other:?}"),
        }
    }

    /// A folder's tensors: name, dtype and shape.
    pub(crate) type Entries = Vec<(String, Dtype, Vec<usize>)>;

    /// The entries of a committed tensor-header file.
    pub(crate) fn entries(header: &Json) -> Entries {
        header
            .as_object()
            .unwrap()
            .iter()
            .map(|(n, e)| {
                let shape = e["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_u64().unwrap() as usize)
                    .collect();
                (
                    n.clone(),
                    Dtype::of(e["dtype"].as_str().unwrap()).unwrap(),
                    shape,
                )
            })
            .collect()
    }

    /// The GGUF fixture's tensors, `[name, ggml dims, type id]`, by name.
    pub(crate) fn gguf_tensors(gguf: &Json) -> BTreeMap<String, (Vec<u64>, GgmlType)> {
        gguf["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| {
                let dims = t[1].as_array().unwrap().iter().map(|d| d.as_u64().unwrap());
                (
                    t[0].as_str().unwrap().to_string(),
                    (
                        dims.collect(),
                        GgmlType::from_u32(t[2].as_u64().unwrap() as u32),
                    ),
                )
            })
            .collect()
    }

    /// One stored tensor: name, dtype, shape, bytes.
    pub(crate) type Stored = (String, &'static str, Vec<usize>, Vec<u8>);

    /// An MLX linear `stem` of `bits`-bit codes over the leading dims `lead` and `k` values a row:
    /// its weight (a counting pattern of words), its scales (0.5) and its biases (−0.25), the pair
    /// in `pair`'s dtype.
    pub(crate) fn mlx_linear_pair(
        stem: &str,
        bits: usize,
        lead: &[usize],
        k: usize,
        pair: Dtype,
    ) -> Vec<Stored> {
        let (ty, scale, bias) = match pair {
            Dtype::Bf16 => ("BF16", 0x3f00u16, 0xbe80),
            Dtype::F16 => ("F16", 0x3800, 0xb400),
            other => panic!("{other:?}: a pair this builder does not write"),
        };
        let shape = |last: usize| [lead, &[last]].concat();
        let n = shape(k * bits / 32).iter().product::<usize>();
        let words = (0..n as u32).flat_map(|i| i.wrapping_mul(0x9e37_79b9).to_le_bytes());
        let groups = shape(k / 64).iter().product::<usize>();
        let pair = |bits16: u16| (0..groups).flat_map(|_| bits16.to_le_bytes()).collect();
        vec![
            (
                format!("{stem}.weight"),
                "U32",
                shape(k * bits / 32),
                words.collect(),
            ),
            (format!("{stem}.scales"), ty, shape(k / 64), pair(scale)),
            (format!("{stem}.biases"), ty, shape(k / 64), pair(bias)),
        ]
    }

    /// [`mlx_linear_pair`] with a BF16 pair, the real 4-bit folders' dtype.
    pub(crate) fn mlx_linear(stem: &str, bits: usize, lead: &[usize], k: usize) -> Vec<Stored> {
        mlx_linear_pair(stem, bits, lead, k, Dtype::Bf16)
    }

    /// A BF16 tensor of `shape` whose element `i` is `value(i)`.
    pub(crate) fn bf16_tensor(name: &str, shape: &[usize], value: impl Fn(usize) -> u16) -> Stored {
        let n = shape.iter().product::<usize>();
        let bytes = (0..n).flat_map(|i| value(i).to_le_bytes()).collect();
        (name.to_string(), "BF16", shape.to_vec(), bytes)
    }

    /// The byte-level BPE tokenizer `tokenizer.json` of eight ids, its one special token `</s>`
    /// id 2, with the pre-tokenizer row `pre` of `hf::vocab::PRE`.
    pub(crate) fn tokenizer(pre: &str) -> Json {
        let (_, row, ignore_merges) = hf::vocab::PRE
            .iter()
            .find(|(name, _, _)| *name == pre)
            .unwrap_or_else(|| panic!("no pre-tokenizer row {pre}"));
        let pre_tokenizer: Json = serde_json::from_str(row).unwrap();
        json!({"normalizer":{"type":"NFC"},"pre_tokenizer":pre_tokenizer,
            "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
            "post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
            "model":{"type":"BPE","vocab":{"A":0,"a":1,"Aa":7},"merges":[["A","a"]],
                "ignore_merges":ignore_merges},
            "added_tokens":[{"id":2,"content":"</s>","special":true}]})
    }

    /// The folder `dir` of `tensors` in two shards (the first half, then the rest), their index,
    /// `config`, `tokenizer` and the two tokenizer files of no content.
    pub(crate) fn write_folder(dir: &Path, tensors: &[Stored], config: &Json, tokenizer: &Json) {
        let write_shard = |file: &str, part: &[Stored]| {
            let list: Vec<(&str, &str, &[usize], &[u8])> = part
                .iter()
                .map(|(n, d, s, b)| (n.as_str(), *d, s.as_slice(), b.as_slice()))
                .collect();
            std::fs::write(dir.join(file), safetensors::write(&list)).unwrap();
        };
        let (first, second) = tensors.split_at(tensors.len() / 2);
        write_shard("a.safetensors", first);
        write_shard("b.safetensors", second);
        let weight_map: BTreeMap<&str, &str> = first
            .iter()
            .map(|t| (t.0.as_str(), "a.safetensors"))
            .chain(second.iter().map(|t| (t.0.as_str(), "b.safetensors")))
            .collect();
        std::fs::write(
            dir.join(safetensors::Folder::INDEX),
            json!({ "weight_map": weight_map }).to_string(),
        )
        .unwrap();
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
        for file in ["tokenizer_config.json", "generation_config.json"] {
            std::fs::write(dir.join(file), "{}").unwrap();
        }
    }

    /// A folder `dir` of two layers of a Qwen3-MoE as small as the groups allow, in MLX's names
    /// with the real file's `quantization` block (no `mode`): the 4-bit linears and stacked
    /// experts, the 8-bit routers, the BF16 vectors, and a tokenizer of eight ids.
    pub(crate) fn small_qwen3_folder(dir: &Path) {
        let mut quantization = json!({"group_size": 64, "bits": 4});
        for l in 0..2 {
            quantization.as_object_mut().unwrap().insert(
                format!("model.layers.{l}.mlp.gate"),
                json!({"group_size": 64, "bits": 8}),
            );
        }
        let config = json!({
            "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
            "attention_bias": false, "decoder_sparse_step": 1, "mlp_only_layers": [],
            "norm_topk_prob": true, "rope_scaling": null, "sliding_window": null,
            "tie_word_embeddings": false, "hidden_act": "silu",
            "bos_token_id": 2, "eos_token_id": 2, "vocab_size": 8,
            "num_hidden_layers": 2, "max_position_embeddings": 4096, "hidden_size": 64,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
            "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 64,
            "intermediate_size": 128, "rms_norm_eps": 1e-6, "rope_theta": 1e6,
            "quantization": quantization, "quantization_config": quantization,
        });
        let mut t: Vec<Stored> = vec![];
        for stem in ["model.embed_tokens", "lm_head"] {
            t.extend(mlx_linear(stem, 4, &[8], 64));
        }
        t.push(bf16_tensor("model.norm.weight", &[64], |_| 0x3fc0));
        for l in 0..2 {
            let m = format!("model.layers.{l}");
            for (norm, bf16) in [
                ("input_layernorm", 0x3fc0),
                ("post_attention_layernorm", 0x3f80),
            ] {
                t.push(bf16_tensor(&format!("{m}.{norm}.weight"), &[64], |_| bf16));
            }
            t.extend(mlx_linear(&format!("{m}.mlp.gate"), 8, &[4], 64));
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                t.extend(mlx_linear(
                    &format!("{m}.mlp.switch_mlp.{proj}"),
                    4,
                    &[4, 64],
                    64,
                ));
            }
            for (stem, rows) in [
                ("q_proj", 64),
                ("k_proj", 32),
                ("v_proj", 32),
                ("o_proj", 64),
            ] {
                t.extend(mlx_linear(&format!("{m}.self_attn.{stem}"), 4, &[rows], 64));
            }
            for norm in ["q_norm", "k_norm"] {
                t.push(bf16_tensor(
                    &format!("{m}.self_attn.{norm}.weight"),
                    &[32],
                    |_| 0x3e80,
                ));
            }
        }
        write_folder(dir, &t, &config, &tokenizer("qwen2"));
    }

    /// A fresh folder under the system's temporary one.
    pub(crate) fn temp_dir(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("model-hf-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        dir
    }
}
