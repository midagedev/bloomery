//! Qwen3-MoE's Hugging Face checkpoint (`qwen3_moe`: Qwen3-30B-A3B and its siblings) in the
//! qwen3moe names, layouts and keys the program reads, as the folder MLX's convert writes it
//! stores it: every layer attends (grouped query, per-head RMS norms on the query and the key)
//! and routes over experts, with no shared expert, no dense layer and no gated delta rule, so
//! each tensor maps to one tensor of the file with no move. The routed experts of a projection
//! arrive stacked, one `switch_mlp` tensor; the norm gammas are the program's own. The
//! trunk's names, keys and checks are [`super::trunk`]'s; what is here is the architecture's own:
//! its config, the sizes each tensor must have, and that every tensor of every layer is there.

use std::collections::BTreeMap;
use std::convert::Infallible;

use gguf::Value;
use hf::modelopt::{Plain, Use};
use hf::schema::{Fields, Rule::*, Type};
use model::arch::names as common;
use model::arch::qwen3moe::names::LAYER;
use serde_json::Value as Json;

use super::section::Section;
use super::trunk;
use crate::folder::{Error, FolderMap, Mapped, Mapping, Origin, VocabSpec, refuse};

const ARCHITECTURE: &str = "qwen3moe";

/// Every layer routes over experts.
const FFN: trunk::Ffn = trunk::Ffn::Routed;

/// `config.json`'s keys beside [`trunk::KEYS`] and the routed block's: the checkpoint's class and
/// type, MLX's `quantization` block (which `hf::quant_config` reads, with its verbatim copy), and
/// the values the program computes with one value and GGUF has no key for.
const ROOT: &Fields = &[
    ("architectures", Need(r#"[["Qwen3MoeForCausalLM"]]"#)),
    ("model_type", Need(r#"["qwen3_moe"]"#)),
    ("quantization", Read),
    ("quantization_config", Read),
    // The dense feed-forward width: the header keeps it as `feed_forward_length`, and no layer is
    // dense, so no layer reads it.
    ("intermediate_size", Read),
    ("rope_theta", Read),
    // Every layer routes: HF's default is every layer, and no layer is listed dense.
    ("decoder_sparse_step", Only("[1]")),
    ("mlp_only_layers", Only("[[]]")),
    // The program renormalizes the chosen experts' weights, which HF's default does not.
    ("norm_topk_prob", Need("[true]")),
    // No bias on the query, key or value projections, the per-head norms on the query and the key
    // are applied on every layer, and there is no shared expert: the values the program computes
    // with, which a Coder config spells out as keys.
    ("qkv_bias", Only("[false]")),
    ("use_qk_norm", Only("[true]")),
    ("shared_expert_intermediate_size", Only("[0]")),
    // Attention is full and its rope is unscaled.
    ("rope_scaling", Only("[null]")),
    ("sliding_window", Only("[null]")),
    ("use_sliding_window", Only("[false]")),
    (
        "max_window_layers",
        Ignore(
            Type::Number,
            "the layers a sliding window applies to, and use_sliding_window is false",
        ),
    ),
    (
        "torch_dtype",
        Ignore(
            Type::String,
            "the dtype HF loads in; each tensor keeps its stored dtype and format",
        ),
    ),
    (
        "transformers_version",
        Ignore(Type::String, "the version that wrote the file"),
    ),
];

/// The config values the map reads, checked, and the header keys they make.
#[derive(Debug)]
pub struct Checkpoint {
    layers: usize,
    experts: usize,
    hidden: u64,
    query_rows: u64,
    key_rows: u64,
    head_dim: u64,
    expert_ff: u64,
    vocab: VocabSpec,
    /// The header's model keys by suffix, under the architecture's name.
    keys: Vec<(String, Value)>,
}

impl FolderMap for Checkpoint {
    /// A tensor needs no move: the stored layout is the program's.
    type Step = Infallible;

    const MODEL_TYPES: &'static [&'static str] = &["qwen3_moe"];

    fn parse(config: &Json) -> Result<Self, Error> {
        let root = Section::<Error>::new(config, "config.json")?;
        root.check(&[ROOT, trunk::KEYS, FFN.keys()])?;
        let mut counts = BTreeMap::new();
        for &field in trunk::COUNTS.iter().chain(FFN.counts()) {
            counts.insert(field, root.count(field)?);
        }
        let n = |key: &str| counts[key];
        trunk::check_counts(&root, FFN, n)?;
        let wide = |key: &str| u64::from(counts[key]);
        let rows = |heads: &str| {
            wide(heads)
                .checked_mul(wide("head_dim"))
                .ok_or_else(|| root.bad(heads, "times head_dim overflows u64"))
        };
        let mut keys = trunk::metadata(
            n,
            FFN,
            root.number("rope_theta")?,
            root.number("rms_norm_eps")?,
        );
        keys.push((
            "feed_forward_length".to_string(),
            Value::U32(root.count("intermediate_size")?),
        ));
        let vocab = hf::vocab::Config::parse(config)?.with_pre("qwen2")?;
        Ok(Self {
            layers: n("num_hidden_layers") as usize,
            experts: n("num_experts") as usize,
            hidden: wide("hidden_size"),
            query_rows: rows("num_attention_heads")?,
            key_rows: rows("num_key_value_heads")?,
            head_dim: wide("head_dim"),
            expert_ff: wide("moe_intermediate_size"),
            vocab,
            keys,
        })
    }

    fn module_class(&self, module: &str) -> Option<hf::quant_config::ModuleClass> {
        super::module_class(module)
    }

    fn map<'a>(&self, tensors: &[(&'a str, &'a [usize])]) -> Result<Mapping<Infallible>, Error> {
        let mut out: BTreeMap<String, Mapped<Infallible>> = BTreeMap::new();
        for &(name, shape) in tensors {
            if !trunk::sized(shape) {
                return Err(refuse(
                    name,
                    format!("shape {shape:?} is empty, zero, or overflows usize"),
                ));
            }
            let (to, plain) = self.tensor(name, shape)?;
            let mapped = Mapped {
                sources: vec![name.to_string()],
                role: Use::Matrix(plain),
                steps: vec![],
                dims: shape.iter().rev().map(|&d| d as u64).collect(),
                origin: Origin::Made,
            };
            if let Some(prev) = out.insert(to, mapped) {
                return Err(refuse(
                    &prev.sources[0],
                    "maps to a qwen3moe name another tensor has",
                ));
            }
        }
        self.complete(&out)?;
        Ok(Mapping {
            tensors: out,
            skipped: BTreeMap::new(),
        })
    }

    fn mover(&self, _name: &str, steps: &[Infallible]) -> Option<crate::part::Step> {
        steps.first().map(|&never| match never {})
    }

    fn architecture(&self) -> &'static str {
        ARCHITECTURE
    }

    fn header(
        &self,
        _planned: &[(&str, &[u64])],
        _folder: &safetensors::Folder,
    ) -> Result<Vec<(String, Value)>, Error> {
        Ok(trunk::header(ARCHITECTURE, &self.keys))
    }

    fn vocab(&self) -> &VocabSpec {
        &self.vocab
    }
}

impl Checkpoint {
    /// The program's tensor of the checkpoint tensor `name` and the role it is read in; a tensor
    /// no Qwen3-MoE folder of MLX's has is refused with why.
    fn tensor(&self, name: &str, shape: &[usize]) -> Result<(String, Plain), Error> {
        if name == "lm_head.weight" {
            return Ok((common::output(), Plain::Q8_0));
        }
        let short = name
            .strip_prefix("model.")
            .ok_or_else(|| refuse(name, "not a Qwen3-MoE tensor"))?;
        if let Some(global) = trunk::global(short) {
            return Ok(global);
        }
        let (l, rest) = short
            .strip_prefix("layers.")
            .and_then(|r| r.split_once('.'))
            .and_then(|(l, rest)| Some((l.parse::<usize>().ok()?, rest)))
            .ok_or_else(|| refuse(name, "not a Qwen3-MoE tensor"))?;
        if l >= self.layers {
            return Err(refuse(name, format!("layer {l} of {}", self.layers)));
        }
        if rest.starts_with("mlp.experts.") {
            return Err(refuse(
                name,
                "an expert of its own; this map reads MLX's stacked `switch_mlp`",
            ));
        }
        if let Some(found) = trunk::attention(l, rest)
            .or_else(|| trunk::block_norms(l, rest, common::ffn_norm))
            .or_else(|| trunk::router(l, rest))
        {
            return Ok(found);
        }
        match trunk::stacked_experts(l, rest, shape, self.experts) {
            Some(stack) => stack.map_err(|why| refuse(name, why)),
            None => Err(refuse(name, "not a Qwen3-MoE tensor")),
        }
    }

    /// The ggml dims of the file's tensor `stem` (the name past `blk.N.`, or a model-level
    /// name) that the config's geometry gives.
    fn dims(&self, stem: &str) -> Vec<u64> {
        let (hidden, experts) = (self.hidden, self.experts as u64);
        match stem {
            "token_embd.weight" | "output.weight" => vec![hidden, u64::from(self.vocab.size())],
            "output_norm.weight" | "attn_norm.weight" | "ffn_norm.weight" => vec![hidden],
            "attn_q.weight" => vec![hidden, self.query_rows],
            "attn_k.weight" | "attn_v.weight" => vec![hidden, self.key_rows],
            "attn_output.weight" => vec![self.query_rows, hidden],
            "attn_q_norm.weight" | "attn_k_norm.weight" => vec![self.head_dim],
            "ffn_gate_inp.weight" => vec![hidden, experts],
            "ffn_gate_exps.weight" | "ffn_up_exps.weight" => {
                vec![hidden, self.expert_ff, experts]
            }
            "ffn_down_exps.weight" => vec![self.expert_ff, hidden, experts],
            other => unreachable!("{other} is a tensor of no Qwen3-MoE file"),
        }
    }

    /// Every tensor the file holds is mapped, at the dims the config gives: the three model-level
    /// tensors and the layer's [`LAYER`] stems on each layer.
    fn complete(&self, mapped: &BTreeMap<String, Mapped<Infallible>>) -> Result<(), Error> {
        let globals = ["token_embd.weight", "output_norm.weight", "output.weight"]
            .into_iter()
            .map(|stem| (stem.to_string(), stem));
        let layers = (0..self.layers).flat_map(|l| {
            LAYER
                .into_iter()
                .map(move |stem| (common::blk(l, stem), stem))
        });
        for (name, stem) in globals.chain(layers) {
            let m = mapped
                .get(&name)
                .ok_or_else(|| refuse(&name, "is missing from the folder"))?;
            let want = self.dims(stem);
            if m.dims != want {
                return Err(refuse(
                    &m.sources[0],
                    format!("holds dims {:?} where the config gives {want:?}", m.dims),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use gguf::GgmlType;
    use hf::quant_config::Config;
    use safetensors::Dtype;
    use serde_json::json;

    use super::*;
    use crate::folder;
    use crate::test_support::{
        Entries, Stored, bf16_tensor, entries, gguf_tensors, json, key_json, mlx_linear_pair,
        temp_dir, tokenizer, write_folder,
    };

    type Planned = folder::Planned<Infallible>;

    /// The MLX type and role a GGUF fixture tensor of type `ty` becomes at the scale pair `pair`:
    /// the 4-bit linears are MLX_A4 (MLX_A4H at an F16 pair), read by Q8_0 roles; the router is
    /// the 8-bit MLX_A8 (MLX_A8H at an F16 pair), read by the BF16 role the map gives a router;
    /// the BF16 vectors are F32.
    fn mlx_type_of(name: &str, ty: GgmlType, pair: Dtype) -> (GgmlType, Use) {
        let (a4, a8) = match pair {
            Dtype::Bf16 => (GgmlType::MLX_A4, GgmlType::MLX_A8),
            Dtype::F16 => (GgmlType::MLX_A4H, GgmlType::MLX_A8H),
            other => panic!("{name}: a pair the fixture is not written at: {other:?}"),
        };
        match ty {
            GgmlType::Q4_K | GgmlType::Q6_K => (a4, Use::Matrix(Plain::Q8_0)),
            GgmlType::F32 if name.contains("ffn_gate_inp") => (a8, Use::Matrix(Plain::Bf16)),
            GgmlType::F32 => (GgmlType::F32, Use::Matrix(Plain::F32)),
            other => panic!("{name}: {other}"),
        }
    }

    /// Every stored byte of the folder is one a planned tensor holds: an MLX tensor holds its
    /// weight, scales and biases whole, and a BF16 vector is planned as F32 at twice the bytes.
    fn assert_every_stored_byte_is_planned(entries: &Entries, planned: &[Planned]) {
        let stored: usize = entries
            .iter()
            .map(|(_, d, s)| s.iter().product::<usize>() * d.size())
            .sum();
        let held: usize = planned
            .iter()
            .map(|p| match GgmlType::from_u32(p.decl.type_id) {
                GgmlType::F32 => p.decl.nbytes as usize / 2,
                _ => p.decl.nbytes as usize,
            })
            .sum();
        assert_eq!(held, stored);
    }

    fn plan(config: &Json, entries: &Entries) -> (Checkpoint, Vec<Planned>) {
        let checkpoint = Checkpoint::parse(config).unwrap();
        let planning = folder::plan_tensors(
            &checkpoint,
            &Config::parse(config, None).unwrap(),
            entries,
            false,
            folder::Keep::InPlace,
        )
        .unwrap();
        assert!(
            planning.plan.refused.is_empty(),
            "{:?}",
            planning.plan.refused
        );
        (checkpoint, planning.planned)
    }

    /// The header keys the map makes, as the committed key files write them, `qwen3moe.<key>`.
    fn keys_of(checkpoint: &Checkpoint) -> serde_json::Map<String, Json> {
        checkpoint
            .keys
            .iter()
            .map(|(k, v)| (format!("{ARCHITECTURE}.{k}"), key_json(v)))
            .collect()
    }

    /// The fixture folder (the GGUF fixture's tensors as MLX's 4-bit checkpoint, its config and
    /// its tensors' headers committed) plans every tensor of the GGUF fixture: the same names and
    /// ggml dims, the router 8-bit and every other linear 4-bit MLX blocks, the BF16 vectors F32;
    /// the bytes a planned tensor holds are the bytes its weight, scales and biases take in the
    /// folder, and nothing of the folder is left unplanned. Its header keys are the GGUF
    /// fixture's `qwen3moe.*` keys.
    #[test]
    fn the_mlx_fixture_folder_plans_the_gguf_fixture_s_tensors() {
        let config = json("quant-configs/bloomery__qwen3moe-mlx-fixture__config.json.gz");
        let header = json("checkpoint-headers/qwen3moe-mlx-fixture.tensors.json.gz");
        let gguf = json("checkpoint-headers/qwen3moe-fixture-gguf.header.json.gz");
        let entries = entries(&header);
        let (checkpoint, planned) = plan(&config, &entries);
        let want = gguf_tensors(&gguf);
        let got: BTreeMap<&str, &Planned> =
            planned.iter().map(|p| (p.decl.name.as_str(), p)).collect();
        assert_eq!(
            got.keys().copied().collect::<Vec<_>>(),
            want.keys().map(String::as_str).collect::<Vec<_>>()
        );
        for (name, (dims, ty)) in &want {
            let planned = got[name.as_str()];
            let folder::Make::Tensor(m) = &planned.make else {
                panic!("{name}: a scale beside a type that reads none");
            };
            let (mlx, role) = mlx_type_of(name, *ty, Dtype::Bf16);
            assert_eq!(planned.decl.dims, *dims, "{name}");
            assert_eq!(GgmlType::from_u32(planned.decl.type_id), mlx, "{name}");
            assert_eq!(m.role, role, "{name}");
            assert!(m.steps.is_empty(), "{name}: {:?}", m.steps);
        }
        assert_every_stored_byte_is_planned(&entries, &planned);
        let mut want = gguf["keys"].as_object().unwrap().clone();
        want.remove(gguf::GENERAL_ARCHITECTURE);
        assert_eq!(Json::Object(keys_of(&checkpoint)), Json::Object(want));
    }

    /// A real 30B-A3B folder's config, with the tensor header both folders store (1,351 tensors:
    /// Coder's names, dtypes and shapes are the 2507's), plans the file's 579 tensors at the
    /// ggml dims of the model, MLX_A4 for the linears and the embedding and head, MLX_A8 for the
    /// routers, F32 for the norms, with no move. `dense_ff` is the config's `intermediate_size`,
    /// the header's `feed_forward_length`.
    fn assert_real_30b_plans(config: &str, dense_ff: u32) {
        let real = json(config);
        let header =
            json("checkpoint-headers/qwen3-30b-a3b-instruct-2507-mlx-4bit.tensors.json.gz");
        let entries = entries(&header);
        assert_eq!(entries.len(), 1351);
        let (checkpoint, planned) = plan(&real, &entries);
        assert_eq!(planned.len(), 3 + 48 * LAYER.len());
        let got: BTreeMap<&str, &Planned> =
            planned.iter().map(|p| (p.decl.name.as_str(), p)).collect();
        for l in [0, 47] {
            for stem in LAYER {
                assert!(
                    got.contains_key(common::blk(l, stem).as_str()),
                    "{l} {stem}"
                );
            }
        }
        for (name, dims, ty) in [
            ("token_embd.weight", vec![2048, 151_936], GgmlType::MLX_A4),
            ("output.weight", vec![2048, 151_936], GgmlType::MLX_A4),
            ("output_norm.weight", vec![2048], GgmlType::F32),
            ("blk.0.attn_norm.weight", vec![2048], GgmlType::F32),
            ("blk.0.attn_q.weight", vec![2048, 4096], GgmlType::MLX_A4),
            ("blk.0.attn_k.weight", vec![2048, 512], GgmlType::MLX_A4),
            ("blk.0.attn_v.weight", vec![2048, 512], GgmlType::MLX_A4),
            (
                "blk.0.attn_output.weight",
                vec![4096, 2048],
                GgmlType::MLX_A4,
            ),
            ("blk.0.attn_q_norm.weight", vec![128], GgmlType::F32),
            ("blk.47.attn_k_norm.weight", vec![128], GgmlType::F32),
            ("blk.47.ffn_norm.weight", vec![2048], GgmlType::F32),
            (
                "blk.47.ffn_gate_inp.weight",
                vec![2048, 128],
                GgmlType::MLX_A8,
            ),
            (
                "blk.0.ffn_gate_exps.weight",
                vec![2048, 768, 128],
                GgmlType::MLX_A4,
            ),
            (
                "blk.47.ffn_up_exps.weight",
                vec![2048, 768, 128],
                GgmlType::MLX_A4,
            ),
            (
                "blk.23.ffn_down_exps.weight",
                vec![768, 2048, 128],
                GgmlType::MLX_A4,
            ),
        ] {
            let planned = got[name];
            assert_eq!(planned.decl.dims, dims, "{name}");
            assert_eq!(GgmlType::from_u32(planned.decl.type_id), ty, "{name}");
            let folder::Make::Tensor(m) = &planned.make else {
                panic!("{name}: a scale beside a type that reads none");
            };
            assert!(m.steps.is_empty(), "{name}");
        }
        assert_every_stored_byte_is_planned(&entries, &planned);
        let want: serde_json::Map<String, Json> = [
            ("block_count", json!(["U32", 48])),
            ("context_length", json!(["U32", 262_144])),
            ("embedding_length", json!(["U32", 2048])),
            ("attention.head_count", json!(["U32", 32])),
            ("attention.head_count_kv", json!(["U32", 4])),
            ("rope.freq_base", json!(["F32", 10_000_000.0])),
            (
                "attention.layer_norm_rms_epsilon",
                json!(["F32", f64::from(1e-6_f32)]),
            ),
            ("expert_used_count", json!(["U32", 8])),
            ("attention.key_length", json!(["U32", 128])),
            ("attention.value_length", json!(["U32", 128])),
            ("expert_count", json!(["U32", 128])),
            ("expert_feed_forward_length", json!(["U32", 768])),
            ("feed_forward_length", json!(["U32", dense_ff])),
        ]
        .into_iter()
        .map(|(k, v)| (format!("{ARCHITECTURE}.{k}"), v))
        .collect();
        assert_eq!(Json::Object(keys_of(&checkpoint)), Json::Object(want));
    }

    /// mlx-community's Qwen3-30B-A3B-Instruct-2507-4bit plans the whole file.
    #[test]
    fn the_real_qwen3_30b_headers_plan_the_whole_file() {
        assert_real_30b_plans(
            "quant-configs/mlx-community__Qwen3-30B-A3B-Instruct-2507-4bit__config.json.gz",
            6144,
        );
    }

    /// mlx-community's Qwen3-Coder-30B-A3B-Instruct-4bit plans the whole file: its config names
    /// `qkv_bias`, `use_qk_norm` and `shared_expert_intermediate_size` at the values the map
    /// computes, and a dense width of 5472.
    #[test]
    fn the_real_qwen3_coder_headers_plan_the_whole_file() {
        assert_real_30b_plans(
            "quant-configs/mlx-community__Qwen3-Coder-30B-A3B-Instruct-4bit__config.json.gz",
            5472,
        );
    }

    /// The config refuses, by key, a model the program does not run as written: a dense layer, a
    /// sliding window, a scaled rope, HF's un-renormalized routing, a bias on the attention, no
    /// norm on the query and key, a shared expert, an unnamed key, heads that do not group, more
    /// experts kept than routed over, and a missing count.
    #[test]
    fn the_config_refuses_what_it_does_not_read_by_name() {
        let base = json("quant-configs/bloomery__qwen3moe-mlx-fixture__config.json.gz");
        assert!(Checkpoint::parse(&base).is_ok());
        let refuse = |edit: &dyn Fn(&mut serde_json::Map<String, Json>)| {
            let mut config = base.clone();
            edit(config.as_object_mut().unwrap());
            Checkpoint::parse(&config).unwrap_err().to_string()
        };
        for (key, value) in [
            ("norm_topk_prob", json!(false)),
            ("decoder_sparse_step", json!(2)),
            ("mlp_only_layers", json!([0])),
            ("rope_scaling", json!({"rope_type": "yarn", "factor": 4.0})),
            ("sliding_window", json!(4096)),
            ("use_sliding_window", json!(true)),
            ("tie_word_embeddings", json!(true)),
            ("hidden_act", json!("gelu")),
            ("architectures", json!(["Qwen3ForCausalLM"])),
            ("zzz", json!(1)),
        ] {
            let e = refuse(&|c| {
                c.insert(key.to_string(), value.clone());
            });
            assert!(e.contains(key), "{key}: {e}");
        }
        // Keys Coder's config names: a value other than the one the map computes is refused as
        // unsupported, not as an unknown key.
        for (key, value, supported) in [
            ("qkv_bias", json!(true), "[false]"),
            ("use_qk_norm", json!(false), "[true]"),
            ("shared_expert_intermediate_size", json!(512), "[0]"),
        ] {
            let e = refuse(&|c| {
                c.insert(key.to_string(), value.clone());
            });
            assert!(
                e.contains(&format!(
                    "config.json.{key} {value} is unsupported; supported: {supported}"
                )),
                "{key}: {e}"
            );
        }
        let e = refuse(&|c| {
            c.insert("num_attention_heads".into(), json!(33));
        });
        assert!(e.contains("must group by num_key_value_heads"), "{e}");
        let e = refuse(&|c| {
            c.insert("num_experts_per_tok".into(), json!(129));
        });
        assert!(e.contains("exceeds num_experts"), "{e}");
        let e = refuse(&|c| {
            c.remove("intermediate_size");
        });
        assert!(
            e.contains("config.json.intermediate_size is missing"),
            "{e}"
        );
        let e = refuse(&|c| {
            c.insert("moe_intermediate_size".into(), json!(0));
        });
        assert!(
            e.contains("config.json.moe_intermediate_size must be a positive u32"),
            "{e}"
        );
    }

    /// The map refuses, by name, a tensor no folder of MLX's has, HF's per-expert layout, a layer
    /// past the config's, a quantized linear without its scales, a tensor the file needs that the
    /// folder lacks, and a tensor of other dims than the config gives.
    #[test]
    fn the_map_refuses_what_it_does_not_read_by_name() {
        let config = json("quant-configs/bloomery__qwen3moe-mlx-fixture__config.json.gz");
        let header = json("checkpoint-headers/qwen3moe-mlx-fixture.tensors.json.gz");
        let base = entries(&header);
        let checkpoint = Checkpoint::parse(&config).unwrap();
        let quant = Config::parse(&config, None).unwrap();
        let refuse = |edit: &dyn Fn(&mut Entries)| {
            let mut e = base.clone();
            edit(&mut e);
            folder::plan_tensors(&checkpoint, &quant, &e, false, folder::Keep::InPlace)
                .err()
                .expect("refused")
                .to_string()
        };
        let at = |e: &mut Entries, name: &str| e.iter().position(|(n, _, _)| n == name).unwrap();
        let l = "model.layers";
        let e = refuse(&|e| e.push((format!("{l}.0.mlp.zzz.weight"), Dtype::Bf16, vec![8])));
        assert!(
            e.contains(&format!("{l}.0.mlp.zzz.weight")) && e.contains("not a Qwen3-MoE tensor"),
            "{e}"
        );
        let e = refuse(&|e| {
            e.push((
                format!("{l}.0.mlp.experts.0.gate_proj.weight"),
                Dtype::Bf16,
                vec![768, 2048],
            ));
        });
        assert!(
            e.contains(&format!("{l}.0.mlp.experts.0.gate_proj.weight"))
                && e.contains("reads MLX's stacked `switch_mlp`"),
            "{e}"
        );
        let e = refuse(&|e| {
            e.push((
                format!("{l}.4.input_layernorm.weight"),
                Dtype::Bf16,
                vec![8],
            ))
        });
        assert!(e.contains("layer 4 of 4"), "{e}");
        // A weight whose scales and biases are gone is a plain U32 tensor of its packed shape:
        // the map refuses it by name at the dims the config gives, before the plan's own refusal.
        let e = refuse(&|e| {
            for suffix in ["scales", "biases"] {
                let i = at(e, &format!("{l}.0.mlp.gate.{suffix}"));
                e.remove(i);
            }
        });
        assert!(
            e.contains(&format!("{l}.0.mlp.gate.weight"))
                && e.contains("holds dims [512, 128] where the config gives [2048, 128]"),
            "{e}"
        );
        let e = refuse(&|e| {
            let i = at(e, &format!("{l}.2.self_attn.q_norm.weight"));
            e.remove(i);
        });
        assert!(
            e.contains("blk.2.attn_q_norm.weight") && e.contains("is missing from the folder"),
            "{e}"
        );
        // Half the key rows of one layer, all three tensors of the linear alike: the plan is
        // whole, the file's dims are not the config's.
        let e = refuse(&|e| {
            for suffix in ["weight", "scales", "biases"] {
                let i = at(e, &format!("{l}.1.self_attn.k_proj.{suffix}"));
                e[i].2[0] /= 2;
            }
        });
        assert!(
            e.contains(&format!("{l}.1.self_attn.k_proj.weight"))
                && e.contains("holds dims [2048, 256] where the config gives [2048, 512]"),
            "{e}"
        );
    }

    /// A folder of two layers of a Qwen3-MoE as small as the groups allow, written in MLX's names
    /// with the real file's `quantization` block (no `mode`), opens at both scale pairs the map
    /// reads (BF16's MLX_A4/MLX_A8 and F16's MLX_A4H/MLX_A8H): its planned tensors and header
    /// keys read as a qwen3moe file through the engine's own header reader, every tensor of the
    /// source reads whole in the planned bytes, the stacked experts and the matrices as the stored
    /// words with their scales and biases, and the norm gammas as stored.
    #[test]
    fn a_small_qwen3_folder_opens_as_a_qwen3moe_file() {
        // Both scale pairs the folder map reads: the BF16 pairs of the real
        // 4-bit folders and their F16 forms.
        for (pair, a4, a8, pair_bytes) in [
            (
                Dtype::Bf16,
                GgmlType::MLX_A4,
                GgmlType::MLX_A8,
                [0x00u8, 0x3f, 0x80, 0xbe],
            ),
            (
                Dtype::F16,
                GgmlType::MLX_A4H,
                GgmlType::MLX_A8H,
                [0x00u8, 0x38, 0x00, 0xb4],
            ),
        ] {
            let lin = |stem: &str, bits: usize, lead: &[usize], k: usize| {
                mlx_linear_pair(stem, bits, lead, k, pair)
            };

            use model::arch::qwen3moe::hparams::Hparams;
            use model::part::{Source, Take};

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
            let dir = temp_dir("qwen3moe-open");
            let mut t: Vec<Stored> = vec![];
            for stem in ["model.embed_tokens", "lm_head"] {
                t.extend(lin(stem, 4, &[8], 64));
            }
            t.push(bf16_tensor("model.norm.weight", &[64], |_| 0x3fc0));
            for l in 0..2 {
                let m = format!("model.layers.{l}");
                t.push(bf16_tensor(
                    &format!("{m}.input_layernorm.weight"),
                    &[64],
                    |_| 0x3fc0,
                ));
                t.push(bf16_tensor(
                    &format!("{m}.post_attention_layernorm.weight"),
                    &[64],
                    |_| 0x3f80,
                ));
                t.extend(lin(&format!("{m}.mlp.gate"), 8, &[4], 64));
                for proj in ["gate_proj", "up_proj", "down_proj"] {
                    t.extend(lin(&format!("{m}.mlp.switch_mlp.{proj}"), 4, &[4, 64], 64));
                }
                for (stem, rows) in [
                    ("q_proj", 64),
                    ("k_proj", 32),
                    ("v_proj", 32),
                    ("o_proj", 64),
                ] {
                    t.extend(lin(&format!("{m}.self_attn.{stem}"), 4, &[rows], 64));
                }
                for norm in ["q_norm", "k_norm"] {
                    t.push(bf16_tensor(
                        &format!("{m}.self_attn.{norm}.weight"),
                        &[32],
                        |_| 0x3e80,
                    ));
                }
            }
            write_folder(&dir, &t, &config, &tokenizer("qwen2"));
            let opened = crate::folder::open(&dir).unwrap();
            let named = crate::folder::architecture(config.to_string().as_bytes()).unwrap();
            assert_eq!(Some(named), opened.header.architecture());
            let hp = Hparams::read(&opened.header).unwrap();
            assert_eq!(
                (hp.n_layer, hp.n_embd, hp.n_head, hp.n_head_kv),
                (2, 64, 2, 1)
            );
            assert_eq!((hp.head_dim, hp.rope.dims), (32, 32));
            assert_eq!(hp.rope.base, 1e6);
            assert_eq!(
                (hp.experts.n_expert, hp.experts.n_used, hp.experts.ff),
                (4, 2, 64)
            );
            let ty = |name: &str| opened.header.find(name).unwrap().1.ty;
            assert_eq!(ty("token_embd.weight"), a4);
            assert_eq!(ty("output.weight"), a4);
            assert_eq!(ty("blk.0.attn_q.weight"), a4);
            assert_eq!(ty("blk.1.ffn_down_exps.weight"), a4);
            assert_eq!(ty("blk.1.ffn_gate_inp.weight"), a8);
            assert_eq!(ty("blk.0.attn_q_norm.weight"), GgmlType::F32);
            assert_eq!(ty("output_norm.weight"), GgmlType::F32);
            assert_eq!(opened.header.iter_tensors().count(), 3 + 2 * LAYER.len());
            let planned = opened
                .source
                .plan(|_, _, _| Ok::<_, String>(Take::Whole))
                .unwrap();
            assert_eq!(planned.len(), opened.header.iter_tensors().count());
            let read = |name: &str| -> Vec<u8> {
                let p = planned.iter().find(|p| p.info.name == name).unwrap();
                let part = opened.source.read(p).unwrap();
                assert_eq!(part.spans().concat().len() as u64, p.info.nbytes, "{name}");
                part.spans().concat()
            };
            for p in &planned {
                assert_eq!(read(&p.info.name).len() as u64, p.info.nbytes);
            }
            let f32s = |name: &str| -> Vec<f32> {
                read(name)
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect()
            };
            assert_eq!(f32s("blk.1.attn_q_norm.weight"), [0.25; 32]);
            assert_eq!(f32s("blk.1.attn_k_norm.weight"), [0.25; 32]);
            assert_eq!(f32s("blk.0.attn_norm.weight"), [1.5; 64]);
            assert_eq!(f32s("blk.0.ffn_norm.weight"), [1.0; 64]);
            assert_eq!(f32s("output_norm.weight"), [1.5; 64]);
            // A group is one row here: its 8 code words, then its scale and its bias (0.5, -0.25).
            let words = |row: usize| -> Vec<u8> {
                let at = (row * 8) as u32;
                (at..at + 8)
                    .flat_map(|i| i.wrapping_mul(0x9e37_79b9).to_le_bytes())
                    .collect()
            };
            let q = read("blk.0.attn_q.weight");
            assert_eq!(q.len(), 64 * 36);
            for row in [0, 1, 63] {
                assert_eq!(q[row * 36..][..32], words(row)[..], "attn_q row {row}");
                assert_eq!(q[row * 36 + 32..][..4], pair_bytes, "attn_q row {row}");
            }
            // The stack is the stored 3-D tensor, expert after expert.
            let up = read("blk.1.ffn_up_exps.weight");
            assert_eq!(up.len(), 4 * 64 * 36);
            for (expert, row) in [(0, 0), (1, 5), (3, 63)] {
                let at = (expert * 64 + row) * 36;
                assert_eq!(
                    up[at..][..32],
                    words(expert * 64 + row)[..],
                    "expert {expert} row {row}"
                );
            }
            std::fs::remove_dir_all(&dir).unwrap();
        }
    }

    /// A folder whose files name no BOS id (Qwen3-Coder's) opens as a qwen3moe file whose split
    /// carries no BOS key and whose tokenizer adds none; the same folder with `add_bos_token` set
    /// is refused by the missing key.
    #[test]
    fn a_folder_that_names_no_bos_opens_without_a_bos_key() {
        let dir = temp_dir("qwen3moe-no-bos");
        crate::test_support::small_qwen3_folder(&dir);
        let path = dir.join("config.json");
        let mut config: Json = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert!(
            config
                .as_object_mut()
                .unwrap()
                .remove("bos_token_id")
                .is_some()
        );
        std::fs::write(&path, config.to_string()).unwrap();
        let opened = crate::folder::open(&dir).unwrap();
        assert_eq!(opened.header.value("tokenizer.ggml.bos_token_id"), None);
        assert_eq!(
            opened.header.value("tokenizer.ggml.add_bos_token"),
            Some(&Value::Bool(false))
        );
        assert_eq!(opened.tokenizer.specials().bos, None);
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"add_bos_token": true}"#,
        )
        .unwrap();
        let e = crate::folder::open(&dir)
            .err()
            .expect("refused")
            .to_string();
        assert!(e.contains("tokenizer.ggml.bos_token_id"), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
