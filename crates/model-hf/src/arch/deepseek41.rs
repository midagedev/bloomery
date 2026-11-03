//! DeepSeek-V4.1-Flash's checkpoint folder (`deepseek_v41`, DeepSeek's own names) in the
//! deepseek41 names, layouts and keys the program reads. Every tensor maps to one deepseek41
//! tensor with no move: the folder's row-major shapes reversed are ggml's dims. A layer's routed
//! experts, one tensor an expert and projection, stack by expert id.
//!
//! The formats are the quantization config's (`fp8`, 32 by 32 blocks, `ue8m0` scales, `fp4`
//! experts): the dense projections are F8_E4M3 beside an E8M0 scale a 32 by 32 block (read as
//! MXFP8), the routed experts MXFP4 codes beside an E8M0 scale a 32 values, an Engram table
//! F8_E4M3 rows beside an E8M0 scale a row's 32 values (an MXFP8 row table read where the folder
//! stores it), and the rest BF16 or F32. The DSpark draft (`mtp.*`) and the vision tower
//! (`vision.*`, `aligner.*`, `image_*`) are skipped with the reason.
//!
//! The header's keys are the config's ([`Ds41Map::parse`]) and the Engram hash constants the
//! reference derives from the config and the tokenizer ([`hf::engram`]); the token map's
//! compressed vocabulary must be the config's `engram_compressed_vocab_size`.

use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;

use gguf::arch::deepseek41::ARCH as ARCHITECTURE;
use gguf::{GgmlType, Value};
use hf::modelopt::{Plain, Use};
use hf::quant_config::ModuleClass;
use hf::schema::{Fields, Rule::*, Type};
use model::arch::deepseek41::names;
use safetensors::Folder;
use serde_json::Value as Json;

pub mod fixture;
mod rank;
pub use rank::{preflight, rank_reads};

use super::section::Section;
use crate::folder::{Error, FolderMap, Mapped, Mapping, Origin, VocabSpec, read_json, refuse};

/// The names a layer's tensors take, each a function of the layer.
type LayerNames = [fn(usize) -> String];

/// One stack's routed experts by id: each stored name and its shape.
type Stack<'a> = BTreeMap<usize, (&'a str, &'a [usize])>;

/// `config.json`'s top-level keys.
const ROOT: &Fields = &[
    ("architectures", Need(r#"[["DeepseekV41ForCausalLM"]]"#)),
    ("model_type", Need(r#"["deepseek_v41"]"#)),
    ("quantization_config", Read),
    ("text_config", Read),
    ("bos_token_id", Read),
    ("eos_token_id", Read),
    ("pad_token_id", Read),
    (
        "image_token_id",
        Ignore(
            Type::Number,
            "the image placeholder; the vision tower is skipped",
        ),
    ),
    (
        "vision_config",
        Ignore(
            Type::Object,
            "the vision tower's geometry; its tensors are skipped",
        ),
    ),
    (
        "dtype",
        Ignore(
            Type::String,
            "the dtype HF loads in; each tensor keeps its stored format",
        ),
    ),
    (
        "transformers_version",
        Ignore(Type::String, "the version that wrote the file"),
    ),
];

/// The DSpark draft's keys, which the draft's own reader takes.
const DSPARK: &str = "the DSpark draft's geometry; its mtp tensors are skipped";

/// `text_config`'s keys.
const TEXT: &Fields = &[
    ("model_type", Need(r#"["deepseek_v41_text"]"#)),
    ("vocab_size", Read),
    ("hidden_size", Read),
    ("moe_intermediate_size", Read),
    ("num_hidden_layers", Read),
    ("num_attention_heads", Read),
    ("num_key_value_heads", Read),
    ("head_dim", Read),
    ("qk_rope_head_dim", Read),
    ("q_lora_rank", Read),
    ("o_lora_rank", Read),
    ("o_groups", Read),
    ("hidden_act", Need(r#"["silu"]"#)),
    ("swiglu_limit", Read),
    ("rms_norm_eps", Read),
    ("attention_bias", Only("[false]")),
    ("tie_word_embeddings", Only("[false]")),
    ("max_position_embeddings", Read),
    ("rope_theta", Read),
    ("rope_scaling", Read),
    ("n_routed_experts", Read),
    ("n_shared_experts", Read),
    ("num_experts_per_tok", Read),
    ("scoring_func", Need(r#"["sqrtsoftplus"]"#)),
    ("topk_method", Only(r#"["noaux_tc"]"#)),
    ("norm_topk_prob", Read),
    ("routed_scaling_factor", Read),
    ("sliding_window", Read),
    ("compress_ratios", Read),
    ("compress_rope_theta", Read),
    // The layers a compressor and an indexer stand on, which the tensors they carry must be.
    ("kv_source_layer_ids", Read),
    ("index_source_layer_ids", Read),
    ("index_n_heads", Read),
    ("index_head_dim", Read),
    ("index_topk", Read),
    ("candidate_source_layer_id", Read),
    ("candidate_topk_blocks", Read),
    ("candidate_block_size", Read),
    ("hc_mult", Read),
    ("hc_sinkhorn_iters", Read),
    ("hc_eps", Read),
    ("engram_layer_ids", Read),
    ("engram_num_embeddings", Read),
    ("engram_max_ngram_size", Read),
    ("engram_vocab_size", Read),
    ("engram_n_heads", Read),
    ("engram_head_dim", Read),
    ("engram_pad_token_id", Read),
    ("engram_compressed_vocab_size", Read),
    ("num_nextn_predict_layers", Read),
    ("dspark_block_size", Ignore(Type::Number, DSPARK)),
    ("dspark_noise_token_id", Ignore(Type::Number, DSPARK)),
    ("dspark_target_layer_ids", Read),
    ("dspark_markov_rank", Ignore(Type::Number, DSPARK)),
    ("dspark_n_routed_experts", Ignore(Type::Number, DSPARK)),
    ("dspark_num_experts_per_tok", Ignore(Type::Number, DSPARK)),
    (
        "attention_dropout",
        Ignore(Type::Number, "a training rate; inference drops nothing"),
    ),
    (
        "initializer_range",
        Ignore(Type::Number, "a training initializer's spread"),
    ),
    (
        "use_cache",
        Ignore(
            Type::Bool,
            "whether HF keeps its cache; outputs are the same",
        ),
    ),
];

/// `rope_scaling`'s keys: YaRN's.
const ROPE: &Fields = &[
    ("rope_type", Need(r#"["yarn"]"#)),
    ("factor", Read),
    ("beta_fast", Read),
    ("beta_slow", Read),
    ("original_max_position_embeddings", Read),
];

/// The Engram the config describes, for its hash constants.
#[derive(Debug)]
struct EngramConfig {
    layer_ids: Vec<u64>,
    heads: usize,
    max_ngram: usize,
    vocab: u64,
    compressed: u64,
    pad: usize,
    /// Each site's table dims: `engram_head_dim` values by `engram_num_embeddings` rows.
    tables: Vec<[u64; 2]>,
}

/// The config values the map reads, checked, and the header keys they make.
#[derive(Debug)]
pub struct Ds41Map {
    layers: usize,
    experts: usize,
    hidden: u64,
    kv_sources: BTreeSet<usize>,
    index_sources: BTreeSet<usize>,
    engram: EngramConfig,
    vocab: VocabSpec,
    /// The header's model keys by suffix, under the architecture's name.
    keys: Vec<(String, Value)>,
}

/// The folder tensors no deepseek41 tensor is made of, and why; `None` for one the map reads.
fn skip_reason(name: &str) -> Option<&'static str> {
    if name.starts_with("mtp.") {
        Some("the DSpark draft's layers, which the draft's own reader takes")
    } else if name.starts_with("vision.")
        || name.starts_with("aligner.")
        || name.starts_with("image_")
    {
        Some("the vision tower and its projector, which the image path reads")
    } else {
        None
    }
}

/// The deepseek41 tensor and role of the layer tensor `rest` of layer `l` (the name past
/// `layers.L.`), but a routed expert's; `None` for no tensor of a layer.
fn layer_tensor(l: usize, rest: &str) -> Option<(String, Plain)> {
    use Plain::{Bf16, F32, Q8_0};
    Some(match rest {
        "attn_norm.weight" => (names::attn_norm(l), F32),
        "ffn_norm.weight" => (names::ffn_norm(l), F32),
        "attn.q_norm.weight" => (names::attn_q_a_norm(l), F32),
        "attn.kv_norm.weight" => (names::attn_kv_a_norm(l), F32),
        "attn.attn_sink" => (names::attn_sinks(l), F32),
        "attn.wq_a.weight" => (names::attn_q_a(l), Q8_0),
        "attn.wq_b.weight" => (names::attn_q_b(l), Q8_0),
        "attn.wkv.weight" => (names::attn_kv(l), Q8_0),
        "attn.wo_a.weight" => (names::attn_output_a(l), Q8_0),
        "attn.wo_b.weight" => (names::attn_output_b(l), Q8_0),
        "attn.compressor.wkv.weight" => (names::attn_compressor_kv(l), Q8_0),
        "attn.compressor.wgate.weight" => (names::attn_compressor_gate(l), Q8_0),
        "attn.compressor.norm.weight" => (names::attn_compressor_norm(l), F32),
        "attn.indexer.wq_b.weight" => (names::indexer_attn_q_b(l), Q8_0),
        "attn.indexer.wk.weight" => (names::indexer_attn_k(l), Q8_0),
        "attn.indexer.weights_proj.weight" => (names::indexer_proj(l), Q8_0),
        "attn.indexer.k_norm.weight" => (names::indexer_k_norm(l), F32),
        "ffn.gate.weight" => (names::ffn_gate_inp(l), Bf16),
        "ffn.gate.bias" => (names::exp_probs_b(l), F32),
        "ffn.gate.bias_vl" => (names::exp_probs_b_vl(l), F32),
        "ffn.shared_experts.w1.weight" => (names::ffn_gate_shexp(l), Q8_0),
        "ffn.shared_experts.w3.weight" => (names::ffn_up_shexp(l), Q8_0),
        "ffn.shared_experts.w2.weight" => (names::ffn_down_shexp(l), Q8_0),
        "hc_attn_fn" => (names::hc_attn_fn(l), F32),
        "hc_attn_base" => (names::hc_attn_base(l), F32),
        "hc_attn_scale" => (names::hc_attn_scale(l), F32),
        "hc_ffn_fn" => (names::hc_ffn_fn(l), F32),
        "hc_ffn_base" => (names::hc_ffn_base(l), F32),
        "hc_ffn_scale" => (names::hc_ffn_scale(l), F32),
        "engram.q_weight" => (names::engram_q(l), Bf16),
        "engram.k_weight" => (names::engram_k(l), Bf16),
        "engram.wkv.weight" => (names::engram_wkv(l), Q8_0),
        _ => return None,
    })
}

/// The tensors every layer carries.
const EVERY_LAYER: [fn(usize) -> String; 22] = [
    names::attn_norm,
    names::ffn_norm,
    names::attn_q_a_norm,
    names::attn_kv_a_norm,
    names::attn_sinks,
    names::attn_q_a,
    names::attn_q_b,
    names::attn_kv,
    names::attn_output_a,
    names::attn_output_b,
    names::ffn_gate_inp,
    names::exp_probs_b,
    names::ffn_gate_shexp,
    names::ffn_up_shexp,
    names::ffn_down_shexp,
    names::ffn_gate_exps,
    names::ffn_up_exps,
    names::ffn_down_exps,
    names::hc_attn_fn,
    names::hc_ffn_fn,
    names::hc_attn_base,
    names::hc_ffn_base,
];

/// `key` of `s` as a list of layer indices.
fn layer_list(s: &Section<'_, Error>, key: &str) -> Result<Vec<usize>, Error> {
    s.array(key)?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| s.bad(key, "is not a list of layer indices"))
        })
        .collect()
}

impl Ds41Map {
    fn read(config: &Json) -> Result<Ds41Map, Error> {
        let root = Section::<Error>::new(config, "config.json")?;
        root.check(&[ROOT])?;
        let text = Section::<Error>::new(root.field("text_config")?, "text_config")?;
        text.check(&[TEXT])?;
        let rope = Section::<Error>::new(text.field("rope_scaling")?, "text_config.rope_scaling")?;
        rope.check(&[ROPE])?;
        let n = |key: &str| text.count(key);
        let f = |key: &str| text.number(key);
        let layers = n("num_hidden_layers")? as usize;
        let ratios = text.array("compress_ratios")?;
        let ratios: Vec<Value> = ratios
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|r| u32::try_from(r).ok())
                    .map(Value::U32)
                    .ok_or_else(|| text.bad("compress_ratios", "is not a list of u32 ratios"))
            })
            .collect::<Result<_, _>>()?;
        if ratios.len() < layers {
            return Err(text.bad("compress_ratios", "names fewer ratios than layers"));
        }
        let engram_ids = layer_list(&text, "engram_layer_ids")?;
        let rows: Vec<u64> = text
            .array("engram_num_embeddings")?
            .iter()
            .map(|v| v.as_u64())
            .collect::<Option<_>>()
            .ok_or_else(|| text.bad("engram_num_embeddings", "is not a list of row counts"))?;
        let width = u64::from(n("engram_head_dim")?);
        let engram = EngramConfig {
            layer_ids: engram_ids.iter().map(|&l| l as u64).collect(),
            heads: n("engram_n_heads")? as usize,
            max_ngram: n("engram_max_ngram_size")? as usize,
            vocab: u64::from(n("engram_vocab_size")?),
            compressed: u64::from(n("engram_compressed_vocab_size")?),
            pad: text
                .field("engram_pad_token_id")?
                .as_u64()
                .and_then(|p| usize::try_from(p).ok())
                .ok_or_else(|| text.bad("engram_pad_token_id", "is not a token id"))?,
            tables: rows.iter().map(|&r| [width, r]).collect(),
        };
        let orders = engram
            .max_ngram
            .checked_sub(1)
            .filter(|&o| o > 0)
            .ok_or_else(|| {
                text.bad(
                    "engram_max_ngram_size",
                    "leaves no n-gram order past the 1-gram",
                )
            })?;
        let sizes = hf::engram::primes(engram.layer_ids.len(), orders, engram.heads, engram.vocab)
            .map_err(|e| text.bad("engram_n_heads", &e))?;
        let sums: Vec<u64> = sizes.iter().map(|s| s.iter().sum()).collect();
        if sums != rows {
            return Err(text.bad(
                "engram_num_embeddings",
                &format!("is {rows:?}, where the sites' buckets sum to {sums:?}"),
            ));
        }
        let norm_topk = text.field("norm_topk_prob")?.as_bool();
        let norm_topk = norm_topk.ok_or_else(|| text.bad("norm_topk_prob", "is not a bool"))?;
        let u32s =
            |list: &[usize]| Value::Array(list.iter().map(|&l| Value::U32(l as u32)).collect());
        let mut keys: Vec<(String, Value)> = vec![
            ("vocab_size", Value::U32(n("vocab_size")?)),
            ("context_length", Value::U32(n("max_position_embeddings")?)),
            ("embedding_length", Value::U32(n("hidden_size")?)),
            ("block_count", Value::U32(n("num_hidden_layers")?)),
            (
                "attention.head_count",
                Value::U32(n("num_attention_heads")?),
            ),
            (
                "attention.head_count_kv",
                Value::U32(n("num_key_value_heads")?),
            ),
            ("attention.key_length", Value::U32(n("head_dim")?)),
            ("attention.value_length", Value::U32(n("head_dim")?)),
            ("rope.dimension_count", Value::U32(n("qk_rope_head_dim")?)),
            ("attention.q_lora_rank", Value::U32(n("q_lora_rank")?)),
            ("attention.output_group_count", Value::U32(n("o_groups")?)),
            ("attention.output_lora_rank", Value::U32(n("o_lora_rank")?)),
            ("attention.sliding_window", Value::U32(n("sliding_window")?)),
            (
                "attention.layer_norm_rms_epsilon",
                Value::F32(f("rms_norm_eps")?),
            ),
            ("rope.freq_base", Value::F32(f("rope_theta")?)),
            (
                "attention.compress_rope_freq_base",
                Value::F32(f("compress_rope_theta")?),
            ),
            ("attention.compress_ratios", Value::Array(ratios)),
            ("rope.scaling.type", Value::String("yarn".into())),
            ("rope.scaling.factor", Value::F32(rope.number("factor")?)),
            (
                "rope.scaling.yarn_beta_fast",
                Value::F32(rope.number("beta_fast")?),
            ),
            (
                "rope.scaling.yarn_beta_slow",
                Value::F32(rope.number("beta_slow")?),
            ),
            (
                "rope.scaling.original_context_length",
                Value::U32(rope.count("original_max_position_embeddings")?),
            ),
            (
                "attention.indexer.head_count",
                Value::U32(n("index_n_heads")?),
            ),
            (
                "attention.indexer.key_length",
                Value::U32(n("index_head_dim")?),
            ),
            ("attention.indexer.top_k", Value::U32(n("index_topk")?)),
            (
                "attention.candidate_source_layer_id",
                Value::U32(
                    text.field("candidate_source_layer_id")?
                        .as_u64()
                        .and_then(|id| u32::try_from(id).ok())
                        .ok_or_else(|| {
                            text.bad("candidate_source_layer_id", "must be a u32 layer index")
                        })?,
                ),
            ),
            (
                "attention.candidate_block_size",
                Value::U32(n("candidate_block_size")?),
            ),
            (
                "attention.candidate_topk_blocks",
                Value::U32(n("candidate_topk_blocks")?),
            ),
            ("expert_count", Value::U32(n("n_routed_experts")?)),
            ("expert_used_count", Value::U32(n("num_experts_per_tok")?)),
            ("expert_shared_count", Value::U32(n("n_shared_experts")?)),
            (
                "expert_feed_forward_length",
                Value::U32(n("moe_intermediate_size")?),
            ),
            (
                "expert_weights_scale",
                Value::F32(f("routed_scaling_factor")?),
            ),
            ("expert_weights_norm", Value::Bool(norm_topk)),
            (
                "expert_gating_func",
                Value::String(gguf::arch::deepseek41::SQRT_SOFTPLUS.into()),
            ),
            ("swiglu_clamp_exp", Value::F32(f("swiglu_limit")?)),
            ("swiglu_clamp_shexp", Value::F32(f("swiglu_limit")?)),
            ("hash_layer_count", Value::U32(0)),
            ("hyper_connection.count", Value::U32(n("hc_mult")?)),
            (
                "hyper_connection.sinkhorn_iterations",
                Value::U32(n("hc_sinkhorn_iters")?),
            ),
            ("hyper_connection.epsilon", Value::F32(f("hc_eps")?)),
            ("engram.layer_ids", u32s(&engram_ids)),
            ("engram.head_count", Value::U32(n("engram_n_heads")?)),
            (
                "engram.max_ngram_size",
                Value::U32(n("engram_max_ngram_size")?),
            ),
            ("engram.key_length", Value::U32(n("engram_head_dim")?)),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let site_values = |values: Vec<Vec<u64>>| {
            Value::Array(values.into_iter().flatten().map(Value::U64).collect())
        };
        let offsets = sizes.iter().map(|s| hf::engram::offsets(s)).collect();
        keys.push(("engram.primes".into(), site_values(sizes)));
        keys.push(("engram.offsets".into(), site_values(offsets)));
        let vocab = hf::vocab::Config::parse(config)?.with_pre("deepseek-v3")?;
        Ok(Ds41Map {
            layers,
            experts: n("n_routed_experts")? as usize,
            hidden: u64::from(n("hidden_size")?),
            kv_sources: layer_list(&text, "kv_source_layer_ids")?
                .into_iter()
                .collect(),
            index_sources: layer_list(&text, "index_source_layer_ids")?
                .into_iter()
                .collect(),
            engram,
            vocab,
            keys,
        })
    }

    /// The Engram keys the tokens make, `tokens` each token's spelling by id: the token map,
    /// the pad as a mapped id and the multipliers of the compressed vocabulary, which must be
    /// the config's.
    fn hash_keys(&self, tokens: &[&str]) -> Result<Vec<(String, Value)>, Error> {
        let e = &self.engram;
        let (map, n) = hf::engram::token_map(tokens);
        if u64::from(n) != e.compressed {
            return Err(Error::Refused(format!(
                "tokenizer.json: its {} tokens compress to {n} ids, where \
                 text_config.engram_compressed_vocab_size is {}",
                tokens.len(),
                e.compressed
            )));
        }
        let pad = *map.get(e.pad).ok_or_else(|| {
            Error::Refused(format!(
                "text_config.engram_pad_token_id {} is past the {} tokens",
                e.pad,
                tokens.len()
            ))
        })?;
        let mult = hf::engram::multipliers(&e.layer_ids, e.max_ngram, e.compressed)
            .map_err(Error::Refused)?;
        let arch = |k: &str| format!("{ARCHITECTURE}.{k}");
        Ok(vec![
            (arch("engram.pad_id"), Value::U32(pad)),
            (
                arch("engram.multipliers"),
                Value::Array(mult.into_iter().map(Value::U64).collect()),
            ),
            (
                arch("engram.token_map"),
                Value::Array(map.into_iter().map(Value::U32).collect()),
            ),
        ])
    }

    /// Every tensor the config says the folder holds is mapped: each layer's [`EVERY_LAYER`],
    /// a compressor (and its norm) on the config's KV source layers alone, the indexer's
    /// projections on its index source layers, its keys and norm on the KV source layers, and
    /// an Engram site's four tensors on its layers alone, its table `engram_head_dim` by
    /// `engram_num_embeddings` rows; the embedding and the head are `[hidden, vocab]`.
    fn complete(&self, mapped: &BTreeMap<String, Mapped<Infallible>>) -> Result<(), Error> {
        let has = |name: &str| mapped.contains_key(name);
        let sites: BTreeSet<usize> = self.engram.layer_ids.iter().map(|&l| l as usize).collect();
        for l in 0..self.layers {
            if let Some(name) = EVERY_LAYER.iter().map(|f| f(l)).find(|n| !has(n)) {
                return Err(refuse(&name, "is missing from the folder"));
            }
            let groups: [(&BTreeSet<usize>, &LayerNames, &str); 3] = [
                (
                    &self.kv_sources,
                    &[
                        names::attn_compressor_kv,
                        names::attn_compressor_norm,
                        names::indexer_attn_k,
                        names::indexer_k_norm,
                    ],
                    "kv_source_layer_ids",
                ),
                (
                    &self.index_sources,
                    &[names::indexer_attn_q_b, names::indexer_proj],
                    "index_source_layer_ids",
                ),
                (
                    &sites,
                    &[
                        names::engram_embd,
                        names::engram_q,
                        names::engram_k,
                        names::engram_wkv,
                    ],
                    "engram_layer_ids",
                ),
            ];
            for (layers, stems, key) in groups {
                for name in stems.iter().map(|f| f(l)) {
                    if has(&name) != layers.contains(&l) {
                        return Err(refuse(
                            &name,
                            format!("is where text_config.{key} {layers:?} does not say"),
                        ));
                    }
                }
            }
        }
        for (&l, want) in self.engram.layer_ids.iter().zip(&self.engram.tables) {
            let name = names::engram_embd(l as usize);
            let m = mapped
                .get(&name)
                .ok_or_else(|| refuse(&name, "is missing"))?;
            if m.dims != want {
                return Err(refuse(
                    &m.sources[0],
                    format!(
                        "dims {:?}, not text_config.engram_head_dim by engram_num_embeddings {want:?}",
                        m.dims
                    ),
                ));
            }
        }
        let vocab = u64::from(self.vocab.size());
        for name in [names::token_embd(), names::output()] {
            let m = mapped
                .get(&name)
                .ok_or_else(|| refuse(&name, "is missing"))?;
            if m.dims != [self.hidden, vocab] {
                return Err(refuse(
                    &m.sources[0],
                    format!(
                        "dims {:?}, not hidden {} by vocabulary {vocab}",
                        m.dims, self.hidden
                    ),
                ));
            }
        }
        Ok(())
    }
}

impl FolderMap for Ds41Map {
    type Step = Infallible;

    const MODEL_TYPES: &'static [&'static str] = &["deepseek_v41"];

    fn parse(config: &Json) -> Result<Self, Error> {
        Ds41Map::read(config)
    }

    fn architecture(&self) -> &'static str {
        ARCHITECTURE
    }

    fn module_class(&self, module: &str) -> Option<ModuleClass> {
        if module == "embed" {
            return Some(ModuleClass::Embedding);
        }
        let (_, rest) = module.strip_prefix("layers.")?.split_once('.')?;
        let linear = rest.starts_with("attn.w")
            || rest.starts_with("ffn.experts.")
            || rest.starts_with("ffn.shared_experts.")
            || rest == "attn.indexer.wq_b"
            || rest == "engram.wkv";
        linear.then_some(ModuleClass::Linear)
    }

    fn map<'a>(&self, tensors: &[(&'a str, &'a [usize])]) -> Result<Mapping<Infallible>, Error> {
        let mut out = Mapping {
            tensors: BTreeMap::new(),
            skipped: BTreeMap::new(),
        };
        let mut stacks: BTreeMap<(usize, &str), Stack> = BTreeMap::new();
        for &(name, shape) in tensors {
            if let Some(why) = skip_reason(name) {
                out.skipped.insert(name.to_string(), why.to_string());
                continue;
            }
            if !super::trunk::sized(shape) {
                return Err(refuse(
                    name,
                    format!("shape {shape:?} is empty, zero, or overflows usize"),
                ));
            }
            let one = |role: Use, origin: Origin| Mapped {
                sources: vec![name.to_string()],
                role,
                steps: vec![],
                dims: shape.iter().rev().map(|&d| d as u64).collect(),
                origin,
            };
            let (to, m) = match name {
                "embed.weight" => (
                    names::token_embd(),
                    one(Use::Matrix(Plain::Bf16), Origin::File(GgmlType::BF16)),
                ),
                "head.weight" => (names::output(), one(Use::Matrix(Plain::Q8_0), Origin::Made)),
                "norm.weight" => (
                    names::output_norm(),
                    one(Use::Matrix(Plain::F32), Origin::Made),
                ),
                _ => {
                    let (l, rest) = name
                        .strip_prefix("layers.")
                        .and_then(|r| r.split_once('.'))
                        .and_then(|(l, rest)| Some((l.parse::<usize>().ok()?, rest)))
                        .ok_or_else(|| refuse(name, "not a DeepSeek-V4.1 tensor"))?;
                    if l >= self.layers {
                        return Err(refuse(name, format!("layer {l} of {}", self.layers)));
                    }
                    if let Some((id, projection)) = rest
                        .strip_prefix("ffn.experts.")
                        .and_then(|e| e.split_once('.'))
                        .and_then(|(id, p)| Some((id.parse::<usize>().ok()?, p)))
                    {
                        if !matches!(projection, "w1.weight" | "w2.weight" | "w3.weight") {
                            return Err(refuse(name, "not a DeepSeek-V4.1 expert projection"));
                        }
                        if stacks
                            .entry((l, projection))
                            .or_default()
                            .insert(id, (name, shape))
                            .is_some()
                        {
                            return Err(refuse(name, format!("duplicate expert id {id}")));
                        }
                        continue;
                    }
                    if rest == "engram.embed.weight" {
                        (names::engram_embd(l), one(Use::Rows, Origin::Made))
                    } else {
                        let (to, plain) = layer_tensor(l, rest)
                            .ok_or_else(|| refuse(name, "not a DeepSeek-V4.1 tensor"))?;
                        (to, one(Use::Matrix(plain), Origin::Made))
                    }
                }
            };
            if let Some(prev) = out.tensors.insert(to, m) {
                return Err(refuse(
                    &prev.sources[0],
                    "maps to a deepseek41 name another tensor has",
                ));
            }
        }
        for ((l, projection), by_id) in stacks {
            let (sources, dims) = safetensors::expert_stack(&by_id, self.experts)
                .map_err(|e| refuse(&format!("layer {l} {projection}"), e.to_string()))?;
            // w1 is the gate, w3 the up and w2 the down projection of an expert's SwiGLU.
            let to = match projection {
                "w1.weight" => names::ffn_gate_exps(l),
                "w3.weight" => names::ffn_up_exps(l),
                _ => names::ffn_down_exps(l),
            };
            out.tensors.insert(
                to,
                Mapped {
                    sources,
                    role: Use::Matrix(Plain::Q8_0),
                    steps: vec![],
                    dims: dims.into_iter().map(|d| d as u64).collect(),
                    origin: Origin::Made,
                },
            );
        }
        self.complete(&out.tensors)?;
        Ok(out)
    }

    fn mover(&self, _name: &str, steps: &[Infallible]) -> Option<crate::part::Step> {
        steps.first().map(|&never| match never {})
    }

    fn header(
        &self,
        _planned: &[(&str, &[u64])],
        folder: &Folder,
    ) -> Result<Vec<(String, Value)>, Error> {
        let tok = read_json(&folder.dir().join("tokenizer.json"))?;
        let tokens = tokens_by_id(&tok, self.vocab.size() as usize)?;
        let mut keys = super::trunk::header(ARCHITECTURE, &self.keys);
        keys.extend(self.hash_keys(&tokens.iter().map(String::as_str).collect::<Vec<_>>())?);
        Ok(keys)
    }

    fn vocab(&self) -> &VocabSpec {
        &self.vocab
    }
}

/// Each id's spelling below `size` in `tokenizer.json` `tok`: an added token's text, else the
/// vocabulary's string, else empty. A token of no text, or of an id that is not below `size`,
/// is refused by name.
fn tokens_by_id(tok: &Json, size: usize) -> Result<Vec<String>, Error> {
    let bad = |what: &str| Error::Refused(format!("tokenizer.json: {what}"));
    let mut out = vec![String::new(); size];
    let vocab = tok["model"]["vocab"]
        .as_object()
        .ok_or_else(|| bad("no model.vocab"))?;
    let added = tok["added_tokens"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default();
    let spelled = vocab
        .iter()
        .map(|(t, id)| (id, Some(t.as_str())))
        .chain(added.iter().map(|a| (&a["id"], a["content"].as_str())));
    for (id, text) in spelled {
        let text = text.ok_or_else(|| bad(&format!("added token {id} has no text")))?;
        let slot = id
            .as_u64()
            .and_then(|i| usize::try_from(i).ok())
            .and_then(|i| out.get_mut(i))
            .ok_or_else(|| bad(&format!("token {text:?}'s id {id} is not below {size}")))?;
        *slot = text.to_string();
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use gguf::write::Layout;
    use hf::quant_config::Config;
    use model::arch::deepseek41::hparams::{Collapse, Hparams};
    use model::arch::deepseek41::tp::{TpUnits, rank_span};

    use super::*;
    use crate::folder::{self, Make};
    use crate::test_support::{Entries, json};

    /// The 1017 folder's config and every tensor of its header ([`header_1017`]).
    fn folder_1017() -> (Json, Entries) {
        let config = json("quant-configs/deepseek-ai__DeepSeek-V4.1-Flash__config.json.gz");
        let experts = |name: &str| if name.starts_with("mtp.") { 128 } else { 384 };
        (config, header_1017(experts))
    }

    /// Every tensor of the 1017 header, each routed stack expanded from its expert 0
    /// (`deepseek-v4.1-flash-1017-expert-0.tensors.json.gz`) to `experts(name)` experts.
    pub(super) fn header_1017(experts: impl Fn(&str) -> usize) -> Entries {
        let header = json("checkpoint-headers/deepseek-v4.1-flash-1017-expert-0.tensors.json.gz");
        let mut entries = Entries::new();
        for (name, dtype, shape) in crate::test_support::entries(&header) {
            match name.split_once(".experts.0.") {
                Some((head, tail)) => entries.extend(
                    (0..experts(&name))
                        .map(|e| (format!("{head}.experts.{e}.{tail}"), dtype, shape.clone())),
                ),
                None => entries.push((name, dtype, shape)),
            }
        }
        entries
    }

    /// The header the map's plan writes, its hash keys from `tokens`.
    fn header(
        map: &Ds41Map,
        planned: &[folder::Planned<Infallible>],
        tokens: &[&str],
    ) -> gguf::Split {
        let mut kvs = super::super::trunk::header(ARCHITECTURE, &map.keys);
        kvs.extend(map.hash_keys(tokens).unwrap());
        let layout = Layout::new(&kvs, planned.iter().map(|p| p.decl.clone()).collect()).unwrap();
        gguf::Split::header_only(PathBuf::from("ds41.gguf"), &layout).unwrap()
    }

    /// The 1017 folder's 96,085 tensors plan from their headers alone: every trunk tensor in
    /// the deepseek41 names at its ggml dims, the dense projections MXFP8, the routed stacks
    /// MXFP4, the Engram tables MXFP8 row tables, the router BF16 and the vectors F32, the draft
    /// and the vision tower skipped by reason. Its header (the config's keys and a hash over a
    /// token list of the config's compressed size) reads as the deepseek41 hyperparameters:
    /// 40 layers, the lagged collapse, two Engram sites, the compressors' and indexers' layers;
    /// and the tensor-parallel cuts at four ranks give each rank its quarter. An Engram table
    /// one row short of `engram_num_embeddings` is refused by name.
    #[test]
    fn the_1017_folder_plans_and_its_header_reads_as_deepseek41() {
        let (mut config, entries) = folder_1017();
        assert_eq!(entries.len(), 96_085);
        let map = Ds41Map::parse(&config).unwrap();
        let planning = folder::plan_tensors(
            &map,
            &Config::parse(&config, None).unwrap(),
            &entries,
            false,
            folder::Keep::InPlace,
        )
        .unwrap();
        assert!(
            planning.plan.refused.is_empty(),
            "{:?}",
            planning.plan.refused
        );
        assert!(planning.skipped.keys().any(|n| n.starts_with("mtp.")));
        assert!(planning.skipped.keys().any(|n| n.starts_with("vision.")));
        assert!(planning.skipped.keys().all(|n| {
            ["mtp.", "vision.", "aligner.", "image_"]
                .iter()
                .any(|p| n.starts_with(p))
        }));
        let ty = |name: &str| {
            let p = planning
                .planned
                .iter()
                .find(|p| p.decl.name == name)
                .unwrap();
            (GgmlType::from_u32(p.decl.type_id), p.decl.dims.clone())
        };
        for (name, want, dims) in [
            ("blk.0.attn_q_b.weight", GgmlType::MXFP8, vec![1280, 32768]),
            (
                "blk.0.attn_output_a.weight",
                GgmlType::MXFP8,
                vec![4096, 8192],
            ),
            (
                "blk.3.ffn_up_exps.weight",
                GgmlType::MXFP4,
                vec![5120, 2304, 384],
            ),
            (
                "blk.3.ffn_down_exps.weight",
                GgmlType::MXFP4,
                vec![2304, 5120, 384],
            ),
            (
                "blk.1.engram_embd.weight",
                GgmlType::MXFP8,
                vec![256, 384_006_168],
            ),
            (
                "blk.14.engram_wkv.weight",
                GgmlType::MXFP8,
                vec![6144, 25600],
            ),
            ("blk.0.ffn_gate_inp.weight", GgmlType::BF16, vec![5120, 384]),
            (
                "blk.2.attn_compressor_kv.weight",
                GgmlType::BF16,
                vec![5120, 512],
            ),
            ("blk.0.hc_attn_fn.weight", GgmlType::F32, vec![20480, 24]),
            ("blk.0.attn_sinks.weight", GgmlType::F32, vec![64]),
            ("token_embd.weight", GgmlType::BF16, vec![5120, 129_280]),
            ("output.weight", GgmlType::BF16, vec![5120, 129_280]),
        ] {
            assert_eq!(ty(name), (want, dims), "{name}");
        }
        let Make::Tensor(m) = &planning
            .planned
            .iter()
            .find(|p| p.decl.name == "blk.1.engram_embd.weight")
            .unwrap()
            .make
        else {
            panic!("the Engram table is a tensor");
        };
        assert_eq!(m.role, Use::Rows);

        let tokens: Vec<String> = (0..129_280).map(|i| format!("t{}", i % 99_092)).collect();
        let tokens: Vec<&str> = tokens.iter().map(String::as_str).collect();
        let split = header(&map, &planning.planned, &tokens);
        let hp = Hparams::read(&split).unwrap();
        assert_eq!((hp.n_layer, hp.n_head, hp.o_groups), (40, 64, 8));
        assert_eq!(hp.collapse, Collapse::Lagged);
        assert_eq!(hp.engram.as_ref().unwrap().layer_ids, [1, 14]);
        assert_eq!(hp.rows.engram, Some(GgmlType::MXFP8));
        let compressors: Vec<usize> = (0..40)
            .filter(|&l| hp.layers[l].compressor.is_some())
            .collect();
        let indexers: Vec<usize> = (0..40).filter(|&l| hp.layers[l].indexer).collect();
        assert_eq!(compressors, [2, 8, 14, 20]);
        assert_eq!(indexers, [2, 8, 14, 20, 24, 28, 32, 36]);
        let hash = engram::hash::Hash::from_keys(&split).unwrap();
        assert_eq!((hash.sites(), hash.n_cols()), (2, 24));

        let u = TpUnits::of(&hp);
        let (q_b, dims) = ty("blk.0.attn_q_b.weight");
        assert_eq!(
            rank_span("blk.0.attn_q_b.weight", q_b, &dims, &u, 3, 4).unwrap(),
            Some((1, 24576..32768))
        );

        let mut short = entries.clone();
        for (name, _, shape) in &mut short {
            if name.starts_with("layers.1.engram.embed.") {
                shape[0] -= 1;
            }
        }
        let e = folder::plan_tensors(
            &map,
            &Config::parse(&config, None).unwrap(),
            &short,
            false,
            folder::Keep::InPlace,
        )
        .err()
        .unwrap()
        .to_string();
        assert!(
            e.contains("layers.1.engram.embed.weight") && e.contains("engram_num_embeddings"),
            "{e}"
        );

        config["text_config"]["kv_source_layer_ids"] = serde_json::json!([2, 8, 14]);
        let e = folder::plan_tensors(
            &Ds41Map::parse(&config).unwrap(),
            &Config::parse(&config, None).unwrap(),
            &entries,
            false,
            folder::Keep::InPlace,
        )
        .err()
        .unwrap()
        .to_string();
        assert!(e.contains("blk.20.attn_compressor_kv.weight"), "{e}");
    }

    /// A token list that compresses to another count than the config's is refused by name.
    #[test]
    fn a_tokenizer_of_another_compressed_vocabulary_is_refused() {
        let (config, _) = folder_1017();
        let map = Ds41Map::parse(&config).unwrap();
        let e = map.hash_keys(&["a", "b"]).unwrap_err().to_string();
        assert!(
            e.contains("compress to 2 ids") && e.contains("99092"),
            "{e}"
        );
    }

    /// An added token of no text, or a token whose id is not below the vocabulary size, is
    /// refused by name.
    #[test]
    fn a_token_of_no_text_or_past_the_vocabulary_is_refused() {
        let tok = |added: Json| serde_json::json!({"model": {"vocab": {"a": 0}}, "added_tokens": [added]});
        let ok = tokens_by_id(&tok(serde_json::json!({"id": 1, "content": "b"})), 2).unwrap();
        assert_eq!(ok, ["a", "b"]);
        for (added, want) in [
            (serde_json::json!({"id": 1}), "added token 1 has no text"),
            (
                serde_json::json!({"id": 2, "content": "b"}),
                "\"b\"'s id 2 is not below 2",
            ),
        ] {
            let e = tokens_by_id(&tok(added), 2).unwrap_err().to_string();
            assert!(e.contains(want), "{e}");
        }
    }
}
