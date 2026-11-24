//! ERNIE 5.5 Flash Preview's checkpoint folder (`ernie5_5_moe`) in the ernie55 names the program
//! reads ([`gguf::arch::ernie55`]). The model is a hybrid of two attention kinds, told apart by the
//! config's `compress_ratios`: a compressed-attention (HCA) layer, DeepSeek-V4's grouped output
//! with one key-value head and a pooling compressor, and a latent sparse-attention (DSA) layer,
//! multi-head latent attention with an indexer. Every layer carries an output gate, a head mix, two
//! hyper-connection mixers and, past the leading dense layers, a latent mixture of experts: two
//! sigmoid routers, a learned per-expert scale and a shared expert.
//!
//! Every tensor maps to one program tensor, but for four moves: a mixer's `mapping_proj` is stored
//! `[in, out]` and written `[out, in]` ([`Step::Transpose`]), a mixer's three scalars
//! (`alpha_pre`, `alpha_post`, `alpha_res`) are the three sources of its one `*_scale` vector of
//! three, and a layer's routed experts, one tensor an expert and projection, stack by expert id.
//! A DSA layer's fused `kv_b_proj` is kept whole as `attn_kv_b`: the per-head key and value halves
//! the card reads are cut from it by a later move.
//!
//! The formats are the quantization config's (`fp8` in 128 by 128 blocks, `mxfp4` experts): the
//! dense projections are F8_E4M3 beside an f32 scale a block (held as FP8 128-blocks), the routed
//! experts MXFP4 codes beside an E8M0 scale a 32 values, the compressor's and the indexer's
//! matrices BF16, the routers BF16 and every norm, mixer and vector F32. The DSpark draft
//! (`model.mtp.*`) and FlashIndexer's gate and bias are skipped with the reason: no reader takes
//! them yet.
//!
//! The header's keys are the config's, by the table `crates/model`'s ernie55 reader holds to
//! ([`Checkpoint::parse`]): every config key is read into a header key, fixed to the value the
//! program assumes (a folder of another value is refused by name) or ignored with its reason.

use std::collections::{BTreeMap, BTreeSet};

use gguf::arch::ernie55::ARCH as ARCHITECTURE;
use gguf::{GgmlType, Value};
use hf::modelopt::{Normal, Plain, Use};
use hf::quant_config::ModuleClass;
use hf::schema::{Fields, Rule::*, Type};
use safetensors::Folder;
use serde_json::Value as Json;

use super::section::Section;
use crate::folder::{Error, FolderMap, Mapped, Mapping, Origin, VocabSpec, refuse};

#[cfg(test)]
mod tests;

/// `expert_gating_func`'s sigmoid router, llama.cpp's `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`
/// (`llama-hparams.h`).
const SIGMOID: u32 = 2;

/// The draft's keys, which the draft's own reader takes.
const DSPARK: &str = "the DSpark draft's geometry; its model.mtp tensors are skipped";

/// What the config's `rope_scaling` holds: YaRN's.
const ROPE: &Fields = &[
    ("type", Need(r#"["yarn"]"#)),
    ("factor", Read),
    ("beta_fast", Read),
    ("beta_slow", Read),
    ("original_max_position_embeddings", Read),
];

/// `config.json`'s keys. A key the program fixes holds that value alone; a key it does not run is
/// ignored with the reason no result depends on it.
const ROOT: &Fields = &[
    ("architectures", Need(r#"[["Ernie5_5_MoeForCausalLM"]]"#)),
    ("model_type", Need(r#"["ernie5_5_moe"]"#)),
    ("quantization_config", Read),
    (
        "torch_dtype",
        Ignore(
            Type::String,
            "the dtype HF loads in; each tensor keeps its stored format",
        ),
    ),
    (
        "transformers_version",
        Ignore(Type::String, "the version that wrote the file"),
    ),
    // The tokenizer's ids and the vocabulary.
    ("vocab_size", Read),
    ("bos_token_id", Read),
    ("eos_token_id", Read),
    ("pad_token_id", Read),
    // The trunk.
    ("hidden_size", Read),
    ("intermediate_size", Read),
    ("num_hidden_layers", Read),
    ("max_position_embeddings", Read),
    ("rms_norm_eps", Read),
    ("first_k_dense_replace", Read),
    ("hidden_act", Need(r#"["situ"]"#)),
    ("gated_linear_unit", Need("[true]")),
    ("activation_situ_beta", Read),
    ("activation_situ_linear_beta", Read),
    ("tie_word_embeddings", Need("[false]")),
    ("multimax", Need(r#"["lm_head"]"#)),
    // The attention both kinds share: the heads, the latents, the output groups, the window.
    ("num_attention_heads", Read),
    ("num_key_value_heads", Need("[1]")),
    ("head_dim", Read),
    ("v_head_dim", Read),
    ("qk_nope_head_dim", Read),
    ("qk_rope_head_dim", Read),
    ("qk_pos_emb_head_dim", Read),
    ("q_lora_rank", Read),
    ("kv_lora_rank", Read),
    ("o_groups", Read),
    ("o_lora_rank", Read),
    ("sliding_window", Read),
    ("compress_ratios", Read),
    ("attention_bias", Need("[false]")),
    ("use_bias", Need("[false]")),
    ("use_mla", Need("[false]")),
    ("use_gated_attn", Need("[true]")),
    ("gated_attn_use_q_lora", Need("[true]")),
    ("gated_attn_act", Need(r#"["sigmoid"]"#)),
    ("use_vha_attention", Need("[true]")),
    ("use_qk_norm", Need("[true]")),
    ("add_full_attention_sink_bias", Need("[true]")),
    ("experimental_attention_variant", Need(r#"["dsv4_hybrid"]"#)),
    ("hybrid_mla_attention", Need(r#"["mqa_dsa"]"#)),
    ("mqa_split_kv_b_proj", Need("[true]")),
    // The latent-attention layers' own widths and the indexer.
    ("hybrid_mla_num_attention_heads", Read),
    ("hybrid_mla_num_key_value_heads", Read),
    ("hybrid_mla_q_lora_rank", Read),
    ("hybrid_mla_kv_lora_rank", Read),
    ("hybrid_mla_qk_nope_head_dim", Read),
    ("hybrid_mla_qk_rope_head_dim", Read),
    ("hybrid_mla_v_head_dim", Read),
    ("index_n_heads", Read),
    ("index_head_dim", Read),
    ("index_topk", Read),
    // The rope: the base, the compressed layers' base and YaRN, a plain rope on the rest.
    ("rope_theta", Read),
    ("rotary_base", Read),
    ("compress_rope_theta", Read),
    ("rope_scaling", Read),
    ("hca_rope_type", Need(r#"["yarn"]"#)),
    ("rope_type", Need(r#"["rope"]"#)),
    ("rope_interleave", Need("[true]")),
    ("rotary_percent", Need("[1.0]")),
    // The hyper-connections.
    ("enable_hyper_connections", Need("[true]")),
    ("hc_mult", Read),
    ("hc_sinkhorn_iters", Read),
    ("hc_eps", Read),
    // The latent mixture of experts.
    ("moe_intermediate_size", Read),
    ("moe_latent_size", Read),
    ("n_routed_experts", Read),
    ("n_shared_experts", Read),
    ("num_experts_per_tok", Read),
    ("moe_layer_freq", Need("[1]")),
    ("moe_shared_expert_gate", Need("[false]")),
    ("latent_moe_use_norm", Need("[true]")),
    ("moe_split_feature_routing", Need("[true]")),
    ("topk_method", Need(r#"["noaux_tc"]"#)),
    ("scoring_func", Need(r#"["sigmoid"]"#)),
    ("norm_topk_prob", Read),
    ("n_group", Read),
    ("topk_group", Read),
    ("routed_scaling_factor", Read),
    ("routed_scaling_factor_learnable", Need("[true]")),
    // The draft.
    ("num_nextn_predict_layers", Need("[0]")),
    ("dspark_block_size", Ignore(Type::Number, DSPARK)),
    ("dspark_noise_token_id", Ignore(Type::Number, DSPARK)),
    ("dspark_markov_rank", Ignore(Type::Number, DSPARK)),
    ("dspark_target_layer_ids", Read),
    ("mtp_shared_last_layer", Ignore(Type::Bool, DSPARK)),
    ("use_dense_mtp", Ignore(Type::Bool, DSPARK)),
    // What no result depends on: a training flag the reference implementations never read, a
    // serving knob of the framework, a kernel choice.
    (
        "use_attn_merger",
        Ignore(Type::Bool, "a training flag no reference reads"),
    ),
    (
        "qk_norm_type",
        Ignore(Type::String, "a training flag no reference reads"),
    ),
    (
        "csa_dense_mode",
        Ignore(Type::Bool, "a training flag no reference reads"),
    ),
    (
        "moe_group_experts",
        Ignore(Type::Bool, "a training flag no reference reads"),
    ),
    (
        "moe_token_dispatcher_type",
        Ignore(Type::String, "the framework's expert-parallel dispatch"),
    ),
    (
        "use_fast_hadamard",
        Ignore(Type::Bool, "a kernel choice of the framework"),
    ),
    (
        "use_cache",
        Ignore(
            Type::Bool,
            "whether HF keeps its cache; outputs are the same",
        ),
    ),
];

/// The attention kind of a layer: the config's `compress_ratios` entry says which.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Compressed attention: a positive pooling ratio.
    Hca,
    /// Latent sparse attention: the ratio `-2`.
    Dsa,
}

/// The move the map makes of a normalized tensor, one of this architecture's own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    /// An F32 matrix stored `[in, out]` rows, written as its `[out, in]` rows.
    Transpose,
}

impl Step {
    /// The move `self` makes of the normalized tensor `t`; a tensor that is not an F32 matrix is
    /// refused and left as it was.
    pub fn apply(&self, t: &mut Normal) -> Result<(), String> {
        match self {
            Step::Transpose => {
                let &[cols, rows] = t.dims.as_slice() else {
                    return Err(format!("Transpose reads a matrix, not dims {:?}", t.dims));
                };
                let (cols, rows) = (cols as usize, rows as usize);
                if t.type_id != GgmlType::F32.as_u32()
                    || cols.checked_mul(rows).and_then(|n| n.checked_mul(4)) != Some(t.bytes.len())
                {
                    return Err(format!(
                        "Transpose reads {rows} rows of {cols} F32 values, not type {} of {} bytes",
                        t.type_id,
                        t.bytes.len()
                    ));
                }
                let mut out = Vec::with_capacity(t.bytes.len());
                for j in 0..cols {
                    for i in 0..rows {
                        out.extend_from_slice(&t.bytes[(i * cols + j) * 4..][..4]);
                    }
                }
                t.bytes = out;
                t.dims = vec![rows as u64, cols as u64];
                Ok(())
            }
        }
    }
}

/// The stem, role and move of a stored layer tensor.
struct Layer {
    stem: &'static str,
    plain: Plain,
    step: Option<Step>,
}

/// The program tensor and role of the layer tensor `rest` (the name past `model.layers.L.`) of a
/// layer of `kind`, but a routed expert's and a mixer's scalars; `None` for no tensor of such a
/// layer.
fn layer_tensor(kind: Kind, rest: &str) -> Option<Layer> {
    use Kind::{Dsa, Hca};
    use Plain::{Bf16, F32, Q8_0};
    let (stem, plain, step) = match (kind, rest) {
        (_, "input_layernorm.weight") => ("attn_norm.weight", F32, None),
        (_, "post_attention_layernorm.weight") => ("ffn_norm.weight", F32, None),
        (_, "self_attn.gate_proj.weight") => ("attn_gate.weight", Q8_0, None),
        (_, "self_attn.vha_postmix_U") => ("attn_vha_u.weight", F32, None),
        (_, "self_attn.vha_postmix_V") => ("attn_vha_v.weight", F32, None),
        (_, "self_attention_hyper_connection.mapping_proj.weight") => {
            ("hc_attn_fn.weight", F32, Some(Step::Transpose))
        }
        (_, "self_attention_hyper_connection.bias") => ("hc_attn_base.weight", F32, None),
        (_, "mlp_hyper_connection.mapping_proj.weight") => {
            ("hc_ffn_fn.weight", F32, Some(Step::Transpose))
        }
        (_, "mlp_hyper_connection.bias") => ("hc_ffn_base.weight", F32, None),
        (Hca, "self_attn.linear_q_down_proj.weight") | (Dsa, "self_attn.q_a_proj.weight") => {
            ("attn_q_a.weight", Q8_0, None)
        }
        (Hca, "self_attn.q_layernorm.weight") | (Dsa, "self_attn.q_a_layernorm.weight") => {
            ("attn_q_a_norm.weight", F32, None)
        }
        (Hca, "self_attn.linear_q_up_proj.weight") | (Dsa, "self_attn.q_b_proj.weight") => {
            ("attn_q_b.weight", Q8_0, None)
        }
        (Hca, "self_attn.kv_layernorm.weight") | (Dsa, "self_attn.kv_a_layernorm.weight") => {
            ("attn_kv_a_norm.weight", F32, None)
        }
        (Hca, "self_attn.core_attention.attn_sink")
        | (Dsa, "self_attn.core_attention.softmax_offset") => ("attn_sinks.weight", F32, None),
        (Hca, "self_attn.linear_kv_proj.weight") => ("attn_kv.weight", Q8_0, None),
        (Hca, "self_attn.linear_o_group_proj") => ("attn_output_a.weight", Q8_0, None),
        (Hca, "self_attn.o_proj.weight") => ("attn_output_b.weight", Q8_0, None),
        (Hca, "self_attn.core_attention.compressor.linear_wkv.weight") => {
            ("attn_compressor_kv.weight", Q8_0, None)
        }
        (Hca, "self_attn.core_attention.compressor.linear_wgate.weight") => {
            ("attn_compressor_gate.weight", Q8_0, None)
        }
        (Hca, "self_attn.core_attention.compressor.ape") => {
            ("attn_compressor_ape.weight", F32, None)
        }
        (Hca, "self_attn.core_attention.compressor.norm.weight") => {
            ("attn_compressor_norm.weight", F32, None)
        }
        (Dsa, "self_attn.kv_a_proj_with_mqa.weight") => ("attn_kv_a_mqa.weight", Q8_0, None),
        (Dsa, "self_attn.kv_b_proj.weight") => ("attn_kv_b.weight", Q8_0, None),
        (Dsa, "self_attn.o_proj.weight") => ("attn_output.weight", Q8_0, None),
        (Dsa, "self_attn.core_attention.indexer.wq_b.weight") => {
            ("indexer.attn_q_b.weight", Q8_0, None)
        }
        (Dsa, "self_attn.core_attention.indexer.wk.weight") => {
            ("indexer.attn_k.weight", Q8_0, None)
        }
        (Dsa, "self_attn.core_attention.indexer.weights_proj.weight") => {
            ("indexer.proj.weight", Q8_0, None)
        }
        (Dsa, "self_attn.core_attention.indexer.k_norm.weight") => {
            ("indexer.k_norm.weight", F32, None)
        }
        (Dsa, "self_attn.core_attention.indexer.k_norm.bias") => ("indexer.k_norm.bias", F32, None),
        (_, "mlp.gate_proj.weight") => ("ffn_gate.weight", Q8_0, None),
        (_, "mlp.up_proj.weight") => ("ffn_up.weight", Q8_0, None),
        (_, "mlp.down_proj.weight") => ("ffn_down.weight", Q8_0, None),
        (_, "block_sparse_moe.gate.weight") => ("ffn_gate_inp.weight", Bf16, None),
        (_, "block_sparse_moe.gate.weight_1") => ("ffn_gate_inp_1.weight", Bf16, None),
        (_, "block_sparse_moe.e_score_correction_bias") => ("exp_probs_b.bias", F32, None),
        (_, "block_sparse_moe.gate.routed_scaling_factor_param") => {
            ("ffn_exp_scale.weight", F32, None)
        }
        (_, "block_sparse_moe.fc1_latent_proj.weight") => ("ffn_routed_down.weight", Q8_0, None),
        (_, "block_sparse_moe.fc2_latent_proj.weight") => ("ffn_routed_up.weight", Q8_0, None),
        (_, "block_sparse_moe.latent_norm.weight") => ("ffn_routed_norm.weight", F32, None),
        (_, "block_sparse_moe.shared_experts.w1.weight") => ("ffn_gate_shexp.weight", Q8_0, None),
        (_, "block_sparse_moe.shared_experts.w3.weight") => ("ffn_up_shexp.weight", Q8_0, None),
        (_, "block_sparse_moe.shared_experts.w2.weight") => ("ffn_down_shexp.weight", Q8_0, None),
        _ => return None,
    };
    Some(Layer { stem, plain, step })
}

/// The stem of the mixer scale a scalar of a layer is a source of, and its place among the three
/// (`alpha_pre`, `alpha_post`, `alpha_res`); `None` for another tensor.
fn mixer_scalar(rest: &str) -> Option<(&'static str, usize)> {
    let (block, alpha) = rest.split_once('.')?;
    let scale = match block {
        "self_attention_hyper_connection" => "hc_attn_scale.weight",
        "mlp_hyper_connection" => "hc_ffn_scale.weight",
        _ => return None,
    };
    let at = ["alpha_pre", "alpha_post", "alpha_res"]
        .iter()
        .position(|a| *a == alpha)?;
    Some((scale, at))
}

/// The model-level tensors: the stem, role and move of each stored name.
fn model_tensor(name: &str) -> Option<Layer> {
    use Plain::{Bf16, F32};
    let (stem, plain, step) = match name {
        "model.embed_tokens.weight" => ("token_embd.weight", Bf16, None),
        "lm_head.weight" => ("output.weight", Bf16, None),
        "model.norm.weight" => ("output_norm.weight", F32, None),
        "model.lm_head.multimax_ranges" => ("output_multimax_ranges.weight", F32, None),
        "model.lm_head.multimax_ts" => ("output_multimax_ts.weight", F32, None),
        "model.mhc_contract.hc_head_fn" => ("output_hc_fn.weight", F32, Some(Step::Transpose)),
        "model.mhc_contract.hc_head_base" => ("output_hc_base.weight", F32, None),
        "model.mhc_contract.hc_head_scale" => ("output_hc_scale.weight", F32, None),
        _ => return None,
    };
    Some(Layer { stem, plain, step })
}

/// The stems every layer carries, whatever its kind.
const EVERY_LAYER: [&str; 16] = [
    "attn_norm.weight",
    "ffn_norm.weight",
    "attn_gate.weight",
    "attn_vha_u.weight",
    "attn_vha_v.weight",
    "attn_q_a.weight",
    "attn_q_a_norm.weight",
    "attn_q_b.weight",
    "attn_kv_a_norm.weight",
    "attn_sinks.weight",
    "hc_attn_fn.weight",
    "hc_attn_base.weight",
    "hc_attn_scale.weight",
    "hc_ffn_fn.weight",
    "hc_ffn_base.weight",
    "hc_ffn_scale.weight",
];

/// The stems a compressed-attention layer adds.
const HCA_ONLY: [&str; 7] = [
    "attn_kv.weight",
    "attn_output_a.weight",
    "attn_output_b.weight",
    "attn_compressor_kv.weight",
    "attn_compressor_gate.weight",
    "attn_compressor_ape.weight",
    "attn_compressor_norm.weight",
];

/// The stems a latent sparse-attention layer adds.
const DSA_ONLY: [&str; 8] = [
    "attn_kv_a_mqa.weight",
    "attn_kv_b.weight",
    "attn_output.weight",
    "indexer.attn_q_b.weight",
    "indexer.attn_k.weight",
    "indexer.proj.weight",
    "indexer.k_norm.weight",
    "indexer.k_norm.bias",
];

/// The stems a leading dense layer's feed-forward block holds.
const DENSE: [&str; 3] = ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"];

/// The stems a routed layer's feed-forward block holds.
const ROUTED: [&str; 13] = [
    "ffn_gate_inp.weight",
    "ffn_gate_inp_1.weight",
    "exp_probs_b.bias",
    "ffn_exp_scale.weight",
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
    "ffn_routed_down.weight",
    "ffn_routed_up.weight",
    "ffn_routed_norm.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
    "ffn_down_shexp.weight",
];

/// The folder tensors no ernie55 tensor is made of, and why; `None` for one the map reads.
fn skip_reason(name: &str) -> Option<&'static str> {
    if name.starts_with("model.mtp.") {
        Some("the DSpark draft's blocks and heads, which no reader takes yet")
    } else if name.ends_with(".indexer.flashindexer_bias")
        || name.ends_with(".indexer.linear_gate.weight")
    {
        Some("FlashIndexer's gate and bias, which no reader takes yet")
    } else {
        None
    }
}

/// The layer `L` and the rest of a name `model.layers.L.rest`.
fn split_layer(name: &str) -> Option<(usize, &str)> {
    let (l, rest) = name.strip_prefix("model.layers.")?.split_once('.')?;
    Some((l.parse().ok()?, rest))
}

/// `blk.{l}.{stem}`.
fn blk(l: usize, stem: &str) -> String {
    format!("blk.{l}.{stem}")
}

/// `shape` reversed: ggml's dims of a row-major tensor.
fn ggml_dims(shape: &[usize]) -> Vec<u64> {
    shape.iter().rev().map(|&d| d as u64).collect()
}

/// A routed projection's experts by id: each one's folder name and shape.
type Experts<'a> = BTreeMap<usize, (&'a str, &'a [usize])>;

/// The ERNIE 5.5 folder map.
#[derive(Debug)]
pub struct Checkpoint {
    layers: usize,
    experts: usize,
    hidden: u64,
    heads: u64,
    dense_layers: usize,
    kinds: Vec<Kind>,
    vocab: VocabSpec,
    /// The header's model keys by suffix, under the architecture's name.
    keys: Vec<(String, Value)>,
}

impl Checkpoint {
    /// The stems layer `l` holds.
    fn expected(&self, l: usize) -> Vec<&'static str> {
        let mut stems: Vec<&'static str> = EVERY_LAYER.to_vec();
        stems.extend(match self.kinds[l] {
            Kind::Hca => &HCA_ONLY[..],
            Kind::Dsa => &DSA_ONLY[..],
        });
        stems.extend(if l < self.dense_layers {
            &DENSE[..]
        } else {
            &ROUTED[..]
        });
        stems
    }

    /// Every tensor the config says the folder holds is mapped, and no tensor the config says it
    /// does not: each layer holds the stems of its kind and block ([`Checkpoint::expected`]), and
    /// the embedding and the head are `[hidden, vocab]`.
    fn complete(&self, mapped: &BTreeMap<String, Mapped<Step>>) -> Result<(), Error> {
        for l in 0..self.layers {
            let want: BTreeSet<&str> = self.expected(l).into_iter().collect();
            if let Some(stem) = want.iter().find(|s| !mapped.contains_key(&blk(l, s))) {
                return Err(refuse(&blk(l, stem), "is missing from the folder"));
            }
        }
        for name in mapped.keys() {
            let Some((l, stem)) = name
                .strip_prefix("blk.")
                .and_then(|r| r.split_once('.'))
                .and_then(|(l, stem)| Some((l.parse::<usize>().ok()?, stem)))
            else {
                continue;
            };
            if l >= self.layers || !self.expected(l).contains(&stem) {
                return Err(refuse(
                    name,
                    format!(
                        "is a tensor of no {:?} layer {l} of {}",
                        self.kinds.get(l),
                        self.layers
                    ),
                ));
            }
        }
        let vocab = u64::from(self.vocab.size());
        for stem in ["token_embd.weight", "output.weight"] {
            let m = mapped
                .get(stem)
                .ok_or_else(|| refuse(stem, "is missing from the folder"))?;
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

    /// The rank of the head mix, which the `U` and `V` tensors state: `[heads, rank]` rows in
    /// every layer, equal across them.
    fn vha_rank(&self, planned: &[(&str, &[u64])]) -> Result<u32, Error> {
        let heads = self.heads;
        let mut rank = None;
        for &(name, dims) in planned {
            if !(name.ends_with(".attn_vha_u.weight") || name.ends_with(".attn_vha_v.weight")) {
                continue;
            }
            let &[r, h] = dims else {
                return Err(refuse(name, format!("dims {dims:?}, not [rank, heads]")));
            };
            if h != heads || rank.is_some_and(|first| first != r) {
                return Err(refuse(
                    name,
                    format!("dims {dims:?}, not [{}, {heads}]", rank.unwrap_or(r)),
                ));
            }
            rank = Some(r);
        }
        let rank =
            rank.ok_or_else(|| refuse("attn_vha_u.weight", "no layer holds the head mix"))?;
        u32::try_from(rank).map_err(|_| refuse("attn_vha_u.weight", "the rank passes u32"))
    }
}

/// The two config keys that name one header key hold one value.
fn same(
    s: &Section<'_, Error>,
    first: &str,
    second: &str,
    n: impl Fn(&str) -> Result<u32, Error>,
) -> Result<u32, Error> {
    let (a, b) = (n(first)?, n(second)?);
    if a != b {
        return Err(s.bad(second, &format!("is {b}, where {first} is {a}")));
    }
    Ok(a)
}

impl FolderMap for Checkpoint {
    type Step = Step;

    const MODEL_TYPES: &'static [&'static str] = &["ernie5_5_moe"];

    fn parse(config: &Json) -> Result<Self, Error> {
        let s = Section::<Error>::new(config, "config.json")?;
        s.check(&[ROOT])?;
        let rope = Section::<Error>::new(s.field("rope_scaling")?, "rope_scaling")?;
        rope.check(&[ROPE])?;
        let n = |key: &str| s.count(key);
        let f = |key: &str| s.number(key);
        let layers = n("num_hidden_layers")? as usize;
        let heads = same(
            &s,
            "num_attention_heads",
            "hybrid_mla_num_attention_heads",
            n,
        )?;
        let rope_dim = same(&s, "qk_rope_head_dim", "hybrid_mla_qk_rope_head_dim", n)?;
        same(&s, "qk_rope_head_dim", "qk_pos_emb_head_dim", n)?;
        let head_dim = same(&s, "head_dim", "v_head_dim", n)?;
        let nope = n("qk_nope_head_dim")?;
        if nope + rope_dim != head_dim {
            return Err(s.bad(
                "qk_nope_head_dim",
                &format!("{nope} and the rotated {rope_dim} are not head_dim {head_dim}"),
            ));
        }
        let kv_rank = same(&s, "kv_lora_rank", "hybrid_mla_kv_lora_rank", n)?;
        let q_rank = same(&s, "q_lora_rank", "hybrid_mla_q_lora_rank", n)?;
        if n("hybrid_mla_num_key_value_heads")? != heads {
            return Err(s.bad(
                "hybrid_mla_num_key_value_heads",
                &format!("is not hybrid_mla_num_attention_heads {heads}"),
            ));
        }
        let (theta, base) = (f("rope_theta")?, f("rotary_base")?);
        if theta != base {
            return Err(s.bad(
                "rotary_base",
                &format!("is {base}, where rope_theta is {theta}"),
            ));
        }
        let groups = n("o_groups")?;
        if !heads.is_multiple_of(groups) {
            return Err(s.bad("o_groups", "must divide num_attention_heads"));
        }
        let (experts, used) = (n("n_routed_experts")?, n("num_experts_per_tok")?);
        if used > experts {
            return Err(s.bad("num_experts_per_tok", "exceeds n_routed_experts"));
        }
        let dense_layers = n("first_k_dense_replace")? as usize;
        if dense_layers > layers {
            return Err(s.bad("first_k_dense_replace", "exceeds num_hidden_layers"));
        }
        let ratios = s.array("compress_ratios")?;
        if ratios.len() != layers {
            return Err(s.bad(
                "compress_ratios",
                &format!("names {} ratios for {layers} layers", ratios.len()),
            ));
        }
        let mut kinds = Vec::with_capacity(layers);
        let mut written = Vec::with_capacity(layers);
        for (l, r) in ratios.iter().enumerate() {
            match r.as_i64().map(|ratio| (ratio, u32::try_from(ratio))) {
                Some((-2, _)) => {
                    kinds.push(Kind::Dsa);
                    written.push(Value::U32(0));
                }
                Some((_, Ok(ratio))) if ratio > 0 => {
                    kinds.push(Kind::Hca);
                    written.push(Value::U32(ratio));
                }
                _ => {
                    return Err(s.bad(
                        "compress_ratios",
                        &format!(
                            "entry {l} is {r}, neither a pooling ratio nor -2 (latent attention)"
                        ),
                    ));
                }
            }
        }
        let targets = s.array("dspark_target_layer_ids")?;
        if targets.is_empty()
            || targets
                .iter()
                .any(|v| v.as_u64().is_none_or(|l| l as usize >= layers))
        {
            return Err(s.bad(
                "dspark_target_layer_ids",
                "is not a list of layer indices below num_hidden_layers",
            ));
        }
        let norm_topk = s
            .field("norm_topk_prob")?
            .as_bool()
            .ok_or_else(|| s.bad("norm_topk_prob", "is not a bool"))?;
        let u = |key: &str, from: u32| (key.to_string(), Value::U32(from));
        let fl = |key: &str, v: f32| (key.to_string(), Value::F32(v));
        let keys = vec![
            u("block_count", n("num_hidden_layers")?),
            u("context_length", n("max_position_embeddings")?),
            u("embedding_length", n("hidden_size")?),
            u("vocab_size", n("vocab_size")?),
            u("feed_forward_length", n("intermediate_size")?),
            u("leading_dense_block_count", dense_layers as u32),
            u("attention.head_count", heads),
            fl("attention.layer_norm_rms_epsilon", f("rms_norm_eps")?),
            (
                "attention.compress_ratios".to_string(),
                Value::Array(written),
            ),
            u("attention.sliding_window", n("sliding_window")?),
            u("attention.q_lora_rank", q_rank),
            u("attention.kv_lora_rank", kv_rank),
            u("attention.key_length", kv_rank + rope_dim),
            u(
                "attention.key_length_mla",
                n("hybrid_mla_qk_nope_head_dim")? + rope_dim,
            ),
            u("attention.value_length_mla", n("hybrid_mla_v_head_dim")?),
            u("rope.dimension_count", rope_dim),
            u("attention.hca.head_length", head_dim),
            u("attention.output_group_count", groups),
            u("attention.output_lora_rank", n("o_lora_rank")?),
            u("attention.indexer.head_count", n("index_n_heads")?),
            u("attention.indexer.key_length", n("index_head_dim")?),
            u("attention.indexer.top_k", n("index_topk")?),
            fl("rope.freq_base", theta),
            fl(
                "attention.compress_rope_freq_base",
                f("compress_rope_theta")?,
            ),
            (
                "rope.scaling.type".to_string(),
                Value::String("yarn".into()),
            ),
            fl("rope.scaling.factor", rope.number("factor")?),
            u(
                "rope.scaling.original_context_length",
                rope.count("original_max_position_embeddings")?,
            ),
            fl("rope.scaling.yarn_beta_fast", rope.number("beta_fast")?),
            fl("rope.scaling.yarn_beta_slow", rope.number("beta_slow")?),
            u("expert_count", experts),
            u("expert_used_count", used),
            u("expert_feed_forward_length", n("moe_intermediate_size")?),
            u("expert_shared_count", n("n_shared_experts")?),
            u("expert_latent_length", n("moe_latent_size")?),
            fl("expert_weights_scale", f("routed_scaling_factor")?),
            ("expert_weights_norm".to_string(), Value::Bool(norm_topk)),
            u("expert_gating_func", SIGMOID),
            u("expert_group_count", n("n_group")?),
            u("expert_group_used_count", n("topk_group")?),
            fl("activation.situ_beta", f("activation_situ_beta")?),
            fl(
                "activation.situ_linear_beta",
                f("activation_situ_linear_beta")?,
            ),
            u("hyper_connection.count", n("hc_mult")?),
            u(
                "hyper_connection.sinkhorn_iterations",
                n("hc_sinkhorn_iters")?,
            ),
            fl("hyper_connection.epsilon", f("hc_eps")?),
        ];
        Ok(Checkpoint {
            layers,
            experts: experts as usize,
            hidden: u64::from(n("hidden_size")?),
            heads: u64::from(heads),
            dense_layers,
            kinds,
            vocab: hf::vocab::Config::parse(config)?,
            keys,
        })
    }

    fn architecture(&self) -> &'static str {
        ARCHITECTURE
    }

    fn module_class(&self, module: &str) -> Option<ModuleClass> {
        if module == "model.embed_tokens" {
            return Some(ModuleClass::Embedding);
        }
        let (_, rest) = module.strip_prefix("model.layers.")?.split_once('.')?;
        let linear = rest.contains("_proj")
            || rest.starts_with("block_sparse_moe.experts.")
            || rest.starts_with("block_sparse_moe.shared_experts.")
            || rest.ends_with("indexer.wq_b")
            || rest == "self_attn.linear_o_group_proj";
        linear.then_some(ModuleClass::Linear)
    }

    fn map<'a>(&self, tensors: &[(&'a str, &'a [usize])]) -> Result<Mapping<Step>, Error> {
        let mut out = Mapping {
            tensors: BTreeMap::new(),
            skipped: BTreeMap::new(),
        };
        let mut stacks: BTreeMap<(usize, &str), Experts<'_>> = BTreeMap::new();
        let mut scalars: BTreeMap<(usize, &str), [Option<&str>; 3]> = BTreeMap::new();
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
            let (to, found) = if let Some(found) = model_tensor(name) {
                (found.stem.to_string(), found)
            } else {
                let (l, rest) =
                    split_layer(name).ok_or_else(|| refuse(name, "not an ERNIE 5.5 tensor"))?;
                let kind = *self
                    .kinds
                    .get(l)
                    .ok_or_else(|| refuse(name, format!("layer {l} of {}", self.layers)))?;
                if let Some((id, projection)) = rest
                    .strip_prefix("block_sparse_moe.experts.")
                    .and_then(|e| e.split_once('.'))
                    .and_then(|(id, p)| Some((id.parse::<usize>().ok()?, p)))
                {
                    if !matches!(projection, "w1.weight" | "w2.weight" | "w3.weight") {
                        return Err(refuse(name, "not an ERNIE 5.5 expert projection"));
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
                if let Some((scale, at)) = mixer_scalar(rest) {
                    let slot = &mut scalars.entry((l, scale)).or_default()[at];
                    if slot.replace(name).is_some() || shape != [1] {
                        return Err(refuse(name, format!("shape {shape:?}, not one scalar")));
                    }
                    continue;
                }
                let found = layer_tensor(kind, rest).ok_or_else(|| {
                    refuse(name, format!("not an ERNIE 5.5 tensor of a {kind:?} layer"))
                })?;
                (blk(l, found.stem), found)
            };
            // The step's output dims: a transpose turns the stored rows into the columns.
            let dims = match found.step {
                Some(Step::Transpose) => shape.iter().map(|&d| d as u64).collect(),
                None => ggml_dims(shape),
            };
            let origin = match (found.stem, found.plain) {
                ("token_embd.weight" | "output.weight", _) => Origin::File(GgmlType::BF16),
                _ => Origin::Made,
            };
            let m = Mapped {
                sources: vec![name.to_string()],
                role: Use::Matrix(found.plain),
                steps: found.step.into_iter().collect(),
                dims,
                origin,
            };
            if let Some(prev) = out.tensors.insert(to, m) {
                return Err(refuse(
                    &prev.sources[0],
                    "maps to an ernie55 name another tensor has",
                ));
            }
        }
        for ((l, scale), parts) in scalars {
            let sources: Vec<String> = parts
                .iter()
                .zip(["alpha_pre", "alpha_post", "alpha_res"])
                .map(|(p, alpha)| {
                    p.map(str::to_string).ok_or_else(|| {
                        refuse(
                            &format!("model.layers.{l}.{scale} {alpha}"),
                            "is missing from the folder",
                        )
                    })
                })
                .collect::<Result<_, _>>()?;
            out.tensors.insert(
                blk(l, scale),
                Mapped {
                    sources,
                    role: Use::Matrix(Plain::F32),
                    steps: vec![],
                    dims: vec![3],
                    origin: Origin::Made,
                },
            );
        }
        for ((l, projection), by_id) in stacks {
            let (sources, dims) = safetensors::expert_stack(&by_id, self.experts)
                .map_err(|e| refuse(&format!("layer {l} {projection}"), e.to_string()))?;
            // w1 is the gate, w3 the up and w2 the down projection of an expert.
            let stem = match projection {
                "w1.weight" => "ffn_gate_exps.weight",
                "w3.weight" => "ffn_up_exps.weight",
                _ => "ffn_down_exps.weight",
            };
            out.tensors.insert(
                blk(l, stem),
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

    fn mover(&self, name: &str, steps: &[Step]) -> Option<crate::part::Step> {
        (!steps.is_empty()).then(|| {
            let (name, steps) = (name.to_string(), steps.to_vec());
            Box::new(move |t: &mut Normal| {
                steps
                    .iter()
                    .try_for_each(|s| s.apply(t).map_err(|why| format!("{name}: {why}")))
            }) as crate::part::Step
        })
    }

    fn header(
        &self,
        planned: &[(&str, &[u64])],
        _folder: &Folder,
    ) -> Result<Vec<(String, Value)>, Error> {
        let mut fields = self.keys.clone();
        fields.push((
            "attention.vha.rank".to_string(),
            Value::U32(self.vha_rank(planned)?),
        ));
        Ok(super::trunk::header(ARCHITECTURE, &fields))
    }

    fn vocab(&self) -> &VocabSpec {
        &self.vocab
    }
}
