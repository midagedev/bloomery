//! K-EXAONE-2.0's checkpoint folder (`exaone_moe`, LG AI Research's names) in the exaone-moe
//! names, layouts and keys the program reads, as llama.cpp spells them (`models/exaone-moe.cpp`,
//! the keys of `src/llama-arch.cpp`'s map). Every tensor maps to one exaone-moe tensor with no
//! move: the folder's row-major shapes reversed are ggml's dims. A routed block's experts, one
//! `mlp.experts.E.{gate,up,down}_proj` tensor an expert, stack by expert id; the MTP layer
//! (`mtp.*`) is the header's last block (`blk.<layers>.`, `block_count` = layers + nextn), its
//! feed-forward block dense as llama.cpp loads it (a dense block for the leading
//! `leading_dense_block_count` blocks and the nextn ones).
//!
//! The stored format is the quantization config's (`fp8`, `weight_block_size` [128, 128], the
//! `weight_scale_inv` grid a block): the attention, dense, shared-expert and routed-expert
//! projections are F8_E4M3, the embedding, the head, the norms and the MTP layer BF16 (the
//! config's `ignored_layers`), the router's gate and its correction bias F32. The engine's dense
//! sites read the trunk as Q8_0, so an FP8 projection is decoded exactly and quantized again
//! (`Use::Q8_0`, the Qwen3.5 HF family's rule for the same body); the routed stacks keep the
//! checkpoint's FP8_B128 blocks for the quantizer to read (`Use::Matrix`), which the card loads
//! as planes. The MTP block is the draft: `prepare` keeps its tensors in their stored form, BF16,
//! as it keeps the head's and the embedding's. Norms, the router and its bias are read as F32.
//!
//! The header's keys are the config's ([`Checkpoint::parse`]): every per-layer key is an array of
//! `block_count` entries — `attention.sliding_window` (0 = full attention), the rope and the
//! window patterns as bools, the clamp limits as F32 — whose last entries are the MTP layer's.
//! `leading_dense_block_count` comes from `mlp_layer_types` (the config names no
//! `first_k_dense_replace` of its own): the dense blocks are its leading run. A field the map
//! does not take is refused by name.
//!
//! The tokenizer is the common path's (`hf::vocab`): a GPT2 BPE of the `exaone-moe`
//! pre-tokenizer (the row `llama-vocab.cpp` reads by that name), a row the vocab table does not
//! hold yet. The map pins no `pre` of its own, so the row the folder's own `tokenizer.json`
//! spells is the one named once it exists; until then the vocab reader refuses a folder of this
//! vocabulary by name rather than read it under another row's pipeline.

use std::collections::BTreeMap;
use std::convert::Infallible;

use gguf::Value;
use gguf::arch::exaone_moe::ARCH as ARCHITECTURE;
use hf::modelopt::{Plain, Use};
use hf::quant_config::ModuleClass;
use hf::schema::{Fields, Rule::*, Type};
use model::arch::glm5next::names as nextn;
use model::arch::names as common;
use safetensors::{Dtype, Folder};
use serde_json::Value as Json;

use super::section::Section;
use super::trunk;
use crate::folder::{Error, FolderMap, Mapped, Mapping, Origin, Stored, VocabSpec, refuse};

#[cfg(test)]
mod tests;

/// `expert_gating_func`'s sigmoid router, llama.cpp's `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`
/// (`llama-hparams.h`).
const SIGMOID: u32 = 2;

/// One stack's routed experts by id: each stored name and its shape.
type Stack<'a> = BTreeMap<usize, (&'a str, &'a [usize])>;

/// `config.json`'s top-level keys.
const ROOT: &Fields = &[
    ("architectures", Need(r#"[["ExaoneMoeForCausalLM"]]"#)),
    ("model_type", Need(r#"["exaone_moe"]"#)),
    ("quantization_config", Read),
    // The tokenizer's ids, which `hf::vocab::Config` reads.
    ("bos_token_id", Read),
    ("eos_token_id", Read),
    ("pad_token_id", Read),
    ("head_dim", Read),
    ("hidden_act", Only(r#"["silu"]"#)),
    ("hidden_size", Read),
    ("intermediate_size", Read),
    // Which blocks attend fully and which through a window, and which hold a dense feed-forward
    // block: the header's per-layer arrays and the dense lead.
    ("layer_types", Read),
    ("mlp_layer_types", Read),
    ("max_position_embeddings", Read),
    ("moe_intermediate_size", Read),
    // The MTP block: its attention kind and window make the per-layer arrays' last entries.
    ("mtp_layer_types", Read),
    ("mtp_sliding_windows", Read),
    ("num_nextn_predict_layers", Read),
    // The router: sigmoid scores, the chosen experts' weights renormalized and scaled, one
    // vacuous group.
    ("scoring_func", Need(r#"["sigmoid"]"#)),
    ("topk_method", Need(r#"["noaux_tc"]"#)),
    ("norm_topk_prob", Read),
    ("routed_scaling_factor", Read),
    ("n_group", Read),
    ("topk_group", Read),
    ("num_experts", Read),
    ("num_experts_per_tok", Read),
    ("num_shared_experts", Read),
    ("num_attention_heads", Read),
    ("num_key_value_heads", Read),
    ("num_hidden_layers", Read),
    ("rms_norm_eps", Read),
    // Each block's SwiGLU clamp limit, 0 where it clamps nothing.
    ("swiglu_limits", Read),
    ("sliding_windows", Read),
    ("rope_parameters", Read),
    ("vocab_size", Read),
    trunk::HEAD_OWN,
    trunk::DTYPE,
    (
        "attention_dropout",
        Ignore(Type::Number, "dropout acts in training only"),
    ),
    (
        "initializer_range",
        Ignore(Type::Number, "initializes weights the checkpoint holds"),
    ),
    (
        "mtp_loss_scaling_factor",
        Ignore(Type::Number, "weighs a training loss"),
    ),
    (
        "mtp_num_speculative_steps",
        Ignore(
            Type::Number,
            "the serving stack's own draft depth; the engine picks its own",
        ),
    ),
    (
        "tokenizer_class",
        Ignore(
            Type::String,
            "the tokenizer's files name their own class; the vocabulary is read from them",
        ),
    ),
    (
        "transformers_version",
        Ignore(Type::String, "the version that wrote the file"),
    ),
    (
        "use_cache",
        Ignore(
            Type::Bool,
            "whether HF keeps its cache; outputs are the same",
        ),
    ),
];

/// `rope_parameters`' keys: an unscaled rope of one base.
const ROPE: &Fields = &[("rope_theta", Read), ("rope_type", Only(r#"["default"]"#))];

/// The config values the map reads, checked, and the header keys they make.
#[derive(Debug)]
pub struct Checkpoint {
    /// The blocks the header names: the layers and the MTP block.
    blocks: usize,
    layers: usize,
    experts: usize,
    hidden: u64,
    query_rows: u64,
    key_rows: u64,
    head_dim: u64,
    expert_ff: u64,
    /// The shared experts' rows together: `num_shared_experts` of the expert width each.
    shared_ff: u64,
    dense_ff: u64,
    vocab_size: u64,
    /// The blocks before the first routed one.
    dense_lead: usize,
    vocab: VocabSpec,
    /// The header's model keys by suffix, under the architecture's name.
    keys: Vec<(String, Value)>,
}

/// A bool of `s`.
fn flag(s: &Section<'_, Error>, key: &str) -> Result<bool, Error> {
    s.field(key)?
        .as_bool()
        .ok_or_else(|| s.bad(key, "is not a bool"))
}

impl Checkpoint {
    fn read(config: &Json) -> Result<Checkpoint, Error> {
        let root = Section::<Error>::new(config, "config.json")?;
        root.check(&[ROOT])?;
        let rope = Section::<Error>::new(root.field("rope_parameters")?, "rope_parameters")?;
        rope.check(&[ROPE])?;
        let n = |key: &str| root.count(key);
        let f = |key: &str| root.number(key);
        let layers = n("num_hidden_layers")? as usize;
        let nextn = n("num_nextn_predict_layers")? as usize;
        let experts = n("num_experts")? as usize;
        if n("num_experts_per_tok")? > n("num_experts")? {
            return Err(root.bad("num_experts_per_tok", "exceeds num_experts"));
        }
        if !n("num_attention_heads")?.is_multiple_of(n("num_key_value_heads")?) {
            return Err(root.bad("num_attention_heads", "must group by num_key_value_heads"));
        }

        // The attention kinds and the windows say the same thing block by block, and the MTP
        // block's own lists name one windowed block each.
        let kinds = root.array("layer_types")?;
        let windows = root.array("sliding_windows")?;
        if kinds.len() != layers || windows.len() != layers {
            return Err(root.bad(
                "layer_types",
                &format!(
                    "and sliding_windows hold {} and {} entries, where num_hidden_layers is \
                     {layers}",
                    kinds.len(),
                    windows.len()
                ),
            ));
        }
        let mut window: Vec<u32> = Vec::with_capacity(layers + nextn);
        for (l, (kind, w)) in kinds.iter().zip(windows).enumerate() {
            let w = w
                .as_u64()
                .and_then(|w| u32::try_from(w).ok())
                .ok_or_else(|| root.bad("sliding_windows", "is not a list of u32 windows"))?;
            match (kind.as_str(), w > 0) {
                (Some("full_attention"), false) | (Some("sliding_attention"), true) => {}
                _ => {
                    return Err(root.bad(
                        "layer_types",
                        &format!("entry {l} is {kind:?}, where sliding_windows entry {l} is {w}"),
                    ));
                }
            }
            window.push(w);
        }
        let mtp_kinds = root.array("mtp_layer_types")?;
        let mtp_windows = root.array("mtp_sliding_windows")?;
        if mtp_kinds.len() != nextn || mtp_windows.len() != nextn {
            return Err(root.bad(
                "mtp_layer_types",
                &format!(
                    "and mtp_sliding_windows hold {} and {} entries, where \
                     num_nextn_predict_layers is {nextn}",
                    mtp_kinds.len(),
                    mtp_windows.len()
                ),
            ));
        }
        for (kind, w) in mtp_kinds.iter().zip(mtp_windows) {
            if kind.as_str() != Some("sliding_attention") {
                return Err(root.bad(
                    "mtp_layer_types",
                    "holds a block that is not sliding_attention",
                ));
            }
            let w = w
                .as_u64()
                .and_then(|w| u32::try_from(w).ok())
                .filter(|&w| w > 0)
                .ok_or_else(|| root.bad("mtp_sliding_windows", "is not a list of windows"))?;
            window.push(w);
        }

        // The dense blocks are the leading run of `mlp_layer_types`'s dense entries; the config
        // names no `first_k_dense_replace` of its own.
        let mlp = root.array("mlp_layer_types")?;
        if mlp.len() != layers {
            return Err(root.bad(
                "mlp_layer_types",
                &format!(
                    "holds {} entries, where num_hidden_layers is {layers}",
                    mlp.len()
                ),
            ));
        }
        let mut dense_lead = 0;
        for (l, kind) in mlp.iter().enumerate() {
            match kind.as_str() {
                Some("dense") if l == dense_lead => dense_lead += 1,
                Some("sparse") => {}
                _ => {
                    return Err(root.bad(
                        "mlp_layer_types",
                        &format!(
                            "entry {l} is {kind:?}, not a dense block of the leading run or a \
                             sparse one after it"
                        ),
                    ));
                }
            }
        }
        if dense_lead == layers {
            return Err(root.bad("mlp_layer_types", "holds no routed block"));
        }

        // The clamp limits: none below zero, one a block, none on the MTP block.
        let limits = root.array("swiglu_limits")?;
        if limits.len() != layers {
            return Err(root.bad(
                "swiglu_limits",
                &format!(
                    "holds {} entries, where num_hidden_layers is {layers}",
                    limits.len()
                ),
            ));
        }
        let mut clamp: Vec<f32> = Vec::with_capacity(layers + nextn);
        for (l, v) in limits.iter().enumerate() {
            let limit = v
                .as_f64()
                .map(|v| v as f32)
                .filter(|v| v.is_finite() && *v >= 0.0)
                .ok_or_else(|| {
                    root.bad(
                        "swiglu_limits",
                        &format!("entry {l} is not a finite limit of at least zero"),
                    )
                })?;
            clamp.push(limit);
        }
        clamp.extend(std::iter::repeat_n(0.0, nextn));

        let head_dim = u64::from(n("head_dim")?);
        let expert_ff = u64::from(n("moe_intermediate_size")?);
        // Every per-layer key is an array of one entry a block, the MTP block's last; the
        // patterns say which blocks run windowed, which is where the rope turns on (every full
        // block runs NoPE).
        let u32s = |v: &[u32]| Value::Array(v.iter().map(|&w| Value::U32(w)).collect());
        let bools = |v: &[bool]| Value::Array(v.iter().map(|&b| Value::Bool(b)).collect());
        let f32s = |v: &[f32]| Value::Array(v.iter().map(|&x| Value::F32(x)).collect());
        let pattern: Vec<bool> = window.iter().map(|&w| w > 0).collect();
        let keys: Vec<(String, Value)> = vec![
            ("vocab_size", Value::U32(n("vocab_size")?)),
            ("context_length", Value::U32(n("max_position_embeddings")?)),
            ("embedding_length", Value::U32(n("hidden_size")?)),
            ("block_count", Value::U32((layers + nextn) as u32)),
            ("feed_forward_length", Value::U32(n("intermediate_size")?)),
            ("leading_dense_block_count", Value::U32(dense_lead as u32)),
            ("nextn_predict_layers", Value::U32(nextn as u32)),
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
            (
                "attention.layer_norm_rms_epsilon",
                Value::F32(f("rms_norm_eps")?),
            ),
            ("rope.freq_base", Value::F32(rope.number("rope_theta")?)),
            ("rope.dimension_count", Value::U32(n("head_dim")?)),
            ("attention.sliding_window", u32s(&window)),
            ("attention.sliding_window_pattern", bools(&pattern)),
            ("attention.rope_pattern", bools(&pattern)),
            // The fork passes the block's limit to the routed and the shared experts alike.
            ("swiglu_clamp_exp", f32s(&clamp)),
            ("swiglu_clamp_shexp", f32s(&clamp)),
            ("expert_count", Value::U32(n("num_experts")?)),
            ("expert_used_count", Value::U32(n("num_experts_per_tok")?)),
            ("expert_shared_count", Value::U32(n("num_shared_experts")?)),
            (
                "expert_feed_forward_length",
                Value::U32(n("moe_intermediate_size")?),
            ),
            (
                "expert_shared_feed_forward_length",
                Value::U32(n("moe_intermediate_size")?),
            ),
            ("expert_gating_func", Value::U32(SIGMOID)),
            (
                "expert_weights_norm",
                Value::Bool(flag(&root, "norm_topk_prob")?),
            ),
            (
                "expert_weights_scale",
                Value::F32(f("routed_scaling_factor")?),
            ),
            ("expert_group_count", Value::U32(n("n_group")?)),
            ("expert_group_used_count", Value::U32(n("topk_group")?)),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        Ok(Checkpoint {
            blocks: layers + nextn,
            layers,
            experts,
            hidden: u64::from(n("hidden_size")?),
            query_rows: u64::from(n("num_attention_heads")?) * head_dim,
            key_rows: u64::from(n("num_key_value_heads")?) * head_dim,
            head_dim,
            expert_ff,
            // The shared experts are `num_shared_experts` of the expert width each: the config
            // names no width of their own, and the checkpoint holds them at the expert's.
            shared_ff: u64::from(n("num_shared_experts")?) * expert_ff,
            dense_ff: u64::from(n("intermediate_size")?),
            vocab_size: u64::from(n("vocab_size")?),
            dense_lead,
            vocab: hf::vocab::Config::parse(config)?,
            keys,
        })
    }

    /// The program's tensor of the block tensor `rest` of block `b` (the name past
    /// `model.layers.` or `mtp.layers.0.`), with the role it is read in; `None` for a routed
    /// expert's own tensor, which stacks.
    fn layer_tensor(&self, b: usize, rest: &str) -> Option<(String, Use)> {
        use Plain::{F32, Q8_0};
        // A trunk projection the engine reads as Q8_0: an FP8 one is decoded exactly and
        // quantized again. The MTP block is the draft, whose stored form stays.
        let trunk = if b >= self.layers {
            Use::Matrix(Q8_0)
        } else {
            Use::Q8_0
        };
        Some(match rest {
            "input_layernorm.weight" => (common::attn_norm(b), Use::Matrix(F32)),
            "post_attention_layernorm.weight" => (common::ffn_norm(b), Use::Matrix(F32)),
            "self_attn.q_norm.weight" => (common::attn_q_norm(b), Use::Matrix(F32)),
            "self_attn.k_norm.weight" => (common::attn_k_norm(b), Use::Matrix(F32)),
            "self_attn.q_proj.weight" => (common::attn_q(b), trunk),
            "self_attn.k_proj.weight" => (common::attn_k(b), trunk),
            "self_attn.v_proj.weight" => (common::attn_v(b), trunk),
            "self_attn.o_proj.weight" => (common::attn_output(b), trunk),
            "mlp.gate.weight" => (common::ffn_gate_inp(b), Use::Matrix(F32)),
            "mlp.e_score_correction_bias" => (common::exp_probs_b(b), Use::Matrix(F32)),
            "nextn.eh_proj.weight" => (nextn::nextn_eh_proj(b), Use::Matrix(Q8_0)),
            "nextn.enorm.weight" => (nextn::nextn_enorm(b), Use::Matrix(F32)),
            "nextn.hnorm.weight" => (nextn::nextn_hnorm(b), Use::Matrix(F32)),
            "nextn.shared_head_norm.weight" => (nextn::nextn_shared_head_norm(b), Use::Matrix(F32)),
            "mlp.gate_proj.weight" if self.dense(b) => (common::ffn_gate(b), trunk),
            "mlp.up_proj.weight" if self.dense(b) => (common::ffn_up(b), trunk),
            "mlp.down_proj.weight" if self.dense(b) => (common::ffn_down(b), trunk),
            // The shared experts sit on the routed blocks alone: an FP8 weight decoded exactly
            // and quantized again.
            "mlp.shared_experts.gate_proj.weight" => (common::ffn_gate_shexp(b), Use::Q8_0),
            "mlp.shared_experts.up_proj.weight" => (common::ffn_up_shexp(b), Use::Q8_0),
            "mlp.shared_experts.down_proj.weight" => (common::ffn_down_shexp(b), Use::Q8_0),
            _ => return None,
        })
    }

    /// Whether block `b` holds a dense feed-forward block: one of the leading run or the MTP
    /// one.
    fn dense(&self, b: usize) -> bool {
        b < self.dense_lead || b >= self.layers
    }

    /// Every tensor the config says a folder holds, by the name the map gives it, with the ggml
    /// dims the config's geometry gives: the model-level three, each block's attention and
    /// feed-forward tensors, and the MTP block's `nextn` tensors beside them.
    fn expected(&self) -> BTreeMap<String, Vec<u64>> {
        let (e, experts) = (self.hidden, self.experts as u64);
        let mut want: BTreeMap<String, Vec<u64>> = [
            (common::token_embd(), vec![e, self.vocab_size]),
            (common::output_norm(), vec![e]),
            (common::output(), vec![e, self.vocab_size]),
        ]
        .into_iter()
        .collect();
        for b in 0..self.blocks {
            let mut layer: Vec<(String, Vec<u64>)> = vec![
                (common::attn_norm(b), vec![e]),
                (common::ffn_norm(b), vec![e]),
                (common::attn_q_norm(b), vec![self.head_dim]),
                (common::attn_k_norm(b), vec![self.head_dim]),
                (common::attn_q(b), vec![e, self.query_rows]),
                (common::attn_k(b), vec![e, self.key_rows]),
                (common::attn_v(b), vec![e, self.key_rows]),
                (common::attn_output(b), vec![self.query_rows, e]),
            ];
            if self.dense(b) {
                layer.extend([
                    (common::ffn_gate(b), vec![e, self.dense_ff]),
                    (common::ffn_up(b), vec![e, self.dense_ff]),
                    (common::ffn_down(b), vec![self.dense_ff, e]),
                ]);
            } else {
                layer.extend([
                    (common::ffn_gate_inp(b), vec![e, experts]),
                    (common::exp_probs_b(b), vec![experts]),
                    (common::ffn_gate_exps(b), vec![e, self.expert_ff, experts]),
                    (common::ffn_up_exps(b), vec![e, self.expert_ff, experts]),
                    (common::ffn_down_exps(b), vec![self.expert_ff, e, experts]),
                    (common::ffn_gate_shexp(b), vec![e, self.shared_ff]),
                    (common::ffn_up_shexp(b), vec![e, self.shared_ff]),
                    (common::ffn_down_shexp(b), vec![self.shared_ff, e]),
                ]);
            }
            if b >= self.layers {
                layer.extend([
                    (nextn::nextn_eh_proj(b), vec![2 * e, e]),
                    (nextn::nextn_enorm(b), vec![e]),
                    (nextn::nextn_hnorm(b), vec![e]),
                    (nextn::nextn_shared_head_norm(b), vec![e]),
                ]);
            }
            want.extend(layer);
        }
        want
    }

    /// Every tensor the config says the folder holds is mapped at the config's dims, and no
    /// other is.
    fn complete(&self, mapped: &BTreeMap<String, Mapped<Infallible>>) -> Result<(), Error> {
        let want = self.expected();
        for (name, dims) in &want {
            let m = mapped
                .get(name)
                .ok_or_else(|| refuse(name, "is missing from the folder"))?;
            if m.dims != *dims {
                return Err(refuse(
                    &m.sources[0],
                    format!("holds dims {:?} where the config gives {dims:?}", m.dims),
                ));
            }
        }
        if let Some((_, m)) = mapped.iter().find(|(n, _)| !want.contains_key(*n)) {
            return Err(refuse(
                &m.sources[0],
                "is a tensor no block of this config holds: mlp_layer_types and \
                 num_nextn_predict_layers say which blocks hold which",
            ));
        }
        Ok(())
    }

    /// The tensors a folder of this config stores, as the checkpoint names, types and shapes
    /// them: the FP8 weights each beside its `weight_scale_inv` grid, BF16 for the embedding,
    /// the head, the norms and the MTP block, F32 for the router and its bias.
    fn stored_tensors(&self) -> Vec<(String, Dtype, Vec<usize>)> {
        use Dtype::{Bf16, F8E4M3, F32};
        let (e, experts) = (self.hidden as usize, self.experts);
        let (q, k) = (self.query_rows as usize, self.key_rows as usize);
        let head = self.head_dim as usize;
        let (eff, shared, ff) = (
            self.expert_ff as usize,
            self.shared_ff as usize,
            self.dense_ff as usize,
        );
        let vocab = self.vocab_size as usize;
        let grid = |rows: usize, cols: usize| vec![rows.div_ceil(128), cols.div_ceil(128)];
        let mut t: Vec<(String, Dtype, Vec<usize>)> = vec![];
        let mut add = |name: String, dtype: Dtype, shape: &[usize]| {
            t.push((name, dtype, shape.to_vec()));
        };
        // The embedding and the head sit in the FP8 export's `ignored_layers`: BF16, no scale.
        add("model.embed_tokens.weight".into(), Bf16, &[vocab, e]);
        add("lm_head.weight".into(), Bf16, &[vocab, e]);
        add("model.norm.weight".into(), Bf16, &[e]);
        for l in 0..self.layers {
            let p = format!("model.layers.{l}.");
            let mut layer = |rest: &str, dtype: Dtype, shape: &[usize]| {
                add(format!("{p}{rest}"), dtype, shape);
            };
            layer("input_layernorm.weight", Bf16, &[e]);
            layer("post_attention_layernorm.weight", Bf16, &[e]);
            layer("self_attn.q_norm.weight", Bf16, &[head]);
            layer("self_attn.k_norm.weight", Bf16, &[head]);
            for (proj, rows, cols) in [
                ("q_proj", q, e),
                ("k_proj", k, e),
                ("v_proj", k, e),
                ("o_proj", e, q),
            ] {
                layer(&format!("self_attn.{proj}.weight"), F8E4M3, &[rows, cols]);
                layer(
                    &format!("self_attn.{proj}.weight_scale_inv"),
                    F32,
                    &grid(rows, cols),
                );
            }
            if l < self.dense_lead {
                for (proj, rows, cols) in [
                    ("gate_proj", ff, e),
                    ("up_proj", ff, e),
                    ("down_proj", e, ff),
                ] {
                    layer(&format!("mlp.{proj}.weight"), F8E4M3, &[rows, cols]);
                    layer(
                        &format!("mlp.{proj}.weight_scale_inv"),
                        F32,
                        &grid(rows, cols),
                    );
                }
                continue;
            }
            layer("mlp.gate.weight", F32, &[experts, e]);
            layer("mlp.e_score_correction_bias", F32, &[experts]);
            for (proj, rows, cols) in [
                ("gate_proj", shared, e),
                ("up_proj", shared, e),
                ("down_proj", e, shared),
            ] {
                layer(
                    &format!("mlp.shared_experts.{proj}.weight"),
                    F8E4M3,
                    &[rows, cols],
                );
                layer(
                    &format!("mlp.shared_experts.{proj}.weight_scale_inv"),
                    F32,
                    &grid(rows, cols),
                );
            }
            for id in 0..experts {
                for (proj, rows, cols) in [
                    ("gate_proj", eff, e),
                    ("up_proj", eff, e),
                    ("down_proj", e, eff),
                ] {
                    layer(
                        &format!("mlp.experts.{id}.{proj}.weight"),
                        F8E4M3,
                        &[rows, cols],
                    );
                    layer(
                        &format!("mlp.experts.{id}.{proj}.weight_scale_inv"),
                        F32,
                        &grid(rows, cols),
                    );
                }
            }
        }
        // The MTP block sits in the FP8 export's `ignored_layers` too: BF16, no scale.
        add("mtp.fc.weight".into(), Bf16, &[e, 2 * e]);
        add("mtp.pre_fc_norm_embedding.weight".into(), Bf16, &[e]);
        add("mtp.pre_fc_norm_hidden.weight".into(), Bf16, &[e]);
        add("mtp.norm.weight".into(), Bf16, &[e]);
        add("mtp.layers.0.input_layernorm.weight".into(), Bf16, &[e]);
        add(
            "mtp.layers.0.post_attention_layernorm.weight".into(),
            Bf16,
            &[e],
        );
        add("mtp.layers.0.self_attn.q_norm.weight".into(), Bf16, &[head]);
        add("mtp.layers.0.self_attn.k_norm.weight".into(), Bf16, &[head]);
        for (proj, rows, cols) in [
            ("q_proj", q, e),
            ("k_proj", k, e),
            ("v_proj", k, e),
            ("o_proj", e, q),
        ] {
            add(
                format!("mtp.layers.0.self_attn.{proj}.weight"),
                Bf16,
                &[rows, cols],
            );
        }
        for (proj, rows, cols) in [
            ("gate_proj", ff, e),
            ("up_proj", ff, e),
            ("down_proj", e, ff),
        ] {
            add(
                format!("mtp.layers.0.mlp.{proj}.weight"),
                Bf16,
                &[rows, cols],
            );
        }
        t.sort_by(|a, b| a.0.cmp(&b.0));
        t
    }
}

impl FolderMap for Checkpoint {
    /// A tensor needs no move: the stored layout is the program's.
    type Step = Infallible;

    const MODEL_TYPES: &'static [&'static str] = &["exaone_moe"];

    fn parse(config: &Json) -> Result<Self, Error> {
        Checkpoint::read(config)
    }

    fn architecture(&self) -> &'static str {
        ARCHITECTURE
    }

    fn module_class(&self, module: &str) -> Option<ModuleClass> {
        super::module_class(module)
    }

    fn map<'a>(&self, tensors: &[(&'a str, &'a [usize])]) -> Result<Mapping<Infallible>, Error> {
        let mut out = Mapping {
            tensors: BTreeMap::new(),
            skipped: BTreeMap::new(),
        };
        let mut stacks: BTreeMap<(usize, &str), Stack> = BTreeMap::new();
        for &(name, shape) in tensors {
            if !trunk::sized(shape) {
                return Err(refuse(
                    name,
                    format!("shape {shape:?} is empty, zero, or overflows usize"),
                ));
            }
            let dims = shape.iter().rev().map(|&d| d as u64).collect::<Vec<_>>();
            let made = |role: Use| Mapped {
                sources: vec![name.to_string()],
                role,
                steps: vec![],
                dims,
                origin: Origin::Made,
            };
            let mut put = |to: String, m: Mapped<Infallible>| match out.tensors.insert(to, m) {
                Some(prev) => Err(refuse(
                    &prev.sources[0],
                    "maps to an exaone-moe name another tensor has",
                )),
                None => Ok(()),
            };
            // The model-level tensors and the MTP block's own names sit outside the layers'
            // prefix: the MTP layer under `mtp.layers.0.`, its input projection and norms under
            // `mtp.` alone.
            let (b, rest) = match name.strip_prefix("model.layers.") {
                Some(layer) => {
                    let (l, rest) = layer
                        .split_once('.')
                        .and_then(|(l, rest)| Some((l.parse::<usize>().ok()?, rest)))
                        .ok_or_else(|| refuse(name, "not a K-EXAONE tensor"))?;
                    if l >= self.layers {
                        return Err(refuse(name, format!("layer {l} of {}", self.layers)));
                    }
                    (l, rest)
                }
                None => match name {
                    "model.embed_tokens.weight" => {
                        put(common::token_embd(), made(Use::Matrix(Plain::Bf16)))?;
                        continue;
                    }
                    "lm_head.weight" => {
                        put(common::output(), made(Use::Matrix(Plain::Q8_0)))?;
                        continue;
                    }
                    "model.norm.weight" => {
                        put(common::output_norm(), made(Use::Matrix(Plain::F32)))?;
                        continue;
                    }
                    "mtp.fc.weight" => (self.layers, "nextn.eh_proj.weight"),
                    "mtp.pre_fc_norm_embedding.weight" => (self.layers, "nextn.enorm.weight"),
                    "mtp.pre_fc_norm_hidden.weight" => (self.layers, "nextn.hnorm.weight"),
                    "mtp.norm.weight" => (self.layers, "nextn.shared_head_norm.weight"),
                    _ if name.starts_with("mtp.layers.0.") => {
                        (self.layers, &name["mtp.layers.0.".len()..])
                    }
                    _ => return Err(refuse(name, "not a K-EXAONE tensor")),
                },
            };
            if let Some((id, projection)) = rest
                .strip_prefix("mlp.experts.")
                .and_then(|e| e.split_once('.'))
                .and_then(|(id, p)| Some((id.parse::<usize>().ok()?, p)))
            {
                if !matches!(
                    projection,
                    "gate_proj.weight" | "up_proj.weight" | "down_proj.weight"
                ) {
                    return Err(refuse(name, "not a K-EXAONE expert projection"));
                }
                if b >= self.layers {
                    return Err(refuse(
                        name,
                        "a routed expert of the MTP block, which holds a dense one",
                    ));
                }
                if stacks
                    .entry((b, projection))
                    .or_default()
                    .insert(id, (name, shape))
                    .is_some()
                {
                    return Err(refuse(name, format!("duplicate expert id {id}")));
                }
                continue;
            }
            // A feed-forward stem of the other block kind than its own is refused before it can
            // map: `mlp_layer_types` says which blocks hold which.
            if matches!(
                rest,
                "mlp.gate_proj.weight" | "mlp.up_proj.weight" | "mlp.down_proj.weight"
            ) && !self.dense(b)
            {
                return Err(refuse(
                    name,
                    "a dense block's projection, where mlp_layer_types holds sparse at this block",
                ));
            }
            if rest.starts_with("mlp.shared_experts.") && self.dense(b) {
                return Err(refuse(
                    name,
                    "a routed block's shared experts, where mlp_layer_types holds dense at this \
                     block",
                ));
            }
            let (to, role) = self
                .layer_tensor(b, rest)
                .ok_or_else(|| refuse(name, "not a K-EXAONE tensor"))?;
            put(to, made(role))?;
        }
        for ((b, projection), by_id) in stacks {
            let (sources, dims) = safetensors::expert_stack(&by_id, self.experts)
                .map_err(|e| refuse(&format!("block {b} {projection}"), e.to_string()))?;
            let to = match projection {
                "gate_proj.weight" => common::ffn_gate_exps(b),
                "up_proj.weight" => common::ffn_up_exps(b),
                _ => common::ffn_down_exps(b),
            };
            out.tensors.insert(
                to,
                Mapped {
                    sources,
                    // The checkpoint's FP8_B128 blocks are kept for the quantizer to read; a
                    // BF16 or F32 stack is refused by the plan, as any other folder's is.
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
        _folder: &Folder,
    ) -> Result<Vec<(String, Value)>, Error> {
        Ok(trunk::header(ARCHITECTURE, &self.keys))
    }

    fn vocab(&self) -> &VocabSpec {
        &self.vocab
    }

    fn stored(&self) -> Option<Stored> {
        Some(Stored {
            tensors: self.stored_tensors(),
            skips: |_| None,
        })
    }
}
