//! Every `inkling2.*` key the description reads, both loop dialects ([`Dialect`]), and the
//! tensors the file is held to: the names every layer's kind carries, their dims and the visits
//! a looped file's per-block arrays describe. A key the loader requires is required here, and a
//! value this reader does not run is refused by the key's name.
//!
//! The two dialects of the loop, chosen by the file's `general.architecture` (both name the one
//! program):
//!
//! - [`Dialect::Unsloth`] (`inkling2`, the word both files on the box carry): `block_count` is
//!   the layer *visits* — the stored layers plus the core's repeats — and the per-block arrays
//!   (head counts, the sliding-window pattern, the used experts) are one value a visit, a
//!   visit's value its layer's first visit's. The loop is read where `loop.core_end` is above 0:
//!   `loop.core_start`, `num_loops` and `loop.gate_rank` then. A core layer's message
//!   controllers are its twelve `loop_{attn,k,v,mlp}_{down,gain,forget}` tensors, the message
//!   queries the model-level `loop_res_query` and `loop_kv_res_query`, two rows a core layer
//!   each.
//! - [`Dialect::Master`] (`inkling2_loop`, llama.cpp master's renamed word): `block_count` is
//!   the *stored* layers, the loop always read from `loop.start`, `loop.end`, `loop.count` and
//!   `loop.gate_rank`. A core layer carries its `loop_msg_query` (the four message queries in
//!   one tensor) and its twelve `loop_{q,k,v,ffn}_ctrl_{down,gain,forget}` tensors, and the
//!   model level carries no query.
//!
//! A key or stem of the other dialect is refused by name, and a `loop.varying` flag (the
//! research checkpoint's serve-time pass count over shared cache slots) is refused by its own
//! name: this reader describes the fixed loop, whose every pass keeps its own slots. The
//! dialect's authority is llama.cpp's `src/models/inkling2.cpp` (`src/models/inkling2-loop.cpp`
//! for the second spelling).

use gguf::Split;
use models::{RouterNorm, Score};

use super::names;
use super::roles::{self, KINDS, LayerKind, stems_of};
use crate::arch::delta_mla::Situ;
use crate::arch::hybrid::{layers_agree, no_nextn};
use crate::arch::{
    INKLING2_LOOP, TOKENS, check_kv, dims_agree, expert_counts, finite_positive, meta_arr,
    meta_f32, meta_usize, metadata, mul, optional_usize, per_layer_count, positive, unread_keys_of,
};
use crate::placement::PlacementError;

/// Which spelling of the loop a file carries, by its `general.architecture`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    /// `inkling2`: the unsloth files' keys and stems, `block_count` the visits.
    Unsloth,
    /// `inkling2_loop`: llama.cpp master's, `block_count` the stored layers.
    Master,
}

/// The loop of a Mini-shaped file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Loop {
    /// `loop.core_start` / `loop.start`.
    pub core_start: usize,
    /// `loop.core_end` / `loop.end`.
    pub core_end: usize,
    /// `num_loops` / `loop.count`.
    pub passes: usize,
    /// `loop.gate_rank`.
    pub gate_rank: usize,
}

/// The hyperparameters of one Inkling 2 file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// The loop's dialect, by the file's architecture word.
    pub dialect: Dialect,
    /// `block_count`: the visits under [`Dialect::Unsloth`], the stored layers under
    /// [`Dialect::Master`].
    pub n_blocks: usize,
    /// The stored layers: `block_count` less the core's repeats.
    pub n_layer: usize,
    /// `embedding_length`.
    pub n_embd: usize,
    /// `attention.head_count`, a value a stored layer.
    pub n_head: Vec<usize>,
    /// `attention.head_count_kv`, a value a stored layer.
    pub n_head_kv: Vec<usize>,
    /// `attention.key_length`, the value width too (the reference holds them equal).
    pub head_dim: usize,
    /// `attention.layer_norm_rms_epsilon`.
    pub rms_eps: f32,
    /// `context_length`.
    pub n_ctx_train: usize,
    /// `vocab_size`: the padded vocabulary the embedding and output tensors hold.
    pub n_vocab: usize,
    /// `vocab_size_unpadded`: the tokenizer's size, the head's columns.
    pub n_vocab_unpadded: usize,
    /// `expert_count`.
    pub n_expert: usize,
    /// `expert_used_count`, a value a stored layer.
    pub n_used: Vec<usize>,
    /// `expert_feed_forward_length`, a value a stored layer.
    pub expert_ff: Vec<usize>,
    /// `expert_weights_scale`.
    pub weights_scale: f32,
    /// `final_logit_softcapping`.
    pub softcap: f32,
    /// `activation.situ_beta` and `activation.situ_linear_beta`.
    pub situ: Situ,
    /// `attention.relative_length`.
    pub d_rel: usize,
    /// `attention.relative_extent`: the windowless layers' bias-bank width.
    pub extent: usize,
    /// `attention.qk_clip`.
    pub qk_clip: f32,
    /// `attention.relative_clip`.
    pub rel_clip: f32,
    /// `attention.log_scale_floor`.
    pub log_floor: u32,
    /// `attention.log_scale_alpha`.
    pub log_alpha: f32,
    /// `shortconv.l_cache`: the taps of the four short convolutions.
    pub conv: usize,
    /// `attention.sliding_window`.
    pub swa: usize,
    /// `attention.sliding_window_pattern`, a value a stored layer.
    pub windowed: Vec<bool>,
    /// `attn_res.block_size`.
    pub block: usize,
    /// The loop; `None`: one pass over every layer.
    pub loop_: Option<Loop>,
    /// The keys this read took a default for: the key, the value, the line.
    pub defaults: Vec<String>,
}

/// Every `inkling2.` key [`Hparams::read`] reads, by suffix.
const READ: &[&str] = &[
    "block_count",
    "context_length",
    "embedding_length",
    "vocab_size",
    "vocab_size_unpadded",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.layer_norm_rms_epsilon",
    "attention.sliding_window",
    "attention.sliding_window_pattern",
    "attention.relative_length",
    "attention.relative_extent",
    "attention.qk_clip",
    "attention.relative_clip",
    "attention.log_scale_floor",
    "attention.log_scale_alpha",
    "attn_res.block_size",
    "shortconv.l_cache",
    "expert_count",
    "expert_used_count",
    "expert_feed_forward_length",
    "expert_weights_scale",
    "final_logit_softcapping",
    "activation.situ_beta",
    "activation.situ_linear_beta",
];

/// The loop keys of [`Dialect::Unsloth`]: `loop.gate_rank` both dialects share.
const LOOP_UNSLOTH: &[&str] = &[
    "num_loops",
    "loop.core_start",
    "loop.core_end",
    "loop.gate_rank",
];

/// The loop keys of [`Dialect::Master`]: `loop.gate_rank` both dialects share.
const LOOP_MASTER: &[&str] = &["loop.start", "loop.end", "loop.count", "loop.gate_rank"];

/// The architecture keys and `general.sampling.*` keys of `split` that [`Hparams::read`] does not
/// read: listed, not refused.
#[must_use]
pub fn unread_keys(split: &Split) -> Vec<String> {
    let loop_keys = match dialect_of(split) {
        Some(Dialect::Master) => LOOP_MASTER,
        _ => LOOP_UNSLOTH,
    };
    unread_keys_of(split, &[READ, loop_keys].concat())
}

/// The dialect of `split`'s architecture word; `None` for any other word.
fn dialect_of(split: &Split) -> Option<Dialect> {
    match split.architecture() {
        Some("inkling2") => Some(Dialect::Unsloth),
        Some(INKLING2_LOOP) => Some(Dialect::Master),
        _ => None,
    }
}

impl Hparams {
    /// Every value from `split`'s headers; a missing required key, a value this reader does not
    /// run, a loop that is not the fixed loop of whole blocks inside the trunk, or a layer whose
    /// tensors disagree with its kind or the keys is an error naming the key or the tensor.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        let dialect = dialect_of(split).ok_or_else(|| {
            crate::arch::architecture_is(&format!(
                "{:?}; this reader reads inkling2 and {INKLING2_LOOP} files",
                split.architecture().unwrap_or("<missing>")
            ))
        })?;
        let mut defaults = Vec::new();
        let n_blocks = positive(split, "block_count")?;
        no_nextn(split)?;
        refuse_other_dialect(split, dialect)?;
        let loop_ = read_loop(split, dialect)?;
        let n_layer = n_layer_of(dialect, n_blocks, &loop_);
        let n_embd = positive(split, "embedding_length")?;
        let head_dim = positive(split, "attention.key_length")?;
        let value = positive(split, "attention.value_length")?;
        if value != head_dim {
            return Err(metadata(
                split,
                "attention.value_length",
                format!("is {value}, not key_length {head_dim}"),
            ));
        }
        let n_head = stored_counts(split, "attention.head_count", n_blocks, &loop_, n_layer)?;
        let n_head_kv = stored_counts(split, "attention.head_count_kv", n_blocks, &loop_, n_layer)?;
        let n_used = stored_counts(split, "expert_used_count", n_blocks, &loop_, n_layer)?;
        let expert_ff = stored_counts(
            split,
            "expert_feed_forward_length",
            n_blocks,
            &loop_,
            n_layer,
        )?;
        let windowed = stored_pattern(split, n_blocks, &loop_, n_layer)?;
        let situ = Situ::read(split)?;
        let n_expert = expert_counts(split)?.0;
        let weights_scale = match split.value(&split.arch_key("expert_weights_scale")) {
            None => {
                // inkling2.cpp:26 reads the key optional over llama.cpp's 0 default
                // (llama-hparams.h:138); no released file lacks it.
                defaults.push("expert_weights_scale = 0 (llama.cpp's default)".into());
                0.0
            }
            Some(_) => finite_positive(split, "expert_weights_scale")?,
        };
        let softcap = match split.value(&split.arch_key("final_logit_softcapping")) {
            None => {
                defaults.push("final_logit_softcapping = 0 (llama.cpp's default)".into());
                0.0
            }
            Some(_) => meta_f32(split, "final_logit_softcapping")?,
        };
        let hp = Hparams {
            dialect,
            n_blocks,
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            rms_eps: finite_positive(split, "attention.layer_norm_rms_epsilon")?,
            n_ctx_train: positive(split, "context_length")?,
            n_vocab: positive(split, "vocab_size")?,
            n_vocab_unpadded: positive(split, "vocab_size_unpadded")?,
            n_expert,
            n_used,
            expert_ff,
            weights_scale,
            softcap,
            situ,
            d_rel: positive(split, "attention.relative_length")?,
            extent: positive(split, "attention.relative_extent")?,
            qk_clip: finite_positive(split, "attention.qk_clip")?,
            rel_clip: finite_positive(split, "attention.relative_clip")?,
            log_floor: log_floor(split)?,
            log_alpha: log_alpha(split)?,
            conv: conv(split)?,
            swa: positive(split, "attention.sliding_window")?,
            block: positive(split, "attn_res.block_size")?,
            windowed,
            loop_,
            defaults,
        };
        hp.check(split)?;
        hp.tensors_agree(split)?;
        dims_agree(
            split,
            hp.n_embd,
            hp.n_vocab,
            (0..hp.n_layer).map(|l| hp.layer_dims(l)),
        )?;
        hp.model_dims_agree(split)?;
        Ok(hp)
    }

    /// Layer `l`'s kind: whether it is a windowed layer, whether it is a core layer of the
    /// loop, and whether it is the trunk's first layer.
    #[must_use]
    pub fn kind(&self, l: usize) -> LayerKind {
        (self.windowed[l], self.in_core(l), l == 0)
    }

    /// Whether layer `l` is inside the loop's core.
    #[must_use]
    pub fn in_core(&self, l: usize) -> bool {
        self.loop_
            .is_some_and(|loop_| (loop_.core_start..loop_.core_end).contains(&l))
    }

    /// The value checks the reference runs once over the whole file
    /// (src/models/inkling2.cpp:38-84, 95-97): every width, count and vocabulary it refuses, by
    /// the key's name.
    fn check(&self, split: &Split) -> Result<(), PlacementError> {
        for l in 0..self.n_layer {
            check_kv(split, self.n_head[l], l, self.n_head_kv[l])?;
            mul(
                split,
                "attention.key_length",
                self.n_head_kv[l],
                self.head_dim,
            )?;
            if self.n_used[l] == 0 || self.n_used[l] > self.n_expert {
                return Err(metadata(
                    split,
                    "expert_used_count",
                    format!(
                        "is {} at layer {l} of expert_count {}",
                        self.n_used[l], self.n_expert
                    ),
                ));
            }
            if self.expert_ff[l] == 0 {
                return Err(metadata(
                    split,
                    "expert_feed_forward_length",
                    format!("is 0 at layer {l}"),
                ));
            }
        }
        if self.n_vocab_unpadded > self.n_vocab {
            return Err(metadata(
                split,
                "vocab_size_unpadded",
                format!(
                    "is {} over the padded vocab_size {}",
                    self.n_vocab_unpadded, self.n_vocab
                ),
            ));
        }
        let tokens = match split.value(TOKENS) {
            Some(gguf::Value::Array(tokens)) => Some(tokens.len()),
            _ => None,
        };
        match tokens {
            Some(n) if n == self.n_vocab_unpadded => {}
            Some(n) => {
                return Err(metadata(
                    split,
                    "vocab_size_unpadded",
                    format!(
                        "is {} and the token list holds {n}: the tokenizer size must equal the \
                         unpadded vocabulary",
                        self.n_vocab_unpadded
                    ),
                ));
            }
            None => {
                return Err(PlacementError::Metadata {
                    key: TOKENS.to_string(),
                    detail: "is absent: the tokenizer size must equal the unpadded vocabulary"
                        .to_string(),
                });
            }
        }
        Ok(())
    }

    /// The model-level tensors a padded or looped file carries besides the three every family
    /// names, at the dims the keys give them.
    fn model_dims_agree(&self, split: &Split) -> Result<(), PlacementError> {
        let e = self.n_embd as u64;
        let mut want = vec![
            (names::TOKEN_EMBD_NORM.to_string(), vec![e]),
            (names::OUTPUT_RES_QUERY.to_string(), vec![e]),
        ];
        if let Some(loop_) = self.loop_
            && self.dialect == Dialect::Unsloth
        {
            let queries = vec![e, 2 * (loop_.core_end - loop_.core_start) as u64];
            want.push((names::LOOP_RES_QUERY.to_string(), queries.clone()));
            want.push((names::LOOP_KV_RES_QUERY.to_string(), queries));
        }
        for (name, dims) in want {
            match split.find(&name) {
                None => {
                    return Err(PlacementError::Tensor {
                        name,
                        detail: "is not in the file (the description needs it)".to_string(),
                    });
                }
                Some((_, t)) if t.dims != dims => {
                    return Err(PlacementError::Tensor {
                        name,
                        detail: format!("has dims {:?}; the keys give {dims:?}", t.dims),
                    });
                }
                Some(_) => {}
            }
        }
        Ok(())
    }

    /// The stored layers' required tensors are in the file, and nothing of another kind's: the
    /// first missing or misplaced name, with how many more there are.
    fn tensors_agree(&self, split: &Split) -> Result<(), PlacementError> {
        let kinds: Vec<LayerKind> = (0..self.n_layer).map(|l| self.kind(l)).collect();
        layers_agree(split, &kinds, |kind| stems_of(self.dialect, kind), &KINDS)
    }

    /// Layer `l`'s tensors by stem and the dims llama.cpp creates them with, ggml order. The
    /// routed stacks are the fused gate and up pair the architecture writes, a windowed layer's
    /// bias bank its window wide and a windowless layer's `attention.relative_extent`.
    #[must_use]
    pub fn layer_dims(&self, l: usize) -> Vec<(&'static str, Vec<u64>)> {
        let (e, d) = (self.n_embd as u64, self.head_dim as u64);
        let (heads, kv) = (self.n_head[l] as u64, self.n_head_kv[l] as u64);
        let (rel, taps) = (self.d_rel as u64, self.conv as u64);
        let extent = if self.windowed[l] {
            self.swa as u64
        } else {
            self.extent as u64
        };
        let mut dims: Vec<(&'static str, Vec<u64>)> = vec![
            ("attn_q.weight", vec![e, d * heads]),
            ("attn_k.weight", vec![e, d * kv]),
            ("attn_v.weight", vec![e, d * kv]),
            ("attn_output.weight", vec![d * heads, e]),
            ("attn_gate.weight", vec![e, d * heads]),
            ("attn_q_clip.weight", vec![d, heads]),
            ("attn_sink_proj.weight", vec![e, heads]),
            ("attn_rel.weight", vec![e, rel * heads]),
            ("attn_rel_bias.weight", vec![rel, extent]),
            ("attn_rel_clip.weight", vec![rel, heads]),
            ("attn_k_conv.weight", vec![taps, d * kv]),
            ("attn_v_conv.weight", vec![taps, d * kv]),
            ("attn_post_conv.weight", vec![taps, e]),
            ("ffn_post_conv.weight", vec![taps, e]),
        ];
        if self.windowed[l] {
            dims.push(("attn_norm.weight", vec![e]));
        } else {
            dims.extend([
                ("attn_q_inp_norm.weight", vec![e]),
                ("attn_k_inp_norm.weight", vec![e]),
                ("attn_v_inp_norm.weight", vec![e]),
                ("attn_k_conv_proj.weight", vec![e, taps * kv]),
                ("attn_v_conv_proj.weight", vec![e, taps * kv]),
            ]);
        }
        if l > 0 {
            let queries = u64::from(!self.windowed[l]) * 2 + 1;
            dims.push(("attn_res_query.weight", vec![e, queries]));
        }
        let (n, ff) = (self.n_expert as u64, self.expert_ff[l] as u64);
        dims.extend([
            ("ffn_res_query.weight", vec![e]),
            ("ffn_norm.weight", vec![e]),
            ("ffn_gate_inp.weight", vec![e, n]),
            ("exp_probs_b.bias", vec![n]),
            ("ffn_gate_up_exps.weight", vec![e, 2 * ff, n]),
            ("ffn_down_exps.weight", vec![ff, e, n]),
        ]);
        if self.in_core(l) {
            let rank = self.loop_.map_or(0, |loop_| loop_.gate_rank) as u64;
            let ctrls: &[crate::arch::stems::Stem] = match self.dialect {
                Dialect::Unsloth => roles::LOOP_UNSLOTH,
                Dialect::Master => roles::LOOP_MASTER,
            };
            dims.extend(ctrls.iter().map(|s| (s.name, ctrl_dims(s.name, e, rank))));
        }
        dims
    }
}

/// The dims of one message-controller projection: `down` reads the joined
/// `[rms(cur); rms(msg)]` row down to the rank, `gain` and `forget` write the hidden row back.
fn ctrl_dims(stem: &str, e: u64, rank: u64) -> Vec<u64> {
    if stem.ends_with("down.weight") {
        vec![2 * e, rank]
    } else if stem.ends_with("msg_query.weight") {
        vec![e, 4]
    } else {
        vec![rank, e]
    }
}

/// The stored layers of a trunk of `n_blocks` blocks under `loop_`: under [`Dialect::Master`]
/// the blocks are the stored layers (llama-model.cpp:1687-1698); under [`Dialect::Unsloth`] the
/// visits are, the stored layers their count less the core's repeats
/// (src/models/inkling2-loop.cpp:29-33).
fn n_layer_of(dialect: Dialect, n_blocks: usize, loop_: &Option<Loop>) -> usize {
    match (dialect, loop_) {
        (Dialect::Master, _) | (_, None) => n_blocks,
        (Dialect::Unsloth, Some(loop_)) => {
            n_blocks - (loop_.passes - 1) * (loop_.core_end - loop_.core_start)
        }
    }
}

/// The loop's key by the dialect: `pr_word` under [`Dialect::Unsloth`], `master_word` under
/// [`Dialect::Master`].
fn loop_key(split: &Split, pr_word: &str, master_word: &str) -> String {
    split.arch_key(match dialect_of(split) {
        Some(Dialect::Master) => master_word,
        _ => pr_word,
    })
}

/// A dialect's loop, refused wherever it is not the fixed loop of whole blocks inside the trunk;
/// `None` where the file carries no loop of that dialect.
fn read_loop(split: &Split, dialect: Dialect) -> Result<Option<Loop>, PlacementError> {
    let loop_ = match dialect {
        Dialect::Unsloth => {
            let Some(core_end) = optional_usize(split, "loop.core_end")? else {
                return Ok(None);
            };
            if core_end == 0 {
                return Ok(None);
            }
            Loop {
                core_start: positive(split, "loop.core_start")?,
                core_end,
                passes: positive(split, "num_loops")?,
                gate_rank: positive(split, "loop.gate_rank")?,
            }
        }
        Dialect::Master => Loop {
            core_start: positive(split, "loop.start")?,
            core_end: positive(split, "loop.end")?,
            passes: positive(split, "loop.count")?,
            gate_rank: positive(split, "loop.gate_rank")?,
        },
    };
    // llama.cpp's LLAMA_MAX_LAYERS bounds the visits a loop expands to.
    const MAX_LAYERS: usize = 512;
    let core = loop_.core_end - loop_.core_start;
    let visits = loop_.core_end + (loop_.passes - 1) * core;
    if loop_.passes < 2 {
        return Err(metadata(
            split,
            &loop_key(split, "num_loops", "loop.count"),
            format!(
                "is {}; the fixed loop runs at least two passes over its core",
                loop_.passes
            ),
        ));
    }
    if loop_.core_start >= loop_.core_end {
        return Err(metadata(
            split,
            &loop_key(split, "loop.core_start", "loop.start"),
            format!("is {} at core_end {}", loop_.core_start, loop_.core_end),
        ));
    }
    let n_blocks = meta_usize(split, "block_count")?;
    let bound = match dialect {
        Dialect::Unsloth => visits,
        Dialect::Master => loop_.core_end,
    };
    if bound > n_blocks {
        return Err(metadata(
            split,
            &loop_key(split, "loop.core_end", "loop.end"),
            format!(
                "is {} of block_count {n_blocks} with {} passes: the looped visits must close \
                 inside the trunk",
                loop_.core_end, loop_.passes
            ),
        ));
    }
    if visits > MAX_LAYERS {
        return Err(metadata(
            split,
            "block_count",
            format!("is {visits} with the loop's visits, above the layer limit {MAX_LAYERS}"),
        ));
    }
    let block = positive(split, "attn_res.block_size")?;
    if !loop_.core_start.is_multiple_of(block) || !core.is_multiple_of(block) {
        return Err(metadata(
            split,
            &loop_key(split, "loop.core_start", "loop.start"),
            format!(
                "is {} with core_end {} for a block size {block}: the looped core must span \
                 whole blocks of the attention residual",
                loop_.core_start, loop_.core_end
            ),
        ));
    }
    Ok(Some(loop_))
}

/// The loop keys and the varying flag of the other dialect, refused by name.
fn refuse_other_dialect(split: &Split, dialect: Dialect) -> Result<(), PlacementError> {
    let (others, word) = match dialect {
        Dialect::Unsloth => (
            ["loop.start", "loop.end", "loop.count", "loop.varying"],
            INKLING2_LOOP,
        ),
        Dialect::Master => (
            [
                "num_loops",
                "loop.core_start",
                "loop.core_end",
                "loop.varying",
            ],
            "inkling2",
        ),
    };
    for key in others {
        if split.value(&split.arch_key(key)).is_some() {
            return Err(metadata(
                split,
                key,
                format!("is the {word} dialect's loop key, and this file is not"),
            ));
        }
    }
    Ok(())
}

/// `attention.log_scale_floor`: a count above 0 (the reference reads it required and refuses 0,
/// src/models/inkling2.cpp:14, 39).
fn log_floor(split: &Split) -> Result<u32, PlacementError> {
    u32::try_from(positive(split, "attention.log_scale_floor")?).map_err(|_| {
        metadata(
            split,
            "attention.log_scale_floor",
            "does not fit u32".to_string(),
        )
    })
}

/// `attention.log_scale_alpha`: finite, 0 allowed (the scale is then 1 everywhere).
fn log_alpha(split: &Split) -> Result<f32, PlacementError> {
    let alpha = meta_f32(split, "attention.log_scale_alpha")?;
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(metadata(
            split,
            "attention.log_scale_alpha",
            format!("is {alpha}, not a finite value at or above 0"),
        ));
    }
    Ok(alpha)
}

/// `shortconv.l_cache`: at least 2 taps, at most 64 (src/models/inkling2.cpp:39-40).
fn conv(split: &Split) -> Result<usize, PlacementError> {
    let conv = positive(split, "shortconv.l_cache")?;
    if !(2..=64).contains(&conv) {
        return Err(metadata(
            split,
            "shortconv.l_cache",
            format!("is {conv}, outside the 2 to 64 the reference runs"),
        ));
    }
    Ok(conv)
}

/// A per-block count array of `n_blocks` values as the stored layers' own: a visit's value must
/// be its layer's first visit's (llama-model.cpp:1703-1711, under [`Dialect::Unsloth`] alone),
/// refused naming the key, the block and both values, and a stored layer's value its first
/// visit's.
fn stored_counts(
    split: &Split,
    key: &str,
    n_blocks: usize,
    loop_: &Option<Loop>,
    n_layer: usize,
) -> Result<Vec<usize>, PlacementError> {
    let values = per_layer_count(split, key, n_blocks)?;
    visits_agree(split, key, &values, loop_)?;
    Ok(stored_layer_values(split, &values, loop_, n_layer))
}

/// `attention.sliding_window_pattern`: one bool a block, required an array
/// (src/models/inkling2.cpp:33-38), the stored layers' own under the visits' agreement.
fn stored_pattern(
    split: &Split,
    n_blocks: usize,
    loop_: &Option<Loop>,
    n_layer: usize,
) -> Result<Vec<bool>, PlacementError> {
    let key = "attention.sliding_window_pattern";
    let items = meta_arr(split, key)?;
    if items.len() != n_blocks {
        return Err(metadata(
            split,
            key,
            format!("has {} values for {n_blocks} blocks", items.len()),
        ));
    }
    let values: Vec<bool> = items
        .iter()
        .enumerate()
        .map(|(l, v)| {
            v.as_bool()
                .ok_or_else(|| metadata(split, key, format!("has no bool at block {l}")))
        })
        .collect::<Result<_, _>>()?;
    visits_agree(split, key, &values, loop_)?;
    Ok(stored_layer_values(split, &values, loop_, n_layer))
}

/// The stored layers' values of a per-block array: under [`Dialect::Unsloth`] the blocks are the
/// visits, a stored layer reading its first visit's.
fn stored_layer_values<T: Copy>(
    split: &Split,
    values: &[T],
    loop_: &Option<Loop>,
    n_layer: usize,
) -> Vec<T> {
    if dialect_of(split) == Some(Dialect::Unsloth) {
        first_visits(loop_, n_layer)
            .into_iter()
            .map(|v| values[v])
            .collect()
    } else {
        values[..n_layer].to_vec()
    }
}

/// The first visit of each of the `n_layer` stored layers under [`Dialect::Unsloth`]
/// (llama-hparams.cpp:381-383): the layer's own index inside the core, its index past the core's
/// repeats after it. The other dialects' blocks are the stored layers.
fn first_visits(loop_: &Option<Loop>, n_layer: usize) -> Vec<usize> {
    match loop_ {
        None => (0..n_layer).collect(),
        Some(loop_) => {
            let extra = (loop_.passes - 1) * (loop_.core_end - loop_.core_start);
            (0..n_layer)
                .map(|l| if l < loop_.core_end { l } else { l + extra })
                .collect()
        }
    }
}

/// A visit's value is its layer's first visit's, under [`Dialect::Unsloth`] alone.
fn visits_agree<T: PartialEq + std::fmt::Debug>(
    split: &Split,
    key: &str,
    values: &[T],
    loop_: &Option<Loop>,
) -> Result<(), PlacementError> {
    let Some(loop_) = loop_ else {
        return Ok(());
    };
    if dialect_of(split) != Some(Dialect::Unsloth) {
        return Ok(());
    }
    let core = loop_.core_end - loop_.core_start;
    let extra = (loop_.passes - 1) * core;
    for v in loop_.core_end..loop_.core_end + extra {
        let layer = loop_.core_start + (v - loop_.core_start) % core;
        if values[v] != values[layer] {
            return Err(metadata(
                split,
                key,
                format!(
                    "is {:?} at block {v} and {:?} at block {layer}, two visits of one layer",
                    values[v], values[layer]
                ),
            ));
        }
    }
    Ok(())
}

// ------------------------------------------------------------ the description's router

/// The router every Inkling 2 layer runs: the sigmoid score over the f32 logits with the
/// selection bias, the kept weights over their sum guarded at llama.cpp's `1e-20`
/// (`expert_weights_norm_eps`, src/models/inkling2.cpp:64-65) and scaled by the file's factor.
#[must_use]
pub fn router(bias: bool, scale: f32) -> models::Router {
    models::Router {
        score: Score::Sigmoid,
        bias,
        norm: RouterNorm::SelectedSum,
        scale,
        hash: false,
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::{Dialect, Hparams, Loop, unread_keys};
    use crate::arch::synthetic::{V, header_shaped};
    use gguf::Split;

    /// The widths of [`keys`]: 6 stored layers, 10 visits (the core [2, 4) of a 3-pass loop),
    /// 64 wide, 4 query heads over 2 key heads of 16, a windowless layer at 2 and its repeats,
    /// 8 experts of 32, the top 2, a bias bank of 4 rows 32 wide.
    pub(in crate::arch::inkling2) const E: u64 = 64;
    pub(in crate::arch::inkling2) const HEADS: u64 = 4;
    pub(in crate::arch::inkling2) const KV: u64 = 2;
    pub(in crate::arch::inkling2) const D: u64 = 16;
    pub(in crate::arch::inkling2) const LAYERS: usize = 6;
    pub(in crate::arch::inkling2) const VISITS: usize = 10;
    pub(in crate::arch::inkling2) const CORE: [usize; 2] = [2, 4];
    pub(in crate::arch::inkling2) const GLOBALS: [usize; 1] = [2];
    pub(in crate::arch::inkling2) const REL: u64 = 4;
    pub(in crate::arch::inkling2) const EXTENT: u64 = 32;
    pub(in crate::arch::inkling2) const SWA: u64 = 16;
    pub(in crate::arch::inkling2) const CONV: u64 = 4;
    pub(in crate::arch::inkling2) const BLOCK: u64 = 2;
    pub(in crate::arch::inkling2) const N_EXP: u64 = 8;
    pub(in crate::arch::inkling2) const EFF: u64 = 32;
    pub(in crate::arch::inkling2) const RANK: u64 = 8;
    /// The two-token vocabulary of the synthetic header (the unpadded size, the padded 4).
    pub(in crate::arch::inkling2) const VOCAB: u64 = 2;
    pub(in crate::arch::inkling2) const VOCAB_PADDED: u64 = 4;

    /// The stored layer of visit `v` (llama-hparams.cpp:370-379): itself before the core, the
    /// core's repeats inside, itself less the core's repeats after.
    fn layer_of(v: usize) -> usize {
        let extra = 4;
        match v {
            v if v < CORE[1] => v,
            v if v < CORE[1] + extra => CORE[0] + (v - CORE[0]) % 2,
            v => v - extra,
        }
    }

    /// The pattern of the visits: layer 2 and its repeats windowless, the rest windowed.
    fn pattern() -> Vec<bool> {
        (0..VISITS)
            .map(|v| !GLOBALS.contains(&layer_of(v)))
            .collect()
    }

    /// The keys of the released file, at this table's widths, `looped` naming whether the loop
    /// keys ride along: without them the blocks are the stored layers and the pattern one a
    /// layer, with them the visits and one a visit.
    pub(in crate::arch::inkling2) fn keys(looped: bool) -> Vec<(&'static str, V)> {
        let mut kv = vec![
            (
                "block_count",
                V::U32(if looped { VISITS as u32 } else { LAYERS as u32 }),
            ),
            ("context_length", V::U32(4096)),
            ("embedding_length", V::U32(E as u32)),
            ("vocab_size", V::U32(VOCAB_PADDED as u32)),
            ("vocab_size_unpadded", V::U32(VOCAB as u32)),
            ("attention.head_count", V::U32(HEADS as u32)),
            ("attention.head_count_kv", V::U32(KV as u32)),
            ("attention.key_length", V::U32(D as u32)),
            ("attention.value_length", V::U32(D as u32)),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-6)),
            ("attention.sliding_window", V::U32(SWA as u32)),
            (
                "attention.sliding_window_pattern",
                V::Bools(if looped {
                    pattern()
                } else {
                    (0..LAYERS).map(|l| !GLOBALS.contains(&l)).collect()
                }),
            ),
            ("attention.relative_length", V::U32(REL as u32)),
            ("attention.relative_extent", V::U32(EXTENT as u32)),
            ("attention.qk_clip", V::F32(22.627_417)),
            ("attention.relative_clip", V::F32(8.0)),
            ("attention.log_scale_floor", V::U32(8192)),
            ("attention.log_scale_alpha", V::F32(0.1)),
            ("attn_res.block_size", V::U32(BLOCK as u32)),
            ("shortconv.l_cache", V::U32(CONV as u32)),
            ("expert_count", V::U32(N_EXP as u32)),
            ("expert_used_count", V::U32(2)),
            ("expert_feed_forward_length", V::U32(EFF as u32)),
            ("expert_weights_scale", V::F32(8.0)),
            ("final_logit_softcapping", V::F32(30.0)),
            ("activation.situ_beta", V::F32(4.0)),
            ("activation.situ_linear_beta", V::F32(25.0)),
        ];
        if looped {
            kv.extend([
                ("loop.core_start", V::U32(CORE[0] as u32)),
                ("loop.core_end", V::U32(CORE[1] as u32)),
                ("num_loops", V::U32(3)),
                ("loop.gate_rank", V::U32(RANK as u32)),
            ]);
        }
        kv
    }

    /// The master dialect's loop keys, at [`keys`]' widths (`loop.gate_rank` both dialects
    /// share, [`keys`] already holds it).
    fn loop_keys_master() -> Vec<(&'static str, V)> {
        vec![
            ("loop.start", V::U32(CORE[0] as u32)),
            ("loop.end", V::U32(CORE[1] as u32)),
            ("loop.count", V::U32(3)),
        ]
    }

    /// The released file's tensor table at [`keys`]' widths: the model level, then each stored
    /// layer by its kind, `looped` naming whether the core layers carry the loop's tensors.
    pub(in crate::arch::inkling2) fn tensors(looped: bool) -> Vec<(String, Vec<u64>)> {
        tensors_of("inkling2", looped)
    }

    /// [`tensors`] under a dialect's architecture word.
    fn tensors_of(arch: &str, looped: bool) -> Vec<(String, Vec<u64>)> {
        let mut t: Vec<(String, Vec<u64>)> = vec![
            ("token_embd.weight".into(), vec![E, VOCAB_PADDED]),
            ("token_embd_norm.weight".into(), vec![E]),
            ("output.weight".into(), vec![E, VOCAB_PADDED]),
            ("output_norm.weight".into(), vec![E]),
            ("output_res_query.weight".into(), vec![E]),
        ];
        if arch == "inkling2" && looped {
            t.push((
                "loop_res_query.weight".into(),
                vec![E, 2 * CORE.len() as u64],
            ));
            t.push((
                "loop_kv_res_query.weight".into(),
                vec![E, 2 * CORE.len() as u64],
            ));
        }
        for l in 0..LAYERS {
            let mut layer: Vec<(String, Vec<u64>)> = [
                ("attn_q.weight", vec![E, D * HEADS]),
                ("attn_k.weight", vec![E, D * KV]),
                ("attn_v.weight", vec![E, D * KV]),
                ("attn_output.weight", vec![D * HEADS, E]),
                ("attn_gate.weight", vec![E, D * HEADS]),
                ("attn_q_clip.weight", vec![D, HEADS]),
                ("attn_sink_proj.weight", vec![E, HEADS]),
                ("attn_rel.weight", vec![E, REL * HEADS]),
                (
                    "attn_rel_bias.weight",
                    vec![REL, if GLOBALS.contains(&l) { EXTENT } else { SWA }],
                ),
                ("attn_rel_clip.weight", vec![REL, HEADS]),
                ("attn_k_conv.weight", vec![CONV, D * KV]),
                ("attn_v_conv.weight", vec![CONV, D * KV]),
                ("attn_post_conv.weight", vec![CONV, E]),
                ("ffn_post_conv.weight", vec![CONV, E]),
            ]
            .into_iter()
            .map(|(s, d)| (format!("blk.{l}.{s}"), d))
            .collect();
            if GLOBALS.contains(&l) {
                layer.extend(
                    [
                        ("attn_q_inp_norm.weight", vec![E]),
                        ("attn_k_inp_norm.weight", vec![E]),
                        ("attn_v_inp_norm.weight", vec![E]),
                        ("attn_k_conv_proj.weight", vec![E, CONV * KV]),
                        ("attn_v_conv_proj.weight", vec![E, CONV * KV]),
                    ]
                    .into_iter()
                    .map(|(s, d)| (format!("blk.{l}.{s}"), d)),
                );
            } else {
                layer.push((format!("blk.{l}.attn_norm.weight"), vec![E]));
            }
            if l > 0 {
                layer.push((
                    format!("blk.{l}.attn_res_query.weight"),
                    vec![E, u64::from(GLOBALS.contains(&l)) * 2 + 1],
                ));
            }
            layer.extend(
                [
                    ("ffn_res_query.weight", vec![E]),
                    ("ffn_norm.weight", vec![E]),
                    ("ffn_gate_inp.weight", vec![E, N_EXP]),
                    ("exp_probs_b.bias", vec![N_EXP]),
                    ("ffn_gate_up_exps.weight", vec![E, 2 * EFF, N_EXP]),
                    ("ffn_down_exps.weight", vec![EFF, E, N_EXP]),
                ]
                .into_iter()
                .map(|(s, d)| (format!("blk.{l}.{s}"), d)),
            );
            if looped && (CORE[0]..CORE[1]).contains(&l) {
                let ctrls: Vec<(String, Vec<u64>)> = match arch {
                    "inkling2" => ["attn", "k", "v", "mlp"]
                        .into_iter()
                        .flat_map(|m| ["down", "gain", "forget"].map(move |p| (m, p)))
                        .map(|(m, p)| {
                            let dims = if p == "down" {
                                vec![2 * E, RANK]
                            } else {
                                vec![RANK, E]
                            };
                            (format!("blk.{l}.loop_{m}_{p}.weight"), dims)
                        })
                        .collect(),
                    _ => ["q", "k", "v", "ffn"]
                        .into_iter()
                        .flat_map(|m| ["down", "gain", "forget"].map(move |p| (m, p)))
                        .map(|(m, p)| {
                            let dims = if p == "down" {
                                vec![2 * E, RANK]
                            } else {
                                vec![RANK, E]
                            };
                            (format!("blk.{l}.loop_{m}_ctrl_{p}.weight"), dims)
                        })
                        .collect(),
                };
                layer.extend(ctrls);
                if arch != "inkling2" {
                    layer.push((format!("blk.{l}.loop_msg_query.weight"), vec![E, 4]));
                }
            }
            t.extend(layer);
        }
        t
    }

    /// [`keys`] with `key` replaced by `v`, or dropped when `v` is `None`.
    pub(in crate::arch::inkling2) fn with(
        key: &'static str,
        v: Option<V>,
    ) -> Vec<(&'static str, V)> {
        with_in(keys(true), key, v)
    }

    /// [`with`] over `kv`, taking it.
    fn with_in(
        mut kv: Vec<(&'static str, V)>,
        key: &'static str,
        v: Option<V>,
    ) -> Vec<(&'static str, V)> {
        kv.retain(|(k, _)| *k != key);
        if let Some(v) = v {
            kv.push((key, v));
        }
        kv
    }

    /// The reader's verdict on a header of `kv` and `tensors` under the architecture word `arch`.
    fn read_of(
        tag: &str,
        arch: &str,
        kv: &[(&'static str, V)],
        tensors: &[(String, Vec<u64>)],
    ) -> Result<Hparams, String> {
        let global = [
            ("tokenizer.ggml.pre", V::Str("inkling2")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = header_shaped(tag, arch, kv, &global, tensors);
        let got = Split::open(&path)
            .map_err(|e| e.to_string())
            .and_then(|s| Hparams::read(&s).map_err(|e| e.to_string()));
        let _ = std::fs::remove_file(&path);
        got
    }

    /// The reader's verdict on an unsloth header of `kv` and `tensors`.
    fn read(
        tag: &str,
        kv: &[(&'static str, V)],
        tensors: &[(String, Vec<u64>)],
    ) -> Result<Hparams, String> {
        read_of(tag, "inkling2", kv, tensors)
    }

    /// One refusal case: a tag, the header's keys, the tensor shapes and the name the refusal
    /// must hold.
    type Case = (
        &'static str,
        Vec<(&'static str, V)>,
        Vec<(String, Vec<u64>)>,
        &'static str,
    );

    /// What is wrong with the verdict on `tag`'s header: it read, or the error lacks `want`.
    fn wrong_verdict(
        tag: &str,
        kv: &[(&'static str, V)],
        tensors: &[(String, Vec<u64>)],
        want: &str,
    ) -> Option<String> {
        match read(tag, kv, tensors) {
            Ok(_) => Some(format!("{tag}: read; expected {want:?}")),
            Err(e) if e.contains(want) => None,
            Err(e) => Some(format!("{tag}: want {want:?} in: {e}")),
        }
    }

    /// The header the table gives reads, to the values its keys hold: six stored layers of ten
    /// visits, the core [2, 4) three times, layer 2 windowless with its repeats, the loop's keys
    /// and no default taken.
    #[test]
    fn a_small_header_reads() {
        let hp = read("small", &keys(true), &tensors(true)).expect("the header reads");
        assert_eq!(hp.dialect, Dialect::Unsloth);
        assert_eq!((hp.n_blocks, hp.n_layer), (VISITS, LAYERS));
        assert_eq!(
            hp.loop_,
            Some(Loop {
                core_start: 2,
                core_end: 4,
                passes: 3,
                gate_rank: 8
            })
        );
        assert!(hp.in_core(2) && hp.in_core(3) && !hp.in_core(1) && !hp.in_core(4));
        assert_eq!(
            (hp.n_embd, hp.head_dim, hp.n_ctx_train),
            (E as usize, D as usize, 4096)
        );
        assert_eq!((hp.n_vocab, hp.n_vocab_unpadded), (4, 2));
        assert_eq!((hp.n_expert, hp.n_used[0], hp.expert_ff[0]), (8, 2, 32));
        assert_eq!(
            (hp.d_rel, hp.extent, hp.swa, hp.block, hp.conv),
            (
                REL as usize,
                EXTENT as usize,
                SWA as usize,
                BLOCK as usize,
                CONV as usize
            )
        );
        assert_eq!((hp.qk_clip, hp.rel_clip), (22.627_417, 8.0));
        assert_eq!((hp.log_floor, hp.log_alpha), (8192, 0.1));
        assert_eq!((hp.weights_scale, hp.softcap), (8.0, 30.0));
        assert_eq!(hp.windowed, [true, true, false, true, true, true]);
        assert!(hp.defaults.is_empty(), "{:?}", hp.defaults);
    }

    /// A file without the loop's keys reads as one pass over every layer: the Small's shape, the
    /// block count the stored layers, the pattern one value a layer and the layer 2 windowless.
    #[test]
    fn a_file_without_the_loop_reads_one_pass() {
        let kv = one_pass_keys();
        let hp = read("small-noloop", &kv, &tensors(false)).expect("the header reads");
        assert_eq!(hp.loop_, None);
        assert_eq!((hp.n_blocks, hp.n_layer), (LAYERS, LAYERS));
        assert_eq!(hp.windowed, [true, true, false, true, true, true]);
    }

    /// The keys of a loopless file: [`keys`] without the loop's.
    fn one_pass_keys() -> Vec<(&'static str, V)> {
        keys(false)
    }

    /// Every refusal names its key or tensor. Mutants: a check dropped, a bound loosened, the
    /// wrong key named.
    #[test]
    fn the_reader_refuses_by_name() {
        let cases: Vec<Case> = vec![
            (
                "one-pass",
                with("num_loops", Some(V::U32(1))),
                tensors(true),
                "num_loops",
            ),
            (
                "no-start",
                with("loop.core_start", None),
                tensors(true),
                "loop.core_start",
            ),
            (
                "no-rank",
                with("loop.gate_rank", None),
                tensors(true),
                "loop.gate_rank",
            ),
            (
                "start-at-zero",
                with("loop.core_start", Some(V::U32(0))),
                tensors(true),
                "loop.core_start",
            ),
            (
                "start-misaligned",
                with("loop.core_start", Some(V::U32(3))),
                tensors(true),
                "whole blocks",
            ),
            (
                "end-misaligned",
                with("loop.core_end", Some(V::U32(5))),
                tensors(true),
                "inside the trunk",
            ),
            (
                "core-past-the-trunk",
                with("loop.core_end", Some(V::U32(8))),
                tensors(true),
                "inside the trunk",
            ),
            (
                "master-keys",
                with_in(
                    with_in(keys(true), "num_loops", None),
                    "loop.start",
                    Some(V::U32(2)),
                ),
                tensors(true),
                "loop.start",
            ),
            (
                "varying",
                with("loop.varying", Some(V::Bool(true))),
                tensors(true),
                "loop.varying",
            ),
            (
                "no-relative",
                with("attention.relative_length", None),
                tensors(false),
                "attention.relative_length",
            ),
            (
                "no-extent",
                with("attention.relative_extent", None),
                tensors(false),
                "attention.relative_extent",
            ),
            (
                "no-floor",
                with("attention.log_scale_floor", None),
                tensors(false),
                "attention.log_scale_floor",
            ),
            (
                "zero-floor",
                with("attention.log_scale_floor", Some(V::U32(0))),
                tensors(false),
                "attention.log_scale_floor",
            ),
            (
                "alpha-negative",
                with("attention.log_scale_alpha", Some(V::F32(-0.1))),
                tensors(false),
                "attention.log_scale_alpha",
            ),
            (
                "no-clip",
                with("attention.qk_clip", None),
                tensors(false),
                "attention.qk_clip",
            ),
            (
                "no-rel-clip",
                with("attention.relative_clip", None),
                tensors(false),
                "attention.relative_clip",
            ),
            (
                "conv-one",
                with("shortconv.l_cache", Some(V::U32(1))),
                tensors(false),
                "shortconv.l_cache",
            ),
            (
                "conv-huge",
                with("shortconv.l_cache", Some(V::U32(65))),
                tensors(false),
                "shortconv.l_cache",
            ),
            (
                "no-block",
                with("attn_res.block_size", None),
                tensors(false),
                "attn_res.block_size",
            ),
            (
                "no-unpadded",
                with("vocab_size_unpadded", None),
                tensors(false),
                "vocab_size_unpadded",
            ),
            (
                "unpadded-over",
                with("vocab_size_unpadded", Some(V::U32(8))),
                tensors(false),
                "over the padded vocab_size",
            ),
            (
                "unpadded-past-the-tokens",
                with("vocab_size_unpadded", Some(V::U32(3))),
                tensors(false),
                "token list holds 2",
            ),
            (
                "no-expert-ff",
                with("expert_feed_forward_length", Some(V::U32(0))),
                tensors(false),
                "expert_feed_forward_length",
            ),
            (
                "used-over",
                with("expert_used_count", Some(V::U32(16))),
                tensors(false),
                "expert_used_count",
            ),
            (
                "pattern-scalar",
                with("attention.sliding_window_pattern", Some(V::Bool(true))),
                tensors(false),
                "attention.sliding_window_pattern",
            ),
            (
                "pattern-short",
                with(
                    "attention.sliding_window_pattern",
                    Some(V::Bools(vec![true; 4])),
                ),
                tensors(false),
                "values for 10 blocks",
            ),
            (
                "visits-disagree",
                with(
                    "attention.sliding_window_pattern",
                    Some(V::Bools({
                        let mut p = pattern();
                        p[4] = true;
                        p
                    })),
                ),
                tensors(true),
                "two visits of one layer",
            ),
            (
                "value-width",
                with("attention.value_length", Some(V::U32(8))),
                tensors(false),
                "attention.value_length",
            ),
            (
                "no-embd-norm",
                keys(false),
                without_noloop("token_embd_norm.weight"),
                "token_embd_norm.weight",
            ),
            (
                "no-final-query",
                keys(false),
                without_noloop("output_res_query.weight"),
                "output_res_query.weight",
            ),
            (
                "no-loop-query",
                keys(true),
                without("loop_res_query.weight"),
                "loop_res_query.weight",
            ),
            (
                "no-ctrl",
                keys(true),
                without("blk.2.loop_attn_down.weight"),
                "blk.2.loop_attn_down.weight",
            ),
            (
                "ctrl-on-a-stem-layer",
                keys(true),
                shaped("blk.1.loop_attn_down.weight", vec![2 * E, RANK]),
                "blk.1.loop_attn_down.weight",
            ),
            (
                "ctrl-rank",
                keys(true),
                shaped("blk.2.loop_attn_down.weight", vec![2 * E, RANK + 1]),
                "blk.2.loop_attn_down.weight",
            ),
            (
                "loop-query-rows",
                keys(true),
                shaped("loop_res_query.weight", vec![E, 6]),
                "loop_res_query.weight",
            ),
            (
                "queries-narrow",
                keys(false),
                shaped_noloop("blk.3.attn_res_query.weight", vec![E, 2]),
                "blk.3.attn_res_query.weight",
            ),
            (
                "gate-up-not-fused",
                keys(false),
                shaped_noloop("blk.0.ffn_gate_up_exps.weight", vec![E, EFF, N_EXP]),
                "blk.0.ffn_gate_up_exps.weight",
            ),
            (
                "rel-bias-narrow",
                keys(false),
                shaped_noloop("blk.0.attn_rel_bias.weight", vec![REL, 8]),
                "blk.0.attn_rel_bias.weight",
            ),
            (
                "global-without-input-norms",
                keys(false),
                without_noloop("blk.2.attn_q_inp_norm.weight"),
                "blk.2.attn_q_inp_norm.weight",
            ),
            (
                "local-with-input-norm",
                keys(false),
                shaped_noloop("blk.0.attn_q_inp_norm.weight", vec![E]),
                "blk.0.attn_q_inp_norm.weight",
            ),
            (
                "layer0-query",
                keys(false),
                shaped_noloop("blk.0.attn_res_query.weight", vec![E, 1]),
                "blk.0.attn_res_query.weight",
            ),
        ];
        let wrong: Vec<String> = cases
            .iter()
            .filter_map(|(tag, kv, t, want)| wrong_verdict(tag, kv, t, want))
            .collect();
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    /// The master dialect's keys at this table's widths: the loopless keys' shared ones, the
    /// stored layers' own block count and pattern, and the master's loop keys.
    pub(in crate::arch::inkling2) fn master_keys() -> Vec<(&'static str, V)> {
        let mut kv: Vec<(&'static str, V)> = keys(true)
            .into_iter()
            .filter(|(k, _)| !matches!(*k, "loop.core_start" | "loop.core_end" | "num_loops"))
            .collect();
        kv.extend(loop_keys_master());
        kv = with_in(kv, "block_count", Some(V::U32(LAYERS as u32)));
        with_in(
            kv,
            "attention.sliding_window_pattern",
            Some(V::Bools(
                (0..LAYERS).map(|l| !GLOBALS.contains(&l)).collect(),
            )),
        )
    }

    /// The master dialect's tensor table at [`keys`]' widths.
    pub(in crate::arch::inkling2) fn master_tensors() -> Vec<(String, Vec<u64>)> {
        tensors_of(crate::arch::INKLING2_LOOP, true)
    }

    /// The master dialect's header reads into the same loop: the stored layers the block count,
    /// its own keys and stems, no model-level query — and its own refusals.
    #[test]
    fn a_master_dialect_header_reads() {
        let kv = master_keys();
        let t = master_tensors();
        let hp = read_of("master", crate::arch::INKLING2_LOOP, &kv, &t).expect("the header reads");
        assert_eq!(hp.dialect, Dialect::Master);
        assert_eq!((hp.n_blocks, hp.n_layer), (LAYERS, LAYERS));
        assert_eq!(
            hp.loop_,
            Some(Loop {
                core_start: 2,
                core_end: 4,
                passes: 3,
                gate_rank: 8
            })
        );
        assert!(hp.defaults.is_empty(), "{:?}", hp.defaults);
        // The unsloth keys are the other dialect's, and so is the varying flag; the master's own
        // keys are required. Each case builds its keys anew: the synthetic value enum is no
        // clone. A stem of the other dialect refuses in the classifier, the spec tests'
        // (`the_dialects_stems_do_not_mix`).
        let base = || master_keys();
        for (tag, kv, t, want) in [
            (
                "master-unsloth-key",
                with_in(base(), "num_loops", Some(V::U32(3))),
                t.clone(),
                "num_loops",
            ),
            (
                "master-varying",
                with_in(base(), "loop.varying", Some(V::Bool(true))),
                t.clone(),
                "loop.varying",
            ),
            (
                "master-no-count",
                with_in(base(), "loop.count", None),
                t.clone(),
                "loop.count",
            ),
            (
                "master-one-pass",
                with_in(base(), "loop.count", Some(V::U32(1))),
                t.clone(),
                "loop.count",
            ),
            (
                "master-past-the-trunk",
                with_in(base(), "loop.end", Some(V::U32(7))),
                t.clone(),
                "loop.end",
            ),
            (
                "master-no-msg-query",
                base(),
                without_in(&t, "blk.2.loop_msg_query.weight"),
                "blk.2.loop_msg_query.weight",
            ),
        ] {
            let e = read_of(tag, crate::arch::INKLING2_LOOP, &kv, &t)
                .err()
                .unwrap_or_else(|| panic!("{tag}: read"));
            assert!(e.contains(want), "{tag}: {e}");
        }
        // A master stem in an unsloth file reads here — the classifier's mixing refusal is the
        // spec tests' (`the_dialects_stems_do_not_mix`).
        read("unsloth-master-stem", &keys(true), &{
            let mut mixed = tensors(true);
            mixed.push(("blk.2.loop_q_ctrl_down.weight".into(), vec![2 * E, RANK]));
            mixed
        })
        .expect("the hparams read stops before the classifier");
    }

    /// Another architecture's file is refused by `general.architecture`, whatever its keys hold.
    #[test]
    fn another_architecture_is_refused_by_name() {
        for arch in ["deepseek41", "glm5next", "raon1"] {
            let got = read_of("arch", arch, &keys(false), &tensors(false));
            match got {
                Ok(_) => panic!("{arch}: read"),
                Err(e) => assert!(
                    e.contains(gguf::GENERAL_ARCHITECTURE) && e.contains(arch),
                    "{arch}: {e}"
                ),
            }
        }
    }

    /// The reader lists, and does not refuse, a key it does not read: the two converter
    /// vintages' stale spellings of keys it reads under another name.
    #[test]
    fn an_unread_key_is_listed() {
        let mut kv = keys(false);
        kv.push(("log_scaling_n_floor", V::U32(128_000)));
        kv.push(("d_rel", V::U32(16)));
        kv.push(("rope.dimension_count", V::U32(0)));
        let global = [
            ("tokenizer.ggml.pre", V::Str("inkling2")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = header_shaped("inkling2-unread", "inkling2", &kv, &global, &tensors(false));
        let split = Split::open(&path).expect("opens");
        let mut unread = unread_keys(&split);
        let _ = std::fs::remove_file(&path);
        unread.sort();
        assert_eq!(
            unread,
            vec![
                "inkling2.d_rel".to_string(),
                "inkling2.log_scaling_n_floor".to_string(),
                "inkling2.rope.dimension_count".to_string(),
            ]
        );
    }

    /// The tensors with `name` dropped, of the looped table.
    fn without(name: &str) -> Vec<(String, Vec<u64>)> {
        without_in(&tensors(true), name)
    }

    /// The loopless tensors with `name` dropped.
    fn without_noloop(name: &str) -> Vec<(String, Vec<u64>)> {
        without_in(&tensors(false), name)
    }

    /// The loopless tensors with `name` given `dims`, or added when the table has none.
    fn shaped_noloop(name: &str, dims: Vec<u64>) -> Vec<(String, Vec<u64>)> {
        let mut t = without_noloop(name);
        t.push((name.to_string(), dims));
        t
    }

    /// The tensors with `name` dropped.
    fn without_in(t: &[(String, Vec<u64>)], name: &str) -> Vec<(String, Vec<u64>)> {
        t.iter().filter(|(n, _)| n != name).cloned().collect()
    }

    /// The tensors with `name` given `dims`, or added when the table has none.
    fn shaped(name: &str, dims: Vec<u64>) -> Vec<(String, Vec<u64>)> {
        let mut t = without(name);
        t.push((name.to_string(), dims));
        t
    }
}
