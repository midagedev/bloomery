//! Every `mimo2.*` key the description reads — the per-layer KV-head and
//! window-pattern arrays as arrays, not scalars, because a full-attention
//! layer and a sliding-window one carry different KV-head counts and rope
//! bases in one file. A key the loader requires is required here; a key it
//! reads with a fallback keeps that fallback, recorded in
//! [`Hparams::defaults`] with the line that sets it. Each layer's tensors
//! are held to the widths its arrays give it ([`Hparams::read`]). The
//! dialect's authority is llama.cpp's `src/models/mimo2.cpp`; the line
//! numbers cited are that file's unless another file is named.

use gguf::{Split, Value};

use super::roles::{
    DENSE, MIXER, MOE, NEXTN, QKV_FUSED, QKV_SPLIT, SELECTION_BIAS, Stem, required,
};
use crate::arch::{meta_arr, meta_f32, meta_u64, meta_usize, metadata, n_vocab};
use crate::placement::PlacementError;

/// llama.cpp's `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`, which the mimo2
/// builder passes as a constant (:220).
const SIGMOID: u64 = 2;

/// What a layer attends, from `attention.sliding_window_pattern`: 0 a full
/// layer, 1 a window of `sliding_window` positions (conversion/mimo.py:160).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Every position before it (pattern 0).
    Full,
    /// The last `sliding_window` positions (pattern 1).
    Swa,
}

/// The hyperparameters of one mimo2 file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// `block_count`: the trunk and the next-token layers.
    pub n_layer: usize,
    /// The trunk's layers: `block_count` less `nextn_predict_layers`.
    pub n_trunk: usize,
    /// `embedding_length`.
    pub n_embd: usize,
    /// `attention.head_count`: query heads, every layer's.
    pub n_head: usize,
    /// `attention.head_count_kv`, per layer: 4 on this file's full layers,
    /// 8 on its window layers and next-token blocks.
    pub kv_heads: Vec<usize>,
    /// `attention.key_length`: key and score values per head.
    pub head_k: usize,
    /// `attention.value_length`: value values per head, narrower than the
    /// key head in this architecture.
    pub head_v: usize,
    /// `attention.sliding_window`: positions a [`Kind::Swa`] layer attends.
    pub window: usize,
    /// Every layer's kind, in order.
    pub kinds: Vec<Kind>,
    /// `rope.dimension_count`: rotated values of a q or k head.
    pub rope_dims: usize,
    /// `rope.freq_base`: the full layers' θ.
    pub rope_base: f32,
    /// `rope.freq_base_swa`: the window layers' θ, another base than the
    /// full layers'.
    pub rope_base_swa: f32,
    /// `attention.value_scale`: the multiplier on every layer's value rows.
    pub value_scale: f32,
    /// `attention.layer_norm_rms_epsilon`.
    pub rms_eps: f32,
    /// `context_length`.
    pub n_ctx_train: usize,
    /// `vocab_size`, else the token list's length.
    pub n_vocab: usize,
    /// `expert_count`; 0 on a file no layer of which routes.
    pub n_expert: usize,
    /// `expert_used_count`; 0 where `n_expert` is.
    pub n_used: usize,
    /// `expert_feed_forward_length`; 0 where `n_expert` is.
    pub expert_ff: usize,
    /// `feed_forward_length`, the dense layers' width; `None` on a file no
    /// layer of which is dense.
    pub dense_ff: Option<usize>,
    /// `expert_weights_scale`.
    pub weights_scale: f32,
    /// The keys this read took a default for: the key, the value, the line.
    pub defaults: Vec<String>,
}

/// Every `mimo2.` key [`Hparams::read`] reads, by suffix.
const READ: &[&str] = &[
    "block_count",
    "nextn_predict_layers",
    "embedding_length",
    "context_length",
    "vocab_size",
    "feed_forward_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.value_scale",
    "attention.layer_norm_rms_epsilon",
    "attention.sliding_window",
    "attention.sliding_window_pattern",
    "rope.dimension_count",
    "rope.freq_base",
    "rope.freq_base_swa",
    "expert_count",
    "expert_used_count",
    "expert_feed_forward_length",
    "expert_gating_func",
    "expert_weights_scale",
    "expert_group_count",
    "expert_group_used_count",
];

/// The architecture keys and `general.sampling.*` keys of `split` that
/// [`Hparams::read`] does not read: listed, not refused.
#[must_use]
pub fn unread_keys(split: &Split) -> Vec<String> {
    let prefix = split.arch_key("");
    split
        .iter_kv()
        .map(|(k, _)| k)
        .filter(|k| match k.strip_prefix(prefix.as_str()) {
            Some(suffix) => !READ.contains(&suffix),
            None => k.starts_with("general.sampling."),
        })
        .map(str::to_string)
        .collect()
}

impl Hparams {
    /// Every value from `split`'s headers; a missing required key, a value
    /// the description cannot hold, a wrong-length per-layer array, or a
    /// layer whose tensors disagree with the arrays is an error naming the
    /// key or the tensor.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        let mut defaults = Vec::new();
        let n_layer = meta_usize(split, "block_count")?;
        let n_nextn = nextn(split, n_layer, &mut defaults)?;
        let n_trunk = n_layer - n_nextn;
        let n_embd = meta_usize(split, "embedding_length")?;
        let n_head = meta_usize(split, "attention.head_count")?;
        if n_head == 0 {
            return Err(metadata(split, "attention.head_count", "is 0"));
        }
        // The KV-head count is a per-layer array: 4 on the full-attention
        // layers of this release, 8 on the window ones (conversion/mimo.py:
        // 151-157), so a scalar broadcast is not this file's fact.
        let kv_heads = per_layer_count(split, "attention.head_count_kv", n_layer)?;
        for (l, &kv) in kv_heads.iter().enumerate() {
            if !n_head.is_multiple_of(kv) {
                return Err(metadata(
                    split,
                    "attention.head_count_kv",
                    format!("is {kv} at layer {l} for {n_head} query heads"),
                ));
            }
        }
        let head_k = positive(split, "attention.key_length")?;
        let head_v = positive(split, "attention.value_length")?;
        // The file always carries it; a missing key is not a scale of 1.
        let value_scale = meta_f32(split, "attention.value_scale")?;
        if !value_scale.is_finite() || value_scale <= 0.0 {
            return Err(metadata(
                split,
                "attention.value_scale",
                format!("is {value_scale}, not a finite positive multiplier"),
            ));
        }
        let rms_eps = meta_f32(split, "attention.layer_norm_rms_epsilon")?;
        let window = positive(split, "attention.sliding_window")?;
        let kinds = pattern(split, n_layer)?;
        let rope_dims = meta_usize(split, "rope.dimension_count")?;
        if rope_dims == 0 || rope_dims % 2 != 0 || rope_dims > head_k {
            return Err(metadata(
                split,
                "rope.dimension_count",
                format!("is {rope_dims}, for a key head of {head_k}"),
            ));
        }
        let rope_base = meta_f32(split, "rope.freq_base")?;
        let rope_base_swa = match split.value(&split.arch_key("rope.freq_base_swa")) {
            Some(_) => meta_f32(split, "rope.freq_base_swa")?,
            None => {
                defaults.push("rope.freq_base_swa = rope.freq_base (:10)".to_string());
                rope_base
            }
        };
        // A layer routes or runs one dense block by the tensors it carries
        // (mimo2.cpp:62-72, both branches TENSOR_NOT_REQUIRED).
        let routed = (0..n_trunk)
            .filter(|&l| {
                split
                    .find(&format!("blk.{l}.ffn_gate_inp.weight"))
                    .is_some()
            })
            .count();
        let dense = n_trunk - routed;
        let (n_expert, n_used, expert_ff) = if routed > 0 {
            let n_expert = meta_usize(split, "expert_count")?;
            let n_used = meta_usize(split, "expert_used_count")?;
            if n_expert == 0 || n_used == 0 || n_used > n_expert {
                return Err(metadata(
                    split,
                    "expert_used_count",
                    format!("is {n_used} of expert_count {n_expert}"),
                ));
            }
            let gating = match split.value(&split.arch_key("expert_gating_func")) {
                Some(_) => meta_u64(split, "expert_gating_func")?,
                None => {
                    defaults.push("expert_gating_func = sigmoid (:220)".to_string());
                    SIGMOID
                }
            };
            if gating != SIGMOID {
                return Err(metadata(
                    split,
                    "expert_gating_func",
                    format!("is {gating}; this reader reads the sigmoid router ({SIGMOID})"),
                ));
            }
            // The group stage runs only past one group (llama-graph.cpp
            // `build_moe_ffn`'s `n_expert_groups > 1`), which no mimo2 file
            // carries.
            for key in ["expert_group_count", "expert_group_used_count"] {
                if let Some(v) = optional_usize(split, key)?
                    && v != 1
                {
                    return Err(metadata(
                        split,
                        key,
                        format!("is {v}; the mimo2 router takes one group"),
                    ));
                }
            }
            (
                n_expert,
                n_used,
                one_width(split, "expert_feed_forward_length", ":8")?,
            )
        } else {
            (0, 0, 0)
        };
        let dense_ff = if dense + n_nextn > 0 {
            Some(one_width(split, "feed_forward_length", "llama-model.cpp")?)
        } else {
            None
        };
        let weights_scale = match split.value(&split.arch_key("expert_weights_scale")) {
            Some(_) => meta_f32(split, "expert_weights_scale")?,
            None => {
                // The builder passes the hparams default, whose scale node
                // the graph skips at 0 and at 1 alike (:219).
                defaults.push("expert_weights_scale = 1 (absent, no scale node)".to_string());
                1.0
            }
        };
        let hp = Hparams {
            n_layer,
            n_trunk,
            n_embd,
            n_head,
            kv_heads,
            head_k,
            head_v,
            window,
            kinds,
            rope_dims,
            rope_base,
            rope_base_swa,
            value_scale,
            rms_eps,
            n_ctx_train: meta_usize(split, "context_length")?,
            n_vocab: n_vocab(split)?,
            n_expert,
            n_used,
            expert_ff,
            dense_ff,
            weights_scale,
            defaults,
        };
        hp.tensors_agree(split)?;
        hp.dims_agree(split)?;
        Ok(hp)
    }

    /// Every layer carries its block's required tensors — the mixer's, one
    /// QKV form, one feed-forward kind, and a next-token block's own — and
    /// nothing of the other kinds': a window layer's sinks are the one
    /// optional tensor (llama.cpp loads them `TENSOR_NOT_REQUIRED`, mimo2.
    /// cpp:57), a full layer's none. The first missing or misplaced name,
    /// with how many more there are.
    fn tensors_agree(&self, split: &Split) -> Result<(), PlacementError> {
        let has = |l: usize, stem: &str| split.find(&format!("blk.{l}.{stem}")).is_some();
        let mut missing = Vec::new();
        let mut misplaced = Vec::new();
        for l in 0..self.n_layer {
            let nextn = l >= self.n_trunk;
            // The next-token blocks are dense-FFN blocks the MTP graph
            // asserts (:372); a trunk layer routes or runs dense by the
            // tensors it carries (:62-72).
            let routed = !nextn && has(l, "ffn_gate_inp.weight");
            let (own, other): (&[Stem], &[Stem]) = if nextn || !routed {
                (DENSE, MOE)
            } else {
                (MOE, DENSE)
            };
            let fused = has(l, QKV_FUSED.name);
            let split_qkv = QKV_SPLIT.iter().any(|s| has(l, s.name));
            let mut need: Vec<&str> = required(MIXER).collect();
            need.extend(match (fused, split_qkv) {
                (true, _) => vec![QKV_FUSED.name],
                (false, true) => QKV_SPLIT.iter().map(|s| s.name).collect(),
                // Neither form: both join the missing list, which names the
                // first.
                (false, false) => [QKV_FUSED.name]
                    .into_iter()
                    .chain(QKV_SPLIT.iter().map(|s| s.name))
                    .collect(),
            });
            need.extend(required(own));
            if nextn {
                need.extend(required(NEXTN));
            }
            missing.extend(
                need.iter()
                    .filter(|s| !has(l, s))
                    .map(|s| format!("blk.{l}.{s}")),
            );
            let mut banned: Vec<&str> = other
                .iter()
                .map(|s| s.name)
                .filter(|s| *s != "ffn_norm.weight")
                .collect();
            if fused {
                banned.extend(QKV_SPLIT.iter().map(|s| s.name));
            } else if split_qkv {
                banned.push(QKV_FUSED.name);
            }
            if self.kinds[l] == Kind::Full {
                banned.push("attn_sinks.weight");
            }
            misplaced.extend(
                banned
                    .iter()
                    .filter(|s| has(l, s))
                    .map(|s| format!("blk.{l}.{s}")),
            );
        }
        if let Some(name) = missing.first() {
            return Err(PlacementError::Tensor {
                name: name.clone(),
                detail: format!(
                    "is not in the file ({} tensor(s) the layers' blocks need are missing: {})",
                    missing.len(),
                    missing.join(", ")
                ),
            });
        }
        if let Some(name) = misplaced.first() {
            return Err(PlacementError::Tensor {
                name: name.clone(),
                detail: format!(
                    "is in the file, and its layer's block carries none ({} such: {})",
                    misplaced.len(),
                    misplaced.join(", ")
                ),
            });
        }
        Ok(())
    }

    /// The tensors whose dims the arrays name — the projections a layer's
    /// KV-head count and the two head widths size, the router's and stacks'
    /// expert count, the dense block's width and a next-token block's input
    /// projection — hold exactly those dims, as llama.cpp creates them with
    /// (:53-54, :68-71, :75). The first tensor of other dims, by name, with
    /// how many more there are.
    fn dims_agree(&self, split: &Split) -> Result<(), PlacementError> {
        let e = self.n_embd as u64;
        let mut off = Vec::new();
        for l in 0..self.n_layer {
            let nextn = l >= self.n_trunk;
            let routed = !nextn
                && split
                    .find(&format!("blk.{l}.ffn_gate_inp.weight"))
                    .is_some();
            let kv = self.kv_heads[l] as u64;
            let (k, v, h) = (self.head_k as u64, self.head_v as u64, self.n_head as u64);
            let mut want: Vec<(&str, Vec<u64>)> = vec![
                ("attn_output.weight", vec![h * v, e]),
                ("attn_sinks.weight", vec![h]),
                ("attn_qkv.weight", vec![e, h * k + kv * (k + v)]),
                ("attn_q.weight", vec![e, h * k]),
                ("attn_k.weight", vec![e, kv * k]),
                ("attn_v.weight", vec![e, kv * v]),
                ("nextn.eh_proj.weight", vec![2 * e, e]),
            ];
            if nextn || !routed {
                if let Some(ff) = self.dense_ff {
                    let ff = ff as u64;
                    want.extend([
                        ("ffn_gate.weight", vec![e, ff]),
                        ("ffn_up.weight", vec![e, ff]),
                        ("ffn_down.weight", vec![ff, e]),
                    ]);
                }
            } else {
                let n = self.n_expert as u64;
                want.extend([
                    ("ffn_gate_inp.weight", vec![e, n]),
                    (SELECTION_BIAS, vec![n]),
                    ("ffn_gate_exps.weight", vec![e, self.expert_ff as u64, n]),
                    ("ffn_up_exps.weight", vec![e, self.expert_ff as u64, n]),
                    ("ffn_down_exps.weight", vec![self.expert_ff as u64, e, n]),
                ]);
            }
            for (stem, dims) in &want {
                let name = format!("blk.{l}.{stem}");
                if let Some((_, info)) = split.find(&name)
                    && info.dims != *dims
                {
                    off.push((name, format!("{:?} (want {dims:?})", info.dims)));
                }
            }
        }
        match off.first() {
            None => Ok(()),
            Some((name, _)) => Err(PlacementError::Tensor {
                name: name.clone(),
                detail: format!(
                    "has other dims than the loader creates the tensor with ({} such: {})",
                    off.len(),
                    off.iter()
                        .map(|(n, d)| format!("{n} {d}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            }),
        }
    }
}

/// `nextn_predict_layers`, optional; believed only when the first
/// next-token layer carries `nextn.eh_proj`, as llama.cpp's loader does
/// (:28-30).
fn nextn(
    split: &Split,
    n_layer: usize,
    defaults: &mut Vec<String>,
) -> Result<usize, PlacementError> {
    let key = "nextn_predict_layers";
    let Some(n) = optional_usize(split, key)? else {
        defaults.push(format!("{key} = 0 (absent)"));
        return Ok(0);
    };
    if n >= n_layer {
        return Err(metadata(
            split,
            key,
            format!("is {n}, not below block_count {n_layer}"),
        ));
    }
    let probe = format!("blk.{}.nextn.eh_proj.weight", n_layer - n);
    if n > 0 && split.find(&probe).is_none() {
        defaults.push(format!("{key} = 0 ({probe} is not in the file)"));
        return Ok(0);
    }
    Ok(n)
}

/// `<architecture>.<suffix>` as a per-layer array of counts, every one
/// above 0: a scalar broadcast is refused by name, the file's own form
/// being the array (conversion/mimo.py:157).
fn per_layer_count(
    split: &Split,
    suffix: &str,
    n_layer: usize,
) -> Result<Vec<usize>, PlacementError> {
    let items = meta_arr(split, suffix)?;
    if items.len() != n_layer {
        return Err(metadata(
            split,
            suffix,
            format!("has {} values for {n_layer} layers", items.len()),
        ));
    }
    items
        .iter()
        .enumerate()
        .map(|(l, v)| {
            v.as_unsigned()
                .and_then(|x| usize::try_from(x).ok())
                .filter(|&x| x > 0)
                .ok_or_else(|| metadata(split, suffix, format!("has no count at layer {l}")))
        })
        .collect()
}

/// `attention.sliding_window_pattern`, a per-layer 0/1 array: 0 a full
/// layer, 1 a window of `attention.sliding_window` positions (mimo2.cpp:12,
/// `get_arr`).
fn pattern(split: &Split, n_layer: usize) -> Result<Vec<Kind>, PlacementError> {
    let key = "attention.sliding_window_pattern";
    let items = meta_arr(split, key)?;
    if items.len() != n_layer {
        return Err(metadata(
            split,
            key,
            format!("has {} values for {n_layer} layers", items.len()),
        ));
    }
    items
        .iter()
        .enumerate()
        .map(|(l, v)| match v.as_unsigned() {
            Some(0) => Ok(Kind::Full),
            Some(1) => Ok(Kind::Swa),
            other => Err(metadata(
                split,
                key,
                format!("is {other:?} at layer {l}; 0 marks a full layer, 1 a window"),
            )),
        })
        .collect()
}

/// A width the loader reads per layer or whole (`get_key_or_arr`, :8): this
/// description holds one width, so the per-layer array is refused by name.
fn one_width(split: &Split, suffix: &str, why: &str) -> Result<usize, PlacementError> {
    if let Some(Value::Array(items)) = split.value(&split.arch_key(suffix)) {
        return Err(metadata(
            split,
            suffix,
            format!(
                "is a per-layer array of {} values; this reader reads one width ({why})",
                items.len()
            ),
        ));
    }
    match meta_usize(split, suffix)? {
        0 => Err(metadata(split, suffix, "is 0")),
        v => Ok(v),
    }
}

/// `<architecture>.<suffix>` as a count above 0.
fn positive(split: &Split, suffix: &str) -> Result<usize, PlacementError> {
    match meta_usize(split, suffix)? {
        0 => Err(metadata(split, suffix, "is 0")),
        v => Ok(v),
    }
}

/// `<architecture>.<suffix>` as a count, `None` when absent.
fn optional_usize(split: &Split, suffix: &str) -> Result<Option<usize>, PlacementError> {
    match split.value(&split.arch_key(suffix)) {
        None => Ok(None),
        Some(_) => meta_usize(split, suffix).map(Some),
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::{Hparams, Kind};
    use crate::arch::synthetic::{V, header_shaped};

    /// Five layers: 0 full and dense, 1 and 2 window and routed, 3 full and
    /// routed, then the next-token layer 4 (window, dense).
    pub(super) const KINDS: [Kind; 5] = [Kind::Full, Kind::Swa, Kind::Swa, Kind::Full, Kind::Swa];

    /// The widths of [`keys`]: 64 wide, 4 heads, key 16, value 8, KV 2 on
    /// the full layers and 1 on the window ones, dense 48, expert 16 of 8.
    pub(super) const E: u64 = 64;
    pub(super) const HEADS: u64 = 4;
    pub(super) const K: u64 = 16;
    pub(super) const VD: u64 = 8;
    pub(super) const FF: u64 = 48;
    pub(super) const N_EXP: u64 = 8;

    pub(in crate::arch::mimo2) fn keys() -> Vec<(&'static str, V)> {
        vec![
            ("block_count", V::U32(5)),
            ("nextn_predict_layers", V::U32(1)),
            ("embedding_length", V::U32(64)),
            ("context_length", V::U32(1024)),
            ("attention.head_count", V::U32(4)),
            ("attention.head_count_kv", V::I32s(vec![2, 1, 1, 2, 1])),
            ("attention.key_length", V::U32(16)),
            ("attention.value_length", V::U32(8)),
            ("attention.value_scale", V::F32(0.707)),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-6)),
            ("attention.sliding_window", V::U32(8)),
            (
                "attention.sliding_window_pattern",
                V::I32s(vec![0, 1, 1, 0, 1]),
            ),
            ("rope.dimension_count", V::U32(4)),
            ("rope.freq_base", V::F32(1e7)),
            ("rope.freq_base_swa", V::F32(1e4)),
            ("expert_count", V::U32(8)),
            ("expert_used_count", V::U32(2)),
            ("expert_feed_forward_length", V::U32(16)),
            ("expert_gating_func", V::U32(2)),
            ("expert_weights_scale", V::F32(1.0)),
            ("expert_group_count", V::U32(1)),
            ("expert_group_used_count", V::U32(1)),
            ("feed_forward_length", V::U32(48)),
        ]
    }

    /// Every tensor of [`keys`]' file, at the loader's dims: the fused qkv
    /// matrix every layer carries (its rows the layer's own KV-head count),
    /// the sinks on the window layers alone, one feed-forward kind a trunk
    /// layer and the dense block a next-token one.
    pub(in crate::arch::mimo2) fn tensors() -> Vec<(String, Vec<u64>)> {
        let qkv = |kv: u64| vec![E, HEADS * K + kv * (K + VD)];
        let mut out: Vec<(String, Vec<u64>)> = [
            ("token_embd.weight", vec![E, 2]),
            ("output_norm.weight", vec![E]),
            ("output.weight", vec![E, 2]),
        ]
        .into_iter()
        .map(|(n, d)| (n.to_string(), d))
        .collect();
        for (l, kind) in KINDS.iter().enumerate() {
            let kv = [2u64, 1, 1, 2, 1][l];
            let mut stems: Vec<(String, Vec<u64>)> = vec![
                (format!("blk.{l}.attn_norm.weight"), vec![E]),
                (format!("blk.{l}.attn_qkv.weight"), qkv(kv)),
                (format!("blk.{l}.attn_output.weight"), vec![HEADS * VD, E]),
                (format!("blk.{l}.ffn_norm.weight"), vec![E]),
            ];
            if *kind == Kind::Swa {
                stems.push((format!("blk.{l}.attn_sinks.weight"), vec![HEADS]));
            }
            let trunk_routed = l < 4 && l != 0;
            if trunk_routed {
                stems.extend([
                    (format!("blk.{l}.ffn_gate_inp.weight"), vec![E, N_EXP]),
                    (format!("blk.{l}.exp_probs_b.bias"), vec![N_EXP]),
                    (format!("blk.{l}.ffn_gate_exps.weight"), vec![E, 16, N_EXP]),
                    (format!("blk.{l}.ffn_up_exps.weight"), vec![E, 16, N_EXP]),
                    (format!("blk.{l}.ffn_down_exps.weight"), vec![16, E, N_EXP]),
                ]);
            } else {
                stems.extend([
                    (format!("blk.{l}.ffn_gate.weight"), vec![E, FF]),
                    (format!("blk.{l}.ffn_up.weight"), vec![E, FF]),
                    (format!("blk.{l}.ffn_down.weight"), vec![FF, E]),
                ]);
            }
            if l == 4 {
                stems.extend([
                    (format!("blk.{l}.nextn.eh_proj.weight"), vec![2 * E, E]),
                    (format!("blk.{l}.nextn.enorm.weight"), vec![E]),
                    (format!("blk.{l}.nextn.hnorm.weight"), vec![E]),
                    (format!("blk.{l}.layer_output_norm.weight"), vec![E]),
                ]);
            }
            out.extend(stems);
        }
        out
    }

    /// The read of a header of `kv` and `tensors`, or its error's text.
    pub(super) fn read(
        tag: &str,
        kv: &[(&str, V)],
        tensors: &[(String, Vec<u64>)],
    ) -> Result<Hparams, String> {
        let path = header_shaped(tag, "mimo2", kv, &[], tensors);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let hp = Hparams::read(&split).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        hp
    }

    fn refused(tag: &str, kv: &[(&str, V)], tensors: &[(String, Vec<u64>)], want: &str) {
        let err = read(tag, kv, tensors).expect_err("the header is refused");
        assert!(err.contains(want), "want {want:?} in: {err}");
    }

    /// `kv` with `key` set to `v`, or dropped for `None`.
    fn with(key: &str, v: Option<V>) -> Vec<(&'static str, V)> {
        let mut v = v;
        keys()
            .into_iter()
            .filter_map(|(k, old)| {
                if k == key {
                    v.take().map(|v| (k, v))
                } else {
                    Some((k, old))
                }
            })
            .collect()
    }

    #[test]
    fn a_small_header_reads() {
        let hp = read("mimo2-ok", &keys(), &tensors()).expect("the header reads");
        assert_eq!(hp.kinds, KINDS);
        assert_eq!(hp.kv_heads, [2, 1, 1, 2, 1]);
        assert_eq!((hp.n_layer, hp.n_trunk), (5, 4));
        assert_eq!((hp.head_k, hp.head_v), (16, 8));
        assert_eq!(hp.defaults, Vec::<String>::new());
    }

    #[test]
    fn a_missing_key_is_named() {
        for key in [
            "attention.sliding_window",
            "rope.freq_base",
            "attention.value_length",
            "attention.value_scale",
        ] {
            let kv: Vec<_> = keys().into_iter().filter(|(k, _)| *k != key).collect();
            refused("mimo2-key", &kv, &tensors(), &format!("mimo2.{key}"));
        }
    }

    /// A value scale that is not a finite positive multiplier is refused by name.
    #[test]
    fn a_value_scale_off_the_positive_reals_is_refused() {
        for bad in [0.0_f32, -0.5, f32::NAN, f32::INFINITY] {
            let kv = with("attention.value_scale", Some(V::F32(bad)));
            refused(
                "mimo2-scale",
                &kv,
                &tensors(),
                "mimo2.attention.value_scale",
            );
        }
    }

    /// A short per-layer array is refused by name — the KV-head count and
    /// the window pattern are the file's per-layer facts, not a scalar a
    /// reader may broadcast.
    #[test]
    fn a_short_layer_array_is_refused() {
        for key in [
            "attention.head_count_kv",
            "attention.sliding_window_pattern",
        ] {
            let kv = with(key, Some(V::I32s(vec![0, 1, 1, 0])));
            refused(
                "mimo2-arr",
                &kv,
                &tensors(),
                &format!("mimo2.{key}: has 4 values for 5 layers"),
            );
        }
    }

    /// A scalar where the file's form is the per-layer array is refused by
    /// name.
    #[test]
    fn a_scalar_layer_array_is_refused() {
        let kv = with("attention.head_count_kv", Some(V::U32(2)));
        refused(
            "mimo2-scalar",
            &kv,
            &tensors(),
            "mimo2.attention.head_count_kv: is absent or not an array",
        );
    }

    #[test]
    fn a_pattern_value_off_0_and_1_is_refused() {
        let kv = with(
            "attention.sliding_window_pattern",
            Some(V::I32s(vec![0, 2, 1, 0, 1])),
        );
        refused(
            "mimo2-pattern",
            &kv,
            &tensors(),
            "0 marks a full layer, 1 a window",
        );
    }

    /// A key head the rope turns more values of than it has is refused, as
    /// an odd turn count is.
    #[test]
    fn a_rope_wider_than_the_head_is_refused() {
        for dims in [18u32, 3] {
            let kv = with("rope.dimension_count", Some(V::U32(dims)));
            refused(
                "mimo2-rope",
                &kv,
                &tensors(),
                &format!("mimo2.rope.dimension_count: is {dims}, for a key head of 16"),
            );
        }
    }

    #[test]
    fn another_gating_function_is_refused() {
        let kv = with("expert_gating_func", Some(V::U32(1)));
        refused(
            "mimo2-gating",
            &kv,
            &tensors(),
            "mimo2.expert_gating_func: is 1; this reader reads the sigmoid router",
        );
    }

    /// A per-layer expert width is refused by name: the description holds
    /// one width, as the dense width's array is.
    #[test]
    fn a_per_layer_width_is_refused() {
        for key in ["expert_feed_forward_length", "feed_forward_length"] {
            let kv = with(key, Some(V::I32s(vec![16; 5])));
            refused(
                "mimo2-ff",
                &kv,
                &tensors(),
                &format!("mimo2.{key}: is a per-layer array of 5 values"),
            );
        }
    }

    /// A layer with both QKV forms, or neither, is refused by name.
    #[test]
    fn a_layer_without_exactly_one_qkv_form_is_refused() {
        let both: Vec<(String, Vec<u64>)> = tensors()
            .into_iter()
            .chain([("blk.1.attn_q.weight".to_string(), vec![E, HEADS * K])])
            .collect();
        refused(
            "mimo2-qkv-both",
            &keys(),
            &both,
            "tensor blk.1.attn_q.weight: is in the file, and its layer's block carries none",
        );
        let neither: Vec<(String, Vec<u64>)> = tensors()
            .into_iter()
            .filter(|(n, _)| n != "blk.1.attn_qkv.weight")
            .collect();
        refused("mimo2-qkv-none", &keys(), &neither, "blk.1.attn_qkv.weight");
    }

    /// A sink on a full-attention layer is refused: the sinks belong to the
    /// window layers (`add_swa_attention_sink_bias`).
    #[test]
    fn a_sink_on_a_full_layer_is_refused() {
        let t: Vec<(String, Vec<u64>)> = tensors()
            .into_iter()
            .chain([("blk.0.attn_sinks.weight".to_string(), vec![HEADS])])
            .collect();
        refused(
            "mimo2-sink",
            &keys(),
            &t,
            "tensor blk.0.attn_sinks.weight: is in",
        );
    }

    /// A window layer without sinks reads: llama.cpp loads them
    /// `TENSOR_NOT_REQUIRED` and the graph runs without them.
    #[test]
    fn a_window_layer_without_sinks_reads() {
        let t: Vec<(String, Vec<u64>)> = tensors()
            .into_iter()
            .filter(|(n, _)| n != "blk.1.attn_sinks.weight")
            .collect();
        let hp = read("mimo2-nosink", &keys(), &t).expect("the header reads");
        assert_eq!(hp.kinds, KINDS);
    }

    /// The qkv matrix's rows are the layer's own KV-head count: a width off
    /// the arrays is refused by the tensor's name, on a window layer and a
    /// full one alike.
    #[test]
    fn a_qkv_of_other_rows_is_refused() {
        // Layer 1 holds 1 KV head and layer 0 holds 2: each is sized with
        // the other's count.
        for (name, dims) in [
            ("blk.1.attn_qkv.weight", vec![E, HEADS * K + 2 * (K + VD)]),
            ("blk.0.attn_qkv.weight", vec![E, HEADS * K + K + VD]),
        ] {
            let t: Vec<(String, Vec<u64>)> = tensors()
                .into_iter()
                .map(|(n, d)| if n == name { (n, dims.clone()) } else { (n, d) })
                .collect();
            let err = read("mimo2-dims", &keys(), &t).expect_err("the header is refused");
            assert!(
                err.contains(&format!("tensor {name}: has other dims")),
                "want {name} in: {err}"
            );
        }
    }

    /// A routed next-token layer is refused: the MTP graph asserts the
    /// block's dense FFN tensors, so its dense gate is missing by name and a
    /// router's weight is misplaced on it.
    #[test]
    fn a_routed_next_token_layer_is_refused() {
        let t: Vec<(String, Vec<u64>)> = tensors()
            .into_iter()
            .filter(|(n, _)| n != "blk.4.ffn_gate.weight")
            .chain([("blk.4.ffn_gate_inp.weight".to_string(), vec![E, N_EXP])])
            .collect();
        refused(
            "mimo2-mtp-route",
            &keys(),
            &t,
            "tensor blk.4.ffn_gate.weight: is not in",
        );
    }

    /// `nextn_predict_layers` above the block count is refused by name.
    #[test]
    fn a_nextn_count_past_the_blocks_is_refused() {
        let kv = with("nextn_predict_layers", Some(V::U32(5)));
        refused(
            "mimo2-nextn",
            &kv,
            &tensors(),
            "mimo2.nextn_predict_layers: is 5, not below block_count 5",
        );
    }
}
