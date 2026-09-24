//! Every hyperparameter the Qwen3-MoE decode step reads, resolved once at load
//! from the file's metadata and the tensors it holds, and a refusal for every
//! value this chain does not run. This is the one reader of `qwen3moe.*` keys.
//!
//! Nothing here has a default of its own. A key ik reads as optional keeps
//! ik's reading of its absence; the router's score function and its
//! renormalization are not keys at all for this architecture — ik's builder
//! passes them as constants — so they are constants here, cited to that line,
//! and a file that spells either differently is refused. The ik line numbers
//! are those of the tree `tools/ref/models/qwen3moe.sh` names as `IK`.

use gguf::{Split, Value};

use super::names;
use crate::arch::{meta_f32, meta_str, meta_usize, metadata, n_vocab};
use crate::placement::PlacementError;

/// ik's `LLM_EXPERT_GATING_FUNC_SOFTMAX` (llama-hparams.h:15).
const IK_SOFTMAX: u64 = 1;

/// The hyperparameters of one qwen3moe file this chain runs.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// `block_count`.
    pub n_layer: usize,
    /// `embedding_length` — the residual width.
    pub n_embd: usize,
    /// `attention.head_count` — query heads.
    pub n_head: usize,
    /// `attention.head_count_kv` — key and value heads; `n_head / n_head_kv`
    /// query heads share each one.
    pub n_head_kv: usize,
    /// `attention.key_length`, which `attention.value_length` equals (ik
    /// asserts it, build_qwen3.cpp:110): one head's query, key and value width.
    pub head_dim: usize,
    /// The rope every layer applies to Q and K.
    pub rope: Rope,
    /// `attention.layer_norm_rms_epsilon` — every RMS norm's, the per-head
    /// query and key norms included.
    pub rms_eps: f32,
    /// The vocabulary size: `vocab_size` when the file carries it, else the
    /// length of `tokenizer.ggml.tokens` (ik's rule, llama-hparams.cpp:155).
    pub n_vocab: usize,
    /// `context_length` — the context the model was trained to, not the one
    /// this engine allocates.
    pub n_ctx_train: usize,
    /// The router and the experts.
    pub experts: Experts,
}

/// How the rope pairs a head's values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeMode {
    /// Value `i` rotates with value `i + dims/2` — `LLAMA_ROPE_TYPE_NEOX`,
    /// ik's rope type for this architecture (`llama_rope_type`,
    /// llama.cpp:9737, 9765).
    Neox,
}

/// The rope ik's graph applies to every layer's Q and K: `ggml_rope_ext` with
/// no frequency factors, no YaRN (the file carries no scaling keys this chain
/// accepts) and a frequency scale of 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rope {
    pub mode: RopeMode,
    /// `rope.freq_base` — θ.
    pub base: f32,
    /// The rotated width: `rope.dimension_count` when the file carries it,
    /// else ik's reading of its absence, the key width
    /// (llama-hparams.cpp:236-238); ik's builder asserts it is the whole head
    /// (build_qwen3.cpp:111).
    pub dims: usize,
}

/// The router's score function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Score {
    /// Softmax over every expert's logit: the constant ik's builder passes
    /// (build_qwen3.cpp:148).
    Softmax,
}

/// The mixture of experts every layer runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Experts {
    /// `expert_count` — routed experts per layer.
    pub n_expert: usize,
    /// `expert_used_count` — routed experts one token runs per layer.
    pub n_used: usize,
    /// `expert_feed_forward_length` — one expert's hidden width.
    pub ff: usize,
    pub score: Score,
    /// Whether the chosen experts' weights are renormalized to sum to one:
    /// always, the `norm_w` constant ik's builder passes, with no scale after
    /// it (build_qwen3.cpp:147).
    pub weights_norm: bool,
}

impl Hparams {
    /// Read the file's hyperparameters, refusing every value this chain does
    /// not run. Load-time only.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        let n_layer = meta_usize(split, "block_count")?;
        let n_embd = meta_usize(split, "embedding_length")?;
        let n_head = meta_usize(split, "attention.head_count")?;
        // Absent, ik takes the query head count (llama-hparams.cpp:192).
        let n_head_kv = optional_usize(split, "attention.head_count_kv")?.unwrap_or(n_head);
        if n_head == 0 || n_head_kv == 0 || !n_head.is_multiple_of(n_head_kv) {
            return Err(metadata(
                split,
                "attention.head_count_kv",
                format!(
                    "is {n_head_kv} for {n_head} query heads; grouped-query attention needs a \
                     whole number of query heads per key head"
                ),
            ));
        }
        // Absent, ik takes n_embd / n_head for both (llama-hparams.cpp:229-233).
        let key = optional_usize(split, "attention.key_length")?.unwrap_or(n_embd / n_head);
        let value = optional_usize(split, "attention.value_length")?.unwrap_or(n_embd / n_head);
        if key != value || key == 0 {
            return Err(metadata(
                split,
                "attention.value_length",
                format!("is {value}, attention.key_length {key}; one width serves Q, K and V"),
            ));
        }
        Ok(Hparams {
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            head_dim: key,
            rope: Rope::read(split, key)?,
            rms_eps: meta_f32(split, "attention.layer_norm_rms_epsilon")?,
            n_vocab: n_vocab(split)?,
            n_ctx_train: meta_usize(split, "context_length")?,
            experts: Experts::read(split, n_layer)?,
        })
    }

    /// Query heads per key and value head.
    #[must_use]
    pub fn group(&self) -> usize {
        self.n_head / self.n_head_kv
    }
}

impl Rope {
    fn read(split: &Split, head_dim: usize) -> Result<Rope, PlacementError> {
        let dims = optional_usize(split, "rope.dimension_count")?.unwrap_or(head_dim);
        if dims != head_dim {
            return Err(metadata(
                split,
                "rope.dimension_count",
                format!("is {dims}, the head {head_dim}; this chain rotates whole heads"),
            ));
        }
        let scaling = "rope.scaling.type";
        if split.value(&split.arch_key(scaling)).is_some() {
            let kind = meta_str(split, scaling)?;
            if kind != "none" {
                return Err(metadata(
                    split,
                    scaling,
                    format!("is {kind:?}; this chain runs the plain rope, no scaling"),
                ));
            }
        }
        let factor = "rope.scaling.factor";
        if split.value(&split.arch_key(factor)).is_some() {
            let f = meta_f32(split, factor)?;
            if f != 0.0 && f != 1.0 {
                return Err(metadata(
                    split,
                    factor,
                    format!("is {f}; this chain runs the rope at frequency scale 1"),
                ));
            }
        }
        Ok(Rope {
            mode: RopeMode::Neox,
            base: meta_f32(split, "rope.freq_base")?,
            dims,
        })
    }
}

impl Experts {
    fn read(split: &Split, n_layer: usize) -> Result<Experts, PlacementError> {
        let n_expert = meta_usize(split, "expert_count")?;
        let n_used = meta_usize(split, "expert_used_count")?;
        if n_expert == 0 || n_used == 0 || n_used > n_expert {
            return Err(metadata(
                split,
                "expert_used_count",
                format!("is {n_used} of expert_count {n_expert}"),
            ));
        }
        // ik reads it as optional and leaves 0 (llama-hparams.cpp:557); a
        // 0 would make every expert empty.
        let ff = meta_usize(split, "expert_feed_forward_length")?;
        if ff == 0 {
            return Err(metadata(split, "expert_feed_forward_length", "is 0"));
        }
        refuse_unless(
            split,
            "expert_shared_count",
            |v| v.as_u64() == Some(0),
            || "this chain has no shared expert (ik's builder passes none, build_qwen3.cpp:143-145)",
        )?;
        refuse_unless(
            split,
            "expert_gating_func",
            |v| v.as_u64() == Some(IK_SOFTMAX),
            || "this chain routes with softmax (1), the constant ik's builder passes",
        )?;
        refuse_unless(
            split,
            "expert_weights_norm",
            |v| v.as_bool() == Some(true),
            || "this chain renormalizes the chosen weights, the constant ik's builder passes",
        )?;
        refuse_unless(
            split,
            "expert_weights_scale",
            |v| v.as_f32() == Some(1.0),
            || "this chain scales the chosen weights by nothing, as ik's builder does",
        )?;
        let dense = (0..n_layer)
            .filter(|&l| split.find(&names::ffn_gate_inp(l)).is_none())
            .collect::<Vec<_>>();
        if let Some(&first) = dense.first() {
            return Err(PlacementError::Tensor {
                name: names::ffn_gate_inp(first),
                detail: format!(
                    "is missing on {} of {n_layer} layers (first {first}); this chain runs \
                     every layer through the router, no dense FFN",
                    dense.len()
                ),
            });
        }
        Ok(Experts {
            n_expert,
            n_used,
            ff,
            score: Score::Softmax,
            weights_norm: true,
        })
    }
}

/// `<architecture>.<suffix>` as a count, `None` when the file does not carry
/// the key; a key of another type is an error naming it.
fn optional_usize(split: &Split, suffix: &str) -> Result<Option<usize>, PlacementError> {
    match split.value(&split.arch_key(suffix)) {
        None => Ok(None),
        Some(_) => meta_usize(split, suffix).map(Some),
    }
}

/// Err naming `suffix` and its value when the file carries the key with a
/// value `ok` refuses; an absent key is this chain's value.
fn refuse_unless(
    split: &Split,
    suffix: &str,
    ok: impl Fn(&Value) -> bool,
    why: impl Fn() -> &'static str,
) -> Result<(), PlacementError> {
    match split.value(&split.arch_key(suffix)) {
        Some(v) if !ok(v) => {
            let shown = v
                .as_unsigned()
                .map_or_else(|| format!("{v:?}"), |n| n.to_string());
            Err(metadata(split, suffix, format!("is {shown}; {}", why())))
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::{Experts, Hparams, Rope, RopeMode, Score};
    use crate::arch::synthetic::{V, header};
    use gguf::Split;

    /// The keys this module reads, with the values the Qwen3-30B-A3B-2507
    /// file's header holds; the optional keys it does not carry are left out,
    /// as there.
    fn a3b() -> Vec<(&'static str, V)> {
        vec![
            ("block_count", V::U32(2)),
            ("context_length", V::U32(262_144)),
            ("embedding_length", V::U32(2048)),
            ("attention.head_count", V::U32(32)),
            ("attention.head_count_kv", V::U32(4)),
            ("rope.freq_base", V::F32(10_000_000.0)),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-6)),
            ("expert_used_count", V::U32(8)),
            ("attention.key_length", V::U32(128)),
            ("attention.value_length", V::U32(128)),
            ("expert_count", V::U32(128)),
            ("expert_feed_forward_length", V::U32(768)),
        ]
    }

    /// `a3b()` with `key` set to `v` (added when absent).
    fn with(key: &'static str, v: V) -> Vec<(&'static str, V)> {
        let mut t = a3b();
        match t.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = v,
            None => t.push((key, v)),
        }
        t
    }

    /// A header-only qwen3moe file of `kv` (a two-token vocabulary) with one
    /// 1-value F32 `ffn_gate_inp` per layer in `routed`, read.
    fn read(tag: &str, kv: &[(&str, V)], routed: &[usize]) -> Result<Hparams, String> {
        let names: Vec<String> = routed
            .iter()
            .map(|l| format!("blk.{l}.ffn_gate_inp.weight"))
            .collect();
        let path = header(&format!("qwen3moe-hparams-{tag}"), "qwen3moe", kv, &names);
        let split = Split::open(&path).map_err(|e| e.to_string())?;
        let hp = Hparams::read(&split).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        hp
    }

    /// The A3B header reads, to the values its chain runs, with the constants
    /// ik's builder passes.
    #[test]
    fn a3b_reads() {
        let hp = read("a3b", &a3b(), &[0, 1]).expect("the A3B header reads");
        assert_eq!(
            hp,
            Hparams {
                n_layer: 2,
                n_embd: 2048,
                n_head: 32,
                n_head_kv: 4,
                head_dim: 128,
                rope: Rope {
                    mode: RopeMode::Neox,
                    base: 10_000_000.0,
                    dims: 128,
                },
                rms_eps: 1e-6,
                n_vocab: 2,
                n_ctx_train: 262_144,
                experts: Experts {
                    n_expert: 128,
                    n_used: 8,
                    ff: 768,
                    score: Score::Softmax,
                    weights_norm: true,
                },
            }
        );
        assert_eq!(hp.group(), 8);
        // The values the file may also spell out, spelled as this chain runs.
        for (i, (k, v)) in [
            ("rope.dimension_count", V::U32(128)),
            ("expert_gating_func", V::U32(1)),
            ("expert_weights_norm", V::Bool(true)),
            ("expert_weights_scale", V::F32(1.0)),
            ("expert_shared_count", V::U32(0)),
            ("rope.scaling.type", V::Str("none")),
        ]
        .into_iter()
        .enumerate()
        {
            let got = read(&format!("spelled{i}"), &with(k, v), &[0, 1]).expect(k);
            assert_eq!(got, hp, "{k} spelled out as this chain runs it");
        }
    }

    /// Each value this chain does not run is refused, and the error names the
    /// key and the value the file holds.
    #[test]
    fn unsupported_values_are_refused_by_key_and_value() {
        let cases = [
            (
                with("attention.head_count_kv", V::U32(5)),
                "metadata qwen3moe.attention.head_count_kv: is 5 for 32 query heads;",
            ),
            (
                with("attention.value_length", V::U32(64)),
                "metadata qwen3moe.attention.value_length: is 64, attention.key_length 128;",
            ),
            (
                with("rope.dimension_count", V::U32(64)),
                "metadata qwen3moe.rope.dimension_count: is 64, the head 128;",
            ),
            (
                with("rope.scaling.type", V::Str("yarn")),
                "metadata qwen3moe.rope.scaling.type: is \"yarn\";",
            ),
            (
                with("rope.scaling.factor", V::F32(4.0)),
                "metadata qwen3moe.rope.scaling.factor: is 4;",
            ),
            (
                with("expert_used_count", V::U32(129)),
                "metadata qwen3moe.expert_used_count: is 129 of expert_count 128",
            ),
            (
                with("expert_shared_count", V::U32(1)),
                "metadata qwen3moe.expert_shared_count: is 1;",
            ),
            (
                with("expert_gating_func", V::U32(2)),
                "metadata qwen3moe.expert_gating_func: is 2;",
            ),
            (
                with("expert_weights_norm", V::Bool(false)),
                "metadata qwen3moe.expert_weights_norm: is Bool(false);",
            ),
            (
                with("expert_weights_scale", V::F32(2.5)),
                "metadata qwen3moe.expert_weights_scale: is F32(2.5);",
            ),
        ];
        let mut wrong: Vec<String> = cases
            .into_iter()
            .enumerate()
            .filter_map(
                |(i, (kv, want))| match read(&format!("refuse{i}"), &kv, &[0, 1]) {
                    Ok(_) => Some(format!("case {i} read; expected \"{want}\"")),
                    Err(e) if e.starts_with(want) => None,
                    Err(e) => Some(format!("case {i}: {e}")),
                },
            )
            .collect();
        // A layer without a router is a dense FFN this chain does not run.
        match read("dense", &a3b(), &[1]) {
            Err(e) if e.starts_with("tensor blk.0.ffn_gate_inp.weight: is missing on 1 of 2") => {}
            other => wrong.push(format!("dense layer: {other:?}")),
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }
}
