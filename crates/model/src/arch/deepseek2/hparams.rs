//! Every hyperparameter the deepseek2 layer chain reads, resolved once at load
//! from the file's metadata, and a refusal for every value that chain does not
//! run.
//!
//! `deepseek2` is the architecture string of more than DeepSeek-V2: files with a
//! low-rank query path, sigmoid routing or the absorbed MLA layout declare it
//! too. This chain is the V2-Lite one, so each of those is refused here with the
//! key and the value the file holds, never branched on. The kernels have their
//! own widths on top (expert and slot counts, the latent block, the flash tile);
//! the device crate holds these values against them.
//!
//! A key ik reads as optional keeps ik's reading of its absence; the ik line
//! numbers are those of the tree `tools/ref/models/deepseek2.sh` names as `IK`.

use gguf::{Gguf, Value};

use crate::placement::PlacementError;

/// ik's `LLM_EXPERT_GATING_FUNC_TYPE_NONE` (llama-hparams.h:14): the key's
/// value when the file does not say.
const IK_GATING_NONE: u64 = 0;
/// ik's `LLM_EXPERT_GATING_FUNC_SOFTMAX` (llama-hparams.h:15).
const IK_SOFTMAX: u64 = 1;
/// ik's `LLM_EXPERT_GATING_FUNC_SIGMOID` (llama-hparams.h:16).
const IK_SIGMOID: u64 = 2;

const GATING_KEY: &str = "expert_gating_func";
const NORM_KEY: &str = "expert_weights_norm";

/// How the router scores the experts' logits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gating {
    /// Softmax over every expert's logit.
    Softmax,
}

/// The hyperparameters of one deepseek2 file this chain runs.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// `block_count`.
    pub n_layer: usize,
    /// `attention.head_count` — `attention.head_count_kv` equals it: every
    /// query head has its own key and value head out of `attn_kv_b`.
    pub n_head: usize,
    /// `attention.key_length` — one head's query/key width, nope plus rope.
    pub key_length: usize,
    /// `attention.value_length` — one head's value width.
    pub value_length: usize,
    /// `attention.kv_lora_rank` — the width of the compressed KV latent.
    pub kv_lora_rank: usize,
    /// `rope.dimension_count` — the rotated tail of each key.
    pub rope_dims: usize,
    pub experts: Experts,
}

/// The routed and shared experts.
#[derive(Clone, Debug, PartialEq)]
pub struct Experts {
    /// `expert_count` — the experts the router scores.
    pub n_expert: usize,
    /// `expert_used_count` — the experts each token runs.
    pub n_used: usize,
    /// `expert_shared_count` — the shared experts folded into `ffn_*_shexp`.
    pub n_shared: usize,
    /// `expert_feed_forward_length` — one routed expert's hidden width.
    pub ff: usize,
    /// `expert_weights_scale`, applied to every router weight.
    pub scale: f32,
    /// `leading_dense_block_count`. Which block is dense is still read from
    /// the tensors (a block without `ffn_gate_inp`), not from this count.
    pub dense_lead: usize,
    /// `expert_gating_func`, or ik's reading of its absence.
    pub gating: Gating,
}

impl Hparams {
    /// Read the file's hyperparameters, refusing every value this chain does
    /// not run. Load-time only.
    pub fn read(gguf: &Gguf) -> Result<Hparams, PlacementError> {
        Hparams::from_meta(gguf)
    }

    fn from_meta(m: &impl Meta) -> Result<Hparams, PlacementError> {
        let n_layer = need_usize(m, "block_count")?;
        let n_head = need_usize(m, "attention.head_count")?;
        // Absent, ik takes the query head count (llama-hparams.cpp:189-192).
        let n_head_kv = match m.value("attention.head_count_kv") {
            Some(_) => need_usize(m, "attention.head_count_kv")?,
            None => n_head,
        };
        if n_head_kv != n_head {
            return Err(refusal(
                m,
                "attention.head_count_kv",
                format!(
                    "is {n_head_kv}, attention.head_count {n_head}; this chain reads a key and a \
                     value head per query head from attn_kv_b, and one KV head is the absorbed \
                     MLA layout (llama-hparams.cpp:1203)"
                ),
            ));
        }
        for suffix in ["attention.key_length_mla", "attention.value_length_mla"] {
            refuse_present(
                m,
                suffix,
                "it names the per-head widths of the absorbed MLA layout (attn_k_b, attn_v_b); \
                 this chain takes them from attention.key_length and attn_kv_b",
            )?;
        }
        refuse_present(
            m,
            "attention.q_lora_rank",
            "a low-rank query path (attn_q_a, attn_q_a_norm, attn_q_b); this chain projects \
             the query with one attn_q",
        )?;
        Ok(Hparams {
            n_layer,
            n_head,
            key_length: need_usize(m, "attention.key_length")?,
            value_length: need_usize(m, "attention.value_length")?,
            kv_lora_rank: need_usize(m, "attention.kv_lora_rank")?,
            rope_dims: need_usize(m, "rope.dimension_count")?,
            experts: Experts::read(m, n_layer)?,
        })
    }
}

impl Experts {
    fn read(m: &impl Meta, n_layer: usize) -> Result<Experts, PlacementError> {
        let n_expert = need_usize(m, "expert_count")?;
        let n_used = need_usize(m, "expert_used_count")?;
        if n_expert == 0 || n_used == 0 || n_used > n_expert {
            return Err(refusal(
                m,
                "expert_used_count",
                format!("is {n_used} of expert_count {n_expert}"),
            ));
        }
        let ff = need_usize(m, "expert_feed_forward_length")?;
        if ff == 0 {
            return Err(refusal(m, "expert_feed_forward_length", "is 0"));
        }
        // Absent, ik leaves it false (llama-hparams.cpp:1232).
        let norm = match m.value(NORM_KEY) {
            Some(v) => v
                .as_bool()
                .ok_or_else(|| refusal(m, NORM_KEY, format!("is {v:?}, not a bool")))?,
            None => false,
        };
        if norm {
            return Err(refusal(
                m,
                NORM_KEY,
                "is true; this chain's router weights are the chosen probabilities times \
                 expert_weights_scale, never divided by their sum",
            ));
        }
        Ok(Experts {
            n_expert,
            n_used,
            n_shared: need_usize(m, "expert_shared_count")?,
            ff,
            scale: need_f32(m, "expert_weights_scale")?,
            dense_lead: need_usize(m, "leading_dense_block_count")?,
            gating: gating(m, n_layer)?,
        })
    }
}

/// The router's score function: the file's `expert_gating_func`, and where the
/// file does not say (the key absent, or ik's "none"), ik's reading of that
/// (llama-hparams.cpp:1233-1245): a 47- or 48-layer file is GLM-4.7-Flash and
/// routes with sigmoid, any other is a DeepSeek-V2 and routes with softmax.
fn gating(m: &impl Meta, n_layer: usize) -> Result<Gating, PlacementError> {
    let declared = match m.value(GATING_KEY) {
        Some(_) => need_u64(m, GATING_KEY)?,
        None => IK_GATING_NONE,
    };
    let func = match declared {
        IK_GATING_NONE if matches!(n_layer, 47 | 48) => IK_SIGMOID,
        IK_GATING_NONE => IK_SOFTMAX,
        f => f,
    };
    if func == IK_SOFTMAX {
        return Ok(Gating::Softmax);
    }
    let read = if declared == IK_GATING_NONE {
        format!("is not set, which ik reads as {func} (sigmoid) on a {n_layer}-layer file")
    } else {
        format!("is {func}")
    };
    Err(refusal(
        m,
        GATING_KEY,
        format!("{read}; this chain routes with softmax ({IK_SOFTMAX})"),
    ))
}

// --------------------------------------------------------------- readers

/// Where the `<architecture>.<suffix>` keys come from: the file, or a table
/// in the tests.
trait Meta {
    /// `<architecture>.<suffix>`'s value, if the file carries the key.
    fn value(&self, suffix: &str) -> Option<&Value>;
    /// The full key `<architecture>.<suffix>`, for an error to name.
    fn key(&self, suffix: &str) -> String;
}

/// The `Gguf::arch_get_*` lookup: no value when the file names no
/// architecture.
impl Meta for Gguf {
    fn value(&self, suffix: &str) -> Option<&Value> {
        let arch = self.architecture()?;
        Gguf::value(self, &format!("{arch}.{suffix}"))
    }

    fn key(&self, suffix: &str) -> String {
        self.arch_key(suffix)
    }
}

/// The error that names `<architecture>.<suffix>`.
fn refusal(m: &impl Meta, suffix: &str, detail: impl Into<String>) -> PlacementError {
    PlacementError::Metadata {
        key: m.key(suffix),
        detail: detail.into(),
    }
}

/// Err naming `suffix` and its value when the file carries the key at all: a
/// key whose presence means a layout this chain does not run.
fn refuse_present(m: &impl Meta, suffix: &str, why: &str) -> Result<(), PlacementError> {
    match m.value(suffix) {
        None => Ok(()),
        Some(v) => {
            let v = v
                .as_unsigned()
                .map_or_else(|| format!("{v:?}"), |n| n.to_string());
            Err(refusal(m, suffix, format!("is {v}; {why}")))
        }
    }
}

/// `<architecture>.<suffix>` as an unsigned integer.
fn need_u64(m: &impl Meta, suffix: &str) -> Result<u64, PlacementError> {
    m.value(suffix)
        .and_then(Value::as_u64)
        .ok_or_else(|| refusal(m, suffix, "is absent or not an unsigned integer"))
}

/// `need_u64` for a count that indexes memory.
fn need_usize(m: &impl Meta, suffix: &str) -> Result<usize, PlacementError> {
    let v = need_u64(m, suffix)?;
    usize::try_from(v).map_err(|_| refusal(m, suffix, format!("{v} does not fit usize")))
}

/// `<architecture>.<suffix>` as a float.
fn need_f32(m: &impl Meta, suffix: &str) -> Result<f32, PlacementError> {
    m.value(suffix)
        .and_then(Value::as_f32)
        .ok_or_else(|| refusal(m, suffix, "is absent or not a float"))
}

#[cfg(test)]
mod tests {
    use super::{Experts, Gating, Hparams, Meta};
    use gguf::Value;

    /// A deepseek2 header as a key table.
    struct Table(Vec<(&'static str, Value)>);

    impl Meta for Table {
        fn value(&self, suffix: &str) -> Option<&Value> {
            self.0.iter().find(|(k, _)| *k == suffix).map(|(_, v)| v)
        }

        fn key(&self, suffix: &str) -> String {
            format!("deepseek2.{suffix}")
        }
    }

    /// The keys this module reads, with the values the V2-Lite file's header
    /// holds; the optional keys it does not carry are left out, as there.
    fn v2_lite() -> Table {
        Table(vec![
            ("block_count", Value::U32(27)),
            ("attention.head_count", Value::U32(16)),
            ("attention.head_count_kv", Value::U32(16)),
            ("attention.key_length", Value::U32(192)),
            ("attention.value_length", Value::U32(128)),
            ("attention.kv_lora_rank", Value::U32(512)),
            ("rope.dimension_count", Value::U32(64)),
            ("expert_count", Value::U32(64)),
            ("expert_used_count", Value::U32(6)),
            ("expert_shared_count", Value::U32(2)),
            ("expert_feed_forward_length", Value::U32(1408)),
            ("expert_weights_scale", Value::F32(1.0)),
            ("leading_dense_block_count", Value::U32(1)),
        ])
    }

    /// `v2_lite()` with `key` set to `v` (added when absent).
    fn with(key: &'static str, v: Value) -> Table {
        let mut t = v2_lite();
        match t.0.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = v,
            None => t.0.push((key, v)),
        }
        t
    }

    /// The V2-Lite header reads, to the values its chain runs.
    #[test]
    fn v2_lite_reads() {
        let hp = Hparams::from_meta(&v2_lite()).expect("the V2-Lite header reads");
        assert_eq!(
            hp,
            Hparams {
                n_layer: 27,
                n_head: 16,
                key_length: 192,
                value_length: 128,
                kv_lora_rank: 512,
                rope_dims: 64,
                experts: Experts {
                    n_expert: 64,
                    n_used: 6,
                    n_shared: 2,
                    ff: 1408,
                    scale: 1.0,
                    dense_lead: 1,
                    gating: Gating::Softmax,
                },
            }
        );
        // The values the file may also spell out, spelled as V2-Lite runs.
        for (k, v) in [
            ("expert_gating_func", Value::U32(1)),
            ("expert_weights_norm", Value::Bool(false)),
        ] {
            let got = Hparams::from_meta(&with(k, v)).expect(k);
            assert_eq!(got, hp, "{k} spelled out as V2-Lite runs it");
        }
    }

    /// Each value this chain does not run — GLM-4.7-Flash's, where it has one —
    /// is refused, and the error names the key and the value the file holds.
    #[test]
    fn unsupported_values_are_refused_by_key_and_value() {
        let cases = [
            (
                with("attention.q_lora_rank", Value::U32(768)),
                "metadata deepseek2.attention.q_lora_rank: is 768;",
            ),
            (
                with("attention.head_count_kv", Value::U32(1)),
                "metadata deepseek2.attention.head_count_kv: is 1, attention.head_count 16;",
            ),
            (
                with("attention.key_length_mla", Value::U32(256)),
                "metadata deepseek2.attention.key_length_mla: is 256;",
            ),
            (
                with("attention.value_length_mla", Value::U32(256)),
                "metadata deepseek2.attention.value_length_mla: is 256;",
            ),
            (
                with("expert_gating_func", Value::U32(2)),
                "metadata deepseek2.expert_gating_func: is 2;",
            ),
            (
                with("block_count", Value::U32(47)),
                "metadata deepseek2.expert_gating_func: is not set, which ik reads as 2 \
                 (sigmoid) on a 47-layer file;",
            ),
            (
                with("expert_weights_norm", Value::Bool(true)),
                "metadata deepseek2.expert_weights_norm: is true;",
            ),
            (
                with("expert_used_count", Value::U32(65)),
                "metadata deepseek2.expert_used_count: is 65 of expert_count 64",
            ),
        ];
        let wrong: Vec<String> = cases
            .into_iter()
            .enumerate()
            .filter_map(|(i, (t, want))| match Hparams::from_meta(&t) {
                Ok(_) => Some(format!("case {i} read; expected \"{want}\"")),
                Err(e) if e.to_string().starts_with(want) => None,
                Err(e) => Some(format!("case {i}: {e}")),
            })
            .collect();
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    /// A refusal that reaches the engine is `ModelError::Metadata`, naming the
    /// key, with the reader's own text.
    #[test]
    fn a_refusal_is_a_model_metadata_error() {
        let e = Hparams::from_meta(&with("attention.q_lora_rank", Value::U32(768)))
            .expect_err("q_lora_rank is refused");
        let text = e.to_string();
        let m = crate::ModelError::from(e);
        assert_eq!(
            m.to_string(),
            text,
            "the conversion keeps the reader's text"
        );
        assert!(
            matches!(&m, crate::ModelError::Metadata { key, .. }
                if key == "deepseek2.attention.q_lora_rank"),
            "expected ModelError::Metadata for q_lora_rank, got {m:?}"
        );
    }
}
