//! The V4.1 model file's rope and norm constants, as the gates of the ops
//! that turn and normalize rows read them from its metadata; and the kernels
//! the step launches for a projection, as the file's tensor types decide
//! them (the step and skew gates' shadow tables).

use crate::{GateError, expect_arch};
use gguf::{GgmlType, Split, Value};
use model::arch::Arch;
use model::arch::deepseek41::names;

/// The file's rope and norm constants, from its metadata, with its two
/// ropes as `S` (the device crate's `RopeSpec`).
pub struct RopeMeta<S> {
    /// Values a rope turns at the tail of a row: `rope.dimension_count`.
    pub n_dims: usize,
    /// The rope of the heads of a layer with no compressed stream.
    pub window: S,
    /// YaRN: the heads of a layer with a stream, the pooled rows, the index
    /// keys and the indexer query.
    pub yarn: S,
    /// `attention.compress_ratios`, one per layer.
    pub ratios: Vec<u64>,
    /// The norms' epsilon: `attention.layer_norm_rms_epsilon`.
    pub eps: f32,
    /// Rows of a layer's raw window ring: `attention.sliding_window`.
    pub ring: usize,
}

impl<S> RopeMeta<S> {
    /// The constants of `split`, which must be a V4.1 file ([`expect_arch`]);
    /// `recipe` is the `just` recipe the error for any other file names. The
    /// ropes are built by `window` and `yarn` — `RopeSpec::window` and
    /// `RopeSpec::yarn`, whose argument orders these are. They come in as
    /// arguments because this library's source does not name the V4.1 device
    /// crate: named here, it would be linked, device bundle and all, into
    /// every gate built with the feature, the ones that launch none of its
    /// kernels too.
    pub fn read(
        split: &Split,
        recipe: &str,
        window: fn(f32, usize) -> S,
        yarn: fn(f32, f32, i32, f32, f32, usize) -> S,
    ) -> Result<RopeMeta<S>, GateError> {
        expect_arch(split, Arch::Deepseek41, recipe)?;
        let f = |s: &str| -> Result<f32, GateError> {
            split
                .arch_get_f32(s)
                .ok_or_else(|| format!("metadata {} missing", split.arch_key(s)).into())
        };
        let u = |s: &str| -> Result<u64, GateError> {
            split
                .arch_get_u64(s)
                .ok_or_else(|| format!("metadata {} missing", split.arch_key(s)).into())
        };
        let scaling = split.arch_get_str("rope.scaling.type");
        if scaling != Some("yarn") {
            return Err(format!("rope.scaling.type is {scaling:?}, want \"yarn\"").into());
        }
        let n_dims = usize::try_from(u("rope.dimension_count")?)?;
        let n_ctx_orig = i32::try_from(u("rope.scaling.original_context_length")?)?;
        let key = split.arch_key("attention.compress_ratios");
        let Some(Value::Array(list)) = split.value(&key) else {
            return Err(format!("metadata {key} is absent or not an array").into());
        };
        let ratios = list
            .iter()
            .map(|v| v.as_unsigned().ok_or_else(|| format!("{key} holds {v:?}")))
            .collect::<Result<Vec<u64>, String>>()?;
        Ok(RopeMeta {
            n_dims,
            window: window(f("rope.freq_base")?, n_dims),
            yarn: yarn(
                f("attention.compress_rope_freq_base")?,
                f("rope.scaling.factor")?,
                n_ctx_orig,
                f("rope.scaling.yarn_beta_fast")?,
                f("rope.scaling.yarn_beta_slow")?,
                n_dims,
            ),
            ratios,
            eps: f("attention.layer_norm_rms_epsilon")?,
            ring: usize::try_from(u("attention.sliding_window")?)?,
        })
    }

    /// The rope of layer `l`'s heads: ik's `use_compress_rope` is
    /// `compress_ratios[l] != 0`.
    pub fn head_spec(&self, l: usize) -> Result<(&S, &'static str), GateError> {
        match self.ratios.get(l) {
            Some(0) => Ok((&self.window, "window")),
            Some(_) => Ok((&self.yarn, "yarn")),
            None => Err(format!("layer {l} has no compress ratio").into()),
        }
    }
}

/// Tensor `name`'s type in `split`.
fn type_of(split: &Split, name: &str) -> Result<GgmlType, GateError> {
    split
        .find(name)
        .map(|(_, t)| t.ty)
        .ok_or_else(|| format!("{name} is not in the file").into())
}

/// The kernels the step launches for one dense projection of file type `ty`,
/// in stream order: the q8_1 form of its input first where a K-quant gemv
/// reads one and the step makes it for this projection alone.
fn projection(ty: GgmlType, name: &str) -> Result<&'static [&'static str], GateError> {
    match ty {
        GgmlType::Q8_0 => Ok(&["q8_0_gemv"]),
        GgmlType::Q3_K => Ok(&["q3k_quantize_q8_1", "q3k_gemv"]),
        GgmlType::Q4_K => Ok(&["q3k_quantize_q8_1", "q4k_gemv"]),
        GgmlType::Q5_K => Ok(&["ds41_q5k_gemv_f32"]),
        other => Err(format!("{name}: {other} has no dense kernel").into()),
    }
}

/// The MoE piece's own work in layer `l`'s host-leg shadow, in stream order:
/// HC_PRE, with card experts the routed gate·up, h's q8_1 and the routed
/// down, then the shared expert's gate·up (`ds41_shexp_gate_up`, or its
/// q3_K twin when both weights are q3_K) and down ([`projection`]).
pub fn shadow_kernels(
    split: &Split,
    l: usize,
    card_experts: bool,
) -> Result<Vec<&'static str>, GateError> {
    let mut k = vec!["ds41_hc_pre"];
    if card_experts {
        k.extend(["ds41_expert_gate_up", "q3k_quantize_q8_1", "q4k_gemv_sel"]);
    }
    let gate = type_of(split, &names::ffn_gate_shexp(l))?;
    let up = type_of(split, &names::ffn_up_shexp(l))?;
    k.push(if (gate, up) == (GgmlType::Q3_K, GgmlType::Q3_K) {
        "ds41_shexp_gate_up_q3k"
    } else {
        "ds41_shexp_gate_up"
    });
    let down = names::ffn_down_shexp(l);
    k.extend(projection(type_of(split, &down)?, &down)?);
    Ok(k)
}

/// Engram site `l`'s token-only work, in stream order: its rows decoded
/// (`ds41_glue_engram_rows`, or `_q3k` for a q3_K table), `engram_wkv`
/// ([`projection`]) and the key norm. It reads the step image and weights
/// alone.
pub fn engram_kv_kernels(split: &Split, l: usize) -> Result<Vec<&'static str>, GateError> {
    let table = names::engram_embd(l);
    let mut k = vec![match type_of(split, &table)? {
        GgmlType::Q8_0 => "ds41_glue_engram_rows",
        GgmlType::Q3_K => "ds41_glue_engram_rows_q3k",
        other => return Err(format!("{table}: {other} has no row kernel").into()),
    }];
    let wkv = names::engram_wkv(l);
    k.extend(projection(type_of(split, &wkv)?, &wkv)?);
    k.push("ds41_engram_key_norm");
    Ok(k)
}
