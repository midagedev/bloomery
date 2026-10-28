//! The shared Gemma encoder trunk: header checks, tensor shapes and GQA layers.

use gguf::Split;
use models::{Act, Ffn, Gqa, LayerSpec, Mixer, OutGate, Residual, Rope, RopeMode};

use crate::arch::stems::{Stem, req};
use crate::arch::{finite_positive, meta_bool, metadata, mul, optional_usize, positive};
use crate::placement::{PlacementError, Role};
use names::*;

/// `pooling_type`'s mean (llama.h `LLAMA_POOLING_TYPE_MEAN`).
const POOLING_MEAN: usize = 1;

/// The head and rope of one kind of layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Attn {
    /// `attention.key_length{,_swa}`, which `value_length{,_swa}` equals.
    pub head: usize,
    /// `rope.dimension_count{,_swa}`: the values rotated, NeoX halves.
    pub rope_dims: usize,
    /// `rope.freq_base{,_swa}`.
    pub rope_base: f32,
}

/// The head and rope keys of one kind of layer, `suffix` `""` or `"_swa"`.
/// The value head is the key head (llama.cpp throws otherwise), and the
/// rope turns an even count of values within it.
pub(crate) fn attn(split: &Split, suffix: &str, rope_dims: usize) -> Result<Attn, PlacementError> {
    let key = format!("attention.key_length{suffix}");
    let value = format!("attention.value_length{suffix}");
    let dims = format!("rope.dimension_count{suffix}");
    let base = format!("rope.freq_base{suffix}");
    let head = positive(split, &key)?;
    let v = positive(split, &value)?;
    if v != head {
        return Err(metadata(split, &value, format!("is {v}, not {key} {head}")));
    }
    if rope_dims == 0 || !rope_dims.is_multiple_of(2) || rope_dims > head {
        return Err(metadata(
            split,
            &dims,
            format!("is {rope_dims}, for a head of {head}"),
        ));
    }
    let rope_base = finite_positive(split, &base)?;
    Ok(Attn {
        head,
        rope_dims,
        rope_base,
    })
}

/// `pooling_type` is the mean, `attention.causal` is absent or false and
/// `attention.shared_kv_layers` is absent or 0: the model this reader
/// describes. RoPE scaling keys must be absent. Any other value is refused
/// by name.
pub(crate) fn encoder_keys(split: &Split) -> Result<(), PlacementError> {
    let pooling = positive(split, "pooling_type")?;
    if pooling != POOLING_MEAN {
        return Err(metadata(
            split,
            "pooling_type",
            format!("is {pooling}; this reader pools by the mean ({POOLING_MEAN})"),
        ));
    }
    let causal = "attention.causal";
    if split.value(&split.arch_key(causal)).is_some() && meta_bool(split, causal)? {
        return Err(metadata(
            split,
            causal,
            "is true; the encoder attends both ways",
        ));
    }
    let shared = "attention.shared_kv_layers";
    if let Some(n) = optional_usize(split, shared)?
        && n != 0
    {
        return Err(metadata(
            split,
            shared,
            format!("is {n}; every layer here projects its own keys and values"),
        ));
    }
    for key in [
        "rope.scaling.type",
        "rope.scaling.factor",
        "rope.scale_linear",
    ] {
        if split.value(&split.arch_key(key)).is_some() {
            return Err(metadata(split, key, "RoPE scaling is not supported"));
        }
    }
    Ok(())
}

/// Validate the per-layer lists and give each sliding layer half the window.
pub(crate) fn reach(
    n_layer: usize,
    kv_len: usize,
    sliding: &[bool],
    window: usize,
) -> Result<Vec<Option<u32>>, PlacementError> {
    for (key, len) in [
        ("attention.sliding_window_pattern", sliding.len()),
        ("attention.head_count_kv", kv_len),
    ] {
        if len != n_layer {
            return Err(PlacementError::Metadata {
                key: key.to_string(),
                detail: format!("has {len} values for {n_layer} layers"),
            });
        }
    }
    let window = crate::arch::spec_u32("attention.sliding_window / 2", window / 2)?;
    Ok(sliding.iter().map(|&s| s.then_some(window)).collect())
}

pub(crate) mod names {
    /// The RMS gain before the attention.
    pub const ATTN_NORM: &str = "attn_norm.weight";
    /// The query projection.
    pub const ATTN_Q: &str = "attn_q.weight";
    /// The key projection.
    pub const ATTN_K: &str = "attn_k.weight";
    /// The value projection. Its rows are normed with no gain.
    pub const ATTN_V: &str = "attn_v.weight";
    /// The attention output projection.
    pub const ATTN_OUTPUT: &str = "attn_output.weight";
    /// The per-head RMS gain on the queries.
    pub const ATTN_Q_NORM: &str = "attn_q_norm.weight";
    /// The per-head RMS gain on the keys.
    pub const ATTN_K_NORM: &str = "attn_k_norm.weight";
    /// The RMS gain on the attention's output, before the residual.
    pub const POST_ATTENTION_NORM: &str = "post_attention_norm.weight";

    /// The RMS gain before the feed-forward block.
    pub const FFN_NORM: &str = "ffn_norm.weight";
    /// The GELU GLU's gate.
    pub const FFN_GATE: &str = "ffn_gate.weight";
    /// The GELU GLU's up projection.
    pub const FFN_UP: &str = "ffn_up.weight";
    /// The down projection.
    pub const FFN_DOWN: &str = "ffn_down.weight";
    /// The RMS gain on the feed-forward output, before the residual.
    pub const POST_FFW_NORM: &str = "post_ffw_norm.weight";
}

/// The shared trunk's required stems.
pub(crate) const LAYER: &[Stem] = &[
    req(ATTN_NORM, Role::Attention),
    req(ATTN_Q, Role::Attention),
    req(ATTN_K, Role::Attention),
    req(ATTN_V, Role::Attention),
    req(ATTN_OUTPUT, Role::Attention),
    req(ATTN_Q_NORM, Role::Attention),
    req(ATTN_K_NORM, Role::Attention),
    req(POST_ATTENTION_NORM, Role::Attention),
    req(FFN_NORM, Role::FfnNorm),
    req(FFN_GATE, Role::DenseFfn),
    req(FFN_UP, Role::DenseFfn),
    req(FFN_DOWN, Role::DenseFfn),
    req(POST_FFW_NORM, Role::DenseFfn),
];

/// A block name's role, if it is in range and one table holds its stem.
pub(crate) fn layer_role(
    name: &str,
    n_layer: usize,
    extra: &[Stem],
) -> Option<(Role, Option<usize>)> {
    let (layer, stem) = name.strip_prefix("blk.")?.split_once('.')?;
    let layer: usize = layer.parse().ok().filter(|&l| l < n_layer)?;
    LAYER
        .iter()
        .chain(extra)
        .find(|s| s.name == stem)
        .map(|s| (s.role, Some(layer)))
}

/// One layer's shared tensors, with GGML dimensions derived from the header.
pub(crate) fn layer_tensors(
    split: &Split,
    embd: usize,
    ff: usize,
    heads: usize,
    kv: usize,
    a: Attn,
    key: &str,
) -> Result<Vec<(&'static str, Vec<u64>)>, PlacementError> {
    let e = embd as u64;
    let ff = ff as u64;
    let q = mul(split, key, heads, a.head)? as u64;
    let k = mul(split, key, kv, a.head)? as u64;
    let d = a.head as u64;
    Ok(vec![
        (names::ATTN_NORM, vec![e]),
        (names::ATTN_Q, vec![e, q]),
        (names::ATTN_K, vec![e, k]),
        (names::ATTN_V, vec![e, k]),
        (names::ATTN_OUTPUT, vec![q, e]),
        (names::ATTN_Q_NORM, vec![d]),
        (names::ATTN_K_NORM, vec![d]),
        (names::POST_ATTENTION_NORM, vec![e]),
        (names::FFN_NORM, vec![e]),
        (names::FFN_GATE, vec![e, ff]),
        (names::FFN_UP, vec![e, ff]),
        (names::FFN_DOWN, vec![ff, e]),
        (names::POST_FFW_NORM, vec![e]),
    ])
}

/// A bidirectional Gemma trunk layer; its reach belongs to EncoderSpec.
pub(crate) fn layer_spec(
    heads: usize,
    kv: usize,
    a: Attn,
    ff: usize,
    head_key: &str,
    rope_key: &str,
) -> Result<LayerSpec, PlacementError> {
    use crate::arch::spec_u32;
    let head = spec_u32(head_key, a.head)?;
    Ok(LayerSpec {
        mixer: Mixer::Gqa(Gqa {
            heads: spec_u32("attention.head_count", heads)?,
            kv_heads: spec_u32("attention.head_count_kv", kv)?,
            head_dim: head,
            value_dim: head,
            rope: Rope {
                mode: RopeMode::Neox,
                dims: spec_u32(rope_key, a.rope_dims)?,
                base: a.rope_base,
                yarn: None,
                factors: false,
            },
            qk_norm: true,
            o_norm: false,
            out_gate: OutGate::None,
            select: None,
            window: None,
            sinks: false,
            value_scale: None,
        }),
        ffn: Ffn::Dense {
            ff: spec_u32("feed_forward_length", ff)?,
            act: Act::GeGlu,
        },
        residual: Residual::Plain,
        extras: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::encoder_keys;
    use crate::arch::synthetic::{V, header_shaped};
    use crate::placement::PlacementError;

    #[test]
    fn rope_scaling_keys_are_refused_by_name() {
        let mut failures = Vec::new();
        for (case, key, value) in [
            ("type", "rope.scaling.type", V::Str("linear")),
            ("factor", "rope.scaling.factor", V::F32(2.0)),
            ("alias", "rope.scale_linear", V::F32(2.0)),
            ("type-wrong-type", "rope.scaling.type", V::Bool(true)),
            ("factor-wrong-type", "rope.scaling.factor", V::Bool(true)),
            ("alias-wrong-type", "rope.scale_linear", V::Bool(true)),
        ] {
            let path = header_shaped(
                &format!("gemma-rope-{case}"),
                "gemma-embedding2",
                &[("pooling_type", V::U32(1)), (key, value)],
                &[],
                &[],
            );
            let split = gguf::Split::open(&path).unwrap();
            let result = encoder_keys(&split);
            let full_key = split.arch_key(key);
            drop(split);
            std::fs::remove_file(&path).expect("remove the synthetic header");
            if !matches!(result, Err(PlacementError::Metadata { key, .. }) if key == full_key) {
                failures.push(format!("{case}: expected Metadata naming {full_key}"));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
