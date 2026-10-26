//! The role of every tensor in a dense llama file, by exact name: the three
//! model-level names, then the per-layer stems. There is no prefix rule and no
//! catch-all: a name the table does not hold — an attention or FFN bias, a
//! merged `attn_qkv`, a rope frequency table, a router — fails the file with
//! every such name listed, because this chain would not read it. The tensors
//! the chain does read must all be in the file, at the dims the keys give, and
//! the head is its own matrix: the description has no tied head.

use gguf::Split;

use super::hparams::Hparams;
use crate::arch::classify::{Counts, block_stem, classify_with, model_level_role};
use crate::arch::{dims_agree, names, refuse_tensor_set};
use crate::placement::{ModelTensors, PlacementError, Role};

/// The per-layer stems every layer carries.
const LAYER: [&str; 9] = [
    "attn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];

/// The per-head q and k norms a layer carries both of or neither: the
/// description's `qk_norm`.
const QK_NORM: [&str; 2] = ["attn_q_norm.weight", "attn_k_norm.weight"];

/// The role of a per-layer stem.
fn layer_role(stem: &str) -> Option<Role> {
    match stem {
        "attn_norm.weight" | "attn_q.weight" | "attn_k.weight" | "attn_v.weight"
        | "attn_output.weight" | "attn_q_norm.weight" | "attn_k_norm.weight" => {
            Some(Role::Attention)
        }
        "ffn_norm.weight" => Some(Role::FfnNorm),
        "ffn_gate.weight" | "ffn_up.weight" | "ffn_down.weight" => Some(Role::DenseFfn),
        _ => None,
    }
}

/// A tensor's role and layer by its name; `None` when the table does not
/// hold it, or its layer is past `n_layer`.
fn role(name: &str, n_layer: usize) -> Option<(Role, Option<usize>)> {
    if let Some(model_level) = model_level_role(name) {
        return Some(model_level);
    }
    let (layer, stem) = block_stem(name, n_layer)?;
    layer_role(stem).map(|r| (r, Some(layer)))
}

/// Every tensor of `split` with its role and header facts, and the model's
/// layer count from `hp`, the file's [`Hparams`]; no expert.
pub(crate) fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    let tensors = classify_with(
        split,
        |name| role(name, hp.n_layer),
        Counts {
            layers: hp.n_layer,
            experts: 0,
            experts_used: 0,
        },
    )?;
    require(split, hp)?;
    Ok(tensors)
}

/// The dims of one layer's tensors as the keys give them, ggml order.
fn layer_dims(hp: &Hparams) -> Vec<(&'static str, Vec<u64>)> {
    let e = hp.n_embd as u64;
    let ff = hp.n_ff as u64;
    let q = (hp.n_head * hp.head_dim) as u64;
    let kv = (hp.n_head_kv * hp.head_dim) as u64;
    let d = hp.head_dim as u64;
    vec![
        ("attn_norm.weight", vec![e]),
        ("attn_q.weight", vec![e, q]),
        ("attn_k.weight", vec![e, kv]),
        ("attn_v.weight", vec![e, kv]),
        ("attn_output.weight", vec![q, e]),
        ("attn_q_norm.weight", vec![d]),
        ("attn_k_norm.weight", vec![d]),
        ("ffn_norm.weight", vec![e]),
        ("ffn_gate.weight", vec![e, ff]),
        ("ffn_up.weight", vec![e, ff]),
        ("ffn_down.weight", vec![ff, e]),
    ]
}

/// The head is a matrix of its own, every tensor the chain reads is in the
/// file, a layer's q and k norms come as a pair, and every tensor has the dims
/// the keys give it; the first tensor that is not so, by name.
fn require(split: &Split, hp: &Hparams) -> Result<(), PlacementError> {
    let head = names::output();
    if split.find(&head).is_none() {
        return Err(PlacementError::Tensor {
            name: head,
            detail: "is not in the file; the description has no tied head, and this chain runs \
                     the head as a matrix of its own"
                .to_string(),
        });
    }
    let missing: Vec<String> = [names::token_embd(), names::output_norm()]
        .into_iter()
        .chain((0..hp.n_layer).flat_map(|l| LAYER.map(|stem| names::blk(l, stem))))
        .filter(|name| split.find(name).is_none())
        .collect();
    refuse_tensor_set(&missing, &[], "a dense llama trunk needs", "")?;
    for l in 0..hp.n_layer {
        let [q, k] = QK_NORM.map(|stem| names::blk(l, stem));
        let (has_q, has_k) = (split.find(&q).is_some(), split.find(&k).is_some());
        if has_q != has_k {
            let (present, absent) = if has_q { (q, k) } else { (k, q) };
            return Err(PlacementError::Tensor {
                name: absent,
                detail: format!(
                    "is not in the file, and {present} is; a per-head q/k norm is both \
                     tensors or neither"
                ),
            });
        }
    }
    dims_agree(
        split,
        hp.n_embd,
        hp.n_vocab,
        (0..hp.n_layer).map(|_| layer_dims(hp)),
    )
}

#[cfg(test)]
mod tests {
    use super::{LAYER, QK_NORM, role};
    use crate::placement::Role;

    /// Every name of the table has its role on its layer; a name the table
    /// does not hold, or a layer past the model, has none, so `classify`
    /// lists it.
    #[test]
    fn the_table_is_exact() {
        assert_eq!(
            role("token_embd.weight", 52),
            Some((Role::TokenEmbedding, None))
        );
        assert_eq!(role("output.weight", 52), Some((Role::Head, None)));
        assert_eq!(role("output_norm.weight", 52), Some((Role::Head, None)));
        for stem in LAYER.into_iter().chain(QK_NORM) {
            let name = format!("blk.51.{stem}");
            assert!(
                matches!(role(&name, 52), Some((_, Some(51)))),
                "{name} has no role"
            );
        }
        assert_eq!(
            role("blk.3.ffn_norm.weight", 52),
            Some((Role::FfnNorm, Some(3)))
        );
        assert_eq!(
            role("blk.3.ffn_down.weight", 52),
            Some((Role::DenseFfn, Some(3)))
        );
        for name in [
            "blk.52.attn_q.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.attn_q.bias",
            "blk.0.attn_output.bias",
            "blk.0.ffn_up.bias",
            "blk.0.ffn_gate_inp.weight",
            "blk.0.ffn_up_exps.weight",
            "blk.0.post_attention_norm.weight",
            "rope_freqs.weight",
            "output.bias",
        ] {
            assert_eq!(role(name, 52), None, "{name}");
        }
    }
}
