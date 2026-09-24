//! The role of every tensor in a Qwen3-MoE file, by exact name: the three
//! model-level names, then the per-layer stems of [`names::LAYER`]. There is
//! no prefix rule and no catch-all: a name the table does not hold — a merged
//! `attn_qkv`, a dense `ffn_up`, a shared expert, a router bias — fails the
//! file with every such name listed, because this chain would not read it.

use gguf::Split;

use super::hparams::Hparams;
use super::names;
use crate::placement::{ModelTensor, ModelTensors, PlacementError, Role};

/// The role of a per-layer stem of [`names::LAYER`].
fn layer_role(stem: &str) -> Option<Role> {
    match stem {
        "attn_norm.weight" | "attn_q.weight" | "attn_k.weight" | "attn_v.weight"
        | "attn_q_norm.weight" | "attn_k_norm.weight" | "attn_output.weight" => {
            Some(Role::Attention)
        }
        "ffn_norm.weight" => Some(Role::FfnNorm),
        "ffn_gate_inp.weight" => Some(Role::Router),
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" | "ffn_down_exps.weight" => {
            Some(Role::RoutedExperts)
        }
        _ => None,
    }
}

/// A tensor's role and layer by its name; `None` when the table does not
/// hold it, or its layer is past `n_layer`.
fn role(name: &str, n_layer: usize) -> Option<(Role, Option<usize>)> {
    if name == names::token_embd() {
        return Some((Role::TokenEmbedding, None));
    }
    if name == names::output_norm() || name == names::output() {
        return Some((Role::Head, None));
    }
    let (layer, stem) = name.strip_prefix("blk.")?.split_once('.')?;
    let layer: usize = layer.parse().ok().filter(|&l| l < n_layer)?;
    layer_role(stem).map(|r| (r, Some(layer)))
}

/// Every tensor of `split` with its role and header facts, and the model's
/// layer and expert counts from `hp`, the file's [`Hparams`].
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    let mut tensors = Vec::with_capacity(split.tensor_count());
    let mut unclassified = Vec::new();
    for (shard, t) in split.iter_tensors() {
        let Some((role, layer)) = role(&t.name, hp.n_layer) else {
            unclassified.push(t.name.clone());
            continue;
        };
        tensors.push(ModelTensor {
            name: t.name.clone(),
            shard,
            layer,
            role,
            ty: t.ty,
            dims: t.dims.clone(),
            file_bytes: t.nbytes,
            gathered_rows: (role == Role::TokenEmbedding).then_some(1),
        });
    }
    if !unclassified.is_empty() {
        return Err(PlacementError::Unclassified {
            names: unclassified,
        });
    }
    Ok(ModelTensors {
        tensors,
        layers: hp.n_layer,
        experts: hp.experts.n_expert as u64,
        experts_used: hp.experts.n_used as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::role;
    use crate::arch::qwen3moe::names;
    use crate::placement::Role;

    /// Every name of the table has its role on its layer; a name the table
    /// does not hold, or a layer past the model, has none, so `classify`
    /// lists it.
    #[test]
    fn the_table_is_exact() {
        assert_eq!(
            role("token_embd.weight", 48),
            Some((Role::TokenEmbedding, None))
        );
        assert_eq!(role("output.weight", 48), Some((Role::Head, None)));
        assert_eq!(role("output_norm.weight", 48), Some((Role::Head, None)));
        for stem in names::LAYER {
            let name = format!("blk.47.{stem}");
            assert!(
                matches!(role(&name, 48), Some((_, Some(47)))),
                "{name} has no role"
            );
        }
        for name in [
            "blk.48.attn_q.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_up_shexp.weight",
            "blk.0.exp_probs_b.bias",
            "blk.0.attn_q.bias",
            "rope_freqs.weight",
            "output.bias",
        ] {
            assert_eq!(role(name, 48), None, "{name}");
        }
    }
}
