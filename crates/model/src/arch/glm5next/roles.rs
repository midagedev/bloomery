//! The role of every tensor in a glm5next file, by exact name: the three
//! model-level names, then the per-layer stems. Every tensor of a
//! next-token layer is [`Role::Unused`]: carried, never loaded. A name the
//! table does not hold fails the file with every such name listed.

use gguf::Split;

use super::hparams::Hparams;
use crate::placement::{ModelTensor, ModelTensors, PlacementError, Role};

/// The role of a trunk layer's stem: the KDA and latent mixers with the
/// indexer, the hyper-connections, the pre-FFN norm, a dense layer's FFN,
/// the router with its selection bias, the shared expert, the routed stacks.
fn layer_role(stem: &str) -> Option<Role> {
    match stem {
        "attn_norm.weight"
        | "attn_q.weight"
        | "attn_k.weight"
        | "attn_v.weight"
        | "ssm_conv1d_q.weight"
        | "ssm_conv1d_k.weight"
        | "ssm_conv1d_v.weight"
        | "ssm_f_a.weight"
        | "ssm_f_b.weight"
        | "ssm_g_a.weight"
        | "ssm_g_b.weight"
        | "ssm_beta.weight"
        | "ssm_a"
        | "ssm_dt.bias"
        | "ssm_norm.weight"
        | "attn_q_a.weight"
        | "attn_q_a_norm.weight"
        | "attn_q_b.weight"
        | "attn_kv_a_mqa.weight"
        | "attn_kv_a_norm.weight"
        | "attn_k_b.weight"
        | "attn_v_b.weight"
        | "attn_output.weight"
        | "indexer.attn_k.weight"
        | "indexer.attn_q_b.weight"
        | "indexer.k_norm.weight"
        | "indexer.k_norm.bias"
        | "indexer.proj.weight"
        | "indexer_compressor_gate.weight"
        | "indexer_compressor_ape.weight" => Some(Role::Attention),
        "hc_attn_fn.weight"
        | "hc_attn_base.weight"
        | "hc_attn_scale.weight"
        | "hc_ffn_fn.weight"
        | "hc_ffn_base.weight"
        | "hc_ffn_scale.weight" => Some(Role::HyperConnection),
        "ffn_norm.weight" => Some(Role::FfnNorm),
        "ffn_gate.weight" | "ffn_up.weight" | "ffn_down.weight" => Some(Role::DenseFfn),
        "ffn_gate_inp.weight" | "exp_probs_b.bias" => Some(Role::Router),
        "ffn_gate_shexp.weight" | "ffn_up_shexp.weight" | "ffn_down_shexp.weight" => {
            Some(Role::SharedExpert)
        }
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" | "ffn_down_exps.weight" => {
            Some(Role::RoutedExperts)
        }
        _ => None,
    }
}

/// A next-token layer's own stems, beside the trunk's.
fn nextn_stem(stem: &str) -> bool {
    matches!(
        stem,
        "nextn.eh_proj.weight"
            | "nextn.enorm.weight"
            | "nextn.hnorm.weight"
            | "nextn.shared_head_norm.weight"
    )
}

/// A tensor's role and layer by its name; `None` when the table does not
/// hold it, its layer is past the file's, or a trunk layer carries a
/// next-token stem.
fn role(name: &str, hp: &Hparams) -> Option<(Role, Option<usize>)> {
    match name {
        "token_embd.weight" => return Some((Role::TokenEmbedding, None)),
        "output_norm.weight" | "output.weight" => return Some((Role::Head, None)),
        _ => {}
    }
    let (layer, stem) = name.strip_prefix("blk.")?.split_once('.')?;
    let layer: usize = layer.parse().ok().filter(|&l| l < hp.n_layer)?;
    if layer >= hp.n_trunk {
        return (layer_role(stem).is_some() || nextn_stem(stem))
            .then_some((Role::Unused, Some(layer)));
    }
    layer_role(stem).map(|r| (r, Some(layer)))
}

/// Every tensor of `split` with its role and header facts.
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    let mut tensors = Vec::with_capacity(split.tensor_count());
    let mut unclassified = Vec::new();
    for (shard, t) in split.iter_tensors() {
        let Some((role, layer)) = role(&t.name, hp) else {
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
        layers: hp.n_trunk,
        experts: hp.n_expert as u64,
        experts_used: hp.n_used as u64,
    })
}
