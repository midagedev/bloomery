//! The role of every tensor in a qwen35moe file, by exact name: the three
//! model-level names, then the per-layer stems. A name the table does not
//! hold fails the file with every such name listed.

use gguf::Split;

use super::hparams::Hparams;
use crate::placement::{ModelTensor, ModelTensors, PlacementError, Role};

/// The role of a per-layer stem: the GDN layers' mixer, the GQA layers'
/// mixer, the post-attention (pre-FFN) norm, the router, the shared expert
/// with its gate, the routed stacks.
fn layer_role(stem: &str) -> Option<Role> {
    match stem {
        "attn_norm.weight" | "attn_qkv.weight" | "attn_gate.weight" | "ssm_conv1d.weight"
        | "ssm_dt.bias" | "ssm_a" | "ssm_alpha.weight" | "ssm_beta.weight" | "ssm_norm.weight"
        | "ssm_out.weight" | "attn_q.weight" | "attn_k.weight" | "attn_v.weight"
        | "attn_q_norm.weight" | "attn_k_norm.weight" | "attn_output.weight" => {
            Some(Role::Attention)
        }
        "post_attention_norm.weight" => Some(Role::FfnNorm),
        "ffn_gate_inp.weight" => Some(Role::Router),
        "ffn_gate_inp_shexp.weight"
        | "ffn_gate_shexp.weight"
        | "ffn_up_shexp.weight"
        | "ffn_down_shexp.weight" => Some(Role::SharedExpert),
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" | "ffn_down_exps.weight" => {
            Some(Role::RoutedExperts)
        }
        _ => None,
    }
}

/// A tensor's role and layer by its name; `None` when the table does not
/// hold it, or its layer is past `n_layer`.
fn role(name: &str, n_layer: usize) -> Option<(Role, Option<usize>)> {
    match name {
        "token_embd.weight" => return Some((Role::TokenEmbedding, None)),
        "output_norm.weight" | "output.weight" => return Some((Role::Head, None)),
        _ => {}
    }
    let (layer, stem) = name.strip_prefix("blk.")?.split_once('.')?;
    let layer: usize = layer.parse().ok().filter(|&l| l < n_layer)?;
    layer_role(stem).map(|r| (r, Some(layer)))
}

/// Every tensor of `split` with its role and header facts.
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
        experts: hp.n_expert as u64,
        experts_used: hp.n_used as u64,
    })
}
