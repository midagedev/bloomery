//! The role of every tensor in a qwen35moe or qwen4exp file, by exact name:
//! the model-level names, then the per-layer stems, one table per variant (a
//! stem one variant carries is unclassified in the other). A name the table
//! does not hold fails the file with every such name listed.

use gguf::Split;

use super::hparams::{Hparams, Variant};
use crate::placement::{ModelTensor, ModelTensors, PlacementError, Role};

/// The role of a qwen35moe stem: the GDN layers' mixer, the GQA layers'
/// mixer, the post-attention (pre-FFN) norm, the router, the shared expert
/// with its gate, the routed stacks.
fn qwen35moe_role(stem: &str) -> Option<Role> {
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

/// The role of a qwen4exp stem: the GDN and GQA mixers with the indexer, the
/// two hyper-connection modules, the router, the shared expert, the routed
/// stacks, the PLE site's projections and its gains (the norms' and the
/// conv's, which no gemv reads). No block norm: the modules' own replace it.
fn qwen4exp_role(stem: &str) -> Option<Role> {
    match stem {
        "attn_qkv.weight"
        | "attn_gate.weight"
        | "ssm_conv1d.weight"
        | "ssm_dt.bias"
        | "ssm_a"
        | "ssm_alpha.weight"
        | "ssm_beta.weight"
        | "ssm_norm.weight"
        | "ssm_out.weight"
        | "attn_q.weight"
        | "attn_k.weight"
        | "attn_v.weight"
        | "attn_q_norm.weight"
        | "attn_k_norm.weight"
        | "attn_output.weight"
        | "indexer.q_proj.weight"
        | "indexer.k_proj.weight"
        | "indexer.q_norm.weight"
        | "indexer.k_norm.weight" => Some(Role::Attention),
        "hc_attn_norm.weight"
        | "hc_attn_down.weight"
        | "hc_attn_up.weight"
        | "hc_attn_inject.weight"
        | "hc_ffn_norm.weight"
        | "hc_ffn_down.weight"
        | "hc_ffn_up.weight"
        | "hc_ffn_inject.weight" => Some(Role::HyperConnection),
        "ffn_gate_inp.weight" => Some(Role::Router),
        "ffn_gate_inp_shexp.weight"
        | "ffn_gate_shexp.weight"
        | "ffn_up_shexp.weight"
        | "ffn_down_shexp.weight" => Some(Role::SharedExpert),
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" | "ffn_down_exps.weight" => {
            Some(Role::RoutedExperts)
        }
        "ple_key.weight" | "ple_value.weight" => Some(Role::EngramDense),
        "ple_norm_key.weight"
        | "ple_norm_query.weight"
        | "ple_norm_conv.weight"
        | "ple_conv1d.weight" => Some(Role::EngramGain),
        _ => None,
    }
}

/// A tensor's role and layer by its name; `None` when the table does not
/// hold it, or its layer is past `n_layer`. The PLE table has no layer in
/// its name and belongs to the site's.
fn role(name: &str, hp: &Hparams) -> Option<(Role, Option<usize>)> {
    let exp = hp.variant == Variant::Qwen4Exp;
    match name {
        "token_embd.weight" => return Some((Role::TokenEmbedding, None)),
        "output.weight" => return Some((Role::Head, None)),
        "output_norm.weight" if !exp => return Some((Role::Head, None)),
        "output_hc_norm.weight" | "output_hc_down.weight" | "output_hc_up.weight" if exp => {
            return Some((Role::Head, None));
        }
        "per_layer_token_embd.weight" => {
            let site = hp.exp.as_ref()?.ple?.layer;
            return Some((Role::EngramTable, Some(site)));
        }
        _ => {}
    }
    let (layer, stem) = name.strip_prefix("blk.")?.split_once('.')?;
    let layer: usize = layer.parse().ok().filter(|&l| l < hp.n_layer)?;
    let role = if exp {
        qwen4exp_role(stem)
    } else {
        qwen35moe_role(stem)
    };
    role.map(|r| (r, Some(layer)))
}

/// Every tensor of `split` with its role and header facts.
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    let ple_rows = hp
        .exp
        .as_ref()
        .and_then(|e| e.ple)
        .map(|p| p.heads() as u64);
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
            gathered_rows: match role {
                Role::TokenEmbedding => Some(1),
                Role::EngramTable => ple_rows,
                _ => None,
            },
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
