//! The role of every tensor in a qwen35moe, qwen35 or qwen4exp file, by exact
//! name: the model-level names, then the per-layer stems, one table per
//! variant (a stem one variant carries is unclassified in the others; qwen35
//! is qwen35moe's mixer and norm stems with a dense FFN's in place of the
//! routed ones, and its file may carry a decision head's tensors, which are
//! never loaded). A name the table does not hold fails the file with every
//! such name listed.

use gguf::Split;

use super::hparams::{Hparams, Variant};
use crate::arch::classify::{Counts, classify_with};
use crate::placement::{ModelTensors, PlacementError, Role};

/// The role of a stem qwen35moe and qwen35 share: the GDN layers' mixer, the
/// GQA layers' mixer, the post-attention (pre-FFN) norm.
fn trunk_role(stem: &str) -> Option<Role> {
    match stem {
        "attn_norm.weight" | "attn_qkv.weight" | "attn_gate.weight" | "ssm_conv1d.weight"
        | "ssm_dt.bias" | "ssm_a" | "ssm_alpha.weight" | "ssm_beta.weight" | "ssm_norm.weight"
        | "ssm_out.weight" | "attn_q.weight" | "attn_k.weight" | "attn_v.weight"
        | "attn_q_norm.weight" | "attn_k_norm.weight" | "attn_output.weight" => {
            Some(Role::Attention)
        }
        "post_attention_norm.weight" => Some(Role::FfnNorm),
        _ => None,
    }
}

/// The role of a qwen35moe stem: the trunk's ([`trunk_role`]), the router,
/// the shared expert with its gate, the routed stacks.
fn qwen35moe_role(stem: &str) -> Option<Role> {
    match stem {
        "ffn_gate_inp.weight" => Some(Role::Router),
        "ffn_gate_inp_shexp.weight"
        | "ffn_gate_shexp.weight"
        | "ffn_up_shexp.weight"
        | "ffn_down_shexp.weight" => Some(Role::SharedExpert),
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" | "ffn_down_exps.weight" => {
            Some(Role::RoutedExperts)
        }
        _ => trunk_role(stem),
    }
}

/// The role of a qwen35 stem: the trunk's ([`trunk_role`]) and the dense
/// FFN's three matrices.
fn qwen35_role(stem: &str) -> Option<Role> {
    match stem {
        "ffn_gate.weight" | "ffn_up.weight" | "ffn_down.weight" => Some(Role::DenseFfn),
        _ => trunk_role(stem),
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

/// Whether `name` is a tensor of the decision head a Clef-layout file
/// (`clef`, llama.cpp) carries beside the trunk: the head's blocks, its
/// `decision.*` tensors and its token types. `crates/decision` reads them; the
/// body loads none.
fn decision_head(name: &str) -> bool {
    name == "token_types.weight" || name.starts_with("dec.blk.") || name.starts_with("decision.")
}

/// A tensor's role and layer by its name; `None` when the table does not
/// hold it, or its layer is past `n_layer`. The PLE table has no layer in
/// its name and belongs to the site's.
fn role(name: &str, hp: &Hparams) -> Option<(Role, Option<usize>)> {
    let exp = hp.variant == Variant::Qwen4Exp;
    if hp.variant == Variant::Qwen35 && decision_head(name) {
        return Some((Role::Unused, None));
    }
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
    let role = match hp.variant {
        Variant::Qwen35Moe => qwen35moe_role(stem),
        Variant::Qwen35 => qwen35_role(stem),
        Variant::Qwen4Exp => qwen4exp_role(stem),
    };
    role.map(|r| (r, Some(layer)))
}

/// Every tensor of `split` with its role and header facts.
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    // The per-layer embedding table reads a row per PLE head, a count only
    // qwen4exp's header carries (`Ple::heads`); the trunk's token embedding
    // reads one.
    let ple_rows = hp
        .exp
        .as_ref()
        .and_then(|e| e.ple)
        .map(|p| p.heads() as u64);
    let mut model = classify_with(
        split,
        |name| role(name, hp),
        Counts {
            layers: hp.n_layer,
            experts: hp.n_expert as u64,
            experts_used: hp.n_used as u64,
        },
    )?;
    for t in &mut model.tensors {
        if t.role == Role::EngramTable {
            t.gathered_rows = ple_rows;
        }
    }
    Ok(model)
}
