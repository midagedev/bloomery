//! The role of every tensor in a qwen35moe, qwen35 or qwen4exp file, by exact
//! name: the model-level names, then the per-layer stems, one table per
//! variant (a stem one variant carries is unclassified in the others; qwen35
//! is qwen35moe's mixer and norm stems with a dense FFN's in place of the
//! routed ones, and its file may carry a decision head's tensors, which are
//! never loaded). A name the table does not hold fails the file with every
//! such name listed. A layer at or past the trunk (`nextn_predict_layers`
//! next-token layers close the file) is never loaded: any of its variant's
//! layer stems or next-token stems is [`Role::Unused`], and a trunk layer
//! carrying a next-token stem is unclassified.

// PIN(2026-10-14): draft-role checks removed with nextn_role, which had no engine consumer.

use gguf::Split;

use super::hparams::{Hparams, Variant};
use crate::arch::classify::{Counts, block_role_with_nextn, block_stem, classify_with};
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

/// A next-token layer's own stems in every variant's file: the input join and
/// its two norms, and the head's norm (ik's tensor tables,
/// `src/llama-model.cpp:667-670` for qwen35moe, :699-702 for qwen35).
const NEXTN: &[&str] = &[
    super::mtp::EH_PROJ,
    super::mtp::ENORM,
    super::mtp::HNORM,
    super::mtp::SHARED_HEAD_NORM,
];

/// The gated-residual head a qwen4exp next-token layer carries beside
/// [`NEXTN`] (`src/llama-model.cpp:549-555`).
const NEXTN_HC: &[&str] = &[
    "nextn.hc_head_norm.weight",
    "nextn.hc_head_down.weight",
    "nextn.hc_head_up.weight",
];

/// Whether `stem` is one of a next-token layer's own in `variant`'s file.
fn nextn_stem(stem: &str, variant: Variant) -> bool {
    NEXTN.contains(&stem) || (variant == Variant::Qwen4Exp && NEXTN_HC.contains(&stem))
}

/// The role of a trunk layer's `stem` in `variant`'s file.
fn layer_role(stem: &str, variant: Variant) -> Option<Role> {
    match variant {
        Variant::Qwen35Moe => qwen35moe_role(stem),
        Variant::Qwen35 => qwen35_role(stem),
        Variant::Qwen4Exp => qwen4exp_role(stem),
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
/// hold it, its layer is past `n_layer`, or a trunk layer carries a
/// next-token stem. The PLE table has no layer in its name and belongs to the
/// site's. A tensor's scale (an F8 or NVFP4 table's, an NVFP4 stack's) is the
/// shared walk's ([`classify_with`]), on its tensor's layer.
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
    block_role_with_nextn(
        name,
        hp.n_layer,
        hp.n_trunk,
        |stem| layer_role(stem, hp.variant),
        |stem| nextn_stem(stem, hp.variant),
    )
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
        Counts::new(hp.n_trunk, hp.n_expert, hp.n_used),
    )?;
    for t in &mut model.tensors {
        if t.role == Role::EngramTable {
            t.gathered_rows = ple_rows;
        }
    }
    // Body38 reads these bf16 tensors in their words: the matrices through
    // the bf16 site, head and embedding rows, the router and its shared
    // gate through the router's bf16 entries when their file types match on
    // every layer; otherwise both are widened before joining. Every other one
    // (the convs, the injects, the parts of its joined beta/alpha stack) is
    // read as f32 values;
    // Qwen3.5's body reads only its embedding in its words (a checkpoint folder's table as stored).
    const WORDS: fn(&crate::placement::ModelTensor) -> bool = |t| {
        let stem = block_stem(&t.name, usize::MAX).map_or(t.name.as_str(), |(_, stem)| stem);
        matches!(
            stem,
            "output.weight"
                | "token_embd.weight"
                | "output_hc_down.weight"
                | "output_hc_up.weight"
                | "attn_qkv.weight"
                | "attn_gate.weight"
                | "ssm_out.weight"
                | "attn_q.weight"
                | "attn_k.weight"
                | "attn_v.weight"
                | "attn_output.weight"
                | "indexer.q_proj.weight"
                | "indexer.k_proj.weight"
                | "hc_attn_down.weight"
                | "hc_attn_up.weight"
                | "hc_ffn_down.weight"
                | "hc_ffn_up.weight"
                | "ffn_gate_shexp.weight"
                | "ffn_up_shexp.weight"
                | "ffn_down_shexp.weight"
                | "ffn_gate_inp.weight"
                | "ffn_gate_inp_shexp.weight"
                | "ple_key.weight"
                | "ple_value.weight"
        )
    };
    let pairs_match = (0..hp.n_trunk).all(|l| {
        match (
            split.find(&super::names::ffn_gate_inp(l)),
            split.find(&super::names::ffn_gate_inp_shexp(l)),
        ) {
            (Some((_, r)), Some((_, s))) => r.ty == s.ty,
            _ => false,
        }
    });
    model.bf16_words = match (hp.variant, pairs_match) {
        (Variant::Qwen4Exp, true) => WORDS,
        (Variant::Qwen4Exp, false) => |t| {
            WORDS(t) && t.role != Role::Router && !t.name.ends_with(".ffn_gate_inp_shexp.weight")
        },
        _ => |t| t.name == "token_embd.weight",
    };
    Ok(model)
}
