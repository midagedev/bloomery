//! The role of every tensor in a mimo2 file, by exact name: the three
//! model-level names, then the per-layer stems — one table, by the group a
//! layer carries ([`MIXER`] … [`NEXTN`]), which the header reader's tensor
//! check reads too. The value projection writes a narrower head than the key
//! (`attn_qkv`'s row count and `attn_output`'s width say which), and a
//! sliding-window layer may carry attention sinks a full layer never does.
//! Every tensor of a next-token layer is [`Role::Unused`]: carried, never
//! loaded. A name the table does not hold fails the file with every such
//! name listed.

use gguf::Split;

use super::hparams::Hparams;
use crate::arch::classify::{Counts, classify_with};
use crate::placement::{ModelTensors, PlacementError, Role};

/// A tensor a layer of some kind carries, by stem: its role, and whether the
/// file must carry it (llama.cpp creates it without `TENSOR_NOT_REQUIRED`).
#[derive(Clone, Copy, Debug)]
pub(super) struct Stem {
    pub(super) name: &'static str,
    pub(super) role: Role,
    pub(super) required: bool,
}

const fn req(name: &'static str, role: Role) -> Stem {
    Stem {
        name,
        role,
        required: true,
    }
}

const fn opt(name: &'static str, role: Role) -> Stem {
    Stem {
        name,
        role,
        required: false,
    }
}

/// Every layer's mixer block beside the projections: the norms. The fused
/// QKV matrix (`attn_qkv`) and the split one (`attn_q` + `attn_k` +
/// `attn_v`) are alternatives, exactly one of which a layer carries
/// (`create_tensor_qkv`, src/models/mimo2.cpp:53); the sinks are the
/// sliding-window layers' alone (mimo2.cpp:57, `TENSOR_NOT_REQUIRED`).
pub(super) const MIXER: &[Stem] = &[
    req("attn_norm.weight", Role::Attention),
    req("attn_output.weight", Role::Attention),
    opt("attn_sinks.weight", Role::Attention),
];

/// The fused qkv matrix, `[embedding, q·key + kv·(key + value)]` rows.
pub(super) const QKV_FUSED: Stem = req("attn_qkv.weight", Role::Attention);

/// The split projections, in the q, k, v order llama.cpp reads them.
pub(super) const QKV_SPLIT: &[Stem] = &[
    req("attn_q.weight", Role::Attention),
    req("attn_k.weight", Role::Attention),
    req("attn_v.weight", Role::Attention),
];

/// A dense layer's feed-forward block.
pub(super) const DENSE: &[Stem] = &[
    req("ffn_norm.weight", Role::FfnNorm),
    req("ffn_gate.weight", Role::DenseFfn),
    req("ffn_up.weight", Role::DenseFfn),
    req("ffn_down.weight", Role::DenseFfn),
];

/// A routed layer's feed-forward block: the router with its selection bias,
/// which the graph adds for the selection only, and the routed stacks.
pub(super) const MOE: &[Stem] = &[
    req("ffn_norm.weight", Role::FfnNorm),
    req("ffn_gate_inp.weight", Role::Router),
    opt(SELECTION_BIAS, Role::Router),
    req("ffn_gate_exps.weight", Role::RoutedExperts),
    req("ffn_up_exps.weight", Role::RoutedExperts),
    req("ffn_down_exps.weight", Role::RoutedExperts),
];

/// The router's selection bias (`ffn_exp_probs_b`), which the file carries
/// per layer (mimo2.cpp:72) and the graph keeps out of the weights.
pub(super) const SELECTION_BIAS: &str = "exp_probs_b.bias";

/// A next-token layer's own stems beside the trunk's: the input projection of
/// the two norms, the two norms, and the head's own norm and matrices some
/// conversions write instead of sharing the model's (mimo2.cpp:74-81, every
/// one of these four `TENSOR_NOT_REQUIRED` but the first three).
pub(super) const NEXTN: &[Stem] = &[
    req("nextn.eh_proj.weight", Role::Unused),
    req("nextn.enorm.weight", Role::Unused),
    req("nextn.hnorm.weight", Role::Unused),
    opt("nextn.embed_tokens.weight", Role::Unused),
    opt("nextn.shared_head_head.weight", Role::Unused),
    opt("nextn.shared_head_norm.weight", Role::Unused),
    opt("layer_output_norm.weight", Role::Unused),
];

/// Every stem a trunk layer may carry, by group.
const TRUNK: [&[Stem]; 4] = [MIXER, DENSE, MOE, QKV_SPLIT];

/// The names of `stems` the file must carry.
pub(super) fn required(stems: &[Stem]) -> impl Iterator<Item = &'static str> + '_ {
    stems.iter().filter(|s| s.required).map(|s| s.name)
}

/// The role of a trunk layer's stem.
pub(super) fn layer_role(stem: &str) -> Option<Role> {
    if stem == QKV_FUSED.name {
        return Some(QKV_FUSED.role);
    }
    TRUNK
        .iter()
        .flat_map(|g| g.iter())
        .find(|s| s.name == stem)
        .map(|s| s.role)
}

/// A next-token layer's own stems, beside the trunk's.
pub(super) fn nextn_stem(stem: &str) -> bool {
    NEXTN.iter().any(|s| s.name == stem)
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
    classify_with(
        split,
        |name| role(name, hp),
        Counts {
            layers: hp.n_trunk,
            experts: hp.n_expert as u64,
            experts_used: hp.n_used as u64,
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::placement::Role;

    use super::{MIXER, NEXTN, QKV_SPLIT, Stem, TRUNK, layer_role, nextn_stem};

    /// A stem in two groups names one role, and no group lists a stem twice;
    /// the fused qkv matrix and each split projection are the only stems
    /// their counterpart's check refuses by name.
    #[test]
    fn a_stem_has_one_role() {
        let all: Vec<&Stem> = TRUNK.iter().flat_map(|g| g.iter()).collect();
        for s in &all {
            assert_eq!(layer_role(s.name), Some(s.role), "{}", s.name);
        }
        for g in TRUNK.iter().chain([&NEXTN]) {
            for (i, s) in g.iter().enumerate() {
                assert!(!g[..i].iter().any(|t| t.name == s.name), "{} twice", s.name);
            }
        }
        assert_eq!(layer_role("attn_qkv.weight"), Some(Role::Attention));
        assert!(NEXTN.iter().all(|s| layer_role(s.name).is_none()));
        assert!(MIXER.iter().chain(QKV_SPLIT).all(|s| !nextn_stem(s.name)));
    }
}
