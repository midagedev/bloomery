//! The role of every tensor in a glm5next file, by exact name: the three
//! model-level names, then the per-layer stems — one table, by the group a
//! layer's kind carries ([`KDA`] … [`NEXTN`]), which the header reader's
//! tensor check reads too. Every tensor of a next-token layer is
//! [`Role::Unused`]: carried, never loaded. A name the table does not hold
//! fails the file with every such name listed.

use gguf::Split;

use super::hparams::Hparams;
use crate::arch::classify::{Counts, classify_with};
use crate::placement::{ModelTensors, PlacementError, Role};

/// A tensor a layer of some kind carries, by stem: its role, and whether
/// the file must carry it (ik creates it without `TENSOR_NOT_REQUIRED`).
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

/// A KDA delta-rule layer's mixer; `ssm_f_b` and `ssm_g_b` are required
/// because without them ik takes another decay rule (`src/llama-kda.cpp`,
/// `build_kda_beta_gate`).
pub(super) const KDA: &[Stem] = &[
    req("attn_norm.weight", Role::Attention),
    req("attn_q.weight", Role::Attention),
    req("attn_k.weight", Role::Attention),
    req("attn_v.weight", Role::Attention),
    req("ssm_conv1d_q.weight", Role::Attention),
    req("ssm_conv1d_k.weight", Role::Attention),
    req("ssm_conv1d_v.weight", Role::Attention),
    req("ssm_f_a.weight", Role::Attention),
    req("ssm_f_b.weight", Role::Attention),
    req("ssm_g_a.weight", Role::Attention),
    req("ssm_g_b.weight", Role::Attention),
    req("ssm_beta.weight", Role::Attention),
    req("ssm_a", Role::Attention),
    req("ssm_dt.bias", Role::Attention),
    req("ssm_norm.weight", Role::Attention),
    req("attn_output.weight", Role::Attention),
];

/// A latent attention layer's mixer with its indexer, which is required
/// because without it ik runs dense attention (`src/llama-load-tensors.cpp`,
/// `create_glm5next_tensors`).
pub(super) const LATENT: &[Stem] = &[
    req("attn_norm.weight", Role::Attention),
    req("attn_q_a.weight", Role::Attention),
    req("attn_q_a_norm.weight", Role::Attention),
    req("attn_q_b.weight", Role::Attention),
    req("attn_kv_a_mqa.weight", Role::Attention),
    req("attn_kv_a_norm.weight", Role::Attention),
    req("attn_k_b.weight", Role::Attention),
    req("attn_v_b.weight", Role::Attention),
    req("attn_output.weight", Role::Attention),
    req("indexer.attn_k.weight", Role::Attention),
    req("indexer.attn_q_b.weight", Role::Attention),
    req("indexer.k_norm.weight", Role::Attention),
    req("indexer.k_norm.bias", Role::Attention),
    req("indexer.proj.weight", Role::Attention),
    req("indexer_compressor_gate.weight", Role::Attention),
    req("indexer_compressor_ape.weight", Role::Attention),
];

/// A dense layer's feed-forward block.
pub(super) const DENSE: &[Stem] = &[
    req("ffn_norm.weight", Role::FfnNorm),
    req("ffn_gate.weight", Role::DenseFfn),
    req("ffn_up.weight", Role::DenseFfn),
    req("ffn_down.weight", Role::DenseFfn),
];

/// The router's selection bias, which ik loads when present: the spec's
/// `Router::bias` is its presence.
pub(super) const SELECTION_BIAS: &str = "exp_probs_b.bias";

/// A routed layer's feed-forward block: the router with its selection bias,
/// and the routed stacks.
pub(super) const MOE: &[Stem] = &[
    req("ffn_norm.weight", Role::FfnNorm),
    req("ffn_gate_inp.weight", Role::Router),
    opt(SELECTION_BIAS, Role::Router),
    req("ffn_gate_exps.weight", Role::RoutedExperts),
    req("ffn_up_exps.weight", Role::RoutedExperts),
    req("ffn_down_exps.weight", Role::RoutedExperts),
];

/// A routed layer's shared expert, when the file has one.
pub(super) const SHARED: &[Stem] = &[
    req("ffn_gate_shexp.weight", Role::SharedExpert),
    req("ffn_up_shexp.weight", Role::SharedExpert),
    req("ffn_down_shexp.weight", Role::SharedExpert),
];

/// The hyper-connection tensors: every trunk layer's, no next-token layer's.
pub(super) const HC: &[Stem] = &[
    req("hc_attn_fn.weight", Role::HyperConnection),
    req("hc_attn_base.weight", Role::HyperConnection),
    req("hc_attn_scale.weight", Role::HyperConnection),
    req("hc_ffn_fn.weight", Role::HyperConnection),
    req("hc_ffn_base.weight", Role::HyperConnection),
    req("hc_ffn_scale.weight", Role::HyperConnection),
];

/// A next-token layer's own stems, beside the trunk's; `shared_head_norm` is
/// optional (the block shares the trunk's embedding and head).
pub(super) const NEXTN: &[Stem] = &[
    req("nextn.eh_proj.weight", Role::Unused),
    req("nextn.enorm.weight", Role::Unused),
    req("nextn.hnorm.weight", Role::Unused),
    opt("nextn.shared_head_norm.weight", Role::Unused),
];

/// Every stem a trunk layer may carry, by group.
const TRUNK: [&[Stem]; 6] = [KDA, LATENT, DENSE, MOE, SHARED, HC];

/// The groups a next-token layer carries besides its own stems: a latent
/// mixer, a routed block and a shared expert, no hyper-connection.
const NEXTN_BLOCK: [&[Stem]; 3] = [LATENT, MOE, SHARED];

/// The role a next-token layer's stem takes in the NextN load's own plan
/// (`place::NextnInputs`), where the layer is loaded: its mixer, routed block
/// and shared expert in their trunk roles, the input projection and its two
/// norms on the card as the mixer's input, the head's norm as the head's.
/// `None` for a stem no next-token layer carries. The file's own plan keeps
/// every such tensor [`Role::Unused`] ([`classify`]).
#[must_use]
pub fn nextn_role(stem: &str) -> Option<Role> {
    if stem == "nextn.shared_head_norm.weight" {
        return Some(Role::Head);
    }
    if nextn_stem(stem) {
        return Some(Role::Attention);
    }
    NEXTN_BLOCK
        .iter()
        .flat_map(|g| g.iter())
        .find(|s| s.name == stem)
        .map(|s| s.role)
}

/// The names of `stems` the file must carry.
pub(super) fn required(stems: &[Stem]) -> impl Iterator<Item = &'static str> + '_ {
    stems.iter().filter(|s| s.required).map(|s| s.name)
}

/// The role of a trunk layer's stem.
fn layer_role(stem: &str) -> Option<Role> {
    TRUNK
        .iter()
        .flat_map(|g| g.iter())
        .find(|s| s.name == stem)
        .map(|s| s.role)
}

/// A next-token layer's own stems, beside the trunk's.
fn nextn_stem(stem: &str) -> bool {
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
    use super::{KDA, LATENT, NEXTN, Stem, TRUNK, layer_role, nextn_role};
    use crate::placement::Role;

    /// A stem in two groups (the norm and output every mixer carries, the
    /// pre-FFN norm) names one role, and no group lists a stem twice.
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
        assert!(NEXTN.iter().all(|s| layer_role(s.name).is_none()));
        // The header check refuses the other mixer's stems by name: each is
        // one the file must carry where its kind is.
        assert!(KDA.iter().chain(LATENT).all(|s| s.required));
    }

    /// A next-token layer loads its mixer, routed block and shared expert in
    /// their trunk roles, its own stems on the card and the head's norm as
    /// the head's; a hyper-connection or KDA stem is none of its.
    #[test]
    fn nextn_roles() {
        assert_eq!(nextn_role("nextn.eh_proj.weight"), Some(Role::Attention));
        assert_eq!(nextn_role("nextn.enorm.weight"), Some(Role::Attention));
        assert_eq!(nextn_role("nextn.hnorm.weight"), Some(Role::Attention));
        assert_eq!(
            nextn_role("nextn.shared_head_norm.weight"),
            Some(Role::Head)
        );
        assert_eq!(nextn_role("attn_kv_a_mqa.weight"), Some(Role::Attention));
        assert_eq!(nextn_role("ffn_gate_inp.weight"), Some(Role::Router));
        assert_eq!(nextn_role("ffn_up_exps.weight"), Some(Role::RoutedExperts));
        assert_eq!(
            nextn_role("ffn_down_shexp.weight"),
            Some(Role::SharedExpert)
        );
        assert_eq!(nextn_role("hc_attn_fn.weight"), None);
        assert_eq!(nextn_role("ssm_a"), None);
    }
}
