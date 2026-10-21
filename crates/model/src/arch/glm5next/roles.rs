//! The role of every tensor in a glm5next file, by exact name: the three
//! model-level names, then the per-layer stems — one table, by the group a
//! layer's kind carries (the KDA mixer's stems are the common
//! [`kda`](crate::arch::kda) module's, the rest this family's), which the
//! header reader's tensor check reads too. Every tensor of a next-token
//! layer is [`Role::Unused`]: carried, never loaded. A name the table does
//! not hold fails the file with every such name listed.

use gguf::Split;

use super::hparams::Hparams;
use crate::arch::classify::{Counts, classify_with, role_in, role_with_nextn};
use crate::arch::kda;
use crate::arch::stems::{opt, req};
use crate::placement::{ModelTensors, PlacementError, Role};

pub use crate::arch::stems::{DENSE, MOE, SELECTION_BIAS, Stem, required};
pub(super) use crate::arch::stems::{DENSE_DOWN, DENSE_GATE};

/// The KDA mixer's stems: the common [`kda`] module's table, which every
/// family with a KDA layer reads.
pub use kda::STEMS as KDA;

/// A latent attention layer's mixer with its indexer, which is required
/// because without it ik runs dense attention (`src/llama-load-tensors.cpp`,
/// `create_glm5next_tensors`).
pub const LATENT: &[Stem] = &[
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

/// A routed layer's shared expert, when the file has one.
pub const SHARED: &[Stem] = &[
    req("ffn_gate_shexp.weight", Role::SharedExpert),
    req("ffn_up_shexp.weight", Role::SharedExpert),
    req("ffn_down_shexp.weight", Role::SharedExpert),
];

/// The hyper-connection tensors: every trunk layer's, no next-token layer's.
pub use crate::arch::stems::HC;

/// A next-token layer's own stems, beside the trunk's; `shared_head_norm` is
/// optional (the block shares the trunk's embedding and head).
pub const NEXTN: &[Stem] = &[
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
    if NEXTN.iter().any(|s| s.name == stem) {
        return Some(Role::Attention);
    }
    NEXTN_BLOCK
        .iter()
        .flat_map(|g| g.iter())
        .find(|s| s.name == stem)
        .map(|s| s.role)
}

/// The role of a model-level or trunk tensor by its GGUF name, without a layer-count bound.
/// A name outside the role table has no role.
#[must_use]
pub fn tensor_role(name: &str) -> Option<Role> {
    role_in(&TRUNK, name, usize::MAX).map(|(role, _)| role)
}

/// Every tensor of `split` with its role and header facts.
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    classify_with(
        split,
        |name| role_with_nextn(name, hp.n_layer, hp.n_trunk, &TRUNK, NEXTN),
        Counts::new(hp.n_trunk, hp.n_expert, hp.n_used),
    )
}

#[cfg(test)]
mod tests {
    use super::{KDA, LATENT, NEXTN, Stem, TRUNK, nextn_role, tensor_role};
    use crate::placement::Role;

    /// A stem in two groups (the norm and output every mixer carries, the
    /// pre-FFN norm) names one role, and no group lists a stem twice.
    #[test]
    fn a_stem_has_one_role() {
        let all: Vec<&Stem> = TRUNK.iter().flat_map(|g| g.iter()).collect();
        for s in &all {
            assert_eq!(
                tensor_role(&format!("blk.0.{}", s.name)),
                Some(s.role),
                "{}",
                s.name
            );
        }
        for g in TRUNK.iter().chain([&NEXTN]) {
            for (i, s) in g.iter().enumerate() {
                assert!(!g[..i].iter().any(|t| t.name == s.name), "{} twice", s.name);
            }
        }
        assert!(
            NEXTN
                .iter()
                .all(|s| tensor_role(&format!("blk.0.{}", s.name)).is_none())
        );
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
