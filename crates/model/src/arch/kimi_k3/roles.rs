//! The role of every tensor in a Kimi K3 or K3.1 file, by exact name: the three model-level names
//! and the final mix's scores, then the per-layer stems by the groups a layer's kind carries
//! ([`stems_of`]: [`KDA`] or [`MLA`] with [`INDEXER`] on a K3.1 latent layer, the layer's [`MIX`]
//! scores, and [`DENSE`] or [`ROUTED`]), which the header reader's tensor check reads too. A name
//! the layer's kind does not hold — a fused `attn_qkv`, a block past the model, a next-token
//! tensor, the other kind's stem, the low-rank gate pair, an indexer tensor on a layer that runs
//! no indexer — fails the file with every such name listed, and an indexer tensor a layer that
//! runs one lacks fails it by name.
//!
//! The latent experts' pair and norm, like the router, are matrices of the routed block every
//! token reads on the layer's card, serial with the experts. The cross-layer residual's scores sit
//! with their sublayer (the mixer's with the attention rows, the feed-forward block's with its
//! norm), and the final mix's with the head.

use gguf::Split;

use super::hparams::{Hparams, Kind};
use super::names::OUTPUT_RES_SCORE;
use crate::arch::classify::{Counts, classify_with, role_by_layer};
use crate::arch::kda;
use crate::arch::noaux;
use crate::arch::stems::{DENSE, Stem, join, req};
use crate::placement::{ModelTensors, PlacementError, Role};

/// A KDA layer's output gate: the one full-width projection. The loader reads no low-rank pair
/// beside it, so [`kda::LOW_RANK_GATE`] is not among the stems.
const GATE: &[Stem] = &[req("ssm_g.weight", Role::Attention)];

/// A KDA layer's mixer: the common [`kda`] head and tail around the full-width gate.
pub(super) const KDA: &[Stem] = &join::<15>(&[kda::HEAD, GATE, kda::TAIL]);

/// A latent layer's mixer, the common `delta_mla` owner's.
use crate::arch::delta_mla::MLA;

/// A K3.1 latent layer's lightning indexer: its query and key projections, the head weights and
/// the key's LayerNorm gain and bias. Read every token on the layer's card beside the attention
/// rows, so they sit with them.
pub(super) const INDEXER: &[Stem] = &[
    req("indexer.attn_q_b.weight", Role::Attention),
    req("indexer.attn_k.weight", Role::Attention),
    req("indexer.proj.weight", Role::Attention),
    req("indexer.k_norm.weight", Role::Attention),
    req("indexer.k_norm.bias", Role::Attention),
];

/// The scores of the two mixes a layer reads: the mixer's and the feed-forward block's.
pub(super) const MIX: &[Stem] = &[
    req("attn_res_score.weight", Role::Attention),
    req("ffn_res_score.weight", Role::FfnNorm),
];

/// The latent experts' input and output projections and the norm on their sum.
const LATENT: &[Stem] = &[
    req("ffn_routed_down.weight", Role::Router),
    req("ffn_routed_up.weight", Role::Router),
    req("ffn_routed_norm.weight", Role::Router),
];

/// A routed layer's feed-forward block: the common noaux owner's, then the latent experts' own.
pub(super) const ROUTED: &[Stem] = &join::<12>(&[noaux::BLOCK, LATENT]);

/// What a layer is: its mixer, a dense (`true`) or routed feed-forward block, and whether it runs
/// the lightning indexer.
pub(super) type LayerKind = (Kind, bool, bool);

/// Every kind of layer a file may hold: a delta or latent mixer, a dense (`true`) or routed
/// feed-forward block, and an indexer on the latent layers of a K3.1 file.
pub(super) const KINDS: [LayerKind; 6] = [
    (Kind::Kda, true, false),
    (Kind::Kda, false, false),
    (Kind::Mla, true, false),
    (Kind::Mla, false, false),
    (Kind::Mla, true, true),
    (Kind::Mla, false, true),
];

/// The stem groups a layer of `kind` carries.
pub(super) fn stems_of((mixer, dense, indexer): LayerKind) -> Vec<&'static [Stem]> {
    let mixer = match mixer {
        Kind::Kda => KDA,
        Kind::Mla => MLA,
    };
    let mut groups = vec![mixer, MIX, if dense { DENSE } else { ROUTED }];
    if indexer {
        groups.push(INDEXER);
    }
    groups
}

/// Every tensor of `split` with its role and header facts.
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    let e = &hp.experts;
    classify_with(
        split,
        |name| {
            if name == OUTPUT_RES_SCORE {
                return Some((Role::Head, None));
            }
            role_by_layer(name, hp.n_layer, |l| {
                stems_of((hp.kind(l), hp.dense(l), hp.dsa(l)))
            })
        },
        Counts::new(hp.n_layer, e.n_expert, e.n_used),
    )
}

#[cfg(test)]
mod tests {
    use super::{DENSE, INDEXER, KDA, KINDS, MIX, MLA, ROUTED, stems_of};
    use crate::arch::classify::{assert_one_role_per_stem, role_by_layer};
    use crate::arch::kimi_k3::hparams::Kind;
    use crate::placement::Role;

    /// A tensor's role and layer by its name in a 6-layer file whose layer 0 is a dense KDA layer,
    /// 1, 3 and 4 routed KDA layers and 2 and 5 routed latent ones, the latent ones running the
    /// indexer when `indexer` (a K3.1 file).
    fn role_of(name: &str, indexer: bool) -> Option<(Role, Option<usize>)> {
        role_by_layer(name, 6, |l| {
            let kind = if l == 2 || l == 5 {
                Kind::Mla
            } else {
                Kind::Kda
            };
            stems_of((kind, l == 0, indexer && kind == Kind::Mla))
        })
    }

    /// [`role_of`] in a K3 file.
    fn role(name: &str) -> Option<(Role, Option<usize>)> {
        role_of(name, false)
    }

    /// A stem in two groups names one role (the two mixers both hold `attn_norm` and
    /// `attn_output`, the two feed-forward blocks `ffn_norm`), and no group lists a stem twice.
    /// Mutant: `attn_norm` given the router's role in one group.
    #[test]
    fn a_stem_has_one_role() {
        let groups: Vec<_> = KINDS.into_iter().flat_map(stems_of).collect();
        assert_one_role_per_stem(&groups);
    }

    /// The table is exact: a stem of each group has its role on a layer of its kind, and a name
    /// the layer's kind does not hold — the other kind's stem, the low-rank gate pair, a fused
    /// projection, a layer past the model, a next-token tensor, a bias, an indexer, a hyper-connection
    /// tensor — has none, so `classify` lists it. Mutant: every layer given the stems of every kind.
    #[test]
    fn the_table_is_exact() {
        assert_eq!(
            role("token_embd.weight"),
            Some((Role::TokenEmbedding, None))
        );
        assert_eq!(role("output.weight"), Some((Role::Head, None)));
        assert_eq!(
            role("blk.2.attn_gate.weight"),
            Some((Role::Attention, Some(2)))
        );
        assert_eq!(role("blk.1.ssm_g.weight"), Some((Role::Attention, Some(1))));
        assert_eq!(
            role("blk.4.attn_res_score.weight"),
            Some((Role::Attention, Some(4)))
        );
        assert_eq!(
            role("blk.4.ffn_res_score.weight"),
            Some((Role::FfnNorm, Some(4)))
        );
        for stem in ["ffn_routed_down", "ffn_routed_up", "ffn_routed_norm"] {
            assert_eq!(
                role(&format!("blk.3.{stem}.weight")),
                Some((Role::Router, Some(3))),
                "{stem}"
            );
        }
        for stem in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
            assert_eq!(
                role(&format!("blk.3.{stem}.weight")),
                Some((Role::RoutedExperts, Some(3))),
                "{stem}"
            );
        }
        assert_eq!(
            role("blk.3.ffn_gate_shexp.weight"),
            Some((Role::SharedExpert, Some(3)))
        );
        assert_eq!(
            role("blk.3.exp_probs_b.bias"),
            Some((Role::Router, Some(3)))
        );
        assert_eq!(
            role("blk.0.ffn_gate.weight"),
            Some((Role::DenseFfn, Some(0)))
        );
        assert_eq!(
            (KDA.len(), MLA.len(), MIX.len(), DENSE.len(), ROUTED.len()),
            (15, 10, 2, 4, 12)
        );
        for name in [
            "blk.2.ssm_g.weight",
            "blk.1.attn_gate.weight",
            "blk.1.ssm_g_a.weight",
            "blk.1.ssm_g_b.weight",
            "blk.1.ssm_rda.weight",
            "blk.0.ffn_routed_down.weight",
            "blk.1.ffn_gate.weight",
            "blk.6.attn_q.weight",
            "blk.0.attn_qkv.weight",
            "blk.3.hc_attn_fn.weight",
            "blk.3.indexer.proj.weight",
            "blk.2.indexer.proj.weight",
            "blk.2.indexer.k_norm.bias",
            "blk.3.nextn.eh_proj.weight",
            "blk.3.attn_q.bias",
            "output_res_score.weight",
            "rope_freqs.weight",
        ] {
            assert_eq!(role(name), None, "{name}");
        }
    }

    /// A K3.1 file's latent layers carry the five indexer stems with the attention's role, and no
    /// other layer does: a delta layer's indexer tensor, the indexer's other spellings (a query
    /// norm, a compressor, a head-weight spelled `weights_proj`) and the K3 file's latent layer
    /// have no role, so the classifier lists them. Mutant: the indexer given to every layer, or
    /// left out of the latent layer's groups.
    #[test]
    fn the_indexer_is_the_k31_latent_layers_alone() {
        assert_eq!(INDEXER.len(), 5);
        for stem in [
            "indexer.attn_q_b.weight",
            "indexer.attn_k.weight",
            "indexer.proj.weight",
            "indexer.k_norm.weight",
            "indexer.k_norm.bias",
        ] {
            for l in [2, 5] {
                assert_eq!(
                    role_of(&format!("blk.{l}.{stem}"), true),
                    Some((Role::Attention, Some(l))),
                    "blk.{l}.{stem}"
                );
            }
            for l in [0, 1, 3, 4] {
                assert_eq!(
                    role_of(&format!("blk.{l}.{stem}"), true),
                    None,
                    "blk.{l}.{stem}"
                );
            }
        }
        for name in [
            "blk.2.indexer.weights_proj.weight",
            "blk.2.indexer.q_norm.weight",
            "blk.2.indexer.k_norm.bias.weight",
            "blk.2.indexer_compressor_kv.weight",
        ] {
            assert_eq!(role_of(name, true), None, "{name}");
        }
        let groups: Vec<_> = KINDS.into_iter().flat_map(stems_of).collect();
        assert_one_role_per_stem(&groups);
    }
}
