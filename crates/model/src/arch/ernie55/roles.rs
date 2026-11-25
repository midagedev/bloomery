//! The role of every tensor in an ERNIE 5.5 file, by exact name: the model-level names (the head's
//! stream fold and MultiMax among them) and the per-layer stems by the groups a layer's kind
//! carries ([`stems_of`]: the HCA mixer or the latent mixer with its lightning indexer, the
//! hyper-connection mixes, and a dense or routed feed-forward block), which the header reader's
//! tensor check reads too. A name the layer's kind does not hold (the key and value halves of the
//! fused projection, a block past the model, the other kind's stem, an indexer tensor on an HCA
//! layer, a compressor on a latent layer) fails the file with every such name listed.
//!
//! The router pair, the learned expert scale and the latent experts' pair and norm are matrices of
//! the routed block every token reads on the layer's card, serial with the experts, so they sit
//! with the router. The head mix and the output gate of a mixer are read every token beside the
//! attention rows, so they sit with them.

use gguf::Split;

use super::hparams::Hparams;
use super::names::{
    OUTPUT_HC_BASE, OUTPUT_HC_FN, OUTPUT_HC_SCALE, OUTPUT_MULTIMAX_RANGES, OUTPUT_MULTIMAX_TS,
};
use crate::arch::classify::{Counts, classify_with, is_side_matrix, role_by_layer};
use crate::arch::noaux;
use crate::arch::stems::{DENSE, HC, Stem, join, req};
use crate::placement::{ModelTensor, ModelTensors, PlacementError, Role};

/// An HCA layer's mixer: the query latent and its norm, the one key-value latent and its norm,
/// the sink, the grouped output projections, the pooling compressor, the output gate and the head
/// mix.
pub(super) const HCA: &[Stem] = &[
    req("attn_norm.weight", Role::Attention),
    req("attn_q_a.weight", Role::Attention),
    req("attn_q_a_norm.weight", Role::Attention),
    req("attn_q_b.weight", Role::Attention),
    req("attn_kv.weight", Role::Attention),
    req("attn_kv_a_norm.weight", Role::Attention),
    req("attn_sinks.weight", Role::Attention),
    req("attn_output_a.weight", Role::Attention),
    req("attn_output_b.weight", Role::Attention),
    req("attn_compressor_kv.weight", Role::Attention),
    req("attn_compressor_gate.weight", Role::Attention),
    req("attn_compressor_norm.weight", Role::Attention),
    req("attn_compressor_ape.weight", Role::Attention),
    req("attn_gate.weight", Role::Attention),
    req("attn_vha_u.weight", Role::Attention),
    req("attn_vha_v.weight", Role::Attention),
];

/// A latent layer's mixer: the query and key-value latents with their norms, the fused per-head up
/// projection as the checkpoint stores it (the card's key and value halves are cut from it by a
/// later move, so no file holds `attn_k_b` or `attn_v_b`), the output gate and the output
/// projection.
const LATENT_MIXER: &[Stem] = &[
    req("attn_norm.weight", Role::Attention),
    req("attn_q_a.weight", Role::Attention),
    req("attn_q_a_norm.weight", Role::Attention),
    req("attn_q_b.weight", Role::Attention),
    req("attn_kv_a_mqa.weight", Role::Attention),
    req("attn_kv_a_norm.weight", Role::Attention),
    req("attn_kv_b.weight", Role::Attention),
    req("attn_gate.weight", Role::Attention),
    req("attn_output.weight", Role::Attention),
];

/// What a latent layer carries beside the mixer's stems: the sink and the head mix.
const LATENT_EXTRA: &[Stem] = &[
    req("attn_sinks.weight", Role::Attention),
    req("attn_vha_u.weight", Role::Attention),
    req("attn_vha_v.weight", Role::Attention),
];

/// A latent layer's mixer with the sink and the head mix.
pub(super) const LATENT: &[Stem] = &join::<12>(&[LATENT_MIXER, LATENT_EXTRA]);

/// A latent layer's lightning indexer: its query and key projections, the head weights and the
/// key's LayerNorm gain and bias. Read every token on the layer's card beside the attention rows,
/// so they sit with them.
pub(super) const INDEXER: &[Stem] = &[
    req("indexer.attn_q_b.weight", Role::Attention),
    req("indexer.attn_k.weight", Role::Attention),
    req("indexer.proj.weight", Role::Attention),
    req("indexer.k_norm.weight", Role::Attention),
    req("indexer.k_norm.bias", Role::Attention),
];

/// A routed layer's second router, its learned expert scale, and the latent experts' input and
/// output projections and the norm on their sum.
const ROUTED_EXTRA: &[Stem] = &[
    req("ffn_gate_inp_1.weight", Role::Router),
    req("ffn_exp_scale.weight", Role::Router),
    req("ffn_routed_down.weight", Role::Router),
    req("ffn_routed_up.weight", Role::Router),
    req("ffn_routed_norm.weight", Role::Router),
];

/// A routed layer's feed-forward block: the common noaux owner's, then the second router, the
/// scale and the latent experts' own.
pub(super) const ROUTED: &[Stem] = &join::<14>(&[noaux::BLOCK, ROUTED_EXTRA]);

/// What a layer is: an HCA or a latent mixer, a dense or routed feed-forward block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Kind {
    pub latent: bool,
    pub dense: bool,
}

/// Every kind of layer a file may hold.
pub(super) const KINDS: [Kind; 4] = [
    Kind {
        latent: false,
        dense: false,
    },
    Kind {
        latent: false,
        dense: true,
    },
    Kind {
        latent: true,
        dense: false,
    },
    Kind {
        latent: true,
        dense: true,
    },
];

/// The stem groups a layer of `kind` carries.
pub(super) fn stems_of(kind: Kind) -> Vec<&'static [Stem]> {
    let mut groups = vec![
        if kind.latent { LATENT } else { HCA },
        HC,
        if kind.dense { DENSE } else { ROUTED },
    ];
    if kind.latent {
        groups.push(INDEXER);
    }
    groups
}

/// The role and layer of `name` in a file of `hp`: the model-level names, then each layer's stems
/// by its kind.
fn role_of(name: &str, hp: &Hparams) -> Option<(Role, Option<usize>)> {
    if [
        OUTPUT_HC_FN,
        OUTPUT_HC_BASE,
        OUTPUT_HC_SCALE,
        OUTPUT_MULTIMAX_RANGES,
        OUTPUT_MULTIMAX_TS,
    ]
    .contains(&name)
    {
        return Some((Role::Head, None));
    }
    role_by_layer(name, hp.n_layer, |l| stems_of(hp.kind(l)))
}

/// Every tensor of `split` with its role and header facts, and the trunk's layer and expert counts
/// from `hp`.
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    let e = &hp.experts;
    let mut model = classify_with(
        split,
        |name| role_of(name, hp),
        Counts::new(hp.n_layer, e.n_expert, e.n_used),
    )?;
    model.bf16_words = bf16_words;
    Ok(model)
}

/// Whether the body reads BF16 tensor `t` in its words ([`ModelTensors::bf16_words`]): a
/// compressor or an indexer matrix, which the V4.1 chain's dense sites take; every other BF16
/// tensor loads widened to f32.
fn bf16_words(t: &ModelTensor) -> bool {
    t.role == Role::Attention && is_side_matrix(&t.name)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use gguf::arch::ernie55::STEM_TYPES;

    use super::{
        DENSE, HC, HCA, INDEXER, KINDS, LATENT, OUTPUT_HC_BASE, OUTPUT_HC_FN, OUTPUT_HC_SCALE,
        OUTPUT_MULTIMAX_RANGES, OUTPUT_MULTIMAX_TS, ROUTED, stems_of,
    };
    use crate::arch::classify::{assert_one_role_per_stem, role_by_layer};
    use crate::placement::Role;

    use super::Kind;

    /// A tensor's role and layer by its name in a 4-layer file: layer 0 a dense HCA layer, 1 a
    /// routed HCA layer, 2 a routed latent layer, 3 a routed HCA layer.
    fn role(name: &str) -> Option<(Role, Option<usize>)> {
        role_by_layer(name, 4, |l| {
            stems_of(Kind {
                latent: l == 2,
                dense: l == 0,
            })
        })
    }

    /// A stem in two groups names one role (the two mixers both hold `attn_norm`, the sink and the
    /// head mix, the two feed-forward blocks `ffn_norm`), and no group lists a stem twice. Mutant:
    /// `attn_norm` given the router's role in one group.
    #[test]
    fn a_stem_has_one_role() {
        let groups: Vec<_> = KINDS.into_iter().flat_map(stems_of).collect();
        assert_one_role_per_stem(&groups);
    }

    /// The description and the loader name the same stems: every stem a layer's groups carry (of
    /// every kind), with the model-level names, is a stem of the loader's table
    /// (`gguf::arch::ernie55::STEM_TYPES`, which is what the folder map writes), and every stem of
    /// that table is named by the description. Mutant: a stem renamed in one group (the fused
    /// `attn_kv_b` spelled as the split halves `attn_k_b`), or a stem dropped from a group.
    #[test]
    fn the_stems_are_the_loaders() {
        let model_level = [
            "token_embd.weight",
            "output.weight",
            "output_norm.weight",
            OUTPUT_HC_FN,
            OUTPUT_HC_BASE,
            OUTPUT_HC_SCALE,
            OUTPUT_MULTIMAX_RANGES,
            OUTPUT_MULTIMAX_TS,
        ];
        let described: BTreeSet<&str> = KINDS
            .into_iter()
            .flat_map(stems_of)
            .flatten()
            .map(|s| s.name)
            .chain(model_level)
            .collect();
        let loaded: BTreeSet<&str> = STEM_TYPES.iter().map(|&(stem, _)| stem).collect();
        let only_described: Vec<_> = described.difference(&loaded).collect();
        let only_loaded: Vec<_> = loaded.difference(&described).collect();
        assert!(
            only_described.is_empty() && only_loaded.is_empty(),
            "stems the description names and the loader's table lacks: {only_described:?}; \
             stems the loader's table holds and the description does not name: {only_loaded:?}"
        );
    }

    /// The table is exact: a stem of each group has its role on a layer of its kind, and a name the
    /// layer's kind does not hold has none, so `classify` lists it. Mutant: every layer given the
    /// stems of every kind.
    #[test]
    fn the_table_is_exact() {
        assert_eq!(
            role("token_embd.weight"),
            Some((Role::TokenEmbedding, None))
        );
        assert_eq!(role("output.weight"), Some((Role::Head, None)));
        for (name, want) in [
            ("blk.1.attn_compressor_ape.weight", Role::Attention),
            ("blk.1.attn_output_a.weight", Role::Attention),
            ("blk.1.attn_gate.weight", Role::Attention),
            ("blk.1.attn_vha_u.weight", Role::Attention),
            ("blk.2.attn_kv_b.weight", Role::Attention),
            ("blk.2.attn_vha_v.weight", Role::Attention),
            ("blk.2.indexer.k_norm.bias", Role::Attention),
            ("blk.1.hc_attn_fn.weight", Role::HyperConnection),
            ("blk.1.ffn_gate_inp.weight", Role::Router),
            ("blk.1.ffn_gate_inp_1.weight", Role::Router),
            ("blk.1.ffn_exp_scale.weight", Role::Router),
            ("blk.1.exp_probs_b.bias", Role::Router),
            ("blk.1.ffn_routed_down.weight", Role::Router),
            ("blk.1.ffn_routed_norm.weight", Role::Router),
            ("blk.1.ffn_gate_exps.weight", Role::RoutedExperts),
            ("blk.1.ffn_down_exps.weight", Role::RoutedExperts),
            ("blk.1.ffn_up_shexp.weight", Role::SharedExpert),
            ("blk.1.ffn_norm.weight", Role::FfnNorm),
            ("blk.0.ffn_gate.weight", Role::DenseFfn),
        ] {
            let layer = name.split('.').nth(1).and_then(|l| l.parse().ok());
            assert_eq!(role(name), Some((want, layer)), "{name}");
        }
        assert_eq!(
            (
                HCA.len(),
                LATENT.len(),
                INDEXER.len(),
                HC.len(),
                DENSE.len(),
                ROUTED.len()
            ),
            (16, 12, 5, 6, 4, 14)
        );
        for name in [
            "blk.2.attn_kv.weight",
            "blk.2.attn_compressor_kv.weight",
            "blk.2.attn_output_a.weight",
            "blk.2.attn_k_b.weight",
            "blk.2.attn_v_b.weight",
            "blk.1.attn_kv_b.weight",
            "blk.1.attn_kv_a_mqa.weight",
            "blk.1.indexer.proj.weight",
            "blk.3.indexer.attn_k.weight",
            "blk.0.ffn_gate_inp.weight",
            "blk.1.ffn_gate.weight",
            "blk.1.post_attention_norm.weight",
            "blk.4.attn_norm.weight",
            "blk.1.attn_q.weight",
            "blk.1.attn_qkv.weight",
            "blk.1.nextn.eh_proj.weight",
            "blk.1.attn_q.bias",
            "rope_freqs.weight",
        ] {
            assert_eq!(role(name), None, "{name}");
        }
    }

    /// The indexer is the latent layers' alone: its five stems carry the attention's role on a
    /// latent layer and none on an HCA layer. Mutant: the indexer given to every layer, or left out
    /// of the latent layer's groups.
    #[test]
    fn the_indexer_is_the_latent_layers_alone() {
        for stem in [
            "indexer.attn_q_b.weight",
            "indexer.attn_k.weight",
            "indexer.proj.weight",
            "indexer.k_norm.weight",
            "indexer.k_norm.bias",
        ] {
            assert_eq!(
                role(&format!("blk.2.{stem}")),
                Some((Role::Attention, Some(2))),
                "{stem}"
            );
            for l in [0, 1, 3] {
                assert_eq!(role(&format!("blk.{l}.{stem}")), None, "blk.{l}.{stem}");
            }
        }
    }
}
