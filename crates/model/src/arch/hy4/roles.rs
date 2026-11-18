//! The role of every tensor in a Hy4-flash file, by exact name: the model-level names (the head's
//! stream fold among them) and the per-layer stems by the groups a layer's kind carries
//! ([`stems_of`]: the latent mixer with its sink and sandwich norm, the hyper-connection mixes,
//! the indexer of a sparse-attention layer, the engram site's delta, and a dense or routed
//! feed-forward block with its sandwich norm), which the header reader's tensor check reads too.
//! The next-token block (`blk.{n_trunk}`) is carried and never loaded: every one of its tensors is
//! [`Role::Unused`]. The per-layer table is the one [`Role::EngramTable`], a few rows gathered a
//! token on the first site's layer. A name the layer's kind does not hold (a fused projection, a
//! block past the model, the other kind's stem, an indexer tensor on a window layer, a mix on the
//! next-token block) fails the file with every such name listed.
//!
//! The latent experts' pair and the router are matrices of the routed block every token reads on
//! the layer's card, serial with the experts, so they sit with the router.

use gguf::Split;

use super::hparams::Hparams;
use super::names::{ENGRAM_TABLE, OUTPUT_HC_BASE, OUTPUT_HC_FN, OUTPUT_HC_SCALE};
use crate::arch::classify::{Counts, block_stem, classify_with, model_level_role};
use crate::arch::delta_mla::MLA;
use crate::arch::noaux;
use crate::arch::stems::{DENSE, HC, Stem, join, req};
use crate::placement::{ModelTensors, PlacementError, Role};

/// The mixer's sink and the norm on its output, beside the latent layer's own stems.
const MIXER_EXTRA: &[Stem] = &[
    req("attn_sinks.weight", Role::Attention),
    req("post_attention_norm.weight", Role::Attention),
];

/// Every layer's mixer: the common latent layer's stems, the per-head sink and the sandwich norm.
pub(super) const MIXER: &[Stem] = &join::<12>(&[MLA, MIXER_EXTRA]);

/// A sparse-attention layer's indexer: the query and key projections, the head weights, the key's
/// gain, and the pooling compressor's value and score projections, position table and gain. Read
/// every token on the layer's card beside the attention rows, so they sit with them.
pub(super) const INDEXER: &[Stem] = &[
    req("indexer.attn_q_b.weight", Role::Attention),
    req("indexer.attn_k.weight", Role::Attention),
    req("indexer.proj.weight", Role::Attention),
    req("indexer.k_norm.weight", Role::Attention),
    req("indexer_compressor_value.weight", Role::Attention),
    req("indexer_compressor_gate.weight", Role::Attention),
    req("indexer_compressor_ape.weight", Role::Attention),
    req("indexer_compressor_norm.weight", Role::Attention),
];

/// An engram site's delta: the fused key and value projection every token reads in full, its bias,
/// the gate, the stream scales, the gathered rows' norm and the value's gain (gains and gates the
/// card multiplies by, f32).
pub(super) const ENGRAM: &[Stem] = &[
    req("engram_kv.weight", Role::EngramDense),
    req("engram_kv_b.weight", Role::EngramGain),
    req("engram_gate.weight", Role::EngramGain),
    req("engram_hc_scale.weight", Role::EngramGain),
    req("engram_lookup_norm.weight", Role::EngramGain),
    req("engram_pre_gamma.weight", Role::EngramGain),
];

/// The norm on the feed-forward block's output.
const POST_FFW: &[Stem] = &[req("post_ffw_norm.weight", Role::FfnNorm)];

/// A dense layer's feed-forward block with its output norm.
pub(super) const DENSE_FFN: &[Stem] = &join::<5>(&[DENSE, POST_FFW]);

/// The latent experts' input and output projections.
const LATENT: &[Stem] = &[
    req("ffn_routed_down.weight", Role::Router),
    req("ffn_routed_up.weight", Role::Router),
];

/// A routed layer's feed-forward block: the common noaux owner's, the latent pair and the output
/// norm.
pub(super) const ROUTED: &[Stem] = &join::<12>(&[noaux::BLOCK, LATENT, POST_FFW]);

/// The next-token block's own tensors, carried and never loaded.
pub(super) const NEXTN: &[Stem] = &[
    req("nextn.eh_proj.weight", Role::Unused),
    req("nextn.enorm.weight", Role::Unused),
    req("nextn.hnorm.weight", Role::Unused),
    req("nextn.shared_head_norm.weight", Role::Unused),
];

/// What a block is: sparse attention (`dsa`) or a window, a dense or routed feed-forward block,
/// an engram site before its mixer, or the next-token block (no hyper-connection mixes, no site).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Kind {
    pub dsa: bool,
    pub dense: bool,
    pub engram: bool,
    pub nextn: bool,
}

/// Every kind of block a file may hold: the trunk's eight (sparse or window, dense or routed,
/// with or without a site) and the next-token block's two (sparse or window, routed).
pub(super) const KINDS: [Kind; 10] = [
    Kind {
        dsa: false,
        dense: false,
        engram: false,
        nextn: false,
    },
    Kind {
        dsa: false,
        dense: false,
        engram: true,
        nextn: false,
    },
    Kind {
        dsa: false,
        dense: true,
        engram: false,
        nextn: false,
    },
    Kind {
        dsa: false,
        dense: true,
        engram: true,
        nextn: false,
    },
    Kind {
        dsa: true,
        dense: false,
        engram: false,
        nextn: false,
    },
    Kind {
        dsa: true,
        dense: false,
        engram: true,
        nextn: false,
    },
    Kind {
        dsa: true,
        dense: true,
        engram: false,
        nextn: false,
    },
    Kind {
        dsa: true,
        dense: true,
        engram: true,
        nextn: false,
    },
    Kind {
        dsa: false,
        dense: false,
        engram: false,
        nextn: true,
    },
    Kind {
        dsa: true,
        dense: false,
        engram: false,
        nextn: true,
    },
];

/// The stem groups a block of `kind` carries.
pub(super) fn stems_of(kind: Kind) -> Vec<&'static [Stem]> {
    let mut groups = vec![MIXER, if kind.dense { DENSE_FFN } else { ROUTED }];
    if kind.nextn {
        groups.push(NEXTN);
    } else {
        groups.push(HC);
    }
    if kind.dsa {
        groups.push(INDEXER);
    }
    if kind.engram {
        groups.push(ENGRAM);
    }
    groups
}

/// The role and layer of `name` in a file of `hp`: the model-level names, the per-layer table, and
/// each block's stems by its kind. A next-token block's every stem is [`Role::Unused`].
fn role_of(name: &str, hp: &Hparams) -> Option<(Role, Option<usize>)> {
    if let Some(r) = model_level_role(name) {
        return Some(r);
    }
    if [OUTPUT_HC_FN, OUTPUT_HC_BASE, OUTPUT_HC_SCALE].contains(&name) {
        return Some((Role::Head, None));
    }
    if name == ENGRAM_TABLE {
        return Some((Role::EngramTable, hp.engram.as_ref().map(|e| e.layers[0])));
    }
    let (layer, stem) = block_stem(name, hp.n_block)?;
    let held = stems_of(hp.kind(layer))
        .into_iter()
        .flat_map(|g| g.iter())
        .find(|s| s.name == stem)?;
    let role = if hp.is_nextn(layer) {
        Role::Unused
    } else {
        held.role
    };
    Some((role, Some(layer)))
}

/// Every tensor of `split` with its role and header facts, and the trunk's layer and expert counts
/// from `hp`. The per-layer table gathers one row of each site's column a token.
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    let e = &hp.experts;
    let mut model = classify_with(
        split,
        |name| role_of(name, hp),
        Counts::new(hp.n_layer, e.n_expert, e.n_used),
    )?;
    for t in &mut model.tensors {
        if t.role == Role::EngramTable {
            t.gathered_rows = hp.engram.as_ref().map(|g| g.rows_a_token() as u64);
        }
    }
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::{
        DENSE_FFN, ENGRAM, INDEXER, KINDS, Kind, MIXER, NEXTN, ROUTED, block_stem, stems_of,
    };
    use crate::arch::classify::{assert_one_role_per_stem, model_level_role};
    use crate::arch::stems::HC;
    use crate::placement::Role;

    /// The role of `name` in a 6-block file whose block 0 is a dense sparse-attention layer, 1 a
    /// routed sparse layer with an engram site, 2 and 3 routed window layers, 4 a routed window
    /// layer and 5 the next-token block (a sparse routed layer).
    fn role(name: &str) -> Option<(Role, Option<usize>)> {
        if let Some(r) = model_level_role(name) {
            return Some(r);
        }
        let (layer, stem) = block_stem(name, 6)?;
        let k = match layer {
            0 => Kind {
                dsa: true,
                dense: true,
                engram: false,
                nextn: false,
            },
            1 => Kind {
                dsa: true,
                dense: false,
                engram: true,
                nextn: false,
            },
            5 => Kind {
                dsa: true,
                dense: false,
                engram: false,
                nextn: true,
            },
            _ => Kind {
                dsa: false,
                dense: false,
                engram: false,
                nextn: false,
            },
        };
        let s = stems_of(k)
            .into_iter()
            .flat_map(|g| g.iter())
            .find(|s| s.name == stem)?;
        Some((if k.nextn { Role::Unused } else { s.role }, Some(layer)))
    }

    /// A stem in two groups names one role (`attn_norm` of every mixer, `ffn_norm` of both
    /// feed-forward blocks), and no group lists a stem twice. Mutant: `ffn_norm` given the router's
    /// role in one block.
    #[test]
    fn a_stem_has_one_role() {
        let groups: Vec<_> = KINDS.into_iter().flat_map(stems_of).collect();
        assert_one_role_per_stem(&groups);
    }

    /// The table is exact: a stem of each group has its role on a block of its kind, and a name the
    /// block's kind does not hold has none, so `classify` lists it. Mutant: every block given the
    /// stems of every kind.
    #[test]
    fn the_table_is_exact() {
        assert_eq!(
            (
                MIXER.len(),
                INDEXER.len(),
                ENGRAM.len(),
                DENSE_FFN.len(),
                ROUTED.len(),
                NEXTN.len(),
                HC.len()
            ),
            (12, 8, 6, 5, 12, 4, 6)
        );
        assert_eq!(
            role("blk.2.attn_sinks.weight"),
            Some((Role::Attention, Some(2)))
        );
        assert_eq!(
            role("blk.2.post_attention_norm.weight"),
            Some((Role::Attention, Some(2)))
        );
        assert_eq!(
            role("blk.2.hc_attn_fn.weight"),
            Some((Role::HyperConnection, Some(2)))
        );
        assert_eq!(
            role("blk.1.indexer_compressor_ape.weight"),
            Some((Role::Attention, Some(1)))
        );
        assert_eq!(
            role("blk.1.engram_kv.weight"),
            Some((Role::EngramDense, Some(1)))
        );
        assert_eq!(
            role("blk.1.engram_gate.weight"),
            Some((Role::EngramGain, Some(1)))
        );
        for stem in ["ffn_routed_down", "ffn_routed_up"] {
            assert_eq!(
                role(&format!("blk.3.{stem}.weight")),
                Some((Role::Router, Some(3))),
                "{stem}"
            );
        }
        assert_eq!(
            role("blk.3.post_ffw_norm.weight"),
            Some((Role::FfnNorm, Some(3)))
        );
        assert_eq!(
            role("blk.0.ffn_down.weight"),
            Some((Role::DenseFfn, Some(0)))
        );
        assert_eq!(
            role("blk.3.ffn_down_exps.weight"),
            Some((Role::RoutedExperts, Some(3)))
        );
        // The next-token block: every tensor Unused, its own and the trunk's.
        for stem in [
            "nextn.eh_proj.weight",
            "nextn.enorm.weight",
            "attn_q_a.weight",
            "indexer.proj.weight",
            "ffn_gate_exps.weight",
        ] {
            assert_eq!(
                role(&format!("blk.5.{stem}")),
                Some((Role::Unused, Some(5))),
                "{stem}"
            );
        }
        for name in [
            "blk.2.indexer.proj.weight",
            "blk.2.engram_kv.weight",
            "blk.3.engram_kv.weight",
            "blk.0.ffn_gate_inp.weight",
            "blk.1.ffn_gate.weight",
            "blk.5.hc_attn_fn.weight",
            "blk.2.nextn.eh_proj.weight",
            "blk.2.attn_q.weight",
            "blk.2.attn_qkv.weight",
            "blk.2.ffn_routed_norm.weight",
            "blk.6.attn_norm.weight",
            "rope_freqs.weight",
        ] {
            assert_eq!(role(name), None, "{name}");
        }
    }

    /// The kinds list covers every combination a file can hold and none twice, so the misplaced-stem
    /// check of `layers_agree` sees every stem group. Mutant: a kind dropped or repeated.
    #[test]
    fn the_kinds_are_the_combinations() {
        let mut seen: Vec<Kind> = Vec::new();
        for k in KINDS {
            assert!(!seen.contains(&k), "{k:?} is listed twice");
            assert!(!(k.nextn && (k.engram || k.dense)), "{k:?}");
            seen.push(k);
        }
        assert_eq!(seen.len(), 8 + 2);
    }
}
