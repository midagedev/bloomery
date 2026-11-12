//! Layer groups shared by the tensor-set check, shape check and classification.

use crate::arch::classify::{Counts, block_stem, classify_with, model_level_role};
pub(super) use crate::arch::stems::{ATTENTION, DENSE, MOE, SELECTION_BIAS, SHARED};
use crate::arch::stems::{Stem, opt, req};
use crate::placement::{ModelTensors, PlacementError, Role};

use super::hparams::Hparams;

pub(super) const SINKS: &[Stem] = &[req("attn_sinks.weight", Role::Attention)];

/// A next-token layer's own stems, beside the trunk's; `shared_head_norm` is
/// optional (the block shares the trunk's embedding and head, exaone-moe.cpp:94-98).
pub(super) const NEXTN: &[Stem] = &[
    req("nextn.eh_proj.weight", Role::Unused),
    req("nextn.enorm.weight", Role::Unused),
    req("nextn.hnorm.weight", Role::Unused),
    opt("nextn.shared_head_norm.weight", Role::Unused),
];

/// A block's kind: what its feed-forward side is, the next-token layers their own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Dense,
    Routed,
    /// A block past the trunk that carries tensors: its every tensor is
    /// [`Role::Unused`] (described, run by no program).
    Nextn,
}

const DENSE_GROUPS: &[&[Stem]] = &[ATTENTION, DENSE];
const ROUTED_GROUPS: &[&[Stem]] = &[ATTENTION, MOE, SHARED];
const NEXTN_GROUPS: &[&[Stem]] = &[ATTENTION, DENSE, NEXTN];

/// The stem groups a block of `kind` carries, `sinks` telling whether every
/// block carries `attn_sinks.weight`.
pub(super) fn groups(kind: Kind, sinks: bool) -> Vec<&'static [Stem]> {
    let mut out = match kind {
        Kind::Dense => DENSE_GROUPS.to_vec(),
        Kind::Routed => ROUTED_GROUPS.to_vec(),
        Kind::Nextn => NEXTN_GROUPS.to_vec(),
    };
    if sinks {
        out.insert(1, SINKS);
    }
    out
}

/// `blk.{l}.attn_sinks.weight`, the name the sinks check reads.
pub(super) fn sink_name(l: usize) -> String {
    format!("blk.{l}.attn_sinks.weight")
}

/// Every tensor of `split` with its role: the model-level three, the trunk
/// blocks' stems by their kind, and a carried next-token block's trunk and own
/// stems as [`Role::Unused`]. A block past the trunk the file stripped carries
/// no tensor, so any name on it is unclassified; a name no block's groups hold
/// fails the file with every such name listed.
pub(super) fn classify(split: &gguf::Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    let nextn = hp.mtp > 0;
    classify_with(
        split,
        |name| {
            model_level_role(name).or_else(|| {
                let (l, stem) = block_stem(name, hp.blocks)?;
                let kind = if l < hp.dense_lead {
                    Kind::Dense
                } else if l < hp.trunk {
                    Kind::Routed
                } else {
                    Kind::Nextn
                };
                let held = groups(kind, hp.sinks)
                    .iter()
                    .flat_map(|g| g.iter())
                    .find(|s| s.name == stem);
                match (kind, held) {
                    (Kind::Nextn, Some(_)) if nextn => Some((Role::Unused, Some(l))),
                    // A stripped next-token block carries no tensor; no name is
                    // its own, so any name there fails the file.
                    (Kind::Nextn, _) => None,
                    (_, held) => held.map(|s| (s.role, Some(l))),
                }
            })
        },
        Counts::new(hp.trunk, hp.experts, hp.used),
    )
}
