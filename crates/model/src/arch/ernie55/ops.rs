//! What an ERNIE 5.5 file asks that [`models::ModelSpec`] has no field for, in this family's own
//! types: each is a part of the layers or the head that the common description cannot state
//! without a field every family would fill, so it is listed here and refused by name
//! ([`Ops::unrun`]) until a program of this family runs it. A common owner that takes one up
//! would add the field named beside it:
//! - the head mix ([`Ops::vha_rank`]): `models::Latent` would gain a head-mix rank, set to 0 by its
//!   literals in deepseek41, glm5next, hy4 and mistrallarge4 (`spec.rs`) and in `delta_mla`'s
//!   `Mla::latent` (kimi_k3, raon1);
//! - the router sum ([`Ops::routers`]): `models::Router` would gain a router count and a
//!   learned-scale flag, set by each of its literals (qwen3moe, `noaux`'s `Experts::moe` and so
//!   glm5next, kimi_k3, raon1, solar2; deepseek41, nemotron_h_moe, flint, hy4, inkling2,
//!   musespark, qwen35moe, mistrallarge4, mimo2);
//! - the head's MultiMax ([`Ops::multimax`]): `models::LogitEpilogue` would gain a variant or a
//!   field, named by the two readers that set an epilogue (inkling2, musespark);
//! - the indexer's own rope ([`Ops::index_rope`]): `models::IndexKeys::FromInput` would gain a
//!   rope, named by its two readers (kimi_k3, mistrallarge4) and read by `place_hybrid` and the
//!   card's `index_key` entry.

use models::Unimplemented;

use super::hparams::{Hparams, MULTIMAX};
use crate::arch::spec_u32;
use crate::placement::PlacementError;

/// The ops of one file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ops {
    /// `attention.vha.rank`: every layer mixes its heads' outputs by `U` and `V` of this rank
    /// (`attn_vha_u`, `attn_vha_v`).
    pub vha_rank: u32,
    /// Router matrices a routed layer scores with: `sigmoid(gate · x) + sigmoid(gate_1 · x)`
    /// picks and weighs an expert, and the kept weights multiply a learned per-expert scale
    /// (`ffn_exp_scale`).
    pub routers: u32,
    /// The values of the head's MultiMax: ranges and scales (`output_multimax_ranges`,
    /// `output_multimax_ts`) applied to the logits.
    pub multimax: u32,
    /// The rotated values of the DSA indexer's query and key: NeoX halves over the leading
    /// values, where the layer's own rope is pairwise.
    pub index_rope: u32,
}

/// Router matrices an ERNIE routed layer carries.
pub const ROUTERS: u32 = 2;

impl Ops {
    /// The ops of `hp`'s file.
    pub fn of(hp: &Hparams) -> Result<Ops, PlacementError> {
        Ok(Ops {
            vha_rank: spec_u32("attention.vha.rank", hp.vha_rank)?,
            routers: ROUTERS,
            multimax: spec_u32("multimax", MULTIMAX)?,
            index_rope: spec_u32("rope.dimension_count", hp.mla.pe)?,
        })
    }

    /// Every op a description of this family holds that no program runs, one item each: the whole
    /// list, since no program of this family exists.
    #[must_use]
    pub fn unrun(&self) -> Vec<Unimplemented> {
        let item = |feature: String| Unimplemented {
            layer: None,
            feature,
        };
        vec![
            item(format!(
                "the head mix of rank {}: every layer's heads mixed by attn_vha_u and attn_vha_v",
                self.vha_rank
            )),
            item(format!(
                "{} summed sigmoid routers (ffn_gate_inp, ffn_gate_inp_1) with a learned per-expert \
                 scale (ffn_exp_scale)",
                self.routers
            )),
            item(format!(
                "the head's MultiMax: {} ranges and scales on the logits",
                self.multimax
            )),
            item(format!(
                "the indexer's NeoX rope over its leading {} values",
                self.index_rope
            )),
        ]
    }
}
