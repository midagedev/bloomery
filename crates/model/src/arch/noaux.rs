//! The routed feed-forward layer of the noaux_tc families (DeepSeek-V3's router): sigmoid scores,
//! a selection bias that steers the choice only, the kept scores renormalized and scaled, and an
//! ungated shared expert of `expert_feed_forward_length · expert_shared_count` values. Which
//! layers are routed and the experts' activation stay with the family.

use gguf::Split;
use models::{Act, Ffn, Moe, Router, Score, Shared};

use crate::arch::stems::{Stem, req};
use crate::arch::{
    expert_counts, finite_positive, meta_bool, mul, one_expert_group, positive, sigmoid_gating,
    spec_u32,
};
use crate::placement::{PlacementError, Role};

/// A routed layer's feed-forward block: the router with its selection bias, the routed stacks and
/// the shared expert.
pub(in crate::arch) const BLOCK: &[Stem] = &[
    req("ffn_norm.weight", Role::FfnNorm),
    req("ffn_gate_inp.weight", Role::Router),
    req("exp_probs_b.bias", Role::Router),
    req("ffn_gate_exps.weight", Role::RoutedExperts),
    req("ffn_up_exps.weight", Role::RoutedExperts),
    req("ffn_down_exps.weight", Role::RoutedExperts),
    req("ffn_gate_shexp.weight", Role::SharedExpert),
    req("ffn_up_shexp.weight", Role::SharedExpert),
    req("ffn_down_shexp.weight", Role::SharedExpert),
];

/// Every key [`Experts::read`] reads, by suffix.
pub(in crate::arch) const KEYS: &[&str] = &[
    "expert_count",
    "expert_used_count",
    "expert_feed_forward_length",
    "expert_shared_count",
    "expert_gating_func",
    "expert_weights_scale",
    "expert_weights_norm",
    "expert_group_count",
    "expert_group_used_count",
];

/// A routed layer's sigmoid router and experts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Experts {
    /// `expert_count`.
    pub n_expert: usize,
    /// `expert_used_count`.
    pub n_used: usize,
    /// `expert_feed_forward_length`: one routed expert's width, and one shared expert's.
    pub ff: usize,
    /// `expert_shared_count`: shared experts, run as one block of `ff · n_shared` values.
    pub n_shared: usize,
    /// `expert_weights_scale`: the factor on the kept weights.
    pub weights_scale: f32,
    /// `expert_weights_norm`: the kept weights are renormalized to sum to one.
    pub weights_norm: bool,
}

impl Experts {
    /// The shared block's width.
    #[must_use]
    pub fn shared_ff(&self) -> usize {
        self.ff * self.n_shared
    }

    /// The router and experts of `split`'s keys; a count of 0, a router other than the sigmoid one
    /// and more than one expert group are errors naming the key, `router` the family's router in
    /// the group refusal.
    pub(in crate::arch) fn read(
        split: &Split,
        router: &str,
        defaults: &mut Vec<String>,
    ) -> Result<Experts, PlacementError> {
        let (n_expert, n_used) = expert_counts(split)?;
        let ff = positive(split, "expert_feed_forward_length")?;
        let n_shared = positive(split, "expert_shared_count")?;
        mul(split, "expert_shared_count", ff, n_shared)?;
        sigmoid_gating(split, None, defaults)?;
        one_expert_group(split, router)?;
        Ok(Experts {
            n_expert,
            n_used,
            ff,
            n_shared,
            weights_scale: finite_positive(split, "expert_weights_scale")?,
            weights_norm: meta_bool(split, "expert_weights_norm")?,
        })
    }

    /// The block's tensors by stem and the dims llama.cpp creates them with on an `e`-value row,
    /// ggml order, `ffn_norm` aside.
    #[must_use]
    pub(in crate::arch) fn dims(&self, e: u64) -> [(&'static str, Vec<u64>); 8] {
        let (n, ff, shared) = (
            self.n_expert as u64,
            self.ff as u64,
            self.shared_ff() as u64,
        );
        [
            ("ffn_gate_inp.weight", vec![e, n]),
            ("exp_probs_b.bias", vec![n]),
            ("ffn_gate_exps.weight", vec![e, ff, n]),
            ("ffn_up_exps.weight", vec![e, ff, n]),
            ("ffn_down_exps.weight", vec![ff, e, n]),
            ("ffn_gate_shexp.weight", vec![e, shared]),
            ("ffn_up_shexp.weight", vec![e, shared]),
            ("ffn_down_shexp.weight", vec![shared, e]),
        ]
    }

    /// The layer's feed-forward description: routed experts and the shared one both of `act`.
    pub(in crate::arch) fn ffn(&self, act: Act) -> Result<Ffn, PlacementError> {
        Ok(Ffn::Moe(Moe {
            experts: spec_u32("expert_count", self.n_expert)?,
            top_k: spec_u32("expert_used_count", self.n_used)?,
            expert_ff: spec_u32("expert_feed_forward_length", self.ff)?,
            latent: None,
            act,
            router: Router {
                score: Score::Sigmoid,
                bias: true,
                norm: self.weights_norm,
                scale: self.weights_scale,
                hash: false,
            },
            shared: Some(Shared {
                ff: spec_u32("expert_shared_count", self.shared_ff())?,
                act,
                sigmoid_gate: false,
            }),
        }))
    }
}
