//! Qwen3.8's side of adaptive expert residency ([`crate::host::swap`]): the
//! model's stack table for the common file source
//! ([`crate::host::swap_source::FileSwap`]) — [`stacks`], its gate, up and
//! down — with the machine's live delay and deadline
//! ([`crate::host::swap_source::LIVE_DELAY`]).
//!
//! **Parts.** An expert is three parts in stack order: the gate and up
//! (`geo::FF` rows of `geo::HIDDEN` values each) and the down (`geo::HIDDEN`
//! rows of `geo::FF`), each of its layer's own type as the file holds it —
//! most layers a Q4_K gate and up and a Q5_1 down, some a Q5_K gate and up or
//! a Q8_0 down, and the UD-Q3_K_XL file's an IQ3_XXS or IQ4_XS gate and up
//! with an IQ4_NL down
//! ([`stacks`] reads each layer's from the plan). On
//! the card each part of slot `s` is the file's bytes of that expert at byte
//! `s · part` of its layer's stack, which the placed load uploads as file
//! bytes in slot order ([`place::routed_rule`] keeps all three stacks in
//! `CardFormat::KQuant`, and `super::card38` reads them exactly so). Nothing
//! converts: a staged part is its slot's bytes as they are, so the r8
//! sidecar, which exists for Q3_K stacks alone (`qdot::repack_q3k_r8`), is
//! undefined input here, refused by name.

use super::swap::{self, CardLeg};
use crate::GpuError;
use crate::host::swap_source::PlanStacks;
use model::placement::accepts::Program;
use model::placement::{ModelTensors, Plan};

pub use crate::host::swap_source::{DEADLINE, LIVE_DELAY};

/// Qwen3.8's card leg ([`CardLeg`]): the Qwen3.8 body's arms
/// (`placement::accepts::leg_arms`: a Q4_K, Q5_K, IQ3_XXS or IQ4_XS gate and
/// up, a Q5_1, Q8_0 or IQ4_NL down), which its place rule reads too.
pub(super) struct Leg38;

impl CardLeg for Leg38 {
    const WHAT: &'static str = "swap38::stacks";
    const PROGRAM: Program = Program::Qwen38Body;
}

/// Qwen3.8's stack table: the common one ([`PlanStacks`]) over this family's
/// facts ([`swap::stacks`]).
pub type Qwen38Stacks = PlanStacks;

/// The stacks of `model`'s layers, each layer's types read from its tensors
/// and every layer `plan` gives card experts (on its stage card or a tier
/// card) checked against [`Leg38`] ([`swap::stacks`]). Refused by name: a
/// layer without all three stacks, and a carded layer the leg refuses.
/// Load-time only.
pub fn stacks(model: &ModelTensors, plan: &Plan<'_>) -> Result<Qwen38Stacks, GpuError> {
    swap::stacks::<Leg38>(model, |l| {
        let card = plan.n_l.get(l).copied();
        let tiers = plan.tier_n_l.iter().map(|t| t.get(l).copied());
        match std::iter::once(card)
            .chain(tiers)
            .collect::<Option<Vec<u64>>>()
        {
            Some(n) => Ok(n.iter().any(|&n| n > 0)),
            None => Err(GpuError::shape(
                "swap38::stacks",
                format!("layer {l} is past the plan's {} layers", plan.n_l.len()),
            )),
        }
    })
}

/// The stacks of `model`'s layers, the layers `set` gives tier experts
/// checked against [`Leg38`] ([`swap::stacks`]): a tier node's, which holds
/// one tier's set and no plan. Load-time only.
pub fn stacks_of_set(
    model: &ModelTensors,
    set: &crate::host::tier::TierSet,
) -> Result<Qwen38Stacks, GpuError> {
    swap::stacks::<Leg38>(model, |l| Ok(set.ids(l).is_some_and(|ids| !ids.is_empty())))
}
