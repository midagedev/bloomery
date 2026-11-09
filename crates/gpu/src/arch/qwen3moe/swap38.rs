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
//! bytes in slot order ([`place::routed_rule`] keeps these stacks in
//! `CardFormat::KQuant`, and `super::card38` reads them exactly so). Such a
//! part converts nothing: it is its slot's bytes as they are, so the r8
//! sidecar, which exists for Q3_K stacks alone (`qdot::repack_q3k_r8`), is
//! undefined input here, refused by name. An NVFP4 stack loads in its planes,
//! three parts a matrix, each staged from the expert's run of the file's
//! blocks by its transform (`crate::host::swap::Transform`), and the host
//! view reads the same blocks with qdot's NVFP4 dot.

use super::swap::{self, CardLeg};
use crate::GpuError;
use crate::host::swap_source::PlanStacks;
use model::placement::accepts::Program;
use model::placement::{ModelTensors, Plan, Role, elastic, per_expert};

pub use crate::host::swap_source::{DEADLINE, LIVE_DELAY};

/// Qwen3.8's card leg ([`CardLeg`]): the Qwen3.8 body's arms
/// (`placement::accepts::leg_arms`: a Q4_K, Q5_K, IQ3_XXS, IQ4_XS or NVFP4
/// gate and up, a Q5_1, Q8_0, IQ4_NL or NVFP4 down), which its place rule
/// reads too.
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

/// [`stacks`] for an elastic load: each layer the plan puts experts on reserves
/// its card stacks for the slots the card's assumed terms could hold
/// ([`elastic::capacity`]: the plan's count plus the quanta of
/// [`elastic::reclaimable`] bytes, a quantum one slot on every such layer at
/// the stack's own bytes an expert), the plan's rows backed and the rest
/// withdrawn until the machine fills them. A layer with no card expert
/// reserves nothing. Load-time only.
pub fn elastic_stacks(
    model: &ModelTensors,
    plan: &Plan<'_>,
    card: usize,
) -> Result<Qwen38Stacks, GpuError> {
    const WHAT: &str = "swap38::elastic_stacks";
    let table = stacks(model, plan)?;
    let (Some(spec), Some(totals)) = (plan.machine.cards.get(card), plan.cards.get(card)) else {
        return Err(GpuError::shape(
            WHAT,
            format!("the plan has no card {card}"),
        ));
    };
    let mut slot_bytes = vec![0u64; model.layers];
    for t in model
        .tensors
        .iter()
        .filter(|t| t.role == Role::RoutedExperts)
    {
        if let Some(b) = t.layer.and_then(|l| slot_bytes.get_mut(l)) {
            let (_, per) = per_expert(t, model.experts).map_err(|e| GpuError::plan(WHAT, e))?;
            *b = b.saturating_add(per);
        }
    }
    let live = |l: usize| plan.n_l.get(l).copied().unwrap_or(0);
    let quantum = (0..model.layers)
        .filter(|&l| live(l) > 0)
        .fold(0u64, |q, l| q.saturating_add(slot_bytes[l]));
    let room = elastic::reclaimable(totals, spec.margin_bytes);
    let capacity = (0..model.layers)
        .map(|l| {
            (live(l) > 0)
                .then(|| {
                    usize::try_from(elastic::capacity(live(l), room, quantum, model.experts))
                        .map_err(|_| {
                            GpuError::shape(WHAT, format!("layer {l}'s capacity passes usize"))
                        })
                })
                .transpose()
        })
        .collect::<Result<Vec<Option<usize>>, GpuError>>()?;
    table.with_capacity(capacity)
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
