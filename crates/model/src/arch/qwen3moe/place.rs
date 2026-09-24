//! A Qwen3-MoE file planned onto a machine: the one path from the file's
//! headers to a plan that keeps its invariants. The hyperparameters
//! ([`Hparams`]), every tensor's role ([`roles::classify`]) and the KV bytes
//! ([`KvLayout`]) are read once ([`PlanInputs::read`]); then the shared
//! placement and its invariants ([`PlanInputs::plan`]); then, for a plan that
//! is meant to hold the whole model on its cards, that every routed expert is
//! on one ([`PlanInputs::whole`]). The token embedding is a row-gathered table
//! and stays on the host by the shared rule; one row of it is read per token.

use gguf::Split;

use super::hparams::Hparams;
use super::kv::KvLayout;
use super::roles;
use crate::placement::{self, Machine, ModelTensors, PlacementError, Plan, Violation};

/// What a plan of a Qwen3-MoE file is made from, read from its headers.
#[derive(Debug)]
pub struct PlanInputs {
    /// The file's hyperparameters.
    pub hp: Hparams,
    /// Every tensor with its role, and the model's layer and expert counts.
    pub model: ModelTensors,
    /// Each layer's cache bytes.
    pub kv: KvLayout,
}

/// Why a plan was refused.
#[derive(Debug, thiserror::Error)]
pub enum PlaceError {
    /// The placement could not be built.
    #[error(transparent)]
    Placement(#[from] PlacementError),
    /// The plan was built and breaks these invariants, every one of them.
    #[error("the plan breaks its invariants: {}", joined(.0))]
    Broken(Vec<Violation>),
    /// The plan keeps some routed experts off the cards: each such layer
    /// with the experts its card holds.
    #[error(
        "the plan is not whole: {} of {layers} layers keep experts on the host ({})",
        short.len(),
        short.iter().map(|(l, n)| format!("layer {l}: {n} of {experts}")).collect::<Vec<_>>().join(", ")
    )]
    NotWhole {
        layers: usize,
        experts: u64,
        short: Vec<(usize, u64)>,
    },
}

/// The violations, `; `-separated.
fn joined(broken: &[Violation]) -> String {
    let list: Vec<String> = broken.iter().map(ToString::to_string).collect();
    list.join("; ")
}

impl PlanInputs {
    /// `split`'s hyperparameters, then its tensors' roles, then its KV bytes;
    /// the first that fails is the error.
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let model = roles::classify(split, &hp)?;
        let kv = KvLayout::from_file(split, &hp)?;
        Ok(PlanInputs { hp, model, kv })
    }

    /// The placement of the file on `machine` at `ctx_max` positions under
    /// the placement levers (`BLOOMERY_HOT_LIST`, `BLOOMERY_CARD_BUDGET`),
    /// refused when it cannot be built or breaks an invariant.
    pub fn plan<'a>(&'a self, machine: &'a Machine, ctx_max: u64) -> Result<Plan<'a>, PlaceError> {
        checked(placement::plan(&self.model, machine, ctx_max, &self.kv)?)
    }

    /// [`PlanInputs::plan`] with no hot list and the card budget
    /// `card_budget` instead of the levers' — what a gate plans with.
    pub fn plan_with<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        card_budget: Option<u64>,
    ) -> Result<Plan<'a>, PlaceError> {
        checked(placement::plan_with(
            &self.model,
            machine,
            ctx_max,
            &self.kv,
            None,
            card_budget,
        )?)
    }

    /// `plan` itself when it holds every routed expert of every layer on a
    /// card, else the layers it does not.
    pub fn whole<'a>(&self, plan: Plan<'a>) -> Result<Plan<'a>, PlaceError> {
        let experts = self.model.experts;
        let short: Vec<(usize, u64)> = plan
            .n_l
            .iter()
            .enumerate()
            .filter(|&(_, &n)| n < experts)
            .map(|(l, &n)| (l, n))
            .collect();
        if short.is_empty() {
            Ok(plan)
        } else {
            Err(PlaceError::NotWhole {
                layers: self.model.layers,
                experts,
                short,
            })
        }
    }
}

/// `plan`, or the invariants it breaks.
fn checked(plan: Plan<'_>) -> Result<Plan<'_>, PlaceError> {
    let broken = plan.violations();
    if broken.is_empty() {
        Ok(plan)
    } else {
        Err(PlaceError::Broken(broken))
    }
}
