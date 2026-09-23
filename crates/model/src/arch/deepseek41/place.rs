//! A DeepSeek-V4.1 file planned onto a machine: the one path from the file's
//! headers to a plan that keeps its invariants. The hyperparameters
//! ([`Hparams`]), every tensor's role ([`roles::classify`]) and each layer's
//! KV bytes ([`KvLayout`]) are read once ([`PlanInputs::read`]); the caller
//! lays its machine out for the model's layers; then the placement and its
//! invariants ([`PlanInputs::plan`]). The engine's entry and the load gates
//! all take this path, so they refuse the same file for the same reason, in
//! the same order.

use gguf::Split;

use super::hparams::Hparams;
use super::kv::KvLayout;
use super::roles;
use crate::placement::{self, Machine, ModelTensors, PlacementError, Plan, Violation};

/// What a plan of a V4.1 file is made from, read from its headers.
#[derive(Debug)]
pub struct PlanInputs {
    /// The file's hyperparameters.
    pub hp: Hparams,
    /// Every tensor with its role, and the model's layer and expert counts.
    pub model: ModelTensors,
    /// Each layer's cache and compressor bytes.
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
}

/// The violations, `; `-separated.
fn joined(broken: &[Violation]) -> String {
    let list: Vec<String> = broken.iter().map(ToString::to_string).collect();
    list.join("; ")
}

impl PlanInputs {
    /// `split`'s hyperparameters, then its tensors' roles, then its layers'
    /// KV bytes; the first that fails is the error.
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let model = roles::classify(split, &hp)?;
        let kv = KvLayout::from_file(split, &hp)?;
        Ok(PlanInputs { hp, model, kv })
    }

    /// The placement of the file on `machine` at `ctx_max` positions, refused
    /// when it cannot be built or breaks an invariant.
    pub fn plan<'a>(&'a self, machine: &'a Machine, ctx_max: u64) -> Result<Plan<'a>, PlaceError> {
        let plan = placement::plan(&self.model, machine, ctx_max, &self.kv)?;
        let broken = plan.violations();
        if broken.is_empty() {
            Ok(plan)
        } else {
            Err(PlaceError::Broken(broken))
        }
    }
}
