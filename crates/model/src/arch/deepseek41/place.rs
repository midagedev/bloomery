//! A DeepSeek-V4.1 or V4 file planned onto a machine: the one path from the
//! file's headers to a plan that keeps its invariants. The hyperparameters
//! ([`Hparams`]), every tensor's role ([`roles::classify`]), the typed model
//! description ([`ModelSpec`]) and each layer's KV bytes ([`KvLayout`]) are
//! read once ([`PlanInputs::read`]), and a file with a feature the engine does
//! not run yet is refused there, every such feature listed by the coverage
//! check ([`PlanInputs::unimplemented`]); the caller lays its machine
//! out for the model's layers; then the placement and its invariants
//! ([`PlanInputs::plan`]). The engine's entry and the load gates all take this
//! path, so they refuse the same file for the same reason, in the same order.
//! [`PlanInputs::describe`] is the same read without that refusal, for what
//! only describes a file (its inventory gate).

use gguf::Split;
use models::ModelSpec;

use super::hparams::Hparams;
use super::kv::KvLayout;
use super::{roles, spec};
use crate::arch::coverage;
use crate::placement::{
    self, Machine, ModelTensors, PlacementError, Plan, PlanLevers, Unimplemented, Violation,
};

/// What a plan of a V4.1 or V4 file is made from, read from its headers.
#[derive(Debug)]
pub struct PlanInputs {
    /// The file's hyperparameters.
    pub hp: Hparams,
    /// Every tensor with its role, and the model's layer and expert counts.
    pub model: ModelTensors,
    /// The typed model description the coverage check reads.
    pub spec: ModelSpec,
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
    /// `split`'s hyperparameters, then its tensors' roles, then its
    /// description, then its layers' KV bytes, the first that fails being the
    /// error; then the file is refused if it has a feature the engine does
    /// not run yet ([`PlacementError::Unimplemented`], every one listed).
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        let inputs = PlanInputs::describe(split)?;
        let missing = inputs.unimplemented();
        if missing.is_empty() {
            Ok(inputs)
        } else {
            Err(PlacementError::Unimplemented(missing))
        }
    }

    /// [`PlanInputs::read`] without the refusal of unimplemented features: the
    /// file as it is, for a reader that only describes it.
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let model = roles::classify(split, &hp)?;
        let spec = spec::spec_of(&hp, &model, spec::chat(split)?)?;
        let kv = KvLayout::from_file(split, &hp)?;
        Ok(PlanInputs {
            hp,
            model,
            spec,
            kv,
        })
    }

    /// Every feature of the file the engine does not run yet, layer by layer
    /// in layer order, then the tensor formats, then the model-wide ones
    /// ([`coverage::check`]). Empty for a file the chain runs whole.
    pub fn unimplemented(&self) -> Vec<Unimplemented> {
        coverage::check(&self.spec, &self.model)
    }

    /// The placement of the file on `machine` at `ctx_max` positions under
    /// the placement's `levers`, refused when it cannot be built or breaks an
    /// invariant.
    pub fn plan<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, PlaceError> {
        let plan = placement::plan(&self.model, machine, ctx_max, &self.kv, levers)?;
        let broken = plan.violations();
        if broken.is_empty() {
            Ok(plan)
        } else {
            Err(PlaceError::Broken(broken))
        }
    }
}
