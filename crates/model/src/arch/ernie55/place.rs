//! An ERNIE 5.5 file planned onto a machine: the inputs ([`PlanInputs`]) the placement owner's
//! `plan_routed_reserving` reads, the routed rule of [`Program::Ernie55Body`] and this family's KV
//! byte function ([`super::kv::KvLayout`]). The common hybrid owner's inputs
//! (`place_hybrid::PlanInputs`) carry a KV layout that refuses a latent layer which compresses,
//! so this family holds its own; the roles and the description are the registry's reader's
//! ([`spec::read`]), so the plan and the description of a file are one reading.

use gguf::Split;
use models::{ModelSpec, Unimplemented};

use super::hparams::Hparams;
use super::kv::{KvLayout, kv_layout};
use super::ops::Ops;
use super::spec;
use crate::arch::coverage;
use crate::placement::{ModelTensors, PlacementError};

/// What a plan of an ERNIE 5.5 file is made from, read from its headers.
#[derive(Clone, Debug)]
pub struct PlanInputs {
    /// The file's hyperparameters.
    pub hp: Hparams,
    /// Every tensor with its role, and the model's layer and expert counts.
    pub model: ModelTensors,
    /// The typed model description the coverage check reads.
    pub spec: ModelSpec,
    /// Each layer's stores.
    pub kv: KvLayout,
    /// What the file asks beyond the description.
    pub ops: Ops,
}

impl PlanInputs {
    /// The file as it is, whatever the coverage list holds.
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let read = spec::read(split)?;
        let kv = kv_layout(&hp);
        let ops = Ops::of(&hp)?;
        Ok(PlanInputs {
            hp,
            model: read.tensors,
            spec: read.spec,
            kv,
            ops,
        })
    }

    /// [`PlanInputs::describe`], then the file refused with every feature the engine does not run
    /// ([`PlacementError::Unimplemented`]).
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        PlanInputs::describe(split)?.runnable()
    }

    /// Every feature of the file the engine does not run: the coverage check's items and the ops
    /// the description cannot state ([`Ops::unrun`]). Empty for a file a program runs.
    #[must_use]
    pub fn unimplemented(&self) -> Vec<Unimplemented> {
        let mut items = coverage::check(&self.spec, &self.model);
        items.extend(self.ops.unrun());
        items
    }

    /// `self`, or [`PlacementError::Unimplemented`] with every feature the engine does not run.
    pub fn runnable(self) -> Result<PlanInputs, PlacementError> {
        let missing = self.unimplemented();
        if missing.is_empty() {
            Ok(self)
        } else {
            Err(PlacementError::Unimplemented(missing))
        }
    }

    /// A context of `ctx` positions: at least one and at most the trained context
    /// (`context_length`), refused by name past either.
    pub fn check_ctx(&self, ctx: u64) -> Result<(), PlacementError> {
        let train = self.hp.n_ctx_train as u64;
        if ctx == 0 || ctx > train {
            return Err(PlacementError::Metadata {
                key: "context_length".to_string(),
                detail: format!("a context of {ctx} positions; the file trains {train}"),
            });
        }
        Ok(())
    }
}
