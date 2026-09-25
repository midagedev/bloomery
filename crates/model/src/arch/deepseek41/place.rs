//! A DeepSeek-V4.1 or V4 file planned onto a machine: the one path from the
//! file's headers to a plan that keeps its invariants. The hyperparameters
//! ([`Hparams`]), every tensor's role ([`roles::classify`]) and each layer's
//! KV bytes ([`KvLayout`]) are read once ([`PlanInputs::read`]), and a file
//! with a feature the engine does not run yet is refused there, every such
//! feature listed ([`PlanInputs::unimplemented`]); the caller lays its machine
//! out for the model's layers; then the placement and its invariants
//! ([`PlanInputs::plan`]). The engine's entry and the load gates all take this
//! path, so they refuse the same file for the same reason, in the same order.
//! [`PlanInputs::describe`] is the same read without that refusal, for what
//! only describes a file (its inventory gate).

use gguf::Split;

use super::hparams::{Collapse, Hparams};
use super::kv::KvLayout;
use super::roles;
use crate::placement::{
    self, CardFormat, Machine, ModelTensors, PlacementError, Plan, Role, Unimplemented, Violation,
};

/// What a plan of a V4.1 or V4 file is made from, read from its headers.
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
    /// KV bytes, the first that fails being the error; then the file is
    /// refused if it has a feature the engine does not run yet
    /// ([`PlacementError::Unimplemented`], every one listed).
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
        let kv = KvLayout::from_file(split, &hp)?;
        Ok(PlanInputs { hp, model, kv })
    }

    /// Every feature of the file the engine does not run yet, layer by layer
    /// in layer order, then the model-wide ones. Empty for a file the chain
    /// runs whole.
    pub fn unimplemented(&self) -> Vec<Unimplemented> {
        let hp = &self.hp;
        let mut out = Vec::new();
        let mut at = |layer: Option<usize>, feature: &str| {
            let f = Unimplemented {
                layer,
                feature: feature.to_string(),
            };
            if !out.contains(&f) {
                out.push(f);
            }
        };
        for (l, k) in hp.layers.iter().enumerate() {
            let l = Some(l);
            if k.hash_routed {
                at(l, "hash routing by ffn_gate_tid2eid");
            }
            if k.dense.is_some() {
                at(l, "a compressed stream attended whole, without a top-k");
            }
            if let Some(c) = k.compressor {
                if c.ape {
                    at(l, "the compressor's position table attn_compressor_ape");
                }
                if c.overlap {
                    at(l, "overlapping compressor groups");
                }
            }
            if k.index_compressor.is_some() {
                at(l, "index keys from the indexer's own compressor");
            }
        }
        for t in &self.model.tensors {
            if t.role == Role::RoutedExperts && CardFormat::of(t.ty).is_none() {
                at(t.layer, &format!("{} routed experts on a card", t.ty));
            }
        }
        if hp.q_head_norm {
            at(None, "the per-head query RMS norm");
        }
        if hp.collapse == Collapse::Head {
            at(None, "the hyper-connection head output_hc_*");
        }
        if hp.engram.is_none() {
            at(None, "a model without engram sites");
        }
        out
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
