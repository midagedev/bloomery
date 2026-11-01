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

use std::num::NonZeroUsize;

use bloomery_placement::slots::{SeqTerms, Stores};
use gguf::Split;
use models::ModelSpec;

use super::hparams::Hparams;
use super::kv::KvLayout;
use super::{roles, spec};
use crate::arch::coverage;
use crate::placement::accepts::{Program, routed_rule};
use crate::placement::ctx_ask::PlanRefusal;
use crate::placement::workstation::{self, HostRead, TierBatchBytes};
use crate::placement::{
    self, Machine, ModelTensors, PlacementError, Plan, PlanLevers, Unimplemented, Violation,
    checked, joined,
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
    /// The host's available bytes when the inputs were read, with the
    /// reading that decided ([`workstation::host_available_read`]): the
    /// routed experts the room cannot hold follow it to the NVMe tier
    /// ([`placement::expert_nvme_tier`]). The engram row tables are on the
    /// NVMe tier by their role at any room, so no row-table tier runs here. A
    /// gate that plans an arm by a room of its own sets it, read
    /// [`HostRead::Given`].
    ///
    /// The effective room is [`Plan::plan_at`]'s: the reading's plan room,
    /// capped by the machine's host room and unified pool.
    pub room: (u64, HostRead),
    /// The arena's bytes `BLOOMERY_NVTIER_BYTES` gave the split dial
    /// ([`workstation::nvtier_arena_given`]), `None` unset; the split takes it
    /// as its `given`. A gate that plans an arm by an arena of its own sets it.
    pub arena: Option<u64>,
}

/// The expert tier's prompt-batch bytes plan (b′) reserves for a file of
/// hyperparameters `hp` (`workstation::plan_bp`): blocks of the host union's
/// [`crate::moe::UNION_MAX_COLS`] columns of rows of `n_embd`, routed to
/// `n_used` experts of `ff` rows — the sizes the tier's staging and tile
/// scratch are allocated at ([`workstation::tier_batch_bytes`]).
#[must_use]
pub fn tier_batch(hp: &Hparams) -> TierBatchBytes {
    workstation::tier_batch_bytes(
        hp.n_embd as u64,
        hp.experts.ff as u64,
        hp.experts.n_used as u64,
        crate::moe::UNION_MAX_COLS as u64,
    )
}

/// Why a plan was refused.
#[derive(Debug, thiserror::Error)]
pub enum PlaceError {
    /// The placement could not be built.
    #[error("{0}")]
    Placement(#[from] PlacementError),
    /// The plan was built and breaks these invariants, every one of them.
    #[error("the plan breaks its invariants: {}", joined(.0))]
    Broken(Vec<Violation>),
}

impl PlanRefusal for PlaceError {
    fn placement(&self) -> Option<&PlacementError> {
        match self {
            Self::Placement(e) => Some(e),
            Self::Broken(_) => None,
        }
    }

    fn violations(&self) -> Option<&[Violation]> {
        match self {
            Self::Broken(v) => Some(v),
            Self::Placement(_) => None,
        }
    }
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
    /// file as it is, for a reader that only describes it, and the host's
    /// room ([`workstation::host_available_read`]) and the arena's given
    /// bytes, a room that cannot be read refused by name. The room is read
    /// after the file's description, through the same steps as
    /// [`PlanInputs::describe_in`].
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        Self::describe_with(split, || {
            workstation::host_available_read().map_err(PlacementError::HostRoom)
        })
    }

    /// The file's description with `room` as the host's room, no host read
    /// for it: another machine's plan checked from this one, `room` that
    /// machine's bytes and how they were read. The arena is still this
    /// process's `BLOOMERY_NVTIER_BYTES`, as in [`PlanInputs::describe`].
    pub fn describe_in(split: &Split, room: (u64, HostRead)) -> Result<PlanInputs, PlacementError> {
        Self::describe_with(split, || Ok(room))
    }

    /// The file's hyperparameters, roles, spec and KV layout, then the room
    /// `room` gives, then the arena, the first that fails being the error.
    fn describe_with(
        split: &Split,
        room: impl FnOnce() -> Result<(u64, HostRead), PlacementError>,
    ) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let model = roles::classify(split, &hp)?;
        let spec = spec::spec_of(&hp, &model, spec::chat(split)?)?;
        let kv = KvLayout::from_file(split, &hp)?;
        let room = room()?;
        let arena = workstation::nvtier_arena_given().map_err(PlacementError::HostRoom)?;
        Ok(PlanInputs {
            hp,
            model,
            spec,
            kv,
            room,
            arena,
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
    /// invariant: the plan of one resident sequence
    /// ([`PlanInputs::plan_with_slots`]).
    pub fn plan<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, PlaceError> {
        self.plan_with_slots(machine, ctx_max, levers, NonZeroUsize::MIN)
    }

    /// The placement of a load that serves `slots` resident sequences of
    /// `ctx_max` positions each: every layer's cache counted `slots` times on
    /// its card and every ring shadow `slots` times on the host
    /// ([`SeqTerms::slots_of`] over [`PlanInputs::seq_terms`]), refused as
    /// [`PlanInputs::plan`] refuses. A sequence holds nothing beside its
    /// stores, so the card's kv class is `slots` sequences' stores.
    pub fn plan_with_slots<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        slots: NonZeroUsize,
    ) -> Result<Plan<'a>, PlaceError> {
        self.plan_slots_nodes(machine, ctx_max, levers, slots, None)
    }

    /// Lower explicit node-owned expert ranges through the common placement owner.
    pub fn plan_nodes<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        owned: &[u64],
    ) -> Result<Plan<'a>, PlaceError> {
        self.plan_slots_nodes(machine, ctx_max, levers, NonZeroUsize::MIN, Some(owned))
    }

    fn plan_slots_nodes<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        slots: NonZeroUsize,
        owned: Option<&[u64]>,
    ) -> Result<Plan<'a>, PlaceError> {
        let kv = self.seq_terms().slots_of(slots.get() as u64);
        let rule = routed_rule(Program::Deepseek41Chain);
        let plan = placement::plan_in_pool(machine, levers, |levers| {
            let mut plan = placement::nodes::loaded_routed(
                &self.model,
                machine,
                ctx_max,
                &kv,
                levers,
                rule,
                0,
                owned,
            )?;
            let room = plan.plan_at(&self.room);
            placement::expert_nvme_tier(&mut plan, room, self.arena)?;
            Ok(plan)
        })?;
        checked(plan).map_err(PlaceError::Broken)
    }

    /// What one resident sequence of a load of the file holds on its card
    /// ([`SeqTerms`]): its stores over the file's layers ([`KvLayout`]: each
    /// layer's window ring, compressed rows, index keys and compressor
    /// state), no draft (the DSpark draft's window is its own card's), and
    /// nothing beside them — the body's sequence is its layers' caches alone.
    /// Its ring shadows are host bytes, which [`SeqTerms::slots_of`] counts
    /// on the host ([`crate::placement::KvBytes::shadow_bytes`]).
    #[must_use]
    pub fn seq_terms(&self) -> SeqTerms<'_> {
        SeqTerms {
            layers: Stores {
                kv: &self.kv,
                count: self.model.layers,
            },
            draft: None,
            beside: 0,
        }
    }
}
