//! The plan-then-open of a Qwen3.8 (`qwen4exp`) gate's load: the file described, the host's room
//! read or given, the machine for where the routed experts sit, the plan for the resident slots
//! ([`Open38::plan`]), then the body opened by it and its steps captured as graphs
//! ([`Planned38::open`]). What a gate prints of the plan and of the load stays in the gate: it
//! prints [`PlanFacts`] between the two calls, where the load's own lines to stderr follow it, and
//! times the load itself.
//!
//! Every parameter is a value ([`Open38`]), so a gate that loads another way changes a value, not
//! this code: the card a plan is made on, the cache rows, the ubatch, where the experts sit, the
//! slots, the plan's levers, the host config, the host's room and the residency.

use std::path::Path;

use bloomery_gpu::arch::qwen3moe::{Body38, Qwen38Model};
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::model::StepMode;
use bloomery_gpu_gates::GateError;
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::qwen35moe::place::{Experts, PlanInputs, machine_for_experts};
use model::placement::workstation::{CardSpec, HostNeed, HostRead};
use model::placement::{Device, Machine, Plan, PlanLevers};

/// The file at `path`, opened as a split of architecture `qwen4exp`; any other architecture is
/// refused by name.
pub fn open_split(path: &Path) -> Result<Split, GateError> {
    let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    if file.architecture() != Some("qwen4exp") {
        return Err(format!(
            "{} is {:?}, not qwen4exp",
            path.display(),
            file.architecture()
        )
        .into());
    }
    Ok(file)
}

/// What a Qwen3.8 load is made from.
#[derive(Clone, Copy)]
pub struct Open38<'a> {
    /// The card the plan is made on.
    pub card: CardSpec,
    /// Cache rows.
    pub ctx: usize,
    /// The ubatch the machine's arenas and the body's walk are sized for.
    pub ub: usize,
    /// Where the routed experts sit: on the host, or on the card's id prefix.
    pub experts: Experts,
    /// Resident sequences the plan counts and the body serves.
    pub slots: usize,
    pub plan_levers: &'a PlanLevers,
    pub host: HostCfg,
    /// The host's room in bytes, given; `None` takes the reading
    /// ([`PlanInputs::room`]).
    pub room: Option<u64>,
    /// The residency machine over the card's routed stacks; [`Residency::Off`] is the load's slot
    /// map for the model's life.
    pub residency: Residency,
}

/// What the plan the load was made by holds, as a gate prints it.
#[derive(Clone, Debug)]
pub struct PlanFacts {
    pub ctx_max: u64,
    /// Routed experts on the host and on the card.
    pub host_experts: u64,
    pub card_experts: u64,
    /// The layers holding routed experts on the card, and each layer's count.
    pub card_layers: usize,
    pub n_l: Vec<u64>,
    /// The tier the plan reads the PLE table from.
    pub ple_tier: Option<Device>,
    /// The host room the plan was made under, and how it was read.
    pub room: (u64, HostRead),
    /// The host's bytes set aside for the table's rows when it sits on the NVMe tier.
    pub row_reserve_bytes: u64,
    /// What the plan asks of the host's available bytes, with nothing held beside the host set.
    pub host_need: HostNeed,
}

/// A load made: the model, in graph step mode, and the plan facts it was made by.
pub struct Opened38 {
    pub model: Qwen38Model,
    pub plan: PlanFacts,
}

/// A file described and planned, not yet loaded: a gate prints the plan's facts, then loads
/// ([`Planned38::open`]); the load writes lines of its own to stderr, which a gate's log orders
/// after the plan line.
pub struct Planned38<'a> {
    file: Split,
    inputs: PlanInputs,
    machine: Machine,
    args: Open38<'a>,
    facts: PlanFacts,
}

impl<'a> Open38<'a> {
    /// `file` described and planned. `describe`, not `read`, so the chat surface's two coverage
    /// items are not refused here; the load refuses what `ALLOWED` does not name.
    pub fn plan(&self, file: Split) -> Result<Planned38<'a>, GateError> {
        let mut inputs = PlanInputs::describe(&file)?;
        if let Some(r) = self.room {
            inputs.room = (r, HostRead::Given);
        }
        let machine = machine_for_experts(
            self.card,
            inputs.spec.layers.len(),
            u64::try_from(self.ub)?,
            self.experts,
        );
        let plan = self.plan_of(&inputs, &machine)?;
        let facts = PlanFacts {
            ctx_max: plan.ctx_max,
            host_experts: plan.host.experts,
            card_experts: plan.cards[0].experts,
            card_layers: plan.n_l.iter().filter(|&&n| n > 0).count(),
            n_l: plan.n_l.clone(),
            ple_tier: plan.row_tier()?,
            room: inputs.room.clone(),
            row_reserve_bytes: plan.host.row_reserve_bytes,
            host_need: HostNeed::of(&plan, 0),
        };
        drop(plan);
        Ok(Planned38 {
            file,
            inputs,
            machine,
            args: *self,
            facts,
        })
    }

    /// The plan of `inputs` on `machine` under these values.
    fn plan_of<'p>(
        &self,
        inputs: &'p PlanInputs,
        machine: &'p Machine,
    ) -> Result<Plan<'p>, GateError> {
        Ok(inputs.plan_with_slots(
            machine,
            self.ctx as u64,
            self.plan_levers,
            self.experts,
            self.slots,
        )?)
    }
}

impl Planned38<'_> {
    /// What the plan holds.
    pub fn facts(&self) -> &PlanFacts {
        &self.facts
    }

    /// The load: the same plan made again over the same inputs and machine (deterministic), the
    /// body opened by it and its steps captured as graphs.
    pub fn open(self) -> Result<Opened38, GateError> {
        let a = self.args;
        let plan = a.plan_of(&self.inputs, &self.machine)?;
        let mut model = Body38::open_placed_residency(
            self.file,
            &plan,
            &self.inputs,
            0,
            a.host,
            a.ub,
            a.residency,
            a.slots,
        )?;
        model.set_mode(StepMode::Graph);
        Ok(Opened38 {
            model,
            plan: self.facts,
        })
    }
}
