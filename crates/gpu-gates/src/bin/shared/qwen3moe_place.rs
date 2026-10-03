//! A placed load of a qwen3moe (Qwen3-30B-A3B) or qwen35moe
//! (Qwen3.6-35B-A3B) file, as `generate_qwen3moe` and the qwen3 serve seat
//! open one under `--place`: the placement word's stage card
//! (`generate::Place`, an alias or a card list of one card; a tier card is
//! refused by name, the program hangs none), the machine
//! `model::arch::qwen3moe::place::machine` lays out on it with the scratch
//! the body's placed load holds past the m = 1 scratch, the plan by the
//! expert rule over the stacks `card_routed` loads — the trunk on the card,
//! each layer's routed id prefix as the budget holds, the rest on the host
//! tier — and the `plan` record of it. A plan that holds no host expert
//! opens the whole model on the plan's card, as the unplaced load does; any
//! other opens the body's placed load.

use bloomery_gpu::arch::qwen3moe::{Body, Body35, Open35, OpenOpts, Qwen35moeModel};
use bloomery_gpu::model::GpuModel;
use bloomery_gpu::{Gpu, Qwen3moeModel};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::generate::Place;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::qwen3moe::place::{self as q3, card_routed};
use model::arch::qwen35moe::place as q35;
use model::placement::{Machine, Plan, PlanLevers};

/// The levers a placed load acts on: the plan's card budget and the host
/// set's load (`Levers::host`). A binary lists them as acted on only when
/// `--place` is given, so one set beside an unplaced load is refused by
/// name.
pub const PLACED_LEVERS: [&str; 4] = [
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
];

/// The plan inputs of one of the two architectures; the qwen35moe ones,
/// which carry the typed model description too, boxed.
enum Inputs {
    Qwen3(q3::PlanInputs),
    Qwen35(Box<q35::PlanInputs>),
}

/// What a placed load plans from: the placement on this process's devices,
/// the file's plan inputs and the machine its stage card lays out.
pub struct PlaceQ3 {
    place: Place,
    arch: &'static str,
    inputs: Inputs,
    machine: Machine,
}

impl PlaceQ3 {
    /// `place` resolved against this process's devices, refused by name
    /// when it names an expert tier card.
    fn resolve(place: Place) -> Result<Place, GateError> {
        let place = place.on_host()?;
        place.serves("qwen3moe", 0)?;
        Ok(place)
    }

    /// A qwen3moe file's placed load at `ctx` positions on `place`'s card.
    pub fn qwen3(file: &Split, place: Place, ctx: usize) -> Result<PlaceQ3, GateError> {
        let place = PlaceQ3::resolve(place)?;
        let inputs = q3::PlanInputs::read(file)?;
        let arena = Qwen3moeModel::placed_arena_bytes(file, ctx)?;
        let machine = q3::machine(place.card_specs()?[0], inputs.hp.n_layer, arena);
        Ok(PlaceQ3 {
            place,
            arch: "qwen3moe",
            inputs: Inputs::Qwen3(inputs),
            machine,
        })
    }

    /// A qwen35moe file's placed load under `o` on `place`'s card.
    pub fn qwen35(file: &Split, place: Place, o: Open35) -> Result<PlaceQ3, GateError> {
        let place = PlaceQ3::resolve(place)?;
        let inputs = q35::PlanInputs::describe(file)?;
        let arena = Qwen35moeModel::placed_arena_bytes(file, o)?;
        let machine = q3::machine(place.card_specs()?[0], inputs.hp.n_layer, arena);
        Ok(PlaceQ3 {
            place,
            arch: "qwen35moe",
            inputs: Inputs::Qwen35(Box::new(inputs)),
            machine,
        })
    }

    /// The plan at `ctx` positions under `levers` (the card budget), every
    /// refusal by name with its terms.
    pub fn plan(&self, ctx: usize, levers: &PlanLevers) -> Result<Plan<'_>, GateError> {
        let ctx = u64::try_from(ctx)?;
        Ok(match &self.inputs {
            Inputs::Qwen3(i) => i.plan(&self.machine, ctx, levers)?,
            Inputs::Qwen35(i) => i.plan_rule(&self.machine, ctx, levers, card_routed)?,
        })
    }

    /// The `plan` record of `plan`: the card, the architecture, the expert
    /// rule, the routed experts on the card and the host, the fewest and
    /// the most one layer keeps on the card, and its device.
    pub fn record(&self, plan: &Plan<'_>) -> Record {
        let card = &plan.cards[0];
        Record::new(&record::PLAN38)
            .w("place", self.place.name())
            .w("card", &self.machine.cards[0].name)
            .w("arch", self.arch)
            .w("experts", "card")
            .u("ctx_max", plan.ctx_max)
            .u("host_experts", plan.host.experts)
            .u("card_experts", card.experts)
            .w(
                "n_l",
                format!(
                    "{}-{}",
                    plan.n_l.iter().min().copied().unwrap_or(0),
                    plan.n_l.iter().max().copied().unwrap_or(0)
                ),
            )
            .csv("devices", record::plan_devices(plan.machine))
            .w("cuda_order", record::cuda_order())
    }
}

/// The Qwen3-30B model of `file` by `plan`: whole on the plan's card when
/// the plan holds no host expert, else the placed load, its host set as
/// `host` asks.
pub fn open_qwen3(
    file: Split,
    plan: &Plan<'_>,
    opts: OpenOpts,
    host: HostCfg,
) -> Result<Qwen3moeModel, GateError> {
    if plan.host.experts == 0 {
        let card = &plan.machine.cards[0];
        let gpu = Gpu::open_card(&card.name, card.device)?;
        return Ok(GpuModel::<Body>::open(gpu, file, opts)?);
    }
    Ok(GpuModel::<Body>::open_placed(file, plan, opts, host)?)
}

/// The Qwen3.6 model of `file` by `plan`, as [`open_qwen3`].
pub fn open_qwen35(
    file: Split,
    plan: &Plan<'_>,
    o: Open35,
    host: HostCfg,
) -> Result<Qwen35moeModel, GateError> {
    if plan.host.experts == 0 {
        let card = &plan.machine.cards[0];
        let gpu = Gpu::open_card(&card.name, card.device)?;
        return Ok(GpuModel::<Body35>::open(gpu, file, o)?);
    }
    Ok(GpuModel::<Body35>::open_placed(file, plan, o, host)?)
}
