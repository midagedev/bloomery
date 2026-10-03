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
//!
//! The same two binaries' `--place`-unset default lives here too
//! ([`unplaced_qwen3`]): today's whole model on device 0 while that fits
//! the card's free bytes — the census reading the plan's own budget term
//! uses — and when it does not, the placed plan on `a`'s card.

use bloomery_gpu::arch::qwen3moe::{Body, Body35, Open35, OpenOpts, Qwen35moeModel};
use bloomery_gpu::model::GpuModel;
use bloomery_gpu::{Gpu, Qwen3moeModel};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::generate::Place;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_levers::{HostCfg, Levers};
use gguf::Split;
use model::arch::qwen3moe::place::{self as q3, card_routed};
use model::arch::qwen35moe::place as q35;
use model::placement::{
    CardFormat, KvBytes, Machine, ModelTensor, ModelTensors, Plan, PlanLevers, Role,
};

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
    /// the most one layer keeps on the card, the card's free bytes when the
    /// census read them, and its device. `why` names what made a run with
    /// `--place` unset take this placed plan.
    pub fn record(&self, plan: &Plan<'_>, why: Option<&str>) -> Record {
        let card = &plan.cards[0];
        let r = Record::new(&record::PLAN38)
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
            );
        let r = match self.machine.cards.first().and_then(|c| c.free_bytes) {
            Some(free) => r.u("card_free", free),
            None => r,
        };
        let r = match why {
            Some(w) => r.w("why", w),
            None => r,
        };
        r.csv("devices", record::plan_devices(plan.machine))
            .w("cuda_order", record::cuda_order())
    }
}

/// The device bytes of the whole model of `inputs` on one card: every
/// tensor its role puts on the card, in the format it loads it in — the
/// file's own expert stacks whole, and none of the roles a whole load keeps
/// off the card (a hash table's, an engram table's). The need of today's
/// unplaced load before its KV, context, scratch and margin; a tensor no
/// card format loads gives `None`.
fn whole_card_bytes(model: &ModelTensors) -> Option<u64> {
    let mut sum = 0u64;
    let on_card = |t: &&ModelTensor| {
        !matches!(
            t.role,
            Role::Unused | Role::Unread | Role::HashTable | Role::EngramTable
        )
    };
    for t in model.tensors.iter().filter(on_card) {
        let format = CardFormat::of_role(t.ty, t.role)?;
        let rows = t.dims.iter().skip(1).product::<u64>();
        sum = sum.checked_add(format.resident_bytes(t.ty, *t.dims.first()?, rows)?)?;
    }
    Some(sum)
}

/// The KV bytes of `q`'s model at `ctx` positions, its own layout's.
fn kv_bytes(q: &PlaceQ3, ctx: u64) -> u64 {
    let layers = q.machine.cards[0].layers.clone();
    layers
        .map(|l| match &q.inputs {
            Inputs::Qwen3(i) => i.kv.layer_bytes(l, ctx),
            Inputs::Qwen35(i) => i.kv.layer_bytes(l, ctx),
        })
        .sum()
}

/// The model of `q`'s file as `inputs` holds it.
fn inputs_model(q: &PlaceQ3) -> &ModelTensors {
    match &q.inputs {
        Inputs::Qwen3(i) => &i.model,
        Inputs::Qwen35(i) => &i.model,
    }
}

/// Whether the whole model fits what device 0 — today's unplaced load's
/// card — had free at plan time: the whole-card bytes plus the KV, context,
/// scratch and margin of the machine the planner lays out for the file,
/// against the card's free-capped bytes (the margin covers the whole load's
/// own arena, which the placed machine's scratch understates). A model that
/// does not fit and has no routed expert to move to the host tier is
/// refused by name, the card, its free and usable bytes, the need and the
/// holders the census named.
fn whole_fits_free(q: &PlaceQ3, ctx: u64) -> Result<bool, GateError> {
    let card = &q.machine.cards[0];
    let model = inputs_model(q);
    let whole = whole_card_bytes(model)
        .ok_or("a tensor of this file has no card format, and the whole load takes the card")?;
    let need = whole
        .checked_add(kv_bytes(q, ctx))
        .and_then(|n| n.checked_add(card.set_aside_bytes()))
        .and_then(|n| n.checked_add(card.margin_bytes))
        .ok_or("the whole load's need passes u64 bytes")?;
    if need <= card.planned_bytes(None) {
        return Ok(true);
    }
    if !model.tensors.iter().any(|t| t.role == Role::RoutedExperts) {
        let free = card.free_bytes.map_or_else(
            || "no free reading".to_string(),
            |f| {
                format!(
                    "{f} B free of its usable {} B at plan time",
                    card.usable_bytes
                )
            },
        );
        return Err(format!(
            "the whole model needs {need} B on {}, which had {free}{}: a dense backbone has no \
             routed experts to move to the host tier",
            card.name,
            card.held_by
                .map_or(String::new(), |h| format!(" (held by {h})")),
        )
        .into());
    }
    Ok(false)
}

/// What a run with `--place` unset loads, decided before any load: the
/// whole model on device 0 (`Whole` — no plan, no record, today's load
/// byte for byte) while that fits the card's free bytes
/// ([`whole_fits_free`]); else the placed plan on `a`'s card (`Placed`).
/// A file with no routed experts that does not fit is refused by name.
pub fn unplaced_qwen3(file: &Split, ctx: usize) -> Result<Unplaced, GateError> {
    let probe = PlaceQ3::qwen3(file, Place::parse("cuda0")?, ctx)?;
    if whole_fits_free(&probe, u64::try_from(ctx)?)? {
        return Ok(Unplaced::Whole);
    }
    Ok(Unplaced::Placed(Box::new(PlaceQ3::qwen3(
        file,
        Place::A.on_host()?,
        ctx,
    )?)))
}

/// What a run with `--place` unset loads, as [`unplaced_qwen3`] decides it,
/// of a qwen35moe file.
pub fn unplaced_qwen35(file: &Split, o: &Open35) -> Result<Unplaced, GateError> {
    let probe = PlaceQ3::qwen35(file, Place::parse("cuda0")?, *o)?;
    if whole_fits_free(&probe, u64::try_from(o.ctx)?)? {
        return Ok(Unplaced::Whole);
    }
    Ok(Unplaced::Placed(Box::new(PlaceQ3::qwen35(
        file,
        Place::A.on_host()?,
        *o,
    )?)))
}

/// The load a run with `--place` unset takes ([`unplaced_qwen3`]): today's
/// whole model on device 0, or the placed plan on `a`'s card.
pub enum Unplaced {
    /// The whole-card load on device 0, as before the free-bytes rule.
    Whole,
    /// The placed plan on `a`'s card: its `plan` record prints with
    /// `why=whole_does_not_fit` before the open.
    Placed(Box<PlaceQ3>),
}

/// What a run with `--place` unset loads: today's whole model on device 0,
/// byte for byte — no plan, no record — while that fits the card's free
/// bytes ([`whole_fits_free`]); else the placed plan on `a`'s card, its
/// `plan` record with `why=whole_does_not_fit` handed to `emit` before the
/// open, the host set as `levers` asks.
pub fn open_unplaced_qwen3(
    file: Split,
    ctx: usize,
    opts: OpenOpts,
    levers: &Levers,
    emit: fn(Record),
) -> Result<Qwen3moeModel, GateError> {
    match unplaced_qwen3(&file, ctx)? {
        Unplaced::Whole => Ok(Qwen3moeModel::open(Gpu::new()?, file, opts)?),
        Unplaced::Placed(q) => {
            let plan = q.plan(ctx, &PlanLevers::from_levers(levers)?)?;
            emit(q.record(&plan, Some(WHY_NOT_WHOLE)));
            open_qwen3(file, &plan, opts, levers.host())
        }
    }
}

/// What a run with `--place` unset loads, as [`open_unplaced_qwen3`] loads
/// it, of a qwen35moe file.
pub fn open_unplaced_qwen35(
    file: Split,
    o: Open35,
    levers: &Levers,
    emit: fn(Record),
) -> Result<Qwen35moeModel, GateError> {
    match unplaced_qwen35(&file, &o)? {
        Unplaced::Whole => Ok(Qwen35moeModel::open(Gpu::new()?, file, o)?),
        Unplaced::Placed(q) => {
            let plan = q.plan(o.ctx, &PlanLevers::from_levers(levers)?)?;
            emit(q.record(&plan, Some(WHY_NOT_WHOLE)));
            open_qwen35(file, &plan, o, levers.host())
        }
    }
}

/// The `why` of a placed plan a run with `--place` unset took because its
/// whole file did not fit the card's free bytes.
pub const WHY_NOT_WHOLE: &str = "whole_does_not_fit";

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
