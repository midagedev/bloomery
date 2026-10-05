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
//! what the card had — the whole-fit verdict (`placement::whole_need`) over
//! what the whole load asks for: its weights' granules, its cache, and the
//! program's own arena and the reserve its load keeps free past it — and
//! when it does not, the placed plan on `a`'s card. The serve seat's unset
//! `--ctx` defaults live here too ([`whole_ctx_qwen3`] for the whole load,
//! [`placed_ctx_qwen3`] for a placed one), and its resident slots open a
//! placed plan through [`open_qwen3_slots`]. Every census reading the module
//! takes and every probe its searches run is counted ([`reads`]).

use std::sync::atomic::{AtomicUsize, Ordering};

use bloomery_gpu::arch::qwen3moe::{Body, Body35, KvQ8, Open35, OpenOpts, Qwen35moeModel};
use bloomery_gpu::model::{GpuModel, Slots};
use bloomery_gpu::{Gpu, Qwen3moeModel};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::generate::Place;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_levers::{HostCfg, Levers};
use gguf::Split;
use model::arch::qwen3moe::place::{self as q3, card_routed};
use model::arch::qwen35moe::place as q35;
use model::placement::workstation::{CardSpec, DeviceInfo};
use model::placement::{
    self, KvBytes, Machine, ModelTensors, PlacementError, Plan, PlanLevers, Role, WholeLoad,
    WholeNeed,
};

/// The census readings this module took and the probes its context searches
/// ran, since the process began: a caller reads them around a search
/// ([`reads`], [`Reads::since`]), so a search that reads the census per
/// probe shows in its counts.
static CENSUSES: AtomicUsize = AtomicUsize::new(0);
static PROBES: AtomicUsize = AtomicUsize::new(0);

/// The module's census readings and search probes at one moment ([`reads`]).
#[allow(
    dead_code,
    reason = "the serve seat prints a search's counts; the CLI and the e2e gates include the planner without them"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reads {
    pub censuses: usize,
    pub probes: usize,
}

impl Reads {
    /// The readings and probes taken since `before`.
    #[allow(
        dead_code,
        reason = "the serve seat prints a search's counts; the CLI and the e2e gates include the planner without them"
    )]
    #[must_use]
    pub fn since(self, before: Reads) -> Reads {
        Reads {
            censuses: self.censuses - before.censuses,
            probes: self.probes - before.probes,
        }
    }
}

/// The census readings and search probes so far.
#[allow(
    dead_code,
    reason = "the serve seat prints a search's counts; the CLI and the e2e gates include the planner without them"
)]
#[must_use]
pub fn reads() -> Reads {
    Reads {
        censuses: CENSUSES.load(Ordering::Relaxed),
        probes: PROBES.load(Ordering::Relaxed),
    }
}

/// This process's census ([`bloomery_gpu_gates::gpu_census::census`]),
/// counted.
fn census() -> Result<Vec<DeviceInfo>, GateError> {
    CENSUSES.fetch_add(1, Ordering::Relaxed);
    bloomery_gpu_gates::gpu_census::census()
}

/// One probe of a context search, counted.
fn probed() {
    PROBES.fetch_add(1, Ordering::Relaxed);
}

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
    /// `place` resolved against `census`, refused by name when it names an
    /// expert tier card.
    fn resolve(place: Place, census: &[DeviceInfo]) -> Result<Place, GateError> {
        let place = place.on(census)?;
        place.serves("qwen3moe", 0)?;
        Ok(place)
    }

    /// A qwen3moe file's placed load at `ctx` positions on `place`'s card,
    /// its cache planes in the `kv` format ([`KvQ8`]), resolved against one
    /// census of this process's devices.
    pub fn qwen3(file: &Split, place: Place, ctx: usize, kv: KvQ8) -> Result<PlaceQ3, GateError> {
        PlaceQ3::qwen3_on(file, place, &census()?, ctx, kv)
    }

    /// [`PlaceQ3::qwen3`] resolved against `census`.
    fn qwen3_on(
        file: &Split,
        place: Place,
        census: &[DeviceInfo],
        ctx: usize,
        kv: KvQ8,
    ) -> Result<PlaceQ3, GateError> {
        let place = PlaceQ3::resolve(place, census)?;
        let mut inputs = q3::PlanInputs::read(file)?;
        if kv == KvQ8::Q8 {
            inputs.kv = inputs.kv.in_q8();
        }
        let arena = Qwen3moeModel::placed_arena_bytes(file, ctx)?;
        let machine = q3::machine(place.card_specs()?[0], inputs.hp.n_layer, arena);
        Ok(PlaceQ3 {
            place,
            arch: "qwen3moe",
            inputs: Inputs::Qwen3(inputs),
            machine,
        })
    }

    /// A qwen35moe file's placed load under `o` on `place`'s card, its
    /// attention planes in `o.kv`'s format, resolved against one census of
    /// this process's devices.
    pub fn qwen35(file: &Split, place: Place, o: Open35) -> Result<PlaceQ3, GateError> {
        PlaceQ3::qwen35_on(file, place, &census()?, o)
    }

    /// [`PlaceQ3::qwen35`] resolved against `census`.
    fn qwen35_on(
        file: &Split,
        place: Place,
        census: &[DeviceInfo],
        o: Open35,
    ) -> Result<PlaceQ3, GateError> {
        let place = PlaceQ3::resolve(place, census)?;
        let mut inputs = q35::PlanInputs::describe(file)?;
        if o.kv == KvQ8::Q8 {
            inputs.kv = inputs.kv.in_q8();
        }
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
        self.plan_over(&self.machine, ctx, levers)
    }

    /// [`Self::plan`] over a machine [`Self::machine_at`] relayed at another
    /// context.
    fn plan_over<'a>(
        &'a self,
        machine: &'a Machine,
        ctx: usize,
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, GateError> {
        let ctx = u64::try_from(ctx)?;
        Ok(match &self.inputs {
            Inputs::Qwen3(i) => i.plan(machine, ctx, levers)?,
            Inputs::Qwen35(i) => i.plan_rule(machine, ctx, levers, card_routed)?,
        })
    }

    /// The same placed load's machine at `ctx`: the place and the inputs this
    /// one resolved and read stay — the census and the file's headers a
    /// context search's every probe retook before — and the machine is relaid
    /// over the arena `ctx` sets, its only ctx-dependent term.
    fn machine_at(&self, arena: u64) -> Result<Machine, GateError> {
        Ok(q3::machine(
            self.place.card_specs()?[0],
            self.layers(),
            arena,
        ))
    }

    /// The card experts the plan over `machine` at `ctx` keeps: one probe of
    /// a placed `--ctx` search. `None` when no plan builds, or when the plan
    /// keeps every routed expert on the card — a load [`open_qwen3`] opens
    /// whole, its arena the whole load's — and the whole-fit verdict at
    /// `ctx` ([`Self::whole_at`], what the program holds past its cache as
    /// `load` gives it) does not take that load.
    fn card_experts(
        &self,
        machine: &Machine,
        ctx: usize,
        levers: &PlanLevers,
        load: impl FnOnce() -> Result<WholeLoad, String>,
    ) -> Result<Option<u64>, GateError> {
        let Ok(plan) = self.plan_over(machine, ctx, levers) else {
            return Ok(None);
        };
        if plan.host.experts == 0 && !self.whole_at(u64::try_from(ctx)?, load())?.fits() {
            return Ok(None);
        }
        Ok(Some(plan.cards[0].experts))
    }

    /// The file's layers.
    fn layers(&self) -> usize {
        match &self.inputs {
            Inputs::Qwen3(i) => i.hp.n_layer,
            Inputs::Qwen35(i) => i.hp.n_layer,
        }
    }

    /// The same placed load on `place`'s card, resolved against `census`:
    /// the inputs this one read stay, and the machine is relaid on that card
    /// over the arena this one's counts.
    fn moved_to(self, place: Place, census: &[DeviceInfo]) -> Result<PlaceQ3, GateError> {
        let place = PlaceQ3::resolve(place, census)?;
        let arena = q3::counted_arena_bytes(&self.machine.cards[0]);
        let machine = q3::machine(place.card_specs()?[0], self.layers(), arena);
        Ok(PlaceQ3 {
            place,
            machine,
            ..self
        })
    }

    /// The whole-fit verdict of this file on the card's device at `ctx`
    /// positions ([`whole_on`]), the cache in the planes' format this load
    /// counts, `load` what the program holds past it or its refusal.
    fn whole_at(&self, ctx: u64, load: Result<WholeLoad, String>) -> Result<Whole, GateError> {
        let layers = 0..self.layers();
        let kv = kv_bytes(&layers, &self.inputs, ctx);
        whole_on(
            self.place.card_specs()?[0],
            inputs_model(&self.inputs),
            layers.end,
            kv,
            load,
        )
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

/// The KV bytes of the model `inputs` holds at `ctx` positions over
/// `layers`, its own layout's.
fn kv_bytes(layers: &std::ops::Range<usize>, inputs: &Inputs, ctx: u64) -> u64 {
    layers
        .clone()
        .map(|l| match inputs {
            Inputs::Qwen3(i) => i.kv.layer_bytes(l, ctx),
            Inputs::Qwen35(i) => i.kv.layer_bytes(l, ctx),
        })
        .sum()
}

/// The model of `inputs`'s file as it holds it.
fn inputs_model(inputs: &Inputs) -> &ModelTensors {
    match inputs {
        Inputs::Qwen3(i) => &i.model,
        Inputs::Qwen35(i) => &i.model,
    }
}

/// The qwen3moe body's whole load. It allocates its ubatch arena with no
/// fit check, and the arena's bytes have no formula outside the
/// allocation, so the card keeps the expert rule's margin past the step
/// arenas in its place, as the placed machine does.
const QWEN3_WHOLE: WholeLoad = WholeLoad::Unchecked;

/// A qwen35moe file's whole load under `o`: its ubatch arena at `o.ubatch`
/// and the reserve its fit check keeps free past it
/// (`GpuModel::whole_load`); the load's own refusal of the file or of `o`,
/// by name.
fn qwen35_whole(file: &Split, o: Open35) -> Result<WholeLoad, String> {
    Qwen35moeModel::whole_load(file, o).map_err(|e| e.to_string())
}

/// The whole-fit verdict ([`whole_on`]).
enum Whole {
    /// The whole load fits what the card had.
    Fits(WholeNeed),
    /// It does not.
    Short(WholeNeed),
    /// The whole load refuses the file, the card and why: a tensor no card
    /// format loads, or the program's own read of the file.
    Refused(String),
}

impl Whole {
    /// Whether the whole load is the load.
    fn fits(&self) -> bool {
        matches!(self, Whole::Fits(_))
    }

    /// The verdict at `ctx` positions in one line: the card, the answer,
    /// the need term by term and what the card had.
    fn line(&self, ctx: usize) -> String {
        match self {
            Whole::Fits(n) | Whole::Short(n) => format!("whole fit at ctx {ctx}: {n}"),
            Whole::Refused(why) => format!("whole fit at ctx {ctx}: {why}"),
        }
    }

    /// The refusal of a whole load this verdict does not take, of a model
    /// with no routed expert to move to the host tier.
    fn dense_refusal(&self, ctx: usize) -> GateError {
        format!(
            "{}: a dense backbone has no routed experts to move to the host tier",
            self.line(ctx)
        )
        .into()
    }
}

/// The whole-fit verdict of a load of the whole of `model` — `layers`
/// layers, `kv` bytes of cache — on `spec`'s card: the card the plan lays
/// out for the program ([`q3::machine`], its step arenas' scratch) and what
/// the program holds past it (`load`, or the program's refusal), sized by
/// `placement::whole_need`, the verdict's one owner. A tensor no card
/// format loads is a refusal; any other planner error is the error.
fn whole_on(
    spec: CardSpec,
    model: &ModelTensors,
    layers: usize,
    kv: u64,
    load: Result<WholeLoad, String>,
) -> Result<Whole, GateError> {
    let refused = |why: &dyn std::fmt::Display| {
        Whole::Refused(format!("the whole load on {}: refused: {why}", spec.name))
    };
    let load = match load {
        Ok(l) => l,
        Err(why) => return Ok(refused(&why)),
    };
    let machine = q3::machine(spec, layers, 0);
    Ok(
        match placement::whole_need(model, &machine.cards[0], kv, load) {
            Ok(need) if need.fits() => Whole::Fits(need),
            Ok(need) => Whole::Short(need),
            Err(e @ PlacementError::NoCardFormat { .. }) => refused(&e),
            Err(e) => return Err(e.into()),
        },
    )
}

impl PlaceQ3 {
    /// The load a run with `--place` unset takes by `whole`, this probe's
    /// verdict at `ctx`, its line on stderr: the whole load when the verdict
    /// takes it, else this file's placed plan on `a`'s card, resolved against
    /// `census` — the probe's own reading. A model with no routed expert that
    /// the verdict does not take is refused by name with its terms.
    fn unplaced(
        self,
        census: &[DeviceInfo],
        ctx: usize,
        whole: &Whole,
    ) -> Result<Unplaced, GateError> {
        eprintln!("{}", whole.line(ctx));
        if whole.fits() {
            return Ok(Unplaced::Whole);
        }
        let routed = inputs_model(&self.inputs)
            .tensors
            .iter()
            .any(|t| t.role == Role::RoutedExperts);
        if !routed {
            return Err(whole.dense_refusal(ctx));
        }
        Ok(Unplaced::Placed(Box::new(self.moved_to(Place::A, census)?)))
    }
}

/// What a run with `--place` unset loads, decided before any load on one
/// census reading: the whole model on device 0 (`Whole` — no plan, no
/// record, today's load byte for byte) while the whole-fit verdict takes it
/// ([`whole_on`]: the qwen3moe body's whole load, [`QWEN3_WHOLE`]); else the
/// placed plan on `a`'s card (`Placed`). The verdict's line goes to stderr;
/// a file with no routed experts the verdict does not take is refused by
/// name.
pub fn unplaced_qwen3(file: &Split, ctx: usize, kv: KvQ8) -> Result<Unplaced, GateError> {
    let census = census()?;
    let probe = PlaceQ3::qwen3_on(file, Place::parse("cuda0")?, &census, ctx, kv)?;
    let whole = probe.whole_at(u64::try_from(ctx)?, Ok(QWEN3_WHOLE))?;
    probe.unplaced(&census, ctx, &whole)
}

/// The step the searched default context moves in ([`whole_ctx_qwen3`]):
/// a round power of two a cache row count stays readable in.
pub const CTX_GRAN: usize = 1024;

/// The file's trained context (`<arch>.context_length`), `None` when the
/// file states none.
pub fn trained_ctx(file: &Split) -> Option<usize> {
    file.arch_get_u64("context_length")
        .and_then(|v| usize::try_from(v).ok())
}

/// The `--ctx` a whole-card load defaults to when the flag is unset: the
/// file's trained context ([`trained_ctx`]) capped to the largest multiple
/// of [`CTX_GRAN`] at or above `floor` whose whole load the whole-fit
/// verdict takes on device 0 ([`whole_on`], one census reading for every
/// probe): a context whose load fits what the card had with its arena and
/// reserve counted. `None` when the file states no trained context or
/// nothing at `floor` fits — the caller keeps `floor`, and the load falls to
/// the placed plan, whose own default [`placed_ctx_qwen3`] searches.
#[allow(
    dead_code,
    reason = "the serve seat defaults its --ctx through these; the CLI and the e2e gates include the planner without them"
)]
pub fn whole_ctx_qwen3(file: &Split, floor: usize, kv: KvQ8) -> Result<Option<usize>, GateError> {
    let probe = PlaceQ3::qwen3(file, Place::parse("cuda0")?, floor, kv)?;
    let fits = |ctx: usize| {
        probed();
        Ok(probe.whole_at(u64::try_from(ctx)?, Ok(QWEN3_WHOLE))?.fits())
    };
    searched_ctx(file, floor, &fits)
}

/// [`whole_ctx_qwen3`] of a qwen35moe file, `o` giving its non-context terms.
#[allow(
    dead_code,
    reason = "the serve seat defaults its --ctx through these; the CLI and the e2e gates include the planner without them"
)]
pub fn whole_ctx_qwen35(
    file: &Split,
    o: &Open35,
    floor: usize,
) -> Result<Option<usize>, GateError> {
    let probe = PlaceQ3::qwen35(file, Place::parse("cuda0")?, *o)?;
    let fits = |ctx: usize| {
        probed();
        let mut at = *o;
        at.ctx = ctx;
        Ok(probe
            .whole_at(u64::try_from(ctx)?, qwen35_whole(file, at))?
            .fits())
    };
    searched_ctx(file, floor, &fits)
}

/// The `--ctx` a placed load defaults to when the flag is unset: the largest
/// multiple of [`CTX_GRAN`] at or above `floor`, capped to the trained
/// context, whose plan on `place`'s card under `levers` keeps the card
/// experts the floor's plan keeps — the solver's context-for-experts trade
/// never below the floor's split, so the search spends only the plan's own
/// headroom — and, when that plan keeps every routed expert on the card (a
/// load that opens whole), whose whole load the whole-fit verdict takes
/// ([`PlaceQ3::card_experts`]). `None` when the file states no trained
/// context or no plan at `floor` builds — the caller keeps `floor`, and the
/// load's own plan call names what refused it.
#[allow(
    dead_code,
    reason = "the serve seat defaults its --ctx through these; the CLI and the e2e gates include the planner without them"
)]
pub fn placed_ctx_qwen3(
    file: &Split,
    place: Place,
    floor: usize,
    levers: &PlanLevers,
    kv: KvQ8,
) -> Result<Option<usize>, GateError> {
    let probe = PlaceQ3::qwen3(file, place, floor, kv)?;
    let experts = |ctx: usize| {
        probed();
        let machine = probe.machine_at(Qwen3moeModel::placed_arena_bytes(file, ctx)?)?;
        probe.card_experts(&machine, ctx, levers, || Ok(QWEN3_WHOLE))
    };
    searched_placed_ctx(file, floor, &experts)
}

/// [`placed_ctx_qwen3`] of a qwen35moe file, `o` giving its non-context terms.
#[allow(
    dead_code,
    reason = "the serve seat defaults its --ctx through these; the CLI and the e2e gates include the planner without them"
)]
pub fn placed_ctx_qwen35(
    file: &Split,
    o: &Open35,
    place: Place,
    floor: usize,
    levers: &PlanLevers,
) -> Result<Option<usize>, GateError> {
    let probe = PlaceQ3::qwen35(file, place, *o)?;
    let experts = |ctx: usize| {
        probed();
        let mut at = *o;
        at.ctx = ctx;
        let machine = probe.machine_at(Qwen35moeModel::placed_arena_bytes(file, at)?)?;
        probe.card_experts(&machine, ctx, levers, || qwen35_whole(file, at))
    };
    searched_placed_ctx(file, floor, &experts)
}

/// The search both `placed_ctx_*` run: [`searched_ctx`] over the plan's own
/// expert split — a context whose plan keeps the floor's card experts fits,
/// one whose plan cannot build (the card is held, a budget binds), that
/// drops below the split, or whose plan keeps every expert while its whole
/// load does not fit ([`PlaceQ3::card_experts`]) does not. Monotone in the
/// context: the KV term grows with it, the expert rule's budget falls, and
/// the rule's fill is a prefix through the allocator's granules, so more
/// budget never holds fewer experts; the whole need grows with the cache
/// against a fixed reading. One census and one header read serve the whole search
/// ([`PlaceQ3::machine_at`]), so the reading cannot move between the
/// probes the premise rides on.
#[allow(
    dead_code,
    reason = "the serve seat defaults its --ctx through these; the CLI and the e2e gates include the planner without them"
)]
fn searched_placed_ctx(
    file: &Split,
    floor: usize,
    card_experts: &dyn Fn(usize) -> Result<Option<u64>, GateError>,
) -> Result<Option<usize>, GateError> {
    let Some(at_floor) = card_experts(floor)? else {
        return Ok(None);
    };
    let fits = |ctx: usize| Ok(card_experts(ctx)?.is_some_and(|e| e >= at_floor));
    searched_ctx(file, floor, &fits)
}

/// The search both `whole_ctx_*` run: the largest `floor + k·CTX_GRAN` that
/// `fits` takes, `Some(trained)` when the trained context itself fits.
#[allow(
    dead_code,
    reason = "the serve seat defaults its --ctx through these; the CLI and the e2e gates include the planner without them"
)]
fn searched_ctx(
    file: &Split,
    floor: usize,
    fits: &dyn Fn(usize) -> Result<bool, GateError>,
) -> Result<Option<usize>, GateError> {
    let Some(trained) = trained_ctx(file) else {
        return Ok(None);
    };
    if trained <= floor {
        return Ok(Some(trained));
    }
    if !fits(floor)? {
        return Ok(None);
    }
    let mut lo = floor;
    let mut hi = trained;
    while hi - lo > CTX_GRAN {
        let mid = lo + (hi - lo) / 2 / CTX_GRAN * CTX_GRAN;
        if mid == lo {
            break;
        }
        if fits(mid)? {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(Some(lo))
}

/// What a run with `--place` unset loads, as [`unplaced_qwen3`] decides it,
/// of a qwen35moe file under `o`: the whole-fit verdict over the program's
/// own arena at `o.ubatch` and its reserve ([`qwen35_whole`]).
pub fn unplaced_qwen35(file: &Split, o: &Open35) -> Result<Unplaced, GateError> {
    let census = census()?;
    let probe = PlaceQ3::qwen35_on(file, Place::parse("cuda0")?, &census, *o)?;
    let whole = probe.whole_at(u64::try_from(o.ctx)?, qwen35_whole(file, *o))?;
    probe.unplaced(&census, o.ctx, &whole)
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
/// byte for byte — no plan, no record — while the whole-fit verdict takes it
/// ([`unplaced_qwen3`]); else the placed plan on `a`'s card, its
/// `plan` record with `why=whole_does_not_fit` handed to `emit` before the
/// open, the host set as `levers` asks.
#[allow(
    dead_code,
    reason = "generate_qwen3moe opens its unset load through it; the serve seat decides its load at the total before the open"
)]
pub fn open_unplaced_qwen3(
    file: Split,
    ctx: usize,
    opts: OpenOpts,
    levers: &Levers,
    emit: fn(Record),
) -> Result<Qwen3moeModel, GateError> {
    match unplaced_qwen3(&file, ctx, opts.kv)? {
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

/// The Qwen3-30B model of `file` by `plan` ([`open_qwen3`]) for `slots`
/// resident sequences: `plan` made at the total context the slots split,
/// the model holding its one sequence at `opts.ctx` rows — a slot's share —
/// so the sequences the caller adds after it sit inside the cache the plan
/// counts. `slots · opts.ctx` past the plan's `ctx_max` is refused by name
/// before anything loads; the open is handed the plan with its `ctx_max` at
/// `opts.ctx` (the rows the caches hold, which the open checks and the model
/// steps against), every other term the plan's own at the total; once open,
/// the `slots` sequences' cache — the body's own sequence bytes, `slots`
/// times — past the plan's KV term is refused by name, and both stand on one
/// stderr line. One slot is [`open_qwen3`] itself.
#[allow(
    dead_code,
    reason = "the serve seat's resident slots; the CLI and the e2e gates load one sequence"
)]
pub fn open_qwen3_slots(
    file: Split,
    plan: &Plan<'_>,
    slots: usize,
    opts: OpenOpts,
    host: HostCfg,
) -> Result<Qwen3moeModel, GateError> {
    const WHAT: &str = "placed slots";
    if slots <= 1 {
        return open_qwen3(file, plan, opts, host);
    }
    let rows = u64::try_from(slots.saturating_mul(opts.ctx))?;
    if rows > plan.ctx_max {
        return Err(format!(
            "{WHAT}: {slots} sequences of {} rows hold {rows} positions, past the {} the plan \
             counts",
            opts.ctx, plan.ctx_max
        )
        .into());
    }
    let at = Plan {
        ctx_max: u64::try_from(opts.ctx)?,
        ..plan.clone()
    };
    let m = open_qwen3(file, &at, opts, host)?;
    let held = u64::try_from(m.body(WHAT)?.seq_bytes())?.saturating_mul(u64::try_from(slots)?);
    let counted = plan
        .cards
        .first()
        .ok_or_else(|| format!("{WHAT}: a plan of no card"))?
        .kv_bytes;
    let line = format!(
        "{WHAT}: {slots} sequences of {} rows hold {held} B of cache; the plan at {} positions \
         counts {counted} B",
        opts.ctx, plan.ctx_max
    );
    if held > counted {
        return Err(format!("{line}: past its count").into());
    }
    eprintln!("{line}");
    Ok(m)
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
