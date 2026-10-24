//! The tiered seats' run: one owner of the flags, the placement, the context, the plan's records,
//! the engine thread and the bind, for a model whose file one architecture names and whose routed
//! experts the plan keeps on the card and on the host. A seat gives its family's facts — the
//! architectures it accepts, its plan inputs, its own flags, its `--parallel` rule, its residency
//! machine, its think tags and its load (on the engine thread, with its records).
//!
//! [`solar2`](super::solar2) and [`mimo2`](super::mimo2) are the first two seats here. A seat
//! whose body is opened whole on the card ([`nemotron`](super::nemotron)) plans a placed load of
//! its own and keeps its run.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;

use bloomery_gpu::host::swap::Residency;
use bloomery_gpu_gates::beside::{self, Beside};
use bloomery_gpu_gates::bind::{
    ChatSurface, Pass, Seat, SeatEngine, SeatFlags, Vocab, arm_screen, model_props, parallel_line,
    placement_of, slots_given, slots_of,
};
use bloomery_gpu_gates::generate::{Lane, Place, choose_ctx, lane_at};
use bloomery_gpu_gates::need;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::side_seat;
use bloomery_gpu_gates::{GateError, gpu_census, ref_model_path};
use bloomery_levers::{HostCfg, Levers, ResidencyPick};
use gguf::Split;
use model::arch::Arch;
use model::arch::place_hybrid::PlanInputs;
use model::placement::ctx_ask::{DEFAULT_SLOTS, probe_of};
use model::placement::workstation::TierBatchBytes;
use model::placement::{PlacementError, Plan, PlanLevers};
use serve::reasoning::ThinkTags;
use serve::{EngineProps, ServeError};
use tokenizer::Tokenizer;

/// What the common run hands the engine thread: the placement and the plan's levers, the
/// context, the file, the side model's reserve, the plan's residency and the seat's own flags.
pub struct Open<F> {
    /// The planned card or placement word.
    pub place: Place,
    /// The lane's rate probed before the plan, and the card it ran on: the `lane` line beside
    /// the `load` record. `None` under `--plan`.
    pub lane: Option<Lane>,
    /// A slot's positions.
    pub ctx: usize,
    /// The resident slots the plan counted; the body of a seat that holds one sequence reads 1.
    pub slots: NonZeroUsize,
    pub path: PathBuf,
    pub plan: PlanLevers,
    pub host: HostCfg,
    /// `BLOOMERY_PIN_MAIN`.
    pub pin_main: bool,
    /// The batches a prompt group runs layer by layer (`BLOOMERY_PREFILL_GROUP`).
    pub group: usize,
    /// The side model's reserve on the stage card, which the plan counted.
    pub embedding: Option<u64>,
    /// The expert tier's prompt-batch bytes the plan reserved on the placement's tier card
    /// ([`Tiered::tier_batch`]); `None` under a placement with no tier card.
    pub tier_batch: Option<TierBatchBytes>,
    /// The residency the plan chose, `Off` where the body keeps the experts the plan placed.
    pub residency: Residency,
    /// The seat's flags beside the common ones ([`SeatFlags`]).
    pub flags: F,
}

/// A serving seat whose file one architecture (or one set of them) names and whose routed experts
/// the plan keeps on the card and on the host: what its family owns, and the load that opens it.
pub trait Tiered: Seat + Sized + 'static {
    /// The family's hyperparameters: what its plan inputs are read for.
    type H;
    /// The flags this seat takes beside the common ones ([`SeatFlags`]).
    type Flags: Default + Send;

    /// The binary's name, its records' owner and the `note` line's.
    const NAME: &'static str;
    const USAGE: &'static str;

    /// The levers this seat acts on, parsed once at `main`.
    const ACTS_ON: &'static [&'static str];

    /// The alias a file that names none serves under.
    const ALIAS: &'static str;
    /// The seat word a foreign file's refusal names, and the files it serves.
    const FAMILY: &'static str;
    const SERVES: &'static str;

    /// `--parallel N` resident slots, which the plan counts: false where the body holds one
    /// sequence and `--parallel` above 1 is refused by name.
    const RESIDENT: bool;

    /// Whether `arch` is a file this seat serves.
    fn accepts(arch: Arch) -> bool;

    /// The file's plan inputs, its own `place::read`.
    fn read(split: &Split) -> Result<PlanInputs<Self::H>, PlacementError>;

    /// One of the seat's own flags (`SeatFlags::parse`'s `own`): `Ok(true)` took it.
    fn flags(flag: &str, value: &str, out: &mut Self::Flags) -> Result<bool, GateError>;

    /// The expert tier's prompt-batch bytes the plan reserves on `place`'s tier card, for the file
    /// `inputs` describes; `None` under a placement with no tier card. A body that serves no tier
    /// card reserves none (the default), and its plan refuses the card by name.
    fn tier_batch(
        place: Place,
        _inputs: &PlanInputs<Self::H>,
    ) -> Result<Option<TierBatchBytes>, GateError> {
        Ok((!place.tier_cards().is_empty()).then_some(TierBatchBytes {
            staging: 0,
            scratch: 0,
            host: 0,
            leg: None,
        }))
    }

    /// The think tags the model's description names, if the family splits its reasoning on them.
    fn think(_inputs: &PlanInputs<Self::H>) -> Result<Option<ThinkTags>, GateError> {
        Ok(None)
    }

    /// The residency machine's pick, before the plan's record: the lever and the plan's own rule.
    /// `None` where the body keeps the experts where the plan put them, which records nothing.
    /// The run prints the pick between the lane and the plan, then asks [`Tiered::settle`] for the
    /// load's word.
    /// Registers the record kinds the seat's server prints (`record::at_main` with its
    /// `record::BLOOMERY_SERVE_*` list), first thing after the levers.
    fn register();

    /// The seat's `listening` record, which the run fills once the port is bound.
    fn listening() -> Record;

    /// The residency the seat opens with, its `residency_lever` record and detail printed here.
    fn residency(
        _levers: &Levers,
        _flags: &Self::Flags,
        _place: Place,
        _plan: &Plan<'_>,
    ) -> Result<Option<ResidencyPick>, GateError> {
        Ok(None)
    }

    /// After the plan's record: the room check, the host's pool and the load's word (the residency
    /// the seat opens with).
    fn settle(_plan: &Plan<'_>, _pick: ResidencyPick) -> Result<(Residency, u64), GateError> {
        Ok((Residency::Off, 0))
    }

    /// The seat on the engine thread: the session of the file at `a.path` on `a.place`, the step
    /// captured, its slots resident where the family has them, the `load` record on stderr.
    fn open(a: Open<Self::Flags>) -> Result<Self, GateError>;
}

/// Loads the model, and the embedding model `side` names beside it, and serves until the listener
/// or the engine fails; `Ok` carries why the server ended.
pub fn run<S: Tiered>(args: &[String], side: &side_seat::Flags) -> Result<ServeError, GateError> {
    let levers = bloomery_levers::at_main(S::ACTS_ON)?;
    model::ops::set_host_lanes(levers.host_lanes());
    S::register();
    record::require(record::groups::tiered::RUN, "tiered::run");
    arm_screen(levers.screen());
    let mut flags = S::Flags::default();
    let a = SeatFlags::parse(args, S::USAGE, |f, v| S::flags(f, v, &mut flags))?;
    if !S::RESIDENT
        && let Some(n) = a.parallel.filter(|&n| n > 1)
    {
        return Err(format!(
            "--parallel {n}: the {} body holds one sequence and no resident slots; the seat \
             serves --parallel 1",
            S::FAMILY
        )
        .into());
    }
    let side = side.read()?;
    let path = ref_model_path()?;
    let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let first = split
        .shard(0)
        .ok_or_else(|| format!("{} opened with no shard", path.display()))?;
    let arch = Arch::detect(first).map_err(|e| format!("{}: {e}", path.display()))?;
    if !S::accepts(arch) {
        return Err(format!(
            "{} is a {} file; the {} seat serves {} files",
            path.display(),
            arch.name(),
            S::FAMILY,
            S::SERVES
        )
        .into());
    }
    let chat = ChatSurface::read(
        &path,
        split.value("tokenizer.chat_template"),
        split.value("general.name"),
        a.template_file.path(),
        S::ALIAS,
    )?;
    let inputs = S::read(&split)?;
    let think = S::think(&inputs)?;
    let plan_levers = PlanLevers::from_levers(&levers)?;
    // The slot count and what named it (`from` on the `parallel` line); a seat of one sequence
    // reads `--parallel` as the one slot it holds, with no note about a count it cannot offer.
    let (slots, from) = if S::RESIDENT {
        slots_given(a.parallel, a.ctx, DEFAULT_SLOTS)?
    } else {
        slots_of(a.parallel, a.ctx.is_some(), DEFAULT_SLOTS)?
    };
    // The common unset rule, once, on one census reading.
    let census = gpu_census::census()?;
    let chosen = Place::choose_untiered(a.place, &census, S::FAMILY)?;
    chosen.record().eprint();
    let place = chosen.place;
    // The side model's seat before the plan, which holds its bytes when it sits on the stage card.
    let side_seat = side.as_ref().map(|s| s.seat(place)).transpose()?;
    let embedding = side_seat.as_ref().and_then(|s| s.reserve);
    // The lane's rate, taken before the plan reads the room (`lane_of`).
    let lane = if a.plan_only {
        None
    } else {
        Some(lane_at(place)?)
    };
    let tier_batch = S::tier_batch(place, &inputs)?;
    let machine = beside::with_reserve(
        place.machine(None, tier_batch)?,
        Beside {
            image: None,
            embedding,
        },
    )(inputs.model.layers);
    let n = NonZeroUsize::new(slots).ok_or("--parallel 0: the server serves no slot")?;
    // The context, then the plan of every slot around it (`choose_ctx`, every seat's): a tier
    // card, a second card, a context the card cannot hold, a host room that cannot hold the host
    // experts and a budget that binds are refused here by name, before any load; the load plans
    // the same.
    let asked = choose_ctx(&split, (a.ctx, slots), (None, None), |c| {
        probe_of(inputs.plan_with_slots(&machine, c, &plan_levers, n))
            .map_err(|e| format!("--place {}: {e}", place.name()))
    })?;
    eprintln!("{}", asked.line());
    let slot_ctx = usize::try_from(asked.slot_ctx)?;
    let plan = asked.plan;
    let pick = S::residency(&levers, &flags, place, &plan)?;
    record::plan(place.name(), &machine, &plan).eprint();
    let (residency, extra) = match pick {
        Some(pick) => S::settle(&plan, pick)?,
        None => (Residency::Off, 0),
    };
    need::print(&plan, extra, 0, true)?;
    parallel_line(slots, slot_ctx, from, Pass::of(slots, true));
    if a.plan_only {
        // The records before the load are out; nothing was opened on a card.
        std::process::exit(0);
    }
    let props = EngineProps {
        model: Some(model_props(&split, &inputs.model)),
        placement: placement_of(S::NAME, &machine, &plan),
        ..EngineProps::default()
    };
    let card = machine.cards[0].name.clone();
    drop(plan);
    drop(split);
    let vocab = Arc::new(Vocab::new(Tokenizer::from_gguf(&path)?)?);
    let open = Open {
        place,
        lane,
        ctx: slot_ctx,
        slots: n,
        path: path.clone(),
        plan: plan_levers,
        host: levers.host(),
        pin_main: levers.pin_main(),
        group: levers.prefill_group(),
        embedding,
        tier_batch,
        residency,
        flags,
    };
    let engine = SeatEngine::spawn(move || S::open(open), slot_ctx, vocab, card, props, 0)?;
    let side = side_seat::Side::start(side.zip(side_seat))?;
    let server = a.bind(engine, chat, &path, slots, &levers, side)?;
    let server = match think {
        Some(tags) => server.with_think_tags(tags),
        None => server,
    };
    S::listening()
        .w("place", place.name())
        .u("ctx", slot_ctx)
        .u("slots", slots)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}
