//! The tiered seats' run: one owner of the flags, the placement, the context, the plan's records,
//! the engine thread and the bind, for a model whose file one architecture names and whose routed
//! experts the plan keeps on the card and on the host. A seat gives its family's facts — the
//! architectures it accepts, its plan inputs, its own flags, its `--parallel` rule, its residency
//! machine, its think tags and its load (on the engine thread, with its records). A seat of
//! resident slots takes `--parallel auto`, counted over the plan once the placement is chosen
//! at one slot (`bind::AutoSlots::count`).
//!
//! [`mimo2`](super::mimo2), [`minimaxm3`](super::minimaxm3) and [`musespark`](super::musespark) run here.
//! [`solar2`](super::solar2) keeps the hybrid seat's run. A seat
//! whose body is opened whole on the card ([`nemotron`](super::nemotron)) plans a placed load of
//! its own and keeps its run. A seat may name a prompt-cache budget
//! ([`Tiered::cache_budget`]; [`CacheFlag`] is the `--cache-ram` flag and the budget over the
//! plan for a seat that saves states in host RAM, whose copy lane the open makes of
//! [`Open::cache_ram`] bytes); the default is 0 and prints nothing.
//!
//! A model path is a GGUF split, or a checkpoint folder where the seat reads one
//! ([`Tiered::read_model`]): the common run opens the folder once, plans by its located header,
//! serves its tokenizer and hands the header and the part source to the engine thread
//! ([`Folder`]). A folder's residency is a file's ([`Tiered::residency`]): the machine stages a
//! located folder's routed experts from its parts.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bloomery_gpu::host::swap::Residency;
use bloomery_gpu_gates::beside::{self, Beside};
use bloomery_gpu_gates::bind::{
    CacheRam, ChatSurface, Pass, Seat, SeatEngine, SeatFlags, Vocab, arm_screen, eprint_noted,
    model_props, parallel_line, placement_of, slots_given, slots_of,
};
use bloomery_gpu_gates::generate::{Lane, Place, TierRule, choose_ctx, lane_at};
use bloomery_gpu_gates::need;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::residency;
use bloomery_gpu_gates::residency38::{
    checkpoints_reserved, churn_bytes, residency_room, residency_set,
};
use bloomery_gpu_gates::side_seat;
use bloomery_gpu_gates::{GateError, gpu_census, ref_model_path};
use bloomery_levers::{
    ChurnHosting, HostCfg, Levers, ResidencyAt, ResidencyPick, ResidencyWhy, UnhostedPool,
};
use gguf::Split;
use model::arch::Arch;
use model::arch::place_hybrid::{PlaceError, PlanInputs};
use model::placement::ctx_ask::{Chosen, DEFAULT_SLOTS, probe_of};
use model::placement::workstation::{self, HostNeed, TierBatchBytes};
use model::placement::{Machine, PlacementError, Plan, PlanLevers};
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
    /// The machine the run's plan was made on, with every reserve the run set on it (the side
    /// model's, a folder's part reads, the expert stream's staging): the open plans on it.
    pub machine: Machine,
    pub path: PathBuf,
    pub plan: PlanLevers,
    pub host: HostCfg,
    /// `BLOOMERY_PIN_MAIN`.
    pub pin_main: bool,
    /// The batches a prompt group runs layer by layer (`BLOOMERY_PREFILL_GROUP`).
    pub group: usize,
    /// The residency the plan chose, `Off` where the body keeps the experts the plan placed.
    pub residency: Residency,
    /// The prompt cache's byte budget ([`Tiered::cache_budget`]): the pinned staging a seat's save
    /// lane is made of at the open; 0, the cache off, makes none.
    pub cache_ram: u64,
    /// `BLOOMERY_XSTREAM` as set ([`Levers::xstream`]), `None` unset: the seat whose prompt calls
    /// stream host experts resolves it by the common rule (`shared/xstream_rule.rs`).
    pub xstream: Option<&'static str>,
    /// The plan kept the expert stream's ring out of the expert budget ([`Tiered::ring`]): the
    /// open plans the same inputs ([`Tiered::with_ring`]).
    pub ring: bool,
    /// The checkpoint folder `path` is, opened once by the common run ([`Tiered::read_model`]);
    /// `None` for a file, which the engine thread opens again.
    pub folder: Option<Folder>,
    /// The seat's flags beside the common ones ([`SeatFlags`]).
    pub flags: F,
}

/// What a checkpoint folder's load reads card tensors through (`model_hf::folder::Opened::source`).
/// A build without the `checkpoint` feature opens no folder, so it has none.
#[cfg(feature = "checkpoint")]
pub type FolderSource = model_hf::part::Checkpoint;
#[cfg(not(feature = "checkpoint"))]
pub type FolderSource = ();

/// A checkpoint folder as the engine thread loads it: the located header the plan was made from and
/// the part source its card tensors are read through.
pub struct Folder {
    pub header: Split,
    pub source: FolderSource,
}

/// `--cache-ram`, the flag of a seat that keeps saved sequence states in host RAM: the budget in MiB
/// as given, unset the default ([`CacheRam`]).
#[derive(Default)]
pub struct CacheFlag {
    ram: Option<u64>,
}

impl CacheFlag {
    /// `flag` and `value` as the seat's own flag ([`Tiered::flags`]): `Ok(true)` took `--cache-ram`.
    pub fn take(&mut self, flag: &str, value: &str) -> Result<bool, GateError> {
        if flag != "--cache-ram" {
            return Ok(false);
        }
        self.ram = Some(CacheRam::parse_mib(flag, value)?);
        Ok(true)
    }

    /// The budget in bytes ([`Tiered::cache_budget`]), its `cache` line printed: the flag as given,
    /// else the default over the plan's host need and the residency's `pool`; a default under one
    /// saved state is off. `state` is the host bytes a saved state of that many positions holds. A
    /// budget that holds no full session at the slot's context `slot_ctx` stays on for the shorter
    /// ones and says so.
    pub fn budget(
        &self,
        plan: &Plan<'_>,
        pool: u64,
        slot_ctx: usize,
        state: impl Fn(usize) -> u64,
    ) -> Result<u64, GateError> {
        let cache = CacheRam::of(
            self.ram,
            HostNeed::of(plan, 0).bytes(),
            pool,
            checkpoints_reserved(plan.machine),
        )?
        .holding(state(1))
        .short_of(slot_ctx, state(slot_ctx));
        eprint_noted(&cache.line());
        Ok(cache.ram)
    }
}

/// The model at a path as the common run reads it ([`Tiered::read_model`]): the header it plans
/// by, which is a split's first shard or a folder's located header, and, for a folder, its
/// tokenizer and its part source.
pub struct Read {
    pub split: Split,
    /// The folder's tokenizer; `None` for a file, whose vocabulary is built after the plan.
    pub tokenizer: Option<Tokenizer>,
    /// The folder's part source; `None` for a file.
    pub source: Option<FolderSource>,
}

/// The GGUF split at `path`, which the engine thread opens again for its load.
pub fn read_file(path: &Path) -> Result<Read, GateError> {
    let split = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    Ok(Read {
        split,
        tokenizer: None,
        source: None,
    })
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

    /// `--parallel N` resident slots, which the plan counts, and `--parallel auto`: false where
    /// the body holds one sequence and `--parallel` above 1 or `auto` is refused by name.
    const RESIDENT: bool;

    /// Whether a round of several slots is one pass of the body ([`Pass::of`]). The default is
    /// one pass: a load of one slot prints `pass=one` either way, and a seat that already runs
    /// one pass keeps the word. A seat whose body serves the slots by a select and a step
    /// overrides false, and a load of several slots prints `pass=turns`.
    const ONE_PASS: bool = true;

    /// Whether the body serves its expert tier card beside several resident slots. The default
    /// does; a body whose pass of several slots runs their rows as the columns of one row, which
    /// the tier's handoff does not write, overrides false: with several slots an unset `--place`
    /// does not offer the tier card, a set placement with one is refused by name before the
    /// plan, and `--parallel auto` on a placement with one serves one slot.
    const TIER_SLOTS: bool = true;

    /// The pool support of the body this seat loads.
    const UNHOSTED_POOL: UnhostedPool = UnhostedPool::NotCarried;

    /// Whether `arch` is a file this seat serves.
    fn accepts(arch: Arch) -> bool;

    /// The model at `path`. The default is a GGUF split and refuses a directory by name; a seat
    /// that reads a checkpoint folder overrides it, opening the folder once ([`Read`]).
    fn read_model(path: &Path) -> Result<Read, GateError> {
        if path.is_dir() {
            return Err(format!(
                "{} is a directory: the {} seat serves {} files",
                path.display(),
                Self::FAMILY,
                Self::SERVES
            )
            .into());
        }
        read_file(path)
    }

    /// Whether this seat's explicit flags select the one-token prompt feed.
    fn prefill_steps(_flags: &Self::Flags) -> bool {
        false
    }

    /// The file's plan inputs, its own `place::read`.
    fn read(split: &Split) -> Result<PlanInputs<Self::H>, PlacementError>;

    /// Price the selected prompt schedule before any context probe or allocation.
    /// A family whose schedule has fixed buffers keeps its header-derived inputs.
    fn configure(_inputs: &mut PlanInputs<Self::H>, _group: usize) -> Result<(), GateError> {
        Ok(())
    }

    /// One of the seat's own flags (`SeatFlags::parse`'s `own`): `Ok(true)` took it.
    fn flags(flag: &str, value: &str, out: &mut Self::Flags) -> Result<bool, GateError>;

    /// The machine the plan of a file of `layers` layers runs over on `place`, the run's, which
    /// the open plans on ([`Open::machine`]): the placement's ([`Place::machine`]: the token
    /// embedding on the host, a row gathered a token, and `tier_batch` on a tier card) by default. A body that reads its
    /// token embedding from a table on its card plans a placed load's machine instead
    /// ([`Place::machine_placed`]).
    fn machine(
        place: Place,
        tier_batch: Option<TierBatchBytes>,
        layers: usize,
    ) -> Result<Machine, GateError> {
        Ok(place.machine(None, tier_batch)?(layers))
    }

    /// The family's input to the common unset rule ([`Place::choose`]): the tier cards its body
    /// serves and the tier experts that keep the tier card. The default, [`TierRule::UNTIERED`],
    /// is a body that serves none: an unset `--place` runs `a`, the plan never asked.
    const TIER_RULE: TierRule = TierRule::UNTIERED;

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

    /// Whether the plan of a load on `place` keeps the expert stream's ring out of the expert budget
    /// ([`Tiered::with_ring`]): the body's prompt calls stream host experts there (`BLOOMERY_XSTREAM`
    /// by the common rule, `shared/xstream_rule.rs`). Asked before the plan; the run keeps the ring
    /// only where the rule would start the stream on the plan (a residency pick other than `off`,
    /// and, unset, host experts to stream and room for the ring), and the open plans what the run
    /// kept ([`Open::ring`]). False by default: a body that streams none.
    fn ring(_levers: &Levers, _place: Place) -> bool {
        false
    }

    /// `inputs` for a plan that keeps the ring out of the expert budget; asked where [`Tiered::ring`]
    /// is true. A seat that answers true without this is refused by name.
    fn with_ring(_inputs: PlanInputs<Self::H>) -> Result<PlanInputs<Self::H>, GateError> {
        Err(format!(
            "the {} seat answered a stream ring and reserves none (Tiered::with_ring)",
            Self::FAMILY
        )
        .into())
    }

    /// The think tags the model's description names, if the family splits its reasoning on them.
    fn think(_inputs: &PlanInputs<Self::H>) -> Result<Option<ThinkTags>, GateError> {
        Ok(None)
    }

    /// Registers the record kinds the seat's server prints (`record::at_main_with_flags` with its
    /// `record::BLOOMERY_SERVE_*` list), before the run checks its arguments.
    fn register(args: &[String]);

    /// The seat's `listening` record, which the run fills once the port is bound.
    fn listening() -> Record;

    /// The residency the seat opens with. The method prints the `residency lever` record, and
    /// the plan-side detail beside it when the plan moved or refused the word. `None` where the
    /// body keeps the experts the plan placed, which records nothing. The default is the common
    /// machine ([`residency_pick`]): set as given, else the serving word on every placement but
    /// `gate`, for a body that hosts its churn pool and feeds no step prompt.
    fn residency(
        levers: &Levers,
        _flags: &Self::Flags,
        place: Place,
        plan: &Plan<'_>,
    ) -> Result<Option<ResidencyPick>, GateError> {
        // The hybrid bodies host every churn pool and feed the prompt in batches.
        residency_pick(levers, place, plan, false, Self::UNHOSTED_POOL).map(Some)
    }

    /// After the plan's record: the room check of a set word, the host pool's record under
    /// `mid`, and the load's word. The default is [`settle_pick`].
    fn settle(plan: &Plan<'_>, pick: ResidencyPick) -> Result<(Residency, u64), GateError> {
        settle_pick(plan, pick)
    }

    /// The prompt-cache budget in bytes, printed on the seat's `cache` line when the seat has one.
    /// The default is 0 and prints nothing: a seat with no saved state serves every request from
    /// the slot it holds. `pool` is the host bytes [`Tiered::settle`] counted beside the plan.
    fn cache_budget(
        _flags: &Self::Flags,
        _plan: &Plan<'_>,
        _pool: u64,
        _slot_ctx: usize,
        _inputs: &PlanInputs<Self::H>,
    ) -> Result<u64, GateError> {
        Ok(0)
    }

    /// The seat on the engine thread: the session of the file at `a.path` on `a.place`, the step
    /// captured, its slots resident where the family has them, the `load` record on stderr.
    fn open(a: Open<Self::Flags>) -> Result<Self, GateError>;
}

/// Checks an explicit residency against the body's pool support and prompt feed.
pub fn check_residency_feed(
    levers: &Levers,
    prefill_steps: bool,
    carry: UnhostedPool,
) -> Result<(), GateError> {
    if let Some(word) = levers.residency() {
        match carry {
            UnhostedPool::NotCarried => Residency::parse(word)?,
            UnhostedPool::Carried => Residency::parse_asking(word)?,
        };
        if word != "off" && prefill_steps {
            return Err(format!(
                "BLOOMERY_RESIDENCY={word} is refused beside --prefill steps (each prompt id would \
                 end a pass the rule counts)"
            )
            .into());
        }
    }
    Ok(())
}

/// `BLOOMERY_RESIDENCY` for a tiered seat, printed as the `residency lever` record: set, as
/// given; unset, the serving word on a serving placement (`place` other than `gate`), which
/// the plan keeps, moves or turns `off`. `prefill_steps` is the seat's step feed, which the
/// rule counts and which refuses a set word other than `off`. `carry` is the loading body's
/// `ChainBody::UNHOSTED_POOL`.
pub fn residency_pick(
    levers: &Levers,
    place: Place,
    plan: &Plan<'_>,
    prefill_steps: bool,
    carry: UnhostedPool,
) -> Result<ResidencyPick, GateError> {
    record::require(record::groups::tiered::LEVER, "tiered::residency_pick");
    let pick = levers.residency_at(ResidencyAt {
        serving_place: place != Place::Gate,
        check_finite: false,
        route_trace: false,
        prefill_steps,
    });
    check_residency_feed(levers, prefill_steps, carry)?;
    let pick = residency::at_plan(carry, plan, 0, pick)?;
    record::residency_lever(pick).eprint();
    if let Some(d) = pick.why.detail() {
        eprintln!("{d}");
    }
    Ok(pick)
}

/// After the plan's record: a set word's room check, the hosted churn pool's `residency host`
/// record under `mid`, and the load's word with the pool's host bytes.
pub fn settle_pick(plan: &Plan<'_>, pick: ResidencyPick) -> Result<(Residency, u64), GateError> {
    let residency = Residency::parse(pick.word)?;
    if pick.why == ResidencyWhy::Set {
        residency_room(plan, residency, pick.word)?;
    }
    residency_set(
        plan,
        0,
        residency,
        ChurnHosting::Hosted,
        pick.word,
        0,
        Record::eprint,
    )?;
    // The hosted churn pool is host bytes the load holds beside the plan.
    Ok((
        residency,
        churn_bytes(plan, 0, (residency, ChurnHosting::Hosted))?,
    ))
}

/// The routed experts `offer`'s plan holds on its expert tier cards, which the common unset rule
/// weighs ([`Place::choose`] by [`Tiered::TIER_RULE`]): one slot's plan at the asked context, or
/// at one position unset — a tier card holds no store, so its experts do not move with the
/// context.
fn tier_experts<S: Tiered>(
    offer: Place,
    inputs: &PlanInputs<S::H>,
    levers: &PlanLevers,
    ctx: Option<usize>,
) -> Result<u64, GateError> {
    let machine = S::machine(offer, S::tier_batch(offer, inputs)?, inputs.model.layers)?;
    let ctx = ctx.map_or(Ok(1), u64::try_from)?;
    let plan = inputs.plan(&machine, ctx, levers)?;
    Ok(plan.tier_n_l.iter().flatten().sum())
}

/// `place` beside `slots` resident slots, refused by name when it names a tier card and the body
/// serves its tier beside one sequence ([`Tiered::TIER_SLOTS`]).
fn refuse_tier_slots<S: Tiered>(place: Place, slots: usize) -> Result<(), GateError> {
    if S::TIER_SLOTS || slots <= 1 || place.tier_cards().is_empty() {
        return Ok(());
    }
    Err(format!(
        "--place {} with --parallel {slots}: the {} body serves its expert tier card beside one \
         sequence; serve --parallel 1 there, or a placement with no tier card",
        place.name(),
        S::FAMILY
    )
    .into())
}

/// The context and the plan of every slot around it over `inputs` ([`choose_ctx`], every seat's):
/// `asked` the context and slots the flags asked, then the machine, the plan's levers, the slots
/// the plan counts and the placement its refusals name.
fn plan_at<'a, H>(
    split: &Split,
    asked: (Option<usize>, usize),
    inputs: &'a PlanInputs<H>,
    (machine, levers, slots, place): (&'a Machine, &PlanLevers, NonZeroUsize, Place),
) -> Result<Chosen<Plan<'a>>, GateError> {
    choose_ctx(split, asked, (None, None), |c| {
        probe_of(inputs.plan_with_slots(machine, c, levers, slots))
            .map_err(|e| format!("--place {}: {e}", place.name()))
    })
}

/// The plan a streaming seat loads on its staged machine, with whether it keeps the ring: the
/// ring's plan `with_ring` where `set` (the lever asks the stream) or the plain plan `hosts`
/// experts, `None` (the plain plan) where it holds every expert on the card, the ring's plan
/// refused or not: nothing streams there. Where the ring does not fit, the lever is unset and the
/// plain plan hosts experts, the plan `without` it on the same machine, its refusal the seat's:
/// the open's stream rule may still start the stream from the card's free bytes, and the plain
/// plan holds no staging.
fn ringed<P>(
    set: bool,
    hosts: bool,
    with_ring: impl FnOnce() -> Result<P, GateError>,
    without: impl FnOnce() -> Result<P, GateError>,
) -> Result<Option<(P, bool)>, GateError> {
    match with_ring() {
        Ok(p) if set || hosts => Ok(Some((p, true))),
        Ok(_) => Ok(None),
        Err(_) if !set && !hosts => Ok(None),
        Err(e) if !set => {
            eprintln!("xstream ring: the plan cannot hold it, so it plans without: {e}");
            Ok(Some((without()?, false)))
        }
        Err(e) => Err(e),
    }
}

/// The plan a streaming seat loads, whether it keeps the ring, and the residency pick it opens
/// with, `pick` of the plan (the seat's `Tiered::residency`, asked once: it prints its record):
/// the staged plan of `attempt` ([`ringed`]), or the `plain` plan where the attempt keeps none.
/// A pick that streams nothing (`off`, or none) ends the stream rule at `off`: the load runs the
/// plain plan, which holds no staging, so the card keeps the ring's bytes for experts and context
/// and the host for its set; the pick holds there, for `off` reads no room. A refused attempt under
/// an unset lever (`set` false) is the seat's refusal only where the plain plan's pick streams.
fn picked<P, K>(
    attempt: Result<Option<(P, bool)>, GateError>,
    set: bool,
    plain: P,
    pick: impl Fn(&P) -> Result<Option<K>, GateError>,
    off: impl Fn(&K) -> bool,
) -> Result<(P, bool, Option<K>), GateError> {
    let streams = |k: &Option<K>| k.as_ref().is_some_and(|k| !off(k));
    let (p, ring) = match attempt {
        Ok(Some(staged)) => staged,
        Ok(None) => {
            let k = pick(&plain)?;
            return Ok((plain, false, k));
        }
        Err(e) if !set => {
            let k = pick(&plain)?;
            return if streams(&k) {
                Err(e)
            } else {
                Ok((plain, false, k))
            };
        }
        Err(e) => return Err(e),
    };
    let k = pick(&p)?;
    Ok(if streams(&k) {
        (p, ring, k)
    } else {
        (plain, false, k)
    })
}

/// Selects the load's plan for both the slot probe and the final context choice. Staging
/// carries whether the plain plan hosts experts and the plans with and without the ring.
fn streamed<P, K>(
    set: bool,
    plain: P,
    staging: Option<(
        bool,
        impl FnOnce() -> Result<P, GateError>,
        impl FnOnce() -> Result<P, GateError>,
    )>,
    pick: impl Fn(&P) -> Result<Option<K>, GateError>,
    off: impl Fn(&K) -> bool,
) -> Result<(P, bool, Option<K>), GateError> {
    let attempt = match staging {
        Some((hosts, with_ring, without)) => ringed(set, hosts, with_ring, without),
        None => Ok(None),
    };
    picked(attempt, set, plain, pick, off)
}

/// The common flags of `args` and the seat's own; `--parallel` above one is refused by name on a
/// body that holds one sequence.
fn parse_args<S: Tiered>(args: &[String]) -> Result<(SeatFlags, S::Flags), GateError> {
    let mut flags = S::Flags::default();
    let a = SeatFlags::parse(args, S::USAGE, |f, v| S::flags(f, v, &mut flags))?;
    if !S::RESIDENT {
        a.one_sequence(S::FAMILY)?;
    }
    Ok((a, flags))
}

/// The levers (unless `read_levers` is false) and flags [`run`] reads first, checked before the
/// model is fetched ([`bloomery_gpu_gates::bind::check_seat`]).
pub fn check<S: Tiered>(
    args: &[String],
    read_levers: bool,
) -> Result<bloomery_gpu_gates::bind::CheckedSeat<(SeatFlags, S::Flags)>, GateError> {
    bloomery_gpu_gates::bind::check_seat(read_levers.then_some(S::ACTS_ON), args, |args, levers| {
        let a = parse_args::<S>(args)?;

        if let Some(place) = a.0.place {
            place.serves(S::FAMILY, S::TIER_RULE.tiers)?;
            refuse_tier_slots::<S>(place, a.0.parallel.unwrap_or(1))?;
        }
        if let Some(levers) = levers {
            check_residency_feed(levers, S::prefill_steps(&a.1), S::UNHOSTED_POOL)?;
        }
        Ok(a)
    })
}

/// Loads the model, and the embedding model `side` names beside it, and serves until the listener
/// or the engine fails; `Ok` carries why the server ended.
pub fn run<S: Tiered>(args: &[String], side: &side_seat::Flags) -> Result<ServeError, GateError> {
    S::register(args);
    let (a, levers) = check::<S>(args, true)?.run();
    model::ops::set_host_lanes(levers.host_lanes());
    record::require(record::groups::tiered::RUN, "tiered::run");
    arm_screen(levers.screen());
    let (a, flags) = a;
    let side = side.read()?;
    let path = ref_model_path()?;
    let Read {
        split,
        tokenizer,
        source,
    } = S::read_model(&path)?;
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
    let mut inputs = S::read(&split)?;
    S::configure(&mut inputs, levers.prefill_group())?;
    let think = S::think(&inputs)?;
    let plan_levers = PlanLevers::from_levers(&levers)?;
    // The slot count and what named it (`from` on the `parallel` line); a seat of one sequence
    // reads `--parallel` as the one slot it holds, with no note about a count it cannot offer.
    // Under `--parallel auto` the placement is chosen at one slot, and the count below.
    let (slots, from) = if S::RESIDENT {
        slots_given(a.parallel, a.ctx, a.auto.is_some(), DEFAULT_SLOTS)?
    } else {
        slots_of(a.parallel, a.ctx.is_some(), DEFAULT_SLOTS)?
    };
    // The common unset rule, once, on one census reading.
    let census = gpu_census::census()?;
    let chosen = if S::TIER_RULE.tiers == 0 {
        Place::choose_untiered(a.place, &census, S::FAMILY)?
    } else {
        Place::choose(a.place, &census, S::TIER_RULE, |offer| {
            refuse_tier_slots::<S>(offer, slots)?;
            tier_experts::<S>(offer, &inputs, &plan_levers, a.ctx)
        })?
    };
    refuse_tier_slots::<S>(chosen.place, slots)?;
    chosen.record().eprint();
    let place = chosen.place;
    // The plan's inputs on this placement where the seat streams, read again from the headers: the
    // stream's ring out of the expert budget. Whether the load keeps them is settled below.
    let ring_inputs = if S::ring(&levers, place) {
        Some(S::with_ring(S::read(&split)?)?)
    } else {
        None
    };
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
    let mut machine = S::machine(place, tier_batch, inputs.model.layers)?;
    beside::reserve_on(
        &mut machine,
        Beside {
            image: None,
            embedding,
        },
    );
    let n = NonZeroUsize::new(slots).ok_or("--parallel 0: the server serves no slot")?;
    // A folder's load reads its card's parts before its host set: the plan reserves the least of
    // those reads on the host, read from the plan at the least context a seat asks (the most
    // card tensors any context holds), so the NVMe expert tier's split leaves them room and the
    // load holds the plan's need. A plan the floor cannot make is refused by the choice below.
    #[cfg(feature = "checkpoint")]
    if let Some(source) = &source
        && let Some(reads) = workstation::least_part_reads(|ctx| {
            let probe = inputs
                .plan_with_slots(&machine, ctx, &plan_levers, n)
                .ok()?;
            Some(super::folder::least_reads(&split, source, &probe, 0))
        })?
    {
        workstation::reserve_part_reads(&mut machine, reads);
    }
    // A load that streams pins the stream's lane staging for its life: its plan reserves it on
    // the host (`workstation::reserve_stream_staging`).
    let staged = ring_inputs.as_ref().map(|r| {
        let mut staged = machine.clone();
        workstation::reserve_stream_staging(&mut staged, r.stream_slot);
        staged
    });
    let set = levers.xstream().is_some();
    // `--parallel auto`'s count ([`AutoSlots::count`]) over the plans the load runs: the plain
    // plan and the staged plans chosen by the same stream and residency rules as the load. A body
    // that serves its tier card beside one sequence serves one slot there.
    let (slots, ctx) = match a.auto {
        None => (slots, a.ctx),
        Some(auto) if !S::TIER_SLOTS && !place.tier_cards().is_empty() => auto.one(&format!(
            "the {} body serves its expert tier card beside one sequence",
            S::FAMILY
        )),
        Some(auto) => auto.count::<PlaceError, _>(
            false,
            |set| {
                let one = (&machine, &plan_levers, NonZeroUsize::MIN, place);
                let chosen = plan_at(&split, (set, 1), &inputs, one)?;
                Ok((usize::try_from(chosen.slot_ctx)?, chosen.rule, ()))
            },
            |(), n, ctx| {
                let n = NonZeroUsize::new(n).ok_or("--parallel 0: the server serves no slot")?;
                let ctx = u64::try_from(ctx)?;
                let plain = inputs.plan_with_slots(&machine, ctx, &plan_levers, n)?;
                let Some((r, staged)) = ring_inputs.as_ref().zip(staged.as_ref()) else {
                    return Ok(true);
                };
                let staging = Some((
                    workstation::plan_streams(&plain),
                    || Ok(r.plan_with_slots(staged, ctx, &plan_levers, n)?),
                    || Ok(inputs.plan_with_slots(staged, ctx, &plan_levers, n)?),
                ));
                streamed(
                    set,
                    plain,
                    staging,
                    |p| S::residency(&levers, &flags, place, p),
                    |pick| pick.word == "off",
                )
                .map(|_| true)
            },
        )?,
    };
    let n = NonZeroUsize::new(slots).ok_or("--parallel 0: the server serves no slot")?;
    // The context, then the plan of every slot around it (`choose_ctx`, every seat's): a tier
    // card, a second card, a context the card cannot hold, a host room that cannot hold the host
    // experts and a budget that binds are refused here by name, before any load; the load plans
    // the same.
    let asking = (ctx, slots);
    let plain = plan_at(&split, asking, &inputs, (&machine, &plan_levers, n, place))?;
    // Where the seat streams, the ring's plan on the staged machine beside it, and the pick is
    // fitted to the plan the load runs. An unset `BLOOMERY_XSTREAM` keeps no ring where the plan
    // without it holds every expert on the card (the rule then ends at `off`: nothing to stream).
    // Where the plan cannot hold the ring, the rule sizes it from what the card leaves free at the
    // load ([`PlanInputs::streaming`]), so that plan keeps the staging on the host.
    let staging = ring_inputs
        .as_ref()
        .zip(staged.as_ref())
        .map(|(r, staged)| {
            let on = (staged, &plan_levers, n, place);
            let split = &split;
            let inputs = &inputs;
            (
                workstation::plan_streams(&plain.plan),
                move || plan_at(split, asking, r, on),
                move || plan_at(split, asking, inputs, on),
            )
        });
    let (asked, ring, pick) = streamed(
        set,
        plain,
        staging,
        |p| S::residency(&levers, &flags, place, &p.plan),
        |pick| pick.word == "off",
    )?;
    eprintln!("{}", asked.line());
    let slot_ctx = usize::try_from(asked.slot_ctx)?;
    let plan = asked.plan;
    record::plan(place.name(), plan.machine, &plan).eprint();
    let (residency, extra) = match pick {
        Some(pick) => S::settle(&plan, pick)?,
        None => (Residency::Off, 0),
    };
    let cache_ram = S::cache_budget(&flags, &plan, extra, slot_ctx, &inputs)?;
    need::print(&plan, extra, cache_ram, true)?;
    parallel_line(slots, slot_ctx, from, Pass::of(slots, S::ONE_PASS));
    if a.plan_only {
        // The records before the load are out; nothing was opened on a card.
        bloomery_gpu_gates::start::plan_exit();
    }
    let props = EngineProps {
        model: Some(model_props(&split, &inputs.model)),
        placement: placement_of(S::NAME, plan.machine, &plan),
        ..EngineProps::default()
    };
    let card = machine.cards[0].name.clone();
    let machine = plan.machine.clone();
    drop(plan);
    // A folder's header goes to the engine thread with its parts; a file's is reopened there.
    let folder = match source {
        Some(source) => Some(Folder {
            header: split,
            source,
        }),
        None => {
            drop(split);
            None
        }
    };
    let tokenizer = match tokenizer {
        Some(tokenizer) => tokenizer,
        None => Tokenizer::from_gguf(&path)?,
    };
    let vocab = Arc::new(Vocab::new(tokenizer)?);
    let open = Open {
        place,
        lane,
        ctx: slot_ctx,
        slots: n,
        machine,
        path: path.clone(),
        plan: plan_levers,
        host: levers.host(),
        pin_main: levers.pin_main(),
        group: levers.prefill_group(),
        residency,
        cache_ram,
        xstream: levers.xstream(),
        ring,
        folder,
        flags,
    };
    let engine = SeatEngine::spawn(
        move || S::open(open),
        slot_ctx,
        vocab,
        card,
        props,
        cache_ram,
    )?;
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

#[cfg(test)]
mod tests {
    use bloomery_gpu_gates::GateError;

    /// The fixed-context count holds the staging of an unset stream as the final load does.
    /// The plain eight-slot plan fits, both staged seven-slot plans refuse, and six fits only
    /// without the ring. A set stream needs its ring; an off residency needs no staging.
    #[test]
    fn auto_slots_hold_default_stream_staging() {
        use bloomery_gpu_gates::bind::AutoSlots;
        use model::arch::place_hybrid::PlaceError;
        use model::placement::PlacementError;
        use model::placement::ctx_ask::Rule;

        let plan = |n: usize, ctx: usize, reserve: usize| -> Result<usize, GateError> {
            let need = n * ctx + reserve;
            if need > 8192 {
                return Err(PlacementError::DraftHostRoom {
                    room: 8192,
                    draft: u64::try_from(need)?,
                }
                .into());
            }
            Ok(n)
        };
        let select = |set, off, staging: bool, n, ctx| {
            super::streamed(
                set,
                plan(n, ctx, 0)?,
                staging.then_some((true, || plan(n, ctx, 2560), || plan(n, ctx, 2048))),
                |_| Ok(Some(off)),
                |off| *off,
            )
        };
        assert!(plan(8, 1024, 0).is_ok(), "the plain maximum fits");
        assert!(plan(7, 1024, 2560).is_err(), "staging and ring exceed room");
        assert!(plan(7, 1024, 2048).is_err(), "staging alone exceeds room");
        assert_eq!(
            select(false, false, true, 6, 1024).unwrap(),
            (6, false, Some(false)),
            "the smaller staged plan fits without its ring"
        );
        for (set, off, staging, expected) in [
            (false, false, true, 6),
            (true, false, true, 5),
            (false, true, true, 8),
            (false, false, false, 8),
        ] {
            let chosen = AutoSlots {
                slot_ctx: Some(1024),
            }
            .count::<PlaceError, _>(
                false,
                |_| Ok((1024, Rule::Set, ())),
                |(), n, ctx| select(set, off, staging, n, ctx).map(|_| true),
            )
            .unwrap();
            assert_eq!(
                chosen,
                (expected, Some(expected * 1024)),
                "the count follows the selected load: set={set} off={off} staging={staging}"
            );
            select(set, off, staging, chosen.0, 1024).expect("the final load fits");
        }
    }

    /// A streaming seat's refused staged plans under an unset lever load the plain plan where its
    /// residency pick is `off` (a plain plan of no card expert streams nothing), with that one
    /// pick; a pick that streams keeps the refusal, as does a set lever. A staged plan whose pick
    /// is `off` loads the plain plan. FAIL-first: the refusal returned before the pick, and a
    /// fitting plain plan of no card expert did not load.
    #[test]
    fn a_plain_plan_that_streams_nothing_loads_past_refused_staging() {
        let refused = || Err::<Option<(u8, bool)>, GateError>("the host room".into());
        let off = |w: &&str| *w == "off";
        let picks = |w: &'static str| move |_: &u8| Ok(Some(w));
        assert_eq!(
            super::picked(refused(), false, 1, picks("off"), off).unwrap(),
            (1, false, Some("off"))
        );
        let e = super::picked(refused(), false, 1, picks("mid"), off).expect_err("streams");
        assert!(e.to_string().contains("the host room"), "{e}");
        assert!(super::picked(refused(), true, 1, picks("off"), off).is_err());
        assert_eq!(
            super::picked(Ok(Some((2, true))), false, 1, picks("mid"), off).unwrap(),
            (2, true, Some("mid"))
        );
        assert_eq!(
            super::picked(Ok(Some((2, true))), false, 1, picks("off"), off).unwrap(),
            (1, false, Some("off"))
        );
        assert_eq!(
            super::picked(Ok(None), false, 1, picks("mid"), off).unwrap(),
            (1, false, Some("mid"))
        );
    }

    /// A streaming seat whose ring does not fit under an unset lever plans without it on the
    /// staged machine, and that plan's refusal is the seat's; where the plain plan hosts no
    /// expert, nothing streams and the seat loads the plain plan. FAIL-first: the second refusal
    /// was dropped and the seat loaded the plain plan, which holds no staging (`None`); then a
    /// plain plan that hosts no expert was refused by the staged plans' room.
    #[test]
    fn a_refused_plan_without_the_ring_is_the_seats_refusal() {
        let refused = || Err::<u8, GateError>("the host room".into());
        let e = super::ringed(false, true, refused, refused).expect_err("the refusal");
        assert!(e.to_string().contains("the host room"), "{e}");
        assert_eq!(
            super::ringed(false, true, refused, || Ok(1)).unwrap(),
            Some((1, false))
        );
        assert!(
            super::ringed(true, true, refused, || Ok(1)).is_err(),
            "a set lever's refusal"
        );
        assert_eq!(
            super::ringed(false, false, || Ok(2), || Ok(1)).unwrap(),
            None
        );
        assert_eq!(
            super::ringed(false, false, refused, refused).unwrap(),
            None,
            "a plain plan that hosts no expert streams nothing"
        );
    }
}
