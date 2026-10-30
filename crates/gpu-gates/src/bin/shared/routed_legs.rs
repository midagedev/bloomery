//! The common owner of gate_routed_legs: the census reading and checking,
//! the arch → family dispatch, the card-room derivation, the retype sweep
//! and the checks L1–L6. A family is added by one adapter file
//! (`routed_legs/<family>.rs`, holding only what the family forces — how its
//! production `--place`-unset defaults plan one split, the loader's plan-time
//! stack table where it has one, a leg-adjacent sweep triple where its leg is
//! exact) and one row in [`family_of`]; no check and no per-family branch
//! lives here. Every stub is planned from its header alone — the census
//! writer's tensor bytes are all zero — so no planner step that reads data
//! changes an answer.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::error::Error as StdError;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::{checks_failed, verdict};
use gguf::Split;
use gguf::quant::GgmlType;
use gguf::write::{Layout, TensorDecl, Writer};
use model::arch::models::Arch;
use model::arch::qwen3moe::place::PlaceError as Place3Error;
use model::arch::qwen35moe::place::PlaceError as Place35Error;
use model::placement::accepts::{Program, leg_arms, leg_runs, program_of};
use model::placement::workstation::DeviceInfo;
use model::placement::{PlacementError, Plan, PlanLevers};
use refset::census::{Manifest, StubSet};

#[path = "qwen3moe_place.rs"]
#[allow(
    dead_code,
    reason = "the arms plan qwen3moe and qwen35moe files only; the module's other half serves the CLI and the seats"
)]
mod q3place;

#[path = "qwen38_place.rs"]
#[allow(
    dead_code,
    reason = "the qwen38 arms read the family's unset-place rule; the module's other half serves the CLI and the seat"
)]
mod q38place;

#[path = "glm5next_place.rs"]
#[allow(
    dead_code,
    reason = "the glm5next arms read the family's unset-place rule and tier bytes; the module's other half serves the CLI and the seat"
)]
mod glmplace;

#[path = "ds41_place.rs"]
#[allow(
    dead_code,
    reason = "the deepseek41 arms read the family's unset-place rule and tier bytes; the module's other half serves the binaries"
)]
mod ds41place;

// A file the bin loads by `#[path]` owns its directory, so the adapters'
// paths are named here rather than resolved under `routed_legs/`.
#[path = "routed_legs/deepseek41.rs"]
mod deepseek41;
#[path = "routed_legs/glm5next.rs"]
mod glm5next;
#[path = "routed_legs/mimo2.rs"]
mod mimo2;
#[path = "routed_legs/qwen35moe.rs"]
mod qwen35moe;
#[path = "routed_legs/qwen38.rs"]
mod qwen38;
#[path = "routed_legs/qwen3moe.rs"]
mod qwen3moe;

/// The families the gate plans, by the file's architecture (read with the
/// planners' own reader, `model::arch::spec`, which reads `glm5-next` and
/// `clef` too). V4 and V4.1 files run one program and one planner.
fn family_of(arch: Arch) -> Option<&'static dyn Adapter> {
    match arch {
        Arch::Qwen3Moe => Some(&qwen3moe::Row),
        Arch::Qwen35Moe => Some(&qwen35moe::Row),
        Arch::Qwen4Exp => Some(&qwen38::Row),
        Arch::Glm5Next => Some(&glm5next::Row),
        Arch::MiMo2 => Some(&mimo2::Row),
        Arch::Deepseek41 | Arch::Deepseek4 => Some(&deepseek41::Row),
        Arch::Qwen35
        | Arch::Llama
        | Arch::Lfm3
        | Arch::MiniMaxM3
        | Arch::Raon1
        | Arch::SolarOpen2
        | Arch::NemotronH
        | Arch::NemotronHMoe
        | Arch::GemmaEmbedding
        | Arch::GemmaEmbedding2
        | Arch::Flint
        | Arch::MuseSpark
        | Arch::MistralLarge4 => None,
    }
}

/// Why a set the gate does not plan is skipped: the Qwen3.5, Llama and LFM3 dense
/// trunks and the EmbeddingGemma 2 encoder route nothing, and an architecture
/// no adapter plans is never planned as a model.
fn skip_of(arch: Arch) -> &'static str {
    match arch {
        Arch::Qwen35 | Arch::Llama | Arch::Lfm3 => "dense trunk",
        Arch::GemmaEmbedding2 => "no routed expert",
        Arch::Qwen3Moe
        | Arch::Qwen35Moe
        | Arch::Qwen4Exp
        | Arch::Glm5Next
        | Arch::MiMo2
        | Arch::MiniMaxM3
        | Arch::Raon1
        | Arch::SolarOpen2
        | Arch::Deepseek41
        | Arch::Deepseek4
        | Arch::NemotronH
        | Arch::NemotronHMoe
        | Arch::GemmaEmbedding
        | Arch::Flint
        | Arch::MuseSpark
        | Arch::MistralLarge4 => "no program",
    }
}

/// One plan the gate holds: the layers' card counts and the byte terms the
/// rooms derive from.
#[derive(Clone)]
struct PlanView {
    /// Per layer: the experts the plan's stage card holds.
    n_l: Vec<u64>,
    /// Per tier card, per layer: the experts the tier holds.
    tier_n_l: Vec<Vec<u64>>,
    /// The routed-expert bytes on the stage card.
    card_expert_bytes: u64,
    /// The routed experts the plan leaves on the host, and the bytes it
    /// pages through the NVMe tier.
    host_experts: u64,
    nvme_expert_bytes: u64,
    /// What the family's loader-side stack table said of this plan (L6):
    /// `None` where the family has no plan-time table.
    loader: Option<Result<(), String>>,
}

impl PlanView {
    /// `self` with the loader table's answer for this plan: `Ok` when the
    /// table took the plan's header, else the refusal's text.
    fn with_loader<E: std::fmt::Display>(mut self, answer: Result<(), E>) -> PlanView {
        self.loader = Some(answer.map_err(|e| e.to_string()));
        self
    }
}

/// A named planner refusal — an outcome L2 counts by kind, never red. Its
/// `floor` is the refusal's own where it carries one (a
/// `CardBudgetFloor`), which the room derivation reads.
#[derive(Clone)]
struct Refusal {
    kind: &'static str,
    floor: Option<u64>,
    text: String,
}

/// One plan attempt's outcome.
#[derive(Clone)]
enum Outcome {
    Plan(PlanView),
    Refused(Refusal),
    /// Any other error: red, with its text.
    Red(String),
}

impl std::fmt::Display for Outcome {
    /// The text of a refusal or a red outcome, where a context probe or the
    /// seat's slot fit reads it as an error.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Outcome::Plan(_) => f.write_str("a plan, where an error was read"),
            Outcome::Refused(r) => f.write_str(&r.text),
            Outcome::Red(text) => f.write_str(text),
        }
    }
}

/// One arm of a family's production default: the unbudgeted answer and the
/// placed planner the forced rooms go through, at the arm's context and
/// slots.
struct Arm {
    /// "cli" or "serve".
    name: &'static str,
    /// The default answered the whole load — no plan; the forced rooms plan
    /// the placed path at the same context.
    whole: bool,
    /// The unbudgeted plan (`None` where the default answered whole).
    unbudgeted: Option<Outcome>,
    /// The placed planner under `levers`.
    placed: Box<dyn Fn(&PlanLevers) -> Outcome>,
}

/// A family's adapter: how its production `--place`-unset defaults plan one
/// split, one arm per binary that opens the family's files. An adapter is a
/// unit struct shared by the worker threads.
trait Adapter: Sync {
    /// The family word the lines carry.
    fn name(&self) -> &'static str;
    /// The arms on `split`, each its own attempt (one arm's refusal does not
    /// stop the other).
    fn arms(&self, split: &Split) -> Vec<Result<Arm, Outcome>>;
    /// The name of the family's loader-side stack table that a plan's header
    /// must take (L6), where the loader reads the header at plan time
    /// without a card; `None` where the family has none. An adapter that
    /// names one attaches its answer to every plan it makes
    /// ([`PlanView::with_loader`]).
    fn loader_table(&self) -> Option<&'static str> {
        None
    }
    /// Sweep triples beside the common ones, for a family whose card leg is
    /// exact and so shares no triple with them: the leg itself and its
    /// nearest neighbours.
    fn extra_triples(&self) -> Vec<[GgmlType; 3]> {
        Vec::new()
    }
}

/// Where a plan line sits: the family word, the set, the arm and the room
/// (the retype sweep's arm word carries the triple).
struct At<'a> {
    family: &'a str,
    set: &'a str,
    arm: &'a str,
    room: &'a str,
}

/// The counts one family's run holds.
#[derive(Default)]
struct Counts {
    sets: u64,
    plans: u64,
    partial: u64,
    whole: u64,
    floors: u64,
    /// Arms whose rooms did not derive because the planner refuses the file
    /// at every card budget, by one named kind ([`Rooms::Standing`]).
    standing: u64,
    retyped: u64,
    /// The planner calls the gate made: one a plan line, one a floor probe and
    /// one a probe of an adapter's context choice (a qwen3moe or qwen35moe
    /// default's census reads are `q3place::census_reads`, not here).
    calls: u64,
    /// Plans L6 asked the loader table about, and the ones it refused.
    loader_asked: u64,
    loader_refused: u64,
    refusals: BTreeMap<&'static str, u64>,
}

impl Counts {
    /// One more refusal of `kind`.
    fn refused(&mut self, kind: &'static str) {
        *self.refusals.entry(kind).or_default() += 1;
    }

    /// `other`'s counts added to these.
    fn add(&mut self, other: &Counts) {
        self.sets += other.sets;
        self.plans += other.plans;
        self.partial += other.partial;
        self.whole += other.whole;
        self.floors += other.floors;
        self.standing += other.standing;
        self.retyped += other.retyped;
        self.calls += other.calls;
        self.loader_asked += other.loader_asked;
        self.loader_refused += other.loader_refused;
        for (kind, n) in &other.refusals {
            *self.refusals.entry(kind).or_default() += n;
        }
    }

    /// The family's counts line.
    fn line(&self, family: &str, traps: usize, loader: Option<&str>) -> String {
        let refusals = self
            .refusals
            .iter()
            .map(|(k, n)| format!("{k}={n}"))
            .collect::<Vec<_>>();
        let refusals = if refusals.is_empty() {
            "none".to_string()
        } else {
            refusals.join(" ")
        };
        let loader = loader.map_or(String::new(), |table| {
            format!(
                " loader {table} asked {} refused {}",
                self.loader_asked, self.loader_refused
            )
        });
        format!(
            "legs: {family}: sets {} plans {} partial {} whole {} floors {} standing {} \
             refusals {refusals} retyped {} traps {traps}{loader}",
            self.sets,
            self.plans,
            self.partial,
            self.whole,
            self.floors,
            self.standing,
            self.retyped
        )
    }
}

/// What one unit of work (a set, or one swept header) printed and counted:
/// its lines, in the order it made them, and its counts. A worker thread
/// builds one; the main thread prints them in the manifest's order.
struct Out {
    lines: Vec<String>,
    counts: Counts,
    ok: bool,
}

impl Out {
    fn new() -> Out {
        Out {
            lines: Vec::new(),
            counts: Counts::default(),
            ok: true,
        }
    }

    /// One more line of the unit's output.
    fn say(&mut self, line: String) {
        self.lines.push(line);
    }

    /// A line that passed or failed, with its verdict word; a failure marks
    /// the unit red.
    fn judged(&mut self, line: String, held: bool) {
        self.ok &= held;
        self.lines.push(format!("{line} {}", verdict(held)));
    }
}

/// What a family's checks read of it: its adapter and the program its files
/// run.
#[derive(Clone, Copy)]
struct Who {
    adapter: &'static dyn Adapter,
    program: Program,
}

/// One family's state across the run: its counts, the program its files run
/// and its sweep seed.
struct Family {
    who: Who,
    counts: Counts,
    /// The smallest set with a carded layer, that layer and its arms' rooms.
    seed: Option<Seed>,
    /// The sweep triples the retyping wrote (the trap count's domain).
    swept: Vec<[GgmlType; 3]>,
    /// Seconds of worker time the sets and the sweep took, and the plan
    /// calls the sets made (the rest are the sweep's).
    sets_secs: f64,
    sweep_secs: f64,
    sets_calls: u64,
}

/// The retype sweep's seed: the family's smallest planned set, a layer its
/// unbudgeted plan cards, and each arm's derived rooms.
struct Seed {
    set: StubSet,
    /// The set's total `file_bytes`, the smallest-ness the seed was picked
    /// by.
    total: u64,
    layer: usize,
    /// Per arm name: its three rooms.
    rooms: Vec<(&'static str, [u64; 3])>,
}

/// The most worker threads the gate plans on. The planners are host work on
/// headers alone; the box's cores are shared with other tracks' builds.
const WORKERS: usize = 8;

thread_local! {
    /// The probes of context choices the adapters made on this thread
    /// ([`probed`]); a unit of work runs on one thread, so its share is the
    /// difference across it.
    static PROBES: Cell<u64> = const { Cell::new(0) };
}

/// One plan an adapter's context choice probed, counted for the unit of work
/// running on this thread.
fn probed() {
    PROBES.with(|p| p.set(p.get() + 1));
}

/// The probes counted on this thread so far.
fn probes() -> u64 {
    PROBES.with(Cell::get)
}

/// This process's census, read once: the gate holds one card for its whole
/// run and plans from headers alone, so the free bytes the arms resolve a
/// placement against are one reading, not one per arm (each reading asks the
/// driver and spawns nvidia-smi). The qwen3moe and qwen35moe defaults read
/// their own (`q3place`).
fn census() -> Result<&'static [DeviceInfo], Outcome> {
    static CENSUS: OnceLock<Result<Vec<DeviceInfo>, String>> = OnceLock::new();
    CENSUS
        .get_or_init(|| bloomery_gpu_gates::gpu_census::census().map_err(|e| e.to_string()))
        .as_ref()
        .map(Vec::as_slice)
        .map_err(|e| Outcome::Red(format!("the device census: {e}")))
}

/// `work(0..count)` on up to [`WORKERS`] threads, the results in index order:
/// the order the lines print in is the manifest's, not the threads'.
fn map_ordered<T: Send>(count: usize, work: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let next = AtomicUsize::new(0);
    let slots: Mutex<Vec<Option<T>>> = Mutex::new((0..count).map(|_| None).collect());
    let workers = WORKERS.min(count).max(1);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            // A planner may recurse over a large header: a worker takes the
            // stack the main thread's default is not the measure of.
            std::thread::Builder::new()
                .stack_size(16 << 20)
                .spawn_scoped(scope, || {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= count {
                            break;
                        }
                        let done = work(i);
                        slots.lock().expect("a worker panicked holding the results")[i] =
                            Some(done);
                    }
                })
                .expect("a worker thread starts");
        }
    });
    slots
        .into_inner()
        .expect("a worker panicked holding the results")
        .into_iter()
        .map(|done| done.expect("every item ran"))
        .collect()
}

/// What planning one census set produced: its output, why it was skipped or
/// the family it was planned as, and the terms the sweep's seed is picked by.
struct SetDone {
    out: Out,
    /// Why the set is not planned (counted and printed in `out`).
    skip: Option<&'static str>,
    /// The family and program of a planned set.
    who: Option<Who>,
    /// The first layer a plan of the set gave card experts.
    carded: Option<usize>,
    /// Per arm name: its three rooms.
    rooms: Vec<(&'static str, [u64; 3])>,
    /// Seconds of worker time.
    secs: f64,
}

/// Plan one census set through its family's arms at derived rooms: the
/// manifest check, the header's reading, then each arm's unbudgeted plan,
/// floor probe and three rooms, with the checks (L1, L2, L3's partial count,
/// L6) over each plan.
fn plan_set(set: &StubSet, root: &Path) -> Result<SetDone, String> {
    let started = Instant::now();
    let before = probes();
    let mut done = SetDone {
        out: Out::new(),
        skip: None,
        who: None,
        carded: None,
        rooms: Vec::new(),
        secs: 0.0,
    };
    plan_set_into(set, root, &mut done).map_err(|e| e.to_string())?;
    done.out.counts.calls += probes() - before;
    done.secs = started.elapsed().as_secs_f64();
    Ok(done)
}

fn plan_set_into(set: &StubSet, root: &Path, done: &mut SetDone) -> Result<(), GateError> {
    let name = set.name();
    let out = &mut done.out;
    // A draft set's first shard's basename — the census writer's naming.
    // Drafts carry their model's architecture but are never planned as a
    // model.
    let first_shard = set.shards[0].file.rsplit('/').next().unwrap_or("");
    if first_shard.starts_with("mtp-") || first_shard.starts_with("dflash-") {
        done.skip = Some("draft");
        out.say(format!("legs: skip {name}: draft"));
        return Ok(());
    }
    let first = match set.check(root) {
        Ok(first) => first,
        Err(e) => {
            out.judged(format!("legs: census {name}: {e}"), false);
            return Ok(());
        }
    };
    let split = match Split::open(&first) {
        Ok(split) => split,
        Err(e) => {
            // A stub that checks against its row but the reader cannot
            // open (a quant type this build's table does not size) is a
            // read refusal, not a gate defect: the census holds users'
            // files of every quant.
            done.skip = Some("no reader");
            out.say(format!("legs: skip {name}: no reader: {e}"));
            return Ok(());
        }
    };
    let arch = match model::arch::spec(&split) {
        Ok(read) => read.spec.arch,
        Err(e) => {
            done.skip = Some("no reader");
            out.say(format!("legs: skip {name}: no reader: {e}"));
            return Ok(());
        }
    };
    let Some(adapter) = family_of(arch) else {
        done.skip = Some(skip_of(arch));
        out.say(format!("legs: skip {name}: {}", skip_of(arch)));
        return Ok(());
    };
    let program = program_of(arch).expect("every architecture the dispatch plans has a program");
    let who = Who { adapter, program };
    done.who = Some(who);
    out.counts.sets += 1;
    let stacks = routed_stacks(&split)?;
    for attempt in adapter.arms(&split) {
        let arm = match attempt {
            Ok(arm) => arm,
            Err(Outcome::Refused(r)) => {
                out.counts.refused(r.kind);
                out.judged(
                    format!(
                        "legs: {} {name}: refused {}: {}",
                        adapter.name(),
                        r.kind,
                        r.text
                    ),
                    true,
                );
                continue;
            }
            Err(Outcome::Red(text)) => {
                out.judged(format!("legs: {} {name}: {text}", adapter.name()), false);
                continue;
            }
            // An adapter's construction error is a refusal or red,
            // never a plan.
            Err(Outcome::Plan(_)) => {
                unreachable!("an arm's construction error holds no plan")
            }
        };
        let at = At {
            family: adapter.name(),
            set: &name,
            arm: arm.name,
            room: "-",
        };
        // The unbudgeted answer first: the default as it is.
        if arm.whole {
            out.counts.whole += 1;
            out.judged(
                format!(
                    "legs: {} {name} {}: room -: whole load",
                    adapter.name(),
                    arm.name
                ),
                true,
            );
        } else if let Some(outcome) = &arm.unbudgeted {
            plan_line(out, who, &at, outcome, &stacks, &mut done.carded);
        } else {
            out.judged(
                format!(
                    "legs: {} {name} {}: an arm that answered neither whole nor a plan",
                    adapter.name(),
                    arm.name
                ),
                false,
            );
        }
        // The rooms: the floor from the placed planner's own refusal,
        // then floor + f·E.
        let e = expert_span(&arm, &stacks);
        let unbudgeted_kind = match &arm.unbudgeted {
            Some(Outcome::Refused(r)) => Some(r.kind),
            _ => None,
        };
        out.counts.calls += 1;
        let (floor, arm_rooms) = match rooms_of(arm.placed.as_ref(), e, unbudgeted_kind) {
            Ok(Rooms::Floor(floor, rooms)) => (floor, rooms),
            Ok(Rooms::Standing(r)) => {
                out.counts.standing += 1;
                out.judged(
                    format!(
                        "legs: {} {name} {}: no rooms: the planner refuses the file at a zero \
                         card budget as it does unbudgeted, {}: {}",
                        adapter.name(),
                        arm.name,
                        r.kind,
                        r.text
                    ),
                    true,
                );
                continue;
            }
            Err(text) => {
                out.judged(
                    format!(
                        "legs: {} {name} {}: the rooms did not derive: {text}",
                        adapter.name(),
                        arm.name
                    ),
                    false,
                );
                continue;
            }
        };
        out.counts.floors += 1;
        out.say(format!(
            "legs: {} {name} {}: floor {floor}",
            adapter.name(),
            arm.name
        ));
        done.rooms.push((arm.name, arm_rooms));
        for &bytes in &arm_rooms {
            let outcome = (arm.placed)(&budget(bytes));
            let room = bytes.to_string();
            let at = At { room: &room, ..at };
            plan_line(out, who, &at, &outcome, &stacks, &mut done.carded);
        }
    }
    Ok(())
}

/// The seconds per call of a family's work, `-` for no call.
fn per_call(secs: f64, calls: u64) -> String {
    if calls == 0 {
        "-".to_string()
    } else {
        format!("{:.3}", secs / calls as f64)
    }
}

/// The gate's run: read the census, plan every family's sets through its
/// production defaults at derived rooms, sweep one set's retyped headers per
/// family, and answer green only when every carded layer is one its file's
/// architecture runs, every loader table takes the plans, and every family's
/// sweep holds a trap. The sets and the swept headers are planned on
/// [`WORKERS`] threads and printed in the manifest's order.
pub(super) fn run() -> Result<(), GateError> {
    bloomery_levers::at_main(&[bloomery_gpu::arch::qwen3moe::ubatch::UBATCH_ENV])?;
    let (manifest, root) = args()?;
    let sets = Manifest::read(&manifest)
        .map_err(|e| format!("the census manifest {}: {e}", manifest.display()))?;
    println!(
        "legs: {} sets under the manifest {} (stubs under {})",
        sets.sets().len(),
        manifest.display(),
        root.display()
    );
    let started = Instant::now();
    // The driver and the census cache are brought up before any worker.
    let _ = census();
    let all = sets.sets();
    let planned = map_ordered(all.len(), |i| plan_set(&all[i], &root));
    let mut ok = true;
    let mut skips: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut families: BTreeMap<&'static str, Family> = BTreeMap::new();
    for (set, done) in all.iter().zip(planned) {
        let done = done?;
        for line in &done.out.lines {
            println!("{line}");
        }
        ok &= done.out.ok;
        if let Some(why) = done.skip {
            *skips.entry(why).or_default() += 1;
        }
        let Some(who) = done.who else {
            continue;
        };
        let family = families
            .entry(who.adapter.name())
            .or_insert_with(|| Family {
                who,
                counts: Counts::default(),
                seed: None,
                swept: Vec::new(),
                sets_secs: 0.0,
                sweep_secs: 0.0,
                sets_calls: 0,
            });
        family.counts.add(&done.out.counts);
        family.sets_secs += done.secs;
        // The sweep seed: the smallest set that cards a layer.
        if let Some(layer) = done.carded {
            let total = set.shards.iter().map(|s| s.file_bytes).sum();
            if family.seed.as_ref().is_none_or(|seed| total < seed.total) {
                family.seed = Some(Seed {
                    set: set.clone(),
                    total,
                    layer,
                    rooms: done.rooms,
                });
            }
        }
    }
    for (name, family) in families.iter_mut() {
        family.sets_calls = family.counts.calls;
        println!(
            "legs: time {name} sets: {} plan calls in {:.1} s of work ({} s a call)",
            family.sets_calls,
            family.sets_secs,
            per_call(family.sets_secs, family.sets_calls)
        );
    }
    // The retype sweep: every family's seed written once per triple, planned
    // through its arms; then L3, L5 and the loader table's vacuity per
    // family.
    let mut sweeps = Vec::new();
    for (&name, family) in families.iter_mut() {
        if let Some(seed) = family.seed.take() {
            sweeps.push(Sweep::prepare(name, family.who, seed, &root)?);
        }
    }
    let dir =
        std::env::temp_dir().join(format!("bloomery-routedlegs-{}-sweep", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    let jobs: Vec<(usize, [GgmlType; 3])> = sweeps
        .iter()
        .enumerate()
        .flat_map(|(j, sweep)| sweep.triples.iter().map(move |&triple| (j, triple)))
        .collect();
    let headers = map_ordered(jobs.len(), |n| {
        let (j, triple) = jobs[n];
        sweeps[j].header(triple, n, &root, &dir)
    });
    for (&(j, triple), done) in jobs.iter().zip(headers) {
        let done = done?;
        for line in &done.out.lines {
            println!("{line}");
        }
        ok &= done.out.ok;
        let family = families
            .get_mut(sweeps[j].name)
            .expect("a sweep's family ran");
        family.counts.add(&done.out.counts);
        family.sweep_secs += done.secs;
        if done.wrote {
            family.swept.push(triple);
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    for (name, family) in &families {
        let calls = family.counts.calls - family.sets_calls;
        println!(
            "legs: time {name} sweep: {calls} plan calls in {:.1} s of work ({} s a call)",
            family.sweep_secs,
            per_call(family.sweep_secs, calls)
        );
    }
    for family in families.values() {
        let name = family.who.program.name();
        let traps = trap_count(family.who.program, &family.swept);
        if traps == 0 {
            println!("legs: {name}: the sweep has no trap {}", verdict(false));
            ok = false;
        }
        if family.counts.partial == 0 {
            println!("legs: {name}: vacuous: no partial plan {}", verdict(false));
            ok = false;
        }
        if let Some(table) = family.who.adapter.loader_table()
            && family.counts.loader_asked == 0
        {
            println!(
                "legs: {name}: vacuous: the loader table {table} was never asked {}",
                verdict(false)
            );
            ok = false;
        }
    }
    let skip_line = skips
        .iter()
        .map(|(why, n)| format!("{why}={n}"))
        .collect::<Vec<_>>()
        .join(" ");
    println!("legs: skips: {skip_line}");
    for (name, family) in &families {
        println!(
            "{}",
            family.counts.line(
                name,
                trap_count(family.who.program, &family.swept),
                family.who.adapter.loader_table()
            )
        );
    }
    let work: f64 = families.values().map(|f| f.sets_secs + f.sweep_secs).sum();
    println!(
        "legs: wall {:.1} s, {:.1} s of work on up to {WORKERS} workers; the qwen3moe and \
         qwen35moe defaults made {} census reads of their own",
        started.elapsed().as_secs_f64(),
        work,
        q3place::census_reads(),
    );
    println!("gate_routed_legs: {}", verdict(ok));
    if !ok {
        return Err(checks_failed());
    }
    Ok(())
}

/// The bin's arguments: `--manifest PATH` and `--root PATH`, defaulting to
/// the census reader's own
/// ([`MANIFEST`](refset::census::MANIFEST), [`ROOT`](refset::census::ROOT)).
fn args() -> Result<(PathBuf, PathBuf), GateError> {
    let mut of = (None, None);
    let mut words = std::env::args().skip(1);
    while let Some(word) = words.next() {
        let missing = |flag: &str| {
            format!("gate_routed_legs: {flag} names no path: --manifest PATH, --root PATH")
        };
        match word.as_str() {
            "--manifest" => {
                of.0 = Some(words.next().ok_or_else(|| missing("--manifest"))?);
            }
            "--root" => {
                of.1 = Some(words.next().ok_or_else(|| missing("--root"))?);
            }
            other => {
                return Err(format!(
                    "gate_routed_legs: {other}: the arguments are --manifest PATH and --root PATH"
                )
                .into());
            }
        }
    }
    Ok((
        PathBuf::from(of.0.unwrap_or_else(|| refset::census::MANIFEST.to_string())),
        PathBuf::from(of.1.unwrap_or_else(|| refset::census::ROOT.to_string())),
    ))
}

/// One plan's line (L1 over it, L6 over the loader's answer, L3's partial
/// count, the carded-layer pick): `legs: <family> <set> <arm> room <room>:
/// <what> PASS|FAIL`, each FAIL line naming family, set, arm, room, layer and
/// the three types. A failure marks the unit of work red.
fn plan_line(
    out: &mut Out,
    who: Who,
    at: &At<'_>,
    outcome: &Outcome,
    stacks: &Stacks,
    carded: &mut Option<usize>,
) {
    out.counts.calls += 1;
    match outcome {
        Outcome::Refused(r) => {
            out.counts.refused(r.kind);
            out.judged(
                format!(
                    "legs: {} {} {}: room {}: refused {}",
                    at.family, at.set, at.arm, at.room, r.kind
                ),
                true,
            );
        }
        Outcome::Red(text) => {
            out.judged(
                format!(
                    "legs: {} {} {}: room {}: {}",
                    at.family, at.set, at.arm, at.room, text
                ),
                false,
            );
        }
        Outcome::Plan(view) => {
            let mut fails = l1(at, view, stacks, who.program);
            if let Some(answer) = &view.loader {
                out.counts.loader_asked += 1;
                // L6 asks the loader's table about the plans L1 passes: a
                // carded layer the leg refuses is the table's to refuse.
                if let (Err(text), true) = (answer, fails.is_empty()) {
                    out.counts.loader_refused += 1;
                    fails.push(format!(
                        "{} {} {} room {}: the loader table {} refuses a plan the leg runs: {text}",
                        at.family,
                        at.set,
                        at.arm,
                        at.room,
                        who.adapter.loader_table().unwrap_or("-")
                    ));
                }
            }
            let routed = (0..view.n_l.len())
                .filter(|&l| stacks.types.get(l).is_some_and(Option::is_some))
                .count();
            let held = (0..view.n_l.len())
                .filter(|&l| carded_layer(view, l))
                .count();
            if held > 0 && held < routed {
                out.counts.partial += 1;
            }
            if held > 0 && carded.is_none() {
                *carded = Some(
                    (0..view.n_l.len())
                        .find(|&l| carded_layer(view, l))
                        .expect("held > 0 names a carded layer"),
                );
            }
            out.counts.plans += 1;
            out.judged(
                format!(
                    "legs: {} {} {}: room {}: placed {held} of {routed} routed layers carded",
                    at.family, at.set, at.arm, at.room
                ),
                fails.is_empty(),
            );
            for fail in &fails {
                out.say(format!("legs:   {fail} {}", verdict(false)));
            }
        }
    }
}

/// L1: every layer `view` gives card experts — its stage card's `n_l` or any
/// tier's `tier_n_l` — is one `program`'s card leg runs, the types read
/// from the header. Each break is one line naming family, set, arm, room,
/// layer and the three types.
fn l1(at: &At<'_>, view: &PlanView, stacks: &Stacks, program: Program) -> Vec<String> {
    let mut broken = Vec::new();
    for l in 0..view.n_l.len() {
        if !carded_layer(view, l) {
            continue;
        }
        let Some(tys) = stacks.types.get(l).copied().flatten() else {
            broken.push(format!(
                "{} {} {} room {}: layer {l} holds card experts and no routed stacks",
                at.family, at.set, at.arm, at.room
            ));
            continue;
        };
        if leg_runs(program, tys) != Some(true) {
            broken.push(format!(
                "{} {} {} room {}: layer {l} of {tys:?} stacks, which the {} card leg does not \
                 run",
                at.family, at.set, at.arm, at.room, at.family
            ));
        }
    }
    broken
}

/// Whether `view` gives layer `l` card experts: its stage card's count or
/// any tier's.
fn carded_layer(view: &PlanView, l: usize) -> bool {
    view.n_l.get(l).is_some_and(|&n| n > 0)
        || view
            .tier_n_l
            .iter()
            .any(|t| t.get(l).is_some_and(|&n| n > 0))
}

/// The routed expert bytes the rooms scale by (`E`): what the unbudgeted
/// plan put on the card, or the file's whole routed bytes where it put all
/// of them (or answered the whole load, or refused to plan).
fn expert_span(arm: &Arm, stacks: &Stacks) -> u64 {
    if arm.whole {
        return stacks.total();
    }
    match &arm.unbudgeted {
        Some(Outcome::Plan(view))
            if view.host_experts == 0
                && view.nvme_expert_bytes == 0
                && view.tier_n_l.iter().all(|t| t.iter().all(|&n| n == 0)) =>
        {
            stacks.total()
        }
        Some(Outcome::Plan(view)) => view.card_expert_bytes,
        _ => stacks.total(),
    }
}

/// What an arm's room derivation answered.
enum Rooms {
    /// The card budget floor and the three rooms above it.
    Floor(u64, [u64; 3]),
    /// The planner refuses the file by one named kind that no card budget
    /// moves: the unbudgeted plan and the zero-budget probe state the same
    /// kind and neither a `CardBudgetFloor` — a tensor of a role a card
    /// holds with no device format, say. No room exists to force.
    Standing(Refusal),
}

/// The rooms an arm plans at: the card budget floor — read off the
/// `CardBudgetFloor` refusal the placed planner states at a zero budget —
/// and `floor + f·E` for f ∈ {1/4, 1/2, 3/4}. Derived from the planner's
/// own refusal and the plan's own bytes, never from a constant. A zero
/// budget refused by the kind the unbudgeted plan was refused by
/// (`unbudgeted_kind`), and by no floor, is [`Rooms::Standing`]; any other
/// refusal without a floor is an error naming it.
fn rooms_of(
    placed: &dyn Fn(&PlanLevers) -> Outcome,
    e: u64,
    unbudgeted_kind: Option<&'static str>,
) -> Result<Rooms, String> {
    let floor = match placed(&budget(0)) {
        Outcome::Refused(r) => match (r.floor, unbudgeted_kind) {
            (Some(floor), _) => floor,
            (None, Some(kind)) if kind == r.kind => return Ok(Rooms::Standing(r)),
            (None, _) => {
                return Err(format!(
                    "the zero-budget refusal names no floor: not the planner's own \
                     CardBudgetFloor ({}: {})",
                    r.kind, r.text
                ));
            }
        },
        Outcome::Plan(_) => {
            return Err(
                "a zero card budget planned: the floor refusal is missing — the placed planner \
                 never refused the trunk"
                    .to_string(),
            );
        }
        Outcome::Red(text) => return Err(text),
    };
    Ok(Rooms::Floor(
        floor,
        [floor + e / 4, floor + e / 2, floor + 3 * e / 4],
    ))
}

/// The levers of a forced room: one card budget, everything else default.
fn budget(bytes: u64) -> PlanLevers {
    PlanLevers {
        card_budget_bytes: Some(bytes).into(),
    }
}

/// L5, pure from `leg_arms`: the triples of the sweep a per-type union rule
/// would card — every stack's type admitted by the gate·up arm or the down
/// arm, the rule's shape before the leg — that `leg_runs` refuses. The
/// sweep must hold at least one.
fn trap_count(program: Program, triples: &[[GgmlType; 3]]) -> usize {
    let Some(arms) = leg_arms(program) else {
        return 0;
    };
    triples
        .iter()
        .filter(|&&[g, u, d]| {
            [g, u, d]
                .iter()
                .all(|&ty| (arms.gate_up)(ty) || (arms.down)(ty))
                && leg_runs(program, [g, u, d]) == Some(false)
        })
        .count()
}

/// The file's routed stacks, by layer: each layer's `[gate, up, down]` types
/// as the header holds them (`blk.{l}.ffn_{part}_exps.weight`, the three
/// families' one naming) and the stacks' file bytes. A layer the plan never
/// loads (a next-token layer) keeps its entry past the plan's own layer
/// count, where no check reads it.
struct Stacks {
    types: Vec<Option<[GgmlType; 3]>>,
    /// Per layer: the three stacks' file bytes.
    bytes: Vec<u64>,
}

impl Stacks {
    /// The file's whole routed bytes — the `E` of a plan that put all of
    /// them on the card.
    fn total(&self) -> u64 {
        self.bytes.iter().copied().sum()
    }
}

/// Read `split`'s routed stacks ([`Stacks`]); a layer word that is not a
/// number is an error naming the tensor.
fn routed_stacks(split: &Split) -> Result<Stacks, GateError> {
    let mut stacks = Stacks {
        types: Vec::new(),
        bytes: Vec::new(),
    };
    for (_, t) in split.iter_tensors() {
        let Some(rest) = t.name.strip_prefix("blk.") else {
            continue;
        };
        let Some((l, stem)) = rest.split_once('.') else {
            continue;
        };
        let part = match stem {
            "ffn_gate_exps.weight" => 0,
            "ffn_up_exps.weight" => 1,
            "ffn_down_exps.weight" => 2,
            _ => continue,
        };
        let l: usize = l
            .parse()
            .map_err(|_| format!("{}: a layer word that is not a number", t.name))?;
        if stacks.types.len() <= l {
            stacks.types.resize(l + 1, None);
            stacks.bytes.resize(l + 1, 0);
        }
        let cell = stacks.types[l].get_or_insert([GgmlType::F32; 3]);
        cell[part] = t.ty;
        stacks.bytes[l] = stacks.bytes[l].saturating_add(t.nbytes);
    }
    Ok(stacks)
}

/// One family's retype sweep (L4), prepared: the seed's set opened once, its
/// carded layer's types and the triples to write. The shards holding the
/// layer are rewritten per triple (their kvs and tensors verbatim but the
/// three stacks' type and recomputed bytes, over a sparse file of the
/// layout's length), every other shard a symlink to its stub, the file names
/// kept so `Split::open` finds them; each triple has a directory of its own,
/// removed once it is planned, so the headers are planned at once.
struct Sweep {
    name: &'static str,
    who: Who,
    seed: Seed,
    set_name: String,
    split: Split,
    tys: [GgmlType; 3],
    triples: Vec<[GgmlType; 3]>,
}

/// What one swept header produced.
struct HeaderDone {
    out: Out,
    /// Whether the header was written (a block-mis-dividing triple is
    /// skipped by name).
    wrote: bool,
    /// Seconds of worker time.
    secs: f64,
}

impl Sweep {
    /// `seed`'s set re-checked against the manifest and opened, its layer's
    /// types read.
    fn prepare(name: &'static str, who: Who, seed: Seed, root: &Path) -> Result<Sweep, GateError> {
        let set_name = seed.set.name();
        let first = seed
            .set
            .check(root)
            .map_err(|e| format!("legs: the sweep's set {set_name} re-checked: {e}"))?;
        let split = Split::open(&first)?;
        let stacks = routed_stacks(&split)?;
        let Some(tys) = stacks.types.get(seed.layer).copied().flatten() else {
            return Err(format!(
                "legs: {name} {set_name}: the sweep's layer {} holds no routed stacks",
                seed.layer
            )
            .into());
        };
        Ok(Sweep {
            name,
            who,
            set_name,
            split,
            tys,
            triples: sweep_triples(who.adapter),
            seed,
        })
    }

    /// The `n`th swept header, `triple`, written under `dir` and planned
    /// through both arms at the rooms the original stub derived.
    fn header(
        &self,
        triple: [GgmlType; 3],
        n: usize,
        root: &Path,
        dir: &Path,
    ) -> Result<HeaderDone, String> {
        let started = Instant::now();
        let before = probes();
        let mut done = self
            .header_into(triple, n, root, dir)
            .map_err(|e| e.to_string())?;
        done.out.counts.calls += probes() - before;
        done.secs = started.elapsed().as_secs_f64();
        Ok(done)
    }

    fn header_into(
        &self,
        triple: [GgmlType; 3],
        n: usize,
        root: &Path,
        dir: &Path,
    ) -> Result<HeaderDone, GateError> {
        let mut done = HeaderDone {
            out: Out::new(),
            wrote: false,
            secs: 0.0,
        };
        let out = &mut done.out;
        let (name, set, tys) = (self.name, &self.set_name, self.tys);
        let here = dir.join(n.to_string());
        std::fs::create_dir_all(&here)?;
        let retyped = retype_split(
            &self.seed.set,
            root,
            &self.split,
            self.seed.layer,
            triple,
            &here,
            &mut out.lines,
        )?;
        // A skipped triple's named line was made by the writer.
        let Some(retyped) = retyped else {
            std::fs::remove_dir_all(&here)?;
            return Ok(done);
        };
        done.wrote = true;
        out.counts.retyped += 1;
        let split = Split::open(&retyped)?;
        let stacks = routed_stacks(&split)?;
        let mut pick = None;
        for attempt in self.who.adapter.arms(&split) {
            let arm = match attempt {
                Ok(arm) => arm,
                Err(Outcome::Refused(r)) => {
                    out.counts.refused(r.kind);
                    out.judged(
                        format!(
                            "legs: {name} {set} [{tys:?} → {triple:?}]: refused {}: {}",
                            r.kind, r.text
                        ),
                        true,
                    );
                    continue;
                }
                Err(Outcome::Red(text)) => {
                    out.judged(
                        format!("legs: {name} {set} [{tys:?} → {triple:?}]: {text}"),
                        false,
                    );
                    continue;
                }
                // An adapter's construction error is a refusal or red,
                // never a plan.
                Err(Outcome::Plan(_)) => {
                    unreachable!("an arm's construction error holds no plan")
                }
            };
            let word = format!("{}[{tys:?} → {triple:?}]", arm.name);
            let at = At {
                family: name,
                set,
                arm: &word,
                room: "-",
            };
            if !arm.whole
                && let Some(outcome) = &arm.unbudgeted
            {
                plan_line(out, self.who, &at, outcome, &stacks, &mut pick);
            }
            let Some((_, rooms)) = self.seed.rooms.iter().find(|(word, _)| *word == arm.name)
            else {
                continue;
            };
            for &bytes in rooms {
                let outcome = (arm.placed)(&budget(bytes));
                let room = bytes.to_string();
                let at = At { room: &room, ..at };
                plan_line(out, self.who, &at, &outcome, &stacks, &mut pick);
            }
        }
        drop(split);
        std::fs::remove_dir_all(&here)?;
        Ok(done)
    }
}

/// The retype sweep's single-type set: the K-quants around the legs' own,
/// the Q8_0 the per-type union admitted in a part no leg runs, and the
/// i-quant IQ4_XS.
const SWEEP_TYPES: [GgmlType; 5] = [
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::Q8_0,
    GgmlType::IQ4_XS,
];

/// L4's triples: every T×T×T of [`SWEEP_TYPES`] (125), the two gate≠up
/// mixed triples (127 in all) and the family's own
/// ([`Adapter::extra_triples`]) that the others do not hold.
fn sweep_triples(adapter: &dyn Adapter) -> Vec<[GgmlType; 3]> {
    let mut triples = Vec::new();
    for &g in &SWEEP_TYPES {
        for &u in &SWEEP_TYPES {
            for &d in &SWEEP_TYPES {
                triples.push([g, u, d]);
            }
        }
    }
    triples.push([GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q4_K]);
    triples.push([GgmlType::Q5_K, GgmlType::Q4_K, GgmlType::Q4_K]);
    for triple in adapter.extra_triples() {
        if !triples.contains(&triple) {
            triples.push(triple);
        }
    }
    triples
}

/// Write `set`'s shards into `dir` with layer `l`'s three routed stacks
/// retyped to `triple`, and return the first shard's path there. A type
/// whose block does not divide its stack's first dim skips the whole triple
/// with one named line (`None` back, never forced).
fn retype_split(
    set: &StubSet,
    root: &Path,
    split: &Split,
    l: usize,
    triple: [GgmlType; 3],
    dir: &Path,
    lines: &mut Vec<String>,
) -> Result<Option<PathBuf>, GateError> {
    let parts = ["gate", "up", "down"];
    let mut retype: Vec<(String, GgmlType)> = Vec::new();
    for (part, &ty) in parts.iter().zip(&triple) {
        let name = format!("blk.{l}.ffn_{part}_exps.weight");
        let (_, t) = split.find(&name).ok_or_else(|| {
            format!("the sweep's stack {name}: not in the file the plan was read from")
        })?;
        let Some((_, blck, _)) = gguf::ggml_type_info(ty.as_u32()) else {
            return Err(format!("{ty}: no block info, which a retype needs").into());
        };
        if !t.dims.first().copied().unwrap_or(0).is_multiple_of(blck) {
            lines.push(format!(
                "legs: retype {} {triple:?} skipped: {} dims {:?} hold no {ty} block",
                set.name(),
                name,
                t.dims
            ));
            return Ok(None);
        }
        retype.push((name, ty));
    }
    // The shards that hold the layer's stacks are rewritten; the rest are
    // symlinks to their stubs, the file names kept.
    let mut rewrite = Vec::new();
    for (name, _) in &retype {
        let (shard, _) = split.find(name).expect("the stack was just found");
        if !rewrite.contains(&shard) {
            rewrite.push(shard);
        }
    }
    let mut first = None;
    for (i, shard) in set.paths(root).iter().enumerate() {
        let path = dir.join(
            shard
                .file_name()
                .ok_or_else(|| format!("{}: no file name", shard.display()))?,
        );
        if rewrite.contains(&i) {
            let gg = split
                .shard(i)
                .ok_or_else(|| format!("{}: no shard {}", shard.display(), i))?;
            let kvs: Vec<(String, gguf::Value)> = gg
                .iter_kv()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect();
            let mut tensors = Vec::new();
            for t in gg.iter_tensors() {
                let ty = retype
                    .iter()
                    .find(|(name, _)| *name == t.name)
                    .map_or(t.ty, |&(_, ty)| ty);
                let (_, blck, tsz) = gguf::ggml_type_info(ty.as_u32())
                    .ok_or_else(|| format!("{}: {ty} has no block info", t.name))?;
                let bytes = gguf::tensor_bytes(blck, tsz, &t.dims)
                    .ok_or_else(|| format!("{}: its bytes overflow u64", t.name))?;
                tensors.push(TensorDecl {
                    name: t.name.clone(),
                    dims: t.dims.clone(),
                    type_id: ty.as_u32(),
                    nbytes: bytes,
                });
            }
            let layout = Layout::new(&kvs, tensors).map_err(|e| e.to_string())?;
            let len = layout.file_len();
            let mut file = File::create(&path)?;
            drop(Writer::new(&mut file, layout).map_err(|e| e.to_string())?);
            file.set_len(len)?;
        } else {
            let _ = std::fs::remove_file(&path);
            std::os::unix::fs::symlink(shard, &path)
                .map_err(|e| format!("{} → {}: {e}", shard.display(), path.display()))?;
        }
        if i == 0 {
            first = Some(path);
        }
    }
    Ok(first)
}

/// The counts the gate reads off `plan`.
fn view_of(plan: &Plan) -> PlanView {
    PlanView {
        n_l: plan.n_l.clone(),
        tier_n_l: plan.tier_n_l.clone(),
        card_expert_bytes: plan.cards.first().map_or(0, |c| c.expert_bytes),
        host_experts: plan.host.experts,
        nvme_expert_bytes: plan.host.nvme_expert_bytes,
        loader: None,
    }
}

/// A seat default's slot context for an arm: the choice's, or the fit's
/// floor where the machine holds not even that
/// (`generate::FloorShort`), so the arm's plan there names the refusal
/// itself; any other error stands.
fn floor_or(chosen: Result<u64, GateError>) -> Result<u64, GateError> {
    chosen.or_else(
        |e| match e.downcast_ref::<bloomery_gpu_gates::generate::FloorShort>() {
            Some(short) => Ok(short.floor),
            None => Err(e),
        },
    )
}

/// The outcome of a planner error: a named refusal L2 counts — a
/// [`PlacementError`] the planner stated by name (the read and budget
/// refusals), or a context refusal of the qwen plan wrappers — else red
/// with its text.
fn outcome_of(e: &(dyn StdError + 'static)) -> Outcome {
    let mut cur = Some(e);
    while let Some(c) = cur {
        if let Some(p) = c.downcast_ref::<PlacementError>() {
            return refused_of(p);
        }
        if let Some(p) = c.downcast_ref::<Place35Error>() {
            return place35_of(p);
        }
        if let Some(p) = c.downcast_ref::<Place3Error>() {
            return place3_of(p);
        }
        // The bodies' own read refusals — a site's form (`GpuError::Shape`),
        // a metadata key, a GGUF the reader cannot take — are stated by name
        // before any plan; the machine state (driver, module, launch) is
        // red, and a plan refusal keeps walking its source below.
        if let Some(g) = c.downcast_ref::<bloomery_gpu::GpuError>() {
            match g {
                bloomery_gpu::GpuError::Shape { .. }
                | bloomery_gpu::GpuError::Metadata { .. }
                | bloomery_gpu::GpuError::Load(_) => {
                    return Outcome::Refused(Refusal {
                        kind: "reader",
                        floor: None,
                        text: g.to_string(),
                    });
                }
                _ => {}
            }
        }
        cur = c.source();
    }
    Outcome::Red(e.to_string())
}

/// A family planner's error as an outcome: the first `P` in `e`'s chain as
/// the family's `place_of` reads it, else [`outcome_of`]'s reading. A
/// family error that wraps the placement's own transparently hides it from
/// [`outcome_of`]'s walk, so a refusal the placement names would read red.
fn outcome_by<P: StdError + 'static>(e: &GateError, place_of: fn(&P) -> Outcome) -> Outcome {
    let mut cur: Option<&(dyn StdError + 'static)> = Some(e.as_ref());
    while let Some(c) = cur {
        if let Some(p) = c.downcast_ref::<P>() {
            return place_of(p);
        }
        cur = c.source();
    }
    outcome_of(e.as_ref())
}

/// A [`PlacementError`] as an outcome: the read refusals a real census file
/// can state and the budget floor are named; the machine state and the
/// invariant breaks (a card another process holds, a plan that breaks its
/// own checks) are red.
fn refused_of(p: &PlacementError) -> Outcome {
    let (kind, floor) = match p {
        PlacementError::CardBudgetFloor { floor, .. } => ("card-budget-floor", Some(*floor)),
        PlacementError::Unimplemented(_) => ("unimplemented", None),
        PlacementError::HostRoomFloor { .. } => ("host-room-floor", None),
        PlacementError::Unclassified { .. } => ("unclassified", None),
        PlacementError::Metadata { .. } | PlacementError::Tensor { .. } => ("metadata", None),
        PlacementError::HostRoom(_) => ("host-room", None),
        PlacementError::NoCardFormat { .. } => ("no-card-format", None),
        // A census file whose dense trunk alone passes the visible card's
        // free reading (a BF16 set on the gate machine): the planner's own
        // named refusal of the card, not a defect.
        PlacementError::CardFreeFloor(_) => ("card-free-floor", None),
        // A census file whose routed stacks the card rule leaves on the host
        // in a type or width no host kernel runs (a BF16 set): the planner's
        // own named refusal of the file on this machine, not a defect.
        PlacementError::NoHostKernel { .. } => ("no-host-kernel", None),
        _ => return Outcome::Red(p.to_string()),
    };
    Outcome::Refused(Refusal {
        kind,
        floor,
        text: p.to_string(),
    })
}

/// A qwen35moe/qwen4exp plan wrapper's refusal as an outcome: its context
/// refusals are named, its invariant breaks red.
fn place35_of(p: &Place35Error) -> Outcome {
    let kind = match p {
        Place35Error::Positions { .. } => "positions",
        Place35Error::PastTrained { .. } => "past-trained",
        Place35Error::Tier { .. } | Place35Error::TierHost { .. } => "tier",
        Place35Error::Placement(e) => return refused_of(e),
        _ => return Outcome::Red(p.to_string()),
    };
    Outcome::Refused(Refusal {
        kind,
        floor: None,
        text: p.to_string(),
    })
}

/// A qwen3moe plan wrapper's refusal as an outcome, as [`place35_of`].
fn place3_of(p: &Place3Error) -> Outcome {
    let kind = match p {
        Place3Error::Positions => "positions",
        Place3Error::Placement(e) => return refused_of(e),
        Place3Error::Broken(_) => return Outcome::Red(p.to_string()),
    };
    Outcome::Refused(Refusal {
        kind,
        floor: None,
        text: p.to_string(),
    })
}
