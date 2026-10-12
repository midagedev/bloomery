//! The common owner of gate_routed_legs: the census reading and checking,
//! the arch → family dispatch, the card-room derivation, the retype sweep
//! and the checks L1–L5. A family is added by one adapter file
//! (`routed_legs/<family>.rs`, holding only what the family forces — how its
//! production `--place`-unset defaults plan one split) and one row in
//! [`family_of`]; no check and no per-family branch lives here. Every stub
//! is planned from its header alone — the census writer's tensor bytes are
//! all zero — so no planner step that reads data changes an answer.

use std::collections::BTreeMap;
use std::error::Error as StdError;
use std::fs::File;
use std::path::{Path, PathBuf};

use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::{checks_failed, verdict};
use gguf::Split;
use gguf::quant::GgmlType;
use gguf::write::{Layout, TensorDecl, Writer};
use model::arch::models::Arch;
use model::arch::qwen3moe::place::PlaceError as Place3Error;
use model::arch::qwen35moe::place::PlaceError as Place35Error;
use model::placement::accepts::{Program, leg_arms, leg_runs, program_of};
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

// A file the bin loads by `#[path]` owns its directory, so the adapters'
// paths are named here rather than resolved under `routed_legs/`.
#[path = "routed_legs/qwen35moe.rs"]
mod qwen35moe;
#[path = "routed_legs/qwen38.rs"]
mod qwen38;
#[path = "routed_legs/qwen3moe.rs"]
mod qwen3moe;

/// The families this round plans, by the file's architecture (read with the
/// planners' own reader, `model::arch::spec`, which reads `glm5-next` and
/// `clef` too). Part 2 adds glm5next, mimo2 and deepseek41 rows.
fn family_of(arch: Arch) -> Option<&'static dyn Adapter> {
    match arch {
        Arch::Qwen3Moe => Some(&qwen3moe::ADAPTER),
        Arch::Qwen35Moe => Some(&qwen35moe::ADAPTER),
        Arch::Qwen4Exp => Some(&qwen38::ADAPTER),
        Arch::Qwen35 | Arch::Glm5Next | Arch::MiMo2 | Arch::Deepseek41 | Arch::Deepseek4 => None,
    }
}

/// Why a set the gate does not plan is skipped: the Qwen3.5 dense trunk
/// routes nothing, the part 2 families land in a later round, and an
/// architecture no program runs is never planned as a model.
fn skip_of(arch: Arch) -> &'static str {
    match arch {
        Arch::Qwen35 => "dense trunk",
        Arch::Glm5Next | Arch::MiMo2 | Arch::Deepseek41 | Arch::Deepseek4 => "part 2's arch",
        Arch::Qwen3Moe | Arch::Qwen35Moe | Arch::Qwen4Exp => "no program",
    }
}

/// One plan the gate holds: the layers' card counts and the byte terms the
/// rooms derive from.
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
}

/// A named planner refusal — an outcome L2 counts by kind, never red. Its
/// `floor` is the refusal's own where it carries one (a
/// `CardBudgetFloor`), which the room derivation reads.
struct Refusal {
    kind: &'static str,
    floor: Option<u64>,
    text: String,
}

/// One plan attempt's outcome.
enum Outcome {
    Plan(PlanView),
    Refused(Refusal),
    /// Any other error: red, with its text.
    Red(String),
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
/// split, one arm per binary that opens the family's files.
trait Adapter {
    /// The family word the lines carry.
    fn name(&self) -> &'static str;
    /// The arms on `split`, each its own attempt (one arm's refusal does not
    /// stop the other).
    fn arms(&self, split: &Split) -> Vec<Result<Arm, Outcome>>;
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
    retyped: u64,
    refusals: BTreeMap<&'static str, u64>,
}

impl Counts {
    /// One more refusal of `kind`.
    fn refused(&mut self, kind: &'static str) {
        *self.refusals.entry(kind).or_default() += 1;
    }

    /// The family's counts line.
    fn line(&self, family: &str, traps: usize) -> String {
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
        format!(
            "legs: {family}: sets {} plans {} partial {} whole {} floors {} refusals {refusals} \
             retyped {} traps {traps}",
            self.sets, self.plans, self.partial, self.whole, self.floors, self.retyped
        )
    }
}

/// One family's state across the run: its counts, the program its files run
/// and its sweep seed.
struct Family {
    counts: Counts,
    program: Program,
    /// The smallest set with a carded layer, that layer and its arms' rooms.
    seed: Option<Seed>,
    /// The sweep triples the retyping wrote (the trap count's domain).
    swept: Vec<[GgmlType; 3]>,
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

/// The gate's run: read the census, plan every B1 family's sets through its
/// production defaults at derived rooms, sweep one set's retyped headers,
/// and answer green only when every carded layer is one its file's
/// architecture runs and every family's sweep holds a trap.
pub(super) fn run() -> Result<(), GateError> {
    bloomery_levers::at_main(&[])?;
    let (manifest, root) = args()?;
    let sets = Manifest::read(&manifest)
        .map_err(|e| format!("the census manifest {}: {e}", manifest.display()))?;
    println!(
        "legs: {} sets under the manifest {} (stubs under {})",
        sets.sets().len(),
        manifest.display(),
        root.display()
    );
    let mut ok = true;
    let mut skips: BTreeMap<String, u64> = BTreeMap::new();
    let mut families: BTreeMap<&'static str, Family> = BTreeMap::new();
    for set in sets.sets() {
        let name = set.name();
        // A draft set's first shard's basename — the census writer's naming.
        // Drafts carry their model's architecture but are never planned as a
        // model.
        let first_shard = set.shards[0].file.rsplit('/').next().unwrap_or("");
        if first_shard.starts_with("mtp-") || first_shard.starts_with("dflash-") {
            *skips.entry("draft".to_string()).or_default() += 1;
            println!("legs: skip {name}: draft");
            continue;
        }
        let first = match set.check(&root) {
            Ok(first) => first,
            Err(e) => {
                println!("legs: census {name}: {e} {}", verdict(false));
                ok = false;
                continue;
            }
        };
        let split = match Split::open(&first) {
            Ok(split) => split,
            Err(e) => {
                // A stub that checks against its row but the reader cannot
                // open (a quant type this build's table does not size) is a
                // read refusal, not a gate defect: the census holds users'
                // files of every quant.
                *skips.entry("no reader".to_string()).or_default() += 1;
                println!("legs: skip {name}: no reader: {e}");
                continue;
            }
        };
        let arch = match model::arch::spec(&split) {
            Ok(read) => read.spec.arch,
            Err(e) => {
                *skips.entry("no reader".to_string()).or_default() += 1;
                println!("legs: skip {name}: no reader: {e}");
                continue;
            }
        };
        let Some(adapter) = family_of(arch) else {
            *skips.entry(skip_of(arch).to_string()).or_default() += 1;
            println!("legs: skip {name}: {}", skip_of(arch));
            continue;
        };
        let program =
            program_of(arch).expect("every architecture the dispatch plans has a program");
        let family = families.entry(adapter.name()).or_insert_with(|| Family {
            counts: Counts::default(),
            program,
            seed: None,
            swept: Vec::new(),
        });
        family.counts.sets += 1;
        let stacks = routed_stacks(&split)?;
        let mut rooms: Vec<(&'static str, [u64; 3])> = Vec::new();
        let mut carded: Option<usize> = None;
        for attempt in adapter.arms(&split) {
            let arm = match attempt {
                Ok(arm) => arm,
                Err(Outcome::Refused(r)) => {
                    family.counts.refused(r.kind);
                    println!(
                        "legs: {} {name}: refused {}: {} {}",
                        adapter.name(),
                        r.kind,
                        r.text,
                        verdict(true)
                    );
                    continue;
                }
                Err(Outcome::Red(text)) => {
                    println!(
                        "legs: {} {name}: {} {}",
                        adapter.name(),
                        text,
                        verdict(false)
                    );
                    ok = false;
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
                family.counts.whole += 1;
                println!(
                    "legs: {} {name} {}: room -: whole load {}",
                    adapter.name(),
                    arm.name,
                    verdict(true)
                );
            } else if let Some(outcome) = &arm.unbudgeted {
                ok &= plan_line(family, &at, outcome, &stacks, &mut carded);
            } else {
                println!(
                    "legs: {} {name} {}: an arm that answered neither whole nor a plan {}",
                    adapter.name(),
                    arm.name,
                    verdict(false)
                );
                ok = false;
            }
            // The rooms: the floor from the placed planner's own refusal,
            // then floor + f·E.
            let e = expert_span(&arm, &stacks);
            let (floor, arm_rooms) = match rooms_of(arm.placed.as_ref(), e) {
                Ok(rooms) => rooms,
                Err(text) => {
                    println!(
                        "legs: {} {name} {}: the rooms did not derive: {text} {}",
                        adapter.name(),
                        arm.name,
                        verdict(false)
                    );
                    ok = false;
                    continue;
                }
            };
            family.counts.floors += 1;
            println!(
                "legs: {} {name} {}: floor {floor}",
                adapter.name(),
                arm.name
            );
            rooms.push((arm.name, arm_rooms));
            for &bytes in &arm_rooms {
                let outcome = (arm.placed)(&budget(bytes));
                let room = bytes.to_string();
                let at = At { room: &room, ..at };
                ok &= plan_line(family, &at, &outcome, &stacks, &mut carded);
            }
        }
        // The sweep seed: the smallest set that cards a layer.
        if let Some(layer) = carded {
            let total = set.shards.iter().map(|s| s.file_bytes).sum();
            if family.seed.as_ref().is_none_or(|seed| total < seed.total) {
                family.seed = Some(Seed {
                    set: set.clone(),
                    total,
                    layer,
                    rooms,
                });
            }
        }
    }
    // The retype sweep, then L3 and L5 per family.
    for (_, family) in families.iter_mut() {
        if let Some(seed) = family.seed.take() {
            ok &= sweep(family, seed, &root)?;
        }
        let name = family_name(family.program);
        let traps = trap_count(family.program, &family.swept);
        if traps == 0 {
            println!("legs: {name}: the sweep has no trap {}", verdict(false));
            ok = false;
        }
        if family.counts.partial == 0 {
            println!("legs: {name}: vacuous: no partial plan {}", verdict(false));
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
            family
                .counts
                .line(name, trap_count(family.program, &family.swept))
        );
    }
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

/// One plan's line (L1 over it, L3's partial count, the carded-layer pick):
/// `legs: <family> <set> <arm> room <room>: <what> PASS|FAIL`, each FAIL
/// line naming family, set, arm, room, layer and the three types. Returns
/// whether the plan held.
fn plan_line(
    family: &mut Family,
    at: &At<'_>,
    outcome: &Outcome,
    stacks: &Stacks,
    carded: &mut Option<usize>,
) -> bool {
    match outcome {
        Outcome::Refused(r) => {
            family.counts.refused(r.kind);
            println!(
                "legs: {} {} {}: room {}: refused {} {}",
                at.family,
                at.set,
                at.arm,
                at.room,
                r.kind,
                verdict(true)
            );
            true
        }
        Outcome::Red(text) => {
            println!(
                "legs: {} {} {}: room {}: {} {}",
                at.family,
                at.set,
                at.arm,
                at.room,
                text,
                verdict(false)
            );
            false
        }
        Outcome::Plan(view) => {
            let fails = l1(at, view, stacks, family.program);
            let routed = (0..view.n_l.len())
                .filter(|&l| stacks.types.get(l).is_some_and(Option::is_some))
                .count();
            let held = (0..view.n_l.len())
                .filter(|&l| carded_layer(view, l))
                .count();
            if held > 0 && held < routed {
                family.counts.partial += 1;
            }
            if held > 0 && carded.is_none() {
                *carded = Some(
                    (0..view.n_l.len())
                        .find(|&l| carded_layer(view, l))
                        .expect("held > 0 names a carded layer"),
                );
            }
            family.counts.plans += 1;
            println!(
                "legs: {} {} {}: room {}: placed {held} of {routed} routed layers carded {}",
                at.family,
                at.set,
                at.arm,
                at.room,
                verdict(fails.is_empty())
            );
            for fail in &fails {
                println!("legs:   {fail} {}", verdict(false));
            }
            fails.is_empty()
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

/// The rooms an arm plans at: the card budget floor — read off the
/// `CardBudgetFloor` refusal the placed planner states at a zero budget —
/// and `floor + f·E` for f ∈ {1/4, 1/2, 3/4}. Derived from the planner's
/// own refusal and the plan's own bytes, never from a constant.
fn rooms_of(placed: &dyn Fn(&PlanLevers) -> Outcome, e: u64) -> Result<(u64, [u64; 3]), String> {
    let floor = match placed(&budget(0)) {
        Outcome::Refused(r) => r.floor,
        Outcome::Plan(_) => {
            return Err(
                "a zero card budget planned: the floor refusal is missing — the placed planner \
                 never refused the trunk"
                    .to_string(),
            );
        }
        Outcome::Red(text) => return Err(text),
    };
    let Some(floor) = floor else {
        return Err(
            "the zero-budget refusal names no floor: not the planner's own CardBudgetFloor"
                .to_string(),
        );
    };
    Ok((floor, [floor + e / 4, floor + e / 2, floor + 3 * e / 4]))
}

/// The levers of a forced room: one card budget, everything else default.
fn budget(bytes: u64) -> PlanLevers {
    PlanLevers {
        card_budget_bytes: Some(bytes),
        ..PlanLevers::default()
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

/// The family word of `program`, for the lines the adapter's own name does
/// not reach (the counts and the check lines).
fn family_name(program: Program) -> &'static str {
    match program {
        Program::Qwen3moeBody => "qwen3moe",
        Program::Qwen35Body => "qwen35moe",
        Program::Qwen38Body => "qwen38",
        Program::Glm5nextBody => "glm5next",
        Program::Mimo2Body => "mimo2",
        Program::Deepseek41Chain => "deepseek41",
    }
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

/// The retype sweep (L4): `seed`'s set with its carded layer's three routed
/// stacks set to each sweep triple, planned through both arms at the rooms
/// the original stub derived. Every retyped split goes into a fresh temp
/// directory — the shards holding the layer rewritten (their kvs and
/// tensors verbatim but the three stacks' type and recomputed bytes, over a
/// sparse file of the layout's length), every other shard a symlink to its
/// stub, the file names kept so `Split::open` finds them. Returns whether
/// every swept plan held.
fn sweep(family: &mut Family, seed: Seed, root: &Path) -> Result<bool, GateError> {
    let name = family_name(family.program);
    let set = seed.set.name();
    let first = seed
        .set
        .check(root)
        .map_err(|e| format!("legs: the sweep's set {set} re-checked: {e}"))?;
    let split = Split::open(&first)?;
    let stacks = routed_stacks(&split)?;
    let Some(tys) = stacks.types.get(seed.layer).copied().flatten() else {
        return Err(format!(
            "legs: {name} {set}: the sweep's layer {} holds no routed stacks",
            seed.layer
        )
        .into());
    };
    let dir =
        std::env::temp_dir().join(format!("bloomery-routedlegs-{}-sweep", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    let adapter = sweep_adapter(family.program);
    let mut ok = true;
    for triple in sweep_triples() {
        let Some(retyped) = retype_split(&seed.set, root, &split, seed.layer, triple, &dir)? else {
            // The named skip line was printed by the writer.
            continue;
        };
        family.counts.retyped += 1;
        family.swept.push(triple);
        let split = Split::open(&retyped)?;
        let stacks = routed_stacks(&split)?;
        let mut pick = None;
        for attempt in adapter.arms(&split) {
            let arm = match attempt {
                Ok(arm) => arm,
                Err(Outcome::Refused(r)) => {
                    family.counts.refused(r.kind);
                    println!(
                        "legs: {name} {set} [{tys:?} → {triple:?}]: refused {}: {} {}",
                        r.kind,
                        r.text,
                        verdict(true)
                    );
                    continue;
                }
                Err(Outcome::Red(text)) => {
                    println!(
                        "legs: {name} {set} [{tys:?} → {triple:?}]: {} {}",
                        text,
                        verdict(false)
                    );
                    ok = false;
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
                set: &set,
                arm: &word,
                room: "-",
            };
            if !arm.whole {
                if let Some(outcome) = &arm.unbudgeted {
                    ok &= plan_line(family, &at, outcome, &stacks, &mut pick);
                }
            }
            let Some((_, rooms)) = seed.rooms.iter().find(|(word, _)| *word == arm.name) else {
                continue;
            };
            for &bytes in rooms {
                let outcome = (arm.placed)(&budget(bytes));
                let room = bytes.to_string();
                let at = At { room: &room, ..at };
                ok &= plan_line(family, &at, &outcome, &stacks, &mut pick);
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    Ok(ok)
}

/// The adapter of `program`'s family, for the sweep's arm construction.
fn sweep_adapter(program: Program) -> &'static dyn Adapter {
    match program {
        Program::Qwen3moeBody => &qwen3moe::ADAPTER,
        Program::Qwen35Body => &qwen35moe::ADAPTER,
        Program::Qwen38Body => &qwen38::ADAPTER,
        Program::Glm5nextBody | Program::Mimo2Body | Program::Deepseek41Chain => {
            unreachable!("part 2's families have no adapter this round")
        }
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

/// L4's triples: every T×T×T of [`SWEEP_TYPES`] (125) and the two gate≠up
/// mixed triples, 127 in all.
fn sweep_triples() -> Vec<[GgmlType; 3]> {
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
            println!(
                "legs: retype {} {triple:?} skipped: {} dims {:?} hold no {ty} block",
                set.name(),
                name,
                t.dims
            );
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
                let bytes = t.dims[1..]
                    .iter()
                    .try_fold(tsz * (t.dims[0] / blck), |acc, d| acc.checked_mul(*d))
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
    }
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
        Place3Error::Tier { .. } => "tier",
        Place3Error::Placement(e) => return refused_of(e),
        _ => return Outcome::Red(p.to_string()),
    };
    Outcome::Refused(Refusal {
        kind,
        floor: None,
        text: p.to_string(),
    })
}
