//! V4.1's adaptive expert residency clauses on the real model
//! (`bloomery_gpu_deepseek41::swap`, `BLOOMERY_RESIDENCY`), run by
//! `gate_ds41_callstream` on its load: plan (b′) on both cards (`--place
//! bp`: the stage on the A6000, the expert tier on the 3090), or plan (a) on
//! the A6000 alone (`--place a`), the residency
//! machine over the stage card's routed stacks at `mid-p40-s1`, set there
//! (the lever itself is refused, so the environment cannot move it). The
//! stage's seed is the plan's id prefix. `refuse` runs before that load, the
//! rest on it before the callstream clauses.
//!
//! A history is: the session cleared (the residency back to its seed), the
//! first [`PROMPT`] ids of the prose corpus as one prompt call, then
//! [`STEPS`] greedy steps, each step's argmax and the FNV-1a 64 of its logits
//! row read. Clauses; each names its mutant, and the machine's own rules
//! (c1, c3) have theirs in `gate_swap`:
//!
//! - `refuse` (the host-set refusal at load): a machine whose host holds one
//!   byte less than the churn pool needs — the stage card's experts past the
//!   pinned ones, which the host set must hold too — is refused by name
//!   before anything loads, within [`REFUSE_BOUND_S`] (mutant: the open's
//!   churn pool check removed).
//! - `c6` (a pass counts its kept rows only): a pair pass `[t, t]` after the
//!   prompt with one row kept, then a boundary, leaves the rule equal to one
//!   step of `t` then a boundary; the same pair with both rows kept does not
//!   (mutant: the session's commit keeps every row of the pass).
//! - `c1` (green-only): the history twice gives the same tokens and logits,
//!   and flips land on at least one layer.
//! - `transform`: after the history, every expert the machine admitted holds
//!   in its slot, part by part, the bytes a static load uploads for it —
//!   the source's Q3_K gate and up, the Q4_K down — though the host reads
//!   the gate and up from the r8 sidecar (mutant: no unpack on the card).
//! - `resident` (`host_resident` reads the truth): every stage expert past
//!   the pinned ones and every host expert of each stage layer is
//!   host-resident, and no pinned one is — they are not in the host set
//!   (mutant: `host_resident` answers true).
//! - `tier` (bp): every tier entry of the host map is where the load put it
//!   after the histories, though flips landed on tier layers (mutant: the
//!   machine's layout does not hold the tier's experts away). Plan (a) has no
//!   tier card: there the clause asks only that the host map holds no tier
//!   entry, at the load and after the histories.
//! - `c3` (green-only): the history with the machine's copy stream held by a
//!   host flag for [`HOLD`] from before its first pass gives the first run's
//!   tokens and logits.
//! - `c7`: a residency reset after the held history brings every layer's
//!   live set back to its seed (`diff` 0, and the ledger), and lets go of no
//!   host byte (`dropped_bytes` 0: the churn pool stays in the host set for
//!   the model's life) (mutant: the reset sends the admitted experts home
//!   and leaves the seed's off the card).
//! - `keep` (serve's cache miss): the reset a server makes when a request
//!   misses its cache ([`runtime::Target::reset`], the seat's reset) leaves
//!   the residency where use took it — the live sets and the rule as before
//!   it, no `residency reset` (mutant: that reset resets the residency).
//! - `table`: after the held history, the stage card's copy of the map read
//!   back (`generate::Residence`, `CardTable`) is off neither the host map
//!   nor the machine's ledger, names no slot twice in a layer, and is off
//!   the map the history started from by at least one entry and at most two
//!   a landed flip (each moves its admitted expert onto a slot and its victim
//!   off) (mutant: a landing writes the admitted expert's word and not the
//!   victim's).
//! - `static` (`--place a`): the residency computes the placement its card
//!   copy holds. The copy as the held history left it is read; after the
//!   seat's reset the history's prompt is fed the serve's way
//!   (`generate::ServeFeed`), and the row of its last id (row 0) is kept,
//!   the boundaries before that row landing no flip (fed again, up to
//!   `SETTLE_FEEDS` feeds, until one lands none; named). After the
//!   teardown, one more load — residency off, host streaming off as the
//!   history ran, each routed layer's card experts the read copy's
//!   (`Loaded::open_edited`, `generate::place_table`; named: no machine, and
//!   the card holds the copy's sets) — is fed the same ids the same way: its
//!   row 0 equals the residency's bit for bit (mutants: a flip's copy
//!   written one slot below its slot; the admitted expert's bytes left
//!   unconverted on the card). Under plan (b′) the streaming clauses follow
//!   on the same load, so the clause runs under `--place a` alone.
//! - `passes` (green-only): a history's boundaries end, in order, no pass,
//!   the prompt call (one pass, 0 rows kept) and each step (1 kept), the
//!   last step's own included: its boundary is made ahead of its readback.
//!
//! The gate runs with the r8 sidecar (`BLOOMERY_R8` on, its default): without
//! it no flip unpacks, and `transform` is red by name.
//!
//! Tiers (`crate::ds41_tier`, `bloomery_gpu_gates::tier`): every equality and structure clause is
//! self-consistency and runs on a fixture. What the real file's routing skew makes true — flips
//! land, an expert is admitted, the copy of the map moves off its start, the live sets leave the
//! seed — is one file-bound premise ([`skew`]); a fixture's generated router has no skew, so the
//! fixture tier defers it and the clauses' equalities hold the rest of each arm.

use crate::ds41_open::{open, open_edited};
use std::path::Path;
use std::time::{Duration, Instant};

use app::{Session, SessionError};
use bloomery_gpu::HostFlags;
use bloomery_gpu::host::PassKind;
use bloomery_gpu::host::slots::Slot;
use bloomery_gpu::host::swap::{Residency, SlotState, SwapSource};
use bloomery_gpu_deepseek41::body::{Body, OpenCfg};
use bloomery_gpu_deepseek41::swap;
use bloomery_gpu_gates::generate::{Place, Residence, place_table};
use bloomery_gpu_gates::record;
use bloomery_gpu_gates::tier::{self, Tag};
use bloomery_gpu_gates::{Fnv1a64, GateError, data_dir, verdict};
use model::arch::deepseek41::place::PlanInputs;
use model::placement::{Machine, workstation};
use runtime::swaprule::SwapRule;
use runtime::{Target, Verify, Want};

pub use crate::residency_clauses::StaticProbe;

const NAME: &str = "residency clauses";
/// The residency the gate runs: 40 pinned, one spare a layer.
const RESIDENCY: Residency = Residency::Mid {
    pinned: 40,
    spares: 1,
};
/// Prompt positions: one prompt call, one batch.
const PROMPT: usize = 64;
/// Greedy steps after the prompt: 24 planning boundaries of the rule.
const STEPS: usize = 96;
/// How long c3 holds the copy stream: past several landings at ~25–40
/// ms a step.
const HOLD: Duration = Duration::from_secs(1);
/// A refusal before the load reads the files' headers only.
const REFUSE_BOUND_S: f64 = 120.0;

/// The first `n` ids of `$BLOOMERY_DATA/engram/corpus-prose.ids`.
fn prose(n: usize) -> Result<Vec<u32>, GateError> {
    let path = data_dir().join("engram").join("corpus-prose.ids");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let ids = text
        .split_whitespace()
        .take(n)
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    if ids.len() < n {
        return Err(format!("{}: {} ids, the gate reads {n}", path.display(), ids.len()).into());
    }
    Ok(ids)
}

/// The premise every flip-dependent clause shares: the history's flips land, an expert is
/// admitted, the card's copy moves off its start, the live sets leave the seed. It is the real
/// file's routing skew (file-bound): `true` when this tier asserts it, `false` when the fixture
/// tier leaves it to the real one, its line printed once.
fn skew() -> Result<bool, GateError> {
    static SKEW: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if let Some(&s) = SKEW.get() {
        return Ok(s);
    }
    let s = tier::run_clause(
        "c1, transform, keep, table: flips landed and an expert admitted (the file's routing skew)",
        Tag::FileBound,
    )?;
    Ok(*SKEW.get_or_init(|| s))
}

/// Milliseconds since `t`.
fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// What a history saw: every step's argmax and logits FNV, each
/// boundary's ended pass (its kind and kept rows), and the flips that
/// landed; and the host map's stage view after its clear, where those
/// flips started from.
struct History {
    tokens: Vec<u32>,
    fnvs: Vec<u64>,
    passes: Vec<(PassKind, usize)>,
    landed: usize,
    start: Vec<u32>,
}

/// Two histories are one when they saw the same: the map one starts from
/// is no part of it, since a reset puts the seed back in whichever slots
/// are free.
impl PartialEq for History {
    fn eq(&self, other: &History) -> bool {
        (&self.tokens, &self.fnvs, &self.passes, self.landed)
            == (&other.tokens, &other.fnvs, &other.passes, other.landed)
    }
}

/// The history from a clear ([`Session::clear`]), the machine's copy
/// stream held by `hold` from before the prompt until [`HOLD`] later
/// when given.
fn history(
    s: &mut Session<Body>,
    ids: &[u32],
    hold: Option<&HostFlags>,
) -> Result<History, GateError> {
    let t_clear = Instant::now();
    s.clear()?;
    let clear_ms = ms(t_clear);
    take_passes(s)?;
    let start = s.model().body(NAME)?.hybrid().slots().stage_view();
    if let Some(flags) = hold {
        flags.clear(0)?;
        let m = s.model();
        let machine = m
            .body(NAME)?
            .hybrid()
            .swap()
            .ok_or("the load runs no residency machine")?;
        flags.enqueue_wait(machine.copy_stream(), 0)?;
    }
    let mut walls = Vec::with_capacity(STEPS + 1);
    let mut run = |s: &mut Session<Body>| -> Result<History, GateError> {
        let t = Instant::now();
        let mut next = s.prompt(ids, Want::Argmax)?.argmax();
        walls.push(ms(t));
        let mut h = History {
            tokens: Vec::with_capacity(STEPS),
            fnvs: Vec::with_capacity(STEPS),
            passes: Vec::new(),
            landed: 0,
            start: Vec::new(),
        };
        for _ in 0..STEPS {
            let t = Instant::now();
            let out = s.step(next, Want::Logits)?;
            walls.push(ms(t));
            if let runtime::Out::Logits { row, .. } = out {
                h.fnvs.push(Fnv1a64::default().f32s(row).value());
            }
            next = out.argmax();
            h.tokens.push(next);
        }
        Ok(h)
    };
    let mut h = match hold {
        None => run(s)?,
        Some(flags) => std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(HOLD);
                // A raise that fails leaves the stream held: the step's
                // deadline names it.
                let _ = flags.raise(0);
            });
            run(s)
        })?,
    };
    h.start = start;
    let passes = take_passes(s)?;
    h.landed = passes.iter().map(|(_, r)| r.landed).sum();
    let late: usize = passes.iter().map(|(_, r)| r.late).sum();
    let steps = walls.get(1..).unwrap_or(&[]);
    let (slowest, max) = steps
        .iter()
        .copied()
        .enumerate()
        .fold((0, 0.0f64), |a, (i, w)| if w > a.1 { (i, w) } else { a });
    println!(
        "history: clear {clear_ms:.1} ms, prompt {:.1} ms, {} steps mean {:.2} ms, slowest \
         step {slowest} {max:.1} ms, {} boundaries, {} flips landed, {late} late",
        walls.first().copied().unwrap_or(f64::NAN),
        steps.len(),
        steps.iter().sum::<f64>() / steps.len().max(1) as f64,
        passes.len(),
        h.landed
    );
    h.passes = passes.iter().map(|&(k, r)| (k, r.kept)).collect();
    Ok(h)
}

/// The boundaries' reports since the last take.
fn take_passes(
    s: &mut Session<Body>,
) -> Result<Vec<(PassKind, bloomery_gpu::host::swap::PassReport)>, GateError> {
    let (_, _, b) = s.model_mut().body_parts(NAME)?;
    Ok(b.take_residency_passes())
}

/// Each seed layer's live experts, sorted, in `seeds`' order.
fn live_sets(s: &Session<Body>, seeds: &[(usize, Vec<u32>)]) -> Result<Vec<Vec<u32>>, GateError> {
    let m = s.model();
    let b = m.body(NAME)?;
    let machine = b
        .hybrid()
        .swap()
        .ok_or("the load runs no residency machine")?;
    Ok(seeds
        .iter()
        .map(|(l, _)| {
            let mut live: Vec<u32> = machine
                .ledger()
                .row(*l)
                .unwrap_or(&[])
                .iter()
                .filter_map(|st| match st {
                    SlotState::Live(e) => Some(*e),
                    _ => None,
                })
                .collect();
            live.sort_unstable();
            live
        })
        .collect())
}

/// The machine's rule as it stands.
fn rule_of(s: &Session<Body>) -> Result<SwapRule, GateError> {
    Ok(s.model()
        .body(NAME)?
        .hybrid()
        .swap()
        .ok_or("the load runs no residency machine")?
        .rule()
        .clone())
}

/// `refuse`: the host one byte short of the churn pool, on placement
/// `place`'s machine.
pub fn refuse_clause(
    path: &Path,
    inputs: &PlanInputs,
    cfg: &OpenCfg,
    place: Place,
    machine_of: &dyn Fn(usize) -> Machine,
) -> Result<bool, GateError> {
    let machine = machine_of(inputs.model.layers);
    let plan = inputs.plan(&machine, workstation::CTX_MAX, &cfg.place)?;
    let pool = swap::churn(&plan, 0, RESIDENCY)?.ok_or("no churn pool under mid")?;
    record::residency_host("mid-p40-s1", &pool, &plan).print();
    let (short, small) =
        crate::residency_clauses::refuse_head(&machine, plan.host.headroom_bytes, pool.bytes)?;
    crate::residency_clauses::refuse_tail(short, pool.bytes, REFUSE_BOUND_S, move || {
        open(path, place, move |_| small.clone(), cfg)
    })
}

/// The layers the stage card holds routed experts of, with their seeds.
fn seeds(s: &Session<Body>) -> Result<Vec<(usize, Vec<u32>)>, GateError> {
    let m = s.model();
    let b = m.body(NAME)?;
    let machine = b
        .hybrid()
        .swap()
        .ok_or("the load runs no residency machine")?;
    crate::residency_clauses::seeds(machine, b.hybrid().slots().layers())
}

/// Per layer, every tier entry of the host map.
fn tier_entries(s: &Session<Body>) -> Result<Vec<(usize, u32, Slot)>, GateError> {
    let m = s.model();
    let map = m.body(NAME)?.hybrid().slots();
    let mut out = Vec::new();
    for l in map.layers() {
        for id in 0..map.n_expert() as u32 {
            if let Some(t @ Slot::Tier { .. }) = map.slot(l, id) {
                out.push((l, id, t));
            }
        }
    }
    Ok(out)
}

/// `transform`: each admitted expert's slot against its static bytes.
fn transform_clause(s: &Session<Body>) -> Result<bool, GateError> {
    let m = s.model();
    let b = m.body(NAME)?;
    let machine = b
        .hybrid()
        .swap()
        .ok_or("the load runs no residency machine")?;
    let source = b
        .residency_source()
        .ok_or("the load has no residency source")?;
    if !source.unpacks() {
        println!(
            "transform: the host reads the source, so no flip unpacks; the clause needs the \
             r8 sidecar (BLOOMERY_R8=on, just r8-sidecar): {}",
            verdict(false)
        );
        return Ok(false);
    }
    let layers = crate::residency_clauses::layer_seeds(machine, b.hybrid().slots().layers())?;
    let (checked, bad) =
        crate::residency_clauses::transform_check(m.gpu(), machine, source, &layers, |_| 3)?;
    Ok(crate::residency_clauses::transform_verdict_with(
        checked,
        &bad,
        skew()?,
    ))
}

/// `resident`: the churn pool and the host experts in, the pinned out.
fn resident_clause(s: &Session<Body>, seeds: &[(usize, Vec<u32>)]) -> Result<bool, GateError> {
    let t = Instant::now();
    let m = s.model();
    let b = m.body(NAME)?;
    let source = b
        .residency_source()
        .ok_or("the load has no residency source")?;
    let machine = b
        .hybrid()
        .swap()
        .ok_or("the load runs no residency machine")?;
    let map = b.hybrid().slots();
    let (mut asked, mut wrong) = (0usize, Vec::new());
    for (l, seed) in seeds {
        let l = *l;
        let pinned = machine.pinned(l)?;
        for id in 0..map.n_expert() as u32 {
            let rank = seed.iter().position(|&e| e == id);
            let want = match (rank, map.slot(l, id)) {
                (Some(r), _) => r >= pinned,
                (None, Some(Slot::Host)) => true,
                // The tier's experts, and admitted ones the seed does not
                // rank, are not asked.
                _ => continue,
            };
            asked += 1;
            if source.host_resident(l, id)? != want {
                wrong.push(format!("layer {l} expert {id} (want {want})"));
            }
        }
    }
    let ok = asked > 0 && wrong.is_empty();
    println!(
        "resident: {asked} experts asked in {:.1} s, {} wrong{}: {}",
        t.elapsed().as_secs_f64(),
        wrong.len(),
        wrong
            .first()
            .map(|f| format!(" (first {f})"))
            .unwrap_or_default(),
        verdict(ok)
    );
    Ok(ok)
}

/// The residency views of `s`'s body: its stage card's copy of the map
/// (`Body::slots`) and its host tier.
fn residence(s: &Session<Body>) -> Result<Residence<'_>, GateError> {
    let m = s.model();
    let b = m.body(NAME)?;
    Ok(Residence::of(m.gpu(), b.slots(), b.hybrid()))
}

/// `table`: the stage card's copy after `held`, the held history, against
/// the host map, the ledger and the map `held` started from.
fn table_clause(s: &Session<Body>, held: &History) -> Result<bool, GateError> {
    crate::residency_clauses::table_clause_with(
        &residence(s)?,
        &held.start,
        held.landed,
        "the held history",
        skew()?,
    )
}

/// [`StaticProbe`] on `s`, after the held history and `keep`.
fn static_probe(s: &mut Session<Body>, ids: &[u32]) -> Result<StaticProbe, GateError> {
    crate::residency_clauses::static_probe(s, ids, residence, take_passes)
}

/// `static`, its second half, after the residency load's teardown: a load by
/// `machine` under `cfg` with the residency and host streaming off, each
/// routed layer's card experts `p`'s copy's, fed `p`'s ids the serve's way;
/// its row 0 against `p`'s, bit for bit.
pub fn static_clause(
    path: &Path,
    place: Place,
    machine: impl Fn(usize) -> Machine,
    cfg: &OpenCfg,
    p: &StaticProbe,
) -> Result<bool, GateError> {
    let mut cfg = cfg.clone();
    cfg.body.residency = Residency::Off;
    cfg.body.hoststream = false;
    let t0 = Instant::now();
    let mut s = open_edited(path, place, machine, &cfg, |plan| {
        place_table(plan, &p.table).map_err(|e| SessionError::Refused(e.to_string()))
    })?;
    crate::residency_clauses::static_row0(&mut s, residence, p, t0.elapsed().as_secs_f64())
}

/// Every clause after `refuse` on `s`, the load just made under
/// [`RESIDENCY`] by placement `place` (plan (b′) or (a)), before its first
/// call; `flags` holds the copy stream for `c3`. Host streaming is off for
/// them, as on the load they were made on (their prompt calls are under the
/// streaming floor anyway). `false` when one is red; under plan (a), with
/// `static`'s residency half for [`static_clause`] to finish on its own load.
pub fn clauses(
    s: &mut Session<Body>,
    flags: &HostFlags,
    place: Place,
) -> Result<(bool, Option<StaticProbe>), GateError> {
    {
        let body = s.model_mut().body_parts(NAME)?.2;
        body.set_hoststream(false)?;
        body.log_residency(0);
    }
    let ids = prose(PROMPT)?;
    let mut pass = true;
    let tier_at_load = tier_entries(s)?;
    let seeds = seeds(s)?;

    let step = crate::residency_clauses::rule_after(s, &ids, rule_of, take_passes, |s, t| {
        s.step(t, Want::Argmax)?;
        Ok(())
    })?;
    let one = crate::residency_clauses::rule_after(s, &ids, rule_of, take_passes, |s, t| {
        s.verify::<2>([t, t])?;
        s.commit(1)?;
        Ok(())
    })?;
    let both = crate::residency_clauses::rule_after(s, &ids, rule_of, take_passes, |s, t| {
        s.verify::<2>([t, t])?;
        s.commit(2)?;
        Ok(())
    })?;
    crate::ds41_tier::sc("c6: a pair pass counts its kept rows only")?;
    let c6 = one == step && both != one;
    println!(
        "c6: a pair pass keeping one row leaves the rule a step leaves ({}), keeping both \
         does not ({}): {}",
        one == step,
        both != one,
        verdict(c6)
    );
    pass &= c6;

    let first = history(s, &ids, None)?;
    crate::ds41_tier::sc("transform: an admitted expert's slot is a static load's bytes")?;
    pass &= transform_clause(s)?;
    crate::ds41_tier::sc("resident: the host set answers the truth")?;
    pass &= resident_clause(s, &seeds)?;
    let again = history(s, &ids, None)?;
    crate::ds41_tier::sc("c1: the history twice gives the same tokens and logits")?;
    pass &= crate::residency_clauses::c1_clause_with(
        first.tokens.len(),
        first.landed,
        first == again,
        skew()?,
    );

    crate::ds41_tier::sc("passes: a history's boundaries end the pass kinds in order")?;
    pass &= crate::residency_clauses::passes_clause(&first.passes, STEPS);

    crate::ds41_tier::sc("tier: every tier entry is where the load put it")?;
    let tier_ok = if place == Place::A {
        let after = tier_entries(s)?.len();
        let ok = tier_at_load.is_empty() && after == 0;
        println!(
            "tier: plan (a) has no tier card: {} tier entries at the load, {after} after the \
             histories: {}",
            tier_at_load.len(),
            verdict(ok)
        );
        ok
    } else {
        let tier_layers: Vec<usize> = tier_at_load.iter().map(|&(l, _, _)| l).collect();
        let ok = tier_entries(s)? == tier_at_load && !tier_at_load.is_empty();
        println!(
            "tier: {} tier entries on {} layers unchanged: {}",
            tier_at_load.len(),
            {
                let mut t = tier_layers.clone();
                t.dedup();
                t.len()
            },
            verdict(ok)
        );
        ok
    };
    pass &= tier_ok;

    let held = history(s, &ids, Some(flags))?;
    crate::ds41_tier::sc("c3: the held copy stream gives the first run's tokens and logits")?;
    let c3 = held.tokens == first.tokens && held.fnvs == first.fnvs;
    println!(
        "c3: the history with the copy stream held {HOLD:?}: {} flips landed, same tokens \
         and logits: {}",
        held.landed,
        verdict(c3)
    );
    pass &= c3;

    // The seat's reset on a cache miss is `Target::reset`.
    crate::ds41_tier::sc("keep: the seat's reset leaves the residency where use took it")?;
    let _ = s.take_cleared();
    let before = (live_sets(s, &seeds)?, rule_of(s)?);
    runtime::Target::reset(s)?;
    let after = (live_sets(s, &seeds)?, rule_of(s)?);
    let moved = seeds.iter().zip(&before.0).any(|((_, seed), live)| {
        live.len() != seed.len() || seed.iter().any(|e| !live.contains(e))
    });
    let cleared = s.take_cleared().is_some();
    let keep = (moved || !skew()?) && before == after && !cleared;
    println!(
        "keep: the serve seat's reset (Target::reset) after the held history, the live sets \
         off the seed {moved}: live sets and rule unchanged {}, no residency reset {}: {}",
        before == after,
        !cleared,
        verdict(keep)
    );
    pass &= keep;

    crate::ds41_tier::sc("table: the card's copy is the host map and the ledger, no slot twice")?;
    pass &= table_clause(s, &held)?;
    let probe = match place == Place::A {
        true => {
            crate::ds41_tier::sc("static: the residency computes the placement its copy holds")?;
            Some(static_probe(s, &ids)?)
        }
        false => None,
    };

    let r = s
        .residency_reset()?
        .ok_or("the load runs no residency machine")?;
    record::residency_reset(&r).print();
    let machine = {
        let m = s.model();
        let b = m.body(NAME)?;
        b.hybrid().swap().ok_or("no machine")?
    };
    crate::ds41_tier::sc("c7: a residency reset brings every layer back to its seed")?;
    pass &= crate::residency_clauses::c7_clause(&r, machine, &seeds);
    Ok((pass, probe))
}
