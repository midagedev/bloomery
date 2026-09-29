//! GPU gate for V4.1's adaptive expert residency on the real model
//! (`bloomery_gpu_deepseek41::swap`, `BLOOMERY_RESIDENCY`): plan (b′) on
//! both cards (`--place bp`: the stage on the A6000, the expert tier on the
//! 3090), the residency machine over the stage card's routed stacks at
//! `mid-p40-s1`, set here (the lever itself is refused, so the environment
//! cannot move it). The stage's seed is the plan's id prefix. One load.
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
//!   machine's layout does not hold the tier's experts away).
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
//! - `passes` (green-only): a history's boundaries end, in order, no pass,
//!   the prompt call (one pass, 0 rows kept) and each step (1 kept).
//!
//! The gate runs with the r8 sidecar (`BLOOMERY_R8` on, its default): without
//! it no flip unpacks, and `transform` is red by name.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_deepseek41_residency: built without the `deepseek41` feature; see `just \
         gate-gpu-ds41-residency`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_residency", gate::run())
}

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_open.rs"]
mod ds41_open;

#[cfg(feature = "deepseek41")]
mod gate {
    use crate::ds41_open::{fnv, open};
    use std::path::Path;
    use std::time::{Duration, Instant};

    use app::Session;
    use bloomery_gpu::host::PassKind;
    use bloomery_gpu::host::slots::Slot;
    use bloomery_gpu::host::swap::{Residency, SlotState, SwapSource};
    use bloomery_gpu::{HostFlags, window};
    use bloomery_gpu_deepseek41::body::{Body, OpenCfg, PrefillMode};
    use bloomery_gpu_deepseek41::swap;
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::{GateError, checks_failed, data_dir, ref_model_path, verdict};
    use bloomery_levers::{CARD_DONTNEED, ENGRAM_HELPER, HOST_LOCK, HOST_POPULATE, R8};
    use gguf::Split;
    use model::arch::deepseek41::place::{self, PlanInputs};
    use model::placement::{Machine, workstation};
    use runtime::swaprule::SwapRule;
    use runtime::{Target, Verify, Want};

    const NAME: &str = "gate_deepseek41_residency";
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
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let ids = text
            .split_whitespace()
            .take(n)
            .map(str::parse::<u32>)
            .collect::<Result<Vec<_>, _>>()?;
        if ids.len() < n {
            return Err(
                format!("{}: {} ids, the gate reads {n}", path.display(), ids.len()).into(),
            );
        }
        Ok(ids)
    }

    /// Milliseconds since `t`.
    fn ms(t: Instant) -> f64 {
        t.elapsed().as_secs_f64() * 1e3
    }

    /// What a history saw: every step's argmax and logits FNV, each
    /// boundary's ended pass (its kind and kept rows), and the flips that
    /// landed.
    #[derive(PartialEq)]
    struct History {
        tokens: Vec<u32>,
        fnvs: Vec<u64>,
        passes: Vec<(PassKind, usize)>,
        landed: usize,
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
            };
            for _ in 0..STEPS {
                let t = Instant::now();
                let out = s.step(next, Want::Logits)?;
                walls.push(ms(t));
                if let runtime::Out::Logits { row, .. } = out {
                    h.fnvs.push(fnv(row));
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

    /// The rule after `pass` from a clear: the prompt, `pass`, then the next
    /// boundary, which folds the pass's kept rows.
    fn rule_after(
        s: &mut Session<Body>,
        ids: &[u32],
        pass: impl FnOnce(&mut Session<Body>, u32) -> Result<(), GateError>,
    ) -> Result<SwapRule, GateError> {
        s.clear()?;
        let next = s.prompt(ids, Want::Argmax)?.argmax();
        pass(s, next)?;
        s.model_mut().pass_boundary()?;
        let rule = rule_of(s)?;
        take_passes(s)?;
        Ok(rule)
    }

    /// Each seed layer's live experts, sorted, in `seeds`' order.
    fn live_sets(
        s: &Session<Body>,
        seeds: &[(usize, Vec<u32>)],
    ) -> Result<Vec<Vec<u32>>, GateError> {
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

    /// `refuse`: the host one byte short of the churn pool.
    fn refuse_clause(
        path: &Path,
        inputs: &PlanInputs,
        cfg: &OpenCfg,
        bp: &dyn Fn(usize) -> Machine,
    ) -> Result<bool, GateError> {
        let machine = bp(inputs.model.layers);
        let plan = inputs.plan(&machine, workstation::CTX_MAX, &cfg.place)?;
        let pool = swap::churn(&plan, 0, RESIDENCY)?.ok_or("no churn pool under mid")?;
        record::residency_host("mid-p40-s1", &pool, &plan).print();
        let short = i128::from(machine.host.usable_bytes) - plan.host.headroom_bytes
            + i128::from(pool.bytes)
            - 1;
        let mut small = machine.clone();
        small.host.usable_bytes = u64::try_from(short)?;
        let t0 = Instant::now();
        let opened = open(path, move |_| small.clone(), cfg);
        let secs = t0.elapsed().as_secs_f64();
        let (ok, why) = match opened {
            Err(e) => {
                let text = e.to_string();
                (text.contains("churn pool") && secs < REFUSE_BOUND_S, text)
            }
            Ok(_) => (false, "the open loaded".to_string()),
        };
        println!(
            "refuse: a host of {short} B, one byte short of the {} B churn pool: {} in {secs:.1} \
             s — {why}",
            pool.bytes,
            verdict(ok)
        );
        Ok(ok)
    }

    /// The layers the stage card holds routed experts of, with their seeds.
    fn seeds(s: &Session<Body>) -> Result<Vec<(usize, Vec<u32>)>, GateError> {
        let m = s.model();
        let b = m.body(NAME)?;
        let machine = b
            .hybrid()
            .swap()
            .ok_or("the load runs no residency machine")?;
        let mut out = Vec::new();
        for l in b.hybrid().slots().layers() {
            let seed = machine.seed(l)?;
            if !seed.is_empty() {
                out.push((l, seed));
            }
        }
        Ok(out)
    }

    /// Per layer, every tier entry of the host map.
    fn tier_entries(s: &Session<Body>) -> Result<Vec<(usize, u32, Slot)>, GateError> {
        let m = s.model();
        let map = m.body(NAME)?.hybrid().slots();
        let mut out = Vec::new();
        for l in map.layers() {
            for id in 0..map.n_expert() as u32 {
                if let Some(t @ Slot::Tier(_)) = map.slot(l, id) {
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
        let (gpu, stream) = (m.gpu(), m.gpu().stream());
        stream.synchronize()?;
        let (mut checked, mut bad) = (0usize, Vec::new());
        for l in b.hybrid().slots().layers() {
            let seed = machine.seed(l)?;
            let Some(row) = machine.ledger().row(l) else {
                continue;
            };
            for (slot, st) in row.iter().enumerate() {
                let SlotState::Live(e) = *st else { continue };
                if seed.contains(&e) {
                    continue;
                }
                checked += 1;
                for part in 0..3 {
                    let want = source.card_bytes(l, e, part)?;
                    let at = source.dest(l, part, slot as u32)?;
                    // SAFETY: `at` is slot `slot` of the stage card's stack
                    // of layer `l`, which holds `want.len()` bytes there and
                    // stays allocated while the model lives.
                    let view = unsafe { window::<u32>(at, want.len() / 4, gpu.context()) };
                    let mut got = vec![0u32; want.len() / 4];
                    view.copy_to_host(stream, &mut got)?;
                    let got: Vec<u8> = got.iter().flat_map(|w| w.to_le_bytes()).collect();
                    if got != want {
                        let first = got.iter().zip(want).position(|(a, b)| a != b);
                        bad.push(format!(
                            "layer {l} expert {e} slot {slot} part {part} at {first:?}"
                        ));
                    }
                }
            }
        }
        let ok = checked > 0 && bad.is_empty();
        println!(
            "transform: {checked} admitted experts, {} parts differ from a static load{}: {}",
            bad.len(),
            bad.first()
                .map(|f| format!(" (first {f})"))
                .unwrap_or_default(),
            verdict(ok)
        );
        Ok(ok)
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

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[
            ENGRAM_HELPER,
            HOST_POPULATE,
            HOST_LOCK,
            CARD_DONTNEED,
            R8,
        ])?;
        let mut cfg = OpenCfg::from_levers(&levers)?;
        cfg.body.residency = RESIDENCY;
        cfg.body.prefill = PrefillMode::Batch;
        let path = ref_model_path()?;
        let inputs = PlanInputs::read(
            &Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?,
        )?;
        let bp = Place::Bp.machine(None, Some(place::tier_batch(&inputs.hp)))?;
        let ids = prose(PROMPT)?;
        let mut pass = refuse_clause(&path, &inputs, &cfg, &bp)?;

        let t0 = Instant::now();
        let mut s = open(&path, bp, &cfg)?;
        s.model_mut().body_parts(NAME)?.2.log_residency(0);
        println!("load in {:.1} s", t0.elapsed().as_secs_f64());
        let tier_at_load = tier_entries(&s)?;
        let seeds = seeds(&s)?;

        let step = rule_after(&mut s, &ids, |s, t| {
            s.step(t, Want::Argmax)?;
            Ok(())
        })?;
        let one = rule_after(&mut s, &ids, |s, t| {
            s.verify::<2>([t, t])?;
            s.commit(1)?;
            Ok(())
        })?;
        let both = rule_after(&mut s, &ids, |s, t| {
            s.verify::<2>([t, t])?;
            s.commit(2)?;
            Ok(())
        })?;
        let c6 = one == step && both != one;
        println!(
            "c6: a pair pass keeping one row leaves the rule a step leaves ({}), keeping both \
             does not ({}): {}",
            one == step,
            both != one,
            verdict(c6)
        );
        pass &= c6;

        let first = history(&mut s, &ids, None)?;
        pass &= transform_clause(&s)?;
        pass &= resident_clause(&s, &seeds)?;
        let again = history(&mut s, &ids, None)?;
        let c1 = first == again && first.landed > 0;
        println!(
            "c1: the history twice, {} tokens, {} flips landed: same {}: {}",
            first.tokens.len(),
            first.landed,
            first == again,
            verdict(c1)
        );
        pass &= c1;

        // The prompt call is one pass that keeps 0 rows; each step keeps 1.
        let mut want = vec![(PassKind::None, 0), (PassKind::Prompt, 0)];
        want.extend(std::iter::repeat_n((PassKind::Step, 1), STEPS - 1));
        let passes_ok = first.passes == want;
        println!(
            "passes: a history's boundaries end none, the prompt call (0 kept), then {} steps (1 \
             kept each): {} boundaries, same {passes_ok}: {}",
            STEPS - 1,
            first.passes.len(),
            verdict(passes_ok)
        );
        pass &= passes_ok;

        let tier_layers: Vec<usize> = tier_at_load.iter().map(|&(l, _, _)| l).collect();
        let tier_ok = tier_entries(&s)? == tier_at_load && !tier_at_load.is_empty();
        println!(
            "tier: {} tier entries on {} layers unchanged: {}",
            tier_at_load.len(),
            {
                let mut t = tier_layers.clone();
                t.dedup();
                t.len()
            },
            verdict(tier_ok)
        );
        pass &= tier_ok;

        let flags = HostFlags::new(s.model().gpu().context(), 1)?;
        let held = history(&mut s, &ids, Some(&flags))?;
        let c3 = held.tokens == first.tokens && held.fnvs == first.fnvs;
        println!(
            "c3: the history with the copy stream held {HOLD:?}: {} flips landed, same tokens \
             and logits: {}",
            held.landed,
            verdict(c3)
        );
        pass &= c3;

        // The seat's reset on a cache miss is `Target::reset`.
        let _ = s.take_cleared();
        let before = (live_sets(&s, &seeds)?, rule_of(&s)?);
        runtime::Target::reset(&mut s)?;
        let after = (live_sets(&s, &seeds)?, rule_of(&s)?);
        let moved = seeds.iter().zip(&before.0).any(|((_, seed), live)| {
            live.len() != seed.len() || seed.iter().any(|e| !live.contains(e))
        });
        let cleared = s.take_cleared().is_some();
        let keep = moved && before == after && !cleared;
        println!(
            "keep: the serve seat's reset (Target::reset) after the held history, the live sets \
             off the seed {moved}: live sets and rule unchanged {}, no residency reset {}: {}",
            before == after,
            !cleared,
            verdict(keep)
        );
        pass &= keep;

        let r = s
            .residency_reset()?
            .ok_or("the load runs no residency machine")?;
        record::residency_reset(&r).print();
        let live_is_seed = {
            let m = s.model();
            let b = m.body(NAME)?;
            let machine = b.hybrid().swap().ok_or("no machine")?;
            seeds.iter().all(|(l, seed)| {
                let live: Vec<u32> = machine
                    .ledger()
                    .row(*l)
                    .unwrap_or(&[])
                    .iter()
                    .filter_map(|st| match st {
                        SlotState::Live(e) => Some(*e),
                        _ => None,
                    })
                    .collect();
                live.len() == seed.len() && seed.iter().all(|e| live.contains(e))
            })
        };
        let c7 = r.diff == 0 && live_is_seed && r.dropped_bytes == 0;
        println!(
            "c7: the reset's diff {}, live sets the seed {live_is_seed}, dropped {} B (0: the \
             churn pool stays): {}",
            r.diff,
            r.dropped_bytes,
            verdict(c7)
        );
        pass &= c7;

        if pass {
            println!("{NAME}: every clause passed");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
