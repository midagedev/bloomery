//! GPU gate for Qwen3.8's adaptive expert residency on the real model
//! (`bloomery_gpu::arch::qwen3moe::swap38`, `BLOOMERY_RESIDENCY`): the card
//! plan (`place::Experts::Card`, each eligible layer's id prefix on the
//! card), the residency machine over the card's routed stacks at
//! `mid-p<P>-s1`, set here with `P` taken from the plan's card experts a
//! layer (the lever itself is refused, so the environment cannot move it).
//! One load, one card, no MTP draft.
//!
//! A history is: the session cleared (the residency back to its seed), the
//! first [`PROMPT`] ids of the prose corpus as one prompt call (a ubatch), a
//! greedy step, a verify of three rows committed at two, then [`STEPS`]
//! greedy steps, each call's argmax read and each step's logits row hashed.
//! Clauses; each names its mutant, and the machine's own rules have theirs
//! in `gate_swap`:
//!
//! - `refuse` (the host-set refusal at load): a machine whose host holds one
//!   byte less than the churn pool needs — the card's experts past the pinned
//!   ones, which the host set must hold too — is refused by name before
//!   anything loads, within [`REFUSE_BOUND_S`] (mutant: the open's churn
//!   pool check removed).
//! - `c6` (a pass counts its kept rows only): a verify `[t, t]` after the
//!   prompt with one row kept, then a boundary, leaves the rule equal to one
//!   step of `t` then a boundary; the same verify with both rows kept does
//!   not (mutant: the session's commit keeps every row of the pass).
//! - `c1` (green-only): the history twice gives the same tokens, logits and
//!   passes, and flips land (mutant: the `keep_rows` after the prompt call
//!   removed — the next step's boundary then refuses the prompt pass by
//!   name).
//! - `transform`: after the history, every expert the machine admitted holds
//!   in its slot, part by part, the bytes a static load uploads for it —
//!   the file's Q4_K gate and up, Q5_1 down (mutant: `Qwen38Stacks::names`
//!   returns the gate and the up swapped, so the up's bytes stage into the
//!   gate's stack).
//! - `passes` (green-only): a history's boundaries end, in order, no pass,
//!   the prompt call (one pass, 0 rows kept), a step (1 kept), a verify (its
//!   accepted rows) and the steps after it (1 kept each).
//! - `c7`: a residency reset after the history brings every layer's live set
//!   back to its seed (`diff` 0, and the ledger), and lets go of no host
//!   byte (`dropped_bytes` 0: the churn pool stays in the host set for the
//!   model's life) (mutant: the reset's copies of the seed experts back
//!   skipped).

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen38_residency: built without the `gpu` feature; see `just \
         gate-gpu-qwen38-residency`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen38_residency", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use std::path::Path;
    use std::time::Instant;

    use app::Session;
    use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for;
    use bloomery_gpu::arch::qwen3moe::{Body38, Qwen38Model};
    use bloomery_gpu::host::PassKind;
    use bloomery_gpu::host::swap::{PassReport, Residency, SlotState, SwapSource};
    use bloomery_gpu::window;
    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::{GateError, checks_failed, data_dir, verdict};
    use bloomery_levers::{CARD_DONTNEED, HOST_LOCK, HOST_POPULATE};
    use gguf::Split;
    use model::arch::qwen35moe::place::{Experts, PlanInputs, machine_for_experts};
    use model::placement::Machine;
    use model::placement::PlanLevers;
    use model::placement::churn::ChurnPool;
    use model::placement::workstation::A6000;
    use refset::arch::qwen4exp::MODEL;
    use runtime::swaprule::SwapRule;
    use runtime::{Out, Target, Verify, Want};

    const NAME: &str = "gate_qwen38_residency";
    /// The stores' positions, `gate_qwen4exp_e2e`'s.
    const CTX: usize = 3072;
    /// Prompt positions: one prompt call, one ubatch.
    const PROMPT: usize = 64;
    /// Greedy steps after the verify: planning boundaries of the rule.
    const STEPS: usize = 96;
    /// A refusal before the load reads the file's headers only.
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

    /// FNV-1a 64 of an f32 row's bits (`shared/ds41_open.rs`'s `fnv`, the
    /// one hash of this gate's family).
    fn fnv(row: &[f32]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for v in row {
            for b in v.to_bits().to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        }
        h
    }

    /// The machine's card plan of `inputs` on `card`.
    fn plan_of<'m>(
        inputs: &'m PlanInputs,
        card: &'m Machine,
    ) -> Result<model::placement::Plan<'m>, GateError> {
        inputs
            .plan_with(card, CTX as u64, &PlanLevers::default(), Experts::Card)
            .map_err(|e| format!("the plan: {e}").into())
    }

    /// The residency the gate runs over `plan`: half of the plan's smallest
    /// layer of card experts pinned (the seed is the plan's id prefix, not a
    /// frequency list, so pinning most of it protects nothing), one spare a
    /// layer. `None` when no layer holds the machine's `pinned + spares + 1`
    /// card experts.
    fn residency_of(plan: &model::placement::Plan<'_>) -> Option<Residency> {
        let slots = plan.n_l.iter().copied().filter(|&n| n > 0).min()? as usize;
        let pinned = slots / 2;
        (pinned >= 1 && slots >= pinned + 2).then_some(Residency::Mid { pinned, spares: 1 })
    }

    /// The lever word of [`residency_of`]'s answer.
    fn word_of(r: &Residency) -> String {
        match *r {
            Residency::Off => "off".to_string(),
            Residency::Mid { pinned, spares } => format!("mid-p{pinned}-s{spares}"),
        }
    }

    /// The model placed on the gate's card by its plan, the residency machine
    /// over the card's routed stacks at `residency`.
    fn open(
        path: &Path,
        inputs: &PlanInputs,
        machine: Machine,
        residency: Residency,
        host: bloomery_levers::HostCfg,
        ub: usize,
    ) -> Result<Qwen38Model, GateError> {
        let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        if file.architecture() != Some("qwen4exp") {
            return Err(format!("{} is not qwen4exp", path.display()).into());
        }
        let plan = plan_of(inputs, &machine)?;
        Ok(Body38::open_placed_residency(
            file, &plan, inputs, 0, host, ub, residency,
        )?)
    }

    /// What a history saw: every call's argmax, each step's logits FNV, each
    /// boundary's ended pass (its kind and kept rows), and the flips that
    /// landed.
    #[derive(PartialEq)]
    struct History {
        tokens: Vec<u32>,
        fnvs: Vec<u64>,
        passes: Vec<(PassKind, usize)>,
        landed: usize,
    }

    /// The boundaries' reports since the last take.
    fn take_passes(s: &mut Session<Body38>) -> Result<Vec<(PassKind, PassReport)>, GateError> {
        let (_, _, b) = s.model_mut().body_parts(NAME)?;
        Ok(b.take_residency_passes())
    }

    /// The history from a clear: the prompt, a step, a verify of three rows
    /// committed at two, then [`STEPS`] greedy steps.
    fn history(s: &mut Session<Body38>, ids: &[u32]) -> Result<History, GateError> {
        s.clear()?;
        take_passes(s)?;
        let mut h = History {
            tokens: Vec::with_capacity(STEPS + 4),
            fnvs: Vec::with_capacity(STEPS),
            passes: Vec::new(),
            landed: 0,
        };
        let mut next = s.prompt(ids, Want::Argmax)?.argmax();
        h.tokens.push(next);
        next = s.step(next, Want::Argmax)?.argmax();
        h.tokens.push(next);
        let out = s.verify::<3>([next; 3])?;
        h.tokens.extend_from_slice(&out[1..]);
        s.commit(2)?;
        for _ in 0..STEPS {
            let out = s.step(next, Want::Logits)?;
            if let Out::Logits { row, .. } = out {
                h.fnvs.push(fnv(row));
            }
            next = out.argmax();
            h.tokens.push(next);
        }
        let passes = take_passes(s)?;
        h.landed = passes.iter().map(|(_, r)| r.landed).sum();
        h.passes = passes.iter().map(|&(k, r)| (k, r.kept)).collect();
        Ok(h)
    }

    /// The rule after `pass` from a clear: the prompt, `pass`, then the next
    /// boundary, which folds the pass's kept rows.
    fn rule_after(
        s: &mut Session<Body38>,
        ids: &[u32],
        pass: impl FnOnce(&mut Session<Body38>, u32) -> Result<(), GateError>,
    ) -> Result<SwapRule, GateError> {
        s.clear()?;
        let next = s.prompt(ids, Want::Argmax)?.argmax();
        pass(s, next)?;
        s.model_mut().pass_boundary()?;
        let rule = rule_of(s)?;
        take_passes(s)?;
        Ok(rule)
    }

    /// The machine's rule as it stands.
    fn rule_of(s: &Session<Body38>) -> Result<SwapRule, GateError> {
        Ok(s.model()
            .body(NAME)?
            .hybrid()
            .swap()
            .ok_or("the load runs no residency machine")?
            .rule()
            .clone())
    }

    /// The layers the card holds routed experts of, with their seeds.
    fn seeds(s: &Session<Body38>) -> Result<Vec<(usize, Vec<u32>)>, GateError> {
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

    /// `transform`: each admitted expert's slot against its static bytes.
    fn transform_clause(s: &Session<Body38>) -> Result<bool, GateError> {
        let m = s.model();
        let b = m.body(NAME)?;
        let machine = b
            .hybrid()
            .swap()
            .ok_or("the load runs no residency machine")?;
        let source = b
            .residency_source()
            .ok_or("the load has no residency source")?;
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
                    // SAFETY: `at` is slot `slot` of the card's stack
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

    /// `refuse`: the host one byte short of the churn pool.
    fn refuse_clause(
        path: &Path,
        inputs: &PlanInputs,
        machine: &Machine,
        residency: &Residency,
        host: bloomery_levers::HostCfg,
        ub: usize,
    ) -> Result<bool, GateError> {
        let plan = plan_of(inputs, machine)?;
        let Residency::Mid { pinned, .. } = *residency else {
            return Err("the gate's residency is mid".into());
        };
        let pool = ChurnPool::of(&plan, 0, pinned).map_err(|e| format!("the churn pool: {e}"))?;
        pool.check(&plan)
            .map_err(|e| format!("the churn pool does not fit the plan's host: {e}"))?;
        record::residency_host(&word_of(residency), &pool, &plan).print();
        let short = i128::from(machine.host.usable_bytes) - plan.host.headroom_bytes
            + i128::from(pool.bytes)
            - 1;
        let mut small = machine.clone();
        small.host.usable_bytes = u64::try_from(short)?;
        let t0 = Instant::now();
        let opened = open(path, inputs, small, *residency, host, ub);
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

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[HOST_POPULATE, HOST_LOCK, CARD_DONTNEED])?;
        let host = levers.host();
        let path = Path::new(MODEL);
        let inputs = PlanInputs::describe(
            &Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?,
        )?;
        let ub = ubatch_for(CTX)?;
        let machine =
            machine_for_experts(A6000, inputs.spec.layers.len(), ub as u64, Experts::Card);
        let (residency, slots) = {
            let plan = plan_of(&inputs, &machine)?;
            let slots = plan
                .n_l
                .iter()
                .copied()
                .filter(|&n| n > 0)
                .min()
                .unwrap_or(0);
            (
                residency_of(&plan).ok_or(
                    "the plan's card experts leave no layer with room for the machine's pinned, \
                     spare and one that moves",
                )?,
                slots,
            )
        };
        println!(
            "residency: {} over the plan's {slots} card experts a layer at least",
            word_of(&residency)
        );

        let mut pass = refuse_clause(path, &inputs, &machine, &residency, host, ub)?;

        let t0 = Instant::now();
        let mut s = Session::from_model(
            open(path, &inputs, machine, residency, host, ub)?,
            CTX as u32,
        );
        s.model_mut().body_parts(NAME)?.2.log_residency(0);
        println!("load in {:.1} s", t0.elapsed().as_secs_f64());
        let seeds = seeds(&s)?;

        let ids = prose(PROMPT)?;
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
            "c6: a verify keeping one row leaves the rule a step leaves ({}), keeping both does \
             not ({}): {}",
            one == step,
            both != one,
            verdict(c6)
        );
        pass &= c6;

        let first = history(&mut s, &ids)?;
        pass &= transform_clause(&s)?;
        let again = history(&mut s, &ids)?;
        let c1 = first == again && first.landed > 0;
        println!(
            "c1: the history twice, {} tokens, {} flips landed: same {}: {}",
            first.tokens.len(),
            first.landed,
            first == again,
            verdict(c1)
        );
        pass &= c1;

        // The prompt call is one pass that keeps 0 rows; a step keeps 1; a
        // verify keeps its accepted rows.
        let mut want = vec![
            (PassKind::None, 0),
            (PassKind::Prompt, 0),
            (PassKind::Step, 1),
            (PassKind::Pair, 2),
        ];
        want.extend(std::iter::repeat_n((PassKind::Step, 1), STEPS - 1));
        let passes_ok = first.passes == want;
        println!(
            "passes: a history's boundaries end none, the prompt call (0 kept), a step (1), a \
             verify (2), then {} steps (1 kept each): {} boundaries, same {passes_ok}: {}",
            STEPS - 1,
            first.passes.len(),
            verdict(passes_ok)
        );
        pass &= passes_ok;

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
