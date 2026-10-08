//! GPU gate for Qwen3.8's adaptive expert residency on the real model
//! (`bloomery_gpu::arch::qwen3moe::swap38`, `BLOOMERY_RESIDENCY`): the card
//! plan (`place::Experts::Card`, each eligible layer's id prefix on the
//! card), the residency machine over the card's routed stacks at
//! `mid-p<P>-s1`, set here with `P` taken from the plan's card experts a
//! layer (the lever itself is refused, so the environment cannot move it).
//! One card, no MTP draft, two loads one after the other: the gate's own,
//! planned for one sequence, and the slots clauses' (`slots_drafted`,
//! `commit_counts`), planned for the two slots they serve — a pass of
//! several slots on a load planned for one is refused by name.
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
//!   the file's Q4_K gate and up, Q5_1 down (mutant: `swap38::stacks`' names
//!   returns the gate and the up swapped, so the up's bytes stage into the
//!   gate's stack).
//! - `passes` (green-only): a history's boundaries end, in order, no pass,
//!   the prompt call (one pass, 0 rows kept), a step (1 kept), a verify (its
//!   accepted rows) and the steps after it (1 kept each), the last step's
//!   own included.
//! - `order`: every step's own boundary — the next pass's — runs ahead of
//!   the step's readback: when the step returns, the last boundary ends it,
//!   was made ahead (`PassReport::ahead`), and carries the readbacks from
//!   before the step (`PassReport::reads`), while the model has read once
//!   more since (mutants: the boundary made after the head's readback, which
//!   carries the step's own readback; the boundary back at the next launch,
//!   which leaves the step unended when it returns).
//! - `c7`: a residency reset after the history brings every layer's live set
//!   back to its seed (`diff` 0, and the ledger), and lets go of no host
//!   byte (`dropped_bytes` 0: the churn pool stays in the host set for the
//!   model's life) (mutant: the reset's copies of the seed experts back
//!   skipped).
//! - `stream` (host streaming, `Body38::set_xstream`'s `admit`): from a clear, the
//!   first [`STREAM_PROMPT`] ids of the qwen4exp prose corpus as one prompt
//!   call, then [`STREAM_STEPS`] greedy steps, once with the prompt's
//!   streaming off and once on. On, the call's end leaves a live set other
//!   than the seed on some layer, the call closed, and admitted experts; off,
//!   every live set is the seed. The first decode step's host slots are
//!   fewer on than off. The two runs agree as the unstreamed walk and the
//!   streamed one can (the card and the host sum a token's experts in
//!   another split): the prompt's last logits row within
//!   [`GREEDY_MARGIN`](bloomery_gpu_gates::GREEDY_MARGIN) of the off run's
//!   everywhere, and the greedy ids equal or, where they first part, the on
//!   run's top-1 margin below it (a near tie, V4.1's `gate_ds41_callstream`
//!   `s3` rule). A prompt of one id fewer than `Body38::STREAM_FLOOR`,
//!   streaming on, opens no call (no pick record, no end). Mutants, one a
//!   rule: the call's end returns each layer to
//!   the set it started with (`stream_end` with `kept` false: the live sets
//!   and the first step's host slots are the seed's); the pick admits
//!   nothing (a floor past any count: the same, and no expert admitted); the
//!   card route enqueued before the pick (it reads the call's start words
//!   while the union reads the moved map, so an admitted expert's columns
//!   run nowhere and the logits part); the short prompt's guard removed (its
//!   call opens and ends, admitting nothing).
//! - `split` (the expert stream, `Body38::set_xstream`'s `split`, its rule
//!   at the gate's [`SPLIT_COSTS`]: a host column dear enough that the
//!   balance streams nearly every routed host expert the pick leaves, so the
//!   ring carries hundreds a layer): from a clear, the first
//!   [`SPLIT_PROMPT`] ids of the qwen4exp prose corpus as one prompt call
//!   (one ubatch past the stream's least width), then [`STREAM_STEPS`]
//!   greedy steps, once off and twice split. Count pins: an `xstream` record
//!   per card layer, experts streamed, each record's streamed at most its
//!   tail and the ring's half, and the call's excluded host slots — what
//!   the serve left out, counted by the serve — the records' streamed
//!   columns (none off). The band: a streamed expert's columns run on the
//!   card route's blocks, so the split run agrees with the off run as the
//!   admit run does (`stream`'s rule). Reproducible: the two split runs'
//!   records, prompt logits (bit for bit) and greedy ids are the same.
//!   Mutants, one a rule: the serve's exclusion set dropped (the union
//!   computes the streamed experts again: the count pin and the band); the
//!   ring row's slots reversed against the copies (each streamed column
//!   reads another expert's weights: the band); the floor in the stream's
//!   way back dropped is `xstream`'s unit test's (here every routed host
//!   expert clears the gate's floor).
//! - `slots_drafted`: a drafted pass of two slots' verify rows folds each
//!   slot's accepted rows — slot 0 keeping one of its three rows, slot 1
//!   two of theirs — so the boundary that ends it reports 3 kept rows as
//!   the mask {0, 3, 4}, no prefix (mutant: `commit_slots` passing a prefix
//!   of the kept total).
//! - `commit_counts`: a pass of two slots' three verify rows each, then a
//!   commit with a count a slot cannot keep — slot 0 keeping none, or
//!   slot 1 keeping past its rows while slot 0's count is one it can keep
//!   — refused by name before any slot moves: slot 0 still stands past
//!   its rows and the pass still waits (the session's select and the
//!   model's own refused until its commit); a commit of every row then
//!   lands and a step runs (mutant: each count checked in the keeps' loop,
//!   just before its slot's keep, so slot 0 keeps its row before slot 1's
//!   count is refused).

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

// `pub`: this gate runs neither `table` nor `static`, and its `passes` list
// holds a verify row the shared clause does not build, so a private module
// would count those items dead in this bin.
#[cfg(feature = "gpu")]
#[path = "shared/residency_clauses.rs"]
pub mod residency_clauses;

#[cfg(feature = "gpu")]
mod gate {
    use std::path::Path;
    use std::time::Instant;

    use app::Session;
    use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for;
    use bloomery_gpu::arch::qwen3moe::{Body38, Qwen38Model, XSTREAM_COSTS};
    use bloomery_gpu::host::PassKind;
    use bloomery_gpu::host::swap::{CallReport, PassReport, Residency, SlotState};
    use bloomery_gpu::host::xstream::{Costs, XLayer, XMode, XReport};
    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::{Fnv1a64, GREEDY_MARGIN, GateError, checks_failed, data_dir, verdict};
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
    /// The streaming clause's prompt: one ubatch of real text, whose routing
    /// clears the pick's floor on most card layers.
    const STREAM_PROMPT: usize = 512;
    /// Greedy steps after the streaming clause's prompt.
    const STREAM_STEPS: usize = 32;
    /// The `split` clause's prompt: one ubatch past the stream's least
    /// width at the family's costs and at the gate's.
    const SPLIT_PROMPT: usize = 2048;
    /// The `split` clause's rule costs: the family's, with a host column a
    /// thousand microseconds, so the stream's floor sits under one column and
    /// the balance cuts only the last few experts of a layer.
    const SPLIT_COSTS: Costs = Costs {
        host_us_per_col: 1000.0,
        ..XSTREAM_COSTS
    };

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

    /// The first `n` ids of `$BLOOMERY_DATA/qwen4exp/corpus-prose.ids`, the
    /// Qwen3.8 tokenizer's prose corpus (the runner's `prose:<P>` ids).
    fn prose38(n: usize) -> Result<Vec<u32>, GateError> {
        let path = data_dir().join("qwen4exp").join("corpus-prose.ids");
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
            file, &plan, inputs, 0, host, ub, residency, 1,
        )?)
    }

    /// What a history saw: every call's argmax, each step's logits FNV, each
    /// boundary's ended pass (its kind and kept rows), the flips that
    /// landed, and the steps whose own boundary ran ahead of their readback
    /// ([`ended_ahead`]).
    #[derive(PartialEq)]
    struct History {
        tokens: Vec<u32>,
        fnvs: Vec<u64>,
        passes: Vec<(PassKind, usize)>,
        landed: usize,
        ahead: usize,
    }

    /// Whether the step just run ended at a boundary made ahead of its
    /// readback: the last of `passes`, the boundaries since the step began,
    /// ends a step, was made ahead and carries `before`, the model's
    /// readbacks before the step, and the model has read once since
    /// (`after`).
    fn ended_ahead(passes: &[(PassKind, PassReport)], before: u64, after: u64) -> bool {
        after == before + 1
            && passes
                .last()
                .is_some_and(|(k, r)| *k == PassKind::Step && r.ahead && r.reads == before)
    }

    /// One greedy step of `next` in the history `h`, `want` its readback:
    /// its argmax and logits FNV noted, its boundaries taken into `passes`
    /// and whether it ended ahead counted. The step's argmax.
    fn history_step(
        s: &mut Session<Body38>,
        h: &mut History,
        passes: &mut Vec<(PassKind, PassReport)>,
        next: u32,
        want: Want,
    ) -> Result<u32, GateError> {
        let before = s.model().reads();
        let out = s.step(next, want)?;
        if let Out::Logits { row, .. } = out {
            h.fnvs.push(Fnv1a64::default().f32s(row).value());
        }
        let next = out.argmax();
        h.tokens.push(next);
        let after = s.model().reads();
        let taken = take_passes(s)?;
        h.ahead += usize::from(ended_ahead(&taken, before, after));
        passes.extend(taken);
        Ok(next)
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
            ahead: 0,
        };
        let mut next = s.prompt(ids, Want::Argmax)?.argmax();
        h.tokens.push(next);
        let mut passes = take_passes(s)?;
        next = history_step(s, &mut h, &mut passes, next, Want::Argmax)?;
        let out = s.verify::<3>([next; 3])?;
        h.tokens.extend_from_slice(&out[1..]);
        s.commit(2)?;
        for _ in 0..STEPS {
            next = history_step(s, &mut h, &mut passes, next, Want::Logits)?;
        }
        passes.extend(take_passes(s)?);
        h.landed = passes.iter().map(|(_, r)| r.landed).sum();
        h.passes = passes.iter().map(|&(k, r)| (k, r.kept)).collect();
        Ok(h)
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
        crate::residency_clauses::seeds(machine, b.hybrid().slots().layers())
    }

    /// `transform`: each admitted expert's slot against its static bytes
    /// (three parts an expert: the gate and up, the down).
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
        let layers = crate::residency_clauses::layer_seeds(machine, b.hybrid().slots().layers())?;
        let (checked, bad) =
            crate::residency_clauses::transform_check(m.gpu(), machine, source, &layers, |_| 3)?;
        Ok(crate::residency_clauses::transform_verdict(checked, &bad))
    }

    /// What one run of the streaming clause saw: the greedy ids, the prompt
    /// call's host slots, the first decode step's, the layers whose live set
    /// is not the seed after the call, whether a call is left open, and the
    /// call's end.
    struct StreamRun {
        tokens: Vec<u32>,
        /// The prompt call's last logits row, and each greedy id's top-1
        /// margin, the call's first.
        logits: Vec<f32>,
        margins: Vec<f32>,
        prompt_slots: u64,
        first_slots: u64,
        moved: usize,
        open: bool,
        end: Option<CallReport>,
        /// The host slots the prompt call's serves left out, by the serve's
        /// own count, and the call's `xstream` records.
        excluded: u64,
        xlayers: Vec<(usize, XLayer)>,
        xend: Option<XReport>,
    }

    /// The layers whose live set is not their seed.
    fn moved_layers(s: &Session<Body38>) -> Result<usize, GateError> {
        let m = s.model();
        let b = m.body(NAME)?;
        let machine = b
            .hybrid()
            .swap()
            .ok_or("the load runs no residency machine")?;
        let mut moved = 0;
        for l in b.hybrid().slots().layers() {
            let seed = machine.seed(l)?;
            let live: Vec<u32> = machine
                .ledger()
                .row(l)
                .unwrap_or(&[])
                .iter()
                .filter_map(|st| match st {
                    SlotState::Live(e) => Some(*e),
                    _ => None,
                })
                .collect();
            if live.len() != seed.len() || !seed.iter().all(|e| live.contains(e)) {
                moved += 1;
            }
        }
        Ok(moved)
    }

    /// The prompt calls' host streaming (`Body38::set_xstream`): the
    /// residency pool's pick under `admit`, the pick and the expert stream
    /// under `split`, nothing off.
    fn set_stream(s: &mut Session<Body38>, mode: XMode) -> Result<(), GateError> {
        let (gpu, _, body) = s.model_mut().body_parts(NAME)?;
        Ok(body.set_xstream(gpu, mode)?)
    }

    /// One run of a streaming clause from a clear, the prompt call's
    /// streaming `mode`; the lever off again after it.
    fn stream_run(
        s: &mut Session<Body38>,
        ids: &[u32],
        mode: XMode,
    ) -> Result<StreamRun, GateError> {
        s.clear()?;
        take_passes(s)?;
        set_stream(s, mode)?;
        let stats = |s: &Session<Body38>| -> Result<_, GateError> {
            Ok(s.model().body(NAME)?.hybrid().stats())
        };
        let before = stats(s)?.batch_host_slots;
        let excluded_before = stats(s)?.batch_excluded_slots;
        let out = s.prompt(ids, Want::Logits)?;
        let mut next = out.argmax();
        let Out::Logits { row, .. } = out else {
            return Err("the prompt call returned no logits row".into());
        };
        let logits = row.to_vec();
        let mut margins = vec![margin(&logits)?];
        let prompt_slots = stats(s)?.batch_host_slots - before;
        let excluded = stats(s)?.batch_excluded_slots - excluded_before;
        let (_, end) = s.model_mut().body_parts(NAME)?.2.take_stream_records();
        let (xlayers, xend) = s.model_mut().body_parts(NAME)?.2.take_xstream_records();
        let moved = moved_layers(s)?;
        let open = s
            .model()
            .body(NAME)?
            .hybrid()
            .swap()
            .is_some_and(|m| m.call_open());
        let mut tokens = vec![next];
        let mut first_slots = 0;
        for i in 0..STREAM_STEPS {
            let before = stats(s)?.host_slots;
            let out = s.step(next, Want::Logits)?;
            next = out.argmax();
            if let Out::Logits { row, .. } = out {
                margins.push(margin(row)?);
            }
            if i == 0 {
                first_slots = stats(s)?.host_slots - before;
            }
            tokens.push(next);
        }
        take_passes(s)?;
        set_stream(s, XMode::Off)?;
        Ok(StreamRun {
            tokens,
            logits,
            margins,
            prompt_slots,
            first_slots,
            moved,
            open,
            end,
            excluded,
            xlayers,
            xend,
        })
    }

    /// How a streamed run `on` agrees with the unstreamed `off` (the card and
    /// the host sum a token's experts in another split): the prompt's last
    /// logits row's max |diff|, within [`GREEDY_MARGIN`]; where the greedy
    /// ids first part, if anywhere; and whether they are equal or part at a
    /// near tie (`on`'s top-1 margin there below it).
    fn agree(on: &StreamRun, off: &StreamRun) -> (f32, bool, Option<usize>, bool) {
        let dlogit = on
            .logits
            .iter()
            .zip(&off.logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let logits_ok = on.logits.len() == off.logits.len() && dlogit < GREEDY_MARGIN;
        let parted = on.tokens.iter().zip(&off.tokens).position(|(a, b)| a != b);
        let ids_ok = on.tokens.len() == off.tokens.len()
            && parted.is_none_or(|i| on.margins.get(i).is_some_and(|&m| m < GREEDY_MARGIN));
        (dlogit, logits_ok, parted, ids_ok)
    }

    /// `split`: the expert stream at the gate's costs, off once and split
    /// twice, from a clear each.
    fn split_clause(s: &mut Session<Body38>) -> Result<bool, GateError> {
        let ids = prose38(SPLIT_PROMPT)?;
        let off = stream_run(s, &ids, XMode::Off)?;
        // `split`'s first setting starts the stream; the gate's costs then.
        set_stream(s, XMode::Split)?;
        let body = s.model_mut().body_parts(NAME)?.2;
        body.set_xstream_costs(SPLIT_COSTS)?;
        println!("split: xstream={}", body.xstream_word());
        set_stream(s, XMode::Off)?;
        let a = stream_run(s, &ids, XMode::Split)?;
        let b = stream_run(s, &ids, XMode::Split)?;
        for (u, x) in &a.xlayers {
            record::xstream_layer(*u, x).print();
        }
        if let Some(e) = &a.xend {
            record::xstream_end(e).print();
        }
        let half = a.xend.map_or(0, |e| e.half_slots);
        let streamed: usize = a.xlayers.iter().map(|(_, x)| x.streamed).sum();
        let columns =
            |r: &StreamRun| -> u64 { r.xlayers.iter().map(|(_, x)| x.streamed_columns).sum() };
        let records_ok = !a.xlayers.is_empty()
            && a.xend.is_some_and(|e| e.streamed == streamed)
            && streamed > 0
            && a.xlayers.iter().all(|(_, x)| {
                x.streamed <= x.tail
                    && x.tail <= x.host
                    && x.streamed <= half
                    && (x.streamed == 0) == (x.streamed_columns == 0)
            })
            && off.xlayers.is_empty()
            && off.xend.is_none();
        let count_ok = a.excluded == columns(&a) && b.excluded == columns(&b) && off.excluded == 0;
        let (dlogit, logits_ok, parted, ids_ok) = agree(&a, &off);
        let key = |r: &StreamRun| -> Vec<(usize, usize, usize, usize, u64, u64)> {
            r.xlayers
                .iter()
                .map(|(u, x)| {
                    (
                        *u,
                        x.layer,
                        x.tail,
                        x.streamed,
                        x.streamed_columns,
                        x.host_columns,
                    )
                })
                .collect()
        };
        let same_bits = a.logits.len() == b.logits.len()
            && a.logits
                .iter()
                .zip(&b.logits)
                .all(|(x, y)| x.to_bits() == y.to_bits());
        let repro_ok = key(&a) == key(&b) && same_bits && a.tokens == b.tokens;
        let ok = records_ok && count_ok && logits_ok && ids_ok && repro_ok;
        println!(
            "split: a {SPLIT_PROMPT}-id prompt call: {} xstream records, {streamed} experts \
             streamed (half {half}), each within its tail and the half {records_ok}; the serve's \
             excluded host slots {} / {} / {} (off / split / split again) against the records' \
             streamed columns {} / {} {count_ok}; the prompt's logits max|diff| against off \
             {dlogit} (under {GREEDY_MARGIN}: {logits_ok}), {} greedy ids equal or parted at a \
             near tie {ids_ok}{}; the two split runs' records, logits bits and ids the same \
             {repro_ok}: {}",
            a.xlayers.len(),
            off.excluded,
            a.excluded,
            b.excluded,
            columns(&a),
            columns(&b),
            a.tokens.len(),
            parted
                .map(|i| format!(
                    " (first parted at {i}, split's margin {:?})",
                    a.margins.get(i)
                ))
                .unwrap_or_default(),
            verdict(ok)
        );
        Ok(ok)
    }

    /// The top-1 margin of a logits row; a value that is not finite is
    /// refused by name.
    fn margin(row: &[f32]) -> Result<f32, GateError> {
        let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for (i, &v) in row.iter().enumerate() {
            if !v.is_finite() {
                return Err(format!("logit {i} is {v}").into());
            }
            if v > a {
                b = a;
                a = v;
            } else if v > b {
                b = v;
            }
        }
        Ok(a - b)
    }

    /// `stream`: the prompt call streaming against not, from a clear each.
    fn stream_clause(s: &mut Session<Body38>) -> Result<bool, GateError> {
        let ids = prose38(STREAM_PROMPT)?;
        let off = stream_run(s, &ids, XMode::Off)?;
        let on = stream_run(s, &ids, XMode::Admit)?;
        if let Some(r) = &on.end {
            record::call_report(r).print();
        }
        let admitted = on.end.map_or(0, |r| r.admitted);
        let map_ok = on.moved > 0 && off.moved == 0 && admitted > 0 && !on.open && !off.open;
        let slots_ok = on.first_slots < off.first_slots;
        let (dlogit, logits_ok, parted, ids_ok) = agree(&on, &off);
        // A prompt too short for the floor opens no call.
        let short_ids = prose38(Body38::STREAM_FLOOR as usize - 1)?;
        s.clear()?;
        take_passes(s)?;
        set_stream(s, XMode::Admit)?;
        s.prompt(&short_ids, Want::Argmax)?;
        let (short_picks, short_end) = s.model_mut().body_parts(NAME)?.2.take_stream_records();
        set_stream(s, XMode::Off)?;
        take_passes(s)?;
        let short_ok = short_picks.is_empty() && short_end.is_none();
        println!(
            "stream: a {}-id prompt, streaming on: {} picks, an end {}: {}",
            short_ids.len(),
            short_picks.len(),
            short_end.is_some(),
            verdict(short_ok)
        );
        let ok = map_ok && slots_ok && logits_ok && ids_ok && short_ok;
        println!(
            "stream: a {STREAM_PROMPT}-id prompt call, off / on: layers moved from the seed {} / \
             {} ({admitted} experts admitted, a call left open {} / {}), the prompt's host slots \
             {} / {}, the first step's host slots {} / {}, the prompt's logits max|diff| {dlogit} \
             (under {GREEDY_MARGIN}: {logits_ok}), {} greedy ids equal or parted at a near tie \
             {ids_ok}{}: {}",
            off.moved,
            on.moved,
            off.open,
            on.open,
            off.prompt_slots,
            on.prompt_slots,
            off.first_slots,
            on.first_slots,
            on.tokens.len(),
            parted
                .map(|i| format!(
                    " (first parted at {i}, on's margin {:?})",
                    on.margins.get(i)
                ))
                .unwrap_or_default(),
            verdict(ok)
        );
        Ok(ok)
    }

    /// The load the slots clauses share: planned for the two slots it serves
    /// ([`PlanInputs::plan_with_slots`]; the gate's own load plans one, and
    /// a pass of several slots on it is refused by name), opened after the
    /// gate's session is dropped — the card holds one load — and serving
    /// both slots, the boundaries its load made taken.
    fn open_slots(
        path: &Path,
        inputs: &PlanInputs,
        machine: &Machine,
        host: bloomery_levers::HostCfg,
        ub: usize,
    ) -> Result<Session<Body38>, GateError> {
        let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        if file.architecture() != Some("qwen4exp") {
            return Err(format!("{} is not qwen4exp", path.display()).into());
        }
        let plan = inputs
            .plan_with_slots(
                machine,
                CTX as u64,
                &PlanLevers::default(),
                Experts::Card,
                2,
            )
            .map_err(|e| format!("the plan: {e}"))?;
        let residency = residency_of(&plan)
            .ok_or("the two-slot plan's card experts leave the machine no room")?;
        let mut s = Session::from_model(
            Body38::open_placed_residency(file, &plan, inputs, 0, host, ub, residency, 2)?,
            CTX as u32,
        );
        s.model_mut().body_parts(NAME)?.2.log_residency(0);
        take_passes(&mut s)?;
        s.add_slots(2)?;
        Ok(s)
    }

    /// `slots_drafted`: a drafted pass of two slots' verify rows folds each
    /// slot's accepted rows — slot 0 keeping one of its three rows, slot 1
    /// two of theirs — so the boundary that ends it kept 3 rows as the mask
    /// {0, 3, 4}, row 1 and row 2 rejected, not the prefix a count of the
    /// kept rows folds (mutant: `commit_slots` passing `KeptRows::prefix`
    /// of the kept total). On the slots clauses' load ([`open_slots`]).
    fn slots_drafted_clause(s: &mut Session<Body38>, ids: &[u32]) -> Result<bool, GateError> {
        let mut last = [0u32; 2];
        for (slot, last) in last.iter_mut().enumerate() {
            s.select_slot(slot)?;
            *last = s.prompt(ids, Want::Argmax)?.argmax();
        }
        let drafted: Vec<Vec<u32>> = last.iter().map(|&t| vec![t; 3]).collect();
        let rows: Vec<(usize, &[u32])> = drafted
            .iter()
            .enumerate()
            .map(|(slot, ids)| (slot, &ids[..]))
            .collect();
        s.verify_slots(&rows)?;
        s.commit_slots(&[1, 2])?;
        s.model_mut().pass_boundary()?;
        let (kind, r) = take_passes(s)?
            .pop()
            .ok_or("slots_drafted: the commit's boundary")?;
        let want: u64 = [0usize, 3, 4].iter().map(|&r| 1u64 << r).sum();
        let ok = kind == PassKind::SlotsDrafted && r.kept == 3 && r.rows == want;
        println!(
            "slots_drafted: a pass of two slots' three drafted rows, slot 0 keeping 1 and slot 1 \
             2: the boundary ends a {} pass of {} kept rows, mask 0b{:b} ({{0, 3, 4}}: {}): {}",
            kind.word(),
            r.kept,
            r.rows,
            r.rows == want,
            verdict(ok)
        );
        Ok(ok)
    }

    /// `commit_counts` (module doc), on the slots clauses' load
    /// ([`open_slots`]): slot 0 from a clear and its prompt, slot 1 where
    /// `slots_drafted` left it. The second refused commit keeps slot 0's
    /// row by a count it can keep, so a check made slot by slot would have
    /// moved slot 0 before refusing slot 1.
    fn commit_counts_clause(s: &mut Session<Body38>, ids: &[u32]) -> Result<bool, GateError> {
        s.select_slot(0)?;
        s.clear()?;
        take_passes(s)?;
        let t = s.prompt(ids, Want::Argmax)?.argmax();
        let verified = s.model().pos() + 3;
        s.verify_slots(&[(0, &[t, t, t]), (1, &[t, t, t])])?;
        let under = s.commit_slots(&[0, 3]).err().map(|e| e.to_string());
        let over = s.commit_slots(&[1, 4]).err().map(|e| e.to_string());
        let named = under
            .as_deref()
            .is_some_and(|e| e.contains("slot 0 keeping 0 of its 3 rows"))
            && over
                .as_deref()
                .is_some_and(|e| e.contains("slot 1 keeping 4 of its 3 rows"));
        // The pass leaves slot 0 selected: its position is the model's.
        let at = s.model().pos();
        let unmoved = at == verified;
        let waits = |e: Option<String>| e.is_some_and(|e| e.contains("waits for its commit"));
        let session_waits = waits(s.select_slot(0).err().map(|e| e.to_string()));
        let model_waits = waits(s.model_mut().select_slot(0).err().map(|e| e.to_string()));
        let landed = s.commit_slots(&[3, 3]).is_ok();
        let stepped = landed && s.step(t, Want::Argmax).is_ok();
        let ok = named && unmoved && session_waits && model_waits && stepped;
        println!(
            "commit_counts: a verify of two slots' three rows each, a commit keeping slot 0 none \
             and one keeping slot 1 four (slot 0 one) refused by name {named} ({} / {}), slot 0 \
             at {at} where the verify stood it ({verified}) {unmoved}, the pass waiting for the \
             session {session_waits} and the model {model_waits}, a commit of every row then a \
             step {stepped}: {}",
            under.as_deref().unwrap_or("committed"),
            over.as_deref().unwrap_or("committed"),
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
        let (short, small) =
            crate::residency_clauses::refuse_head(machine, plan.host.headroom_bytes, pool.bytes)?;
        crate::residency_clauses::refuse_tail(short, pool.bytes, REFUSE_BOUND_S, || {
            open(path, inputs, small, *residency, host, ub)
        })
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
            open(path, &inputs, machine.clone(), residency, host, ub)?,
            CTX as u32,
        );
        s.model_mut().body_parts(NAME)?.2.log_residency(0);
        println!("load in {:.1} s", t0.elapsed().as_secs_f64());
        let seeds = seeds(&s)?;

        let ids = prose(PROMPT)?;
        let step =
            crate::residency_clauses::rule_after(&mut s, &ids, rule_of, take_passes, |s, t| {
                s.step(t, Want::Argmax)?;
                Ok(())
            })?;
        let one =
            crate::residency_clauses::rule_after(&mut s, &ids, rule_of, take_passes, |s, t| {
                s.verify::<2>([t, t])?;
                s.commit(1)?;
                Ok(())
            })?;
        let both =
            crate::residency_clauses::rule_after(&mut s, &ids, rule_of, take_passes, |s, t| {
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
        pass &=
            crate::residency_clauses::c1_clause(first.tokens.len(), first.landed, first == again);

        // The prompt call is one pass that keeps 0 rows; a step keeps 1; a
        // verify keeps its accepted rows. A step's own boundary is made ahead
        // of its readback, so the history's last step ends one too.
        // PIN(2026-10-01): STEPS steps after the verify, not STEPS - 1: the last one's boundary runs ahead.
        let mut want = vec![
            (PassKind::None, 0),
            (PassKind::Prompt, 0),
            (PassKind::Step, 1),
            (PassKind::Pair, 2),
        ];
        want.extend(std::iter::repeat_n((PassKind::Step, 1), STEPS));
        let passes_ok = first.passes == want;
        println!(
            "passes: a history's boundaries end none, the prompt call (0 kept), a step (1), a \
             verify (2), then {STEPS} steps (1 kept each): {} boundaries, same {passes_ok}: {}",
            first.passes.len(),
            verdict(passes_ok)
        );
        pass &= passes_ok;

        // Every plain step of the history: the one after the prompt and the
        // STEPS after the verify.
        let order = first.ahead == STEPS + 1;
        println!(
            "order: {} of {} steps ended at a boundary made ahead of their readback, stamped \
             with the readbacks before the step: {}",
            first.ahead,
            STEPS + 1,
            verdict(order)
        );
        pass &= order;

        let r = s
            .residency_reset()?
            .ok_or("the load runs no residency machine")?;
        record::residency_reset(&r).print();
        // The placement `machine` stays bound below (the slots clauses' load
        // plans on it), so the residency machine's borrow stays in the call.
        pass &= crate::residency_clauses::c7_clause(
            &r,
            {
                let m = s.model();
                let b = m.body(NAME)?;
                b.hybrid().swap().ok_or("no machine")?
            },
            &seeds,
        );

        pass &= stream_clause(&mut s)?;
        pass &= split_clause(&mut s)?;
        // The gate's load is done: the card holds one load, and the slots
        // clauses bring their own.
        drop(s);
        let mut s = open_slots(path, &inputs, &machine, host, ub)?;
        pass &= slots_drafted_clause(&mut s, &ids)?;
        pass &= commit_counts_clause(&mut s, &ids)?;

        if pass {
            println!("{NAME}: every clause passed");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
