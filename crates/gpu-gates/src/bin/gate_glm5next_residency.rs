//! GPU gate for GLM's adaptive expert residency on the real model
//! (`bloomery_gpu_glm5next::swap`, `BLOOMERY_RESIDENCY`): the file planned
//! onto the gate card (`workstation::plan_gate`, one card — the trunk's
//! release scope; a plan over more than one card is refused at the body's
//! load), the residency machine over the stage card's routed stacks, the
//! lever set here from the plan's own card slots a layer (the environment
//! cannot move it). The seed is the plan's id prefix. One load; a prompt is
//! fed in batches ([`bloomery_gpu_glm5next::feed`]), one residency pass.
//!
//! A history is: the session cleared (the residency back to its seed), the
//! first [`PROMPT`] ids of an lcg over the vocabulary as one prompt call,
//! then [`STEPS`] greedy steps, each step's argmax and the FNV-1a 64 of its
//! logits row read. Clauses; each names its mutant:
//!
//! - `refuse` (the host-set refusal at load): a machine whose host holds one
//!   byte less than the churn pool needs — the stage card's experts past the
//!   pinned ones, which the host set must hold too — is refused by name
//!   before anything loads, within [`REFUSE_BOUND_S`] (mutant: the churn
//!   pool check removed from the load).
//! - `passes` (a pass counts its kept rows only): a history's boundaries
//!   end, in order, no pass, the prompt call (one pass, 0 rows kept — the
//!   batch service notes no id) and each step (1 kept) (mutant: the call
//!   keeps its rows as a step's kind).
//! - `transform`: after the history, every expert the machine admitted holds
//!   in its slot, part by part, the bytes a static load uploads for it; and
//!   the source's parts are the header's: each card layer's part sizes are
//!   its three stacks' own (each stack's type and shape as the header states
//!   them), the card layers hold one layout, and no layer whose down the
//!   header gives as Q6_K holds a card part (mutant: every part of a layer
//!   sized at its first stack's — the down's slots then sit at the gate's
//!   stride, which the load's stack-size check, the byte check and `c1` all
//!   pass, and the header's down size does not).
//!   PIN(2026-10-01): the file's card layers share one part layout —
//!   `card_routed` admits Q4_K and Q5_K alone and `eligible` needs every
//!   stack of a layer routable, so the one Q5_K gate/up layer, whose down is
//!   Q6_K, has no card slot — and per-layer part sizes are not exercised by
//!   this file.
//! - `steps` (the steps feed refused under the machine): a prompt fed one
//!   decode step an id is refused by name while the machine runs, and the
//!   batch feed still serves after it (mutant: the steps arm of the feed
//!   runs the steps without the refusal).
//! - `c1` (green-only): the history twice gives the same tokens and logits,
//!   and flips land on at least one layer (mutant: the staging thread stages
//!   an expert's first part alone).
//! - `c7`: a residency reset after the history brings every layer's live set
//!   back to its seed (`diff` 0, and the ledger), and lets go of no host
//!   byte (`dropped_bytes` 0: the churn pool stays in the host set for the
//!   model's life) (mutant: the reset copies only the first seed expert
//!   back).

#[cfg(not(feature = "glm5next"))]
fn main() {
    eprintln!(
        "gate_glm5next_residency: built without the `glm5next` feature; see the lead's recipe."
    );
    std::process::exit(2);
}

#[cfg(feature = "glm5next")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_glm5next_residency", gate::run())
}

#[cfg(feature = "glm5next")]
mod gate {
    use std::time::Instant;

    use app::arch::glm5next::GlmCfg;
    use app::{Loaded, OpenLog, Session, SessionError};
    use bloomery_gpu::host::PassKind;
    use bloomery_gpu::host::swap::{Residency, SlotState, SwapMachine, SwapSource};
    use bloomery_gpu::{GpuError, window};
    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use bloomery_gpu_glm5next::{Body, Glm5nextModel, PrefillMode, feed, set_prefill};
    use bloomery_levers::{CARD_BUDGET, CARD_DONTNEED, HOST_LOCK, HOST_POPULATE, R8};
    use gguf::Split;
    use gguf::quant::GgmlType;
    use model::arch::glm5next::names;
    use model::arch::glm5next::place::PlanInputs;
    use model::placement::churn::ChurnPool;
    use model::placement::{Machine, ModelTensors, Plan, PlanLevers, workstation};
    use runtime::{Out, Target, Want};

    const NAME: &str = "gate_glm5next_residency";
    /// Positions the stores hold: a prompt and the steps with room.
    const CTX: usize = 1024;
    /// Prompt positions: one prompt call, one batch.
    const PROMPT: usize = 64;
    /// Greedy steps after the prompt: planning boundaries of the rule.
    const STEPS: usize = 96;
    /// A refusal before the load reads the files' headers and the plan only.
    const REFUSE_BOUND_S: f64 = 120.0;

    /// The lever's word, as the `residency host` record prints it.
    fn word_of(r: Residency) -> String {
        match r {
            Residency::Off => "off".to_string(),
            Residency::Mid { pinned, spares } => format!("mid-p{pinned}-s{spares}"),
        }
    }

    /// The lever the plan's card slots take: one spare a layer (the least —
    /// a flip lands in it), and the pinned count the least of the layers'
    /// slot counts halved, so every layer leaves the machine its bound
    /// (pinned + spares + 1 slots at least) and half the plan's card experts
    /// a layer can churn.
    fn lever_of(plan: &Plan<'_>) -> Result<Residency, GateError> {
        const WHAT: &str = "gate lever";
        let min = plan
            .n_l
            .iter()
            .copied()
            .filter(|&n| n > 0)
            .min()
            .ok_or("the plan puts no routed expert on the card")?;
        let min = usize::try_from(min).map_err(|e| format!("{WHAT}: {e}"))?;
        let pinned = min / 2;
        if min < pinned + 2 {
            return Err(format!(
                "{WHAT}: the plan's least card slots a layer is {min}, fewer than {pinned} \
                 pinned, one spare and one that moves"
            )
            .into());
        }
        Ok(Residency::Mid { pinned, spares: 1 })
    }

    /// `n` ids of an lcg over the vocabulary: every id a row of the
    /// embedding, the routing spread over the experts.
    fn lcg_ids(n: usize, vocab: usize) -> Vec<u32> {
        let mut x = 0x9e37_79b9_7f4a_7c15_u64;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((x >> 33) % vocab as u64) as u32
            })
            .collect()
    }

    /// FNV-1a 64 over a logits row's f32 bits.
    fn fnv(row: &[f32]) -> u64 {
        row.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, v| {
            v.to_bits()
                .to_le_bytes()
                .iter()
                .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
        })
    }

    /// The open's records: none; the gate prints its own lines.
    struct Quiet;

    impl OpenLog<Body> for Quiet {
        fn plan(
            &mut self,
            _: &'static str,
            _: &PlanInputs,
            _: &Machine,
            _: &Plan<'_>,
        ) -> Result<bool, SessionError> {
            Ok(true)
        }

        fn load(&mut self, _: &Glm5nextModel) -> Result<(), SessionError> {
            Ok(())
        }

        fn capture(&mut self, _: usize) -> Result<(), SessionError> {
            Ok(())
        }

        fn prompt_buffers(&mut self, _: &Glm5nextModel) -> Result<(), SessionError> {
            Ok(())
        }
    }

    /// The file planned onto `machine` and loaded under `residency`, its
    /// session's prompts fed in batches.
    fn open(
        path: &str,
        machine: &Machine,
        levers: &bloomery_levers::Levers,
        residency: Residency,
    ) -> Result<Session<Body>, GateError> {
        let file = Split::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let inputs = PlanInputs::read(&file).map_err(|e| format!("{path}: {e}"))?;
        let plan = inputs
            .plan(machine, CTX as u64, &PlanLevers::from_levers(levers)?)
            .map_err(|e| format!("{path}: {e}"))?;
        let ctx = u32::try_from(plan.ctx_max).map_err(|e| format!("{path}: {e}"))?;
        let m = Body::open_placed_with(file, &plan, &inputs, 0, levers.host(), residency)?;
        let cfg = GlmCfg {
            place: PlanLevers::from_levers(levers)?,
            host: levers.host(),
            prefill: PrefillMode::Batch,
        };
        let loaded = Loaded::<Body>::from_model(m, cfg, ctx);
        loaded.ready(&mut Quiet).map_err(Into::into)
    }

    /// The residency machine the loaded body runs, refused by name without
    /// one.
    fn machine_of(s: &Session<Body>) -> Result<&SwapMachine, GateError> {
        Ok(s.model()
            .body(NAME)?
            .hybrid()
            .swap()
            .ok_or("the load runs no residency machine")?)
    }

    /// The layers the stage card holds routed experts of, with their seeds.
    fn seeds(s: &Session<Body>) -> Result<Vec<(usize, Vec<u32>)>, GateError> {
        let b = s.model().body(NAME)?;
        let machine = machine_of(s)?;
        let mut out = Vec::new();
        for l in b.hybrid().slots().layers() {
            let seed = machine.seed(l)?;
            if !seed.is_empty() {
                out.push((l, seed));
            }
        }
        Ok(out)
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

    /// The history from a clear ([`Session::clear`], the residency back to
    /// its seed): the prompt call, then the greedy steps.
    fn history(s: &mut Session<Body>, ids: &[u32]) -> Result<History, GateError> {
        let t = Instant::now();
        s.clear()?;
        let clear_s = t.elapsed().as_secs_f64();
        let (_, _, b) = s.model_mut().body_parts(NAME)?;
        b.take_residency_passes();
        let mut next = s.prompt(ids, Want::Argmax)?.argmax();
        let mut h = History {
            tokens: Vec::with_capacity(STEPS),
            fnvs: Vec::with_capacity(STEPS),
            passes: Vec::new(),
            landed: 0,
        };
        let t = Instant::now();
        for _ in 0..STEPS {
            let out = s.step(next, Want::Logits)?;
            if let Out::Logits { row, .. } = out {
                h.fnvs.push(fnv(row));
            }
            next = out.argmax();
            h.tokens.push(next);
        }
        let steps_s = t.elapsed().as_secs_f64();
        let (_, _, b) = s.model_mut().body_parts(NAME)?;
        let passes = b.take_residency_passes();
        h.landed = passes.iter().map(|(_, r)| r.landed).sum();
        h.passes = passes.iter().map(|&(k, r)| (k, r.kept)).collect();
        println!(
            "history: clear {clear_s:.1} s, prompt + {STEPS} steps in {steps_s:.1} s (runtime \
             values), {} boundaries, {} flips landed",
            h.passes.len(),
            h.landed
        );
        Ok(h)
    }

    /// `refuse`: the host one byte short of the churn pool, on `machine`
    /// with its host shrunk to it.
    fn refuse_clause(
        path: &str,
        machine: &Machine,
        levers: &bloomery_levers::Levers,
        plan: &Plan<'_>,
        residency: Residency,
    ) -> Result<bool, GateError> {
        let Residency::Mid { pinned, .. } = residency else {
            return Err("the gate's lever is off".into());
        };
        let pool = ChurnPool::of(plan, 0, pinned).map_err(|e| format!("{path}: {e}"))?;
        pool.check(plan).map_err(|e| format!("{path}: {e}"))?;
        record::residency_host(&word_of(residency), &pool, plan).print();
        let short = i128::from(machine.host.usable_bytes) - plan.host.headroom_bytes
            + i128::from(pool.bytes)
            - 1;
        let mut small = machine.clone();
        small.host.usable_bytes = u64::try_from(short)?;
        let t0 = Instant::now();
        let opened = open(path, &small, levers, residency);
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

    /// Layer `l`'s parts as the header states them: per stack (gate, up,
    /// down), its type and one expert's bytes from that type and the stack's
    /// first two dims; `None` on a layer without the three stacks.
    fn header_parts(
        model: &ModelTensors,
        l: usize,
    ) -> Result<Option<Vec<(GgmlType, usize)>>, GateError> {
        let names = [
            names::ffn_gate_exps(l),
            names::ffn_up_exps(l),
            names::ffn_down_exps(l),
        ];
        let mut out = Vec::with_capacity(names.len());
        for n in &names {
            let Some(t) = model.tensors.iter().find(|t| &t.name == n) else {
                return Ok(None);
            };
            let (Some(blck), Some(size)) = (t.ty.blck_size(), t.ty.type_size()) else {
                return Err(format!("{n}: {:?}, a type the header cannot size", t.ty).into());
            };
            let [row, rows, ..] = t.dims[..] else {
                return Err(format!("{n}: dims {:?}, not a stack of experts", t.dims).into());
            };
            if !row.is_multiple_of(blck) {
                return Err(
                    format!("{n}: rows of {row} values, not whole blocks of {blck}").into(),
                );
            }
            out.push((t.ty, usize::try_from(size * (row / blck) * rows)?));
        }
        Ok(Some(out))
    }

    /// `transform`: each admitted expert's slot against its static bytes,
    /// part by part; then every layer's parts in the source against the
    /// header's.
    fn transform_clause(
        s: &Session<Body>,
        model: &ModelTensors,
        seeds: &[(usize, Vec<u32>)],
    ) -> Result<bool, GateError> {
        let m = s.model();
        let b = m.body(NAME)?;
        let machine = machine_of(s)?;
        let source = b
            .residency_source()
            .ok_or("the load has no residency source")?;
        let (gpu, stream) = (m.gpu(), m.gpu().stream());
        stream.synchronize()?;
        let mut checked = 0usize;
        let mut bad = Vec::new();
        for (l, seed) in seeds {
            let l = *l;
            let parts = source.part_bytes(l).len();
            let Some(row) = machine.ledger().row(l) else {
                continue;
            };
            for (slot, st) in row.iter().enumerate() {
                let SlotState::Live(e) = *st else { continue };
                if seed.contains(&e) {
                    continue;
                }
                checked += 1;
                for part in 0..parts {
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
        // The source's parts against the header, over every layer: the byte
        // check above reads the source's own part sizes on both sides.
        let mut cards = Vec::new();
        let mut layouts: Vec<Vec<(GgmlType, usize)>> = Vec::new();
        let mut off_header = Vec::new();
        let mut q6k_down = Vec::new();
        let mut on_q6k = Vec::new();
        for l in 0..model.layers {
            let parts = source.part_bytes(l);
            let header = header_parts(model, l)?;
            if header
                .as_ref()
                .is_some_and(|h| h.last().is_some_and(|&(ty, _)| ty == GgmlType::Q6_K))
            {
                q6k_down.push(l);
                if !parts.is_empty() {
                    on_q6k.push(l);
                }
                continue;
            }
            if parts.is_empty() {
                continue;
            }
            cards.push(l);
            match header {
                Some(h) if h.iter().map(|&(_, n)| n).eq(parts.iter().copied()) => {
                    if !layouts.contains(&h) {
                        layouts.push(h);
                    }
                }
                _ => off_header.push(l),
            }
        }
        let layout = layouts
            .iter()
            .map(|h| {
                let parts: Vec<String> = h.iter().map(|(ty, n)| format!("{ty:?} {n} B")).collect();
                format!("[{}]", parts.join(", "))
            })
            .collect::<Vec<_>>()
            .join(" and ");
        let ok = checked > 0
            && bad.is_empty()
            && !cards.is_empty()
            && off_header.is_empty()
            && layouts.len() == 1
            && !q6k_down.is_empty()
            && on_q6k.is_empty();
        println!(
            "transform: {checked} admitted experts on {} card layers, {} parts differing from a \
             static load{}; card parts {layout} (layouts: {}), the header's on all but {off_header:?}; \
             Q6_K-down layers {q6k_down:?}, those with card parts {on_q6k:?}: {}",
            cards.len(),
            bad.len(),
            bad.first()
                .map(|f| format!(" (first {f})"))
                .unwrap_or_default(),
            layouts.len(),
            verdict(ok)
        );
        Ok(ok)
    }

    /// `steps`: the steps feed refused by name under the machine, the batch
    /// feed still serving after it. The served call's positions stand; the
    /// next clause's history clears them.
    fn steps_clause(s: &mut Session<Body>, ids: &[u32]) -> Result<bool, GateError> {
        let refused = {
            let m = s.model_mut();
            set_prefill(m, PrefillMode::Steps)?;
            feed(m, &ids[..8])
        };
        let named = matches!(&refused, Err(GpuError::Shape { detail, .. })
            if detail.contains("steps beside BLOOMERY_RESIDENCY"));
        set_prefill(s.model_mut(), PrefillMode::Batch)?;
        let served = s.prompt(&ids[..8], Want::Argmax).is_ok();
        println!(
            "steps: under the machine the steps feed: {}; the batch feed after it: {}: {}",
            match &refused {
                Ok(t) => format!("token {t}"),
                Err(e) => format!("error \"{e}\""),
            },
            verdict(served),
            verdict(named && served)
        );
        Ok(named && served)
    }

    pub fn run() -> Result<(), GateError> {
        let levers =
            bloomery_levers::at_main(&[CARD_BUDGET, HOST_POPULATE, HOST_LOCK, CARD_DONTNEED, R8])?;
        let path = refset::arch::glm5next::MODEL.to_string();
        let file = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        let inputs = PlanInputs::read(&file).map_err(|e| format!("{path}: {e}"))?;
        let machine = workstation::plan_gate(inputs.model.layers);
        let place = PlanLevers::from_levers(&levers)?;
        let plan = inputs
            .plan(&machine, CTX as u64, &place)
            .map_err(|e| format!("{path}: {e}"))?;
        let residency = lever_of(&plan)?;
        println!(
            "{NAME}: the plan's least card slots a layer, halved for the pinned count, one spare: \
             {}",
            word_of(residency)
        );
        let mut pass = refuse_clause(&path, &machine, &levers, &plan, residency)?;

        let t0 = Instant::now();
        let mut s = open(&path, &machine, &levers, residency)?;
        s.model_mut().body_parts(NAME)?.2.log_residency(0);
        println!(
            "load in {:.1} s (runtime value)",
            t0.elapsed().as_secs_f64()
        );
        let seeds = seeds(&s)?;
        let ids = lcg_ids(PROMPT + 8, inputs.hp.n_vocab);

        let first = history(&mut s, &ids[..PROMPT])?;
        // The prompt call is one pass that keeps 0 rows; each step keeps 1. A
        // step's own boundary is made ahead of its readback, so the history's
        // last step ends one too.
        // PIN(2026-10-01): STEPS steps, not STEPS - 1: the last one's boundary runs ahead.
        let mut want = vec![(PassKind::None, 0), (PassKind::Prompt, 0)];
        want.extend(std::iter::repeat_n((PassKind::Step, 1), STEPS));
        let passes_ok = first.passes == want;
        println!(
            "passes: a history's boundaries end none, the prompt call (0 kept), then {STEPS} steps \
             (1 kept each): {} boundaries, same {passes_ok}: {}",
            first.passes.len(),
            verdict(passes_ok)
        );
        pass &= passes_ok;

        pass &= transform_clause(&s, &inputs.model, &seeds)?;
        pass &= steps_clause(&mut s, &ids)?;

        let again = history(&mut s, &ids[..PROMPT])?;
        let c1 = first == again && first.landed > 0;
        println!(
            "c1: the history twice, {} tokens, {} flips landed: same {}: {}",
            first.tokens.len(),
            first.landed,
            first == again,
            verdict(c1)
        );
        pass &= c1;

        let r = s
            .residency_reset()?
            .ok_or("the load runs no residency machine")?;
        record::residency_reset(&r).print();
        let live_is_seed = {
            let machine = machine_of(&s)?;
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
