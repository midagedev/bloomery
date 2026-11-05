//! GPU gate for Mistral Large 4's adaptive expert residency (`BLOOMERY_RESIDENCY`,
//! `bloomery_gpu_mistrallarge4::resid`): the common machine over the stage card's routed stacks, started by
//! `Body::open_placed_with` under the word the unset rule gives each plan, against the same history under `off`, and the
//! `tier census` record of the decode steps that reads the hit rate.
//!
//! Three placed loads in turn, each under the plan the real tier serves (no card budget; the fixture tier's stage keeps
//! half its routed expert bytes), the plans priced for the prompt group the loads run at (`place::configure_group`, [`GROUP`]):
//!
//! - arm `a` (plan (a), the A6000 alone): a load under `off`, then a load under the word the unset rule resolves
//!   (`residency::at_plan` over the plan: the serving word `mid-p0-s1` where the plan has room);
//! - arm `bp` (plan (b′), the A6000 the stage, the 3090 an expert tier): a load under the same rule's word on its plan. The
//!   rule is `off` (`no_host_experts`) where the plan holds no routed expert on the host, and the fixture's plan (b′) is such
//!   a plan: its 3090 holds every expert the stage does not, so there the arm runs the `off` load beside the tier card and
//!   the real tier's plan, which leaves the host its experts, runs the `mid` one.
//!
//! A history is: the session cleared (the residency back to its seed), a prompt of [`HIST_P`] ids (the first ids of
//! `$BLOOMERY_DATA/mistrallarge4/corpus-prose.ids` in the real tier, the fixed walk of `e2e::walk` in the fixture tier)
//! as one prompt call, then [`STEPS`] greedy steps in [`WINDOWS`] windows of [`WINDOW`], each step's argmax, top-1 margin
//! and the digest of its logits row read, the `tier census` of each window taken. The real tier's `d4096` run is the same
//! history over [`DEEP_P`] prose ids.
//!
//! Clauses (all self-consistency clauses but `residency_hit_depth`, which needs the real shapes). FAIL-first (box): the
//! one-patch mutants of the round's mutant folder name the clause each reddens.
//!
//! - `residency_unset_serves`: on a plan that holds routed experts on the host the unset word is a `mid` word (the serving
//!   word as it is, or the pinned count the plan's card slots or host leave), on a plan that holds none it is `off` with the
//!   reason `no_host_experts` (the machine would have no host expert to move onto the card), and beside a fixed placement and
//!   beside the step feed it is `off`; the real tier's plans hold host experts, and one that holds none fails by name
//!   (mutants: the serving word `off`; the rule's `no_host_experts` branch dropped, which the fixture's plan (b′) reads as
//!   `mid`);
//! - `residency_started`: the `off` load runs no machine and no source; the `mid` load runs a machine whose source holds
//!   parts for every layer the plan gives a card expert and whose seed of each layer is the plan's card experts less the
//!   spares, in id order (mutant: `resid::spec` dropping the lever);
//! - `residency_passes`: a `mid` history's boundaries end none, the prompt call (0 rows kept) and each step (1 kept) (mutant: a
//!   prompt call ending as a step's kind); an `off` history (arm `bp` on the fixture's plan) ends no boundary, its load
//!   running no machine though the gate's log is armed (mutant: the machine-less glue logging a boundary);
//! - `residency_c1`: the history twice gives the same tokens, logits bits and flips, and in the real tier flips land (the
//!   fixture's routing has no skew, so none need) (mutant: a reset that keeps the rule's heat; real tier);
//! - `residency_c7`: the reset brings every seed layer's live set back to its seed, its diff 0, and lets go of no host byte
//!   (the shared clause, with the V4.1 and MiMo gates' FAIL-first);
//! - `residency_near_off`: the `mid` history's greedy ids are the `off` history's, or part from them where the `mid` run's
//!   top-1 margin is under `GREEDY_MARGIN` (a near tie); the prompt's last logits row's greatest distance from the `off` run's
//!   is printed, not judged: the card and the host sum a token's experts in another split under another seed, and no band of
//!   that distance has been measured for this family (no mutant of its own: a flip that copies wrong bytes reddens it in the
//!   real tier, and the V4.1 and GLM gates' byte-level `transform` clause is the guard);
//! - `residency_tier_stays` (arm `bp`): the expert tier card's slot map is the load's after the histories (mutant: the layout
//!   leaving a tier expert out of its away list; real tier, where hot tier experts exist);
//! - `residency_hit_record`: every window's `tier census` record counts the window's steps, its picks are the steps' times the
//!   routed layers' times the router's pick count, its card, arena and file picks sum to its picks, none is streamed, a load
//!   with no machine moves no byte, and the record is printed with its hit rate (mutant: the census counting the routed width
//!   less one);
//! - `residency_hit_depth` (real tier): the same record over the [`DEEP_P`]-id prose prompt on each `mid` load.
//!
//! The hit rate itself is printed with the plan's share of the picks at uniform routing (the card experts of every layer over
//! the router's 128), and judged by no band: the first real run's values are the `[provisional]` pins.

#[cfg(feature = "mistrallarge4")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_mistrallarge4_residency", gate::run())
}

#[cfg(feature = "mistrallarge4")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "mistrallarge4")]
#[allow(
    dead_code,
    reason = "the residency clauses serve several gates; each takes the helpers it reads"
)]
#[path = "shared/residency_clauses.rs"]
mod residency_clauses;

#[cfg(feature = "mistrallarge4")]
mod gate {
    use std::time::Instant;

    use bloomery_gpu::GpuModel;
    use bloomery_gpu::host::PassKind;
    use bloomery_gpu::host::census::TierCensus;
    use bloomery_gpu::host::served::prompt_pass;
    use bloomery_gpu::host::swap::{Residency, SwapSource};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{
        Fnv1a64, GREEDY_MARGIN, GateError, checks_failed, flip, open_split, prose_ids, record,
        residency, verdict,
    };
    use bloomery_gpu_mistrallarge4::body::Body;
    use bloomery_gpu_mistrallarge4::ffn::N_USED;
    use bloomery_gpu_mistrallarge4::prefill::{prefill, set_prefill_group};
    use bloomery_levers::{ResidencyAt, ResidencyWhy, residency_unset};
    use model::arch::Arch;
    use model::arch::mistrallarge4::place::{self, PlanInputs};
    use model::placement::{Machine, Plan, PlanLevers, workstation};

    use crate::e2e::walk;
    use crate::residency_clauses::{c1_clause_with, c7_clause, passes_clause, seeds};

    type Model = GpuModel<Body>;

    const NAME: &str = "gate_mistrallarge4_residency";

    /// The recipe the model file is opened for.
    const RECIPE: &str = "gate-gpu-mistrallarge4-residency";

    /// The host tier's configuration every load opens with: the host set read in, which the residency machine's source
    /// stages its parts from, and no other lever read, so one left in the runner's environment cannot move a clause.
    const HOST_CFG: bloomery_levers::HostCfg = bloomery_levers::HostCfg {
        populate: true,
        lock: false,
        card_dontneed: false,
        r8: false,
        route_trace: false,
    };

    /// The directory under `$BLOOMERY_DATA` of the family's prose ids, tokenized by the family's mainline tokenizer.
    const CORPUS_DIR: &str = "mistrallarge4";

    /// The history's prompt: the depth 6 of the decode headline.
    const HIST_P: usize = 6;

    /// The deep run's prompt (real tier).
    const DEEP_P: usize = 4096;

    /// The steps of one census window, the windows of a run and the steps of a history. A window is 12 planning intervals
    /// of the rule (every 4 kept rows), so a flip made in a window lands in it or the next.
    const WINDOW: usize = 48;
    const WINDOWS: usize = 4;
    const STEPS: usize = WINDOW * WINDOWS;

    /// Positions of the cache beyond the deepest prompt and the steps.
    const CTX_SPARE: usize = 16;

    /// The prompt group the clauses run at: the lever's default, which a serving load runs.
    const GROUP: usize = bloomery_levers::PREFILL_GROUP_DEFAULT as usize;

    /// The context both tiers' plans are made at.
    fn ctx_of(tier_now: Tier) -> u64 {
        let deepest = match tier_now {
            Tier::Real => DEEP_P,
            Tier::Fixture => HIST_P,
        };
        (deepest + STEPS + CTX_SPARE) as u64
    }

    /// One run of a history: what it decoded and counted.
    struct Run {
        /// The prompt's token, then the steps'.
        tokens: Vec<u32>,
        /// Each token's top-1 margin.
        margins: Vec<f32>,
        /// Each step's logits row digest.
        fnvs: Vec<u64>,
        /// The prompt's last logits row.
        logits: Vec<f32>,
        /// Each residency boundary's ended pass: its kind and kept rows.
        passes: Vec<(PassKind, usize)>,
        /// The flips that landed at the boundaries.
        landed: usize,
        /// The `tier census` of each window of steps.
        spans: Vec<TierCensus>,
    }

    impl Run {
        /// The runs' contents: tokens, logits bits, flips.
        fn same(&self, o: &Run) -> bool {
            self.tokens == o.tokens
                && self.fnvs == o.fnvs
                && self.passes == o.passes
                && self.landed == o.landed
                && self.logits.len() == o.logits.len()
                && self
                    .logits
                    .iter()
                    .zip(&o.logits)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
        }
    }

    /// The tier census since the load, refused by name on a load whose host counts no reads.
    fn census_now(m: &Model) -> Result<TierCensus, GateError> {
        record::census_of(m.body(NAME)?.hybrid())
            .ok_or_else(|| "the host tier counts no reads: no tier census".into())
    }

    /// The prompt `ids` from the model's position by the prompt batch, and its token.
    fn prompt(m: &mut Model, ids: &[u32]) -> Result<u32, GateError> {
        let token = prompt_pass(m, |m| {
            m.run_rows(
                ids.len(),
                "mistrallarge4 residency prompt",
                |gpu, w, body, head, pos| prefill(gpu, w, body, head, ids, pos),
            )
        })?;
        token.ok_or_else(|| "the prompt call enqueued no head".into())
    }

    /// The history of `ids` from a reset and the residency's seed: the prompt call, then [`WINDOWS`] windows of
    /// [`WINDOW`] greedy steps.
    fn history(m: &mut Model, ids: &[u32]) -> Result<Run, GateError> {
        m.reset()?;
        m.residency_reset()?;
        let _ = m.body_parts(NAME)?.2.take_residency_passes();
        let mut next = prompt(m, ids)?;
        let logits = m.logits()?;
        let mut tokens = vec![next];
        let mut margins = vec![flip::head_margin(&logits)?];
        let mut fnvs = Vec::with_capacity(STEPS);
        let mut spans = Vec::with_capacity(WINDOWS);
        let mut last = census_now(m)?;
        for _ in 0..WINDOWS {
            for _ in 0..WINDOW {
                next = m.step(&[next])?;
                let row = m.logits()?;
                margins.push(flip::head_margin(&row)?);
                fnvs.push(Fnv1a64::default().f32s(&row).value());
                tokens.push(next);
            }
            let now = census_now(m)?;
            spans.push(now.since(&last));
            last = now;
        }
        let ended = m.body_parts(NAME)?.2.take_residency_passes();
        Ok(Run {
            tokens,
            margins,
            fnvs,
            logits,
            landed: ended.iter().map(|(_, r)| r.landed).sum(),
            passes: ended.iter().map(|&(k, r)| (k, r.kept)).collect(),
            spans,
        })
    }

    /// How `on` agrees with `off` (the card and the host sum a token's experts in another split): the prompt's last logits
    /// row's greatest distance, printed; where the greedy ids first part, if anywhere; and whether they are equal or part at a
    /// near tie (`on`'s top-1 margin there under `GREEDY_MARGIN`).
    fn agree(on: &Run, off: &Run) -> (f32, Option<usize>, bool) {
        let dlogit = on
            .logits
            .iter()
            .zip(&off.logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let parted = on.tokens.iter().zip(&off.tokens).position(|(a, b)| a != b);
        let ids_ok = on.tokens.len() == off.tokens.len()
            && on.logits.len() == off.logits.len()
            && parted.is_none_or(|i| on.margins.get(i).is_some_and(|&m| m < GREEDY_MARGIN));
        (dlogit, parted, ids_ok)
    }

    /// The ids of a history of `n` prompt ids: the real tier's prose (a missing corpus is a named error, with the command
    /// that makes it), the fixture tier's fixed walk.
    fn history_ids(tier_now: Tier, n_vocab: usize, n: usize) -> Result<Vec<u32>, GateError> {
        match tier_now {
            Tier::Fixture => Ok(walk(n, n_vocab)),
            Tier::Real => {
                let ids = prose_ids(CORPUS_DIR, n).map_err(|e| {
                    format!(
                        "{e}: the real tier's histories are prose ids of the family's tokenizer; make \
                         $BLOOMERY_DATA/{CORPUS_DIR}/corpus-prose.ids with the recipe's comment"
                    )
                })?;
                if let Some(bad) = ids.iter().find(|&&id| id as usize >= n_vocab) {
                    return Err(format!(
                        "$BLOOMERY_DATA/{CORPUS_DIR}/corpus-prose.ids holds id {bad}, the vocabulary {n_vocab}"
                    )
                    .into());
                }
                Ok(ids)
            }
        }
    }

    /// The plan's levers: the real tier plans the card's own bytes, the fixture's stage keeps half its experts (so the slot map
    /// has host marks and the machine has experts to move).
    fn plan_levers(
        tier_now: Tier,
        inputs: &PlanInputs,
        machine: &Machine,
        ctx: u64,
    ) -> Result<PlanLevers, GateError> {
        if tier_now == Tier::Real {
            return Ok(PlanLevers::default());
        }
        let all = inputs.plan(machine, ctx, &PlanLevers::default())?;
        let (c, card) = (&all.cards[0], &machine.cards[0]);
        let kept = c.dense_bytes
            + c.rounding_bytes
            + c.kv_bytes
            + c.stage_rows_bytes
            + card
                .set_aside_bytes()
                .ok_or("the card's set-aside is unbounded")?
            + card.margin_bytes
            + inputs.prompt_reserve
            + c.expert_bytes / 2;
        Ok(PlanLevers {
            card_budget_bytes: Some(kept).into(),
            ..PlanLevers::default()
        })
    }

    /// Refused unless `m` runs on the A6000, the card the plans' stage is.
    fn on_stage_card(what: &str, m: &Model) -> Result<(), GateError> {
        let device = m.gpu().device_name()?;
        if !device.contains(workstation::A6000.name) {
            return Err(format!(
                "the {what} opened on {device}, not the {}: the plan's stage card",
                workstation::A6000.name
            )
            .into());
        }
        Ok(())
    }

    /// The load of `plan` under `lever`, set to the prompt group of the clauses.
    fn open(
        what: &str,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        lever: Residency,
    ) -> Result<Model, GateError> {
        let t0 = Instant::now();
        let mut m = Body::open_placed_with(
            open_split(Arch::MistralLarge4, RECIPE)?,
            plan,
            inputs,
            0,
            HOST_CFG,
            lever,
        )?;
        on_stage_card(what, &m)?;
        m.set_mode(StepMode::Eager);
        set_prefill_group(&mut m, GROUP)?;
        m.body_parts(NAME)?.2.log_residency(0);
        println!(
            "{what} load residency={lever:?} resident_bytes={} layers={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            m.layers().len(),
            t0.elapsed().as_secs_f64()
        );
        Ok(m)
    }

    /// The picks the plan's card experts serve at uniform routing: every layer's stage and tier card experts over the router's
    /// experts, as a share of the routed layers' picks. Not a hit rate; the rate the record reads is compared with it.
    fn uniform_share(plan: &Plan<'_>, layers: std::ops::Range<usize>) -> f64 {
        let n_expert = bloomery_gpu_mistrallarge4::ffn::N_EXPERT as f64;
        let held: u64 = layers
            .clone()
            .map(|l| plan.n_l[l] + plan.tier_n_l.iter().map(|t| t[l]).sum::<u64>())
            .sum();
        held as f64 / (layers.len() as f64 * n_expert)
    }

    /// `residency_unset_serves` on `plan`: the unset word at a serving placement, at a fixed placement and beside the step feed,
    /// each as the seat's own two calls resolve it (`residency_unset`, `residency::at_plan`). The serving word is the plan's:
    /// a `mid` word where it holds routed experts on the host, `off` with `no_host_experts` where it holds none, which the
    /// real tier's plans never do. Returns the word a serving load runs and whether the clause held.
    fn unset_word(
        what: &str,
        plan: &Plan<'_>,
        tier_now: Tier,
    ) -> Result<(Residency, bool), GateError> {
        let carry = <Body as bloomery_gpu::model::ChainBody>::UNHOSTED_POOL;
        let at = |serving_place, prefill_steps| ResidencyAt {
            serving_place,
            check_finite: false,
            route_trace: false,
            prefill_steps,
        };
        let serving = residency::at_plan(carry, plan, 0, residency_unset(at(true, false)))?;
        record::residency_lever(serving).print();
        if let Some(d) = serving.why.detail() {
            println!("{d}");
        }
        let word = Residency::parse(serving.word)?;
        let hosted = plan.host.experts > 0;
        let precondition = hosted || tier_now == Tier::Fixture;
        if !precondition {
            println!(
                "FAIL precondition residency_unset_serves: {what}: the real tier's plan holds no routed expert on the host: the \
                 serving word is off by the rule, and the mid clauses prove nothing"
            );
        }
        let on = precondition
            && if hosted {
                matches!(word, Residency::Mid { .. })
                    && matches!(
                        serving.why,
                        ResidencyWhy::Place
                            | ResidencyWhy::Shrunk { .. }
                            | ResidencyWhy::PoolShrunk { .. }
                    )
            } else {
                matches!(word, Residency::Off) && serving.why == ResidencyWhy::NoHostExperts
            };
        let fixed = residency::at_plan(carry, plan, 0, residency_unset(at(false, false)))?;
        let steps = residency::at_plan(carry, plan, 0, residency_unset(at(true, true)))?;
        let off = (fixed.word, fixed.why) == ("off", ResidencyWhy::FixedPlace)
            && (steps.word, steps.why) == ("off", ResidencyWhy::PrefillSteps);
        println!(
            "residency_unset_serves: {what}: the plan holds {} host experts, the serving word {} ({}), beside a fixed placement {} and the step feed {} {}",
            plan.host.experts,
            serving.word,
            serving.why.name(),
            fixed.word,
            steps.word,
            verdict(on && off)
        );
        Ok((word, on && off))
    }

    /// `residency_started` on the `off` load `m`: no machine and no source.
    fn started_off(m: &Model) -> Result<bool, GateError> {
        let body = m.body(NAME)?;
        let (machine, source) = (
            body.hybrid().swap().is_some(),
            body.residency_source().is_some(),
        );
        let ok = !machine && !source;
        println!(
            "residency_started: the off load runs a machine {machine}, holds a source {source} (want neither) {}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// `residency_started` on the `mid` load `m` of `plan` under `word`: a machine, a source with parts for every layer the plan
    /// gives a card expert (a layer whose stacks the card leg does not run, a Mix layer of two gate·up types, holds none), and
    /// each layer's seed the plan's card experts less the spares, in id order.
    fn started_mid(m: &Model, plan: &Plan<'_>, word: Residency) -> Result<bool, GateError> {
        let Residency::Mid { spares, .. } = word else {
            return Err("the mid load's word is off".into());
        };
        let body = m.body(NAME)?;
        let (Some(machine), Some(source)) = (body.hybrid().swap(), body.residency_source()) else {
            println!("residency_started: the mid load runs no machine or holds no source FAIL");
            return Ok(false);
        };
        let layers = body.hybrid().slots().layers();
        let (mut with_parts, mut seeds_ok) = (0usize, true);
        let mut bare = Vec::new();
        for l in layers.clone() {
            if source.part_bytes(l).is_empty() {
                if plan.n_l[l] > 0 {
                    bare.push(l);
                }
            } else {
                with_parts += 1;
            }
            let want: Vec<u32> =
                (0..u32::try_from(plan.n_l[l].saturating_sub(spares as u64))?).collect();
            seeds_ok &= machine.seed(l)? == want;
        }
        let ok = with_parts > 0 && bare.is_empty() && seeds_ok;
        println!(
            "residency_started: the mid load runs a machine over {} routed layers, {with_parts} with source parts, the layers with a card              expert and no part {bare:?}, each seed the plan's card experts less {spares} spare(s) in id order {seeds_ok} {}",
            layers.len(),
            verdict(ok)
        );
        Ok(ok)
    }

    /// `residency_hit_record` for the run `r` named `label`: its windows' `tier census` records printed, each held to the
    /// identities of the record; `machine` whether the load runs one. Prints the windows' hit rates beside `share`.
    fn hit_record(label: &str, r: &Run, routed: usize, machine: bool, share: f64) -> bool {
        let mut ok = !r.spans.is_empty();
        let mut rates = Vec::with_capacity(r.spans.len());
        for (w, s) in r.spans.iter().enumerate() {
            record::census_record(&format!("{label}_w{}", w + 1), s).print();
            ok &= s.tokens == WINDOW as u64
                && s.picks == s.tokens * routed as u64 * N_USED as u64
                && s.adds_up()
                && s.card <= s.picks
                && s.streamed == 0
                && (machine || (s.promote_bytes == 0 && s.demote_bytes == 0));
            rates.push(100.0 * s.card as f64 / s.picks.max(1) as f64);
        }
        let moved: u64 = r.spans.iter().map(|s| s.promote_bytes).sum();
        println!(
            "residency_hit_record: {label}: {} windows of {WINDOW} steps, card picks of picks per window {rates:.2?} % (the plan's \
             share at uniform routing {:.2} %), {} flips landed, {moved} B promoted {}",
            r.spans.len(),
            100.0 * share,
            r.landed,
            verdict(ok)
        );
        ok
    }

    pub fn run() -> Result<(), GateError> {
        let tier_now = Tier::from_env()?;
        let split = open_split(Arch::MistralLarge4, RECIPE)?;
        // The plans are priced for the units of the group every load here runs at, so the load's
        // prompt batch buffers stay inside the reserve the plans keep for them.
        let mut inputs = PlanInputs::describe(&split)?;
        place::configure_group(&mut inputs, GROUP)?;
        drop(split);
        let layers = inputs.hp.n_layer;
        let routed = inputs.hp.dense_lead..layers;
        let ctx = ctx_of(tier_now);
        let ids = history_ids(tier_now, inputs.hp.n_vocab, HIST_P)?;

        let spec = tier::real_card(Place::A.on_host()?.card_specs()?[0]);
        let machine_a = workstation::placed_on(spec, layers, workstation::SCRATCH);
        let plan_a = inputs.plan(
            &machine_a,
            ctx,
            &plan_levers(tier_now, &inputs, &machine_a, ctx)?,
        )?;
        let bp = Place::Bp
            .on_host()?
            .machine(None, Some(place::tier_batch(&inputs)?))?;
        let two = bp(layers);
        let flat = Machine {
            tiers: Vec::new(),
            ..two.clone()
        };
        let plan_bp = inputs.plan(&two, ctx, &plan_levers(tier_now, &inputs, &flat, ctx)?)?;
        for (what, plan) in [("arm a", &plan_a), ("arm bp", &plan_bp)] {
            println!(
                "{what} plan: stage card experts {} tier card experts {} host experts {} (plan values), the card experts serve {:.2} % of \
                 the picks at uniform routing (derived)",
                plan.cards[0].experts,
                plan.cards.get(1).map_or(0, |c| c.experts),
                plan.host.experts,
                100.0 * uniform_share(plan, routed.clone())
            );
        }

        tier::sc("residency_unset_serves")?;
        let (word_a, ok_a) = unset_word("arm a", &plan_a, tier_now)?;
        let (word_bp, ok_bp) = unset_word("arm bp", &plan_bp, tier_now)?;
        let mut pass = ok_a && ok_bp;
        if !pass {
            return Err(checks_failed());
        }

        let (arm_pass, off) = arm_a(
            tier_now,
            (&plan_a, &inputs, ids.as_slice()),
            word_a,
            routed.clone(),
        )?;
        pass &= arm_pass;
        pass &= arm_bp(
            tier_now,
            (&plan_bp, &inputs, ids.as_slice()),
            word_bp,
            &off,
            routed,
        )?;

        println!("{NAME}: {}", tier::tally_line());
        if !pass {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} the unset word is a mid word on each serving plan that holds host experts (off on one that holds \
             none) and off beside a fixed placement and the step feed, a mid load runs a machine seeded by the plan and decodes \
             the off load's ids (or parts at a near tie), its boundaries end none, the prompt call and each step (an off load's \
             none), the history repeats bit for bit and the reset restores the \
             seed, the expert tier's slot map stays, and the windows' tier census records count the steps' picks"
        );
        Ok(())
    }

    /// Arm `a` (module doc): the `off` load's history, then the `mid` load's clauses. Returns whether they held and the `off`
    /// run, which arm `bp` agrees with.
    fn arm_a(
        tier_now: Tier,
        (plan, inputs, ids): (&Plan<'_>, &PlanInputs, &[u32]),
        word: Residency,
        routed: std::ops::Range<usize>,
    ) -> Result<(bool, Run), GateError> {
        let share = uniform_share(plan, routed.clone());
        let mut pass = true;

        tier::sc("residency_started")?;
        let off = {
            let mut m = open("arm a off", plan, inputs, Residency::Off)?;
            pass &= started_off(&m)?;
            let off = history(&mut m, ids)?;
            pass &= hit_record("a_off_d6", &off, routed.len(), false, share);
            off
        };

        let mut m = open("arm a mid", plan, inputs, word)?;
        pass &= started_mid(&m, plan, word)?;

        let first = history(&mut m, ids)?;
        tier::sc("residency_passes")?;
        pass &= passes_clause(&first.passes, STEPS);

        tier::sc("residency_near_off")?;
        let (dlogit, parted, ids_ok) = agree(&first, &off);
        println!(
            "residency_near_off: arm a: the prompt's logits max|diff| against off {dlogit} (printed, not judged), {} greedy ids equal or \
             parted at a near tie{} {}",
            first.tokens.len(),
            parted
                .map(|i| format!(
                    " (first parted at {i}, mid's margin {:?})",
                    first.margins.get(i)
                ))
                .unwrap_or_default(),
            verdict(ids_ok)
        );
        pass &= ids_ok;

        tier::sc("residency_hit_record")?;
        pass &= hit_record("a_mid_d6", &first, routed.len(), true, share);

        tier::sc("residency_c1")?;
        let again = history(&mut m, ids)?;
        pass &= c1_clause_with(
            first.tokens.len(),
            first.landed,
            first.same(&again),
            tier_now == Tier::Real,
        );

        tier::sc("residency_c7")?;
        let reset = m
            .residency_reset()?
            .ok_or("the mid load runs no residency machine")?;
        {
            let body = m.body(NAME)?;
            let machine = body.hybrid().swap().ok_or("the mid load runs no machine")?;
            let lists = seeds(machine, body.hybrid().slots().layers())?;
            pass &= c7_clause(&reset, machine, &lists);
        }

        if tier::run_clause("residency_hit_depth", Tag::Scale)? {
            let deep = history_ids(tier_now, inputs.hp.n_vocab, DEEP_P)?;
            let r = history(&mut m, &deep)?;
            pass &= hit_record(&format!("a_mid_d{DEEP_P}"), &r, routed.len(), true, share);
        }
        Ok((pass, off))
    }

    /// `residency_passes` of an `off` history: its load runs no machine (`ResidencyGlue::off`, whose boundary calls do
    /// nothing), so the log [`open`] armed holds no boundary.
    fn no_passes(passes: &[(PassKind, usize)]) -> bool {
        let none = passes.is_empty();
        println!(
            "passes: an off load runs no machine, so its history ends no boundary: {} boundaries, same {none}: {}",
            passes.len(),
            verdict(none)
        );
        none
    }

    /// Arm `bp` (module doc): the load under the word the rule gives the plan with an expert tier card: `mid` where the plan
    /// holds host experts, else `off` (the fixture's plan), whose load runs no machine.
    fn arm_bp(
        tier_now: Tier,
        (plan, inputs, ids): (&Plan<'_>, &PlanInputs, &[u32]),
        word: Residency,
        off: &Run,
        routed: std::ops::Range<usize>,
    ) -> Result<bool, GateError> {
        let share = uniform_share(plan, routed.clone());
        let tier_experts = plan.cards.get(1).map_or(0, |c| c.experts);
        let mut pass = true;
        if tier_experts == 0 {
            println!(
                "FAIL precondition residency_tier_stays: the plan holds no expert on the tier: the run proves nothing"
            );
            pass = false;
        }
        let mid = matches!(word, Residency::Mid { .. });
        let name = if mid { "mid" } else { "off" };
        let mut m = open(&format!("arm bp {name}"), plan, inputs, word)?;
        pass &= if mid {
            started_mid(&m, plan, word)?
        } else {
            started_off(&m)?
        };
        let before = m.body(NAME)?.hybrid().slots().tier_view(0)?;

        let r = history(&mut m, ids)?;
        tier::sc("residency_near_off")?;
        let (dlogit, parted, ids_ok) = agree(&r, off);
        println!(
            "residency_near_off: arm bp: the prompt's logits max|diff| against arm a's off {dlogit} (printed, not judged), {} greedy ids \
             equal or parted at a near tie{} {}",
            r.tokens.len(),
            parted
                .map(|i| format!(
                    " (first parted at {i}, mid's margin {:?})",
                    r.margins.get(i)
                ))
                .unwrap_or_default(),
            verdict(ids_ok)
        );
        pass &= ids_ok;

        tier::sc("residency_passes")?;
        pass &= if mid {
            passes_clause(&r.passes, STEPS)
        } else {
            no_passes(&r.passes)
        };
        tier::sc("residency_hit_record")?;
        pass &= hit_record(&format!("bp_{name}_d6"), &r, routed.len(), mid, share);

        if tier::run_clause("residency_hit_depth", Tag::Scale)? {
            let deep = history_ids(tier_now, inputs.hp.n_vocab, DEEP_P)?;
            let d = history(&mut m, &deep)?;
            pass &= hit_record(
                &format!("bp_{name}_d{DEEP_P}"),
                &d,
                routed.len(),
                mid,
                share,
            );
        }

        tier::sc("residency_tier_stays")?;
        let after = m.body(NAME)?.hybrid().slots().tier_view(0)?;
        let stays = before == after;
        println!(
            "residency_tier_stays: the expert tier card's slot map after the histories ({} flips landed) is the load's {stays} {}",
            r.landed,
            verdict(stays)
        );
        pass &= stays;
        Ok(pass)
    }
}
