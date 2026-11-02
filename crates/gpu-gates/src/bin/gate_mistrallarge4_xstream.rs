//! GPU gate for Mistral Large 4's prompt calls through the common walk's host streaming (`BLOOMERY_XSTREAM`,
//! `bloomery_gpu_mistrallarge4::stream`): `split` (the expert stream: the host experts the rule sends to the card land in
//! the ring's half and run there as route GEMMs beside the card's stacks) and `admit` (V4.1's host streaming under the
//! lever's name: the residency's pick), each against the same prompt with the lever `off`.
//!
//! Two arms, each one placed load of its plan under a residency machine (`Body::open_placed_with`), which the lever's
//! `admit` and `split` run on:
//!
//! - arm `a`, plan (a) (the A6000 alone, the host tier beside it) under a card budget the gate sets: the plan keeps the
//!   stream's ring in its prompt reserve (`PlanInputs::with_stream`), and the card keeps, beside the ring, what the stage
//!   holds without experts plus [`REAL_EXPERTS`] of them in the real tier or half the experts the card would hold in the
//!   fixture tier, so every card layer leaves the residency its pinned, spare and moving experts (`residency_of`);
//! - arm `bp`, plan (b′) (the stage on the A6000, an expert tier on the 3090) under the same budget of the stage card,
//!   the plan keeping the stream's ring as arm `a`'s does: a tier layer's join sums the stage card's, the streamed and
//!   the tier's slots by the stream's union places, so `split` streams there as on arm `a`.
//!
//! The prompt is [`STREAM_P`] ids of the fixed walk of `e2e::walk` from a reset, the residency back at its seed, at the
//! default prompt group (one walk of two units), then [`STEPS`] greedy steps. A streamed expert's columns run on the card's
//! route GEMM where the host union would run them, so the streamed run's bits are the band's, not the unstreamed run's:
//! the clauses hold counts, the contract in the ids and the streamed run to itself bit for bit.
//!
//! Clauses (all self-consistency clauses: each compares the engine's runs to each other and reads no oracle, so both
//! tiers run them):
//! - `xstream_started` (arm `a`): the load's first `split` starts the stream: the lever's word says `split`, the ring's
//!   half holds slots, and a `split` refused by name fails the clause with the refusal printed;
//! - `xstream_split_streams`: the `split` run's records and counts agree with each other: some layer streamed experts
//!   (or the run proves nothing and fails by name), each layer's streamed experts are within its tail, the host experts
//!   it routes and the ring's half, a layer streams columns exactly when it streams experts, the call's end counts the
//!   layers' sum, and the host slots the batch services left out agree with the records by the walk's timing rows
//!   (`set_prompt_stats`, one row a layer-batch): a walk's units share one pick a layer, which prices the first unit's
//!   routing at the walk's width, so each streamed record's columns are its walk's units times the slots its walk's
//!   first unit left out at the layer, exactly; the walk's later units left out slots on the streamed layers (they share
//!   the layer's stream), no layer-batch of a layer no record streams left out any, and the rows sum to the serve's
//!   counter (`HybridStats::batch_excluded_slots`); the `off` run's records are empty and it left out none;
//! - `xstream_split_repeats`: a second `split` run's records, prompt logits (bit for bit) and greedy ids are the first's;
//! - `xstream_split_near_off`: the `split` run's greedy ids are the `off` run's, or part from them where the `split`
//!   run's top-1 margin is under `GREEDY_MARGIN` (a near tie); the prompt's last logits row's greatest distance from the
//!   `off` run's is printed, not judged: `GREEDY_MARGIN` is the top-1 gap at a parted id, not a bound on that distance
//!   over the vocabulary, and no band of it has been measured for this family;
//! - `xstream_admit_opens_call`: under `admit` a prompt of [`STREAM_P`] ids opens a call (its end report is there) and
//!   streams no expert, its ids agree with the `off` run's as above, and a prompt of one id fewer than
//!   [`STREAM_FLOOR`] opens none (no pick, no end);
//! - `xstream_split_beside_tier` (arm `bp`): `split` starts the stream beside the expert tier card, and a `split` run
//!   streams experts on some layer the tier holds experts of (or the run proves nothing and fails by name); the call's
//!   end counts the layers' sum and the host's left-out slots agree with the records as `xstream_split_streams` reads
//!   them; every unit's back joined
//!   each streamed layer by the records' streamed count and no back joined a layer no record names
//!   (`Body::take_stream_joined`); a second `split` run's records, joins, prompt logits (bit for bit) and greedy ids are
//!   the first's; the `split` run's ids agree with the `off` run's as `xstream_split_near_off` reads them; and `admit`,
//!   the unset lever's mode there, opens a call over the same prompt that streams and joins nothing, its ids agreeing
//!   with the `off` run's.
//!
//! The taps are not armed. FAIL-first (box): the one-patch mutants under the round's mutant folder name the clause each
//! reddens.

#[cfg(feature = "mistrallarge4")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_mistrallarge4_xstream", gate::run())
}

#[cfg(feature = "mistrallarge4")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "mistrallarge4")]
mod gate {
    use std::time::Instant;

    use bloomery_gpu::GpuModel;
    use bloomery_gpu::host::served::prompt_pass;
    use bloomery_gpu::host::swap::{CallPick, CallReport, Residency};
    use bloomery_gpu::host::xstream::{Costs, XLayer, XMode, XReport};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::prompt_timing::PromptStats;
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::tier::{self, Tier};
    use bloomery_gpu_gates::{GREEDY_MARGIN, GateError, checks_failed, flip, open_split, verdict};
    use bloomery_gpu_mistrallarge4::body::Body;
    use bloomery_gpu_mistrallarge4::prefill::{
        prefill, set_prefill_group, set_prompt_stats, take_prompt_stats,
    };
    use bloomery_gpu_mistrallarge4::stream::STREAM_FLOOR;
    use model::arch::Arch;
    use model::arch::mistrallarge4::place::{self, PlanInputs};
    use model::placement::{Machine, Plan, PlanLevers, workstation};

    use crate::e2e::walk;

    type Model = GpuModel<Body>;

    const NAME: &str = "gate_mistrallarge4_xstream";

    /// The recipe the model file is opened for.
    const RECIPE: &str = "gate-gpu-mistrallarge4-xstream";

    /// The host tier's configuration both loads open with: the host set read in, which the residency machine's source
    /// stages its parts from, and no other lever read, so one left in the runner's environment cannot move a clause.
    const HOST_CFG: bloomery_levers::HostCfg = bloomery_levers::HostCfg {
        populate: true,
        lock: false,
        card_dontneed: false,
        r8: false,
        route_trace: false,
    };

    /// The prompt's length: two whole units of the plan's prompt rows, one walk of the default prompt group.
    const STREAM_P: usize = 2 * place::PROMPT_ROWS;

    /// The prompt group the clauses run at: the lever's default, which a serving load runs.
    const GROUP: usize = bloomery_levers::PREFILL_GROUP_DEFAULT as usize;

    /// Greedy steps after the prompt.
    const STEPS: usize = 8;

    /// Positions of the cache beyond the prompt and the steps.
    const CTX_SPARE: usize = 16;

    /// The routed expert bytes the stage card keeps beside the ring in the real tier: at 23.5 MB an expert, about five a
    /// layer over the 71 routed layers, so every card layer leaves the residency its pinned expert, its spare and its
    /// moving slot [derived].
    const REAL_EXPERTS: u64 = 8 << 30;

    /// The stream's fixed rule input: a host column of a tenth of a second, so the stream's floor sits under one column and
    /// every host expert a unit routes streams until the ring's half or the balance cuts it. The host union's fixed cost of
    /// an expert is held under the card's fixed cost of a streamed one: the rule refuses a fixed cost past one expert's
    /// lane copy plus the card's fixed cost (`stream_tail`), and a copy of the fixture's experts (about 0.17 MB, tens of
    /// microseconds at the lane's rate [derived]) is shorter than the real ones' (23.5 MB), so only a fixed cost under the
    /// card's own holds at every expert size. A column at the host's per-column cost is dearer than any fixed cost, so
    /// the value moves no decision of the stream.
    const SPLIT_COSTS: Costs = Costs {
        host_us_per_col: 1.0e5,
        host_us_fixed: 45.0,
        card_us_fixed: 90.0,
        card_us_per_col: 2.0,
    };

    const _: () = assert!(
        SPLIT_COSTS.host_us_fixed <= SPLIT_COSTS.card_us_fixed
            && SPLIT_COSTS.host_us_fixed < SPLIT_COSTS.host_us_per_col,
        "SPLIT_COSTS.host_us_fixed must stay at most the card's fixed cost, or the rule refuses a layer of small experts",
    );

    /// One run of the prompt: what it moved and what the steps after it read.
    struct Run {
        /// The prompt's token, then the greedy steps'.
        tokens: Vec<u32>,
        /// Each token's top-1 margin.
        margins: Vec<f32>,
        /// The prompt's last logits row.
        logits: Vec<f32>,
        /// The host slots the batch services left out during the prompt call.
        excluded: u64,
        /// The same by layer-batch, from the walk's timing rows ([`left_rows`]).
        left: Vec<LeftRow>,
        /// The call's expert stream records and the residency's picks.
        xlayers: Vec<(usize, XLayer)>,
        xend: Option<XReport>,
        picks: Vec<(usize, CallPick)>,
        end: Option<CallReport>,
        /// The call's ring joins: per unit back, its walk, layer and streamed experts.
        joined: Vec<(usize, usize, usize)>,
    }

    /// The host slots the batch services left out of their union since `m` loaded.
    fn left_out(m: &Model) -> Result<u64, GateError> {
        Ok(m.body(NAME)?.hybrid().stats().batch_excluded_slots)
    }

    /// One layer-batch's serve: its walk, layer, unit within the walk and the host slots it left out.
    struct LeftRow {
        walk: usize,
        layer: usize,
        unit: usize,
        slots: u64,
    }

    /// The walk's timing rows as [`LeftRow`]s; empty where no batch walked. A row's walk is its batch's less its unit,
    /// over [`GROUP`]: every walk before the last runs [`GROUP`] units.
    fn left_rows(stats: Option<PromptStats>) -> Result<Vec<LeftRow>, GateError> {
        let Some(stats) = stats else {
            return Ok(Vec::new());
        };
        stats
            .rows
            .iter()
            .map(|r| -> Result<LeftRow, GateError> {
                let first = usize::try_from(r.b)?.checked_sub(r.unit).ok_or_else(|| {
                    format!("a timing row of batch {} names unit {}", r.b, r.unit)
                })?;
                Ok(LeftRow {
                    walk: first / GROUP,
                    layer: r.layer,
                    unit: r.unit,
                    slots: r.excluded,
                })
            })
            .collect()
    }

    /// How a run's left-out host slots agree with its stream records (`xstream_split_streams`'s count term).
    struct LeftOut {
        /// The slots the walks' first units left out on the streamed layers, and their later units.
        first: u64,
        later: u64,
        /// The slots left out on a layer of a walk no record streams.
        stray: u64,
        /// The records whose columns are not their walk's units times the first unit's left-out slots, named.
        bad: Vec<String>,
        ok: bool,
    }

    /// [`LeftOut`] of `r`, a call of `units` units: a layer's pick prices the first unit's routing of its walk at the
    /// walk's width (each routed id once a unit of the walk), so each streamed record's columns are the walk's units
    /// times the slots the walk's first unit left out at the layer; the later units, which share the layer's stream,
    /// leave out their own routing's slots of the same experts (some, where a streamed walk has one); no layer no
    /// record streams leaves any out; and the rows sum to the serve's counter.
    fn left_out_of(r: &Run, units: usize) -> LeftOut {
        let walk_units = |w: usize| units.saturating_sub(w * GROUP).min(GROUP);
        let at = |w: usize, l: usize, lead: bool| -> u64 {
            r.left
                .iter()
                .filter(|x| x.walk == w && x.layer == l && (x.unit == 0) == lead)
                .map(|x| x.slots)
                .sum()
        };
        let streams: Vec<(usize, XLayer)> = r
            .xlayers
            .iter()
            .filter(|(_, x)| x.streamed > 0)
            .copied()
            .collect();
        let (mut first, mut later, mut shared) = (0u64, 0u64, false);
        let mut bad = Vec::new();
        for &(w, x) in &streams {
            let k = walk_units(w);
            let f = at(w, x.layer, true);
            if x.streamed_columns != k as u64 * f {
                bad.push(format!(
                    "walk {w} layer {}: {} columns against {k} × {f}",
                    x.layer, x.streamed_columns
                ));
            }
            first += f;
            later += at(w, x.layer, false);
            shared |= k > 1;
        }
        let stray: u64 = r
            .left
            .iter()
            .filter(|x| {
                !streams
                    .iter()
                    .any(|&(w, s)| w == x.walk && s.layer == x.layer)
            })
            .map(|x| x.slots)
            .sum();
        let rows: u64 = r.left.iter().map(|x| x.slots).sum();
        let ok = !r.left.is_empty()
            && bad.is_empty()
            && stray == 0
            && rows == r.excluded
            && (!shared || later > 0);
        LeftOut {
            first,
            later,
            stray,
            bad,
            ok,
        }
    }

    /// `lo`'s bad records, the first few, for a red line.
    fn bad_records(clause: &str, lo: &LeftOut) {
        if !lo.bad.is_empty() {
            println!(
                "{clause}: {} records off their walk's units times the first unit's left-out slots: {}",
                lo.bad.len(),
                lo.bad
                    .iter()
                    .take(8)
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join("; ")
            );
        }
    }

    /// `mode` on the load's body (the stream's first `split` starts it).
    fn set_mode(m: &mut Model, mode: XMode) -> Result<(), GateError> {
        let (gpu, _, body) = m.body_parts(NAME)?;
        body.set_xstream(gpu, mode)?;
        Ok(())
    }

    /// The prompt `ids` from the model's position by the prompt batch, and its token.
    fn prompt(m: &mut Model, ids: &[u32]) -> Result<u32, GateError> {
        let token = prompt_pass(m, |m| {
            m.run_rows(
                ids.len(),
                "mistrallarge4 xstream prompt",
                |gpu, w, body, head, pos| prefill(gpu, w, body, head, ids, pos),
            )
        })?;
        token.ok_or_else(|| "the prompt call enqueued no head".into())
    }

    /// The prompt `ids` from a reset and the residency's seed under `mode`, then [`STEPS`] greedy steps, the stream records
    /// taken at the call's end; the lever is left `off`.
    fn run_once(m: &mut Model, ids: &[u32], mode: XMode) -> Result<Run, GateError> {
        m.reset()?;
        m.residency_reset()?;
        set_mode(m, mode)?;
        {
            let body = m.body_parts(NAME)?.2;
            let _ = body.take_stream_records();
            let _ = body.take_xstream_records();
            let _ = body.take_stream_joined();
        }
        let _ = take_prompt_stats(m)?;
        let before = left_out(m)?;
        let mut next = prompt(m, ids)?;
        let logits = m.logits()?;
        let excluded = left_out(m)? - before;
        let left = left_rows(take_prompt_stats(m)?)?;
        let (picks, end, xlayers, xend, joined) = {
            let body = m.body_parts(NAME)?.2;
            let (picks, end) = body.take_stream_records();
            let (xlayers, xend) = body.take_xstream_records();
            (picks, end, xlayers, xend, body.take_stream_joined())
        };
        let mut tokens = vec![next];
        let mut margins = vec![flip::head_margin(&logits)?];
        for _ in 0..STEPS {
            next = m.step(&[next])?;
            tokens.push(next);
            margins.push(flip::head_margin(&m.logits()?)?);
        }
        set_mode(m, XMode::Off)?;
        Ok(Run {
            tokens,
            margins,
            logits,
            excluded,
            left,
            xlayers,
            xend,
            picks,
            end,
            joined,
        })
    }

    /// How a streamed run `on` agrees with the unstreamed `off` (the card and the host sum a token's experts in another
    /// split): the prompt's last logits row's greatest distance, printed; where the greedy ids first part, if anywhere;
    /// and whether they are equal or part at a near tie (`on`'s top-1 margin there under `GREEDY_MARGIN`).
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

    /// The joins `r`'s records ask for, sorted: each streamed layer's `(walk, layer, streamed)` once a unit of its walk,
    /// the call's `units` units cut into walks of [`GROUP`] units (the last holds the rest).
    fn joins_wanted(r: &Run, units: usize) -> Vec<(usize, usize, usize)> {
        let mut want = Vec::new();
        for (w, x) in r.xlayers.iter().filter(|(_, x)| x.streamed > 0) {
            let in_walk = units.saturating_sub(w * GROUP).min(GROUP);
            for _ in 0..in_walk {
                want.push((*w, x.layer, x.streamed));
            }
        }
        want.sort_unstable();
        want
    }

    /// `r`'s joins that summed streamed experts, sorted.
    fn joins_got(r: &Run) -> Vec<(usize, usize, usize)> {
        let mut got: Vec<_> = r.joined.iter().copied().filter(|j| j.2 > 0).collect();
        got.sort_unstable();
        got
    }

    /// A run's records as the repeat clause compares them.
    fn key(r: &Run) -> Vec<(usize, usize, usize, usize, u64, u64)> {
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
    }

    /// The residency over `plan`: half the smallest card layer's experts pinned (the seed is the plan's id prefix, not a
    /// frequency list), one spare a layer. Refused by name: a plan whose smallest layer of card experts holds no pinned
    /// expert beside the spare and the slot the machine swaps into.
    fn residency_of(plan: &Plan<'_>) -> Result<Residency, GateError> {
        let slots = plan
            .n_l
            .iter()
            .copied()
            .filter(|&n| n > 0)
            .min()
            .ok_or("the plan puts no expert on the card")?;
        let slots = usize::try_from(slots)?;
        let pinned = slots / 2;
        if pinned == 0 || slots < pinned + 2 {
            return Err(format!(
                "the plan keeps {slots} card experts on its smallest layer: no residency machine of a pinned expert, a \
                 spare and its swap slot"
            )
            .into());
        }
        println!("residency mid pinned={pinned} spares=1 over {slots} card experts (plan values)");
        Ok(Residency::Mid { pinned, spares: 1 })
    }

    /// A card budget that keeps the stage card's bytes without experts (its dense, rounding, cache and stage rows,
    /// set-aside, margin and `inputs`' prompt reserve, the stream's ring among it) plus the share of the routed expert
    /// bytes the stage would keep unbudgeted on `machine`, the stage card alone: half of them in the fixture tier,
    /// [`REAL_EXPERTS`] at most in the real tier.
    fn stage_budget(
        inputs: &PlanInputs,
        machine: &Machine,
        ctx: u64,
        tier_now: Tier,
    ) -> Result<PlanLevers, GateError> {
        let all = inputs.plan(machine, ctx, &PlanLevers::default())?;
        let (c, card) = (&all.cards[0], &machine.cards[0]);
        let no_experts = c.dense_bytes
            + c.rounding_bytes
            + c.kv_bytes
            + c.stage_rows_bytes
            + card
                .set_aside_bytes()
                .ok_or("the card's set-aside is unbounded")?
            + card.margin_bytes
            + inputs.prompt_reserve;
        let experts = match tier_now {
            Tier::Fixture => c.expert_bytes / 2,
            Tier::Real => c.expert_bytes.min(REAL_EXPERTS),
        };
        Ok(PlanLevers {
            card_budget_bytes: Some(no_experts + experts).into(),
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

    /// The load of `plan` under its residency, set to the prompt group of the clauses, its walks' timing armed for the
    /// left-out slots by layer-batch ([`left_rows`]; every run armed alike).
    fn open(what: &str, plan: &Plan<'_>, inputs: &PlanInputs) -> Result<Model, GateError> {
        let residency = residency_of(plan)?;
        let t0 = Instant::now();
        let mut m = Body::open_placed_with(
            open_split(Arch::MistralLarge4, RECIPE)?,
            plan,
            inputs,
            0,
            HOST_CFG,
            residency,
        )?;
        on_stage_card(what, &m)?;
        m.set_mode(StepMode::Eager);
        set_prefill_group(&mut m, GROUP)?;
        set_prompt_stats(&mut m, true)?;
        println!(
            "{what} load resident_bytes={} layers={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            m.layers().len(),
            t0.elapsed().as_secs_f64()
        );
        Ok(m)
    }

    /// The ring's slots a half in the lever's word (`xstream_half=N`), 0 where the word has none.
    fn half_of(word: &str) -> usize {
        word.split_whitespace()
            .find_map(|w| w.strip_prefix("xstream_half="))
            .and_then(|n| n.parse().ok())
            .unwrap_or(0)
    }

    pub fn run() -> Result<(), GateError> {
        let tier_now = Tier::from_env()?;
        let mut pass = arm_a(tier_now)?;
        pass &= arm_bp(tier_now)?;
        println!("{NAME}: {}", tier::tally_line());
        if !pass {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} the stream starts under a residency machine, a split prompt call streams experts whose \
             counts agree with the serve's left-out slots, repeats bit for bit and keeps the unstreamed run's ids (or \
             parts at a near tie), admit opens a call a prompt of the floor opens and none below it, and beside an \
             expert tier split streams on the tier's layers, joins them by the records' counts, repeats bit for bit \
             and keeps the unstreamed run's ids"
        );
        Ok(())
    }

    /// Arm `a` (module doc): `xstream_started`, `xstream_split_streams`, `xstream_split_repeats`,
    /// `xstream_split_near_off` and `xstream_admit_opens_call`.
    fn arm_a(tier_now: Tier) -> Result<bool, GateError> {
        let split = open_split(Arch::MistralLarge4, RECIPE)?;
        let inputs = PlanInputs::describe(&split)?.with_stream()?;
        let layers = inputs.hp.n_layer;
        let ctx = u64::try_from(STREAM_P + STEPS + CTX_SPARE)?;
        let spec = tier::real_card(Place::A.on_host()?.card_specs()?[0]);
        let machine = workstation::placed_on(spec, layers, workstation::SCRATCH);
        let levers = stage_budget(&inputs, &machine, ctx, tier_now)?;
        let plan = inputs.plan(&machine, ctx, &levers)?;
        println!(
            "arm a plan: card experts {} ({} B) host experts {} ({} B), prompt reserve {} B with the stream's ring \
             (plan values)",
            plan.cards[0].experts,
            plan.cards[0].expert_bytes,
            plan.host.experts,
            plan.host.expert_bytes,
            inputs.prompt_reserve,
        );
        let mut m = open("arm a", &plan, &inputs)?;
        let ids = walk(STREAM_P, inputs.hp.n_vocab);
        let mut pass = true;

        tier::sc("xstream_started")?;
        let started = {
            let (gpu, _, body) = m.body_parts(NAME)?;
            match body.set_xstream(gpu, XMode::Split) {
                Ok(()) => {
                    body.set_xstream_costs(SPLIT_COSTS)?;
                    let word = body.xstream_word();
                    let half = half_of(&word);
                    let ok = body.xstream_mode() == XMode::Split && half > 0;
                    println!(
                        "xstream_started: xstream={word}: a ring of {half} slots a half {}",
                        verdict(ok)
                    );
                    ok
                }
                Err(e) => {
                    println!("xstream_started: split refused: {e} FAIL");
                    false
                }
            }
        };
        pass &= started;
        if !started {
            return Ok(false);
        }
        set_mode(&mut m, XMode::Off)?;

        let off = run_once(&mut m, &ids, XMode::Off)?;
        let a = run_once(&mut m, &ids, XMode::Split)?;

        tier::sc("xstream_split_streams")?;
        let streamed: usize = a.xlayers.iter().map(|(_, x)| x.streamed).sum();
        let columns: u64 = a.xlayers.iter().map(|(_, x)| x.streamed_columns).sum();
        let half = a.xend.map_or(0, |e| e.half_slots);
        if streamed == 0 {
            println!(
                "FAIL precondition xstream_split_streams: the {STREAM_P}-id prompt streamed no expert (a plan that \
                 leaves no host expert, or a ring too small): the run proves nothing"
            );
        }
        let records_ok = !a.xlayers.is_empty()
            && streamed > 0
            && a.xend.is_some_and(|e| e.streamed == streamed)
            && a.xlayers.iter().all(|(_, x)| {
                x.streamed <= x.tail
                    && x.tail <= x.host
                    && x.streamed <= half
                    && (x.streamed == 0) == (x.streamed_columns == 0)
            })
            && off.xlayers.is_empty()
            && off.xend.is_none();
        let units = STREAM_P.div_ceil(place::PROMPT_ROWS);
        let lo = left_out_of(&a, units);
        let count_ok = lo.ok && off.excluded == 0;
        println!(
            "xstream_split_streams: a {STREAM_P}-id prompt: {} xstream records, {streamed} experts streamed (half \
             {half}), each within its tail, its host experts and the half {records_ok}; the records' {columns} \
             streamed columns are each walk's units times its first unit's left-out slots ({} in all, {} records \
             off), its later units left out {}, layers that streamed nothing {}, the serve {} in all over {} \
             layer-batch rows; the unstreamed run left out {} {count_ok}: {}",
            a.xlayers.len(),
            lo.first,
            lo.bad.len(),
            lo.later,
            lo.stray,
            a.excluded,
            a.left.len(),
            off.excluded,
            verdict(records_ok && count_ok)
        );
        bad_records("xstream_split_streams", &lo);
        pass &= records_ok && count_ok;

        tier::sc("xstream_split_repeats")?;
        let b = run_once(&mut m, &ids, XMode::Split)?;
        let same_bits = a.logits.len() == b.logits.len()
            && a.logits
                .iter()
                .zip(&b.logits)
                .all(|(x, y)| x.to_bits() == y.to_bits());
        let repro_ok = key(&a) == key(&b) && same_bits && a.tokens == b.tokens;
        println!(
            "xstream_split_repeats: the two split runs' records, prompt logits bits and ids the same {}",
            verdict(repro_ok)
        );
        pass &= repro_ok;

        tier::sc("xstream_split_near_off")?;
        let (dlogit, parted, ids_ok) = agree(&a, &off);
        println!(
            "xstream_split_near_off: the prompt's logits max|diff| against off {dlogit} (printed, not judged), {} \
             greedy ids equal or parted at a near tie{} {}",
            a.tokens.len(),
            parted
                .map(|i| format!(
                    " (first parted at {i}, split's margin {:?})",
                    a.margins.get(i)
                ))
                .unwrap_or_default(),
            verdict(ids_ok)
        );
        pass &= ids_ok;

        tier::sc("xstream_admit_opens_call")?;
        let admit = run_once(&mut m, &ids, XMode::Admit)?;
        let short: Vec<u32> = ids[..STREAM_FLOOR as usize - 1].to_vec();
        let none = run_once(&mut m, &short, XMode::Admit)?;
        let (adm_dlogit, adm_parted, adm_ids_ok) = agree(&admit, &off);
        let opens = admit.end.is_some() && admit.xlayers.is_empty() && admit.excluded == 0;
        let short_ok = none.picks.is_empty() && none.end.is_none();
        println!(
            "xstream_admit_opens_call: a {STREAM_P}-id prompt: {} picks, an end {}, {} experts admitted, no expert \
             streamed {opens}; its logits max|diff| against off {adm_dlogit} (printed), ids equal or parted at a near \
             tie{} {adm_ids_ok}; a {}-id prompt (one fewer than the floor {STREAM_FLOOR}): {} picks, an end {} {}",
            admit.picks.len(),
            admit.end.is_some(),
            admit.end.map_or(0, |r| r.admitted),
            adm_parted
                .map(|i| format!(
                    " (first parted at {i}, admit's margin {:?})",
                    admit.margins.get(i)
                ))
                .unwrap_or_default(),
            short.len(),
            none.picks.len(),
            none.end.is_some(),
            verdict(short_ok)
        );
        pass &= opens && adm_ids_ok && short_ok;
        Ok(pass)
    }

    /// Arm `bp` (module doc): `xstream_split_beside_tier`.
    fn arm_bp(tier_now: Tier) -> Result<bool, GateError> {
        const CLAUSE: &str = "xstream_split_beside_tier";
        let split = open_split(Arch::MistralLarge4, RECIPE)?;
        let inputs = PlanInputs::describe(&split)?.with_stream()?;
        let layers = inputs.hp.n_layer;
        let ctx = u64::try_from(STREAM_P + STEPS + CTX_SPARE)?;
        let bp = Place::Bp
            .on_host()?
            .machine(None, Some(place::tier_batch(&inputs)?))?;
        let mut two = bp(layers);
        // The stage card alone, the budget's machine.
        let flat = Machine {
            tiers: Vec::new(),
            ..two.clone()
        };
        let levers = stage_budget(&inputs, &flat, ctx, tier_now)?;
        if tier_now == Tier::Fixture {
            // The fixture's experts fit the tier card whole with room to spare, and a split needs
            // host experts: the tier card keeps free its headroom and half the expert bytes it
            // would plan, so it holds half of them and the rest goes to the host.
            let full = inputs.plan(&two, ctx, &levers)?;
            let (held, headroom) = full.cards.get(1).map_or((0, 0), |c| {
                (c.expert_bytes, u64::try_from(c.headroom_bytes).unwrap_or(0))
            });
            let tier_card = two
                .tiers
                .first_mut()
                .ok_or("arm bp: the bp machine has no tier card")?;
            tier_card.margin_bytes += headroom + held / 2;
        }
        let plan = inputs.plan(&two, ctx, &levers)?;
        if plan.host.experts == 0 {
            return Err(format!(
                "{CLAUSE}: the arm's plan leaves no expert on the host, so a split has nothing to stream"
            )
            .into());
        }
        let tier_experts = plan.cards.get(1).map_or(0, |c| c.experts);
        let tier_layers: Vec<usize> = plan.tier_n_l.first().map_or_else(Vec::new, |per| {
            per.iter()
                .enumerate()
                .filter(|&(_, &n)| n > 0)
                .map(|(l, _)| l)
                .collect()
        });
        println!(
            "arm bp plan: stage card experts {} tier card experts {tier_experts} on {} layers, host experts {}, \
             prompt reserve {} B with the stream's ring (plan values)",
            plan.cards[0].experts,
            tier_layers.len(),
            plan.host.experts,
            inputs.prompt_reserve,
        );
        let mut m = open("arm bp", &plan, &inputs)?;
        let ids = walk(STREAM_P, inputs.hp.n_vocab);

        tier::sc(CLAUSE)?;
        let started = {
            let (gpu, _, body) = m.body_parts(NAME)?;
            match body.set_xstream(gpu, XMode::Split) {
                Ok(()) => {
                    body.set_xstream_costs(SPLIT_COSTS)?;
                    let word = body.xstream_word();
                    let half = half_of(&word);
                    let ok = body.xstream_mode() == XMode::Split && half > 0;
                    println!(
                        "{CLAUSE}: beside an expert tier of {tier_experts} experts xstream={word}: a ring of {half} \
                         slots a half {}",
                        verdict(ok)
                    );
                    ok
                }
                Err(e) => {
                    println!("{CLAUSE}: split refused beside the expert tier: {e} FAIL");
                    false
                }
            }
        };
        if tier_experts == 0 || tier_layers.is_empty() {
            println!(
                "FAIL precondition {CLAUSE}: the plan holds no expert on the tier: the run proves nothing"
            );
        }
        if !started || tier_experts == 0 || tier_layers.is_empty() {
            return Ok(false);
        }
        set_mode(&mut m, XMode::Off)?;

        let off = run_once(&mut m, &ids, XMode::Off)?;
        let a = run_once(&mut m, &ids, XMode::Split)?;
        let b = run_once(&mut m, &ids, XMode::Split)?;

        let on_tier: Vec<usize> = a
            .xlayers
            .iter()
            .filter(|(_, x)| x.streamed > 0 && tier_layers.contains(&x.layer))
            .map(|(_, x)| x.layer)
            .collect();
        if on_tier.is_empty() {
            println!(
                "FAIL precondition {CLAUSE}: the {STREAM_P}-id prompt streamed no expert on a layer the tier holds \
                 experts of (a plan that leaves those layers no host expert, or a ring too small): the run proves \
                 nothing"
            );
        }
        let streamed: usize = a.xlayers.iter().map(|(_, x)| x.streamed).sum();
        let columns: u64 = a.xlayers.iter().map(|(_, x)| x.streamed_columns).sum();
        let units = STREAM_P.div_ceil(place::PROMPT_ROWS);
        let lo = left_out_of(&a, units);
        let records_ok = a.xend.is_some_and(|e| e.streamed == streamed)
            && lo.ok
            && off.excluded == 0
            && off.joined.is_empty();
        let (want, got) = (joins_wanted(&a, units), joins_got(&a));
        let joins_ok = !want.is_empty() && got == want;
        println!(
            "{CLAUSE}: {streamed} experts streamed, on the tier's layers {on_tier:?}; the end counts the layers' sum and \
             the records' {columns} streamed columns are each walk's units times its first unit's left-out slots ({} \
             in all, {} records off), its later units left out {}, layers that streamed nothing {}, the serve {} in \
             all over {} layer-batch rows (the unstreamed run {}, {} joins) {records_ok}; the backs joined {} layer \
             units against the records' {} {joins_ok}",
            lo.first,
            lo.bad.len(),
            lo.later,
            lo.stray,
            a.excluded,
            a.left.len(),
            off.excluded,
            off.joined.len(),
            got.len(),
            want.len(),
        );
        bad_records(CLAUSE, &lo);
        if !joins_ok {
            println!("{CLAUSE}: the records ask for joins {want:?}, the backs joined {got:?}");
        }
        let same_bits = a.logits.len() == b.logits.len()
            && a.logits
                .iter()
                .zip(&b.logits)
                .all(|(x, y)| x.to_bits() == y.to_bits());
        let repro_ok =
            key(&a) == key(&b) && joins_got(&b) == got && same_bits && a.tokens == b.tokens;
        let (dlogit, parted, ids_ok) = agree(&a, &off);
        println!(
            "{CLAUSE}: the two split runs' records, joins, prompt logits bits and ids the same {repro_ok}; the prompt's \
             logits max|diff| against off {dlogit} (printed, not judged), {} greedy ids equal or parted at a near tie{} \
             {ids_ok}: {}",
            a.tokens.len(),
            parted
                .map(|i| format!(
                    " (first parted at {i}, split's margin {:?})",
                    a.margins.get(i)
                ))
                .unwrap_or_default(),
            verdict(!on_tier.is_empty() && records_ok && joins_ok && repro_ok && ids_ok)
        );
        let admit = run_once(&mut m, &ids, XMode::Admit)?;
        let (dlogit, parted, admit_ids) = agree(&admit, &off);
        let opens = admit.end.is_some() && admit.xlayers.is_empty() && admit.joined.is_empty();
        println!(
            "{CLAUSE}: admit beside the tier: a {STREAM_P}-id prompt opened a call with no stream and no join {opens}, \
             logits max|diff| against off {dlogit} (printed), ids equal or parted at a near tie{} {}",
            parted
                .map(|i| format!(
                    " (first parted at {i}, admit's margin {:?})",
                    admit.margins.get(i)
                ))
                .unwrap_or_default(),
            verdict(opens && admit_ids)
        );
        Ok(!on_tier.is_empty()
            && records_ok
            && joins_ok
            && repro_ok
            && ids_ok
            && opens
            && admit_ids)
    }
}
