//! GPU gate for Mistral Large 4 on plan (b′) (`--place bp`): the stage on the largest card in view and the expert tier on
//! the next, against one card holding the same expert sets.
//!
//! Two loads of one plan (b′), one after the other, both through the body's own entry (`Body::open_placed`, which hangs
//! the plan's tier card under the host tier):
//!
//! 1. the reference: the plan with every tier segment moved onto the stage card (each routed stack's stage and tier
//!    lists joined, the host's untouched, [`union_plan`]) and the tier card dropped from the plan's machine: the stage card
//!    holds the union and no tier code runs;
//! 2. the two cards: the stage on the largest card, each routed layer's next ids on the next card, the host the rest.
//!
//! Memory: both plans are made under one card budget the gate sets (`PlanLevers::card_budget_bytes`; no environment
//! lever moves it). It caps every card, the tier's too. In the real tier the budget is what the stage card holds without
//! experts plus [`TWO_STAGE_EXPERTS`], so the stage keeps that many routed expert bytes, the tier up to the same budget
//! less its own set-aside and batch reserve, and the reference's union is at most twice the budget [derived from the
//! placement's rule]. The gate asks the view for the facts it needs before any load, two cards in view and the stage
//! card's usable bytes for the union (`tier::union_lack_of`, the plan's own values), and leaves every clause by name
//! (`tier::defer_for_room`, `no-card-room`), in either tier, where the view lacks them. In
//! the fixture tier the stage keeps half of the routed expert bytes the stage card alone would hold, so the tier takes the
//! other half and both cards run. Each layer's stage card keeps its id prefix and the tier the next ids, so the
//! preconditions below are met by the walk's routing, which the run checks.
//!
//! The walk is the fixed one of `e2e::walk` through the file's vocabulary: the first [`PROMPT`] ids one decode step
//! each, then [`STEPS`] greedy steps, every step's token and logits row read. The body has no prompt batch, so the
//! walk is the only leg.
//!
//! Clauses (all self-consistency clauses: each compares the engine's two loads to each other and reads no oracle, so
//! both tiers run them):
//! - `two_card_load`: the plan puts experts on the tier, the load's tier opened on the second card beside the stage card,
//!   and the experts the load holds on the tier, by layer, are the plan's;
//! - `two_card_step_bits`: the walk's tokens and logits rows on the two cards, run eager and as the captured step
//!   graph's replays, are the reference's replayed walk bit for bit, every logit of the reference finite. The tier's
//!   slot rows are the card leg's over the tier's stacks, each the row the stage card would write for that slot, and
//!   the card sum over both cards' slots in slot order is the one-card sum over their union
//!   (`bloomery_gpu_deepseek41::tier_join`); every other launch of the step reads the same bits on both loads, so the
//!   band is bits and any band wider hides a tier row apart from the reference's;
//! - `two_card_tier_sent`: the replayed walk sent every layer the tier holds experts of at least one routed slot, or the
//!   run proves nothing there and fails by name. A miss is a walk too short for the routing (raise [`PROMPT`] or
//!   [`STEPS`]), not a defect of the tier.
//!
//! The taps are not armed: a tier layer's card sum is made inside the join, in no buffer a tap could copy
//! (`Body::arm_taps` refuses such a load by name), so the clauses read the step's tokens and logits.
//!
//! The node arm (`--node`, `just gate-gpu-mistrallarge4-node`; no argument runs the two-card arm above) runs the clauses
//! with the plan's one expert tier on a node (`BLOOMERY_TIER_NODES`, served by a loopback `tier_server` built with the
//! `mistrallarge4` feature, [`bloomery_gpu_mistrallarge4::NODE_FAMILY`]) and needs one card in view: on one card the tier
//! node's card is the stage's own, so the plan's card budget splits the device between the two (`node_machine`). The lever
//! is read at every load, so the arm is two runs of this binary, each a process of its own with the same card budget:
//! `--union` (no lever) loads the reference, the plan's tier segments joined onto the stage card alone, and prints its plan
//! and its walk's tokens and logits digests; `--node0` (the lever set) loads the plan with its tier on the node and prints
//! the same for the eager walk and the replayed one. This process compares the two (the plans first, then the digests,
//! FNV-1a over every logits row's bits) as the clauses:
//! - `node_load` (the node run's): the plan puts experts on the tier, the load's tier opened on the node's card and the
//!   experts it holds, by layer, are the plan's;
//! - `node_step_bits` (this process's): the node tier's eager and replayed walks are the union's bit for bit, over the
//!   same plan;
//! - `node_tier_sent` (the node run's): the replayed walk sent every layer the tier holds experts of at least one routed
//!   slot.

#[cfg(feature = "mistrallarge4")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_mistrallarge4_twocard", gate::run())
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
    reason = "the two-card gates share the module; each takes the helpers it reads"
)]
#[path = "shared/tier_twocard.rs"]
mod tier_twocard;

#[cfg(feature = "mistrallarge4")]
#[allow(
    dead_code,
    reason = "the gates that start a node share the module; each takes the helpers it reads"
)]
#[path = "shared/node_proc.rs"]
mod node_proc;

#[cfg(feature = "mistrallarge4")]
mod gate {
    use std::net::SocketAddr;
    use std::process::Command;
    use std::sync::mpsc::RecvTimeoutError;
    use std::time::{Duration, Instant};

    use bloomery_gpu::GpuModel;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::tier::{self, Tier};
    use bloomery_gpu_gates::{GateError, checks_failed, open_split, ref_model_path};
    use bloomery_gpu_mistrallarge4::body::Body;
    use model::arch::Arch;
    use model::arch::mistrallarge4::place::{self, PlanInputs};
    use model::placement::workstation::{self, CardSpec, TierBatchBytes};
    use model::placement::{Device, ExpertList, Machine, Plan, PlanLevers, Role, union_plan};

    use crate::e2e::walk;
    use crate::node_proc::{Proc, tier_server, tier_server_command};
    use crate::tier_twocard::{
        Leg, Sent, same, sent_between, sent_every, tier_counts, tier_on_card, tier_on_node,
    };

    type Model = GpuModel<Body>;

    const NAME: &str = "gate_mistrallarge4_twocard";

    /// The recipe the model file is opened for.
    const RECIPE: &str = "gate-gpu-mistrallarge4-twocard";

    /// The host tier's configuration both loads open with: no lever is read, so one left in the runner's environment
    /// cannot move a clause (the e2e gate's).
    const HOST_CFG: bloomery_levers::HostCfg = bloomery_levers::HostCfg {
        populate: false,
        lock: false,
        card_dontneed: false,
        r8: false,
        route_trace: false,
    };

    /// Prompt positions of the walk, one decode step each: past the fixture's index list width, so every step past its
    /// sixteenth position selects.
    const PROMPT: usize = 32;

    /// Greedy steps after the walk's prompt positions.
    const STEPS: usize = 48;

    /// Rows of the caches beyond the walk.
    const CTX_SPARE: usize = 8;

    /// Cache rows of both loads: the walk and a few more.
    const CTX: usize = PROMPT + STEPS + CTX_SPARE;

    /// The routed expert bytes on the stage card in the real tier: the stage card's budget is what it holds without
    /// experts plus these, and the tier's experts are capped by the same budget, so the reference's union is at most
    /// twice the budget [derived from the placement's rule].
    const TWO_STAGE_EXPERTS: u64 = 2 << 30;

    /// The clauses of the arm: each is declared where it runs and left by name where the cards in view cannot run it.
    const TWO_CARD_LOAD: &str = "two_card_load";
    const STEP_BITS: &str = "two_card_step_bits";
    const TIER_SENT: &str = "two_card_tier_sent";
    const CLAUSES: [&str; 3] = [TWO_CARD_LOAD, STEP_BITS, TIER_SENT];

    /// The node arm's clauses, decided by the run that can: the node run's, and the comparison of the two runs'.
    const NODE_LOAD: &str = "node_load";
    const NODE_STEP_BITS: &str = "node_step_bits";
    const NODE_TIER_SENT: &str = "node_tier_sent";
    const NODE_CLAUSES: [&str; 3] = [NODE_LOAD, NODE_STEP_BITS, NODE_TIER_SENT];

    /// What a run of the node arm tells its parent, a keyword and its text a line: the layer splits it planned, the
    /// reason it left its clauses, a clause it decided, and a leg's tokens and row digests.
    const PLAN: &str = "NODEARM plan";
    const DEFERRED: &str = "NODEARM deferred";
    const RESULT: &str = "NODEARM result";
    const LEG: &str = "NODEARM leg";

    /// The node arm (`--node`) and its two runs of this binary: the reference (no node lever), and the load with its tier
    /// on the node `BLOOMERY_TIER_NODES` names.
    const NODE: &str = "--node";
    const UNION: &str = "--union";
    const NODE0: &str = "--node0";

    /// The bound on each of the node arm's runs: a load and the walks, or the real file's.
    const RUN_BOUND: Duration = Duration::from_secs(1800);

    /// How long the parent waits for a line a finished run printed.
    const LINE_WITHIN: Duration = Duration::from_secs(10);

    /// What the walk read: its tokens and logits rows, and what it sent the tier.
    struct Run {
        leg: Leg,
        /// What the walk alone sent the tier; `None` on a load with none.
        sent: Option<Sent>,
    }

    /// The tier's per-layer hits and batch counters of `m`'s load since it began; `None` without a tier.
    fn counts(m: &Model) -> Result<Option<Sent>, GateError> {
        Ok(tier_counts(m.body(NAME)?.hybrid()))
    }

    /// The model file, a Mistral Large 4 file (`open_split` refuses another architecture by name).
    fn split() -> Result<gguf::Split, GateError> {
        open_split(Arch::MistralLarge4, RECIPE)
    }

    /// The placed load of `plan`, which `inputs` made, on its card 0.
    fn open(plan: &Plan<'_>, inputs: &PlanInputs) -> Result<Model, GateError> {
        Ok(Body::open_placed(split()?, plan, inputs, 0, HOST_CFG)?)
    }

    /// Refused unless `m` runs on the largest card in view, the card the plan's stage is.
    fn on_stage_card(what: &str, m: &Model) -> Result<(), GateError> {
        tier::opened_on_rank(what, m.gpu(), 0)
    }

    /// Per routed layer, the experts its stage card, its tier card and the host hold (the first routed stack's segments:
    /// every stack of a layer is split alike).
    fn layer_splits(plan: &Plan<'_>) -> Vec<(usize, u64, u64, u64)> {
        let tier = Device::Card(plan.machine.cards.len());
        let mut out: Vec<(usize, u64, u64, u64)> = Vec::new();
        for row in &plan.rows {
            let t = &plan.model.tensors[row.tensor];
            let Some(l) = t.layer.filter(|_| t.role == Role::RoutedExperts) else {
                continue;
            };
            if out.iter().any(|o| o.0 == l) {
                continue;
            }
            let on = |d: Device| -> u64 {
                row.segments
                    .iter()
                    .filter(|s| s.device == d)
                    .filter_map(|s| s.experts.as_ref())
                    .map(ExpertList::len)
                    .sum()
            };
            out.push((l, on(Device::Card(0)), on(tier), on(Device::Host)));
        }
        out
    }

    /// A card budget that keeps the stage card's bytes without experts (its dense, rounding, cache and stage rows,
    /// set-aside and margin) plus the share of the routed expert bytes the stage would keep unbudgeted on `machine`,
    /// the stage card alone: half of them in the fixture tier, [`TWO_STAGE_EXPERTS`] at most in the real tier. The
    /// unbudgeted plan is made on the stage card alone because a plan of both cards refuses a stage that holds every
    /// expert (the fixture's), the tier then holding none.
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
            Tier::Real => c.expert_bytes.min(TWO_STAGE_EXPERTS),
        };
        Ok(PlanLevers {
            card_budget_bytes: Some(no_experts + experts).into(),
            ..PlanLevers::default()
        })
    }

    /// One decode step of `id`, its token and logits into `leg`.
    fn step(m: &mut Model, id: u32, leg: &mut Leg) -> Result<u32, GateError> {
        let tok = m.step(&[id])?;
        leg.tokens.push(tok);
        leg.logits.push(m.logits()?);
        Ok(tok)
    }

    /// The walk (module header) from a reset in the load's current mode, and what it sent the tier.
    fn walk_leg(m: &mut Model, prompt: &[u32]) -> Result<Run, GateError> {
        let mut leg = Leg::default();
        m.reset()?;
        let before = counts(m)?;
        let mut tok = 0;
        for &id in &prompt[..PROMPT] {
            tok = step(m, id, &mut leg)?;
        }
        for _ in 0..STEPS {
            tok = step(m, tok, &mut leg)?;
        }
        let sent = sent_between(before, counts(m)?);
        Ok(Run { leg, sent })
    }

    /// `m` in graph mode with its step captured.
    fn capture(m: &mut Model) -> Result<(), GateError> {
        m.set_mode(StepMode::Graph);
        m.capture_step()?;
        Ok(())
    }

    fn verdict(pass: bool) -> &'static str {
        if pass { "ok" } else { "FAIL" }
    }

    pub fn run() -> Result<(), GateError> {
        let tier_now = Tier::from_env()?;
        let inputs = PlanInputs::describe(&split()?)?;
        let args: Vec<String> = std::env::args().skip(1).collect();
        let (pass, proves) = match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
            [] => (
                bits(&inputs, tier_now)?,
                "plan (b′) loads the expert tier on the second card beside the stage card with the plan's experts on it, \
                 the walk eager and replayed reads the one card's ids and logits bit for bit, and every tier layer was \
                 sent a routed slot",
            ),
            [NODE] => (
                node_arm()?,
                "plan (b′) opens the expert tier on a node with the plan's experts on it, the walk eager and replayed \
                 reads the no-node load's ids and logits bit for bit, and every tier layer was sent a routed slot",
            ),
            [UNION] => return union_run(&inputs, tier_now),
            [NODE0] => return node_run(&inputs, tier_now),
            _ => {
                return Err(format!(
                    "unknown arguments {args:?}: usage: {NAME} [{NODE} | {UNION} | {NODE0}]"
                )
                .into());
            }
        };
        println!("{NAME}: {}", tier::tally_line());
        if tier::ran_any() && pass {
            println!("PASSED: {NAME} {proves}");
        }
        if !pass {
            return Err(checks_failed());
        }
        Ok(())
    }

    /// The clauses (module header): the plan, the union reference, then the two cards.
    fn bits(inputs: &PlanInputs, tier_now: Tier) -> Result<bool, GateError> {
        if let Some(why) = tier::two_cards_lack()? {
            tier::defer_for_room(&CLAUSES, &why)?;
            return Ok(true);
        }
        let layers = inputs.hp.n_layer;
        let bp = Place::Bp
            .on_host()?
            .machine(None, Some(place::tier_batch(inputs)?))?;
        let two = bp(layers);
        let ctx = u64::try_from(CTX)?;
        // The stage card alone, the reference's machine.
        let flat = Machine {
            tiers: Vec::new(),
            ..two.clone()
        };
        let levers = stage_budget(inputs, &flat, ctx, tier_now)?;
        let plan = inputs.plan(&two, ctx, &levers)?;
        let splits = layer_splits(&plan);
        let tier_layers: Vec<usize> = splits.iter().filter(|o| o.2 > 0).map(|o| o.0).collect();
        print_plan(&plan, &levers, &splits);
        if let Some(why) = tier::union_lack_of(&plan) {
            tier::defer_for_room(&CLAUSES, &why)?;
            return Ok(true);
        }
        let prompt = walk(PROMPT, inputs.hp.n_vocab);

        // The reference: the union on the stage card alone, its walk replayed, then released.
        let uplan = union_plan(&plan, &flat)?;
        println!(
            "reference: the stage card holds {} experts, the union of the stage's and the tier's",
            uplan.n_l.iter().sum::<u64>()
        );
        let want = {
            let t0 = Instant::now();
            let mut r = open(&uplan, inputs)?;
            on_stage_card("reference load", &r)?;
            if !r.body(NAME)?.hybrid().tiers().is_empty() {
                return Err("the reference load holds a tier card".into());
            }
            println!(
                "reference load in {:.1} s: {} ({} B) holds the union, no tier",
                t0.elapsed().as_secs_f64(),
                r.gpu().device_name()?,
                r.resident_bytes(),
            );
            capture(&mut r)?;
            walk_leg(&mut r, &prompt)?
        };

        let t0 = Instant::now();
        let mut m = open(&plan, inputs)?;
        on_stage_card("two-card load", &m)?;
        let mut pass = true;

        tier::sc(TWO_CARD_LOAD)?;
        tier_on_card(
            "two-card load",
            t0,
            (&m.gpu().device_name()?, m.resident_bytes()),
            m.body(NAME)?.hybrid(),
        )?;
        let loaded: Vec<u64> = m
            .body(NAME)?
            .tier_experts()
            .into_iter()
            .map(u64::try_from)
            .collect::<Result<_, _>>()?;
        let planned: Vec<u64> = (0..layers)
            .map(|l| splits.iter().find(|o| o.0 == l).map_or(0, |o| o.2))
            .collect();
        let load_ok = !tier_layers.is_empty() && loaded == planned;
        println!(
            "two_card_load: the tier holds experts on {} layers; the load's tier experts by layer {} the plan's {}",
            tier_layers.len(),
            if loaded == planned { "are" } else { "are not" },
            verdict(load_ok)
        );
        if tier_layers.is_empty() {
            println!(
                "FAIL precondition two_card_load: the plan holds no expert on the tier: the run proves nothing"
            );
        }
        if loaded != planned {
            println!("  loaded {loaded:?}\n  planned {planned:?}");
        }
        pass &= load_ok;

        tier::sc(STEP_BITS)?;
        m.set_mode(StepMode::Eager);
        let eager = walk_leg(&mut m, &prompt)?;
        capture(&mut m)?;
        let replay = walk_leg(&mut m, &prompt)?;
        let finite = want.leg.logits.iter().flatten().all(|v| v.is_finite());
        if !finite {
            println!("FAIL two_card_step_bits: a logit of the reference walk is not finite");
        }
        let eager_ok = same("two_card_step_bits: eager walk", &want.leg, &eager.leg);
        let replay_ok = same("two_card_step_bits: replayed walk", &want.leg, &replay.leg);
        pass &= finite && eager_ok && replay_ok;

        tier::sc(TIER_SENT)?;
        pass &= sent_every(
            "two_card_tier_sent replayed walk",
            replay.sent.as_ref(),
            &tier_layers,
            false,
        );
        Ok(pass)
    }

    /// The plan's one line and its per-layer splits.
    fn print_plan(plan: &Plan<'_>, levers: &PlanLevers, splits: &[(usize, u64, u64, u64)]) {
        let tier_bytes = plan.cards.get(1).map_or(0, |c| c.expert_bytes);
        let stage = &plan.cards[0];
        let card = &plan.machine.cards[0];
        println!(
            "plan (b′) under a card budget of {} at ctx {CTX}: stage {} experts ({} B), tier {} experts ({tier_bytes} B) on \
             {} layers, host {} experts ({} B); the reference's union {} B of the stage card {}'s {} usable (plan values)",
            model::placement::card_budget::in_bytes(&levers.card_budget_bytes),
            stage.experts,
            stage.expert_bytes,
            plan.cards.get(1).map_or(0, |c| c.experts),
            splits.iter().filter(|o| o.2 > 0).count(),
            plan.host.experts,
            plan.host.expert_bytes,
            tier::union_bytes(plan),
            card.name,
            card.usable_bytes,
        );
        for (l, on_stage, on_tier, on_host) in splits {
            println!("  layer {l}: stage {on_stage}, tier {on_tier}, host {on_host}");
        }
    }

    /// Plan (b′)'s machine over `stage` and `tier`, which may be one device: the stage card for `layers` layers, the
    /// tier card beside it holding the prompt batch's card bytes, and the host's rows of it. `Place::machine` refuses
    /// a tier on the stage's own device (`workstation::check_holding`); the planner's card budget, which the gate sets,
    /// is what splits one device between the two.
    fn node_machine(
        layers: usize,
        stage: CardSpec,
        tier: CardSpec,
        batch: TierBatchBytes,
    ) -> Machine {
        let mut m = workstation::plan_on(stage, layers);
        m.tiers.push(workstation::tier_card(tier, batch));
        m.host_mut()
            .reserves
            .push((workstation::TIER_BATCH_HOST_RESERVE.to_string(), batch.host));
        m
    }

    /// The stage card of [`node_machine`] alone, the reference's machine.
    fn flat_machine(layers: usize, stage: CardSpec, batch: TierBatchBytes) -> Machine {
        Machine {
            tiers: Vec::new(),
            ..node_machine(layers, stage, stage, batch)
        }
    }

    /// The layers' stage, tier and host expert counts as the one token both runs print and the parent compares.
    fn splits_key(splits: &[(usize, u64, u64, u64)]) -> String {
        splits
            .iter()
            .map(|(l, s, t, h)| format!("{l}:{s}:{t}:{h}"))
            .collect::<Vec<_>>()
            .join(";")
    }

    /// FNV-1a over the bits of `row`.
    fn fnv(row: &[f32]) -> u64 {
        row.iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .fold(0xcbf2_9ce4_8422_2325, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
            })
    }

    /// What a run tells its parent about a leg: its tokens and the digest of every logits row.
    fn print_leg(name: &str, leg: &Leg) {
        let tokens: Vec<String> = leg.tokens.iter().map(u32::to_string).collect();
        let rows: Vec<String> = leg
            .logits
            .iter()
            .map(|r| format!("{:016x}", fnv(r)))
            .collect();
        println!(
            "{LEG} {name} tokens {} rows {}",
            tokens.join(","),
            rows.join(",")
        );
    }

    /// The reference run of the node arm (module header): the plan the node run makes, with its tier segments
    /// joined onto the stage card, loaded with no node and walked as the captured step graph.
    fn union_run(inputs: &PlanInputs, tier_now: Tier) -> Result<(), GateError> {
        let layers = inputs.hp.n_layer;
        let ctx = u64::try_from(CTX)?;
        let batch = place::tier_batch(inputs)?;
        let stage = *Place::A
            .on_host()?
            .card_specs()?
            .first()
            .ok_or("no stage card in view")?;
        let flat = flat_machine(layers, stage, batch);
        let mut two = node_machine(layers, stage, stage, batch);
        two.nodes.push(workstation::node_of(&stage));
        let levers = stage_budget(inputs, &flat, ctx, tier_now)?;
        let plan = inputs.plan(&two, ctx, &levers)?;
        let splits = layer_splits(&plan);
        print_plan(&plan, &levers, &splits);
        println!("{PLAN} {}", splits_key(&splits));
        if let Some(why) = tier::union_lack_of(&plan) {
            println!("{DEFERRED} {why}");
            return Ok(());
        }
        let uplan = union_plan(&plan, &flat)?;
        let t0 = Instant::now();
        let mut r = open(&uplan, inputs)?;
        on_stage_card("reference load", &r)?;
        if !r.body(NAME)?.hybrid().tiers().is_empty() {
            return Err("the reference load holds a tier card".into());
        }
        println!(
            "reference load in {:.1} s: {} ({} B) holds the union, {} experts, no tier",
            t0.elapsed().as_secs_f64(),
            r.gpu().device_name()?,
            r.resident_bytes(),
            uplan.n_l.iter().sum::<u64>(),
        );
        capture(&mut r)?;
        let want = walk_leg(&mut r, &walk(PROMPT, inputs.hp.n_vocab))?;
        if !want.leg.logits.iter().flatten().all(|v| v.is_finite()) {
            return Err("a logit of the reference walk is not finite".into());
        }
        print_leg("union", &want.leg);
        Ok(())
    }

    /// The node run of the node arm (module header): the plan with its tier on the node `BLOOMERY_TIER_NODES` names,
    /// the clauses `node_load` and `node_tier_sent` decided here, the eager and replayed walks handed to the parent.
    fn node_run(inputs: &PlanInputs, tier_now: Tier) -> Result<(), GateError> {
        let layers = inputs.hp.n_layer;
        let ctx = u64::try_from(CTX)?;
        let batch = place::tier_batch(inputs)?;
        let place = Place::Bp.on_host()?;
        let stage = *place.card_specs()?.first().ok_or("no stage card in view")?;
        let two = place.machine_with(|first, tiers| match tiers {
            &[tier] => Ok(node_machine(layers, first, tier, batch)),
            _ => Err(format!(
                "--place {}: the node arm takes a stage card and one tier card",
                place.name()
            )
            .into()),
        })?;
        let flat = flat_machine(layers, stage, batch);
        let levers = stage_budget(inputs, &flat, ctx, tier_now)?;
        let plan = inputs.plan(&two, ctx, &levers)?;
        let splits = layer_splits(&plan);
        print_plan(&plan, &levers, &splits);
        println!("{PLAN} {}", splits_key(&splits));
        if let Some(why) = tier::union_lack_of(&plan) {
            println!("{DEFERRED} {why}");
            return Ok(());
        }
        let tier_layers: Vec<usize> = splits.iter().filter(|o| o.2 > 0).map(|o| o.0).collect();
        let prompt = walk(PROMPT, inputs.hp.n_vocab);

        let t0 = Instant::now();
        let mut m = open(&plan, inputs)?;
        on_stage_card("node load", &m)?;
        tier_on_node(
            "node load",
            t0,
            (&m.gpu().device_name()?, m.resident_bytes()),
            m.body(NAME)?.hybrid(),
        )?;
        let loaded: Vec<u64> = m
            .body(NAME)?
            .tier_experts()
            .into_iter()
            .map(u64::try_from)
            .collect::<Result<_, _>>()?;
        let planned: Vec<u64> = (0..layers)
            .map(|l| splits.iter().find(|o| o.0 == l).map_or(0, |o| o.2))
            .collect();
        let load_ok = !tier_layers.is_empty() && loaded == planned;
        println!(
            "{NODE_LOAD}: the tier holds experts on {} layers; the node's experts by layer {} the plan's {}",
            tier_layers.len(),
            if loaded == planned { "are" } else { "are not" },
            verdict(load_ok)
        );
        if tier_layers.is_empty() {
            println!(
                "FAIL precondition {NODE_LOAD}: the plan holds no expert on the tier: the run proves nothing"
            );
        }
        if loaded != planned {
            println!("  loaded {loaded:?}\n  planned {planned:?}");
        }
        println!("{RESULT} {NODE_LOAD} {}", verdict(load_ok));

        m.set_mode(StepMode::Eager);
        let eager = walk_leg(&mut m, &prompt)?;
        print_leg("eager", &eager.leg);
        capture(&mut m)?;
        let replay = walk_leg(&mut m, &prompt)?;
        print_leg("replay", &replay.leg);
        let sent_ok = sent_every(
            "node_tier_sent replayed walk",
            replay.sent.as_ref(),
            &tier_layers,
            false,
        );
        println!("{RESULT} {NODE_TIER_SENT} {}", verdict(sent_ok));
        Ok(())
    }

    /// One run of this binary under `arg`, with `BLOOMERY_TIER_NODES` naming `node` or, with none, unset; its stdout's
    /// lines and its exit, within [`RUN_BOUND`].
    fn child(
        arg: &str,
        node: Option<SocketAddr>,
        tag: &'static str,
    ) -> Result<Vec<String>, GateError> {
        let mut cmd = Command::new(std::env::current_exe()?);
        cmd.arg(arg).env_remove(bloomery_levers::TIER_NODES);
        if let Some(addr) = node {
            cmd.env(bloomery_levers::TIER_NODES, addr.to_string());
        }
        let mut p = Proc::spawn(cmd, tag)?;
        let end = Instant::now() + RUN_BOUND;
        let mut lines = Vec::new();
        loop {
            match p.next_line(end.saturating_duration_since(Instant::now())) {
                Ok(l) => lines.push(l),
                Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => {
                    return Err(format!("{tag}: not done within {} s", RUN_BOUND.as_secs()).into());
                }
            }
        }
        let status = p
            .wait_within(LINE_WITHIN)?
            .ok_or_else(|| format!("{tag}: closed its output and did not exit"))?;
        if !status.success() {
            return Err(format!("{tag}: {status}; stderr: {}", p.stderr()).into());
        }
        Ok(lines)
    }

    /// The text after `prefix` on the first of `lines` that starts with it.
    fn field<'a>(lines: &'a [String], prefix: &str, tag: &str) -> Result<&'a str, GateError> {
        lines
            .iter()
            .find_map(|l| l.strip_prefix(prefix))
            .map(str::trim)
            .ok_or_else(|| format!("{tag}: printed no {prefix:?} line").into())
    }

    /// The tokens and row digests of leg `name`, which a run printed through [`print_leg`].
    fn leg_of<'a>(
        lines: &'a [String],
        name: &str,
        tag: &str,
    ) -> Result<(&'a str, &'a str), GateError> {
        let text = field(lines, &format!("{LEG} {name} tokens "), tag)?;
        text.split_once(" rows ")
            .ok_or_else(|| format!("{tag}: the {name} leg line has no rows").into())
    }

    /// Whether leg `name` of the node run is the union's bit for bit; prints the first difference.
    fn same_leg(name: &str, union: &[String], node: &[String]) -> Result<bool, GateError> {
        let (ut, ur) = leg_of(union, "union", "union run")?;
        let (nt, nr) = leg_of(node, name, "node run")?;
        let what = format!("{NODE_STEP_BITS}: {name} walk");
        if ut != nt {
            let at = ut.split(',').zip(nt.split(',')).position(|(a, b)| a != b);
            println!("FAIL {what}: tokens apart at {at:?}");
            return Ok(false);
        }
        if ur != nr {
            let at = ur.split(',').zip(nr.split(',')).position(|(a, b)| a != b);
            println!("FAIL {what}: logits row {at:?} apart (digests of every logit's bits)");
            return Ok(false);
        }
        println!(
            "ok {what}: {} tokens, {} logits rows bit for bit the union's",
            ut.split(',').count(),
            ur.split(',').count()
        );
        Ok(true)
    }

    /// The node arm (module header): the union run, then a loopback `tier_server` and the node run, and the clauses.
    fn node_arm() -> Result<bool, GateError> {
        let model = ref_model_path()?;
        let union = child(UNION, None, "union")?;
        if let Ok(why) = field(&union, DEFERRED, "union run") {
            tier::defer_for_room(&NODE_CLAUSES, why)?;
            return Ok(true);
        }
        let (server, addr) = tier_server(tier_server_command(Some(&model))?, "node")?;
        let node = child(NODE0, Some(addr), "node 0")?;
        drop(server);
        if let Ok(why) = field(&node, DEFERRED, "node run") {
            tier::defer_for_room(&NODE_CLAUSES, why)?;
            return Ok(true);
        }
        for c in NODE_CLAUSES {
            tier::sc(c)?;
        }
        let mut pass = true;
        let (up, np) = (
            field(&union, PLAN, "union run")?,
            field(&node, PLAN, "node run")?,
        );
        if up != np {
            println!(
                "FAIL precondition {NODE_STEP_BITS}: the two runs planned other splits (layer:stage:tier:host)\n  \
                 union run {up}\n  node run  {np}"
            );
            pass = false;
        }
        for c in [NODE_LOAD, NODE_TIER_SENT] {
            let v = field(&node, &format!("{RESULT} {c} "), "node run")?;
            println!("{c} (node run): {v}");
            pass &= v == verdict(true);
        }
        for name in ["eager", "replay"] {
            pass &= same_leg(name, &union, &node)?;
        }
        Ok(pass)
    }
}
