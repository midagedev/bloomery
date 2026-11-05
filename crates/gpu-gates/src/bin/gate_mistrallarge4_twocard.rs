//! GPU gate for Mistral Large 4 on plan (b′) (`--place bp`): the stage on the A6000 and the expert tier on the 3090,
//! against one card holding the same expert sets.
//!
//! Two loads of one plan (b′), one after the other, both through the body's own entry (`Body::open_placed`, which hangs
//! the plan's tier card under the host tier):
//!
//! 1. the reference: the plan with every tier segment moved onto the stage card (each routed stack's stage and tier
//!    lists joined, the host's untouched, [`union_plan`]) and the tier card dropped from the plan's machine: the A6000
//!    holds the union and no tier code runs;
//! 2. the two cards: the stage on the A6000, each routed layer's next ids on the 3090, the host the rest.
//!
//! Memory: both plans are made under one card budget the gate sets (`PlanLevers::card_budget_bytes`; no environment
//! lever moves it). It caps every card, the tier's too. In the real tier the budget is what the stage card holds without
//! experts plus [`TWO_STAGE_EXPERTS`], so the stage keeps that many routed expert bytes, the tier up to the same budget
//! less its own set-aside and batch reserve, and the reference's union is at most twice the budget, under the A6000's
//! usable bytes [derived from the placement's rule; the gate prints both and refuses a union past the card by name]. In
//! the fixture tier the stage keeps half of the routed expert bytes the A6000 alone would hold, so the tier takes the
//! other half and both cards run. Each layer's stage card keeps its id prefix and the tier the next ids, so the
//! preconditions below are met by the walk's routing, which the run checks.
//!
//! The walk is the fixed one of `e2e::walk` through the file's vocabulary: the first [`PROMPT`] ids one decode step
//! each, then [`STEPS`] greedy steps, every step's token and logits row read. The body has no prompt batch, so the
//! walk is the only leg.
//!
//! Clauses (all self-consistency clauses: each compares the engine's two loads to each other and reads no oracle, so
//! both tiers run them):
//! - `two_card_load`: the plan puts experts on the tier, the load's tier opened on the 3090 beside the A6000 stage card,
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
//! FAIL-first (box): the one-line mutants under the round's mutant folder name the clause each reddens.

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
mod gate {
    use std::time::Instant;

    use bloomery_gpu::GpuModel;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::tier::{self, Tier};
    use bloomery_gpu_gates::{GateError, checks_failed, open_split};
    use bloomery_gpu_mistrallarge4::body::Body;
    use model::arch::Arch;
    use model::arch::mistrallarge4::place::{self, PlanInputs};
    use model::placement::{
        Device, ExpertList, Machine, Plan, PlanLevers, Role, union_plan, workstation,
    };

    use crate::e2e::walk;
    use crate::tier_twocard::{
        Leg, Sent, same, sent_between, sent_every, tier_counts, tier_on_3090,
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
    /// twice the budget, under the A6000's usable bytes [derived from the placement's rule; the gate prints both and
    /// refuses a union past the card by name].
    const TWO_STAGE_EXPERTS: u64 = 2 << 30;

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

    /// Refused unless `m` runs on the A6000, the card the plan's stage is.
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
        let pass = bits(&inputs, tier_now)?;
        println!("{NAME}: {}", tier::tally_line());
        if !pass {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} plan (b′) loads the expert tier on the 3090 beside the A6000 stage with the plan's experts on \
             it, the walk eager and replayed reads the one card's ids and logits bit for bit, and every tier layer was \
             sent a routed slot"
        );
        Ok(())
    }

    /// The clauses (module header): the plan, the union reference, then the two cards.
    fn bits(inputs: &PlanInputs, tier_now: Tier) -> Result<bool, GateError> {
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
        let tier_bytes = plan.cards.get(1).map_or(0, |c| c.expert_bytes);
        let stage = &plan.cards[0];
        let stage_bytes = stage.dense_bytes + stage.expert_bytes + stage.kv_bytes;
        let usable = workstation::A6000.usable_bytes();
        println!(
            "plan (b′) under a card budget of {} at ctx {CTX}: stage {} experts ({} B), tier {} experts ({tier_bytes} B) on \
             {} layers, host {} experts ({} B); the reference's union {} B of the A6000's {usable} (plan values)",
            model::placement::card_budget::in_bytes(&levers.card_budget_bytes),
            stage.experts,
            stage.expert_bytes,
            plan.cards.get(1).map_or(0, |c| c.experts),
            tier_layers.len(),
            plan.host.experts,
            plan.host.expert_bytes,
            stage_bytes + tier_bytes,
        );
        for (l, on_stage, on_tier, on_host) in &splits {
            println!("  layer {l}: stage {on_stage}, tier {on_tier}, host {on_host}");
        }
        if stage_bytes + tier_bytes > usable {
            return Err(format!(
                "the reference's union of {} B passes the A6000's {usable} B: lower the stage's share",
                stage_bytes + tier_bytes
            )
            .into());
        }
        let prompt = walk(PROMPT, inputs.hp.n_vocab);

        // The reference: the union on the A6000 alone, its walk replayed, then released.
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

        tier::sc("two_card_load")?;
        tier_on_3090(
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

        tier::sc("two_card_step_bits")?;
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

        tier::sc("two_card_tier_sent")?;
        pass &= sent_every(
            "two_card_tier_sent replayed walk",
            replay.sent.as_ref(),
            &tier_layers,
            false,
        );
        Ok(pass)
    }
}
