//! GPU gate for Solar-Open2 on plan (b′) (`--place bp`): the stage on the A6000 and the expert tier on the 3090, against
//! one card holding the same expert sets.
//!
//! Two loads of one plan (b′), one after the other:
//!
//! 1. the reference: the plan with every tier segment moved onto the stage card (each routed stack's stage and tier lists
//!    joined, the host's untouched) and the tier card dropped from the plan's machine, opened by `Body::open_placed`: the
//!    A6000 holds the union, and no tier code runs;
//! 2. the two cards, through the binaries' own open (`app::Loaded::open`, whose load hangs the plan's tier card under the
//!    host tier).
//!
//! Memory: both plans are made under the card budget [`BUDGET`] (`PlanLevers::card_budget_bytes`; `BLOOMERY_CARD_BUDGET` is
//! refused here, so the environment cannot move the sets). It caps every card, the tier's too. Without it the union is the
//! stage's set plus the tier's, each planned up to its own card's usable bytes, and the two together pass the A6000's. Under
//! it the stage's resident bytes are at most `BUDGET` and the tier's experts at most `BUDGET` less its batch reserve, so
//! the reference holds at most 2 · `BUDGET` = 32 GiB of the A6000's 48 GiB [derived from the placement's rule]. Each layer's
//! stage card keeps its id prefix and the tier the next ids, so the preconditions below are met by the prompts' routing,
//! which the run checks.
//!
//! The prompt is the fixed walk of `e2e::walk` through the file's vocabulary. Each load runs two legs, each from a reset:
//!
//! - step: the first [`PROMPT`] ids one decode step each, then [`STEPS`] greedy steps, every position's argmax and logits
//!   read;
//! - call: the first [`CALL_PROMPT`] ids as one prompt call (one batch: the tier serves the slots of its experts from the
//!   batch), its token and logits row read.
//!
//! The two cards' load then runs a group leg, from a reset each: the first [`GROUP_PROMPT`] ids (two batches) as one prompt
//! call at groups of one batch and at groups of two (`set_prefill_group`), each call's token and logits row read.
//!
//! Clauses, against the reference:
//! - decode bits: the step leg's tokens and logits rows bit for bit the reference's;
//! - call bits: the call leg's token and logits row bit for bit the reference's (no decode step follows the call, so a
//!   decode defect cannot turn this clause red);
//! - group bits: the group leg's call at groups of two bit for bit its call at groups of one — a tiered layer's join reads
//!   the unit's route weights and tier places after the next unit's front has routed;
//! - structure: the stage's captured step graph holds the reference's node count;
//! - precondition: every layer the tier holds experts of was sent at least one routed slot by the step leg, and by the call
//!   alone, whose batch the tier served (its batch services); or the run proves nothing and fails by name. A miss is a
//!   prompt too short for the routing (raise [`PROMPT`], [`STEPS`] or [`CALL_PROMPT`]), not a defect of the tier.
//!
//! Tiers (`BLOOMERY_TIER`): the card budget is [`BUDGET`] in the real tier and the header's in the fixture tier (the budget
//! the generator planned to put half of each card-eligible layer's experts on a card). Every clause is a self-consistency
//! clause, so the fixture tier runs them all.

#[cfg(feature = "solar2")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_solar2_twocard", gate::run())
}

#[cfg(feature = "solar2")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "solar2")]
#[path = "shared/quiet.rs"]
mod quiet;

#[cfg(feature = "solar2")]
#[allow(
    dead_code,
    reason = "the two-card gates share the module; each takes the helpers it reads"
)]
#[path = "shared/tier_twocard.rs"]
mod tier_twocard;

#[cfg(feature = "solar2")]
mod gate {
    use std::path::Path;
    use std::time::Instant;

    use app::arch::HybridCfg;
    use app::{Loaded, OpenArgs, Prompt as _, Session};
    use bloomery_gpu::host::swap::Residency;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::tier::{self, Tier};
    use bloomery_gpu_gates::{GateError, checks_failed, ref_model_path};
    use bloomery_gpu_solar2::{Body, Solar2Model, set_prefill_group};
    use bloomery_levers::{CARD_DONTNEED, HOST_LOCK, HOST_POPULATE};
    use gguf::Split;
    use model::arch::solar2::place::{self, PlanInputs};
    use model::placement::{Device, ExpertList, Machine, Plan, PlanLevers, Role, workstation};
    use refset::arch::solar2::ARCH;

    use crate::e2e::walk;
    use crate::quiet::Quiet;
    use crate::tier_twocard::{Leg, Sent, same, sent_between, sent_every, union_plan};

    const NAME: &str = "gate_solar2_twocard";
    /// The card budget both plans are made under in the real tier (module header: the union it leaves fits the A6000).
    const BUDGET: u64 = 16 << 30;
    /// Cache rows: past every leg's last position (the group leg's [`GROUP_PROMPT`] ids and the step leg's
    /// [`PROMPT`] + [`STEPS`] are the most).
    const CTX: usize = 2048;
    /// Prompt positions of the step leg, one decode step each.
    const PROMPT: usize = 32;
    /// Greedy steps after the step leg's prompt.
    const STEPS: usize = 48;
    /// Prompt positions of the call leg's prompt call, one whole batch (`bloomery_gpu_solar2::body::T_MAX`): with each layer's
    /// tier on the ids after its stage prefix, the call's routing must reach every tier layer.
    const CALL_PROMPT: usize = 512;
    /// Prompt positions of the group leg's calls: two whole batches, one group at groups of two.
    const GROUP_PROMPT: usize = 1024;

    /// What a load's two legs read, and what they sent the tier.
    struct Run {
        step: Leg,
        call: Leg,
        stage_nodes: usize,
        /// What the step leg and the call alone sent the tier; `None` without one.
        sent_step: Option<Sent>,
        sent_call: Option<Sent>,
    }

    /// The tier's per-layer hits and batch counters of `m`'s load since it began; `None` without a tier.
    fn counts(m: &Solar2Model) -> Result<Option<Sent>, GateError> {
        Ok(crate::tier_twocard::tier_counts(m.body(NAME)?.hybrid()))
    }

    /// The tiered load through the binaries' open, refused unless its tier's `Gpu` is the 3090.
    fn open_two(
        file: &Path,
        machine: impl Fn(usize) -> Machine,
        cfg: &HybridCfg,
    ) -> Result<Session<Body>, GateError> {
        let t0 = Instant::now();
        let args = OpenArgs {
            place: Place::Bp.name(),
            machine,
            ctx: CTX,
            mode: StepMode::Graph,
            cfg: cfg.clone(),
        };
        let s = Loaded::<Body>::open(split_of(file)?, args, &mut Quiet)?
            .ok_or("the open planned nothing")?
            .ready(&mut Quiet)?;
        let m = s.model();
        let stage = (m.gpu().device_name()?, m.resident_bytes());
        crate::tier_twocard::tier_on_3090("load", t0, (&stage.0, stage.1), m.body(NAME)?.hybrid())?;
        Ok(s)
    }

    /// The reference: `plan`, whose stage card holds the union ([`union_plan`]), opened on card 0 with no tier.
    fn open_union(
        file: &Path,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        cfg: &HybridCfg,
    ) -> Result<Session<Body>, GateError> {
        let t0 = Instant::now();
        let mut model = Body::open_placed(split_of(file)?, plan, inputs, 0, cfg.host)?;
        model.set_mode(StepMode::Graph);
        if !model.body(NAME)?.hybrid().tiers().is_empty() {
            return Err("the reference load holds a tier card".into());
        }
        let device = model.gpu().device_name()?;
        if !device.contains(workstation::A6000.name) {
            return Err(format!(
                "the reference opened on {device}, not the {}: the union is planned for its bytes",
                workstation::A6000.name
            )
            .into());
        }
        let s = Loaded::from_model(model, cfg.clone(), u32::try_from(plan.ctx_max)?)
            .ready(&mut Quiet)?;
        println!(
            "reference load in {:.1} s: {} ({} B) holds the union, no tier",
            t0.elapsed().as_secs_f64(),
            s.model().gpu().device_name()?,
            s.model().resident_bytes(),
        );
        Ok(s)
    }

    fn split_of(file: &Path) -> Result<Split, GateError> {
        Split::open(file).map_err(|e| format!("open {}: {e}", file.display()).into())
    }

    /// The model file the profile names, a solar-open2 file (`ref_model_path` holds a whole fixture in the fixture tier).
    fn model_path() -> Result<std::path::PathBuf, GateError> {
        let path = ref_model_path()?;
        let arch = split_of(&path)?.architecture().map(str::to_owned);
        if arch.as_deref() != Some(ARCH) {
            return Err(format!(
                "the model file is {arch:?}, want {ARCH}: run through `just gate-gpu-solar2-twocard`, which picks the solar-open2 profile"
            )
            .into());
        }
        Ok(path)
    }

    /// Per routed layer, the experts its stage card, its tier card and the host hold (the first routed stack's segments: every
    /// stack of a layer is split alike).
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

    /// One decode step of `id`, its token and logits into `leg`.
    fn step(m: &mut Solar2Model, id: u32, leg: &mut Leg) -> Result<u32, GateError> {
        let tok = m.step(&[id])?;
        leg.tokens.push(tok);
        leg.logits.push(m.logits()?);
        Ok(tok)
    }

    /// The two legs (module header).
    fn legs(s: &mut Session<Body>, prompt: &[u32]) -> Result<Run, GateError> {
        let m = s.model_mut();
        let mut leg = Leg::default();
        m.reset()?;
        let before = counts(m)?;
        let mut tok = 0;
        for &id in &prompt[..PROMPT] {
            tok = step(m, id, &mut leg)?;
        }
        let stage_nodes = m.step_graph_nodes()?.len();
        for _ in 0..STEPS {
            tok = step(m, tok, &mut leg)?;
        }
        let sent_step = sent_between(before, counts(m)?);
        let step_leg = leg;

        let mut leg = Leg::default();
        m.reset()?;
        let before = counts(m)?;
        let tok = Body::prompt(m, &prompt[..CALL_PROMPT])?;
        let sent_call = sent_between(before, counts(m)?);
        leg.tokens.push(tok);
        leg.logits.push(m.logits()?);
        Ok(Run {
            step: step_leg,
            call: leg,
            stage_nodes,
            sent_step,
            sent_call,
        })
    }

    /// The group leg on `s`: the first [`GROUP_PROMPT`] ids of `prompt` as one call at groups of one batch, then at groups of
    /// two, each from a reset, its token and logits row; the session left at groups of one.
    fn group_leg(s: &mut Session<Body>, prompt: &[u32]) -> Result<[Leg; 2], GateError> {
        let m = s.model_mut();
        let mut out = [Leg::default(), Leg::default()];
        for (leg, g) in out.iter_mut().zip([1, 2]) {
            set_prefill_group(m, g)?;
            m.reset()?;
            let tok = Body::prompt(m, &prompt[..GROUP_PROMPT])?;
            leg.tokens.push(tok);
            leg.logits.push(m.logits()?);
        }
        set_prefill_group(m, 1)?;
        m.reset()?;
        Ok(out)
    }

    pub fn run() -> Result<(), GateError> {
        let levers =
            bloomery_levers::at_main(&tier::acts_on(&[HOST_POPULATE, HOST_LOCK, CARD_DONTNEED])?)?;
        let file = model_path()?;
        let split = split_of(&file)?;
        let inputs = PlanInputs::read(&split)?;
        // Both plans run under one budget that no environment moves: the real tier's [`BUDGET`], a parameter of the real
        // file (the union it leaves fits the A6000); the fixture tier's, the header's, which the generator planned to put
        // half of each card-eligible layer's experts on a card.
        let place_levers = match Tier::from_env()? {
            Tier::Real => PlanLevers {
                card_budget_bytes: Some(BUDGET),
            },
            Tier::Fixture => tier::plan_levers(&split, &levers, 0)?,
        };
        drop(split);
        let cfg = HybridCfg {
            place: place_levers,
            host: levers.host(),
            residency: Residency::Off,
            group: 1,
        };
        let pass = bits(&file, &cfg, &inputs)?;
        println!("{NAME}: {}", tier::tally_line());
        if !pass {
            return Err(checks_failed());
        }
        println!("PASSED: {NAME}");
        Ok(())
    }

    /// The bits arm: the union reference, then the two cards.
    fn bits(file: &Path, cfg: &HybridCfg, inputs: &PlanInputs) -> Result<bool, GateError> {
        let layers = inputs.model.layers;
        let bp = Place::Bp
            .on_host()?
            .machine(None, Some(place::tier_batch(&inputs.hp)))?;
        let two = bp(layers);
        let plan = inputs.plan(&two, u64::try_from(CTX)?, &cfg.place)?;
        let budget = cfg
            .place
            .card_budget_bytes
            .ok_or("the gate plans under a card budget")?;
        let splits = layer_splits(&plan);
        let tier_layers: Vec<usize> = splits.iter().filter(|o| o.2 > 0).map(|o| o.0).collect();
        let stage_layers = splits.iter().filter(|o| o.1 > 0).count();
        println!(
            "plan (b′) under a card budget of {budget} B at ctx {CTX}: stage {} experts ({} B), tier {} experts ({} B), host {} \
             experts ({} B); the tier holds experts on {} layers, the stage on {stage_layers}",
            plan.cards[0].experts,
            plan.cards[0].expert_bytes,
            plan.cards.get(1).map_or(0, |c| c.experts),
            plan.cards.get(1).map_or(0, |c| c.expert_bytes),
            plan.host.experts,
            plan.host.expert_bytes,
            tier_layers.len(),
        );
        for (l, stage, tier, host) in &splits {
            println!("  layer {l}: stage {stage}, tier {tier}, host {host}");
        }
        let mut pass = true;
        tier::sc("precondition tier: the plan puts experts on the tier")?;
        if tier_layers.is_empty() {
            println!(
                "FAIL precondition tier: the plan holds no expert on the tier: the run proves nothing"
            );
            pass = false;
        } else {
            println!(
                "ok precondition tier: the tier holds experts on {} layers",
                tier_layers.len()
            );
        }
        let vocab = inputs.hp.n_vocab;
        let prompt = walk(GROUP_PROMPT, vocab);

        let flat = Machine {
            tiers: Vec::new(),
            ..two.clone()
        };
        let uplan = union_plan(&plan, &flat)?;
        println!(
            "reference: the stage card holds {} experts, the union of the stage's and the tier's",
            uplan.n_l.iter().sum::<u64>()
        );
        let want = {
            let mut s = open_union(file, &uplan, inputs, cfg)?;
            legs(&mut s, &prompt)?
        };
        let (got, [g1, g2]) = {
            let mut s = open_two(file, bp, cfg)?;
            let run = legs(&mut s, &prompt)?;
            (run, group_leg(&mut s, &prompt)?)
        };
        for name in [
            "decode bits: step",
            "call bits",
            "group bits: groups of two against groups of one",
            "structure: the stage's step graph has the reference's node count",
        ] {
            tier::sc(name)?;
        }
        pass &= same("decode bits: step", &want.step, &got.step);
        pass &= same("call bits", &want.call, &got.call);
        pass &= same("group bits: groups of two against groups of one", &g1, &g2);
        if got.stage_nodes == want.stage_nodes {
            println!(
                "ok structure: stage graph of {} nodes, the reference's",
                got.stage_nodes
            );
        } else {
            println!(
                "FAIL structure: stage graph of {} nodes, the reference's {}",
                got.stage_nodes, want.stage_nodes
            );
            pass = false;
        }
        tier::sc("precondition sent: every tier layer was sent a routed slot")?;
        pass &= sent_every("decode", got.sent_step.as_ref(), &tier_layers, false);
        pass &= sent_every("call", got.sent_call.as_ref(), &tier_layers, true);
        Ok(pass)
    }
}
