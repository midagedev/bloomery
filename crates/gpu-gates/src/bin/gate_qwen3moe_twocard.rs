//! GPU gate for Qwen3-30B (`Body`) and Qwen3.6 (`Body35`) on plan (b′)
//! (`--place bp`): the stage on the A6000 and one expert tier on the 3090,
//! against one card holding the same expert sets.
//!
//! Each arm is two loads of one plan (b′), one after the other. The reference
//! is the plan with every tier segment joined onto the stage card and the
//! tier card dropped ([`union_plan`]), opened by `GpuModel::open_placed`: the
//! A6000 holds the union, and no tier code runs. The two-card load is the
//! same plan through the same open, which hangs the tier.
//!
//! Memory: both plans are made under one card budget. In the real tier that
//! budget is [`BUDGET`] (`PlanLevers::card_budget_bytes`; `BLOOMERY_CARD_BUDGET`
//! is not a lever here, so the environment cannot move the sets). It caps
//! every card, the tier's too, so the reference holds at most 2 · `BUDGET` =
//! 32 GiB of the A6000's 48 GiB [derived from the placement's rule, the same
//! budget the solar-open2 two-card gate uses]. The budget is what puts
//! experts on the tier: without a positive tier count the arm fails by name
//! before either load. The fixture tier's budget is the file header's
//! ([`tier::plan_levers`]).
//!
//! The prompt is the fixed walk of `e2e::walk` through the file's vocabulary.
//! Each load, from a reset: the first [`PROMPT`] ids one decode step each,
//! then [`STEPS`] greedy steps. Every token and logits row is read. No prompt
//! call: a prompt unit beside a tier expert is refused by name.
//!
//! Clauses, against the reference, each arm that runs:
//! - precondition tier: the plan puts experts on the tier, or the run proves
//!   nothing and fails by name before either load;
//! - decode bits: the tokens and logits rows bit for bit the reference's;
//! - precondition sent: every layer the tier holds experts of was sent at
//!   least one routed slot by that decode (no batch service is required).
//!
//! The gate does not compare either arm to a reference set. Qwen3.6's sets,
//! named and unread here, are `ref_qwen35moe`, `ref_qwen35moe_step4_every_node`
//! and `ref_qwen35moe_d1k` (fixture `fx_ref_qwen35moe`,
//! `fx_ref_qwen35moe_step4_every_node`, `fx_ref_qwen35moe_d1k`, family
//! `fx-ik-qwen35moe`). This gate reads no Qwen3-30B set: the fixture tier checks
//! that arm's header is `qwen3moe` and skips it, and opens neither the real file
//! nor the fixture's.
//!
//! Arguments: the Qwen3-30B file, then the Qwen3.6 file. The second file's
//! architecture is `qwen35` or `qwen35moe`.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen3moe_twocard", gate::run())
}

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; this one reads the fixed walk"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the gate plans a qwen3moe or qwen35moe file; the module also serves the CLI and the seat"
)]
#[path = "shared/qwen3moe_place.rs"]
mod q3place;

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the two-card gates share the module; this one reads the union, the bits and the tier counters"
)]
#[path = "shared/tier_twocard.rs"]
mod tier_twocard;

#[cfg(feature = "gpu")]
mod gate {
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use bloomery_gpu::arch::qwen3moe::{Body, Body35, FlashKind, Open35, OpenOpts, Placed};
    use bloomery_gpu::kv_format::KvQ8;
    use bloomery_gpu::model::{ChainBody, GpuModel};
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::tier::{self, Tier};
    use bloomery_gpu_gates::{GateError, checks_failed};
    use bloomery_levers::{CARD_DONTNEED, HOST_LOCK, HOST_POPULATE, Levers};
    use gguf::Split;
    use model::placement::{Plan, PlanLevers, workstation};

    use crate::e2e::walk;
    use crate::q3place::PlaceQ3;
    use crate::tier_twocard::{
        Leg, Sent, same, sent_between, sent_every, tier_counts, tier_on_3090,
    };
    use model::placement::union_plan;

    const NAME: &str = "gate_qwen3moe_twocard";
    /// The card budget both plans are made under in the real tier (module header).
    const BUDGET: u64 = 16 << 30;
    /// Cache rows: past the step leg's last position ([`PROMPT`] + [`STEPS`]).
    const CTX: usize = 2048;
    /// Prompt positions, one decode step each.
    const PROMPT: usize = 32;
    /// Greedy steps after the prompt.
    const STEPS: usize = 24;

    /// The Qwen3-30B file and the Qwen3.6 file, in that order.
    fn paths() -> Result<(PathBuf, PathBuf), GateError> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let [q3, q35] = args.as_slice() else {
            return Err(format!(
                "{NAME}: want the Qwen3-30B file and the Qwen3.6 file, got {} arguments",
                args.len()
            )
            .into());
        };
        Ok((PathBuf::from(q3), PathBuf::from(q35)))
    }

    fn reopen(path: &Path) -> Result<Split, GateError> {
        Split::open(path).map_err(|e| format!("open {}: {e}", path.display()).into())
    }

    fn place_levers(split: &Split, tier: Tier, levers: &Levers) -> Result<PlanLevers, GateError> {
        match tier {
            Tier::Real => Ok(PlanLevers {
                card_budget_bytes: Some(BUDGET).into(),
            }),
            Tier::Fixture => tier::plan_levers(split, levers, 0),
        }
    }

    /// The layers whose tier count is positive, and the experts on every tier.
    fn tier_of(plan: &Plan<'_>) -> (u64, Vec<usize>) {
        let experts = plan.tier_n_l.iter().flatten().sum();
        let layers = plan
            .tier_n_l
            .first()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter(|(_, n)| **n > 0)
                    .map(|(i, _)| i)
                    .collect()
            })
            .unwrap_or_default();
        (experts, layers)
    }

    /// One decode of `prompt`: [`PROMPT`] steps of its ids, then [`STEPS`]
    /// greedy steps. `counts` is read after the reset and after the last step.
    fn history<B: ChainBody>(
        m: &mut GpuModel<B>,
        prompt: &[u32],
        counts: &impl Fn(&GpuModel<B>) -> Result<Option<Sent>, GateError>,
    ) -> Result<(Leg, Option<Sent>), GateError> {
        m.reset()?;
        let before = counts(m)?;
        let mut leg = Leg::default();
        let mut tok = 0;
        for &id in &prompt[..PROMPT] {
            tok = m.step(&[id])?;
            leg.tokens.push(tok);
            leg.logits.push(m.logits()?);
        }
        for _ in 0..STEPS {
            tok = m.step(&[tok])?;
            leg.tokens.push(tok);
            leg.logits.push(m.logits()?);
        }
        Ok((leg, sent_between(before, counts(m)?)))
    }

    /// The qwen3moe bodies' placed side, which the gate reads on both programs.
    trait PlacedBody: ChainBody {
        fn placed_side(&self) -> Option<&Placed>;
    }

    impl PlacedBody for Body {
        fn placed_side(&self) -> Option<&Placed> {
            self.placed()
        }
    }

    impl PlacedBody for Body35 {
        fn placed_side(&self) -> Option<&Placed> {
            self.placed()
        }
    }

    /// The reference, then the two cards. An open or a card check that fails
    /// is the error; a bits or sent miss is `Ok(false)`.
    fn bits<B: PlacedBody>(
        union: &Plan<'_>,
        tiered: &Plan<'_>,
        prompt: &[u32],
        layers: &[usize],
        open: &mut impl FnMut(&Plan<'_>) -> Result<GpuModel<B>, GateError>,
        counts: &impl Fn(&GpuModel<B>) -> Result<Option<Sent>, GateError>,
    ) -> Result<bool, GateError> {
        let want = {
            let t0 = Instant::now();
            let mut model = open(union)?;
            {
                let placed = model
                    .body(NAME)?
                    .placed_side()
                    .ok_or("the joined reference holds no placed side")?;
                if !placed.hybrid().tiers().is_empty() {
                    return Err("the joined reference holds a tier card".into());
                }
            }
            let device = model.gpu().device_name()?;
            if !device.contains(workstation::A6000.name) {
                return Err(format!(
                    "the reference opened on {device}, not the {}: the union is planned for its bytes",
                    workstation::A6000.name
                )
                .into());
            }
            println!(
                "reference load in {:.1} s: {device} ({} B) holds the union, no tier",
                t0.elapsed().as_secs_f64(),
                model.resident_bytes(),
            );
            history(&mut model, prompt, counts)?.0
        };
        let (got, sent) = {
            let t0 = Instant::now();
            let mut model = open(tiered)?;
            {
                let stage = (model.gpu().device_name()?, model.resident_bytes());
                let placed = model
                    .body(NAME)?
                    .placed_side()
                    .ok_or("the two-card load holds no placed side")?;
                tier_on_3090("load", t0, (&stage.0, stage.1), placed.hybrid())?;
            }
            history(&mut model, prompt, counts)?
        };
        tier::sc("decode bits: the step's tokens and logits")?;
        let mut pass = same("decode bits: step", &want, &got);
        tier::sc("precondition sent: every tier layer was sent a routed slot")?;
        pass &= sent_every("decode", sent.as_ref(), layers, false);
        Ok(pass)
    }

    /// Plan `path` under the arm's budget and run [`bits`] when the plan puts
    /// experts on the tier. `Ok(false)` is a failed clause; an open is an error.
    #[allow(
        clippy::too_many_arguments,
        reason = "one arm's file, tier, architecture check and the body's open and count hooks"
    )]
    fn arm<B: PlacedBody>(
        path: &Path,
        tier: Tier,
        levers: &Levers,
        want: &str,
        arch_ok: impl Fn(Option<&str>) -> bool,
        vocab_of: impl Fn(&Split) -> Result<usize, GateError>,
        place_of: impl FnOnce(&Split) -> Result<PlaceQ3, GateError>,
        mut open: impl FnMut(&Plan<'_>) -> Result<GpuModel<B>, GateError>,
        counts: &impl Fn(&GpuModel<B>) -> Result<Option<Sent>, GateError>,
    ) -> Result<bool, GateError> {
        let split = reopen(path)?;
        let arch = split.architecture().map(str::to_owned);
        if !arch_ok(arch.as_deref()) {
            return Err(format!("the file is {arch:?}, want {want}").into());
        }
        let vocab = vocab_of(&split)?;
        if vocab <= 1_000 {
            return Err(format!("{NAME}: vocabulary {vocab} does not hold the walk's ids").into());
        }
        let place = place_of(&split)?;
        let place_levers = place_levers(&split, tier, levers)?;
        drop(split);
        let plan = place.plan(CTX, &place_levers)?;
        let n_tiers = plan.machine.expert_tiers().count();
        if n_tiers > 1 {
            return Err(
                format!("a plan of {n_tiers} expert tiers; the program hangs at most one").into(),
            );
        }
        let budget = plan
            .card_budget
            .every()
            .ok_or("the gate plans under a card budget")?;
        let (experts, layers) = tier_of(&plan);
        let stage_layers = plan.n_l.iter().filter(|n| **n > 0).count();
        let stage = plan.cards.first().ok_or("a plan of no card")?;
        println!(
            "plan (b′) under a card budget of {budget} B at ctx {CTX}: stage {} experts ({} B), tier {} experts ({} B), host {} \
             experts ({} B); the tier holds experts on {} layers, the stage on {stage_layers}",
            stage.experts,
            stage.expert_bytes,
            plan.cards.get(1).map_or(0, |c| c.experts),
            plan.cards.get(1).map_or(0, |c| c.expert_bytes),
            plan.host.experts,
            plan.host.expert_bytes,
            layers.len(),
        );
        tier::sc("precondition tier: the plan puts experts on the tier")?;
        if experts == 0 {
            println!(
                "FAIL precondition tier: the plan holds no expert on the tier: the run proves nothing"
            );
            return Ok(false);
        }
        println!(
            "ok precondition tier: the tier holds experts on {} layers",
            layers.len()
        );
        let mut flat = plan.machine.clone();
        flat.tiers.clear();
        flat.cards.truncate(1);
        let union = union_plan(&plan, &flat)?;
        println!(
            "reference: the stage card holds {} experts, the union of the stage's and the tier's",
            union.n_l.iter().sum::<u64>()
        );
        let prompt = walk(PROMPT, vocab);
        bits(&union, &plan, &prompt, &layers, &mut open, counts)
    }

    fn q3_counts(m: &GpuModel<Body>) -> Result<Option<Sent>, GateError> {
        let side = m
            .body(NAME)?
            .placed()
            .ok_or("the load holds no placed side")?;
        Ok(tier_counts(side.hybrid()))
    }

    fn q35_counts(m: &GpuModel<Body35>) -> Result<Option<Sent>, GateError> {
        let side = m
            .body(NAME)?
            .placed()
            .ok_or("the load holds no placed side")?;
        Ok(tier_counts(side.hybrid()))
    }

    fn qwen3(path: &Path, tier: Tier, levers: &Levers) -> Result<bool, GateError> {
        if tier == Tier::Fixture {
            let split = reopen(path)?;
            let arch = split.architecture().map(str::to_owned);
            if arch.as_deref() != Some("qwen3moe") {
                return Err(format!("the first file is {arch:?}, want qwen3moe").into());
            }
            println!(
                "{NAME}: the fixture tier skips the Qwen3-30B arm: this gate has no fixture \
                 conversion of it, so it opens neither the real file nor the fixture there"
            );
            return Ok(true);
        }
        let opts = OpenOpts {
            ctx: CTX,
            ubatch: 1,
            flash: FlashKind::Mma,
            kv: KvQ8::F16,
        };
        let host = levers.host();
        arm(
            path,
            tier,
            levers,
            "qwen3moe",
            |a| a == Some("qwen3moe"),
            |file| Ok(model::arch::qwen3moe::hparams::Hparams::read(file)?.n_vocab),
            |file| PlaceQ3::qwen3(file, Place::Bp, CTX, KvQ8::F16),
            |plan| {
                Ok(GpuModel::<Body>::open_placed(
                    reopen(path)?,
                    plan,
                    opts,
                    host,
                )?)
            },
            &q3_counts,
        )
    }

    fn qwen35(path: &Path, tier: Tier, levers: &Levers) -> Result<bool, GateError> {
        let opts = Open35 {
            ctx: CTX,
            mma: true,
            ubatch: 1,
            kv: KvQ8::F16,
            media: false,
        };
        let host = levers.host();
        arm(
            path,
            tier,
            levers,
            "qwen35 or qwen35moe",
            |a| a == Some("qwen35") || a == Some("qwen35moe"),
            |file| Ok(model::arch::qwen35moe::hparams::Hparams::read(file)?.n_vocab),
            |file| PlaceQ3::qwen35(file, Place::Bp, opts),
            |plan| {
                Ok(GpuModel::<Body35>::open_placed(
                    reopen(path)?,
                    plan,
                    opts,
                    host,
                )?)
            },
            &q35_counts,
        )
    }

    pub fn run() -> Result<(), GateError> {
        let levers =
            bloomery_levers::at_main(&tier::acts_on(&[HOST_POPULATE, HOST_LOCK, CARD_DONTNEED])?)?;
        let tier = Tier::from_env()?;
        let (q3, q35) = paths()?;
        let mut pass = true;
        pass &= qwen3(&q3, tier, &levers)?;
        pass &= qwen35(&q35, tier, &levers)?;
        println!("{NAME}: {}", tier::tally_line());
        if !pass {
            return Err(checks_failed());
        }
        println!("PASSED: {NAME}");
        Ok(())
    }
}
