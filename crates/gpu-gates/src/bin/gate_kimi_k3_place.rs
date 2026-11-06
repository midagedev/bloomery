//! GPU gate for Kimi K3's placed load (`--place bp`): the stage card holds the dense trunk and each
//! routed layer's first experts, the second card is the expert tier (`LegTier`, in process or
//! served by a loopback `tier_server` through [`bloomery_gpu::arch::kimi_k3::NODE_FAMILY`]), the
//! embedding is a host row, and a plan's share the cards cannot hold is the host tier's (SiTU
//! combined by `model::ops::ExpertForm::Situ`). The fixture tier only: the plans are found by
//! card budgets over a fixture's experts, and the real file's run is
//! `generate_kimi_k3 --place bp`'s.
//!
//! The plan is made on the gate's `bp` machine (the A6000 the stage, the 3090 the tier, each
//! counting its uploads at [`GRANULE`]: the fixture's experts, 38 KB each, fit the shared granules
//! the dense trunk leaves open at the card's 2 MiB, so only a finer granule gives a card budget a
//! window; every child the gate runs gets the same `--granule`). The window is found by bisection
//! from the header's budget: from the first budget that plans (the stage's floor, every expert on
//! the tier) to the first that leaves the tier idle (every expert on the stage); the plans between
//! move one expert at a time from the tier to the stage, and the gate takes the first budget with
//! experts on both cards and none on the host.
//!
//! Clauses, each decided by `tier::sc` (every one runs in the tier the gate has):
//!
//! - `precondition tier`: the plan holds experts on both cards and none on the host, and the load
//!   holds what the plan says per layer (`Placed::n_card`, `Placed::tier_k`).
//! - `placed ids`: the placed load's greedy ids after the prompt equal the whole-card body's
//!   (`GpuModel::open`) on the same ids. The routed sum is `(card + tier) + host` where the whole
//!   card sums every slot in one pass, so the rows may differ by rounding (printed as a relative
//!   L2, not pinned) and the clause is on the argmax, as the e2e gate holds the free chain.
//! - `placed ids, stage without experts`: the same on the first budget of the window, whose plan
//!   holds every expert on the tier card and none on the stage or the host, the real file's shape
//!   (every layer a hosted layer with no card expert).
//! - `placed eager`: the same load stepped eagerly after a reset reads the graph run's ids and
//!   logits bits.
//! - `precondition sent`: every tier layer was sent a routed slot, so the tier's legs ran.
//! - `node ids`: the tier served by a loopback `tier_server` on the 3090 alone
//!   (`BLOOMERY_TIER_NODES`) through `generate_kimi_k3 --place bp` reads the in-process tier's ids
//!   and the same logits bits (an FNV-1a over the last row's bits).
//! - `reserve`: a plan made with a tier reserve of 0 where the load makes tier places is refused
//!   by name at the load (`tier places`).
//! - `host head`: a plan whose stage card carries no head (the head read on the host) is refused
//!   by name at the load: the body ends its step in a card head.
//! - `host share`: a plan whose tier card is budgeted under what the experts need (per-card
//!   budgets: the stage at the window's first, the tier below) leaves experts on the host; the
//!   load holds the plan's counts per layer, and fed the whole-card body's ids each step's logits
//!   row is within [`HOST_REL_BAND`] relative L2 of the whole-card row (the host's SiTU against
//!   the card's rule). The free-running ids are not the clause: a flat fixture row flips its
//!   argmax on a rounding, and the rows after that are another history's.
//! - `host layer without tier`: a plan where a layer holds host experts and no tier expert is
//!   refused by name at the load (the host handoff's entries are not built for K3's slot count).
//! - `node family`: a node that serves only another family refuses the open by name.
//! - `plan records`: `generate_kimi_k3 --plan --place bp` prints the `plan` record of the placed
//!   machine and no host expert (the gate's plan holds none).
//!
//! The prompt is the fixed walk of `e2e::walk` through the vocabulary: [`PROMPT`] ids, [`STEPS`]
//! greedy ids.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    if std::env::args().nth(1).as_deref() == Some(gate::SERVE_OTHER) {
        return bloomery_gpu_gates::exit_with("gate_kimi_k3_place node", gate::serve_other());
    }
    bloomery_gpu_gates::exit_with("gate_kimi_k3_place", gate::run())
}

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/node_proc.rs"]
mod node_proc;

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/tier_twocard.rs"]
mod tier_twocard;

#[cfg(feature = "gpu")]
mod gate {
    use std::net::TcpListener;
    use std::num::NonZeroU64;
    use std::process::Command;
    use std::sync::Arc;

    use app::Prompt;
    use bloomery_gpu::arch::kimi_k3::Body;
    use bloomery_gpu::host::node;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Gpu, GpuModel};
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::tier::{self, Tier};
    use bloomery_gpu_gates::{
        GateError, beside, checks_failed, die_with_parent, open_split, ref_model_path,
    };
    use bloomery_levers::{
        CARD_BUDGET, CARD_DONTNEED, CardBudget, HOST_LOCK, HOST_POPULATE, HostCfg, TIER_NODES,
    };
    use gguf::Split;
    use model::arch::Arch;
    use model::arch::kimi_k3::place::{PlaceError, PlanInputs, tier_batch};
    use model::placement::{
        Device, ExpertList, Machine, PlacementError, Plan, PlanLevers, Role, workstation,
    };

    use crate::node_proc::{LISTENING, tier_server, tier_server_command};
    use crate::tier_twocard::{Sent, sent_every, tier_counts};

    /// The gate's name: the recipe's and the refusals'.
    const NAME: &str = "gate_kimi_k3_place";
    /// The recipe that runs it.
    const RECIPE: &str = "gate-gpu-kimik3-place";
    /// The argument that makes this binary a tier node serving another family.
    pub const SERVE_OTHER: &str = "--serve-other-family";

    /// Cache rows: past the prompt and the generated ids.
    const CTX: usize = 128;
    /// Prompt positions, one decode step each.
    const PROMPT: usize = 6;
    /// Greedy ids after the prompt.
    const STEPS: usize = 8;
    /// The worst step's logits row's relative L2 distance of a load with host experts from the
    /// whole-card body's, both fed the same ids: the host's activation rules (q8_K, q8_2) against
    /// the card's (q8_1) set the floor, `gate_hybrid`'s error model.
    /// PIN(2026-11-06): twice the worst measured step (8.16e-2); the swapped betas read 5.8e-1.
    const HOST_REL_BAND: f64 = 1.6e-1;

    /// The fixed walk: a multiplicative walk through ids the vocabulary holds.
    fn walk(n: usize, vocab: usize) -> Vec<u32> {
        (0..n)
            .map(|i| u32::try_from(1_000 + (i * 7_919 + 13) % (vocab - 1_000)).unwrap_or(1))
            .collect()
    }

    /// FNV-1a over the bits of a logits row: how two processes' rows are held equal.
    fn fnv(row: &[f32]) -> u64 {
        row.iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
            })
    }

    /// What a run read: the greedy ids and the last logits row.
    struct Run {
        ids: Vec<u32>,
        /// The last logits row.
        logits: Vec<f32>,
        /// The logits row after the prompt and after each id fed back: `ids.len()` rows.
        rows: Vec<Vec<f32>>,
    }

    impl Run {
        fn same_bits(&self, other: &Run) -> bool {
            self.ids == other.ids && fnv(&self.logits) == fnv(&other.logits)
        }
    }

    /// The prompt, then [`STEPS`] ids greedy, the last logits row and every step's.
    fn greedy(m: &mut GpuModel<Body>, prompt: &[u32]) -> Result<Run, GateError> {
        let mut ids = vec![Body::prompt(m, prompt)?];
        let mut rows = vec![m.logits()?];
        while ids.len() < STEPS {
            let last = ids[ids.len() - 1];
            ids.push(m.step(&[last])?);
            rows.push(m.logits()?);
        }
        Ok(Run {
            ids,
            logits: m.logits()?,
            rows,
        })
    }

    /// The prompt, then `ids` fed back one a step (all but the last): the logits row after the
    /// prompt and after each id, the rows a free-running load would read had it chosen `ids`.
    fn forced(
        m: &mut GpuModel<Body>,
        prompt: &[u32],
        ids: &[u32],
    ) -> Result<Vec<Vec<f32>>, GateError> {
        Body::prompt(m, prompt)?;
        let mut rows = vec![m.logits()?];
        for &id in &ids[..ids.len() - 1] {
            m.step(&[id])?;
            rows.push(m.logits()?);
        }
        Ok(rows)
    }

    /// The relative L2 distance of row `b` from row `a`.
    fn rel(a: &[f32], b: &[f32]) -> f64 {
        let (mut num, mut den) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            let (x, y) = (f64::from(*x), f64::from(*y));
            num += (x - y) * (x - y);
            den += x * x;
        }
        (num / den).sqrt()
    }

    /// One clause's verdict line.
    fn line(ok: bool, clause: &str, detail: &str) -> bool {
        println!("{} {clause}: {detail}", if ok { "ok" } else { "FAIL" });
        ok
    }

    /// The allocator granule the gate's plans count each card's uploads at. The fixture's
    /// experts (about 38 KB each) fit the shared granules the dense trunk leaves open at the
    /// card's 2 MiB, so no budget at or above the plan's floor splits them between the cards; at
    /// a page the budget has a window of the experts' own bytes. A plan input only: the driver
    /// allocates as it does, and every child the gate runs gets the same `--granule`.
    const GRANULE: u64 = 4 << 10;

    /// The `bp` machine of the gate: the A6000 the stage, the 3090 the tier, each at [`GRANULE`],
    /// the tier's batch reserved.
    fn bp_machine(inputs: &PlanInputs) -> Result<Machine, GateError> {
        let bp = Place::Bp
            .on_host()?
            .machine(None, Some(tier_batch(&inputs.hp)?))?;
        let mut machine = bp(inputs.model.layers);
        let granule = NonZeroU64::new(GRANULE).ok_or("the gate's granule is at least one byte")?;
        for card in machine.cards.iter_mut().chain(machine.tiers.iter_mut()) {
            card.granule_bytes = granule;
        }
        Ok(machine)
    }

    /// The plan levers of the card budget `budget`.
    fn levers_of(budget: CardBudget) -> PlanLevers {
        PlanLevers {
            card_budget_bytes: budget,
            ..PlanLevers::default()
        }
    }

    /// The plan levers of a card budget of `bytes` on every card.
    fn under(bytes: u64) -> PlanLevers {
        levers_of(CardBudget::Every(bytes))
    }

    /// Per routed layer, the experts its stage card, its tier card and the host hold (the first
    /// routed stack's segments: every stack of a layer is split alike).
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

    /// The experts the stage, the tier and the host hold across the layers.
    fn totals(splits: &[(usize, u64, u64, u64)]) -> (u64, u64, u64) {
        splits
            .iter()
            .fold((0, 0, 0), |a, s| (a.0 + s.1, a.1 + s.2, a.2 + s.3))
    }

    /// One split of the experts and the first card budget that gives it.
    struct Budget {
        from: u64,
        stage: u64,
        tier: u64,
        host: u64,
    }

    /// How the plan under a card budget reads.
    enum Cut {
        /// Refused for room (the budget does not hold the stage's dense trunk), with the reason.
        Short(String),
        /// Every expert on the stage: the tier card would hold none.
        Idle,
        /// A plan: the experts the stage, the tier and the host hold.
        Split(u64, u64, u64),
    }

    fn cut(inputs: &PlanInputs, machine: &Machine, levers: &PlanLevers) -> Cut {
        match inputs.plan(machine, CTX as u64, levers) {
            Ok(plan) => {
                let (stage, tier, host) = totals(&layer_splits(&plan));
                Cut::Split(stage, tier, host)
            }
            Err(PlaceError::Placement(PlacementError::IdleTier { .. })) => Cut::Idle,
            Err(e) => Cut::Short(e.to_string()),
        }
    }

    /// The plans the card budgets give, one [`Budget`] per distinct split:
    /// the budgets between the first that plans and the first that leaves the tier idle (a plan
    /// is refused for room below the one and every expert sits on the stage above the other, the
    /// window being the experts' bytes and the allocator's shared granules), found by bisection
    /// from the header's budget and then scanned in 512 steps.
    fn split_table(inputs: &PlanInputs, header: u64) -> Result<Vec<Budget>, GateError> {
        let machine = bp_machine(inputs)?;
        let top = header
            .checked_mul(2)
            .ok_or("the header's budget doubles past u64")?;
        match cut(inputs, &machine, &under(top)) {
            Cut::Idle => {}
            Cut::Short(why) => {
                return Err(
                    format!("twice the header's budget ({top} B) is short of room: {why}").into(),
                );
            }
            Cut::Split(s, t, h) => {
                return Err(format!(
                    "twice the header's budget ({top} B) still splits the experts ({s} stage, {t} tier, {h} host)"
                )
                .into());
            }
        }
        // The first budget that plans: bisect (short, not short).
        let (mut short, mut plans) = (0u64, top);
        while plans - short > 1 {
            let mid = short + (plans - short) / 2;
            if matches!(cut(inputs, &machine, &under(mid)), Cut::Short(_)) {
                short = mid;
            } else {
                plans = mid;
            }
        }
        let first = plans;
        // The first budget that leaves the tier idle: bisect (not idle, idle).
        let (mut busy, mut idle) = (first, top);
        if matches!(cut(inputs, &machine, &under(first)), Cut::Idle) {
            return Err(format!(
                "the first budget that plans ({first} B) already leaves the tier idle: no split"
            )
            .into());
        }
        while idle - busy > 1 {
            let mid = busy + (idle - busy) / 2;
            if matches!(cut(inputs, &machine, &under(mid)), Cut::Idle) {
                idle = mid;
            } else {
                busy = mid;
            }
        }
        let step = ((idle - first) / 512).max(1);
        let mut out: Vec<Budget> = Vec::new();
        let mut b = first;
        while b < idle {
            if let Cut::Split(stage, tier, host) = cut(inputs, &machine, &under(b))
                && out
                    .last()
                    .is_none_or(|l| (l.stage, l.tier, l.host) != (stage, tier, host))
            {
                out.push(Budget {
                    from: b,
                    stage,
                    tier,
                    host,
                });
            }
            b += step;
        }
        println!(
            "budgets: plans from {first} B, leaves the tier idle from {idle} B (header {header} B); \
             {} distinct splits (stage, tier, host experts)",
            out.len()
        );
        for o in &out {
            println!("  from {} B: {}, {}, {}", o.from, o.stage, o.tier, o.host);
        }
        Ok(out)
    }

    /// A plan that leaves experts on the host: its levers and the experts each tier holds.
    struct HostShare {
        levers: PlanLevers,
        stage: u64,
        tier: u64,
        host: u64,
    }

    /// The card budgets of a plan that leaves experts on the host: the stage card at `stage`, the
    /// tier card at the first budget above its floor that gives it some experts and the host the
    /// rest, and the split that makes; with `tierless`, a plan where some layer holds host experts
    /// and no tier expert, and without, one where every layer with host experts holds tier ones
    /// too. None when no budget does.
    fn host_share_plan(
        inputs: &PlanInputs,
        stage: u64,
        tierless: bool,
    ) -> Result<Option<HostShare>, GateError> {
        let machine = bp_machine(inputs)?;
        let (Some(s), Some(t)) = (
            machine.cards.first().and_then(|c| c.device),
            machine.tiers.first().and_then(|c| c.device),
        ) else {
            return Err(
                "the gate's cards carry no census device, which a per-card budget names".into(),
            );
        };
        let of =
            |tier: u64| levers_of(CardBudget::Per(vec![(s.ordinal, stage), (t.ordinal, tier)]));
        if matches!(cut(inputs, &machine, &of(stage)), Cut::Short(_)) {
            return Ok(None);
        }
        let (mut short, mut plans) = (0u64, stage);
        while plans - short > 1 {
            let mid = short + (plans - short) / 2;
            if matches!(cut(inputs, &machine, &of(mid)), Cut::Short(_)) {
                short = mid;
            } else {
                plans = mid;
            }
        }
        let step = ((stage - plans) / 512).max(1);
        let mut b = plans;
        while b < stage {
            if let Ok(plan) = inputs.plan(&machine, CTX as u64, &of(b)) {
                let splits = layer_splits(&plan);
                let (stage_n, tier, host) = totals(&splits);
                let has_tierless = splits.iter().any(|&(_, _, t, h)| h > 0 && t == 0);
                if tier > 0 && host > 0 && has_tierless == tierless {
                    return Ok(Some(HostShare {
                        levers: of(b),
                        stage: stage_n,
                        tier,
                        host,
                    }));
                }
            }
            b += step;
        }
        Ok(None)
    }

    /// The budget of the most even split with experts on both cards and none on the host: the
    /// first such budget whose stage and tier counts differ least, so the routed slots reach both
    /// cards' experts and the card sum and the tier sum are both read.
    fn pick_budget(table: &[Budget]) -> Result<u64, GateError> {
        table
            .iter()
            .filter(|o| o.stage > 0 && o.tier > 0 && o.host == 0)
            .min_by_key(|o| o.stage.abs_diff(o.tier))
            .map(|o| o.from)
            .ok_or_else(|| {
                "no budget puts experts on both cards and none on the host (the splits above)"
                    .into()
            })
    }

    /// The whole-card body's run: the reference.
    fn whole(prompt: &[u32]) -> Result<Run, GateError> {
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let mut m = GpuModel::<Body>::open(Gpu::new()?, file, CTX)?;
        m.set_mode(StepMode::Graph);
        greedy(&mut m, prompt)
    }

    /// `generate_kimi_k3` over `prompt` on `--place bp` with the given environment: its status,
    /// stdout and stderr.
    fn generate(
        prompt: &[u32],
        env: &[(&str, String)],
        plan_only: bool,
    ) -> Result<(bool, String, String), GateError> {
        let bin = beside("generate_kimi_k3")?;
        let ids = prompt
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let mut cmd = Command::new(&bin);
        cmd.args(["--tokens", &ids, "-n", &STEPS.to_string()])
            .args(["--ctx", &CTX.to_string(), "--place", "bp"])
            .args(["--granule", &GRANULE.to_string()]);
        if plan_only {
            cmd.arg("--plan");
        }
        for (k, v) in env {
            cmd.env(k, v);
        }
        die_with_parent(&mut cmd);
        let out = cmd
            .output()
            .map_err(|e| format!("{}: {e}", bin.display()))?;
        Ok((
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        ))
    }

    /// The ids and the logits hash `generate_kimi_k3` printed.
    fn printed(stdout: &str) -> Result<(Vec<u32>, u64), GateError> {
        let ids = stdout
            .lines()
            .find_map(|l| l.strip_prefix("ids ["))
            .and_then(|l| l.strip_suffix(']'))
            .ok_or_else(|| format!("generate_kimi_k3 printed no ids line:\n{stdout}"))?
            .split(',')
            .map(|t| t.trim().parse::<u32>().map_err(GateError::from))
            .collect::<Result<Vec<_>, _>>()?;
        let hash = stdout
            .lines()
            .find_map(|l| l.strip_prefix("logits fnv1a64="))
            .and_then(|l| l.split_whitespace().next())
            .ok_or_else(|| format!("generate_kimi_k3 printed no logits line:\n{stdout}"))?;
        Ok((ids, u64::from_str_radix(hash, 16)?))
    }

    /// `CUDA_VISIBLE_DEVICES` of a node: the 3090 alone, by the UUID the driver reports for it.
    fn node_card() -> Result<String, GateError> {
        let census = bloomery_gpu::census()?;
        let device = census
            .iter()
            .find(|d| d.name.contains(workstation::RTX_3090.name))
            .ok_or_else(|| {
                format!(
                    "the node takes the {} and the census has none: {}",
                    workstation::RTX_3090.name,
                    census
                        .iter()
                        .map(|d| d.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;
        Ok(bloomery_gpu_gates::gpu_census::gpu_uuid(&device.uuid))
    }

    /// The error text of a load that must be refused; `None` if it opened.
    fn refused(r: Result<GpuModel<Body>, bloomery_gpu::GpuError>) -> Option<String> {
        r.err().map(|e| e.to_string())
    }

    /// This binary as a tier node serving the Qwen3.8 family alone over the file: what a K3
    /// stage's open is refused by. Prints the address it listens on.
    pub fn serve_other() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let path = ref_model_path()?;
        let file =
            Arc::new(Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?);
        let listener = TcpListener::bind("127.0.0.1:0")?;
        println!("{LISTENING}{}", listener.local_addr()?);
        std::io::Write::flush(&mut std::io::stdout())?;
        node::serve(
            &listener,
            &file,
            &[bloomery_gpu::arch::qwen3moe::NODE_FAMILY],
            |words| println!("{words}"),
        )?;
        Ok(())
    }

    pub fn run() -> Result<(), GateError> {
        if Tier::from_env()? != Tier::Fixture {
            return Err(format!(
                "{NAME}: the gate's plans split the fixture's experts by card budget, so it runs \
                 the fixture tier: BLOOMERY_TIER=fixture just {RECIPE}; the real file's run is \
                 generate_kimi_k3 --place bp"
            )
            .into());
        }
        let levers =
            bloomery_levers::at_main(&tier::acts_on(&[HOST_POPULATE, HOST_LOCK, CARD_DONTNEED])?)?;
        let split = open_split(Arch::KimiK3, RECIPE)?;
        let inputs = PlanInputs::read_without_chat(&split)?;
        let header = tier::plan_levers(&split, &levers, 0)?
            .card_budget_bytes
            .every()
            .ok_or("the fixture tier plans under the header's one budget")?;
        drop(split);
        if PROMPT + STEPS > CTX {
            return Err(format!("{PROMPT} + {STEPS} positions, the caches hold {CTX}").into());
        }
        let prompt = walk(PROMPT, inputs.hp.n_vocab);
        let table = split_table(&inputs, header)?;
        let budget = pick_budget(&table)?;
        let host = levers.host();
        let want = whole(&prompt)?;
        println!("whole-card ids {:?}", want.ids);
        let mut pass = true;
        pass &= placed(&inputs, budget, host, (&prompt, &want))?;
        pass &= tier_only(&inputs, &table, host, (&prompt, &want))?;
        pass &= refusals(&inputs, budget, host)?;
        pass &= host_share(&inputs, &table, (budget, host), (&prompt, &want))?;
        pass &= host_without_tier(&inputs, &table, (budget, host))?;
        pass &= node_family_and_records(&prompt, budget)?;
        println!("{NAME}: {}", tier::tally_line());
        if !pass {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} the placed load's greedy ids equal the whole-card body's, eager reads \
             the graph's bits, the loopback node's tier reads the in-process tier's bits, a plan's \
             host share reads the whole-card logits rows within the band, and the load refuses a wrong tier \
             reserve and a node of another family by name"
        );
        Ok(())
    }

    /// The placed load against the whole-card body, its eager run, and the tier on a node.
    fn placed(
        inputs: &PlanInputs,
        budget: u64,
        host: HostCfg,
        (prompt, want): (&[u32], &Run),
    ) -> Result<bool, GateError> {
        let machine = bp_machine(inputs)?;
        let plan = inputs.plan(&machine, CTX as u64, &under(budget))?;
        let splits = layer_splits(&plan);
        let (stage, tier_n, host_n) = totals(&splits);
        println!(
            "plan under a card budget of {budget} B at ctx {CTX}: stage {stage} experts, tier \
             {tier_n}, host {host_n}"
        );
        for (l, s, t, h) in &splits {
            println!("  layer {l}: stage {s}, tier {t}, host {h}");
        }
        let tier_layers: Vec<usize> = splits.iter().filter(|o| o.2 > 0).map(|o| o.0).collect();
        let mut pass = true;
        tier::sc("precondition tier: the plan holds experts on both cards and none on the host")?;

        let t = std::time::Instant::now();
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let mut m = GpuModel::<Body>::open_placed(file, &plan, inputs, 0, host)?;
        m.set_mode(StepMode::Graph);
        println!(
            "placed load in {:.1} s: resident_bytes={} layers={}",
            t.elapsed().as_secs_f64(),
            m.resident_bytes(),
            m.layers().len()
        );
        {
            let p = m
                .body(NAME)?
                .placed()
                .ok_or("the load is not a placed load")?;
            let wrong: Vec<_> = splits
                .iter()
                .filter(|&&(l, s, t, _)| p.n_card(l) as u64 != s || p.tier_k(l) as u64 != t)
                .map(|&(l, ..)| (l, p.n_card(l), p.tier_k(l)))
                .collect();
            pass &= line(
                wrong.is_empty() && !tier_layers.is_empty() && host_n == 0,
                "precondition tier",
                &format!(
                    "the tier holds experts on {} layers, the host none, and the load holds the \
                     plan's counts (layers off the plan: {wrong:?})",
                    tier_layers.len()
                ),
            );
        }

        let got = greedy(&mut m, prompt)?;
        tier::sc("placed ids: the placed load's greedy ids are the whole-card body's")?;
        pass &= line(
            got.ids == want.ids,
            "placed ids",
            &format!(
                "placed {:?}, whole-card {:?}, last logits row relative L2 {:.2e}",
                got.ids,
                want.ids,
                rel(&want.logits, &got.logits)
            ),
        );
        let sent = tier_counts(m.body(NAME)?.placed().ok_or("not placed")?.hybrid());
        tier::sc("precondition sent: every tier layer was sent a routed slot")?;
        pass &= sent_every("placed", sent.as_ref(), &tier_layers, false);
        pass &= stage_hit(&splits, inputs.hp.experts.n_used, sent.as_ref());

        m.reset()?;
        m.set_mode(StepMode::Eager);
        let eager = greedy(&mut m, prompt)?;
        tier::sc("placed eager: the eager run reads the graph run's bits")?;
        pass &= line(
            eager.same_bits(&got),
            "placed eager",
            &format!("eager {:?}, graph {:?}", eager.ids, got.ids),
        );
        drop(m);

        pass &= node(prompt, budget, &got)?;
        Ok(pass)
    }

    /// The stage card's own experts were routed slots: the run evaluated [`PROMPT`] + [`STEPS`] - 1
    /// positions of `n_used` slots a layer, the tier took `hits` of them, and the rest went to the
    /// layer's stage experts, so some layer holding stage experts must show a remainder; a run whose
    /// slots all went to the tier leaves the card sum unread.
    fn stage_hit(splits: &[(usize, u64, u64, u64)], n_used: usize, sent: Option<&Sent>) -> bool {
        let Some(sent) = sent else {
            return false;
        };
        let slots = ((PROMPT + STEPS - 1) * n_used) as u64;
        let stage_hits: Vec<(usize, u64)> = splits
            .iter()
            .filter(|o| o.1 > 0 && o.2 > 0)
            .map(|&(l, ..)| {
                let tier = l
                    .checked_sub(sent.first)
                    .and_then(|i| sent.hits.get(i))
                    .copied()
                    .unwrap_or(0);
                (l, slots.saturating_sub(tier))
            })
            .collect();
        line(
            stage_hits.iter().any(|&(_, n)| n > 0),
            "precondition stage",
            &format!(
                "slots routed to the stage card's experts per layer {stage_hits:?} of {slots}; \
                 the card sum and the tier sum both ran"
            ),
        )
    }

    /// The first budget of a split with every expert on the tier card and none on the stage or
    /// the host: the real file's shape, where the dense trunk leaves the stage no room for an
    /// expert.
    fn pick_tier_only(table: &[Budget]) -> Option<u64> {
        table
            .iter()
            .find(|o| o.stage == 0 && o.tier > 0 && o.host == 0)
            .map(|o| o.from)
    }

    /// The placed load of a plan with no expert on the stage: its greedy ids are the whole-card
    /// body's, as on the gate's own plan.
    fn tier_only(
        inputs: &PlanInputs,
        table: &[Budget],
        host: HostCfg,
        (prompt, want): (&[u32], &Run),
    ) -> Result<bool, GateError> {
        let clause = "placed ids, stage without experts";
        tier::sc("placed ids, stage without experts: the ids are the whole-card body's")?;
        let Some(budget) = pick_tier_only(table) else {
            return Ok(line(
                false,
                clause,
                "no budget up to the header's puts every expert on the tier card: the clause has \
                 no input",
            ));
        };
        let machine = bp_machine(inputs)?;
        let plan = inputs.plan(&machine, CTX as u64, &under(budget))?;
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let mut m = GpuModel::<Body>::open_placed(file, &plan, inputs, 0, host)?;
        m.set_mode(StepMode::Graph);
        let on_stage: usize = {
            let p = m.body(NAME)?.placed().ok_or("not a placed load")?;
            (0..inputs.hp.n_layer).map(|l| p.n_card(l)).sum()
        };
        let got = greedy(&mut m, prompt)?;
        Ok(line(
            on_stage == 0 && got.ids == want.ids,
            clause,
            &format!(
                "stage holds {on_stage} experts; placed {:?}, whole-card {:?}, last logits row \
                 relative L2 {:.2e}",
                got.ids,
                want.ids,
                rel(&want.logits, &got.logits)
            ),
        ))
    }

    /// The tier on a loopback `tier_server` on the 3090 alone, through `generate_kimi_k3`.
    fn node(prompt: &[u32], budget: u64, in_process: &Run) -> Result<bool, GateError> {
        let mut cmd = tier_server_command(None)?;
        cmd.env("CUDA_VISIBLE_DEVICES", node_card()?);
        let (_server, addr) = tier_server(cmd, "node")?;
        let (ok, stdout, stderr) = generate(
            prompt,
            &[
                (TIER_NODES, addr.to_string()),
                (CARD_BUDGET, budget.to_string()),
            ],
            false,
        )?;
        tier::sc("node ids: the loopback node's tier reads the in-process tier's bits")?;
        if !ok {
            println!("generate_kimi_k3 on the node tier failed:\n{stdout}\n{stderr}");
            return Ok(line(false, "node ids", "generate_kimi_k3 ended nonzero"));
        }
        let (ids, hash) = printed(&stdout)?;
        Ok(line(
            ids == in_process.ids && hash == fnv(&in_process.logits),
            "node ids",
            &format!(
                "node {ids:?} logits fnv1a64={hash:016x}, in-process {:?} fnv1a64={:016x}",
                in_process.ids,
                fnv(&in_process.logits)
            ),
        ))
    }

    /// The loads the body refuses by name.
    fn refusals(inputs: &PlanInputs, budget: u64, host: HostCfg) -> Result<bool, GateError> {
        let machine = bp_machine(inputs)?;

        // A tier reserve of 0 where the load makes tier places.
        let wrong = inputs.clone().with_tier_reserve(0);
        let plan = wrong.plan(&machine, CTX as u64, &under(budget))?;
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let text = refused(GpuModel::<Body>::open_placed(file, &plan, &wrong, 0, host));
        tier::sc("reserve: a tier reserve of 0 is refused by name at the load")?;
        let mut pass = line(
            text.as_deref()
                .is_some_and(|t| t.contains("tier places") && t.contains("the plan reserves 0")),
            "reserve",
            &format!("the load of a plan with a tier reserve of 0: {text:?}"),
        );

        // A plan with no head card reads the head on the host, which the body has no stage for.
        // The head's bytes leave the stage card, so its budget loses them and the experts split
        // as the gate's plan does.
        let head_bytes: u64 = plan
            .rows
            .iter()
            .filter(|r| plan.model.tensors[r.tensor].role == Role::Head)
            .flat_map(|r| &r.segments)
            .filter(|s| matches!(s.device, Device::Card(_)))
            .map(|s| s.resident_bytes)
            .sum();
        let mut headless = bp_machine(inputs)?;
        for card in &mut headless.cards {
            card.head = false;
        }
        let plan = inputs.plan(&headless, CTX as u64, &under(budget - head_bytes))?;
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let text = refused(GpuModel::<Body>::open_placed(file, &plan, inputs, 0, host));
        tier::sc("host head: a plan that reads the head on the host is refused by name")?;
        pass &= line(
            text.as_deref()
                .is_some_and(|t| t.contains("reads the head on the host")),
            "host head",
            &format!("the load of a plan with no head card: {text:?}"),
        );
        Ok(pass)
    }

    /// A plan that leaves experts on the host (the tier card's budget squeezed under what the
    /// experts need): the load holds the plan's counts, and its greedy ids and last logits row are
    /// the whole-card body's within [`HOST_REL_BAND`].
    fn host_share(
        inputs: &PlanInputs,
        table: &[Budget],
        (budget, host): (u64, HostCfg),
        (prompt, want): (&[u32], &Run),
    ) -> Result<bool, GateError> {
        let clause = "host share";
        tier::sc("host share: the host tier's experts read the whole-card body's logits rows")?;
        let Some(HostShare {
            levers,
            stage,
            tier: held,
            host: left,
        }) = host_share_plan(inputs, pick_tier_only(table).unwrap_or(budget), false)?
        else {
            return Ok(line(
                false,
                clause,
                "no tier budget leaves an expert on the host: the clause has no input",
            ));
        };
        let machine = bp_machine(inputs)?;
        let plan = inputs.plan(&machine, CTX as u64, &levers)?;
        let splits = layer_splits(&plan);
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let mut m = GpuModel::<Body>::open_placed(file, &plan, inputs, 0, host)?;
        m.set_mode(StepMode::Graph);
        let wrong: Vec<_> = {
            let p = m.body(NAME)?.placed().ok_or("not a placed load")?;
            splits
                .iter()
                .filter(|&&(l, s, t, _)| p.n_card(l) as u64 != s || p.tier_k(l) as u64 != t)
                .map(|&(l, ..)| (l, p.n_card(l), p.tier_k(l)))
                .collect()
        };
        let rows = forced(&mut m, prompt, &want.ids)?;
        let dists: Vec<f64> = want
            .rows
            .iter()
            .zip(&rows)
            .map(|(w, g)| rel(w, g))
            .collect();
        let shown: Vec<String> = dists.iter().map(|d| format!("{d:.2e}")).collect();
        let (at, worst) = dists
            .iter()
            .copied()
            .enumerate()
            .fold((0, 0.0f64), |a, (i, d)| if d > a.1 { (i, d) } else { a });
        Ok(line(
            wrong.is_empty() && rows.len() == want.rows.len() && worst <= HOST_REL_BAND,
            clause,
            &format!(
                "per-card budgets {} hold stage {stage}, tier {held}, host {left} experts (layers \
                 off the plan: {wrong:?}); fed the whole-card ids {:?}, each step's logits row \
                 relative L2 {shown:?}, worst {worst:.3e} at step {at}, band {HOST_REL_BAND:.1e}",
                levers.card_budget_bytes, want.ids
            ),
        ))
    }

    /// A plan where some layer holds host experts and no tier expert: the load refuses it by name,
    /// since the host handoff's entries are built for other slot counts than K3's.
    fn host_without_tier(
        inputs: &PlanInputs,
        table: &[Budget],
        (budget, host): (u64, HostCfg),
    ) -> Result<bool, GateError> {
        let clause = "host layer without tier";
        tier::sc("host layer without tier: a host share with no tier expert is refused by name")?;
        let Some(HostShare {
            levers,
            stage,
            tier: held,
            host: left,
        }) = host_share_plan(inputs, pick_tier_only(table).unwrap_or(budget), true)?
        else {
            return Ok(line(
                false,
                clause,
                "no tier budget leaves a layer with host experts and none on the tier: the clause \
                 has no input",
            ));
        };
        let machine = bp_machine(inputs)?;
        let plan = inputs.plan(&machine, CTX as u64, &levers)?;
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let text = refused(GpuModel::<Body>::open_placed(file, &plan, inputs, 0, host));
        Ok(line(
            text.as_deref()
                .is_some_and(|t| t.contains("keep experts on the host and none on a tier card")),
            clause,
            &format!(
                "per-card budgets {} hold stage {stage}, tier {held}, host {left} experts; its \
                 load: {text:?}",
                levers.card_budget_bytes
            ),
        ))
    }

    /// The node of another family, and the plan records.
    fn node_family_and_records(prompt: &[u32], budget: u64) -> Result<bool, GateError> {
        let mut pass = true;
        let mut cmd = Command::new(std::env::current_exe()?);
        cmd.arg(SERVE_OTHER)
            .env("CUDA_VISIBLE_DEVICES", node_card()?)
            .env_remove(TIER_NODES)
            .env_remove(CARD_BUDGET)
            .env_remove(HOST_POPULATE)
            .env_remove(HOST_LOCK)
            .env_remove(CARD_DONTNEED);
        let (_other, addr) = tier_server(cmd, "other")?;
        let (ok, _, stderr) = generate(
            prompt,
            &[
                (TIER_NODES, addr.to_string()),
                (CARD_BUDGET, budget.to_string()),
            ],
            false,
        )?;
        tier::sc("node family: a node that serves another family refuses the open by name")?;
        pass &= line(
            !ok && stderr.contains("a tier of the family \"kimi-k3\"")
                && stderr.contains("this server opens"),
            "node family",
            &format!(
                "generate_kimi_k3 against a node of another family: ok={ok}, stderr {stderr:?}"
            ),
        );

        let (ok, _, stderr) = generate(prompt, &[(CARD_BUDGET, budget.to_string())], true)?;
        let plan = stderr.lines().find(|l| l.starts_with("plan "));
        tier::sc("plan records: --plan --place bp prints the plan record of the placed machine")?;
        pass &= line(
            ok && plan.is_some_and(|l| l.contains("place=bp") && l.contains("host_experts=0")),
            "plan records",
            &format!("generate_kimi_k3 --plan --place bp: ok={ok}, record {plan:?}"),
        );
        Ok(pass)
    }
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_kimi_k3_place: built without the `gpu` feature");
    std::process::exit(2);
}
