//! GPU gate for the V4.1 expert tier (`bloomery_gpu::host::tier`), on one
//! card: the loopback. The stage and the tier both run on the gate card
//! (`workstation::plan_gate`), two `Gpu`s on it. The gate plans with a hot
//! list (`BLOOMERY_HOT_LIST`, refused unset), so each routed layer's card
//! list is its hottest `n_l`. The tiered plan moves the coldest [`K3`] of
//! them, ranks `n_l - K3 .. n_l`, from card 0 to device 1 (the tier) through
//! the placement's own split (`placement::routed_row`): the (b′) shape, the
//! stage the hottest ranks and the tier the next ones, on the gate machine
//! with (b′)'s tier card and its prompt-batch reserves beside the stage card
//! (`tier_machine`). The reference is the gate plan itself, whose card holds
//! the union — the same experts on one card. Loads run one after the other,
//! the reference first.
//!
//! - `--union` (T1): the prose prompt [`PROMPT`] fed one decode step per
//!   position, then [`STEPS`] greedy steps, then [`PAIRS`] pair passes, all
//!   in graph mode, then the prompt and four steps eagerly: every position's
//!   argmax and logits bit for bit the reference's. The tiered stage graph
//!   has the reference's node count. Precondition: every layer the tier
//!   holds experts of was sent at least one routed slot over the run (the
//!   host's per-layer tier hits), or the run proves nothing and fails by
//!   name.
//! - `--fault` (T2): a fault raised on the tier card's word at the first
//!   tier layer before a step is that step's error, by its layer and site,
//!   and poisons the model; a reset lifts it and the next step is the
//!   reference's.
//! - `--lost` (T3): the tier's stream held behind a host flag before a
//!   step: within the go deadline and its grace the step fails naming the
//!   lost card, the host tier is poisoned as a lost card, the next step is
//!   refused by that poison (`GpuError::HostPoisoned`, a lost card) and a
//!   reset by name; the flag is then raised and both streams drain.
//! - `--two` (T7): a tiered plan with one expert on both devices is refused
//!   by name before anything is uploaded: as the load's check before the
//!   uploads (`body::TIER_MAP_BEFORE_UPLOAD`), not the one after them.
//! - `--batch` (B7): the prose prompt of [`BATCH_P`] positions fed as one
//!   prompt batch (`body::prefill`), then [`BATCH_STEPS`] greedy steps, bit
//!   for bit the same prompt fed one decode step per position on the tiered
//!   load: the call's last position and every greedy step, argmax and
//!   logits. No reference load.
//! - `--batch2` (B7, a group): [`BATCH_P2`] positions, two batches run as one
//!   group of [`BATCH_GROUP`], then the greedy steps, bit for bit the
//!   reference's prompt call of the same ids.
//!
//!   Both: precondition, the prompt call itself sent every tier layer at
//!   least one routed slot (the tier's per-layer hits across the call), and
//!   `--batch2`'s call ran as a group of two, or the clause proves nothing
//!   and fails by name.
//! - `--bfault` (B2): a fault raised on the tier card's word at the first
//!   tier layer before a prompt call is the call's error, by its layer and
//!   site, and poisons the model; a reset lifts it and the call is then the
//!   same prompt fed one decode step per position. No reference load.
//! - `--blost` (B3): the tier's stream held behind a host flag before a
//!   prompt call: within the go deadline and its grace the call fails naming
//!   the lost card, the host tier is poisoned as a lost card, the next call
//!   is refused by that poison at its group's start (`HostTier::begin_group`,
//!   before it enqueues anything) and a reset by name; the flag is then
//!   raised and both streams drain. A lost tier stays lost, so with `--lost`
//!   too the tiered plan is loaded again for T3.
//! - `--bfirst` (B3f): B3 as the first prompt call of a fresh tiered load,
//!   before anything else runs on it: the call must still fail by name
//!   within the same bound, so nothing the call would make for itself (a
//!   module load, an allocation) waits on the held tier stream. The tiered
//!   plan is loaded again for any clause after it.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_deepseek41_tier: built without the `deepseek41` feature; see `just gate-gpu-ds41-tier`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_tier", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use std::time::Instant;

    use bloomery_gpu::host::batch::TierBatchStats;
    use bloomery_gpu::hybrid::{BEGIN_GROUP, Chain, PoisonKind};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::q4k_sel::QuantSel;
    use bloomery_gpu::{Fault, FaultSite, GpuError, HostFlags, Q8Act};
    use bloomery_gpu_deepseek41::body::{
        self, Body, BodyMeta, DECODE_INPUT, Deepseek41Model, OpenCfg, PrefillMode,
        TIER_MAP_BEFORE_UPLOAD, TierOpen,
    };
    use bloomery_gpu_gates::{GateError, checks_failed, data_dir};
    use bloomery_levers::{
        CARD_BUDGET, CARD_DONTNEED, ENGRAM_HELPER, HOST_LOCK, HOST_POPULATE, HOT_LIST, R8,
    };
    use cuda_core::DeviceBuffer;
    use gguf::Split;
    use model::arch::deepseek41::hparams::Hparams;
    use model::arch::deepseek41::place::PlanInputs;
    use model::arch::deepseek41::place::tier_batch;
    use model::placement::{
        self, Device, ExpertList, HotList, Machine, Plan, Role, Row, workstation,
    };

    const NAME: &str = "gate_deepseek41_tier";
    /// Experts per routed layer the tier takes from the cold end of the gate
    /// plan's card list: the (b′) plan's tier depth, 873 tier experts over
    /// its 38 hybrid layers ≈ 23 a layer [derived]. A carve from the
    /// gate plan's own card list costs no card bytes, so the 3090's free
    /// bytes after the load bound nothing here.
    const K3: usize = 23;
    /// Prompt positions, fed one decode step each.
    const PROMPT: usize = 16;
    /// Greedy steps after the prompt.
    const STEPS: usize = 32;
    /// Pair passes after the prompt.
    const PAIRS: usize = 4;
    /// Steps the eager arm runs after the prompt.
    const EAGER_STEPS: usize = 4;
    /// Positions of the prompt call `--batch` feeds against the steps: one
    /// batch.
    const BATCH_P: usize = 512;
    /// Positions of the prompt call `--batch2` feeds against the reference's:
    /// two batches, one group of two under the default group lever.
    const BATCH_P2: usize = 1024;
    /// The batches a group holds that `--batch2`'s call must run as.
    const BATCH_GROUP: usize = 2;
    /// Greedy steps after a prompt call.
    const BATCH_STEPS: usize = 4;
    /// Positions of the prompt calls B2 and B3 feed: one batch.
    const BATCH_FAULT_P: usize = 64;
    /// The device index the tier's segments sit on in the tiered plan.
    const TIER_DEVICE: usize = 1;
    /// The go deadline and the grace a lost card is named within, plus room.
    const LOST_BOUND_S: f64 = 25.0;

    struct Args {
        union: bool,
        fault: bool,
        lost: bool,
        two: bool,
        batch: bool,
        batch2: bool,
        bfault: bool,
        blost: bool,
        bfirst: bool,
    }

    impl Args {
        /// Whether a clause other than T7 and B3f runs on the tiered load.
        fn on_load(&self) -> bool {
            self.union
                || self.fault
                || self.lost
                || self.batch
                || self.batch2
                || self.bfault
                || self.blost
        }
    }

    fn parse_args() -> Result<Args, GateError> {
        const USAGE: &str = "usage: gate_deepseek41_tier [--union] [--fault] [--lost] [--two] \
                             [--batch] [--batch2] [--bfault] [--blost] [--bfirst]";
        let mut a = Args {
            union: false,
            fault: false,
            lost: false,
            two: false,
            batch: false,
            batch2: false,
            bfault: false,
            blost: false,
            bfirst: false,
        };
        for arg in std::env::args().skip(1) {
            match arg.as_str() {
                "--union" => a.union = true,
                "--fault" => a.fault = true,
                "--lost" => a.lost = true,
                "--two" => a.two = true,
                "--batch" => a.batch = true,
                "--batch2" => a.batch2 = true,
                "--bfault" => a.bfault = true,
                "--blost" => a.blost = true,
                "--bfirst" => a.bfirst = true,
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        if !(a.on_load() || a.two || a.bfirst) {
            return Err(USAGE.into());
        }
        Ok(a)
    }

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

    /// The gate plan's machine `machine` with the (b′) tier card beside its
    /// one stage card, at index [`TIER_DEVICE`]: the tier card and the host
    /// carry the tier's prompt-batch reserves for the file of `hp`
    /// (`workstation::plan_bp`'s), which a tiered load checks its
    /// allocations against. The tier card is the gate card by name: the
    /// stage and the tier share it.
    fn tier_machine(machine: &Machine, hp: &Hparams) -> Result<Machine, GateError> {
        if machine.cards.len() != TIER_DEVICE || !machine.tiers.is_empty() {
            return Err(format!(
                "the gate machine has {} stage cards and {} tiers; the tier goes at index \
                 {TIER_DEVICE}, after one stage card",
                machine.cards.len(),
                machine.tiers.len()
            )
            .into());
        }
        let bp = workstation::plan_bp(hp.n_layer, None, tier_batch(hp));
        if bp
            .tiers
            .iter()
            .map(|t| t.name.as_str())
            .ne([machine.cards[0].name.as_str()])
        {
            return Err(format!(
                "plan (b′)'s tier cards {:?} are not the gate card {}",
                bp.tiers.iter().map(|t| &t.name).collect::<Vec<_>>(),
                machine.cards[0].name
            )
            .into());
        }
        let mut out = machine.clone();
        out.tiers = bp.tiers;
        out.host = bp.host;
        Ok(out)
    }

    /// Whether `r` is the host tier's refusal ([`GpuError::HostPoisoned`])
    /// for the poison of a lost card, as `what`: the entry that refuses it
    /// before it enqueues anything.
    fn lost_refusal<T>(r: &Result<T, GpuError>, what: &str) -> bool {
        matches!(
            r,
            Err(GpuError::HostPoisoned { what: w, poison })
                if *w == what
                    && poison.mark().kind == PoisonKind::CardLost
        )
    }

    /// `plan` on `machine` (the gate plan's with the tier card,
    /// [`tier_machine`]) with the coldest `k3` experts of each routed stack's
    /// card-0 list by `hot`'s ranks moved to device [`TIER_DEVICE`]; a stack
    /// whose card list holds `k3` or fewer is refused by name. With `twice`,
    /// the first stack's tier list also keeps its first expert on card 0 (an
    /// expert on two devices). The host segments stay as they are, so the
    /// host set is the plan's.
    fn tiered<'a>(
        plan: &Plan<'a>,
        machine: &'a Machine,
        hot: &HotList,
        k3: usize,
        twice: bool,
    ) -> Result<Plan<'a>, GateError> {
        let model = plan.model;
        let mut out = plan.clone();
        out.machine = machine;
        let mut doubled = false;
        for row in &mut out.rows {
            let t = &model.tensors[row.tensor];
            if t.role != Role::RoutedExperts {
                continue;
            }
            let Some(card) = row.segments.iter().find(|s| s.device == Device::Card(0)) else {
                continue;
            };
            let layer = t
                .layer
                .ok_or_else(|| format!("{}: a routed stack without a layer", t.name))?;
            let mut ids = card
                .experts
                .as_ref()
                .ok_or_else(|| format!("{}: a card segment without experts", t.name))?
                .ids()
                .to_vec();
            let ranked = hot.ranked(layer);
            let rank = |e: u32| ranked.iter().position(|&r| r == e);
            if let Some(&e) = ids.iter().find(|&&e| rank(e).is_none()) {
                return Err(
                    format!("{}: card expert {e} has no rank in {}", t.name, hot.path()).into(),
                );
            }
            ids.sort_by_key(|&e| rank(e));
            if ids.len() <= k3 {
                return Err(format!(
                    "{}: the card keeps {} experts, the gate moves {k3} to the tier",
                    t.name,
                    ids.len()
                )
                .into());
            }
            let cut = ids.len() - k3;
            let mut stage_ids = ids[..cut].to_vec();
            if twice && !doubled {
                stage_ids.push(ids[cut]);
            }
            let stage = placement::routed_row(
                row.tensor,
                t,
                0,
                ExpertList::new(stage_ids, model.experts)?,
                model,
            )?;
            let tier = placement::routed_row(
                row.tensor,
                t,
                TIER_DEVICE,
                ExpertList::new(ids[cut..].to_vec(), model.experts)?,
                model,
            )?;
            let host = row
                .segments
                .iter()
                .filter(|s| s.device != Device::Card(0))
                .cloned();
            let segments = stage
                .segments
                .into_iter()
                .filter(|s| s.device == Device::Card(0))
                .chain(
                    tier.segments
                        .into_iter()
                        .filter(|s| s.device == Device::Card(TIER_DEVICE)),
                )
                .chain(host)
                .collect();
            *row = Row {
                segments,
                ..row.clone()
            };
            if let Some(n) = out.n_l.get_mut(layer) {
                *n = cut as u64;
            }
            doubled = true;
        }
        Ok(out)
    }

    /// Every argmax and logits vector a run left, in order.
    #[derive(Default)]
    struct Run {
        tokens: Vec<u32>,
        logits: Vec<Vec<f32>>,
        stage_nodes: Option<usize>,
    }

    impl Run {
        fn push(&mut self, token: u32, logits: Vec<f32>) {
            self.tokens.push(token);
            self.logits.push(logits);
        }
    }

    /// The T1 run on `m`: the prompt one step a position, the greedy steps,
    /// the pair passes, all in graph mode from a reset; then the prompt and
    /// [`EAGER_STEPS`] steps eagerly from a reset.
    fn union_run(m: &mut Deepseek41Model, prompt: &[u32]) -> Result<Run, GateError> {
        let mut run = Run::default();
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let mut tok = 0;
        for &id in prompt {
            tok = m.step(&[id])?;
            run.push(tok, m.logits()?);
        }
        run.stage_nodes = Some(m.step_graph_nodes()?.len());
        for _ in 0..STEPS {
            tok = m.step(&[tok])?;
            run.push(tok, m.logits()?);
        }
        for k in 0..PAIRS {
            let second = prompt[k % prompt.len()];
            let [a, b] = m.step_rows([tok, second])?;
            let [la, lb] = m.rows_logits::<2>()?;
            run.push(a, la);
            run.push(b, lb);
            tok = b;
        }
        m.set_mode(StepMode::Eager);
        m.reset()?;
        for &id in prompt {
            tok = m.step(&[id])?;
            run.push(tok, m.logits()?);
        }
        for _ in 0..EAGER_STEPS {
            tok = m.step(&[tok])?;
            run.push(tok, m.logits()?);
        }
        m.set_mode(StepMode::Graph);
        Ok(run)
    }

    /// The reference load's runs each clause compares with.
    #[derive(Default)]
    struct Want {
        union: Option<Run>,
        batch2: Option<Run>,
    }

    /// What one prompt call sent the tier: its hits per layer from the
    /// tier's first layer, and the batch counters it moved.
    struct Sent {
        first: usize,
        hits: Vec<u64>,
        stats: TierBatchStats,
    }

    /// The tier's per-layer hits and batch counters since load; `None`
    /// without a tier.
    fn tier_counts(m: &Deepseek41Model) -> Result<Option<Sent>, GateError> {
        let hybrid = m.body(NAME)?.hybrid();
        Ok(hybrid.tier().map(|t| Sent {
            first: t.set().layers().start,
            hits: t.stats().layer_hits,
            stats: hybrid.tier_batch_stats(),
        }))
    }

    /// From a reset in graph mode: `ids` fed as one prompt call, then
    /// [`BATCH_STEPS`] greedy steps; the call's last position and each step,
    /// and on a tiered load what the call alone sent the tier.
    fn batch_run(m: &mut Deepseek41Model, ids: &[u32]) -> Result<(Run, Option<Sent>), GateError> {
        let mut run = Run::default();
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let before = tier_counts(m)?;
        let mut tok = body::prefill(m, ids)?;
        let sent = match (before, tier_counts(m)?) {
            (Some(b), Some(a)) => Some(Sent {
                first: a.first,
                hits: a.hits.iter().zip(&b.hits).map(|(a, b)| a - b).collect(),
                stats: TierBatchStats {
                    served: a.stats.served - b.stats.served,
                    settles: a.stats.settles - b.stats.settles,
                    settle_early: a.stats.settle_early - b.stats.settle_early,
                    settle_ns: a.stats.settle_ns - b.stats.settle_ns,
                },
            }),
            _ => None,
        };
        run.push(tok, m.logits()?);
        for _ in 0..BATCH_STEPS {
            tok = m.step(&[tok])?;
            run.push(tok, m.logits()?);
        }
        Ok((run, sent))
    }

    /// The precondition of a prompt-call clause `what`: the call sent every
    /// layer of `tier_layers` at least one routed slot; prints its counters.
    fn sent_every(what: &str, sent: Option<&Sent>, tier_layers: &[usize]) -> bool {
        let Some(sent) = sent else {
            println!("FAIL {what} precondition: the tiered load holds no tier card");
            return false;
        };
        let hit = |l: usize| {
            l.checked_sub(sent.first)
                .and_then(|i| sent.hits.get(i))
                .copied()
                .unwrap_or(0)
        };
        let missing: Vec<usize> = tier_layers
            .iter()
            .copied()
            .filter(|&l| hit(l) == 0)
            .collect();
        let slots: u64 = tier_layers.iter().map(|&l| hit(l)).sum();
        let st = sent.stats;
        println!(
            "{what} tier: {} services, {} settles ({} early, {:.2} ms waited), {slots} tier slots \
             over {} layers",
            st.served,
            st.settles,
            st.settle_early,
            st.settle_ns as f64 / 1e6,
            tier_layers.len()
        );
        if missing.is_empty() && st.served > 0 {
            println!("ok {what} precondition: the prompt call sent every tier layer a routed slot");
            true
        } else {
            println!(
                "FAIL {what} precondition: tier layers {missing:?} were sent no routed slot by the \
                 prompt call ({} services); the clause proves nothing there",
                st.served
            );
            false
        }
    }

    /// [`batch_run`] with `ids` fed one decode step per position.
    fn steps_run(m: &mut Deepseek41Model, ids: &[u32]) -> Result<Run, GateError> {
        let mut run = Run::default();
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let mut tok = 0;
        for &id in ids {
            tok = m.step(&[id])?;
        }
        run.push(tok, m.logits()?);
        for _ in 0..BATCH_STEPS {
            tok = m.step(&[tok])?;
            run.push(tok, m.logits()?);
        }
        Ok(run)
    }

    /// Whether `got` is `want` bit for bit; prints the first difference.
    fn same(what: &str, want: &Run, got: &Run) -> bool {
        if want.tokens.len() != got.tokens.len() {
            println!(
                "FAIL {what}: {} positions, the reference ran {}",
                got.tokens.len(),
                want.tokens.len()
            );
            return false;
        }
        for (i, ((wt, wl), (gt, gl))) in want
            .tokens
            .iter()
            .zip(&want.logits)
            .zip(got.tokens.iter().zip(&got.logits))
            .enumerate()
        {
            let bits =
                wl.len() == gl.len() && wl.iter().zip(gl).all(|(a, b)| a.to_bits() == b.to_bits());
            if wt != gt || !bits {
                let first = wl
                    .iter()
                    .zip(gl)
                    .position(|(a, b)| a.to_bits() != b.to_bits());
                println!(
                    "FAIL {what}: position {i}: token {gt} (reference {wt}); first logit apart {first:?}"
                );
                return false;
            }
        }
        println!(
            "ok {what}: {} positions, tokens and logits bit for bit the reference's",
            got.tokens.len()
        );
        true
    }

    fn open(
        file: Split,
        plan: &Plan<'_>,
        tier: Option<TierOpen>,
        meta: &BodyMeta,
    ) -> Result<Deepseek41Model, GpuError> {
        Body::open_placed_tiered(file, plan, 0, tier, meta)
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[
            ENGRAM_HELPER,
            HOT_LIST,
            CARD_BUDGET,
            HOST_POPULATE,
            HOST_LOCK,
            CARD_DONTNEED,
            R8,
        ])?;
        let args = parse_args()?;
        let mut cfg = OpenCfg::from_levers(&levers)?;
        // The batch clauses run the prompt call, so the tiered load makes its
        // buffers (`Body::open_placed_tiered`).
        cfg.body.prefill = PrefillMode::Batch;
        let path = workstation::model_v41();
        let split = || Split::open(&path).map_err(|e| format!("open {path}: {e}"));
        let inputs = PlanInputs::read(&split()?)?;
        let machine = workstation::plan_gate(inputs.model.layers);
        let plan = inputs.plan(&machine, workstation::CTX_MAX, &cfg.place)?;
        let meta = BodyMeta {
            hp: inputs.hp.clone(),
            levers: cfg.body,
        };
        let hot = cfg
            .place
            .hot
            .as_ref()
            .ok_or("the gate plans with a hot list: set BLOOMERY_HOT_LIST")?;
        let card = machine.cards[0].name.clone();
        let tier_open = || TierOpen {
            card: TIER_DEVICE,
            name: card.clone(),
        };
        let tmachine = tier_machine(&machine, &inputs.hp)?;
        let tplan = tiered(&plan, &tmachine, hot, K3, false)?;
        let tier_layers: Vec<usize> = (0..inputs.model.layers)
            .filter(|&l| tplan.n_l.get(l) != plan.n_l.get(l))
            .collect();
        println!(
            "plan: gate card {card}, {} experts on it by {}; tier: the coldest {K3} of each routed \
             layer's list on {} layers ({} experts), stage keeps the rest",
            plan.cards[0].experts,
            hot.path(),
            tier_layers.len(),
            K3 * tier_layers.len()
        );
        let mut pass = true;

        if args.two {
            let bad = tiered(&plan, &tmachine, hot, K3, true)?;
            let t0 = Instant::now();
            match open(split()?, &bad, Some(tier_open()), &meta) {
                Err(e @ GpuError::Plan { what, .. })
                    if what == TIER_MAP_BEFORE_UPLOAD && e.to_string().contains("placed twice") =>
                {
                    println!(
                        "ok T7: an expert on two devices refused before any upload in {:.2} s: {e}",
                        t0.elapsed().as_secs_f64()
                    );
                }
                Err(e) => {
                    println!(
                        "FAIL T7: refused, but not as {TIER_MAP_BEFORE_UPLOAD} naming an expert \
                         placed twice: {e}"
                    );
                    pass = false;
                }
                Ok(_) => {
                    println!("FAIL T7: a plan with an expert on two devices loaded");
                    pass = false;
                }
            }
        }

        let prompt = prose(PROMPT)?;
        let long = if args.batch || args.batch2 || args.bfault || args.blost || args.bfirst {
            prose(BATCH_P2)?
        } else {
            Vec::new()
        };
        let mut want = Want::default();
        if args.union || args.fault || args.batch2 {
            let t0 = Instant::now();
            let mut m = open(split()?, &plan, None, &meta)?;
            println!(
                "reference load (gate plan, one card) in {:.1} s",
                t0.elapsed().as_secs_f64()
            );
            if args.union || args.fault {
                want.union = Some(union_run(&mut m, &prompt)?);
            }
            if args.batch2 {
                body::prepare_prefill(&mut m)?;
                want.batch2 = Some(batch_run(&mut m, &long)?.0);
            }
            drop(m);
        }

        let load = || -> Result<Deepseek41Model, GateError> {
            let t0 = Instant::now();
            let m = open(split()?, &tplan, Some(tier_open()), &meta)?;
            let (free, total) = m.gpu().mem_info()?;
            println!(
                "tiered load (stage + tier on {card}) in {:.1} s; card free {free} of {total} B \
                 after it",
                t0.elapsed().as_secs_f64()
            );
            Ok(m)
        };
        let mut m = load()?;
        if args.bfirst {
            pass &= batch_lost_case(&mut m, &long[..BATCH_FAULT_P], "B3f")?;
            if args.on_load() {
                drop(m);
                m = load()?;
            }
        }

        if let Some(want) = &want.union {
            if args.union {
                let got = union_run(&mut m, &prompt)?;
                pass &= same("T1 (step, 32 steps, pairs, eager)", want, &got);
                if got.stage_nodes != want.stage_nodes {
                    println!(
                        "FAIL T1 structure: stage graph of {:?} nodes, the reference's {:?}",
                        got.stage_nodes, want.stage_nodes
                    );
                    pass = false;
                } else {
                    println!(
                        "ok T1 structure: stage graph of {:?} nodes, the reference's",
                        got.stage_nodes
                    );
                }
                let body = m.body(NAME)?;
                let tier = body
                    .hybrid()
                    .tier()
                    .ok_or("the tiered load holds no tier card")?;
                let st = tier.stats();
                let first = tier.set().layers().start;
                let missing: Vec<usize> = tier_layers
                    .iter()
                    .copied()
                    .filter(|&l| st.layer_hits.get(l - first).copied().unwrap_or(0) == 0)
                    .collect();
                let hits: u64 = st.layer_hits.iter().sum();
                println!(
                    "tier: {} layers asked, {} settles ({} early), hits {hits} over {} layers; graph \
                     nodes step {:?} pair {:?}",
                    st.issued,
                    st.settles,
                    st.settle_early,
                    tier_layers.len(),
                    tier.graph_nodes(Chain::Step),
                    tier.graph_nodes(Chain::Pair)
                );
                if missing.is_empty() {
                    println!("ok precondition: every tier layer was sent a routed slot");
                } else {
                    println!(
                        "FAIL precondition: tier layers {missing:?} were sent no routed slot; T1 proves \
                         nothing there"
                    );
                    pass = false;
                }
            }
            if args.fault {
                let first = *tier_layers
                    .first()
                    .ok_or("the tiered plan has no tier layer")?;
                pass &= fault_case(&mut m, &prompt, want, first)?;
            }
        }
        if args.batch {
            let want = steps_run(&mut m, &long[..BATCH_P])?;
            let (got, sent) = batch_run(&mut m, &long[..BATCH_P])?;
            pass &= same(
                "B7 (one batch against the steps, then greedy steps)",
                &want,
                &got,
            );
            pass &= sent_every("B7", sent.as_ref(), &tier_layers);
        }
        if let Some(want) = &want.batch2 {
            let (got, sent) = batch_run(&mut m, &long)?;
            pass &= same(
                "B7 group (two batches against the reference's call, then greedy steps)",
                want,
                &got,
            );
            pass &= sent_every("B7 group", sent.as_ref(), &tier_layers);
            let group = m.body(NAME)?.prefill_group().map(|(g, _)| g);
            if group == Some(BATCH_GROUP) {
                println!("ok B7 group structure: the call ran as a group of {BATCH_GROUP}");
            } else {
                println!(
                    "FAIL B7 group structure: the prompt group is {group:?}, want {BATCH_GROUP}: \
                     the held routing across a group is not exercised"
                );
                pass = false;
            }
        }
        if args.bfault {
            let first = *tier_layers
                .first()
                .ok_or("the tiered plan has no tier layer")?;
            pass &= batch_fault_case(&mut m, &long[..BATCH_FAULT_P], first)?;
        }
        if args.blost {
            pass &= batch_lost_case(&mut m, &long[..BATCH_FAULT_P], "B3")?;
            if args.lost {
                drop(m);
                m = load()?;
            }
        }
        if args.lost {
            pass &= lost_case(&mut m, &prompt)?;
        }
        drop(m);
        if !pass {
            return Err(checks_failed());
        }
        println!("PASSED: {NAME}");
        Ok(())
    }

    /// T2: a fault on the tier card's word at `layer` before a step.
    fn fault_case(
        m: &mut Deepseek41Model,
        prompt: &[u32],
        want: &Run,
        layer: usize,
    ) -> Result<bool, GateError> {
        m.set_mode(StepMode::Graph);
        m.reset()?;
        for &id in &prompt[..prompt.len() - 1] {
            m.step(&[id])?;
        }
        let want_fault = raise_tier_fault(m, layer)?;
        let r = m.step(&[prompt[prompt.len() - 1]]);
        let mut ok = match &r {
            Err(GpuError::Fault { fault, .. }) if *fault == want_fault => {
                println!("ok T2: the step fails with the tier's fault: {fault}");
                true
            }
            other => {
                println!("FAIL T2: the step gave {other:?}, want the tier's fault {want_fault}");
                false
            }
        };
        if m.poisoned() != Some(want_fault) {
            println!("FAIL T2: the model's poison is {:?}", m.poisoned());
            ok = false;
        }
        m.reset()?;
        let mut tok = 0;
        for &id in prompt {
            tok = m.step(&[id])?;
        }
        let (i, logits) = (prompt.len() - 1, m.logits()?);
        let bits = logits
            .iter()
            .zip(&want.logits[i])
            .all(|(a, b)| a.to_bits() == b.to_bits());
        if tok == want.tokens[i] && bits {
            println!("ok T2: after the reset the prompt's last position is the reference's");
        } else {
            println!(
                "FAIL T2: after the reset token {tok} (reference {})",
                want.tokens[i]
            );
            ok = false;
        }
        Ok(ok)
    }

    /// Raise a fault on the tier card's word at `layer`, as a quantizer that
    /// met a NaN there; returns the fault a read of the word gives.
    fn raise_tier_fault(m: &mut Deepseek41Model, layer: usize) -> Result<Fault, GateError> {
        let body = m.body(NAME)?;
        let tier = body.hybrid().tier().ok_or("no tier card")?;
        let g = tier.gpu();
        let s = g.stream();
        let x = DeviceBuffer::from_host(s, &[f32::NAN; 256])?;
        let sel = DeviceBuffer::from_host(s, &[0u32])?;
        let mut act = Q8Act::with_k(s, 1, 256)?;
        let q = QuantSel {
            x: &x,
            cols: 0..1,
            sel: &sel,
            n_card: 1,
        };
        g.q4k_sel()
            .enqueue_quantize_sel(s, &q, g.layer_sink(layer)?, &mut act)?;
        s.synchronize()?;
        m.gpu().context().bind_to_thread()?;
        Ok(Fault::at(u32::try_from(layer)?, FaultSite::QuantColumn))
    }

    /// B2: a fault on the tier card's word at `layer` before a prompt call.
    fn batch_fault_case(
        m: &mut Deepseek41Model,
        ids: &[u32],
        layer: usize,
    ) -> Result<bool, GateError> {
        let want = steps_run(m, ids)?;
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let want_fault = raise_tier_fault(m, layer)?;
        let r = body::prefill(m, ids);
        let mut ok = match &r {
            Err(GpuError::Fault { fault, .. }) if *fault == want_fault => {
                println!("ok B2: the prompt call fails with the tier's fault: {fault}");
                true
            }
            other => {
                println!(
                    "FAIL B2: the prompt call gave {other:?}, want the tier's fault {want_fault}"
                );
                false
            }
        };
        if m.poisoned() != Some(want_fault) {
            println!("FAIL B2: the model's poison is {:?}", m.poisoned());
            ok = false;
        }
        let (got, _) = batch_run(m, ids)?;
        ok &= same(
            "B2 (after the reset, the prompt call against the steps)",
            &want,
            &got,
        );
        Ok(ok)
    }

    /// B3 (B3f on a fresh load, clause `what`): the tier's stream held behind
    /// a host flag before a prompt call.
    fn batch_lost_case(
        m: &mut Deepseek41Model,
        ids: &[u32],
        what: &str,
    ) -> Result<bool, GateError> {
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let flags = {
            let body = m.body(NAME)?;
            let tier = body.hybrid().tier().ok_or("no tier card")?;
            let flags = HostFlags::new(tier.gpu().context(), 1)?;
            flags.enqueue_wait(tier.gpu().stream(), 0)?;
            flags
        };
        m.gpu().context().bind_to_thread()?;
        let t0 = Instant::now();
        let r = body::prefill(m, ids);
        let secs = t0.elapsed().as_secs_f64();
        let mut ok = match &r {
            Err(e) if e.to_string().contains("the card is lost") && secs < LOST_BOUND_S => {
                println!(
                    "ok {what}: the prompt call fails in {secs:.1} s naming the lost card: {e}"
                );
                true
            }
            other => {
                println!(
                    "FAIL {what}: after {secs:.1} s the prompt call gave {other:?}, want the lost card"
                );
                false
            }
        };
        let kind = m.body(NAME)?.hybrid().last_poison().map(|p| p.mark().kind);
        if kind != Some(PoisonKind::CardLost) {
            println!("FAIL {what}: the host tier's poison is {kind:?}, want CardLost");
            ok = false;
        }
        let next = body::prefill(m, ids);
        if lost_refusal(&next, BEGIN_GROUP) {
            println!(
                "ok {what}: the next prompt call is refused at its group's start by the lost card's \
                 poison: {}",
                next.as_ref()
                    .err()
                    .map_or_else(String::new, ToString::to_string)
            );
        } else {
            println!(
                "FAIL {what}: the next prompt call gave {next:?}, want the host tier's refusal as \
                 {BEGIN_GROUP} naming the lost card"
            );
            ok = false;
        }
        match m.reset() {
            Err(e) if e.to_string().contains("lost") => {
                println!("ok {what}: a reset is refused: {e}")
            }
            other => {
                println!(
                    "FAIL {what}: a reset gave {other:?}, want a refusal naming the lost card"
                );
                ok = false;
            }
        }
        flags.raise(0)?;
        let body = m.body(NAME)?;
        body.hybrid()
            .tier()
            .ok_or("no tier card")?
            .gpu()
            .stream()
            .synchronize()?;
        m.gpu().stream().synchronize()?;
        println!("{what}: flag raised, both streams drained");
        Ok(ok)
    }

    /// T3: the tier's stream held behind a host flag before a step.
    fn lost_case(m: &mut Deepseek41Model, prompt: &[u32]) -> Result<bool, GateError> {
        m.set_mode(StepMode::Graph);
        m.reset()?;
        m.step(&prompt[..4])?;
        let flags = {
            let body = m.body(NAME)?;
            let tier = body.hybrid().tier().ok_or("no tier card")?;
            let flags = HostFlags::new(tier.gpu().context(), 1)?;
            flags.enqueue_wait(tier.gpu().stream(), 0)?;
            flags
        };
        let t0 = Instant::now();
        let r = m.step(&[prompt[4]]);
        let secs = t0.elapsed().as_secs_f64();
        let mut ok = match &r {
            Err(e) if e.to_string().contains("the card is lost") && secs < LOST_BOUND_S => {
                println!("ok T3: the step fails in {secs:.1} s naming the lost card: {e}");
                true
            }
            other => {
                println!("FAIL T3: after {secs:.1} s the step gave {other:?}, want the lost card");
                false
            }
        };
        let kind = m.body(NAME)?.hybrid().last_poison().map(|p| p.mark().kind);
        if kind != Some(PoisonKind::CardLost) {
            println!("FAIL T3: the host tier's poison is {kind:?}, want CardLost");
            ok = false;
        }
        let next = m.step(&[prompt[5]]);
        if lost_refusal(&next, DECODE_INPUT) {
            println!(
                "ok T3: the next step is refused by the lost card's poison: {}",
                next.as_ref()
                    .err()
                    .map_or_else(String::new, ToString::to_string)
            );
        } else {
            println!(
                "FAIL T3: the next step gave {next:?}, want the host tier's refusal as \
                 {DECODE_INPUT} naming the lost card"
            );
            ok = false;
        }
        match m.reset() {
            Err(e) if e.to_string().contains("lost") => println!("ok T3: a reset is refused: {e}"),
            other => {
                println!("FAIL T3: a reset gave {other:?}, want a refusal naming the lost card");
                ok = false;
            }
        }
        flags.raise(0)?;
        let body = m.body(NAME)?;
        body.hybrid()
            .tier()
            .ok_or("no tier card")?
            .gpu()
            .stream()
            .synchronize()?;
        m.gpu().stream().synchronize()?;
        println!("T3: flag raised, both streams drained");
        Ok(ok)
    }
}
