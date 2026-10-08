//! GPU gate for the V4.1 expert tier (`bloomery_gpu::host::tier`), on one
//! card: the loopback. The stage and the tier both run on the gate card
//! (`crate::gate_card::plan_gate`: the 3090's bytes in the real tier, the largest visible card
//! under the header's card budget in the fixture tier), two `Gpu`s on it. Each routed layer's card
//! list is its id prefix `[0, n_l)`. The tiered plan moves the last `k3` ([`tier_depth`]) of
//! them, ids `n_l - k3 .. n_l`, from card 0 to device 1 (the tier) through
//! the placement's own split (`placement::routed_row`): the (b′) shape, the
//! stage the first ids and the tier the next ones, on the gate machine
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
//! - `--bfeat` (B8): on a load of its own, the DSpark draft's feature tap
//!   (`body::attach_features`, the layers of `$BLOOMERY_DSPARK_MODEL`'s
//!   `target_layers`, header only: no draft is loaded) attached after the
//!   tiered load made the prompt batch's buffers, then `body::prepare_prefill`
//!   again — the order `--place bp` opens a draft in (`app::Loaded::open`,
//!   `CardDraft::open`, `Loaded::ready`). Then the [`BATCH_P2`] prose ids fed
//!   one decode step each, each position's features read after its step
//!   (`Body::read_features`); against them, a prompt call of the same ids
//!   (`body::prefill_with`) handing over its last `window` positions' rows,
//!   for the draft's `attention.sliding_window` and for every position: the
//!   positions and every row bit for bit. Preconditions: the calls ran two
//!   batches as one group of [`BATCH_GROUP`], the wide call handed rows over
//!   from both batches, and it sent every tier layer a routed slot.
//!
//! Every clause is self-consistency (the tiered load against the reference load or against the
//! same file fed another way), so the fixture tier runs all of them; each names its tag once
//! through `crate::ds41_tier::sc` before it runs, and the closing line counts them.

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
#[path = "shared/ds41_dspark.rs"]
#[allow(
    dead_code,
    reason = "B8 reads the draft's header only; the loop half serves generate_ds41"
)]
mod dspark;

#[cfg(feature = "deepseek41")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_tier.rs"]
#[allow(
    dead_code,
    reason = "the gate reads the path and the clause tags; the triangle's facts serve the prefill gate"
)]
mod ds41_tier;

#[cfg(feature = "deepseek41")]
mod gate {
    use std::time::Instant;

    use bloomery_gpu::host::batch::TierBatchStats;
    use bloomery_gpu::hybrid::{BEGIN_GROUP, Chain, PoisonKind};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::q4k_sel::QuantSel;
    use bloomery_gpu::{Fault, FaultSite, GpuError, HostFlags, Q8Act};
    use bloomery_gpu_deepseek41::body::{
        self, Body, BodyMeta, DECODE_INPUT, Deepseek41Model, FeatureRows, OpenCfg, PrefillMode,
        TIER_MAP_BEFORE_UPLOAD, TierOpen,
    };
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::{GateError, checks_failed, prose_ids};
    use bloomery_levers::{
        CARD_BUDGET, CARD_DONTNEED, ENGRAM_HELPER, HOST_LOCK, HOST_POPULATE, R8,
    };
    use cuda_core::DeviceBuffer;
    use gguf::Split;
    use model::arch::deepseek41::hparams::Hparams;
    use model::arch::deepseek41::place::PlanInputs;
    use model::arch::deepseek41::place::tier_batch;
    use model::placement::{self, Device, ExpertList, Machine, Plan, Role, Row, workstation};

    use crate::dspark;

    const NAME: &str = "gate_deepseek41_tier";
    /// The real file's tier depth, as the literal this gate carried before [`tier_depth`] derived
    /// it: the (b′) plan's 881 tier experts over its 38 hybrid layers are 23.2 a layer, which
    /// rounds to 23. The real tier prints the derived depth beside it and requires them equal.
    const REAL_K3: usize = 23;
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
        bfeat: bool,
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
                             [--batch] [--batch2] [--bfault] [--blost] [--bfirst] [--bfeat]";
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
            bfeat: false,
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
                "--bfeat" => a.bfeat = true,
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        if !(a.on_load() || a.two || a.bfirst || a.bfeat) {
            return Err(USAGE.into());
        }
        Ok(a)
    }

    /// The gate plan's machine `machine` with the (b′) tier card beside its
    /// one stage card, at index [`TIER_DEVICE`]: the tier card and the host
    /// carry the tier's prompt-batch reserves for the file of `hp`
    /// (`workstation::plan_bp`'s), which a tiered load checks its
    /// allocations against. (b′)'s tier is the 3090, whose bytes the gate
    /// card plans; it goes on the gate card by name, so the stage and the
    /// tier share it.
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
            .ne([workstation::RTX_3090.name])
        {
            return Err(format!(
                "plan (b′)'s tier cards {:?} are not the 3090, whose bytes the gate card {} plans",
                bp.tiers.iter().map(|t| &t.name).collect::<Vec<_>>(),
                machine.cards[0].name
            )
            .into());
        }
        let mut out = machine.clone();
        out.tiers = bp.tiers;
        for t in &mut out.tiers {
            t.name.clone_from(&machine.cards[0].name);
            t.device = machine.cards[0].device;
        }
        out.host = bp.host;
        Ok(out)
    }

    /// Experts per routed layer the tier takes from the cold end of the gate plan's card list: the
    /// (b′) plan's tier depth — the mean of its tier experts over the layers that have any, rounded
    /// down after a half added, read from `workstation::plan_bp` planned over the same file with
    /// the DSpark draft's reserve (`reserve`, from the draft's header) under `place`, the levers
    /// of that plan. A carve from the gate plan's own card list costs no card bytes, so the 3090's
    /// free bytes after the load bound nothing here; a depth the gate plan's lists cannot give is
    /// [`tiered`]'s named error. The real tier prints the derived depth beside [`REAL_K3`].
    fn tier_depth(
        inputs: &PlanInputs,
        place: &model::placement::PlanLevers,
        reserve: u64,
    ) -> Result<usize, GateError> {
        let bp = workstation::plan_bp(inputs.hp.n_layer, Some(reserve), tier_batch(&inputs.hp));
        let plan = inputs.plan(&bp, workstation::CTX_MAX, place)?;
        let per: Vec<u64> = plan
            .tier_n_l
            .first()
            .ok_or("plan (b′) has no tier card")?
            .iter()
            .copied()
            .filter(|&n| n > 0)
            .collect();
        let layers = per.len() as u64;
        if layers == 0 {
            return Err("plan (b′) puts no expert on its tier card".into());
        }
        let total: u64 = per.iter().sum();
        let depth = usize::try_from((total + layers / 2) / layers)?;
        println!(
            "tier depth: plan (b′) with the draft's reserve of {reserve} B puts {total} experts on \
             its tier over {layers} layers: {depth} a layer"
        );
        if !bloomery_gpu_gates::tier::witness("tier depth K3", depth, REAL_K3) {
            return Err(
                format!("the derived tier depth {depth} is not the literal {REAL_K3}").into(),
            );
        }
        Ok(depth)
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
    /// [`tier_machine`]) with the last `k3` experts of each routed stack's
    /// card-0 list, in id order, moved to device [`TIER_DEVICE`]; a stack
    /// whose card list holds `k3` or fewer is refused by name. With `twice`,
    /// the first stack's tier list also keeps its first expert on card 0 (an
    /// expert on two devices). The host segments stay as they are, so the
    /// host set is the plan's.
    fn tiered<'a>(
        plan: &Plan<'a>,
        machine: &'a Machine,
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
            let ids = card
                .experts
                .as_ref()
                .ok_or_else(|| format!("{}: a card segment without experts", t.name))?
                .ids()
                .to_vec();
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
        Ok(hybrid.tiers().first().map(|t| Sent {
            first: t.set().layers().start,
            hits: t.stats().layer_hits,
            stats: hybrid
                .tier_batch_stats()
                .first()
                .copied()
                .unwrap_or_default(),
        }))
    }

    /// The counters `after` less `before`: what the span between sent the
    /// tier; `None` without a tier.
    fn sent_between(before: Option<Sent>, after: Option<Sent>) -> Option<Sent> {
        let (b, a) = (before?, after?);
        Some(Sent {
            first: a.first,
            hits: a.hits.iter().zip(&b.hits).map(|(a, b)| a - b).collect(),
            stats: TierBatchStats {
                served: a.stats.served - b.stats.served,
                settles: a.stats.settles - b.stats.settles,
                settle_early: a.stats.settle_early - b.stats.settle_early,
                settle_ns: a.stats.settle_ns - b.stats.settle_ns,
            },
        })
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
        let sent = sent_between(before, tier_counts(m)?);
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
            CARD_BUDGET,
            HOST_POPULATE,
            HOST_LOCK,
            CARD_DONTNEED,
            R8,
        ])?;
        crate::gate_card::init()?;
        let args = parse_args()?;
        let mut cfg = OpenCfg::from_levers(&levers)?;
        // The batch clauses run the prompt call, so the tiered load makes its
        // buffers (`Body::open_placed_tiered`).
        cfg.body.prefill = PrefillMode::Batch;
        let path = crate::ds41_tier::model_path()?;
        let split = || Split::open(&path).map_err(|e| format!("open {path}: {e}"));
        // The draft is read for its header only (B8), so the plan reserves nothing for it.
        cfg.place = bloomery_gpu_gates::tier::plan_levers(&split()?, &levers, 0)?;
        let inputs = PlanInputs::read(&split()?)?;
        let machine = crate::gate_card::plan_gate(inputs.model.layers);
        let plan = inputs.plan(&machine, workstation::CTX_MAX, &cfg.place)?;
        for (on, name) in [
            (
                args.union,
                "T1: the tiered load is the reference load, step, pairs, eager",
            ),
            (
                args.fault,
                "T2: a fault on the tier card's word is the step's error and poison",
            ),
            (
                args.lost,
                "T3: the held tier stream is named as a lost card",
            ),
            (
                args.two,
                "T7: an expert on two devices is refused before any upload",
            ),
            (args.batch, "B7: one batch is the steps"),
            (
                args.batch2,
                "B7 group: two batches as one group are the reference's call",
            ),
            (
                args.bfault,
                "B2: a tier fault before a prompt call is the call's error",
            ),
            (
                args.blost,
                "B3: the held tier stream fails a prompt call by name",
            ),
            (
                args.bfirst,
                "B3f: the first prompt call of a fresh load fails by name",
            ),
            (
                args.bfeat,
                "B8: the feature tap through a tiered prompt call is the steps'",
            ),
        ] {
            if on {
                crate::ds41_tier::sc(name)?;
            }
        }
        let meta = BodyMeta {
            hp: inputs.hp.clone(),
            levers: cfg.body,
        };
        let card = machine.cards[0].name.clone();
        let device = machine.cards[0].device;
        let tier_open = || TierOpen {
            card: TIER_DEVICE,
            name: card.clone(),
            device,
        };
        let tmachine = tier_machine(&machine, &inputs.hp)?;
        let draft = dspark::draft_hparams()?;
        let reserve = dspark::draft_reserve(Place::Bp, &draft.0, std::path::Path::new(&path))?
            .ok_or("plan (b′) made no draft reserve")?;
        let k3 = tier_depth(
            &inputs,
            &bloomery_gpu_gates::tier::plan_levers(&split()?, &levers, reserve)?,
            reserve,
        )?;
        let tplan = tiered(&plan, &tmachine, k3, false)?;
        let tier_layers: Vec<usize> = (0..inputs.model.layers)
            .filter(|&l| tplan.n_l.get(l) != plan.n_l.get(l))
            .collect();
        println!(
            "plan: gate card {card}, {} experts on it by the id prefix; tier: the last {k3} of each \
             routed layer's list on {} layers ({} experts), stage keeps the rest",
            plan.cards[0].experts,
            tier_layers.len(),
            k3 * tier_layers.len()
        );
        let mut pass = true;

        if args.two {
            let bad = tiered(&plan, &tmachine, k3, true)?;
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

        let prompt = prose_ids("engram", PROMPT)?;
        let long = if args.batch
            || args.batch2
            || args.bfault
            || args.blost
            || args.bfirst
            || args.bfeat
        {
            prose_ids("engram", BATCH_P2)?
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
        if args.bfeat {
            let (_, dhp) = dspark::draft_hparams()?;
            let mut f = load()?;
            pass &= batch_feature_case(
                &mut f,
                &long,
                &dhp.target_layers,
                &[dhp.window, long.len()],
                &tier_layers,
            )?;
            drop(f);
            if !(args.on_load() || args.bfirst) {
                return verdict(pass);
            }
        }
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
                    .tiers()
                    .first()
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
        verdict(pass)
    }

    fn verdict(pass: bool) -> Result<(), GateError> {
        println!("{NAME}: {}", bloomery_gpu_gates::tier::tally_line());
        if !pass {
            return Err(checks_failed());
        }
        println!("PASSED: {NAME}");
        Ok(())
    }

    /// B8 (module header): the tap attached to the fresh tiered load `m`,
    /// whose prompt batch's buffers the load made, then those buffers
    /// prepared again; `ids` fed one step each against a prompt call of them
    /// for each of `windows`.
    fn batch_feature_case(
        m: &mut Deepseek41Model,
        ids: &[u32],
        layers: &[usize],
        windows: &[usize],
        tier_layers: &[usize],
    ) -> Result<bool, GateError> {
        body::attach_features(m, layers)?;
        body::prepare_prefill(m)?;
        let width = m.body(NAME)?.feature_width();
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let mut want: Vec<Vec<f32>> = Vec::with_capacity(ids.len());
        for (p, &id) in ids.iter().enumerate() {
            m.step(&[id])?;
            let (gpu, _, b) = m.body_parts(NAME)?;
            let f = b.read_features(gpu, 1)?;
            if f.pos as usize != p || f.values.len() != width {
                return Err(format!(
                    "B8: the step at position {p} read features of position {} ({} values, the \
                     tap's {width})",
                    f.pos,
                    f.values.len()
                )
                .into());
            }
            want.push(f.values.to_vec());
        }
        let mut ok = true;
        let batches = body::batch_count(ids.len());
        for &window in windows {
            let what = format!("B8 (window {window})");
            m.reset()?;
            let before = tier_counts(m)?;
            let mut got: Vec<(usize, Vec<f32>)> = Vec::new();
            let mut sink = |first: u32, rows: &[f32]| -> Result<(), GpuError> {
                for (r, row) in rows.chunks_exact(width).enumerate() {
                    got.push((first as usize + r, row.to_vec()));
                }
                Ok(())
            };
            let call = body::prefill_with(
                m,
                ids,
                Some(FeatureRows {
                    window,
                    sink: &mut sink,
                }),
            );
            if let Err(e) = call {
                println!("FAIL {what}: the prompt call that hands features over failed: {e}");
                ok = false;
                continue;
            }
            let sent = sent_between(before, tier_counts(m)?);
            let from = ids.len() - window.min(ids.len());
            let positions: Vec<usize> = got.iter().map(|(p, _)| *p).collect();
            if positions.iter().copied().ne(from..ids.len()) {
                println!(
                    "FAIL {what}: rows of positions {:?}..{:?} ({} rows), want {from}..{}",
                    positions.first(),
                    positions.last(),
                    positions.len(),
                    ids.len()
                );
                ok = false;
                continue;
            }
            let apart = got.iter().find_map(|(p, row)| {
                row.iter()
                    .zip(&want[*p])
                    .position(|(a, b)| a.to_bits() != b.to_bits())
                    .map(|v| (*p, v))
            });
            if let Some((p, v)) = apart {
                println!("FAIL {what}: position {p}'s features apart from the step's at value {v}");
                ok = false;
            } else {
                println!(
                    "ok {what}: {} positions' features, {from}..{}, bit for bit the steps'",
                    got.len(),
                    ids.len()
                );
            }
            if window >= ids.len() {
                ok &= sent_every(&what, sent.as_ref(), tier_layers);
                let first_batch = body::batches(0, ids.len()).first().map_or(0, |r| r.end);
                if from < first_batch {
                    println!("ok {what} precondition: the first batch handed its rows over too");
                } else {
                    println!(
                        "FAIL {what} precondition: no row from the first batch (ends at \
                         {first_batch}); the group's first set's rows are not exercised"
                    );
                    ok = false;
                }
            }
        }
        let group = m.body(NAME)?.prefill_group().map(|(g, _)| g);
        if group == Some(BATCH_GROUP) && batches == BATCH_GROUP {
            println!("ok B8 structure: {batches} batches, one group of {BATCH_GROUP}");
        } else {
            println!(
                "FAIL B8 structure: {batches} batches under a group of {group:?}, want \
                 {BATCH_GROUP} of each"
            );
            ok = false;
        }
        Ok(ok)
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
        let tier = body.hybrid().tiers().first().ok_or("no tier card")?;
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
            let tier = body.hybrid().tiers().first().ok_or("no tier card")?;
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
            .tiers()
            .first()
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
            let tier = body.hybrid().tiers().first().ok_or("no tier card")?;
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
            .tiers()
            .first()
            .ok_or("no tier card")?
            .gpu()
            .stream()
            .synchronize()?;
        m.gpu().stream().synchronize()?;
        println!("T3: flag raised, both streams drained");
        Ok(ok)
    }
}
