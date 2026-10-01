//! GPU gate for plan (b′) on both cards (`--place bp`): V4.1's stage on the
//! A6000 and its expert tier on the 3090, the DSpark draft beside the tier
//! on the 3090 on its own `Gpu` and stream, against one card holding the
//! same expert sets.
//!
//! Three loads of one plan (b′), one after the other, the DSpark draft on the
//! 3090 in each:
//!
//! 1. the reference: the plan with every tier segment moved onto the stage
//!    card (each routed stack's stage and tier lists joined, the host's
//!    untouched), opened with no tier — the A6000 holds the union, and no
//!    tier code runs;
//! 2. with `--loopback`, the loopback of `gate_deepseek41_tier` on the A6000:
//!    the same plan with the tier card renamed to the A6000, so the stage and
//!    the tier are two `Gpu`s on one card and the tier's rows cross no link;
//! 3. the two cards, through the binaries' own open (`app::Loaded` over
//!    `app::arch::deepseek41`'s `Open`, whose `tier_of` hangs the plan's tier
//!    card under the host tier).
//!
//! The loopback plan is refused unless every row's segments sit on the same
//! device indices with the same experts as the two-card plan's, before any
//! load. The plans are made under the card budget [`BUDGET`]
//! (`PlanLevers::card_budget_bytes`, the field `BLOOMERY_CARD_BUDGET` sets;
//! the lever itself is refused here so the environment cannot move the
//! sets): it caps every card, the tier's too, so the A6000 holds the stage's
//! and the tier's sets at once. Every load feeds a prompt call as a prompt
//! batch (`PrefillMode::Batch`, set here; `BLOOMERY_PREFILL` is refused, so
//! the environment cannot move the feed, and the reference and the tiered
//! loads run the same one): on a tiered load the tier serves its experts'
//! slots of each batch. Each layer's stage card keeps its id prefix and the
//! tier the next ids, so the preconditions below are met by the prompts'
//! routing, which the run checks.
//!
//! - `--union`: from a reset, the first [`PROMPT`] ids of the prose prompt
//!   one decode step per id and [`STEPS`] greedy steps, every position's
//!   argmax and logits read; then from a reset with the draft started over,
//!   the first [`CALL_PROMPT`] ids as one prompt call through the draft's
//!   (one batch, the draft fed the call's feature rows) and [`PASSES`] DSpark
//!   passes, each pass's kept tokens and its rows' logits read. Every token,
//!   kept count and logits vector of the two-card run (and of the loopback)
//!   bit for bit the reference's, the stage's step graph of the reference's
//!   node count. Preconditions: every layer the tier holds experts of was
//!   sent at least one routed slot over the two-card run (the tier's
//!   per-layer hits), and by the prompt call alone, whose batch the tier
//!   served (its batch services); the pair passes kept and rejected a
//!   proposal each at least once; or the run proves nothing and fails by
//!   name.
//! - `--lost`: on the two-card model, the tier's stream held behind a host
//!   flag before a step — the tier stops signalling: within the go deadline
//!   and its grace the step fails naming the lost card, no token comes out,
//!   the host tier is poisoned as a lost card, and the next step is refused
//!   by that poison at its entry (`GpuError::HostPoisoned` as
//!   `DECODE_INPUT`, a lost card); the flag is then
//!   raised and both streams drain.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_deepseek41_twocard: built without the `deepseek41` feature; see `just gate-gpu-ds41-twocard`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_twocard", gate::run())
}

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_dspark.rs"]
#[allow(
    dead_code,
    reason = "the gate puts the draft on the 3090 itself; the card rule serves the binaries"
)]
mod dspark;

#[cfg(feature = "deepseek41")]
mod gate {
    use std::path::Path;
    use std::sync::Arc;
    use std::time::Instant;

    use app::arch::deepseek41::{CardDraft, Ds41Cfg};
    use app::{Loaded, OpenArgs, OpenLog, RowsLog, Session, SessionError};
    use bloomery_gpu::host::batch::TierBatchStats;
    use bloomery_gpu::hybrid::PoisonKind;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{GpuError, HostFlags};
    use bloomery_gpu_deepseek41::body::{
        Body, BodyMeta, DECODE_INPUT, Deepseek41Model, OpenCfg, PAIR_ROWS, PrefillMode,
    };
    use bloomery_gpu_deepseek41::draft::DraftBody;
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::{GateError, checks_failed, data_dir, ref_model_path};
    use bloomery_levers::{CARD_DONTNEED, ENGRAM_HELPER, HOST_LOCK, HOST_POPULATE, R8};
    use gguf::Split;
    use model::arch::deepseek41::hparams::Hparams;
    use model::arch::deepseek41::place::{self, PlanInputs};
    use model::arch::dspark::DraftHparams;
    use model::placement::{self, Device, ExpertList, Machine, Plan, Role, Row, workstation};
    use runtime::{Advance, Speculative, Target};

    use crate::dspark;

    const NAME: &str = "gate_deepseek41_twocard";
    /// The card budget both plans are made under: the stage's dense bytes,
    /// cache and set-asides take about 5.7 GB of it, the tier loses the
    /// draft's reserve besides, so the loopback's A6000 holds both sets with
    /// room [derived from the placement pins].
    const BUDGET: u64 = 16 << 30;
    /// Prompt positions, fed one decode step each.
    const PROMPT: usize = 32;
    /// Prompt positions of the DSpark leg's prompt call, one whole batch:
    /// with each layer's tier on the ids after its stage prefix, the call's
    /// routing must reach every tier layer, and 128 positions left a layer's
    /// ten tier experts with no slot on the prose prompt.
    const CALL_PROMPT: usize = 512;
    /// Greedy steps after the prompt.
    const STEPS: usize = 48;
    /// DSpark passes after the prompt.
    const PASSES: usize = 16;
    /// The go deadline and the grace a lost card is named within, plus room.
    const LOST_BOUND_S: f64 = 25.0;

    struct Args {
        union: bool,
        loopback: bool,
        lost: bool,
    }

    fn parse_args() -> Result<Args, GateError> {
        const USAGE: &str = "usage: gate_deepseek41_twocard [--union [--loopback]] [--lost]";
        let mut a = Args {
            union: false,
            loopback: false,
            lost: false,
        };
        for arg in std::env::args().skip(1) {
            match arg.as_str() {
                "--union" => a.union = true,
                "--loopback" => a.loopback = true,
                "--lost" => a.lost = true,
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        if !(a.union || a.lost) || (a.loopback && !a.union) {
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

    /// Plan (b′) `m` with its tier card renamed to the A6000, every figure
    /// kept: the loopback's.
    fn looped(mut m: Machine) -> Machine {
        for t in &mut m.tiers {
            t.name = workstation::A6000.name.to_owned();
        }
        m
    }

    /// The first row whose segments `a` and `b` place differently (device
    /// index, format or experts), or `None`: the same sets.
    fn apart(a: &Plan<'_>, b: &Plan<'_>) -> Option<String> {
        if a.n_l != b.n_l || a.tier_n_l != b.tier_n_l {
            return Some("the per-layer counts".into());
        }
        a.rows.iter().zip(&b.rows).find_map(|(ra, rb)| {
            let key = |r: &model::placement::Row| {
                r.segments
                    .iter()
                    .map(|s| {
                        (
                            s.device,
                            s.format,
                            s.experts.as_ref().map(|e| e.ids().to_vec()),
                            s.resident_bytes,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            (key(ra) != key(rb)).then(|| a.model.tensors[ra.tensor].name.clone())
        })
    }

    /// `plan` with each routed stack's tier segment joined to its stage-card
    /// segment on card 0 (`placement::routed_row` over the two lists), the
    /// host's segments as they are: the stage card holds the union, and no
    /// segment sits on the tier.
    fn union_plan<'a>(plan: &Plan<'a>) -> Result<Plan<'a>, GateError> {
        let model = plan.model;
        let tier = Device::Card(plan.machine.cards.len());
        let mut out = plan.clone();
        for row in &mut out.rows {
            let t = &model.tensors[row.tensor];
            if t.role != Role::RoutedExperts {
                continue;
            }
            let ids = |d: Device| -> Vec<u32> {
                row.segments
                    .iter()
                    .filter(|s| s.device == d)
                    .filter_map(|s| s.experts.as_ref())
                    .flat_map(|e| e.ids().iter().copied())
                    .collect()
            };
            let on_tier = ids(tier);
            if on_tier.is_empty() {
                continue;
            }
            let mut union = ids(Device::Card(0));
            union.extend(on_tier);
            let n = union.len();
            let card = placement::routed_row(
                row.tensor,
                t,
                0,
                ExpertList::new(union, model.experts)?,
                model,
            )?;
            let segments = card
                .segments
                .into_iter()
                .filter(|s| s.device == Device::Card(0))
                .chain(
                    row.segments
                        .iter()
                        .filter(|s| !matches!(s.device, Device::Card(_)))
                        .cloned(),
                )
                .collect();
            *row = Row {
                segments,
                ..row.clone()
            };
            if let Some(l) = t.layer.and_then(|l| out.n_l.get_mut(l)) {
                *l = u64::try_from(n)?;
            }
        }
        for t in &mut out.tier_n_l {
            t.iter_mut().for_each(|n| *n = 0);
        }
        Ok(out)
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
        fn load(&mut self, _: &Deepseek41Model) -> Result<(), SessionError> {
            Ok(())
        }
        fn capture(&mut self, _: usize) -> Result<(), SessionError> {
            Ok(())
        }
        fn prompt_buffers(&mut self, _: &Deepseek41Model) -> Result<(), SessionError> {
            Ok(())
        }
    }

    impl RowsLog for Quiet {
        fn capture_rows(&mut self, _: usize, _: usize) -> Result<(), SessionError> {
            Ok(())
        }
    }

    type Spec = Speculative<CardDraft<DraftBody>, PAIR_ROWS>;

    /// A load: the session and its draft.
    struct Opened {
        s: Session<Body>,
        spec: Spec,
    }

    /// The target by `machine` under `cfg` through the binaries' open, and
    /// its draft ([`with_draft`]); refused unless the tier's `Gpu` is the
    /// card named `tier_card`.
    fn open(
        path: &Path,
        machine: impl Fn(usize) -> Machine,
        cfg: &OpenCfg,
        draft: &(Split, DraftHparams),
        tier_card: &str,
    ) -> Result<Opened, GateError> {
        let t0 = Instant::now();
        let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let args = OpenArgs {
            place: Place::Bp.name(),
            machine,
            ctx: usize::try_from(workstation::CTX_MAX)?,
            mode: StepMode::Graph,
            cfg: ds41(cfg),
        };
        let loaded =
            Loaded::<Body>::open(file, args, &mut Quiet)?.ok_or("the open planned nothing")?;
        let o = with_draft(loaded, path, draft)?;
        let m = o.s.model();
        let tier = m
            .body(NAME)?
            .hybrid()
            .tiers()
            .first()
            .ok_or("the (b′) load holds no tier card")?;
        let tier_device = tier.gpu().device_name()?;
        if !tier_device.contains(tier_card) {
            return Err(format!("the tier opened on {tier_device}, not the {tier_card}").into());
        }
        println!(
            "load in {:.1} s: stage {} ({} B), tier {} experts on {tier_device} ({} B), draft on {}",
            t0.elapsed().as_secs_f64(),
            m.gpu().device_name()?,
            m.resident_bytes(),
            tier.set().experts(),
            tier.weights().resident_bytes(),
            workstation::RTX_3090.name,
        );
        Ok(o)
    }

    /// The reference: `plan`, whose stage card holds the union
    /// ([`union_plan`]), opened on card 0 with no tier, and its draft.
    fn open_union(
        path: &Path,
        plan: &Plan<'_>,
        hp: &Hparams,
        cfg: &OpenCfg,
        draft: &(Split, DraftHparams),
    ) -> Result<Opened, GateError> {
        let t0 = Instant::now();
        let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let meta = BodyMeta {
            hp: hp.clone(),
            levers: cfg.body,
        };
        let mut model = Body::open_placed_tiered(file, plan, 0, None, &meta)?;
        model.set_mode(StepMode::Graph);
        if !model.body(NAME)?.hybrid().tiers().is_empty() {
            return Err("the reference load holds a tier card".into());
        }
        let loaded = Loaded::from_model(model, ds41(cfg), u32::try_from(plan.ctx_max)?);
        let o = with_draft(loaded, path, draft)?;
        println!(
            "reference load in {:.1} s: {} ({} B) holds the union, no tier",
            t0.elapsed().as_secs_f64(),
            o.s.model().gpu().device_name()?,
            o.s.model().resident_bytes(),
        );
        Ok(o)
    }

    /// The session's configuration: `cfg`, its prompt feed the body's.
    fn ds41(cfg: &OpenCfg) -> Ds41Cfg {
        Ds41Cfg {
            open: cfg.clone(),
            feed: cfg.body.prefill,
            card_timing: false,
        }
    }

    /// `loaded` with the DSpark draft on the 3090, its taps attached before
    /// the captures, then the step's and the pair pass's captures.
    fn with_draft(
        mut loaded: Loaded<Body>,
        path: &Path,
        draft: &(Split, DraftHparams),
    ) -> Result<Opened, GateError> {
        let target =
            Arc::new(Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?);
        let d = CardDraft::open(
            &mut loaded,
            &draft.0,
            &draft.1,
            target,
            workstation::RTX_3090.name,
        )?;
        let mut s = loaded.ready(&mut Quiet)?;
        let spec = s.with_draft::<_, PAIR_ROWS>(d, &mut Quiet)?;
        Ok(Opened { s, spec })
    }

    /// Every token, kept count and logits vector a run read, in order.
    #[derive(Default)]
    struct Run {
        tokens: Vec<u32>,
        logits: Vec<Vec<f32>>,
        kept: Vec<usize>,
        stage_nodes: Option<usize>,
        /// Pair passes that kept their proposal, and that rejected it.
        accepts: usize,
        rejects: usize,
        /// What the prompt call alone sent the tier; `None` without one.
        call: Option<Sent>,
    }

    /// What a span of a run sent the tier: its hits per layer from the
    /// tier's first layer, and the batch services it enqueued.
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

    /// The counters `after` less `before`: what the span between sent.
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

    /// The step run, then the DSpark run (module header).
    fn union_run(o: &mut Opened, prompt: &[u32]) -> Result<Run, GateError> {
        let mut run = Run::default();
        let Opened { s, spec } = o;
        s.reset()?;
        spec.draft_mut().restart()?;
        let mut tok = 0;
        for &id in &prompt[..PROMPT] {
            tok = s.model_mut().step(&[id])?;
            run.tokens.push(tok);
            run.logits.push(s.model().logits()?);
        }
        run.stage_nodes = Some(s.model().step_graph_nodes()?.len());
        for _ in 0..STEPS {
            tok = s.model_mut().step(&[tok])?;
            run.tokens.push(tok);
            run.logits.push(s.model().logits()?);
        }
        s.reset()?;
        spec.draft_mut().restart()?;
        let before = tier_counts(s.model())?;
        let mut last = spec.draft_mut().feed_call(s, &prompt[..CALL_PROMPT])?;
        run.call = sent_between(before, tier_counts(s.model())?);
        run.tokens.push(last);
        run.logits.push(s.model().logits()?);
        let mut out = Vec::with_capacity(PAIR_ROWS);
        for _ in 0..PASSES {
            out.clear();
            let c = Advance::pass(spec, s, last, &mut out)?;
            run.kept.push(c.kept);
            run.tokens.extend_from_slice(&out);
            if c.rows == PAIR_ROWS {
                let [a, b] = s.model().rows_logits::<PAIR_ROWS>()?;
                run.logits.push(a);
                run.logits.push(b);
                if c.kept == PAIR_ROWS {
                    run.accepts += 1;
                } else {
                    run.rejects += 1;
                }
            } else {
                run.logits.push(s.model().logits()?);
            }
            last = *out.last().ok_or("a pass kept no token")?;
        }
        spec.draft_mut().check_fault()?;
        Ok(run)
    }

    /// Whether `got`, run `what`, is `want` bit for bit; prints the first
    /// difference.
    fn same(what: &str, want: &Run, got: &Run) -> bool {
        if want.tokens != got.tokens || want.kept != got.kept {
            let at = want
                .tokens
                .iter()
                .zip(&got.tokens)
                .position(|(a, b)| a != b);
            println!(
                "FAIL {what}: tokens apart at {at:?} ({} and {} tokens), kept {:?} against the \
                 reference's {:?}",
                got.tokens.len(),
                want.tokens.len(),
                got.kept,
                want.kept
            );
            return false;
        }
        if want.logits.len() != got.logits.len() {
            println!(
                "FAIL {what}: {} logits rows, the reference read {}",
                got.logits.len(),
                want.logits.len()
            );
            return false;
        }
        for (i, (w, g)) in want.logits.iter().zip(&got.logits).enumerate() {
            let first = w
                .iter()
                .zip(g)
                .position(|(a, b)| a.to_bits() != b.to_bits());
            if w.len() != g.len() || first.is_some() {
                println!("FAIL {what}: logits row {i}: first logit apart {first:?}");
                return false;
            }
        }
        println!(
            "ok {what}: {} tokens, {} logits rows, {} passes' kept counts bit for bit the \
             reference's",
            got.tokens.len(),
            got.logits.len(),
            got.kept.len()
        );
        true
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[
            ENGRAM_HELPER,
            HOST_POPULATE,
            HOST_LOCK,
            CARD_DONTNEED,
            R8,
        ])?;
        let args = parse_args()?;
        let mut cfg = OpenCfg::from_levers(&levers)?;
        cfg.place.card_budget_bytes = Some(BUDGET);
        cfg.body.prefill = PrefillMode::Batch;
        let draft = dspark::draft_hparams()?;
        let path = ref_model_path()?;
        let reserve = dspark::draft_reserve(Place::Bp, &draft.0, &path)?
            .ok_or("plan (b′) made no draft reserve")?;
        let inputs = PlanInputs::read(
            &Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?,
        )?;
        let bp = Place::Bp.machine(Some(reserve), Some(place::tier_batch(&inputs.hp)))?;
        let layers = inputs.model.layers;
        let (two, lmachine) = (bp(layers), looped(bp(layers)));
        let plan = inputs.plan(&two, workstation::CTX_MAX, &cfg.place)?;
        let lplan = inputs.plan(&lmachine, workstation::CTX_MAX, &cfg.place)?;
        let tier_n = plan.tier_n_l.first().ok_or("plan (b′) has no tier")?;
        let tier_layers: Vec<usize> = (0..layers)
            .filter(|&l| tier_n.get(l).is_some_and(|&n| n > 0))
            .collect();
        let held = |v: &[u64]| {
            let h: Vec<u64> = v.iter().copied().filter(|&n| n > 0).collect();
            (
                h.iter().min().copied().unwrap_or(0),
                h.iter().max().copied().unwrap_or(0),
            )
        };
        println!(
            "plan (b′) under a card budget of {BUDGET} B, the draft's reserve {reserve} B: stage \
             {} experts (n_l {:?}), tier {} experts (per layer {:?}) on {} layers, host {} experts",
            plan.cards[0].experts,
            held(&plan.n_l),
            plan.cards.get(1).map_or(0, |c| c.experts),
            held(tier_n),
            tier_layers.len(),
            plan.host.experts
        );
        let hybrid: Vec<usize> = (0..layers)
            .filter(|&l| plan.n_l.get(l).is_some_and(|&n| n > 0))
            .collect();
        let mut pass = true;
        if hybrid != tier_layers {
            println!(
                "FAIL sets: the tier holds experts on layers {tier_layers:?}, the stage's hybrid \
                 layers are {hybrid:?}"
            );
            pass = false;
        }
        match apart(&plan, &lplan) {
            None => println!(
                "ok sets: the loopback plan places every row as the two-card plan (device \
                 indices, formats, experts, bytes)"
            ),
            Some(what) => {
                return Err(format!(
                    "the loopback plan places {what} otherwise than the two-card plan: the \
                     reference would not hold the same sets"
                )
                .into());
            }
        }
        let prompt = prose(PROMPT.max(CALL_PROMPT))?;

        let reference = if args.union {
            let uplan = union_plan(&plan)?;
            println!(
                "reference: the stage card holds {} experts, the union of the stage's and the tier's",
                uplan.n_l.iter().sum::<u64>()
            );
            let mut o = open_union(&path, &uplan, &inputs.hp, &cfg, &draft)?;
            let r = union_run(&mut o, &prompt)?;
            drop(o);
            Some(r)
        } else {
            None
        };
        if let (Some(want), true) = (&reference, args.loopback) {
            let mut o = open(
                &path,
                |l| looped(bp(l)),
                &cfg,
                &draft,
                workstation::A6000.name,
            )?;
            let got = union_run(&mut o, &prompt)?;
            drop(o);
            pass &= same("loopback", want, &got);
            pass &= call_sent_every("loopback", got.call.as_ref(), &tier_layers);
        }
        let mut o = open(&path, bp, &cfg, &draft, workstation::RTX_3090.name)?;
        if let Some(want) = &reference {
            let got = union_run(&mut o, &prompt)?;
            pass &= same("two cards", want, &got);
            if got.stage_nodes == want.stage_nodes {
                println!(
                    "ok structure: stage graph of {:?} nodes, the reference's",
                    got.stage_nodes
                );
            } else {
                println!(
                    "FAIL structure: stage graph of {:?} nodes, the reference's {:?}",
                    got.stage_nodes, want.stage_nodes
                );
                pass = false;
            }
            if got.accepts > 0 && got.rejects > 0 {
                println!(
                    "ok precondition: the pair passes kept {} proposals and rejected {}",
                    got.accepts, got.rejects
                );
            } else {
                println!(
                    "FAIL precondition: the pair passes kept {} proposals and rejected {}: the \
                     DSpark clause needs both",
                    got.accepts, got.rejects
                );
                pass = false;
            }
            let body = o.s.model().body(NAME)?;
            let tier = body
                .hybrid()
                .tiers()
                .first()
                .ok_or("the (b′) load holds no tier")?;
            let st = tier.stats();
            let first = tier.set().layers().start;
            let hits = |l: usize| st.layer_hits.get(l - first).copied().unwrap_or(0);
            let missing: Vec<usize> = tier_layers
                .iter()
                .copied()
                .filter(|&l| hits(l) == 0)
                .collect();
            println!(
                "tier: {} layers asked, {} settles ({} early), hits per layer {:?}",
                st.issued,
                st.settles,
                st.settle_early,
                tier_layers.iter().map(|&l| hits(l)).collect::<Vec<_>>()
            );
            if missing.is_empty() {
                println!("ok precondition: every tier layer was sent a routed slot");
            } else {
                println!(
                    "FAIL precondition: tier layers {missing:?} were sent no routed slot; the \
                     union clause proves nothing there"
                );
                pass = false;
            }
            pass &= call_sent_every("two cards", got.call.as_ref(), &tier_layers);
        }
        if args.lost {
            pass &= lost_case(o.s.model_mut(), &prompt)?;
        }
        drop(o);
        if !pass {
            return Err(checks_failed());
        }
        println!("PASSED: {NAME}");
        Ok(())
    }

    /// The prompt call's precondition on the load `what`: the call alone
    /// sent every layer of `tier_layers` at least one routed slot and the
    /// tier served its batch; prints its counters.
    fn call_sent_every(what: &str, sent: Option<&Sent>, tier_layers: &[usize]) -> bool {
        let Some(sent) = sent else {
            println!("FAIL {what} call precondition: the load holds no tier card");
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
        let st = sent.stats;
        println!(
            "{what} call tier: {CALL_PROMPT} positions, {} batch services, {} settles ({} early, \
             {:.2} ms waited), hits per layer {:?}",
            st.served,
            st.settles,
            st.settle_early,
            st.settle_ns as f64 / 1e6,
            tier_layers.iter().map(|&l| hit(l)).collect::<Vec<_>>()
        );
        if missing.is_empty() && st.served > 0 {
            println!(
                "ok {what} call precondition: the prompt batch sent every tier layer a routed \
                 slot, the tier served it"
            );
            true
        } else {
            println!(
                "FAIL {what} call precondition: tier layers {missing:?} were sent no routed slot \
                 by the prompt call, {} batch services; the batched clause proves nothing there",
                st.served
            );
            false
        }
    }

    /// `--lost`: the tier's stream held behind a host flag before a step.
    fn lost_case(m: &mut Deepseek41Model, prompt: &[u32]) -> Result<bool, GateError> {
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
            Err(e)
                if e.to_string().contains("the card is lost")
                    && e.to_string().contains(workstation::RTX_3090.name)
                    && secs < LOST_BOUND_S =>
            {
                println!(
                    "ok lost: the step fails in {secs:.1} s naming the lost card, no token: {e}"
                );
                true
            }
            other => {
                println!(
                    "FAIL lost: after {secs:.1} s the step gave {other:?}, want the lost card"
                );
                false
            }
        };
        let kind = m.body(NAME)?.hybrid().last_poison().map(|p| p.mark().kind);
        if kind != Some(PoisonKind::CardLost) {
            println!("FAIL lost: the host tier's poison is {kind:?}, want CardLost");
            ok = false;
        }
        match m.step(&[prompt[5]]) {
            Err(GpuError::HostPoisoned { what, poison })
                if what == DECODE_INPUT && poison.mark().kind == PoisonKind::CardLost =>
            {
                println!(
                    "ok lost: the next step is refused by the lost card's poison: {what}: {poison}"
                );
            }
            other => {
                println!(
                    "FAIL lost: the next step gave {other:?}, want the host tier's refusal as \
                     {DECODE_INPUT} naming the lost card"
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
        println!("lost: flag raised, both streams drained");
        Ok(ok)
    }
}
