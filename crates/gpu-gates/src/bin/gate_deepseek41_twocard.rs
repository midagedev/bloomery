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
//!    `app::arch::deepseek41`'s `Open`, whose `TierOpen::of_machine` hangs the plan's tier
//!    card under the host tier).
//!
//! The loopback plan is refused unless every row's segments sit on the same
//! device indices with the same experts as the two-card plan's, before any
//! load. The plans are made under a card budget
//! (`PlanLevers::card_budget_bytes`, the field `BLOOMERY_CARD_BUDGET` sets;
//! the lever itself is refused here in the real tier so the environment cannot move the
//! sets): [`BUDGET`] in the real tier, the header's `bloomery.fixture.card_budget` plus the
//! draft's card bytes in the fixture tier (the lever there must equal the header's). It caps
//! every card, the tier's too, so the A6000 holds the stage's
//! and the tier's sets at once. Every load feeds a prompt call as a prompt
//! batch (`PrefillMode::Batch`, set here; `BLOOMERY_PREFILL` is refused, so
//! the environment cannot move the feed, and the reference and the tiered
//! loads run the same one): on a tiered load the tier serves its experts'
//! slots of each batch. Each layer's stage card keeps its id prefix and the
//! tier the next ids, so the preconditions below are met by the prompts'
//! routing, which the run checks.
//!
//! - `--union`: from a reset, the first [`PROMPT`] ids of the prose prompt
//!   (the corpus from [`PROMPT_FROM`]) one decode step per id and [`STEPS`]
//!   greedy steps, every position's
//!   argmax and logits read; then from a reset with the draft started over,
//!   the first [`CALL_PROMPT`] ids as one prompt call through the draft's
//!   (one batch, the draft fed the call's feature rows) and [`PASSES`] DSpark
//!   passes (more, on each load, until a pass has rejected a proposal, up to
//!   [`PASSES_MAX`]), each pass's kept tokens and the logits of the rows it
//!   kept read.
//!   The stepped part, every token and logits vector of the decode steps, of
//!   the two-card run (and of the loopback) is bit for bit the reference's,
//!   the stage's step graph of the reference's node count. The call and the
//!   passes after it are walked in order against the reference's
//!   (`logits_band::walk_at`, a position clear from a margin of
//!   [`CLEAR_MARGIN`]): the call's stage experts run the grouped GEMM and
//!   its tier's the tile path, the reference's all on the GEMM, so their
//!   logits sit within the logits band of the reference's, not on its bits.
//!   The tokens agree until a position the reference was not clear about
//!   (clear top-1 equality is the band's hard clause), after which the
//!   contexts differ and the rest is not compared; the positions both loads
//!   hold are walked, since how many rows the passes keep follows the draft's
//!   proposals, and the passes' accepted proposals are within [`KEPT_PIN`] of
//!   the reference's over the passes the walk compared. Preconditions: every
//!   layer the tier holds experts of was sent at least [`MIN_SLOTS`] routed
//!   slots over the two-card run (the tier's per-layer hits), and by the prompt
//!   call alone, whose batch the tier served (its batch services); the pair
//!   passes kept and rejected a
//!   proposal each at least once (a property of the draft file's content, so the fixture
//!   tier defers it: its random draft keeps none); or the run proves nothing and fails by
//!   name.
//! - `--windows N`: the evidence for [`PROMPT_FROM`], no clause: over the first N starts at
//!   [`WINDOW_STRIDE`] of the prose corpus, each window's prompt call and passes on the
//!   reference, then on the two-card load (one load of each, the decode steps left out). Per window
//!   it prints the tier layers' call hits, the call's band walk and the passes' kept counts, then
//!   a `pos` line per compared position.
//! - `--lost`: on the two-card model, the tier's stream held behind a host
//!   flag before a step — the tier stops signalling: within the go deadline
//!   and its grace the step fails naming the lost card, no token comes out,
//!   the host tier is poisoned as a lost card, and the next step is refused
//!   by that poison at its entry (`GpuError::HostPoisoned` as
//!   `DECODE_INPUT`, a lost card); the flag is then
//!   raised and both streams drain.

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
#[path = "shared/quiet.rs"]
mod quiet;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_pair.rs"]
mod ds41_pair;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_tier.rs"]
#[allow(
    dead_code,
    reason = "the gate reads the levers and the clause tags; the triangle's facts serve the other gates"
)]
mod ds41_tier;

#[cfg(feature = "deepseek41")]
mod gate {
    use std::path::Path;
    use std::sync::Arc;
    use std::time::Instant;

    use app::arch::deepseek41::{Ds41Cfg, Ds41Draft, open_draft};
    use app::{Loaded, OpenArgs, Session};
    use bloomery_gpu::host::batch::TierBatchStats;
    use bloomery_gpu::hybrid::PoisonKind;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{GpuError, HostFlags};
    use bloomery_gpu_deepseek41::body::{
        Body, BodyMeta, DECODE_INPUT, Deepseek41Model, OpenCfg, PAIR_ROWS, PrefillMode,
    };
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::kld;
    use bloomery_gpu_gates::logits_band::{self, Positions};
    use bloomery_gpu_gates::{GateError, checks_failed, prose_ids, ref_model_path};
    use bloomery_levers::{CARD_DONTNEED, ENGRAM_HELPER, HOST_LOCK, HOST_POPULATE, R8};
    use gguf::Split;
    use model::arch::deepseek41::hparams::Hparams;
    use model::arch::deepseek41::place::{self, PlanInputs};
    use model::arch::dspark::DraftHparams;
    use model::placement::{self, Device, ExpertList, Machine, Plan, Role, Row, workstation};
    use runtime::{Advance, Speculative, Target};

    use crate::dspark;
    use crate::quiet::Quiet;

    const NAME: &str = "gate_deepseek41_twocard";
    /// The card budget both plans are made under on the real file: the stage's dense bytes,
    /// cache and set-asides take about 5.7 GB of it, the tier loses the
    /// draft's reserve besides, so the loopback's A6000 holds both sets with
    /// room [derived from the placement pins]. A free parameter of the real file, not a value
    /// the header gives; the fixture tier plans under its header's budget plus the draft's.
    const BUDGET: u64 = 16 << 30;
    /// Prompt positions, fed one decode step each.
    const PROMPT: usize = 32;
    /// Prompt positions of the DSpark leg's prompt call, one whole batch:
    /// with each layer's tier on the ids after its stage prefix, the call's
    /// routing must reach every tier layer, and 128 positions left a layer's
    /// ten tier experts with no slot on the prose prompt.
    const CALL_PROMPT: usize = 512;
    /// The routed slots every tier layer must be sent, over the two-card run and by the prompt
    /// call alone: a layer's tier experts are routed to rarely, and a layer sent fewer has
    /// no tier row for the clause to compare.
    const MIN_SLOTS: u64 = 3;
    /// The distance between the corpus starts a prompt window is chosen among.
    const WINDOW_STRIDE: usize = 1024;
    /// The corpus id the prompts start at: the first multiple of [`WINDOW_STRIDE`] at which every
    /// tier layer is sent [`MIN_SLOTS`] or more routed slots by the prompt call. The preconditions
    /// are the rule and decide nothing else: no other clause's result picks the window, and
    /// `--windows N` prints each start's per-layer hits and every clause's numbers.
    // PIN(2026-11-03): `--windows 74` on the real file: start 0 sends layer 6 no slot (the
    // precondition's red since the grouped IMMA of the default prompt block moved routing
    // near-ties); start 1024 sends every tier layer 6 or more, layer 39 the least. The routing
    // moves with the engine, so a start whose least hits fall under the rule fails by name.
    const PROMPT_FROM: usize = 1024;
    /// Greedy steps after the prompt.
    const STEPS: usize = 48;
    /// DSpark passes after the prompt, and more until a pass has rejected a proposal, up to
    /// [`PASSES_MAX`]: a load whose draft has rejected nothing leaves the reject path unrun.
    const PASSES: usize = 16;
    /// The most passes a load runs looking for a rejected proposal.
    const PASSES_MAX: usize = 48;
    /// What the reference's top1−top2 must be at a position for a flipped top-1 there to be the
    /// band's red: the distance the tier's tile order moves a top-two gap, doubled. The e2e rule's
    /// `MARGIN_FLOOR` is set for one order's rounding and sits inside this gate's.
    // PIN(2026-11-03): `--windows 74` on the real file, 1893 compared positions: the two-card
    // order moved the reference's top-two gap by 0.49 rms and 3.13 at most, and by more than 0.5
    // at 7.9 % of the positions. The flips at a margin of 0.5 or more (five windows) sat at 0.52
    // to 1.28. Twice 3.13, rounded up; 55 % of the positions are still clear.
    const CLEAR_MARGIN: f32 = 6.5;
    /// The accepted proposals the two loads' passes may differ by, over the passes the walk
    /// compared. The draft reads the target's tap rows, which differ by the tier's f32 order, and
    /// its argmax flips at a near-tie, so a pass keeps one row on one load and two on the other.
    // PIN(2026-11-03): `--windows 74` on the real file: the pass-by-pass equality parted at a pass
    // in 16 of 74 windows (1.8 % of 904 compared passes, 0.24 flips a window), and the accepted
    // counts over the 981 compared passes differed by 0 in 64 windows and by 1 in 10. A Poisson
    // tail of 0.24 puts a difference of 5 or more at 5e-6 a window [derived].
    const KEPT_PIN: usize = 4;
    /// The go deadline and the grace a lost card is named within, plus room.
    const LOST_BOUND_S: f64 = 25.0;

    struct Args {
        union: bool,
        loopback: bool,
        lost: bool,
        /// `--windows N`: the prompt windows to print the evidence of.
        windows: Option<usize>,
    }

    fn parse_args() -> Result<Args, GateError> {
        const USAGE: &str =
            "usage: gate_deepseek41_twocard [--union [--loopback]] [--lost] | --windows N";
        let mut a = Args {
            union: false,
            loopback: false,
            lost: false,
            windows: None,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--union" => a.union = true,
                "--loopback" => a.loopback = true,
                "--lost" => a.lost = true,
                "--windows" => {
                    let n = args.next().ok_or(USAGE)?;
                    let n = n.parse::<usize>().ok().filter(|&n| n > 0).ok_or_else(|| {
                        format!("--windows {n:?} is not a count of 1 or more: {USAGE}")
                    })?;
                    a.windows = Some(n);
                }
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        let valid = if a.windows.is_some() {
            !(a.union || a.lost || a.loopback)
        } else {
            (a.union || a.lost) && (a.union || !a.loopback)
        };
        if !valid {
            return Err(USAGE.into());
        }
        Ok(a)
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
            for n in t.iter_mut() {
                *n = 0;
            }
        }
        Ok(out)
    }

    type Spec = Speculative<Ds41Draft, PAIR_ROWS>;

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
        let tier_device = tier.gpu()?.device_name()?;
        if !tier_device.contains(tier_card) {
            return Err(format!("the tier opened on {tier_device}, not the {tier_card}").into());
        }
        println!(
            "load in {:.1} s: stage {} ({} B), tier {} experts on {tier_device} ({} B), draft on {}",
            t0.elapsed().as_secs_f64(),
            m.gpu().device_name()?,
            m.resident_bytes(),
            tier.set().experts(),
            tier.resident_bytes(),
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
        let d = open_draft(
            &mut loaded,
            &draft.0,
            &draft.1,
            target,
            (workstation::RTX_3090.name, None),
        )?;
        let mut s = loaded.ready(&mut Quiet)?;
        let spec = s.with_draft::<_, PAIR_ROWS>(d, &mut Quiet)?;
        Ok(Opened { s, spec })
    }

    /// Every token, kept count and logits vector a run read, in order: one token and one logits
    /// row a position, the row the token was taken from. The decode steps' positions come first,
    /// then the call's last position, then the rows each pass kept.
    #[derive(Default)]
    struct Run {
        tokens: Vec<u32>,
        logits: Vec<Vec<f32>>,
        /// The leading positions the decode steps recorded; the call's and the passes' follow.
        stepped: usize,
        /// The rows each DSpark pass kept.
        kept: Vec<usize>,
        stage_nodes: Option<usize>,
        /// Pair passes that kept their proposal, and that rejected it.
        accepts: usize,
        rejects: usize,
        /// What the prompt call alone sent the tier; `None` without one.
        call: Option<Sent>,
    }

    impl Run {
        /// The call's last position and the rows the passes kept, with the tokens taken from them.
        fn called(&self) -> Positions<'_> {
            Positions {
                tokens: &self.tokens[self.stepped..],
                rows: &self.logits[self.stepped..],
            }
        }
    }

    /// What a span of a run sent the tier: its hits per layer from the
    /// tier's first layer, and the batch services it enqueued.
    struct Sent {
        first: usize,
        hits: Vec<u64>,
        stats: TierBatchStats,
    }

    impl Sent {
        /// The routed slots layer `l` was sent; 0 for a layer the tier holds nothing of.
        fn hit(&self, l: usize) -> u64 {
            l.checked_sub(self.first)
                .and_then(|i| self.hits.get(i))
                .copied()
                .unwrap_or(0)
        }
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
                skipped: a.stats.skipped - b.stats.skipped,
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
        run.stepped = run.tokens.len();
        called(o, &prompt[..CALL_PROMPT], &mut run)?;
        Ok(run)
    }

    /// The DSpark run (module header) of `prompt`, [`CALL_PROMPT`] ids, from a reset with the draft
    /// started over, appended to `run`: [`PASSES`] passes, and more until a proposal was rejected,
    /// up to [`PASSES_MAX`].
    fn called(o: &mut Opened, prompt: &[u32], run: &mut Run) -> Result<(), GateError> {
        let Opened { s, spec } = o;
        s.reset()?;
        spec.draft_mut().restart()?;
        let before = tier_counts(s.model())?;
        let mut last = spec.draft_mut().feed_call(s, prompt)?;
        run.call = sent_between(before, tier_counts(s.model())?);
        run.tokens.push(last);
        run.logits.push(s.model().logits()?);
        let mut out = Vec::with_capacity(PAIR_ROWS);
        let mut done = 0;
        while done < PASSES || (run.rejects == 0 && done < PASSES_MAX) {
            done += 1;
            out.clear();
            let c = Advance::pass(spec, s, last, &mut out)?;
            run.kept.push(c.kept);
            // Row `r` of a pass holds the logits kept token `r` was taken from; the rows past
            // `kept` follow a proposal the target rejected and are taken back.
            let rows: Vec<Vec<f32>> = if c.rows == PAIR_ROWS {
                if c.kept == PAIR_ROWS {
                    run.accepts += 1;
                } else {
                    run.rejects += 1;
                }
                s.model().rows_logits::<PAIR_ROWS>()?.into()
            } else {
                vec![s.model().logits()?]
            };
            if out.len() != c.kept || c.kept > rows.len() {
                return Err(format!(
                    "a pass kept {} rows of its {} and gave {} tokens over {} logits rows",
                    c.kept,
                    c.rows,
                    out.len(),
                    rows.len()
                )
                .into());
            }
            run.tokens.extend_from_slice(&out);
            run.logits.extend(rows.into_iter().take(c.kept));
            last = *out.last().ok_or("a pass kept no token")?;
        }
        spec.draft_mut().check_fault()?;
        Ok(())
    }

    /// Whether `got`, run `what`, is `want`: the stepped part bit for bit, the call and the passes
    /// after it within the logits band of `want`'s ([`walk_called`]), and the passes' accepted
    /// proposals within [`KEPT_PIN`] over the passes the walk compared. The call's stage experts run the grouped
    /// GEMM and its tier's the tile path where the reference's all run the GEMM, so the same exact
    /// integer block sums fold in more than one f32 order and the logits are near the reference's,
    /// not on its bits. Prints each part's line.
    fn same(what: &str, want: &Run, got: &Run) -> bool {
        let stepped = same_stepped(what, want, got);
        let walk = walk_called(what, want, got);
        let kept = same_kept(what, want, got, walk.agreed);
        stepped && walk.ok && kept
    }

    /// The call's last position and the passes' rows of `got` walked against `want`'s at
    /// [`CLEAR_MARGIN`], over the positions both hold: how many rows the passes keep follows the
    /// draft's proposals ([`same_kept`]), so one load can hold more than the other.
    fn walk_called(what: &str, want: &Run, got: &Run) -> logits_band::Walk {
        let (w, g) = (want.called(), got.called());
        let n = w.tokens.len().min(g.tokens.len());
        fn cut(p: Positions<'_>, n: usize) -> Positions<'_> {
            Positions {
                tokens: &p.tokens[..n],
                rows: &p.rows[..n],
            }
        }
        logits_band::walk_at(
            &format!("{what} (the call and its passes)"),
            cut(w, n),
            cut(g, n),
            CLEAR_MARGIN,
        )
    }

    /// Whether `got`'s decode steps, run `what`, are `want`'s bit for bit; prints the first
    /// difference.
    fn same_stepped(what: &str, want: &Run, got: &Run) -> bool {
        let n = want.stepped;
        if got.stepped != n {
            println!(
                "FAIL {what} (the steps): {} positions, the reference ran {n}",
                got.stepped
            );
            return false;
        }
        let (w_tok, w_row) = (&want.tokens[..n], &want.logits[..n]);
        let (g_tok, g_row) = (&got.tokens[..n], &got.logits[..n]);
        let steps = w_tok.iter().zip(w_row).zip(g_tok.iter().zip(g_row));
        for (i, ((wt, wl), (gt, gl))) in steps.enumerate() {
            let first = wl
                .iter()
                .zip(gl)
                .position(|(a, b)| a.to_bits() != b.to_bits());
            if wt != gt || wl.len() != gl.len() || first.is_some() {
                println!(
                    "FAIL {what} (the steps): position {i}: token {gt} (reference {wt}), first \
                     logit apart {first:?}"
                );
                return false;
            }
        }
        println!(
            "ok {what} (the steps): {n} positions, tokens and logits bit for bit the reference's"
        );
        true
    }

    /// Whether the accepted proposals of `got`'s passes are within [`KEPT_PIN`] of `want`'s, over
    /// the passes that end within the first `agreed` positions of [`Run::called`] (the call's last
    /// position, then each pass's kept rows): a pass past them ran on contexts the walk found
    /// apart. Prints both counts.
    fn same_kept(what: &str, want: &Run, got: &Run, agreed: usize) -> bool {
        let at = |r: &Run| logits_band::accepts_in(&r.kept, PAIR_ROWS, agreed);
        let (w, g) = (at(want), at(got));
        let apart = w.accepted.abs_diff(g.accepted);
        let ok = apart <= KEPT_PIN;
        println!(
            "{} {what} (the passes): {} of {} passes kept every row, the reference's {} of {}: apart \
             by {apart} (pin {KEPT_PIN}), over the passes the walk compared",
            if ok { "ok" } else { "FAIL" },
            g.accepted,
            g.passes,
            w.accepted,
            w.passes,
        );
        ok
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&crate::ds41_tier::acts_on(&[
            ENGRAM_HELPER,
            HOST_POPULATE,
            HOST_LOCK,
            CARD_DONTNEED,
            R8,
            bloomery_gpu_gates::llama_place::DRAFT_CARD,
        ])?)?;
        let args = parse_args()?;
        let mut cfg = OpenCfg::from_levers(&levers)?;
        cfg.body.prefill = PrefillMode::Batch;
        let draft = dspark::draft_hparams()?;
        let path = ref_model_path()?;
        let on = dspark::draft_seat(Place::Bp, &draft.0, &path)?;
        let reserve = on.reserve.bytes;
        let head = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        // The real tier plans under the gate's own budget (the lever is refused there); the
        // fixture tier under its header's plus the draft's card bytes, the lever held equal to it.
        let tier = bloomery_gpu_gates::tier::Tier::from_env()?;
        let lever = match tier {
            bloomery_gpu_gates::tier::Tier::Real => bloomery_levers::CardBudget::Every(BUDGET),
            bloomery_gpu_gates::tier::Tier::Fixture => levers.card_budget(),
        };
        cfg.place = bloomery_gpu_gates::tier::budget_levers(
            tier,
            bloomery_gpu_gates::tier::header_budget(&head)?,
            lever,
            reserve,
        )?;
        let budget = cfg
            .place
            .card_budget_bytes
            .every()
            .ok_or("the plan has no card budget")?;
        let inputs = PlanInputs::read(&head)?;
        drop(head);
        let bp = Place::Bp.machine(Some(on), Some(place::tier_batch(&inputs.hp)))?;
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
            "plan (b′) under a card budget of {budget} B, the draft's reserve {reserve} B: stage \
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
        // The tier holds experts on exactly the hybrid layers the pair rule lets a tier run.
        let mut admitted = Vec::new();
        for &l in &hybrid {
            if crate::ds41_pair::tier_admits(&inputs.hp, l)? {
                admitted.push(l);
            }
        }
        let mut pass = true;
        if admitted != tier_layers {
            println!(
                "FAIL sets: the tier holds experts on layers {tier_layers:?}, the stage's hybrid \
                 layers the pair rule admits on a tier are {admitted:?} (hybrid {hybrid:?})"
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
        if let Some(n) = args.windows {
            if !pass {
                return Err(checks_failed());
            }
            let scan = Scan {
                path: &path,
                plan: &plan,
                hp: &inputs.hp,
                cfg: &cfg,
                draft: &draft,
                tier_layers: &tier_layers,
                tier_n,
            };
            return scan.run(bp, n);
        }
        let prompt =
            prose_ids("engram", PROMPT_FROM + PROMPT.max(CALL_PROMPT))?.split_off(PROMPT_FROM);
        println!(
            "prompt: corpus ids {PROMPT_FROM}.. (the first start at which every tier layer is \
             sent {MIN_SLOTS} or more routed slots by the prompt call)"
        );
        if args.union {
            crate::ds41_tier::sc(
                "--union: two cards (and the loopback) are the union reference, the steps bit for \
                 bit, the call and its passes within the logits band",
            )?;
        }
        if args.lost {
            crate::ds41_tier::sc("--lost: the held tier stream is named as a lost card")?;
        }

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
            pass &= call_sent_every("loopback", got.call.as_ref(), &tier_layers, tier_n);
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
            // What a draft keeps is its file's content: the real draft keeps and rejects, a
            // fixture's random one keeps none, so the fixture tier prints the counts and the
            // equalities above hold the passes' tokens and kept counts.
            let both = got.accepts > 0 && got.rejects > 0;
            let held = bloomery_gpu_gates::tier::premise(
                "pair passes kept and rejected a proposal each (the real draft's accept rate)",
                bloomery_gpu_gates::tier::Tag::FileBound,
                both,
            )?;
            if both {
                println!(
                    "ok precondition: the pair passes kept {} proposals and rejected {}",
                    got.accepts, got.rejects
                );
            } else if held {
                println!(
                    "precondition left to the real tier: the pair passes kept {} proposals and \
                     rejected {}",
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
                .filter(|&l| hits(l) < MIN_SLOTS)
                .collect();
            println!(
                "tier: {} layers asked, {} settles ({} early), per layer (layer: tier experts/hits) {}",
                st.issued,
                st.settles,
                st.settle_early,
                layer_counts(&tier_layers, tier_n, hits)
            );
            if missing.is_empty() {
                println!(
                    "ok precondition: every tier layer was sent {MIN_SLOTS} or more routed slots"
                );
            } else {
                println!(
                    "FAIL precondition: tier layers {missing:?} were sent fewer than {MIN_SLOTS} \
                     routed slots; the union clause proves little there"
                );
                pass = false;
            }
            pass &= call_sent_every("two cards", got.call.as_ref(), &tier_layers, tier_n);
        }
        if args.lost {
            pass &= lost_case(o.s.model_mut(), &prompt)?;
        }
        drop(o);
        println!("{NAME}: {}", bloomery_gpu_gates::tier::tally_line());
        if !pass {
            return Err(checks_failed());
        }
        println!("PASSED: {NAME}");
        Ok(())
    }

    /// Each tier layer as `layer: experts/hits`: the experts the tier holds of it (`tier_n`, from
    /// layer 0) and the routed slots it was sent, comma separated.
    fn layer_counts(tier_layers: &[usize], tier_n: &[u64], hit: impl Fn(usize) -> u64) -> String {
        tier_layers
            .iter()
            .map(|&l| format!("{l}: {}/{}", tier_n.get(l).copied().unwrap_or(0), hit(l)))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The prompt call's precondition on the load `what`: the call alone
    /// sent every layer of `tier_layers` at least [`MIN_SLOTS`] routed slots and the
    /// tier served its batch; prints its counters.
    fn call_sent_every(
        what: &str,
        sent: Option<&Sent>,
        tier_layers: &[usize],
        tier_n: &[u64],
    ) -> bool {
        let Some(sent) = sent else {
            println!("FAIL {what} call precondition: the load holds no tier card");
            return false;
        };
        let missing: Vec<usize> = tier_layers
            .iter()
            .copied()
            .filter(|&l| sent.hit(l) < MIN_SLOTS)
            .collect();
        let st = sent.stats;
        println!(
            "{what} call tier: {CALL_PROMPT} positions, {} batch services, {} settles ({} early, \
             {:.2} ms waited), per layer (layer: tier experts/hits) {}",
            st.served,
            st.settles,
            st.settle_early,
            st.settle_ns as f64 / 1e6,
            layer_counts(tier_layers, tier_n, |l| sent.hit(l))
        );
        if missing.is_empty() && st.served > 0 {
            println!(
                "ok {what} call precondition: the prompt batch sent every tier layer \
                 {MIN_SLOTS} or more routed slots, the tier served it"
            );
            true
        } else {
            println!(
                "FAIL {what} call precondition: tier layers {missing:?} were sent fewer than \
                 {MIN_SLOTS} routed slots by the prompt call, {} batch services; the batched \
                 clause proves little there",
                st.served
            );
            false
        }
    }

    /// What `--windows N` reads (module header): the plan's facts, the files and the layers.
    struct Scan<'a> {
        path: &'a Path,
        plan: &'a Plan<'a>,
        hp: &'a Hparams,
        cfg: &'a OpenCfg,
        draft: &'a (Split, DraftHparams),
        tier_layers: &'a [usize],
        tier_n: &'a [u64],
    }

    impl Scan<'_> {
        /// The first `n` windows' prompt calls and passes on the reference, then on the two-card
        /// load; per window, the lines of [`Scan::window`]. No clause: the window's start is read
        /// off the minimum of its call hits.
        fn run(&self, bp: impl Fn(usize) -> Machine, n: usize) -> Result<(), GateError> {
            let corpus = prose_ids("engram", (n - 1) * WINDOW_STRIDE + CALL_PROMPT)?;
            let prompt = |w: usize| &corpus[w * WINDOW_STRIDE..][..CALL_PROMPT];
            let uplan = union_plan(self.plan)?;
            let mut o = open_union(self.path, &uplan, self.hp, self.cfg, self.draft)?;
            let mut want = Vec::with_capacity(n);
            for w in 0..n {
                let mut run = Run::default();
                called(&mut o, prompt(w), &mut run)?;
                want.push(run);
            }
            drop(o);
            let mut o = open(
                self.path,
                bp,
                self.cfg,
                self.draft,
                workstation::RTX_3090.name,
            )?;
            for (w, want) in want.iter().enumerate() {
                let mut got = Run::default();
                called(&mut o, prompt(w), &mut got)?;
                self.window(w * WINDOW_STRIDE, want, &got)?;
            }
            Ok(())
        }

        /// The window at `start`: its tier layers' call hits and the least of them, the walk of the
        /// two-card call and passes against the reference's, the kept counts of both arms and the
        /// proposals each kept and rejected, then one `pos` line per compared position: the
        /// reference's top1-top2 margin, how far the two-card order closed that gap (the margin
        /// less the two-card logits' gap between the reference's top two ids; a value over the
        /// margin is a flipped top-1), and whether the tokens part there.
        fn window(&self, start: usize, want: &Run, got: &Run) -> Result<(), GateError> {
            let sent = got.call.as_ref().ok_or("the two-card load sent no call")?;
            let least = self
                .tier_layers
                .iter()
                .map(|&l| (sent.hit(l), l))
                .min()
                .ok_or("no tier layer")?;
            println!(
                "window {start}: call hits least {} (layer {}) of {MIN_SLOTS} wanted, per layer \
                 (layer: tier experts/hits) {}",
                least.0,
                least.1,
                layer_counts(self.tier_layers, self.tier_n, |l| sent.hit(l))
            );
            let walk = walk_called(&format!("window {start}"), want, got);
            same_kept(&format!("window {start}"), want, got, walk.agreed);
            println!(
                "window {start}: kept reference {:?} two cards {:?}; two cards kept {} proposals \
                 and rejected {}, the reference {} and {}",
                want.kept, got.kept, got.accepts, got.rejects, want.accepts, want.rejects
            );
            let (w, g) = (want.called(), got.called());
            for i in 0..walk.compared {
                let (wr, gr) = (&w.rows[i], &g.rows[i]);
                let (first, top1, second, top2) = kld::top2(wr);
                let margin = top1 - top2;
                let moved = margin - (gr[first] - gr[second]);
                println!(
                    "pos {start} {i} margin {margin:.4} moved {moved:.4}{}",
                    if w.tokens[i] == g.tokens[i] {
                        ""
                    } else {
                        " tokens-apart"
                    }
                );
            }
            Ok(())
        }
    }

    /// `--lost`: the tier's stream held behind a host flag before a step.
    fn lost_case(m: &mut Deepseek41Model, prompt: &[u32]) -> Result<bool, GateError> {
        m.reset()?;
        m.step(&prompt[..4])?;
        let flags = {
            let body = m.body(NAME)?;
            let tier = body.hybrid().tiers().first().ok_or("no tier card")?;
            let flags = HostFlags::new(tier.gpu()?.context(), 1)?;
            flags.enqueue_wait(tier.gpu()?.stream(), 0)?;
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
            .gpu()?
            .stream()
            .synchronize()?;
        m.gpu().stream().synchronize()?;
        println!("lost: flag raised, both streams drained");
        Ok(ok)
    }
}
