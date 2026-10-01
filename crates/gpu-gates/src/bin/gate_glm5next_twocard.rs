//! GPU gate for GLM-5.3-Flash on plan (b′) (`--place bp`): the stage on the
//! A6000 and the expert tier on the 3090, against one card holding the same
//! expert sets.
//!
//! Two loads of one plan (b′) at two KDA lanes (`PlanInputs::plan_lanes`,
//! `KdaLanes::Two`: the load the two-row verify runs on), one after the
//! other:
//!
//! 1. the reference: the plan with every tier segment moved onto the stage
//!    card (each routed stack's stage and tier lists joined, the host's
//!    untouched) and the tier card dropped from the plan's machine, opened by
//!    `Body::open_placed_lanes` — the A6000 holds the union, and no tier code
//!    runs;
//! 2. the two cards, through the binaries' own open
//!    (`app::arch::glm5next::open_pair`, whose load hangs the plan's tier
//!    card under the host tier).
//!
//! Memory: both plans are made under the card budget [`BUDGET`]
//! (`PlanLevers::card_budget_bytes`; `BLOOMERY_CARD_BUDGET` is refused here,
//! so the environment cannot move the sets). It caps every card, the tier's
//! too. Without it the union is the stage's set plus the tier's, each
//! planned up to its own card's usable bytes, and the two together pass the
//! A6000's. Under it the stage's resident bytes are at most `BUDGET` and the
//! tier's experts at most `BUDGET` less its batch reserve, so the reference
//! holds at most 2 · `BUDGET` = 32 GiB of the A6000's 48 GiB [derived from
//! the placement's rule]. Every load feeds a
//! prompt call as a prompt batch (`PrefillMode::Batch`, set here). Each
//! layer's stage card keeps its id prefix and the tier the next ids, so the
//! preconditions below are met by the prompt's routing, which the run checks.
//!
//! The prompt is the GLM prose of the `d1k` reference set (its prefill ids,
//! through `refset::arch::glm5next`). Each load runs three legs, each from a
//! reset:
//!
//! - step: the first [`PROMPT`] ids one decode step each, then [`STEPS`]
//!   greedy steps, every position's argmax and logits read;
//! - pair: the same [`PROMPT`] decode steps, then [`PASSES`] verifies of two
//!   rows (`GpuModel::step_rows`), the draft of each row 1 the step leg's
//!   next greedy token, made wrong on every third pass; each pass keeps
//!   both rows when row 0 is the draft and its first row otherwise
//!   (`rollback` and `keep_rows(…, PassKind::Pair)`), its rows' tokens, kept
//!   count and both rows' logits read;
//! - call: the first [`CALL_PROMPT`] ids as one prompt call (one batch: the
//!   tier serves the slots of its experts from the batch), its token and
//!   logits row read.
//!
//! Clauses, against the reference:
//! - decode bits: the step and pair legs' tokens, kept counts and logits
//!   rows bit for bit the reference's;
//! - call bits: the call leg's token and logits row bit for bit the
//!   reference's (no decode step follows the call, so a decode defect cannot
//!   turn this clause red);
//! - structure: the stage's captured step graph holds the reference's node
//!   count;
//! - precondition: every layer the tier holds experts of was sent at least
//!   one routed slot by the decode legs (the step and pair legs together, the
//!   tier's per-layer hits), and by the call alone, whose batch the tier
//!   served (its batch services); the pair passes kept and rejected a draft
//!   each at least once; the tier's layers are the stage's hybrid layers; or
//!   the run proves nothing and fails by name. A miss is a prompt too short
//!   for the routing (raise [`PROMPT`], [`STEPS`] or [`PASSES`]), not a
//!   defect of the tier.

#[cfg(not(feature = "glm5next"))]
fn main() {
    eprintln!(
        "gate_glm5next_twocard: built without the `glm5next` feature; see `just gate-gpu-glm5next-twocard`."
    );
    std::process::exit(2);
}

#[cfg(feature = "glm5next")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_glm5next_twocard", gate::run())
}

#[cfg(feature = "glm5next")]
mod gate {
    use std::time::Instant;

    use app::arch::glm5next::{GlmCfg, open_pair};
    use app::{Loaded, OpenArgs, OpenLog, Session, SessionError};
    use bloomery_gpu::host::PassKind;
    use bloomery_gpu::host::batch::TierBatchStats;
    use bloomery_gpu::host::swap::Residency;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::{GateError, RefManifest, checks_failed, data_dir};
    use bloomery_gpu_glm5next::{Body, Glm5nextModel, PrefillMode, feed};
    use bloomery_levers::{CARD_DONTNEED, HOST_LOCK, HOST_POPULATE};
    use gguf::Split;
    use model::arch::glm5next::place::{self, KdaLanes, PlanInputs};
    use model::placement::{
        self, Device, ExpertList, Machine, Plan, PlanLevers, Role, Row, workstation,
    };
    use refset::arch::glm5next::{D1K, IK, MODEL};

    const NAME: &str = "gate_glm5next_twocard";
    /// The card budget both plans are made under (module header: the union
    /// it leaves fits the A6000).
    const BUDGET: u64 = 16 << 30;
    /// Cache rows: every position the latent layers attend whole
    /// (`place::dense_positions`), past every leg's last position.
    const CTX: usize = 2051;
    /// Prompt positions of the step and pair legs, one decode step each.
    const PROMPT: usize = 32;
    /// Greedy steps after the step leg's prompt: past the pair leg's last
    /// draft (two tokens a pass at most, and the next).
    const STEPS: usize = 48;
    /// Verifies of two rows after the pair leg's prompt.
    const PASSES: usize = 16;
    /// Prompt positions of the call leg's prompt call, one whole batch
    /// (`prefill::T_MAX`): with each layer's tier on the ids after its stage
    /// prefix, the call's routing must reach every tier layer.
    const CALL_PROMPT: usize = 512;
    /// The vocabulary: a wrong draft is the right one plus one, within it.
    const N_VOCAB: u32 = 154_880;

    /// The prompt: the `d1k` set's prefill ids.
    fn prompt() -> Result<Vec<u32>, GateError> {
        let man = RefManifest::open(&data_dir().join(D1K), &IK)?;
        let (_, _, prefill) = man.step()?;
        let n = PROMPT.max(CALL_PROMPT);
        if prefill.len() < n {
            return Err(format!("{D1K}: {} prefill ids, the gate reads {n}", prefill.len()).into());
        }
        Ok(prefill[..n].to_vec())
    }

    /// `plan` with each routed stack's tier segment joined to its stage-card
    /// segment on card 0 (`placement::routed_row` over the two lists), the
    /// host's segments as they are, on `flat` — the plan's machine with no
    /// tier card: the stage card holds the union, and no segment sits on a
    /// tier.
    fn union_plan<'a>(plan: &Plan<'a>, flat: &'a Machine) -> Result<Plan<'a>, GateError> {
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
        out.machine = flat;
        out.cards.truncate(flat.cards.len());
        out.tier_n_l.clear();
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
        fn load(&mut self, _: &Glm5nextModel) -> Result<(), SessionError> {
            Ok(())
        }
        fn capture(&mut self, _: usize) -> Result<(), SessionError> {
            Ok(())
        }
        fn prompt_buffers(&mut self, _: &Glm5nextModel) -> Result<(), SessionError> {
            Ok(())
        }
    }

    /// The tiered load through the binaries' open, refused unless its tier's
    /// `Gpu` is the 3090.
    fn open_two(
        machine: impl Fn(usize) -> Machine,
        cfg: &GlmCfg,
    ) -> Result<Session<Body>, GateError> {
        let t0 = Instant::now();
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        let args = OpenArgs {
            place: Place::Bp.name(),
            machine,
            ctx: CTX,
            mode: StepMode::Graph,
            cfg: cfg.clone(),
        };
        let s = open_pair(file, args, &mut Quiet)?.ok_or("the open planned nothing")?;
        let m = s.model();
        let tier = m
            .body(NAME)?
            .hybrid()
            .tiers()
            .first()
            .ok_or("the (b′) load holds no tier card")?;
        let tier_device = tier.gpu().device_name()?;
        if !tier_device.contains(workstation::RTX_3090.name) {
            return Err(format!(
                "the tier opened on {tier_device}, not the {}",
                workstation::RTX_3090.name
            )
            .into());
        }
        println!(
            "load in {:.1} s: stage {} ({} B), tier {} experts on {tier_device} ({} B)",
            t0.elapsed().as_secs_f64(),
            m.gpu().device_name()?,
            m.resident_bytes(),
            tier.set().experts(),
            tier.weights().resident_bytes(),
        );
        Ok(s)
    }

    /// The reference: `plan`, whose stage card holds the union
    /// ([`union_plan`]), opened on card 0 with no tier.
    fn open_union(
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        cfg: &GlmCfg,
    ) -> Result<Session<Body>, GateError> {
        let t0 = Instant::now();
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        let mut model = Body::open_placed_lanes(
            file,
            plan,
            inputs,
            0,
            cfg.host,
            Residency::Off,
            KdaLanes::Two,
        )?;
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

    /// Every token, kept count and logits row a leg read, in order.
    #[derive(Default)]
    struct Leg {
        tokens: Vec<u32>,
        logits: Vec<Vec<f32>>,
        kept: Vec<usize>,
    }

    /// What a load's three legs read, and what they sent the tier.
    struct Run {
        step: Leg,
        pair: Leg,
        call: Leg,
        stage_nodes: usize,
        /// Pair passes that kept their draft, and that rejected it.
        accepts: usize,
        rejects: usize,
        /// What the decode legs together and the call alone sent the tier;
        /// `None` without one.
        sent_decode: Option<Sent>,
        sent_call: Option<Sent>,
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
    fn tier_counts(m: &Glm5nextModel) -> Result<Option<Sent>, GateError> {
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

    /// The counters of two spans together.
    fn plus(a: Option<Sent>, b: Option<Sent>) -> Option<Sent> {
        let (a, b) = (a?, b?);
        Some(Sent {
            first: a.first,
            hits: a.hits.iter().zip(&b.hits).map(|(a, b)| a + b).collect(),
            stats: TierBatchStats {
                served: a.stats.served + b.stats.served,
                settles: a.stats.settles + b.stats.settles,
                settle_early: a.stats.settle_early + b.stats.settle_early,
                settle_ns: a.stats.settle_ns + b.stats.settle_ns,
            },
        })
    }

    /// One decode step of `id`, its token and logits into `leg`.
    fn step(m: &mut Glm5nextModel, id: u32, leg: &mut Leg) -> Result<u32, GateError> {
        let tok = m.step(&[id])?;
        leg.tokens.push(tok);
        leg.logits.push(m.logits()?);
        Ok(tok)
    }

    /// The three legs (module header).
    fn legs(s: &mut Session<Body>, prompt: &[u32]) -> Result<Run, GateError> {
        let m = s.model_mut();
        let mut leg = Leg::default();
        m.reset()?;
        let before = tier_counts(m)?;
        let mut tok = 0;
        for &id in &prompt[..PROMPT] {
            tok = step(m, id, &mut leg)?;
        }
        let stage_nodes = m.step_graph_nodes()?.len();
        // The greedy run from the prompt's last token: g[0] the prompt's
        // argmax, g[i + 1] the argmax after g[i].
        let mut greedy = vec![tok];
        for _ in 0..STEPS {
            tok = step(m, tok, &mut leg)?;
            greedy.push(tok);
        }
        let sent_step = sent_between(before, tier_counts(m)?);
        let step_leg = leg;

        let mut leg = Leg::default();
        let (mut accepts, mut rejects) = (0, 0);
        m.reset()?;
        for &id in &prompt[..PROMPT] {
            m.step(&[id])?;
        }
        let before = tier_counts(m)?;
        let mut k = 0;
        for pass in 0..PASSES {
            let (last, next) = (greedy[k], greedy[k + 1]);
            let draft = if pass % 3 == 2 {
                (next + 1) % N_VOCAB
            } else {
                next
            };
            let at = m.pos();
            let rows = m.step_rows::<2>([last, draft])?;
            let [a, b] = m.rows_logits::<2>()?;
            let kept = if rows[0] == draft { 2 } else { 1 };
            m.rollback(at + u32::try_from(kept)?)?;
            m.keep_rows(kept, PassKind::Pair)?;
            leg.tokens.extend_from_slice(&rows[..kept]);
            leg.logits.push(a);
            leg.logits.push(b);
            leg.kept.push(kept);
            if kept == 2 {
                accepts += 1;
            } else {
                rejects += 1;
            }
            k += kept;
            if k + 1 >= greedy.len() {
                return Err(format!(
                    "pass {pass} reached greedy token {k} of {}; the step leg runs too few steps",
                    greedy.len()
                )
                .into());
            }
        }
        let sent_pair = sent_between(before, tier_counts(m)?);
        let pair_leg = leg;

        let mut leg = Leg::default();
        m.reset()?;
        let before = tier_counts(m)?;
        let tok = feed(m, &prompt[..CALL_PROMPT])?;
        let sent_call = sent_between(before, tier_counts(m)?);
        leg.tokens.push(tok);
        leg.logits.push(m.logits()?);
        Ok(Run {
            step: step_leg,
            pair: pair_leg,
            call: leg,
            stage_nodes,
            accepts,
            rejects,
            sent_decode: plus(sent_step, sent_pair),
            sent_call,
        })
    }

    /// Whether leg `got` is `want` bit for bit; prints the first difference.
    fn same(what: &str, want: &Leg, got: &Leg) -> bool {
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

    /// The precondition of leg `what`: it alone sent every layer of
    /// `tier_layers` at least one routed slot, and, when `batch`, the tier
    /// served its batch; prints its counters.
    fn sent_every(what: &str, sent: Option<&Sent>, tier_layers: &[usize], batch: bool) -> bool {
        let Some(sent) = sent else {
            println!("FAIL precondition {what}: the load holds no tier card");
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
            "{what} tier: {} batch services, {} settles ({} early, {:.2} ms waited), hits per \
             layer {:?}",
            st.served,
            st.settles,
            st.settle_early,
            st.settle_ns as f64 / 1e6,
            tier_layers.iter().map(|&l| hit(l)).collect::<Vec<_>>()
        );
        let served = !batch || st.served > 0;
        if missing.is_empty() && served {
            println!(
                "ok precondition {what}: every tier layer was sent a routed slot{}",
                if batch {
                    ", the tier served the batch"
                } else {
                    ""
                }
            );
            true
        } else {
            println!(
                "FAIL precondition {what}: tier layers {missing:?} were sent no routed slot, {} \
                 batch services; the leg proves nothing there",
                st.served
            );
            false
        }
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[HOST_POPULATE, HOST_LOCK, CARD_DONTNEED])?;
        let mut place_levers = PlanLevers::from_levers(&levers)?;
        place_levers.card_budget_bytes = Some(BUDGET);
        let cfg = GlmCfg {
            place: place_levers,
            host: levers.host(),
            prefill: PrefillMode::Batch,
        };
        let inputs =
            PlanInputs::read(&Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?)?;
        let bp = Place::Bp.machine(None, Some(place::tier_batch(&inputs.hp)))?;
        let layers = inputs.model.layers;
        let two = bp(layers);
        let plan = inputs.plan_lanes(&two, u64::try_from(CTX)?, &cfg.place, KdaLanes::Two)?;
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
            "plan (b′) under a card budget of {BUDGET} B: stage {} experts (n_l {:?}), tier {} \
             experts (per layer {:?}) on {} layers, host {} experts",
            plan.cards[0].experts,
            held(&plan.n_l),
            plan.cards.get(1).map_or(0, |c| c.experts),
            held(tier_n),
            tier_layers.len(),
            plan.host.experts
        );
        let free = PlanLevers::default();
        let unbudgeted = inputs.plan_lanes(&two, u64::try_from(CTX)?, &free, KdaLanes::One)?;
        println!(
            "plan (b′) with no card budget at ctx {} and one KDA lane (not loaded): stage {} \
             experts, tier {} experts, host {} experts; per layer stage {:?}, tier {:?}",
            CTX,
            unbudgeted.cards[0].experts,
            unbudgeted.cards.get(1).map_or(0, |c| c.experts),
            unbudgeted.host.experts,
            unbudgeted.n_l,
            unbudgeted.tier_n_l.first()
        );
        let hybrid: Vec<usize> = (0..layers)
            .filter(|&l| plan.n_l.get(l).is_some_and(|&n| n > 0))
            .collect();
        let mut pass = true;
        if hybrid == tier_layers {
            println!(
                "ok precondition sets: the tier holds experts on the stage's {} hybrid layers",
                hybrid.len()
            );
        } else {
            println!(
                "FAIL precondition sets: the tier holds experts on layers {tier_layers:?}, the \
                 stage's hybrid layers are {hybrid:?}"
            );
            pass = false;
        }
        let prompt = prompt()?;

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
            let mut s = open_union(&uplan, &inputs, &cfg)?;
            legs(&mut s, &prompt)?
        };
        let got = {
            let mut s = open_two(bp, &cfg)?;
            legs(&mut s, &prompt)?
        };
        pass &= same("decode bits: step", &want.step, &got.step);
        pass &= same("decode bits: pair", &want.pair, &got.pair);
        pass &= same("call bits", &want.call, &got.call);
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
        if got.accepts > 0 && got.rejects > 0 {
            println!(
                "ok precondition pair: the passes kept {} drafts and rejected {}",
                got.accepts, got.rejects
            );
        } else {
            println!(
                "FAIL precondition pair: the passes kept {} drafts and rejected {}: the pair clause \
                 needs both",
                got.accepts, got.rejects
            );
            pass = false;
        }
        pass &= sent_every("decode", got.sent_decode.as_ref(), &tier_layers, false);
        pass &= sent_every("call", got.sent_call.as_ref(), &tier_layers, true);
        if !pass {
            return Err(checks_failed());
        }
        println!("PASSED: {NAME}");
        Ok(())
    }
}
