//! GPU gate for the V4.1 expert tier (`bloomery_gpu::host::tier`), on one
//! card: the loopback. The stage and the tier both run on the gate card
//! (`workstation::plan_gate`), two `Gpu`s on it. The gate plans with a hot
//! list (`BLOOMERY_HOT_LIST`, refused unset), so each routed layer's card
//! list is its hottest `n_l`. The tiered plan moves the coldest [`K3`] of
//! them, ranks `n_l - K3 .. n_l`, from card 0 to device 1 (the tier) through
//! the placement's own split (`placement::routed_row`): the (b′) shape, the
//! stage the hottest ranks and the tier the next ones. The reference is the
//! gate plan itself, whose card holds the union — the same experts on one
//! card. Loads run one after the other, the reference first.
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
//!   lost card, the host tier is poisoned as a lost card, the next step and
//!   a reset are refused by name; the flag is then raised and both streams
//!   drain.
//! - `--two` (T7): a tiered plan with one expert on both devices is refused
//!   by name before anything is uploaded.

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

    use bloomery_gpu::hybrid::{Chain, PoisonKind};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::q4k_sel::QuantSel;
    use bloomery_gpu::{Fault, FaultSite, GpuError, HostFlags, Q8Act};
    use bloomery_gpu_deepseek41::body::{Body, BodyMeta, Deepseek41Model, OpenCfg, TierOpen};
    use bloomery_gpu_gates::{GateError, checks_failed, data_dir};
    use bloomery_levers::{
        CARD_BUDGET, CARD_DONTNEED, ENGRAM_HELPER, HOST_LOCK, HOST_POPULATE, HOT_LIST, R8,
    };
    use cuda_core::DeviceBuffer;
    use gguf::Split;
    use model::arch::deepseek41::place::PlanInputs;
    use model::placement::{self, Device, ExpertList, HotList, Plan, Role, Row, workstation};

    const NAME: &str = "gate_deepseek41_tier";
    /// Experts per routed layer the tier takes from the cold end of the gate
    /// plan's card list: the (b′) plan's tier depth, 873 tier experts over
    /// its 38 hybrid layers ≈ 23 a layer [derived, twoeng]. A carve from the
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
    /// The device index the tier's segments sit on in the tiered plan.
    const TIER_DEVICE: usize = 1;
    /// The go deadline and the grace a lost card is named within, plus room.
    const LOST_BOUND_S: f64 = 25.0;
    /// A refusal before any upload returns within this; a V4.1 upload alone
    /// takes longer.
    const REFUSE_BOUND_S: f64 = 20.0;

    struct Args {
        union: bool,
        fault: bool,
        lost: bool,
        two: bool,
    }

    fn parse_args() -> Result<Args, GateError> {
        const USAGE: &str = "usage: gate_deepseek41_tier [--union] [--fault] [--lost] [--two]";
        let mut a = Args {
            union: false,
            fault: false,
            lost: false,
            two: false,
        };
        for arg in std::env::args().skip(1) {
            match arg.as_str() {
                "--union" => a.union = true,
                "--fault" => a.fault = true,
                "--lost" => a.lost = true,
                "--two" => a.two = true,
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        if !(a.union || a.fault || a.lost || a.two) {
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

    /// `plan` with the coldest `k3` experts of each routed stack's card-0
    /// list by `hot`'s ranks moved to device [`TIER_DEVICE`]; a stack whose
    /// card list holds `k3` or fewer is refused by name. With `twice`, the
    /// first stack's tier list also keeps its first expert on card 0 (an
    /// expert on two devices). The host segments stay as they are, so the
    /// host set is the plan's.
    fn tiered<'a>(
        plan: &Plan<'a>,
        hot: &HotList,
        k3: usize,
        twice: bool,
    ) -> Result<Plan<'a>, GateError> {
        let model = plan.model;
        let mut out = plan.clone();
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
        let cfg = OpenCfg::from_levers(&levers)?;
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
        let tplan = tiered(&plan, hot, K3, false)?;
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
            let bad = tiered(&plan, hot, K3, true)?;
            let t0 = Instant::now();
            match open(split()?, &bad, Some(tier_open()), &meta) {
                Err(e)
                    if e.to_string().contains("placed twice")
                        && t0.elapsed().as_secs_f64() < REFUSE_BOUND_S =>
                {
                    println!(
                        "ok T7: an expert on two devices refused before any upload in {:.2} s: {e}",
                        t0.elapsed().as_secs_f64()
                    );
                }
                Err(e) => {
                    println!(
                        "FAIL T7: refused after {:.1} s (want the two-device rule within \
                         {REFUSE_BOUND_S} s): {e}",
                        t0.elapsed().as_secs_f64()
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
        let reference = if args.union || args.fault {
            let t0 = Instant::now();
            let mut m = open(split()?, &plan, None, &meta)?;
            println!(
                "reference load (gate plan, one card) in {:.1} s",
                t0.elapsed().as_secs_f64()
            );
            let r = union_run(&mut m, &prompt)?;
            drop(m);
            Some(r)
        } else {
            None
        };

        let t0 = Instant::now();
        let mut m = open(split()?, &tplan, Some(tier_open()), &meta)?;
        let (free, total) = m.gpu().mem_info()?;
        println!(
            "tiered load (stage + tier on {card}) in {:.1} s; card free {free} of {total} B after it",
            t0.elapsed().as_secs_f64()
        );

        if let Some(want) = &reference {
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
        {
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
        }
        let want_fault = Fault::at(u32::try_from(layer)?, FaultSite::QuantColumn);
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
        match m.step(&[prompt[5]]) {
            Err(e) => println!("ok T3: the next step is refused: {e}"),
            Ok(t) => {
                println!("FAIL T3: the next step gave token {t}");
                ok = false;
            }
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
