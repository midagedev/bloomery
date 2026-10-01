//! GPU gate for Qwen3.8's plan (b′) on both cards (`--place bp`): the stage
//! on the A6000 and its expert tier on the 3090 (`tier38`), against one card
//! holding the same expert sets. No MTP draft: a verify of each width the
//! draft's windows use (2, 3 and 4 rows) is driven directly.
//!
//! The plans are made under a card budget (`PlanLevers::card_budget_bytes`)
//! that caps both cards, so the A6000 can hold the stage's and the tier's
//! sets at once: the gate takes the largest budget of [`BUDGETS`] whose plan
//! puts tier experts on every layer, whose union fits the A6000 (its stage
//! bytes and the tier's expert and rounding bytes under the card's usable
//! bytes less its margin) and whose union counts a layer hold at most two
//! values (the ubatch walk's card route is built for two), and prints why each
//! larger one failed. The reference is that plan with every tier segment
//! joined to the stage card's (each routed stack's stage list, then its tier
//! list, in the stage segment's format; the host's untouched) on a machine
//! with no tier: one card, no tier code. The stage keeps each layer's id
//! prefix and the tier the next ids.
//!
//! A history: from a clear, the first [`PROMPT`] ids of the qwen4exp prose
//! corpus one decode step each, [`STEPS`] greedy steps, verifies at 4, 3, 2,
//! 4, 2 and 3 rows — the greedy token, then the corpus' next ids — each
//! committed at a fixed count, then [`STEPS`] greedy steps more; every
//! token and every logits row read.
//!
//! - `--union` (residency off): the tier load's history bit for bit the
//!   reference's, every token and logits row. Structure [derived]: the
//!   tier's stage step graph and each verify graph hold the reference's
//!   nodes less one a tier layer (its card leg drops the card sum, which the
//!   join takes after the wait); the tier's graph of the step and of each
//!   verify width holds eight nodes a tier layer (the go wait, the two copies
//!   in, the four launches, the signal) and the fault copy. Precondition:
//!   every tier layer was sent a routed slot by the steps.
//! - `--residency`: the tier load under `mid-p<P>-s1`, `P` half the plan's
//!   fewest stage experts a layer (the stage's count, not the union's): the
//!   history twice the same, flips landed; against the residency-off tier
//!   run, `gate_qwen38_residency`'s stream rule (the machine moves experts
//!   between the card and the host, whose sums run in another order): the
//!   prompt's last logits row within
//!   [`GREEDY_MARGIN`](bloomery_gpu_gates::GREEDY_MARGIN) of the off run's,
//!   the greedy ids equal or parted first at the on run's near tie;
//!   after it the tier's set is the live slot map's tier rows (`tier`), and
//!   no tier expert's file bytes are in the load's host set (`host`, by
//!   [`HostSet::holds`]). The union bit rule cannot hold here: the union's
//!   machine would move tier ids, and its pinned count is the union's.
//! - `--lost`: on the last load, the tier's stream held behind a host flag
//!   before a step — the tier stops signalling: within the go deadline and
//!   its grace the step fails naming the lost card, the host tier is
//!   poisoned as a lost card, and the next step at that position is refused
//!   by the recurrent stores (the delta layers ran it already); the flag is
//!   then raised, both streams drain, and a reset is refused by the lost
//!   card's poison, which a reset does not lift.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen38_twocard: built without the `gpu` feature; see `just gate-gpu-qwen38-twocard`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen38_twocard", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use std::path::Path;
    use std::time::Instant;

    use app::Session;
    use bloomery_gpu::HostFlags;
    use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for;
    use bloomery_gpu::arch::qwen3moe::{Body38, Qwen38Model};
    use bloomery_gpu::host::swap::{PassReport, Residency};
    use bloomery_gpu::host::tier::{TierAct, TierCard, TierSet};
    use bloomery_gpu::host::{PassKind, PoisonKind};
    use bloomery_gpu::hybrid::Chain;
    use bloomery_gpu_gates::{GREEDY_MARGIN, GateError, checks_failed, data_dir, verdict};
    use bloomery_levers::{CARD_DONTNEED, HOST_LOCK, HOST_POPULATE, HostCfg};
    use gguf::Split;
    use model::arch::qwen35moe::place::{Experts, PlanInputs, machine_bp, tier_batch};
    use model::placement::host_lock::{HostFile, HostSet};
    use model::placement::workstation::RTX_3090;
    use model::placement::{Device, ExpertList, Machine, Plan, PlanLevers, Role, Row, Segment};
    use refset::arch::qwen4exp::MODEL;
    use runtime::{Out, Target, Verify, Want};

    const NAME: &str = "gate_qwen38_twocard";
    /// The stores' positions, `gate_qwen38_residency`'s.
    const CTX: usize = 3072;
    /// Prompt positions, fed one decode step each.
    const PROMPT: usize = 32;
    /// Greedy steps before the verifies and after them.
    const STEPS: usize = 24;
    /// The card budgets tried, largest first, GiB.
    const BUDGETS: [u64; 10] = [26, 25, 24, 23, 22, 21, 20, 19, 18, 16];
    /// The go deadline and the grace a lost card is named within, plus room.
    const LOST_BOUND_S: f64 = 25.0;
    /// Nodes of a tier graph a tier layer: the go wait, the activation's and
    /// the places' copies in, the card leg's four launches, the signal.
    const TIER_LAYER_NODES: usize = TierCard::nodes_per_layer(TierAct::F32, 4);

    struct Args {
        union: bool,
        residency: bool,
        lost: bool,
    }

    fn parse_args() -> Result<Args, GateError> {
        const USAGE: &str = "usage: gate_qwen38_twocard [--union] [--residency] [--lost]";
        let mut a = Args {
            union: false,
            residency: false,
            lost: false,
        };
        for arg in std::env::args().skip(1) {
            match arg.as_str() {
                "--union" => a.union = true,
                "--residency" => a.residency = true,
                "--lost" => a.lost = true,
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        if !(a.union || a.residency || a.lost) {
            return Err(USAGE.into());
        }
        Ok(a)
    }

    /// The first `n` ids of `$BLOOMERY_DATA/qwen4exp/corpus-prose.ids`.
    fn prose38(n: usize) -> Result<Vec<u32>, GateError> {
        let path = data_dir().join("qwen4exp").join("corpus-prose.ids");
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

    /// The tier's device index in the plan's cards.
    fn tier_device(plan: &Plan<'_>) -> Device {
        Device::Card(plan.machine.cards.len())
    }

    /// Why plan (b′) under a budget cannot be the gate's: no tier expert on
    /// some layer, a union past the A6000, or more than two union counts.
    fn unfit(plan: &Plan<'_>) -> Option<String> {
        let tier = plan.tier_n_l.first()?;
        if let Some(l) = tier.iter().position(|&n| n == 0) {
            return Some(format!("layer {l} holds no tier expert"));
        }
        let (stage, t) = (&plan.cards[0], plan.cards.get(1)?);
        let card = &plan.machine.cards[0];
        let union = stage.dense_bytes
            + stage.expert_bytes
            + stage.rounding_bytes
            + stage.kv_bytes
            + stage.scratch_bytes
            + stage.context_bytes
            + stage.reserve_bytes
            + t.expert_bytes
            + t.rounding_bytes;
        let limit = card.usable_bytes.saturating_sub(card.margin_bytes);
        if union > limit {
            return Some(format!("the union takes {union} B of the A6000's {limit}"));
        }
        let mut counts: Vec<u64> = plan.n_l.iter().zip(tier).map(|(a, b)| a + b).collect();
        counts.sort_unstable();
        counts.dedup();
        (counts.len() > 2).then(|| format!("union counts {counts:?}: the card route takes two"))
    }

    /// `plan` with each routed stack's tier segment joined to its stage-card
    /// segment on card 0 — the stage's ids, then the tier's, in the stage
    /// segment's format — the host's segments as they are, on `machine`, a
    /// copy of the plan's with no tier: the stage card holds the union.
    fn union_plan<'a>(plan: &Plan<'a>, machine: &'a Machine) -> Result<Plan<'a>, GateError> {
        let model = plan.model;
        let tier = tier_device(plan);
        let mut out = plan.clone();
        out.machine = machine;
        out.cards.truncate(1);
        out.tier_n_l.clear();
        for row in &mut out.rows {
            let t = &model.tensors[row.tensor];
            if t.role != Role::RoutedExperts {
                continue;
            }
            let of = |d: Device| row.segments.iter().find(|s| s.device == d).cloned();
            let (Some(stage), Some(on_tier)) = (of(Device::Card(0)), of(tier)) else {
                continue;
            };
            if stage.format != on_tier.format {
                return Err(format!(
                    "{}: the stage's segment and the tier's are of two formats",
                    t.name
                )
                .into());
            }
            let ids = |s: &Segment| {
                s.experts
                    .as_ref()
                    .map(|e| e.ids().to_vec())
                    .unwrap_or_default()
            };
            let mut union = ids(&stage);
            union.extend(ids(&on_tier));
            let n = union.len();
            let joined = Segment {
                device: Device::Card(0),
                format: stage.format,
                experts: Some(ExpertList::new(union, model.experts)?),
                resident_bytes: stage.resident_bytes + on_tier.resident_bytes,
            };
            let segments = std::iter::once(joined)
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
        Ok(out)
    }

    /// A load of `plan` under `residency`.
    fn open(
        path: &Path,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        (host, ub): (HostCfg, usize),
        residency: Residency,
    ) -> Result<Session<Body38>, GateError> {
        let t0 = Instant::now();
        let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let m = Body38::open_placed_residency(file, plan, inputs, 0, host, ub, residency)?;
        let tiers = m.body(NAME)?.hybrid().tiers();
        let tier = match tiers.first() {
            Some(t) => format!(
                ", tier {} experts on {} ({} B)",
                t.set().experts(),
                t.gpu().device_name()?,
                t.weights().resident_bytes()
            ),
            None => ", no tier".to_string(),
        };
        println!(
            "load in {:.1} s: stage {} ({} B){tier}",
            t0.elapsed().as_secs_f64(),
            m.gpu().device_name()?,
            m.resident_bytes()
        );
        let mut s = Session::from_model(m, CTX as u32);
        s.model_mut().body_parts(NAME)?.2.log_residency(0);
        Ok(s)
    }

    /// Every token and logits row a history read, in order, its stage graph
    /// nodes (the step's, then each verify width's), and the flips that
    /// landed over it.
    #[derive(Default)]
    struct Run {
        tokens: Vec<u32>,
        logits: Vec<Vec<f32>>,
        stage_nodes: Vec<usize>,
        landed: usize,
    }

    /// The verify windows: their rows and the rows each keeps.
    const WINDOWS: [(usize, usize); 6] = [(4, 2), (3, 3), (2, 1), (4, 4), (2, 2), (3, 1)];

    /// One greedy step of `next`, its token and row into `run`.
    fn step(s: &mut Session<Body38>, run: &mut Run, next: u32) -> Result<u32, GateError> {
        let out = s.step(next, Want::Logits)?;
        let Out::Logits { argmax, row } = out else {
            return Err("a step asked for its logits gave the argmax alone".into());
        };
        run.tokens.push(argmax);
        run.logits.push(row.to_vec());
        Ok(argmax)
    }

    /// A verify of `M` rows — `next`, then `fill`'s ids — committed at
    /// `keep`: every row's token and logits into `run`; the next token, the
    /// last kept row's argmax.
    fn window<const M: usize>(
        s: &mut Session<Body38>,
        run: &mut Run,
        next: u32,
        fill: &[u32],
        keep: usize,
    ) -> Result<u32, GateError> {
        let mut rows = [next; M];
        rows[1..].copy_from_slice(&fill[..M - 1]);
        let out = s.verify::<M>(rows)?;
        run.tokens.extend_from_slice(&out);
        run.logits.extend(s.model().rows_logits::<M>()?);
        s.commit(keep)?;
        Ok(out[keep - 1])
    }

    /// The history (module doc) from a clear.
    fn history(s: &mut Session<Body38>, ids: &[u32]) -> Result<Run, GateError> {
        s.clear()?;
        take_passes(s)?;
        let mut run = Run::default();
        let mut next = 0;
        for &id in &ids[..PROMPT] {
            next = step(s, &mut run, id)?;
        }
        for _ in 0..STEPS {
            next = step(s, &mut run, next)?;
        }
        let mut at = PROMPT;
        for (m, keep) in WINDOWS {
            let fill = &ids[at..];
            next = match m {
                2 => window::<2>(s, &mut run, next, fill, keep)?,
                3 => window::<3>(s, &mut run, next, fill, keep)?,
                _ => window::<4>(s, &mut run, next, fill, keep)?,
            };
            at += m;
        }
        for _ in 0..STEPS {
            next = step(s, &mut run, next)?;
        }
        let m = s.model();
        run.stage_nodes = vec![
            m.step_graph_nodes()?.len(),
            m.rows_graph_nodes::<2>()?.len(),
            m.rows_graph_nodes::<3>()?.len(),
            m.rows_graph_nodes::<4>()?.len(),
        ];
        run.landed = take_passes(s)?.iter().map(|(_, r)| r.landed).sum();
        Ok(run)
    }

    /// The boundaries' reports since the last take.
    fn take_passes(s: &mut Session<Body38>) -> Result<Vec<(PassKind, PassReport)>, GateError> {
        let (_, _, b) = s.model_mut().body_parts(NAME)?;
        Ok(b.take_residency_passes())
    }

    /// Whether `got` is `want` bit for bit; prints the first difference.
    fn same(what: &str, want: &Run, got: &Run) -> bool {
        if want.tokens != got.tokens {
            let at = want
                .tokens
                .iter()
                .zip(&got.tokens)
                .position(|(a, b)| a != b);
            println!(
                "FAIL {what}: tokens apart at {at:?} ({} and {} tokens)",
                got.tokens.len(),
                want.tokens.len()
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
        let ok = want.logits.len() == got.logits.len();
        println!(
            "{} {what}: {} tokens, {} logits rows bit for bit the reference's",
            if ok { "ok" } else { "FAIL" },
            got.tokens.len(),
            got.logits.len()
        );
        ok
    }

    /// The top-1 margin of `row`: its largest value less its second.
    fn margin(row: &[f32]) -> f32 {
        let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &v in row {
            if v > a {
                b = a;
                a = v;
            } else if v > b {
                b = v;
            }
        }
        a - b
    }

    /// The tier layers: the plan's layers with a tier expert.
    fn tier_layers(plan: &Plan<'_>) -> Vec<usize> {
        plan.tier_n_l
            .first()
            .map(|t| (0..t.len()).filter(|&l| t[l] > 0).collect())
            .unwrap_or_default()
    }

    /// `--union`'s structure and precondition lines over the tier load `s`
    /// and the two runs.
    fn structure(
        s: &Session<Body38>,
        want: &Run,
        got: &Run,
        layers: &[usize],
    ) -> Result<bool, GateError> {
        let t = layers.len();
        let stage: Vec<usize> = want.stage_nodes.iter().map(|n| n - t).collect();
        let mut ok = got.stage_nodes == stage;
        println!(
            "structure: the stage graphs of the step and the 2, 3, 4-row verifies hold {:?} nodes, \
             the reference's {:?} less one a tier layer ({t}): {}",
            got.stage_nodes,
            want.stage_nodes,
            verdict(ok)
        );
        let body = s.model().body(NAME)?;
        let tier = body
            .hybrid()
            .tiers()
            .first()
            .ok_or("the load holds no tier")?;
        let want_tier = Some(TIER_LAYER_NODES * t + 1);
        let graphs: Vec<Option<usize>> =
            [Chain::Step, Chain::Cols(2), Chain::Cols(3), Chain::Cols(4)]
                .iter()
                .map(|&c| tier.graph_nodes(c))
                .collect();
        let graphs_ok = graphs.iter().all(|&g| g == want_tier);
        println!(
            "structure: the tier's graphs of the step and the 2, 3, 4-column verifies hold {graphs:?} \
             nodes, {TIER_LAYER_NODES} a tier layer and the fault copy ({want_tier:?}): {}",
            verdict(graphs_ok)
        );
        ok &= graphs_ok;
        let st = tier.stats();
        let first = tier.set().layers().start;
        let hits = |l: usize| st.layer_hits.get(l - first).copied().unwrap_or(0);
        let missing: Vec<usize> = layers.iter().copied().filter(|&l| hits(l) == 0).collect();
        println!(
            "tier: {} layers asked, {} settles ({} early), step hits per tier layer {:?}",
            st.issued,
            st.settles,
            st.settle_early,
            layers.iter().map(|&l| hits(l)).collect::<Vec<_>>()
        );
        let pre = missing.is_empty();
        println!(
            "precondition: every tier layer was sent a routed slot by the steps (missing \
             {missing:?}): {}",
            verdict(pre)
        );
        Ok(ok && pre)
    }

    /// `--residency`'s clauses (module doc) over the tier load `s` under the
    /// machine, `off` the residency-off tier run, `split` the file and
    /// `plan` the tier load's plan.
    fn residency_clauses(
        s: &mut Session<Body38>,
        ids: &[u32],
        off: Option<&Run>,
        (split, plan): (&Split, &Plan<'_>),
    ) -> Result<bool, GateError> {
        let first = history(s, ids)?;
        let again = history(s, ids)?;
        let det = first.tokens == again.tokens
            && first.logits.len() == again.logits.len()
            && first
                .logits
                .iter()
                .zip(&again.logits)
                .all(|(a, b)| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
        let flips = first.landed > 0 && again.landed > 0;
        println!(
            "residency: the history twice the same {det}, flips landed {} and {}: {}",
            first.landed,
            again.landed,
            verdict(det && flips)
        );
        let mut ok = det && flips;
        if let Some(off) = off {
            // gate_qwen38_residency's stream rule: the prompt's last logits
            // row within GREEDY_MARGIN of the off run's, the greedy ids equal
            // or parted first at the on run's near tie.
            let last = PROMPT - 1;
            let row_diff = |i: usize| {
                off.logits[i]
                    .iter()
                    .zip(&first.logits[i])
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max)
            };
            let dlogit = row_diff(last);
            let parted = off
                .tokens
                .iter()
                .zip(&first.tokens)
                .position(|(a, b)| a != b);
            let near = parted.is_none_or(|i| margin(&first.logits[i]) < GREEDY_MARGIN);
            let near_ok = near
                && dlogit < GREEDY_MARGIN
                && off.tokens.len() == first.tokens.len()
                && off.logits.len() == first.logits.len();
            let (worst, worst_diff) = (0..off.logits.len().min(first.logits.len()))
                .map(|i| (i, row_diff(i)))
                .fold((0, 0.0f32), |a, b| if b.1 > a.1 { b } else { a });
            println!(
                "residency: against the residency-off tier run, the prompt's last logits row \
                 max|diff| {dlogit} (under {GREEDY_MARGIN}), the ids parted at {parted:?}{} (a \
                 near tie {near}): {}; every row's largest max|diff| {worst_diff} at row {worst} \
                 (not a clause)",
                parted
                    .map(|i| format!(", the on run's margin {}", margin(&first.logits[i])))
                    .unwrap_or_default(),
                verdict(near_ok)
            );
            ok &= near_ok;
        }
        let body = s.model().body(NAME)?;
        let hybrid = body.hybrid();
        let tier = hybrid.tiers().first().ok_or("the load holds no tier")?;
        let live = TierSet::of_map(hybrid.slots(), 0)?;
        let set_ok = &live == tier.set();
        println!(
            "tier: after the histories the live slot map's tier rows are the tier card's set \
             ({} experts): {}",
            tier.set().experts(),
            verdict(set_ok)
        );
        ok &= set_ok;
        let host_ok = no_tier_bytes(s, split, plan)?;
        Ok(ok && host_ok)
    }

    /// `host`: no tier expert's file bytes in the load's host set.
    fn no_tier_bytes(
        s: &mut Session<Body38>,
        split: &Split,
        plan: &Plan<'_>,
    ) -> Result<bool, GateError> {
        let tier = tier_device(plan);
        let set: HostSet = s
            .model_mut()
            .host_residency()
            .ok_or("the load holds no host set")?
            .set()
            .clone();
        let (mut checked, mut held) = (0usize, Vec::new());
        for row in &plan.rows {
            let t = &plan.model.tensors[row.tensor];
            for seg in row.segments.iter().filter(|s| s.device == tier) {
                let (sh, info) = split
                    .find(&t.name)
                    .ok_or_else(|| format!("{} is not in the file", t.name))?;
                let base = split
                    .shard(sh)
                    .ok_or_else(|| format!("{}: no shard {sh}", t.name))?
                    .data_base()
                    + info.offset;
                for span in seg.spans(t, plan.model.experts)? {
                    checked += 1;
                    let at = base + span.bytes.start..base + span.bytes.end;
                    if set.holds(&HostFile::Shard(sh), &at) {
                        held.push(format!("{} bytes {at:?}", t.name));
                    }
                }
            }
        }
        let ok = checked > 0 && held.is_empty();
        println!(
            "host: {checked} tier runs checked, {} in the load's host set{}: {}",
            held.len(),
            held.first()
                .map(|h| format!(" (first {h})"))
                .unwrap_or_default(),
            verdict(ok)
        );
        Ok(ok)
    }

    /// `--lost`: the tier's stream held behind a host flag before a step.
    fn lost_case(m: &mut Qwen38Model, ids: &[u32]) -> Result<bool, GateError> {
        m.reset()?;
        m.step(&ids[..4])?;
        let flags = {
            let body = m.body(NAME)?;
            let tier = body.hybrid().tiers().first().ok_or("no tier card")?;
            let flags = HostFlags::new(tier.gpu().context(), 1)?;
            flags.enqueue_wait(tier.gpu().stream(), 0)?;
            flags
        };
        let t0 = Instant::now();
        let r = m.step(&[ids[4]]);
        let secs = t0.elapsed().as_secs_f64();
        let mut ok = match &r {
            Err(e)
                if e.to_string().contains("the card is lost")
                    && e.to_string().contains(RTX_3090.name)
                    && secs < LOST_BOUND_S =>
            {
                println!("ok lost: the step fails in {secs:.1} s naming the lost card: {e}");
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
        // The delta layers ran the failed step's recurrence already, so the
        // next step at its position is refused before the tier is asked.
        match m.step(&[ids[4]]) {
            Err(e)
                if e.to_string()
                    .contains("failed after its chain was launched") =>
            {
                println!("ok lost: the next step is refused by the recurrent stores: {e}");
            }
            other => {
                println!(
                    "FAIL lost: the next step gave {other:?}, want the recurrent stores' refusal"
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
        // A reset does not lift a lost card's poison: it is refused by name.
        match m.reset() {
            Err(e)
                if e.to_string().contains("the card is lost")
                    && e.to_string().contains("reload the model") =>
            {
                println!("ok lost: a reset is refused by the lost card's poison: {e}");
            }
            other => {
                println!("FAIL lost: a reset gave {other:?}, want the lost card's refusal");
                ok = false;
            }
        }
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[HOST_POPULATE, HOST_LOCK, CARD_DONTNEED])?;
        let host = levers.host();
        let args = parse_args()?;
        let path = Path::new(MODEL);
        let split = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::describe(&split)?;
        let ub = ubatch_for(CTX)?;
        let layers = inputs.spec.layers.len();
        let bp = machine_bp(layers, ub as u64, None, tier_batch(&inputs.hp, ub as u64));
        let mut chosen = None;
        for gib in BUDGETS {
            let levers = PlanLevers {
                card_budget_bytes: Some(gib << 30),
            };
            let plan = inputs
                .plan_with(&bp, CTX as u64, &levers, Experts::Card)
                .map_err(|e| format!("plan (b′) under {gib} GiB: {e}"))?;
            match unfit(&plan) {
                None => {
                    chosen = Some((gib, levers));
                    break;
                }
                Some(why) => println!("budget {gib} GiB: {why}; next"),
            }
        }
        let (gib, levers) = chosen.ok_or("no budget of BUDGETS fits the gate")?;
        let plan = inputs.plan_with(&bp, CTX as u64, &levers, Experts::Card)?;
        let tl = tier_layers(&plan);
        let held = |v: &[u64]| {
            (
                v.iter().copied().min().unwrap_or(0),
                v.iter().copied().max().unwrap_or(0),
            )
        };
        println!(
            "plan (b′) under a card budget of {gib} GiB: stage {} experts (a layer {:?}), tier {} \
             experts (a layer {:?}) on {} layers, host {} experts",
            plan.cards[0].experts,
            held(&plan.n_l),
            plan.cards.get(1).map_or(0, |c| c.experts),
            held(plan.tier_n_l.first().map_or(&[][..], Vec::as_slice)),
            tl.len(),
            plan.host.experts
        );
        let ids = prose38(PROMPT + 32)?;
        let mut pass = true;
        let mut last: Option<Session<Body38>> = None;
        let mut off_run = None;
        if args.union {
            let mut um = bp.clone();
            um.tiers.clear();
            let uplan = union_plan(&plan, &um)?;
            println!(
                "reference: the A6000 holds {} experts, the union of the stage's and the tier's",
                uplan.n_l.iter().sum::<u64>()
            );
            let mut r = open(path, &uplan, &inputs, (host, ub), Residency::Off)?;
            if !r.model().body(NAME)?.hybrid().tiers().is_empty() {
                return Err("the reference load holds a tier card".into());
            }
            let want = history(&mut r, &ids)?;
            drop(r);
            let mut t = open(path, &plan, &inputs, (host, ub), Residency::Off)?;
            let got = history(&mut t, &ids)?;
            pass &= same("two cards", &want, &got);
            pass &= structure(&t, &want, &got, &tl)?;
            off_run = Some(got);
            last = Some(t);
        }
        if args.residency {
            drop(last.take());
            let slots = plan
                .n_l
                .iter()
                .copied()
                .filter(|&n| n > 0)
                .min()
                .unwrap_or(0) as usize;
            let residency = Residency::Mid {
                pinned: slots / 2,
                spares: 1,
            };
            println!(
                "residency: mid-p{}-s1 over the plan's {slots} stage experts a layer at least",
                slots / 2
            );
            let mut t = open(path, &plan, &inputs, (host, ub), residency)?;
            pass &= residency_clauses(&mut t, &ids, off_run.as_ref(), (&split, &plan))?;
            last = Some(t);
        }
        if args.lost {
            let mut s = match last.take() {
                Some(s) => s,
                None => open(path, &plan, &inputs, (host, ub), Residency::Off)?,
            };
            pass &= lost_case(s.model_mut(), &ids)?;
        }
        if !pass {
            return Err(checks_failed());
        }
        println!("PASSED: {NAME}");
        Ok(())
    }
}
