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
//!   reference's, every token and logits row; then, from a clear, the first
//!   [`PROMPT_U`] ids as one prompt call (one ubatch walk, the tier serving
//!   its slots through the batch port) and [`PROMPT_STEPS`] greedy steps: the
//!   prompt's last logits row, every step's row and every greedy id bit for
//!   bit the reference's; the tier's batch services served every tier layer
//!   (precondition); then a [`PROMPT_SHORT`]-id prompt call the same way, the
//!   tier serving at least one layer — a tier load runs every prompt by
//!   ubatches — and an explicit pass prompt call refused by name before
//!   anything moves. Each precondition line prints the host's wall waiting
//!   for the tier, not a clause. Structure [derived]: the
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
//!   then the [`PROMPT_U`]-id prompt call with host streaming on against
//!   the residency-off tier run's by the same stream rule (the call's picks
//!   move host experts onto the stage card only); after it the tier's set
//!   is the live slot map's tier rows (`tier`), and no tier expert's file
//!   bytes are in the load's host set (`host`, by [`HostSet::holds`]). The
//!   union bit rule cannot hold here: the union's machine would move tier
//!   ids, and its pinned count is the union's.
//! - `--prompt4k` (residency off): plan (b′) and its union reference at
//!   [`CTX_4K`] positions, whose ubatch is the full [`PROMPT_4K`] window;
//!   from a clear the first [`PROMPT_4K`] ids as one prompt call (one ubatch
//!   of two card-route runs, the tier's block route over both) and
//!   [`PROMPT_STEPS`] greedy steps, bit for bit the reference's as under
//!   `--union`.
//! - `--slots` (residency off): the plan of two sequences through the slot
//!   harness (`slots_gate`: H1 interleave, H2 one pass, H3 bytes, H4 reset,
//!   H6 refusals, H7 captures; two disjoint [`PROMPT_U`]-id prose windows,
//!   both by the ubatch path the tier load serves), then, on the tier load
//!   and its union reference of the same plan, each slot's prompt and the
//!   windows 4+4, 4+3, 3+4, 4+2 and 3+2 rows as one `verify_slots` pass of
//!   both slots, committed at fixed counts, and [`STEPS`] rounds of one row
//!   a slot: every row's id, each slot's state hash after the history and
//!   its next step's logits row bit for bit the reference's (a); each slot's
//!   rows its solo run's on the tier load, the same windows as verifies of
//!   that slot alone, the other parked (b); and the tier's graphs of the
//!   history's widths, 5 to 8 columns, eight nodes a tier layer and the
//!   fault copy (c).
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
    use bloomery_gpu::arch::qwen3moe::{Body38, Prompt38, Qwen38Model, Store38Host};
    use bloomery_gpu::host::swap::{PassReport, Residency};
    use bloomery_gpu::host::tier::{TierAct, TierCard, TierSet};
    use bloomery_gpu::host::{PassKind, PoisonKind};
    use bloomery_gpu::hybrid::Chain;
    use bloomery_gpu_gates::slots_gate::{self, Derived, Launches, PassAdapter, SlotsAdapter};
    use bloomery_gpu_gates::{Fnv1a64, GREEDY_MARGIN, GateError, checks_failed, data_dir, verdict};
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
    /// The prompt call's ids under `--union` and `--residency`: one ubatch
    /// at [`CTX`].
    const PROMPT_U: usize = 600;
    /// `--union`'s short prompt call: under the pass walk's eight rows, which
    /// a tier load runs as one ubatch (`Body38::resolve_prompt`).
    const PROMPT_SHORT: usize = 5;
    /// The greedy steps after a prompt call.
    const PROMPT_STEPS: usize = 8;
    /// `--prompt4k`'s stores and prompt: the full ubatch window, with room
    /// for the steps after it.
    const CTX_4K: usize = 4352;
    const PROMPT_4K: usize = 4096;
    /// The card budgets tried, largest first, GiB.
    const BUDGETS: [u64; 10] = [26, 25, 24, 23, 22, 21, 20, 19, 18, 16];
    /// The go deadline and the grace a lost card is named within, plus room.
    const LOST_BOUND_S: f64 = 25.0;
    /// Nodes of a tier graph a tier layer: the go wait, the activation's and
    /// the places' copies in, the card leg's four launches, the signal.
    const TIER_LAYER_NODES: usize = TierCard::nodes_per_layer(TierAct::F32, 4);
    /// The f16 pattern every selecting store's planes hold before a run
    /// (100.0): `reset` leaves those rows as they were, so a path that reads
    /// a row it did not write must read this, not a previous run's value —
    /// gate_qwen4exp_e2e's `PLANE_FILL`, the same rule.
    const PLANE_FILL: u16 = 0x5640;
    /// The model's delta and selecting layers, the e2e gate's terms of the
    /// launches a slot binds.
    const N_GDN: usize = 36;
    const N_QSA: usize = 12;
    /// The model's streams, hidden width and pass rows, the e2e gate's terms
    /// of a sequence's arena rows.
    const STREAMS: usize = 4;
    const HIDDEN: usize = 2560;
    const PASS_ROWS: usize = 8;
    /// PIN(2026-10-06): the nodes one more busy slot adds to a captured pass
    /// of several slots' rows, derived before the pass was built
    /// (gate_qwen4exp_e2e's `NODES_SLOT`, the same derivation): the
    /// launches bound to one sequence — the embedding row, the PLE site's
    /// conv on layer 1, each delta layer's conv and delta step, each
    /// selecting layer's 8 — and the copy of a parked slot's rows into its
    /// own: 1 + 1 + 36·2 + 12·8 + 1.
    const NODES_SLOT: usize = 171;
    /// The verify windows of `--slots`'s history: each pass's rows a slot.
    const SLOT_WINDOWS: [(usize, usize); 5] = [(4, 4), (4, 3), (3, 4), (4, 2), (3, 2)];
    /// Each [`SLOT_WINDOWS`] pass's kept rows a slot, fixed counts.
    const SLOT_KEEPS: [(usize, usize); 5] = [(2, 3), (4, 1), (1, 2), (3, 2), (2, 1)];

    const _: () = assert!(NODES_SLOT == 1 + 1 + N_GDN * 2 + N_QSA * 8 + 1);

    struct Args {
        union: bool,
        residency: bool,
        lost: bool,
        prompt4k: bool,
        slots: bool,
    }

    fn parse_args() -> Result<Args, GateError> {
        const USAGE: &str =
            "usage: gate_qwen38_twocard [--union] [--residency] [--lost] [--prompt4k] [--slots]";
        let mut a = Args {
            union: false,
            residency: false,
            lost: false,
            prompt4k: false,
            slots: false,
        };
        for arg in std::env::args().skip(1) {
            match arg.as_str() {
                "--union" => a.union = true,
                "--residency" => a.residency = true,
                "--lost" => a.lost = true,
                "--prompt4k" => a.prompt4k = true,
                "--slots" => a.slots = true,
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        if !(a.union || a.residency || a.lost || a.prompt4k || a.slots) {
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

    /// A load of `plan` under `residency` for `slots` sequences, its session
    /// over `ctx` positions.
    fn open(
        path: &Path,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        (host, ub, ctx): (HostCfg, usize, usize),
        residency: Residency,
        slots: usize,
    ) -> Result<Session<Body38>, GateError> {
        let t0 = Instant::now();
        let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let m = Body38::open_placed_residency(file, plan, inputs, 0, host, ub, residency, slots)?;
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
        let mut s = Session::from_model(m, u32::try_from(ctx)?);
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
        (ids, prompt_ids): (&[u32], &[u32]),
        (off, off_prompt): (Option<&Run>, Option<&Run>),
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
        if let Some(off) = off_prompt {
            let on = prompt_run(s, prompt_ids, true)?;
            ok &= stream_rule(&format!("a {PROMPT_U}-id prompt call streaming"), off, &on);
        }
        let body = s.model().body(NAME)?;
        let hybrid = body.hybrid();
        let tier = hybrid.tiers().first().ok_or("the load holds no tier")?;
        let live = TierSet::of_map(hybrid.slots(), 0)?;
        let set_ok = &live == tier.set();
        println!(
            "tier: after the histories and the prompt call the live slot map's tier rows are the \
             tier card's set ({} experts): {}",
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

    /// Plan (b′) on `bp` at `ctx` positions under the largest budget of
    /// [`BUDGETS`] that fits the gate ([`unfit`]), printed with why each
    /// larger one did not.
    fn chosen_plan<'a>(
        inputs: &'a PlanInputs,
        bp: &'a Machine,
        ctx: usize,
    ) -> Result<Plan<'a>, GateError> {
        chosen_plan_slots(inputs, bp, ctx, 1)
    }

    /// [`chosen_plan`] for a load that serves `slots` sequences
    /// (`PlanInputs::plan_with_slots`; at 1 the plan itself): two sequences'
    /// stores shrink the union's room on the A6000, so the search runs on
    /// the plan it picks.
    fn chosen_plan_slots<'a>(
        inputs: &'a PlanInputs,
        bp: &'a Machine,
        ctx: usize,
        slots: usize,
    ) -> Result<Plan<'a>, GateError> {
        for gib in BUDGETS {
            let levers = PlanLevers {
                card_budget_bytes: Some(gib << 30),
            };
            let plan = if slots == 1 {
                inputs.plan_with(bp, ctx as u64, &levers, Experts::Card)
            } else {
                inputs.plan_with_slots(bp, ctx as u64, &levers, Experts::Card, slots)
            }
            .map_err(|e| format!("plan (b′) under {gib} GiB: {e}"))?;
            if let Some(why) = unfit(&plan) {
                println!("budget {gib} GiB at {ctx} positions: {why}; next");
                continue;
            }
            let held = |v: &[u64]| {
                (
                    v.iter().copied().min().unwrap_or(0),
                    v.iter().copied().max().unwrap_or(0),
                )
            };
            println!(
                "plan (b′) of {slots} at {ctx} positions under a card budget of {gib} GiB: stage \
                 {} experts (a layer {:?}), tier {} experts (a layer {:?}) on {} layers, host {} \
                 experts",
                plan.cards[0].experts,
                held(&plan.n_l),
                plan.cards.get(1).map_or(0, |c| c.experts),
                held(plan.tier_n_l.first().map_or(&[][..], Vec::as_slice)),
                tier_layers(&plan).len(),
                plan.host.experts
            );
            return Ok(plan);
        }
        Err(format!("no budget of BUDGETS fits the gate at {ctx} positions").into())
    }

    /// From a clear, `ids` as one prompt call (the session's path: ubatches
    /// for a prompt this long) with the call's host streaming `stream`, then
    /// [`PROMPT_STEPS`] greedy steps: the prompt's last logits row and its
    /// argmax first, then every step's.
    fn prompt_run(s: &mut Session<Body38>, ids: &[u32], stream: bool) -> Result<Run, GateError> {
        s.clear()?;
        take_passes(s)?;
        s.model_mut().body_parts(NAME)?.2.set_hoststream(stream)?;
        let mut run = Run::default();
        let out = s.prompt(ids, Want::Logits)?;
        let Out::Logits { argmax, row } = out else {
            return Err("a prompt call asked for its logits gave the argmax alone".into());
        };
        run.tokens.push(argmax);
        run.logits.push(row.to_vec());
        let mut next = argmax;
        for _ in 0..PROMPT_STEPS {
            next = step(s, &mut run, next)?;
        }
        s.model_mut().body_parts(NAME)?.2.set_hoststream(false)?;
        run.landed = take_passes(s)?.iter().map(|(_, r)| r.landed).sum();
        Ok(run)
    }

    /// From a clear, `ids` as one explicit ubatch prompt call
    /// ([`Prompt38::Gemm`]: the reference load would take the pass below
    /// eight ids) and [`PROMPT_STEPS`] greedy steps: the call's argmax, then
    /// every step's token and row (the call's own row is not read).
    fn gemm_run(s: &mut Session<Body38>, ids: &[u32]) -> Result<Run, GateError> {
        s.clear()?;
        take_passes(s)?;
        let mut run = Run::default();
        let mut next = s.model_mut().prompt38(ids, Prompt38::Gemm)?;
        run.tokens.push(next);
        for _ in 0..PROMPT_STEPS {
            next = step(s, &mut run, next)?;
        }
        take_passes(s)?;
        Ok(run)
    }

    /// A prompt call's tier batch services: served, settled, settled with
    /// the rows already in, and the host's wall waiting for the rest.
    type Served = (u64, u64, u64, u64);

    /// [`prompt_run`] unstreamed on the tier load `s`, with the tier's batch
    /// services over it ([`Served`]).
    fn served_prompt(s: &mut Session<Body38>, ids: &[u32]) -> Result<(Run, Served), GateError> {
        let stats = |s: &Session<Body38>| -> Result<Served, GateError> {
            let st = s.model().body(NAME)?.hybrid().tier_batch_stats();
            let t = st.first().ok_or("the load holds no tier batch leg")?;
            Ok((t.served, t.settles, t.settle_early, t.settle_ns))
        };
        let before = stats(s)?;
        let run = prompt_run(s, ids, false)?;
        let after = stats(s)?;
        Ok((
            run,
            (
                after.0 - before.0,
                after.1 - before.1,
                after.2 - before.2,
                after.3 - before.3,
            ),
        ))
    }

    /// The precondition of a tier prompt call's bit clause: its one walk's
    /// tier services, one a tier layer (`layers`; a call too short to route
    /// a slot to every tier layer: at least one, `None`), each waited for (a
    /// call whose tier served nothing would agree with the reference
    /// trivially on any layer the tier holds no routed slot of). The host's
    /// wall in the waits is printed beside it, not a clause.
    fn batch_precondition(
        (served, settles, early, wait_ns): Served,
        layers: Option<&[usize]>,
    ) -> bool {
        let (ok, want) = match layers {
            Some(l) => (
                served == l.len() as u64,
                format!("one a tier layer, {}", l.len()),
            ),
            None => (served > 0, "at least one".to_string()),
        };
        let ok = ok && settles == served;
        println!(
            "precondition: the prompt call's tier services {served} ({want}), each waited for \
             ({settles} settles, {early} with the rows already in, {:.2} ms waiting): {}",
            wait_ns as f64 / 1e6,
            verdict(ok)
        );
        ok
    }

    /// An explicit `--prefill pass` on the tier load `s`: refused by name
    /// (the pass walk has no tier leg), the model where it stood.
    fn pass_refused(s: &mut Session<Body38>, ids: &[u32]) -> Result<bool, GateError> {
        s.clear()?;
        let r = s.model_mut().prompt38(ids, Prompt38::Pass);
        let at = s.pos();
        let ok = match &r {
            Err(e) => e.to_string().contains("the pass walk has no tier leg") && at == 0,
            Ok(_) => false,
        };
        println!(
            "pass: an explicit pass prompt call of {} ids on the tier load is refused by name \
             ({}), the session at position {at}: {}",
            ids.len(),
            match &r {
                Err(e) => e.to_string(),
                Ok(t) => format!("ran, next {t}"),
            },
            verdict(ok)
        );
        s.clear()?;
        Ok(ok)
    }

    /// `gate_qwen38_residency`'s stream rule between the residency-off tier
    /// run `off` and `on`, each a prompt call's row then its steps': the
    /// prompt's row within [`GREEDY_MARGIN`] of the off run's, the greedy
    /// ids equal or parted first at the on run's near tie.
    fn stream_rule(what: &str, off: &Run, on: &Run) -> bool {
        let dlogit = match (off.logits.first(), on.logits.first()) {
            (Some(a), Some(b)) if a.len() == b.len() => a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max),
            _ => f32::INFINITY,
        };
        let parted = off.tokens.iter().zip(&on.tokens).position(|(a, b)| a != b);
        let near =
            parted.is_none_or(|i| on.logits.get(i).is_some_and(|r| margin(r) < GREEDY_MARGIN));
        let ok = near
            && dlogit < GREEDY_MARGIN
            && off.tokens.len() == on.tokens.len()
            && off.logits.len() == on.logits.len();
        println!(
            "residency: {what} against the residency-off tier run: the prompt's last logits row \
             max|diff| {dlogit} (under {GREEDY_MARGIN}), the ids parted at {parted:?} (a near tie \
             {near}), flips landed {}: {}",
            on.landed,
            verdict(ok)
        );
        ok
    }

    /// `--prompt4k` (module doc): plan (b′) and its union at [`CTX_4K`]
    /// positions, a [`PROMPT_4K`]-id prompt call on each.
    fn prompt4k(path: &Path, inputs: &PlanInputs, host: HostCfg) -> Result<bool, GateError> {
        let ub = ubatch_for(CTX_4K)?;
        if ub != PROMPT_4K {
            return Err(format!(
                "a ubatch of {ub} at {CTX_4K} positions: --prompt4k walks the full {PROMPT_4K} \
                 window (unset BLOOMERY_QWEN3_UBATCH)"
            )
            .into());
        }
        let layers = inputs.spec.layers.len();
        let bp = machine_bp(layers, ub as u64, None, tier_batch(&inputs.hp, ub as u64));
        let plan = chosen_plan(inputs, &bp, CTX_4K)?;
        let tl = tier_layers(&plan);
        let ids = prose38(PROMPT_4K)?;
        let mut um = bp.clone();
        um.tiers.clear();
        let uplan = union_plan(&plan, &um)?;
        let mut r = open(path, &uplan, inputs, (host, ub, CTX_4K), Residency::Off, 1)?;
        if !r.model().body(NAME)?.hybrid().tiers().is_empty() {
            return Err("the reference load holds a tier card".into());
        }
        let want = prompt_run(&mut r, &ids, false)?;
        drop(r);
        let mut t = open(path, &plan, inputs, (host, ub, CTX_4K), Residency::Off, 1)?;
        let (got, served) = served_prompt(&mut t, &ids)?;
        let ok = same(
            &format!("two cards, a {PROMPT_4K}-id prompt call at {CTX_4K} positions"),
            &want,
            &got,
        );
        Ok(batch_precondition(served, Some(&tl)) && ok)
    }

    // ---------------------------------------------- --slots: (s2) and (s8)

    /// `reset`, then every selecting store's planes set to [`PLANE_FILL`]:
    /// gate_qwen4exp_e2e's `fresh`, the same rule (the digest reads the
    /// planes whole, and `reset` leaves their rows as they were).
    fn tier_fresh(m: &mut Qwen38Model) -> Result<(), GateError> {
        m.reset()?;
        let (gpu, _, b) = m.body_parts("fresh")?;
        b.fill_planes(gpu, PLANE_FILL)?;
        Ok(())
    }

    /// The digest of the selected slot's whole resident state: every store
    /// and the PLE ring ([`Body38::stores_host`]) and the lane word —
    /// gate_qwen4exp_e2e's `slot_digest`, the same rule.
    fn slot_digest(m: &mut Qwen38Model) -> Result<u64, GateError> {
        let (gpu, _, b) = m.body_parts("stores")?;
        let (stores, ring) = b.stores_host(gpu)?;
        let lane = m.body("slots")?.lane();
        let mut h = Fnv1a64::default();
        let u16s = |h: Fnv1a64, v: &[u16]| v.iter().fold(h, |h, &w| h.bytes(&w.to_le_bytes()));
        for s in &stores {
            match s {
                Store38Host::Rec { state, ring } => h = h.f32s(state).f32s(ring),
                Store38Host::Qsa { k, v, raw, pooled } => {
                    h = u16s(u16s(u16s(u16s(h, k), v), raw), pooled);
                }
            }
        }
        Ok(h.f32s(&ring).bytes(&lane.to_le_bytes()).value())
    }

    /// (s2)'s body for the slot harness ([`slots_gate`]): the tier load of
    /// the plan of two, stream 0 window `a` and stream 1 window `b` (two
    /// disjoint [`PROMPT_U`]-id prose windows) by the ubatch path — the tier
    /// load's prompt path; the eager pass is refused there by name.
    struct TierSlots<'a> {
        path: &'a Path,
        inputs: &'a PlanInputs,
        plan: &'a Plan<'a>,
        host: HostCfg,
        ub: usize,
        a: Vec<u32>,
        b: Vec<u32>,
    }

    impl SlotsAdapter for TierSlots<'_> {
        type Body = Body38;

        const STEPS: usize = 8;
        const TAIL: usize = 4;

        fn open(&self, slots: usize) -> Result<Qwen38Model, GateError> {
            let t0 = Instant::now();
            let file =
                Split::open(self.path).map_err(|e| format!("open {}: {e}", self.path.display()))?;
            let m = Body38::open_placed_residency(
                file,
                self.plan,
                self.inputs,
                0,
                self.host,
                self.ub,
                Residency::Off,
                slots,
            )?;
            println!(
                "slots load in {:.1} s: stage {} ({} B)",
                t0.elapsed().as_secs_f64(),
                m.gpu().device_name()?,
                m.resident_bytes()
            );
            Ok(m)
        }

        fn rewind(&self, m: &mut Qwen38Model) -> Result<(), GateError> {
            tier_fresh(m)
        }

        fn prompt(&self, m: &mut Qwen38Model, stream: usize) -> Result<u32, GateError> {
            Ok(match stream {
                0 => m.prompt38(&self.a, Prompt38::Gemm)?,
                1 => m.prompt38(&self.b, Prompt38::Gemm)?,
                _ => return Err(format!("--slots runs streams 0 and 1, not {stream}").into()),
            })
        }

        fn state_hash(&self, m: &mut Qwen38Model) -> Result<u64, GateError> {
            slot_digest(m)
        }

        /// The stores ([`Body38::store_bytes`]), the lane word (one device
        /// word) and the two arena-row buffers (`(1 + PASS_ROWS) · STREAMS ·
        /// HIDDEN` f32) — (y)'s derivation in gate_qwen4exp_e2e.
        fn seq_bytes_derived(&self, m: &Qwen38Model) -> Result<Derived, GateError> {
            let stores = m.body("slots")?.store_bytes();
            let rows = (1 + PASS_ROWS) * STREAMS * HIDDEN * 4;
            Ok(Derived {
                bytes: stores + 4 + rows,
                terms: format!("store_bytes {stores} + the lane word 4 + the arena rows {rows}"),
            })
        }
    }

    impl PassAdapter for TierSlots<'_> {
        /// [`NODES_SLOT`], (y)'s terms.
        fn added_slot_launches(&self, _m: &Qwen38Model) -> Result<Launches, GateError> {
            Ok(Launches {
                n: NODES_SLOT,
                terms: format!(
                    "the embedding 1 + the PLE conv 1 + {N_GDN} delta layers' 2 + {N_QSA} \
                     selecting layers' 8 + the parked rows' copy 1"
                ),
            })
        }

        fn slots_launches(
            &self,
            m: &Qwen38Model,
            key: &[(usize, usize)],
        ) -> Result<usize, GateError> {
            Ok(m.body("slots")?.slots_launches(key))
        }
    }

    /// One run of (s8)'s history: the ids each slot gave (its prompt's
    /// argmax, every verify row's, every step's), each slot's state hash
    /// after them and its next step's token and logits row.
    struct SlotsRun {
        ids: [Vec<u32>; 2],
        hash: [u64; 2],
        step: [(u32, Vec<f32>); 2],
    }

    /// (s8)'s history (module doc) over `s`: each slot's prompt window (`a`,
    /// `b`) by the session's prompt call, the [`SLOT_WINDOWS`] each as one
    /// `verify_slots` pass of both slots (each slot's greedy token, then
    /// `fill`'s next ids) committed at its [`SLOT_KEEPS`], [`STEPS`] rounds
    /// of one row a slot — then each slot's state hash and its next step's
    /// token and logits row. Both slots' ids count `2 · PROMPT_U` plus the
    /// windows' rows plus [`STEPS`] positions.
    fn slots_history(
        s: &mut Session<Body38>,
        (a, b): (&[u32], &[u32]),
        fill: &[u32],
    ) -> Result<SlotsRun, GateError> {
        let windows = [a, b];
        let mut last = [0; 2];
        let mut ids = [Vec::new(), Vec::new()];
        for slot in 0..2 {
            s.select_slot(slot)?;
            s.clear()?;
            let out = s.prompt(windows[slot], Want::Argmax)?;
            last[slot] = out.argmax();
            ids[slot].push(last[slot]);
        }
        let mut at = [0; 2];
        for (&(m0, m1), &(k0, k1)) in SLOT_WINDOWS.iter().zip(&SLOT_KEEPS) {
            let mut rows = [Vec::new(), Vec::new()];
            for (slot, m) in [(0, m0), (1, m1)] {
                let f = fill
                    .get(at[slot]..at[slot] + m - 1)
                    .ok_or("the corpus holds too few ids for the slots' windows")?;
                rows[slot] = vec![last[slot]; m];
                rows[slot][1..].copy_from_slice(f);
                at[slot] += m;
            }
            let out = s.verify_slots(&[(0, &rows[0]), (1, &rows[1])])?;
            if out.ids.len() != m0 + m1 {
                return Err(format!(
                    "a pass of {m0} and {m1} rows read back {} ids",
                    out.ids.len()
                )
                .into());
            }
            ids[0].extend_from_slice(&out.ids[..m0]);
            ids[1].extend_from_slice(&out.ids[m0..]);
            last = [out.ids[k0 - 1], out.ids[m0 + k1 - 1]];
            s.commit_slots(&[k0, k1])?;
        }
        for _ in 0..STEPS {
            let out = s.step_slots(&[(0, &[last[0]]), (1, &[last[1]])])?;
            if out.ids.len() != 2 {
                return Err(
                    format!("a pass of one row a slot read back {} ids", out.ids.len()).into(),
                );
            }
            last = [out.ids[0], out.ids[1]];
            ids[0].push(last[0]);
            ids[1].push(last[1]);
        }
        let mut hash = [0; 2];
        let mut step = [(0, Vec::new()), (0, Vec::new())];
        for slot in 0..2 {
            s.select_slot(slot)?;
            hash[slot] = slot_digest(s.model_mut())?;
            let out = s.step(last[slot], Want::Logits)?;
            let Out::Logits { argmax, row } = out else {
                return Err("a step asked for its logits gave the argmax alone".into());
            };
            step[slot] = (argmax, row.to_vec());
        }
        Ok(SlotsRun { ids, hash, step })
    }

    /// Slot `slot`'s half of (s8)'s history alone, the other slot parked
    /// from its clear: its prompt window, the same windows as `verify_slots`
    /// passes of it alone committed at the same counts, then greedy steps —
    /// its ids.
    fn slots_solo(
        s: &mut Session<Body38>,
        slot: usize,
        window: &[u32],
        fill: &[u32],
    ) -> Result<Vec<u32>, GateError> {
        s.select_slot(slot)?;
        s.clear()?;
        let mut last = s.prompt(window, Want::Argmax)?.argmax();
        let mut ids = vec![last];
        let mut at = 0;
        for (&(m0, m1), &(k0, k1)) in SLOT_WINDOWS.iter().zip(&SLOT_KEEPS) {
            let (m, k) = if slot == 0 { (m0, k0) } else { (m1, k1) };
            let f = fill
                .get(at..at + m - 1)
                .ok_or("the corpus holds too few ids for the slots' windows")?;
            let mut rows = vec![last; m];
            rows[1..].copy_from_slice(f);
            at += m;
            let out = s.verify_slots(&[(slot, &rows)])?;
            if out.ids.len() != m {
                return Err(format!("a pass of {m} rows read back {} ids", out.ids.len()).into());
            }
            ids.extend_from_slice(&out.ids);
            last = out.ids[k - 1];
            s.commit_slots(&[k])?;
        }
        for _ in 0..STEPS {
            last = s.step(last, Want::Argmax)?.argmax();
            ids.push(last);
        }
        Ok(ids)
    }

    /// (s8)(a): every row's id, each slot's state hash after the history and
    /// its next step's token and logits row bit for bit the union
    /// reference's.
    fn slots_union_clause(want: &SlotsRun, got: &SlotsRun) -> bool {
        let mut off = Vec::new();
        for slot in 0..2 {
            if want.ids[slot] != got.ids[slot] {
                let at = want.ids[slot]
                    .iter()
                    .zip(&got.ids[slot])
                    .position(|(a, b)| a != b);
                off.push(format!("slot {slot}'s ids apart at {at:?}"));
            }
            if want.hash[slot] != got.hash[slot] {
                off.push(format!("slot {slot}'s state hash"));
            }
            let (w, g) = (&want.step[slot], &got.step[slot]);
            let row =
                w.1.iter()
                    .zip(&g.1)
                    .position(|(x, y)| x.to_bits() != y.to_bits());
            if w.0 != g.0 || w.1.len() != g.1.len() || row.is_some() {
                off.push(format!("slot {slot}'s next step's logits row"));
            }
        }
        println!(
            "slots union: each slot's {} ids, its state hash after the history and its next \
             step's logits row bit for bit the union reference's{}: {}",
            got.ids[0].len(),
            if off.is_empty() {
                String::new()
            } else {
                format!(" (differs in {})", off.join(", "))
            },
            verdict(off.is_empty())
        );
        off.is_empty()
    }

    /// (s8)(b): on the tier load each slot's rows its solo run's (the same
    /// windows as verifies of that slot alone, the other parked).
    fn slots_solo_clause(got: &SlotsRun, solo: &[Vec<u32>; 2]) -> bool {
        let mut off = Vec::new();
        for slot in 0..2 {
            if got.ids[slot] != solo[slot] {
                let at = got.ids[slot]
                    .iter()
                    .zip(&solo[slot])
                    .position(|(a, b)| a != b);
                let a = at.unwrap_or(0);
                let around =
                    |v: &[u32]| format!("{:?}", &v[a.saturating_sub(3)..(a + 4).min(v.len())]);
                off.push(format!(
                    "slot {slot}'s ids apart at {at:?} (joint {} solo {})",
                    around(&got.ids[slot]),
                    around(&solo[slot])
                ));
            }
        }
        println!(
            "slots solo: on the tier load each slot's rows its solo run's, the same windows as \
             verifies of that slot alone{}: {}",
            if off.is_empty() {
                String::new()
            } else {
                format!(" (differs in {})", off.join(", "))
            },
            verdict(off.is_empty())
        );
        off.is_empty()
    }

    /// (s8)(c): after the history the tier's graphs of its widths (5 to 8
    /// columns) hold eight nodes a tier layer and the fault copy.
    fn slots_tier_graphs(s: &Session<Body38>, layers: usize) -> Result<bool, GateError> {
        let tier = s
            .model()
            .body(NAME)?
            .hybrid()
            .tiers()
            .first()
            .ok_or("the load holds no tier")?;
        let want = Some(TIER_LAYER_NODES * layers + 1);
        let got: Vec<Option<usize>> = (5..=8).map(|m| tier.graph_nodes(Chain::Cols(m))).collect();
        let ok = got.iter().all(|&g| g == want);
        println!(
            "slots structure: the tier's graphs of the history's widths 5..=8 columns hold \
             {got:?} nodes, {TIER_LAYER_NODES} a tier layer and the fault copy ({want:?}): {}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// `--slots` (module doc): (s2) the slot harness over the tier load of
    /// the plan of two, then (s8) the passes of 5..8 rows on that load and
    /// its union reference. Three loads.
    fn slots_clauses(
        path: &Path,
        inputs: &PlanInputs,
        bp: &Machine,
        host: HostCfg,
        ub: usize,
    ) -> Result<bool, GateError> {
        let plan = chosen_plan_slots(inputs, bp, CTX, 2)?;
        let tl = tier_layers(&plan);
        let ids = prose38(2 * PROMPT_U + 32)?;
        let (a, b) = (&ids[..PROMPT_U], &ids[PROMPT_U..2 * PROMPT_U]);
        let fill = &ids[2 * PROMPT_U..];
        let body = TierSlots {
            path,
            inputs,
            plan: &plan,
            host,
            ub,
            a: a.to_vec(),
            b: b.to_vec(),
        };
        let mut s = slots_gate::interleave(&body)?;
        s.one_pass();
        let mut ok = s.finish()?;
        let mut um = bp.clone();
        um.tiers.clear();
        let uplan = union_plan(&plan, &um)?;
        let mut r = open(path, &uplan, inputs, (host, ub, CTX), Residency::Off, 2)?;
        if !r.model().body(NAME)?.hybrid().tiers().is_empty() {
            return Err("the reference load holds a tier card".into());
        }
        r.add_slots(2)?;
        let want = slots_history(&mut r, (a, b), fill)?;
        drop(r);
        let mut t = open(path, &plan, inputs, (host, ub, CTX), Residency::Off, 2)?;
        t.add_slots(2)?;
        let got = slots_history(&mut t, (a, b), fill)?;
        ok &= slots_union_clause(&want, &got);
        let solo = [
            slots_solo(&mut t, 0, a, fill)?,
            slots_solo(&mut t, 1, b, fill)?,
        ];
        ok &= slots_solo_clause(&got, &solo);
        ok &= slots_tier_graphs(&t, tl.len())?;
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
        let plan = chosen_plan(&inputs, &bp, CTX)?;
        let tl = tier_layers(&plan);
        let ids = prose38(PROMPT + 32)?;
        let prompt_ids = prose38(PROMPT_U)?;
        let mut off_prompt = None;
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
            let mut r = open(path, &uplan, &inputs, (host, ub, CTX), Residency::Off, 1)?;
            if !r.model().body(NAME)?.hybrid().tiers().is_empty() {
                return Err("the reference load holds a tier card".into());
            }
            let want = history(&mut r, &ids)?;
            let want_prompt = prompt_run(&mut r, &prompt_ids, false)?;
            let want_short = gemm_run(&mut r, &prompt_ids[..PROMPT_SHORT])?;
            drop(r);
            let mut t = open(path, &plan, &inputs, (host, ub, CTX), Residency::Off, 1)?;
            let got = history(&mut t, &ids)?;
            pass &= same("two cards", &want, &got);
            pass &= structure(&t, &want, &got, &tl)?;
            let (got_prompt, served) = served_prompt(&mut t, &prompt_ids)?;
            pass &= same(
                &format!("two cards, a {PROMPT_U}-id prompt call"),
                &want_prompt,
                &got_prompt,
            );
            pass &= batch_precondition(served, Some(&tl));
            let (mut got_short, served) = served_prompt(&mut t, &prompt_ids[..PROMPT_SHORT])?;
            got_short.logits.remove(0);
            pass &= same(
                &format!(
                    "two cards, a {PROMPT_SHORT}-id prompt call by the session's path (the \
                     reference's an explicit ubatch call; the steps' rows)"
                ),
                &want_short,
                &got_short,
            );
            pass &= batch_precondition(served, None);
            pass &= pass_refused(&mut t, &prompt_ids[..PROMPT_SHORT])?;
            off_prompt = Some(got_prompt);
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
            let mut t = open(path, &plan, &inputs, (host, ub, CTX), residency, 1)?;
            pass &= residency_clauses(
                &mut t,
                (&ids, &prompt_ids),
                (off_run.as_ref(), off_prompt.as_ref()),
                (&split, &plan),
            )?;
            last = Some(t);
        }
        if args.lost {
            let mut s = match last.take() {
                Some(s) => s,
                None => open(path, &plan, &inputs, (host, ub, CTX), Residency::Off, 1)?,
            };
            pass &= lost_case(s.model_mut(), &ids)?;
        }
        if args.prompt4k {
            drop(last.take());
            pass &= prompt4k(path, &inputs, host)?;
        }
        if args.slots {
            drop(last.take());
            pass &= slots_clauses(path, &inputs, &bp, host, ub)?;
        }
        if !pass {
            return Err(checks_failed());
        }
        println!("PASSED: {NAME}");
        Ok(())
    }
}
