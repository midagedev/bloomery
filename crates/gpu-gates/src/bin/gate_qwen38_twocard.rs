//! GPU gate for Qwen3.8's plan (b′) on both cards (`--place bp`): the stage
//! on the stage card and its expert tier on the tier card (`tier38`), against
//! one card holding the same expert sets. No MTP draft: a verify of each width
//! the draft's windows use (2, 3 and 4 rows) is driven directly. The cards are
//! the two in view ([`bp_machine`]): the real tier plans the workstation
//! pair's bytes on them, the fixture tier the cards as they report themselves.
//!
//! The plans are made under a card budget (`PlanLevers::card_budget_bytes`)
//! that caps both cards, so the stage card can hold the stage's and the tier's
//! sets at once: the gate takes the largest budget of [`BUDGETS`] whose plan
//! puts tier experts on every layer, whose union fits the stage card (its stage
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
//!   nodes less one a tier layer the reference's plan hosts experts on (its
//!   card leg drops the card sum, which the join takes after the wait; a
//!   layer the reference hosts nothing on already joins its card rows there)
//!   plus [`HOST_LEG_NODES`] a tier layer the
//!   reference's plan hosts nothing on, read per layer from that plan (the
//!   reference's load runs such a layer without a host leg, no go and no
//!   wait; the tier load keeps the leg on every tier layer, its stage card
//!   holding fewer than all the layer's experts); the tier's graph of the
//!   step and of each verify width holds nine nodes a tier layer (the go
//!   wait, the two copies in, the four launches into the tier card's rows,
//!   the rows' copy out to the page, the signal) and the fault copy; each
//!   chain the history runs is fed and captured once, at its first pass, and
//!   replayed by every later one. Precondition: every tier layer was sent a
//!   routed slot by the steps.
//! - `--residency`: the tier load under `mid-p<P>-s1`, `P` half the plan's
//!   fewest stage experts a layer (the stage's count, not the union's): the
//!   history twice the same, flips landed (in the real tier: that flips land
//!   at all is the router's skew, which a fixture's generated router lacks, so
//!   the fixture tier prints the counts and holds the rest); against the
//!   residency-off tier run, `gate_qwen38_residency`'s stream rule (the machine moves experts
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
//!   `--union`; on the reference the same call by the split walk
//!   (`SplitWalk::Every`: two units of half its rows, the host streaming off)
//!   bit for bit its one-unit run, the walk two units; on the tier load the
//!   split walk set by the rule refused by name (a tier load), the load's
//!   own cut one unit. Every load opens with the split walk off.
//! - `--lost`: on the last load, the tier's stream held behind a host flag
//!   before a step — the tier stops signalling: within the go deadline and
//!   its grace the step fails naming the lost card, the host tier is
//!   poisoned as a lost card, and the next step at that position is refused
//!   by the recurrent stores (the delta layers ran it already); the flag is
//!   then raised, both streams drain, and a reset is refused by the lost
//!   card's poison, which a reset does not lift.
//! - `--slots`: a tier load planned for two slots (the plan counts them, under
//!   the budget `--union` found), so the tier's page, places and card leg are
//!   cut to the step port's [`Prompt38::PASS_ROWS`] columns; a load of one
//!   slot keeps four. The shared slot harness (`slots_gate`) drives it through
//!   an adapter for this body: its interleave (H1), one pass of one row a slot
//!   (H2), bytes (H3), reset (H4), poison (H5), refusals (H6) and captures
//!   (H7); both streams are prompted by the ubatch walk (a tier load has no
//!   pass walk), from two windows of the prose corpus of two lengths. Then
//!   `slots_wide`, on that load with both slots rewound: five passes of 4+4,
//!   4+3, 3+4, 4+2 and 3+2 rows, which the tier serves as 8, 7, 7, 6 and 5
//!   columns, each slot's every row's id and logits, two plain steps after them,
//!   state hash and position bit for bit its own rows run alone as verifies
//!   kept whole on the same load. And `structure`
//!   on that load after the passes: the tier's graph of each of 5 to 8 columns
//!   holds [`TIER_LAYER_NODES`] nodes a tier layer and the fault copy, and the
//!   tier is opened for the port's columns. The clauses run in both tiers
//!   (self-consistency); the harness's one-slot clause opens the second load.
//!
//! Before the arms' loads, one child run of `generate_qwen3moe` beside this
//! binary with no `--place` takes the common unset rule
//! (`generate::Place::choose`, by the Qwen3.8 family's rule): on this gate's
//! two cards its `place unset` record keeps `bp`, the offer's tier at or
//! past the rule's break-even, naming the tier's experts in the plan at the
//! child's default context and the rule's break-even and basis. The clause
//! is file-bound (a fixture's plan puts another count on the tier), and the
//! child takes this gate's environment less the placed levers — a run with
//! no `--place` refuses them by name.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen38_twocard", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/q38_arch.rs"]
mod q38_arch;

#[cfg(feature = "gpu")]
#[path = "shared/q38_history.rs"]
mod q38_history;

#[cfg(feature = "gpu")]
mod gate {
    use std::path::Path;
    use std::time::Instant;

    use app::Session;
    use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for;
    use bloomery_gpu::arch::qwen3moe::{Body38, Prompt38, Qwen38Model, SplitWalk, Store38Host};
    use bloomery_gpu::host::swap::{PassReport, Residency};
    use bloomery_gpu::host::tier::TierSet;
    use bloomery_gpu::host::xstream::XMode;
    use bloomery_gpu::host::{PassKind, PoisonKind};
    use bloomery_gpu::hybrid::Chain;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{GpuError, HostFlags};
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::q38_fixture::Shape;
    use bloomery_gpu_gates::record::{self, Fields};
    use bloomery_gpu_gates::slots_gate::{self, Derived, Launches, PassAdapter, SlotsAdapter};
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{
        Fnv1a64, GREEDY_MARGIN, GateError, bits_equal, checks_failed, prose_ids, ref_model_path,
        verdict,
    };
    use bloomery_levers::{CARD_BUDGET, CARD_DONTNEED, HOST_LOCK, HOST_POPULATE, HostCfg};
    use gguf::Split;
    use model::arch::qwen35moe::place::{
        Experts, PlaceError, PlanInputs, Stack, machine_bp_on, tier_batch,
    };
    use model::placement::host_lock::{HostFile, HostSet};
    use model::placement::workstation::{self, A6000, Pick, RTX_3090, TierBatchBytes};
    use model::placement::{
        Device, ExpertList, Machine, PlacementError, Plan, PlanLevers, Role, Row, Segment,
    };
    use refset::arch::qwen4exp::MODEL;
    use runtime::swaprule::KeptRows;
    use runtime::{Out, Target, Want};

    use crate::q38_history::{
        self, PROMPT, Run, TIER_LAYER_NODES, same, step, tier_graphs_and_chains, tier_layers,
    };

    const NAME: &str = "gate_qwen38_twocard";
    /// The stores' positions, `gate_qwen38_residency`'s.
    const CTX: usize = 3072;
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
    /// The card budget the real file's gate takes on the workstation pair's
    /// bytes ([`bp_machine`]), GiB: the literal the budget search
    /// ([`chosen_plan`]) replaced, the witness the real tier prints the
    /// search's pick beside and requires equal.
    const REAL_BUDGET_GIB: u64 = 26;
    /// The go deadline and the grace a lost card is named within, plus room.
    const LOST_BOUND_S: f64 = 25.0;
    /// Nodes of a stage graph a layer with a host leg, two: the go and the
    /// wait, stream memory-operation batches ([`Shape::memops`]). A load with
    /// no residency machine and no route trace runs a layer with every routed
    /// expert on the card without them.
    const HOST_LEG_NODES: usize = 2;
    /// Prompt ids the unset-rule clause's child runs.
    const UNSET_TOKENS: usize = 8;
    /// The Qwen3.8 family's unset rule as the child names it
    /// (`shared/qwen38_place.rs`'s `Q38_RULE`): the break-even experts and
    /// the card file that decides them.
    const BREAK_EVEN: u64 = 1_152;
    const TIER_BASIS: &str = "docs/cards/q38bpbug-ab.card";
    /// The experts the child's plan of `bp` puts on this box's 3090 at its
    /// default context, a band for the free bytes each census reads.
    const BP_TIER_EXPERTS: std::ops::RangeInclusive<u64> = 7_230..=7_242;
    /// `--slots`' two streams: each one's prompt ids, then the ids its wide
    /// passes' rows are filled from; the two windows of the prose corpus come
    /// one after the other.
    const SLOT_PROMPTS: [usize; 2] = [24, 40];
    const SLOT_FILL: usize = 16;
    /// `--slots`' wide passes: the rows a pass gives slot 0 and slot 1. The
    /// tier serves a pass as the sum of them in columns: 8, 7, 7, 6 and 5.
    const WIDE_PASSES: [(usize, usize); 5] = [(4, 4), (4, 3), (3, 4), (4, 2), (3, 2)];
    /// The fewest columns a wide pass takes: one past a verify's four.
    const WIDE_FROM: usize = 5;
    /// The plain steps after the wide passes, which read what the passes left.
    const WIDE_STEPS: usize = 2;
    /// The file's model width and its residual streams, as the header states
    /// them (`gate_qwen4exp_e2e`'s): the arena rows of one sequence's bytes.
    const HIDDEN: usize = 2560;
    const PLE_STREAMS: usize = 4;

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

    /// Plan (b′)'s machine for `layers` layers at ubatches of `ub` on the two
    /// cards in view, the stage card then its tier. The real tier resolves them
    /// as the workstation pair's room (`Pick::Room` of the A6000, then of the
    /// 3090 on the devices the stage card leaves: each plans that card's total
    /// and driver reserve on the device it lands on, held as
    /// `tier::real_card` holds a census card), the bytes
    /// [`REAL_BUDGET_GIB`] and the union limit were found on, and refuses a
    /// pair without that room by name; the fixture tier plans `bp` as the
    /// devices report themselves (`--place bp`).
    fn bp_machine(layers: Stack, ub: u64, batch: TierBatchBytes) -> Result<Machine, GateError> {
        let census = bloomery_gpu::census()?;
        let specs = match tier::Tier::from_env()? {
            tier::Tier::Real => {
                let stage = workstation::resolve(&[Pick::Room(&A6000)], &census)?[0];
                let rest: Vec<_> = census
                    .iter()
                    .filter(|d| Some(d.device()) != stage.device)
                    .cloned()
                    .collect();
                let tier_card = workstation::resolve(&[Pick::Room(&RTX_3090)], &rest)?[0];
                vec![tier::real_card(stage), tier::real_card(tier_card)]
            }
            tier::Tier::Fixture => Place::Bp.on(&census)?.card_specs()?,
        };
        let [stage, tier_card] = specs[..] else {
            return Err(format!("plan (b′) takes two cards, {} resolved", specs.len()).into());
        };
        Ok(machine_bp_on((stage, tier_card), layers, ub, None, batch))
    }

    /// The tier's device index in the plan's cards.
    fn tier_device(plan: &Plan<'_>) -> Device {
        Device::Card(plan.machine.cards.len())
    }

    /// Why plan (b′) under a budget cannot be the gate's: no tier expert on
    /// some layer, a union past the stage card, or more than two union counts.
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
            return Some(format!(
                "the union takes {union} B of the stage card {}'s {limit}",
                card.name
            ));
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

    /// A load of `plan` under `residency`, its session over `ctx`
    /// positions.
    fn open(
        path: &Path,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        (host, ub, ctx): (HostCfg, usize, usize),
        residency: Residency,
    ) -> Result<Session<Body38>, GateError> {
        let t0 = Instant::now();
        let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let m = Body38::open_placed_residency(
            file,
            plan,
            inputs,
            0,
            host,
            ub,
            residency,
            bloomery_levers::ChurnHosting::Hosted,
            1,
        )?;
        let tiers = m.body(NAME)?.hybrid().tiers();
        let tier = match tiers.first() {
            Some(t) => format!(
                ", tier {} experts on {} ({} B)",
                t.set().experts(),
                t.gpu()?.device_name()?,
                t.resident_bytes()
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
        let b = s.model_mut().body_parts(NAME)?.2;
        b.log_residency(0);
        b.set_split_walk(SplitWalk::Off)?;
        Ok(s)
    }

    /// The history (`q38_history`) from a clear, with the flips that landed
    /// over it.
    fn history(s: &mut Session<Body38>, ids: &[u32]) -> Result<Run, GateError> {
        let mut run = q38_history::history(s, ids, |s| take_passes(s).map(drop))?;
        run.landed = take_passes(s)?.iter().map(|(_, r)| r.landed).sum();
        Ok(run)
    }

    /// The boundaries' reports since the last take.
    fn take_passes(s: &mut Session<Body38>) -> Result<Vec<(PassKind, PassReport)>, GateError> {
        let (_, _, b) = s.model_mut().body_parts(NAME)?;
        Ok(b.take_residency_passes())
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

    /// `--union`'s structure and precondition lines over the tier load `s`
    /// and the two runs; `reference` is the plan the reference loaded.
    fn structure(
        s: &Session<Body38>,
        want: &Run,
        got: &Run,
        layers: &[usize],
        reference: &Plan<'_>,
    ) -> Result<bool, GateError> {
        let t = layers.len();
        // A tier layer the reference's plan hosts nothing on runs without a
        // host leg in the reference's load; the tier load's stage card holds
        // fewer than all its experts and keeps the leg.
        let hosted = layers
            .iter()
            .map(|&l| reference.host_experts(l))
            .collect::<Result<Vec<u64>, _>>()?;
        let unhosted = hosted.iter().filter(|&&n| n == 0).count();
        // PIN(2026-10-29): a hostless layer joins its retained card rows with the shared add in
        // one launch (c7b642055), as a tier layer does, so only a tier layer the reference hosts
        // experts on drops a card sum: the reference's [121, 125, 125, 125] with all four tier
        // layers unhosted gives [129, 133, 133, 133], the count this clause was red on.
        let hosted_tier = t - unhosted;
        let stage: Vec<usize> = want
            .stage_nodes
            .iter()
            .map(|n| n + HOST_LEG_NODES * unhosted - hosted_tier)
            .collect();
        let mut ok = got.stage_nodes == stage;
        println!(
            "structure: the stage graphs of the step and the 2, 3, 4-row verifies hold {:?} nodes, \
             the reference's {:?} less one a tier layer it hosts experts on ({hosted_tier}) plus the go and the wait \
             ({HOST_LEG_NODES}) of each tier layer its plan hosts nothing on ({unhosted}; host \
             experts a tier layer {hosted:?}) = {stage:?}: {}",
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
        ok &= tier_graphs_and_chains(tier, t);
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
        skewed: bool,
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
        let flips = !skewed || (first.landed > 0 && again.landed > 0);
        println!(
            "residency: the history twice the same {det}, flips landed {} and {}{}: {}",
            first.landed,
            again.landed,
            if skewed {
                ""
            } else {
                " (printed, not pinned in this tier)"
            },
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
        let (flags, lost) = {
            let body = m.body(NAME)?;
            let tier = body.hybrid().tiers().first().ok_or("no tier card")?;
            let flags = HostFlags::new(tier.gpu()?.context(), 1)?;
            flags.enqueue_wait(tier.gpu()?.stream(), 0)?;
            (flags, tier.name().to_string())
        };
        let t0 = Instant::now();
        let r = m.step(&[ids[4]]);
        let secs = t0.elapsed().as_secs_f64();
        let mut ok = match &r {
            Err(e)
                if e.to_string().contains("the card is lost")
                    && e.to_string().contains(&lost)
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
            .gpu()?
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

    /// What [`chosen_plan`] found: the plan, whether its budget is the
    /// largest whole GiB the gate fits (the next one is refused) and, on the
    /// real file, the literal's, and that budget in GiB.
    struct Chosen<'a> {
        plan: Plan<'a>,
        tight: bool,
        gib: u64,
    }

    /// Plan (b′) under a whole-GiB budget on both cards: the plan (`None` when the
    /// stage card holds every routed expert and the tier card is left idle,
    /// `PlacementError::IdleTier`) and why it cannot be the gate's ([`unfit`]).
    /// Any other refusal is an error.
    fn plan_under<'a>(
        inputs: &'a PlanInputs,
        bp: &'a Machine,
        ctx: usize,
        gib: u64,
    ) -> Result<(Option<Plan<'a>>, Option<String>), GateError> {
        let levers = PlanLevers {
            card_budget_bytes: Some(gib << 30).into(),
        };
        match inputs.plan_with(
            bp,
            (ctx as u64, u64::try_from(ubatch_for(ctx)?)?),
            &levers,
            Experts::Card,
        ) {
            Ok(plan) => {
                let why = unfit(&plan);
                Ok((Some(plan), why))
            }
            Err(PlaceError::Placement(PlacementError::IdleTier { card, tier })) => Ok((
                None,
                Some(format!(
                    "tier card {card} (tier {tier}) holds no expert: the stage card holds every one"
                )),
            )),
            Err(e) => Err(format!("plan (b′) under {gib} GiB: {e}").into()),
        }
    }

    /// Plan (b′) on `bp` at `ctx` positions under the largest whole-GiB budget
    /// that fits the gate ([`unfit`]), searched down from the stage card's
    /// usable GiB, printed with why each larger one did not. The pick is held
    /// to be the largest: the next whole GiB above it is refused (re-planned
    /// here, not read off the search). On the real file the search lands on
    /// [`REAL_BUDGET_GIB`], where the next one's union passes the stage card's bytes.
    fn chosen_plan<'a>(
        inputs: &'a PlanInputs,
        bp: &'a Machine,
        ctx: usize,
    ) -> Result<Chosen<'a>, GateError> {
        let top = bp.cards[0].usable_bytes >> 30;
        for gib in (1..=top).rev() {
            let (plan, why) = plan_under(inputs, bp, ctx, gib)?;
            if let Some(why) = why {
                println!("budget {gib} GiB at {ctx} positions: {why}; next");
                continue;
            }
            let plan = plan.ok_or("a plan under a budget that fits the gate has no plan")?;
            let held = |v: &[u64]| {
                (
                    v.iter().copied().min().unwrap_or(0),
                    v.iter().copied().max().unwrap_or(0),
                )
            };
            println!(
                "plan (b′) at {ctx} positions under a card budget of {gib} GiB: stage {} experts \
                 (a layer {:?}), tier {} experts (a layer {:?}) on {} layers, host {} experts",
                plan.cards[0].experts,
                held(&plan.n_l),
                plan.cards.get(1).map_or(0, |c| c.experts),
                held(plan.tier_n_l.first().map_or(&[][..], Vec::as_slice)),
                tier_layers(&plan).len(),
                plan.host.experts
            );
            let next = gib + 1;
            let tight = if next > top {
                true
            } else {
                let (_, why) = plan_under(inputs, bp, ctx, next)?;
                println!(
                    "budget {gib} GiB is the largest whole GiB the gate fits: {next} GiB {} {}",
                    why.as_deref().unwrap_or("fits"),
                    verdict(why.is_some())
                );
                why.is_some()
            };
            let witnessed = tier::witness("twocard budget GiB", gib, REAL_BUDGET_GIB);
            return Ok(Chosen {
                plan,
                tight: tight && witnessed,
                gib,
            });
        }
        Err(format!("no whole-GiB budget fits the gate at {ctx} positions").into())
    }

    /// The prompt calls' host streaming: the residency pool's pick
    /// (`Body38::set_xstream`'s `admit`) when `on`, nothing when off.
    fn set_stream(s: &mut Session<Body38>, on: bool) -> Result<(), GateError> {
        let (gpu, _, body) = s.model_mut().body_parts(NAME)?;
        let mode = if on { XMode::Admit } else { XMode::Off };
        Ok(body.set_xstream(gpu, mode)?)
    }

    /// From a clear, `ids` as one prompt call (the session's path: ubatches
    /// for a prompt this long) with the call's host streaming `stream`, then
    /// [`PROMPT_STEPS`] greedy steps: the prompt's last logits row and its
    /// argmax first, then every step's.
    fn prompt_run(s: &mut Session<Body38>, ids: &[u32], stream: bool) -> Result<Run, GateError> {
        s.clear()?;
        take_passes(s)?;
        set_stream(s, stream)?;
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
        set_stream(s, false)?;
        run.landed = take_passes(s)?.iter().map(|(_, r)| r.landed).sum();
        Ok(run)
    }

    /// [`prompt_run`] with the host streaming off by the split walk
    /// ([`SplitWalk::Every`]), and the units the prompt call's walk ran; the
    /// cut set back to one unit either way.
    fn split_run(s: &mut Session<Body38>, ids: &[u32]) -> Result<(Run, usize), GateError> {
        let set = |s: &mut Session<Body38>, w| -> Result<(), GateError> {
            Ok(s.model_mut().body_parts(NAME)?.2.set_split_walk(w)?)
        };
        set(s, SplitWalk::Every)?;
        let r = prompt_run(s, ids, false)
            .and_then(|run| Ok((run, s.model().body(NAME)?.ubatch_units())));
        set(s, SplitWalk::Off)?;
        r
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
        let layers = inputs.stack();
        let bp = bp_machine(layers, ub as u64, tier_batch(&inputs.spec, ub as u64))?;
        let Chosen { plan, tight, .. } = chosen_plan(inputs, &bp, CTX_4K)?;
        let tl = tier_layers(&plan);
        let ids = prose_ids("qwen4exp", PROMPT_4K)?;
        let mut um = bp.clone();
        um.tiers.clear();
        let uplan = union_plan(&plan, &um)?;
        let mut r = open(path, &uplan, inputs, (host, ub, CTX_4K), Residency::Off)?;
        if !r.model().body(NAME)?.hybrid().tiers().is_empty() {
            return Err("the reference load holds a tier card".into());
        }
        let want = prompt_run(&mut r, &ids, false)?;
        let (on, units) = split_run(&mut r, &ids)?;
        let split_ok = same(
            &format!("the reference's {PROMPT_4K}-id prompt call by the split walk vs one unit"),
            &want,
            &on,
        ) && units == 2;
        println!(
            "the reference's split walk ran {units} units (want 2): {}",
            verdict(units == 2)
        );
        drop(r);
        let mut t = open(path, &plan, inputs, (host, ub, CTX_4K), Residency::Off)?;
        let refused = refuses_split(&mut t)?;
        let (got, served) = served_prompt(&mut t, &ids)?;
        let ok = same(
            &format!("two cards, a {PROMPT_4K}-id prompt call at {CTX_4K} positions"),
            &want,
            &got,
        );
        let one = t.model().body(NAME)?.ubatch_units() == 1;
        println!(
            "the tier load's prompt call ran {} unit(s) (want 1): {}",
            t.model().body(NAME)?.ubatch_units(),
            verdict(one)
        );
        Ok(batch_precondition(served, Some(&tl)) && ok && tight && split_ok && refused && one)
    }

    /// The split walk by the rule set on a tier load: refused by name (a
    /// tier load), the load's cut left one unit.
    fn refuses_split(t: &mut Session<Body38>) -> Result<bool, GateError> {
        let b = t.model_mut().body_parts(NAME)?.2;
        let e = b.set_split_walk(SplitWalk::Rule);
        let named = e.as_ref().is_err_and(|e| {
            e.to_string().contains("BLOOMERY_SPLIT_WALK") && e.to_string().contains("tier load")
        });
        let kept = b.split_walk() == SplitWalk::Off;
        println!(
            "the split walk by the rule on the tier load: {} (want refused, naming the lever and a \
             tier load), the cut {:?} (want Off): {}",
            match &e {
                Ok(()) => "set".to_string(),
                Err(e) => format!("refused: {e}"),
            },
            b.split_walk(),
            verdict(named && kept)
        );
        Ok(named && kept)
    }

    /// `generate_qwen3moe`'s first record of `kind` of a short run of `ids`
    /// under `args`: the binary beside this one, its stdout read whole by
    /// its kinds, an error naming the run on a non-zero exit. The child takes
    /// this gate's environment less the placed levers — a run with no
    /// `--place` refuses them by name, and the clause holds the default run.
    fn child_record(
        kind: &'static record::Kind,
        args: &[&str],
        ids: &[u32],
    ) -> Result<Fields, GateError> {
        let exe = std::env::current_exe()?.with_file_name("generate_qwen3moe");
        let tokens: Vec<String> = ids.iter().map(u32::to_string).collect();
        let out = std::process::Command::new(&exe)
            .args(args)
            .arg("--tokens")
            .arg(tokens.join(","))
            .arg("-n")
            .arg("2")
            .env_remove(CARD_BUDGET)
            .env_remove(HOST_POPULATE)
            .env_remove(HOST_LOCK)
            .env_remove(CARD_DONTNEED)
            .stdin(std::process::Stdio::null())
            .output()
            .map_err(|e| format!("spawn {}: {e}", exe.display()))?;
        if !out.status.success() {
            return Err(format!(
                "generate_qwen3moe {}: {}; stderr: {}",
                args.join(" "),
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            )
            .into());
        }
        let stdout = String::from_utf8(out.stdout)?;
        let log = record::Log::of(&stdout, record::GENERATE_QWEN3MOE)
            .named(format!("generate_qwen3moe {}", args.join(" ")));
        Ok(log.one(kind)?)
    }

    /// The `place unset` record a child run with no `--place` prints: the
    /// common rule's choice on this process's cards ([`Self::child_record`]).
    fn unset_record(ids: &[u32]) -> Result<Fields, GateError> {
        child_record(&record::PLACE_UNSET, &[], ids)
    }

    /// The unset-rule clause (the module header): a child run with no
    /// `--place` keeps the offer `bp` at or past the family's break-even,
    /// its record naming the word, why, the offer's tier count at the
    /// child's default context, and the rule's break-even and basis.
    /// FAIL-first: a CLI that keeps a placement of its own prints no `place
    /// unset` line, and a rule with no break-even answers "no sitting has
    /// shown the tier not slower" — both red here.
    fn records_unset(ids: &[u32]) -> Result<bool, GateError> {
        let p = unset_record(ids)?;
        let (place, why) = (p.word("place")?, p.text("why")?);
        let tier = p.opt_u64("tier_experts")?;
        let ok = place == "bp"
            && why == "two cards, tier at or past the break-even"
            && tier.is_some_and(|t| BP_TIER_EXPERTS.contains(&t) && t >= BREAK_EVEN)
            && p.opt_u64("break_even")? == Some(BREAK_EVEN)
            && p.opt_word("basis")? == Some(TIER_BASIS);
        println!(
            "no --place: place unset place={place} why={why} tier_experts={tier:?} \
             break_even={:?} basis={:?}: {}",
            p.opt_u64("break_even")?,
            p.opt_word("basis")?,
            verdict(ok)
        );
        Ok(ok)
    }

    /// `--slots`' body for the shared slot harness ([`slots_gate`]): the tier
    /// load of plan (b′) planned for `slots` sequences under the budget the
    /// gate found, its two streams two windows of the prose corpus.
    struct SlotsTier<'a> {
        path: &'a Path,
        inputs: &'a PlanInputs,
        bp: &'a Machine,
        gib: u64,
        host: HostCfg,
        ub: usize,
        shape: Shape,
        windows: [Vec<u32>; 2],
    }

    impl<'a> SlotsTier<'a> {
        /// Plan (b′) for `slots` sequences at [`CTX`] positions under the
        /// gate's budget: one slot is [`PlanInputs::plan_with`] itself.
        fn plan(&self, slots: usize) -> Result<Plan<'a>, GateError> {
            let levers = PlanLevers {
                card_budget_bytes: Some(self.gib << 30).into(),
            };
            Ok(self.inputs.plan_with_slots(
                self.bp,
                (CTX as u64, u64::try_from(ubatch_for(CTX)?)?),
                &levers,
                Experts::Card,
                slots,
            )?)
        }

        /// The ids stream `s` fills its wide passes' rows from.
        fn fill(&self, s: usize) -> &[u32] {
            &self.windows[s][SLOT_PROMPTS[s]..]
        }
    }

    /// `reset`, then every selecting store's planes set to the one pattern
    /// the state hash reads whole: a row no call writes must read it, never a
    /// previous run's value (`gate_qwen4exp_e2e`'s `fresh`).
    fn fresh(m: &mut Qwen38Model) -> Result<(), GateError> {
        /// The f16 pattern (100.0): finite, and far from any value the model
        /// writes there.
        const PLANE_FILL: u16 = 0x5640;
        m.reset()?;
        let (gpu, _, b) = m.body_parts("fresh")?;
        b.fill_planes(gpu, PLANE_FILL)?;
        Ok(())
    }

    /// The digest of the selected slot's whole resident state: every store,
    /// the PLE ring and the lane word.
    fn slot_digest(m: &mut Qwen38Model) -> Result<u64, GateError> {
        let (stores, ring) = {
            let (gpu, _, b) = m.body_parts("slot digest")?;
            b.stores_host(gpu)?
        };
        let lane = m.body("slot digest")?.lane();
        let u16s = |h: Fnv1a64, v: &[u16]| v.iter().fold(h, |h, &w| h.bytes(&w.to_le_bytes()));
        let mut h = Fnv1a64::default();
        for s in &stores {
            match s {
                Store38Host::Rec { state, ring, carry } => {
                    h = h.f32s(state).f32s(ring).f32s(carry);
                }
                Store38Host::Qsa { k, v, ring, pooled } => {
                    h = u16s(u16s(u16s(u16s(h, k), v), ring), pooled);
                }
            }
        }
        Ok(h.f32s(&ring).bytes(&lane.to_le_bytes()).value())
    }

    impl SlotsAdapter for SlotsTier<'_> {
        type Body = Body38;

        const STEPS: usize = 8;
        const TAIL: usize = 4;

        /// The tier load planned for `slots`: its plan counts them, and its
        /// tier is cut to the port's columns for more than one.
        fn open(&self, slots: usize) -> Result<Qwen38Model, GateError> {
            let t0 = Instant::now();
            let plan = self.plan(slots)?;
            let file =
                Split::open(self.path).map_err(|e| format!("open {}: {e}", self.path.display()))?;
            let mut m = Body38::open_placed_residency(
                file,
                &plan,
                self.inputs,
                0,
                self.host,
                self.ub,
                Residency::Off,
                bloomery_levers::ChurnHosting::Hosted,
                slots,
            )?;
            let b = m.body_parts(NAME)?.2;
            b.log_residency(0);
            b.set_split_walk(SplitWalk::Off)?;
            let tiers = m.body(NAME)?.hybrid().tiers();
            let tier = tiers.first().ok_or("the slots load holds no tier")?;
            println!(
                "slots load in {:.1} s, planned for {slots}: tier {} experts on {}, opened for {} \
                 columns a row",
                t0.elapsed().as_secs_f64(),
                tier.set().experts(),
                tier.gpu()?.device_name()?,
                tier.cols()
            );
            Ok(m)
        }

        fn rewind(&self, m: &mut Qwen38Model) -> Result<(), GateError> {
            fresh(m)
        }

        /// Each stream's window by the ubatch walk: a tier load refuses the
        /// pass walk.
        fn prompt(&self, m: &mut Qwen38Model, stream: usize) -> Result<u32, GateError> {
            let window = self
                .windows
                .get(stream)
                .ok_or_else(|| format!("--slots runs streams 0 and 1, not {stream}"))?;
            Ok(m.prompt38(&window[..SLOT_PROMPTS[stream]], Prompt38::Gemm)?)
        }

        fn state_hash(&self, m: &mut Qwen38Model) -> Result<u64, GateError> {
            slot_digest(m)
        }

        /// The stores ([`Body38::store_bytes`]), the commit's replay table
        /// (`runtime::stores::DELTA_REPLAY_WORDS` u64 words a delta layer),
        /// the lane word (one device word) and the two arena-row buffers
        /// (`(1 + PASS_ROWS) · PLE_STREAMS · HIDDEN` f32).
        fn seq_bytes_derived(&self, m: &Qwen38Model) -> Result<Derived, GateError> {
            let stores = m.body("slots")?.store_bytes();
            let table = self.shape.n_gdn() * runtime::stores::DELTA_REPLAY_WORDS * 8;
            let rows = (1 + Prompt38::PASS_ROWS) * PLE_STREAMS * HIDDEN * 4;
            Ok(Derived {
                bytes: stores + table + 4 + rows,
                terms: format!(
                    "store_bytes {stores} + the replay table {table} + the lane word 4 + the \
                     arena rows {rows}"
                ),
            })
        }

        fn seq_terms_bytes(&self, _m: &Qwen38Model) -> Result<Option<usize>, GateError> {
            Ok(Some(usize::try_from(
                self.inputs.seq_terms(None).bytes(CTX as u64, 1),
            )?))
        }

        /// H5's planter: the host tier's refusal, through the body's own mut
        /// path to the tier.
        fn plant_refusal(&self, m: &mut Qwen38Model) -> Result<bool, GateError> {
            m.body_parts("gate_qwen38_twocard slots")?
                .2
                .hybrid_mut()
                .plant_refusal("a planted refusal (the slots harness's seam)");
            Ok(true)
        }

        /// H5's window: the tier's own refusal, read through the body.
        fn tier_poisoned(&self, m: &mut Qwen38Model) -> Result<bool, GateError> {
            Ok(m.body_parts("gate_qwen38_twocard slots")?
                .2
                .hybrid()
                .refuse_if_poisoned("slots H5")
                .is_err())
        }

        /// H5's round of several slots: the body's one pass.
        fn step_all(&self, m: &mut Qwen38Model, last: &[u32]) -> Result<Vec<u32>, GpuError> {
            let ones: Vec<[u32; 1]> = last.iter().map(|&t| [t]).collect();
            let rows: Vec<(usize, &[u32])> = ones.iter().map(|t| &t[..]).enumerate().collect();
            Ok(m.step_slots(&rows)?.ids)
        }
    }

    impl PassAdapter for SlotsTier<'_> {
        /// The header's slot nodes ([`Shape::nodes_slot`]): the tier takes
        /// none of a slot's sequence-bound launches.
        fn added_slot_launches(&self, _m: &Qwen38Model) -> Result<Launches, GateError> {
            Ok(Launches {
                n: self.shape.nodes_slot(),
                terms: "the embedding 1 + the PLE conv 1 + the parked rows' copy 1".to_owned(),
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

    /// One slot's side of `slots_wide`: every row's id and logits over the
    /// windows, then the slot's state hash and position.
    struct Side {
        ids: Vec<u32>,
        logits: Vec<Vec<f32>>,
        hash: u64,
        pos: u32,
    }

    /// A window's `n` rows of a slot: `last`, then `fill`'s ids from `at`.
    fn wide_rows(fill: &[u32], last: u32, at: usize, n: usize) -> Result<Vec<u32>, GateError> {
        let more = fill
            .get(at..at + n - 1)
            .ok_or("the slot's window holds too few ids for the wide passes' rows")?;
        let mut rows = vec![last; n];
        rows[1..].copy_from_slice(more);
        Ok(rows)
    }

    /// Slot `s`'s rows of pass `(n0, n1)`.
    fn rows_of((n0, n1): (usize, usize), s: usize) -> usize {
        if s == 0 { n0 } else { n1 }
    }

    /// `rows` (2 to 4) as one verify of the selected slot kept whole, settled
    /// as `Session::commit` of every row settles it (a rollback to the
    /// position past them, the rows kept for the residency): each row's greedy
    /// id and logits.
    fn solo_rows(
        m: &mut Qwen38Model,
        rows: &[u32],
    ) -> Result<(Vec<u32>, Vec<Vec<f32>>), GateError> {
        fn run<const M: usize>(
            m: &mut Qwen38Model,
            rows: &[u32],
        ) -> Result<(Vec<u32>, Vec<Vec<f32>>), GateError> {
            let rows: [u32; M] = rows
                .try_into()
                .map_err(|_| format!("{} rows for a verify of {M}", rows.len()))?;
            let first = m.pos();
            let ids = m.step_rows::<M>(rows)?;
            let logits = m.rows_logits::<M>()?;
            m.rollback(first + u32::try_from(M)?)?;
            m.keep_rows(KeptRows::prefix(M), PassKind::Pair)?;
            Ok((ids.to_vec(), Vec::from(logits)))
        }
        match rows.len() {
            2 => run::<2>(m, rows),
            3 => run::<3>(m, rows),
            4 => run::<4>(m, rows),
            n => Err(format!("a verify of {n} rows: the wide passes give a slot 2 to 4").into()),
        }
    }

    /// [`WIDE_STEPS`] plain steps of the selected slot from `last`, each id
    /// and logits row appended.
    fn wide_steps(
        m: &mut Qwen38Model,
        mut last: u32,
        ids: &mut Vec<u32>,
        logits: &mut Vec<Vec<f32>>,
    ) -> Result<(), GateError> {
        for _ in 0..WIDE_STEPS {
            last = m.step(&[last])?;
            ids.push(last);
            logits.push(m.logits()?);
        }
        Ok(())
    }

    /// Slot `s` alone, from a rewind: its prompt, each wide pass's rows of it
    /// as one verify kept whole (the same rows the pass gives it, which the
    /// tier serves at 2 to 4 columns), every row's id and logits read, then
    /// the plain steps.
    fn wide_alone(a: &SlotsTier<'_>, m: &mut Qwen38Model, s: usize) -> Result<Side, GateError> {
        m.select_slot(s)?;
        a.rewind(m)?;
        let mut last = a.prompt(m, s)?;
        let (mut ids, mut logits, mut at) = (Vec::new(), Vec::new(), 0);
        for pass in WIDE_PASSES {
            let n = rows_of(pass, s);
            let (out, rows) = solo_rows(m, &wide_rows(a.fill(s), last, at, n)?)?;
            last = out[n - 1];
            ids.extend(out);
            logits.extend(rows);
            at += n - 1;
        }
        wide_steps(m, last, &mut ids, &mut logits)?;
        Ok(Side {
            ids,
            logits,
            hash: a.state_hash(m)?,
            pos: m.pos(),
        })
    }

    /// Both slots from a rewind, each its prompt, then every [`WIDE_PASSES`]
    /// pass as one pass of both slots' rows (`GpuModel::step_slots`, in graph
    /// mode, so the tier captures each width), every row's id and logits
    /// read, then each slot's plain steps.
    fn wide_together(a: &SlotsTier<'_>, m: &mut Qwen38Model) -> Result<[Side; 2], GateError> {
        for s in (0..2).rev() {
            m.select_slot(s)?;
            a.rewind(m)?;
        }
        let mut last = [0; 2];
        for (s, l) in last.iter_mut().enumerate() {
            m.select_slot(s)?;
            *l = a.prompt(m, s)?;
        }
        let mut side = [(Vec::new(), Vec::new()), (Vec::new(), Vec::new())];
        let mut at = [0; 2];
        for pass in WIDE_PASSES {
            let n = [pass.0, pass.1];
            let rows = [
                wide_rows(a.fill(0), last[0], at[0], n[0])?,
                wide_rows(a.fill(1), last[1], at[1], n[1])?,
            ];
            let out = m.step_slots(&[(0, &rows[0]), (1, &rows[1])])?.ids;
            let all = m.slots_logits()?;
            if out.len() != n[0] + n[1] || all.len() != n[0] + n[1] {
                return Err(format!(
                    "a pass of {} and {} rows read back {} ids and {} logits rows",
                    n[0],
                    n[1],
                    out.len(),
                    all.len()
                )
                .into());
            }
            let mut from = 0;
            for (s, (ids, logits)) in side.iter_mut().enumerate() {
                ids.extend_from_slice(&out[from..from + n[s]]);
                logits.extend_from_slice(&all[from..from + n[s]]);
                last[s] = out[from + n[s] - 1];
                at[s] += n[s] - 1;
                from += n[s];
            }
        }
        let mut done = Vec::with_capacity(2);
        for (s, (mut ids, mut logits)) in side.into_iter().enumerate() {
            m.select_slot(s)?;
            wide_steps(m, last[s], &mut ids, &mut logits)?;
            done.push(Side {
                ids,
                logits,
                hash: a.state_hash(m)?,
                pos: m.pos(),
            });
        }
        done.try_into()
            .map_err(|_| "the wide passes' two slots".into())
    }

    /// What differs between slot `s`'s side `got` and its solo run's `want`,
    /// by the harness's four parts: ids, last logits (here every row's, bit
    /// for bit), state hash and position.
    fn wide_off(s: usize, want: &Side, got: &Side) -> Vec<String> {
        let mut off = Vec::new();
        if want.ids != got.ids {
            let at = want.ids.iter().zip(&got.ids).position(|(x, y)| x != y);
            off.push(format!("slot {s} ids (first apart at {at:?})"));
        }
        let row = want
            .logits
            .iter()
            .zip(&got.logits)
            .position(|(x, y)| !bits_equal(x, y));
        if row.is_some() || want.logits.len() != got.logits.len() {
            off.push(format!("slot {s} logits (first row apart {row:?})"));
        }
        if want.hash != got.hash {
            off.push(format!("slot {s} state"));
        }
        if want.pos != got.pos {
            off.push(format!("slot {s} position {} (solo {})", got.pos, want.pos));
        }
        off
    }

    /// `slots_wide` (module doc) on the load `m`: each slot's solo run first,
    /// then the passes of both, in graph mode.
    fn slots_wide(a: &SlotsTier<'_>, m: &mut Qwen38Model) -> Result<bool, GateError> {
        m.set_mode(StepMode::Graph);
        let want = [wide_alone(a, m, 0)?, wide_alone(a, m, 1)?];
        let got = wide_together(a, m)?;
        let off: Vec<String> = (0..2)
            .flat_map(|s| wide_off(s, &want[s], &got[s]))
            .collect();
        let ok = off.is_empty();
        let rows: Vec<usize> = WIDE_PASSES.iter().map(|&(n0, n1)| n0 + n1).collect();
        println!(
            "slots wide: passes of {WIDE_PASSES:?} rows a slot (the tier's {rows:?} columns) and \
             {WIDE_STEPS} plain steps after them, {} ids and logits rows against each slot's own \
             rows run alone as verifies kept whole on the same load: {} {}",
            want[0].ids.len() + want[1].ids.len(),
            if ok {
                "every row's id and logits, each slot's state hash and position bit for bit"
                    .to_string()
            } else {
                format!("differs in {}", off.join(", "))
            },
            verdict(ok)
        );
        Ok(ok)
    }

    /// `--slots`' structure line over the load `m` after its wide passes:
    /// the tier's graph of each of 5 to 8 columns holds nine nodes a tier
    /// layer of `layers` and the fault copy, and the tier is opened for the
    /// port's columns.
    fn structure_wide(m: &Qwen38Model, layers: &[usize]) -> Result<bool, GateError> {
        let tier = m
            .body(NAME)?
            .hybrid()
            .tiers()
            .first()
            .ok_or("the slots load holds no tier")?;
        let want = Some(TIER_LAYER_NODES * layers.len() + 1);
        let graphs: Vec<Option<usize>> = (WIDE_FROM..=Prompt38::PASS_ROWS)
            .map(|c| tier.graph_nodes(Chain::Cols(c)))
            .collect();
        let ok = graphs.iter().all(|&g| g == want) && tier.cols() == Prompt38::PASS_ROWS;
        println!(
            "structure: the tier's graphs of the 5, 6, 7, 8-column passes hold {graphs:?} nodes, \
             {TIER_LAYER_NODES} a tier layer ({}) and the fault copy ({want:?}), the tier opened \
             for {} columns (want {}): {}",
            layers.len(),
            tier.cols(),
            Prompt38::PASS_ROWS,
            verdict(ok)
        );
        Ok(ok)
    }

    /// A clause's verdict: a call that failed inside it is its red, the error
    /// printed on its line.
    fn red(clause: &str, r: Result<bool, GateError>) -> bool {
        r.unwrap_or_else(|e| {
            println!("{clause}: a call failed: {e} {}", verdict(false));
            false
        })
    }

    /// `--slots` (module doc) on plan (b′) under `gib` GiB: the harness's
    /// contracts over the load planned for two slots, then `slots_wide` and
    /// the wide structure line on that load, then the harness's one-slot
    /// clause on a load of one.
    fn slots_clauses(
        (path, file, inputs, bp): (&Path, &Split, &PlanInputs, &Machine),
        (gib, host, ub): (u64, HostCfg, usize),
    ) -> Result<bool, GateError> {
        let [w0, w1] = [0, 1].map(|s| SLOT_PROMPTS[s] + SLOT_FILL);
        let ids = prose_ids("qwen4exp", w0 + w1)?;
        let (a0, a1) = ids.split_at(w0);
        let a = SlotsTier {
            path,
            inputs,
            bp,
            gib,
            host,
            ub,
            shape: crate::q38_arch::shape_of(&inputs.spec, file)?,
            windows: [a0.to_vec(), a1.to_vec()],
        };
        let plan = a.plan(slots_gate::STREAMS)?;
        let layers = tier_layers(&plan);
        let missing: Vec<usize> = plan
            .tier_n_l
            .first()
            .map(|t| (0..t.len()).filter(|&l| t[l] == 0).collect())
            .unwrap_or_default();
        let mut ok = missing.is_empty();
        println!(
            "slots plan: plan (b′) for {} slots under a card budget of {gib} GiB: stage {} \
             experts, tier {} experts on {} layers; precondition: a tier expert on every layer \
             (missing {missing:?}): {}",
            slots_gate::STREAMS,
            plan.cards[0].experts,
            plan.cards.get(1).map_or(0, |c| c.experts),
            layers.len(),
            verdict(ok)
        );
        drop(plan);
        let mut s = slots_gate::interleave(&a)?;
        s.one_pass();
        let (harness, mut m, one_slot) = s.finish_keep();
        ok &= harness;
        ok &= red("slots wide", slots_wide(&a, &mut m));
        ok &= red("structure", structure_wide(&m, &layers));
        Ok(one_slot.run(m) && ok)
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[
            CARD_BUDGET,
            HOST_POPULATE,
            HOST_LOCK,
            CARD_DONTNEED,
            bloomery_gpu::arch::qwen3moe::ubatch::UBATCH_ENV,
        ])?;
        let host = levers.host();
        let args = parse_args()?;
        let path_buf = ref_model_path()?;
        let path = path_buf.as_path();
        let split = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::describe(&split)?;
        let mut pass = tier::witness(
            "model file",
            path.to_string_lossy().into_owned(),
            MODEL.to_string(),
        );
        // The plans' budget is the search's ([`chosen_plan`]): a caller's budget
        // lever is not read, so one that disagrees with the fixture's header is
        // refused here as every other gate refuses it.
        tier::plan_levers(&split, &levers, 0)?;
        let ub = ubatch_for(CTX)?;
        let layers = inputs.stack();
        let bp = bp_machine(layers, ub as u64, tier_batch(&inputs.spec, ub as u64))?;
        let Chosen {
            plan,
            tight: tight_3k,
            gib,
        } = chosen_plan(&inputs, &bp, CTX)?;
        pass &= tight_3k;
        let tl = tier_layers(&plan);
        let ids = prose_ids("qwen4exp", PROMPT + 32)?;
        let prompt_ids = prose_ids("qwen4exp", PROMPT_U)?;
        // The unset-rule clause (the module header), before this gate's own
        // loads: the cards as empty as the clause's band holds.
        if tier::run_clause(
            "records-unset: the unset rule keeps bp at the real file's tier size",
            Tag::FileBound,
        )? {
            pass &= match records_unset(&ids[..UNSET_TOKENS]) {
                Ok(p) => p,
                Err(e) => {
                    println!("records-unset: {e}");
                    false
                }
            };
        }
        let mut off_prompt = None;
        let mut last: Option<Session<Body38>> = None;
        let mut off_run = None;
        if args.union {
            tier::sc(
                "--union: the tier load's history, prompt calls and structure = the union reference's",
            )?;
            let mut um = bp.clone();
            um.tiers.clear();
            let uplan = union_plan(&plan, &um)?;
            println!(
                "reference: the stage card holds {} experts, the union of the stage's and the tier's",
                uplan.n_l.iter().sum::<u64>()
            );
            let mut r = open(path, &uplan, &inputs, (host, ub, CTX), Residency::Off)?;
            if !r.model().body(NAME)?.hybrid().tiers().is_empty() {
                return Err("the reference load holds a tier card".into());
            }
            let want = history(&mut r, &ids)?;
            let want_prompt = prompt_run(&mut r, &prompt_ids, false)?;
            let want_short = gemm_run(&mut r, &prompt_ids[..PROMPT_SHORT])?;
            drop(r);
            let mut t = open(path, &plan, &inputs, (host, ub, CTX), Residency::Off)?;
            let got = history(&mut t, &ids)?;
            pass &= same("two cards", &want, &got);
            pass &= structure(&t, &want, &got, &tl, &uplan)?;
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
            // The flips landing at all is the router's skew (the double of the router is R4's):
            // a fixture's generated router has none, so the fixture tier holds the rest of the
            // arm and prints the counts.
            let skewed = tier::run_clause(
                "--residency: flips landed under the router's skew (the double of the router is R4's)",
                Tag::FileBound,
            )?;
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
            let mut t = open(path, &plan, &inputs, (host, ub, CTX), residency)?;
            pass &= residency_clauses(
                &mut t,
                (&ids, &prompt_ids),
                (off_run.as_ref(), off_prompt.as_ref()),
                (&split, &plan),
                skewed,
            )?;
            last = Some(t);
        }
        if args.lost {
            tier::sc("--lost: the tier's stream held behind a host flag is named as a lost card")?;
            let mut s = match last.take() {
                Some(s) => s,
                None => open(path, &plan, &inputs, (host, ub, CTX), Residency::Off)?,
            };
            pass &= lost_case(s.model_mut(), &ids)?;
        }
        if args.prompt4k {
            tier::sc("--prompt4k: a 4,096-id prompt call, two cards = the union")?;
            drop(last.take());
            pass &= prompt4k(path, &inputs, host)?;
        }
        if args.slots {
            tier::sc(
                "--slots: a tier load planned for two slots through the slot harness, wide passes of 5 to 8 columns",
            )?;
            drop(last.take());
            pass &= slots_clauses((path, &split, &inputs, &bp), (gib, host, ub))?;
        }
        let (ran, deferred) = tier::tally();
        println!(
            "clauses: {ran} ran, {deferred} left to the real tier (deferred(real) lines above)"
        );
        if !pass {
            return Err(checks_failed());
        }
        println!("PASSED: {NAME}");
        Ok(())
    }
}
