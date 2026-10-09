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
//! The two cards' load then runs a group leg, from a reset each: the first
//! [`GROUP_PROMPT`] ids (two batches) as one prompt call at groups of one
//! batch and at groups of two (`set_prefill_group`), each call's token and
//! logits row read.
//!
//! Clauses, against the reference:
//! - decode bits: the step and pair legs' tokens, kept counts and logits
//!   rows bit for bit the reference's;
//! - call bits: the call leg's token and logits row bit for bit the
//!   reference's (no decode step follows the call, so a decode defect cannot
//!   turn this clause red);
//! - group bits: the group leg's call at groups of two bit for bit its call
//!   at groups of one — a tiered layer's join reads its unit's route weights
//!   and tier places after the next unit's front has routed;
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
//!
//! `--residency`: the serving placements' residency
//! (`bloomery_levers::GLM_RESIDENCY_UNSET`, `mid-p0-s1`) beside the tier, two
//! loads of plan (b′) through the binaries' opens, one after the other: the
//! plain load (`app::arch::glm5next::open_resident`, one KDA lane), then the
//! NextN load (`open_nextn`, the draft beside the target), each under the
//! word. The residency machine moves the stage card's experts, so a card
//! expert served on the host computes other bits than on the card and no
//! one-card load is a bit reference here; the clauses hold the schedule to
//! itself:
//!
//! - `res-twice` (determinism): a history — a clear (the residency back to
//!   its seed), the first [`PROMPT`] ids as one prompt call, then [`STEPS`]
//!   greedy steps, each step's argmax and logits FNV — run twice gives the
//!   same tokens, logits, boundaries and landings, and flips landed;
//! - `res-mtp-twice`: on the NextN load, the drafted history (the MTP window,
//!   [`DRAFT_STEPS`] ids) twice gives the same ids, windows, boundaries and
//!   landings, flips landed;
//! - `res-tier-fixed` and `res-mtp-tier-fixed` (no tier expert moves): after
//!   the histories, the host slot map's tier entries are the load's, the
//!   tier card's set is the map's tier rows, and the stage card's copy of the
//!   map, read back, is the host map's stage view with every tier expert on
//!   the host mark — no card slot names a tier expert; and the histories sent
//!   every tier layer a routed slot, so the tier served under the machine.
//!
//! `--nextn`: the NextN draft beside the tier at residency `off`, against the
//! one-card NextN load holding the same expert sets:
//!
//! - `nextn-refuse`: a NextN plan of (b′) whose next-token layer's routed
//!   stack puts expert 0 on the tier is refused by name before anything
//!   uploads (`bloomery_gpu_glm5next::NEXTN_ON_TIER`: the walk serves that
//!   layer on the host through the batch port and runs no tier join);
//! - then two loads, one after the other: the reference, the NextN plan's
//!   target plan with its tier segments joined onto the stage card
//!   ([`union_plan`]) and no tier, opened by `Body::open_placed_nextn_with`;
//!   the two cards through `open_nextn`. `nextn-bits`: the drafted history's
//!   ids and windows (each window's proposal and kept rows) on the two cards
//!   are the reference's, a draft kept and one rejected among them;
//! - `nextn-stage` (the draft's walk stays on the stage card): on the two
//!   cards the slot map's row for the next-token layer holds no tier and no
//!   card entry, the windows enqueued no tier batch service (the walks use
//!   the batch port, the verifies the step port), and the windows sent every
//!   tier layer a routed slot.
//!
//! `--records`: the `load` record `generate_glm5next` prints (the binary
//! built beside this one, run as a child: [`RECORD_TOKENS`] prompt ids, `-n
//! 2`) names the cards the model runs on — `cards` the A6000 alone under
//! `--place a` with no tier fields, the A6000 then the 3090 under `--place bp`
//! with the tier's experts (`tier_experts` > 0) — as a V4.1 binary's load
//! record names them (`generate::with_cards`): the record read by the
//! binary's kinds (`record::Log`, `tools/bloomery/records.py`'s rules), each
//! `cards` item exactly the driver name of the device the placement resolves
//! to; the same check on
//! the line with its first card's name grown by a letter inside the brackets
//! is red. A child run with no `--place` takes the common unset rule
//! (`generate::Place::choose`, by the GLM family's rule): on this gate's two
//! cards its `place unset` record keeps `bp`, the offer's tier at or past the
//! rule's break-even, naming the tier's experts in the plan at the child's
//! default context and the rule's break-even and basis. Three loads, one a
//! process.
//!
//! Tiers (`BLOOMERY_TIER`): the card budget is [`BUDGET`] in the real tier and the header's in the fixture tier
//! (`shared/glm5next_tier.rs`). The clauses that need the file's trained weights (a pair that accepts a proposal, flips
//! landed and an expert admitted, the plan with no budget leaving tier experts, the unset rule's bp at the real file's
//! tier size) are `Tag::FileBound` and the fixture tier defers them to the real tier by name; every bit-for-bit clause
//! runs on the fixture.

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
#[path = "shared/glm5next_tier.rs"]
mod glm5next_tier;

#[cfg(feature = "glm5next")]
#[path = "shared/quiet.rs"]
mod quiet;

#[cfg(feature = "glm5next")]
mod gate {
    use std::time::Instant;

    use app::arch::glm5next::{GlmCfg, open_nextn, open_pair, open_resident};
    use app::mtp::MtpDraft;
    use app::{Loaded, OpenArgs, Session};
    use bloomery_gpu::host::PassKind;
    use bloomery_gpu::host::batch::TierBatchStats;
    use bloomery_gpu::host::slots::Slot;
    use bloomery_gpu::host::swap::Residency;
    use bloomery_gpu::host::tier::TierSet;
    use bloomery_gpu::hybrid::HOST;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::record::{self, Fields, ReadError};
    use bloomery_gpu_gates::residency38::glm_machine;
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{Fnv1a64, GateError, checks_failed, verdict};
    use bloomery_gpu_glm5next::{
        Body, Glm5nextModel, NEXTN_ON_TIER, PrefillMode, TIER_BEFORE_UPLOAD, feed,
        set_prefill_group,
    };
    use bloomery_levers::{CARD_DONTNEED, GLM_RESIDENCY_UNSET, HOST_LOCK, HOST_POPULATE};
    use model::arch::glm5next::names;
    use model::arch::glm5next::place::{self, KdaLanes, NextnInputs, NextnPlan, PlanInputs};
    use model::placement::{
        self, Device, ExpertList, Format, Machine, Plan, PlanLevers, Role, Row, Segment,
        workstation,
    };
    use runtime::swaprule::KeptRows;
    use runtime::{Advance as _, Committed, Out, PassSink, Stop, Target, Want};

    use crate::glm5next_tier;
    use crate::quiet::Quiet;

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
    /// Prompt positions of the group leg's calls: two whole batches, one
    /// group at groups of two.
    const GROUP_PROMPT: usize = 1024;
    /// The vocabulary: a wrong draft is the right one plus one, within it.
    const N_VOCAB: u32 = 154_880;
    /// Ids the drafted histories generate after their prompt call.
    const DRAFT_STEPS: usize = 48;
    /// The rows a drafted window verifies: the target's next token and one
    /// proposal.
    const PAIR: usize = 2;
    /// A refusal before the upload reads the files' headers and the plan only.
    const REFUSE_BOUND_S: f64 = 120.0;
    /// Prompt ids the `--records` arm's child runs.
    const RECORD_TOKENS: usize = 8;
    /// The GLM family's unset rule as the child names it
    /// (`shared/glm5next_place.rs`'s `TIER_RULE`): the tier experts that keep
    /// the offer's tier card, and the card file that decides them.
    const BREAK_EVEN: u64 = 526;
    const TIER_BASIS: &str = "docs/cards/glmbp-ab.card";
    /// The experts the child's plan of `bp` puts on this box's 3090 at its
    /// default context, a band for the free bytes each census reads.
    const BP_TIER_EXPERTS: std::ops::RangeInclusive<u64> = 1503..=1529;

    /// The prompt: the `d1k` set's prefill ids.
    fn prompt() -> Result<Vec<u32>, GateError> {
        let prefill = glm5next_tier::d1k_prefill()?;
        let n = PROMPT.max(CALL_PROMPT).max(GROUP_PROMPT);
        if prefill.len() < n {
            return Err(format!(
                "the d1k set's prefill is {} ids, the gate reads {n}",
                prefill.len()
            )
            .into());
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

    /// The tiered load through the binaries' open, refused unless its tier's
    /// `Gpu` is the 3090.
    fn open_two(
        machine: impl Fn(usize) -> Machine,
        cfg: &GlmCfg,
    ) -> Result<Session<Body>, GateError> {
        let t0 = Instant::now();
        let file = glm5next_tier::open()?;
        let args = OpenArgs {
            place: Place::Bp.name(),
            machine,
            ctx: CTX,
            mode: StepMode::Graph,
            cfg: cfg.clone(),
        };
        let s = open_pair(file, args, &mut Quiet)?.ok_or("the open planned nothing")?;
        tier_on_3090(&s, "load", t0)?;
        Ok(s)
    }

    /// Refused unless `s`'s tier `Gpu` is the 3090; prints the load line,
    /// `what` its name.
    fn tier_on_3090(s: &Session<Body>, what: &str, t0: Instant) -> Result<(), GateError> {
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
            "{what} in {:.1} s: stage {} ({} B), tier {} experts on {tier_device} ({} B)",
            t0.elapsed().as_secs_f64(),
            m.gpu().device_name()?,
            m.resident_bytes(),
            tier.set().experts(),
            tier.weights().resident_bytes(),
        );
        Ok(())
    }

    /// The reference: `plan`, whose stage card holds the union
    /// ([`union_plan`]), opened on card 0 with no tier.
    fn open_union(
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        cfg: &GlmCfg,
    ) -> Result<Session<Body>, GateError> {
        let t0 = Instant::now();
        let file = glm5next_tier::open()?;
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
            m.keep_rows(KeptRows::prefix(kept), PassKind::Pair)?;
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

    /// The group leg on `s`: the first [`GROUP_PROMPT`] ids of `prompt` as one
    /// call at groups of one batch, then at groups of two, each from a
    /// reset, its token and logits row; the session left at groups of one.
    fn group_leg(s: &mut Session<Body>, prompt: &[u32]) -> Result<[Leg; 2], GateError> {
        let m = s.model_mut();
        let mut out = [Leg::default(), Leg::default()];
        for (leg, g) in out.iter_mut().zip([1, 2]) {
            set_prefill_group(m, g)?;
            m.reset()?;
            let tok = feed(m, &prompt[..GROUP_PROMPT])?;
            leg.tokens.push(tok);
            leg.logits.push(m.logits()?);
        }
        set_prefill_group(m, 1)?;
        m.reset()?;
        Ok(out)
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

    /// Which of the gate's runs the command line asks for (module header).
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Arm {
        /// No flag: the tier against the union, residency off, no draft.
        Bits,
        /// `--residency`.
        Residency,
        /// `--nextn`.
        Nextn,
        /// `--records`.
        Records,
    }

    /// The arm the command line names; any other argument is refused by name.
    fn arm_of() -> Result<Arm, GateError> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
            [] => Ok(Arm::Bits),
            ["--residency"] => Ok(Arm::Residency),
            ["--nextn"] => Ok(Arm::Nextn),
            ["--records"] => Ok(Arm::Records),
            _ => Err(format!(
                "{NAME} takes no flag, --residency, --nextn or --records (the module header), not \
                 {args:?}"
            )
            .into()),
        }
    }

    pub fn run() -> Result<(), GateError> {
        let arm = arm_of()?;
        let levers =
            bloomery_levers::at_main(&tier::acts_on(&[HOST_POPULATE, HOST_LOCK, CARD_DONTNEED])?)?;
        glm5next_tier::init_file()?;
        // Both plans run under one budget that no environment moves: the real tier's
        // [`BUDGET`], a parameter of the real file (the union it leaves fits the A6000); the
        // fixture tier's, the header's, which the generator planned to put half of the experts
        // on a card.
        let place_levers = match Tier::from_env()? {
            Tier::Real => PlanLevers {
                card_budget_bytes: Some(BUDGET),
            },
            Tier::Fixture => glm5next_tier::plan_levers(&levers, 0)?,
        };
        let cfg = GlmCfg {
            place: place_levers,
            host: levers.host(),
            prefill: PrefillMode::Batch,
            group: 1,
        };
        let inputs = PlanInputs::read(&glm5next_tier::open()?)?;
        let pass = match arm {
            Arm::Bits => bits(&cfg, &inputs)?,
            Arm::Residency => residency_arm(&cfg, &inputs)?,
            Arm::Nextn => nextn_arm(&cfg, &inputs)?,
            Arm::Records => records_arm()?,
        };
        println!("{NAME}: {}", tier::tally_line());
        if !pass {
            return Err(checks_failed());
        }
        println!("PASSED: {NAME}");
        Ok(())
    }

    /// The bits arm (no flag): the union reference, then the two cards.
    fn bits(cfg: &GlmCfg, inputs: &PlanInputs) -> Result<bool, GateError> {
        let cfg = cfg.clone();
        let bp = glm_machine(Place::Bp, Some(place::tier_batch(&inputs.hp)), 1)?;
        let layers = inputs.model.layers;
        let two = bp(layers);
        let plan = inputs.plan_lanes(&two, u64::try_from(CTX)?, &cfg.place, KdaLanes::Two)?;
        let budget = cfg
            .place
            .card_budget_bytes
            .ok_or("the gate plans under a card budget")?;
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
            "plan (b′) under a card budget of {budget} B: stage {} experts (n_l {:?}), tier {} \
             experts (per layer {:?}) on {} layers, host {} experts",
            plan.cards[0].experts,
            held(&plan.n_l),
            plan.cards.get(1).map_or(0, |c| c.experts),
            held(tier_n),
            tier_layers.len(),
            plan.host.experts
        );
        // The card's whole room holds every routed expert of a file this small, which leaves the
        // tier idle and the plan refused by name; only the real file's host set needs a tier there.
        if tier::run_clause(
            "the plan with no card budget leaves the tier experts",
            Tag::FileBound,
        )? {
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
        }
        let hybrid: Vec<usize> = (0..layers)
            .filter(|&l| plan.n_l.get(l).is_some_and(|&n| n > 0))
            .collect();
        let mut pass = true;
        tier::sc("precondition sets: the tier holds experts on the stage's hybrid layers")?;
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
            let mut s = open_union(&uplan, inputs, &cfg)?;
            legs(&mut s, &prompt)?
        };
        let (got, [g1, g2]) = {
            let mut s = open_two(bp, &cfg)?;
            let run = legs(&mut s, &prompt)?;
            (run, group_leg(&mut s, &prompt)?)
        };
        for name in [
            "decode bits: step",
            "decode bits: pair",
            "call bits",
            "group bits: groups of two against groups of one",
            "structure: the stage's step graph has the reference's node count",
        ] {
            tier::sc(name)?;
        }
        pass &= same("decode bits: step", &want.step, &got.step);
        pass &= same("decode bits: pair", &want.pair, &got.pair);
        pass &= same("call bits", &want.call, &got.call);
        pass &= same("group bits: groups of two against groups of one", &g1, &g2);
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
        // A pass that keeps its draft is the file's NextN layer: a fixture's is random, whose
        // proposals the target refuses, so only the real tier requires one kept.
        let accepts = tier::premise(
            "precondition pair: a pass keeps its draft",
            Tag::FileBound,
            got.accepts > 0,
        )?;
        if accepts && got.rejects > 0 {
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
        tier::sc("precondition sent: every tier layer was sent a routed slot")?;
        pass &= sent_every("decode", got.sent_decode.as_ref(), &tier_layers, false);
        pass &= sent_every("call", got.sent_call.as_ref(), &tier_layers, true);
        Ok(pass)
    }

    // ------------------------------------------ residency beside the tier

    /// A self-consistency clause's verdict, registered as one ([`reported`]).
    fn held(clause: &str, r: Result<bool, GateError>) -> bool {
        reported(clause, tier::sc(clause).and(r))
    }

    /// A clause's verdict, an error it met printed as its red line.
    fn reported(clause: &str, r: Result<bool, GateError>) -> bool {
        r.unwrap_or_else(|e| {
            println!("FAIL {clause}: error \"{e}\"");
            false
        })
    }

    /// What a history saw (module header): the prompt's argmax, each step's
    /// argmax and logits FNV, each boundary's ended pass and the flips that
    /// landed there.
    #[derive(PartialEq)]
    struct History {
        first: u32,
        tokens: Vec<u32>,
        fnvs: Vec<u64>,
        passes: Vec<(PassKind, usize)>,
        landings: Vec<usize>,
    }

    /// The history from a clear (the residency back to its seed), with what
    /// it sent the tier.
    fn history(s: &mut Session<Body>, ids: &[u32]) -> Result<(History, Option<Sent>), GateError> {
        s.clear()?;
        s.model_mut().body_parts(NAME)?.2.take_residency_passes();
        let before = tier_counts(s.model())?;
        let mut next = s.prompt(ids, Want::Argmax)?.argmax();
        let first = next;
        let (mut tokens, mut fnvs) = (Vec::with_capacity(STEPS), Vec::with_capacity(STEPS));
        for _ in 0..STEPS {
            let out = s.step(next, Want::Logits)?;
            if let Out::Logits { row, .. } = out {
                fnvs.push(Fnv1a64::default().f32s(row).value());
            }
            next = out.argmax();
            tokens.push(next);
        }
        let sent = sent_between(before, tier_counts(s.model())?);
        let passes = s.model_mut().body_parts(NAME)?.2.take_residency_passes();
        let h = History {
            first,
            tokens,
            fnvs,
            passes: passes.iter().map(|&(k, r)| (k, r.kept)).collect(),
            landings: passes.iter().map(|(_, r)| r.landed).collect(),
        };
        println!(
            "history: prompt + {STEPS} steps, {} boundaries, {} flips landed",
            h.passes.len(),
            h.landings.iter().sum::<usize>()
        );
        Ok((h, sent))
    }

    /// The windows' proposals and kept rows, as the generation reports them.
    #[derive(Default)]
    struct Windows(Vec<(bool, usize)>);

    impl PassSink<Session<Body>> for Windows {
        type Error = GateError;

        fn begin(&mut self, _: &Session<Body>) -> Result<(), GateError> {
            Ok(())
        }

        fn pass(
            &mut self,
            _: &Session<Body>,
            c: &Committed,
            _: &[u32],
            _: std::time::Duration,
        ) -> Result<(), GateError> {
            self.0.push((c.proposed, c.kept));
            Ok(())
        }
    }

    /// What a drafted history saw: the generated ids, each window's proposal
    /// and kept rows, each boundary's ended pass and the flips that landed.
    #[derive(PartialEq)]
    struct Drafted {
        tokens: Vec<u32>,
        windows: Vec<(bool, usize)>,
        passes: Vec<(PassKind, usize)>,
        landings: Vec<usize>,
    }

    /// The drafted history from a clear: the prompt call with the draft's
    /// store walks, then MTP windows (`Speculative<MtpDraft<Body>, 2>`) for
    /// [`DRAFT_STEPS`] ids; with what the windows alone sent the tier.
    fn drafted(s: &mut Session<Body>, ids: &[u32]) -> Result<(Drafted, Option<Sent>), GateError> {
        s.clear()?;
        s.model_mut().body_parts(NAME)?.2.take_residency_passes();
        let draft = MtpDraft::open(s.model(), PrefillMode::Batch, StepMode::Eager)?;
        let mut spec = s.with_draft::<MtpDraft<Body>, PAIR>(draft, &mut Quiet)?;
        let first = spec.prompt(s, ids)?;
        let before = tier_counts(s.model())?;
        let mut w = Windows::default();
        let stop = Stop::new(DRAFT_STEPS, s.ctx())?;
        let out = runtime::generate(s, &mut spec, ids, first, &stop, &mut w)?;
        let sent = sent_between(before, tier_counts(s.model())?);
        let passes = s.model_mut().body_parts(NAME)?.2.take_residency_passes();
        let d = Drafted {
            tokens: out.tokens,
            windows: w.0,
            passes: passes.iter().map(|&(k, r)| (k, r.kept)).collect(),
            landings: passes.iter().map(|(_, r)| r.landed).collect(),
        };
        println!(
            "drafted history: prompt + {} ids in {} windows, {} boundaries, {} flips landed",
            d.tokens.len(),
            d.windows.len(),
            d.passes.len(),
            d.landings.iter().sum::<usize>()
        );
        Ok((d, sent))
    }

    /// Every tier entry of the load's host slot map, `(layer, id, slot)`,
    /// and the tier card's set.
    struct TierHeld {
        entries: Vec<(usize, u32, Slot)>,
        set: TierSet,
    }

    fn tier_held(s: &Session<Body>) -> Result<TierHeld, GateError> {
        let hybrid = s.model().body(NAME)?.hybrid();
        let map = hybrid.slots();
        let mut entries = Vec::new();
        for l in map.layers() {
            for id in 0..u32::try_from(map.n_expert())? {
                if let Some(slot @ Slot::Tier { .. }) = map.slot(l, id) {
                    entries.push((l, id, slot));
                }
            }
        }
        let set = hybrid
            .tiers()
            .first()
            .ok_or("the (b′) load holds no tier card")?
            .set()
            .clone();
        Ok(TierHeld { entries, set })
    }

    /// The layers the loaded map sends tier experts to.
    fn tier_layers_of(s: &Session<Body>) -> Result<Vec<usize>, GateError> {
        let map = s.model().body(NAME)?.hybrid().slots();
        let mut out = Vec::new();
        for l in map.layers() {
            if map.on_tier(l)? > 0 {
                out.push(l);
            }
        }
        Ok(out)
    }

    /// `what` (the tier-fixed clause, module header): `s`'s tier entries are
    /// `load`'s, the tier card's set is the host map's tier rows, and the
    /// stage card's copy of the map, read back, is the host map's stage view
    /// with every tier expert on the host mark.
    fn tier_fixed(what: &str, s: &Session<Body>, load: &TierHeld) -> Result<bool, GateError> {
        let now = tier_held(s)?;
        let m = s.model();
        let body = m.body(NAME)?;
        let map = body.hybrid().slots();
        let same_entries = now.entries == load.entries;
        let set_is_map = now.set == load.set && TierSet::of_map(map, 0)? == load.set;
        m.gpu().stream().synchronize()?;
        let copy = body.slot_copy().buf().to_host_vec(m.gpu().stream())?;
        let view_ok = copy == map.stage_view();
        let mut named = Vec::new();
        for &(l, id, _) in &load.entries {
            let at = map
                .row_offset(l)
                .ok_or_else(|| format!("no row offset for layer {l}"))?
                + usize::try_from(id)?;
            if copy.get(at) != Some(&HOST) {
                named.push((l, id, copy.get(at).copied()));
            }
        }
        let moved = load
            .entries
            .iter()
            .filter(|e| !now.entries.contains(e))
            .count();
        let ok =
            !load.entries.is_empty() && same_entries && set_is_map && view_ok && named.is_empty();
        println!(
            "{what}: {} tier entries at load, {moved} of them moved, {} now; the tier's set the \
             map's tier rows {set_is_map}; the stage card's copy the host map's stage view \
             {view_ok}, tier experts at a card slot {named:?}: {}",
            load.entries.len(),
            now.entries.len(),
            verdict(ok)
        );
        Ok(ok)
    }

    /// `what` (determinism): `a` and `b` equal, and flips landed in `a`.
    fn twice<T: PartialEq>(what: &str, a: &T, b: &T, landings: &[usize], skew: bool) -> bool {
        let landed: usize = landings.iter().sum();
        let ok = a == b && (landed > 0 || !skew);
        println!(
            "{what}: the same schedule twice the same {}, {landed} flips landed: {}",
            a == b,
            verdict(ok)
        );
        ok
    }

    /// The `--residency` arm (module header).
    fn residency_arm(cfg: &GlmCfg, inputs: &PlanInputs) -> Result<bool, GateError> {
        let lever = Residency::parse(GLM_RESIDENCY_UNSET)?;
        let bp = glm_machine(Place::Bp, Some(place::tier_batch(&inputs.hp)), 1)?;
        let prompt = prompt()?;
        let ids = &prompt[..PROMPT];
        let open_args = || OpenArgs {
            place: Place::Bp.name(),
            machine: bp,
            ctx: CTX,
            mode: StepMode::Graph,
            cfg: cfg.clone(),
        };
        let file = glm5next_tier::open;
        println!("{NAME} --residency: plan (b′) under BLOOMERY_RESIDENCY {GLM_RESIDENCY_UNSET}");
        let mut pass = true;
        {
            let t0 = Instant::now();
            let mut s = open_resident(file()?, open_args(), lever, &mut Quiet)?
                .ok_or("the resident open planned nothing")?;
            tier_on_3090(&s, "resident load", t0)?;
            s.model_mut().body_parts(NAME)?.2.log_residency(0);
            let load = tier_held(&s)?;
            let tier_layers = tier_layers_of(&s)?;
            let a = history(&mut s, ids);
            let b = history(&mut s, ids);
            match (a, b) {
                (Ok((a, sent)), Ok((b, _))) => {
                    tier::sc("res-twice: the same history twice")?;
                    pass &= twice("res-twice", &a, &b, &a.landings, glm5next_tier::skew()?);
                    pass &= sent_every("residency", sent.as_ref(), &tier_layers, false);
                }
                (a, b) => {
                    let e = a.err().or(b.err()).map(|e| e.to_string());
                    println!(
                        "FAIL res-twice: a history: error \"{}\"",
                        e.unwrap_or_default()
                    );
                    pass = false;
                }
            }
            pass &= held("res-tier-fixed", tier_fixed("res-tier-fixed", &s, &load));
        }
        {
            let t0 = Instant::now();
            let mut s = open_nextn(file()?, open_args(), lever, &mut Quiet)?
                .ok_or("the NextN open planned nothing")?;
            tier_on_3090(&s, "resident NextN load", t0)?;
            s.model_mut().body_parts(NAME)?.2.log_residency(0);
            let load = tier_held(&s)?;
            let tier_layers = tier_layers_of(&s)?;
            let a = drafted(&mut s, ids);
            let b = drafted(&mut s, ids);
            match (a, b) {
                (Ok((a, sent)), Ok((b, _))) => {
                    tier::sc("res-mtp-twice: the same drafted history twice")?;
                    pass &= twice("res-mtp-twice", &a, &b, &a.landings, glm5next_tier::skew()?);
                    pass &= sent_every("residency drafted", sent.as_ref(), &tier_layers, false);
                }
                (a, b) => {
                    let e = a.err().or(b.err()).map(|e| e.to_string());
                    println!(
                        "FAIL res-mtp-twice: a drafted history: error \"{}\"",
                        e.unwrap_or_default()
                    );
                    pass = false;
                }
            }
            pass &= held(
                "res-mtp-tier-fixed",
                tier_fixed("res-mtp-tier-fixed", &s, &load),
            );
        }
        Ok(pass)
    }

    // ------------------------------------------------ NextN beside the tier

    /// `np` with the next-token layer's routed gate stack made a routed row
    /// whose expert 0 sits on the tier card: the plan `nextn-refuse` loads.
    /// `model` is the target's tensors with that stack's role routed.
    fn nextn_on_tier_plan<'a>(
        np: &NextnPlan<'a>,
        model: &'a placement::ModelTensors,
        nextn: &NextnInputs,
    ) -> Result<NextnPlan<'a>, GateError> {
        let name = names::ffn_gate_exps(nextn.index);
        let i = model
            .tensors
            .iter()
            .position(|t| t.name == name)
            .ok_or_else(|| format!("the target's tensors hold no {name}"))?;
        let mut plan = np.plan.clone();
        plan.model = model;
        let row = plan
            .rows
            .iter_mut()
            .find(|r| r.tensor == i)
            .ok_or_else(|| format!("the plan has no row of {name}"))?;
        *row = Row {
            tensor: i,
            segments: vec![Segment {
                device: Device::Card(plan.machine.cards.len()),
                format: Format::Unused,
                experts: Some(ExpertList::new(vec![0], model.experts)?),
                resident_bytes: 0,
            }],
            read_bytes: 0,
            stage: 0,
        };
        Ok(NextnPlan {
            plan,
            nextn: np.nextn.clone(),
            arena_bytes: np.arena_bytes,
            headroom_bytes: np.headroom_bytes,
            host_headroom_bytes: np.host_headroom_bytes,
        })
    }

    /// `nextn-refuse` (module header).
    fn nextn_refuse(
        np: &NextnPlan<'_>,
        inputs: &PlanInputs,
        nextn: &NextnInputs,
        cfg: &GlmCfg,
    ) -> Result<bool, GateError> {
        let mut model = np.plan.model.clone();
        let name = names::ffn_gate_exps(nextn.index);
        let t = model
            .tensors
            .iter_mut()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("the target's tensors hold no {name}"))?;
        if t.layer != Some(nextn.index) {
            return Err(format!("{name} is on layer {:?}, not {}", t.layer, nextn.index).into());
        }
        t.role = Role::RoutedExperts;
        let doctored = nextn_on_tier_plan(np, &model, nextn)?;
        let file = glm5next_tier::open()?;
        let t0 = Instant::now();
        let opened = Body::open_placed_nextn_with(
            file,
            &doctored,
            inputs,
            nextn,
            0,
            cfg.host,
            Residency::Off,
        );
        let secs = t0.elapsed().as_secs_f64();
        let (ok, why) = match opened {
            Err(e) => {
                let text = e.to_string();
                (
                    text.contains(NEXTN_ON_TIER)
                        && text.contains(TIER_BEFORE_UPLOAD)
                        && secs < REFUSE_BOUND_S,
                    text,
                )
            }
            Ok(_) => (false, "the open loaded".to_string()),
        };
        println!(
            "nextn-refuse: layer {}'s expert 0 on the tier: the open in {secs:.1} s — {why}: {}",
            nextn.index,
            verdict(ok)
        );
        Ok(ok)
    }

    /// The reference NextN load: `np`, whose stage card holds the union, on
    /// card 0 with no tier.
    fn open_union_nextn(
        np: &NextnPlan<'_>,
        inputs: &PlanInputs,
        nextn: &NextnInputs,
        cfg: &GlmCfg,
    ) -> Result<Session<Body>, GateError> {
        let t0 = Instant::now();
        let file = glm5next_tier::open()?;
        let mut model =
            Body::open_placed_nextn_with(file, np, inputs, nextn, 0, cfg.host, Residency::Off)?;
        model.set_mode(StepMode::Graph);
        if !model.body(NAME)?.hybrid().tiers().is_empty() {
            return Err("the reference NextN load holds a tier card".into());
        }
        let s = Loaded::from_model(model, cfg.clone(), u32::try_from(np.plan.ctx_max)?)
            .ready(&mut Quiet)?;
        println!(
            "reference NextN load in {:.1} s: {} ({} B) holds the union, no tier",
            t0.elapsed().as_secs_f64(),
            s.model().gpu().device_name()?,
            s.model().resident_bytes(),
        );
        Ok(s)
    }

    /// `nextn-stage` (module header): the next-token layer's map row holds no
    /// tier and no card entry, the windows enqueued no tier batch service,
    /// and they sent every tier layer a routed slot.
    fn nextn_stage(
        s: &Session<Body>,
        nextn: usize,
        windows: Option<&Sent>,
        tier_layers: &[usize],
    ) -> Result<bool, GateError> {
        let map = s.model().body(NAME)?.hybrid().slots();
        let (on_tier, on_card) = (map.on_tier(nextn)?, map.on_card(nextn)?);
        let batch = windows.map(|w| w.stats.served);
        let ok = on_tier == 0 && on_card == 0 && batch == Some(0);
        println!(
            "nextn-stage: layer {nextn}'s row: {on_tier} tier, {on_card} card entries; tier batch \
             services in the windows {batch:?}: {}",
            verdict(ok)
        );
        Ok(ok && sent_every("nextn windows", windows, tier_layers, false))
    }

    /// The `--nextn` arm (module header).
    fn nextn_arm(cfg: &GlmCfg, inputs: &PlanInputs) -> Result<bool, GateError> {
        let nextn = NextnInputs::read(inputs)?;
        let bp = glm_machine(Place::Bp, Some(place::tier_batch(&inputs.hp)), 1)?;
        let two = bp(inputs.model.layers);
        let np = inputs.plan_nextn(&two, u64::try_from(CTX)?, &cfg.place, &nextn)?;
        println!(
            "{NAME} --nextn: NextN plan (b′): stage {} experts, tier {} experts, host {} experts; \
             layer {} after the trunk",
            np.plan.cards[0].experts,
            np.plan.cards.get(1).map_or(0, |c| c.experts),
            np.plan.host.experts,
            nextn.index
        );
        let mut pass = held("nextn-refuse", nextn_refuse(&np, inputs, &nextn, cfg));
        let prompt = prompt()?;
        let ids = &prompt[..PROMPT];
        let flat = Machine {
            tiers: Vec::new(),
            ..two.clone()
        };
        let unp = NextnPlan {
            plan: union_plan(&np.plan, &flat)?,
            nextn: np.nextn.clone(),
            arena_bytes: np.arena_bytes,
            headroom_bytes: np.headroom_bytes,
            host_headroom_bytes: np.host_headroom_bytes,
        };
        let want = {
            let mut s = open_union_nextn(&unp, inputs, &nextn, cfg)?;
            drafted(&mut s, ids)?.0
        };
        let t0 = Instant::now();
        let args = OpenArgs {
            place: Place::Bp.name(),
            machine: bp,
            ctx: CTX,
            mode: StepMode::Graph,
            cfg: cfg.clone(),
        };
        let file = glm5next_tier::open()?;
        let mut s = open_nextn(file, args, Residency::Off, &mut Quiet)?
            .ok_or("the NextN open planned nothing")?;
        tier_on_3090(&s, "NextN load", t0)?;
        let tier_layers = tier_layers_of(&s)?;
        let (got, windows) = drafted(&mut s, ids)?;
        let rejected = got.windows.iter().filter(|&&(p, k)| p && k == 1).count();
        let accepted = got.windows.iter().filter(|&&(p, k)| p && k == PAIR).count();
        let differ = got
            .tokens
            .iter()
            .zip(&want.tokens)
            .position(|(a, b)| a != b);
        let bits = got.tokens == want.tokens && got.windows == want.windows;
        let accepts = tier::premise("nextn-bits: a draft is kept", Tag::FileBound, accepted > 0)?;
        let ok = bits && accepts && rejected > 0;
        tier::sc("nextn-bits: the drafted history's ids and windows are the reference's")?;
        println!(
            "nextn-bits: {} ids in {} windows, the reference's {} ids in {} windows, the same \
             {bits} (first id apart {differ:?}); {accepted} drafts kept, {rejected} rejected: {}",
            got.tokens.len(),
            got.windows.len(),
            want.tokens.len(),
            want.windows.len(),
            verdict(ok)
        );
        pass &= ok;
        pass &= held(
            "nextn-stage",
            nextn_stage(&s, nextn.index, windows.as_ref(), &tier_layers),
        );
        Ok(pass)
    }

    // ------------------------------------------------ the load record's cards

    /// The binary's first record of `kind` of a short run of `ids` under
    /// `args`: `generate_glm5next` beside this one, its stdout read whole by
    /// its kinds, an error naming the run on a non-zero exit.
    fn child_record(
        kind: &'static record::Kind,
        args: &[&str],
        ids: &[u32],
    ) -> Result<Fields, GateError> {
        let exe = std::env::current_exe()?.with_file_name("generate_glm5next");
        let tokens: Vec<String> = ids.iter().map(u32::to_string).collect();
        let out = std::process::Command::new(&exe)
            .args(args)
            .arg("--tokens")
            .arg(tokens.join(","))
            .arg("-n")
            .arg("2")
            .stdin(std::process::Stdio::null())
            .output()
            .map_err(|e| format!("spawn {}: {e}", exe.display()))?;
        if !out.status.success() {
            return Err(format!(
                "generate_glm5next {}: {}; stderr: {}",
                args.join(" "),
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            )
            .into());
        }
        let stdout = String::from_utf8(out.stdout)?;
        let log = record::Log::of(&stdout, record::GENERATE_GLM5NEXT)
            .named(format!("generate_glm5next {}", args.join(" ")));
        Ok(log.one(kind)?)
    }

    /// The `load` record `generate_glm5next --place <place>` prints: a short
    /// child run of `ids` ([`Self::child_record`]).
    fn load_record(place: &str, ids: &[u32]) -> Result<Fields, GateError> {
        child_record(&record::LOAD_GENERATOR, &["--place", place], ids)
    }

    /// The `place unset` record a child run with no `--place` prints: the
    /// common rule's choice on this process's cards ([`Self::child_record`]).
    fn unset_record(ids: &[u32]) -> Result<Fields, GateError> {
        child_record(&record::PLACE_UNSET, &[], ids)
    }

    /// Whether `load`'s `cards` are exactly `want`, in order: each the
    /// driver's name of the device the placement resolved, each space
    /// written `_` (`generate::with_cards`).
    fn cards_named(load: &Fields, want: &[String]) -> Result<bool, ReadError> {
        Ok(load.csv("cards")? == want.iter().map(String::as_str).collect::<Vec<_>>())
    }

    /// The record words of `place`'s cards on this process's devices: the
    /// driver's name of each device it resolves to, spaces written `_`.
    fn card_words(place: Place) -> Result<Vec<String>, GateError> {
        let census = bloomery_gpu::census()?;
        place
            .on(&census)?
            .card_specs()?
            .iter()
            .map(|s| {
                census
                    .iter()
                    .find(|d| s.device.is_some_and(|dev| dev.uuid == d.uuid))
                    .map(|d| d.name.replace(' ', "_"))
                    .ok_or_else(|| format!("{}: no census device is the plan's", s.name).into())
            })
            .collect()
    }

    /// `records` (module header): the cards each placement's load record
    /// names, read as records.py reads a csv field, against the driver names
    /// of the devices the placement resolves to, in order; and the same
    /// check on the line with its first card's name grown by one letter
    /// inside the brackets, which must be red.
    fn records_arm() -> Result<bool, GateError> {
        let prompt = prompt()?;
        let ids = &prompt[..RECORD_TOKENS];
        let mut pass = true;
        for place in [Place::A, Place::Bp] {
            let want = card_words(place)?;
            let line = held(
                "records",
                load_record(place.name(), ids).and_then(|l| {
                    let named = cards_named(&l, &want);
                    let mutant = l.line().replacen(
                        &format!("cards=[{}", want[0]),
                        &format!("cards=[{}X", want[0]),
                        1,
                    );
                    let mutant = record::Log::of(&mutant, record::GENERATE_GLM5NEXT)
                        .one(&record::LOAD_GENERATOR)?;
                    let mutant_red =
                        mutant.line() != l.line() && cards_named(&mutant, &want) == Ok(false);
                    let tier = l.opt_u64("tier_experts")?;
                    let tier_bytes = l.opt_u64("tier_bytes")?;
                    let tier_ok = match place.tier_cards().is_empty() {
                        true => tier.is_none() && tier_bytes.is_none(),
                        false => tier.is_some_and(|n| n > 0),
                    };
                    let ok = named == Ok(true) && mutant_red && tier_ok;
                    println!(
                        "records --place {}: cards {:?} (want {want:?}), a mutant first card red \
                     {mutant_red}, tier_experts {tier:?} tier_bytes {tier_bytes:?}: {}",
                        place.name(),
                        l.csv("cards"),
                        verdict(ok)
                    );
                    if !ok {
                        println!("  {}", l.line());
                    }
                    Ok(ok)
                }),
            );
            pass &= line;
        }
        // A run with no `--place` takes the common rule: on this gate's two
        // cards the offer is `bp`, kept at or past the family's break-even —
        // the `place unset` record names the word, why, the tier's experts
        // in the plan at the child's default context, and the rule's
        // break-even and basis. FAIL-first: a CLI that keeps a placement of
        // its own prints no `place unset` line, and the record read is red.
        // The rule's choice reads the plan's tier experts at the real file's size against the
        // family's break-even: a clause of the real file.
        let unset = !tier::run_clause(
            "records-unset: the unset rule keeps bp at the real file's tier size",
            Tag::FileBound,
        )? || reported(
            "records-unset",
            unset_record(ids).and_then(|p| {
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
            }),
        );
        pass &= unset;
        Ok(pass)
    }
}
