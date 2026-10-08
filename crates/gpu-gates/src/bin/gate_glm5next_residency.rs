//! GPU gate for GLM's adaptive expert residency on the real model
//! (`bloomery_gpu_glm5next::swap`, `BLOOMERY_RESIDENCY`): the file planned
//! onto the gate card (`crate::gate_card::plan_gate`, one card — the trunk's
//! release scope; a plan over more than one card is refused at the body's
//! load), the residency machine over the stage card's routed stacks, the
//! lever set here from the plan's own card slots a layer (the environment
//! cannot move it). The seed is the plan's id prefix. Two loads, one after
//! the other: the plain load, then the NextN load (the file's next-token
//! layer beside the target, `app::arch::glm5next::open_nextn`) under
//! `mid-p0-s1`; a prompt is fed in batches
//! ([`bloomery_gpu_glm5next::feed`]), one residency pass.
//!
//! A history is: the session cleared (the residency back to its seed), the
//! first [`PROMPT`] ids of an lcg over the vocabulary as one prompt call,
//! then [`STEPS`] greedy steps, each step's argmax and the FNV-1a 64 of its
//! logits row read. Clauses; each names its mutant:
//!
//! - `refuse` (the host-set refusal at load): a machine whose host holds one
//!   byte less than the churn pool needs — the stage card's experts past the
//!   pinned ones, which the host set must hold too — is refused by name
//!   before anything loads, within [`REFUSE_BOUND_S`] (mutant: the churn
//!   pool check removed from the load).
//! - `front-refuse` (the prompt batch's GEMM front and group units in the
//!   plan, `place::prompt_reserve_bytes`): the gate card one byte short of the
//!   plan's floor — its dense granules, cache, context, scratch, reserves,
//!   margin and the front, found from the plan's own shortfalls — is
//!   refused by name, the front and its bytes named, by the planner and by
//!   `open_resident` (the tape's open, under the gate's lever) before
//!   anything loads, within [`REFUSE_BOUND_S`]; the card at the floor plans,
//!   its card experts printed (mutant: the front out of the plan's reserve and
//!   check — the card one byte short then fails by the card's bound or
//!   plans, the front unnamed).
//! - `passes` (a pass counts its kept rows only): a history's boundaries
//!   end, in order, no pass, the prompt call (one pass, 0 rows kept — the
//!   batch service notes no id) and each step (1 kept) (mutant: the call
//!   keeps its rows as a step's kind).
//! - `transform`: after the history, every expert the machine admitted holds
//!   in its slot, part by part, the bytes a static load uploads for it; and
//!   the source's parts are the header's: each card layer's part sizes are
//!   its three stacks' own (each stack's type and shape as the header states
//!   them), the card layers hold one layout, and no layer whose down the
//!   header gives as Q6_K holds a card part (mutant: every part of a layer
//!   sized at its first stack's — the down's slots then sit at the gate's
//!   stride, which the load's stack-size check, the byte check and `c1` all
//!   pass, and the header's down size does not).
//!   PIN(2026-10-01): the file's card layers share one part layout —
//!   `card_routed` admits Q4_K and Q5_K alone and `eligible` needs every
//!   stack of a layer routable, so the one Q5_K gate/up layer, whose down is
//!   Q6_K, has no card slot — and per-layer part sizes are not exercised by
//!   this file.
//! - `steps` (the steps feed refused under the machine): a prompt fed one
//!   decode step an id is refused by name while the machine runs, and the
//!   batch feed still serves after it (mutant: the steps arm of the feed
//!   runs the steps without the refusal).
//! - `c1` (green-only): the history twice gives the same tokens and logits,
//!   and flips land on at least one layer (mutant: the staging thread stages
//!   an expert's first part alone).
//! - `table`: after the history, the stage card's copy of the map read back
//!   (`generate::Residence`, `CardTable`) is off neither the host map nor
//!   the machine's ledger, names no slot twice in a layer, and is off the
//!   map the history started from by at least one entry and at most two a
//!   landed flip (each moves its admitted expert onto a slot and its victim
//!   off) (mutant: a landing writes the admitted expert's word and not the
//!   victim's).
//! - `static`: the residency computes the placement its card copy holds.
//!   The copy as the history left it is read; after the seat's reset the
//!   history's prompt is fed the serve's way (`generate::ServeFeed`), and
//!   the row of its last id (row 0) is kept, the boundaries before that row
//!   landing no flip (fed again, up to `SETTLE_FEEDS` feeds, until one lands
//!   none; named). After the teardown, one more load — the
//!   residency off, each routed layer's card experts the read copy's
//!   (`Loaded::open_edited`, `generate::place_table`; named: no machine, and
//!   the card holds the copy's sets) — is fed the same ids the same way:
//!   its row 0 equals the residency's bit for bit (mutants: `place_table`
//!   drops each layer's lowest card id).
//! - `c7`: a residency reset after the history brings every layer's live set
//!   back to its seed (`diff` 0, and the ledger), and lets go of no host
//!   byte (`dropped_bytes` 0: the churn pool stays in the host set for the
//!   model's life) (mutant: the reset copies only the first seed expert
//!   back).
//!
//! On the NextN load, whose slot map is the host run and the next-token
//! layer after it (every expert of that layer on the host):
//!
//! - `nextn-refuse`: a host one byte short of the churn pool and the
//!   next-token layer's host experts — the host set holds both beside the
//!   plan's segments — is refused by name before anything loads, while the
//!   planner at that host still plans (mutant: the load's pool check without
//!   the layer's bytes).
//! - `nextn-front-refuse`: the same floor walk the plain `front-refuse`
//!   holds, planned through `plan_nextn` (the serving default's path: the
//!   draft's card bytes and arena beside the front in the reserve, the front
//!   named on the sum) and opened through [`open_nextn`]: a stage card one
//!   byte short of that floor is refused by name, the front and its bytes
//!   named, before anything loads (mutant: the front zeroed in `plan_nextn`
//!   — the card one byte under the floor then plans or fails by the card's
//!   own bound, the front unnamed).
//! - `nextn-open`: the machine starts over the map's layers — the rule's
//!   layer count, the tally's edges — every pinned count 0, the next-token
//!   layer's ledger row empty, no card part of it in the source, and it alone
//!   listed as routed by no pass (the trunk's layers with no card slot stay
//!   routed) (mutant:
//!   the machine's pinned list at the trunk run's length, which the machine
//!   refuses by name at the load).
//! - `pair-map` (bits): from a clear, the step history; from a clear again,
//!   verifies of two rows over its fed tokens, both kept. Before the first
//!   landing of either, each verify's two tokens and logits FNVs are the
//!   steps' at those positions (mutant: the verify's replay skips its
//!   boundary).
//! - `pair-fold`: a drafted history (the MTP window,
//!   `Speculative<MtpDraft<Body>, 2>`) ends at its boundaries no pass, the
//!   prompt call (0 kept), then each window but the last: a verify as a
//!   `pair` pass keeping its commit's accepted rows, a window with no
//!   proposal as a `step` keeping 1, a rejected row among them (mutant: the
//!   commit keeps the verify's rows, not its accepted ones).
//! - `draft-quiet`: after it, the next-token layer is the one layer the rule
//!   holds routed by no pass, no flip in flight names it, and its card set
//!   and ledger row stay empty (mutant: the batch port notes the walk's ids —
//!   the tally refuses a note of that layer by name, and the history ends in
//!   that error).
//! - `c1-mtp`: the drafted history twice, a residency reset between, gives
//!   the same ids, windows and flips, flips landed (mutant: the walk reads
//!   one NextN key past its position, so the second history reads a row the
//!   first left behind).
//!
//! Tiers (`BLOOMERY_TIER`): every clause runs on the fixture but the ones that need the file's routing skew (flips
//! landed and an expert admitted: `Tag::FileBound`), which the fixture tier defers to the real tier by name. The prompt
//! is the prose corpus's first ids in both tiers, witnessed equal to the MTP set's in the real tier.

#[cfg(not(feature = "glm5next"))]
fn main() {
    eprintln!(
        "gate_glm5next_residency: built without the `glm5next` feature; see the lead's recipe."
    );
    std::process::exit(2);
}

#[cfg(feature = "glm5next")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_glm5next_residency", gate::run())
}

// `pub`: this gate prints its own `transform` line over its own seed list
// and runs no rule walk, so a private module would count `transform_verdict`,
// `layer_seeds` and `rule_after` dead in this bin.
#[cfg(feature = "glm5next")]
#[path = "shared/residency_clauses.rs"]
pub mod residency_clauses;

#[cfg(feature = "glm5next")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "glm5next")]
#[path = "shared/glm5next_tier.rs"]
mod glm5next_tier;

#[cfg(feature = "glm5next")]
#[path = "shared/quiet.rs"]
mod quiet;

#[cfg(feature = "glm5next")]
mod gate {
    use crate::residency_clauses::StaticProbe;
    use std::ops::Range;
    use std::time::Instant;

    use app::arch::glm5next::{GlmCfg, open_nextn, open_resident};
    use app::mtp::MtpDraft;
    use app::{Loaded, OpenArgs, Session, SessionError};
    use bloomery_gpu::GpuError;
    use bloomery_gpu::host::PassKind;
    use bloomery_gpu::host::swap::{Residency, SlotState, SwapMachine, SwapSource};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::generate::{Residence, place_table};
    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::tier;
    use bloomery_gpu_gates::{Fnv1a64, GateError, checks_failed, verdict};
    use bloomery_gpu_glm5next::{Body, PrefillMode, feed, set_prefill};
    use bloomery_levers::{CARD_BUDGET, CARD_DONTNEED, HOST_LOCK, HOST_POPULATE, R8};
    use gguf::Split;
    use gguf::quant::GgmlType;
    use model::arch::glm5next::names;
    use model::arch::glm5next::place::{
        NEXTN_ARENA_BYTES, NextnInputs, NextnPlan, PlaceError, PlanInputs, prompt_reserve_bytes,
    };
    use model::placement::churn::ChurnPool;
    use model::placement::{Machine, ModelTensors, Plan, Violation};
    use runtime::layer::hosted;
    use runtime::{Advance as _, Committed, Out, PassSink, Stop, Target, Verify, Want};

    use crate::glm5next_tier;
    use crate::quiet::Quiet;

    const NAME: &str = "gate_glm5next_residency";
    /// Positions the stores hold: a prompt and the steps with room.
    const CTX: usize = 1024;
    /// Prompt positions: one prompt call, one batch.
    const PROMPT: usize = 64;
    /// Greedy steps after the prompt: planning boundaries of the rule.
    const STEPS: usize = 96;
    /// A refusal before the load reads the files' headers and the plan only.
    const REFUSE_BOUND_S: f64 = 120.0;

    /// The lever's word, as the `residency host` record prints it.
    fn word_of(r: Residency) -> String {
        match r {
            Residency::Off => "off".to_string(),
            Residency::Mid { pinned, spares } => format!("mid-p{pinned}-s{spares}"),
        }
    }

    /// The lever the plan's card slots take: one spare a layer (the least —
    /// a flip lands in it), and the pinned count the least of the layers'
    /// slot counts halved, so every layer leaves the machine its bound
    /// (pinned + spares + 1 slots at least) and half the plan's card experts
    /// a layer can churn.
    fn lever_of(plan: &Plan<'_>) -> Result<Residency, GateError> {
        const WHAT: &str = "gate lever";
        let min = plan
            .n_l
            .iter()
            .copied()
            .filter(|&n| n > 0)
            .min()
            .ok_or("the plan puts no routed expert on the card")?;
        let min = usize::try_from(min).map_err(|e| format!("{WHAT}: {e}"))?;
        let pinned = min / 2;
        if min < pinned + 2 {
            return Err(format!(
                "{WHAT}: the plan's least card slots a layer is {min}, fewer than {pinned} \
                 pinned, one spare and one that moves"
            )
            .into());
        }
        Ok(Residency::Mid { pinned, spares: 1 })
    }

    /// `n` ids of an lcg over the vocabulary: every id a row of the
    /// embedding, the routing spread over the experts.
    fn lcg_ids(n: usize, vocab: usize) -> Vec<u32> {
        let mut x = 0x9e37_79b9_7f4a_7c15_u64;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((x >> 33) % vocab as u64) as u32
            })
            .collect()
    }

    /// The file planned onto `machine` and loaded under `residency`, its
    /// session's prompts fed in batches.
    fn open(
        path: &str,
        machine: &Machine,
        levers: &bloomery_levers::Levers,
        residency: Residency,
    ) -> Result<Session<Body>, GateError> {
        let file = Split::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let inputs = PlanInputs::read(&file).map_err(|e| format!("{path}: {e}"))?;
        let plan = inputs
            .plan(machine, CTX as u64, &glm5next_tier::plan_levers(levers, 0)?)
            .map_err(|e| format!("{path}: {e}"))?;
        let ctx = u32::try_from(plan.ctx_max).map_err(|e| format!("{path}: {e}"))?;
        let m = Body::open_placed_with(file, &plan, &inputs, 0, levers.host(), residency)?;
        let cfg = GlmCfg {
            place: glm5next_tier::plan_levers(levers, 0)?,
            host: levers.host(),
            prefill: PrefillMode::Batch,
            group: 1,
        };
        let loaded = Loaded::<Body>::from_model(m, cfg, ctx);
        loaded.ready(&mut Quiet).map_err(Into::into)
    }

    /// The residency machine the loaded body runs, refused by name without
    /// one.
    fn machine_of(s: &Session<Body>) -> Result<&SwapMachine, GateError> {
        Ok(s.model()
            .body(NAME)?
            .hybrid()
            .swap()
            .ok_or("the load runs no residency machine")?)
    }

    /// The layers the stage card holds routed experts of, with their seeds.
    fn seeds(s: &Session<Body>) -> Result<Vec<(usize, Vec<u32>)>, GateError> {
        let b = s.model().body(NAME)?;
        let machine = machine_of(s)?;
        crate::residency_clauses::seeds(machine, b.hybrid().slots().layers())
    }

    /// What a history saw: the prompt's argmax, every step's argmax and
    /// logits FNV, each boundary's ended pass (its kind and kept rows) and
    /// the flips that landed there, and their sum; and the host map's stage
    /// view after its clear, where those flips started from.
    struct History {
        first: u32,
        tokens: Vec<u32>,
        fnvs: Vec<u64>,
        passes: Vec<(PassKind, usize)>,
        landings: Vec<usize>,
        landed: usize,
        start: Vec<u32>,
    }

    /// Two histories are one when they saw the same: the map one starts from
    /// is no part of it, since a reset puts the seed back in whichever slots
    /// are free.
    impl PartialEq for History {
        fn eq(&self, other: &History) -> bool {
            (
                &self.first,
                &self.tokens,
                &self.fnvs,
                &self.passes,
                &self.landings,
                self.landed,
            ) == (
                &other.first,
                &other.tokens,
                &other.fnvs,
                &other.passes,
                &other.landings,
                other.landed,
            )
        }
    }

    /// The history from a clear ([`Session::clear`], the residency back to
    /// its seed): the prompt call, then the greedy steps.
    fn history(s: &mut Session<Body>, ids: &[u32]) -> Result<History, GateError> {
        let t = Instant::now();
        s.clear()?;
        let clear_s = t.elapsed().as_secs_f64();
        let (_, _, b) = s.model_mut().body_parts(NAME)?;
        b.take_residency_passes();
        let start = s.model().body(NAME)?.hybrid().slots().stage_view();
        let mut next = s.prompt(ids, Want::Argmax)?.argmax();
        let mut h = History {
            first: next,
            tokens: Vec::with_capacity(STEPS),
            fnvs: Vec::with_capacity(STEPS),
            passes: Vec::new(),
            landings: Vec::new(),
            landed: 0,
            start,
        };
        let t = Instant::now();
        for _ in 0..STEPS {
            let out = s.step(next, Want::Logits)?;
            if let Out::Logits { row, .. } = out {
                h.fnvs.push(Fnv1a64::default().f32s(row).value());
            }
            next = out.argmax();
            h.tokens.push(next);
        }
        let steps_s = t.elapsed().as_secs_f64();
        let (_, _, b) = s.model_mut().body_parts(NAME)?;
        let passes = b.take_residency_passes();
        h.landings = passes.iter().map(|(_, r)| r.landed).collect();
        h.landed = h.landings.iter().sum();
        h.passes = passes.iter().map(|&(k, r)| (k, r.kept)).collect();
        println!(
            "history: clear {clear_s:.1} s, prompt + {STEPS} steps in {steps_s:.1} s (runtime \
             values), {} boundaries, {} flips landed",
            h.passes.len(),
            h.landed
        );
        Ok(h)
    }

    /// `refuse`: the host one byte short of the churn pool, on `machine`
    /// with its host shrunk to it.
    fn refuse_clause(
        path: &str,
        machine: &Machine,
        levers: &bloomery_levers::Levers,
        plan: &Plan<'_>,
        residency: Residency,
    ) -> Result<bool, GateError> {
        let Residency::Mid { pinned, .. } = residency else {
            return Err("the gate's lever is off".into());
        };
        let pool = ChurnPool::of(plan, 0, pinned).map_err(|e| format!("{path}: {e}"))?;
        pool.check(plan).map_err(|e| format!("{path}: {e}"))?;
        record::residency_host(&word_of(residency), &pool, plan).print();
        let (short, small) =
            crate::residency_clauses::refuse_head(machine, plan.host.headroom_bytes, pool.bytes)?;
        crate::residency_clauses::refuse_tail(short, pool.bytes, REFUSE_BOUND_S, || {
            open(path, &small, levers, residency)
        })
    }

    /// `front-refuse` (module header): the gate card's floor with the prompt
    /// batch's GEMM front and group units found from the plan's own
    /// shortfalls, from a guess
    /// at or under it (the plan's card terms but its experts and rounding,
    /// the margin and the front): each refusal adds the bytes it names, so
    /// the walk lands on the floor.
    fn front_refuse_clause(
        path: &str,
        machine: &Machine,
        levers: &bloomery_levers::Levers,
        inputs: &PlanInputs,
        plan: &Plan<'_>,
        residency: Residency,
    ) -> Result<bool, GateError> {
        let place = glm5next_tier::plan_levers(levers, 0)?;
        let front = prompt_reserve_bytes(&inputs.hp, CTX as u64, false);
        let (Some(card), Some(t)) = (machine.cards.first(), plan.cards.first()) else {
            return Err("the gate machine has no stage card".into());
        };
        let with = |usable: u64| {
            let mut m = machine.clone();
            m.cards[0].usable_bytes = usable;
            m
        };
        let mut usable = t.dense_bytes
            + t.kv_bytes
            + t.scratch_bytes
            + t.context_bytes
            + t.reserve_bytes
            + card.margin_bytes
            + front;
        let mut walked = 0usize;
        let floor = loop {
            if walked == 8 {
                return Err(format!("the floor walk did not settle by {usable} B").into());
            }
            walked += 1;
            match inputs.plan(&with(usable), CTX as u64, &place) {
                Ok(_) => break usable,
                Err(PlaceError::FrontOver { short, .. }) => usable += short,
                Err(PlaceError::Broken(v)) => {
                    let over = v.iter().find_map(|v| match v {
                        Violation::CardOver { total, limit, .. } => Some(total - limit),
                        _ => None,
                    });
                    usable += over.ok_or_else(|| format!("{path}: {}", PlaceError::Broken(v)))?;
                }
                Err(e) => return Err(format!("{path}: {e}").into()),
            }
        };
        let at_floor = inputs
            .plan(&with(floor), CTX as u64, &place)
            .map(|p| p.cards[0].experts);
        let short_card = with(floor - 1);
        let below = inputs.plan(&short_card, CTX as u64, &place);
        let named = matches!(
            below,
            Err(PlaceError::FrontOver { front: f, short: 1, .. }) if f == front
        );
        let t0 = Instant::now();
        let file = Split::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let small = short_card.clone();
        let args = OpenArgs {
            place: "gate",
            machine: move |_| small.clone(),
            ctx: CTX,
            mode: StepMode::Graph,
            cfg: GlmCfg {
                place: place.clone(),
                host: levers.host(),
                prefill: PrefillMode::Batch,
                group: 1,
            },
        };
        let opened = open_resident(file, args, residency, &mut Quiet);
        let secs = t0.elapsed().as_secs_f64();
        let (refused, why) = match opened {
            Err(e) => {
                let text = e.to_string();
                (
                    text.contains("GEMM front")
                        && text.contains(&format!("{front} B"))
                        && secs < REFUSE_BOUND_S,
                    text,
                )
            }
            Ok(_) => (false, "the open loaded".to_string()),
        };
        let ok = front > 0 && at_floor.is_ok() && named && refused;
        println!(
            "front-refuse: the GEMM front and group units {front} B in the plan of the {} card: \
             its floor {floor} B ({walked} plans) plans with {} card experts; one byte under it \
             the planner {}; \
             open_resident under {} there in {secs:.1} s — {why}: {}",
            card.name,
            at_floor.map_or_else(|e| format!("no ({e})"), |n| n.to_string()),
            match &below {
                Err(e @ PlaceError::FrontOver { .. }) => format!("refuses: {e}"),
                Err(e) => format!("refuses by another name: {e}"),
                Ok(_) => "plans".to_string(),
            },
            word_of(residency),
            verdict(ok)
        );
        Ok(ok)
    }

    /// Layer `l`'s parts as the header states them: per stack (gate, up,
    /// down), its type and one expert's bytes from that type and the stack's
    /// first two dims; `None` on a layer without the three stacks.
    fn header_parts(
        model: &ModelTensors,
        l: usize,
    ) -> Result<Option<Vec<(GgmlType, usize)>>, GateError> {
        let names = [
            names::ffn_gate_exps(l),
            names::ffn_up_exps(l),
            names::ffn_down_exps(l),
        ];
        let mut out = Vec::with_capacity(names.len());
        for n in &names {
            let Some(t) = model.tensors.iter().find(|t| &t.name == n) else {
                return Ok(None);
            };
            let (Some(blck), Some(size)) = (t.ty.blck_size(), t.ty.type_size()) else {
                return Err(format!("{n}: {:?}, a type the header cannot size", t.ty).into());
            };
            let [row, rows, ..] = t.dims[..] else {
                return Err(format!("{n}: dims {:?}, not a stack of experts", t.dims).into());
            };
            if !row.is_multiple_of(blck) {
                return Err(
                    format!("{n}: rows of {row} values, not whole blocks of {blck}").into(),
                );
            }
            out.push((t.ty, usize::try_from(size * (row / blck) * rows)?));
        }
        Ok(Some(out))
    }

    /// `transform`: each admitted expert's slot against its static bytes,
    /// part by part; then every layer's parts in the source against the
    /// header's.
    fn transform_clause(
        s: &Session<Body>,
        model: &ModelTensors,
        seeds: &[(usize, Vec<u32>)],
    ) -> Result<bool, GateError> {
        let m = s.model();
        let b = m.body(NAME)?;
        let machine = machine_of(s)?;
        let source = b
            .residency_source()
            .ok_or("the load has no residency source")?;
        let (checked, bad) =
            crate::residency_clauses::transform_check(m.gpu(), machine, source, seeds, |l| {
                source.part_bytes(l).len()
            })?;
        // The source's parts against the header, over every layer: the byte
        // check above reads the source's own part sizes on both sides.
        let mut cards = Vec::new();
        let mut layouts: Vec<Vec<(GgmlType, usize)>> = Vec::new();
        let mut off_header = Vec::new();
        let mut q6k_down = Vec::new();
        let mut on_q6k = Vec::new();
        for l in 0..model.layers {
            let parts = source.part_bytes(l);
            let header = header_parts(model, l)?;
            if header
                .as_ref()
                .is_some_and(|h| h.last().is_some_and(|&(ty, _)| ty == GgmlType::Q6_K))
            {
                q6k_down.push(l);
                if !parts.is_empty() {
                    on_q6k.push(l);
                }
                continue;
            }
            if parts.is_empty() {
                continue;
            }
            cards.push(l);
            match header {
                Some(h) if h.iter().map(|&(_, n)| n).eq(parts.iter().copied()) => {
                    if !layouts.contains(&h) {
                        layouts.push(h);
                    }
                }
                _ => off_header.push(l),
            }
        }
        let layout = layouts
            .iter()
            .map(|h| {
                let parts: Vec<String> = h.iter().map(|(ty, n)| format!("{ty:?} {n} B")).collect();
                format!("[{}]", parts.join(", "))
            })
            .collect::<Vec<_>>()
            .join(" and ");
        let ok = (checked > 0 || !glm5next_tier::skew()?)
            && bad.is_empty()
            && !cards.is_empty()
            && off_header.is_empty()
            && layouts.len() == 1
            && !q6k_down.is_empty()
            && on_q6k.is_empty();
        println!(
            "transform: {checked} admitted experts on {} card layers, {} parts differing from a \
             static load{}; card parts {layout} (layouts: {}), the header's on all but {off_header:?}; \
             Q6_K-down layers {q6k_down:?}, those with card parts {on_q6k:?}: {}",
            cards.len(),
            bad.len(),
            bad.first()
                .map(|f| format!(" (first {f})"))
                .unwrap_or_default(),
            layouts.len(),
            verdict(ok)
        );
        Ok(ok)
    }

    /// `steps`: the steps feed refused by name under the machine, the batch
    /// feed still serving after it. The served call's positions stand; the
    /// next clause's history clears them.
    fn steps_clause(s: &mut Session<Body>, ids: &[u32]) -> Result<bool, GateError> {
        let refused = {
            let m = s.model_mut();
            set_prefill(m, PrefillMode::Steps)?;
            feed(m, &ids[..8])
        };
        let named = matches!(&refused, Err(GpuError::Shape { detail, .. })
            if detail.contains("steps beside BLOOMERY_RESIDENCY"));
        set_prefill(s.model_mut(), PrefillMode::Batch)?;
        let served = s.prompt(&ids[..8], Want::Argmax).is_ok();
        println!(
            "steps: under the machine the steps feed: {}; the batch feed after it: {}: {}",
            match &refused {
                Ok(t) => format!("token {t}"),
                Err(e) => format!("error \"{e}\""),
            },
            verdict(served),
            verdict(named && served)
        );
        Ok(named && served)
    }

    /// The residency views of `s`'s body ([`Residence`]): its stage card's
    /// copy of the slot map (`Body::slot_copy`) and its host tier.
    fn residence(s: &Session<Body>) -> Result<Residence<'_>, GateError> {
        let m = s.model();
        let b = m.body(NAME)?;
        Ok(Residence::of(m.gpu(), b.slot_copy(), b.hybrid()))
    }

    /// `table` (module header): the stage card's copy of the map after the
    /// history `held` ends, against the host map, the machine's ledger and
    /// the map the history started from.
    fn table_clause(s: &Session<Body>, held: &History) -> Result<bool, GateError> {
        crate::residency_clauses::table_clause_with(
            &residence(s)?,
            &held.start,
            held.landed,
            "the history",
            glm5next_tier::skew()?,
        )
    }

    /// [`StaticProbe`] on `s`, after the last history.
    fn static_probe(s: &mut Session<Body>, ids: &[u32]) -> Result<StaticProbe, GateError> {
        crate::residency_clauses::static_probe(s, ids, residence, |s| {
            let (_, _, b) = s.model_mut().body_parts(NAME)?;
            Ok(b.take_residency_passes())
        })
    }

    /// `static`, its other half, after the residency load's teardown: a load
    /// of the same file on the same machine with the residency off, each
    /// routed layer's card experts `p`'s copy's (`generate::place_table` on
    /// the plan), fed `p`'s ids the serve's way; its row 0 against `p`'s,
    /// bit for bit.
    fn static_clause(
        path: &str,
        machine: &Machine,
        levers: &bloomery_levers::Levers,
        p: &StaticProbe,
    ) -> Result<bool, GateError> {
        let t0 = Instant::now();
        let file = Split::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let args = OpenArgs {
            place: "gate",
            machine: move |_| machine.clone(),
            ctx: CTX,
            mode: StepMode::Graph,
            cfg: GlmCfg {
                place: glm5next_tier::plan_levers(levers, 0)?,
                host: levers.host(),
                prefill: PrefillMode::Batch,
                group: 1,
            },
        };
        let loaded = Loaded::<Body>::open_edited(file, args, &mut Quiet, |_, plan| {
            place_table(plan, &p.table).map_err(|e| SessionError::Refused(e.to_string()))
        })?
        .ok_or("the static open stopped at its plan")?;
        let mut s = loaded.ready(&mut Quiet)?;
        crate::residency_clauses::static_row0(&mut s, residence, p, t0.elapsed().as_secs_f64())
    }

    // ------------------------------------ the NextN load under the machine

    /// The NextN clauses' lever: no pinned expert and one spare, the default
    /// the router-set replay picked.
    const NEXTN_LEVER: Residency = Residency::Mid {
        pinned: 0,
        spares: 1,
    };

    /// The rows a verify runs: the target's next token and one proposal.
    const PAIR: usize = 2;

    /// A clause's verdict, an error it met printed as its red line.
    fn held(clause: &str, r: Result<bool, GateError>) -> bool {
        tier::sc(clause).and(r).unwrap_or_else(|e| {
            println!("{clause}: error \"{e}\": {}", verdict(false));
            false
        })
    }

    /// The file planned with its next-token layer onto `machine`
    /// ([`open_nextn`]) and loaded under `residency` in graph mode, its
    /// session's prompts fed in batches.
    fn open_nextn_on(
        path: &str,
        machine: Machine,
        levers: &bloomery_levers::Levers,
        residency: Residency,
    ) -> Result<Session<Body>, GateError> {
        let file = Split::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let args = OpenArgs {
            place: "gate",
            machine: move |_| machine.clone(),
            ctx: CTX,
            mode: StepMode::Graph,
            cfg: GlmCfg {
                place: glm5next_tier::plan_levers(levers, 0)?,
                host: levers.host(),
                prefill: PrefillMode::Batch,
                group: 1,
            },
        };
        open_nextn(file, args, residency, &mut Quiet)?
            .ok_or_else(|| "the NextN open stopped at its plan".into())
    }

    /// `nextn-refuse`: a host one byte short of the churn pool and the
    /// next-token layer's host experts, which the load's host set holds
    /// beside the plan's own: the planner at that host still plans (its own
    /// bound is the layer's experts), and the load refuses by name before
    /// anything loads.
    fn nextn_refuse_clause(
        path: &str,
        machine: &Machine,
        levers: &bloomery_levers::Levers,
        inputs: &PlanInputs,
        nextn: &NextnInputs,
        np: &NextnPlan<'_>,
    ) -> Result<bool, GateError> {
        let pool = ChurnPool::of(&np.plan, 0, 0).map_err(|e| format!("{path}: {e}"))?;
        let (_, hosted) = np.host_runs().map_err(|e| format!("{path}: {e}"))?;
        let need = i128::from(pool.bytes) + i128::from(hosted);
        let short = i128::from(machine.host.usable_bytes) - np.plan.host.headroom_bytes + need - 1;
        let mut small = machine.clone();
        small.host.usable_bytes = u64::try_from(short)?;
        let place = glm5next_tier::plan_levers(levers, 0)?;
        let planner = match inputs.plan_nextn(&small, CTX as u64, &place, nextn) {
            Ok(p) => {
                let same_pool = ChurnPool::of(&p.plan, 0, 0).is_ok_and(|q| q.bytes == pool.bytes);
                (p.plan.host.headroom_bytes == need - 1 && same_pool).then_some(())
            }
            Err(_) => None,
        };
        let t0 = Instant::now();
        let opened = open_nextn_on(path, small, levers, NEXTN_LEVER);
        let secs = t0.elapsed().as_secs_f64();
        let beside = format!("{hosted} B of it held");
        let (named, why) = match opened {
            Err(e) => {
                let text = e.to_string();
                (
                    text.contains("churn pool") && text.contains(&beside) && secs < REFUSE_BOUND_S,
                    text,
                )
            }
            Ok(_) => (false, "the open loaded".to_string()),
        };
        let ok = planner.is_some() && named;
        println!(
            "nextn-refuse: a host of {short} B, one byte short of the {} B churn pool and the \
             layer's {hosted} B of host experts; the planner there {}; the load in {secs:.1} s — \
             {why}: {}",
            pool.bytes,
            if planner.is_some() {
                "plans, its headroom the two less one byte"
            } else {
                "refused or planned another headroom"
            },
            verdict(ok)
        );
        Ok(ok)
    }

    /// `nextn-front-refuse` (module header): the gate card's floor under
    /// `plan_nextn` — the front found from the nextn plan's own refusals,
    /// from a guess at or under it (the target plan's terms but its experts
    /// and rounding, the margin, the draft's card bytes and the arena, and
    /// the front) — each refusal adding the bytes it names, so the walk
    /// lands on the floor. The clause holds the shape `front-refuse` holds
    /// on the plain plan, on the serving default's own planner: the card at
    /// the floor plans, one byte under it the planner refuses by name with
    /// the front and its bytes, and `open_nextn` under [`NEXTN_LEVER`]
    /// refuses the same way before anything loads.
    fn nextn_front_refuse_clause(
        path: &str,
        machine: &Machine,
        levers: &bloomery_levers::Levers,
        inputs: &PlanInputs,
        nextn: &NextnInputs,
    ) -> Result<bool, GateError> {
        let place = glm5next_tier::plan_levers(levers, 0)?;
        let front = prompt_reserve_bytes(&inputs.hp, CTX as u64, false);
        let Some(card) = machine.cards.first() else {
            return Err("the gate machine has no stage card".into());
        };
        if !machine.tiers.is_empty() {
            return Err(
                "the gate machine holds tier cards; the front walked here is the \
                        no-tiers one"
                    .into(),
            );
        }
        let np = inputs
            .plan_nextn(machine, CTX as u64, &place, nextn)
            .map_err(|e| format!("{path}: {e}"))?;
        let t = np.plan.cards.first().ok_or("the plan names no card")?;
        let draft = np.nextn_card_bytes();
        let with = |usable: u64| {
            let mut m = machine.clone();
            m.cards[0].usable_bytes = usable;
            m
        };
        let mut usable = t.dense_bytes
            + t.kv_bytes
            + t.scratch_bytes
            + t.context_bytes
            + t.reserve_bytes
            + card.margin_bytes
            + draft
            + NEXTN_ARENA_BYTES
            + front;
        let mut walked = 0usize;
        let floor = loop {
            if walked == 8 {
                return Err(format!("the floor walk did not settle by {usable} B").into());
            }
            walked += 1;
            match inputs.plan_nextn(&with(usable), CTX as u64, &place, nextn) {
                Ok(_) => break usable,
                Err(PlaceError::FrontOver { short, .. }) => usable += short,
                Err(PlaceError::Broken(v)) => {
                    let over = v.iter().find_map(|v| match v {
                        Violation::CardOver { total, limit, .. } => Some(total - limit),
                        _ => None,
                    });
                    usable += over.ok_or_else(|| format!("{path}: {}", PlaceError::Broken(v)))?;
                }
                Err(e) => return Err(format!("{path}: {e}").into()),
            }
        };
        let floor_m = with(floor);
        let at_floor = inputs
            .plan_nextn(&floor_m, CTX as u64, &place, nextn)
            .is_ok();
        let short_card = with(floor - 1);
        let below = inputs.plan_nextn(&short_card, CTX as u64, &place, nextn);
        let named = matches!(
            below,
            Err(PlaceError::FrontOver { front: f, short: 1, .. }) if f == front
        );
        let t0 = Instant::now();
        let opened = open_nextn_on(path, short_card.clone(), levers, NEXTN_LEVER);
        let secs = t0.elapsed().as_secs_f64();
        let (refused, why) = match opened {
            Err(e) => {
                let text = e.to_string();
                (
                    text.contains("GEMM front")
                        && text.contains(&format!("{front} B"))
                        && secs < REFUSE_BOUND_S,
                    text,
                )
            }
            Ok(_) => (false, "the open loaded".to_string()),
        };
        let ok = front > 0 && at_floor && named && refused;
        println!(
            "nextn-front-refuse: the GEMM front and group units {front} B beside the draft's \
             {draft} B of card bytes and the {} B arena, in the nextn plan of the {} card: its \
             floor {floor} B ({walked} plans) plans; one byte under it the planner {}; \
             open_nextn under {} there in {secs:.1} s — {why}: {}",
            NEXTN_ARENA_BYTES,
            card.name,
            match &below {
                Err(e @ PlaceError::FrontOver { .. }) => format!("refuses: {e}"),
                Err(e) => format!("refuses by another name: {e}"),
                Ok(_) => "plans".to_string(),
            },
            word_of(NEXTN_LEVER),
            verdict(ok)
        );
        Ok(ok)
    }

    /// `nextn-open`: the machine's layers are the slot map's, the trunk's
    /// routed run and the next-token layer `nextn` after it, every layer's
    /// pinned count 0; that layer's ledger row is empty and the source holds
    /// no card part of it.
    fn nextn_open_clause(
        s: &Session<Body>,
        trunk: &Range<usize>,
        nextn: usize,
    ) -> Result<bool, GateError> {
        let b = s.model().body(NAME)?;
        let machine = machine_of(s)?;
        let source = b
            .residency_source()
            .ok_or("the load has no residency source")?;
        let map = b.hybrid().slots().layers();
        let want = trunk.start..nextn + 1;
        let tally = machine.tally();
        let covers = tally.covers(want.start)
            && tally.covers(nextn)
            && !tally.covers(want.end)
            && want.start.checked_sub(1).is_none_or(|l| !tally.covers(l));
        let layers_ok = map == want && machine.rule().layers() == want.len() && covers;
        let mut pinned = Vec::new();
        for l in want.clone() {
            pinned.push(machine.pinned(l)?);
        }
        let row = machine.ledger().row(nextn).map(<[SlotState]>::len);
        let empty = row == Some(0)
            && source.part_bytes(nextn).is_empty()
            && machine.seed(nextn)?.is_empty();
        let p0 = pinned.iter().all(|&p| p == 0);
        let unrouted: Vec<usize> = want
            .clone()
            .zip(machine.rule().unrouted())
            .filter_map(|(l, &u)| u.then_some(l))
            .collect();
        let ok = layers_ok && empty && p0 && unrouted == [nextn];
        println!(
            "nextn-open: the map's layers {map:?} (want {want:?}), the rule's {}, the tally's \
             edges {covers}; layer {nextn}'s ledger row {row:?} slots, its card parts {}, its \
             seed {}; pinned 0 on every layer {p0}; routed by no pass {unrouted:?} (want \
             [{nextn}]): {}",
            machine.rule().layers(),
            source.part_bytes(nextn).len(),
            machine.seed(nextn)?.len(),
            verdict(ok)
        );
        Ok(ok)
    }

    /// The first boundary whose flips landed, as the pass it opens: boundary
    /// 0 opens the prompt call and boundary `j` the `j`-th pass after it.
    /// `passes` when none landed.
    fn seed_passes(landings: &[usize], passes: usize) -> usize {
        landings
            .iter()
            .position(|&n| n > 0)
            .map_or(passes, |j| j.saturating_sub(1))
            .min(passes)
    }

    /// `pair-map` (bits): from a clear, the step history; from a clear
    /// again, the same prompt, then verifies of two rows over the history's
    /// fed tokens, both rows kept. Before the first landing of either, each
    /// verify's two tokens and logits FNVs are the steps' at the same
    /// positions: both rows read one map, the seed.
    fn pair_map_clause(s: &mut Session<Body>, ids: &[u32]) -> Result<bool, GateError> {
        let steps = history(s, ids)?;
        let seed_steps = seed_passes(&steps.landings, STEPS);
        let fed: Vec<u32> = std::iter::once(steps.first)
            .chain(steps.tokens.iter().copied())
            .collect();
        s.clear()?;
        s.model_mut().body_parts(NAME)?.2.take_residency_passes();
        let first = s.prompt(ids, Want::Argmax)?.argmax();
        s.model_mut().capture_rows::<PAIR>()?;
        let runs = seed_steps / PAIR;
        let mut got = Vec::with_capacity(runs);
        for k in 0..runs {
            let tokens = s.verify([fed[PAIR * k], fed[PAIR * k + 1]])?;
            let logits = s.model().rows_logits::<PAIR>()?;
            s.commit(PAIR)?;
            got.push((
                tokens,
                [
                    Fnv1a64::default().f32s(&logits[0]).value(),
                    Fnv1a64::default().f32s(&logits[1]).value(),
                ],
            ));
        }
        let passes = s.model_mut().body_parts(NAME)?.2.take_residency_passes();
        let landings: Vec<usize> = passes.iter().map(|(_, r)| r.landed).collect();
        let compared = seed_passes(&landings, runs);
        let differ = (0..compared).find(|&k| {
            let (tokens, fnvs) = got[k];
            let at = PAIR * k;
            tokens[..] != steps.tokens[at..at + PAIR] || fnvs[..] != steps.fnvs[at..at + PAIR]
        });
        let ok = first == steps.first && compared > 0 && differ.is_none();
        println!(
            "pair-map: the steps on the seed for {seed_steps} positions, the verifies for {} of \
             {runs}; {compared} verifies compared, tokens and logits FNV equal the steps' {} \
             (first differing verify {differ:?}), the prompt's token {first} (steps {}): {}",
            seed_passes(&landings, runs),
            differ.is_none(),
            steps.first,
            verdict(ok)
        );
        Ok(ok)
    }

    /// What a drafted history saw: the generated ids, each window's
    /// proposal and kept rows, and each boundary's ended pass with the flips
    /// that landed there.
    #[derive(PartialEq)]
    struct Drafted {
        tokens: Vec<u32>,
        windows: Vec<(bool, usize)>,
        passes: Vec<(PassKind, usize)>,
        landings: Vec<usize>,
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

    /// The stacked history from a clear (the residency back to its seed):
    /// the prompt call with the draft's store walks, then MTP windows
    /// (`Speculative<MtpDraft<Body>, 2>`) for [`STEPS`] ids.
    fn drafted_history(s: &mut Session<Body>, ids: &[u32]) -> Result<Drafted, GateError> {
        let t = Instant::now();
        s.clear()?;
        s.model_mut().body_parts(NAME)?.2.take_residency_passes();
        let draft = MtpDraft::open(s.model(), PrefillMode::Batch, StepMode::Eager)?;
        let mut spec = s.with_draft::<MtpDraft<Body>, PAIR>(draft, &mut Quiet)?;
        let first = spec.prompt(s, ids)?;
        let mut w = Windows::default();
        let stop = Stop::new(STEPS, s.ctx())?;
        let out = runtime::generate(s, &mut spec, ids, first, &stop, &mut w)?;
        let passes = s.model_mut().body_parts(NAME)?.2.take_residency_passes();
        let d = Drafted {
            tokens: out.tokens,
            windows: w.0,
            passes: passes.iter().map(|&(k, r)| (k, r.kept)).collect(),
            landings: passes.iter().map(|(_, r)| r.landed).collect(),
        };
        println!(
            "drafted history: prompt + {} ids in {} windows in {:.1} s (runtime value), {} \
             boundaries, {} flips landed",
            d.tokens.len(),
            d.windows.len(),
            t.elapsed().as_secs_f64(),
            d.passes.len(),
            d.landings.iter().sum::<usize>()
        );
        Ok(d)
    }

    /// `pair-fold`: the boundaries end no pass, the prompt call (0 kept),
    /// then each window but the last, whose pass the next boundary would
    /// end: a verify as a `pair` pass keeping its commit's accepted rows, a
    /// window with no proposal as a `step` keeping 1; a rejected row among
    /// them.
    fn pair_fold_clause(d: &Drafted) -> bool {
        let mut want = vec![(PassKind::None, 0), (PassKind::Prompt, 0)];
        let ended = d.windows.len().saturating_sub(1);
        want.extend(
            d.windows[..ended]
                .iter()
                .map(|&(proposed, kept)| match proposed {
                    true => (PassKind::Pair, kept),
                    false => (PassKind::Step, 1),
                }),
        );
        let rejected = d.windows.iter().filter(|&&(p, k)| p && k == 1).count();
        let accepted = d.windows.iter().filter(|&&(p, k)| p && k == PAIR).count();
        let differ = d.passes.iter().zip(&want).position(|(a, b)| a != b);
        let ok = d.passes == want && rejected > 0;
        println!(
            "pair-fold: {} boundaries (want {}), each verify a pair pass keeping its accepted \
             rows {} (first differing boundary {differ:?}); {accepted} accepted, {rejected} with \
             a rejected row: {}",
            d.passes.len(),
            want.len(),
            d.passes == want,
            verdict(ok)
        );
        ok
    }

    /// `draft-quiet`: after a drafted history, the rule holds the next-token
    /// layer `nextn` routed by no pass, no flip in flight names it, and its
    /// card set and ledger row stay empty.
    fn draft_quiet_clause(
        s: &Session<Body>,
        trunk: &Range<usize>,
        nextn: usize,
    ) -> Result<bool, GateError> {
        let machine = machine_of(s)?;
        let at = nextn - trunk.start;
        let flips = machine
            .rule()
            .in_flight()
            .iter()
            .filter(|f| f.layer == at)
            .count();
        let live = machine
            .rule()
            .live(at)
            .map_err(|e| format!("the rule's layer {at}: {e:?}"))?
            .count();
        let row = machine.ledger().row(nextn).map(<[SlotState]>::len);
        let listed = machine.rule().unrouted().get(at) == Some(&true);
        let ok = listed && flips == 0 && live == 0 && row == Some(0);
        println!(
            "draft-quiet: layer {nextn} (the rule's {at}) routed by no pass {listed}: {flips} \
             flips in flight, {live} live on the card, its ledger row {row:?} slots: {}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// `c1-mtp`: the stacked history twice, a residency reset between, gives
    /// the same ids, windows and flips, flips landed.
    fn c1_mtp_clause(a: &Drafted, b: &Drafted, skew: bool) -> bool {
        let landed: usize = a.landings.iter().sum();
        let ok = a == b && (landed > 0 || !skew);
        println!(
            "c1-mtp: the stacked history twice, {} ids, {} windows, {landed} flips landed: same \
             {}: {}",
            a.tokens.len(),
            a.windows.len(),
            a == b,
            verdict(ok)
        );
        ok
    }

    /// The prose prompt of ik's MTP draft set (refset `mtp-glm5next`), on
    /// which a draft both keeps and rejects rows: the lcg prompt's greedy
    /// output repeats, so every window keeps both and `pair-fold` has no
    /// rejected row to fold. The prose corpus's first ids in both tiers
    /// ([`glm5next_tier::mtp_prompt`]).
    fn prose_ids() -> Result<Vec<u32>, GateError> {
        let ids = glm5next_tier::mtp_prompt()?;
        if ids.len() < PROMPT {
            return Err(format!(
                "a prompt of {} ids, the NextN clauses read {PROMPT}",
                ids.len()
            )
            .into());
        }
        Ok(ids)
    }

    /// The NextN clauses, on a NextN load of the gate placement under
    /// [`NEXTN_LEVER`]: `nextn-refuse` before it, then `nextn-open`,
    /// `pair-map`, `pair-fold`, `draft-quiet` and `c1-mtp` on it.
    fn nextn_clauses(
        path: &str,
        machine: &Machine,
        levers: &bloomery_levers::Levers,
        inputs: &PlanInputs,
        ids: &[u32],
    ) -> Result<bool, GateError> {
        let nextn = NextnInputs::read(inputs).map_err(|e| format!("{path}: {e}"))?;
        let place = glm5next_tier::plan_levers(levers, 0)?;
        let np = inputs
            .plan_nextn(machine, CTX as u64, &place, &nextn)
            .map_err(|e| format!("{path}: {e}"))?;
        let trunk = hosted(&inputs.spec.layers).map_err(|e| format!("{path}: {e}"))?;
        println!(
            "{NAME}: the NextN load under {}, layer {} after the host run {trunk:?}",
            word_of(NEXTN_LEVER),
            nextn.index
        );
        let mut ok = held(
            "nextn-refuse",
            nextn_refuse_clause(path, machine, levers, inputs, &nextn, &np),
        );
        ok &= held(
            "nextn-front-refuse",
            nextn_front_refuse_clause(path, machine, levers, inputs, &nextn),
        );
        let t0 = Instant::now();
        let mut s = match open_nextn_on(path, machine.clone(), levers, NEXTN_LEVER) {
            Ok(s) => s,
            Err(e) => {
                println!(
                    "nextn-open: the load: error \"{e}\": {}; pair-map, pair-fold, \
                     draft-quiet and c1-mtp not run",
                    verdict(false)
                );
                return Ok(false);
            }
        };
        s.model_mut().body_parts(NAME)?.2.log_residency(0);
        println!(
            "NextN load in {:.1} s (runtime value)",
            t0.elapsed().as_secs_f64()
        );
        ok &= held("nextn-open", nextn_open_clause(&s, &trunk, nextn.index));
        ok &= held("pair-map", pair_map_clause(&mut s, &ids[..PROMPT]));
        match drafted_history(&mut s, &ids[..PROMPT]) {
            Ok(a) => {
                tier::sc("pair-fold: a drafted history's passes")?;
                ok &= pair_fold_clause(&a);
                ok &= held("draft-quiet", draft_quiet_clause(&s, &trunk, nextn.index));
                let b = drafted_history(&mut s, &ids[..PROMPT]);
                ok &= held(
                    "c1-mtp",
                    b.and_then(|b| Ok(c1_mtp_clause(&a, &b, glm5next_tier::skew()?))),
                );
            }
            Err(e) => {
                println!(
                    "pair-fold, draft-quiet: the drafted history: error \"{e}\": {}; draft-quiet \
                     and c1-mtp not run",
                    verdict(false)
                );
                ok = false;
            }
        }
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&tier::acts_on(&[
            CARD_BUDGET,
            HOST_POPULATE,
            HOST_LOCK,
            CARD_DONTNEED,
            R8,
        ])?)?;
        crate::gate_card::init()?;
        glm5next_tier::init_file()?;
        let path = glm5next_tier::model_path()?;
        let file = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        let inputs = PlanInputs::read(&file).map_err(|e| format!("{path}: {e}"))?;
        let machine = crate::gate_card::plan_gate(inputs.model.layers);
        let place = glm5next_tier::plan_levers(&levers, 0)?;
        let plan = inputs
            .plan(&machine, CTX as u64, &place)
            .map_err(|e| format!("{path}: {e}"))?;
        let residency = lever_of(&plan)?;
        println!(
            "{NAME}: the plan's least card slots a layer, halved for the pinned count, one spare: \
             {}",
            word_of(residency)
        );
        tier::sc("refuse: the churn pool's host refusal at load")?;
        let mut pass = refuse_clause(&path, &machine, &levers, &plan, residency)?;
        pass &= held(
            "front-refuse",
            front_refuse_clause(&path, &machine, &levers, &inputs, &plan, residency),
        );

        let t0 = Instant::now();
        let mut s = open(&path, &machine, &levers, residency)?;
        s.model_mut().body_parts(NAME)?.2.log_residency(0);
        println!(
            "load in {:.1} s (runtime value)",
            t0.elapsed().as_secs_f64()
        );
        let seeds = seeds(&s)?;
        let ids = lcg_ids(PROMPT + 8, inputs.hp.n_vocab);

        let first = history(&mut s, &ids[..PROMPT])?;
        tier::sc("passes: a pass counts its kept rows only")?;
        pass &= crate::residency_clauses::passes_clause(&first.passes, STEPS);

        tier::sc("transform: every admitted expert holds the bytes a static load uploads")?;
        pass &= transform_clause(&s, &inputs.model, &seeds)?;
        tier::sc("steps: the steps feed is refused under the machine")?;
        pass &= steps_clause(&mut s, &ids)?;

        let again = history(&mut s, &ids[..PROMPT])?;
        tier::sc("c1: the history twice gives the same tokens and logits")?;
        pass &= crate::residency_clauses::c1_clause_with(
            first.tokens.len(),
            first.landed,
            first == again,
            glm5next_tier::skew()?,
        );

        tier::sc("table: the stage card's copy of the map")?;
        pass &= table_clause(&s, &again)?;
        let probe = static_probe(&mut s, &ids[..PROMPT])?;

        let r = s
            .residency_reset()?
            .ok_or("the load runs no residency machine")?;
        record::residency_reset(&r).print();
        tier::sc("c7: a residency reset brings every live set back to its seed")?;
        pass &= crate::residency_clauses::c7_clause(&r, machine_of(&s)?, &seeds);
        drop(s);

        pass &= held("static", static_clause(&path, &machine, &levers, &probe));
        pass &= nextn_clauses(&path, &machine, &levers, &inputs, &prose_ids()?)?;

        println!("{NAME}: {}", tier::tally_line());
        if pass {
            println!("{NAME}: every clause passed");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
