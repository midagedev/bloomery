//! GPU gate for Qwen3.8's adaptive expert residency on the real model
//! (`bloomery_gpu::arch::qwen3moe::swap38`, `BLOOMERY_RESIDENCY`): the card
//! plan (`place::Experts::Card`, each eligible layer's id prefix on the
//! card), the residency machine over the card's routed stacks at
//! `mid-p<P>-s1`, set here with `P` taken from the plan's card experts a
//! layer (the lever itself is refused, so the environment cannot move it).
//! One card, no MTP draft, two loads one after the other: the gate's own,
//! planned for one sequence, and the slots clauses' (`slots_drafted`,
//! `commit_counts`), planned for the two slots they serve — a pass of
//! several slots on a load planned for one is refused by name.
//!
//! A history is: the session cleared (the residency back to its seed), the
//! first [`PROMPT`] ids of the prose corpus as one prompt call (a ubatch), a
//! greedy step, a verify of three rows committed at two, then [`STEPS`]
//! greedy steps, each call's argmax read and each step's logits row hashed.
//! Clauses; each names its mutant, and the machine's own rules have theirs
//! in `gate_swap`:
//!
//! - `refuse` (the host-set refusal at load): a machine whose host holds one
//!   byte less than the churn pool needs — the card's experts past the pinned
//!   ones, which the host set must hold too — is refused by name before
//!   anything loads, within [`REFUSE_BOUND_S`] (mutant: the open's churn
//!   pool check removed).
//! - `c6` (a pass counts its kept rows only): a verify `[t, t]` after the
//!   prompt with one row kept, then a boundary, leaves the rule equal to one
//!   step of `t` then a boundary; the same verify with both rows kept does
//!   not (mutant: the session's commit keeps every row of the pass).
//! - `c1` (green-only): the history twice gives the same tokens, logits and
//!   passes, and flips land (mutant: the `keep_rows` after the prompt call
//!   removed — the next step's boundary then refuses the prompt pass by
//!   name).
//! - `transform`: after the history, every expert the machine admitted holds
//!   in its slot, part by part, the bytes a static load uploads for it —
//!   the file's Q4_K gate and up, Q5_1 down (mutant: `routed_stacks`
//!   (`model::arch::names`) returns the gate and the up swapped, so the up's
//!   bytes stage into the gate's stack).
//! - `passes` (green-only): a history's boundaries end, in order, no pass,
//!   the prompt call (one pass, 0 rows kept), a step (1 kept), a verify (its
//!   accepted rows) and the steps after it (1 kept each), the last step's
//!   own included.
//! - `order`: every step's own boundary — the next pass's — runs ahead of
//!   the step's readback: when the step returns, the last boundary ends it,
//!   was made ahead (`PassReport::ahead`), and carries the readbacks from
//!   before the step (`PassReport::reads`), while the model has read once
//!   more since (mutants: the boundary made after the head's readback, which
//!   carries the step's own readback; the boundary back at the next launch,
//!   which leaves the step unended when it returns).
//! - `c7`: a residency reset after the history brings every layer's live set
//!   back to its seed (`diff` 0, and the ledger), and lets go of no host
//!   byte (`dropped_bytes` 0: the churn pool stays in the host set for the
//!   model's life) (mutant: the reset's copies of the seed experts back
//!   skipped).
//! - `stream` (host streaming, `Body38::set_xstream`'s `admit`): from a clear, the
//!   first [`STREAM_PROMPT`] ids of the qwen4exp prose corpus as one prompt
//!   call, then [`STREAM_STEPS`] greedy steps, once with the prompt's
//!   streaming off and once on. On, the call's end leaves a live set other
//!   than the seed on some layer, the call closed, and admitted experts; off,
//!   every live set is the seed. The first decode step's host slots are
//!   fewer on than off. The two runs agree as the unstreamed walk and the
//!   streamed one can (the card and the host sum a token's experts in
//!   another split): the prompt's last logits row within
//!   [`GREEDY_MARGIN`](bloomery_gpu_gates::GREEDY_MARGIN) of the off run's
//!   everywhere, and the greedy ids equal or, where they first part, the on
//!   run's top-1 margin below it (a near tie, V4.1's `gate_ds41_callstream`
//!   `s3` rule). A prompt of one id fewer than `Body38::STREAM_FLOOR`,
//!   streaming on, opens no call (no pick record, no end). Mutants, one a
//!   rule: the call's end returns each layer to
//!   the set it started with (`stream_end` with `kept` false: the live sets
//!   and the first step's host slots are the seed's); the pick admits
//!   nothing (a floor past any count: the same, and no expert admitted); the
//!   card route enqueued before the pick (it reads the call's start words
//!   while the union reads the moved map, so an admitted expert's columns
//!   run nowhere and the logits part); the short prompt's guard removed (its
//!   call opens and ends, admitting nothing).
//! - `history`: a prompt call before a decode history does not keep its
//!   placement — the decode has been routing experts of its own, so the
//!   call's end puts every layer back at the live set it started from and
//!   sends the experts it admitted to the host again — while a fresh
//!   server's call keeps (the `stream` clause's gain). The call's last unit
//!   returns each layer in its walk; against the same history with the
//!   return whole at the call's end (`Body38::set_return_at_end`, the twin)
//!   it admits and copies back as many, and the prompt's last logits row and
//!   the first step's after it are the twin's bit for bit (mutants: the end
//!   keeps before a history too; the end always restores; the walk's return
//!   asked before the layer's serve, refused by name).
//! - `split` (the expert stream, `Body38::set_xstream`'s `split`, its rule
//!   at the gate's [`SPLIT_COSTS`]: a host column dear enough that the
//!   balance streams nearly every routed host expert the pick leaves, so the
//!   ring carries hundreds a layer): from a clear, the first
//!   [`SPLIT_PROMPT`] ids of the qwen4exp prose corpus as one prompt call
//!   (one ubatch past the stream's least width), then [`STREAM_STEPS`]
//!   greedy steps, once off and twice split. Count pins: an `xstream` record
//!   per card layer, experts streamed, each record's streamed at most its
//!   tail and the ring's half, and the call's excluded host slots — what
//!   the serve left out, counted by the serve — the records' streamed
//!   columns (none off). The band: a streamed expert's columns run on the
//!   card route's blocks, so the split run agrees with the off run as the
//!   admit run does (`stream`'s rule). Reproducible: the two split runs'
//!   records, prompt logits (bit for bit) and greedy ids are the same.
//!   Mutants, one a rule: the serve's exclusion set dropped (the union
//!   computes the streamed experts again: the count pin and the band); the
//!   ring row's slots reversed against the copies (each streamed column
//!   reads another expert's weights: the band); the floor in the stream's
//!   way back dropped is `xstream`'s unit test's (here every routed host
//!   expert clears the gate's floor).
//! - `history_split`: the `history` call under `split` at [`SPLIT_COSTS`],
//!   [`SPLIT_PROMPT`] ids, its return in the walk against the twin's at the
//!   call's end. The walk's last unit prices an admit of a layer that still
//!   owes return copies at two copies, so a pick may part from the twin's:
//!   where every pick is the twin's the logits are the twin's bit for bit,
//!   where one parts the prompt's last row and the first step's stay within
//!   [`GREEDY_MARGIN`](bloomery_gpu_gates::GREEDY_MARGIN); either way the
//!   call returns in its walk, admits, sends its admits back and leaves every
//!   live set where the history left it.
//! - `units` (the split walk, `Body38::set_split_walk`'s `Every`, against
//!   one unit; every other clause walks one unit): from a clear, the first
//!   [`UNITS_PROMPT`] ids of the qwen4exp prose corpus as one prompt call,
//!   the walk's timing and route taps armed, then [`STREAM_STEPS`] greedy
//!   steps, with the call's streaming off, `admit` and `split`. The call
//!   picks once a card layer, at the layers one unit picks at (the group's
//!   pick, from the first unit's counts at twice); every layer-batch's
//!   slots — the host's its serve listed, those it left out to the stream,
//!   and the card's by the taps against the slot map the call left — are
//!   its columns' every slot, each unit's columns its half; under `split`
//!   each layer's streamed columns are twice the first unit's left-out
//!   slots (one unit's: once), and the second unit leaves slots out too
//!   (the group shares the stream); on every layer the stream ran the
//!   second unit's front and download between the first unit's download
//!   and its shadow. Off, the split walk's logits and greedy ids are one
//!   unit's bit for bit; streamed, they agree as `stream`'s rule allows.
//!   Mutants: a pick per unit (the pick count); no share of the stream
//!   (the second unit's left-out slots); the stream read at the first unit
//!   only (the call's end refuses a layer read by one of its two units, by
//!   name); the walk in today's order (the order).
//! - `history_split_on`: `history_split` with the split walk on for the
//!   walk and its twin, both calls walked as two units, under its rule
//!   (mutant: the return owed at every unit's back — the next unit's shadow
//!   asks for a layer its sibling has not served, refused by name).
//! - `kvfloor` (a growing sequence frees expert bytes; a third load, planned at
//!   a KV floor of [`FLOOR_POSITIONS`] positions, after the slots clauses';
//!   real tier only like the gate, no fixture runs a residency machine): a
//!   [`FLOOR_PROMPT`]-id prompt call and [`STREAM_STEPS`] greedy steps from a
//!   clear, three times, each from the machine's full capacity. With the card
//!   as the load left it (what the machine withdrew is printed, not judged).
//!   With the card squeezed until it refuses a page: the fit runs out whatever
//!   the plan's slack, the same top slots are withdrawn from every layer, the
//!   call completes, the reports' withdrawn bytes equal the page-and-parts
//!   derivation and the ids are the first run's or part at a near tie. After
//!   a restore of every withdrawn slot (no engine path restores one) every
//!   layer's withdrawn count and the rooms' released bytes are 0, and the
//!   third run is the first run's bit for bit. Mutants: the retry in
//!   `Body38::fit_stores` removed (the squeezed call ends in `NoRoom`);
//!   `restore_stacks` skipped in `SwapMachine::restore_tail` (the rooms'
//!   released bytes stay above 0 and the third run's bits part).
//! - `kvspill` (`BLOOMERY_KV_HOST=spill`, a third load planned under it
//!   after the kvfloor clause's; real tier only like the gate): its plan is
//!   the card-backed one's, its selecting stores' K/V planes backing on the
//!   host wherever the card cannot make a page. A prompt call and steps with
//!   the card as the load left it back every page the reach touches on the
//!   card; then the card squeezed until it refuses a page, the same history
//!   replayed from one clear (a clear keeps the reached pages) with steps
//!   past the first run's reach: the growth asks the plane buffers' next
//!   pages, the host takes them, no slot anywhere is withdrawn (the host
//!   took the pages, not the machine; the pooled keys, already backed, ask
//!   the card nothing) and the replayed ids are the first run's bit for
//!   bit. Mutant: the store ignoring the lever, which makes the machine
//!   withdraw for the pages instead.
//! - `slots_drafted`: a drafted pass of two slots' verify rows folds each
//!   slot's accepted rows — slot 0 keeping one of its three rows, slot 1
//!   two of theirs — so the boundary that ends it reports 3 kept rows as
//!   the mask {0, 3, 4}, no prefix (mutant: `commit_slots` passing a prefix
//!   of the kept total).
//! - `commit_counts`: a pass of two slots' three verify rows each, then a
//!   commit with a count a slot cannot keep — slot 0 keeping none, or
//!   slot 1 keeping past its rows while slot 0's count is one it can keep
//!   — refused by name before any slot moves: slot 0 still stands past
//!   its rows and the pass still waits (the session's select and the
//!   model's own refused until its commit); a commit of every row then
//!   lands and a step runs (mutant: each count checked in the keeps' loop,
//!   just before its slot's keep, so slot 0 keeps its row before slot 1's
//!   count is refused).

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen38_residency", gate::run())
}

// `pub`: this gate runs neither `table` nor `static`, and its `passes` list
// holds a verify row the shared clause does not build, so a private module
// would count those items dead in this bin.
#[cfg(feature = "gpu")]
#[path = "shared/residency_clauses.rs"]
pub mod residency_clauses;

#[cfg(feature = "gpu")]
mod gate {
    use std::path::Path;
    use std::time::Instant;

    use app::Session;
    use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for;
    use bloomery_gpu::arch::qwen3moe::{Body38, Qwen38Model, SplitWalk};
    use bloomery_gpu::host::PassKind;
    use bloomery_gpu::host::slots::Slot;
    use bloomery_gpu::host::swap::{
        CallPick, CallReport, PassReport, Residency, SlotState, SwapSource,
    };
    use bloomery_gpu::host::xstream::{Costs, XLayer, XMode, XReport};
    use bloomery_gpu_gates::flip;
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::tier;
    use bloomery_gpu_gates::{
        Fnv1a64, GREEDY_MARGIN, GateError, checks_failed, prose_ids, ref_model_path, verdict,
    };
    use bloomery_levers::{CARD_DONTNEED, HOST_LOCK, HOST_POPULATE};
    use gguf::Split;
    use model::arch::qwen35moe::place::{Experts, PlanInputs, machine_for_experts};
    use model::placement::Machine;
    use model::placement::PlanLevers;
    use model::placement::churn::ChurnPool;
    use model::placement::workstation::{CardSpec, HostRead, PLAN_ROOM};
    use runtime::swaprule::SwapRule;
    use runtime::{Out, Target, Verify, Want};

    const NAME: &str = "gate_qwen38_residency";
    /// The stores' positions, `gate_qwen4exp_e2e`'s.
    const CTX: usize = 3072;
    /// Prompt positions: one prompt call, one ubatch.
    const PROMPT: usize = 64;
    /// Greedy steps after the verify: planning boundaries of the rule.
    const STEPS: usize = 96;
    /// A refusal before the load reads the file's headers only.
    const REFUSE_BOUND_S: f64 = 120.0;
    /// The streaming clause's prompt: one ubatch of real text, whose routing
    /// clears the pick's floor on most card layers.
    const STREAM_PROMPT: usize = 512;
    /// Greedy steps after the streaming clause's prompt.
    const STREAM_STEPS: usize = 32;
    /// The `split` clause's prompt: one ubatch past the stream's least
    /// width at the load's prices and at the gate's fixed rule input.
    const SPLIT_PROMPT: usize = 2048;
    /// The `split` clause's fixed rule input: a host column a
    /// thousand microseconds, so the stream's floor sits under one column and
    /// the balance cuts only the last few experts of a layer.
    const SPLIT_COSTS: Costs = Costs {
        host_us_per_col: 1000.0,
        host_us_fixed: 19.95,
        card_us_fixed: 13.0,
        card_us_per_col: 0.16,
    };
    /// The `kvfloor` clause's KV floor: the positions its plan spreads the
    /// card experts at, so every position the prompt call grows the stores
    /// by past it is bytes the plan gave to experts.
    const FLOOR_POSITIONS: u64 = 64;
    /// The `kvfloor` clause's prompt: ids of the qwen4exp prose corpus as
    /// one call, far past [`FLOOR_POSITIONS`] and, with the greedy steps
    /// after it, inside the first page of each store plane (a K or V row is
    /// 1,024 B a position, a page 2 MiB).
    const FLOOR_PROMPT: usize = STREAM_PROMPT;
    /// The `units` clause's prompt: the `stream` clause's, one ubatch the
    /// split walk cuts into two units of half its rows.
    const UNITS_PROMPT: usize = STREAM_PROMPT;
    /// A token's routed slots, the file's top-k.
    const N_USED: u64 = 10;
    /// The stream's event marks, to their resolution [ms].
    const EVENT_MS: f64 = 0.002;

    /// The machine's card plan of `inputs` on `card`.
    fn plan_of<'m>(
        inputs: &'m PlanInputs,
        card: &'m Machine,
    ) -> Result<model::placement::Plan<'m>, GateError> {
        plan_under(inputs, card, &PlanLevers::default())
    }

    /// [`plan_of`] under `levers`.
    fn plan_under<'m>(
        inputs: &'m PlanInputs,
        card: &'m Machine,
        levers: &PlanLevers,
    ) -> Result<model::placement::Plan<'m>, GateError> {
        plan_under_at(inputs, card, levers, CTX)
    }

    /// [`plan_under`] at a store context of `ctx` rows.
    fn plan_under_at<'m>(
        inputs: &'m PlanInputs,
        card: &'m Machine,
        levers: &PlanLevers,
        ctx: usize,
    ) -> Result<model::placement::Plan<'m>, GateError> {
        inputs
            .plan_with(
                card,
                (ctx as u64, u64::try_from(ubatch_for(ctx)?)?),
                levers,
                Experts::Card,
            )
            .map_err(|e| format!("the plan: {e}").into())
    }

    /// The machine the gate plans on: the card `a` resolves on the census (the
    /// placement the Qwen3.8 defaults treat as plan (a)), held as the real tier
    /// holds a census card (`tier::real_card`; a unified card plans its pool),
    /// laid out by `machine_for_experts`, at the largest usable bytes at which
    /// every card layer leaves the residency its pinned, spare and moving
    /// experts ([`residency_of`]) and the plan's host holds the churn pool
    /// beside it (`ChurnPool::check`): the card's own bytes where they do, else
    /// the largest whole GiB below them that does, each larger one printed
    /// with why it did not.
    fn residency_machine(inputs: &PlanInputs, ub: usize) -> Result<Machine, GateError> {
        residency_machine_under(inputs, ub, &PlanLevers::default())
    }

    /// [`residency_machine`] for the plan made under `levers`.
    fn residency_machine_under(
        inputs: &PlanInputs,
        ub: usize,
        levers: &PlanLevers,
    ) -> Result<Machine, GateError> {
        let card = tier::real_card(Place::A.on(&bloomery_gpu::census()?)?.card_specs()?[0]);
        let layers = inputs.stack();
        let at = |c: CardSpec| machine_for_experts(c, layers, ub as u64, Experts::Card);
        let why_not = |m: &Machine| -> Result<Option<String>, GateError> {
            let plan = match plan_under(inputs, m, levers) {
                Ok(plan) => plan,
                Err(e) => return Ok(Some(e.to_string())),
            };
            let Some(Residency::Mid { pinned, .. }) = residency_of(&plan) else {
                return Ok(Some(
                    "the plan's card experts leave no layer the machine's pinned, spare and one \
                     that moves"
                        .into(),
                ));
            };
            let pool =
                ChurnPool::of(&plan, 0, pinned).map_err(|e| format!("the churn pool: {e}"))?;
            Ok(pool.check(&plan).err().map(|e| e.to_string()))
        };
        let own = at(card);
        let Some(mut why) = why_not(&own)? else {
            println!(
                "residency card: {} at its own {} B",
                card.name,
                card.usable_bytes()
            );
            return Ok(own);
        };
        let mut above = card.usable_bytes();
        for gib in (1..=card.usable_bytes().saturating_sub(1) >> 30).rev() {
            let budget = CardSpec {
                total_bytes: (gib << 30) + card.driver_reserve_bytes,
                ..card
            };
            let machine = at(budget);
            match why_not(&machine)? {
                None => {
                    println!(
                        "residency card: {} under {gib} GiB, the largest whole GiB its plan fits; \
                         at {above} B: {why}",
                        card.name
                    );
                    return Ok(machine);
                }
                Some(w) => (why, above) = (w, budget.usable_bytes()),
            }
        }
        Err(format!(
            "no budget on {} gives the residency its card experts and the churn pool its host \
             room: {why}",
            card.name
        )
        .into())
    }

    /// The residency the gate runs over `plan`: half of the plan's smallest
    /// layer of card experts pinned (the seed is the plan's id prefix, not a
    /// frequency list, so pinning most of it protects nothing), one spare a
    /// layer. `None` when no layer holds the machine's `pinned + spares + 1`
    /// card experts.
    fn residency_of(plan: &model::placement::Plan<'_>) -> Option<Residency> {
        let slots = plan.n_l.iter().copied().filter(|&n| n > 0).min()? as usize;
        let pinned = slots / 2;
        (pinned >= 1 && slots >= pinned + 2).then_some(Residency::Mid { pinned, spares: 1 })
    }

    /// The lever word of [`residency_of`]'s answer.
    fn word_of(r: &Residency) -> String {
        match *r {
            Residency::Off => "off".to_string(),
            Residency::Mid { pinned, spares } => format!("mid-p{pinned}-s{spares}"),
        }
    }

    /// The model placed on the gate's card by its plan, the residency machine
    /// over the card's routed stacks at `residency`.
    fn open(
        path: &Path,
        inputs: &PlanInputs,
        machine: Machine,
        residency: Residency,
        host: bloomery_levers::HostCfg,
        ub: usize,
    ) -> Result<Qwen38Model, GateError> {
        open_under(
            path,
            inputs,
            machine,
            (residency, &PlanLevers::default()),
            host,
            ub,
        )
    }

    /// [`open`] for the plan made under `levers`.
    fn open_under(
        path: &Path,
        inputs: &PlanInputs,
        machine: Machine,
        (residency, levers): (Residency, &PlanLevers),
        host: bloomery_levers::HostCfg,
        ub: usize,
    ) -> Result<Qwen38Model, GateError> {
        open_under_at(path, inputs, machine, (residency, levers), host, CTX, ub)
    }

    /// [`open_under`] at a store context of `ctx` rows.
    fn open_under_at(
        path: &Path,
        inputs: &PlanInputs,
        machine: Machine,
        (residency, levers): (Residency, &PlanLevers),
        host: bloomery_levers::HostCfg,
        ctx: usize,
        ub: usize,
    ) -> Result<Qwen38Model, GateError> {
        let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        if file.architecture() != Some("qwen4exp") {
            return Err(format!("{} is not qwen4exp", path.display()).into());
        }
        let plan = plan_under_at(inputs, &machine, levers, ctx)?;
        Ok(Body38::open_placed_residency(
            file,
            &plan,
            inputs,
            0,
            host,
            ub,
            residency,
            bloomery_levers::ChurnHosting::Hosted,
            1,
        )?)
    }

    /// What a history saw: every call's argmax, each step's logits FNV, each
    /// boundary's ended pass (its kind and kept rows), the flips that
    /// landed, and the steps whose own boundary ran ahead of their readback
    /// ([`ended_ahead`]).
    #[derive(PartialEq)]
    struct History {
        tokens: Vec<u32>,
        fnvs: Vec<u64>,
        passes: Vec<(PassKind, usize)>,
        landed: usize,
        ahead: usize,
    }

    /// Whether the step just run ended at a boundary made ahead of its
    /// readback: the last of `passes`, the boundaries since the step began,
    /// ends a step, was made ahead and carries `before`, the model's
    /// readbacks before the step, and the model has read once since
    /// (`after`).
    fn ended_ahead(passes: &[(PassKind, PassReport)], before: u64, after: u64) -> bool {
        after == before + 1
            && passes
                .last()
                .is_some_and(|(k, r)| *k == PassKind::Step && r.ahead && r.reads == before)
    }

    /// One greedy step of `next` in the history `h`, `want` its readback:
    /// its argmax and logits FNV noted, its boundaries taken into `passes`
    /// and whether it ended ahead counted. The step's argmax.
    fn history_step(
        s: &mut Session<Body38>,
        h: &mut History,
        passes: &mut Vec<(PassKind, PassReport)>,
        next: u32,
        want: Want,
    ) -> Result<u32, GateError> {
        let before = s.model().reads();
        let out = s.step(next, want)?;
        if let Out::Logits { row, .. } = out {
            h.fnvs.push(Fnv1a64::default().f32s(row).value());
        }
        let next = out.argmax();
        h.tokens.push(next);
        let after = s.model().reads();
        let taken = take_passes(s)?;
        h.ahead += usize::from(ended_ahead(&taken, before, after));
        passes.extend(taken);
        Ok(next)
    }

    /// The boundaries' reports since the last take.
    fn take_passes(s: &mut Session<Body38>) -> Result<Vec<(PassKind, PassReport)>, GateError> {
        let (_, _, b) = s.model_mut().body_parts(NAME)?;
        Ok(b.take_residency_passes())
    }

    /// The history from a clear: the prompt, a step, a verify of three rows
    /// committed at two, then [`STEPS`] greedy steps.
    fn history(s: &mut Session<Body38>, ids: &[u32]) -> Result<History, GateError> {
        s.clear()?;
        take_passes(s)?;
        let mut h = History {
            tokens: Vec::with_capacity(STEPS + 4),
            fnvs: Vec::with_capacity(STEPS),
            passes: Vec::new(),
            landed: 0,
            ahead: 0,
        };
        let mut next = s.prompt(ids, Want::Argmax)?.argmax();
        h.tokens.push(next);
        let mut passes = take_passes(s)?;
        next = history_step(s, &mut h, &mut passes, next, Want::Argmax)?;
        let out = s.verify::<3>([next; 3])?;
        h.tokens.extend_from_slice(&out[1..]);
        s.commit(2)?;
        for _ in 0..STEPS {
            next = history_step(s, &mut h, &mut passes, next, Want::Logits)?;
        }
        passes.extend(take_passes(s)?);
        h.landed = passes.iter().map(|(_, r)| r.landed).sum();
        h.passes = passes.iter().map(|&(k, r)| (k, r.kept)).collect();
        Ok(h)
    }

    /// The machine's rule as it stands.
    fn rule_of(s: &Session<Body38>) -> Result<SwapRule, GateError> {
        Ok(s.model()
            .body(NAME)?
            .hybrid()
            .swap()
            .ok_or("the load runs no residency machine")?
            .rule()
            .clone())
    }

    /// The layers the card holds routed experts of, with their seeds.
    fn seeds(s: &Session<Body38>) -> Result<Vec<(usize, Vec<u32>)>, GateError> {
        let m = s.model();
        let b = m.body(NAME)?;
        let machine = b
            .hybrid()
            .swap()
            .ok_or("the load runs no residency machine")?;
        crate::residency_clauses::seeds(machine, b.hybrid().slots().layers())
    }

    /// `transform`: each admitted expert's slot against its static bytes,
    /// every part of it the source stages.
    fn transform_clause(s: &Session<Body38>) -> Result<bool, GateError> {
        let m = s.model();
        let b = m.body(NAME)?;
        let machine = b
            .hybrid()
            .swap()
            .ok_or("the load runs no residency machine")?;
        let source = b
            .residency_source()
            .ok_or("the load has no residency source")?;
        let layers = crate::residency_clauses::layer_seeds(machine, b.hybrid().slots().layers())?;
        let (checked, bad) =
            crate::residency_clauses::transform_check(m.gpu(), machine, source, &layers)?;
        Ok(crate::residency_clauses::transform_verdict(checked, &bad))
    }

    /// What one run of the streaming clause saw: the greedy ids, the prompt
    /// call's host slots, the first decode step's, the layers whose live set
    /// is not the seed after the call, whether a call is left open, and the
    /// call's end.
    struct StreamRun {
        tokens: Vec<u32>,
        /// The prompt call's last logits row, and each greedy id's top-1
        /// margin, the call's first.
        logits: Vec<f32>,
        margins: Vec<f32>,
        prompt_slots: u64,
        first_slots: u64,
        moved: usize,
        open: bool,
        end: Option<CallReport>,
        /// The host slots the prompt call's serves left out, by the serve's
        /// own count, and the call's `xstream` records.
        excluded: u64,
        xlayers: Vec<(usize, XLayer)>,
        xend: Option<XReport>,
    }

    /// The layers whose live set is not their seed.
    fn moved_layers(s: &Session<Body38>) -> Result<usize, GateError> {
        let m = s.model();
        let b = m.body(NAME)?;
        let machine = b
            .hybrid()
            .swap()
            .ok_or("the load runs no residency machine")?;
        let mut moved = 0;
        for l in b.hybrid().slots().layers() {
            let seed = machine.seed(l)?;
            let live: Vec<u32> = machine
                .ledger()
                .row(l)
                .unwrap_or(&[])
                .iter()
                .filter_map(|st| match st {
                    SlotState::Live(e) => Some(*e),
                    _ => None,
                })
                .collect();
            if live.len() != seed.len() || !seed.iter().all(|e| live.contains(e)) {
                moved += 1;
            }
        }
        Ok(moved)
    }

    /// The prompt calls' host streaming (`Body38::set_xstream`): the
    /// residency pool's pick under `admit`, the pick and the expert stream
    /// under `split`, nothing off.
    fn set_stream(s: &mut Session<Body38>, mode: XMode) -> Result<(), GateError> {
        let (gpu, _, body) = s.model_mut().body_parts(NAME)?;
        Ok(body.set_xstream(gpu, mode)?)
    }

    /// One run of a streaming clause from a clear, the prompt call's
    /// streaming `mode`; the lever off again after it.
    fn stream_run(
        s: &mut Session<Body38>,
        ids: &[u32],
        mode: XMode,
    ) -> Result<StreamRun, GateError> {
        s.clear()?;
        take_passes(s)?;
        set_stream(s, mode)?;
        let stats = |s: &Session<Body38>| -> Result<_, GateError> {
            Ok(s.model().body(NAME)?.hybrid().stats())
        };
        let before = stats(s)?.batch_host_slots;
        let excluded_before = stats(s)?.batch_excluded_slots;
        let out = s.prompt(ids, Want::Logits)?;
        let mut next = out.argmax();
        let Out::Logits { row, .. } = out else {
            return Err("the prompt call returned no logits row".into());
        };
        let logits = row.to_vec();
        let mut margins = vec![flip::head_margin(&logits)?];
        let prompt_slots = stats(s)?.batch_host_slots - before;
        let excluded = stats(s)?.batch_excluded_slots - excluded_before;
        let (_, end) = s.model_mut().body_parts(NAME)?.2.take_stream_records();
        let (xlayers, xend) = s.model_mut().body_parts(NAME)?.2.take_xstream_records();
        let moved = moved_layers(s)?;
        let open = s
            .model()
            .body(NAME)?
            .hybrid()
            .swap()
            .is_some_and(|m| m.call_open());
        let mut tokens = vec![next];
        let mut first_slots = 0;
        for i in 0..STREAM_STEPS {
            let before = stats(s)?.host_slots;
            let out = s.step(next, Want::Logits)?;
            next = out.argmax();
            if let Out::Logits { row, .. } = out {
                margins.push(flip::head_margin(row)?);
            }
            if i == 0 {
                first_slots = stats(s)?.host_slots - before;
            }
            tokens.push(next);
        }
        take_passes(s)?;
        set_stream(s, XMode::Off)?;
        Ok(StreamRun {
            tokens,
            logits,
            margins,
            prompt_slots,
            first_slots,
            moved,
            open,
            end,
            excluded,
            xlayers,
            xend,
        })
    }

    /// A run's greedy facts [`agree`] reads: the prompt call's last logits
    /// row, every greedy id (the call's first) and each id's top-1 margin.
    struct Greedy<'r> {
        logits: &'r [f32],
        tokens: &'r [u32],
        margins: &'r [f32],
    }

    impl StreamRun {
        fn greedy(&self) -> Greedy<'_> {
            Greedy {
                logits: &self.logits,
                tokens: &self.tokens,
                margins: &self.margins,
            }
        }
    }

    /// How a streamed run `on` agrees with the unstreamed `off` (the card and
    /// the host sum a token's experts in another split): the prompt's last
    /// logits row's max |diff|, within [`GREEDY_MARGIN`]; where the greedy
    /// ids first part, if anywhere; and whether they are equal or part at a
    /// near tie (`on`'s top-1 margin there below it).
    fn agree(on: &Greedy<'_>, off: &Greedy<'_>) -> (f32, bool, Option<usize>, bool) {
        let dlogit = on
            .logits
            .iter()
            .zip(off.logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let logits_ok = on.logits.len() == off.logits.len() && dlogit < GREEDY_MARGIN;
        let parted = on.tokens.iter().zip(off.tokens).position(|(a, b)| a != b);
        let ids_ok = on.tokens.len() == off.tokens.len()
            && parted.is_none_or(|i| on.margins.get(i).is_some_and(|&m| m < GREEDY_MARGIN));
        (dlogit, logits_ok, parted, ids_ok)
    }

    /// `split`: the expert stream at the gate's costs, off once and split
    /// twice, from a clear each.
    fn split_clause(s: &mut Session<Body38>) -> Result<bool, GateError> {
        let ids = prose_ids("qwen4exp", SPLIT_PROMPT)?;
        let off = stream_run(s, &ids, XMode::Off)?;
        // `split`'s first setting starts the stream; the gate's costs then.
        set_stream(s, XMode::Split)?;
        let body = s.model_mut().body_parts(NAME)?.2;
        body.set_xstream_costs(SPLIT_COSTS)?;
        println!("split: xstream={}", body.xstream_word());
        set_stream(s, XMode::Off)?;
        let a = stream_run(s, &ids, XMode::Split)?;
        let b = stream_run(s, &ids, XMode::Split)?;
        for (u, x) in &a.xlayers {
            record::xstream_layer(*u, x).print();
        }
        if let Some(e) = &a.xend {
            record::xstream_end(e).print();
        }
        let half = a.xend.map_or(0, |e| e.half_slots);
        let streamed: usize = a.xlayers.iter().map(|(_, x)| x.streamed).sum();
        let columns =
            |r: &StreamRun| -> u64 { r.xlayers.iter().map(|(_, x)| x.streamed_columns).sum() };
        let records_ok = !a.xlayers.is_empty()
            && a.xend.is_some_and(|e| e.streamed == streamed)
            && streamed > 0
            && a.xlayers.iter().all(|(_, x)| {
                x.streamed <= x.tail
                    && x.tail <= x.host
                    && x.streamed <= half
                    && (x.streamed == 0) == (x.streamed_columns == 0)
            })
            && off.xlayers.is_empty()
            && off.xend.is_none();
        let count_ok = a.excluded == columns(&a) && b.excluded == columns(&b) && off.excluded == 0;
        let (dlogit, logits_ok, parted, ids_ok) = agree(&a.greedy(), &off.greedy());
        let key = |r: &StreamRun| -> Vec<(usize, usize, usize, usize, u64, u64)> {
            r.xlayers
                .iter()
                .map(|(u, x)| {
                    (
                        *u,
                        x.layer,
                        x.tail,
                        x.streamed,
                        x.streamed_columns,
                        x.host_columns,
                    )
                })
                .collect()
        };
        let same_bits = a.logits.len() == b.logits.len()
            && a.logits
                .iter()
                .zip(&b.logits)
                .all(|(x, y)| x.to_bits() == y.to_bits());
        let repro_ok = key(&a) == key(&b) && same_bits && a.tokens == b.tokens;
        // PIN(2026-10-12): the logits term is printed, not judged. The bound it
        // read, GREEDY_MARGIN, is the top-1 gap at a parted id, not a bound on
        // the max over the vocabulary of |split − off|. That max moved when the
        // arena's half lost a slot to the checkpoint reserve a deeper default
        // context takes (the ctx rule), which moves the split's unit
        // boundaries. The ids term holds the contract (equal, or parted only
        // at a near tie): a split that drops or doubles an expert's share
        // parts the ids outside a near tie and turns this clause red.
        let ok = records_ok && count_ok && ids_ok && repro_ok;
        println!(
            "split: a {SPLIT_PROMPT}-id prompt call: {} xstream records, {streamed} experts \
             streamed (half {half}), each within its tail and the half {records_ok}; the serve's \
             excluded host slots {} / {} / {} (off / split / split again) against the records' \
             streamed columns {} / {} {count_ok}; the prompt's logits max|diff| against off \
             {dlogit} (under {GREEDY_MARGIN}: {logits_ok}; printed, not judged), {} greedy ids \
             equal or parted at a \
             near tie {ids_ok}{}; the two split runs' records, logits bits and ids the same \
             {repro_ok}: {}",
            a.xlayers.len(),
            off.excluded,
            a.excluded,
            b.excluded,
            columns(&a),
            columns(&b),
            a.tokens.len(),
            parted
                .map(|i| format!(
                    " (first parted at {i}, split's margin {:?})",
                    a.margins.get(i)
                ))
                .unwrap_or_default(),
            verdict(ok)
        );
        Ok(ok)
    }

    /// `stream`: the prompt call streaming against not, from a clear each.
    fn stream_clause(s: &mut Session<Body38>) -> Result<bool, GateError> {
        let ids = prose_ids("qwen4exp", STREAM_PROMPT)?;
        let off = stream_run(s, &ids, XMode::Off)?;
        let on = stream_run(s, &ids, XMode::Admit)?;
        if let Some(r) = &on.end {
            record::call_report(r).print();
        }
        let admitted = on.end.map_or(0, |r| r.admitted);
        let map_ok = on.moved > 0 && off.moved == 0 && admitted > 0 && !on.open && !off.open;
        let slots_ok = on.first_slots < off.first_slots;
        let (dlogit, logits_ok, parted, ids_ok) = agree(&on.greedy(), &off.greedy());
        // A prompt too short for the floor opens no call.
        let short_ids = prose_ids("qwen4exp", Body38::STREAM_FLOOR as usize - 1)?;
        s.clear()?;
        take_passes(s)?;
        set_stream(s, XMode::Admit)?;
        s.prompt(&short_ids, Want::Argmax)?;
        let (short_picks, short_end) = s.model_mut().body_parts(NAME)?.2.take_stream_records();
        set_stream(s, XMode::Off)?;
        take_passes(s)?;
        let short_ok = short_picks.is_empty() && short_end.is_none();
        println!(
            "stream: a {}-id prompt, streaming on: {} picks, an end {}: {}",
            short_ids.len(),
            short_picks.len(),
            short_end.is_some(),
            verdict(short_ok)
        );
        let ok = map_ok && slots_ok && logits_ok && ids_ok && short_ok;
        println!(
            "stream: a {STREAM_PROMPT}-id prompt call, off / on: layers moved from the seed {} / \
             {} ({admitted} experts admitted, a call left open {} / {}), the prompt's host slots \
             {} / {}, the first step's host slots {} / {}, the prompt's logits max|diff| {dlogit} \
             (under {GREEDY_MARGIN}: {logits_ok}), {} greedy ids equal or parted at a near tie \
             {ids_ok}{}: {}",
            off.moved,
            on.moved,
            off.open,
            on.open,
            off.prompt_slots,
            on.prompt_slots,
            off.first_slots,
            on.first_slots,
            on.tokens.len(),
            parted
                .map(|i| format!(
                    " (first parted at {i}, on's margin {:?})",
                    on.margins.get(i)
                ))
                .unwrap_or_default(),
            verdict(ok)
        );
        Ok(ok)
    }

    /// Every layer's live ids, ascending, as the machine's ledger holds
    /// them: a placement to compare across a prompt call.
    fn live_sets(s: &Session<Body38>) -> Result<Vec<Vec<u32>>, GateError> {
        let m = s.model();
        let b = m.body(NAME)?;
        let machine = b
            .hybrid()
            .swap()
            .ok_or("the load runs no residency machine")?;
        let mut sets = Vec::new();
        for l in b.hybrid().slots().layers() {
            let mut live: Vec<u32> = machine
                .ledger()
                .row(l)
                .unwrap_or(&[])
                .iter()
                .filter_map(|st| match st {
                    SlotState::Live(e) => Some(*e),
                    _ => None,
                })
                .collect();
            live.sort_unstable();
            sets.push(live);
        }
        Ok(sets)
    }

    /// One `history` run's facts: the call's end record and its picks,
    /// whether every layer's live set is back where the history left it,
    /// the prompt's last logits row and the first step's after it.
    struct HistoryRun {
        end: CallReport,
        picks: Vec<(usize, CallPick)>,
        back: bool,
        prompt: Vec<f32>,
        step: Vec<f32>,
    }

    /// One `history` run from a clear: a 64-id prompt streaming off, eight
    /// greedy steps (the history), then the first `n` corpus ids as one call
    /// streaming `mode`, its return in the walk or, `at_end`, whole at the
    /// call's end (`Body38::set_return_at_end`), then one greedy step.
    fn history_run(
        s: &mut Session<Body38>,
        mode: XMode,
        n: usize,
        at_end: bool,
    ) -> Result<HistoryRun, GateError> {
        let ids = prose_ids("qwen4exp", n)?;
        let warm = prose_ids("qwen4exp", 64)?;
        s.clear()?;
        take_passes(s)?;
        set_stream(s, XMode::Off)?;
        s.model_mut()
            .body_parts(NAME)?
            .2
            .set_return_at_end(at_end)?;
        let mut next = s.prompt(&warm, Want::Argmax)?.argmax();
        for _ in 0..8 {
            next = s.step(next, Want::Argmax)?.argmax();
        }
        take_passes(s)?;
        let before = live_sets(s)?;
        set_stream(s, mode)?;
        let out = s.prompt(&ids, Want::Logits)?;
        let next = out.argmax();
        let Out::Logits { row, .. } = out else {
            return Err("the prompt call returned no logits row".into());
        };
        let prompt = row.to_vec();
        let (picks, end) = s.model_mut().body_parts(NAME)?.2.take_stream_records();
        s.model_mut().body_parts(NAME)?.2.take_xstream_records();
        take_passes(s)?;
        set_stream(s, XMode::Off)?;
        let after = live_sets(s)?;
        let step = match s.step(next, Want::Logits)? {
            Out::Logits { row, .. } => row.to_vec(),
            Out::Argmax(_) => return Err("the step after the call returned no logits row".into()),
        };
        take_passes(s)?;
        s.model_mut().body_parts(NAME)?.2.set_return_at_end(false)?;
        let end = end.ok_or("the prompt opened no call")?;
        record::call_report(&end).print();
        Ok(HistoryRun {
            end,
            picks,
            back: before == after,
            prompt,
            step,
        })
    }

    /// Whether two rows are equal bit for bit.
    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// The largest |diff| between two rows of one length; infinite for two
    /// lengths.
    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::INFINITY;
        }
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// A call's picks as the facts that name what each moved: unit, layer,
    /// admitted, kept, the admits' and the victims' columns, the counts'
    /// digest.
    fn pick_keys(picks: &[(usize, CallPick)]) -> Vec<(usize, usize, usize, usize, u64, u64, u64)> {
        picks
            .iter()
            .map(|(u, p)| {
                (
                    *u,
                    p.layer,
                    p.admitted,
                    p.kept,
                    p.admit_cols,
                    p.victim_cols,
                    p.counts,
                )
            })
            .collect()
    }

    /// `history`: a prompt call before a decode history does not keep its
    /// placement: the decode has been routing experts of its own, so the
    /// call's end puts every layer back at the live set it started from and
    /// sends the experts it admitted to the host again, while a fresh
    /// server's call keeps its placement (the `stream` clause's gain). The
    /// call's last unit returns each layer in the walk; the same history
    /// with the return whole at the call's end (the twin) admits and copies
    /// back as many, and the prompt's last logits row and the first step's
    /// after it are the twin's bit for bit — the same picks, the same reads
    /// and serves in the call, the same end state slot for slot ([`history_run`])
    /// (mutants: the end keeps before a history too — the restore red here;
    /// the end always restores — the `stream` clause's fresh gain red; the
    /// walk's return asked before the layer's serve — refused by name).
    fn history_clause(s: &mut Session<Body38>) -> Result<bool, GateError> {
        let walk = history_run(s, XMode::Admit, STREAM_PROMPT, false)?;
        let twin = history_run(s, XMode::Admit, STREAM_PROMPT, true)?;
        let (end, back) = (&walk.end, walk.back);
        let twin_back = twin.back;
        let walked = end.return_us > 0 && twin.end.return_us == 0;
        let prompt_same = same_bits(&walk.prompt, &twin.prompt);
        let step_same = same_bits(&walk.step, &twin.step);
        let as_twin = end.admitted == twin.end.admitted
            && end.restored == twin.end.restored
            && prompt_same
            && step_same;
        let ok = !end.kept
            && end.restored > 0
            && end.admitted > 0
            && back
            && !twin.end.kept
            && twin_back
            && walked
            && as_twin;
        println!(
            "history: a call after 8 greedy steps, its return in the walk / at its end: {} / {} \
             experts admitted, the end kept {} / {} and restored {} / {}, the walk's return time \
             {} / {} us, every layer's live set back where the history left it {back} / \
             {twin_back}; the prompt's last logits and the first step's after it bit for bit \
             the end's {prompt_same} / {step_same}: {}",
            end.admitted,
            twin.end.admitted,
            end.kept,
            twin.end.kept,
            end.restored,
            twin.end.restored,
            end.return_us,
            twin.end.return_us,
            verdict(ok)
        );
        Ok(ok)
    }

    /// `history_split`: the `history` call under `split` at the gate's
    /// [`SPLIT_COSTS`], [`SPLIT_PROMPT`] ids, its return in the walk against
    /// the twin's at the call's end. The walk's last unit prices an admit
    /// whose layer still owes return copies at two copies
    /// (`xsplit::ReturnCost::Exposed`), so its picks may part from the
    /// twin's; where every pick is the twin's, the logits are the twin's bit
    /// for bit, and where one parts, the prompt's last logits row and the
    /// first step's stay within [`GREEDY_MARGIN`] of the twin's (a streamed
    /// or admitted expert sums its columns in another split). Either way the
    /// call returns in its walk, admits, sends its admits back and leaves
    /// every layer's live set where the history left it.
    fn history_split_clause(s: &mut Session<Body38>) -> Result<bool, GateError> {
        // `split`'s first setting starts the stream; the gate's costs then.
        set_stream(s, XMode::Split)?;
        s.model_mut()
            .body_parts(NAME)?
            .2
            .set_xstream_costs(SPLIT_COSTS)?;
        set_stream(s, XMode::Off)?;
        let walk = history_run(s, XMode::Split, SPLIT_PROMPT, false)?;
        let twin = history_run(s, XMode::Split, SPLIT_PROMPT, true)?;
        let same_picks = pick_keys(&walk.picks) == pick_keys(&twin.picks);
        let dprompt = max_diff(&walk.prompt, &twin.prompt);
        let dstep = max_diff(&walk.step, &twin.step);
        let bits = same_bits(&walk.prompt, &twin.prompt) && same_bits(&walk.step, &twin.step);
        let values_ok = if same_picks {
            bits
        } else {
            dprompt < GREEDY_MARGIN && dstep < GREEDY_MARGIN
        };
        let ok = !walk.end.kept
            && !twin.end.kept
            && walk.end.admitted > 0
            && walk.end.restored > 0
            && twin.end.restored > 0
            && walk.end.return_us > 0
            && twin.end.return_us == 0
            && walk.back
            && twin.back
            && values_ok;
        println!(
            "history_split: a {SPLIT_PROMPT}-id split call after 8 greedy steps, its return in \
             the walk / at its end: {} / {} experts admitted, restored {} / {}, kept {} / {}, the \
             walk's return time {} / {} us, every layer's live set back {} / {}; the picks the \
             twin's {same_picks}, the prompt's last logits and the first step's max|diff| \
             {dprompt} / {dstep} (bit for bit {bits}; under {GREEDY_MARGIN} where a pick parts) \
             {values_ok}: {}",
            walk.end.admitted,
            twin.end.admitted,
            walk.end.restored,
            twin.end.restored,
            walk.end.kept,
            twin.end.kept,
            walk.end.return_us,
            twin.end.return_us,
            walk.back,
            twin.back,
            verdict(ok)
        );
        Ok(ok)
    }

    /// `history_split_on`: the `history_split` call with the split walk on
    /// ([`SplitWalk::Every`]) for the walk and its twin alike — the 64-id
    /// prompt before the history and the call each walked as two units —
    /// under the same rule: the call returns in its walk (its last unit's
    /// back owes each layer's return, so no return asks for a layer one
    /// unit still has to serve), admits, sends its admits back and leaves
    /// every live set where the history left it; where every pick is the
    /// twin's the logits are the twin's bit for bit, else within
    /// [`GREEDY_MARGIN`]; both calls walked two units (mutant: the return
    /// owed at every unit's back — the next unit's shadow asks for a layer
    /// its sibling has not served, refused by name).
    fn history_split_on_clause(s: &mut Session<Body38>) -> Result<bool, GateError> {
        set_stream(s, XMode::Split)?;
        s.model_mut()
            .body_parts(NAME)?
            .2
            .set_xstream_costs(SPLIT_COSTS)?;
        set_stream(s, XMode::Off)?;
        let set = |s: &mut Session<Body38>, w| -> Result<(), GateError> {
            Ok(s.model_mut().body_parts(NAME)?.2.set_split_walk(w)?)
        };
        let units = |s: &Session<Body38>| -> Result<usize, GateError> {
            Ok(s.model().body(NAME)?.ubatch_units())
        };
        set(s, SplitWalk::Every)?;
        let r = (|| -> Result<_, GateError> {
            let walk = history_run(s, XMode::Split, SPLIT_PROMPT, false)?;
            let walk_units = units(s)?;
            let twin = history_run(s, XMode::Split, SPLIT_PROMPT, true)?;
            Ok((walk, walk_units, twin, units(s)?))
        })();
        set(s, SplitWalk::Off)?;
        let (walk, walk_units, twin, twin_units) = r?;
        let same_picks = pick_keys(&walk.picks) == pick_keys(&twin.picks);
        let dprompt = max_diff(&walk.prompt, &twin.prompt);
        let dstep = max_diff(&walk.step, &twin.step);
        let bits = same_bits(&walk.prompt, &twin.prompt) && same_bits(&walk.step, &twin.step);
        let values_ok = if same_picks {
            bits
        } else {
            dprompt < GREEDY_MARGIN && dstep < GREEDY_MARGIN
        };
        // The step after the call is a ubatch of none: the units read are
        // the call's.
        let units_ok = walk_units == 2 && twin_units == 2;
        let ok = !walk.end.kept
            && !twin.end.kept
            && walk.end.admitted > 0
            && walk.end.restored > 0
            && twin.end.restored > 0
            && walk.end.return_us > 0
            && twin.end.return_us == 0
            && walk.back
            && twin.back
            && values_ok
            && units_ok;
        println!(
            "history_split_on: the {SPLIT_PROMPT}-id split call by the split walk, its return in \
             the walk / at its end: {walk_units} / {twin_units} units (want 2), {} / {} experts \
             admitted, restored {} / {}, kept {} / {}, the walk's return time {} / {} us, every \
             layer's live set back {} / {}; the picks the twin's {same_picks}, the prompt's last \
             logits and the first step's max|diff| {dprompt} / {dstep} (bit for bit {bits}; \
             under {GREEDY_MARGIN} where a pick parts) {values_ok}: {}",
            walk.end.admitted,
            twin.end.admitted,
            walk.end.restored,
            twin.end.restored,
            walk.end.kept,
            twin.end.kept,
            walk.end.return_us,
            twin.end.return_us,
            walk.back,
            twin.back,
            verdict(ok)
        );
        Ok(ok)
    }

    /// A layer-batch of a `units` run: its layer and unit, its columns, the
    /// host slots its serve listed and those it left out to the expert
    /// stream, the first unit's stream gap, and — by the route taps against
    /// the slot map the call left — its slots on the card and in all.
    struct UnitRow {
        layer: usize,
        unit: usize,
        cols: usize,
        host: u64,
        excluded: u64,
        card: u64,
        all: u64,
    }

    /// One `units` run ([`units_run`]).
    struct UnitsRun {
        tokens: Vec<u32>,
        logits: Vec<f32>,
        margins: Vec<f32>,
        units: usize,
        picks: Vec<(usize, CallPick)>,
        xlayers: Vec<(usize, XLayer)>,
        rows: Vec<UnitRow>,
        /// The layers whose first unit's stream gap — from its download to
        /// its shadow — does not hold the second unit's front and download.
        out_of_order: Vec<usize>,
    }

    impl UnitsRun {
        fn greedy(&self) -> Greedy<'_> {
            Greedy {
                logits: &self.logits,
                tokens: &self.tokens,
                margins: &self.margins,
            }
        }

        /// The host slots the serves of `unit` left out at `layer`.
        fn excluded(&self, layer: usize, unit: usize) -> u64 {
            self.rows
                .iter()
                .filter(|r| r.layer == layer && r.unit == unit)
                .map(|r| r.excluded)
                .sum()
        }
    }

    /// One run of the `units` clause from a clear: the prompt call's
    /// streaming `mode`, its one ubatch walked under `walk` with the walk's
    /// timing and route taps armed, then [`STREAM_STEPS`] greedy steps;
    /// every setting back off after it.
    fn units_run(
        s: &mut Session<Body38>,
        ids: &[u32],
        mode: XMode,
        walk: SplitWalk,
    ) -> Result<UnitsRun, GateError> {
        let n = ids.len();
        s.clear()?;
        take_passes(s)?;
        set_stream(s, mode)?;
        let arm = |s: &mut Session<Body38>, on: bool| -> Result<(), GateError> {
            let (gpu, _, b) = s.model_mut().body_parts(NAME)?;
            b.set_split_walk(if on { walk } else { SplitWalk::Off })?;
            b.set_ubatch_route_taps(gpu, if on { n } else { 0 })?;
            Ok(s.model_mut().set_prompt38_stats(on)?)
        };
        arm(s, true)?;
        let r = units_prompt(s, ids);
        let disarmed = arm(s, false);
        let mut run = r?;
        disarmed?;
        let mut next = run.tokens[0];
        for _ in 0..STREAM_STEPS {
            let out = s.step(next, Want::Logits)?;
            next = out.argmax();
            if let Out::Logits { row, .. } = out {
                run.margins.push(flip::head_margin(row)?);
            }
            run.tokens.push(next);
        }
        take_passes(s)?;
        set_stream(s, XMode::Off)?;
        Ok(run)
    }

    /// [`units_run`]'s prompt call and what it read back before any step:
    /// the walk's units, its picks and `xstream` records, and a row per
    /// layer-batch — each unit's half of the taps' ids counted against the
    /// slot map the call left (a step's boundary may move it) beside the
    /// serve's counts — and the units' order on the stream.
    fn units_prompt(s: &mut Session<Body38>, ids: &[u32]) -> Result<UnitsRun, GateError> {
        let n = ids.len();
        let out = s.prompt(ids, Want::Logits)?;
        let next = out.argmax();
        let Out::Logits { row, .. } = out else {
            return Err("the prompt call returned no logits row".into());
        };
        let logits = row.to_vec();
        let margins = vec![flip::head_margin(&logits)?];
        let (gpu, _, b) = s.model_mut().body_parts(NAME)?;
        let units = b.ubatch_units();
        let widths = if units == 2 {
            vec![n.div_ceil(2), n / 2]
        } else {
            vec![n]
        };
        let taps = b.ubatch_route_taps(gpu, 0, n)?;
        // Each (layer, unit)'s slots on the card, and all its slots.
        let mut card: Vec<Vec<(u64, u64)>> = Vec::new();
        {
            let map = b.hybrid().slots();
            let mut t0 = 0;
            for (u, &w) in widths.iter().enumerate() {
                for tap in &taps[t0..t0 + w] {
                    if card.len() < tap.len() {
                        card.resize(tap.len(), vec![(0, 0); widths.len()]);
                    }
                    for (l, r) in tap.iter().enumerate() {
                        let on_card = r
                            .ids
                            .iter()
                            .filter(|&&id| {
                                matches!(map.slot(l, id), Some(Slot::Card(_) | Slot::Tier { .. }))
                            })
                            .count();
                        card[l][u].0 += on_card as u64;
                        card[l][u].1 += r.ids.len() as u64;
                    }
                }
                t0 += w;
            }
        }
        let (picks, _) = b.take_stream_records();
        let (xlayers, _) = b.take_xstream_records();
        let stats = s
            .model_mut()
            .take_prompt38_stats()?
            .ok_or("the prompt call's walk kept no timing record")?;
        let mut rows = Vec::with_capacity(stats.rows.len());
        for r in &stats.rows {
            let (on_card, all) = card
                .get(r.layer)
                .and_then(|c| c.get(r.unit))
                .copied()
                .ok_or_else(|| format!("a timing row of layer {} unit {}", r.layer, r.unit))?;
            rows.push(UnitRow {
                layer: r.layer,
                unit: r.unit,
                cols: r.cols,
                host: r.slots,
                excluded: r.excluded,
                card: on_card,
                all,
            });
        }
        let mut out_of_order = Vec::new();
        if units == 2 {
            for l in 0..card.len() {
                let row = |u: usize| stats.rows.iter().find(|r| r.layer == l && r.unit == u);
                let in_order = match (row(0), row(1)) {
                    (Some(a), Some(b)) => {
                        a.gap_ms.unwrap_or(0.0) + EVENT_MS >= b.front_ms + b.down_ms
                            && b.gap_ms.is_some()
                    }
                    _ => false,
                };
                if !in_order {
                    out_of_order.push(l);
                }
            }
        }
        Ok(UnitsRun {
            tokens: vec![next],
            logits,
            margins,
            units,
            picks,
            xlayers,
            rows,
            out_of_order,
        })
    }

    /// `units`: the split walk ([`SplitWalk::Every`]) against one unit, from
    /// a clear each, the prompt call's streaming off, `admit` and `split`.
    /// Its call picks once a card layer (the group's pick: unit 0's counts
    /// at twice, the same layers as one unit's); every layer-batch's slots —
    /// the host's its serve listed, those it left out to the stream and the
    /// card's by the route taps against the map the call left — are its
    /// columns' every slot, each unit's columns its half; under `split` each
    /// layer's streamed columns are twice the first unit's left-out slots
    /// and the second unit left out slots too (the shared stream); on every
    /// layer the second unit's front and download ran between the first
    /// unit's download and its shadow; off, the split walk's bits are one
    /// unit's, streamed it agrees with one unit as the stream rule allows
    /// (`stream`'s rule). Mutants: a pick per unit (the pick count); no
    /// share of the stream (the second unit's left-out slots); the stream
    /// read at the first unit (the call's end refuses a layer read by one of
    /// its two units, by name); the walk in today's order (the order).
    fn units_clause(s: &mut Session<Body38>) -> Result<bool, GateError> {
        let ids = prose_ids("qwen4exp", UNITS_PROMPT)?;
        set_stream(s, XMode::Split)?;
        s.model_mut()
            .body_parts(NAME)?
            .2
            .set_xstream_costs(SPLIT_COSTS)?;
        set_stream(s, XMode::Off)?;
        let widths = [ids.len().div_ceil(2), ids.len() / 2];
        let mut ok = true;
        for mode in [XMode::Off, XMode::Admit, XMode::Split] {
            let off = units_run(s, &ids, mode, SplitWalk::Off)?;
            let on = units_run(s, &ids, mode, SplitWalk::Every)?;
            let layers_of = |r: &UnitsRun| -> Vec<usize> {
                let mut l: Vec<usize> = r.picks.iter().map(|(_, p)| p.layer).collect();
                l.sort_unstable();
                l
            };
            let (on_layers, off_layers) = (layers_of(&on), layers_of(&off));
            let distinct = on_layers.windows(2).all(|w| w[0] < w[1]);
            let streams = mode != XMode::Off;
            let picks_ok = on_layers == off_layers
                && distinct
                && on.picks.iter().all(|(u, _)| *u == 0)
                && on_layers.is_empty() != streams;
            let sum_ok = |r: &UnitsRun, w: &[usize]| {
                r.rows.len() == w.len() * r.rows.iter().map(|x| x.layer + 1).max().unwrap_or(0)
                    && r.rows.iter().all(|x| {
                        x.unit < w.len()
                            && x.cols == w[x.unit]
                            && x.all == x.cols as u64 * N_USED
                            && x.host + x.excluded + x.card == x.all
                    })
            };
            let slots_ok = on.units == 2
                && off.units == 1
                && sum_ok(&on, &widths)
                && sum_ok(&off, &[ids.len()]);
            // The stream prices unit 0's counts at twice: its streamed
            // columns are twice what unit 0's serve left out.
            let streamed = |r: &UnitsRun, twice: u64| {
                r.xlayers
                    .iter()
                    .all(|(_, x)| x.streamed_columns == twice * r.excluded(x.layer, 0))
            };
            let shared: u64 = on
                .rows
                .iter()
                .filter(|r| r.unit == 1)
                .map(|r| r.excluded)
                .sum();
            let share_ok = if mode == XMode::Split {
                !on.xlayers.is_empty() && streamed(&on, 2) && streamed(&off, 1) && shared > 0
            } else {
                on.xlayers.is_empty() && on.rows.iter().all(|r| r.excluded == 0)
            };
            let order_ok = on.out_of_order.is_empty();
            let (dlogit, logits_ok, parted, ids_ok) = agree(&on.greedy(), &off.greedy());
            let values_ok = if streams {
                logits_ok && ids_ok
            } else {
                same_bits(&on.logits, &off.logits) && on.tokens == off.tokens
            };
            let clause_ok = picks_ok && slots_ok && share_ok && order_ok && values_ok;
            println!(
                "units ({}): a {}-id prompt call by the split walk / one unit: {} / {} units, \
                 picks at {} / {} card layers, one a layer, the group's {picks_ok}; every \
                 layer-batch's host, left-out and card slots its columns' every slot, each unit's \
                 columns its half {slots_ok}; the second unit's left-out slots {shared}, each \
                 layer's streamed columns twice the first unit's left-out {share_ok}; layers out \
                 of the fronts' order {:?} {order_ok}; the prompt's last logits max|diff| \
                 {dlogit}, the greedy ids parted at {parted:?} ({}) {values_ok}: {}",
                mode.word(),
                ids.len(),
                on.units,
                off.units,
                on_layers.len(),
                off_layers.len(),
                on.out_of_order,
                if streams {
                    "under the stream rule"
                } else {
                    "bit for bit"
                },
                verdict(clause_ok)
            );
            ok &= clause_ok;
        }
        Ok(ok)
    }

    /// What one run of the `kvfloor` clause saw: the greedy ids, the prompt
    /// call's last logits row and each id's top-1 margin (the call's first),
    /// the device's free bytes after the clear, the most expert-stack bytes
    /// the boundaries' reports held withdrawn, and each machine layer's
    /// withdrawn slots at the end.
    struct FloorRun {
        tokens: Vec<u32>,
        logits: Vec<f32>,
        margins: Vec<f32>,
        idle: usize,
        withdrawn_bytes: u64,
        withdrawn: Vec<(usize, usize)>,
    }

    impl FloorRun {
        fn greedy(&self) -> Greedy<'_> {
            Greedy {
                logits: &self.logits,
                tokens: &self.tokens,
                margins: &self.margins,
            }
        }

        /// The slots withdrawn from every layer, when it is the same count
        /// on all of them.
        fn k(&self) -> Option<usize> {
            let k = self.withdrawn.first()?.1;
            self.withdrawn.iter().all(|&(_, w)| w == k).then_some(k)
        }
    }

    /// Device memory held until the card refuses one more page: buffers of a
    /// gigabyte, halving to one page as the card refuses them, until less
    /// than a page is free or a page-sized buffer is refused for lack of
    /// memory (the driver keeps a reserve the free bytes count). Any other
    /// refusal is refused by name.
    fn squeeze(gpu: &bloomery_gpu::Gpu) -> Result<Vec<cuda_core::DeviceBuffer<u32>>, GateError> {
        let page = gpu.allocation_granularity()?;
        let mut held = Vec::new();
        let mut chunk = 1usize << 30;
        loop {
            let free = gpu.mem_info()?.0;
            if free < page {
                return Ok(held);
            }
            let take = chunk.min(free / page * page);
            match cuda_core::DeviceBuffer::<u32>::zeroed(gpu.stream(), take / 4) {
                Ok(b) => held.push(b),
                Err(_) if chunk > page => chunk /= 2,
                Err(e) if e.0 == cuda_core::sys::cudaError_enum_CUDA_ERROR_OUT_OF_MEMORY => {
                    return Ok(held);
                }
                Err(e) => {
                    return Err(format!(
                        "the squeeze: a {take} B buffer refused with {free} B free: {e}"
                    )
                    .into());
                }
            }
        }
    }

    /// Each layer of the machine and the slots withdrawn from it.
    fn withdrawn_slots(s: &Session<Body38>) -> Result<Vec<(usize, usize)>, GateError> {
        let b = s.model().body(NAME)?;
        let machine = b
            .hybrid()
            .swap()
            .ok_or("the load runs no residency machine")?;
        Ok(b.hybrid()
            .slots()
            .layers()
            .map(|l| (l, machine.ledger().withdrawn(l)))
            .collect())
    }

    /// The bytes the layers' stacks have given back and not had backed
    /// again, by the rooms' own counters.
    fn released_total(s: &Session<Body38>) -> Result<u64, GateError> {
        let b = s.model().body(NAME)?;
        let source = b
            .residency_source()
            .ok_or("the load has no residency source")?;
        Ok(b.hybrid()
            .slots()
            .layers()
            .map(|l| source.released_bytes(l))
            .sum())
    }

    /// The bytes withdrawing `k` slots from every layer gives back, derived
    /// from the page and each part's bytes a slot holds: a part's stack
    /// lends the pages wholly at or past slot `cap - k`, `pages(cap) -
    /// pages(cap - k)` of them, `pages` rounding bytes up to the page.
    fn stack_bytes_gone(s: &Session<Body38>, k: usize) -> Result<u64, GateError> {
        let b = s.model().body(NAME)?;
        let machine = b
            .hybrid()
            .swap()
            .ok_or("the load runs no residency machine")?;
        let source = b
            .residency_source()
            .ok_or("the load has no residency source")?;
        let page = s.model().gpu().allocation_granularity()? as u64;
        let pages = |bytes: usize| (bytes as u64).div_ceil(page);
        let mut gone = 0;
        for l in b.hybrid().slots().layers() {
            let cap = machine.ledger().row(l).map_or(0, <[SlotState]>::len);
            for &part in source.part_bytes(l) {
                gone += (pages(cap * part) - pages((cap - k) * part)) * page;
            }
        }
        Ok(gone)
    }

    /// The sequence cleared and every withdrawn slot of every layer
    /// restored, so the next run starts from the machine's full capacity: no
    /// engine path restores a withdrawn tail (a reset leaves it withdrawn).
    fn restore_all(s: &mut Session<Body38>) -> Result<(), GateError> {
        s.clear()?;
        take_passes(s)?;
        let held = withdrawn_slots(s)?;
        let tier = s.model_mut().body_parts(NAME)?.2.hybrid_mut();
        for (l, k) in held {
            if k > 0 {
                tier.swap_restore_tail(l, k)?;
            }
        }
        Ok(())
    }

    /// One run of the `kvfloor` clause from a clear: with `tight`, the card
    /// squeezed until it refuses a page after the clear, the prompt call of
    /// `ids`, then [`STREAM_STEPS`] greedy steps; the squeeze let go at the
    /// end. An error is the named refusal of a call.
    fn floor_run(s: &mut Session<Body38>, ids: &[u32], tight: bool) -> Result<FloorRun, GateError> {
        s.clear()?;
        take_passes(s)?;
        let idle = s.model().gpu().mem_info()?.0;
        let _held = if tight {
            squeeze(s.model().gpu())?
        } else {
            Vec::new()
        };
        let out = s.prompt(ids, Want::Logits)?;
        let mut next = out.argmax();
        let Out::Logits { row, .. } = out else {
            return Err("the prompt call returned no logits row".into());
        };
        let logits = row.to_vec();
        let mut margins = vec![flip::head_margin(&logits)?];
        let mut tokens = vec![next];
        for _ in 0..STREAM_STEPS {
            let out = s.step(next, Want::Logits)?;
            next = out.argmax();
            if let Out::Logits { row, .. } = out {
                margins.push(flip::head_margin(row)?);
            }
            tokens.push(next);
        }
        let passes = take_passes(s)?;
        Ok(FloorRun {
            tokens,
            logits,
            margins,
            idle,
            withdrawn_bytes: passes
                .iter()
                .map(|(_, r)| r.withdrawn_bytes)
                .max()
                .unwrap_or(0),
            withdrawn: withdrawn_slots(s)?,
        })
    }

    /// `kvfloor` (a growing sequence frees expert bytes): the load planned
    /// at a KV floor of [`FLOOR_POSITIONS`] positions, the same history of a
    /// [`FLOOR_PROMPT`]-id prompt call and [`STREAM_STEPS`] greedy steps from
    /// a clear, three times, each from the machine's full capacity. First
    /// with the card as the load left it: what the machine withdrew, if
    /// anything, is printed, not judged (the plan's slack decides it). Then
    /// with the card squeezed until it refuses a page: the call's first page
    /// has no room, so the fit runs out whatever the slack, the machine
    /// withdraws the same top slots of every layer (the fewest whose stacks
    /// give back the pages the stores ask and one more; every page of the
    /// prompt's lies in the stores' first page of each plane, so the steps
    /// ask nothing more), the fit runs again and the call completes; its
    /// reports hold the stacks' bytes withdrawn, equal to the bytes the page
    /// and the parts derive ([`stack_bytes_gone`]), and its greedy ids equal
    /// the first run's or part at a near tie (a layer's card set is smaller,
    /// so a token's experts sum in another split: `stream`'s rule). Then
    /// every withdrawn slot restored ([`restore_all`]): every layer's
    /// withdrawn count and the rooms' released bytes are 0, directly; and
    /// the third run, with the card as the load left it, is the first run's
    /// bit for bit (mutants: the fit's retry removed, which ends the
    /// squeezed run's call in `NoRoom`; the restore's stack pages left
    /// unbacked, which leaves the rooms' released bytes above 0 and the
    /// third run's bits apart).
    fn kvfloor_clause(
        path: &Path,
        inputs: &PlanInputs,
        host: bloomery_levers::HostCfg,
        ub: usize,
    ) -> Result<bool, GateError> {
        let levers = PlanLevers {
            kv_floor: Some(FLOOR_POSITIONS),
            ..PlanLevers::default()
        };
        let machine = residency_machine_under(inputs, ub, &levers)?;
        let (residency, floored) = {
            let plan = plan_under(inputs, &machine, &levers)?;
            (
                residency_of(&plan).ok_or(
                    "the floored plan's card experts leave no layer with room for the \
                     machine's pinned, spare and one that moves",
                )?,
                plan.kv_floor,
            )
        };
        let t0 = Instant::now();
        let mut s = Session::from_model(
            open_under(path, inputs, machine, (residency, &levers), host, ub)?,
            CTX as u32,
        );
        let b = s.model_mut().body_parts(NAME)?.2;
        b.log_residency(0);
        b.set_split_walk(SplitWalk::Off)?;
        take_passes(&mut s)?;
        println!(
            "kvfloor: {} at a floor of {FLOOR_POSITIONS} positions, plan's floor {floored:?}, \
             load in {:.1} s",
            word_of(&residency),
            t0.elapsed().as_secs_f64()
        );
        let ids = prose_ids("qwen4exp", FLOOR_PROMPT)?;
        let loaded = withdrawn_slots(&s)?.iter().all(|&(_, w)| w == 0);
        let room = floor_run(&mut s, &ids, false)?;
        println!(
            "kvfloor: the card as the load left it: idle free {} B, withdrawn {:?} slots a layer, \
             {} B (printed, not judged)",
            room.idle,
            room.k(),
            room.withdrawn_bytes
        );
        restore_all(&mut s)?;
        let tight = match floor_run(&mut s, &ids, true) {
            Ok(r) => r,
            Err(e) => {
                println!(
                    "kvfloor: the call with no page left FAILED: {e} {}",
                    verdict(false)
                );
                return Ok(false);
            }
        };
        let k = tight.k().filter(|&k| k > 0);
        let gone = match k {
            Some(k) => stack_bytes_gone(&s, k)?,
            None => 0,
        };
        let freed_ok = floored == Some(FLOOR_POSITIONS)
            && loaded
            && k.is_some()
            && tight.withdrawn_bytes == gone;
        let (dlogit, logits_ok, parted, ids_ok) = agree(&tight.greedy(), &room.greedy());
        println!(
            "kvfloor: with no page left, the machine withdrew {k:?} slots a layer over {} \
             layers, {} B in the reports against {gone} B derived {freed_ok}; the prompt's logits \
             max|diff| against the first run {dlogit} (under {GREEDY_MARGIN}: {logits_ok}; \
             printed, not judged), {} greedy ids equal or parted at a near tie {ids_ok}{}: {}",
            tight.withdrawn.len(),
            tight.withdrawn_bytes,
            tight.tokens.len(),
            parted
                .map(|i| format!(
                    " (first parted at {i}, squeezed run's margin {:?})",
                    tight.margins.get(i)
                ))
                .unwrap_or_default(),
            verdict(freed_ok && ids_ok)
        );
        restore_all(&mut s)?;
        let slots_back = withdrawn_slots(&s)?.iter().all(|&(_, w)| w == 0);
        let released = released_total(&s)?;
        let restored_ok = slots_back && released == 0;
        println!(
            "kvfloor: after the restore every layer's withdrawn count is 0 {slots_back}, the \
             rooms' released bytes {released}: {}",
            verdict(restored_ok)
        );
        let again = floor_run(&mut s, &ids, false)?;
        let same_ok = again.tokens == room.tokens
            && again.withdrawn_bytes == room.withdrawn_bytes
            && again.logits.len() == room.logits.len()
            && again
                .logits
                .iter()
                .zip(&room.logits)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        println!(
            "kvfloor: the third run, the card as the load left it (idle free {} B against {} B, \
             printed), is the first run's ids, logits bits and withdrawn bytes: {}",
            again.idle,
            room.idle,
            verdict(same_ok)
        );
        Ok(freed_ok && ids_ok && restored_ok && same_ok)
    }

    /// The `kvspill` clause's store context: at it a selecting store's plane
    /// buffers hold whole pages past the reach of its first run — every
    /// head's rows through a [`FLOOR_PROMPT`]-id prompt and [`STREAM_STEPS`]
    /// steps stay inside each buffer's first pages — and the reach that
    /// crosses [`SPILL_STEPS`] steps past them asks the buffers' next pages,
    /// pages the first run never touched.
    const SPILL_CTX: usize = 8192;
    /// The `kvspill` clause's greedy steps past the first run's, taking the
    /// reach across the plane buffers' next pages at [`SPILL_CTX`].
    const SPILL_STEPS: usize = 6_800;

    /// `kvspill` (a load planned under `BLOOMERY_KV_HOST=spill`; its plan is
    /// the card-backed one's, the stores' planes alone backing on the host
    /// when the card cannot make a page): from a clear, a
    /// [`FLOOR_PROMPT`]-id prompt call and [`STREAM_STEPS`] greedy steps with
    /// the card as the load left it — the pages the reach touches back on
    /// the card, the pooled keys' one page among them. Then the card
    /// squeezed until it refuses a page ([`squeeze`], the kvfloor clause's)
    /// and the same history replayed from one clear — a clear keeps the
    /// reached pages — with [`SPILL_STEPS`] more greedy steps: the growth
    /// asks the plane buffers' next pages, the card cannot make them and the
    /// host takes them, so the store's K/V planes hold host bytes, no slot
    /// anywhere is withdrawn (the host took the pages, not the machine; the
    /// pooled keys, already backed, ask the card nothing) and the replayed
    /// ids are the first run's bit for bit, the pages reading through the
    /// same pointer whatever backs them. Mutant: the store ignoring `spill`
    /// — the growth's pages refuse on the card, the machine withdraws
    /// expert slots for them and the clause reds by a layer's withdrawn
    /// count above 0.
    fn kvspill_clause(
        path: &Path,
        inputs: &mut PlanInputs,
        host: bloomery_levers::HostCfg,
    ) -> Result<bool, GateError> {
        let levers = PlanLevers {
            kv_host: bloomery_levers::KvHost::Spill,
            ..PlanLevers::default()
        };
        // The page a host-capable room takes, which the plan's store
        // rounding under the lever and the clause's spilled rooms read.
        let spec = tier::real_card(Place::A.on(&bloomery_gpu::census()?)?.card_specs()?[0]);
        inputs.kv_host_page = Some(bloomery_gpu_gates::generate::stage_host_page(spec)?);
        let ub = ubatch_for(SPILL_CTX)?;
        let machine = residency_machine_under(inputs, ub, &levers)?;
        let residency = {
            let plan = plan_under_at(inputs, &machine, &levers, SPILL_CTX)?;
            residency_of(&plan).ok_or(
                "the spill plan's card experts leave no layer the machine's pinned, spare and one \
                 that moves",
            )?
        };
        let t0 = Instant::now();
        let mut s = Session::from_model(
            open_under_at(
                path,
                inputs,
                machine,
                (residency, &levers),
                host,
                SPILL_CTX,
                ub,
            )?,
            SPILL_CTX as u32,
        );
        let b = s.model_mut().body_parts(NAME)?.2;
        b.log_residency(0);
        b.set_split_walk(SplitWalk::Off)?;
        take_passes(&mut s)?;
        println!(
            "kvspill: {} at a context of {SPILL_CTX} under spill, load in {:.1} s",
            word_of(&residency),
            t0.elapsed().as_secs_f64()
        );
        let ids = prose_ids("qwen4exp", FLOOR_PROMPT)?;
        let run = |s: &mut Session<Body38>, steps: usize| -> Result<Vec<u32>, GateError> {
            let mut next = s.prompt(&ids, Want::Argmax)?.argmax();
            let mut tokens = vec![next];
            for _ in 0..steps {
                next = s.step(next, Want::Argmax)?.argmax();
                tokens.push(next);
            }
            Ok(tokens)
        };
        let loose = run(&mut s, STREAM_STEPS)?;
        let idle = s.model().gpu().mem_info()?.0;
        let _held = squeeze(s.model().gpu())?;
        // One clear: the reached pages stay backed, the pooled keys' among
        // them, so the replay asks only the pages past the first run's
        // reach.
        s.clear()?;
        take_passes(&mut s)?;
        let tight = run(&mut s, STREAM_STEPS + SPILL_STEPS)?;
        let withdrawn = withdrawn_slots(&s)?;
        let hosted = s.model().body(NAME)?.kv_planes_room().1;
        let none_withdrawn = withdrawn.iter().all(|&(_, w)| w == 0);
        let ids_ok = tight[..loose.len()] == loose[..];
        println!(
            "kvspill: with no page left, {} new steps put {} B of the planes on the host, \
             withdrawn {:?} slots a layer over {} layers (0 everywhere: {}), and the replayed \
             {} ids are the unsqueezed run's: {}",
            SPILL_STEPS,
            hosted,
            withdrawn.iter().map(|&(_, w)| w).collect::<Vec<_>>(),
            withdrawn.len(),
            none_withdrawn,
            loose.len(),
            verdict(ids_ok)
        );
        let spill_ok = hosted > 0 && none_withdrawn;
        println!(
            "kvspill: the host took the pages, not the machine (idle free {} B before the \
             squeeze, printed): {}",
            idle,
            verdict(spill_ok && ids_ok)
        );
        Ok(spill_ok && ids_ok)
    }

    /// The load the slots clauses share: planned for the two slots it serves
    /// ([`PlanInputs::plan_with_slots`]; the gate's own load plans one, and
    /// a pass of several slots on it is refused by name), opened after the
    /// gate's session is dropped — the card holds one load — and serving
    /// both slots, the boundaries its load made taken.
    fn open_slots(
        path: &Path,
        inputs: &PlanInputs,
        machine: &Machine,
        host: bloomery_levers::HostCfg,
        ub: usize,
    ) -> Result<Session<Body38>, GateError> {
        let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        if file.architecture() != Some("qwen4exp") {
            return Err(format!("{} is not qwen4exp", path.display()).into());
        }
        let plan = inputs
            .plan_with_slots(
                machine,
                (CTX as u64, u64::try_from(ub)?),
                &PlanLevers::default(),
                Experts::Card,
                2,
            )
            .map_err(|e| format!("the plan: {e}"))?;
        let residency = residency_of(&plan)
            .ok_or("the two-slot plan's card experts leave the machine no room")?;
        let mut s = Session::from_model(
            Body38::open_placed_residency(
                file,
                &plan,
                inputs,
                0,
                host,
                ub,
                residency,
                bloomery_levers::ChurnHosting::Hosted,
                2,
            )?,
            CTX as u32,
        );
        let b = s.model_mut().body_parts(NAME)?.2;
        b.log_residency(0);
        b.set_split_walk(SplitWalk::Off)?;
        take_passes(&mut s)?;
        s.add_slots(2)?;
        Ok(s)
    }

    /// `slots_drafted`: a drafted pass of two slots' verify rows folds each
    /// slot's accepted rows — slot 0 keeping one of its three rows, slot 1
    /// two of theirs — so the boundary that ends it kept 3 rows as the mask
    /// {0, 3, 4}, row 1 and row 2 rejected, not the prefix a count of the
    /// kept rows folds (mutant: `commit_slots` passing `KeptRows::prefix`
    /// of the kept total). On the slots clauses' load ([`open_slots`]).
    fn slots_drafted_clause(s: &mut Session<Body38>, ids: &[u32]) -> Result<bool, GateError> {
        let mut last = [0u32; 2];
        for (slot, last) in last.iter_mut().enumerate() {
            s.select_slot(slot)?;
            *last = s.prompt(ids, Want::Argmax)?.argmax();
        }
        let drafted: Vec<Vec<u32>> = last.iter().map(|&t| vec![t; 3]).collect();
        let rows: Vec<(usize, &[u32])> = drafted
            .iter()
            .enumerate()
            .map(|(slot, ids)| (slot, &ids[..]))
            .collect();
        s.verify_slots(&rows)?;
        s.commit_slots(&[1, 2])?;
        s.model_mut().pass_boundary()?;
        let (kind, r) = take_passes(s)?
            .pop()
            .ok_or("slots_drafted: the commit's boundary")?;
        let want: u64 = [0usize, 3, 4].iter().map(|&r| 1u64 << r).sum();
        let ok = kind == PassKind::SlotsDrafted && r.kept == 3 && r.rows == want;
        println!(
            "slots_drafted: a pass of two slots' three drafted rows, slot 0 keeping 1 and slot 1 \
             2: the boundary ends a {} pass of {} kept rows, mask 0b{:b} ({{0, 3, 4}}: {}): {}",
            kind.word(),
            r.kept,
            r.rows,
            r.rows == want,
            verdict(ok)
        );
        Ok(ok)
    }

    /// `commit_counts` (module doc), on the slots clauses' load
    /// ([`open_slots`]): slot 0 from a clear and its prompt, slot 1 where
    /// `slots_drafted` left it. The second refused commit keeps slot 0's
    /// row by a count it can keep, so a check made slot by slot would have
    /// moved slot 0 before refusing slot 1.
    fn commit_counts_clause(s: &mut Session<Body38>, ids: &[u32]) -> Result<bool, GateError> {
        s.select_slot(0)?;
        s.clear()?;
        take_passes(s)?;
        let t = s.prompt(ids, Want::Argmax)?.argmax();
        let verified = s.model().pos() + 3;
        s.verify_slots(&[(0, &[t, t, t]), (1, &[t, t, t])])?;
        let under = s.commit_slots(&[0, 3]).err().map(|e| e.to_string());
        let over = s.commit_slots(&[1, 4]).err().map(|e| e.to_string());
        let named = under
            .as_deref()
            .is_some_and(|e| e.contains("slot 0 keeping 0 of its 3 rows"))
            && over
                .as_deref()
                .is_some_and(|e| e.contains("slot 1 keeping 4 of its 3 rows"));
        // The pass leaves slot 0 selected: its position is the model's.
        let at = s.model().pos();
        let unmoved = at == verified;
        let waits = |e: Option<String>| e.is_some_and(|e| e.contains("waits for its commit"));
        let session_waits = waits(s.select_slot(0).err().map(|e| e.to_string()));
        let model_waits = waits(s.model_mut().select_slot(0).err().map(|e| e.to_string()));
        let landed = s.commit_slots(&[3, 3]).is_ok();
        let stepped = landed && s.step(t, Want::Argmax).is_ok();
        let ok = named && unmoved && session_waits && model_waits && stepped;
        println!(
            "commit_counts: a verify of two slots' three rows each, a commit keeping slot 0 none \
             and one keeping slot 1 four (slot 0 one) refused by name {named} ({} / {}), slot 0 \
             at {at} where the verify stood it ({verified}) {unmoved}, the pass waiting for the \
             session {session_waits} and the model {model_waits}, a commit of every row then a \
             step {stepped}: {}",
            under.as_deref().unwrap_or("committed"),
            over.as_deref().unwrap_or("committed"),
            verdict(ok)
        );
        Ok(ok)
    }

    /// `refuse`: the host one byte short of the churn pool.
    fn refuse_clause(
        path: &Path,
        inputs: &PlanInputs,
        machine: &Machine,
        residency: &Residency,
        host: bloomery_levers::HostCfg,
        ub: usize,
    ) -> Result<bool, GateError> {
        let plan = plan_of(inputs, machine)?;
        let Residency::Mid { pinned, .. } = *residency else {
            return Err("the gate's residency is mid".into());
        };
        let pool = ChurnPool::of(&plan, 0, pinned).map_err(|e| format!("the churn pool: {e}"))?;
        pool.check(&plan)
            .map_err(|e| format!("the churn pool does not fit the plan's host: {e}"))?;
        record::residency_host(&word_of(residency), &pool, &plan).print();
        let (short, small) =
            crate::residency_clauses::refuse_head(machine, plan.host.headroom_bytes, pool.bytes)?;
        crate::residency_clauses::refuse_tail(short, pool.bytes, REFUSE_BOUND_S, || {
            open(path, inputs, small, *residency, host, ub)
        })
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[
            HOST_POPULATE,
            HOST_LOCK,
            CARD_DONTNEED,
            bloomery_gpu::arch::qwen3moe::ubatch::UBATCH_ENV,
        ])?;
        let host = levers.host();
        let path_buf = ref_model_path()?;
        let path = path_buf.as_path();
        // PIN(2026-10-28): the plan read `MemAvailable`, which another process's load moves; the
        // room is the workstation's own, so the machine's host figure alone binds `refuse`. A
        // unified card's pool caps it (`Plan::plan_at`).
        let mut inputs = PlanInputs::describe_in(
            &Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?,
            (PLAN_ROOM, HostRead::Given),
        )?;
        let ub = ubatch_for(CTX)?;
        let machine = residency_machine(&inputs, ub)?;
        let (residency, slots) = {
            let plan = plan_of(&inputs, &machine)?;
            let slots = plan
                .n_l
                .iter()
                .copied()
                .filter(|&n| n > 0)
                .min()
                .unwrap_or(0);
            (
                residency_of(&plan).ok_or(
                    "the plan's card experts leave no layer with room for the machine's pinned, \
                     spare and one that moves",
                )?,
                slots,
            )
        };
        println!(
            "residency: {} over the plan's {slots} card experts a layer at least",
            word_of(&residency)
        );

        let mut pass = refuse_clause(path, &inputs, &machine, &residency, host, ub)?;

        let t0 = Instant::now();
        let mut s = Session::from_model(
            open(path, &inputs, machine.clone(), residency, host, ub)?,
            CTX as u32,
        );
        // Every clause walks its ubatches as one unit, the split walk's own
        // (`units`, `history_split_on`) aside, which set it and set it back.
        let b = s.model_mut().body_parts(NAME)?.2;
        b.log_residency(0);
        b.set_split_walk(SplitWalk::Off)?;
        println!("load in {:.1} s", t0.elapsed().as_secs_f64());
        let seeds = seeds(&s)?;

        let ids = prose_ids("engram", PROMPT)?;
        let step =
            crate::residency_clauses::rule_after(&mut s, &ids, rule_of, take_passes, |s, t| {
                s.step(t, Want::Argmax)?;
                Ok(())
            })?;
        let one =
            crate::residency_clauses::rule_after(&mut s, &ids, rule_of, take_passes, |s, t| {
                s.verify::<2>([t, t])?;
                s.commit(1)?;
                Ok(())
            })?;
        let both =
            crate::residency_clauses::rule_after(&mut s, &ids, rule_of, take_passes, |s, t| {
                s.verify::<2>([t, t])?;
                s.commit(2)?;
                Ok(())
            })?;
        let c6 = one == step && both != one;
        println!(
            "c6: a verify keeping one row leaves the rule a step leaves ({}), keeping both does \
             not ({}): {}",
            one == step,
            both != one,
            verdict(c6)
        );
        pass &= c6;

        let first = history(&mut s, &ids)?;
        pass &= transform_clause(&s)?;
        let again = history(&mut s, &ids)?;
        pass &=
            crate::residency_clauses::c1_clause(first.tokens.len(), first.landed, first == again);

        // The prompt call is one pass that keeps 0 rows; a step keeps 1; a
        // verify keeps its accepted rows. A step's own boundary is made ahead
        // of its readback, so the history's last step ends one too.
        // PIN(2026-10-01): STEPS steps after the verify, not STEPS - 1: the last one's boundary runs ahead.
        let mut want = vec![
            (PassKind::None, 0),
            (PassKind::Prompt, 0),
            (PassKind::Step, 1),
            (PassKind::Pair, 2),
        ];
        want.extend(std::iter::repeat_n((PassKind::Step, 1), STEPS));
        let passes_ok = first.passes == want;
        println!(
            "passes: a history's boundaries end none, the prompt call (0 kept), a step (1), a \
             verify (2), then {STEPS} steps (1 kept each): {} boundaries, same {passes_ok}: {}",
            first.passes.len(),
            verdict(passes_ok)
        );
        pass &= passes_ok;

        // Every plain step of the history: the one after the prompt and the
        // STEPS after the verify.
        let order = first.ahead == STEPS + 1;
        println!(
            "order: {} of {} steps ended at a boundary made ahead of their readback, stamped \
             with the readbacks before the step: {}",
            first.ahead,
            STEPS + 1,
            verdict(order)
        );
        pass &= order;

        let r = s
            .residency_reset()?
            .ok_or("the load runs no residency machine")?;
        record::residency_reset(&r).print();
        // The placement `machine` stays bound below (the slots clauses' load
        // plans on it), so the residency machine's borrow stays in the call.
        pass &= crate::residency_clauses::c7_clause(
            &r,
            {
                let m = s.model();
                let b = m.body(NAME)?;
                b.hybrid().swap().ok_or("no machine")?
            },
            &seeds,
        );

        pass &= stream_clause(&mut s)?;
        pass &= history_clause(&mut s)?;
        pass &= split_clause(&mut s)?;
        pass &= history_split_clause(&mut s)?;
        pass &= units_clause(&mut s)?;
        pass &= history_split_on_clause(&mut s)?;
        // The gate's load is done: the card holds one load, and the slots
        // clauses bring their own.
        drop(s);
        let mut s = open_slots(path, &inputs, &machine, host, ub)?;
        pass &= slots_drafted_clause(&mut s, &ids)?;
        pass &= commit_counts_clause(&mut s, &ids)?;
        // The card holds one load: the floored plan's own comes after the
        // slots clauses', and the spilled one's after it.
        drop(s);
        pass &= kvfloor_clause(path, &inputs, host, ub)?;
        pass &= kvspill_clause(path, &mut inputs, host)?;

        if pass {
            println!("{NAME}: every clause passed");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
