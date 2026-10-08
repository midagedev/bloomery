//! The GLM-5.3-Flash seat — the llama-server-compatible HTTP API on the
//! glm5next engine, opened and stepped as `generate_glm5next` opens and steps
//! the model. `bloomery-serve --model glm` is one call of [`run`], which
//! takes the process's arguments (`--model` already taken out by the
//! one-binary server).
//!
//!     [--host 127.0.0.1] [--port 8080] [--place a|gate|bp|<stage>[+<tier>…]] [--ctx C]
//!     [--alias NAME] [--cache-ram MIB] [--slot-save-path DIR] [--chat-template-file PATH]
//!     [--prefill batch|steps] [--parallel N] [--queue-depth Q] [--plan]
//!
//! `--ctx` is also spelled `--ctx-size` and `-c`, as llama-server spells it
//! (`serve::flag::CTX`); a number flag's value that is not a number is
//! refused naming the flag and the value.
//!
//! The model is `$BLOOMERY_REF_MODEL`; its first shard gives the vocabulary,
//! `tokenizer.chat_template` the chat template (`--chat-template-file`
//! replaces it, as the qwen38 seat takes it) and `general.name` the default
//! alias. Tool calls are the template's: the server reads their markup out of
//! the generation (`serve::glmxml`, which this file's template teaches), so
//! the seat adds nothing there. The `plan` record, then the `load` and
//! `capture` lines of the open, go to stderr as `generate_glm5next` prints
//! them, then the `listening on http://<addr>` line once the model is loaded
//! and the port is bound (`--port 0` binds a free one). Sampling is the
//! sampler crate's chain with no repetition penalty; `temperature <= 0` is
//! the engine's argmax, the
//! ids `generate_glm5next --tokens <the prompt's ids>` prints.
//!
//! `--place` is the shared placement word (`generate::Place`, as
//! `generate_glm5next` takes it): `a` (the serving plan, `workstation::plan_a`:
//! the stage on the largest visible card, the A6000 here), `gate` (the gate
//! card's), or `bp` (plan (b′): plan (a)'s stage and the next-largest card an
//! expert tier under the host tier — the 3090 under the A6000 here — its
//! prompt-batch bytes `place::tier_batch`'s) and its list spelling
//! `a6000+3090`; a list of more tier cards than the GLM body serves
//! (`bloomery_gpu::host::SERVED_TIERS`) is refused by name before the plan.
//! Unset, the common rule every serving seat takes decides
//! (`Place::choose`, by the GLM family's `glm_place::TIER_RULE`: the tier
//! card kept only when the plan puts at least the rule's break-even experts
//! on it and more than none) on the census the placement resolves against,
//! read once. A `place unset` record after the levers' records and before
//! the `ctx` line names the word, why (`set` under the flag; `one card`;
//! on two cards `tier at or past the break-even`, `tier under the
//! break-even`, `the tier holds no expert`, or `the tier's plan refused`
//! (the refusal on stderr)), the tier's experts in that plan of `bp` where
//! the rule asked the plan for them (never under the flag or on one card),
//! and the rule's break-even and basis. A refusal of the chosen placement's
//! plan is the seat's refusal by name. The
//! positions a slot serves are the stores the load sized for each sequence
//! (`--ctx` names one request's context: while `--parallel` names no count
//! one slot serves the whole of it, and under `--parallel N` it is the
//! total the N slots split): `/props`' `n_ctx` is that
//! number, a prompt that long is a 400 before it reaches the engine, and
//! generation stops there with `truncated`. Unset, a slot's context is the
//! file's trained context (`context_length`) capped to what the plan takes —
//! the largest context whose plan of every slot stands, itself never past
//! `place::ORACLE_POSITIONS` — pulled back to the largest multiple of
//! `CTX_STEP` that stays within the plan's `MARGIN` of stage-card expert
//! bytes (`placement::ctx`'s guard, qwen38's margin rule: more positions
//! on the card push card experts to the host, and decode crawls), and never
//! under 2048 unless the card holds less than that (qwen38's `card` rule);
//! a `ctx` line on stderr names the rule, the chosen context, the slots and
//! their total, the trained context, the fit and the margin. Set, the flag
//! is the total and each slot takes `total / N` positions rounded down
//! (llama-server's `-np N` without `-kvu`) under a set `--parallel N` —
//! alone it is one request's context, one slot at the whole of it
//! (`placement::ctx::slots_of`; one line on stderr says so) — a slot
//! under the body's floor
//! refused by name before the plan — one position, or the MTP draft's
//! window under `BLOOMERY_DRAFT=mtp` — and the plan refusing it by name past
//! the oracle and when the card cannot hold every slot.
//!
//! The keep rule: each KDA layer holds one recurrent state, and its history
//! only in the checkpoints a prompt call takes (its start, every 512th
//! position of the whole context, its end — `CHECKPOINT_EVERY`), so a cut
//! keeps every fed position, the empty model, or the checkpoint at or below
//! it — never a position between two checkpoints. A request that shares less
//! than that with the slot is a `cache reuse` record with the rule. The seat
//! marks no user-start token, so the server asks nowhere to cut a prompt
//! call: the checkpoint spacing owns where a cut can land.
//!
//! The prompt cache (llama-server's `--cache-ram`, in MiB; 0 turns it off)
//! holds the slot's sequence state when a request of another session takes
//! the slot, never for the session's next turn that carries its last prompt
//! whole (the reply rendered again without its reasoning, which never comes
//! back), and also for one of the session that leaves part of that prompt
//! out and keeps less than half of the slot
//! (`bloomery_gpu_glm5next::seq_save`): every latent layer's rows
//! of the held positions, each KDA layer's state and conv ring at the held
//! position and at the last checkpoint below it (the last prompt call's
//! end), and on a NextN load the layer's store rows and the target's rows
//! its draft reads next, with the draft's waiting rows beside them
//! ([`SlotDrafts::park`]). A returning session's state comes back whole
//! after the session's reset (`seq_resume`), its checkpoints that one point,
//! its draft joining where it left (`SlotDrafts::unpark`); the residency is
//! the model's and stays where use has taken it, the state carrying no slot
//! map. Its default is the lesser of `bind::CACHE_RAM_CAP` and half of what
//! `MemAvailable` leaves at load past the plan's host need, the residency's
//! churn pool and the checkpoints' host budget (`bind::CacheRam`); a `cache`
//! line on stderr after the `residency host` record (`--plan` too) prints it
//! with each term. A state of another model, card, context or store layout
//! is refused by name; every save, load, eviction and skip prints as a line.
//! `--slot-save-path` names the directory the slot actions answer from
//! (none refuses every one, as llama-server does): `erase` drops the slot,
//! and `save` and `restore` answer the server's own 501, a state being a
//! host value and not a file.
//!
//! `--parallel N` (`-np N`) serves N resident sequences inside
//! the one model (`Session::add_slots` over the body's `Slots`): the server
//! steps every running slot in each round, the sequences switched by
//! pointer exchange and each slot's own draft held beside it
//! ([`SlotDrafts`]), so each request's tokens are its solo run's;
//! `--parallel 1` is the one-sequence server and `--parallel 0` is refused
//! by name. `--parallel` naming no count, the seat serves one slot beside a
//! set `--ctx` (the flag is one request's context, `placement::ctx::slots_of`) and its
//! default two over the automatic context. The round's shape the open decided once, from the load and the
//! body's `SlotRows::MAX_ROWS`: a plain load (no NextN draft) runs a round of
//! steps as one pass of the busy rows
//! (`serve_seats::rounds::step_rows_one_pass`, cut at that bound); a NextN
//! load runs a round of drafted passes as one pass of the busy slots'
//! windows (`serve_seats::rounds::pass_rows_one_pass`, each slot's window
//! over its own draft) when the body's pass holds two windows of one
//! proposal each, else as a select and a pass a row. A slot whose draft
//! skips — it proposes nothing until the slot's next reset, its row one
//! plain step in every shape — is turned off before such a round and steps
//! alone, the table's off path (`SlotDrafts::turn_off`): the NextN pass of
//! several slots takes each slot's whole verify, its two rows. A NextN
//! load's round of steps — sampled and id-banning requests, which the
//! server steps — is a select and a step a row: its body refuses a pass of
//! plain steps by name, each of its slots being drafted and such a pass
//! telling no slot's draft what it ran, so a round of greedy and sampled
//! requests runs the steps' call and the passes' apart. A refusal
//! mid-round is the server's error, never a fallback to the loop. A
//! `parallel` line on stderr names the rule (`slots`), the slots, a slot's
//! context, the total, what set the count (`from`: the `--parallel` flag, a
//! set `--ctx`, the default) and the shape: `pass=one` where the open decided one
//! pass, `pass=turns` for a NextN load on a body whose pass holds fewer
//! rows than two windows; under
//! `BLOOMERY_STEP_STATS=1` each round of several slots prints a `slots
//! round` record naming its command, rows, passes and the slots the seat
//! serves. The plan counts every sequence in its KV term
//! (`PlanInputs::plan_slots`, `plan_nextn_slots`), so the context search, the
//! load and `/props`' `vram_kv_bytes` see them all, and a card that cannot
//! hold them is refused by name. Nothing parks — no slot waits for another —
//! so `--park-ram` is refused by name. `--queue-depth Q` bounds the requests
//! that wait for a slot.
//!
//! `BLOOMERY_DRAFT=mtp` loads the file's next-token layer beside the target
//! (`app::arch::glm5next::open_nextn`, the plan `PlanInputs::plan_nextn`
//! makes: the layer's card bytes and arena reserved, each KDA layer's state
//! two lanes) and drives the session through the runtime's speculative loop
//! with the shared window `app::mtp::MtpDraft`, as `generate_glm5next` does:
//! windows of two rows, the target's next token and the draft's one
//! proposal, the draft's walks eager, `pass_rows` 2; the greedy ids are the
//! plain run's of the same load. A sampling or id-banning request takes
//! plain steps (the server's loop asks a pass only of a greedy request with
//! no banned id), each step's row the target's, read before the step is told
//! to the draft. A `load draft=mtp` record follows the `load` line, and
//! `/props`' `engine.draft` names the draft (the target file, whose NextN
//! layer it is) with the plan's bytes for it as the card's `draft` class.
//! `BLOOMERY_DRAFT=off` is the plain path, one step a pass, with a `load
//! draft=off (<why>)` record. Unset follows the placement
//! (`bloomery_levers::glm_unset`), its word and why a `draft unset` record
//! before the plan: `mtp` under `--place a` and `bp` on a file of one
//! next-token layer; the plain path under `--place gate`, on a file of other than one,
//! and with stores too short for one window, never a refusal. Unset with the
//! ctx rule's own default, the draft then yields to the context
//! (`bloomery_levers::DraftYield::of`, qwen38's module doc, the one rule):
//! it goes off when the plan with it leaves a slot under the base the plain
//! rule aims for, one `draft yield` record naming its card bytes, the
//! positions a slot gets either way and the base, the plain rule's `ctx`
//! line following. Set, refused by
//! name: `mtp` with stores too short for one window (`--ctx` under 3: the
//! prompt's last id, its first token and the window's second row), and every
//! other word.
//!
//! `BLOOMERY_RESIDENCY` set prints as a `residency lever` record first
//! thing. Unset follows the placement and the plan
//! (`bloomery_levers::glm_unset`, `glm_residency_at_plan`), its word and why a
//! `residency unset` record after the `plan` line: under `--place a` and
//! `bp` `mid-p0-s1`; `off` under `--place gate`, with stores too short for one
//! window, beside `--prefill steps`, and when the plan holds no card expert,
//! its fewest leave no room for the word, or the churn pool does not fit the
//! plan's host headroom less the NextN layer's host experts or what
//! `MemAvailable` leaves past the load's host need — never a refusal.
//! `--plan` prints these records and the `plan` line and exits 0 before any
//! card is opened. Running `mid-p<P>-s<S>` (plain or
//! drafted), the load runs the common residency machine over the card's
//! routed stacks (`open_resident`, or `open_nextn` with the word, the
//! next-token layer's experts on the host): a `residency host` record
//! follows the `plan` line, the word refused by name when the plan's fewest
//! card experts a layer leave no room for P pinned, S spares and one that
//! moves or its host headroom cannot take the churn pool, and each call (a
//! prompt, a step, a pass) prints its boundaries' `residency pass` records
//! after it. A request's reset keeps the residency where use has taken it;
//! `POST /residency/reset` moves it back to its seed and prints a `residency
//! reset` record (without the residency, the server's 501). `mid-…` set is
//! refused by name beside `--prefill steps`, as the CLI refuses it.
//!
//! An engine error ends the process: the request gets a 500, `/health` a 503
//! for a moment, then the crash block (card, position, error) goes to stderr
//! and the exit code is 70. The levers this seat acts on ([`ACTS_ON`]) are
//! parsed once, at `main` (`bloomery_levers::at_main`), which refuses by name
//! a lever set outside them; `--levers` prints them with this process's
//! values and exits. The stderr lines named above are records of the kinds
//! `bloomery_gpu_gates::record` declares, but for the `cache` line and the
//! prompt cache's notes past `cache reuse`, which print as their own lines;
//! `--records-schema` prints those kinds and exits.

use std::any::Any;
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use app::arch::glm5next::{GlmCfg, open_nextn_slots, open_resident_slots};
use app::mtp::{MtpBody, MtpDraft};
use app::{OpenLog, RowsLog, Session, SessionError};
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::model::{SlotRows, StepMode};
use bloomery_gpu_gates::bind::{
    CacheRam, Seat, SeatEngine, SlotPassRow, SlotStep, Vocab, model_props, nvidia_smi_index,
    placement_props, sampler_factory,
};
use bloomery_gpu_gates::generate::{Place, mode_name, with_cards};
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::residency38::{GLM_CARD, residency_room, residency_set};
use bloomery_gpu_gates::{GateError, gpu_census, ref_model_path};
use bloomery_gpu_glm5next::{
    Body, Glm5nextModel, GlmArena, GlmSeq, PrefillMode, seq_resume, seq_save,
};
use bloomery_levers::{
    GlmAt, GlmPick, ResidencyPick, ResidencyWhy, glm_residency_at_plan, glm_unset,
};
use gguf::Split;
use model::arch::glm5next::place::{
    KdaLanes, NextnInputs, ORACLE_POSITIONS, PROMPT_GROUP, PlanInputs,
};
use model::placement::churn::ChurnPool;
use model::placement::slots::{SplitError, split_ctx};
use model::placement::workstation::{HostNeed, MARGIN, TierBatchBytes, host_available};
use model::placement::{Machine, Plan, PlanLevers};
use runtime::Target;
use runtime::seqstate::Why;
use runtime::width::Mode as WidthMode;
use serve::flag::number;
use serve::{
    CacheNote, DraftProps, Drafted, EngineProps, FATAL_LINGER, ResidencyReset, Saved, ServeError,
    Server, ServerConfig, SlotConfig,
};
use tokenizer::Tokenizer;

use super::drafted::{ParkedDraft, SlotDrafts};
use crate::glm_place;

/// The seat's name, as its records and errors print it.
const WHAT: &str = "bloomery-serve-glm";

const USAGE: &str = "usage: bloomery-serve [--model glm] [--host H] [--port P] \
                     [--place a|gate|bp|<stage>[+<tier>…]] \
                     [--ctx C] [--alias NAME] [--cache-ram MIB] [--slot-save-path DIR] \
                     [--chat-template-file PATH] [--prefill batch|steps] [--parallel N] \
                     [--queue-depth Q] [--plan] [--api-key KEY] [--api-key-file FNAME]";

/// The positions the stores are sized for when `--ctx` names none:
/// `generate_glm5next`'s default, and the floor [`ctx_of`]'s rule never
/// goes under while the card holds it.
const CTX: usize = 2048;

/// The default context is a multiple of this many positions
/// (`placement::ctx`'s guard rounds to it).
const CTX_STEP: usize = 256;

/// The levers this seat acts on: those `generate_glm5next` reads for its load
/// and its draft (`BLOOMERY_ROUTE_TRACE` left out — one run's instrument,
/// which a server that serves many prompts does not wire) and
/// `BLOOMERY_STEP_STATS`, which the seat reads to count its rounds of several
/// slots (a `slots round` record each); a lever set outside this list is
/// refused by name at `main`, with `BLOOMERY_PIN_MAIN` for the engine
/// thread's cpu slot, as every serving seat reads it; `gate_glm5next_serve`
/// holds the same list.
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
    bloomery_levers::R8,
    bloomery_levers::PIN_MAIN,
    bloomery_levers::DRAFT,
    bloomery_levers::MTP_WIDTH,
    bloomery_levers::RESIDENCY,
    bloomery_levers::STEP_STATS,
];

/// The drafted window's verify: the target's next token and the draft's one
/// proposal.
const VERIFY_ROWS: usize = <Body as MtpBody>::VERIFY_ROWS;

/// The shape the seat runs a round of several slots in, decided once at the
/// open from the load and a fact of the body; a refusal mid-round is the
/// server's error, never a fallback to another shape.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rounds {
    /// A plain load: one pass of the busy rows' steps
    /// (`serve_seats::rounds::step_rows_one_pass`, cut at the body's
    /// [`SlotRows::MAX_ROWS`]).
    Steps,
    /// A NextN load whose body's pass of several slots holds two drafted
    /// windows: one pass of the busy slots' windows
    /// (`serve_seats::rounds::pass_rows_one_pass`, cut between windows).
    Windows,
    /// A NextN load whose body's pass does not: a select and a pass a row
    /// ([`pass_rows_in_turn`]).
    Turns,
}

impl Rounds {
    /// The shape of a load that drafts (`drafts`) or not: a drafted round
    /// is one pass when the body's pass of several slots holds two windows
    /// of the seat's depth — the token at a slot's position and its
    /// proposal of [`MtpBody::WIDTH`] ids — so two slots' windows run
    /// together; a body whose pass holds fewer rows keeps the fallback loop.
    fn of(drafts: bool) -> Rounds {
        if !drafts {
            Rounds::Steps
        } else if <Body as SlotRows>::MAX_ROWS >= 2 * (1 + <Body as MtpBody>::WIDTH) {
            Rounds::Windows
        } else {
            Rounds::Turns
        }
    }

    /// The `parallel` line's `pass=` word: `one` or `turns`.
    fn word(self) -> &'static str {
        match self {
            Rounds::Steps | Rounds::Windows => "one",
            Rounds::Turns => "turns",
        }
    }
}

/// The positions the server's loop needs for one window from an empty
/// model: it takes one only while its rows fit, the first at a one-id
/// prompt's first generated token, 1 + 1 + rows − 1.
const NEED: usize = VERIFY_ROWS + 1;
/// [`NEED`] as the least a slot of a split holds under the draft.
const NEED_FLOOR: NonZeroU64 = NonZeroU64::new(NEED as u64).expect("a window of at least one row");

/// `BLOOMERY_DRAFT` on stores of `ctx` positions, `unset` the seat's rule
/// for it: `None` drafts with the NextN layer; `Some` runs the plain path,
/// with why (the `load draft=off` record's). Unset, the rule's word and why
/// print as a `draft unset` record. Refused by name: `mtp` set with no room
/// for one window, and every word but `mtp` and `off`, as `generate_glm5next`
/// refuses it.
fn draft_of(
    levers: &bloomery_levers::Levers,
    ctx: usize,
    unset: GlmPick,
) -> Result<Option<String>, GateError> {
    match levers.draft() {
        Some("mtp") if ctx < NEED => Err(format!(
            "BLOOMERY_DRAFT=mtp needs --ctx {NEED} or more: token 0 comes out of the feed, and \
             a window's {VERIFY_ROWS} rows run after it (--ctx {ctx})"
        )
        .into()),
        Some("mtp") => Ok(None),
        Some("off") => Ok(Some("BLOOMERY_DRAFT=off".to_owned())),
        Some(other) => Err(format!(
            "BLOOMERY_DRAFT={other}: on a glm5next file mtp drafts the window (the file's NextN \
             layer); lookup and dspark are the V4.1 binaries'"
        )
        .into()),
        None => {
            Record::new(&record::DRAFT_UNSET_GLM)
                .w("draft", unset.word)
                .w("why", unset.why)
                .eprint();
            match unset.word {
                "mtp" => Ok(None),
                _ => Ok(Some(unset.why.to_string())),
            }
        }
    }
}

/// The seat's `--ctx` and what decided it, for its `ctx` line: every
/// context a slot's own, the plan counting every slot.
struct GlmCtx {
    ctx: usize,
    /// The resident sequences that split the total.
    slots: usize,
    /// `set` (`--ctx`), or what bounded the default: `card` (the largest
    /// context the plan takes, fewer than [`CTX`]), `base` ([`CTX`]: no
    /// step past it stays within the margin), `margin` (the largest that
    /// does), `fit` (the plan's largest context, the margin binding nothing
    /// below it).
    rule: &'static str,
    /// The file's trained context (`context_length`).
    trained: usize,
    /// The largest context the plan takes, capped to the trained context and
    /// [`ORACLE_POSITIONS`], and its stage-card expert bytes.
    fit: usize,
    fit_bytes: u64,
    /// The largest context within the plan's margin
    /// (`placement::ctx`'s guard).
    margin_ctx: usize,
    /// The plan's stage-card expert bytes at [`CTX`] and at `ctx`.
    base_bytes: u64,
    card_bytes: u64,
}

/// The seat's context (the module doc), every number a slot's own and every
/// plan counting `slots` sequences ([`plan_of`]): `set`, a slot's
/// share of the flag ([`split_ctx`]), as given — the plan the load runs
/// refuses it by name past [`ORACLE_POSITIONS`] and when the card cannot
/// hold every slot, so does this, with the plan's own words; unset, [`CTX`]
/// or the trained context capped to what the plan takes within its
/// [`MARGIN`] of stage-card expert bytes. The searches are planning-time
/// only: one plan a probe of the bisection, none past the cap. A card whose
/// plan stands nowhere, not even at one position a slot, refuses by name
/// here, as the load's own plan would.
fn ctx_of(
    inputs: &PlanInputs,
    machine: &Machine,
    levers: &PlanLevers,
    nextn: Option<&NextnInputs>,
    set: Option<usize>,
    slots: usize,
) -> Result<GlmCtx, GateError> {
    let card = |ctx: usize| -> Result<u64, GateError> {
        let plan = plan_of(inputs, machine, u64::try_from(ctx)?, levers, nextn, slots)?;
        Ok(plan
            .cards
            .first()
            .ok_or("a plan with no card")?
            .expert_bytes)
    };
    let trained = inputs.hp.n_ctx_train;
    let cap = trained.min(usize::try_from(ORACLE_POSITIONS)?);
    let fits = |c: usize| Ok(card(c).is_ok());
    if let Err(e) = card(1) {
        return Err(format!(
            "no context fits the card: the plan of {slots} slots at 1 position a slot is \
             refused ({e})"
        )
        .into());
    }
    let fit = model::placement::ctx::largest::<GateError>(1, cap, fits)?;
    let base_at = CTX.min(fit);
    let base_bytes = card(base_at)?;
    let margin_ctx =
        model::placement::ctx::within_margin(fit, base_at, base_bytes, CTX_STEP, MARGIN, &card)?;
    let (ctx, rule) = match set {
        Some(c) => (c, "set"),
        None if fit < CTX => (base_at, "card"),
        None if margin_ctx == CTX => (CTX, "base"),
        None if margin_ctx == fit => (fit, "fit"),
        None => (margin_ctx, "margin"),
    };
    Ok(GlmCtx {
        ctx,
        slots,
        rule,
        trained,
        fit,
        fit_bytes: card(fit)?,
        margin_ctx,
        base_bytes,
        card_bytes: card(ctx)?,
    })
}

impl GlmCtx {
    /// The `ctx` line on stderr, qwen38's shape with the trained context
    /// named: every context a slot's, `total` the slots'.
    fn print(&self) {
        eprintln!(
            "ctx rule={} ctx={} slots={} total={} trained={} fit={} fit_card_expert_bytes={} \
             margin_ctx={} base={CTX} base_card_expert_bytes={} card_expert_bytes={} \
             lost_bytes={} margin_bytes={MARGIN}",
            self.rule,
            self.ctx,
            self.slots,
            self.slots * self.ctx,
            self.trained,
            self.fit,
            self.fit_bytes,
            self.margin_ctx,
            self.base_bytes,
            self.card_bytes,
            self.base_bytes.saturating_sub(self.card_bytes)
        );
    }
}

/// The target's plan of `slots` resident sequences at `ctx` positions a slot
/// on the load `nextn` names: with the next-token layer
/// (`PlanInputs::plan_nextn_slots`, two KDA lanes) or without it
/// (`PlanInputs::plan_slots` at one lane) — the plans the load runs
/// ([`open_nextn_slots`], [`open_resident_slots`]).
fn plan_of<'a>(
    inputs: &'a PlanInputs,
    machine: &'a Machine,
    ctx: u64,
    levers: &PlanLevers,
    nextn: Option<&'a NextnInputs>,
    slots: usize,
) -> Result<Plan<'a>, GateError> {
    Ok(match nextn {
        None => inputs.plan_slots(machine, ctx, levers, KdaLanes::One, slots)?,
        Some(n) => {
            inputs
                .plan_nextn_slots(machine, ctx, levers, n, slots)?
                .plan
        }
    })
}

/// A placement as the seat plans it: the tier's prompt-batch bytes, the
/// machine, the context rule over it, and the experts its tier cards hold
/// in the plan at the rule's context (0 with no tier card).
struct Placed {
    place: Place,
    tier_batch: Option<TierBatchBytes>,
    machine: Machine,
    rule: GlmCtx,
    tier_experts: u64,
}

impl Placed {
    /// `place` planned for `slots` sequences on the load `nextn` names, its
    /// context `set` or the rule's ([`ctx_of`]); every refusal of the plan's
    /// by name.
    fn of(
        place: Place,
        inputs: &PlanInputs,
        levers: &PlanLevers,
        nextn: Option<&NextnInputs>,
        set: Option<usize>,
        slots: usize,
    ) -> Result<Placed, GateError> {
        let tier_batch = glm_place::tier_batch(place, &inputs.hp);
        let machine = place.machine(None, tier_batch)?(inputs.model.layers);
        let rule = ctx_of(inputs, &machine, levers, nextn, set, slots)?;
        let ctx = u64::try_from(rule.ctx)?;
        let tier_experts =
            glm_place::tier_experts(place, inputs, ctx, levers, nextn, KdaLanes::One, slots)?;
        Ok(Placed {
            place,
            tier_batch,
            machine,
            rule,
            tier_experts,
        })
    }
}

/// `BLOOMERY_RESIDENCY` as this seat takes it before the plan.
enum ResidencyLever {
    /// Set: its parse and word.
    Set(Residency, &'static str),
    /// Unset: the seat's rule before the plan, which the plan then decides
    /// ([`bloomery_levers::glm_residency_at_plan`]).
    Unset(GlmPick),
}

/// `BLOOMERY_RESIDENCY` before the plan: set, as given, its `residency
/// lever` record printed, `mid-…` refused by name beside the steps feed (each
/// prompt id would end a pass the rule counts; the body refuses it at the
/// feed, this before the load); unset, `unset`.
fn residency_of(
    levers: &bloomery_levers::Levers,
    prefill: PrefillMode,
    unset: GlmPick,
) -> Result<ResidencyLever, GateError> {
    let Some(word) = levers.residency() else {
        return Ok(ResidencyLever::Unset(unset));
    };
    let r = Residency::parse(word)?;
    record::residency_lever(ResidencyPick {
        word,
        why: ResidencyWhy::Set,
    })
    .eprint();
    if r != Residency::Off && prefill == PrefillMode::Steps {
        return Err(format!(
            "BLOOMERY_RESIDENCY={word} is refused beside --prefill steps (each prompt id would \
             end a pass the rule counts)"
        )
        .into());
    }
    Ok(ResidencyLever::Set(r, word))
}

/// The residency the load of `plan` runs, `beside` the bytes the load hosts
/// outside the plan (the NextN layer's experts): a set word as given, refused
/// by name when the plan's card slots leave it no room; unset, the rule's
/// word where the plan has room and the host the churn pool, else `off`
/// with why, its `residency unset` record printed. Then, under `mid`, the
/// `residency host` record ([`residency_set`]).
fn residency_at(
    plan: &Plan<'_>,
    lever: ResidencyLever,
    beside: u64,
) -> Result<Residency, GateError> {
    let (r, word) = match lever {
        ResidencyLever::Set(r, word) => {
            residency_room(plan, r, word)?;
            (r, word)
        }
        ResidencyLever::Unset(pick) => {
            // The load refuses a host set past `MemAvailable` before any
            // upload; the default leaves the pool out instead.
            let mem_left = match pick.word {
                "off" => 0,
                _ => i128::from(host_available()?) - i128::from(HostNeed::of(plan, beside).bytes()),
            };
            let pick = glm_residency_at_plan(
                pick,
                plan.n_l.iter().copied(),
                |pinned| ChurnPool::of(plan, GLM_CARD, pinned).map(|pool| pool.bytes),
                plan.host.headroom_bytes - i128::from(beside),
                mem_left,
            )
            .map_err(|e| format!("BLOOMERY_RESIDENCY unset: the churn pool: {e}"))?;
            Record::new(&record::RESIDENCY_UNSET_GLM)
                .w("residency", pick.word)
                .w("why", pick.why)
                .eprint();
            (Residency::parse(pick.word)?, pick.word)
        }
    };
    residency_set(plan, GLM_CARD, r, word, beside, Record::eprint)
}

struct Args {
    host: String,
    port: u16,
    /// `--place`; `None` takes the common rule's choice ([`Place::choose`]).
    place: Option<Place>,
    /// `--ctx`; `None` takes the rule's default.
    ctx: Option<usize>,
    alias: Option<String>,
    /// `--cache-ram` in bytes; `None` takes the default.
    cache_ram: Option<u64>,
    /// `--slot-save-path`: the directory the slot actions answer from;
    /// `None` refuses every one, as llama-server does.
    slot_save_path: Option<PathBuf>,
    /// `--chat-template-file`, replacing the file's own template.
    template_file: Option<PathBuf>,
    prefill: PrefillMode,
    /// `--plan`: the records before the load, then exit.
    plan_only: bool,
    /// `--parallel`: the resident sequences the seat serves; `None` takes
    /// one slot beside a set `--ctx`, else the default 2
    /// ([`model::placement::ctx::slots_of`]).
    parallel: Option<usize>,
    queue_depth: Option<usize>,
    /// `--api-key`/`--api-key-file`: the keys every request is checked
    /// against ([`serve::flag::ApiKeys`]).
    api_keys: serve::flag::ApiKeys,
}

fn parse_args(args: &[String]) -> Result<Args, GateError> {
    let mut a = Args {
        host: "127.0.0.1".to_owned(),
        port: 8080,
        place: None,
        ctx: None,
        alias: None,
        cache_ram: None,
        slot_save_path: None,
        template_file: None,
        prefill: PrefillMode::Batch,
        plan_only: false,
        parallel: None,
        queue_depth: None,
        api_keys: serve::flag::ApiKeys::default(),
    };
    let mut it = args.iter().map(|s| s.as_str());
    while let Some(flag) = it.next() {
        if flag == "--help" || flag == "-h" {
            return Err(USAGE.into());
        }
        if flag == "--plan" {
            a.plan_only = true;
            continue;
        }
        let v = it
            .next()
            .ok_or_else(|| format!("{flag} needs a value, or is unknown: {USAGE}"))?;
        match flag {
            "--host" => a.host = v.to_owned(),
            "--port" => a.port = number(flag, v)?,
            "--place" => a.place = Some(glm_place::parse(v)?),
            f if serve::flag::CTX.contains(&f) => a.ctx = Some(number(flag, v)?),
            "--alias" => a.alias = Some(v.to_owned()),
            "--cache-ram" => a.cache_ram = Some(CacheRam::parse_mib(flag, v)?),
            "--slot-save-path" => a.slot_save_path = Some(PathBuf::from(v)),
            "--parallel" | "-np" => a.parallel = Some(number(flag, v)?),
            "--queue-depth" => a.queue_depth = Some(number(flag, v)?),
            f if serve::flag::KEYS.contains(&f) => a.api_keys.add(f, v)?,
            "--park-ram" => {
                return Err(format!(
                    "--park-ram {v}: it holds the states of slots that take the model in turns; \
                     this seat's slots are resident sequences, which park nothing — the plan \
                     counts their state"
                )
                .into());
            }
            "--chat-template-file" => a.template_file = Some(PathBuf::from(v)),
            "--prefill" => {
                a.prefill = PrefillMode::from_name(v)
                    .ok_or_else(|| format!("--prefill is batch or steps, not {v}"))?;
            }
            other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
        }
    }
    if a.ctx == Some(0) {
        return Err("--ctx 0: the stores hold no position".into());
    }
    if a.parallel == Some(0) {
        return Err("--parallel 0: the server serves no slot".into());
    }
    Ok(a)
}

/// Loads the model and serves until the listener or the engine fails; `Ok`
/// carries why the server ended.
pub fn run(args: &[String]) -> Result<ServeError, GateError> {
    let levers = bloomery_levers::at_main(ACTS_ON)?;
    let width = WidthMode::of(levers.mtp_width())?;
    record::at_main(WHAT, record::BLOOMERY_SERVE_GLM);
    let a = parse_args(args)?;
    // The census the placement resolves against, read once: `--place` as
    // given, or unset the common rule's choice on the same reading.
    let census = gpu_census::census()?;
    let plan_levers = PlanLevers::from_levers(&levers)?;
    let path = ref_model_path()?;
    let vocab = Arc::new(Vocab::new(Tokenizer::from_gguf(&path)?)?);
    let inv = gguf::inventory_of(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let template = match &a.template_file {
        Some(file) => std::fs::read_to_string(file)
            .map_err(|e| format!("--chat-template-file {}: {e}", file.display()))?,
        None => inv
            .value("tokenizer.chat_template")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("{}: no tokenizer.chat_template", path.display()))?
            .to_owned(),
    };
    let name = inv
        .value("general.name")
        .and_then(|v| v.as_str())
        .unwrap_or("glm-5.3-flash")
        .to_owned();
    drop(inv);

    // The plan record, and `/props` from the same plan the load runs by:
    // drafting, the target's plan beside the NextN layer, whose card bytes
    // and arena are the card's `draft` class and whose host experts the
    // load's host set holds beside the plan's.
    let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let inputs = PlanInputs::read(&split)?;
    // A slot's share of `--ctx`, refused by name under the body's least a
    // slot before anything is planned: one position, or the MTP draft's
    // window under `BLOOMERY_DRAFT=mtp` (unset, the rule below drafts only
    // where a window fits, and never refuses). The slot count itself: a set
    // `--ctx` with no `--parallel` is one request's context — one slot at
    // the whole of it (`placement::ctx::slots_of`).
    let (slots, from) = model::placement::ctx::slots_of(a.parallel, a.ctx.is_some(), 2)?;
    if from == "ctx" {
        eprintln!(
            "--ctx-size {} is one request's context; add --parallel N to serve N requests at \
             once (they split it)",
            a.ctx.unwrap_or_default()
        );
    }
    let (floor, why) = match levers.draft() {
        Some("mtp") => (
            NEED_FLOOR,
            format!(
                "BLOOMERY_DRAFT=mtp: token 0 comes out of the feed, and a window's {VERIFY_ROWS} \
                 rows run after it"
            ),
        ),
        _ => (
            NonZeroU64::MIN,
            "a slot holds at least one position".to_owned(),
        ),
    };
    // The split is `placement::slots::split_ctx`'s, ⌊total / slots⌋
    // (llama-server's `-np N` without `-kvu`); its refusal in the flags'
    // words.
    let set = a
        .ctx
        .map(|total| {
            let share = split_ctx(total as u64, slots as u64, floor).map_err(|e| match e {
                SplitError::NoSlots { .. } => {
                    format!("--ctx {total}: --parallel 0 splits it among no slot ({why})")
                }
                SplitError::BelowFloor { split, floor, .. } => format!(
                    "--ctx {total}: --parallel {slots} splits it to {split} positions a slot, \
                     under the {floor} a slot needs ({why})"
                ),
            })?;
            usize::try_from(share).map_err(|_| format!("a slot's {share} positions"))
        })
        .transpose()?;
    // The unset rule's window bound reads a slot's floor context — a default
    // is never under it while the card holds it — and the context the rule
    // chooses is asked again below.
    let at_ctx = set.unwrap_or(CTX);
    // Whether the placement serves: an unset flag does (the common rule
    // never picks `gate`), and a set one does unless it is `gate`.
    let unset = glm_unset(GlmAt {
        serving_place: a.place.is_none_or(|p| p != Place::Gate),
        nextn_layers: inputs.hp.n_layer.saturating_sub(inputs.hp.n_trunk),
        need: NEED,
        ctx: at_ctx,
        prefill_steps: a.prefill == PrefillMode::Steps,
    });
    let lever = residency_of(&levers, a.prefill, unset.residency)?;
    let draft_off = draft_of(&levers, at_ctx, unset.draft)?;
    let model = model_props(&split, &inputs.model);
    let nextn = match draft_off {
        None => Some(NextnInputs::read(&inputs)?),
        Some(_) => None,
    };
    let plan_place = |p: Place| Placed::of(p, &inputs, &plan_levers, nextn.as_ref(), set, slots);
    // The placement the seat runs by (`Place::choose`, by the GLM family's
    // `TIER_RULE`): the plan of the offer names its tier count, and an unset
    // flag never refuses a load that `--place a` serves — a refused plan of
    // the offer runs `a`, the refusal named on stderr. Planning the chosen
    // placement again is milliseconds.
    let chosen = Place::choose(a.place, &census, glm_place::TIER_RULE, |p| {
        Ok(plan_place(p)?.tier_experts)
    })?;
    chosen.record().eprint();
    let drafted = plan_place(chosen.place)?;
    // The unset draft's yield to the context (qwen38's module doc, the one
    // rule through `bloomery_levers::DraftYield`): the chosen placement's
    // search is the drafted load's own — its fit is the positions a slot
    // gets with the draft — and under [`CTX`] the plain search, the
    // draft-off load's own, decides: it holds more, and the NextN layer's
    // card bytes leave a slot under the base the plain rule aims for, so
    // the draft goes off, the plain placement replaces this one and one
    // `draft yield` record names both; a serving cap that binds the two
    // fits together keeps the draft. A set `--ctx` or `BLOOMERY_DRAFT`
    // never asks, and the placement's own choice (`chosen`, planned with
    // the NextN layer) stands: a card small enough to yield holds no tier
    // either way.
    let (placed, draft_off) = if draft_off.is_none()
        && a.ctx.is_none()
        && levers.draft().is_none()
        && drafted.rule.fit < CTX
    {
        let plain = Placed::of(chosen.place, &inputs, &plan_levers, None, set, slots)?;
        let with = inputs.plan_nextn_slots(
            &drafted.machine,
            u64::try_from(drafted.rule.ctx)?,
            &plan_levers,
            nextn.as_ref().expect("a drafted load read the NextN layer"),
            slots,
        )?;
        let bytes = with.nextn_card_bytes() + with.arena_bytes;
        match bloomery_levers::DraftYield::of(
            drafted.rule.ctx,
            plain.rule.ctx,
            CTX.min(plain.rule.fit),
            bytes,
        ) {
            Some(y) => {
                record::draft_yield(&y).eprint();
                (plain, Some(y.to_string()))
            }
            None => (drafted, draft_off),
        }
    } else {
        (drafted, draft_off)
    };
    // A width lever set on a server that drafts nothing is refused, with
    // why — after the yield said its last word.
    if let (Some(why), Some(_)) = (&draft_off, levers.mtp_width()) {
        return Err(format!(
            "BLOOMERY_MTP_WIDTH picks the width a drafted window verifies; the server drafts \
             nothing ({why})"
        )
        .into());
    }
    let nextn = if draft_off.is_none() { nextn } else { None };
    let Placed {
        place,
        tier_batch,
        machine,
        rule,
        ..
    } = placed;
    rule.print();
    if draft_off.is_none() && rule.ctx < NEED {
        return Err(format!(
            "the --ctx {} leaves no positions for the MTP draft's window (token 0 comes out of \
             the feed, and a window's {VERIFY_ROWS} rows run after it)",
            rule.ctx
        )
        .into());
    }
    let ctx = u64::try_from(rule.ctx)?;
    // The plans count every slot in their KV terms.
    let (plan, beside, draft_bytes) = match &nextn {
        None => (
            inputs.plan_slots(&machine, ctx, &plan_levers, KdaLanes::One, slots)?,
            0,
            0,
        ),
        Some(n) => {
            let with = inputs.plan_nextn_slots(&machine, ctx, &plan_levers, n, slots)?;
            let beside = with.host_runs()?.1;
            let bytes = with.nextn_card_bytes() + with.arena_bytes;
            (with.plan, beside, bytes)
        }
    };
    record::plan(place.name(), &machine, &plan).eprint();
    let residency = residency_at(&plan, lever, beside)?;
    let pool = match residency {
        Residency::Mid { pinned, .. } => {
            ChurnPool::of(&plan, GLM_CARD, pinned)
                .map_err(|e| format!("the churn pool: {e}"))?
                .bytes
        }
        Residency::Off => 0,
    };
    let cache = CacheRam::of(a.cache_ram, HostNeed::of(&plan, beside).bytes(), pool)?;
    eprintln!("{}", cache.line());
    eprintln!(
        "parallel rule=slots slots={slots} slot_ctx={} total={} from={from} pass={}",
        rule.ctx,
        slots * rule.ctx,
        // The round's shape the open below runs (the module doc).
        Rounds::of(draft_off.is_none()).word()
    );
    if a.plan_only {
        // The records before the load are out; nothing was opened on a card.
        std::process::exit(0);
    }
    let gpu = machine
        .all_cards()
        .map(|c| nvidia_smi_index(&c.name, c.device).map(|i| format!("GPU{i}")))
        .collect::<Result<Vec<String>, String>>()
        .and_then(|g| placement_props(&plan, &g));
    if let Err(e) = &gpu {
        eprintln!("{WHAT}: /props leaves the placement out: {e}");
    }
    let props = EngineProps {
        model: Some(model),
        placement: gpu.ok(),
        ..EngineProps::default()
    };

    let open = SeatArgs {
        place,
        tier_batch,
        ctx: rule.ctx,
        slots,
        cfg: GlmCfg {
            place: plan_levers,
            host: levers.host(),
            prefill: a.prefill,
            group: PROMPT_GROUP,
        },
        pin_main: levers.pin_main(),
        path: path.clone(),
        draft_off,
        draft_bytes,
        residency,
        stats: levers.step_stats(),
        width,
    };
    let engine = SeatEngine::spawn(
        move || Glm::open(open),
        rule.ctx,
        vocab,
        machine.cards[0].name.clone(),
        props,
        cache.ram,
    )?;

    let config = ServerConfig {
        model_alias: a.alias.unwrap_or(name),
        model_path: path.display().to_string(),
        chat_template: template,
        sampler: Some(sampler_factory()),
        fatal_linger: FATAL_LINGER,
        slot_save_path: a.slot_save_path,
        api_keys: a.api_keys,
    };
    // The seat's resident slots are the server's, one sequence each: the
    // server selects and steps them together (the engine declares its
    // per-slot draft), no turns and no park.
    let config_slots = SlotConfig {
        parallel: slots,
        queue_depth: a.queue_depth,
        ..SlotConfig::default()
    };
    let server = Server::bind_with(
        (a.host.as_str(), a.port),
        Box::new(engine),
        config,
        config_slots,
    )?;
    Record::new(&record::LISTENING_GLM)
        .w("place", place.name())
        .u("ctx", rule.ctx)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}

/// What the engine thread opens the seat with.
struct SeatArgs {
    place: Place,
    /// The expert tier's prompt-batch bytes the plan reserves under a
    /// placement with a tier card.
    tier_batch: Option<TierBatchBytes>,
    /// A slot's context: the stores' rows one sequence serves.
    ctx: usize,
    /// The resident sequences the plan counts and the load makes
    /// ([`Session::add_slots`]).
    slots: usize,
    cfg: GlmCfg,
    pin_main: bool,
    path: PathBuf,
    /// Drafting nothing, why; `None` drafts with the NextN layer.
    draft_off: Option<String>,
    /// The plan's bytes for the NextN layer (its card terms and arena), the
    /// card's `draft` class in `/props`.
    draft_bytes: u64,
    /// The residency the load runs.
    residency: Residency,
    /// Whether the seat counts its rounds of several slots
    /// (`BLOOMERY_STEP_STATS`, the binary's `main` parsed).
    stats: bool,
    /// `BLOOMERY_MTP_WIDTH`: the width a drafted window verifies, the
    /// chooser's (`cost`) or the draft's own (`fixed`).
    width: WidthMode,
}

/// The GLM session on the engine thread: the session over the model, the
/// draft it drives when one runs (its windows of two rows through the
/// runtime's speculative loop), and the positions its stores were sized
/// for.
struct Glm {
    s: Session<Body>,
    drafted: SlotDrafts<Body, VERIFY_ROWS>,
    ctx: usize,
    /// The plan's bytes for the NextN layer, for `/props`' `draft` class.
    draft_bytes: u64,
    /// The target file, whose NextN layer `/props`' `draft` names.
    path: PathBuf,
    /// The load runs the residency machine: each call prints its
    /// boundaries' `residency pass` records.
    residency: bool,
    /// The shape a round of several slots runs in ([`Rounds::of`], decided
    /// at the open): a plain load's step rounds one pass
    /// ([`Glm::step_slots`]), a NextN load's drafted rounds one pass when the
    /// body's pass holds two windows ([`Glm::pass_slots`]); every other
    /// round the fallback loop — a NextN load's step rounds among them, its
    /// body refusing a pass of plain steps by name (a pass of plain steps
    /// tells no slot's draft what it ran).
    rounds: Rounds,
    /// [`Seat::step_stats`]: the `BLOOMERY_STEP_STATS` the binary parsed.
    stats: bool,
}

impl Glm {
    /// The session by `a.place` (the `plan` was printed before the engine
    /// thread started; the `load` and `capture` lines go to stderr as
    /// `generate_glm5next` prints them, the draft's `load draft=…` line after
    /// the `load` line), on the calling thread, pinned to the dispatcher's
    /// cpu slot when asked: drafting, the NextN load under the residency
    /// (`open_nextn_slots`) and the window over it, its verify captured (a
    /// `capture` line of its nodes); else the one-lane load under it
    /// (`open_resident_slots`). Either plans and serves the seat's slots.
    fn open(a: SeatArgs) -> Result<Glm, GateError> {
        let pinned = a.pin_main && threads::pool().pin_caller();
        let t = Instant::now();
        let file = Split::open(&a.path).map_err(|e| format!("open {}: {e}", a.path.display()))?;
        let mut log = Log {
            top_k: 0,
            place: a.place.name(),
            placement: a.place,
            prefill: a.cfg.prefill,
            group: a.cfg.group,
            ctx: a.ctx,
            pin_main: a.pin_main,
            pinned,
            t,
            draft_off: a.draft_off.clone(),
            draft_bytes: a.draft_bytes,
        };
        let prefill = a.cfg.prefill;
        let args = app::OpenArgs {
            place: a.place.name(),
            machine: a.place.machine(None, a.tier_batch)?,
            ctx: a.ctx,
            mode: StepMode::Graph,
            cfg: a.cfg,
        };
        let planned = || format!("{WHAT}: the open planned nothing");
        // Every load serves the seat's slots, the plan counting each.
        let mut s = match &a.draft_off {
            None => open_nextn_slots(file, args, a.residency, a.slots, &mut log)?,
            Some(_) => open_resident_slots(file, args, a.residency, a.slots, &mut log)?,
        }
        .ok_or_else(planned)?;
        let residency = a.residency != Residency::Off;
        if residency {
            // A request's passes are not known at load: the log grows as it
            // must, and every call takes it.
            s.model_mut().body_parts(WHAT)?.2.log_residency(0);
        }
        // The resident sequences, parked empty: the live slot keeps the
        // chains already captured, each parked slot capturing its own on its
        // first use.
        s.add_slots(a.slots)?;
        let drafted = match a.draft_off {
            Some(_) => SlotDrafts::none(),
            None => SlotDrafts::open(&mut s, a.slots, a.width, &mut PairCapture, |m| {
                MtpDraft::open(m, prefill, StepMode::Eager)
            })?,
        };
        Ok(Glm {
            s,
            drafted,
            ctx: a.ctx,
            draft_bytes: a.draft_bytes,
            path: a.path,
            residency,
            rounds: Rounds::of(a.draft_off.is_none()),
            stats: a.stats,
        })
    }

    /// The `residency pass` records of the boundaries the last call made, on
    /// stderr; nothing without the residency.
    fn print_passes(&mut self) -> Result<(), GateError> {
        if !self.residency {
            return Ok(());
        }
        for (kind, r) in self
            .s
            .model_mut()
            .body_parts(WHAT)?
            .2
            .take_residency_passes()
        {
            record::residency_pass_of(kind, &r).eprint();
        }
        Ok(())
    }

    /// Before a round of one pass ([`Rounds::Windows`]), each row's slot
    /// whose draft skips turned off ([`SlotDrafts::turn_off`], for
    /// [`SKIP_OFF_WHY`]): its window would be the one plain row, which the
    /// NextN pass of several slots refuses by name (each slot's rows there
    /// are its verify's two), and the round runs an off slot alone, its
    /// plain step. A skipping draft proposes nothing until the slot's next
    /// reset — the server resets a slot before every request that keeps
    /// nothing — so the slot's ids and counts are those of the fallback
    /// loop, whose pass steps a skipping slot plainly.
    fn off_skipping(&mut self, rows: &[SlotPassRow]) -> Result<(), GateError> {
        for r in rows {
            let skips = self
                .drafted
                .specs_mut()
                .get(r.slot)
                .and_then(Option::as_ref)
                .is_some_and(|spec| spec.draft().draft().skipping());
            if skips && !self.drafted.is_off(r.slot) {
                self.drafted.turn_off(r.slot, SKIP_OFF_WHY)?;
            }
        }
        Ok(())
    }
}

/// Why a slot whose draft skips is off in a round of one pass
/// ([`Glm::off_skipping`]); its next call prints it as an `mtp prompt`
/// record.
const SKIP_OFF_WHY: &str = "the draft skips until the slot's next reset, and a NextN pass of \
                            several slots takes each slot's whole verify";

/// A GLM sequence state as the server's prompt cache holds it ([`GlmSeq`]),
/// with the draft's side of it when the seat drafts. The cache ranks it by
/// the body's rule: every position it holds, or the point it carries at or
/// below the shared prefix.
struct SavedGlm {
    state: GlmSeq,
    draft: Option<ParkedDraft<GlmArena>>,
}

impl Saved for SavedGlm {
    fn n_tokens(&self) -> usize {
        self.state.positions() as usize
    }

    fn n_bytes(&self) -> u64 {
        self.state.bytes() as u64
    }

    fn keepable(&self, n: usize) -> usize {
        self.state.keep_point(u32::try_from(n).unwrap_or(u32::MAX)) as usize
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The verify pass's capture: its nodes, one line.
struct PairCapture;

impl RowsLog for PairCapture {
    fn capture_rows(&mut self, _rows: usize, nodes: usize) -> Result<(), SessionError> {
        Record::new(&record::CAPTURE_PAIR)
            .u("pair_graph_nodes", nodes)
            .eprint();
        Ok(())
    }
}

/// What the open prints: the plan's top_k, the load, the draft's load, the
/// capture.
struct Log {
    /// The file's indexer top-k, which the plan read.
    top_k: usize,
    place: &'static str,
    /// The placement, whose cards the `load` record names.
    placement: Place,
    prefill: PrefillMode,
    /// The batches a prompt group runs (`GlmCfg::group`).
    group: usize,
    ctx: usize,
    pin_main: bool,
    pinned: bool,
    t: Instant,
    /// Drafting nothing, why.
    draft_off: Option<String>,
    /// The plan's bytes for the NextN layer.
    draft_bytes: u64,
}

impl OpenLog<Body> for Log {
    /// The plan was printed before the engine thread started.
    fn plan(
        &mut self,
        _place: &'static str,
        inputs: &PlanInputs,
        _machine: &Machine,
        _plan: &model::placement::Plan<'_>,
    ) -> Result<bool, app::SessionError> {
        self.top_k = inputs.hp.indexer.top_k;
        Ok(true)
    }

    fn load(&mut self, m: &Glm5nextModel) -> Result<(), app::SessionError> {
        let b = m.body(WHAT)?;
        let r = Record::new(&record::LOAD_GENERATOR)
            .u("resident_bytes", m.resident_bytes())
            .u("ctx", self.ctx)
            .u("layers", b.kinds().len())
            .u("top_k", self.top_k)
            .w("shadow", "none")
            .u("shadow_bytes", 0)
            .u("unified_addressing", 0);
        with_cards(r, self.placement, m.gpu(), b.hybrid().tiers())
            .map_err(app::SessionError::Caller)?
            .w("prefill", self.prefill.name())
            .u("group", self.group)
            .w("mode", mode_name(StepMode::Graph))
            .w("place", self.place)
            .w("pin_main", if self.pin_main { "on" } else { "off" })
            .w("pinned", self.pinned)
            .f("load_s", self.t.elapsed().as_secs_f64())
            .eprint();
        if let Some(h) = b.hybrid().residency() {
            for r in record::host_residency(h) {
                r.eprint();
            }
        }
        match &self.draft_off {
            Some(why) => Record::new(&record::LOAD_DRAFT_OFF_GLM)
                .w("why", why)
                .eprint(),
            None => {
                let d = b.nextn().ok_or_else(|| {
                    SessionError::Refused(format!("{WHAT}: the NextN load holds no NextN layer"))
                })?;
                Record::new(&record::LOAD_DRAFT_GLM)
                    .u("layer", d.index())
                    .u("resident", d.resident_bytes())
                    .u("arena", d.arena_bytes())
                    .w("head", if d.head_rows() { "rows" } else { "full" })
                    .u("plan_bytes", self.draft_bytes)
                    .f("load_s", self.t.elapsed().as_secs_f64())
                    .eprint();
            }
        }
        Ok(())
    }

    fn capture(&mut self, nodes: usize) -> Result<(), app::SessionError> {
        Record::new(&record::CAPTURE)
            .u("graph_nodes", nodes)
            .eprint();
        Ok(())
    }

    fn prompt_buffers(&mut self, _m: &Glm5nextModel) -> Result<(), app::SessionError> {
        Ok(())
    }
}

impl Seat for Glm {
    fn pos(&self) -> usize {
        self.s.pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    /// The resident sequences the load made ([`Session::slots`]).
    fn slots(&self) -> usize {
        self.s.slots()
    }

    /// The session's slot ([`Session::select_slot`]): the slot's draft
    /// ([`Glm::drafted`], one a slot) needs nothing moved.
    fn select(&mut self, slot: usize) -> Result<(), GateError> {
        if self.s.selected() == slot {
            return Ok(());
        }
        Ok(self.s.select_slot(slot)?)
    }

    /// The draft's state is per slot ([`Glm::select`]): a slot's drafted
    /// passes are the passes it would run alone.
    fn slot_drafts(&self) -> bool {
        self.drafted.drafts()
    }

    /// The lever the binary parsed ([`Glm::stats`]).
    fn step_stats(&self) -> bool {
        self.stats
    }

    /// One round of several slots ([`super::rounds::step_round`]): one pass
    /// of the busy rows on a plain load ([`Rounds::Steps`], through
    /// `serve_seats::rounds::step_rows_one_pass` — cut at the body's
    /// [`SlotRows::MAX_ROWS`], the rows' answers and lent logits rows from
    /// the pass's own per-row heads, slot 0 left selected), then the call's
    /// `residency pass` records, else the seat's fallback — a select and a
    /// step a row — the NextN load's step round, whose body refuses a pass
    /// of plain steps by name (each of its slots is drafted, and such a pass
    /// tells no slot's draft what it ran). A refusal on either path is the
    /// server's to die on: the open decides once, so a load that cannot run
    /// one pass never tries it at run time.
    fn step_slots(&mut self, rows: &mut [SlotStep]) -> Result<(), String> {
        let one_pass = self.rounds == Rounds::Steps;
        super::rounds::step_round(
            self,
            one_pass,
            rows,
            |g, rows| super::rounds::step_rows_one_pass(&mut g.s, rows),
            Glm::print_passes,
        )
    }

    /// One round of several slots' drafted passes
    /// ([`super::rounds::pass_round`]): one pass of the busy slots' windows
    /// while the open decided so ([`Rounds::Windows`], through
    /// `serve_seats::rounds::pass_rows_one_pass` — each slot's window over
    /// its own draft at the depth a one-slot pass proposes, the windows laid
    /// in slot order and cut between them at the body's
    /// [`SlotRows::MAX_ROWS`], slot 0 left selected), each slot whose draft
    /// skips turned off first ([`Glm::off_skipping`]) and stepped alone,
    /// then the call's `residency pass` records; else the fallback (a select
    /// and a pass a row, which prints its own `slots round` record). A
    /// refusal on either path is the server's to die on, never a fallback:
    /// the open decides once.
    fn pass_slots(&mut self, rows: &mut [SlotPassRow]) -> Result<(), String> {
        let one_pass = self.rounds == Rounds::Windows;
        super::rounds::pass_round(
            self,
            one_pass,
            rows,
            |g, rows| {
                g.off_skipping(rows)?;
                super::rounds::pass_rows_one_pass(
                    &mut g.s,
                    rows,
                    &mut g.drafted,
                    <Body as MtpBody>::WIDTH,
                )
            },
            Glm::print_passes,
        )
    }

    /// The body's feed ([`bloomery_gpu_glm5next::feed`]): the batched prompt
    /// call or one step a position, as `--prefill` says, each call taking the
    /// checkpoints its marks name; under the draft the draft's own prompt
    /// call, its store walked over the prompt's units.
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        let sel = self.s.selected();
        let next = self.drafted.prefill(&mut self.s, sel, ids)?;
        self.print_passes()?;
        Ok(next)
    }

    /// One step; under the draft the rows it left waiting walked first
    /// (`MtpDraft::before_step`: a request that continues the held sequence
    /// joins it here when its prompt call is empty).
    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        let sel = self.s.selected();
        let next = self.drafted.step(&mut self.s, sel, last)?;
        self.print_passes()?;
        Ok(next)
    }

    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        Ok(self.s.model().logits_into(row)?)
    }

    /// One step and the target's row of it, read before the step is told to
    /// the draft ([`SlotDrafts::step_with_row`]).
    fn step_row(&mut self, last: u32, row: &mut [f32]) -> Result<u32, GateError> {
        let sel = self.s.selected();
        let next = self.drafted.step_with_row(&mut self.s, sel, last, row)?;
        self.print_passes()?;
        Ok(next)
    }

    /// One pass from `last`: under the draft the window of two rows, its
    /// kept tokens and counts; without it one step.
    fn pass(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, GateError> {
        let sel = self.s.selected();
        let d = self.drafted.pass(&mut self.s, sel, last, out)?;
        self.print_passes()?;
        Ok(d)
    }

    /// The most positions one pass runs: the window's two rows, or one step
    /// without the draft.
    fn pass_rows(&self) -> usize {
        self.drafted.pass_rows(self.s.selected())
    }

    /// One sampled pass from `last` ([`SlotDrafts::pass_sampled`]): under
    /// the draft the window of two rows, each row's id the sampler's draw.
    fn pass_sampled(
        &mut self,
        last: u32,
        history: &mut Vec<u32>,
        sampler: &mut serve::Sampler,
        out: &mut Vec<u32>,
    ) -> Result<Drafted, GateError> {
        let sel = self.s.selected();
        let d = self
            .drafted
            .pass_sampled(&mut self.s, sel, last, history, sampler, out)?;
        self.print_passes()?;
        Ok(d)
    }

    /// The sampled pass drafts while the MTP draft runs.
    fn drafts_sampled(&self) -> bool {
        self.drafted.drafts()
    }

    /// The session's reset, then the draft started over: the residency stays
    /// where use has taken it (only [`Seat::residency_reset`] moves it back).
    fn reset(&mut self) -> Result<(), GateError> {
        let sel = self.s.selected();
        self.drafted.reset(&mut self.s, sel)
    }

    /// [`Session::residency_reset`], its `residency reset` record on stderr;
    /// `None` without the residency (the server's 501).
    fn residency_reset(&mut self) -> Result<Option<ResidencyReset>, GateError> {
        let Some(r) = self.s.residency_reset()? else {
            return Ok(None);
        };
        record::residency_reset(&r).eprint();
        let n = |v: usize| v as u64;
        Ok(Some(ResidencyReset {
            cancelled: n(r.cancelled),
            copies: n(r.copies),
            diff: n(r.diff),
            dropped_bytes: r.dropped_bytes,
        }))
    }

    /// `/props`' `engine.draft` under the MTP draft: its kind, the target
    /// file whose NextN layer it is, its width, and the plan's bytes for it
    /// as the card's `draft` class.
    fn props(&self, mut p: EngineProps) -> EngineProps {
        if !self.drafted.drafts() {
            return p;
        }
        if let Some(place) = p.placement.as_mut() {
            for d in place.devices.iter_mut() {
                if d.device.starts_with("GPU") {
                    d.class_bytes.insert("draft".to_owned(), self.draft_bytes);
                }
            }
        }
        p.draft = Some(DraftProps {
            model: self.path.file_name().map_or_else(
                || self.path.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            ),
            n_max: Some(<Body as MtpBody>::WIDTH as u64),
            kind: Some("mtp".to_owned()),
            path: Some(self.path.display().to_string()),
            device: None,
        });
        p
    }

    /// The body's rule ([`Body::kept`]): every fed position, the empty model,
    /// or the checkpoint at or below the cut, with the rule that kept less.
    /// A query past u32 positions is the query for every position.
    fn keep(&self, n: usize) -> (usize, Option<String>) {
        let asked = u32::try_from(n).unwrap_or(u32::MAX);
        let k = self.s.kept(asked);
        let why = match k.why {
            Why::Current | Why::Empty => None,
            _ => Some(k.to_string()),
        };
        (k.at as usize, why)
    }

    /// The checkpoint the cut keeps, taken back by the body's own rollback;
    /// any other position is refused by name, the body's own message.
    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        Ok(self.s.model_mut().rollback(pos)?)
    }

    /// Nowhere: this seat marks no user start, and a cut keeps the
    /// checkpoints the body's own rule takes — a call cut at a mark would add
    /// one the checkpoint spacing did not choose.
    fn splits(&self, _: usize, _: usize, _: &[usize]) -> Vec<usize> {
        Vec::new()
    }

    /// The sequence state ([`seq_save`]) and, under the draft, its side of
    /// it ([`SlotDrafts::park`]), as the prompt cache holds them.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
        Ok(Arc::new(SavedGlm {
            state: seq_save(self.s.model_mut())?,
            draft: self.drafted.park(self.s.selected())?,
        }))
    }

    /// The state put back ([`seq_resume`]) after the session's reset, then
    /// the draft's side of it ([`SlotDrafts::unpark`]): the sequence's next
    /// call runs as it would have with no switch between, its draft joining
    /// where it left. Refused by name, before the reset, for a state this
    /// seat did not take or one saved with the draft on put back with it off
    /// (or the other way); the body refuses another model's state or layout.
    fn resume(&mut self, state: &dyn Saved) -> Result<(), GateError> {
        let saved = state
            .as_any()
            .downcast_ref::<SavedGlm>()
            .ok_or("a saved state that is not a glm5next body's")?;
        let sel = self.s.selected();
        self.drafted.takes(saved.draft.as_ref())?;
        self.drafted.reset(&mut self.s, sel)?;
        seq_resume(self.s.model_mut(), &saved.state)?;
        self.drafted.unpark(sel, saved.draft.as_ref())
    }

    /// A prefix kept less of than shared is a `cache reuse` record; every
    /// other note of the prompt cache prints as its line.
    fn note(note: &CacheNote) {
        if let CacheNote::Reuse {
            common,
            ask,
            kept,
            held,
            reason,
        } = note
        {
            Record::new(&record::CACHE_REUSE)
                .u("common", common)
                .u("ask", ask)
                .u("kept", kept)
                .u("held", held)
                .w("reason", reason.as_deref().unwrap_or("unstated"))
                .eprint();
            return;
        }
        eprintln!("{WHAT}: {note}");
    }
}
