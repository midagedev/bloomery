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
//! `generate_glm5next` takes it): `a` (the serving plan, `workstation::plan_a`,
//! on the A6000 — the default, as both other serving seats), `gate` (the gate
//! card's), or `bp` (plan (b′): plan (a) on the A6000 and the 3090 an expert
//! tier under the host tier, its prompt-batch bytes `place::tier_batch`'s) and
//! its list spelling `a6000+3090`; a list of more tier cards than the GLM
//! body serves (`bloomery_gpu::host::SERVED_TIERS`) and a stage other than the
//! A6000 are refused by name before the plan. The
//! positions a slot serves are the stores the load sized for each sequence
//! (`--ctx` names the total the slots split): `/props`' `n_ctx` is that
//! number, a prompt that long is a 400 before it reaches the engine, and
//! generation stops there with `truncated`. Unset, a slot's context is the
//! file's trained context (`context_length`) capped to what the plan takes —
//! the largest context whose plan of every slot stands, itself never past
//! `place::ORACLE_POSITIONS` — pulled back to the largest multiple of
//! `CTX_STEP` that stays within the plan's `MARGIN` of stage-card expert
//! bytes (`serve_seats::ctx`'s guard, qwen38's margin rule: more positions
//! on the card push card experts to the host, and decode crawls), and never
//! under 2048 unless the card holds less than that (qwen38's `card` rule);
//! a `ctx` line on stderr names the rule, the chosen context, the slots and
//! their total, the trained context, the fit and the margin. Set, the flag
//! is the total and each slot takes `total / N` positions rounded down
//! (llama-server's `-np N` without `-kvu`), a slot under the body's floor
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
//! ([`DraftedSeat::park`]). A returning session's state comes back whole
//! after the session's reset (`seq_resume`), its checkpoints that one point,
//! its draft joining where it left (`DraftedSeat::unpark`); the residency is
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
//! `--parallel N` (`-np N`, default 2) serves N resident sequences inside
//! the one model (`Session::add_slots` over the body's `Slots`): the server
//! steps every running slot in each round, one pass a slot, the sequences
//! switched by pointer exchange and the draft's side of each slot kept with
//! it ([`DraftedSeat::select`]), so each request's tokens are its solo
//! run's; `--parallel 1` is the one-sequence server and `--parallel 0` is
//! refused by name. The plan counts every sequence in its KV term
//! (`PlanInputs::plan_slots`, `plan_nextn_slots`), so the context search, the
//! load and `/props`' `vram_kv_bytes` see them all, and a card that cannot
//! hold them is refused by name. Nothing parks — no slot waits for another —
//! so `--park-ram` is refused by name. A `parallel` line on stderr names the
//! rule (`slots`), the slots, a slot's context and the total. `--queue-depth
//! Q` bounds the requests that wait for a slot.
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
//! and with stores too short for one window, never a refusal. Set, refused by
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
use bloomery_gpu::model::StepMode;
use bloomery_gpu_gates::bind::{
    CacheRam, Seat, SeatEngine, Vocab, model_props, nvidia_smi_index, placement_props,
    sampler_factory,
};
use bloomery_gpu_gates::generate::{Place, mode_name, with_cards};
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::residency38::{GLM_CARD, residency_room, residency_set};
use bloomery_gpu_gates::{GateError, ref_model_path};
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
use serve::{
    CacheNote, DraftProps, Drafted, EngineProps, FATAL_LINGER, ResidencyReset, Saved, ServeError,
    Server, ServerConfig, SlotConfig,
};
use tokenizer::Tokenizer;

use super::drafted::{DraftedSeat, ParkedDraft};
use crate::glm_place;

/// The seat's name, as its records and errors print it.
const WHAT: &str = "bloomery-serve-glm";

const USAGE: &str = "usage: bloomery-serve [--model glm] [--host H] [--port P] \
                     [--place a|gate|bp|<stage>[+<tier>…]] \
                     [--ctx C] [--alias NAME] [--cache-ram MIB] [--slot-save-path DIR] \
                     [--chat-template-file PATH] [--prefill batch|steps] [--parallel N] \
                     [--queue-depth Q] [--plan]";

/// The positions the stores are sized for when `--ctx` names none:
/// `generate_glm5next`'s default, and the floor [`ctx_of`]'s rule never
/// goes under while the card holds it.
const CTX: usize = 2048;

/// The default context is a multiple of this many positions
/// (`serve_seats::ctx`'s guard rounds to it).
const CTX_STEP: usize = 256;

/// The levers this seat acts on: those `generate_glm5next` reads for its load
/// and its draft (`BLOOMERY_ROUTE_TRACE` and `BLOOMERY_STEP_STATS` left out —
/// each is one run's instrument, which a server that serves many prompts
/// does not wire, and a lever set outside this list is refused by name at
/// `main`), with `BLOOMERY_PIN_MAIN` for the engine thread's cpu slot, as
/// every serving seat reads it; `gate_glm5next_serve` holds the same list.
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
    bloomery_levers::R8,
    bloomery_levers::PIN_MAIN,
    bloomery_levers::DRAFT,
    bloomery_levers::RESIDENCY,
];

/// The drafted window's verify: the target's next token and the draft's one
/// proposal.
const VERIFY_ROWS: usize = <Body as MtpBody>::VERIFY_ROWS;

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
    /// (`serve_seats::ctx`'s guard).
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
    let fit = super::ctx::largest(1, cap, fits)?;
    let base_at = CTX.min(fit);
    let base_bytes = card(base_at)?;
    let margin_ctx = super::ctx::within_margin(fit, base_at, base_bytes, CTX_STEP, &card)?;
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
    place: Place,
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
    /// `--parallel`: the resident sequences the seat serves.
    parallel: usize,
    queue_depth: Option<usize>,
}

fn parse_args(args: &[String]) -> Result<Args, GateError> {
    let mut a = Args {
        host: "127.0.0.1".to_owned(),
        port: 8080,
        place: Place::A,
        ctx: None,
        alias: None,
        cache_ram: None,
        slot_save_path: None,
        template_file: None,
        prefill: PrefillMode::Batch,
        plan_only: false,
        parallel: 2,
        queue_depth: None,
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
            "--port" => a.port = v.parse()?,
            "--place" => a.place = glm_place::parse(v)?,
            "--ctx" => a.ctx = Some(v.parse()?),
            "--alias" => a.alias = Some(v.to_owned()),
            "--cache-ram" => a.cache_ram = Some(CacheRam::parse_mib(flag, v)?),
            "--slot-save-path" => a.slot_save_path = Some(PathBuf::from(v)),
            "--parallel" | "-np" => a.parallel = v.parse()?,
            "--queue-depth" => a.queue_depth = Some(v.parse()?),
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
    if a.parallel == 0 {
        return Err("--parallel 0: the server serves no slot".into());
    }
    a.place = a.place.on_host()?;
    Ok(a)
}

/// Loads the model and serves until the listener or the engine fails; `Ok`
/// carries why the server ended.
pub fn run(args: &[String]) -> Result<ServeError, GateError> {
    let levers = bloomery_levers::at_main(ACTS_ON)?;
    record::at_main(WHAT, record::BLOOMERY_SERVE_GLM);
    let a = parse_args(args)?;
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
    // where a window fits, and never refuses).
    let slots = a.parallel;
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
    let unset = glm_unset(GlmAt {
        serving_place: a.place != Place::Gate,
        nextn_layers: inputs.hp.n_layer.saturating_sub(inputs.hp.n_trunk),
        need: NEED,
        ctx: at_ctx,
        prefill_steps: a.prefill == PrefillMode::Steps,
    });
    let lever = residency_of(&levers, a.prefill, unset.residency)?;
    let draft_off = draft_of(&levers, at_ctx, unset.draft)?;
    let model = model_props(&split, &inputs.model);
    let tier_batch = glm_place::tier_batch(a.place, &inputs.hp);
    let machine = a.place.machine(None, tier_batch)?(inputs.model.layers);
    let nextn = match draft_off {
        None => Some(NextnInputs::read(&inputs)?),
        Some(_) => None,
    };
    let rule = ctx_of(&inputs, &machine, &plan_levers, nextn.as_ref(), set, slots)?;
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
    record::plan(a.place.name(), &machine, &plan).eprint();
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
        "parallel rule=slots slots={slots} slot_ctx={} total={}",
        rule.ctx,
        slots * rule.ctx
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
        place: a.place,
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
        .w("place", a.place.name())
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
}

/// The GLM session on the engine thread: the session over the model, the
/// draft it drives when one runs (its windows of two rows through the
/// runtime's speculative loop), and the positions its stores were sized
/// for.
struct Glm {
    s: Session<Body>,
    drafted: DraftedSeat<Body, VERIFY_ROWS>,
    ctx: usize,
    /// The plan's bytes for the NextN layer, for `/props`' `draft` class.
    draft_bytes: u64,
    /// The target file, whose NextN layer `/props`' `draft` names.
    path: PathBuf,
    /// The load runs the residency machine: each call prints its
    /// boundaries' `residency pass` records.
    residency: bool,
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
        let mut drafted = DraftedSeat::new(match a.draft_off {
            Some(_) => None,
            None => {
                let draft = MtpDraft::open(s.model(), prefill, StepMode::Eager)?;
                Some(s.with_draft::<MtpDraft<Body>, VERIFY_ROWS>(draft, &mut PairCapture)?)
            }
        });
        drafted.hold_slots(a.slots);
        Ok(Glm {
            s,
            drafted,
            ctx: a.ctx,
            draft_bytes: a.draft_bytes,
            path: a.path,
            residency,
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
}

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

    /// The session's slot through the draft's table
    /// ([`DraftedSeat::select`]): the live slot's draft side parked and the
    /// target's put back around the exchange.
    fn select(&mut self, slot: usize) -> Result<(), GateError> {
        self.drafted.select(&mut self.s, slot)
    }

    /// The draft's state is per slot ([`Glm::select`]): a slot's drafted
    /// passes are the passes it would run alone.
    fn slot_drafts(&self) -> bool {
        self.drafted.drafts()
    }

    /// The body's feed ([`bloomery_gpu_glm5next::feed`]): the batched prompt
    /// call or one step a position, as `--prefill` says, each call taking the
    /// checkpoints its marks name; under the draft the draft's own prompt
    /// call, its store walked over the prompt's units.
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        let next = self.drafted.prefill(&mut self.s, ids)?;
        self.print_passes()?;
        Ok(next)
    }

    /// One step; under the draft the rows it left waiting walked first
    /// (`MtpDraft::before_step`: a request that continues the held sequence
    /// joins it here when its prompt call is empty).
    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        let next = self.drafted.step(&mut self.s, last)?;
        self.print_passes()?;
        Ok(next)
    }

    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        Ok(self.s.model().logits_into(row)?)
    }

    /// One step and the target's row of it, read before the step is told to
    /// the draft ([`DraftedSeat::step_with_row`]).
    fn step_row(&mut self, last: u32, row: &mut [f32]) -> Result<u32, GateError> {
        let next = self.drafted.step_with_row(&mut self.s, last, row)?;
        self.print_passes()?;
        Ok(next)
    }

    /// One pass from `last`: under the draft the window of two rows, its
    /// kept tokens and counts; without it one step.
    fn pass(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, GateError> {
        let d = self.drafted.pass(&mut self.s, last, out)?;
        self.print_passes()?;
        Ok(d)
    }

    /// The most positions one pass runs: the window's two rows, or one step
    /// without the draft.
    fn pass_rows(&self) -> usize {
        self.drafted.pass_rows()
    }

    /// The session's reset, then the draft started over: the residency stays
    /// where use has taken it (only [`Seat::residency_reset`] moves it back).
    fn reset(&mut self) -> Result<(), GateError> {
        self.drafted.reset(&mut self.s)
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
    /// it ([`DraftedSeat::park`]), as the prompt cache holds them.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
        Ok(Arc::new(SavedGlm {
            state: seq_save(self.s.model_mut())?,
            draft: self.drafted.park(),
        }))
    }

    /// The state put back ([`seq_resume`]) after the session's reset, then
    /// the draft's side of it ([`DraftedSeat::unpark`]): the sequence's next
    /// call runs as it would have with no switch between, its draft joining
    /// where it left. Refused by name, before the reset, for a state this
    /// seat did not take or one saved with the draft on put back with it off
    /// (or the other way); the body refuses another model's state or layout.
    fn resume(&mut self, state: &dyn Saved) -> Result<(), GateError> {
        let saved = state
            .as_any()
            .downcast_ref::<SavedGlm>()
            .ok_or("a saved state that is not a glm5next body's")?;
        self.drafted.takes(saved.draft.as_ref())?;
        self.drafted.reset(&mut self.s)?;
        seq_resume(self.s.model_mut(), &saved.state)?;
        self.drafted.unpark(saved.draft.as_ref())
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
