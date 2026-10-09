//! `bloomery-serve-qwen38` — the llama-server-compatible HTTP API on the
//! Qwen3.8-Flash-Next (qwen4exp) engine, for a client to attach to.
//!
//! This module is the seat: `bloomery-serve-qwen38` and `bloomery-serve --model
//! qwen38` are each one call of [`run`], which takes the process's arguments
//! (`--model` already taken out by the one-binary server).
//!
//!     bloomery-serve-qwen38 [--host 127.0.0.1] [--port 8080] [--place a|gate|bp]
//!                           [--ctx-size C] [--alias NAME] [--cache-ram MIB]
//!                           [--chat-template-file PATH] [--parallel N]
//!                           [--queue-depth Q] [--park-ram MIB] [--slot-save-path DIR]
//!
//! `--slot-save-path` names the directory the slot actions answer from, as
//! llama-server's; without it every slot action is refused. `--ctx` is `--ctx-size` under its family's other spelling. The model is
//! `$BLOOMERY_REF_MODEL` (a `--model` flag is not taken: the family's
//! binaries read the one the profile pins); its first shard gives the
//! vocabulary, `tokenizer.chat_template` the chat template
//! (`--chat-template-file` replaces it) and `general.name` the default alias.
//! The `plan` record, then the `load` and `capture` lines of the qwen4exp
//! open, go to stderr as `generate_qwen3moe` prints them, then `listening on
//! http://<addr>` once the model is loaded and the port is bound (`--port 0`
//! binds a free one). Sampling is the sampler crate's chain with no
//! repetition penalty; `temperature <= 0` is the engine's argmax, the ids
//! `generate_qwen3moe --tokens <the prompt's ids>` prints for a prompt of at
//! most eight ids (past that the ubatch walk the server and the CLI both take
//! leaves a position's bits a function of its own inputs, so a longer
//! prompt's greedy ids are the walk's, not a step-fed run's).
//!
//! `--place` set runs its word as given; unset, the common rule every
//! serving seat takes decides (`generate::Place::choose`, by this family's
//! `q38place::Q38_RULE`: two cards keep the offer `bp` while its plan holds the
//! rule's break-even experts on the tier card or more and more than none —
//! else `a`; one card `a` on the one card), its `place unset` record among
//! the seat's first records after a set residency lever's.
//!
//! The positions a slot serves are the stores the load sized
//! (`--ctx-size` names one request's context: while `--parallel` names no
//! count one slot serves the whole of it, and under `--parallel N` it is
//! the total the N slots split): `/props`' `n_ctx` is a
//! slot's number, a prompt that long is a 400 before it reaches the engine,
//! and generation stops there with `truncated`. Unset, the context is
//! [`margin38`]'s answer over a slot's share — the largest multiple of
//! [`CTX_STEP`] up to the fit (the largest a slot's N-slot plan holds on the
//! card, at most the file's serving cap) whose plan holds at most the plan's
//! own margin (`MARGIN`) fewer card expert bytes than the N-slot plan at
//! [`CTX`] a slot, the fit when every context does — the largest the card
//! holds when that is fewer than [`CTX`], and at most what lets one
//! session's sequence state with its two recurrent copies fit the prompt
//! cache's budget (one session's whole slot context can be saved). Set, a
//! total a slot's share of which passes the file's `context_length`
//! (`place::serve_ctx`: YaRN scaling past it is not built), then one whose
//! N-slot plan the card cannot hold beside the plan's dense weights — the
//! largest slot context any N-slot plan of the file and the placement takes
//! up to that cap ([`fit38`]) — is refused by name before the load. The
//! `load` line prints the stores' `ctx` (a slot's rows) with the slot count
//! and each slot's context, the cap as `ctx_max`, `ctx_train` and
//! `verified`, the deepest context the reference sets hold our numbers to
//! ik's at (`refset::arch::qwen4exp::VERIFIED_POSITIONS`, `/props`'
//! `engine.ctx_verified`), which bounds nothing. A `ctx` line on stderr names the rule, the slot context, the slots and
//! the total they serve, the largest context the card holds with its card
//! expert bytes, and the largest whose plan holds at most the plan's own
//! margin (`MARGIN`) fewer card expert bytes than the plan at [`CTX`] a slot
//! ([`margin38`]): more positions on the card push card experts to the host.
//!
//! A request keeps the longest prefix it shares with what the slot holds that
//! the recurrent layers can stand at: every held position, or the nearest
//! checkpoint at or below the shared prefix. A prompt call copies the
//! recurrent stores (each delta layer's committed state lane and conv ring,
//! the PLE ring) and the PLE hash's history to the host at its start when it
//! continues a sequence, every `CHECKPOINT_EVERY` positions inside it and at
//! its end (`Body38::set_checkpoints`), so a request that resends the
//! conversation with the last turn's reasoning taken out keeps everything up
//! to the last prompt's end. The server cuts a prompt call at the first and
//! the last message start inside it (`<|im_start|>`, when the vocabulary and
//! the chat template have it), where both runs keep at least
//! `Prompt38::GEMM_FROM` ids, so a later session that shares the system
//! prompt keeps it; each run is the ubatch walk, whose bits do not depend on
//! where a call is cut. Every prefix a request keeps less of than it shares is
//! a `cache reuse` record with the rule.
//!
//! The prompt cache (llama-server's `--cache-ram`, in MiB; 0 turns it off)
//! holds the slot's sequence state when a request of another session takes
//! the slot: the positional rows (K/V, raw and pooled keys) of the held
//! positions, the recurrent stores at the held position and at the last
//! prompt call's end, each with its PLE history, and under the draft the
//! layer's side (`Seq38`): its store rows and the step's and the pass's
//! arena rows with the positions they hold, the draft's waiting rows beside
//! them ([`SlotDrafts::park`]). A returning session's state comes back
//! whole, and its checkpoints are those two points. Its default is the
//! lesser of `bind::CACHE_RAM_CAP` and half of what `MemAvailable` leaves at
//! load past the plan's host need, the residency's churn pool and the
//! checkpoints' host budget; a `cache` line on stderr prints it with each
//! term. A state of another model, card, context or store layout (a load
//! with the draft beside one without it) is refused by name; every save,
//! load, eviction and skip prints as a line.
//!
//! `--parallel N` (`-np N`) serves N resident sequences inside the one
//! model (`Session::add_slots` over `Body38`'s [`Slots`]): on a
//! drafted load the server's round of the busy slots' drafted passes runs
//! as one pass of their windows (`app::mtp::pass_slots`, every slot's rows
//! verified together, the round cut into passes at the body's
//! `SlotRows::MAX_ROWS`), and its round of plain steps (a sampling or
//! id-banning request's) a select and a step a row, each step told to its
//! slot's draft; on a load that drafts nothing every request steps, and the
//! round of the busy slots' steps runs as one pass of their rows
//! (`GpuModel::step_slots`, each row bit for bit its step alone). The
//! sequences are switched by pointer exchange, and one draft is held a
//! resident slot — its host state never moved, its device side parking
//! with the sequence on every switch — so each request's tokens are its
//! solo run's. Under `--place bp` (an expert tier card, beside which the
//! body runs no pass of several slots) every round runs a slot at a time,
//! and the open prints a line that says so. `--parallel 1` is exactly the
//! one-sequence server, its rounds one pass a slot. `--parallel` naming no
//! count, the seat serves one slot beside a set `--ctx-size` (the flag is
//! one request's context, `placement::ctx::slots_of`; one line on stderr says so) and
//! its default two over the automatic context. Under more slots than one
//! the context is split
//! as llama-server splits it with
//! `-np N` and no `-kvu`: the total (the `--ctx-size` the flags named, or
//! the automatic choice when unset) is the slots' sum, each slot `total /
//! N` positions rounded
//! down, and the search for the default — the largest total whose N-slot
//! plan fits the card within the margin rule `ctx38` applies, each slot a
//! multiple of [`CTX_STEP`] — runs over a slot's context, the plan counting
//! every sequence ([`PlanInputs::plan_with_slots`]). An explicit `--ctx` a
//! slot of whose N-slot plan the card cannot hold is refused by name before
//! the load, naming the split. Nothing parks: no slot ever waits for another
//! (a slot's round runs whether the others stream), so `--park-ram` is
//! refused by name — resident slots hold their state on the card, in the
//! plan. A `parallel` line on stderr names the rule (`slots`), the slots,
//! the split they serve and what set the count (`from`: the `--parallel`
//! flag, a set `--ctx-size`, the default). `--queue-depth Q` bounds the
//! requests that wait for a slot.
//!
//! The MTP draft keeps the same rule past a break-even, and the server cuts
//! its prompt calls at the same message starts (the draft's prompt call joins
//! each run where the one before left it). The draft rejoins a sequence only
//! where its last call left it: a state put back carries its waiting rows
//! ([`SlotDrafts::park`]) and it drafts on as if no other sequence had run,
//! while a cut to a checkpoint — the rows past the cut belonging to the
//! branch the cut dropped — leaves it proposing nothing until a request
//! starts from position 0. The seat turns the draft off there
//! ([`SlotDrafts::turn_off`]), prints a `bloomery-serve-qwen38: the MTP draft
//! proposes nothing …` line at the cut, and each later prompt call's `mtp
//! prompt` record names why. A request that keeps every held position with
//! the draft on keeps the draft. One whose kept prefix would leave the draft
//! off (a cut, or a draft already off) keeps it only at or past the
//! break-even (`Q38::draft_keep`): the prefix whose re-prefill costs what the
//! draft saves over the reply — the request's tokens through passes, at most
//! `NOMINAL_REPLY`, that when it bounds nothing — at the placement's prompt
//! rate (`docs/plan.md`, Qwen3.8 serve). Below it the server resets, the draft
//! on again, and prefills the whole prompt. A `draft keep` line after the
//! `cache` line prints the break-even, and each such request an `mtp keep`
//! record (`kept` or `reset`, the prefix, the break-even, the reply).

//! `BLOOMERY_DRAFT` unset follows the placement as `generate_qwen3moe`'s
//! does (`bloomery_levers::draft38_unset`): under `--place a` or `bp` the MTP draft
//! runs when a regular file is where it would be opened
//! (`BLOOMERY_MTP_DRAFT`, else the shared draft file beside the target); the
//! plain path runs under `--place gate`, with no file there, and with stores
//! too short for one window (`--ctx-size` under 5), each printed as a `load
//! draft=off (<why>)` record after the `load` line (`no file at <path>` for
//! the missing file), never a refusal. Unset with the ctx rule's own
//! default, the draft then yields to the context
//! (`bloomery_levers::DraftYield::of`, below at the search): it goes off
//! when the plan with it leaves a slot under the base the plain rule aims
//! for, one `draft yield` record naming its card bytes, the positions a
//! slot gets either way and the base, and the plain rule's `ctx` line
//! following; `BLOOMERY_DRAFT=mtp` set never yields, and a set
//! `--ctx-size` keeps today's answer. `BLOOMERY_DRAFT=off` is the plain
//! path with the same record; `mtp` drafts wherever the draft loads. The
//! CLI's `--logits` and route-trace conditions have no seat equivalent: a
//! request that reads the logits row steps plainly, and the seat does not
//! take the route trace.
//!
//! Drafting, the seat drives the session through the runtime's speculative
//! loop with the shared window `app::mtp::MtpDraft` (the shared draft file
//! beside the target or `BLOOMERY_MTP_DRAFT`'s, its head
//! `BLOOMERY_MTP_HEAD_ROWS`'s: unset the shipped list on a target of its
//! tokenizer, `full` the full head) behind the shared width chooser
//! (`BLOOMERY_MTP_WIDTH`, `runtime::width`): windows of at most four rows
//! — `cost`, the default, verifies the width whose measured pass wall pays
//! most and none while no width beats the plain step, `fixed` the draft's
//! three ids whole — the greedy ids the plain server's, `pass_rows` 4;
//! under `cost` a request's `mtp width` record prints at the slot's next
//! prompt call. A sampling or id-banning request takes plain
//! steps (the server's loop asks a pass only of a greedy request with no
//! banned id), each step's row the target's, read before the step is told
//! to the draft; its ids are the plain server's. A `load draft=mtp` line
//! follows the `load` line, then the `mtp head` record (which head, what
//! picked it and why), and `/props`' `engine.draft` names the draft
//! with its resident bytes as the card's `draft` class. Every other word of
//! the lever is refused by name, and so are `BLOOMERY_MTP_HEAD_ROWS` and
//! `BLOOMERY_MTP_DRAFT` set on a server that drafts nothing, with why.
//!
//! `BLOOMERY_RESIDENCY` set prints as a `residency lever` record first
//! thing. Unset, the Qwen3.8 rule picks the word as in `generate_qwen3moe`
//! (`bloomery_levers::residency38_unset`, `residency38_at_plan`) and a
//! `residency unset` record after the `plan` line prints it with why: under
//! `--place a` or `bp` `mid-p<P>-s1`, P half the fewest card experts a layer of the
//! plan the load runs (the plain or the MTP plan, at `--ctx-size`); `off`
//! under `--place gate`, when the plan holds no card expert or its fewest
//! leave no room, and when the churn pool does not fit the plan's host
//! headroom or what `MemAvailable` leaves past the plan's host need — never
//! a refusal. Running `mid-p<P>-s<S>` (plain or drafted, either `--place`),
//! the load runs the common residency machine over the card's routed stacks
//! (`Body38::open_placed_residency`, `open_placed_mtp_residency`): a
//! `residency host` record follows the `plan` line, and each call (a prompt,
//! a step, a pass) prints its boundaries' `residency pass` records after it.
//! A request's reset keeps the residency where use has taken it; `POST
//! /residency/reset` moves it back to its seed on a free slot and prints a
//! `residency reset` record (without the residency, the server's 501). The
//! seat's prompt path is `auto` — passes below nine ids, ubatches from nine
//! on, never one step an id — so the body's refusal of a step-fed prompt
//! beside the machine is never reached.
//!
//! `BLOOMERY_XSTREAM` resolves by the rule `generate_qwen3moe` runs
//! (`shared/xstream38.rs`): set as given; unset `split` under `--place a`
//! with a residency machine (`admit` when the card has no room for the
//! stream's ring), `admit` under `--place bp` with one, `off` everywhere
//! else. The open resolves it last, after the slots and the drafts, and
//! prints its `xstream=` line (the word, then why) after the capture lines.
//!
//! An engine error ends the process: the request gets a 500, `/health` a 503
//! for a moment, then the crash block (card, position, error) goes to stderr
//! and the exit code is 70.
//!
//! The binary builds under the `deepseek41` feature, whose name is the
//! server surface's scoping (gpu-gates' `bind` and `serve_client`, the
//! `serve` and `sampler` crates), not a model: it runs no V4.1 code.
//!
//! The levers it acts on ([`ACTS_ON`]) are parsed once, at `main`
//! (`bloomery_levers::at_main`), which refuses by name a lever set outside
//! them and a `BLOOMERY_*` name no registry row names; `--levers` prints them
//! with this process's values and exits. The Qwen3.8 levers are
//! `BLOOMERY_QWEN38_EXPERTS` (the plan's expert rule) with
//! `BLOOMERY_CARD_BUDGET` bounding its card plan, the host
//! tier's load settings, `BLOOMERY_PIN_MAIN`, the draft's levers,
//! `BLOOMERY_RESIDENCY`, `BLOOMERY_XSTREAM` and `BLOOMERY_STEP_STATS` (the
//! `slots round` record a round of several slots prints); the ubatch size
//! (`BLOOMERY_QWEN3_UBATCH`) is read where the load sizes its arena. The
//! stderr lines named above are records of the kinds
//! `bloomery_gpu_gates::record` declares; `--records-schema` prints those
//! kinds and exits.

use std::any::Any;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use app::mtp::{MtpBody, MtpDraft};
use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for;
use bloomery_gpu::arch::qwen3moe::{Body38, Prompt38, Seq38, TargetRows, seq38_bytes};
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::model::StepMode;
use bloomery_gpu_gates::bind::{
    CacheRam, Seat, SeatEngine, SlotPassRow, SlotStep, Vocab, model_props, nvidia_smi_index,
    placement_props, sampler_factory,
};
use bloomery_gpu_gates::generate::{BreakEven, Place};
use bloomery_gpu_gates::nodes::count_kinds;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::residency38::{CARD38, Lever38, residency38};
use bloomery_gpu_gates::{GateError, gpu_census, ref_model_path};
use bloomery_levers::{
    Draft38At, Draft38Off, Residency38At, ResidencyPick, ResidencyWhy, draft38_unset,
    residency38_unset,
};
use cuda_core::sys;
use gguf::Split;
use model::arch::models::Mixer;
use model::arch::qwen35moe::head_list::{HeadPick, head_rows_of};
use model::arch::qwen35moe::place::{
    Experts, MtpInputs, PlaceError, PlanInputs, machine_bp_on, machine_for_experts, serve_ctx,
    tier_batch,
};
use model::placement::churn::ChurnPool;
use model::placement::workstation::{CardSpec, HostNeed, MARGIN};
use model::placement::{Machine, PlacementError, Plan, PlanLevers};
use refset::arch::qwen4exp::VERIFIED_POSITIONS;
use refset::arch::qwen4exp::mtp::{DraftFrom, draft_file};
use runtime::Target as _;
use runtime::width::Mode as WidthMode;
use serve::{
    CacheNote, DraftProps, Drafted, EngineProps, FATAL_LINGER, ResidencyReset, Saved, ServeError,
    Server, ServerConfig, SlotConfig,
};
use tokenizer::Tokenizer;

use super::drafted::{ParkedDraft, SlotDrafts};

#[path = "../qwen38_place.rs"]
mod q38place;
#[path = "../xstream38.rs"]
mod xstream38;
use xstream38::{Stage38, xstream38};

/// The levers `bloomery-serve-qwen38` acts on, for its own `main` and for
/// `gate_qwen38_serve`'s (the gate starts the server with its own
/// environment, so a lever the server would refuse is refused by the gate
/// first; the two lists must stay one).
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::QWEN38_EXPERTS,
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::PIN_MAIN,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
    bloomery_levers::R8,
    bloomery_levers::DRAFT,
    bloomery_levers::MTP_HEAD_ROWS,
    bloomery_levers::MTP_DRAFT,
    bloomery_levers::MTP_WIDTH,
    bloomery_levers::RESIDENCY,
    bloomery_levers::XSTREAM,
    bloomery_levers::STEP_STATS,
];

const USAGE: &str = "usage: bloomery-serve-qwen38 [--host H] [--port P] [--place a|gate|bp] \
                     [--ctx-size C] [--alias NAME] [--cache-ram MIB] [--chat-template-file PATH] \
                     [--parallel N] [--queue-depth Q] [--park-ram MIB] [--slot-save-path DIR] \
                     [--api-key KEY] [--api-key-file FNAME]";

/// The context the default's expert cost is counted against: the plan
/// every Qwen3.8 measurement ran at, and `generate_qwen3moe`'s default.
const CTX: usize = 4096;

/// The default context is a multiple of this many positions.
const CTX_STEP: usize = 256;

/// The token every message of the chat template opens with: a prompt call is
/// cut at the first and the last inside it.
const MESSAGE_START: &str = "<|im_start|>";

/// Why a cut leaves the MTP draft proposing nothing
/// ([`SlotDrafts::turn_off`]).
const DRAFT_OFF_WHY: &str = "the draft rejoins a sequence only where its last call left it, \
                             and it holds no rows at the kept position";

/// The reply the break-even weighs ([`Q38::draft_keep`]) when a request
/// bounds nothing, and the most it weighs: the mean greedy reply of the
/// Korean chat prompts the shipped head list was chosen on (`docs/plan.md`,
/// Qwen3.8 serve).
const NOMINAL_REPLY: usize = 277;

/// The plain step's and the drafted decode's positions per second, and the
/// drafted prompt's ids per second under plan (a), that the break-even
/// weighs (`docs/plan.md`, Qwen3.8 serve).
const PLAIN_TPS: f64 = 57.88;
const DRAFTED_TPS: f64 = 79.33;
const PROMPT_IDS_PER_S: f64 = 1224.1;

/// The prompt rate under `bp` over `a`'s, the same lease
/// (docs/cards/q38bpbug-ab.card: P 512, draft off, A6000 + 3090).
const BP_PROMPT_RATIO: f64 = 1.379;

/// The drafted seat's break-even at `place` ([`Q38::draft_keep`]): the
/// draft's gain a token times the rate a reset re-prefills at — the ubatch
/// walk's under `a` and `gate` (plan (a)'s stands for the 3090's: the product
/// is a ratio of one card's own rates), and that walk under `bp` at its
/// measured prompt ratio over `a`'s ([`BP_PROMPT_RATIO`],
/// docs/cards/q38bpbug-ab.card).
fn break_even_of(place: Place38) -> BreakEven {
    let rate = match place.kind {
        Kind38::A | Kind38::Gate => PROMPT_IDS_PER_S,
        Kind38::Bp => PROMPT_IDS_PER_S * BP_PROMPT_RATIO,
    };
    BreakEven::new(PLAIN_TPS, DRAFTED_TPS, rate)
}

/// The branch a drafted request took at its kept prefix ([`Q38::draft_keep`]),
/// its `mtp keep` record's fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Branch38 {
    /// The prefix was kept (at or past the break-even); else the request
    /// reset.
    kept: bool,
    /// The prefix the body's rule granted.
    prefix: usize,
    break_even: usize,
    /// The reply tokens the break-even was derived for.
    reply: usize,
}

/// Which placement `--place` names, as `generate_qwen3moe` takes it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind38 {
    /// The stage on the largest visible card (the A6000 here, the timing
    /// card; the default).
    A,
    /// The 3090, the gate card.
    Gate,
    /// Plan (b′): the stage as under `a`, the next-largest card its expert
    /// tier (`place::machine_bp_on`).
    Bp,
}

/// Where `--place` puts the plan's stage card, and its expert tier card
/// when it has one: the kind, and its cards (`generate::Place`), resolved
/// against this process's devices once the arguments are read.
#[derive(Clone, Copy)]
struct Place38 {
    kind: Kind38,
    cards: Place,
}

impl Place38 {
    const A: Place38 = Place38 {
        kind: Kind38::A,
        cards: Place::A,
    };

    fn parse(v: &str) -> Result<Place38, GateError> {
        let (kind, cards) = match v {
            "a" => (Kind38::A, Place::A),
            "gate" => (Kind38::Gate, Place::Gate),
            "bp" => (Kind38::Bp, Place::Bp),
            other => {
                return Err(format!(
                    "--place is a, gate or bp (plan (b′): the next-largest card as the \
                     largest's expert tier), not {other}"
                )
                .into());
            }
        };
        Ok(Place38 { kind, cards })
    }

    fn name(self) -> &'static str {
        self.cards.name()
    }

    /// The stage card's spec.
    fn spec(self) -> Result<CardSpec, GateError> {
        Ok(self.cards.card_specs()?[0])
    }

    /// The stage card is plan (a)'s: the placement the Qwen3.8 defaults
    /// treat as plan (a).
    fn stage_a(self) -> bool {
        matches!(self.kind, Kind38::A | Kind38::Bp)
    }

    /// The stage as the unset `BLOOMERY_XSTREAM` rule reads it.
    fn stage38(self) -> Stage38 {
        match self.kind {
            Kind38::A => Stage38::A,
            Kind38::Bp => Stage38::Tiered,
            Kind38::Gate => Stage38::Other,
        }
    }

    /// The placement holds an expert tier card (plan (b′)): the body runs
    /// no pass of several slots on such a load (`Body38::plan_slots`
    /// refuses a load with an expert tier card by name).
    fn tiered(self) -> bool {
        matches!(self.kind, Kind38::Bp)
    }

    /// The machine a plan of `inputs` at `ctx` positions and ubatches of
    /// `ub` runs on under `experts`, `mtp` the draft when it runs beside
    /// the target, for `slots` resident sequences: plan (b′) reserves the
    /// draft's card bytes at `ctx` for every slot on its stage card
    /// (`MtpInputs::card_bytes_of`, the reserve `plan_mtp_with_slots`
    /// checks; `place::machine_bp_on`), the one-card plans count them in
    /// `plan_mtp_with_slots`.
    fn machine(
        self,
        inputs: &PlanInputs,
        (ctx, ub): (u64, u64),
        experts: Experts,
        mtp: Option<&MtpInputs>,
        slots: usize,
    ) -> Result<Machine, GateError> {
        let layers = inputs.spec.layers.len();
        Ok(match self.kind {
            Kind38::A | Kind38::Gate => machine_for_experts(self.spec()?, layers, ub, experts),
            Kind38::Bp => {
                let cards = self.cards.card_specs()?;
                let draft = mtp.map(|m| m.card_bytes_of(ctx, slots)).transpose()?;
                machine_bp_on(
                    (cards[0], cards[1]),
                    layers,
                    ub,
                    draft,
                    tier_batch(&inputs.hp, ub),
                )
            }
        })
    }
}

/// The MTP draft file at `path`, which `from` picked; an error names both.
fn open_draft(path: &Path, from: DraftFrom) -> Result<Split, GateError> {
    Split::open(path).map_err(|e| {
        format!(
            "open the MTP draft {} ({}): {e}",
            path.display(),
            from.describe()
        )
        .into()
    })
}

/// `BLOOMERY_QWEN38_EXPERTS` as the plan's expert rule.
fn experts38(levers: &bloomery_levers::Levers) -> Result<Experts, GateError> {
    match levers.qwen38_experts() {
        "host" => Ok(Experts::Host),
        "card" => Ok(Experts::Card),
        other => Err(format!("BLOOMERY_QWEN38_EXPERTS={other}: host or card").into()),
    }
}

/// `BLOOMERY_DRAFT` on the seat at `place` with stores of `ctx` positions,
/// `file` the MTP draft file the load would open, `inputs` the target's:
/// `mtp` the draft, `off` the plain path; unset, `generate_qwen3moe`'s rule
/// (`bloomery_levers::draft38_unset`), then `off` when the target's matrices
/// the draft borrows are not its format (`PlanInputs::mtp_borrows`, which
/// refuses a set `mtp` at `MtpInputs::read`); and, drafting nothing, why (the
/// `load draft=off` record's). This is the rule before the plan: the unset
/// draft's yield to the context (`bloomery_levers::DraftYield::of`, the
/// module doc) is decided after the context search, at the caller. The V4.1
/// words and any other are refused by name.
///
/// The rule's run conditions as the seat meets them: no `--logits` (a
/// request that reads the logits row steps plainly, the row the target's),
/// no route trace (the seat does not act on `BLOOMERY_ROUTE_TRACE`, so
/// `at_main` refuses it set), and the positions the server's loop needs for
/// one window — it takes one only while its rows fit, the first at a one-id
/// prompt's first generated token: 1 + 1 + rows − 1.
fn draft38(
    levers: &bloomery_levers::Levers,
    place: Place38,
    ctx: usize,
    file: &Path,
    inputs: &PlanInputs,
) -> Result<(bool, Option<Draft38Off>), GateError> {
    match levers.draft() {
        Some("mtp") => Ok((true, None)),
        Some("off") => Ok((false, Some(Draft38Off::Set))),
        Some(other) => Err(format!(
            "BLOOMERY_DRAFT={other}: on a qwen4exp file mtp drafts the window and off runs the \
             plain path; lookup and dspark are the V4.1 binaries'"
        )
        .into()),
        None => {
            let at = Draft38At {
                place_a: place.stage_a(),
                logits: false,
                route_trace: false,
                file,
                file_is_there: file.is_file(),
                need: <Body38 as MtpBody>::VERIFY_ROWS + 1,
                ctx,
            };
            Ok(match draft38_unset(&at) {
                None => match inputs.mtp_borrows() {
                    Ok(()) => (true, None),
                    Err(PlaceError::DraftBorrow { name, ty }) => {
                        let ty = ty.map_or("absent".to_string(), |t| t.to_string());
                        (false, Some(Draft38Off::Borrowed { name, ty }))
                    }
                    Err(e) => return Err(e.into()),
                },
                Some(off) => (false, Some(off)),
            })
        }
    }
}

/// The residency the load of `plan` at `place` runs: `set` (the word and
/// its parse) as given; unset, the Qwen3.8 rule's
/// ([`residency38`]: before the plan `off` under `--place gate`, else on plan
/// (a) from it), its records on stderr.
fn residency38_at(
    plan: &Plan<'_>,
    place: Place38,
    set: Option<(Residency, &str)>,
) -> Result<Residency, GateError> {
    let lever = match set {
        Some((r, word)) => Lever38::Set(r, word),
        None => {
            // The seat feeds no prompt by steps (its path is `auto`) and
            // takes no route trace.
            let at = Residency38At {
                qwen38_file: true,
                dump_taps: false,
                place_a: place.stage_a(),
                route_trace: false,
                prefill_step: false,
            };
            Lever38::Unset(residency38_unset(at))
        }
    };
    residency38(plan, lever, Record::eprint)
}

/// The plans the seat can load, by context: the file's inputs at `place`
/// under the expert rule, the placement levers and the draft, every
/// per-sequence term counted `slots` times
/// ([`PlanInputs::plan_with_slots`]).
struct Plans<'a> {
    inputs: &'a PlanInputs,
    place: Place38,
    experts: Experts,
    levers: &'a PlanLevers,
    mtp: Option<&'a MtpInputs>,
    /// The resident sequences every plan counts; 1 is the one-sequence plan
    /// itself.
    slots: usize,
}

impl Plans<'_> {
    /// The plan's card expert bytes at `ctx` positions a sequence serves, or
    /// why no plan takes that context.
    fn card(&self, ctx: usize) -> Result<u64, GateError> {
        let ub = ubatch_for(ctx)?;
        let ctx = u64::try_from(ctx)?;
        let machine = self.place.machine(
            self.inputs,
            (ctx, u64::try_from(ub)?),
            self.experts,
            self.mtp,
            self.slots,
        )?;
        let plan = match self.mtp {
            None => {
                self.inputs
                    .plan_with_slots(&machine, ctx, self.levers, self.experts, self.slots)?
            }
            Some(mi) => {
                self.inputs
                    .plan_mtp_with_slots(&machine, ctx, self.levers, mi, self.experts, self.slots)?
                    .plan
            }
        };
        Ok(plan
            .cards
            .first()
            .ok_or("a plan with no card")?
            .expert_bytes)
    }

    /// The plan's host need (`HostNeed`) at `ctx`, the churn pool the
    /// residency it runs holds beside it (`set` the lever as `run` read it;
    /// the rule's records unprinted), and the NVMe expert tier's arena the
    /// load's paged experts fill beside them
    /// (`HostTotals::nvme_arena_bytes`, 0 without one).
    fn host(
        &self,
        ctx: usize,
        set: Option<(Residency, &str)>,
    ) -> Result<(u64, u64, u64), GateError> {
        let ub = ubatch_for(ctx)?;
        let c = u64::try_from(ctx)?;
        let machine = self.place.machine(
            self.inputs,
            (c, u64::try_from(ub)?),
            self.experts,
            self.mtp,
            self.slots,
        )?;
        let plan = match self.mtp {
            None => {
                self.inputs
                    .plan_with_slots(&machine, c, self.levers, self.experts, self.slots)?
            }
            Some(mi) => {
                self.inputs
                    .plan_mtp_with_slots(&machine, c, self.levers, mi, self.experts, self.slots)?
                    .plan
            }
        };
        let lever = match set {
            Some((r, word)) => Lever38::Set(r, word),
            None => Lever38::Unset(residency38_unset(Residency38At {
                qwen38_file: true,
                dump_taps: false,
                place_a: self.place.stage_a(),
                route_trace: false,
                prefill_step: false,
            })),
        };
        let pool = match residency38(&plan, lever, |_| {})? {
            Residency::Mid { pinned, .. } => {
                ChurnPool::of(&plan, CARD38, pinned)
                    .map_err(|e| format!("the churn pool: {e}"))?
                    .bytes
            }
            _ => 0,
        };
        Ok((
            HostNeed::of(&plan, 0).bytes(),
            pool,
            plan.host.nvme_arena_bytes,
        ))
    }
}

/// The context the seat loads ([`ctx38`]) and what decided it: every number
/// a slot's own — the stores' rows one sequence serves — the total they hold
/// `slots` times over.
struct Ctx38 {
    /// A slot's context.
    ctx: usize,
    /// The resident sequences the plan counted.
    slots: usize,
    /// `set` (`--ctx-size`: under a set `--parallel N` the total the slots
    /// split, alone one slot's whole context), or what decided the
    /// default: `margin` ([`margin38`]'s answer), `card` (the largest
    /// context the card holds, fewer than [`CTX`]), `cache` (one state in
    /// the prompt cache's budget).
    rule: &'static str,
    /// What the default's search found; `None` for a set context, which
    /// plans its own split alone.
    search: Option<Search38>,
    /// The plan's card expert bytes at `ctx`.
    card_bytes: u64,
}

/// The default's search: the largest slot context the card holds
/// ([`fit38`]), the largest within the plan's margin ([`margin38`]), and the
/// plan's card expert bytes at [`CTX`] a slot.
#[derive(Clone, Copy)]
struct Search38 {
    fit: Fit38,
    margin_ctx: usize,
    base_bytes: u64,
}

/// The largest slot context the card holds beside the plan's dense weights,
/// and the plan's card expert bytes there.
#[derive(Clone, Copy)]
struct Fit38 {
    ctx: usize,
    card_bytes: u64,
}

/// The largest slot context any plan of the file and the placement takes —
/// every plan counting the seat's slots — up to the file's serving cap
/// (`place::serve_ctx`: its `context_length`): the most a `--ctx-size`'s
/// split may give a slot. A tier card the plan leaves idle is refused as
/// itself (`PlacementError::IdleTier`): no context gives it an expert.
fn fit38(plans: &Plans<'_>) -> Result<Fit38, GateError> {
    let fits = |c: usize| Ok(plans.card(c).is_ok());
    if let Err(e) = plans.card(1) {
        if let Some(PlaceError::Placement(PlacementError::IdleTier { .. })) =
            e.downcast_ref::<PlaceError>()
        {
            return Err(e);
        }
        return Err(
            format!("no context fits the card: the plan at 1 position is refused ({e})").into(),
        );
    }
    let most = usize::try_from(serve_ctx(1, &plans.inputs.hp)?)?;
    let ctx = model::placement::ctx::largest::<GateError>(1, most, fits)?;
    Ok(Fit38 {
        ctx,
        card_bytes: plans.card(ctx)?,
    })
}

/// The largest multiple of [`CTX_STEP`] up to `fit` whose plan holds at most
/// `MARGIN` fewer card expert bytes than the plan at `base_at` a slot
/// (`base_bytes` there), `fit` when every context does, `base_at` when no
/// step past it does ([`model::placement::ctx::within_margin`], the seats'
/// shared guard).
fn margin38(
    plans: &Plans<'_>,
    fit: usize,
    base_at: usize,
    base_bytes: u64,
) -> Result<usize, GateError> {
    model::placement::ctx::within_margin(fit, base_at, base_bytes, CTX_STEP, MARGIN, &|c| {
        plans.card(c)
    })
}

/// The seat's context (the module doc), every number a slot's own: `set`
/// when the file's serving cap takes a slot's share of the total
/// (`place::serve_ctx`, refused by name past it) and it fits the card
/// ([`fit38`]), else refused by name; unset, [`margin38`]'s answer, or the
/// largest the card holds when that is fewer than [`CTX`]. The prompt
/// cache's bound comes after ([`Ctx38::host_bound`]).
fn ctx38(plans: &Plans<'_>, set: Option<usize>) -> Result<Ctx38, GateError> {
    let n = plans.slots;
    if let Some(c) = set {
        let ctx = c / n;
        if ctx == 0 {
            return Err(
                format!("--ctx-size {c}: --parallel {n} splits it to no position a slot").into(),
            );
        }
        serve_ctx(u64::try_from(ctx)?, &plans.inputs.hp)?;
        // A set context plans its own split alone: the default's search
        // probes contexts the flag did not name, and a refusal at one of them
        // is not this load's.
        return match plans.card(ctx) {
            Ok(card_bytes) => Ok(Ctx38 {
                ctx,
                slots: n,
                rule: "set",
                search: None,
                card_bytes,
            }),
            Err(refused) => {
                let fit = fit38(plans)?;
                if ctx > fit.ctx {
                    Err(format!(
                        "--ctx-size {c}: with --parallel {n} each slot takes {ctx} positions and \
                         the card holds at most {} a slot beside the plan's dense weights \
                         (`--place {}`)",
                        fit.ctx,
                        plans.place.name()
                    )
                    .into())
                } else {
                    Err(refused)
                }
            }
        };
    }
    let fit = fit38(plans)?;
    let base_at = CTX.min(fit.ctx);
    let base_bytes = plans.card(base_at)?;
    let margin_ctx = margin38(plans, fit.ctx, base_at, base_bytes)?;
    let (ctx, rule) = if base_at < CTX {
        (base_at, "card")
    } else {
        // Unset takes the fit's context without paying its card experts.
        (margin_ctx, "margin")
    };
    Ok(Ctx38 {
        ctx,
        slots: n,
        rule,
        search: Some(Search38 {
            fit,
            margin_ctx,
            base_bytes,
        }),
        card_bytes: plans.card(ctx)?,
    })
}

impl Ctx38 {
    /// A default bounded by the prompt cache too: one session's state at its
    /// slot's context ([`Seq38`: its positional rows and two recurrent
    /// copies, the draft's side under `drafts`](seq38_bytes)) fits `ram`
    /// bytes, when the cache is on. A set context keeps its value.
    fn host_bound(self, inputs: &PlanInputs, ram: u64, drafts: bool) -> Result<Ctx38, GateError> {
        if self.rule == "set" || ram == 0 {
            return Ok(self);
        }
        let gdn = inputs
            .spec
            .layers
            .iter()
            .filter(|l| matches!(l.mixer, Mixer::DeltaRule(_)))
            .count();
        let qsa = inputs.spec.layers.len() - gdn;
        let state = |n: usize| seq38_bytes(gdn, qsa, n, drafts);
        if state(self.ctx) <= ram {
            return Ok(self);
        }
        if state(1) > ram {
            return Err(format!(
                "the prompt cache's {ram} bytes hold no sequence state (two recurrent copies \
                 among its terms); --cache-ram 0 turns it off"
            )
            .into());
        }
        let c = model::placement::ctx::largest::<GateError>(1, self.ctx, |n| Ok(state(n) <= ram))?;
        let ctx = (c / CTX_STEP * CTX_STEP).max(c.min(CTX_STEP));
        Ok(Ctx38 {
            ctx,
            rule: "cache",
            ..self
        })
    }

    /// The `ctx` line on stderr: every context a slot's, `total` the slots'
    /// sum, `base` the plan at [`CTX`] a slot the margin counts against; a
    /// set context's line carries no search terms (it ran none).
    fn print(&self) {
        let Some(s) = self.search else {
            eprintln!(
                "ctx rule={} ctx={} slots={} total={} card_expert_bytes={}",
                self.rule,
                self.ctx,
                self.slots,
                self.slots * self.ctx,
                self.card_bytes
            );
            return;
        };
        eprintln!(
            "ctx rule={} ctx={} slots={} total={} fit={} fit_card_expert_bytes={} margin_ctx={} \
             base={CTX} base_card_expert_bytes={} card_expert_bytes={} lost_bytes={} \
             margin_bytes={MARGIN}",
            self.rule,
            self.ctx,
            self.slots,
            self.slots * self.ctx,
            s.fit.ctx,
            s.fit.card_bytes,
            s.margin_ctx,
            s.base_bytes,
            self.card_bytes,
            s.base_bytes.saturating_sub(self.card_bytes)
        );
    }
}

struct Args {
    host: String,
    port: u16,
    /// `--place`; `None` takes the common rule's choice ([`Place::choose`]).
    flag: Option<Place38>,
    /// The placement the seat runs by: the flag's word, or the common rule's
    /// choice, resolved against this process's devices.
    place: Place38,
    /// `--ctx-size`; `None` takes the rule's default.
    ctx: Option<usize>,
    alias: Option<String>,
    /// `--cache-ram` in bytes; `None` takes the default.
    cache_ram: Option<u64>,
    /// `--chat-template-file`, replacing the file's own template.
    template_file: Option<PathBuf>,
    /// `--slot-save-path`: the directory the slot actions answer from;
    /// `None` refuses every one, as llama-server does.
    slot_save_path: Option<PathBuf>,
    /// `--parallel`: the resident sequences the seat serves; `None` takes
    /// one slot beside a set `--ctx-size`, else the default 2
    /// ([`model::placement::ctx::slots_of`]).
    parallel: Option<usize>,
    queue_depth: Option<usize>,
    /// `--park-ram` in bytes; set is refused by name (resident slots park
    /// nothing).
    park_ram: Option<u64>,
    /// `--api-key`/`--api-key-file`: the keys every request is checked
    /// against ([`serve::flag::ApiKeys`]).
    api_keys: serve::flag::ApiKeys,
}

fn parse_args(args: &[String]) -> Result<Args, GateError> {
    let mut a = Args {
        host: "127.0.0.1".to_owned(),
        port: 8080,
        flag: None,
        place: Place38::A,
        ctx: None,
        alias: None,
        cache_ram: None,
        template_file: None,
        slot_save_path: None,
        parallel: None,
        queue_depth: None,
        park_ram: None,
        api_keys: serve::flag::ApiKeys::default(),
    };
    let mut it = args.iter().map(|s| s.as_str());
    while let Some(flag) = it.next() {
        if flag == "--help" || flag == "-h" {
            return Err(USAGE.into());
        }
        let v = it
            .next()
            .ok_or_else(|| format!("{flag} needs a value, or is unknown: {USAGE}"))?;
        match flag {
            "--host" => a.host = v.to_owned(),
            "--port" => a.port = serve::flag::number(flag, v)?,
            "--place" => a.flag = Some(Place38::parse(v)?),
            f if serve::flag::CTX.contains(&f) => a.ctx = Some(serve::flag::number(flag, v)?),
            "--alias" => a.alias = Some(v.to_owned()),
            "--cache-ram" => a.cache_ram = Some(CacheRam::parse_mib(flag, v)?),
            "--chat-template-file" => a.template_file = Some(PathBuf::from(v)),
            "--slot-save-path" => a.slot_save_path = Some(PathBuf::from(v)),
            "--parallel" | "-np" => a.parallel = Some(serve::flag::number(flag, v)?),
            "--queue-depth" => a.queue_depth = Some(serve::flag::number(flag, v)?),
            "--park-ram" => a.park_ram = Some(CacheRam::parse_mib(flag, v)?),
            f if serve::flag::KEYS.contains(&f) => a.api_keys.add(f, v)?,
            other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
        }
    }
    if a.ctx == Some(0) {
        return Err("--ctx-size 0: the stores hold no position".into());
    }
    if a.parallel == Some(0) {
        return Err("--parallel 0: the server serves no slot".into());
    }
    if a.park_ram.is_some() {
        return Err(
            "--park-ram holds the states of slots that take the model in turns; this seat's \
             slots are resident sequences, which park nothing — the plan counts their state"
                .into(),
        );
    }
    Ok(a)
}

/// Loads the model and serves until the listener or the engine fails;
/// `Ok` carries why the server ended.
pub fn run(args: &[String]) -> Result<ServeError, GateError> {
    let levers = bloomery_levers::at_main(ACTS_ON)?;
    record::at_main("bloomery-serve-qwen38", record::BLOOMERY_SERVE_QWEN38);
    // Set, the word runs as given; unset, the Qwen3.8 rule picks it once the
    // placement and the plan are known (`residency unset`).
    if let Some(word) = levers.residency() {
        record::residency_lever(ResidencyPick {
            word,
            why: ResidencyWhy::Set,
        })
        .eprint();
    }
    let set = match levers.residency() {
        Some(word) => Some((Residency::parse(word)?, word)),
        None => None,
    };
    let mut a = parse_args(args)?;
    // The placement the seat runs by: the flag's word, or unset the common
    // rule's (`Place::choose`) on the census, read once — chosen below, once
    // the plan's inputs are known, the rule asking the offer's plan for its
    // tier count. The kind is read by structure — the flag's when set, and
    // unset `Bp` only where the chosen placement holds a tier card — never
    // off the chosen word, which `Place::on` spells as card names when an
    // alias lands on other devices. The draft's rule reads the placement's
    // stage alone (`place_a`), and an unset flag's offer never names the
    // gate card, so the flag's own place — `a`'s stage — stands for the
    // offer until the rule chooses.
    let census = gpu_census::census()?;
    let draft_at = a.flag.unwrap_or(Place38::A);
    let path = ref_model_path()?;
    let (draft_path, draft_from) = draft_file(levers.mtp_draft(), &path);
    let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let inputs = PlanInputs::describe(&split)?;
    // The draft's one context condition (a window's positions) holds at any
    // context the rule grants; it is asked again at the final context below.
    let (mtp, draft_off) = draft38(
        &levers,
        draft_at,
        a.ctx.unwrap_or(CTX),
        &draft_path,
        &inputs,
    )?;
    // A draft lever set on a server that drafts nothing is refused, with
    // why — asked once the yield below has said its last word, so a draft
    // the plan turns off refuses them the same as one the rule did.
    let refuse_mtp_levers = |draft_off: &Option<Draft38Off>| -> Result<(), GateError> {
        if let Some(why) = draft_off {
            if levers.mtp_head_rows().is_some() {
                return Err(format!(
                    "BLOOMERY_MTP_HEAD_ROWS picks the MTP draft's head; the server drafts \
                     nothing ({why})"
                )
                .into());
            }
            if levers.mtp_draft().is_some() {
                return Err(format!(
                    "BLOOMERY_MTP_DRAFT names the MTP draft file; the server drafts nothing \
                     ({why})"
                )
                .into());
            }
            if levers.mtp_width().is_some() {
                return Err(format!(
                    "BLOOMERY_MTP_WIDTH picks the width a drafted window verifies; the server \
                     drafts nothing ({why})"
                )
                .into());
            }
        }
        Ok(())
    };
    let width = WidthMode::of(levers.mtp_width())?;
    let experts = experts38(&levers)?;
    let plan_levers = PlanLevers::from_levers(&levers)?;
    let tok = Tokenizer::from_gguf(&path)?;
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
        .unwrap_or("qwen3.8")
        .to_owned();
    drop(inv);
    // The message start the server cuts prompt calls at, when both the
    // vocabulary and the template have it.
    let has_start = tok
        .special_tokens()
        .iter()
        .any(|&id| tok.text(id) == Some(MESSAGE_START));
    let in_template = template.contains(MESSAGE_START);
    let vocab = Vocab::new(tok)?;
    let vocab = Arc::new(if has_start && in_template {
        vocab.with_user_start(MESSAGE_START)?
    } else {
        vocab
    });

    // The plan record, and `/props` from the same plan the load runs by:
    // under the draft `plan_mtp_with`'s, its draft's card bytes (granules,
    // store, row map) and its program's arena the card's `draft` class.
    let model = model_props(&split, &inputs.model);
    // The head is picked once, here; the load prints the pick.
    let head = match mtp {
        false => None,
        true => Some(head_rows_of(
            levers.mtp_head_rows(),
            &split,
            inputs.spec.vocab,
        )?),
    };
    let mut mtp_inputs = match &head {
        None => None,
        Some(h) => {
            let draft = open_draft(&draft_path, draft_from)?;
            Some(MtpInputs::read(&draft, &split, &inputs, h.rows.clone())?)
        }
    };
    drop(split);
    // The slot count first: it shapes the context search itself, every plan
    // the seat asks for counting each sequence (`Plans::slots`). A set
    // `--ctx-size` with no `--parallel` is one request's context — one slot
    // at the whole of it (`placement::ctx::slots_of`).
    let (slots, from) = model::placement::ctx::slots_of(a.parallel, a.ctx.is_some(), 2)?;
    if from == "ctx" {
        eprintln!(
            "--ctx-size {} is one request's context; add --parallel N to serve N requests at \
             once (they split it)",
            a.ctx.unwrap_or_default()
        );
    }
    // The common rule's choice (the comment at the draft's rule above): the
    // plan of the offer at the seat's own terms — the draft the flag's stage
    // resolved, the slots, the asked-or-default context, which the context
    // search below asks again at the chosen placement — names the tier count
    // the rule holds against its break-even; a refused plan of the offer
    // runs `a`, the refusal named on stderr, and an unset flag never refuses
    // a load that `--place a` serves.
    let chosen = q38place::choose(a.flag.map(|p| p.cards), &census, |offer| {
        let specs = offer.card_specs()?;
        q38place::tier_experts(
            &inputs,
            (specs[0], specs[1]),
            (
                u64::try_from(a.ctx.unwrap_or(CTX))?,
                u64::try_from(ubatch_for(a.ctx.unwrap_or(CTX))?)?,
            ),
            experts,
            &plan_levers,
            mtp_inputs.as_ref(),
            slots,
        )
    })?;
    chosen.record().eprint();
    a.place = Place38 {
        kind: match a.flag {
            Some(f) => f.kind,
            None if chosen.place.tier_cards().is_empty() => Kind38::A,
            None => Kind38::Bp,
        },
        cards: chosen.place,
    };
    let drafted = Plans {
        inputs: &inputs,
        place: a.place,
        experts,
        levers: &plan_levers,
        mtp: mtp_inputs.as_ref(),
        slots,
    };
    let mut rule = ctx38(&drafted, a.ctx)?;
    // The unset draft's yield to the context (the module doc), judged on
    // the drafted search the load just ran — its fit is the positions a
    // slot gets with the draft, and a fit under [`CTX`] served the card
    // rule's own fit exactly. One plain search, the draft-off load's own,
    // decides the rest: it holds more, and the draft's bytes leave a slot
    // under the base the plain rule aims for, so the draft goes off, the
    // plain rule replaces this one and one `draft yield` record names both;
    // a serving cap that binds the two fits together keeps the draft. A set
    // `--ctx-size` or `BLOOMERY_DRAFT` never asks.
    let (plans, mtp, head, draft_off) = if mtp
        && draft_off.is_none()
        && a.ctx.is_none()
        && levers.draft().is_none()
        && rule.search.is_some_and(|s| s.fit.ctx < CTX)
    {
        let plain_plans = Plans {
            inputs: &inputs,
            place: a.place,
            experts,
            levers: &plan_levers,
            mtp: None,
            slots,
        };
        let plain = ctx38(&plain_plans, a.ctx)?;
        let without = plain.search.expect("an unset rule searched").fit.ctx;
        let bytes = mtp_inputs
            .as_ref()
            .expect("a drafted search held the draft")
            .card_bytes_of(u64::try_from(rule.ctx)?, slots)?;
        match bloomery_levers::DraftYield::of(rule.ctx, plain.ctx, CTX.min(without), bytes) {
            Some(y) => {
                record::draft_yield(&y).eprint();
                rule = plain;
                mtp_inputs = None;
                (plain_plans, false, None, Some(Draft38Off::Yield(y)))
            }
            None => (drafted, mtp, head, draft_off),
        }
    } else {
        (drafted, mtp, head, draft_off)
    };
    refuse_mtp_levers(&draft_off)?;
    let (need, pool, arena) = plans.host(rule.ctx, set)?;
    let cache = CacheRam::of_tier(a.cache_ram, need, pool, arena)?;
    let rule = rule.host_bound(&inputs, cache.ram, mtp)?;
    rule.print();
    eprintln!(
        "{} message_start={MESSAGE_START} in_vocab={has_start} in_template={in_template}",
        cache.line()
    );
    // The slots the seat serves ([`Session::add_slots`]): the flag's, one
    // beside a set `--ctx-size`, or 2 — one drafted pass a slot a round,
    // nothing parked.
    eprintln!(
        "parallel rule=slots slots={slots} slot_ctx={} total={} from={from}",
        rule.ctx,
        slots * rule.ctx
    );
    if mtp {
        let be = break_even_of(a.place);
        eprintln!(
            "draft keep place={} break_even={} reply={NOMINAL_REPLY} per_reply_token={} (the \
             kept prefix below which a request that would leave the MTP draft off resets; \
             docs/plan.md, Qwen3.8 serve)",
            a.place.name(),
            be.at(NOMINAL_REPLY),
            be.per_token
        );
    }
    let ctx = rule.ctx;
    // A drafted load's window re-checked at the context the rule chose; a
    // draft the yield turned off needs no window (the plain rule's context
    // is the larger one it fell to).
    if mtp && draft38(&levers, a.place, ctx, &draft_path, &inputs)?.0 != mtp {
        return Err(format!(
            "a slot's context of {ctx} leaves no positions for the MTP draft's window; \
             --ctx-size names a total one past {}",
            <Body38 as MtpBody>::VERIFY_ROWS * slots
        )
        .into());
    }
    let ub = ubatch_for(ctx)?;
    let machine = a.place.machine(
        &inputs,
        (u64::try_from(ctx)?, u64::try_from(ub)?),
        experts,
        mtp_inputs.as_ref(),
        slots,
    )?;
    let (plan, draft_bytes) = match &mtp_inputs {
        None => (
            inputs.plan_with_slots(&machine, u64::try_from(ctx)?, &plan_levers, experts, slots)?,
            0,
        ),
        Some(mi) => {
            let with = inputs.plan_mtp_with_slots(
                &machine,
                u64::try_from(ctx)?,
                &plan_levers,
                mi,
                experts,
                slots,
            )?;
            let bytes = with.draft_card_bytes() + with.arena_bytes;
            (with.plan, bytes)
        }
    };
    let line = Record::new(&record::PLAN38)
        .w("place", a.place.name())
        .w("card", machine.cards[0].name.as_str())
        .w(
            "experts",
            if experts == Experts::Card {
                "card"
            } else {
                "host"
            },
        )
        .u("ctx_max", plan.ctx_max)
        .u("host_experts", plan.host.experts)
        .u("card_experts", plan.cards[0].experts);
    let line = match (machine.tiers.first(), plan.tier_n_l.first()) {
        (Some(t), Some(n)) => line
            .w("tier", t.name.as_str())
            .u("tier_experts", n.iter().sum::<u64>()),
        _ => line,
    };
    let line = match machine.cards.first().and_then(|c| c.free_bytes) {
        Some(free) => line.u("card_free", free),
        None => line,
    };
    line.csv("devices", record::plan_devices(&machine))
        .w("cuda_order", record::cuda_order())
        .eprint();
    let residency = residency38_at(&plan, a.place, set)?;
    let gpu = machine
        .all_cards()
        .map(|c| nvidia_smi_index(&c.name, c.device).map(|i| format!("GPU{i}")))
        .collect::<Result<Vec<_>, _>>()
        .and_then(|g| placement_props(&plan, &g));
    if let Err(e) = &gpu {
        eprintln!("bloomery-serve-qwen38: /props leaves the placement out: {e}");
    }
    let props = EngineProps {
        model: Some(model),
        placement: gpu.ok(),
        ctx_verified: Some(VERIFIED_POSITIONS),
        ..EngineProps::default()
    };

    let open = SeatArgs {
        place: a.place,
        ctx,
        slots,
        experts,
        plan_levers,
        host: levers.host(),
        pin_main: levers.pin_main(),
        path: path.clone(),
        mtp,
        head,
        draft_path,
        draft_from,
        draft_bytes,
        draft_off,
        residency,
        xstream: levers.xstream(),
        stats: levers.step_stats(),
        width,
    };
    let engine = SeatEngine::spawn(
        move || Q38::open(open),
        ctx,
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
    Record::new(&record::LISTENING38)
        .w("place", a.place.name())
        .u("ctx", ctx)
        .u("slots", slots)
        .u("slot_ctx", ctx)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}

/// What the engine thread opens the seat with.
struct SeatArgs {
    place: Place38,
    /// A slot's context: the stores' rows one sequence serves.
    ctx: usize,
    /// The resident sequences the plan counted and the load makes
    /// ([`Session::add_slots`] after the captures).
    slots: usize,
    experts: Experts,
    plan_levers: PlanLevers,
    host: bloomery_levers::HostCfg,
    pin_main: bool,
    path: PathBuf,
    /// `BLOOMERY_DRAFT=mtp`, the MTP draft; unset the plain path.
    mtp: bool,
    /// The draft's head under `mtp` (`head_list::head_rows_of`), which the
    /// load prints.
    head: Option<HeadPick>,
    /// The MTP draft file under `mtp`, and what picked it.
    draft_path: PathBuf,
    draft_from: DraftFrom,
    /// The plan's `draft` class bytes, which `/props` files.
    draft_bytes: u64,
    /// Drafting nothing, why: the `load draft=off` record's.
    draft_off: Option<Draft38Off>,
    /// The residency the load runs ([`residency38`]).
    residency: Residency,
    /// `BLOOMERY_XSTREAM` as set, or unset for the shared rule
    /// ([`xstream38`]).
    xstream: Option<&'static str>,
    /// `BLOOMERY_STEP_STATS`, the round records the seat prints.
    stats: bool,
    /// `BLOOMERY_MTP_WIDTH`: the width a drafted window verifies, the
    /// chooser's (`cost`) or the draft's own (`fixed`).
    width: WidthMode,
}

/// The Qwen3.8 session on the engine thread: the session over the model,
/// one draft a resident slot (its windows of four rows through the
/// runtime's speculative loop — every busy slot's draft reachable in one
/// round, none moved on a [`Q38::select`]), what the open decided about the
/// rounds of several slots, and the positions its stores were sized for.
struct Q38 {
    s: app::Session<Body38>,
    /// One draft a resident slot ([`SlotDrafts`]): a round of several
    /// slots' drafted passes verifies every busy slot's rows in one pass
    /// ([`Q38::pass_slots`]), which needs every busy slot's draft at once,
    /// with no select between. Empty without a draft.
    drafts: SlotDrafts<Body38, { <Body38 as MtpBody>::VERIFY_ROWS }>,
    ctx: usize,
    /// The plan's draft card bytes and arena, for `/props`' `draft` class.
    draft_bytes: u64,
    /// The MTP draft file, which `/props`' `draft` names.
    draft_path: PathBuf,
    /// The load runs the residency machine: each call prints its
    /// boundaries' `residency pass` records.
    residency: bool,
    /// Under the draft, its break-even at the seat's placement.
    break_even: Option<BreakEven>,
    /// Whether a round of several slots' drafted passes runs as one pass of
    /// the busy rows ([`Q38::pass_slots`]): a drafted load of more than one
    /// resident slot with no expert tier card — the body runs several slots'
    /// rows as one pass ([`SlotRows`]) except beside a tier
    /// ([`Place38::tiered`]), and a round of drafted rows only a drafting
    /// server runs. A `--parallel 1` load, a plain one and a tiered one keep
    /// the fallback loop.
    one_pass: bool,
    /// Whether a round of several slots' plain steps runs as one pass of the
    /// busy rows ([`Q38::step_slots`]): a load that drafts nothing, of more
    /// than one resident slot, with no expert tier card. A drafted load's
    /// plain rounds — a sampling or id-banning request's steps — keep the
    /// fallback loop: each of its steps is told to its slot's draft
    /// ([`SlotDrafts::step_with_row`]), which a pass of plain steps does
    /// not do.
    step_pass: bool,
    /// [`Seat::step_stats`]: the `BLOOMERY_STEP_STATS` the binary parsed.
    stats: bool,
    /// The branch the last keep query took ([`Q38::draft_keep`]), one a
    /// slot, printed at that slot's next call; a `RefCell` because the keep
    /// query runs on `&self` (the engine thread owns the seat, so the borrow
    /// never contends).
    branch: std::cell::RefCell<Vec<Option<Branch38>>>,
}

impl Q38 {
    /// The session by `a.place` (the `load` and `capture` lines, as
    /// `generate_qwen3moe` prints them; under `mtp` the draft loaded
    /// beside the target, its own `load draft=mtp` line and the verify
    /// passes' capture lines after them), on the calling thread, pinned
    /// to the dispatcher's cpu slot when asked, its `a.slots` resident
    /// sequences parked after the captures ([`Session::add_slots`]).
    fn open(a: SeatArgs) -> Result<Q38, GateError> {
        const WHAT: &str = "bloomery-serve-qwen38";
        if a.pin_main {
            // The engine thread runs every step; give it the pool's
            // dispatcher slot.
            let _ = threads::pool().pin_caller();
        }
        let t = Instant::now();
        let file = Split::open(&a.path).map_err(|e| format!("open {}: {e}", a.path.display()))?;
        let inputs = PlanInputs::describe(&file)?;
        let cap = serve_ctx(u64::try_from(a.ctx)?, &inputs.hp)?;
        let ub = ubatch_for(a.ctx)?;
        let ctx_ub = (u64::try_from(a.ctx)?, u64::try_from(ub)?);
        let mut m = match a.mtp {
            false => {
                let machine = a.place.machine(&inputs, ctx_ub, a.experts, None, a.slots)?;
                let plan = inputs.plan_with_slots(
                    &machine,
                    u64::try_from(a.ctx)?,
                    &a.plan_levers,
                    a.experts,
                    a.slots,
                )?;
                Body38::open_placed_residency(
                    file,
                    &plan,
                    &inputs,
                    CARD38,
                    a.host,
                    ub,
                    a.residency,
                    a.slots,
                )?
            }
            true => {
                let head = a
                    .head
                    .as_ref()
                    .ok_or("a drafted load with no head picked")?;
                let draft = open_draft(&a.draft_path, a.draft_from)?;
                let mtp = MtpInputs::read(&draft, &file, &inputs, head.rows.clone())?;
                let machine = a
                    .place
                    .machine(&inputs, ctx_ub, a.experts, Some(&mtp), a.slots)?;
                let plan = inputs.plan_mtp_with_slots(
                    &machine,
                    u64::try_from(a.ctx)?,
                    &a.plan_levers,
                    &mtp,
                    a.experts,
                    a.slots,
                )?;
                Body38::open_placed_mtp_residency(
                    file,
                    &plan,
                    &inputs,
                    CARD38,
                    a.host,
                    ub,
                    &draft,
                    &mtp,
                    a.residency,
                    a.slots,
                )?
            }
        };
        m.set_mode(StepMode::Graph);
        eprintln!(
            "load arch=qwen4exp resident_bytes={} ctx={} slots={} slot_ctx={} ctx_max={cap} \
             ctx_train={} verified={VERIFIED_POSITIONS} layers={} mode=graph store_bytes={} \
             prefill=auto ubatch={} place={} card_layers={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            a.ctx,
            a.slots,
            a.ctx,
            inputs.hp.n_ctx_train,
            m.layers().len(),
            m.body(WHAT)?.store_bytes(),
            m.body(WHAT)?.ubatch_rows(),
            a.place.name(),
            m.body(WHAT)?.card_layers(),
            t.elapsed().as_secs_f64()
        );
        // The load's NVMe expert tier, when its plan paged routed experts
        // and the arena was built: the same `nvtier` record
        // `generate_qwen3moe` prints, at load — the arena's budget, the
        // paged bytes and the counters (all zero this early). A load that
        // attached no tier prints nothing, as the CLI's honest absence.
        if let Some(r) = record::nvtier_of(m.body(WHAT)?.nvme_tier().map(|t| &**t)) {
            r.eprint();
        }
        if let Some(why) = &a.draft_off {
            Record::new(&record::LOAD_DRAFT_OFF38)
                .w("why", why)
                .eprint();
        }
        if a.mtp {
            let d = m.body(WHAT)?.mtp().ok_or("the load opened no MTP draft")?;
            let bytes = (d.resident_bytes() + d.arena_bytes()) as u64;
            let head = match d.head_map() {
                Some((_, n)) => format!("rows={n}"),
                None => "full".to_string(),
            };
            eprintln!(
                "load draft=mtp resident={} arena={} head={head} card_bytes={bytes} (the \
                 plan's {}) in {:.1} s (runtime value)",
                d.resident_bytes(),
                d.arena_bytes(),
                a.draft_bytes,
                t.elapsed().as_secs_f64(),
            );
            if let Some(pick) = &a.head {
                Record::new(&record::MTP_HEAD38)
                    .w("head", pick.head_word())
                    .u("rows", pick.rows_of(inputs.spec.vocab))
                    .w("from", pick.why.from_word())
                    .w("why", &pick.why)
                    .eprint();
            }
        }
        let (launches, memops) = m.body(WHAT)?.step_launches();
        let nodes = m.capture_step()?;
        let kernel = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL;
        let memop = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP;
        let ([k, b], other) = count_kinds(&m.step_graph_nodes()?, [kernel, memop]);
        eprintln!(
            "capture graph_nodes={nodes} kernel={k} batch_mem_op={b} other={other} (the \
             program counts {launches}, {memops} of them batch_mem_op)"
        );
        if nodes != launches || b != memops || k + b != nodes || other != 0 {
            return Err(format!(
                "the captured step is not the program's: {nodes} nodes ({k} kernel, {b} \
                 batch_mem_op, {other} other) against {launches} launches, {memops} of them \
                 batch_mem_op"
            )
            .into());
        }
        // Every prompt call copies the recurrent stores at its marks: the
        // checkpoints a later request's cut and a saved state come from.
        m.body_parts(WHAT)?.2.set_checkpoints(true);
        let mut s = app::Session::from_model(m, u32::try_from(a.ctx)?);
        let residency = a.residency != Residency::Off;
        if residency {
            // A request's passes are not known at load: the log grows as it
            // must, and every call takes it.
            s.model_mut().body_parts(WHAT)?.2.log_residency(0);
        }
        // The resident sequences, parked after the captures: the live slot's
        // step and verify chains stay with it, each parked slot capturing on
        // its first use.
        s.add_slots(a.slots)?;
        // One draft a resident slot ([`SlotDrafts`]): every busy slot's
        // draft reachable in one round with no select between.
        let drafts = match a.mtp {
            false => SlotDrafts::none(),
            true => {
                struct Captures;
                impl app::RowsLog for Captures {
                    fn capture_rows(
                        &mut self,
                        rows: usize,
                        nodes: usize,
                    ) -> Result<(), app::SessionError> {
                        eprintln!("capture verify rows={rows} nodes={nodes}");
                        Ok(())
                    }
                }
                SlotDrafts::open(&mut s, a.slots, a.width, &mut Captures, |m| {
                    MtpDraft::open(m, Prompt38::Auto, StepMode::Graph)
                })?
            }
        };
        // The rounds of several slots, decided once: one pass of the busy
        // rows where the body runs one, the drafted rounds on a drafted load
        // and the plain rounds on a plain one.
        let pass_of_slots = a.slots > 1 && !a.place.tiered();
        if a.slots > 1 && a.place.tiered() {
            eprintln!(
                "bloomery-serve-qwen38: --place {}: the rounds of several slots run a slot at a \
                 time: the body runs no pass of several slots beside an expert tier card",
                a.place.name()
            );
        }
        // The expert stream last: a started ring takes the card's free bytes
        // past its keep, so every card buffer the open makes comes before it.
        let (gpu, _, body) = s.model_mut().body_parts(WHAT)?;
        let line = xstream38(gpu, body, a.xstream, a.place.stage38(), a.residency)?;
        eprintln!("{line}");
        Ok(Q38 {
            s,
            drafts,
            ctx: a.ctx,
            draft_bytes: a.draft_bytes,
            draft_path: a.draft_path,
            residency,
            break_even: a.mtp.then(|| break_even_of(a.place)),
            one_pass: pass_of_slots && a.mtp,
            step_pass: pass_of_slots && !a.mtp,
            stats: a.stats,
            branch: std::cell::RefCell::new(vec![None; a.slots]),
        })
    }

    /// The drafted seat's keep rule, its one owner: a prefix of `at`
    /// positions the body's rule keeps, where keeping it leaves the draft
    /// off for the reply — a cut below the held positions, or a draft already
    /// off — is kept only at or past the break-even for the reply's `reply`
    /// tokens through passes (the request's, at most [`NOMINAL_REPLY`], and
    /// that when it bounds nothing); below it the request resets, which turns
    /// the draft back on, and prefills its whole prompt. `None` where the rule
    /// weighs nothing: no draft, nothing kept, or every held position kept
    /// with the draft on.
    fn draft_keep(&self, at: usize, reply: Option<usize>) -> Option<Branch38> {
        let be = self.break_even?;
        let held = self.s.pos() as usize;
        if at == 0 || (at >= held && !self.drafts.is_off(self.s.selected())) {
            return None;
        }
        let reply = reply.map_or(NOMINAL_REPLY, |r| r.min(NOMINAL_REPLY));
        let break_even = be.at(reply);
        Some(Branch38 {
            kept: at >= break_even,
            prefix: at,
            break_even,
            reply,
        })
    }

    /// What the request's reuse left to say, at its first call on the slot
    /// it runs: the `mtp keep` record of the branch its keep took.
    fn before_call(&mut self) {
        self.before_call_at(self.s.selected());
    }

    /// [`Q38::before_call`] at `slot`, a round's row printing at its own
    /// slot as the fallback loop's per-row pass would.
    fn before_call_at(&mut self, slot: usize) {
        if let Some(b) = self.branch.borrow_mut()[slot].take() {
            Record::new(&record::MTP_KEEP38)
                .w("branch", if b.kept { "kept" } else { "reset" })
                .u("prefix", b.prefix)
                .u("break_even", b.break_even)
                .u("reply", b.reply)
                .eprint();
        }
    }

    /// The `call stream` records of the last prompt call — each of its
    /// picks, then its end — and the `residency pass` records of the
    /// boundaries since, on stderr; nothing without the residency.
    fn print_passes(&mut self) -> Result<(), GateError> {
        if !self.residency {
            return Ok(());
        }
        let (picks, end) = self
            .s
            .model_mut()
            .body_parts("bloomery-serve-qwen38")?
            .2
            .take_stream_records();
        for (ubatch, p) in &picks {
            record::call_stream(*ubatch, p).eprint();
        }
        if let Some(r) = end {
            record::call_report(&r).eprint();
        }
        for (kind, r) in self
            .s
            .model_mut()
            .body_parts("bloomery-serve-qwen38")?
            .2
            .take_residency_passes()
        {
            record::residency_pass_of(kind, &r).eprint();
        }
        Ok(())
    }
}

/// The line that says the MTP draft proposes nothing from `pos` on, for
/// `what` (a cut or a state put back).
fn draft_off(pos: u32, what: &str) {
    eprintln!(
        "bloomery-serve-qwen38: the MTP draft proposes nothing from position {pos} until a \
         request starts from position 0: {what}; {DRAFT_OFF_WHY}"
    );
}

/// A Qwen3.8 sequence state as the server's prompt cache holds it, with the
/// draft's side of it ([`SlotDrafts::park`]) beside the body's
/// ([`Seq38`], the draft's rows within it). The cache ranks it by the
/// seat's rule: every position it holds, or the point it carries at or
/// below the shared prefix.
struct Saved38 {
    state: Seq38,
    draft: Option<ParkedDraft<TargetRows>>,
}

impl Saved for Saved38 {
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

impl Seat for Q38 {
    fn pos(&self) -> usize {
        self.s.pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    /// The resident sequences the load made ([`Session::slots`]): one a
    /// `--parallel 1` load, the `--parallel` the seat served past it.
    fn slots(&self) -> usize {
        self.s.slots()
    }

    /// The session's slot ([`Session::select_slot`]): the model exchanges
    /// its live sequence — the draft's device side with it (its store and
    /// the step's and the pass's arena rows, [`Seq38`]) — and the slot's own
    /// draft ([`Q38::drafts`]) needs nothing moved, its host side never
    /// having left the slot, so the slot's next call drafts as if no other
    /// had run. The exchange is refused by name while a verify waits for its
    /// commit (the session's and the body's own checks; nothing has moved
    /// then).
    fn select(&mut self, slot: usize) -> Result<(), GateError> {
        if self.s.selected() == slot {
            return Ok(());
        }
        Ok(self.s.select_slot(slot)?)
    }

    /// The draft's state is per slot ([`Q38::drafts`]): a slot's drafted
    /// passes are the passes it would run alone.
    fn slot_drafts(&self) -> bool {
        self.drafts.drafts()
    }

    /// The prompt through the ubatch walk `--prefill auto` takes: `gemm`
    /// from nine positions on, `pass` below — never one step per id;
    /// under the draft the draft's own prompt call, its store walked over
    /// the prompt's units.
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        self.before_call();
        let sel = self.s.selected();
        let next = self.drafts.prefill(&mut self.s, sel, ids)?;
        self.print_passes()?;
        Ok(next)
    }

    /// One step; under the draft the rows it left waiting walked first
    /// (`MtpDraft::before_step`: a request that continues the held
    /// sequence joins it here when its prompt call is empty).
    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        self.before_call();
        let sel = self.s.selected();
        let next = self.drafts.step(&mut self.s, sel, last)?;
        self.print_passes()?;
        Ok(next)
    }

    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        Ok(self.s.model().logits_into(row)?)
    }

    /// One step and the target's row of it, read before the step is told
    /// to the draft ([`SlotDrafts::step_with_row`]).
    fn step_row(&mut self, last: u32, row: &mut [f32]) -> Result<u32, GateError> {
        self.before_call();
        let sel = self.s.selected();
        let next = self.drafts.step_with_row(&mut self.s, sel, last, row)?;
        self.print_passes()?;
        Ok(next)
    }

    /// The session's reset, the selected slot's draft on again: the
    /// residency stays where use has taken it (only [`Seat::residency_reset`]
    /// moves it back), and the other slots' drafts stand as they are.
    fn reset(&mut self) -> Result<(), GateError> {
        let sel = self.s.selected();
        self.drafts.reset(&mut self.s, sel)
    }

    /// [`app::Session::residency_reset`], its `residency reset` record on
    /// stderr; `None` without the residency (the server's 501).
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

    /// One pass from `last`: under the draft the window of at most its four
    /// rows (the width chooser's cut under `BLOOMERY_MTP_WIDTH=cost`), its
    /// kept tokens and counts; without it one step.
    fn pass(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, GateError> {
        self.before_call();
        let sel = self.s.selected();
        let d = self.drafts.pass(&mut self.s, sel, last, out)?;
        self.print_passes()?;
        Ok(d)
    }

    /// The most positions one pass runs: the draft's four rows (a window
    /// may run narrower under the width chooser), or one step without it.
    fn pass_rows(&self) -> usize {
        self.drafts.pass_rows(self.s.selected())
    }

    /// One sampled pass from `last` ([`SlotDrafts::pass_sampled`]): under
    /// the draft the draft's window of four rows, each row's id the sampler's draw.
    fn pass_sampled(
        &mut self,
        last: u32,
        history: &mut Vec<u32>,
        sampler: &mut serve::Sampler,
        out: &mut Vec<u32>,
    ) -> Result<Drafted, GateError> {
        self.before_call();
        let sel = self.s.selected();
        let d = self
            .drafts
            .pass_sampled(&mut self.s, sel, last, history, sampler, out)?;
        self.print_passes()?;
        Ok(d)
    }

    /// The sampled pass drafts while the MTP draft runs.
    fn drafts_sampled(&self) -> bool {
        self.drafts.drafts()
    }

    /// The lever the binary parsed ([`Q38::stats`]).
    fn step_stats(&self) -> bool {
        self.stats
    }

    /// One round of several slots' plain steps
    /// ([`super::rounds::step_round`]): while the open decided so
    /// ([`Q38::step_pass`], a plain load) one pass of the busy rows through
    /// [`super::rounds::step_rows_one_pass`] — laid in slot order, each
    /// row's answer and lent logits row its own, slot 0 left selected — and
    /// the call's `residency pass` records after it, else the fallback
    /// (`bind::step_rows_in_turn`: a select and a step a row, each step told
    /// to its slot's draft on a drafted load).
    fn step_slots(&mut self, rows: &mut [SlotStep]) -> Result<(), String> {
        let one_pass = self.step_pass;
        super::rounds::step_round(
            self,
            one_pass,
            rows,
            |q, rows| {
                // What the fallback loop's per-row step would print at its
                // slot.
                for r in rows.iter() {
                    q.before_call_at(r.slot);
                }
                super::rounds::step_rows_one_pass(&mut q.s, rows)
            },
            Q38::print_passes,
        )
    }

    /// One round of several slots' drafted passes
    /// ([`super::rounds::pass_round`]): while the open decided so
    /// ([`Q38::one_pass`]) one pass of the busy rows — every slot's window
    /// verified together through [`super::rounds::pass_rows_one_pass`], the
    /// round cut into passes at the body's `SlotRows::MAX_ROWS`, a slot
    /// whose draft a cut turned off stepped alone before them, and what the
    /// fallback loop's per-row `pass` would print at its slot printed first
    /// — else the fallback (a select and a pass a row). The open decides
    /// once, so a load that cannot run one pass never tries it at run time:
    /// a refusal on either path is the server's to die on, never a
    /// fallback. The rows are distinct slots below [`Seat::slots`] (the
    /// engine's own named refusal), a slot whose draft skips rides the pass
    /// as one plain row and an off slot runs its own plain step, so no
    /// row-level fallback exists.
    fn pass_slots(&mut self, rows: &mut [SlotPassRow]) -> Result<(), String> {
        let one_pass = self.one_pass;
        super::rounds::pass_round(
            self,
            one_pass,
            rows,
            |q, rows| {
                // What the fallback loop's per-row `pass` would print at its
                // slot.
                for r in rows.iter() {
                    q.before_call_at(r.slot);
                }
                super::rounds::pass_rows_one_pass(
                    &mut q.s,
                    rows,
                    &mut q.drafts,
                    <Body38 as MtpBody>::WIDTH,
                )
            },
            Q38::print_passes,
        )
    }

    /// The body's rule (`Body38::kept`): every held position, or the
    /// nearest checkpoint at or below `n`, with the rule's sentence when it
    /// keeps less.
    fn keep(&self, n: usize) -> (usize, Option<String>) {
        let pos = self.s.pos() as usize;
        let k = self.s.kept(u32::try_from(n).unwrap_or(u32::MAX));
        let at = k.at as usize;
        (at, (at < n.min(pos)).then(|| k.to_string()))
    }

    /// [`Seat::keep`] under the MTP draft's break-even ([`Q38::draft_keep`]):
    /// a prefix whose keeping leaves the draft off is granted only at or past
    /// it, which a cut then leaves proposing nothing ([`Seat::rollback`]);
    /// below it nothing is granted, with the rule's sentence, and the server
    /// resets. Without the draft, the body's rule.
    fn keep_for(&self, n: usize, reply: Option<usize>) -> (usize, Option<String>) {
        let (at, why) = self.keep(n);
        let branch = self.draft_keep(at, reply);
        self.branch.borrow_mut()[self.s.selected()] = branch;
        match branch {
            Some(b) if !b.kept => (
                0,
                Some(format!(
                    "under the MTP draft a kept prefix of {at} positions is below the \
                     break-even of {} for a reply of {} tokens: the prompt is prefilled whole \
                     with the draft on",
                    b.break_even, b.reply
                )),
            ),
            _ => (at, why),
        }
    }

    /// The body's commit or its cut to a checkpoint ([`Seat::keep`]
    /// granted it); any other position is refused by name, the body's own
    /// message. A cut under the MTP draft prints that the draft proposes
    /// nothing from there ([`DRAFT_OFF_WHY`]).
    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        let held = self.s.pos();
        self.s.model_mut().rollback(pos)?;
        if self.drafts.drafts() && pos < held {
            self.drafts.turn_off(self.s.selected(), DRAFT_OFF_WHY)?;
            draft_off(pos, &format!("a cut back from {held}"));
        }
        Ok(())
    }

    /// The message starts inside the call where both runs keep at least
    /// `Prompt38::GEMM_FROM` ids, so each run is a ubatch walk and the bits
    /// are the uncut call's; under the MTP draft the draft's prompt call
    /// joins each run where the one before left it.
    fn splits(&self, first: usize, end: usize, marks: &[usize]) -> Vec<usize> {
        let mut at = Vec::new();
        let mut last = first;
        for &u in marks {
            if u >= last + Prompt38::GEMM_FROM && u + Prompt38::GEMM_FROM <= end {
                at.push(u);
                last = u;
            }
        }
        at
    }

    /// `/props`' `engine.draft` under the MTP draft: its kind, the draft
    /// file, its width and its resident bytes as the card's `draft`
    /// class.
    fn props(&self, mut p: EngineProps) -> EngineProps {
        if !self.drafts.drafts() {
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
            model: self.draft_path.file_name().map_or_else(
                || self.draft_path.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            ),
            n_max: Some(<Body38 as MtpBody>::WIDTH as u64),
            kind: Some("mtp".to_owned()),
            path: Some(self.draft_path.display().to_string()),
            device: None,
        });
        p
    }

    /// The sequence state (`GpuModel::seq_save`, the draft's rows within it)
    /// as the prompt cache holds it, with the draft's side of it
    /// ([`SlotDrafts::park`]) beside the body's.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
        let sel = self.s.selected();
        Ok(Arc::new(Saved38 {
            state: self.s.model_mut().seq_save()?,
            draft: self.drafts.park(sel)?,
        }))
    }

    /// The state put back (`GpuModel::seq_resume`) after the session's reset,
    /// then the draft's side of it ([`SlotDrafts::unpark`]): the sequence's
    /// next call runs as it would have with no switch between, its draft
    /// joining where it left. Refused by name, before the reset, for a state
    /// this seat did not take or one saved with the draft on put back with it
    /// off (or the other way); the body refuses another model's state or
    /// layout.
    fn resume(&mut self, state: &dyn Saved) -> Result<(), GateError> {
        let saved = state
            .as_any()
            .downcast_ref::<Saved38>()
            .ok_or("a saved state that is not a qwen4exp body's")?;
        let sel = self.s.selected();
        self.drafts.takes(saved.draft.as_ref())?;
        self.drafts.reset(&mut self.s, sel)?;
        self.s.model_mut().seq_resume(&saved.state)?;
        self.drafts.unpark(sel, saved.draft.as_ref())
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
        eprintln!("bloomery-serve-qwen38: {note}");
    }
}
