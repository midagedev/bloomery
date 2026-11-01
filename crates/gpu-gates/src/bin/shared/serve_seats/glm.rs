//! The GLM-5.3-Flash seat — the llama-server-compatible HTTP API on the
//! glm5next engine, opened and stepped as `generate_glm5next` opens and steps
//! the model. `bloomery-serve --model glm` is one call of [`run`], which
//! takes the process's arguments (`--model` already taken out by the
//! one-binary server).
//!
//!     [--host 127.0.0.1] [--port 8080] [--place a|gate|bp|<stage>[+<tier>…]] [--ctx C]
//!     [--alias NAME] [--cache-ram MIB] [--slot-save-path DIR] [--chat-template-file PATH]
//!     [--prefill batch|steps] [--parallel N|auto] [--slot-ctx S] [--queue-depth Q] [--plan]
//!     [--mmproj FILE | --no-mmproj] [--image-min-tokens N] [--image-max-tokens N]
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
//! generation stops there with `truncated`. The context comes first and the
//! plan fills the machine around it (`generate::choose_ctx_of`, every seat's):
//! the model's limit is the trained context (`context_length`) verified up
//! to `place::ORACLE_POSITIONS`, a set share past it stops by name, and one
//! the machine cannot hold stops with the largest it would hold. Unset, a
//! slot asks the limit and, where the plan of every slot is short of the card
//! there, takes the largest context under it, a multiple of 1024 never under
//! 4096, whose plan holds. A `ctx` line on stderr names the rule (`set`,
//! `target` or `fitted`, with the memory the context one step above was short
//! of), the slot context, the slots, their total and the limit. Set, the flag
//! is the total and each slot takes `total / N` positions rounded down
//! (llama-server's `-np N` without `-kvu`) under a set `--parallel N` —
//! alone it is one request's context, one slot at the whole of it
//! (`bind::slots_of`; one line on stderr says so) — a slot
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
//! ([`SlotDrafts::park`]). With the cache on, the copy runs on a stream of its
//! own into pinned staging that the cache's budget pins whole at the open
//! (`seq_save_async`), begun when a request ends and waited for only when the
//! state is read back; a pool with no room is the server's `NoRoom`
//! (`bind::begin`), which makes room or saves on the request path. A returning
//! session's state comes back after the session's reset (`seq_resume`, or
//! `seq_resume_prefix` for the first positions the server keeps, of which
//! only those rows go to the card), its checkpoints the points up to there,
//! its draft joining where it left (`SlotDrafts::unpark`); the residency is
//! the model's and stays where use has taken it, the state carrying no slot
//! map. Its default is the lesser of `bind::CACHE_RAM_CAP` and half of what
//! `MemAvailable` leaves at load past the plan's host need, every slot's
//! checkpoints in it, and the residency's churn pool (`bind::CacheRam`); a `cache`
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
//! by name. `--parallel` naming no count, the seat serves one slot; a count
//! the flags set is never lowered. `--parallel auto` counts the slots once the
//! placement and the draft are chosen at one slot (`bind::AutoSlots::count`):
//! the most, up to `ctx_ask::AUTO_MOST`, whose plan holds one request's context
//! (`--slot-ctx`, or the default's where that is the model's limit). The round's shape the open decided once, from the load and the
//! body's row and sequence limits (`SlotRows::MAX_ROWS`, `SlotRows::MAX_SLOTS`):
//! a plain load (no NextN draft) runs a round of
//! steps as passes of the busy rows
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
//! (`PlanInputs::plan_slots`, `plan_nextn_slots`), so the context choice, the
//! load and `/props`' `vram_kv_bytes` see them all, and a card that cannot
//! hold them is refused by name. Nothing parks — no slot waits for another —
//! so `--park-ram` is refused by name. `--queue-depth Q` bounds the requests
//! that wait for a slot.
//!
//! `--mmproj FILE` (with `--no-mmproj`, `--image-min-tokens` and `--image-max-tokens`:
//! [`vision_seat::Flags`]) gives the seat image input (behind the `vision` feature) from GLM-5.3-Flash's
//! own encoder file (projector type `glm5v`; a limit not given is the file's own budget, 16 to 8000
//! tokens); with no `--mmproj` and a model from `--hf`, the repo's own projector when the plan
//! gives the text model the same experts and context with it as without
//! ([`vision_seat::Flags::encoder`], [`vision_seat::kept`]), unless `--no-mmproj`. The file is held against the text model — its row width, and its three image ids in the
//! vocabulary — before anything is planned ([`vision_seat::open_seat`]). The tower sits where the
//! placement leaves room ([`vision_seat::encoder_seat`]): on a held card the plan leaves off (the 3090
//! under `--place a`, the A6000 under `gate`), else as a named reserve of the stage card (`bp`) that every
//! plan this seat makes counts, the offer's, the context rule's, the printed `plan` line's and the
//! load's, so they all see one stage card. The `vision_seat` record follows the `plan` record, the `vision`
//! record the `load` record. The server
//! renders an image part as `<|begin_of_image|><|image|><|end_of_image|>` and expands the middle id to the
//! merged-token count of the image's grid, positions normal (`serve::media` over [`Seat::media_model`]);
//! a prompt call that holds an image encodes it on the tower and feeds the rows in place of their
//! positions' embeddings ([`Seat::prefill_media`], `bloomery_gpu_glm5next::prefill_media`; the load is
//! set for media). Refused by name: `--mmproj` beside `--prefill steps` (a span's rows enter only
//! through the batched call) and in a build without the `vision` feature. The NextN draft is
//! planned and loaded as without `--mmproj`, and stays on for a request that holds an image: its
//! prompt walk is the text call's over the media call's units (`app::mtp::MtpDraft::prompt_media`,
//! [`SlotDrafts::prefill_media`]), the call a draft prompt of its own that opens its one
//! residency pass. At a span's positions the draft reads the span's row, the row the target
//! consumed there, in place of the image token's embedding, paired with the target's hidden row
//! there (feed B, the reference's: llama.cpp's MTP draft adds the target's vision row at an image
//! position).
//!
//! `BLOOMERY_DRAFT=mtp` loads the file's next-token layer beside the target
//! (`app::arch::glm5next::open_nextn`, the plan `PlanInputs::plan_nextn`
//! makes: the layer's card bytes and arena reserved, each KDA layer's state
//! two lanes) and drives the session through the runtime's speculative loop
//! with the shared window `app::mtp::MtpDraft`, as `generate_glm5next` does:
//! windows of two rows, the target's next token and the draft's one
//! proposal, the draft's walks eager, `pass_rows` 2; the greedy ids are the
//! plain run's of the same load. A sampling or id-banning request drafts
//! too, each kept id its draw from its verified row (`Seat::pass_sampled`
//! alone in the round, its window's rows of the round's one pass beside
//! another busy slot, `serve_seats::rounds::pass_rows_one_pass`), the ids
//! its pass alone draws. A `load draft=mtp` record follows the `load` line, and
//! `/props`' `engine.draft` names the draft (the target file, whose NextN
//! layer it is) with the plan's bytes for it as the card's `draft` class.
//! `BLOOMERY_DRAFT=off` is the plain path, one step a pass, with a `load
//! draft=off (<why>)` record. Unset follows the placement
//! (`bloomery_levers::glm_unset`), its word and why a `draft unset` record
//! before the plan: `mtp` under `--place a` and `bp` on a file of one
//! next-token layer; the plain path under `--place gate`, on a file of other than one,
//! and with stores too short for one window, never a refusal. Unset, the
//! draft then yields to the context, set or not (`bloomery_gpu_gates::generate::draft_yield`,
//! the one rule): when the drafted load fitted its context, or holds not even
//! the floor, and the plain load holds twice as much or more, the draft goes off (any
//! other refusal of the drafted load stops it by name), one `draft
//! yield` record naming its card bytes and the positions a slot gets either
//! way, the plain load's `ctx` line following. Set, refused by
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
//! plan's host headroom less the NextN layer's host experts or what the
//! host's room leaves past the load's host need — never a refusal.
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

use std::num::NonZeroU64;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use app::arch::glm5next::{GlmCfg, open_nextn_slots, open_resident_slots};
use app::mtp::{MtpBody, MtpDraft};
use app::{OpenLog, RowsLog, Session, SessionError};
use bloomery_gpu::checkpoint::saved::SaveLane;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::model::StepMode;
use bloomery_gpu_gates::beside::{self, Beside};
use bloomery_gpu_gates::bind::{
    AutoSlots, CacheRam, Cached, ChatSurface, Landing, Listen, Pass, PutBack, Seat, SeatEngine,
    SlotFlags, SlotPassRow, SlotStep, TemplateFile, Vocab, arm_screen, begin, bind_server,
    eprint_noted, model_props, note_line, parallel_line, placement_of, put_back, slots_given,
    verifies_slots,
};
use bloomery_gpu_gates::generate::{
    DraftYield, Lane, Place, draft_yield, lane_at, mode_name, with_cards,
};
use bloomery_gpu_gates::need;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::residency::PlanRoom;
use bloomery_gpu_gates::residency38::{
    GLM_CARD, check_reserve, checkpoints_reserved, glm_machine, residency_room, residency_set,
};
use bloomery_gpu_gates::side_seat::{self, Side};
use bloomery_gpu_gates::vision_seat::{self, Opened, PleImage, TextFacts};
use bloomery_gpu_gates::{GateError, gpu_census, ref_model_path};
#[cfg(feature = "vision")]
use bloomery_gpu_glm5next::set_prefill_media;
use bloomery_gpu_glm5next::{
    Body, Glm5nextModel, GlmArena, GlmSeq, PrefillMode, checkpoint_budget, checkpoint_price,
    seq_bytes, seq_lane, seq_resume, seq_resume_prefix, seq_save, seq_save_async, seq_save_bytes,
};
use bloomery_levers::{
    ChurnHosting, GlmAt, GlmPick, ResidencyPick, ResidencyWhy, glm_residency_at_plan, glm_unset,
};
use gguf::Split;
use model::arch::glm5next::place::{KdaLanes, NextnInputs, PROMPT_GROUP, PlaceError, PlanInputs};
use model::placement::churn::ChurnPool;
use model::placement::ctx_ask::{DEFAULT_SLOTS, Probe, Rule};
use model::placement::slots::{SplitError, split_ctx};
use model::placement::workstation::{HostNeed, TierBatchBytes};
use model::placement::{Machine, Plan, PlanLevers};
use runtime::Target;
use runtime::prompt::CallEnd;
use runtime::seqstate::{Price, Why};
use runtime::width::Mode as WidthMode;
use serve::flag::number;
#[cfg(feature = "vision")]
use serve::media::{MediaFeed, SharedMediaModel};
use serve::{
    CacheNote, DraftProps, Drafted, EngineProps, Pending, ResidencyReset, Saved, ServeError,
};
use tokenizer::Tokenizer;
use vision::arch::glm5v;

use super::drafted::{ParkedDraft, SlotDrafts};
use crate::glm_place;

/// The seat's name, as its records and errors print it.
const WHAT: &str = "bloomery-serve-glm";

const USAGE: &str = "usage: bloomery-serve [--model glm] [--host H] [--port P] \
                     [--place a|gate|bp|<stage>[+<tier>…]] \
                     [--ctx C] [--alias NAME] [--cache-ram MIB] [--slot-save-path DIR] \
                     [--chat-template-file PATH] [--prefill batch|steps] [--parallel N|auto] \
                     [--slot-ctx S] [--queue-depth Q] [--plan] [--mmproj FILE | --no-mmproj] \
                     [--image-min-tokens N] [--image-max-tokens N] [--api-key KEY] \
                     [--api-key-file FNAME]";

/// `generate_glm5next`'s default context: the unset draft rule reads a
/// slot's window bound at it before the context is chosen.
const CTX: usize = 2048;

/// The levers this seat acts on: those `generate_glm5next` reads for its load
/// and its draft (`BLOOMERY_ROUTE_TRACE` left out — one run's instrument,
/// which a server that serves many prompts does not wire) and
/// `BLOOMERY_STEP_STATS`, which the seat reads to count its rounds of several
/// slots (a `slots round` record each); a lever set outside this list is
/// refused by name at `main`, with `BLOOMERY_PIN_MAIN` for the engine
/// thread's cpu slot, as every serving seat reads it, and `BLOOMERY_HOST_LANES`
/// for the lanes of the host tier's legs on that thread's CCD map, and
/// `BLOOMERY_SCREEN` for the start screen, and `BLOOMERY_SCHEMAJUMP` for how a
/// structured-output request's fixed runs are fed; `gate_glm5next_serve` holds
/// the same list.
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
    bloomery_levers::R8,
    bloomery_levers::PIN_MAIN,
    bloomery_levers::HOST_LANES,
    bloomery_levers::DRAFT,
    bloomery_levers::MTP_WIDTH,
    bloomery_levers::RESIDENCY,
    bloomery_levers::STEP_STATS,
    bloomery_levers::SCREEN,
    bloomery_levers::SCHEMAJUMP,
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
    /// `slots_a_pass`).
    Steps,
    /// A NextN load whose body verifies several slots' drafted windows
    /// together: one pass of the busy slots' windows
    /// (`serve_seats::rounds::pass_rows_one_pass`, cut between windows).
    Windows,
    /// A NextN load whose body does not: a select and a pass a row
    /// ([`pass_rows_in_turn`]).
    Turns,
}

impl Rounds {
    /// The shape of a load that drafts (`drafts`) or not: a drafted round
    /// is one pass when the body verifies several slots' windows of the
    /// seat's depth together (`bind::verifies_slots`: the token at a slot's
    /// position and its proposal of [`MtpBody::WIDTH`] ids, two of them in
    /// its pass of several slots, a partial keep settled at the commit); a
    /// body that does not keeps the fallback loop.
    fn of(drafts: bool) -> Rounds {
        if !drafts {
            Rounds::Steps
        } else if verifies_slots::<Body>(VERIFY_ROWS) {
            Rounds::Windows
        } else {
            Rounds::Turns
        }
    }

    /// The shape the `parallel` line names for `slots` slots: the rounds
    /// that run as one pass are [`Pass::One`].
    fn pass(self, slots: usize) -> Pass {
        Pass::of(slots, self != Rounds::Turns)
    }
}

/// The positions the server's loop needs for one window from an empty
/// model: it takes one only while its rows fit, the first at a one-id
/// prompt's first generated token, 1 + 1 + rows − 1.
const NEED: usize = VERIFY_ROWS + 1;
/// [`NEED`] as the least a slot of a split holds under the draft.
const NEED_FLOOR: NonZeroU64 = NonZeroU64::new(NEED as u64).expect("a window of at least one row");

/// The draft kinds this seat serves without inspecting its NextN layers.
fn check_draft_word(levers: &bloomery_levers::Levers) -> Result<(), GateError> {
    match levers.draft() {
        None | Some("mtp" | "off") => Ok(()),
        Some(other) => Err(format!(
            "BLOOMERY_DRAFT={other}: on a glm5next file mtp drafts the window (the file's NextN \
             layer); lookup needs a load that verifies two rows, which a glm5next load holds \
             (two KDA lanes) only with its NextN layer, and dspark is the V4.1 binaries'"
        )
        .into()),
    }
}

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
    check_draft_word(levers)?;
    match levers.draft() {
        Some("mtp") if ctx < NEED => Err(format!(
            "BLOOMERY_DRAFT=mtp needs --ctx {NEED} or more: token 0 comes out of the feed, and \
             a window's {VERIFY_ROWS} rows run after it (--ctx {ctx})"
        )
        .into()),
        Some("mtp") => Ok(None),
        Some("off") => Ok(Some("BLOOMERY_DRAFT=off".to_owned())),
        Some(_) => unreachable!("the draft word was checked"),
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

/// The target's plan as the load runs it ([`glm_place::plan_of`]), its
/// refusal named for the seat ([`glm_place::named`]).
fn plan_of<'a>(
    inputs: &'a PlanInputs,
    machine: &'a Machine,
    ctx: u64,
    levers: &PlanLevers,
    nextn: Option<&'a NextnInputs>,
    slots: usize,
) -> Result<Plan<'a>, GateError> {
    glm_place::plan_of(inputs, machine, ctx, levers, nextn, slots)
        .map_err(|e| glm_place::named(machine, e))
}

/// What `--mmproj` asks of a placement: the flags, the encoder file they
/// resolve to ([`vision_seat::Flags::encoder`]: `--mmproj`'s, else the `--hf`
/// repo's projector), and the text model's embedding width and vocabulary,
/// which the file is held against.
struct Images<'a> {
    flags: &'a vision_seat::Flags,
    file: Option<PathBuf>,
    n_embd: usize,
    token: &'a dyn Fn(u32) -> Option<String>,
}

impl Images<'_> {
    /// The encoder file opened, held against the text model and seated for
    /// a plan on `place` ([`vision_seat::open_seat`]); `None` without an
    /// encoder file. Where the tower sits depends on the placement, so each
    /// placement the seat plans opens its own.
    fn seat(&self, place: Place) -> Result<Option<Opened>, GateError> {
        let facts = TextFacts {
            n_embd: self.n_embd,
            imrope: None,
            ple: PleImage::Absent,
        };
        vision_seat::open_seat(
            self.flags,
            self.file.as_deref(),
            place,
            &[glm5v::PROJECTOR_TYPE],
            &facts,
            self.token,
        )
    }
}

/// A placement as the seat plans it: the tier's prompt-batch bytes, the
/// machine (carrying the image encoder's reserve when the tower sits on its
/// stage card), the context rule over it, the experts its tier cards hold
/// in the plan at the rule's context (0 with no tier card), the encoder
/// it seats under `--mmproj`.
struct Placed {
    place: Place,
    tier_batch: Option<TierBatchBytes>,
    machine: Machine,
    /// The stage card's reserves the machine carries: the encoder's and the side model's.
    held: Beside,
    /// The context a slot gets, the rule that set it and the choice's `ctx` line
    /// ([`glm_place::serve_ctx`]).
    ctx: usize,
    rule: Rule,
    line: String,
    tier_experts: u64,
    vision: Option<Opened>,
}

/// `place`'s machine for `slots` sequences of `ctx` positions: its checkpoints reserved at the
/// load's `price` ([`glm_machine`]) and the stage card's reserves `held` on it.
fn machine_of(
    (place, tier_batch): (Place, Option<TierBatchBytes>),
    inputs: &PlanInputs,
    (slots, ctx): (usize, u64),
    price: Price,
    held: Beside,
) -> Result<Machine, GateError> {
    Ok(beside::with_reserve(
        glm_machine(place, tier_batch, slots, price, ctx)?,
        held,
    )(inputs.model.layers))
}

impl Placed {
    /// `place` planned for `slots` sequences on the load `nextn` names, its
    /// context `set` or the rule's ([`glm_place::serve_ctx`]), the encoder `images` seat and
    /// the side model's seat counted; every refusal of the plan's by name.
    fn of(
        place: Place,
        inputs: &PlanInputs,
        levers: &PlanLevers,
        nextn: Option<&NextnInputs>,
        (set, slots): (Option<usize>, usize),
        (images, side): (&Images<'_>, Option<&Side>),
    ) -> Result<Placed, GateError> {
        let tier_batch = glm_place::tier_batch(place, inputs)?;
        // Every machine reserves its checkpoints at the slot context it is
        // laid for: the choice's probes each at their own, the load's at the
        // context it is granted; and the encoder's reserve, as the printed plan does.
        let price = checkpoint_price(inputs)?;
        let side_seat = side
            .map(|s| vision_seat::encoder_seat(place, s.bytes))
            .transpose()?;
        let embedding = side_seat.as_ref().and_then(|s| s.reserve);
        // A projector the `--hf` repo picked stays only where the plan gives the text model the
        // same experts and context with its reserve as without ([`vision_seat::kept`]).
        let cost = |v: Option<&Opened>| -> Result<vision_seat::Cost, GateError> {
            let held = Beside {
                image: v.and_then(|v| v.seat.reserve),
                embedding,
            };
            let machine_at =
                |ctx: u64| machine_of((place, tier_batch), inputs, (slots, ctx), price, held);
            let ctx =
                glm_place::serve_ctx(inputs, &machine_at, levers, nextn, (set, slots))?.slot_ctx;
            let machine = machine_at(ctx)?;
            let plan = plan_of(inputs, &machine, ctx, levers, nextn, slots)?;
            Ok(vision_seat::Cost::of(&plan, ctx))
        };
        let vision = vision_seat::kept(images.seat(place)?, images.flags, cost)?;
        let held = Beside {
            image: vision.as_ref().and_then(|v| v.seat.reserve),
            embedding,
        };
        let machine_at =
            |ctx: u64| machine_of((place, tier_batch), inputs, (slots, ctx), price, held);
        let chosen = glm_place::serve_ctx(inputs, &machine_at, levers, nextn, (set, slots))?;
        let ctx = chosen.slot_ctx;
        let machine = machine_at(ctx)?;
        // `glm_place::tier_experts` plans on a machine of its own, which carries no reserve: a
        // stage-reserve seat counts its tier on the machine the plan runs by.
        let tier_experts = if held == Beside::default() {
            glm_place::tier_experts(place, inputs, ctx, levers, nextn, KdaLanes::One, slots)?
        } else {
            plan_of(inputs, &machine, ctx, levers, nextn, slots)?
                .tier_n_l
                .iter()
                .flatten()
                .sum()
        };
        Ok(Placed {
            place,
            tier_batch,
            machine,
            held,
            ctx: usize::try_from(ctx)?,
            rule: chosen.rule,
            line: chosen.line(),
            tier_experts,
            vision,
        })
    }

    /// Whether `slots` sequences of `ctx` positions hold on this placement with its reserves
    /// ([`glm_place::probe`]): `--parallel auto`'s count over the one-slot placement.
    fn holds(
        &self,
        inputs: &PlanInputs,
        levers: &PlanLevers,
        nextn: Option<&NextnInputs>,
        (slots, ctx): (usize, usize),
    ) -> Result<bool, GateError> {
        let ctx = u64::try_from(ctx)?;
        let machine = machine_of(
            (self.place, self.tier_batch),
            inputs,
            (slots, ctx),
            checkpoint_price(inputs)?,
            self.held,
        )?;
        let probe = glm_place::probe(inputs, &machine, (ctx, slots), levers, nextn)?;
        Ok(matches!(probe, Probe::Holds(())))
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
    super::tiered::check_residency_feed(
        levers,
        prefill == PrefillMode::Steps,
        bloomery_levers::UnhostedPool::NotCarried,
    )?;
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
            // The load refuses a host set past the host's available bytes
            // before any upload; the default leaves the pool out instead.
            let room = PlanRoom::new(plan, GLM_CARD, beside);
            let climb = match pick.word {
                "off" => None,
                _ => Some(room.climb_read()?),
            };
            let pick = glm_residency_at_plan(
                <Body as bloomery_gpu::model::ChainBody>::UNHOSTED_POOL,
                pick,
                room.card_experts(),
                |pinned| room.pool_bytes(pinned),
                room.headroom(),
                climb.as_ref().map_or(0, |c| c.mem_left),
            )
            .map_err(|e| format!("BLOOMERY_RESIDENCY unset: the churn pool: {e}"))?;
            record::residency_unset_glm(pick.word, pick.why, climb.as_ref()).eprint();
            (Residency::parse(pick.word)?, pick.word)
        }
    };
    residency_set(
        plan,
        GLM_CARD,
        r,
        ChurnHosting::Hosted,
        word,
        beside,
        Record::eprint,
    )
}

pub(crate) struct Args {
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
    template_file: TemplateFile,
    prefill: PrefillMode,
    /// `--plan`: the records before the load, then exit.
    plan_only: bool,
    /// `--parallel`: the resident sequences the seat serves; `None` takes
    /// one slot beside a set `--ctx`, else `ctx_ask::DEFAULT_SLOTS` (one).
    parallel: Option<usize>,
    /// `--parallel auto`: the seat counts the slots from its plans ([`AutoSlots::count`]);
    /// `parallel` is unset beside it and `ctx` is `--slot-ctx`.
    auto: Option<AutoSlots>,
    queue_depth: Option<usize>,
    /// The image-input flags: `--mmproj` is GLM's own encoder file, whose
    /// images the seat serves; none serves no image.
    vision: vision_seat::Flags,
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
        template_file: TemplateFile::default(),
        prefill: PrefillMode::Batch,
        plan_only: false,
        parallel: None,
        auto: None,
        queue_depth: None,
        vision: vision_seat::Flags::default(),
        api_keys: serve::flag::ApiKeys::default(),
    };
    let mut slots = SlotFlags::default();
    let mut it = args.iter().map(|s| s.as_str());
    while let Some(flag) = it.next() {
        if flag == "--help" || flag == "-h" {
            return Err(USAGE.into());
        }
        if flag == "--plan" {
            a.plan_only = true;
            continue;
        }
        if a.vision.take(flag, &mut it)? {
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
            TemplateFile::FLAG => a.template_file.set(v),
            "--prefill" => {
                a.prefill = PrefillMode::from_name(v)
                    .ok_or_else(|| format!("--prefill is batch or steps, not {v}"))?;
            }
            f if slots.take(f, v)? => {}
            other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
        }
    }
    (a.ctx, a.parallel, a.auto) = slots.asked(a.ctx)?;
    if a.ctx == Some(0) {
        return Err("--ctx 0: the stores hold no position".into());
    }
    if a.vision.mmproj()?.is_some() && a.prefill == PrefillMode::Steps {
        return Err(
            "--mmproj with --prefill steps: a media span's rows enter only through \
                        the batched prompt call (--prefill batch)"
                .into(),
        );
    }
    bloomery_gpu_gates::bind::check_slot_ctx(a.ctx, a.parallel)?;
    Ok(a)
}

/// The explicitly requested context split, including a set MTP draft's floor.
fn checked_ctx(
    a: &Args,
    slots: usize,
    levers: &bloomery_levers::Levers,
) -> Result<Option<usize>, String> {
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
    a.ctx
        .map(|total| {
            let share = split_ctx(total as u64, slots as u64, floor).map_err(|e| match e {
                SplitError::NoSlots { .. } => {
                    format!("--ctx {total}: --parallel 0 splits it among no slot ({why})")
                }
                SplitError::BelowFloor { split, floor, .. } => format!(
                    "--ctx {total}: --parallel {slots} splits it to {split} positions a \
                         slot, under the {floor} a slot needs ({why})"
                ),
            })?;
            usize::try_from(share).map_err(|_| format!("a slot's {share} positions"))
        })
        .transpose()
}

/// The levers (unless `read_levers` is false) and flags [`run`] reads first, checked before the
/// model is fetched ([`bloomery_gpu_gates::bind::check_seat`]).
pub(crate) fn check(
    args: &[String],
    read_levers: bool,
    from_repo: bool,
) -> Result<bloomery_gpu_gates::bind::CheckedSeat<Args>, GateError> {
    bloomery_gpu_gates::bind::check_seat(read_levers.then_some(ACTS_ON), args, |args, levers| {
        let a = parse_args(args)?;
        a.template_file.check()?;
        a.vision.check_before_model(from_repo)?;
        if let Some(levers) = levers {
            WidthMode::of(levers.mtp_width())?;
            check_draft_word(levers)?;
            super::tiered::check_residency_feed(
                levers,
                a.prefill == PrefillMode::Steps,
                bloomery_levers::UnhostedPool::NotCarried,
            )?;
            if levers.draft() == Some("off") {
                bloomery_gpu_gates::bind::refuse_plain_width(levers, Some("BLOOMERY_DRAFT=off"))?;
            }
            let (slots, _) = slots_given(a.parallel, a.ctx, a.auto.is_some(), DEFAULT_SLOTS)?;
            checked_ctx(&a, slots, levers)?;
        }
        Ok(a)
    })
}

/// Loads the model, and the embedding model `side` names beside it, and serves until the listener
/// or the engine fails; `Ok` carries why the server ended.
pub fn run(args: &[String], side: &side_seat::Flags) -> Result<ServeError, GateError> {
    record::at_main_with_flags(WHAT, record::BLOOMERY_SERVE_GLM, |flag| {
        bloomery_gpu_gates::llama_launch::serve_option(args, flag)
    });
    let (a, levers) = check(
        args,
        true,
        bloomery_gpu_gates::model_file::hf_repo().is_some(),
    )?
    .run();
    model::ops::set_host_lanes(levers.host_lanes());
    let width = WidthMode::of(levers.mtp_width())?;

    arm_screen(levers.screen());
    a.vision.wanted()?;
    // The side model is read before the plans, each of which holds its bytes when it sits on the
    // placement's stage card.
    let side = side.read()?;
    // The census the placement resolves against, read once: `--place` as
    // given, or unset the common rule's choice on the same reading.
    let census = gpu_census::census()?;
    let plan_levers = PlanLevers::from_levers(&levers)?;
    let path = ref_model_path()?;
    let vocab = Arc::new(Vocab::new(Tokenizer::from_gguf(&path)?)?);
    let inv = gguf::inventory_of(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let ChatSurface { template, name, .. } = ChatSurface::read(
        &path,
        inv.value("tokenizer.chat_template"),
        inv.value("general.name"),
        a.template_file.path(),
        "glm-5.3-flash",
    )?;
    drop(inv);

    // The plan record, and `/props` from the same plan the load runs by:
    // drafting, the target's plan beside the NextN layer, whose card bytes
    // and arena are the card's `draft` class and whose host experts the
    // load's host set holds beside the plan's.
    let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let inputs = PlanInputs::read(&split)?;
    // `--mmproj`'s encoder file, held against the text model's embedding width and vocabulary
    // before anything is planned; each placement the plans below ask about seats its own tower.
    // A refusal of the file is read here, by name: inside the offer's plan it would read as the
    // tier's plan refusing.
    let token = |id| vocab.tokenizer().text(id).map(str::to_owned);
    let images = Images {
        flags: &a.vision,
        // The `--hf` repo's projector is taken only where a span's rows can enter.
        file: a.vision.encoder(a.prefill != PrefillMode::Steps)?,
        n_embd: inputs.hp.n_embd,
        token: &token,
    };
    images.seat(a.place.unwrap_or(Place::A))?;
    // A slot's share of `--ctx`, refused by name under the body's least a
    // slot before anything is planned: one position, or the MTP draft's
    // window under `BLOOMERY_DRAFT=mtp` (unset, the rule below drafts only
    // where a window fits, and never refuses). The slot count itself: a set
    // `--ctx` with no `--parallel` is one request's context — one slot at
    // the whole of it (`bind::slots_of`).
    let (slots, from) = slots_given(a.parallel, a.ctx, a.auto.is_some(), DEFAULT_SLOTS)?;
    let ctx_share = |slots| checked_ctx(&a, slots, &levers);
    let set = ctx_share(slots)?;
    // The unset rule's window bound reads a slot's floor context — a default
    // is never under it while the card holds it — and the context the rule
    // chooses is asked again below. A default count has no `--ctx`, so what
    // this reads is the same at the one slot the load may fall to.
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
    let plan_place = |p: Place, (set, slots): (Option<usize>, usize)| {
        Placed::of(
            p,
            &inputs,
            &plan_levers,
            nextn.as_ref(),
            (set, slots),
            (&images, side.as_ref()),
        )
    };
    // The placement the seat runs by (`Place::choose`, by the GLM family's
    // `TIER_RULE`): the plan of the offer names its tier count, and an unset
    // flag never refuses a load that `--place a` serves — a refused plan of
    // the offer runs `a`, the refusal named on stderr. Planning the chosen
    // placement again is milliseconds.
    let chosen = Place::choose(a.place, &census, glm_place::TIER_RULE, |p| {
        Ok(plan_place(p, (a.ctx, slots))?.tier_experts)
    })?;
    chosen.record().eprint();
    // The lane's rate, taken before the plan reads the room (`lane_of`).
    // `--plan` returns before any card is opened, so it probes no lane.
    let lane = if a.plan_only {
        None
    } else {
        Some(lane_at(chosen.place)?)
    };
    // `--parallel auto`'s count ([`AutoSlots::count`]) over the placement and the draft chosen
    // at one slot: a drafted default that holds no context at one slot is the yield's below.
    let (slots, asked) = match a.auto {
        None => (slots, a.ctx),
        Some(auto) => auto.count::<PlaceError, _>(
            nextn.is_some(),
            |set| plan_place(chosen.place, (set, 1)).map(|p| (p.ctx, p.rule, p)),
            |one, n, ctx| one.holds(&inputs, &plan_levers, nextn.as_ref(), (n, ctx)),
        )?,
    };
    // An unset draft yields when the plain load serves twice the positions or more.
    // A set context can yield if only its plain plan holds; a set draft never yields.
    // The selected placement and slot count stay fixed.
    let drafted = plan_place(chosen.place, (asked, slots));
    // The draft is unset, so it yields to the context, set or not: the
    // context comes first (`generate::draft_yield`).
    let asks_yield = draft_off.is_none() && levers.draft().is_none();
    let chosen_draft = draft_yield::<PlaceError, _, _>(
        asks_yield,
        drafted,
        |d| matches!(d.rule, Rule::Fitted { .. }).then_some(d.ctx),
        || {
            let plain = Placed::of(
                chosen.place,
                &inputs,
                &plan_levers,
                None,
                (asked, slots),
                (&images, side.as_ref()),
            )?;
            let bytes = nextn
                .as_ref()
                .expect("a drafted load read the NextN layer")
                .card_bytes(u64::try_from(plain.ctx)?, slots)?;
            let ctx = plain.ctx;
            Ok((plain, ctx, bytes))
        },
    )?;
    let (placed, draft_off, yield_record) = match chosen_draft {
        DraftYield::Keep(d) => (d, draft_off, None),
        DraftYield::Yield(plain, y) => (plain, Some(y.to_string()), Some(record::draft_yield(&y))),
    };
    if let Some(said) = yield_record {
        said.eprint();
    }
    // A width lever set on a server that drafts nothing is refused, with
    // why — after the yield said its last word.
    bloomery_gpu_gates::bind::refuse_plain_width(&levers, draft_off.as_deref())?;
    let nextn = if draft_off.is_none() { nextn } else { None };
    let Placed {
        place,
        tier_batch,
        machine,
        ctx: slot_ctx,
        line,
        vision,
        ..
    } = placed;
    let side_seat = side.as_ref().map(|s| s.seat(place)).transpose()?;
    eprintln!("{line}");
    if draft_off.is_none() && slot_ctx < NEED {
        return Err(format!(
            "the --ctx {} leaves no positions for the MTP draft's window (token 0 comes out of \
             the feed, and a window's {VERIFY_ROWS} rows run after it)",
            slot_ctx
        )
        .into());
    }
    let ctx = u64::try_from(slot_ctx)?;
    // The sequences the load makes pin the body's budget at this context;
    // the plan's machine must reserve at least that.
    check_reserve(&machine, slots, checkpoint_budget(&inputs, ctx)?)?;
    let price = checkpoint_price(&inputs)?;
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
    if let Some(v) = &vision {
        vision_seat::seat_record(v).eprint();
    }
    let residency = residency_at(&plan, lever, beside)?;
    let pool = match residency {
        Residency::Mid { pinned, .. } => {
            ChurnPool::of(&plan, GLM_CARD, pinned)
                .map_err(|e| format!("the churn pool: {e}"))?
                .bytes
        }
        Residency::Off => 0,
    };
    // A default too small for one saved state (one position, the KDA
    // layers' two recurrent copies whole, `seq_bytes`) is off.
    let cache = CacheRam::of(
        a.cache_ram,
        HostNeed::of(&plan, beside).bytes(),
        pool,
        checkpoints_reserved(plan.machine),
    )?
    .holding(seq_bytes(&inputs, 1, nextn.is_some()));
    eprint_noted(&cache.line());
    need::print(&plan, beside + pool, cache.ram, true)?;
    // The round's shape the open below runs (the module doc).
    parallel_line(
        slots,
        slot_ctx,
        from,
        Rounds::of(draft_off.is_none()).pass(slots),
    );
    if a.plan_only {
        // The records before the load are out; nothing was opened on a card.
        bloomery_gpu_gates::start::plan_exit();
    }
    let props = EngineProps {
        model: Some(model),
        placement: placement_of(WHAT, &machine, &plan),
        ..EngineProps::default()
    };

    let open = SeatArgs {
        place,
        lane: lane.expect("a serving run probed the lane before the plan"),
        tier_batch,
        ctx: slot_ctx,
        checkpoint_price: price,
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
        cache_ram: cache.ram,
        stats: levers.step_stats(),
        width,
        embedding: side_seat.as_ref().and_then(|s| s.reserve),
        vision,
    };
    let engine = SeatEngine::spawn(
        move || Glm::open(open),
        slot_ctx,
        vocab,
        machine.cards[0].name.clone(),
        props,
        cache.ram,
    )?;
    let side = side_seat::Side::start(side.zip(side_seat))?;

    // The seat's resident slots are the server's, one sequence each: the
    // server selects and steps them together (the engine declares its
    // per-slot draft), no turns and no park.
    let server = bind_server(
        engine,
        Listen {
            host: &a.host,
            port: a.port,
            alias: a.alias.unwrap_or(name),
            path: &path,
            template,
            preparation: None,
            slot_save_path: a.slot_save_path,
            api_keys: a.api_keys,
            parallel: slots,
            queue_depth: a.queue_depth,
            schema_jump: levers.schema_jump(),
            side,
        },
    )?;
    Record::new(&record::LISTENING_GLM)
        .w("place", place.name())
        .u("ctx", slot_ctx)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}

/// What the engine thread opens the seat with.
struct SeatArgs {
    place: Place,
    /// The lane's rate probed before the plan, and the card it ran on: the
    /// `lane` line beside the `load` line.
    lane: Lane,
    /// The expert tier's prompt-batch bytes the plan reserves under a
    /// placement with a tier card.
    tier_batch: Option<TierBatchBytes>,
    /// A slot's context: the stores' rows one sequence serves.
    ctx: usize,
    /// What one sequence's checkpoints cost ([`checkpoint_price`]), which
    /// the load's machine reserves a slot at that context.
    checkpoint_price: Price,
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
    /// The prompt cache's byte budget: the pinned staging its saves are
    /// copied into on a lane of their own, pinned whole at the open
    /// (`SaveLane::new`); 0 (the cache off) makes no lane.
    cache_ram: u64,
    /// Whether the seat counts its rounds of several slots
    /// (`BLOOMERY_STEP_STATS`, the binary's `main` parsed).
    stats: bool,
    /// `BLOOMERY_MTP_WIDTH`: the width a drafted window verifies, the
    /// chooser's (`cost`) or the draft's own (`fixed`).
    width: WidthMode,
    /// The encoder `--mmproj` seated for the placement the plan ran on: its
    /// reserve rides the load's machine, and its tower is loaded after the
    /// model (`open_encoder`).
    vision: Option<Opened>,
    /// The side model's reserve on the stage card, which the plan counted.
    embedding: Option<u64>,
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
    /// The copy stream and the pinned staging the prompt cache's saves are
    /// made on ([`Seat::snapshots_async`]); none with the cache off.
    lane: Option<SaveLane>,
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
    /// The resident image encoder under `--mmproj` ([`open_encoder`]).
    #[cfg(feature = "vision")]
    vision: Option<vision_seat::Resident>,
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
            opened: Duration::ZERO,
            draft_off: a.draft_off.clone(),
            draft_bytes: a.draft_bytes,
            lane: a.lane,
        };
        let prefill = a.cfg.prefill;
        // The plan the open runs by counts the encoder's and the side model's reserves, as the
        // printed plan did.
        let reserve = a.vision.as_ref().and_then(|v| v.seat.reserve);
        let args = app::OpenArgs {
            place: a.place.name(),
            machine: beside::with_reserve(
                glm_machine(
                    a.place,
                    a.tier_batch,
                    a.slots,
                    a.checkpoint_price,
                    u64::try_from(a.ctx)?,
                )?,
                Beside {
                    image: reserve,
                    embedding: a.embedding,
                },
            ),
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
        let lane = match a.cache_ram {
            0 => None,
            ram => Some(seq_lane(s.model_mut(), ram)?),
        };
        // The tower after the model's captures and the slots, as the V4.1 seat opens its encoder.
        #[cfg(feature = "vision")]
        let vision = open_encoder(&mut s, a.vision)?;
        Ok(Glm {
            s,
            drafted,
            ctx: a.ctx,
            draft_bytes: a.draft_bytes,
            path: a.path,
            residency,
            lane,
            rounds: Rounds::of(a.draft_off.is_none()),
            stats: a.stats,
            #[cfg(feature = "vision")]
            vision,
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
            if self.drafted.skipping(r.slot) && !self.drafted.is_off(r.slot) {
                self.drafted.turn_off(r.slot, SKIP_OFF_WHY)?;
            }
        }
        Ok(())
    }
}

/// The resident image encoder of `--mmproj` ([`vision_seat::Resident`]), its tower loaded on the
/// card the plan seated, and the load set for media (the id every position of a span carries is
/// the media description's, [`vision::MediaModel::image_token`]); the `vision` record that follows
/// the `load` record names the card, which of the two seats it is and the tower's bytes.
#[cfg(feature = "vision")]
fn open_encoder(
    s: &mut Session<Body>,
    at: Option<Opened>,
) -> Result<Option<vision_seat::Resident>, GateError> {
    let Some(v) = at else {
        return Ok(None);
    };
    let t = Instant::now();
    set_prefill_media(
        s.model_mut(),
        Some(
            v.media
                .image_token()
                .ok_or("the projector has no chat image token")?,
        ),
    )?;
    let resident = vision_seat::Resident::open(s.model().gpu().context(), &v)?;
    resident.announce(&v, vision_seat::ScratchHome::Own, t);
    Ok(Some(resident))
}

/// The draft's three steps around a GLM state's put-back ([`put_back`]).
impl PutBack for Glm {
    type Body = GlmSeq;
    type Side = Option<ParkedDraft<GlmArena>>;

    fn takes(&self, side: &Self::Side) -> Result<(), GateError> {
        self.drafted.takes(side.as_ref())
    }

    fn reset_selected(&mut self) -> Result<(), GateError> {
        let sel = self.s.selected();
        self.drafted.reset(&mut self.s, sel)
    }

    fn unpark(&mut self, side: &Self::Side) -> Result<(), GateError> {
        self.drafted.unpark(self.s.selected(), side.as_ref())
    }
}

/// Why a slot whose draft skips is off in a round of one pass
/// ([`Glm::off_skipping`]); its next call prints it as an `mtp prompt`
/// record.
const SKIP_OFF_WHY: &str = "the draft skips until the slot's next reset, and a NextN pass of \
                            several slots takes each slot's whole verify";

/// A GLM sequence state as the server's prompt cache holds it ([`GlmSeq`],
/// the body's rule ranking it), with the draft's side of it when the seat
/// drafts: the one common [`Cached`], whose draft side is this seat's.
type SavedGlm = Cached<GlmSeq, Option<ParkedDraft<GlmArena>>>;

/// What a refusal of another seat's state names ([`put_back`]).
const WHOSE: &str = "a glm5next body's";

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
    /// The wall from `t` to the load, once the plan is made: the phases
    /// record's `open_s`.
    opened: Duration,
    /// Drafting nothing, why.
    draft_off: Option<String>,
    /// The plan's bytes for the NextN layer.
    draft_bytes: u64,
    /// The lane's rate probed before the plan, and the card it ran on: the
    /// `lane` line beside the `load` line.
    lane: Lane,
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
        self.opened = self.t.elapsed();
        Ok(true)
    }

    fn load(&mut self, m: &Glm5nextModel) -> Result<(), app::SessionError> {
        let b = m.body(WHAT)?;
        let total = self.t.elapsed();
        let r = Record::new(&record::LOAD_GENERATOR)
            .u("resident_bytes", m.resident_bytes())
            .u("ctx", self.ctx)
            .u("layers", b.kinds().len())
            .u("top_k", self.top_k)
            .w("shadow", "none")
            .u("shadow_bytes", 0)
            .u("unified_addressing", 0);
        record::lane(self.lane.0, &self.lane.1).eprint();
        with_cards(r, self.placement, m.gpu(), b.hybrid().tiers())
            .map_err(app::SessionError::Caller)?
            .w("prefill", self.prefill.name())
            .u("group", self.group)
            .w("mode", mode_name(StepMode::Graph))
            .w("place", self.place)
            .w("pin_main", if self.pin_main { "on" } else { "off" })
            .w("pinned", self.pinned)
            .f("load_s", total.as_secs_f64())
            .eprint();
        record::eprint_load_phases(m, total, self.opened);
        if let Some(h) = b.hybrid().residency() {
            for r in record::host_residency(h) {
                r.eprint();
            }
        }
        // The NVMe expert tier's arena as the load was granted it, beside
        // the one its plan reserved; a load with no tier prints nothing.
        record::host_tier::eprint_nvtier(b.nvme_tier());
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
    /// `slots_a_pass`, the rows' answers and lent logits rows from
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
    /// `slots_a_pass`, slot 0 left selected), each slot whose draft
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
                    1..=<Body as MtpBody>::WIDTH,
                )
            },
            Glm::print_passes,
        )
    }

    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        self.prompt_call(ids, CallEnd::Argmax)
    }

    /// The body's prompt call ([`bloomery_gpu_glm5next::prompt_with`]): the
    /// batched call, its head only when `end` reads it, or one step a
    /// position, as `--prefill` says, each call taking the checkpoints its
    /// marks name; under the draft the draft's own prompt call, its store
    /// walked over the prompt's units ([`SlotDrafts::prompt_call`]). A KDA
    /// layer takes no checkpoint inside a batch, so the seat folds no last
    /// id ([`Seat::prefill_next`]'s default).
    fn prompt_call(&mut self, ids: &[u32], end: CallEnd) -> Result<u32, GateError> {
        let sel = self.s.selected();
        let next = self.drafted.prompt_call(&mut self.s, sel, ids, end)?;
        self.print_passes()?;
        Ok(next)
    }

    /// The resident encoder's media description ([`vision_seat::Resident::media`]); `None`
    /// without `--mmproj`, a seat that takes no image.
    #[cfg(feature = "vision")]
    fn media_model(&self) -> Option<SharedMediaModel> {
        self.vision.as_ref().map(vision_seat::Resident::media)
    }

    /// Each feed's image encoded on the resident encoder and read back as bf16, one row a cell of
    /// its grid and nothing around them ([`vision_seat::Resident::encode`]: the ids carry the
    /// delimiters), then one prompt call with the rows in place of their positions' embeddings
    /// ([`vision_seat::Encoded::prompt_by`]): under the draft the draft's own media prompt call
    /// ([`SlotDrafts::prefill_media`]), its store walked over the call's units like a text
    /// prompt's; without it, or while it is off, the session's ([`app::Session::prompt_media`]).
    /// The encode binds the model's card back before the prompt call.
    #[cfg(feature = "vision")]
    fn prefill_media(&mut self, ids: &[u32], feeds: &[MediaFeed]) -> Result<u32, GateError> {
        let Some(v) = self.vision.as_mut() else {
            return Err("no --mmproj: this seat serves no image".into());
        };
        let encoded = v.encode(
            self.s.model().gpu().context(),
            feeds
                .iter()
                .map(|f| vision_seat::Image::of(f.at, &f.prepared)),
        )?;
        let sel = self.s.selected();
        let next = encoded.prompt_by::<Body, _, _>(
            |rows| rows.media_spans(),
            |spans| self.drafted.prefill_media(&mut self.s, sel, ids, spans),
        )?;
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
        best: Option<usize>,
        out: &mut Vec<u32>,
    ) -> Result<Drafted, GateError> {
        let sel = self.s.selected();
        let d = self
            .drafted
            .pass_sampled(&mut self.s, sel, last, history, sampler, best, out)?;
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

    /// Every slot's draft record printed ([`SlotDrafts::close`]), then the
    /// tier nodes closed.
    fn close(&mut self) -> Result<(), GateError> {
        self.drafted.close();
        Ok(self.s.model_mut().close_nodes()?)
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

    fn snapshot_size(&mut self) -> Result<Option<u64>, GateError> {
        Ok(Some(seq_save_bytes(self.s.model_mut())?))
    }

    /// The sequence state ([`seq_save`]) and, under the draft, its side of
    /// it ([`SlotDrafts::park`]), as the prompt cache holds them.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
        let state = seq_save(self.s.model_mut())?;
        let draft = self.drafted.park(self.s.selected())?;
        Ok(Arc::new(SavedGlm::landed(state, draft)))
    }

    /// Whether the saves copy on the lane: the seat holds one.
    fn snapshots_async(&self) -> bool {
        self.lane.is_some()
    }

    /// The state's copy begun on the lane ([`seq_save_async`]), the draft's
    /// side taken beside it; a pool with no room is the server's `NoRoom`
    /// ([`begin`]). Without a lane, the snapshot on the call.
    fn snapshot_begin(&mut self) -> Result<Pending, GateError> {
        let Some(lane) = &self.lane else {
            return Ok(Pending::landed(self.snapshot()?));
        };
        let copy = seq_save_async(self.s.model_mut(), lane)?
            .map(|(state, pending)| (state, Arc::new(pending) as Arc<dyn Landing>));
        let sel = self.s.selected();
        begin(copy, || self.drafted.park(sel))
    }

    /// The state put back ([`seq_resume`]) after the session's reset, then
    /// the draft's side of it ([`put_back`]).
    fn resume(&mut self, state: &dyn Saved) -> Result<(), GateError> {
        put_back(self, state, WHOSE, |g, body| {
            seq_resume(g.s.model_mut(), body).map(|()| body.positions())
        })
        .map(drop)
    }

    /// The first `n` positions the state can be cut to put back after the
    /// session's reset ([`seq_resume_prefix`]), then the draft's side of it
    /// ([`put_back`]): the sequence's next call runs as it would have
    /// after the whole state's resume and a rollback to the returned
    /// position, its draft joining where the cut leaves it.
    fn resume_prefix(&mut self, state: &dyn Saved, n: usize) -> Result<usize, GateError> {
        let n = u32::try_from(n).unwrap_or(u32::MAX);
        put_back(self, state, WHOSE, |g, body| {
            seq_resume_prefix(g.s.model_mut(), body, n)
        })
    }

    /// A prefix kept less of than shared is a `cache reuse` record; every
    /// other note of the prompt cache prints as its line.
    fn note(note: &CacheNote) {
        note_line(WHAT, note);
    }
}

#[cfg(test)]
mod tests {
    use super::parse_args;
    use model::placement::workstation::{A6000, RTX_3090};

    /// The seat reads `--place` as the shared grammar with the one tier card
    /// the GLM body serves: each word it took before parses to the placement it
    /// names, a list of more tier cards is refused by name at parse, and a
    /// word no card or ordinal spells is too.
    #[test]
    fn the_place_word_is_the_shared_grammar_with_the_glm_tier_count() {
        let args = |word: &str| [String::from("--place"), word.to_owned()];
        for (word, name, cards) in [
            ("a", "a", vec![A6000.name]),
            ("gate", "gate", vec![RTX_3090.name]),
            ("bp", "bp", vec![A6000.name, RTX_3090.name]),
            ("a6000+3090", "bp", vec![A6000.name, RTX_3090.name]),
            ("cuda1+0", "cuda1+0", vec!["cuda1", "cuda0"]),
            ("tp", "tp", vec![A6000.name]),
        ] {
            let a = parse_args(&args(word)).unwrap_or_else(|e| panic!("{word}: {e}"));
            let p = a.place.expect(word);
            assert_eq!((p.name(), p.cards()), (name, cards), "{word}");
        }
        let Err(e) = parse_args(&args("cuda0+cuda1+cuda2")) else {
            panic!("two tier cards parsed");
        };
        assert_eq!(
            e.to_string(),
            "--place cuda0+cuda1+cuda2: 2 expert tier cards, and the glm5next body serves 1 at most"
        );
        let Err(e) = parse_args(&args("b2")) else {
            panic!("b2 is no card or ordinal");
        };
        assert!(e.to_string().starts_with("--place is a, gate, bp"), "{e}");
    }
}
