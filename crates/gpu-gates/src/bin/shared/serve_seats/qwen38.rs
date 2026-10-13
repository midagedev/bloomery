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
//! `placement::ctx::default_slot_ctx_by_card`'s answer over a slot's share
//! (the one rule every serving seat takes, [`CTX`] a slot, steps of
//! [`CTX_STEP`], a margin of `MARGIN` card expert bytes): the card decides it
//! and the host room only vetoes it. The card's context is the largest
//! multiple of [`CTX_STEP`] up to the fit (the largest a slot's N-slot plan
//! holds on the card at an unbounded host room, at most the file's serving
//! cap) whose plan holds at most the plan's own margin (`MARGIN`) fewer card
//! expert bytes than the N-slot plan at [`CTX`] a slot, the fit when every
//! context does — the largest the card holds when that is fewer than
//! [`CTX`]. Where the plan at the room read pages no expert at [`CTX`], the
//! card's context stands only if the plan at the room read pages none there
//! either, else the answer is [`CTX`] (the `host` rule), so a reading of the
//! room never picks a depth between the two; on a machine whose card
//! allocates from the host's memory (`Machine.unified`) the pool is the
//! card's room too, so the answer is instead the first rung of the halving
//! ladder from the serving cap down to [`CTX`] whose plan at the room read
//! stands and is within the margin and pages nothing past the base's (the
//! `pool` rule below the cap's rung), and the reading moves it only where it
//! crosses one rung's need (`placement::ctx::Veto`). Where that plan pages
//! already, the answer is the largest such context whose plan misses the arena
//! on at most a hundredth of the base's missed expert bytes more
//! (`placement::ctx::NVME_SLACK_DIV`). The answer is also at most what lets
//! one session's sequence state with its two recurrent copies fit the prompt
//! cache's budget (one session's whole slot context can be saved): the first
//! rung of the halving ladder from the answer down to [`CTX`] that holds it,
//! and [`CTX`] when none does (the `cache` line then names that its budget
//! holds no session there; the cache keeps the shorter ones). Set, a
//! total a slot's share of which passes the file's
//! `context_length` (`place::serve_ctx`: YaRN scaling past it is not built),
//! then one whose N-slot plan the card cannot hold beside the plan's dense
//! weights — the largest slot context any N-slot plan of the file and the
//! placement takes up to that cap ([`fit38`]) — is refused by name before
//! the load; a set context keeps the MTP draft, so that refusal names the
//! draft's card bytes and the context a slot holds without it. The
//! `load` line prints the stores' `ctx` (a slot's rows) with the slot count
//! and each slot's context, the cap as `ctx_max`, `ctx_train` and
//! `verified`, the deepest context the reference sets hold our numbers to
//! ik's at (`refset::arch::qwen4exp::VERIFIED_POSITIONS`, `/props`'
//! `engine.ctx_verified`), which bounds nothing. A `ctx` line on stderr names
//! the rule (`card`, `base`, `fit`, `margin`, `host` where the host room
//! vetoed the card's context, `pool` where the unified pool's reading took a
//! lower rung, or `cache` where the prompt cache's budget lowered it), the slot context, the slots and the total they serve, the
//! cap, the largest context the card holds with its card expert bytes, and
//! the largest whose plan holds at most the plan's own margin (`MARGIN`)
//! fewer card expert bytes than the plan at [`CTX`] a slot (`margin_ctx`;
//! under `host` the context the room vetoed): more positions on the card push
//! card experts to the host.
//! `--plan` prints the records the decisions before the load make (the `ctx`
//! line, the `plan` and `residency` records) and exits 0 before the route
//! trace's directory is made or any model is loaded.
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
//! the slot: the positional rows (K/V, raw and pooled keys, and under the
//! draft its store's rows) of the held positions, the recurrent stores at
//! the held position and at the last prompt call's end, each with its PLE
//! history, and under the draft the layer's side (`Seq38`): the positions
//! its store holds and the step's and the pass's arena rows with the
//! positions they hold, the draft's waiting rows beside them
//! ([`SlotDrafts::park`]). A returning session's state comes back
//! whole, and its checkpoints are those two points. Its default is the
//! lesser of `bind::CACHE_RAM_CAP` and half of what `MemAvailable` leaves at
//! load past the plan's host need, the residency's churn pool and the
//! checkpoints' host budget, and 0 on a plan the NVMe tier split (the split
//! fills the room) or one too small for a one-position state; a `cache` line
//! on stderr prints it with each term. A state of another model, card,
//! context or store layout (a load with the draft beside one without it) is
//! refused by name; every save,
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
//! one request's context, `bind::slots_of`; one line on stderr says so),
//! one under the route trace (below), and its default two over the automatic
//! context. Under more slots than one
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
//! the split they serve, what set the count (`from`: the `--parallel`
//! flag, a set `--ctx-size`, the default, `route-trace`, or `paged`, the rule
//! below) and the shape of a round of several slots (`pass=one`, or
//! `pass=turns` under `--place bp`, which runs the slots a slot at a time).
//! `--queue-depth Q` bounds the requests that wait for a slot.
//!
//! A plan that pages host experts through the NVMe tier's RAM arena (a room
//! under half the host leg's experts, or `BLOOMERY_NVTIER_BYTES` set:
//! `placement::nvme_arena_of`) serves its slots as one pass over the arena
//! (`bloomery_levers::paged_columns`): the placement's floor counts a column
//! a slot (`ArenaFloor::of`), so the paged experts of every slot's rows are
//! read from the arena, not through the file mapping. A set `--parallel` is
//! served as asked; an unset count falls to one slot (`from=paged`: a second
//! slot's KV takes card bytes that experts would hold, and a later budget
//! rule prices that trade) and an unset draft goes off (`load draft=off
//! (unset: the plan pages …)` — also for a draft whose yield to the context,
//! judged on the plan of several slots the load abandons, had turned it off;
//! whether a drafted verify pays on a paged plan is the machine's drive, not
//! the rule's, to answer, so a set `BLOOMERY_DRAFT=mtp` runs), one `paged`
//! line on stderr naming the arena and what was asked (`asked_draft` the
//! draft before that yield, `draft` what the plan serves); a set count the
//! room's floor cannot hold is refused by name with the room and the floor.
//! Under `--place bp` each slot steps alone (the
//! rounds below), so slots add no column there: they are served as asked, and
//! only an unset draft goes off. The rule reads the plan the flags name — a set
//! context's plain plan before any search, else the plan the default's search
//! and the draft's yield chose — and the load's own plan is held to it: the
//! engine thread plans from the room the main thread read
//! (`PlanInputs::room`, never a reading of its own), and its plan must have
//! the terms the records and rules read (`PlanKey`: the card's and the tiers'
//! experts, the host's experts, resident bytes and tables, the NVMe tier's
//! bytes and arena) or the load is refused by name. The plan's host reserves
//! every slot's checkpoints (the body's budget at the slot's context,
//! [`reserve_checkpoints`]),
//! so the host need, the split's floor and the arena count them, so a default
//! count whose plan the room leaves under the floor is planned at one slot
//! with no draft where that plan pages, under `--place bp` too: that one slot
//! is the room's, not a column's (the same `paged` line, the floor named at
//! its end); a set `--parallel` there is refused by the floor, by name. The
//! residency machine's victim fills read through the NVMe tier's reader
//! while the step's ids stay pinned in the arena, so a `BLOOMERY_RESIDENCY`
//! word set on such a plan is served as given (unset resolves `off`).
//!
//! The MTP draft keeps the same rule past a break-even, and the server cuts
//! its prompt calls at the same message starts (the draft's prompt call joins
//! each run where the one before left it). A state put back whole carries
//! the draft's waiting rows ([`SlotDrafts::park`]) and it drafts on as if no
//! other sequence had run. A cut to a checkpoint, or a state put back as its
//! first positions, rejoins the draft at the point ([`SlotDrafts::rejoin`])
//! where the draft was on and the body kept its side there
//! ([`Body38::draft_rejoins`]: the draft's store holds every position below
//! the point, and the point the target's hidden row before it). Otherwise
//! the draft proposes nothing until a request starts from position 0: the
//! seat turns it off there ([`SlotDrafts::turn_off`]), prints a
//! `bloomery-serve-qwen38: the MTP draft proposes nothing …` line at the
//! cut, and each later prompt call's `mtp prompt` record names why. A
//! request that keeps every held position, or a point the draft rejoins,
//! with the draft on keeps the draft. One whose kept prefix would leave the
//! draft off (a cut to a point the draft cannot rejoin, or a draft already
//! off) keeps it only at or past the
//! break-even (`Q38::draft_keep`): the prefix whose re-prefill costs what the
//! draft saves over the reply — the request's tokens through passes, at most
//! `NOMINAL_REPLY`, that when it bounds nothing — at the placement's prompt
//! rate (<https://github.com/midagedev/rig-log/blob/main/docs/bloomery/plan.md>, Qwen3.8 serve). Below it the server resets, the draft
//! on again, and prefills the whole prompt. A `draft keep` line after the
//! `cache` line prints the break-even, and each such request an `mtp keep`
//! record (`kept` or `reset`, the prefix, the break-even, the reply).

//! `BLOOMERY_DRAFT` unset follows the placement as `generate_qwen3moe`'s
//! does (`bloomery_levers::draft38_unset`): under `--place a` or `bp` the MTP draft
//! runs when a regular file is where it would be opened
//! (`BLOOMERY_MTP_DRAFT`, else the shared draft file beside the target); the
//! plain path runs under `--place gate`, with no file there, and with stores
//! too short for one window (a slot's share of `--ctx-size` under 5: the
//! total under 5 a slot), each printed as a `load draft=off (<why>)` record
//! after the `load` line (`no file at <path>[, <path>…]`, every place the pick looked), never a
//! refusal; `BLOOMERY_DRAFT=mtp` set with stores that short is refused by
//! name before the plan. Unset with the ctx rule's own
//! default, the draft then yields to the context
//! (`bloomery_levers::DraftYield::of`, below at the search): it goes off
//! when the plan with it leaves a slot under the base the plain rule aims
//! for, one `draft yield` record naming its card bytes, the positions a
//! slot gets either way and the base, and the plain rule's `ctx` line
//! following; `BLOOMERY_DRAFT=mtp` set never yields, and a set
//! `--ctx-size` keeps today's answer. `BLOOMERY_DRAFT=off` is the plain
//! path with the same record; `mtp` drafts wherever the draft loads. The
//! CLI's `--logits` condition has no seat equivalent: a request that reads
//! the logits row steps plainly. The route trace's condition is the seat's
//! own (the paragraph on `BLOOMERY_ROUTE_TRACE` below).
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
//! headroom or what the host's room leaves past the plan's host need — never
//! a refusal. Running `mid-p<P>-s<S>` (plain or drafted, either `--place`),
//! the load runs the common residency machine over the card's routed stacks
//! (`Body38::open_placed_residency`, `open_placed_mtp_residency`): a
//! `residency host` record follows the `plan` line, and each call (a prompt,
//! a step, a pass) prints its boundaries' `residency pass` records after it.
//! A request's reset keeps the residency where use has taken it; `POST
//! /residency/reset` moves it back to its seed on a free slot and prints a
//! `residency reset` record (without the residency, the server's 501). The
//! seat's prompt path is `auto` — passes below nine ids, ubatches from nine
//! on, never one step an id — unless `BLOOMERY_PREFILL=steps` feeds one step
//! an id (below), which the machine does not run beside: the unset word
//! resolves `off`, and a set word other than `off` is refused by name before
//! the load, where the body would refuse the first prompt.
//!
//! `BLOOMERY_PREFILL` is `batch` (unset), the `auto` path above, or `steps`,
//! which feeds a prompt call one captured step an id (`Prompt38::Step`; the
//! `load` line prints `prefill=step`) and cuts no prompt call at a message
//! start ([`Seat::splits`]: each cut would be a call of its own). It is
//! refused by name beside the MTP draft, whose prompt call walks the draft
//! over the target's ubatches: `BLOOMERY_DRAFT=off` is the way out.
//!
//! `BLOOMERY_ROUTE_TRACE=<dir>` writes a route trace
//! (`bloomery_gpu::host::route_trace`) of every position the engine runs
//! into `<dir>`, which `main` creates as a new directory before the load: a
//! `call` row for each request's prompt call, the routed ids and slot kinds
//! of every position after it. `tools/ref/route-trace-chat.py` drives the
//! server and seals the set, since a TERM runs no drop. The trace records one
//! sequence's one-row steps under a fixed placement, so it needs
//! `BLOOMERY_PREFILL=steps` and is refused by name beside a set
//! `BLOOMERY_DRAFT`, a `BLOOMERY_RESIDENCY` word other than `off` and a
//! `--parallel` past one; unset, the draft and the residency resolve `off`
//! with their records (`load draft=off`, `residency unset`) and the slots
//! count one (`parallel … from=route-trace`).
//!
//! `BLOOMERY_XSTREAM` resolves by the rule `generate_qwen3moe` runs
//! (`shared/xstream_rule.rs`): set as given; unset `split` under `--place a`
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
//! tier's load settings, `BLOOMERY_PIN_MAIN` with `BLOOMERY_HOST_LANES`,
//! the draft's levers,
//! `BLOOMERY_RESIDENCY`, `BLOOMERY_XSTREAM`, `BLOOMERY_PREFILL`,
//! `BLOOMERY_ROUTE_TRACE` and `BLOOMERY_STEP_STATS` (the `slots round` record
//! a round of several slots prints); the ubatch size
//! (`BLOOMERY_QWEN3_UBATCH`) is read where the load sizes its arena. The
//! stderr lines named above are records of the kinds
//! `bloomery_gpu_gates::record` declares; `--records-schema` prints those
//! kinds and exits.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use app::mtp::{MtpBody, MtpDraft};
use bloomery_gpu::Gpu;
use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for;
use bloomery_gpu::arch::qwen3moe::{
    Body38, Prompt38, Seq38, TargetRows, checkpoint_budget38, checkpoint_price38,
    point_row_budget38, seq38_bytes,
};
use bloomery_gpu::checkpoint::saved::SaveLane;
use bloomery_gpu::host::census::TierCensus;
use bloomery_gpu::host::route_trace::RouteTrace;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::model::StepMode;
use bloomery_gpu_gates::bind::{
    CacheRam, Cached, ChatSurface, Landing, Listen, Pass, PutBack, Seat, SeatEngine, SlotPassRow,
    SlotStep, SlotsFrom, Vocab, arm_screen, begin, bind_server, eprint_noted, model_props,
    note_line, parallel_line, placement_of, put_back, slots_of,
};
use bloomery_gpu_gates::generate::{BreakEven, Lane, Place, lane_of};
use bloomery_gpu_gates::nodes::count_kinds;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::residency38::{
    CARD38, Lever38, Seqs, check_reserve, checkpoints_reserved, reserve_checkpoints, residency38,
};
use bloomery_gpu_gates::{GateError, gpu_census, ref_model_path};
use bloomery_levers::{
    ChurnHosting, Draft38At, Draft38Off, Paged, Residency38At, ResidencyPick, ResidencyWhy,
    draft38_unset, residency38_unset,
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
use model::placement::ctx::{
    ExpertBytes, Refused, SlotCtx, Veto, default_slot_ctx_by_card, fit_of, within_cache,
};
use model::placement::workstation::{CardSpec, HostNeed, HostRead, MARGIN, host_available_read};
use model::placement::{Machine, PlacementError, Plan, PlanLevers};
use refset::arch::qwen4exp::VERIFIED_POSITIONS;
use refset::arch::qwen4exp::mtp::{DraftFrom, draft_file, draft_places};
use runtime::Target as _;
use runtime::seqstate::pinned;
use runtime::width::Mode as WidthMode;
use serve::{
    CacheNote, DraftProps, Drafted, EngineProps, Pending, ResidencyReset, Saved, ServeError,
};
use tokenizer::Tokenizer;

use super::drafted::{ParkedDraft, SlotDrafts};

#[allow(
    clippy::duplicate_mod,
    reason = "the serving binary compiles both seats that plan a qwen35moe file, each including this one file"
)]
#[path = "../plan_room.rs"]
mod plan_room;
#[path = "../qwen38_place.rs"]
mod q38place;
#[path = "../qwen38_levers.rs"]
mod qwen38_levers;
pub(crate) use qwen38_levers::ACTS_ON;
#[path = "../route_trace38.rs"]
mod route_trace38;
use route_trace38::{TraceRun, experts_name};
#[path = "../xstream_rule.rs"]
mod xstream_rule;
use xstream_rule::{Stage, xstream_rule};

const USAGE: &str = "usage: bloomery-serve-qwen38 [--host H] [--port P] [--place a|gate|bp] \
                     [--ctx-size C] [--alias NAME] [--cache-ram MIB] [--chat-template-file PATH] \
                     [--parallel N] [--queue-depth Q] [--park-ram MIB] [--slot-save-path DIR] \
                     [--plan] [--api-key KEY] [--api-key-file FNAME]";

/// The context the default's expert cost is counted against: the plan
/// every Qwen3.8 measurement ran at, and `generate_qwen3moe`'s default.
const CTX: usize = 4096;

/// The default context is a multiple of this many positions.
const CTX_STEP: usize = 256;

/// The token every message of the chat template opens with: a prompt call is
/// cut at the first and the last inside it.
const MESSAGE_START: &str = "<|im_start|>";

/// Why a cut leaves the MTP draft proposing nothing
/// ([`SlotDrafts::turn_off`]): the body did not keep its side at the kept
/// position ([`Body38::draft_rejoins`]), or it was off already.
const DRAFT_OFF_WHY: &str = "the draft rejoins a kept prefix only while it drafts, with its \
                             store holding every position below the prefix's end and the \
                             checkpoint there holding the target's hidden row before it";

/// The reply the break-even weighs ([`Q38::draft_keep`]) when a request
/// bounds nothing, and the most it weighs: the mean greedy reply of the
/// Korean chat prompts the shipped head list was chosen on (<https://github.com/midagedev/rig-log/blob/main/docs/bloomery/plan.md>,
/// Qwen3.8 serve).
const NOMINAL_REPLY: usize = 277;

/// The plain step's and the drafted decode's positions per second, and the
/// drafted prompt's ids per second under plan (a), that the break-even
/// weighs (<https://github.com/midagedev/rig-log/blob/main/docs/bloomery/plan.md>, Qwen3.8 serve).
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
    fn stage38(self) -> Stage {
        match self.kind {
            Kind38::A => Stage::A,
            Kind38::Bp => Stage::Tiered,
            Kind38::Gate => Stage::Other,
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
    /// `plan_mtp_with_slots`. The host reserves every slot's checkpoints
    /// ([`seqs38`] through [`reserve_checkpoints`], priced at `ctx` positions
    /// a slot) and, under the draft, every slot's checkpoint hidden rows
    /// ([`POINT_ROWS_RESERVE`]), which the plan's host need, the NVMe tier's
    /// floor and its arena then count.
    fn machine(
        self,
        inputs: &PlanInputs,
        (ctx, ub): (u64, u64),
        experts: Experts,
        mtp: Option<&MtpInputs>,
        slots: usize,
    ) -> Result<Machine, GateError> {
        let layers = inputs.spec.layers.len();
        let mut machine = match self.kind {
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
        };
        reserve_checkpoints(&mut machine, seqs38(inputs, slots), ctx)?;
        if mtp.is_some() {
            let bytes = point_row_budget38(gdn_layers(inputs), ctx)?;
            machine
                .host
                .reserves
                .push((POINT_ROWS_RESERVE.to_owned(), pinned(slots, bytes)));
        }
        Ok(machine)
    }
}

/// The host reserve row of a load with the MTP draft that carries its
/// checkpoints' hidden rows: each slot's points at most, one row each
/// ([`point_row_budget38`]), beside the checkpoints themselves.
const POINT_ROWS_RESERVE: &str = "mtp point rows";

/// What the seat's load of `slots` resident sequences of the file `inputs`
/// describes declares ([`reserve_checkpoints`]): every slot takes checkpoints,
/// the seat turning them on (`set_checkpoints`), each priced by the body's own
/// derivation ([`checkpoint_price38`]) from the file's delta layers.
fn seqs38(inputs: &PlanInputs, slots: usize) -> Seqs {
    Seqs::priced(slots, checkpoint_price38(gdn_layers(inputs)))
}

/// The delta layers of the file `inputs` describes.
fn gdn_layers(inputs: &PlanInputs) -> usize {
    inputs
        .spec
        .layers
        .iter()
        .filter(|l| matches!(l.mixer, Mixer::DeltaRule(_)))
        .count()
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

/// The positions in all that give each of `slots` resident sequences one MTP
/// window: the server's loop takes one only while its rows fit, the first at
/// a one-id prompt's first generated token — 1 + 1 + rows − 1 positions a
/// slot — and a total split among the slots gives each the floor of its
/// share, so a total of this many or more holds a window a slot.
fn window_positions(slots: usize) -> usize {
    (<Body38 as MtpBody>::VERIFY_ROWS + 1).saturating_mul(slots)
}

/// How the run feeds a prompt and whether it records the routing
/// (`BLOOMERY_PREFILL`, `BLOOMERY_ROUTE_TRACE`): the inputs the unset
/// residency's rule reads beside the placement, and what [`refuse_feed`]
/// holds the rest of the run to.
#[derive(Clone, Copy)]
pub(crate) struct Feed38 {
    /// [`Prompt38::Step`] under `BLOOMERY_PREFILL=steps`, else
    /// [`Prompt38::Auto`]: the ubatch walk, passes below nine ids.
    pub(crate) prompt: Prompt38,
    /// `BLOOMERY_ROUTE_TRACE` is set.
    pub(crate) traced: bool,
}

impl Feed38 {
    fn of(levers: &bloomery_levers::Levers) -> Result<Feed38, GateError> {
        let prompt = match levers.prefill() {
            "steps" => Prompt38::Step,
            "batch" => Prompt38::Auto,
            other => return Err(format!("BLOOMERY_PREFILL={other}: batch or steps").into()),
        };
        Ok(Feed38 {
            prompt,
            traced: levers.route_trace().is_some(),
        })
    }

    /// Each prompt id is a step.
    fn stepped(self) -> bool {
        self.prompt == Prompt38::Step
    }

    /// What decides `BLOOMERY_RESIDENCY` unset before the plan, at `place`.
    fn residency_at(self, place: Place38) -> Residency38At {
        Residency38At {
            qwen38_file: true,
            dump_taps: false,
            place_a: place.stage_a(),
            route_trace: self.traced,
            prefill_step: self.stepped(),
        }
    }
}

/// What the feed and the trace refuse of the run, each by name before the
/// load, with `slots` resident sequences, `mtp` the draft on and `residency`
/// the set residency word when it is not `off`. The trace records one
/// sequence's one-row steps under a fixed placement: it takes no prompt path
/// but the step feed, no MTP draft, no slot beyond the first and no
/// residency machine. The step feed itself runs no MTP draft (its prompt
/// call walks the draft over the target's ubatches) and no residency machine
/// (each prompt id would end a decode pass the rule counts, where a prompt
/// call keeps 0).
pub(crate) fn refuse_feed(
    feed: Feed38,
    (slots, mtp): (usize, bool),
    residency: Option<&str>,
) -> Result<(), GateError> {
    if feed.traced {
        if !feed.stepped() {
            return Err(
                "BLOOMERY_ROUTE_TRACE records one-row steps: set BLOOMERY_PREFILL=steps (a \
                 pass runs a multi-row service and a ubatch a prompt batch, which the trace \
                 does not record)"
                    .into(),
            );
        }
        if mtp {
            return Err(route_trace38::draft_refusal());
        }
        if slots > 1 {
            return Err(format!(
                "BLOOMERY_ROUTE_TRACE records one sequence's steps after its call row; \
                 --parallel {slots} puts another sequence's steps between them"
            )
            .into());
        }
        if let Some(word) = residency {
            return Err(route_trace38::residency_refusal(word));
        }
    }
    if feed.stepped() {
        if mtp {
            return Err(
                "BLOOMERY_PREFILL=steps feeds a prompt one step an id, and the MTP draft's \
                 prompt call walks the draft over the target's ubatches: set \
                 BLOOMERY_DRAFT=off"
                    .into(),
            );
        }
        if let Some(word) = residency {
            return Err(format!(
                "BLOOMERY_RESIDENCY={word} beside BLOOMERY_PREFILL=steps: each prompt id would \
                 end a decode pass the residency rule counts, where a prompt call keeps 0; \
                 unset BLOOMERY_RESIDENCY (or set it to off), or feed the prompt in batches \
                 (BLOOMERY_PREFILL=batch)"
            )
            .into());
        }
    }
    Ok(())
}

/// `BLOOMERY_DRAFT` on the seat at `place` with stores of `ctx` positions
/// in all, split among `slots` resident sequences, `file` the MTP draft file
/// the load would open, `inputs` the target's: `mtp` the draft, `off` the
/// plain path; unset, `generate_qwen3moe`'s rule
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
/// no route trace (`BLOOMERY_ROUTE_TRACE` records one-row steps; a set `mtp`
/// beside it is [`refuse_feed`]'s), and the positions of one window a slot
/// ([`window_positions`]). A set `mtp` meets that condition at the caller,
/// where the context the rule chose is known.
fn draft38(
    levers: &bloomery_levers::Levers,
    place: Place38,
    (ctx, slots): (usize, usize),
    file: &Path,
    looked: &[std::path::PathBuf],
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
                route_trace: levers.route_trace().is_some(),
                file,
                file_is_there: file.is_file(),
                looked,
                need: window_positions(slots),
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

/// The default slot count's plan, refused because the host room is under the
/// NVMe expert tier's floor, answered by the one-column rule's own plan: the
/// first plan the context search asks, at one position (`fit38`), is the
/// asked slots' — its refusal, [`PlacementError::HostRoomFloor`], matched by
/// type through the [`PlaceError`] that [`Plans::expert_bytes`] hands back as it came
/// — and the arena of the one-slot plan with no draft, which the rule loads
/// (`one`), when that plan pages (an arena past 0: the rule's own condition;
/// a plan with none serves as asked, so the refusal stands for [`ctx38`] to
/// name). The floor counts every slot's checkpoints ([`reserve_checkpoints`]),
/// so a room between the one-slot floor and the asked slots' is one only the
/// one-slot plan clears. That one slot is the room's, so it holds under
/// `--place bp` too, where the one-column rule alone keeps every slot. Any
/// other refusal of the asked plan is `ctx38`'s, and a one-slot plan that is
/// refused too is named by the context search of `one`.
fn floor_fallback(
    asked: &Plans<'_>,
    set: Option<(Residency, &str)>,
) -> Result<Option<u64>, GateError> {
    let Err(refused) = asked.expert_bytes(1) else {
        return Ok(None);
    };
    let Some(PlaceError::Placement(PlacementError::HostRoomFloor { .. })) =
        refused.downcast_ref::<PlaceError>()
    else {
        return Ok(None);
    };
    let one = Plans {
        mtp: None,
        slots: 1,
        ..*asked
    };
    let rule = ctx38(&one, None)?;
    let arena = one.host(rule.ctx(), set)?.2;
    Ok((arena > 0).then_some(arena))
}

/// The residency the load of `plan` at `place` runs: `set` (the word and
/// its parse) as given; unset, the Qwen3.8 rule's
/// ([`residency38`]: before the plan `off` under `--place gate`, beside the
/// route trace and under the step feed, else on plan (a) from it), its
/// records on stderr.
fn residency38_at(
    plan: &Plan<'_>,
    (place, feed): (Place38, Feed38),
    set: Option<(Residency, &str)>,
) -> Result<(Residency, ChurnHosting), GateError> {
    let lever = match set {
        Some((r, word)) => Lever38::Set(r, word),
        None => Lever38::Unset(residency38_unset(feed.residency_at(place))),
    };
    residency38(
        <Body38 as bloomery_gpu::model::ChainBody>::UNHOSTED_POOL,
        plan,
        lever,
        Record::eprint,
    )
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
    /// The prompt feed and the trace, which the unset residency's pool reads.
    feed: Feed38,
}

impl Plans<'_> {
    /// How far a default the host room vetoed falls on this placement's
    /// machine ([`Veto::of`]), read at the base: the machine's card either
    /// has memory of its own or shares the host's, whatever the context.
    fn veto(&self) -> Result<Veto, GateError> {
        let ub = ubatch_for(CTX)?;
        let machine = self.place.machine(
            self.inputs,
            (u64::try_from(CTX)?, u64::try_from(ub)?),
            self.experts,
            self.mtp,
            self.slots,
        )?;
        Ok(Veto::of(&machine))
    }

    /// The plan's routed-expert bytes by tier at `ctx` positions a sequence
    /// serves — the stage card's and the NVMe tier's, what the default rule
    /// counts — or why no plan takes that context.
    fn expert_bytes(&self, ctx: usize) -> Result<ExpertBytes, GateError> {
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
        Ok(ExpertBytes {
            card: plan
                .cards
                .first()
                .ok_or("a plan with no card")?
                .expert_bytes,
            nvme: plan.host.nvme_expert_bytes,
            arena: plan.host.nvme_arena_bytes,
        })
    }

    /// The plan's host need (`HostNeed::bytes`, the NVMe expert tier's
    /// arena inside it) at `ctx`, the churn pool the residency it runs
    /// holds beside it (`set` the lever as `run` read it; the rule's
    /// records unprinted), and the arena itself
    /// (`HostTotals::nvme_arena_bytes`, 0 without one) for the `cache`
    /// line's `tier` term — [`CacheRam::of_tier`] counts it once, inside
    /// the need — the checkpoint bytes the machine reserves
    /// ([`checkpoints_reserved`]: every slot's, inside the need), and the
    /// routed-expert bytes the host leg reads from the NVMe tier
    /// (`HostTotals::nvme_expert_bytes`, 0 on a plan it did not split).
    fn host(
        &self,
        ctx: usize,
        set: Option<(Residency, &str)>,
    ) -> Result<(u64, u64, u64, u64, u64), GateError> {
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
            None => Lever38::Unset(residency38_unset(self.feed.residency_at(self.place))),
        };
        // An unhosted pool takes no host bytes: the load only knows it.
        let pool = match residency38(
            <Body38 as bloomery_gpu::model::ChainBody>::UNHOSTED_POOL,
            &plan,
            lever,
            |_| {},
        )? {
            (Residency::Mid { pinned, .. }, ChurnHosting::Hosted) => {
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
            checkpoints_reserved(&machine),
            plan.host.nvme_expert_bytes,
        ))
    }
}

/// The host bytes a sequence state of `positions` positions holds on the
/// file `inputs` describes ([`Seq38`]: its positional rows and two recurrent
/// copies, the draft's side under `drafts`, [`seq38_bytes`]) — at one
/// position the least one saved state of the prompt cache holds.
fn seq38_state(inputs: &PlanInputs, positions: usize, drafts: bool) -> u64 {
    let gdn = gdn_layers(inputs);
    seq38_bytes(gdn, inputs.spec.layers.len() - gdn, positions, drafts)
}

/// The context the seat loads ([`ctx38`]) and what decided it: every number
/// a slot's own — the stores' rows one sequence serves — the total they hold
/// `slots` times over.
enum Ctx38 {
    /// `--ctx-size`: a slot's share of the flag, the plan's card expert
    /// bytes there. The default's search probes contexts the flag did not
    /// name, so a set context runs none.
    Set {
        ctx: usize,
        slots: usize,
        card_bytes: u64,
    },
    /// The unset default: the common rule's answer ([`ctx38`]), which the
    /// prompt cache's bound may lower ([`Ctx38::host_bound`]) — `rule` and
    /// `chosen` then name the bound and the plan at the lowered context.
    Rule { found: SlotCtx, slots: usize },
}

/// The refusal of a probe the common rule's searches ran, as this seat
/// names it: a tier card the plan leaves idle is refused as itself
/// (`PlacementError::IdleTier`): no context gives it an expert.
fn refusal38(refused: Refused<GateError>) -> GateError {
    match refused.source.downcast_ref::<PlaceError>() {
        Some(PlaceError::Placement(PlacementError::IdleTier { .. })) => refused.source,
        _ => refused.into(),
    }
}

/// The most a slot serves: the file's serving cap (`place::serve_ctx`: its
/// `context_length`).
fn cap38(plans: &Plans<'_>) -> Result<usize, GateError> {
    Ok(usize::try_from(serve_ctx(1, &plans.inputs.hp)?)?)
}

/// The largest slot context any plan of the file and the placement takes —
/// every plan counting the seat's slots — up to the file's serving cap: the
/// most a `--ctx-size`'s split may give a slot.
fn fit38(plans: &Plans<'_>) -> Result<usize, GateError> {
    let (fit, _) = fit_of(cap38(plans)?, &|c| plans.expert_bytes(c)).map_err(refusal38)?;
    Ok(fit)
}

/// The seat's context (the module doc), every number a slot's own: `set`
/// when the file's serving cap takes a slot's share of the total
/// (`place::serve_ctx`, refused by name past it) and it fits the card
/// ([`fit38`]), else refused by name; unset, the common rule's answer
/// (`placement::ctx::default_slot_ctx_by_card`: [`CTX`] a slot, [`CTX_STEP`],
/// a margin of `MARGIN` card expert bytes) over two probes of the same plan,
/// the card's at an unbounded host room ([`plan_room::unbounded`]) and the
/// load's at the room read. The prompt cache's bound comes after ([`Ctx38::host_bound`]).
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
        return match plans.expert_bytes(ctx) {
            Ok(bytes) => Ok(Ctx38::Set {
                ctx,
                slots: n,
                card_bytes: bytes.card,
            }),
            Err(refused) => {
                let fit = fit38(plans)?;
                if ctx <= fit {
                    return Err(refused);
                }
                let head = format!(
                    "--ctx-size {c}: with --parallel {n} each slot takes {ctx} positions and the \
                     card holds at most {fit} a slot beside the plan's dense weights"
                );
                let place = plans.place.name();
                // A set context keeps the MTP draft (the unset one yields to
                // the context): the refusal names it, and what a slot holds
                // without it.
                let Some(m) = plans.mtp else {
                    return Err(format!("{head} (`--place {place}`)").into());
                };
                let bytes = m.card_bytes_of(u64::try_from(ctx)?, n)?;
                let without = fit38(&Plans {
                    mtp: None,
                    ..*plans
                })?;
                let tail = if ctx <= without {
                    format!(
                        "; without the draft a slot holds {without}: set BLOOMERY_DRAFT=off to serve \
                         {ctx} a slot, or leave --ctx-size unset (the unset context turns the draft \
                         off when it costs positions)"
                    )
                } else {
                    format!(", and {without} without the draft")
                };
                Err(
                    format!("{head} and the MTP draft's {bytes} B (`--place {place}`){tail}")
                        .into(),
                )
            }
        };
    }
    // The card decides, the host room only vetoes: the plan at an unbounded
    // room is the card's probe, the plan at the room read the veto's. How far
    // a veto falls is the placement's fact (`Veto::of`).
    let free = plan_room::unbounded(plans.inputs);
    let card = Plans {
        inputs: &free,
        ..*plans
    };
    let found = default_slot_ctx_by_card(
        cap38(plans)?,
        CTX,
        CTX_STEP,
        MARGIN,
        plans.veto()?,
        &|c| card.expert_bytes(c),
        &|c| plans.expert_bytes(c),
    )
    .map_err(refusal38)?;
    Ok(Ctx38::Rule { found, slots: n })
}

impl Ctx38 {
    /// A slot's context.
    fn ctx(&self) -> usize {
        match self {
            Ctx38::Set { ctx, .. } => *ctx,
            Ctx38::Rule { found, .. } => found.ctx,
        }
    }

    /// The largest slot context the card holds, which the default's search
    /// found; `None` for a set context, which ran none.
    fn fit(&self) -> Option<usize> {
        match self {
            Ctx38::Set { .. } => None,
            Ctx38::Rule { found, .. } => Some(found.fit),
        }
    }

    /// A default bounded by the prompt cache too: the first rung of the
    /// halving ladder from the rule's context down to [`CTX`] whose one
    /// session's state at its slot's context ([`seq38_state`]) fits the
    /// budget `ram_at` gives there, when the cache is on
    /// ([`model::placement::ctx::within_cache`]: a unified pool's need moves
    /// with the context, so the budget is read at the context bounded, not at
    /// the one the rule read), and the plan's bytes at the lowered context are
    /// `plans`' ([`Plans::expert_bytes`]). A set context keeps its value, as
    /// does a default whose budget is 0 at the rule's context (the cache is
    /// off: `--cache-ram 0`, or the NVMe tier's split fills the room); a set
    /// budget too small for one position's state is refused by name (a
    /// default one is 0 already, [`CacheRam::holding`]). The second term is
    /// whether the budget holds one session at the context returned: false
    /// when not even [`CTX`]'s does, which the cache line says
    /// ([`CacheRam::short_of`]) — the cache keeps the shorter sessions it
    /// holds.
    fn host_bound(
        self,
        plans: &Plans<'_>,
        ram_at: &dyn Fn(usize) -> Result<u64, GateError>,
        drafts: bool,
    ) -> Result<(Ctx38, bool), GateError> {
        let Ctx38::Rule { mut found, slots } = self else {
            return Ok((self, true));
        };
        if ram_at(found.ctx)? == 0 {
            return Ok((Ctx38::Rule { found, slots }, true));
        }
        let state = |n: usize| seq38_state(plans.inputs, n, drafts);
        let bound = within_cache(found.ctx, CTX, CTX_STEP, &state, ram_at)?;
        if bound.ctx != found.ctx {
            found.ctx = bound.ctx;
            found.rule = "cache";
            found.chosen = plans.expert_bytes(bound.ctx)?;
        }
        Ok((Ctx38::Rule { found, slots }, bound.holds))
    }

    /// The `ctx` line on stderr: the common rule's ([`SlotCtx::line`]); a
    /// set context's line carries no search terms (it ran none).
    fn print(&self) {
        match self {
            Ctx38::Set {
                ctx,
                slots,
                card_bytes,
            } => eprintln!(
                "ctx rule=set ctx={ctx} slots={slots} total={} card_expert_bytes={card_bytes}",
                slots * ctx
            ),
            Ctx38::Rule { found, slots } => eprintln!("{}", found.line(*slots)),
        }
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
    /// (`bind::slots_of`).
    parallel: Option<usize>,
    queue_depth: Option<usize>,
    /// `--park-ram` in bytes; set is refused by name (resident slots park
    /// nothing).
    park_ram: Option<u64>,
    /// `--api-key`/`--api-key-file`: the keys every request is checked
    /// against ([`serve::flag::ApiKeys`]).
    api_keys: serve::flag::ApiKeys,
    /// `--plan`: the records before the load, then exit.
    plan_only: bool,
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
        plan_only: false,
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
    model::ops::set_host_lanes(levers.host_lanes());
    record::at_main("bloomery-serve-qwen38", record::BLOOMERY_SERVE_QWEN38);
    super::drafted::serves_widths();
    arm_screen(levers.screen());
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
        Some(word) => Some((Residency::parse_asking(word)?, word)),
        None => None,
    };
    let feed = Feed38::of(&levers)?;
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
    let draft_looked = draft_places(&path);
    let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    // The stage card's first open is made before the room is read
    // (`Gpu::hold_card`: its context and the bundle), so the plans' room has
    // lost its host bytes as the load's check after its open has; plan
    // (b′)'s tier card is held once chosen, below.
    let stage = draft_at.cards.on(&census)?.card_specs()?[0];
    Gpu::hold_card(stage.name, stage.device)?;
    // The lane's rate, taken before the plan reads the room (`lane_of`).
    let lane = lane_of(stage)?;
    let mut inputs = PlanInputs::describe(&split)?;
    // The slot count first: it shapes the draft's window condition, the
    // context search itself, and every plan the seat asks for, each counting
    // every sequence (`Plans::slots`). A set `--ctx-size` with no
    // `--parallel` is one request's context — one slot at the whole of it
    // (`bind::slots_of`).
    // The route trace records one sequence's steps, so its default is one.
    let (slots, from) = match slots_of(a.parallel, a.ctx.is_some(), 2)? {
        (_, SlotsFrom::Default) if feed.traced => (1, SlotsFrom::RouteTrace),
        given => given,
    };
    // The draft's one context condition (a window's positions a slot) holds
    // at any context the default rule grants, which each slot gets [`CTX`]
    // of; a set `--ctx-size` is the total the slots split. It is asked again
    // at the final context below.
    let (mtp, draft_off) = draft38(
        &levers,
        draft_at,
        (a.ctx.unwrap_or(CTX.saturating_mul(slots)), slots),
        &draft_path,
        &draft_looked,
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
    let ChatSurface { template, name } = ChatSurface::read(
        &path,
        inv.value("tokenizer.chat_template"),
        inv.value("general.name"),
        a.template_file.as_deref(),
        "qwen3.8",
    )?;
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
    if from == SlotsFrom::Ctx {
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
            seqs38(&inputs, slots),
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
    // Plan (b′)'s tier cards held too, and the room read again: every plan
    // below and the engine thread's take it.
    let tiers = &a.place.cards.card_specs()?[1..];
    if !tiers.is_empty() {
        for t in tiers {
            Gpu::hold_card(t.name, t.device)?;
        }
        inputs.room = host_available_read()?;
    }
    // The paged tier's rule (`bloomery_levers::paged_columns`, the module
    // doc), asked once of the plan the flags name: a set context's slot is
    // known before any search, so its plain plan's arena decides ahead of the
    // drafted plan; the default's is asked below, of the plan its search and
    // the draft's yield chose. Under the rule the plan serves a set draft, an
    // unset one off, a set `--parallel` as asked, and a default count of
    // slots that step together at one slot (`from=paged`); beside an expert
    // tier card each slot steps alone ([`Place38::tiered`], the rounds' own
    // rule below), and the slots stay.
    let paged_at = |arena: u64, slots: usize, mtp: bool| bloomery_levers::PagedAt {
        arena,
        slots,
        slots_set: from == SlotsFrom::Flag,
        together: !a.place.tiered(),
        draft: mtp,
        draft_set: levers.draft().is_some(),
    };
    let paged = |arena: u64, slots: usize, mtp: bool| {
        bloomery_levers::paged_columns(&paged_at(arena, slots, mtp))
    };
    // The draft a reduced load runs ([`bloomery_levers::paged_draft`]): a set
    // one runs, and an unset one the plan would draft — or that only its
    // yield to the context turned off on the plan this load abandons — goes
    // off naming the paged rule, at either point.
    let reduced = |arena: u64,
                   (slots, served): (usize, usize),
                   mtp: bool,
                   draft_off: Option<Draft38Off>,
                   floor: bool| {
        let (drafts, draft_off) =
            bloomery_levers::paged_draft(&paged_at(arena, slots, mtp), draft_off);
        eprintln!(
            "paged rule=reduced arena={arena} asked_slots={slots} asked_draft={} \
             slots={served} draft={} (the plan pages host experts through the NVMe tier's RAM \
             arena; {}{})",
            if mtp { "on" } else { "off" },
            if drafts { "on" } else { "off" },
            if served < slots {
                "an unset count serves one slot: a second slot's KV takes card bytes that \
                 experts would hold, and a later budget rule prices that trade"
            } else {
                "a set count is served as one pass over the arena"
            },
            if floor {
                "; the asked slots' plan leaves the room under the tier's floor"
            } else {
                ""
            }
        );
        (
            if served < slots {
                SlotsFrom::Paged
            } else {
                from
            },
            drafts,
            draft_off,
        )
    };
    let early = match a.ctx.map(|c| c / slots) {
        Some(ctx) if ctx > 0 && (slots > 1 || mtp) => {
            let plain = Plans {
                inputs: &inputs,
                place: a.place,
                experts,
                levers: &plan_levers,
                mtp: None,
                slots,
                feed,
            };
            if plain.expert_bytes(ctx).is_ok() {
                plain.host(ctx, set)?.2
            } else {
                0
            }
        }
        _ => 0,
    };
    let (slots, from, mtp, head, draft_off) = match paged(early, slots, mtp) {
        Paged::AsAsked => (slots, from, mtp, head, draft_off),
        Paged::Reduced { slots: served } => {
            let (from, drafts, draft_off) = reduced(early, (slots, served), mtp, draft_off, false);
            if !drafts {
                mtp_inputs = None;
            }
            (
                served,
                from,
                drafts,
                if drafts { head } else { None },
                draft_off,
            )
        }
    };
    // A default count whose plan the room leaves under the tier's floor
    // ([`floor_fallback`]) falls to one slot, and no draft where unset, on
    // the plan that pages. The one slot is the room's, not a column's, so it
    // holds beside a tier card too, where each slot steps alone and the
    // column rule keeps them (`bloomery_levers::paged_floor`).
    let floor = if from == SlotsFrom::Default && slots > 1 {
        let asked = Plans {
            inputs: &inputs,
            place: a.place,
            experts,
            levers: &plan_levers,
            mtp: mtp_inputs.as_ref(),
            slots,
            feed,
        };
        floor_fallback(&asked, set)?
    } else {
        None
    };
    let (slots, from, mtp, head, draft_off) = match floor {
        Some(arena) => match bloomery_levers::paged_floor(&paged_at(arena, slots, mtp)) {
            Paged::AsAsked => (slots, from, mtp, head, draft_off),
            Paged::Reduced { slots: served } => {
                let (from, drafts, draft_off) =
                    reduced(arena, (slots, served), mtp, draft_off, true);
                if !drafts {
                    mtp_inputs = None;
                }
                (
                    served,
                    from,
                    drafts,
                    if drafts { head } else { None },
                    draft_off,
                )
            }
        },
        None => (slots, from, mtp, head, draft_off),
    };
    let drafted = Plans {
        inputs: &inputs,
        place: a.place,
        experts,
        levers: &plan_levers,
        mtp: mtp_inputs.as_ref(),
        slots,
        feed,
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
        && rule.fit().is_some_and(|fit| fit < CTX)
    {
        let plain_plans = Plans {
            inputs: &inputs,
            place: a.place,
            experts,
            levers: &plan_levers,
            mtp: None,
            slots,
            feed,
        };
        let plain = ctx38(&plain_plans, a.ctx)?;
        let without = plain.fit().expect("an unset rule searched");
        let bytes = mtp_inputs
            .as_ref()
            .expect("a drafted search held the draft")
            .card_bytes_of(u64::try_from(rule.ctx())?, slots)?;
        match bloomery_levers::DraftYield::of(rule.ctx(), plain.ctx(), CTX.min(without), bytes) {
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
    let host = plans.host(rule.ctx(), set)?;
    let (slots, from, mtp, head, draft_off, host, one) = match paged(host.2, slots, mtp) {
        Paged::AsAsked => {
            if host.2 > 0 && slots > 1 {
                eprintln!(
                    "paged rule=served arena={} asked_slots={slots} slots={slots} (the plan \
                     pages host experts through the NVMe tier's RAM arena; a set count is \
                     served as one pass over it)",
                    host.2
                );
            }
            (slots, from, mtp, head, draft_off, host, None)
        }
        Paged::Reduced { slots: served } => {
            let (from, drafts, draft_off) = reduced(host.2, (slots, served), mtp, draft_off, false);
            let one = Plans {
                mtp: if drafts { plans.mtp } else { None },
                slots: served,
                ..plans
            };
            rule = ctx38(&one, a.ctx)?;
            let host = one.host(rule.ctx(), set)?;
            (
                served,
                from,
                drafts,
                if drafts { head } else { None },
                draft_off,
                host,
                Some(one),
            )
        }
    };
    refuse_mtp_levers(&draft_off)?;
    let residency_word = set.filter(|&(r, _)| r != Residency::Off).map(|(_, w)| w);
    refuse_feed(feed, (slots, mtp), residency_word)?;
    // The prompt cache's budget, read from the host need at the context the
    // bound asks about, on one reading of the host's available bytes: the
    // bound searches contexts, and a reading that moved between two of its
    // probes would make its answer and its check disagree.
    let paged_one = one.is_some();
    let live = one.unwrap_or(plans);
    let reading = host_available_read()?;
    let entry = seq38_state(&inputs, 1, mtp);
    let cache_of = |(need, pool, arena, checkpoints, paged): (u64, u64, u64, u64, u64)| {
        CacheRam::at(
            a.cache_ram,
            reading.clone(),
            need,
            pool,
            (arena, paged),
            checkpoints,
        )
        .map(|c| c.holding(entry))
    };
    let first = cache_of(host)?;
    // A set `--cache-ram` is the same at every context.
    let ram_at = |n: usize| -> Result<u64, GateError> {
        match a.cache_ram {
            Some(_) => Ok(first.ram),
            None => Ok(cache_of(live.host(n, set)?)?.ram),
        }
    };
    let before = rule.ctx();
    let (rule, holds) = rule.host_bound(&live, &ram_at, mtp)?;
    // The cache line names the need of the context the load serves at, and
    // that its budget holds no session there when it does not.
    let cache = if rule.ctx() == before {
        first
    } else {
        cache_of(live.host(rule.ctx(), set)?)?
    };
    let cache = if holds {
        cache
    } else {
        cache.short_of(rule.ctx(), seq38_state(&inputs, rule.ctx(), mtp))
    };
    if paged_one && !mtp {
        mtp_inputs = None;
    }
    rule.print();
    eprint_noted(&format!(
        "{} message_start={MESSAGE_START} in_vocab={has_start} in_template={in_template}",
        cache.line()
    ));
    // The rounds of several slots, decided once (the open takes it through
    // `SeatArgs`): one pass of the busy rows, a slot at a time beside an
    // expert tier card ([`Place38::tiered`]).
    let pass = Pass::of(slots, !a.place.tiered());
    // The slots the seat serves ([`Session::add_slots`]): the flag's, one
    // beside a set `--ctx-size`, or 2 — one drafted pass a slot a round,
    // nothing parked.
    parallel_line(slots, rule.ctx(), from, pass);
    if mtp {
        let be = break_even_of(a.place);
        eprintln!(
            "draft keep place={} break_even={} reply={NOMINAL_REPLY} per_reply_token={} (the \
             kept prefix below which a request that would leave the MTP draft off resets; \
             https://github.com/midagedev/rig-log/blob/main/docs/bloomery/plan.md, Qwen3.8 serve)",
            a.place.name(),
            be.at(NOMINAL_REPLY),
            be.per_token
        );
    }
    let ctx = rule.ctx();
    // A drafted load's window checked at the context the rule chose — the
    // unset draft's own condition at the default's, the set draft's at any;
    // a draft the yield turned off needs no window (the plain rule's context
    // is the larger one it fell to).
    if mtp && slots.saturating_mul(ctx) < window_positions(slots) {
        return Err(format!(
            "a slot's context of {ctx} leaves no positions for the MTP draft's window; \
             --ctx-size names a total of at least {}",
            window_positions(slots)
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
    check_reserve(
        &machine,
        slots,
        checkpoint_budget38(gdn_layers(&inputs), u64::try_from(ctx)?)?,
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
    // The one-column rule holds at the plan the load runs: the cache's
    // bound only lowers the context the rule read, which leaves the card
    // more experts and the host fewer, so a paged load of several columns
    // — slots stepped together, or a draft — is the seat's own breach.
    if plan.host.nvme_arena_bytes > 0 && (pass.together(slots) || mtp) {
        return Err(format!(
            "the load's plan pages host experts through the NVMe tier's {} B RAM arena at \
             {slots} slots, the draft {}: the one-column rule read another plan",
            plan.host.nvme_arena_bytes,
            if mtp { "on" } else { "off" }
        )
        .into());
    }
    let line = Record::new(&record::PLAN38)
        .w("place", a.place.name())
        .w("card", machine.cards[0].name.as_str())
        .w("experts", experts_name(experts))
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
    let line = record::with_room(line, plan.host.room);
    line.csv("devices", record::plan_devices(&machine))
        .w("cuda_order", record::cuda_order())
        .eprint();
    let (residency, pool) = residency38_at(&plan, (a.place, feed), set)?;
    if a.plan_only {
        // The records before the load are out; nothing was loaded and the
        // route trace's directory is not made.
        std::process::exit(0);
    }
    // Made last among the refusals, so a load that stops before the engine
    // leaves no directory behind.
    let trace = levers
        .route_trace()
        .map(|dir| {
            let build = format!("bloomery-serve-qwen38 {}", serve::VERSION);
            let run = TraceRun {
                model: &path,
                build: &build,
                place: a.place.name(),
                experts,
                prefill: feed.prompt,
                chunk: None,
            };
            route_trace38::create(dir, &inputs, &run)
        })
        .transpose()?;
    let props = EngineProps {
        model: Some(model),
        placement: placement_of("bloomery-serve-qwen38", &machine, &plan),
        ctx_verified: Some(VERIFIED_POSITIONS),
        ..EngineProps::default()
    };

    let open = SeatArgs {
        place: a.place,
        lane,
        ctx,
        slots,
        pass,
        experts,
        plan_levers,
        room: inputs.room.clone(),
        plan: PlanKey::of(&plan)?,
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
        pool,
        feed: feed.prompt,
        trace,
        xstream: levers.xstream(),
        stats: levers.step_stats(),
        width,
        cache_ram: cache.ram,
    };
    let engine = SeatEngine::spawn(
        move || Q38::open(open),
        ctx,
        vocab,
        machine.cards[0].name.clone(),
        props,
        cache.ram,
    )?;

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
            slot_save_path: a.slot_save_path,
            api_keys: a.api_keys,
            parallel: slots,
            queue_depth: a.queue_depth,
        },
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

/// The terms of a plan that the seat's records and rules read: the stage
/// card's experts a layer and in all, the host leg's experts, the routed
/// bytes it holds resident and on the NVMe tier, that tier's RAM arena (the
/// one-column rule's term), the tables the host holds and the experts the
/// tier cards hold. The main thread records them from the plan it ruled by;
/// the engine thread holds the plan it loads by to them ([`Q38::open`]).
#[derive(Clone, Debug)]
struct PlanKey {
    n_l: Vec<u64>,
    card_experts: u64,
    tier_experts: u64,
    host_experts: u64,
    host_expert_bytes: u64,
    host_table_bytes: u64,
    nvme_expert_bytes: u64,
    nvme_arena_bytes: u64,
}

impl PlanKey {
    /// `plan`'s terms.
    fn of(plan: &Plan<'_>) -> Result<PlanKey, GateError> {
        Ok(PlanKey {
            n_l: plan.n_l.clone(),
            card_experts: plan.cards.first().ok_or("a plan with no card")?.experts,
            tier_experts: plan.tier_n_l.iter().flatten().sum(),
            host_experts: plan.host.experts,
            host_expert_bytes: plan.host.expert_bytes,
            host_table_bytes: plan.host.table_bytes,
            nvme_expert_bytes: plan.host.nvme_expert_bytes,
            nvme_arena_bytes: plan.host.nvme_arena_bytes,
        })
    }

    /// Every term by name, in the order [`PlanKey::hold`] reads them.
    fn terms(&self) -> [(&'static str, String); 8] {
        [
            ("n_l", format!("{:?}", self.n_l)),
            ("cards[0].experts", self.card_experts.to_string()),
            ("tiers.experts", self.tier_experts.to_string()),
            ("host.experts", self.host_experts.to_string()),
            ("host.expert_bytes", self.host_expert_bytes.to_string()),
            ("host.table_bytes", self.host_table_bytes.to_string()),
            ("host.nvme_expert_bytes", self.nvme_expert_bytes.to_string()),
            ("host.nvme_arena_bytes", self.nvme_arena_bytes.to_string()),
        ]
    }

    /// `Ok` when `engine`, the plan the load runs by, has every term of
    /// this one; else the first term that differs, named with both values.
    fn hold(&self, engine: &PlanKey) -> Result<(), GateError> {
        for ((name, seat), (_, load)) in self.terms().into_iter().zip(engine.terms()) {
            if seat != load {
                return Err(format!(
                    "the engine's plan differs from the plan the seat recorded: {name} {seat} \
                     vs {load}"
                )
                .into());
            }
        }
        Ok(())
    }
}

/// What the engine thread opens the seat with.
struct SeatArgs {
    place: Place38,
    /// The lane's rate probed before the plans, and the card it ran on:
    /// the `lane` line beside the `load` line.
    lane: Lane,
    /// A slot's context: the stores' rows one sequence serves.
    ctx: usize,
    /// The resident sequences the plan counted and the load makes
    /// ([`Session::add_slots`] after the captures).
    slots: usize,
    /// The shape of a round of several slots, which the `parallel` line
    /// named and the open runs.
    pass: Pass,
    experts: Experts,
    plan_levers: PlanLevers,
    /// The host's room and its reading as the main thread's plans took them
    /// (`PlanInputs::room`): the engine thread plans from it, not from a
    /// reading of its own.
    room: (u64, HostRead),
    /// The plan the main thread's rules and records read, which the load's
    /// plan is held to.
    plan: PlanKey,
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
    /// Whether the load hosts the residency's churn pool, as the word's
    /// resolution decided ([`residency38`]).
    pool: ChurnHosting,
    /// How the session feeds a prompt call: [`Prompt38::Step`] (one captured
    /// step an id) or [`Prompt38::Auto`].
    feed: Prompt38,
    /// The route trace the host tier records into, its directory made by the
    /// main thread.
    trace: Option<RouteTrace>,
    /// `BLOOMERY_XSTREAM` as set, or unset for the shared rule
    /// ([`xstream_rule`]).
    xstream: Option<&'static str>,
    /// `BLOOMERY_STEP_STATS`, the round records the seat prints.
    stats: bool,
    /// `BLOOMERY_MTP_WIDTH`: the width a drafted window verifies, the
    /// chooser's (`cost`) or the draft's own (`fixed`).
    width: WidthMode,
    /// The prompt cache's byte budget: the pinned staging its saves are
    /// copied into on a lane of their own, pinned whole at the open
    /// (`SaveLane::new`); 0 (the cache off) makes no lane.
    cache_ram: u64,
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
    /// How a prompt call is fed ([`SeatArgs::feed`]).
    feed: Prompt38,
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
    /// The load's tier census as the last call left it, on a load whose plan
    /// pages through the NVMe tier: each call prints its span
    /// ([`Q38::print_census`]). `None` on a load with no tier.
    census: Option<TierCensus>,
    /// The branch the last keep query took ([`Q38::draft_keep`]), one a
    /// slot, printed at that slot's next call; a `RefCell` because the keep
    /// query runs on `&self` (the engine thread owns the seat, so the borrow
    /// never contends).
    branch: std::cell::RefCell<Vec<Option<Branch38>>>,
    /// The draft-off line a prefix put back owes ([`Seat::resume_prefix`]), one a
    /// slot: the position and the sentence, printed at the slot's next call
    /// ([`Q38::before_call_at`]) if the draft is still off then, so a request the
    /// server resets after the load (its keep below the break-even) prints none.
    cut_line: Vec<Option<(u32, String)>>,
    /// The copy stream and the pinned staging the prompt cache's saves are
    /// made on ([`Seat::snapshots_async`]); none with the cache off.
    lane: Option<SaveLane>,
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
        // The room is the main thread's: a reading of the host's available
        // bytes of its own would plan the load from another room than the
        // plan the seat ruled by, and the paged tier's arena is that room's
        // difference.
        let mut inputs = PlanInputs::describe(&file)?;
        inputs.room = a.room.clone();
        let cap = serve_ctx(u64::try_from(a.ctx)?, &inputs.hp)?;
        let ub = ubatch_for(a.ctx)?;
        let ctx_ub = (u64::try_from(a.ctx)?, u64::try_from(ub)?);
        // The sequences the load makes pin the body's budget at this context;
        // the plan's machine must reserve at least that.
        let ckpt = checkpoint_budget38(gdn_layers(&inputs), ctx_ub.0)?;
        // The wall before the loader's call, which the `load phases` record names.
        let opened: Duration;
        let mut m = match a.mtp {
            false => {
                let machine = a.place.machine(&inputs, ctx_ub, a.experts, None, a.slots)?;
                check_reserve(&machine, a.slots, ckpt)?;
                let plan = inputs.plan_with_slots(
                    &machine,
                    u64::try_from(a.ctx)?,
                    &a.plan_levers,
                    a.experts,
                    a.slots,
                )?;
                a.plan.hold(&PlanKey::of(&plan)?)?;
                opened = t.elapsed();
                Body38::open_placed_residency(
                    file,
                    &plan,
                    &inputs,
                    CARD38,
                    a.host,
                    ub,
                    a.residency,
                    a.pool,
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
                check_reserve(&machine, a.slots, ckpt)?;
                let plan = inputs.plan_mtp_with_slots(
                    &machine,
                    u64::try_from(a.ctx)?,
                    &a.plan_levers,
                    &mtp,
                    a.experts,
                    a.slots,
                )?;
                a.plan.hold(&PlanKey::of(&plan.plan)?)?;
                opened = t.elapsed();
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
                    a.pool,
                    a.slots,
                )?
            }
        };
        m.set_mode(StepMode::Graph);
        record::lane(a.lane.0, &a.lane.1).eprint();
        let total = t.elapsed();
        eprint_noted(&format!(
            "load arch=qwen4exp resident_bytes={} ctx={} slots={} slot_ctx={} ctx_max={cap} \
             ctx_train={} verified={VERIFIED_POSITIONS} layers={} mode=graph store_bytes={} \
             prefill={} ubatch={} place={} card_layers={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            a.ctx,
            a.slots,
            a.ctx,
            inputs.hp.n_ctx_train,
            m.layers().len(),
            m.body(WHAT)?.store_bytes(),
            a.feed.name(),
            m.body(WHAT)?.ubatch_rows(),
            a.place.name(),
            m.body(WHAT)?.card_layers(),
            total.as_secs_f64()
        ));
        record::eprint_load_phases(&m, total, opened);
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
        if let Some(t) = a.trace {
            s.model_mut()
                .body_parts(WHAT)?
                .2
                .hybrid_mut()
                .attach_route_trace(t)?;
        }
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
        // The rounds of several slots, decided once (`SeatArgs::pass`): one
        // pass of the busy rows where the body runs one, the drafted rounds
        // on a drafted load and the plain rounds on a plain one.
        let pass_of_slots = a.pass.together(a.slots);
        if a.pass == Pass::Turns {
            eprintln!(
                "bloomery-serve-qwen38: --place {}: the rounds of several slots run a slot at a \
                 time: the body runs no pass of several slots beside an expert tier card",
                a.place.name()
            );
        }
        // The expert stream last: a started ring takes the card's free bytes
        // past its keep, so every card buffer the open makes comes before it.
        let (gpu, _, body) = s.model_mut().body_parts(WHAT)?;
        let line = xstream_rule(gpu, body, a.xstream, a.place.stage38(), a.residency)?;
        eprintln!("{line}");
        let census = {
            let b = s.model().body(WHAT)?;
            b.hybrid().census(b.nvme_tier().map(|t| &**t))
        };
        let lane = match a.cache_ram {
            0 => None,
            ram => Some(s.model_mut().seq_lane(ram)?),
        };
        Ok(Q38 {
            s,
            drafts,
            ctx: a.ctx,
            feed: a.feed,
            draft_bytes: a.draft_bytes,
            draft_path: a.draft_path,
            residency,
            break_even: a.mtp.then(|| break_even_of(a.place)),
            one_pass: pass_of_slots && a.mtp,
            step_pass: pass_of_slots && !a.mtp,
            stats: a.stats,
            census,
            branch: std::cell::RefCell::new(vec![None; a.slots]),
            cut_line: vec![None; a.slots],
            lane,
        })
    }

    /// The prompt call of `ids` one captured step an id
    /// ([`Prompt38::Step`]) with one readback after the last, the host tier's
    /// route trace told the call's positions first (`Hybrid::route_prompt`).
    /// The session has no call of this path, so the call is the model's own,
    /// under the engine watchdog's prompt mark as the session's is; the open
    /// refuses the draft beside this feed ([`refuse_feed`]), so none is
    /// told.
    fn prefill_steps(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        const WHAT: &str = "bloomery-serve-qwen38";
        let _busy = self.s.model().prompt_busy();
        let pos = self.s.pos();
        let m = self.s.model_mut();
        m.body_parts(WHAT)?
            .2
            .hybrid_mut()
            .route_prompt(pos, ids.len())?;
        Ok(m.prompt38(ids, Prompt38::Step)?)
    }

    /// The drafted seat's keep rule, its one owner: a prefix of `at`
    /// positions the body's rule keeps, where keeping it leaves the draft
    /// off for the reply (a cut below the held positions to a point the
    /// draft cannot rejoin at, [`Body38::draft_rejoins`], or a draft already
    /// off) is kept only at or past the break-even for the reply's `reply`
    /// tokens through passes (the request's, at most [`NOMINAL_REPLY`], and
    /// that when it bounds nothing); below it the request resets, which turns
    /// the draft back on, and prefills its whole prompt. `None` where the rule
    /// weighs nothing: no draft, nothing kept, or, with the draft on, every
    /// held position kept or a cut it rejoins.
    fn draft_keep(&self, at: usize, reply: Option<usize>) -> Option<Branch38> {
        let be = self.break_even?;
        let held = self.s.pos() as usize;
        let on = !self.drafts.is_off(self.s.selected());
        if at == 0 || on && (at >= held || self.draft_rejoins(at)) {
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

    /// Whether the body kept the MTP draft's side at `at`
    /// ([`Body38::draft_rejoins`]), where a cut to `at` rejoins the draft.
    fn draft_rejoins(&self, at: usize) -> bool {
        u32::try_from(at).is_ok_and(|at| {
            self.s
                .model()
                .body("bloomery-serve-qwen38")
                .is_ok_and(|b| b.draft_rejoins(at))
        })
    }

    /// Whether a cut or a put-back to `at` on slot `sel` rejoins its draft:
    /// the body kept the draft's side there ([`Q38::draft_rejoins`]). Refused
    /// by name when the draft is off yet the body kept its side: an off draft
    /// holds an empty store, because the seat turns it off only after the body
    /// forgot the store and an off draft walks nothing, so a rejoin possible
    /// there means one of the two stopped holding.
    fn rejoins(&self, sel: usize, at: usize) -> Result<bool, GateError> {
        let kept = self.draft_rejoins(at);
        if kept && self.drafts.is_off(sel) {
            return Err(format!(
                "slot {sel}: the MTP draft is off, yet the body kept its side at position {at}; \
                 an off draft holds an empty store"
            )
            .into());
        }
        Ok(kept)
    }

    /// What the request's reuse left to say, at its first call on the slot
    /// it runs: the draft-off line of a prefix put back that the request kept,
    /// then the `mtp keep` record of the branch its keep took.
    fn before_call(&mut self) {
        self.before_call_at(self.s.selected());
    }

    /// [`Q38::before_call`] at `slot`, a round's row printing at its own
    /// slot as the fallback loop's per-row pass would.
    fn before_call_at(&mut self, slot: usize) {
        if let Some((pos, what)) = self.cut_line[slot].take()
            && self.drafts.is_off(slot)
        {
            draft_off(pos, &what);
        }
        if let Some(b) = self.branch.borrow_mut()[slot].take() {
            Record::new(&record::MTP_KEEP38)
                .w("branch", if b.kept { "kept" } else { "reset" })
                .u("prefix", b.prefix)
                .u("break_even", b.break_even)
                .u("reply", b.reply)
                .eprint();
        }
    }

    /// The call's `tier census` record ([`Q38::print_census`]); then the
    /// `call stream` records of the last prompt call — each of its picks,
    /// then its end — and the `residency pass` records of the boundaries
    /// since, on stderr, nothing of them without the residency.
    fn print_passes(&mut self) -> Result<(), GateError> {
        self.print_census()?;
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

    /// The `tier census` record of the span since the last call's, phase
    /// `call`, on stderr: its picks by where they were served and the bytes
    /// it moved between the tiers, the dropper's calls and wall and the
    /// pool's dispatch waits; then the load's cumulative `nvtier` record, so
    /// the last one of a run holds the arena's counters at its end. Nothing
    /// on a load with no tier.
    fn print_census(&mut self) -> Result<(), GateError> {
        let Some(last) = self.census else {
            return Ok(());
        };
        let b = self.s.model().body("bloomery-serve-qwen38")?;
        let Some(now) = b.hybrid().census(b.nvme_tier().map(|t| &**t)) else {
            return Ok(());
        };
        now.since(&last)
            .fields()
            .iter()
            .fold(
                Record::new(&record::TIER_CENSUS).w("phase", "call"),
                |r, &(k, v)| r.u(k, v),
            )
            .eprint();
        self.census = Some(now);
        if let Some(r) = record::nvtier_of(b.nvme_tier().map(|t| &**t)) {
            r.eprint();
        }
        Ok(())
    }
}

/// The cuts of a prompt call of the positions `first..end` fed by `feed`: the
/// message starts (`marks`) inside the call where both runs keep at least
/// `Prompt38::GEMM_FROM` ids, so each run is a ubatch walk and the bits are
/// the uncut call's; under the MTP draft the draft's prompt call joins each
/// run where the one before left it. None under the step feed, which leaves
/// no hole and needs none: a cut would be a prompt call of its own, which
/// the route trace marks as a call row.
pub(crate) fn prompt_cuts(feed: Prompt38, first: usize, end: usize, marks: &[usize]) -> Vec<usize> {
    if feed == Prompt38::Step {
        return Vec::new();
    }
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

/// The line that says the MTP draft proposes nothing from `pos` on, for
/// `what` (a cut or a state put back).
fn draft_off(pos: u32, what: &str) {
    eprintln!(
        "bloomery-serve-qwen38: the MTP draft proposes nothing from position {pos} until a \
         request starts from position 0: {what}; {DRAFT_OFF_WHY}"
    );
}

/// A Qwen3.8 sequence state as the server's prompt cache holds it ([`Seq38`],
/// the draft's rows within it, the body's rule ranking it), with the draft's
/// side of it ([`SlotDrafts::park`]) beside the body's: the one common
/// [`Cached`], whose draft side is this seat's.
type Saved38 = Cached<Seq38, Option<ParkedDraft<TargetRows>>>;

/// What a refusal of another seat's state names ([`put_back`]).
const WHOSE: &str = "a qwen4exp body's";

/// The draft's three steps around a Qwen3.8 state's put-back ([`put_back`]).
impl PutBack for Q38 {
    type Body = Seq38;
    type Side = Option<ParkedDraft<TargetRows>>;

    fn takes(&self, side: &Self::Side) -> Result<(), GateError> {
        self.drafts.takes(side.as_ref())
    }

    fn reset_selected(&mut self) -> Result<(), GateError> {
        let sel = self.s.selected();
        self.drafts.reset(&mut self.s, sel)
    }

    fn unpark(&mut self, side: &Self::Side) -> Result<(), GateError> {
        self.drafts.unpark(self.s.selected(), side.as_ref())
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
    /// the prompt's units. Under `BLOOMERY_PREFILL=steps` one step per id
    /// ([`Q38::prefill_steps`]).
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        self.before_call();
        let sel = self.s.selected();
        let next = if self.feed == Prompt38::Step {
            self.prefill_steps(ids)?
        } else {
            self.drafts.prefill(&mut self.s, sel, ids)?
        };
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
        self.s.keep_query(n)
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
    /// message. A cut under the MTP draft rejoins the slot's draft where the
    /// body kept its side ([`SlotDrafts::rejoin`]; an off draft has none,
    /// [`Q38::rejoins`]); otherwise it prints that the draft proposes nothing
    /// from there ([`DRAFT_OFF_WHY`]).
    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        let held = self.s.pos();
        self.s.model_mut().rollback(pos)?;
        if self.drafts.drafts() && pos < held {
            let sel = self.s.selected();
            self.cut_line[sel] = None;
            if self.rejoins(sel, pos as usize)? {
                self.drafts.rejoin(sel, pos)?;
            } else {
                self.drafts.turn_off(sel, DRAFT_OFF_WHY)?;
                draft_off(pos, &format!("a cut back from {held}"));
            }
        }
        Ok(())
    }

    /// [`prompt_cuts`] under the seat's feed.
    fn splits(&self, first: usize, end: usize, marks: &[usize]) -> Vec<usize> {
        prompt_cuts(self.feed, first, end, marks)
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
        let state = self.s.model_mut().seq_save()?;
        let draft = self.drafts.park(sel)?;
        Ok(Arc::new(Saved38::landed(state, draft)))
    }

    /// Whether the saves copy on the lane: the seat holds one.
    fn snapshots_async(&self) -> bool {
        self.lane.is_some()
    }

    /// The state's copy begun on the lane (`GpuModel::seq_save_async`), the
    /// draft's side taken beside it; a pool with no room is the server's
    /// `NoRoom` ([`begin`]). Without a lane, the snapshot on the call.
    fn snapshot_begin(&mut self) -> Result<Pending, GateError> {
        let Some(lane) = &self.lane else {
            return Ok(Pending::landed(self.snapshot()?));
        };
        let copy = self
            .s
            .model_mut()
            .seq_save_async(lane)?
            .map(|(state, pending)| (state, Arc::new(pending) as Arc<dyn Landing>));
        let sel = self.s.selected();
        begin(copy, || self.drafts.park(sel))
    }

    /// The state put back (`GpuModel::seq_resume`) after the session's reset,
    /// then the draft's side of it ([`put_back`]): the sequence's next call
    /// runs as it would have with no switch between, its draft joining where
    /// it left. Refused by name, before the reset, for a state this seat did
    /// not take or one saved with the draft on put back with it off (or the
    /// other way); the body refuses another model's state or layout.
    fn resume(&mut self, state: &dyn Saved) -> Result<(), GateError> {
        put_back(self, state, WHOSE, |q, body| {
            q.s.model_mut().seq_resume(body).map(|()| body.positions())
        })
        .map(drop)
    }

    /// The first `n` positions the state can be cut to put back after the
    /// session's reset (`GpuModel::seq_resume_prefix`), then the draft's side
    /// of it ([`put_back`]): the sequence's next call runs as it would have
    /// after the whole state's resume and a rollback to the returned position
    /// ([`Seat::rollback`]): a cut short of the state's positions under the
    /// draft rejoins the slot's draft where the body kept its side
    /// ([`SlotDrafts::rejoin`]; a state saved with the draft off has none,
    /// [`Q38::rejoins`]), else leaves it proposing nothing from there
    /// ([`DRAFT_OFF_WHY`]); the line
    /// says so at the slot's next call, unless the server resets the slot
    /// first (its keep is below the break-even, [`Q38::keep_for`]).
    fn resume_prefix(&mut self, state: &dyn Saved, n: usize) -> Result<usize, GateError> {
        let n = u32::try_from(n).unwrap_or(u32::MAX);
        let at = put_back(self, state, WHOSE, |q, body| {
            q.s.model_mut().seq_resume_prefix(body, n)
        })?;
        if self.drafts.drafts() && at < state.n_tokens() {
            let sel = self.s.selected();
            let pos = u32::try_from(at).unwrap_or(u32::MAX);
            if self.rejoins(sel, at)? {
                self.drafts.rejoin(sel, pos)?;
            } else {
                self.drafts.turn_off(sel, DRAFT_OFF_WHY)?;
                let from = state.n_tokens();
                self.cut_line[sel] = Some((
                    pos,
                    format!("a state of {from} positions put back as its first {at}"),
                ));
            }
        }
        Ok(at)
    }

    /// A prefix kept less of than shared is a `cache reuse` record; every
    /// other note of the prompt cache prints as its line.
    fn note(note: &CacheNote) {
        note_line("bloomery-serve-qwen38", note);
    }
}
