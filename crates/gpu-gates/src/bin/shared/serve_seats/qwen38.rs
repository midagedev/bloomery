//! `bloomery-serve-qwen38` — the llama-server-compatible HTTP API on the
//! Qwen3.8-Flash-Next (qwen4exp) engine, for a client to attach to.
//!
//! This module is the seat: `bloomery-serve-qwen38` and `bloomery-serve --model
//! qwen38` are each one call of [`run`], which takes the process's arguments
//! (`--model` already taken out by the one-binary server).
//!
//!     bloomery-serve-qwen38 [--host 127.0.0.1] [--port 8080]
//!                           [--place a|gate|bp|<stage>[+<tier>…]] [--ctx-size C]
//!                           [--alias NAME] [--cache-ram MIB]
//!                           [--chat-template-file PATH] [--parallel N]
//!                           [--queue-depth Q] [--park-ram MIB] [--slot-save-path DIR]
//!                           [--mmproj FILE | --no-mmproj] [--image-min-tokens N]
//!                           [--image-max-tokens N] [-ub N | --ubatch-size N]
//!                           [--cache-type-k f16|q8_0]
//!
//! `--cache-type-k` is the KV cache's format, llama-server's flag and words: a set word wins, then
//! the `BLOOMERY_QWEN3_KV` lever's; unset, a checkpoint folder (FP4-Local) runs q8_0 and a GGUF
//! file f16 ([`KvQ8::choose`]). `--cache-type-v` does not exist: both planes quantize together.
//!
//! `-ub N` (`--ubatch-size`, llama-server's spelling and its `LLAMA_ARG_UBATCH`) is the most
//! prompt positions one ubatch walk runs, over `BLOOMERY_QWEN3_UBATCH`, else 4096; a size outside
//! `1..=4096` is refused by name before the model opens. The one size the seat reads plans every
//! context it asks and sizes the load's ubatch arena, each at most the slot's context.
//!
//! `--slot-save-path` names the directory the slot actions answer from, as
//! llama-server's; without it every slot action is refused. `--ctx` is `--ctx-size` under its family's other spelling. The model is
//! `$BLOOMERY_REF_MODEL` (a `--model` flag is not taken: the family's
//! binaries read the one the profile pins); its first shard gives the
//! vocabulary, `tokenizer.chat_template` the chat template
//! (`--chat-template-file` replaces it) and `general.name` the default alias.
//! A model path that is a directory is a Qwen3.8 checkpoint folder, served as
//! stored (`qwen38_folder::open`): its header (`gguf::Split::header_only`)
//! takes the first shard's place in every rule and plan, its tokenizer is the
//! vocabulary, and the engine loads every card tensor from the folder's parts
//! ([`Folder38`]) and reads its PLE rows where the folder stores them. Such a
//! load holds every tensor on the stage card and runs no residency machine.
//! Unless `BLOOMERY_MTP_DRAFT` names a file, its MTP draft is the folder's own
//! MTP layer, under the draft rule below; a folder without the layer turns the
//! unset draft off naming the folder and refuses `BLOOMERY_DRAFT=mtp` by name
//! ([`draft_off_for`]).
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
//! `--place` is `a`, `gate`, `bp` or a card list `<stage>[+<tier>…]`
//! (`generate::Place`, read by `bind::place_word` as every seat reads it); a
//! list of more tier cards than the Qwen3.8 body serves (`q38place::Q38_RULE`)
//! and `tp` are refused by name before any plan. The word runs as given (a
//! single named card plans on its own bytes); unset, the common rule every
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
//! and generation stops there with `truncated`. The context is the seats'
//! common choice ([`ctx38`], `generate::choose_ctx_of`): it comes first, the
//! experts fill what is left. The limit is the file's `context_length`
//! within `place::KERNEL_POSITIONS` (YaRN scaling past it is not built). Set,
//! a total a slot's share of which passes the limit is refused by name, and
//! one whose N-slot plan the machine cannot hold is refused with the largest
//! share it holds, before the load. Unset, a slot asks the limit and, where
//! that plan is short of the card or of the unified pool the card shares with
//! the host, takes the largest context under it, a multiple of 1024 down to
//! the floor, whose plan holds. Where a residency runs the machine with
//! experts on the card (a set `mid` word, or the lever unset on the stage
//! plan (a) or (b′)) a plan holds only if it also leaves every layer of the
//! card the experts that word runs on ([`default_floor`]), so the context
//! never takes the room the residency needs; where not even the floor of the search leaves them, the
//! context is the plain fit, the unset word resolves `off` (a set word is
//! refused by name), and a
//! `ctx experts-floor=unmet` line says why. An unset MTP draft yields to the
//! context:
//! when the drafted plan fits a context at most half the one the plain plan
//! holds, or holds not even the floor, the draft goes off and a `draft
//! yield` record names both. The
//! prompt cache's budget never lowers the context: the `cache` line names
//! when it holds no session there. The
//! `load` line prints the stores' `ctx` (a slot's rows) with the slot count
//! and each slot's context, the cap as `ctx_max`, `ctx_train` and
//! `verified`, the deepest context the reference sets hold our numbers to
//! ik's at (`refset::arch::qwen4exp::VERIFIED_POSITIONS`, `/props`'
//! `engine.ctx_verified`), which bounds nothing. One `ctx` line on stderr
//! names the rule (`set`, `target` or `fitted`, with the target it fitted
//! under and the memory the context one step above was short of), the slot
//! context, the slots, their total and the limit with its source. The
//! `setup` line before the address repeats the slot context and its rule.
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
//! `SlotRows::MAX_ROWS`, a sampling or id-banning request's window among
//! them), and its round of plain steps (of requests that take no pass: one
//! that reads its logits row, a forced id) as one pass of their rows, each row told to its
//! slot's draft as its step alone would be (`rounds::step_rows_drafted`:
//! every slot's waiting draft rows walked in one store walk before the
//! pass), slot 0 among the busy slots or not (a pass that leaves it out
//! leaves its waiting rows where its draft reads them,
//! `Body38::enqueue_slots`). On a load that drafts nothing every request steps, and the
//! round of the busy slots' steps runs as one pass of their rows (`GpuModel::step_slots`,
//! each row bit for bit its step alone). A lookup load runs its rounds a
//! select and a call a row. The
//! sequences are switched by pointer exchange, and one draft is held a
//! resident slot — its host state never moved, its device side parking
//! with the sequence on every switch — so each request's tokens are its
//! solo run's. Under `--place bp` (an expert tier card, beside which the
//! body runs no pass of several slots) every round runs a slot at a time,
//! and the open prints a line that says so. `--parallel 1` is exactly the
//! one-sequence server, its rounds one pass a slot. `--parallel` naming no
//! count, the seat serves one slot beside a set `--ctx-size` (the flag is
//! one request's context, `bind::slots_of`; one line on stderr says so),
//! one under the route trace (below), and its default one over the automatic
//! context. Under more slots than one
//! the context is split
//! as llama-server splits it with
//! `-np N` and no `-kvu`: the total (the `--ctx-size` the flags named, or
//! the automatic choice when unset) is the slots' sum, each slot `total /
//! N` positions rounded
//! down, and the default's fit runs over a slot's context, the plan
//! counting every sequence ([`PlanInputs::plan_with_slots`]). An explicit
//! `--ctx` a slot of whose N-slot plan the machine cannot hold is refused by
//! name before the load, naming the split. Nothing parks: no slot ever waits for another
//! (a slot's round runs whether the others stream), so `--park-ram` is
//! refused by name — resident slots hold their state on the card, in the
//! plan. A `parallel` line on stderr names the rule (`slots`), the slots,
//! the split they serve, what set the count (`from`: the `--parallel`
//! flag, a set `--ctx-size`, the default, or `route-trace`) and the shape of a
//! round of several slots (`pass=one`, or
//! `pass=turns` under `--place bp`, which runs the slots a slot at a time).
//! `--queue-depth Q` bounds the requests that wait for a slot.
//!
//! A plan that pages host experts through the NVMe tier's RAM arena (a room
//! under half the host leg's experts, or `BLOOMERY_NVTIER_BYTES` set:
//! `placement::nvme_arena_of`) serves its slots as one pass over the arena
//! (`bloomery_levers::paged_columns`): the placement's floor counts a column
//! a slot (`ArenaFloor::of`), so the paged experts of every slot's rows are
//! read from the arena, not through the file mapping. The count is served
//! as asked — the default is already one slot, so the rule has no count to
//! reduce — and an unset draft goes off (`load draft=off
//! (unset: the plan pages …)` — also for a draft whose yield to the context,
//! judged on the plan the load abandons, had turned it off;
//! whether a drafted verify pays on a paged plan is the machine's drive, not
//! the rule's, to answer, so a set `BLOOMERY_DRAFT=mtp` runs), one `paged`
//! line on stderr naming the arena and what was asked (`asked_draft` the
//! draft before that yield, `draft` what the plan serves). The rule reads
//! the plan the flags name — a set context's plain plan before any choice,
//! else the plan the default's choice and the draft's yield chose — and the
//! load's own plan is held to it: the engine thread plans from the room the
//! main thread read (`PlanInputs::room`, never a reading of its own), and
//! its plan must have the terms the records and rules read (`PlanKey`: the
//! card's and the tiers' experts, the host's experts, resident bytes and
//! tables, the NVMe tier's bytes and arena) or the load is refused by name.
//! The plan's host reserves every slot's checkpoints (the body's budget at
//! the slot's context, [`reserve_checkpoints`]), so the host need, the
//! split's floor and the arena count them. The
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
//!
//! `--mmproj FILE` (llama-server's spelling, with `--no-mmproj` and the two token limits:
//! [`vision_seat::Flags`]) gives the file its image input. The encoder file (Qwen3.8's is a
//! `qwen3vl_merger` projector) is held against the text model — its rows' width, the
//! vocabulary's three image ids, the file's IMROPE sections and its PLE image token — and refused
//! by name when they differ ([`vision_seat::open_seat`]), before any plan. Without `--mmproj` the
//! `--hf` repo's projector is the file where the draft rule's first word is off and the prompt is
//! not fed a step an id ([`vision_of`]); it stays only where it costs the plan no expert and no
//! context ([`vision_seat::kept`]), else a `vision_seat` record says it was dropped, and
//! `--mmproj`'s file is never dropped. The tower sits on the smallest card outside the plan, else
//! as a named reserve of the stage card ([`vision_seat::EncoderSeat`]), which every plan counts
//! and the `vision_seat` record names, printed with the plan (`--plan` too). The placement is the
//! one the same run without images takes: the tower is seated on it ([`reseated`]). The load is
//! a media load (`GpuModel::enable_media`, once after the open): every sequence holds the
//! rotation table an image's positions rewrite, which every plan counts a position as a card
//! reserve ([`vision_seat::TABLES_RESERVE`]). The `vision` record follows the `load` line. The
//! server renders an image part as the model's placeholder and expands it to its grid's span
//! (`serve::media` over [`Seat::media_model`]); a prompt call that holds images encodes each on
//! the tower and feeds the prompt with the rows in place of their positions' embeddings
//! ([`Seat::prefill_media`]: the ubatch walk, eager). An image with no `--mmproj` is the server's
//! own 400.
//!
//! An image load drafts nothing, the engine's media load refusing a load that carries the MTP
//! draft ([`draft_off_for`]): `BLOOMERY_DRAFT=mtp` beside `--mmproj` is refused by name
//! before the plan; unset, the draft goes off with a `load draft=off` record that names
//! `--mmproj`, and the lever words that name the draft are then refused as they are on any
//! server that drafts nothing. `BLOOMERY_PREFILL=steps` (one step an id) and the route trace
//! that needs it are refused by name beside `--mmproj` ([`refuse_media_feed`]). A sequence that
//! holds an image is kept out of the prompt cache: its rotation table is no part of a saved
//! state, so [`Seat::snapshot`] refuses it by name (the server keeps no state of it and serves
//! on).
//!
//! `BLOOMERY_DRAFT` unset follows the placement as `generate_qwen3moe`'s
//! does (`bloomery_levers::draft38_unset`): under `--place a` or `bp` the MTP draft
//! runs when a regular file is where it would be opened
//! (`BLOOMERY_MTP_DRAFT`, else the shared draft file beside the target); the
//! plain path runs under `--place gate`, with no file there, and with stores
//! too short for one window (a slot's share of `--ctx-size` under 5: the
//! total under 5 a slot), each printed as a `load draft=off (<why>)` record
//! after the `load` line (`no file at <path>[, <path>…]`, every place the pick looked), never a
//! refusal; `BLOOMERY_DRAFT=mtp` set with stores that short is refused by
//! name before the plan. Unset, the draft drafts on any `--parallel`, a
//! round of the busy slots' sampled and greedy requests one pass of their
//! windows. Unset, the draft
//! then yields to the context
//! (`bloomery_levers::RungCost::of`, at [`ctx38`]'s caller): it goes off
//! when the plan with it leaves a slot half the positions of the plan
//! without it or fewer, a set `--ctx-size` included, one `draft yield` record naming
//! its card bytes and the positions a slot gets either way, and the plain
//! plan's `ctx` line following; `BLOOMERY_DRAFT=mtp` set never yields. `BLOOMERY_DRAFT=off` is the plain
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
//! prompt call, a slot's last request's at the server's stop. A sampling or id-banning request drafts
//! too, alone in the round or beside other busy slots (its window among the round's one pass,
//! `rounds::pass_rows_one_pass`), each kept id its draw from its verified row
//! (`Seat::pass_sampled`; an id-banning request's draw sets the banned
//! logits to -inf first), the ids its pass alone draws.
//! Each step's row is the target's, read before the step is told to the
//! draft; its ids are the plain server's. A `load draft=mtp` line
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
//! (`BLOOMERY_QWEN3_UBATCH`, under `-ub`) is read once, with the flags. The
//! stderr lines named above are records of the kinds
//! `bloomery_gpu_gates::record` declares; `--records-schema` prints those
//! kinds and exits.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use app::mtp::{MtpBody, MtpDraft};
use bloomery_gpu::arch::qwen3moe::ubatch::UBATCH;
use bloomery_gpu::arch::qwen3moe::{
    Body38, Folder38, KvQ8, Model38, Prompt38, Seq38, TargetRows, checkpoint_budget38,
    checkpoint_price38, point_row_budget38, seq38_bytes,
};
use bloomery_gpu::checkpoint::saved::SaveLane;
use bloomery_gpu::host::census::TierCensus;
use bloomery_gpu::host::route_trace::RouteTrace;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::model::StepMode;
use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_gates::beside::{self, Beside};
use bloomery_gpu_gates::bind::{
    CacheRam, Cached, ChatSurface, Landing, Listen, Pass, PutBack, Seat, SeatEngine, SlotPassRow,
    SlotStep, SlotsFrom, TemplateFile, Vocab, arm_screen, begin, bind_server, eprint_noted,
    model_props, note_line, parallel_line, place_word, placement_of, put_back, slots_of,
};
use bloomery_gpu_gates::generate::{BreakEven, FloorShort, Lane, Place, lane_of};
use bloomery_gpu_gates::need;
use bloomery_gpu_gates::nodes::count_kinds;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::residency38::{
    CARD38, Lever38, Seqs, check_reserve, checkpoints_reserved, churn_bytes, reserve_checkpoints,
    residency38,
};
use bloomery_gpu_gates::{GateError, gpu_census, ref_model_path, side_seat, vision_seat};
use bloomery_levers::{
    ChurnHosting, Draft38At, Draft38Off, Paged, Residency38At, ResidencyPick, ResidencyWhy,
    draft38_unset, residency_word_least_fewest, residency38_least_fewest, residency38_unset,
};
use cuda_core::sys;
use gguf::Split;
use model::arch::models::headpick::HeadSet;
use model::arch::models::{Mixer, ModelSpec};
use model::arch::qwen35moe::head_list::{HeadPick, head_rows_of};
use model::arch::qwen35moe::place::{
    Experts, MtpInputs, PlaceError, PlanInputs, machine_for_experts, serve_ctx, tier_batch,
    widen_card_route,
};
use model::placement::churn::ChurnPool;
use model::placement::ctx_ask::{DEFAULT_SLOTS, Rule, card_experts_floor};
use model::placement::workstation::{
    CardSpec, HostNeed, HostRead, POINT_ROWS_RESERVE, host_available_read,
};
use model::placement::{ArenaFloor, Machine, Plan, PlanLevers};
use refset::arch::qwen4exp::VERIFIED_POSITIONS;
use refset::arch::qwen4exp::mtp::{DraftFrom, draft_file, draft_places};
use runtime::Target as _;
use runtime::seqstate::pinned;
use runtime::width::Mode as WidthMode;
#[cfg(feature = "vision")]
use serve::media::{MediaFeed, SharedMediaModel};
use serve::{
    CacheNote, DraftProps, Drafted, EngineProps, Pending, ResidencyReset, Saved, ServeError,
};
use tokenizer::Tokenizer;

use super::drafted::{ParkedDraft, SlotDrafts, lookup_props};

#[path = "../qwen38_place.rs"]
mod q38place;
#[path = "../qwen38_levers.rs"]
mod qwen38_levers;
pub(crate) use qwen38_levers::ACTS_ON;
#[path = "../route_trace38.rs"]
mod route_trace38;
use super::xstream_rule::{Stage, xstream_rule};
#[allow(
    clippy::duplicate_mod,
    reason = "three seats of one bin include this file"
)]
#[path = "../ubatch_rule.rs"]
mod ubatch_rule;
use route_trace38::{TraceRun, experts_name};
use ubatch_rule::ubatch_rule;

const USAGE: &str = "usage: bloomery-serve-qwen38 [--host H] [--port P] \
                     [--place a|gate|bp|<stage>[+<tier>…]] [--ctx-size C] [--alias NAME] \
                     [--cache-ram MIB] [--chat-template-file PATH] [--parallel N] \
                     [--queue-depth Q] [--park-ram MIB] [--slot-save-path DIR] \
                     [--mmproj FILE | --no-mmproj] [--image-min-tokens N] \
                     [--image-max-tokens N] [-ub N | --ubatch-size N] \
                     [--cache-type-k f16|q8_0] [--plan] [--api-key KEY] [--api-key-file FNAME]";

/// The context a slot asks before the context is chosen (the placement's
/// offer, the draft's window): the plan every Qwen3.8 measurement ran at, and
/// `generate_qwen3moe`'s default.
const CTX: usize = 4096;

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind38 {
    /// The stage on the largest visible card (the A6000 here, the timing
    /// card; the default), and any single card a list names: the plan is
    /// made on that card's own bytes.
    A,
    /// The word `gate`: the 3090, the gate card.
    Gate,
    /// Plan (b′): the stage as under `a`, the next-largest card its expert
    /// tier (`q38place::machine_bp38`), and any card list with a tier card.
    Bp,
}

/// Where `--place` puts the plan's stage card, and its expert tier card
/// when it has one: the kind, and its cards (`generate::Place`), resolved
/// against this process's devices once the arguments are read.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Place38 {
    pub(crate) kind: Kind38,
    pub(crate) cards: Place,
    /// The side model's reserve on the stage card, which every machine of the placement
    /// carries ([`Place38::machine`]).
    embedding: Option<u64>,
    /// The image encoder's reserve on the stage card, when its tower sits there
    /// ([`vision_seat::images_of`]), which every machine of the placement carries
    /// ([`Place38::machine`]).
    image: Option<u64>,
    /// The load is a media load (`--mmproj`): every sequence holds a rotation table of its own,
    /// which every machine of the placement counts ([`Place38::machine`]).
    media: bool,
}

impl Place38 {
    const A: Place38 = Place38 {
        kind: Kind38::A,
        cards: Place::A,
        embedding: None,
        image: None,
        media: false,
    };

    /// The seat's reading of a `--place` word: the shared grammar
    /// ([`place_word`]) with the one expert tier card the body serves
    /// ([`q38place::Q38_RULE`]), and the kind it names ([`Place38::of`]). `tp`
    /// is refused by name: the body has no tensor-parallel path.
    pub(crate) fn parse(v: &str) -> Result<Place38, GateError> {
        let cards = place_word(v, Some(("qwen4exp", q38place::Q38_RULE.tiers)))?;
        if cards.tp() {
            return Err(
                "--place tp: a tensor-parallel run is planned by its family's own \
                        path, and the Qwen3.8 seat has none"
                    .into(),
            );
        }
        Ok(Place38::of(cards))
    }

    /// `cards` as a placement of the seat, its kind read by structure: a tier
    /// card is plan (b′), the word `gate` the gate card, any other stage —
    /// `a`, or a single card a list names — plan (a) on that card's own
    /// bytes.
    fn of(cards: Place) -> Place38 {
        let kind = if !cards.tier_cards().is_empty() {
            Kind38::Bp
        } else if cards == Place::Gate {
            Kind38::Gate
        } else {
            Kind38::A
        };
        Place38 {
            kind,
            cards,
            embedding: None,
            image: None,
            media: false,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        self.cards.name()
    }

    /// The stage card's spec.
    fn spec(self) -> Result<CardSpec, GateError> {
        Ok(self.cards.card_specs()?[0])
    }

    /// The stage card is plan (a)'s: the placement the Qwen3.8 defaults
    /// treat as plan (a).
    pub(crate) fn stage_a(self) -> bool {
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
    pub(crate) fn tiered(self) -> bool {
        matches!(self.kind, Kind38::Bp)
    }

    /// The machine a plan of `inputs` at `ctx` positions and ubatches of
    /// `ub` runs on under `experts`, `mtp` the draft when it runs beside
    /// the target, for `slots` resident sequences: plan (b′) reserves the
    /// draft's card bytes at `ctx` for every slot on its stage card
    /// (`MtpInputs::card_bytes_of`, the reserve `plan_mtp_with_slots`
    /// checks; `q38place::machine_bp38`), the one-card plans count them in
    /// `plan_mtp_with_slots`. The host reserves every slot's checkpoints
    /// ([`seqs38`] through [`reserve_checkpoints`], priced at `ctx` positions
    /// a slot) and, under the draft, every slot's checkpoint hidden rows
    /// ([`POINT_ROWS_RESERVE`], one row a point, [`point_row_budget38`]), which the plan's host need, the NVMe tier's
    /// floor and its arena then count. The stage card carries the side model's reserve, the
    /// image encoder's when its tower sits there and, on a media load, a rotation table of `ctx`
    /// rows for every slot ([`vision_seat::TABLES_RESERVE`]). Its card route runs the whole
    /// ubatch at once when the plan under `levers` keeps every card expert by it
    /// ([`widen_card_route`]).
    fn machine(
        self,
        inputs: &PlanInputs,
        ctx_ub: (u64, u64),
        experts: Experts,
        levers: &PlanLevers,
        mtp: Option<&MtpInputs>,
        slots: usize,
    ) -> Result<Machine, GateError> {
        Ok(self
            .machine_read(inputs, ctx_ub, (experts, levers, mtp, slots), |_| ())?
            .0)
    }

    /// [`Self::machine`] and `read` of its plan ([`PlanInputs::plan_target`] at
    /// `ctx`), which the machine's own choice made, so the caller plans it no
    /// more.
    fn machine_read<T>(
        self,
        inputs: &PlanInputs,
        (ctx, ub): (u64, u64),
        (experts, levers, mtp, slots): (Experts, &PlanLevers, Option<&MtpInputs>, usize),
        read: impl Fn(&Plan<'_>) -> T,
    ) -> Result<(Machine, T), GateError> {
        let layers = inputs.spec.layers.len();
        let mut machine = match self.kind {
            Kind38::A | Kind38::Gate => machine_for_experts(self.spec()?, layers, ub, experts),
            Kind38::Bp => {
                let draft = mtp.map(|m| m.card_bytes_of(ctx, slots)).transpose()?;
                q38place::machine_bp38(self.cards, layers, ub, draft, tier_batch(&inputs.spec, ub))?
            }
        };
        beside::reserve_on(
            &mut machine,
            Beside {
                image: self.image,
                embedding: self.embedding,
            },
        );
        if self.media {
            let rows = u64::try_from(slots)?.saturating_mul(ctx);
            let head = inputs.spec.gqa().ok_or(
                "the description holds no attention layer to rotate an image's positions by",
            )?;
            let width = head.rope.dims as usize;
            machine.cards[0].reserves.push((
                vision_seat::TABLES_RESERVE.to_owned(),
                rows.saturating_mul(vision_seat::table_bytes_per_position(width)),
            ));
        }
        reserve_checkpoints(&mut machine, seqs38(inputs, slots), ctx)?;
        if mtp.is_some() {
            let (gdn, qsa) = layer_counts(inputs);
            let bytes = point_row_budget38(gdn, qsa, ctx)?;
            machine
                .host_mut()
                .reserves
                .push((POINT_ROWS_RESERVE.to_owned(), pinned(slots, bytes)));
        }
        Ok(widen_card_route(
            machine,
            ub,
            |m, on| {
                inputs
                    .plan_target(m, (ctx, ub), levers, experts, mtp, slots)
                    .map(|p| on(&p))
            },
            read,
        )?)
    }
}

/// What the seat's load of `slots` resident sequences of the file `inputs`
/// describes declares ([`reserve_checkpoints`]): every slot takes checkpoints,
/// the seat turning them on (`set_checkpoints`), each priced by the body's own
/// derivation ([`checkpoint_price38`]) from the file's delta and selecting
/// layers.
fn seqs38(inputs: &PlanInputs, slots: usize) -> Seqs {
    let (gdn, qsa) = layer_counts(inputs);
    Seqs::priced(slots, checkpoint_price38(gdn, qsa))
}

/// The delta layers of the file `inputs` describes, and its other
/// (selecting) layers.
fn layer_counts(inputs: &PlanInputs) -> (usize, usize) {
    let layers = &inputs.spec.layers;
    let gdn = layers
        .iter()
        .filter(|l| matches!(l.mixer, Mixer::DeltaRule(_)))
        .count();
    (gdn, layers.len() - gdn)
}

/// The MTP draft file at `path`, which `from` picked; an error names both.
fn open_draft(path: &Path, from: DraftFrom) -> Result<Split, GateError> {
    if path.is_dir() {
        return super::qwen38_folder::draft(path);
    }
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

/// A slot must hold the whole MTP window at its chosen context.
fn check_mtp_ctx(ctx: usize, slots: usize) -> Result<(), GateError> {
    if slots.saturating_mul(ctx) < window_positions(slots) {
        return Err(format!(
            "a slot's context of {ctx} leaves no positions for the MTP draft's window; \
             --ctx-size names a total of at least {}",
            window_positions(slots)
        )
        .into());
    }
    Ok(())
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

/// The feed the lookup draft refuses: the route trace and the step feed run
/// one-row steps, where the lookup verifies its proposal in a pass and takes
/// its context from the prompt call's ids.
pub(crate) fn refuse_lookup_feed(feed: Feed38) -> Result<(), GateError> {
    if feed.traced || feed.stepped() {
        return Err(
            "BLOOMERY_DRAFT=lookup verifies its proposal in a pass and takes its context from \
             the prompt call's ids; BLOOMERY_ROUTE_TRACE and BLOOMERY_PREFILL=steps run one-row \
             steps: set BLOOMERY_DRAFT=off"
                .into(),
        );
    }
    Ok(())
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

/// The MTP draft's own lever set, which the lookup draft does not read:
/// its head and its file. `BLOOMERY_MTP_WIDTH` is the width chooser's,
/// which both drafts run behind.
fn mtp_only_lever(levers: &bloomery_levers::Levers) -> Option<&'static str> {
    if levers.mtp_head_rows().is_some() {
        Some(bloomery_levers::MTP_HEAD_ROWS)
    } else if levers.mtp_draft().is_some() {
        Some(bloomery_levers::MTP_DRAFT)
    } else {
        None
    }
}

/// Refuses draft-only levers when the draft's final rule serves the plain path.
fn refuse_mtp_levers(
    levers: &bloomery_levers::Levers,
    draft_off: &Option<Draft38Off>,
) -> Result<(), GateError> {
    if let Some(why) = draft_off
        && let Some(lever) = mtp_only_lever(levers)
    {
        return Err(
            format!("{lever} is the MTP draft's; the server drafts nothing ({why})").into(),
        );
    }
    bloomery_gpu_gates::bind::refuse_plain_width(levers, draft_off.as_ref())
}

/// What an image load (`--mmproj`) refuses of the feed: `BLOOMERY_PREFILL=steps`, which feeds a
/// prompt one step an id. A prompt call that holds images enters their rows only through the
/// ubatch walk's media call (`Seat::prefill_media`), so the step feed would serve one path to the
/// text and another to the images; `BLOOMERY_PREFILL=batch` is the way out. The route trace needs
/// the step feed ([`refuse_feed`]), so it is refused here too.
pub(crate) fn refuse_media_feed(feed: Feed38) -> Result<(), GateError> {
    if feed.stepped() {
        return Err(
            "--mmproj with BLOOMERY_PREFILL=steps: an image's rows enter only through the ubatch \
             walk's media call, never one step an id; set BLOOMERY_PREFILL=batch, or drop \
             --mmproj"
                .into(),
        );
    }
    Ok(())
}

/// The draft kinds this seat serves, before any model is read.
fn check_draft_word(levers: &bloomery_levers::Levers) -> Result<(), GateError> {
    match levers.draft() {
        None | Some("mtp" | "lookup" | "off") => Ok(()),
        Some(other) => Err(format!(
            "BLOOMERY_DRAFT={other}: on a qwen4exp file mtp drafts the window, lookup the n-gram \
             lookup and off runs the plain path; dspark is the V4.1 binaries'"
        )
        .into()),
    }
}

/// `BLOOMERY_DRAFT` on the seat at `place` with stores of `ctx` positions
/// in all, split among `slots` resident sequences, `file` the MTP draft file
/// the load would open, `inputs` the target's: `mtp` the draft, `off` the
/// plain path; unset, `generate_qwen3moe`'s rule
/// (`bloomery_levers::draft38_unset`), then `off` when the target's matrices
/// the draft borrows are not its format (`PlanInputs::mtp_borrows`, which
/// refuses a set `mtp` at `MtpInputs::read`); and, drafting nothing, why (the
/// `load draft=off` record's); `lookup` the lookup draft in its place
/// (`drafted::Lookups`), the MTP draft off with no why. This is the rule
/// before the plan: the unset draft's yield to the context
/// (`bloomery_levers::RungCost::of`, the module doc) is decided after the
/// context choice, at the caller. `dspark` and any other word are refused by
/// name.
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
    check_draft_word(levers)?;
    match levers.draft() {
        Some("mtp") => Ok((true, None)),
        Some("lookup") => Ok((false, None)),
        Some("off") => Ok((false, Some(Draft38Off::Set))),
        Some(_) => unreachable!("the draft word was checked"),
        None => {
            let at = Draft38At {
                place_a: place.stage_a(),
                logits: false,
                route_trace: levers.route_trace().is_some(),
                file,
                file_is_there: file.is_file() || file.is_dir(),
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

/// [`draft38`]'s answer `(mtp, off)` on a load that carries no MTP draft, for the reason `why`
/// ([`Draft38Off::Media`]: an image load, whose sequences the engine's `GpuModel::enable_media`
/// refuses a draft beside; [`Draft38Off::Folder`]: a checkpoint folder without an MTP layer;
/// [`Draft38Off::FolderMtp`]: one whose MTP layer the draft does not read): `BLOOMERY_DRAFT=mtp`
/// is refused by name; the other words keep the plain path, `off`
/// as set, and a draft the rule would have run goes off with `why`, while a reason it was
/// already off for stands. The lever words that name the draft are then refused by the seat's
/// own rule (`refuse_mtp_levers`).
pub(crate) fn draft_off_for(
    why: Draft38Off,
    word: Option<&str>,
    (mtp, off): (bool, Option<Draft38Off>),
) -> Result<(bool, Option<Draft38Off>), GateError> {
    if word == Some("mtp") {
        let load = match &why {
            Draft38Off::Folder => "with a checkpoint folder that holds no MTP layer".to_string(),
            Draft38Off::FolderMtp { why } => {
                format!("with a checkpoint folder whose MTP layer the draft does not read ({why})")
            }
            _ => "with --mmproj: an image load carries no MTP draft (the media load refuses it); \
                  drop --mmproj"
                .to_string(),
        };
        return Err(format!(
            "BLOOMERY_DRAFT=mtp {load}, or set BLOOMERY_DRAFT=off (unset, the draft goes off by \
             itself)"
        )
        .into());
    }
    Ok(if mtp {
        (false, Some(why))
    } else {
        (false, off)
    })
}

/// [`draft38`]'s answer `(mtp, off)` on a load whose draft is the checkpoint folder's own MTP
/// layer, `layer` as the folder prepared it (`Ok(None)`: the folder holds none; `Err`: the
/// preparation's refusal): a prepared layer drafts when `reads`, the draft's reader, takes it,
/// asked only while the draft is on; none, or one the preparation or the reader refuses, is
/// [`draft_off_for`]'s ([`Draft38Off::Folder`], [`Draft38Off::FolderMtp`]).
pub(crate) fn folder_draft_for(
    layer: &Result<Option<Split>, String>,
    reads: impl FnOnce(&Split) -> Result<(), GateError>,
    word: Option<&str>,
    (mtp, off): (bool, Option<Draft38Off>),
) -> Result<(bool, Option<Draft38Off>), GateError> {
    let why = match layer {
        Ok(None) => Draft38Off::Folder,
        Err(why) => Draft38Off::FolderMtp { why: why.clone() },
        Ok(Some(_)) if !mtp => return Ok((mtp, off)),
        Ok(Some(layer)) => match reads(layer) {
            Ok(()) => return Ok((mtp, off)),
            Err(e) => Draft38Off::FolderMtp { why: e.to_string() },
        },
    };
    draft_off_for(why, word, (mtp, off))
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
/// ([`PlanInputs::plan_target`]).
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
    /// The ubatch size the seat read ([`Args::ubatch`]), before a context clips it.
    ubatch: usize,
    /// The checkpoint folder the load reads its card's parts from; `None` for a GGUF file.
    folder: Option<&'a Folder38>,
    /// The card experts a layer a default context leaves ([`default_floor`]);
    /// `None` where the load runs no residency machine to leave them for.
    floor: Option<u64>,
}

impl Plans<'_> {
    /// Whether a plan takes `ctx` positions a sequence serves, or why none
    /// does. A folder's plan also holds the least of its parts' reads
    /// ([`Folder38::least_reads`], [`folder_holds`]).
    fn holds(&self, ctx: usize) -> Result<(), GateError> {
        self.holds_leaving(ctx, None)
    }

    /// [`Plans::holds`], and with `least` the plan also leaves every layer
    /// of the card that many experts ([`card_experts_floor`]): a default
    /// context's fit, which would otherwise spend the card's room on context
    /// and leave the unset residency none.
    fn holds_leaving(&self, ctx: usize, least: Option<u64>) -> Result<(), GateError> {
        self.read(ctx, |plan| -> Result<(), GateError> {
            if let Some(folder) = self.folder {
                folder_holds(plan, folder.least_reads(plan, CARD38)?)?;
            }
            if let Some(least) = least {
                card_experts_floor(plan, CARD38, least)?;
            }
            Ok(())
        })?
        .1
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
        let (machine, read) = self.read(ctx, |plan| -> Result<_, GateError> {
            let lever = match set {
                Some((r, word)) => Lever38::Set(r, word),
                None => Lever38::Unset(residency38_unset(self.feed.residency_at(self.place))),
            };
            // An unhosted pool takes no host bytes: the load only knows it.
            let pool = match residency38(
                <Body38 as bloomery_gpu::model::ChainBody>::UNHOSTED_POOL,
                plan,
                lever,
                |_| {},
            )? {
                (Residency::Mid { pinned, .. }, ChurnHosting::Hosted) => {
                    ChurnPool::of(plan, CARD38, pinned)
                        .map_err(|e| format!("the churn pool: {e}"))?
                        .bytes
                }
                _ => 0,
            };
            Ok((
                HostNeed::of(plan, 0).bytes(),
                pool,
                plan.host.nvme_arena_bytes,
                plan.host.nvme_expert_bytes,
            ))
        })?;
        let (need, pool, arena, nvme) = read?;
        Ok((need, pool, arena, checkpoints_reserved(&machine), nvme))
    }

    /// The machine a load of `ctx` positions a sequence runs on
    /// ([`Place38::machine`]) and `read` of its plan.
    fn read<T>(
        &self,
        ctx: usize,
        read: impl Fn(&Plan<'_>) -> T,
    ) -> Result<(Machine, T), GateError> {
        let ub = self.ubatch.min(ctx);
        self.place.machine_read(
            self.inputs,
            (u64::try_from(ctx)?, u64::try_from(ub)?),
            (self.experts, self.levers, self.mtp, self.slots),
            read,
        )
    }
}

/// The most host bytes a sequence state of `positions` positions holds on the
/// file `inputs` describes ([`Seq38`]: its positional rows and bounded recurrent
/// copies, the draft's side under `drafts`, [`seq38_bytes`]) — at one
/// position the least one saved state of the prompt cache holds.
fn seq38_state(inputs: &PlanInputs, positions: usize, drafts: bool) -> u64 {
    let (gdn, qsa) = layer_counts(inputs);
    seq38_bytes(gdn, qsa, positions, drafts, KvQ8::of(inputs.kv.is_q8()))
}

/// The context the seat loads ([`ctx38`]): a slot's own, the stores' rows
/// one sequence serves, the choice's one `ctx` line and what set it.
struct Ctx38 {
    ctx: usize,
    line: String,
    rule: Rule,
    /// The line that says the default could not leave the unset residency
    /// its experts at any context ([`ctx38`]), `None` when it did or the
    /// context was not the default's.
    unmet: Option<String>,
}

/// Whether a checkpoint folder's `plan` holds `reads`, the least of its
/// load's part reads (`HostNeed::read_ahead`), beside its host need at the
/// room the plan was made at (`HostNeed::check_planned`), its NVMe tier arena
/// at the floor the load's own check may shrink it to (`HostNeed::grant`,
/// `ArenaFloor::of`). On a unified pool a shorter context is what that
/// refusal asks for (`HostShort::pool_short`).
pub(crate) fn folder_holds(plan: &Plan<'_>, reads: u64) -> Result<(), GateError> {
    let need = HostNeed::of(plan, 0);
    HostNeed {
        read_ahead: reads,
        arena: if need.arena > 0 {
            ArenaFloor::of(plan)?.bytes
        } else {
            0
        },
        ..need
    }
    .check_planned(plan)?;
    Ok(())
}

/// The card experts a layer a default context leaves the residency, where
/// the plan's expert rule puts experts on the card: a set `mid` word's own
/// least ([`residency_word_least_fewest`]); unset,
/// [`residency38_least_fewest`], the fewest the unset word runs on, where the
/// rule runs the machine on this load's plan — the stage plan (a) or (b′), no
/// route trace or step feed. `None` where nothing runs to leave them for
/// (`off`, the host's experts, the unset rule off), and the context takes the
/// room.
fn default_floor(
    place: Place38,
    feed: Feed38,
    experts: Experts,
    set: Option<(Residency, &str)>,
) -> Option<u64> {
    if !matches!(experts, Experts::Card) {
        return None;
    }
    match set {
        Some((_, word)) => residency_word_least_fewest(word),
        None => residency38_unset(feed.residency_at(place))
            .is_none()
            .then(residency38_least_fewest),
    }
}

/// The seat's context (the module doc) by the family's common choice
/// (`q38place::ctx_choice`): a set `--ctx-size` (`set`, the total the slots
/// split) whose share passes the limit stops by name, and one the machine
/// cannot hold stops with the largest share it would hold; unset, a slot asks
/// the target and fits under it while the plan of every slot is short there
/// ([`Plans::holds`]) or leaves a layer of the card fewer experts than the
/// unset residency runs on ([`Plans::floor`]). Where not even the least
/// context leaves them the card cannot run the residency at any context the
/// default tries: the context is fitted as if it left none, the residency
/// resolves `off`, and `unmet` names the floor as the cause. A tier card the
/// plan leaves idle is refused as itself: no context gives it an expert.
fn ctx38(plans: &Plans<'_>, set: Option<usize>) -> Result<Ctx38, GateError> {
    let ask = (set, plans.slots);
    let mut unmet = None;
    let floored = match plans.floor.filter(|_| set.is_none()) {
        None => None,
        Some(least) => {
            let probe = |c: u64| plans.holds_leaving(usize::try_from(c)?, Some(least));
            match q38place::ctx_choice(plans.inputs, ask, probe) {
                Ok(chosen) => Some(chosen),
                Err(e) => {
                    let short = e.downcast_ref::<FloorShort>().map(|f| (f.floor, f.short));
                    let Some((floor, short)) = short else {
                        return Err(e);
                    };
                    unmet = Some(format!(
                        "ctx experts-floor=unmet least={least} floor_ctx={floor} slots={}: the \
                         residency needs {least} experts in every layer of the card, {} B, and \
                         the card holds {} B of them at {floor} positions a slot, the least \
                         context the default tries: the unset word resolves off and a set word \
                         is refused; fewer slots or a smaller model file leave room",
                        plans.slots, short.need, short.room
                    ));
                    None
                }
            }
        }
    };
    let chosen = match floored {
        Some(chosen) => chosen,
        None => q38place::ctx_choice(plans.inputs, ask, |c| plans.holds(usize::try_from(c)?))?,
    };
    Ok(Ctx38 {
        ctx: usize::try_from(chosen.slot_ctx)?,
        line: chosen.line(),
        rule: chosen.rule,
        unmet,
    })
}

pub(crate) struct Args {
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
    template_file: TemplateFile,
    /// `--slot-save-path`: the directory the slot actions answer from;
    /// `None` refuses every one, as llama-server does.
    slot_save_path: Option<PathBuf>,
    /// `--parallel`: the resident sequences the seat serves; `None` takes
    /// one slot beside a set `--ctx-size`, else `ctx_ask::DEFAULT_SLOTS`
    /// (one).
    parallel: Option<usize>,
    queue_depth: Option<usize>,
    /// `--park-ram` in bytes; set is refused by name (resident slots park
    /// nothing).
    park_ram: Option<u64>,
    /// The image-input flags: `--mmproj` is the encoder file whose images the seat serves, none
    /// serves no image.
    vision: vision_seat::Flags,
    /// `--api-key`/`--api-key-file`: the keys every request is checked
    /// against ([`serve::flag::ApiKeys`]).
    api_keys: serve::flag::ApiKeys,
    /// `-ub` / `--ubatch-size`: the most positions a ubatch walk runs, over
    /// `BLOOMERY_QWEN3_UBATCH` ([`ubatch_rule`]).
    ubatch: Option<usize>,
    /// `--cache-type-k`: the KV cache's format ([`KvQ8::choose`]).
    cache_type_k: Option<String>,
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
        template_file: TemplateFile::default(),
        slot_save_path: None,
        parallel: None,
        queue_depth: None,
        park_ram: None,
        vision: vision_seat::Flags::default(),
        api_keys: serve::flag::ApiKeys::default(),
        ubatch: None,
        cache_type_k: None,
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
        // `--no-mmproj` has no value: the image flags read theirs from `it` themselves.
        if a.vision.take(flag, &mut it)? {
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
            TemplateFile::FLAG => a.template_file.set(v),
            "--slot-save-path" => a.slot_save_path = Some(PathBuf::from(v)),
            "--parallel" | "-np" => a.parallel = Some(serve::flag::number(flag, v)?),
            "--queue-depth" => a.queue_depth = Some(serve::flag::number(flag, v)?),
            "--park-ram" => a.park_ram = Some(CacheRam::parse_mib(flag, v)?),
            f if serve::flag::KEYS.contains(&f) => a.api_keys.add(f, v)?,
            f if serve::flag::UBATCH.contains(&f) => a.ubatch = Some(serve::flag::number(flag, v)?),
            "--cache-type-k" => a.cache_type_k = Some(v.to_owned()),
            "--cache-type-v" => {
                return Err(
                    "--cache-type-v does not exist: the qwen4exp cache quantizes its K and V \
                     planes together, so --cache-type-k names the one format both take"
                        .into(),
                );
            }
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
    bloomery_gpu_gates::bind::check_slot_ctx(a.ctx, a.parallel)?;
    Ok(a)
}

/// The levers (unless `read_levers` is false), flags and ubatch size ([`ubatch_rule`]) [`run`]
/// reads first, checked before the model is fetched ([`bloomery_gpu_gates::bind::check_seat`]).
pub(crate) fn check(
    args: &[String],
    read_levers: bool,
    from_repo: bool,
) -> Result<bloomery_gpu_gates::bind::CheckedSeat<Args>, GateError> {
    bloomery_gpu_gates::bind::check_seat(read_levers.then_some(ACTS_ON), args, |args, levers| {
        let a = parse_args(args)?;
        a.template_file.check()?;
        a.vision.check_before_model(from_repo)?;
        ubatch_rule(a.ubatch)?;
        let kv_set = if let Some(levers) = levers {
            levers.qwen3_kv_set()
        } else {
            None
        };
        KvQ8::choose(a.cache_type_k.as_deref(), kv_set, KvQ8::F16)?;
        if let Some(levers) = levers {
            check_draft_word(levers)?;
            let feed = Feed38::of(levers)?;
            if levers.draft() == Some("lookup") {
                refuse_lookup_feed(feed)?;
                if let Some(lever) = mtp_only_lever(levers) {
                    return Err(format!(
                        "{lever} is the MTP draft's; BLOOMERY_DRAFT=lookup serves the lookup draft"
                    )
                    .into());
                }
            }
            if levers.draft() == Some("off") {
                refuse_mtp_levers(levers, &Some(Draft38Off::Set))?;
            }
            let residency = levers.residency().filter(|&r| r != "off");
            refuse_feed(
                feed,
                (a.parallel.unwrap_or(1), levers.draft() == Some("mtp")),
                residency,
            )?;
            if a.vision.mmproj()?.is_some() {
                refuse_media_feed(feed)?;
                draft_off_for(
                    Draft38Off::Media,
                    levers.draft(),
                    (levers.draft() == Some("mtp"), None),
                )?;
            }
            if levers.draft() == Some("mtp") {
                let (slots, _) = slots_of(a.parallel, a.ctx.is_some(), DEFAULT_SLOTS)?;
                if let Some(ctx) = a.ctx {
                    check_mtp_ctx(ctx / slots, slots)?;
                }
            }
            WidthMode::of(levers.mtp_width())?;
            experts38(levers)?;
        }
        Ok(a)
    })
}

/// What `--mmproj` opened for the plans: the encoder file held against the text model's embedding
/// width, its vocabulary's image ids, the description's IMROPE sections and its PLE image token
/// (`token` the vocabulary's id to text), and seated for
/// a plan on `place` ([`vision_seat::open_seat`]); the file is `--mmproj`'s, else, where the seat
/// `takes` the `--hf` repo's projector, that one ([`vision_seat::Flags::encoder`]), and `None`
/// without either. A build without the `vision` feature, whose tower is not built, refuses
/// `--mmproj` by name.
fn vision_of(
    flags: &vision_seat::Flags,
    (place, takes): (Place, bool),
    spec: &ModelSpec,
    token: &dyn Fn(u32) -> Option<String>,
) -> Result<Option<vision_seat::Opened>, GateError> {
    flags.check_before_model(bloomery_gpu_gates::model_file::hf_repo().is_some())?;
    let file = flags.encoder(takes)?;
    vision_seat::open_seat(
        flags,
        file.as_deref(),
        place,
        &[vision::arch::qwen3vl::PROJECTOR_TYPE],
        &vision_seat::TextFacts {
            ple: vision_seat::PleImage::of_spec(spec),
            ..vision_seat::TextFacts::of_imrope(spec)?
        },
        token,
    )
}

/// `at` seated for the plan on `place`, the placement the rule chose: where the tower sits depends
/// on the cards the plan uses ([`vision_seat::encoder_seat`]), and the placement is known only
/// after the rule asked the offer's plan, so the file is opened and held against the text model
/// first, on the flag's own placement, and seated here.
fn reseated(at: vision_seat::Opened, place: Place) -> Result<vision_seat::Opened, GateError> {
    let seat = vision_seat::encoder_seat(place, at.card.total())?;
    Ok(vision_seat::Opened { seat, ..at })
}

/// Loads the model, and the embedding model `side` names beside it, and serves until the listener
/// or the engine fails; `Ok` carries why the server ended.
pub fn run(args: &[String], side: &side_seat::Flags) -> Result<ServeError, GateError> {
    record::at_main_with_flags(
        "bloomery-serve-qwen38",
        record::BLOOMERY_SERVE_QWEN38,
        |flag| bloomery_gpu_gates::llama_launch::serve_option(args, flag),
    );
    let (mut a, levers) = check(
        args,
        true,
        bloomery_gpu_gates::model_file::hf_repo().is_some(),
    )?
    .run();
    model::ops::set_host_lanes(levers.host_lanes());

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
    // The one ubatch size the plans and the load take, each clipped to its context.
    let ubatch = ubatch_rule(a.ubatch)?
        .asked()
        .map_or(UBATCH, |ask| ask.size);
    let side = side.read()?;
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
    // A draft path that is a directory is a checkpoint folder's MTP layer, the target's or another's.
    let draft_in_folder = draft_path.is_dir();
    let draft_looked = draft_places(&path);
    // A checkpoint folder is opened once, here: its header plans as a split's first shard
    // does, its tokenizer is the vocabulary, its MTP layer is the draft's own, and the engine
    // thread loads its parts.
    let (split, folder, folder_tok, folder_draft) = if path.is_dir() {
        let f = super::qwen38_folder::open(&path)?;
        (f.header, Some(f.parts), Some(f.tokenizer), Some(f.draft))
    } else {
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        (split, None, None, None)
    };
    // The stage card's first open is made before the room is read
    // (`Gpu::hold_card`: its context and the bundle), so the plans' room has
    // lost its host bytes as the load's check after its open has; plan
    // (b′)'s tier card is held once chosen, below.
    let stage = draft_at.cards.on(&census)?.card_specs()?[0];
    Gpu::hold_card(stage.name, stage.device)?;
    // The lane's rate, taken before the plan reads the room (`lane_of`).
    // `--plan` probes no lane. The hold above stays first, for a load and
    // for `--plan`.
    let lane = if a.plan_only {
        None
    } else {
        Some(lane_of(stage)?)
    };
    let mut inputs = PlanInputs::describe(&split)?;
    // The cache format, before any plan: every plan the seat asks counts its bytes.
    let kv = KvQ8::choose(
        a.cache_type_k.as_deref(),
        levers.qwen3_kv_set(),
        KvQ8::of(folder.is_some()),
    )?;
    inputs.kv = inputs.kv.in_q8_if(kv == KvQ8::Q8);
    // The slot count first: it shapes the draft's window condition, the
    // context choice itself, and every plan the seat asks for, each counting
    // every sequence (`Plans::slots`). A set `--ctx-size` with no
    // `--parallel` is one request's context — one slot at the whole of it
    // (`bind::slots_of`).
    // The route trace records one sequence's steps, so its default is one.
    let (slots, from) = match slots_of(a.parallel, a.ctx.is_some(), DEFAULT_SLOTS)? {
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
    // The folder's own MTP layer drafts when the draft's reader takes it, whatever head the
    // load picks below; none, or one the preparation or the reader refuses, refuses a set `mtp`
    // and turns an unset draft off before the image seat reads the draft's word, the target
    // served as stored.
    let (mtp, draft_off) = match folder_draft.as_ref().filter(|_| draft_path == path) {
        Some(layer) => folder_draft_for(
            layer,
            |d| {
                MtpInputs::read(d, &split, &inputs, HeadSet::Full)?;
                Ok(())
            },
            levers.draft(),
            (mtp, draft_off),
        )?,
        None => (mtp, draft_off),
    };
    // The folder's own MTP layer, opened with the folder above, is the draft both threads read:
    // the folder is not opened again for it.
    let own_draft = folder_draft
        .and_then(|layer| layer.ok().flatten())
        .filter(|_| draft_path == path);
    // A draft lever set on a server that drafts nothing is refused, with
    // why — asked once the yield below has said its last word, so a draft
    // the plan turns off refuses them the same as one the rule did.
    let lookup = levers.draft() == Some("lookup");
    let width = WidthMode::of(levers.mtp_width())?;
    let experts = experts38(&levers)?;
    let plan_levers = PlanLevers::from_levers(&levers)?;
    let tok = match folder_tok {
        Some(tok) => tok,
        None => Tokenizer::from_gguf(&path)?,
    };
    // `--mmproj`'s encoder file, held against the text model and the vocabulary before anything
    // is planned. The `--hf` repo's projector is taken only where the load can serve images and
    // keep what it runs without them: the draft rule's first word is off and the prompt is not
    // fed a step an id. An image load takes no MTP draft ([`draft_off_for`]).
    let token = |id: u32| tok.text(id).map(str::to_owned);
    let vision = vision_of(
        &a.vision,
        (draft_at.cards, !mtp && !feed.stepped()),
        &inputs.spec,
        &token,
    )?;
    let (mtp, draft_off) = if vision.is_some() {
        refuse_media_feed(feed)?;
        draft_off_for(Draft38Off::Media, levers.draft(), (mtp, draft_off))?
    } else {
        (mtp, draft_off)
    };
    let ChatSurface { template, name, .. } = ChatSurface::read(
        &path,
        split.value("tokenizer.chat_template"),
        split.value("general.name"),
        a.template_file.path(),
        "qwen3.8",
    )?;
    let vocab = Arc::new(Vocab::new(tok)?.with_chat_template(&template)?);

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
            let opened;
            let draft = match &own_draft {
                Some(d) => d,
                None => {
                    opened = open_draft(&draft_path, draft_from)?;
                    &opened
                }
            };
            Some(MtpInputs::read(draft, &split, &inputs, h.rows.clone())?)
        }
    };
    // The folder's header goes to the engine thread with its parts; a split's is reopened there.
    let folder = folder.map(|parts| parts(split));
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
    // choice below asks again at the chosen placement — names the tier count
    // the rule holds against its break-even; a refused plan of the offer
    // runs `a`, the refusal named on stderr, and an unset flag never refuses
    // a load that `--place a` serves.
    let chosen = q38place::choose(a.flag.map(|p| p.cards), &census, |offer| {
        let seat = side
            .as_ref()
            .map(|s| vision_seat::encoder_seat(offer, s.bytes))
            .transpose()?;
        q38place::tier_experts(
            &inputs,
            offer,
            (
                u64::try_from(a.ctx.unwrap_or(CTX))?,
                u64::try_from(ubatch.min(a.ctx.unwrap_or(CTX)))?,
            ),
            experts,
            &plan_levers,
            mtp_inputs.as_ref(),
            q38place::Residents {
                seqs: seqs38(&inputs, slots),
                held: Beside {
                    image: None,
                    embedding: seat.as_ref().and_then(|s| s.reserve),
                },
            },
        )
    })?;
    chosen.record().eprint();
    // The side model's seat before the plans, each of which holds its bytes when it sits on the
    // stage card.
    let side_seat = side.as_ref().map(|s| s.seat(chosen.place)).transpose()?;
    // The encoder on the placement the rule chose: images never move the placement, which is the
    // one the same run without them takes.
    let vision = vision.map(|v| reseated(v, chosen.place)).transpose()?;
    a.place = Place38 {
        kind: match a.flag {
            Some(f) => f.kind,
            None if chosen.place.tier_cards().is_empty() => Kind38::A,
            None => Kind38::Bp,
        },
        cards: chosen.place,
        embedding: side_seat.as_ref().and_then(|s| s.reserve),
        image: None,
        media: false,
    };
    // What a default context leaves the unset residency, decided once for every plan below.
    let floor = default_floor(a.place, feed, experts, set);
    // Plan (b′)'s tier cards held too, and the room read again: every plan
    // below and the engine thread's take it.
    let tiers = &a.place.cards.card_specs()?[1..];
    if !tiers.is_empty() {
        for t in tiers {
            Gpu::hold_card(t.name, t.device)?;
        }
        inputs.room = host_available_read()?;
    }
    // A projector the `--hf` repo picked stays only where the load gives the text model the same
    // experts and context with its reserve as without ([`vision_seat::kept`]); that load's draft
    // is off, which the projector's pick required. `--mmproj`'s file is never dropped.
    let vision = vision_seat::kept(vision, &a.vision, |v| {
        let (media, image) = vision_seat::images_of(v);
        let place = Place38 {
            image,
            media,
            ..a.place
        };
        let plans = Plans {
            inputs: &inputs,
            place,
            experts,
            levers: &plan_levers,
            mtp: None,
            slots,
            feed,
            ubatch,
            folder: folder.as_ref(),
            floor,
        };
        let ctx = ctx38(&plans, a.ctx)?.ctx;
        let at = u64::try_from(ctx)?;
        let ub = u64::try_from(ubatch.min(ctx))?;
        let machine = place.machine(&inputs, (at, ub), experts, &plan_levers, None, slots)?;
        let plan = inputs.plan_with_slots(&machine, (at, ub), &plan_levers, experts, slots)?;
        Ok(vision_seat::Cost::of(&plan, at))
    })?;
    let (media, image) = vision_seat::images_of(vision.as_ref());
    a.place = Place38 {
        image,
        media,
        ..a.place
    };
    // The paged tier's rule (`bloomery_levers::paged_columns`, the module
    // doc), asked once of the plan the flags name: a set context's slot is
    // known before any choice, so its plain plan's arena decides ahead of the
    // drafted plan; the default's is asked below, of the plan its choice and
    // the draft's yield chose. Under the rule the plan serves the asked
    // count, a set draft, and an unset one off.
    let paged_at = |arena: u64, mtp: bool| bloomery_levers::PagedAt {
        arena,
        draft: mtp,
        draft_set: levers.draft().is_some(),
    };
    let paged = |arena: u64, mtp: bool| bloomery_levers::paged_columns(&paged_at(arena, mtp));
    // The draft a reduced load runs ([`bloomery_levers::paged_draft`]): a set
    // one runs, and an unset one the plan would draft — or that only its
    // yield to the context turned off on the plan this load abandons — goes
    // off naming the paged rule, at either point. The asked count stays.
    let reduced = |arena: u64, slots: usize, mtp: bool, draft_off: Option<Draft38Off>| {
        let at = paged_at(arena, mtp);
        let asked_on = at.asked_draft(draft_off.as_ref());
        let (drafts, draft_off) = bloomery_levers::paged_draft(&at, draft_off);
        eprintln!(
            "paged rule=reduced arena={arena} asked_slots={slots} asked_draft={} \
             slots={slots} draft={} (the plan pages host experts through the NVMe tier's RAM \
             arena; the asked count is served, and an unset draft goes off)",
            if asked_on { "on" } else { "off" },
            if drafts { "on" } else { "off" },
        );
        (drafts, draft_off)
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
                ubatch,
                folder: folder.as_ref(),
                floor,
            };
            if plain.holds(ctx).is_ok() {
                plain.host(ctx, set)?.2
            } else {
                0
            }
        }
        _ => 0,
    };
    let (mtp, head, draft_off) = match paged(early, mtp) {
        Paged::AsAsked => (mtp, head, draft_off),
        Paged::Reduced => {
            let (drafts, draft_off) = reduced(early, slots, mtp, draft_off);
            if !drafts {
                mtp_inputs = None;
            }
            (drafts, if drafts { head } else { None }, draft_off)
        }
    };
    let drafted = Plans {
        inputs: &inputs,
        place: a.place,
        experts,
        levers: &plan_levers,
        mtp: mtp_inputs.as_ref(),
        slots,
        feed,
        ubatch,
        folder: folder.as_ref(),
        floor,
    };
    // The unset draft's yield to the context (the module doc): the context
    // comes first, so when the drafted load fitted its context, or holds not
    // even the floor, the plain load decides; it holds twice as much or more,
    // the draft goes off and one `draft yield` record names both. A set
    // `--ctx-size` asks too; `BLOOMERY_DRAFT` never does.
    let asks_yield = mtp && draft_off.is_none() && levers.draft().is_none();
    let (plans, mut rule, mtp, head, draft_off) = match ctx38(&drafted, a.ctx) {
        Ok(r) if !matches!(r.rule, Rule::Fitted { .. }) => (drafted, r, mtp, head, draft_off),
        r if !asks_yield => (drafted, r?, mtp, head, draft_off),
        drafted_rule => {
            let plain_plans = Plans {
                mtp: None,
                ..drafted
            };
            let plain = ctx38(&plain_plans, a.ctx)?;
            let with = drafted_rule.as_ref().map_or(0, |r| r.ctx);
            let bytes = mtp_inputs
                .as_ref()
                .expect("a drafted choice held the draft")
                .card_bytes_of(u64::try_from(plain.ctx)?, slots)?;
            match bloomery_levers::RungCost::of(
                bloomery_levers::Reserve::Draft,
                u64::try_from(with)?,
                u64::try_from(plain.ctx)?,
                bytes,
            ) {
                Some(y) => {
                    record::draft_yield(&y).eprint();
                    (plain_plans, plain, false, None, Some(Draft38Off::Yield(y)))
                }
                None => (drafted, drafted_rule?, mtp, head, draft_off),
            }
        }
    };
    let host = plans.host(rule.ctx, set)?;
    let (mtp, head, draft_off, host) = match paged(host.2, mtp) {
        Paged::AsAsked => {
            if host.2 > 0 && slots > 1 {
                eprintln!(
                    "paged rule=served arena={} asked_slots={slots} slots={slots} (the plan \
                     pages host experts through the NVMe tier's RAM arena; a set count is \
                     served as one pass over it)",
                    host.2
                );
            }
            (mtp, head, draft_off, host)
        }
        Paged::Reduced => {
            let (drafts, draft_off) = reduced(host.2, slots, mtp, draft_off);
            let one = Plans {
                mtp: if drafts { plans.mtp } else { None },
                ..plans
            };
            rule = ctx38(&one, a.ctx)?;
            let host = one.host(rule.ctx, set)?;
            (drafts, if drafts { head } else { None }, draft_off, host)
        }
    };
    refuse_mtp_levers(&levers, &draft_off)?;
    let residency_word = set.filter(|&(r, _)| r != Residency::Off).map(|(_, w)| w);
    if lookup {
        refuse_lookup_feed(feed)?;
    }
    refuse_feed(feed, (slots, mtp), residency_word)?;
    // The prompt cache's budget, read from the host need at the chosen
    // context on one reading of the host's available bytes.
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
    // The context comes first: the cache holds the sessions its budget
    // takes at it, and says so when that is not even one.
    let state = seq38_state(&inputs, rule.ctx, mtp);
    let cache = if first.ram == 0 || state <= first.ram {
        first
    } else {
        first.short_of(rule.ctx, state)
    };
    // A draft the yield or the paged rule turned off plans nothing more.
    if !mtp {
        mtp_inputs = None;
    }
    eprintln!("{}", rule.line);
    if let Some(unmet) = &rule.unmet {
        eprintln!("{unmet}");
    }
    eprint_noted(&cache.line());
    // The rounds of several slots, decided once (the open takes it through
    // `SeatArgs`): one pass of the busy rows, a slot at a time beside an
    // expert tier card ([`Place38::tiered`]).
    let pass = Pass::of(slots, !a.place.tiered());
    // The slots the seat serves ([`Session::add_slots`]): the flag's, or
    // one — one drafted pass a slot a round, nothing parked.
    parallel_line(slots, rule.ctx, from, pass);
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
    let (ctx, ctx_rule) = (rule.ctx, rule.rule);
    // A drafted load's window checked at the context the rule chose — the
    // unset draft's own condition at the default's, the set draft's at any;
    // a draft the yield turned off needs no window (the plain rule's context
    // is the larger one it fell to).
    if mtp {
        check_mtp_ctx(ctx, slots)?;
    }

    let ub = ubatch.min(ctx);
    let machine = a.place.machine(
        &inputs,
        (u64::try_from(ctx)?, u64::try_from(ub)?),
        experts,
        &plan_levers,
        mtp_inputs.as_ref(),
        slots,
    )?;
    check_reserve(&machine, slots, {
        let (gdn, qsa) = layer_counts(&inputs);
        checkpoint_budget38(gdn, qsa, u64::try_from(ctx)?)?
    })?;
    let (plan, draft_bytes) = match &mtp_inputs {
        None => (
            inputs.plan_with_slots(
                &machine,
                (u64::try_from(ctx)?, u64::try_from(ubatch.min(ctx))?),
                &plan_levers,
                experts,
                slots,
            )?,
            0,
        ),
        Some(mi) => {
            let with = inputs.plan_mtp_with_slots(
                &machine,
                (u64::try_from(ctx)?, u64::try_from(ubatch.min(ctx))?),
                &plan_levers,
                mi,
                experts,
                slots,
            )?;
            let bytes = with.draft_card_bytes() + with.arena_bytes;
            (with.plan, bytes)
        }
    };
    // PIN(2026-10-23): the planner's slot-aware arena floor replaces the one-column
    // refusal; a paged pass reads every slot's experts through the same arena.
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
    let line = record::with_room(line, plan.host.room.as_ref());
    let line = line
        .csv("devices", record::plan_devices(&machine))
        .w("cuda_order", record::cuda_order());
    record::with_requantized(line, plan.model).eprint();
    for line in record::host_only(
        plan.model,
        matches!(experts, Experts::Card).then(|| {
            model::placement::accepts::routed_rule(model::arch::qwen35moe::place::PROGRAM)
        }),
    )? {
        line.eprint();
    }
    // The encoder's seat and the reserve it takes on the stage card, with the plan.
    vision_seat::print_seat(vision.as_ref());
    let (residency, pool) = residency38_at(&plan, (a.place, feed), set)?;
    need::print(
        &plan,
        churn_bytes(&plan, CARD38, (residency, pool))?,
        cache.ram,
        true,
    )?;
    if a.plan_only {
        // The records before the load are out; nothing was loaded and the
        // route trace's directory is not made.
        bloomery_gpu_gates::start::plan_exit();
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

    let off_why = draft_off.clone();
    let open = SeatArgs {
        kv,
        place: a.place,
        lane: lane.expect("a serving run probed the lane before the plan"),
        ctx,
        ubatch,
        slots,
        pass,
        experts,
        plan_levers,
        room: inputs.room.clone(),
        plan: PlanKey::of(&plan)?,
        host: levers.host(),
        pin_main: levers.pin_main(),
        path: path.clone(),
        folder,
        mtp,
        lookup,
        head,
        own_draft,
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
        vision,
    };
    let engine = SeatEngine::spawn(
        move || Q38::open(open),
        ctx,
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
    setup38(
        (mtp, lookup, draft_in_folder),
        off_why.as_ref(),
        (slots, ctx, ctx_rule),
        kv,
    )
    .eprint();
    Record::new(&record::LISTENING38)
        .w("place", a.place.name())
        .u("ctx", ctx)
        .u("slots", slots)
        .u("slot_ctx", ctx)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}

/// The `setup` record of a server that drafts by `mtp` (its layer a checkpoint folder's when
/// `in_folder`, else a draft file's) or `lookup`, or drafts nothing for the reason `off`, with
/// `slots` resident sequences of `ctx` positions set by `rule` and its KV cache in `kv`: the
/// choices, why the draft is off where a rule turned it off, and the settings that change them.
pub fn setup38(
    (mtp, lookup, in_folder): (bool, bool, bool),
    off: Option<&Draft38Off>,
    (slots, ctx, rule): (usize, usize, Rule),
    kv: KvQ8,
) -> Record {
    let r = Record::new(&record::SETUP38);
    let (r, draft) = match (mtp, lookup) {
        (true, _) => (
            r.w("draft", "mtp")
                .w("draft_from", if in_folder { "folder" } else { "file" }),
            "BLOOMERY_DRAFT=off serves without the draft, BLOOMERY_MTP_DRAFT=FILE drafts from a \
             draft file",
        ),
        (false, true) => (
            r.w("draft", "lookup"),
            "BLOOMERY_DRAFT=mtp drafts with the MTP layer",
        ),
        (false, false) => (
            r.w("draft", "off"),
            "BLOOMERY_DRAFT=mtp drafts with the MTP layer",
        ),
    };
    let draft = match off.filter(|_| !mtp && !lookup) {
        // These reasons name their own setting, or a set draft is refused beside them.
        Some(
            why @ (Draft38Off::Paged { .. }
            | Draft38Off::Media
            | Draft38Off::Folder
            | Draft38Off::FolderMtp { .. }),
        ) => format!("the draft is off, {why}"),
        Some(why) => format!("the draft is off, {why}; {draft}"),
        None => draft.to_string(),
    };
    r.u("slots", slots)
        .u("ctx", ctx)
        .w("ctx_rule", rule.word())
        .w("cache", kv.name())
        .w(
            "change",
            format!(
                "{draft}; --parallel N sets the slots, --ctx-size N their context and \
                 --cache-type-k f16|q8_0 the KV cache"
            ),
        )
}

/// The terms of a plan that the seat's records and rules read: the stage
/// card's experts a layer and in all, the host leg's experts, the routed
/// bytes it holds resident and on the NVMe tier, that tier's RAM arena (the
/// paged rule's term), the tables the host holds and the experts the
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
    /// The ubatch size the seat read, which the load clips to `ctx` as the
    /// plans did.
    ubatch: usize,
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
    /// The KV cache's format the main thread's plans counted ([`KvQ8::choose`]).
    kv: KvQ8,
    /// The plan the main thread's rules and records read, which the load's
    /// plan is held to.
    plan: PlanKey,
    host: bloomery_levers::HostCfg,
    pin_main: bool,
    path: PathBuf,
    /// The checkpoint folder at `path`, opened by the main thread; `None` for a split, which
    /// the open reads from `path`.
    folder: Option<Folder38>,
    /// `BLOOMERY_DRAFT=mtp`, the MTP draft; unset the plain path.
    mtp: bool,
    /// `BLOOMERY_DRAFT=lookup`, the lookup draft (`drafted::Lookups`).
    lookup: bool,
    /// The draft's head under `mtp` (`head_list::head_rows_of`), which the
    /// load prints.
    head: Option<HeadPick>,
    /// The checkpoint folder's own MTP layer when it is the draft, as the main thread opened it
    /// with the folder; `None` for a draft file, which the open reads from `draft_path`.
    own_draft: Option<Split>,
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
    /// The image encoder `--mmproj` seated: the load is a media load
    /// (`GpuModel::enable_media`) and its tower is opened after the captures
    /// ([`vision_seat::open_resident`]).
    vision: Option<vision_seat::Opened>,
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
    /// busy rows ([`Q38::step_slots`]): a load of more than one resident
    /// slot with no expert tier card, plain or MTP-drafted. A drafted load's
    /// plain rounds (the steps of requests that take no pass) tell each row
    /// to its slot's draft around the pass
    /// ([`super::rounds::step_rows_drafted`]). A lookup load keeps the
    /// fallback loop: its contexts follow each step at the selected slot's
    /// position.
    step_pass: bool,
    /// Whether a plain step reads the PLE rows of the id it answers ahead
    /// ([`Q38::read_ahead`]): the body's next call is then that slot's, on
    /// a load of one slot or one whose rounds run as one pass
    /// ([`Pass::One`]); a round of turns steps another slot next.
    reads_ahead: bool,
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
    /// The resident image encoder under `--mmproj` ([`vision_seat::open_resident`]).
    #[cfg(feature = "vision")]
    vision: Option<vision_seat::Resident>,
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
        let media = a.vision.is_some();
        let model: Model38 = match a.folder {
            Some(folder) => folder.into(),
            None => Split::open(&a.path)
                .map_err(|e| format!("open {}: {e}", a.path.display()))?
                .into(),
        };
        // The room is the main thread's: a reading of the host's available
        // bytes of its own would plan the load from another room than the
        // plan the seat ruled by, and the paged tier's arena is that room's
        // difference.
        let mut inputs = PlanInputs::describe(model.header())?;
        inputs.room = a.room.clone();
        inputs.kv = inputs.kv.in_q8_if(a.kv == KvQ8::Q8);
        let cap = serve_ctx(u64::try_from(a.ctx)?, &inputs.spec)?;
        let ub = a.ubatch.min(a.ctx);
        let ctx_ub = (u64::try_from(a.ctx)?, u64::try_from(ub)?);
        // The sequences the load makes pin the body's budget at this context;
        // the plan's machine must reserve at least that.
        let (gdn, qsa) = layer_counts(&inputs);
        let ckpt = checkpoint_budget38(gdn, qsa, ctx_ub.0)?;
        // The wall before the loader's call, which the `load phases` record names.
        let opened: Duration;
        let mut m = match a.mtp {
            false => {
                let machine =
                    a.place
                        .machine(&inputs, ctx_ub, a.experts, &a.plan_levers, None, a.slots)?;
                check_reserve(&machine, a.slots, ckpt)?;
                let plan =
                    inputs.plan_with_slots(&machine, ctx_ub, &a.plan_levers, a.experts, a.slots)?;
                a.plan.hold(&PlanKey::of(&plan)?)?;
                opened = t.elapsed();
                Body38::open_placed_residency(
                    model,
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
                let draft = match a.own_draft {
                    Some(d) => d,
                    None => open_draft(&a.draft_path, a.draft_from)?,
                };
                let mtp = MtpInputs::read(&draft, model.header(), &inputs, head.rows.clone())?;
                let machine = a.place.machine(
                    &inputs,
                    ctx_ub,
                    a.experts,
                    &a.plan_levers,
                    Some(&mtp),
                    a.slots,
                )?;
                check_reserve(&machine, a.slots, ckpt)?;
                let plan = inputs.plan_mtp_with_slots(
                    &machine,
                    ctx_ub,
                    &a.plan_levers,
                    &mtp,
                    a.experts,
                    a.slots,
                )?;
                a.plan.hold(&PlanKey::of(&plan.plan)?)?;
                opened = t.elapsed();
                Body38::open_placed_mtp_residency(
                    model,
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
        // An image load gives every sequence its rotation table, once, before any call, slot or
        // capture; the plan counted them ([`Place38::machine`]).
        if media {
            m.enable_media()?;
        }
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
            inputs.spec.ctx_train,
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
        struct Captures;
        impl app::RowsLog for Captures {
            fn capture_rows(&mut self, rows: usize, nodes: usize) -> Result<(), app::SessionError> {
                eprintln!("capture verify rows={rows} nodes={nodes}");
                Ok(())
            }
        }
        let drafts = match (a.mtp, a.lookup) {
            (true, _) => SlotDrafts::open(&mut s, a.slots, a.width, &mut Captures, |m| {
                MtpDraft::open(m, Prompt38::Auto, StepMode::Graph)
            })?,
            (false, true) => {
                let drafts = SlotDrafts::lookup(&mut s, a.slots, a.width, &mut Captures)?;
                eprintln!("load draft=lookup slots={} (no weights)", a.slots);
                drafts
            }
            (false, false) => SlotDrafts::none(),
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
        // The tower after the model's captures and slots, as the qwen3 seat opens its encoder: its
        // bytes are the plan's reserve, so it comes before the expert stream below.
        #[cfg(feature = "vision")]
        let vision = vision_seat::open_resident(s.model().gpu().context(), a.vision)?;
        #[cfg(not(feature = "vision"))]
        let _ = a.vision;
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
            step_pass: pass_of_slots && !a.lookup,
            reads_ahead: a.pass == Pass::One,
            stats: a.stats,
            census,
            branch: std::cell::RefCell::new(vec![None; a.slots]),
            cut_line: vec![None; a.slots],
            lane,
            #[cfg(feature = "vision")]
            vision,
        })
    }

    /// A sequence that holds an image is kept out of the prompt cache: its rotation table
    /// (`GpuModel::rope_saved`) is no part of a saved state, so a resume would read the text
    /// table over the image's rows. Refused by name as the request's (a `GpuError::Shape`): the
    /// server keeps no state and serves on. A text load holds no table.
    fn refuse_images(&self, what: &'static str) -> Result<(), GateError> {
        if !self.s.model().rope_saved()?.is_empty() {
            return Err(GpuError::shape(
                what,
                "the sequence holds an image, whose rotation table a saved state does not carry",
            )
            .into());
        }
        Ok(())
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

    /// The PLE rows of `out`'s one id read ahead ([`Body38::prefetch_input`])
    /// while the server takes the answer, where that id is the input of the
    /// body's next call ([`Q38::reads_ahead`]), one plain step of the slot
    /// ([`Seat::pass_rows`] 1: no draft, or the slot's draft off).
    fn read_ahead(&mut self, out: &[u32]) -> Result<(), GateError> {
        if let &[next] = out
            && self.reads_ahead
            && self.pass_rows() == 1
        {
            let pos = self.s.pos();
            self.s
                .model_mut()
                .body_parts("bloomery-serve-qwen38")?
                .2
                .prefetch_input(next, pos)?;
        }
        Ok(())
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

    fn prefill_only(&mut self, ids: &[u32]) -> Result<(), GateError> {
        self.before_call();
        let sel = self.s.selected();
        if self.feed == Prompt38::Step {
            self.prefill_steps(ids)?;
        } else {
            self.drafts.prefill_only(&mut self.s, sel, ids, self.feed)?;
        }
        self.print_passes()?;
        Ok(())
    }

    /// The resident encoder's media description ([`vision_seat::Resident::media`]); `None`
    /// without `--mmproj`, a seat that takes no image.
    #[cfg(feature = "vision")]
    fn media_model(&self) -> Option<SharedMediaModel> {
        self.vision.as_ref().map(vision_seat::Resident::media)
    }

    /// Each feed's image encoded on the resident encoder and read back as bf16, then one prompt
    /// call with the images spliced in: a patch grid is one row a cell and nothing around it
    /// ([`vision_seat::merged_cells`]: the tokens carry the delimiters), a tile group is refused
    /// by name by the encode ([`vision_seat::Resident::encode`]), and the call is the session's
    /// media call ([`vision_seat::Encoded::prompt`]) — the ubatch walk, eager, the rows in place
    /// of their positions' embeddings and the sequence's rotation table shifted by each image's
    /// grid. An image load drafts nothing, so no draft is told. The encode binds the model's
    /// card back before the prompt call.
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
        let next = encoded.prompt(&mut self.s, ids, |rows| rows.image_spans())?;
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
        self.read_ahead(&[next])?;
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

    /// Every slot's draft record printed ([`SlotDrafts::close`]), then the
    /// tier nodes closed.
    fn close(&mut self) -> Result<(), GateError> {
        self.drafts.close();
        Ok(self.s.model_mut().close_nodes()?)
    }

    /// One pass from `last`: under the draft the window of at most its four
    /// rows (the width chooser's cut under `BLOOMERY_MTP_WIDTH=cost`), its
    /// kept tokens and counts; without it one step.
    fn pass(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, GateError> {
        self.before_call();
        let sel = self.s.selected();
        let d = self.drafts.pass(&mut self.s, sel, last, out)?;
        self.read_ahead(out)?;
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
        self.read_ahead(out)?;
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
    /// ([`Q38::step_pass`]) one pass of the busy rows through
    /// [`super::rounds::step_rows_one_pass`] — laid in slot order, each
    /// row's answer and lent logits row its own, slot 0 left selected; on
    /// an MTP-drafted load each row told to its slot's draft as its step
    /// would be ([`super::rounds::step_rows_drafted`]) — and the call's
    /// `residency pass` records after it. A round that leaves slot 0 out
    /// leaves its waiting hidden rows where its draft reads them
    /// (`Body38::enqueue_slots`).
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
                if q.drafts.mtp() {
                    super::rounds::step_rows_drafted(&mut q.s, rows, &mut q.drafts)
                } else {
                    super::rounds::step_rows_one_pass(&mut q.s, rows)
                }
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
    /// else the fallback (a select and a pass a row). A round that leaves
    /// slot 0 out leaves its refresh where its draft reads it
    /// (`Body38::enqueue_slots`). The open decides which loads can run one pass,
    /// so a load without that support never tries it at run time:
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
                    0..=<Body38 as MtpBody>::WIDTH,
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
        if self.drafts.mtp() && pos < held {
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
        if self.drafts.looks_up() {
            return lookup_props(p);
        }
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

    fn snapshot_size(&mut self) -> Result<Option<u64>, GateError> {
        Ok(Some(self.s.model_mut().seq_save_bytes()?))
    }

    /// The sequence state (`GpuModel::seq_save`, the draft's rows within it)
    /// as the prompt cache holds it, with the draft's side of it
    /// ([`SlotDrafts::park`]) beside the body's.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
        self.refuse_images("qwen38::snapshot")?;
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
        self.refuse_images("qwen38::snapshot_begin")?;
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
        if self.drafts.mtp() && at < state.n_tokens() {
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

#[cfg(test)]
mod tests {
    use super::parse_args;
    use super::ubatch_rule;
    use bloomery_gpu::arch::qwen3moe::ubatch::UBATCH;

    /// The seat reads the ubatch flag under both of llama-server's spellings
    /// (`LLAMA_ARG_UBATCH` reaches it as `--ubatch-size`), and the size it
    /// takes is the flag's over `BLOOMERY_QWEN3_UBATCH`; a value that is not a
    /// number, and a size outside the engine's, is refused naming the flag.
    #[test]
    fn the_ubatch_flag_is_read_and_its_size_refused_outside_the_engines() {
        let args = |f: &str, v: &str| [f.to_owned(), v.to_owned()];
        for f in ["-ub", "--ubatch-size"] {
            let a = parse_args(&args(f, "48")).unwrap_or_else(|e| panic!("{f}: {e}"));
            assert_eq!(a.ubatch, Some(48), "{f}");
        }
        let Err(e) = parse_args(&args("-ub", "4k")) else {
            panic!("4k is no size");
        };
        assert!(e.to_string().starts_with("-ub \"4k\": "), "{e}");
        let size = |flag| ubatch_rule(flag).map(|r| r.asked().map(|a| a.size));
        assert_eq!(size(Some(48)).expect("in range"), Some(48));
        assert_eq!(size(Some(UBATCH)).expect("the cap"), Some(UBATCH));
        for bad in [0, UBATCH + 1] {
            let e = size(Some(bad)).expect_err("outside 1..=UBATCH").to_string();
            assert!(
                e.contains(&format!(
                    "--ubatch-size {bad} is not a ubatch size in 1..={UBATCH}"
                )),
                "{e}"
            );
        }
    }

    /// `check` refuses a `--cache-type-k` word the cache does not take, before
    /// `--hf` fetches the model, and passes both words it takes.
    /// FAIL-first: drop the `KvQ8::choose` call from `check` and `q4` passes.
    #[test]
    fn check_refuses_a_cache_word_before_the_model() {
        let check =
            |w: &str| super::check(&["--cache-type-k".to_owned(), w.to_owned()], false, true);
        for w in ["f16", "q8_0"] {
            check(w).unwrap_or_else(|e| panic!("{w}: {e}"));
        }
        let Err(e) = check("q4") else {
            panic!("q4 is no cache word");
        };
        assert!(
            e.to_string()
                .contains("--cache-type-k takes f16 or q8_0, not \"q4\""),
            "{e}"
        );
    }
    /// A default context leaves the residency its experts only where a word
    /// runs the machine with experts on the card: a set `mid` word asks its
    /// own least, and unset, the rule's least where it runs on the plan (the
    /// stage plan (a) or (b′), no route trace and no step feed).
    /// FAIL-first: a floor that reads a set lever as unset (`set` ignored),
    /// or asks one of `off`, the gate placement, a traced or step-fed run or
    /// the host's experts turns a row red.
    #[test]
    fn a_default_context_leaves_the_residency_its_experts_only_where_it_runs() {
        use bloomery_gpu_gates::generate::Place;
        use model::arch::qwen35moe::place::Experts;

        use super::{Feed38, Place38, Prompt38, Residency, default_floor};

        let batch = Feed38 {
            prompt: Prompt38::Auto,
            traced: false,
        };
        let a = Place38::A;
        let least = bloomery_levers::residency38_least_fewest();
        assert_eq!(
            default_floor(a, batch, Experts::Card, None),
            Some(least),
            "the unset word on plan (a) with experts on the card runs the machine"
        );
        let word = "mid-p3-s1-unhosted";
        let mid = Some((Residency::parse_asking(word).expect("a mid word"), word));
        assert_eq!(
            default_floor(a, batch, Experts::Card, mid),
            Some(5),
            "a set mid word asks its own P, its spare and one that moves"
        );
        let off = Some((Residency::Off, "off"));
        for (place, feed, experts, set) in [
            (a, batch, Experts::Host, mid),
            (a, batch, Experts::Card, off),
            (a, batch, Experts::Host, None),
            (Place38::of(Place::Gate), batch, Experts::Card, None),
            (
                a,
                Feed38 {
                    traced: true,
                    ..batch
                },
                Experts::Card,
                None,
            ),
            (
                a,
                Feed38 {
                    prompt: Prompt38::Step,
                    ..batch
                },
                Experts::Card,
                None,
            ),
        ] {
            assert_eq!(
                default_floor(place, feed, experts, set),
                None,
                "{place:?}, lever set {}: no machine runs, so the context takes the room",
                set.is_some()
            );
        }
    }

    /// A checkpoint folder's plan holds its least part reads where its NVMe
    /// tier arena shrinks to its floor for them, as the load's own check
    /// grants (`HostNeed::grant`), and one byte past that room is refused.
    /// Where each card has memory of its own the refusal stands; on a unified
    /// pool it is short, so through the seat's context choice an unset
    /// context takes the largest step that holds and a set one is refused.
    #[test]
    fn a_folder_plan_holds_its_reads_on_the_arena_floor() {
        const TRAINED: u64 = 262_144;
        use bloomery_gpu_gates::generate::{Place, short_in};
        use model::arch::qwen35moe::place::PlaceError;
        use model::placement::ctx_ask::Rule;
        use model::placement::workstation::{self, HostNeed, HostRead};
        use model::placement::{
            ArenaFloor, KvBytes, ModelTensor, ModelTensors, PlanLevers, Role, plan_host_routed,
        };

        use super::{Feed38, Place38, Plans, ctx38, folder_holds};

        struct NoKv;
        impl KvBytes for NoKv {
            fn layer_bytes(&self, _: usize, _: u64) -> u64 {
                0
            }
        }
        let (rows, k, experts) = (2560, 640, 8);
        let model = ModelTensors {
            tensors: vec![ModelTensor {
                name: "blk.0.ffn_down_exps.weight".to_string(),
                shard: 0,
                layer: Some(0),
                role: Role::RoutedExperts,
                ty: gguf::GgmlType::Q8_0,
                dims: vec![k, rows, experts],
                file_bytes: k * rows * experts / 32 * 34,
                gathered_rows: None,
            }],
            layers: 1,
            experts,
            experts_used: 2,
            bf16_words: |_| false,
        };
        let machine = workstation::plan_on(workstation::A6000, 1);
        let mut plan =
            plan_host_routed(&model, &machine, 4096, &NoKv, &PlanLevers::default()).expect("plans");
        let floor = ArenaFloor::of(&plan).expect("the arena's floor").bytes;
        assert!(floor > 0);
        plan.host.nvme_arena_bytes = 24 * floor;
        let reads = 2 * floor;
        let room = HostNeed::of(&plan, 0).plan_bytes() + floor + reads;
        assert_eq!(plan.plan_at(&(room, HostRead::Given)), room);
        // The planned arena beside the reads passes the room; at its floor they fit.
        let whole = HostNeed {
            read_ahead: reads,
            ..HostNeed::of(&plan, 0)
        };
        assert!(whole.check_planned(&plan).is_err());
        folder_holds(&plan, reads).expect("the reads fit on the arena's floor");
        let e = folder_holds(&plan, reads + 1).expect_err("one byte past the floor's room");
        assert!(
            e.to_string()
                .contains(&format!("the host room is {room} B")),
            "{e}"
        );
        assert_eq!(short_in::<PlaceError>(&e), None);

        // The actual seat checks a folder's reads before choosing its context.
        use gguf::write::{Layout, TensorDecl};
        use model::arch::models::{
            Act, Arch, ChatSpec, Ffn, Gqa, LayerSpec, Mixer, ModelSpec, OutGate, Residual, Rope,
            RopeMode,
        };
        use model::arch::qwen35moe::place::{Experts, KvLayout, PlanInputs};
        use model::placement::{RowTableSource, workstation::DeviceInfo};

        let tensor = ModelTensor {
            name: "blk.0.attn_norm.weight".to_string(),
            shard: 0,
            layer: Some(0),
            role: Role::Attention,
            ty: gguf::GgmlType::F32,
            dims: vec![2560],
            file_bytes: 2560 * 4,
            gathered_rows: None,
        };
        let layout = Layout::new(
            &[],
            vec![TensorDecl {
                name: tensor.name.clone(),
                dims: tensor.dims.clone(),
                type_id: tensor.ty.as_u32(),
                nbytes: tensor.file_bytes,
            }],
        )
        .expect("the header");
        let header = || gguf::Split::header_only("seat-test".into(), &layout).expect("the header");
        let folder = super::Folder38::with(header(), header(), None);
        let spec = ModelSpec {
            arch: Arch::Qwen4Exp,
            hidden: 2560,
            vocab: 8,
            ctx_train: u32::try_from(TRAINED).expect("context"),
            rms_eps: 1e-6,
            layers: vec![LayerSpec {
                mixer: Mixer::Gqa(Gqa {
                    heads: 1,
                    kv_heads: 1,
                    head_dim: 8,
                    value_dim: 8,
                    rope: Rope {
                        mode: RopeMode::NormTail,
                        dims: 8,
                        base: 1e4,
                        yarn: None,
                        factors: false,
                    },
                    qk_norm: false,
                    o_norm: false,
                    out_gate: OutGate::None,
                    select: None,
                    window: None,
                    sinks: false,
                    value_scale: None,
                }),
                ffn: Ffn::Dense {
                    ff: 640,
                    act: Act::SwiGlu { limit: None },
                },
                residual: Residual::Plain,
                extras: Vec::new(),
            }],
            mtp: Vec::new(),
            hc: None,
            engram: None,
            encoder: None,
            logit_epilogue: None,
            chat: ChatSpec {
                pre: String::new(),
                template: None,
                tools: None,
                reasoning: None,
            },
        };
        let mut inputs = PlanInputs {
            kv: KvLayout::of_spec(&spec).expect("the stores"),
            spec,
            model: ModelTensors {
                tensors: vec![tensor],
                layers: 1,
                experts: 8,
                experts_used: 2,
                bf16_words: |_| false,
            },
            room: (u64::MAX, HostRead::Given),
            row_source: RowTableSource::Parts,
            arena: None,
        };
        let card = DeviceInfo {
            ordinal: 0,
            name: "GB10".to_string(),
            total_bytes: 32 << 30,
            free_bytes: 32 << 30,
            integrated: true,
            uuid: [1; 16],
            pci_bus: "0000:01:00.0".to_string(),
            held_by: None,
            compute_capability: Some((12, 1)),
        };
        let place = Place38::of(
            Place::parse("0")
                .expect("a card")
                .on(&[card])
                .expect("census"),
        );
        let levers = PlanLevers::default();
        let half = usize::try_from(TRAINED / 2).expect("context");
        let machine = place
            .machine(&inputs, (TRAINED / 2, 1), Experts::Host, &levers, None, 1)
            .expect("the machine");
        let plan = inputs
            .plan_target(&machine, (TRAINED / 2, 1), &levers, Experts::Host, None, 1)
            .expect("the half-context plan");
        let reads = folder
            .least_reads(&plan, super::CARD38)
            .expect("part reads");
        assert!(reads > 0);
        let room = HostNeed::of(&plan, 0).bytes() + reads;
        drop(plan);
        inputs.room = (room, HostRead::Given);
        let plans = Plans {
            inputs: &inputs,
            place,
            experts: Experts::Host,
            levers: &levers,
            mtp: None,
            slots: 1,
            feed: Feed38 {
                prompt: super::Prompt38::Auto,
                traced: false,
            },
            ubatch: 1,
            folder: Some(&folder),
            floor: None,
        };
        plans
            .holds(half)
            .expect("exactly enough room for the folder");
        let target = usize::try_from(TRAINED).expect("context");
        let e = plans
            .holds(target)
            .expect_err("the folder's reads do not fit");
        assert!(short_in::<PlaceError>(&e).is_some(), "{e}");
        Plans {
            folder: None,
            ..plans
        }
        .holds(target)
        .expect("the plan fits without part reads");
        let unset = ctx38(&plans, None).expect("fits");
        assert_eq!(unset.ctx, half);
        // PIN(2026-10-28): fitted, not halved: an unset context takes the largest step that holds.
        assert!(
            matches!(unset.rule, Rule::Fitted { .. }),
            "{:?}",
            unset.rule
        );
        assert!(ctx38(&plans, Some(target)).is_err());
    }
}
