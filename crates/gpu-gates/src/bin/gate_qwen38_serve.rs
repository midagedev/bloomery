//! `gate_qwen38_serve` — `bloomery-serve-qwen38` on the 3090 (placement
//! gate), driven over HTTP, as one process under the GPU gate lock.
//!
//!     gate_qwen38_serve --arm plain|drafted --gen <generate_qwen3moe log>
//!                       --prompt <text> --ids <a,b,…> --dir <out>
//!
//! Two arms, one a process, the recipe's two gate-lock holds: `plain`, the
//! main server with the draft off and then every clause of its own below,
//! and `drafted`, the main server with the draft on and its drafted clauses
//! alone (the bullets that name it), stopped before any other server. The
//! servers a run loads are those two mains, the slots clause's two, the
//! width chooser's one, the sampled rounds' one and the draft share clause's
//! unset-draft one (stopped at its `load draft=off` record); the `ctx`
//! clause's three starts and the residency word's one stop before the load.
//! Each arm starts the server beside this binary (`--host 127.0.0.1
//! --port 0 --place gate --ctx-size 4096`), reads its address from its
//! stderr, waits for `/health`, then checks:
//!
//! - `/props`' `engine` object, printed once, against this gate's own plan of
//!   the file the server opens (the gate card, the server's default context,
//!   the expert rule the inherited levers name): the server's name, argv and
//!   pid; the model's architecture (`qwen4exp`), shards, bytes on disk and
//!   counts as the header states them; a card `GPU<n>` and the host `CPU`,
//!   each device's `bytes` the sum of its classes and equal to the plan's
//!   card (dense + experts) and host (experts + tables) bytes; the cards' KV
//!   bytes; no draft; `ctx_verified` the deepest reference set's
//!   positions (`refset::arch::qwen4exp::VERIFIED_POSITIONS`);
//! - `/completion` of `--prompt` at temperature 0 with `return_tokens`: its
//!   ids are `generate_qwen3moe --tokens <--ids> -n 16`'s `tokens` line — all
//!   16, or a prefix ending in the end-of-generation id when the server
//!   stopped there (`generate_qwen3moe` does not stop at it). The prompt must
//!   hold at most eight ids: the passes both engines take below
//!   [`GEMM_FROM`](bloomery_gpu::arch::qwen3moe::Prompt38::GEMM_FROM) are bit
//!   for bit steps, so the server's prompt feed (the prompt less its last id)
//!   and `generate_qwen3moe`'s (the whole prompt) leave the same state, while
//!   past that each runs the prompt's last position through a different arm;
//! - `/completion` once more after one chat turn of another conversation
//!   (`/v1/chat/completions` at temperature 0): the same ids again (a
//!   request that does not extend the held sequence keeps the checkpoint at
//!   its prompt call's end, or under the draft prefills from a reset —
//!   never a wrong state);
//! - the `xstream=` line ([`xstream_agrees`]) of this server and the slots
//!   clause's second: the unset `BLOOMERY_XSTREAM` rule, stated here apart
//!   from the seat's resolver — `off` at `--place gate` (the first),
//!   `split` at `--place a` with the adaptive residency (the second, the
//!   user's case; `admit` only with the ring's room refusal named on the
//!   line). FAIL-first mutants: the seat resolving no stream prints no
//!   line, red on both; the seat handing the rule `--place gate`'s stage,
//!   or no residency, prints `off` on the second, red there;
//! - the positions the server serves: `/props`' `n_ctx` is the context it
//!   was started at ([`props_ctx`]);
//! - `cache` ([`cache`]): two sessions' requests through the prompt cache —
//!   a verbatim resend after the other session keeps every held position
//!   (under the draft the turn's prompt end, its last window having fed
//!   rows past the reply) and gives the ids of the same session run with no
//!   switch; a resend with the reply's reasoning stripped keeps the
//!   checkpoint at the first turn's end and gives a fresh run's ids; under
//!   the draft a cut keeps that checkpoint only at or past the seat's
//!   break-even for the reply (its `draft keep` line), drafting nothing and
//!   saying why by name, and below it resets with the draft on; both sides
//!   of the break-even at the turn's end, by reply length, each with a fresh
//!   run's ids and its `mtp keep` record. The plain arm's main server runs
//!   the clause with the draft off; its drafted half runs on the drafted
//!   arm's main server, after that server's drafted clauses;
//! - under the draft (`--arm drafted`, that arm's main server): a greedy
//!   `/completion` gives the plain run's ids with its draft counts carried
//!   (a draft changes which passes run, never a token), and `/props`'
//!   `engine.draft` names the draft file
//!   `refset::arch::qwen4exp::mtp::draft_file` picks, by name and path;
//!   requests that extend the sequence the
//!   server holds (`continued`): each keeps the held prefix (`cache_n`), a
//!   prompt call past it prints one `mtp prompt` record — the draft caught
//!   up at the kept position, nothing skipped — and none prints a skip, and
//!   each drafts, with the plain run's ids (the generated ones); the prompt
//!   resent keeps its end by a cut, after which the extension drafts
//!   nothing, by name, with the same prompt's ids fed fresh;
//!   and a sampled `/completion` (temperature 0.8, a fixed seed) drafts, its
//!   passes' rows drawn by its sampler (`Engine::advance_sampled`), and the
//!   same request at `top_k` 1 drafts and gives this server's greedy ids;
//! - the seats' `parallel` lines ([`parallel_agrees`]): the resident-slot
//!   rule names its slots, and the line's own terms hold them — the slots
//!   are the flag's (`--parallel 2`) or the default's (2, the `ctx` clause's
//!   flagless server), and the split they serve (`slot_ctx`, `total`) is the
//!   `ctx` line's own context times the slots — with the word that set the
//!   count (`from`: `flag`, the `--parallel`; `ctx`, a set `--ctx-size`
//!   under no `--parallel`, the one-slot rule the `ctx` clause holds too;
//!   `default`);
//! - the slots clause ([`slots_flow_together`], a server of its own under
//!   `BLOOMERY_DRAFT=mtp`, `BLOOMERY_RESIDENCY=off`, the fixed window — the
//!   clauses that compare draft counts across two runs of a server hold it,
//!   for a chooser its clock drives promises no count twice — and
//!   `BLOOMERY_STEP_STATS=1`): two greedy streamed
//!   requests of distinct prompts on `--parallel 2`, each first run alone
//!   on that same server and then both together —
//!   (v1) each request's together ids are its alone ids, and its
//!   `draft_n`/`draft_n_accepted` are its alone run's: the slots hold
//!   separate sequences, and the draft's host state is parked with its slot
//!   on every switch ([`Seat::select`]) so a slot's passes are its own —
//!   the coverage the swap clause held (a preempted request's draft
//!   rejoining) lives here now: the rejoin is the per-slot park, and this
//!   equality is red without it;
//!   (v2) `swaps_total` is absent or 0 (nothing parks), and over the
//!   together run the deltas of the busy slots booked per engine call
//!   (busy total = `n_busy_slots_per_decode` × `n_decode_total`) sit at ≥
//!   1.5: the worker books one call a round carrying every running slot
//!   (`worker.rs`'s `Stats`), so two drafted requests that decode together
//!   book 2 busy a call — at 96 tokens each and a mean ~2.5 kept ids a
//!   drafted pass, ~40 rounds carry both and a handful of head and tail
//!   rounds one, (2·N + ~4)/(N + ~4) ≈ 1.9 — while a turn-taking server
//!   books exactly 1 busy a call, 1.0;
//!   (v3) the server's own `slots round` records over the together run
//!   (the server runs under `BLOOMERY_STEP_STATS=1`, which moves no
//!   computation): at least one round carries both slots (`rows=2`), and
//!   every record of the window is `cmd=pass` of both slots in one pass
//!   (`passes=1`) — both requests advance in the same drafted pass while
//!   both are live. The records are the engine's own count, so no load on
//!   the box moves them; how many rounds carry both is (v2)'s, from the
//!   server's own call counters. A turn-taking server prints no round of
//!   both, and a seat that runs its rounds a slot at a time prints
//!   `passes=2`;
//!   (v4) the `load` line names `slots=2` and `slot_ctx` = the `--ctx-size`
//!   the server was started at over two (2048 of the 4096 total), the
//!   `listening` record the same split, and `/props`' `n_ctx` is the slot
//!   context.
//!   On the same server, the off slot and the late request
//!   ([`slots_drafted_beside`]): the break-even clause's long turn fresh,
//!   then `TURN_A`'s drafted request on the other slot and, once it
//!   decodes, the turn resent with its reply stripped, cut on the turn's
//!   slot with its draft off by name — the off slot's plain steps run in
//!   rounds of two rows beside the drafted windows, its ids the same ids
//!   fed fresh, and the drafted request's ids and draft counts its alone
//!   run's (`slots_drafted_off_slot_steps_beside_a_window`). Then the late
//!   request, in both orders: the first posted alone and the second once a
//!   few of its rounds have passed, its prompt between the first's rounds,
//!   both ids again their alone ids — `TURN_A` first
//!   (`slots_drafted_late_request_together_alone`), then `SLOTS_B` first
//!   (`slots_drafted_late_request_other_order_together_alone`). Each
//!   request takes the slot that holds its own last run, so one of the two
//!   orders lays a round's rows with slot 1's first, which the pass must
//!   run in slot order (the qwen4exp body holds slot 0's rows from row 0).
//!   A late request's first window is not empty (its draft's refresh was
//!   recorded at its prompt's last-id step); the mixed round covered is its
//!   young window beside the first's decoding one, a window with no
//!   proposal riding a pass only while its draft skips (`pass_slots`' own
//!   case).
//!   Then one more server of `--parallel 2` under the seat's own defaults
//!   — no `--place`, no `BLOOMERY_DRAFT`, no `BLOOMERY_RESIDENCY`, no
//!   `BLOOMERY_MTP_WIDTH` (the user's case: the unset rule picks the
//!   placement, the adaptive residency runs, the draft runs behind the
//!   width chooser; `BLOOMERY_STEP_STATS=1` for (v3)'s records). Its
//!   residency and width clauses, like the `xstream=` clause's second line,
//!   rest on the unset rule picking an adaptive word on the box: where it
//!   picks `off` (MemAvailable under the churn pool), each of them fails by
//!   name, the pick and its why printed, and none passes or is skipped. It
//!   runs the together pair once and holds (v2) and (v3) alone, no id
//!   equality: under
//!   the adaptive residency the other stream moves experts between the host
//!   and the card, so a stream's bits need not equal its alone run's. Its
//!   `place unset` record is the common rule's on this gate's one visible
//!   card ([`unset_place_is_the_rule`]); the residency the seat opened is
//!   the word the unset rule picked, adaptive, with a `residency host`
//!   record (`slots_default_residency_opens_the_rule_picked`); the requests
//!   crossed residency boundaries, a prompt pass and a decode pass
//!   (`slots_default_residency_passes_end_prompt_and_decode`); after a
//!   one-id flush the chooser's `mtp width` record is its own aggregate —
//!   windows the widths above 0 and E inside the window's rows
//!   (`slots_default_width_record_is_the_choosers`; a round of both slots
//!   runs the draft's whole width, so whether a run cuts a narrower window
//!   here is not fixed, and the cut is the width chooser's server's,
//!   `width_cost_cut_a_window`); and `POST /residency/reset` is a
//!   200 whose `diff` is 0, with the server's `residency reset` record
//!   carrying the same counts
//!   (`slots_default_residency_reset_is_a_200_with_its_record`).
//!   The swap clause this replaces is gone: the turn path it held
//!   (`serve::SwapEngine`, the park of a preempted request's state) left
//!   the seat with the resident slots — no load of this seat takes turns
//!   any more (every plan counts its sequences
//!   ([`PlanInputs::plan_with_slots`]), a plan that cannot hold them is
//!   refused by name), so nothing remains for it to check; a preempted
//!   request cannot happen (no slot waits for another), and the draft's
//!   park-and-rejoin under a switch is (v1)'s draft-count equality.
//!   FAIL-first mutants, each red on its line: the seat still building the
//!   turn-taking engine leaves (v2) at ~1.0 and (v3) with no round of both;
//!   the seat's drafted rounds a slot at a time (its open's `one_pass`
//!   false) print `passes=2`, (v3) red on both servers; a select
//!   that parks no draft state leaves (v1)'s draft counts red (a slot's
//!   pass refuses or drafts another's rows); a reply that hands row
//!   answers back in the wrong order scrambles (v1)'s ids;
//!   `slot_drafts` false ends the server at `check_slots`'s named refusal
//!   before it listens; a context search that ignores the slot count loads
//!   a plan of one sequence and makes the second slot's load fail (the
//!   server never listens) or (v4) red.
//!   FAIL-first mutants of the drafted clauses: the seat left on the default
//!   loop prints `passes=2` a round, red on the round clause alone (the off
//!   and late clauses hold, the default loop's ids being its own alone
//!   ids' — the seat's one-pass bits are what the round clause holds); a
//!   round that asks the off slot's draft to propose (its stale refresh
//!   refused by name, the server dead, or a proposal from rows another
//!   sequence left) turns the off clause red; a pass laid in the rows'
//!   arrival order is refused by name in the order that puts slot 1's row
//!   first and the server dies, red on that late line and every line after
//!   it. Of the seat's own defaults: a seat that opens with
//!   `Body38::open_placed` runs no residency machine whatever the rule — no
//!   `residency host` record and no `residency pass`, and the reset is the
//!   server's 501; a draft run with no chooser prints no `mtp width` record.
//! - the width chooser's clause ([`width_cost`], a server of its own under
//!   `BLOOMERY_DRAFT=mtp`, `BLOOMERY_RESIDENCY=off` and
//!   `BLOOMERY_MTP_WIDTH=cost`): one greedy request then a one-id flush,
//!   its ids the plain run's, its draft counts carried, and the chooser's
//!   `mtp width` record its own aggregate — windows the widths above 0,
//!   E inside the window's rows, a width below the draft's own among the
//!   passes. It is the one clause that runs the default chooser on a
//!   loaded server and holds its tokens to the plain run: every other
//!   drafted server this gate starts pins `BLOOMERY_MTP_WIDTH=fixed` (the
//!   clauses that compare draft counts across two runs of a server hold the
//!   fixed window; a chooser its clock drives promises no count twice), and
//!   the seat under its own defaults (the slots clause's second server)
//!   holds the chooser's records but no id equality.
//! - the sampled rounds clauses ([`slots_sampled_rounds`], a server of its
//!   own under `BLOOMERY_DRAFT=off`, `BLOOMERY_RESIDENCY=off` and
//!   `BLOOMERY_STEP_STATS=1`, `--parallel 2`): two sampled streamed
//!   requests of the slots clause's prompts (temperature
//!   [`SAMPLED_TEMPERATURE`], a fixed seed each, so their ids are a function
//!   of their logits rows alone: the server seeds each request's sampler
//!   from its own `seed`), each first run alone on that server and then
//!   both together — over the together run every `slots round` record is
//!   `cmd=step` of `passes=1` carrying both slots (`rows=2`)
//!   (`slots_sampled_rounds_run_one_pass`: a load that drafts nothing steps
//!   every request, and the seat runs a round of plain steps as one pass of
//!   its rows), and each request's together ids are its alone run's
//!   (`slots_sampled_together_alone`: each row of the pass and its logits
//!   bit for bit its step alone, each answer and logits row handed back to
//!   its own request). FAIL-first mutant: the seat's plain rounds a slot at
//!   a time (its open's `step_pass` false) print `passes=2`, red on the
//!   round clause alone (the fallback loop's ids are its alone ids too).
//!
//! Then the server is killed by the handle this binary spawned it with and
//! waited for — the drafted arm ends here — and, on the plain arm, more
//! servers start on the card, one at a time: the `ctx` clause's three, the
//! `paged` clause's eight, the draft share clause's two and the residency
//! word's one (they stop before the load, or are refused before it, but the
//! `paged` clause's small card and the draft share clause's unset-draft
//! server, stopped at their load's draft record), the `paged` clause's
//! `generate_qwen3moe` run (whole, two tokens), then the slots clause's two
//! and the sampled rounds' one (above).
//!
//! - `ctx` ([`ctx`]): a server with no `--ctx-size` prints its default
//!   (the margin rule's answer — or the largest the card holds when that is
//!   fewer, or the prompt cache's clamp of it), the largest context the card
//!   holds with its card expert bytes and the largest within the plan's
//!   margin, before its load, each the rule's against this gate's own plans
//!   (stopped there), the fit at most the file's serving cap
//!   (`place::serve_ctx`); one with `--ctx-size` set and no `--parallel`
//!   takes one slot at the whole flag (`from=ctx` on its `parallel` line,
//!   and the one-request note on stderr), and one asked for a position past
//!   the one-slot fit is
//!   refused by name before it listens (the card's rule, or the cap's when
//!   the card holds more). The rules' own terms — the slot count a flag
//!   buys, the bisection, the margin guard — are held by
//!   `bloomery_placement::placement::ctx`'s tests; these servers hold the
//!   seat's composition and its lines;
//! - `paged` ([`paged`], deferred by name in the fixture tier: its rooms
//!   are the real file's shapes): the NVMe tier's one-column rule at two
//!   rooms (`BLOOMERY_HOST_ROOM`): at 27 GiB, where the gate plan pages its
//!   host experts through the tier's RAM arena, the server with no
//!   `--ctx-size` serves one slot (`from=paged`, its `paged` line naming the
//!   arena, the two slots asked and the draft the placement left off;
//!   `paged_default_serves_one_column`), and `--parallel 2` and
//!   `BLOOMERY_DRAFT=mtp` are refused by name before the load
//!   (`paged_parallel_is_refused_by_name`, `paged_draft_is_refused_by_name`);
//!   at 40 GiB, where the tier pages experts with no arena, `--parallel 2`
//!   serves its two slots (`unpaged_parallel_serves_as_asked`). The small
//!   card's case last: a server with no `--place` (the box pin's one card,
//!   `a`) under the budget whose drafted two-slot plan holds half of `CTX`
//!   a slot, at 27 GiB — the unset draft yields to the context on that plan
//!   (`paged_small_card_yields_on_one_card`, the premise), the plan pages,
//!   and the one-slot load's `load draft=off` record names the paged rule,
//!   never the yield judged on the plan it abandoned
//!   (`paged_small_card_load_names_the_rule`). Beside them, at 27 GiB: the
//!   checkpoints every resident sequence may pin sit outside the room the
//!   split fills — the server's arena is the gate's own plan's (no host
//!   reserve for them) less `checkpoint_bytes(slots)`, and its host need that
//!   plan's plus the same (`paged_checkpoints_sit_outside_the_arena`, read
//!   off the `cache` line); a residency word set to `off` reaches the `plan`
//!   record (`paged_set_off_residency_is_accepted`), and one set to
//!   `mid-p0-s1` is refused by name before it (`paged_set_residency_is_refused_by_name`:
//!   the residency machine's fills share the arena with the step). The
//!   floor counts every slot's checkpoints, so between the one-slot plan's
//!   floor and the asked two slots' lies a band of rooms only the one-slot
//!   plan clears: the gate reads both floors off its own plans
//!   (`paged_floor_band_is_between_the_slot_floors`), and at the band's
//!   middle the flagless server serves one slot, `from=paged`, its `paged`
//!   line naming the floor (`paged_floor_fallback_serves_one_column`), while
//!   `--parallel 2` there is refused by the floor, by name
//!   (`paged_floor_set_parallel_is_refused_by_the_floor`).
//!   Then `generate_qwen3moe` with no `--place` at 27 GiB: its unset
//!   draft at `a` falls to the same rule, and the run's `load draft=off`
//!   record names it (`paged_cli_drafts_nothing_by_the_rule`). FAIL-first:
//!   the seat with no
//!   rule serves two slots at 27 GiB and refuses neither; a rule on the
//!   paged experts' bytes refuses the 40 GiB server; a one-column load that
//!   keeps the yield's reason names it on the small card; a CLI with no rule
//!   drafts on the paged plan; a machine with no
//!   checkpoint reserve gives the arena of the gate's own plan; a seat that
//!   takes the set word on a paged plan prints its `plan` record and loads;
//!   one that refuses `off` too never reaches the 27 GiB `plan` record; a
//!   seat that plans the asked two slots with no fallback refuses the
//!   flagless band server by the floor, red on the fallback clause;
//! - `draft_slot_share` ([`draft_slot_share`]): the unset draft judged on a
//!   slot's share of `--ctx-size`: at `--ctx-size 8 --parallel 2` (four
//!   positions a slot, a window needs five) an unset draft turns off, its
//!   `load draft=off` record naming the ten positions of the two slots past
//!   the eight (`draft_unset_yields_to_the_slot_share`), and a set one is
//!   refused by name before its plan (`draft_set_is_refused_by_name`); the
//!   fixture tier defers the clause by name (the draft file is the real
//!   file's);
//! - `residency_loads_the_word` ([`residency_loads_the_word`]): a server of
//!   the main arguments under `BLOOMERY_RESIDENCY=mid-p0-s1`, stopped before
//!   its load, prints the `residency lever` record of that word with why
//!   `set` and the `residency host` record of the same word — the seat's
//!   open takes a set word as given. FAIL-first mutants: a seat that ignores
//!   the set word runs the unset rule's `off` at `--place gate` and prints
//!   no host record; one that records the lever's why as unset names another
//!   why.
//!
//! Held by another test, and not run here:
//!
//! - the streamed chat's framing and its closing `[DONE]`:
//!   `crates/serve/tests/serve.rs`, `hw_chat_stream_framing_matches_non_stream`;
//! - a prompt of the served positions' length is a 400 naming the context,
//!   and `/health` after it: `crates/serve/tests/limits.rs`,
//!   `hw_requests_stop_at_the_context`;
//! - a resent completion's ids: `serve.rs`'s
//!   `hw_same_prompt_keeps_all_but_its_last_id` and this gate's `cache`
//!   clause's resends;
//! - `/tokenize` of the prompt is the reference ids:
//!   `crates/tokenizer/tests/tokenizer.rs`,
//!   `hw_qwen38_corpus_ids_equal_reference` and
//!   `hw_qwen38_cases_equal_reference` (the `tokenizer-qwen38` reference
//!   set), the handler by `serve.rs`'s `hw_tokenize_detokenize_round_trip`
//!   and the seat's binding by `completion_ids_are_generate_qwen3moe`;
//! - the residency lever's word reaching the seat: `crates/levers/src/tests.rs`
//!   hands it over, `residency_loads_the_word` above takes it at the open;
//!   the reset route's 200 and 501, its report shape and that a request
//!   resets nothing: `serve.rs`'s
//!   `hw_residency_reset_is_the_one_explicit_call`; the ids after a reset
//!   equal the first request's: `gate_qwen38_residency`'s `c7` clause; the
//!   seat's own wiring of the residency machine is the slots clause's second
//!   server's;
//! - the `parallel` line at the default placement: `parallel_agrees` over
//!   the `ctx` clause's flagless server and the slots clause's first;
//! - the drafted rounds' pass-width call: `crates/serve/tests/slots/mod.rs`,
//!   `hw_a_drafted_round_is_one_advance_slots_call_of_every_running_slot`.
//!
//! Logs and the raw stream go to `--dir`.
//!
//! The server inherits this binary's environment, so the levers it acts on
//! are the server's (`ACTS_ON`, the same list): one the server would refuse
//! is refused here, at `main`, before the server starts. The gate sets two of
//! them itself on every server it starts, so neither the placement's defaults
//! nor the environment move what a clause means: `BLOOMERY_DRAFT` (`mtp` on
//! the drafted arm's main server and the drafting clauses' own servers, `off`
//! on the plain arm's main server and the plain clauses' own) and
//! `BLOOMERY_RESIDENCY` (`off`; the residency word's server sets its word) —
//! the slots clause's second server alone sets neither: it is the seat under
//! its own defaults (the user's case).
//! `BLOOMERY_DRAFT`, `BLOOMERY_RESIDENCY` and `BLOOMERY_XSTREAM` set in this
//! binary's environment are refused by name:
//! the arm names the draft (a set `BLOOMERY_MTP_DRAFT` needs the drafted
//! arm), and every server holds the unset `xstream=` rule.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen38_serve: built without the `deepseek41` feature; see `just gate-gpu-qwen38-serve`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen38_serve", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use std::fs::File;
    use std::os::unix::process::CommandExt;
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, Stdio};
    use std::sync::mpsc;
    use std::thread::JoinHandle;
    use std::time::Duration;

    use app::mtp::MtpBody;
    use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for;
    use bloomery_gpu::arch::qwen3moe::{Body38, checkpoint_bytes, seq_positional_bytes};
    use bloomery_gpu::host::xstream::XSTREAM_ROOM;
    use bloomery_gpu_gates::record::{self, Fields, ReadError};
    use bloomery_gpu_gates::serve_client::{
        curl, ids_of, json_of, metric, parse_ids, server_log, stage_usable,
    };
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{GateError, checks_failed, ref_model_path, verdict};
    use gguf::Split;
    use model::arch::models::Mixer;
    use model::arch::qwen35moe::head_list::head_rows_of;
    use model::arch::qwen35moe::place::{
        Experts, MtpInputs, PlaceError, PlanInputs, machine_for_experts, serve_ctx,
    };
    use model::placement::workstation::{HostNeed, HostRead, RTX_3090};
    use model::placement::{PlacementError, PlanLevers};
    use refset::arch::qwen4exp::VERIFIED_POSITIONS;
    use refset::arch::qwen4exp::mtp::draft_file;
    use serde_json::{Value, json};
    use threads::helper::{Placement, spawn_helper};

    /// The levers the server acts on — the same list
    /// `bloomery_serve_qwen38` parses, kept one with it: this gate starts the
    /// server with its own environment, so a lever the server would refuse is
    /// refused here first.
    const ACTS_ON: &[&str] = &[
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
    ];

    const USAGE: &str = "usage: gate_qwen38_serve --arm plain|drafted --gen <generate_qwen3moe log> \
                         --prompt <text> --ids <a,b,…> --dir <out>";

    /// The server's arguments after its path; `/props` must echo them. The
    /// context is named: the clauses below count on [`CTX`] positions, and
    /// the default is the `ctx` clause's. The slot actions need a save
    /// directory; the gate asks only for `erase`, which writes nothing. The
    /// one slot is named: the clauses below hold the one-slot path's prompt
    /// cache, whose choreography — which state the slot-0 `erase` drops,
    /// what the cache can serve a resend — the streams of several slots
    /// would move with a request-arrival race (the free slots' LRU); the
    /// slots clause's server runs the two-slot streams on its own, and the
    /// `ctx` clause's flagless server holds the resident default's own
    /// lines.
    const SERVER_ARGS: [&str; 12] = [
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--place",
        "gate",
        "--ctx-size",
        "4096",
        "--slot-save-path",
        "/tmp",
        "--parallel",
        "1",
    ];
    /// The same arguments with no context named: the default's.
    const DEFAULT_ARGS: [&str; 6] = ["--host", "127.0.0.1", "--port", "0", "--place", "gate"];
    /// The load takes tens of seconds; the bound is the spec's 120 polls × 5 s.
    const POLLS: usize = 120;
    const POLL: Duration = Duration::from_secs(5);
    /// The greedy requests' length, `generate_qwen3moe -n 16`'s.
    const N_PREDICT: usize = 16;
    /// The context the servers are started at (`SERVER_ARGS`), and the one
    /// the default's expert cost is counted against.
    const CTX: usize = 4096;
    /// The default context's multiple.
    const CTX_STEP: usize = 256;
    /// Session A's first turn, session B's, a later turn A resends after its
    /// reply (`cache` clause (a)), and the text A resends instead of its
    /// reply (clause (b)): the last's first id is not the reply's.
    const TURN_A: &str = "Name three rivers that flow through Germany and say in one sentence \
                          which of them is the longest.";
    const TURN_B: &str = "Write a haiku about a lighthouse in winter.";
    const LATER_A: &str = "<|im_end|>\n<|im_start|>user\nAnd which of them reaches the sea \
                           first?<|im_end|>\n<|im_start|>assistant\n";
    const STRIPPED_A: &str = "The Rhine, the Danube and the Elbe; the Danube is the longest.\
                              <|im_end|>\n<|im_start|>user\nAnd which of them reaches the sea \
                              first?<|im_end|>\n<|im_start|>assistant\n";
    /// The break-even clause's two sessions, one a side, each a first turn and
    /// the text it resends in place of its reply: no other clause sends them,
    /// so no state the prompt cache holds shares more than their turn.
    const BREAK_EVEN_TURNS: [(&str, &str); 2] = [
        (
            "Name three mountains of the Alps and say in one sentence which of them is the \
             highest.",
            "Mont Blanc, the Matterhorn and the Eiger; Mont Blanc is the highest.<|im_end|>\n\
             <|im_start|>user\nAnd which of them was climbed first?<|im_end|>\n\
             <|im_start|>assistant\n",
        ),
        (
            "Name three lakes of Italy and say in one sentence which of them is the largest.",
            "Lake Garda, Lake Como and Lake Maggiore; Lake Garda is the largest.<|im_end|>\n\
             <|im_start|>user\nAnd which of them lies furthest north?<|im_end|>\n\
             <|im_start|>assistant\n",
        ),
    ];
    /// The words of the seat's line that says the MTP draft proposes nothing
    /// after a cut or a state put back.
    const DRAFT_OFF: &str = "the MTP draft proposes nothing from position";
    /// The slots clause's second prompt: `TURN_A`'s shape over another
    /// country's rivers — distinct from `TURN_A` from its first content ids,
    /// and its greedy reply runs as long.
    const SLOTS_B: &str = "Name three rivers that flow through Russia and say in one sentence \
                           which of them is the longest.";
    /// The slots clause's requests: long enough that the together run holds
    /// dozens of rounds with both streams live (and `TURN_A`'s greedy reply
    /// runs past it without an end-of-generation id — no `ignore_eos`, which
    /// steps a request plainly).
    const SLOTS_PREDICT: usize = 96;
    /// The drafted rounds clause's late request: the second posted once the
    /// first has run this many decode calls, polled at
    /// [`DRAFTED_LATE_POLL`] up to [`DRAFTED_LATE_POLLS`] times.
    const DRAFTED_LATE_ROUNDS: u64 = 4;
    const DRAFTED_LATE_POLL: Duration = Duration::from_millis(50);
    const DRAFTED_LATE_POLLS: usize = 600;
    /// The one chat turn the chat clauses send.
    const CHAT: &str = "What is the capital of France? Answer in one word.";
    /// The chat's reply length.
    const CHAT_PREDICT: usize = 32;
    /// The sampled request's temperature (llama-server's default) and seed.
    const SAMPLED_TEMPERATURE: f64 = 0.8;
    const SAMPLED_SEED: u64 = 42;
    /// The residency clause's word: set explicitly, one the lever takes; no
    /// seed expert pinned, one spare a layer.
    const RESIDENCY_WORD: &str = "mid-p0-s1";
    /// The Qwen3.8 family's unset rule as the server names it
    /// (`shared/qwen38_place.rs`'s `Q38_RULE`): the break-even experts and
    /// the card file that decides them.
    const BREAK_EVEN38: u64 = 1_152;

    /// `BLOOMERY_QWEN38_EXPERTS` as the plan's expert rule — the server's
    /// reading of the inherited environment.
    fn experts38(levers: &bloomery_levers::Levers) -> Result<Experts, GateError> {
        match levers.qwen38_experts() {
            "host" => Ok(Experts::Host),
            "card" => Ok(Experts::Card),
            other => Err(format!("BLOOMERY_QWEN38_EXPERTS={other}: host or card").into()),
        }
    }

    /// The server this binary started; killed and reaped on every way out.
    /// `serve_client::Served`'s core with this gate's server's name — that
    /// module's `exe` names `bloomery-serve-ds41`.
    struct Served38 {
        child: Child,
    }

    impl Served38 {
        /// The server's path: `bloomery-serve-qwen38` beside this binary.
        fn exe() -> Result<PathBuf, GateError> {
            Ok(std::env::current_exe()?.with_file_name("bloomery-serve-qwen38"))
        }

        /// Starts `bloomery-serve-qwen38` beside this binary with `args` on
        /// `cmd` (the server with the environment the caller set on it),
        /// stdout to `<dir>/server.out` and stderr to `<dir>/server.err`. The
        /// child is killed when this process dies, so a runner's bound that
        /// ends this process does not leave the server holding a card.
        fn spawn_with(args: &[&str], dir: &Path, cmd: &mut Command) -> Result<Served38, GateError> {
            let exe = Self::exe()?;
            cmd.args(args)
                .stdin(Stdio::null())
                .stdout(File::create(dir.join("server.out"))?)
                .stderr(File::create(dir.join("server.err"))?);
            // SAFETY: the closure runs in the child between fork and exec and
            // calls only `prctl`, which is async-signal-safe and touches no
            // memory of ours.
            unsafe {
                cmd.pre_exec(|| {
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) == 0 {
                        Ok(())
                    } else {
                        Err(std::io::Error::last_os_error())
                    }
                });
            }
            let child = cmd
                .spawn()
                .map_err(|e| format!("spawn {}: {e}", exe.display()))?;
            Ok(Served38 { child })
        }

        /// Waits for the `listening on http://<addr>` line in the server's
        /// stderr, `polls` reads `poll` apart.
        fn address(
            &mut self,
            err_log: &Path,
            polls: usize,
            poll: Duration,
        ) -> Result<String, GateError> {
            for _ in 0..polls {
                let text = std::fs::read_to_string(err_log).unwrap_or_default();
                if let Some(addr) = text
                    .lines()
                    .find_map(|l| l.split_once("listening on http://").map(|(_, a)| a.trim()))
                {
                    return Ok(addr.to_owned());
                }
                if let Some(status) = self.child.try_wait()? {
                    return Err(format!(
                        "the server exited ({status}) before listening; {}:\n{text}",
                        err_log.display()
                    )
                    .into());
                }
                std::thread::sleep(poll);
            }
            Err(format!("the server did not listen within {polls} polls").into())
        }

        fn stop(&mut self) -> Result<String, GateError> {
            self.child.kill()?;
            Ok(format!("{}", self.child.wait()?))
        }
    }

    impl Drop for Served38 {
        fn drop(&mut self) {
            if matches!(self.child.try_wait(), Ok(None)) {
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
        }
    }

    /// The `tokens [..]` line of a `generate_qwen3moe` log.
    fn gen_tokens(log: &Path) -> Result<Vec<u32>, GateError> {
        let text = std::fs::read_to_string(log).map_err(|e| format!("{}: {e}", log.display()))?;
        let line = text
            .lines()
            .find_map(|l| l.strip_prefix("tokens "))
            .ok_or_else(|| format!("{}: no `tokens` line", log.display()))?;
        parse_ids(line)
    }

    /// `ids` agree with `reference`, a run that did not stop at the
    /// end-of-generation id: all of them, or, when `stop` is `eos`, a prefix
    /// ending there.
    fn agree(ids: &[u32], stop: &str, reference: &[u32]) -> bool {
        match stop {
            "eos" => !ids.is_empty() && reference.starts_with(ids),
            _ => ids == reference,
        }
    }

    /// The arms, one a process (the module header): `plain`, the main
    /// server with the draft off and every clause of its own; `drafted`,
    /// the main server with the draft on and its drafted clauses alone.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Arm {
        Plain,
        Drafted,
    }

    impl Arm {
        fn parse(v: &str) -> Result<Arm, GateError> {
            match v {
                "plain" => Ok(Arm::Plain),
                "drafted" => Ok(Arm::Drafted),
                other => Err(format!("--arm is plain or drafted, not {other}").into()),
            }
        }
    }

    struct Args {
        arm: Arm,
        gen_log: PathBuf,
        prompt: String,
        ids: Vec<u32>,
        dir: PathBuf,
    }

    fn parse_args() -> Result<Args, GateError> {
        let (mut arm, mut gen_log, mut prompt, mut ids, mut dir) = (None, None, None, None, None);
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value: {USAGE}"))?;
            match flag.as_str() {
                "--arm" => arm = Some(Arm::parse(&v)?),
                "--gen" => gen_log = Some(PathBuf::from(v)),
                "--prompt" => prompt = Some(v),
                "--ids" => ids = Some(parse_ids(&v)?),
                "--dir" => dir = Some(PathBuf::from(v)),
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        match (arm, gen_log, prompt, ids, dir) {
            (Some(arm), Some(gen_log), Some(prompt), Some(ids), Some(dir)) => Ok(Args {
                arm,
                gen_log,
                prompt,
                ids,
                dir,
            }),
            _ => Err(USAGE.into()),
        }
    }

    fn check(ok: &mut bool, name: &str, pass: bool) {
        println!("check {name}: {}", verdict(pass));
        *ok &= pass;
    }

    /// `/props`' `engine` object (the module header) against the plan of the
    /// file the server opens, made here from its headers the way the server
    /// makes it, under the levers the server inherits; `mtp` the arm's main
    /// server (the draft's terms in its plan). `argv` and `pid` are the
    /// process this gate spawned.
    fn props_engine(
        url: &dyn Fn(&str) -> String,
        argv: &[String],
        pid: u32,
        levers: &bloomery_levers::Levers,
        err_log: &Path,
        mtp: bool,
    ) -> Result<bool, GateError> {
        let (st, body) = curl(&url("/props"), None, false)?;
        let e = json_of("/props", st, &body)?["engine"].clone();
        println!("props engine {e}");
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|err| format!("open {}: {err}", path.display()))?;
        let inputs = PlanInputs::describe(&split)?;
        let experts = experts38(levers)?;
        let ub = ubatch_for(CTX)?;
        let mut machine = machine_for_experts(
            RTX_3090,
            inputs.spec.layers.len(),
            u64::try_from(ub)?,
            experts,
        );
        machine.cards[0].free_bytes = Some(server_card_free(err_log)?);
        // Under the draft the plan carries it (its granules, its store, its
        // row map and its program's arena beside the target's card terms),
        // and `/props` files its bytes as the card's `draft` class.
        let levers_plan = PlanLevers::from_levers(levers)?;
        let (draft_path, from) = draft_file(levers.mtp_draft(), &path);
        let terms = |dense: u64, experts_at: u64, host: u64, tables: u64, kv: u64, draft: u64| {
            (dense + experts_at + draft, host + tables, kv, draft)
        };
        let (card_bytes, host_bytes, kv, draft_bytes) = match mtp {
            false => {
                let plan =
                    inputs.plan_with(&machine, u64::try_from(CTX)?, &levers_plan, experts)?;
                let c = plan.cards.first().ok_or("the gate's plan has no card")?;
                terms(
                    c.dense_bytes,
                    c.expert_bytes,
                    plan.host.expert_bytes,
                    plan.host.table_bytes,
                    plan.cards.iter().map(|c| c.kv_bytes).sum(),
                    0,
                )
            }
            true => {
                let rows = head_rows_of(levers.mtp_head_rows(), &split, inputs.spec.vocab)?.rows;
                let draft = Split::open(&draft_path).map_err(|e| {
                    format!(
                        "open the MTP draft {} ({}): {e}",
                        draft_path.display(),
                        from.describe()
                    )
                })?;
                let mtp = MtpInputs::read(&draft, &split, &inputs, rows)?;
                let with = inputs.plan_mtp_with(
                    &machine,
                    u64::try_from(CTX)?,
                    &levers_plan,
                    &mtp,
                    experts,
                )?;
                let bytes = with.draft_card_bytes() + with.arena_bytes;
                let c = with
                    .plan
                    .cards
                    .first()
                    .ok_or("the gate's plan has no card")?;
                terms(
                    c.dense_bytes,
                    c.expert_bytes,
                    with.plan.host.expert_bytes,
                    with.plan.host.table_bytes,
                    with.plan.cards.iter().map(|c| c.kv_bytes).sum(),
                    bytes,
                )
            }
        };
        let mut file_bytes = 0u64;
        for i in 0..split.shard_count() {
            let shard = split.shard_path(i).ok_or("a shard without a path")?;
            file_bytes += std::fs::metadata(shard)?.len();
        }
        println!(
            "plan card bytes {card_bytes} (the draft's {draft_bytes} of it) host bytes \
             {host_bytes} kv {kv}; file {file_bytes} B in {} shards",
            split.shard_count()
        );
        let devices = e["placement"]["devices"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let class_sum = |d: &Value| {
            d["classes"]
                .as_object()
                .map(|c| c.values().filter_map(Value::as_u64).sum::<u64>())
        };
        let layers = format!("{}-{}", 0, inputs.spec.layers.len().saturating_sub(1));
        let m = &e["model"];
        let mut ok = true;
        check(
            &mut ok,
            "props_engine_identity",
            e["name"] == "bloomery"
                && e["version"]
                    .as_str()
                    .is_some_and(|v| !v.is_empty() && !v.ends_with(" mock"))
                && e["args"] == json!(argv)
                && e["server_pid"] == json!(pid),
        );
        check(
            &mut ok,
            "props_engine_model",
            m["format"] == "gguf"
                && m["arch"] == json!(split.architecture())
                && m["files"] == json!(split.shard_count())
                && m["bytes"] == json!(file_bytes)
                && m["n_layers"] == json!(inputs.model.layers)
                && m["n_experts"] == json!(inputs.model.experts)
                && m["n_experts_used"] == json!(inputs.model.experts_used)
                && m["ctx_train"] == json!(split.arch_get_u64("context_length"))
                && m["quant"].as_str().is_some_and(|q| !q.is_empty()),
        );
        check(
            &mut ok,
            "props_engine_devices",
            devices.len() == 2
                && devices[0]["device"]
                    .as_str()
                    .and_then(|d| d.strip_prefix("GPU"))
                    .is_some_and(|n| n.parse::<u32>().is_ok())
                && devices[0]["layers"] == json!(layers)
                && devices[1]["device"] == "CPU",
        );
        check(
            &mut ok,
            "props_engine_bytes_are_the_plans",
            devices
                .iter()
                .all(|d| d["bytes"].as_u64().is_some_and(|b| Some(b) == class_sum(d)))
                && devices
                    .first()
                    .is_some_and(|d| d["bytes"] == json!(card_bytes))
                && devices
                    .get(1)
                    .is_some_and(|d| d["bytes"] == json!(host_bytes)),
        );
        check(
            &mut ok,
            "props_engine_vram_kv",
            e["placement"]["vram_kv_bytes"] == json!(kv),
        );
        check(
            &mut ok,
            "props_engine_ctx_verified",
            e["ctx_verified"] == json!(VERIFIED_POSITIONS),
        );
        match mtp {
            false => check(&mut ok, "props_engine_no_draft", e.get("draft").is_none()),
            true => {
                let d = &e["draft"];
                check(
                    &mut ok,
                    "props_engine_names_the_draft",
                    d["kind"] == json!("mtp")
                        && d["n_max"] == json!(3)
                        && draft_path
                            .file_name()
                            .is_some_and(|n| d["model"] == json!(n.to_string_lossy()))
                        && d["path"] == json!(draft_path.display().to_string())
                        && devices
                            .first()
                            .is_some_and(|d| d["classes"]["draft"].as_u64() == Some(draft_bytes)),
                );
            }
        }
        Ok(ok)
    }

    /// `/props`' `n_ctx` is the context the server was started at: the seat
    /// hands the flag to the server's settings.
    fn props_ctx(url: &dyn Fn(&str) -> String) -> Result<bool, GateError> {
        let (st, body) = curl(&url("/props"), None, false)?;
        let n_ctx = json_of("/props", st, &body)?["default_generation_settings"]["n_ctx"].clone();
        println!("positions served {CTX}: /props n_ctx {n_ctx}");
        let mut ok = true;
        check(
            &mut ok,
            "props_n_ctx_is_the_default_context",
            n_ctx == json!(CTX),
        );
        Ok(ok)
    }

    /// The server's stderr so far, read by the seat's record kinds.
    fn seat_log(err_log: &Path) -> Result<record::Log, GateError> {
        server_log(err_log, record::BLOOMERY_SERVE_QWEN38)
    }

    /// An `mtp prompt` record's join ([`record::MTP_PROMPT`]): its start, the
    /// rows the draft caught up and why it skipped (`none` when it drafts).
    #[derive(Debug)]
    struct Join {
        start: u64,
        caught_up: u64,
        skipped: String,
    }

    impl Join {
        fn read(f: &Fields) -> Result<Join, ReadError> {
            Ok(Join {
                start: f.u64("start")?,
                caught_up: f.u64("caught_up")?,
                skipped: f.text("skipped")?.to_owned(),
            })
        }
    }

    /// An `mtp keep` record ([`record::MTP_KEEP38`]): its branch (`kept` or
    /// `reset`), the prefix the keep rule granted, the break-even and the
    /// reply tokens it was derived for.
    #[derive(Debug, PartialEq)]
    struct MtpKeep {
        branch: String,
        prefix: u64,
        break_even: u64,
        reply: u64,
    }

    impl MtpKeep {
        fn read(f: &Fields) -> Result<MtpKeep, ReadError> {
            Ok(MtpKeep {
                branch: f.word("branch")?.to_owned(),
                prefix: f.u64("prefix")?,
                break_even: f.u64("break_even")?,
                reply: f.u64("reply")?,
            })
        }
    }

    /// The drafted seat's break-even as its `draft keep` line prints it: the
    /// ids a reply token and the most reply tokens it weighs.
    #[derive(Clone, Copy)]
    struct Keep38 {
        per_token: f64,
        nominal: usize,
    }

    impl Keep38 {
        fn read(err_log: &Path) -> Result<Keep38, GateError> {
            let line = record_line(err_log, "draft keep")?
                .ok_or("the drafted server printed no `draft keep` line")?;
            let field = |k: &str| {
                line.split_whitespace()
                    .find_map(|w| w.strip_prefix(k)?.strip_prefix('='))
            };
            match (
                field("per_reply_token").and_then(|v| v.parse::<f64>().ok()),
                field("reply").and_then(|v| v.parse::<usize>().ok()),
            ) {
                (Some(per_token), Some(nominal)) => Ok(Keep38 { per_token, nominal }),
                _ => Err(
                    format!("a `draft keep` line without per_reply_token and reply: {line}").into(),
                ),
            }
        }

        /// The reply a greedy request of `n` tokens is weighed for: the
        /// tokens after its first, at most the nominal.
        fn reply(self, n: usize) -> usize {
            n.saturating_sub(1).min(self.nominal)
        }

        /// The break-even of a greedy request of `n` tokens, the seat's
        /// arithmetic.
        fn at(self, n: usize) -> usize {
            let r = f64::from(u32::try_from(self.reply(n)).unwrap_or(u32::MAX));
            (r * self.per_token).ceil() as usize
        }

        /// The `mtp keep` record a request of `n` tokens that weighed a
        /// prefix of `prefix` positions prints, its branch `kept` or `reset`.
        fn record(self, n: usize, prefix: u64, kept: bool) -> MtpKeep {
            MtpKeep {
                branch: if kept { "kept" } else { "reset" }.to_owned(),
                prefix,
                break_even: self.at(n) as u64,
                reply: self.reply(n) as u64,
            }
        }
    }

    /// The seat's lines on the server's stderr so far that say the MTP draft
    /// proposes nothing after a cut or a state put back.
    fn draft_offs(err_log: &Path) -> Result<Vec<String>, GateError> {
        Ok(std::fs::read_to_string(err_log)?
            .lines()
            .filter(|l| l.contains(DRAFT_OFF))
            .map(str::to_owned)
            .collect())
    }

    /// One greedy `/completion` of the token array `ids`: its status, ids,
    /// `cache_n` and `draft_n`, and the `mtp prompt` records and draft-off
    /// lines it made the server print. A request the server does not answer is a status of 0
    /// and its error as the ids' line.
    fn greedy(
        url: &dyn Fn(&str) -> String,
        err_log: &Path,
        ids: &[u32],
        n: usize,
        cache: bool,
    ) -> Result<Greedy, GateError> {
        let log = seat_log(err_log)?;
        let before = log.all(&record::MTP_PROMPT)?.len();
        let offs_before = draft_offs(err_log)?.len();
        let keeps_before = log.all(&record::MTP_KEEP38)?.len();
        let body = json!({
            "prompt": ids, "n_predict": n, "temperature": 0, "return_tokens": true,
            "cache_prompt": cache,
        });
        let g = match curl(&url("/completion"), Some(&body), false) {
            Ok((200, text)) => {
                let v = json_of("/completion", 200, &text)?;
                Greedy {
                    ok: true,
                    tokens: ids_of(&v["tokens"]),
                    cache_n: v["timings"]["cache_n"].as_u64().unwrap_or(u64::MAX),
                    draft_n: v["timings"]["draft_n"].as_u64().unwrap_or(0),
                    joins: Vec::new(),
                    offs: Vec::new(),
                    keeps: Vec::new(),
                    said: String::new(),
                }
            }
            Ok((st, text)) => Greedy::failed(format!("HTTP {st}: {text}")),
            Err(e) => Greedy::failed(e.to_string()),
        };
        let log = seat_log(err_log)?;
        let joins = log.all(&record::MTP_PROMPT)?.split_off(before);
        let joins = joins.iter().map(Join::read).collect::<Result<_, _>>()?;
        let offs = draft_offs(err_log)?.split_off(offs_before);
        let keeps = log.all(&record::MTP_KEEP38)?.split_off(keeps_before);
        let keeps = keeps.iter().map(MtpKeep::read).collect::<Result<_, _>>()?;
        Ok(Greedy {
            joins,
            offs,
            keeps,
            ..g
        })
    }

    struct Greedy {
        ok: bool,
        tokens: Vec<u32>,
        cache_n: u64,
        draft_n: u64,
        joins: Vec<Join>,
        offs: Vec<String>,
        /// The `mtp keep` records the request made the server print.
        keeps: Vec<MtpKeep>,
        said: String,
    }

    impl Greedy {
        fn failed(said: String) -> Greedy {
            Greedy {
                ok: false,
                tokens: Vec::new(),
                cache_n: u64::MAX,
                draft_n: 0,
                joins: Vec::new(),
                offs: Vec::new(),
                keeps: Vec::new(),
                said,
            }
        }

        /// The request's line.
        fn show(&self, what: &str) {
            if self.ok {
                println!(
                    "{what}: tokens {:?} cache_n={} draft_n={} joins {:?} offs {:?} keeps {:?}",
                    self.tokens, self.cache_n, self.draft_n, self.joins, self.offs, self.keeps
                );
            } else {
                println!("{what}: {} joins {:?}", self.said, self.joins);
            }
        }

        /// Under the draft, a request that kept `cache_n` by a cut: its last
        /// draft-off line names that position, its join there names why the
        /// draft skips, and it drafted nothing.
        fn skipped_by_name(&self) -> bool {
            let at = format!(" from position {} ", self.cache_n);
            self.ok
                && self.draft_n == 0
                && self.offs.last().is_some_and(|l| l.contains(&at))
                && self
                    .joins
                    .iter()
                    .any(|j| j.start == self.cache_n && j.caught_up == 0 && j.skipped != "none")
        }

        /// Under the draft, a request that reset below the break-even: it
        /// kept nothing, printed `record` (its `mtp keep`) and no draft-off
        /// line, and drafted.
        fn reset_by_rule(&self, record: &MtpKeep) -> bool {
            self.ok
                && self.cache_n == 0
                && self.draft_n > 0
                && self.offs.is_empty()
                && self.keeps.last() == Some(record)
        }

        /// One join printed for this request, at `start`, that walked the
        /// draft up to it and skipped nothing.
        fn joined_at(&self, start: u64) -> bool {
            matches!(self.joins.as_slice(), [j] if j.start == start
                && j.caught_up >= 1
                && j.skipped == "none")
        }
    }

    /// The drafted server's continuations (`BLOOMERY_DRAFT=mtp`): requests
    /// that extend the sequence the server holds keep its prefix
    /// (`cache_prompt`), and the draft joins them — the rows an earlier
    /// request left waiting walked, the token the new request puts at their
    /// last position — so it drafts from the first window, with the plain
    /// run's ids. Three joins: after a request fed fresh that ended on its
    /// first step (`n_predict` 1), extended by the id it generated: its prompt call is
    /// empty, the join is its first step's, and no `mtp prompt` record says
    /// it skipped; after drafted windows (the held sequence is at most 11
    /// ids past the prompt: 8 generated, the last window's 3 past them),
    /// extended by the plain run's next ids, a prompt call of 1 to 4 ids and
    /// its record; and after a first step again, extended by ids the model
    /// did not generate, against the same prompt fed fresh. That first step
    /// is the prompt resent, which keeps the prompt's end (one short of it)
    /// by a cut: the draft proposes nothing from there, by name, and the
    /// extension, with the draft off, keeps the prompt only at or past the
    /// break-even of its reply (`mtp keep` `kept`, plain steps), else resets
    /// (`reset`, drafted) — with the fresh run's ids either way. Every prompt
    /// call stays below
    /// [`GEMM_FROM`](bloomery_gpu::arch::qwen3moe::Prompt38::GEMM_FROM), so
    /// a continued and a fresh feed leave the same state.
    fn continued(
        url: &dyn Fn(&str) -> String,
        err_log: &Path,
        ids: &[u32],
        reference: &[u32],
    ) -> Result<bool, GateError> {
        let p = ids.len() as u64;
        let with = |tail: &[u32]| -> Vec<u32> { ids.iter().chain(tail).copied().collect() };
        let mut ok = true;

        // PIN(2026-10-01): fed fresh, so the draft starts on: under the keep rule's checkpoints
        // the prompt kept from an earlier request's cached state is a cut, after which the draft
        // proposes nothing; was `cache_prompt` true.
        let a = greedy(url, err_log, ids, 1, false)?;
        a.show("continued: the prompt, one id");
        check(
            &mut ok,
            "continued_first_request_is_the_plain_first_id",
            a.ok && a.tokens == reference[..1],
        );

        let x1 = greedy(url, err_log, &with(&reference[..1]), 8, true)?;
        x1.show("continued: extended by its id");
        check(
            &mut ok,
            "continued_by_its_own_id_keeps_the_prompt_and_skips_nothing",
            x1.ok && x1.cache_n == p && x1.joins.is_empty(),
        );
        check(
            &mut ok,
            "continued_by_its_own_id_drafts_the_plain_ids",
            x1.ok && x1.draft_n > 0 && x1.tokens == reference[1..9],
        );

        let x2 = greedy(url, err_log, &with(&reference[..13]), 3, true)?;
        x2.show("continued: after windows, extended by the plain ids");
        check(
            &mut ok,
            "continued_after_windows_keeps_them_and_joins_at_their_end",
            x2.ok && x2.cache_n > p + 1 && x2.joined_at(x2.cache_n),
        );
        check(
            &mut ok,
            "continued_after_windows_drafts_the_plain_ids",
            x2.ok && x2.draft_n > 0 && x2.tokens == reference[13..16],
        );

        let ext: Vec<u32> = ids.iter().copied().skip(1).take(3).collect();
        if ext.len() != 3 || ext[0] == reference[0] {
            return Err(format!(
                "--ids {ids:?}: the extension {ext:?} must be three ids, its first not the \
                 generated id {}",
                reference[0]
            )
            .into());
        }
        let b = greedy(url, err_log, ids, 1, true)?;
        b.show("continued: the prompt again, one id");
        let x3 = greedy(url, err_log, &with(&ext), 8, true)?;
        x3.show("continued: extended by other ids");
        let fresh = greedy(url, err_log, &with(&ext), 8, false)?;
        fresh.show("continued: the same ids fed fresh");
        // PIN(2026-10-01): the keep rule grants checkpoints under the draft (lead, round recsave),
        // and the draft's rejoin after a cut is not built (app/src/mtp.rs): `b` keeps p − 1 by a
        // cut, so `x3` drafts nothing and its join names why; was a join at p that drafts.
        // PIN(2026-10-02): the seat's break-even weighs a keep that leaves the draft off (round
        // q38rules): `b`'s reply of one token makes no pass, so it keeps p − 1 by a cut; `x3`,
        // with the draft off, keeps the whole prompt only at or past the break-even of its 8
        // tokens, else it resets and drafts; was the prompt kept and nothing drafted, always.
        let k = Keep38::read(err_log)?;
        let x3_kept = p as usize >= k.at(8);
        let skips_at_p = x3
            .joins
            .iter()
            .any(|j| j.start == p && j.caught_up == 0 && j.skipped != "none");
        let x3_ok = if x3_kept {
            x3.ok
                && x3.cache_n == p
                && x3.draft_n == 0
                && x3.offs.is_empty()
                && skips_at_p
                && x3.keeps.last() == Some(&k.record(8, p, true))
        } else {
            // The reset may follow a state the prompt cache put back first,
            // whose prefix the rule then declined: any prefix up to p.
            (1..=p).any(|q| x3.reset_by_rule(&k.record(8, q, false)))
        };
        check(
            &mut ok,
            "continued_by_other_ids_keeps_the_prompt_and_skips_by_name",
            b.cache_n == p - 1
                && b.skipped_by_name()
                && b.keeps.last() == Some(&k.record(1, p - 1, true))
                && x3_ok,
        );
        check(
            &mut ok,
            "continued_by_other_ids_runs_the_fresh_ids",
            x3.ok
                && fresh.ok
                && fresh.cache_n == 0
                && fresh.draft_n > 0
                && x3.tokens == fresh.tokens
                && x3.tokens.len() == 8,
        );
        Ok(ok)
    }

    /// The drafted server's sampled requests, each drafted, every kept id the
    /// sampler's draw from its verified row: `sampled` (answered `st` `body`)
    /// is served and carries the draft's counts; `top1` (answered `st1`
    /// `body1`), the same request cut to `top_k` 1, samples each row's own
    /// argmax, so it drafts too and its ids are this server's greedy
    /// `first`. A plain server is no reference
    /// for the sampled ids: the drafted load plans fewer target experts on the
    /// card (the draft takes card bytes), and an expert served on the host
    /// rounds its sums another way, so a sampled id can differ at a near tie.
    /// Mutants: the refusal of a sampled request under a draft restored (a
    /// 400); a row read after the draft's walk from a buffer the walk writes
    /// (`top1` then follows the draft's argmax, not `first`).
    fn sampled_served(
        sampled: (u16, &str),
        top1: (u16, &str),
        first: &[u32],
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let (st, body) = sampled;
        let drafted = serde_json::from_str::<Value>(body).unwrap_or(Value::Null);
        let got = ids_of(&drafted["tokens"]);
        println!("sampled on the drafted server: HTTP {st} tokens {got:?}");
        check(
            &mut ok,
            "drafted_serves_a_sampled_request",
            st == 200 && !got.is_empty(),
        );
        check(
            &mut ok,
            "drafted_sampled_request_drafts",
            drafted["timings"]["draft_n"]
                .as_u64()
                .is_some_and(|n| n > 0),
        );
        let (st1, body1) = top1;
        let k1 = json_of("/completion", st1, body1)?;
        let got1 = ids_of(&k1["tokens"]);
        println!("top_k 1 on the drafted server: tokens {got1:?}, greedy {first:?}");
        check(
            &mut ok,
            "drafted_top1_sample_is_the_greedy_ids",
            got1 == first && k1["timings"]["draft_n"].as_u64().is_some_and(|n| n > 0),
        );
        Ok(ok)
    }

    /// The card free bytes the server's own `plan` record
    /// ([`record::PLAN38`]) named: its census reading at its load. The
    /// gate's re-derivations take the same one — a fresh census read beside
    /// the loaded server would see the server's own bytes as taken and size
    /// another plan. A server that printed no `plan` record or more than
    /// one, or a record without its `card_free`, is a named error, never an
    /// uncapped re-plan: the seat prints its `ctx` line before its `plan`
    /// record, so a reader that stops the server at the `ctx` line waits for
    /// the record too.
    fn server_card_free(err_log: &Path) -> Result<u64, GateError> {
        let free = seat_log(err_log)?.one(&record::PLAN38)?.u64("card_free")?;
        println!("the server's plan record: card_free={free}");
        Ok(free)
    }

    /// The first line on the server's stderr at `err_log` that starts with
    /// `head` and a space: a seat's own line no record kind reads.
    fn record_line(err_log: &Path, head: &str) -> Result<Option<String>, GateError> {
        let prefix = format!("{head} ");
        Ok(std::fs::read_to_string(err_log)?
            .lines()
            .find(|l| l.starts_with(&prefix))
            .map(str::to_owned))
    }

    /// The seat's `xstream=` line against the unset rule (the module
    /// header), stated here apart from the seat's resolver: at `--place a`
    /// (`place_a`) with the residency the server's `residency` records name
    /// running, `split` (`admit` only with the ring's room refusal named);
    /// at `--place gate` or with none, `off`. Each why is the line's own.
    fn xstream_agrees(err_log: &Path, place_a: bool) -> Result<bool, GateError> {
        let Some(line) = std::fs::read_to_string(err_log)?
            .lines()
            .find(|l| l.starts_with("xstream="))
            .map(str::to_owned)
        else {
            println!("the server printed no `xstream=` line");
            return Ok(false);
        };
        let log = seat_log(err_log)?;
        let residency = match log.first(&record::RESIDENCY_UNSET)? {
            Some(r) => r.word("residency")?.to_owned(),
            None => log
                .one(&record::RESIDENCY_LEVER)?
                .word("residency")?
                .to_owned(),
        };
        let word = line["xstream=".len()..].split(' ').next().unwrap_or("");
        let agrees = match (place_a && residency != "off", word) {
            (true, "split") => line.ends_with("(unset: --place a with a residency)"),
            (true, "admit") => {
                line.contains("(unset: split has no room: ") && line.contains(XSTREAM_ROOM)
            }
            (false, "off") => line.ends_with("(unset: no residency or not --place a)"),
            _ => false,
        };
        println!("{line} (residency {residency}, --place a {place_a}): agrees {agrees}");
        Ok(agrees)
    }

    /// The ids of `messages` as the server's chat template renders them
    /// (`/apply-template`, then `/tokenize`).
    fn rendered(url: &dyn Fn(&str) -> String, messages: Value) -> Result<Vec<u32>, GateError> {
        let (st, body) = curl(
            &url("/apply-template"),
            Some(&json!({ "messages": messages })),
            false,
        )?;
        let text = json_of("/apply-template", st, &body)?["prompt"]
            .as_str()
            .ok_or("/apply-template: no prompt")?
            .to_owned();
        tokenized(url, &text)
    }

    /// `/tokenize` of `text`.
    fn tokenized(url: &dyn Fn(&str) -> String, text: &str) -> Result<Vec<u32>, GateError> {
        let (st, body) = curl(&url("/tokenize"), Some(&json!({ "content": text })), false)?;
        Ok(ids_of(&json_of("/tokenize", st, &body)?["tokens"]))
    }

    /// The slot dropped and not saved (`POST /slots/0?action=erase`): the
    /// prompt cache keeps what it held, and nothing more.
    fn erase(url: &dyn Fn(&str) -> String) -> Result<(), GateError> {
        let (st, body) = curl(&url("/slots/0?action=erase"), Some(&json!({})), false)?;
        json_of("/slots/0?action=erase", st, &body)?;
        Ok(())
    }

    /// The `cache` clause on one server (`drafted`: its MTP draft runs):
    /// session A's first turn fed fresh, session B's turn, then A again — its
    /// state saved when B took the slot and put back — and the greedy ids
    /// against a run of A with no switch between:
    /// - (a) A's conversation resent verbatim with a later turn: it keeps
    ///   every position A held (under the draft, whose last window fed rows
    ///   past the reply it returned, the resend shares fewer than held and
    ///   keeps the turn's prompt end, as the same session does with no
    ///   switch), and its ids are those of the same two requests run back to
    ///   back (fresh, then the resend), run first. Not a fresh run
    ///   of the resend: the positions A's generation fed were steps, which a
    ///   fresh prompt call would run through the ubatch walk;
    /// - (b) A's first turn resent with the reasoning-stripped reply in place
    ///   of the reply, so the shared prefix ends at the first turn's end: it
    ///   keeps the checkpoint there (the turn's prompt call's end, one short
    ///   of the turn: its last id is fed by the first step), the draft on
    ///   or off, and its ids are a fresh run's of the same ids (every prompt
    ///   call of at least nine ids is the ubatch walk, whose bits do not
    ///   depend on where a call is cut);
    /// - under the draft, the cuts (the resends with and without the switch,
    ///   the stripped resend) weigh the turn's end against the seat's
    ///   break-even for the reply ([`Keep38`], the `draft keep` line): at or
    ///   past it each keeps the turn's end, drafts nothing and says so by
    ///   name (the seat's draft-off line at the kept position, an `mtp
    ///   prompt` record there with why, a `kept` `mtp keep` record); below it
    ///   each keeps nothing, prints a `reset` record and no draft-off line,
    ///   and drafts. The fresh run after them drafts. Without the draft no
    ///   such line prints. The draft's rejoin after a cut is not built: when
    ///   it is, this clause turns over;
    /// - (c) under the draft, both sides of the break-even at the turn's end
    ///   ([`break_even`]).
    ///
    /// Mutants: the state keeping the current position's recurrent stores
    /// alone ((b) keeps 0); a resume that leaves the PLE history behind (the
    /// resend is refused); a keep rule that grants any position (B's request
    /// cuts where no checkpoint stands, refused); under the draft every
    /// shorter prefix kept as none (the cuts' clause sees no cut).
    fn cache(
        url: &dyn Fn(&str) -> String,
        err_log: &Path,
        drafted: bool,
        label: &str,
    ) -> Result<bool, GateError> {
        let p1 = rendered(url, json!([{ "role": "user", "content": TURN_A }]))?;
        let pb = rendered(url, json!([{ "role": "user", "content": TURN_B }]))?;
        let later = tokenized(url, LATER_A)?;
        let stripped = tokenized(url, STRIPPED_A)?;
        let gemm = bloomery_gpu::arch::qwen3moe::Prompt38::GEMM_FROM;
        if p1.len() < gemm + 1 || later.len() < gemm || stripped.len() < gemm {
            return Err(format!(
                "the cache clause's turns are {}, {} and {} ids; each needs at least {gemm} past \
                 the last kept position",
                p1.len(),
                later.len(),
                stripped.len()
            )
            .into());
        }
        let n = N_PREDICT;
        let with = |a: &[u32], b: &[u32], c: &[u32]| -> Vec<u32> {
            a.iter().chain(b).chain(c).copied().collect()
        };
        let mut ok = true;

        // The run with no switch first, then the slot dropped unsaved, so no
        // cached state holds the resend when the switched run asks for it.
        let r1 = greedy(url, err_log, &p1, n, false)?;
        r1.show(&format!("cache {label}: A's turn fresh"));
        let resend = with(&p1, &r1.tokens, &later);
        let r2 = greedy(url, err_log, &resend, n, true)?;
        r2.show(&format!("cache {label}: A resent verbatim, no switch"));
        erase(url)?;
        let a1 = greedy(url, err_log, &p1, n, false)?;
        a1.show(&format!("cache {label}: A's turn fresh again"));
        let b1 = greedy(url, err_log, &pb, n, true)?;
        b1.show(&format!("cache {label}: B's turn"));
        let a2 = greedy(url, err_log, &resend, n, true)?;
        a2.show(&format!("cache {label}: A resent verbatim after B"));
        let held = (p1.len() + a1.tokens.len()).saturating_sub(1) as u64;
        let turn_end = p1.len() as u64 - 1;
        // PIN(2026-10-02): under the draft a cut keeps the turn's end only at or past the seat's
        // break-even for the reply (round q38rules: `Q38::draft_keep`), else the request resets
        // and prefills whole with the draft on; was the turn's end, the draft off, always.
        let keep38 = drafted.then(|| Keep38::read(err_log)).transpose()?;
        let cut_kept = keep38.is_none_or(|k| turn_end >= k.at(n) as u64);
        let cut_at = if cut_kept { turn_end } else { 0 };
        let want = if drafted { cut_at } else { held };
        if let Some(k) = keep38 {
            println!(
                "cache {label}: the turn's end {turn_end} against the break-even {} of a reply of \
                 {} tokens: {}",
                k.at(n),
                k.reply(n),
                if cut_kept { "kept" } else { "reset" }
            );
        }
        let kept = a2.cache_n == want && r2.cache_n == want;
        check(
            &mut ok,
            &format!("cache_{label}_a_keeps_every_held_position_after_the_switch"),
            r1.ok && a1.ok && b1.ok && a2.ok && !a1.tokens.is_empty() && kept,
        );
        check(
            &mut ok,
            &format!("cache_{label}_a_ids_are_the_run_with_no_switch"),
            r2.ok && r1.tokens == a1.tokens && !a2.tokens.is_empty() && a2.tokens == r2.tokens,
        );

        if stripped.first() == a1.tokens.first() {
            return Err(format!(
                "the stripped reply's first id {:?} is the reply's: the shared prefix would not \
                 end at the turn",
                stripped.first()
            )
            .into());
        }
        erase(url)?;
        let c1 = greedy(url, err_log, &p1, n, false)?;
        c1.show(&format!("cache {label}: A's turn fresh (b)"));
        let b2 = greedy(url, err_log, &pb, n, true)?;
        b2.show(&format!("cache {label}: B's turn (b)"));
        let strip = with(&p1, &stripped, &[]);
        let a3 = greedy(url, err_log, &strip, n, true)?;
        a3.show(&format!("cache {label}: A resent stripped after B"));
        let f3 = greedy(url, err_log, &strip, n, false)?;
        f3.show(&format!("cache {label}: the same ids fresh"));
        check(
            &mut ok,
            &format!("cache_{label}_b_keeps_the_turns_end"),
            c1.ok && b2.ok && a3.ok && a3.cache_n == if drafted { cut_at } else { turn_end },
        );
        check(
            &mut ok,
            &format!("cache_{label}_b_ids_are_a_fresh_runs"),
            f3.ok && f3.cache_n == 0 && !a3.tokens.is_empty() && a3.tokens == f3.tokens,
        );
        let cuts = [&r2, &a2, &a3];
        let named = match keep38 {
            Some(k) if cut_kept => {
                let rec = k.record(n, turn_end, true);
                cuts.iter()
                    .all(|g| g.skipped_by_name() && g.keeps.last() == Some(&rec))
                    && f3.draft_n > 0
                    && f3.offs.is_empty()
            }
            Some(k) => {
                let rec = k.record(n, turn_end, false);
                cuts.iter().all(|g| g.reset_by_rule(&rec)) && f3.draft_n > 0 && f3.offs.is_empty()
            }
            None => cuts.iter().chain([&f3].iter()).all(|g| g.offs.is_empty()),
        };
        check(
            &mut ok,
            &format!("cache_{label}_cuts_leave_the_draft_off_by_name"),
            named,
        );
        if let Some(k) = keep38 {
            ok &= break_even(url, err_log, k, &pb)?;
        }
        Ok(ok)
    }

    /// The drafted seat's break-even at a turn's end, as the `cache`
    /// clause's (b) reaches it (the turn, B's turn, the turn resent with its
    /// reply stripped: its state put back, the turn's end kept by a cut): a
    /// reply short enough that the turn's end is at or past its break-even
    /// keeps it — the draft off from there, by name, a `kept` record — and
    /// one whose break-even is past it resets — nothing kept, the draft on, a
    /// `reset` record, no draft-off line; each gives the ids of the same
    /// request fed fresh. Each side has a session of its own
    /// ([`BREAK_EVEN_TURNS`]), so the prompt cache holds no state that shares
    /// more than its turn. The replies are the seat's own arithmetic on its
    /// `draft keep` line: the longest of at most [`N_PREDICT`] tokens whose
    /// break-even the turn's end reaches, and [`N_PREDICT`], whose
    /// break-even it must not. Mutants: a break-even of 0 (the short reply
    /// keeps the turn's end) and of `usize::MAX` (the long reply resets).
    fn break_even(
        url: &dyn Fn(&str) -> String,
        err_log: &Path,
        k: Keep38,
        pb: &[u32],
    ) -> Result<bool, GateError> {
        let mut ok = true;
        for (side, (turn, reply_text)) in ["long", "short"].into_iter().zip(BREAK_EVEN_TURNS) {
            let p1 = rendered(url, json!([{ "role": "user", "content": turn }]))?;
            let stripped = tokenized(url, reply_text)?;
            let turn_end = p1.len() - 1;
            let long = (2..=N_PREDICT)
                .rev()
                .find(|&n| k.at(n) <= turn_end)
                .ok_or_else(|| {
                    format!(
                        "the turn's end {turn_end} is below the break-even of every reply of 2 to \
                     {N_PREDICT} tokens ({} ids a token): the clause needs a longer turn",
                        k.per_token
                    )
                })?;
            if k.at(N_PREDICT) <= turn_end {
                return Err(format!(
                    "the turn's end {turn_end} reaches the break-even {} of a reply of {N_PREDICT} \
                 tokens: the clause needs a shorter turn",
                    k.at(N_PREDICT)
                )
                .into());
            }
            let (n, kept) = if side == "long" {
                (long, true)
            } else {
                (N_PREDICT, false)
            };
            let strip: Vec<u32> = p1.iter().chain(&stripped).copied().collect();
            {
                erase(url)?;
                let a = greedy(url, err_log, &p1, n, false)?;
                a.show(&format!("break-even {side}: A's turn fresh, {n} tokens"));
                if stripped.first() == a.tokens.first() {
                    return Err(format!(
                    "the {side} side's stripped reply's first id {:?} is the reply's: the shared \
                     prefix would not end at the turn",
                    stripped.first()
                )
                .into());
                }
                let b = greedy(url, err_log, pb, n, true)?;
                b.show(&format!("break-even {side}: B's turn"));
                let s = greedy(url, err_log, &strip, n, true)?;
                s.show(&format!("break-even {side}: A resent stripped after B"));
                let f = greedy(url, err_log, &strip, n, false)?;
                f.show(&format!("break-even {side}: the same ids fresh"));
                let rec = k.record(n, turn_end as u64, kept);
                let branch = if kept {
                    s.cache_n == turn_end as u64
                        && s.skipped_by_name()
                        && s.keeps.last() == Some(&rec)
                } else {
                    s.reset_by_rule(&rec)
                };
                let fresh = a.ok && b.ok && f.ok && f.cache_n == 0 && f.offs.is_empty();
                let same = !s.tokens.is_empty() && s.tokens == f.tokens;
                println!(
                    "break-even {side}: the turn's end {turn_end}, a reply of {n} tokens weighed as {} \
                 against the break-even {}: want {rec:?} {}",
                    k.reply(n),
                    k.at(n),
                    verdict(branch && fresh && same)
                );
                check(
                    &mut ok,
                    if kept {
                        "break_even_past_it_keeps_the_prefix_the_draft_off"
                    } else {
                        "break_even_below_it_resets_the_draft_on"
                    },
                    branch && fresh,
                );
                check(
                    &mut ok,
                    if kept {
                        "break_even_kept_ids_are_a_fresh_runs"
                    } else {
                        "break_even_reset_ids_are_a_fresh_runs"
                    },
                    same,
                );
            }
        }
        Ok(ok)
    }

    /// The server's `parallel` line (the seat's own, printed before its
    /// load).
    fn parallel_line(err_log: &Path) -> Result<Option<String>, GateError> {
        Ok(std::fs::read_to_string(err_log)?
            .lines()
            .find(|l| l.starts_with("parallel rule="))
            .map(str::to_owned))
    }

    /// The `parallel` line's rule against its own terms (the module header):
    /// under `slots` the line's slots are the flag's (`--parallel`'s value,
    /// the caller naming it) or the default's (2), and the split the line
    /// names holds — `slot_ctx` the one context the whole round of clauses
    /// pins a slot at (the server's `--ctx-size` total over the slots) and
    /// `total` the slots' sum. `None` a line that is not the seat's;
    /// `Some(false)` a rule whose terms do not hold.
    fn parallel_agrees(line: &str, flag: Option<usize>) -> Option<bool> {
        let field = |k: &str| {
            line.split_whitespace()
                .find_map(|w| w.strip_prefix(k)?.strip_prefix('='))
        };
        let rule = field("rule")?;
        let slots: usize = field("slots")?.parse().ok()?;
        let slot_ctx: usize = field("slot_ctx")?.parse().ok()?;
        let total: usize = field("total")?.parse().ok()?;
        let want = match (rule, flag) {
            ("slots", f) => f.unwrap_or(2),
            _ => return None,
        };
        Some(slots == want && total == slots * slot_ctx && slot_ctx > 0)
    }

    /// One streamed request's answer: the final event's tokens, its draft
    /// counts, and each token event's arrival on its reader thread's clock.
    struct Streamed {
        tokens: Vec<u32>,
        /// The last event's `timings.draft_n`/`draft_n_accepted`.
        drafts: (u64, u64),
    }

    /// A streamed request's answer channel: what its reader thread sends
    /// when the stream ends.
    type StreamedRx = mpsc::Receiver<Result<Streamed, String>>;

    /// `/completion` of `ids` at temperature 0, streamed
    /// ([`streamed_with`]).
    fn streamed(
        addr: &str,
        ids: &[u32],
        n: usize,
    ) -> Result<(JoinHandle<()>, StreamedRx), GateError> {
        streamed_with(
            addr,
            json!({
                "prompt": ids, "n_predict": n, "temperature": 0, "return_tokens": true,
                "cache_prompt": false, "stream": true,
            }),
        )
    }

    /// `/completion` of the streamed request `body`: a helper thread of its
    /// own runs `curl -N` and reads the SSE lines as they land. The final
    /// event carries the tokens and the request's timings, its draft counts
    /// among them.
    fn streamed_with(addr: &str, body: Value) -> Result<(JoinHandle<()>, StreamedRx), GateError> {
        let (tx, rx) = mpsc::channel();
        let url = format!("http://{addr}/completion");
        let (h, _) = spawn_helper("slots-request", Placement::Float, move || {
            let run = (|| -> Result<Streamed, String> {
                use std::io::{BufRead, BufReader, Write};
                let mut child = Command::new("curl")
                    .args([
                        "-sS",
                        "-N",
                        "--max-time",
                        "600",
                        "-H",
                        "Content-Type: application/json",
                        "--data-binary",
                        "@-",
                    ])
                    .arg(&url)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::null())
                    .spawn()
                    .map_err(|e| format!("curl {url}: {e}"))?;
                let mut stdin = child
                    .stdin
                    .take()
                    .ok_or_else(|| format!("curl {url}: no stdin"))?;
                stdin
                    .write_all(body.to_string().as_bytes())
                    .map_err(|e| format!("curl {url}: {e}"))?;
                drop(stdin);
                let out = child
                    .stdout
                    .take()
                    .ok_or_else(|| format!("curl {url}: no stdout"))?;
                let mut tokens = Vec::new();
                let mut drafts = (0, 0);
                for line in BufReader::new(out).lines() {
                    let line = line.map_err(|e| format!("curl {url}: {e}"))?;
                    let Some(v) = line
                        .strip_prefix("data: ")
                        .and_then(|d| serde_json::from_str::<Value>(d).ok())
                    else {
                        continue;
                    };
                    if let (Some(n), Some(a)) = (
                        v["timings"]["draft_n"].as_u64(),
                        v["timings"]["draft_n_accepted"].as_u64(),
                    ) {
                        drafts = (n, a);
                    }
                    if v.get("tokens").is_some_and(Value::is_array) {
                        tokens = ids_of(&v["tokens"]);
                    }
                }
                let status = child.wait().map_err(|e| format!("curl {url}: {e}"))?;
                if !status.success() {
                    return Err(format!("curl {url}: {status}"));
                }
                Ok(Streamed { tokens, drafts })
            })();
            let _ = tx.send(run);
        })
        .map_err(|e| format!("slots: {}", e.what()))?;
        Ok((h, rx))
    }

    /// One streamed request run to its end on its own thread: its answer, or
    /// `None` a request that failed (a red check, not the clause's end).
    fn run_streamed(
        addr: &str,
        ids: &[u32],
        n: usize,
        what: &str,
    ) -> Result<Option<Streamed>, GateError> {
        let (h, rx) = streamed(addr, ids, n)?;
        h.join()
            .map_err(|_| format!("slots: the {what} request's thread panicked"))?;
        let ran = rx
            .recv()
            .map_err(|_| format!("slots: the {what} request's thread gave no answer"))?;
        match ran {
            Ok(s) => Ok(Some(s)),
            Err(e) => {
                println!("slots: the {what} request failed: {e}");
                Ok(None)
            }
        }
    }

    /// `/completion` of `ids` sampled at [`SAMPLED_TEMPERATURE`] with
    /// [`SAMPLED_SEED`], streamed ([`streamed_with`]): a request that never
    /// drafts (any sampling steps), its ids a function of its logits rows
    /// and the seed alone (the server seeds each request's sampler from its
    /// own `seed`).
    fn sampled_stream(
        addr: &str,
        ids: &[u32],
        n: usize,
    ) -> Result<(JoinHandle<()>, StreamedRx), GateError> {
        streamed_with(
            addr,
            json!({
                "prompt": ids, "n_predict": n, "temperature": SAMPLED_TEMPERATURE,
                "seed": SAMPLED_SEED, "return_tokens": true, "cache_prompt": false, "stream": true,
            }),
        )
    }

    /// One sampled streamed request ([`sampled_stream`]) run to its end on
    /// its own thread ([`run_streamed`]'s shape).
    fn run_sampled(
        addr: &str,
        ids: &[u32],
        n: usize,
        what: &str,
    ) -> Result<Option<Streamed>, GateError> {
        let (h, rx) = sampled_stream(addr, ids, n)?;
        h.join()
            .map_err(|_| format!("slots: the {what} request's thread panicked"))?;
        let ran = rx
            .recv()
            .map_err(|_| format!("slots: the {what} request's thread gave no answer"))?;
        match ran {
            Ok(s) => Ok(Some(s)),
            Err(e) => {
                println!("slots: the {what} request failed: {e}");
                Ok(None)
            }
        }
    }

    /// The busy slots the server has booked over its `decodes` engine calls:
    /// `/metrics` carries them as the running mean `n_busy_slots_per_decode`
    /// (`serve::api`'s metrics), so the total is that mean times the calls.
    fn busy_total(url: &dyn Fn(&str) -> String, decodes: f64) -> Result<f64, GateError> {
        Ok(metric(url, "n_busy_slots_per_decode")?.unwrap_or(f64::NAN) * decodes)
    }

    /// The (v3) round check (the module header) on the server whose stderr
    /// is `err_log`: its `slots round` records past the first `before` (the
    /// count read before the together run) — at least one round carried
    /// both slots (`rows=2`), and every such round ran them as one pass
    /// (`passes=1`) of the seat's two slots. With `cmd` named, every record
    /// of the window is also a round of that command (the drafted rounds'
    /// `pass`) of two rows.
    // PIN(2026-10-05): (v3) reads the engine's own round records, not the SSE
    // arrival timestamps — a client-side clock on a shared box clumped one
    // stream's events under load average 15.6 and turned the window rule red
    // with the server's rounds unchanged. The threshold stays zero: no thin
    // window then, no round of both run in more than one pass now.
    fn rounds_carry_both_in_one_pass(
        ok: &mut bool,
        name: &str,
        err_log: &Path,
        before: usize,
        cmd: Option<&str>,
    ) -> Result<(), GateError> {
        let rounds = seat_log(err_log)?.all(&record::SLOTS_ROUND)?;
        let window = &rounds[before.min(rounds.len())..];
        let both = window.iter().filter(|r| r.u64("rows") == Ok(2)).count();
        let split = window
            .iter()
            .filter(|r| {
                r.u64("rows") == Ok(2) && (r.u64("passes") != Ok(1) || r.u64("slots") != Ok(2))
            })
            .count();
        let other = cmd.map_or(0, |c| {
            window
                .iter()
                .filter(|r| r.word("cmd") != Ok(c) || r.u64("rows") != Ok(2))
                .count()
        });
        println!(
            "{name}: {} record(s) in the together window, {both} of both slots, {split} of them \
             in more than one pass, {other} of another command or width than {cmd:?} of two rows",
            window.len(),
        );
        check(ok, name, both > 0 && split == 0 && other == 0);
        Ok(())
    }

    /// The slots clause (the module header) in two parts. The first server,
    /// into `<dir>/slots`, is this gate's own shape at `--parallel 2` with
    /// the draft on, the residency off, the fixed window and
    /// `BLOOMERY_STEP_STATS=1`: the ids and draft counts of two streamed
    /// requests run together are their alone runs' ((v1) — the per-slot
    /// draft park is the rejoin the swap clause held), a round carries both
    /// slots and nothing swaps ((v2)), every round of the together run is
    /// one `pass` of both slots ((v3), [`rounds_carry_both_in_one_pass`]),
    /// and the load, `listening` and `/props` name the split ((v4)). Then,
    /// on the same server, the off slot beside a drafted window and the late
    /// request in both orders ([`slots_drafted_beside`]). The second server,
    /// into `<dir>/slots-default`, is the seat under its own defaults at
    /// `--parallel 2` — `--place a`, neither lever set (the user's case: the
    /// A6000 stages, the adaptive residency and the draft run) — and runs
    /// the together pair once ([`slots_default_residency`]).
    // PIN(2026-10-05): the swap clause (`swap_rejoins_the_draft`) is removed — the
    /// turn path it checked (`serve::SwapEngine`, the park of a preempted request's
    /// state) left the seat with the resident slots, and its coverage (a preempted
    /// request's draft rejoining) is this clause's (v1) draft-count equality.
    fn slots_flow_together(dir: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let own = dir.join("slots");
        std::fs::create_dir_all(&own)?;
        let err_log = own.join("server.err");
        // SERVER_ARGS ends in the one-slot `--parallel 1`; this clause's
        // server takes the two-slot value, the total context the same 4096.
        let mut args: Vec<&str> = SERVER_ARGS.to_vec();
        let parallel = args.len() - 1;
        args[parallel] = "2";
        let mut cmd = Command::new(Served38::exe()?);
        cmd.env(bloomery_levers::DRAFT, "mtp")
            .env(bloomery_levers::RESIDENCY, "off")
            // The fixed window: the clause holds draft counts equal across
            // two runs, which a clock-driven chooser does not promise.
            .env(bloomery_levers::MTP_WIDTH, "fixed")
            .env(bloomery_levers::STEP_STATS, "1");
        let mut served = Served38::spawn_with(&args, &own, &mut cmd)?;
        println!("slots server pid {}", served.child.id());
        let addr = match served.address(&err_log, POLLS, POLL) {
            Ok(a) => a,
            // A server that never listens names no split: the checks are red
            // by name, and the clause's second server still runs.
            Err(e) => {
                println!(
                    "slots: the server never listened: {e}; it said: {}",
                    std::fs::read_to_string(&err_log).unwrap_or_default()
                );
                for name in [
                    "slots_parallel_line_names_the_rule",
                    "slots_load_names_the_split",
                    "slots_alone_ids_stay",
                    "slots_alone_draft_counts_stay",
                    "slots_rounds_carry_both_slots",
                    "slots_rounds_of_both_run_one_pass",
                    "slots_drafted_off_slot_steps_beside_a_window",
                    "slots_drafted_late_request_together_alone",
                    "slots_drafted_late_request_other_order_together_alone",
                ] {
                    check(&mut ok, name, false);
                }
                let _ = served.stop();
                return slots_default_residency(dir, &mut ok);
            }
        };
        let url = |p: &str| format!("http://{addr}{p}");
        let line = parallel_line(&err_log)?.ok_or("the slots server printed no `parallel` line")?;
        println!("slots {line}");
        check(
            &mut ok,
            "slots_parallel_line_names_the_rule",
            line.contains("rule=slots") && parallel_agrees(&line, Some(2)) == Some(true),
        );
        // (v4) The load names the split, the `listening` record the same,
        // and /props the slot ctx.
        // The `load arch=` line, not the host tier's own `load host_tier …`
        // note that can precede it.
        let load = std::fs::read_to_string(&err_log)?
            .lines()
            .find(|l| l.starts_with("load arch="))
            .unwrap_or("")
            .to_owned();
        let field = |l: &str, k: &str| {
            l.split_whitespace()
                .find_map(|w| w.strip_prefix(&format!("{k}=")))
                .and_then(|v| v.parse::<u64>().ok())
        };
        let listening = seat_log(&err_log)?.one(&record::LISTENING38)?;
        let (l_slots, l_slot_ctx) = (listening.u64("slots")?, listening.u64("slot_ctx")?);
        let (st, body) = curl(&url("/props"), None, false)?;
        let props = json_of("/props", st, &body)?;
        let n_ctx = props["n_ctx"].as_u64().unwrap_or(u64::MAX);
        println!(
            "slots: {load}; props n_ctx {n_ctx}, load slots={:?} slot_ctx={:?}, listening \
             slots={l_slots} slot_ctx={l_slot_ctx}",
            field(&load, "slots"),
            field(&load, "slot_ctx"),
        );
        check(
            &mut ok,
            "slots_load_names_the_split",
            field(&load, "slots") == Some(2)
                && field(&load, "slot_ctx") == Some(u64::try_from(CTX / 2).unwrap())
                && l_slots == 2
                && l_slot_ctx == u64::try_from(CTX / 2).unwrap()
                && n_ctx == u64::try_from(CTX / 2).unwrap(),
        );
        // Two distinct prompts, each first run alone, then both together.
        let (a_ids, b_ids) = (
            rendered(&url, json!([{ "role": "user", "content": TURN_A }]))?,
            rendered(&url, json!([{ "role": "user", "content": SLOTS_B }]))?,
        );
        let gemm = bloomery_gpu::arch::qwen3moe::Prompt38::GEMM_FROM;
        if a_ids.len() < gemm || b_ids.len() < gemm {
            return Err(format!(
                "the slots clause's prompts are {} and {} ids; each needs at least {gemm}",
                a_ids.len(),
                b_ids.len()
            )
            .into());
        }
        let mut alone = Vec::new();
        for (ids, what) in [(&a_ids, "alone A"), (&b_ids, "alone B")] {
            alone.push(run_streamed(&addr, ids, SLOTS_PREDICT, what)?);
        }
        let decode0 = metric(&url, "n_decode_total")?.unwrap_or(f64::NAN);
        let busy0 = busy_total(&url, decode0)?;
        // The alone runs print no record (a round of one active request is a
        // select and an `advance`), so the window opens empty; taken anyway,
        // so the together run's records are exactly the ones after this
        // point.
        let before = seat_log(&err_log)?.all(&record::SLOTS_ROUND)?.len();
        // Both posted at once: the prompts serialize on the engine thread (a
        // round or two of skew), so nearly every round carries both slots —
        // the (v2) bound's shape.
        let first = streamed(&addr, &a_ids, SLOTS_PREDICT)?;
        let second = streamed(&addr, &b_ids, SLOTS_PREDICT)?;
        let mut together = Vec::new();
        for ((h, rx), what) in [(first, "together A"), (second, "together B")] {
            h.join()
                .map_err(|_| format!("slots: the {what} request's thread panicked"))?;
            let ran = rx
                .recv()
                .map_err(|_| format!("slots: the {what} request's thread gave no answer"))?;
            match ran {
                Ok(s) => together.push(Some(s)),
                Err(e) => {
                    println!("slots: the {what} request failed: {e}");
                    together.push(None);
                }
            }
        }
        let decode1 = metric(&url, "n_decode_total")?.unwrap_or(f64::NAN);
        let busy1 = busy_total(&url, decode1)?;
        let swaps = metric(&url, "swaps_total")?;
        println!(
            "slots alone: {} and {} ids; together {} and {} ids, drafts {:?} -> {:?} and {:?} \
             -> {:?}; decode {decode0} -> {decode1}, busy {busy0} -> {busy1}, swaps {swaps:?}",
            alone[0].as_ref().map_or(0, |s| s.tokens.len()),
            alone[1].as_ref().map_or(0, |s| s.tokens.len()),
            together[0].as_ref().map_or(0, |s| s.tokens.len()),
            together[1].as_ref().map_or(0, |s| s.tokens.len()),
            alone[0].as_ref().map(|s| s.drafts),
            together[0].as_ref().map(|s| s.drafts),
            alone[1].as_ref().map(|s| s.drafts),
            together[1].as_ref().map(|s| s.drafts),
        );
        // (v1) The slots hold separate sequences and each slot's draft its
        // own state: together ids and draft counts are alone ones (a failed
        // request is red, not the clause's end).
        check(
            &mut ok,
            "slots_alone_ids_stay",
            together[0].as_ref().is_some_and(|t| !t.tokens.is_empty())
                && together[1].as_ref().is_some_and(|t| !t.tokens.is_empty())
                && alone[0]
                    .as_ref()
                    .is_some_and(|a| a.tokens == together[0].as_ref().unwrap().tokens)
                && alone[1]
                    .as_ref()
                    .is_some_and(|a| a.tokens == together[1].as_ref().unwrap().tokens),
        );
        check(
            &mut ok,
            "slots_alone_draft_counts_stay",
            together[0].as_ref().is_some_and(|t| t.drafts.0 > 0)
                && together[1].as_ref().is_some_and(|t| t.drafts.0 > 0)
                && alone[0]
                    .as_ref()
                    .is_some_and(|a| a.drafts == together[0].as_ref().unwrap().drafts)
                && alone[1]
                    .as_ref()
                    .is_some_and(|a| a.drafts == together[1].as_ref().unwrap().drafts),
        );
        // (v2) Nothing parks, and a round carries both slots (the module
        // header's derivation): >= 1.5 against the turns' 1.0.
        let ratio = (busy1 - busy0) / (decode1 - decode0);
        check(
            &mut ok,
            "slots_rounds_carry_both_slots",
            swaps.is_none_or(|v| v == 0.0) && ratio >= 1.5,
        );
        // (v3) While both stream, every round of both runs one pass.
        rounds_carry_both_in_one_pass(
            &mut ok,
            "slots_rounds_of_both_run_one_pass",
            &err_log,
            before,
            Some("pass"),
        )?;
        ok &= slots_drafted_beside(&url, &addr, &err_log, [&a_ids, &b_ids], &alone)?;
        println!("slots server stopped: {}", served.stop()?);
        slots_default_residency(dir, &mut ok)
    }

    /// The slots clause's second server (the module header): the seat under
    /// its own defaults at `--parallel 2` — no `--place`, no lever set (the
    /// user's case) — runs the together pair once and holds (v2) and (v3)
    /// alone; under the adaptive residency a stream's bits need not equal
    /// its alone run's, so no id equality is asked. The recipe leaves the
    /// box env file's CUDA_VISIBLE_DEVICES pin (the 3090) in place, so the
    /// seat sees one card: the unset rule's `place unset` record names it
    /// (`unset_place_is_the_rule`), and the server's load waits out a timing
    /// sitting by the box guard this gate runs under. The seat's own wiring
    /// of the two machines it runs by default is read here too: the
    /// residency it opens is the one the unset rule picked and its
    /// boundaries print `residency pass` records after the requests; its
    /// reset route answers 200 and prints its `residency reset` record;
    /// and, after a one-id flush, the draft's width chooser (`cost`, the
    /// unset lever) prints its `mtp width` record as the chooser's own
    /// aggregate.
    fn slots_default_residency(dir: &Path, ok: &mut bool) -> Result<bool, GateError> {
        let own = dir.join("slots-default");
        std::fs::create_dir_all(&own)?;
        let err_log = own.join("server.err");
        let mut cmd = Command::new(Served38::exe()?);
        cmd.env_remove(bloomery_levers::DRAFT)
            .env_remove(bloomery_levers::RESIDENCY)
            .env_remove(bloomery_levers::MTP_HEAD_ROWS)
            .env_remove(bloomery_levers::MTP_DRAFT)
            .env_remove(bloomery_levers::MTP_WIDTH)
            .env(bloomery_levers::STEP_STATS, "1");
        let args = ["--host", "127.0.0.1", "--port", "0", "--parallel", "2"];
        let mut served = Served38::spawn_with(&args, &own, &mut cmd)?;
        println!("slots-default server pid {}", served.child.id());
        let addr = match served.address(&err_log, POLLS, POLL) {
            Ok(a) => a,
            Err(e) => {
                println!(
                    "slots-default: the server never listened: {e}; it said: {}",
                    std::fs::read_to_string(&err_log).unwrap_or_default()
                );
                for name in [
                    "unset_place_is_the_rule",
                    "slots_default_xstream_line_is_the_unset_rules",
                    "slots_default_residency_opens_the_rule_picked",
                    "slots_default_rounds_carry_both_slots",
                    "slots_default_rounds_of_both_run_one_pass",
                    "slots_default_residency_passes_end_prompt_and_decode",
                    "slots_default_width_record_is_the_choosers",
                    "slots_default_residency_reset_is_a_200_with_its_record",
                ] {
                    check(ok, name, false);
                }
                let _ = served.stop();
                return Ok(*ok);
            }
        };
        let url = |p: &str| format!("http://{addr}{p}");
        // The common rule's own record on the unset flag: the pin leaves one
        // card visible, so the offer is `a` on it and the plan is never
        // asked, the rule's break-even and basis named beside it. FAIL-first:
        // a seat that keeps a hard-coded default placement names `why=set`;
        // one that runs `bp` on a one-card view never listens; a rule that
        // asks the plan on one card names a tier count.
        let placed = seat_log(&err_log)?.one(&record::PLACE_UNSET)?;
        let (place, why, tier, break_even, basis) = (
            placed.word("place")?,
            placed.text("why")?,
            placed.opt_u64("tier_experts")?,
            placed.opt_u64("break_even")?,
            placed.opt_word("basis")?.unwrap_or(""),
        );
        println!(
            "slots-default: place unset place={place} why={why} tier_experts={tier:?} \
             break_even={break_even:?} basis={basis}"
        );
        check(
            ok,
            "unset_place_is_the_rule",
            place == "a"
                && why == "one card"
                && tier.is_none()
                && break_even == Some(BREAK_EVEN38)
                && basis == "docs/cards/q38bpbug-ab.card",
        );
        check(
            ok,
            "slots_default_xstream_line_is_the_unset_rules",
            xstream_agrees(&err_log, true)?,
        );
        // The residency the seat opened is the one the unset rule picked, and
        // it is the adaptive one on this placement. FAIL-first: a seat that
        // opens with `Body38::open_placed`, which runs no machine whatever
        // the rule, prints no `residency host` record; one that ignores the
        // pick opens another word.
        let log = seat_log(&err_log)?;
        let picked = log.first(&record::RESIDENCY_UNSET)?;
        let host = log.first(&record::RESIDENCY_HOST)?;
        let word_of = |f: &Option<Fields>| -> Result<Option<String>, ReadError> {
            f.as_ref()
                .map(|f| f.word("residency").map(str::to_owned))
                .transpose()
        };
        let (rule_word, host_word) = (word_of(&picked)?, word_of(&host)?);
        let why = picked.as_ref().map(|p| p.text("why")).transpose()?;
        println!("slots-default: residency unset {rule_word:?} ({why:?}), host {host_word:?}");
        // The premise of every clause below that holds the seat's residency
        // and width wiring: the unset rule picks an adaptive word on this
        // box (a load with MemAvailable under the churn pool picks `off`, and
        // `xstream_agrees` rests on the same pick). A box where it does not
        // fails each of them by name here, the pick and its why printed once;
        // none passes vacuously and none is skipped.
        let adaptive = rule_word.as_deref().is_some_and(|w| w != "off");
        if !adaptive {
            println!(
                "slots-default: the unset rule picked {rule_word:?} ({why:?}), not an adaptive \
                 word: the residency and width clauses below hold nothing on this server and fail"
            );
        }
        check(
            ok,
            "slots_default_residency_opens_the_rule_picked",
            adaptive && host_word == rule_word,
        );
        let (a_ids, b_ids) = (
            rendered(&url, json!([{ "role": "user", "content": TURN_A }]))?,
            rendered(&url, json!([{ "role": "user", "content": SLOTS_B }]))?,
        );
        let decode0 = metric(&url, "n_decode_total")?.unwrap_or(f64::NAN);
        let busy0 = busy_total(&url, decode0)?;
        let before = seat_log(&err_log)?.all(&record::SLOTS_ROUND)?.len();
        let first = streamed(&addr, &a_ids, SLOTS_PREDICT)?;
        let second = streamed(&addr, &b_ids, SLOTS_PREDICT)?;
        let mut together = Vec::new();
        for ((h, rx), what) in [(first, "default A"), (second, "default B")] {
            h.join()
                .map_err(|_| format!("slots-default: the {what} request's thread panicked"))?;
            let ran = rx.recv().map_err(|_| {
                format!("slots-default: the {what} request's thread gave no answer")
            })?;
            match ran {
                Ok(s) => together.push(Some(s)),
                Err(e) => {
                    println!("slots-default: the {what} request failed: {e}");
                    together.push(None);
                }
            }
        }
        let decode1 = metric(&url, "n_decode_total")?.unwrap_or(f64::NAN);
        let busy1 = busy_total(&url, decode1)?;
        let swaps = metric(&url, "swaps_total")?;
        println!(
            "slots-default together: {} and {} ids; decode {decode0} -> {decode1}, busy {busy0} \
             -> {busy1}, swaps {swaps:?}",
            together[0].as_ref().map_or(0, |s| s.tokens.len()),
            together[1].as_ref().map_or(0, |s| s.tokens.len()),
        );
        let ratio = (busy1 - busy0) / (decode1 - decode0);
        check(
            ok,
            "slots_default_rounds_carry_both_slots",
            swaps.is_none_or(|v| v == 0.0) && ratio >= 1.5,
        );
        rounds_carry_both_in_one_pass(
            ok,
            "slots_default_rounds_of_both_run_one_pass",
            &err_log,
            before,
            None,
        )?;
        // The residency boundaries the requests crossed: a prompt pass, and
        // the decode passes of the running slots (a step, a pair, resident
        // slots' rows, a drafted pass of them). FAIL-first: a seat that
        // opens with `Body38::open_placed` runs no machine and prints none.
        let kinds = seat_log(&err_log)?
            .all(&record::RESIDENCY_PASS)?
            .iter()
            .map(|f| f.word("pass").map(str::to_owned))
            .collect::<Result<Vec<_>, _>>()?;
        println!(
            "slots-default residency passes: {} of kinds {:?}",
            kinds.len(),
            {
                let mut seen: Vec<&str> = kinds.iter().map(String::as_str).collect();
                seen.sort_unstable();
                seen.dedup();
                seen
            }
        );
        check(
            ok,
            "slots_default_residency_passes_end_prompt_and_decode",
            adaptive
                && kinds.iter().any(|k| k == "prompt")
                && kinds
                    .iter()
                    .any(|k| matches!(k.as_str(), "step" | "pair" | "slots" | "slots_drafted")),
        );
        // The chooser's record of each slot's request prints at that slot's
        // next prompt call: a one-id flush on each prompt, which takes the
        // slot that holds its ids. A round of both slots runs the draft's
        // whole width and is only counted ([`Choosing::round_ran`]): a pass
        // below the draft's width comes only from a head or tail round beside
        // the other's prompt, which a run need not have, so the cut is
        // [`width_cost`]'s clause. FAIL-first: a seat that runs the draft
        // with no chooser prints no record; a record off by a window turns
        // the sum red.
        let mut flushed = true;
        for ids in [&a_ids, &b_ids] {
            let flush = json!({
                "prompt": ids, "n_predict": 1, "temperature": 0, "return_tokens": true,
            });
            let (st, _) = curl(&url("/completion"), Some(&flush), false)?;
            flushed &= st == 200;
        }
        let records = seat_log(&err_log)?.all(&record::MTP_WIDTH)?;
        let mut parsed = Vec::new();
        for rec in &records {
            let widths: Vec<u64> = rec
                .csv("widths")?
                .iter()
                .map(|v| v.parse::<u64>())
                .collect::<Result<_, _>>()
                .map_err(|e| format!("mtp width widths: {e}"))?;
            parsed.push((rec.u64("windows")?, widths, rec.f64("e")?));
        }
        println!("slots-default flushed {flushed}: mtp width (windows, widths, e) {parsed:?}");
        check(
            ok,
            "slots_default_width_record_is_the_choosers",
            adaptive
                && flushed
                && !parsed.is_empty()
                && parsed.iter().any(|(w, _, _)| *w > 0)
                && parsed.iter().all(|(w, widths, e)| {
                    *w == widths.iter().skip(1).sum::<u64>() && (*w == 0 || (1.0..=4.0).contains(e))
                }),
        );
        // The reset route answers on the seat's residency: a 200 whose `diff`
        // is 0, and the server prints its `residency reset` record with the
        // same counts. FAIL-first: a seat whose open runs no machine answers
        // the server's 501 and prints none; one that prints other counts
        // than it answers turns the equality red.
        let (st, body) = curl(&url("/residency/reset"), Some(&json!({})), false)?;
        let reset: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
        let names = ["cancelled", "copies", "diff"];
        let counts = match seat_log(&err_log)?.first(&record::RESIDENCY_RESET)? {
            Some(r) => Some(
                names
                    .iter()
                    .map(|k| r.u64(k))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            None => None,
        };
        println!("slots-default residency reset: HTTP {st} {reset}; record {names:?} {counts:?}");
        check(
            ok,
            "slots_default_residency_reset_is_a_200_with_its_record",
            adaptive
                && st == 200
                && reset["diff"] == json!(0)
                && counts.is_some_and(|c| {
                    names
                        .iter()
                        .zip(&c)
                        .all(|(k, v)| reset[*k].as_u64() == Some(*v))
                }),
        );
        println!("slots-default server stopped: {}", served.stop()?);
        Ok(*ok)
    }

    /// The drafted clauses beside the slots clause's pair, on its server
    /// (the draft on, the residency off, the fixed window, the round records
    /// on) after its alone and together runs, `alone` the alone answers of
    /// `ids`: the off slot ([`slots_drafted_off_slot`]), then the late
    /// request in both orders ([`late_pair`]): the first posted and the
    /// second once `/metrics`' `n_decode_total` has moved
    /// [`DRAFTED_LATE_ROUNDS`] rounds — its prompt between the first's
    /// rounds — both requests' ids again their alone ids
    /// (`slots_drafted_late_request_together_alone`, `TURN_A` first;
    /// `slots_drafted_late_request_other_order_together_alone`, `SLOTS_B`
    /// first: one of the two lays slot 1's row first).
    ///
    /// The late request's first window is not empty: a request's prompt call
    /// ends in its last id's step, which records the draft's refresh
    /// (`genloop`'s prompt), so its first pass proposes beside the first
    /// request's deep one. A window rides a pass with no proposal only
    /// while its draft skips, `pass_slots`' own case; a slot a cut turned
    /// off steps beside the pass ([`slots_drafted_off_slot`]). The mixed
    /// round these clauses cover is that young window beside the decoding
    /// one.
    fn slots_drafted_beside(
        url: &dyn Fn(&str) -> String,
        addr: &str,
        err_log: &Path,
        ids: [&[u32]; 2],
        alone: &[Option<Streamed>],
    ) -> Result<bool, GateError> {
        let [a_ids, b_ids] = ids;
        let mut ok = slots_drafted_off_slot(url, addr, err_log, a_ids, alone[0].as_ref())?;
        // The late request, in both orders: the first decodes alone, and once
        // a few of its rounds have passed the second arrives — its prompt
        // between the first's rounds, its first drafted pass in a round
        // beside the first's deep one. Each request takes the slot that holds
        // its own last run, so one of the two orders puts slot 1's row first.
        for (name, order, ids) in [
            (
                "slots_drafted_late_request_together_alone",
                ["A", "B"],
                [a_ids, b_ids],
            ),
            (
                "slots_drafted_late_request_other_order_together_alone",
                ["B", "A"],
                [b_ids, a_ids],
            ),
        ] {
            let (polls, late) = late_pair(url, addr, ids, order)?;
            let want = match order {
                ["A", _] => [&alone[0], &alone[1]],
                _ => [&alone[1], &alone[0]],
            };
            println!(
                "slots-drafted late {}{}: the second posted after {polls} poll(s); {} and {} ids",
                order[0],
                order[1],
                late[0].as_ref().map_or(0, |s| s.tokens.len()),
                late[1].as_ref().map_or(0, |s| s.tokens.len()),
            );
            check(
                &mut ok,
                name,
                late.iter().zip(want).all(|(l, a)| {
                    l.as_ref()
                        .zip(a.as_ref())
                        .is_some_and(|(l, a)| !l.tokens.is_empty() && l.tokens == a.tokens)
                }),
            );
        }
        Ok(ok)
    }

    /// The late pair of the drafted rounds clauses: `ids[0]` posted, and
    /// once `/metrics`' `n_decode_total` has moved [`DRAFTED_LATE_ROUNDS`]
    /// rounds (polled at [`DRAFTED_LATE_POLL`], at most
    /// [`DRAFTED_LATE_POLLS`] times) `ids[1]`, both streamed greedy for
    /// [`SLOTS_PREDICT`] tokens: the polls taken and each request's answer,
    /// `None` a request that failed (a red check, not the clause's end).
    fn late_pair(
        url: &dyn Fn(&str) -> String,
        addr: &str,
        ids: [&[u32]; 2],
        what: [&str; 2],
    ) -> Result<(usize, Vec<Option<Streamed>>), GateError> {
        let first = streamed(addr, ids[0], SLOTS_PREDICT)?;
        let decode0 = metric(url, "n_decode_total")?.unwrap_or(f64::NAN);
        let mut polls = 0;
        while polls < DRAFTED_LATE_POLLS
            && metric(url, "n_decode_total")?.unwrap_or(decode0)
                < decode0 + DRAFTED_LATE_ROUNDS as f64
        {
            std::thread::sleep(DRAFTED_LATE_POLL);
            polls += 1;
        }
        let second = streamed(addr, ids[1], SLOTS_PREDICT)?;
        let mut late = Vec::new();
        for ((h, rx), what) in [(first, what[0]), (second, what[1])] {
            h.join()
                .map_err(|_| format!("slots-drafted: the late {what} request's thread panicked"))?;
            let ran = rx.recv().map_err(|_| {
                format!("slots-drafted: the late {what} request's thread gave no answer")
            })?;
            match ran {
                Ok(s) => late.push(Some(s)),
                Err(e) => {
                    println!("slots-drafted: the late {what} request failed: {e}");
                    late.push(None);
                }
            }
        }
        Ok((polls, late))
    }

    /// The off slot clause (the module header) on the slots clause's
    /// server: the break-even clause's long turn (`BREAK_EVEN_TURNS`'
    /// first) run fresh on a free slot for a reply its break-even keeps a
    /// cut for; then `TURN_A`'s drafted request (`a_ids`, [`SLOTS_PREDICT`]
    /// tokens) posted — it takes the other slot, whose held ids share more
    /// of its prompt — and once it has run [`DRAFTED_LATE_ROUNDS`] decode
    /// calls, the turn resent with its reply stripped, which only the turn's
    /// slot is free to take: that slot keeps the turn's end by a cut, its
    /// draft off by name (its `mtp keep` record `kept`), and its plain steps
    /// run in the rounds the drafted request's windows run in (a `slots
    /// round` record of two rows inside its window). The off slot's ids are
    /// the same ids fed fresh, and the drafted request's ids and draft
    /// counts its alone run's (`alone_a`): an off slot steps beside a
    /// drafted window and is never asked to propose
    /// (`slots_drafted_off_slot_steps_beside_a_window`).
    fn slots_drafted_off_slot(
        url: &dyn Fn(&str) -> String,
        addr: &str,
        err_log: &Path,
        a_ids: &[u32],
        alone_a: Option<&Streamed>,
    ) -> Result<bool, GateError> {
        let k = Keep38::read(err_log)?;
        let (turn, reply_text) = BREAK_EVEN_TURNS[0];
        let p1 = rendered(url, json!([{ "role": "user", "content": turn }]))?;
        let stripped = tokenized(url, reply_text)?;
        let turn_end = p1.len() - 1;
        let n = (2..=N_PREDICT)
            .rev()
            .find(|&n| k.at(n) <= turn_end)
            .ok_or_else(|| {
                format!(
                    "slots-drafted off: the turn's end {turn_end} is below the break-even of \
                     every reply of 2 to {N_PREDICT} tokens ({} ids a token): the clause needs \
                     a longer turn",
                    k.per_token
                )
            })?;
        let strip: Vec<u32> = p1.iter().chain(&stripped).copied().collect();
        let x = greedy(url, err_log, &p1, n, false)?;
        x.show("slots-drafted off: the turn fresh");
        if stripped.first() == x.tokens.first() {
            return Err(format!(
                "slots-drafted off: the stripped reply's first id {:?} is the reply's: the \
                 shared prefix would not end at the turn",
                stripped.first()
            )
            .into());
        }
        let decode0 = metric(url, "n_decode_total")?.unwrap_or(f64::NAN);
        let (h, rx) = streamed(addr, a_ids, SLOTS_PREDICT)?;
        let mut polls = 0;
        while polls < DRAFTED_LATE_POLLS
            && metric(url, "n_decode_total")?.unwrap_or(decode0)
                < decode0 + DRAFTED_LATE_ROUNDS as f64
        {
            std::thread::sleep(DRAFTED_LATE_POLL);
            polls += 1;
        }
        let before = seat_log(err_log)?.all(&record::SLOTS_ROUND)?.len();
        let cut = greedy(url, err_log, &strip, n, true)?;
        cut.show("slots-drafted off: the turn resent stripped beside the drafted request");
        let rounds = seat_log(err_log)?.all(&record::SLOTS_ROUND)?;
        let beside = rounds[before.min(rounds.len())..]
            .iter()
            .filter(|r| r.u64("rows") == Ok(2))
            .count();
        h.join()
            .map_err(|_| "slots-drafted off: the drafted request's thread panicked")?;
        let drafted = match rx
            .recv()
            .map_err(|_| "slots-drafted off: the drafted request's thread gave no answer")?
        {
            Ok(s) => Some(s),
            Err(e) => {
                println!("slots-drafted off: the drafted request failed: {e}");
                None
            }
        };
        let f = greedy(url, err_log, &strip, n, false)?;
        f.show("slots-drafted off: the same ids fresh");
        let rec = k.record(n, turn_end as u64, true);
        let off = cut.cache_n == turn_end as u64
            && cut.skipped_by_name()
            && cut.keeps.last() == Some(&rec);
        let fresh = x.ok && f.ok && f.cache_n == 0 && f.offs.is_empty();
        let off_same = !cut.tokens.is_empty() && cut.tokens == f.tokens;
        let drafted_same = drafted.as_ref().zip(alone_a).is_some_and(|(d, a)| {
            !d.tokens.is_empty() && d.tokens == a.tokens && d.drafts == a.drafts
        });
        println!(
            "slots-drafted off: posted after {polls} poll(s); the cut kept {} of the turn's end \
             {turn_end} (want {rec:?}), {beside} round(s) of two rows beside the drafted \
             request; off ids the fresh run's: {off_same}; drafted ids and counts its alone \
             run's: {drafted_same}",
            cut.cache_n
        );
        let mut ok = true;
        check(
            &mut ok,
            "slots_drafted_off_slot_steps_beside_a_window",
            off && fresh && off_same && drafted_same && beside > 0,
        );
        Ok(ok)
    }

    /// The width chooser's own server (the module header): the draft on,
    /// the residency off and `BLOOMERY_MTP_WIDTH=cost`, one greedy request
    /// of the gate's prompt then a one-id flush — the chooser's `mtp width`
    /// record prints at the slot's next prompt call — holding: the request's
    /// ids the plain run's (the chooser changes which rows a pass runs,
    /// never a token), its draft counts carried, and the record the
    /// chooser's own: its windows the widths above 0 it names, E inside the
    /// window's rows, and at least one pass of a width below the draft's
    /// own (the warm-up rotation's measurements, on this server too). The
    /// per-window width is the CLI gate's clause (its window records); this
    /// server's records are the chooser's aggregates.
    fn width_cost(dir: &Path, prompt: &str, reference: &[u32]) -> Result<bool, GateError> {
        let own = dir.join("width-cost");
        std::fs::create_dir_all(&own)?;
        let err_log = own.join("server.err");
        let mut cmd = Command::new(Served38::exe()?);
        cmd.env(bloomery_levers::DRAFT, "mtp")
            .env(bloomery_levers::RESIDENCY, "off")
            .env(bloomery_levers::MTP_WIDTH, "cost");
        let mut served = Served38::spawn_with(&SERVER_ARGS, &own, &mut cmd)?;
        println!("width-cost server pid {}", served.child.id());
        let addr = match served.address(&err_log, POLLS, POLL) {
            Ok(a) => a,
            Err(e) => {
                println!(
                    "width-cost: the server never listened: {e}; it said: {}",
                    std::fs::read_to_string(&err_log).unwrap_or_default()
                );
                let _ = served.stop();
                for name in [
                    "width_cost_ids_are_the_plain_runs",
                    "width_cost_draft_counts_carried",
                    "width_cost_record_is_the_choosers",
                    "width_cost_cut_a_window",
                ] {
                    check(&mut false, name, false);
                }
                return Ok(false);
            }
        };
        let url = |p: &str| format!("http://{addr}{p}");
        let mut ok = true;
        let completion = json!({
            "prompt": prompt, "n_predict": N_PREDICT, "temperature": 0, "return_tokens": true,
        });
        let (st, body) = curl(&url("/completion"), Some(&completion), false)?;
        let c = json_of("/completion", st, &body)?;
        let ids = ids_of(&c["tokens"]);
        let stop = c["stop_type"].as_str().unwrap_or("").to_owned();
        println!("width-cost tokens {ids:?} stop_type={stop}");
        check(
            &mut ok,
            "width_cost_ids_are_the_plain_runs",
            agree(&ids, &stop, reference),
        );
        let d = &c["timings"];
        check(
            &mut ok,
            "width_cost_draft_counts_carried",
            d["draft_n"].as_u64().is_some_and(|n| n > 0)
                && d["draft_n_accepted"].as_u64().is_some_and(|n| n > 0),
        );
        // The flush: the chooser's record of the request above prints at
        // this prompt call.
        let flush = json!({
            "prompt": prompt, "n_predict": 1, "temperature": 0, "return_tokens": true,
        });
        let (st, _) = curl(&url("/completion"), Some(&flush), false)?;
        check(&mut ok, "width_cost_flush_ok", st == 200);
        let records = seat_log(&err_log)?.all(&record::MTP_WIDTH)?;
        println!("width-cost {records:?}");
        let Some(rec) = records.first() else {
            check(&mut ok, "width_cost_record_is_the_choosers", false);
            check(&mut ok, "width_cost_cut_a_window", false);
            println!("width-cost server stopped: {}", served.stop()?);
            return Ok(ok);
        };
        let windows = rec.u64("windows")?;
        let widths: Vec<u64> = rec
            .csv("widths")?
            .iter()
            .map(|v| v.parse::<u64>())
            .collect::<Result<_, _>>()
            .map_err(|e| format!("mtp width widths: {e}"))?;
        let verified: u64 = widths
            .iter()
            .enumerate()
            .skip(1)
            .map(|(k, &n)| n * k as u64)
            .sum();
        let e = rec.f64("e")?;
        println!(
            "width-cost record windows={windows} widths={widths:?} e={e:.3} (verified \
             {verified} positions)"
        );
        check(
            &mut ok,
            "width_cost_record_is_the_choosers",
            windows > 0
                && windows == widths.iter().skip(1).sum::<u64>()
                && (1.0..=4.0).contains(&e),
        );
        check(
            &mut ok,
            "width_cost_cut_a_window",
            widths.iter().take(3).any(|&n| n > 0),
        );
        println!("width-cost server stopped: {}", served.stop()?);
        Ok(ok)
    }

    /// The sampled rounds clauses (the module header) on one server of
    /// `--parallel 2` under `BLOOMERY_DRAFT=off`, `BLOOMERY_RESIDENCY=off`
    /// and `BLOOMERY_STEP_STATS=1`, into its own directory: two sampled
    /// streamed requests of the slots clause's prompts — temperature
    /// [`SAMPLED_TEMPERATURE`], [`SAMPLED_SEED`] each, so a request's ids
    /// are a function of its logits rows alone (the server seeds each
    /// request's sampler from its own `seed`) — each first run alone on
    /// that server and then both together, the together run's `slots
    /// round` records read back — every one `cmd=step` of `passes=1`
    /// carrying both slots (`rows=2`, `slots=2`), and each request's
    /// together ids its alone run's: the seat runs a round of plain steps
    /// as one pass of its rows, each row's answer and lent logits row its
    /// own. The alone runs are on this server, the load the together run
    /// runs (the plan of 2 sequences); a `--parallel 1` load's ids are no
    /// reference (the placement differs).
    fn slots_sampled_rounds(dir: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let d = dir.join("slots-sampled");
        std::fs::create_dir_all(&d)?;
        let err_log = d.join("server.err");
        // SERVER_ARGS ends in the one-slot `--parallel 1`; this clause's
        // server takes the two-slot value, the total context the same 4096.
        let mut args: Vec<&str> = SERVER_ARGS.to_vec();
        let parallel = args.len() - 1;
        args[parallel] = "2";
        let mut cmd = Command::new(Served38::exe()?);
        cmd.env(bloomery_levers::DRAFT, "off")
            .env(bloomery_levers::RESIDENCY, "off")
            .env_remove(bloomery_levers::MTP_HEAD_ROWS)
            .env_remove(bloomery_levers::MTP_DRAFT)
            .env(bloomery_levers::STEP_STATS, "1");
        let mut served = Served38::spawn_with(&args, &d, &mut cmd)?;
        println!("slots-sampled server pid {}", served.child.id());
        let addr = match served.address(&err_log, POLLS, POLL) {
            Ok(a) => a,
            // A server that never listens names no round: the checks are red
            // by name, and the server is stopped.
            Err(e) => {
                println!(
                    "slots-sampled: the server never listened: {e}; it said: {}",
                    std::fs::read_to_string(&err_log).unwrap_or_default()
                );
                for name in [
                    "slots_sampled_rounds_run_one_pass",
                    "slots_sampled_together_alone",
                ] {
                    check(&mut ok, name, false);
                }
                let _ = served.stop();
                return Ok(ok);
            }
        };
        let url = |p: &str| format!("http://{addr}{p}");
        let (a_ids, b_ids) = (
            rendered(&url, json!([{ "role": "user", "content": TURN_A }]))?,
            rendered(&url, json!([{ "role": "user", "content": SLOTS_B }]))?,
        );
        let mut alone = Vec::new();
        for (ids, what) in [(&a_ids, "alone A"), (&b_ids, "alone B")] {
            alone.push(run_sampled(&addr, ids, SLOTS_PREDICT, what)?);
        }
        // The alone runs print no record (a round of one active request is a
        // select and a step); taken anyway, so the together run's records are
        // exactly the ones after this point.
        let before = seat_log(&err_log)?.all(&record::SLOTS_ROUND)?.len();
        let first = sampled_stream(&addr, &a_ids, SLOTS_PREDICT)?;
        let second = sampled_stream(&addr, &b_ids, SLOTS_PREDICT)?;
        let mut together = Vec::new();
        for ((h, rx), what) in [(first, "together A"), (second, "together B")] {
            h.join()
                .map_err(|_| format!("slots-sampled: the {what} request's thread panicked"))?;
            let ran = rx.recv().map_err(|_| {
                format!("slots-sampled: the {what} request's thread gave no answer")
            })?;
            match ran {
                Ok(s) => together.push(Some(s)),
                Err(e) => {
                    println!("slots-sampled: the {what} request failed: {e}");
                    together.push(None);
                }
            }
        }
        let rounds = seat_log(&err_log)?.all(&record::SLOTS_ROUND)?;
        let window = &rounds[before.min(rounds.len())..];
        let every = !window.is_empty()
            && window.iter().all(|r| {
                r.word("cmd") == Ok("step")
                    && r.u64("passes") == Ok(1)
                    && r.u64("rows") == Ok(2)
                    && r.u64("slots") == Ok(2)
            });
        println!(
            "slots-sampled: {} record(s) in the together window, {} of them one pass of 2 rows \
             (alone {} and {} ids, together {} and {} ids)",
            window.len(),
            window
                .iter()
                .filter(|r| r.word("cmd") == Ok("step") && r.u64("passes") == Ok(1))
                .count(),
            alone[0].as_ref().map_or(0, |s| s.tokens.len()),
            alone[1].as_ref().map_or(0, |s| s.tokens.len()),
            together[0].as_ref().map_or(0, |s| s.tokens.len()),
            together[1].as_ref().map_or(0, |s| s.tokens.len()),
        );
        check(&mut ok, "slots_sampled_rounds_run_one_pass", every);
        check(
            &mut ok,
            "slots_sampled_together_alone",
            together[0].as_ref().is_some_and(|t| !t.tokens.is_empty())
                && together[1].as_ref().is_some_and(|t| !t.tokens.is_empty())
                && alone[0]
                    .as_ref()
                    .is_some_and(|a| a.tokens == together[0].as_ref().unwrap().tokens)
                && alone[1]
                    .as_ref()
                    .is_some_and(|a| a.tokens == together[1].as_ref().unwrap().tokens),
        );
        println!("slots-sampled server stopped: {}", served.stop()?);
        Ok(ok)
    }

    /// The plan's card expert bytes on the gate card at a slot's `ctx`,
    /// `slots` resident sequences counted (`plan_with_slots`), plain, as the
    /// server makes it — the clause's flagless server's two, the refusal
    /// arm's one; `None` when no plan takes the context.
    fn card_at(
        inputs: &PlanInputs,
        levers: &PlanLevers,
        experts: Experts,
        ctx: usize,
        free: Option<u64>,
        slots: usize,
    ) -> Result<Option<u64>, GateError> {
        let mut machine = machine_for_experts(
            RTX_3090,
            inputs.spec.layers.len(),
            u64::try_from(ubatch_for(ctx)?)?,
            experts,
        );
        machine.cards[0].free_bytes = free;
        Ok(inputs
            .plan_with_slots(&machine, u64::try_from(ctx)?, levers, experts, slots)
            .ok()
            .and_then(|p| p.cards.first().map(|c| c.expert_bytes)))
    }

    /// The gate's own plan of the file on the gate card (`free` bytes free,
    /// as the server's `plan` record names them) at a slot's `ctx` positions
    /// for `slots` resident sequences, plain, under the host room `room`
    /// (given, not read), handed to `then` — the server's plan but for its
    /// machine's host reserve of the checkpoints, which this machine, the
    /// OS's alone, does not hold.
    fn gate_plan_at<T>(
        levers: &bloomery_levers::Levers,
        (ctx, slots): (usize, usize),
        free: u64,
        room: u64,
        then: impl FnOnce(Result<model::placement::Plan<'_>, PlaceError>) -> Result<T, GateError>,
    ) -> Result<T, GateError> {
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mut inputs = PlanInputs::describe(&split)?;
        inputs.room = (room, HostRead::Given);
        let experts = experts38(levers)?;
        let mut machine = machine_for_experts(
            RTX_3090,
            inputs.spec.layers.len(),
            u64::try_from(ubatch_for(ctx)?)?,
            experts,
        );
        machine.cards[0].free_bytes = Some(free);
        then(inputs.plan_with_slots(
            &machine,
            u64::try_from(ctx)?,
            &PlanLevers::from_levers(levers)?,
            experts,
            slots,
        ))
    }

    /// The NVMe tier's RAM arena and the host's need ([`HostNeed::bytes`],
    /// the arena inside it) of [`gate_plan_at`]'s plan at the host room
    /// `room`.
    fn host_split_at(
        levers: &bloomery_levers::Levers,
        shape: (usize, usize),
        free: u64,
        room: u64,
    ) -> Result<(u64, u64), GateError> {
        gate_plan_at(levers, shape, free, room, |plan| {
            let plan = plan?;
            Ok((plan.host.nvme_arena_bytes, HostNeed::of(&plan, 0).bytes()))
        })
    }

    /// The NVMe expert tier's floor of [`gate_plan_at`]'s plan: the one
    /// `HostRoomFloor` names for a room of one byte (the planner's one owner
    /// of the floor), read by type.
    fn host_floor(
        levers: &bloomery_levers::Levers,
        shape: (usize, usize),
        free: u64,
    ) -> Result<u64, GateError> {
        gate_plan_at(levers, shape, free, 1, |plan| match plan {
            Err(PlaceError::Placement(PlacementError::HostRoomFloor { floor, .. })) => Ok(floor),
            Err(e) => Err(e.into()),
            Ok(_) => Err("a host room of one byte planned: no floor to read".into()),
        })
    }

    /// The `ctx` clause: a plain server with no `--ctx-size` (and no
    /// `--parallel`, so the seat's default two resident slots) prints its
    /// `ctx` line before its load (it is stopped there), against this gate's
    /// own plans of the file — every plan counting the two slots, as the
    /// seat's does: its context is a slot's, the margin rule's answer
    /// (`margin`: the line's own `margin_ctx`, which the clause below holds
    /// to the gate's plans), or the largest the card holds when that is
    /// fewer than [`CTX`] (`card`), or the prompt cache's clamp of the
    /// margin's answer (`cache`, composed the seat's way from the cache
    /// line's own ram — the clamp rides MemAvailable as the fit rides the
    /// census); the line's `fit` is the largest slot context the card holds
    /// up to the file's serving cap (`place::serve_ctx`, its
    /// `context_length`) — its plan taken, and the one past it refused when
    /// the card bounds it, else the cap itself — with that plan's card
    /// expert bytes; its `margin_ctx` holds at most the plan's margin fewer
    /// card expert bytes than the two-slot plan at [`CTX`], and the next
    /// multiple of [`CTX_STEP`] past it more (or it is the fit). A server
    /// with `--ctx-size` set and no `--parallel` takes one slot at the whole
    /// flag (the `from=ctx` rule; the clause's last server holds the line
    /// and the one-request note beside it), and one asked for a context one
    /// past the one-slot fit is refused by name
    /// before it listens: by the card's rule when the card bounds that fit,
    /// by the cap's (the trained context, YaRN not built) when the cap
    /// does. Mutants: the default ignoring the fit (staying at the base);
    /// the default passing the fit through without the margin rule; the
    /// refusal of a context past the card taken out (the load's own plan
    /// refuses it, not by the rule's name); the cap's refusal taken out
    /// (the card's rule names the clamped fit instead); a set `--ctx-size`
    /// still split by the default two (the one-slot line names two slots,
    /// and the one-request note prints nowhere).
    // PIN(2026-10-05): re-derived for the seat's default two resident slots — every
    // `at` below is `plan_with_slots(2)` at a slot's context, and the default
    // `--parallel` names two (the `parallel` line's check). Was the
    // one-sequence search (round q38rules). The refusal arm's own count moved
    // to one slot by the PIN(2026-10-07) below.
    // PIN(2026-10-07): the refusal arm and the one-slot arm take one slot — a set
    // `--ctx-size` with no `--parallel` is one request's context now (`placement::ctx::slots_of`),
    // so the arm's `at1` below is `plan_with_slots(1)` and its ask is one past the
    // one-slot fit.
    fn ctx(dir: &Path, levers: &bloomery_levers::Levers) -> Result<bool, GateError> {
        let own = dir.join("ctx");
        std::fs::create_dir_all(&own)?;
        let err_log = own.join("server.err");
        let spawn = |args: &[&str]| -> Result<Served38, GateError> {
            let mut cmd = Command::new(Served38::exe()?);
            cmd.env(bloomery_levers::DRAFT, "off")
                .env(bloomery_levers::RESIDENCY, "off")
                .env_remove(bloomery_levers::MTP_HEAD_ROWS)
                .env_remove(bloomery_levers::MTP_DRAFT);
            Served38::spawn_with(args, &own, &mut cmd)
        };
        let mut served = spawn(&DEFAULT_ARGS)?;
        let mut line = None;
        for _ in 0..POLLS {
            let text = std::fs::read_to_string(&err_log).unwrap_or_default();
            line = text
                .lines()
                .find(|l| l.starts_with("ctx rule="))
                .map(str::to_owned);
            // The `plan` record carries the free reading the re-derivations
            // below take ([`server_card_free`]), and the seat prints its `ctx`
            // line before it: both must be in the log before the server
            // stops, or the reading is raced away.
            // A poll of a log still being written: a line that does not read
            // whole yet is not there yet ([`server_card_free`] reads it once
            // the server stops).
            let plan = matches!(
                record::Log::of(&text, record::BLOOMERY_SERVE_QWEN38).first(&record::PLAN38),
                Ok(Some(_))
            );
            if (line.is_some() && plan) || served.child.try_wait()?.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
        if served.child.try_wait()?.is_none() {
            println!("ctx: the default server stopped: {}", served.stop()?);
        }
        let line = line.ok_or("the default server printed no `ctx` line")?;
        println!("ctx: {line}");
        // The resident-slot default's own line (the module header): this
        // flagless server is the one that takes it, [`SERVER_ARGS`] naming
        // the one slot its clauses count on. The slots its rule names are
        // what the line's own terms hold — the default's 2 and the split it
        // names the `ctx` line's own context. FAIL-first: a default that
        // serves another count (or a split off the `ctx` line) turns this
        // red.
        let parallel =
            parallel_line(&err_log)?.ok_or("the default server printed no `parallel` line")?;
        println!("ctx: {parallel}");
        let field = |k: &str| -> Option<String> {
            line.split_whitespace()
                .find_map(|w| w.strip_prefix(k)?.strip_prefix('='))
                .map(str::to_owned)
        };
        let num = |k: &str| field(k).and_then(|v| v.parse::<usize>().ok());
        let (rule, ctx, fit) = (field("rule"), num("ctx"), num("fit"));
        let fit_bytes = field("fit_card_expert_bytes").and_then(|v| v.parse::<u64>().ok());
        let margin_ctx = num("margin_ctx");
        let (Some(rule), Some(ctx), Some(fit), Some(fit_bytes), Some(margin_ctx)) =
            (rule, ctx, fit, fit_bytes, margin_ctx)
        else {
            return Err(format!(
                "a `ctx` line without its rule, ctx, fit, fit_card_expert_bytes and margin_ctx: \
                 {line}"
            )
            .into());
        };
        // The rule's own terms: the default's 2 slots at the `ctx` line's own
        // context (the line's `total` twice it, held by `parallel_agrees`).
        let pfield = |k: &str| {
            parallel
                .split_whitespace()
                .find_map(|w| w.strip_prefix(k)?.strip_prefix('='))
                .and_then(|v| v.parse::<usize>().ok())
        };
        let parallel_ok = parallel_agrees(&parallel, None).unwrap_or(false)
            && pfield("slots") == Some(2)
            && pfield("slot_ctx") == Some(ctx);
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::describe(&split)?;
        let experts = experts38(levers)?;
        let plan_levers = PlanLevers::from_levers(levers)?;
        let free = Some(server_card_free(&err_log)?);
        // The default the seat's rules compose to also answers the prompt
        // cache's bound, whose ram rides MemAvailable as the fit rides the
        // census: the gate takes the server's own `cache` line's ram, not a
        // fresh reading.
        let ram = record_line(&err_log, "cache")?
            .ok_or("the default server printed no `cache` line")?
            .split_whitespace()
            .find_map(|w| w.strip_prefix("ram="))
            .and_then(|v| v.parse::<u64>().ok())
            .ok_or("a `cache` line without its ram")?;
        let at = |c: usize| card_at(&inputs, &plan_levers, experts, c, free, 2);
        let base = at(CTX)?.ok_or("no plan at the base context")?;
        let lost = |c: usize| -> Result<Option<u64>, GateError> {
            Ok(at(c)?.map(|e| base.saturating_sub(e)))
        };
        let margin = model::placement::workstation::MARGIN;
        let here = lost(margin_ctx)?;
        let past = lost(margin_ctx + CTX_STEP)?;
        let (at_fit, past_fit) = (at(fit)?, at(fit + 1)?);
        let cap = usize::try_from(serve_ctx(1, &inputs.hp)?)?;
        // PIN(2026-10-02): the fit is the card's largest context up to the file's serving cap
        // (round q38rules: `serve_ctx`, 262,144 for Qwen3.8); where the card holds more, the fit
        // is the cap and a plan past it is still a plan; was the card's largest, `past_fit` none.
        let card_bounds = fit < cap;
        // One session's whole-context state (`Seq38`), the bound the seat's
        // `cache` rule clamps the default with (`Ctx38::host_bound`).
        let gdn = inputs
            .spec
            .layers
            .iter()
            .filter(|l| matches!(l.mixer, Mixer::DeltaRule(_)))
            .count();
        let qsa = inputs.spec.layers.len() - gdn;
        let state = |n: usize| seq_positional_bytes(qsa, n) + 2 * checkpoint_bytes(gdn);
        let cache_ctx = |from: usize| {
            let mut lo = 1;
            let mut hi = from;
            while lo < hi {
                let mid = lo + (hi - lo).div_ceil(2);
                if state(mid) <= ram {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            (lo / CTX_STEP * CTX_STEP).max(lo.min(CTX_STEP))
        };
        let (want_rule, want_ctx) = if fit < CTX {
            ("card", fit)
        } else if ram == 0 || state(margin_ctx) <= ram {
            ("margin", margin_ctx)
        } else {
            ("cache", cache_ctx(margin_ctx))
        };
        println!(
            "ctx: the gate's plans: lost at {margin_ctx} {here:?}, at {} {past:?}, fit {fit} \
             {at_fit:?}, past it {past_fit:?}; the serving cap {cap} ({}); the default the rules \
             compose to: rule {want_rule} ctx {want_ctx} (the cache line's ram {ram})",
            margin_ctx + CTX_STEP,
            if card_bounds {
                "the card bounds the fit"
            } else {
                "the cap bounds the fit"
            }
        );
        let mut ok = true;
        check(
            &mut ok,
            "ctx_default_is_the_rules",
            rule.as_str() == want_rule && ctx == want_ctx,
        );
        check(
            &mut ok,
            "parallel_default_holds_what_its_rule_names",
            parallel_ok,
        );
        check(
            &mut ok,
            "ctx_line_prints_the_fit_and_the_margin",
            at_fit == Some(fit_bytes)
                && fit <= cap
                && (if card_bounds {
                    past_fit.is_none()
                } else {
                    fit == cap
                })
                && here.is_some_and(|l| l <= margin)
                && margin_ctx <= fit
                && (margin_ctx == fit
                    || (margin_ctx.is_multiple_of(CTX_STEP) && past.is_none_or(|l| l > margin))),
        );
        // PIN(2026-10-07): the refusal arm's server takes one slot — a set
        // `--ctx-size` with no `--parallel` is one request's context now, the
        // one-slot `from=ctx` rule the arm below holds — so the arm asks for
        // one past the one-slot fit (every `at1` a `plan_with_slots(1)` at a
        // slot's context), not the total whose two-slot share was one past
        // the two-slot fit. Re-pin by specification (the seat's
        // `--ctx-size` rule), not a relaxation: the refusal is still asked
        // one past the boundary the card holds, at the slot count the flag
        // set now serves, and still by name.
        let at1 = |c: usize| card_at(&inputs, &plan_levers, experts, c, free, 1);
        let (mut lo, mut hi) = (1, cap);
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if at1(mid)?.is_some() {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let fit1 = lo;
        let card_bounds1 = fit1 < cap;
        let over = (fit1 + 1).to_string();
        let args: Vec<&str> = DEFAULT_ARGS
            .iter()
            .copied()
            .chain(["--ctx-size", &over])
            .collect();
        let mut refused = spawn(&args)?;
        let mut status = None;
        for _ in 0..POLLS {
            status = refused.child.try_wait()?;
            if status.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
        let text = std::fs::read_to_string(&err_log).unwrap_or_default();
        let named = if card_bounds1 {
            format!(
                "--ctx-size {over}: with --parallel 1 each slot takes {over} positions and the \
                 card holds at most {fit1} a slot beside the plan's dense weights (`--place gate`)"
            )
        } else {
            format!(
                "a context of {over} positions: the file was trained at {cap} (context_length), \
                 and a load serves at most {cap}; YaRN scaling past the trained context is not \
                 implemented"
            )
        };
        let listened = record::Log::of(&text, record::BLOOMERY_SERVE_QWEN38)
            .first(&record::LISTENING38)?
            .is_some();
        println!(
            "ctx: --ctx-size {over}: exit {status:?}; named {}; listened {listened}",
            text.contains(&named)
        );
        if status.is_none() {
            println!("ctx: the refused server stopped: {}", refused.stop()?);
        }
        check(
            &mut ok,
            "ctx_past_the_fit_is_refused_by_name",
            status.is_some_and(|s| !s.success()) && text.contains(&named) && !listened,
        );

        // The one-slot rule itself, the flag set the refusal arm now shares:
        // `--ctx-size 8192` with no `--parallel` prints its `parallel` line
        // (before the load; the server is stopped there) with one slot at
        // the whole 8192, `from=ctx`, and the one-request line beside it.
        // FAIL-first on the old split: that line read `slots=2
        // slot_ctx=4096 total=8192` and carried no `from=`, and the
        // one-request line printed nowhere.
        let ctx_arg = "8192".to_owned();
        let one_args: Vec<&str> = DEFAULT_ARGS
            .iter()
            .copied()
            .chain(["--ctx-size", ctx_arg.as_str()])
            .collect();
        let mut served = spawn(&one_args)?;
        let mut pline = None;
        for _ in 0..POLLS {
            let text = std::fs::read_to_string(&err_log).unwrap_or_default();
            pline = text
                .lines()
                .find(|l| l.starts_with("parallel rule="))
                .map(str::to_owned);
            if pline.is_some() || served.child.try_wait()?.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
        if served.child.try_wait()?.is_none() {
            println!("ctx: the one-slot server stopped: {}", served.stop()?);
        }
        let text = std::fs::read_to_string(&err_log).unwrap_or_default();
        let one_request = format!(
            "--ctx-size {ctx_arg} is one request's context; add --parallel N to serve N requests \
             at once (they split it)"
        );
        let agrees = pline.as_deref().is_some_and(|l| {
            let f = |k: &str| {
                l.split_whitespace()
                    .find_map(|w| w.strip_prefix(&format!("{k}=")))
            };
            f("slots") == Some("1")
                && f("slot_ctx") == Some("8192")
                && f("total") == Some("8192")
                && f("from") == Some("ctx")
        });
        println!(
            "ctx: one-slot {pline:?}; one-request line {}; it said: {text}",
            text.contains(&one_request)
        );
        check(
            &mut ok,
            "ctx_set_parallel_unset_is_one_slot",
            agrees && text.contains(&one_request),
        );
        Ok(ok)
    }

    /// `residency_loads_the_word` (the module header): a server of
    /// [`SERVER_ARGS`] under [`RESIDENCY_WORD`] prints, before its load (it
    /// is stopped there), the `residency lever` record of that word with
    /// why `set` — the seat took the word as given, not the unset rule's
    /// pick — and the `residency host` record of the same word, the churn
    /// pool its plan holds. FAIL-first: a seat that ignores a set word
    /// leaves the unset rule's `off` at `--place gate` and prints no host
    /// record (the server then loads and listens, which ends the poll); one
    /// that records the lever as unset names another why.
    fn residency_loads_the_word(dir: &Path) -> Result<bool, GateError> {
        let own = dir.join("residency-word");
        std::fs::create_dir_all(&own)?;
        let err_log = own.join("server.err");
        let mut cmd = Command::new(Served38::exe()?);
        cmd.env(bloomery_levers::RESIDENCY, RESIDENCY_WORD)
            .env(bloomery_levers::DRAFT, "off")
            .env_remove(bloomery_levers::MTP_HEAD_ROWS)
            .env_remove(bloomery_levers::MTP_DRAFT)
            .env_remove(bloomery_levers::MTP_WIDTH);
        let mut served = Served38::spawn_with(&SERVER_ARGS, &own, &mut cmd)?;
        println!("residency-word server pid {}", served.child.id());
        for _ in 0..POLLS {
            let text = std::fs::read_to_string(&err_log).unwrap_or_default();
            // A log still being written: a line that does not read whole yet
            // is not there yet.
            let log = record::Log::of(&text, record::BLOOMERY_SERVE_QWEN38);
            let printed = matches!(log.first(&record::RESIDENCY_HOST), Ok(Some(_)))
                || matches!(log.first(&record::LISTENING38), Ok(Some(_)));
            if printed || served.child.try_wait()?.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
        if served.child.try_wait()?.is_none() {
            println!("residency-word: the server stopped: {}", served.stop()?);
        }
        let log = seat_log(&err_log)?;
        let lever = log.first(&record::RESIDENCY_LEVER)?;
        let host = log.first(&record::RESIDENCY_HOST)?;
        let said = lever
            .as_ref()
            .map(|l| -> Result<(String, String), ReadError> {
                Ok((l.word("residency")?.to_owned(), l.word("why")?.to_owned()))
            })
            .transpose()?;
        let host_word = host
            .as_ref()
            .map(|h| h.word("residency").map(str::to_owned))
            .transpose()?;
        println!("residency-word: lever {said:?}, host {host_word:?}");
        let mut ok = true;
        check(
            &mut ok,
            "residency_loads_the_word",
            said.as_ref()
                .is_some_and(|(word, why)| word == RESIDENCY_WORD && why == "set")
                && host_word.as_deref() == Some(RESIDENCY_WORD),
        );
        Ok(ok)
    }

    /// The MTP draft's window condition judged on a slot's share of the total
    /// context (the seat's module doc): `--ctx-size 8 --parallel 2` gives
    /// each slot four positions, one short of the window's (`VERIFY_ROWS` +
    /// 1 = five a slot, ten for the two). With `BLOOMERY_DRAFT` unset and no
    /// `--place` (the box pin's one card, `a`, whose stage runs the draft
    /// where its file is) the draft turns off: the server prints no `draft
    /// keep` line, refuses nothing and loads, its `load draft=off` record
    /// naming the ten positions past the flag's eight
    /// (`draft_unset_yields_to_the_slot_share`); with `BLOOMERY_DRAFT=mtp`
    /// set the server is refused by name before its plan record, the
    /// message's total the same ten positions (`draft_set_is_refused_by_name`).
    /// FAIL-first: a seat that judges the window on the total `--ctx-size`
    /// (eight, past five) keeps the unset draft and refuses it at the context
    /// the rule chose, so the first clause's server exits with the refusal
    /// before its load; a refusal whose total is one window short (the
    /// `VERIFY_ROWS` a slot) turns the second clause red; a seat whose check
    /// passes a set draft loads it, so its server prints a `plan` record and
    /// no refusal. The fixture tier defers the clause by name: the draft file
    /// is the real file's.
    fn draft_slot_share(dir: &Path) -> Result<bool, GateError> {
        // The draft is the real file's: its draft file sits beside the target,
        // which a fixture's directory is not known to hold.
        if !tier::run_clause("draft_slot_share", Tag::FileBound)? {
            return Ok(true);
        }
        const FLAGS: [&str; 4] = ["--ctx-size", "8", "--parallel", "2"];
        let own = dir.join("draft-share");
        std::fs::create_dir_all(&own)?;
        let err_log = own.join("server.err");
        let positions = (<Body38 as MtpBody>::VERIFY_ROWS + 1) * 2;
        let mut ok = true;

        // The set draft: refused before the plan record and any load.
        let mut cmd = Command::new(Served38::exe()?);
        cmd.env(bloomery_levers::DRAFT, "mtp")
            .env(bloomery_levers::RESIDENCY, "off")
            .env_remove(bloomery_levers::MTP_HEAD_ROWS)
            .env_remove(bloomery_levers::MTP_DRAFT)
            .env_remove(bloomery_levers::MTP_WIDTH);
        let args: Vec<&str> = DEFAULT_ARGS.iter().chain(&FLAGS).copied().collect();
        let mut served = Served38::spawn_with(&args, &own, &mut cmd)?;
        let mut status = None;
        for _ in 0..POLLS {
            status = served.child.try_wait()?;
            let text = std::fs::read_to_string(&err_log)?;
            let planned = matches!(
                record::Log::of(&text, record::BLOOMERY_SERVE_QWEN38).first(&record::PLAN38),
                Ok(Some(_))
            );
            if status.is_some() || planned {
                break;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
        let text = std::fs::read_to_string(&err_log)?;
        println!("draft-share: set draft: exit {status:?}");
        if status.is_none() {
            println!(
                "draft-share: the set-draft server stopped: {}",
                served.stop()?
            );
        }
        let want = format!(
            "a slot's context of 4 leaves no positions for the MTP draft's window; --ctx-size \
             names a total of at least {positions}"
        );
        let planned = record::Log::of(&text, record::BLOOMERY_SERVE_QWEN38)
            .first(&record::PLAN38)?
            .is_some();
        if !text.contains(&want) {
            println!("draft-share: the set-draft server's last lines:");
            for l in text.lines().rev().take(20).collect::<Vec<_>>().iter().rev() {
                println!("  {l}");
            }
        }
        check(
            &mut ok,
            "draft_set_is_refused_by_name",
            status.is_some_and(|s| !s.success()) && text.contains(&want) && !planned,
        );

        // The unset draft: off, and the load's record names why.
        let mut cmd = Command::new(Served38::exe()?);
        cmd.env(bloomery_levers::RESIDENCY, "off")
            .env_remove(bloomery_levers::DRAFT)
            .env_remove(bloomery_levers::MTP_HEAD_ROWS)
            .env_remove(bloomery_levers::MTP_DRAFT)
            .env_remove(bloomery_levers::MTP_WIDTH);
        let args: Vec<&str> = DEFAULT_ARGS[..4].iter().chain(&FLAGS).copied().collect();
        let mut served = Served38::spawn_with(&args, &own, &mut cmd)?;
        let mut off = None;
        for _ in 0..POLLS {
            let text = std::fs::read_to_string(&err_log)?;
            off = record::Log::of(&text, record::BLOOMERY_SERVE_QWEN38)
                .first(&record::LOAD_DRAFT_OFF38)
                .ok()
                .flatten()
                .map(|f| f.text("why").map(str::to_owned))
                .transpose()?;
            if off.is_some() || served.child.try_wait()?.is_some() {
                break;
            }
            std::thread::sleep(POLL);
        }
        let text = std::fs::read_to_string(&err_log)?;
        if served.child.try_wait()?.is_none() {
            println!(
                "draft-share: the unset-draft server stopped: {}",
                served.stop()?
            );
        }
        let split = parallel_line(&err_log)?;
        println!(
            "draft-share: unset draft: {}; load draft=off ({})",
            split.as_deref().unwrap_or("no parallel line"),
            off.as_deref().unwrap_or("none")
        );
        if off.is_none() {
            for l in text.lines().rev().take(20).collect::<Vec<_>>().iter().rev() {
                println!("  {l}");
            }
        }
        let why = bloomery_levers::Draft38Off::Ctx {
            need: positions,
            ctx: 8,
        }
        .to_string();
        let two_slots = split
            .as_deref()
            .is_some_and(|l| l.contains(" slots=2 ") && l.contains(" slot_ctx=4 "));
        check(
            &mut ok,
            "draft_unset_yields_to_the_slot_share",
            two_slots
                && off.as_deref() == Some(why.as_str())
                && !text.contains("leaves no positions for the MTP draft's window")
                && !text.lines().any(|l| l.starts_with("draft keep ")),
        );
        Ok(ok)
    }

    /// The NVMe tier's one-column rule (the seat's module doc,
    /// `bloomery_levers::paged_columns`) on the gate plan at two rooms
    /// (`BLOOMERY_HOST_ROOM`, the split dial's default arena), each server
    /// stopped at its `parallel` line, before its load, or refused before it
    /// prints one. At 27 GiB the plan pages its host experts through the
    /// tier's RAM arena (`bloomery_placement`'s
    /// `the_one_column_rule_reads_the_arena` derives both rooms on the Mac):
    /// the flagless server serves one slot (`from=paged`), its `paged` line
    /// naming the arena and the two slots asked; `--parallel 2` and
    /// `BLOOMERY_DRAFT=mtp` are each refused by name (`PagedRefused`'s own
    /// text at the arena the refusal names, its own plan's). At 40 GiB the tier
    /// pages experts with no arena (the mapping path), and `--parallel 2`
    /// serves its two slots (`from=flag`, no `paged` line). At 27 GiB too,
    /// the checkpoints of the resident sequences sit outside the room the
    /// split fills (`paged_checkpoints_sit_outside_the_arena`: the `cache`
    /// line's arena (`tier`) is the gate's own plan's, which holds no
    /// checkpoint reserve, less `checkpoint_bytes(slots)`, and its `need` that
    /// plan's plus the same); a residency word set to `off` reaches the
    /// `plan` record (`paged_set_off_residency_is_accepted`), and one set to
    /// `mid-p0-s1` is refused by name before it
    /// (`paged_set_residency_is_refused_by_name`: the machine's fills share
    /// the arena with the step). The floor counts the checkpoints too: at the
    /// middle of the band of rooms between the one-slot plan's floor and the
    /// asked two slots' (the gate's own plans' floors, read off
    /// `HostRoomFloor`, plus `checkpoint_bytes(slots)`) the flagless server
    /// serves one slot, `from=paged`, its `paged` line naming the floor, with
    /// the arena the gate's own plan has less `checkpoint_bytes(1)`
    /// (`paged_floor_fallback_serves_one_column`), and `--parallel 2` is
    /// refused by the floor, by name
    /// (`paged_floor_set_parallel_is_refused_by_the_floor`). Last, the small
    /// card's case (below, at its server). The fixture tier defers the clause by name: its rooms and
    /// budget are the real file's shapes.
    /// FAIL-first: the seat before the rule serves the default's two slots
    /// at 27 GiB and refuses neither; a rule that reads the paged experts'
    /// bytes rather than the arena refuses the 40 GiB `--parallel 2`; a
    /// one-column load that keeps the yield's reason names it on the small
    /// card; a machine with no checkpoint reserve gives the arena the gate's
    /// own plan has, red on the checkpoint clause; a seat that takes a set
    /// word on the paged plan prints its `plan` record, red on the refusal
    /// clause; one that refuses `off` as well never reaches that record, red
    /// on the acceptance clause; a seat that plans the asked two slots with
    /// no floor fallback refuses the flagless band server by the floor, red
    /// on the fallback clause.
    fn paged(
        dir: &Path,
        levers: &bloomery_levers::Levers,
        prompt: &str,
    ) -> Result<bool, GateError> {
        // The rooms and the budget are the real file's shapes: a fixture's
        // host set pages nothing at them.
        if !tier::run_clause("paged", Tag::Scale)? {
            return Ok(true);
        }
        let own = dir.join("paged");
        std::fs::create_dir_all(&own)?;
        let err_log = own.join("server.err");
        // One server under `env` (the room, and the draft where it is set)
        // to its `parallel` line or its exit: the line, whether it exited
        // unsuccessfully, and its log.
        let run = |env: &[(&str, &str)],
                   flags: &[&str]|
         -> Result<(Option<String>, bool, String), GateError> {
            let mut cmd = Command::new(Served38::exe()?);
            cmd.env(bloomery_levers::RESIDENCY, "off")
                .env_remove(bloomery_levers::DRAFT)
                .env_remove(bloomery_levers::NVTIER_BYTES)
                .env_remove(bloomery_levers::MTP_HEAD_ROWS)
                .env_remove(bloomery_levers::MTP_DRAFT)
                .env_remove(bloomery_levers::MTP_WIDTH)
                .envs(env.iter().copied());
            let args: Vec<&str> = DEFAULT_ARGS.iter().chain(flags).copied().collect();
            let mut served = Served38::spawn_with(&args, &own, &mut cmd)?;
            let mut status = None;
            for _ in 0..POLLS {
                status = served.child.try_wait()?;
                // The `plan` record follows the `parallel` line before the
                // load: both are read before the server stops.
                let text = std::fs::read_to_string(&err_log)?;
                let plan = matches!(
                    record::Log::of(&text, record::BLOOMERY_SERVE_QWEN38).first(&record::PLAN38),
                    Ok(Some(_))
                );
                if status.is_some() || (parallel_line(&err_log)?.is_some() && plan) {
                    break;
                }
                std::thread::sleep(Duration::from_secs(1));
            }
            let line = parallel_line(&err_log)?;
            let text = std::fs::read_to_string(&err_log)?;
            let failed = status.is_some_and(|s| !s.success());
            println!(
                "paged: {env:?} {flags:?}: exit {status:?}, {}",
                line.as_deref().unwrap_or("no parallel line")
            );
            if status.is_none() {
                println!("paged: the server stopped: {}", served.stop()?);
            }
            Ok((line, failed, text))
        };
        let field = |line: &str, k: &str| -> Option<String> {
            line.split_whitespace()
                .find_map(|w| w.strip_prefix(k)?.strip_prefix('='))
                .map(str::to_owned)
        };
        let paged_line = |text: &str| {
            text.lines()
                .find(|l| l.starts_with("paged "))
                .map(str::to_owned)
        };
        let listened = |text: &str| -> Result<bool, GateError> {
            Ok(record::Log::of(text, record::BLOOMERY_SERVE_QWEN38)
                .first(&record::LISTENING38)?
                .is_some())
        };
        let mut ok = true;

        let (line, _, text) = run(&[(bloomery_levers::HOST_ROOM, "27G")], &[])?;
        let paged = paged_line(&text);
        println!("paged: {}", paged.as_deref().unwrap_or("no paged line"));
        let arena = paged
            .as_deref()
            .and_then(|l| field(l, "arena"))
            .and_then(|v| v.parse::<u64>().ok());
        let one = line.as_deref().is_some_and(|l| {
            let n = |k: &str| field(l, k).and_then(|v| v.parse::<usize>().ok());
            n("slots") == Some(1)
                && field(l, "from").as_deref() == Some("paged")
                && n("slot_ctx").is_some_and(|c| c > 0 && n("total") == Some(c))
        });
        // At `--place gate` the unset draft is off by the placement, a reason
        // that holds whatever the plan: it was not asked, and stays.
        let named = paged.as_deref().is_some_and(|l| {
            field(l, "rule").as_deref() == Some("one-column")
                && field(l, "asked_slots").as_deref() == Some("2")
                && field(l, "asked_draft").as_deref() == Some("off")
                && field(l, "slots").as_deref() == Some("1")
                && field(l, "draft").as_deref() == Some("off")
        });
        check(
            &mut ok,
            "paged_default_serves_one_column",
            one && named && arena.is_some_and(|a| a > 0),
        );
        // A set residency word on a plan that pages through the arena is
        // refused by name (`paged_set_residency_is_refused_by_name`
        // below); `off`, set on every server of this clause, is not: the
        // 27 GiB server reaches its `plan` record.
        let refusal = |word: &str| {
            format!(
                "BLOOMERY_RESIDENCY={word}: the plan pages host experts through the NVMe \
                 tier's "
            )
        };
        let planned = record::Log::of(&text, record::BLOOMERY_SERVE_QWEN38)
            .first(&record::PLAN38)?
            .is_some();
        check(
            &mut ok,
            "paged_set_off_residency_is_accepted",
            planned && !text.contains(&refusal("off")),
        );
        let free = record::Log::of(&text, record::BLOOMERY_SERVE_QWEN38)
            .first(&record::PLAN38)?
            .ok_or("the 27 GiB server printed no `plan` record")?
            .u64("card_free")?;

        // The checkpoints every resident sequence may pin sit outside the
        // room the split fills: the host reserves them, so the plan's need
        // counts them and the arena is the room less the floor they raise.
        // The gate's own plan at the same room, slots and context, whose
        // machine holds no such reserve, differs from the server's by exactly
        // `checkpoint_bytes(slots)` in each — the split keeps the same host
        // experts (its kept bytes are the need less the room past the
        // arena, both moved by the reserve) — read off the `cache` line's
        // `need` and `tier`, the seat's own terms of the plan it loads. The
        // need holds the arena (`HostNeed::bytes`), so the reserve's move is
        // in the need beside the arena (`need − tier`); the need itself, the
        // room the split fills, stays where it was.
        let cache = text
            .lines()
            .find(|l| l.starts_with("cache "))
            .map(str::to_owned);
        let term = |l: Option<&str>, k: &str| -> Option<u64> {
            l.and_then(|l| field(l, k)).and_then(|v| v.parse().ok())
        };
        let shape = term(line.as_deref(), "slots")
            .zip(term(line.as_deref(), "slot_ctx"))
            .map(|(slots, ctx)| {
                Ok::<_, GateError>((usize::try_from(ctx)?, usize::try_from(slots)?))
            })
            .transpose()?;
        let held =
            bloomery_gpu_gates::residency38::checkpoint_bytes(shape.map_or(1, |(_, slots)| slots));
        let seat = (
            term(cache.as_deref(), "tier"),
            term(cache.as_deref(), "need"),
        );
        let bare = shape
            .map(|shape| host_split_at(levers, shape, free, 27 << 30))
            .transpose()?;
        println!(
            "paged: the seat's arena {:?} and need {:?}, the gate's plan with no checkpoint \
             reserve {bare:?}, the checkpoints {held} B",
            seat.0, seat.1
        );
        check(
            &mut ok,
            "paged_checkpoints_sit_outside_the_arena",
            match (seat, bare) {
                ((Some(arena), Some(need)), Some((bare_arena, bare_need))) => {
                    arena > 0
                        && arena + held == bare_arena
                        && need.checked_sub(arena)
                            == bare_need.checked_sub(bare_arena).map(|b| b + held)
                }
                _ => false,
            },
        );

        // A refusal's arena, read off its own line (the plan the refused
        // server read, whose context rides the card's free bytes at its
        // start), and whether the line is `PagedRefused`'s text at it.
        let refused = |text: &str, head: &str, parallel: Option<usize>, draft| {
            let arena = text.lines().find_map(|l| {
                let tail = &l[l.find(head)? + head.len()..];
                tail.split_once(" B RAM arena")?.0.parse::<u64>().ok()
            });
            let named = arena.is_some_and(|arena| {
                let want = bloomery_levers::PagedRefused {
                    arena,
                    slots: parallel.map(|n| (bloomery_levers::SlotsBy::Parallel, n)),
                    draft,
                }
                .to_string();
                arena > 0 && text.contains(&want)
            });
            println!("paged: refused at arena {arena:?}, by name {named}");
            named
        };
        let tier = ": the plan pages host experts through the NVMe tier's ";

        let (line, failed, text) =
            run(&[(bloomery_levers::HOST_ROOM, "27G")], &["--parallel", "2"])?;
        check(
            &mut ok,
            "paged_parallel_is_refused_by_name",
            failed
                && line.is_none()
                && refused(&text, &format!("--parallel 2{tier}"), Some(2), false)
                && !listened(&text)?,
        );

        let (line, failed, text) = run(
            &[
                (bloomery_levers::HOST_ROOM, "27G"),
                (bloomery_levers::DRAFT, "mtp"),
            ],
            &[],
        )?;
        check(
            &mut ok,
            "paged_draft_is_refused_by_name",
            failed
                && line.is_none()
                && refused(&text, &format!("BLOOMERY_DRAFT=mtp{tier}"), None, true)
                && !listened(&text)?,
        );

        // The residency machine's fills share the arena with the step: a
        // set word other than `off` on a plan that pages through the arena
        // is refused by name, after the slot rule's lines and before the
        // plan's record.
        let (line, failed, text) = run(
            &[
                (bloomery_levers::HOST_ROOM, "27G"),
                (bloomery_levers::RESIDENCY, RESIDENCY_WORD),
            ],
            &[],
        )?;
        check(
            &mut ok,
            "paged_set_residency_is_refused_by_name",
            failed
                && line.is_some()
                && text.contains(&refusal(RESIDENCY_WORD))
                && !record::Log::of(&text, record::BLOOMERY_SERVE_QWEN38)
                    .first(&record::PLAN38)?
                    .is_some()
                && !listened(&text)?,
        );

        let (line, _, text) = run(&[(bloomery_levers::HOST_ROOM, "40G")], &["--parallel", "2"])?;
        let asked = line.as_deref().is_some_and(|l| {
            field(l, "slots").as_deref() == Some("2") && field(l, "from").as_deref() == Some("flag")
        });
        check(
            &mut ok,
            "unpaged_parallel_serves_as_asked",
            asked && paged_line(&text).is_none(),
        );

        // The default count's floor fallback. The floor counts every slot's
        // checkpoints: the gate's own floors (the planner's, read off
        // `HostRoomFloor` at a room of one byte, with no checkpoint reserve)
        // plus `checkpoint_bytes(slots)` are the server's, so the one-slot
        // plan's floor sits `checkpoint_bytes(1)` above the two-slot plan's
        // less its own extra checkpoints — a band of rooms the one-slot plan
        // clears and the two the flagless server asks first does not. The
        // server's first plan, at one position (the context search's first),
        // is the asked slots', so the band is [one slot's floor, two slots'
        // floor at one position); its middle is the room. A flagless server
        // there serves one slot (`from=paged`, its `paged` line naming the
        // floor), and `--parallel 2` is refused by the floor, by name (the
        // context search's first plan fails before the `paged` rule).
        let held = bloomery_gpu_gates::residency38::checkpoint_bytes(1);
        let one_floor = host_floor(levers, (CTX, 1), free)? + held;
        let two_floor = host_floor(levers, (1, 2), free)? + 2 * held;
        let band = one_floor < two_floor;
        let room = one_floor + two_floor.saturating_sub(one_floor) / 2;
        println!(
            "paged: floors with the checkpoints: one slot at {CTX} positions {one_floor} B, two \
             slots at one position {two_floor} B; the band's middle {room} B"
        );
        check(&mut ok, "paged_floor_band_is_between_the_slot_floors", band);
        let room_arg = room.to_string();
        let (line, failed, text) = run(&[(bloomery_levers::HOST_ROOM, &room_arg)], &[])?;
        let paged = paged_line(&text);
        println!("paged: {}", paged.as_deref().unwrap_or("no paged line"));
        if paged.is_none() {
            println!("paged: the flagless server's last lines:");
            for l in text.lines().rev().take(12).collect::<Vec<_>>().iter().rev() {
                println!("  {l}");
            }
        }
        let cache = text
            .lines()
            .find(|l| l.starts_with("cache "))
            .map(str::to_owned);
        let shape = term(line.as_deref(), "slots")
            .zip(term(line.as_deref(), "slot_ctx"))
            .map(|(slots, ctx)| {
                Ok::<_, GateError>((usize::try_from(ctx)?, usize::try_from(slots)?))
            })
            .transpose()?;
        let arena = term(cache.as_deref(), "tier");
        let bare = shape
            .map(|shape| host_split_at(levers, shape, free, room))
            .transpose()?;
        let one = line.as_deref().is_some_and(|l| {
            field(l, "slots").as_deref() == Some("1")
                && field(l, "from").as_deref() == Some("paged")
        });
        let named = paged.as_deref().is_some_and(|l| {
            field(l, "rule").as_deref() == Some("one-column")
                && field(l, "asked_slots").as_deref() == Some("2")
                && field(l, "slots").as_deref() == Some("1")
                && l.ends_with("; the asked slots' plan leaves the room under the tier's floor)")
        });
        println!("paged: the seat's arena {arena:?}, the gate's plan with no reserve {bare:?}");
        check(
            &mut ok,
            "paged_floor_fallback_serves_one_column",
            !failed
                && one
                && named
                && arena.zip(bare).is_some_and(|(arena, (bare_arena, _))| {
                    arena > 0 && arena + held == bare_arena
                }),
        );
        let (line, failed, text) = run(
            &[(bloomery_levers::HOST_ROOM, &room_arg)],
            &["--parallel", "2"],
        )?;
        check(
            &mut ok,
            "paged_floor_set_parallel_is_refused_by_the_floor",
            failed
                && line.is_none()
                && text.contains(&format!(
                    "the host room {room} B is under the NVMe expert tier's floor"
                ))
                && !listened(&text)?,
        );

        // The small card's case: a flagless server — no `--place`, so the
        // box pin's one-card view runs `a` on the 3090, as
        // `slots_default_residency`'s does — under the budget whose drafted
        // two-slot plan holds half of [`CTX`] a slot ([`small_card_budget`]),
        // at 27 GiB. The unset draft yields to the context on that plan, the
        // plan pages, and the load runs one slot with no draft: its `load
        // draft=off` record names the paged rule at the `paged` line's own
        // arena, never the yield judged on the plan it abandoned, and the
        // `paged` line names the draft as asked before that yield. The
        // residency is pinned off: the clause holds the draft, and a paged
        // plan's residency is its own rule's. Stopped at that record.
        let budget = small_card_budget(levers, free)?;
        let budget_bytes = budget.to_string();
        let mut cmd = Command::new(Served38::exe()?);
        cmd.env(bloomery_levers::HOST_ROOM, "27G")
            .env(bloomery_levers::CARD_BUDGET, &budget_bytes)
            .env(bloomery_levers::RESIDENCY, "off")
            .env_remove(bloomery_levers::DRAFT)
            .env_remove(bloomery_levers::NVTIER_BYTES)
            .env_remove(bloomery_levers::MTP_HEAD_ROWS)
            .env_remove(bloomery_levers::MTP_DRAFT)
            .env_remove(bloomery_levers::MTP_WIDTH);
        let mut served = Served38::spawn_with(&DEFAULT_ARGS[..4], &own, &mut cmd)?;
        let mut off = None;
        for _ in 0..POLLS {
            let text = std::fs::read_to_string(&err_log)?;
            off = record::Log::of(&text, record::BLOOMERY_SERVE_QWEN38)
                .first(&record::LOAD_DRAFT_OFF38)
                .ok()
                .flatten()
                .map(|f| f.text("why").map(str::to_owned))
                .transpose()?;
            if off.is_some() || served.child.try_wait()?.is_some() {
                break;
            }
            std::thread::sleep(POLL);
        }
        let text = std::fs::read_to_string(&err_log)?;
        if served.child.try_wait()?.is_none() {
            println!("paged: the small-card server stopped: {}", served.stop()?);
        }
        let log = record::Log::of(&text, record::BLOOMERY_SERVE_QWEN38);
        let placed = log.first(&record::PLACE_UNSET)?;
        let one_card = match &placed {
            Some(p) => p.word("place")? == "a" && p.text("why")? == "one card",
            None => false,
        };
        let with = log
            .first(&record::DRAFT_YIELD)?
            .map(|y| y.u64("with"))
            .transpose()?;
        let yielded = with.is_some();
        if off.is_none() {
            let tail: Vec<&str> = text.lines().rev().take(20).collect();
            println!(
                "paged: the small-card server printed no `load draft=off` record; its last lines:"
            );
            for l in tail.iter().rev() {
                println!("  {l}");
            }
        }
        let paged = paged_line(&text);
        let arena = paged
            .as_deref()
            .and_then(|l| field(l, "arena"))
            .and_then(|v| v.parse::<u64>().ok());
        println!(
            "paged: small card, budget {budget} B: one card {one_card}, draft yield with={with:?} \
             (the gate's plan: the drafted fit reaches {} a slot), {}; load draft=off ({})",
            CTX / 2,
            paged.as_deref().unwrap_or("no paged line"),
            off.as_deref().unwrap_or("none")
        );
        // The premise: the box's one-card view, and the yield judged on the
        // two-slot drafted plan.
        check(
            &mut ok,
            "paged_small_card_yields_on_one_card",
            one_card && yielded,
        );
        let line = parallel_line(&err_log)?;
        let one = line.as_deref().is_some_and(|l| {
            field(l, "slots").as_deref() == Some("1")
                && field(l, "from").as_deref() == Some("paged")
        });
        let asked = paged.as_deref().is_some_and(|l| {
            field(l, "asked_slots").as_deref() == Some("2")
                && field(l, "asked_draft").as_deref() == Some("on")
                && field(l, "draft").as_deref() == Some("off")
        });
        let why = arena.map(|arena| bloomery_levers::Draft38Off::Paged { arena }.to_string());
        check(
            &mut ok,
            "paged_small_card_load_names_the_rule",
            one && asked && arena.is_some_and(|a| a > 0) && why.is_some() && off == why,
        );

        // `generate_qwen3moe` under the same rule: with no `--place` (the
        // box pin's one card, `a`) at 27 GiB, the unset draft runs at `a`
        // with its file there, its plan pages, and the run drafts nothing —
        // its `load draft=off` record names the paged rule at the arena it
        // names, and no `load draft=mtp` line prints. The residency is pinned
        // off, as at the small card. It runs whole: `-n 2` past its load.
        let mut cmd = Command::new(std::env::current_exe()?.with_file_name("generate_qwen3moe"));
        cmd.env(bloomery_levers::HOST_ROOM, "27G")
            .env(bloomery_levers::RESIDENCY, "off")
            .env_remove(bloomery_levers::DRAFT)
            .env_remove(bloomery_levers::NVTIER_BYTES)
            .env_remove(bloomery_levers::MTP_HEAD_ROWS)
            .env_remove(bloomery_levers::MTP_DRAFT)
            .env_remove(bloomery_levers::MTP_WIDTH);
        let mut run = Served38::spawn_with(&["--prompt", prompt, "-n", "2"], &own, &mut cmd)?;
        let mut status = None;
        for _ in 0..POLLS {
            status = run.child.try_wait()?;
            if status.is_some() {
                break;
            }
            std::thread::sleep(POLL);
        }
        if status.is_none() {
            println!(
                "paged: generate_qwen3moe stopped at the bound: {}",
                run.stop()?
            );
        }
        let out = std::fs::read_to_string(own.join("server.out"))?;
        let head = "load draft=off (unset: the plan pages host experts through the NVMe tier's ";
        let arena = out.lines().find_map(|l| {
            l.strip_prefix(head)?
                .split_once(" B RAM arena")?
                .0
                .parse::<u64>()
                .ok()
        });
        let off = record::Log::of(&out, record::GENERATE_QWEN3MOE)
            .first(&record::LOAD_DRAFT_OFF38)?
            .map(|f| f.text("why").map(str::to_owned))
            .transpose()?;
        let drafted = out.lines().any(|l| l.starts_with("load draft=mtp"));
        println!(
            "paged: generate_qwen3moe exit {status:?}: load draft=off ({}), load draft=mtp {drafted}",
            off.as_deref().unwrap_or("none")
        );
        if off.is_none() {
            let err = std::fs::read_to_string(own.join("server.err")).unwrap_or_default();
            let tail: Vec<&str> = err.lines().rev().take(20).collect();
            println!("paged: generate_qwen3moe's last stderr lines:");
            for l in tail.iter().rev() {
                println!("  {l}");
            }
        }
        let why = arena.map(|arena| bloomery_levers::Draft38Off::Paged { arena }.to_string());
        check(
            &mut ok,
            "paged_cli_drafts_nothing_by_the_rule",
            status.is_some_and(|s| s.success())
                && arena.is_some_and(|a| a > 0)
                && why.is_some()
                && off == why
                && !drafted,
        );
        Ok(ok)
    }

    /// The card budget under which the file's drafted two-slot plan on the
    /// 3090 (its free bytes `free`, the seat's census reading) holds half of
    /// [`CTX`] a slot, to a MiB: the least under which the plan at `CTX / 2`
    /// a slot holds, so the drafted fit reaches it. The seat's unset draft
    /// yields to the context there — the drafted fit under [`CTX`], the plain
    /// one past it — as on a 12 GB card. The plan is the small-card server's:
    /// the unset head and the draft file beside the target (that server sets
    /// neither lever), the census's 3090 as `a` resolves it
    /// (`devices::spec_of_device`: the known spec with the census's free
    /// bytes) and no draft reserve in the machine, which a one-card plan
    /// counts in `plan_mtp_with_slots`.
    fn small_card_budget(levers: &bloomery_levers::Levers, free: u64) -> Result<u64, GateError> {
        const MIB: u64 = 1 << 20;
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::describe(&split)?;
        let experts = experts38(levers)?;
        let (draft_path, from) = draft_file(None, &path);
        let draft = Split::open(&draft_path).map_err(|e| {
            format!(
                "open the MTP draft {} ({}): {e}",
                draft_path.display(),
                from.describe()
            )
        })?;
        let rows = head_rows_of(None, &split, inputs.spec.vocab)?.rows;
        let mtp = MtpInputs::read(&draft, &split, &inputs, rows)?;
        let holds = |budget: u64, c: usize| -> Result<bool, GateError> {
            let mut machine = machine_for_experts(
                RTX_3090,
                inputs.spec.layers.len(),
                u64::try_from(ubatch_for(c)?)?,
                experts,
            );
            machine.cards[0].free_bytes = Some(free);
            let plan_levers = PlanLevers {
                card_budget_bytes: Some(budget),
            };
            Ok(inputs
                .plan_mtp_with_slots(&machine, u64::try_from(c)?, &plan_levers, &mtp, experts, 2)
                .is_ok())
        };
        let want = CTX / 2;
        let (mut lo, mut hi) = (1, free / MIB);
        if !holds(hi * MIB, want)? {
            return Err(format!(
                "the 3090's {free} free bytes hold no drafted two-slot plan at {want} positions a \
                 slot: no budget makes the small card's case"
            )
            .into());
        }
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if holds(mid * MIB, want)? {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let budget = hi * MIB;
        let cap = usize::try_from(serve_ctx(1, &inputs.hp)?)?;
        let fit = model::placement::ctx::largest(want, cap, |c| holds(budget, c))?;
        println!("paged: the small card's budget {budget} B (drafted two-slot fit {fit} a slot)");
        Ok(budget)
    }

    /// The plain arm (the module header): the main server with the draft
    /// off — every clause that needs no draft — then, one at a time, the
    /// clauses' own servers: `ctx`, the slots pair and the sampled rounds.
    fn plain(
        dir: &Path,
        levers: &bloomery_levers::Levers,
        prompt: &str,
        reference: &[u32],
    ) -> Result<bool, GateError> {
        let exe = Served38::exe()?;
        let err_log = dir.join("server.err");
        let mut cmd = Command::new(&exe);
        cmd.env(bloomery_levers::DRAFT, "off")
            .env(bloomery_levers::RESIDENCY, "off")
            .env_remove(bloomery_levers::MTP_WIDTH);
        let mut served = Served38::spawn_with(&SERVER_ARGS, dir, &mut cmd)?;
        println!("server pid {}", served.child.id());
        let addr = served.address(&err_log, POLLS, POLL)?;
        let url = |p: &str| format!("http://{addr}{p}");
        println!("server listening on {addr}");

        let mut ok = true;
        let (st, body) = curl(&url("/health"), None, false)?;
        println!("health {st} {body}");
        check(&mut ok, "health_ok", st == 200 && body.contains("\"ok\""));
        // The seat's `plan` line names its stage card's free bytes at plan
        // time, at most the device's usable bytes — the census term the
        // expert rule filled within (memguard). FAIL-first: a plan line that
        // drops it, or names it past the usable bytes, turns this red.
        let plan = seat_log(&err_log)?.one(&record::PLAN38)?;
        let free = plan.opt_u64("card_free")?;
        let usable = stage_usable(&plan)?;
        let free_named = free.is_some_and(|f| f <= usable);
        println!(
            "plan names the stage card's free bytes {free_named}: card_free={free:?}, usable \
             {usable}"
        );
        check(&mut ok, "plan_names_the_cards_free_bytes", free_named);
        check(
            &mut ok,
            "xstream_line_is_the_unset_rules",
            xstream_agrees(&err_log, false)?,
        );
        let argv: Vec<String> = std::iter::once(exe.to_string_lossy().into_owned())
            .chain(SERVER_ARGS.iter().map(|s| (*s).to_owned()))
            .collect();
        ok &= props_engine(&url, &argv, served.child.id(), levers, &err_log, false)?;

        let completion = json!({
            "prompt": prompt, "n_predict": N_PREDICT, "temperature": 0, "return_tokens": true,
        });
        let (st, body) = curl(&url("/completion"), Some(&completion), false)?;
        let c1 = json_of("/completion", st, &body)?;
        let first = ids_of(&c1["tokens"]);
        let stop = c1["stop_type"].as_str().unwrap_or("").to_owned();
        println!(
            "completion tokens {first:?} stop_type={stop} content={}",
            c1["content"]
        );
        println!("generate_qwen3moe {reference:?}");
        check(
            &mut ok,
            "completion_ids_are_generate_qwen3moe",
            agree(&first, &stop, reference),
        );

        // One chat turn, so the slot holds another conversation than the
        // completion's before it is resent.
        let chat = json!({
            "messages": [{"role": "user", "content": CHAT}],
            "temperature": 0, "max_tokens": CHAT_PREDICT,
        });
        let (st, body) = curl(&url("/v1/chat/completions"), Some(&chat), false)?;
        let plain = json_of("/v1/chat/completions", st, &body)?;
        println!(
            "chat content={:?} finish_reason={} usage={}",
            plain["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or(""),
            plain["choices"][0]["finish_reason"],
            plain["usage"]
        );

        let (st, body) = curl(&url("/completion"), Some(&completion), false)?;
        let c3 = json_of("/completion", st, &body)?;
        println!(
            "completion after the other requests {:?} cache_n={}",
            ids_of(&c3["tokens"]),
            c3["timings"]["cache_n"]
        );
        check(
            &mut ok,
            "completion_after_other_requests_identical",
            ids_of(&c3["tokens"]) == first,
        );

        ok &= props_ctx(&url)?;
        ok &= cache(&url, &err_log, false, "plain")?;

        println!("server stopped: {}", served.stop()?);
        ok &= ctx(dir, levers)?;
        ok &= paged(dir, levers, prompt)?;
        ok &= draft_slot_share(dir)?;
        ok &= residency_loads_the_word(dir)?;
        // The resident slots together (the module header): its own servers,
        // the first with the draft on, the second the seat's own defaults.
        ok &= slots_flow_together(dir)?;
        // The width chooser's server (the module header): the draft on, the
        // residency off, `BLOOMERY_MTP_WIDTH=cost`.
        ok &= width_cost(dir, prompt, reference)?;
        // The plain rounds of a load that drafts nothing run as one pass
        // (the module header): its own server, sampled requests that step,
        // the round records on.
        ok &= slots_sampled_rounds(dir)?;
        Ok(ok)
    }

    /// The drafted arm (the module header): the main server of
    /// [`SERVER_ARGS`] under `BLOOMERY_DRAFT=mtp`, `BLOOMERY_RESIDENCY=off`
    /// and the fixed window, running the main arm's drafted clauses alone —
    /// `/props`' `engine` against the gate's own plan with the draft (its
    /// card bytes the draft's among them, its `draft` naming the file
    /// [`draft_file`] picks), the greedy pass's ids and its draft counts,
    /// `continued`, the sampled requests and, after them, the `cache`
    /// clause's drafted half — none of the plain arm's other clauses: each
    /// runs there on the plain main server.
    fn drafted_main(
        dir: &Path,
        levers: &bloomery_levers::Levers,
        prompt: &str,
        ids: &[u32],
        reference: &[u32],
    ) -> Result<bool, GateError> {
        let err_log = dir.join("server.err");
        let exe = Served38::exe()?;
        let mut cmd = Command::new(&exe);
        // The fixed window (the module header): this server's drafted
        // clauses hold counts across requests of it.
        cmd.env(bloomery_levers::DRAFT, "mtp")
            .env(bloomery_levers::RESIDENCY, "off")
            .env(bloomery_levers::MTP_WIDTH, "fixed");
        let mut served = Served38::spawn_with(&SERVER_ARGS, dir, &mut cmd)?;
        println!("server pid {}", served.child.id());
        let addr = served.address(&err_log, POLLS, POLL)?;
        let url = |p: &str| format!("http://{addr}{p}");
        println!("server listening on {addr}");

        let mut ok = true;
        let (st, body) = curl(&url("/health"), None, false)?;
        println!("health {st} {body}");
        let argv: Vec<String> = std::iter::once(exe.to_string_lossy().into_owned())
            .chain(SERVER_ARGS.iter().map(|s| (*s).to_owned()))
            .collect();
        ok &= props_engine(&url, &argv, served.child.id(), levers, &err_log, true)?;
        let completion = json!({
            "prompt": prompt, "n_predict": N_PREDICT, "temperature": 0, "return_tokens": true,
        });
        let (st, body) = curl(&url("/completion"), Some(&completion), false)?;
        let c1 = json_of("/completion", st, &body)?;
        let first = ids_of(&c1["tokens"]);
        let stop = c1["stop_type"].as_str().unwrap_or("").to_owned();
        println!(
            "completion tokens {first:?} stop_type={stop} content={}",
            c1["content"]
        );
        println!("generate_qwen3moe {reference:?}");
        // The drafted server's own clause: the greedy ids are the plain
        // run's (a draft changes which passes run, never a token), and the
        // pass carried its counts.
        let d = &c1["timings"];
        check(
            &mut ok,
            "drafted_ids_are_the_plain_runs",
            agree(&first, &stop, reference),
        );
        check(
            &mut ok,
            "drafted_timings_carry_the_draft_counts",
            d["draft_n"].as_u64().is_some_and(|n| n > 0)
                && d["draft_n_accepted"].as_u64().is_some_and(|n| n > 0),
        );
        ok &= continued(&url, &err_log, ids, reference)?;
        let sampled = json!({
            "prompt": prompt, "n_predict": N_PREDICT, "temperature": SAMPLED_TEMPERATURE,
            "seed": SAMPLED_SEED, "return_tokens": true,
        });
        let (st, body) = curl(&url("/completion"), Some(&sampled), false)?;
        let top1 = json!({
            "prompt": prompt, "n_predict": N_PREDICT, "temperature": SAMPLED_TEMPERATURE,
            "top_k": 1, "seed": SAMPLED_SEED, "return_tokens": true,
        });
        let (st1, body1) = curl(&url("/completion"), Some(&top1), false)?;
        ok &= sampled_served((st, &body), (st1, &body1), &first)?;
        ok &= cache(&url, &err_log, true, "drafted")?;
        println!("server stopped: {}", served.stop()?);
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(ACTS_ON)?;
        if let Some(word) = levers.residency() {
            return Err(format!(
                "BLOOMERY_RESIDENCY={word}: the gate sets it on every server it starts but the \
                 seat's own defaults (off, or the residency word's own)"
            )
            .into());
        }
        if let Some(word) = levers.xstream() {
            return Err(format!(
                "BLOOMERY_XSTREAM={word}: the gate holds its servers' `xstream=` lines to the \
                 unset rule"
            )
            .into());
        }
        if let Some(word) = levers.draft() {
            return Err(format!(
                "BLOOMERY_DRAFT={word}: the gate sets it on every server it starts (the plain \
                 arm off, the drafted arm mtp)"
            )
            .into());
        }
        let a = parse_args()?;
        if a.arm != Arm::Drafted && levers.mtp_draft().is_some() {
            return Err(
                "BLOOMERY_MTP_DRAFT names the MTP draft file; it needs --arm drafted".into(),
            );
        }
        let reference = gen_tokens(&a.gen_log)?;
        std::fs::create_dir_all(&a.dir)?;
        let ok = match a.arm {
            Arm::Plain => plain(&a.dir, &levers, &a.prompt, &reference)?,
            Arm::Drafted => drafted_main(&a.dir, &levers, &a.prompt, &a.ids, &reference)?,
        };
        if ok {
            println!("gate-gpu-qwen38-serve: PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
