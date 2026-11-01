//! `gate_qwen38_serve` — `bloomery-serve-qwen38` on the card in view
//! (placement gate: the 3090's bytes on it), driven over HTTP, as one process
//! under the GPU gate lock. The gate's own plans resolve the card as each
//! server does, so a unified card plans its pool there too.
//!
//!     gate_qwen38_serve --arm plain|drafted|lookup --gen <generate_qwen3moe log>
//!                       --prompt <text> --ids <a,b,…> --dir <out>
//!
//! Three arms, each a process under the recipe's gate lock. `lookup` runs
//! two two-slot servers, draft off then lookup: their ids agree, including
//! a seeded sampled request the lookup drafts, whose draws leave the greedy
//! ids, and one with logprobs that drafts too; two-row capture nodes and draft
//! counts prove that verifies keep both one and two positions (file-bound),
//! and a completion whose draft counts are malformed, or one without the
//! other, fails the arm. Both
//! conversations restore their cached prefixes on different slots after
//! the resident slots are erased, with the ids each gave re-run from the
//! cache in its own slot. A third two-slot lookup server, under the
//! width chooser's cost mode, runs the code and the prose requests
//! together: their slots take turns at the passes, their ids are the
//! draft-off server's, and each slot's chooser ends its warm-up
//! ([`lookup_cost_turns`]). `plain` runs the main server with the
//! draft off, then every clause of its own below. `drafted` runs the main
//! server with MTP and its drafted clauses alone, stopped before the one
//! other server it starts, the slot-file clause's. These last two arms load
//! their mains, the slots clause's two, the width chooser's one, the sampled
//! rounds' one, the slot-file clause's one, the unhosted-pool clause's one
//! and the draft share clause's unset-draft one (stopped at its `load
//! draft=off` record); the `ctx` clause's three starts, the residency word's
//! one and the unhosted-pool clause's plan-only one stop before the load.
//! The plain and drafted arms start the server beside this binary (`--host
//! 127.0.0.1 --port 0 --place gate --ctx-size 4096`), read its address from
//! stderr, wait for `/health`, then check:
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
//!   for bit steps, so the server's prompt feed (the prompt less its last id,
//!   then its step) and `generate_qwen3moe`'s (the whole prompt) leave the
//!   same state, while past that the server runs the last id as the ubatch
//!   walk's last row (`Q38::prefill_next`), a delta chunk it completes left
//!   to the next step to fold where the CLI's walk folds it;
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
//! - `shared_system_prompt_stays_in_the_slot`: different questions sharing one
//!   system message keep its message-start checkpoint in the live slot and
//!   answer a fresh run's ids. Removing the message-start registration keeps zero;
//! - `systemone_is_the_schema_path` (`shared/systemone_clause.rs`, the clause
//!   `gate_qwen3_serve` holds too; the plain arm's main server, after
//!   `cache`): the rows of `tools/ref/clef/suite.jsonl` that fit the
//!   server's context, POSTed to `/v1/systemone` and answered by this model
//!   through the structured-output path, are 200s with the decision seats'
//!   keys, and their picks are `/v1/chat/completions`' on the same request
//!   converted with `response_format`. FAIL-first: a server without the
//!   route answers every row 404;
//! - `cache` ([`cache`]): two sessions' requests through the prompt cache —
//!   a verbatim resend after the other session keeps every held position
//!   (under the draft the turn's prompt end, its last window having fed
//!   rows past the reply) and gives the ids of the same session run with no
//!   switch; a resend with the reply's reasoning stripped keeps the
//!   checkpoint at the first turn's end and gives a fresh run's ids; under
//!   the draft each cut keeps its checkpoint with the draft rejoined there
//!   and drafting, no draft-off line and no `mtp keep` record, on both sides
//!   of the seat's break-even for the reply (its `draft keep` line) at the
//!   turn's end, by reply length, each with a fresh run's ids. The plain
//!   arm's main server runs
//!   the clause with the draft off; its drafted half runs on the drafted
//!   arm's main server, after that server's drafted clauses;
//! - under the draft (`--arm drafted`, that arm's main server): a greedy
//!   `/completion` gives the ids of the same server's steps (the request
//!   one token a request, reusing the prefix; a plain load places
//!   more target experts on the card, so its ids are no reference) with its
//!   draft counts carried
//!   (a draft changes which passes run, never a token; the fixture tier
//!   defers the counts by name: its draft proposes and its random-weight
//!   target accepts none), and `/props`'
//!   `engine.draft` names the draft file
//!   `refset::arch::qwen4exp::mtp::draft_file` picks, by name and path;
//!   requests that extend the sequence the
//!   server holds (`continued`): each keeps the held prefix (`cache_n`), a
//!   prompt call past it prints one `mtp prompt` record — the draft caught
//!   up at the kept position, nothing skipped — and none prints a skip, and
//!   each drafts, with those steps' ids (the generated ones); the prompt
//!   resent keeps its end by a cut that rejoins the draft, after which the
//!   extension drafts from its join, with the same prompt's ids fed fresh;
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
//!   `default`); and the slots clause's server (`--place gate`, two slots,
//!   no expert tier card) names the shape of its rounds, `pass=one`
//!   (`slots_parallel_line_names_one_pass`; FAIL-first: the word printed
//!   `turns`). No server here runs `--place bp`, the one placement that
//!   prints `pass=turns` for the seat (`Pass::of` in `bind`'s unit test
//!   holds the word);
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
//!   On the same server, the cut slot and the late request
//!   ([`slots_drafted_beside`]): the break-even clause's long turn fresh,
//!   then `TURN_A`'s drafted request on the other slot and, once it
//!   decodes, the turn resent with its reply stripped, cut on the turn's
//!   slot with its draft rejoined there: the cut slot's windows run in
//!   rounds of two rows beside the drafted request's, its ids the same ids
//!   fed fresh, and the drafted request's ids its alone run's
//!   (`slots_drafted_off_slot_steps_beside_a_window`). Then the late
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
//!   Then the drafted sampled clauses ([`slots_drafted_sampled`]): the
//!   sampled rounds clauses' pair (below) banning the end ids
//!   (`ignore_eos`), so each request draws in drafted passes alone and
//!   beside the other on this drafted load: over the together run every
//!   record is `cmd=pass` of `passes=1` carrying both slots
//!   (`slots_drafted_sampled_rounds_run_one_pass`: the seat runs both
//!   sampled windows as one pass) and each request's together ids are its
//!   alone ones (`slots_drafted_sampled_together_alone`: each window's
//!   draws its own); the same pair asking one probability a token also
//!   drafts: every together record `cmd=pass` of `passes=1` carrying both
//!   slots (`slots_drafted_probs_rounds_run_one_pass`: both windows share a
//!   pass and collect their probabilities) and the together ids the alone ones
//!   (`slots_drafted_probs_together_alone`); then the pair with no ban,
//!   the second asking 8 ids: the first drafts beside it and then alone
//!   from the state the shared passes left its draft, its draft
//!   proposals an id past the second's at least half its alone run's
//!   (`slots_drafted_sampled_tail_drafts`) and some accepted
//!   (`slots_drafted_sampled_tail_accepts`, deferred by name in the
//!   fixture tier, whose target accepts no proposal).
//!   Then one more server of `--parallel 2` under the seat's own defaults
//!   but the draft (no `--place`, no `BLOOMERY_RESIDENCY`, no
//!   `BLOOMERY_MTP_WIDTH`, and `BLOOMERY_DRAFT=mtp`, so the clause holds the
//!   drafted load whatever the unset draft's context rule picks): the unset rule picks
//!   the placement, the adaptive residency runs, the draft runs behind the
//!   width chooser; `BLOOMERY_STEP_STATS=1` for (v3)'s records. Its
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
//!   runs the widths its timings choose, so whether a run cuts a narrower
//!   window here is not fixed, and the cut is the width chooser's server's,
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
//!   it; a drafted load's plain rounds a slot at a time (its open's
//!   `step_pass` false under the draft) print `passes=2`, red on the
//!   drafted sampled round clause alone; a row of the pass its draft is not
//!   told of leaves no refresh, and the tail's first proposal is refused by
//!   name, red on the tail clause. Of the seat's own defaults: a seat that
//!   opens with
//!   `Body38::open_placed` runs no residency machine whatever the rule — no
//!   `residency host` record and no `residency pass`, and the reset is the
//!   server's 501; a draft run with no chooser prints no `mtp width` record.
//! - the width chooser's clause ([`width_cost`], a server of its own under
//!   `BLOOMERY_DRAFT=mtp`, `BLOOMERY_RESIDENCY=off` and
//!   `BLOOMERY_MTP_WIDTH=cost`): the prompt's steps (one token a request),
//!   one greedy request then the server's orderly stop, which prints the
//!   request's record; its ids its steps', its draft counts carried, and the chooser's `mtp width` record
//!   its own aggregate — windows the widths above 0,
//!   E inside the window's rows, a width below the draft's own among the
//!   passes. It is the one clause that runs the default chooser on a
//!   loaded server and holds its tokens to its steps: every other
//!   drafted server this gate starts pins `BLOOMERY_MTP_WIDTH=fixed` (the
//!   clauses that compare draft counts across two runs of a server hold the
//!   fixed window; a chooser its clock drives promises no count twice), and
//!   the seat under its own defaults but the draft (the slots clause's
//!   second server) holds the chooser's records but no id equality.
//! - the shared-prefix clause ([`prefix`], a server of its own under
//!   `BLOOMERY_DRAFT=off` and `BLOOMERY_RESIDENCY=off` at a context of two
//!   checkpoint spacings): a prompt past the first spacing answered and
//!   another taking the slot, then a request that shares only the prompt's
//!   first ids restores from the host cache at the spacing's multiple
//!   below them (the saved state carries the doubling multiples beside its
//!   last point), its ids a fresh run's.
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
//! waited for (the drafted arm then runs the slot-file clause on a server
//! of its own and ends, [`slot_file`]: a slot saved to a file, erased and
//! restored under the draft continues as the straight run does, and the file
//! cut short is refused by name), and, on the plain arm, more
//! servers start on the card, one at a time: the `ctx` clause's three, the
//! `paged` clause's probes, the draft share clause's three and the residency
//! word's one (they stop before the load, or are refused before it, but the
//! `paged` clause's small card and the draft share clause's unset-draft
//! server, stopped at their load's draft record), the `paged` clause's
//! `generate_qwen3moe` runs (one and four slots, two tokens each), then the slots clause's two
//! and the sampled rounds' one (above).
//!
//! - `ctx` ([`ctx`]): a server with no `--ctx-size` prints its default's
//!   one `ctx` line before its load (stopped there): the seats' fit at one
//!   slot, its word held against this gate's own plans (the default held,
//!   the step above not when it fitted); one with `--ctx-size` set and
//!   no `--parallel` takes one slot at the whole flag (`from=ctx` on its
//!   `parallel` line, and the one-request note on stderr), and one asked a
//!   position past the one-slot fit is refused by name before it listens
//!   (with the share the card holds, or by the limit when the card holds
//!   more). The fit's own terms are held by
//!   `bloomery_placement::placement::ctx_ask`'s tests; these servers hold
//!   the seat's composition and its lines;
//! - `paged` ([`paged`], deferred by name in the fixture tier: its rooms
//!   are the real file's shapes): the NVMe tier's paged rule at two
//!   rooms (`BLOOMERY_HOST_ROOM`), each its bytes beside the gate card and,
//!   on a unified card, the card's share of the pool in the plan at that
//!   room ([`room_beside_card`]), the gate's own plans made on the card the
//!   servers resolved: at 27 GiB, where the gate plan pages its
//!   host experts through the tier's RAM arena, the server with no
//!   `--ctx-size` serves its default one slot with no `paged` line (the
//!   rule has no count to reduce, and the placement left the draft off) on a
//!   plan that pages (`paged_default_serves_one_column`, the `cache` line's
//!   arena), `--parallel 4` serves four requests as one pass, each with its
//!   alone ids (`paged_parallel_serves_one_pass`, `paged_parallel_ids_as_alone`), with
//!   a round of all four slots in one pass (`paged_parallel_rounds_carry_all_slots`), and
//!   `BLOOMERY_DRAFT=mtp` serves one slot with the draft on
//!   (`paged_draft_mtp_serves_one_slot`);
//!   at 40 GiB, where the tier pages experts with no arena, `--parallel 2`
//!   serves its two slots (`unpaged_parallel_serves_as_asked`). The small
//!   card's case last: a server with no `--place` (the one card in view,
//!   `a`) under the budget whose drafted one-slot plan fits a context the
//!   plain plan holds twice of, at 27 GiB — the unset draft yields to the
//!   context on that plan (`paged_small_card_yields_on_one_card`, the
//!   premise), and the load runs the plain plan the yield was judged on, its
//!   `load draft=off` record naming the yield
//!   (`paged_small_card_load_names_the_yield`). Beside them, at 27 GiB: the
//!   checkpoints every resident sequence may pin sit outside the room the
//!   split fills — the server's arena is the gate's own plan's (no host
//!   reserve for them) less `pinned_checkpoints(slots)`, and its host need that
//!   plan's plus the same (`paged_checkpoints_sit_outside_the_arena`, read
//!   off the `cache` line); the prompt cache's default takes nothing beside
//!   the room the split fills (`paged_default_cache_is_off_by_the_split`:
//!   `ram=0`, the split named in `why=`); a residency word set to `off`
//!   reaches the `plan` record and loads no residency machine
//!   (`paged_set_off_residency_is_accepted`), and one set to `mid-p0-s1`
//!   loads it and serves a request (`paged_set_residency_serves`: the
//!   machine's victim fills read through the NVMe tier's reader, the lane's
//!   reads of the arena's ids go through its slots and none through the
//!   model file's mapping, read off the run's last `nvtier` record). The
//!   floor counts every slot's checkpoints, so between the one-slot plan's
//!   floor and two slots' lies a band of rooms a set `--parallel 2` cannot
//!   clear: the gate reads both floors off its own plans
//!   (`paged_floor_band_is_between_the_slot_floors`), and at the band's
//!   middle `--parallel 2` is refused by the floor, by name
//!   (`paged_floor_set_parallel_is_refused_by_the_floor`). On a unified card
//!   the pool lowers the card budget until the floor holds, so the gate asks
//!   `--parallel 2` at the two-slot floor at one position instead: its plan
//!   is the pool's (`paged_floor_pool_lowers_the_two_slot_budget`), and the
//!   server stands on it (`paged_floor_set_parallel_stands_on_the_pool_budget`).
//!   Then `generate_qwen3moe` with no `--place` at 27 GiB: its unset
//!   draft at `a` falls to the same rule, and the run's `load draft=off`
//!   record names it (`paged_cli_drafts_nothing_by_the_rule`). FAIL-first:
//!   a default of two slots prints a `paged` line at 27 GiB; a rule on the
//!   paged experts' bytes refuses the 40 GiB server; a seat whose yield
//!   leaves the draft on, or whose paged rule overrides the yield's reason,
//!   is red on the small card; a CLI with no rule drafts on the paged plan;
//!   a machine with no checkpoint reserve gives the arena of the gate's own
//!   plan; a seat that takes the set word on a paged plan prints its `plan`
//!   record and loads; one that refuses `off` too never reaches the 27 GiB
//!   `plan` record;
//! - `draft_slot_share` ([`draft_slot_share`]): the unset draft judged on a
//!   slot's share of `--ctx-size`: at `--ctx-size 8 --parallel 2` (four
//!   positions a slot, a window needs five) an unset draft turns off, its
//!   `load draft=off` record naming the ten positions of the two slots past
//!   the eight (`draft_unset_yields_to_the_slot_share`), and a set one is
//!   refused by name before its plan (`draft_set_is_refused_by_name`); at
//!   the default context of `--parallel 2` the unset draft drafts, so a
//!   server with `BLOOMERY_MTP_WIDTH` set beside it plans, no refusal
//!   naming a draft that drafts nothing (`draft_unset_drafts_on_two_slots`);
//!   the fixture tier defers the clause by name (the draft file is the real
//!   file's);
//! - `residency_loads_the_word` ([`residency_loads_the_word`]): a server of
//!   the main arguments under `BLOOMERY_RESIDENCY=mid-p0-s1`, stopped before
//!   its load, prints the `residency lever` record of that word with why
//!   `set` and the `residency host` record of the same word — the seat's
//!   open takes a set word as given. FAIL-first mutants: a seat that ignores
//!   the set word runs the unset rule's `off` at `--place gate` and prints
//!   no host record; one that records the lever's why as unset names another
//!   why;
//! - `churn_unhosted` ([`churn_unhosted`], both tiers): a residency word
//!   that asks for the pool unhosted (`mid-p0-s1-unhosted`) at a host room
//!   that holds the plan's host need and not the churn pool beside it keeps
//!   its P and leaves the pool out of the host set, so a victim outside the
//!   set is served from the file and counted; the unset word on the same
//!   room climbs to a hosted pool instead. Three servers of `--parallel 2`
//!   and `--cache-ram 0` (no cache budget, so the card rule alone sets the
//!   context and no room moves it) and the draft off. The first stops at
//!   `--plan` under a room no plan needs, the residency unset: its `cache`
//!   line gives the plan's host need, its `residency host` record the pool
//!   at P 0 (hosted, pinned 0) and the plan's host headroom, the machine's
//!   there (the room asked is past its figure). A stated room binds the
//!   headroom of a plan whose need it holds (`Plan::plan_at`: the tighter of
//!   the room and the machine's figure, past the need), and the unset rule
//!   reads the available bytes past the need too (`BLOOMERY_HOST_ROOM`
//!   replaces the reading): both budgets are what the room sets. The second
//!   server runs under the room `need + pool / 2` with
//!   `BLOOMERY_RESIDENCY=mid-p0-s1-unhosted`, loads, and answers one greedy
//!   request of [`CHURN_PREDICT`] ids: the `residency lever` record names
//!   that word with why `set`
//!   (`churn_unhosted_lever_record_names_the_asked_word`); the `residency
//!   host` record carries the word and reads `hosted=unhosted` at `pinned=0`
//!   with a pool that is not empty
//!   (`churn_unhosted_host_record_keeps_p0_unhosted`); the request is a 200
//!   of at least [`CHURN_MIN_TOKENS`] ids (`churn_unhosted_request_completes`);
//!   `rereads` plus `unresident` over the `residency pass` records sum above
//!   0, and the log holds no refused flip
//!   (`churn_unhosted_victims_are_served_from_the_file`). The third stops at
//!   `--plan` under the same room with the residency unset: its `residency
//!   unset` record names the climb (`p0→p<q>`, `host: the churn pool did not
//!   fit, raised`) and its `residency host` record the climbed word at
//!   `pinned=<q>` with the pool hosted (`churn_unset_climbs_to_a_hosted_pool`).
//!   A plan-only run whose terms cannot be read fails the five checks by
//!   name, none skipped. FAIL-first mutants: an unset word of this body
//!   keeping P 0 with the pool unhosted prints a `residency unset` record
//!   that is not the climb and `hosted=unhosted` at `pinned=0`, red on the
//!   last check; a seat that hands the load `Hosted` beside the asked word
//!   holds the pool against the plan's headroom, and the pool's refusal
//!   (`residency pinning 0 experts a layer: the churn pool's <n> experts take
//!   <pool> B of host RAM, and the plan leaves the host <pool / 2> B`,
//!   `ChurnPool::check_beside`) ends the server before it listens, red on
//!   the checks after the lever record.
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
//! are the server's (`qwen38_levers::ACTS_ON`, the one list): one the server
//! would refuse is refused here, at `main`, before the server starts. The gate sets two of
//! them itself on every server it starts, so neither the placement's defaults
//! nor the environment move what a clause means: `BLOOMERY_DRAFT` (`mtp` on
//! the drafted arm's main server and the drafting clauses' own servers, `off`
//! on the plain arm's main server and the plain clauses' own) and
//! `BLOOMERY_RESIDENCY` (`off`; the residency word's server sets its word) —
//! the slots clause's second server sets the draft `mtp` and leaves the
//! residency unset: it is the seat under its own defaults but the draft;
//! the unhosted-pool clause's
//! servers set the draft `off` and leave the residency unset. The drafted arm's main
//! server also takes the drafting load's `BLOOMERY_CARD_BUDGET`
//! (`tier::plan_levers`: in the fixture tier the header's budget plus the
//! draft's card bytes at `CTX`, as a drafting load plans there; the runner
//! exports the target's alone), and `/props`' engine clauses plan under the
//! same; the slots clause's second server takes it at its two slots
//! ([`drafted_card_budget`]).
//! `BLOOMERY_DRAFT`, `BLOOMERY_RESIDENCY` and `BLOOMERY_XSTREAM` set in this
//! binary's environment are refused by name:
//! the arm names the draft (a set `BLOOMERY_MTP_DRAFT` needs the drafted
//! arm), and every server holds the unset `xstream=` rule.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen38_serve", gate::run())
}

// The levers the server acts on: the one list the server's `main` parses, so a lever
// it would refuse is refused here first.
#[cfg(feature = "gpu")]
#[path = "shared/qwen38_levers.rs"]
mod qwen38_levers;

// The `/v1/systemone` clause the generative serve gates share.
#[cfg(feature = "gpu")]
#[path = "shared/systemone_clause.rs"]
mod systemone_clause;

#[cfg(feature = "gpu")]
mod gate {
    use super::qwen38_levers::ACTS_ON;
    use std::fs::File;
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, Stdio};
    use std::sync::{OnceLock, mpsc};
    use std::thread::JoinHandle;
    use std::time::Duration;

    use app::mtp::MtpBody;
    use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for;
    use bloomery_gpu::arch::qwen3moe::{
        Body38, CHECKPOINT_EVERY, KvQ8, Prompt38, checkpoint_budget38, seq38_bytes,
    };
    use bloomery_gpu::host::xstream::XSTREAM_ROOM;
    use bloomery_gpu_gates::ctx_ladder::{self, ctx_line, default_line_held, step_above};
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::record::{self, Fields, ReadError};
    use bloomery_gpu_gates::serve_client::{
        LookupCompletion, PrefixRestore, curl, draft_counts, ids_of, json_of, metric, parse_ids,
        prefix_restore, server_log, shared_system_prompt, slot_action, slot_erased,
        stepped_completion, tokenized,
    };
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{GateError, checks_failed, ref_model_path, verdict};
    use gguf::Split;
    use model::arch::models::Mixer;
    use model::arch::qwen35moe::head_list::head_rows_of;
    use model::arch::qwen35moe::place::{
        Experts, KERNEL_POSITIONS, MtpInputs, PlaceError, PlanInputs, machine_for_route,
    };
    use model::placement::ctx::CtxLimit;
    use model::placement::workstation::{
        CHECKPOINTS_RESERVE, CardSpec, DeviceInfo, HostNeed, HostRead,
    };
    use model::placement::{PlacementError, PlanLevers};
    use refset::arch::qwen4exp::VERIFIED_POSITIONS;
    use refset::arch::qwen4exp::mtp::draft_file;
    use serde_json::{Value, json};
    use threads::helper::{Placement, spawn_helper};

    const USAGE: &str = "usage: gate_qwen38_serve --arm plain|drafted|lookup --gen <generate_qwen3moe log> \
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
    /// The shared-prefix clause's prompt A in ids, the ids C shares with it,
    /// the `cache_n` C keeps (the checkpoints' first multiple, the doubling
    /// multiple nearest at or below the shared ids) and the context its
    /// server holds. The checkpoints' spacing is the ubatch's, so A passes it
    /// and C shares more than it but not the next multiple.
    const PREFIX_A: usize = 4_400;
    const PREFIX_SHARED: usize = 4_200;
    const PREFIX_KEPT: u64 = CHECKPOINT_EVERY as u64;
    const PREFIX_CTX: usize = 2 * CHECKPOINT_EVERY as usize;
    const _: () = assert!(
        PREFIX_A + N_PREDICT < PREFIX_CTX
            && PREFIX_KEPT as usize <= PREFIX_SHARED
            && PREFIX_SHARED < 2 * CHECKPOINT_EVERY as usize
            && PREFIX_SHARED < PREFIX_A
    );
    /// The context the servers are started at (`SERVER_ARGS`).
    const CTX: usize = 4096;
    /// The slots the seat serves under `--parallel 2`, the count of the slots clause's second
    /// server, which runs the seat at its own defaults.
    const SLOTS_DEFAULT: usize = 2;
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
    /// The drafted sampled tail clause's short request: the long one steps
    /// beside it this many rounds, then drafts alone.
    const DRAFTED_SAMPLED_SHORT: usize = 8;
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
    /// The unhosted-pool clause's plan-only server asks for a host room no
    /// plan of the file needs (1 TiB), so its pool fits whatever the file.
    const CHURN_PLAN_ROOM: u64 = 1 << 40;
    /// The unhosted-pool clause's servers: the seat's defaults at
    /// `--parallel 2` but for the prompt cache, off, and a named context, so
    /// neither a cache budget nor the default context's fit moves the plan
    /// between the rooms. PIN(2026-10-29): the context is named (8,192
    /// positions a slot): the default context fills the card down to the
    /// residency's least experts a layer, which holds P 0 and no higher P, so
    /// the unset word on this room had no P to climb to; the clause is about
    /// the host's room, not the default context (`slots_default_*` holds that).
    const CHURN_ARGS: [&str; 10] = [
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--parallel",
        "2",
        "--cache-ram",
        "0",
        "--ctx-size",
        "16384",
    ];
    /// The unhosted-pool clause's request: ids asked, and the least that
    /// must come back for a victim outside the host set to have been picked.
    const CHURN_PREDICT: usize = 128;
    const CHURN_MIN_TOKENS: usize = 64;
    /// A flip the host cannot serve is refused with this sentence
    /// (`swap::Unresident`'s Display); the clause's server must never print it.
    const UNRESIDENT_REFUSED: &str = "is not host-resident after its pages were read in again";
    /// The word that asks the load to keep P 0 with the churn pool unhosted.
    const CHURN_WORD: &str = "mid-p0-s1-unhosted";
    /// The unhosted-pool clause's checks, one line each, and the unset
    /// word's one beside them.
    const CHURN_WHY: &str = "churn_unhosted_lever_record_names_the_asked_word";
    const CHURN_HOST: &str = "churn_unhosted_host_record_keeps_p0_unhosted";
    const CHURN_REQUEST: &str = "churn_unhosted_request_completes";
    const CHURN_VICTIMS: &str = "churn_unhosted_victims_are_served_from_the_file";
    const CHURN_CLIMBS: &str = "churn_unset_climbs_to_a_hosted_pool";

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

        /// The child's exit status within `POLLS` seconds; `None` while it
        /// still runs.
        fn wait_exit(&mut self) -> Result<Option<std::process::ExitStatus>, GateError> {
            for _ in 0..POLLS {
                if let Some(status) = self.child.try_wait()? {
                    return Ok(Some(status));
                }
                std::thread::sleep(Duration::from_secs(1));
            }
            Ok(None)
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
            bloomery_gpu_gates::die_with_parent(cmd);
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
        Lookup,
    }

    impl Arm {
        fn parse(v: &str) -> Result<Arm, GateError> {
            match v {
                "plain" => Ok(Arm::Plain),
                "drafted" => Ok(Arm::Drafted),
                "lookup" => Ok(Arm::Lookup),
                other => Err(format!("--arm is plain, drafted or lookup, not {other}").into()),
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
        // The card the server's `--place gate` resolved to and the room the
        // server planned at, as its plan record names them ([`server_stage`]).
        // A reading beside the loaded server sees its bytes as taken; a
        // unified card's pool is the free bytes, and the room places the
        // row-gathered tables.
        let (stage, plan_room) = server_stage(err_log)?;
        let inputs = PlanInputs::describe_in(&split, (plan_room, HostRead::Given))?;
        let experts = experts38(levers)?;
        let ub = ubatch_for(CTX)?;
        let machine = route_machine(
            &inputs,
            stage,
            (u64::try_from(ub)?, u64::try_from(CTX)?),
            experts,
            (1, mtp),
        )?;
        // Under the draft the plan carries it (its granules, its store, its
        // row map and its program's arena beside the target's card terms),
        // and `/props` files its bytes as the card's `draft` class.
        let (draft_path, _) = draft_file(levers.mtp_draft(), &path);
        let terms = |dense: u64, experts_at: u64, host: u64, tables: u64, kv: u64, draft: u64| {
            (dense + experts_at + draft, host + tables, kv, draft)
        };
        let (card_bytes, host_bytes, kv, draft_bytes) = match mtp {
            false => {
                let levers_plan = PlanLevers::from_levers(levers)?;
                let plan = inputs.plan_with(
                    &machine,
                    (u64::try_from(CTX)?, u64::try_from(ub)?),
                    &levers_plan,
                    experts,
                )?;
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
                let (mtp, levers_plan) = drafted_plan_inputs(levers, &path, &split, &inputs, 1)?;
                let with = inputs.plan_mtp_with(
                    &machine,
                    (u64::try_from(CTX)?, u64::try_from(ub)?),
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

        /// Under the draft, a request of `sent` ids that kept `at` by a cut or
        /// a prefix put back with the draft rejoined there: no draft-off line,
        /// no `mtp keep` record, and it drafted. Its joins when ids are left
        /// for a prompt call past `at`: the first at `at`, walking the
        /// rejoined row, and one more at the start of each further call of
        /// the same prompt (a prompt fed beside another slot's windows runs
        /// in more than one call), each walking the draft up to it and
        /// skipping nothing; none when one id is left, which the first step
        /// feeds, the step walking the rejoined row with it and printing no
        /// record (as a request that ended on its first step and is extended
        /// by one id).
        // PIN(2026-10-13): the joins after the first are each later call of the prompt; was one
        // join only, which a cut prompt fed in two calls beside a drafted request's windows failed
        // while its ids and drafts were right.
        fn rejoined_at(&self, at: u64, sent: usize) -> bool {
            let join = if sent as u64 > at + 1 {
                self.joined_from(at)
            } else {
                self.joins.is_empty()
            };
            self.ok
                && self.cache_n == at
                && self.draft_n > 0
                && self.offs.is_empty()
                && self.keeps.is_empty()
                && join
        }

        /// One join printed for this request, at `start`, that walked the
        /// draft up to it and skipped nothing.
        fn joined_at(&self, start: u64) -> bool {
            matches!(self.joins.as_slice(), [j] if j.start == start
                && j.caught_up >= 1
                && j.skipped == "none")
        }

        /// The joins printed for this request: the first at `start`, the
        /// starts rising, each walking the draft up to it and skipping nothing.
        fn joined_from(&self, start: u64) -> bool {
            self.joins.first().is_some_and(|j| j.start == start)
                && self.joins.windows(2).all(|w| w[0].start < w[1].start)
                && self
                    .joins
                    .iter()
                    .all(|j| j.caught_up >= 1 && j.skipped == "none")
        }
    }

    /// The drafted server's continuations (`BLOOMERY_DRAFT=mtp`): requests
    /// that extend the sequence the server holds keep its prefix
    /// (`cache_prompt`), and the draft joins them — the rows an earlier
    /// request left waiting walked, the token the new request puts at their
    /// last position — so it drafts from the first window, with
    /// `reference`'s ids (the server's steps). Three joins: after a request
    /// fed fresh that ended on its first step (`n_predict` 1), extended by the id it generated: its prompt call is
    /// empty, the join is its first step's, and no `mtp prompt` record says
    /// it skipped; after drafted windows (the held sequence is at most 11
    /// ids past the prompt: 8 generated, the last window's 3 past them),
    /// extended by `reference`'s next ids, a prompt call of 1 to 4 ids and
    /// its record; and after a first step again, extended by ids the model
    /// did not generate, against the same prompt fed fresh. That first step
    /// is the prompt resent, which keeps the prompt's end (one short of it)
    /// by a cut that rejoins the draft there (the point at the first prompt
    /// call's end holds the target's row before it, the draft's store holds
    /// past it): no draft-off line, no `mtp keep` record; the extension keeps
    /// the whole prompt and drafts from the join at its end, with the fresh
    /// run's ids. Every prompt call stays below
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
        // PIN(2026-10-12): the cut rejoins the draft at the point it keeps (round kp-draft): `b`
        // keeps p − 1 by a cut with the draft on (no draft-off line, no `mtp keep` record, its
        // first step walking the rejoined row), and `x3` keeps the whole prompt and drafts from
        // the join at p; was the draft off from p − 1 by name, `x3` keeping the prompt with the
        // draft off at or past the break-even of its 8 tokens, else the state the lane saved.
        check(
            &mut ok,
            "continued_by_other_ids_keeps_the_prompt_and_skips_by_name",
            b.ok && b.cache_n == p - 1
                && b.offs.is_empty()
                && b.keeps.is_empty()
                && b.joins.is_empty()
                && x3.ok
                && x3.cache_n == p
                && x3.draft_n > 0
                && x3.offs.is_empty()
                && x3.keeps.is_empty()
                && x3.joined_at(p),
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

    /// The gate card a `--place gate` server plans on ([`stage_at`]) at the free
    /// reading its `plan` record names ([`server_card_free`]), and the host
    /// room that record names.
    fn server_stage(err_log: &Path) -> Result<(CardSpec, u64), GateError> {
        let room = seat_log(err_log)?.one(&record::PLAN38)?.u64("plan_room")?;
        Ok((stage_at(Place::Gate, server_card_free(err_log)?)?, room))
    }

    /// This process's census, read once before the arm's first server holds
    /// a card ([`run`]): a census opens a context on each device, which beside
    /// a loaded server takes from the pool the server planned, and the plans
    /// the gate derives take the server's own free reading
    /// ([`server_card_free`]), not this one's.
    fn census() -> Result<&'static [DeviceInfo], GateError> {
        static CENSUS: OnceLock<Vec<DeviceInfo>> = OnceLock::new();
        if let Some(c) = CENSUS.get() {
            return Ok(c);
        }
        let read = bloomery_gpu::census()?;
        Ok(CENSUS.get_or_init(|| read))
    }

    /// The card `place` resolves to on this census, as the server resolves
    /// it — `--place gate` the 3090's bytes on the card in view, no `--place`
    /// (`a`) the card as it reports itself; a unified card plans its pool —
    /// at the free reading `free` a server's `plan` record names.
    fn stage_at(place: Place, free: u64) -> Result<CardSpec, GateError> {
        let mut stage = place.on(census()?)?.card_specs()?[0];
        stage.free_bytes = Some(free);
        Ok(stage)
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

    /// The slot emptied (`POST /slots/0?action=erase`): its rows are reset and
    /// its ids forgotten, and nothing is saved by it. The prompt cache keeps
    /// every state it holds, among them the one the lane saved when the last
    /// request ended.
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
    ///   fresh prompt call would run through the ubatch walk. The plain
    ///   server saves each request's state on its lane when the request
    ///   ends and an erase keeps the cache, so the switched resend finds the
    ///   state the first resend left (its ids and its reply) and keeps the
    ///   whole resend but its last id, where the first resend kept what A
    ///   held (under the draft that prefix is kept like any other, the draft
    ///   rejoined there);
    /// - (b) A's first turn resent with the reasoning-stripped reply in place
    ///   of the reply, so the shared prefix ends at the first turn's end: it
    ///   keeps the checkpoint there (the end mark of the turn's prompt call,
    ///   one short of the turn: that call ran its last id as the ubatch
    ///   walk's last row, `Q38::prefill_next`), the draft on
    ///   or off, and its ids are a fresh run's of the same ids (every prompt
    ///   call of at least nine ids is the ubatch walk, whose bits do not
    ///   depend on where a call is cut);
    /// - under the draft, the cuts (the resend with no switch, the stripped
    ///   resend: the turn's end; the switched resend: all but its last id)
    ///   each keep their point with the draft rejoined there: each point is
    ///   a prompt call's end, which holds the target's row before it, and
    ///   the draft's store reaches it, so the seat's break-even weighs
    ///   nothing ([`Q38::draft_keep`]); each request prints no draft-off line
    ///   and no `mtp keep` record, joins the draft at its point and drafts.
    ///   The fresh run after them drafts. Without the draft no such line
    ///   prints;
    /// - (c) under the draft, the turn's end kept on both sides of the
    ///   break-even ([`break_even`]).
    ///
    /// Mutants: the state keeping the current position's recurrent stores
    /// alone ((b) keeps 0); a resume that leaves the PLE history behind (the
    /// resend is refused); a keep rule that grants any position (B's request
    /// cuts where no checkpoint stands, refused); under the draft every
    /// shorter prefix kept as none (the cuts' clause sees no cut); the seat
    /// saving on the request path (`Seat::snapshots_async` false: the plain
    /// switched resend keeps A's held positions, not the whole resend).
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

        // The run with no switch first, then the slot dropped: an erase keeps the prompt cache,
        // which holds the states the lane saved when each request ended.
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
        // PIN(2026-10-12): under the draft a cut to the turn's end rejoins the draft there
        // (round kp-draft), so the break-even weighs nothing and the cut keeps the turn's end;
        // was the turn's end only at or past the seat's break-even for the reply, the draft off,
        // else a reset.
        let keep38 = drafted.then(|| Keep38::read(err_log)).transpose()?;
        let want = if drafted { turn_end } else { held };
        // PIN(2026-10-11): the plain switched resend keeps every id of the resend but its last,
        // not A's held positions; the lane saved the first resend's end state (its ids and its
        // reply) when that request ended, the erase kept it, and the switched resend takes the
        // prefix of it up to the nearest point it carries at or below all but the resend's last
        // id, which is the end of the first resend's prompt call. A request keeps at most all
        // but its last id, and the first resend still keeps A's held positions.
        // PIN(2026-10-12): under the draft the switched resend keeps that prefix with the draft
        // rejoined at its end (round kp-draft: the saved state carries the point's row); was the
        // prefix weighed against the break-even, kept with the draft off by name at or past it
        // (`cache load positions=121 common=105 kept=105`, `mtp keep branch=kept prefix=105
        // break_even=86`), else a reset.
        let all_but_last = resend.len() as u64 - 1;
        let kept = if drafted {
            a2.cache_n == all_but_last && r2.cache_n == want
        } else {
            r2.cache_n == held && a2.cache_n == all_but_last && a2.cache_n >= held
        };
        // Under the draft, whether a verbatim resend cuts depends on where the draft's last call
        // left its rows, which the file's acceptance decides: the fixture tier defers it by name.
        let name = format!("cache_{label}_a_keeps_every_held_position_after_the_switch");
        if !drafted || tier::run_clause(&name, Tag::FileBound)? {
            check(
                &mut ok,
                &name,
                r1.ok && a1.ok && b1.ok && a2.ok && !a1.tokens.is_empty() && kept,
            );
        }
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
            c1.ok && b2.ok && a3.ok && a3.cache_n == turn_end,
        );
        check(
            &mut ok,
            &format!("cache_{label}_b_ids_are_a_fresh_runs"),
            f3.ok && f3.cache_n == 0 && !a3.tokens.is_empty() && a3.tokens == f3.tokens,
        );
        // PIN(2026-10-12): each cut keeps its point with the draft rejoined (round kp-draft): the
        // first resend's and the stripped resend's at the turn's end, the switched resend's at
        // all but its last id; was each weighed against the break-even, kept with the draft off
        // by name at or past it, else a reset by the rule. The check keeps its name.
        let named = if drafted {
            r2.rejoined_at(turn_end, resend.len())
                && a2.rejoined_at(all_but_last, resend.len())
                && a3.rejoined_at(turn_end, strip.len())
                && f3.draft_n > 0
                && f3.offs.is_empty()
        } else {
            [&r2, &a2, &a3, &f3].iter().all(|g| g.offs.is_empty())
        };
        let name = format!("cache_{label}_cuts_leave_the_draft_off_by_name");
        if !drafted || tier::run_clause(&name, Tag::FileBound)? {
            check(&mut ok, &name, named);
        }
        if let Some(k) = keep38 {
            ok &= break_even(url, err_log, k, &pb)?;
        }
        Ok(ok)
    }

    /// The drafted seat's break-even at a turn's end, as the `cache`
    /// clause's (b) reaches it (the turn, B's turn, the turn resent with its
    /// reply stripped: its state put back as the prefix to the turn's end):
    /// the draft rejoins at the turn's end, so the break-even weighs nothing
    /// on either side of it ([`Q38::draft_keep`]). A reply short enough that
    /// the turn's end is at or past its break-even and one whose break-even
    /// is past it each keep the turn's end with the draft rejoined
    /// ([`Greedy::rejoined_at`]) and give the ids of the same request fed
    /// fresh. Each side has a session of its own ([`BREAK_EVEN_TURNS`]), so
    /// the prompt cache holds no state that shares more than its turn. The
    /// replies are the seat's own arithmetic on its `draft keep` line: the
    /// longest of at most [`N_PREDICT`] tokens whose break-even the turn's
    /// end reaches, and [`N_PREDICT`], whose break-even it must not.
    /// Mutant: the put-back that turns the draft off (the long side keeps
    /// the turn's end with the draft off by name, the short side resets).
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
                // PIN(2026-10-12): both sides keep the turn's end with the draft rejoined (round
                // kp-draft); was the long side's turn's end kept with the draft off by name (a
                // `kept` record), the short side's a reset by the rule (a `reset` record). The
                // checks keep their names.
                let branch = s.rejoined_at(turn_end as u64, strip.len());
                let fresh = a.ok && b.ok && f.ok && f.cache_n == 0 && f.offs.is_empty();
                let same = !s.tokens.is_empty() && s.tokens == f.tokens;
                println!(
                    "break-even {side}: the turn's end {turn_end}, a reply of {n} tokens weighed as {} \
                 against the break-even {}: kept with the draft rejoined {}",
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

    /// The value of `key` on one of the seat's `key=value` lines.
    fn word_of<'a>(line: &'a str, key: &str) -> Option<&'a str> {
        line.split_whitespace()
            .find_map(|w| w.strip_prefix(key)?.strip_prefix('='))
    }

    /// The `parallel` line's rule against its own terms (the module header):
    /// under `slots` the line's slots are the flag's (`--parallel`'s value,
    /// the caller naming it) or the default's (`ctx_ask::DEFAULT_SLOTS`), and the split the line
    /// names holds — `slot_ctx` the one context the whole round of clauses
    /// pins a slot at (the server's `--ctx-size` total over the slots) and
    /// `total` the slots' sum. `None` a line that is not the seat's;
    /// `Some(false)` a rule whose terms do not hold.
    fn parallel_agrees(line: &str, flag: Option<usize>) -> Option<bool> {
        let field = |k: &str| word_of(line, k);
        let rule = field("rule")?;
        let slots: usize = field("slots")?.parse().ok()?;
        let slot_ctx: usize = field("slot_ctx")?.parse().ok()?;
        let total: usize = field("total")?.parse().ok()?;
        let want = match (rule, flag) {
            ("slots", f) => f.unwrap_or(model::placement::ctx_ask::DEFAULT_SLOTS),
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

    /// One streamed request already posted, run to its end on its own thread
    /// ([`answers`]): its answer, or `None` a request that failed (a red
    /// check, not the clause's end).
    fn answer(
        clause: &str,
        run: (JoinHandle<()>, StreamedRx),
        what: &str,
    ) -> Result<Option<Streamed>, GateError> {
        Ok(answers(clause, vec![(run, what)])?.remove(0))
    }

    /// `/completion` of `ids` sampled at [`SAMPLED_TEMPERATURE`] with
    /// [`SAMPLED_SEED`], streamed ([`streamed_with`]), banning the end ids
    /// when `ignore_eos`, asking `n_probs` probabilities a token (0 asks
    /// none): its ids a function of its logits rows and the seed alone (the
    /// server seeds each request's sampler from its own `seed`). It steps on
    /// a load that drafts nothing, and on a drafted load while it asks
    /// probabilities; else on a drafted load it drafts, banning or not,
    /// alone in its round or beside another busy slot.
    fn sampled_stream(
        addr: &str,
        ids: &[u32],
        n: usize,
        ignore_eos: bool,
        n_probs: usize,
    ) -> Result<(JoinHandle<()>, StreamedRx), GateError> {
        streamed_with(
            addr,
            json!({
                "prompt": ids, "n_predict": n, "temperature": SAMPLED_TEMPERATURE,
                "seed": SAMPLED_SEED, "ignore_eos": ignore_eos, "return_tokens": true,
                "cache_prompt": false, "stream": true, "n_probs": n_probs,
            }),
        )
    }

    /// The answers of streamed requests already posted, in order, each
    /// thread joined: `None` a request that failed (a red check, not the
    /// clause's end), named in a line under `clause`.
    fn answers(
        clause: &str,
        runs: Vec<((JoinHandle<()>, StreamedRx), &str)>,
    ) -> Result<Vec<Option<Streamed>>, GateError> {
        let mut out = Vec::new();
        for ((h, rx), what) in runs {
            h.join()
                .map_err(|_| format!("{clause}: the {what} request's thread panicked"))?;
            let ran = rx
                .recv()
                .map_err(|_| format!("{clause}: the {what} request's thread gave no answer"))?;
            match ran {
                Ok(s) => out.push(Some(s)),
                Err(e) => {
                    println!("{clause}: the {what} request failed: {e}");
                    out.push(None);
                }
            }
        }
        Ok(out)
    }

    /// Whether every request of `together` answered ids, each the ids of the
    /// same request in `alone`.
    fn ids_as_alone<'a>(
        alone: impl ExactSizeIterator<Item = &'a Option<Streamed>>,
        together: &[Option<Streamed>],
    ) -> bool {
        alone.len() == together.len()
            && alone.zip(together).all(|(a, t)| {
                a.as_ref()
                    .zip(t.as_ref())
                    .is_some_and(|(a, t)| !t.tokens.is_empty() && a.tokens == t.tokens)
            })
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
    /// into `<dir>/slots-default`, is the seat under its own defaults but the
    /// draft at `--parallel 2` (`--place a`, the residency unset and
    /// `BLOOMERY_DRAFT=mtp`: the A6000 stages, the adaptive residency and the
    /// draft run), and runs the together pair once
    /// ([`slots_default_residency`]).
    // PIN(2026-10-05): the swap clause (`swap_rejoins_the_draft`) is removed — the
    /// turn path it checked (`serve::SwapEngine`, the park of a preempted request's
    /// state) left the seat with the resident slots, and its coverage (a preempted
    /// request's draft rejoining) is this clause's (v1) draft-count equality.
    fn slots_flow_together(
        dir: &Path,
        levers: &bloomery_levers::Levers,
    ) -> Result<bool, GateError> {
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
                    "slots_parallel_line_names_one_pass",
                    "slots_load_names_the_split",
                    "slots_alone_ids_stay",
                    "slots_alone_draft_counts_stay",
                    "slots_rounds_carry_both_slots",
                    "slots_rounds_of_both_run_one_pass",
                    "slots_drafted_off_slot_steps_beside_a_window",
                    "slots_drafted_late_request_together_alone",
                    "slots_drafted_late_request_other_order_together_alone",
                    "slots_drafted_sampled_rounds_run_one_pass",
                    "slots_drafted_sampled_together_alone",
                    "slots_drafted_probs_rounds_run_one_pass",
                    "slots_drafted_probs_together_alone",
                    "slots_drafted_sampled_tail_drafts",
                ] {
                    check(&mut ok, name, false);
                }
                let _ = served.stop();
                return slots_default_residency(dir, levers, &mut ok);
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
        // The round's shape the seat's open runs (`--place gate`: no expert
        // tier card, so the slots step as one pass).
        check(
            &mut ok,
            "slots_parallel_line_names_one_pass",
            word_of(&line, "pass") == Some("one"),
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
            alone.push(answer("slots", streamed(&addr, ids, SLOTS_PREDICT)?, what)?);
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
        let runs = vec![
            (streamed(&addr, &a_ids, SLOTS_PREDICT)?, "together A"),
            (streamed(&addr, &b_ids, SLOTS_PREDICT)?, "together B"),
        ];
        let together = answers("slots", runs)?;
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
            ids_as_alone(alone.iter(), &together),
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
        ok &= slots_drafted_sampled(&addr, &err_log, [&a_ids, &b_ids])?;
        println!("slots server stopped: {}", served.stop()?);
        slots_default_residency(dir, levers, &mut ok)
    }

    /// The slots clause's second server (the module header): the seat under
    /// its own defaults at `--parallel 2` but the draft (no `--place`, no
    /// residency or width lever, `BLOOMERY_DRAFT=mtp` set, so the clause holds
    /// the drafted load whatever the unset draft's context rule picks) runs the
    /// together pair once and holds (v2) and (v3)
    /// alone; under the adaptive residency a stream's bits need not equal
    /// its alone run's, so no id equality is asked. The recipe leaves the
    /// box env file's CUDA_VISIBLE_DEVICES pin (the 3090) in place, so the
    /// seat sees one card (under the gate card's budget,
    /// [`gate_card_budget`]): the unset rule's `place unset` record names it
    /// (`unset_place_is_the_rule`), and the server's load waits out a timing
    /// sitting by the box guard this gate runs under. The seat's own wiring
    /// of the two machines it runs by default is read here too: the
    /// residency it opens is the one the unset rule picked and its
    /// boundaries print `residency pass` records after the requests; its
    /// reset route answers 200 and prints its `residency reset` record;
    /// and, after a one-id flush, the draft's width chooser (`cost`, the
    /// unset lever) prints its `mtp width` record as the chooser's own
    /// aggregate. The fixture tier defers that last clause by name: the
    /// draft is the real file's, which proposes nothing a random-weight
    /// target accepts.
    fn slots_default_residency(
        dir: &Path,
        levers: &bloomery_levers::Levers,
        ok: &mut bool,
    ) -> Result<bool, GateError> {
        let own = dir.join("slots-default");
        std::fs::create_dir_all(&own)?;
        let err_log = own.join("server.err");
        // PIN(2026-10-23): the draft is set, where this server took the unset draft, so it
        // keeps the drafted residency, rounds and chooser it holds by the set word whatever
        // the unset draft's context rule picks; the unset rule's two-slot answer is
        // `draft_unset_drafts_on_two_slots`.
        let mut cmd = Command::new(Served38::exe()?);
        cmd.env(bloomery_levers::DRAFT, "mtp")
            .env_remove(bloomery_levers::RESIDENCY)
            .env_remove(bloomery_levers::MTP_HEAD_ROWS)
            .env_remove(bloomery_levers::MTP_DRAFT)
            .env_remove(bloomery_levers::MTP_WIDTH)
            .env(bloomery_levers::STEP_STATS, "1");
        // PIN(2026-10-25): the drafting load's budget at the two slots, where this server took the
        // fixture runner's, the target's alone: the set draft's card bytes (PIN above) leave the
        // least default context of two slots no room beside it, so the seat refused by name.
        drafted_card_budget(&mut cmd, levers, SLOTS_DEFAULT)?;
        gate_card_budget(&mut cmd, levers)?;
        let parallel = SLOTS_DEFAULT.to_string();
        let args = [
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--parallel",
            parallel.as_str(),
        ];
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
        let runs = vec![
            (streamed(&addr, &a_ids, SLOTS_PREDICT)?, "default A"),
            (streamed(&addr, &b_ids, SLOTS_PREDICT)?, "default B"),
        ];
        let together = answers("slots-default", runs)?;
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
        // slot that holds its ids. A round of both slots runs the widths the
        // round's timings choose and is counted at the rows it ran
        // ([`Choosing::round_ran`]): whether a run cuts a window depends on
        // those timings, so the cut is [`width_cost`]'s clause. FAIL-first: a
        // seat that runs the draft with no chooser prints no record; a record
        // off by a window turns the sum red.
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
        if tier::run_clause("slots_default_width_record_is_the_choosers", Tag::FileBound)? {
            check(
                ok,
                "slots_default_width_record_is_the_choosers",
                adaptive
                    && flushed
                    && !parsed.is_empty()
                    && parsed.iter().any(|(w, _, _)| *w > 0)
                    && parsed.iter().all(|(w, widths, e)| {
                        *w == widths.iter().skip(1).sum::<u64>()
                            && (*w == 0 || (1.0..=4.0).contains(e))
                    }),
            );
        }
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

    /// What the unhosted-pool clause reads off its plan-only server
    /// ([`churn_plan`]): the plan's host need, its host headroom, and the
    /// churn pool at the word's P, which is 0.
    struct ChurnRoom {
        /// The plan's host need (`HostNeed::bytes`), the `cache` line's.
        need: u64,
        /// The plan's host headroom at the plan-only room, the `residency
        /// host` record's: the machine's figure less the need, that room
        /// being past the figure.
        headroom: i64,
        /// The pool's bytes at P 0.
        churn: u64,
    }

    impl ChurnRoom {
        /// The room that holds the plan's need and half the pool: the
        /// unset rule reads the host's available bytes past the need
        /// (`BLOOMERY_HOST_ROOM` replaces that reading) and the plan's
        /// headroom takes the room too, so the pool finds half of what it
        /// takes, whatever the file and the card.
        fn room(&self) -> u64 {
            self.need + self.churn / 2
        }

        /// The plan's host headroom under [`ChurnRoom::room`]: the tighter
        /// of the machine's, read off the plan-only run, and what the room
        /// leaves past the need (`Plan::plan_at`).
        fn headroom_under_room(&self) -> i128 {
            i128::from(self.headroom).min(i128::from(self.room() - self.need))
        }
    }

    /// `cmd`'s `BLOOMERY_CARD_BUDGET` on any card where neither `cmd`
    /// nor this process's `levers` set one: the gate card's usable bytes
    /// ([`Place::Gate`], the 3090's on the card in view, as `gate_card` plans
    /// them), so a flagless server's `a` holds part of the routed experts on
    /// the card, as the 3090 does, even when the card's own bytes could hold
    /// every routed expert. On a unified card the set budget also keeps the
    /// plan off the pool's budget search (`plan_in_pool` lowers only an unset
    /// one): these servers hold the 3090's split, and the pool's search is the
    /// [`paged`] clause's.
    fn gate_card_budget(
        cmd: &mut Command,
        levers: &bloomery_levers::Levers,
    ) -> Result<(), GateError> {
        let gate = Place::Gate.on(census()?)?.card_specs()?[0];
        let set = match cmd
            .get_envs()
            .find(|&(k, _)| k == bloomery_levers::CARD_BUDGET)
        {
            Some((_, v)) => v.is_some(),
            None => levers.card_budget().is_set(),
        };
        if !set {
            cmd.env(
                bloomery_levers::CARD_BUDGET,
                gate.usable_bytes().to_string(),
            );
        }
        Ok(())
    }

    /// A server of the unhosted-pool clause under `room` bytes of host, the
    /// residency word `word` (unset for `None`), on any card under the
    /// gate card's budget ([`gate_card_budget`]).
    fn churn_server(
        room: u64,
        word: Option<&str>,
        levers: &bloomery_levers::Levers,
    ) -> Result<Command, GateError> {
        let mut cmd = Command::new(Served38::exe()?);
        match word {
            Some(word) => cmd.env(bloomery_levers::RESIDENCY, word),
            None => cmd.env_remove(bloomery_levers::RESIDENCY),
        };
        cmd.env(bloomery_levers::DRAFT, "off")
            .env_remove(bloomery_levers::NVTIER_BYTES)
            .env_remove(bloomery_levers::MTP_HEAD_ROWS)
            .env_remove(bloomery_levers::MTP_DRAFT)
            .env_remove(bloomery_levers::MTP_WIDTH)
            .env(bloomery_levers::HOST_ROOM, room.to_string());
        gate_card_budget(&mut cmd, levers)?;
        Ok(cmd)
    }

    /// A server of the unhosted-pool clause stopped at `--plan` under `room`
    /// bytes of host, the residency word unset, into `own`: its error log and
    /// that log's records, or why it printed none, as text.
    fn churn_plan_only(
        own: &Path,
        room: u64,
        levers: &bloomery_levers::Levers,
    ) -> Result<Result<(PathBuf, record::Log), String>, GateError> {
        std::fs::create_dir_all(own)?;
        let err_log = own.join("server.err");
        let mut cmd = churn_server(room, None, levers)?;
        let args: Vec<&str> = CHURN_ARGS.iter().copied().chain(["--plan"]).collect();
        let mut served = Served38::spawn_with(&args, own, &mut cmd)?;
        println!("churn-unhosted plan server pid {}", served.child.id());
        let status = served.wait_exit()?;
        let Some(status) = status else {
            println!(
                "churn-unhosted: the plan-only server stopped: {}",
                served.stop()?
            );
            return Ok(Err(format!(
                "the plan-only server did not exit in {POLLS} s"
            )));
        };
        let text = std::fs::read_to_string(&err_log)?;
        if !status.success() {
            return Ok(Err(format!("the plan-only server exited {status}: {text}")));
        }
        let log = seat_log(&err_log)?;
        Ok(Ok((err_log, log)))
    }

    /// The unhosted-pool clause's first server, stopped at `--plan` under
    /// [`CHURN_PLAN_ROOM`], into `own`: the plan's host need, headroom and
    /// churn pool ([`ChurnRoom`]), or why they cannot be read, as text. The
    /// pool must be hosted at P 0 on this room and non-empty, and its `cache`
    /// line must name the room asked and the same pool, so the terms are the
    /// plan the second server loads.
    fn churn_plan(
        own: &Path,
        levers: &bloomery_levers::Levers,
    ) -> Result<Result<ChurnRoom, String>, GateError> {
        let (err_log, log) = match churn_plan_only(own, CHURN_PLAN_ROOM, levers)? {
            Ok(ran) => ran,
            Err(why) => return Ok(Err(why)),
        };
        let Some(host) = log.first(&record::RESIDENCY_HOST)? else {
            let unset = log
                .first(&record::RESIDENCY_UNSET)?
                .map(|u| u.text("why").map(str::to_owned))
                .transpose()?;
            return Ok(Err(format!(
                "the plan-only server printed no `residency host` record: the unset word runs \
                 no machine here (residency unset why {unset:?})"
            )));
        };
        let (hosted, pinned, churn) = (
            host.word("hosted")?.to_owned(),
            host.u64("pinned")?,
            host.u64("churn_bytes")?,
        );
        if hosted != "hosted" || pinned != 0 || churn < 2 {
            return Ok(Err(format!(
                "the pool must be hosted at P 0 and not empty on a room of {CHURN_PLAN_ROOM} B: \
                 residency host hosted={hosted} pinned={pinned} churn_bytes={churn}"
            )));
        }
        let Some(cache) = record_line(&err_log, "cache")? else {
            return Ok(Err(
                "the plan-only server printed no `cache` line".to_owned()
            ));
        };
        let term = |k: &str| word_of(&cache, k).and_then(|v| v.parse::<u64>().ok());
        let (Some(available), Some(need), Some(pool)) =
            (term("available"), term("need"), term("pool"))
        else {
            return Ok(Err(format!(
                "the `cache` line has no available, need and pool terms: {cache}"
            )));
        };
        if available != CHURN_PLAN_ROOM || pool != churn || need >= available {
            return Ok(Err(format!(
                "the `cache` line is not the plan the pool was read from: {cache} (room asked \
                 {CHURN_PLAN_ROOM} B, churn pool {churn} B)"
            )));
        }
        Ok(Ok(ChurnRoom {
            need,
            headroom: host.i64("headroom")?,
            churn,
        }))
    }

    /// The unhosted-pool clause (the module header), both tiers: a plan-only
    /// server gives the plan's host need and churn pool ([`churn_plan`]); a
    /// second server runs under the room that holds the need and half the
    /// pool, loads, and answers one greedy request of [`CHURN_PREDICT`] ids.
    /// The word asks for P 0 with the pool unhosted ([`CHURN_WORD`]): its
    /// `residency lever` record names that word with why `set`
    /// ([`CHURN_WHY`]), its `residency host` record carries the same word and
    /// reads `hosted=unhosted` at `pinned=0` ([`CHURN_HOST`]), the request
    /// completes ([`CHURN_REQUEST`]), and the `residency pass` records count
    /// a victim served from the file ([`CHURN_VICTIMS`]). The unset word on
    /// the same room climbs ([`churn_unset_climbs`], [`CHURN_CLIMBS`]).
    /// FAIL-first: the module header's mutants.
    fn churn_unhosted(dir: &Path, levers: &bloomery_levers::Levers) -> Result<bool, GateError> {
        let mut ok = true;
        let room = match churn_plan(&dir.join("churn-plan"), levers)? {
            Ok(room) => room,
            Err(why) => {
                println!("churn-unhosted: no room can be derived: {why}");
                for name in [
                    CHURN_WHY,
                    CHURN_HOST,
                    CHURN_REQUEST,
                    CHURN_VICTIMS,
                    CHURN_CLIMBS,
                ] {
                    check(&mut ok, name, false);
                }
                return Ok(ok);
            }
        };
        let asked = room.room();
        println!(
            "churn-unhosted: room asked {CHURN_PLAN_ROOM} B; the plan's host need {} B, host \
             headroom {} B (the machine's: that room is past its figure), churn pool at P 0 \
             {} B; derived room {asked} B = need + pool / 2, host headroom there {} B",
            room.need,
            room.headroom,
            room.churn,
            room.headroom_under_room()
        );
        let own = dir.join("churn-unhosted");
        std::fs::create_dir_all(&own)?;
        let err_log = own.join("server.err");
        let mut cmd = churn_server(asked, Some(CHURN_WORD), levers)?;
        let mut served = Served38::spawn_with(&CHURN_ARGS, &own, &mut cmd)?;
        println!("churn-unhosted server pid {}", served.child.id());
        let completed = match served.address(&err_log, POLLS, POLL) {
            Ok(addr) => {
                let url = |p: &str| format!("http://{addr}{p}");
                let ids = rendered(&url, json!([{ "role": "user", "content": TURN_A }]))?;
                let body = json!({
                    "prompt": ids, "n_predict": CHURN_PREDICT, "temperature": 0,
                    "return_tokens": true, "cache_prompt": false,
                });
                let (st, reply) = curl(&url("/completion"), Some(&body), false)?;
                let tokens = json_of("/completion", st, &reply)
                    .map(|c| ids_of(&c["tokens"]))
                    .unwrap_or_default();
                println!(
                    "churn-unhosted: the request: HTTP {st}, {} ids",
                    tokens.len()
                );
                st == 200 && tokens.len() >= CHURN_MIN_TOKENS
            }
            Err(e) => {
                println!("churn-unhosted: the server never listened: {e}");
                false
            }
        };
        if served.child.try_wait()?.is_none() {
            println!("churn-unhosted server stopped: {}", served.stop()?);
        }
        let text = std::fs::read_to_string(&err_log)?;
        let log = seat_log(&err_log)?;
        // A set word prints the lever record: the word as asked, why `set`.
        let lever = log
            .first(&record::RESIDENCY_LEVER)?
            .map(|l| -> Result<(String, String), ReadError> {
                Ok((l.word("residency")?.to_owned(), l.word("why")?.to_owned()))
            })
            .transpose()?;
        println!("churn-unhosted: residency lever (word, why) {lever:?}, asked {CHURN_WORD:?}");
        check(
            &mut ok,
            CHURN_WHY,
            lever
                .as_ref()
                .is_some_and(|(word, why)| word == CHURN_WORD && why == "set"),
        );
        let host = log
            .first(&record::RESIDENCY_HOST)?
            .map(|h| -> Result<(String, String, u64, u64), ReadError> {
                Ok((
                    h.word("residency")?.to_owned(),
                    h.word("hosted")?.to_owned(),
                    h.u64("pinned")?,
                    h.u64("churn_bytes")?,
                ))
            })
            .transpose()?;
        println!("churn-unhosted: residency host (word, hosted, pinned, churn_bytes) {host:?}");
        check(
            &mut ok,
            CHURN_HOST,
            host.as_ref().is_some_and(|(word, hosted, pinned, bytes)| {
                word == CHURN_WORD && hosted == "unhosted" && *pinned == 0 && *bytes > 0
            }),
        );
        check(&mut ok, CHURN_REQUEST, completed);
        let (mut rereads, mut unresident, mut picked, mut landed) = (0u64, 0u64, 0u64, 0u64);
        let passes = log.all(&record::RESIDENCY_PASS)?;
        for pass in &passes {
            rereads = rereads.saturating_add(pass.u64("rereads")?);
            unresident = unresident.saturating_add(pass.opt_u64("unresident")?.unwrap_or(0));
            picked = picked.saturating_add(pass.opt_u64("picked")?.unwrap_or(0));
            landed = landed.saturating_add(pass.u64("landed")?);
        }
        let refused = text.contains(UNRESIDENT_REFUSED);
        println!(
            "churn-unhosted: {} residency passes, rereads {rereads}, unresident {unresident}, a \
             refused flip in the log: {refused}",
            passes.len()
        );
        // The premise line names a missed setup apart from a victim the engine did not serve
        // from the file; the verdict below is the same either way.
        println!(
            "churn-unhosted: premise: {picked} flips picked by the prompt call, {landed} landed; \
             {} of their victims were outside the page cache",
            rereads.saturating_add(unresident)
        );
        if picked == 0 {
            println!("churn-unhosted: setup missed: the prompt call picked no flip");
        } else if rereads.saturating_add(unresident) == 0 {
            println!(
                "churn-unhosted: setup missed: every picked victim was page-cache resident \
                 (another process maps the model file, or the load's page drop did not hold)"
            );
        }
        check(
            &mut ok,
            CHURN_VICTIMS,
            rereads.saturating_add(unresident) > 0 && !refused,
        );
        churn_unset_climbs(dir, &room, levers, &mut ok)?;
        Ok(ok)
    }

    /// The unset word on the unhosted-pool clause's room ([`ChurnRoom::room`]:
    /// the plan's need and half the pool), a plan-only server: the pool does
    /// not fit, so the `residency unset` record names the climb (`p0→p<q>`,
    /// the first P whose pool fits), and the `residency host` record carries
    /// the climbed word at `pinned=<q>` with the pool hosted, whatever the
    /// body carries ([`CHURN_CLIMBS`]). FAIL-first: an unset word that keeps
    /// P 0 with the pool unhosted prints `p0` kept and `hosted=unhosted`.
    fn churn_unset_climbs(
        dir: &Path,
        room: &ChurnRoom,
        levers: &bloomery_levers::Levers,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let log = match churn_plan_only(&dir.join("churn-climbs"), room.room(), levers)? {
            Ok((_, log)) => log,
            Err(why) => {
                println!("churn-unset-climbs: {why}");
                check(ok, CHURN_CLIMBS, false);
                return Ok(());
            }
        };
        let why = log
            .first(&record::RESIDENCY_UNSET)?
            .map(|u| u.text("why").map(str::to_owned))
            .transpose()?;
        let host = log
            .first(&record::RESIDENCY_HOST)?
            .map(|h| -> Result<(String, String, u64), ReadError> {
                Ok((
                    h.word("residency")?.to_owned(),
                    h.word("hosted")?.to_owned(),
                    h.u64("pinned")?,
                ))
            })
            .transpose()?;
        println!("churn-unset-climbs: residency unset why {why:?}, residency host {host:?}");
        // `unset: p0→p<q> (host: the churn pool did not fit, raised)`.
        let climbed = why.as_deref().and_then(|w| {
            let q = w.strip_prefix("unset: p0→p")?;
            let (q, rest) = q.split_once(' ')?;
            (rest == "(host: the churn pool did not fit, raised)")
                .then(|| q.parse::<u64>().ok())
                .flatten()
        });
        check(
            ok,
            CHURN_CLIMBS,
            climbed.is_some_and(|q| {
                q > 0
                    && host.as_ref().is_some_and(|(word, hosted, pinned)| {
                        hosted == "hosted" && *pinned == q && *word == format!("mid-p{q}-s1")
                    })
            }),
        );
        Ok(())
    }

    /// The drafted clauses beside the slots clause's pair, on its server
    /// (the draft on, the residency off, the fixed window, the round records
    /// on) after its alone and together runs, `alone` the alone answers of
    /// `ids`: the cut slot ([`slots_drafted_off_slot`]), then the late
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
    /// off steps beside the pass. The mixed
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
            check(&mut ok, name, ids_as_alone(want.into_iter(), &late));
        }
        Ok(ok)
    }

    /// A sampled pair's rounds (the sampled rounds clauses, the module
    /// header) on a `--parallel 2` server that prints its `slots round`
    /// records: the sampled streamed requests of `ids`, banning the end ids
    /// when `ignore_eos` and asking `n_probs` probabilities a token, each run
    /// alone and then both together. Over the together run every record is
    /// one `cmd` pass of both slots — `step` where they step, `pass` where
    /// they draft (`<name>_rounds_run_one_pass`), and each request's together ids are
    /// its alone ones (`<name>_together_alone`): each row of the pass and its
    /// logits bit for bit its step alone, each answer and logits row its own
    /// request's. The alone runs print no record (a round of one active
    /// request is a select and a call).
    fn sampled_pair(
        name: &str,
        addr: &str,
        err_log: &Path,
        ids: [&[u32]; 2],
        (ignore_eos, n_probs): (bool, usize),
        cmd: &str,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let alone = [
            answer(
                name,
                sampled_stream(addr, ids[0], SLOTS_PREDICT, ignore_eos, n_probs)?,
                "alone A",
            )?,
            answer(
                name,
                sampled_stream(addr, ids[1], SLOTS_PREDICT, ignore_eos, n_probs)?,
                "alone B",
            )?,
        ];
        let before = seat_log(err_log)?.all(&record::SLOTS_ROUND)?.len();
        let runs = vec![
            (
                sampled_stream(addr, ids[0], SLOTS_PREDICT, ignore_eos, n_probs)?,
                "together A",
            ),
            (
                sampled_stream(addr, ids[1], SLOTS_PREDICT, ignore_eos, n_probs)?,
                "together B",
            ),
        ];
        let together = answers(name, runs)?;
        println!(
            "{name}: alone {:?} ids, together {:?} ids",
            alone
                .iter()
                .map(|s| s.as_ref().map(|s| s.tokens.len()))
                .collect::<Vec<_>>(),
            together
                .iter()
                .map(|s| s.as_ref().map(|s| s.tokens.len()))
                .collect::<Vec<_>>(),
        );
        rounds_carry_both_in_one_pass(
            &mut ok,
            &format!("{name}_rounds_run_one_pass"),
            err_log,
            before,
            Some(cmd),
        )?;
        check(
            &mut ok,
            &format!("{name}_together_alone"),
            ids_as_alone(alone.iter(), &together),
        );
        Ok(ok)
    }

    /// The drafted sampled clauses (the module header) on the slots clause's
    /// server, after [`slots_drafted_beside`]: the sampled pair of `ids`
    /// banning the end ids, which draws in drafted passes alone and together
    /// on this drafted load, every together round one `pass` of both slots'
    /// windows ([`sampled_pair`], `slots_drafted_sampled_*`); the same pair
    /// asking a probability a token, which drafts on this load too, every
    /// together round one `pass` of both windows (`slots_drafted_probs_*`);
    /// then the pair with no ban,
    /// `ids[1]` asking [`DRAFTED_SAMPLED_SHORT`] ids, so `ids[0]` drafts
    /// beside it and then alone from the state its windows in the shared
    /// passes left its draft: its proposals an id past the short request's
    /// are at least half its alone run's an id
    /// (`slots_drafted_sampled_tail_drafts`), and some are accepted
    /// (`slots_drafted_sampled_tail_accepts`, which the fixture tier defers).
    /// Refused by name before either check unless the short request answered
    /// its [`DRAFTED_SAMPLED_SHORT`] ids and a `slots round` record of the
    /// tail run is one `pass` of both slots: the tail checks hold only for a
    /// long request whose window rode a pass beside the short one.
    fn slots_drafted_sampled(
        addr: &str,
        err_log: &Path,
        ids: [&[u32]; 2],
    ) -> Result<bool, GateError> {
        let mut ok = sampled_pair(
            "slots_drafted_sampled",
            addr,
            err_log,
            ids,
            (true, 0),
            "pass",
        )?;
        // PIN(2026-11-01): probabilities are collected from verified rows,
        // so asking for them no longer forces plain steps.
        ok &= sampled_pair("slots_drafted_probs", addr, err_log, ids, (true, 1), "pass")?;
        let alone = answer(
            "slots-drafted-sampled",
            sampled_stream(addr, ids[0], SLOTS_PREDICT, false, 0)?,
            "alone long",
        )?;
        let before = seat_log(err_log)?.all(&record::SLOTS_ROUND)?.len();
        let runs = vec![
            (
                sampled_stream(addr, ids[0], SLOTS_PREDICT, false, 0)?,
                "tail long",
            ),
            (
                sampled_stream(addr, ids[1], DRAFTED_SAMPLED_SHORT, false, 0)?,
                "tail short",
            ),
        ];
        let mut tail = answers("slots-drafted-sampled", runs)?;
        let short = tail
            .pop()
            .flatten()
            .ok_or("sampled tail: short request failed")?;
        if short.tokens.len() != DRAFTED_SAMPLED_SHORT {
            return Err("sampled tail: short request ended before its requested length".into());
        }
        let rounds = seat_log(err_log)?.all(&record::SLOTS_ROUND)?;
        let shared = rounds.iter().skip(before).any(|r| {
            r.word("cmd") == Ok("pass") && r.u64("rows") == Ok(2) && r.u64("passes") == Ok(1)
        });
        if !shared {
            return Err("sampled tail: no shared pass carried the long request's window".into());
        }
        let tail = tail.remove(0);
        // Proposals an id the request could draft: every id alone, the ids
        // past the short request's on the tail.
        let per_id = |s: &Streamed, from: usize| {
            (s.tokens.len() > from).then(|| s.drafts.0 as f64 / (s.tokens.len() - from) as f64)
        };
        let (a, t) = (
            alone.as_ref().and_then(|s| per_id(s, 0)),
            tail.as_ref().and_then(|s| per_id(s, DRAFTED_SAMPLED_SHORT)),
        );
        println!(
            "slots-drafted-sampled: alone {:?} (ids, drafted, accepted), beside the short one \
             {:?}; proposals an id {a:?} and {t:?}",
            alone.as_ref().map(|s| (s.tokens.len(), s.drafts)),
            tail.as_ref().map(|s| (s.tokens.len(), s.drafts)),
        );
        check(
            &mut ok,
            "slots_drafted_sampled_tail_drafts",
            a.zip(t).is_some_and(|(a, t)| a > 0.0 && 2.0 * t >= a),
        );
        if tier::run_clause("slots_drafted_sampled_tail_accepts", Tag::FileBound)? {
            check(
                &mut ok,
                "slots_drafted_sampled_tail_accepts",
                tail.is_some_and(|s| s.drafts.1 > 0),
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
        let late = answers(
            "slots-drafted late",
            vec![(first, what[0]), (second, what[1])],
        )?;
        Ok((polls, late))
    }

    /// The cut slot clause (the module header) on the slots clause's
    /// server: the break-even clause's long turn (`BREAK_EVEN_TURNS`'
    /// first) run fresh on a free slot; then `TURN_A`'s drafted request
    /// (`a_ids`, [`SLOTS_PREDICT`] tokens) posted (it takes the other slot,
    /// whose held ids share more of its prompt), and once it has run
    /// [`DRAFTED_LATE_ROUNDS`] decode calls, the turn resent with its reply
    /// stripped, which only the turn's slot is free to take: that slot keeps
    /// the turn's end by a cut with its draft rejoined there
    /// ([`Greedy::rejoined_at`]), and its windows run in the rounds the
    /// drafted request's windows run in (`slots round` records of two rows).
    /// The cut slot's ids are the same ids fed fresh, and the drafted
    /// request's ids its alone run's (`alone_a`)
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
        let run = streamed(addr, a_ids, SLOTS_PREDICT)?;
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
        let drafted = answer("slots-drafted off", run, "drafted")?;
        let f = greedy(url, err_log, &strip, n, false)?;
        f.show("slots-drafted off: the same ids fresh");
        // PIN(2026-10-12): the cut rejoins the slot's draft at the turn's end (round kp-draft),
        // so the slot drafts beside the drafted request and both slots' windows share rounds; the
        // drafted request's draft counts are no longer its alone run's (a pass of two drafted
        // slots caps each window, `app::mtp::SLOT_DEPTH`), its ids still are; was the slot's
        // draft off by name (a `kept` `mtp keep` record), its plain steps beside the drafted
        // windows, the drafted request's counts its alone run's. The check keeps its name.
        let off = cut.rejoined_at(turn_end as u64, strip.len());
        let fresh = x.ok && f.ok && f.cache_n == 0 && f.offs.is_empty();
        let off_same = !cut.tokens.is_empty() && cut.tokens == f.tokens;
        let drafted_same = drafted
            .as_ref()
            .zip(alone_a)
            .is_some_and(|(d, a)| !d.tokens.is_empty() && d.tokens == a.tokens);
        println!(
            "slots-drafted off: posted after {polls} poll(s); the cut kept {} of the turn's end \
             {turn_end}, the draft rejoined {off}, {beside} round(s) of two rows beside the \
             drafted request; cut ids the fresh run's: {off_same}; drafted ids its alone run's: \
             {drafted_same}",
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

    /// The shared-prefix clause ([`prefix_restore`]) on a server of its own at
    /// [`PREFIX_CTX`]: prompt A of [`PREFIX_A`] ids from the gate's turn, B
    /// another turn, and C sharing [`PREFIX_SHARED`] of A's ids keeps
    /// [`PREFIX_KEPT`] of them from the host cache (the checkpoint at the
    /// first spacing multiple, which the saved state carries beside its last
    /// one), its ids a fresh run's.
    ///
    /// Mutant: a save that carries its last checkpoint alone (C keeps 0).
    fn prefix(dir: &Path) -> Result<bool, GateError> {
        let own = dir.join("prefix");
        std::fs::create_dir_all(&own)?;
        let err_log = own.join("server.err");
        let ctx = PREFIX_CTX.to_string();
        let args = [
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--place",
            "gate",
            "--ctx-size",
            &ctx,
            "--slot-save-path",
            "/tmp",
            "--parallel",
            "1",
        ];
        let mut cmd = Command::new(Served38::exe()?);
        cmd.env(bloomery_levers::DRAFT, "off")
            .env(bloomery_levers::RESIDENCY, "off")
            .env_remove(bloomery_levers::MTP_WIDTH);
        let mut served = Served38::spawn_with(&args, &own, &mut cmd)?;
        println!("prefix server pid {}", served.child.id());
        let addr = served.address(&err_log, POLLS, POLL)?;
        let url = |p: &str| format!("http://{addr}{p}");
        let mut ok = true;
        let t = PrefixRestore {
            a: TURN_A,
            b: TURN_B,
            tail: STRIPPED_A,
            a_len: PREFIX_A,
            shared: PREFIX_SHARED,
            kept: PREFIX_KEPT,
            n: N_PREDICT,
        };
        prefix_restore(&url, &own, &mut ok, &t, &|| erase(&url))?;
        println!("prefix server stopped: {}", served.stop()?);
        Ok(ok)
    }

    /// The width chooser's own server (the module header): the draft on,
    /// the residency off and `BLOOMERY_MTP_WIDTH=cost`, one greedy request
    /// of the gate's prompt then the server's orderly stop — the chooser's
    /// `mtp width` record of the slot's last request prints there, with no
    /// later request to close it out, after its steps ([`stepped_completion`], which
    /// run no pass and so print no record) — holding: the request's
    /// ids its steps' (the chooser changes which rows a pass runs,
    /// never a token), its draft counts carried, and the record the
    /// chooser's own: its windows the widths above 0 it names, E inside the
    /// window's rows, and at least one pass of a width below the draft's
    /// own (the warm-up rotation's measurements, on this server too). The
    /// per-window width is the CLI gate's clause (its window records); this
    /// server's records are the chooser's aggregates. The fixture tier
    /// defers the four clauses by name: the draft is the real file's, which
    /// proposes nothing a random target accepts, and the fixture's card
    /// budget holds no slot of the gate's context beside it.
    fn width_cost(dir: &Path, prompt: &str) -> Result<bool, GateError> {
        if !tier::run_clause("width_cost", Tag::FileBound)? {
            return Ok(true);
        }
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
                    "width_cost_reference_steps_draft_nothing",
                    "width_cost_ids_are_its_steps",
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
        let (reference, steps_draft_n) =
            stepped_completion(&url, &tokenized(&url, prompt)?, N_PREDICT)?;
        println!("width-cost steps {reference:?} draft_n={steps_draft_n}");
        check(
            &mut ok,
            "width_cost_reference_steps_draft_nothing",
            steps_draft_n == 0,
        );
        let completion = json!({
            "prompt": prompt, "n_predict": N_PREDICT, "temperature": 0, "return_tokens": true,
            "cache_prompt": false,
        });
        let (st, body) = curl(&url("/completion"), Some(&completion), false)?;
        let c = json_of("/completion", st, &body)?;
        let ids = ids_of(&c["tokens"]);
        let stop = c["stop_type"].as_str().unwrap_or("").to_owned();
        println!("width-cost tokens {ids:?} stop_type={stop}");
        // PIN(2026-10-30): this load's steps replace the plain run's ids: the draft's card
        // reserve moves target experts to the host, whose sums round another way; was
        // `generate_qwen3moe`'s ids.
        check(
            &mut ok,
            "width_cost_ids_are_its_steps",
            agree(&ids, &stop, &reference),
        );
        let d = &c["timings"];
        check(
            &mut ok,
            "width_cost_draft_counts_carried",
            d["draft_n"].as_u64().is_some_and(|n| n > 0)
                && d["draft_n_accepted"].as_u64().is_some_and(|n| n > 0),
        );
        // PIN(2026-10-28): Replace width_cost_flush_ok with width_cost_stop_ok:
        // a second prompt would hide a missing shutdown flush of the last request.
        let (st, _) = curl(&url("/shutdown"), Some(&json!({})), false)?;
        let exited = served.wait_exit()?;
        println!("width-cost server stopped: {exited:?}");
        check(
            &mut ok,
            "width_cost_stop_ok",
            st == 200 && exited.is_some_and(|s| s.success()),
        );
        let records = seat_log(&err_log)?.all(&record::MTP_WIDTH)?;
        println!("width-cost {records:?}");
        let Some(rec) = records.first() else {
            check(&mut ok, "width_cost_record_is_the_choosers", false);
            check(&mut ok, "width_cost_cut_a_window", false);
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
        Ok(ok)
    }

    /// The sampled rounds clauses (the module header) on one server of
    /// `--parallel 2` under `BLOOMERY_DRAFT=off`, `BLOOMERY_RESIDENCY=off`
    /// and `BLOOMERY_STEP_STATS=1`, into its own directory: the sampled
    /// pair of the slots clause's prompts ([`sampled_pair`]), every round of
    /// its together run one `step` pass of both slots and each request's
    /// together ids its alone run's. The alone runs are on this server, the
    /// load the together run runs (the plan of 2 sequences); a
    /// `--parallel 1` load's ids are no reference (the placement differs).
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
        ok &= sampled_pair(
            "slots_sampled",
            &addr,
            &err_log,
            [&a_ids, &b_ids],
            (false, 0),
            "step",
        )?;
        println!("slots-sampled server stopped: {}", served.stop()?);
        Ok(ok)
    }

    /// The seat's machine of the file `inputs` describes on the stage card
    /// `stage` (`Place38::machine_read`'s for `--place a` and `gate`): the
    /// file's route form at ubatches of `ub`, and the room its `slots`
    /// sequences of a slot's `ctx` positions reserve (`drafted` the draft
    /// beside the target), rounded to the stage card's allocation granule as
    /// the census read it. The seat's checkpoint reserves and its route
    /// widening are its own.
    fn route_machine(
        inputs: &PlanInputs,
        stage: CardSpec,
        (ub, ctx): (u64, u64),
        experts: Experts,
        (slots, drafted): (u64, bool),
    ) -> Result<model::placement::Machine, GateError> {
        let mut machine = machine_for_route(
            stage,
            inputs.spec.layers.len(),
            ub,
            experts,
            inputs.route_form(bloomery_gpu::NATIVE_FP4),
        );
        inputs.reserve_room(
            &mut machine,
            (ctx, slots, drafted),
            stage.granule_bytes.get(),
        )?;
        Ok(machine)
    }

    /// The gate's own plan of the file on the gate card `stage` (its free
    /// bytes the server's `plan` record names, [`stage_at`]) at a slot's
    /// `ctx` positions for `slots` resident sequences, plain, under the host
    /// room `room` (given, not read), handed to `then` — the server's plan on
    /// the seat's machine (the file's route form and the store page rounding
    /// on the stage card, which a unified card counts in its share of the
    /// host room) but for its host reserve of the checkpoints, which this
    /// machine holds only as `held` ([`CHECKPOINTS_RESERVE`], the OS's alone
    /// for `None`).
    fn gate_plan_at<T>(
        levers: &bloomery_levers::Levers,
        (ctx, slots): (usize, usize),
        stage: CardSpec,
        (room, held): (u64, Option<u64>),
        then: impl FnOnce(Result<model::placement::Plan<'_>, PlaceError>) -> Result<T, GateError>,
    ) -> Result<T, GateError> {
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mut inputs = PlanInputs::describe(&split)?;
        inputs.room = (room, HostRead::Given);
        let experts = experts38(levers)?;
        let ub = u64::try_from(ubatch_for(ctx)?)?;
        let mut machine = route_machine(
            &inputs,
            stage,
            (ub, u64::try_from(ctx)?),
            experts,
            (u64::try_from(slots)?, false),
        )?;
        if let Some(bytes) = held {
            machine
                .host_mut()
                .reserves
                .push((CHECKPOINTS_RESERVE.to_owned(), bytes));
        }
        then(inputs.plan_with_slots(
            &machine,
            (u64::try_from(ctx)?, ub),
            &PlanLevers::from_levers(levers)?,
            experts,
            slots,
        ))
    }

    /// The rounds [`room_beside_card`] moves the room by the plan's card share
    /// before it refuses by name.
    const SHARE_ROUNDS: usize = 8;

    /// The host room the [`paged`] clause asks for `host` bytes beside the
    /// gate card `stage`: `host` plus the card's share of a unified pool in
    /// the gate plan at that room (`pool_card_bytes`, 0 on a card of memory
    /// of its own, where the room is `host`). The share moves with the room
    /// (at a small room the pool plan lowers the card's experts), so the room
    /// is taken where the plan at `host + share` holds that same share,
    /// starting from the share at `host`; a share that does not settle within
    /// [`SHARE_ROUNDS`] is a named error.
    fn room_beside_card(
        levers: &bloomery_levers::Levers,
        stage: CardSpec,
        host: u64,
    ) -> Result<u64, GateError> {
        let share_at = |room: u64| {
            gate_plan_at(levers, (CTX, 1), stage, (room, None), |plan| {
                Ok(plan?.host.pool_card_bytes)
            })
        };
        let mut share = share_at(host)?;
        for _ in 0..SHARE_ROUNDS {
            let next = share_at(host + share)?;
            if next == share {
                return Ok(host + share);
            }
            share = next;
        }
        Err(format!(
            "paged: the card share of {} at {host} B beside it did not settle in {SHARE_ROUNDS} \
             rounds (last {share} B)",
            stage.name
        )
        .into())
    }

    /// The NVMe tier's RAM arena and the host's need ([`HostNeed::bytes`],
    /// the arena inside it) of [`gate_plan_at`]'s plan at the host room
    /// `room`.
    fn host_split_at(
        levers: &bloomery_levers::Levers,
        shape: (usize, usize),
        stage: CardSpec,
        room: u64,
    ) -> Result<(u64, u64), GateError> {
        gate_plan_at(levers, shape, stage, (room, None), |plan| {
            let plan = plan?;
            Ok((plan.host.nvme_arena_bytes, HostNeed::of(&plan, 0).bytes()))
        })
    }

    /// The NVMe expert tier's floor of [`gate_plan_at`]'s plan beside the
    /// gate card, and the card's share of a unified pool it carries: the
    /// floor `HostRoomFloor` names for a room of one byte (the planner's one
    /// owner of the floor), read by type, less the share
    /// (`pool_card_bytes`, 0 on a card of memory of its own) of the plan at
    /// that floor, whose cards are the refused plan's.
    fn host_floor(
        levers: &bloomery_levers::Levers,
        shape: (usize, usize),
        stage: CardSpec,
    ) -> Result<(u64, u64), GateError> {
        let floor = gate_plan_at(levers, shape, stage, (1, None), |plan| match plan {
            Err(PlaceError::Placement(PlacementError::HostRoomFloor { floor, .. })) => Ok(floor),
            Err(e) => Err(e.into()),
            Ok(_) => Err("a host room of one byte planned: no floor to read".into()),
        })?;
        let share = gate_plan_at(levers, shape, stage, (floor, None), |plan| {
            Ok(plan?.host.pool_card_bytes)
        })?;
        let beside = floor.checked_sub(share).ok_or_else(|| {
            format!("the floor {floor} B at {shape:?} is under its plan's card share {share} B")
        })?;
        Ok((beside, share))
    }

    /// The host bytes the seat's `slots` sequences of a slot's `ctx` positions
    /// pin in checkpoints on the file: `slots` of the body's budget at that
    /// context (`checkpoint_budget38` over the file's delta and selecting layers), the
    /// reserve the server's machine holds and [`gate_plan_at`]'s does not.
    fn pinned_checkpoints((ctx, slots): (usize, usize)) -> Result<u64, GateError> {
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::describe(&split)?;
        let layers = &inputs.spec.layers;
        let gdn = layers
            .iter()
            .filter(|l| matches!(l.mixer, Mixer::DeltaRule(_)))
            .count();
        Ok(runtime::seqstate::pinned(
            slots,
            checkpoint_budget38(gdn, layers.len() - gdn, u64::try_from(ctx)?)?,
        ))
    }

    /// The `ctx` clause: a plain server with no `--ctx-size` and no
    /// `--parallel` (the seat's default one slot) prints its `ctx` and
    /// `parallel` lines before its load (it is stopped there). Its one `ctx`
    /// line holds its word under the model's limit (the trained context
    /// within `place::KERNEL_POSITIONS`; `ctx_ladder::default_line_held`),
    /// and this gate's own one-slot plans at the card's free bytes agree: the
    /// card holds the default, and not the step above it when it fitted. A
    /// server asked one past the one-slot fit is refused by name before it
    /// listens: with the largest share the card holds when the card bounds
    /// the fit, by the limit when the limit does. Mutants: a default that
    /// starts a step low (the step above holds); a default of two slots (the
    /// `parallel` line names two); the refusal of a context past the card
    /// taken out (the load's own plan refuses it, not by the common words).
    // PIN(2026-10-14): the default is fitted: one slot takes the largest step
    // under the model's limit that the card holds (the context comes first, the experts
    // fill what is left), so the clause holds the line's word and the step above it; the
    // refusal names the seats' common words.
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
        line.ok_or("the default server printed no `ctx` line")?;
        let line = ctx_line(&std::fs::read_to_string(&err_log)?)
            .ok_or("the default server printed no one `ctx` line")?
            .to_owned();
        println!("ctx: {line}");
        // The resident-slot default's own line (the module header): this
        // flagless server is the one that takes it. The slots its rule names
        // are what the line's own terms hold — the default's one slot at the
        // `ctx` line's own context. FAIL-first: a default that
        // serves another count (or a split off the `ctx` line) turns this
        // red.
        let parallel =
            parallel_line(&err_log)?.ok_or("the default server printed no `parallel` line")?;
        println!("ctx: {parallel}");
        let pfield = |k: &str| ctx_ladder::field(&parallel, k);
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let (stage, plan_room) = server_stage(&err_log)?;
        let inputs = PlanInputs::describe_in(&split, (plan_room, HostRead::Given))?;
        let limit = limit38(&inputs)?;
        // The gate's own one-slot plan at the server's free reading and room,
        // on the seat's machine with its checkpoint reserve: the card's and
        // the NVMe tier's routed-expert bytes, `None` when no plan takes `c`.
        let at = |c: u64| -> Result<Option<(u64, u64)>, GateError> {
            let shape = (usize::try_from(c)?, 1);
            gate_plan_at(
                levers,
                shape,
                stage,
                (plan_room, Some(pinned_checkpoints(shape)?)),
                |plan| {
                    Ok(plan.ok().and_then(|p| {
                        p.cards
                            .first()
                            .map(|c| (c.expert_bytes, p.host.nvme_expert_bytes))
                    }))
                },
            )
        };
        let ctx = ctx_ladder::field(&line, "ctx").ok_or("the `ctx` line names no ctx")?;
        let above = step_above(limit.0, ctx);
        let (at_ctx, at_above) = (at(ctx)?, above.map(at).transpose()?);
        let parallel_ok = parallel_agrees(&parallel, None).unwrap_or(false)
            && pfield("slots") == Some(1)
            && pfield("slot_ctx") == Some(ctx);
        // The plan's floor for the PLE table's tier bounds the state the cache
        // saves ([`PlanInputs::seq_state_bytes`]): the engine's `seq38_bytes`,
        // plain and with the draft the seat's unset default carries. FAIL-first:
        // a term of the plan's state off by a byte (the saved checkpoint
        // copies, the draft's rows) turns this red.
        let gdn = inputs
            .spec
            .layers
            .iter()
            .filter(|l| matches!(l.mixer, Mixer::DeltaRule(_)))
            .count();
        let qsa = inputs.spec.layers.len() - gdn;
        let (draft_path, _) = draft_file(None, &path);
        let draft = Split::open(&draft_path)
            .map_err(|e| format!("open the MTP draft {}: {e}", draft_path.display()))?;
        let rows = head_rows_of(None, &split, inputs.spec.vocab)?.rows;
        let mtp = MtpInputs::read(&draft, &split, &inputs, rows)?;
        let state_is_the_engines = [1usize, 4096, 252_416, 262_144].iter().all(|&n| {
            let p = n as u64;
            inputs.seq_state_bytes(p, None) == seq38_bytes(gdn, qsa, n, false, KvQ8::F16)
                && inputs.seq_state_bytes(p, Some(&mtp))
                    == seq38_bytes(gdn, qsa, n, true, KvQ8::F16)
        });
        println!(
            "ctx: the gate's plans: at {ctx} {at_ctx:?}, at the step above {above:?} {at_above:?}; \
             the limit {limit:?}"
        );
        let mut ok = true;
        // The line holds its word, and the gate's own plans agree: the card
        // holds the default, and not the step above it when it fitted.
        check(
            &mut ok,
            "ctx_default_is_the_largest_step_held",
            default_line_held(&line, 1, limit) == Some(true)
                && at_ctx.is_some()
                && at_above.is_none_or(|a| a.is_none()),
        );
        check(
            &mut ok,
            "plan_floor_counts_the_engines_state",
            state_is_the_engines,
        );
        check(
            &mut ok,
            "parallel_default_holds_what_its_rule_names",
            parallel_ok,
        );
        // The one-slot fit at the default server's free reading: the largest
        // share whose plan the card holds, up to the limit.
        let at1 = |c: usize| at(u64::try_from(c)?);
        let cap = usize::try_from(limit.0)?;
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
        let status = refused.wait_exit()?;
        let text = std::fs::read_to_string(&err_log).unwrap_or_default();
        // The seats' common words: past the card, the share the card holds
        // (`ctx_ask::AskError::DoesNotFit`, its largest a step multiple at or
        // over the floor); past the limit, the limit
        // (`placement::ctx::PastLimit`).
        let named = if card_bounds1 {
            let largest = fit1 as u64 / ctx_ladder::STEP * ctx_ladder::STEP;
            let head = format!(
                "--ctx {over} over 1 slot(s) asks {over} positions a slot, and the card holds "
            );
            let tail = if largest >= ctx_ladder::FLOOR {
                format!(
                    "a slot holds at most {largest} here; ask {largest} or less, or fewer slots"
                )
            } else {
                format!(
                    "not even {} positions a slot fit; ask fewer slots or a smaller model file",
                    ctx_ladder::FLOOR
                )
            };
            text.contains(&head) && text.contains(&tail)
        } else {
            let (limit, source) = limit;
            text.contains(&format!(
                "--ctx asks {over} positions a slot, past this model's limit of {limit} \
                 ({source}; the file was trained at {}): the load stops; ask {limit} or fewer a \
                 slot",
                inputs.spec.ctx_train
            ))
        };
        let listened = record::Log::of(&text, record::BLOOMERY_SERVE_QWEN38)
            .first(&record::LISTENING38)?
            .is_some();
        println!("ctx: --ctx-size {over}: exit {status:?}; named {named}; listened {listened}");
        if status.is_none() {
            println!("ctx: the refused server stopped: {}", refused.stop()?);
        }
        check(
            &mut ok,
            "ctx_past_the_fit_is_refused_by_name",
            status.is_some_and(|s| !s.success()) && named && !listened,
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

        // The auto count preserves the requested share in both lines. Passing the share
        // as the total makes the context line read 4096 / slots and turns this clause red.
        // The clause asks for two slots or more to tell a share from a total, so a card
        // that holds one 4096-position slot cannot run it.
        let args: Vec<&str> = DEFAULT_ARGS
            .iter()
            .copied()
            .chain(["--parallel", "auto", "--slot-ctx", "4096", "--plan"])
            .collect();
        let mut served = spawn(&args)?;
        let status = served.wait_exit()?;
        if status.is_none() {
            println!("ctx: the auto plan stopped: {}", served.stop()?);
        }
        let text = std::fs::read_to_string(&err_log)?;
        let parallel = parallel_line(&err_log)?.ok_or("the auto plan printed no parallel line")?;
        let context = ctx_line(&text).ok_or("the auto plan printed no context line")?;
        let slots = ctx_ladder::field(&parallel, "slots").ok_or("no auto slot count")?;
        println!("ctx auto: {parallel}; {context}");
        check(
            &mut ok,
            "hw_auto_slots_preserve_the_requested_context",
            status.is_some_and(|s| s.success())
                && (2..=model::placement::ctx_ask::AUTO_MOST as u64).contains(&slots)
                && word_of(&parallel, "from") == Some("auto")
                && ctx_ladder::field(&parallel, "slot_ctx") == Some(4096)
                && ctx_ladder::field(&parallel, "total") == Some(4096 * slots)
                && ctx_ladder::field(context, "ctx") == Some(4096)
                && ctx_ladder::field(context, "slots") == Some(slots)
                && ctx_ladder::field(context, "total") == Some(4096 * slots),
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
    /// At the default context the window fits and the unset draft drafts on
    /// two slots: a server of `--parallel 2` with `BLOOMERY_MTP_WIDTH` set
    /// and the draft unset prints its plan record, refused by no word that
    /// the server drafts nothing (`draft_unset_drafts_on_two_slots`).
    /// FAIL-first: a seat that judges the window on the total `--ctx-size`
    /// (eight, past five) keeps the unset draft and refuses it at the context
    /// the rule chose, so the first clause's server exits with the refusal
    /// before its load; a refusal whose total is one window short (the
    /// `VERIFY_ROWS` a slot) turns the second clause red; a seat whose check
    /// passes a set draft loads it, so its server prints a `plan` record and
    /// no refusal; a seat that turns the unset draft off on two slots refuses
    /// the width lever before its plan, red on the third. The fixture tier defers the clause by name:
    /// the draft file is the real file's.
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
        // Each server runs with the residency off and the draft's levers unset
        // but `env`.
        let server = |env: &[(&str, &str)]| -> Result<Command, GateError> {
            let mut cmd = Command::new(Served38::exe()?);
            cmd.env(bloomery_levers::RESIDENCY, "off")
                .env_remove(bloomery_levers::DRAFT)
                .env_remove(bloomery_levers::MTP_HEAD_ROWS)
                .env_remove(bloomery_levers::MTP_DRAFT)
                .env_remove(bloomery_levers::MTP_WIDTH)
                .envs(env.iter().copied());
            Ok(cmd)
        };
        // A server refused before its plan, to its exit or its `plan` record:
        // its exit, its log and whether it planned.
        let refused =
            |what: &str,
             env: &[(&str, &str)],
             args: &[&str]|
             -> Result<(Option<std::process::ExitStatus>, String, bool), GateError> {
                let mut served = Served38::spawn_with(args, &own, &mut server(env)?)?;
                let mut status = None;
                for _ in 0..POLLS {
                    status = served.child.try_wait()?;
                    let text = std::fs::read_to_string(&err_log)?;
                    let planned = matches!(
                        record::Log::of(&text, record::BLOOMERY_SERVE_QWEN38)
                            .first(&record::PLAN38),
                        Ok(Some(_))
                    );
                    if status.is_some() || planned {
                        break;
                    }
                    std::thread::sleep(Duration::from_secs(1));
                }
                let text = std::fs::read_to_string(&err_log)?;
                println!("draft-share: {what}: exit {status:?}");
                if status.is_none() {
                    println!("draft-share: the {what} server stopped: {}", served.stop()?);
                }
                let planned = record::Log::of(&text, record::BLOOMERY_SERVE_QWEN38)
                    .first(&record::PLAN38)?
                    .is_some();
                Ok((status, text, planned))
            };
        let last_lines = |what: &str, text: &str| {
            println!("draft-share: the {what} server's last lines:");
            for l in text.lines().rev().take(20).collect::<Vec<_>>().iter().rev() {
                println!("  {l}");
            }
        };

        // The set draft: refused before the plan record and any load.
        let args: Vec<&str> = DEFAULT_ARGS.iter().chain(&FLAGS).copied().collect();
        let (status, text, planned) =
            refused("set-draft", &[(bloomery_levers::DRAFT, "mtp")], &args)?;
        let want = format!(
            "a slot's context of 4 leaves no positions for the MTP draft's window; --ctx-size \
             names a total of at least {positions}"
        );
        if !text.contains(&want) {
            last_lines("set-draft", &text);
        }
        check(
            &mut ok,
            "draft_set_is_refused_by_name",
            status.is_some_and(|s| !s.success()) && text.contains(&want) && !planned,
        );

        // The unset draft at the default context of two slots: it drafts, so
        // the width lever beside it plans.
        let args: Vec<&str> = DEFAULT_ARGS[..4]
            .iter()
            .chain(&["--parallel", "2"])
            .copied()
            .collect();
        let (status, text, planned) =
            refused("two-slot", &[(bloomery_levers::MTP_WIDTH, "fixed")], &args)?;
        let refused_width = text.contains("the server drafts nothing");
        if !planned || refused_width {
            last_lines("two-slot", &text);
        }
        check(
            &mut ok,
            "draft_unset_drafts_on_two_slots",
            status.is_none() && planned && !refused_width,
        );

        // The unset draft: off, and the load's record names why.
        let args: Vec<&str> = DEFAULT_ARGS[..4].iter().chain(&FLAGS).copied().collect();
        let mut served = Served38::spawn_with(&args, &own, &mut server(&[])?)?;
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
            last_lines("unset-draft", &text);
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

    /// The NVMe tier's paged rule (the seat's module doc,
    /// `bloomery_levers::paged_columns`) on the gate plan at two rooms
    /// (`BLOOMERY_HOST_ROOM`, the split dial's default arena), each its bytes
    /// beside the gate card ([`room_beside_card`]). Planning probes
    /// stop before the load; request clauses wait for listening and completion.
    /// At 27 GiB the plan pages its host experts through the
    /// tier's RAM arena (`bloomery_placement`'s
    /// `the_paged_rule_reads_the_arena` derives both rooms on the Mac):
    /// the flagless server serves its default one slot with no `paged` line,
    /// the `cache` line naming the arena; `--parallel 4` serves four
    /// slots as one pass over the arena, and `BLOOMERY_DRAFT=mtp` serves
    /// one slot with the draft on (PIN(2026-10-12): the rule refused it by
    /// name; a paged step reads its columns' ids from the arena in one union
    /// call, so the machine's drive, not the rule, answers whether the draft
    /// pays — its `paged` line names `draft=on`). At 40 GiB the tier
    /// pages experts with no arena (the mapping path), and `--parallel 2`
    /// serves its two slots (`from=flag`, no `paged` line). At 27 GiB too,
    /// the checkpoints of the resident sequences sit outside the room the
    /// split fills (`paged_checkpoints_sit_outside_the_arena`: the `cache`
    /// line's arena (`tier`) is the gate's own plan's, which holds no
    /// checkpoint reserve, less `pinned_checkpoints(slots)`, and its `need` that
    /// plan's plus the same); the prompt cache's default is 0 beside the room
    /// the split fills (`paged_default_cache_is_off_by_the_split`); a
    /// residency word set to `off` reaches the `plan` record and loads no
    /// machine (`paged_set_off_residency_is_accepted`), and one set to
    /// `mid-p0-s1` loads it and serves a request
    /// (`paged_set_residency_serves`: the machine's victim fills read through
    /// the NVMe tier's reader, so the lane's arena reads count as slot reads
    /// and none as mapping reads, `slot_reads` and `mapping_reads` of the
    /// run's last `nvtier` record). The floor counts the checkpoints too: at the
    /// middle of the band of rooms between the one-slot plan's floor and two
    /// slots' (the gate's own plans' floors, read off `HostRoomFloor`, plus
    /// `pinned_checkpoints(slots)`) `--parallel 2` is refused by the floor, by
    /// name
    /// (`paged_floor_set_parallel_is_refused_by_the_floor`); on a unified card,
    /// whose pool lowers the card budget instead, it stands at the two-slot
    /// floor on the plan the pool gives the gate too
    /// (`paged_floor_set_parallel_stands_on_the_pool_budget`). Last, the small
    /// card's case (below, at its server). The fixture tier defers the clause by name: its rooms and
    /// budget are the real file's shapes.
    /// FAIL-first: a default of two slots prints a `paged` line at 27 GiB; a
    /// rule that reads the paged experts' bytes rather than the arena refuses
    /// the 40 GiB `--parallel 2`; a yield the paged rule overrides is red on
    /// the small card; a machine with no checkpoint reserve gives the arena the gate's
    /// own plan has, red on the checkpoint clause; a seat that refuses a set
    /// word on the paged plan by name never listens, red on the residency
    /// clause, and so is one that hands the machine's lane no tier (no slot
    /// read) or lets it read the mapping (a mapping read); one that refuses
    /// `off` as well never reaches the `plan` record, red on the acceptance
    /// clause; one that turns a set draft off on a paged plan
    /// serves it plain, red on the drafted clause's `draft=on`.
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
        let gate_card = Place::Gate.on(census()?)?.card_specs()?[0];
        let paged_room = room_beside_card(levers, gate_card, 27 << 30)?;
        let mapped_room = room_beside_card(levers, gate_card, 40 << 30)?;
        println!(
            "paged: the rooms {paged_room} B and {mapped_room} B, 27 GiB and 40 GiB beside the card \
             and its share of a unified pool"
        );
        let (paged_arg, mapped_arg) = (paged_room.to_string(), mapped_room.to_string());

        let (line, _, text) = run(&[(bloomery_levers::HOST_ROOM, &paged_arg)], &[])?;
        let paged = paged_line(&text);
        println!("paged: {}", paged.as_deref().unwrap_or("no paged line"));
        let arena = text
            .lines()
            .find(|l| l.starts_with("cache "))
            .and_then(|l| field(l, "tier"))
            .and_then(|v| v.parse::<u64>().ok());
        let one = line.as_deref().is_some_and(|l| {
            let n = |k: &str| field(l, k).and_then(|v| v.parse::<usize>().ok());
            n("slots") == Some(1)
                && field(l, "from").as_deref() == Some("default")
                && n("slot_ctx").is_some_and(|c| c > 0 && n("total") == Some(c))
        });
        // PIN(2026-10-14): the default is one slot by itself (ctx-first, D2), so the
        // paged rule has no count to reduce and, with the draft off by `--place
        // gate`'s placement, nothing to turn off: it prints no `paged` line. Was two
        // slots asked, one served (`from=paged`).
        check(
            &mut ok,
            "paged_default_serves_one_column",
            one && paged.is_none() && arena.is_some_and(|a| a > 0),
        );
        // `off`, set on every server of this clause, reaches the `plan`
        // record: its `residency lever` record names `off` with why `set`
        // and the load runs no residency machine, so no `residency host`
        // record follows (a set word that runs one is the clause
        // `paged_set_residency_serves` below).
        let word_of = |log: &record::Log| -> Result<Option<(String, String)>, ReadError> {
            log.first(&record::RESIDENCY_LEVER)?
                .map(|l| -> Result<(String, String), ReadError> {
                    Ok((l.word("residency")?.to_owned(), l.word("why")?.to_owned()))
                })
                .transpose()
        };
        let off_log = record::Log::of(&text, record::BLOOMERY_SERVE_QWEN38);
        check(
            &mut ok,
            "paged_set_off_residency_is_accepted",
            off_log.first(&record::PLAN38)?.is_some()
                && word_of(&off_log)? == Some(("off".to_owned(), "set".to_owned()))
                && off_log.first(&record::RESIDENCY_HOST)?.is_none(),
        );
        // The split fills the room past the tier's floor, so the prompt
        // cache's default takes nothing beside it: `ram=0`, the split named
        // in `why=`, the arena and the drive's bytes on the line
        // (`CacheRam::of_tier`).
        let cache_off = text
            .lines()
            .find(|l| l.starts_with("cache "))
            .is_some_and(|l| {
                let n = |k: &str| field(l, k).and_then(|v| v.parse::<u64>().ok());
                n("ram") == Some(0)
                    && field(l, "rule").as_deref() == Some("default")
                    && n("tier").is_some_and(|t| t > 0)
                    && n("paged").is_some_and(|b| b > 0)
                    && l.contains(" why=the NVMe tier's split fills the room past its floor")
            });
        check(
            &mut ok,
            "paged_default_cache_is_off_by_the_split",
            cache_off,
        );
        let free = record::Log::of(&text, record::BLOOMERY_SERVE_QWEN38)
            .first(&record::PLAN38)?
            .ok_or("the 27 GiB server printed no `plan` record")?
            .u64("card_free")?;
        let stage = stage_at(Place::Gate, free)?;

        // The checkpoints every resident sequence may pin sit outside the
        // room the split fills: the host reserves them, so the plan's need
        // counts them and the arena is the room less the floor they raise.
        // The gate's own plan at the same room, slots and context, whose
        // machine holds no such reserve, differs from the server's by exactly
        // `pinned_checkpoints(slots)` in each — the split keeps the same host
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
        let held = shape.map(pinned_checkpoints).transpose()?.unwrap_or(0);
        let seat = (
            term(cache.as_deref(), "tier"),
            term(cache.as_deref(), "need"),
        );
        let bare = shape
            .map(|shape| host_split_at(levers, shape, stage, paged_room))
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

        // PIN(2026-10-23): judge the paged parallel clause again, through listening
        // and requests; the plan's arena floor counts every resident slot's columns.
        let mut cmd = Command::new(Served38::exe()?);
        cmd.env(bloomery_levers::HOST_ROOM, &paged_arg)
            .env(bloomery_levers::DRAFT, "off")
            .env(bloomery_levers::RESIDENCY, "off")
            .env(bloomery_levers::STEP_STATS, "1")
            .env_remove(bloomery_levers::NVTIER_BYTES);
        let args: Vec<&str> = SERVER_ARGS[..SERVER_ARGS.len() - 1]
            .iter()
            .copied()
            .chain(["4"])
            .collect();
        let mut parallel = Served38::spawn_with(&args, &own, &mut cmd)?;
        let addr = parallel.address(&err_log, POLLS, POLL)?;
        let text = std::fs::read_to_string(&err_log)?;
        let line = parallel_line(&err_log)?.ok_or("paged: no parallel line")?;
        let paged = paged_line(&text).ok_or("paged: no paged line")?;
        let served = parallel_agrees(&line, Some(4)) == Some(true)
            && field(&line, "pass").as_deref() == Some("one");
        let named = field(&paged, "rule").as_deref() == Some("served")
            && field(&paged, "asked_slots").as_deref() == Some("4")
            && field(&paged, "slots").as_deref() == Some("4")
            && field(&paged, "arena").and_then(|v| v.parse::<u64>().ok()) > Some(0);
        check(
            &mut ok,
            "paged_parallel_serves_one_pass",
            served && named && listened(&text)?,
        );
        let prompts = [prompt, TURN_A, TURN_B, SLOTS_B];
        let request = |prompt: &str| {
            streamed_with(
                &addr,
                json!({
                    "prompt": prompt, "n_predict": N_PREDICT, "temperature": 0,
                    "return_tokens": true, "ignore_eos": true,
                    "cache_prompt": false, "stream": true,
                }),
            )
        };
        let mut alone = Vec::new();
        for p in prompts {
            alone.extend(answers("paged", vec![(request(p)?, p)])?);
        }
        let before = seat_log(&err_log)?.all(&record::SLOTS_ROUND)?.len();
        let runs = prompts
            .iter()
            .map(|p| Ok((request(p)?, *p)))
            .collect::<Result<Vec<_>, GateError>>()?;
        let together = answers("paged", runs)?;
        check(
            &mut ok,
            "paged_parallel_ids_as_alone",
            ids_as_alone(alone.iter(), &together)
                && together
                    .iter()
                    .all(|s| s.as_ref().is_some_and(|s| s.tokens.len() == N_PREDICT)),
        );
        let rounds = seat_log(&err_log)?.all(&record::SLOTS_ROUND)?;
        let rounds = &rounds[before..];
        check(
            &mut ok,
            "paged_parallel_rounds_carry_all_slots",
            rounds.iter().any(|r| r.u64("rows") == Ok(4))
                && rounds.iter().all(|r| r.u64("passes") == Ok(1)),
        );
        println!("paged: the parallel server stopped: {}", parallel.stop()?);

        // PIN(2026-10-12): `paged_draft_is_refused_by_name` became this
        // clause. The refusal is gone by rule: a paged step of several
        // columns reads its ids from the arena in one union call (the
        // placement's floor counts a column a slot, `gate_nvtier`'s census
        // clause), so whether the draft pays on a paged plan is the
        // machine's drive, not the rule's, to answer — a set
        // `BLOOMERY_DRAFT=mtp` runs, and a user the refusal stopped now
        // runs, possibly slower on a drive flat in queue depth. Stopped at
        // the `parallel` line, before its load.
        // PIN(2026-10-14): the default is one slot (ctx-first, D2), so a set draft
        // on it is served as asked: no `paged` line, the drafted load's `draft keep`
        // line. Was two slots asked, one served (`from=paged`, `draft=on`).
        let (line, failed, text) = run(
            &[
                (bloomery_levers::HOST_ROOM, &paged_arg),
                (bloomery_levers::DRAFT, "mtp"),
            ],
            &[],
        )?;
        let drafted = paged_line(&text);
        println!("paged: {}", drafted.as_deref().unwrap_or("no paged line"));
        check(
            &mut ok,
            "paged_draft_mtp_serves_one_slot",
            !failed
                && line.as_deref().is_some_and(|l| {
                    field(l, "slots").as_deref() == Some("1")
                        && field(l, "from").as_deref() == Some("default")
                })
                && drafted.is_none()
                && text.lines().any(|l| l.starts_with("draft keep ")),
        );

        // A residency word set on a plan that pages through the arena is
        // served as given. PIN(2026-10-11): this clause replaced
        // `paged_set_residency_is_refused_by_name`; the refusal's cause is
        // gone — the lane's victim fills read through the NVMe tier's reader
        // and the step's ids stay pinned until its reads end, so a fill
        // cannot evict a slot the step is reading. The server loads in full
        // (it is the one server of this clause that does) and answers one
        // request, and stops before the next server starts. Its words come
        // from the `residency lever` and `residency host` records (the
        // `plan` record names no residency), and the lane's reads of the
        // arena's ids from the run's last `nvtier` record, which each call's
        // end prints with the counters since the load: a promotion read an
        // arena slot (`slot_reads` past zero) and none read the model file's
        // mapping (`mapping_reads` zero).
        let refusal = format!(
            "BLOOMERY_RESIDENCY={RESIDENCY_WORD}: the plan pages host experts through the NVMe \
             tier's "
        );
        let mut cmd = Command::new(Served38::exe()?);
        cmd.env(bloomery_levers::RESIDENCY, RESIDENCY_WORD)
            .env(bloomery_levers::HOST_ROOM, &paged_arg)
            .env_remove(bloomery_levers::DRAFT)
            .env_remove(bloomery_levers::NVTIER_BYTES)
            .env_remove(bloomery_levers::MTP_HEAD_ROWS)
            .env_remove(bloomery_levers::MTP_DRAFT)
            .env_remove(bloomery_levers::MTP_WIDTH);
        let mut served = Served38::spawn_with(&DEFAULT_ARGS, &own, &mut cmd)?;
        println!("paged: residency server pid {}", served.child.id());
        let completed = match served.address(&err_log, POLLS, POLL) {
            Ok(addr) => {
                let body = json!({
                    "prompt": prompt, "n_predict": N_PREDICT, "temperature": 0,
                    "return_tokens": true,
                });
                let (st, reply) = curl(&format!("http://{addr}/completion"), Some(&body), false)?;
                let tokens = json_of("/completion", st, &reply)
                    .map(|c| ids_of(&c["tokens"]))
                    .unwrap_or_default();
                println!(
                    "paged: the residency server's request: HTTP {st}, {} ids",
                    tokens.len()
                );
                st == 200 && !tokens.is_empty()
            }
            Err(e) => {
                println!("paged: the residency server never listened: {e}");
                false
            }
        };
        if served.child.try_wait()?.is_none() {
            println!("paged: the residency server stopped: {}", served.stop()?);
        }
        let text = std::fs::read_to_string(&err_log)?;
        let log = record::Log::of(&text, record::BLOOMERY_SERVE_QWEN38);
        let host_word = log
            .first(&record::RESIDENCY_HOST)?
            .map(|h| h.word("residency").map(str::to_owned))
            .transpose()?;
        let reads = log
            .all(&record::NVTIER)?
            .last()
            .map(|n| Ok::<_, ReadError>((n.u64("slot_reads")?, n.u64("mapping_reads")?)))
            .transpose()?;
        let lever = word_of(&log)?;
        println!(
            "paged: residency lever {lever:?}, host {host_word:?}, (slot_reads, mapping_reads) \
             {reads:?}"
        );
        check(
            &mut ok,
            "paged_set_residency_serves",
            log.first(&record::PLAN38)?.is_some()
                && lever == Some((RESIDENCY_WORD.to_owned(), "set".to_owned()))
                && host_word.as_deref() == Some(RESIDENCY_WORD)
                && !text.contains(&refusal)
                && listened(&text)?
                && completed
                && reads.is_some_and(|(slot, mapping)| slot > 0 && mapping == 0),
        );

        let (line, _, text) = run(
            &[(bloomery_levers::HOST_ROOM, &mapped_arg)],
            &["--parallel", "2"],
        )?;
        let asked = line.as_deref().is_some_and(|l| {
            field(l, "slots").as_deref() == Some("2") && field(l, "from").as_deref() == Some("flag")
        });
        check(
            &mut ok,
            "unpaged_parallel_serves_as_asked",
            asked && paged_line(&text).is_none(),
        );

        // The floor counts every slot's checkpoints: the gate's own floors
        // (the planner's, read off `HostRoomFloor` at a room of one byte, with
        // no checkpoint reserve) plus `pinned_checkpoints(slots)` are the
        // server's, so the two-slot plan's floor at one position sits over the
        // one-slot plan's at [`CTX`] — a band of rooms a set `--parallel 2`
        // cannot clear. At its middle that server is refused by the floor, by
        // name (its context choice's first plan fails before the `paged` rule).
        // On a unified card the floors are read beside the card
        // ([`host_floor`]), and the band is not the rule: the cards and the
        // host read one pool, and a plan the floor refuses at the cards' full
        // share is planned again under a lowered card budget (`plan_in_pool`),
        // so no room refuses the server by the floor while the pool can give
        // way. There the server is asked at the two-slot plan's floor at one
        // position, its full share in, under which every context its choice
        // takes is refused at the full share: the gate's own plan at the
        // server's shape and room, with the checkpoint reserve the server
        // holds, is the pool's (`paged_floor_pool_lowers_the_two_slot_budget`),
        // and the server stands at two slots on that plan, its card and host
        // experts the gate plan's (`paged_floor_set_parallel_stands_on_the_pool_budget`).
        // PIN(2026-10-14): `paged_floor_fallback_serves_one_column` is gone with the
        // seat's floor fallback: it served a default of two slots as one, and the
        // default is one slot now (ctx-first, D2).
        let (one_floor, one_share) = host_floor(levers, (CTX, 1), stage)?;
        let one_floor = one_floor + pinned_checkpoints((CTX, 1))?;
        let (two_floor, two_share) = host_floor(levers, (1, 2), stage)?;
        let two_floor = two_floor + pinned_checkpoints((1, 2))?;
        // PIN(2026-10-31): a unified card's pool lowers the card budget, so the floor's
        // refusal is the discrete card's only.
        let unified = stage.integrated;
        let room = if unified {
            two_floor + two_share
        } else {
            one_floor + two_floor.saturating_sub(one_floor) / 2
        };
        println!(
            "paged: floors beside the card with the checkpoints: one slot at {CTX} positions \
             {one_floor} B (card share {one_share} B), two slots at one position {two_floor} B \
             (card share {two_share} B); the server's room {room} B"
        );
        if !unified {
            check(
                &mut ok,
                "paged_floor_band_is_between_the_slot_floors",
                one_floor < two_floor,
            );
        }
        let room_arg = room.to_string();
        let (line, failed, text) = run(
            &[(bloomery_levers::HOST_ROOM, &room_arg)],
            &["--parallel", "2"],
        )?;
        if unified {
            let seat = record::Log::of(&text, record::BLOOMERY_SERVE_QWEN38)
                .first(&record::PLAN38)?
                .map(|p| {
                    Ok::<_, GateError>((
                        p.u64("card_experts")?,
                        p.u64("host_experts")?,
                        p.u64("card_free")?,
                    ))
                })
                .transpose()?;
            let shape = term(line.as_deref(), "slot_ctx").zip(term(line.as_deref(), "slots"));
            let pool = match (seat, shape) {
                (Some((_, _, free)), Some((ctx, slots))) => {
                    let shape = (usize::try_from(ctx)?, usize::try_from(slots)?);
                    let held = pinned_checkpoints(shape)?;
                    Some(gate_plan_at(
                        levers,
                        shape,
                        stage_at(Place::Gate, free)?,
                        (room, Some(held)),
                        |plan| {
                            Ok(plan
                                .map(|p| (p.budget_from_pool, p.cards[0].experts, p.host.experts))
                                .map_err(|e| e.to_string()))
                        },
                    )?)
                }
                _ => None,
            };
            println!(
                "paged: the two-slot server's plan (card experts, host experts, card free) \
                 {seat:?} at (slot_ctx, slots) {shape:?}; the gate's plan there with the \
                 checkpoints (from the pool, card experts, host experts) {pool:?}"
            );
            check(
                &mut ok,
                "paged_floor_pool_lowers_the_two_slot_budget",
                matches!(pool, Some(Ok((true, _, _)))),
            );
            let from_flag = line
                .as_deref()
                .is_some_and(|l| field(l, "from").as_deref() == Some("flag"));
            check(
                &mut ok,
                "paged_floor_set_parallel_stands_on_the_pool_budget",
                !failed
                    && from_flag
                    && shape.is_some_and(|(_, slots)| slots == 2)
                    && matches!(
                        (seat, &pool),
                        (Some((card, host, _)), Some(Ok((true, c, h)))) if card == *c && host == *h
                    ),
            );
        } else {
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
        }

        // The small card's case: a flagless server — no `--place`, so the
        // one card in view runs `a`, as `slots_default_residency`'s does —
        // under the budget whose drafted one-slot plan fits a context the
        // plain plan holds twice of ([`small_card_budget`]), at 27 GiB. The
        // unset draft yields to the context on that plan, and the load runs
        // the plain plan the yield was judged on: its `load draft=off` record
        // names the yield, the positions the `draft yield` record holds. The
        // residency is pinned off: the clause holds the draft. Stopped at that
        // record.
        // PIN(2026-10-14): was the two-slot default, whose yield the paged rule
        // overrode on the one-slot plan it fell to (`paged_small_card_load_names_the_rule`);
        // with one default slot (ctx-first, D2) the yield's plan is the load's, so its
        // record names the yield (`paged_small_card_load_names_the_yield`).
        // A unified card's reading beside the paged server is the pool its
        // room caps; the search plans the card's own pool, the census's.
        let small = match stage_at(Place::A, free)? {
            card if card.integrated => Place::A.on(census()?)?.card_specs()?[0],
            card => card,
        };
        let (budget, fitted) = small_card_budget(levers, small)?;
        let budget_bytes = budget.to_string();
        let mut cmd = Command::new(Served38::exe()?);
        cmd.env(bloomery_levers::HOST_ROOM, &paged_arg)
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
        let yielded = log
            .first(&record::DRAFT_YIELD)?
            .map(|y| -> Result<_, ReadError> {
                Ok(bloomery_levers::RungCost {
                    reserve: bloomery_levers::Reserve::Draft,
                    with: y.u64("with")?,
                    without: y.u64("without")?,
                    card_bytes: y.u64("card_bytes")?,
                })
            })
            .transpose()?;
        if off.is_none() {
            let tail: Vec<&str> = text.lines().rev().take(20).collect();
            println!(
                "paged: the small-card server printed no `load draft=off` record; its last lines:"
            );
            for l in tail.iter().rev() {
                println!("  {l}");
            }
        }
        println!(
            "paged: small card, budget {budget} B: one card {one_card}, draft yield \
             {yielded:?} (the gate's plan: the drafted choice takes {fitted}); load draft=off ({})",
            off.as_deref().unwrap_or("none")
        );
        // The premise: the box's one-card view, and the yield at the context
        // the gate's plans put it.
        check(
            &mut ok,
            "paged_small_card_yields_on_one_card",
            one_card && yielded.is_some_and(|y| y.with == fitted),
        );
        let line = parallel_line(&err_log)?;
        let one = line.as_deref().is_some_and(|l| {
            field(l, "slots").as_deref() == Some("1")
                && field(l, "from").as_deref() == Some("default")
        });
        let why = yielded.map(|y| bloomery_levers::Draft38Off::Yield(y).to_string());
        check(
            &mut ok,
            "paged_small_card_load_names_the_yield",
            one && paged_line(&text).is_none() && why.is_some() && off == why,
        );

        // `generate_qwen3moe` under the same rule: at `a` (the one card in
        // view) at the paged room, the unset draft runs at `a` with its file
        // there, its plan pages, and the run drafts nothing — its `load
        // draft=off` record names the paged rule at the arena it names, and no
        // `load draft=mtp` line prints. The residency is pinned off, as at the
        // small card. It runs whole: `-n 2` past its load. The four-slot run
        // sets the draft off and answers each slot from the arena. A card of
        // memory of its own runs `--place a` under the gate card's bytes (the
        // binary takes `BLOOMERY_CARD_BUDGET` only beside a `--place`): a
        // larger card holds more experts at 27 GiB, so its plan has no arena
        // and the draft stays on. A unified card's room is beside its share
        // already, so it runs with no `--place`.
        // PIN(2026-10-31): was no `--place` on the 3090 alone, gpu-gate.sh's `PIN_3090`.
        let place_a: &[&str] = if gate_card.integrated {
            &[]
        } else {
            &["--place", "a"]
        };
        for slots in [1, 4] {
            let mut cmd =
                Command::new(std::env::current_exe()?.with_file_name("generate_qwen3moe"));
            cmd.env(bloomery_levers::HOST_ROOM, &paged_arg)
                .env(bloomery_levers::GEN_SLOTS, slots.to_string())
                .env(bloomery_levers::RESIDENCY, "off")
                .env_remove(bloomery_levers::DRAFT)
                .env_remove(bloomery_levers::NVTIER_BYTES)
                .env_remove(bloomery_levers::MTP_HEAD_ROWS)
                .env_remove(bloomery_levers::MTP_DRAFT)
                .env_remove(bloomery_levers::MTP_WIDTH);
            if !gate_card.integrated {
                cmd.env(
                    bloomery_levers::CARD_BUDGET,
                    gate_card.usable_bytes().to_string(),
                );
            }
            if slots > 1 {
                cmd.env(bloomery_levers::DRAFT, "off");
            }
            let fed = if slots == 1 {
                ["--prompt", prompt, "-n", "2"]
            } else {
                ["--tokens", "1,2,3,4", "-n", "2"]
            };
            let args: Vec<&str> = place_a.iter().chain(&fed).copied().collect();
            let mut run = Served38::spawn_with(&args, &own, &mut cmd)?;
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
            let head =
                "load draft=off (unset: the plan pages host experts through the NVMe tier's ";
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
                if slots == 1 {
                    "paged_cli_drafts_nothing_by_the_rule"
                } else {
                    "paged_cli_parallel_serves_as_asked"
                },
                status.is_some_and(|s| s.success())
                    && if slots == 1 {
                        arena.is_some_and(|a| a > 0) && why.is_some() && off == why
                    } else {
                        record::Log::of(&out, record::GENERATE_QWEN3MOE)
                            .all(&record::NVTIER)?
                            .iter()
                            .any(|n| n.u64("arena_granted").is_ok_and(|a| a > 0))
                    }
                    && !drafted
                    && (slots == 1
                        || (0..slots).all(|slot| {
                            out.lines().any(|l| {
                                l.strip_prefix("tokens ")
                                    .and_then(|l| l.strip_suffix(&format!(" slot={slot}")))
                                    .is_some_and(|ids| {
                                        parse_ids(ids).is_ok_and(|ids| ids.len() == 2)
                                    })
                            })
                        })),
            );
        }
        Ok(ok)
    }

    /// The card budget under which the unset draft yields to the context on
    /// `card` (`a` as the small-card server resolves it, at the seat's free
    /// reading, [`stage_at`], or a unified card's own pool), to a MiB, and
    /// the context the drafted choice fits there: for each context `c` from
    /// half the target down by halves to the floor
    /// ([`ctx_ladder::target_of`] under [`limit38`]), the least budget under
    /// which the file's drafted one-slot plan holds `c`, kept when the
    /// drafted plan holds not the step above and the plain plan holds twice
    /// `c` there — the drafted choice fits to `c` and the plain one holds at
    /// least twice as much, as on a 12 GB card. The plan is the small-card server's: the unset head and
    /// the draft file beside the target (that server sets neither lever),
    /// `card` on the seat's machine (route form and store page rounding) and
    /// no draft reserve in it, which a one-card plan counts in
    /// `plan_mtp_with_slots`.
    fn small_card_budget(
        levers: &bloomery_levers::Levers,
        card: CardSpec,
    ) -> Result<(u64, u64), GateError> {
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
        let holds = |budget: u64, c: u64, drafted: bool| -> Result<bool, GateError> {
            let ub = u64::try_from(ubatch_for(usize::try_from(c)?)?)?;
            let machine = route_machine(&inputs, card, (ub, c), experts, (1, drafted))?;
            let plan_levers = PlanLevers {
                card_budget_bytes: Some(budget).into(),
            };
            Ok(inputs
                .plan_target(
                    &machine,
                    (c, ub),
                    &plan_levers,
                    experts,
                    drafted.then_some(&mtp),
                    1,
                )
                .is_ok())
        };
        let free = card
            .free_bytes
            .ok_or("the small card's spec carries no free reading")?;
        let target = ctx_ladder::target_of(limit38(&inputs)?.0);
        let mut tried = Vec::new();
        let mut c = target / 2 / ctx_ladder::STEP * ctx_ladder::STEP;
        while c >= ctx_ladder::FLOOR {
            tried.push(c);
            let (mut lo, mut hi) = (1, free / MIB);
            if holds(hi * MIB, c, true)? {
                while lo + 1 < hi {
                    let mid = lo + (hi - lo) / 2;
                    if holds(mid * MIB, c, true)? {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                let budget = hi * MIB;
                if !holds(budget, c + ctx_ladder::STEP, true)? && holds(budget, 2 * c, false)? {
                    println!(
                        "paged: the small card's budget {budget} B (the drafted plan fits {c}, \
                         the plain plan holds {})",
                        2 * c
                    );
                    return Ok((budget, c));
                }
            }
            c = c / 2 / ctx_ladder::STEP * ctx_ladder::STEP;
        }
        Err(format!(
            "the {}'s {free} free bytes give no budget under which the drafted one-slot plan \
             fits a context the plain plan holds twice of ({tried:?}): no budget makes the small \
             card's case",
            card.name
        )
        .into())
    }

    /// The model's limit as the seat names it: the file's trained context
    /// within `place::KERNEL_POSITIONS`.
    fn limit38(inputs: &PlanInputs) -> Result<(u64, &'static str), GateError> {
        Ok(CtxLimit {
            trained: u64::from(inputs.spec.ctx_train),
            kernel: Some((KERNEL_POSITIONS, "KERNEL_POSITIONS")),
            verified: None,
        }
        .limit())
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
        // The devices' totals for the plan line's bound, read before the
        // server holds any of them.
        let census = census()?;
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
        // PIN(2026-10-14): the bound is the stage device's, not the bytes its
        // card plans: under `--place gate` on a larger device the card plans
        // fewer bytes than the device's, and `card_free` is still the
        // device's reading. The seat's `plan` line names its stage card's
        // free bytes at plan time, at most the device's usable bytes and under
        // its total: the census term the expert rule filled within
        // (memguard). FAIL-first: a plan line that drops it, or names it past
        // that bound, turns this red.
        let plan = seat_log(&err_log)?.one(&record::PLAN38)?;
        let free = plan.opt_u64("card_free")?;
        let limit = record::stage_device_free_limit(&plan, census)?;
        let free_named = free.is_some_and(|f| f <= limit);
        println!(
            "plan names the stage card's free bytes {free_named}: card_free={free:?}, usable \
             {limit}"
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
        ok &= shared_system_prompt(&url, dir, TURN_A, [TURN_A, TURN_B], N_PREDICT)?;
        // Last on the main server: its requests leave the slot another conversation's.
        ok &= crate::systemone_clause::is_the_schema_path(&url, dir)?;

        println!("server stopped: {}", served.stop()?);
        ok &= prefix(dir)?;
        ok &= ctx(dir, levers)?;
        ok &= paged(dir, levers, prompt)?;
        ok &= draft_slot_share(dir)?;
        ok &= residency_loads_the_word(dir)?;
        // The resident slots together (the module header): its own servers,
        // the first with the draft on, the second the seat's own defaults but
        // the draft.
        ok &= slots_flow_together(dir, levers)?;
        // An unset residency whose churn pool the host's room cannot take
        // keeps its P and runs the pool from the file (the module header):
        // its own two servers.
        ok &= churn_unhosted(dir, levers)?;
        // The width chooser's server (the module header): the draft on, the
        // residency off, `BLOOMERY_MTP_WIDTH=cost`.
        ok &= width_cost(dir, prompt)?;
        // The plain rounds of a load that drafts nothing run as one pass
        // (the module header): its own server, sampled requests that step,
        // the round records on.
        ok &= slots_sampled_rounds(dir)?;
        Ok(ok)
    }

    /// The MTP draft the drafted main server opens ([`draft_file`]) read
    /// against the target at `path` (`split`, `inputs`), its head the
    /// lever's rows, and the plan levers of that load at [`CTX`] a slot of
    /// `slots` resident sequences: [`tier::plan_levers`] with the draft's
    /// card bytes there as its reserve — in the fixture tier the header's
    /// budget plus them (the header's is the target's alone, so the target
    /// keeps the card experts its plain load holds), in the real tier the
    /// caller's lever as it stands.
    fn drafted_plan_inputs(
        levers: &bloomery_levers::Levers,
        path: &Path,
        split: &Split,
        inputs: &PlanInputs,
        slots: usize,
    ) -> Result<(MtpInputs, PlanLevers), GateError> {
        let (draft_path, from) = draft_file(levers.mtp_draft(), path);
        let draft = Split::open(&draft_path).map_err(|e| {
            format!(
                "open the MTP draft {} ({}): {e}",
                draft_path.display(),
                from.describe()
            )
        })?;
        let rows = head_rows_of(levers.mtp_head_rows(), split, inputs.spec.vocab)?.rows;
        let mtp = MtpInputs::read(&draft, split, inputs, rows)?;
        let plan_levers = tier::plan_levers(
            split,
            levers,
            mtp.card_bytes_of(u64::try_from(CTX)?, slots)?,
        )?;
        Ok((mtp, plan_levers))
    }

    /// `cmd`'s `BLOOMERY_CARD_BUDGET` for a drafting load of `slots` resident
    /// sequences ([`drafted_plan_inputs`]): the fixture tier's runner exports
    /// the target's alone, beside which the draft's card bytes leave no slot
    /// of [`CTX`]. The real tier sets none unless the caller's lever is set.
    fn drafted_card_budget(
        cmd: &mut Command,
        levers: &bloomery_levers::Levers,
        slots: usize,
    ) -> Result<(), GateError> {
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::describe(&split)?;
        let (_, plan_levers) = drafted_plan_inputs(levers, &path, &split, &inputs, slots)?;
        if plan_levers.card_budget_bytes.is_set() {
            cmd.env(
                bloomery_levers::CARD_BUDGET,
                plan_levers.card_budget_bytes.to_string(),
            );
        }
        Ok(())
    }

    /// The drafted arm (the module header): the main server of
    /// [`SERVER_ARGS`] under `BLOOMERY_DRAFT=mtp`, `BLOOMERY_RESIDENCY=off`
    /// and the fixed window, running the main arm's drafted clauses alone —
    /// `/props`' `engine` against the gate's own plan with the draft (its
    /// card bytes the draft's among them, its `draft` naming the file
    /// [`draft_file`] picks), the greedy pass's ids against this server's
    /// steps and its draft counts, `continued`, the sampled requests and,
    /// after them, the `cache` clause's drafted half — none of the plain
    /// arm's other clauses: each runs there on the plain main server.
    /// `generated` (`generate_qwen3moe`'s ids) is printed, not compared: that
    /// run places more target experts on the card.
    fn drafted_main(
        dir: &Path,
        levers: &bloomery_levers::Levers,
        prompt: &str,
        ids: &[u32],
        generated: &[u32],
    ) -> Result<bool, GateError> {
        let err_log = dir.join("server.err");
        let exe = Served38::exe()?;
        let mut cmd = Command::new(&exe);
        // The fixed window (the module header): this server's drafted
        // clauses hold counts across requests of it.
        cmd.env(bloomery_levers::DRAFT, "mtp")
            .env(bloomery_levers::RESIDENCY, "off")
            .env(bloomery_levers::MTP_WIDTH, "fixed");
        drafted_card_budget(&mut cmd, levers, 1)?;
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
        let (reference, steps_draft_n) = stepped_completion(&url, ids, N_PREDICT)?;
        println!("steps on the drafted server: tokens {reference:?} draft_n={steps_draft_n}");
        println!("generate_qwen3moe {generated:?} (another placement, not compared)");
        check(
            &mut ok,
            "drafted_reference_steps_draft_nothing",
            steps_draft_n == 0,
        );
        if reference.len() != N_PREDICT {
            return Err(format!(
                "--prompt {prompt:?}: the drafted server's steps gave {} ids, not {N_PREDICT}: \
                 `continued` extends the prompt by all of them",
                reference.len()
            )
            .into());
        }
        let completion = json!({
            "prompt": prompt, "n_predict": N_PREDICT, "temperature": 0, "return_tokens": true,
            "cache_prompt": false,
        });
        let (st, body) = curl(&url("/completion"), Some(&completion), false)?;
        let c1 = json_of("/completion", st, &body)?;
        let first = ids_of(&c1["tokens"]);
        let stop = c1["stop_type"].as_str().unwrap_or("").to_owned();
        println!(
            "completion tokens {first:?} stop_type={stop} content={}",
            c1["content"]
        );
        // The drafted server's own clause: the greedy ids are its steps'
        // (a draft changes which passes run, never a token), and the pass
        // carried its counts.
        // PIN(2026-10-30): same-load steps replace the plain-load oracle: the draft's card
        // reserve moves target experts to the host, whose sums round another way; was
        // `generate_qwen3moe`'s ids.
        let d = &c1["timings"];
        check(
            &mut ok,
            "drafted_ids_are_its_steps",
            agree(&first, &stop, &reference),
        );
        // The accepted count is the file's: the fixture's draft proposes,
        // and its random-weight target accepts none of it.
        if tier::run_clause("drafted_timings_carry_the_draft_counts", Tag::FileBound)? {
            check(
                &mut ok,
                "drafted_timings_carry_the_draft_counts",
                d["draft_n"].as_u64().is_some_and(|n| n > 0)
                    && d["draft_n_accepted"].as_u64().is_some_and(|n| n > 0),
            );
        }
        ok &= continued(&url, &err_log, ids, &reference)?;
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
        ok &= slot_file(dir, levers, ids)?;
        Ok(ok)
    }

    /// The slot-file clause (the module header): a server of its own under
    /// `BLOOMERY_DRAFT=mtp`, `BLOOMERY_RESIDENCY=off`, the fixed window and
    /// `--cache-ram 0`, so an erased slot leaves no state any other owner
    /// could give back. `--ids` fed fresh, then `--ids` plus its greedy reply
    /// continued on the slot that holds it (the straight run); `--ids` fed
    /// fresh again, the slot saved to a file, erased and restored from it,
    /// and the same continuation. The restore puts back the positions the
    /// save wrote, reading every byte of the engine's part
    /// (`slot_file_round_trip`), and the continuation after it is the
    /// straight run's: its ids, its kept prefix, its draft counts and joins,
    /// no draft-off line on either (`slot_file_restore_is_the_straight_run`:
    /// the draft's side came back with the body's). The same file cut short
    /// by one byte is refused by name, a 400, and the server serves on with
    /// the fresh run's ids (`slot_file_cut_short_is_refused_by_name`), and
    /// so is each file whose current or one checkpoint PLE history holds an
    /// id past the vocabulary or the model's reserved image placeholder,
    /// one history and one bad id a file
    /// (`slot_file_past_vocab_ple_history_is_refused_by_name`,
    /// `slot_file_image_ple_history_is_refused_by_name`). Changing
    /// only a pending refresh's arena to the uncarried ubatch arena is also
    /// a 400 and leaves the server serving the same fresh ids
    /// (`slot_file_uncarried_refresh_arena_is_refused_by_name`), as do an
    /// overflowing refresh first row and a last unit past its arena
    /// (`slot_file_refresh_past_its_arena_is_refused_by_name`,
    /// `slot_file_last_unit_past_its_arena_is_refused_by_name`), a refresh
    /// id past the vocabulary
    /// (`slot_file_past_vocab_refresh_id_is_refused_by_name`) and a refresh
    /// whose first row is one on, reading rows of other positions
    /// (`slot_file_shifted_refresh_rows_are_refused_by_name`): the draft's
    /// side is checked against the restored state before it is put back.
    /// Then a slot whose draft a cut turned off saves and restores
    /// ([`slot_file_off_draft`]).
    /// FAIL-first mutants: a restore that puts no state back keeps 0; one
    /// whose state carries no draft side is refused (`a state saved with the
    /// MTP draft off put back with it on`); a draft side left parked drafts
    /// other counts or leaves the draft off by name.
    fn slot_file(
        dir: &Path,
        levers: &bloomery_levers::Levers,
        ids: &[u32],
    ) -> Result<bool, GateError> {
        const FILE: &str = "q38.slot";
        const CUT: &str = "q38-cut.slot";
        const BAD_PLE: &str = "q38-bad-ple.slot";
        const BAD_ARENA: &str = "q38-bad-arena.slot";
        const BAD_FIRST: &str = "q38-bad-first.slot";
        const BAD_LAST: &str = "q38-bad-last.slot";
        const BAD_ID: &str = "q38-bad-id.slot";
        const BAD_ROW: &str = "q38-bad-row.slot";
        let own = dir.join("slot-file");
        let save = own.join("save");
        std::fs::create_dir_all(&save)?;
        let save_dir = save
            .to_str()
            .ok_or_else(|| format!("{}: not UTF-8", save.display()))?;
        let err_log = own.join("server.err");
        let ctx = CTX.to_string();
        let args = [
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--place",
            "gate",
            "--ctx-size",
            &ctx,
            "--parallel",
            "1",
            "--cache-ram",
            "0",
            "--slot-save-path",
            save_dir,
        ];
        let mut cmd = Command::new(Served38::exe()?);
        cmd.env(bloomery_levers::DRAFT, "mtp")
            .env(bloomery_levers::RESIDENCY, "off")
            .env(bloomery_levers::MTP_WIDTH, "fixed");
        drafted_card_budget(&mut cmd, levers, 1)?;
        let mut served = Served38::spawn_with(&args, &own, &mut cmd)?;
        println!("slot-file server pid {}", served.child.id());
        let addr = served.address(&err_log, POLLS, POLL)?;
        let url = |p: &str| format!("http://{addr}{p}");

        let first = greedy(&url, &err_log, ids, N_PREDICT, false)?;
        first.show("slot file: --ids fresh");
        let cont: Vec<u32> = ids.iter().chain(&first.tokens).copied().collect();
        let straight = greedy(&url, &err_log, &cont, N_PREDICT, true)?;
        straight.show("slot file: the continuation straight");
        let again = greedy(&url, &err_log, ids, N_PREDICT, false)?;
        again.show("slot file: --ids fresh again");
        let file = json!({ "filename": FILE });
        let (st_save, saved) = slot_action(&url, 0, "save", &file)?;
        let (st_erase, _) = slot_action(&url, 0, "erase", &json!({}))?;
        let (st_restore, restored) = slot_action(&url, 0, "restore", &file)?;
        let back = greedy(&url, &err_log, &cont, N_PREDICT, true)?;
        back.show("slot file: the continuation after the restore");
        let mut ok = true;
        check(
            &mut ok,
            "slot_file_round_trip",
            first.ok
                && again.tokens == first.tokens
                && st_save == 200
                && saved["n_saved"].as_u64().is_some_and(|n| n > 0)
                && st_erase == 200
                && st_restore == 200
                && restored["n_restored"] == saved["n_saved"]
                && restored["n_read"] == saved["n_written"],
        );
        check(
            &mut ok,
            "slot_file_restore_is_the_straight_run",
            straight.ok
                && back.ok
                && back.tokens == straight.tokens
                && back.cache_n == straight.cache_n
                && back.cache_n > 0
                && back.draft_n == straight.draft_n
                && back.offs.is_empty()
                && straight.offs.is_empty()
                && format!("{:?}", back.joins) == format!("{:?}", straight.joins),
        );

        let bytes = std::fs::read(save.join(FILE))?;
        std::fs::write(save.join(CUT), &bytes[..bytes.len().saturating_sub(1)])?;
        let (st_cut, cut) = slot_action(&url, 0, "restore", &json!({ "filename": CUT }))?;
        let (hst, health) = curl(&url("/health"), None, false)?;
        let after = greedy(&url, &err_log, ids, N_PREDICT, false)?;
        after.show("slot file: --ids fresh after the refusal");
        check(
            &mut ok,
            "slot_file_cut_short_is_refused_by_name",
            st_cut == 400
                && cut["error"]["type"] == "invalid_request_error"
                && cut["error"]["message"]
                    .as_str()
                    .is_some_and(|m| m.contains("a short stream"))
                && hst == 200
                && health.contains("\"ok\"")
                && after.tokens == first.tokens,
        );

        let fed: Vec<u32> = ids.iter().chain(&again.tokens).copied().collect();
        let n = saved["n_saved"].as_u64().unwrap_or(0) as usize;
        let path = ref_model_path()?;
        let split = Split::open(&path)?;
        let vocab = PlanInputs::describe(&split)?.spec.vocab;
        let (arena_at, side_at, arena, rows) = pending_refresh(&bytes, &fed, n, vocab)?;
        println!("slot file: the pending refresh holds {rows} rows of arena {arena} from row 0");
        let mut bad_arena = bytes.clone();
        bad_arena[arena_at] = 2;
        let mut bad_first = bytes.clone();
        bad_first[arena_at + 1..arena_at + 9].copy_from_slice(&u64::MAX.to_le_bytes());
        // The refresh's first id past the vocabulary, after its tags and its
        // count.
        let mut bad_id = bytes.clone();
        bad_id[side_at + 10..side_at + 14].copy_from_slice(&u32::MAX.to_le_bytes());
        // Its first row one on: inside the pass arena it reads rows that hold
        // other positions, past the step arena's one row it is outside it.
        let mut bad_row = bytes.clone();
        bad_row[arena_at + 1..arena_at + 9].copy_from_slice(&1u64.to_le_bytes());
        let shifted = if arena == 1 && rows < Prompt38::PASS_ROWS {
            "arena as positions"
        } else {
            "a parked draft's refresh outside its arena"
        };
        let mut bad_last = bytes[..side_at].to_vec();
        let mut w = bloomery_gpu::checkpoint::bytes::Put::new(&mut bad_last);
        w.vals(&[1u8, 0, 1, 0])?; // MTP, no refresh, last unit in Step.
        w.one(u64::MAX)?;
        w.vals(&[0u8; 2])?; // No skip or off reason.
        w.flush()?;
        for (name, bad, clause, says) in [
            (
                BAD_ARENA,
                bad_arena,
                "slot_file_uncarried_refresh_arena_is_refused_by_name",
                "a parked draft's arena of code 2",
            ),
            (
                BAD_FIRST,
                bad_first,
                "slot_file_refresh_past_its_arena_is_refused_by_name",
                "a parked draft's refresh outside its arena",
            ),
            (
                BAD_LAST,
                bad_last,
                "slot_file_last_unit_past_its_arena_is_refused_by_name",
                "a parked draft's last unit past its arena",
            ),
            (
                BAD_ID,
                bad_id,
                "slot_file_past_vocab_refresh_id_is_refused_by_name",
                "a saved draft id 4294967295 past the vocabulary",
            ),
            (
                BAD_ROW,
                bad_row,
                "slot_file_shifted_refresh_rows_are_refused_by_name",
                shifted,
            ),
        ] {
            std::fs::write(save.join(name), &bad)?;
            let (st_bad, refused) = slot_action(&url, 0, "restore", &json!({ "filename": name }))?;
            let (hst, health) = curl(&url("/health"), None, false)?;
            let after = greedy(&url, &err_log, ids, N_PREDICT, false)?;
            after.show(&format!("slot file: --ids fresh after {name} was refused"));
            check(
                &mut ok,
                clause,
                st_bad == 400
                    && refused["error"]["type"] == "invalid_request_error"
                    && refused["error"]["message"]
                        .as_str()
                        .is_some_and(|m| m.contains(says))
                    && hst == 200
                    && health.contains("\"ok\"")
                    && after.ok
                    && after.tokens == first.tokens,
            );
        }

        let hit = ple_histories(&bytes, &fed[..n.min(fed.len())]);
        println!(
            "slot file: PLE histories at {:?}",
            hit.iter().map(|&(at, _)| at).collect::<Vec<_>>()
        );
        let hash = engram::Hash::ple_from_keys(split.keys(), "qwen4exp", vocab)?;
        let engram::hash::Window::Eos(window) = hash.window() else {
            return Err("slot file: the model has no raw-id PLE hash".into());
        };
        for (id, clause, says) in std::iter::once((
            u32::MAX,
            "slot_file_past_vocab_ple_history_is_refused_by_name",
            "vocabulary",
        ))
        .chain(window.image.map(|id| {
            (
                id,
                "slot_file_image_ple_history_is_refused_by_name",
                "image placeholder",
            )
        })) {
            let mut ple_ok = hit.first().is_some_and(|&(at, _)| at == n) && hit.len() > 1;
            for &(at, newest) in &hit {
                let mut bad = bytes.clone();
                bad[newest..newest + 4].copy_from_slice(&id.to_le_bytes());
                std::fs::write(save.join(BAD_PLE), &bad)?;
                let (st_bad, refused) =
                    slot_action(&url, 0, "restore", &json!({ "filename": BAD_PLE }))?;
                let (hst, health) = curl(&url("/health"), None, false)?;
                let after = greedy(&url, &err_log, ids, N_PREDICT, false)?;
                after.show(&format!(
                    "slot file: --ids fresh after id {id} in the PLE history at {at} was refused"
                ));
                ple_ok &= st_bad == 400
                    && refused["error"]["type"] == "invalid_request_error"
                    && refused["error"]["message"]
                        .as_str()
                        .is_some_and(|m| m.contains("a saved PLE history") && m.contains(says))
                    && hst == 200
                    && health.contains("\"ok\"")
                    && after.ok
                    && after.tokens == first.tokens;
            }
            check(&mut ok, clause, ple_ok);
        }
        ok &= slot_file_off_draft(&url, &err_log, &save)?;
        println!("slot-file server stopped: {}", served.stop()?);
        Ok(ok)
    }

    /// The slot-file clause's draft-off round trip (`slot_file_off_draft_round_trip`):
    /// the break-even clause's long turn fed fresh, saved, its draft's point
    /// rows taken out of the file ([`without_draft_points`]) and the file
    /// restored, so the body keeps no draft side at the turn's end; the turn
    /// resent with its reply stripped keeps the turn's end with the draft off
    /// by name and steps past the first state's position. That slot saves,
    /// and its file restores every position it saved: an off draft holds no
    /// rows waiting for a walk. Mutant: a draft turned off that keeps its
    /// waiting refresh (the restore is a 400 naming the refresh's position).
    fn slot_file_off_draft(
        url: &dyn Fn(&str) -> String,
        err_log: &Path,
        save: &Path,
    ) -> Result<bool, GateError> {
        const TURN: &str = "q38-turn.slot";
        const NO_POINTS: &str = "q38-turn-no-points.slot";
        const OFF: &str = "q38-off.slot";
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
                    "slot file off: the turn's end {turn_end} is below the break-even of every \
                     reply of 2 to {N_PREDICT} tokens: the clause needs a longer turn"
                )
            })?;
        let x = greedy(url, err_log, &p1, n, false)?;
        x.show("slot file off: the turn fresh");
        if stripped.first() == x.tokens.first() {
            return Err(format!(
                "slot file off: the stripped reply's first id {:?} is the reply's: the shared \
                 prefix would not end at the turn",
                stripped.first()
            )
            .into());
        }
        let (st_turn, turned) = slot_action(url, 0, "save", &json!({ "filename": TURN }))?;
        let held = turned["n_saved"].as_u64().unwrap_or(0) as usize;
        let fed: Vec<u32> = p1.iter().chain(&x.tokens).copied().collect();
        let bytes = std::fs::read(save.join(TURN))?;
        let path = ref_model_path()?;
        let split = Split::open(&path)?;
        let vocab = PlanInputs::describe(&split)?.spec.vocab;
        let (_, side_at, _, _) = pending_refresh(&bytes, &fed, held, vocab)?;
        let (cut, points) = without_draft_points(&bytes, side_at)?;
        println!("slot file off: {points} draft point rows taken out of the turn's file");
        std::fs::write(save.join(NO_POINTS), cut)?;
        let (st_cut, _) = slot_action(url, 0, "restore", &json!({ "filename": NO_POINTS }))?;
        let strip: Vec<u32> = p1.iter().chain(&stripped).copied().collect();
        let off = greedy(url, err_log, &strip, n, true)?;
        off.show("slot file off: the turn resent stripped, the draft off");
        let file = json!({ "filename": OFF });
        let (st_save, saved) = slot_action(url, 0, "save", &file)?;
        let (st_erase, _) = slot_action(url, 0, "erase", &json!({}))?;
        let (st_restore, restored) = slot_action(url, 0, "restore", &file)?;
        let past = saved["n_saved"].as_u64().unwrap_or(0) as usize;
        println!(
            "slot file off: the off slot saved {past} positions, the turn's file {held}; restore \
             HTTP {st_restore}"
        );
        let mut ok = true;
        check(
            &mut ok,
            "slot_file_off_draft_round_trip",
            x.ok && st_turn == 200
                && points > 0
                && st_cut == 200
                && off.ok
                && off.cache_n == turn_end as u64
                && !off.offs.is_empty()
                && past > held
                && st_save == 200
                && st_erase == 200
                && st_restore == 200
                && restored["n_restored"] == saved["n_saved"],
        );
        Ok(ok)
    }

    /// The pending refresh's place in a Qwen3.8 slot file `b` of `n`
    /// positions of the sequence `fed`, laid out as `Parked::put` and
    /// `ParkedDraft::put` write it (the owners of these offsets): its arena
    /// byte's offset, the draft side's first byte, the arena's code and the
    /// refresh's rows. An active parked refresh ends with its first row (0),
    /// no last unit, no skip and no off reason, its last row at position `n`.
    /// Its ids are read from the file: on the positions `fed` holds they are
    /// `fed`'s; past it they are the server's, the ids a verify window kept
    /// past the request's last one, each under the file's `vocab`, and the id
    /// it waits on, a placeholder the next call's id replaces, for which no
    /// id is expected. Its whole draft side must match once.
    fn pending_refresh(
        b: &[u8],
        fed: &[u32],
        n: usize,
        vocab: u32,
    ) -> Result<(usize, usize, u8, usize), GateError> {
        let suffix = |tokens: &[u32], pos0: usize, arena: u8| -> Result<Vec<u8>, GateError> {
            let mut suffix = Vec::new();
            let mut w = bloomery_gpu::checkpoint::bytes::Put::new(&mut suffix);
            w.vals(&[1u8, 1])?; // MTP side, pending refresh.
            w.array(tokens)?;
            w.one(u32::try_from(pos0).expect("a refresh position fits the saved u32 positions"))?;
            w.one(arena)?;
            w.count(0)?;
            w.vals(&[0u8; 3])?; // No last unit, skip or off reason.
            w.flush()?;
            Ok(suffix)
        };
        let mut found = Vec::new();
        for rows in 1..=<Body38 as MtpBody>::VERIFY_ROWS {
            let Some(pos0) = (n + 1).checked_sub(rows) else {
                continue;
            };
            let Some(side_at) = b.len().checked_sub(suffix(&vec![0; rows], pos0, 0)?.len()) else {
                continue;
            };
            // PIN(2026-10-31): was every id `fed`'s; a verify window keeps ids past the
            // request's last.
            // The refresh's ids, after 2 tag bytes and the 8-byte count.
            let tokens: Vec<u32> = b[side_at + 10..][..4 * rows]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| u32::from_le_bytes(c))
                .collect();
            if tokens
                .iter()
                .zip(pos0..)
                .any(|(t, p)| fed.get(p).is_some_and(|f| f != t))
                || tokens[..rows - 1].iter().any(|&t| t >= vocab)
            {
                continue;
            }
            for arena in [0u8, 1] {
                if b.ends_with(&suffix(&tokens, pos0, arena)?) {
                    found.push((b.len() - 12, side_at, arena, rows));
                }
            }
        }
        match found.as_slice() {
            [one] => Ok(*one),
            _ => Err(format!(
                "slot file: expected one pending refresh, its kept ids under the vocabulary \
                 {vocab}, found {}",
                found.len()
            )
            .into()),
        }
    }

    /// A Qwen3.8 slot file `b` with its draft's point rows taken out, and
    /// how many it held: they end at the draft side `side_at`, laid out as
    /// `DraftRows38::put` writes them (a count, u64, then each point, u32,
    /// and its row, a count, u64, then the f32s, each row of one width).
    fn without_draft_points(b: &[u8], side_at: usize) -> Result<(Vec<u8>, usize), GateError> {
        let u64_at = |at: usize| {
            b.get(at..at + 8)
                .map(|v| u64::from_le_bytes(v.try_into().expect("eight bytes")))
        };
        for wide in 1..side_at / 4 {
            let rec = 12 + 4 * wide;
            if side_at < rec + 8 || u64_at(side_at - 4 * wide - 8) != Some(wide as u64) {
                continue;
            }
            for k in (1..).take_while(|k| k * rec + 8 <= side_at) {
                let row_len = side_at - k * rec + 4;
                if u64_at(row_len) != Some(wide as u64) {
                    break;
                }
                let count = side_at - k * rec - 8;
                if u64_at(count) == Some(k as u64) {
                    let mut out = b[..count].to_vec();
                    out.extend(0u64.to_le_bytes());
                    out.extend(&b[side_at..]);
                    return Ok((out, k));
                }
            }
        }
        Err("slot file: no draft point rows end at the draft's side".into())
    }

    /// The PLE histories found in a Qwen3.8 slot file `b` of the sequence
    /// `fed` (each a count, then the ids before its position, u32, then the
    /// position, u64: the current one at `fed.len()` first, then the
    /// checkpoints'): each one's position and its newest id's offset.
    fn ple_histories(b: &[u8], fed: &[u32]) -> Vec<(usize, usize)> {
        let mut hit = Vec::new();
        for at in (1..=fed.len()).rev() {
            for k in 1..=at {
                let mut pat = (k as u64).to_le_bytes().to_vec();
                pat.extend(fed[at - k..at].iter().flat_map(|t| t.to_le_bytes()));
                pat.extend((at as u64).to_le_bytes());
                if let Some(i) = b.windows(pat.len()).position(|w| w == pat) {
                    hit.push((at, i + 8 + 4 * (k - 1)));
                    break;
                }
            }
        }
        hit
    }

    /// How a lookup arm request takes its tokens after the first.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Taken {
        /// Greedy, through the draft's passes.
        Greedy,
        /// Drawn with [`SAMPLED_SEED`] from every token (`top_k` 0, `top_p`
        /// 1, `min_p` 0) at temperature 1.5, through the draft's passes, each
        /// kept id the sampler's draw from its verified row. The code
        /// prompt's rows are confident enough that the default truncation
        /// at [`SAMPLED_TEMPERATURE`] draws the greedy ids; these draws leave
        /// them.
        Sampled,
        /// Drawn at [`SAMPLED_TEMPERATURE`] with [`SAMPLED_SEED`] and two
        /// probabilities a token, collected from the verified rows.
        Logprobs,
    }

    /// Two resident slots, each conversation re-run from the cache in its own
    /// slot, then both restored from the prompt cache in reverse order after
    /// the slots are erased: the lookup follows the sequence, not its slot,
    /// and a moved restore gives the ids its own slot's re-run gave.
    /// The lookup server's ids are the draft-off server's, sampled requests
    /// included: the lookup takes no card bytes, so both load one plan, and
    /// a seeded draw from a verified row is the draw from that position's
    /// step.
    fn lookup_main(dir: &Path, prompt: &str, reference: &[u32]) -> Result<bool, GateError> {
        let code = format!(
            "{}def add_12(a, b):\n",
            (0..12)
                .map(|i| format!("def add_{i}(a, b):\n    return a + b + {i}\n\n"))
                .collect::<String>()
        );
        let prose = "Write the numbers from one to two hundred in English words, separated by \
                     commas: one, two, three, four, five,";
        // The first request after the erase that empties both slots: the prose
        // conversation restored on the code's slot, then the code on the prose's.
        const MOVED: usize = 5;
        let mut args = SERVER_ARGS.to_vec();
        args[11] = "2";
        let mut plain = Vec::<LookupCompletion>::new();
        let mut ok = true;
        for draft in ["off", "lookup"] {
            let own = dir.join(draft);
            std::fs::create_dir_all(&own)?;
            let err_log = own.join("server.err");
            let mut cmd = Command::new(Served38::exe()?);
            cmd.env(bloomery_levers::DRAFT, draft)
                .env(bloomery_levers::RESIDENCY, "off")
                .env_remove(bloomery_levers::MTP_WIDTH);
            if draft == "lookup" {
                cmd.env(bloomery_levers::MTP_WIDTH, "fixed");
            }
            let mut served = Served38::spawn_with(&args, &own, &mut cmd)?;
            let addr = served.address(&err_log, POLLS, POLL)?;
            let url = |p: &str| format!("http://{addr}{p}");
            let mut answers = Vec::<LookupCompletion>::new();
            let mut proposed = 0_u64;
            let mut accepted = 0_u64;
            for (i, (text, n, taken)) in [
                (prompt, N_PREDICT, Taken::Greedy),
                (code.as_str(), 64, Taken::Greedy),
                (prose, 64, Taken::Greedy),
                (code.as_str(), 64, Taken::Greedy),
                (prose, 64, Taken::Greedy),
                (prose, 64, Taken::Greedy),
                (code.as_str(), 64, Taken::Greedy),
                (code.as_str(), 64, Taken::Logprobs),
                (code.as_str(), 64, Taken::Sampled),
            ]
            .into_iter()
            .enumerate()
            {
                if i == 1 || i == MOVED {
                    for slot in 0..2 {
                        let (st, body) = curl(
                            &url(&format!("/slots/{slot}?action=erase")),
                            Some(&json!({})),
                            false,
                        )?;
                        slot_erased(&json_of("lookup erase", st, &body)?, slot, CTX as u64)?;
                    }
                }
                let mut request = json!({
                    "prompt": text, "n_predict": n, "temperature": 0,
                    "return_tokens": true, "cache_prompt": true,
                });
                match taken {
                    Taken::Greedy => {}
                    Taken::Sampled => {
                        for (k, v) in [
                            ("temperature", json!(1.5)),
                            ("top_k", json!(0)),
                            ("top_p", json!(1.0)),
                            ("min_p", json!(0.0)),
                            ("seed", json!(SAMPLED_SEED)),
                        ] {
                            request[k] = v;
                        }
                    }
                    Taken::Logprobs => {
                        request["temperature"] = json!(SAMPLED_TEMPERATURE);
                        request["seed"] = json!(SAMPLED_SEED);
                        request["n_probs"] = json!(2);
                    }
                }
                let (st, body) = curl(&url("/completion"), Some(&request), false)?;
                let answer = json_of("lookup completion", st, &body)?;
                println!("lookup {draft} request={i} {answer}");
                let answer = LookupCompletion::read(&answer, 2, CTX as u64)?;
                check(
                    &mut ok,
                    &format!("lookup_{draft}_{i}_completes"),
                    answer.tokens.len() == n,
                );
                if taken == Taken::Sampled {
                    check(
                        &mut ok,
                        &format!("lookup_{draft}_{i}_draws_off_the_greedy_ids"),
                        answer.tokens != answers[MOVED + 1].tokens,
                    );
                }
                if i == 0 {
                    check(
                        &mut ok,
                        &format!("lookup_{draft}_ids_are_generate_qwen3moe"),
                        agree(&answer.tokens, &answer.stop, reference),
                    );
                }
                let (d, a) = (answer.proposed, answer.accepted);
                if draft == "lookup" {
                    check(
                        &mut ok,
                        &format!("lookup_{i}_ids_are_draft_off"),
                        answer.tokens == plain[i].tokens,
                    );
                    if i > 0 {
                        // The draft proposes only where the text repeats an n-gram of its own: the
                        // fixture's random-weight target writes none, so the file answers it.
                        // PIN(2026-11-01): logprobs use verified rows too; the old
                        // `lookup_logprobs_step_without_a_draft` clause is replaced by
                        // this request's proposal check under the same file-bound rule.
                        let name = format!("lookup_{i}_verifies_proposals");
                        if tier::run_clause(&name, Tag::FileBound)? {
                            check(&mut ok, &name, d > 0);
                        }
                    }
                    proposed = proposed
                        .checked_add(d)
                        .ok_or("lookup draft_n sum overflow")?;
                    accepted = accepted
                        .checked_add(a)
                        .ok_or("lookup draft_n_accepted sum overflow")?;
                }
                answers.push(answer);
            }
            // PIN(2026-11-01): a moved restore is held to its conversation re-run from the cache
            // in its own slot, not to the fresh run: since `Engine::prefill_next` a fresh prompt
            // of at least nine ids runs its last id as the ubatch walk's last row, while a cached
            // re-run with that id alone left steps it, so fresh and cached ids are no bit pin;
            // was `answers[fresh].tokens == answers[moved].tokens`.
            for (fresh, stay, moved) in [(1, 3, MOVED + 1), (2, 4, MOVED)] {
                check(
                    &mut ok,
                    &format!("lookup_{draft}_{fresh}_moves_slots_with_its_cache"),
                    answers[stay].slot == answers[fresh].slot
                        && answers[stay].cache_n > 0
                        && answers[moved].slot != answers[stay].slot
                        && answers[moved].cache_n == answers[stay].cache_n
                        && answers[moved].tokens == answers[stay].tokens,
                );
            }
            if draft == "lookup" {
                let capture = record_line(&err_log, "capture verify")?
                    .ok_or("lookup capture verify: missing record")?;
                println!("lookup {capture}; proposed={proposed} accepted={accepted}");
                check(
                    &mut ok,
                    "lookup_captures_two_verify_rows",
                    word_of(&capture, "rows") == Some("2")
                        && word_of(&capture, "nodes")
                            .and_then(|n| n.parse::<u64>().ok())
                            .is_some_and(|n| n > 0),
                );
                // A random-weight fixture need not produce both acceptance cases.
                if tier::run_clause("lookup_verify_keeps_one_and_two_positions", Tag::FileBound)? {
                    check(
                        &mut ok,
                        "lookup_verify_keeps_one_and_two_positions",
                        accepted > 0 && proposed > accepted,
                    );
                }
            } else {
                plain = answers;
            }
            println!("lookup {draft} server stopped: {}", served.stop()?);
        }
        let texts = [code.as_str(), prose];
        ok &= lookup_cost_turns(
            dir,
            texts,
            [plain[1].tokens.as_slice(), plain[2].tokens.as_slice()],
        )?;
        Ok(ok)
    }

    /// The lookup under the width chooser's cost mode, on `--parallel 2` with
    /// `BLOOMERY_STEP_STATS=1`, into `<dir>/cost`: the two `texts` (the arm's
    /// code and prose requests) run together, so their slots take turns at
    /// the passes (a `slots round` of two rows in two passes), and their ids
    /// are `plain`, the draft-off server's. Each slot's chooser reads its own
    /// passes, so both end their warm-up: erasing the two slots prints each
    /// slot's `mtp width` record of its request, every width's cost read.
    fn lookup_cost_turns(
        dir: &Path,
        texts: [&str; 2],
        plain: [&[u32]; 2],
    ) -> Result<bool, GateError> {
        let own = dir.join("cost");
        std::fs::create_dir_all(&own)?;
        let err_log = own.join("server.err");
        let mut args = SERVER_ARGS.to_vec();
        args[11] = "2";
        let mut cmd = Command::new(Served38::exe()?);
        cmd.env(bloomery_levers::DRAFT, "lookup")
            .env(bloomery_levers::RESIDENCY, "off")
            .env(bloomery_levers::MTP_WIDTH, "cost")
            .env(bloomery_levers::STEP_STATS, "1");
        let mut served = Served38::spawn_with(&args, &own, &mut cmd)?;
        let addr = served.address(&err_log, POLLS, POLL)?;
        let mut runs = Vec::new();
        for text in texts {
            runs.push(streamed_with(
                &addr,
                json!({
                    "prompt": text, "n_predict": 64, "temperature": 0, "return_tokens": true,
                    "cache_prompt": false, "stream": true,
                }),
            )?);
        }
        let mut ok = true;
        for (i, ((h, rx), want)) in runs.into_iter().zip(plain).enumerate() {
            h.join()
                .map_err(|_| format!("lookup cost: request {i}'s thread panicked"))?;
            let ran = rx
                .recv()
                .map_err(|_| format!("lookup cost: request {i}'s thread gave no answer"))??;
            println!("lookup cost request={i} tokens {:?}", ran.tokens);
            check(
                &mut ok,
                &format!("lookup_cost_{i}_ids_are_draft_off"),
                ran.tokens == want,
            );
        }
        let rounds = seat_log(&err_log)?.all(&record::SLOTS_ROUND)?;
        let turns = rounds
            .iter()
            .filter(|r| {
                r.word("cmd") == Ok("pass") && r.u64("rows") == Ok(2) && r.u64("passes") == Ok(2)
            })
            .count();
        println!(
            "lookup cost: {turns} of {} rounds both slots' passes in turn",
            rounds.len()
        );
        check(&mut ok, "lookup_cost_slots_take_turns", turns > 0);
        let before = seat_log(&err_log)?.all(&record::MTP_WIDTH)?.len();
        for slot in 0..2 {
            let (st, body) = curl(
                &format!("http://{addr}/slots/{slot}?action=erase"),
                Some(&json!({})),
                false,
            )?;
            slot_erased(&json_of("lookup cost erase", st, &body)?, slot, CTX as u64)?;
        }
        // Each erase prints its slot's record of the request. The warm-up's
        // readings need a few proposals a slot, which a random-weight
        // fixture need not make.
        let widths = seat_log(&err_log)?.all(&record::MTP_WIDTH)?;
        println!("lookup cost: width records {widths:?}, {before} before the erases");
        if tier::run_clause("lookup_cost_warm_up_ends_on_both_slots", Tag::FileBound)? {
            check(
                &mut ok,
                "lookup_cost_warm_up_ends_on_both_slots",
                widths.len() == before + 2
                    && widths[before..].iter().all(|r| {
                        r.csv("costs")
                            .is_ok_and(|c| c.len() == 2 && !c.contains(&"-"))
                    }),
            );
        }
        println!("lookup cost server stopped: {}", served.stop()?);
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
                 arm off, the drafted arm mtp, the lookup arm off and lookup)"
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
        census()?;
        let ok = match a.arm {
            Arm::Plain => plain(&a.dir, &levers, &a.prompt, &reference)?,
            Arm::Drafted => drafted_main(&a.dir, &levers, &a.prompt, &a.ids, &reference)?,
            Arm::Lookup => lookup_main(&a.dir, &a.prompt, &reference)?,
        };
        if ok {
            println!("gate-gpu-qwen38-serve: PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
