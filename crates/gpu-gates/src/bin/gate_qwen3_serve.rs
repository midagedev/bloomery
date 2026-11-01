//! `gate_qwen3_serve` — the qwen3 seat of `bloomery-serve` against
//! `generate_qwen3moe`, on each model file given, one process at a time on
//! the card the runner pins.
//!
//!     gate_qwen3_serve --model <gguf> [--model <gguf> ...] --dir <dir> [--only-mixed]
//!
//! `--only-mixed` runs the two-slot greedy and seeded sampled requests alone
//! and together, holding their ids and the shared round shape on one server.
//!
//! Before any file, touching no card, it holds
//! `unplaced_card_is_the_largest`: on fake censuses of this workstation's
//! two cards, the card a run with `--place` unset loads on — whole or placed
//! (`q3place::PlaceQ3::unplaced_on`) — is the A6000 at its own ordinal with
//! the 3090 at device 0 and with it at device 1, a lone 3090 is itself, and
//! a census of no device is `a`'s refusal by name; that card's allocation
//! granule is the census's, the unit the seat counts its cache planes in.
//! FAIL-first: a card that is the census's first device turns the
//! 3090-first census red, and the workstation's 2 MiB granule a 64 KiB one.
//! Then it reads this process's census once, before any server holds the
//! card, for the granule the cache clauses below count in.
//!
//! For each file it starts `bloomery-serve --model qwen3 -m <file> --port 0
//! --parallel 1` (beside this binary, `BLOOMERY_REF_MODEL` removed from its
//! environment: the file is named once, by `-m`; the plain engine pinned,
//! the prefix clauses holding the one-slot path's keeps) and holds:
//!
//! - `default_seat_listens`: that server — no `--ctx`, the seat's own
//!   default — listens on the card the runner pins, both cards in turn
//!   (`BLOOMERY_GATE_CARD`): its default context is one whose whole load
//!   passes the load's own arena check, or the placed plan's. A server that
//!   never listens is this check red, its words printed, and the file's
//!   other clauses skipped. FAIL-first: a whole fit that leaves the arena out
//!   of its need picks, on the 3090, a Qwen3.6 context whose load the arena
//!   check refuses;
//! - `default_load_is_its_whole_fit`: the seat prints a whole-fit verdict
//!   line at each context its default asked, one at the loaded context, its
//!   terms named; that one says `fits` exactly when the load is the
//!   whole-card one (no `plan` record), and none at a larger context fits;
//! - `default_load_counts_its_arena`: that line's arena and the reserve
//!   past it are the pair the load's own `whole_load` hands the verdict for
//!   the file (a qwen3moe file's line names the margin in their place). The
//!   terms are the file's, not the card's, so a seat whose whole load
//!   carries no arena is red on either card, where `default_seat_listens`
//!   is red only on a card whose room past the weights and the cache is
//!   under the arena. FAIL-first: that seat prints `arena 0 B`;
//! - `load_and_listen`: the server prints its `load` record for the file's
//!   architecture before its `listening` record;
//! - `unplaced_default_names_itself`: with no `--place` the default is the
//!   whole-card load printing no `plan` record (a card with room; today's
//!   lines), or the auto-placed plan whose one record names
//!   `whole_does_not_fit` (FAIL-first: a default that plans without the why);
//! - the chat turn [`MESSAGES`], rendered by the server's own template
//!   (`/apply-template`, `/tokenize`), is longer than the eight ids a pass
//!   takes, so the prompt call runs the ubatch walk;
//! - `completion_ids_are_the_cli_ids`: `/completion` at temperature 0, [`N`]
//!   tokens, of those ids and of [`PROSE`] (`/tokenize`, a raw continuation,
//!   which a model's next ids follow more loosely than a chat answer, so a
//!   prompt call fed other ids moves them sooner), gives for each the ids
//!   `generate_qwen3moe --arm <ids>/N --ctx <the server's loaded context>
//!   … --last-step` prints for the same file (the server's cut: the prompt
//!   less its last id, then a step). The CLI's own default context plans a
//!   different split of the routed experts, and the card leg and the host
//!   leg sum in different orders, so the ids are the server's context's —
//!   all of them, or a prefix ending at an end-of-generation id, and the
//!   two loads' plans agree (both absent, or the same host experts, card
//!   experts and context);
//! - `chat_is_those_ids`: `/v1/chat/completions` of the same turn at
//!   temperature 0 and `max_tokens` N is a 200 whose `usage` counts the
//!   prompt's ids and the completion's tokens, and whose message text (the
//!   reasoning and the content) is inside the text of those ids
//!   (`/detokenize`);
//! - `schema_jump_is_the_step_arms`: `/completion` of [`JUMP_ASK`]'s turn
//!   under [`jump_schema`] (an intent of four options and a boolean, the
//!   suites' median request) on the jump arm gives the step arm's text,
//!   stop and token count, and its ids but the last (the jump arm names
//!   its stop) — the step arm the same request asking `n_probs`, which
//!   reads every row; thinking off. The jump arm's counts are the design's: the lead rode
//!   the prompt's call, the run between the two picks was one
//!   `next_after` (the seat's prompt call of the run and its last id, its
//!   head's row the second pick's: below `GEMM_FROM` ids the pass path,
//!   whose rows the e2e gates pin to the steps' bit for bit), no `next`,
//!   no `prefill`, and the closing text a tail. FAIL-first: the jump arm
//!   off (`BLOOMERY_SCHEMAJUMP=off`) reads `arm` `step` and no
//!   `next_after`;
//! - `answer_spans_take_no_marks`, a `Body35` file, between the two arms: the
//!   answer's run took no checkpoint, as the steps it replaces take none —
//!   a request of the prompt and the answer the jump arm fed, but its last
//!   id, keeps the prompt call's end (`P + lead - 1`). FAIL-first: a run fed
//!   with the body's marks on keeps its start, `P + lead`;
//! - `shared_system_prompt_stays_in_the_slot`: two chats with the same system
//!   message and different users keep at least the second message's start and
//!   answer the ids of a fresh run. FAIL-first: a recurrent seat without its
//!   message-start marker keeps no checkpoint inside the shared prefix;
//! - `systemone_is_the_schema_path` (`shared/systemone_clause.rs`, the clause
//!   `gate_qwen38_serve` holds too): the rows of `tools/ref/clef/suite.jsonl`
//!   POSTed to `/v1/systemone`, answered by this model through the
//!   structured-output path, are 200s with the decision seats' keys, and
//!   their picks are `/v1/chat/completions`' on the same request converted
//!   with `response_format`. FAIL-first: a server without the route
//!   answers every row 404;
//! - on a qwen3moe file, whose seat keeps a prefix of what the slot holds,
//!   the prefix clauses: the edit clause — the turn of [`EDIT_A`] answered,
//!   then resent with its user message changed to [`EDIT_B`], which diverges
//!   at `j` inside the rows the first run's prompt call wrote with at least
//!   [`GEMM_FROM`] ids after it — keeps `j` (`cache_n`) and its ids are the
//!   same ids fed fresh (`cache_prompt: false`: a cut call of at least
//!   `GEMM_FROM` rows is the GEMM walk wherever it is cut); and the
//!   extension clause — the turn resent with its reply and [`LATER`]'s user
//!   turn — keeps every position the slot held (`cache_n == held`), its ids
//!   printed only, for the rows the first run's steps wrote are not a fresh
//!   prompt call's rows;
//! - on a qwen35moe file, whose seat keeps a prefix back to the checkpoints
//!   a marked prompt call took (every ubatch of positions and its end), the
//!   prefix clauses: the stripped clause — the turn of [`EDIT_A`] answered,
//!   then resent with a reasoning-free reply ([`STRIPPED_35`]) in place of
//!   the reply, so the shared prefix ends at the turn's prompt end — keeps
//!   the checkpoint there (`cache_n == len(p1) - 1`) and its ids are the
//!   same ids fed fresh (a re-fed call of at least [`GEMM_FROM`] rows is the
//!   wide walk wherever it is cut); and the extension clause — the turn
//!   resent with its reply and [`LATER`]'s user turn — keeps every position
//!   the slot held (`cache_n == held`: the ask reaches the standing
//!   position, which no cut takes back), its ids printed only.
//!   FAIL-first mutants, each red on its line: a keep rule that grants the
//!   common prefix instead of the checkpoint makes the session's cut refuse
//!   by name and both clauses' requests fail; the cut's restore omitted
//!   leaves the re-fed ids a fresh run's (the stripped clause's ids red).
//! - `slots_flow_together`, on both arms (both bodies hold resident
//!   sequences: a qwen3moe file's per-position caches, a qwen35moe file's
//!   `Slot35`): a server of
//!   `--parallel 2` (the gate's ctx handling — no `--ctx`, the auto
//!   default), two greedy streamed requests of distinct prompts
//!   (`n_predict` [`SLOTS_PREDICT`], `ignore_eos`, no cache), each first run
//!   alone on that same server and then both together —
//!   (v1) each request's together ids are its alone ids (the slots hold
//!   separate sequences; nothing one runs feeds the other);
//!   (v2) `swaps_total` is absent or 0 (no park), and over the together run
//!   the deltas of the busy slots booked / `n_decode_total` sit at ≥ 1.5
//!   (`/metrics` carries the busy count as the running mean
//!   `n_busy_slots_per_decode`; times `n_decode_total` it is the total):
//!   the worker books one call a round carrying every running slot
//!   (`worker.rs`'s `Stats`), so two requests that decode together book 2
//!   busy a call, the two prompts and any non-overlapping head or tail
//!   rounds 1 — at 96 tokens each the ratio is (2·94 + 2 + ~2)/(94 + 2 +
//!   ~2) ≈ 1.9 — while a turn-taking server books exactly 1 busy a call
//!   (every round is one slot's), 1.0 past the two prompts;
//!   (v3) the load record names `slots=2` and `slot_ctx` = the arm's set
//!   `--ctx` of 8192 over two (a cache row the granularity, the floor
//!   exact), which the whole load holds on either card, and `/props`'
//!   `n_ctx` is that slot ctx.
//!   The same clause runs a second time on a placed load, its checks named
//!   `placed_…`: `--place a` under `BLOOMERY_CARD_BUDGET=12G` (a 12 GiB
//!   card's budget, which keeps routed experts on the host tier on either
//!   card) and no `--ctx`, so (v3)'s total is the placed default the server's
//!   own `plan` record was made at (`ctx_max`) — the default listening at
//!   all is the split's check on the smallest card the seat serves. Then
//!   `placed_slots_hold_the_plan` reads that server's stderr: the plan keeps
//!   experts on the card and on the host (the arm ran the placed body, not
//!   the whole load a plan of every expert opens), and the seat's `placed
//!   slots` line names two sequences of half the plan's `ctx_max` holding at
//!   most what the plan counts for them — its KV term at the total (every
//!   slot's rows: two of `⌊T/2⌋` never past `T`) and its card's `resident
//!   slots` reserve — and `placed_slots_reserve_the_fixed_stores` holds that
//!   line's terms to the file's own planes ([`cache_planes`]): the two
//!   sequences hold twice one slot's planes at `⌊T/2⌋` rows, and the reserve
//!   is one sequence's planes no position moves (a qwen35moe file's delta
//!   layers' states and rings: the plan's KV term at the total counts them
//!   once, the second slot holds them again), named on the line only where
//!   they are not 0 (none on a qwen3moe file). FAIL-first: a reserve left
//!   out holds the second slot's fixed stores past the plan's count — the
//!   open refuses by name and the placed qwen35moe server never listens
//!   (`placed_slots_load_names_the_split` red); one counted for every slot
//!   names twice the planes and the clause is red.
//!   `placed_props_engine_placement_is_the_plans`: the placed arm's `/props`
//!   `engine.placement` is its plan's — the card, then the host, routed experts
//!   on the host exactly when the plan record's `host_experts` is not 0;
//!   `props_engine_whole_load_names_no_placement`:
//!   the whole-card arm opens with no plan, so it reports none and its line
//!   says why. FAIL-first: a seat that reports no placement is red on the
//!   placed arm. `slots_count_their_checkpoints`
//!   reads both servers' stderr: a qwen35moe file's two sequences each pin
//!   up to the body's budget at the slot's context of checkpoints on the host, and
//!   the seat's `checkpoints:` line names two sequences and twice the budget
//!   — against the host's available bytes on the whole-card arm, as the
//!   placed plan's host reserve on the placed one; a qwen3moe file's
//!   sequences take none and the line is absent. FAIL-first: the term
//!   counted for one sequence names one budget and the clause is red.
//!   FAIL-first mutants, each red on its line: the seat still building the
//!   turn-taking engine leaves (v2) at ~1.0 and the round clause's
//!   `slots_rounds_hold_a_round_of_both` red (its together window holds no
//!   record of both); a select that
//!   ignores the slot (both requests on one sequence) moves the together
//!   ids off the alone ones; a reply that hands row answers back in the
//!   wrong order scrambles the ids; a ctx not divided gives each slot the
//!   total, and the second sequence does not fit past the first (the auto
//!   default spent the card's free bytes on the one-sequence cache): the
//!   server never listens, which the clause names as the split's check red;
//!   a placed load kept on the turns leaves the placed arm's (v2)
//!   `placed_slots_rounds_carry_both_slots` at ~1.0 and
//!   `placed_slots_rounds_run_one_pass` red (`passes=rows`). The hold
//!   clause still sees a record of `rows=2`, and `placed_slots_are_resident`
//!   and `placed_ctx_is_the_plans_ctx_max` do not move.
//! - `slots_set_total_is_its_whole_fit`, the whole slots arm: its one
//!   whole-fit line is at the arm's set total `t` and says the load fits. The
//!   line's need less its cache term is the rest no context moves, and the
//!   two-slot need at `t` is that rest plus the allocator's bytes of the two
//!   slots' planes — each slot one slot's list at ⌊t/2⌋ rows, the per-slot
//!   list the seat's own verdict counts (`q3place::kv_bytes`: N copies of one
//!   slot's planes) — inside what the card had. A whole-fit line that says
//!   the load does not fit is a named error: the arm is not a whole arm.
//!   FAIL-first: a seat that decides a set total's load at a slot's share
//!   puts its line at half the total.
//! - `slots_rounds_run_one_pass`, on both arms (a qwen3moe file's `Body`, a
//!   qwen35moe file's `Body35`), and the seat's `parallel` line names the
//!   shape the open decided. Both arms are one pass
//!   (`slots_parallel_line_names_one_pass`,
//!   `placed_slots_parallel_line_names_one_pass`): a placed load serves
//!   several slots' rows as one pass (`Body3::serves_one_pass`), and a
//!   printed plan does not name turns. The lookup draft stays turns in
//!   `pass_of` and is not this clause. FAIL-first: the word `turns`, red
//!   on that arm. A server
//!   of `--parallel 2` under `BLOOMERY_STEP_STATS=1` prints a `slots round`
//!   record a round of several slots — two greedy streamed requests of the
//!   slots clause's prompts, each first run alone on that server and then
//!   both together — and over the together run every record holds the
//!   shape the line named: `passes=1` over `rows` of 2..=`MAX_TOKENS` on
//!   `slots=2` (`slots_rounds_run_one_pass`, and
//!   `placed_slots_rounds_run_one_pass` on the placed arm);
//!   the window holds a round of both
//!   (`slots_rounds_hold_a_round_of_both`: at least one record of `rows=2`)
//!   — the one thing the records can see of both advancing in the same
//!   rounds, for a round of one stepping slot prints no record (the
//!   worker runs it as a plain select and `next`), so the records cannot
//!   see one-slot stretches: a seat that serves one slot at a time leaves
//!   the window empty, a server that only stretches one-slot runs between
//!   two-row rounds keeps its records of both — that shape is the slots
//!   clause's (v2) busy ratio's to catch, which one busy a round pulls to
//!   ~1.0 — and the window's solo count (the decode deltas less the
//!   records and the two prompt-call bookings) prints beside it,
//!   unpinned: a round server's solo rounds are the two requests'
//!   admission skew and its mirror at the tail, a race the worker counts
//!   in passes, which no code bounds. The hold clause is not a pass-count
//!   pin (a turns mutant still prints a record of `rows=2`; the pass count
//!   is `slots_rounds_run_one_pass`); and the rows the
//!   records carry plus
//!   the solo rounds between them (a round of one row is a select and a
//!   `next`, no record; their count the `/metrics` `n_decode_total` deltas
//!   give, less the two requests' prompt-phase calls booked at their end)
//!   sum to the tokens the two requests decoded beyond their
//!   prompts' first — each request's `n_predict` less one, its first token
//!   being the prompt's own last step
//!   (`slots_rounds_rows_sum_the_decoded_tokens`). The same clause on the
//!   placed arm (`slots-rounds-placed`,
//!   `--place a` under the slots clause's budget, the lever on too) holds
//!   `passes=1` (`placed_slots_rounds_run_one_pass`),
//!   and its window holds a round of both
//!   (`placed_slots_rounds_hold_a_round_of_both`).
//!   FAIL-first: a together record of `passes=rows`, or the word `turns`,
//!   is red on `slots_rounds_run_one_pass` or
//!   `placed_slots_rounds_run_one_pass`. A seat that serves one slot at a
//!   time prints no record of both and the hold clause is red on both arms,
//!   while a server that only stretches one-slot runs between two-row rounds
//!   is the (v2) busy ratio's red, not the records'.
//! - `slots_late_request_together_alone`, on both arms: a
//!   whole-card server of `--parallel 2`, the first request decoding alone
//!   and the second posted once `/metrics`' `n_decode_total` has moved a
//!   few rounds — slot 1 taken mid-run, its prompt run between the first's
//!   rounds — answers with each request's alone ids, the first's included:
//!   the rounds the two ran together leave each slot's sequence whole
//!   whatever slot the pass left selected (a qwen35moe file's second
//!   prompt call taking its checkpoints in its own sequence). FAIL-first: a
//!   round that leaves the wrong slot selected runs the second request's
//!   prompt over the first's sequence, and both requests' ids move at once.
//! - `slot_ctx_too_small_is_refused`, `ctx_set_parallel_unset_is_one_slot` and
//!   `one_slot_parallel_line_names_one_pass` (the arms): the refusal, `--ctx 15 --parallel 2` — a slot of 7 rows,
//!   one under the 8 rows
//!   a slot loads at (`router::MAX_TOKENS`: the widest pass a whole-card
//!   load captures, and one whole prompt pass of a placed load) — ends the
//!   process before the load, naming the total, the slot count and the
//!   slot ctx; the same `--ctx 15` with no `--parallel` takes one slot at
//!   the whole flag, its `parallel` line (`from=ctx`) and the one-request
//!   note on stderr before the load, the server stopped there; the line's
//!   `pass=` word is `one` (`one_slot_parallel_line_names_one_pass`: a load
//!   of one slot runs no round of several, `Pass::of` is `Pass::One`
//!   whatever the body or the placement, so the word holds on both files).
//!   PIN(2026-10-07): the refusal's flag set gained its `--parallel 2` — a
//!   set `--ctx` with no `--parallel` is one request's context now
//!   (`bind::slots_of`), so the split refusal needs the count said, and the
//!   unset flag set is the one-slot rule's own check. Re-pin by
//!   specification, not a relaxation: the refusal holds its named message
//!   at the same boundary, and the one-slot rule is a new check beside it.
//!   FAIL-first: the refusal dropped loads the slot and dies in
//!   the pass capture under the launcher's own message; a seat that still
//!   splits a set `--ctx` by its default two prints `slots=2 slot_ctx=7`
//!   and dies in the load instead of the one-slot line.
//! - `ctx_choice_reads_the_census_once`, the main server and
//!   `ctx_default`'s placed arm, the servers whose `--ctx` is unset: each
//!   `--ctx` choice the seat ran names its probes and its census readings on
//!   one line, and every one read the census once for all its probes.
//!   FAIL-first: a choice that resolves its placement per probe reads it
//!   once a probe where the card is short of the target.
//! - `ctx_flag_wins`, the whole-fit clause's server: `/props`' `n_ctx` is
//!   the flagged context, not the seat's default.
//! - `whole_fit_counts_the_planes_granules`: a server at a flagged `--ctx`
//!   (one row past an odd multiple of 1024, so the rounding binds on any
//!   card's granule) and `--parallel 1`
//!   prints the seat's one whole-fit line before any load — fits or not,
//!   the term prints either way — and its cache term equals the
//!   allocator's bytes of the file's own planes at the card's granule:
//!   f16's `k` and `v` a plain-attention layer, a qwen35moe
//!   file's recurrent layers' `state` and `ring` (the layers its
//!   `attn_qkv` tensor marks). FAIL-first: the raw sum the verdict
//!   counted before the planes went through the allocator is 95.9 MiB short
//!   at that ctx on the Qwen3-30B file at a 2 MiB granule (every plane 1023
//!   KiB under the granules it takes) and about 21.7 MiB on the Qwen3.6 one.
//! - `ctx_never_passes_the_trained_context`, `default_load_names_f16` and
//!   `unset_place_is_the_rule`, the main server's default arm: `/props`'
//!   `n_ctx` is at most the file's trained context, the load record names
//!   the f16 planes, and the common rule's `place unset` record names `a`,
//!   its why one card (or two the body serves no tier card of), and no
//!   break-even, basis or tier count; and `ctx_default_is_the_largest_step_held`,
//!   the same server, on either card: the default is a context an unset
//!   `--ctx` takes — the file's trained context (read by the seat's own
//!   `trained_ctx`), or a `CTX_GRAN` step under it, the base a slot (4096)
//!   the least — and either the target or one a step under a context whose
//!   whole load does not fit (the placed plan there was short too, which the
//!   `ctx` line's `fitted` word holds). The fit is the seat's own whole-fit
//!   line at the default — the census reading, its verdict, what the card
//!   had, the need term by term — with its cache term moved a step above.
//!   FAIL-first: a default that starts a step under the target defaults
//!   under the target where the whole load fits the target.
//! - `default_ctx_line_names_its_rule`, on the main server's default arm
//!   (`--parallel 1`) and `ctx_default`'s placed arm (`--parallel 2`): the
//!   one `ctx` line of an unset `--ctx` (`placement::ctx_ask::Chosen::line`)
//!   names a rule word and holds the relations the word claims: `target`
//!   the target, `fitted` a step under it, its `from` the target and its
//!   `need` past its `room` (the step above was short); its slots are
//!   the arm's, its `total` the slots' sum, its `limit` the file's trained
//!   context. The answer is also what the server serves (`/props`' `n_ctx`
//!   the slot context, the plan's `ctx_max` the total on the placed arm).
//!   FAIL-first: a seat that prints no `ctx` line, two, or a context off the
//!   steps turns this red.
//! - `ctx_past_the_trained_is_refused_by_name`: two `--plan` runs of a set
//!   `--ctx` on `--place a`. One position past the file's trained context
//!   on one slot exits non-zero with the shared limit's words
//!   (`placement::ctx::PastLimit`); twice the trained context on two slots,
//!   a share of exactly it each, is not refused by the limit. FAIL-first: a
//!   seat that skips the check plans the first; one that holds the total to
//!   the limit refuses the second.
//! - `plan_flag_stops_before_the_load`: `--plan` (a placed load under a
//!   12 GiB card budget, two slots) prints the `ctx` line and the `plan`
//!   record, whose `ctx_max` is the line's `total`, and the process exits 0
//!   with no `load` or `listening` record. FAIL-first: a seat that ignores
//!   the flag loads and listens, the clause red.
//! - `defaults_take_a_long_prompt_on_a_12gb_card`, on a qwen35moe file
//!   alone: `--plan` with the seat's defaults — no `--ctx`, no `--parallel` —
//!   under [`SMALL_CARD_BUDGET`] and [`SMALL_HOST_ROOM`], a 12 GB card in a
//!   27.5 GiB host, prints a `ctx` line of one slot (the default count) of at
//!   least [`LONG_PROMPT_SLOT_CTX`] positions that holds its word (the clause
//!   above), and stops as the `--plan` clause holds (the line's `total` the
//!   `plan` record's `ctx_max`, exit 0, no `load` or `listening` record). The
//!   plan is asked for by name, `--place a`: on a 12 GB card the unset
//!   `--place` falls to the placed plan on `a`'s card because the whole file
//!   does not fit, but the whole-fit verdict reads the card's census, not the
//!   budget, so a card that holds the file whole would print no plan; the
//!   placed probes and the `ctx` line are the same either way. FAIL-first: a
//!   default that keeps the card's experts before the context fits to the
//!   floor and the clause is red.
//! - `placed_slots_are_resident`, `ctx_default`'s placed arm (`--place a
//!   --parallel 2`, no budget), on both arms: the placed load holds its two
//!   slots resident — its plan made at the total, which counts every slot's
//!   rows and, on a qwen35moe file, the second slot's fixed stores as its
//!   `resident slots` reserve — so the load and listening records name two
//!   slots of half the plan's `ctx_max` each. FAIL-first: the seat's placed
//!   load kept on one sequence (the turns a qwen35moe file took before its
//!   body held resident slots) names one sequence of `ctx_max` in its load.
//! - `placed_answers_a_prompt_past_the_gemm_walk`, a placed server whose
//!   card budget is [`HOST_TIER_BUDGET`] (`--place a --parallel 1`): its
//!   plan record puts routed experts on the host, and a greedy
//!   `/completion` of [`PROSE`]'s ids (at least [`GEMM_FROM`]) answers 200
//!   with tokens — a placed load's prompt call runs the wide units from [`GEMM_FROM`] ids and passes below.
//!   FAIL-first: a prompt call that asks a placed load for the GEMM walk is
//!   refused by `prefill_plan` and the request answers no tokens.
//!
//! With `--images` (the file's own clauses first) or `--only-images`, the image clauses on the
//! Qwen3.6 file (`images`), against the oracle sets of `refset::arch::qwen35moe::vis` — llama.cpp
//! mainline's mtmd on the same model file and the family's bf16 projector — and on the Qwen3-30B
//! file the open's refusal:
//!
//! - `image_without_mmproj_is_refused`: an image part on the seat with no `--mmproj` is a 400 in the
//!   server's own words for every seat that takes no image. FAIL-first: a seat that hands the
//!   server an image input it has no tower for answers another status;
//! - `mmproj_on_a_qwen3moe_file_is_refused`: `--mmproj` on the Qwen3-30B file exits at the open
//!   naming the flag and the file. FAIL-first: an open that ignores the flag listens;
//! - `mmproj_props_vision_true` and `mmproj_vision_record_is_the_towers_figure`: the server with
//!   `--mmproj` (`--parallel 2`, the seat's own default context) answers `/props` with vision on, and
//!   its `vision` record names the card and the seat word the plan gives the tower
//!   (`vision_seat::encoder_seat`, on this process's cards: a reserve of the stage card when it
//!   holds one) and the weight and activation bytes of the projector's card figure at the default
//!   limits. FAIL-first: a record that swaps the two byte counts;
//! - `mmproj_whole_fit_names_the_reserve` and `mmproj_whole_fit_counts_the_rotation_tables`: the
//!   seat's whole-fit line (read from the lines it prints before its load) carries the encoder's
//!   reserve by name and bytes when the tower sits on the stage card (none named off the plan), and
//!   its cache term is the allocator's bytes of the file's own planes ([`cache_planes`]) with each
//!   slot's rotation table after them, one f32 row of `rope.dimension_count` a position of the
//!   slot's share. FAIL-first: a verdict that leaves the reserve out; one that leaves the tables
//!   out;
//! - `mmproj_plan_holds_the_reserve` and `mmproj_ctx_default_names_the_reserve`: under
//!   [`PLACED_BUDGET`] at one `--ctx` the placed plan keeps as many routed experts on the card as
//!   the plan of a text load under the budget less what the image load sets aside (the tower's
//!   reserve and each slot's rotation table), and fewer than the plan without `--mmproj` when the
//!   tower sits on the stage card; the default `--ctx` line of the placed load names the reserve.
//!   FAIL-first: a plan that leaves the reserve out keeps as many as without `--mmproj`; a line
//!   that does not name it;
//! - `mmproj_prompt_ids_are_set_e`: for each request of set E (`e1`: text, image, text; `e2`: a
//!   system turn and an image-only turn; `e3`: an image in each of two user turns) the prompt the
//!   server renders (`/apply-template`, `/tokenize`) is the set's ids with each image's pad run
//!   cut to the one placeholder id, and the tokens the server evaluates are the set's count.
//!   FAIL-first: parts of a message joined by a newline throughout, not llama-server's rule;
//! - `mmproj_image_chat_follows_set_f`: on a server of its own (`--mmproj`, `--parallel 2`, a
//!   flagged `--ctx` of [`SET_F_CTX`]: the host experts' rule is not the oracle's, so set F runs
//!   on a load whose routed experts are all on the card) the greedy ids of [`F_STEPS`] steps after
//!   the prompts `c1` and `c2` equal set F's CUDA ids over the prefix on which the set's CPU twin
//!   (a second reading of mainline) gives the same ids, at least [`F_MIN_PREFIX`] long and
//!   printed; at the first difference the oracle's id must be the server's own runner-up within
//!   [`F_NEAR_TIE`] of its top (`ChatReply::follows`). FAIL-first: the span's rows zeroed in the seat's
//!   `prefill_media`. The grid's order is not this clause's pin (a grid taken as `(ny, nx)` stays
//!   inside the near-tie band): the vision crate's `a_spans_grid_runs_along_the_images_width`
//!   holds it. The clause's precondition,
//!   `mmproj_set_f_server_keeps_every_expert_on_the_card`, reads that server's load by record kind
//!   (its `load` record, then its `plan` record or, with none, a whole-fit verdict that says
//!   `fits`) and refuses by name a plan that keeps a routed expert on the host (a server at the
//!   seat's own default `--ctx` does) and a load it cannot read as whole or placed;
//! - `mmproj_image_slot_beside_a_text_slot`: an image request posted on the second slot while a text
//!   request decodes on the first answers the ids it answered alone, the text request too, and the
//!   server ran rounds of both slots. FAIL-first: a media call that lands on the first slot.
//!
//! A dense `qwen35` file (the Qwen3.8-27B) is a third model. Its body is `Body35`, so the clauses
//! above that name a qwen35moe file hold on it too — the checkpoint prefix clauses, the resident
//! slots and their `checkpoints:` line, the planes and the arena of the whole-fit line, the late
//! arrival, the rounds — and, comparing the engine with itself (the server against the CLI, the
//! load against its own verdict), they read no reference set. The clauses that assume routed
//! experts have no arm on it and print `n/a (dense: …)` with the reason, no verdict:
//! `defaults_take_a_long_prompt_on_a_12gb_card`, the placed slots arms (`placed_…`),
//! `placed_slots_hold_the_plan` and the images. `plan_flag_stops_before_the_load` plans it with
//! no card budget, and `ctx_default`'s placed arm (`--place a`, no budget) is a plan record of
//! no experts around the whole-card load. Beside them, on the clauses' own terms:
//!
//! - `records_name_the_files_architecture`, every file: the `load` and the `listening` record
//!   name the header's architecture word. FAIL-first: a seat that prints `qwen35moe` for the
//!   dense file leaves both records red.
//! - `tool_call_is_parsed`, a `Body35` file: a server over a scripted engine (no model behind it)
//!   whose every reply is the file's template's XML call (`<tool_call><function=…>`), asked one tool
//!   turn with the file's own chat template, answers 200 with `finish_reason` `tool_calls` and one
//!   call of the listed function with the arguments the script gave
//!   (`serve_client::scripted_call`, which `gate_llama_serve` runs too): the markup the template
//!   teaches is the format the reader binds (`qwen35moe::spec`) and the server parses, and the
//!   clause reads nothing the model chose to reply. FAIL-first: a scanner that drops a call's
//!   parameters answers `{}` and the clause is red.
//! - `dense_below_its_floor_is_refused_by_name`, the dense file: `--plan` on `--place a` under
//!   [`PLACED_BUDGET`] exits non-zero with the context choice's refusal (the least the default
//!   tries takes more bytes than the budget's, `ask fewer slots or a smaller model file`) and
//!   opens nothing: a dense backbone has no routed expert to move to the host tier. FAIL-first:
//!   a budget over the file's floor plans and exits 0.
//! - `q8_cache_is_refused_on_the_scalar_pair_flash`, a `Body35` file whose head shape runs the
//!   scalar pair flash (the 27B: 24 query heads over 4 key-value heads, `decode_mma` false): the
//!   server of `--cache-type-k q8_0` at the default arm's context exits before it listens and its
//!   stderr names the cache's flash; the q8_0 delta and cache-word clauses are `n/a` for it. When
//!   the pass exists the clause is red and its owner swaps it for those two. FAIL-first: a server
//!   that listens leaves it red.
//!
//! The spawn census, every server this gate starts and its `--parallel`:
//! [`spawn`] and `ctx_default`'s q8 arm pin `--parallel 1` (the prefix and
//! ctx clauses hold the one-slot path's keeps and defaults), its placed arm
//! passes `--parallel 2` (resident, above), and its whole-fit clause pins
//! `--parallel 1` at a flagged `--ctx` (one sequence's planes; the resident
//! split's clause is the slots clause's); `cache_refusals`' two refusal
//! arms pass none — they die at flag parsing before the seat splits
//! anything — and its flag-wins arm pins `--parallel 1`; the slots clause's
//! two arms pass `--parallel 2` (resident on both files: the whole-card load
//! and the budgeted placed one), the refusal arm passes `--parallel 2` too
//! (the split it refuses is the flag's), and its one-slot arm passes none (a
//! set `--ctx` alone is one request's context, one slot at the whole flag); the
//! defaults clause's plan-only server passes none (the default count is its
//! claim); the slots-round clause's two
//! arms (both files) pass `--parallel 2` with `BLOOMERY_STEP_STATS=1` in their
//! environment (the whole-card load and the budgeted placed one), and the
//! late arm passes `--parallel 2` (whole-card, no lever); the host-tier
//! arm (qwen3moe) pins `--parallel 1` (one prompt on
//! the placed path is the clause), and set F's image server passes `--parallel 2` at the
//! flagged `--ctx` [`SET_F_CTX`] (the other image clauses' servers take the seat's own default
//! `--ctx`).
//!
//! Held by another test, and not run here: the choice of `--ctx` — the
//! target, the fit, the floor, a set total's share and the largest a
//! short card would hold, the line's words — by
//! `bloomery_placement::placement::ctx_ask`'s tests; a set share past the
//! file's limit, by `bloomery_placement::placement::ctx`'s.
//!
//! - `mtp_ids_match_the_plain_path`: on a file whose header carries the in-file
//!   next-token layer, greedy output under `BLOOMERY_DRAFT=mtp` equals the same
//!   prompt under `BLOOMERY_DRAFT=off` for [`MTP_N`] tokens, and the drafted
//!   completion's `draft_n_accepted` is above 0. A file without the layer prints
//!   that the clause is pinned to [`MTP_FILE`] and starts no second server. The
//!   pinned file with no layer is red. FAIL-first: dropping the id compare leaves
//!   a draft that changes tokens green; treating accepted as `>= 0` leaves a
//!   server that proposes nothing green.
//!
//! The server is stopped by the handle this binary spawned it with before
//! the CLI loads. Logs per file in `<dir>/<n>/` (`server.err`, `gen.log`,
//! the responses).

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen3_serve", gate::run())
}

#[cfg(feature = "deepseek41")]
#[path = "shared/systemone_clause.rs"]
mod systemone_clause;

#[cfg(feature = "deepseek41")]
#[path = "shared/qwen3moe_place.rs"]
#[allow(
    dead_code,
    reason = "the gate reads the file's trained context through the seat's own owner; the planner serves the seat and the CLI"
)]
mod q3place;

#[cfg(feature = "deepseek41")]
mod gate {
    use std::collections::BTreeMap;
    use std::num::NonZeroU64;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::{OnceLock, mpsc};
    use std::thread::JoinHandle;
    use std::time::Duration;

    use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_size;
    use bloomery_gpu::arch::qwen3moe::{Body35, KvQ8, Open35, Qwen35moeModel, checkpoint_budget35};
    use bloomery_gpu::linear::{KHeadMap, LinearShape};
    use bloomery_gpu::route_softmax::MAX_TOKENS;
    use bloomery_gpu_gates::beside;
    use bloomery_gpu_gates::ctx_ladder::{
        ctx_line, default_line_held, field as field_u64, fits as default_fits, step_above,
        target_of,
    };
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::qwen_seat::{self, decode_mma};
    use bloomery_gpu_gates::record::{self, Fields};
    use bloomery_gpu_gates::serve_client::{
        self, Answer, PrefixTurns, Served, WholeTerms, agree, answered, chat_is_those_ids, check,
        curl, edit_and_extension, greedy_chat, ids_of, json_of, metric, rendered, scripted_call,
        serve_cmd, server_log, sse_event, sse_events, tokenized, whole_terms,
    };
    use bloomery_gpu_gates::ubatch_pick;
    use bloomery_gpu_gates::vision_seat::{self, EncoderSeat};
    use bloomery_gpu_gates::{GateError, checks_failed};
    use gguf::{Gguf, Split};
    use model::arch::models::Mixer;
    use model::arch::qwen35moe::mtp::{InFile, in_file};
    use model::placement::WholeLoad;
    use model::placement::workstation::{A6000, DeviceInfo, RTX_3090};
    use refset::arch::qwen35moe::vis::{self, CHATIDS, DECODE};
    use refset::clefvis::{ClefvisSet, Span};
    use refset::family::Family;
    use serde_json::{Value, json};
    use threads::helper::{Placement, spawn_helper};
    use tokenizer::Tokenizer;
    use vision::arch::card::CardBytes;
    use vision::arch::qwen3vl::{self, TokenLimits};

    use crate::q3place;

    const USAGE: &str = "usage: gate_qwen3_serve --model <gguf> [--model <gguf> ...] --dir <dir> \
                         [--images | --only-images | --only-mixed]";

    /// The tokens each request makes.
    const N: usize = 64;

    /// The chat turn every file answers.
    fn messages() -> Value {
        json!([{ "role": "user", "content": "Name the three primary colors of light, and say in one sentence why a screen mixes them." }])
    }

    /// The raw prompt every file continues.
    const PROSE: &str = "The lighthouse keeper counted the steps as he climbed: one hundred and twelve, the same as every night, until the hundred and thirteenth";

    /// How many greedy tokens the drafted and undrafted runs are compared over.
    /// A comparison length, not a measured result.
    const MTP_N: usize = 16;

    /// The file `gate_qwen35moe_mtp` loads: the one whose header carries the
    /// next-token layer. Fixture files do not, and the clause is pinned here.
    const MTP_FILE: &str = "/models/qwen36-hf/unsloth-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";

    /// The edit clause's turn, and the same turn with its user message
    /// changed: the resend diverges inside the rows the first run's prompt
    /// call wrote, with the rest of the prompt after it.
    const EDIT_A: &str =
        "Name the three primary colors of light, and say in one sentence why a screen mixes them.";
    const EDIT_B: &str =
        "Name the three primary colors of ink, and say in one sentence why a page reflects them.";

    /// The whole-call clause's system message, ahead of [`EDIT_A`]'s user
    /// turn: the chat's prompt opens two messages.
    const SYSTEM_MSG: &str = "You answer in one short sentence, naming each item exactly as the \
                              question names it.";

    /// The user turn the extension clause appends after the reply, as the
    /// template renders it past the reply's end.
    const LATER: &str = "<|im_end|>\n<|im_start|>user\nAnd which of the three does a screen show when it shows none of them?\n<|im_end|>\n";

    /// The stripped clause's resend on a qwen35moe file (the module header):
    /// a reasoning-free reply in place of the reply, then a later user turn
    /// — the template's rendering of the reply with its reasoning dropped.
    /// Its first id is a word of the answer, never the reply's first id.
    const STRIPPED_35: &str = "Red, green and blue; a screen mixes them because each of its \
                               pixels emits those three lights side by side.<|im_end|>\n\
                               <|im_start|>user\nAnd which of the three does a screen show when \
                               it shows none of them?<|im_end|>\n<|im_start|>assistant\n";

    /// The prompt ids a pass takes at most: the rendered turn must be
    /// longer, so the clause covers the ubatch walk.
    const PASS_IDS: usize = 8;

    /// The slots clause's requests: long enough that the together run holds
    /// dozens of rounds with both streams live.
    const SLOTS_PREDICT: usize = 96;

    /// The slots clause's poll for the refusal arm's exit.
    const SLOTS_POLL: Duration = Duration::from_millis(500);

    /// The late clause's arrival point: the first request this many decode
    /// rounds in (`/metrics`' `n_decode_total` past its start) when the
    /// second is posted.
    const LATE_ROUNDS: u64 = 4;

    /// The late clause's poll of the arrival point, and the most polls it
    /// takes before the second is posted anyway.
    const LATE_POLL: Duration = Duration::from_millis(50);
    const LATE_POLLS: usize = 600;

    /// The refusal arm's `--ctx`: `--parallel 2` of it leaves a slot of one
    /// row under the seat's least slot context, the widest pass a
    /// whole-card load captures ([`MAX_TOKENS`] rows); the one-slot arm
    /// takes the whole of it, over the floor.
    const SLOTS_REFUSED_CTX: usize = 2 * MAX_TOKENS - 1;

    /// The fewest rows the seat's prompt call runs as the GEMM walk
    /// (`app::arch::qwen3moe::GEMM_FROM`): the rows a kept prefix leaves
    /// behind are a whole fresh run's.
    const GEMM_FROM: usize = app::arch::qwen3moe::GEMM_FROM;

    /// The seat's family of `model`, from the program its header's description names;
    /// a file the Qwen seat takes none of is a named error.
    fn seat_of(model: &Path) -> Result<qwen_seat::Family, GateError> {
        let split = Split::open(model).map_err(|e| format!("open {}: {e}", model.display()))?;
        qwen_seat::Family::of(&split).map_err(|e| format!("{}: {e}", model.display()).into())
    }

    /// Whether the family's body is `Body35`: GDN layers beside gated
    /// attention, so recurrent state and checkpoints, on a routed file and a
    /// dense one alike.
    fn body35(seat: qwen_seat::Family) -> bool {
        matches!(
            seat,
            qwen_seat::Family::Qwen35moe | qwen_seat::Family::Qwen35
        )
    }

    /// Whether the family's files have routed experts: a host tier, and so a
    /// placement under a card budget the weights alone overrun.
    fn routed(seat: qwen_seat::Family) -> bool {
        seat != qwen_seat::Family::Qwen35
    }

    /// The line of a clause a dense file has no arm of, with the reason: no
    /// verdict, so the gate's result does not move.
    fn not_for_dense(clause: &str, why: &str) {
        println!("n/a {clause} (dense: {why})");
    }

    struct Args {
        only_mixed: bool,
        models: Vec<PathBuf>,
        dir: PathBuf,
        /// `--images`: the image clauses ([`images`]) after each file's own.
        images: bool,
        /// `--only-images`: the image clauses alone.
        only_images: bool,
    }

    fn parse_args() -> Result<Args, GateError> {
        let (mut models, mut dir, mut images, mut only_images) = (Vec::new(), None, false, false);
        let mut only_mixed = false;
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            match flag.as_str() {
                "--only-mixed" => only_mixed = true,
                "--images" => images = true,
                "--only-images" => only_images = true,
                "--model" | "--dir" => {
                    let v = it
                        .next()
                        .ok_or_else(|| format!("{flag} needs a value: {USAGE}"))?;
                    if flag == "--model" {
                        models.push(PathBuf::from(v));
                    } else {
                        dir = Some(PathBuf::from(v));
                    }
                }
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        match (models.is_empty(), dir) {
            (false, Some(dir)) => Ok(Args {
                only_mixed,
                models,
                dir,
                images: images || only_images,
                only_images,
            }),
            _ => Err(USAGE.into()),
        }
    }

    /// A fake census of this workstation's cards by short name, in the
    /// order given: each its CUDA ordinal by position, its measured total,
    /// all of it free.
    fn fake_census(names: &[&str]) -> Vec<DeviceInfo> {
        names
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let (name, total_bytes) = match *n {
                    "3090" => ("NVIDIA GeForce RTX 3090", 25_351_356_416),
                    "A6000" => ("NVIDIA RTX A6000", 50_952_536_064),
                    other => panic!("no fake device {other}"),
                };
                let ordinal = u32::try_from(i).expect("a short list");
                DeviceInfo {
                    ordinal,
                    name: name.to_owned(),
                    total_bytes,
                    free_bytes: total_bytes,
                    uuid: [u8::try_from(i + 1).expect("a short list"); 16],
                    pci_bus: format!("0000:{:02x}:00.0", 0x41 + i),
                    held_by: None,
                    compute_capability: Some((8, 6)),
                    granule_bytes: None,
                    integrated: false,
                }
            })
            .collect()
    }

    /// The allocation granule of the card a run with `--place` unset loads
    /// on, as `census` reads it: the unit the seat counts its cache planes
    /// in (`q3place::kv_bytes`).
    fn granule_on(census: &[DeviceInfo]) -> Result<NonZeroU64, GateError> {
        Ok(q3place::PlaceQ3::unplaced_on(census)?.card_specs()?[0].granule_bytes)
    }

    /// [`granule_on`] this process's census, read once: [`run`] reads it
    /// before any server holds the card, so the census's context never sits
    /// beside a load.
    fn serve_granule() -> Result<NonZeroU64, GateError> {
        static READ: OnceLock<NonZeroU64> = OnceLock::new();
        if let Some(g) = READ.get() {
            return Ok(*g);
        }
        let g = granule_on(&bloomery_gpu::census()?)?;
        Ok(*READ.get_or_init(|| g))
    }

    /// `unplaced_card_is_the_largest` (module doc): the card the unset
    /// default loads on, by name and ordinal, on each fake census, and its
    /// granule.
    fn unplaced_card_is_the_largest(ok: &mut bool) {
        let card = |census: &[DeviceInfo]| {
            q3place::PlaceQ3::unplaced_on(census)
                .and_then(|p| p.card_specs())
                .map(|s| (s[0].name, s[0].device.map(|d| d.ordinal)))
                .map_err(|e| e.to_string())
        };
        let mut pass = true;
        for (order, a6000) in [(["3090", "A6000"], 1), (["A6000", "3090"], 0)] {
            let got = card(&fake_census(&order));
            println!("unplaced card on {order:?}: {got:?}");
            pass &= got == Ok((A6000.name, Some(a6000)));
        }
        let lone = card(&fake_census(&["3090"]));
        println!("unplaced card on [\"3090\"]: {lone:?}");
        pass &= lone == Ok((RTX_3090.name, Some(0)));
        let mut fine = fake_census(&["3090", "A6000"]);
        fine[1].granule_bytes = NonZeroU64::new(64 << 10);
        let granule = granule_on(&fine)
            .map(NonZeroU64::get)
            .map_err(|e| e.to_string());
        println!("unplaced card's granule on [\"3090\", \"A6000\" at 64 KiB]: {granule:?}");
        pass &= granule == Ok(64 << 10);
        let none = card(&[]);
        println!("unplaced card on []: {none:?}");
        pass &= none.is_err_and(|e| {
            e.starts_with("--place a: the placement takes the largest visible card, and 0 devices")
        });
        check(ok, "unplaced_card_is_the_largest", pass);
    }

    /// The busy slots the server has booked over its `decodes` engine calls:
    /// `/metrics` carries them as the running mean `n_busy_slots_per_decode`
    /// (`serve::api`'s metrics), so the total is that mean times the calls.
    fn busy_total(url: &dyn Fn(&str) -> String, decodes: f64) -> Result<f64, GateError> {
        Ok(metric(url, "n_busy_slots_per_decode")?.unwrap_or(f64::NAN) * decodes)
    }

    /// The server's stderr so far, read by the seat's record kinds.
    fn seat_log(err_log: &Path) -> Result<record::Log, GateError> {
        server_log(err_log, record::BLOOMERY_SERVE_QWEN3)
    }

    /// A seat's stderr `text`, read at `err_log` by the seat's record kinds;
    /// its errors name the file, whose path carries the arm's directory.
    fn seat_records(text: &str, err_log: &Path) -> record::Log {
        record::Log::of(text, record::BLOOMERY_SERVE_QWEN3).named(err_log.display().to_string())
    }

    /// A record read as a clause's input: a read that fails (the record
    /// absent, repeated or broken, a field of another type) prints its named
    /// error and is `None`, which the clause's own checks hold red.
    fn or_named<T>(read: Result<T, record::ReadError>) -> Option<T> {
        read.inspect_err(|e| println!("{e}")).ok()
    }

    /// The whole-number field `name` of `rec`, [`or_named`]; `None` for a record
    /// that was not read.
    fn u64_of(rec: Option<&Fields>, name: &str) -> Option<u64> {
        rec.and_then(|r| or_named(r.u64(name)))
    }

    /// The word ` name=…` of a `parallel` line; `None` when the line
    /// carries none.
    fn parallel_word<'a>(line: &'a str, name: &str) -> Option<&'a str> {
        line.split_whitespace()
            .find_map(|w| w.strip_prefix(&format!("{name}=")))
    }

    /// The `--ctx` choices a server's stderr names, one line a choice
    /// (`… --ctx choice ran P probes on C census reading(s)`): each one's
    /// probes and census readings.
    fn choices(log: &str) -> Vec<(u64, u64)> {
        log.lines()
            .filter_map(|l| {
                let (probes, rest) = l
                    .split(" --ctx choice ran ")
                    .nth(1)?
                    .split_once(" probes on ")?;
                let censuses = rest.split(' ').next()?;
                Some((probes.parse().ok()?, censuses.parse().ok()?))
            })
            .collect()
    }

    /// `default_ctx_line_names_its_rule` (module header), on a server of
    /// `slots` slots and an unset `--ctx` whose `/props` named `props_n` and
    /// whose file's trained context is `trained`; `plan_total` the `ctx_max`
    /// of its `plan` record when it made one: the one `ctx` line holds its
    /// word ([`default_line_held`]), and it is what the server serves
    /// (`/props`' `n_ctx` the slot context, the plan's `ctx_max` the total).
    /// A missing line or term is a failed clause.
    fn default_ctx_line_names_its_rule(
        log: &str,
        (slots, props_n, trained): (u64, u64, u64),
        plan_total: Option<u64>,
        ok: &mut bool,
    ) {
        let held = (|| {
            let line = ctx_line(log)?;
            println!("default ctx line: {line}");
            let num = |k: &str| field_u64(line, k);
            Some(
                default_line_held(line, slots, (trained, "trained"))?
                    && props_n == num("ctx")?
                    && plan_total.is_none_or(|t| Some(t) == num("total")),
            )
        })();
        check(ok, "default_ctx_line_names_its_rule", held == Some(true));
    }

    /// One `--plan` run of the seat on `model`: `args` after the file's own
    /// flags and before `--plan`, `env` on the child, whose logs go to
    /// `<dir>/<name>/`. The run's stderr and whether it exited on its own.
    fn plan_run(
        name: &str,
        (model, dir): (&Path, &Path),
        env: &[(&str, &str)],
        args: &[&str],
    ) -> Result<PlanRun, GateError> {
        let d = dir.join(name);
        std::fs::create_dir_all(&d)?;
        let err_log = d.join("server.err");
        let mut cmd = Command::new(beside("bloomery-serve")?);
        cmd.env_remove("BLOOMERY_REF_MODEL")
            .envs(env.iter().copied());
        let m = model.to_str().ok_or("the model path is not UTF-8")?;
        let mut argv = vec!["--model", "qwen3", "--port", "0", "-m", m];
        argv.extend_from_slice(args);
        argv.push("--plan");
        let mut s = Served::spawn_cmd(cmd, &argv, &d)?;
        let mut said = String::new();
        let mut exited = None;
        for _ in 0..360 {
            said = std::fs::read_to_string(&err_log).unwrap_or_default();
            if let Some(status) = s.child.try_wait()? {
                exited = Some(status.success());
                // Its last lines may have come after the read above.
                said = std::fs::read_to_string(&err_log).unwrap_or_default();
                break;
            }
            if said.contains("listening on http://") {
                break;
            }
            std::thread::sleep(SLOTS_POLL);
        }
        if exited.is_none() {
            println!("{name}: server stopped: {}", s.stop()?);
        }
        Ok(PlanRun {
            name: name.to_owned(),
            said,
            exited,
        })
    }

    /// What a `--plan` run printed and how it ended ([`plan_run`]).
    struct PlanRun {
        /// The run's name, its directory under the gate's.
        name: String,
        /// The server's stderr.
        said: String,
        /// Whether it exited zero on its own; `None` when it was still up
        /// at the poll bound and was stopped.
        exited: Option<bool>,
    }

    impl PlanRun {
        /// The whole-number field `name` of the one `ctx` line.
        fn ctx_field(&self, name: &str) -> Option<u64> {
            ctx_line(&self.said).and_then(|l| field_u64(l, name))
        }

        /// The run's stderr by the seat's record kinds.
        fn records(&self) -> record::Log {
            record::Log::of(&self.said, record::BLOOMERY_SERVE_QWEN3)
                .named(format!("{}'s stderr", self.name))
        }

        /// The `ctx_max` of the one `plan` record; a run with none, two or a
        /// broken line prints that error and has none.
        fn ctx_max(&self) -> Option<u64> {
            u64_of(
                or_named(self.records().one(&record::PLAN38)).as_ref(),
                "ctx_max",
            )
        }

        /// Whether the run stopped as `--plan` says: exit 0 with no `load`
        /// or `listening` record (a line that opens one and does not read
        /// whole counts as one), the `ctx` line's total the `plan` record's
        /// `ctx_max`.
        fn stopped_with_its_plan(&self) -> bool {
            let total = self.ctx_field("total");
            self.exited == Some(true)
                && total.is_some()
                && total == self.ctx_max()
                && !self.opened()
        }

        /// Whether the run printed a `load` or `listening` record (a line that
        /// opens one and does not read whole counts as one).
        fn opened(&self) -> bool {
            let records = self.records();
            or_named(records.first(&record::LOAD_QWEN3)).is_none_or(|l| l.is_some())
                || or_named(records.first(&record::LISTENING_QWEN3)).is_none_or(|l| l.is_some())
        }
    }

    /// `plan_flag_stops_before_the_load` (module header): a placed server
    /// of two slots under [`PLACED_BUDGET`] given `--plan` prints its `ctx`
    /// line and its `plan` record, the line's total the record's `ctx_max`,
    /// and exits 0 with no `load` or `listening` record. A seat that ignores
    /// the flag loads and listens, which the polls end and the clause names
    /// red.
    fn plan_flag_stops_before_the_load(
        model: &Path,
        seat: qwen_seat::Family,
        dir: &Path,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        // A dense file is planned on the card as it stands: under the budget a
        // routed file's plan splits, its weights alone are over the floor
        // ([`dense_is_refused_below_its_floor`]).
        let env: &[(&str, &str)] = if routed(seat) {
            &[("BLOOMERY_CARD_BUDGET", PLACED_BUDGET)]
        } else {
            &[]
        };
        let run = plan_run(
            "plan-only",
            (model, dir),
            env,
            &["--parallel", "2", "--place", "a"],
        )?;
        println!(
            "plan-only arm: exit ok {:?}, ctx line total {:?}, plan ctx_max {:?}",
            run.exited,
            run.ctx_field("total"),
            run.ctx_max()
        );
        check(
            ok,
            "plan_flag_stops_before_the_load",
            run.stopped_with_its_plan(),
        );
        Ok(())
    }

    /// `dense_below_its_floor_is_refused_by_name` (module header), on a dense
    /// file alone: `--plan` of two slots under [`PLACED_BUDGET`] on `--place a`
    /// exits non-zero with the context choice's refusal (the least the default
    /// tries, `… take N bytes on a card that holds` the budget's bytes, `ask
    /// fewer slots or a smaller model file`) and prints no `load` or
    /// `listening` record. A dense backbone has no routed expert to move to the
    /// host tier, so a budget under its weights is a refusal, not a placement.
    /// FAIL-first: a budget over the file's floor plans and exits 0, and the
    /// clause is red.
    fn dense_below_its_floor_is_refused_by_name(
        model: &Path,
        dir: &Path,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let run = plan_run(
            "plan-floor",
            (model, dir),
            &[("BLOOMERY_CARD_BUDGET", PLACED_BUDGET)],
            &["--parallel", "2", "--place", "a"],
        )?;
        let budget = format!(
            "on a card that holds {}:",
            bloomery_levers::parse_bytes(PLACED_BUDGET).map_err(|e| format!("{e}"))?
        );
        let named = run.said.contains("the least the default tries")
            && run.said.contains(&budget)
            && run.said.contains("ask fewer slots or a smaller model file");
        println!(
            "floor arm: exit ok {:?}, names the refusal {named}, opened {}",
            run.exited,
            run.opened()
        );
        check(
            ok,
            "dense_below_its_floor_is_refused_by_name",
            run.exited == Some(false) && named && !run.opened(),
        );
        Ok(())
    }

    /// `ctx_past_the_trained_is_refused_by_name` (module header): two
    /// `--plan` runs of a set `--ctx` on `--place a`. One slot asked one
    /// position past the file's `trained` context exits non-zero with the
    /// shared limit's words (`placement::ctx::PastLimit`); two slots asked
    /// twice the trained context, a share of exactly it each, are not refused
    /// by the limit (the plan may still find the card short). The words are
    /// printed with each run's exit.
    fn ctx_past_the_trained_is_refused_by_name(
        model: &Path,
        dir: &Path,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let split = Split::open(model).map_err(|e| format!("open {}: {e}", model.display()))?;
        let trained = bloomery_gpu_gates::generate::trained_ctx(&split)
            .ok_or_else(|| format!("{}: no context_length", model.display()))?;
        let past = (trained + 1).to_string();
        let over = plan_run(
            "ctx-past",
            (model, dir),
            &[],
            &["--place", "a", "--ctx", &past],
        )?;
        let refused = format!(
            "--ctx asks {past} positions a slot, past this model's limit of {trained} (trained; \
             the file was trained at {trained}): the load stops; ask {trained} or fewer a slot"
        );
        let twice = (2 * trained).to_string();
        let shared = plan_run(
            "ctx-share",
            (model, dir),
            &[],
            &["--place", "a", "--ctx", &twice, "--parallel", "2"],
        )?;
        let limit_words = "past this model's limit";
        println!(
            "ctx past the trained {trained}: --ctx {past} exit ok {:?}, named {}; --ctx {twice} \
             --parallel 2 exit ok {:?}, refused by the limit {}",
            over.exited,
            over.said.contains(&refused),
            shared.exited,
            shared.said.contains(limit_words),
        );
        check(
            ok,
            "ctx_past_the_trained_is_refused_by_name",
            over.exited == Some(false)
                && over.said.contains(&refused)
                && shared.exited.is_some()
                && !shared.said.contains(limit_words),
        );
        Ok(())
    }

    /// `defaults_take_a_long_prompt_on_a_12gb_card` (module header): the
    /// seat's defaults — no `--ctx`, no `--parallel` — planned under
    /// [`SMALL_CARD_BUDGET`] and [`SMALL_HOST_ROOM`] print a `ctx` line of
    /// one slot (the default count) of at least [`LONG_PROMPT_SLOT_CTX`]
    /// positions that holds its word ([`default_line_held`]), and stop as
    /// `--plan` says ([`PlanRun::stopped_with_its_plan`]). The plan is asked
    /// for by name (`--place a`): the census and the whole-fit verdict read
    /// the card, not the budget, so a card that holds the file whole would
    /// not plan at all.
    fn defaults_take_a_long_prompt_on_a_12gb_card(
        model: &Path,
        dir: &Path,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let split = Split::open(model).map_err(|e| format!("open {}: {e}", model.display()))?;
        let trained = bloomery_gpu_gates::generate::trained_ctx(&split)
            .ok_or_else(|| format!("{}: no context_length", model.display()))?;
        let trained = u64::try_from(trained)?;
        let run = plan_run(
            "plan-defaults",
            (model, dir),
            &[
                (bloomery_levers::CARD_BUDGET, SMALL_CARD_BUDGET),
                (bloomery_levers::HOST_ROOM, SMALL_HOST_ROOM),
            ],
            &["--place", "a"],
        )?;
        let (ctx, slots) = (run.ctx_field("ctx"), run.ctx_field("slots"));
        println!(
            "defaults arm: exit ok {:?}, plan ctx_max {:?}, {}",
            run.exited,
            run.ctx_max(),
            ctx_line(&run.said).unwrap_or("no ctx line")
        );
        check(
            ok,
            "defaults_take_a_long_prompt_on_a_12gb_card",
            run.stopped_with_its_plan()
                && ctx.is_some_and(|c| c >= LONG_PROMPT_SLOT_CTX)
                && slots == Some(1)
                && ctx_line(&run.said).and_then(|l| default_line_held(l, 1, (trained, "trained")))
                    == Some(true),
        );
        Ok(())
    }

    /// The seat's `--ctx` against the file and this gate's card, beyond the
    /// main server's own default arm ([`served`]: unset, the largest step
    /// under the target the load holds,
    /// `ctx_default_is_the_largest_step_held`): a load under `--place` takes
    /// the context its plan holds, its two slots resident on either file
    /// (`placed_slots_are_resident`), and the q8_0 cache's load at the main
    /// server's context holds the planes' bytes — the whole-fit verdict's
    /// cache term — down by their saving, the freed bytes spent on more card
    /// experts where both loads are placed (a file whose head shape has no
    /// q8_0 flash is refused instead,
    /// [`q8_cache_is_refused_on_the_scalar_pair_flash`]).
    /// The choice itself — the target, the fit, the floor and the set
    /// share — is held by `bloomery_placement::placement::ctx_ask`'s tests.
    /// FAIL-first: a choice that keeps the target whatever the card makes
    /// the placed arm's load a plan the card cannot hold (the spawn never
    /// listens).
    fn ctx_default(
        model: &Path,
        seat: qwen_seat::Family,
        dir: &Path,
        default_n: u64,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        // The trained context read from the file beside the server, not the
        // server's own echo of it, by the owner the seat's choice reads it
        // through.
        let split = Split::open(model).map_err(|e| format!("open {}: {e}", model.display()))?;
        let trained = bloomery_gpu_gates::generate::trained_ctx(&split)
            .ok_or_else(|| format!("{}: no context_length", model.display()))?;
        let trained = u64::try_from(trained)?;
        // Two slots on the placed load, resident on either file (the module
        // header's `placed_slots_are_resident`).
        let d = dir.join("ctx-placed");
        std::fs::create_dir_all(&d)?;
        let err_log = d.join("server.err");
        let mut cmd = Command::new(beside("bloomery-serve")?);
        cmd.env_remove("BLOOMERY_REF_MODEL");
        let m = model.to_str().ok_or("the model path is not UTF-8")?;
        let args = [
            "--model",
            "qwen3",
            "--port",
            "0",
            "-m",
            m,
            "--parallel",
            "2",
            "--place",
            "a",
        ];
        let mut s = Served::spawn_cmd(cmd, &args, &d)?;
        let addr = s.address(&err_log, 600, Duration::from_secs(1))?;
        let url = |p: &str| format!("http://{addr}{p}");
        let (st, body) = curl(&url("/props"), None, false)?;
        let n = json_of("/props", st, &body)?["n_ctx"]
            .as_u64()
            .unwrap_or(u64::MAX);
        println!("ctx arm placed: props n_ctx {n} of trained {trained}");
        check(ok, "ctx_never_passes_the_trained_context", n <= trained);
        // The placed default is chosen over the plan's own
        // expert split under the common rule, and the relation is what
        // holds on every card — the fit rides the census reading: the one
        // `ctx` line's total is the total the one `plan` record was made
        // at, and `/props` serves a slot's share of it. FAIL-first: a
        // choice that ignores its bound plans past what the card holds
        // and the server dies before listening — on the 3090, a
        // Qwen3.6 plan that keeps every expert at a context its
        // whole load (which that plan opens) does not fit.
        // PIN(2026-10-05): the total is the plan's `ctx_max`, not
        // `/props`' `n_ctx`, and on both files each of the two
        // resident placed slots serves `ctx_max / 2` rows (the
        // split's floor: the seat splits the total the plan was
        // made at across `--parallel 2`, `slot_ctx`); a qwen35moe
        // file's slot was `ctx_max` while its slots took the one
        // sequence in turns. Mutant: the seat's qwen35moe placed
        // load kept on one sequence serves `ctx_max`.
        let log = std::fs::read_to_string(&err_log)?;
        let records = seat_log(&err_log)?;
        let total = records.one(&record::PLAN38)?.u64("ctx_max")?;
        let slot = total / 2;
        println!("placed arm: plan ctx_max {total}, props n_ctx {n} against a slot's {slot}");
        check(ok, "placed_ctx_is_the_plans_ctx_max", n == slot);
        // PIN(2026-10-11): `placed_names_its_default_ctx` read the seat's
        // `--ctx defaults to` lines, present when the search stopped under the
        // trained context; the seat prints the common rule's one `ctx` line
        // on every unset `--ctx` now, so the check is this one, which holds
        // the line's word and numbers (the total the plan's `ctx_max`). Re-pin
        // by specification (the seat's lines, one rule for every seat), not a
        // relaxation: the default is held to more terms than the old
        // biconditional on the trained context.
        default_ctx_line_names_its_rule(&log, (2, n, trained), Some(total), ok);
        // The two slots are resident sequences of a slot's
        // share each, on both files (the module header).
        let load = records.one(&record::LOAD_QWEN3)?;
        let listening = records.one(&record::LISTENING_QWEN3)?;
        let terms = [
            load.u64("slots")?,
            load.u64("slot_ctx")?,
            listening.u64("slots")?,
            listening.u64("slot_ctx")?,
        ];
        println!(
            "placed arm: load slots={} slot_ctx={}, listening slots={} \
             slot_ctx={}",
            terms[0], terms[1], terms[2], terms[3]
        );
        // PIN(2026-10-05): `placed_slots_take_turns` (qwen35moe,
        // `[1, ctx_max, 2, ctx_max]`) turned into this clause's
        // qwen35moe arm: the seat's qwen35moe slots are resident
        // sequences (`Slot35`), so the turns it pinned are gone.
        check(ok, "placed_slots_are_resident", terms == [2, slot, 2, slot]);
        // Every choice the unset `--ctx` ran — the placed one a run under
        // `--place` runs — read the census once for all its probes (the
        // seat's line a choice). FAIL-first: a choice that resolves its
        // placement per probe reads the census once a probe.
        let log = std::fs::read_to_string(&err_log)?;
        let runs = choices(&log);
        println!("ctx arm placed: choices (probes, census readings) {runs:?}");
        check(
            ok,
            "ctx_choice_reads_the_census_once",
            !runs.is_empty() && runs.iter().all(|&(p, c)| p >= 1 && c == 1),
        );
        println!("ctx arm placed: server stopped: {}", s.stop()?);
        if body35(seat) && !flash_has_q8(&split)? {
            return q8_cache_is_refused_on_the_scalar_pair_flash(model, dir, default_n, ok);
        }
        let resident = |log: &str| -> Result<u64, GateError> {
            Ok(record::Log::of(log, record::BLOOMERY_SERVE_QWEN3)
                .one(&record::LOAD_QWEN3)?
                .u64("resident_bytes")?)
        };
        let kv_heads = split
            .arch_get_u64("attention.head_count_kv")
            .or_else(|| split.arch_get_u64("attention.head_count"))
            .ok_or("no attention.head_count(_kv)")?;
        let head_dim = split
            .arch_get_u64("attention.key_length")
            .ok_or("no attention.key_length")?;
        let layers = split.arch_get_u64("block_count").ok_or("no block_count")?;
        // The two loads compare at one context: the default can take a
        // larger context under q8_0 than under f16 where the card is
        // short at f16's, so the q8_0 server is held to the main server's
        // context by `--ctx`, started once the placed arm's server has
        // stopped (two loads do not fit one card).
        let same_n = default_n;
        let same_dir = dir.join("ctx-q8-same");
        std::fs::create_dir_all(&same_dir)?;
        let same_err = same_dir.join("server.err");
        let mut same_cmd = Command::new(beside("bloomery-serve")?);
        same_cmd.env_remove("BLOOMERY_REF_MODEL");
        let same_ctx = same_n.to_string();
        let same_args = [
            "--model",
            "qwen3",
            "--port",
            "0",
            "-m",
            m,
            "--parallel",
            "1",
            "--cache-type-k",
            "q8_0",
            "--ctx",
            same_ctx.as_str(),
        ];
        let mut same = Served::spawn_cmd(same_cmd, &same_args, &same_dir)?;
        same.address(&same_err, 600, Duration::from_secs(1))?;
        println!(
            "ctx arm q8 at the f16 context: server stopped: {}",
            same.stop()?
        );
        let same_log = std::fs::read_to_string(&same_err)?;
        let ceiling = layers * same_n * kv_heads * head_dim * 15 / 8;
        let f16_log = std::fs::read_to_string(dir.join("server.err"))?;
        let (f16_resident, q8_resident) = (resident(&f16_log)?, resident(&same_log)?);
        let d = f16_resident.saturating_sub(q8_resident);
        // PIN(2026-10-22): a placed plan spends the bytes the q8 planes free
        // on more card experts, so the resident no longer falls by the planes'
        // saving. The clause holds the whole-fit verdict's cache term (the
        // planes' bytes) and, where both loads print a plan, that the q8 plan
        // keeps more experts on the card at the same context. A card the whole
        // load fits prints no plan, and there the resident falls by that term.
        // A qwen3moe file's every layer carries the planes, so the cache term's
        // drop is the ceiling; a qwen35moe file's delta layers carry none, so
        // the same product is only the upper bound (their exact 17/32 term is
        // the model crate's own unit pin).
        // FAIL-first: a q8 load that keeps the f16 planes — the cache term does
        // not drop, and a placed plan does not take more card experts.
        let f16_cache = cache_term(&f16_log, same_n)?;
        let q8_cache = cache_term(&same_log, same_n)?;
        let plane_drop = f16_cache.saturating_sub(q8_cache);
        // PIN(2026-10-30): each plane is allocated whole granules, so at a context that is no
        // power of two the drop is the granule-rounded planes' difference, not the raw
        // ceiling: f16 two planes of 2b bytes, q8_0 two of codes b and two of scales b/16, b
        // the K or V values of a layer (kv_heads x n x head_dim). The default context became
        // a 1024-step (cd84d529f); at 226304 on Qwen3-30B the clause was red at 153 MiB under
        // the ceiling, which this term equals.
        let g = serve_granule()?.get();
        let b = kv_heads * same_n * head_dim;
        let granule_drop = layers
            * (2 * (2 * b).next_multiple_of(g)
                - 2 * b.next_multiple_of(g)
                - 2 * (b / 16).next_multiple_of(g));
        let planes_held = match split.architecture() {
            Some("qwen3moe") => plane_drop == granule_drop,
            _ => plane_drop > 0 && plane_drop <= ceiling,
        };
        let f16_split = routed_split(&record::Log::of(&f16_log, record::BLOOMERY_SERVE_QWEN3))?;
        let q8_split = routed_split(&record::Log::of(&same_log, record::BLOOMERY_SERVE_QWEN3))?;
        let spent = match (f16_split, q8_split) {
            (Some((_, fc, fx)), Some((_, qc, qx))) => {
                println!(
                    "ctx arm q8: placed, card_experts f16 {fc} q8_0 {qc}, ctx_max {fx} and {qx}"
                );
                qc > fc && fx == same_n && qx == same_n
            }
            (None, None) => {
                println!(
                    "ctx arm q8: both loads whole, the resident's drop {d} against the planes' \
                     {plane_drop}"
                );
                d == plane_drop
            }
            _ => {
                println!("ctx arm q8: one load placed and the other whole");
                false
            }
        };
        println!(
            "ctx arm q8: cache f16 {f16_cache} q8_0 {q8_cache}, drop {plane_drop} against ceiling \
             {ceiling}; residents f16 {f16_resident} q8_0 {q8_resident} at context {same_n}",
        );
        check(ok, "q8_resident_drops_by_the_planes", planes_held && spent);
        // The load names the format it holds. FAIL-first: a flag the seat
        // ignores loads the f16 planes and says so.
        let caches = seat_log(&same_err)?
            .all(&record::LOAD_QWEN3)?
            .iter()
            .map(|l| l.word("cache").map(str::to_owned))
            .collect::<Result<Vec<_>, _>>()?;
        println!("ctx arm q8: the load records' cache {caches:?}");
        check(ok, "q8_load_names_its_cache", caches == ["q8_0"]);
        Ok(())
    }

    /// Whether the file's decode flash has a q8_0 cache pass: `Body35`'s
    /// tensor-core pass, the eight-head layout's (`decode_mma`), is the one that
    /// reads a q8_0 K cache; the scalar pair pass of a group of six has none.
    fn flash_has_q8(split: &Split) -> Result<bool, GateError> {
        let spec = model::arch::spec(split)?.spec;
        let head = spec
            .gqa()
            .ok_or("the description holds no attention layer to choose the decode flash by")?;
        decode_mma(
            head.head_dim as usize,
            (head.heads / head.kv_heads) as usize,
        )
    }

    /// `q8_cache_is_refused_on_the_scalar_pair_flash` (module header), on a
    /// `Body35` file whose head shape runs the scalar pair flash (the dense
    /// 27B: 24 query heads over 4 key-value heads): the server of
    /// `--cache-type-k q8_0` at the default arm's context exits before it
    /// listens, and its stderr names the cache's flash as the reason. The
    /// q8_0 plane and cache-word clauses (`q8_resident_drops_by_the_planes`,
    /// `q8_load_names_its_cache`) print `n/a` for it. FAIL-first: a server
    /// that listens on this file (a pair pass for the q8_0 cache) leaves the
    /// clause red; its owner then replaces the clause with those two.
    fn q8_cache_is_refused_on_the_scalar_pair_flash(
        model: &Path,
        dir: &Path,
        default_n: u64,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        not_for_dense(
            "q8_resident_drops_by_the_planes, q8_load_names_its_cache",
            "the q8_0 cache has no scalar pair flash",
        );
        let d = dir.join("ctx-q8-refused");
        std::fs::create_dir_all(&d)?;
        let err_log = d.join("server.err");
        let mut cmd = Command::new(beside("bloomery-serve")?);
        cmd.env_remove("BLOOMERY_REF_MODEL");
        let m = model.to_str().ok_or("the model path is not UTF-8")?;
        let ctx = default_n.to_string();
        let args = [
            "--model",
            "qwen3",
            "--port",
            "0",
            "-m",
            m,
            "--parallel",
            "1",
            "--cache-type-k",
            "q8_0",
            "--ctx",
            ctx.as_str(),
        ];
        let mut s = Served::spawn_cmd(cmd, &args, &d)?;
        let listened = s.address(&err_log, 600, Duration::from_secs(1)).is_ok();
        if listened {
            println!("q8 arm: server stopped: {}", s.stop()?);
        }
        let said = std::fs::read_to_string(&err_log)?;
        let named = said.contains("a q8_0 cache's flash");
        println!("q8 arm: listened {listened}, names the cache's flash {named}");
        check(
            ok,
            "q8_cache_is_refused_on_the_scalar_pair_flash",
            !listened && named,
        );
        Ok(())
    }

    /// `ctx_default_is_the_largest_step_held`, on the default arm's server
    /// (no `--ctx`, `--parallel 1`, its stderr `log`, its default `n`): the
    /// default is the largest step under the target the load holds. `n` is a
    /// context the default takes ([`default_fits`]); the seat's one whole-fit
    /// line at `n` — the census reading the load was decided on, its `fits`,
    /// what the card had and the need term by term — with the cache term
    /// moved from `n` rows to the step above `n` ([`step_above`]) says the
    /// whole load did not fit there, so the seat did not fit under a context
    /// the card held whole (the placed plan there was short too, which the
    /// `ctx` line's `fitted` word holds). The need is a sum (`Card::floor_bytes`) whose
    /// other terms read no context at or past the floor, and the cache at a
    /// context is the allocator's bytes of [`cache_planes`] there, the list
    /// [`whole_fit_counts_the_planes_granules`] holds equal to the verdict's
    /// own. A line missing or without its terms is a named error.
    /// FAIL-first: a default that starts a step under the target defaults
    /// under the target where the whole load fits the target, red here.
    fn ctx_default_is_the_largest_step_held(
        model: &Path,
        split: &Split,
        log: &str,
        n: u64,
        trained: u64,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let head = format!("whole fit at ctx {n}: ");
        let line = log
            .lines()
            .find(|l| l.starts_with(&head))
            .ok_or_else(|| format!("the default arm's seat printed no whole-fit line at {n}"))?;
        let WholeTerms {
            need,
            rest,
            budget,
            census,
            fits: whole,
            ..
        } = whole_terms(line)?;
        // The whole load's need at `ctx` positions, the line's other terms kept.
        let granule = serve_granule()?;
        let need_at = |ctx: u64| -> Result<u64, GateError> {
            let planes = cache_planes(model, split, ctx, default_lanes(split))?;
            rest.checked_add(model::placement::allocator_bytes(granule, planes))
                .ok_or_else(|| format!("the whole load's need at {ctx} passes u64 bytes").into())
        };
        let target = target_of(trained);
        let above = step_above(trained, n);
        let at_above = above.map(need_at).transpose()?;
        println!(
            "default arm: the default {n} under the target {target} (the file's trained {trained}),              the whole load {} there; it needs {need} B at {n}, {at_above:?} B at the step above              {above:?}, of {budget} B the card had ({census})",
            if whole { "fits" } else { "does not fit" },
        );
        let held =
            default_fits(trained, n) && (n == target || at_above.is_some_and(|b| b > budget));
        check(ok, "ctx_default_is_the_largest_step_held", held);
        Ok(())
    }

    /// The arena and the reserve past it one whole-fit `line` counts
    /// (`+ arena A B + reserve past it R B`); `None` for a line that names
    /// the margin in their place.
    fn arena_terms(line: &str) -> Option<(u64, u64)> {
        let bytes = |after: &str| -> Option<u64> {
            line.split(after).nth(1)?.split(" B").next()?.parse().ok()
        };
        Some((bytes(" + arena ")?, bytes(" B + reserve past it ")?))
    }

    /// `default_load_counts_its_arena`, on the default arm's server (its
    /// stderr `log`, its default `n`, the load record's `arch`): the seat's
    /// whole-fit line at `n` counts the arena the load itself holds. A
    /// qwen35moe or qwen35 file's line names `arena A B + reserve past it R B`, the
    /// pair `Qwen35moeModel::whole_load` hands the verdict for this file at
    /// `n` rows and the load's own ubatch, read here from the file; a
    /// qwen3moe body allocates its arena with no fit check and has no
    /// formula for it, so its line names the margin in the arena's place.
    /// The terms are the file's, so the clause holds on every card, where
    /// `default_seat_listens` sees a need without its arena only on a card
    /// whose room past the weights and the cache is under the arena.
    /// FAIL-first: a seat whose whole load carries no arena prints
    /// `arena 0 B` and is red here on a card with room to spare.
    fn default_load_counts_its_arena(
        split: &Split,
        log: &str,
        n: u64,
        arch: &str,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let head = format!("whole fit at ctx {n}: ");
        let line = log
            .lines()
            .find(|l| l.starts_with(&head))
            .ok_or_else(|| format!("the default arm's seat printed no whole-fit line at {n}"))?;
        let held = match qwen_seat::Family::of(split).ok() {
            Some(seat) if body35(seat) => {
                // The open the serve seat makes, from its one owner; a file with no routed expert
                // settles its ubatch in the verdict, which the seat's `ubatch` line at `n` names.
                let at = usize::try_from(n)?;
                let o = q3place::serve_open35(split, at, KvQ8::F16, false)?;
                let o = match ubatch_pick::choices(log)
                    .iter()
                    .rev()
                    .find(|c| c.ctx == at)
                    .and_then(|c| c.pick)
                {
                    Some(ubatch) => Open35 { ubatch, ..o },
                    None => o,
                };
                let WholeLoad::Checked { arena, reserve } = Qwen35moeModel::whole_load(split, o)?
                else {
                    return Err(format!("a {arch} whole load carries no arena term").into());
                };
                println!(
                    "default arm: the load's arena {arena} B + reserve {reserve} B at {n}; the \
                     line names {:?}",
                    arena_terms(line)
                );
                arena_terms(line) == Some((arena, reserve))
            }
            Some(_) => {
                println!("default arm: the qwen3moe line names the margin: {line}");
                arena_terms(line).is_none() && line.contains("(the program's arena under it)")
            }
            None => {
                return Err(format!("the default arm's load names architecture {arch:?}").into());
            }
        };
        check(ok, "default_load_counts_its_arena", held);
        Ok(())
    }

    /// The ctx [`whole_fit_counts_the_planes_granules`] runs at: one row
    /// past an odd multiple of 1024, over the seat's floor, so a qwen3moe
    /// file's f16 plane (1 KiB a row) is no whole number of granules of any
    /// power-of-two size past 1 KiB, and the allocator's rounding moves the
    /// verdict's cache term off the raw sum on any card.
    const GRANULE_CTX: usize = 5 * 1024 + 1;

    /// `whole_fit_counts_the_planes_granules`: the whole-fit verdict's cache
    /// term is the allocator's bytes of the planes the body allocates, one
    /// sequence at [`GRANULE_CTX`] rows. A server at that flagged ctx prints
    /// its one whole-fit line before any load — fits or not, the term prints
    /// either way — and the clause reads the line's cache bytes and holds
    /// them equal to `allocator_bytes` over the file's own planes at the
    /// card's granule: f16's `k` and `v` a plain-attention layer, a
    /// qwen35moe file's recurrent layers' `state` and `ring`
    /// (`LinearShape`'s lengths, one lane as `Body35` allocates them).
    /// FAIL-first: the raw sum the verdict counted before the planes went
    /// through the allocator is 95.9 MiB short at this ctx on the Qwen3-30B
    /// file at a 2 MiB granule (every plane 1023 KiB under the granules it
    /// takes) and about 21.7 MiB on the Qwen3.6 one (its GQA planes just past
    /// 2.5 granules each, its delta rings sharing granules), so the clause is
    /// red on it.
    fn whole_fit_counts_the_planes_granules(
        model: &Path,
        dir: &Path,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let split = Split::open(model).map_err(|e| format!("open {}: {e}", model.display()))?;
        let planes = cache_planes(model, &split, GRANULE_CTX as u64, default_lanes(&split))?;
        let expected = model::placement::allocator_bytes(serve_granule()?, planes.iter().copied());
        let raw: u64 = planes.iter().sum();
        let d = dir.join("ctx-granule");
        std::fs::create_dir_all(&d)?;
        let err_log = d.join("server.err");
        let mut cmd = Command::new(beside("bloomery-serve")?);
        cmd.env_remove("BLOOMERY_REF_MODEL");
        let m = model.to_str().ok_or("the model path is not UTF-8")?;
        let ctx = GRANULE_CTX.to_string();
        let args = [
            "--model",
            "qwen3",
            "--port",
            "0",
            "-m",
            m,
            "--parallel",
            "1",
            "--ctx",
            ctx.as_str(),
        ];
        let mut s = Served::spawn_cmd(cmd, &args, &d)?;
        // The verdict's line prints before the open, so a listening server
        // has printed it. The flag wins over the default: `/props` names the
        // context it named. FAIL-first: a seat that ignores the flag serves
        // its default.
        let addr = s.address(&err_log, 600, Duration::from_secs(1))?;
        let (st, body) = curl(&format!("http://{addr}/props"), None, false)?;
        let n = json_of("/props", st, &body)?["n_ctx"]
            .as_u64()
            .unwrap_or(u64::MAX);
        println!("granule arm: props n_ctx {n} at the flagged {GRANULE_CTX}");
        check(ok, "ctx_flag_wins", n == GRANULE_CTX as u64);
        let log = std::fs::read_to_string(&err_log)?;
        println!("granule arm: server stopped: {}", s.stop()?);
        let line = log
            .lines()
            .find(|l| l.starts_with(&format!("whole fit at ctx {GRANULE_CTX}:")))
            .ok_or("the seat printed no whole-fit line at the flagged ctx")?;
        let said = line
            .split(" + cache ")
            .nth(1)
            .and_then(|t| t.split(" B").next())
            .and_then(|t| t.parse::<u64>().ok())
            .ok_or("the whole-fit line names no cache bytes")?;
        println!(
            "granule arm: the verdict's cache {said} B of {raw} B of planes, the allocator's \
             bytes {expected} B"
        );
        check(ok, "whole_fit_counts_the_planes_granules", said == expected);
        Ok(())
    }

    /// The lanes a delta store of the seat's default load holds on `split`: a
    /// file that carries a usable next-token layer serves the MTP draft with
    /// the draft unset (one slot, the whole load, no media), and that load
    /// verifies on [`Body35::VERIFY_LANES`] lanes; any other file one lane.
    fn default_lanes(split: &Split) -> usize {
        if matches!(in_file(split), Ok(InFile::Draft(_))) {
            Body35::VERIFY_LANES
        } else {
            1
        }
    }

    /// The cache planes the body allocates for one sequence of `ctx` rows of
    /// `model`'s file (`split`), in its allocation order — the order the
    /// allocator's shared granules are counted in: f16's `k` and `v` a
    /// plain-attention layer, a qwen35moe file's recurrent layers' `state`
    /// and `ring` (`LinearShape`'s lengths), which no context moves: the
    /// state `lanes` times, and a load of more than one lane a word a lane
    /// past the ring (`RecStore`'s stamps), as `Body35` allocates them. A
    /// qwen35moe file's layers are the description's trunk: a next-token
    /// layer past the trunk holds no sequence plane.
    fn cache_planes(
        model: &Path,
        split: &Split,
        ctx: u64,
        lanes: usize,
    ) -> Result<Vec<u64>, GateError> {
        let num = |key: &str| -> Result<u64, GateError> {
            split
                .arch_get_u64(key)
                .ok_or_else(|| format!("{}: no {key}", model.display()).into())
        };
        let layers = num("block_count")? as usize;
        let kv_heads = split
            .arch_get_u64("attention.head_count_kv")
            .or_else(|| split.arch_get_u64("attention.head_count"))
            .ok_or_else(|| format!("{}: no attention.head_count(_kv)", model.display()))?
            as usize;
        let head_dim = num("attention.key_length")? as usize;
        let plane = kv_heads as u64 * ctx * head_dim as u64 * 2;
        let mut planes = Vec::new();
        match split.architecture() {
            Some("qwen3moe") => {
                for _ in 0..layers {
                    planes.push(plane);
                    planes.push(plane);
                }
            }
            Some("qwen35moe" | "qwen35") => {
                let shape = LinearShape {
                    n_k: num("ssm.group_count")? as usize,
                    n_v: num("ssm.time_step_rank")? as usize,
                    // Neither plane's length reads the map; a qwen35moe
                    // file's delta layers are the tiled ones.
                    map: KHeadMap::Tiled,
                };
                let trunk = model::arch::spec(split)
                    .map_err(|e| format!("{}: {e}", model.display()))?
                    .spec;
                for (l, layer) in trunk.layers.iter().enumerate() {
                    match &layer.mixer {
                        Mixer::DeltaRule(_) => {
                            planes.push(4 * lanes as u64 * shape.state_len() as u64);
                            planes.push(4 * shape.ring_len() as u64);
                            if lanes > 1 {
                                planes.push(4 * lanes as u64);
                            }
                        }
                        Mixer::Gqa(_) => {
                            planes.push(plane);
                            planes.push(plane);
                        }
                        other => {
                            return Err(format!(
                                "{}: layer {l} mixes with {}, which holds no planes listed here",
                                model.display(),
                                other.what()
                            )
                            .into());
                        }
                    }
                }
            }
            other => {
                return Err(format!(
                    "{} is a {other:?} file; the planes are listed for qwen3moe, qwen35moe and qwen35",
                    model.display()
                )
                .into());
            }
        }
        Ok(planes)
    }

    /// `bloomery-serve --model qwen3 -m <model> --port 0 --parallel 1` beside
    /// this binary, `BLOOMERY_REF_MODEL` removed, its logs in `dir`. The
    /// plain engine is pinned: the prefix clauses below hold the one-slot
    /// path's keeps.
    fn spawn(model: &Path, dir: &Path) -> Result<Served, GateError> {
        let m = model.to_str().ok_or("the model path is not UTF-8")?;
        Served::spawn_cmd(
            serve_cmd()?,
            &[
                "--model",
                "qwen3",
                "--port",
                "0",
                "--parallel",
                "1",
                "-m",
                m,
            ],
            dir,
        )
    }

    /// `/completion` of `prompt` at temperature 0 for [`N`] tokens — `cache`
    /// asks the server to keep the prefix the prompt shares with the slot —
    /// its body in `<dir>/<name>.json`.
    fn greedy(
        url: &dyn Fn(&str) -> String,
        prompt: Vec<u32>,
        dir: &Path,
        name: &str,
        cache: bool,
    ) -> Result<(Answer, Value), GateError> {
        serve_client::greedy(url, prompt, N, dir, name, cache)
    }

    /// The prefix clauses (the module header), on the qwen3moe arm alone:
    /// the edit clause — the turn answered, then resent with its user
    /// message changed — diverges at `j`, inside the rows the first run's
    /// prompt call wrote, with at least [`GEMM_FROM`] ids after it, keeps
    /// `j` (`cache_n`) and answers the ids of the same ids fed fresh; the
    /// extension clause — the turn resent with its reply and [`LATER`]'s
    /// user turn — keeps every position the slot held, its ids printed
    /// only; the whole-call clause — a chat whose prompt opens a system and
    /// a user message, each at least [`GEMM_FROM`] ids, on a slot holding
    /// another conversation — runs as one prompt call: no `prefill split`
    /// note (the per-position cache keeps every mark as it stands, so a cut
    /// would buy nothing and cost a walk), and the answer it returns is
    /// held to the CLI's below, a fresh run's ids.
    fn prefix(
        url: &dyn Fn(&str) -> String,
        dir: &Path,
        ok: &mut bool,
    ) -> Result<Answer, GateError> {
        let later = tokenized(url, LATER)?;
        // The edit clause and the extension clause, the owner's.
        edit_and_extension(
            url,
            dir,
            ok,
            &PrefixTurns {
                a: EDIT_A,
                b: EDIT_B,
                n: N,
                min_after: GEMM_FROM,
            },
            &later,
        )?;

        // The whole-call clause: the two-message chat on the slot the
        // extension left holding another conversation. Each message at
        // least GEMM_FROM ids, so the user message's start sits GEMM_FROM
        // ids inside the call either side — a mark a cutting rule could
        // take.
        let n_sys = tokenized(url, SYSTEM_MSG)?.len();
        let n_user = tokenized(url, EDIT_A)?.len();
        if n_sys < GEMM_FROM || n_user < GEMM_FROM {
            return Err(format!(
                "the whole-call clause's messages hold {n_sys} and {n_user} ids; each needs at \
                 least {GEMM_FROM} so a message-start mark sits inside the call"
            )
            .into());
        }
        let chat = rendered(
            url,
            json!([
                { "role": "system", "content": SYSTEM_MSG },
                { "role": "user", "content": EDIT_A },
            ]),
        )?;
        let (whole, _) = greedy(url, chat, dir, "whole", true)?;
        let log = std::fs::read_to_string(dir.join("server.err"))?;
        let cuts = log.lines().filter(|l| l.contains("prefill split")).count();
        println!(
            "whole-call: the chat holds {} ids; the server's log carries {cuts} prefill split \
             note(s)",
            whole.prompt.len(),
        );
        check(ok, "two_message_chat_runs_one_call", cuts == 0);
        Ok(whole)
    }

    /// The prefix clauses of the qwen35moe arm (the module header): the
    /// stripped clause — the turn answered, then resent with a
    /// reasoning-free reply in place of the reply, so the shared prefix
    /// ends at the turn's prompt end — keeps the checkpoint there and
    /// answers the ids of the same ids fed fresh; the extension clause —
    /// the turn resent with its reply and [`LATER`]'s user turn — keeps
    /// every position the slot held, its ids printed only.
    fn prefix35(url: &dyn Fn(&str) -> String, dir: &Path, ok: &mut bool) -> Result<(), GateError> {
        let p1 = rendered(url, json!([{ "role": "user", "content": EDIT_A }]))?;
        let later = tokenized(url, LATER)?;
        let stripped = tokenized(url, STRIPPED_35)?;
        if stripped.len() < GEMM_FROM || later.len() < GEMM_FROM {
            return Err(format!(
                "the prefix clauses' resends hold {} and {} ids past the turn; each needs at \
                 least {GEMM_FROM}",
                stripped.len(),
                later.len()
            )
            .into());
        }
        // The stripped clause: the turn answered fresh, resent with the
        // reasoning-free reply, and the same ids fed fresh.
        let (held_run, _) = greedy(url, p1.clone(), dir, "strip35_first", false)?;
        if held_run.tokens.is_empty() || stripped.first() == held_run.tokens.first() {
            return Err(format!(
                "the reply's first id {:?} is the stripped reply's {:?}: the shared prefix \
                 would not end at the turn",
                held_run.tokens.first(),
                stripped.first()
            )
            .into());
        }
        let mut strip = p1.clone();
        strip.extend_from_slice(&stripped);
        let (resend, _) = greedy(url, strip.clone(), dir, "strip35", true)?;
        let (fresh, _) = greedy(url, strip, dir, "strip35_fresh", false)?;
        let turn_end = p1.len() as u64 - 1;
        println!(
            "strip35: the turn ends at {turn_end}; the resend kept {} and answered {:?}; the \
             fresh run's {:?}",
            resend.cache_n, resend.tokens, fresh.tokens
        );
        check(
            ok,
            "qwen35_stripped_resend_keeps_the_turns_end",
            held_run.cache_n == 0 && resend.cache_n == turn_end,
        );
        check(
            ok,
            "qwen35_stripped_resend_ids_are_a_fresh_runs",
            !resend.tokens.is_empty() && resend.tokens == fresh.tokens && fresh.cache_n == 0,
        );

        // The extension clause: the turn resent with its reply and a later
        // user turn. The stripped clause's runs left the slot holding the
        // stripped conversation, so the fresh turn runs again first — the
        // slot back at the reply's end, its prompt call's end mark standing.
        let (held_again, _) = greedy(url, p1, dir, "extend35_first", false)?;
        if held_again.tokens != held_run.tokens {
            return Err(
                "the extension's fresh turn answered ids the first one did not: the clause                  cannot hold a shared prefix"
                    .into(),
            );
        }
        let held = (held_again.prompt.len() + held_again.tokens.len()).saturating_sub(1) as u64;
        let mut extend = held_again.prompt.clone();
        extend.extend_from_slice(&held_again.tokens);
        extend.extend_from_slice(&later);
        let (resend, _) = greedy(url, extend, dir, "extend35", true)?;
        println!(
            "extension35: the slot held {held}; the resend kept {} and answered {:?}",
            resend.cache_n, resend.tokens
        );
        check(
            ok,
            "qwen35_extension_keeps_every_held_position",
            resend.cache_n == held && !resend.tokens.is_empty(),
        );
        Ok(())
    }

    /// `tool_call_is_parsed` (module header), on a `Body35` file's chat template: a server over a
    /// scripted engine whose every reply is the template's XML call, asked one tool turn at
    /// temperature 0 and thinking off, answers 200 with `finish_reason` `tool_calls` and one call of
    /// the listed function with the arguments the script gave (`serve_client::scripted_call`). The
    /// markup is the file's template's (`<tool_call><function=…>`), the format the reader binds and the
    /// server parses; the clause reads nothing the model chose to reply. FAIL-first: a scanner that
    /// drops a call's parameters answers `{}` and the clause is red.
    fn tool_call_is_parsed(split: &Split, dir: &Path, ok: &mut bool) -> Result<(), GateError> {
        let template = model::arch::spec(split)
            .map_err(|e| format!("the header: {e}"))?
            .spec
            .chat
            .template
            .ok_or("the file carries no chat template")?;
        check(ok, "tool_call_is_parsed", scripted_call(&template, dir)?);
        Ok(())
    }

    /// One streamed request's answer: the final event's tokens.
    struct Streamed {
        tokens: Vec<u32>,
    }

    /// A streamed request's answer channel: what its reader thread sends
    /// when the stream ends.
    type StreamedRx = mpsc::Receiver<Result<Streamed, String>>;

    /// `/completion` of `ids` at temperature 0, streamed: a helper thread of
    /// its own runs [`curl`] and reads its SSE body ([`sse_events`]). The
    /// final event carries the tokens.
    fn streamed(
        addr: &str,
        ids: &[u32],
        n: usize,
    ) -> Result<(JoinHandle<()>, StreamedRx), GateError> {
        streamed_body(
            addr,
            json!({
                "prompt": ids, "n_predict": n, "temperature": 0, "return_tokens": true,
                "cache_prompt": false, "ignore_eos": true, "stream": true,
            }),
        )
    }

    /// A streamed request with its own sampler and logits requirements.
    fn streamed_body(addr: &str, body: Value) -> Result<(JoinHandle<()>, StreamedRx), GateError> {
        let (tx, rx) = mpsc::channel();
        let url = format!("http://{addr}/completion");
        let (h, _) = spawn_helper("slots-request", Placement::Float, move || {
            let run = (|| -> Result<Streamed, String> {
                let (status, sse) =
                    curl(&url, Some(&body), true).map_err(|e| format!("curl {url}: {e}"))?;
                if status != 200 {
                    return Err(format!("curl {url}: HTTP {status}: {sse}"));
                }
                let mut tokens = Vec::new();
                for e in sse_events(&sse).map_err(|e| format!("curl {url}: {e}"))? {
                    let v = sse_event(e).map_err(|e| format!("curl {url}: {e}"))?;
                    if v.get("tokens").is_some() {
                        tokens = ids_of(&v["tokens"]).map_err(|e| format!("curl {url}: {e}"))?;
                    }
                }
                Ok(Streamed { tokens })
            })();
            let _ = tx.send(run);
        })
        .map_err(|e| format!("slots: {}", e.what()))?;
        Ok((h, rx))
    }

    /// Where a slots arm's total context comes from: the arm's own `--ctx`
    /// (a set total, which its one whole-fit line is at), or its own `plan`
    /// record (`ctx_max`: the placed default's answer, the total the plan
    /// was made at).
    #[derive(Clone, Copy, Debug)]
    enum Total {
        Set,
        Plan,
    }

    /// One server of the slots clause (module header): its directory under
    /// the file's, the prefix of its checks' names, the flags past
    /// `--parallel 2`, its environment, and where its total comes from.
    struct SlotsArm<'a> {
        dir: &'a str,
        prefix: &'a str,
        flags: &'a [&'a str],
        env: &'a [(&'a str, &'a str)],
        total: Total,
    }

    /// The whole-card arm: no `--place`, a set `--ctx` of `ctx_ladder::FLOOR` a
    /// slot, which the whole load holds on every card the gate runs on (an
    /// unset `--ctx` asks the target, which a card may hold only placed).
    const WHOLE_SLOTS: SlotsArm<'static> = SlotsArm {
        dir: "slots",
        prefix: "",
        flags: &["--ctx", "8192"],
        env: &[],
        total: Total::Set,
    };

    /// The slots clause's placed arm (its checks `placed_…`): `--place a`
    /// under a card budget of [`PLACED_BUDGET`], no `--ctx` — the placed
    /// default the plan record names, split.
    const PLACED_SLOTS: SlotsArm<'static> = SlotsArm {
        dir: "slots-placed",
        prefix: "placed_",
        flags: &["--place", "a"],
        env: &[("BLOOMERY_CARD_BUDGET", PLACED_BUDGET)],
        total: Total::Plan,
    };

    /// The placed slots arm's card budget: a 12 GiB card's, the smallest
    /// card the seat serves Qwen3-30B on, under which the plan splits the
    /// routed experts between the card and the host tier on either of this
    /// machine's cards (`gate_qwen3moe_e2e`'s (o) budget).
    const PLACED_BUDGET: &str = "12G";

    /// The card budget that stands for a 12 GB card whose census reads
    /// 11,789,139,968 free bytes: those bytes plus the context's own creation
    /// cost (`workstation::CONTEXT_SELF`), which a plan's cap adds back to a
    /// census reading.
    const SMALL_CARD_BUDGET: &str = "12073304064";

    /// The host room of the machine that card sits in, 27.5 GiB (the lever
    /// takes whole MiB or GiB).
    const SMALL_HOST_ROOM: &str = "28160M";

    /// The slot context the seat's defaults must reach on that machine: a
    /// prompt of up to 8K tokens fits one slot.
    const LONG_PROMPT_SLOT_CTX: u64 = 8192;

    /// The slots clause (module header) on a server of `--parallel 2` and
    /// `arm`'s flags and environment started into `<dir>/<arm.dir>`: two
    /// greedy streamed requests of distinct prompts, each first run alone on
    /// that same server and then both together, the ids and the round
    /// counters held, the load record's split and
    /// `/props`' context pinned to half the arm's total. Each check's name
    /// carries the arm's prefix.
    fn slots_flow_together(
        model: &Path,
        dir: &Path,
        arm: &SlotsArm<'_>,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let named = |c: &str| format!("{}{c}", arm.prefix);
        let dir = dir.join(arm.dir);
        std::fs::create_dir_all(&dir)?;
        let err_log = dir.join("server.err");
        let mut cmd = Command::new(beside("bloomery-serve")?);
        cmd.env_remove("BLOOMERY_REF_MODEL");
        for &(k, v) in arm.env {
            cmd.env(k, v);
        }
        let m = model.to_str().ok_or("the model path is not UTF-8")?;
        let mut args = vec![
            "--model",
            "qwen3",
            "--port",
            "0",
            "--parallel",
            "2",
            "-m",
            m,
        ];
        args.extend_from_slice(arm.flags);
        let mut s = Served::spawn_cmd(cmd, &args, &dir)?;
        let addr = match s.address(&err_log, 600, Duration::from_secs(1)) {
            Ok(a) => a,
            // A load that never listens names no split: the split's check
            // is red by name, and the clauses after this one still run.
            Err(e) => {
                println!("{}: the server never listened: {e}", arm.dir);
                check(ok, &named("slots_load_names_the_split"), false);
                let _ = s.stop();
                return Ok(());
            }
        };
        let url = |p: &str| format!("http://{addr}{p}");
        // (v4) The load names the split and /props the slot ctx.
        let log = std::fs::read_to_string(&err_log)?;
        let records = seat_records(&log, &err_log);
        let load = or_named(records.one(&record::LOAD_QWEN3));
        let field = |name: &str| u64_of(load.as_ref(), name);
        // The placed arm's total is its one `plan` record's `ctx_max`, so
        // the record is due there; the whole arm makes no plan of its own
        // and prints one only for a placed load, but a line that opens one
        // and does not read whole is no read on either arm (`None`, the
        // split's check red).
        let plan = or_named(match arm.total {
            Total::Plan => records.one(&record::PLAN38).map(Some),
            Total::Set => records.first(&record::PLAN38),
        });
        let plan_line = plan.as_ref().and_then(Option::as_ref).map(Fields::line);
        // The load's decision, printed: the whole-fit verdicts the seat ran
        // (one, at the total, when it made one) and its plan record.
        let verdicts: Vec<&str> = log
            .lines()
            .filter(|l| l.starts_with("whole fit at ctx "))
            .collect();
        println!(
            "{}: {} whole-fit verdict(s) {verdicts:?}; plan record {plan_line:?}",
            arm.dir,
            verdicts.len()
        );
        let total = match arm.total {
            Total::Set => arm
                .flags
                .iter()
                .position(|&f| f == "--ctx")
                .and_then(|i| arm.flags.get(i + 1))
                .and_then(|t| t.parse::<u64>().ok()),
            Total::Plan => u64_of(plan.as_ref().and_then(Option::as_ref), "ctx_max"),
        };
        let (st, body) = curl(&url("/props"), None, false)?;
        let props = json_of("/props", st, &body)?;
        let n_ctx = props["n_ctx"].as_u64().unwrap_or(u64::MAX);
        println!(
            "{}: {:?}; the total {total:?} ({:?}), props n_ctx {n_ctx}, record slots={:?} \
             slot_ctx={:?}",
            arm.dir,
            load.as_ref().map(Fields::line),
            arm.total,
            field("slots"),
            field("slot_ctx"),
        );
        check(
            ok,
            &named("slots_load_names_the_split"),
            plan.is_some()
                && field("slots") == Some(2)
                && total.is_some_and(|t| field("slot_ctx") == Some(t / 2)),
        );
        check(
            ok,
            &named("props_names_the_slot_ctx"),
            total.is_some_and(|t| n_ctx == t / 2),
        );
        // `/props`' placement is the plan's when the load placed one (its plan record printed): the
        // card, then the host, and routed experts on the host exactly when the record puts them
        // there. A whole-card load has no plan: no placement, and the seat's line says why.
        let placement = &props["engine"]["placement"];
        match plan.as_ref().and_then(Option::as_ref) {
            Some(rec) => {
                let devices = placement["devices"].as_array().cloned().unwrap_or_default();
                let host_experts = devices
                    .get(1)
                    .and_then(|d| d["classes"]["experts"].as_u64())
                    .unwrap_or(0);
                println!("{}: props placement {placement}", arm.dir);
                check(
                    ok,
                    &named("props_engine_placement_is_the_plans"),
                    devices.len() == 2
                        && devices[0]["device"]
                            .as_str()
                            .and_then(|d| d.strip_prefix("GPU"))
                            .is_some_and(|n| n.parse::<u32>().is_ok())
                        && devices[1]["device"] == "CPU"
                        && devices[0]["bytes"].as_u64().is_some_and(|b| b > 0)
                        && u64_of(Some(rec), "host_experts")
                            .is_some_and(|h| (h > 0) == (host_experts > 0)),
                );
            }
            None => check(
                ok,
                &named("props_engine_whole_load_names_no_placement"),
                placement.is_null()
                    && log.contains(
                        "/props leaves the placement out: a whole-card load opens with no plan",
                    ),
            ),
        }
        // The whole arm's set total is its whole-fit verdict's, held by the
        // two slots' planes; the placed arm's is the plan's.
        if let (Total::Set, Some(t)) = (arm.total, total) {
            slots_set_total_is_its_whole_fit(model, &verdicts, t, ok)?;
        }
        // Two distinct prompts, each first run alone, then both together.
        let a_ids = rendered(&url, messages())?;
        let b_ids = tokenized(&url, PROSE)?;
        let mut alone = Vec::new();
        for ids in [&a_ids, &b_ids] {
            let (h, rx) = streamed(&addr, ids, SLOTS_PREDICT)?;
            h.join()
                .map_err(|_| format!("{}: an alone request's thread panicked", arm.dir))?;
            let ran = rx
                .recv()
                .map_err(|_| format!("{}: an alone request's thread gave no answer", arm.dir))??;
            println!("{} alone: {} ids", arm.dir, ran.tokens.len());
            alone.push(ran.tokens);
        }
        let decode0 = metric(&url, "n_decode_total")?.unwrap_or(f64::NAN);
        let busy0 = busy_total(&url, decode0)?;
        // Both posted at once: the prompts serialize on the engine thread
        // (a round or two of skew), so nearly every decode round carries
        // both slots — the (v2) bound's shape.
        let first = streamed(&addr, &a_ids, SLOTS_PREDICT)?;
        let second = streamed(&addr, &b_ids, SLOTS_PREDICT)?;
        let mut together = Vec::new();
        for ((h, rx), what) in [(first, "first"), (second, "second")] {
            h.join()
                .map_err(|_| format!("{}: the {what} request's thread panicked", arm.dir))?;
            let ran = rx.recv().map_err(|_| {
                format!("{}: the {what} request's thread gave no answer", arm.dir)
            })??;
            together.push(ran);
        }
        let decode1 = metric(&url, "n_decode_total")?.unwrap_or(f64::NAN);
        let busy1 = busy_total(&url, decode1)?;
        let swaps = metric(&url, "swaps_total")?;
        // (v1) The slots hold separate sequences: together ids are alone ids.
        check(
            ok,
            &named("slots_alone_ids_stay"),
            !together[0].tokens.is_empty()
                && !together[1].tokens.is_empty()
                && together[0].tokens == alone[0]
                && together[1].tokens == alone[1],
        );
        // (v2) No park, and a round carries both slots (the derivation in
        // the module header): ≥ 1.5 against the turns' 1.0.
        let ratio = (busy1 - busy0) / (decode1 - decode0);
        println!(
            "{} together: {} and {} ids; decode {decode0} -> {decode1}, busy {busy0} -> \
             {busy1} (ratio {ratio:.3}); swaps {swaps:?}",
            arm.dir,
            together[0].tokens.len(),
            together[1].tokens.len(),
        );
        check(
            ok,
            &named("slots_rounds_carry_both_slots"),
            swaps.is_none_or(|v| v == 0.0) && ratio >= 1.5,
        );
        println!("{} server stopped: {}", arm.dir, s.stop()?);
        Ok(())
    }

    /// `slots_set_total_is_its_whole_fit` (the module header), on the whole
    /// slots arm's server of a set total `t`: its one whole-fit line
    /// (`verdicts`) is at `t` and says the load fits, and the line's terms —
    /// its need, its cache, what the card had ([`whole_terms`]) — leave a
    /// rest no context moves, so the two-slot need at `t` is that rest plus
    /// the allocator's bytes of the two slots' planes: each slot the
    /// one-slot list [`cache_planes`] gives at ⌊t/2⌋ rows, once per slot
    /// concatenated — the per-slot list the seat's own verdict counts
    /// (`q3place::kv_bytes` over `PlaceQ3::whole_at`: N copies of one slot's
    /// planes, each slot `ctx / N` rows, through the card's granule) — and
    /// it is inside what the card had. A line that says the load does not
    /// fit is a named error: the arm is not a whole arm. FAIL-first: a seat
    /// that decides a set total's load at a slot's share puts its line at
    /// half the total, red on every card.
    // PIN(2026-10-14): the clause was `slots_default_is_the_whole_search`,
    // the whole arm an unset `--ctx` whose two-slot fit search it held: (b)
    // a slot never past the one-slot default, (c) the largest slot context
    // the card holds. An unset `--ctx` now asks the target and halves only
    // on a short card (`placement::ctx_ask`), which a card may hold only
    // placed, so the whole arm sets its total and (b) and (c) left with the
    // search; the fit's relations are `ctx_default_is_the_largest_step_held`'s
    // and `default_ctx_line_names_its_rule`'s.
    fn slots_set_total_is_its_whole_fit(
        model: &Path,
        verdicts: &[&str],
        t: u64,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let split = Split::open(model).map_err(|e| format!("open {}: {e}", model.display()))?;
        let line = match verdicts {
            [line] => *line,
            _ => {
                return Err(format!(
                    "the whole slots arm's server printed {} whole-fit line(s), not the one its \
                     total comes from",
                    verdicts.len()
                )
                .into());
            }
        };
        let at = line
            .strip_prefix("whole fit at ctx ")
            .and_then(|x| x.split(':').next())
            .and_then(|x| x.parse::<u64>().ok())
            .ok_or_else(|| format!("the whole-fit line names no context: {line}"))?;
        let WholeTerms {
            rest, budget, fits, ..
        } = whole_terms(line)?;
        if !fits {
            return Err(format!(
                "the whole slots arm's whole-fit line at {t} says the load does not fit: the arm \
                 is not a whole arm"
            )
            .into());
        }
        // The two-slot whole load's need at a total `x`: the line's other
        // terms plus two slots' planes, each slot one slot's list at half
        // the total, as the allocator rounds them.
        let granule = serve_granule()?;
        let need2 = |x: u64| -> Result<u64, GateError> {
            let rows = x / 2;
            let mut planes = cache_planes(model, &split, rows, 1)?;
            planes.extend(cache_planes(model, &split, rows, 1)?);
            rest.checked_add(model::placement::allocator_bytes(granule, planes))
                .ok_or_else(|| {
                    format!("the two-slot whole load's need at {x} passes u64 bytes").into()
                })
        };
        let at_t = need2(t)?;
        println!(
            "slots arm: the set total {t}, its whole-fit line at {at}; the two-slot load needs \
             {at_t} B at {t}, of {budget} B the card had",
        );
        check(
            ok,
            "slots_set_total_is_its_whole_fit",
            at == t && at_t <= budget,
        );
        Ok(())
    }

    /// One server of the slots-round clause ([`slots_rounds_run_one_pass`]): its
    /// directory under the file's, the prefix of its checks' names, the flags
    /// past `--parallel 2`, its environment, whether a round is one pass, and
    /// whether its second request samples, seeded, beside a greedy first.
    struct RoundsArm<'a> {
        dir: &'a str,
        prefix: &'a str,
        flags: &'a [&'a str],
        env: &'a [(&'a str, &'a str)],
        one_pass: bool,
        sampled: bool,
    }

    /// The whole-card rounds arm: no flag past `--parallel 2`.
    const WHOLE_ROUNDS: RoundsArm<'static> = RoundsArm {
        dir: "slots-rounds",
        prefix: "",
        flags: &[],
        env: &[],
        one_pass: true,
        sampled: false,
    };

    /// The placed rounds arm (its checks `placed_…`): `--place a` under a
    /// card budget of [`PLACED_BUDGET`].
    const PLACED_ROUNDS: RoundsArm<'static> = RoundsArm {
        dir: "slots-rounds-placed",
        prefix: "placed_",
        flags: &["--place", "a"],
        env: &[("BLOOMERY_CARD_BUDGET", PLACED_BUDGET)],
        one_pass: true,
        sampled: false,
    };

    /// The sampled rounds arm (its checks `mixed_…`): a greedy and a seeded
    /// sampled request at the seat's own `--ctx`.
    const SAMPLED_ROUNDS: RoundsArm<'static> = RoundsArm {
        dir: "slots-rounds-mixed",
        prefix: "mixed_",
        flags: &["--ctx", "4096"],
        env: &[],
        one_pass: true,
        sampled: true,
    };

    /// The slots-round clause (module header) on a server of `--parallel 2`
    /// started with `BLOOMERY_STEP_STATS=1` and the arm's flags and
    /// environment into its own directory: two greedy streamed requests, or
    /// on a `sampled` arm one greedy and one seeded sampled request, each first
    /// run alone and then together. The sampled arm's pair keeps its alone ids;
    /// its greedy request asks for no logits. The `slots round` records
    /// between the requests hold their pass shape.
    /// Every arm is one pass: every together-run record holds `passes=1`
    /// (`slots_rounds_run_one_pass`, `placed_…` and `mixed_…` on those
    /// arms). On every arm the window holds a round
    /// of both ([`rounds_hold_a_round_of_both`]: at least one record of
    /// `rows=2` — the records carry no round of one stepping slot, so the
    /// records cannot see one-slot stretches, and the window's solo count
    /// prints beside, unpinned);
    /// and the rows the records carry, with
    /// the solo rounds the `/metrics` decode deltas count, sum to the
    /// tokens the two requests decoded beyond their prompts' first
    /// (`slots_rounds_rows_sum_the_decoded_tokens`). A printed plan names
    /// the split and does not choose turns.
    /// `BLOOMERY_STEP_STATS=1` moves no computation on this server — the seat
    /// and the fallback loop branch on it only to print the record — so the
    /// together-equals-alone bits of these runs stand for the default server.
    fn slots_rounds_run_one_pass(
        model: &Path,
        dir: &Path,
        arm: &RoundsArm<'_>,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let &RoundsArm {
            dir: dir_name,
            prefix,
            flags,
            env,
            one_pass,
            sampled,
        } = arm;
        let named = |c: &str| format!("{prefix}{c}");
        let d = dir.join(dir_name);
        std::fs::create_dir_all(&d)?;
        let err_log = d.join("server.err");
        let mut cmd = stats_cmd()?;
        for &(k, v) in env {
            cmd.env(k, v);
        }
        let m = model.to_str().ok_or("the model path is not UTF-8")?;
        let mut args = vec![
            "--model",
            "qwen3",
            "--port",
            "0",
            "--parallel",
            "2",
            "-m",
            m,
        ];
        args.extend_from_slice(flags);
        let mut s = Served::spawn_cmd(cmd, &args, &d)?;
        let addr = s.address(&err_log, 600, Duration::from_secs(1))?;
        let url = |p: &str| format!("http://{addr}{p}");
        // The shape the seat's open decided, by the line it printed before its load: one pass
        // of the busy rows (`qwen3::pass_of`, `Body3::serves_one_pass`), whole-card and
        // placed. The rounds below hold the records to that shape. A printed plan is still
        // read: it names the split and does not choose turns.
        let log = std::fs::read_to_string(&err_log)?;
        // PIN(2026-10-22): a placed load of `Body` or `Body35` serves
        // several slots' rows as one pass. A printed plan is that load
        // and names `one`, the same as a whole-card load. Lookup stays
        // turns in `pass_of` and is not an arm here.
        // FAIL-first: the word `turns`, or a together record of `passes=rows`.
        let plan = routed_split(&seat_records(&log, &err_log))?;
        println!(
            "{dir_name}: plan (host, card, ctx) {plan:?}; a round is {}",
            if one_pass { "one pass" } else { "turns" }
        );
        let parallel = log.lines().find(|l| l.starts_with("parallel rule="));
        println!("{dir_name}: {parallel:?}");
        check(
            ok,
            &named(if one_pass {
                "slots_parallel_line_names_one_pass"
            } else {
                "slots_parallel_line_names_turns"
            }),
            parallel.is_some_and(|l| {
                parallel_word(l, "slots") == Some("2")
                    && parallel_word(l, "pass") == Some(if one_pass { "one" } else { "turns" })
            }),
        );
        let a_ids = rendered(&url, messages())?;
        let b_ids = tokenized(&url, PROSE)?;
        let request = |ids: &[u32], second: bool| {
            if sampled {
                streamed_body(
                    &addr,
                    json!({
                        "prompt": ids, "n_predict": SLOTS_PREDICT,
                        "temperature": if second { 0.8 } else { 0.0 }, "seed": 42,
                        "return_tokens": true, "cache_prompt": false, "stream": true,
                    }),
                )
            } else {
                streamed(&addr, ids, SLOTS_PREDICT)
            }
        };
        let mut alone = Vec::new();
        for (j, ids) in [&a_ids, &b_ids].into_iter().enumerate() {
            let (h, rx) = request(ids, j == 1)?;
            h.join()
                .map_err(|_| format!("{dir_name}: an alone request's thread panicked"))?;
            let ran = rx
                .recv()
                .map_err(|_| format!("{dir_name}: an alone request's thread gave no answer"))??;
            alone.push(ran.tokens);
        }
        let decode0 = metric(&url, "n_decode_total")?.unwrap_or(f64::NAN);
        // The alone runs print no record (a round of one row is a select
        // and a `next`), so the window opens empty; taken anyway, so the
        // together run's records are exactly the ones after this point.
        let before = seat_log(&err_log)?.all(&record::SLOTS_ROUND)?.len();
        let first = request(&a_ids, false)?;
        let second = request(&b_ids, true)?;
        let mut together = Vec::new();
        for ((h, rx), what) in [(first, "first"), (second, "second")] {
            h.join()
                .map_err(|_| format!("{dir_name}: the {what} request's thread panicked"))?;
            together.push(rx.recv().map_err(|_| {
                format!("{dir_name}: the {what} request's thread gave no answer")
            })??);
        }
        if sampled {
            let ids: Vec<&[u32]> = together.iter().map(|s| s.tokens.as_slice()).collect();
            println!("{dir_name}: alone {alone:?}, together {ids:?}");
            check(
                ok,
                &named("slots_together_alone"),
                alone
                    .iter()
                    .zip(&together)
                    .all(|(a, t)| !a.is_empty() && *a == t.tokens),
            );
        }
        let decode1 = metric(&url, "n_decode_total")?.unwrap_or(f64::NAN);
        let rounds = seat_log(&err_log)?.all(&record::SLOTS_ROUND)?;
        let window = &rounds[before.min(rounds.len())..];
        let rows: Vec<u64> = window.iter().map(|r| r.u64("rows").unwrap_or(0)).collect();
        let every = !window.is_empty()
            && window.iter().all(|r| {
                r.word("cmd") == Ok("step")
                    && r.u64("slots") == Ok(2)
                    && matches!(r.u64("rows"), Ok(n) if (2..=MAX_TOKENS as u64).contains(&n))
                    && r.u64("passes") == if one_pass { Ok(1) } else { r.u64("rows") }
            });
        println!(
            "{dir_name}: {} record(s) in the together window, rows {rows:?} (alone {} and \
                 {} ids, together {} and {})",
            window.len(),
            alone[0].len(),
            alone[1].len(),
            together[0].tokens.len(),
            together[1].tokens.len(),
        );
        check(
            ok,
            &named(if one_pass {
                "slots_rounds_run_one_pass"
            } else {
                "slots_rounds_run_in_turn"
            }),
            every,
        );
        // The window's solo rounds (a round of one stepping slot is a
        // select and a `next`, no record): the decode deltas less the
        // records and the two requests' prompt-phase calls, booked at
        // their end (`worker.rs`'s `end_request`).
        let solo = decode1 - decode0 - window.len() as f64 - 2.0;
        rounds_hold_a_round_of_both(
            ok,
            &named("slots_rounds_hold_a_round_of_both"),
            dir_name,
            window,
            solo,
        );
        if prefix.is_empty() {
            // The tokens decoded beyond the two prompts' first: each
            // request's `n_predict` less one, its first token being the
            // prompt's own last step. The no-flag arm, whole or placed.
            let sum: u64 = rows.iter().sum();
            let want = 2 * (SLOTS_PREDICT - 1) as u64;
            println!(
                "{dir_name}: rows sum {sum} + {solo} solo round(s) against the {want} \
                     decoded beyond the prompts' first (decode {decode0} -> {decode1})",
            );
            check(
                ok,
                "slots_rounds_rows_sum_the_decoded_tokens",
                solo >= 0.0 && sum + solo as u64 == want,
            );
        }
        println!("{dir_name} server stopped: {}", s.stop()?);
        Ok(())
    }

    /// The slots clause's both-advance-in-the-same-rounds criterion (the
    /// module header) on one slots-round arm's together window `window`,
    /// its solo-round count `solo` printed beside: the window holds at
    /// least one round that carried both slots (a record of `rows=2`) —
    /// the one thing the records can see of it. A round of one stepping
    /// slot prints no record at all (the worker runs it as a plain select
    /// and `next`), so the records cannot see one-slot stretches: a seat
    /// that serves one slot at a time leaves the window with no record of
    /// both, while a server that only stretches one-slot runs between
    /// two-row rounds keeps its records of both — that shape is the slots
    /// clause's (v2) busy ratio's to catch (`slots_rounds_carry_both_slots`,
    /// ≥ 1.5, pulled to ~1.0 by one busy a round). The solo count is
    /// printed, not pinned: on a round server it is the two requests'
    /// admission skew and its mirror at the tail (equal `n_predict`), a
    /// race the worker counts in passes, which no code bounds. The hold
    /// is not a pass-count pin: a turns mutant still prints a record of
    /// `rows=2`. The pass count is `slots_rounds_run_one_pass`.
    // PIN(2026-10-06): the criterion reads the engine's own `slots round`
    // records, not the SSE arrival timestamps it read before — a client-side
    // clock on a shared box clumps one stream's events under load (the same
    // clause in `gate_qwen38_serve` went red at load average 15.6 with the
    // server's rounds unchanged), so the arrival window was a flake by
    // construction. The threshold stays zero: no thin arrival window then,
    // no empty together window now — a window with no record of `rows=2` is
    // a seat that served one slot at a time.
    fn rounds_hold_a_round_of_both(
        ok: &mut bool,
        name: &str,
        what: &str,
        window: &[Fields],
        solo: f64,
    ) {
        let both = window.iter().filter(|r| r.u64("rows") == Ok(2)).count();
        println!(
            "{what}: {both} round(s) of both slots; {solo} solo round(s) beside them, printed \
             not pinned",
        );
        check(ok, name, both > 0);
    }

    /// The late-request clause (module header): the first request decodes
    /// alone on a whole-card server of `--parallel 2`, and the second is
    /// posted once `/metrics`' `n_decode_total` has moved [`LATE_ROUNDS`]
    /// rounds — slot 1 taken mid-run, its prompt run between the first's
    /// rounds. Both answer with their alone ids.
    fn slots_late_request_together_alone(
        model: &Path,
        dir: &Path,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let d = dir.join("slots-late");
        std::fs::create_dir_all(&d)?;
        let err_log = d.join("server.err");
        let mut cmd = Command::new(beside("bloomery-serve")?);
        cmd.env_remove("BLOOMERY_REF_MODEL");
        let m = model.to_str().ok_or("the model path is not UTF-8")?;
        let mut s = Served::spawn_cmd(
            cmd,
            &[
                "--model",
                "qwen3",
                "--port",
                "0",
                "--parallel",
                "2",
                "-m",
                m,
            ],
            &d,
        )?;
        let addr = s.address(&err_log, 600, Duration::from_secs(1))?;
        let url = |p: &str| format!("http://{addr}{p}");
        let a_ids = rendered(&url, messages())?;
        let b_ids = tokenized(&url, PROSE)?;
        let mut alone = Vec::new();
        for ids in [&a_ids, &b_ids] {
            let (h, rx) = streamed(&addr, ids, SLOTS_PREDICT)?;
            h.join()
                .map_err(|_| "slots-late: an alone request's thread panicked")?;
            let ran = rx
                .recv()
                .map_err(|_| "slots-late: an alone request's thread gave no answer")??;
            alone.push(ran.tokens);
        }
        // The late run: the first decodes alone, and once a few of its rounds
        // have passed the second arrives on the empty slot.
        let first = streamed(&addr, &a_ids, SLOTS_PREDICT)?;
        let decode0 = metric(&url, "n_decode_total")?.unwrap_or(f64::NAN);
        let mut polls = 0;
        while polls < LATE_POLLS
            && metric(&url, "n_decode_total")?.unwrap_or(decode0) < decode0 + LATE_ROUNDS as f64
        {
            std::thread::sleep(LATE_POLL);
            polls += 1;
        }
        let second = streamed(&addr, &b_ids, SLOTS_PREDICT)?;
        let mut late = Vec::new();
        for ((h, rx), what) in [(first, "first"), (second, "second")] {
            h.join()
                .map_err(|_| format!("slots-late: the {what} request's thread panicked"))?;
            late.push(rx.recv().map_err(|_| {
                format!("slots-late: the {what} request's thread gave no answer")
            })??);
        }
        println!(
            "slots-late: the second posted after {polls} poll(s); {} and {} ids",
            late[0].tokens.len(),
            late[1].tokens.len(),
        );
        check(
            ok,
            "slots_late_request_together_alone",
            !late[0].tokens.is_empty()
                && !late[1].tokens.is_empty()
                && late[0].tokens == alone[0]
                && late[1].tokens == alone[1],
        );
        println!("slots-late server stopped: {}", s.stop()?);
        Ok(())
    }

    /// The placed slots arm's own terms, from its server's stderr
    /// (`<dir>/slots-placed/server.err`) once it stopped: its `plan` record
    /// keeps routed experts on the card and on the host tier (the arm ran
    /// the placed body: the host tier, the placed step walk, the eager
    /// prompt passes — not the whole load a plan of every expert opens),
    /// and the seat's `placed slots` line holds the two sequences' cache
    /// bytes at or under what the plan counts for them at the total — its
    /// KV term and its card's `resident slots` reserve — the slot ctx half
    /// the plan's `ctx_max` (`placed_slots_cache_within_the_plan`); the
    /// line's terms are the file's own planes ([`cache_planes`]): the two
    /// sequences hold twice one slot's planes at the slot ctx, and the
    /// reserve, named only where it is not 0, is one sequence's planes at
    /// no position — what the second slot holds again past the KV term's
    /// count (`placed_slots_reserve_the_fixed_stores`). FAIL-first: the
    /// seat's placed load kept on one sequence prints no `placed slots`
    /// line; a reserve counted for both slots names twice the fixed planes.
    fn placed_slots_hold_the_plan(
        model: &Path,
        dir: &Path,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let err_log = dir.join(PLACED_SLOTS.dir).join("server.err");
        let log =
            std::fs::read_to_string(&err_log).map_err(|e| format!("{}: {e}", err_log.display()))?;
        let records = seat_records(&log, &err_log);
        let plan = or_named(records.one(&record::PLAN38));
        let (host, card, total) = (
            u64_of(plan.as_ref(), "host_experts"),
            u64_of(plan.as_ref(), "card_experts"),
            u64_of(plan.as_ref(), "ctx_max"),
        );
        let load = or_named(records.one(&record::LOAD_QWEN3));
        // `placed slots: N sequences of S rows hold H B of cache; the plan
        // at T positions counts K B`, then ` (its KV term V B and its
        // resident slots reserve R B)` where R is not 0
        // (`q3place::open_slots`).
        let line = log
            .lines()
            .find(|l| l.starts_with("placed slots: "))
            .unwrap_or("");
        let num = |after: &str| -> Option<u64> {
            line.split(after).nth(1)?.split(' ').next()?.parse().ok()
        };
        let (n, slot, held, at, counted) = (
            num("placed slots: "),
            num(" sequences of "),
            num(" hold "),
            num(" the plan at "),
            num(" counts "),
        );
        println!(
            "placed slots: plan host_experts {host:?} card_experts {card:?} ctx_max {total:?}; \
             load resident_bytes {:?}; cache line {line:?}",
            u64_of(load.as_ref(), "resident_bytes")
        );
        check(
            ok,
            "placed_slots_plan_splits_the_experts",
            host.is_some_and(|h| h > 0) && card.is_some_and(|c| c > 0),
        );
        check(
            ok,
            "placed_slots_cache_within_the_plan",
            total.is_some()
                && n == Some(2)
                && at == total
                && slot == total.map(|t| t / 2)
                && held.zip(counted).is_some_and(|(h, k)| h > 0 && h <= k),
        );
        // The line's terms against the file's own planes: two slots' planes
        // at the slot ctx held, one slot's planes at no position reserved
        // (named only where not 0), the count its KV term plus that reserve.
        let split = Split::open(model).map_err(|e| format!("open {}: {e}", model.display()))?;
        let fixed: u64 = cache_planes(model, &split, 0, 1)?.iter().sum();
        let both = match slot {
            Some(s) => Some(2 * cache_planes(model, &split, s, 1)?.iter().sum::<u64>()),
            None => None,
        };
        let (kv, reserve) = (num("(its KV term "), num(" reserve "));
        println!(
            "placed slots: held {held:?} of two slots' planes {both:?}; reserve {reserve:?} of \
             one slot's fixed planes {fixed} B; KV term {kv:?}, counted {counted:?}"
        );
        let terms = if fixed == 0 {
            reserve.is_none() && kv.is_none()
        } else {
            reserve == Some(fixed) && kv.zip(counted).is_some_and(|(v, k)| v + fixed == k)
        };
        check(
            ok,
            "placed_slots_reserve_the_fixed_stores",
            held.is_some() && held == both && terms,
        );
        Ok(())
    }

    /// `slots_count_their_checkpoints` (module header), from both slots
    /// arms' servers' stderr once they stopped: a qwen35moe file's two
    /// sequences each pin up to the body's budget at the slot's context
    /// (`checkpoint_budget35` of the file's own spec at the asked ubatch, the
    /// slot context read off the arm's `parallel` line), so the seat's one
    /// `checkpoints:` line names two sequences and twice that budget — on the
    /// whole-card arm against the host's available bytes, on the placed arm
    /// as its plan's host reserve; a qwen3moe file's sequences take no
    /// checkpoints and print none. FAIL-first: the term counted for one
    /// sequence names one budget; a budget priced at the host's cap
    /// (`runtime::seqstate::HOST_BUDGET`) names the 4 GiB of the default.
    fn slots_count_their_checkpoints(
        model: &Path,
        seat: qwen_seat::Family,
        dir: &Path,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let split = Split::open(model).map_err(|e| format!("open {}: {e}", model.display()))?;
        let takes = body35(seat);
        let spec = takes
            .then(|| model::arch::qwen35moe::spec::read(&split))
            .transpose()?
            .map(|read| read.spec);
        let mut held = true;
        // A dense file has no placed arm (its budgeted plan is a refusal).
        let placed = routed(seat).then_some((
            PLACED_SLOTS.dir,
            "counted in the placed plan's host reserves",
        ));
        for (arm, counted_in) in [("slots", "of ")].into_iter().chain(placed) {
            let log = std::fs::read_to_string(dir.join(arm).join("server.err")).unwrap_or_default();
            let lines: Vec<&str> = log
                .lines()
                .filter(|l| l.starts_with("checkpoints: "))
                .collect();
            println!("{arm}: checkpoints lines {lines:?}");
            held &= match &spec {
                Some(spec) => {
                    // The size the load opened at: a file with no routed expert settles it in its
                    // verdict (the seat's last `ubatch` line), any other takes the levers'.
                    let ubatch = match ubatch_pick::choices(&log).last().and_then(|c| c.pick) {
                        Some(u) => u,
                        None => ubatch_size()?,
                    };
                    let slot_ctx = log
                        .lines()
                        .filter(|l| l.starts_with("parallel "))
                        .find_map(|l| parallel_word(l, "slot_ctx")?.parse::<u64>().ok());
                    let each = slot_ctx
                        .map(|c| checkpoint_budget35(spec, ubatch, c))
                        .transpose()?;
                    let want = each.zip(slot_ctx).map(|(each, c)| {
                        format!(
                            "checkpoints: 2 sequences pin at most {} B of host ({each} B each at \
                             {c} positions), {counted_in}",
                            2 * each
                        )
                    });
                    match (want, &lines[..]) {
                        (Some(want), [l]) => l.starts_with(&want),
                        _ => false,
                    }
                }
                None => lines.is_empty(),
            };
        }
        check(ok, "slots_count_their_checkpoints", held);
        Ok(())
    }

    /// The card budget the host-tier clause plans under: a card this size
    /// holds the trunk and only part of the routed experts of the qwen3moe
    /// file, so the placed plan puts the rest on the host on either box card.
    const HOST_TIER_BUDGET: &str = "12G";

    /// The host-tier prompt clause (module header): a placed server under
    /// [`HOST_TIER_BUDGET`] — a large card planning as the smallest one the
    /// placed path serves — whose plan record names routed experts on the
    /// host, answers a prompt of at least [`GEMM_FROM`] ids. A plan with no
    /// host experts opens the whole-card body, which never meets the placed
    /// prompt path, so the clause asks for both.
    fn placed_answers_a_prompt_past_the_gemm_walk(
        model: &Path,
        dir: &Path,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let d = dir.join("host-tier");
        std::fs::create_dir_all(&d)?;
        let err_log = d.join("server.err");
        let mut cmd = Command::new(beside("bloomery-serve")?);
        cmd.env_remove("BLOOMERY_REF_MODEL");
        cmd.env(bloomery_levers::CARD_BUDGET, HOST_TIER_BUDGET);
        let m = model.to_str().ok_or("the model path is not UTF-8")?;
        let mut s = Served::spawn_cmd(
            cmd,
            &[
                "--model",
                "qwen3",
                "--port",
                "0",
                "--parallel",
                "1",
                "--place",
                "a",
                "--ctx",
                "2048",
                "-m",
                m,
            ],
            &d,
        )?;
        let addr = s.address(&err_log, 600, Duration::from_secs(1))?;
        let url = |p: &str| format!("http://{addr}{p}");
        let log = std::fs::read_to_string(&err_log)?;
        let host = u64_of(
            or_named(seat_records(&log, &err_log).one(&record::PLAN38)).as_ref(),
            "host_experts",
        );
        let ids = tokenized(&url, PROSE)?;
        let body = json!({
            "prompt": ids, "n_predict": N, "temperature": 0,
            "return_tokens": true, "cache_prompt": false,
        });
        let (st, text) = match curl(&url("/completion"), Some(&body), false) {
            Ok(r) => r,
            Err(e) => (0, e.to_string()),
        };
        let tokens = match st {
            200 => ids_of(&json_of("/completion", st, &text)?["tokens"])?,
            _ => Vec::new(),
        };
        println!(
            "host-tier arm: plan host_experts={host:?}; a prompt of {} ids answers HTTP {st} \
             with {} tokens",
            ids.len(),
            tokens.len()
        );
        check(
            ok,
            "placed_answers_a_prompt_past_the_gemm_walk",
            host.is_some_and(|h| h > 0)
                && ids.len() >= GEMM_FROM
                && st == 200
                && !tokens.is_empty(),
        );
        println!("host-tier server stopped: {}", s.stop()?);
        Ok(())
    }

    /// The refusal clause (module header), two arms.
    /// PIN(2026-10-07): the refusal arm passes `--parallel 2` — a set `--ctx`
    /// with no `--parallel` is one request's context now (`bind::slots_of`),
    /// so the two-slot split the floor refuses needs the count said — and
    /// the one-slot arm holds that unset-flag rule on the same `--ctx`: the
    /// server takes one slot at the whole flag and says so (`from=ctx`, the
    /// one-request note) before its load, stopped there. Re-pin by
    /// specification, not a relaxation: the refusal keeps its named message
    /// at the same boundary, and the one-slot rule is a new check beside it.
    /// A refusal that loads instead shows as the
    /// server still running (or listening) past the polls and is stopped
    /// here, red; one that dies inside the load names another cause, red.
    fn slot_ctx_too_small_is_refused(
        model: &Path,
        dir: &Path,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let d = dir.join("refuse");
        std::fs::create_dir_all(&d)?;
        let err_log = d.join("server.err");
        let mut cmd = Command::new(beside("bloomery-serve")?);
        cmd.env_remove("BLOOMERY_REF_MODEL");
        let m = model.to_str().ok_or("the model path is not UTF-8")?;
        let total = SLOTS_REFUSED_CTX.to_string();
        let mut s = Served::spawn_cmd(
            cmd,
            &[
                "--model",
                "qwen3",
                "--port",
                "0",
                "--ctx",
                &total,
                "--parallel",
                "2",
                "-m",
                m,
            ],
            &d,
        )?;
        let mut said = String::new();
        let mut refused = false;
        for _ in 0..360 {
            said = std::fs::read_to_string(&err_log).unwrap_or_default();
            match s.child.try_wait()? {
                Some(status) => {
                    refused = !status.success()
                        && said.contains("--parallel 2")
                        && said.contains(&format!("of a --ctx of {total}:"))
                        && said.contains(&format!(
                            "a slot's context of {} rows is below the {MAX_TOKENS} rows",
                            SLOTS_REFUSED_CTX / 2
                        ));
                    break;
                }
                None if said.contains("listening on http://") => break,
                None => std::thread::sleep(SLOTS_POLL),
            }
        }
        println!(
            "slots refusal: exit {:?}, said {said}",
            s.child.try_wait()?.map(|s| s.code())
        );
        check(ok, "slot_ctx_too_small_is_refused", refused);
        let _ = s.stop();

        // The one-slot arm: the same `--ctx` with no `--parallel` is one
        // request's context — one slot at the whole flag, the `parallel`
        // line and the one-request note on stderr before the load, the
        // server stopped there (a set `--parallel` still splitting it would
        // print `slots=2 slot_ctx=7` and refuse to load).
        let d = dir.join("one-slot");
        std::fs::create_dir_all(&d)?;
        let err_log = d.join("server.err");
        let mut cmd = Command::new(beside("bloomery-serve")?);
        cmd.env_remove("BLOOMERY_REF_MODEL");
        let mut s = Served::spawn_cmd(
            cmd,
            &["--model", "qwen3", "--port", "0", "--ctx", &total, "-m", m],
            &d,
        )?;
        let mut said = String::new();
        let mut line = None;
        for _ in 0..360 {
            said = std::fs::read_to_string(&err_log).unwrap_or_default();
            line = said
                .lines()
                .find(|l| l.starts_with("parallel rule="))
                .map(str::to_owned);
            if line.is_some() || s.child.try_wait()?.is_some() {
                break;
            }
            if said.contains("listening on http://") {
                break;
            }
            std::thread::sleep(SLOTS_POLL);
        }
        if s.child.try_wait()?.is_none() {
            println!("one-slot arm: server stopped: {}", s.stop()?);
        }
        let one_request = format!(
            "--ctx-size {total} is one request's context; add --parallel N to serve N requests at \
             once (they split it)"
        );
        let agrees = line.as_deref().is_some_and(|l| {
            let f = |k: &str| parallel_word(l, k);
            f("slots") == Some("1")
                && f("slot_ctx") == Some(total.as_str())
                && f("total") == Some(total.as_str())
                && f("from") == Some("ctx")
        });
        println!("one-slot arm: {line:?}; said {said}");
        check(
            ok,
            "ctx_set_parallel_unset_is_one_slot",
            agrees && said.contains(&one_request),
        );
        // One slot runs no round of several: `Pass::of(1, _)` is `Pass::One`
        // for every body and placement (`bind::Pass::of`, `qwen3::pass_of`).
        check(
            ok,
            "one_slot_parallel_line_names_one_pass",
            line.as_deref()
                .is_some_and(|l| parallel_word(l, "pass") == Some("one")),
        );
        let _ = s.stop();
        Ok(())
    }

    /// The server's clauses for `model`, its logs in `dir`; the answers the
    /// CLI is held to, the chat turn's first, and the file's architecture
    /// (which arm carries which clause below). `None` when the server never
    /// listened — `default_seat_listens` red, the server's own words
    /// printed — and the file's other clauses do not run.
    fn served(
        model: &Path,
        seat: qwen_seat::Family,
        dir: &Path,
        ok: &mut bool,
    ) -> Result<Option<(Vec<Answer>, u64)>, GateError> {
        let err_log = dir.join("server.err");
        let mut s = spawn(model, dir)?;
        // The load reads the whole file: up to ten minutes from a cold cache.
        let addr = match s.address(&err_log, 600, Duration::from_secs(1)) {
            Ok(addr) => addr,
            Err(e) => {
                println!("the default seat never listened: {e}");
                check(ok, "default_seat_listens", false);
                // A server still up past the polls is killed when `s` drops.
                return Ok(None);
            }
        };
        check(ok, "default_seat_listens", true);
        let url = |p: &str| format!("http://{addr}{p}");
        let log = std::fs::read_to_string(&err_log)?;
        let records = seat_log(&err_log)?;
        let load = records.first(&record::LOAD_QWEN3)?;
        let listen = records.first(&record::LISTENING_QWEN3)?;
        let arch = match &load {
            Some(l) => l.word("arch")?.to_owned(),
            None => String::new(),
        };
        println!(
            "server {} at {addr}; load record {:?}",
            model.display(),
            load.as_ref().map(Fields::line)
        );
        check(
            ok,
            "load_and_listen",
            matches!((&load, &listen), (Some(a), Some(b)) if a.at() < b.at()),
        );
        // Both records name the file's own architecture, the header's word
        // (`qwen35` for a dense file, not its routed sibling's).
        let listening_arch = match &listen {
            Some(l) => l.word("arch")?.to_owned(),
            None => String::new(),
        };
        println!(
            "records name architecture {arch:?} and {listening_arch:?}, the file's {:?}",
            seat.name()
        );
        check(
            ok,
            "records_name_the_files_architecture",
            arch == seat.name() && listening_arch == seat.name(),
        );
        // The server ran with no `--place`: the default is exactly one of
        // two things — the whole-card load, printing no `plan` record (a
        // card with room; today's lines), or the auto-placed plan, its one
        // record naming `whole_does_not_fit`. FAIL-first: a default that
        // plans without the why, or prints more than one plan record, turns
        // this red.
        let plans = records.all(&record::PLAN38)?;
        let whys = plans
            .iter()
            .map(|p| p.opt_word("why"))
            .collect::<Result<Vec<_>, _>>()?;
        let default_ok = whys.is_empty() || whys == [Some("whole_does_not_fit")];
        println!(
            "unplaced default: {} plan record(s), why {whys:?}",
            plans.len()
        );
        check(
            ok,
            "unplaced_default_names_itself",
            load.is_some() && default_ok,
        );
        // The whole-fit verdicts of the contexts the default asked, one at
        // the loaded context: `fits` exactly when the load is the whole-card
        // one (no `plan` record), and none at a larger context fits (a
        // context whose whole load fits holds). FAIL-first: a seat that
        // loads whole on a verdict that does not fit, or prints none, turns
        // this red.
        let verdicts: Vec<&str> = log
            .lines()
            .filter(|l| l.starts_with("whole fit at ctx "))
            .collect();
        let loaded_ctx = match &load {
            Some(l) => Some(l.u64("ctx")?),
            None => None,
        };
        println!("whole-fit verdicts {verdicts:?}");
        check(
            ok,
            "default_load_is_its_whole_fit",
            loaded_ctx.is_some_and(|c| {
                let at = |v: &str| -> Option<u64> {
                    v.strip_prefix("whole fit at ctx ")?
                        .split(':')
                        .next()?
                        .parse()
                        .ok()
                };
                let mut loaded = verdicts.iter().filter(|v| at(v) == Some(c));
                loaded
                    .next()
                    .is_some_and(|v| v.contains(": fits: ") == plans.is_empty())
                    && loaded.next().is_none()
                    && verdicts
                        .iter()
                        .all(|v| at(v).is_some_and(|x| x <= c || !v.contains(": fits: ")))
            }),
        );

        // The seat's default `--ctx` on this server (no `--ctx`, `--parallel
        // 1`): the largest step under the target the load holds, on either
        // card ([`ctx_default_is_the_largest_step_held`]).
        // FAIL-first: a choice that keeps the target whatever the card
        // makes the load a plan the card cannot hold (the spawn never
        // listens), and one that starts a step low is red there.
        let split = Split::open(model).map_err(|e| format!("open {}: {e}", model.display()))?;
        let trained = bloomery_gpu_gates::generate::trained_ctx(&split)
            .ok_or_else(|| format!("{}: no context_length", model.display()))?;
        let trained = u64::try_from(trained)?;
        let (st, body) = curl(&url("/props"), None, false)?;
        let default_n = json_of("/props", st, &body)?["n_ctx"]
            .as_u64()
            .unwrap_or(u64::MAX);
        println!("ctx arm default: props n_ctx {default_n} of trained {trained}");
        check(
            ok,
            "ctx_never_passes_the_trained_context",
            default_n <= trained,
        );
        let cache = records.one(&record::LOAD_QWEN3)?.word("cache")?.to_owned();
        println!("default arm: the load record's cache {cache}");
        check(ok, "default_load_names_f16", cache == "f16");
        // The common rule's record on the unset flag: a body with no tier
        // card runs `a`, on one card or two, names no break-even, basis or
        // tier count. FAIL-first: a seat that skips the record finds none;
        // one that keeps a tier names `bp` or a count.
        let placed = records.one(&record::PLACE_UNSET)?;
        let (place, why) = (placed.word("place")?, placed.text("why")?);
        let (tier, even, basis) = (
            placed.opt_u64("tier_experts")?,
            placed.opt_u64("break_even")?,
            placed.opt_word("basis")?,
        );
        println!(
            "default arm: place unset place={place} why={why} tier_experts={tier:?} \
             break_even={even:?} basis={basis:?}"
        );
        check(
            ok,
            "unset_place_is_the_rule",
            place == "a"
                && (why == "one card" || why == "two cards, the body serves no tier card")
                && tier.is_none()
                && even.is_none()
                && basis.is_none(),
        );
        ctx_default_is_the_largest_step_held(model, &split, &log, default_n, trained, ok)?;
        default_ctx_line_names_its_rule(&log, (1, default_n, trained), None, ok);
        default_load_counts_its_arena(&split, &log, default_n, &arch, ok)?;
        // Every choice the unset `--ctx` ran read the census once for all its
        // probes (the seat's line a choice). FAIL-first: a choice that
        // resolves its placement per probe reads the census once a probe.
        let runs = choices(&log);
        println!("ctx arm default: choices (probes, census readings) {runs:?}");
        check(
            ok,
            "ctx_choice_reads_the_census_once",
            !runs.is_empty() && runs.iter().all(|&(p, c)| p >= 1 && c == 1),
        );

        let (st, body) = curl(
            &url("/apply-template"),
            Some(&json!({ "messages": messages() })),
            false,
        )?;
        let text = json_of("/apply-template", st, &body)?["prompt"]
            .as_str()
            .ok_or("/apply-template: no prompt")?
            .to_owned();
        std::fs::write(dir.join("prompt.txt"), &text)?;
        let prompt = tokenized(&url, &text)?;
        println!("rendered prompt: {} ids", prompt.len());
        check(ok, "the_turn_is_past_a_pass", prompt.len() > PASS_IDS);

        let (chat_ids, v) = greedy(&url, prompt, dir, "completion", false)?;
        let prose = tokenized(&url, PROSE)?;
        println!("prose prompt: {} ids", prose.len());
        check(ok, "the_prose_is_past_a_pass", prose.len() > PASS_IDS);
        let (prose_ids, _) = greedy(&url, prose, dir, "prose", false)?;

        let chat_ok = chat_is_those_ids(&url, dir, messages(), N, (&chat_ids, &v))?;
        check(ok, "chat_is_those_ids", chat_ok);
        schema_jump_is_the_step_arms(&url, dir, body35(seat), ok)?;
        // The qwen3moe arm's whole-call answer joins the CLI comparison: a
        // fresh run's ids for the two-message chat.
        let mut whole = None;
        if seat == qwen_seat::Family::Qwen3moe {
            whole = Some(prefix(&url, dir, ok)?);
        }
        if body35(seat) {
            prefix35(&url, dir, ok)?;
            tool_call_is_parsed(&split, dir, ok)?;
        }
        *ok &= serve_client::shared_system_prompt(&url, dir, SYSTEM_MSG, [EDIT_A, EDIT_B], N)?;
        *ok &= crate::systemone_clause::is_the_schema_path(&url, dir)?;
        println!("server stopped: {}", s.stop()?);
        let mut answers = vec![chat_ids, prose_ids];
        answers.extend(whole);
        Ok(Some((answers, default_n)))
    }

    /// The turn the jump clause asks: a support message to classify.
    const JUMP_ASK: &str = "Classify this support message. Message: \"I was charged twice for my \
        order and I want my money back now!\" Give its intent and whether the customer is angry.";

    /// The jump clause's schema: the suites' median request, an intent of
    /// four options and a boolean.
    fn jump_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "intent": { "type": "string", "enum": ["cancel", "other", "refund", "tracking"] },
                "angry": { "type": "boolean" },
            },
            "required": ["intent", "angry"],
            "additionalProperties": false,
        })
    }

    /// `schema_jump_is_the_step_arms` (module header): [`JUMP_ASK`]'s turn
    /// under [`jump_schema`] on the jump arm and on the step arm (the same
    /// request asking `n_probs`), each body in `<dir>/jump-<arm>.json`; on a
    /// `Body35` file (`marks`), between them, `answer_spans_take_no_marks`.
    fn schema_jump_is_the_step_arms(
        url: &dyn Fn(&str) -> String,
        dir: &Path,
        marks: bool,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        // Thinking off: the answer opens on the schema's text, the lead.
        let (st, text) = curl(
            &url("/apply-template"),
            Some(&json!({
                "messages": [{ "role": "user", "content": JUMP_ASK }],
                "chat_template_kwargs": { "enable_thinking": false },
            })),
            false,
        )?;
        let text = json_of("/apply-template", st, &text)?["prompt"]
            .as_str()
            .ok_or("/apply-template: no prompt")?
            .to_owned();
        let prompt = tokenized(url, &text)?;
        let ask = |probs: bool| -> Result<Value, GateError> {
            let mut body = json!({
                "prompt": prompt, "n_predict": 64, "temperature": 0, "return_tokens": true,
                "cache_prompt": false,
                "response_format": { "type": "json_schema",
                    "json_schema": { "name": "ticket", "schema": jump_schema() } },
            });
            if probs {
                body["n_probs"] = json!(1);
            }
            let (st, text) = curl(&url("/completion"), Some(&body), false)?;
            let name = if probs {
                "jump-step.json"
            } else {
                "jump-jump.json"
            };
            std::fs::write(dir.join(name), &text)?;
            json_of("/completion", st, &text)
        };
        let jump = ask(false)?;
        let jt = ids_of(&jump["tokens"])?;
        let j = &jump["timings"]["constrained"];
        let count = |v: &Value, k: &str| v[k].as_u64().unwrap_or(u64::MAX);
        if marks {
            answer_spans_take_no_marks(url, dir, &prompt, &jt, j, ok)?;
        }
        let step = ask(true)?;
        let st = ids_of(&step["tokens"])?;
        let s = &step["timings"]["constrained"];
        println!(
            "schema jump: content {:?} / {:?}, stop {:?} / {:?}, {} / {} ids; jump {j}; step {s}",
            jump["content"],
            step["content"],
            jump["stop_type"],
            step["stop_type"],
            jt.len(),
            st.len(),
        );
        let same_answer = jump["content"] == step["content"]
            && jump["stop_type"] == step["stop_type"]
            && jt.len() == st.len()
            && !jt.is_empty()
            && jt[..jt.len() - 1] == st[..st.len() - 1];
        let designed = j["arm"] == "jump"
            && s["arm"] == "step"
            && count(j, "lead") > 0
            && count(j, "next_after_calls") == 1
            && count(j, "next_calls") == 0
            && count(j, "prefill_calls") == 0
            && count(j, "tail") > 0
            && count(j, "decisions") == count(s, "decisions");
        check(ok, "schema_jump_is_the_step_arms", same_answer && designed);
        Ok(())
    }

    /// `answer_spans_take_no_marks` (module header), on the slot the jump
    /// arm's request `jt` (counts `j`) left: the slot holds the prompt, the
    /// lead and the run up to the last pick, and a request of the prompt and
    /// all of that but its last id keeps the last checkpoint below — the
    /// prompt call's end, `P + lead - 1` (the call fed the prompt and the
    /// lead but its last id, which the call's step fed). The run's call took
    /// none; one that took marks keeps the run's start, `P + lead`.
    fn answer_spans_take_no_marks(
        url: &dyn Fn(&str) -> String,
        dir: &Path,
        prompt: &[u32],
        jt: &[u32],
        j: &Value,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let count = |k: &str| j[k].as_u64().and_then(|n| usize::try_from(n).ok());
        let (lead, fed) = match (count("lead"), count("span_ids")) {
            (Some(lead), Some(span)) => (lead, lead + span),
            _ => (0, 0),
        };
        if lead == 0 || fed < lead + 2 || fed > jt.len() {
            println!(
                "answer marks: lead {lead}, {fed} answer ids fed of {}: no run",
                jt.len()
            );
            check(ok, "answer_spans_take_no_marks", false);
            return Ok(());
        }
        let ids: Vec<u32> = prompt.iter().chain(&jt[..fed - 1]).copied().collect();
        let body = json!({
            "prompt": ids, "n_predict": 1, "temperature": 0, "cache_prompt": true,
        });
        let (st, text) = curl(&url("/completion"), Some(&body), false)?;
        std::fs::write(dir.join("jump-marks.json"), &text)?;
        let v = json_of("/completion", st, &text)?;
        let cache_n = v["timings"]["cache_n"].as_u64();
        let want = u64::try_from(prompt.len() + lead - 1)?;
        println!(
            "answer marks: the slot held {} ids, a request of {} keeps {cache_n:?}; the prompt \
             call's end is {want}",
            prompt.len() + fed,
            ids.len()
        );
        check(ok, "answer_spans_take_no_marks", cache_n == Some(want));
        Ok(())
    }

    /// The whole-fit verdict's cache term at `ctx`: the planes' bytes.
    fn cache_term(log: &str, ctx: u64) -> Result<u64, GateError> {
        let head = format!("whole fit at ctx {ctx}: ");
        let line = log
            .lines()
            .find(|l| l.starts_with(&head))
            .ok_or_else(|| format!("no whole-fit line at context {ctx}"))?;
        Ok(whole_terms(line)?.cache)
    }

    /// One `plan` record's routed split `(host_experts, card_experts, ctx_max)`,
    /// or none when the load printed no plan (the whole model on the card).
    /// More than one is a named error.
    fn routed_split(log: &record::Log) -> Result<Option<(u64, u64, u64)>, GateError> {
        let plans = log.all(&record::PLAN38)?;
        match plans.as_slice() {
            [] => Ok(None),
            [p] => Ok(Some((
                p.u64("host_experts")?,
                p.u64("card_experts")?,
                p.u64("ctx_max")?,
            ))),
            _ => Err(format!("{} plan records, the clause wants one split", plans.len()).into()),
        }
    }

    /// The cache flag's own surface (`--cache-type-k`): a word outside the
    /// two spellings refused by name before anything loads, `--cache-type-v`
    /// refused by name with the why (both planes quantize together), and
    /// the flag winning over the `BLOOMERY_QWEN3_KV` lever — a spawn with
    /// the lever at `q8_0` and the flag at `f16` loads the f16 planes and
    /// says so. FAIL-first: a parser that takes any word listens; one that
    /// lets the lever override the flag loads `cache=q8_0`.
    fn cache_refusals(model: &Path, dir: &Path, ok: &mut bool) -> Result<(), GateError> {
        let m = model.to_str().ok_or("the model path is not UTF-8")?;
        let refuses = |name: &str, want: &str| -> Result<bool, GateError> {
            let d = dir.join(format!("cache-{name}"));
            std::fs::create_dir_all(&d)?;
            let out = Command::new(beside("bloomery-serve")?)
                .args(["--model", "qwen3", "--port", "0", "-m", m])
                .args(
                    match name {
                        "word" => vec!["--cache-type-k", "q6_0"],
                        _ => vec!["--cache-type-v", "q8_0"],
                    }
                    .as_slice(),
                )
                .env_remove("BLOOMERY_REF_MODEL")
                .output()?;
            let said = String::from_utf8_lossy(&out.stderr).into_owned();
            std::fs::write(d.join("server.err"), &said)?;
            let refused = !out.status.success() && said.contains(want);
            println!(
                "cache arm {name}: exit {:?}, said {said}",
                out.status.code()
            );
            Ok(refused)
        };
        check(
            ok,
            "cache_type_k_refuses_other_words",
            refuses("word", "--cache-type-k takes f16 or q8_0, not \"q6_0\"")?,
        );
        check(
            ok,
            "cache_type_v_refused_naming_why",
            refuses("v", "--cache-type-v does not exist")?
                && std::fs::read_to_string(dir.join("cache-v").join("server.err"))?
                    .contains("together"),
        );
        // The flag wins over the lever: the spawn listens and its load names
        // the f16 planes.
        let d = dir.join("cache-flag-wins");
        std::fs::create_dir_all(&d)?;
        let err_log = d.join("server.err");
        let mut cmd = Command::new(beside("bloomery-serve")?);
        cmd.env_remove("BLOOMERY_REF_MODEL")
            .env("BLOOMERY_QWEN3_KV", "q8_0");
        let mut s = Served::spawn_cmd(
            cmd,
            &[
                "--model",
                "qwen3",
                "--port",
                "0",
                "--parallel",
                "1",
                "-m",
                m,
                "--cache-type-k",
                "f16",
            ],
            &d,
        )?;
        let addr = s.address(&err_log, 600, Duration::from_secs(1))?;
        let url = |p: &str| format!("http://{addr}{p}");
        let (st, body) = curl(&url("/props"), None, false)?;
        json_of("/props", st, &body)?;
        let cache = seat_log(&err_log)?
            .one(&record::LOAD_QWEN3)?
            .word("cache")?
            .to_owned();
        let f16 = cache == "f16";
        println!("cache arm flag-wins: the load record's cache {cache}");
        println!("cache arm flag-wins: server stopped: {}", s.stop()?);
        check(ok, "cache_type_k_wins_over_the_lever", f16);
        Ok(())
    }

    // ------------------------------------------------------------ images

    /// The chat requests the oracle sets E and F were dumped from, one JSON line a request
    /// (`tools/ref/qvis/make_requests.py`), and the images they name, under the tree this binary
    /// was built from.
    const REQUESTS: &str = "../../tools/ref/qvis/requests.jsonl";
    const IMAGE_DIR: &str = "../../tools/ref/clefvis/images";

    /// The tokens an image request answers: set F's decode steps.
    const F_STEPS: usize = 32;

    /// Set F's server's `--ctx`, a total over `--parallel 2`: a context at which the seat's plan
    /// keeps every routed expert on a 24 GB card beside the encoder's reserve, its load placed or,
    /// where the card has room for the whole load, whole. The oracle's ids are the CUDA path's;
    /// a host expert runs another rule, so past this context the clause would hold the host rule
    /// to the card's.
    const SET_F_CTX: usize = 16384;

    /// The widest gap between the server's top two logprobs at which an id that parts from the
    /// oracle's counts as a near tie, in nats.
    // PIN(2026-10-12): the near-tie floor of the reference gates (`MARGIN_FLOOR` of `gate_e2e`), from
    // this set's spread: two configurations of this server (the 3090 with every expert on the card,
    // the A6000, a placed plan) differ in the gap between the top two logprobs by 0.15-0.23 nats rms
    // where that gap is under 1 (8-9 steps a pair, `c1` and `c2`). The server-to-oracle spread is at
    // least that. At that spread a harmless first difference has a gap above 0.1 in 38-55 % of runs
    // and above 0.5 in 0.03-1.3 % [derived: the gap uniform on (0, |d|), d ~ N(0, spread^2), one
    // flip a run]. Not the grid's pin: a grid taken as `(ny, nx)` read 0.23-0.41 here (0.4134 on the
    // 3090, 0.2288 placed), inside this band; the vision crate's
    // `a_spans_grid_runs_along_the_images_width` holds that order.
    const F_NEAR_TIE: f64 = 0.5;

    /// The fewest ids a prompt's CUDA and CPU oracle runs must agree on for set F to hold the
    /// server to them: a shorter prefix says too little of the image to be a clause.
    const F_MIN_PREFIX: usize = 8;

    /// One prompt of set F: the oracle's CUDA decode ids and the length of the prefix on which its
    /// CPU twin (`decode_<n>.cpu`, a second reading of mainline) gives the same ids.
    struct Decoded {
        ids: Vec<u32>,
        agree: usize,
    }

    /// The set of `family` whose name ends in `suffix`, opened as a set of that family.
    fn open_set(family: &'static Family, suffix: &str) -> Result<ClefvisSet, GateError> {
        let name = family
            .sets
            .iter()
            .find(|s| s.ends_with(suffix))
            .ok_or_else(|| format!("the {} family names no set ending in {suffix}", family.name))?;
        Ok(ClefvisSet::open(&family.path(name), family)?)
    }

    /// The image clauses' facts: the encoder file (the family's projector, held to its digest) and
    /// its card figure and seat under `--place a`, the oracle sets of the Qwen3.6 image-input
    /// families (`refset::arch::qwen35moe::vis`; llama.cpp mainline's mtmd on the same model
    /// file) — E's chat ids of e1..e3 and c1, c2, F's 32 greedy ids after c1 and c2 — and the
    /// requests and images they were dumped from.
    struct Images {
        mmproj: PathBuf,
        /// The tower's weight and activation bytes at the default token limits.
        bytes: CardBytes,
        /// Where the plan seats the tower: a reserve of `bytes.total()` on the stage card when
        /// every card this process holds is on the plan, else off it.
        seat: EncoderSeat,
        e: Vec<ClefvisSet>,
        c: Vec<ClefvisSet>,
        f: Vec<Decoded>,
        requests: BTreeMap<String, Value>,
        pngs: BTreeMap<String, Vec<u8>>,
    }

    impl Images {
        fn read() -> Result<Images, GateError> {
            let mmproj = PathBuf::from(vis::MMPROJ_BF16);
            let got = gguf::digest::sha256_file(&mmproj)?;
            if got != vis::MMPROJ_BF16_SHA256 {
                return Err(format!(
                    "{} is {got}; the Qwen3.6 sets were dumped with {}",
                    mmproj.display(),
                    vis::MMPROJ_BF16_SHA256
                )
                .into());
            }
            let file = Gguf::open(&mmproj)?;
            let hp = qwen3vl::Hparams::read(&file)?;
            let bytes = qwen3vl::card::of(&hp, TokenLimits::DEFAULT);
            let seat = vision_seat::encoder_seat(Place::A, bytes.total())?;
            let set = |f: &'static Family, tail: &str| open_set(f, tail);
            let e = ["e1", "e2", "e3"]
                .map(|n| set(&CHATIDS, &format!("chatids_{n}")))
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;
            let c = ["c1", "c2"]
                .map(|n| set(&CHATIDS, &format!("chatids_{n}")))
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;
            let mut f = Vec::new();
            for n in ["c1", "c2"] {
                // The CUDA run's ids; the CPU twin of the same prompt is a second reading of
                // mainline, whose agreement with it is printed.
                let cuda = open_set(&DECODE, &format!("decode_{n}"))?.decode_ids()?;
                let twin = open_set(&DECODE, &format!("decode_{n}.cpu"))?.decode_ids()?;
                let agree = cuda.iter().zip(&twin).take_while(|(a, b)| a == b).count();
                println!(
                    "set F {n}: {} CUDA ids, the CPU twin agrees on the first {agree}",
                    cuda.len()
                );
                if agree < F_MIN_PREFIX {
                    return Err(format!(
                        "set F {n}: the CUDA and CPU ids agree on {agree} ids, fewer than \
                         {F_MIN_PREFIX}: no prefix to hold the server to"
                    )
                    .into());
                }
                f.push(Decoded { ids: cuda, agree });
            }
            let root = Path::new(env!("CARGO_MANIFEST_DIR"));
            let text = std::fs::read_to_string(root.join(REQUESTS))?;
            let mut requests = BTreeMap::new();
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                let v: Value = serde_json::from_str(line)?;
                let id = v["id"].as_str().ok_or("a request with no id")?.to_owned();
                requests.insert(id, v["messages"].clone());
            }
            let mut pngs = BTreeMap::new();
            for set in e.iter().chain(&c) {
                for image in &set.images {
                    let path = root.join(IMAGE_DIR).join(format!("{}.png", image.name));
                    if gguf::digest::sha256_file(&path)? != image.png_sha256 {
                        return Err(format!(
                            "{} is not the image the sets were dumped from ({})",
                            path.display(),
                            image.png_sha256
                        )
                        .into());
                    }
                    pngs.insert(image.name.clone(), std::fs::read(&path)?);
                }
            }
            Ok(Images {
                mmproj,
                bytes,
                seat,
                e,
                c,
                f,
                requests,
                pngs,
            })
        }

        /// The chat body of request `id` of the sets, its images as `data:` URLs, the thinking
        /// switch the set's chat line records.
        fn chat(&self, id: &str, thinking: bool) -> Result<Value, GateError> {
            let mut messages = self
                .requests
                .get(id)
                .cloned()
                .ok_or_else(|| format!("no request {id} in {REQUESTS}"))?;
            for m in messages.as_array_mut().into_iter().flatten() {
                for part in m["content"].as_array_mut().into_iter().flatten() {
                    if part["type"] != "image_url" {
                        continue;
                    }
                    let name = part["image_url"]["url"]
                        .as_str()
                        .ok_or("an image part with no name")?
                        .to_owned();
                    let png = self
                        .pngs
                        .get(&name)
                        .ok_or_else(|| format!("request {id}: no image {name} among the sets'"))?;
                    part["image_url"]["url"] = json!(format!(
                        "data:image/png;base64,{}",
                        serve_client::base64(png)
                    ));
                }
            }
            Ok(json!({
                "messages": messages,
                "chat_template_kwargs": { "enable_thinking": thinking },
            }))
        }

        /// [`Images::chat`] of the case `n` (`c1`, `c2`, `e1`, `e2`, `e3`) as its chat-ids set's
        /// chat line renders it.
        fn case(&self, n: &str) -> Result<Value, GateError> {
            let (sets, at) = match n {
                "c1" => (&self.c, 0),
                "c2" => (&self.c, 1),
                "e1" => (&self.e, 0),
                "e2" => (&self.e, 1),
                "e3" => (&self.e, 2),
                other => return Err(format!("no chat-ids set of case {other}").into()),
            };
            self.chat(n, sets[at].chat()?.enable_thinking)
        }
    }

    /// `ids` with each of `spans` — a run of the image-pad id — cut to the one id the chat
    /// template's placeholder tokenizes to: the prompt before the server expands its images.
    fn collapsed(ids: &[u32], spans: &[Span]) -> Result<Vec<u32>, GateError> {
        let (mut out, mut at) = (Vec::new(), 0);
        for s in spans {
            let run = ids
                .get(s.at..s.at + s.len)
                .filter(|r| !r.is_empty() && r.iter().all(|&i| i == r[0]))
                .ok_or_else(|| format!("span {} is not one id repeated", s.index))?;
            out.extend_from_slice(&ids[at..s.at]);
            out.push(run[0]);
            at = s.at + s.len;
        }
        out.extend_from_slice(&ids[at..]);
        Ok(out)
    }

    /// A server of `args` over `env` beside `--model qwen3 -m <model>` whose stderr is read until
    /// a line starts with `until`, then stopped: its stderr. The seat prints its plan and its
    /// whole-fit verdict before it opens the model, so no clause on them waits on a load.
    fn before_load(
        model: &Path,
        dir: &Path,
        args: &[&str],
        env: &[(&str, &str)],
        until: &str,
    ) -> Result<String, GateError> {
        std::fs::create_dir_all(dir)?;
        let m = model.to_str().ok_or("the model path is not UTF-8")?;
        let mut cmd = serve_cmd()?;
        for &(k, v) in env {
            cmd.env(k, v);
        }
        let mut all = vec!["--model", "qwen3", "--port", "0", "-m", m];
        all.extend_from_slice(args);
        let mut s = Served::spawn_cmd(cmd, &all, dir)?;
        let err_log = dir.join("server.err");
        for _ in 0..720 {
            let text = std::fs::read_to_string(&err_log).unwrap_or_default();
            if text.lines().any(|l| l.starts_with(until)) || s.child.try_wait()?.is_some() {
                s.stop()?;
                return Ok(std::fs::read_to_string(&err_log)?);
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        Err(format!("the server printed no `{until}` line in 3 minutes ({all:?})").into())
    }

    /// The server command with its step counters on (`BLOOMERY_STEP_STATS=1`, which only prints
    /// the `slots round` records the slot clauses read): the one place this gate sets it.
    fn stats_cmd() -> Result<Command, GateError> {
        let mut cmd = serve_cmd()?;
        cmd.env(bloomery_levers::STEP_STATS, "1");
        Ok(cmd)
    }

    /// A chat request posted on a helper thread ([`post_chat`]): the thread and the receiver of
    /// its status and body.
    type Posted = (
        JoinHandle<()>,
        mpsc::Receiver<Result<(u16, String), String>>,
    );

    /// One chat request `body` posted on a helper thread of its own, as [`streamed`] posts a
    /// completion.
    fn post_chat(addr: &str, body: Value) -> Result<Posted, GateError> {
        let (tx, rx) = mpsc::channel();
        let url = format!("http://{addr}/v1/chat/completions");
        let (h, _) = spawn_helper("image-request", Placement::Float, move || {
            let _ = tx.send(curl(&url, Some(&body), false).map_err(|e| e.to_string()));
        })
        .map_err(|e| format!("image slots: {}", e.what()))?;
        Ok((h, rx))
    }

    /// The `vision` record and `/props` of the image server (`--mmproj` over `--parallel 2`,
    /// `BLOOMERY_STEP_STATS=1`, no `--ctx`: the seat's own default) and the clauses that read
    /// them: `mmproj_props_vision_true` and `mmproj_vision_record_is_the_towers_figure`; then the
    /// requests' clauses ([`image_requests`]).
    fn image_server(
        model: &Path,
        img: &Images,
        dir: &Path,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let d = dir.join("server");
        std::fs::create_dir_all(&d)?;
        let err_log = d.join("server.err");
        let cmd = stats_cmd()?;
        let m = model.to_str().ok_or("the model path is not UTF-8")?;
        let mm = img
            .mmproj
            .to_str()
            .ok_or("the projector path is not UTF-8")?;
        let mut s = Served::spawn_cmd(
            cmd,
            &[
                "--model",
                "qwen3",
                "--port",
                "0",
                "--parallel",
                "2",
                "-m",
                m,
                "--mmproj",
                mm,
            ],
            &d,
        )?;
        let addr = match s.address(&err_log, 900, Duration::from_secs(1)) {
            Ok(a) => a,
            Err(e) => {
                println!("the image server never listened: {e}");
                check(ok, "mmproj_server_listens", false);
                return Ok(());
            }
        };
        let url = |p: &str| format!("http://{addr}{p}");
        let (st, body) = curl(&url("/props"), None, false)?;
        let props = json_of("/props", st, &body)?;
        println!("props modalities {}", props["modalities"]);
        check(
            ok,
            "mmproj_props_vision_true",
            props["modalities"]["vision"] == json!(true),
        );

        let log = seat_log(&err_log)?;
        let vision = log.first(&record::VISION)?;
        let seat = &img.seat;
        let figure = (
            seat.name.replace(' ', "_"),
            seat.word(),
            img.bytes.weights,
            img.bytes.scratch,
        );
        let seen = match &vision {
            Some(f) => Some((
                f.word("card")?.to_owned(),
                f.word("seat")?,
                f.u64("weights")?,
                f.u64("scratch")?,
                f.u64("cap")?,
                f.word("scratch_home")?,
            )),
            None => None,
        };
        println!("vision record {seen:?}; the figure {figure:?}");
        check(
            ok,
            "mmproj_vision_record_is_the_towers_figure",
            seen.as_ref().is_some_and(|(card, word, w, sc, cap, home)| {
                (card.as_str(), *word, *w, *sc) == (figure.0.as_str(), figure.1, figure.2, figure.3)
                    && *cap == u64::try_from(TokenLimits::DEFAULT.max()).unwrap_or(0)
                    && *home == "own"
            }),
        );

        image_requests(img, &addr, &err_log, ok)?;
        println!("image server stopped: {}", s.stop()?);
        Ok(())
    }

    /// How a Qwen3 seat's load holds the routed experts.
    enum Experts {
        /// The whole-card load.
        Whole,
        /// A placed load: the host experts its `plan` record keeps and the context it was made at.
        Placed { host: u64, ctx_max: u64 },
    }

    /// The load a server's stderr `text` (`what` names it) says it made, by record kind: a
    /// `load_qwen3` record, and either one `plan` record, or none and the one whole-fit verdict
    /// saying it fits (`default_load_is_its_whole_fit`'s pairing). A record absent where one is
    /// due, a line that opens one and does not read whole, and a load with no `plan` record whose
    /// verdict is not there or does not say it fits are errors naming it, never a whole load by
    /// default.
    fn experts_in(text: &str, what: &str) -> Result<Experts, GateError> {
        let log = record::Log::of(text, record::BLOOMERY_SERVE_QWEN3).named(what);
        log.one(&record::LOAD_QWEN3)?;
        if let Some(plan) = log.first(&record::PLAN38)? {
            return Ok(Experts::Placed {
                host: plan.u64("host_experts")?,
                ctx_max: plan.u64("ctx_max")?,
            });
        }
        let verdicts: Vec<&str> = text
            .lines()
            .filter(|l| l.starts_with("whole fit at ctx "))
            .collect();
        match verdicts[..] {
            [v] if v.contains(": fits: ") => Ok(Experts::Whole),
            _ => Err(format!(
                "{what}: the load printed no `plan` record and its whole-fit verdicts are not one \
                 line saying it fits: {verdicts:?}"
            )
            .into()),
        }
    }

    /// `mmproj_image_chat_follows_set_f`, on a server of its own: `--mmproj` over `--parallel 2` at
    /// [`SET_F_CTX`]. Its load is read first ([`experts_in`],
    /// `mmproj_set_f_server_keeps_every_expert_on_the_card`): a plan that keeps a routed expert on
    /// the host is refused by name and set F is not run, since a host expert's rule is not the
    /// oracle's, and so is a load it cannot read. A load with no plan record is the whole model
    /// on the card when its whole-fit verdict says so.
    fn image_set_f(model: &Path, img: &Images, dir: &Path, ok: &mut bool) -> Result<(), GateError> {
        let d = dir.join("set-f");
        std::fs::create_dir_all(&d)?;
        let err_log = d.join("server.err");
        let m = model.to_str().ok_or("the model path is not UTF-8")?;
        let mm = img
            .mmproj
            .to_str()
            .ok_or("the projector path is not UTF-8")?;
        let ctx = SET_F_CTX.to_string();
        let mut s = Served::spawn_cmd(
            serve_cmd()?,
            &[
                "--model",
                "qwen3",
                "--port",
                "0",
                "--parallel",
                "2",
                "-m",
                m,
                "--mmproj",
                mm,
                "--ctx",
                ctx.as_str(),
            ],
            &d,
        )?;
        let addr = match s.address(&err_log, 900, Duration::from_secs(1)) {
            Ok(a) => a,
            Err(e) => {
                println!("set F's server never listened: {e}");
                check(ok, "mmproj_set_f_server_listens", false);
                return Ok(());
            }
        };
        let text = std::fs::read_to_string(&err_log)?;
        let on_card = match experts_in(&text, "set F's server") {
            Ok(Experts::Whole) => true,
            Ok(Experts::Placed { host, ctx_max }) => {
                if host > 0 {
                    println!(
                        "set F needs every routed expert on the card: the plan at ctx {ctx_max} \
                         keeps {host} on the host"
                    );
                }
                host == 0
            }
            Err(e) => {
                println!("set F needs every routed expert on the card: its load is unread: {e}");
                false
            }
        };
        check(
            ok,
            "mmproj_set_f_server_keeps_every_expert_on_the_card",
            on_card,
        );
        if on_card {
            let url = |p: &str| format!("http://{addr}{p}");
            let mut f = true;
            for (k, n) in ["c1", "c2"].into_iter().enumerate() {
                let body = greedy_chat(&img.case(n)?, F_STEPS);
                let (st, text) = curl(&url("/v1/chat/completions"), Some(&body), false)?;
                std::fs::write(dir.join(format!("chat-{n}.json")), &text)?;
                let a = answered(st, &text)?;
                let (agreed, gap, holds) = a.follows(&img.f[k].ids, img.f[k].agree, F_NEAR_TIE);
                println!(
                    "F {n}: {} ids, the prompt {} tokens (set {}); equal to the oracle's CUDA ids \
                     for {agreed} of the {} the oracle's CPU twin agrees on (the gap at the first \
                     difference {gap:?}), the rule {}; said {:?}",
                    a.ids.len(),
                    a.prompt,
                    img.c[k].ids()?.len(),
                    img.f[k].agree,
                    if holds { "holds" } else { "fails" },
                    a.text.chars().take(80).collect::<String>()
                );
                f &= holds && a.prompt == img.c[k].ids()?.len() as u64;
            }
            check(ok, "mmproj_image_chat_follows_set_f", f);
        }
        println!("set F's server stopped: {}", s.stop()?);
        Ok(())
    }

    /// The whole-fit line of an image load (`--mmproj` over `--parallel 2`, no `--ctx`: the seat's
    /// own default), read from the lines the seat prints before it loads, so the verdict stands on
    /// its own: a verdict that leaves the reserve out would load the whole model and leave the tower
    /// no room. `mmproj_whole_fit_names_the_reserve` and
    /// `mmproj_whole_fit_counts_the_rotation_tables`.
    fn image_whole_fit(
        model: &Path,
        img: &Images,
        dir: &Path,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let mm = img
            .mmproj
            .to_str()
            .ok_or("the projector path is not UTF-8")?;
        let err = before_load(
            model,
            &dir.join("whole-fit"),
            &["--parallel", "2", "--mmproj", mm],
            &[],
            "parallel rule=",
        )?;
        let fit = err
            .lines()
            .find(|l| l.starts_with("whole fit at ctx "))
            .ok_or("the image seat printed no whole-fit line")?;
        println!("{fit}");
        let seat = &img.seat;
        let named = format!(
            "; the reserves term holds the {}'s {} B",
            beside::IMAGE_ENCODER,
            seat.reserve.unwrap_or(0)
        );
        check(
            ok,
            "mmproj_whole_fit_names_the_reserve",
            match seat.reserve {
                Some(r) => r == img.bytes.total() && fit.ends_with(&named),
                None => !fit.contains(beside::IMAGE_ENCODER),
            },
        );
        check(
            ok,
            "mmproj_whole_fit_counts_the_rotation_tables",
            tables_in_the_cache(model, fit, 2)?,
        );
        Ok(())
    }

    /// `rope.dimension_count` of the file: the f32 a sequence's rotation table holds a position.
    fn rope_width(model: &Path) -> Result<u64, GateError> {
        let split = Split::open(model).map_err(|e| format!("open {}: {e}", model.display()))?;
        split
            .arch_get_u64("rope.dimension_count")
            .ok_or_else(|| format!("{}: no rope.dimension_count", model.display()).into())
    }

    /// The cache term of the whole-fit line `fit` of a load of `slots` slots is the allocator's
    /// bytes of the file's own planes ([`cache_planes`]) with, after each slot's, the rotation
    /// table of a media load: one f32 row of the file's rope width a position of the slot's
    /// share (`rope.dimension_count`, 64 on Qwen3.6, 256 B a position).
    fn tables_in_the_cache(model: &Path, fit: &str, slots: u64) -> Result<bool, GateError> {
        let total: u64 = fit
            .strip_prefix("whole fit at ctx ")
            .and_then(|t| t.split(':').next())
            .and_then(|t| t.parse().ok())
            .ok_or_else(|| format!("no context in the whole-fit line: {fit}"))?;
        let said: u64 = fit
            .split(" + cache ")
            .nth(1)
            .and_then(|t| t.split(" B").next())
            .and_then(|t| t.parse().ok())
            .ok_or("the whole-fit line names no cache bytes")?;
        let split = Split::open(model).map_err(|e| format!("open {}: {e}", model.display()))?;
        let rope = rope_width(model)?;
        let rows = total / slots;
        let mut planes = Vec::new();
        for _ in 0..slots {
            planes.extend(cache_planes(model, &split, rows, 1)?);
            planes.push(rows * rope * 4);
        }
        let want = model::placement::allocator_bytes(serve_granule()?, planes);
        println!(
            "tables: {slots} slots of {rows} rows, the line's cache {said} B, the allocator's \
             bytes of the planes and the {rope}-wide tables {want} B"
        );
        Ok(said == want)
    }

    /// The image server's request clauses (module header): `mmproj_prompt_ids_are_set_e` and
    /// `mmproj_image_slot_beside_a_text_slot`; set F has a server of its own ([`image_set_f`]).
    fn image_requests(
        img: &Images,
        addr: &str,
        err_log: &Path,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let url = |p: &str| format!("http://{addr}{p}");

        // J1: the prompt each request of set E renders to is mainline's, an image one
        // placeholder (`/apply-template`, `/tokenize`) the server expands to its span's ids —
        // and the count it evaluates is the set's.
        let mut j1 = true;
        for (k, set) in img.e.iter().enumerate() {
            let n = format!("e{}", k + 1);
            let body = img.case(&n)?;
            let (st, text) = curl(&url("/apply-template"), Some(&body), false)?;
            let prompt = json_of("/apply-template", st, &text)?["prompt"]
                .as_str()
                .ok_or("/apply-template: no prompt")?
                .to_owned();
            let ids = tokenized(&url, &prompt)?;
            let oracle = set.ids()?;
            let want = collapsed(&oracle, &set.spans)?;
            let mut chat = body.clone();
            chat["max_tokens"] = json!(1);
            chat["temperature"] = json!(0);
            chat["cache_prompt"] = json!(false);
            let (st, text) = curl(&url("/v1/chat/completions"), Some(&chat), false)?;
            let served =
                json_of("/v1/chat/completions", st, &text)?["usage"]["prompt_tokens"].as_u64();
            println!(
                "J1 {n}: the rendered prompt {} ids (set {}); the server evaluated {served:?} \
                 (set {})",
                ids.len(),
                want.len(),
                oracle.len()
            );
            j1 &= ids == want && served == u64::try_from(oracle.len()).ok();
        }
        check(ok, "mmproj_prompt_ids_are_set_e", j1);

        // S6: an image request on one slot beside a text request on the other. Each answers
        // what it answered alone, and the server ran rounds of both slots together.
        let text = tokenized(&url, PROSE)?;
        let image = greedy_chat(&img.case("c1")?, SLOTS_PREDICT);
        let alone_text = {
            let (h, rx) = streamed(addr, &text, SLOTS_PREDICT)?;
            h.join().map_err(|_| "image slots: a thread panicked")?;
            rx.recv().map_err(|_| "image slots: no answer")??.tokens
        };
        let (st, body) = curl(&url("/v1/chat/completions"), Some(&image), false)?;
        let alone_image = answered(st, &body)?;
        let first = streamed(addr, &text, SLOTS_PREDICT)?;
        let decode0 = metric(&url, "n_decode_total")?.unwrap_or(f64::NAN);
        let mut polls = 0;
        while polls < LATE_POLLS
            && metric(&url, "n_decode_total")?.unwrap_or(decode0) < decode0 + LATE_ROUNDS as f64
        {
            std::thread::sleep(LATE_POLL);
            polls += 1;
        }
        let (second, second_rx) = post_chat(addr, image)?;
        let (h, rx) = first;
        h.join().map_err(|_| "image slots: a thread panicked")?;
        let together_text = rx.recv().map_err(|_| "image slots: no answer")??.tokens;
        second
            .join()
            .map_err(|_| "image slots: the image request's thread panicked")?;
        let (st, body) = second_rx
            .recv()
            .map_err(|_| "image slots: no image answer")??;
        let together_image = answered(st, &body)?;
        let both = seat_log(err_log)?
            .all(&record::SLOTS_ROUND)?
            .iter()
            .filter(|r| r.u64("rows") == Ok(2))
            .count();
        println!(
            "image slots: the image request posted after {polls} poll(s); {} text ids and {} \
             image ids; {both} round(s) of both slots",
            together_text.len(),
            together_image.ids.len(),
        );
        check(
            ok,
            "mmproj_image_slot_beside_a_text_slot",
            both > 0
                && !together_text.is_empty()
                && together_text == alone_text
                && !together_image.ids.is_empty()
                && together_image.ids == alone_image.ids,
        );
        Ok(())
    }

    /// The placed plan of an image load (`--place a` under [`PLACED_BUDGET`]): at one context it
    /// keeps as many routed experts on the card as the plan of a text load under the budget less
    /// what the image load sets aside of it — the tower's reserve when the tower sits on the stage
    /// card, and each slot's rotation table of its share of the context — and fewer than the plan
    /// without `--mmproj` at the full budget when the tower sits there
    /// (`mmproj_plan_holds_the_reserve`); and the default `--ctx` line names the reserve
    /// (`mmproj_ctx_default_names_the_reserve`).
    fn image_plan(model: &Path, img: &Images, dir: &Path, ok: &mut bool) -> Result<(), GateError> {
        let mm = img
            .mmproj
            .to_str()
            .ok_or("the projector path is not UTF-8")?;
        let env = [(bloomery_levers::CARD_BUDGET, PLACED_BUDGET)];
        let (ctx, slots) = (16384_u64, 2_u64);
        let ctx_arg = ctx.to_string();
        let experts = |name: &str, extra: &[&str], budget: &str| -> Result<u64, GateError> {
            let mut args = vec!["--place", "a", "--parallel", "2", "--ctx", ctx_arg.as_str()];
            args.extend_from_slice(extra);
            let d = dir.join(name);
            before_load(
                model,
                &d,
                &args,
                &[(bloomery_levers::CARD_BUDGET, budget)],
                "plan ",
            )?;
            let plan = seat_log(&d.join("server.err"))?
                .first(&record::PLAN38)?
                .ok_or_else(|| format!("{name}: the seat printed no plan record"))?;
            Ok(plan.u64("card_experts")?)
        };
        let without = experts("plan-text", &[], PLACED_BUDGET)?;
        let with = experts("plan-image", &["--mmproj", mm], PLACED_BUDGET)?;
        let tables = slots * (ctx / slots) * rope_width(model)? * 4;
        let aside = img.seat.reserve.unwrap_or(0) + tables;
        let budget = bloomery_levers::parse_bytes(PLACED_BUDGET)
            .map_err(|e| format!("{PLACED_BUDGET}: {e}"))?
            - aside;
        let same = experts("plan-text-less", &[], &budget.to_string())?;
        println!(
            "placed plan at ctx {ctx}: {without} card experts without --mmproj, {with} with it (the \
             tower {}), {same} of a text load under the budget less the {aside} B it sets aside \
             ({tables} B of tables)",
            img.seat.word()
        );
        check(
            ok,
            "mmproj_plan_holds_the_reserve",
            with == same && (img.seat.reserve.is_none() || with < without),
        );

        let d = dir.join("plan-ctx");
        let err = before_load(
            model,
            &d,
            &["--place", "a", "--parallel", "2", "--mmproj", mm],
            &env,
            "plan ",
        )?;
        let line = ctx_line(&err).ok_or("the placed image server printed no one ctx line")?;
        println!("{line}");
        let counted = format!(
            ", the {} reserve of {} B counted",
            beside::IMAGE_ENCODER,
            img.seat.reserve.unwrap_or(0)
        );
        check(
            ok,
            "mmproj_ctx_default_names_the_reserve",
            match img.seat.reserve {
                Some(_) => line.contains(&counted),
                None => !line.contains(beside::IMAGE_ENCODER),
            },
        );
        Ok(())
    }

    /// An image part on a seat with no `--mmproj` is the server's own refusal of every seat that
    /// takes no image: a 400 naming the kind, never a prompt run without the image
    /// (`image_without_mmproj_is_refused`).
    fn image_without_mmproj_is_refused(
        model: &Path,
        img: &Images,
        dir: &Path,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let d = dir.join("no-mmproj");
        std::fs::create_dir_all(&d)?;
        let err_log = d.join("server.err");
        let mut s = spawn(model, &d)?;
        let addr = s.address(&err_log, 600, Duration::from_secs(1))?;
        let (st, body) = curl(
            &format!("http://{addr}/v1/chat/completions"),
            Some(&img.case("c1")?),
            false,
        )?;
        println!("an image on the seat with no --mmproj: {st} {body}");
        check(
            ok,
            "image_without_mmproj_is_refused",
            st == 400 && body.contains("image input is not supported: this server reads text only"),
        );
        println!("no-mmproj server stopped: {}", s.stop()?);
        Ok(())
    }

    /// `--mmproj` on a Qwen3-30B file is refused at the open, by name, before any card work: the
    /// server exits on its own with the file named, never listens
    /// (`mmproj_on_a_qwen3moe_file_is_refused`).
    fn mmproj_on_a_qwen3moe_file_is_refused(
        model: &Path,
        mmproj: &Path,
        dir: &Path,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let d = dir.join("mmproj-qwen3moe");
        std::fs::create_dir_all(&d)?;
        let m = model.to_str().ok_or("the model path is not UTF-8")?;
        let mm = mmproj.to_str().ok_or("the projector path is not UTF-8")?;
        let mut s = Served::spawn_cmd(
            serve_cmd()?,
            &["--model", "qwen3", "--port", "0", "-m", m, "--mmproj", mm],
            &d,
        )?;
        let mut status = None;
        for _ in 0..240 {
            status = s.child.try_wait()?;
            if status.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        let said = std::fs::read_to_string(d.join("server.err"))?;
        println!("--mmproj on the qwen3moe file: exit {status:?}, said {said}");
        let refused = status.is_some_and(|s| !s.success())
            && said.contains(&format!("--mmproj {}", mmproj.display()))
            && said.contains("is a qwen3moe file")
            && said.contains("no image input");
        check(ok, "mmproj_on_a_qwen3moe_file_is_refused", refused);
        if status.is_none() {
            println!("the server listened: stopped {}", s.stop()?);
        }
        Ok(())
    }

    /// The image clauses (module header) on `model`: the qwen3moe arm holds the open's refusal,
    /// the qwen35moe arm the seat with the encoder.
    fn images(model: &Path, dir: &Path, ok: &mut bool) -> Result<(), GateError> {
        let arch = {
            let split = Split::open(model).map_err(|e| format!("open {}: {e}", model.display()))?;
            split.architecture().map(str::to_owned)
        };
        let d = dir.join("images");
        std::fs::create_dir_all(&d)?;
        match arch.as_deref() {
            Some("qwen3moe") => {
                mmproj_on_a_qwen3moe_file_is_refused(model, Path::new(vis::MMPROJ_BF16), &d, ok)
            }
            Some("qwen35moe") => {
                let img = Images::read()?;
                println!(
                    "the encoder {} ({} B of weights, {} B of activations): {} under --place a",
                    img.mmproj.display(),
                    img.bytes.weights,
                    img.bytes.scratch,
                    img.seat.word()
                );
                image_without_mmproj_is_refused(model, &img, &d, ok)?;
                image_plan(model, &img, &d, ok)?;
                image_whole_fit(model, &img, &d, ok)?;
                image_server(model, &img, &d, ok)?;
                image_set_f(model, &img, &d, ok)
            }
            Some("qwen35") => {
                not_for_dense(
                    "images",
                    "the 27B's multimodal projector clauses are a later round",
                );
                Ok(())
            }
            other => Err(format!("{}: images on a {other:?} file", model.display()).into()),
        }
    }

    /// Greedy drafted ids equal the plain path's, and the drafted completion
    /// accepted at least one token. A file with no in-file next-token layer
    /// prints the pin and starts no server; the pinned path with no layer is red.
    /// A spawn, tokenize or completion failure is red and does not stop the
    /// file's later clauses or the next file.
    fn mtp_ids_match_the_plain_path(
        model: &Path,
        dir: &Path,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let file = match Split::open(model) {
            Ok(file) => file,
            Err(e) => {
                println!("mtp open {}: {e}", model.display());
                check(ok, "mtp_ids_match_the_plain_path", false);
                return Ok(());
            }
        };
        let drafted = matches!(in_file(&file), Ok(InFile::Draft(_)));
        let passed = if !drafted {
            if model == Path::new(MTP_FILE) {
                println!(
                    "{}: the pinned file carries no in-file next-token layer",
                    model.display()
                );
                Some(false)
            } else {
                println!(
                    "{}: no in-file next-token layer; mtp_ids_match_the_plain_path is pinned to \
                     {MTP_FILE}",
                    model.display()
                );
                None
            }
        } else {
            Some(mtp_pair(model, dir))
        };
        if let Some(passed) = passed {
            check(ok, "mtp_ids_match_the_plain_path", passed);
        }
        Ok(())
    }

    /// The two serves, one after the other so one card is free for the second.
    /// `false` when either fails to listen or to complete.
    fn mtp_pair(model: &Path, dir: &Path) -> bool {
        let ids = match Tokenizer::from_gguf(model) {
            Ok(tok) => tok.encode(PROSE, false, true),
            Err(e) => {
                println!("mtp tokenize {}: {e}", model.display());
                return false;
            }
        };
        let Some(utf8) = model.to_str() else {
            println!("mtp: the model path is not UTF-8");
            return false;
        };
        let once = |name: &str, word: &str| -> Option<(Answer, Value)> {
            let sub = dir.join(name);
            if let Err(e) = std::fs::create_dir_all(&sub) {
                println!("mtp {name} dir: {e}");
                return None;
            }
            let mut cmd = match serve_cmd() {
                Ok(cmd) => cmd,
                Err(e) => {
                    println!("mtp {name} command: {e}");
                    return None;
                }
            };
            cmd.env("BLOOMERY_DRAFT", word);
            let mut s = match Served::spawn_cmd(
                cmd,
                &[
                    "--model",
                    "qwen3",
                    "--port",
                    "0",
                    "--parallel",
                    "1",
                    "-m",
                    utf8,
                ],
                &sub,
            ) {
                Ok(s) => s,
                Err(e) => {
                    println!("mtp {name} spawn: {e}");
                    return None;
                }
            };
            let err_log = sub.join("server.err");
            let addr = match s.address(&err_log, 600, Duration::from_secs(1)) {
                Ok(addr) => addr,
                Err(e) => {
                    println!("mtp {name} never listened: {e}");
                    return None;
                }
            };
            let url = |p: &str| format!("http://{addr}{p}");
            match serve_client::greedy(&url, ids.clone(), MTP_N, &sub, name, false) {
                Ok(v) => Some(v),
                Err(e) => {
                    println!("mtp {name} greedy: {e}");
                    None
                }
            }
        };
        let Some((on, on_body)) = once("mtp-on", "mtp") else {
            return false;
        };
        let Some((off, _)) = once("mtp-off", "off") else {
            return false;
        };
        let accepted = on_body["timings"]["draft_n_accepted"].as_u64();
        let same = on.tokens == off.tokens;
        println!(
            "mtp drafted {} ids, plain {} ids, draft_n_accepted {accepted:?}",
            on.tokens.len(),
            off.tokens.len()
        );
        same && accepted.is_some_and(|n| n > 0)
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[bloomery_gpu::arch::qwen3moe::ubatch::UBATCH_ENV])?;
        let a = parse_args()?;
        let mut ok = true;
        unplaced_card_is_the_largest(&mut ok);
        println!("serve granule: {} B, the unplaced card's", serve_granule()?);
        for (i, model) in a.models.iter().enumerate() {
            let dir = a.dir.join(i.to_string());
            std::fs::create_dir_all(&dir)?;
            println!("== {}", model.display());
            let seat = seat_of(model)?;
            if a.only_mixed {
                slots_rounds_run_one_pass(model, &dir, &SAMPLED_ROUNDS, &mut ok)?;
                continue;
            }
            if a.only_images {
                images(model, &dir, &mut ok)?;
                continue;
            }
            let Some((answers, default_n)) = served(model, seat, &dir, &mut ok)? else {
                continue;
            };
            let prompts: Vec<&[u32]> = answers.iter().map(|a| a.prompt.as_slice()).collect();
            let references = serve_client::generate_arms(
                "generate_qwen3moe",
                model,
                &prompts,
                N,
                default_n,
                &dir,
            )?;
            let mut agree_all = true;
            for (ans, reference) in answers.iter().zip(&references) {
                println!("cli tokens {reference:?}");
                agree_all &= agree(&ans.tokens, &ans.stop, reference);
            }
            // Both absent is the whole model on the card. Both present must
            // name the same host experts, card experts and context, and that
            // context is the server's loaded one. One of each is a different
            // split, which the ids are not held across.
            let server_split = routed_split(&seat_log(&dir.join("server.err"))?)?;
            let cli_text = std::fs::read_to_string(dir.join("gen.log"))?;
            let cli_split = routed_split(
                &record::Log::of(&cli_text, &[&record::PLAN38])
                    .named(dir.join("gen.log").display().to_string()),
            )?;
            println!(
                "cli split (host, card, ctx) {cli_split:?} against the server's {server_split:?} \
                 at context {default_n}"
            );
            let same_split = match (server_split, cli_split) {
                (None, None) => true,
                (Some((sh, sc, sx)), Some((ch, cc, cx))) => {
                    sh == ch && sc == cc && sx == cx && sx == default_n
                }
                _ => false,
            };
            check(
                &mut ok,
                "completion_ids_are_the_cli_ids",
                agree_all && same_split,
            );
            ctx_default(model, seat, &dir, default_n, &mut ok)?;
            plan_flag_stops_before_the_load(model, seat, &dir, &mut ok)?;
            ctx_past_the_trained_is_refused_by_name(model, &dir, &mut ok)?;
            match seat {
                qwen_seat::Family::Qwen35moe => {
                    // The Qwen3.6 file alone: the defaults on the 12 GB card it
                    // is served from.
                    defaults_take_a_long_prompt_on_a_12gb_card(model, &dir, &mut ok)?;
                }
                qwen_seat::Family::Qwen35 => {
                    // The dense file alone: its weights are over a 12 GB card's.
                    not_for_dense(
                        "defaults_take_a_long_prompt_on_a_12gb_card",
                        "its weights alone are over a 12 GB card's budget",
                    );
                    dense_below_its_floor_is_refused_by_name(model, &dir, &mut ok)?;
                }
                qwen_seat::Family::Qwen3moe => {}
            }
            whole_fit_counts_the_planes_granules(model, &dir, &mut ok)?;
            cache_refusals(model, &dir, &mut ok)?;
            // Both arms: the resident slots and the split they serve, on the
            // whole-card load and on a placed one, the late arrival and the
            // refused split.
            // PIN(2026-10-05): `swap_reprefills_the_parked_ids` (the qwen35moe
            // arm's turns, `serve::SwapEngine` under `Park::Ids`) left with
            // the seat's last swap engine: both bodies hold resident slots, so
            // the qwen35moe arm runs these slot clauses instead; `Park::Ids`
            // keeps its pure tests in `crates/serve/tests/slots`.
            slots_flow_together(model, &dir, &WHOLE_SLOTS, &mut ok)?;
            if routed(seat) {
                slots_flow_together(model, &dir, &PLACED_SLOTS, &mut ok)?;
                placed_slots_hold_the_plan(model, &dir, &mut ok)?;
            } else {
                not_for_dense(
                    "placed_slots_*",
                    "no routed expert to place: a budgeted plan is the floor refusal",
                );
            }
            slots_count_their_checkpoints(model, seat, &dir, &mut ok)?;
            slot_ctx_too_small_is_refused(model, &dir, &mut ok)?;
            slots_late_request_together_alone(model, &dir, &mut ok)?;
            slots_rounds_run_one_pass(model, &dir, &WHOLE_ROUNDS, &mut ok)?;
            if routed(seat) {
                slots_rounds_run_one_pass(model, &dir, &PLACED_ROUNDS, &mut ok)?;
            } else {
                not_for_dense(
                    "placed_slots_rounds_*",
                    "no host tier: a budgeted placement is the floor refusal",
                );
            }
            slots_rounds_run_one_pass(model, &dir, &SAMPLED_ROUNDS, &mut ok)?;
            if seat == qwen_seat::Family::Qwen3moe {
                // The qwen3moe arm alone: the placed prompt past the GEMM
                // walk.
                placed_answers_a_prompt_past_the_gemm_walk(model, &dir, &mut ok)?;
            }
            mtp_ids_match_the_plain_path(model, &dir, &mut ok)?;
            if a.images {
                images(model, &dir, &mut ok)?;
            }
        }
        if ok {
            println!("PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
