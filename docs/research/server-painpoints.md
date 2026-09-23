# Server and CLI pain points: llama-server, llama-cli, ik_llama.cpp, ollama, koboldcpp

Survey date: 2026-09-24. Question: people who run big MoE models at home (a 24 GB card, a
big-RAM host, CPU expert offload) — where do llama-server / llama-cli / ik's server / ollama
hurt them, and which of those hurts can bloomery remove **without changing the API shape**
it already serves (`crates/serve`: `/v1/chat/completions`, `/completion`, `/tokenize`,
`/detokenize`, `/apply-template`, `/v1/models`, `/health`, `/props`, `/slots`, `/metrics`).

## Method and limits

- Every row cites a primary source that was fetched for this survey: GitHub issues and PRs
  through the GitHub API (state and creation date as returned on 2026-09-24), Hacker News
  comments through the HN Algolia API, and the reference's own docs
  (`tools/server/README.md` and `tools/cli/README.md` of ggml-org/llama.cpp at commit
  `217f81c266`, 2026-09-22).
- "Recurrence" is the number of distinct sources **cited in that row** (a "#N above" reference to
  a source linked in an earlier row counts), not a search hit count. The shortlist's "Sources"
  column is the distinct union over its pain rows, so a source shared by two rows counts once.
- reddit (r/LocalLLaMA) could not be fetched: the search tool refuses the domain for this
  user agent. No reddit thread is cited; nothing is recalled from memory in its place.
- PRs are marked "PR". A PR is evidence that someone hit the pain hard enough to write code,
  not evidence of the pain's frequency.
- "Closed (stale)" means the stale bot closed it without a fix; the pain is still live.
- Sizes: XS < 1 day, S ≈ 1–3 days, M ≈ a round, L ≈ several rounds (lead's estimate class,
  not measured).
- "Compatible" means no default response shape changes: an extra field, an extra endpoint,
  an opt-in request field, or a flag. Returning 400 for a field we do not implement is
  counted as compatible (an honest error, not a new shape).

## What bloomery already has (referenced below)

| Short name | What it is today |
|---|---|
| placement plan | `crates/model/src/placement/workstation.rs`: the card/host split is computed from the card's and the host's memory before any upload (plans `a`, `gate`) |
| populate / lock | `BLOOMERY_HOST_POPULATE` (default on: `MADV_POPULATE_READ` of the host set at load), `BLOOMERY_HOST_LOCK=1` (`mlock` it), `BLOOMERY_CARD_DONTNEED` (drop page cache of uploaded card segments) |
| hot list | `BLOOMERY_HOT_LIST`: each routed layer keeps the router's most-used experts on the card instead of the id prefix |
| step stats | `BLOOMERY_STEP_STATS=1`: per step, host-tier `HybridStats` deltas, page faults, `vram_free`; a summary with `vram_free_load` and `vram_free_min` |
| lookup draft | `BLOOMERY_DRAFT=lookup`: n-gram draft, no draft model; `just gate-gpu-ds41-draft` pins the token line equal to the plain run; prints a `draft summary` line |
| one slot, reset per request | `crates/serve`: one engine behind a mutex, `reset` + full prefill every request; `/health`, `/props`, `/slots`, `/metrics` never take the engine mutex |
| own tokenizer | `crates/tokenizer`, gated id-identical against `llama-tokenize` on five corpora |

---

## 1. Startup and load

| # | Pain | Evidence (date, state) | Rec. | Reference today | What we could do | Size | Compat. |
|---|---|---|---|---|---|---|---|
| L1 | Cold load of a model near or above RAM size blocks for minutes, or thrashes, because the whole file is prefetched | [llama.cpp PR #29250](https://github.com/ggml-org/llama.cpp/pull/29250) (2026-09-21, open: `MADV_WILLNEED` blocks, 434 GB model on a 128 GB host); [ik PR #1634](https://github.com/ikawrakow/ik_llama.cpp/pull/1634) (2026-04-14, closed: `--defer-experts`, "load times from minutes down to ~10s"); [ik PR #2101](https://github.com/ikawrakow/ik_llama.cpp/pull/2101) (2026-07-09, closed: `--prefetch-experts`); [llama.cpp #10478](https://github.com/ggml-org/llama.cpp/issues/10478) (2024-11-25, closed: slow load with mmap) | 4 | mainline prefetches with `MADV_WILLNEED` by default; ik has opt-in defer/prefetch flags | Populate/lock already choose "resident up front" deliberately. Add what is missing: a load-phase line (seconds per phase: map, populate, upload, lock) and a `/health` 503 body with `"status":"loading"` plus bytes done/total while loading | S | yes |
| L2 | The one memory mode MoE offloaders need ("no mmap, but pinned") disappears or changes name between builds | [llama.cpp #26110](https://github.com/ggml-org/llama.cpp/issues/26110) (2026-07-25, closed: `--load-mode` refactor removed "no mmap + mlock"; swap filled, TG dropped); [llama.cpp PR #26934](https://github.com/ggml-org/llama.cpp/pull/26934) (2026-08-11, closed: migrate `--mmap/--no-mmap` to `--load-mode`); [llama.cpp #26025](https://github.com/ggml-org/llama.cpp/issues/26025) (2026-07-23, closed: fit + no-mmap incompatibility, "30+ tuning experiments") | 3 | `--load-mode none|mmap|mlock|dio` (CLI README) | One default (populate the host set; optional lock), verified at load and printed (the gate `gate-gpu-load-v41-lock` already checks residency). Keep the knob count at one and never rename it silently | XS | yes |
| L3 | Offload flag interplay: `-ngl`, `-ot` regex, `--n-cpu-moe`, `--fit` — and `--fit` gives up silently when an override is set, then OOMs later | [llama.cpp #27872](https://github.com/ggml-org/llama.cpp/issues/27872) (2026-08-28, open: RTX 3090 + DeepSeek-V4-Flash, fit "abort" logged as a warning, OOM at compute-buffer allocation); [llama.cpp #27501](https://github.com/ggml-org/llama.cpp/issues/27501) (2026-08-21, closed: n-cpu-moe in fit; a commenter replaced a hand regex with fit); [ik #2289](https://github.com/ikawrakow/ik_llama.cpp/issues/2289) (2026-08-10, closed: DeepSeek-V4 `--fit` crashes at load, manual `-ot` works); [llama.cpp PR #29343](https://github.com/ggml-org/llama.cpp/pull/29343) (2026-09-23, open: docs for `--n-cpu-moe` with several GPUs); [ollama #11772](https://github.com/ollama/ollama/issues/11772) (2025-08-07, open, 41 comments, 78 reactions: no MoE CPU offload at all); [HN comment](https://news.ycombinator.com/item?id=48968808) (2026-07-19: "a year and still no implementation" in ollama, `--n-cpu-moe` works in llama.cpp) | 6 | `--fit on` by default; `-ncmoe N` keeps the first N layers' experts on CPU; `-ot` regex for the rest | The placement plan is the answer: one flag names a plan, the plan is computed from free card memory and host RAM, and the load line prints it (per layer `n_l`, bytes per device) before any upload. Today plans are keyed to card names (`3090`, `A6000`); generalize to "the card's free bytes" | M | yes |
| L4 | "Which experts go on the card?" — people pin experts by index or regex, not by use | [llama.cpp PR #26414](https://github.com/ggml-org/llama.cpp/pull/26414) (2026-08-01, open: `--pin-hot-experts`); [llama.cpp PR #25932](https://github.com/ggml-org/llama.cpp/pull/25932) (2026-07-20, closed: earlier version); [llama.cpp PR #28414](https://github.com/ggml-org/llama.cpp/pull/28414) (2026-09-04, open: lookahead H2D prefetch of host experts); [ik PR #1813](https://github.com/ikawrakow/ik_llama.cpp/pull/1813) (2026-05-17, closed: per-byte MoE offload threshold) | 4 | no merged frequency-based placement in mainline | Hot list exists (4.5 ms/step faster, rig-log 2026-09-24, per BUILD.md). Expose: a documented command that builds the list from the user's own corpus, and `/props` field naming which ids sit on the card per layer | S | yes |
| L5 | OOM at load (or at first request) with a message far from the cause; the fitter forgets buffers | [llama.cpp #23472](https://github.com/ggml-org/llama.cpp/issues/23472) (2026-05-21, closed: fit ignores built-in MTP VRAM, OOM); [llama.cpp #18181](https://github.com/ggml-org/llama.cpp/issues/18181) (2025-12-18, closed: fit ignores vision encoders); [llama.cpp #21323](https://github.com/ggml-org/llama.cpp/issues/21323) (2026-04-02, closed: OOM loading Gemma 4 with CPU offload); [llama.cpp #28964](https://github.com/ggml-org/llama.cpp/issues/28964) (2026-09-15, open: "Invalid vector subscript" when devices report zero free VRAM); plus #27872 above | 5 | fit estimates, then allocates; failures surface as allocator errors | The plan is a budget: compare need vs. free per device before the first allocation and refuse with a table (weights, KV at `--ctx`, scratch, margin; need / free) and a nonzero exit | S | yes |
| L6 | Context size: silent truncation, and nobody can tell which setting produced the loaded context | [HN comment](https://news.ycombinator.com/item?id=43828277) (2025-04-29: ollama "defaults to a 2048 context length and silently truncates"); [HN comment](https://news.ycombinator.com/item?id=43866722) (2025-05-02: same, thinking models break); [ollama #18229](https://github.com/ollama/ollama/issues/18229) (2026-09-04, open: loaded context length has no visible source); [ollama #15848](https://github.com/ollama/ollama/issues/15848) (2026-04-27, open: undocumented default on Linux) | 4 | llama-server: `-c 0` = from model, `--fit-ctx` minimum 4096 (server README) | We already reject an over-long prompt with 400 `exceed_context_size_error`. Add: print `ctx` and its source (flag / default) on the load line and in `/props` | XS | yes |
| L7 | Reloading the same model (switch, idle unload) throws away the warm state | [llama.cpp #25066](https://github.com/ggml-org/llama.cpp/issues/25066) (2026-06-26, closed: router reloads files already loaded); [ollama #17247](https://github.com/ollama/ollama/issues/17247) (2026-07-18, open: warm prefill cache across unload/reload; ~7 s TTFT at ~31K tokens after an unload); [llama.cpp #24372](https://github.com/ggml-org/llama.cpp/issues/24372) (2026-06-09, closed: memory pressure on reload after sleep) | 3 | router mode, `--sleep-idle-seconds` (off by default) | One model per long-lived process, no idle unload (a 200 GB load is not something to repeat). Warm state across restarts = slot save to disk, see S3 | M | yes |

## 2. Serving behavior

| # | Pain | Evidence (date, state) | Rec. | Reference today | What we could do | Size | Compat. |
|---|---|---|---|---|---|---|---|
| S1 | Prefix reuse fails: the next turn re-processes the whole prompt | [llama.cpp #21831](https://github.com/ggml-org/llama.cpp/issues/21831) (2026-04-13, open, 52 comments, 30 reactions); [llama.cpp #22746](https://github.com/ggml-org/llama.cpp/issues/22746) (2026-05-06, closed, 126 comments); [llama.cpp #20225](https://github.com/ggml-org/llama.cpp/issues/20225) (2026-03-08, closed: every turn); [llama.cpp #19394](https://github.com/ggml-org/llama.cpp/issues/19394) (2026-02-06, closed); [llama.cpp #25567](https://github.com/ggml-org/llama.cpp/issues/25567) (2026-07-11, closed: deepseek4 logs an LCP match of 0.9 and re-processes anyway; later retest reuses); [ik #2033](https://github.com/ikawrakow/ik_llama.cpp/issues/2033) (2026-06-25, open: prompt comparison runs before checkpoint restore); [ik #1294](https://github.com/ikawrakow/ik_llama.cpp/issues/1294) (2026-02-21, closed: prompt caching doesn't work); [koboldcpp #2464](https://github.com/LostRuins/koboldcpp/issues/2464) (2026-09-13, open: DeepSeek V4 Flash classified as hybrid and forced into checkpoint mode) | 8 | `cache_prompt` default true, LCP slot selection, context checkpoints for SWA/hybrid, `--cache-ram` 8192 MiB | **We are worse today**: `cache_prompt` is accepted and ignored, every request prefills its whole prompt. Build: keep the slot's token history, cut the cache to the longest common prefix, prefill only the suffix. Needs an `Engine` method beyond `reset` (e.g. `truncate(n)`). V4.1's compressed KV and indexer state set the cut granularity; derive that before building | L | yes (`cache_prompt` honored, `timings.cache_n` real) |
| S2 | Cache reuse changes the output: same request, different answer depending on what came before | [server README](https://github.com/ggml-org/llama.cpp/blob/217f81c266/tools/server/README.md) (`cache_prompt`: "enabling this option can cause nondeterministic results"); [llama.cpp #28368](https://github.com/ggml-org/llama.cpp/issues/28368) (2026-09-04, open: warm vs fresh top-2 margin 0.131 vs 0.866 nats); [llama.cpp #19034](https://github.com/ggml-org/llama.cpp/issues/19034) (2026-01-22, open: identical requests differ with `cache_prompt`); [llama.cpp #9477](https://github.com/ggml-org/llama.cpp/issues/9477) (2024-09-14, closed: `logit_bias` persists across requests with the cache on) | 4 | documented as nondeterministic | Make it a gate, not a claim: greedy ids with `cache_prompt: true` equal those with it off, on a multi-turn fixture. This holds only if prefill logits and step logits agree for the same position; measure that before promising it | S (on top of S1) | yes |
| S3 | Slot save/restore does not restore, or restores wrong state | [llama.cpp #26676](https://github.com/ggml-org/llama.cpp/issues/26676) (2026-08-06, open: restore is a no-op, `cache_n=0`); [llama.cpp #25913](https://github.com/ggml-org/llama.cpp/issues/25913) (2026-07-20, open: hybrid checkpoints never persisted); [llama.cpp #28194](https://github.com/ggml-org/llama.cpp/issues/28194) (2026-09-01, open: no reuse after restore on hybrid/SWA); [llama.cpp #28619](https://github.com/ggml-org/llama.cpp/issues/28619) (2026-09-08, open: draft context never persisted); [llama.cpp #27068](https://github.com/ggml-org/llama.cpp/issues/27068) (2026-08-14, open: failed restore corrupts K/V) | 5 | `POST /slots/{id}?action=save|restore` with `--slot-save-path` | Same endpoint and body. One owner of "slot state" (KV + compressed-KV/indexer state + token history + draft history), saved and restored as one unit; gate: restore then continue = never-saved run, token for token | M | yes |
| S4 | State leaks between requests or clients | [llama.cpp #27148](https://github.com/ggml-org/llama.cpp/issues/27148) (2026-08-15, open: RAM prompt cache restores an unrelated conversation); [llama.cpp #25992](https://github.com/ggml-org/llama.cpp/issues/25992) (2026-07-22, open: `-np 4 --kv-unified` returns other requests' responses); [llama.cpp #27422](https://github.com/ggml-org/llama.cpp/issues/27422) (2026-08-19, open: system prompt leaking between consumers); [ik #2451](https://github.com/ikawrakow/ik_llama.cpp/issues/2451) (2026-09-15, open: one `ignore_eos` request makes later requests unable to end); [llama.cpp #26207](https://github.com/ggml-org/llama.cpp/issues/26207) (2026-07-28, open: prompt cache reused across different `lora`) | 5 | shared cross-slot caches on by default (`--cache-ram`, `--cache-idle-slots`) | Structurally absent today (one slot, reset per request, sampling params per request). Keep it absent when S1 lands: reuse keyed by exact token prefix only, per-request state never outlives the request; gate: an `ignore_eos` request followed by a plain one ends normally | XS (keep) | yes |
| S5 | `/health`, `/slots`, `/metrics` stop answering during a long prefill | [ik #1210](https://github.com/ikawrakow/ik_llama.cpp/issues/1210) (2026-01-31, open: unresponsive for "30-60+ minutes" of CPU prefill); [llama.cpp #24866](https://github.com/ggml-org/llama.cpp/issues/24866) (2026-06-21, closed 2026-08-14: `/metrics` blocked during decode, fixed in mainline); [llama.cpp #27388](https://github.com/ggml-org/llama.cpp/issues/27388) (2026-08-19, open: `/slots` hangs); [llama.cpp #29104](https://github.com/ggml-org/llama.cpp/issues/29104) (2026-09-18, open: server stops when `/metrics` is scraped) | 4 | mainline fixed `/metrics`; ik still blocks | **Already answered** for `/health`, `/props`, `/slots`, `/metrics`. Remaining: `/tokenize` and `/detokenize` wait for the generation mutex though `encode`/`decode` are `&self`; move the tokenizer out from behind the engine lock | XS | yes |
| S6 | No prompt-processing progress, or a wrong fraction | [llama.cpp #6586](https://github.com/ggml-org/llama.cpp/issues/6586) (2024-04-10, closed); [llama.cpp #14685](https://github.com/ggml-org/llama.cpp/issues/14685) (2025-07-15, closed); [llama.cpp #15432](https://github.com/ggml-org/llama.cpp/issues/15432) (2025-08-19, closed: fraction wrong with cached prompts); [llama.cpp #24822](https://github.com/ggml-org/llama.cpp/issues/24822) (2026-06-19, open: loading progress over SSE) | 4 | `return_progress` streams `prompt_progress {total, cache, processed, time_ms}` | We send `prompt_progress` once, complete, after prefill. Send it per prefill chunk (needs a progress callback from `Engine::prefill`); for a 30K prompt on the host tier this is the difference between "hung" and "working" | S | yes |
| S7 | Throughput gauges read 0 almost always | [llama.cpp #27436](https://github.com/ggml-org/llama.cpp/issues/27436) (2026-08-20, open); [llama.cpp #27364](https://github.com/ggml-org/llama.cpp/issues/27364) (2026-08-19, open) | 2 | gauge = tokens/time in a bucket reset at each scrape | **We copied this behavior** (`crates/serve/src/api.rs` metrics resets the bucket on read). Report the last completed request's rate, or a fixed window, instead; same metric names | XS | yes |
| S8 | Missing metrics: KV use, speculative counters, per-device memory | [llama.cpp #23632](https://github.com/ggml-org/llama.cpp/issues/23632) (2026-05-25, closed: expose KV cache use); [llama.cpp #26516](https://github.com/ggml-org/llama.cpp/issues/26516) (2026-08-03, open — but the [server README](https://github.com/ggml-org/llama.cpp/blob/217f81c266/tools/server/README.md) at `217f81c266` already lists `llamacpp:spec_decode_num_draft_tokens_total`, `…_accepted_tokens_total`, `…_drafts_total`); [llama.cpp #26129](https://github.com/ggml-org/llama.cpp/issues/26129) (2026-07-26, closed (stale) 2026-09-23: per-device weights/context/compute bytes) | 4 | spec counters yes; per-device memory no (log lines only) | Our `kv_cache_usage_ratio` is hard-coded `0`: make it `n_past / ctx_max`. Add the three `spec_decode_*` names as the README spells them, and `memory_{model,context,compute,free}_bytes{device=…}` as #26129 proposes, filled from the placement plan and `cuMemGetInfo` | S | yes |
| S9 | Unsupported request features are accepted with HTTP 200 and ignored | [llama.cpp #24097](https://github.com/ggml-org/llama.cpp/issues/24097) (2026-06-04, closed (stale): `json_schema` not enforced, 200); [llama.cpp #24096](https://github.com/ggml-org/llama.cpp/issues/24096) (2026-06-04, closed: `json_object` returns free text); [llama.cpp #27129](https://github.com/ggml-org/llama.cpp/issues/27129) (2026-08-15, open: `tools` silently dropped when the template has no tool support); [llama.cpp #10732](https://github.com/ggml-org/llama.cpp/issues/10732) (2024-12-09, open: `json_object` works, `json_schema` does not) | 4 | varies by template and parser | **We do this too**: `n_probs` gets 400, but `response_format`, `grammar`, `json_schema`, `logprobs` are ignored. One policy: a field we cannot honor is a 400 naming the field, until it is implemented | XS | yes |
| S10 | Reasoning text misfiled: thinking lands in `content`, the answer lands in `reasoning_content`, or a tool call lands in reasoning | [llama.cpp #27134](https://github.com/ggml-org/llama.cpp/issues/27134) (2026-08-15, open: reply misfiled into `reasoning_content` when the generation prompt ends with `</think>`); [llama.cpp #22684](https://github.com/ggml-org/llama.cpp/issues/22684) (2026-05-04, closed: tool call emitted in `reasoning_content`, Copilot client); [llama.cpp #20717](https://github.com/ggml-org/llama.cpp/issues/20717) (2026-03-18, closed: DeepSeek V3.2 returns no `reasoning_content`); [llama.cpp #22577](https://github.com/ggml-org/llama.cpp/issues/22577) (2026-05-01, closed); [ollama #18009](https://github.com/ollama/ollama/issues/18009) (2026-08-26, open: thinking parser drops trailing content at end of stream) | 5 | `--reasoning-format none|deepseek|…`, general autoparser | No split today. V4.1's template emits `</think>` at the end of the generation prompt when thinking is off — exactly #27134's case. Build one parser for the one template we serve, with `reasoning_format` (llama-server's name) and `none` as opt-out; gate on fixtures for both thinking modes | S | yes |
| S11 | Tool calls: DeepSeek DSML parsing, argument encoding, ids | [llama.cpp #27613](https://github.com/ggml-org/llama.cpp/issues/27613) (2026-08-23, open: DeepSeek-V4-Flash stops mid-tag when reasoning contains DSML closing fragments); [llama.cpp #20198](https://github.com/ggml-org/llama.cpp/issues/20198) (2026-03-07, closed: `arguments` returned as object, OpenAI SDK crashes); [llama.cpp #27966](https://github.com/ggml-org/llama.cpp/issues/27966) (2026-08-29, open: streamed parallel tool calls reuse an id); [llama.cpp #26359](https://github.com/ggml-org/llama.cpp/issues/26359) (2026-07-31, closed: tool-call id rejected by the model's own template) | 4 | PEG parser per format (`peg-native`) | We pass `tools` into the template, so V4.1 will emit DSML, but nothing parses it: raw DSML reaches `content`. Build a DSML parser for V4.1 into `tool_calls` (`arguments` a JSON string, unique ids), and parse only outside the reasoning span (the #27613 trap) | M | yes |
| S12 | `logprobs` / `n_probs` gaps | [llama.cpp #27174](https://github.com/ggml-org/llama.cpp/issues/27174) (2026-08-16, open: no prompt logprobs with `echo`, lm-eval scores at chance); [llama.cpp #27972](https://github.com/ggml-org/llama.cpp/issues/27972) (2026-08-29, open: every logprob 0.0 with speculative decoding); [llama.cpp #28478](https://github.com/ggml-org/llama.cpp/issues/28478) (2026-09-06, open: streaming + tools rejects `logprobs`, accepts `n_probs`); [llama.cpp #16329](https://github.com/ggml-org/llama.cpp/issues/16329) (2025-09-29, closed: logprobs cut speed 3×) | 4 | `n_probs`, `post_sampling_probs`, OpenAI `logprobs` on generated tokens | `n_probs` is refused today. Top-N on generated tokens is cheap from the logits `Engine::next` already writes; do it when requested only (the bind round wants the logits copy optional for greedy — keep it on for `n_probs`). With the lookup draft, logprobs must come from the verifying pass, not be faked (#27972) | S | yes |
| S13 | Generation not bounded: `max_tokens` ignored or a runaway request holds the only slot | [ollama #18575](https://github.com/ollama/ollama/issues/18575) (2026-09-21, open: `/v1/chat/completions` ignores `max_tokens`, one request blocks all); [ollama #2963](https://github.com/ollama/ollama/issues/2963) (2024-03-06, open: no way to pass `options` through the OpenAI endpoints); plus ik #2451 above | 3 | llama-server honors `max_tokens`/`n_predict`; `-1` = until context | We honor `n_predict`/`max_tokens`/`max_completion_tokens`; `-1` runs to the context. With one slot, add a server flag `--n-predict-max` (unset = today's behavior) and cancel on client disconnect | XS | yes |
| S14 | DeepSeek V4 tokenizer crashes on long letter runs | [llama.cpp #26965](https://github.com/ggml-org/llama.cpp/issues/26965) (2026-08-12, open: stack overflow on a long ASCII run in a tool result) | 1 | recursive split in the reference tokenizer | Our tokenizer is separate code: add the 128 Ki-letter run to the tokenizer gate corpus and check it neither recurses nor goes quadratic | XS | yes |

## 3. Speculative decoding and MoE specifics

| # | Pain | Evidence (date, state) | Rec. | Reference today | What we could do | Size | Compat. |
|---|---|---|---|---|---|---|---|
| P1 | Draft setup is hard and often does not speed anything up | [llama.cpp #28939](https://github.com/ggml-org/llama.cpp/issues/28939) (2026-09-15, open: DSpark/DFlash on DeepSeek-V4-Flash, same or lower speed); [llama.cpp #28255](https://github.com/ggml-org/llama.cpp/issues/28255) (2026-09-02, open: no docs to set a draft by env var); [llama.cpp #27839](https://github.com/ggml-org/llama.cpp/issues/27839) (2026-08-28, open: combined spec types fail at init); [llama.cpp #26337](https://github.com/ggml-org/llama.cpp/issues/26337) (2026-07-30, open: model fails to load with its DSpark draft) | 4 | many spec types (`draft-mtp`, `draft-dspark`, `draft-dflash`, n-gram), each with its own flags | Lookup draft needs no second model and is gated token-identical. Expose it as a server flag, off by default, and report its effect per request (P2) so users see whether it helps on their text | S | yes |
| P2 | Acceptance collapses and nobody can see why | [llama.cpp #26750](https://github.com/ggml-org/llama.cpp/issues/26750) (2026-08-08, open: 40.7 % on CUDA vs ~92 % on Vulkan); [llama.cpp #27151](https://github.com/ggml-org/llama.cpp/issues/27151) (2026-08-15, open: 1 of 633 accepted); [llama.cpp #29168](https://github.com/ggml-org/llama.cpp/issues/29168) (2026-09-20, open: a fusion breaks exactness, acceptance 0.82 → 0.48, MTP becomes a slowdown); [llama.cpp #27572](https://github.com/ggml-org/llama.cpp/issues/27572) (2026-08-22, open: acceptance 0 under `-np N`) | 4 | per-request `draft_n`, `draft_n_accepted`, `draft_accept_ratio` in `timings` (per #26516's body: added by PR #12603); `/metrics` totals (README) | Our `draft summary` line already has the counts. Put them in `timings` under those three names and in `/metrics` under the README's `spec_decode_*` names | XS | yes |
| P3 | Speculative decoding changes results | #27972 (logprob 0.0 with spec), #29168 (exactness broken) above; [llama.cpp #28049](https://github.com/ggml-org/llama.cpp/issues/28049) (2026-08-30, open: accepted tokens kept past end-of-generation) | 3 | no exactness gate visible to users | **Already answered** by `just gate-gpu-ds41-draft` (token line equal to the plain run). Extend the gate to the server path (stream with and without draft, same ids) and to end-of-generation inside a draft | XS | yes |
| P4 | "Why is my tok/s low?" — regressions with CPU experts found only by long bisects, no per-step breakdown | [llama.cpp #26025](https://github.com/ggml-org/llama.cpp/issues/26025) (2026-07-23, closed: ~35 % regression on 8 GB VRAM, "30+ tuning experiments"); [ik #2199](https://github.com/ikawrakow/ik_llama.cpp/issues/2199) (2026-07-28, closed: TG −36 % with `n-cpu-moe=99`); [ik #1699](https://github.com/ikawrakow/ik_llama.cpp/issues/1699) (2026-04-27, closed: slower than mainline with CPU-MoE); [ik #1765](https://github.com/ikawrakow/ik_llama.cpp/issues/1765) (2026-05-09, closed: CPU-expert offload much slower than mainline); [llama.cpp #26876](https://github.com/ggml-org/llama.cpp/issues/26876) (2026-08-11, open: partial MoE offload 2× slower than CPU-only on an iGPU) | 5 | `timings` gives prompt/predicted totals only; host-vs-card split exists nowhere | Step stats already split host tier vs card per step. Expose them opt-in per request (an extra `timings` object, only when asked, since instrumented steps are 2.5–3 ms slower per BUILD.md) and as a CLI flag | S | yes |
| P5 | Streaming stalls for seconds every few lines | [ik #464](https://github.com/ikawrakow/ik_llama.cpp/issues/464) (2025-05-27, closed, 43 comments: 5–8 s blocks, Qwen3-235B with `-ot`); [llama.cpp #27388](https://github.com/ggml-org/llama.cpp/issues/27388) (2026-08-19, open: generation stalls mid-decode) | 2 | no per-step timing to show where the stall is | Step stats already carry page faults per step; populate/lock remove the fault source. Report a per-request `max step ms` and the page-fault count so a stall is visible in the response, not only by eye | XS | yes |
| P6 | Memory creeps until OOM on DeepSeek V4 serving | [llama.cpp #27155](https://github.com/ggml-org/llama.cpp/issues/27155) (2026-08-16, open: DSpark draft KV grows ~10 MB per cycle); [llama.cpp #25566](https://github.com/ggml-org/llama.cpp/issues/25566) (2026-07-11, closed: deepseek4 slot KV never reclaimed, "Context size has been exceeded"); [llama.cpp #28933](https://github.com/ggml-org/llama.cpp/issues/28933) (2026-09-15, open: host RSS grows during chat) | 3 | none built in | Step stats read `vram_free` per step. Make the M2 30-minute soak a gate: `vram_free_min` after N requests equals the value after the first, and RSS is flat | S | yes |

## 4. CLI ergonomics

| # | Pain | Evidence (date, state) | Rec. | Reference today | What we could do | Size | Compat. |
|---|---|---|---|---|---|---|---|
| C1 | Non-interactive use hangs: `-no-cnv` ignored, the process waits at `>` forever | [llama.cpp #27214](https://github.com/ggml-org/llama.cpp/issues/27214) (2026-08-17, open) | 1 | new `llama-cli` is conversation-only; `-st` exits after one turn; `llama-completion` keeps the old mode (CLI README) | `bloomery-chat` (in flight): when stdin is not a TTY or `-p` is given without `-i`, answer once and exit 0; nonzero exit on error | XS | yes |
| C2 | Multi-turn breaks because the CLI diffs re-rendered prompts and the template strips old reasoning | [llama.cpp #13404](https://github.com/ggml-org/llama.cpp/issues/13404) (2025-05-09, open) | 1 | incremental diff of template renders | Render the whole conversation each turn with the server's template code and rely on prefix reuse (S1) for speed; the CLI and server then share one path and one gate | S | yes |
| C3 | Multiline input mangled | [llama.cpp #21464](https://github.com/ggml-org/llama.cpp/issues/21464) (2026-04-05, closed: `\n` stripped in multiline input) | 1 | `-mli` flag | Read stdin to EOF when piped; in a TTY, a documented end-of-message key | XS | yes |
| C4 | No session persistence in the chat CLI; the older one's session file had a replay bug | [CLI README](https://github.com/ggml-org/llama.cpp/blob/217f81c266/tools/cli/README.md) at `217f81c266` lists `-sys`, `-st`, `-mli`, no session or prompt-cache flag; [llama.cpp #23400](https://github.com/ggml-org/llama.cpp/issues/23400) (2026-05-20, closed: `llama-completion --prompt-cache` replays the last token at the wrong position) | 2 | sessions only in `llama-completion` | `--session <file>`: the conversation as JSON plus the slot state from S3; resume = restore and continue, gated like S3 | M | yes |

## 5. Observability and diagnostics

| # | Pain | Evidence (date, state) | Rec. | Reference today | What we could do | Size | Compat. |
|---|---|---|---|---|---|---|---|
| O1 | Memory per device is only in log lines; no API | [llama.cpp #26129](https://github.com/ggml-org/llama.cpp/issues/26129) (2026-07-26, closed (stale) 2026-09-23); #23632 above | 2 | log lines | Same as S8's memory rows, plus a JSON form in `/props` (an extra key); source: placement plan + `vram_free_load` | XS | yes |
| O2 | Crashes with no useful context: silent exits, misleading HTTP errors, orphaned processes | [llama.cpp #20509](https://github.com/ggml-org/llama.cpp/issues/20509) (2026-03-13, closed: server down without clear log, later identified as OOM); [ik #2411](https://github.com/ikawrakow/ik_llama.cpp/issues/2411) (2026-09-04, open: DeepSeek-V4-Flash assert returns 500 "Input prompt is too big", a child keeps the port); [ik #2344](https://github.com/ikawrakow/ik_llama.cpp/issues/2344) (2026-08-20, closed: all-NaN logits abort, the server's handler "can never run"); [llama.cpp #26609](https://github.com/ggml-org/llama.cpp/issues/26609) (2026-08-05, open: illegal memory access with partial expert offload); [llama.cpp #27330](https://github.com/ggml-org/llama.cpp/issues/27330) (2026-08-18, open: CUDA graphs hang the GPU channel, Xid 8) | 5 | abort or generic 500 | On an `EngineError`: a 500 whose message is the engine's (not a guess), `/health` → 503 with that reason, one stderr block with the card name, last step's stats, and position; then exit nonzero so a supervisor restarts it. A NaN in logits becomes an `EngineError` at the step, not an abort in the sampler | S | yes |
| O3 | Errors hidden by the verbosity filter | [llama.cpp #28107](https://github.com/ggml-org/llama.cpp/issues/28107) (2026-08-31, open: llama-bench drops the Metal OOM line unless `-v`) | 1 | level filter applies to errors too | Errors always reach stderr; data (tokens, JSON) only on stdout | XS | yes |
| O4 | Log noise or misleading log lines | [llama.cpp #20309](https://github.com/ggml-org/llama.cpp/issues/20309) (2026-03-09, closed: log spam on `/completion`); [llama.cpp #22127](https://github.com/ggml-org/llama.cpp/issues/22127) (2026-04-19, closed: "prompt cache is enabled" logged with `--cache-ram 0`) | 2 | `--log-verbosity` levels | One line per request at default level (method, prompt n, predicted n, ms), everything else behind a flag | XS | yes |

## 6. Client compatibility

| # | Pain | Evidence (date, state) | Rec. | Reference today | What we could do | Size | Compat. |
|---|---|---|---|---|---|---|---|
| K1 | Usage fields in a slightly different place or missing sub-objects; strict clients refuse to parse | [llama.cpp #15443](https://github.com/ggml-org/llama.cpp/issues/15443) (2025-08-20, closed: `usage` in the finish chunk, OpenAI sends a final chunk with empty `choices`); [llama.cpp #22659](https://github.com/ggml-org/llama.cpp/issues/22659) (2026-05-03, closed: `output_tokens_details` missing, Rust `rig` refused the JSON); [llama.cpp #27971](https://github.com/ggml-org/llama.cpp/issues/27971) (2026-08-29, closed: reasoning token count in usage) | 3 | usage with `timings` in the last chunk | We match llama-server's placement today. Where llama-server and OpenAI differ, follow llama-server (the compatibility target) and fill every sub-object OpenAI marks required, zero-valued | XS | yes |
| K2 | Non-standard fields in the default response break strict clients | [ollama #16853](https://github.com/ollama/ollama/issues/16853) (2026-06-22, closed: extra `reasoning` field breaks Warp); [llama.cpp #29091](https://github.com/ggml-org/llama.cpp/issues/29091) (2026-09-18, closed: `vocab_type` serialized as bool) | 2 | — | Rule for every row above: our additions go only under names llama-server already uses, or behind an opt-in request field | XS (policy) | yes |
| K3 | `/v1/models` `created` changes every call; clients that cache by it break | [llama.cpp #26169](https://github.com/ggml-org/llama.cpp/issues/26169) (2026-07-27, closed); [llama.cpp PR #27719](https://github.com/ggml-org/llama.cpp/pull/27719) (2026-08-25, open: make it stable) | 2 | fix in review | **We have this bug**: `created` is `unix_now()` per call. Use the process start time (already stored) | XS | yes |
| K4 | `/v1/responses` missing; newer SDKs and agents need it | [llama.cpp #19138](https://github.com/ggml-org/llama.cpp/issues/19138) (2026-01-27, open, 56 reactions); [llama.cpp #14702](https://github.com/ggml-org/llama.cpp/issues/14702) (2025-07-15, open, 41 comments); [llama.cpp #24115](https://github.com/ggml-org/llama.cpp/issues/24115) (2026-06-04, closed: Responses shim throws on a missing field, breaks Copilot CLI); [ollama #17673](https://github.com/ollama/ollama/issues/17673) (2026-08-11, open) | 4 | llama-server serves `/responses` (#22659 uses it) | An extra endpoint over the same generation loop, after S10/S11 (Responses carries reasoning and tool items) | M | yes |
| K5 | Coding agents (Claude Code, Copilot, Cline) fail against local servers | [llama.cpp #27107](https://github.com/ggml-org/llama.cpp/issues/27107) (2026-08-15, closed); [llama.cpp #27893](https://github.com/ggml-org/llama.cpp/issues/27893) (2026-08-28, open: Anthropic endpoint, server sampling args vs client's); [llama.cpp #27612](https://github.com/ggml-org/llama.cpp/issues/27612) (2026-08-23, open: Cline tools partly work); plus #22684 above | 4 | llama-server serves an Anthropic Messages endpoint (server README) | Out of scope until S1, S10, S11 exist: agents send 20–60K-token prompts every turn, so without prefix reuse a host-tier engine is unusable for them regardless of API shape | L | yes |

---

## Ranked shortlist — what to build first

"Expose" = the engine already has it, the server does not show it. "Build" = new code.

| Rank | Item | Pain rows | Sources | Kind | Size |
|---|---|---|---|---|---|
| 1 | Prefix reuse (`cache_prompt` honored) | S1, S2, C2 | 13 | Build | L |
| 2 | Honest refusals and small correctness fixes in `crates/serve` | S9, S7, K3, S5 | 12 | Build (small) | XS |
| 3 | V4.1 reasoning split and DSML tool calls | S10, S11 | 9 | Build | M |
| 4 | Load plan and memory report | L3, L5, O1, S8 | 14 | Expose | S |
| 5 | Per-request host/card attribution | P4, P5 | 7 | Expose | S |
| 6 | Lookup draft on the server, with counters | P1, P2, P3 | 10 | Expose | S |
| 7 | Crash context and error surfacing | O2, O3 | 6 | Build | S |
| 8 | Slot save/restore as one owned unit | S3, L7, C4 | 10 | Build | M |
| 9 | Load-phase progress and prefill progress | L1, S6 | 8 | Build | S |
| 10 | Soak gate for memory creep | P6 | 3 | Expose | S |

**1. Prefix reuse.** The most-reported pain in every reference, and bloomery is behind here
today: every request resets and prefills its whole prompt. On a host-tier engine a 30K-token
agent turn is minutes, so this gates every chat and agent use. Feature: the slot keeps its
token history; a new request computes the longest common prefix, the engine cuts its cache
to that length and prefills only the suffix; `timings.cache_n` reports the reused count.
Needs one new `Engine` method (a cut to position `n`); how finely V4.1's compressed KV and
indexer state can be cut is the design question and should be derived before code.
Compatible: `cache_prompt` already exists in the request shape; honoring it changes values,
not shape. Gate: on a mock engine and on `gate-gpu-ds41-serve`, a two- and a five-turn chat
with `cache_prompt: true` returns greedy ids identical to `cache_prompt: false`, and turn N's
`prompt_n` equals its new-suffix length. The first half of that gate is the differentiator
llama-server's own README disclaims (S2); it holds only if prefill and step logits agree per
position, which must be measured first.

**2. Honest refusals and small fixes.** Four XS changes in `crates/serve`, one policy: a
field we cannot honor (`response_format`, `grammar`, `json_schema`, `logprobs`, `tools`
until S11 parses them) returns 400 naming it, like `n_probs` does now; the throughput
gauges report the last request's rate instead of resetting on scrape; `/v1/models`
`created` uses the start time; `/tokenize` stops waiting on the generation mutex.
Compatible: errors and value fixes, no new shape. Gate: mock-engine tests, one per field
and one per fix, each FAIL-first on today's code.

**3. V4.1 reasoning and DSML tool calls.** V4.1's template ends the generation prompt with
`</think>` when thinking is off (the #27134 trap) and emits DSML for tools (the #27613
trap); today both reach `content` raw. Feature: one parser for the one template we serve,
producing `reasoning_content` (llama-server's field, `reasoning_format: none` as opt-out)
and `tool_calls` with `arguments` as a JSON string and unique ids; DSML inside the
reasoning span is text, not a call. Compatible: both fields are llama-server's. Gate:
fixture streams (thinking on/off, a tool call, DSML fragments inside reasoning) parsed to
expected JSON; streamed and non-streamed results identical.

**4. Load plan and memory report (expose).** The placement plan already decides every
byte's device before upload, and `vram_free_load` is measured. Feature: print the plan
(per layer `n_l`, bytes per device, ctx and its source) before upload; refuse a plan that
does not fit with a need/free table and a nonzero exit; publish it in `/props` as an
extra key and in `/metrics` as `memory_*_bytes{device=…}` (the names #26129 proposed) plus
a real `kv_cache_usage_ratio`. Compatible: extra keys and metric names. Gate: the reported
per-device bytes sum to the plan's totals; `/props` still carries every llama-server key
(existing gate-serve shape check).

**5. Per-request host/card attribution (expose).** Nobody in the references can say where a
slow step's time went; users bisect for days (P4). Step stats already split host tier and
card per step, with page faults. Feature: an opt-in request field that adds a `timings`
sub-object (host ms, card ms, wait ms, faults, max step ms), off by default because it costs
2.5–3 ms per step. Compatible: opt-in field, extra sub-object. Gate: the sub-object's per-step
sums match the CLI's `stat step` lines for the same run; absent when not requested.

**6. Lookup draft on the server (expose).** The draft is already token-identical by gate and
needs no second model, which removes P1's setup class entirely. Feature: a server flag (off
by default) and `draft_n`, `draft_n_accepted`, `draft_accept_ratio` in `timings` plus
`spec_decode_*` totals in `/metrics`, all under llama-server's names. Compatible: flag and
existing field names. Gate: `gate-gpu-ds41-serve` streams with and without the flag return
identical ids; counters equal the `draft summary` line.

**7. Crash context.** Feature: an `EngineError` becomes a 500 with the engine's message,
`/health` turns 503 with the reason, stderr gets one block (card, last step's stats,
position), and the process exits nonzero instead of hanging with the port held (#2411's
orphan). NaN logits are detected at the step. Compatible: error bodies only. Gate: a mock
engine that fails at step k yields the 500, the 503, and the exit code.

**8. Slot save/restore.** After 1: the same `/slots/{id}?action=save|restore` body, with
one owner of "slot state" (KV, compressed-KV and indexer state, token history, draft
history). Also backs the CLI's `--session`. Gate: save at turn k, restart, restore, continue
= an uninterrupted run, token for token (the check #26676 shows the reference lacks).

**9. Progress.** `/health` 503 `loading` with bytes done/total during a 200 GB load, and
`prompt_progress` per prefill chunk instead of once at the end. Needs a progress callback
from `prefill`. Gate: mock engine with a chunked prefill emits monotonically increasing
`processed` ending at `total`.

**10. Soak gate (expose).** Turn M2's 30-minute soak into a gate on the step stats already
read: `vram_free_min` and RSS flat across N requests. It closes the DeepSeek V4 leak class
(P6) as a contract rather than a hope.

### Which items our architecture already answers

| Architecture piece | Rows it answers | Work left |
|---|---|---|
| one slot, reset per request | S4, S5 (except tokenize) | keep it true when S1 lands |
| placement plan | L3, L5 | print, refuse with a table, generalize past card names |
| populate / lock | L1, L2, P5 | report phases and faults |
| hot list | L4 | a documented build command, `/props` listing |
| step stats | P4, P5, P6, O1 | an opt-in response field, a soak gate |
| lookup draft + its gate | P1, P2, P3 | server flag, counters under llama-server's names |
| own tokenizer + gate | S14 | one more corpus line |

Not in the shortlist on purpose: `/v1/responses` and agent clients (K4, K5) come after 1 and
3, because without prefix reuse an agent turn on a host-tier engine is not usable whatever
the endpoint; `/v1/completions` echo logprobs (S12's lm-eval case) needs prefill logits and
waits for an evaluation use.
