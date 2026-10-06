# Serving

How `bloomery-serve` behaves: the llama-server API and flags it shares, the seats' own behaviour, and where it ends
today. Install and the one-command start are in the [README](../README.md).

## llama-server compatibility

Your OpenAI client works unchanged. bloomery speaks llama-server's HTTP API and reads the same files:

- **The API**: `/v1/chat/completions`, `/completion`, streaming, tool calls, `reasoning_content`,
  `cache_prompt`, `/tokenize`, `/detokenize`, `/slots`, `/metrics`, `/v1/models`, `/props`. OpenAI SDKs, curl,
  and llama-server tutorials apply as they are.
- **The Anthropic Messages API**: `POST /v1/messages` and `/v1/messages/count_tokens`, as llama-server serves
  them. Point Claude Code or the Anthropic SDK at `ANTHROPIC_BASE_URL=http://localhost:8080`.
- **Tool calls on every generative model**: DeepSeek-V4.1-Flash (DSML), GLM-5.3-Flash (its XML), Qwen3-30B
  (Hermes JSON), Qwen3.6-35B and Qwen3.8-Flash-Next (the `<function=…><parameter=…>` markup) parse into
  `tool_calls` and `tool_use` blocks, so agent clients work against each of them.
- **`timings`** carry llama-server's fields and one of ours, `cache_ms`: the prompt cache's work before the
  prompt (the slot's state saved, a cached state put back, the cut), which `prompt_ms` does not count.
- **`reasoning_budget`**, per request on `/v1/chat/completions` and `/completion`: llama-server's
  `--reasoning-budget` — the think span's budget in generated ids taken while the span is open, `0` closing it at
  once, `-1` or absent unrestricted, and silently ignored when the template already closed the span.
- **The files**: the GGUF uploads as downloaded, the same quantizations, the chat template read from the file,
  and a tokenizer bit-identical to `llama-tokenize`.
- **The flags**: `-m`/`--model-file`, `--hf <repo>[:<quant>]` (download, resume, sha256 check, never fetched
  twice; it also fetches the model's MTP draft when the repo publishes it, as unsloth's Qwen3.8-Flash-Next GGUFs
  do under `MTP/`, landing it beside the set, where the run picks it up), `--parallel/-np`, `--queue-depth`,
  `--cache-ram`, `--ctx/--ctx-size` (the qwen3, qwen38 and glm seats default it to what the card's free memory
  fits — the trained context capped to it on the qwen3 seat's whole-card loads, the largest context that keeps the
  card's experts on the qwen38 seat, the placed plan's margin on the glm seat), `--cache-type-k f16|q8_0` (the qwen3
  seats; llama-server's spelling, also the `BLOOMERY_QWEN3_KV` lever — q8_0 halves the KV bytes a position, so the
  auto context nearly doubles on the same card; `--cache-type-v` does not exist, both planes quantize together),
  `--host/--port`, `--alias`.
- **No placement flag needed**: leave `--place` out on every model and the load picks for you — the largest
  visible card, a split onto the CPU when the file does not fit it, and on GLM-5.3 the second card as an expert
  tier when its plan puts enough experts there to pay for it (the GLM row of the
  [A6000 + 3090 table](../README.md#numbers)). The plan line names what was picked.
- **Two-card serving like `-ts`**: `--place a` for the largest visible card, `--place bp` to add the next one
  as an expert tier, or a list (`0+1`) by CUDA index (not on the qwen38 seat, which takes `a`, `gate` or `bp`).
- **Clef-Flash's `/v1/systemone`** follows llama.cpp's decision server wire (`model` optional, `/v1/models`,
  501 for images and video, upstream's `confidence` formula).

Where it ends today: one model a server, sm_86+ GPUs, and the seats' own defaults where llama-server has none
(see [Limits](#limits)).

## The seats

The decide seat answers SystemOne requests, one at a time, text states only:

```sh
curl -s http://127.0.0.1:8080/v1/systemone -d '{"state": "User: what is the weather in Seoul tomorrow? Tools available: web_search, calculator.",
  "questions": {"tool": {"type": "choice", "instructions": "Which tool should the agent call next?",
    "criteria": {"web_search": "Look something up online", "calculator": "Do arithmetic", "none": "Answer directly"}}}}'
```

**Concurrent requests.** Every generative seat takes `--parallel N` (unset, 2). The N slots are resident sequences
inside the one model, switched by pointer exchange: nothing parks, and each round advances every busy slot by one
token or one drafted pass. Where the body runs several slots' rows as one pass, the round reads the weights once for
all of them: on a whole-card Qwen3-30B or Qwen3.6, on V4.1 (two slots' rows a pass), and on GLM-5.3 and Qwen3.8, where a greedy
request's drafted verify window rides the same pass (two windows a pass; with the draft off, the plain rows). A
placed Qwen3-30B or Qwen3.6 (`--place a`), a sampled request on a drafting GLM-5.3 or Qwen3.8 load, V4.1 with the
lookup draft and Qwen3.8 under `--place bp` step their slots in turn, a select and a step a slot each round. At a
fixed expert placement each request answers the tokens of its run alone on the same server; under adaptive residency
the placement follows every stream's passes (see [Limits](#limits)). The context splits as llama-server splits it
with `-np N` and no `-kvu`: `--ctx-size` (or the automatic choice) is the total and each slot holds `total / N` rows,
except on V4.1, where every slot holds the whole context; `--parallel 1` keeps one sequence with the whole context.
The plan counts every slot, so the slots' caches together never pass what it holds, and `--park-ram` is refused by
name. A new request's prompt runs in one call between rounds, and the other streams wait for it.

**Small cards.** Qwen3.6 and Qwen3-30B at `Q4_K_M` (19-21 GB) run whole on a 24 GB card. On a 12-16 GB card
the load sees that the file does not fit the card's free bytes, sends the routed experts to the CPU, and says so
on its plan line; no flag is needed (`--place a` asks for that split on any card).

**A plan that fits the card you have.** The load reads each card's free bytes and the host's available RAM
before anything uploads; the expert share sizes to them, and a plan that cannot fit is refused by name with
every term (dense, KV, context, scratch, margin) and the processes holding the card — before minutes of
loading, not after.

**Speculative decoding.** `BLOOMERY_DRAFT=mtp` drafts with the model's own MTP head (Qwen3.8 and GLM-5.3, on by
default in their servers); `BLOOMERY_DRAFT=dspark` drafts V4.1 with DeepSeek's DSpark head on a second card
(`--place bp`). The drafts run on greedy requests (`"temperature": 0`), and the MTP drafts also on sampled ones
(the default, temperature 0.8 as in llama-server) while one request runs: each kept id is the request's own
sampler's draw from its verified row, as llama.cpp's speculative decoding takes it. A sampled request beside
another running one, a request that bans an id (`ignore_eos`), and a sampled request under DSpark decode one token
a step.

**Adaptive residency.** V4.1, Qwen3.8 and GLM-5.3 count their own routing as they run and swap routed experts
between the card and the host between steps; a prompt call streams its hottest host experts onto the card, so
decode starts warm (on by default, with `--place` unset, `a` or `bp`).

**The engines behind the server.** The release ships `bloomery-serve` alone. A source build also has
`generate_ds41` (token ids in, greedy ids out), `bloomery-chat` (streams text) and the standalone V4.1 and Qwen3.8
servers `bloomery-serve-ds41` and `bloomery-serve-qwen38` ([`BUILD.md`](BUILD.md)). Every seat and flag is its
binary's `--help`.

## Limits

- **sm_86+** GPUs and NVIDIA driver R580+ (CUDA 13); the prebuilt archive carries sm_86 PTX, compiled for the card
  at the first start.
- GLM-5.3 holds at most 16,384 positions a slot, whatever `--ctx-size` asks; a longer prompt is refused with
  `exceed_context_size_error`. A 12–16 GB card's split Qwen3 load picks a small context (2,048 positions a slot at
  `--parallel 2`); agent clients with long system prompts want `--parallel 1` there.
- One model a server; one expert tier card at most (`--place bp`).
- Under WSL2, Qwen3.8, GLM-5.3 and DeepSeek-V4.1 run only with `BLOOMERY_RESIDENCY=off` for now: the residency
  swap's wait on a host word queues behind the request's readback there, and the request stops ([#1](https://github.com/midagedev/bloomery/issues/1)).
- Concurrent streams run as one pass on a whole-card Qwen3-30B or Qwen3.6, on V4.1, GLM-5.3 and Qwen3.8; a placed
  Qwen3-30B or Qwen3.6 and a sampled request on a drafting load step in turn. A new request's prompt runs whole while the other
  streams wait. The decide seat serves one request at a time; DSpark drafts never rejoin — `--parallel > 1` with
  `BLOOMERY_DRAFT=dspark` is refused by name.
- With adaptive residency on, a request's tokens follow the placement its passes ran on, and the placement follows
  the history of passes (the requests before it and the streams beside it): the same history gives the same tokens
  bit for bit, another history can reword an answer at a near tie, since an expert on the card and the same expert
  on the host round their activations differently. `POST /residency/reset` returns to the load's placement;
  `BLOOMERY_RESIDENCY=off` gives repeatable tokens at residency's cost in speed.
- A placed load's plan counts every slot's cache, so `--parallel 2` puts fewer experts on the card than
  `--parallel 1` (8 fewer on V4.1), and the two servers' answers can differ at a near tie, with residency on or off.
- Clef takes text states only, and reads backbone weights of Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 and F32 (not the IQ
  types, Q2_K, Q4_0 or Q4_1).
- V4.1 decode is bound by host memory bandwidth in each step's expert part, and by the card's own serial work
  before it; a long V4.1 prompt (P = 4096) is bound by the card, since the prompt call streams host experts in.
- A pinned nightly with a pinned cuda-oxide revision from our fork ([`BUILD.md`](BUILD.md)).

In progress: serving on N cards for every model; Clef from the IQ and Q2_K files; DeepSeek-V4-Flash-0731.
