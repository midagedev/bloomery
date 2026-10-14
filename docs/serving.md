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
- **The OpenAI text completion API**: `POST /v1/completions` runs the `/completion` path under OpenAI's shape —
  `object: "text_completion"`, `choices` with `finish_reason` `stop`/`length`, `usage`, and a stream of chunks ending
  in `data: [DONE]`. One completion a request: a `prompt` array of several prompts, `echo`, a non-empty `suffix` and
  `best_of > 1` are refused by name, as is `logprobs`.
- **Chat token counts**: `POST /chat/completions/input_tokens` (and its `/v1/` spelling) answers
  `{"input_tokens": N}` with the same chat request's prompt ids — the chat template's render, an image as its span;
  `max_tokens` is not needed.
- **Tool calls on every generative model**: DeepSeek-V4.1-Flash (DSML), GLM-5.3-Flash (its XML), Qwen3-30B
  (Hermes JSON), Qwen3.6-35B and Qwen3.8-Flash-Next (the `<function=…><parameter=…>` markup) parse into
  `tool_calls` and `tool_use` blocks, so agent clients work against each of them.
- **Refused by name** (a 400 naming the field): `samplers` in an order other than llama-server's default list for the
  stages that are on, or naming a sampler this server lacks; typical_p, mirostat, `dry_*`, `xtc_*`, `top_n_sigma` and
  `dynatemp_*` when a request turns them on; `n_predict` or `max_tokens` outside -1..=i32::MAX (-2 does not mean "no
  limit"). `/tokenize` refuses a missing or non-string `content` by name.
- **`timings`** carry llama-server's fields and one of ours, `cache_ms`: the prompt cache's work before the
  prompt (the slot's state saved, a cached state put back, the cut), which `prompt_ms` does not count.
- **`reasoning_budget`**, per request on `/v1/chat/completions` and `/completion`: llama-server's
  `--reasoning-budget` — the think span's budget in generated ids taken while the span is open, `0` closing it at
  once, `-1` or absent unrestricted, and silently ignored when the template already closed the span.
- **`developer` messages** reach the chat template as `system`, the mapping llama-server applies before it — every
  template but GPT-OSS's, whose `<|channel|>` marker in the template source keeps the role. A template with no
  branch for the role would drop the message, and OpenAI clients send one every turn.
- **`reasoning_effort: "none"`** turns thinking off before the template, as llama-server does: the template reads
  `enable_thinking: false` (`chat_template_kwargs` sets it too — a bool, a quoted string is a 400 naming it, and
  `"none"` wins over a kwargs `true`), no `reasoning_effort` reaches the template, and the whole output lands in
  `content`. A template with no thinking switch but an effort header (GLM-5.3's: its generation prompt always opens
  `<think>`) has nothing to turn off, so there `"none"` is sent as `reasoning_effort: "low"`, the template's lowest
  effort, replacing a kwargs `reasoning_effort`; the server prints one line at start saying so. `/props` names which
  case the template is in: `thinking_off` is `"switch"`, `"low_effort"` or `"neither"` (neither knob changes the
  prompt, so `"none"` is `enable_thinking: false` as above).
- **The files**: the GGUF uploads as downloaded, the same quantizations, the chat template read from the file,
  and a tokenizer bit-identical to `llama-tokenize`.
- **The flags**: `-m`/`--model-file`, `--hf <repo>[:<quant>]` (download, resume, sha256 check, never fetched
  twice; it also fetches the model's MTP draft when the repo publishes it, as unsloth's Qwen3.8-Flash-Next GGUFs
  do under `MTP/`, landing it beside the set, where the run picks it up), `--parallel/-np`, `--queue-depth`,
  `--cache-ram`, `--ctx/--ctx-size` (unset, every generative seat and the decide seat take one slot at the lesser of
  262,144 and the file's limit, and halve that while the plan is short of the card — and of the host, where the body
  keeps no tier past it — down to 4,096; one `ctx` line names the rule. A set value is the total the slots split, and
  one that does not fit is refused, never cut. PIN(2026-10-14): `generate::choose_ctx` over `ctx_ask::choose`),
  `--cache-type-k f16|q8_0` (the qwen3
  seats; llama-server's spelling, also the `BLOOMERY_QWEN3_KV` lever — q8_0 halves the KV bytes a position, so the
  auto context nearly doubles on the same card; `--cache-type-v` does not exist, both planes quantize together),
  `--host/--port`, `--alias`, `--mmproj <file>` and `--no-mmproj` (image input, below).
- **No placement flag needed**: leave `--place` out on every model and the load picks for you — the largest
  visible card, a split onto the CPU when the file does not fit it, and on GLM-5.3 the second card as an expert
  tier when its plan puts enough experts there to pay for it (the GLM row of the
  [A6000 + 3090 table](../README.md#numbers)). The plan line names what was picked.
- **Two-card serving like `-ts`**: `--place a` for the largest visible card, `--place bp` to add the next one
  as an expert tier, or a list (`0+1`) by CUDA index (not on the qwen38 seat, which takes `a`, `gate` or `bp`).
- **Clef-Flash's `/v1/systemone`** follows llama.cpp's decision server wire (`model` optional, `/v1/models`,
  501 for video, and for an image when the seat has no encoder, upstream's `confidence` formula).
- **`--api-key`/`--api-key-file`**, llama-server's authentication: a comma-separated list (a quoted item may
  hold commas) and a file (one key per line, `#` lines comments), both flags adding to one set, carried by
  `Authorization: Bearer <key>` or `X-Api-Key: <key>`. `/health`, `/v1/health` and `OPTIONS` stay public;
  every other route — generation, `/props`, `/slots`, `/shutdown` — answers llama-server's 401
  (`Invalid API Key`) without a valid one, and a flag that names no key refuses to start rather than serve
  unlocked. The server binds `127.0.0.1` by default; `--host 0.0.0.0` without a key serves anyone who can
  reach the port.
- **Stopping the server**: `SIGINT`/`SIGTERM` stop it as llama-server's do — no new request is admitted, the
  engine thread ends between its calls, and the process exits 0 after one stderr line naming the cause; a second
  signal terminates at once. `POST /shutdown` does the same over HTTP, accepted only from a loopback peer (a
  machine on the network cannot stop your server): a WSL2 user cannot safely signal a GPU process — the host has
  locked up when one was signalled — so the route stops the server without a signal.

Where it ends today: one model a server, sm_86+ GPUs, and the seats' own defaults where llama-server has none
(see [Limits](#limits)).

## The seats

The decide seat answers SystemOne requests, one at a time. Clef-Flash also reads images (Image input, below); lev reads text states only. It serves two decision models: Clef-Flash (a joint schema head) and lev (`--hf ggml-org/lev-GGUF:Q4_K_M`, a label readout; its file's `qwen35.decision.type` picks it, and its request is rendered by the file's own `systemone` template):

```sh
curl -s http://127.0.0.1:8080/v1/systemone -d '{"state": "User: what is the weather in Seoul tomorrow? Tools available: web_search, calculator.",
  "questions": {"tool": {"type": "choice", "instructions": "Which tool should the agent call next?",
    "criteria": {"web_search": "Look something up online", "calculator": "Do arithmetic", "none": "Answer directly"}}}}'
```

**Two layouts of the Clef-Flash GGUF.** The file's architecture picks the head's source. llama.cpp's `clef` layout
carries the joint head inside the file (bartowski's current upload), so `bloomery-serve --hf
bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M` needs no head flag. The earlier `qwen35` layout takes the release's head
file: `--head joint_head.safetensors`, its `joint_head_config.json` beside it. A `--head` beside a `clef` file names the
head instead of the file's own.

**Concurrent requests.** Every generative seat takes `--parallel N` (unset, one slot; a set
`--ctx-size` with no `--parallel` serves one slot at the whole of it — the flag is one request's context, as
llama-server reads it, and the startup line says so). A count you set is never lowered. The N slots are resident sequences
inside the one model, switched by pointer exchange: nothing parks, and each round advances every busy slot by one
token or one drafted pass. Where the body runs several slots' rows as one pass, the round reads the weights once for
all of them: on a whole-card Qwen3-30B or Qwen3.6, on V4.1 and MiMo-V2.6-Flash (two slots' rows a pass), and on GLM-5.3 and Qwen3.8, where a greedy
request's drafted verify window rides the same pass (two windows a pass; with the draft off, the plain rows). A
placed Qwen3-30B or Qwen3.6 (`--place a`), a sampled request on a drafting GLM-5.3 or Qwen3.8 load, V4.1 with the
lookup draft step their slots in turn (Qwen3.8 under `--place bp` runs one pass since 0.3.1), a select and a step a slot each round. At a
fixed expert placement each request answers the tokens of its run alone on the same server; under adaptive residency
the placement follows every stream's passes (see [Limits](#limits)). Under more slots than one the context splits as
llama-server splits it with `-np N` and no `-kvu`: a set `--ctx-size` is the total and each slot holds `total / N`
rows, on V4.1 the same way; with no context named, each slot holds the ladder's choice and the total is that times N.
`--parallel 1` keeps one sequence with the whole context.
The plan counts every slot, so the slots' caches together never pass what it holds, and `--park-ram` is refused by
name. A new request's prompt runs in one call between rounds, and the other streams wait for it.

**Small cards.** Qwen3.6 and Qwen3-30B at `Q4_K_M` (19-21 GB) run whole on a 24 GB card. On a 12-16 GB card
the load sees that the file does not fit the card's free bytes, sends the routed experts to the CPU, and says so
on its plan line; no flag is needed (`--place a` asks for that split on any card).

**A plan that fits the card you have.** The load reads each card's free bytes and the host's available RAM
before anything uploads; the expert share sizes to them, and a plan that cannot fit is refused by name with
every term (dense, KV, context, scratch, margin, and what the load holds beside the plan) and the processes holding
the card — before minutes of loading, not after. Inside a container the host's RAM is the smaller of `MemAvailable`
and the cgroup's own memory limit (`docker run --memory`, a systemd scope's `MemoryMax`), read the same way.

**When the host cannot hold its experts either.** A Qwen3.8, GLM-5.3 or V4.1 plan whose host need passes the
host's available RAM pages the overflow on the NVMe tier and serves it from a RAM arena filled from the file. The
tokens equal the all-RAM load's. The load prints an `nvtier` record (the arena's budget, the paged bytes and its
counters; a load that attached no tier prints none). Limits of a paged load, as the code has them:
- Qwen3.8 steps one column on a paged plan. With no `--parallel` it serves one slot and the MTP draft stays off; a set
  `--parallel` is served as asked, and one the room's floor cannot hold is refused by name. A set `BLOOMERY_DRAFT=mtp`
  runs. Under `--place bp` each slot steps alone, so a set count stays and only an unset draft goes off.
- Qwen3.8's prompt cache default is 0 on a plan the tier splits, and its `cache ram=…` line names the arena beside
  the load's own host need, so the two anonymous heaps cannot outrun `MemAvailable` into the OOM killer. GLM-5.3's
  default counts the arena inside the host need. `--cache-ram` sets the budget past both.
- Unset, adaptive residency is off on a paged plan (the `residency` record says `paged`): a promotion would read
  the expert cold through the file mapping. A paged Qwen3.8 seat serves with `BLOOMERY_RESIDENCY` set.
- GLM-5.3 has no refusal of its own for slots or the draft. Its NextN draft layer's routed experts are host
  experts too, so the trunk's room is the host room less them.
- V4.1 refuses a paged plan under an r8 pair that reads a sidecar, by name (`BLOOMERY_R8=off` reads the source and
  gives the same bits). The release uses no sidecar.

**Speculative decoding.** `BLOOMERY_DRAFT=mtp` drafts with the model's own MTP head (Qwen3.8 and GLM-5.3, on by
default in their servers); `BLOOMERY_DRAFT=dspark` drafts V4.1 with DeepSeek's DSpark head on a second card
(`--place bp`). Unset, the MTP draft yields to the context: it goes off when the plan with it would leave a slot
fewer positions than the plan without it, a set `--ctx-size`/`--ctx` included, a `draft yield` record naming its
card bytes and the positions a slot gets either way, and the context line that follows is the plain plan's.
`BLOOMERY_DRAFT=mtp` keeps it on whatever the context. The
drafts run on greedy requests (`"temperature": 0`), and the MTP drafts also on sampled ones
(the default, temperature 0.8 as in llama-server) while one request runs: each kept id is the request's own
sampler's draw from its verified row, as llama.cpp's speculative decoding takes it. A sampled request beside
another running one, a request that bans an id (`ignore_eos`), and a sampled request under DSpark decode one token
a step.

**Adaptive residency.** V4.1, Qwen3.8 and GLM-5.3 count their own routing as they run and swap routed experts
between the card and the host between steps; a prompt call streams its hottest host experts onto the card, so
decode starts warm (on by default, with `--place` unset, `a` or `bp`). On Qwen3.8 a call that arrives once the
decode has routed experts of its own puts the pool back where it found it at its end, so a long prompt does not
reset the placement its reply decodes from; a fresh server's calls keep it.

**Prompt cache (Qwen3.8, GLM-5.3).** `--cache-ram` (MiB; 0 turns it off) keeps saved sequence states in host RAM.
By default the budget is the lesser of 8 GiB and half the host RAM the plan leaves, with every slot's checkpoints and
the residency's churn pool counted in; the checkpoint budget follows the slot context. A request that continues an
earlier conversation (its prompt holds the whole earlier prompt) restores that state even after another request used
the slot. A request that shares only part of an earlier prompt, such as the same system prompt with another question,
restores too: a saved state keeps checkpoints at 512·2^i positions below its end and the last one, and a shared
prefix restores at the largest of them inside it. On Qwen3.8 the card's own checkpoints are 4,096 positions
apart, so there a shared prefix shorter than 4,096 tokens does not restore. Saving runs off the request path. The
MiMo and Qwen3 seats run no prompt cache: a request keeps the longest prefix the slot holds, and a save is refused
by name. `timings.cache_ms` and `/metrics`'
`prompt_cache_seconds_total` count the cache's work.

**Image input (V4.1, Qwen3.6, GLM-5.3, Clef).** These seats read images when they have a vision encoder: `--mmproj <file>`,
llama-server's flag, naming the model's encoder GGUF. On V4.1 the file is `mmproj-DeepSeek-V4.1-Flash-BF16.gguf`.
Under `--hf` with no `--mmproj`, every image seat takes the repo's own projector beside the model, unless `--no-mmproj`
says none. The V4.1, Qwen3.6 and GLM-5.3 seats load it only when the text model keeps the same card experts and
context with the encoder's reserve; otherwise one vision line gives the projector's bytes and cost, and the seat serves text. Qwen3-30B-A3B has no encoder, and `--mmproj` there is refused by name. A file of
another encoder is refused by name. Images travel inside the request as PNG or JPEG bytes: an OpenAI `image_url`
part holding a `data:` URL, or an Anthropic `image` block with a base64 source; any other URL is refused, not
fetched. The encoder loads once at start and stays resident: on a card the load's plan leaves off it, or, when every
card is on the plan, as a named reserve whose bytes the plan counts on the stage card; the `vision` line at start
names the card and the bytes (V4.1's is 1.31 GB of weights and activations). On V4.1 a request's image spans ride
the prompt cache like its ids: the same image sent again keeps its span. Without an encoder the server serves no image, and `/props`
answers `vision: false`. A generative seat answers a request with an image with a 400 (`image input is not supported:
this server reads text only`). The Clef decide seat answers it with a 501 and llama.cpp's hint (`image input is not
supported - hint: if this is unexpected, you may need to provide the mmproj`). Lev takes no image: `--mmproj` beside
it is refused, and a request with an image is a 501.

**The engines behind the server.** The release ships `bloomery-serve` alone. A source build also has
`generate_ds41` (token ids in, greedy ids out), `bloomery-chat` (streams text) and the standalone V4.1 and Qwen3.8
servers `bloomery-serve-ds41` and `bloomery-serve-qwen38` ([`BUILD.md`](BUILD.md)). Every seat and flag is its
binary's `--help`.

## Limits

- **sm_86+** GPUs and NVIDIA driver R580+ (CUDA 13); the prebuilt archive carries sm_86 PTX, compiled for the card
  at the first start. The release binaries are x86-64 only (x86-64-v3). A DGX Spark (GB10, aarch64) builds from source
  with `BLOOMERY_ARCH` setting the CUDA arch ([`contrib/dgx-spark.md`](contrib/dgx-spark.md)); the AVX2 kernels stay
  x86-64 only, and a quant type without an aarch64 kernel runs through a scalar mirror there.
- GLM-5.3 holds at most 16,384 positions a slot (`ORACLE_POSITIONS`), whatever `--ctx-size` asks; a longer prompt is refused with
  `exceed_context_size_error`. A 12–16 GB card's split Qwen3 load takes the ladder: unset halves until the plan fits. The sitting measured under the old rule was 2,048 positions a slot at
  `--parallel 2`, and that card has not been remeasured. Agent clients with long system prompts want the context to one request there (`--parallel 1`, or
  a set `--ctx-size` with no `--parallel`).
- One model a server; one expert tier card at most (`--place bp`).
- Concurrent streams run as one pass on a whole-card Qwen3-30B or Qwen3.6, on V4.1, MiMo-V2.6-Flash, GLM-5.3 and Qwen3.8; a placed
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
- Clef takes text states and, with an encoder, images; it reads backbone weights of Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 and F32 (not the IQ
  types, Q2_K, Q4_0 or Q4_1). A head inside a `clef` file is read as the file stores it (bartowski's: Q8_0 matrices,
  which move a probability by under 3e-4 against the release's bf16 head on the eight reference requests).
- V4.1 decode is bound by host memory bandwidth in each step's expert part, and by the card's own serial work
  before it; a long V4.1 prompt (P = 4096) is bound by the card, since the prompt call streams host experts in.
- A pinned nightly with a pinned cuda-oxide revision from our fork ([`BUILD.md`](BUILD.md)).

In progress: serving on N cards for every model; Clef from the IQ and Q2_K files; DeepSeek-V4-Flash-0731.
