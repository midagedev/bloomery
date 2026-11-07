# Serving with bloomery-serve

This page is for someone who has installed bloomery and wants to run `bloomery-serve` as an HTTP server for a client
or an agent. Sections 1 to 8 are the path most readers follow: start a server, send it a request, size its slots and
context, and choose how it uses the cards and memory. Sections 9 to 12 are reference: what each seat (model family
engine) does differently, images and embeddings, the API field by field, and the limits. Install is in the
[README](../README.md#install), and an agent's way to start a model by name is [`agents.md`](agents.md).

`bloomery-serve` speaks llama-server's HTTP API and reads its flags, so an OpenAI, Anthropic or llama-server client
works unchanged. Where bloomery differs from llama-server, this page says so.

## 1. Start a server

Give the server a model and nothing else. Every flag has a default, and the start prints what it chose.

```sh
bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M   # download from Hugging Face, then serve
bloomery-serve -m /path/to/model.gguf                                 # serve a file you have
bloomery-serve -m /path/to/Qwen3.8-Flash-Next-FP4-Local               # serve a checkpoint folder as downloaded
```

**The model.** `-m` (`--model-file`) names a GGUF file (its first shard for a split set) or a checkpoint folder.
`--hf <repo>[@<rev>][:<quant>]` downloads from Hugging Face into `~/.cache/bloomery/hf` (`BLOOMERY_CACHE` moves it):
`:<quant>` picks a quantization of a GGUF repo, and `@<rev>` a branch, tag or commit. The architecture in the file
picks the seat, the engine of that model family; the [README's Use table](../README.md#use) lists them. `--model
<seat>` names the seat yourself, and `bloomery-serve --model <seat> --help` prints the flags that seat takes.

**What the start prints.** Before it listens, the start prints one line for each choice it made, on stderr: `plan`
(which cards hold which part of the model), `ctx` (the context and why), `parallel` (the slot count), `setup` (the
draft, slots and context of a Qwen3.8 load) and others named in later sections. Read these lines first when a start
does not do what you expected.

**When it is ready.** The server binds `127.0.0.1:8080` (`--host`, `--port`; `--port 0` takes a free port) before it
downloads or loads anything. Until the model is loaded, every request answers 503 `loading` with the phase (`fetch`,
`plan` or `load`). `GET /health` answers 200 once the server takes requests. The first start on a card also compiles
the GPU code for it once, which takes tens of seconds; the GB10 archive carries that code compiled already.

**Check a start without loading.** `bloomery-serve --plan ...` prints the plan records for the same flags and exits
before any weight is read onto a card. It still needs the model file (`--hf` downloads it first), the driver and the
cards' free bytes, so run it on the machine that will serve.

**Stop it.** `Ctrl-C`, `SIGINT` or `SIGTERM` stops the server as llama-server's does (section 8.3).

## 2. Send a request

OpenAI clients take the base URL `http://localhost:8080/v1`. Claude Code and the Anthropic SDK take
`ANTHROPIC_BASE_URL=http://localhost:8080`.

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}'
```

The server answers OpenAI's chat completions, completions and Responses APIs, Anthropic's Messages API and
llama-server's own routes (`/completion`, `/tokenize`, `/slots`, `/metrics`, `/props` and the rest). Section 11.1
lists every route.

**Sampling defaults.** Unset, a request samples at llama-server's defaults: `temperature` 0.8, `top_k` 40, `top_p`
0.95, `min_p` 0.05, `repeat_penalty` 1, `repeat_last_n` 64, no token limit but the context (`n_predict` -1) and
`cache_prompt` true. Send `"temperature": 0` for greedy output.

**Unknown fields are ignored**, as in llama-server. A misspelled field is therefore no error: `"temprature": 0`
answers 200 and samples at the default temperature. A field the server knows but cannot honor is a 400 that names it
(section 11.2).

**The model's name** in `/v1/models` and in every answer is `--alias`, else the GGUF's `general.name`, else the seat's
default. llama-server uses the file name instead; this name stays the same when a file is renamed or split. A file
whose `general.name` is a placeholder is named with `--alias`. The qwen3 and decide seats refuse `--alias`.

## 3. Slots and context

A slot serves one request at a time. The server has one slot unless `--parallel N` asks for N, so a second request
waits for the first. Requests that wait queue up to `--queue-depth`, which defaults to 64 minus the slot count and
cannot be set higher.

| You want | Flags | What the server does |
|---|---|---|
| One request at a time, the longest context that fits | none | one slot at the model's limit, or the largest context under it whose plan holds |
| One request at a time, a set context | `--ctx-size C` | one slot of `C` positions |
| N requests at once, a set context each | `--parallel N --slot-ctx S` | N slots of `S` positions each |
| N requests at once sharing a set total | `--parallel N --ctx-size C` | N slots of `C / N` positions each, as llama-server's `-np N` without `-kvu` |
| As many requests as memory holds | `--parallel auto`, optionally `--slot-ctx S` | the most slots, up to 8, whose plan holds one request's context |

**The default context.** With no `--ctx-size`, each slot asks the model's trained limit. Where that plan does not fit
the card (or the host, or the unified pool of a DGX Spark), the slot takes the largest context under it whose plan
holds, a multiple of 1,024 down to 4,096. The `ctx` line names the rule: `set`, `target` or `fitted`. A context you
set is never cut: one that does not fit is refused by name, with the largest one that does.

**`--slot-ctx S`** names one request's context apart from the slot count: `--parallel 4 --slot-ctx 32768` is the same
load as `--parallel 4 --ctx-size 131072`. Alone it serves one slot of `S`. Beside `--ctx-size` it is refused, since
one names the total and the other a slot's share. Every seat that takes `--parallel` takes it.

**`--parallel auto`** counts the slots from memory on every seat that serves several slots: the most, up to 8, whose
plan holds one request's context. That context is `--slot-ctx` when set; unset, it is the context one slot takes with
no flags, when that is the model's limit. Where one slot already has to fit under the limit, auto serves one slot. The
placement and the draft are chosen at one slot first. `--parallel auto` beside `--ctx-size` is refused. The `parallel`
line names the count with `from=auto`. The default stays one slot; auto is never chosen for you.

Where the load chosen at one slot serves one sequence, auto serves one slot and says so: `from=route-trace` under
`BLOOMERY_ROUTE_TRACE` (qwen38, ds41), `from=dspark` under the DSpark draft (ds41, qwen3), and a `parallel auto
slots=1` line under the qwen3 seat's in-file MTP draft (`BLOOMERY_DRAFT=off` counts the slots) and beside an expert
tier card on the mimo2 seat.

**How slots share the card.** The slots are resident sequences inside one model, switched by pointer exchange, and
the plan counts every slot's cache, so their caches together never pass what it holds (`--park-ram` is refused). Each
round advances every busy slot by one token or one drafted pass. Where the seat runs several slots' rows as one pass,
the round reads the weights once for all of them; which seats do that is in section 9.1. A placed load's plan counts
every slot, so `--parallel 2` puts fewer experts on the card than `--parallel 1`.

**A new request's prompt.** A new prompt runs in one call between rounds while the other streams wait. On Qwen3.8,
unless `--mmproj` loads an image encoder, a prompt with more than 4,096 ids past its cached prefix runs instead as
calls that end at multiples of 4,096 positions, one a round between the other streams' rounds. The cut calls give the
same bits and the same cached prefixes as the one call.

**Seats with one slot.** The nemotron, raon1, solar2 and musespark seats serve one slot, and the decide seat one
request at a time; `--parallel` above 1 is refused by name there.

## 4. Speculative decoding

Where a model has a draft, the server drafts by default: a small draft proposes a few tokens and the model checks them
in one pass, so a reply decodes faster with the same tokens a plain run gives. `BLOOMERY_DRAFT` chooses the draft:

| `BLOOMERY_DRAFT` | What drafts |
|---|---|
| unset | on Qwen3.8 and GLM-5.3, the model's own MTP draft by the yield rule below; on the qwen3 seat, a `qwen35moe` or `qwen35` file's in-file MTP draft under section 9.3's limits; on Raon-1, its DFlash2 draft on requests that think; no draft on V4.1 |
| `mtp` | the MTP draft, kept on whatever the context |
| `off` | nothing |
| `dspark` | DeepSeek's DSpark head on V4.1 (`BLOOMERY_DSPARK_MODEL` names it); on the qwen3 seat, an HF draft directory for a whole-card `qwen35moe` target |
| `lookup` | the n-gram lookup draft (Qwen3.8, and on the qwen3 seat a whole-card verifying load) |

**The yield rule (Qwen3.8 and GLM-5.3).** Unset, the MTP draft yields to the context. It goes off when the plan with it leaves a slot half
the positions of the plan without it or fewer, or holds no context at all, a set `--ctx-size` included. A `draft
yield` record names the draft's card bytes and the positions a slot gets either way, and the `ctx` line that follows
is the plain plan's. Any other refusal of the drafted plan stops the load by name. The draft also goes off under
`--place gate` and with stores shorter than one window. On Qwen3.8 it also goes off on a plan that pages experts from
NVMe (section 5.4) and on a checkpoint folder without the MTP layer. Each prints a `load draft=off` record with the
reason, never a refusal.

**Which requests draft.** The MTP drafts run on greedy and sampled requests, on ones that ban an id (`ignore_eos`) or
bias ids (`logit_bias`), on ones that ask for probabilities and on structured output, alone or beside other running
requests. Each kept id is the request's own pick from its verified row, so a request of one seed takes the same ids
alone and beside others, and its probabilities are the values its steps would report. On Qwen3.8 the MTP draft runs
on any slot count. A request under DSpark decodes one token a step.

The model-specific details of each draft (where the DSpark draft sits, which files a qwen3-seat draft fits, the
checkpoint folder's own MTP layer) are in section 9.

## 5. Placement: cards, host RAM and NVMe

Leave placement alone first. With `--place` and the `BLOOMERY_*` variables unset, the load picks the largest visible
card, sends the routed experts the card cannot hold to the host, and on GLM-5.3 and Qwen3.8 takes a second card as an
expert tier when its plan puts enough experts there to pay for it. The `plan` line names what was picked.

### 5.1 Fitting the machine you have

The load reads each card's free bytes and the host's available RAM before anything uploads, and sizes the expert share
to them. A plan that cannot fit is refused by name, before minutes of loading, with every term (dense weights, KV,
context, scratch, margin, what the load holds beside the plan) and the processes holding the card. Inside a container
the host's RAM is the smaller of `MemAvailable` and the container's memory limit (`docker run --memory`, a systemd
scope's `MemoryMax`). On a DGX Spark the card and the host share one memory pool, and the plan counts both against it.

**Small cards.** Qwen3.6 and Qwen3-30B at `Q4_K_M` (19 to 21 GB) run whole on a 24 GB card. On a 12 to 16 GB card the
load sees that the file does not fit, sends the routed experts to the CPU and says so on its plan line; no flag is
needed.

### 5.2 Choosing cards with llama-server's flags

`bloomery-serve --list-devices` prints the devices as llama-server does and exits:

    Available devices:
      CUDA0: NVIDIA RTX A6000 (… MiB, … MiB free)
      CUDA1: NVIDIA GeForce RTX 3090 (… MiB, … MiB free)

The first figure is the bytes a plan may use, the second what was free at the start. `-dev`, `-mg`, `-ts` and `-devd`
count by the `CUDA<N>` ordinal. Each flag prints one line saying what it was read as, and a flag bloomery cannot honor
ends the start with a refusal that names it; none is ignored.

| Flag | What bloomery does |
|---|---|
| `-dev CUDA0,CUDA1` | The cards the plan may use; default every visible card. `none` is refused: no seat runs on the CPU alone. |
| `-sm layer` (default) | Every listed card takes part: the largest runs every layer and the head, the others hold experts under the host tier. bloomery does not split layers; one line says so. |
| `-sm none -mg N` | The one card at index N of the list (`-mg` default 0). |
| `-sm row`, `-sm tensor` | Refused: bloomery has no row split, and its tensor parallel is `--place tp` over `BLOOMERY_TP_NODES`, whose ranks are other nodes. |
| `-ts 3,1` | The cards that stay: a zero or missing share leaves the card out. With two or more cards left, the shares are each card's share of the routed experts (`BLOOMERY_CARD_BUDGET=CUDA0=750000x,CUDA1=250000x`, in millionths). Each card plans its own floor (dense tensors, cache, context, scratch, reserves, margin) plus its part of the experts, in the proportion of the shares, as many in all as the tightest card holds; the host tier takes the rest. A share that rounds to no millionth, a card whose floor passes its bytes and shares on a seat that puts no routed expert on a card are refused. One card left, or `-sm none`, sets nothing. |
| `-mg N` under `-sm layer` | Refused unless N is 0: the stage here is the largest listed card. |
| `-ngl` | `all`, `auto`, a negative count, or at least the file's layers + 1: as the load is. A smaller count is refused: every layer runs on the stage card. |
| `--cpu-moe`, `-ncmoe N`, `-ot <experts>=CPU` | The qwen38 seat plans every routed expert on the host tier (`BLOOMERY_QWEN38_EXPERTS=host`). `-ncmoe` counts only as every layer; `-ot` reads only the all-experts patterns (`exps`, `ffn_.*_exps`, `.ffn_.*_exps.`, `\.ffn_.*_exps\.`, `--cpu-moe`'s own regex) to `CPU`. Every other seat refuses them: its plan already puts on each card what the card holds and the rest on the host. |
| `-fit on` (default) | The plan always fits. `-fit off` is refused. Size the load with `-dev` or `-sm none -mg N`, `-fitt` and `--ctx-size`. |
| `-fitc N` | N must be 4096, the least context an unset context steps down to; pin a context with `--ctx-size`. |
| `-fitt MiB[,MiB…]` | The margin kept free on each card: one value for every device, or one per listed device. Sets `BLOOMERY_CARD_MARGIN`. The plan's own margin is 1024 MiB, also llama-server's default, so `-fitt 1024` changes nothing. A list of the wrong length and a margin that leaves a card no usable byte are refused. |
| `-devd CUDA1` | The card the V4.1 seat's DSpark draft opens on (`BLOOMERY_DSPARK_CARD=cuda1`). Another seat refuses it. |

The cards the flags name become the `--place` word the seat parses, so a seat that serves no expert tier card refuses
the tier as a `--place` refusal after the line that says what the flags were read as. Giving a flag twice is refused.
Each flag's `LLAMA_ARG_*` variable is read too (section 7.3).

Examples, with the A6000 as CUDA0 and the 3090 as CUDA1:

- `bloomery-serve -m model.gguf -sm none -mg 1` runs on CUDA1 alone (`--place cuda1`); `-dev CUDA1` is the same.
- `-sm none -mg 1 -fitt 4096` also keeps 4096 MiB free on CUDA1, and the host tier holds the experts that no longer fit.
- `-dev CUDA0,CUDA1` is the plan `bp`: the larger card runs every layer and the head, the other holds experts.
- `-dev CUDA1,CUDA0 -ts 1,0` indexes that list, so CUDA0 is left out; `-ts 3,1` keeps both and splits the routed
  experts 3:1.
- `--model qwen38 -m model.gguf --cpu-moe` plans every routed expert on the host tier.
- `BLOOMERY_DRAFT=dspark BLOOMERY_DSPARK_MODEL=draft.gguf bloomery-serve -m model.gguf -devd CUDA1` opens the V4.1
  draft on CUDA1.

**Margins and budgets without llama's flags.** `BLOOMERY_CARD_MARGIN=4G` keeps 4 GiB free on every card, and
`BLOOMERY_CARD_MARGIN=CUDA0=2G,CUDA1=4G` gives each named card its own. `BLOOMERY_CARD_BUDGET=38G` caps the bytes every
card plans with, and `BLOOMERY_CARD_BUDGET=CUDA0=38G,CUDA1=20G` caps each named card; it sizes the cards, while
`--place` or the load still chooses them. A device the census does not list, a card named twice, and a value that
leaves a card no usable byte are refused by name.

### 5.3 The `--place` words

`--place` is bloomery's older spelling, and it keeps its meaning: `a` (the largest visible card), `gate` (the 3090's
bytes alone, on the smallest card with room for them), `bp` (the next-largest card as the largest's expert tier) and a
list by card name or CUDA index (`0+1`, `cuda1+cuda0`, the stage first). `--place` together with a llama-server
placement flag is refused.

### 5.4 When the host cannot hold the experts either

A Qwen3.8, GLM-5.3, V4.1, MiniMax-M3, Muse Spark, Mistral Large 4, Solar-Open2, Flint or Nemotron-H MoE plan whose host
need passes the host's available RAM pages the overflow on the NVMe tier and serves it from a RAM arena filled from the file. MiMo's
plan refuses that room by name: its host run attaches no tier. A checkpoint folder's plan counts the host bytes its load
reads the card's parts in, so the tier pages what they take instead of the load refusing the room. The load prints an
`nvtier` record with the arena's budget, the paged bytes and its counters.

Who computes the arena's experts decides whether the tokens equal an all-RAM load's:

- On an anonymous arena (`BLOOMERY_NVTIER_MEM=anon`, the default on a card with its own memory) the host computes
  them, and the tokens equal the all-RAM load's.
- On a pinned arena (`pinned`, the default on an integrated GPU such as the DGX Spark's GB10) GLM-5.3's decode steps
  compute them on the card from the arena's slots. The card quantizes the activation as q8_1 and sums in its own
  order, so the tokens can part from the all-RAM load's after some steps. A prompt's batch, a load whose residency
  machine runs, and the other families stay with the host. The `nvtier` record's `card_leg` says `on` or why not.

A paged load serves differently:

- Qwen3.8 steps one column on a paged plan. With no `--parallel` it serves one slot and the MTP draft stays off; a set
  `--parallel` is served as asked, and one the room's floor cannot hold is refused by name. A set
  `BLOOMERY_DRAFT=mtp` runs. Under `--place bp` each slot steps alone.
- Qwen3.8's prompt cache defaults to 0 on a plan the tier splits; GLM-5.3's default counts the arena inside the host
  need. `--cache-ram` sets the budget past both.
- Adaptive residency (section 5.5) follows the model's placement and room rules on a paged plan too; paging alone
  does not turn it off. Set `BLOOMERY_RESIDENCY=off` for a fixed placement.
- GLM-5.3's NextN draft layer's routed experts are host experts too, so the trunk's room is the host room less them.
- V4.1 refuses a paged plan under an r8 pair that reads a sidecar (`BLOOMERY_R8=off` reads the source and gives the
  same bits). The release uses no sidecar.

NVMe arena eviction options are in the [lever registry](../crates/levers/src/registry.rs).

### 5.5 Adaptive residency

V4.1, Qwen3.8 and GLM-5.3 count their own routing as they run and swap routed experts between the card and the host
between steps. A prompt call streams its hottest host experts onto the card, so decode starts warm. It is on by default
with `--place` unset, `a` or `bp`. On Qwen3.8 a call that arrives once decode has routed experts of its own puts the
pool back where it found it at its end, so a long prompt does not reset the placement its reply decodes from.

Residency makes tokens depend on history: section 12 says how, and `BLOOMERY_RESIDENCY=off` gives repeatable tokens at
its cost in speed. `POST /residency/reset` returns to the load's placement.

**Why a Qwen layer stays on the host.** Under the card expert rule, each printed plan is followed by `host only
layer=… tensor=… why=…` records for routed tensors whose kernel contract keeps their layer off the card (the tensor's
gate, up or down role, and matching gate and up types). A byte limit alone, a dense model and an explicit all-host
choice print none. The records go to stderr in the server.

## 6. Memory beside the weights

**The prompt cache** (V4.1, Qwen3.8, GLM-5.3, MiMo, MiniMax-M3, Muse Spark, Mistral Large 4). `--cache-ram` (MiB; `0`
turns it off) keeps saved sequence states in host RAM, so a request that shares a prefix with an earlier one restores
it, even after another request used the slot. By default the budget is the lesser of 8 GiB and half the host RAM the
plan leaves, with every slot's checkpoints and the residency's churn pool counted in. llama-server's `-1` (no limit) is
refused: the cache is host memory beside the load, so it always has a budget.

- A request that continues an earlier conversation restores that state. A request that shares only part of an earlier
  prompt restores too: a saved state keeps checkpoints at 512 × 2^i positions below its end and the last one, and a
  shared prefix restores at the largest of them inside it. On Qwen3.8 the card's own checkpoints are 4,096 positions
  apart, so there a shared prefix shorter than 4,096 tokens does not restore.
- On MiniMax-M3, Muse Spark and Mistral Large 4 a saved state is the rows of the held positions (K and V, or Mistral
  Large 4's latent rows; the index keys of MiniMax-M3 and Mistral Large 4), with no checkpoints, so any shared prefix
  restores.
- The qwen3 seat keeps only the longest prefix the live slot holds and runs no saved prompt cache. For Qwen3.5 and
  Qwen3.6, when the vocabulary and chat template both carry `<|im_start|>`, it cuts prompt calls at message starts, so
  a shared system prompt stays in the live slot for another question.
- Saving runs off the request path. `timings.cache_ms` and `/metrics`' `prompt_cache_seconds_total` count the cache's
  work.

**Slot files.** `--slot-save-path DIR` lets llama-server's slot actions save a slot to a file and restore it. The
qwen3 seat refuses it.

**The KV cache format.** `--cache-type-k f16|q8_0` (the qwen3 and qwen38 seats; also the `BLOOMERY_QWEN3_KV` lever)
sets the format of both planes; `--cache-type-v` does not exist. Unset is f16, for a GGUF file and a checkpoint folder
alike. q8_0 takes 17/16 bytes a value against f16's 2, which nearly doubles the default context on the same card for
the qwen3 seat; on a GB10 (Qwen3.8) and an RTX 5090 (Qwen3.6) its prompt pass ran slower than f16's, with no faster
decode step. `setup` and the qwen3 seat's `load` line print the format as `cache=` and what chose it as
`cache_from=flag|lever|default`.

**The prompt batch size.** `-ub N` (`--ubatch-size`) is the most positions one prompt pass runs. On the qwen38 seat it
takes 1 to 4096 (default 4096, also `BLOOMERY_QWEN3_UBATCH`). A smaller size frees card bytes for a context or slot
count the default does not fit and runs long prompts slower; `--plan` prints the card bytes at each size. On the qwen3
and decide seats it applies to a file with no routed experts: unset, the load takes the largest of 4096, 2048, 1024
and 512 the card holds, and one `ubatch` line names the sizes it tried. A routed file there takes
`BLOOMERY_QWEN3_UBATCH` and refuses the flag.

## 7. Settings

### 7.1 The settings file

`bloomery-serve --config FILE` reads a start's flags from one JSON object, so a long start line can live in a file
that people and services share. Each key is a flag's long name without its dashes:

```json
{
  "model-file": "/models/Qwen3.8-Flash-Next-FP4-Local",
  "ctx-size": 65536,
  "parallel": 2,
  "host": "0.0.0.0",
  "api-key-file": "/etc/bloomery/keys",
  "no-mmproj": true
}
```

This starts as `bloomery-serve -m /models/Qwen3.8-Flash-Next-FP4-Local --ctx-size 65536 --parallel 2 --host 0.0.0.0
--api-key-file /etc/bloomery/keys --no-mmproj` does.

- **Values.** A string or a number is the flag's value. `true` gives a switch; leave out a switch that is off. A list
  gives the flag once for each value (`"api-key": ["k1", "k2"]`). A path is read from the directory the server starts
  in.
- **Keys.** Every flag that sets something has a key, its long name: `ctx-size` for `-c`, `device` for `-dev`, `model`
  for `--model`. A key the seat does not take is refused by name before anything is fetched.
- **The command line replaces the file's entry** for the same flag, in any spelling (`-c 4096` replaces
  `"ctx-size"`), and for a flag that sets the same thing another way: `-m` or `--hf` replaces `"model-file"`,
  `--embd-hf` replaces `"embd-model"`, `--no-mmproj` replaces `"mmproj"`, `--embd-no-mmproj` replaces
  `"embd-mmproj"` (and the reverse of each), `-ncmoe` or `-ot` replaces `"cpu-moe"`, and `--place` replaces
  `"device"`, `"split-mode"`, `"tensor-split"` and `"main-gpu"` while those flags replace `"place"`. A catalog variant
  (`"model": "glm:Q8_0"`) is a model source; a bare seat word (`"model": "glm"`) names the seat.
- **One stderr line for each entry** says what it was read as, or which flag replaced it. The line prints the value
  only for `ctx-size`, `ubatch-size`, `cache-ram`, `parallel`, `alias`, `host`, `port`, `api-key-file`,
  `slot-save-path`, `chat-template-file` and the image flags, never a key.
- **The file is read before `--help`**, because it can name the seat whose usage `--help` prints.
- **Refused by name**, before the model is read: a file that is not one JSON object, a key named twice, a key with its
  dashes, a key that asks the start to do something rather than set it (`plan`, `levers`, `help`, `version`,
  `list-devices`, `records-schema`, `recommend`, `dry-run`, `json`, `config`), `false`, `null`, an object or an empty
  list as a value, a value for a switch, `true` for a flag that takes a value, and `--config` given twice.

### 7.2 Which setting wins

One value a setting, chosen in this order:

1. **The command line.** A flag that stands for a `BLOOMERY_*` lever replaces it: `-ub` replaces
   `BLOOMERY_QWEN3_UBATCH`, `--cache-type-k` replaces `BLOOMERY_QWEN3_KV`, `--autotune` replaces `BLOOMERY_AUTOTUNE`,
   and the placement flags replace the levers they set.
2. **The settings file** (`--config`). Its entries count as the command line's against the variables and levers below.
3. **llama-server's variables.** A `LLAMA_ARG_*` variable is read as its flag (section 7.3), so it also replaces the
   lever: `LLAMA_ARG_UBATCH=48` beside `BLOOMERY_QWEN3_UBATCH=64` runs 48.
4. **The levers**, the `BLOOMERY_*` variables no flag or `LLAMA_ARG_*` variable names.
5. **The seat's default.**

Two cases are not ranked. API keys add: `LLAMA_API_KEY`, `--api-key` and `--api-key-file` make one set, though a
command-line `--api-key` or `--api-key-file` replaces the settings file's list for that flag. Two sources of one thing
are refused by name, never picked between: the model named twice (`-m` beside `--hf`, `LLAMA_ARG_MODEL` beside
`LLAMA_ARG_HF_REPO`, or `BLOOMERY_REF_MODEL` naming another file), `--place` beside a placement flag, and the routed
experts' place named twice.

### 7.3 llama-server's environment variables

Every launcher reads llama-server's variable for each flag it takes, as that flag with that value. The names are the
`.set_env(...)` of each flag in llama.cpp's `common/arg.cpp` (commit f5027fb).

| Flag | Variable |
|---|---|
| `-dev`, `-sm`, `-ts`, `-mg` | `LLAMA_ARG_DEVICE`, `LLAMA_ARG_SPLIT_MODE`, `LLAMA_ARG_TENSOR_SPLIT`, `LLAMA_ARG_MAIN_GPU` |
| `-ngl` | `LLAMA_ARG_N_GPU_LAYERS` |
| `--cpu-moe`, `-ncmoe`, `-ot` | `LLAMA_ARG_CPU_MOE`, `LLAMA_ARG_N_CPU_MOE`, `LLAMA_ARG_OVERRIDE_TENSOR` |
| `-fit`, `-fitt`, `-fitc` | `LLAMA_ARG_FIT`, `LLAMA_ARG_FIT_TARGET`, `LLAMA_ARG_FIT_CTX` |
| `-c`, `--ctx-size` | `LLAMA_ARG_CTX_SIZE` |
| `-ub`, `--ubatch-size` | `LLAMA_ARG_UBATCH` |
| `-cram`, `--cache-ram` | `LLAMA_ARG_CACHE_RAM` |
| `-mm`, `--mmproj` | `LLAMA_ARG_MMPROJ` |
| `--image-min-tokens`, `--image-max-tokens` | `LLAMA_ARG_IMAGE_MIN_TOKENS`, `LLAMA_ARG_IMAGE_MAX_TOKENS` |
| `-np`, `--parallel` | `LLAMA_ARG_N_PARALLEL` |
| `-a`, `--alias` | `LLAMA_ARG_ALIAS` |
| `--host`, `--port` | `LLAMA_ARG_HOST`, `LLAMA_ARG_PORT` |
| `--api-key`, `--api-key-file` | `LLAMA_API_KEY`, `LLAMA_ARG_API_KEY_FILE` |
| `--slot-save-path` | `LLAMA_ARG_SLOT_SAVE_PATH` |
| `--chat-template-file` | `LLAMA_ARG_CHAT_TEMPLATE_FILE` |
| `-m`, `--hf` | `LLAMA_ARG_MODEL`, `LLAMA_ARG_HF_REPO` (`bloomery-serve` only; both set is refused) |

- **The command line wins**, and one line says so. `--place` replaces the variables that name cards
  (`LLAMA_ARG_DEVICE`, `LLAMA_ARG_SPLIT_MODE`, `LLAMA_ARG_TENSOR_SPLIT`, `LLAMA_ARG_MAIN_GPU`).
- **A variable whose flag would be refused** ends the start under the variable's name, for example
  `LLAMA_ARG_SPLIT_MODE=row`.
- **`LLAMA_ARG_CPU_MOE` is a switch**: on for `on`, `enabled`, `true`, `1`, off for `off`, `disabled`, `false`, `0`;
  any other value is refused, where llama-server ignores it.
- **Keys add.** llama-server adds every key it is given, so a variable's keys count beside the command line's. A
  variable that names no key (`LLAMA_API_KEY=`) ends the start, as `--api-key ""` does. No line prints a key.
- **Short spellings** `-c`, `-ub`, `-np`, `-a`, `-mm` and `-cram` are respelled as their long flags before a seat
  parses, so a seat that takes none of the flag refuses it by the long name.
- **A variable bloomery does not read** (`LLAMA_ARG_FLASH_ATTN`, `LLAMA_ARG_THREADS` and the rest) gets a line saying
  so, never silently dropped. The decide seat names each placement variable that is set as one it does not act on.

### 7.4 The levers

A lever is a `BLOOMERY_*` variable that changes one behavior, such as `BLOOMERY_DRAFT` or `BLOOMERY_RESIDENCY`.
`bloomery-serve --levers` prints each lever, whether this binary and seat apply it, and its value. The list is the rows
of [`crates/levers/src/registry.rs`](../crates/levers/src/registry.rs) ([`BUILD.md`](BUILD.md#levers)).

## 8. Running it as a service

### 8.1 API keys and binding

The server binds `127.0.0.1` by default. `--host 0.0.0.0` without a key serves anyone who can reach the port.

`--api-key` takes a comma-separated list (a quoted item may hold commas), and `--api-key-file` a file of one key a
line (`#` lines are comments; blank lines are skipped; a key holds no whitespace). Both add to one set, read once at
start, so a pipe such as `/dev/stdin` works and a later edit takes a restart. A client sends `Authorization: Bearer
<key>` or `X-Api-Key: <key>`. `/health`, `/v1/health` and `OPTIONS` stay public; every other route answers
llama-server's 401 (`Invalid API Key`) without a valid key. The check runs on the headers before `100 Continue`, and a
denied request's connection closes without its body being read. A flag that names no key refuses to start.

A key file line can narrow its key with fields after it, separated by whitespace
(`sk-app scope=inference inflight=2 rate=30/m burst=5`). A key with none reaches every route unlimited.

- `scope=` takes `inference` (generation, token counts, the tokenizer, embeddings, stored responses, `/v1/systemone`),
  `read` (`GET /props`, `/slots`, `/metrics`) or `admin` (every route, the default), comma-separated. `inference` and
  `read` both reach the model list and lookups. A route outside the scope answers 403 (`permission_error`) and closes
  the connection. Slot actions, `/residency/reset` and `/shutdown` need `admin`.
- `inflight=N` is the most requests the key may have answered at once. A request holds its place until its answer is
  written and the engine has left its generation, also after a cancel or a disconnect.
- `rate=N/s` or `rate=N/m` is a bucket of `burst=B` requests (`N` unless set) that refills `N` a second or a minute.
- A request past either limit answers 429 (`rate_limit_error`) with `Retry-After` in whole seconds; the connection
  stays open. Each key counts apart.
- A field the line does not take, or names twice, refuses the start, naming the file, the line and the field's place,
  never the key. A key given twice with different fields does too.

### 8.2 What it logs

**One stderr line a request.** When a generation request ends, the server prints:
`bloomery-serve: request <task id> <endpoint path>: prompt <N> ids (<C> reused) in <S> s, <R> ids/s; generated <G> ids
in <S> s, <R> ids/s; stop <reason>; ctx <n_past>/<n_ctx>`. `<reason>` is `eos`, `stop`, `length`, `cancel` or `error`.
A request that generates for more than 30 s prints a shorter `generating` line every 30 s. A request of several choices
prints one line a choice.

**`/metrics`** carries llama-server's counters and these of bloomery's:

- The decode rounds by `width`, the slots one round stepped: `rounds_total`, `round_seconds_total`,
  `round_card_wait_seconds_total` (the part the engine waited for the card) and `round_ple_wait_seconds_total` (the
  part a Qwen3.8 call waited for PLE rows read from NVMe); the rest is host time. `card_wait_seconds_total` and
  `ple_wait_seconds_total` count the same waits over every call.
- `spec_decode_num_draft_tokens_by_depth_total` and `spec_decode_num_accepted_tokens_by_depth_total` count a drafted
  pass's ids by `depth` (the id's place in its draft) under `source`, the drafter's kind (`/props`'
  `engine.draft.kind`). They sum to `spec_decode_num_draft_tokens_total` and `spec_decode_num_accepted_tokens_total`.
- `prompt_cache_seconds_total`, and on a server with an image encoder `prepared_image_hits_total`,
  `prepared_image_misses_total` and the prepared-image cache's bytes and images.

**`timings`** in an answer carry llama-server's fields and `cache_ms`, the prompt cache's work before the prompt.

**Kernel picks at load.** The qwen38 seat times, after the load and before its first capture, each way to run a kernel
that gives the same output bits (today the MXFP8 projections at four to eight columns, the widths a drafted pass
verifies), each on weights read from the card's memory rather than its L2, as a decode step reads them. A way replaces
the built-in rule only when it is faster on every one of five repeats by more than the repeats' spread. The load prints
an `autotune pick` line for each kernel shape and an `autotune` line with the time it took, and `/props`'
`engine.autotune` lists the same. Each pick is kept in the cost store (`$BLOOMERY_COSTS`, else
`~/.cache/bloomery/costs`) under the card, the driver, the build and the shape, so the next load of the same card and
build reads it and times nothing; an entry that does not decode is timed again. A store that cannot be created, locked,
read or written stops the load, and the error names `--autotune off`. `--autotune off` (or `BLOOMERY_AUTOTUNE=off`)
runs the built-in rule with no timing and no store.

### 8.3 Stopping

`SIGINT` or `SIGTERM` admits no new request, ends the engine between its calls and exits 0 after one stderr line naming
the cause; a second signal ends it at once. `POST /shutdown` does the same over HTTP, accepted only from a loopback
peer. Use it under WSL2, where signalling a GPU process has locked the host up. An engine error ends the process with
exit code 70 after the request gets a 500.

### 8.4 Starting under a broker

A placed load prints a `need` record beside its `plan` record (a seat with `--plan` prints both before it exits). It
states what the load asks of the machine:

    need bytes=… pinned=… anon=… cache=… locked=… unified=… os=… cache_ram=… lock=false cards=[A6000:…] threads=… cpus=[…] caller=… allowed=[…]

- `bytes` is the host need the load checks against `MemAvailable`, less the OS reserve `os`. `pinned`, `anon`,
  `cache`, `locked` and `unified` split it by what can take a page back and sum to it: driver-pinned memory (ring
  shadows, the tier's rows, a pinned arena, the expert stream's staging), anonymous pages (the NVMe tier's arena),
  reclaimable file pages, file pages `mlock`ed under `BLOOMERY_HOST_LOCK`, and an integrated device's share of its
  pool. `cache_ram` is the prompt cache, beside `bytes`.
- `cards` is the bytes the plan puts on each card. `threads` counts the worker pool, `cpus` the cpu each worker is
  pinned to, `caller` the cpu `BLOOMERY_PIN_MAIN` pins the dispatching thread to, and `allowed` every cpu the process
  may run on (`taskset`, a container's cpuset, a unit's `AllowedCPUs`).

`BLOOMERY_NEED_HOOK=<absolute path>` runs a program once per placed load, after the plan and before a card is opened,
with that line on its standard input. Exit 0 goes on. Exit 75 ends the process with 75, naming the hook's last output
line, so a broker can queue the start and ask again. Any other status ends the load with a named error. A logger is a
hook: `cat >> need.log` as a script. A load that opens a whole model on one card with no plan states no need and asks
no hook; a set hook is named on stderr as not asked.

## 9. Seats

A seat is the engine of one model family. Sections 1 to 8 hold for every seat; this section lists what differs.

### 9.1 Flags by seat

`bloomery-serve` reads its own flags first: `--model <seat>`, the model (`-m` or `--hf`), `--head` and `--head-config`
(the decide seat), the embedding side model (`--embd-*`), the placement flags (not on the decide seat), `--config`,
`--levers`, `--list-devices` and `--version`. The seat parses every other flag. Every generative seat takes `--host`,
`--port`, `--ctx-size` (`-c`, `--ctx`), `--parallel` (`-np`), `--slot-ctx`, `--place`, `--plan`, `--queue-depth`,
`--chat-template-file`, `--api-key`, `--api-key-file`, `--alias` and `--slot-save-path`; the qwen3 seat refuses the
last two. A flag the seat does not take is refused by name with the seat's usage, before `--hf` downloads anything.

| Seat | `--parallel` | Several slots in one pass | `--cache-ram` | `-ub` | `--mmproj` ¹ | Its own flags |
|---|---|---|---|---|---|---|
| `ds41` | N or `auto` | yes, two slots' rows a pass | yes | no | yes | |
| `glm` | N or `auto` | yes, drafted windows too | yes | no | yes | `--prefill batch\|steps` |
| `qwen38` | N or `auto` | yes, drafted and lookup windows too, except under `--place bp` | yes | 1 to 4096 | yes | `--cache-type-k f16\|q8_0`, `--autotune on\|off` |
| `qwen3` | N or `auto` | on a whole-card load; a placed load steps in turn | no | a file with no routed expert | yes | `--cache-type-k f16\|q8_0` |
| `mimo2` | N or `auto` | yes, two slots' rows a pass | yes | no | no | `--prefill batch\|steps` |
| `minimaxm3` | N or `auto` | no, slots step in turn | yes | no | no | |
| `mistrallarge4` | N or `auto` | no, slots step in turn | yes | no | yes (Pixtral) | |
| `flint` | N or `auto` | | no | no | no | |
| `musespark` | 1 | | yes | no | no | |
| `solar2`, `nemotron`, `raon1`, `kimik3` | 1 | | no | no | no | |
| `decide` | one request at a time | | no | yes | Clef only | `--head`, `--head-config` |

`N` is any count, 1 by default. `1` is the seat's one slot, and `--parallel` above 1 or `auto` is refused by name. `--mmproj`
stands for the image flags together. `--park-ram` is refused on every seat that reads it, since slots are resident
sequences. The decide seat takes only `--host`, `--port`, `--ctx-size`, `-ub`, the key flags and the image flags.
V4.1 and Qwen3.6 with the lookup draft step their slots in turn.

¹ A build with the `vision` feature; the release has it, and another build refuses the image flags by name.

### 9.2 Qwen3.8 checkpoint folders

The qwen38 seat serves NVIDIA's Qwen3.8 safetensors checkpoint folders (NVFP4 and FP4-Local) as downloaded, with no
conversion, in a build with the `checkpoint` feature (the release has it). `-m` names the folder that holds
`model.safetensors.index.json`, and `--hf <repo>` names a repo that holds such a checkpoint and no GGUF set (a
`:<quant>` there is refused). Every part keeps its stored format. Running these folders on a DGX Spark, with the
slots and memory each start takes, is [`dgx-spark.md`](dgx-spark.md).

- **Where the experts go.** The load reads the PLE table from the folder on the NVMe tier. Either folder's plan splits
  the routed experts between the card and the host as a GGUF's does, and `BLOOMERY_RESIDENCY` runs over them. A plan
  that pages them to the NVMe tier reads each matrix's codes and block scales into the arena from the shards that hold
  them.
- **FP4-Local's scale store.** The FP4-Local folder stores its experts' block scales compressed, so a load whose host
  or residency machine reads an expert first decodes them once into the scale store, one file per folder generation
  under `BLOOMERY_SCALES` (`~/.cache/bloomery/scales` when unset). Later loads of the unchanged folder read that file.
  A load that holds every expert on its card writes no scale, but its open still sizes the file, so the store must be
  writable.
- **The MTP draft.** Unless `BLOOMERY_MTP_DRAFT` names a file, the draft is the folder's own MTP layer in its stored
  formats, under the rule of section 4. It borrows the target's embedding and vocabulary head, so the folder is the
  one download a drafted server needs. A folder whose MTP layer the draft does not read still serves: `mtp` is refused
  with the layer's refusal, and unset turns the draft off.
- **The `setup` line.** Before its address the seat prints the draft (`mtp`, `lookup` or `off`) and where it reads it,
  or why a rule turned it off, the slots, the context each serves and whether it was set, and the settings that
  change them.

### 9.3 Drafts by seat

- **V4.1 (DSpark).** `BLOOMERY_DRAFT=dspark` sits on the tier card under `--place bp`, else on the smallest visible
  card: the stage itself (which then holds fewer experts), or a smaller card that joins the plan to hold the draft and
  no expert. `-devd` or `BLOOMERY_DSPARK_CARD` names the card. `--parallel` above 1 with DSpark is refused.
- **The qwen3 seat.** Unset or `mtp`, `BLOOMERY_DRAFT` drafts a `qwen35moe` or `qwen35` file's own next-token layer on
  a whole-card load of one sequence; a set `mtp` is refused on a media load, a placed load and `--parallel` above 1, and
  unset goes off there with the reason on `load draft=off`. A `qwen3moe` file has no such
  layer and refuses `mtp`. `BLOOMERY_DRAFT=dspark` with `BLOOMERY_DSPARK_MODEL` naming an HF draft directory serves a
  whole-card `qwen35moe` target from one slot: RedHat's Qwen3.6 speculator carries its own reduced head, and
  MiniCPM-V 4.7's full-vocabulary draft borrows the target's Q6_K `output.weight` and `token_embd.weight` rows in
  place. The draft fits a target by its shapes, not its name, so fine-tunes with Qwen3.6's shapes (Sage-A1.5,
  Ornith-1.5) take it. A draft whose ids, mask or taps do not fit is refused before a weight is read.
  `BLOOMERY_DRAFT=lookup` serves the n-gram lookup draft on the same whole-card load.
- **The mistrallarge4 seat.** `BLOOMERY_DRAFT=lookup` serves the n-gram draft, one context a slot. `BLOOMERY_DRAFT=dspark`
  serves the DSpark draft of `BLOOMERY_DSPARK_MODEL` on the card `BLOOMERY_DSPARK_CARD` names (else the common placement
  rule picks), from one slot: `--parallel` above 1 and an image under it are refused by name. The draft's card bytes (its
  weights, the target's `output` it copies, its rings and a scratch allowance) are a reserve of the plan on its card
  before the context is fitted: the unset context is the largest that leaves the draft its room, and a `--ctx` that
  leaves none is refused by name with the largest that fits. Unset serves the plain path.

### 9.4 Nemotron

The nemotron seat serves Nemotron 3 Nano 4B (`--model nemotron -m PATH`), Nemotron 3.5 Bolt 74B-A9B and
Nemotron-3.5-Lightning 30B-A3B, one sequence: `--parallel` above 1, a placement of more than one card or with a tier
card, and a draft lever are refused. A Mamba-2 state cannot step back, so a request keeps what the slot holds only
when its prompt extends the whole held sequence; any other prompt resets the slot. The seat runs no prompt cache. A
Bolt or Lightning load the card cannot hold keeps the rest of its routed experts on the host, and pages past the
host's room on the NVMe tier (section 5.4).

### 9.5 Decision models

The decide seat answers SystemOne requests, one at a time, for two decision models: Clef-Flash (a joint schema head,
and images with an encoder) and lev (`--hf ggml-org/lev-GGUF:Q4_K_M`, a label readout; its file's
`qwen35.decision.type` picks it, and its request is rendered by the file's own `systemone` template).

```sh
curl -s http://127.0.0.1:8080/v1/systemone -d '{"state": "User: what is the weather in Seoul tomorrow? Tools available: web_search, calculator.",
  "questions": {"tool": {"type": "choice", "instructions": "Which tool should the agent call next?",
    "criteria": {"web_search": "Look something up online", "calculator": "Do arithmetic", "none": "Answer directly"}}}}'
```

**Two layouts of the Clef-Flash GGUF.** llama.cpp's `clef` layout carries the joint head inside the file (bartowski's
current upload), so `bloomery-serve --hf bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M` needs no head flag. The earlier
`qwen35` layout takes the release's head file: `--head joint_head.safetensors`, its `joint_head_config.json` beside
it. A `--head` beside a `clef` file replaces the file's own head.

`/v1/systemone` follows llama.cpp's decision server wire (`model` optional, `/v1/models`, 501 for video and for an
image when the seat has no encoder, upstream's `confidence` formula). A generative seat answers the same route too,
section 11.1.

### 9.6 The other launchers

The release ships `bloomery-serve` alone. A source build also has `generate_ds41` (token ids in, greedy ids out),
`generate_glm5next`, `generate_qwen3moe`, `bloomery-chat` (streams text) and the standalone servers
`bloomery-serve-ds41` and `bloomery-serve-qwen38` ([`BUILD.md`](BUILD.md)). They read the same placement flags and
variables, before any lever of theirs, and `--list-devices` prints the devices on each. Their model is
`$BLOOMERY_REF_MODEL`, which `-ngl N` compares with. Every seat and flag is its binary's `--help`.

| Launcher | `--cpu-moe`, `-ncmoe`, `-ot` | `-devd` |
|---|---|---|
| `bloomery-serve` (by seat) | the qwen38 seat | the V4.1 seat |
| `bloomery-serve-ds41`, `generate_ds41` | refused | accepted |
| `bloomery-serve-qwen38` | accepted | refused |
| `generate_qwen3moe` | on a qwen4exp file; refused on any other | refused |
| `generate_glm5next`, `bloomery-chat` | refused | refused |

`-ts`'s shares set a budget only on a load that names its card: `generate_qwen3moe` refuses them unless a flag that
names cards gives the load a `--place`. `-fitt` is read on every launcher. Each launcher refuses an argument it has no
flag for, by name, with its usage.

## 10. Images, video and embeddings

### 10.1 Image input

The V4.1, Qwen3.8, Qwen3.6, MiniCPM-V 4.7, Sage-A1.5, GLM-5.3 and Clef seats read images when they have a vision
encoder: `--mmproj <file>` names the model's encoder GGUF (on V4.1, `mmproj-DeepSeek-V4.1-Flash-BF16.gguf`). Under
`--hf` with no `--mmproj`, every image seat takes the repo's own projector, unless `--no-mmproj` says none. Qwen3.8
skips that projector when its MTP draft is on or its prompt is fed a step an id, so it needs `--mmproj <file>` for
images with MTP. The V4.1,
qwen38, qwen3 and GLM-5.3 seats keep that projector only when the text model keeps the same card experts and context
with it; otherwise one `vision` line gives its bytes and cost, and the seat serves text. A file you name with
`--mmproj` is never dropped. Qwen3-30B-A3B has no encoder and refuses `--mmproj`. A file of another encoder is refused.

- **Sending an image.** PNG or JPEG bytes inside the request: an OpenAI `image_url` part holding a `data:` URL, or an
  Anthropic `image` block with a base64 source. Any other URL is refused, not fetched.
- **Where the encoder sits.** It loads once at start and stays resident, on a card the plan leaves off, or as a named
  reserve the plan counts on the stage card. The `vision` line names the card and the bytes (V4.1's is 1.31 GB).
- **Caching.** On V4.1 an image span rides the prompt cache like its ids. The prepared-image cache keeps at most 512
  MiB per server, least recently used first out (a 1280×720 screenshot prepares to about 5.6 MB on V4.1 and about 11.3
  MB on the Qwen3-VL and GLM encoders [derived]). A cached image skips decoding but is still hashed.
- **Memory bound.** A chat request's new images, and every image of a batched embedding input, share a fixed 64 MiB
  decoded-RGB budget. Headers are checked before decoder output is allocated, and a request whose new images pass the
  budget is a named 400. The bound covers retained RGB and prepared patches, not total process memory.
- **Without an encoder** `/props` answers `vision: false`. A generative seat answers a request with an image with a 400
  (`image input is not supported: this server reads text only`); the Clef decide seat with a 501 and llama.cpp's hint.
  Lev takes no image: `--mmproj` beside it is refused, and a request with an image is a 501.
- **Qwen3.8** drafts on an image load by the same rule as a text load, and refuses `BLOOMERY_PREFILL=steps` beside
  `--mmproj`. A sequence that holds an image is kept out of the prompt cache.

### 10.2 Video input

A model with a video rule reads a clip when `ffmpeg` and `ffprobe` (5.1 or newer) are on `PATH`; the server looks once
at start, prints one `video:` line and answers `/props` with `modalities.video` true. A clip is an `input_video` part
(`{"data": <base64>, "format": "mp4"}`) or a `video_url` part holding a `data:video/*;base64,` URL; any other URL is
refused, and so is a body over 64 MiB. The server writes the clip to a private (0600) temp file and reads it only in
one of these containers: MP4/MOV, Matroska/WebM, AVI, FLV, MPEG-PS, MPEG-TS and Ogg. It plans the frames by the model's
rule (rate, frame bounds, group size, token cap) under the context the request's text leaves, and has ffmpeg decode
only those frames. Without ffmpeg, a clip is a 400 that names the fix.

### 10.3 Embeddings

`POST /v1/embeddings` (OpenAI's shape: `encoding_format` `float` or `base64`, `dimensions`) and `/embedding` or
`/embeddings` (llama-server's own array) answer from an embedding model held beside the chat model: `--embd-model FILE`
or `--embd-hf <repo>[:<quant>]`. With none they answer 501 `not_supported_error`.

- **Models.** The `gemma-embedding2` feature serves EmbeddingGemma v1 (`gemma-embedding`, 300M, up to 2048 tokens) and
  EmbeddingGemma 2 (up to 8192 tokens), both Q8_0, on a separate thread and card stream. Both serve 128, 256 and 512
  MRL cuts and the full width, cut before normalization. A file whose output width is below 128 is refused.
- **Input.** `input`, or `content`, as llama-server reads it: a string, a list of ids, a list of strings and ids, or a
  list of these. `embd_normalize` is `-1` (none), `0` (int16 range), `1` (taxicab), `2` (euclidean, the default) or a
  higher p-norm. An input past the model's context is a 400 `exceed_context_size_error`; there is no batch limit.
- **`prompt_name`** selects a prefix by exact name from the model's `config_sentence_transformers.json` `prompts` map,
  such as `query` or `document`, read at startup beside `--embd-model` or from the `--embd-hf` repo. The configured
  bytes are prepended to each text input with no separator. Without `prompt_name`, input is unchanged, even if the
  config sets `default_prompt_name`. An unknown or non-string name, and token-id or media inputs with it, are a 400.
- **Where it sits.** On the smallest card outside the chat plan, else as a named `embedding model` reserve on the stage
  card, which lowers the chat expert budget. `--mmproj` and `--embd-model` on the same off-plan card do not count each
  other's bytes; the second open fails by name if the card is short.
- **Names.** The response `model` is the embedding file name, listed separately in `/v1/models` with `meta.n_embd`,
  `meta.n_ctx` and `meta.prompt_names`; `/props` adds `embedding_model`. On `/v1/embeddings`, a `model` that is not the
  embedding model's name is a 400; `/embedding` and `/embeddings` ignore `model`.
- **`--embd-cache-bytes N`** turns on an LRU cache of pooled vectors and image tower outputs with one byte budget (off,
  `0`, by default). Audio inputs do not use the vector cache.

### 10.4 EmbeddingGemma 2 images and audio

Add these flags to the chat model's command (the release build has the features):

```text
--embd-model embeddinggemma-2-Q8_0.gguf
--embd-mmproj mmproj-embeddinggemma-2-BF16.gguf
--embd-ctx 8192
```

`--embd-hf` selects the text file and the BF16 projector from one repository; `--embd-no-mmproj` turns that projector
off. A local `--embd-model` loads images only with an explicit `--embd-mmproj`. The projector's vision matrices must be
BF16. With `clip.audio.projector_type=gemma4a`, the side seat also loads the audio tower, and an embedding input may
carry decoded f32 PCM audio.

Send one object per input inside OpenAI's `input` array:

```json
{
  "input": [
    {
      "content": [
        {"type": "text", "text": "A description of this picture:"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }
  ],
  "dimensions": 256
}
```

Image parts stay in input order. Each image is resized with mtmd's aspect-preserving canvas, bicubic filter and
centered black padding, within 70 to 1120 pooled image tokens (`--embd-image-min-tokens`, `--embd-image-max-tokens`).
`<|image>` and `<image|>` wrap each image's rows in the single bidirectional pass, as in the pinned mainline server.
Video frames go in as a first text part `Video:` followed by `image_url` frames, or as the video parts of section 10.2.

To compare a running server with saved llama-server vectors:

```sh
python3 tools/ref/emb2/server_vectors.py compare --url http://127.0.0.1:8080 \
  --set "$BLOOMERY_DATA/ref_gemma_embedding2_server" \
  --image free-448x448=/path/to/free-448x448.png --mix-image free-448x448
```

Add `--ids-dir "$BLOOMERY_DATA/ref-tokenizer-gemma4"` and `--case NAME` for text cases. Each row prints cosine and
maximum absolute difference; the tool applies no acceptance band. The gates are `just gate-gpu-gemma4v-tower` and
`just gate-gpu-gemma-embedding2-e2e`.

## 11. API reference

### 11.1 Routes

- **llama-server's routes**: `/completion`, `/apply-template`, `/tokenize`, `/detokenize`, `/slots`, `/metrics`,
  `/props`, `/health`, streaming, `cache_prompt`. OpenAI SDKs, curl and llama-server tutorials apply as they are.
- **OpenAI chat completions**: `/v1/chat/completions`, with tool calls and `reasoning_content` (section 11.3).
- **OpenAI text completions**: `POST /v1/completions` runs `/completion` in OpenAI's shape (`object:
  "text_completion"`, `choices`, `usage`, a stream ending in `data: [DONE]`). One prompt a request: a `prompt` array
  of several prompts, `echo`, a non-empty `suffix`, `best_of > 1` and `logprobs` are refused by name.
- **Several choices**: `n` on `/v1/chat/completions` and `/v1/completions` answers that many choices of the prompt.
  Choice `i` samples with the request's `seed` plus `i`, so choice 0 is the answer `n` 1 gives. The choices queue
  together, so `n` is at most the slots plus the queue depth. `usage` counts the prompt once; `timings` are the first
  choice's. Other routes refuse `n` above 1.
- **OpenAI Responses**: `POST /v1/responses` and `/v1/responses/input_tokens`. A response is kept unless the request
  sends `store: false`: `GET /v1/responses/{id}` reads it, `DELETE` drops it, and `previous_response_id` continues its
  conversation (its input and output, not its `instructions`). `background: true` answers `queued` at once; poll `GET`,
  or end it with `POST /v1/responses/{id}/cancel`. The server keeps at most 1,024 responses and 256 MiB of JSON text,
  oldest finished first out, and loses them all when it stops. A background request gets 503 with `Retry-After` while
  the running ones fill either bound. A `namespace` tool (Codex's `multi_agent_v1`) is flattened: each function inside
  it serves under its bare name, and its calls carry the namespace in the `function_call` item's `namespace` field.
  `conversation`, an `item_reference` item and `background: true` with `store: false` or `stream: true` are refused.
- **Anthropic Messages**: `POST /v1/messages` and `/v1/messages/count_tokens`, as llama-server serves them.
- **Chat token counts**: `POST /chat/completions/input_tokens` (and `/v1/`) answers `{"input_tokens": N}` for the
  same chat request, an image as its span; `max_tokens` is not needed.
- **Prompt ids**: `POST /apply-template` takes a chat request's `messages`, tools and template options and answers
  `{"prompt": "..."}`. `"return_prompt_ids": true` adds `"prompt_ids"` with the same guarded tokenization as chat;
  tokenizing the returned `prompt` with `/tokenize` is not the same, because that route parses special-token
  spellings. Media placeholders are not expanded in either field.
- **Model lookup**: `GET /v1/models/{model}` (and `/models/{model}`) answers the one `/v1/models` entry of that name;
  a name the server does not serve is a 404 naming the models it does. For Ollama clients, `GET /api/tags` answers the
  model list in Ollama's shape, and `POST /api/show` with `{"model": NAME}` the chat model's template, format,
  architecture and quant, the served context as `<arch>.context_length` and its `capabilities` (`completion`,
  `tools`, `vision`; `embedding` for the embedding model). A decision model's server answers `/api/tags` and the
  lookup but not `/api/show`. Ollama's other routes are not served.
- **`/v1/systemone` on a generative seat**: the decision models' request, answered by the general model. The schema
  text goes in the system message and the state in the user message, and the reply is held to a JSON object by the
  structured-output path. Each question's scores are the logits of the token its options start with, read where the
  model picks, and `choice`, `probabilities`, `confidence` and `score` follow the decision models' formulas. Options one
  token does not tell apart are asked by letter (`A = ab`, `B = ac`); more such options than 26 letters is a 400. A
  request may add `cache_prompt` (default `true`).
- **Compressed bodies**: a body sent with `Content-Encoding: gzip` (or `x-gzip`) or `deflate` is read decoded. The 64
  MiB body limit holds for the bytes sent and for the decoded body. A body that does not decode is a 400; any other
  coding, or a list of two or more, is a 415 whose `Accept-Encoding` names `gzip, deflate`.
- **Errors** answer llama-server's object, `{"error": {"code": 400, "message": "...", "type":
  "invalid_request_error"}}`, plus OpenAI's `param` when the error is about one request field (its top-level name). The
  Anthropic Messages API keeps its own error object. A route the server does not have is a 404 that starts with the
  method and path.

### 11.2 Request fields

A request field is read, refused or ignored.

- **Read.** Sampling fields such as `temperature` reject wrong types and non-finite values; `n_predict` and
  `repeat_last_n` keep llama-server's count limits. `null` is unset.
- **Refused by name** (a 400 naming the field): `samplers` in another order than llama-server's default for the stages
  that are on, or naming a sampler this server lacks; `typical_p`, mirostat, `dry_*`, `xtc_*`, `top_n_sigma` and
  `dynatemp_*` when turned on; `n_predict` or `max_tokens` outside -1 to i32::MAX; a non-empty `grammar`;
  probabilities on a route that renders none (`/completion` takes `n_probs`, `/v1/chat/completions` takes `logprobs`
  and `top_logprobs`); `n` above 1 outside the routes of section 11.1. `/tokenize` refuses a missing or non-string
  `content`.
- **Ignored.** A field the server does not know (section 2).
- **`logit_bias`**: OpenAI's object from token ids to a number in -100 to 100, added to that id's logit before every
  pick, on `/completion`, `/v1/completions`, `/v1/chat/completions` and `/v1/responses`. It applies to greedy, sampled
  and structured-output requests, with or without a draft. Under `ignore_eos` the stop ids stay at -inf. Under a schema
  the bias reorders the ids the schema allows and lets in no other. `logprobs` report the model's own row, before the
  bias. Other forms (an array of pairs, `false`, a token's text), a key that is no id, a value out of range and an id
  named twice are refused.
- **Special-token spellings in client text are text.** llama-server parses special tokens in the rendered chat prompt,
  so a message that spells `<|im_start|>` writes a turn marker. This server keeps what a client sends as text: a
  special-token spelling in a user, system or tool message, a role or name, a tool call's name or arguments, a tool's
  description or schema, or a `chat_template_kwargs` value tokenizes as its characters, while the tokens the template
  writes stay tokens. An assistant message's content and reasoning keep the special ids the model printed. The marking
  is in `crates/serve/src/guard.rs`; a template that slices a marked text through its mark is a 400 naming it. Not
  covered: a template that builds a spelling from its own pieces around a client string. A `/completion` prompt is
  the client's own tokenization and still parses special tokens.

### 11.3 Tool calls and reasoning

**Tool calls on every generative model.** DeepSeek-V4.1-Flash (DSML), GLM-5.3-Flash (its XML), Qwen3-30B (Hermes
JSON), Qwen3.6-35B and Qwen3.8-Flash-Next (the `<function=…><parameter=…>` markup, a value on lines of its own) and
MiMo-V2.6 (the same tags, a value bare between them) parse into `tool_calls` and `tool_use` blocks. Which framing a
template uses is read from the call it renders, and a `tool_choice` that forces a call writes it in that framing.

- **A cut call.** A generation the token limit cuts inside a call answers 200 with `finish_reason` `length`: the
  call's text as `content` while its name is not whole, and on the chat route the call with the arguments read so far
  once it is. A generation that ends on its own inside a call is a 500 naming the call.
- **DSML spellings.** DeepSeek-V4.1 writes its DSML tags with a space after the token (`<｜DSML｜ invoke>`), V4
  without. The V4.1 GGUFs carry V4's template, so a `deepseek41` file is served with that template respelled as V4.1's,
  and one line at start says so. A template given by `--chat-template-file` (the V4.1, Qwen3, Qwen3.8, GLM, MiMo and
  Nemotron seats take it) is respelled the same way. The file is read once at startup.
- **Streaming.** A streamed `/v1/chat/completions` sends a call as the model writes it, in OpenAI's shape: the first
  delta carries `index`, `id`, `type` and `function.name`, and `function.arguments` fragments follow. A text value is
  released as it arrives up to a tail that could still become a closing tag; a number or JSON value when it is
  complete. Markup that stops being a call after its name went out ends the stream with an error event naming the
  call. `/v1/messages` and `/v1/responses` streams send a call when it closes.
- **Markup tokens.** A think tag or a Qwen3-Coder, Hermes, GLM or Solar call tag held as one special token is markup
  only where the model generated that token. The same characters in ordinary tokens stay reasoning or content. A call
  opened by a special token needs a special closing token too. A Hermes `<tool_call>` opener at the end of output is an
  unterminated call. `bash tools/ref/serve-control-markers.sh URL OUTPUT_DIR` saves reproductions of these cases
  against a running server.
- **Watched calls.** Under `tool_choice` `"auto"` (or none) with tools, the reply is free until the model writes its
  format's call opener; from there the call follows the program a forced call compiles, until it closes. A tool schema
  that is invalid is a 400; a valid one outside the subset of section 11.4 leaves that request's calls unwatched. Muse
  Spark's replies and Qwen3-Coder's bare first call are not watched. With a draft on, the watched reply keeps the
  draft's passes.

**Reasoning fields.**

- **`reasoning_budget`** (on `/v1/chat/completions` and `/completion`): llama-server's think span budget in generated
  ids, `0` closing it at once, `-1` or absent unrestricted, ignored when the template already closed the span. A
  Qwen3-Coder `<tool_call>` ends the span too. A finite budget beside a forced `tool_choice` is a 400.
- **`reasoning_loop`** (on every generating route): `true` ends a generation whose open think span repeats a cycle of
  up to 1,024 ids over at least 256 ids and one whole cycle. It ends with the reasoning so far and `finish_reason`
  `reasoning_loop` (`stop_reason` on `/v1/messages`, `stop_type` on `/completion`; on `/v1/responses` `status`
  `incomplete`). Absent, `null` or `false` turns it off; another type is a 400.
- **`reasoning_effort`** takes OpenAI's words (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`) and `max` (V4.1's
  maximum-effort header), on `/v1/chat/completions`, `/apply-template` and the Responses API's `reasoning.effort`;
  another word is a 400. `"none"` turns thinking off before the template, as llama-server does: the template reads
  `enable_thinking: false`, and the whole output lands in `content`. A template with no thinking switch but an effort
  header (GLM-5.3's) gets `reasoning_effort: "low"` instead, and one line at start says so. Every other word asks for
  reasoning: a template whose thinking is off until `enable_thinking` is set (V4.1's) gets `enable_thinking: true`, and
  `max` also adds its maximum-effort header. A `chat_template_kwargs` `enable_thinking` in the same request wins.
  `/props`' `thinking_off` names the template's case: `"switch"`, `"low_effort"` or `"neither"`. Unlike llama-server,
  a request that names neither field keeps the template's own default (thinking is off for V4.1).
- **`developer` messages** reach the template as `system`, as llama-server maps them, except on GPT-OSS's template.

### 11.4 Structured output

Structured output is served for `response_format` of type `json_schema` or `json_object` and for `tool_choice`
`"required"` or a named function (Anthropic `any` and `tool`, Responses `text.format`), on every generative seat.

- **The schema subset.** An object (its `required` properties in order, then each optional one in order, each of which
  may be left out), an array of `items`, `anyOf`, a `$ref` into the schema itself (`#/$defs/...`), string, integer and
  boolean enums, `const`, booleans, `null`, integers, numbers and free strings with an optional `maxLength`.
  `json_object` is any JSON object with at least one member. The JSON is compact with one space after `:` and `,`, and
  a free string takes no escape.
- **Sampling.** `temperature` defaults to 0, so the request decodes greedily over the ids the schema allows. A positive
  `temperature` or a penalty draws with the request's sampler from the masked row, so `top_k`, `top_p`, `min_p` and
  `seed` apply. Arrays, numbers and `json_object` have no length bound, so the token or context limit can leave
  incomplete JSON with a `length` finish.
- **Thinking.** A JSON `response_format` closes an open think span at once and writes the document after it. A forced
  call lets the model reason freely until it generates the span's close itself, and the call starts after that; a
  model that ends before closing the span is a 500 naming the missing call.
- **The jump arm.** By default each run of ids the schema fixes is fed as one call instead of one step each, so a fixed
  run costs one engine call. `BLOOMERY_SCHEMAJUMP=off` returns to stepping every id. The two arms give the same text and
  finish reason; their token lists can differ in the last id, and a decision inside a rounding tie can differ. A
  request that asks for `logprobs` or sets a penalty takes the step arm. `timings.constrained` counts the arm's work
  (`spans`, `span_ids`, `decisions`, `free_steps`, `next_calls`, `prefill_calls`, `next_after_calls`, `lead`, `tail`).
- **Refused by name**: any other schema keyword (`oneOf`, `allOf`, `pattern`, `minLength`, `minItems`, `minimum`,
  `minProperties`, a list of types, ...), a `$ref` outside the schema or a recursive one, a schema that writes out to
  more than 4,096 schemas or past 128 nested levels, a schema or tools that write out past 64 MiB, a program of more than
  16 MiB of fixed text, a forced tool's `parameters` with a root `anyOf` or `$ref`, an `anyOf` with a text option in a
  tool argument written raw, a top-level `json_schema`, `grammar`, a JSON `response_format` beside a forced call,
  `ignore_eos`, `reasoning_budget`, an enum whose options tokenize so that one is a prefix of another, and a constant,
  option or property name whose text tokenizes to a stop id.
- **`<` in tool calls.** A free JSON string inside a tool call takes no token with a `<`, and neither does one in a
  `response_format` reply that comes with tools, since the reply is scanned for a call. A `<` in a constant, option or
  property name is written as `\u003c`. A raw string in a tool call's XML markup is one line (Qwen) or holds no `<`
  (GLM, DSML); a raw constant or option with a `<` is a 400.

## 12. Limits

- **Hardware.** sm_86 or newer NVIDIA GPUs and driver R580 or newer (CUDA 13). The x86-64 archive carries sm_86 PTX,
  compiled for the card at the first start; its sm_120a twin is the installer's pick where every GPU is compute
  capability 12.0 (the RTX 50 series). The GB10 archive (Linux aarch64) carries sm_121a code compiled for a DGX Spark.
  Another aarch64 machine builds from source with `BLOOMERY_ARCH` setting the CUDA arch. The AVX2 kernels are x86-64
  only; a quant type without an aarch64 kernel runs through a scalar mirror there.
- **One model a server**, and one expert tier card at most (`--place bp`).
- **GLM-5.3** holds at most 16,384 positions a slot: a `--ctx-size` whose slot share passes it is refused at start,
  and a longer prompt answers `exceed_context_size_error`.
- **Kimi K3 and K3.1 Pro** load at `--place bp` only, one slot. K3.1 Pro holds `--ctx-size` to 2,048 positions: the
  reader runs the lightning indexer's selection up to its top-k, and a larger context is refused at the start by name.
- **Small cards.** On a 12 to 16 GB card the split Qwen3 load's default context fits to the largest the plan holds.
  Agent clients with long system prompts want the whole context for one request there (`--parallel 1`).
- **Repeatable tokens.** With adaptive residency on, a request's tokens follow the placement its passes ran on, and
  that placement follows the history of passes. The same history gives the same tokens bit for bit; another history
  can reword an answer at a near tie, since an expert on the card and on the host round their activations differently.
  A placed load's slot count changes how many experts sit on the card (8 fewer on V4.1 at `--parallel 2`), so two
  servers' answers can differ at a near tie with residency on or off.
- **Clef** reads text states and, with an encoder, images, from backbone weights of Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 and F32
  (not the IQ types, Q2_K, Q4_0 or Q4_1). A head inside a `clef` file is read as stored (bartowski's: Q8_0 matrices,
  which move a probability by under 3e-4 against the release's bf16 head on the eight reference requests).
- **V4.1** decode is bound by host memory bandwidth in each step's expert part and by the card's serial work before it;
  a long V4.1 prompt is bound by the card.
- **Toolchain.** A source build needs the pinned toolchain and cuda-oxide revision ([`BUILD.md`](BUILD.md)).

In progress: serving on N cards for every model; Clef from the IQ and Q2_K files; DeepSeek-V4-Flash-0731.
