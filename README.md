<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img src="docs/assets/logo.svg" width="128" alt="bloomery">
  </picture>
</p>

<h1 align="center">bloomery</h1>

<p align="center">
  <b>A fast inference engine for mixture-of-experts models, built for offloading.</b><br>
  Written in Rust from the HTTP server down to the CUDA kernels. A drop-in
  <a href="https://github.com/ggml-org/llama.cpp/blob/master/tools/server">llama-server</a>.
</p>

- **MoE models larger than the card.** Routed experts live on the GPU and in host RAM; the engine counts its own
  routing and moves the experts it calls most onto the card while it runs. On one RTX 3090 24 GB beside a 256 GB
  host, DeepSeek-V4.1-Flash `Q3_K_M` (347 GB) serves two requests at 34.4 tok/s in total and GLM-5.3-Flash at 26.9;
  Qwen3-30B runs whole on the card, two requests at 254 tok/s in total ([Numbers](#numbers)). Qwen3.6-35B fits a 12–16 GB card with
  `--place a`.
- **Several requests, one pass.** With `--parallel 2` two busy streams run through the model together: 25–37 %
  more tokens a second in total than one stream, on V4.1, GLM-5.3, Qwen3.8 and Qwen3-30B
  ([measured on the A6000](https://github.com/midagedev/rig-log/blob/main/log/2026-10-06.md#rel021-slots)).
- **llama-server compatible.** The same HTTP API and GGUF files, and llama-server's spelling for the flags both
  have: your OpenAI client works unchanged.
- **Rust all the way down.** Every CUDA kernel is written in Rust and compiled with
  [cuda-oxide](https://github.com/NVIDIA/cuda-rust).

```sh
brew install midagedev/tap/bloomery        # Linux x86-64, NVIDIA sm_86+
bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M --port 8080
```

Models: DeepSeek-V4.1-Flash, GLM-5.3-Flash, Qwen3.8-Flash-Next, Qwen3.6-35B-A3B, Qwen3-30B-A3B, and the decision
model Clef-Flash — [what each needs](#hardware) · [all install channels](#install) · [numbers](#numbers)

## llama-server compatibility

Your OpenAI client works unchanged. bloomery speaks llama-server's HTTP API and reads the same files:

- **The API**: `/v1/chat/completions`, `/completion`, streaming, tool calls, `reasoning_content`,
  `cache_prompt`, `/tokenize`, `/detokenize`, `/slots`, `/metrics`, `/v1/models`, `/props`. OpenAI SDKs, curl,
  and llama-server tutorials apply as they are.
- **`timings`** carry llama-server's fields and one of ours, `cache_ms`: the prompt cache's work before the
  prompt (the slot's state saved, a cached state put back, the cut), which `prompt_ms` does not count.
- **`reasoning_budget`**, per request on `/v1/chat/completions` and `/completion`: llama-server's
  `--reasoning-budget` — the think span's budget in generated ids taken while the span is open, `0` closing it at
  once, `-1` or absent unrestricted, and silently ignored when the template already closed the span.
- **The files**: the GGUF uploads as downloaded, the same quantizations, the chat template read from the file,
  and a tokenizer bit-identical to `llama-tokenize`.
- **The flags**: `-m`/`--model-file`, `--hf <repo>[:<quant>]` (download, resume, sha256 check, never fetched
  twice), `--parallel/-np`, `--queue-depth`, `--cache-ram`, `--ctx/--ctx-size` (the qwen3, qwen38 and glm
  seats default it to what the card's free memory fits — the trained context capped to it on the qwen3 seat's
  whole-card loads, the largest context that keeps the card's experts on the qwen38 seat, the placed plan's
  margin on the glm seat), `--cache-type-k f16|q8_0` (the qwen3 seats; llama-server's spelling, also the
  `BLOOMERY_QWEN3_KV` lever — q8_0 halves the KV bytes a position, so the auto context nearly doubles on the
  same card; `--cache-type-v` does not exist, both planes quantize together), `--host/--port`,)
  `--alias`.
- **Two-card serving like `-ts`**: `--place a` for the largest visible card, `--place bp` to add the next one
  as an expert tier, or a list (`0+1`) by CUDA index.
- **Clef-Flash's `/v1/systemone`** follows llama.cpp's decision server wire (`model` optional, `/v1/models`,
  501 for images and video, upstream's `confidence` formula).

Where it ends today: one model a server, sm_86+ GPUs, and the seats' own defaults where llama-server has none
(see [Limits](#limits)).

## Install

Prebuilt, one binary, no CUDA toolkit and no Rust toolchain — Linux x86-64 with AVX2/FMA (x86-64-v3), glibc
2.34+, an NVIDIA GPU of compute capability 8.6 or newer and its driver. On Windows, run it inside WSL2
(`wsl --install`, reboot, then the same commands in the WSL shell). What each model needs in GPU, RAM and
disk: [Hardware](#hardware).

Three ways to get it:

```sh
# Homebrew (Linux x86-64)
brew install midagedev/tap/bloomery

# or the installer: latest release, sha256-checked, symlink into ~/.local/bin
curl -fsSL https://raw.githubusercontent.com/midagedev/bloomery/main/tools/release/install.sh | sh

# or Docker (needs the host's NVIDIA Container Toolkit; model downloads live in the bloomery-cache volume)
docker run --gpus all -p 8080:8080 -v bloomery-cache:/root/.cache/bloomery \
  ghcr.io/midagedev/bloomery --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M
```

The plain tarball, when you want to lay it down yourself:

```sh
tar -xzf bloomery-0.2.1-linux-x86_64-cuda-sm86.tar.gz && cd bloomery-0.2.1-linux-x86_64-cuda-sm86
bin/bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M --port 8080
```

[bloomery 0.2.1](https://github.com/midagedev/bloomery/releases/tag/v0.2.1) · [all releases](https://github.com/midagedev/bloomery/releases)

From source: [`docs/BUILD.md`](docs/BUILD.md) (Linux x86-64, CUDA 13.3, LLVM/Clang 21, `cargo-oxide` from our
cuda-oxide fork).

## Use

One binary, five seats. With `-m` or `--hf` and no `--model` word, the file's architecture picks the seat
(`--help` lists them, one line each with a working `--hf` example):

| Seat | Serves | One command |
|---|---|---|
| `ds41` | [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | `bin/bloomery-serve --hf vcruz305/DeepSeek-V4.1-Flash-GGUF:Q3_K_M` |
| `glm` | [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | `bin/bloomery-serve --hf unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL` |
| `qwen38` | [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | `bin/bloomery-serve --hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL` |
| `qwen3` | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | `bin/bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M` |
| `decide` | [Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) (a decision model) | `bin/bloomery-serve --hf bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M` |

The generative seats answer `/v1/chat/completions` and `/completion`, streaming, and reuse a cached prompt
prefix:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}'
```

The decide seat answers SystemOne requests, one at a time, text states only:

```sh
curl -s http://127.0.0.1:8080/v1/systemone -d '{"state": "User: what is the weather in Seoul tomorrow? Tools available: web_search, calculator.",
  "questions": {"tool": {"type": "choice", "instructions": "Which tool should the agent call next?",
    "criteria": {"web_search": "Look something up online", "calculator": "Do arithmetic", "none": "Answer directly"}}}}'
```

**Concurrent requests.** Every generative seat takes `--parallel N` (unset, 2). The N slots are resident sequences
inside the one model, switched by pointer exchange: nothing parks, and each round advances every busy slot by one
token or one drafted pass. Where the body runs several slots' rows as one pass, the round reads the weights once for
all of them: on a whole-card Qwen3-30B, on V4.1 (two slots' rows a pass), and on GLM-5.3 and Qwen3.8, where a greedy
request's drafted verify window rides the same pass (two windows a pass; with the draft off, the plain rows). A
placed Qwen3-30B (`--place a`), Qwen3.6, a sampled request on a drafting GLM-5.3 or Qwen3.8 load, V4.1 with the
lookup draft and Qwen3.8 under `--place bp` step their slots in turn, a select and a step a slot each round. At a
fixed expert placement each request answers the tokens of its run alone on the same server; under adaptive residency
the placement follows every stream's passes (see [Limits](#limits)). The context splits as llama-server splits it
with `-np N` and no `-kvu`: `--ctx-size` (or the automatic choice) is the total and each slot holds `total / N` rows,
except on V4.1, where every slot holds the whole context; `--parallel 1` keeps one sequence with the whole context.
The plan counts every slot, so the slots' caches together never pass what it holds, and `--park-ram` is refused by
name. A new request's prompt runs in one call between rounds, and the other streams wait for it.

**Small cards.** Qwen3.6 and Qwen3-30B at `Q4_K_M` (19-21 GB) run whole on a 24 GB card, or on a 12-16 GB card
with `--place a`: the routed experts go to the CPU. With no `--place` at all, a file that does not fit the
card's free bytes plans that split itself and says so on its plan line.

**A plan that fits the card you have.** The load reads each card's free bytes and the host's available RAM
before anything uploads; the expert share sizes to them, and a plan that cannot fit is refused by name with
every term (dense, KV, context, scratch, margin) and the processes holding the card — before minutes of
loading, not after.

**Speculative decoding.** `BLOOMERY_DRAFT=mtp` drafts with the model's own MTP head (Qwen3.8 and GLM-5.3, on by
default in their servers); `BLOOMERY_DRAFT=dspark` drafts V4.1 with DeepSeek's DSpark head on a second card
(`--place bp`).

**Adaptive residency.** V4.1, Qwen3.8 and GLM-5.3 count their own routing as they run and swap routed experts
between the card and the host between steps; a prompt call streams its hottest host experts onto the card, so
decode starts warm (on by default under `--place a`/`bp`).

**The engines behind the server.** `generate_ds41` takes token ids and prints greedy ids; `bloomery-chat`
streams text; `bloomery-serve-ds41` and `bloomery-serve-qwen38` are the standalone V4.1 and Qwen3.8 servers.
Every seat and flag is its binary's `--help`.

## Models

| Model | Recommended file | How it runs |
|---|---|---|
| DeepSeek-V4.1-Flash | [`Q3_K_M`](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF) (vcruz305) | GPU + CPU experts; 256 GB of RAM |
| GLM-5.3-Flash | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) (unsloth) | GPU + CPU experts; sparse attention past 2,051 positions |
| Qwen3.8-Flash-Next | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (unsloth) | GPU + CPU experts |
| Qwen3.6-35B-A3B | [`Q4_K_M`](https://huggingface.co/lmstudio-community/Qwen3.6-35B-A3B-GGUF) (lmstudio-community) | whole on one 24 GB card, or `--place a` on 12-16 GB |
| Qwen3-30B-A3B-Instruct-2507 | [`Q4_K_M`](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) (unsloth) | whole on one 24 GB card, or `--place a` on 12-16 GB |
| Clef-Flash | [`Q8_0` to `Q3_K_S`](https://huggingface.co/bartowski/Cloudflare_clef-flash-GGUF) (bartowski), plus the release's [head file](docs/BUILD.md#clef-flash) | whole on one GPU, its head on the CPU |
| DeepSeek-V2-Lite-Chat | [`Q3_K_M`](https://huggingface.co/mradermacher/DeepSeek-V2-Lite-Chat-GGUF) (mradermacher) | CPU or GPU; the first model, still gated |

Each file is the public upload as downloaded; the downloads are checked against the uploads' sha256. How close
each Clef quantization answers to the BF16 release, per question: [agreement table](https://huggingface.co/midagedev/clef-flash-bloomery).
The DSpark draft for V4.1 is converted from DeepSeek's checkpoint (the converter in llama.cpp's V4.1 pull
request).

## Numbers

Two requests at once (`--parallel 2`, both busy) after a 512-token prompt, decode tok/s of both together. The host
is the development machine's: a 32-core AVX2 CPU with 8 DDR4 channels and 256 GB of RAM. The RTX 3090 runs at a
250 W cap (a stock 3090 draws 350 W). Each number comes from the runners in `tools/ref/` under the quiet-machine
protocol ([conditions](https://github.com/midagedev/rig-log/blob/main/log/2026-10-06.md#num3090)); one-request rows, the A6000's own rows, the measured history and the other
engines' rows live on [rig-log's bench page](https://github.com/midagedev/rig-log/blob/main/docs/bloomery-bench.md).

**One RTX 3090 24 GB**

| Model | Where it runs | Two requests, tok/s in total | Prompt tok/s (P = 512) |
|---|---|---|---|
| Qwen3-30B-A3B `Q4_K_M` | the whole model on the card | 254.0 | 6,482 |
| Qwen3.6-35B-A3B `Q4_K_M` | the whole model on the card | 193.9 ¹ | 5,834 |
| Qwen3.8-Flash-Next `UD-Q4_K_XL` | card + CPU | 60.7 | 486 |
| GLM-5.3-Flash `UD-Q4_K_XL` | card + CPU | 26.9 | 179 |
| DeepSeek-V4.1-Flash `Q3_K_M` | card + CPU | 34.4 ² | 199 |

**RTX A6000 48 GB + RTX 3090 24 GB (`--place bp`)**

| Model | Two requests, tok/s in total | Prompt tok/s (P = 512) |
|---|---|---|
| DeepSeek-V4.1-Flash `Q3_K_M` | 55.9 | 206 |
| GLM-5.3-Flash `UD-Q4_K_XL` | 35.3 | 223 |
| Qwen3.8-Flash-Next `UD-Q4_K_XL` | — ³ | — ³ |

- Qwen3.8 and GLM-5.3 ran with the MTP draft off and adaptive residency on.
- ¹ Qwen3.6 runs two requests in turn today, so its total is one request's rate.
- ² A fixed placement with adaptive residency off: a server with the 3090 alone turns residency on, which this row
  does not count yet.
- ³ On two cards Qwen3.8 runs one-row steps only: two requests take turns and the prompt runs a position at a time.
  Batched passes over the second card are in progress.

Each of two requests runs at 0.62–0.68× its speed alone, so one user waits longer and the machine serves more
([A6000](https://github.com/midagedev/rig-log/blob/main/log/2026-10-06.md#rel021-slots)).

A first start compiles the GPU code for the card (tens of seconds); later starts take seconds. A warm V4.1
load takes about 16 s.

**Other engines.** The bench page also holds rows measured side by side with llama.cpp, mistral.rs and
exllamav3 on the same card in the same window, each with its build and flags; on some rows bloomery is ahead
and on some it is behind. They are not a claim about the other engines: each ran at the fastest flags we found,
which is not a proof of its best, and some rows did not hold both engines to the same conditions: in the
V4.1 rows of 2026-09-28, llama.cpp read the engram table cold on new prompt ids while bloomery's prompt repeated
one ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-27.md#v41-xeng)). If you know faster flags for a row, please open an issue.

## Why it is fast

Each point names the measurement that isolates it: the same engine, or a bench of that one path, with and
without that one choice, on the development machine. A point with no such measurement carries no number. The language is not one of the reasons: the
kernels are Rust (274 of them, through [cuda-oxide](https://github.com/NVIDIA/cuda-rust),
[how](docs/cuda-oxide.md)), but the speed comes from the choices below.

- **Experts move to where they are used.** V4.1, Qwen3.8 and GLM-5.3 count their own routing as they run and
  swap routed experts between the card and the host between steps; a prompt call streams its hottest host
  experts onto the card, so decode starts warm. V4.1 `Q3_K_M`, 96 decode steps after a 512-token prose prompt:
  29.48 tok/s with it off, 35.70 with the swap rule, 43.86 with the prompt call's streaming as well (the
  default). The card served 18 % of the routed calls with it off; with streaming it served 57 % over the first
  16 steps and 71 % over the last 16 ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-30.md#residency-clip)).

<p align="center">
  <a href="https://github.com/midagedev/rig-log/blob/main/assets/residency-explainer-v5.mp4">
    <img src="https://raw.githubusercontent.com/midagedev/rig-log/main/assets/residency-explainer-v5-poster.png" width="420" alt="Adaptive expert residency explainer (42 s video)">
  </a>
</p>

- **The card and the CPU work on a prompt at once (V4.1).** A prompt runs in batches of up to 512 positions:
  every routed expert a batch calls is read once for the whole batch. Batching took P = 512 from 30.7 to
  91 tok/s against one position at a time ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-25.md#ds41batch-pp)). Two batches stay in
  flight, layer first, so the card routes and attends one while the CPU runs the other's host experts: 1.20×
  at P = 4096, the host's wait for routes falling from 17.8 to 0.8 ms per layer-batch
  ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-26.md#prefillgroup-ab)). A batched prompt leaves the same state as one step per
  token, bit for bit.
- **Qwen prompts as tensor-core matrix products.** The Qwens' prompt weights and routed experts run as grouped
  int8 tensor-core GEMMs: P = 4096 went from 357 tok/s (eight positions a pass) to 2,615
  ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-25.md#qwen3prefill-ab)). A prefill-only flash attention kernel, where a block owns 64
  query rows and loads each 64-key tile once, then took it 2.12× further
  ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-25.md#q3pflash-ab)), and ubatches of 4096 tokens instead of 512 another 1.35×
  ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-25.md#q3ubatch-ab)).
- **Disk rows read ahead of the step (V4.1).** V4.1's engram table (195 GiB) stays on NVMe, and every token
  reads 48 scattered rows. In a bench of that path, read on demand from a cold cache they took 4.69 ms a token;
  with the next token's read-ahead issued early, 0.31 ms ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-22.md#p-the-engram-path-costs-a-third-of-a-millisecond-and-it-is-syscalls)).
- **CPU experts at memory speed.** AVX2 kernels take the dot products on the quantized weights as stored, and
  a step's host experts go out as 82 grouped dispatches on a pinned worker pool instead of 720 per-matrix
  ones. In a bench of that leg on an earlier V4.1 file: 31.1 ms against 35.5 ms for the same bytes, at 130 GB/s of the machine's 148
  ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-23.md#v41-host-leg)). The kernels are not faster than ik_llama.cpp's on one core;
  the leg is bound by memory.
- **One CUDA graph a decode step, the CPU experts inside it.** Where experts run on the host, the card waits on
  a counter in pinned memory that the CPU workers write, not on a host synchronization. Not isolated by a
  measurement on today's models.
- **Speculative decoding** with greedy output unchanged: DSpark for V4.1 (the draft on a second card), the MTP
  head for Qwen3.8 and GLM-5.3.
- **Loads stream through a pinned ring** with one sync: a warm V4.1 load went from 36.3–36.6 s to 15.9–16.6 s
  ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-30.md#v41-load-upload)).

## How it is verified

- **Against ik_llama.cpp.** Intermediate tensors are dumped from its CPU backend and compared per layer.
  Integer outputs (router ids, engram rows, indexer top-k) match exactly outside tie bands; float outputs stay
  inside bands derived from both engines' rounding. On the public V4.1 file: PPL 2.2401 against ik's 2.2378,
  KLD 0.00987, same top token 97.46 %.
- **Against Cloudflare's own Python, for Clef.** The request encoder's token ids equal the release's; the
  head's logits stay within 2e-5 of an f64 copy of the release's head; the backbone's hidden states gate
  against llama.cpp mainline's final-norm rows. End to end, the top option equals the BF16 release's on 29 to
  31 of 31 questions for every bartowski file from `Q8_0` down to `Q3_K_S` — every miss a question the release
  itself answers within 0.02 of a tie ([table](tools/ref/clef/agreement.md)).
- **Bit gates.** Graph replay equals eager execution; the skewed pass equals two single steps; a rollback and
  a V4.1 batched prompt leave the same state as one step per token.
- **Each gate is shown to fail** on its defect before the fix lands. Bands are not relaxed to pass.

## Hardware

**Required of every host**: an NVIDIA GPU of compute capability 8.6 or newer and its driver; a CPU with AVX2
and FMA (the prebuilt archive targets x86-64-v3); glibc 2.34+; Linux, or Windows through WSL2. NVMe is
recommended — a load streams the file to the card, and V4.1 reads its engram table from disk as it runs.

**What each model needs:**

| Model | GPU | Host RAM | Disk (the file) |
|---|---|---|---|
| Clef-Flash | 12 GB covers every quantization (the files run 4.26–9.55 GB) | any | 4.3–9.6 GB, plus the head (243 MB, fetched) |
| Qwen3.6-35B `Q4_K_M` | 24 GB whole, or 12–16 GB with `--place a` | 32 GB whole; about 10 GB free beside a small card | 21.2 GB |
| Qwen3-30B `Q4_K_M` | 24 GB whole, or 12–16 GB with `--place a` | 32 GB whole; about 8 GB free beside a small card | 18.6 GB |
| Qwen3.8-Flash-Next | 24 GB recommended (the expert share sizes to the card's free bytes) | 256 GB | 111.3 GB, plus the 2.8 GB MTP draft |
| GLM-5.3-Flash | 24 GB recommended | 256 GB | 199.7 GB |
| DeepSeek-V4.1-Flash | 24 GB recommended | 256 GB | 347.3 GB, plus 155.7 GB for the optional r8 sidecar |

**Recommended** (the configuration every number in this README ran on): a 32-core AVX2 CPU with 8 DDR4
channels, 256 GB of RAM, an RTX A6000 48 GB, and an RTX 3090 24 GB beside it for `--place bp`. Cards are found
by device, not by name; a list word such as `0+1` names cards by CUDA index (`CUDA_DEVICE_ORDER=PCI_BUS_ID`
matches `nvidia-smi`'s numbering).

Only the A6000, the 3090, and an RTX 3060 12 GB under WSL2 (running Clef-Flash, its download and its cache)
have been run; other cards and drivers are untested — please open an issue with `--version` and the error
text. Costs per part: [`docs/HARDWARE.md`](docs/HARDWARE.md).

## Limits

- **sm_86+** GPUs; the prebuilt archive carries sm_86 PTX.
- One model a server; one expert tier card at most (`--place bp`).
- Concurrent streams run as one pass on a whole-card Qwen3-30B, on V4.1, GLM-5.3 and Qwen3.8; a placed Qwen3-30B,
  Qwen3.6 and a sampled request on a drafting load step in turn. A new request's prompt runs whole while the other
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
- A pinned nightly with a pinned cuda-oxide revision from our fork ([`docs/BUILD.md`](docs/BUILD.md)).

In progress: serving on N cards for every model; Clef from the IQ and Q2_K files; DeepSeek-V4-Flash-0731.

## Build

See [`docs/BUILD.md`](docs/BUILD.md): the toolchain and one command block per model (download, build,
generate, serve, and `--place gate` on a single RTX 3090). In short: Linux x86-64, CUDA 13.3, LLVM 21 and
`cargo-oxide` from our cuda-oxide fork, then `cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates
--features deepseek41 --release --bin generate_ds41`.

## Upstream

- ik_llama.cpp: DeepSeek-V4.1 support ([#2455](https://github.com/ikawrakow/ik_llama.cpp/pull/2455)),
  `GGML_CUDA_NO_PINNED_WEIGHTS` ([#2444](https://github.com/ikawrakow/ik_llama.cpp/pull/2444)), V4.1 DSpark
  drafts ([#2522](https://github.com/ikawrakow/ik_llama.cpp/pull/2522),
  [#2546](https://github.com/ikawrakow/ik_llama.cpp/pull/2546)), and eleven more fixes, merged;
  [#2507](https://github.com/ikawrakow/ik_llama.cpp/pull/2507) and
  [#2562](https://github.com/ikawrakow/ik_llama.cpp/pull/2562) open.
- cuda-oxide: [#1314](https://github.com/NVIDIA/cuda-rust/pull/1314) and
  [#1321](https://github.com/NVIDIA/cuda-rust/pull/1321) merged;
  [#1329](https://github.com/NVIDIA/cuda-rust/pull/1329) and
  [#1346](https://github.com/NVIDIA/cuda-rust/pull/1346) open. Unfiled toolchain issues:
  [`docs/upstream/nvlabs-ledger.md`](docs/upstream/nvlabs-ledger.md).
- llama.cpp [#29008](https://github.com/ggml-org/llama.cpp/pull/29008) merged; mistral.rs
  [#2430](https://github.com/EricLBuehler/mistral.rs/pull/2430) and cutile-rs
  [#309](https://github.com/NVlabs/cutile-rs/pull/309) open.

## More

- How bloomery uses cuda-oxide: [`docs/cuda-oxide.md`](docs/cuda-oxide.md). Every other page, reference and
  working records: [`docs/README.md`](docs/README.md).
- Measurements and command lines: [rig-log](https://github.com/midagedev/rig-log) (Korean).
- Recorded sessions, not benchmark rows. bloomery 0.2.1: [V4.1 answering a code review on one
  A6000](https://tape.midagedev.com/r/un7mimv23csn2pyhfk3d), [GLM-5.3 on two cards
  (`--place bp`)](https://tape.midagedev.com/r/wcp952uyjebbpui7sgch), [Qwen3.8 on one
  A6000](https://tape.midagedev.com/r/9d6bv7ssftr4e9cu8wdi), [Qwen3-30B serving four streams on one
  3090](https://tape.midagedev.com/r/uxkad26d6jjr26nixrkh). Earlier: [V4.1 on two cards answering a coding
  review](https://tape.midagedev.com/r/6w4t9r5nqwtt5c9sagn3); [GLM-5.3's server at its
  defaults](https://tape.midagedev.com/r/6kf3sxuqpza6m7k7iwi6) against [llama.cpp's GLM pull request on the
  same card](https://tape.midagedev.com/r/6nn6grc6hztpssp88hz5); [Clef-Flash answering seven Korean SystemOne
  requests](https://tape.midagedev.com/r/zd3asiqegffcmvky9hti).
- Working contract: [`AGENTS.md`](AGENTS.md), [`CONTRIBUTING.md`](CONTRIBUTING.md).

## References and credits

[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and ggml are the numerical reference: the kernels
were written by reading ggml's k-quant code and ik_llama.cpp's `iqk_mul_mat`, and accuracy is defined against
their output. [exllamav3](https://github.com/turboderp-org/exllamav3) and
[mistral.rs](https://github.com/EricLBuehler/mistral.rs) are design references, cited where used. All four are
MIT. The GPU kernels compile with [cuda-oxide](https://github.com/NVIDIA/cuda-rust) (Apache-2.0).

AI assistants helped write the code and the documentation. Every number here was measured by the runners in
`tools/ref/`.

## License

MIT, see [`LICENSE`](LICENSE). `crates/oxide-ice-unroll` carries NVIDIA's Apache-2.0 headers. Third-party
notices are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
