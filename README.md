<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img src="docs/assets/logo.svg" width="128" alt="bloomery">
  </picture>
</p>

# bloomery

An LLM inference engine in Rust, from the HTTP server down to the CUDA kernels — and a **drop-in
[llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server)** for it.

## llama-server compatibility

Your OpenAI client works unchanged. bloomery speaks llama-server's HTTP API and reads the same files:

- **The API**: `/v1/chat/completions`, `/completion`, streaming, tool calls, `reasoning_content`,
  `cache_prompt`, `/tokenize`, `/detokenize`, `/slots`, `/metrics`, `/v1/models`, `/props`. OpenAI SDKs, curl,
  and llama-server tutorials apply as they are.
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
tar -xzf bloomery-0.2.0-linux-x86_64-cuda-sm86.tar.gz && cd bloomery-0.2.0-linux-x86_64-cuda-sm86
bin/bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M --port 8080
```

[bloomery 0.2.0](https://github.com/midagedev/bloomery/releases/tag/v0.2.0) · [all releases](https://github.com/midagedev/bloomery/releases)

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

**Concurrent requests.** Every seat takes `--parallel N`. On a whole-card qwen3moe load (Qwen3-30B and its
kin) the N slots are resident sequences inside the one model: each running request decodes one token a round, so
the streams flow together, and each answers its solo run's tokens exactly. The context is split as llama-server
splits it with `-np N` and no `-kvu`: `--ctx-size` (or the automatic choice) is the total, each slot holds
`total / N` rows; unset, `--parallel` is 2, and `--parallel 1` keeps one sequence with the whole context. The
ds41, glm and qwen38 seats, a qwen35moe file and a placed load still take turns of 64 tokens on one sequence: a
later arrival preempts at the next step, and both answer their solo runs' tokens exactly — V4.1, GLM and
Qwen3.8 by snapshot/resume (their MTP drafts rejoining), the qwen35moe file and a placed load by re-prefill.
Unset, the ds41, glm and qwen38 seats size the slots to what the park budget holds of one slot's whole-context
state (a `parallel` line on stderr names the rule, the slots and each term; never below one, and `--parallel 1`
keeps the plain engine). Their N slots share one context-sized cache in turns.

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

Single-stream tok/s on the development machine (an RTX A6000 48 GB and a 32-core AVX2 CPU; decode `n = 96`).
Every number comes from the runners in `tools/ref/` under the quiet-machine protocol, and every measured row,
the other engines, and progress over time live on [rig-log's bench page](https://github.com/midagedev/rig-log/blob/main/docs/bloomery-bench.md).

Against llama.cpp, DeepSeek-V4.1-Flash `Q3_K_M` ([pull #28696](https://github.com/ggml-org/llama.cpp/pull/28696) at its fastest
flags, 2026-09-28, [conditions and caveats](https://github.com/midagedev/rig-log/blob/main/log/2026-09-28.md#v41-release) — a warm
re-measure replaces these rows):

| A6000, tok/s | bloomery | llama.cpp | ratio |
|---|---:|---:|---:|
| decode, depth 6 | **29.66** | 22.75 | 1.304 |
| decode, depth 4096 | **30.04** | 21.78 | 1.379 |
| prompt, P = 512 | **190.6** | 104.9 | 1.817 |
| prompt, P = 4096 | **358.4** | 76.5 | 4.68 |

On this machine, today's defaults ([per-row conditions in rig-log](https://github.com/midagedev/rig-log/blob/main/docs/bloomery-bench.md)):

| Model | Decode | Prompt |
|---|---|---|
| V4.1-Flash `Q3_K_M`, A6000 + CPU | 43.30 after P = 512, 36.77 after 4096 | 191.97 (P = 512), 376.67 (P = 4096) |
| V4.1-Flash, A6000 + 3090 (`--place bp`) | 44.28 / 37.45 | 205.48 / 388.93 |
| GLM-5.3-Flash `UD-Q4_K_XL`, A6000 + CPU | 20.95 (depth 512) | — (the batched prompt and the MTP draft landed after this window) |
| Qwen3.8-Flash-Next `UD-Q4_K_XL`, A6000 + CPU | 91.24 after P = 512, 79.33 after 4096 | 861.7 (P = 512), 1,024.3 (P = 4096) |
| Qwen3.6-35B `Q4_K_M`, A6000 whole | 204.4 (depth 6), 195.8 (depth 4096) | 6,464 (P = 512), 8,311 (P = 4096) |
| Qwen3-30B `Q4_K_M`, A6000 whole | 209.2 (depth 6), 175.2 (depth 4096) | 7,827 (P = 512), 9,276 (P = 4096) |

A first start compiles the GPU code for the card (tens of seconds); later starts take seconds. A warm V4.1
load takes about 16 s.

## How it works

- More than 200 CUDA kernels, all written in Rust with [cuda-oxide](https://github.com/NVIDIA/cuda-rust)
  ([how](docs/cuda-oxide.md)).
- Experts on the GPU and the AVX2 CPU in one CUDA graph step; V4.1's batched prompts leave the state of one
  step per token, bit for bit.
- Adaptive residency: routed experts move between the card and the host by the engine's own routing as it
  runs; V4.1's engram table is read from NVMe, 48 rows a token, prefetched by a helper thread.
- Speculative decoding with greedy output unchanged: DSpark for V4.1 (the draft on a second card), the MTP
  head for Qwen3.8 and GLM-5.3.
- Prompts run batched — V4.1 in batches of up to 512 positions with two in flight so the card and the CPU
  overlap, the Qwens through an int8 tensor-core GEMM in ubatches of up to 4096 tokens.
- A load streams the file's bytes to the card through a pinned ring with one sync.

<p align="center">
  <a href="https://github.com/midagedev/rig-log/blob/main/assets/residency-explainer-v5.mp4">
    <img src="https://raw.githubusercontent.com/midagedev/rig-log/main/assets/residency-explainer-v5-poster.png" width="420" alt="Adaptive expert residency explainer (42 s video)">
  </a>
</p>

Adaptive residency on one A6000, V4.1 `Q3_K_M`, 96 decode steps after a 512-token prose prompt: 29.48 tok/s
with it off, 35.70 with the swap rule, 43.86 with the prompt call's streaming as well (the default). The card
served 18 % of the routed calls with it off; with streaming it served 57 % over the first 16 steps and 71 %
over the last 16 ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-30.md#residency-clip)).

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
- Streams flow together only on a whole-card qwen3moe load so far; the other seats' `--parallel` slots take
  turns of 64 tokens. The decide seat serves one request at a time; DSpark drafts never rejoin —
  `--parallel > 1` with `BLOOMERY_DRAFT=dspark` is refused by name.
- Clef takes text states only, and reads backbone weights of Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 and F32 (not the IQ
  types, Q2_K, Q4_0 or Q4_1).
- V4.1 prompts are bound by the CPU expert tier; V4.1 decode is bound by host memory bandwidth.
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

- How bloomery uses cuda-oxide: [`docs/cuda-oxide.md`](docs/cuda-oxide.md). Plan and cost models:
  [`docs/plan.md`](docs/plan.md); GPU design: [`docs/gpu-design.md`](docs/gpu-design.md).
- Measurements and command lines: [rig-log](https://github.com/midagedev/rig-log) (Korean).
- Recorded sessions, not benchmark rows: [V4.1 on two cards answering a coding
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
