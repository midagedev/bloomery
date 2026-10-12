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

- **MoE models larger than the card, and larger than the host's RAM.** Routed experts live on the GPU and in host
  RAM; the engine counts its own routing and moves the experts it calls most onto the card while it runs. On one
  RTX 3090 24 GB beside a 256 GB host, DeepSeek-V4.1-Flash `Q3_K_M` (347 GB) serves two requests at 34.4 tok/s in
  total and GLM-5.3-Flash at 26.9. When the RAM cannot hold the host's experts either, Qwen3.8, GLM-5.3-Flash and
  DeepSeek-V4.1-Flash page the rest from the model file on NVMe.
- **A prompt cache that survives other requests.** The Qwen3.8 and GLM-5.3-Flash seats keep saved prompt states in
  host RAM: a request that shares a prefix with an earlier one restores it, even after another request used the
  slot ([recorded on tape](https://tape.midagedev.com/r/u8fnay7pn4djqmy6vvaq)).
- **Several requests, one pass.** With `--parallel 2` two busy streams run through the model together. With the MTP
  draft off, that gave 25–37 % more tokens a second in total than one stream (0.2.1,
  [measured on the A6000](https://github.com/midagedev/rig-log/blob/main/log/2026-10-06.md#rel021-slots)). The draft
  is on by default and already fills the pass's rows: in 0.2.10's release check (A6000), two streams of Qwen3.8
  (`--place a`) and GLM-5.3 `UD-IQ4_XS` gave about the total of one stream.
- **llama-server compatible.** The same HTTP API (OpenAI and Anthropic), GGUF files and flag spellings: your client
  works unchanged.
- **Rust all the way down.** Every CUDA kernel is written in Rust and compiled with
  [cuda-oxide](https://github.com/NVIDIA/cuda-rust).

```sh
brew install midagedev/tap/bloomery        # Linux x86-64, NVIDIA sm_86+
bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M --port 8080
```

## Status (0.3.0)

| Area | State |
|---|---|
| Linux x86-64, NVIDIA sm_86 (RTX 3090, RTX A6000, RTX 3060) | Runs. Every number here comes from these cards |
| NVIDIA driver | R580 or newer (CUDA 13); an older driver is refused at the start |
| Ada and Blackwell (sm_89, sm_120) | Blackwell ran on a user's RTX 5090 under WSL2 ([#2](https://github.com/midagedev/bloomery/issues/2)): the sm_86 PTX compiles for the card at the first start. Ada not yet run. [Reports welcome](docs/evaluating.md#help-wanted-hardware-we-have-not-run) |
| NVIDIA DGX Spark (GB10, aarch64) | Builds and runs from source (a contributor's port, [`docs/contrib/dgx-spark.md`](docs/contrib/dgx-spark.md)): the plan treats the unified memory as one pool, `BLOOMERY_ARCH` sets sm_121, and the quant types without an aarch64 kernel run through a scalar mirror. The release binaries are x86-64 only |
| Host CPU | x86-64 with AVX2 (x86-64-v3), AMD or Intel. Every number here comes from an 8-channel host; a 2-channel desktop host is not measured here and runs the CPU experts slower |
| macOS, Apple silicon · AMD GPUs, Windows without WSL2 | Not supported yet · Not supported |
| OpenAI and Anthropic APIs, streaming, tool calls | Every generative model |
| Several requests at once (`--parallel`, default 2) | One pass on V4.1, GLM-5.3, Qwen3.8, MiMo-V2.6-Flash and a whole-card Qwen3-30B or Qwen3.6; in turn on the rest. When the default's second slot does not fit, MiMo, GLM-5.3 and Qwen3 serve one and say so on stderr; a `--parallel` you set is never lowered. `--ctx-size` alone is one request's context: the server then serves one request at that length, as llama-server does |
| Prompt cache (Qwen3.8, GLM-5.3) | Saved states in host RAM (`--cache-ram`; by default the lesser of 8 GiB and half the RAM the plan leaves; on Qwen3.8, 0 on a plan that pages experts from NVMe). A shared prefix restores the last checkpoint inside it; a saved state keeps checkpoints at 512·2^i positions below its end. On Qwen3.8 the card's checkpoints are 4,096 positions apart, so a shared prefix shorter than that does not restore |
| MTP draft (Qwen3.8, GLM-5.3) | On by default; its width follows measured cost |
| Qwen3.8 prompts | Each layer seats the prompt's most-used CPU experts on the card (`--place a` and `bp`); the extra stream ring runs only under `--place a` |
| Quant types | Card experts in Q4_K–Q6_K, MXFP4, IQ2_XS, IQ3_XXS, IQ4_XS and IQ4_NL; host kernels add Q2_K, IQ2_XS and Q6_K (`UD-Q2_K_XL` files load). Qwen3.8 `UD-Q3_K_XL` runs on the card: faster than `UD-Q4_K_XL` on long prompts and decode, slower at P = 512. IQ3_S runs on the card too, so GLM-5.3 `UD-IQ4_XS` and `UD-Q2_K_XL` keep routed experts on the card. A file's q4_0 and q4_1 tensors open. IQ1_M and BF16 tensors are not loaded yet |
| Cards | One card, or one card plus one expert-tier card (`--place bp`). With `--place` unset and two cards visible, Qwen3.8 and GLM-5.3 add the second card when their plan finds it pays |
| Host RAM under the model | Runs on Qwen3.8, GLM-5.3 and V4.1: the routed experts the RAM cannot hold are paged from the model file on NVMe through a RAM arena. Tokens equal the all-RAM load. A plan keeps 512 MiB under the free RAM it reads, and a paged load shrinks its arena when it finds less RAM than planned. A paged Qwen3.8 load serves one slot by default with the MTP draft off (a set `BLOOMERY_DRAFT=mtp` is refused by name); adaptive residency runs on it. In 0.3.1 a set `--parallel` of 2 or more on a paged Qwen3.8 plan fails at load. On WSL2 with the model on a Windows drive, set `BLOOMERY_NVTIER_READ=buffered` |
| MiMo-V2.6-Flash | Its hot routed experts sit on the card, the rest on the host. After a 5,620-token prompt it decodes at 34.0 tok/s (0.2.10: 19.1), first token at 52.9 s (RTX A6000, `--place` unset, 0.3.0's release check) |
| Image input | V4.1, Qwen3.6, GLM-5.3 and the Clef decide seat, with `--mmproj <file>` (llama-server's flag). Under `--hf` with no `--mmproj`, a seat takes the projector in the model's repo when it costs the text model no card experts or context; otherwise it prints the projector's bytes and serves text (`--no-mmproj` skips it). Images go in as base64 (`data:` URLs), not fetched URLs. A request with an image to a seat started without an encoder is refused by name |

### Known issues and limits (0.3.1)

- **A paged load's default context counts the NVMe arena it gives up.** The context grows while the misses a step
  (the experts it moves to NVMe, plus the arena the per-slot checkpoint reserve takes) stay within 1 % of the base's,
  and the `ctx` line prints `arena=` and `base_arena=`. On a 12 GB card the one-slot default is 12,288 positions
  under that bound [derived on the Flash-Next fixture]; `--ctx-size` sets it by hand.
- **Qwen3.8's MTP draft:** a request that keeps a cached prefix runs with the draft off. The prompt-state save after a
  request adds 100–160 ms to the next request's first token ([#46](https://github.com/midagedev/bloomery/issues/46)).
- **Quant types not loaded yet:** Q2_0, IQ2_S, IQ2_XXS and IQ1_M routed experts
  ([#48](https://github.com/midagedev/bloomery/issues/48)).
- **Not measured here:** Ada cards (sm_89), and a 2-channel desktop host's CPU experts.

To try it and judge it fairly (against llama-server too): [`docs/evaluating.md`](docs/evaluating.md). Full limits:
[`docs/serving.md`](docs/serving.md#limits).

## Install

One binary, no CUDA toolkit and no Rust toolchain: Linux x86-64 with AVX2, glibc 2.34+, an NVIDIA GPU of compute
capability 8.6 or newer and driver R580 or newer. On Windows, run it inside WSL2 (`wsl --install`, reboot, then the same commands).

```sh
# Homebrew (Linux x86-64)
brew install midagedev/tap/bloomery

# or the installer: latest release, sha256-checked, symlink into ~/.local/bin
curl -fsSL https://raw.githubusercontent.com/midagedev/bloomery/main/tools/release/install.sh | sh

# or Docker (needs the host's NVIDIA Container Toolkit; model downloads live in the bloomery-cache volume)
docker run --gpus all -p 8080:8080 -v bloomery-cache:/root/.cache/bloomery \
  ghcr.io/midagedev/bloomery --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M

# or the plain tarball
tar -xzf bloomery-0.3.0-linux-x86_64-cuda-sm86.tar.gz && cd bloomery-0.3.0-linux-x86_64-cuda-sm86
bin/bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M --port 8080
```

The tarballs are on the [releases page](https://github.com/midagedev/bloomery/releases)
([0.3.0](https://github.com/midagedev/bloomery/releases/tag/v0.3.0)). From source: [`docs/BUILD.md`](docs/BUILD.md).

## Use

One binary, six seats; the file's architecture picks the seat, and `--place` can stay unset on every model.

| Seat | Serves | One command |
|---|---|---|
| `ds41` | [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | `bloomery-serve --hf vcruz305/DeepSeek-V4.1-Flash-GGUF:Q3_K_M` |
| `glm` | [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | `bloomery-serve --hf unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL` |
| `qwen38` | [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | `bloomery-serve --hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL` |
| `qwen3` | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (images with `--mmproj`), [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | `bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M` |
| `mimo2` | [MiMo-V2.6-Flash](https://huggingface.co/ggml-org/MiMo-V2.6-Flash-MOPD-GGUF) | `bloomery-serve --hf ggml-org/MiMo-V2.6-Flash-MOPD-GGUF:MXFP4` |
| `decide` | [Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) and [lev](https://huggingface.co/ggml-org/lev-GGUF) (decision models) | `bloomery-serve --hf bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M` |

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}'
```

Claude Code: `ANTHROPIC_BASE_URL=http://localhost:8080`. The API, the flags, concurrent requests, drafts and
adaptive residency: [`docs/serving.md`](docs/serving.md). What each model needs in GPU, RAM and disk:
[`docs/models.md`](docs/models.md). In short, Qwen3-30B and Qwen3.6 run on a 24 GB card (or 12–16 GB, split onto
the CPU by itself); Qwen3.8 wants 128 GB of RAM, GLM-5.3 and V4.1 256 GB, and less RAM runs with the overflow
paged from NVMe.

## Numbers

**0.3.0's release check: one request at the serving defaults** (every lever unset, no `--parallel`,
`--ctx-size 16384`), greedy, thinking off, RTX A6000 (+ RTX 3090 for `bp`), one load per row, the release
candidate's `bloomery-serve`. Prompt sizes are the server's prompt tokens.

| Model, place | Long prompt: tokens, prefill tok/s, first token | Decode after it | Short prompt decode |
|---|---|---|---|
| Qwen3.8 `UD-Q4_K_XL`, `a` | 5,673, 1,693, 3.38 s | 75.1 tok/s | 86.3 tok/s |
| Qwen3.8 `UD-Q4_K_XL`, `bp` | 5,673, 1,700, 3.36 s | 80.5 tok/s | 97.3 tok/s |
| Qwen3.6-35B-A3B `UD-Q4_K_XL`, unset (picks `a`) | 5,673, 7,122, 0.81 s | 115 tok/s | 118 tok/s |
| GLM-5.3-Flash `UD-IQ4_XS`, unset (picks `a`) | 5,548, 108, 51.2 s | 24.4 tok/s | 23.7 tok/s |
| DeepSeek-V4.1-Flash `Q3_K_M`, unset (picks `a`) | 5,644, 407, 13.9 s | 42.4 tok/s | 41.1 tok/s |
| MiMo-V2.6-Flash `MXFP4`, unset (picks `a`) | 5,620, 106, 52.9 s | 34.0 tok/s | 37.7 tok/s |

**Two requests at once** (`--parallel 2`, both busy) after a 512-token prompt, decode tok/s of both together, and
one request's decode at depth 4096. Measured on 0.3.0 (2026-10-12), on a 32-core AVX2 host with 8 DDR4 channels and
256 GB of RAM; the 3090 at a 250 W cap; the MTP draft off. Conditions and the earlier rows:
[`docs/performance.md`](docs/performance.md#conditions-of-the-numbers) and
[rig-log's bench page](https://github.com/midagedev/rig-log/blob/main/docs/bloomery-bench.md).

**One RTX 3090 24 GB**

| Model | Where it runs | Two requests, tok/s in total | Decode at depth 4096 | Prompt tok/s (P = 512 / 4096) |
|---|---|---|---|---|
| Qwen3-30B-A3B `Q4_K_M` | the whole model on the card | 252.5 | 164.1 | 6,496 / 6,801 |
| Qwen3.6-35B-A3B `Q4_K_M` | the whole model on the card | 257.3 | 181.8 | 5,766 / 6,685 |
| Qwen3.8-Flash-Next `UD-Q4_K_XL` | card + CPU | 64.3 ¹ | 51.5 ¹ | 502 / 664 ¹ |
| GLM-5.3-Flash `UD-Q4_K_XL` | card + CPU | 25.3 ² | 19.5 | 176 / 205 |
| DeepSeek-V4.1-Flash `Q3_K_M` | card + CPU | 33.5 | 26.9 | 195 / 376 |

**RTX A6000 48 GB + RTX 3090 24 GB (`--place bp`)**

| Model | Two requests, tok/s in total | Decode at depth 4096 | Prompt tok/s (P = 512 / 4096) |
|---|---|---|---|
| DeepSeek-V4.1-Flash `Q3_K_M` | 56.1 | 39.0 | 233 / 398 |
| GLM-5.3-Flash `UD-Q4_K_XL` | 34.0 | 22.7 | 218 / 262 |
| Qwen3.8-Flash-Next `UD-Q4_K_XL` | — ³ | 90.5 | — / 1,777 |

¹ with `BLOOMERY_RESIDENCY=mid-p32-s1` · ² 29.1 with `BLOOMERY_RESIDENCY=mid-p0-s1` · ³ 0.3.0's measurement: two requests ran in
turn on an expert-tier load. 0.3.1 runs them in one pass; not re-measured here.

**One RTX 3060 12 GB** — one request at the serving defaults (`--ctx-size 16384`, every lever unset), 0.3.0's
release tarball, an i7-12700F desktop with 32 GB of RAM under WSL2, greedy, thinking off.

| Model | Long prompt: tokens, prefill tok/s, first token | Decode after it | Short prompt decode |
|---|---|---|---|
| Qwen3-30B-A3B `Q4_K_M`, card + CPU | 5,619, 103 ⁵, 54.4 s | 38.8 tok/s | 40.4 tok/s |
| Qwen3.8-Flash-Next `UD-Q4_K_XL`, experts paged from NVMe ⁶ | 5,673, 84.9, 66.8 s | 4.5 tok/s | 5.4 tok/s |

⁵ 0.3.0's measurement: the prompt ran one row at a time on this load. 0.3.1 runs it in wide units; not re-measured here.
⁶ 0.3.1's release candidate with no flags at all: the plan chose one slot of 28,881 positions and pages 75 GB of
experts from NVMe through a 19 GB RAM arena; a paged plan runs with the MTP draft off. 0.3.0 refused this load at
the defaults.

**Contributors' cards** — one request at the serving defaults, measured by
[@juliankang4](https://github.com/juliankang4) on their own hardware at 0.3.0, greedy, thinking off, 256 tokens.
Each row is their measurement on that card; the conditions, logs and tapes are in
[#46](https://github.com/midagedev/bloomery/issues/46).

| Card | Model | ~4K prompt: prefill tok/s, first token | Decode after it | Short prompt decode |
|---|---|---|---|---|
| RTX 5090 32 GB | Qwen3.6-35B-A3B `Q4_K_M`, whole on the card | 14,937, 276 ms | 345.2 tok/s | 356.1 tok/s |
| RTX 5090 32 GB | Qwen3.8-Flash-Next `UD-Q4_K_XL` ⁴ | 447, 9.12 s | 54.5 tok/s | 47.7 tok/s |
| DGX Spark (GB10) | Qwen3.6-35B-A3B `Q4_K_M`, whole | 4,576, 899 ms | 89.8 tok/s | 93.5 tok/s |
| DGX Spark (GB10) | Qwen3.8-Flash-Next `UD-Q4_K_XL`, whole, MTP draft on | 1,038, 4.11 s | 41.6 tok/s | 41.5 tok/s |

⁴ with `BLOOMERY_HOST_ROOM=56G` on 64 GB of RAM: 0.3.0 refuses this load at the defaults.

## Why it is fast

Each point has a same-engine measurement in [`docs/performance.md`](docs/performance.md#why-it-is-fast) or a
release's check:

- **Experts move to where they are used**: V4.1 decode 29.48 → 43.86 tok/s with adaptive residency.
- **The card and the CPU work on a prompt at once** (V4.1): two batches in flight, 1.20× at P = 4096.
- **A prompt's CPU experts move to the card** (Qwen3.8): each layer seats the experts the prompt uses most, and
  they stay for the answer; P = 4096 from 733 to 1,258 tok/s on the A6000.
- **Qwen prompts as int8 tensor-core GEMMs**: P = 4096 from 357 to 2,615 tok/s, then a prefill flash kernel.
- **MiMo's hot experts on the card**: decode after a 5,620-token prompt 19.1 → 34.0 tok/s (0.2.10 → 0.3.0 release
  checks, A6000).
- **CPU experts at memory speed**: AVX2 kernels on the stored quantization, grouped dispatch on a pinned pool.
- **One CUDA graph a decode step**, with the CPU experts inside it, and speculative decoding (MTP, DSpark).

Output is checked layer by layer against ik_llama.cpp (V4.1: PPL 2.2401 against 2.2378, same top token 97.46 %),
and every gate is shown to fail on its defect before the fix lands ([how](docs/performance.md#how-it-is-verified)).

## Contributing

Issues and pull requests are welcome, in Korean or English: [`CONTRIBUTING.md`](CONTRIBUTING.md)
([한국어](CONTRIBUTING.ko.md)). Two tracks we hope for are a new model architecture
([`docs/contrib/new-model.md`](docs/contrib/new-model.md)) and the NVIDIA DGX Spark, which runs since 0.3.0 and
next scales across nodes ([`docs/contrib/dgx-spark.md`](docs/contrib/dgx-spark.md)). Starter work carries the
[good first issue](https://github.com/midagedev/bloomery/labels/good%20first%20issue) label; questions go to
[Discussions](https://github.com/midagedev/bloomery/discussions). Review looks first at code shape: one owner for
shared logic, no second copy, no silent failure.

## Docs

| Page | What it covers |
|---|---|
| [`docs/evaluating.md`](docs/evaluating.md) | A checklist to run it and compare it fairly; hardware reports |
| [`docs/serving.md`](docs/serving.md) | llama-server compatibility, flags, seats, concurrent requests, limits |
| [`docs/models.md`](docs/models.md) | Recommended files and what each model needs |
| [`docs/performance.md`](docs/performance.md) | Number conditions, why it is fast, verification, recorded sessions |
| [`docs/BUILD.md`](docs/BUILD.md), [`docs/cuda-oxide.md`](docs/cuda-oxide.md) | Building from source; the kernels in Rust |
| [`docs/upstream/`](docs/upstream/) | The toolchain-defect ledger; our upstream pull-request drafts are in [rig-log](https://github.com/midagedev/rig-log/tree/main/docs/bloomery/upstream) |
| [`docs/README.md`](docs/README.md) | Every other page and the design records |

Measurements and command lines: [rig-log](https://github.com/midagedev/rig-log) (Korean). Working contract:
[`AGENTS.md`](AGENTS.md), [`CONTRIBUTING.md`](CONTRIBUTING.md). AI-assisted pull requests are welcome, and pull
requests for hardware we do not have most of all.

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
