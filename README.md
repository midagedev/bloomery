<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
    <img src="docs/assets/banner.svg" width="720" alt="bloomery">
  </picture>
</p>

<p align="center">
  <b>A fast inference engine for mixture-of-experts models that are larger than the GPU.</b><br>
  Rust from the HTTP server down to the CUDA kernels. NVIDIA cards, and Apple Silicon through Metal.<br>
  A drop-in <a href="https://github.com/ggml-org/llama.cpp/blob/master/tools/server">llama-server</a>.
</p>

<p align="center">
  <a href="https://github.com/midagedev/bloomery/releases/tag/v0.3.10">0.3.10 release notes</a> ·
  <a href="docs/models.md">Models</a> ·
  <a href="docs/serving.md">Serving</a> ·
  <a href="docs/evaluating.md">Evaluating</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

```sh
brew install midagedev/tap/bloomery        # Linux x86-64 with NVIDIA sm_86+, or an Apple Silicon Mac
bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M --port 8080
```

## What it does

- **Models larger than the cards, and larger than the host's RAM.** The routed experts live on the cards and in host
  RAM, and the engine moves the experts it calls most onto the card while it runs. bloomery serves Muse Spark Open
  816B-A42B (225.6 GB of weights) and MiniMax-M3.1-Flash (250.1 GB, or 143.9 GB as Unsloth's 2-bit GGUF) on two cards
  with 72 GB between them. When the RAM
  cannot hold the host's share either, Qwen3.8, GLM-5.3-Flash and DeepSeek-V4.1-Flash page the rest from NVMe.
- **Checkpoint folders as downloaded.** MiniMax-M3.1-Flash loads from its vendor's NVFP4 safetensors folder, and
  Qwen3.8 from its NVFP4 or FP4-Local folder, with no conversion and no requant. Everything else loads from GGUF.
- **Apple Silicon.** A Metal backend serves Qwen3.6-35B-A3B and Qwen3-30B-A3B on a Mac GPU, from mlx-community's 4-bit
  folders as downloaded or from Q4_K_M GGUF files (Qwen3.6 also from `UD-Q4_K_XL`), and runs Ornith-1.5-35B and MiniCPM-V 4.7 35B-A3B (Q4_K_M, text only); prompts run
  in batched passes, and a follow-up turn reads only its new messages ([below](#apple-silicon)).
- **llama-server compatible.** The same HTTP API (OpenAI and Anthropic), GGUF files and flag spellings, including the
  placement flags (`-dev`, `-ts`, `-ngl`, `--cpu-moe`, `-ot`, `--fit`) with llama's meaning. Your client works
  unchanged; Claude Code runs with `ANTHROPIC_BASE_URL=http://localhost:8080`.
- **A prompt cache that survives other requests.** Saved prompt states stay in host RAM: a request that shares a prefix
  with an earlier one restores it, even after another request used the slot
  ([recorded on tape](https://tape.midagedev.com/r/u8fnay7pn4djqmy6vvaq)).
- **Several requests in one pass**, speculative decoding (MTP, DSpark) on by default where a model has a draft, and
  one CUDA graph per decode step with the CPU experts inside it.
- **Rust all the way down.** Every CUDA kernel is written in Rust and compiled with
  [cuda-oxide](https://github.com/NVIDIA/cuda-rust); the Metal backend's host side is Rust, its kernels Metal
  Shading Language.

## Status (0.3.10)

Large mixture-of-experts models on one workstation: an RTX A6000 48 GB and an RTX 3090 24 GB (at its 250 W cap),
32-core AVX2 host, 256 GB of RAM. Serving defaults: `--place` unset (it picks `bp`: the A6000 holds the model's stage,
the 3090 holds experts, the host runs the rest), every lever unset, no `--parallel`, `--ctx-size 4300`. Means of two
rounds, tok/s.

| Model | Prompt, 512 tokens | Prompt, 4,096 tokens | Decode at depth 512 | Decode at depth 4,096 |
|---|---|---|---|---|
| MiniMax-M3.1-Flash `UD-Q2_K_XL` (GGUF) | 160.16 | 165.92 | 28.63 | 29.41 |
| MiniMax-M3.1-Flash, NVFP4 folder | 129.08 | 131.56 | 21.12 | 19.53 |

On the RTX 3090 alone, the same GGUF runs at 21.36 tok/s decode at depth 512, and Nanbeige4.3-33B `Q4_K_M` loads whole
on the card (a 512-token prompt at 745.46 tok/s, decode 24.22). Flint 40B-A2.5B (`UD-Q4_K_XL`) loads whole on the
RTX A6000. The release notes hold the commits, the release check on the tagged binary, every off cell and the
contributors' work.

## Install

### Linux and WSL2 (NVIDIA)

One binary, no CUDA toolkit and no Rust toolchain: Linux x86-64 with AVX2 and glibc 2.34+, an NVIDIA GPU of compute
capability 8.6 or newer, driver R580 or newer. On Windows, run it inside WSL2 (`wsl --install`, reboot, then the same
commands). On a DGX Spark or another GB10 machine (Linux aarch64), the installer takes the GB10 archive
(`linux-aarch64-cuda-sm121a`, the GB10's FP4 tensor cores) where the release has one; then
`bloomery-serve -m <folder>` serves NVIDIA's FP4-Local and NVFP4 checkpoints as downloaded.

```sh
# Homebrew (Linux x86-64)
brew install midagedev/tap/bloomery

# or the installer: latest release, sha256-checked, symlink into ~/.local/bin
curl -fsSL https://raw.githubusercontent.com/midagedev/bloomery/main/tools/release/install.sh | sh

# or Docker (needs the host's NVIDIA Container Toolkit; model downloads live in the bloomery-cache volume)
docker run --gpus all -p 8080:8080 -v bloomery-cache:/root/.cache/bloomery \
  ghcr.io/midagedev/bloomery --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M

# or the plain tarball
tar -xzf bloomery-0.3.10-linux-x86_64-cuda-sm86.tar.gz && cd bloomery-0.3.10-linux-x86_64-cuda-sm86
bin/bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M --port 8080
```

The tarballs are on the [releases page](https://github.com/midagedev/bloomery/releases)
([0.3.10](https://github.com/midagedev/bloomery/releases/tag/v0.3.10)). From source: [`docs/BUILD.md`](docs/BUILD.md).

### Apple Silicon

macOS 26.2 or newer; the binaries are signed and notarized. Two architectures run: `qwen35moe` and `qwen3moe`. On a
build after 0.3.10, an mlx-community 4-bit folder opens as downloaded (`--hf mlx-community/<repo>` or `-m <folder>`):
Qwen3.6-35B-A3B and Qwen3-30B-A3B-Instruct-2507. 0.3.10 converts the Qwen3.6 folder once with `bloomery-prepare`, which
still writes one GGUF split for those who want one file. Both models also run from their Q4_K_M GGUF, and Qwen3.6 from
`UD-Q4_K_XL`. Two more Q4_K_M files with Qwen3.6's shapes run, text only: Ornith-1.5-35B, and MiniCPM-V 4.7 35B-A3B (its
`mmproj` is not read). Sage-A1.5 has the same shapes and has not been run on a Mac yet. A file the Metal backend cannot
run is refused by name when it opens. A file whose shapes the kernels take but no build has registered compiles its
kernels when it opens, and the server says that no gate checked that shape.

```sh
# Homebrew, or the same installer as on Linux (bloomery-0.3.10-macos-arm64-metal.tar.gz on the releases page)
brew install midagedev/tap/bloomery
curl -fsSL https://raw.githubusercontent.com/midagedev/bloomery/main/tools/release/install.sh | sh

bloomery-serve-metal --hf mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --port 8080   # after 0.3.10: fetched once
bloomery-serve-metal -m PATH/Qwen3.6-35B-A3B-4bit --port 8080                         # after 0.3.10: a folder on disk
bloomery-serve-metal -m PATH/Qwen3.6-35B-A3B-Q4_K_M.gguf --port 8080
```

On an M4 Pro 48 GB (development loop, three rounds, tok/s): decode 101.2 and a 4,096-id prompt at 1,076.5 on the MLX
4-bit weights; 85.6 and 1,044.5 on the Q4_K_M file. mlx-lm and llama.cpp ran on the same files in the same sitting:
[their rows](https://github.com/midagedev/rig-log/blob/main/log/2026-10-25.md#mac-m4pro-xeng). On
[@avlp12](https://github.com/avlp12)'s machines at 0.3.9's Metal code, a 4,096-id prompt runs at 3,300 tok/s on an M3
Ultra and 5,972 on an M5 Ultra, whose neural accelerators run the prompt's GEMMs, and decode after 512 ids at 141–142
and 211–212 tok/s ([#593](https://github.com/midagedev/bloomery/issues/593)). One slot, text only;
the flags, the limits and every step: [`docs/mac.md`](docs/mac.md).

## Use

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}'
```

The file's architecture picks the seat (the model family's engine), and `--place` can stay unset on every model.

| Seat | Serves | One command |
|---|---|---|
| `ds41` | [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | `bloomery-serve --hf vcruz305/DeepSeek-V4.1-Flash-GGUF:Q3_K_M` |
| `glm` | [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | `bloomery-serve --hf unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL` |
| `qwen38` | [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | `bloomery-serve --hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL` |
| `qwen3` | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507), [MiniCPM-V 4.7 35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf), [Sage-A1.5](https://huggingface.co/InternScience/Sage-A1.5) (images with `--mmproj`) | `bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M` |
| `mimo2` | [MiMo-V2.6-Flash](https://huggingface.co/ggml-org/MiMo-V2.6-Flash-MOPD-GGUF) | `bloomery-serve --hf ggml-org/MiMo-V2.6-Flash-MOPD-GGUF:MXFP4` |
| `decide` | [Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) and [lev](https://huggingface.co/ggml-org/lev-GGUF) (decision models) | `bloomery-serve --hf bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M` |

These seats serve a file or a checkpoint folder on disk (`-m PATH`):

| Seat | Serves | The file |
|---|---|---|
| `minimaxm3` | MiniMax-M3.1-Flash | the vendor's NVFP4 safetensors folder as downloaded (MiniMax-M3: unverified) |
| `musespark` | Muse Spark Open 816B-A42B | `Muse-Spark-Open-816B-A42B-UD-IQ2_XXS-00001-of-00006.gguf` |
| `qwen38` | Qwen3.8-Flash-Next, checkpoint folders | NVIDIA's NVFP4 folder or the FP4-Local folder, no conversion, on a card that holds every routed expert (a DGX Spark's GB10); also `--hf nvidia/Qwen3.8-Flash-Next-NVFP4` |
| `nemotron` | Nemotron 3 Nano 4B, Nemotron 3.5 Bolt 74B-A9B, Nemotron-3.5-Lightning 30B-A3B | the GGUF |
| `solar2` | Solar-Open2-250B, two cards | the GGUF, with `--place bp` |
| `raon1` | Raon-1-mini-preview | `raon-1-mini-preview-Q4_K_M.gguf`, its DFlash2 draft file beside it |

What each model needs in GPU memory, RAM and disk is in [`docs/models.md`](docs/models.md). In short: Qwen3-30B and
Qwen3.6 run on a 24 GB card (or 12–16 GB, split onto the CPU by itself); Qwen3.8 wants 128 GB of RAM, GLM-5.3 and
V4.1 256 GB, and less RAM runs with the overflow paged from NVMe. The API, every flag, concurrent requests, drafts and
adaptive residency: [`docs/serving.md`](docs/serving.md).

## Hardware

| What runs | State |
|---|---|
| NVIDIA cards | Compute capability 8.6 or newer, driver R580 or newer (CUDA 13); an older driver is refused at the start. Run here: RTX 3090, RTX A6000, RTX 3060 (WSL2). Run by users: RTX 5090 ([#2](https://github.com/midagedev/bloomery/issues/2)), DGX Spark (GB10: the release's GB10 archive, or from source: [`docs/contrib/dgx-spark.md`](docs/contrib/dgx-spark.md)), one model across several Sparks. Ada (sm_89): no report yet |
| Apple Silicon | macOS 26.2 or newer, the release's signed binaries (Homebrew, the installer or the tarball); Qwen3.6-35B-A3B, Qwen3-30B-A3B, Ornith-1.5-35B and MiniCPM-V 4.7, one slot, text only ([`docs/mac.md`](docs/mac.md)). Run here: M1 Pro, M4 Pro 48 GB. Run by users: M3 Ultra, M5 Ultra ([#593](https://github.com/midagedev/bloomery/issues/593)). On an M5 the prompt's GEMMs run on the neural accelerators, picked by the device |
| Host CPU | x86-64 with AVX2 (x86-64-v3), AMD or Intel, for the CUDA build, or a GB10's Arm cores; the Metal build runs on Apple Silicon |
| OS | Linux x86-64 with glibc 2.34 or newer; Linux aarch64 on a GB10 (DGX OS 7, glibc 2.39); Windows through WSL2; macOS 26.2 or newer on Apple Silicon. AMD GPUs are not supported yet |

## Numbers

Every number below is a release check or a sitting of the runners in `tools/ref/`, with its machine and conditions.
The reference engines' rows are on
[rig-log's bench page](https://github.com/midagedev/rig-log/blob/main/docs/bloomery-bench.md) with their flags,
measured in the same sitting as ours (the Mac's in [its own log](https://github.com/midagedev/rig-log/blob/main/log/2026-10-25.md#mac-m4pro-xeng)). Conditions: [`docs/performance.md`](docs/performance.md#conditions-of-the-numbers).

**One request at the serving defaults** (every lever unset, no `--parallel`, `--ctx-size 16384`), greedy, thinking
off, RTX A6000 (+ RTX 3090 for `bp`), one load per row, the release candidate's `bloomery-serve`; the release each row
was checked at is in its first column. Prompt sizes are the server's prompt tokens.

| Model, place (release) | Long prompt: tokens, prefill tok/s, first token | Decode after it | Short prompt decode |
|---|---|---|---|
| Qwen3.8 `UD-Q4_K_XL`, `a` (0.3.0) | 5,673, 1,693, 3.38 s | 75.1 tok/s | 86.3 tok/s |
| Qwen3.8 `UD-Q4_K_XL`, `bp` (0.3.1) | 5,673, 1,707, 3.35 s | 84.3 tok/s | 94.0 tok/s |
| Qwen3.6-35B-A3B `UD-Q4_K_XL`, unset → `a` (0.3.1) | 5,673, 7,083, 0.81 s | 115.0 tok/s | 117.5 tok/s |
| GLM-5.3-Flash `UD-IQ4_XS`, unset → `bp` (0.3.1) | 5,548, 172, 32.3 s | 35.1 tok/s | 35.0 tok/s |
| DeepSeek-V4.1-Flash `Q3_K_M`, unset → `a` (0.3.1) | 5,642, 403, 14.0 s | 41.9 tok/s | 42.7 tok/s |
| MiMo-V2.6-Flash `MXFP4`, unset → `a` (0.3.1) | 5,620, 104, 53.9 s | 34.8 tok/s | 36.1 tok/s |
| Nemotron 3.5 Bolt 74B-A9B `24GB`, unset → `a` (0.3.4) | 5,680, 2,716, 2.10 s | 87.5 tok/s | 88.7 tok/s |

<details>
<summary><b>One RTX 3090, two cards, and two requests at once</b> (0.3.0)</summary>

Two requests at once (`--parallel 2`, both busy) after a 512-token prompt, decode tok/s of both together, and one
request's decode at depth 4096. Measured on 0.3.0 (2026-10-12), on a 32-core AVX2 host with 8 DDR4 channels and 256 GB
of RAM; the 3090 at a 250 W cap; the MTP draft off.

**One RTX 3090 24 GB**

| Model | Where it runs | Two requests, tok/s in total | Decode at depth 4096 | Prompt tok/s (P = 512 / 4096) |
|---|---|---|---|---|
| Qwen3-30B-A3B `Q4_K_M` | the whole model on the card | 252.5 | 164.1 | 6,496 / 6,801 |
| Qwen3.6-35B-A3B `Q4_K_M` | the whole model on the card | 257.3 | 181.8 | 5,766 / 6,685 |
| Qwen3.8-Flash-Next `UD-Q4_K_XL` | card + CPU | 64.3 ¹ | 51.5 ¹ | 502 / 664 ¹ |
| GLM-5.3-Flash `UD-Q4_K_XL` | card + CPU | 25.3 ² | 19.5 | 176 / 205 |
| DeepSeek-V4.1-Flash `Q3_K_M` | card + CPU | 33.5 | 26.9 | 195 / 376 |
| Nemotron 3.5 Bolt 74B-A9B `24GB` (0.3.4) | the whole model on the card | — ⁴ | 86.9 | 1,773 / 2,047 |

**RTX A6000 48 GB + RTX 3090 24 GB (`--place bp`)**

| Model | Two requests, tok/s in total | Decode at depth 4096 | Prompt tok/s (P = 512 / 4096) |
|---|---|---|---|
| DeepSeek-V4.1-Flash `Q3_K_M` | 56.1 | 39.0 | 233 / 398 |
| GLM-5.3-Flash `UD-Q4_K_XL` | 34.0 | 22.7 | 218 / 262 |
| Qwen3.8-Flash-Next `UD-Q4_K_XL` | — ³ | 90.5 | — / 1,777 |

¹ with `BLOOMERY_RESIDENCY=mid-p32-s1` · ² 29.1 with `BLOOMERY_RESIDENCY=mid-p0-s1` · ³ under `--place bp` the Qwen3.8
seat runs two requests in turn, a slot at a time · ⁴ the nemotron seat serves one slot. Bolt's rows are 0.3.4's
sittings (n=4, prompt at e92fa8bde, decode at 44495f0ba); on one RTX A6000 the same sittings read 87.4 tok/s at depth
4096 and 2,143 / 2,772 prompt tok/s.

</details>

<details>
<summary><b>One RTX 3060 12 GB</b> (0.3.0, WSL2)</summary>

One request at the serving defaults (`--ctx-size 16384`, every lever unset), 0.3.0's release tarball, an i7-12700F
desktop with 32 GB of RAM under WSL2, greedy, thinking off.

| Model | Long prompt: tokens, prefill tok/s, first token | Decode after it | Short prompt decode |
|---|---|---|---|
| Qwen3-30B-A3B `Q4_K_M`, card + CPU | 5,619, 103 ¹, 54.4 s | 38.8 tok/s | 40.4 tok/s |
| Qwen3.8-Flash-Next `UD-Q4_K_XL`, experts paged from NVMe ² | 5,673, 84.9, 66.8 s | 4.5 tok/s | 5.4 tok/s |

¹ 0.3.0's measurement: the prompt ran one row at a time on this load. 0.3.1 runs it in wide units; not re-measured
here. ² 0.3.1's release candidate with no flags at all: the plan chose one slot of 28,881 positions and pages 75 GB of
experts from NVMe through a 19 GB RAM arena; a paged plan runs with the MTP draft off. 0.3.0 refused this load at the
defaults.

</details>

<details>
<summary><b>Contributors' cards</b>: RTX 5090, DGX Spark (0.3.0, by @juliankang4)</summary>

One request at the serving defaults, measured by [@juliankang4](https://github.com/juliankang4) on their own hardware
at 0.3.0, greedy, thinking off, 256 tokens. Each row is their measurement on that card; the conditions, logs and tapes
are in [#46](https://github.com/midagedev/bloomery/issues/46). Their 0.3.9 DGX Spark work (prompts up to about twice
as fast) is in the [release notes](https://github.com/midagedev/bloomery/releases/tag/v0.3.9#from-our-contributors).

| Card | Model | ~4K prompt: prefill tok/s, first token | Decode after it | Short prompt decode |
|---|---|---|---|---|
| RTX 5090 32 GB | Qwen3.6-35B-A3B `Q4_K_M`, whole on the card | 14,937, 276 ms | 345.2 tok/s | 356.1 tok/s |
| RTX 5090 32 GB | Qwen3.8-Flash-Next `UD-Q4_K_XL` ¹ | 447, 9.12 s | 54.5 tok/s | 47.7 tok/s |
| DGX Spark (GB10) | Qwen3.6-35B-A3B `Q4_K_M`, whole | 4,576, 899 ms | 89.8 tok/s | 93.5 tok/s |
| DGX Spark (GB10) | Qwen3.8-Flash-Next `UD-Q4_K_XL`, whole, MTP draft on | 1,038, 4.11 s | 41.6 tok/s | 41.5 tok/s |

¹ with `BLOOMERY_HOST_ROOM=56G` on 64 GB of RAM: 0.3.0 refuses this load at the defaults.

</details>

## Why it is fast

Each point has a same-engine measurement in [`docs/performance.md`](docs/performance.md#why-it-is-fast) or a
release's check:

- **Experts move to where they are used.** The engine counts its own routing and seats the experts decode calls most
  on the card: V4.1 decode 29.48 → 43.86 tok/s, MiniMax-M3.1-Flash 16.52 → 21.97 at depth 512 (0.3.9).
- **The card and the CPU work on a prompt at once** (V4.1): two batches in flight, 1.20× at P = 4096.
- **A prompt's CPU experts move to the card** (Qwen3.8): each layer seats the experts the prompt uses most, and they
  stay for the answer; P = 4096 from 733 to 1,258 tok/s on the A6000.
- **Prompts as tensor-core GEMMs**: Qwen's int8 path took P = 4096 from 357 to 2,615 tok/s; MiniMax-M3.1-Flash's card
  experts run [@juliankang4](https://github.com/juliankang4)'s NVFP4 kernels.
- **CPU experts at memory speed**: AVX2 kernels on the stored quantization (Q4_K switches to AVX-512 VNNI on a CPU that
  has it), grouped dispatch on a pinned pool; the host's NVFP4 products run a column tile, MiniMax-M3.1-Flash's
  512-token prompt 88.35 → 124.19 tok/s (0.3.9).
- **One CUDA graph a decode step**, with the CPU experts inside it, and speculative decoding (MTP, DSpark).

Output is checked layer by layer against ik_llama.cpp (V4.1: PPL 2.2401 against 2.2378, same top token 97.46 %), and
every gate is shown to fail on its defect before the fix lands ([how](docs/performance.md#how-it-is-verified)).

## Known limits (0.3.10)

- DeepSeek V4.1 on two cards (`--place bp`) is not faster than one card (`--place a`). Unset `--place` picks `a`.
- The `nemotron`, `raon1`, `solar2` and `musespark` seats serve one slot; `--parallel` above 1 is refused by name.
  The `nemotron`, `raon1` and `solar2` seats keep no saved prompt states.
- `--place bp` is refused by name on the `nemotron` seat, Qwen3.6 and Qwen3-30B; they run one card plus the host tier.
- Nemotron-3.5-Lightning `UD-Q4_K_XL` is a placed load: its whole-card expert body is IQ2_XXS, and the other expert
  types stay on the host.
- Muse Spark Open always thinks: a turn's first tokens are its reasoning, so a short `max_tokens` can end with empty
  `content`.
- MiniMax-M3 (not M3.1) was not checked for 0.3.10; each family's newest model was.
- Flint's greedy text has no CPU or llama.cpp referee yet; the card path's gates and the serving checks hold it.
- DeepSeek V4.1 runs routed experts on a card only as Q3_K gate·up with a Q4_K down; other expert types (ISTA-DASLab's
  IQ1_M, IQ2_XXS and IQ2_XS) run on the host tier.
- GLM-5.3 holds at most 16,384 positions a slot; a `--ctx-size` whose slot share is larger is refused at the start.

Every limit, by seat: [`docs/serving.md`](docs/serving.md#limits).

## Contributing

Issues and pull requests are welcome, in Korean or English: [`CONTRIBUTING.md`](CONTRIBUTING.md)
([한국어](CONTRIBUTING.ko.md)). Tracks we hope for: a new model architecture
([`docs/contrib/new-model.md`](docs/contrib/new-model.md)), the NVIDIA DGX Spark
([`docs/contrib/dgx-spark.md`](docs/contrib/dgx-spark.md)), Apple Silicon ([`docs/contrib/mac.md`](docs/contrib/mac.md)),
and hardware we do not have. Starter work carries the
[good first issue](https://github.com/midagedev/bloomery/labels/good%20first%20issue) label; questions go to
[Discussions](https://github.com/midagedev/bloomery/discussions). Review looks first at code shape: one owner for
shared logic, no second copy, no silent failure. AI-assisted pull requests are welcome.

Thanks to [@juliankang4](https://github.com/juliankang4) (DGX Spark, NVFP4 and FP4-Local, several requests at once, the
start screen), [@Burntgogi](https://github.com/Burntgogi) (the common owners every model shares, Flint) and
[@avlp12](https://github.com/avlp12) (M3 Ultra and M5 Ultra measurements), and to everyone who filed an issue.

## Docs

| Page | What it covers |
|---|---|
| [`docs/evaluating.md`](docs/evaluating.md) | A checklist to run it and compare it fairly; hardware reports |
| [`docs/serving.md`](docs/serving.md) | llama-server compatibility, flags, seats, concurrent requests, limits |
| [`docs/models.md`](docs/models.md) | Recommended files and what each model needs |
| [`docs/performance.md`](docs/performance.md) | Number conditions, why it is fast, verification, recorded sessions |
| [`docs/mac.md`](docs/mac.md) | Apple Silicon: build, open an MLX folder, serve, limits |
| [`docs/BUILD.md`](docs/BUILD.md), [`docs/cuda-oxide.md`](docs/cuda-oxide.md) | Building from source; the kernels in Rust |
| [`docs/upstream/`](docs/upstream/) | The toolchain-defect ledger; our upstream pull-request drafts are in [rig-log](https://github.com/midagedev/rig-log/tree/main/docs/bloomery/upstream) |
| [`docs/README.md`](docs/README.md) | Every other page and the design records |

Measurements and command lines: [rig-log](https://github.com/midagedev/rig-log) (Korean). Working contract:
[`AGENTS.md`](AGENTS.md).

## References and credits

[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and ggml are the numerical reference: the kernels were written
by reading ggml's k-quant code and ik_llama.cpp's `iqk_mul_mat`, and accuracy is defined against their output.
[exllamav3](https://github.com/turboderp-org/exllamav3) and [mistral.rs](https://github.com/EricLBuehler/mistral.rs)
are design references, cited where used. All four are MIT. The GPU kernels compile with
[cuda-oxide](https://github.com/NVIDIA/cuda-rust) (Apache-2.0).

AI assistants helped write the code and the documentation. Every number here was measured by the runners in
`tools/ref/`, or by a contributor on their own machine with its evidence.

## License

MIT, see [`LICENSE`](LICENSE). `crates/oxide-ice-unroll` carries NVIDIA's Apache-2.0 headers. Third-party notices are
in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
