<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
    <img src="docs/assets/banner.svg" width="720" alt="bloomery">
  </picture>
</p>

<p align="center">
  <b>bloomery is a fast LLM inference server for mixture-of-experts models that are larger than the GPU.</b><br>
  It runs on NVIDIA cards, the DGX Spark included, and on Apple Silicon through Metal.<br>
  A drop-in <a href="https://github.com/ggml-org/llama.cpp/blob/master/tools/server">llama-server</a>, in Rust from the HTTP server down to the CUDA kernels.
</p>

<p align="center">
  <a href="https://github.com/midagedev/bloomery/releases/tag/v0.3.11">0.3.11 release notes</a> ·
  <a href="docs/models.md">Models</a> ·
  <a href="docs/serving.md">Serving</a> ·
  <a href="docs/agents.md">Agents</a> ·
  <a href="docs/evaluating.md">Evaluating</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

```sh
curl -fsSL https://raw.githubusercontent.com/midagedev/bloomery/main/tools/release/install.sh | sh
~/.local/bin/bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M
```

These two commands install the latest release on Linux x86-64 with an NVIDIA GPU (or on Windows inside WSL2) and
serve Qwen3-30B-A3B on port 8080; [Use](#use) sends it a first request. A DGX Spark or another GB10 installs its own
archive or Docker image once a release carries them ([DGX Spark](#dgx-spark-and-other-gb10-machines)), and an Apple
Silicon Mac runs `bloomery-serve-metal` ([Apple Silicon](#apple-silicon)). Homebrew, Docker and the plain tarball are
under [Install](#install).

**From an agent.** Hand Claude Code, Codex or a script [`docs/agents.md`](docs/agents.md). The agent asks the machine
what fits, then starts the model by its name. stdout carries JSON, and every refusal ends with a named exit code and
the command to run next:

```sh
bloomery-serve --recommend --json                           # what this machine serves, and the fastest variant of each
bloomery-serve --model qwen3.8-flash-next --dry-run --json  # the start it would run; downloads nothing
bloomery-serve --model qwen3.8-flash-next --port 0          # start it: one JSON event a line, then "ready" with the url
```

On a Mac the server is `bloomery-serve-metal`, and the agent follows the "On a Mac" steps in the same page. To run a
coding agent on the model you just started — pi on a Mac, dsh on DeepSeek-V4.1-Flash, Claude Code or OpenCode on a
fast card set — see [Which coding agent](docs/agents.md#which-coding-agent).

## What it does

- **Models larger than the cards, and larger than the host's RAM.** The routed experts live on the cards and in host
  RAM, and the engine moves the experts it calls most onto the card while it runs. bloomery serves Muse Spark Open
  816B-A42B (225.6 GB of weights) and MiniMax-M3.1-Flash (250.1 GB, or 143.9 GB as Unsloth's 2-bit GGUF) on two cards
  with 72 GB between them. When the RAM
  cannot hold the host's share either, Qwen3.8, GLM-5.3-Flash, DeepSeek-V4.1-Flash, MiniMax-M3.1-Flash, Muse Spark
  Open, Solar-Open2 and the Nemotron-3.5 MoE models page the rest from NVMe.
- **Checkpoint folders as downloaded.** MiniMax-M3.1-Flash loads from its vendor's NVFP4 safetensors folder, and
  Qwen3.8 from its NVFP4 or FP4-Local folder, with no conversion and no requant. Everything else loads from GGUF.
- **Apple Silicon.** A Metal backend serves Qwen3.6-35B-A3B and Qwen3-30B-A3B on a Mac GPU. The verified path is
  mlx-community's 4-bit folders as downloaded. Q4_K_M GGUF files run, unverified (Qwen3.6 also from `UD-Q4_K_XL`).
  Ornith-1.5-35B and MiniCPM-V 4.7 35B-A3B (Q4_K_M, text only) run, unverified. Prompts run in batched passes, and a
  follow-up turn reads only its new messages ([below](#apple-silicon)).
- **llama-server compatible.** The same HTTP API (OpenAI and Anthropic), GGUF files and flag spellings, including the
  placement flags (`-dev`, `-ts`, `-ngl`, `--cpu-moe`, `-ot`, `--fit`) with llama's meaning. Your client works
  unchanged ([Use](#use)).
- **A prompt cache that survives other requests.** Saved prompt states stay in host RAM: a request that shares a prefix
  with an earlier one restores it, even after another request used the slot
  ([recorded on tape](https://tape.midagedev.com/r/u8fnay7pn4djqmy6vvaq)).
- **Several requests in one pass**, speculative decoding (MTP, DSpark) on by default where a model has a draft, and
  one CUDA graph per decode step with the CPU experts inside it.
- **Rust all the way down.** Every CUDA kernel is written in Rust and compiled with
  [cuda-oxide](https://github.com/NVIDIA/cuda-rust); the Metal backend's host side is Rust, its kernels Metal
  Shading Language.

## Install

Every path below installs a prebuilt release: no CUDA toolkit and no Rust toolchain. The installer, the Docker image
and Homebrew all take the latest release. To build from source instead: [`docs/BUILD.md`](docs/BUILD.md) on Linux,
[`docs/mac.md`](docs/mac.md#build-from-source) on a Mac.

### DGX Spark and other GB10 machines

0.3.11's GB10 archive (`linux-aarch64-cuda-sm121a`, built for the GB10's FP4 tensor cores) is built on a GB10 from the
tag and added to the release when it is ready, with a linux/arm64 variant of the Docker image `ghcr.io/midagedev/bloomery`
made from it. On a GB10, Docker then pulls that variant by itself and the installer picks that archive. Until the
archive is on the release, a GB10 builds from source: [`docs/contrib/dgx-spark.md`](docs/contrib/dgx-spark.md#21-build),
section 2.1.

The machine needs DGX OS 7 (Ubuntu 24.04, glibc 2.39) and an NVIDIA driver of R580 or newer (CUDA 13).

```sh
# Docker (needs the host's NVIDIA Container Toolkit; model downloads live in the bloomery-cache volume)
docker run --gpus all -p 8080:8080 -v bloomery-cache:/root/.cache/bloomery \
  ghcr.io/midagedev/bloomery --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M

# or the installer: it checks that every GPU is a GB10 and links bloomery-serve into ~/.local/bin
curl -fsSL https://raw.githubusercontent.com/midagedev/bloomery/main/tools/release/install.sh | sh
~/.local/bin/bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M
```

A 128 GB GB10 holds every routed expert of Qwen3.8-Flash-Next on the GPU. NVIDIA's FP4-Local and NVFP4 checkpoint
folders serve as downloaded, with no conversion: `bloomery-serve -m <folder>`, or `--hf nvidia/Qwen3.8-Flash-Next-NVFP4`.
In Docker, mount the folder and name it inside the container, for example
`-v /path/to/models:/models ... ghcr.io/midagedev/bloomery -m /models/Qwen3.8-Flash-Next-FP4-Local`. The slots, the
context each gets, the MTP draft and the memory each folder takes on a GB10 are in
[`docs/contrib/dgx-spark.md`](docs/contrib/dgx-spark.md#23-qwen38-on-one-spark-slots-mtp-and-memory), section 2.3.

### Linux x86-64

The machine needs an x86-64 CPU with AVX2, FMA and BMI2 (x86-64-v3), glibc 2.34 or newer, an NVIDIA GPU of compute
capability 8.6 or newer and driver R580 or newer (CUDA 13). An older driver is refused at the start.

```sh
# the installer: latest release, sha256-checked, symlink into ~/.local/bin
curl -fsSL https://raw.githubusercontent.com/midagedev/bloomery/main/tools/release/install.sh | sh

# or Homebrew
brew install midagedev/tap/bloomery

# or Docker (needs the host's NVIDIA Container Toolkit; model downloads live in the bloomery-cache volume)
docker run --gpus all -p 8080:8080 -v bloomery-cache:/root/.cache/bloomery \
  ghcr.io/midagedev/bloomery --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M

# or the plain tarball
tar -xzf bloomery-0.3.11-linux-x86_64-cuda-sm86.tar.gz && cd bloomery-0.3.11-linux-x86_64-cuda-sm86
bin/bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M
```

The tarballs are on the [releases page](https://github.com/midagedev/bloomery/releases)
([0.3.11](https://github.com/midagedev/bloomery/releases/tag/v0.3.11)).

From 0.3.11 the release carries an archive for the RTX 50 series (`linux-x86_64-cuda-sm120a`, compute capability 12.0:
the RTX 5090 and the RTX PRO Blackwell cards), built for the Blackwell FP4 and FP8 tensor cores. The installer takes it
when every GPU the machine lists is compute capability 12.0, and the `linux-x86_64-cuda-sm86` archive on any other
machine, or from a release without the new one. Docker and Homebrew stay on the sm_86 archive. No maintainer machine
has a 12.0 card, so this archive is checked by compiling it and by the users who run it; to take the sm_86 archive on
such a card, download it by name from the releases page.

### Windows (WSL2)

Run the Linux x86-64 release inside WSL2: `wsl --install` from an administrator PowerShell, reboot, then run the
Linux x86-64 commands above in the WSL shell. The binary reaches the GPU through the Windows driver.

### Apple Silicon

The Mac needs macOS 26.2 or newer; the binaries are signed and notarized. The server is `bloomery-serve-metal`.

```sh
# Homebrew
brew install midagedev/tap/bloomery
bloomery-serve-metal -m PATH/Qwen3.6-35B-A3B-Q4_K_M.gguf

# or the same installer as on Linux (bloomery-0.3.11-macos-arm64-metal.tar.gz on the releases page)
curl -fsSL https://raw.githubusercontent.com/midagedev/bloomery/main/tools/release/install.sh | sh
~/.local/bin/bloomery-serve-metal -m PATH/Qwen3.6-35B-A3B-Q4_K_M.gguf
```

Two architectures run, `qwen35moe` (Qwen3.6-35B-A3B and files with its shapes) and `qwen3moe` (Qwen3-30B-A3B), from
Q4_K_M GGUF files, one slot, text only. From 0.3.11 an mlx-community 4-bit folder also opens as downloaded
(`--hf mlx-community/<repo>` or `-m <folder>`; 0.3.10 converted it once with `bloomery-prepare`), and the MLX folders
of LFM3-1.7B, the dense Qwen3.8-27B and Qwen3-Coder-30B-A3B run too. Every file that runs,
the flags and the limits: [`docs/mac.md`](docs/mac.md).

## Use

Start the server with a model, then send it a request from another shell. After the installer, run it as
`~/.local/bin/bloomery-serve` unless `~/.local/bin` is on your `PATH`.

```sh
bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M

curl -s http://127.0.0.1:8080/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}'
```

The server listens on `127.0.0.1:8080` and is ready when `GET /health` answers 200. The first start downloads the
model into `~/.cache/bloomery/hf` (`BLOOMERY_CACHE` moves it) and compiles the GPU code for the card once, which takes
tens of seconds; the GB10 archive carries that code compiled already. `-m PATH` serves a file or folder you already
have. OpenAI clients take the base URL
`http://localhost:8080/v1`, and Claude Code runs with `ANTHROPIC_BASE_URL=http://localhost:8080`. In Docker, the
arguments after the image name go to `bloomery-serve`.

An agent or a script starts the server by a model's name: [`docs/agents.md`](docs/agents.md) has the exact commands,
the events to wait on and every exit code.

Start with `--place` and the `BLOOMERY_*` variables unset: the load picks the cards and the split onto the CPU itself, and
its plan line names what it picked. The file's architecture picks the seat (the model family's engine):

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
| `qwen38` | Qwen3.8-Flash-Next, checkpoint folders | NVIDIA's NVFP4 folder, its routed experts split between the card and the host, or the FP4-Local folder on a card that holds every routed expert (a DGX Spark's GB10), no conversion; also `--hf nvidia/Qwen3.8-Flash-Next-NVFP4` |
| `nemotron` | Nemotron 3 Nano 4B, Nemotron 3.5 Bolt 74B-A9B, Nemotron-3.5-Lightning 30B-A3B | the GGUF |
| `solar2` | Solar-Open2-250B, two cards | the GGUF, with `--place bp` |
| `raon1` | Raon-1-mini-preview | `raon-1-mini-preview-Q4_K_M.gguf`, its DFlash2 draft file beside it |

The server has one slot unless `--parallel N` asks for N, so a second request waits for the first. Unset,
`--ctx-size` gives each slot the largest context its plan holds; a set `--ctx-size` is the total the slots share,
and one slot takes it whole. `--slot-ctx S` names one request's context instead, so `--parallel 4 --slot-ctx 32768`
gives each of four slots 32,768 positions. On Qwen3.8, `--parallel auto` serves the most slots, up to 8, that memory
holds at that context. `--config FILE` reads the flags from one JSON file. A flag on the command line wins over
the file's, and both win over the flag's `LLAMA_ARG_*` variable and the `BLOOMERY_*` lever it stands for. A request
field the server does not know is ignored, as in llama-server, so a misspelled sampling field runs at its default. Which flags each seat takes, which setting wins and what a request's fields do:
[`docs/serving.md`](docs/serving.md#settings).

What each model needs in GPU memory, RAM and disk is in [`docs/models.md`](docs/models.md). In short: Qwen3-30B and
Qwen3.6 run on a 24 GB card (or 12–16 GB, split onto the CPU by itself); Qwen3.8 wants 128 GB of RAM, GLM-5.3 and
V4.1 256 GB, and less RAM runs with the overflow paged from NVMe. The API, every flag, concurrent requests, drafts and
adaptive residency: [`docs/serving.md`](docs/serving.md).

## Hardware

What each platform needs is under [Install](#install), and what each model needs in GPU memory, RAM and disk is in
[`docs/models.md`](docs/models.md). These machines have run it:

| What runs | Run here | Run by users |
|---|---|---|
| NVIDIA cards | RTX 3090, RTX A6000, RTX 3060 (WSL2) | RTX 5090 ([#2](https://github.com/midagedev/bloomery/issues/2)), DGX Spark (GB10: [`docs/contrib/dgx-spark.md`](docs/contrib/dgx-spark.md)), one model across several Sparks |
| Apple Silicon | M1 Pro, M4 Pro 48 GB | M3 Ultra, M5 Ultra ([#593](https://github.com/midagedev/bloomery/issues/593)) |

Ada cards (sm_89) have no report yet, and AMD GPUs are not supported yet. On an M5 the device picks the neural
accelerators for the prompt's GEMMs. What each part of the machine costs in a decode step:
[`docs/HARDWARE.md`](docs/HARDWARE.md).

## Status (0.3.11)

Four common machines, one table each, tok/s. Every other record (two cards, older releases, other cards and Macs, the
reference engines) is in [`docs/numbers.md`](docs/numbers.md), with its machine and conditions. The rows below were
measured on 0.3.10; 0.3.11's own rows, Qwen3.8-27B's among them, are in its
[release notes](https://github.com/midagedev/bloomery/releases/tag/v0.3.11).

**One RTX 3090 24 GB**, on our workstation (32-core AVX2 host, 256 GB of RAM, the 3090 at its 250 W cap): the tagged
0.3.10 tree at the serving defaults (`--place` unset, every lever unset, speculative drafts on where a model has one, no
`--parallel`, `--ctx-size 4300`). Means of two rounds; four for DeepSeek-V4.1-Flash and Qwen3.8-Flash-Next.

| Model | Where it runs | Prompt, 512 tokens | Prompt, 4,096 tokens | Decode at depth 512 | Decode at depth 4,096 |
|---|---|---|---|---|---|
| MiniMax-M3.1-Flash `UD-Q2_K_XL` | card + CPU | 105.16 | 97.56 | 23.41 | 20.40 |
| Flint 40B-A2.5B `UD-Q4_K_XL` | card + CPU | 879.20 | 750.19 | 87.86 | 85.84 |
| Nanbeige4.3-33B `Q4_K_M` | the whole model on the card | 745.46 | 693.76 | 24.22 | 23.23 |
| DeepSeek-V4.1-Flash `Q3_K_M` | card + CPU | 167.41 | 300.46 | 34.73 ¹ | 34.94 |
| GLM-5.3-Flash `UD-Q4_K_XL` | card + CPU | 178.99 | 204.47 | 30.06 | 26.18 |
| Qwen3.8-Flash-Next `UD-Q4_K_XL` | card + CPU | 736.66 | 1,662.72 ² | 83.36 | 74.34 |
| Qwen3.6-35B-A3B `Q4_K_M` | the whole model on the card | 5,123.05 | 6,652.24 | 188.31 | 180.68 |

¹ rounds from 31.90 to 40.67 · ² the first round after a load read 1,440.33, the next three 1,735 to 1,739.

**DGX Spark (GB10, 128 GB)**, measured by [@juliankang4](https://github.com/juliankang4) on their own machine, one
request at a time, greedy, 128 tokens, median of three rounds. The Qwen3.8-Flash-Next rows are the 0.3.10 release on
NVIDIA's FP4-Local and NVFP4 folders as downloaded and on `UD-Q4_K_XL`, with the MTP draft from Unsloth's
`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`; the Qwen3.6 row is 0.3.0's
([#46](https://github.com/midagedev/bloomery/issues/46#issuecomment-6116080838)).

| Model (release) | ~4,096-token prompt: first token | Decode after it | Short prompt decode |
|---|---|---|---|
| Qwen3.8-Flash-Next, FP4-Local folder, MTP on (0.3.10) | 2.05 s code, 2.08 s prose | 54.2 code, 39.9 prose | — |
| Qwen3.8-Flash-Next, FP4-Local folder, draft off (0.3.10) | 1.98 s code, 2.08 s prose | 30.8 code, 29.8 prose | — |
| Qwen3.8-Flash-Next, NVFP4 folder, MTP on (0.3.10) | 2.22 s code, 2.19 s prose | 47.9 code, 33.4 prose | — |
| Qwen3.8-Flash-Next, NVFP4 folder, draft off (0.3.10) | 2.11 s code, 2.07 s prose | 22.7 code, 22.7 prose | — |
| Qwen3.8-Flash-Next `UD-Q4_K_XL`, MTP on (0.3.10) | 2.80 s code, 2.75 s prose | 50.0 code, 35.2 prose | — |
| Qwen3.6-35B-A3B `Q4_K_M` (0.3.0) | 0.90 s | 89.8 | 93.5 |

**RTX 5090 32 GB**, measured by [@juliankang4](https://github.com/juliankang4) at 0.3.10 on native Linux with 126 GB of
RAM, one request at a time at the serving defaults, greedy, 128 tokens, median of three rounds.

| Model | ~4K prompt: prefill tok/s, first token | Decode after it | Short prompt decode |
|---|---|---|---|
| Qwen3.6-35B-A3B `Q4_K_M`, whole on the card | 14,249, 287 ms code; 14,298, 286 ms prose | 344.1 code, 344.2 prose | — |

With its MTP draft, on the same card at a power limit of 400 W (default 600 W), at `3ebdff767`: unsloth's
`UD-Q4_K_M`, which carries the next-token layer, `BLOOMERY_DRAFT=mtp`, 4,096-id prompts, 128 greedy tokens, one request
at a time, median of three rounds
([`docs/numbers.md`](docs/numbers.md#rtx-5090-qwen36-35b-a3b-with-its-mtp-draft-by-juliankang4)).

| Model | Prefill tok/s | Decode tok/s | First token |
|---|---|---|---|
| Qwen3.6-35B-A3B `UD-Q4_K_M`, MTP draft on | 8,747 | 268.2 | 470 ms |

**M5 Ultra**, measured by [@avlp12](https://github.com/avlp12) at 0.3.9's Metal code
([#593](https://github.com/midagedev/bloomery/issues/593); conditions in [`docs/mac.md`](docs/mac.md#numbers)).

| Model | Prompt, 4,096 ids | Decode after 512 ids |
|---|---|---|
| Qwen3.6-35B-A3B | 5,972 | 211–212 |

The release notes hold the commits, the release check on the tagged binary, every off cell and the contributors' work.

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

## Known limits (0.3.11)

- DeepSeek V4.1 on two cards (`--place bp`) is not faster than one card (`--place a`). Unset `--place` picks `a`.
- The `nemotron`, `raon1`, `solar2` and `musespark` seats serve one slot; `--parallel` above 1 is refused by name.
  The `nemotron`, `raon1` and `solar2` seats keep no saved prompt states.
- `--place bp` is refused by name on the `nemotron` seat, Qwen3.6 and Qwen3-30B; they run one card plus the host tier.
- Nemotron-3.5-Lightning `UD-Q4_K_XL` is a placed load: its whole-card expert body is IQ2_XXS, and the other expert
  types stay on the host.
- Muse Spark Open always thinks: a turn's first tokens are its reasoning, so a short `max_tokens` can end with empty
  `content`.
- MiniMax-M3 (not M3.1) was not checked for 0.3.11; each family's newest model was.
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
| [`docs/agents.md`](docs/agents.md) | Starting the server from an agent or a script: the picks, the events, the exit codes |
| [`docs/claude-code.md`](docs/claude-code.md) | Claude Code on a local model, on an NVIDIA card or a Mac, with measured first-turn times |
| [`docs/pi.md`](docs/pi.md) | pi on a local model: the smallest first request of the surveyed agents, on an NVIDIA card or a Mac |
| [`docs/dsh.md`](docs/dsh.md) | DeepSeek Harness (dsh) on DeepSeek-V4.1-Flash: the server, the profile patch, what to expect |
| [`docs/models.md`](docs/models.md) | Recommended files and what each model needs |
| [`docs/numbers.md`](docs/numbers.md) | Every measured table beyond the Status tables: two cards, older releases, other cards and Macs |
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
