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
  <a href="https://github.com/midagedev/bloomery/releases/tag/v0.3.13">0.3.13 release notes</a> ·
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
serve Qwen3-30B-A3B on port 8080; [Use](#use) sends it a first request. The same installer takes the GB10 archive on a
DGX Spark ([DGX Spark](#dgx-spark-and-other-gb10-machines)), and an Apple Silicon Mac runs `bloomery-serve-metal`
([Apple Silicon](#apple-silicon)). Homebrew, Docker and the plain tarball are under [Install](#install).

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
fast card set, see [Which coding agent](docs/agents.md#which-coding-agent).

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
- **A quantizer that writes the final set.** `bloomery-quantize` writes a GGUF set's routed-expert stacks as IQ1_S or
  IQ2_XXS on the card and every other tensor by the host's K-quant rules, the metadata and shard layout as
  `llama-quantize` writes them, byte-identical to its output for the same input and options ([`docs/quantize.md`](docs/quantize.md)).
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

The release carries a GB10 archive (`linux-aarch64-cuda-sm121a`, built for the GB10's FP4 tensor cores, its GPU code
compiled ahead) and a linux/arm64 variant of the Docker image `ghcr.io/midagedev/bloomery` made from it. On a GB10
the installer picks that archive and Docker pulls that variant by itself. The machine needs DGX OS 7 (Ubuntu 24.04,
glibc 2.39) and an NVIDIA driver of R580 or newer (CUDA 13). 0.3.13's GB10 archive is added to the release when it is
ready; until then the installer stops at its download by name, and `install.sh --version v0.3.12` installs 0.3.12.

```sh
# Docker (needs the host's NVIDIA Container Toolkit; model downloads live in the bloomery-cache volume)
docker run --gpus all -p 8080:8080 -v bloomery-cache:/root/.cache/bloomery \
  ghcr.io/midagedev/bloomery --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M

# or the installer: it checks that every GPU is a GB10 and links bloomery-serve into ~/.local/bin
curl -fsSL https://raw.githubusercontent.com/midagedev/bloomery/main/tools/release/install.sh | sh
~/.local/bin/bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M
```

A 128 GB GB10 holds every routed expert of Qwen3.8-Flash-Next on the GPU, and NVIDIA's FP4-Local and NVFP4 checkpoint
folders serve as downloaded: `bloomery-serve -m <folder>`. In Docker, mount the folder and name it inside the
container, for example `-v /path/to/models:/models ... ghcr.io/midagedev/bloomery -m
/models/Qwen3.8-Flash-Next-FP4-Local`. Which folder to pick, the slots, the context, the MTP draft and the memory each
start takes: [`docs/dgx-spark.md`](docs/dgx-spark.md).

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
tar -xzf bloomery-0.3.13-linux-x86_64-cuda-sm86.tar.gz && cd bloomery-0.3.13-linux-x86_64-cuda-sm86
bin/bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M
```

The tarballs are on the [releases page](https://github.com/midagedev/bloomery/releases)
([0.3.13](https://github.com/midagedev/bloomery/releases/tag/v0.3.13)).

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

# or the same installer as on Linux (bloomery-0.3.13-macos-arm64-metal.tar.gz on the releases page)
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
`~/.local/bin/bloomery-serve` unless `~/.local/bin` is on your `PATH`. Every flag has a default, so a model is all a
start needs.

```sh
bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M

curl -s http://127.0.0.1:8080/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}'
```

The server listens on `127.0.0.1:8080` and is ready when `GET /health` answers 200. The first start downloads the
model into `~/.cache/bloomery/hf` and compiles the GPU code for the card once. `-m PATH` serves a file or folder you
already have. OpenAI clients take the base URL `http://localhost:8080/v1`, and Claude Code runs with
`ANTHROPIC_BASE_URL=http://localhost:8080`. In Docker, the arguments after the image name go to `bloomery-serve`.

Leave `--place` and the `BLOOMERY_*` variables unset at first: the load picks the cards and the split onto the CPU, and
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
| `qwen38` | Qwen3.8-Flash-Next, checkpoint folders | NVIDIA's NVFP4 or FP4-Local folder, its routed experts split between the card and the host, no conversion; also `--hf nvidia/Qwen3.8-Flash-Next-NVFP4` |
| `nemotron` | Nemotron 3 Nano 4B, Nemotron 3.5 Bolt 74B-A9B, Nemotron-3.5-Lightning 30B-A3B | the GGUF |
| `solar2` | Solar-Open2-250B, two cards | the GGUF, with `--place bp` |
| `mistrallarge4` | Mistral Large 4 (1T, 52B active) | unsloth's `UD-IQ1_S` or Mix-256G GGUF; `--place a` on one 24 GB card |
| `raon1` | Raon-1-mini-preview | `raon-1-mini-preview-Q4_K_M.gguf`, its DFlash2 draft file beside it |

The server has one slot, so a second request waits for the first. `--parallel 4 --slot-ctx 32768` serves four
requests at once with 32,768 positions each; unset, a slot takes the largest context its plan holds. `--parallel auto`
serves the most slots, up to 8, that memory holds at one request's context. A request field the server does not know is ignored, as in llama-server, so a misspelled sampling field runs at its default.
[`docs/serving.md`](docs/serving.md) is the serving guide: slots and context, drafts, placement, the settings file,
API keys and every route.

An agent or a script starts the server by a model's name: [`docs/agents.md`](docs/agents.md) has the exact commands,
the events to wait on and every exit code.

What each model needs in GPU memory, RAM and disk is in [`docs/models.md`](docs/models.md). In short: Qwen3-30B and
Qwen3.6 run on a 24 GB card (or 12 to 16 GB, split onto the CPU by itself); Qwen3.8 wants 128 GB of RAM, GLM-5.3 and
V4.1 256 GB, and less RAM runs with the overflow paged from NVMe.

A paged run reads faster from two NVMe drives than from one: a byte-identical copy of the shards on a second drive,
named by a `<prefix>.shard-dirs` sidecar beside the first shard ([`docs/serving.md` 5.4](docs/serving.md#54-when-the-host-cannot-hold-the-experts-either)),
lets the NVMe tier's reader pick each expert's drive by id, so a layer's misses land on both drives at once. Measured
on the 0.3.13 tarball's `bloomery-serve`, RTX A6000 + RTX 3090, one cold request of 512 ids then 128 generated, three
rounds each, the host room pinned so the plan pages (`BLOOMERY_HOST_ROOM`), the same set with and without the sidecar
(`docs/cards/mirror-ab-*.card`):

| Model | Room | `--place` | Decode tok/s, one drive → two | Prompt tok/s, one drive → two |
|---|---|---|---|---|
| DeepSeek V4.1-Flash Q3_K_M | 80G | `bp` | 16.9 → 22.4 (+32 %) | 26.1 → 40.5 (+55 %) |
| Mistral Large 4 UD-IQ1_S | 80G | `a` | 3.02 → 4.23 (+40 %) | 13.5 → 21.5 (+59 %) |
| GLM-5.3-Flash UD-Q4_K_XL | 48G | `bp` | 5.51 → 8.22 (+49 %) | 30.3 → 54.0 (+78 %) |
| Kimi K3.1 Pro IQ1_S | 232G | `bp` | 2.94 with two drives (the prompt pass, the default); 4.99 under `--prefill steps`, in the [0.3.13 notes](https://github.com/midagedev/bloomery/releases/tag/v0.3.13) | 12.5; 3.80 under `--prefill steps` |

One drive reads at ~4.2–4.7 GB/s on these runs and two at ~5.9–7.2 GB/s. The sidecar does nothing when the plan
pages through the file mapping instead of the tier's arena (a room that moves under half the host leg's bytes prints
`arena=0`); the gain is in the paged part of the step, so a plan that pages little reads near 0.

## Hardware

What each platform needs is under [Install](#install), and what each model needs in GPU memory, RAM and disk is in
[`docs/models.md`](docs/models.md). These machines have run it:

| What runs | Run here | Run by users |
|---|---|---|
| NVIDIA cards | RTX 3090, RTX A6000, RTX 3060 (WSL2) | RTX 5090 ([#2](https://github.com/midagedev/bloomery/issues/2)), DGX Spark (GB10: [`docs/dgx-spark.md`](docs/dgx-spark.md)), one model across several Sparks |
| Apple Silicon | M1 Pro, M4 Pro 48 GB | M3 Ultra, M5 Ultra ([#593](https://github.com/midagedev/bloomery/issues/593)) |

Ada cards (sm_89) have no report yet, and AMD GPUs are not supported yet. On an M5 the device picks the neural
accelerators for the prompt's GEMMs. What each part of the machine costs in a decode step:
[`docs/HARDWARE.md`](docs/HARDWARE.md).

## Status (0.3.13)

Four common machines, one table each, tok/s. Every other record (two cards, older releases, other cards and Macs, the
reference engines) is in [`docs/numbers.md`](docs/numbers.md), with its machine and conditions. Each table names the
release or commit it was measured on; 0.3.13's own rows, the DGX Spark expert tier's among them, are in its
[release notes](https://github.com/midagedev/bloomery/releases/tag/v0.3.13).

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

¹ rounds from 31.90 to 40.67.

² the first round after a load read 1,440.33, the next three 1,735 to 1,739.

The DGX Spark and RTX 5090 tables were measured by [@juliankang4](https://github.com/juliankang4) at main
`e7180ea9`, on the same four models with the MTP draft on: NVIDIA's FP4-Local and NVFP4 folders as downloaded (their
own MTP layer), Unsloth's Qwen3.8 `UD-Q4_K_XL` with its shared Q8_0 draft file, and Unsloth's Qwen3.6 `UD-Q4_K_M` with
its in-file MTP layer. Each server ran one slot of 16,896 positions (`--parallel 1 --ctx-size 16896`) with
`BLOOMERY_HOST_ROOM=96G`. A row is one request of a 512 or 4,096-id prompt from a code, an English prose or a Korean
prose corpus, sent as token ids to `/completion` with streaming, `cache_prompt` false, `temperature` 0, `top_k` 1 and
128 tokens. Prefill and decode are the server's timings, and first token is the client's time from sending the request
to the first streamed token. Each value is the median of three rounds; each round started a new server and sent two
warm-up requests first.

**DGX Spark (GB10, 128 GB)**, every model wholly on the GB10. Earlier Spark rows are in
[`docs/numbers.md`](docs/numbers.md#dgx-spark-qwen38-flash-next-on-the-fp4-local-and-nvfp4-folders-0310).

| Model | Prompt | Prefill tok/s, 512 ids | Prefill tok/s, 4,096 ids | Decode tok/s, 512 ids | Decode tok/s, 4,096 ids | First token ms, 512 ids | First token ms, 4,096 ids |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-Flash-Next, FP4-Local folder | Code | 1,317 | 2,840 | 59.1 | 69.6 | 393 | 1,448 |
|  | English prose | 1,651 | 2,894 | 54.8 | 39.4 | 314 | 1,421 |
|  | Korean prose | 1,447 | 2,850 | 49.4 | 49.5 | 359 | 1,443 |
| Qwen3.8-Flash-Next, NVFP4 folder | Code | 1,225 | 2,544 | 45.3 | 54.8 | 423 | 1,615 |
|  | English prose | 1,526 | 2,596 | 38.4 | 29.5 | 340 | 1,583 |
|  | Korean prose | 1,334 | 2,558 | 36.0 | 33.8 | 388 | 1,607 |
| Qwen3.8-Flash-Next `UD-Q4_K_XL` | Code | 824 | 1,708 | 46.4 | 48.8 | 626 | 2,403 |
|  | English prose | 1,086 | 1,747 | 45.1 | 38.6 | 476 | 2,351 |
|  | Korean prose | 929 | 1,718 | 47.3 | 41.6 | 556 | 2,390 |
| Qwen3.6-35B-A3B `UD-Q4_K_M` | Code | 1,860 | 2,848 | 78.0 | 83.8 | 279 | 1,444 |
|  | English prose | 2,189 | 2,828 | 84.4 | 68.0 | 238 | 1,454 |
|  | Korean prose | 2,090 | 2,826 | 81.6 | 75.3 | 249 | 1,454 |

**RTX 5090 32 GB**, on native Linux. The Qwen3.8 models do not fit the card: it holds part of the routed experts and
reads the rest from host RAM ("card + host RAM"). Qwen3.6 runs wholly on the card. Earlier RTX 5090 rows are in
[`docs/numbers.md`](docs/numbers.md#rtx-5090-qwen36-35b-a3b-with-its-mtp-draft-by-juliankang4).

| Model | Prompt | Prefill tok/s, 512 ids | Prefill tok/s, 4,096 ids | Decode tok/s, 512 ids | Decode tok/s, 4,096 ids | First token ms, 512 ids | First token ms, 4,096 ids |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-Flash-Next, FP4-Local folder, card + host RAM | Code | 558 | 1,382 | 73.7 | 82.9 | 920 | 2,966 |
|  | English prose | 659 | 1,522 | 53.7 | 52.3 | 780 | 2,694 |
|  | Korean prose | 528 | 1,602 | 49.1 | 55.7 | 973 | 2,560 |
| Qwen3.8-Flash-Next, NVFP4 folder, card + host RAM | Code | 554 | 1,187 | 59.5 | 63.9 | 927 | 3,454 |
|  | English prose | 644 | 1,378 | 48.1 | 46.7 | 798 | 2,975 |
|  | Korean prose | 523 | 1,476 | 46.0 | 46.4 | 981 | 2,779 |
| Qwen3.8-Flash-Next `UD-Q4_K_XL`, card + host RAM | Code | 517 | 1,124 | 67.6 | 64.8 | 992 | 3,647 |
|  | English prose | 616 | 1,284 | 49.1 | 44.9 | 833 | 3,193 |
|  | Korean prose | 509 | 1,385 | 52.0 | 50.6 | 1,008 | 2,961 |
| Qwen3.6-35B-A3B `UD-Q4_K_M` | Code | 7,922 | 11,223 | 273.8 | 294.3 | 67 | 368 |
|  | English prose | 8,326 | 11,238 | 296.0 | 244.5 | 64 | 367 |
|  | Korean prose | 8,182 | 11,245 | 274.9 | 275.0 | 65 | 367 |

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

## Known limits (0.3.13)

- DeepSeek V4.1 on two cards (`--place bp`) is not faster than one card (`--place a`). Unset `--place` picks `a`.
- The `nemotron`, `raon1`, `solar2` and `musespark` seats serve one slot; `--parallel` above 1 is refused by name.
  The `nemotron`, `raon1` and `solar2` seats keep no saved prompt states.
- `--place bp` is refused by name on the `nemotron` seat, Qwen3.6 and Qwen3-30B; they run one card plus the host tier.
- Nemotron-3.5-Lightning `UD-Q4_K_XL` is a placed load: its whole-card expert body is IQ2_XXS, and the other expert
  types stay on the host.
- Muse Spark Open always thinks: a turn's first tokens are its reasoning, so a short `max_tokens` can end with empty
  `content`.
- MiniMax-M3 (not M3.1) was not checked for 0.3.13; each family's newest model was.
- Flint's greedy text has no CPU or llama.cpp referee yet; the card path's gates and the serving checks hold it.
- DeepSeek V4.1 runs routed experts on a card only as Q3_K gate·up with a Q4_K down; other expert types (ISTA-DASLab's
  IQ1_M, IQ2_XXS and IQ2_XS) run on the host tier.
- DeepSeek V4.1 on one 24 GB card from a file with an r8 sidecar beside it is refused at the load by name (the NVMe
  tier does not fill from the sidecar yet); `BLOOMERY_R8=off` reads the source file with the same bits.
- GLM-5.3 holds at most 16,384 positions a slot; a `--ctx-size` whose slot share is larger is refused at the start.
- Kimi K3.1 Pro (our `IQ1_S` set, a Q4_K trunk) loads at `--place bp` only (two cards, the host and the NVMe expert
  tier). Past the lightning indexer's top-k (2,048 positions) the latent layers attend the rows the indexer picks,
  so `--ctx-size` is not held to the top-k. The seat runs a prompt as placed passes; the pass's plan keeps fewer experts in
  the NVMe arena, so a step decodes slower than under `--prefill steps` (2.94 against 4.99 tok/s, the prompt 12.5
  against 3.80, at 512 ids and 512 generated): steps is the faster pick when the reply is longer than about 1.3 times
  the prompt. The seat serves the model's think span as plain text. The set's chat template is unsloth's
  Kimi-K3 template (the checkpoint ships none), and its importance matrix came from `llama-imatrix`; the quantization
  itself was written by `bloomery-quantize`. Kimi K3 (unsloth's `UD-IQ1_S`, 594 GB) does not load at this tag: its
  dense trunk is 51.4 GB, past one 48 GB card.
- The expert tier on another machine (`BLOOMERY_TIER_NODES`, `tier_server`) serves DeepSeek V4.1-Flash, GLM-5.3, Qwen3.8,
  Solar-Open2 and Kimi K3 / K3.1; a node serves no prompt batch (the batch is the host's), and the RAM arena
  (`BLOOMERY_NVME_ARENA`) refuses a plan with a rank file (`BLOOMERY_EXPERT_RANK`) by name.

Every limit, by seat: [`docs/serving.md`](docs/serving.md#12-limits).

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
| [`docs/serving.md`](docs/serving.md) | The serving guide: start, slots and context, drafts, placement, settings, API keys, every route, limits |
| [`docs/agents.md`](docs/agents.md) | Starting the server from an agent or a script: the picks, the events, the exit codes |
| [`docs/claude-code.md`](docs/claude-code.md) | Claude Code on a local model, on an NVIDIA card or a Mac, with measured first-turn times |
| [`docs/pi.md`](docs/pi.md) | pi on a local model: the smallest first request of the surveyed agents, on an NVIDIA card or a Mac |
| [`docs/dsh.md`](docs/dsh.md) | DeepSeek Harness (dsh) on DeepSeek-V4.1-Flash: the server, the profile patch, what to expect |
| [`docs/quantize.md`](docs/quantize.md) | `bloomery-quantize`: the final quantized set in one pass (IQ1_S and IQ2_XXS experts on the card, the trunk by the host's K-quant rules), how its bytes are proved, the resource timeline |
| [`docs/models.md`](docs/models.md) | Recommended files and what each model needs |
| [`docs/numbers.md`](docs/numbers.md) | Every measured table beyond the Status tables: two cards, older releases, other cards and Macs |
| [`docs/performance.md`](docs/performance.md) | Number conditions, why it is fast, verification, recorded sessions |
| [`docs/dgx-spark.md`](docs/dgx-spark.md) | DGX Spark (GB10): install, Qwen3.8's checkpoint folders, slots, context and memory, speed |
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
