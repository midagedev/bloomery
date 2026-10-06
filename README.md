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
  host, DeepSeek-V4.1-Flash `Q3_K_M` (347 GB) serves two requests at 34.4 tok/s in total and GLM-5.3-Flash at 26.9.
- **Several requests, one pass.** With `--parallel 2` two busy streams run through the model together: 25–37 %
  more tokens a second in total than one stream
  ([measured on the A6000](https://github.com/midagedev/rig-log/blob/main/log/2026-10-06.md#rel021-slots)).
- **llama-server compatible.** The same HTTP API (OpenAI and Anthropic), GGUF files and flag spellings: your client
  works unchanged.
- **Rust all the way down.** Every CUDA kernel is written in Rust and compiled with
  [cuda-oxide](https://github.com/NVIDIA/cuda-rust).

```sh
brew install midagedev/tap/bloomery        # Linux x86-64, NVIDIA sm_86+
bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M --port 8080
```

## Status (0.2.4)

| Area | State |
|---|---|
| Linux x86-64, NVIDIA sm_86 (RTX 3090, RTX A6000, RTX 3060) | Runs. Every number here comes from these cards |
| NVIDIA driver | R580 or newer (CUDA 13); an older driver is refused at the start |
| Ada and Blackwell (sm_89, sm_120) | Not yet run here. The sm_86 PTX compiles for the card at the first start. [Reports welcome](docs/evaluating.md#help-wanted-hardware-we-have-not-run) |
| Host CPU | x86-64 with AVX2 (x86-64-v3), AMD or Intel |
| macOS, Apple silicon · AMD GPUs, Windows without WSL2 | Not supported yet · Not supported |
| WSL2 (Windows) | Qwen3.8, GLM-5.3 and DeepSeek-V4.1 run only with `BLOOMERY_RESIDENCY=off` for now ([#1](https://github.com/midagedev/bloomery/issues/1)); the other models run as they are |
| OpenAI and Anthropic APIs, streaming, tool calls | Every generative model |
| Several requests at once (`--parallel`, default 2) | One pass on V4.1, GLM-5.3, Qwen3.8 and a whole-card Qwen3-30B or Qwen3.6; in turn on the rest |
| MTP draft (Qwen3.8, GLM-5.3) | On by default |
| Cards | One card, or one card plus one expert-tier card (`--place bp`) |
| Vision input (V4.1) | Not in this release |

**WSL2 limit** ([#1](https://github.com/midagedev/bloomery/issues/1)): adaptive residency's expert swap waits on a
word that a WSL2 driver queues behind the request's own readback, so Qwen3.8, GLM-5.3 and DeepSeek-V4.1 stop a few
tokens into a request (one CPU core busy, the GPU idle). On WSL2, start them with the setting in front:
`BLOOMERY_RESIDENCY=off bloomery-serve …`. Native Linux is not affected.

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
tar -xzf bloomery-0.2.4-linux-x86_64-cuda-sm86.tar.gz && cd bloomery-0.2.4-linux-x86_64-cuda-sm86
bin/bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M --port 8080
```

The tarballs are on the [releases page](https://github.com/midagedev/bloomery/releases)
([0.2.4](https://github.com/midagedev/bloomery/releases/tag/v0.2.4)). From source: [`docs/BUILD.md`](docs/BUILD.md).

## Use

One binary, five seats; the file's architecture picks the seat, and `--place` can stay unset on every model.

| Seat | Serves | One command |
|---|---|---|
| `ds41` | [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | `bloomery-serve --hf vcruz305/DeepSeek-V4.1-Flash-GGUF:Q3_K_M` |
| `glm` | [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | `bloomery-serve --hf unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL` |
| `qwen38` | [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | `bloomery-serve --hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL` |
| `qwen3` | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | `bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M` |
| `decide` | [Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) (a decision model) | `bloomery-serve --hf bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M` |

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}'
```

Claude Code: `ANTHROPIC_BASE_URL=http://localhost:8080`. The API, the flags, concurrent requests, drafts and
adaptive residency: [`docs/serving.md`](docs/serving.md). What each model needs in GPU, RAM and disk:
[`docs/models.md`](docs/models.md). In short, Qwen3-30B and Qwen3.6 run on a 24 GB card (or 12–16 GB, split onto
the CPU by itself); Qwen3.8 wants 128 GB of RAM, GLM-5.3 and V4.1 256 GB.

## Numbers

Two requests at once (`--parallel 2`, both busy) after a 512-token prompt, decode tok/s of both together, on a
32-core AVX2 host with 8 DDR4 channels and 256 GB of RAM; the 3090 at a 250 W cap. Conditions and footnotes:
[`docs/performance.md`](docs/performance.md#conditions-of-the-numbers); every row and the history:
[rig-log's bench page](https://github.com/midagedev/rig-log/blob/main/docs/bloomery-bench.md).

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

¹ measured when Qwen3.6 ran two requests in turn (one request's rate); one pass since 0.2.3, not re-measured · ² adaptive residency off · ³ two requests in turn on two cards
today. Qwen3.8 and GLM-5.3 ran with the MTP draft off.

## Why it is fast

Each point has a same-engine measurement in [`docs/performance.md`](docs/performance.md#why-it-is-fast):

- **Experts move to where they are used**: V4.1 decode 29.48 → 43.86 tok/s with adaptive residency.
- **The card and the CPU work on a prompt at once** (V4.1): two batches in flight, 1.20× at P = 4096.
- **Qwen prompts as int8 tensor-core GEMMs**: P = 4096 from 357 to 2,615 tok/s, then a prefill flash kernel.
- **CPU experts at memory speed**: AVX2 kernels on the stored quantization, grouped dispatch on a pinned pool.
- **One CUDA graph a decode step**, with the CPU experts inside it, and speculative decoding (MTP, DSpark).

Output is checked layer by layer against ik_llama.cpp (V4.1: PPL 2.2401 against 2.2378, same top token 97.46 %),
and every gate is shown to fail on its defect before the fix lands ([how](docs/performance.md#how-it-is-verified)).

## Docs

| Page | What it covers |
|---|---|
| [`docs/evaluating.md`](docs/evaluating.md) | A checklist to run it and compare it fairly; hardware reports |
| [`docs/serving.md`](docs/serving.md) | llama-server compatibility, flags, seats, concurrent requests, limits |
| [`docs/models.md`](docs/models.md) | Recommended files and what each model needs |
| [`docs/performance.md`](docs/performance.md) | Number conditions, why it is fast, verification, recorded sessions |
| [`docs/BUILD.md`](docs/BUILD.md), [`docs/cuda-oxide.md`](docs/cuda-oxide.md) | Building from source; the kernels in Rust |
| [`docs/upstream/`](docs/upstream/) | Pull requests to ik_llama.cpp, cuda-oxide, llama.cpp and others |
| [`docs/README.md`](docs/README.md) | Every other page and the working records |

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
