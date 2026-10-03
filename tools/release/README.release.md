# bloomery — prebuilt Linux release

bloomery is an LLM inference engine for one workstation: a Rust host, CUDA kernels written in Rust, and AVX2 kernels
for the experts that stay on the CPU. Source, measurements and the full README: https://github.com/midagedev/bloomery

## What is in this archive

| File | What it serves |
|---|---|
| `bin/bloomery-serve` | every seat in one binary (the model file's architecture picks it; `--help` lists them): llama-server's HTTP API (`/v1/chat/completions`, `/completion`, streaming) for DeepSeek-V4.1-Flash, GLM-5.3-Flash, Qwen3.8-Flash-Next, Qwen3.6-35B-A3B, Qwen3-30B-A3B, and Cloudflare Clef-Flash's SystemOne API (`POST /v1/systemone`) as its decide seat |

## Requirements

- Linux x86-64 with glibc 2.34 or newer (Ubuntu 22.04+, Debian 12+, RHEL 9+). The CPU needs AVX2 and FMA (x86-64-v3).
- An NVIDIA GPU of compute capability 8.6 or newer: the archive carries sm_86 PTX, which the driver compiles for the
  card on first start. Run on an RTX A6000 and an RTX 3090 (driver 615.71, Ubuntu 24.04) and, for Clef-Flash, on an
  RTX 3060 under WSL2 (Windows driver 591.86), where the answers equal the A6000's byte for byte. Other cards and
  drivers are untested.
- The NVIDIA driver (`libcuda.so.1`). No CUDA toolkit is needed. The first start compiles the GPU code for the card
  and caches it (about 10–25 s measured); later starts take a few seconds. Do not set `CUDA_CACHE_DISABLE=1`: every
  start then recompiles and takes minutes.
- What fits where:
  - Clef-Flash: run on a 12 GB card with the `Q3_K_S` file (4.26 GB); the files run from 4.26 GB (`Q3_K_S`) to 9.55 GB (`Q8_0`).
  - Qwen3-30B-A3B and Qwen3.6-35B-A3B at `Q4_K_M`: a 24 GB card whole on device 0, or a 12-16 GB card with `--place a` (the routed experts on the CPU; the server plans the split itself when the file does not fit the card's free bytes).
  - V4.1, GLM-5.3, Qwen3.8 (their experts partly on the CPU): a 24–48 GB card and a host with 256 GB of RAM.

## Run

Without unpacking by hand, the installer fetches the latest release, checks its sha256 and puts a
`bloomery-serve` symlink into `~/.local/bin`:

```sh
curl -fsSL https://raw.githubusercontent.com/midagedev/bloomery/main/tools/release/install.sh | sh
```

On Windows, run this archive inside WSL2 (Ubuntu): `wsl --install` from an administrator PowerShell, reboot, then
extract and run as below inside the WSL shell — the binaries see the GPU through Windows' driver (tested on an RTX
3060 12 GB under WSL2, Windows driver 591.86; the installer runs there too).

```sh
tar -xzf bloomery-*-linux-x86_64-cuda-sm86.tar.gz
cd bloomery-*/

# A model that fits one 24 GB card, downloaded on first start:
bin/bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M --port 8080

# A file you already have:
bin/bloomery-serve -m /models/Qwen3.6-35B-A3B-Q4_K_M.gguf --port 8080

curl -s http://127.0.0.1:8080/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}'
```

Clef-Flash, with the head file fetched from Cloudflare's release (the repo bartowski's model card names as the model
it quantizes):

```sh
bin/bloomery-serve --hf bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M --port 8091
```

A local file takes its head by name: `bin/bloomery-serve -m <file.gguf> --head <dir>/joint_head.safetensors`, the
config `joint_head_config.json` beside it.

Downloads go to `~/.cache/bloomery/hf` (`BLOOMERY_CACHE` overrides it). A gated repo reads `HF_TOKEN` or
`~/.cache/huggingface/token`.

## Cards

The ds41, glm and qwen3 seats take `--parallel N` (two by default): requests alternate in 64-token turns on one model, and a lone request pays nothing for the second slot. A plan that cannot fit the card's free bytes is refused by name with its terms and the processes holding the card.

`--place` picks the cards for the models with CPU experts: `a` runs the model on the visible card with the most memory,
`bp` adds the next card as an expert tier, and `0+1` names cards by CUDA index (`CUDA_DEVICE_ORDER=PCI_BUS_ID` makes it `nvidia-smi`'s numbering). The single-card models run on
device 0; set `CUDA_VISIBLE_DEVICES` to pick another.

## Licence

MIT (`LICENSE`). Third-party code and the patched cuda-oxide toolchain: `THIRD_PARTY_NOTICES.md`.
