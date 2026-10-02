# bloomery — prebuilt Linux release

bloomery is an LLM inference engine for one workstation: a Rust host, CUDA kernels written in Rust, and AVX2 kernels
for the experts that stay on the CPU. Source, measurements and the full README: https://github.com/midagedev/bloomery

## What is in this archive

| File | What it serves |
|---|---|
| `bin/bloomery-serve` | llama-server's HTTP API (`/v1/chat/completions`, `/completion`, streaming): DeepSeek-V4.1-Flash, GLM-5.3-Flash, Qwen3.8-Flash-Next, Qwen3.6-35B-A3B, Qwen3-30B-A3B |
| `bin/bloomery_serve_clef` | Cloudflare Clef-Flash's SystemOne API (`POST /v1/systemone`) |

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
  - Qwen3-30B-A3B and Qwen3.6-35B-A3B at `Q4_K_M`: a 24 GB card; the whole model sits on device 0.
  - V4.1, GLM-5.3, Qwen3.8 (their experts partly on the CPU): a 24–48 GB card and a host with 256 GB of RAM.

## Run

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

Clef-Flash, with the head file fetched from Cloudflare's release:

```sh
bin/bloomery_serve_clef --hf bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M --port 8091
```

Downloads go to `~/.cache/bloomery/hf` (`BLOOMERY_CACHE` overrides it). A gated repo reads `HF_TOKEN` or
`~/.cache/huggingface/token`.

## Cards

`--place` picks the cards for the models with CPU experts: `a` runs the model on the visible card with the most memory,
`bp` adds the next card as an expert tier, and `0+1` names cards by CUDA index (`CUDA_DEVICE_ORDER=PCI_BUS_ID` makes it `nvidia-smi`'s numbering). The single-card models run on
device 0; set `CUDA_VISIBLE_DEVICES` to pick another.

## Licence

MIT (`LICENSE`). Third-party code and the patched cuda-oxide toolchain: `THIRD_PARTY_NOTICES.md`.
