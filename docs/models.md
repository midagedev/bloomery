# Models and hardware

## Models

| Model | Recommended file | How it runs |
|---|---|---|
| DeepSeek-V4.1-Flash | [`Q3_K_M`](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF) (vcruz305) | GPU + CPU experts; 256 GB of RAM |
| GLM-5.3-Flash | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) (unsloth) | GPU + CPU experts; sparse attention past 2,051 positions |
| Qwen3.8-Flash-Next | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (unsloth) | GPU + CPU experts |
| Qwen3.6-35B-A3B | [`Q4_K_M`](https://huggingface.co/lmstudio-community/Qwen3.6-35B-A3B-GGUF) (lmstudio-community) | whole on one 24 GB card, or split onto the CPU on 12-16 GB (automatic) |
| Qwen3-30B-A3B-Instruct-2507 | [`Q4_K_M`](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) (unsloth) | whole on one 24 GB card, or split onto the CPU on 12-16 GB (automatic) |
| Clef-Flash | [`Q8_0` to `Q3_K_S`](https://huggingface.co/bartowski/Cloudflare_clef-flash-GGUF) (bartowski), plus the release's [head file](BUILD.md#clef-flash) | whole on one GPU, its head on the CPU |
| DeepSeek-V2-Lite-Chat | [`Q3_K_M`](https://huggingface.co/mradermacher/DeepSeek-V2-Lite-Chat-GGUF) (mradermacher) | CPU or GPU; the first model, still gated |

Each file is the public upload as downloaded; the downloads are checked against the uploads' sha256. How close
each Clef quantization answers to the BF16 release, per question: [agreement table](https://huggingface.co/midagedev/clef-flash-bloomery).
The DSpark draft for V4.1 is converted from DeepSeek's checkpoint (the converter in llama.cpp's V4.1 pull
request).

## Hardware

**Required of every host**: an NVIDIA GPU of compute capability 8.6 or newer and driver R580 or newer (CUDA 13); a CPU with AVX2
and FMA (the prebuilt archive targets x86-64-v3); glibc 2.34+; Linux, or Windows through WSL2. NVMe is
recommended — a load streams the file to the card, and V4.1 reads its engram table from disk as it runs.

### What each model needs

| Model | GPU | Host RAM | Disk (the file) |
|---|---|---|---|
| Clef-Flash | 12 GB covers every quantization (the files run 4.26–9.55 GB) | any | 4.3–9.6 GB, plus the head (243 MB, fetched) |
| Qwen3.6-35B `Q4_K_M` | 24 GB whole, or 12–16 GB split (automatic) | 32 GB whole; about 10 GB free beside a small card | 21.2 GB |
| Qwen3-30B `Q4_K_M` | 24 GB whole, or 12–16 GB split (automatic) | 32 GB whole; about 8 GB free beside a small card | 18.6 GB |
| Qwen3.8-Flash-Next | 24 GB recommended (the expert share sizes to the card's free bytes); a 96 GB card holds every routed expert at a 4k context, and a second card adds none | 128 GB beside a 24 or 48 GB card, 64 GB beside a 96 GB card (about 101, 75 and 34 GB free, plus adaptive residency's pool where the host has it) | 111.3 GB, plus the 2.8 GB MTP draft |
| GLM-5.3-Flash | 24 GB recommended | 256 GB | 199.7 GB |
| DeepSeek-V4.1-Flash | 24 GB recommended | 256 GB | 347.3 GB (a source build can add a 155.7 GB r8 sidecar; the release does not use one) |

Qwen3.8's 28.8 GB PLE table is read from host RAM whatever the cards, so no number of cards holds the whole
file. `UD-Q3_K_XL` (90.0 GB, not yet run) keeps every routed expert on the host whatever the card, since no card
kernel reads its IQ3_XXS and IQ4_NL stacks: about 89 GB free. The MTP draft reads a Q8_0 head and this file's is
Q6_K, so the draft is off on it by name (a set `BLOOMERY_DRAFT=mtp` is refused at the load).

**Recommended** (the configuration every number in the README ran on): a 32-core AVX2 CPU with 8 DDR4
channels, 256 GB of RAM, an RTX A6000 48 GB, and an RTX 3090 24 GB beside it for `--place bp`. Cards are found
by device, not by name; a list word such as `0+1` names cards by CUDA index (`CUDA_DEVICE_ORDER=PCI_BUS_ID`
matches `nvidia-smi`'s numbering; the qwen38 seat takes `a`, `gate` or `bp` only).

Only the A6000, the 3090, and an RTX 3060 12 GB under WSL2 (running Clef-Flash, its download and its cache)
have been run; other cards and drivers are untested — please open an issue with `--version` and the error
text. Costs per part: [`HARDWARE.md`](HARDWARE.md).
