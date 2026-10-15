# Models and hardware

## Models

| Model | Recommended file | How it runs |
|---|---|---|
| DeepSeek-V4.1-Flash | [`Q3_K_M`](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF) (vcruz305) | GPU + CPU experts; 256 GB of RAM, or less with the NVMe expert tier |
| GLM-5.3-Flash | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) (unsloth) | GPU + CPU experts (the NVMe expert tier where RAM falls short); sparse attention past 2,051 positions |
| Qwen3.8-Flash-Next | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (unsloth) | GPU + CPU experts (the NVMe expert tier where RAM falls short) |
| MiMo-V2.6-Flash | [`MXFP4`](https://huggingface.co/ggml-org/MiMo-V2.6-Flash-MOPD-GGUF) (ggml-org) | GPU + CPU experts; the hot routed experts sit on the card |
| Qwen3.6-35B-A3B | [`Q4_K_M`](https://huggingface.co/lmstudio-community/Qwen3.6-35B-A3B-GGUF) (lmstudio-community) | whole on one 24 GB card, or split onto the CPU on 12-16 GB (automatic); image input with `--mmproj` |
| Qwen3-30B-A3B-Instruct-2507 | [`Q4_K_M`](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) (unsloth) | whole on one 24 GB card, or split onto the CPU on 12-16 GB (automatic) |
| MiniCPM-V 4.7 35B-A3B | [`Q4_K_M`](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf) (openbmb) | on the `qwen3` seat (a `qwen35moe` file); image input with `--mmproj` since 0.3.3, a tile capped at 70 tokens |
| Clef-Flash | [`Q8_0` to `Q3_K_S`](https://huggingface.co/bartowski/Cloudflare_clef-flash-GGUF) (bartowski's upload of 2026-10-05; llama.cpp's `clef` layout carries the head, and the Q3_K files' Q3_K token embedding runs on the card; an older `qwen35`-layout file takes the release's [head file](BUILD.md#clef-flash)) | whole on one GPU, its head on the CPU; image input with an encoder |
| lev | [`Q4_K_M`](https://huggingface.co/ggml-org/lev-GGUF) (ggml-org) | whole on one GPU; its label readout in the file |
| DeepSeek-V2-Lite-Chat | [`Q3_K_M`](https://huggingface.co/mradermacher/DeepSeek-V2-Lite-Chat-GGUF) (mradermacher) | CPU or GPU; the first model, still gated |

Each file is the public upload as downloaded; the downloads are checked against the uploads' sha256. How close
each Clef quantization answers to the BF16 release, per question: [agreement table](https://huggingface.co/midagedev/clef-flash-bloomery).
The DSpark draft for V4.1 is converted from DeepSeek's checkpoint (the converter in llama.cpp's V4.1 pull
request).

## Hardware

**Required of every host**: an NVIDIA GPU of compute capability 8.6 or newer and driver R580 or newer (CUDA 13); a CPU with AVX2
and FMA (the prebuilt archive targets x86-64-v3; an aarch64 host builds from source, below); glibc 2.34+; Linux, or Windows through WSL2. NVMe is
recommended — a load streams the file to the card, and V4.1 reads its engram table from disk as it runs.

### What each model needs

| Model | GPU | Host RAM | Disk (the file) |
|---|---|---|---|
| Clef-Flash | 12 GB covers every quantization (the files run 4.80–9.68 GB) | any | 4.8–9.7 GB; a `qwen35`-layout file adds the release's head (243 MB, fetched) |
| lev `Q4_K_M` | 6.12 GB at its default 16,384 context (the server's fit line) | any | 3.0 GB |
| Qwen3.6-35B `Q4_K_M` | 24 GB whole, or 12–16 GB split (automatic) | 32 GB whole; about 10 GB free beside a small card | 21.2 GB |
| Qwen3-30B `Q4_K_M` | 24 GB whole, or 12–16 GB split (automatic) | 32 GB whole; about 8 GB free beside a small card | 18.6 GB |
| Qwen3.8-Flash-Next | 24 GB recommended (the expert share sizes to the card's free bytes); a 96 GB card holds every routed expert at a 4k context, and a second card adds none | 128 GB beside a 24 or 48 GB card, 64 GB beside a 96 GB card (about 101, 75 and 34 GB free, plus adaptive residency's pool where the host has it) | 111.3 GB, plus the 2.8 GB MTP draft |
| GLM-5.3-Flash | 24 GB recommended | 256 GB to hold every routed expert; less pages the rest from NVMe | 199.7 GB |
| MiMo-V2.6-Flash | the dense layers, attention and KV cache, plus the routed experts the card has room for (the plan sizes the share to the card) | the routed experts the card does not hold; with none on the card, about 161 GB (the 0.2.10 figure) | the file on the repo page |
| DeepSeek-V4.1-Flash | 24 GB recommended | 256 GB to hold every routed expert; less pages the rest from NVMe | 347.3 GB (a source build can add a 155.7 GB r8 sidecar; the release does not use one) |

**The NVMe expert tier** (Qwen3.8, GLM-5.3, V4.1): when the host RAM cannot hold the experts the plan leaves on the
host, the plan pages the overflow through a RAM arena filled from the model file on NVMe, and the tokens equal the
all-RAM load's. The plan line and the
`nvtier` record name the arena and the paged bytes. The limits of a paged load are in [Serving](serving.md#the-seats).

Qwen3.8's 28.8 GB PLE table is read from host RAM whatever the cards, so no number of cards holds the whole
file. Qwen3.8's `UD-Q3_K_XL` (90.0 GB) runs: its IQ3_XXS, IQ4_XS and IQ4_NL expert stacks are on the card, and the
README's status rows name the types still not loaded.

**Quant types.** Card experts run in Q4_K–Q6_K, MXFP4, IQ2_XS, IQ3_XXS, IQ4_XS and IQ4_NL. The host kernels add
Q2_K, IQ2_XS and Q6_K, so `UD-Q2_K_XL` files load. GLM-5.3's token embedding may be q4_K, q5_K or q6_K. GLM-5.3
`UD-IQ4_XS` keeps every routed expert on the host, because its IQ3_S gate/up have no card kernel.

**Recommended** (the configuration every number in the README ran on): a 32-core AVX2 CPU with 8 DDR4
channels, 256 GB of RAM, an RTX A6000 48 GB, and an RTX 3090 24 GB beside it for `--place bp`. Cards are found
by device, not by name; a list word such as `0+1` names cards by CUDA index (`CUDA_DEVICE_ORDER=PCI_BUS_ID`
matches `nvidia-smi`'s numbering; the qwen38 seat takes `a`, `gate` or `bp` only).

A DGX Spark (GB10, aarch64, sm_121) builds and runs from source, a contributor's port
([`contrib/dgx-spark.md`](contrib/dgx-spark.md)): the plan treats its unified memory as one pool, and `BLOOMERY_ARCH`
sets the CUDA arch. The release binaries are x86-64 only.

Only the A6000, the 3090, an RTX 3060 12 GB under WSL2 (running Clef-Flash, its download and its cache) and the
Spark have been run; other cards and drivers are untested — please open an issue with `--version` and the error
text. Costs per part: [`HARDWARE.md`](HARDWARE.md).
