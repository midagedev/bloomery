# Models and hardware

## Models

| Model | Recommended file | How it runs |
|---|---|---|
| DeepSeek-V4.1-Flash | [`Q3_K_M`](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF) (vcruz305) | GPU + CPU experts; 256 GB of RAM, or less with the NVMe expert tier |
| GLM-5.3-Flash | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) (unsloth) | GPU + CPU experts (the NVMe expert tier where RAM falls short; on an integrated GPU the card computes the tier's experts at decode); sparse attention past 2,051 positions |
| Qwen3.8-Flash-Next | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (unsloth) | GPU + CPU experts (the NVMe expert tier where RAM falls short) |
| MiMo-V2.6-Flash | [`MXFP4`](https://huggingface.co/ggml-org/MiMo-V2.6-Flash-MOPD-GGUF) (ggml-org); the RL release's [`MXFP4`](https://huggingface.co/ggml-org/MiMo-V2.6-Flash-RL-GGUF) runs the same way | GPU + CPU experts; the hot routed experts sit on the card |
| Qwen3.6-35B-A3B | [`Q4_K_M`](https://huggingface.co/lmstudio-community/Qwen3.6-35B-A3B-GGUF) (lmstudio-community) | whole on one 24 GB card, or split onto the CPU on 12-16 GB (automatic); image input with `--mmproj` |
| Ornith-1.5-35B-A3B | [`Q4_K_M`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) (ornith-ai) | on the `qwen3` seat (a `qwen35moe` file with Qwen3.6-35B-A3B's shapes; the next-token layer it carries is not read); whole on one 24 GB card, or split onto the CPU on 12-16 GB (automatic); text only, its `mmproj` has not been run |
| Sage-A1.5 | [`Q4_K_M`](https://huggingface.co/InternScience/Sage-A1.5-Q4_K_M-GGUF) (InternScience) | on the `qwen3` seat (a `qwen35moe` fine-tune of Qwen3.6-35B-A3B with its shapes; the next-token layer it carries is not read); whole on one 24 GB card, or split onto the CPU on 12-16 GB (automatic); image input with its `mmproj` |
| Qwen3-30B-A3B-Instruct-2507 | [`Q4_K_M`](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) (unsloth) | whole on one 24 GB card, or split onto the CPU on 12-16 GB (automatic) |
| MiniCPM-V 4.7 35B-A3B | [`Q4_K_M`](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf) (openbmb) | on the `qwen3` seat (a `qwen35moe` file); image input with `--mmproj` since 0.3.3, a tile capped at 70 tokens |
| Clef-Flash | [`Q8_0` to `Q3_K_S`](https://huggingface.co/bartowski/Cloudflare_clef-flash-GGUF) (bartowski's upload of 2026-10-05; llama.cpp's `clef` layout carries the head, and the Q3_K files' Q3_K token embedding runs on the card; an older `qwen35`-layout file takes the release's [head file](BUILD.md#clef-flash)) | whole on one GPU, its head on the CPU; image input with an encoder |
| Nemotron 3.5 Bolt 74B-A9B | `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf` (by `-m`) | whole on one 24 GB GPU, one slot (`nemotron` seat); a prompt runs in batched passes |
| Nemotron-3.5-Lightning 30B-A3B | `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_XL.gguf` (by `-m`) | on the `nemotron` seat, one slot. No latent: the experts act on the hidden (2,688), and the router keeps the top 6 of 128. The shared expert is Q8_0. The next-token block is carried and not loaded. Placed load: the card runs its routed Q5_0, Q5_1 and Q8_0 stacks (Q5_0 ups with Q5_1 and Q8_0 downs, the first layer's up Q5_1), so the plan holds as many experts on the card as its room allows and the rest on the host. Served on a 24 GB card and on an A6000. RTX A6000 at the serving defaults before the card ran these types (`--ctx-size 16384`, 2,944 experts on the host): 89.2 tok/s after a 25-token prompt, 79.9 tok/s after a 5,866-token prompt (0.3.8's release check) |
| Nemotron 3 Nano 4B | `Q4_K_M` (`NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf`, by `-m`) | whole on one GPU, one slot (`--model nemotron`); a prompt runs in batched passes |
| MiniMax-M3.1-Flash | the vendor's NVFP4 safetensors folder (48 shards, by `-m <folder>`), no conversion | GPU + CPU experts in their own NVFP4 form (`minimaxm3` seat; `--parallel N` serves N slots). On two cards the unset placement picks `bp` (the stage on the larger card, the other card an expert tier) |
| Muse Spark Open 816B-A42B | `UD-IQ2_XXS` (`Muse-Spark-Open-816B-A42B-UD-IQ2_XXS-00001-of-00006.gguf`, by `-m`) | GPU + CPU experts; one slot (`musespark` seat). On two cards the unset placement picks `bp`. Text only |
| MiniMax-M3 | [`UD-IQ4_XS`](https://huggingface.co/unsloth/MiniMax-M3-GGUF) (unsloth) | GPU + CPU experts (the plan sizes the share to the card); one slot (`minimaxm3` seat). Unverified: no gate runs on this file; MiniMax-M3.1-Flash is the MiniMax the gates hold |
| Raon-1-mini-preview | `raon-1-mini-preview-Q4_K_M.gguf` (by `-m`) | whole on one card, one slot (`raon1` seat); its DFlash2 draft (`dflash-raon-1-mini-preview-Q8_0.gguf`) beside it |
| Solar-Open2-250B | [`Q4_K_M`](https://huggingface.co/llamaraspberryrabbit/Solar-Open2-250B-GGUF) (llamaraspberryrabbit) | GPU + CPU experts; one slot (`solar2` seat); on two cards under `--place bp` (the stage card and an expert tier) |
| lev | [`Q4_K_M`](https://huggingface.co/ggml-org/lev-GGUF) (ggml-org) | whole on one GPU; its label readout in the file |
| DeepSeek-V2-Lite-Chat | [`Q3_K_M`](https://huggingface.co/mradermacher/DeepSeek-V2-Lite-Chat-GGUF) (mradermacher) | CPU or GPU; the first model, still gated |

A model starts by name (`bloomery-serve --model NAME`, [`agents.md`](agents.md)) from a row of `tools/catalog.tsv`.
`tools/check-catalog-docs.sh` holds the Recommended file of each row above that the catalog names, and the README's Use
table, to those rows.

Each file is the public upload as downloaded; the downloads are checked against the uploads' sha256. How close
each Clef quantization answers to the BF16 release, per question: [agreement table](https://huggingface.co/midagedev/clef-flash-bloomery).
The DSpark draft for V4.1 is converted from DeepSeek's checkpoint (the converter in llama.cpp's V4.1 pull
request). Qwen3.6-35B-A3B and MiniCPM-V 4.7 keep their HF DSpark draft directory beside the GGUF
(`BLOOMERY_DSPARK_MODEL`): RedHatAI's Qwen3.6 speculator for the former, and MiniCPM-V 4.7's own draft, whose
full vocabulary takes the target's head and embedding rows, for the latter.

## Hardware

What each platform needs (the GPU, its driver, the CPU, glibc, the OS) is in the README's [Install](../README.md#install)
section. NVMe is recommended: a load streams the file to the card, and V4.1 reads its engram table from disk as it runs.

### What each model needs

| Model | GPU | Host RAM | Disk (the file) |
|---|---|---|---|
| Clef-Flash | 12 GB covers every quantization (the files run 4.80–9.68 GB) | any | 4.8–9.7 GB; a `qwen35`-layout file adds the release's head (243 MB, fetched) |
| lev `Q4_K_M` | 6.12 GB at its default 16,384 context (the server's fit line) | any | 3.0 GB |
| Qwen3.6-35B `Q4_K_M` | 24 GB whole, or 12–16 GB split (automatic) | 32 GB whole; about 10 GB free beside a small card | 21.2 GB |
| Ornith-1.5-35B `Q4_K_M` | 24 GB whole, or 12–16 GB split (automatic) | 32 GB whole; about 10 GB free beside a small card | 21.7 GB |
| Sage-A1.5 `Q4_K_M` | 24 GB whole, or 12–16 GB split (automatic) | 32 GB whole; about 10 GB free beside a small card | 21.7 GB, plus the 0.90 GB `mmproj` |
| Qwen3-30B `Q4_K_M` | 24 GB whole, or 12–16 GB split (automatic) | 32 GB whole; about 8 GB free beside a small card | 18.6 GB |
| Qwen3.8-Flash-Next | 24 GB recommended (the expert share sizes to the card's free bytes); a 96 GB card holds every routed expert at a 4k context, and a second card adds none | 128 GB beside a 24 or 48 GB card, 64 GB beside a 96 GB card (about 101, 75 and 34 GB free, plus adaptive residency's pool where the host has it) | 111.3 GB, plus the 2.8 GB MTP draft |
| GLM-5.3-Flash | 24 GB recommended | 256 GB to hold every routed expert; less pages the rest from NVMe | 199.7 GB |
| MiMo-V2.6-Flash | the dense layers, attention and KV cache, plus the routed experts the card has room for (the plan sizes the share to the card) | the routed experts the card does not hold; with none on the card, about 161 GB (the 0.2.10 figure) | the file on the repo page |
| DeepSeek-V4.1-Flash | 24 GB recommended | 256 GB to hold every routed expert; less pages the rest from NVMe | 347.3 GB (a source build can add a 155.7 GB r8 sidecar; the release does not use one) |
| MiniMax-M3.1-Flash (NVFP4 folder) | RTX A6000 + RTX 3090 at `bp`, ctx 16,384: 1,063 routed experts on the A6000 and 726 on the 3090 tier, beside the dense layers, attention and KV cache | the routed experts neither card holds (5,507 on that plan) | 250.1 GB (48 shards) |
| Muse Spark Open 816B-A42B `UD-IQ2_XXS` | RTX A6000 + RTX 3090 at `bp`, ctx 16,384: 1,900 routed experts on the A6000 plus the 3090 tier, beside the dense layers, attention and KV cache | the routed experts neither card holds (12,209 on that plan) | 210.1 GiB (six shards) |
| MiniMax-M3 `UD-IQ4_XS` | the dense layers, the attention and the KV cache, plus the routed experts the card has room for (662 of them on the RTX A6000 plan at ctx 131,072) | the routed experts the card does not hold (6,634 of them on that plan) | 208.5 GB (six shards) |
| Raon-1-mini-preview `Q4_K_M` | whole on one card (measured on a 24 GB RTX 3090) | any | 19.7 GB (one file) |
| Solar-Open2-250B `Q4_K_M` | one card, plus an expert tier card under `--place bp` | the routed experts the card does not hold | 151.8 GB (four shards) |

Ornith-1.5-35B's `Q4_K_M` plans the same weight bytes as Qwen3.6-35B's (21,311,258,624 B at the allocator's granules, `--plan` on both files), so its GPU and host RAM cells are Qwen3.6's; the 0.55 GB its file holds beyond is the next-token layer, which the load does not read.
Sage-A1.5's `Q4_K_M` plans the same weight bytes as Qwen3.6-35B's too (21,311,258,624 B, `--plan` on both files). It passes the `qwen3` serve gate, and it answers with thinking on and off, calls tools on `/v1/chat/completions` and reads an image through its `mmproj`. `BLOOMERY_DRAFT=mtp` drafts its next-token layer on a whole-card load of one sequence; a media load, a placed load and `--parallel` above 1 refuse it by name.

**The NVMe expert tier** (Qwen3.8, GLM-5.3, V4.1): when the host RAM cannot hold the experts the plan leaves on the
host, the plan pages the overflow through a RAM arena filled from the model file on NVMe. The tokens equal the
all-RAM load's where the host computes the arena's experts; where GLM-5.3's card computes them at decode, they can
part from it ([Serving](serving.md)). The plan line and the
`nvtier` record name the arena and the paged bytes. The limits of a paged load are in [Serving](serving.md#the-seats).

Qwen3.8's 28.8 GB PLE table is read from host RAM whatever the cards, so no number of cards holds the whole
file. Qwen3.8's `UD-Q3_K_XL` (90.0 GB) runs: its IQ3_XXS, IQ4_XS and IQ4_NL expert stacks are on the card, and the
README's status rows name the types still not loaded.

**Quant types.** A card runs a routed expert stack of Q4_K, Q5_K, Q6_K, Q8_0, MXFP4, NVFP4, IQ1_S, IQ1_M, IQ2_XXS,
IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS (a gate·up and a down), and the down of a Q5_0, Q5_1 or IQ4_NL stack. The
types with no card kernel are Q2_K, Q2_0, Q4_0, Q4_1, Q8_1, Q8_K, Q1_0, TQ1_0 and TQ2_0; of those only Q2_K and Q2_0
have a host matmul, so a `UD-Q2_K_XL` file loads. GLM-5.3's token embedding may be q4_K, q5_K or q6_K. On GLM-5.3 a
layer whose down is Q6_K runs its experts on the card beside a card gate and up.

**Recommended** (the configuration the maintainers' numbers in the README ran on): a 32-core AVX2 CPU with 8 DDR4
channels, 256 GB of RAM, an RTX A6000 48 GB, and an RTX 3090 24 GB beside it for `--place bp`. Cards are found
by device, not by name; a list word such as `0+1` names cards by CUDA index (`CUDA_DEVICE_ORDER=PCI_BUS_ID`
matches `nvidia-smi`'s numbering; the qwen38 seat takes `a`, `gate` or `bp` only).

A DGX Spark (GB10, aarch64, sm_121) is a contributor's port ([`contrib/dgx-spark.md`](contrib/dgx-spark.md)): the
plan treats its unified memory as one pool, and the recipes build `sm_121a` on it unless `BLOOMERY_ARCH` names another
arch. It installs the release's GB10 archive where the release carries one, or builds from source (the README's
[Install](../README.md#dgx-spark-and-other-gb10-machines)).

The machines that have run it are in the README's [Hardware](../README.md#hardware) section; an RTX 3060 12 GB under
WSL2 ran Clef-Flash, its download and its cache. Other cards and drivers are untested: please open an issue with
`--version` and the error text. Costs per part: [`HARDWARE.md`](HARDWARE.md).
