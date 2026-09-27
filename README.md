<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img src="docs/assets/logo.svg" width="128" alt="bloomery">
  </picture>
</p>

# bloomery

Hybrid GPU + CPU inference for mixture-of-experts models, written in Rust down to the CUDA kernels.

bloomery runs mixture-of-experts models that do not fit on one GPU. Each routed layer keeps as many of its experts on the card as fit, and the rest run on the CPU inside the same decode step. The host code is Rust, the GPU kernels are CUDA written in Rust ([cuda-oxide](https://github.com/NVlabs/cuda-oxide)), and the CPU expert kernels are AVX2 Rust. Today it runs DeepSeek-V4.1-Flash, GLM-5.3-Flash and Qwen3.8-Flash-Next on one GPU with an eight-channel CPU and 256 GB of RAM (see [Target hardware](#target-hardware)), and Qwen3.6-35B-A3B and Qwen3-30B-A3B whole on one GPU.

## Models

The files below are the ones the gates and the numbers use; ~~each is byte-identical (sha256) to the upload linked~~ each was downloaded from the upload linked. For the three added on 2026-09-28 the files on the timing machine were compared with the uploads' listings that day: every shard's byte size matches, and so do the sha256 of GLM-5.3's and Qwen3.8's first shards and of the whole Qwen3.6 file. For the others a sha256 comparison has not been recorded yet.

| Model | File | Runs as | Checked by |
|---|---|---|---|
| [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | [`Q3_K_M`, 9 shards](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF) (vcruz305) | one GPU + CPU experts: `generate_ds41`, `bloomery-chat`, `bloomery-serve-ds41` | per-op reference gates; step, skewed-pass, rollback and prompt = steps bit gates; load, draft, chat and server gates |
| [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | [`Q4_K_M`](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) (unsloth) | the whole model on one GPU: `generate_qwen3moe` | per-kernel gates and an end-to-end gate (greedy tokens, PPL/KLD against the reference) |
| [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | [`UD-Q4_K_XL`, 6 shards](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) (unsloth) | one GPU + CPU experts: `generate_glm5next`, context up to 2,051 positions | an end-to-end gate against ik_llama.cpp's batch sets (every layer's streams, node count, graph = eager), the card-expert, router and MLA gates |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | [`UD-Q4_K_XL`, 4 shards](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (unsloth) | one GPU + every routed expert on the CPU: `generate_qwen3moe` | an end-to-end gate against ik_llama.cpp's batch sets (every layer's streams, graph = eager = one pass, prompt by passes = by steps) |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | [`Q4_K_M`](https://huggingface.co/lmstudio-community/Qwen3.6-35B-A3B-GGUF) (lmstudio-community) | the whole model on one GPU: `generate_qwen3moe` | an end-to-end gate (node counts, decode steps = one pass = eager, per-layer taps against ik_llama.cpp's sets), the MoE, attention and linear-attention kernel gates |
| [DeepSeek-V2-Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat) | [`Q3_K_M`](https://huggingface.co/mradermacher/DeepSeek-V2-Lite-Chat-GGUF) (mradermacher) | CPU (`bloomery-decode`) and GPU | the engine's first model; every subsystem gate |

The DSpark draft for V4.1 speculative decoding is not a public upload. It is converted from the draft tensors in DeepSeek's V4.1 checkpoint with the converter in llama.cpp's V4.1 pull request, and then its `dflash.target_layers` is rewritten from [38, 39, 40] to [37, 38, 39]: the converter keeps V4's value, which is one layer off for V4.1. It is not the file in [JigSawPT's DSpark upload](https://huggingface.co/JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF).

In progress: [DeepSeek-V4-Flash-0731](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF).

## What it does

V4.1-Flash does not fit on one consumer card, so each decode step is split across the machine:

- **Experts are split between the card and the host.** Each routed layer keeps some of its experts on the GPU. The rest run on an AVX2 host tier inside the same captured CUDA graph step. A router-frequency list (`BLOOMERY_HOT_LIST`) can pick which experts stay on the card.
- **engram rows come from NVMe.** V4.1's n-gram lookup table does not fit in RAM beside the host experts, so it stays in a memory-mapped file. Each token reads 48 rows; a helper thread asks the kernel for all of them first, then copies them.
- **A skewed two-row pass.** Two positions run one layer apart in one step, so the card's dense work hides behind the host leg. It is the verification pass for speculative decoding.
- **Speculative decoding with the DSpark draft** (`BLOOMERY_DRAFT=dspark`). The draft runs on a second card and proposes one token; the skewed pass checks it. Greedy output is the same with and without the draft; a gate checks that token for token.
- **An n-gram lookup draft** (`BLOOMERY_DRAFT=lookup`) proposes the next token from the text so far, with the same guarantee.

A prompt does not run one decode step per token:

- **It runs in batches of up to 512 positions**, and leaves the model bit for bit in the state that one decode step per token would leave. Each layer runs only at the positions a later reader needs.
- **Each expert's work is batched.** A host expert computes all the batch's tokens routed to it in one pass. A card expert reads its weights once for every tile of up to eight of its tokens.
- **Two batches go through the layers together** (`BLOOMERY_PREFILL_GROUP`, default 2), so the card routes the next batch while the CPU runs this one's host experts.

## Status

**It runs on the public `Q3_K_M` GGUF as uploaded**, and the maintainers' gates run on that file by default ([`docs/BUILD.md`](docs/BUILD.md#the-model-file)). You can run `generate_ds41` (token ids in, greedy ids out), `bloomery-chat` (text in, streamed text out, sampled or greedy) and `bloomery-serve-ds41` (llama-server's HTTP API with streaming, prompt-prefix reuse, reasoning and tool calls; one request at a time). The tokenizer is bit-identical to `llama-tokenize` on its gate's corpora.

The reference sets the gates read come from the public file: ik_llama.cpp's greedy tokens, the PPL/KLD baseline and the DSpark draft's intermediate dumps were made again from it on 2026-09-27 ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-27.md#sit10)). A gate refuses by name a set made from another file.

In progress: faster V4.1 prompts on the CPU side; batched prompts for GLM-5.3 and Qwen3.8; GLM-5.3's sparse-attention selector, for context past 2,051 positions; DeepSeek-V4-Flash-0731.

As far as we know (2026-09-24), this is the only Rust engine that runs DeepSeek-V4.1-Flash with CPU expert offloading, with the GPU kernels also written in Rust. If you know of another one, please open an issue.

## Measured numbers

Built for Ampere GPUs (sm_86) and AVX2 CPUs. Every number is a single stream on one RTX A6000 (48 GB, 300 W), or where a row says so on the workstation's RTX 3090 (24 GB, 250 W), in a 32-core AVX2 workstation, instrumentation off unless a row says otherwise, ~~one fresh process per arm~~ one process per reference arm and, for our arms, one process per load setting with the engine reset between arms (bit for bit a fresh process's state), measured under the quiet-machine protocol below. Decode rows generate `n = 96` tokens. Arms are alternated within a window, a table names its windows, and a ratio is only taken between arms of the same window.

The cross-engine tables for DeepSeek-V4.1-Flash, Qwen3.6, Qwen3.8 and GLM-5.3 come from one release sitting on 2026-09-28 (main at `53e2def`; rig-log [release-sit](https://github.com/midagedev/rig-log/blob/main/log/2026-09-28.md#release-sit)). In that sitting llama.cpp ran through `llama-bench` (GLM-5.3's MTP arm through `llama-server`), which feeds new random token ids on every repetition, so several of llama.cpp's rows below read their weights or n-gram rows cold; each such row says so. A warm re-measure of V4.1, GLM-5.3 and Qwen3.8, with llama.cpp through `llama-server` on the same token ids as ours and a warm-up first, is scheduled after the batched-prompt work for GLM-5.3 and Qwen3.8 lands.

### DeepSeek-V4.1-Flash, public `Q3_K_M` — one GPU plus CPU experts

Placement plan (a): every layer and the head on the card, 2,668 routed experts on the card, the other 12,692 on the host.

**Against llama.cpp** (release sitting, 2026-09-28; synthetic prompt ids, no hot list; llama.cpp's V4.1 pull request [#28696](https://github.com/ggml-org/llama.cpp/pull/28696) at `5210c7c`, which carries three commits by this repository's author, `-ngl 999 --n-cpu-moe 33 -fa on -t 32 -nopo 1`; rig-log [v41-release](https://github.com/midagedev/rig-log/blob/main/log/2026-09-28.md#v41-release)):

| A6000, tok/s | bloomery | llama.cpp | bloomery / llama.cpp |
|---|---:|---:|---:|
| decode, depth 6 | **29.66** | 22.75 | 1.304 |
| decode, depth 4096 | **30.04** | 21.78 | 1.379 |
| prompt, P = 512 | **190.6** | 104.9 | 1.817 |
| prompt, P = 4096 | **358.4** | 76.5 | 4.68 |

- **llama.cpp's column is its most favorable row.** Its decode values are the one round whose timed window had no page faults; its other five decode rounds, at either placement, read 20.65 at depth 6 and 19.86 at 4096, with faults bounding up to 7.8 % of the window. Its P = 512 value is the re-sit's clean round.
- **At P = 4096 llama.cpp faults in every repetition**, bounding up to 27.9 % of the window, the second repetition inside one process included: `llama-bench` feeds new ids each time, and this branch reads each n-gram (engram) row it has not seen through a page fault. 76.5 is its level on input it has not seen; bloomery prefetches those rows. The 103.8 it read on 2026-09-25 (below) came from repeated ids.
- **llama.cpp's own automatic placement** (`-fitt 1024`) was within 0.5 % of the hand-set one on every row. With `-ub 4096 -b 4096` the hand-set placement could not create its context (the ubatch buffer does not fit beside 33 layers of card experts), and the automatic one ran at 77.5, faulting as above.
- **Our rows carry the runner's `[cpu-busy]` tag**, which here counted our own process: a known defect of the runner's CPU check, being fixed. These rows will be taken again after the fix. Each ratio is our two-round mean over that llama.cpp value.

**Decode.** The router-ranked hot list picks the card experts. The prompt is the first 512 tokens of a prose or a code corpus, and the DSpark draft runs on the workstation's RTX 3090. Three rounds, rig-log [2026-09-25, dspark-loop-tps](https://github.com/midagedev/rig-log/blob/main/log/2026-09-25.md#dspark-loop-tps):

| Prompt | plain tok/s | DSpark tok/s | DSpark / plain |
|---|---:|---:|---:|
| prose 512 | 44.8 | **51.3** | 1.146 ± 0.014 |
| code 512 | 43.8 | **50.6** | 1.156 ± 0.062 |

**On one RTX 3090** (24 GB, capped at 250 W; placement `gate`: 1,146 routed experts on the card, the rest on the host; hot list), V4.1 decodes at **36.7 tok/s** after the prose 512 prompt and 36.5 after the code 512 prompt (plain, first of two rounds; the second round's rows carried page faults in their timed windows; 2026-09-27, rig-log [e21-3090](https://github.com/midagedev/rig-log/blob/main/log/2026-09-27.md#e21-3090)). The power cap was active for about a third of that window.

Against llama.cpp on the RTX 3090 (release sitting, 2026-09-28; synthetic prompt ids, hot list; llama.cpp #28696 with `--n-cpu-moe 37` and 38; two rounds; rig-log [e21ref-release](https://github.com/midagedev/rig-log/blob/main/log/2026-09-28.md#e21ref-release)):

| RTX 3090, tok/s | decode, depth 6 | decode, depth 4096 | prompt, P = 4096 |
|---|---:|---:|---:|
| bloomery (`gate`) | **28.14** | **28.92** | 329.3 |
| llama.cpp `--n-cpu-moe 37` | 21.59 | 20.85 | — |
| llama.cpp `--n-cpu-moe 38` | 21.49 | — | — |

bloomery decodes at 1.303× (depth 6) and 1.387× (depth 4096) llama.cpp's `--n-cpu-moe 37`. llama.cpp's rows here carry no tag; ours carry the same `[cpu-busy]` runner tag as the A6000 table and will be taken again with it.

On the A6000, the same placement (a) decodes at 39.6 tok/s after a 4096-token prose prompt (plain, with `BLOOMERY_STEP_STATS=1` card-event timing on; 2026-09-26, [cardtile-ab](https://github.com/midagedev/rig-log/blob/main/log/2026-09-26.md#cardtile-ab)). With the card budget of a 24 GB card, emulated on the A6000 (1,171 card experts), prose 512 decodes at 35.8 tok/s (plain; 2026-09-24, [public-q3km-prose-code-and-budget](https://github.com/midagedev/rig-log/blob/main/log/2026-09-24.md#public-q3km-prose-code-and-budget)).

~~Against llama.cpp's V4.1 pull request ([#28696](https://github.com/ggml-org/llama.cpp/pull/28696) at `5210c7c`, `--n-cpu-moe 33`), in one window with three rounds: on synthetic prompt ids at depth 6, with no hot list, bloomery decodes at 29.7 tok/s and llama.cpp at 21.5, 1.38 ± 0.19. llama.cpp's first round read 11 % below its other two, which is most of that interval (rig-log [2026-09-25, launch-thread-lever-ab](https://github.com/midagedev/rig-log/blob/main/log/2026-09-25.md#launch-thread-lever-ab)).~~ Superseded by the release table above.

**Prompt processing, earlier windows** (pp tok/s over a prompt of P tokens; every row ran with `BLOOMERY_STEP_STATS=1`, card-event timing on, so these rows do not share a table with the release table above):

| Prompt | Card experts by | P = 512 | P = 4096 | Measured |
|---|---|---:|---:|---|
| synthetic ids | id order (no hot list) | 144.6 | **247.5** | 2026-09-26; main at `0bcee2c` (512) and `e690f54` (4096) |
| prose corpus | hot list | **216.1** | 292.1 | 2026-09-26; main at `efc202f`, before the paired batches |

Sources: rig-log [b1-pp-ab](https://github.com/midagedev/rig-log/blob/main/log/2026-09-26.md#b1-pp-ab), [prefillgroup-ab](https://github.com/midagedev/rig-log/blob/main/log/2026-09-26.md#prefillgroup-ab) and [cardtile-ab](https://github.com/midagedev/rig-log/blob/main/log/2026-09-26.md#cardtile-ab). The 247.5 is the clean round of two. The other round read 235.2, because its row was the window's first run and its prompt planning read the file cold (0.7 s). ~~Later changes to the tree are not expected to move the 144.6 [derived: at 512 synthetic tokens the card work already runs under the CPU's].~~ The r8 sidecar (2026-09-27) changed the CPU side, which bounds P = 512, so the 144.6 is out of date; it has not been re-measured in a clean window. llama.cpp's V4.1 branch (`-ngl 999 --n-cpu-moe 33 -fa on -t 32 -nopo 1`, `llama-bench -p P -n 0`, which also feeds random ids) ran at 104.6 tok/s at P = 512 and 103.8 at P = 4096 on 2026-09-25 ([P = 512](https://github.com/midagedev/rig-log/blob/main/log/2026-09-25.md#v41-prefill-baseline-p512), [P = 4096](https://github.com/midagedev/rig-log/blob/main/log/2026-09-25.md#v41-prefill-baseline-p4096)). That was a different window from ours, so no ratio is given here. Those two values are warm ones: that run fed the random ids it had fed minutes before, so the n-gram (engram) rows the branch reads were already in memory. With ids it has not seen, the branch reads each such row through a single-threaded 4 KB page fault, about 1.9 s per 512 tokens on this machine [derived], and ran at 73–76 tok/s (2026-09-27, rig-log [v41-xeng](https://github.com/midagedev/rig-log/blob/main/log/2026-09-27.md#v41-xeng)). bloomery prefetches those rows. ~~A table from one window, with both engines' rows warm, is next.~~ The release table above is one window; its llama.cpp rows are not all warm, and the warm table is scheduled (see the top of this section).

### Qwen3-30B-A3B-Instruct-2507, `Q4_K_M` — the whole model on the GPU

Against mainline llama.cpp (`53ed051ce`, `llama-bench -ngl 99 -fa on`) and mistral.rs (`d5ae0f18f`, built with `--features "cuda flash-attn"`, its recommended build), in one window, arms alternated, three rounds (2026-09-27, rig-log [qwen3-xeng](https://github.com/midagedev/rig-log/blob/main/log/2026-09-27.md#qwen3-xeng)).

**Decode** (tok/s, `n = 96`):

| Depth | bloomery | llama.cpp | mistral.rs | bloomery / llama.cpp | bloomery / mistral.rs |
|---:|---:|---:|---:|---:|---:|
| 6 | **209.2** | 193.8 | 194.2 | 1.080 ± 0.006 | 1.077 ± 0.007 |
| 1024 | **202.3** | 188.9 | 167.0 | 1.071 ± 0.001 | 1.211 ± 0.007 |
| 4096 | **175.2** | 173.3 | 156.6 | 1.011 ± 0.002 | 1.119 ± 0.004 |

**Prompt processing** (pp tok/s over a prompt of P tokens; `llama-bench -p P -n 0`, mistral.rs `bench --prompt-len P --gen-len 1`):

| Prompt | bloomery | llama.cpp | llama.cpp `-ub 4096 -b 4096` | mistral.rs |
|---:|---:|---:|---:|---:|
| 512 | **7,827** | 4,247 | — | 4,816 |
| 4096 | **9,276** | 4,157 | 6,831 | 7,812 |

At 4096 tokens bloomery is 1.358 ± 0.002× llama.cpp with `-ub 4096` and 1.187 ± 0.002× mistral.rs. The prompt runs in ubatches of up to 4096 tokens (a load-time size, `BLOOMERY_QWEN3_UBATCH`) through a grouped int8 tensor-core GEMM that reads each expert once per ubatch, and a prefill attention kernel that stages each 64-key tile once for 64 query rows and runs both products on the tensor cores. The router's logits for a ubatch run as register tiles, 32 tokens by 32 experts a block.

### Qwen3.6-35B-A3B, `Q4_K_M` — the whole model on the GPU

Against mainline llama.cpp (`53ed051ce`, `-ngl 99 -fa on`) and mistral.rs (`d5ae0f18f`), in one window, arms alternated, three rounds; all 45 rows clean (release sitting, 2026-09-28; rig-log [q36-release](https://github.com/midagedev/rig-log/blob/main/log/2026-09-28.md#q36-release)).

| Decode, tok/s @ n = 96 | depth 6 | 1024 | 4096 |
|---|---:|---:|---:|
| bloomery | **204.4** | **201.6** | **195.8** |
| llama.cpp | 163.8 | 162.8 | 159.2 |
| mistral.rs | 114.5 | — | 110.3 |

| Prompt, pp tok/s | P = 512 | P = 4096 |
|---|---:|---:|
| bloomery | **6,464** | **8,311** |
| llama.cpp | 3,225 | 3,378 |
| llama.cpp `-ub 4096 -b 4096` | — | 5,036 |
| mistral.rs | 3,695 | — |

bloomery decodes at 1.23–1.25× llama.cpp and 1.78× mistral.rs, and runs a prompt at 2.00× llama.cpp at P = 512 and 1.65× llama.cpp's `-ub 4096` at P = 4096. llama.cpp's automatic placement put the whole model on the card, as the hand-set one does, and read the same (0.997–0.999×).

### Qwen3.8-Flash-Next, `UD-Q4_K_XL` — one GPU plus every routed expert on the CPU: llama.cpp is ahead

Against mainline llama.cpp (`53ed051ce`, `-ngl 99 -fa on -lzm off -ncmoe 26 -t 32`), in one window, engine blocks, three rounds (release sitting, 2026-09-28; rig-log [q38-release](https://github.com/midagedev/rig-log/blob/main/log/2026-09-28.md#q38-release)).

| A6000, tok/s | decode, depth 6 | 1024 | 3000 | prompt, P = 512 | P = 4096 |
|---|---:|---:|---:|---:|---:|
| bloomery | 40.46 | 39.45 | 38.76 | 105.7 | 104.0 |
| llama.cpp | **44.68** | **44.03** | **43.21** | **340.4** | **356.6** |
| llama.cpp `-ub 4096 -b 4096` | — | — | — | — | **776.3** |

bloomery decodes at about 0.90× llama.cpp and runs a prompt at 0.29–0.31× (0.13× against `-ub 4096`). The reasons: our engine runs every routed expert of this model on the CPU, and it has no batched prompt path for it yet; it feeds the prompt in passes of eight positions. Batched prefill is the next round for this model, then keeping experts on the card (two cards). Our first-round decode rows read about 10 % below the other rounds for a reason not yet named (their page faults bound at most 2.6 % of the window); the depth-1024 and 3000 values include that round, and the depth-6 value is the two rounds without a fault tag. llama.cpp's automatic placement was not measured: with `-ub 4096` it ran out of card memory while capturing its CUDA graph.

### GLM-5.3-Flash, `UD-Q4_K_XL` — one GPU plus CPU experts, context up to 2,051 positions

Against two llama.cpp pull-request builds for this model ([#27752](https://github.com/ggml-org/llama.cpp/pull/27752) at `1d0c76f3c6`, [#27754](https://github.com/ggml-org/llama.cpp/pull/27754) at `86ebfef2c6`), in one window, engine blocks, two rounds (release sitting, 2026-09-28; rig-log [glm-release](https://github.com/midagedev/rig-log/blob/main/log/2026-09-28.md#glm-release)). Our prompt is prose; llama.cpp's is `llama-bench`'s random ids, and its MTP arm runs through `llama-server`.

| Decode, tok/s @ n = 96, A6000 | depth 512 | 1024 |
|---|---:|---:|
| bloomery | 20.95 | 20.99 |
| bloomery, hot list | ≥ 28.14 | ≥ 26.77 |
| llama.cpp #27752 | 17.80 | 17.81 |
| llama.cpp #27754 | 16.78 | 16.63 |
| llama.cpp #27754, MTP draft | 23.29 | — |

- **bloomery with no hot list** decodes at 1.18× #27752 and 1.25–1.26× #27754. llama.cpp's MTP arm is faster than that (23.29); its acceptance line, the same in both rounds, reads `draft acceptance = 0.79452 (58 accepted / 73 generated)`.
- **The hot-list rows are lower bounds.** Every one carried page faults that this runner counts over the whole process, not the timed window, so the values can only be low. Read as bounds, they are at least 1.21× the MTP arm.
- **Context stops at 2,051 positions.** Past that the model's sparse attention selector is needed, which bloomery does not have yet. Our prompt runs one step per token (about 21 tok/s), so there is no prompt row for bloomery; llama.cpp's runs at 110–113 tok/s at P = 512.
- #27752's depth-512 row carries a fault tag in both rounds. llama.cpp's automatic placement read 0.87–0.94× its hand-set one. #27752 with `-ub 4096` could not create its context. exllamav3 (`0740edc2da`) on a 4.05 bpw EXL3 quantization, another file, decodes at 18.48 / 18.02 and runs a prompt at 156.6 (P = 512) and 614.9 (P = 4096).

### Read these numbers with their conditions

- **Rows are plain decoding unless they say DSpark.** The DSpark rows use two cards: the model on the A6000 and the draft on the RTX 3090.
- **The mistral.rs columns are its recommended build** (`flash-attn` on, as upstream's release builds and install script build it on Ampere). The prefill rows we published before 2026-09-26, 3,460 and 1,317 tok/s, came from a build without `flash-attn`, whose prompt attention took the eager P × P path (measured in [q3router-ab](https://github.com/midagedev/rig-log/blob/main/log/2026-09-26.md#q3router-ab); the cause is in [mrs-noflash](https://github.com/midagedev/rig-log/blob/main/log/2026-09-26.md#mrs-noflash)).
- **The card is an A6000 unless a row says RTX 3090.** The 24 GB row emulates a 3090's budget on the A6000; the RTX 3090 rows ran on the real card.
- **The V4.1 hot list was built from routing traces of the same corpora** the prose and code prompts come from, so those rows are its favorable case. With synthetic prompt ids, whose output collapses into a few repeating tokens, the same placement decoded at 34.7 tok/s at depth 6 and 32.3 at depth 4096 (2026-09-24, rig-log [public-q3km-first-timing](https://github.com/midagedev/rig-log/blob/main/log/2026-09-24.md#public-q3km-first-timing)).
- **A row's tags are the runner's witness, not decoration.** `[cold]` means the page faults inside the row's timed window could account for at least 1 % of it (75 µs a fault); a fault can only slow a row. `[cpu-busy]` means the runner saw other CPU work beside the arm; in the 2026-09-28 sitting it counted our own process, a runner defect, and those rows will be taken again.
- **Synthetic and prose prompts do not share a table row.** Random ids route to different experts than text does, and the prose rows run with the hot list.

For V4.1 decode, the host tier's memory bandwidth sets the step: 135–137 GB/s at 32 threads ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-24.md#v41-host-tier-k-rows)). The cost of each part is in [`docs/HARDWARE.md`](docs/HARDWARE.md).

## Target hardware

| | Minimum | Extended |
|---|---|---|
| GPU | RTX 3090 24 GB ×1 (sm_86) | RTX 3090 ×2: the second card runs the DSpark draft (splitting the model across two cards is not built) |
| CPU | AVX2, 8 DDR4 channels | same |
| RAM | 256 GB | same |
| Storage | NVMe for the engram table | same |

The DSpark draft (about 8 GB) has been timed only on a second card. On one card it would take the place of card experts, and that case has not been measured. The development machine is a Threadripper PRO 5975WX (32 cores, 8 DDR4 channels, 264 GB) with an RTX A6000 and an RTX 3090. Details and costs: [`docs/HARDWARE.md`](docs/HARDWARE.md).

## How it is verified

- **Oracle sets against ik_llama.cpp.** Intermediate tensors are dumped from ik_llama.cpp's CPU backend for V4.1 and compared per tensor and per layer. Integer outputs (engram row ids, router top-6 ids, indexer top-k lists) must match exactly, except where two candidates are within a tie band. Float outputs must stay inside bands derived from the two engines' rounding rules.
- **Bit gates.** A captured graph replay must equal the eager run bit for bit. The skewed two-row pass must equal two single steps bit for bit, and so must a rollback. A batched prompt must leave the model in the state one step per token leaves, bit for bit, for prompts of 1 to 4096 tokens and under each prompt setting. Node and launch counts are pinned at build time.
- **Drafts change no output.** With the DSpark or the lookup draft, greedy tokens must equal the plain run's.
- **PPL and KLD.** On the public file, wikitext-2 at 2048 context, 4 chunks (4,092 positions), on the RTX 3090: PPL 2.2401 (bloomery) against 2.2378 (ik_llama.cpp CPU), +0.105 %; KLD 0.00987 ± 0.00049; same top token 97.46 % ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-27.md#sit10)).
- **A gate is shown to fail** on the defect it guards before the fix goes in. Bands are not relaxed to make a gate pass.

Timing runs only on a quiet machine, under a machine-wide lease, with a witness block around every timed region. The protocol is rig-log's [`docs/quiet-machine.md`](https://github.com/midagedev/rig-log/blob/main/docs/quiet-machine.md).

## Limits

- **sm_86 only.** Every kernel is built and gated for Ampere (`--arch sm_86`). Other architectures are not tested.
- **A pinned nightly.** The toolchain is `nightly-2026-08-28`, pinned together with a cuda-oxide git revision. The source comes from our fork of that revision, where fixes wait until upstream takes them (`THIRD_PARTY_NOTICES.md` lists them).
- **One machine.** Timed numbers come from one A6000 in one workstation. The tooling (`tools/box.sh`) assumes a Mac editor and that workstation; [`docs/BUILD.md`](docs/BUILD.md) says what to run on your own host.
- **GLM-5.3 context stops at 2,051 positions**, and its prompt runs one step per token; the sparse-attention selector and batched prompts are next.
- **Qwen3.8 runs every routed expert on the CPU**, and feeds its prompt in passes of eight positions; llama.cpp is ahead on it (see its table).
- **V4.1 prompts are bound by the CPU expert tier.** At 4096 synthetic tokens each layer and batch takes about 74 ms, 69 of them in the host experts; the card's work runs underneath ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-26.md#prefillgroup-ab)). A 4096-token prompt takes about 16.5 s. The next steps are on the host side: faster host-expert kernels, and streaming the hottest host experts to the card over PCIe. The server reuses a cached prompt prefix, so a follow-up turn pays only for its new tokens.

## Build

See [`docs/BUILD.md`](docs/BUILD.md). In one line: a Linux x86-64 host with AVX2, CUDA 13.3, LLVM 21 and `cargo-oxide`, then `cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41` (or `--bin bloomery-chat`, `--bin bloomery-serve-ds41`).

## More

- Measurements with their command lines: [rig-log](https://github.com/midagedev/rig-log), under `log/` (Korean).
- Plan and cost models: [`docs/plan.md`](docs/plan.md) (Korean). GPU design: [`docs/gpu-design.md`](docs/gpu-design.md) (Korean). Placement: [`docs/v41-placement.md`](docs/v41-placement.md) (Korean).
- The working contract for contributors and agents: [`AGENTS.md`](AGENTS.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).

The engine started on DeepSeek-V2-Lite-Chat Q3_K_M, and that path still runs and is gated. On the A6000 (300 W), V2-Lite decode ran at 229.5 tok/s at depth 6 and 199.6 at depth 4096 (2026-09-22, [rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-22-o-the-depth-table-on-the-new-default-and-the-reversal-is-gone.md)).

## Upstream

What this work needed from its reference and its toolchain went upstream:

- ik_llama.cpp [#2455](https://github.com/ikawrakow/ik_llama.cpp/pull/2455): DeepSeek-V4.1 support, the oracle above. Merged 2026-09-21.
- ik_llama.cpp [#2444](https://github.com/ikawrakow/ik_llama.cpp/pull/2444): `GGML_CUDA_NO_PINNED_WEIGHTS`, which loading V4.1 with CPU experts needs. Merged 2026-09-15.
- ik_llama.cpp [#2522](https://github.com/ikawrakow/ik_llama.cpp/pull/2522): loading DeepSeek-V4.1 DSpark drafts. Merged 2026-09-25.
- ik_llama.cpp [#2507](https://github.com/ikawrakow/ik_llama.cpp/pull/2507): V4.1 index keys from the pre-RoPE latent. Open, draft.
- ik_llama.cpp [#2546](https://github.com/ikawrakow/ik_llama.cpp/pull/2546): the SwiGLU limits in DeepSeek-V4 DSpark drafts. Open.
- cuda-oxide [#1314](https://github.com/NVlabs/cuda-oxide/pull/1314) (a constant-folder crash on mixed-width shifts, merged 2026-09-23) and [#1321](https://github.com/NVlabs/cuda-oxide/pull/1321) (unrolling loops whose exit test adds a constant to the counter, merged 2026-09-24); [#1329](https://github.com/NVlabs/cuda-oxide/pull/1329) (installing the cached backend by rename) and [#1346](https://github.com/NVlabs/cuda-oxide/pull/1346) (unrolling range `for` loops) are open.
- llama.cpp [#29008](https://github.com/ggml-org/llama.cpp/pull/29008): message delimiters in the DeepSeek V3.2/V4 chat parser. Merged 2026-09-17.
- Open: mistral.rs [#2430](https://github.com/EricLBuehler/mistral.rs/pull/2430) (non-F32 activations in the CPU GGUF MoE gather) and cutile-rs [#309](https://github.com/NVlabs/cutile-rs/pull/309) (`CudaContext::mem_info`).

Ten more of our fixes are merged in ik_llama.cpp; the full lists are [our ik_llama.cpp PRs](https://github.com/ikawrakow/ik_llama.cpp/pulls?q=is%3Apr+author%3Amidagedev) and [our cuda-oxide PRs](https://github.com/NVlabs/cuda-oxide/pulls?q=is%3Apr+author%3Amidagedev). Toolchain issues not filed yet are in [`docs/upstream/nvlabs-ledger.md`](docs/upstream/nvlabs-ledger.md).

## References and credits

[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and ggml are the numerical reference. The kernels were written by reading ggml's k-quant code and ik_llama.cpp's `iqk_mul_mat` kernels, the tokenizer follows the reference's `llama_tokenize`, and accuracy is defined against their output. Where code or an algorithm was taken, the source comment points at the original file. [exllamav3](https://github.com/turboderp-org/exllamav3) and [mistral.rs](https://github.com/EricLBuehler/mistral.rs) are design references, cited where used (for example `crates/gpu/src/route_core.rs`). All four are MIT. The GPU kernels compile with [cuda-oxide](https://github.com/NVlabs/cuda-oxide) (Apache-2.0).

AI assistants helped write the code and the documentation in this repository. Every number here was measured on the machine by the runners in `tools/ref/` and the GPU gate runner.

## License

MIT, see [`LICENSE`](LICENSE). `crates/oxide-ice-unroll`, a compiler-bug reproducer kept out of the workspace, carries NVIDIA's Apache-2.0 headers. Third-party notices, including the ggml authors' MIT notice for the generated tokenizer table, are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
