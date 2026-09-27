<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img src="docs/assets/logo.svg" width="128" alt="bloomery">
  </picture>
</p>

# bloomery

Hybrid GPU + CPU inference for mixture-of-experts models, written in Rust down to the CUDA kernels.

bloomery runs MoE models that do not fit on one GPU. Each routed layer keeps as many experts on the card as fit; the rest run on the CPU in the same decode step. The host code is Rust, the GPU kernels are Rust compiled with [cuda-oxide](https://github.com/NVlabs/cuda-oxide), and the CPU expert kernels are AVX2 Rust. How the kernels use cuda-oxide: [`docs/cuda-oxide.md`](docs/cuda-oxide.md). It targets one workstation: an Ampere GPU, an eight-channel AVX2 CPU and 256 GB of RAM.

## Models

| Model | File | Runs as |
|---|---|---|
| [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | [`Q3_K_M`](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF) (vcruz305) | GPU + CPU experts; `generate_ds41`, `bloomery-chat`, `bloomery-serve-ds41` |
| [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) (unsloth) | GPU + CPU experts; context up to 2,051 positions |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (unsloth) | GPU + every routed expert on the CPU |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | [`Q4_K_M`](https://huggingface.co/lmstudio-community/Qwen3.6-35B-A3B-GGUF) (lmstudio-community) | whole model on one GPU |
| [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | [`Q4_K_M`](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) (unsloth) | whole model on one GPU |
| [DeepSeek-V2-Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat) | [`Q3_K_M`](https://huggingface.co/mradermacher/DeepSeek-V2-Lite-Chat-GGUF) (mradermacher) | CPU or GPU; the first model, still gated |

Each file is the public upload as downloaded. For GLM-5.3, Qwen3.8 and Qwen3.6 the files were checked against the uploads on 2026-09-28 (every shard's size, the first shard's sha256); the others have no recorded sha256 check yet.

The DSpark draft for V4.1 speculative decoding is not a public upload: it is converted from DeepSeek's V4.1 checkpoint with the converter in llama.cpp's V4.1 pull request, with `dflash.target_layers` set to [37, 38, 39] (the converter writes V4's [38, 39, 40]).

## How it works

- **Experts split between card and host.** Each routed layer keeps some experts on the GPU; the rest run on the AVX2 host tier inside the same captured CUDA graph step. A router-frequency list (`BLOOMERY_HOT_LIST`) can pick which experts stay on the card.
- **V4.1's engram table stays on NVMe.** It does not fit in RAM beside the host experts. Each token reads 48 rows; a helper thread prefetches them.
- **Speculative decoding.** A skewed two-row pass runs two positions one layer apart in one step and verifies a draft token: the DSpark draft on a second card (`BLOOMERY_DRAFT=dspark`) or an n-gram lookup (`BLOOMERY_DRAFT=lookup`). Greedy output is the same with and without a draft.
- **Batched prompts.** V4.1 runs a prompt in batches of up to 512 positions, each expert over all its tokens at once, two batches in flight so the card and the CPU overlap. Qwen3-30B runs ubatches of up to 4096 tokens through an int8 tensor-core GEMM. The result is bit for bit the state one step per token leaves.

## Status

V4.1 runs on the public `Q3_K_M` file as uploaded. `generate_ds41` takes token ids and prints greedy ids; `bloomery-chat` streams text; `bloomery-serve-ds41` serves llama-server's HTTP API (streaming, prefix reuse, reasoning, tool calls; one request at a time). The tokenizer is bit-identical to `llama-tokenize` on its gate's corpora.

In progress: batched prompts for GLM-5.3 and Qwen3.8; GLM-5.3's sparse-attention selector (context past 2,051); faster V4.1 host experts; DeepSeek-V4-Flash-0731.

## Measured numbers

Single stream, on one RTX A6000 (48 GB, 300 W) unless a row says RTX 3090 (24 GB, 250 W), in a 32-core AVX2 workstation. Decode rows generate `n = 96` tokens. Arms are alternated in one window and ratios are taken only within a window. Every row comes from the runners in `tools/ref/` under the quiet-machine protocol; the command lines are in [rig-log](https://github.com/midagedev/rig-log).

**The llama.cpp rows from 2026-09-28 are provisional.** In that sitting llama.cpp ran through `llama-bench`, which feeds new random ids each repetition, so several of its rows read weights or n-gram rows cold (the captions say which). Its V4.1 prompt rows ran with op offload off (`-nopo 1`) only, and its Qwen3.8 rows at the default only, so they may not be its fastest setting. Our rows from that sitting carry a `[cpu-busy]` tag that counted our own process, a runner defect. All of these are re-measured warm, with both engines on the same token ids and llama.cpp at its fastest flags; the conditions are in [`docs/fair-measure.md`](docs/fair-measure.md).

### DeepSeek-V4.1-Flash, `Q3_K_M` — GPU + CPU experts

Placement (a): 2,668 routed experts on the card, 12,692 on the host. Against llama.cpp's V4.1 pull request [#28696](https://github.com/ggml-org/llama.cpp/pull/28696) at `5210c7c` (`-ngl 999 --n-cpu-moe 33 -fa on -t 32 -nopo 1`); synthetic prompt ids, no hot list; 2026-09-28, rig-log [v41-release](https://github.com/midagedev/rig-log/blob/main/log/2026-09-28.md#v41-release).

| A6000, tok/s | bloomery | llama.cpp | ratio |
|---|---:|---:|---:|
| decode, depth 6 | **29.66** | 22.75 | 1.304 |
| decode, depth 4096 | **30.04** | 21.78 | 1.379 |
| prompt, P = 512 | **190.6** | 104.9 | 1.817 |
| prompt, P = 4096 | **358.4** | 76.5 | 4.68 |

- llama.cpp's values are its best round. Its other decode rounds read 20.65 (depth 6) and 19.86 (depth 4096), with page faults bounding up to 7.8 % of the window.
- At P = 4096 llama.cpp faulted in every repetition (up to 27.9 % of the window): this branch reads each n-gram row it has not seen through a page fault, and `llama-bench` feeds new ids each time. bloomery prefetches those rows. With repeated ids the branch ran at 103.8 on 2026-09-25.
- llama.cpp's automatic placement (`-fitt 1024`) was within 0.5 % of the hand-set one. With `-ub 4096` the hand-set placement could not create its context.

On the RTX 3090 (placement `gate`: 1,146 routed experts on the card; hot list), against #28696 at `--n-cpu-moe 37`, two rounds; rig-log [e21ref-release](https://github.com/midagedev/rig-log/blob/main/log/2026-09-28.md#e21ref-release):

| RTX 3090, tok/s | decode, depth 6 | decode, depth 4096 | prompt, P = 4096 |
|---|---:|---:|---:|
| bloomery | **28.14** | **28.92** | 329.3 |
| llama.cpp | 21.59 | 20.85 | — |

With the hot list and a 512-token prose or code prompt, and the DSpark draft on the RTX 3090, three rounds; rig-log [dspark-loop-tps](https://github.com/midagedev/rig-log/blob/main/log/2026-09-25.md#dspark-loop-tps):

| A6000, prompt | plain tok/s | DSpark tok/s | DSpark / plain |
|---|---:|---:|---:|
| prose 512 | 44.8 | **51.3** | 1.146 ± 0.014 |
| code 512 | 43.8 | **50.6** | 1.156 ± 0.062 |

The hot list was built from routing traces of the same corpora, so these rows are its favorable case.

### Qwen3-30B-A3B-Instruct-2507, `Q4_K_M` — whole model on the GPU

Against mainline llama.cpp (`53ed051ce`, `-ngl 99 -fa on`) and mistral.rs (`d5ae0f18f`, `--features "cuda flash-attn"`), three rounds; 2026-09-27, rig-log [qwen3-xeng](https://github.com/midagedev/rig-log/blob/main/log/2026-09-27.md#qwen3-xeng).

| Decode, tok/s | bloomery | llama.cpp | mistral.rs | vs llama.cpp | vs mistral.rs |
|---:|---:|---:|---:|---:|---:|
| depth 6 | **209.2** | 193.8 | 194.2 | 1.080 ± 0.006 | 1.077 ± 0.007 |
| depth 1024 | **202.3** | 188.9 | 167.0 | 1.071 ± 0.001 | 1.211 ± 0.007 |
| depth 4096 | **175.2** | 173.3 | 156.6 | 1.011 ± 0.002 | 1.119 ± 0.004 |

| Prompt, tok/s | bloomery | llama.cpp | llama.cpp `-ub 4096 -b 4096` | mistral.rs |
|---:|---:|---:|---:|---:|
| P = 512 | **7,827** | 4,247 | — | 4,816 |
| P = 4096 | **9,276** | 4,157 | 6,831 | 7,812 |

### Qwen3.6-35B-A3B, `Q4_K_M` — whole model on the GPU

Same llama.cpp and mistral.rs builds, three rounds, all rows clean; 2026-09-28, rig-log [q36-release](https://github.com/midagedev/rig-log/blob/main/log/2026-09-28.md#q36-release).

| Decode, tok/s | depth 6 | 1024 | 4096 |
|---|---:|---:|---:|
| bloomery | **204.4** | **201.6** | **195.8** |
| llama.cpp | 163.8 | 162.8 | 159.2 |
| mistral.rs | 114.5 | — | 110.3 |

| Prompt, tok/s | P = 512 | P = 4096 |
|---|---:|---:|
| bloomery | **6,464** | **8,311** |
| llama.cpp | 3,225 | 3,378 |
| llama.cpp `-ub 4096 -b 4096` | — | 5,036 |
| mistral.rs | 3,695 | — |

### Qwen3.8-Flash-Next, `UD-Q4_K_XL` — GPU + every routed expert on the CPU

llama.cpp is ahead on this model. Mainline llama.cpp (`53ed051ce`, `-ngl 99 -fa on -lzm off -ncmoe 26 -t 32`), three rounds; 2026-09-28, rig-log [q38-release](https://github.com/midagedev/rig-log/blob/main/log/2026-09-28.md#q38-release).

| A6000, tok/s | decode, depth 6 | 1024 | 3000 | prompt, P = 512 | P = 4096 |
|---|---:|---:|---:|---:|---:|
| bloomery | 40.46 | ~~39.45~~ 40.90 | ~~38.76~~ 40.1 | 105.7 | 104.0 |
| llama.cpp | **44.68** | **44.03** | **43.21** | **340.4** | **356.6** |
| llama.cpp `-ub 4096 -b 4096` | — | — | — | — | **776.3** |

bloomery runs every routed expert of this model on the CPU and has no batched prompt path for it yet (it feeds the prompt eight positions at a time). Batched prompts come next, then experts on the card. Our decode values leave out the first round. ~~Our first decode round read about 10 % below the others for a reason not yet found; the depth-6 value is the two rounds without it.~~ Corrected 2026-09-28: the first round read cold rows of the model's PLE table, about 16 page faults a token, and the depth 1024 and 3000 cells had averaged it in. Without it, decode is 0.906×, 0.929× and 0.928× llama.cpp at depth 6, 1024 and 3000 (rig-log [q38-release](https://github.com/midagedev/rig-log/blob/main/log/2026-09-28.md#q38-release)).

### GLM-5.3-Flash, `UD-Q4_K_XL` — GPU + CPU experts

Against two llama.cpp pull-request builds for this model, [#27752](https://github.com/ggml-org/llama.cpp/pull/27752) at `1d0c76f3c6` and [#27754](https://github.com/ggml-org/llama.cpp/pull/27754) at `86ebfef2c6`, two rounds; 2026-09-28, rig-log [glm-release](https://github.com/midagedev/rig-log/blob/main/log/2026-09-28.md#glm-release).

| Decode, tok/s | depth 512 | 1024 |
|---|---:|---:|
| bloomery | 20.95 | 20.99 |
| bloomery, hot list | ≥ 28.14 | ≥ 26.77 |
| llama.cpp #27752 | 17.80 | 17.81 |
| llama.cpp #27754 | 16.78 | 16.63 |
| llama.cpp #27754, MTP draft | 23.29 | — |

- The hot-list rows are lower bounds: they carried page faults that this runner counts over the whole process, not the timed window.
- llama.cpp's MTP arm accepted 58 of 73 draft tokens.
- Context stops at 2,051 positions until the sparse-attention selector is built. The prompt runs one step per token (about 21 tok/s), so there is no prompt row for bloomery; llama.cpp's is 110–113 tok/s at P = 512.
- exllamav3 (`0740edc2da`) on a 4.05 bpw EXL3 file, another quantization, decodes at 18.48 / 18.02 and runs a prompt at 156.6 (P = 512) and 614.9 (P = 4096).

### Tags

`[cold]` means page faults inside a row's timed window could account for at least 1 % of it (75 µs a fault); a fault only slows a row. `[cpu-busy]` means the runner saw other CPU work beside the arm. Synthetic and prose prompts never share a row: they route to different experts.

## Target hardware

| | Minimum | Extended |
|---|---|---|
| GPU | RTX 3090 24 GB (sm_86) | a second RTX 3090 for the DSpark draft |
| CPU | AVX2, 8 DDR4 channels | same |
| RAM | 256 GB | same |
| Storage | NVMe for the engram table | same |

The development machine is a Threadripper PRO 5975WX (32 cores, 8 DDR4 channels, 264 GB) with an RTX A6000 and an RTX 3090. V4.1 decode is bound by host memory bandwidth, 135–137 GB/s at 32 threads. Costs per part: [`docs/HARDWARE.md`](docs/HARDWARE.md).

## How it is verified

- **Against ik_llama.cpp.** Intermediate tensors are dumped from ik_llama.cpp's CPU backend and compared per layer. Integer outputs (router ids, engram rows, indexer top-k) must match exactly outside tie bands; float outputs stay inside bands derived from both engines' rounding.
- **Bit gates.** Graph replay equals eager execution; the skewed pass equals two single steps; a rollback and a batched prompt leave the same state as one step per token.
- **PPL and KLD** on the public V4.1 file (wikitext-2, 2048 context, 4 chunks, RTX 3090): PPL 2.2401 against ik_llama.cpp's 2.2378; KLD 0.00987 ± 0.00049; same top token 97.46 % ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-27.md#sit10)).
- **Each gate is shown to fail** on its defect before the fix lands. Bands are not relaxed to pass.

## Limits

- **sm_86 only**, and one machine: the tooling (`tools/box.sh`) assumes a Mac editor and that workstation.
- **A pinned nightly** (`nightly-2026-08-28`) with a pinned cuda-oxide revision from our fork, where fixes wait for upstream (`THIRD_PARTY_NOTICES.md`).
- **V4.1 prompts are bound by the CPU expert tier**: about 69 of 74 ms per layer-batch at 4096 tokens is host experts. The server reuses a cached prompt prefix.
- **GLM-5.3** stops at 2,051 positions and has no batched prompt; **Qwen3.8** runs all routed experts on the CPU.

## Build

See [`docs/BUILD.md`](docs/BUILD.md). In short: Linux x86-64 with AVX2, CUDA 13.3, LLVM 21 and `cargo-oxide`, then `cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41`.

## Upstream

- ik_llama.cpp: DeepSeek-V4.1 support ([#2455](https://github.com/ikawrakow/ik_llama.cpp/pull/2455)), `GGML_CUDA_NO_PINNED_WEIGHTS` ([#2444](https://github.com/ikawrakow/ik_llama.cpp/pull/2444)), V4.1 DSpark drafts ([#2522](https://github.com/ikawrakow/ik_llama.cpp/pull/2522)), and ten more fixes, merged; [#2507](https://github.com/ikawrakow/ik_llama.cpp/pull/2507) and [#2546](https://github.com/ikawrakow/ik_llama.cpp/pull/2546) open.
- cuda-oxide: [#1314](https://github.com/NVlabs/cuda-oxide/pull/1314) and [#1321](https://github.com/NVlabs/cuda-oxide/pull/1321) merged; [#1329](https://github.com/NVlabs/cuda-oxide/pull/1329) and [#1346](https://github.com/NVlabs/cuda-oxide/pull/1346) open. Unfiled toolchain issues: [`docs/upstream/nvlabs-ledger.md`](docs/upstream/nvlabs-ledger.md).
- llama.cpp [#29008](https://github.com/ggml-org/llama.cpp/pull/29008) merged; mistral.rs [#2430](https://github.com/EricLBuehler/mistral.rs/pull/2430) and cutile-rs [#309](https://github.com/NVlabs/cutile-rs/pull/309) open.

## More

- How bloomery uses cuda-oxide (layout, launch contracts, pinning): [`docs/cuda-oxide.md`](docs/cuda-oxide.md).
- Measurements and command lines: [rig-log](https://github.com/midagedev/rig-log) (Korean).
- Plan and cost models: [`docs/plan.md`](docs/plan.md); GPU design: [`docs/gpu-design.md`](docs/gpu-design.md); placement: [`docs/v41-placement.md`](docs/v41-placement.md) (Korean).
- Working contract: [`AGENTS.md`](AGENTS.md), [`CONTRIBUTING.md`](CONTRIBUTING.md).

## References and credits

[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and ggml are the numerical reference: the kernels were written by reading ggml's k-quant code and ik_llama.cpp's `iqk_mul_mat`, and accuracy is defined against their output. [exllamav3](https://github.com/turboderp-org/exllamav3) and [mistral.rs](https://github.com/EricLBuehler/mistral.rs) are design references, cited where used. All four are MIT. The GPU kernels compile with [cuda-oxide](https://github.com/NVlabs/cuda-oxide) (Apache-2.0).

AI assistants helped write the code and the documentation. Every number here was measured by the runners in `tools/ref/`.

## License

MIT, see [`LICENSE`](LICENSE). `crates/oxide-ice-unroll` carries NVIDIA's Apache-2.0 headers. Third-party notices are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
