<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img src="docs/assets/logo.svg" width="128" alt="bloomery">
  </picture>
</p>

# bloomery

MoE inference in Rust, from the HTTP server down to the CUDA kernels.

DeepSeek-V4.1-Flash `Q3_K_M` on an RTX A6000 and a 32-core CPU: 29.66 tok/s decode at depth 6 and 30.04 at depth 4096, 358.4 tok/s prompt at P = 4096; provisional until the warm re-measure ([conditions](#against-llamacpp-deepseek-v41-flash), [all numbers](https://github.com/midagedev/rig-log/blob/main/docs/bloomery-bench.md)).

- More than 200 CUDA kernels, all written in Rust with [cuda-oxide](https://github.com/NVlabs/cuda-oxide) ([how](docs/cuda-oxide.md)).
- Experts on the GPU and the AVX2 CPU in one CUDA graph step.
- DSpark speculative decoding with the draft on a second card; greedy output is unchanged.
- V4.1's batched prompts leave the state of one step per token, bit for bit.
- A llama-server-compatible HTTP API.
- Every number comes from a runner and links its log.

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

- **Placement.** Each layer keeps a fixed number of routed experts on the card, the lowest ids by default. A router-frequency list (`BLOOMERY_HOT_LIST`) can pick them instead, but our lists were learned from the same corpora as our test prompts, so the headline numbers use none. Adaptive residency, which moves experts by the engine's own routing as it runs, is in progress.
- **Engram rows from NVMe.** V4.1's engram table is read from NVMe, 48 rows a token, prefetched by a helper thread.
- **Skewed pass.** Speculative decoding runs two positions one layer apart in one step and verifies a draft token: the DSpark draft (`BLOOMERY_DRAFT=dspark`) or an n-gram lookup (`BLOOMERY_DRAFT=lookup`).
- **Batched prompts.** V4.1 runs a prompt in batches of up to 512 positions, each expert over all its tokens at once, two batches in flight so the card and the CPU overlap. Qwen3-30B runs ubatches of up to 4096 tokens through an int8 tensor-core GEMM.

## Status

V4.1 runs on the public `Q3_K_M` file as uploaded. `generate_ds41` takes token ids and prints greedy ids; `bloomery-chat` streams text; `bloomery-serve-ds41` serves llama-server's HTTP API (streaming, prefix reuse, reasoning, tool calls; one request at a time). The tokenizer is bit-identical to `llama-tokenize` on its gate's corpora.

In progress: batched prompts for GLM-5.3 and Qwen3.8; GLM-5.3's sparse-attention selector (context past 2,051); faster V4.1 host experts; DeepSeek-V4-Flash-0731.

## Measured numbers

All numbers are single-stream tok/s on the development machine: an RTX A6000 (48 GB, 300 W) unless a row says RTX 3090 (24 GB, 250 W), with a 32-core AVX2 CPU. Decode generates `n = 96` tokens. Every number comes from the runners in `tools/ref/` under the quiet-machine protocol, and each row links to its rig-log entry with the command lines. Every measured number, the other engines' rows and bloomery's progress over time are on the living benchmark page, [rig-log `docs/bloomery-bench.md`](https://github.com/midagedev/rig-log/blob/main/docs/bloomery-bench.md), rendered from the runner logs after each sitting.

### Against llama.cpp: DeepSeek-V4.1-Flash

Placement (a): 2,668 routed experts on the card, 12,692 on the host. Against llama.cpp's V4.1 pull request [#28696](https://github.com/ggml-org/llama.cpp/pull/28696) at `5210c7c` (`-ngl 999 --n-cpu-moe 33 -fa on -t 32 -nopo 1`), synthetic prompt ids, no hot list, one window; 2026-09-28, rig-log [v41-release](https://github.com/midagedev/rig-log/blob/main/log/2026-09-28.md#v41-release).

| A6000, tok/s | bloomery | llama.cpp | ratio |
|---|---:|---:|---:|
| decode, depth 6 | **29.66** | 22.75 | 1.304 |
| decode, depth 4096 | **30.04** | 21.78 | 1.379 |
| prompt, P = 512 | **190.6** | 104.9 | 1.817 |
| prompt, P = 4096 | **358.4** | 76.5 | 4.68 |

This table is provisional. llama.cpp ran through `llama-bench`, which feeds new random ids on every repetition, so its rows read n-gram (engram) rows it had not seen through page faults, in every repetition at P = 4096; with repeated ids the branch ran at 103.8 there on 2026-09-25. Its prompt rows ran with op offload off only. A warm re-measure, with both engines on the same token ids and llama.cpp at its fastest flags, replaces it; the conditions are in [`docs/fair-measure.md`](docs/fair-measure.md).

### On this machine

| Model | Setup | Decode, tok/s | Prompt, tok/s | vs llama.cpp | rig-log |
|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash `Q3_K_M` | A6000 + CPU, hot list (in-sample), prose prompt of 512 | 44.8; **51.3** with the DSpark draft on the 3090 | — | not measured | [dspark-loop-tps](https://github.com/midagedev/rig-log/blob/main/log/2026-09-25.md#dspark-loop-tps) |
| DeepSeek-V4.1-Flash `Q3_K_M` | RTX 3090 + CPU, hot list (not the default), synthetic ids | 28.14 (depth 6), 28.92 (depth 4096) | 329.3 (P = 4096) | decode faster; prompt not measured | [e21ref-release](https://github.com/midagedev/rig-log/blob/main/log/2026-09-28.md#e21ref-release) |
| GLM-5.3-Flash `UD-Q4_K_XL` | A6000 + CPU, prose prompt | 20.95 (depth 512), 20.99 (depth 1024) | — | decode faster, but slower than its MTP draft; prompt slower | [glm-release](https://github.com/midagedev/rig-log/blob/main/log/2026-09-28.md#glm-release) |
| Qwen3.8-Flash-Next `UD-Q4_K_XL` | A6000 + every routed expert on the CPU | 40.46 (depth 6), 40.90 (depth 1024) | 105.7 (P = 512), 104.0 (P = 4096) | slower | [q38-release](https://github.com/midagedev/rig-log/blob/main/log/2026-09-28.md#q38-release) |
| Qwen3.6-35B-A3B `Q4_K_M` | A6000, whole model | 204.4 (depth 6), 195.8 (depth 4096) | 6,464 (P = 512), 8,311 (P = 4096) | faster | [q36-release](https://github.com/midagedev/rig-log/blob/main/log/2026-09-28.md#q36-release) |
| Qwen3-30B-A3B-Instruct-2507 `Q4_K_M` | A6000, whole model | 209.2 (depth 6), 175.2 (depth 4096) | 7,827 (P = 512), 9,276 (P = 4096) | faster; near par in decode at depth 4096 | [qwen3-xeng](https://github.com/midagedev/rig-log/blob/main/log/2026-09-27.md#qwen3-xeng) |

- "vs llama.cpp" is the direction in the same window as the linked rig-log entry (decode and prompt alike unless it says otherwise): mainline llama.cpp, or the model's llama.cpp pull request. The rows from 2026-09-28 are provisional, as above.
- The V4.1 hot list was built from routing traces of the same corpora the prose prompt comes from, and the prompt's 512 ids are inside the list's training range: the row is in-sample, its favorable case. Since 2026-09-28 the default placement is the id prefix, and no headline uses a hot list. The warm re-measure uses held-out prompts ([`docs/fair-measure.md`](docs/fair-measure.md) §4.2).
- GLM-5.3's prompt runs one step per token for now (about 21 tok/s), so it has no prompt value. With a hot list (learned from earlier text of the same corpus: held out by position, but in-domain) its decode is at least 28.14 at depth 512; that row is a lower bound, because its page faults were counted over the whole process.
- Qwen3.8's prompt is fed eight positions at a time; batched prompts are next. The rig-log entries carry the other engines' rows from the same windows, including where llama.cpp is ahead.

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
- **Bit gates.** Graph replay equals eager execution; the skewed pass equals two single steps; a rollback and a V4.1 batched prompt leave the same state as one step per token.
- **PPL and KLD** on the public V4.1 file (wikitext-2, 2048 context, 4 chunks, RTX 3090): PPL 2.2401 against ik_llama.cpp's 2.2378; KLD 0.00987 ± 0.00049; same top token 97.46 % ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-27.md#sit10)).
- **Each gate is shown to fail** on its defect before the fix lands. Bands are not relaxed to pass.

## Limits

- **sm_86 only.** The tooling (`tools/box.sh`) assumes our development setup; [`docs/BUILD.md`](docs/BUILD.md) says what to run on your own host.
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
