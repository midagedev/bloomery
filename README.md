<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img src="docs/assets/logo.svg" width="128" alt="bloomery">
  </picture>
</p>

# bloomery

MoE inference in Rust, from the HTTP server down to the CUDA kernels.

DeepSeek-V4.1-Flash `Q3_K_M` on an RTX A6000 and a 32-core CPU: 29.66 tok/s decode at depth 6 and 30.04 at depth 4096, 358.4 tok/s prompt at P = 4096; provisional until the warm re-measure ([conditions](#against-llamacpp-deepseek-v41-flash), [all numbers](https://github.com/midagedev/rig-log/blob/main/docs/bloomery-bench.md)).

- More than 200 CUDA kernels, all written in Rust with [cuda-oxide](https://github.com/NVIDIA/cuda-rust) ([how](docs/cuda-oxide.md)).
- Experts on the GPU and the AVX2 CPU in one CUDA graph step.
- Adaptive residency: routed experts move between the card and the host by the engine's own routing as it runs.
- Speculative decoding with greedy output unchanged: DSpark for V4.1 (the draft on a second card), the MTP head for Qwen3.8.
- V4.1's batched prompts leave the state of one step per token, bit for bit.
- A llama-server-compatible HTTP API.
- Every number comes from a runner and links its log.

## Models

| Model | File | Runs as |
|---|---|---|
| [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | [`Q3_K_M`](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF) (vcruz305) | GPU + CPU experts; `generate_ds41`, `bloomery-chat`, `bloomery-serve-ds41` |
| [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) (unsloth) | GPU + CPU experts; sparse attention past 2,051 positions; `generate_glm5next` |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | [`UD-Q4_K_XL`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (unsloth) | GPU + CPU experts; `generate_qwen3moe`, `bloomery-serve-qwen38` |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | [`Q4_K_M`](https://huggingface.co/lmstudio-community/Qwen3.6-35B-A3B-GGUF) (lmstudio-community) | whole model on one GPU; `generate_qwen3moe` |
| [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | [`Q4_K_M`](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) (unsloth) | whole model on one GPU; `generate_qwen3moe` |
| [DeepSeek-V2-Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat) | [`Q3_K_M`](https://huggingface.co/mradermacher/DeepSeek-V2-Lite-Chat-GGUF) (mradermacher) | CPU or GPU; the first model, still gated |

Each file is the public upload as downloaded. For GLM-5.3, Qwen3.8 and Qwen3.6 the files were checked against the uploads on 2026-09-28 (every shard's size, the first shard's sha256); the others have no recorded sha256 check yet.

The DSpark draft for V4.1 speculative decoding is not a public upload: it is converted from DeepSeek's V4.1 checkpoint with the converter in llama.cpp's V4.1 pull request, with `dflash.target_layers` set to [37, 38, 39] (the converter writes V4's [38, 39, 40]).

## How it works

- **Placement.** Each layer starts with a fixed number of routed experts on the card, the lowest ids. With `--place bp`, V4.1 puts the model on the A6000 and more routed experts (and the DSpark draft) on the 3090. Qwen3.8 runs its card share of the experts on the card (`BLOOMERY_QWEN38_EXPERTS=card`, the default).
- **Adaptive residency.** Under V4.1's serving placements (`--place a` and `bp`) the engine counts its own routing and swaps experts between the card and the host between steps (`BLOOMERY_RESIDENCY`, on by default there). A prompt call also streams its hottest host experts into the card's pool (`BLOOMERY_HOSTSTREAM`), so decode starts on experts that fit the prompt.
- **Engram rows from NVMe.** V4.1's engram table is read from NVMe, 48 rows a token, prefetched by a helper thread.
- **Skewed pass.** Speculative decoding runs two positions one layer apart in one step and verifies a draft token: the DSpark draft (`BLOOMERY_DRAFT=dspark`) or an n-gram lookup (`BLOOMERY_DRAFT=lookup`).
- **Batched prompts.** V4.1 runs a prompt in batches of up to 512 positions, each expert over all its tokens at once, two batches in flight so the card and the CPU overlap. Qwen3-30B and Qwen3.6 run ubatches of up to 4096 tokens through an int8 tensor-core GEMM; Qwen3.8 runs ubatches of up to 4096 with its card experts beside the host tier; GLM-5.3 runs its prompt in batches, its sparse-attention selector included.
- **Loading.** A load streams the file's bytes to the card through a pinned ring with one sync; a warm V4.1 load takes about 16 s ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-30.md#v41-load-upload)).

<p align="center">
  <a href="https://github.com/midagedev/rig-log/blob/main/assets/residency-explainer-v5.mp4">
    <img src="https://raw.githubusercontent.com/midagedev/rig-log/main/assets/residency-explainer-v5-poster.png" width="420" alt="Adaptive expert residency explainer (42 s video)">
  </a>
</p>

Adaptive residency on one A6000, V4.1-Flash `Q3_K_M`, 96 decode steps after a 512-token prose prompt: 29.48 tok/s with it off, 35.70 with the swap rule, 43.86 with the prompt call's streaming as well (the default). The card served 18 % of the routed calls with it off; with streaming it served 57 % over the first 16 steps and 71 % over the last 16 ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-30.md#residency-clip); the video's source and a source for every number: [`tools/clip/residency`](https://github.com/midagedev/rig-log/tree/main/tools/clip/residency)).

## Status

V4.1 runs on the public `Q3_K_M` file as uploaded. `generate_ds41` takes token ids and prints greedy ids; `bloomery-chat` streams text; `bloomery-serve-ds41` and `bloomery-serve-qwen38` serve llama-server's HTTP API, streaming, one request at a time. The V4.1 server also reuses a cached prompt prefix, splits reasoning and returns tool calls; the Qwen3.8 server keeps no prefix (every request prefills from a reset). GLM-5.3 and Qwen3.6 run from their generator bins and have no server yet.<!-- pending: no-glm-qwen36-server --> The Qwen and GLM chat templates render as jinja2 does on their gates' cases, and the tokenizer is bit-identical to `llama-tokenize` on its gate's corpora.

In progress: the warm re-measure against llama.cpp (below); Qwen3.8's MTP draft speed; GLM-5.3's prompt speed since batching; faster V4.1 host experts; DeepSeek-V4-Flash-0731.

## Measured numbers

All numbers are single-stream tok/s on the development machine: an RTX A6000 (48 GB, 300 W) unless a row says RTX 3090 (24 GB, 250 W), with a 32-core AVX2 CPU. Decode generates `n = 96` tokens. Every number comes from the runners in `tools/ref/` under the quiet-machine protocol, and each row links to its rig-log entry with the command lines. Every measured number, the other engines' rows and bloomery's progress over time are on the living benchmark page, [rig-log `docs/bloomery-bench.md`](https://github.com/midagedev/rig-log/blob/main/docs/bloomery-bench.md), rendered from the runner logs after each sitting.

### Against llama.cpp: DeepSeek-V4.1-Flash

Placement (a): 2,668 routed experts on the card, 12,692 on the host. Against llama.cpp's V4.1 pull request [#28696](https://github.com/ggml-org/llama.cpp/pull/28696) at `5210c7c` (`-ngl 999 --n-cpu-moe 33 -fa on -t 32 -nopo 1`), synthetic prompt ids, the id-prefix placement with adaptive residency off (it was not yet the default), one window; 2026-09-28, rig-log [v41-release](https://github.com/midagedev/rig-log/blob/main/log/2026-09-28.md#v41-release).

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
| DeepSeek-V4.1-Flash `Q3_K_M` | A6000 + CPU, adaptive residency with prompt streaming (today's default), prose prompt | 43.30 after a prompt of 512, 36.77 after 4096 | 191.97 (P = 512), 376.67 (P = 4096) | not measured | [callstream-pp-a](https://github.com/midagedev/rig-log/blob/main/log/2026-09-30.md#callstream-pp-a) |
| GLM-5.3-Flash `UD-Q4_K_XL` | A6000 + CPU, prose prompt | 20.95 (depth 512), 20.99 (depth 1024) | — | decode faster, but slower than its MTP draft; prompt slower | [glm-release](https://github.com/midagedev/rig-log/blob/main/log/2026-09-28.md#glm-release) |
| Qwen3.8-Flash-Next `UD-Q4_K_XL` | A6000 + CPU, card experts (today's default), prose prompt | 53.85 (depth 512), 53.47 (depth 4096) | 659.2 (P = 512), 618.2 (P = 4096) | not measured | [q38prose-pp](https://github.com/midagedev/rig-log/blob/main/log/2026-09-29.md#q38prose-pp) |
| Qwen3.6-35B-A3B `Q4_K_M` | A6000, whole model | 204.4 (depth 6), 195.8 (depth 4096) | 6,464 (P = 512), 8,311 (P = 4096) | faster | [q36-release](https://github.com/midagedev/rig-log/blob/main/log/2026-09-28.md#q36-release) |
| Qwen3-30B-A3B-Instruct-2507 `Q4_K_M` | A6000, whole model | 209.2 (depth 6), 175.2 (depth 4096) | 7,827 (P = 512), 9,276 (P = 4096) | faster; near par in decode at depth 4096 | [qwen3-xeng](https://github.com/midagedev/rig-log/blob/main/log/2026-09-27.md#qwen3-xeng) |

- "vs llama.cpp" is the direction in the same window as the linked rig-log entry (decode and prompt alike unless it says otherwise): mainline llama.cpp, or the model's llama.cpp pull request. "Not measured" means that window had no llama.cpp rows. The rows from 2026-09-28 are provisional, as above.
- The V4.1 and Qwen3.8 rows are two-round windows of bloomery alone; decode follows the prompt it names. Qwen3.8 on 2026-09-28, with every routed expert on the CPU, ran 40.46 decode and 104.0 prompt at P = 4096, slower than llama.cpp in that window.
- GLM-5.3's row fed its prompt one step per token (about 21 tok/s); its batched prompt has landed since and is not re-measured yet.

### On two cards: DeepSeek-V4.1-Flash

`--place bp`: the A6000 with the model and 2,668 routed experts, the RTX 3090 (250 W) with more routed experts. These rows do not share a table with the one-card rows.

| A6000 + 3090, tok/s | Decode | Prompt | rig-log |
|---|---|---|---|
| adaptive residency with prompt streaming, prose prompt | 44.28 after a prompt of 512, 37.45 after 4096 | 205.48 (P = 512), 388.93 (P = 4096) | [callstream-pp-bp](https://github.com/midagedev/rig-log/blob/main/log/2026-09-30.md#callstream-pp-bp) |

## Target hardware

| | Minimum | Extended |
|---|---|---|
| GPU | RTX 3090 24 GB (sm_86) | an RTX A6000 48 GB with the RTX 3090 (`--place bp`) |
| CPU | AVX2, 8 DDR4 channels | same |
| RAM | 256 GB | same |
| Storage | NVMe for the engram table | same |

The development machine is a Threadripper PRO 5975WX (32 cores, 8 DDR4 channels, 264 GB) with an RTX A6000 and an RTX 3090. A card is picked by its name (`A6000`, `3090`), so two RTX 3090s are not supported: make one visible with `CUDA_VISIBLE_DEVICES`. The CPU build targets Zen 3 (`target-cpu=znver3`); another CPU needs the flag changed ([`docs/BUILD.md`](docs/BUILD.md#the-cpu-flag)).

Every throughput number in this README ran on the A6000: the one-card rows on the A6000 alone, the two-card rows on the A6000 with the 3090 (`--place bp`). No row ran on a 3090 alone yet; those rows come with the release re-measure. V4.1 decode is bound by host memory bandwidth, 135–137 GB/s at 32 threads. Costs per part: [`docs/HARDWARE.md`](docs/HARDWARE.md).

## How it is verified

- **Against ik_llama.cpp.** Intermediate tensors are dumped from ik_llama.cpp's CPU backend and compared per layer. Integer outputs (router ids, engram rows, indexer top-k) must match exactly outside tie bands; float outputs stay inside bands derived from both engines' rounding.
- **Bit gates.** Graph replay equals eager execution; the skewed pass equals two single steps; a rollback and a V4.1 batched prompt leave the same state as one step per token.
- **PPL and KLD** on the public V4.1 file (wikitext-2, 2048 context, 4 chunks, RTX 3090): PPL 2.2401 against ik_llama.cpp's 2.2378; KLD 0.00987 ± 0.00049; same top token 97.46 % ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-27.md#sit10)).
- **Each gate is shown to fail** on its defect before the fix lands. Bands are not relaxed to pass.

## Limits

- **sm_86 only.** The `just` recipes and `tools/box.sh` are the maintainers' tooling for one remote workstation; [`docs/BUILD.md`](docs/BUILD.md) gives the direct commands for your own host.
- **A pinned nightly** (`nightly-2026-08-28`) with a pinned cuda-oxide revision from our fork, where fixes wait for upstream (`THIRD_PARTY_NOTICES.md`).
- **V4.1 prompts are bound by the CPU expert tier**: the host experts are most of each layer-batch. The V4.1 server reuses a cached prompt prefix; the Qwen3.8 server does not.
- **One request at a time** in both servers. GLM-5.3 and Qwen3.6 have no server yet.<!-- pending: no-glm-qwen36-server -->
- **Qwen3.8** keeps the routed experts past its card share on the CPU, and its MTP draft's speed is not measured yet.

## Build

See [`docs/BUILD.md`](docs/BUILD.md): the toolchain, one command block per model (download, build, generate, serve, and `--place gate` on a single RTX 3090), and the CPU flag. In short: Linux x86-64 (the build targets Zen 3), CUDA 13.3, LLVM 21, Clang 21 and `cargo-oxide` from our cuda-oxide fork, then `cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41` and `BLOOMERY_REF_MODEL=<first shard> target/release/generate_ds41 --place gate --tokens 671,6102,294,8760,344`.

## Upstream

- ik_llama.cpp: DeepSeek-V4.1 support ([#2455](https://github.com/ikawrakow/ik_llama.cpp/pull/2455)), `GGML_CUDA_NO_PINNED_WEIGHTS` ([#2444](https://github.com/ikawrakow/ik_llama.cpp/pull/2444)), V4.1 DSpark drafts ([#2522](https://github.com/ikawrakow/ik_llama.cpp/pull/2522), [#2546](https://github.com/ikawrakow/ik_llama.cpp/pull/2546)), and eleven more fixes, merged; [#2507](https://github.com/ikawrakow/ik_llama.cpp/pull/2507) and [#2562](https://github.com/ikawrakow/ik_llama.cpp/pull/2562) open.
- cuda-oxide: [#1314](https://github.com/NVIDIA/cuda-rust/pull/1314) and [#1321](https://github.com/NVIDIA/cuda-rust/pull/1321) merged; [#1329](https://github.com/NVIDIA/cuda-rust/pull/1329) and [#1346](https://github.com/NVIDIA/cuda-rust/pull/1346) open. Unfiled toolchain issues: [`docs/upstream/nvlabs-ledger.md`](docs/upstream/nvlabs-ledger.md).
- llama.cpp [#29008](https://github.com/ggml-org/llama.cpp/pull/29008) merged; mistral.rs [#2430](https://github.com/EricLBuehler/mistral.rs/pull/2430) and cutile-rs [#309](https://github.com/NVlabs/cutile-rs/pull/309) open.

## More

- How bloomery uses cuda-oxide (layout, launch contracts, pinning): [`docs/cuda-oxide.md`](docs/cuda-oxide.md).
- Measurements and command lines: [rig-log](https://github.com/midagedev/rig-log) (Korean).
- A recorded session, V4.1 on two cards answering a coding review (507 prompt tokens, 1,500 generated): [toktape](https://tape.midagedev.com/r/6w4t9r5nqwtt5c9sagn3). A recording, not a benchmark row.
- Plan and cost models: [`docs/plan.md`](docs/plan.md); GPU design: [`docs/gpu-design.md`](docs/gpu-design.md); placement: [`docs/v41-placement.md`](docs/v41-placement.md) (Korean).
- Working contract: [`AGENTS.md`](AGENTS.md), [`CONTRIBUTING.md`](CONTRIBUTING.md).

## References and credits

[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and ggml are the numerical reference: the kernels were written by reading ggml's k-quant code and ik_llama.cpp's `iqk_mul_mat`, and accuracy is defined against their output. [exllamav3](https://github.com/turboderp-org/exllamav3) and [mistral.rs](https://github.com/EricLBuehler/mistral.rs) are design references, cited where used. All four are MIT. The GPU kernels compile with [cuda-oxide](https://github.com/NVIDIA/cuda-rust) (Apache-2.0).

AI assistants helped write the code and the documentation. Every number here was measured by the runners in `tools/ref/`.

## License

MIT, see [`LICENSE`](LICENSE). `crates/oxide-ice-unroll` carries NVIDIA's Apache-2.0 headers. Third-party notices are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
