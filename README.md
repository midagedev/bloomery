<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img src="docs/assets/logo.svg" width="128" alt="bloomery">
  </picture>
</p>

# bloomery

Hybrid GPU + CPU inference for mixture-of-experts models, written in Rust down to the CUDA kernels.

bloomery runs mixture-of-experts models that do not fit on one GPU. Each routed layer keeps as many of its experts on the card as fit, and the rest run on the CPU inside the same decode step. The host code is Rust, the GPU kernels are CUDA written in Rust ([cuda-oxide](https://github.com/NVlabs/cuda-oxide)), and the CPU expert kernels are AVX2 Rust. Today it runs DeepSeek-V4.1-Flash on one GPU with an eight-channel CPU and 256 GB of RAM (see [Target hardware](#target-hardware)).

## Models

The files below are the ones the gates and the numbers use; each is byte-identical (sha256) to the upload linked.

| Model | File | Runs as | Checked by |
|---|---|---|---|
| [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | [`Q3_K_M`, 9 shards](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF) (vcruz305) | one GPU + CPU experts: `generate_ds41`, `bloomery-chat`, `bloomery-serve-ds41` | step, skewed pass, load, draft, chat and server gates; per-op reference gates (being ported from the Q8_0 mix) |
| [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | [`Q4_K_M`](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) (unsloth) | the whole model on one GPU: `generate_qwen3moe` | per-kernel gates and an end-to-end gate (greedy tokens, PPL/KLD against the reference) |
| [DeepSeek-V2-Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat) | [`Q3_K_M`](https://huggingface.co/mradermacher/DeepSeek-V2-Lite-Chat-GGUF) (mradermacher) | CPU (`bloomery-decode`) and GPU | the engine's first model; every subsystem gate |

In progress: [DeepSeek-V4-Flash-0731](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF), and the [DSpark draft](https://huggingface.co/JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF) for V4.1 speculative decoding.

## What it does

V4.1-Flash does not fit on one consumer card, so each decode step is split across the machine:

- **Experts are split between the card and the host.** Each routed layer keeps some of its experts on the GPU. The rest run on an AVX2 host tier inside the same captured CUDA graph step. A router-frequency list (`BLOOMERY_HOT_LIST`) can pick which experts stay on the card.
- **engram rows come from NVMe.** V4.1's n-gram lookup table does not fit in RAM beside the host experts, so it stays in a memory-mapped file. Each token reads 48 rows; a helper thread asks the kernel for all of them first, then copies them.
- **A skewed two-row pass.** Two positions run one layer apart in one step, so the card's dense work hides behind the host leg. It is the verification pass for speculative decoding.
- **An n-gram lookup draft** (`BLOOMERY_DRAFT=lookup`) proposes the next token from the text so far. Greedy output is the same with and without it; a gate checks that token for token.

## Status

**It runs today on the public `Q3_K_M` GGUF** (`BLOOMERY_V41_MODEL` names its first shard; the default is still a local mix of the same file with attention, shared experts and the engram tables in Q8_0 and the token embedding in BF16 — [`docs/BUILD.md`](docs/BUILD.md#the-model-file)). On the public file the step, the skewed pass, the load, the lookup draft, chat and the server pass their gates. Seven per-op reference gates still check only Q8_0 weights and are being ported; the default moves to the public file when they pass.

On either file you can run `generate_ds41` (token ids in, greedy ids out), `bloomery-chat` (text in, streamed text out, sampled or greedy) and `bloomery-serve-ds41` (llama-server's HTTP API with streaming, prompt-prefix reuse, reasoning and tool calls; one request at a time). The tokenizer is bit-identical to `llama-tokenize` on its gate's corpora.

In progress: the per-op gates on the public file; a timed run on one RTX 3090; a DSpark speculative draft; Qwen3-30B-A3B on one card.

As far as we know (2026-09-24), this is the only Rust engine that runs DeepSeek-V4.1-Flash with CPU expert offloading, with the GPU kernels also written in Rust. If you know of another one, please open an issue.

## Measured numbers

Built for Ampere GPUs (sm_86) and AVX2 CPUs. All numbers are single-stream decode on one RTX A6000 (48 GB) in a 32-core AVX2 workstation, instrumentation off, `n = 96` generated tokens, one fresh process per arm, measured under the quiet-machine protocol below.

**DeepSeek-V4.1-Flash, public `Q3_K_M` — one GPU plus CPU experts.** Placement plan (a): every layer and the head on the card, 2,668 routed experts on the card picked by the router-ranked hot list, the other 12,692 on the host. The prompt is the first 512 tokens of a prose or a code corpus.

| Prompt | Card experts | step p50 | tok/s |
|---|---:|---:|---:|
| prose 512 | 2,668 | 23.30 ms | **42.9** |
| code 512 | 2,668 | 23.72 / 23.57 ms | 42.2 / 42.4 |
| prose 512, card budget 24 GB (as on an RTX 3090) | 1,171 | 27.91 ms | 35.8 |

Source: rig-log [2026-09-24, public file on the headline prompts](https://github.com/midagedev/rig-log/blob/main/log/2026-09-24.md#public-q3km-prose-code-and-budget). llama.cpp support for V4.1 is an open pull request ([#28696](https://github.com/ggml-org/llama.cpp/pull/28696)); a same-window comparison against it is next.

**Qwen3-30B-A3B-Instruct-2507, `Q4_K_M` — the whole model on the GPU**, against mainline llama.cpp (`53ed051ce`, `llama-bench -ngl 99 -fa on`), arms alternated in one window, four rounds:

| Depth | bloomery tok/s | llama.cpp tok/s | bloomery / llama.cpp |
|---:|---:|---:|---:|
| 6 | **206.0** | 193.7 | 1.063 ± 0.004 |
| 1024 | **194.4** | 188.9 | 1.029 ± 0.002 |
| 4096 | 160.7 | **173.1** | 0.929 ± 0.006 |

bloomery is ahead at short context and behind at 4096 keys, where its attention is slower; that is being worked on. Source: rig-log [2026-09-24, Qwen3-30B-A3B](https://github.com/midagedev/rig-log/blob/main/log/2026-09-24.md#qwen3-30b-a3b-e28). A mistral.rs row is next.

Prompt processing on the same card, arms alternated in one window, three rounds (`llama-bench -p P -n 0`; mistral.rs `d5ae0f18f`, `bench --prompt-len P --gen-len 1`):

| Prompt | bloomery pp tok/s | llama.cpp | llama.cpp `-ub 4096` | mistral.rs |
|---:|---:|---:|---:|---:|
| 512 | **6,402** | 4,318 | — | 3,471 |
| 4096 | **7,490** | 4,205 | 6,904 | 1,321 |

The prompt runs in ubatches of up to 4096 tokens (a load-time size, `BLOOMERY_QWEN3_UBATCH`) through a grouped int8 tensor-core GEMM, each expert read once per ubatch, and a prefill attention kernel that stages each 64-key tile once for 64 query rows and runs both products on the tensor cores. At 4096 tokens bloomery's default is 1.085× llama.cpp with `-ub 4096 -b 4096`; the routed GEMM and the per-token glue kernels are about equal shares of what remains [derived]. Source: rig-log [2026-09-25, q3ubatch](https://github.com/midagedev/rig-log/blob/main/log/2026-09-25.md#q3ubatch-ab).

Read these numbers with their conditions:

- **No speculative decoding** in any row. A draft for V4.1 (DSpark) is being built.
- **The card is an A6000.** The 24 GB row emulates a 3090's budget on the A6000; a run on a real 3090 has not been timed.
- **The V4.1 hot list was built from routing traces of the same corpora** the prompts come from, so the prose and code rows are its favorable case. With the synthetic depth prompt, whose output collapses into a few repeating tokens, the same placement ran at 34.7 tok/s at depth 6 and 32.3 at depth 4096.

For V4.1 the host tier's memory bandwidth sets the step: 135–137 GB/s at 32 threads ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-24.md#v41-host-tier-k-rows)). The cost of each part is in [`docs/HARDWARE.md`](docs/HARDWARE.md).

## Target hardware

| | Minimum | Extended |
|---|---|---|
| GPU | RTX 3090 24 GB ×1 (sm_86) | RTX 3090 ×2 (the two-card path is not built yet) |
| CPU | AVX2, 8 DDR4 channels | same |
| RAM | 256 GB | same |
| Storage | NVMe for the engram table | same |

The development machine is a Threadripper PRO 5975WX (32 cores, 8 DDR4 channels, 264 GB) with an RTX A6000 and an RTX 3090. Details and costs: [`docs/HARDWARE.md`](docs/HARDWARE.md).

## How it is verified

- **Oracle sets against ik_llama.cpp.** Intermediate tensors are dumped from ik_llama.cpp's CPU backend for V4.1 and compared per tensor and per layer. Integer outputs (engram row ids, router top-6 ids, indexer top-k lists) must match exactly, except where two candidates are within a tie band. Float outputs must stay inside bands derived from the two engines' rounding rules.
- **Bit gates.** A captured graph replay must equal the eager run bit for bit. The skewed two-row pass must equal two single steps bit for bit, and so must a rollback. Node and launch counts are pinned at build time.
- **PPL and KLD.** Hot list, wikitext-2 at 2048 context, 4 chunks, 3090, the mix file: PPL 1.9003 (bloomery) against 1.8989 (ik_llama.cpp CPU), KLD 0.00601 ± 0.00028, same top token 97.87 % ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-24.md#hot-list-ppl-kld)).
- **A gate is shown to fail** on the defect it guards before the fix goes in. Bands are not relaxed to make a gate pass.

Timing runs only on a quiet machine, under a machine-wide lease, with a witness block around every timed region. The protocol is rig-log's [`docs/quiet-machine.md`](https://github.com/midagedev/rig-log/blob/main/docs/quiet-machine.md).

## Limits

- **The public `Q3_K_M` file is not the default yet**: seven per-op reference gates still check only Q8_0 weights (see Status).
- **sm_86 only.** Every kernel is built and gated for Ampere (`--arch sm_86`). Other architectures are not tested.
- **A pinned nightly.** The toolchain is `nightly-2026-08-28`, pinned together with a cuda-oxide git revision. The source comes from our fork of that revision, where fixes wait until upstream takes them (`THIRD_PARTY_NOTICES.md` lists them).
- **One machine.** Timed numbers come from one A6000 in one workstation. The tooling (`tools/box.sh`) assumes a Mac editor and that workstation; [`docs/BUILD.md`](docs/BUILD.md) says what to run on your own host.
- ~~**No batched prefill for V4.1.** A V4.1 prompt is fed one step per token, so a 4096-token prompt takes about 2.4–2.9 minutes before the first new token [derived: 4096 × 35–43 ms]; batched prefill is in progress.~~ **V4.1 prefill is slow.** Since 2026-09-25 a V4.1 prompt runs in batches of up to 512 positions: ~~91.2 tok/s at 512 tokens and 89.2 at 4096 on the A6000 ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-25.md#ds41batch-pp)), so a 4096-token prompt takes about 46 s~~ 109.9 tok/s at 512 tokens and 152.8 at 4096 on the A6000 by the evening of the same day (llama.cpp 104.6 / 103.8 in the morning's clean lease; ~~77.6 / 76.2 in the same lease~~ the evening lease read the reference's file pages cold after our arms — [rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-25.md#v41-prefill-resit-b)), so a 4096-token prompt takes about 27 s — each layer runs only at the positions a later reader needs, so long prompts gain most. Most of the rest is the CPU expert tier; the next steps are in progress. ~~Qwen3 prefills in ubatches of 512~~ Qwen3 prefills in ubatches of up to 4096 since 2026-09-25 (the table above). The server reuses a cached prompt prefix, so a follow-up turn pays only for its new tokens.

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
- ik_llama.cpp [#2507](https://github.com/ikawrakow/ik_llama.cpp/pull/2507) (V4.1 index keys from the pre-RoPE latent) and [#2522](https://github.com/ikawrakow/ik_llama.cpp/pull/2522) (DSpark draft loading): open, draft.
- cuda-oxide [#1314](https://github.com/NVlabs/cuda-oxide/pull/1314): a constant-folder crash on mixed-width shifts. Merged 2026-09-23.

Eight more of our fixes are merged in ik_llama.cpp; the full list is [our ik_llama.cpp PRs](https://github.com/ikawrakow/ik_llama.cpp/pulls?q=is%3Apr+author%3Amidagedev) and [our cuda-oxide PRs](https://github.com/NVlabs/cuda-oxide/pulls?q=is%3Apr+author%3Amidagedev). Toolchain issues not filed yet are in [`docs/upstream/nvlabs-ledger.md`](docs/upstream/nvlabs-ledger.md).

## References and credits

[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and ggml are the numerical reference. The kernels were written by reading ggml's k-quant code and ik_llama.cpp's `iqk_mul_mat` kernels, the tokenizer follows the reference's `llama_tokenize`, and accuracy is defined against their output. Where code or an algorithm was taken, the source comment points at the original file. [exllamav3](https://github.com/turboderp-org/exllamav3) and [mistral.rs](https://github.com/EricLBuehler/mistral.rs) are design references, cited where used (for example `crates/gpu/src/route_core.rs`). All four are MIT. The GPU kernels compile with [cuda-oxide](https://github.com/NVlabs/cuda-oxide) (Apache-2.0).

AI assistants helped write the code and the documentation in this repository. Every number here was measured on the machine by the runners in `tools/ref/` and the GPU gate runner.

## License

MIT, see [`LICENSE`](LICENSE). `crates/oxide-ice-unroll`, a compiler-bug reproducer kept out of the workspace, carries NVIDIA's Apache-2.0 headers. Third-party notices, including the ggml authors' MIT notice for the generated tokenizer table, are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
