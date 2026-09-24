# bloomery

DeepSeek-V4.1-Flash on one workstation: one GPU, an AVX2 host, and kernels written in Rust.

bloomery is an inference engine for one machine: an RTX 3090-class card, an AVX2 CPU with eight memory channels, 256 GB of RAM and an NVMe drive. The host code is Rust, the GPU kernels are CUDA written in Rust ([cuda-oxide](https://github.com/NVlabs/cuda-oxide)), and the CPU expert kernels are AVX2 Rust. It is for people with that kind of workstation who want to run a model that does not fit on their card.

## What it does

V4.1-Flash does not fit on one consumer card, so each decode step is split across the machine:

- **Experts are split between the card and the host.** Each routed layer keeps some of its experts on the GPU. The rest run on an AVX2 host tier inside the same captured CUDA graph step. A router-frequency list (`BLOOMERY_HOT_LIST`) can pick which experts stay on the card.
- **engram rows come from NVMe.** V4.1's n-gram lookup table does not fit in RAM beside the host experts, so it stays in a memory-mapped file. Each token reads 48 rows; a helper thread asks the kernel for all of them first, then copies them.
- **A skewed two-row pass.** Two positions run one layer apart in one step, so the card's dense work hides behind the host leg. It is the verification pass for speculative decoding.
- **An n-gram lookup draft** (`BLOOMERY_DRAFT=lookup`) proposes the next token from the text so far. Greedy output is the same with and without it; a gate checks that token for token.

## Status

**It runs today, but only on a local model file.** The V4.1 engine loads a local mix of the public `Q3_K_M` GGUF: attention, shared experts and the engram tables in Q8_0, the token embedding in BF16 ([`docs/BUILD.md`](docs/BUILD.md#the-model-file)). The GPU dense path reads only Q8_0, so the public `Q3_K_M` file does not load yet. Support for the public file is the current work, and it comes before a download-and-run release.

On that file you can run `generate_ds41` (token ids in, greedy ids out), `bloomery-chat` (text in, streamed text out, sampled or greedy) and `bloomery-serve-ds41` (llama-server's HTTP API with streaming, prompt-prefix reuse, reasoning and tool calls; one request at a time). The tokenizer is bit-identical to `llama-tokenize` on its gate's corpora.

In progress: the public `Q3_K_M` file; a timed run on one RTX 3090; a DSpark speculative draft; Qwen3-30B-A3B on one card.

As far as we know (2026-09-24), this is the only Rust engine that runs DeepSeek-V4.1-Flash with CPU expert offloading, with the GPU kernels also written in Rust. If you know of another one, please open an issue.

## Measured numbers

Decode on the RTX A6000 (48 GB), placement plan (a): every layer and the head on the card, 2,414 routed experts on the card picked by the router-ranked hot list, the rest on the host. Instrumentation off, `n = 96`, one fresh process per arm. **Both engines read the local Q8_0 mix above**, not the public file.

ik_llama.cpp ran in the same lease window, built from #2455 plus the draft #2507 (below), as `GGML_CUDA_NO_PINNED_WEIGHTS=1 llama-bench -ngl 999 --n-cpu-moe 34 -t 32 --defer-experts -gp 6,96`: six whole layers on the card, about the same host work per token [derived] but not the same experts. No flag sweep was run for ik.

| Depth | bloomery step p50 | bloomery tok/s (p50) | ik_llama.cpp tok/s |
|---:|---:|---:|---:|
| 6 | 35.16 / 35.40 ms | 28.35 | 19.40 / 20.08 |
| 4096 | 34.33 / 34.38 ms | 29.10 | not measured in this window |

Two rounds each, shown as `round 1 / round 2`. The ratio at depth 6 is about 1.4× [derived from the table]. Source: rig-log [2026-09-24, hot-list placement](https://github.com/midagedev/rig-log/blob/main/log/2026-09-24.md#hot-list-placement-and-real-text-prompt).

Read these numbers with their conditions:

- **No draft on either side.** ik_llama.cpp has a DSpark speculative draft for V4.1, which neither arm here ran. Ours is being built; the fair row is draft against draft.
- **The model file is the local mix.** The public file's dense tensors are Q3_K, about 40 % of the mix's dense bytes, so the card's part of the step should shrink [derived]. The host tier's routed experts are the same in both files. Public-file numbers will be measured again.
- **The prompt is synthetic.** The depth prompt is a fixed pseudo-random id sequence, and its output collapses into a few repeating tokens. With a 4096-token prose prompt the step was 42.59 ms against 38.41 ms for the synthetic prompt (instrumentation on). Expect real text to be about 10 % slower [derived from those two]; most of the difference is cold engram rows.
- **The card is an A6000, not a 3090.** A single-3090 run (placement `gate`) has not been timed yet.
- **The skewed pass has a break-even.** It costs 1.21–1.29 plain steps (A6000, depth 6), so a draft wins once more than 21–29 % of its tokens are accepted [derived from that ratio].

The host tier's memory bandwidth sets the step: 135–137 GB/s at 32 threads, 24.89 ms for 3.37 GB per token with five host experts per routed layer ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-24.md#v41-host-tier-k-rows)). The cost of each part is in [`docs/HARDWARE.md`](docs/HARDWARE.md).

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

- **The public `Q3_K_M` file does not load yet** (see Status).
- **sm_86 only.** Every kernel is built and gated for Ampere (`--arch sm_86`). Other architectures are not tested.
- **A pinned nightly.** The toolchain is `nightly-2026-08-28`, pinned together with a cuda-oxide git revision.
- **One machine.** Timed numbers come from one A6000 in one workstation. The tooling (`tools/box.sh`) assumes a Mac editor and that workstation; [`docs/BUILD.md`](docs/BUILD.md) says what to run on your own host.
- **No batched prefill.** A prompt is fed one step per token, so a 4096-token prompt takes about 2.4–2.9 minutes before the first new token [derived: 4096 × 35–43 ms]. The server reuses a cached prompt prefix, so a follow-up turn pays only for its new tokens.

## Build

See [`docs/BUILD.md`](docs/BUILD.md). In one line: a Linux x86-64 host with AVX2, CUDA 13.3, LLVM 21 and `cargo-oxide`, then `cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41` (or `--bin bloomery-chat`, `--bin bloomery-serve-ds41`).

## More

- Measurements with their command lines: [rig-log](https://github.com/midagedev/rig-log), under `log/` (Korean).
- Plan and cost models: [`docs/plan.md`](docs/plan.md) (Korean). GPU design: [`docs/gpu-design.md`](docs/gpu-design.md) (Korean). Placement: [`docs/v41-placement.md`](docs/v41-placement.md) (Korean).
- The working contract for contributors and agents: [`AGENTS.md`](AGENTS.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).

The engine started on DeepSeek-V2-Lite-Chat Q3_K_M, and that path still runs and is gated. On the A6000 (300 W), V2-Lite decode ran at 229.5 tok/s against ik_llama.cpp's 205.5 at depth 6, and 199.6 against 179.0 at depth 4096 (2026-09-22, [rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-22-o-the-depth-table-on-the-new-default-and-the-reversal-is-gone.md)).

## Upstream

What this work needed from its reference and its toolchain went upstream:

- ik_llama.cpp [#2455](https://github.com/ikawrakow/ik_llama.cpp/pull/2455): DeepSeek-V4.1 support, the oracle above. Merged 2026-09-21.
- ik_llama.cpp [#2444](https://github.com/ikawrakow/ik_llama.cpp/pull/2444): `GGML_CUDA_NO_PINNED_WEIGHTS`, which the ik command above uses. Merged 2026-09-15.
- ik_llama.cpp [#2507](https://github.com/ikawrakow/ik_llama.cpp/pull/2507) (V4.1 index keys from the pre-RoPE latent) and [#2522](https://github.com/ikawrakow/ik_llama.cpp/pull/2522) (DSpark draft loading): open, draft.
- cuda-oxide [#1314](https://github.com/NVlabs/cuda-oxide/pull/1314): a constant-folder crash on mixed-width shifts. Merged 2026-09-23.

Eight more of our fixes are merged in ik_llama.cpp; the full list is [our ik_llama.cpp PRs](https://github.com/ikawrakow/ik_llama.cpp/pulls?q=is%3Apr+author%3Amidagedev) and [our cuda-oxide PRs](https://github.com/NVlabs/cuda-oxide/pulls?q=is%3Apr+author%3Amidagedev). Toolchain issues not filed yet are in [`docs/upstream/nvlabs-ledger.md`](docs/upstream/nvlabs-ledger.md).

## References and credits

[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and ggml are the numerical reference and the baseline. The kernels were written by reading ggml's k-quant code and ik_llama.cpp's `iqk_mul_mat` kernels, the tokenizer follows the reference's `llama_tokenize`, and accuracy is defined against their output. Where code or an algorithm was taken, the source comment points at the original file. [exllamav3](https://github.com/turboderp-org/exllamav3) and [mistral.rs](https://github.com/EricLBuehler/mistral.rs) are design references, cited where used (for example `crates/gpu/src/route_core.rs`). All four are MIT. The GPU kernels compile with [cuda-oxide](https://github.com/NVlabs/cuda-oxide) (Apache-2.0).

AI assistants helped write the code and the documentation in this repository. Every number here was measured on the machine by the runners in `tools/ref/` and the GPU gate runner.

## License

MIT, see [`LICENSE`](LICENSE). `crates/oxide-ice-unroll`, a compiler-bug reproducer kept out of the workspace, carries NVIDIA's Apache-2.0 headers. Third-party notices, including the ggml authors' MIT notice for the generated tokenizer table, are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
