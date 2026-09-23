# bloomery

A Rust inference engine for DeepSeek-V4.1-Flash on one workstation. The host code is Rust, the GPU kernels are CUDA written in Rust ([cuda-oxide](https://github.com/NVlabs/cuda-oxide)), and the CPU expert kernels are AVX2 Rust.

V4.1-Flash does not fit on one consumer card, so the engine splits every decode step across the machine:

- **MoE experts are split between the card and the host.** Routed layers keep a set of their experts on the GPU. The rest run on an AVX2 host tier inside the same captured CUDA graph step. Which experts stay on the card can come from a router-frequency list (`BLOOMERY_HOT_LIST`).
- **engram rows come from NVMe.** V4.1's lookup table is about 195 GiB, so it stays in a memory-mapped file. Each token reads 48 rows of 272 bytes. A helper thread advises every row to the kernel first, then copies them.
- **A skewed two-row pass.** Two positions run one layer apart in one step, so the card's dense work hides behind the host leg. It is the verification pass for speculative decoding.
- **An n-gram lookup draft** (`BLOOMERY_DRAFT=lookup`) proposes the next token from the text seen so far. Greedy output is the same with and without it; a gate checks that token for token.

## Status

**Numbers first. The text CLI and the server are in progress.** Today the engine takes token ids and returns token ids (`generate_ds41`, see [`docs/BUILD.md`](docs/BUILD.md)). The tokenizer, a chat binary with sampling and streaming, and an OpenAI-compatible endpoint are the next milestones.

As far as we know (2026-09-24), this is the only Rust engine that runs DeepSeek-V4.1-Flash with CPU expert offloading, with the GPU kernels also written in Rust. If you know of another one, please open an issue.

## Measured numbers

Decode on the RTX A6000 (48 GB), placement plan (a), the router-ranked hot list on the card, per-step instrumentation off, `n = 96`, one fresh process per arm. ik_llama.cpp ran in the same lease window with its closest placement (`llama-bench -gp 6,96`, `-ncmoe 34 --defer-experts`: six whole layers on the card, about the same host work per token, not the same expert set). Source: rig-log [2026-09-24, hot-list placement](https://github.com/midagedev/rig-log/blob/main/log/2026-09-24.md#hot-list-placement-and-real-text-prompt).

| Depth | bloomery step p50 | bloomery tok/s (p50) | ik_llama.cpp tok/s |
|---:|---:|---:|---:|
| 6 | 35.16 / 35.40 ms | 28.35 | 19.40 / 20.08 |
| 4096 | 34.33 / 34.38 ms | 29.10 | not measured in this window |

Two rounds each, shown as `round 1 / round 2`. The ratio at depth 6 is about 1.4× [derived from the table].

Read these numbers with their conditions:

- **The prompt is synthetic.** The depth prompt is a fixed pseudo-random id sequence, and its output collapses into a few repeating tokens. With a 4096-token prose prompt the step was 42.59 ms against 38.41 ms for the synthetic prompt (prefix placement, instrumentation on, same window). Expect real text to be about 10 % slower [derived from those two]. Of the extra 4.2 ms, 0.9 ms is the host leg; the rest is cold engram rows (major faults).
- **The card is an A6000, not a 3090.** A single-3090 run (placement `gate`) has not been timed yet. The target profile and what it is expected to cost are in [`docs/HARDWARE.md`](docs/HARDWARE.md).
- **No draft is on.** The skewed two-row pass costs 1.21–1.29 plain steps (A6000, depth 6), so a draft wins once it is accepted more than 21–29 % of the time [derived from that ratio]. The served lookup draft has not been timed yet.

The host tier's bandwidth sets the step. On the same day the host expert leg read 135–137 GB/s at 32 threads: 24.89 ms for 3.37 GB per token in a bench with five host experts per routed layer ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-24.md#v41-host-tier-k-rows)).

## Target hardware

| | Minimum | Extended |
|---|---|---|
| GPU | RTX 3090 24 GB ×1 (sm_86) | RTX 3090 ×2 (the two-card path is not built yet) |
| CPU | AVX2, 8 DDR4 channels | same |
| RAM | 256 GB | same |
| Storage | NVMe for the engram table | same |

The development machine is a Threadripper PRO 5975WX (32 cores, 8 DDR4 channels, 264 GB) with an RTX A6000 and an RTX 3090. Details and costs are in [`docs/HARDWARE.md`](docs/HARDWARE.md).

## How it is verified

- **Oracle sets against ik_llama.cpp.** Intermediate tensors are dumped from ik_llama.cpp's CPU backend for V4.1 and compared per tensor and per layer. Integer outputs (engram row ids, router top-6 ids, indexer top-k lists) must match exactly, except where two candidates are within a tie band. Float outputs must stay inside bands derived from the two engines' rounding rules.
- **Bit gates.** A captured graph replay must equal the eager run bit for bit. The skewed two-row pass must equal two single steps bit for bit, and so must a rollback. Node and launch counts are pinned at build time.
- **PPL and KLD.** With the hot list, wikitext-2 at 2048 context, 4 chunks, 3090: PPL 1.9003 (bloomery) against 1.8989 (ik_llama.cpp CPU), KLD 0.00601 ± 0.00028, same top token 97.87 % ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-24.md#hot-list-ppl-kld)).
- **A gate is shown to fail** on the defect it guards before the fix goes in. Bands are not relaxed to make a gate pass.

Timing runs only on a quiet machine, under a machine-wide lease, with a witness block around every timed region. The protocol is rig-log's [`docs/quiet-machine.md`](https://github.com/midagedev/rig-log/blob/main/docs/quiet-machine.md).

## Limits

- **sm_86 only.** Every kernel is built and gated for Ampere (`--arch sm_86`). Other architectures are not tested.
- **A pinned nightly.** The toolchain is `nightly-2026-08-28`, pinned together with a cuda-oxide git revision. It moves only when the cuda-oxide pin moves.
- **Measured on an A6000.** Timed numbers come from one card in one machine. The placement plans name the cards they run on (`A6000`, `3090`).
- **One machine.** The tooling (`tools/box.sh`) assumes a Mac editor and one Linux workstation. [`docs/BUILD.md`](docs/BUILD.md) says what to run on your own host instead.
- **Greedy only, one token per step.** There is no batched prefill yet, so a prompt is fed one step per token.

## Build

See [`docs/BUILD.md`](docs/BUILD.md). In one line: a Linux x86-64 host with AVX2, CUDA 13.3, LLVM 21 and `cargo-oxide`, then `cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41`.

## More

- Measurements with their command lines: [rig-log](https://github.com/midagedev/rig-log), under `log/`.
- Plan, stages and cost models: [`docs/plan.md`](docs/plan.md). GPU design: [`docs/gpu-design.md`](docs/gpu-design.md). Placement: [`docs/v41-placement.md`](docs/v41-placement.md). These are written in Korean.
- The working contract for contributors and agents: [`AGENTS.md`](AGENTS.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).

The engine started on DeepSeek-V2-Lite-Chat Q3_K_M, and that path still runs and is gated. On the A6000 (300 W), V2-Lite decode ran at 229.5 tok/s against ik_llama.cpp's 205.5 at depth 6, and 199.6 against 179.0 at depth 4096 (2026-09-22, [rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-22-o-the-depth-table-on-the-new-default-and-the-reversal-is-gone.md)).

## Upstream

What this work needed from its reference and its toolchain went upstream:

- ik_llama.cpp [#2455](https://github.com/ikawrakow/ik_llama.cpp/pull/2455): DeepSeek-V4.1 support, the oracle above. Merged 2026-09-21.
- ik_llama.cpp [#2493](https://github.com/ikawrakow/ik_llama.cpp/pull/2493): padded allocation for the derived MLA weights. Merged 2026-09-21.
- ik_llama.cpp [#2501](https://github.com/ikawrakow/ik_llama.cpp/pull/2501): an element count instead of bytes for the flash-attention fixup pool. Merged 2026-09-22.
- ik_llama.cpp [#2508](https://github.com/ikawrakow/ik_llama.cpp/pull/2508): `-no-fidx` to turn off the fused indexer top-k. Merged 2026-09-23.
- ik_llama.cpp [#2507](https://github.com/ikawrakow/ik_llama.cpp/pull/2507): V4.1 index keys from the pre-RoPE compressed latent. Open.
- cuda-oxide [#1314](https://github.com/NVlabs/cuda-oxide/pull/1314): a constant-folder crash on a shift with mixed integer widths. Merged 2026-09-23.

Toolchain issues not filed yet are in [`docs/upstream/nvlabs-ledger.md`](docs/upstream/nvlabs-ledger.md).

## References and credits

The kernels were written by reading ggml's k-quant code and ik_llama.cpp's `iqk_mul_mat` kernels, and accuracy is defined against their output. Where an algorithm was taken, the source comment points at the original file. Both projects are MIT. Baseline speeds are measured with [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) on the same machine.

AI assistants helped write the code and the documentation in this repository. Every number here was measured on the machine by the runners in `tools/ref/`.

## License

MIT, see [`LICENSE`](LICENSE).
