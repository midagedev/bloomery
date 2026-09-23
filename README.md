# bloomery

An LLM inference engine for one workstation, written to learn what the machine can actually do. Rust host, CUDA kernels written in Rust ([cuda-oxide](https://github.com/NVlabs/cuda-oxide), later [cutile-rs](https://github.com/NVlabs/cutile-rs)), AVX2 kernels for the CPU expert tier. The target model is DeepSeek-V4.1-Flash; the stepping stone everything was first built and gated on is DeepSeek-V2-Lite-Chat Q3_K_M (8.1 GB, MLA attention, 64-expert MoE).

The machine: Threadripper PRO 5975WX (32 cores), 256 GB DDR4, RTX A6000 48 GB, RTX 3090 24 GB. Since 2026-09-22 every timed GPU number is taken on the A6000 (the 3090 dropped off the bus twice under load that day); the 3090 builds and runs the gates. Numbers from the two cards never share a table.

Every claim in this repository is a measurement, and the measurements live in [rig-log](https://github.com/midagedev/rig-log) with the command lines that produced them. Stages and pass criteria are in [`docs/plan.md`](docs/plan.md); the GPU design and its decisions in [`docs/gpu-design.md`](docs/gpu-design.md). Both are written in Korean.

## What runs today

**V2-Lite on the GPU.** The whole decode step runs on one card as one captured CUDA graph: gemv kernels for every weight type the model uses (Q3_K, Q4_K, Q6_K, Q5_0, Q5_1, Q8_0, F32), a q8_1 activation quantizer, rms_norm, rope, embedding, MLA latent attention on tensor cores with an in-place KV append, the top-k router, and a fused MoE step that reads expert ids from a device buffer so the captured graph replays under new routing. The end-to-end gate runs the whole chain — 27 layers, the output head and the argmax — against ik_llama.cpp's CUDA greedy decode on 33 prompts × 32 steps. Decode against ik_llama.cpp on the A6000 (300 W), both engines alternating in one lease, three rounds, 96 tokens each (2026-09-22, [rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-22-o-the-depth-table-on-the-new-default-and-the-reversal-is-gone.md)):

| Context depth | bloomery | ik_llama.cpp | Ratio |
|---:|---:|---:|---:|
| 6 | 229.5 tok/s | 205.5 tok/s | 1.12× |
| 1024 | 222.2 tok/s | 194.0 tok/s | 1.15× |
| 4096 | 199.6 tok/s | 179.0 tok/s | 1.12× |

A prompt still goes through the decode graph one token at a time; the prefill GEMM path is not written yet. Part of each layer's experts can run on the CPU tier inside the captured step (`BLOOMERY_HYBRID_NL`), a rehearsal for V4.1's host experts. It is gated against the all-card run, and its timing waits for a quiet-machine measurement.

**V2-Lite on the CPU.** A full forward pass with a KV cache on a 32-thread pool. The quantized dot products are fused int8 AVX2 kernels for Q3_K, Q4_K, Q5_0, Q5_1 and Q6_K, written against ik_llama.cpp's `iqk_mul_mat` kernels as the reference: the activation encoders are byte-identical to ik's, the kernels agree to 1 ULP on the same rows, and the engine's logits are bit-identical across thread counts. On one core, the four with a counterpart in ik are 2–14 % slower than ik's own (Q3_K has none to compare with; 2026-09-23). Decode against ik_llama.cpp's fastest flag set, the two engines alternating in one lease (2026-09-21): at short context (6–100 tokens) 86.0 vs 84.1 tok/s, ours ahead in 6 of 6 rounds; at 1024 tokens of context 71.3 vs 77.9; at 4096, 45.7 vs 64.4. Short-context decode is bandwidth-bound on this host (the saturation study is [`docs/RESULTS-mul35-saturation.md`](docs/RESULTS-mul35-saturation.md)); the depth gap is the attention key scan, which reads the shared MLA key rows once per head, and is the open CPU item.

**DeepSeek-V4.1-Flash (in progress).** The reference is ik_llama.cpp's V4.1 support, which is our port ([ik_llama.cpp#2455](https://github.com/ikawrakow/ik_llama.cpp/pull/2455), merged 2026-09-21); tensors dumped from its CPU backend are the oracle. V4.1's own operations are GPU kernels in `crates/gpu-deepseek41`, each gated against those dumps: rope, the router and experts with the SwiGLU clamp, the engram key norm and gate, the hyper-connections, the block-diagonal `wo_a`, attention over the prefix and over indexer-selected rows with sinks, the KV compressor and the index keys. The indexer's scoring and top-k are not on the GPU yet, and the step is not assembled: this engine has not produced a V4.1 token. What is measured so far are the legs of one token:

- The GPU matrix-vector products of one decode token, as one captured graph on the A6000 with synthetic weights and every routed expert on the card (a ceiling, not a step): 24.66 ms, then 19.005 ms after four kernel rounds (2026-09-23).
- The CPU expert leg on the real file: 129.9 GB/s at 16 threads and no higher at 32, which is 31.1 ms per token with six host experts in every routed layer (4.04 GB; 2026-09-23).
- engram, the 195 GiB lookup table that stays on NVMe: 48 scattered 272-byte rows per token take 4.69 ms (median, cold) as demand page faults and 0.31 ms when each row's read-ahead is issued a token early (2026-09-22).

## How it is checked

Each stage has a `just gate-*` recipe that exits non-zero on failure. The oracle is ik_llama.cpp: intermediate tensors dumped from its CPU backend for the CPU path and for V4.1, and from its CUDA backend for the V2-Lite GPU path, compared per tensor. Bands are derived from the two engines' rounding rules or pinned from measurement, and they are not relaxed: a threshold change carries a dated note with its derivation, and every gate has been shown to fail on the defect it guards before the fix went in.

A round opens with a prediction — the value and its band, derived from the cost and error models in [`docs/plan.md`](docs/plan.md) — and the measurement checks it. When the two disagree, the wrong term of the model is the finding.

Timing runs only on a quiet machine under a lease file, with the machine state recorded next to each number. The protocol is in rig-log's [`docs/quiet-machine.md`](https://github.com/midagedev/rig-log/blob/main/docs/quiet-machine.md).

## Layout

```
crates/gguf            GGUF reader (split files too) and dequantizers (bit-exact against ggml for the six V2-Lite types)
crates/model           the model: attention, MoE and the host expert tier, forward pass, KV cache, profiling; V4.1's hyperparameters and tensor names
crates/qdot            fused AVX2 int8 kernels and their encoders
crates/threads         the resident worker pool
crates/engram          V4.1's NVMe-resident lookup table: the IO path, the row hash, the row cache
crates/gpu             the CUDA Rust kernels, the V2-Lite device model, graph capture, the hybrid MoE boundary
crates/gpu-deepseek41  V4.1's own GPU kernels (entry points prefixed ds41_)
crates/gpu-gates       GPU gate binaries, benches and the reference dump loader
docs/                  plan, design decisions, result ledgers, research notes
tools/                 box runner, gate runners, timing runners, reference dumpers
```

`crates/q3k-cpu`, `q3k-gemv`, `gpu-spike` and `oxide-ice-unroll` are the spikes and reproducers that preceded the crates above; they are kept because their numbers are cited.

## References

The kernels were written by reading ggml's k-quant code: the Q3_K block geometry, the Q8 activation quantization and the AVX2 integer dot-product chains come from there, and accuracy is still defined against ggml's output. Where an algorithm was taken, the source comment points at the original file and line — the usual form for a Rust port, as in [candle](https://github.com/huggingface/candle)'s `k_quants.rs`. Baseline speeds are measured with [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) on the same machine. Both are MIT.

## Upstream

What this work needed from its reference and its toolchain went upstream; each fix carries a case that fails before the patch:

- ik_llama.cpp [#2455](https://github.com/ikawrakow/ik_llama.cpp/pull/2455) — DeepSeek-V4.1 support (the oracle above), merged 2026-09-21
- ik_llama.cpp [#2493](https://github.com/ikawrakow/ik_llama.cpp/pull/2493) — the derived MLA weights allocated with the backend's padded size (V2-Lite prompt processing on CUDA turned to garbage once a ubatch held 9 or more tokens), merged 2026-09-21
- ik_llama.cpp [#2501](https://github.com/ikawrakow/ik_llama.cpp/pull/2501) — an element count instead of bytes for the flash-attention fixup pool, merged 2026-09-22
- ik_llama.cpp [#2508](https://github.com/ikawrakow/ik_llama.cpp/pull/2508) — `-no-fidx` to turn off the fused indexer top-k, merged 2026-09-23
- ik_llama.cpp [#2507](https://github.com/ikawrakow/ik_llama.cpp/pull/2507) — V4.1 index keys from the pre-RoPE compressed latent, as the reference model does; open — the maintainer measured a higher perplexity at 4096 context, and the cause is being reproduced
- cuda-oxide [#1314](https://github.com/NVlabs/cuda-oxide/pull/1314) — the constant folder crashed on a shift whose amount has a different integer width (`u32 << usize` after `#[unroll]`); open, rebased and extended with tests by the maintainer

Toolchain candidates not filed yet are in [`docs/upstream/nvlabs-ledger.md`](docs/upstream/nvlabs-ledger.md); the full list, with the ones from before this repository, is rig-log's [`docs/upstream-contributions.md`](https://github.com/midagedev/rig-log/blob/main/docs/upstream-contributions.md).

## License

MIT, see `LICENSE`.
