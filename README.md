# bloomery

An LLM inference engine for one workstation, written to learn what the machine can actually do. Rust host, CUDA kernels written in Rust ([cuda-oxide](https://github.com/NVlabs/cuda-oxide), later [cutile-rs](https://github.com/NVlabs/cutile-rs)), AVX2 kernels for the CPU expert tier. The target model is DeepSeek-V4.1-Flash; the stepping stone everything is built and gated on is DeepSeek-V2-Lite-Chat Q3_K_M (8.1 GB, MLA attention, 64-expert MoE).

The machine: Threadripper PRO 5975WX (32 cores), 256 GB DDR4, RTX A6000 48 GB, RTX 3090 24 GB. Development and every GPU number below are on the 3090.

Every claim in this repository is a measurement, and the measurements live in [rig-log](https://github.com/midagedev/rig-log) with the command lines that produced them. Stages and pass criteria are in [`docs/plan.md`](docs/plan.md); the GPU design and its decisions in [`docs/gpu-design.md`](docs/gpu-design.md). Both are written in Korean.

## What runs today

**CPU path.** A full V2-Lite forward pass with a KV cache on a 32-thread pool. The quantized dot products are fused int8 AVX2 kernels for Q3_K, Q4_K, Q5_0, Q5_1 and Q6_K, written against ik_llama.cpp's `iqk_mul_mat` kernels as the reference: the activation encoders are byte-identical to ik's, the kernels agree to 1 ULP on the same rows, and the engine's logits are bit-identical across thread counts. Decode at 8 tokens of context: 37.0 tok/s, against 82.4 tok/s for ik_llama.cpp measured in the same lease (2026-09-20). The remaining gap is orchestration, not kernels — the saturation study is in [`docs/RESULTS-mul35-saturation.md`](docs/RESULTS-mul35-saturation.md).

**GPU path (in progress).** Kernels for every weight type the model uses (Q3_K, Q4_K, Q6_K, Q5_0, Q5_1, Q8_0, F32 gemv; a q8_1 activation quantizer; rms_norm, rope, embedding, residual; MLA latent attention with an in-place KV append; the top-k router; a fused MoE step that reads expert ids from a device buffer so a captured graph replays under new routing). Block 0 of the model runs as one captured CUDA graph, checked tap by tap against ik's CUDA intermediate tensors. There is no end-to-end GPU forward pass yet; the plan's next rows are the MoE layers, the output head, and a first tok/s against ik on the same card.

## How it is checked

Each stage has a `just gate-*` recipe that exits non-zero on failure. The oracle is ik_llama.cpp: intermediate tensors dumped from its CPU backend for the CPU path and from its CUDA backend for the GPU path, compared per tensor with bands that were pinned from measurement and are not relaxed. Threshold changes carry a dated note with the derivation, and every gate has been shown to fail on the defect it guards before the fix went in.

Timing runs only on a quiet machine under a lease file, with the machine state recorded next to each number. The protocol is in rig-log's [`docs/quiet-machine.md`](https://github.com/midagedev/rig-log/blob/main/docs/quiet-machine.md).

## Layout

```
crates/gguf        GGUF reader and dequantizers (bit-exact against ggml for the six types used)
crates/model       the model: attention, MoE, forward pass, KV cache, profiling
crates/qdot        fused AVX2 int8 kernels and their encoders
crates/threads     the resident worker pool
crates/gpu         the CUDA Rust kernels, the device model, graph capture
crates/gpu-gates   GPU gate binaries and the reference dump loader
docs/              plan, design decisions, result ledgers, research notes
tools/             box runner, gate runner, timing runner, reference dumpers
```

`crates/q3k-cpu`, `q3k-gemv`, `gpu-spike` and `oxide-ice-unroll` are the spikes and reproducers that preceded the crates above; they are kept because their numbers are cited.

## References

The kernels were written by reading ggml's k-quant code: the Q3_K block geometry, the Q8 activation quantization and the AVX2 integer dot-product chains come from there, and accuracy is still defined against ggml's output. Where an algorithm was taken, the source comment points at the original file and line — the usual form for a Rust port, as in [candle](https://github.com/huggingface/candle)'s `k_quants.rs`. Baseline speeds are measured with [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) on the same machine. Both are MIT.

Defects found in the toolchain along the way are sent upstream; the ledger is [`docs/upstream/nvlabs-ledger.md`](docs/upstream/nvlabs-ledger.md).

## License

MIT, see `LICENSE`.
