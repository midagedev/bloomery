# IQ1_S / IQ2_XXS quantization on the card — the derivation

`bloomery_gpu::iq_quant` runs ggml's `quantize_row_iq1_s_impl` and `quantize_row_iq2_xxs_impl` (the rules are
`gguf::iq_quant`'s) on the card, so the K3.1 pipeline's routed experts (`tools/ref/k31-quant.sh`, map `tight`) can be
quantized there and handed to `llama-quantize` already in their target type, which it copies (`llama-quant.cpp`,
`quantize = cur_type != new_type`). Every number below is [derived] unless it says measured; this page was written
before the first timed run, and the probe's numbers are set beside it afterwards.

## Why the bytes can be ggml's

- The reference library is `libggml-base.so` of the quantization tree (78feec062), compiled `-O3 -std=gnu11` with no
  `-march` (`build/ggml/src/CMakeFiles/ggml-base.dir/flags.make`): baseline x86-64, so no FMA contraction and no F16C
  (`objdump -d` shows 0 `vfmadd*`/`vfmsub*`/`vfnmadd*`, 0 `vcvtps2ph`, 148 `sqrtss`). Every `*`, `+`, `/` and `sqrtf` rounds
  once; the f16 store is ggml's software round-to-nearest-even. The card spells every operation as its `.rn`
  instruction (`mul_rn_f32`, `add_rn_f32`, `div_rn_f32`, `sqrt_rn_f32`), in C's evaluation order, and rounds `d` with
  `gguf::quant::f32_to_f16_bits`.
- IQ1_S sorts each 32-value sub-block with `qsort` on the value alone. glibc 2.39 (the box) sorts with a merge sort, so
  equal values keep their index order (measured: 0 unstable of 100,000 random tie-heavy trials). The card sorts by
  insertion with a strict comparison: the same order. An MXFP4 sub-block (one E8M0 scale, 16 codes) always has ties.
- The neighbour tables are `iq2xs_init_impl`'s, derived from the dequant grids `gguf::iq_tables` holds; the derived
  2-bit codes equal ggml's `kgrid_2bit_256` and `kgrid_1bit_2048` (checked once against the source).
- MXFP4 → f32 is `kvalues_mxfp4[q] · 2^(e−128)`: a small integer times a power of two, exact.

## The card's rate

Per 32-value sub-block, one thread, serial in ggml's order (SASS instructions, counted from the rule's loops):

| term | IQ1_S | IQ2_XXS |
|---|---|---|
| weights (32 × mul, add, sqrt.rn ≈ 6, mul) | ~300 | ~300 (+ 32 sqrt for `waux`) |
| `sigma2`: 256-term serial sum (each of the 8 threads of a super-block repeats it) | ~770 | ~770 |
| sort (insertion, 32 values, ~250 compare-moves on local memory) | ~3,000 | — |
| split search: 561 (i1, i2) pairs × 2 shifts × ~21 + 6 differences + 2 local loads | ~28,000 | — |
| `make_qp_quants` (9 trial scales × 32, 5 refinement passes) | — | ~5,000 |
| 13 trial scales × 4 groups × (code word, map lookup, neighbour scan when off grid) + 13 × 32 sums | — | ~30,000 |
| neighbour scans (IQ1_S: ~69 % of 3-level words are off its 2048-point grid; lists of 3 shells) | ~7,000 | in the row above |
| total | **~39,000 ≈ 1,220 / weight** | **~37,000 ≈ 1,150 / weight** |

The card issues 4 warp-instructions a clock per SM: RTX 3090 82 SMs × 128 lanes × ~1.7 GHz = 1.8e13 thread-instructions/s,
A6000 84 SMs ≈ 1.8e13. The search is issue-bound, not memory-bound: a sub-block reads 32 values (17 B of MXFP4) and
writes 16 B. Dependent chains of 4-cycle FP ops, L1-resident local arrays and divergent neighbour scans give it an
assumed ~40 % of issue: **~5.7e9 weights/s, band 2e9 – 1.4e10**. One K3.1 expert tensor (3584 × 3072 × 896 =
9.865e9 weights) is then **~1.7 s of card (0.7 – 5 s)**. The CPU path (`k31-quant.sh`'s model: 40 effective threads ×
5.4e6 weights/s, the rate measured on one thread) takes 46 s for the same tensor.

## The resource timeline, per expert tensor

| resource | bytes | rate | s |
|---|---|---|---|
| NVMe read (MXFP4, 17/32 B a weight) | 5.24 GB | 1.6 GB/s (page-fault floor on /models, measured by k3trace) | 3.3 |
| host: bytes → words copy, then H2D (pageable) | 5.24 GB | ~3 GB/s copy, ~10 GB/s H2D | 1.7 + 0.5 |
| card: widen + search + finish | 9.865e9 weights | 5.7e9 /s | 1.7 |
| D2H records (68 B a super-block) + host pack | 2.62 GB → 1.93 GB | ~10 GB/s, ~4 GB/s | 0.3 + 0.5 |
| NVMe write (IQ1_S, 50 B a super-block) | 1.93 GB | 1.5 GB/s | 1.3 |

The probe runs them in series ~~(~9 s a tensor)~~ (22 s measured, below). ~~The set's bin should overlap them: then the
read (3.3 s) is the bound, and the card is under its shadow.~~ `iq_quant_set` reads the next batch while the card works,
and the card (14 s a stack, measured) is the bound. The host copy into words is a term to remove: upload the mapped
bytes directly (a byte-addressed decode), or pinned staging.

## Measured beside the derivation

`just iq-quant-probe` on shard 31 of the K3.1 k3shape set (A6000, unleased, beside a 64-thread `llama-imatrix` that held
the host's cores and memory: load 66, kswapd awake), one stack of 9.865e9 weights, synthetic importance, batches of 32
experts; the bytes of the experts it checked equal ggml's (IQ1_S experts 0, 447, 895 of `blk.91.ffn_gate_exps`;
IQ2_XXS 0 and 895 of `blk.92.ffn_down_exps`):

| term (per stack) | derived | IQ1_S measured | IQ2_XXS measured |
|---|---|---|---|
| read (page-in copy of the mapped tensor) | 3.3 s | 3.4 – 4.4 s | 3.5 – 4.4 s |
| host: bytes → words | 1.7 s | 5.9 s, 1.9 s after the bulk conversion | 2.1 s |
| upload | 0.5 s | 0.33 s | 0.35 s |
| widen + search + finish (kernels) | 1.7 s | 0.13 + **9.05 – 9.12** + 0.06 s | 0.13 + **11.8** + 0.07 s |
| download + pack | 0.8 s | 0.2 – 1.2 + 0.6 – 1.6 s | 1.1 + 1.6 s |
| the quantize call | 4.7 s | **14.0 s** | **17.4 s** |

The search kernel runs at 1.09e9 weights/s (IQ1_S) and 8.3e8 (IQ2_XXS), under the band's floor (2e9): the term that was
wrong is the issue efficiency, assumed 40 %, measured ~7 % of the 1.8e13 thread-instructions/s (if the instruction count
holds). Its mechanism (local-memory arrays of 32 per thread, registers, divergent neighbour scans) is not measured yet:
`ptxas -v` and an `ncu` pass over `iq_quant_subs` are the next reading. The host terms are contended in this sitting
(the imatrix run), so they are an upper bound. The card, not the read, bounds a stack now.

## The full K3.1 quantize with `--gpu-quant`

With the measured call (`k31-quant.sh`'s `GPU_STACK_S` = 14.3 s, the 254:22 weighting): the card set takes 276 × 14.3 s
+ the set written (0.66 TB at 1.5 GB/s, in series after each batch) ≈ 4,390 s; `llama-quantize` over it (the trunk's
5.7e10 weights at the IQ rate, an upper bound, + the set read + the output written) ≈ 1,060 s: **≈ 5,450 s (91 min)
against the CPU path's ≈ 13,970 s (3.9 h)**, both from `k31-quant.sh quantize --pred-s`. The set needs ~0.66 TB of room
besides the output; `k31-quant.sh` refuses a `--gpu-set` drive without it.

## The derivation's own projection (before the measurement; wrong in its card term, corrected above)

- Intermediate set: 276 expert tensors × ~3.3 s (read-bound, overlapped) ≈ 910 s, plus the trunk copied through
  (~0.11 TB read and written, ~140 s); out ≈ 0.53 TB experts + 0.11 TB trunk = ~0.64 TB.
- `llama-quantize` over it: read ~0.64 TB (~400 s at 1.6 GB/s), the trunk's k-quants on the CPU (~5.5e10 weights, small),
  write the 0.58 TB output (387 s).
- ~~**≈ 1,900 s (~32 min) overlapped, ≈ 2,700 s (~45 min) in series, against the CPU path's ~3.9 h.**~~ The card
  term was 5× short (above): ≈ 5,450 s. The intermediate
  set needs ~0.64 TB of room besides the output; nvme1 has 783 GB free today, so both cannot sit there at once.
