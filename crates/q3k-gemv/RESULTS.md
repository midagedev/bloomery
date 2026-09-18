# Stage 0 round 2 — mmvq-shaped Q3_K gemv redesign: RESULTS

Same-card back-to-back measurement, RTX 3090 (sm_86), one `tools/ref/measure.sh`
invocation: ggml mmvq (ik_llama.cpp CUDA backend) then this crate, 20 warm-up +
200 timed iterations per shape, fresh `x` quantization inside every timed
iteration for both engines. Weight tensor: `blk.1.ffn_gate_exps.weight`
(DeepSeek-V2-Lite-Chat Q3_K_M, K=2048, 64×1408 rows).

## Design chosen (accuracy contract)

**q8_1 activation + hardware dp4a** — the same design as ggml's mmvq. Gate:
max rel err ≤ **1e-2**, stated up front. The activation is quantized once per
call to q8_1 (32-value blocks, d = amax/127, int8 q), and the inner product is
`dp4a.s32.s32` over packed 4×int8 words. On these shapes ggml's own mmvq
measures 3.9–4.4e-3 and this kernel lands in the same band (3.9–4.4e-3), i.e.
the residual error is the activation quantization itself, not the decode: a CPU
replica of the exact kernel arithmetic against `y_ref` reproduces the reference
to 5e-3 with quantized `x` and to 1e-7 when fed float `x`.

Kernel shape (both kernels in `src/main.rs`):

- `q3k_quantize_q8_1` — one 32-thread block per 32-value q8_1 block; warp max
  for `d`, lanes pack their int8 byte into u32 words with two `shuffle_down`s.
- `q3k_gemv` — one warp per row. Lanes 0–15 own the even super-blocks, lanes
  16–31 the odd ones (`sbp = 2·it + half`), so each lane's 16-weight qs window
  is exactly one u32 and the warp's weight loads are contiguous runs. Odd
  super-blocks sit 2 mod 4, so every window (qs, hmask, scales, d) is
  assembled from two aligned u32 loads with a 16-bit funnel select. Each
  (qs word, field j) quad is dequantized in registers with SWAR byte
  arithmetic (`vi = vil − 4·(1−hbit)` as signed bytes), dotted with one u32 of
  q8_1 `x` via `dp4a_s32`, scaled by the 6-bit sub-block scale and
  `d8·d` in f32, and reduced with a warp shuffle sum. M ≤ 8 columns per
  weight read (guarded scalar accumulators — a runtime column index would put
  the array in local memory).

## Final numbers (official run, 2026-09-18T23:27Z)

| engine | shape | N | M | µs | GB/s | max rel err | rust/ggml (GB/s) |
|---|---|---|---|---|---|---|---|
| ggml mmvq | expert0 | 1408 | 1 | 10.43 | 118.82 | 4.361e-03 | — |
| mulle rust | expert0 | 1408 | 1 | 9.00 | **137.70** | 4.316e-3 | **1.159** |
| ggml mmvq | expert0 | 1408 | 8 | 16.11 | 76.93 | 4.429e-03 | — |
| mulle rust | expert0 | 1408 | 8 | 21.76 | 56.95 | 4.432e-3 | 0.740 |
| ggml mmvq | stack | 90112 | 1 | 238.64 | 332.30 | 4.119e-03 | — |
| mulle rust | stack | 90112 | 1 | 227.68 | **348.29** | 4.158e-3 | **1.048** |
| ggml mmvq | stack | 90112 | 8 | 473.33 | 167.53 | 3.928e-03 | — |
| mulle rust | stack | 90112 | 8 | 569.96 | 139.13 | 3.868e-3 | 0.830 |

**Gate: stack M=1 ≥ 0.9× ggml → 1.048×, PASS** (reproduced twice; an earlier
run the same minute measured 348.27 vs 332.89 = 1.046×). M=8 is 0.83× — above
the 0.5× "good round" bar, not at parity. `expert0` is L2-resident
(1.2 MB weights) and reported, not tuned: M=1 1.16×, M=8 0.74×.

## Optimisation steps (stack GB/s; ggml ratios from the same runs)

| step | stack M=1 | stack M=8 | kept? |
|---|---|---|---|
| baseline — round 1, f32 activations (`main`) | 140.17 (0.42×) | 32.68 (0.19×) | — |
| redesign — q8_1 + dp4a, warp-per-row, funnel loads | 348.27 (1.046×) | 139.29 (0.829×) | ✔ |
| step 1 — `#[unroll]` on the super-block loop | 226.28 (0.68×) | 99.74 (0.59×) | ✘ reverted |
| final (= redesign; canonical run above) | 348.29 (1.048×) | 139.13 (0.830×) | ✔ |

Why each step, grounded in `ptxas -v` (`sm_86`, `cargo oxide inspect` PTX):

- **Redesign.** dp4a packs 4 int8 multiply-adds into one instruction and cuts
  activation traffic 4× (u32 vs f32). Final `q3k_gemv`: **60 registers,
  0 bytes stack frame, 0 spill stores / 0 spill loads**, 32 `dp4a`, 73
  `ld.global`, 0 `ld.shared`; `q3k_quantize_q8_1`: 24 registers, 0 spills.
  60 regs × 256 threads/block → 4 blocks/SM = 32 warps/SM (67% occupancy) —
  enough latency hiding for the bandwidth-bound M=1 shape.
- **Step 1 (unroll), reverted.** Full unroll of the 4-iteration super-block
  loop moved `q3k_gemv` from 60 → **122 registers**, 73 → 289 `ld.global`,
  32 → 128 `dp4a` (still 0 spills). 122 regs × 256 threads → 2 blocks/SM =
  16 warps/SM (33% occupancy): half the memory-level parallelism on a shape
  whose wall is weight streaming, measured as −35% (M=1) / −28% (M=8).
  The rolled loop with its per-iteration parity selects is the better shape;
  the step was measured and reverted, and the final run is the rolled build.

`.target sm_86` (from `cargo oxide inspect q3k_gemv --arch sm_86`, line 22 of
the PTX dump):

```
.target sm_86
```

## Witnesses (official run, verbatim)

Every block shows the 3090 `compute-apps` list **empty** (header only) — no
other process held the measured GPU during either timed section. The A6000
(index 1) was serving its own workload throughout and is not on the measured
device.

```
--- witness pre-ref 2026-09-18T23:27:41Z ---
index, name, memory.used [MiB], utilization.gpu [%], power.draw [W]
0, NVIDIA GeForce RTX 3090, 1 MiB, 0 %, 30.58 W
1, NVIDIA RTX A6000, 38742 MiB, 100 %, 295.61 W
compute-apps-3090:
pid, used_gpu_memory [MiB]
loadavg: 1.65 1.57 1.63 3/1366 2563683
pressure-io avg10: some avg10=0.00 avg60=0.00 avg300=0.00 total=1046524210
```

```
--- witness pre-rust 2026-09-18T23:27:43Z ---
index, name, memory.used [MiB], utilization.gpu [%], power.draw [W]
0, NVIDIA GeForce RTX 3090, 1 MiB, 57 %, 46.20 W
1, NVIDIA RTX A6000, 38742 MiB, 100 %, 294.44 W
compute-apps-3090:
pid, used_gpu_memory [MiB]
loadavg: 1.68 1.58 1.63 2/1368 2563765
pressure-io avg10: some avg10=0.00 avg60=0.00 avg300=0.00 total=1046538246
```

```
--- witness post-rust 2026-09-18T23:27:46Z ---
index, name, memory.used [MiB], utilization.gpu [%], power.draw [W]
0, NVIDIA GeForce RTX 3090, 1 MiB, 70 %, 140.62 W
1, NVIDIA RTX A6000, 38742 MiB, 100 %, 298.09 W
compute-apps-3090:
pid, used_gpu_memory [MiB]
loadavg: 1.68 1.58 1.63 2/1368 2563917
pressure-io avg10: some avg10=0.00 avg60=0.00 avg300=0.00 total=1046545492
```

Rust engine output of the same run (verbatim):

```
shape expert0_m1 N=  1408 M=1 weight_bytes=  1239040 us=     9.00 GB/s=  137.70 max_rel_err=4.316e-3
shape expert0_m8 N=  1408 M=8 weight_bytes=  1239040 us=    21.76 GB/s=   56.95 max_rel_err=4.432e-3
shape stack_m1   N= 90112 M=1 weight_bytes= 79298560 us=   227.68 GB/s=  348.29 max_rel_err=4.158e-3
shape stack_m8   N= 90112 M=8 weight_bytes= 79298560 us=   569.96 GB/s=  139.13 max_rel_err=3.868e-3
PASSED: all 4 shapes within 1e-2 (q8_1 activation design)
```

## Notes

- `expert0` is not a tuning target this round (L2-resident weights); its M=1
  number exceeding ggml's is a side effect of the same design, reported only.
- The M=8 gap (0.83×) is structural in this mapping: ggml's vecdot merges the
  four sub-block dots of a 32-weight window into one scaled f32 per 32
  weights, while the lane/field mapping here needs one dp4a+scale-FMA per
  4-weight field because each field pairs with its own q8_1 block scale.
  Closing it means changing the q8_1 quantization granularity, which is an
  accuracy-design change, not a tuning step — left for the next round.
