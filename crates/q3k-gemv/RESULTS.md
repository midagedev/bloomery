# Stage 0 round 3 (MUL-8) — M>1 restructure: 128-value q8_1 blocks, int-chained dp4a

Same-card back-to-back measurement, RTX 3090 (sm_86), one `tools/ref/measure.sh`
invocation: ggml mmvq (ik_llama.cpp CUDA backend) then this crate, 20 warm-up +
200 timed iterations per shape, fresh `x` quantization inside every timed
iteration for both engines. Weight tensor: `blk.1.ffn_gate_exps.weight`
(DeepSeek-V2-Lite-Chat Q3_K_M, K=2048; `stack` = 90,112 rows, `expert0` =
1,408 rows). Accuracy gate unchanged: max rel err <= 1e-2 (q8_1-activation
design; ggml's own mmvq error on these shapes is 3.9-5.2e-3).

## Round goal and gates

Take `stack` M=8 from 0.83x to **>= 0.9x ggml** while `stack` M=1 stays
**>= 1.0x ggml**, both in the same `measure.sh` run. Round contract: up to
four optimisation steps after the first working change; stop at the first
step that clears both conditions.

## The change (one step, both gates cleared)

Round 2 left M=8 at 0.83x with a diagnosed cause: every 4-weight field of a
gemv lane paired with its own 32-value q8_1 block, so each column paid
4 dp4a + 4 int multiplies + **4 d8 loads + 4 I2F + 4 FMUL + 4 FMA** per lane
step. ggml's `vec_dot_q3_K_q8_1_impl_mmvq` instead folds each dp4a into one
f32 FMA and applies the super-block scale once.

The fix is the spec's lever (a), and it is a *geometry* change, not a math
change: `q3k_quantize_q8_1` now emits **128-value q8_1 blocks — one per Q3_K
half super-block**, which is exactly the span a gemv lane's four fields cover
in one step. `d = amax/127` and `q = round(x/d)` per value are unchanged; the
quantize kernel gets simpler (one warp per block, each lane quantizes its four
consecutive values into one packed u32 word — the two `shuffle_down`s of the
32-value geometry disappear, and the grid drops from 64 to 16 blocks per
column).

In `q3k_gemv`, the lane/weight mapping and the whole weight-side decode are
untouched; only the per-column body changes: the four dp4a results are
multiplied by their 6-bit sub-block scales and summed in int, then ONE f32
FMA applies the shared q8_1 block scale and the super-block scale:

```
a  = dp4a(vi0, q[uw], 0) * sc0 + dp4a(vi1, q[uw+8], 0) * sc1
   + dp4a(vi2, q[uw+16], 0) * sc2 + dp4a(vi3, q[uw+24], 0) * sc3;
f  += (a as f32) * (d8[d8b] * drow);      // one per column
```

|a| <= 4 * (4*4*127) * 31 < 2^18, so the int chain cannot overflow, and the
i32 -> f32 conversion is exact. Per column per lane step this is 5 loads +
11 ALU ops (4 q + 1 d8 loads; 4 dp4a + 4 IMAD + 1 I2F + 1 FMUL + 1 FMA)
where round 2 paid 8 loads + 20 ALU. M=1 shares the same column-0 body, so
the M=1 path only shed work — no trade-off between the two gates.

## Final numbers (canonical run, 2026-09-19T00:02:20Z)

| engine | shape | N | M | us | GB/s | max rel err | rust/ggml (GB/s) |
|---|---|---|---|---|---|---|---|
| ggml mmvq | expert0 | 1408 | 1 | 10.44 | 118.69 | 4.361e-03 | — |
| mulle rust | expert0 | 1408 | 1 | 8.76 | **141.38** | 4.821e-3 | **1.191** |
| ggml mmvq | expert0 | 1408 | 8 | 16.23 | 76.32 | 4.429e-03 | — |
| mulle rust | expert0 | 1408 | 8 | 17.39 | 71.26 | 3.844e-3 | 0.934 |
| ggml mmvq | stack | 90112 | 1 | 238.89 | 331.94 | 4.119e-03 | — |
| mulle rust | stack | 90112 | 1 | 208.75 | **379.88** | 3.495e-3 | **1.144** |
| ggml mmvq | stack | 90112 | 8 | 472.65 | 167.78 | 3.928e-03 | — |
| mulle rust | stack | 90112 | 8 | 445.55 | **177.98** | 4.558e-3 | **1.061** |

**Gates: `stack` M=8 = 1.061x ggml (>= 0.9x, PASS with 18% margin) and
`stack` M=1 = 1.144x ggml (>= 1.0x, PASS with 14% margin), same run,
`PASSED: all 4 shapes within 1e-2`.** Reproduced three times end-to-end;
the rust numbers were stable to +-0.1 GB/s across runs (M=8: 177.88 /
177.97 / 177.98; M=1: 379.85 / 379.87 / 379.88) and ggml's stack numbers
varied 331.9-333.1 / 167.4-167.8. The gate cleared at the first working
change, so the round stopped there: 0 of the 4 follow-up optimisation steps
were needed, and levers (b: d8 in registers/shuffles beyond the 4->1 load
cut this change already makes), (c: column-major q / vector q loads) and
(d: rows per warp > 1 at M=8) were deliberately not tried.

## Step table (stack GB/s; ratios from the same runs)

| step | stack M=1 | stack M=8 | kept? |
|---|---|---|---|
| baseline — round-2 `main`: 32-value q8_1 blocks, per-field d8 load + FMA | 348.32 (1.046x) | 139.32 (0.832x) | — |
| step 1 — 128-value q8_1 blocks (half-super-block aligned), four dp4a chained in int behind one FMA per column | 379.88 (1.144x) | 177.98 (1.061x) | ✔ |
| final (= step 1; canonical run above) | 379.88 (1.144x) | 177.98 (1.061x) | ✔ |

Baseline row: own baseline run 2026-09-18T23:55Z (ggml 332.86 / 167.44 in the
same invocation; matches the round-2 official 348.29 / 139.13).

## ptxas facts per step (`ptxas -v -arch=sm_86` on the `cargo oxide inspect` PTX)

- **Baseline (round 2, from RESULTS-r2-arm-glm.md):** `q3k_gemv` 60 registers,
  0 bytes stack frame, 0 spill stores / 0 spill loads; PTX per loop body:
  73 `ld.global`, 32 `dp4a`, per-field float scaling (32 FMA / 32 FMUL / 32
  I2F at M=8). `q3k_quantize_q8_1`: 24 registers, 0 spills.
- **Step 1 (this round, same command):** `q3k_gemv` **56 registers, 0 bytes
  stack frame, 0 spill stores / 0 spill loads**; PTX per loop body:
  **48 `ld.global`** (8 weight-side + 8 columns x (4 q + 1 d8)),
  **32 `dp4a`** (unchanged — the cut is in everything around them),
  **24 `mad.lo` (integer), 8 `fma.rn`, 8 `mul.f32`, 8 `cvt.rn`** — the float
  work of the column loop dropped 4x with dp4a count flat.
  `q3k_quantize_q8_1`: 30 registers, 0 spills (4 values per lane instead of
  1, shuffles gone; 16 blocks per column instead of 64).
- 56 regs x 256 threads/block -> still 4 blocks/SM = 32 warps/SM on sm_86
  (67% occupancy, same as round 2's 60-reg build).

`.target sm_86` (line 16 of the `cargo oxide inspect --arch sm_86` PTX dump,
`.version 7.1`):

```
.target sm_86
```

## Accuracy note

The 128-value block is a coarser grid (amax over 128 uniform samples instead
of 32), so worst-case dots move slightly: expert0_m1 4.316 -> 4.821e-3,
stack_m8 3.868 -> 4.558e-3, while stack_m1 landed on 3.495e-3 (worst-case
placement shifts, it is a max over 90k dots). All four shapes stay inside the
1e-2 design gate and ggml's own 3.9-5.2e-3 band; the decode itself is
unchanged and was verified weight-for-weight in round 1 at 1e-7.

## Witnesses (canonical run, verbatim)

All four blocks show the 3090 `compute-apps` list **empty** (header only) —
no other process held the measured GPU. The A6000 (index 1) was idle
throughout this run. loadavg ~0.4, /proc/pressure/io some-avg10 = 0.00 for
every block.

```
--- witness pre-ref 2026-09-19T00:02:20Z ---
index, name, memory.used [MiB], utilization.gpu [%], power.draw [W]
0, NVIDIA GeForce RTX 3090, 1 MiB, 0 %, 29.64 W
1, NVIDIA RTX A6000, 1 MiB, 0 %, 19.13 W
compute-apps-3090:
pid, used_gpu_memory [MiB]
loadavg: 0.40 0.56 0.79 2/1262 2611412
pressure-io avg10: some avg10=0.00 avg60=0.00 avg300=0.00 total=1052031911
```

```
--- witness post-ref 2026-09-19T00:02:22Z ---
index, name, memory.used [MiB], utilization.gpu [%], power.draw [W]
0, NVIDIA GeForce RTX 3090, 1 MiB, 98 %, 76.41 W
1, NVIDIA RTX A6000, 1 MiB, 0 %, 19.69 W
compute-apps-3090:
pid, used_gpu_memory [MiB]
loadavg: 0.40 0.56 0.79 1/1264 2611467
pressure-io avg10: some avg10=0.00 avg60=0.00 avg300=0.00 total=1052039129
```

```
--- witness pre-rust 2026-09-19T00:02:22Z ---
index, name, memory.used [MiB], utilization.gpu [%], power.draw [W]
0, NVIDIA GeForce RTX 3090, 1 MiB, 98 %, 76.41 W
1, NVIDIA RTX A6000, 1 MiB, 0 %, 21.95 W
compute-apps-3090:
pid, used_gpu_memory [MiB]
loadavg: 0.40 0.56 0.79 1/1264 2611483
pressure-io avg10: some avg10=0.00 avg60=0.00 avg300=0.00 total=1052039129
```

```
--- witness post-rust 2026-09-19T00:02:25Z ---
index, name, memory.used [MiB], utilization.gpu [%], power.draw [W]
0, NVIDIA GeForce RTX 3090, 1 MiB, 38 %, 160.72 W
1, NVIDIA RTX A6000, 1 MiB, 0 %, 20.98 W
compute-apps-3090:
pid, used_gpu_memory [MiB]
loadavg: 0.52 0.59 0.79 3/1262 2611657
pressure-io avg10: some avg10=0.00 avg60=0.00 avg300=0.00 total=1052039444
```

Rust engine output of the same run (verbatim):

```
shape expert0_m1 N=  1408 M=1 weight_bytes=  1239040 us=     8.76 GB/s=  141.38 max_rel_err=4.821e-3
shape expert0_m8 N=  1408 M=8 weight_bytes=  1239040 us=    17.39 GB/s=   71.26 max_rel_err=3.844e-3
shape stack_m1   N= 90112 M=1 weight_bytes= 79298560 us=   208.75 GB/s=  379.88 max_rel_err=3.495e-3
shape stack_m8   N= 90112 M=8 weight_bytes= 79298560 us=   445.55 GB/s=  177.98 max_rel_err=4.558e-3
PASSED: all 4 shapes within 1e-2 (q8_1 activation design)
```

## Notes

- The gate closed at step 1, so no "what I would try next" was pursued. If
  the M=8 gate had still been open, the next levers in order would have been
  (c) column-major q layout or 16-byte q loads (4 load instructions -> 1 per
  column) and (d) two rows per warp at M=8 to amortize the weight-side SWAR
  decode the way ggml's `rows_per_cuda_block = 2` does.
- `expert0` remains report-only (L2-resident weights): M=1 1.19x, M=8 0.93x.
- Harness protocol untouched: same 20+200 timing, same four shapes, same
  `shape ... GB/s= ... max_rel_err=...` lines; `tools/ref/measure.sh` ran
  unchanged end-to-end for every number above.
