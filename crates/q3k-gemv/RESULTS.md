# MUL-8: stack M=8 from 0.83x to parity — RESULTS

Same-card back-to-back measurement, RTX 3090 (sm_86), one `tools/ref/measure.sh`
invocation: ggml mmvq (ik_llama.cpp CUDA backend) then this crate, 20 warm-up +
200 timed iterations per shape, fresh `x` quantization inside every timed
iteration for both engines. Weight tensor: `blk.1.ffn_gate_exps.weight`
(DeepSeek-V2-Lite-Chat Q3_K_M, K=2048, 64x1408 rows). Starting point: the
round-2 winner on `main` (q8_1 + dp4a, warp-per-row, funnel loads).

## Final numbers (official run, 2026-09-18T23:56Z)

| engine | shape | N | M | us | GB/s | max rel err | rust/ggml (GB/s) |
|---|---|---|---|---|---|---|---|
| ggml mmvq | expert0 | 1408 | 1 | 10.45 | 118.61 | 4.361e-03 | — |
| mulle rust | expert0 | 1408 | 1 | 7.48 | 165.59 | 4.316e-3 | 1.396 |
| ggml mmvq | expert0 | 1408 | 8 | 16.13 | 76.80 | 4.429e-03 | — |
| mulle rust | expert0 | 1408 | 8 | 18.16 | 68.24 | 4.432e-3 | 0.889 |
| ggml mmvq | stack | 90112 | 1 | 238.93 | 331.89 | 4.119e-03 | — |
| mulle rust | stack | 90112 | 1 | 151.78 | 522.47 | 4.158e-3 | 1.574 |
| ggml mmvq | stack | 90112 | 8 | 474.05 | 167.28 | 3.928e-03 | — |
| mulle rust | stack | 90112 | 8 | 480.39 | 165.07 | 3.868e-3 | 0.987 |

**Gates, same run: stack M=8 0.987x (>= 0.9x) PASS, stack M=1 1.574x (>= 1.0x)
PASS.** Accuracy gate (<= 1e-2, q8_1 design): PASS on all four shapes, same
~4e-3 band as ggml itself. `expert0` is L2-resident, reported not tuned.

## Optimisation steps (stack GB/s)

| step | stack M=1 | stack M=8 | kept? |
|---|---|---|---|
| baseline — round-2 winner (`main`) | 348.26 (1.046x) | 139.92 (0.832x) | — |
| step 1 — hoist `drow` out of the per-field scale | 354.20 | 142.86 | yes |
| step 2 — unchecked raw loads (`get_unchecked`) | 522.47 (1.574x) | 165.07 (0.987x) | yes, final |

Ratios for baseline use its own same-run ggml (333.09 / 168.25); step 1 was a
rust-only run (no ggml in that invocation), so GB/s only. No reverted steps;
tuning stopped at step 2 per the round contract (first step clearing both
gates wins).

Why each step, grounded in `ptxas -v` (`sm_86`, PTX from
`cargo oxide inspect q3k_gemv --arch sm_86`):

- **Step 1 (drow hoist).** The baseline computed
  `(dp4a*sc) as f32 * (d8*drow)` per field: 8 FMULs per column per lane step
  (4 for `d8*drow`, 4 for the product) plus a 4-deep `f0 +=` dependency chain.
  ggml's `vec_dot_q3_K_q8_1_impl_mmvq` instead sums the four `d8`-scaled fields
  and multiplies by the super-block scale once (`return d3 * sumf`). The step
  copies that shape: per column, `p = sum4((dp4a*sc) as f32 * d8); f += p *
  drow` — 5 FMULs instead of 8, one accumulator update, better ILP.
  `q3k_gemv` went 60 -> **55 registers** (0 spills), 32 `dp4a`, 73 `ld.global`
  unchanged. Measured +1.7% (M=1) / +2.1% (M=8). Numerics unchanged to the
  printed precision (4.158e-3 / 3.868e-3 before and after).
- **Step 2 (unchecked loads) — the actual gap.** ncu on one `stack` M=8 launch
  each (same work, same 79.3 MB DRAM read): ggml executed 4.58B
  thread-instructions, this kernel 7.78B (+70%) at 73.7% SM throughput. The
  excess was Rust slice bounds checks on every global load (73 `ld.global` in
  PTX, each with a compare-and-trap prologue ggml's raw pointers never pay).
  A `--unchecked-indexing` probe build confirmed it before any source change
  (M=1 556, M=8 168 GB/s). The committed step converts every device dirty load
  to `get_unchecked` inside `unsafe` blocks with SAFETY bounds arguments —
  an allowed `unsafe` use ("raw loads") that works under the stock
  `measure.sh` flags: 1 `x` load in the quantizer, 8 weight words, 32 `q` + 32
  `d8` words per lane step, grouped so each `unsafe` block carries one
  comment. `q3k_gemv` went 55 -> **40 registers** (0 spills), quantizer 24 ->
  22; 40 regs x 256 threads unlocks 6 blocks/SM (48 warps, up from 32).
  Measured +47% (M=1) / +16% (M=8), accuracy bit-identical to baseline.

`.target sm_86` (from `cargo oxide inspect q3k_gemv --arch sm_86`, line 5 of
the PTX dump):

```
.target sm_86
```

## Witnesses (official run, verbatim)

Every block shows the 3090 `compute-apps` list **empty** (header only) — no
other process held the measured GPU during either timed section.

```
--- witness pre-ref 2026-09-18T23:56:55Z ---
index, name, memory.used [MiB], utilization.gpu [%], power.draw [W]
0, NVIDIA GeForce RTX 3090, 1 MiB, 0 %, 29.45 W
1, NVIDIA RTX A6000, 1 MiB, 0 %, 19.04 W
compute-apps-3090:
pid, used_gpu_memory [MiB]
loadavg: 0.62 0.85 0.94 1/1251 2603557
pressure-io avg10: some avg10=0.00 avg60=0.00 avg300=0.00 total=1051024079
```

```
--- witness post-ref 2026-09-18T23:56:57Z ---
index, name, memory.used [MiB], utilization.gpu [%], power.draw [W]
0, NVIDIA GeForce RTX 3090, 1 MiB, 98 %, 64.06 W
1, NVIDIA RTX A6000, 1 MiB, 0 %, 20.17 W
compute-apps-3090:
pid, used_gpu_memory [MiB]
loadavg: 0.62 0.85 0.94 1/1253 2603618
pressure-io avg10: some avg10=0.00 avg60=0.00 avg300=0.00 total=1051034843
```

```
--- witness pre-rust 2026-09-18T23:56:57Z ---
index, name, memory.used [MiB], utilization.gpu [%], power.draw [W]
0, NVIDIA GeForce RTX 3090, 1 MiB, 98 %, 64.06 W
1, NVIDIA RTX A6000, 1 MiB, 0 %, 20.17 W
compute-apps-3090:
pid, used_gpu_memory [MiB]
loadavg: 0.62 0.85 0.94 1/1253 2603634
pressure-io avg10: some avg10=0.00 avg60=0.00 avg300=0.00 total=1051034843
```

```
--- witness post-rust 2026-09-18T23:57:00Z ---
index, name, memory.used [MiB], utilization.gpu [%], power.draw [W]
0, NVIDIA GeForce RTX 3090, 1 MiB, 3 %, 119.46 W
1, NVIDIA RTX A6000, 1 MiB, 0 %, 22.52 W
compute-apps-3090:
pid, used_gpu_memory [MiB]
loadavg: 0.73 0.87 0.95 2/1253 2603815
pressure-io avg10: some avg10=0.00 avg60=0.00 avg300=0.00 total=1051036016
```

Rust engine output of the same run (verbatim):

```
shape expert0_m1 N=  1408 M=1 weight_bytes=  1239040 us=     7.48 GB/s= 165.59 max_rel_err=4.316e-3
shape expert0_m8 N=  1408 M=8 weight_bytes=  1239040 us=    18.16 GB/s=  68.24 max_rel_err=4.432e-3
shape stack_m1   N= 90112 M=1 weight_bytes= 79298560 us=   151.78 GB/s= 522.47 max_rel_err=4.158e-3
shape stack_m8   N= 90112 M=8 weight_bytes= 79298560 us=   480.39 GB/s= 165.07 max_rel_err=3.868e-3
PASSED: all 4 shapes within 1e-2 (q8_1 activation design)
```

## Notes

- The spec's QR3_K=2 premise does not match the vendored ggml: `ggml-common.h`
  defines QR3_K=4 (QI3_K=16), and `vec_dot_q3_K_q8_1_impl_mmvq` scales each of
  its 4 fields by its own `d8` — there is no int-accumulation across shared
  scales to copy. The M=8 gap was not the scale count but bounds-check
  instructions plus the extra per-field `drow` multiply, both fixed above
  without touching the quantization design (accuracy unchanged).
- Host protocol unchanged (20 warm-up + 200 timed, same four shapes, same
  output line format). No C/CUDA C in the rust crate; `unsafe` only for raw
  loads and lane-0 stores, each with a SAFETY comment.
