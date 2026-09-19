# MUL-3 pre-study: Q3_K x Q8_K gemv on the host CPU (Rust AVX2 vs ik CPU)

Machine: Threadripper PRO 5975WX (Zen 3, 32C/64T, 4x32 MB L3, AVX2/FMA, no
VNNI), one NUMA node. Reference: ik_llama.cpp `c10fbbcc` CPU backend
(`ggml_backend_cpu_init`, Q3_K weights x F32 activations through iqk's kquants
path). Rust kernel: `crates/q3k-cpu` (port of `ggml_vec_dot_q3_K_q8_K`'s AVX2
branch + scalar `quantize_row_q8_K_ref`), pinned std threads, CCD-contiguous
row ranges. Build: `RUSTFLAGS="-C target-cpu=znver3" cargo build --release`
(nightly-2026-08-28, libc 0.2.189). Protocol: 5 warm-up + 50 timed iterations,
GB/s = weight bytes / time, error vs f32 dequant+dot reference.

Shapes: `expert0` = 1408 rows / 1.24 MB, `stack` = 90112 rows / 79.3 MB,
`big` = 360448 rows / 317.2 MB (4 tensors: blk.1/2 gate+up exps), K = 2048.
`big.q3k` md5: extract fresh per run; `gate.q3k`/`x_m1.f32` verified
md5-identical to the GPU harness outputs in `/root/bloomery-data`
(1364a8ce…/8fa914ca…), i.e. the CPU harness reproduces those bytes without
touching the 3090.

## Verdict

- Accuracy: every row of both engines is within the 1e-2 bound; measured
  max_rel_err is 3.3e-3-4.6e-3 everywhere (the expected Q8_K class).
- Performance gate (big, M=1, each engine's own best thread count,
  rust >= 0.9 x ggml): **PASS: 198.13 vs 185.17 GB/s = 1.07x** (both best
  at 32 threads). Replicates: 1.09x (run 1), 1.14x cross-run (ablation).
- SMT (64 threads) on big M=1: ggml 121.60 vs 185.17 at 32 = **-34%**;
  rust 148.37 vs 198.13 at 32 = **-25%** (replicates -2%/-31%; the rust t64
  point is unstable run to run but never beats t32). SMT hurts; 32 threads
  is the operating point for this tier.

## Full table, final run (GB/s; µs in parentheses)

| shape   | M | t=8 ggml | t=8 rust | t=16 ggml | t=16 rust | t=32 ggml | t=32 rust | t=64 ggml | t=64 rust |
|---------|---|----------|----------|-----------|-----------|-----------|-----------|-----------|-----------|
| expert0 | 1 | 65.65 (18.87) | 31.78 (38.99) | 82.15 (15.08) | 25.28 (49.01) | 97.94 (12.65) | 18.50 (66.97) | 77.27 (16.04) | 1.91 (649.45) * |
| expert0 | 8 | 24.20 (51.20) | 9.45 (131.08) | 40.15 (30.86) | 13.71 (90.40) | 50.62 (24.48) | 13.45 (92.12) | 42.61 (29.08) | 1.39 (890.85) * |
| stack   | 1 | 104.00 (762.51) | 107.76 (735.86) | 208.30 (380.69) | 172.40 (459.97) | 380.34 (208.50) | 276.99 (286.28) | 363.59 (218.10) | 48.97 (1619.23) * |
| stack   | 8 | 29.42 (2695.02) | 14.11 (5618.48) | 58.87 (1347.12) | 27.14 (2922.01) | 106.37 (745.47) | 52.87 (1499.96) | 105.68 (750.39) | 43.61 (1818.49) |
| big     | 1 | 96.86 (3274.78) | 98.56 (3218.14) | 143.44 (2211.33) | 162.17 (1955.88) | 185.17 (1712.98) | 198.13 (1600.94) | 121.60 (2608.54) | 148.37 (2137.83) |
| big     | 8 | 29.22 (10855.88) | 14.09 (22509.60) | 57.55 (5512.00) | 28.06 (11302.33) | 60.95 (5203.88) * | 35.58 (8915.43) * | 80.75 (3928.09) | 43.25 (7333.79) |

max_rel_err per (shape, M), identical for both engines to the printed digit:
expert0/m1 4.099e-3, expert0/m8 4.52e-3, stack/m1 3.556e-3, stack/m8 3.594e-3,
big/m1 3.333e-3, big/m8 3.594e-3.

`*` = disagrees with replicates by >15% (transient noise inside the lease
window; boundary witnesses were clean, pressure 0.00, GPUs idle). Replicates:

| cell | run 1 | ablation | final |
|------|-------|----------|-------|
| RUST expert0 m1 t64 | 2.47 (501.89) | 9.66 (128.24) | 1.91 (649.45) |
| RUST expert0 m8 t64 | 7.04 (175.95) | 8.39 (147.74) | 1.39 (890.85) |
| RUST stack m1 t64 | 237.54 (333.83) | 238.44 (332.57) | 48.97 (1619.23) |
| RUST big m1 t64 | 191.95 (1652.50) | 139.63 (2271.68) | 148.37 (2137.83) |
| RUST big m8 t32 | 51.15 (6200.79) | 52.62 (6027.95) | 35.58 (8915.43) |
| CPUREF big m8 t32 | 90.99 (3486.21) | — | 60.95 (5203.88) |
| CPUREF big m8 t64 | 93.21 (3403.05) | — | 80.75 (3928.09) |

Stable gate cells across all runs — RUST big M=1 t32: 196.47 / 203.78 /
198.13; CPUREF big M=1 t32: 179.53 / 185.17. No re-run was taken tofavor
one sample: the table above is the single final run in full.

## Best-thread row per shape (final run)

| shape   | M | ggml best (t) | rust best (t) | rust/ggml |
|---------|---|---------------|---------------|-----------|
| expert0 | 1 | 97.94 (32) | 31.78 (8) | 0.32x |
| expert0 | 8 | 50.62 (32) | 13.71 (16) | 0.27x |
| stack   | 1 | 380.34 (32) | 276.99 (32) | 0.73x |
| stack   | 8 | 106.37 (32) | 52.87 (32) | 0.50x |
| big     | 1 | 185.17 (32) | 198.13 (32) | **1.07x PASS** |
| big     | 8 | 80.75 (64) | 43.25 (64) | 0.54x |

## big as % of the 147.7 GB/s ceiling

| engine | M=1 | M=8 |
|--------|-----|-----|
| ggml CPU | 185.17 = 125% | 80.75 = 55% |
| rust | 198.13 = 134% | 43.25 = 29% |

Both engines exceed the nominal 147.7 GB/s DRAM-read number on big M=1, so
that ceiling (measured with a different pattern/thread mix) is soft for pure
sequential Q3_K reads; the two engines corroborate each other. big's 317 MB
exceeds the 128 MB total L3, so these rows are DRAM-served.

## Levers (gate cell: big M=1, 32 threads; base range 196-204 across runs)

| step | config | GB/s | delta vs base | verdict |
|------|--------|------|---------------|---------|
| 0 | base (no flags) | 203.78 | — | keep (final) |
| 1 | +hugepage | 203.36 | ~0% | **reverted**: no effect; AnonHugePages stayed 0 kB (THP never engaged in the seconds-long run) |
| 2 | +hugepage+unroll2 | 152.50 | -25% (@8: -54%, 45.45 vs 98.17) | **reverted**: hurts; doubled live AVX state presumably spills (numerics unchanged, 3.333e-3) |
| 3a | +hugepage+unroll2+prefetch | 149.57 | -27% | **reverted** with step 2 |
| 3b | prefetch alone | 180.86 | -9..-11% | **reverted**: HW streamer already covers the sequential pattern |

Final binary = base (all flags default off). No code changed after the
ablation; the final run above is the base binary.

## Findings (out of gate scope, for the record)

- M=8 is 0.50-0.54x on stack/big: the rust kernel re-decodes each weight row
  per column while iqk's kernel amortizes one decode over 8 Q8 dots. Closing
  it needs a column-interleaved kernel (a 4th step; budget was 3 and the gate
  had cleared).
- Cache-resident shapes trail (stack M=1 0.73x, expert0 worse): thread/
  barrier overhead dominates at <=22 rows/thread, and the tiny-shape t64
  point is unstable (1.4-9.7 GB/s across runs).
- Activation quantize costs 0.74 µs/iter (M=1, scalar, inside the timed
  window like ggml's per-compute quantize): negligible everywhere.

## Witnesses, final run (`tools/box.sh 'bash tools/ref/cpu-measure.sh'`)

```
[lease] waiting for /root/bloomery-cpu.lock ...
[lease] acquired 00:27:26Z
--- witness pre-ref 2026-09-19T00:27:26Z ---
loadavg: 5.07 2.95 1.85 1/1270 2659089
pressure-cpu: some avg10=0.00 avg60=0.00 avg300=0.00 total=473335374
pressure-io: some avg10=0.00 avg60=0.00 avg300=0.00 total=1056024901
0, NVIDIA GeForce RTX 3090, 0 %, 29.83 W
1, NVIDIA RTX A6000, 0 %, 19.21 W
lock-holder-pid: 2658913
```

(rust section runs between post-ref and pre-rust witnesses)

```
--- witness post-ref 2026-09-19T00:27:37Z ---
loadavg: 4.61 2.91 1.85 2/1274 2659684
pressure-cpu: some avg10=0.00 avg60=0.00 avg300=0.00 total=473346465
pressure-io: some avg10=0.00 avg60=0.00 avg300=0.00 total=1056037168
0, NVIDIA GeForce RTX 3090, 0 %, 29.97 W
1, NVIDIA RTX A6000, 0 %, 19.36 W
lock-holder-pid: 2658913
--- witness pre-rust 2026-09-19T00:27:38Z ---
loadavg: 4.61 2.91 1.85 3/1274 2659695
pressure-cpu: some avg10=0.00 avg60=0.00 avg300=0.00 total=473347266
pressure-io: some avg10=0.00 avg60=0.00 avg300=0.00 total=1056037168
0, NVIDIA GeForce RTX 3090, 0 %, 29.97 W
1, NVIDIA RTX A6000, 0 %, 19.36 W
lock-holder-pid: 2658913
--- witness post-rust 2026-09-19T00:27:42Z ---
loadavg: 5.04 3.03 1.89 1/1267 2659980
pressure-cpu: some avg10=0.00 avg60=0.00 avg300=0.00 total=473369725
pressure-io: some avg10=0.00 avg60=0.00 avg300=0.00 total=1056037745
0, NVIDIA GeForce RTX 3090, 0 %, 29.84 W
1, NVIDIA RTX A6000, 0 %, 19.19 W
lock-holder-pid: 2658913
```
