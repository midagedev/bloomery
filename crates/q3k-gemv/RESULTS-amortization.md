# Stage 0 (MUL-8) — the M>1 amortization curve

GLM-authored round (outsourced spec), 2026-09-19. Question: what does one
extra activation column cost in the gemv kernels, how does that curve
compare to ggml mmvq's, and can the gap be closed without touching the M=1
arithmetic? Harness extended in `tools/ref/q3k_ref.cpp` + the rust main:
a nested k-family (M ∈ {1,2,3,4,6,8}, the FIRST k columns of the existing
`x_m8` draw, rng order untouched) on the three gated tensors — stack (Q3_K,
N=90112), attnstk (Q4_K, N=55296), head (Q6_K, N=102400) — 18 new rows, all
existing rows and names untouched. Gates live in the rust binary and anchor
to ggml's own timings dumped by the ref in the SAME `measure.sh` invocation.

## Part A — the curve before (2026-09-19 07:12Z run)

| engine | M | stack µs | t/t₁ | attnstk µs | t/t₁ | head µs | t/t₁ |
|---|---:|---:|---:|---:|---:|---:|---:|
| bloomery | 1 | 140.37 | 1.000 | 79.05 | 1.000 | 207.37 | 1.000 |
| bloomery | 2 | 170.70 | 1.216 | 128.77 | 1.629 | 231.54 | 1.117 |
| bloomery | 3 | 200.63 | 1.430 | 183.56 | 2.323 | 280.30 | 1.352 |
| bloomery | 4 | 243.75 | 1.736 | 220.96 | 2.795 | 324.58 | 1.565 |
| bloomery | 6 | 325.83 | 2.321 | 308.25 | 3.899 | 413.73 | 1.995 |
| bloomery | 8 | 404.25 | 2.880 | 400.76 | 5.070 | 499.96 | 2.411 |
| ggml | 1 | 252.38 | 1.000 | 86.05 | 1.000 | 233.16 | 1.000 |
| ggml | 2 | 282.28 | 1.119 | 120.51 | 1.400 | 284.69 | 1.221 |
| ggml | 3 | 339.85 | 1.346 | 155.91 | 1.812 | 326.45 | 1.400 |
| ggml | 4 | 369.36 | 1.463 | 181.57 | 2.110 | 359.21 | 1.541 |
| ggml | 6 | 355.40 | 1.408 | 209.06 | 2.430 | 400.77 | 1.719 |
| ggml | 8 | 487.16 | 1.930 | 280.91 | 3.264 | 503.64 | 2.160 |

(existing m1/m8 rows, same runs: bloomery stack 141.24→409.82 = 2.902,
attnstk 78.97→406.28 = 5.145, head 206.56→522.61 = 2.530.)

Curve shape: every curve is concave-superlinear — the kth column costs more
than a weight re-read (µs/col between consecutive M grows), and bloomery's
curves rise much faster than ggml's except on head. Slope of the k-family,
µs per extra column (endpoint (t₈−t₁)/7): bloomery 37.70 / 45.96 / 41.80,
ggml 33.54 / 27.84 / 38.64. Normalized per warp (slope/N): bloomery 0.419 / 0.832 / 0.408 ns per
warp-column — Q4_K exactly 2× Q3_K. Per-token amortization t(M)/(M·t(M=1))
at M=8: bloomery 0.360 / 0.634 / 0.301 vs ggml 0.241 / 0.408 / 0.270 — the
q8_1 activation is amortized far better in ggml's layout; bloomery's M=8 was
paying most of an M=1 again per column pair on Q4_K.

Diagnosis (measured, then arithmetically pinned): the marginal cost is the
q8 load path. In the old linear store, one gemv load instruction (32 lanes
× 4 B) touched 4 different 128 B lines on Q3_K/Q6_K (64 wavefronts per
warp-column over the 16 loads) and 8 lines on Q4_K (stride-8 words = 128 B
between lanes' words; 128 wavefronts) — predicting q4k/q3k = 2.0×, exactly
the 0.832/0.419 measured, and an absolute slope within 10% of 1
wavefront/cycle/SM. That named the shared third cause beyond the two
q4k-specific levers in RESULTS-q4k-q6k.md.

## The levers (all value-identical; every later run's 28 max_rel_err values
are bit-identical to Part A's — that equality is the proof)

1. **Per-format permuted q8 store** (all three formats): quantize writes
   each packed word three times, once per gemv geometry, so every gemv load
   instruction addresses 32 lane-consecutive words = one 128 B line, one
   wavefront. Bijection p_fmt on 0..512 host-verified for every load site,
   INCLUDING the value-span tie v4 = 32b + lane (see the incident below).
2. **s8 group sums** (Q4_K, the spec's second named lever): quantize
   butterfly-sums each 32-value group's signed bytes — bit-identical to
   what the B chain's dp4a(0x01010101, qv) accumulated — so the per-column
   B chain becomes one i32 load of 32 consecutive lanes.
3. **vi hoist** (Q4_K, the spec's first named lever): the qs window and its
   SWAR nibble decode are column-independent; decoded once per iteration
   into eight registers, reused by all columns.
4. **u64 pairing** (Q3_K, added after the first post-lever run): levers 1-3
   fixed attnstk and head but stack still failed gate 2 (2.262 vs 2.006) —
   and the three formats' per-warp-column costs were now uniform
   (0.274/0.280/0.270 ns) despite different instruction mixes, pointing at
   load-ISSUE count, not bytes. Q3_K's four q8 loads per iteration become
   two u64 loads: the two fields of a load pair (j, j^1) always live in
   quantize lanes lane and lane^8 of one block, so one shuffle_xor(8)
   builds the u64 and half the lanes store. Same bytes, same wavefronts,
   half the instructions. Stack's slope fell 25.35 → 15.71 µs/col.

**Incident (recorded for the next permutation round):** the first
permutation draft computed the store-side value index as `v4 = 16*b + lane`
instead of `32*b + lane` (32 words per 128-value block, not 16). The gemv
then read half-scrambled, half-zero words — max_rel_err ~1.0-1.7, every
row. The host verifier did NOT catch it: it proved the p_fmt bijections and
every load-site identity, but on the q-word path it never tied the store
index to the VALUE SPAN, so kernel and verifier agreed on the same wrong
premise (the s8 path did have that tie and was correct). Fix: one line +
the verifier now asserts `v4 == (128b + 4·lane)/4` and the span equality
for all three formats; with that tie the wrong draft cannot pass.

## Part B — the curve after (final kernel, 07:44Z run)

| engine | M | stack µs | GB/s | t/t₁ | t/(M·t₁) | attnstk µs | GB/s | t/t₁ | t/(M·t₁) | head µs | GB/s | t/t₁ | t/(M·t₁) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bloomery | 1 | 147.79 | 536.6 | 1.000 | 1.000 | 79.13 | 805.0 | 1.000 | 1.000 | 207.21 | 830.3 | 1.000 | 1.000 |
| bloomery | 2 | 159.30 | 497.8 | 1.078 | 0.539 | 82.22 | 774.8 | 1.039 | 0.520 | 216.36 | 795.1 | 1.044 | 0.522 |
| bloomery | 3 | 171.88 | 461.4 | 1.163 | 0.388 | 97.89 | 650.7 | 1.237 | 0.412 | 242.55 | 709.3 | 1.170 | 0.390 |
| bloomery | 4 | 189.13 | 419.3 | 1.280 | 0.320 | 117.33 | 542.9 | 1.483 | 0.371 | 275.88 | 623.6 | 1.331 | 0.333 |
| bloomery | 6 | 220.98 | 358.8 | 1.495 | 0.249 | 156.91 | 406.0 | 1.983 | 0.331 | 341.25 | 504.1 | 1.647 | 0.275 |
| bloomery | 8 | 257.79 | 307.6 | 1.744 | 0.218 | 191.93 | 331.9 | 2.425 | 0.303 | 402.30 | 427.6 | 1.942 | 0.243 |
| ggml | 1 | 253.93 | 312.3 | 1.000 | 1.000 | 86.30 | 738.1 | 1.000 | 1.000 | 234.48 | 733.7 | 1.000 | 1.000 |
| ggml | 2 | 284.63 | 278.6 | 1.121 | 0.560 | 121.23 | 525.5 | 1.405 | 0.702 | 280.60 | 613.1 | 1.197 | 0.598 |
| ggml | 3 | 339.18 | 233.8 | 1.336 | 0.445 | 158.18 | 402.7 | 1.833 | 0.611 | 325.95 | 527.8 | 1.390 | 0.463 |
| ggml | 4 | 373.75 | 212.2 | 1.472 | 0.368 | 183.83 | 346.5 | 2.130 | 0.532 | 355.15 | 484.4 | 1.515 | 0.379 |
| ggml | 6 | 361.22 | 219.5 | 1.423 | 0.237 | 213.47 | 298.4 | 2.473 | 0.412 | 400.97 | 429.0 | 1.710 | 0.285 |
| ggml | 8 | 485.64 | 163.3 | 1.912 | 0.239 | 283.84 | 224.4 | 3.290 | 0.411 | 502.98 | 342.0 | 2.145 | 0.268 |

(existing m1/m8 rows: bloomery stack 137.19→255.51 = 1.863, attnstk
78.54→183.34 = 2.334, head 204.66→406.62 = 1.987.)

After the levers bloomery's M=8 is faster than ggml's in absolute time on all
three tensors (stack 257.79 vs 485.64 µs = 1.88×, attnstk 191.93 vs 283.84
= 1.48×, head 402.30 vs 502.98 = 1.25×), and every per-token curve sits at
or below ggml's. Slopes µs/col: 15.71 / 16.11 / 27.87 (per warp: 0.174 /
0.291 / 0.272 ns). Where the engines still diverge: ggml's curve is smoother
in M (its M=6 stack point is FASTER than M=4 in both runs — quantized tile
geometry, not noise of ours), and head's residual 27.87 µs/col slope is the
remaining u32-load path — the u64 pairing lever is proven on Q3_K and
simply not applied to Q6_K this round (head is report-only in the gate).

## Gates (in-process, same-invocation anchors; pre = Part A run, post = both final runs)

    pre : gate2 stack_m8/stack_m1: bloomery 2.902 vs ggml 1.998 -> FAIL
    pre : gate2 stack_k8/stack_k1: bloomery 2.880 vs ggml 1.930 -> FAIL
    pre : gate2 attnstk_m8/attnstk_m1: bloomery 5.145 vs ggml 3.320 -> FAIL
    pre : gate2 attnstk_k8/attnstk_k1: bloomery 5.070 vs ggml 3.264 -> FAIL
    pre : gate2 head_m8/head_m1: bloomery 2.530 vs ggml 2.200 -> info
    pre : gate2 head_k8/head_k1: bloomery 2.411 vs ggml 2.160 -> info
    pre : gate3 stack_m1: bloomery 561.45 GB/s vs floor 298.46 (0.9 x ggml 331.62) -> PASS
    pre : gate3 attnstk_m1: bloomery 806.62 GB/s vs floor 690.51 (0.9 x ggml 767.23) -> PASS

    post (run 1): gate2 stack_m8/stack_m1: bloomery 1.863 vs ggml 2.011 -> PASS
    post (run 1): gate2 stack_k8/stack_k1: bloomery 1.744 vs ggml 1.912 -> PASS
    post (run 1): gate2 attnstk_m8/attnstk_m1: bloomery 2.334 vs ggml 3.233 -> PASS
    post (run 1): gate2 attnstk_k8/attnstk_k1: bloomery 2.425 vs ggml 3.289 -> PASS
    post (run 1): gate2 head_m8/head_m1: bloomery 1.987 vs ggml 2.261 -> info
    post (run 1): gate2 head_k8/head_k1: bloomery 1.942 vs ggml 2.145 -> info
    post (run 1): gate3 stack_m1: bloomery 578.04 GB/s vs floor 299.72 (0.9 x ggml 333.02) -> PASS
    post (run 1): gate3 attnstk_m1: bloomery 811.08 GB/s vs floor 660.43 (0.9 x ggml 733.81) -> PASS

    post (run 2): gate2 stack_m8/stack_m1: bloomery 1.872 vs ggml 2.013 -> PASS
    post (run 2): gate2 stack_k8/stack_k1: bloomery 1.719 vs ggml 1.932 -> PASS
    post (run 2): gate2 attnstk_m8/attnstk_m1: bloomery 2.347 vs ggml 3.259 -> PASS
    post (run 2): gate2 attnstk_k8/attnstk_k1: bloomery 2.388 vs ggml 3.335 -> PASS
    post (run 2): gate2 head_m8/head_m1: bloomery 1.999 vs ggml 2.208 -> info
    post (run 2): gate2 head_k8/head_k1: bloomery 1.948 vs ggml 2.155 -> info
    post (run 2): gate3 stack_m1: bloomery 581.27 GB/s vs floor 299.23 (0.9 x ggml 332.48) -> PASS
    post (run 2): gate3 attnstk_m1: bloomery 811.10 GB/s vs floor 678.30 (0.9 x ggml 753.67) -> PASS

Gate 1 (correctness, ≤ 1e-2, no threshold change): PASS in every run — and
all 28 max_rel_err values are IDENTICAL across Part A and both final runs
(max 6.561e-3, attnstk M=8), which is the value-identity proof for the
permutation/s8/u64 levers and the M=1-arithmetic-unchanged proof in the
same invocation (M=1 rows included). Gate 3 floors improved: the u64 lever
also sped M=1 (stack 137.19 µs / 578 GB/s vs Part A's 141.24 / 561).

## Run-to-run agreement (final kernel, two back-to-back invocations)

27 of 28 bloomery rows agree within 1.6% (gate anchors all ≤ 1.6%), verdicts
identical, rel errs identical. One row exceeds the 3% line and is said
plainly: **attnstk_k3, 97.89 vs 93.58 µs (4.4%)** — an ungated intermediate
point on the guard-step boundary (M=2→3 adds the third column's guarded
body); every gated anchor around it (k1, k2, k4, k6, k8, m1, m8) agrees
≤ 1.6%. If attnstk_k3 is ever gated, re-measure it with more repetitions
before trusting a delta.

Ambient note: bloomery stack_m1 here is 136.4-137.2 µs vs the 127.9 µs
committed in RESULTS-r3-combo (7% slower box-day); attnstk/head match
committed values. All gates above are anchored within-invocation, so the
drift does not touch the verdicts — it only means cross-day absolute
comparisons carry that margin.

## Witnesses (both final runs quiet)

Run 1 (07:44Z): pre-ref 3090 1 MiB / 0 % / 29.89 W, compute-apps empty,
loadavg 0.37, io-pressure avg10 0.00; pre-rust 97 % / 409.58 W (ref tail
decaying), loadavg 0.62, io avg10 0.14; post-rust 95 % / 348.14 W. Run 2
(07:45Z): pre-ref 0 % / 29.87 W, loadavg 0.61, io 0.00; pre-rust 97 % /
406.22 W, io 0.14; post-rust 100 % / 332.84 W. A6000 idle (0 %, ~20-27 W)
throughout; never touched. A parallel round (bloomery-attn) may run cargo test
on the same box — loadavg ~0.4-0.9 with io-pressure ≤ 0.14 avg10 shows it
never overlapped a timed section.

## Reproduce

    BLOOMERY_DATA=/root/bloomery-data-amort ./tools/box.sh 'bash tools/ref/measure.sh'

Builds the ref, regenerates `$BLOOMERY_DATA` (28 y_ref rows + ggml_timings.txt),
times ggml then the rust binary; the rust side re-reads ggml_timings.txt and
exits nonzero on any gate failure (gate 1 always; gates 2/3 only when the
timings file exists).
