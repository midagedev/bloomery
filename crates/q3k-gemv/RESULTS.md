# Stage 0 round 2 — Q3_K gemv (Rust/cuda-oxide) vs ggml mmvq

Same tensor (`blk.1.ffn_gate_exps.weight` of DeepSeek-V2-Lite-Chat Q3_K_M,
Q3_K, `stack` = 90,112 rows x 2048, 79.3 MB; `expert0` = 1,408 rows),
same card (RTX 3090), `tools/ref/measure.sh` back-to-back (ref then rust,
GPU-idle wait, witness blocks). Protocol: 20 warm-up + 200 timed launches
per shape. Truth = CPU f32 dots of ggml's own `dequantize_row_q3_K`.

Design choice: **q8_1 activations with an integer `dp4a` dot product**
(the mmvq design). The accuracy gate for this design is therefore
**<= 1e-2 max rel err** (ggml's own mmvq error is ~4e-3 for the same
reason: it also quantizes x). A 1e-7 figure with q8_1 activations would be
impossible and would be a red flag; the ~4e-3 below is expected.

## Final table (same `measure.sh` run, 2026-09-18T23:22Z)

| engine | shape | M | us/launch | GB/s | max rel err | rust/ggml (GB/s) |
|---|---|---|---|---|---|---|
| ggml mmvq | expert0 | 1 | 10.43 | 118.81 | 4.361e-03 | — |
| mulle rust | expert0 | 1 | 8.36 | 148.21 | 4.316e-03 | 1.25 |
| ggml mmvq | expert0 | 8 | 16.13 | 76.83 | 4.429e-03 | — |
| mulle rust | expert0 | 8 | 19.47 | 63.65 | 4.432e-03 | 0.83 |
| ggml mmvq | stack | 1 | 238.36 | 332.68 | 4.119e-03 | — |
| mulle rust | stack | 1 | 261.00 | 303.83 | 4.158e-03 | **0.91** |
| ggml mmvq | stack | 8 | 473.28 | 167.55 | 3.928e-03 | — |
| mulle rust | stack | 8 | 609.82 | 130.04 | 3.868e-03 | **0.78** |

- Performance gate (`stack` M=1 >= 0.9x ggml in the same run): **PASS** at
  0.91x (303.83 / 332.68). Gate cleared at optimisation step 1, so tuning
  stopped per the round contract (1 of 3 steps used).
- M=8 (`stack`): 0.78x ggml, above the round's 0.5x "good" mark. Reported,
  not tuned further (gate was M=1; `expert0` is L2-resident, report only).
- Accuracy gate (<= 1e-2 for this q8_1 design): **PASS** on all four shapes
  (3.9e-3..4.4e-3, same band as ggml itself).

## Step table (`stack` GB/s; M=1 gate, M=8 report)

| version | stack M=1 | stack M=8 | max rel err | note |
|---|---|---|---|---|
| baseline (round-1 main) | 134.4 | 32.7 | ~1e-7 (f32 path) | warp/row, x in smem, per-lane byte-assembled u32 loads |
| v1 redesign (f32 + cooperative loads) | 172.7 | 32.6 | ~1e-7 (f32 path) | warp-cooperative 28-word load into smem slot, same decode |
| v2 final (q8_1 + dp4a) | 303.8 | 130.0 | ~4.2e-3 (q8_1 path) | fused x quantize, integer dot, per-16 scale apply |

(v1 numbers from the 22:56Z run in this round; v2 from the 23:22Z final
run above. Run-to-run ggml variance on `stack` M=1 was 332.7..333.0.)

## Why each step did what it did (ptxas -v, PTX counts, ncu)

- **Baseline problem (from round-1 SASS diagnosis, confirmed):** each lane
  loaded its scattered qs/hmask bytes with byte loads assembled into u32
  (~960 scalar, mutually redundant global loads per warp per
  super-block), plus one shared-memory x read per weight per column.
- **v1** replaced the weight side with one coalesced 4-byte `LDG` per lane
  (28 aligned u32 words cover any 110 B super-block: `(sb*110)&3` is 0 for
  even `sb`, 2 for odd) into a warp-private 112 B smem slot, then decoded
  from slot bytes. PTX: 2 `ld.global.b32`, 94 `ld.shared` (64 b32 + 30 b8),
  `ptxas -v`: 58 registers, 0 spills. Result: M=1 134 -> 173 GB/s, but M=8
  flat at ~32.6 GB/s.
- **v1 profiling (ncu, `stack` M=1 launch):** DRAM 81.7 MB (1.03x the weight
  bytes — traffic already minimal), L2 83 MB, L1 119 MB, SM throughput 75%,
  but 4.13B thread instructions vs 184M useful FMAs (4.5% FMA density).
  Verdict: instruction-bound in f32 decode (~22 insts/weight/lane), not
  bandwidth-bound. More coalescing could not close the remaining 2x.
- **v2** therefore changed the math, not just the loads: fused q8_1
  quantize of x (64 blocks of 32 per column, warp butterfly absmax, ~160
  insts/thread once per launch), then 4 consecutive weights = 4 consecutive
  `qs` bytes (same pair) + 4 consecutive `hmask` bytes (same bit) = one
  `dp4a` against one packed q8 word. Signed 3-bit values come from one
  branchless combine per word (`vil | (mask * 0xFC)`); per-16 scales use the
  exact aux shuffle; the packed `vi` words are reused across all M columns.
  PTX: 2 `ld.global.b32`, 36 `ld.shared.b32` + 2 b8, 16 `dp4a`, `ptxas -v`:
  **53 registers, 0 spills, 0 stack**. Result: M=1 173 -> 304 GB/s,
  M=8 32.6 -> 130 GB/s (~3/weight/lane vs ~22 before).

## Two decode traps found by measurement (both fixed in v2)

1. `hmask` bit does **not** reset per 128-weight half: in
   `dequantize_row_q3_K` only `shift` resets per half, `m` keeps shifting
   (bits 0..3, then 4..7). Bit select is `(chunk*4+j)`, as round 1 already
   had it. First v2 used bit `j` for both halves: rel err 1.28.
2. `hmask` bytes do **not** advance per half either (`hm` never advances,
   only `q += 32`): both halves share `hm[0..32)`, distinguished by the bit
   above. First v2 read bytes 32..63 for the second half (qs bytes as
   hmask). Rel err stayed ~1.5 until fixed. Cross-checked: `hmask` is 32
   bytes = 256 bits = 1 bit per weight; `qs` is 64 bytes = 2 bits per
   weight. The kernel's per-(byte, pair, bit, scale) mapping was then
   verified weight-for-weight against the round-1 kernel's mapping (which
   measured 1.3e-7): identical on all 256 weights.

## Witness blocks (final measured run, `/tmp/measure-final.log` on the box)

```
--- witness pre-ref 2026-09-18T23:22:19Z ---
index, name, memory.used [MiB], utilization.gpu [%], power.draw [W]
0, NVIDIA GeForce RTX 3090, 1 MiB, 0 %, 30.65 W
1, NVIDIA RTX A6000, 38742 MiB, 100 %, 297.57 W
compute-apps-3090:
pid, used_gpu_memory [MiB]
loadavg: 1.84 1.64 1.68 2/1367 2556418
pressure-io avg10: some avg10=0.00 avg60=0.00 avg300=0.00 total=1046285415
--- witness post-ref 2026-09-18T23:22:21Z ---
index, name, memory.used [MiB], utilization.gpu [%], power.draw [W]
0, NVIDIA GeForce RTX 3090, 1 MiB, 98 %, 75.92 W
1, NVIDIA RTX A6000, 38742 MiB, 100 %, 296.51 W
compute-apps-3090:
pid, used_gpu_memory [MiB]
loadavg: 1.84 1.64 1.68 2/1369 2556447
pressure-io avg10: some avg10=0.00 avg60=0.00 avg300=0.00 total=1046291297
--- witness pre-rust 2026-09-18T23:22:21Z ---
index, name, memory.used [MiB], utilization.gpu [%], power.draw [W]
0, NVIDIA GeForce RTX 3090, 1 MiB, 98 %, 75.92 W
1, NVIDIA RTX A6000, 38742 MiB, 100 %, 296.51 W
compute-apps-3090:
pid, used_gpu_memory [MiB]
loadavg: 1.84 1.64 1.68 3/1369 2556463
pressure-io avg10: some avg10=0.00 avg60=0.00 avg300=0.00 total=1046291297
--- witness post-rust 2026-09-18T23:22:23Z ---
index, name, memory.used [MiB], utilization.gpu [%], power.draw [W]
0, NVIDIA GeForce RTX 3090, 1 MiB, 26 %, 145.33 W
1, NVIDIA RTX A6000, 38742 MiB, 100 %, 297.69 W
compute-apps-3090:
pid, used_gpu_memory [MiB]
loadavg: 1.85 1.65 1.68 2/1367 2556633
pressure-io avg10: some avg10=0.00 avg60=0.00 avg300=0.00 total=1046291702
```

- Both timed sections show an empty 3090 `compute-apps` list (header only,
  no pids); `measure.sh` proceeded without waiting.
- The A6000 holds someone else's job (100 %, ~297 W) for the whole run:
  expected, recorded only.

## Toolchain

```
.version 7.1
.target sm_86
```

(`cargo oxide inspect q3k_gemv --arch sm_86`; `cargo oxide run` has no
`--release` flag, host is always release. Box: nightly-2026-08-28,
LLVM 21.1.8, CUDA 13.0, `CUDA_VISIBLE_DEVICES` pinned to the 3090.)

## Reproduction

```
tools/box.sh bash tools/ref/measure.sh   # ref then rust, one invocation
```

Harness protocol unchanged (20 warm-up + 200 timed, same four shapes, same
`shape ... GB/s= ... max_rel_err=...` line format). No C/CUDA C in the rust
crate; `unsafe` only for shared-memory pointers, raw word loads, the
launch, and lane-0 stores, each with a SAFETY comment.
