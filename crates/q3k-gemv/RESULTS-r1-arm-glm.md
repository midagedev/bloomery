# bloomery stage 0 — Q3_K gemv: bloomery (cuda-oxide Rust) vs ggml mmvq

One `tools/box.sh` invocation, 2026-09-19: `tools/ref` harness (ggml) then the
rust kernel back-to-back, same card (RTX 3090), same minute. Tensor
`blk.1.ffn_gate_exps.weight` (Q3_K, [2048×1408×64]) from
DeepSeek-V2-Lite-Chat.Q3_K_M.gguf; 20 warm-up + 200 timed launches per shape;
truth = CPU f32 dots of ggml's own `dequantize_row_q3_K`.

## Results

| engine | shape | M | µs/launch | GB/s | max rel err | GB/s ratio rust/ggml |
|---|---|---|---|---|---|---|
| ggml mmvq  | expert0 | 1 | 10.428  | 118.8 | 4.155e-3 | — |
| bloomery rust | expert0 | 1 | 30.626  | 40.5  | 1.385e-6 | 0.34 |
| ggml mmvq  | expert0 | 8 | 16.141  | 76.8  | 3.698e-3 | — |
| bloomery rust | expert0 | 8 | 218.280 | 5.7   | 1.295e-6 | 0.07 |
| ggml mmvq  | stack   | 1 | 238.293 | 332.8 | 4.341e-3 | — |
| bloomery rust | stack   | 1 | 1218.965| 65.1  | 1.491e-6 | **0.20** |
| ggml mmvq  | stack   | 8 | 474.368 | 167.2 | 5.206e-3 | — |
| bloomery rust | stack   | 8 | 9208.457| 8.6   | 1.616e-6 | **0.05** |

- **Accuracy gate (max rel err ≤ 1e-4): PASS** on all four shapes — rust lands
  at 1.3–1.6e-6 (the decode is the exact ggml port in f32). ggml's 3.7–5.2e-3
  is its q8_1 activation quantization: expected, recorded, not a failure.
- **Bandwidth target (stack ≥ 0.9× ggml): FAIL** — 0.20× at M=1, 0.05× at
  M=8. One optimisation pass was spent (below), then tuning stopped per spec.
- GB/s counts weight bytes only (880 B/row); `expert0` (1.2 MB) is
  L2-resident and latency-bound — reported, not tuned.

## The one optimisation pass (stack M=1 GB/s)

| variant | change | stack m1 GB/s |
|---|---|---|
| v0 | first working kernel, rolled loops, per-weight `u32_at` | 51.5 |
| v1 | qs/hmask windows pre-loaded as register-held u32s | 60.6 |
| v2 | `sel8` scalar match instead of dynamic `[u32;8]` index (PTX `st.local` 32→0) | 62.7 |
| v2b | `--unchecked-indexing` | 63.4 |
| v3 | `#[unroll]` while-loops (r×l×m constant trip counts; for-loop attrs are rejected) | 59.4 |
| v4 | v3 + branchless hm-bit (`low2-4+4*hb`) + `--unchecked-indexing` | **66.0** (65.1 in final run) |

M=8 was only measured at v3 (8.4) and v4 (8.6) — the tuning probe was stack
M=1 (the gated shape), so no earlier M=8 rows exist.

## Why it is slow (sm_86 SASS, ptxas -v, measured)

- m1 kernel after v4: 888 SASS instructions, 64–71 registers, no spills,
  3 blocks/SM (768 threads). All 89 loads are scalar `LDG.E`.
- Each lane owns a private 64-float x segment (lane owns sub-blocks
  4·lane..4·lane+4), so every warp-wide x load gathers 32 lanes × 4 B spread
  over 8 KB → ~32 L1 sectors per instruction instead of 4. Per warp-dot that
  is ~89 gather loads ≈ 700+ L1 cycles of sector serialization, vs ~222
  cycles of pure instruction issue. Measured 1856 cycles/warp-dot effective.
- m8: same mapping × 8 accumulators → 512 scalar x gathers per warp-dot
  (537 `LDG.E` in SASS), 80 registers, latency-bound. This mapping cannot
  reach memory bandwidth; ggml's mmvq stages x so a warp reads it once,
  broadcast-style.
- Next-round lever (redesign, out of stage-0's one-pass budget): stage x in
  dynamic shared memory (`LaunchConfig1D`'s shared-bytes arg is available,
  currently 0) or remap lanes so a warp cooperates on one super-block with x
  broadcast from registers/shuffles; make x loads provably 16B-aligned so
  they fuse to `LDG.E.128`.

## Witnesses (quiet-machine protocol; other card busy is expected — recorded)

Both blocks bracket each engine's timed section, printed from inside the
binaries. RTX A6000 held someone else's job (95–100 %, ~297 W) for the whole
run; the 3090 is our card.

```
[witness before ggml timing]
name, memory.used [MiB], utilization.gpu [%], power.draw [W]
NVIDIA GeForce RTX 3090, 1 MiB, 0 %, 30.91 W
NVIDIA RTX A6000, 38742 MiB, 95 %, 298.48 W
2.36 1.86 1.72 2/1391 2411160            (/proc/loadavg)
some avg10=0.00 avg60=0.00 avg300=0.25 total=1040666119
full avg10=0.00 avg60=0.00 avg300=0.25 total=1038041329

[witness after ggml timing]
name, memory.used [MiB], utilization.gpu [%], power.draw [W]
NVIDIA GeForce RTX 3090, 268 MiB, 0 %, 30.74 W
NVIDIA RTX A6000, 38742 MiB, 100 %, 297.09 W
2.36 1.86 1.72 2/1398 2411174            (/proc/loadavg)
some avg10=0.00 avg60=0.00 avg300=0.25 total=1040666662
full avg10=0.00 avg60=0.00 avg300=0.25 total=1038041871

[witness before bloomery timing]
name, memory.used [MiB], utilization.gpu [%], power.draw [W]
NVIDIA GeForce RTX 3090, 344 MiB, 3 %, 36.98 W
NVIDIA RTX A6000, 38742 MiB, 100 %, 297.80 W
2.41 1.87 1.73 2/1398 2411411            (/proc/loadavg)
some avg10=0.00 avg60=0.00 avg300=0.24 total=1040669465
full avg10=0.00 avg60=0.00 avg300=0.24 total=1038044674

[witness after bloomery timing]
name, memory.used [MiB], utilization.gpu [%], power.draw [W]
NVIDIA GeForce RTX 3090, 344 MiB, 100 %, 219.15 W
NVIDIA RTX A6000, 38742 MiB, 100 %, 296.13 W
2.38 1.88 1.73 2/1396 2411473
some avg10=0.00 avg60=0.00 avg300=0.24 total=1040669805
full avg10=0.00 avg60=0.00 avg300=0.24 total=1038045014
```

## Reproduction

```
tools/ref/build.sh                       # g++ -O3 -std=c++17 vs ik libggml (CUDA baked in)
/root/bloomery-data/q3k_ref                 # writes /root/bloomery-data + times ggml
cd crates/q3k-gemv && cargo oxide run q3k_gemv --arch sm_86 --unchecked-indexing
cd crates/q3k-gemv && cargo oxide inspect q3k_gemv --arch sm_86 | grep -E "^\.(target|version)"
  → .version 7.1
    .target sm_86
```

- `cargo oxide run` has no `--release` flag; the host binary is always built
  release (cargo-oxide host_cargo.rs, build log prints
  `Finished \`release\` profile [optimized]`).
- Box: nightly-2026-08-28, CUDA 13.0, driver 615.71.09, CUDA_VISIBLE_DEVICES
  pinned to the 3090.
- ggml timing: 200 back-to-back `ggml_backend_graph_compute` +
  `ggml_backend_synchronize`, wall clock (ggml computes on its own
  non-blocking stream; CUDA events on the default stream do not order against
  it — first measured this run, see q3k_ref.cpp).
- Data: `gate.q3k` 79,298,560 B (= 90112×880), activations from LCG seed 1,
  uniform [-1,1); `y_ref_*` recomputed every ref run on CPU (8 threads,
  ggml `to_float` dequant + sequential f32 dots).
