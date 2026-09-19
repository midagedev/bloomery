# Stage 0 (MUL-9) — Q4_K and Q6_K gemv

GLM-authored round (outsourced spec), 2026-09-19. Two new kernels in
`crates/q3k-gemv/src/main.rs`, siblings of `q3k_gemv`, same harness.

**Activation design: q8_1, the Q3_K kernel's.** Both kernels reuse
`q3k_quantize_q8_1` unchanged (K = 2048 → 16 q8_1 blocks per column, d8 block
scales). Chosen because the gated comparison is ggml's mmvq, which also
quantizes activations to int8 before the K-quant dot; the error this adds
(~3–5e-3 measured, both engines) is the design's floor, not a kernel defect.
Correctness gate: max |err| / max |ref| ≤ 1e-2 per shape, the Q3_K precedent
(RESULTS.md), NOT the 1e-4 that f32 activations would demand.

**Q4_K** (`q4k_gemv`): 144 B super-block, always 4-aligned, no funnel. The
min offset forces a second dp4a chain — the sub-block dot is
`d8·(d·sc·A + (8·d·sc − dmin·mi)·B)` with `A = dp4a(nib−8, q8)` and
`B = dp4a(1s, q8)`: 16 dp4a per 32 values, twice Q3_K's density, inherent to
the nibble-sibling layout (sibling sub-blocks share each qs byte). One warp
per row; the warp covers FOUR super-blocks per iteration (lane owns one
32-value sub-block = one scale/min pair), so the B chain stays lane-local.
**Q6_K** (`q6k_gemv`): 210 B super-block, odd blocks sit 2 mod 4 → the Q3_K
16-bit funnel on every window. q6−32 ∈ [−32,31] fits a signed byte, so ONE
SWAR subtract folds the offset and each 16-value sub-block is a single 4-dp4a
chain — no B chain. Same lane geometry as `q3k_gemv`.

Both kernels' word arithmetic was verified on the host against the ggml
scalar dequant ports (inlined from `crates/gguf/src/quant.rs`) on real model
bytes BEFORE any GPU round: kernel algebra vs f64 truth 3.6e-8 (Q4_K) /
5.2e-8 (Q6_K); the q8-activation error it predicted (3.9e-3 / 2.2e-3) is
what the gate below measures. That pre-check caught three packing bugs
(Q4_K's 32-byte sub-block window, Q6_K's per-half qh bit index, the qh
section base) for the price of zero box roundtrips.

Tensors (this model): Q4_K = the 27 `blk.*.attn_output` [2048,2048]
(concatenated in blk order → N=55296) and 26 `ffn_down_shexp` (K=2816, out
of scope — the kernels are K=2048); Q6_K = `output.weight` [2048,102400],
the lm_head. Harness (`tools/ref/q3k_ref.cpp`) slices them from the GGUF
through `gguf_get_tensor_offset`, extends the CPU refs and the ggml timing
table; the x activations and all four Q3_K cases are byte-identical to the
round-1 harness (rng order untouched).

Lead-run conditions matched: `tools/ref/measure.sh`, RTX 3090 (dev card,
A6000 untouched and idle), 2026-09-19 06:02 UTC, single invocation.
Witnesses: `wait_gpu` compute-apps empty before each engine; pre-ref 3090
1 MiB / 0 % / 30.2 W, loadavg 0.79, io-pressure avg10 0.00; pre-rust 6 % /
155 W, loadavg 0.81, io-pressure 0.00; post-rust 26 % / 151 W. Full witness
blocks in the run log (`/tmp/mul9-measure.log` on the box).

| engine | shape | M | µs | GB/s | max rel err | rust/ggml |
|---|---|---:|---:|---:|---:|---:|
| ggml mmvq | attn0 | 1 | 9.96 | 236.9 | 3.1e-3 | — |
| bloomery rust | attn0 | 1 | 7.88 | **299.4** | 3.0e-3 | **1.26** |
| ggml mmvq | attn0 | 8 | 19.24 | 122.6 | 3.3e-3 | — |
| bloomery rust | attn0 | 8 | 24.07 | 98.0 | 3.1e-3 | 0.80 |
| ggml mmvq | attnstk | 1 | 83.29 | 764.8 | 3.0e-3 | — |
| bloomery rust | attnstk | 1 | 78.66 | **809.8** | 3.8e-3 | **1.06** |
| ggml mmvq | attnstk | 8 | 272.19 | 234.0 | 4.6e-3 | — |
| bloomery rust | attnstk | 8 | 407.56 | 156.3 | 6.6e-3 | 0.67 |
| ggml mmvq | head | 1 | 229.55 | 749.4 | 4.0e-3 | — |
| bloomery rust | head | 1 | 205.60 | **836.7** | 3.8e-3 | **1.12** |
| ggml mmvq | head | 8 | 502.46 | 342.4 | 4.5e-3 | — |
| bloomery rust | head | 8 | 529.14 | 325.1 | 3.8e-3 | 0.95 |

Gate: per type, the stack-shape M=1 ratio vs ggml mmvq on the same tensor,
bar ≥ 0.9 — **Q4_K attnstk M=1: 1.06, Q6_K head M=1: 1.12. PASS.**
Correctness: all six new shapes ≤ 6.6e-3 ≤ 1e-2 (q8_1 design). 810/837 GB/s
is 87/89 % of the 3090's 936 GB/s theoretical; ggml's own 765/749 is 82/80 %.
The four Q3_K shapes re-measured in the same invocation still pass (stack
M=1 620 vs 332, 1.87× — r3's 1.86× reproduced).

M=8 is below ggml on Q4_K (0.67×) and at parity on Q6_K (0.95×). The M=8
cost is structural: Q4_K's min-offset B chain doubles dp4a per value-word
per column, and each lane re-reads its qs words per column (the column loop
reuses q8 words but not the weight window). Levers if M=8 ever gets gated:
hoist the vi words out of the column loop, or store per-32-value activation
sums at quantize time (kills the B chain's dp4a entirely). Neither is in
this round's gate.

## Reproduce

    ./tools/box.sh 'cd ~/repo/bloomery-gemv && bash tools/ref/measure.sh'

Builds the harness, regenerates `$BLOOMERY_DATA/{attn.q4k,output.q6k,
y_ref_attn0_*,y_ref_attnstk_*,y_ref_head_*}` and the round-1 files, times
ggml then the rust binary (correctness gate exits nonzero on any rel err
> 1e-2).
