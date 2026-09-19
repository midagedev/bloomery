# Stage 0 round 3 (MUL-8) — the two arms combined

Lead-authored, 2026-09-19. Round 3 ran the same spec on two arms at effort `max`. GLM changed the q8_1 block
geometry (128-value blocks = one Q3_K half super-block, so a lane's four dp4a share one scale and one FMA);
muse kept the geometry and removed the bounds checks on the 73 device loads (`get_unchecked` with SAFETY
arguments), plus hoisted `drow`. The levers are orthogonal, so the lead merged them: GLM's geometry, then
every `q`/`d8`/`x` load in the kernels module rewritten to `get_unchecked` (44 sites), `w` loads already
unchecked from the muse merge. No other change.

Lead run, `tools/ref/measure.sh`, RTX 3090, 2026-09-19 ~00:20 UTC, 3090 compute-apps empty, A6000 idle:

| engine | shape | M | µs | GB/s | max rel err | rust/ggml |
|---|---|---:|---:|---:|---:|---:|
| ggml mmvq | expert0 | 1 | 10.44 | 118.7 | 4.4e-3 | — |
| bloomery rust (combo) | expert0 | 1 | 7.14 | **173.5** | 4.8e-3 | **1.46** |
| ggml mmvq | expert0 | 8 | 16.11 | 76.9 | 4.4e-3 | — |
| bloomery rust (combo) | expert0 | 8 | 14.64 | **84.6** | 3.8e-3 | **1.10** |
| ggml mmvq | stack | 1 | 238.4 | 332.7 | 4.1e-3 | — |
| bloomery rust (combo) | stack | 1 | 127.9 | **620.3** | 3.5e-3 | **1.86** |
| ggml mmvq | stack | 8 | 473.6 | 167.5 | 3.9e-3 | — |
| bloomery rust (combo) | stack | 8 | 396.2 | **200.2** | 4.6e-3 | **1.20** |

For reference, the arms alone (lead re-runs, same card, same hour): GLM 379.8 / 178.0 GB/s (1.14× / 1.06×),
muse 522.4 / 166.4 (1.57× / 0.99×). 620 GB/s is 66 % of the 3090's 936 GB/s theoretical; ggml's 333 is 36 %.
All four shapes now beat ggml mmvq on this card with ggml-class accuracy (q8_1 activations, gate ≤ 1e-2).

Arm reports: `RESULTS-r3-arm-glm.md`, `RESULTS-r3-arm-muse.md`. Spec premise corrected by the muse arm:
`QR3_K` is 4 in the vendored ggml (`ggml-common.h`), not 2 as the spec said.
