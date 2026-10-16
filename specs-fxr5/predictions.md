# fxr5 predictions, written before any box run (phase A, Mac)

Every number is `[derived]` with its terms, or `[measured]` with its source. Qwen3.8 fixture: 4 layers
(delta, selecting, delta, selecting: `qsa` = [1, 3], PLE site on layer 2), 512 experts, top 10, vocab 248,320.

## 0. Decomposition and the one term a box run fixes

The question: do the new fixture-oracle clauses hold on the fixture, and do they turn red on a defect?

Terms, from the code and the numbers already measured:

| term | source | value |
|---|---|---|
| per-site rounding of each engine | `act_rule::site_rel(ty, arm)` at the worst crest | q32 = 1.2858e-2, q128 = 2.5716e-2, q256 = 3.6369e-2 [derived: √block / (127·√12)] |
| dense sites | the engine's plan (`plan38.rs`) | every one Q8_0 (qkv, z, out, q, k, v, shared expert, PLE key/value, head, hc down/up) or F32 (router, β·α); ours f32 on the gemv, q8 blocks of 32 on the GEMM |
| routed experts | the file's types | read at run time; the card/host rules differ, the joint of ik and the q8_1 forms is the conservative one |
| logits scale | measured row of the batch set | sd 0.4391 [measured, design §3.1]; top-2 gap of V iid normals ≈ σ·0.2100·Exp [derived, design §3.1] |
| router boundary spacing | measured median margin 8.09e-3 [measured, design §3.3] | mean spacing d = 8.09e-3 / ln 2 = 1.167e-2 [derived: exponential spacing]; router-logit sd σ_z = d·512·φ(2.04) = 0.298 [derived: order statistics of 512 normals at the 10/11 boundary] |
| typical vs worst crest | mean crest of a block of n normals | 32: 2.37 (0.42 of √32), 128: 2.84 (0.25 of √128), 256: 3.05 (0.19 of √256) [derived: 200k-block Monte Carlo] |

The one term the derivation does not fix is `f`, the ratio of the measured error to its worst-crest band (the
typical crest, the multiplicities that are bounds, and the branch outputs' size against the stream's). Expected
value: **f = 0.30, range 0.15 to 0.45** [derived: crest ratios 0.19 to 0.42 per site, the sites' mix, and bounds in
the multiplicities (silu 1.64 against about 1)]. The first fixture run of crx reads it: logits_rel / head band.
Next action: f inside the range, the bands stand; f above 0.5, the multiplicities or a site are missing from
`ik_bands` (a finding, not a re-tune); f below 0.1, the band is vacuous against a defect (report; do not tighten).

Everything else in this file follows from f, the flip model below, and the clause rules. No lease is taken (gate lock
only), so no prediction card: the runs are functional.

## 1. Resource timeline (per clause, the fixture tier)

Per unit of work: one clause. Times are `[derived]` from the design's anchors (fixture step 4 to 6.5 ms a step,
GLM measured; set reads at 1 GB/s or better) except where marked.

| clause | host CPU | card SMs | PCIe | host DRAM | NVMe | wall |
|---|---|---|---|---|---|---|
| ik_argmax_fx ×3 | f64 head_gap over 248,320 logits: 1 ms; manifest parse + `result_output` read: 5 to 20 ms | 0 | 0 | 1 MB | page cache | sum, ~0.1 s each |
| cfx | manifest, 4 layers' taps, flip rules: ~0.1 s | 0 (the eager steps ran in (p)) | 0 | ~2 MB | ≤ 0.5 s set read (0.21 GB batch set) | sum, ~0.5 s |
| crx | plant rows 20 × 512 logits: <1 ms; compares: ms | one 5-row ubatch walk: <50 ms | 40 KB plant up, 410 KB taps down: 0.1 ms | <1 MB | ≤ 0.5 s | sum, ~0.6 s |
| tfx step4, step4_every_node | ~0.1 s each | 4 steps + 1 eager step: <50 ms | negligible | small | ≤ 0.5 s | ~0.6 s each |
| tfx d1k planted | plant build 1,024 × 4 rows: ~10 ms; compares | one 1,024-row ubatch walk <0.3 s + eager step | 8 MB plant: 0.4 ms | 8 MB | ≤ 0.5 s | ~0.8 s |
| tfx d3k step-fed | ~0.2 s | 3,000 graph steps at 4 to 6 ms: **12 to 18 s** | negligible | small | ≤ 0.5 s | the card's launch latency: one resource's max, **12 to 18 s** |
| tfx d3k planted | plant build 3,000 × 4 rows: ~30 ms | one 3,000-row ubatch walk: ~0.5 s | 24 MB plant: 1 ms | 25 MB | ≤ 0.5 s | ~1 s |
| elhfx | 127 rows of f64 pins and compares: ~1 to 2 s | 127 draft rows in runs of 8 (16 walks, <0.5 s) | host rows 10 KB each: negligible | small | ≤ 0.3 s | ~2 s, host f64 bound |
| tfx shift pairing (mtp) | ms | a 64-row pass (`prompt38`) already run by (t): 0 new | 0 | small | ≤ 0.3 s | <0.1 s |

The wall of the gate is the sum of its clauses (they are sequential in the code); the new terms add about 18 to 25 s
to the fixture batch's 43 to 67 s, of which D3K's 3,000 steps are 12 to 18 s: step latency the warps do not hide,
one resource. Flow levers: none worth taking. The D3K step-fed arm cannot share the 3,000-position gemm feed (its
state is the steps'), and overlapping its host work under the steps saves ms. A term under the card's shadow is worth 0.

## 2. `ik_bands`: the derivation, and the head band before any run

Per site `q = site_rel(ty, arm).joint() = √(q_ik² + q_ours²)` at the worst crest; a site's variance is
`mult · q²` (multiplicity: a projection once; a sigmoid's slope ¼ squared = 1/16; silu's log-slope 1.28 squared = 1.64;
q, k, v of a delta layer 3). The variances of the sites a stream has gained add; a band is
`1.04 · √(Σ)` (the hyper-connection mixes' down and up reach the stream through σ and add under 4 %, the gate's
`gemm_band` derivation, carried). The router of layer l reads the stream before the layer plus the layer's mixer.

Sites, per layer (types read from the file at run time, printed with each term):

- delta layer: `attn_qkv` 3, `attn_gate` (z, silu) 1.64, `ssm_beta` 1/16, `ssm_alpha` 1/16 (by analogy with β: a
  softplus decay), `ssm_out` 1;
- selecting layer: `attn_q` 1 + 1/16 (its rows beside the heads' sigmoid gate), `attn_k` 1, `attn_v` 1,
  `attn_output` 1; the indexer's two projections 0 (they reach the output by which pools are read: a selection,
  never a distance);
- block: shared expert gate 1.64, up 1, down 1; routed experts gate 1.64, up 1, down 1; the router and the shared
  gate `ffn_gate_inp*` (F32) 0;
- the PLE site on layer 2: `ple_key` 1, `ple_value` 1;
- the head: `output` 1.

Joint q by site class [derived, `site_rel`]:

| site | ik | ours | q joint |
|---|---|---|---|
| dense Q8_0, gemv (steps) | q8_2, 32 → 1.2858e-2 | f32 → 0 | 1.2858e-2 |
| dense Q8_0, GEMM (ubatch walk) | 1.2858e-2 | q8 blocks of 32 → 1.2858e-2 | 1.8184e-2 |
| expert Q4_K / Q5_K | q8_2, 32 → 1.2858e-2 | q8_1, 128 → 2.5716e-2 | 2.8750e-2 |
| expert IQ3_XXS / IQ4_XS | q8_K, 256 → 3.6369e-2 | q8_1, 128 → 2.5716e-2 | 4.4545e-2 |
| expert Q5_1 / IQ4_NL / Q8_0 | 1.2858e-2 | q8 blocks of 32 → 1.2858e-2 | 1.8184e-2 |

Three scenarios for the 4 layers' routed types (the file's are read at run time; the fixture copies the real file's
layer types by `source_layers`): all 32-blocks; Q4_K gate/up with 32-block down (UD-Q4_K_XL's majority); IQ3_XXS/IQ4_XS
gate/up with 32-block down.

| scenario | arm | router bands (layers 0..3) | stream bands (layers 0..3) | **head band** |
|---|---|---|---|---|
| all 32-blocks | gemv | 0.0321 0.0609 0.0840 0.0986 | 0.0546 0.0753 0.0949 0.1081 | **0.1089** |
| Q4_K gate/up | gemv | 0.0321 0.0716 0.0994 0.1182 | 0.0663 0.0922 0.1151 0.1317 | **0.1324** |
| IQ3_XXS/IQ4_XS gate/up | gemv | 0.0321 0.0918 0.1284 0.1546 | 0.0878 0.1229 0.1522 0.1748 | **0.1753** |
| all 32-blocks | GEMM | 0.0454 0.0782 0.1072 0.1247 | 0.0683 0.0934 0.1188 0.1348 | **0.1361** |
| Q4_K gate/up | GEMM | 0.0454 0.0868 0.1197 0.1407 | 0.0780 0.1075 0.1355 0.1544 | **0.1555** |
| IQ3_XXS/IQ4_XS gate/up | GEMM | 0.0454 0.1041 0.1447 0.1724 | 0.0969 0.1348 0.1681 0.1925 | **0.1934** |

**Head band prediction: 0.109 to 0.175 on the gemv arm (central 0.132), 0.136 to 0.193 on the GEMM arm (central
0.156)** [derived]. The tie cap is `6 · band · RMS(ik row)` with RMS 0.4391 [measured]: 0.287 to 0.461 (gemv), 0.358 to
0.509 (GEMM), in logit units; the batch row's top-2 gap is 0.108 [measured] and the mean top gap 0.092 [derived].

## 3. Predictions per clause (`[derived]`)

Flip model: a router logit moves by `s = f · band_router(l) · σ_z`, two logits by `τ = √2 · s`; the cell's flip
probability is `P = 0.5 − e^(r²/2) Q(r)` with `r = τ / d` (the pair at the top-10 boundary, an exponential spacing of
mean d = 1.167e-2), times 1.5 for the neighbouring pairs. At f = 0.30 and the gemv central router bands:

| layer | router band | s | r = τ/d | P(cell flips) |
|---|---|---|---|---|
| 0 | 0.0321 | 2.9e-3 | 0.35 | 0.17 |
| 1 | 0.0716 | 6.4e-3 | 0.78 | 0.31 |
| 2 | 0.0994 | 8.9e-3 | 1.08 | 0.37 |
| 3 | 0.1182 | 1.06e-2 | 1.28 | 0.41 |

### e2e

- **crx** (BATCH, ik's picks and final weights planted, GEMM arm). Nothing can flip. Predicted:
  - ik's picks taken at 20 of 20 cells (a pick that is not is a named FAIL; ties of exactly equal f32 logits are ~1e-5
    a cell).
  - Each layer's router logits' distance from ik's, worst of 5 positions: f × the layer's router band, i.e. at the
    central GEMM scenario 0.30 × (0.0454, 0.0868, 0.1197, 0.1407) = 0.014, 0.026, 0.036, 0.042; fraction of band
    0.15 to 0.45.
  - Layer 3 output: 0.30 × 0.1544 = 0.046 (0.15 to 0.45 of the stream band).
  - **logits_rel: 0.30 × 0.1555 = 0.047; range 0.15 to 0.45 of the head band = 0.020 to 0.087** (over the three
    scenarios' bands 0.136 to 0.193). Fraction of the band 0.15 to 0.45.
  - argmax by `head_tie`: our argmax is ik's with P ≈ 1 − 0.564·s_logit/(σ·g) = 1 − 0.564·0.0205/0.0922 = 0.87
    (s_logit = f · band · σ = 0.0205); where it is not, ik's logit at it lies ≪ the cap (≥ 0.36): PASS.
- **cfx** (BATCH, free arm, gemv). Per position and layer the flip probabilities above; the cells after the first
  flip err more (P ≈ 0.45):
  - flips among the 20 cells: **5 to 11, expected 8**;
  - **first-on-path flips (no earlier flip on their path): 1 to 4, expected 2.5** [2.46 derived], each within
    its pair cap `6·band·RMS(z)` (the typical pair error is 0.094 of it: PASS) and its margin bound;
  - layer outputs off every flip's path: **held cells 2 to 9, expected 5.1 of 20**; worst fraction of the stream band
    0.15 to 0.45;
  - **P(cfx holds no layer output) = P(a flip at layer 0, position 0) = 0.17**: then every cell is on a path; the clause prints
    `cfx: unexercised past the first flip's layer (held 0 of 20 cells)` and `unexercised` in place of its verdict, holds what it
    can (the flips under their caps, the argmax by `head_tie`) and is neither a pass nor a fail on that count (lead decision
    2026-10-10; crx holds every layer);
  - first layer a flip lies on the path of, by position: position 4 reaches layer ≥ 1 with probability 0.39,
    ≥ 2 with 0.064, ≥ 3 with 0.006 (the batch's last position is on a flip's path from layer 0 with 0.61);
  - last argmax tie: PASS; logits_rel printed, 0.08 to 0.2 (flip-perturbed).
- **tfx step4, step4_every_node** (steps, band from cfx `firsts[4]`, the step's own flips folded in): layers held =
  min(cfx's first flip layer at position 4, the step's own first flip layer): 0 with P = 0.61, ≥ 1 with 0.39,
  ≥ 2 with 0.06. Layer outputs below it within the band at 0.15 to 0.45 of it; argmax tie PASS;
  logits_rel held only when no flip lies on the path (≈ 0); printed 0.08 to 0.2.
- **tfx d1k planted** (1,024 positions by ubatch, ik's prefill routes planted, then the step): no flip in the
  prefill; the step's own: layers held 0..4 with P = 0.17, 0.25, 0.21, 0.15, 0.21 (E = 2.0). Held layers at 0.15 to
  0.45 of the GEMM band; logits_rel ≤ head band, held with P = 0.21, at 0.15 to 0.45 of it; argmax tie: PASS.
- **tfx d3k step-fed** (3,000 steps, then the step): layer 0 held when the step's layer-0 router does not flip
  (P = 0.83), at 0.15 to 0.45 of 0.0546 to 0.0878; layers ≥ 1 on a flip's path (a layer-0 flip at some of 3,000
  positions); argmax tie PASS; logits printed.
- **tfx d3k planted**: as d1k planted; D3K's step reads a selection (751 pools past 512), where ik keeps up to 3 keys
  of the 513th pool that ours does not: 3 of 2,051 keys in one softmax, ≤ 0.2 % of the attention output, inside the
  band.
- **ik_argmax_fx ×3**: PASS each; our argmax equals ik's with P ≈ 0.87 (g forced, k(iii), no flip on the path) and
  0.75 to 0.87 where a flip lies on the last position's path (g free); the rest within the cap.

### mtp

- **tfx shift pairing**: the shift-1 worst distance is flip-path-bound (the last 8 positions): 0.05 to 0.3; shifts 0
  and 2 are ≈ √2 = 1.41 (random hidden rows at other positions are near orthogonal): shift 1 closer than both by ≥ 4×:
  PASS.
- **elhfx** (127 rows, the draft's sites all Q8_0, ours f32 on the gemvs and `q8_0_gemv_sel_f32`): bands = the real
  PINs' (eh 1.2858e-2, router √3 q = 2.227e-2, l_out √7 q = 3.402e-2, logits √8 q = 3.637e-2) [derived: one term a
  site, the 8 terms of the real clause];
  - (e) eh_proj worst row: 0.42 to 0.55 of 1.2858e-2 (one site; the worst of 127 rows' crests ≈ 0.5);
  - (l) flips: P(row flips) ≈ 0.12 (router band 2.2e-2, f = 0.30: r = 0.24 → P = 0.08 × 1.5), about **15 rows**;
    each allowed under its cap; 112 rows held; l_out worst 0.2 to 0.5 of 3.402e-2;
  - (h) head rows ≈ 64; ties ≈ 13 %, wrong 0; logits distance printed, 0.2 to 0.5 of 3.637e-2.

## 4. The FAIL-first predictions (mutants in `mutants/`)

| mutant | what it does | clauses predicted red |
|---|---|---|
| m1 PLE skipped | the PLE site's conv is not launched on the steps' and the ubatch walk's front (layer 2) | (s) structure (112 ≠ 113 nodes, SC); crx: layer 3's router logits and the last output (certain: planted, every position); cfx: layer 2 or 3 held at some position with P ≈ 0.4 (cell (2,0) is clean with 0.36); tfx d1k / d3k planted: held ≥ 3 with P = 0.36; tfx step4 sets: held ≥ 3 with P = 0.006 (≈ never) |
| m2 logits tap off by one row | the ubatch walk's route-tap logits read from row (t+1) mod m, ids untouched | (crx) each layer's router logits (certain: rel ≈ √2 ≫ band); SC (g) free "route taps are the top ten of their own logits" |
| m3 GDN state not carried | `delta_lanes_body` (the one kernel every delta step, eager pass and ubatch walk launches) reads layer 0's state as zero at each launch (`fault.layer_of() == 0`, `crates/gpu/src/linear/delta.rs:978`); position 0 stays clean (its state is zero), positions >= 1 of every step diverge | **(g) D3K cut walk** (the unmutated one-ubatch walk against the two-ubatch cut: the second ubatch starts from zero instead of the carried state; certain when the load's ubatch is under 3,000 rows); **cfx** layer 0 at positions 1..4 (held where no layer-0 flip lies at an earlier or the same position, P(held per cell) 0.83 to 0.5 by position; red when the mixer's miss, 0.5 to 1.0 of the mixer output [derived: a rank-1 token-0 term missing against the token's own], times the mixer's share s of the stream after layer 0, exceeds the layer-0 band 0.0546 x 0.3 = 0.016: s >= 0.03, P(red | a held cell) ~ 0.8 with s = 0.2 to 0.5 unmeasured); **tfx step4 / step4_every_node** (the step at position 4 reads zero state) and **tfx d3k step-fed** (the final step after 3,000 steps): layer 0 held with P = 0.83, then red at the same s; **not red:** crx (one 5-row launch from position 0), tfx d1k / d3k planted unless the load cuts the prefill into several ubatches (the first launch is clean). The one unmeasured term is s |
| m4 eh halves swapped | `mtp_input` writes `[h_n,s \| e_n]` in place of `[e_n \| h_n,s]` | (elhfx e) `eh_proj` against ik's, then (l), (h); (n) the node-local pins (the host simulation packs `[e \| h]`) |

### Call chain per mutant (Mac evidence: no mutant is built here; each reaches its clause by this chain)

- **m1 PLE skipped.** Steps and the eager pass: `front` `program38.rs:513` → `ple_conv` `:559` (mutated). The ubatch walk:
  `wide38.rs:1096` `front` → `c.k.ple.enqueue_conv` `:1164` (mutated). crx: `forced_fx` `gate_qwen4exp_e2e.rs:4690` →
  `gemm_walk` `:4723` → `m.prompt38(.., Prompt38::Gemm)` `:2517` → `wide38 front`. cfx: `free_fx` `:4607` over the run
  `gemm_free` `:2662` feeds; tfx: `step_fx` `:4868` (steps → `program38 front`, planted → `wide38 front`). Both paths run the
  mutated line, so every new clause whose stream passes layer 2 sees it.
- **m2 route-tap logits one row off.** crx: `forced_fx` `:4690` → `gemm_walk` `:4723` → `Body38::ubatch_route_taps`
  `:2521` (`body38.rs:3036`) → `route_taps_host` `wide38.rs:2187`, whose `lg[t * taps.width..]` (`:2234`, mutated) is read.
  The tap logits are the router launch's own, copied into the tap at `wide38.rs:1271` before the plant overrides the
  router's buffer (`:1330` copies the slots after), so crx holds exactly this read. The spec's "logits tap, m14 shape": the
  gate's m14 is the route tap reading another layer's logits (`gate_qwen4exp_e2e.rs:645`); this patch reads another row
  (position), the same tap and the same line. The head's logits tap is `m.logits()` (`gemm_walk` `:2518`) and is not touched.
- **m3 no carry.** `gdn` `program38.rs:953` → `gdn_seq` `:1056` → `enqueue_delta_lanes` `:1100` (steps and eager) and
  `wide38.rs:1519` `gdn` → `enqueue_delta_lanes` `:1586` (ubatch walk): both reach `delta_lanes_body`
  (`crates/gpu/src/linear/delta.rs:978`, mutated), the sink's layer being `gpu.layer_sink(l)` (`program38.rs:516`).
- **m4 eh_proj halves.** `mtp38.rs:1872` and `:2263` `enqueue_mtp_input` → `mtp_input` (`crates/gpu/src/mtp.rs:105`; mutated
  `:160` and `:176`) → the draft's `eh_proj` GEMV → `elh_fx` (`gate_qwen4exp_mtp.rs:1714`) → `replay` `:1380`
  `(e)` compare with ik's `mtp_eh_proj-N`.

## 5. The cfx `held > 0` premise is gone; "unexercised" is a status (lead, 2026-10-10)

On a deterministic fixture a rounding-only engine change re-rolls which cells a flip leaves off its path, so `held >= 1` is a
flake class (P = 0.17 red on a correct engine). Rule in the code: when no output is held the clause prints by name
`cfx: unexercised past the first flip's layer (held 0 of N cells)`, keeps what it holds (flips under their caps, argmax by
`head_tie`) and prints `unexercised` where its verdict goes ([`outcome`]); it is never a pass and never red by itself. The
phase B report lists every `unexercised`. The same rule, every `held >= k` premise that depends on rounding:

| premise | where | rule now |
|---|---|---|
| cfx `held > 0` | e2e `free_fx` | `unexercised` (held 0 of N cells) |
| tfx `held` layers (0 = only the argmax is held) | e2e `step_fx`, five instances | named line `tfx {set}: unexercised past the first flip's layer (held 0 of 4 layers)`, verdict word `unexercised`; the old rule never required it |
| elhfx (l) `l_out_held > 0` | mtp `elh_fx` | `(elhfx l) unexercised past the flips (l_out held 0 of N rows)` |
| elhfx (h) `heads_held > 0` | mtp `elh_fx` | `(elhfx h) unexercised past the flips (heads held 0 of N head rows)` |

Not changed: the real-tier Oracle clauses' own `held > 0` rules (`gate_qwen4exp_mtp.rs` real replay, "a run that excuses every
row as a flip fails"), which stay byte for byte. Structural counts that no rounding moves stay held: `r.rows > 0`, `taken ==
cells` (the plant's margin is e^30), the declared clause counts.

## 6. The plant's weight round trip (crx, tfx planted), [derived]

`plant_ubatch_routes` takes router logits, so a pick's logit is `ln` of ik's final weight and every other expert's is
`PLANT_FILLER` (30) below the lowest pick's. The launch (`qwen3moe_router_route`: `exp(logit - max)` in f32, the sum of the
512 exps in f64, each probability the f32 divide by `sum as f32`, the 10 chosen probabilities summed in f64 and rounded once
to f32, each weight the f32 divide of its probability by that sum) returns `w' = exp(z_i) / sum_picks exp(z_j)` up to its
roundings. Terms, relative error of one planted weight against ik's, in f32 epsilons (eps = 2^-23 = 1.19e-7):

| term | bound | source |
|---|---|---|
| `z_i = fl32(ln w_i)`, to one ulp of z | ulp(z) = 2^e eps for 2^e <= \|z\| < 2^(e+1); \|ln w\| in [4, 8) (w in 3.4e-4 .. 0.018, the smallest of ten weights summing to 1) gives 2^2 eps = 4 eps = 4.77e-7; \|ln w\| in [2, 4) gives 2 eps; the relative error of exp(z) is the absolute error of z | `planted_route` (`w.ln()` in f32) |
| `exp` | 2 ulp (CUDA `expf` bound; assumed, the device `exp` is not read here) | router.rs:630 |
| the 3 divides/sums (p = e/sum, the f64 sum of picks rounded once, w = p/chosen) | 1.5 eps (half an eps each); the common factor 1/sum cancels except for these roundings | router.rs:640-700 |
| the 502 filler experts | each e^-30 = 9.4e-14 of the lowest pick's probability, together 4.7e-11 of it, so at most 4.7e-11 of the sum `inv` is rounded from (f32 resolves 6e-8); `chosen` excludes them and the top ten by probability is ik's set (a margin of e^30 = 1e13) | router.rs:630 |
| **total** | worst 4 eps + 3.5 eps = **7.5 eps = 8.9e-7** relative (typical picks: 1 to 2 eps from the ln, 4.5 to 5.5 eps total) | printed by `round_trip_line` |

Against the band: 8.9e-7 / 0.136 (the smallest GEMM head band) = **6.5e-6 of the head band**; it reaches the routed branch's
output at most 3x that (cancellation of ten terms of mixed sign, 2.7e-6), against the smallest single-site term of any band,
q32 = 1.2858e-2: 2e-4 of it. Negligible by five orders. The order of the picks: the launch sorts by descending probability,
ik's order is the same up to ties; a tie reorders an f32 sum by an ulp, inside the same bound.

What holds the picks: crx compares our walk's ids (`ubatch_route_taps`, the slots the walk ran) with ik's picks as sets at
every one of the 20 cells, so a pick that is not taken is a named FAIL. The planted tfx walks (D1K, D3K) arm no taps (the
taps hold the last walk's rows, `TAP_ROWS`), so no engine output shows their picks: they are held by construction (the same
plant, 30 below the lowest pick, which crx proves takes ik's set at 20 of 20) and by the steps' compare that follows. The
weights are shown by no engine output in either: this derivation and the printed bound are their proof.

## 7. Every bound against its expectation (lead item 3), [derived]

Sampling spread of a relative error `||e|| / ||x||` over N independent elements: relative sd `1/sqrt(2N)` of the norm (the
variance of `e^2` is twice its mean squared for Gaussian errors). Crest spread of a q8 block of 32: mean crest 2.37, sd 0.45,
so a row of `K/32` blocks has relative sd `0.37 / (2 sqrt(K/32))` of its error (0.021 at K = 2560). Band `B` = worst crest
(`sqrt(block)`, crest 5.66 at 32); expectation `E = f B`, f = 0.30 (0.15 to 0.45).

| bound | expectation E | bound B | B / E | N | spread sigma (of E) | (B - E) / sigma |
|---|---|---|---|---|---|---|
| head band, `logits_rel` (crx, tfx, cfx) | 0.30 B | B (0.136 to 0.193) | 3.3 (2.2 to 6.7) | 248,320 logits | 0.14 % | 1.6e3 (at f = 0.45: 8.6e2) |
| stream band per layer, `l_out_rel` (cfx held cells, crx last layer, tfx held layers) | 0.30 B | B | 3.3 | 10,240 values (4 x 2560) | 0.70 % | 330 |
| router band per layer (crx router logits) | 0.30 B | B | 3.3 | 512 logits | 3.1 % | 75 |
| per-row crest of one q8 site (elhfx (e), `eh_proj`, band q32 = 1.2858e-2) | 0.42 B | B | 2.4 | 127 rows, K/32 = 80 blocks | 2.1 % | 67 (the max of 127 rows adds 2.5 sigma) |
| flip pair cap `6 band RMS(z)` (first-on-path flips) | pair error sd = sqrt(2) f band s_z | cap | 14 | the cell's flips (printed) | | 14 sigma_pair (6 / (sqrt 2 f); 9.4 at f = 0.45) |
| margin bound `6 band spread(z)` (gap of a first-on-path flip) | a flipped pair's gap <= its pair error | bound | 14 | | | 14 sigma_pair |
| `head_tie` cap `6 band RMS(ik row)` | gap of a non-ik argmax <= pair error of two logits, sd sqrt(2) f band RMS = 0.029 at the central GEMM band | 0.36 to 0.51 | 12 to 17 | 248,320 logits | | 14 sigma_pair at the central band (9.4 at f = 0.45) |
| multiplicities (1.64 silu, 1/16 sigma, 3 qkv), HC_SHARE 1.04 | the typical slope (about 1) | the supremum of the log-slope | >= 1.28 in amplitude | | a supremum, not a mean | n/a |
| pairing_fx: shift 1 closer than shifts 0 and 2 (ratio d1 / d_other < 1) | E[d1 / d_other] = 0.05 to 0.3 / about 1.41 = 0.04 to 0.2 | 1 | 5 to 25 | pairs per shift (printed, at most 8) | not derivable on the Mac: d_other is the box set's data (I assumed near orthogonal rows; real residual rows share a large common direction, so d_other could be well under 1.41) and d1 is a worst over at most 8 flip-perturbed rows | the first run prints N, d0, d1, d2; if d1 / d_other > 0.5 the ordering is not separated from the flips and the lead re-decides, no re-tune |

No bound in the code equals its expectation. The 6-sigma rows hold at the least favourable N (K = 128: sigma 9 %, margin
15 sigma). The sampling spread is small against the model's own uncertainty, f (the activations' tails), which the first crx
run reads from `logits_rel / head band` and each layer's printed fraction of its band. The gate prints N where it is a count
(tokens, cells, positions, rows, pairs); it prints no sigma, which is a derived constant here.
