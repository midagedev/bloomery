# Bolt speed plan (perf owns; every role adds its section)

Base: origin/main `d0e8a8e1`. Model: Nemotron 3.5 Bolt GGUF, `/models/nemotron-3.5-bolt/` on ws. Nothing merges from
this plan without its gate and the GPU evidence in the PR body.

Scope (lead 18:38–18:39): bloomery = fast offloading (hybrid GPU+CPU MoE). Covers batched prefill, IQ2 decode kernels
(card `iq2_xxs_sel`, host qdot), offload tiers, speculative decoding, graphs/fusion, MoE routing/expert batching, other
levers. Judged on (a) whole-card A6000 and (b) offload tiers (3090 24 GB / smaller budget, experts on host) against
llama.cpp with `-ot` / `--n-cpu-moe`.

## 0. Blockers

- ~~ws unreachable from the box~~ restored 19:0x (tailscale re-login). The box's `ssh` is openssh-client 10.0p1 at
  `~/.local/bin/ssh`.

## 1. Baseline table (perf)

Measured rows come from `/workspace/repo/bloomery/specs/perf/bolt-lcpp-a6000.md` (box jobs named; mainline
llama.cpp cd0a5b6a1, `-ngl 99 -fa on`, the 24 GB IQ2_XXS file). Every new row is `tok/s @ n, depth, card`, with both
engines in one lease and the commands and output in the PR body. Empty cells are not measured yet (ws, §0).

| config | engine | pp512 | pp2048 | pp4096 | tg (depth 6 / 4096) | source |
|---|---|---|---|---|---|---|
| A6000 whole card | llama.cpp (-ub 512) | 1534.83 / 1537.07 | 1583.51 / 1577.88 | 1603.78 (s1187) | 80.95 / 80.50 @ d6 | **s1642** (2 rounds, r1 / r2), card bolt-speed-base-a6000 |
| A6000 whole card | llama.cpp **-ub 1024** | — | **1892.67 / 1889.88** | — | — | s1642: the ubatch lever is llama.cpp's best, **the line to beat** |
| A6000 whole card | bloomery main d0e8a8e1 | **82.47 / 81.17** (kind=step) | **81.85 / 81.04** | — | 82.23 / 80.86 @ d512; 81.16 / 80.63 @ d2048 | s1642; ours/lcpp pp 0.0533 ± 0.0059 @ 512, 0.0515 ± 0.0021 @ 2048, 0.0431 vs -ub 1024 |
| A6000 whole card | bloomery **E1** bolt-prefill 39fff4df (passes of 448, Q8_0/NVFP4 sites as gemm32, experts `_sel`) | **330.48** [329.52..331.43] | — | **327.07** [326.67..327.47] | 80.84 @ d512, 80.03 @ d4096 (decode unchanged) | **s1700** (one lease with main and G1, 2 rounds, card bolt-e1-pp-a6000; predicted 500..1000 → **below**: the per-slot `_sel` term is larger than derived) |
| A6000 whole card | bloomery **G1** bolt-prefill-g1 f7063282 (E1 + IQ2_XXS grouped GEMM experts) | **927.19** [926.74..927.64] | — | **971.66** [971.46..971.87] | 81.06 @ d512, 80.41 @ d4096 | **s1700**; engine's band 800..1600 → **in**; 11.4× / 12.1× main (81.08 / 80.44 in the same lease), 0.60× / 0.61× llama.cpp -ub 512 |
| A6000 whole card | bloomery **G1 room-gated** bolt-prefill-g1 19863679 (twins kept) | **938.26** [933.67..942.85] | — | **989.04** [984.13..993.95] | 82.07 @ d512, 81.37 @ d4096 (decode unchanged) | **s1730** (card bolt-g1b-pp-a6000, predict 880..1000 → **in**, 2 rounds); same lease llama.cpp -ub 512 **1549.84 / 1603.37** → ours 0.605× / 0.617× |
| 3090 whole card (250 W) | llama.cpp (-ub 512) | 1358.69 / 1349.02 | 1373.19 / 1354.29 | 1384.73 (s1189) | 77.55 / 77.47 @ d6 [cpu-busy] | **s1645**, card bolt-speed-base-3090; tg rows tagged by perf's own gate build |
| 3090 whole card | llama.cpp **-ub 1024** | — | **1673.11 / 1658.67** | — | — | s1645: the line to beat on the 3090 |
| 3090 whole card | bloomery main d0e8a8e1 | 81.95 / 80.68 (kind=step) | 80.90 / 80.71 | — | 81.85 / 80.61 @ d512; 80.28 / 80.25 @ d2048 | s1645: the file fits the 3090 whole |
| 3090 whole card | bloomery **G1 room-gated** bolt-prefill-g1 19863679 (= 30ce3f32 + ptx rows; twins do not fit, NVFP4 sites run E1's 8-column gemv in the passes) | **495.62** [493.25..498.00] | — | **501.45** [498.96..503.95] | 81.47 @ d512, 80.59 @ d4096 (decode unchanged) | **s1729** (card bolt-g1b-pp-3090, predict 300..700 → **in**, 2 rounds); loads (the f7063282 OOM is gone); same lease llama.cpp -ub 512 **1354.70 / 1374.99** → ours 0.366× / 0.365× |
| A6000 whole card | decode A/B, one lease | — | — | — | main d0e8a8e1 **81.19** [80.79..82.21]; **#326** iq2opt 8a16e7a4 **84.84** [84.45..85.45] (+4.5 %); **#329** P3b 77180fa2 **82.90** [82.63..83.30] (+2.1 %, nodes 1084 → 1004) @ d512, n=96 (n=4 each) | **s1724**, card bolt-decode-ab-a6000 (predict 83.5..86.0 for #326 → **in**); rows tagged cpu-busy/other-busy, read beside the means |
| 3090 `--n-cpu-moe 16` (experts of blocks 0–15 on host; K counts all 88 blocks, ~7 of them MoE [derived]) | llama.cpp | **1060.85** [1054.17..1069.50] (n=4) | — | — | **72.42** [71.60..73.07] @ d6 (n=4) | **s1647**, card bolt-lcpp-ncmoe-3090, `-ngl 99 -fa on`, n=96 |
| 3090 `--n-cpu-moe 8` | llama.cpp | **1249.75** [1249.26..1250.24] (n=2) [cpu-busy 2/2] | — | — | **75.83** [75.74..75.92] @ d6 (n=2) | **s1722**, card bolt-lcpp-ncmoe2-3090, `-ngl 99 -fa on`, n=96 |
| 3090 `--n-cpu-moe 24` | llama.cpp | **938.50** [934.48..942.51] (n=2) | — | — | **72.92** [72.47..73.38] @ d6 (n=2) | **s1726**, same card |
| 3090 `--n-cpu-moe 40` | llama.cpp | **806.33** [806.09..806.56] (n=2) | — | — | **68.98** [67.73..70.23] @ d6 (n=2) | **s1727**, same card |
| 3090 `--n-cpu-moe 48` | llama.cpp | **740.19** [716.81..763.58, spread 6.5 %] (n=2) | — | — | **68.40** [68.20..68.60] @ d6 (n=2) | **s1728**, same card |
| 3090 `--n-cpu-moe 88` (**every routed expert on host**) | llama.cpp | **532.19** [525.67..537.34] (n=4) | — | — | **59.09** [53.39..61.26, spread 14.7 %] @ d6 (n=4) | **s1655**; inside our derived ~64 tok/s all-host decode bound (§3.3) |
| 3090 / 12–16 GB budget, experts on host | bloomery | **cannot run** | | | | Bolt has no host tier (§3.3) |

To run first when ws returns (perf, 3090 owned by perf, A6000 through the box queue):
1. llama-bench, both cards: `-p 512,2048,4096 -n 128 -ub 512,1024,2048 -ngl 99 -fa on` (pp2048 + the -ub sweep).
2. 3090 offload: `--n-cpu-moe 8,16,24,40` and `-ot 'blk\.(…)\.ffn_.*_exps=CPU'` at tg128 / pp512, `-t 32`.
3. ours: `generate_qwen3moe` P = 512/2048 (prompt row) on both cards, `depth-serve.sh` tg.
4. nsys: llama.cpp pp512 split by op kind (engine's list in §2), and **our own decode nsys** (none exists for Bolt;
   the 1084 nodes' gap term is unmeasured).

## 2. Profile (perf)

**Measured 19:43 (A6000, `nsys profile --cuda-graph-trace=node`, lease card bolt-speed-nsys-a6000, our main
d0e8a8e1 `generate_qwen3moe` d6 64 steps, `tools/ref/nsys-table.py`): kernel time 12175 µs a step** (the wall step is
12.2 ms, so the graph's gaps are ~0: c_node is near 0 on this graph, and a node fold is worth its kernel's time only).

| kernel (grid) | n/step | mean µs | µs/step | share | bytes floor at ~700 GB/s [derived] | headroom [derived] |
|---|---|---|---|---|---|---|
| nvfp4_gemv ssm_in (2320) | 40 | 63.26 | 2530 | 20.8 % | 61 µs | ~0 |
| nvfp4_gemv ssm_out (512) | 40 | 35.35 | 1414 | 11.6 % | 30 µs | ~0.2 ms |
| **mamba2_step** (512×128) | 40 | 26.32 | 1053 | 8.6 % | 8.4 MB state r+w → 12 µs | **~0.55 ms** |
| **q8_0_gemv** (512: latent down + attn) | 56 | 15.39 | 862 | 7.1 % | latent 4.35 MB → 6.2 µs | **~0.4 ms** |
| q8_0_gemv head (16384) | 1 | 807 | 807 | 6.6 % | 0.57 GB → 810 µs | 0 |
| iq2_xxs_gemv_sel (all) | 160 | 11.9–18.6 | 2570 | 21.1 % | — | iq2opt: ×0.72–0.85 (PR #326) → ~0.55 ms |
| **rms_norm** (**1×1 grid: one block**) | 89 | 7.12 | 634 | 5.2 % | 16–32 KB: ~1 µs | **~0.45 ms** |
| q8_0_gemv latent up (128) | 40 | 8.94 | 358 | 2.9 % | 4.35 MB → 6.2 µs | ~0.1 ms |
| **routers F32** (60–124) | 40 | 13.2–27.8 | 784 | 6.4 % | 432×4096×4 = 7.1 MB → 10 µs | **~0.35 ms** |
| q3k_quantize_q8_1 | 240 | 1.8–2.4 | 488 | 4.0 % | — | folds: ~1.9 µs a launch |
| add | 128 | 1.33 | 170 | 1.4 % | — | fold into the gemv epilogue |
| gqa_flash_seg/merge, kv_append, relu_sqr, weighted_sum, argmax | | | ~430 | 3.5 % | | small |

Read [derived from the measured table]: the new decode levers, beyond iq2opt, are a multi-block rms_norm (or norm
folded into its consumer), mamba2_step at its byte rate, the 512-block q8_0 gemv, and the F32 routers at the rate:
~1.7 ms together, 12.2 → ~10.5 ms (+16 %) on top of iq2opt's ~0.55 ms. Each is its own kernel change with an owning
gate; none lands by 20:40. P3a's fold (40 nodes) is worth ~40 × 1.9 µs quantize + the norm's difference ≈ 0.08 ms
(0.6 %): it lands on bit identity and the structure pin, not an A/B.

llama.cpp pp512 nsys (same sitting, `llama-bench -p 512 -n 0 -r 1`, warm-up + 1 rep = 2 passes in the table): 626 ms
kernel time, ~313 ms a pass. `mul_mat_q` (MMQ: dense and MUL_MAT_ID experts) is ~75 % of it, the SSD scan's
`ssm_ssd_pre_matmul_kernel` 3.1 %, `mm_ids_helper` 4.8 %, rms_norm/bin_bcast/quantize_mmq ~7 %. So llama.cpp's
prefill is MMQ-bound: G1 (experts as GEMM) is the lever, the scan is second-order at P = 512.
Files: ws `$HOME/boltspeed/{ours-d6,lcpp-pp512}.{nsys-rep,sqlite,table.txt}`.

Already measured, llama.cpp's decode token on the A6000 (nsys node trace, 12.36 ms a token):
NVFP4 gemv 3.84 ms (31 %, 80 launches), IQ2_XXS 2.63 (21 %: shared up 40, shared down 40, routed 80), Q8_0 1.74
(14 %), argsort 0.54, quantize_q8_1 0.48 (358 launches), rms_norm 0.47, ssm_scan 0.41, f32 router 0.40, add 0.35.
Our kernels (bench, A6000): nvfp4_18560 66.19 µs (648 GB/s), nvfp4_4096 35.16 µs (539 GB/s), iq2xxs shared
13.89 / 16.55 µs (411 / 345 GB/s), routed 11.35–17.66 µs (324–427 GB/s).

### 2b. #328 G1 prefill nsys at P = 4096 (perf, 19863679, `-n 2`, s1733 A6000 / s1734 3090)

| site | A6000 s (share) | 3090 s (share) | llama.cpp |
|---|---|---|---|
| mamba2_steps (sequential scan, 400 launches) | 1.12 (27.1 %) | 1.26 (15.7 %) | SSD chunked (ssm_ssd_pre_matmul + Kernel2) ~10–15 ms / 512 ids (pp512 nsys) |
| gqa_flash_seg per row (131104 launches) + merge | 0.90 + 0.03 (22.5 %) | 1.06 + 0.03 (13.6 %) | batched flash, not in its top 30 |
| NVFP4 sites: twin gemm_q8_0p (A6000) / nvfp4_gemv 8-col (3090) | 0.54 + 0.24 + 0.05 (20 %) | 2.38 + 1.64 (50.2 %) | mul_mat_q |
| gemm_iq2xxs (all shapes) | ~0.85 | ~1.0 | mul_mat_q (ids) |
| kernel total / prompt | 4.13 / 4.27 s (958.56 tok/s) | 8.02 / 8.27 s (495.38 tok/s) | |

Derived plan (all [derived]): (1) group-16 batched prefill flash −0.85 / −1.0 s; (2) chunked Mamba-2 scan −1.0 s;
(3) G5 NVFP4 GEMM: the 3090's 4.02 s → ~0.86 s. A6000 ≈ 2.3 s → ~1800 tok/s (≥ 1604); 3090 ≈ 2.8 s → ~1460 (≥ 1385).
Without (2) neither card clears. perf proposed: (1) perf kernel, (2) perf spec, (3) engine's G5 (running).

## 3. Cost model (perf, all [derived] unless a job is named)

### 3.1 Bytes a decode step (hparams: `crates/model/tests/nemotron_h_moe_meta.rs`)
| term | bytes a step |
|---|---|
| NVFP4 `ssm_in` 18560×4096 + `ssm_out` 4096×8192 at 4.5 bits, × 40 | 2.47 GB |
| Mamba-2 scan state 8192×128 f32, read + written, × 40 | 0.34 GB |
| shared expert up + down 5376×4096 IQ2_XXS, × 40 | 0.45 GB |
| latent pair Q8_0 2×1024×4096, × 40 | 0.36 GB |
| attention q/k/v/o Q8_0, × 8 | 0.30 GB |
| router F32 ≤ 496×4096, × 40 | 0.32 GB |
| output head Q8_0 131072×4096 | 0.57 GB |
| **routed experts** 9 × ~1.27 MB (17.5 GB / 13,824 experts), × 40 | **0.46 GB** |
| total | **≈ 5.27 GB** |

- At ~700 GB/s that is **7.5 ms → ~133 tok/s, the A6000's decode ceiling.** Ours is 12.2 ms (81.99 tok/s), an
  average of 432 GB/s, so **~4.7 ms of headroom**.
- NVFP4 is already at the rate (4.05 ms for 2.47 GB). The gap sits in everything else: 2.8 GB in ~8 ms, about
  350 GB/s, made of small launches (IQ2 at 324–427 GB/s, Q8_0, the 1084 nodes' gaps).
- **f_routed ≈ 0.087.** llama.cpp's measured routed time is 80 × ~15 µs ≈ 1.2 ms of 12.36 ms. A 2-row verify
  costs about 1.09× a step at zero expert overlap, so a lookup draft wins wherever α > ~0.1 (models' gain formula).

### 3.2 Prompt pass at P = 512 (engine §5.1 agrees)
- Dense weights are read once a pass. The routed stacks are touched nearly whole (4608 slots over 240–496
  experts), so a pass reads ≈ 22 GB, about **31 ms at 700 GB/s, a byte floor of ~16k tok/s**.
- Compute is 2 × 9.1 B × 512 ≈ 9.3 TFLOP. At ~100 TOPS int8 mma effective, that is ~93 ms, **~5.5k tok/s**.
- llama.cpp's 1549 tok/s is 331 ms a pass, far under both floors.
- Stage-1 arms with today's m ≤ 8 kernels (E1): dense once per 8 rows, routed per slot. That is
  4.5 + 8 × 0.46 GB ≈ 8.2 GB per 8 tokens, ~19 ms at ~430 GB/s, about **~420 tok/s (5×)**.
- With the routed IQ2 GEMM (G1) and the NVFP4 GEMM arm (G2), the pass is compute-bound: **≥ 1550 (parity) to ~3k tok/s**.

### 3.3 Offload tier (ws host: TR PRO 5975WX, 147.7 GB/s DRAM, qdot MT cap 122–136 GB/s; PCIe 4.0 ×16, H2D 26.3 GB/s pinned)
- **Bolt has no host tier today:** `type Host = NoHost` (`nemotron_h_moe/body.rs:296`), and the plan refuses any
  host expert (`arch/hybrid.rs:578-583`). The host IQ2_XXS × Q8_K AVX2 dot exists (`qdot/src/lib.rs:8047`, mainline
  sum order); the host MoE has no relu² non-gated latent expert.
- **Decode with all routed experts on the host** (card holds ~5.3 GB):
  - The host reads 9 × 1.27 MB = 11.4 MB a MoE layer, ≈ 90 µs at ~125 GB/s, plus ~20 µs of round trip.
  - Overlapped with the card's shared expert (~25 µs), that adds ~85 µs a layer × 40 ≈ 3.4 ms a token, so 12.2 →
    15.6 ms, **≈ 64 tok/s with Bolt on an 8 GB card.**
  - At half the experts spilled: ≈ 72 tok/s.
  - llama.cpp `--n-cpu-moe` is the row to beat (ws).
- **Prompt under offload (engine E5):**
  - Large m: stream the spilled stacks up (17.5 GB at 26 GB/s ≈ 0.67 s a pass, ≈ 1.3 ms/token at P = 512 for full
    spill, overlappable with the pass's compute).
  - Small m: host union.
  - The crossover is where m × 9 picks ≥ the stack (≈ m ≥ 30–50).

## 4. Levers ranked by gain × cost (all roles; gain [derived] at A6000 unless noted)

| # | lever | predicted gain | cost | owner | lands by 20:40? | stage |
|---|---|---|---|---|---|---|
| 1 | **Batched prefill** E1 (walk, today's kernels) → G1 (IQ2 grouped GEMM) + G2 (NVFP4 GEMM arm) + G3 (router rows) | pp512 81.7 → ~420 (E1) → ≥ 1550 (G1+G2) | 1.5 h + GLM ∥ | engine design, GLM impl | **measured s1700 (A6000):** E1 39fff4df 330.48 / 327.07 @ 512 / 4096; **G1 f7063282 927.19 / 971.66** (11.4× / 12.1× main, 0.60× llama.cpp); engine PR #328 (not merged). G1 nsys at P = 4096 queued (perf) to name the next site | 1 |
| 2 | **P1 iq2opt** (done: block-shared grid/sign table, SASS −36/−39 % a sub-block, bit-identical): branch `perf-bolt-iq2opt` on main | decode −0.4..−0.6 ms/token, **+3.5–5 %** (shared up 13.89 → 8.9–11.5 µs, routed 0.70–0.92×) | gates + A/B only | perf | **PR #326 open (not merged)**, rebased on main 85511826. Gates green on the A6000 (gate-gpu-iq s1644, iq-sel s1651, Bolt e2e fixture s1654 / real s1656, glm-sel s1657, routed-legs s1663, qwen4exp-iqleg s1675). Kernel bench (s1660 / s1671): shared up iq2xxs_5376 **13.824 → 10.549 µs** (in the card's 8.9..11.5), routed 0.72–0.85×; ~0.55 ms a token [derived] | 1 |
| 3 | **Lookup spec decode** (M3 lanes + Rows/Rollback + verify gate + seat lever) | (1+α)/1.09; code/edit workloads with α 0.3–0.6 [unmeasured] → +20–45 % | M–L | models design, GLM impl | no | 2 |
| 4 | P2 `_sel` prologue: u32 division instead of 64-bit `usize` (~140 instructions before the first load; the routed up runs one iteration a warp) | routed up −1..−3 µs × 40, ~−0.1..−0.2 ms/token (1–2 %) | S (md5 move of every `_sel`, bit-identical) | perf | **done** `4b83c062` on `perf-bolt-iq2opt`: SASS iq2_xxs_gemv_sel 401 → 304, iq2_xs 420 → 344, iq3_xxs 442 → 365, iq4_nl_sel32 368 → 287 (build-time counts); in the same gate batch and bench | 1 |
| 5 | P3 fold `quantize_q8_1` into the preceding norm | Δt = −ΔN × the removed launch (graph gaps ~0, §2): ~0.08 ms a 40-node fold [derived] | S each | perf design (P3a GLM impl, P3b perf) | **P3a 4cd38136** (Mamba ssm_in, 1084 → 1044) + **P3b 77180fa2** (MoE pre-norm + shared up quantize, → 1004), branch `perf-bolt-normquant-moe`; real tier PASSED on the 3090 (s1703, s1711), logit lines byte-identical; P3a FAIL-first killed (s1707/s1709). Under the ruler: no A/B claim | 1 |
| 6 | P4 `ActCols` 32-bit column offsets (frees ~14 registers in the five IQ row kernels) | iq2_xxs_rows 4 → 5–6 blocks/SM; shared IQ2 toward the byte term | S–M (md5 move) | perf (GLM impl) | no | 2 |
| 7 | **Host tier for Bolt** (offload: relu² non-gated latent host experts, placement accepts host experts, card/host overlap per MoE layer) | Bolt on an 8–16 GB card at ~64–72 tok/s [derived] vs today: cannot load | L (4 h+) | engine design | no | 3 |
| 8 | Prefill under offload (stream up vs host union, E5) | offload pp | L | engine | no | 3 |
| 9 | Chunked SSD scan (E3), GROUP_16 tile flash (E2) | ≤ 0.1 ms/token each at P = 512; more at 4096 | 2–3 h each | engine | no | 2 |
| 10 | Router F32 → f16 rows (0.32 → 0.16 GB) | ~0.23 ms (2 %) | precision class, routing near ties | — | no; not opened | — |

## 5. Engine — batched prefill / verify forward (engine, 18:50 KST)

Target (lead 18:40): Bolt prompt ≥ 10× today's 81.7 tok/s on the A6000 (≥ 820 tok/s at P = 512). llama.cpp is 1548.68
pp512 / 1603.78 pp4096 (s1187). Prompt today = one decode step per token (`nemotron_h_moe/mod.rs`: "a prompt is one step
per token").

### 5.1 Where a prompt token's time goes [derived, from decode bytes + the llama.cpp nsys split]
Per-token decode reads ≈ 5.0 GB: NVFP4 `ssm_in`/`ssm_out` 2.47 GB (40 Mamba layers), Q8_0 ≈ 1.2 GB (latent pair,
attention, head), IQ2_XXS routed ≈ 0.40 GB + shared ≈ 0.45 GB, F32 router ≈ 0.22 GB. A pass of m rows reads each dense
weight **once** (m× saving). The routed stacks are touched nearly whole at m = 512 (4608 slots over ~346 experts/layer):
≈ 15.9 GiB a pass, ≈ 0.05 ms/token at 650 GB/s. Compute at m = 512: 2 × 9.1 B × 512 ≈ 9.3 TFLOP a pass; NVFP4 sites are
≈ 52 % of it, routed + shared experts ≈ 39 %, attention/latent ≈ 9 %. So the prefill is GEMM-bound on two weight
types that have **no many-row kernel today**: NVFP4 (refused, `site.rs:735`) and IQ2_XXS (`iq2_xxs_gemv_sel` re-reads its
rows per slot, `iq_sel.rs:484-541`).

### 5.2 The design: one layer-major pass walk (Nano's `nemotron_h/prefill.rs` shape), op arm chosen by m
- **Walk:** `nemotron_h_moe/prefill.rs` (new), `GpuModel<Body>::prefill[_with]`, passes of `PASS_ROWS` = 512 ids:
  embed rows (`q38::enqueue_embed_rows` already takes n ids) → per layer norm_rows → mixer → residual_rows → last row
  into the head. Prompt buffers cut at load (`Prompt`, allocation-free pass), as Nano's.
- **Mamba-2:** norm_rows → `ssm_in` [m × 18560] → `Mamba2Kernels::enqueue_steps` (exists, bit-equal U steps, reads the
  GEMM output by `z0/x0/dt0` windows at stride `ld`) → `ssm_out`. Stage 2: chunked (SSD) scan if nsys says the token-serial
  scan matters (≈ 40 launches × U serial tokens).
- **Attention (8 layers):** q/k/v rows GEMM → head-major regroup → `enqueue_kv_append_planes` → `flash.enqueue_pass::<GROUP_16>`
  in runs of `MAX_PASS_ROWS` (accepts GROUP_16, `flash_gqa.rs:4641`; 2·m·n_kv+1 launches — ≈ 0.1 ms/token, stage 1 OK) →
  `attn_output` GEMM. Stage 2: a GROUP_16 entry of the tile prefill flash (`flash_gqa_prefill.rs`, group 8 only today).
- **Latent MoE:** norm_rows → router rows (`route_body` `$scores`/`$pick` take t rows, `enqueue_router_rows`
  `route_body.rs:826`; Bolt's `Routers` wraps the 1-token entry only → add a rows wrapper + m-row `RouteOut` buffers) →
  `ffn_latent_down` Q8_0 GEMM (`gemm32` Q8_0Plane, dense table) → `enqueue_route(ids, 9m)` → routed **up** GEMM over the
  table, `GemmInput::Shared{top_k: 9}` (slot reads its token's latent column) → relu² over 9m rows → quantize → **down** GEMM
  `PerSlot` → `enqueue_weighted_sum(.., m)` (already m rows, `elem.rs:2294`) → `ffn_latent_up` Q8_0 GEMM; shared expert
  up/down IQ2_XXS GEMM on the dense table → add → residual. Exactly qwen3moe `ubatch.rs::ffn` (602-683) minus the gate.
- **Numeric class:** a pass's rows agree with the steps' to the GEMM's band, not bit for bit (Nano's contract). Prompt
  ≤ 8 ids may stay on the step loop (stage 1) — verify needs the bit-exact arm (5.4).

### 5.3 Ranked items (gain × cost), owners, stage
| # | item | gain [derived] | cost | owner | stage |
|---|---|---|---|---|---|
| E1 | the pass walk above with **existing** kernels where they exist; NVFP4 via `nvfp4_gemv` in 8-column chunks, IQ2 via `gemv_sel` 9m slots (stand-in arms) | ~4–6× (IQ2 per-slot re-read ≈ 1.1 ms/tok bounds it) | 1.5 h | engine (Claude) | 1, now |
| G1 | **IQ2_XXS in the grouped int8 GEMM** (`GemmWeight::Iq2Xxs`, file blocks, decode = IQ3_XXS's shape: `iq2xxs_grid` bytes {8,25,43}·sign, scale `2s+1` per 32, `d/8`; `m16n8k32`; `|isum| ≤ 4·31·32·43·127 < 2^31`) + FAIL-first gate vs dequant×q8 reference | routed+shared → read once a tile: E1 → ≥ 10× | 1–1.5 h | GLM-5.3 (spec engine) | 1, now |
| G2 | **NVFP4 → Q8_0Plane requant** per pass into one site-sized scratch (d = max(s0,s1)/16 exact in f16, the larger half exact `q = 8·kv`, the smaller `round(8·kv·r)`; tensor scale at the output), then `gemm32` | NVFP4 0.6 → ~0.15 ms/tok: ~1500 tok/s | 1 h | GLM-5.3 | 1, now |
| G3 | Bolt `Routers::enqueue_rows` + `RouteOut` m-row buffers (wraps `enqueue_router_rows`) | needed by E1 | 30 min | GLM-5.3 | 1, now |
| G4 | gate: `gate_nemotron_h_moe_e2e` prefill clauses ported from `gate_nemotron_e2e` (`prefill_is_mainlines`, `prefill_then_step_is_mainlines`, `prefill_edges`, small ctx) on the fixture tier | correctness | 45 min | models / GLM | 1 |
| E2 | GROUP_16 tile prefill flash | ~0.1 ms/tok at P=512, more at 4096 | 2 h | engine | 2 |
| E3 | chunked Mamba-2 scan | ≤ 0.1 ms/tok [to measure] | 3 h | engine | 2 |
| E4 | **verify forward** = `Rows` for Bolt (m ≤ 8, bit-exact arm: NVFP4 kgemv m cols, Q8_0 gemv m cols, gemv_sel 9m slots) + Mamba-2 rollback: **double-buffered state + replay of the accepted k rows' scan on a partial accept** (one 4 MB/layer write a pass, not m lanes × 4 MB = 1.3 GB at m=8); the conv ring already holds `PASS_ROWS` = 8 extra inputs (`runtime/stores.rs`) | spec decode's verify | 3 h | engine | 2 |
| E5 | prefill under offload: experts past the card's prefix — m ≥ `STREAM_MIN_P` stream their stacks up (PCIe 26 GB/s; per pass of 512 at 4 GB spilled ≈ 0.3 ms/tok) and run in the same GEMM; small m → host union (`experts_union_into`, needs a relu² non-gated `HostLayer`, agent map) | offload tier pp | 4 h+ | engine | 3 |

Stage-1 critical path: G1 ∥ G2 ∥ G3 (GLM, concurrent) while engine writes E1 against today's kernels; swap the arms as
each lands. **Blocker: ws unreachable (§0) — no build on a card, no gate, no Evidence until the route returns.** Static tier
(`cargo oxide check`, mac-static equivalents) runs here.

### 5.4 Verify forward shares the walk
`Rows::enqueue_rows` for Bolt is the same layer walk with the gemv arm (m ≤ `MAX_PASS_ROWS`), every row into a head
(`RowHeads::One`), exactly qwen3moe `wide.rs`'s split (gemv arm ≤ 8 bit-equal to steps, wide arm above).

## 6. Models — gate, n-gram spec decode, mamba2 rollback (models)

Models' section: `/workspace/agent-notes/bolt-speed-plan-models.md` (M1 prefill gate spec
`/workspace/agent-notes/bolt-prefill-gate.md`, M2 lookup spec decode, M3 Mamba-2 rollback, ranking). Bolt has no
MTP head (`model/src/arch/hybrid.rs:269` `no_nextn`; the real file loads).

**Open between engine §5.3 E4 and models M3 (decide before code):**
- Engine proposes a double-buffered scan state plus a replay of the accepted rows.
- Models proposes `seqstate::Lanes` (Body35's way).
- Lane bytes are 40 × 8192 × 128 × 4 B = 168 MB a lane: 336 MB at L = 2 and 1.34 GB at L = 8.
- On the 3090, the whole-card resident file is 22.7 GiB of 24 GiB, so L = 2 lanes fit and L = 8 does not.
- Perf's view:
  - Lanes at L = 2 for the first lookup verify (no replay cost on a partial accept).
  - Double-buffer + replay only if m > 2 drafts arrive.

**Models resolution (19:0x KST): m − 1 lanes, no replay.**
- `mamba2_steps` (the pass engine's verify runs at u = m) stores the state after token t into `lanes[t]` for
  t < m − 1. The final in-place store is unchanged.
- Commit k = m: nothing. Commit k < m: one d2d copy `lanes[k−1]` → state per Mamba layer.
- Lane bytes are (m − 1) × 168 MB: 168 MB at m = 2, the 3090 fits it.
- No lane word, no graph change, and no rows run again.
- GLM round `bolt-m2lanes` is implementing it (kernel arg, refusals, gate_mamba2 clause 5). Engine threads
  `Option<lanes>` through `enqueue_pass` (agreed 19:0x).

**α measured offline on perf's Bolt toktapes** (`tools/ref/draft-accept.py`, lookup-recent):
- Input: the four tapes' prompt + reasoning + content, re-tokenized with Bolt's own tokenizer. The tapes are Bolt's
  greedy output, so this is the engine's acceptance up to re-tokenization.

| tape | ids | proposed/positions q | accepted/proposed a | α = acc/positions |
|---|---:|---:|---:|---:|
| chat | 474 | 0.626 | 0.291 | 0.182 |
| code | 1209 | 0.779 | 0.471 | 0.367 |
| korean | 545 | 0.710 | 0.601 | 0.426 |
| reasoning | 106 | 0.514 | 0.556 | 0.286 |

- Speedup [derived], with a 2-row verify at 1.1× a step (perf §3) and lookup proposing on a fraction q of positions:
  (1 + q·a) / (1 + 0.1·q) ⇒ chat ×1.11, code ×1.27, korean ×1.33, reasoning ×1.22 decode.
- Before the lanes' commit copy (≤ 168 MB d2d on a partial accept, ≈0.4 ms at the A6000's bandwidth [derived]).
- The bench confirms it once the verify is wired.

## 7. Final target

- **Prefill:** pp512 / pp4096 ≥ llama.cpp on both cards (A6000 ≥ 1549 / 1604; 3090 ≥ 1365 / 1385).
  - Stage-1 floor: ≥ 820 at P = 512 (engine's 10×).
  - Stretch: ~3k (compute floor at ~30 % mma).
- **Decode, whole card:** iq2opt + prologue + folds, 81.99 → ~88–90 tok/s at depth 6 [derived, +7–10 %]; the
  card's ceiling is ~133.
  - Lookup spec decode on code/edit workloads stacks on top.
- **Offload (bloomery's core):** Bolt at ≥ 60 tok/s decode with every routed expert on the host. The 3090 or a
  12–16 GB budget must beat llama.cpp's `--n-cpu-moe` at the same budget.
  - Not landable today: no host tier for Bolt.
- **By 20:40:** realistically iq2opt (if ws returns), E1 code, and GLM's G1/G2/G3 code with fixture-tier gates. Spec
  decode and the offload tier are tomorrow's work, said now rather than at 20:40.


### Engine checkpoint 19:10 KST
- `bolt-prefill` 5e5d54e3: Bolt prompt API (`prefill`/`prefill_with`, `PASS_ROWS`) on steps; the pass body in progress
  (gemv-chunk arm = also the verify arm; models builds Rows/rollback on it).
- GLM rounds live: `bolt-iq2xxs-gemm` (G1), `bolt-nvfp4-q8-requant` (G2), `bolt-prefill-gate` (G4). G3 folded into E1.
- Local `cargo check` of device crates works: `source ~/.local/cuda/env.sh`.
- 19:05 update: `bolt-prefill` 5a763150 = E1 pass body compiles (layer-major passes of 448: embed rows, Mamba
  enqueue_steps over the pass, K/V plane append + flash GROUP_16 runs of 8, router rows, routed/shared via `_sel` over
  9m / m slots, NVFP4/Q8_0 8-col gemv runs; last row's routing copied to each layer's RouteOut). 177f5ae6 = Bolt serve
  seat + serve gate use `prefill`. Fixture e2e+serve gates queued (box). Card-unmeasured; derived E1 band 300–700 pp
  tok/s @512 A6000 (derived). Asked perf for the pp arm. Models has the SHA for verify (Mamba lanes pending).

### Engine checkpoint 19:40 KST (written 19:25)
- `bolt-prefill` 78805e48 (pushed, not merged), on top of E1 5a763150:
  - 26015583 Q8_0 sites (latent pair, attention) past 8 rows → one gemm32 over the pass (dense one-expert route).
  - 215ff174 NVFP4 sites (ssm_in/ssm_out) → Q8_0 twin requantized on the host at load (decode × tensor scale,
    quantize_row_q8_0 rule), gemm32 past 8 rows. Replaces G2's device requant (G2 round stopped). +~4.7 GB resident
    (derived from 2.47 GB NVFP4 at 4.5→8.5 bits).
  - experts: grouped GEMM where `GemmWeight::from_ggml(IQ2_XXS)` is Ok (G1 lands it), else `_sel` — so rebasing
    onto G1 turns it on with no further edit.
  - 177f5ae6/6670f374: serve seat, serve gate, generate_qwen3moe run the Bolt prompt through `prefill`.
  - 78805e48: gate_nemotron_h_moe_e2e gains prefill_near_mainlines(+_real), prefill_then_step_near_mainlines(+_real).
- Gates: fixture e2e + serve green at 177f5ae6 (box, 856 s). Fixture + real e2e at 78805e48 queued; real Bolt
  step-vs-auto greedy ids queued (A6000).
- Perf measuring 215ff174 pp @ 512/4096 vs main (card bolt-e1-pp-a6000, band 500–1000 derived).
- G4 round failed (rc 72, degenerate output) → clauses written by engine. G1 (IQ2_XXS grouped GEMM) near its report.
- Not done / at risk for 20:40: G1 GPU proof + ptx pins; FAIL-first runs of the new clauses (not run yet);
  verify forward (models, on 5a763150).
- 19:45 update: 215ff174..78805e48 failed the real load (`insert_derived` wants a `derived.` name; box s1664) —
  fixed in bolt-prefill 39fff4df; the fixture/real/mutant runs queued at 78805e48 were stopped (they would fail at
  load). G1 committed: bolt-g1-iq2gemm d63973f5 (host-side FAIL-first by the round; card gate_gemm queued; ptx pins
  for gemm_iq2xxs not written — need the box JIT scan). Integrated branch bolt-prefill-g1 f7063282 = bolt-prefill
  rebased onto G1 (no merge). Queued: real load + step-vs-auto ids on both trees, real-tier e2e on 39fff4df.
  G1's report: grouped.rs `win` reads one word past a row's end when the row is a whole number of words (IQ3_XXS
  already, IQ2_XXS at K 1024/2048/2560 too) — past the buffer on a stack's last row; pre-existing, open item.
- 19:52: real Bolt A6000 (box queue): 39fff4df and bolt-prefill-g1 f7063282 both load (resident 27,414,672,948 B) and
  `--prefill auto` (729 ids, rows:448 x2) gives the 24 greedy ids `--prefill step` gives, on both. Perf's A/B
  queued: main / 39fff4df / f7063282 @ P=512, 4096. Real-tier e2e queued on both; fixture e2e + FAIL-first mutant
  (head reads row 0) queued at 39fff4df; real-tier gate_gemm queued on bolt-g1-iq2gemm d63973f5.

### Engine checkpoint 20:10 KST (written 20:08)
- Real-tier `gate_nemotron_h_moe_e2e` at bolt-prefill 39fff4df (box, A6000, 806 s, rc 1): every clause PASS except
  `prefill_then_step_near_mainlines_real`, 9 of 10 prompts PASS, prose64 FAIL: cosine 0.997937 against the floor
  0.999 (max |Δ| 2.5779). The step path itself gives 0.999401 (max |Δ| 0.6988) on that prompt, and the same argmax 2082.
  `prefill_near_mainlines_real` 10 of 10 PASS (prose64 0.999875; prose512 0.999992; code4096 0.999937).
  routing/logits/top1 mainline clauses PASS.
  - Triage so far (reasoning, not yet measured): it is not a pass-shape bug. prose512 runs 448 + 63 rows then a step
    and passes at 0.999993; prose255/256/257/1024 and code5/29/4096 (passes then a step) pass. prose64 is the
    near-tie prompt: its step path already crosses mainline's pick at the cut in 3 of 40 MoE layers (3, 23, 61;
    worst gap/band 0.008–0.057). Likely another crossing on the pass + step state, but unproven. A diag run is queued
    (local branch bolt-prefill-diag 0846c15d, not for landing): it prints the layers keeping mainline's experts, run
    against step. The floor is NOT relaxed; the clause stays red in the PR until the diag decides it.
  - ~~The real-tier e2e now takes 806 s, past 0.75× of its 900 s bound~~ — wrong: the binary ran 131 s; 806 s is the
    item's wall incl. box queue wait + build (gate-batch's budget check reads that wall, so a rerun needs --over-budget-ok).
- Queue (A6000, congested): gate_gemm real tier on G1 d63973f5 (s1694, building), fixture e2e 39fff4df (s1696),
  FAIL-first mutant (s1697), real-tier e2e on f7063282 (s1698), perf A/B s1700, diag. Probably not all land by 20:40.

### Engine final 20:40 KST (posted 20:10; pending rows updated as they land)
Landed (pushed as draft PRs, rebase-only, nothing merged):
- #327 bolt-g1-iq2gemm d63973f5: grouped GEMM IQ2_XXS entry (`gemm_iq2xxs`). Card gate_gemm: PASS (real tier, A6000, 70 IQ2_XXS lines, 0 FAIL).
- #328 bolt-prefill-g1 f7063282 (= bolt-prefill 39fff4df rebased on #327): Bolt batched prefill (layer-major passes of
  448; gemm32 at Q8_0 sites, Q8_0 twin of NVFP4 sites, grouped IQ2_XXS experts), wired into the serve seat, the serve
  gate and generate_qwen3moe. Real Bolt A6000: loads, and `--prefill auto` gives the step path's 24 greedy ids on both trees.
- Real-tier e2e 39fff4df: all PASS but prefill_then_step_near_mainlines_real prose64 (0.997937 < 0.999). Suspected near-tie crossing, not shown; diag run queued (local bolt-prefill-diag 0846c15d). Floor not relaxed. #328's head f7063282 is green (below); the 39fff4df red stays open until the diag decides it.
- Real-tier e2e f7063282 (#328's head): PASSED, every clause, prose64 prefill+step 0.999466 (105 s binary). Fixture e2e: PASSED (4 of 10 gated per prompt-path clause; prose64 crossed). FAIL-first mutant (head reads row 0): prefill_near_mainlines red (gated cosines 0.06–0.11), as it should be.
- Speed (perf, s1700, A6000, one lease, 2 rounds): pp tok/s main 81.08 @512 / 80.44 @4096; 39fff4df 330.48 / 327.07; f7063282 927.19 / 971.66 (11.4× / 12.1× main, target ≥820 met). Decode unchanged (nodes 1084). 39fff4df missed its derived 500–1000: the per-slot `_sel` term is larger than derived.
Dropped / not landed by 20:40:
- PTX pins for gemm_iq2xxs and the prefill bins (need the box JIT scan; `gate-ptx-spill` will be red on them).
- Card FAIL-first for G1's decode clauses (host-side FAIL-first only).
- A FAIL-first mutant for prefill_then_step_near_mainlines (state left for the step, e.g. K/V append position or the Mamba state) — not run.
- Verify forward (models' track, on the multi-column arm; not started in engine).
- ~~The real-tier e2e budget (806 s of 900)~~ — not an item: the binary runs 105–131 s.
- G1's report: grouped.rs `win` reads one word past the last row when a row is a whole number of words (pre-existing).
After the deadline (merge-tree dry run at 20:09: both stacks apply cleanly onto 5a735ee9): rebase order glm-m2lanes → bolt-g1-iq2gemm → bolt-prefill → verify; never merge.
- 20:14: G1 card gate PASS: real-tier `gate_gemm` at d63973f5 (A6000, 158 s, rc 0), 70 IQ2_XXS lines PASS, 0 FAIL
  (#327 body updated). Fixture e2e at 39fff4df PASSED (10 clauses ran, 5 left to the real tier). Both prompt-path
  clauses: 4 of 10 prompts gated, PASS. prose64 is the outlier on the fixture too: the prefill path's cosine is 0.819,
  the step path's 0.9995, printed as crossed, not gated. Same prompt as the real-tier red, so the diag covers both.
- 20:16: FAIL-first mutant (head reads the pass's row 0) at 39fff4df, fixture: prefill_near_mainlines FAIL 4 of 10 gated (cosine 0.06–0.11); prefill_then_step stays green (its last row is a step). #328 body updated.
- 20:18: real-tier e2e on f7063282 (#328 head) PASSED, every clause: prefill_near 10/10, prefill_then_step 9/9 gated
  (prose64 0.999466 vs 0.997937 at 39fff4df with `_sel` experts). Corrected the 806 s budget claim (struck above).
- 20:25: perf runs the next-stage nsys (f7063282, generate_qwen3moe, 4096-id LCG prompt, card bolt-g1-nsys-a6000, A6000); per-kernel and by-site split to engine when it lands, possibly after 20:40. Engine prices the next lever from it before any code.

### Engine 20:40 report (written 20:33; per the user's priorities: #326 → #328 → offload decode → n-gram opt-in)
IN (pushed; nothing merged by engine):
- (2) #328 bolt-prefill-g1, now 017a4d6f (on #327 47fee935). Measured (perf s1700, A6000, one lease, 2 rounds):
  pp 927.19 @512 / 971.66 @4096 tok/s vs main 81.08 / 80.44 (11.4× / 12.1×); decode unchanged (80.8–81.1 @ d512).
  Gates on f7063282: real-tier e2e PASS (every clause), fixture e2e PASS, the FAIL-first mutant red as it should be.
  G1 real-tier gate_gemm PASS. CI contrib-check t0 red=4 at f7063282: fmt + check-arch fixed in 017a4d6f
  (spawn-allow row for derive_q8's load-time scoped workers, PENDING triage); still red: check-ptx-names (2 unpinned
  entries; needs the box JIT scan) and check-unsafe (gpu 1865 > pin 1863). So it is NOT ready yet.
- (1) #326 (perf's IQ2 decode): its head 8a16e7a4 passes `cargo fmt --check` locally; CI contrib-check/review
  were still running at 20:30. It merges once green (perf's PR; engine has not merged it).
- 39fff4df prose64 red, root-caused as far as the diag goes: the 63-row pass then a step keeps mainline's experts in
  25 of 40 MoE layers; the step path and the full prefill keep 37. The routes diverge, and the cosine falls with them.
  At f7063282 (grouped experts) the same clause passes at 0.999466. Whether the first diverging layer is a near-tie
  crossing or a fault is not shown. Open, on the base tree only.
LEFT (not by 20:40):
- #328 to ready: the ptx-shapes pins for gemm_iq2xxs plus the second entry (box JIT scan), and the unsafe-ratchet +2
  with a reason. Then pp to ≥1549 @512 A6000 (llama.cpp's line), priced from perf's nsys (bolt-g1-nsys-a6000).
- The 3090 row: Bolt (27.4 GB resident) does not fit 24 GB, so the 3090 prefill and the offload decode both need the
  host expert tier. Spec in progress (engine), with GLM rounds to implement. No 3090 number exists yet.
- (3) Offload decode vs llama.cpp --n-cpu-moe on the 3090: the same tier, not started.
- (4) n-gram spec opt-in: not engine's track this round.
- The FAIL-first mutant for prefill_then_step; card FAIL-first for G1; verify forward (models).

### Engine checkpoint 21:00, written 20:51 (criterion changed to the goal: #328 ships at pp ≥ llama.cpp on both cards, offload measured)
- PRs: #326 (perf) CI green, Shingler review: CHANGES_REQUESTED/dismissed, labels gpu+needs-gpu, blocked.
  #327 G1 c0d5fb3d (rebased on main 85511826; unsafe pin +2 dated; gemm_iq2xxs pinned 0/0 after gate-ptx-spill went
  red on exactly it in generate_ds41 + gate_e2e, 3090 sm_86). #328 19863679 on #327: twins only when the room holds
  them (3090 OOM'd at load with twins: perf s1720), spawn-allow row, fmt. Both still draft, CI rerunning.
- Measured (perf): A6000 whole card, pp tok/s: main 81.08 @512 / 80.44 @4096; #328 (f7063282) 927.19 / 971.66;
  llama.cpp 1535 / 1604. 3090: main 81.33 pp, 81.23 decode @ d512; #328 head queued (s1729), A6000 rerun s1730.
  No 3090 #328 number yet; no offload number (Bolt has no host tier yet).
- Gates at f7063282: real e2e PASS, fixture PASS, FAIL-first red as it should be, G1 gate_gemm PASS. Head 19863679
  real e2e queued on the A6000 and the 3090 (the 3090 runs the no-twin path).
- In flight toward the goal:
  G5 (GLM, bolt-g5-nvfp4-gemm32 on 8dc9ccc0): NVFP4 gemm32 arm from the resident planes, no twins; the 3090 needs it.
  models: prefill_edges clause (pass boundaries, GLM bolt-pf-edges), H1 host relu² union.
  perf: pp reruns, A6000 nsys at P=4096 (by-site split prices the 927 → 1549 gap).
  engine next: Bolt card boundary for the host tier (plan item 7), spec to models before a round.
- 21:03: perf at #328 head 19863679, one lease per card, 2 rounds, pp tok/s @512/@4096:
  A6000 938.26/989.04 vs llama.cpp 1549.84/1603.37 (0.605×/0.617×);
  3090 495.62/501.45 vs llama.cpp 1354.70/1374.99 (0.366×), no twins there.
  Decode unchanged. In flight: G5 (NVFP4 gemm32, GLM), B1 (handoff_9 + weighted_sum_card, GLM), B2 spec (Plan agent),
  H1 (models), prefill_edges (models), nsys s1733/s1734 (perf).
- 21:06: real-tier e2e at #328 head 19863679 on the 3090 (no twins; 116 s binary): rc 1. Every clause PASS except
  both prompt-path clauses, and only prose64 is red there: prefill 0.998342, prefill+step 0.997558 (floor 0.999); the
  step path is 0.999401. The other 9 prompts are ≥0.99981. prose64 has been red on 2 of 3 prompt-path builds.
  Proposed to models (gate owner): classify the prompt path's first divergent layer with the gate's own crossing band
  (Crossed vs Other). Other = FAIL. Whether to un-gate a Crossed prompt below the floor is a gate-rule change for
  models and the user, not a pass. Floor unchanged. The A6000 head run is still queued.
- 21:27: real-tier e2e at #328 head 19863679:
  - A6000 PASS rc 0 (107 s), prose64 0.999920 / 0.999466.
  - 3090 rc 1, prose64 only, 0.998342 / 0.997558.
  - #328's body has both. CI on #328 and on #327 (c0d5fb3d) is green.
- #327 at its head c0d5fb3d: gate_gemm real tier PASS on the A6000 (143 s, IQ2_XXS 69 PASS / 0 FAIL). The card FAIL-first runs of scale-2ls and sign-63 are queued (mutant-run). After them, #327 goes ready.
- Rounds:
  - M1, the Mamba conv pre-pass plus T-token scan stage, bit for bit: bolt-m1-mamba-rows.
  - B2, Bolt's plan with Experts::{Card, Placed, Host}: bolt-b2-place.
  - B1 hung at 21:12 on the provider side and was resumed in its session at 21:27.
  - G5 is running.
  - B3 spec: a Plan agent is drafting it.
- perf: flash split (FlashGqaPrefill::enqueue_split), derived −0.85 s A6000 / −1.0 s 3090 at P=4096; engine wires it at prefill.rs:995.
- Derived at P=4096 with M1, the flash split and G5 together: A6000 ≈ 2.3 s (≈ 1780 tok/s), 3090 ≈ 2.8–3.2 s (≈ 1290–1460).
  The 3090 is the tight one.
