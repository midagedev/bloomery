> 원문 보고(영문) — 읽기 전용 유도 라운드 `b1d`(opus, 2026-09-23 새벽). 리드의 설계 정리는 [`../../v41-placement.md`](../../v41-placement.md)다. 이 디렉터리의 스크립트는 여기서 `python3 inv.py && python3 derive.py && python3 final.py` 순서로 다시 돈다(입력은 `../../v41-inventory.md`). 본문의 스크래치 경로는 라운드 당시의 위치다.

# B1d report: V4.1-Flash placement on this machine (design only)

Labels: **[m]** measured (source given), **[d]** derived (arithmetic given or from the named script), **[asm]** assumption. Every tok/s is for n=1 (a single stream) with ctx_max 32,768 and the engram lookup kept off the step thread, unless a row says otherwise.

Compliance: I edited no file and changed no git state. I took no lease, ran no benchmark, no GPU gate and no `box.sh`. On the box I ran only read-only commands over `ssh ws`: `nvidia-smi --query-gpu=…`, `free -b`, `numactl -H`, and two read-only NVML queries that are not in your literal list (`nvidia-smi topo -p2p r` and `nvidia-smi topo -m`). The scripts are in `/private/tmp/claude-501/-Users-hckim-repo-bloomery/e5714b41-3850-42be-a722-79e1f225c3b5/scratchpad/b1d/`:
- `inv.py` parses `docs/v41-inventory.md` into `inv.json`.
- `derive.py` gives per-type and per-role bytes.
- `v2lite.py` checks the resident formula against the V2-Lite records.
- `v2lite_step.py` gives the A6000 bandwidth anchor.
- `place.py`, `spread.py`, `final.py` and `tables.py` do the placements and tok/s.

## 1. Predictions vs outcomes

**No prediction was written down before the arithmetic.** What I expected after reading the inputs, before running any placement script:
- **(a) fits:** about 42 GB of the A6000 left for experts after about 8.5 GB of dense weights.
- **(a) speed:** about 25 tok/s with the serial executor.
- **(b) speed:** 4–8 % faster than (a).
- **Separating term:** `bytes_host / BW_host`, through the VRAM left for experts.

What came out:
- **(a) fits.** 40.57 GB of experts plus a 1 GiB margin.
- **(a) serial:** 24.5 tok/s, as expected.
- **(b) vs (a):** +8.8 % serial, +10.6 % with the R1 overlap. That is more than I expected, because I missed a second term: half of the dense bytes move to the 3090's faster memory (936 vs 768 GB/s peak), worth about 1.4 ms per token.
- **Not predicted at all:** with the R1 overlap, how experts are spread across layers matters about as much as how much VRAM they get. Spreading experts over all layers instead of filling whole layers is worth +3.7 % in (a) and +6.3 % in (b).
- **The larger finding:** whether we beat ik depends on the GPU bandwidth term, not on placement (see §5).

## 2. Sections 1–8

### §1 Resident bytes per type

The file → resident formula for each `DevWeight` format (`crates/gpu/src/weights.rs`); `rb` = type_size × k/256:

| format | resident bytes | file bytes |
|---|---|---|
| KQuant (Q3_K/Q4_K/Q6_K) | upload: `4·⌈rb·rows/4⌉` (`:301`, refused at `:302` unless rows divides that word count); `resident_size`: `rb·4⌈rows/4⌉` (`:251`) | `rb·rows` |
| Q5_0 | `rows·(1024·⌈k/1024⌉ + 4·k/32)` (`:243`, `q5.rs:925-927`: one byte per 5-bit code, windows of 1024 values, f32 scale) | `rows·22k/32` |
| Q5_1 | `rows·(1024·⌈k/1024⌉ + 8·k/32)` (f32 d and m) | `rows·24k/32` |
| Q8_0 planes (`Q8_0`, `Q8_0Derived`) | `rows·(k + 4k/32)` = 36/34 × file | `rows·34k/32` |
| F32 | = file | |

**Checked against the recorded V2-Lite numbers: no disagreement.**
- `plan.md:406`: Q5_0 is 26 tensors × 131,072 rows × 2,224 B = **7,579,107,328** B resident, against 968 B/row = **3,298,820,096** B file [d]. Both match. The padding per row is 1024·2 − 1408 = 640 B [d].
- `plan-ledger.md:226` says file 8.12 GB → resident 12.45 GB. The formula gives file 8,122,610,688 and resident 12,446,610,432 (files 12,414,759,936 plus derived q_nope2 31,850,496) [d]. Match.
- `plan-ledger.md:219` says "상주 12.46 GB". That is the stage total, including KV (27×512×576×2 = 15.9 MB) and scratch. Consistent at the record's precision.
- `plan-ledger.md:211` records 729,072,372 B. That print is `Stage::resident_bytes()`, i.e. weights + KV + scratch (`c9f5ace:crates/gpu-gates/src/bin/gate_p8b.rs:74`, `c9f5ace:crates/gpu/src/model.rs:796-801`, CTX_MAX 64). The formula gives 728,646,656 B of weights, KV is 73,728 B, so the implied V2-Lite one-layer scratch is **351,988 B** [d, by subtraction].

**V4.1, per type:**

| type | n | file B | today | cheapest format an existing kernel reads | resident B | alternative |
|---|---:|---:|---|---|---:|---|
| f32 | 449 | 2,097,600 | F32 | F32 | 2,097,600 | — |
| q3_K | 179 | 155,738,992,640 | KQuant | KQuant | = file | — |
| q4_K | 38 | 96,825,507,840 | KQuant | KQuant (`_sel` missing, §6) | = file | — |
| q6_K | 1 | 542,976,000 | KQuant | KQuant | = file | — |
| q8_0, non-engram | 330 | 7,264,010,240 | **refused** at `gguf/src/lib.rs:212-219` (tag 8 → `Unknown`), then `weights.rs:253/356` | Q8_0 planes, read by `q8_0_gemv` (`q8f32.rs:467`, f32 activations, M ≤ 8) | 7,691,304,960 (+427,294,720) | f16 scale plane: = file (kernel change) |
| q8_0, `engram_embd` | 2 | 208,902,215,200 | same | never on a GPU; rows are gathered | — | — |
| bf16 | 45 | 1,481,277,440 | **refused** at `lib.rs:212` (tag 30) | decode to F32, read by `f32_gemv` (`q8f32.rs:402`) | router 314,572,800; `engram_{k,q}` 327,680; `token_embd` on the host | bf16 variant of `f32_gemv`: = file |
| q5_K | 2 | 6,228,541,440 | **refused**: `quant.rs:198` (no dequant), `weights.rs:253/356` | Q5_1 layout. Values are exact (d·sc and dmin·m fit in f32), but `q5_1_gemv` has no `_sel` form, so no kernel exists for routed use | 14,344,519,680 (2.30×) | native q5_K kernel (missing): = file; f32: 36,238,786,560 |

### §2 The rows % 4 trap

Row bytes are always even here (type sizes 110, 144, 210), so `rb` is 0 or 2 mod 4. The upload and `resident_size` disagree exactly when:

| case | upload | `resident_size` | effect |
|---|---|---|---|
| `rows % 4 ≠ 0`, `rb ≡ 0 (mod 4)` | succeeds, holds `rb·rows` | overstates by `rb·(4⌈rows/4⌉ − rows)` | `gate_p10` goes red on "resident totals disagree" |
| `rb ≡ 2 (mod 4)`, `rows = 1` | holds `rb + 2` | says `4·rb` | numbers disagree |
| `rb ≡ 2 (mod 4)`, `rows > 1` | **refused** at `:302` | still returns a size | census counts a tensor the loader cannot load. This is the stricter latent trap; it is Q3_K or Q6_K with odd k/256 |

**V4.1 has zero such tensors (V2-Lite also zero).** Every K-quant has rows % 4 = 0 and rb % 4 = 0:
- q3_K: k/256 is 20, 80 or 2.
- q4_K: rb = 1,296.
- q6_K: 129,280 rows × 4,200 B.

Uploading expert slices (2,304 or 5,120 rows per expert) cannot trigger it either. So the `plan.md:406` claim "V4.1 형상에서 함정이다" does not hold.

Fix (XS): make `resident_size` use the upload's own arithmetic, and return `None` wherever `:302` refuses.

### §3 Per-token bytes by role [d, from the inventory]

| role | tensors | per-token read, file format | per-token GPU read, resident format |
|---|---:|---:|---:|
| attention (q_a/q_b/kv/o_a/o_b/norms/sinks; compressor at 2,8,14,20; indexer at 8 layers) | 395 | 5,435,412,480 | 5,754,572,800 |
| hyper-connection (`hc_*`) | 240 | 16,904,640 | 16,904,640 |
| router (`ffn_gate_inp` bf16 + `exp_probs_b`) | 80 | 157,347,840 | 314,634,240 |
| `ffn_norm` | 40 | 819,200 | 819,200 |
| shared experts (q8_0) | 120 | 1,504,051,200 | 1,592,524,800 |
| **engram dense** (`engram_wkv` ×2 is a full 6144→25600 gemv per token; `engram_{k,q}`) | 6 | **334,397,440** | 354,222,080 |
| head (`output` q6_K + norm) | 2 | 542,996,480 | 542,996,480 |
| `token_embd` (one bf16 row) | 1 | 10,240 | host gather |
| routed experts, 6 of 384 per layer (q4_K layers 100,638,720 each; q5_K layers 0–1 109,486,080 each) | 120 | 4,043,243,520 | depends on placement |
| engram table rows (48 × 272 B) | 2 | 13,056 | — |
| `exp_probs_b_vl` (unused in text decode) | 40 | 0 | — |
| **total** | 1046 | **12,035,196,096** | |

- Dense per token = **7,991,939,520 B**. The roofline's 7,657,593,280 leaves out the engram dense tensors; the per-token total goes from 11.701 to 12.035 GB (+2.9 %). See §4 of the report.
- KV read per token at depth D is in §4 below.

### §4 KV and scratch

**Who owns what** (`v41-ops-report.md:386-400`, `:195`):
- Every one of the 40 layers has its own 128-row ring of 512-dim rows: 131,072 B per layer, 5,242,880 B in total.
- Compressed rows and 128-dim index keys are owned by layers 2, 8, 14 and 20. Layers 3–7 alias 2, 9–13 alias 8, 15–19 alias 14, and 21–39 alias 20.
- Top-k ids are issued by layers 2, 8, 14, 20, 24, 28, 32 and 36. The layers after each issuer reuse its ids until the next issuer.

**Size.** Allocated at ctx_max C [d, f16]: `5,242,880 + 3·⌈C/2⌉·1,280 + C·1,280 + 28,672` B. The last term is the compressor state rings.

| depth D | KV occupied (all layers) | KV read per token |
|---:|---:|---:|
| 6 | 293,632 | 433,920 |
| 1,024 | 8,548,352 | 26,869,760 |
| 4,096 | 18,378,752 | 31,981,568 |
| 32,768 | 110,129,152 | 79,691,776 |
| 1,048,576 | 3,360,714,752 | 1,769,996,288 |

- The read per token is: layers 2–39 read (128 + min(visible, 512)) × 1,024 B; layers 0–1 read the window only; the indexer scans 1,664 B per cached token.
- 1,048,576 is the reference config's `max_position_embeddings` (`v41-ops-report.md:44`), not the GGUF header, which was not read (see report §3).
- At 1M depth the indexer scan is about 9 ms per token at 195 GB/s [d]. That depth becomes indexer-bound.
- In (b), at ctx 32k: A6000 layers 0–19 hold 65,560,576 B and 3090 layers 20–39 hold 44,568,576 B. At 1M: 2.02 GB and 1.34 GB.

**Scratch, context, reserve:**
- m=1 scratch: **64 MiB per card [asm]**. V2-Lite's one-layer stage measured 351,988 B [d]. The prefill arena is not determined: mainline's compute buffer at `-ub 512` was 4.3 GB and 2.4 GB [m, rig-log `configs/v41-serve.sh:28-29`, mainline].
- CUDA context: nothing is recorded for our engine, so **512 MiB per card [asm]**.
- Driver reserve [m, `nvidia-smi` 2026-09-23]: 548 MiB on the A6000, 400 MiB on the 3090.

### §5 Placement options

**Expert rule (both options): spread, by id prefix.**
- Layer l keeps experts [0, n_l) in VRAM as one flat row slice of the stack. n_l is equal across a card's eligible layers.
- Layers 0–1 (q5_K) stay entirely on the host.
- The existing `_sel` contract, "an id ≥ n_experts leaves the slot untouched" (`lib.rs:1262-1265`, `moe_fused.rs:196-198`), already implements the GPU/host split when n_experts = n_l. No remap table is needed.
- But "untouched" means **stale**: the combine must mask by id, not rely on zeros.
- Why spread: it wins only with R1 (concurrent legs inside the graph). With the serial executor it ties whole-layer filling (−0.2 %) [d].
- What would change the rule: a router-id histogram per layer (top-6 ids over a real stream). If routing is skewed, the resident set becomes the hottest experts per layer, which needs an id→slot remap table the kernels must read.
- Balance limit: the GPU leg stays hidden under the host leg while n_l ≤ about 281 [d at 575 and 147.7 GB/s]. Every plan below stays under it.

**Headline bandwidths:**
- A6000 575 GB/s [asm: the plan's `BW` 700 (`plan.md:67`, calibrated on the 3090) × 768/936].
- 3090 700 GB/s [m-calibrated, `plan.md:67`].
- Host 147.7 GB/s [m, `plan.md:472`].

**(a) A6000 + DDR4 (3090 unused)**

| device | role | resident GB | read/token GB | ms/token |
|---|---|---:|---:|---:|
| A6000 | attention ×40 (planes) | 5.755 | 5.755 | 10.01 |
| A6000 | shared experts ×40 | 1.593 | 1.593 | 2.77 |
| A6000 | router (f32) + hc + ffn_norm | 0.332 | 0.332 | 0.58 |
| A6000 | engram dense (layers 1, 14) | 0.354 | 0.354 | 0.62 |
| A6000 | head | 0.543 | 0.543 | 0.94 |
| A6000 | routed experts [0, n_l), n_l = 63–64 on layers 2–39 (2,419 experts) | 40.574 | 0.634 | 1.10 |
| A6000 | KV @ 32k | 0.110 | 0.032 @ D=4096 | 0.16 @ 195 GB/s |
| host (mlock) | routed experts: the rest of 2–39 plus all of 0–1, including both q5_K downs | 218.193 | 3.409 | 23.08 |
| host | `token_embd`, one-row gather | 1.324 | 0.00001 | ~0 |
| NVMe | `engram_embd` ×2 | 208.902 | 0.000013 | 0–0.31 on the step thread |

Budget:
- **A6000:** 50,952,404,992 [m: (49,140 − 548) MiB] − 8,576,674,240 dense − 40,574,177,280 experts − 110,129,152 KV − 67,108,864 scratch − 536,870,912 context = **1,087,444,544 headroom**.
- **Host:** 270,071,001,088 [m: `free -b`] − 218,193,408,000 experts − 1,323,827,200 `token_embd` − 4,294,967,296 B3e cache − 6,694,629,376 OS and other [m, `free -b` "used", as a proxy] = **39,564,169,216**.

Host experts should be served from the mmap of the shard files and locked, not copied. The reason is eviction: without the lock, engram's 51 pages per token push expert pages out of the page cache, and each refault costs 92 µs [m: rig-log 09-22-p].

**(b) 3090 + A6000 + DDR4, cut at layer 20: A6000 runs 0–19, 3090 runs 20–39 + head**

Why cut at 20: it is a group boundary, so no compressed KV, index keys or top-k ids cross. It also puts the larger KV (three ratio-2 groups) on the larger card. Cuts at 8 or 14 are also clean and land within ±1 % [d]. A cut inside 20–39 would need a copy of group 20's cache on both cards, plus 1,280 B more per token across the link.

| device | role | resident GB | read/token GB | ms/token |
|---|---|---:|---:|---:|
| A6000 | attention, shared, router, norms of 0–19 + engram dense | 4.191 | 4.191 | 7.29 |
| A6000 | experts [0, n_l), n_l = 149–150 on 2–19 (2,683) | 45.002 | 0.703 | 1.22 |
| 3090 | attention, shared, router, norms of 20–39 + head | 4.386 | 4.386 | 6.27 |
| 3090 | experts, n_l = 57–58 on 20–39 (1,147) | 19.239 | 0.301 | 0.43 |
| host | the rest of the experts | 194.527 | 3.039 | 20.58 |
| link | h4 streams + `ffn_pre`, A6000 → host → 3090 | — | 81,936 B | ≈0.02 |

The link size is 4 × 5120 × 4 + 16 = 81,936 B (`v41-ops-report.md:213`: "block returns (h_next, ffn_pre)"). There is **no P2P between the two cards** [m: `nvidia-smi topo -p2p r` returns GNS]. The transfer is 3.0 µs + 3.2 µs at 27.1 / 25.5 GB/s [m: rig-log 2026-09-18.md:8, measured on the 3090], plus 2 × 7 µs of sync [asm].

Budget:
- **A6000:** 50,952,404,992 − 4,190,828,000 − 45,002,280,960 − 65,560,576 − 67,108,864 − 536,870,912 = **1,089,755,680**.
- **3090:** 25,350,373,376 [m: (24,576 − 400) MiB] − 4,385,846,240 − 19,238,768,640 − 44,568,576 − 67,108,864 − 536,870,912 = **1,077,210,144**.
- **Host:** **63,231,041,536**.

**(b′) 3090 as an experts-only tier hanging off the host tier.** Dense weights stay on the A6000, the 3090 holds 23.58 GB of experts, and no stage cut is needed. It is 3 % below (b) with R1 and 6 % below (b) serial. This matters for A2-2, because it means A2-2's stage cut is not required to use the 3090.

**Composition terms:**
- Serial = Σ device legs + 1,284 nodes × 0.80 µs + 14 µs per host layer [asm] + KV / 195 GB/s [d: V2-Lite MMA default on the A6000, 127.2 MB / 0.653 ms, `plan.md:24-26`].
- R1 = per layer, max(shared expert + GPU experts, host leg) (`hybrid-engines.md` R1).

tok/s at n=1, headline bandwidths, depth 6 / 1024 / 4096:

| option | serial (expected on B5 day) | R1 (B2's target) |
|---|---|---|
| (a) | 24.58 / 24.49 / 24.48 | 27.16 / 27.06 / 27.04 |
| (b) | 26.74 / 26.65 / 26.63 | 30.05 / 29.92 / 29.90 |
| (b′) | 25.09 / 25.00 / 24.99 | 29.13 / 29.02 / 29.00 |

**Band at D=4096, shown as R1/serial.**

| option | host GB/s | GPU low | GPU mid | GPU high |
|---|---:|---|---|---|
| (a) | 122.6 | 21.27 / 19.02 | 23.98 / 21.94 | 24.81 / 22.87 |
| (a) | 147.7 | 23.64 / 20.89 | 27.04 / 24.48 | 28.11 / 25.64 |
| (a) | 230.4 | 29.37 / 25.27 | 34.84 / 30.71 | 36.63 / 32.55 |
| (b) | 122.6 | 23.02 / 20.15 | 26.56 / 23.94 | 27.51 / 24.99 |
| (b) | 147.7 | 25.44 / 22.02 | 29.90 / 26.63 | 31.10 / 27.94 |
| (b) | 230.4 | 31.11 / 26.30 | 38.25 / 33.15 | 40.32 / 35.20 |

- GPU low: A6000 400 GB/s [d: 1,533,695,856 B per V2-Lite step ÷ (4.3565 − 648 × 0.80 µs) ms] and 3090 433 GB/s [d, from `plan-ledger.md:40`].
- GPU mid: 575 / 700 GB/s (the headline).
- GPU high: 85 % of peak [asm].
- Host 122.6 GB/s: the qdot multi-thread ceiling [m, `plan.md:369`].
- Host 230.4 GB/s: the physical ceiling, which is also where the WKS-36 slope would put the host rate.

**The WKS-36 contradiction (`roofline.md:91`) is still open.**
- If the true host rate is high, the (a)/(b) gap shrinks in ms: 3.5 ms at 147.7 vs 2.6 ms at 230.4 GB/s.
- It stays about 10 % in relative terms, and the GPU dense term becomes the long pole.

**Against the references:**
- ik: 20.4–20.7 tok/s; mainline in the same #2455 table: 21.2.
- Mainline 25.05 tok/s was measured **with the DSpark draft** (`roofline.md:67-71`). The comparable no-draft figures are 17.7 tok/s and 19.9 tok/s. Our file has no DSpark tensors.

**Beating ik depends on the GPU term, not the host term.**
- On B5 day (serial executor, CPU kernels at 122.6 GB/s), the band is 19.0–22.9 tok/s for (a) and 20.2–25.0 tok/s for (b). ik's line sits inside that band.
- If our GPU dense path ran at ik's effective 247 GB/s [d, `roofline.md:77`], we would get 16.1 tok/s (a) and 16.4 tok/s (b).
- The measurement that narrows this most: `q8_0_gemv` at K = 1280 / 2304 / 4096 / 5120 / 6144 / 8192 on the A6000. That kernel carries 7.69 of the 8.58 GB per token of GPU dense reads and has never been timed at these shapes.

**Side rows** (R1, mid bandwidths, D=4096):
- Engram lookup on the step thread, 0.31 ms per token [m, rig-log 09-22-p]: 26.82 (a), 29.63 (b).
- ctx_max 1M: 26.79 (a), 29.61 (b).
- D = 32,768: 26.86 (a), 29.68 (b).

### §6 The two pre-B1 decisions

**(i) bf16: read path vs decoding to f32 at load.** Only the 40 router tensors are gemvs.
- Decoding them to f32 costs +157,286,400 B of VRAM and +157,286,400 B read per token, about +0.27 ms per token at 575 GB/s [d]. This may be smaller or larger in practice: the grid is 48 blocks against 84 SMs.
- `engram_{k,q}` are elementwise gains. Decode them to f32 either way: +163,840 B, cost is noise.
- `token_embd` stays on the host, so it needs no device format. Keeping it off the card saves 1.32 GB of VRAM, worth about 0.14 ms per token.
- Recommendation: a bf16 variant of `f32_gemv`. bf16→f32 is `bits << 16`, which is exact, so its gate is bit identity against `f32_gemv`. It saves about 0.29 ms per token (~0.8 %) and is not a blocker.

**(ii) q5_K on the CPU vs the GPU.** Placement puts layers 0–1 on the host in both options: in VRAM they would save the same host bytes per VRAM byte at best, and 0.63× as much in the Q5_1 layout. So this is **CPU work**. About 97.3 MB of q5_K is read on the host per token [d].

Missing on the CPU side:
- `gguf/src/quant.rs:198`: no `dequant_q5_k`.
- `qdot/src/lib.rs:91-96, 101-120, 123-160, 326-338`: no Q5_K kernel and no Q5_K arms.
- The activation format already exists (`quant.rs:548`: Q5_K → Q8_2_X4).

Missing on the GPU side (optional under this placement): `weights.rs:253/356`, no q5_K kernel, and no `q5_1` `_sel` form (`q5.rs:613`).

**Required for either option:** `q4k_gemv_sel` does not exist. Only `enqueue_gemv_q4k` (`lib.rs:1567`), `q3k_gemv_sel` (`:1283`) and `q5_0_gemv_sel` (`q5.rs:713`) do. Without it, none of the 40–64 GB of VRAM experts can run their down projection.

### §7 What B1's gate should check

Split it into two gates.

**① Host-only placement gate** (reads the 9 shard headers; no GPU, no lease, seconds):
- Prints the table: tensor, layer, role, ggml type, file bytes, device, device format, resident bytes, expert range, bytes read per token, stage.
- Asserts each tensor is placed exactly once.
- Asserts that for each layer, the VRAM expert range and the host expert range together cover 0..384 with no overlap.
- Asserts the per-device totals match this design byte for byte:
  - (a): A6000 8,576,674,240 + 40,574,177,280; host 218,193,408,000.
  - (b): the three totals in §5.
- Asserts resident + KV + scratch + context ≤ usable − margin on every device.
- Asserts the format per type matches the design.
- FAIL-first: bump one n_l by 1, or exceed a card's capacity; either must go red.

**② GPU load gate** (needs the A6000 free of timing runs; roughly 49 GB (a) or 73 GB (b) of uploads; cold read ≥ 38–51 s [d]; well under 30 min):
- `resident_bytes()` per device equals ①'s table.
- `cudaMemGetInfo` shows free ≥ margin on each card. This also yields the first measured context overhead.
- `VmLck` equals the host table.
- `mincore` over the host expert ranges is 100 %.
- Readback check on a sample for each format.

Load-time facts that should be ratchets:
- the table and per-device sums;
- per-layer n_l;
- KV and scratch bytes;
- the 81,936 B crossing;
- bytes read per token;
- the captured node count.

Runtime values that should be recorded, not pinned:
- context overhead;
- load time;
- tok/s and host-leg ms;
- GPU idle fraction (58 % in (a) serial [d]);
- host major faults (must be 0);
- link latency.

### §8 Questions only the user can decide

1. **Serve on the 3090?** (b) gains +10.6 % with R1 and +8.8 % serial [d]. The cost: the card fell off the bus twice on 2026-09-22, and it stops being the gate/build card while serving.
2. **Serving ctx_max: 32k or 1M?** The KV grows from 110 MB to 3.36 GB, which costs about 0.9 % tok/s [d].
3. **Lock 203 GiB (a) or 181 GiB (b) of host RAM for experts?** That rules out running other large jobs on the box at the same time. Leaving it unlocked risks expert pages being evicted.
4. **Approve a long router-id trace?** A trace of about 100k tokens through the ik port at ~20 tok/s is about 1.4 hours of box time, over the 30-minute rule.

## 3. What I could not determine

- **The file's max context:** `gguf-inventory` prints no KV metadata (`:456`), so running it would not have helped. I used the reference config's 1,048,576.
- **CUDA context overhead:** not recorded anywhere for our engine.
- **Scratch:** the V4.1 m=1 scratch and the prefill arena.
- **Our kernels' bandwidth at V4.1 shapes,** on either card and on the host. The WKS-36 contradiction is unresolved.
- **Router hotness.**
- **V4.1 node count and host boundary cost:** both are estimates, and `c_node` has not been recalibrated on the A6000.
- **Top-k kernel cost:** no such kernel exists yet.
- **Link figures:** the A6000's H2D/D2H bandwidth and the sync latency.
- **NVMe read rate:** only writes were measured (5.23 GB/s).
- **Delayed hc mix:** whether it allows overlap across layers. Not counted.

## 4. Improvement spots outside this spec (report only)

- `docs/roofline.md:21-23` and `crates/gguf/src/bin/gguf-inventory.rs:18,309-313`: "dense" subtracts the whole `*engram*` group, so `engram_wkv`/`engram_{k,q}` (334.4 MB per token) are missing. The per-token total is 12.035 GB, not 11.701 (XS doc, S tool).
- `docs/plan.md:406`: the rows % 4 claim does not hold for V4.1; the fix is in §2 (XS).
- `docs/research/host-surplus.md:185-187`: uses MLA arithmetic, 46,080 B per token. For V4.1 it is 3,200 B per token plus the 5.24 MB window, so a 13k prefix is about 46.8 MB, not 599 MB. H4 and C2 change accordingly (XS).
- `docs/roofline.md:98`: the KV read uses 576-dim rows. For V4.1 it is 25.17 MB plus 1,664 B × depth (XS).
- `docs/gpu-design.md:37`: says the crossing is 8 KB. For V4.1 it is 81,936 B and goes through the host (no P2P) (XS).
- `crates/gpu/src/weights.rs:44-54` and `q8f32.rs:9-15`: the Q8_0 f32 scale plane costs +427 MB of VRAM and ~0.74 ms per token. An f16 plane converted with hardware `cvt` is exact (M).
- `crates/gpu/src/q8f32.rs:467-490`: the f32-activation Q8_0 gemv was built for the K=128 q_nope2 site. A dp4a q8_1 version is the likely kernel for V4.1 (M, unmeasured).
- `crates/gpu/src/lib.rs:1567`: no `q4k_gemv_sel` (S–M; a prerequisite, see §6).
- `crates/gguf/src/lib.rs:164-240`: the strict reader is single-file, and it refuses tags 8 and 30 at open. V4.1 needs a split-aware reader and `GgmlType` arms for Q8_0 and BF16 (`quant.rs:30-40`) (S–M, B1's core work).
- `crates/gpu/src/weights.rs:143-152`: uploads are whole tensors only. B1 also needs row-slice uploads of expert stacks, and the combine needs an id mask because skipped `_sel` slots are stale (S).
- `crates/gpu/src/weights.rs:243-244`: `resident_size` for Q5 does not check k % 32, but `pack_q5` refuses it (`q5.rs:911`) (XS).

## 5. Model

I ran as Opus 5.5 (`claude-opus-5-5[1m]`), spawned as opus.
