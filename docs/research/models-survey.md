# Models survey — GLM Flash, Qwen3 family, DeepSeek V4 Flash against this engine

Date: 2026-09-24. Question (plan.md, 「공개」 M4): what would it take for this engine to run GLM Flash,
the Qwen3 family (MoE and dense) and DeepSeek V4 Flash, (a) as big MoE through the host tier and
(b) whole on one card against llama.cpp / ik at the same flags.

Rules of this document:

- Every model fact comes from a fetched `config.json`, a model card, a published GGUF header (read
  over HTTP range requests, first shard and, for split files, the second shard for tensor types),
  or the llama.cpp / ik_llama.cpp sources. Each row names its sources by id (list at the end).
- `unverified` marks a fact this round could not fetch. Nothing is filled from memory.
- File sizes are the sum of the published `.gguf` byte counts for that quant (HF tree API).
  "Fit" is **derived**: file bytes plus the KV bytes the config implies, ignoring the CUDA context
  and scratch (about a gigabyte on this engine, not re-measured here). No number in this file was
  measured on the box.
- Engine facts cite `path:line` in this tree at base `7a59449`.

Reference trees read: llama.cpp `master` at `d2e54583c7452353eb35d40431281f6ee984332f`
(2026-09-23), ik_llama.cpp `main` at `f3d6e6e3020ddfebad60113845bf521620766da5` (2026-09-23).

## 1. Four findings that change the plan's M4 entry

1. **"GLM Flash" today is GLM-5.3-Flash, and it is not a GQA + router port.** Its GGUF declares
   `general.architecture = glm5next` (ik only; mainline has no such arch). 34 of 45 trunk layers are
   KDA linear attention (a recurrent state, no KV rows), the other 11 are nope-only MLA with a DSA
   k-pool indexer, every block is wrapped in mHC hyper-connections, and the router is sigmoid + bias
   over 288 experts. [S1, S1b, G2, ik-hp:2370-2470, ik-glm5next:11-14] The attention is the port, and it
   is a new kernel family.
2. **The cheap GLM is GLM-4.7-Flash, and its GGUF says `deepseek2`.** llama.cpp converts
   `Glm4MoeLiteForCausalLM` to `MODEL_ARCH.DEEPSEEK2` [lc-conv-glm:247-250]. The file's attention
   rows have V2-Lite's shape (`key_length 576`, `value_length 512`, `rope.dimension_count 64`,
   `kv_lora_rank 512`) [G1]. What differs from V2-Lite is the query LoRA, the per-head nope/value
   widths, the split `attn_k_b`/`attn_v_b` tensors, the sigmoid router, 4 of 64 experts and 20 heads.
   Today's `deepseek2` GPU path refuses the file loudly on `n_used != 6`
   (`crates/gpu/src/arch/deepseek2/scratch.rs:317-318`).
3. **DeepSeek V4 Flash is not "V4.1 minus engram".** Same ik loader (`deepseek4` and `deepseek41`
   share `ik-hp:1976-1977`), but the V4 file takes ik's non-shared-streams branch
   [ik-ds4:1175-1210, 1365-1400]: every compressed layer owns its compressor, ratios alternate 4
   (CSA, with indexer) and 128 (HCA, dense over all compressed rows, no indexer), each compressor
   carries an absolute-position table `attn_compressor_ape`, the indexer has its own compressor
   (`indexer_compressor_{kv,gate,norm,ape}`) instead of `indexer.attn_k`, 64 index heads (V4.1: 32),
   and the first 3 layers route by a token table `ffn_gate_tid2eid` (I32) instead of the router
   [G11, ik-ds4:1589-1593]. Our `walk_streams` would refuse this file at "no layer carries
   `indexer.attn_k`" (`crates/model/src/arch/deepseek41/hparams.rs:549-556`).
4. **Weight formats decide the order more than any hyperparameter.** The card refuses Q5_K, F16 and
   MXFP4 (`crates/model/src/placement.rs:230-250`, `CardFormat::of`), ~~the GGUF reader parses I32 and every IQ type as
   `Unknown` (`crates/gguf/src/quant.rs:57-71`)~~ (corrected 2026-09-24: the reader names I32 and the IQ
   types since `seamc`, `crates/gguf/src/quant.rs:62-122`), the host tier has no MXFP4 or IQ dot
   (`crates/qdot/src/lib.rs:1-12`), and the card has routed `_sel` gemv for Q3_K, Q4_K and Q5_0 only
   (`q3k_gemv_sel`, `q4k_gemv_sel`, `q5_0_gemv_sel` in `crates/gpu/src`) — plus, since the DSpark
   draft, an MXFP4 gate·up/down pair fixed at 128 experts / 3 used
   (`crates/gpu-deepseek41/src/experts_mxfp4.rs`) that the placement does not yet route to. Of the published files
   read here, only the Qwen3 files (Q4_K + Q6_K) sit inside today's set on both tiers except for a
   Q6_K routed `_sel` on the card. The large new MoE files (GLM-5.3-Flash, Qwen3.8-Flash-Next, V4
   Flash) publish their experts as MXFP4, IQ3_XXS, IQ4_XS or IQ4_NL [§3 table C].

## 2. Which checkpoints exist (Hugging Face, 2026-09-24)

**GLM, Flash-named** (`author=zai-org&search=Flash`, 4 hits) [H1]:

| repo | created | params (HF API `safetensors.total`) | what it is |
|---|---|---:|---|
| zai-org/GLM-5.3-Flash | 2026-08-25 | 321,323,031,390 | text+vision, FP8 e4m3 128×128 blocks; card: "320B total parameters and just 18B active" [S1, C1:25] |
| zai-org/GLM-5.3-Flash-BF16 | 2026-08-25 | 321,323,031,390 | same `config.json` without `quantization_config` [S1b] |
| zai-org/GLM-4.7-Flash | 2026-01-19 | 31,221,488,576 | text; card: "a 30B-A3B MoE model" [S3, C2:27] |
| zai-org/GLM-4.6V-Flash | 2025-12-07 | 10,292,777,472 | dense `glm4v` vision model; out of scope (vision), listed because Flash-named [S4] |

Not Flash-named but in plan.md's track ①: GLM-4.7 (358,337,791,296) and GLM-4.5-Air
(110,468,824,832), both `Glm4MoeForCausalLM` [S5, S6].

**Qwen** (`author=Qwen`, searches Qwen3.5 / Qwen3.6 / Qwen4 / Qwen3-Next and the newest 150) [H2]:
the Qwen3 checkpoints the spec names exist (235B-A22B, 30B-A3B, 8B, 14B, 32B, and the 2507 instruct
refreshes of the two MoE ones). Newer: Qwen3.5 (0.8B … 397B-A17B, 2026-02), Qwen3.6 (27B, 35B-A3B,
2026-04), Qwen3.8 (27B, Flash-Next, 2.4T-A95B [S19], 2026-08), and Qwen3-Next-80B-A3B (2025-09, `qwen3next`, 512 / 10 experts, linear attention on 3 of 4 layers [S14]). Nothing is
named Qwen4 on the hub; `qwen4exp` is the llama.cpp / ik arch of Qwen3.8-Flash-Next
[lc-conv-qwen4exp:16-17]. Qwen3.5, 3.6 and 3.8 dense/MoE share `qwen35`/`qwen35moe`: Gated DeltaNet
on 3 of every 4 layers (`full_attention_interval 4`). They are a different program from Qwen3.

**DeepSeek V4 Flash** [S20, S21]: `deepseek-ai/DeepSeek-V4-Flash` (created 2026-04-22) and
`deepseek-ai/DeepSeek-V4-Flash-0731` (2026-07-31). Their configs differ only by the `dspark_*` keys
and two more trailing zeros in `compress_ratios` (46 entries vs 44). Card: "284B parameters (13B
activated)" [C3:43]; HF API total 290,944,616,402 (not reconciled here).

## 3. Model tables

Legend: GQA ratio = heads / KV heads. RoPE kind is ik's `llama_rope_type` table
[ik-llama:9751-9849] and mainline's [lc-model:2925-3080]: NORM = adjacent pairs (GPT-J style),
NEOX = halves, IMROPE = interleaved multi-section rope (text positions give all sections one value).

### A. Shape and attention

| model | `general.architecture` | params total / active | layers | hidden | heads / KV (ratio) | head dim | attention | QK-norm | RoPE | norm eps | src |
|---|---|---|---:|---:|---|---|---|---|---|---|---|
| GLM-5.3-Flash | `glm5next` (ik only) | 321.3B / 18B | 45 + 1 MTP | 4096 | 64 / 1 in GGUF on the 11 MLA layers (config 64 / 64) | nope 256, v 256 | 34 KDA (64 heads × 128, conv 4, gate bound −5) + 11 MLA nope-only (`q_lora 1536`, `kv_lora 512`, key_length 512) with DSA k-pool indexer (32 heads × 128, top 2048, kpool 4) | indexer k LayerNorm | none (`rope.dimension_count 0`) | RMS 1e-5 | S1, G2, ik-hp:2370-2470 |
| GLM-4.7-Flash | `deepseek2` | 31.2B / 3B | 47 (+1 MTP in HF, not in the GGUF) | 2048 | 20 / 1 in GGUF (MLA) | nope 192 + rope 64, v 256 (`key_length_mla 256`, `value_length_mla 256`) | MLA, `q_lora 768`, `kv_lora 512`, cache row 576 | no | NORM, 64 dims, θ 1e6, no YaRN | RMS 1e-5 | S3, G1 |
| GLM-4.7 | `glm4moe` | 358.3B / unverified | 92 + 1 MTP | 5120 | 96 / 8 (12) | 128 | GQA, `attention_bias true` | yes (`use_qk_norm`) | NEOX, partial 0.5 (64 dims), θ 1e6 | RMS 1e-5 | S5, lc-conv-glm:151-157, lc-model:3074 |
| GLM-4.5-Air | `glm4moe` | 110.5B / unverified | 46 + 1 MTP | 4096 | 96 / 8 (12) | 128 | GQA, `attention_bias true` | no | NEOX, partial 0.5, θ 1e6 | RMS 1e-5 | S6 |
| Qwen3-235B-A22B (-2507) | `qwen3moe` | 235.1B / 22B (name) | 94 | 4096 | 64 / 4 (16) | 128 | GQA | yes (`attn_q_norm`, `attn_k_norm`) | NEOX, full 128, θ 1e6 (2507: 5e6), no YaRN | RMS 1e-6 | S7, S8, G3 |
| Qwen3-30B-A3B (-2507) | `qwen3moe` | 30.5B / 3B (name) | 48 | 2048 | 32 / 4 (8) | 128 | GQA | yes | NEOX, full 128, θ 1e6 (2507: 1e7) | RMS 1e-6 | S9, S10, G4 |
| Qwen3-8B | `qwen3` | 8.19B | 36 | 4096 | 32 / 8 (4) | 128 | GQA | yes | NEOX, full 128, θ 1e6 | RMS 1e-6 | S11, G5 |
| Qwen3-14B | `qwen3` | 14.77B | 40 | 5120 | 40 / 8 (5) | 128 | GQA | yes | NEOX, full 128, θ 1e6 | RMS 1e-6 | S12, G6 |
| Qwen3-32B | `qwen3` | 32.76B | 64 | 5120 | 64 / 8 (8) | 128 | GQA | yes | NEOX, full 128, θ 1e6 | RMS 1e-6 | S13, G7 |
| Qwen3.6-35B-A3B (= Qwen3.5-35B-A3B shape) | `qwen35moe` | 35.95B / 3B | 40 | 2048 | 16 / 2 (8) on 10 full layers | 256 | 30 Gated DeltaNet (16 k-heads, 32 v-heads × 128) + 10 GQA with output gate `attn_gate` | yes | IMROPE sections [11,11,10], 64 of 256 dims, θ 1e7 | RMS 1e-6 | S15, S16, G8 |
| Qwen3.8-27B (= 3.5/3.6-27B shape) | `qwen35` | 27.8B dense | 64 + 1 MTP | 5120 | 24 / 4 (6) on 16 full layers | 256 | 48 Gated DeltaNet + 16 GQA with output gate | yes | IMROPE, 64 of 256, θ 1e7 | RMS 1e-6 | S17, G9 |
| Qwen3.8-Flash-Next | `qwen4exp` | 180.0B (API) / card: "125B with 6B activated, plus 51B n-gram embedding and 4B MTP" | 48 | 2560 | 24 / 2 (12) on 12 full layers | 256 | 36 Gated DeltaNet + 12 GQA with output gate; indexer (4 heads × 128, top 2048) on the full layers at compress ratio 4 | yes | IMROPE, 64 of 256, θ 1e7 | RMS 1e-6 | S18, G10, C4:46 |
| DeepSeek-V4-Flash (-0731) | `deepseek4` | 290.9B (API) / 13B (card) | 43 (+1 MTP in HF) | 4096 | 64 / 1 | 512 (one latent is K and V), rope tail 64 | window 128 + compressed rows: ratio 4 layers (CSA) through a 64-head indexer (top 512), ratio 128 layers (HCA) dense; per-head sinks; grouped output `o_groups 8`, `o_lora 1024`; `q_lora 1024` | `attn_q_a_norm`, `attn_kv_a_norm` | NORM, 64 dims; window layers θ 1e4, compressed θ 1.6e5 under YaRN ×16 (orig 65536, β 32/1) | RMS 1e-6 | S20, S21, G11, ik-ds4:1128-1400 |
| DeepSeek-V4.1-Flash (anchor) | `deepseek41` (ik only) | 763.2B (API; includes the two engram tables, (384,006,168 + 384,016,682) × 256 ≈ 196.6B, derived from `engram_num_embeddings × engram_head_dim`) / unverified | 40 | 5120 | 64 / 1 | 512, rope 64 | window 128 + shared compressed streams (ratios 2 and 1, sources 2/8/14/20), indexer 32 × 128 top 512; engram layers 1, 14 | as V4 | as V4 | RMS `1e-20` in config | S22, `hparams.rs` |

### B. FFN, MoE and router

Router rule, all three families, is one ik function [ik-bc:1453-1581]: score by the gating function
(softmax :1497, sigmoid :1503, sqrt-softplus :1509), add `exp_probs_b` for the selection only
(:1523), `ggml_top_k` (:1543), weights from the unbiased scores (:1547), divide by their sum when
`norm_w` (:1572, no epsilon for these archs), scale when `|w − 1| > 1e-5` (:1578). No model here uses
groups (`n_group 1`, `topk_group 1` wherever the config has them).

| model | FFN act | experts routed / used / shared | expert ff | gating (GGUF value) | bias `exp_probs_b` | norm_topk | scale | dense lead | MTP / draft | engram / indexer | src |
|---|---|---|---:|---|---|---|---:|---:|---|---|---|
| GLM-5.3-Flash | SiLU, SwiGLU clamp 10 | 288 / 8 / 1 | 2048 | sigmoid (2) | yes | yes | 2.5 | 3 | 1 nextn layer in the GGUF | DSA k-pool indexer; mHC 4 streams, 20 Sinkhorn | S1, G2 |
| GLM-4.7-Flash | SiLU | 64 / 4 / 1 | 1536 | sigmoid (2) | yes | yes | 1.8 | 1 | HF has 1, GGUF block_count 47 has none | none | S3, G1, ik-hp:1318-1330 |
| GLM-4.7 | SiLU | 160 / 8 / 1 | 1536 | sigmoid (written by converter) | yes (noaux_tc) | yes | 2.5 | 3 | 1 | none | S5, lc-conv-glm:172-180 |
| GLM-4.5-Air | SiLU | 128 / 8 / 1 | 1408 | sigmoid | yes | yes | 1.0 | 1 | 1 | none | S6 |
| Qwen3-235B-A22B | SiLU | 128 / 8 / 0 | 1536 | softmax (ik hard-codes, no key) | no | yes (`norm_w true`) | none | 0 | no | none | S7, G3, ik-qwen3:137-151 |
| Qwen3-30B-A3B | SiLU | 128 / 8 / 0 | 768 | softmax | no | yes | none | 0 | no | none | S9, G4 |
| Qwen3-8B / 14B / 32B | SiLU dense, ff 12288 / 17408 / 25600 | — | — | — | — | — | — | all | no | none | S11-S13 |
| Qwen3.6-35B-A3B | SiLU | 256 / 8 / 1 (with `ffn_gate_inp_shexp` gate) | 512 (shexp 512) | unverified (not in GGUF header; ik qwen35moe builder not read) | no | unverified | unverified | 0 | MTP 1 in HF | none | S16, G8 |
| Qwen3.8-27B | SiLU dense, ff 17408 | — | — | — | — | — | — | all | nextn 1 in the GGUF | none | S17, G9 |
| Qwen3.8-Flash-Next | SiLU | 512 / 10 / 1 (gated) | 640 | unverified | no | unverified | unverified | 0 | MTP 1 (separate GGUF) | PLE: per-layer 3-gram hashed embedding (layer 1; 51B params) — the engram analogue; hc 4 streams low-rank 320 | S18, G10 |
| DeepSeek-V4-Flash | SiLU, SwiGLU clamp 10 | 256 / 6 / 1 | 2048 | sqrt-softplus (4) | yes | yes | 1.5 | 0 | 1 in HF; ik zeroes it for 43 layers [ik-hp:2019-2031] | no engram; hash routing on layers 0-2 (`hash_layer_count 3`, `ffn_gate_tid2eid` I32) | S20, G11 |
| DeepSeek-V4.1-Flash | as V4 | 384 / 6 / 1 | 2304 | sqrt-softplus (4) | yes | yes | 1.5 | 0 | 3 nextn; DSpark draft | engram 2 sites; `hash_layer_count 0` (`crates/model/tests/ds41_meta.rs:365-372`) | S22 |

### C. Tokenizer, files, formats, fit

KV bytes per token are derived from the config: GQA = layers_with_KV × 2 × KV heads × head dim × 2 B
(f16); MLA = layers × cache row × 2 B. Hybrid models also keep a fixed recurrent state per sequence
whose size this round did not derive. The types column lists the types of the shard(s) read;
other shards may quantize the same role differently (one shard alone mixed Q4_K and Q5_K for the
same role).

| model | vocab / `tokenizer.ggml.pre` | chat template | Q4_K_M GB | Q3_K_M GB | published expert / attention types (file read) | KV B/token (derived) | fit (derived) | who runs it | src |
|---|---|---|---:|---:|---|---:|---|---|---|
| GLM-5.3-Flash | 154,880 / `glm4` | GLM (`[gMASK]<sop>`, reasoning effort) | UD-Q4_K_XL 199.71 | UD-Q3_K_XL 147.54 | Q4_XL: exps gate/up Q4_K (1 Q5_K), down Q5_K/Q6_K, attn and shexp Q8_0; Q3_XL: exps IQ3_XXS / IQ4_XS | 11 × 512 × 2 = 11,264 + KDA state | host tier at 256 GB RAM | ik | G2, F1 |
| GLM-4.7-Flash | 154,880 / `glm4` | GLM | 18.31 | 14.61 | exps gate/up Q4_K, down Q4_K/Q6_K; shexp gate/up **Q5_K**, down Q6_K; `attn_k_b`/`attn_v_b`/`attn_kv_a_mqa` Q8_0; `token_embd` Q4_K | 47 × 576 × 2 = 54,144 | 24 GB (18.3 + 1.8 at 32k) | both | G1, F2 |
| GLM-4.7 | 151,552 / `glm4` (4.5-Air hash) | GLM | 216.46 | 171.27 | not read | 92 × 2 × 8 × 128 × 2 = 376,832 | host tier | both | S5, F3 |
| GLM-4.5-Air | 151,552 / `glm4` | GLM | 72.98 | 57.20 | not read | 46 × 2 × 8 × 128 × 2 = 188,416 | 2 cards + host, or host tier | both | S6, F4 |
| Qwen3-235B-A22B-2507 | 151,936 / `qwen2` | ChatML (`<\|im_start\|>`) | 142.15 | 112.45 | exps gate/up Q4_K, down Q4_K/Q6_K; attn Q4_K, `attn_v` Q4_K/Q6_K | 94 × 2 × 4 × 128 × 2 = 192,512 | host tier | both | G3, F5 |
| Qwen3-30B-A3B-2507 | 151,936 / `qwen2` | ChatML | 18.56 | 14.71 | same pattern as 235B | 48 × 2 × 4 × 128 × 2 = 98,304 | 24 GB (18.6 + 3.2 at 32k) | both | G4, F6 |
| Qwen3-8B | 151,936 / `qwen2` | ChatML | 5.03 | 4.12 | Q4_K, `ffn_down`/`attn_v` half Q6_K, output Q6_K | 36 × 2 × 8 × 128 × 2 = 147,456 | 24 GB | both | G5, F7 |
| Qwen3-14B | 151,936 / `qwen2` | ChatML | 9.00 | 7.32 | as 8B | 163,840 | 24 GB | both | G6, F8 |
| Qwen3-32B | 151,936 / `qwen2` | ChatML | 19.76 | 15.97 | as 8B | 262,144 | 24 GB to ~8k tokens (19.8 + 2.1), 48 GB at 32k | both | G7, F9 |
| Qwen3.6-35B-A3B | 248,320 / `qwen35` | Qwen3.5 (vision-aware) | UD-Q4_K_M 22.13 | UD-Q3_K_M 16.60 | exps gate/up Q4_K, down **Q5_K**/Q6_K; attn, DeltaNet, shexp Q8_0 (unsloth UD files only — the lmstudio-community Q4_K_M, 21.17 GB, carries Q4_K/Q6_K/F32 alone, the same type set as our Qwen3-30B-A3B file; corrected 2026-09-25, qwennext-lit) | 10 × 2 × 2 × 256 × 2 = 20,480 + DeltaNet state | 24 GB tight at Q4, fits at Q3 | both | G8, F10 |
| Qwen3.8-27B | 248,320 / `qwen35` | Qwen3.5 | UD-Q4_K_M 16.46 | UD-Q3_K_XL 13.15 | mixed **IQ4_XS / IQ4_NL / IQ3_S** with Q3-Q6_K and Q8_0 | 16 × 2 × 4 × 256 × 2 = 65,536 + state | 24 GB | both | G9, F11 |
| Qwen3.8-Flash-Next | 248,320 / `qwen35` | Qwen3.5 | UD-Q4_K_XL 111.33 | UD-Q3_K_XL 89.99 | Q4_XL: exps gate/up Q4_K, down **Q5_1**/Q8_0, indexer BF16, PLE table IQ4_NL; Q3_XL: exps IQ3_XXS / IQ4_NL | 12 × 2 × 2 × 256 × 2 = 24,576 + state | host tier (or 2 × 48 GB + host) | both | G10, F12 |
| DeepSeek-V4-Flash | 129,280 / `joyai-llm` | DeepSeek (unsloth-patched in this file) | UD-Q4_K_XL 155.10 | UD-Q3_K_M 129.32 | Q4_XL: exps **MXFP4** (the checkpoint's native FP4), router `ffn_gate_inp` BF16, rest Q8_0/F32, `ffn_gate_tid2eid` **I32**; Q3_M: exps gate/up **IQ3_XXS**, down MXFP4 | window 128 rows × 512 × 2 per layer + ⌈ctx/4⌉ or ⌈ctx/128⌉ rows (+ index keys on CSA layers) | host tier at 256 GB RAM | both | G11, F13 |
| DeepSeek-V4.1-Flash | 129,280 / `joyai-llm` (antirez GGUF; our file not re-read) | DeepSeek | — | — | our serving file is our own mix (plan.md) | `crates/model/src/arch/deepseek41/kv.rs` | host tier (plan.md 타겟 프로필) | ik | G12 |

Files skipped: `unsloth/Qwen3.8-Flash-Next-GGUF` "Q4_K_M" is `MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`
(2.79 GB, the MTP head), and `unsloth/DeepSeek-V4-Flash-0731-GGUF` "Q8_0" is the DSpark draft
(`dspark-…-Q8_0.gguf`, 10.90 GB). Neither is a model file.

## 4. What the engine has today (read, not run)

| piece | where | shape it fixes |
|---|---|---|
| arch selection | `crates/model/src/arch/mod.rs:27-51` | `Arch::{Deepseek2, Deepseek41}` from `general.architecture`; unknown string is an error naming it |
| V4.1 hyperparameters, layer kinds, stream walk | `arch/deepseek41/hparams.rs` | refuses a gating func other than sqrt-softplus (:396-401), a non-YaRN rope (:333-339), a layer that reads a stream without `indexer.attn_k` (:549-556) |
| roles → placement | `arch/deepseek41/roles.rs:33-45`, `placement.rs:36-59` | no catch-all role; no dense-FFN role (V4.1 has none); no rule for `nextn.*` |
| KV accounting | `placement.rs:143` (`KvBytes`), `arch/deepseek41/kv.rs` | one impl per arch |
| card weight formats | `placement.rs:186-194` | Q3_K/Q4_K/Q6_K, Q5_0, Q5_1, Q8_0, F32, BF16→F32; refuses F16, Q5_K, MXFP4 |
| card routed gemv | `crates/gpu/src` `q3k_gemv_sel`, `q4k_sel.rs` `q4k_gemv_sel`, `q5.rs` `q5_0_gemv_sel`; V4.1 fused `ds41_expert_gate_up` (q3_K) | no Q5_K / Q6_K / Q8_0 routed kernel; MXFP4 only as the DSpark draft's 128/3 pair (`gpu-deepseek41/src/experts_mxfp4.rs`) |
| host tier dots | `crates/qdot/src/lib.rs:1-12` | Q3_K×Q8_K; Q4_K/Q5_K/Q6_K/Q5_0/Q5_1×Q8_2_X4; Q8_0 cells |
| host tier service | `crates/gpu/src/hybrid.rs:1051` (`HostExperts`), `chain/ffn.rs:1327` (`Ds41Host`), `crates/model/src/moe.rs:952` (`HostLayerSpec`) | protocol shared; one `HostExperts` impl per arch |
| V2-Lite router | `crates/gpu/src/router.rs:1-33` | softmax, `N_EXPERT 64`, `N_USED 6` consts, ties to the smaller id |
| V4.1 router | `crates/gpu-deepseek41/src/router.rs:1-45` | sqrt-softplus + bias, `N_EXPERT 384`, `N_USED 6` consts, ties to the **larger** id, f64 sum |
| MLA flash (V2-Lite) | `crates/gpu/src/flash.rs:96-202` | `LATENT 512`, cache row 576, MMA tile of 16 query rows (`q_rows.div_ceil(MMA_ROWS)`, :256) |
| V4.1 attention | `crates/gpu-deepseek41/src/{attn,compress,indexer,index_key,hc,rope}.rs`, `chain/attn.rs:1-49` | window ⧺ selected compressed rows with sinks; shared compressed streams |
| V2-Lite MoE pins | `crates/gpu/src/arch/deepseek2/scratch.rs:288-335` | refuses `n_expert != 64 || n_used != 6` |
| draft | `gpu-gates::draft::Lookup` (n-gram over token ids) served through `step_pair` (V4.1 two-row pass) | the n-gram table is model-agnostic; the two-row pass is a `ChainBody` feature |

## 5. Delta table

Cell legend: **R** reuse unchanged · **V** variant of an existing piece · **N** new · — not needed.
Sizes are derived estimates: XS under ~100 lines, S a few hundred, M one round, L several rounds.
Columns: G47F GLM-4.7-Flash · Q3M Qwen3 MoE (30B, 235B) · Q3D Qwen3 dense · G4X GLM-4.7 / 4.5-Air ·
V4F DeepSeek V4 Flash · G53F GLM-5.3-Flash · Q35 Qwen3.5/3.6/3.8 (dense and MoE) · Q38N Qwen3.8-Flash-Next.

| piece | G47F | Q3M | Q3D | G4X | V4F | G53F | Q35 | Q38N |
|---|---|---|---|---|---|---|---|---|
| GGUF types (`GgmlType`) | R | R | R | R | V XS (I32) | R (Q4_XL) | V (IQ4_XS/NL, IQ3_S) | V (IQ4_NL) |
| card format (`CardFormat`) | V S (Q5_K shexp) | R | R | unverified | N M (MXFP4) or self-quantize | V S (Q5_K) | V S (Q5_K) / N (IQ) | V (Q5_1 exps) |
| card routed `_sel` gemv | V S (Q6_K down) | V S (Q6_K down) | — | unverified | N M (MXFP4) | V S (Q5_K, Q6_K) | V S (Q5_K) | V S (Q5_1, Q8_0) |
| card dense gemv | R (Q4_K, Q6_K, Q8_0) | R | R | R | R (Q8_0) | R (Q8_0) | N (IQ4_XS in 3.8-27B file) | R (Q8_0), V (BF16 indexer) |
| host tier dots | R | R | — | R | N M (MXFP4 × Q8) | R (Q4_XL); N (IQ at Q3) | — | V (Q5_1 exps: R) |
| host tier service (`HostExperts` impl, `HostLayerSpec`) | V XS | V XS | — | V XS | V XS | V XS | — | V XS |
| placement roles + `KvBytes` | V XS (deepseek2 table) | N S | N S (dense-FFN role) | N S | V S (per-layer compressors) | N S (KDA state, `nextn` Unread) | N S | N S |
| hot list | R | R | — | R | R | R | — | R |
| draft (n-gram lookup; `step_pair`) | V S (two-row pass in its body) | V S | V S | V S | V S | V M (recurrent state rewind) | V M | V M |
| embedding row | V XS (Q4_K row) | V XS (Q4_K row) | V XS | unverified | R (Q8_0 row, V4.1 reads bf16) | V XS | V XS | V XS + PLE N |
| norms | R | V XS (per-head QK RMS) | V XS | V XS (4.7 only) | R | V XS (LayerNorm for indexer k) | V XS | V XS |
| RoPE | R (NORM 64) | N S (NEOX full 128, θ 1e6-1e7) | N S | V S (NEOX partial 64) | V S (same YaRN, two bases as V4.1) | — (none) | N M (IMROPE sections) | N M |
| attention kernel | R (`flash.rs` 576/512); V XS if 20 heads break the 16-row tile | N M (GQA flash decode, head 128) | N M (same kernel) | V S (GQA + qkv bias + partial rope) | V M-L (CSA with own indexer compressor + APE; HCA dense) | N L (KDA recurrent) + N M (nope-only MLA + k-pool indexer) | N L (Gated DeltaNet) + V S (GQA head 256 + output gate) | N L (DeltaNet) + N M (compressed indexer on GQA) |
| KV / state topology | R (576-row plane) | N S (per-layer K and V, GQA) | N S | N S | V M (per-layer compressed, no sharing) | N M (state + 11 MLA planes) | N M (state + 1-in-4 GQA) | N M |
| query path | V S (q_lora `attn_q_a` → `attn_q_a_norm` → `attn_q_b`; split `attn_k_b`/`attn_v_b` instead of derived `wk_b`; nope 192 / v 256) | — | — | — | R (V4.1's) | V S | — | — |
| router | V S (sigmoid + bias, 64 / 4, scale 1.8) | V S (softmax, 128 / 8, renorm, no scale) | — | V S (sigmoid, 160 or 128 / 8) | V XS (V4.1 rule, 256 / 6) | V S (sigmoid, 288 / 8, 2.5) | unverified | unverified |
| hash routing | — | — | — | — | N S (token → 6 ids table on layers 0-2) | — | — | — |
| shared expert | V XS (Q5_K / Q6_K formats) | — | — | R | R | R | V S (sigmoid `ffn_gate_inp_shexp` gate) | V S |
| hyper-connections | — | — | — | — | R (V4.1 hc kernels; same 4 / 20 constants) | V S (hc fn Q8_0, eps 1e-6) | — | N M (low-rank 320 variant) |
| engram / PLE | — | — | — | — | — (none in V4) | — | — | N M (PLE, 3-gram, 8 heads) |
| head | R | R | R | R | R (hc collapse as V4.1) | R | R | V S (`output_hc_*`) |
| tokenizer pre (`tok` crate) | N S (`glm4`) | N S (`qwen2`) | same | N S (`glm4`) | R if `tok` covers `joyai-llm` | N S (`glm4`) | N S (`qwen35`) | N S |
| chat template | N XS (GLM) | N XS (ChatML) | same | N XS | R | N XS | N XS | N XS |
| reference profile + oracle | N S (a second `deepseek2` model profile under `tools/ref/models/`, ik) | N S (`qwen3moe`, ik and mainline) | N S | N S | N S (`deepseek4`, ik) | N S (ik only) | N S | N S |

Gates, one per new or varied piece, all in the existing form (ik harness dump per tensor, compared
bit for bit or within the crate's recorded band; tap names from ik's `cb()` calls):

| piece | gate |
|---|---|
| router variants | taps `ffn_moe_probs`, `ffn_moe_topk`, `ffn_moe_weights_norm`, `ffn_moe_weights_scaled` [ik-bc:1517-1579]: ids equal, weights bit-identical on the host rule; the tie rule pinned by a constructed tie case per arch |
| GQA flash decode | `kqv_out` (or the arch's attention output tap) against ik at depths 6 / 1024 / 4096, both the scalar and MMA paths, in the form `gate-gpu-e2e` uses for `flash.rs` |
| NEOX / IMROPE rope | Q and K after rope, bit-identical to `ggml_rope_ext` of the reference build on the host rule, like `gate_deepseek41_rope` |
| QK-norm, LayerNorm | per-head normed Q/K taps against the dump |
| card Q5_K / Q6_K / MXFP4 `_sel` | the qdot-style row-dot gate against the harness dump for that type (`gate-qdot` pattern), then the layer's `ffn_moe_down` tap |
| hash routing | `hashed_exps` tap [ik-ds4:1591] ids equal |
| V4 compressor with APE, indexer compressor, HCA | taps `csa`/`hca`/`lid` from `ds4_build_comp` calls [ik-ds4:1179-1206] and `attn_csa` / `attn_hca` [ik-ds4:1390, 1401] |
| a whole model | `l_out-N` per layer and the `tokens` line of greedy decode equal to ik (the `gate-gpu-e2e` count pin form) |

## 6. The seam in `arch/` and the chain

`docs/arch-split.md` already decides the shape and this survey does not change it: no `Arch` trait
with associated types, monomorphic token path, one `Arch` variant and one `AnyEngine` arm per
architecture (the compiler then names every `match` a new arm misses), one `ChainBody` per
architecture in its own device crate when it brings its own `#[cuda_module]`s. What a new model adds:

- `crates/model/src/arch/<name>/{hparams,names,roles,kv,place,host}.rs` — the six files V4.1 has,
  each consumed by what exists (`placement::plan` takes `ModelTensors` and a `KvBytes`;
  `HostLayer::build` takes a `HostLayerSpec`). Name = the GGUF string (`qwen3moe`, `qwen3`,
  `deepseek4`, `glm5next`).
- A `ChainBody` impl per arch: `crates/gpu/src/arch/<name>/` when its kernels are shared ones
  (Qwen3, GLM-4.x), its own crate like `gpu-deepseek41` when it adds device code only it uses
  (V4 Flash could live *in* `gpu-deepseek41` as a second body, since it reuses hc, sink attention,
  rope, router and indexer kernels there).
- **GLM-4.7-Flash is the one case where the file's arch string is an existing variant.** It must
  not become `if` branches in `deepseek2`'s V2-Lite code. The V2-Lite pins move from constants to
  hyperparameters read once (`n_used`, gating function, `q_lora_rank`, `key_length_mla`,
  `value_length_mla`, head count), and every value V2-Lite does not support keeps a loud refusal.

**Router: one device core, instantiated per arch.** The two routers we have already differ in three
things a shared core must carry to keep both bit-identical: the score function (softmax over all
logits vs element-wise sqrt-softplus; sigmoid joins them), the tie rule (V2-Lite: smaller id wins,
`router.rs:11`; V4.1: larger id wins, `gpu-deepseek41/src/router.rs:22-24`), and the compile-time
widths (`N_EXPERT`, `N_USED`, which size shared arrays). Carry them as a const-generic core
`route::<SCORE, TIE, N_EXPERT, N_USED>` marked `#[inline(always)]`, with the bias, norm and scale as
launch arguments; each arch keeps its own `#[kernel]` wrapper, so no existing kernel's PTX changes.
Proof for the refactor is the move class: `just ptx-scan` table identical and the V4.1 and V2-Lite
router gates' verdict lines identical. Each new arch adds a wrapper and its gate.

**Attention: no trait.** Attention kernels, their KV topology and their step inputs are the places
the types differ (arch-split principle 1). What is shared is the kernel library: a GQA flash-decode
kernel goes in `crates/gpu/src/` beside `flash.rs` (shape-named, head dim and GQA ratio as launch
arguments), because Qwen3, GLM-4.x and the full-attention layers of Qwen3.5+ and Qwen3.8-Flash-Next
all use it. A delta-rule recurrent kernel (KDA for GLM-5.3-Flash, Gated DeltaNet for Qwen3.5+) is one
family and one later program; that they can share one kernel is a hypothesis this round did not
check (ik has them in separate files, `src/llama-kda.cpp` and `src/llama-delta-net.cpp`).

## 7. Recommended order

V4.1 stays the headline first (plan.md 「공개」). After it:

1. **Qwen3-30B-A3B, whole on one card** (goal b). The published Q4_K_M uses only Q4_K and Q6_K; the
   new pieces are the GQA flash-decode kernel, NEOX rope, QK-norm, the softmax router variant and a
   Q6_K routed `_sel`. Both ik and mainline run it, so the same-flags comparison has two references.
   It is the explicit user goal ("qwen 계열이 vram에 통짜로 들어갔을 때 llama보다 빠른 것도 증명").
2. **Qwen3 dense 8B / 14B / 32B.** Same kernels plus the dense FFN path and a dense-FFN role; the
   narrow-margin gemv contest plan.md describes.
3. **GLM-4.7-Flash, whole on one card.** It reuses the MLA flash kernel that already beats ik on
   V2-Lite, so its attention is the cheapest in this list; the work is lifting the `deepseek2` pins
   into hyperparameters, the query LoRA path, the sigmoid router, and a Q5_K card format for the
   shared expert (or a self-quantized file). It is second to Qwen3 only because it needs the Q5_K
   format first and because its attention win would repeat V2-Lite's rather than prove a new kernel.
4. **Qwen3-235B-A22B through the host tier** (goal a). No new kernel after steps 1-2; a new
   `HostExperts` impl and placement roles.
5. **DeepSeek V4 Flash through the host tier.** A second body next to V4.1's: per-layer compressors
   with APE, the indexer's own compressor, dense HCA over compressed rows, hash routing on layers
   0-2, 256 experts, and either an MXFP4 expert kernel on both tiers (the checkpoint's native
   format) or a self-quantized K-quant file. Our part ends at sm_86 + AVX2; the DGX Spark port (GB10
   sm_121, arm64 host, 128 GB unified memory) is the "help wanted" item.
6. **GLM-5.3-Flash and Qwen3.5 / 3.6 / 3.8 together**, as one linear-attention program: a
   delta-rule recurrent kernel and its state cache, then GLM-5.3's nope-only MLA with the k-pool
   indexer and mHC (the hc kernels carry over from V4.1), and Qwen3.8-Flash-Next's PLE. This is
   where the user's named target ("glm-5.3 flash") lands, and it is the largest item.

## 8. Is the router the one we already implement?

- **GLM-4.7-Flash, GLM-5.3-Flash, GLM-4.x: no, a variant.** The rule is V4.1's structure (score,
  bias for selection only, top-k, renormalize the unbiased scores, scale) with `sigmoid` in place
  of sqrt-softplus [ik-bc:1503, G1 `expert_gating_func = 2`, G2 same]. `n_group 1`, so there is no
  group stage. The variant is the score function plus the widths (64 / 4, 288 / 8, 160 / 8) — S.
  For GLM-4.7-Flash the attention is not the port either: the flash kernel is ours already; the
  query path and pins are. For GLM-5.3-Flash the attention is the whole port and it is L.
- **DeepSeek V4 Flash: the rule is V4.1's exactly** (gating 4, bias, top 6, norm, 1.5), with
  `N_EXPERT` 256 instead of the compiled 384 and three leading hash layers that bypass it — XS for
  the width, S for the hash table. The attention topology is the port (M-L), plus the expert format.
- **Qwen3 MoE:** softmax, renormalized, no bias, no scale [ik-qwen3:137-151] — the V2-Lite rule with
  `norm_w` and different widths (128 / 8).

## 9. Not verified in this round

- Active parameters of GLM-4.7 / 4.5-Air and V4.1-Flash (the V4.1 card's table columns are
  ambiguous); the Qwen3 active counts are read from the model names.
- The Qwen3.5+ and Qwen3.8-Flash-Next router rule (the ik `qwen35moe` / `qwen4exp` builders were
  not read), their recurrent state sizes, and their gating keys.
- GLM-4.7 and GLM-4.5-Air tensor types (headers not read).
- Whether `flash.rs`'s MMA path is correct for 20 query rows (it groups by `div_ceil(16)`; no gate
  runs 20 heads).
- The pre-tokenizer in our own V4.1 serving file (the `joyai-llm` value comes from a third-party
  GGUF, G12); `unsloth/DeepSeek-V4.1-Flash-GGUF` returned HTTP 401.

## Sources

HF repo commits are the `sha` the HF API returned on 2026-09-24. Config URLs are
`https://huggingface.co/<repo>/resolve/main/config.json`; the commit pins the content.

| id | source | commit |
|---|---|---|
| H1 | https://huggingface.co/api/models?author=zai-org&search=Flash | — |
| H2 | https://huggingface.co/api/models?author=Qwen&limit=150&sort=createdAt&direction=-1 | — |
| S1 | https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/config.json | eb9eb208eb0d |
| S1b | https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/resolve/main/config.json | a5b45eb41df6 |
| S3 | https://huggingface.co/zai-org/GLM-4.7-Flash/resolve/main/config.json | 7dd20894a642 |
| S4 | https://huggingface.co/zai-org/GLM-4.6V-Flash/resolve/main/config.json | 411bb4d77144 |
| S5 | https://huggingface.co/zai-org/GLM-4.7/resolve/main/config.json | 602d01efcdd3 |
| S6 | https://huggingface.co/zai-org/GLM-4.5-Air/resolve/main/config.json | a24ceef6ce4f |
| S7 | https://huggingface.co/Qwen/Qwen3-235B-A22B/resolve/main/config.json | 8efa61729e24 |
| S8 | https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507/resolve/main/config.json | ac9c66cc9b46 |
| S9 | https://huggingface.co/Qwen/Qwen3-30B-A3B/resolve/main/config.json | ad44e777bcd1 |
| S10 | https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/resolve/main/config.json | 0d7cf23991f4 |
| S11 | https://huggingface.co/Qwen/Qwen3-8B/resolve/main/config.json | b968826d9c46 |
| S12 | https://huggingface.co/Qwen/Qwen3-14B/resolve/main/config.json | 40c069824f42 |
| S13 | https://huggingface.co/Qwen/Qwen3-32B/resolve/main/config.json | 9216db5781bf |
| S14 | https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/resolve/main/config.json | 9c7f2fbe8446 |
| S15 | https://huggingface.co/Qwen/Qwen3.5-35B-A3B/resolve/main/config.json | 59d61f3ce65a |
| S16 | https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/main/config.json | 995ad96eacd9 |
| S17 | https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/config.json | 1d4bf0f2ff60 |
| S18 | https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/config.json | de4b8e4d43b9 |
| S19 | https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/resolve/main/config.json (92 layers, 512/10 experts; listed, not a target) | 207bd685a7e3 |
| S20 | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/config.json | 60d8d70770c6 |
| S21 | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/resolve/main/config.json | 7872f01b1d1f |
| S22 | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/config.json | dba1be0a40aa |
| C1 | https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/README.md | eb9eb208eb0d |
| C2 | https://huggingface.co/zai-org/GLM-4.7-Flash/resolve/main/README.md | 7dd20894a642 |
| C3 | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/README.md | 60d8d70770c6 |
| C4 | https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/README.md | de4b8e4d43b9 |
| G1 | https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf (header) | 0d32489ecb9d |
| G2 | https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/main/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf, types from -00002-of-00006 and UD-Q3_K_XL -00002-of-00004 | 621d456e93e9 |
| G3 | https://huggingface.co/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF/resolve/main/Q4_K_M/Qwen3-235B-A22B-Instruct-2507-Q4_K_M-00001-of-00003.gguf | 437d6915c5c5 |
| G4 | https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/main/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf | eea7b2be5805 |
| G5 | https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf | 7c41481f57cb |
| G6 | https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf | 530227a7d994 |
| G7 | https://huggingface.co/Qwen/Qwen3-32B-GGUF/resolve/main/Qwen3-32B-Q4_K_M.gguf | 938a7432affa |
| G8 | https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf | a483e9e6cbd5 |
| G9 | https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_M.gguf | 4ca720788d1e |
| G10 | https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf, types from -00002-of-00004 and UD-Q3_K_XL -00002-of-00003 | 38bb39ee9782 |
| G11 | https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF/resolve/main/UD-Q4_K_XL/DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf, types from -00002-of-00005 and UD-Q3_K_M -00002-of-00004 | e3aa0d6a5fa4 |
| G12 | https://huggingface.co/antirez/deepseek-v4.1-flash-gguf/resolve/main/DeepSeek-V4.1-Flash-Q2.gguf (header: arch, pre) | dd8a266f7145 |
| F1-F13 | `https://huggingface.co/api/models/<repo>/tree/main?recursive=1` for unsloth/GLM-5.3-Flash-GGUF, unsloth/GLM-4.7-Flash-GGUF, unsloth/GLM-4.7-GGUF (70deda3fcbe6), unsloth/GLM-4.5-Air-GGUF (506d64aa8c5c), unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF, unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF, unsloth/Qwen3-8B-GGUF (a6adef130ffb), unsloth/Qwen3-14B-GGUF (a04a82c4739b), unsloth/Qwen3-32B-GGUF (931c84066f88), unsloth/Qwen3.6-35B-A3B-GGUF, unsloth/Qwen3.8-27B-GGUF, unsloth/Qwen3.8-Flash-Next-GGUF, unsloth/DeepSeek-V4-Flash-GGUF | as listed |
| lc-arch | https://github.com/ggml-org/llama.cpp/blob/d2e54583c7452353eb35d40431281f6ee984332f/src/llama-arch.cpp (arch strings :32-159; no `glm5next`, no `deepseek41`) | d2e54583 |
| lc-model | https://github.com/ggml-org/llama.cpp/blob/d2e54583c7452353eb35d40431281f6ee984332f/src/llama-model.cpp (rope types :2925-3080: deepseek2/deepseek4 NORM :2944-2975, qwen3/qwen3moe NEOX :2994-3051, qwen35/qwen35moe/qwen4exp IMROPE :3066-3070, glm4moe NEOX :3074) | d2e54583 |
| lc-conv-glm | https://github.com/ggml-org/llama.cpp/blob/d2e54583c7452353eb35d40431281f6ee984332f/conversion/glm.py (Glm4Moe :111-245, Glm4MoeLite → DEEPSEEK2 :247-250) | d2e54583 |
| lc-conv-ds | https://github.com/ggml-org/llama.cpp/blob/d2e54583c7452353eb35d40431281f6ee984332f/conversion/deepseek.py (DeepseekV2Model :225-400, DeepseekV4 :525) | d2e54583 |
| lc-conv-base | https://github.com/ggml-org/llama.cpp/blob/d2e54583c7452353eb35d40431281f6ee984332f/conversion/base.py (pre hashes: GLM-4.7-Flash → `glm4` :1618-1620, Qwen3 → `qwen2` :1648-1650, Qwen3.5 → `qwen35` :1861-1863) | d2e54583 |
| lc-conv-qwen4exp | https://github.com/ggml-org/llama.cpp/blob/d2e54583c7452353eb35d40431281f6ee984332f/conversion/qwen4exp.py | d2e54583 |
| ik-hp | https://github.com/ikawrakow/ik_llama.cpp/blob/f3d6e6e3020ddfebad60113845bf521620766da5/src/llama-hparams.cpp (n_rot default :236-238; qwen3 :567; qwen3moe :594; deepseek2 incl. GLM-4.7-Flash sigmoid fallback :1282-1330; deepseek4/41 :1976-2230; glm5next :2370-2470) | f3d6e6e3 |
| ik-bc | https://github.com/ikawrakow/ik_llama.cpp/blob/f3d6e6e3020ddfebad60113845bf521620766da5/src/llama-build-context.cpp (`llm_build_moe_ffn` :1453-1581) | f3d6e6e3 |
| ik-llama | https://github.com/ikawrakow/ik_llama.cpp/blob/f3d6e6e3020ddfebad60113845bf521620766da5/src/llama.cpp (rope types :9751-9849; glm5next NONE, deepseek2/4/41 NORM, qwen3/qwen3moe NEOX, qwen35/qwen35moe/qwen4exp IMROPE) | f3d6e6e3 |
| ik-ds4 | https://github.com/ikawrakow/ik_llama.cpp/blob/f3d6e6e3020ddfebad60113845bf521620766da5/src/graphs/build_deepseek4.cpp | f3d6e6e3 |
| ik-qwen3 | https://github.com/ikawrakow/ik_llama.cpp/blob/f3d6e6e3020ddfebad60113845bf521620766da5/src/graphs/build_qwen3.cpp | f3d6e6e3 |
| ik-glm5next | https://github.com/ikawrakow/ik_llama.cpp/blob/f3d6e6e3020ddfebad60113845bf521620766da5/src/graphs/build_glm5next.cpp | f3d6e6e3 |
| ik-vocab | https://github.com/ikawrakow/ik_llama.cpp/blob/f3d6e6e3020ddfebad60113845bf521620766da5/src/llama-vocab.cpp (`glm4` :2074, `qwen2` :2046, `qwen35` :2054, `joyai-llm` :2165) | f3d6e6e3 |
