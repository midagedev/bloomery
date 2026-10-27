# Metal for every model: the op map, the cover order, and a port proven on fixtures

Snapshot: main `85f17e393` (branch mac-allmodels), 2026-10-25. Design only: nothing here was built or run on a card or a
Mac GPU. Every number is a `wc -l`, a header reading, a measured number cited by place, or marked `[derived]` (with its
inputs) or `[unverified]`.

Revised the same day after an adversarial re-review and the lead's decisions on it (the last section says where each
finding landed). In the headline lines (§0 and the phase table), a number the first draft gave and the re-review
corrected is struck through, `~~old~~ new`; elsewhere it is corrected in place.

Sources used throughout:
- **Header readings** (HTTP ranges of the GGUF headers, every shard of a split file, no weights downloaded), 2026-10-25.
  Parser: a script kept outside the repo (single-shard files) and a shard-summing copy for split files (§7 item 5). The
  byte sums over every shard equal the files' sizes on the Hub, which checks the parse. Type tables per op category
  and bytes per token were computed from these readings by script.
- **Mac measurements**: the lead's Mac-numbers record, `mac-numbers.md` §4b (kept outside the repo; the same numbers are
  in `README.md:103-104`; M4 Pro 48 GB, 2026-10-25, 3 rounds, depth about
  0): bloomery decode **101.2 tok/s** on the MLX 4-bit folder and **85.6 tok/s** on the Q4_K_M GGUF of Qwen3.6-35B-A3B.
  `docs/contrib/mac.md` C.5 (M4 Pro: 273 GB/s datasheet, in-chain bandwidth fraction 0.96, so 262 GB/s; about 2.08 µs a
  bandwidth-bound dispatch in a chain) and `docs/contrib/mac.md:44` (M1 Pro 32 GB: `recommendedMaxWorkingSetSize`
  26,800,603,136 B, `maxBufferLength` 20,100,448,256 B).
- **Code** at the snapshot, cited as `path:line`.
- **Design references** at pinned commits: llama.cpp `74bca395` (`ggml/src/ggml-metal/kernels/`) and MLX `56c2bf92`
  (`mlx/backend/metal/kernels/`). Nothing is copied from them; §3 names which design is followed and where they differ.

## 0. The answer

The Mac does not redo the CUDA work. The host side a model runs on is shared already: GGUF reading, the model
description (`crates/models`: `ModelSpec`, `Need`, the shape tables), placement, the runtime, the tokenizer, the sampler,
the server, the reference-set readers and the fixture generator. By crate size that is 161,187 lines of Rust
(`gguf` 10,113, `models` 4,481, `placement` 23,015, `runtime` 21,111, `refset` 28,990, `tokenizer` 10,558, `sampler`
507, `serve` 47,532, `levers` 8,729, `device` 3,475, `chain` 2,676). Metal's own code for the one model it runs is
20,734 lines (MSL 6,382, kernel launchers 5,059, the `arch` directory 5,783 (the qwen3moe module 5,779 and `arch/mod.rs`
4), the core 3,192, `build.rs` 318), plus ~~30,760~~ 30,154 lines of gate and timing bins (`wc -l` of
`crates/metal/src/bin`). So 88.6 % of the lines a Metal model runs through are shared (161,187 of 181,921, by crate, not
by reach).

What a new model costs on Metal is kernels and glue:
- **The op map** (§1) has ~~75~~ 77 kernel needs over 18 models: ~~26~~ 28 weight-type instantiations and 49 op classes.
  **41 are shared by two or more models; ~~34~~ 36 belong to one model.** ~~16~~ 15 are on Metal today [derived: a script
  over the two tables as corrected, §1.3 "Counts"].
- **The cover order** (§2): kernel rounds ordered by models opened per line written, weighted by how many Mac sizes hold
  the file. ~~The first batch is shape-from-the-file plus Q8_0 and Q5_K. It opens five model files for about 1,200 lines
  [derived].~~ The first round is host-only, about 20 lines, and opens three model files (MiniCPM-V, Sage, Ornith). Then
  come the instruments, then Qwen3.6 UD (about 800 lines: the Q8_0 and Q5_K owners, the F32 site, the separate shared
  expert), then the instance mechanism and Qwen3-30B-A3B. lev, Clef and EmbeddingGemma are a seats track at weight 0
  until the user ranks it.
- **The total** to cover all 17 missing models is about ~~19,900~~ 20,600 new MSL and launcher lines [derived, §2 method;
  a floor, §2.2], ~~1.7~~ 1.8 times Qwen3.6's 11,441. On top of that comes about 2,000 lines of Metal glue for each new
  family (11 families) [derived from the qwen3moe arena and ops].
- **A typical new-family port** is about one third shared host code and two thirds Metal-own code [derived, §4]. The
  shared third is the family's plan and walk, moved from its CUDA crate into `crates/chain` (a move, proven by
  `canon-check`). The Metal-own two thirds are about 2,000 lines of glue and a median of ~~about 1,100~~ 1,268 lines of
  new MSL.
- **The move proof** (§3.4) is not a text scan. It is a per-pipeline hash of what the runtime builds (a binary archive per pipeline; an AIR hash where a Metal
  toolchain made the metallib), plus a
  per-token encode trace of the host dispatch. The first draft's `msl-scan` (MSL text equality) is unsound.
- **Proof without real files** (§5): every family can be declared working from fixtures, which are written from a
  file's header alone. A fixture made from a README file's own header carries that file's type mix. What a fixture
  cannot prove is memory fit, speed and the real weights' statistics. A header-only `generate_metal --plan` states the
  first as the room on an idle Mac and the session checks the live free memory at open (the census counts only this
  process: `crates/metal/src/census.rs:26-45`). The second is a predicted band, not a measurement. The last needs the
  real file, and real-file refsets exist only for the families named in §5.3; none exists for the MiniCPM-V, Sage or
  Ornith files, and Clef's are keyed to two other files than the README's (§5.2).

## 1. The op map

### 1.1 The files read (one README quant per model)

| col | model (seat, arch) | file read | bytes | layers | weight bytes a token [derived] |
|---|---|---|---|---|---|
| Q36 | Qwen3.6-35B-A3B (qwen3, qwen35moe) | lmstudio Q4_K_M | 21,166,757,728 | 40 (30 delta, 10 attn) | 1.976 GB |
| UD | Qwen3.6 type variant (census) | unsloth UD-Q4_K_XL | 22.349 GB | 40 | 2.753 GB |
| Q330 | Qwen3-30B-A3B (qwen3, qwen3moe) | Q4_K_M | 18.6 GB | 48 | 1.920 GB |
| kin | MiniCPM-V 4.7 text body (qwen3, qwen35moe; mmproj and vision are an off cell); Sage-A1.5 and Ornith have the same column, plus a nextn layer outside the trunk (`crates/model/src/arch/mod.rs:671`) | Q4_K_M | 21,166,255,008 (Sage 21,713,463,520) | 40 | 1.975 GB |
| Clef | Clef-Flash (decide, clef) | Q5_K_M | 6,982,686,880 | 32 (24 delta, 8 attn) | 6.143 GB (head on; decision head not counted) |
| lev | lev (decide, qwen35) | Q4_K_M | 3,032,093,632 | 32 | 2.664 GB (no `output.weight`) |
| Nano | Nemotron Nano 4B (nemotron_h) | Q4_K_M | 2,837,072,864 | 42 (21 Mamba-2, 4 attn, 17 MLP) | 2.547 GB |
| Ltn | Nemotron Lightning 30B-A3B (nemotron_h_moe) | UD-Q4_K_XL | 25,505,724,480 | 53 | 3.121 GB |
| Bolt | Nemotron Bolt 74B-A9B (nemotron_h_moe) | the 24GB file | 22,018,326,528 | 88 | 4.707 GB |
| Raon | Raon-1-mini (raon1) | KRAFTON Q4_K_M | 19,686,434,304 | 25 (18 RDA, 7 MLA) | 2.990 GB |
| Q38 | Qwen3.8-Flash-Next (qwen38, qwen4exp) | UD-Q4_K_XL, 4 shards | 111.324 GB | 48 (36 GDN, 12 attn) | 6.334 GB |
| MiMo | MiMo-V2.6-Flash (mimo2) | MXFP4, 2 shards | 167.363 GB | 48 | 10.649 GB |
| Sol | Solar-Open2-250B (solar-open2) | Q4_K_M, 4 shards | 151.810 GB | 48 (36 KDA, 12 GQA) | 7.413 GB |
| GLM | GLM-5.3-Flash (glm5next) | UD-Q4_K_XL, 6 shards | 199.698 GB | 46 (34 KDA, 12 MLA) | 13.977 GB |
| MMX | MiniMax-M3 (minimax-m3) | UD-IQ4_XS, 6 shards (M3.1-Flash: an NVFP4 folder of 250.1 GB, same shape class by its `config.json`) | 208.448 GB | 60 | 19.122 GB |
| Muse | Muse Spark (muse-spark) | UD-IQ2_XXS, 6 shards | 225.555 GB | 62 | 25.103 GB |
| V41 | DeepSeek-V4.1-Flash (ds41) | Q3_K_M, 9 shards | 347.266 GB | 40 | 7.774 GB |
| Gem | EmbeddingGemma 1/2 (embeddings side seat, `docs/serving.md:48`) | Q8_0 (`docs/serving.md:48`); header not read `[unverified]` | about 0.3 GB | — | — (one pass, no decode) |

Weight bytes a token [derived] counts every tensor read once a decode token: all non-routed matrices and vectors, top-k
of the routed stacks (`top_k / n_experts` of their bytes), and the head. It skips the gathered tables (one row a token:
`token_embd`, PLE, Engram), nextn/MTP layers and Clef's decision blocks. As a check, Qwen3.6's F32 routers come to
0.084 GB of the 1.976 [derived from the header: 40 × (256 × 2048 + 2048) × 4 B = 84.2 MB]; they are read as F32 at open
(`arena.rs:290`, `f32s_or_a8`).

### 1.2 Weight types per op category

Cell letters: `d` dense mat-vec (attention, delta, shared expert, dense FFN), `r` routed experts, `e` embedding rows,
`h` head, `t` a gathered table. Uppercase means on Metal today. `·` means not needed. The last column counts models.
Source: the header type tables. Metal's types today are Q4_K, Q6_K and MLX_A4 (`crates/metal/src/arch/qwen3moe/arena.rs:334`),
embed Q4_K|A4 (`:618`) and head Q6_K|A4 (`:625`). F32 small tensors are read as they are (`:305`).

| type | Q36 | UD | Q330 | kin | Clef | lev | Nano | Ltn | Bolt | Raon | Q38 | MiMo | Sol | GLM | MMX | Muse | V41 | Gem | models |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q4_K | DRE | r | dre | DRE | · | de | d | · | · | dre | r | · | dre | r | · | e | dr | · | 12 |
| Q6_K | DRH | r | drh | DRH | dh | · | d | · | · | dh | · | · | drh | r | rh | dh | h | · | 12 |
| Q8_0 | · | deh | · | · | d | d | dh | dreh | deh | dr | dreh | deh | · | deh | de | d | · | de | 13 |
| Q5_K | · | r | · | · | de | · | · | · | · | · | r | · | · | r | · | · | dr | · | 5 |
| Q5_0 | · | · | · | · | · | · | de | r | · | r | · | · | d | · | · | · | · | · | 4 |
| Q5_1 | · | · | · | · | · | · | · | r | · | · | r | · | · | · | · | · | · | · | 2 |
| Q3_K | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | dre | · | 1 |
| IQ2_XXS | · | · | · | · | · | · | · | · | dr | · | · | · | · | · | · | r | · | · | 2 |
| IQ3_S | · | · | · | · | · | · | · | · | · | · | · | · | · | · | r | · | · | · | 1 |
| IQ4_XS | · | · | · | · | · | · | · | · | · | · | · | · | · | · | r | · | · | · | 1 |
| IQ4_NL | · | · | · | · | · | · | · | · | · | · | t | · | · | · | · | · | · | · | 1 |
| MXFP4 | · | · | · | · | · | · | · | · | · | · | · | r | · | · | · | · | · | · | 1 |
| NVFP4 | · | · | · | · | · | · | · | · | d | · | · | · | · | · | · | · | · | · | 1 |
| F32 | · | d | · | · | d | · | · | · | · | · | · | · | · | · | · | · | · | · | 2 |

F32 as a mat-vec weight is the table's last row: `ssm_alpha` and `ssm_beta`, [2048, 32] in UD and [4096, 32] in Clef (lev's
are Q8_0). Metal's P1 segment takes quant types only (`arena.rs:322-338`, `:357-379`), so that site has no owner yet
(1c). Small sites outside the table: F32 vectors in 16 models (routers, decay vectors; Metal reads them today), BF16 in
Lightning's router, Qwen3.8's indexer projections and V4.1's routers, and F16 in Raon's hyper-connection and recurrent
vectors.
Every quantized type above has a dequantization port in `crates/gguf/src/quant.rs` (`dequant_row` `:356`; Q5_0 `:534`, Q5_1
`:563`, Q3_K `:649`, Q5_K `:750`, Q8_0 `:838`, BF16 `:878`, MXFP4 `:943`, NVFP4 `:1491`, IQ2_XXS `:1547`, IQ3_S
`:1652`, IQ4_NL `:1683`, IQ4_XS `:1707`), and the IQ grids are in `crates/gguf/src/iq_tables.rs`. So the f64 oracle of
every type exists already (§5.1).

### 1.3 Op classes

`M` means on Metal today, `n` needed, `·` not needed. Each row names its CUDA owner, from the coverage table
`crates/model/src/arch/coverage.rs` (an `AVAILABLE` row's `at`, cited by line), the chain walk
`crates/chain/src/arch/qwen3moe/walk.rs:151-184`, or the family crate.

| class | op | Q36 | UD | Q330 | kin | Clef | lev | Nano | Ltn | Bolt | Raon | Q38 | MiMo | Sol | GLM | MMX | Muse | V41 | Gem | models |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shape | vocab, hidden, heads, expert ff from the file | · | n | n | n | n | n | n | n | n | n | n | n | n | n | n | n | n | n | 17 |
| attn | GQA head 256, q·gate, partial rope 64, per-head norm | M | n | · | M | n | n | · | · | · | · | n | · | · | · | · | · | · | · | 6 |
| attn | GQA head 256, group 4 or 12 (packs) | · | · | · | · | n | n | · | · | · | · | n | · | · | · | · | · | · | · | 3 |
| attn | GQA head 128 (rope full, none or 64; groups 5/8/16) | · | · | n | · | · | · | n | n | n | · | · | · | n | · | n | · | · | · | 6 |
| attn | GQA out gate of its own projection | · | · | · | · | · | · | · | · | · | · | · | · | n | · | · | n | · | · | 2 |
| attn | GQA head 64 with a window pattern | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | n | · | · | 1 |
| attn | GQA K192/V128, window 128, sinks, value scale | · | · | · | · | · | · | · | · | · | · | · | n | · | · | · | · | · | · | 1 |
| attn | bidirectional GQA head 256 (no cache) | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | n | 1 |
| attn | latent/MLA core (absorbed heads, latent 512) | · | · | · | · | · | · | · | · | · | n | · | · | · | n | · | · | n | · | 3 |
| attn | MLA glue, GLM 256/256 | · | · | · | · | · | · | · | · | · | · | · | · | · | n | · | · | · | · | 1 |
| attn | MLA glue, Raon 192/128, pe, out gate | · | · | · | · | · | · | · | · | · | n | · | · | · | · | · | · | · | · | 1 |
| attn | V4.1 compress, stream index, grouped out | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | n | · | 1 |
| select | TokenPool indexer + key LayerNorm | · | · | · | · | · | · | · | · | · | · | · | · | · | n | · | · | · | · | 1 |
| select | BlockMax indexer (MSA) | · | · | · | · | · | · | · | · | · | · | · | · | · | · | n | · | · | · | 1 |
| select | MeanPool indexer (QSA) | · | · | · | · | · | · | · | · | · | · | n | · | · | · | · | · | · | · | 1 |
| rope | rotation by a host table (NeoX, imrope, NormTail; YaRN is in the table) | M | n | n | M | n | n | · | · | · | n | n | n | · | n | n | n | n | n | 14 |
| norm | RMS norm rows, norm on load | M | n | n | M | n | n | n | n | n | n | n | n | n | n | n | n | n | n | 18 |
| norm | per-head RMS (q/k norm) | M | n | n | M | n | n | · | · | · | · | n | · | · | · | n | n | · | n | 10 |
| norm | gated RMS norm, silu gate | M | n | · | M | n | n | · | · | · | · | · | · | · | · | · | · | · | · | 5 |
| norm | gated RMS norm, sigmoid gate | · | · | · | · | · | · | · | · | · | n | n | · | n | n | · | · | · | · | 4 |
| norm | pre/post expert norms | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | n | · | · | 1 |
| norm | post norms, layer scale, embed scale | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | n | 1 |
| router | softmax-norm top-k, no shared gate (128) | · | · | n | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 |
| router | softmax-norm gated, 256 top 8 | M | n | · | M | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 3 |
| router | softmax-norm gated, 512 top 10 | · | · | · | · | · | · | · | · | · | · | n | · | · | · | · | · | · | · | 1 |
| router | biased sigmoid top-k | · | · | · | · | · | · | · | n | n | n | · | n | n | n | n | n | · | · | 8 |
| router | biased sqrt-softplus + hash layers | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | n | · | 1 |
| mixer | gated delta net (conv 4) | M | n | · | M | n | n | · | · | · | · | n | · | · | · | · | · | · | · | 6 |
| mixer | KDA (sigmoid or softplus decay; RDA's write gate) | · | · | · | · | · | · | · | · | · | n | · | · | n | n | · | · | · | · | 3 |
| mixer | Mamba-2 step (state 128, conv 4) | · | · | · | · | · | · | n | n | n | · | · | · | · | · | · | · | · | · | 3 |
| ffn | SwiGLU expert pair + down fold | M | n | n | M | n | n | · | · | · | · | n | n | n | n | · | n | n | · | 12 |
| ffn | activation: ReLU² on the up projection alone (no gate) | · | · | · | · | · | · | n | n | n | · | · | · | · | · | · | · | · | · | 3 |
| ffn | activation: clamped SwiGLU (limit) | · | · | · | · | · | · | · | · | · | · | · | · | · | n | · | · | n | · | 2 |
| ffn | activation: swigluoai (alpha, limit) | · | · | · | · | · | · | · | · | · | · | · | · | · | · | n | · | · | · | 1 |
| ffn | activation: situ | · | · | · | · | · | · | · | · | · | n | · | · | · | · | · | · | · | · | 1 |
| ffn | activation: GeGLU | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | n | 1 |
| ffn | dense gated FFN layer (fixed route, one expert) | · | · | · | · | n | n | · | · | · | n | · | n | · | n | n | · | · | · | 6 |
| ffn | shared expert | M | n | · | M | · | · | · | n | n | n | n | · | n | n | n | n | n | · | 12 |
| ffn | shared expert of another type than its stacks (Apart) | · | n | · | · | · | · | · | · | · | · | n | · | · | · | · | · | · | · | 2 |
| ffn | latent MoE (latent down and up) | · | · | · | · | · | · | · | · | n | · | · | · | · | · | · | n | · | · | 2 |
| resid | mHC, 4 streams | · | · | · | · | · | · | · | · | · | · | · | · | · | n | · | · | n | · | 2 |
| resid | HcChart | · | · | · | · | · | · | · | · | · | n | · | · | · | · | · | · | · | · | 1 |
| resid | HcGated | · | · | · | · | · | · | · | · | · | · | n | · | · | · | · | · | · | · | 1 |
| resid | alpha-beta residual | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | n | · | · | 1 |
| extra | Engram table gate | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | n | · | 1 |
| extra | per-layer embeddings (PLE), IQ4_NL table rows | · | · | · | · | · | · | · | · | · | · | n | · | · | · | · | · | · | · | 1 |
| extra | Qwen3.8 out gate (f32) and shared add | · | · | · | · | · | · | · | · | · | · | n | · | · | · | · | · | · | · | 1 |
| extra | logit scale and softcap | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | n | · | · | 1 |
| extra | mean pool + dense projections | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | n | 1 |

The row sources (CUDA owner, `path:line`):
- **shape**: the instance tables `crates/models/src/shape.rs` `ROUTERS` (`:172`) and `GQA` (`:677`), looked up by
  `gqa_row`/`gqa_row_v` (`:869`, `:860`). The Metal blockers for the head are the host's: `arena.rs:602` (the vocabulary
  must be `HEAD_VOCAB` 248,320; MiniCPM-V has 248,144, Qwen3-30B 151,936, Raon 259,840) and the fixed `HEAD_GROUPS`
  (`arena.rs:678`, `crates/metal/src/kernels/head.rs:20,36`). The head kernel takes runtime rows and K with a tail guard
  (`crates/metal/kernels/head.metal:52-54,65,83`; `head.rs:46-50`: "a launch accepts any positive count"). The others are
  `crates/metal/src/kernels/attention_geom.rs:2-14` and `arena.rs:486` (`EXPERT_K`).
- **GQA head 256 gated**: `coverage.rs:280` (`rope_neox.rs head_norm_neox_append_256`), `:295` (`dispatch.rs gated_256`);
  the packs `:488` (`flash_gqa.rs PACK_4`, `shape.rs` the `_256_p4` row). Clef and lev are 16 q / 4 KV (group 4),
  Qwen3.8 is 24 q / 2 KV (group 12), Qwen3.6 is 16/2 (group 8).
- **GQA head 128**: `coverage.rs:124` (`shape.rs GQA`), `:145` (`gqa_flash_seg_g5`, Nano's group 5), `:162`
  (`GROUP_16`, Lightning, Bolt, MiniMax), `:228` (`rope_neox.rs HEAD`, Qwen3-30B's full NeoX rope), `:385`
  (`hybrid.rs gqa_layer`, Solar's NoPE layer). Nemotron-H's attention is NoPE too
  (`crates/gpu/src/arch/nemotron_h/mod.rs:3`, `crates/gpu/src/arch/hybrid.rs:20`).
- **out gate proj**: `coverage.rs:402` (`hybrid.rs GqaGate::Proj`), `:876` (Muse).
  **head 64**: `:841`. **K192**: `:605`, `:623`, `:641`. **bidirectional**: `:656`.
- **latent/MLA**: `coverage.rs:51` (`gpu-deepseek41/src/attn.rs LATENT`), `:315` (`gpu-glm5next/src/mla.rs` over it),
  `:745` (`gpu/src/arch/raon1/mla.rs` over `gpu/src/latent.rs LATENT`), `:760` (pe), `:765` (out gate).
  **V4.1**: `:65` (`compress.rs WIDTH`), `:70` (`indexer.rs`).
- **selectors**: TokenPool `:329`, key LN `:344` (`latent.rs INDEX_HEAD`); BlockMax
  `crates/gpu-minimaxm3/src/attn.rs:1-20` (no `AVAILABLE` row names `Program::MiniMaxM3Body`; see §7); MeanPool `:525`
  (`qsa.rs`).
- **rope**: the host table `crates/chain/src/rope.rs:1-10` (ggml's `rope_cache`; YaRN is one more fused multiply-add
  in it), so a rope kind is a mode of one rotation, not a kernel. Today's Metal rotation is fused into
  `attn_norm_rope_rows` (`crates/metal/kernels/attention.metal:20-26`).
- **norms**: `crates/gpu/src/elem.rs:1015` (`rms_norm`); per-head `rope_neox.rs`; gated `crates/gpu/src/linear/norm_gate.rs`
  with Qwen3.8's sigmoid gate `coverage.rs:469`; Muse `:902`, `:907`; Gemma `:695-722`.
- **routers**: `shape.rs:172-420` (`Qwen3moe` 128, `Qwen35moe` 256 and 512, `Ds41` 384, `BiasedSigmoid` at
  288/256/240…496/320/128×4/128×6/256×6); `coverage.rs:243`, `:540`, `:179`, `:412`, `:887`, `:75`.
- **mixers**: GDN `coverage.rs:248` (`plan.rs delta_shape`); KDA `:349` (`linear/mod.rs`), `:367` (`mixer_kda.rs`, the
  softplus decay), `:726` (`raon1/rda.rs`: the common KDA mixer with a write gate); Mamba-2 `:103` (`mamba2.rs`,
  `STATE 128`, `CONV_TAPS 4`). Clef, lev and Qwen3.6 have the same delta geometry (inner 4096, 16 groups, state 128,
  32 value heads, conv 4, from their headers). Only the hidden width differs (2048, 4096, 2560). Qwen3.8 does not: 48
  value heads and inner 6,144 (its header), so it is a second delta instance (`VALUE_HEADS` is compile-time:
  `arena.rs:973`). The CUDA KDA entries sit inside `linear/delta.rs` (`kda_delta` `:407`, `kda_delta_lanes` `:519`) and
  `linear/conv.rs` (`kda_conv_prep`, `:575-770`); `mixer_kda.rs`, `norm_gate.rs` and `write_gate.rs` are the glue around
  them.
- **FFN**: the chain's `FfnRoute::Dense` (`crates/chain/src/arch/qwen3moe/plan.rs:146-150`: "one slot a token, on
  expert 0 of the one-expert stacks, at weight 1") and `coverage.rs:300`; dense in other families `:436`; ReLU² `:119`,
  `:194`; situ `:770`, `:776`; GeGLU `:690`; latent MoE `:188`, `:896`; shared `:267`, `:423`, `:545`, `:790`. The
  activations are five epilogues beyond plain SwiGLU (`crates/models/src/lib.rs:647-661`): ReLU² with no gate (Nano's MLP
  is `down(relu²(up(x)))`, `crates/gpu/src/arch/nemotron_h/step.rs:51`, so it is not the one-slot SwiGLU of the dense
  row), the clamped SwiGLU (`crates/model/src/arch/deepseek41/spec.rs:221`, `glm5next/spec.rs:93-110`), swigluoai
  (`minimaxm3/spec.rs:22-25`), situ and GeGLU. `SharedPlan::Apart`, a shared expert of another type than its stacks, is
  refused today at `arena.rs:1074`, and the one-type rules for gate/up and for down at `:1109` and `:1117`; UD and
  Qwen3.8 are both Apart (their headers: a Q8_0 shared expert beside Q4_K or Q5_K gate and up stacks).
- **not a Metal need**: Clef's decision head is host f32 (`crates/decision/src/head.rs:1`), so it has no row, and neither
  has the LayerNorm its blocks own. GLM's key LayerNorm is inside the TokenPool row (`coverage.rs:344`). lev's score is
  one prompt pass and a host readout of the label logits (`crates/decision/src/label.rs:1-5`).
- **residual and extras**: mHC `:93`, `:441`; HcChart `:818`; HcGated `:558`; AlphaBeta `:913`; Engram `:98`; PLE `:576`;
  Qwen3.8 out gate `:520`; softcap `:919` (`head.rs enqueue_with_tail`).

**Counts.** There are 77 kernel needs: 28 type instantiations (17 shared) and 49 op classes (24 shared, 25 family-only).
**41 are shared by ≥ 2 models and 36 are family-only.** 15 are on Metal today. [derived: a script over the two tables
above. Types: 27 (16 shared) from the type table, plus the F32 row, 28 (17). Op rows: 46 (23 shared) from the op table
as first drafted, less Clef's decision head and the LayerNorm row, plus the activation row split into five and the Apart
row, 49 (24 shared). Metal today: 6 type instantiations and 9 op rows.] 11 quant decode owners are missing, and 5 of them
are shared (Q8_0, Q5_K, Q5_0, Q5_1, IQ2_XXS); F32 as a mat-vec weight is a twelfth site without an owner. The family-only
rows are the model's own architecture facts: four sparse selectors, HcChart, HcGated and Muse's alpha-beta residual,
Engram, PLE, Muse's epilogue, three activations (swigluoai, situ, GeGLU), and the attention geometries only one file has
(head 64, K192/V128, the bidirectional pass, the two MLA glues, V4.1's compressor). Everything else (types, the GQA
classes, the delta-rule mixers, the routers, mHC, the FFN rules) is shared.

## 2. The cover order

### 2.1 Weights: which Macs hold each file whole

The working set is 0.78 × RAM, measured on three Macs: `recommendedMaxWorkingSetSize` over RAM reads 0.7800002 on the
two M1 Pro 32 GB and 0.7800001 on the M4 Pro 48 GB (`docs/research/metal-offload.md` §1.1). It is `[unverified]` above
48 GB: no larger Mac was measured. RAM tiers are GiB, files are 10⁹ B, and 1 GB is left for the caches, the recurrent
state and the scratch [derived: Qwen3.6's KV at 8,192 positions is 10 layers × 2 × 2 heads × 256 × 2 B × 8,192 = 168 MB,
and its delta state is 30 × 32 × 128 × 128 × 4 B = 63 MB]. Room per tier (GB) [derived: RAM × 2³⁰ × 0.78 / 10⁹ − 1]:
32 → 25.8, 48 → 39.2, 64 → 52.6, 128 → 106.2, 256 → 213.4, 512 → 427.8.

| weight (tiers that hold it) | models |
|---|---|
| 6 (every tier, 32 GB up) | Q36, UD, Q330, kin (three models), Clef, lev, Nano, Ltn (25.5 GB against 25.8 GB of room), Bolt, Raon, Gem |
| 2 (256 GB up) | Q38, MiMo, Sol, GLM |
| 1 (512 GB only) | MMX, Muse, V41 |

MMX is counted as the README's M3.1-Flash NVFP4 folder (250.1 GB, `README.md:31,133`), which the 256 GB tier does not hold.
The M3 UD-IQ4_XS file read in §1.1 (208.4 GB) does. As that file it would have weight 2 and, in the last round, would go
before GLM (2 / 3.117 kLOC = 0.64 against GLM's 2 / 3.912 = 0.51 per kLOC [derived from §2.2's glue column]). Ltn moves
from weight 5 to 6 at 0.78. The order of the earlier rounds does not change.

A file over 18.72 GiB also needs more than one buffer on a 32 GB Mac (`maxBufferLength` 20,100,448,256 B,
`docs/contrib/mac.md:44`). The arena already gives each weight its own buffer (`arena.rs:1-10`), so no single tensor
comes near that. Offload, the sibling round's design, raises the weights of the big four. This order counts only whole
residency.

### 2.2 The order

The greedy: at each step it picks the model with the most weight per line still to write, builds every missing row it
needs, and counts every model those rows complete. A new family also pays about 2,000 lines of Metal glue: the
qwen3moe module's arena (1,168) and ops (1,340) are 2,508, less the part of the arena a shape-driven binder shares
[judgment]. The session's step modes are not counted, on the condition that they become common first (§4.3). The first
draft's order was re-cut on 2026-10-25 (the re-review and the lead's decisions): kin first as host-only work, Qwen3.6 UD
before the instance mechanism, and lev, Clef and EmbeddingGemma out to a seats track at weight 0.

Line estimates [derived], one method per row kind:
- **A quant owner**: 180 lines, plus 30 per category it is instantiated in. This is measured on today's owners: Q6_K is
  `quant.metal:325-472` (147 lines), A4 is `:484-637` (153), and one instantiation file is `matvec_q6k.metal` (60 lines,
  all modes). IQ2_XXS and IQ3_S add a grid table (300 in all).
- **An op class**: 0.62 × the CUDA owner's `wc -l`. 0.62 is the ratio measured on the one class ported both ways: Metal
  `delta.metal` 926 + `kernels/delta.rs` 755 = 1,681 against CUDA `linear/delta.rs` 1,431 + `linear/conv.rs` 1,293 =
  2,724. Those two CUDA files also hold the KDA entries and their tests (`delta.rs` `:407`, `:519`, `:610`, `:759`, `:1174`;
  `conv.rs` `:575-770`, `:1057-1293`), so the ratio against GDN alone is above 0.62: the estimates that use it are a
  floor, not a ceiling. A GDN-only ratio needs the CUDA files split by entry, which was not done. For attention the ratio
  is 0.27 (Metal 1,310 + 808 against `flash_gqa.rs` 7,858, which holds every head size), a ratio against a file that
  serves many head sizes. CUDA file counts include launchers and tests.
- **Shape rows and small flags**: judged from the Metal files named in §3, marked as judgment in the table below.

| step | opens (weight) | new MSL + launchers | with glue | weight per kLOC | rows built |
|---|---|---|---|---|---|
| 1-0 | kin: MiniCPM-V, Sage, Ornith (18) | 20 (judgment: host only, no MSL) | 20 | 900 | the host's vocab refusal (`arena.rs:602`) and fixed `HEAD_GROUPS` (`:678`) lifted; the head kernel takes runtime rows with a tail guard already; the fixture COPY gap (`rope.section_order`) |
| 1c | UD (6) | 800 (judgment) | 800 | 7.5 | Q8_0 and Q5_K owners with the 16-value GEMM dequant; the prompt GEMM entries (gemm, routed, tensor); the F32 P1 segment (`ssm_alpha`, `ssm_beta`); `SharedPlan::Apart` as one common dispatch arm |
| 1b-0 | none (a move) | 600 (judgment: `build.rs` 318 + `*_geom.rs` 277 regenerated as a function of the `models::shape` row at build time) | 600 | 0 | the instance mechanism; Qwen3.6 the single instance |
| 1b-1 | Q330 (6) | 600 (judgment) | 600 | 10.0 | GQA head 128 (attention rewrite: the norm-rope pairing and rows); softmax-norm router without the shared gate |
| seats | lev (6) | 360 | 360 | 0 (nominal 16.7) | GQA packs (group 4); dense FFN as a fixed route; Q8_0 owner counted in 1c; a Metal seat engine and headless open, not sized |
| seats | Clef (6) | at most 640 | at most 640 | 0 (nominal 9.4) | Q5_K mat-vec and embed (counted in 1c); the first draft's LayerNorm and decision head leave the count (host f32), so the row is not re-sized `[unverified]` |
| seats | Gem (6) | 660 | 2,660 | 0 (nominal 2.3) | bidirectional GQA; Gemma norms; GeGLU; pooling |
| 7 | Nano (6) | 1,268 | 3,268 | 1.8 | Q5_0 owner; Mamba-2 step (`mamba2.rs` 1,658) |
| 8 | Bolt (6) | 1,278 | 3,278 | 1.8 | IQ2_XXS and NVFP4 owners; biased-sigmoid router (`route_body.rs` 948); latent MoE |
| 9 | Ltn (6) | 330 | 330 | 18.2 | Q5_1 owner; BF16 sites |
| 10 | Raon (6) | 2,816 | 4,816 | 1.2 | KDA glue (`mixer_kda.rs`, `norm_gate.rs`, `write_gate.rs`; the core is a variant of `delta.metal`, not sized `[unverified]`, so this row is a floor); MLA core (`latent.rs` 1,185); Raon's MLA glue (`raon1/mla.rs` 1,444); HcChart (`raon1/mhc.rs` 725); F16 sites |
| 11 | Sol (2) | 128 | 2,128 | 0.9 | out gate of its own projection (`linear/out_gate.rs` 206) |
| 12 | MiMo (2) | 756 | 2,756 | 0.7 | MXFP4 owner; K192/V128 window, sinks and value scale (`flash_gqa_h128_swa.rs` 558 + rope K192) |
| 13 | GLM (2) | 1,912 | 3,912 | 0.5 | GLM's MLA glue (`gpu-glm5next/src/mla.rs` 365); TokenPool and key LN; mHC (`gpu-deepseek41/src/hc.rs` 2,119) |
| 14 | Muse (1) | 310 | 2,310 | 0.4 | head-64 window instance; expert norms; alpha-beta; softcap |
| 15 | Q38 (2) | 3,803 | 5,803 | 0.3 | IQ4_NL; MeanPool (`qsa.rs` 1,604); HcGated (`hc_gated.rs` 2,590); PLE (`ple.rs` 1,053); `q38.rs` gate; router 512 |
| 16 | MMX (1) | 1,117 | 3,117 | 0.3 | IQ3_S, IQ4_XS owners; BlockMax (`gpu-minimaxm3/src/attn.rs` 931) |
| 17 | V41 (1) | 3,250 | 5,250 | 0.2 | Q3_K owner; compress, indexer, latent attention (`compress.rs` 1,074 + `indexer.rs` 993 + `attn.rs` 1,230); hash router (`router.rs` 878); Engram gate (`engram_gate.rs` 632) |

Sum: 20,648 new MSL and launcher lines, plus 22,000 lines of glue for 11 families [derived: the column above].

The rounds in order (each is what §6 cuts into rounds):
1. **1-0 kin, host only** (about 20 lines, no MSL). Opens MiniCPM-V (text body only), Sage and Ornith: 18 weight for 20
   lines. Every missing model needs "a shape from its file" (17 of 17 columns), but the head kernel takes runtime rows
   already, so this class needs no instance: only the host's refusals change.
2. **1a instruments**: `generate_metal --plan`, Metal's coverage rows, and the move proof's tools (§3.4).
3. **1c Qwen3.6 UD** (about 800 lines, add class). Opens UD. Its new expert lanes are written to the offload doc's
   slot-table signature, so `experts.metal` is touched once.
4. **1b-0 the instance mechanism** (a move, 600 lines), then **1b-1 Qwen3-30B-A3B** (about 600 lines). Opens Q330.
5. **The seats track** (lev, Clef, EmbeddingGemma): weight 0 until the user ranks it. Its metric is prompt throughput and
   request latency, not decode tok/s; it needs a Metal seat engine and a headless open (§6).
6. **Nemotron** (steps 7-9: Mamba-2, Q5_0/Q5_1, IQ2_XXS, NVFP4, the biased-sigmoid router, ReLU², latent MoE). Opens
   Nano, Bolt, Lightning, with the first chain lift (§4).
7. **The KDA and MLA families** (steps 10-11). Opens Raon, then Solar2 for 128 more lines.
8. **MiMo and Muse** (steps 12, 14).
9. **The 256/512 GB set** (steps 13, 15-17): GLM, Qwen3.8, MiniMax, V4.1. They can move earlier once offload lets them
   run on smaller Macs.

Order against the sibling offload doc (`docs/research/metal-offload.md`): model breadth first (rounds 1-4), the offload
identity round after. The first real offload customer is Qwen3.6 (Q4_K_M and UD) on a 16 or 24 GB Mac, which needs 1c,
and the identity round's A/B base moves when 1b-1 changes attention, so offload is measured once after breadth. The
offload doc fixes the slot-table kernel signature (a slot-index table in place of expert ids, the identity today) in one
paragraph now, so that 1c writes its lanes to it. `experts.metal`, `arena.rs` and `session.rs` have one owner at a time.

## 3. Shape from the file

### 3.1 What is compile-time today, and what each axis becomes

`build.rs` writes `#define BLOOMERY_*` from `src/kernels/{geometry,delta_geom,router_geom,experts_geom,attention_geom}.rs`
into the source (`crates/metal/build.rs:93` `geometry_header`, `:176` `combined_source`). The library is compiled from
that text at load, with fixed options (`crates/metal/src/library.rs:280-291`: safe math, precise functions,
language 3.2), or from an offline metallib when the Metal Toolchain is installed (`build.rs:211`). One axis is already a
function constant: the mat-vec's SIMD groups per threadgroup (`crates/metal/kernels/norm.metal:54`, set by
`Library::pipeline_with_simdgroups`, `library.rs:360`).

| kernel class | today's constants | the file's source | becomes | why |
|---|---|---|---|---|
| mat-vec, all modes | rows, cols already runtime (`MatvecParams`); `ROWS_PER_SIMD_GROUP` 4; SIMD groups by function constant 0 | tensor dims | unchanged | rows and cols are trip counts; the lane map is the type's |
| head argmax | `HEAD_VOCAB` 248,320, `HEAD_K` 2,048, exact-multiple asserts (`head.rs:20-40`) | `output.weight` dims | **kernel unchanged**: rows and K are launch arguments with a tail guard (`head.metal:52-54,65,83`; `head.rs:46-50`); the host drops its fixed-vocab refusal (`arena.rs:602`) and sizes `candidates` from the vocabulary (`:678`) | MiniCPM-V's 248,144 rows make 7,755 groups against 7,760 [derived: 248,144 / 32, rounded up] |
| embed | K | hidden | runtime | trip count |
| norm rows | `args.cols` runtime already (`norm.metal:124`) | hidden | unchanged | |
| experts pair and fold | `EXPERT_SLOTS` 9, `ROUTED_IDS` 8, `EXPERT_K` 512 (`quant.metal:431` folds by it) | top-k + shared, expert ff | **K compile-time per instance**; slots from the instance header | a runtime K puts integer division and modulo on the address path (§3.2); the fixture rule below |
| dense FFN | refused (`arena.rs:471`) | `feed_forward_length` | the experts kernels at one slot (`FfnRoute::Dense`) with the instance's K | no new kernel for a gated FFN; Nano's up-only ReLU² MLP is not this path |
| router | `EXPERTS` 256, `ROWS` 257, K 2,048, `TOP_K_MAX` 8 (`router_geom.rs`) | expert count, top-k, shared gate | instance header from the `ROUTERS` row, **with the hidden width in the instance key** (a `ROUTERS` row holds no hidden: `shape.rs:88-130`; `arena.rs:597` ties hidden to `router::K`); rule by function constant | per-lane expert arrays are register-sized |
| attention | `HEAD_DIM` 256, `ROT` 64, 16 q, 2 KV, `TILE`, `SIMD_GROUPS` 8, `SEGMENT_KEYS`, `SCALE` (`attention_geom.rs:2-14`) | key/value length, heads, rope dims | instance header from the `GQA` row; gate, sinks, window, value scale by function constant | q, k and accumulator arrays are sized by the head |
| attention norm + rope rows | the square pairing written for head 256 (`attention.metal:20-26`) | head, rot | a head-generic pairing (a rewrite for 128 and 64), instance header | |
| delta (GDN) | `CONV_TAPS` 4, `PASS_ROWS` 8, `SNAPS` 2, value heads 32, head cols 128 (`build.rs:96-106`) | `ssm.*` keys | instance header | equal across Qwen3.6, Clef and lev; **Qwen3.8 is another instance** (48 value heads, inner 6,144: its header; `VALUE_HEADS` is compile-time, `arena.rs:973`) |
| rope | `ROT`, imrope sections, one base (`arena.rs:447-460`) | rope keys | host table (`crates/chain/src/rope.rs`) + mode by function constant | |

**The fixture rule.** A fixture keeps every per-layer shape and the source's globals, except the tables a family resizes
and the routed stacks its ff override narrows (`crates/model/src/fixture/mod.rs:1-30`; glm5next and deepseek41 narrow
expert ff to 512). So:
- An axis a fixture narrows (expert ff, table rows, layer count) **must be a runtime trip count**. Then the fixture
  runs the same pipeline as the real file, and a fixture proof carries over.
- An axis a fixture keeps (head dims, heads, groups, experts, hidden, vocab) may be compile-time.

The expert K is where this rule and §3.2 pull apart. K stays compile-time inside an instance. For the qwen3 family that
is consistent with the rule, because its fixtures keep the source's expert ff (`crates/model/src/arch/qwen35moe/fixture/qwen35moe.rs:97`,
`ff: None`). A family whose fixture narrows ff (glm5next and deepseek41, to 512) would run another K than its real
file. Either the narrowed K is a second registered instance (the fixture then proves the shared kernel text and the
clause band, not the real file's pipeline), or K is runtime there and pays the address-path term of §3.2. That is decided
when such a family is ported (Phase 7), with the term priced then `[unverified]`.
Lead's call (2026-10-25): the narrowed K is a second registered instance, so the real file's pipeline keeps its
compile-time K; the fixture proves the walk and the shared kernel text, and a `lab-metal` clause at the real file's K (real-size
synthetic tensors against the f64 oracle) proves the real instance's kernel. Runtime K only if that family's measured c_L
term says it costs nothing.

### 3.2 The cost model

One decode token on Metal is one command buffer per token, with the next token committed before the wait in chain
mode (`crates/metal/src/arch/qwen3moe/session.rs:1-16`). The host's encode sits under the GPU's run, so the wall is the
GPU's:

  t_token ≈ B / 262 GB/s + L × c_L [derived]

Here B is the weight bytes a token (§1.1) and 262 GB/s is the in-chain bandwidth (C.5). c_L = 0.104 ms is the
per-layer remainder, fitted to Qwen3.6 Q4_K_M: 85.6 tok/s is 11.68 ms, the bytes take 1.976 GB / 262 GB/s = 7.54 ms,
and the rest is 4.14 ms over 40 layers. The other edge of the band is one effective rate, 169 GB/s (1.976 GB in
11.68 ms).

A runtime row or column count adds one compare-and-branch per block iteration of a lane. At 262 GB/s one 144-byte Q4_K
block is 0.55 ns of memory time. A bandwidth-bound mat-vec keeps that branch under the load's shadow, so its term is 0.
**Runtime rows and columns cost nothing on mat-vec, head, embed and norm, as long as the loop structure stays.** The
head already takes its rows and K at run time (`head.metal:52-54,65,83`).

The expert K is not that case. The fold maps a block index to its slot with `ib / per_slot` and `ib % per_slot`
(`quant.metal:276-280`, `:431-435`, `:605-609`; `experts.metal:173-175`), which a runtime K turns into integer division
and modulo on the address path. The shadow argument needs the step to be bandwidth-bound, and c_L is not: 4.14 ms of the
11.68 ms token is 35 % [derived: 4.14 / 11.68] outside the byte time, and the busiest unit is not the bound until the
step's cycles are its demand (AGENTS.md). So K stays compile-time per instance, and the router's K (the hidden width)
joins the instance key (`arena.rs:597`). The other fix, the host computing each slot's block offset into `FoldArgs`, is
not taken.

The kernels that need compile-time specialization to keep Qwen3.6's speed are those whose register arrays are sized by
the axis: attention (head dims, group unroll), the router (experts a lane holds) and delta (value heads, head cols).
A runtime size there means dynamic indexing of per-thread arrays [unverified on Apple's compiler; both references avoid
it, below]. These get the instance header.

### 3.3 The mechanism: one instance per registered shape row

The design:
- `geometry_header` becomes a pure function of a shape instance, a `models::shape` row and the file's per-model
  constants. The registered rows are finite, so every instance is generated and compiled at **build time**: `build.rs`
  writes one metallib per instance (offline `xcrun metal -c`, as it does for Qwen3.6 today, `build.rs:208-236`). The
  first round of the mechanism (1b-0) builds one instance, Qwen3.6's, and moves nothing else.
- The expert K stays compile-time inside an instance (§3.2). The router's K joins the instance key (`arena.rs:597`).
- ~~A file whose shape has no registered instance is refused by name through the same `gqa_row`/`router_row` lookups CUDA
  uses. There is no load-time compile path for an unregistered shape: the named refusal is the contract and the
  contributor's route (add a `models::shape` row, and the instance follows).~~ Superseded 2026-10-27: the open renders
  the instance's header from the file's `Geometry` (`crates/metal/src/kernels/header.rs`), the same text `build.rs`
  writes for a registered row. A geometry a kernel cannot take is refused by name (`Geometry::check`); one every kernel
  takes but no row registers compiles from source at open (`instance::open_key`), and the binaries say no gate checked
  that shape. A registered row adds the metallib and the gates. The source path (`LibraryPath::Source`) is also the
  path of a Mac without the Metal toolchain.
- A file that needs two instances of one class links two metallibs; MiMo's full (4 KV) and window (8 KV) layers are the
  one case in the map. Pipelines are cached by (instance, entry, function-constant values).
- The price is `time xcrun metal -c` on `OUT_DIR/bloomery.metal` once, times the instance count `[unverified until
  measured]`. It is paid at build time, not at load.

So both backends compile exactly the rows of `models::shape`, and the shape table stays the one owner of "which
geometries exist". Booleans and small counts (gate, sinks, window, value scale, rope mode, router rule) become function
constants, as `BLOOMERY_MATVEC_SIMD_GROUPS` is today.

Where the references stand (pinned commits):
- **llama.cpp** instantiates a template per head size under a host name, for example
  `template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv96")]]    kernel flash_attn_ext_vec_t` (`fa_vec_f16.metal`,
  twelve DK/DV pairs from 32/32 to 576/512). Flags and counts are function constants:
  `constant bool FC_flash_attn_ext_vec_has_sinks [[function_constant(FC_FLASH_ATTN_EXT_VEC + 1)]];`
  (`fa_vec_common.metal`) and `constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];` (`mul_mv.metal`).
  https://raw.githubusercontent.com/ggml-org/llama.cpp/74bca3950054eed1795f525cb9cf06bfe796492d/ggml/src/ggml-metal/kernels/fa_vec_f16.metal
- **MLX** makes head dims template ints, `template <typename T, int D, int V = D>`, and flags and counts function
  constants, `constant bool has_sinks [[function_constant(25)]];` and `constant int window_size [[function_constant(27)]];`
  (`sdpa_vector.h`). https://raw.githubusercontent.com/ml-explore/mlx/56c2bf9236d929097f61e2c6e55be4e4bb6da249/mlx/backend/metal/kernels/sdpa_vector.h
- **Apple**: "A set of constant values that specialize a graphics or compute GPU function."
  (https://developer.apple.com/tutorials/data/documentation/metal/mtlfunctionconstantvalues.json); `preprocessorMacros`,
  "A list of preprocessor macros to apply when compiling the library source." (macOS 10.11,
  https://developer.apple.com/tutorials/data/documentation/metal/mtlcompileoptions/preprocessormacros.json).

Both references agree with this design on the split: array-sizing axes are compile-time, flags are function constants.
**Ours differs** in where the compile-time value comes from and when it is compiled. They compile one library with every
instance named, at run time. We generate one translation unit per registered instance from a header, at build time. That
mirrors CUDA's one compiled entry per shape row. That Qwen3.6's pipelines did not move is proven by §3.4's move proof,
not by comparing the generated text. A binary archive can cache the source fallback's compile ("A container for pipeline
state descriptors and their associated compiled shader code.", macOS 11.0,
https://developer.apple.com/tutorials/data/documentation/metal/mtlbinaryarchive.json).

### 3.4 The move proof: what shows that Qwen3.6's pipelines and dispatch did not move

The first draft proposed `msl-scan`, a pinned table of the md5 of each entry's complete MSL compiler input. It is unsound
for three reasons:
- The compiler input of the main library is one translation unit: `build.rs:175-176` (`combined_source`) concatenates the
  generated header and every kernel file. A new owner added to `quant.metal` therefore changes the text of every entry,
  so "the base's rows plus exactly the new rows" cannot be printed. The tensor-op GEMM library is a second, small unit
  (`fault`, `quant`, `gemm_shared` and `gemm_tensor`, `build.rs:293-309`), and `quant.metal` moves it too. A runtime K would also have changed the text Qwen3.6 compiles
  (`quant.metal:276-280`, `experts.metal:173`).
- The host dispatch is outside the text: the grid, the bindings, the function-constant values (`library.rs:360`) and
  whether the metallib or the source path is loaded.
- Equal text gives an equal program only under equal options.

What replaces it, for any change that claims to be a move:
1. **A per-entry AIR hash taken from the library the runtime actually loads.** That is the embedded metallib (else the
   source at language 3.2) and, on a GPU with matrix units, the tensor-op GEMM library compiled at run time from
   `TENSOR_SOURCE` at language 4.0 (`library.rs:17-23`); the second has no metallib, so its hash is of an offline compile
   at the same options. The hash input includes the compile options. One proof that the runtime's options (`source_of`: safe math, precise functions, the language version,
   `library.rs:280-291`) and the `xcrun` flags (`build.rs:208-236`) give the same program. A determinism probe first:
   compile twice, and again on another day; whether Apple's AIR output is deterministic is `[unverified]`, and without it
   the hash proves nothing. `xcrun metal -c` needs the Metal toolchain but no GPU.
   **Fact (2026-10-25): neither the lead Mac nor the M4 Pro testbed has the Metal toolchain** (`xcrun -sdk macosx metal
   --version` answers "missing Metal Toolchain"), so their builds take the run-time source path (`build.rs:314-317`) and
   no AIR exists there. On them the hash is of each pipeline serialized to its own `MTLBinaryArchive` (the device
   compiler's output, at the engine's function-constant values), which needs the GPU and no toolchain. The metallib path
   that users with the toolchain run is untested on these Macs: a coverage gap the Mac docs name.
2. **A per-token encode trace**, recorded under the existing tracer wrapper (`session.rs:1740` brackets each `ChainOps`
   op and reads arena nodes; it records no encoder state today, so the recording layer is new). Per encoded dispatch: the
   pipeline, the grid, the bindings, the function-constant values, the `setBytes` contents (`FoldArgs`, `MatvecParams`),
   the buffer offsets, the threadgroup memory length and `maxTotalThreadsPerThreadgroup`; per token: the command buffers
   and their commit and wait order. Taken at depth about 0 and at one deep depth.
3. Bit-identical outputs of every `lab-metal` clause and of `lab-metal-e2e` between the base and the new build.

Wording: after a move that passes 1 to 3, "85.6 and 101.2 tok/s stand by the move proof; the next round that touches the
dispatch path re-measures". A same-sitting A/B on the M4 Pro is the proof only for a change that fails 1 or 2 (§6 1b-1
budgets one from the start). New instances and new entries are the add class: the base's hashes unchanged, new ones added.

## 4. Chain lift: what is shared host code

### 4.1 What exists

`crates/chain` holds device-free plans and walks. Its one whole-model plan, `arch/qwen3moe` (1,897 lines: plan 652,
read 572, walk 460, router 114, ops 89, mod 10), covers qwen3moe, qwen35moe and the qwen35 dense body. Its op trait
`ChainOps` (`crates/chain/src/arch/qwen3moe/ops.rs:54-88`) has embed, attn_in/core/out, delta_in/core/out, route,
experts and head. The walk calls them in layer order (`walk.rs:151-184`). Both backends implement the trait: CUDA's
`crates/gpu/src/arch/qwen3moe/program.rs:195` and Metal's `crates/metal/src/arch/qwen3moe/ops.rs:399` (with
`ops/`, 666 + 674 lines).

The other families walk through `runtime::sched::walk` (`crates/runtime/src/sched.rs:377`) over a `LayerProgram`
(`:330`: begin, front, shadow, back, end). That is already a common, host-only walk of the tier legs. Each family's
program mixes its layer's op order with its CUDA launches in one impl, for example `crates/gpu-solar2/src/step.rs:202-382`
(mixer, gqa, kda, front, shadow, back) and `crates/gpu-minimaxm3/src/step.rs:411-468`.

### 4.2 What moves, per family

The lift splits each family's program into a backend-neutral op order (moved to `crates/chain`) and a backend's ops
(the existing launch bodies, which stay). The trait grows per mixer **kind**, not per model, since AGENTS.md calls a
common owner with a method per model the wrong cut. The common owner holds per-kind steps only, and only for a kind two
or more families share: mamba_in/core/out (Nano, Lightning, Bolt), latent_in/core/out (Raon, GLM, V4.1) and
hc_pre/hc_post for mHC (GLM, V4.1). Dense FFN is the existing `FfnRoute::Dense`. A step one model owns (the four sparse
selectors, HcChart, HcGated, Engram, PLE, softcap) stays in that family's own walk: the common owner takes no per-model
hook, no plan enum for a one-model selector and no one-model selector (AGENTS.md "Common code first"). The decision
head is host f32 and is not a Metal op.

| family (models) | CUDA family code (`wc -l`) | its op order today | moves to `crates/chain` | CUDA lines that change | proof class |
|---|---|---|---|---|---|
| qwen3moe/qwen35moe/qwen35 (Q36, UD, kin, Q330, Clef, lev) | `gpu/src/arch/qwen3moe` 43,170 (with Body38) | chain walk (done) | nothing; 1b-1 and the seats track add plan rows (a `GqaKind` per shape row, packs) | 0 | add: plan rows |
| qwen38 (Q38) | in the same dir: `body38.rs` 6,343, `program38.rs` 1,836, `plan38.rs` 623, `scratch38.rs` 579 | `program38.rs` | the device-free part of `plan38.rs` and program38's op order (hc mix, mixer, MoE, hc combine) | the program's call sites become trait calls | move |
| nemotron_h (Nano) | 828 (`body.rs` 226, `step.rs` 133, `prefill.rs` 431, `mod.rs` 38) + common `hybrid.rs` 1,263, `mamba2.rs` 1,658 | `step.rs`, and the layer order in `hybrid.rs` | the three layer kinds' order (Mamba-2, NoPE GQA, ReLU² MLP) | ≤ 133 (step) + 131 (`hybrid.rs` `mamba_layer` `:569-632` and `gqa_layer` `:964-1030` [derived: 64 + 67]), a module six families import | move |
| nemotron_h_moe (Ltn, Bolt) | 4,286 (`body.rs` 633, `moe.rs` 866, `router.rs` 354, `placed.rs` 628, `prefill.rs` 1,768) | `body.rs`, `moe.rs` | + the MoE order (router, experts, shared, latent), shared with qwen3moe's route/experts | ≤ 1,500 (body + moe) | move |
| solar2 (Sol) | 2,637 (`body.rs` 1,374, `step.rs` 406) | `step.rs:202-382` | KDA and NoPE GQA order over `hybrid.rs` | ≤ 406 | move |
| raon1 (Raon) | 7,730 (`body.rs` 1,300, `mla.rs` 1,444, `rda.rs` 940, `mhc.rs` 725) | `body.rs` Mixer | RDA, MLA, HcChart order | ≤ 1,300 | move |
| mimo2 (MiMo) | 4,879 (`body.rs` 1,394, `seq.rs` 616, `program.rs` 167) | `program.rs`, `body.rs` | full/window GQA order, dense layer | ≤ 1,561 | move |
| glm5next (GLM) | 13,771 (`body.rs` 2,641, `program.rs` 423, `mla.rs` 365, `kda.rs` 186) | `program.rs` | KDA, MLA + TokenPool, mHC order | ≤ 3,064 | move |
| minimaxm3 (MMX) | 3,610 (`body.rs` 1,111, `step.rs` 487, `attn.rs` 931) | `step.rs:411-468` | BlockMax GQA order (its rule is `runtime::msa` already) | ≤ 487 | move |
| musespark (Muse) | 4,911 (`body.rs` 1,876 + `body/` 3,022) | `body.rs` | window GQA, latent MoE, alpha-beta order | ≤ 1,876 | move |
| deepseek41 (V41) | 39,340 (its device chain `src/chain/` 11,775; 8 files use `bloomery_chain`) | `src/chain/*` | the plan parts not yet in `crates/chain`; the latent, compress, mHC and Engram order | not read in full `[unverified]` | move |
| gemma-embedding2 (Gem) | 2,193 | `pass.rs` | the bidirectional pass order | ≤ 2,193 | move |

The proof class for every lift is **move**: `just canon-check BASE` identical for the scan set, plus the host dispatch
structural lines (graph node count, eager = replay, the e2e set identical), with no runtime gate beyond those. The
"CUDA lines that change" column is an upper bound: the op-order file's size. The lines that actually move are the
match-on-kind dispatch, not the launch bodies. A lift that edits `hybrid.rs` is the share of every family that imports
it, not of the lifted family alone: `mamba_layer` and `gqa_layer` are called by nemotron_h, nemotron_h_moe and solar2
(grep of the call sites), and the module's other items (`Acts`, `GqaCx`, `Site`, `project_wide`) by six families
(nemotron_h, nemotron_h_moe, solar2, minimaxm3, raon1, musespark: grep of `arch::hybrid`). The `canon-check` and the
structural lines of such a lift are the six families' share.

**The cheapest family to lift first is nemotron_h.**
- It is the smallest program: 133 lines of step, 226 of body.
- Its pieces are common already: `hybrid.rs gqa_layer` (shared with Solar2) and `mamba2.rs` (shared with
  nemotron_h_moe).
- Nano 4B (2.84 GB) runs on every Mac, and its real file fits the M4 Pro's 16 GB of free disk.
- Its fixture exists (`/models/fixtures/nemotron`, 895 MB).
- It opens nemotron_h_moe next by adding only the MoE ops the qwen3moe walk already has.

### 4.3 The synergy in lines

| port | shared host it uses (written once) | Metal-own lines | shared share of the model-specific lines |
|---|---|---|---|
| Qwen3.6, measured (crate sizes) | chain qwen3moe 1,897 | 17,224 (arch 5,783 + MSL 6,382 + launchers 5,059) | 1,897 / 19,121 = **9.9 %** (the first port wrote the kernels) |
| a neighbour in a lifted family (UD, kin, lev, Q330) [derived] | the whole plan and walk | 20-800 (§2.2) | about 100 % of the host; MSL only |
| a new family, typical [derived] | its plan and walk, moved: 1,000-2,000 (by the qwen3moe analogy, 1,897 for three archs) | glue about 2,000 + MSL median 1,268 (§2.2 steps 7-17) | 1,500 / (1,500 + 2,000 + 1,268) = 31.5 %, about **one third** |

Two more things are shared but not in the table:
- **The whole verification side**: fixtures, refsets, the dequantization ports, the coverage and shape tables, and the
  per-type ik dot dumps (§5).
- **The non-model host**: 161,187 lines.

The Metal glue shrinks if `crates/metal/src/arch/qwen3moe/session.rs` (2,165 lines: the host-fed, chain and ahead step
modes, fault poisoning, `reset`) and its `count.rs`, `rows.rs` and `target.rs` (932 lines) become a Metal-common session
over a per-arch trait before the second family lands. The step-mode logic names no model (`session.rs:1-30`); its
four tracer and profiler wrappers implement the qwen3moe `ChainOps` (`session.rs:1740`, `:1827`, `:2002`, `:2104`)
and go generic with the trait. Without the lift, each family would copy about 3,000 lines.

The lift is its own move round, after 1c and before the second family's session (the lead's decision of 2026-10-25),
proven by the encode trace (§3.4) and bit-identical `lab-metal` and `lab-metal-e2e`. It is not folded into the Nemotron
phase. It carries a per-mixer state contract. The chain and ahead modes drop a token committed ahead, which is safe
because the dropped token wrote only what its replacement writes again, including "the delta lane it writes, not the one
it reads" (`session.rs:9-13`; two lanes, `SNAPS` 2, `delta_geom.rs:15`). A mixer updated in place has no second lane:
CUDA's Mamba-2 reads and writes its conv ring and scan state in place (`crates/gpu/src/mamba2.rs:1-4`). Each mixer kind
therefore states how a dropped token is undone (two lanes, or another rule it names) before it joins a chain or ahead
step.

## 5. Proof without real files

### 5.1 Per op class: the clause pattern

The pattern is the qwen35moe kernel gate's, in `crates/metal/src/bin/gate_metal_qwen35moe_kernels/`:
- One `clauses_<class>.rs` per class: matvec, embed, head, experts, router, route, attention, delta, norm, rows, seg,
  gemm, tensor, a4.
- The **f64 oracle** `shared/oracle.rs` dequantizes the weight's blocks with `crates/gguf` and takes an f64 dot on the
  host. Its band is derived from the evaluation tree's rounding count (`oracle.rs:1-40`: for Q6_K, `n = 9 + ⌈nb/2⌉ + 31`
  with the SIMD sum's worst order).
- Inputs are **the fixture's real tensors** by name, e.g. `clauses_matvec.rs:25-30` reads `blk.0.attn_qkv.weight`,
  `blk.0.ssm_out.weight` and `blk.0.ffn_down_shexp.weight` of the fixture.
- A **mutant per clause** (`mutants.rs`) is red on its clause.

For each new class:
- **A quant owner** reuses `gguf::quant::dequant_row` (`crates/gguf/src/quant.rs:356`) as the oracle, with its band
  written like Q4_K's or Q6_K's in `oracle.rs`. That oracle and the f64 dot are the whole check. ik's per-type dot dumps
  on the box (`/root/bloomery-data/ref/`: `q5k-x4-ik-dot.txt`, `q8f0-`, `q5f0-`, `q5f1-ik-dot.txt`,
  `iq2xxs-mainline-dot.txt`, `iq3s-`, `iq4nl-`, `iq4xs-ik-dot.txt`, `mxfp4-x4-ik-dot.txt`, and the `q3_K`, `q5_0`, `q5_1`,
  `q6_K` dequant `.raw/.meta`, written by `just build-ref`) are no second check for Metal. They are the output of ik's CPU
  kernels over `q8_2_x4`-quantized activations (`tools/ref/q8f0_ref.cpp:1-9`), which the CPU `gate-qdot` reads, and a
  GPU kernel's activation path is not that. At most their weight bytes can serve as inputs. The inputs are the type's
  tensors in a fixture made from a file of that type (§5.3).
- **An op class** (Mamba-2, KDA, MLA, selectors, hyper-connections) gets a host f64 reference of its Metal rule, in
  the form of `shared/oracle_delta.rs`, `oracle_attention.rs`, `oracle_router.rs` and `oracle_experts.rs`. It is
  checked against the CPU engine's op where one exists: `crates/model/src/mamba2.rs` (807), `crates/model/src/arch/kda.rs`
  (210), and the per-family forwards under `crates/model/src/arch/*`, which run every family. That is the f64 referee
  `exact_ref --act f64|ours` (AGENTS.md "Performance first"). So Metal never writes a second model of an op; it writes
  one oracle in its own rule's sum order and ties it to the CPU engine once.

### 5.2 Per family: e2e on the fixture against ik's sets

| family | fixture on the box (`/models/fixtures`) | ik sets on the box (`/root/bloomery-data`) | what is missing |
|---|---|---|---|
| qwen35moe (Q36, kin, UD) | `qwen35moe` 2.6 GB (on the Mac too) | `fx_ref_qwen35moe`, `_d1k`, `_step4_every_node`, `unjoined*` (on the Mac too) | nothing for Q36. UD: the `unjoined` and `unjoinedq6` variants exist (`tools/fixture-variants.tsv:18-19`), with a gap at the Q8_0 embed and head (§5.3). MiniCPM-V: its header fixture, refused until 1-0 closes the COPY gap (`qwen35moe/fixture.rs:187`), and a set keyed to it (a box dump). Sage and Ornith: the Qwen3.6 fixture covers their trunk (same shapes and types); the nextn layer is outside it |
| qwen38 (Q38) | `qwen38` 11 GB | `fx_ref_qwen4exp`, `_d1k`, `_d3k`, `_step4`, `_every_node` | — |
| glm5next (GLM) | `glm5next` 8.0 GB, 3 shards | `fx_ref_glm5next` and 15+ (`step4_every_node`, `d1k`, `d3kdsa`, variants) | — |
| deepseek41 (V41) | `v41` 16 GB | `fx_ref_deepseek41_{d1,d1n,d2,step4}_every_node` | — |
| solar2 (Sol) | `solar2` 13 GB | `fx_ref_solar2_a`…`d` | — |
| musespark (Muse) | `musespark` 309 MB | `fx_ref_musespark_a`…`e` | — |
| minimax-m3 (MMX) | `minimaxm31` 6.6 GB (a Cut folder) | `fx_ref_minimaxm31_a`…`e` | the GGUF M3 file's own fixture, if M3 stays a README row |
| nemotron_h_moe (Bolt, Ltn) | `nemotron-bolt` 2.1 GB | `fx_ref-nemotron-moe-logits` | Lightning's fixture (other types, no latent) |
| nemotron_h (Nano) | `nemotron` 895 MB | none listed (real: `ref-nemotron-logits`, `-n7c`) | the fixture's sets |
| qwen3moe (Q330) | none (`tools/fixture-dirs.tsv`: `self`) | real only: `ref_qwen3moe`, `_d1k`, `_d4k`, `_step4` | fixture and sets |
| qwen35 dense (Clef, lev) | none (`self`) | real only: `ref_qwen35_lev_hidden_p64/p600/p4096`, `ref_qwen35_hidden_p*`, `ref_clefvis_*` | fixture and sets |
| mimo2 (MiMo) | none | real only: `ref_mimo2`, `_d1k`, `_d4096`, `_step4` | fixture and sets |
| raon1 (Raon) | none | real only: `ref_raon1_a`…`d` | fixture and sets |
| gemma-embedding2 (Gem) | none | real only: `ref_gemma_embedding(2)_*` | fixture and sets |

The e2e per family is `lab-metal-e2e`'s form (`justfile:110`): the fixture's ids through the Metal session, every
record against the family's `fx_ref_*` sets, refused by name through `crates/refset`'s family check. The every-node
sets (qwen35moe, qwen4exp, glm5next, deepseek41) localize a red to a node.

A **CUDA↔Metal differential** on the same fixture ids closes no gap the ik sets leave, because CUDA is held to the same
sets. It can only be banded, never bit-exact:
- MSL does not fix `simd_sum`'s order (`oracle.rs:20-22`).
- Apple GPUs have no f64, so the CUDA kernels' f64 sums become f32 with the same tree (`attention.metal:23-24`: "The CUDA
  sum is f64; this GPU has no f64").

Use it only to bisect a red by node in a family whose ik sets have no every-node dump: solar2, musespark, minimaxm31,
nemotron. The box already writes CUDA's per-node records.

### 5.3 What a fixture cannot prove, and the smallest real-file check

A fixture made from a README file's own header **does** carry that file's type mix, because every tensor keeps its
source's dims and type id (`fixture/mod.rs:6-13`). So the type mix is provable without the file: one fixture per
README quant. The cheaper form for a variant inside a family is `just fixture-requant FAMILY TAG` (`justfile:1068`;
ik's quantizer over the existing fixture into the tag's type map, as `glm5next iq4xs` is done today). For
Qwen3.6 UD the variants exist already: `unjoined` and `unjoinedq6` (`tools/fixture-variants.tsv:18-19`) carry the Q8_0
mixers and shared expert, F32 `ssm_alpha` and `ssm_beta`, and a Q5_K or Q6_K routed down, which gives 1c's mat-vec,
expert and F32 paths their e2e without a new box sitting. Their maps name no `token_embd` or `output.weight` role, so
the UD file's Q8_0 embed and head (its header: both Q8_0, 248,320 rows) and its one layer with Q5_K gate and up are
outside them [derived from the variants' maps and the header; whether their sets reach those tensors is unverified].

What it cannot prove:
1. **Memory fit**: byte arithmetic from the header.
2. **Speed**: bytes and shapes from the header; the runtime term is measured once per family.
3. **The real weights' statistics**: scale outliers, real activation ranges and sparse expert use. Fixture codes are
   random with standard deviation 1/√K (`fixture/mod.rs:20-25`).

The check for (1) and (2) needs no download and states them only as far as a header can: **`generate_metal --plan [-m FILE | --hf REPO[:QUANT]]`**.
It is the Metal twin of `generate_ds41 --plan`, which prints the prompt call's records and exits before the load. It
reads the header only: `gguf::Split::header_only` (`crates/gguf/src/lib.rs:771`) for a local file, and `fixture fetch`'s
sparse header ranges (`crates/model-hf/src/bin/fixture.rs:448-530`) for the Hub. It prints:
- the `Program`;
- every `Need` (`crates/models/src/need.rs:547`) with its Metal owner or "missing", by name;
- every weight site's type and its Metal launch family;
- each buffer against `maxBufferLength`;
- the total against the working set (the device's own value on a Mac, a table row elsewhere), stated as the room on an idle Mac;
- bytes a token and the predicted band (§3.2).

It exits 0 when the file runs and 3 when it does not, naming each missing item. Run over the README's files (a list
beside the README rows), it is the Mac-free answer to "does file X run on Metal". The memory line is the room on an idle
Mac. The session checks the live free memory when it opens: the Metal census reports `limit - currentAllocatedSize` as
free (`crates/metal/src/census.rs:26-45`), which counts only this process and so breaks the integrated-device contract
that free is the host's room (`crates/placement/src/placement/devices.rs:34-36`). `coverage::check_with` (`coverage.rs:1834`) accepts operation rows but still applies CUDA's
routed-format rules and weight-type pins (`weight_formats`, `coverage.rs:1942`; the Qwen35 head pin at `:1062` excludes `MLX_A4`).
Phase 1a must isolate Metal's weight validation from those policies, checking Metal weights against Metal-owned type rows while
preserving CUDA's checks. If that separation changes `coverage.rs` (95 gates reach it; `crates/gguf/src/qtype.rs` 230)
[derived: `python3 tools/recipes.py reach`], include its owning CUDA checks in Phase 1a. Completion requires Qwen3.6 Q4_K_M and
MLX_A4 to pass, and an unsupported Metal weight type to fail by name. The rule replaces today's ad hoc arena refusals (`arena.rs:334`, `:471`, `:486`, `:618`,
`:625`) by name.

Real-file check (3) is one run per family on a Mac that holds the file, against the real-file refsets already on the
box (`ref_qwen3moe*`, `ref_mimo2*`, `ref_raon1_*`, `ref_minimaxm3_*`, `ref-nemotron-logits`, `ref_qwen35_lev_hidden_*`).
It is a release-time item, not a per-landing gate. No real-file set is named for the MiniCPM-V, Sage or Ornith files
(§5.2), and a Clef set is refused for the README's Q5_K_M file, because a set names the file it was dumped from (Clef's are
the 27B Q4_K_M and Clef-Flash's Q8_0: `crates/refset/src/arch/qwen35/mod.rs:1-8`). A real Sage run on the second M1 Pro
has nothing to compare with: it shows that the file loads, runs and writes plausible text, not that it is bit-correct
[judgment]. It is background evidence for the first user claim, not a merge blocker.

### 5.4 Families without a fixture or sets: the box jobs

| job | where | price |
|---|---|---|
| a `FixtureSpec` row for qwen3moe, qwen35 (clef/lev), mimo2, raon1 and gemma-embedding2, plus the Lightning fixture and the MiniCPM-V header's fixture (refused by name until 1-0 adds `rope.section_order` to the copy list at `qwen35moe/fixture.rs:187`; `fixture/mod.rs:15`) | Mac code (the family's `fixture.rs`, a registry `Fixture::` row, `tools/fixture-dirs.tsv`) | Mac only for the fixture file; the fixture is a function of the seed and the header (`fixture/mod.rs:29`), so no weights move. A set keyed to a new fixture is a box dump (the rows below) |
| `fixture fetch` + generate | box (or any host with the header) | writes 0.3-13 GB to `/models/fixtures` (the box's 3.7 TB nvme0); wall [unverified]: no fixture-write time is recorded in `gate-times.tsv` |
| `just dump-ref-fixture FAMILY` (`justfile:1060`) | box, `BLOOMERY_TIER=fixture`, **under the CPU lease and a card** (`justfile:1057`, card `docs/cards/fxoracle-dump.card`) | a sitting: the lead's go; sets of 12-456 MB each by today's families; wall [unverified]: dumps are not batch items, so `gate-times.tsv` has none. The first dump's card names it as its one term |
| `just fixture-requant FAMILY TAG` for README-quant variants (UD's exist; a Q8_0 embed and head row would be new) | box, fixture tier | ik's quantizer over a ≤ 2.6 GB fixture; wall [unverified] |
| copy the sets to the Mac | rsync, as Qwen3.6's sets already are | 0.1-0.5 GB per family |

Each dump needs ik to run the arch. Real sets exist for all five families, so ik (or the tree that dumped them) does.

### 5.5 The port discipline: cuda-oxide → MSL

The constructs counted in `crates/gpu/src/*.rs`, `kquant/` and `gemm/` (`grep -oh … | uniq -c`, top uses):
`thread::index_1d` 184,
`thread::sync_threads` 165, `warp::lane_id` 147, `warp::reduce_sum_f32` 116, `warp::shuffle_xor_f32` 57,
`warp::shuffle_xor_f64` 40, `warp::ballot` 24, `SharedArray` 21, `DynamicSharedArray` 17, `wmma` 11, `sync::atomic` 10.

| cuda-oxide | MSL | rule |
|---|---|---|
| `#[kernel] fn` + `#[launch_contract(block = (N,1,1), requires = …)]` | `kernel void f(… [[buffer(i)]])` + the launcher's `dispatchThreadgroups(grid, (N,1,1))` | `requires` becomes the launcher's checks (`kernels/mod.rs` `check_window_len`), refused by name |
| `#[launch_bounds(N)]` | `[[max_total_threads_per_threadgroup(N)]]` | |
| `thread::blockIdx_x()` | `[[threadgroup_position_in_grid]]` | |
| `thread::threadIdx_x()` | `[[thread_position_in_threadgroup]]` | |
| `thread::index_1d()` | `[[thread_position_in_grid]]` | |
| `warp::lane_id()`, `tid / 32` | `[[thread_index_in_simdgroup]]`, `[[simdgroup_index_in_threadgroup]]` | SIMD width 32 on every Apple GPU (`geometry.rs`) |
| `warp::reduce_sum_f32` | `simd_sum` | **order not fixed by MSL**: the band counts 31 adds; where an order is the gate, write the `simd_shuffle_xor` butterfly out |
| `warp::shuffle_xor_f32` / `shuffle_f32` / `shuffle_up` | `simd_shuffle_xor` / `simd_shuffle` / `simd_shuffle_up` | |
| `warp::ballot`, `warp::any` | `simd_ballot`, `simd_any` | `lanemask_lt` has no builtin: `(1u << lane) - 1` against the ballot |
| `warp::shuffle_xor_f64`, f64 sums (9 files: `latent.rs`, `ple.rs`, `qk_rms.rs`, `qsa.rs`, `rope_neox.rs`, `linear/norm_gate.rs`, `linear/conv.rs`, `engram_gate.rs`, `arch/qwen3moe/router.rs`) | **no f64**: the same pairing, butterfly and warp order in f32 (`attention.metal:20-26` is the precedent) | the f64 oracle's band covers it; a CUDA↔Metal compare is banded |
| `thread::sync_threads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` | |
| `threadfence()` (`mamba2.rs:443`, `route_body.rs:271`), and a barrier that must publish device memory | across threadgroups `atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)` with the ticket; within one, `threadgroup_barrier(mem_flags::mem_device)` | precedents `ticket.metal:38-44`, `head.metal:57,73`, `attention.metal:900` |
| `warp::sync_mask(u32::MAX)` (`flash_gqa.rs:601`, `hc_gated.rs:1827`) | `simdgroup_barrier(mem_flags::mem_none)` `[unverified: confirm per use]` | |
| `warp::match_any_sync` (`gemm/kernels.rs:152,308`, `gemm/kernels32.rs:953,1110`) | no MSL builtin: build it from `simd_ballot` or `simd_shuffle` compares | |
| `static mut X: SharedArray<T, N>` | `threadgroup T x[N];` in the kernel | Metal's threadgroup memory is 32,768 B (`docs/contrib/mac.md:44`), so recheck every CUDA tile |
| `DynamicSharedArray` | `threadgroup T* [[threadgroup(i)]]` + `setThreadgroupMemoryLength` | |
| `shared::cvta_generic_to_shared_u32` | none needed (address spaces are explicit) | |
| `sync::atomic`, `cuda_device::atomic` | `atomic_uint` + `atomic_fetch_*_explicit(…, memory_order_relaxed)` | |
| `cuda_device::wmma` | `simdgroup_matrix` + `simdgroup_multiply_accumulate`; tensor ops (`TENSOR_SOURCE`, language 4.0, `library.rs:19`) | prefill only; decode is mat-vec |
| `thread::__unroll_config` | `_Pragma("clang loop unroll(full)")` (`quant.metal:46` `FOR_UNROLL`) | |
| const generics (`HEAD`, `GROUP`, `PACK`) | the instance header (§3.3) or a template under `[[host_name]]` | one instance per `models::shape` row |
| `FaultSink::raise(site)` (`crates/gpu/src/fault.rs:85`: `(layer << 8) \| site` by atomic min, the site bit ORed into the layer's word) | `raise(faults, layer, site)` (`crates/metal/kernels/fault.metal:9-19`: the same min and OR) | one site table, `bloomery_device::fault::FaultSite` (`crates/device/src/fault.rs:73`), written into MSL by `build.rs:66`; **the same refusal must raise the same site** (§7 item 1) |
| `half_to_f32` | `(float)h` | |
| `DisjointSlice<f32>` | `device float*` | disjointness is the launcher's contract |

**The worked example: the RMS norm row.** Both texts exist today: CUDA `rms_norm` (`crates/gpu/src/elem.rs:1015-1060`)
and Metal `norm_rows` (`crates/metal/kernels/norm.metal:124-175`). Abridged:

```rust
// CUDA (cuda-oxide), one block of 256 threads per row
static mut WSUM: SharedArray<f32, RMS_WARPS> = SharedArray::UNINIT;
let t = thread::blockIdx_x() as usize;                    // the row
let tid = thread::threadIdx_x() as usize;
let part = warp::reduce_sum_f32(unsafe { rms_partial_sq(x, base, k, tid) });
if warp::lane_id() == 0 { unsafe { *ws.add(tid / 32) = part; } }
thread::sync_threads();
let sums = unsafe { [*ws.add(0), /* … */ *ws.add(7)] };
// ((w0+w1)+(w2+w3)) + ((w4+w5)+(w6+w7)): "This order is the gate" (elem.rs:490-497)
let scale = rms_scale_checked(rms_warp_tree(sums), k as u32, eps, fault, tid == 0);   // raises F32Product
apply.map(move |g, v| (scale * g) * v);
```

```metal
// Metal (MSL), one threadgroup per row
kernel void norm_rows(device const float *x [[buffer(0)]], /* gain, y, args */
                      device atomic_uint *faults [[buffer(4)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tg_sums[NORM_ROWS_SG];
    float sum_sq = 0.f;                                   // float4 loads: cols % 4 == 0, host-checked
    for (uint f4 = tid; f4 < f4s; f4 += NORM_ROWS_THREADS) { /* sum_sq += x·x */ }
    sum_sq = simd_sum(sum_sq);
    if (lane == 0u) tg_sums[sg] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.f;
    for (uint g = 0u; g < NORM_ROWS_SG; ++g) total += tg_sums[g];   // serial, not CUDA's tree
    const float scale = rms_scale(total, args.cols, args.eps);
    if (!rms_usable(total, scale)) { /* raise NORM_QUANT, write NaN */ }
    /* y = (x * gain) * scale, each output checked finite */
}
```

The mapping is mechanical line for line: block → threadgroup, warp sum → `simd_sum`, `SharedArray` → `threadgroup`,
`sync_threads` → barrier, `FaultSink` → `raise`. The example also shows six places where it is not mechanical, so the
discipline must state them:
1. **The cross-group sum order**: CUDA's fixed tree against Metal's serial loop. That is allowed under "performance
   first", but a port that keeps CUDA's tree is the default, so that a fused norm-on-load and the op path stay equal
   within one backend.
2. **The load width**: scalar strided against `float4`.
3. **The refusal's site**: `F32Product` against `NormQuant`.
4. **The output product order**: CUDA multiplies `(scale·g)·v` (`elem.rs:1068`), Metal `(x·gain)·scale`
   (`norm.metal:115`). The two round differently.
5. **Contraction**: the CUDA device build contracts a multiply and an add into one fused multiply-add (`elem.rs:546-551`),
   while the Metal translation unit sets `#pragma METAL fp contract(off)` in its first file (`fault.metal:11`;
   `build.rs:175-176`). A line that is one fma on CUDA is two roundings on Metal.
6. **Where a non-finite value raises**: Metal checks every output value and raises once per row (`norm.metal:163-172`);
   CUDA raises on the scale (`elem.rs:1067`, `rms_scale_checked`).

**A shared kernel description emitting both: evaluated, not recommended.**
- Cost: a DSL with two code generators, one to cuda-oxide Rust and one to MSL. That is thousands of lines that would
  need its own gates, plus a third language for every kernel author.
- What it can express is only the intersection of the two machines. The kernels that set speed differ in structure,
  not in syntax:
  - Metal's decode mat-vec lane maps follow llama.cpp's `kernel_mul_mv_q4_K_f32`/`q6_K` (`quant.metal:26-33`), CUDA's
    are Walk A (`crates/gpu/src/kquant/mod.rs:3-8`).
  - Threadgroup memory is 32 KB on Metal against larger CUDA tiles.
  - Prefill uses tensor cores on one side and `simdgroup_matrix` or tensor ops on the other.
- Benefit: one text per op, but each backend still runs its own gates, so the proof work does not shrink.

**Recommended instead: share every fact a kernel reads, generated into MSL from Rust**, the way `build.rs` already
writes the fault sites from `bloomery_device`:
- the shape instances (`models::shape`);
- the block layouts and IQ grids (`gguf/src/qtype.rs` `QROWS`, `iq_tables.rs`);
- the fault sites;
- the router rules.

Write each op's numeric contract (sum order, rounding points) once, in the f64 oracle's doc beside its band, and port
by the table above. Cost: about 300 lines of generation in `build.rs` [derived: today's fault and geometry writers are
the bulk of its 318 lines]. Benefit: no drift in tables, sites or shapes, which are the facts a hand port gets wrong
silently.

## 6. Phases

Machines:
- **Lead M1 Pro 32 GB** (34 GB disk free): development and `lab-metal`.
- **M4 Pro 48 GB testbed** (16 GB disk free, so no file over about 15 GB): timing, as in `mac-numbers.md` §4. A sitting
  with rotated arms and witness lines, and only on the lead's go.
- **The second M1 Pro 32 GB** (490 GB disk): the real files over 15 GB.
- **The box**: fixtures and refset dumps only.

Speed predictions use §3.2's band on the M4 Pro [derived]. Its two measured inputs (85.6 and 101.2 tok/s) are the M4
Pro's, and no Qwen3.6 rate is recorded on the second M1 Pro. A row that will be measured there is therefore given as a
ratio to Qwen3.6 on the same Mac [derived], and runs interleaved with Qwen3.6 in one sitting. Models that do not fit 48
GB are given at the contributors' machines (#593, `mac-numbers.md` §2c: M3 Ultra 150.5, M5 Ultra 219.7 tok/s on Qwen3.6
near depth 0), scaled by 1.976 GB over the model's bytes a token [derived; which file #593 ran is unverified].

**Landing rule.** A Metal round touches `crates/metal` and fixture files, so the box's `just affected` may select little.
The landing evidence is `just lab-metal` and `just lab-metal-e2e` on a named Mac at the commit hash, recorded in the PR
body.

**Contributor issues** open only after Phase 1 (1-0, 1a, 1c) is merged, the fixture proof path exists, one template-model
PR is merged (1b-1 is the natural template), and that model's fixture and refsets are already dumped (a contributor cannot
dump on the box). Two may open earlier: NEON qdot, and the M5 measurement on #593.

| phase | scope | files | prediction | proof | gates | machine |
|---|---|---|---|---|---|---|
| **1-0** kin, host only | lift the vocab refusal (`arena.rs:602`) and the fixed `HEAD_GROUPS` (`:678`); close the fixture COPY gap (`rope.section_order`, `crates/model/src/arch/qwen35moe/fixture.rs:187`) so a MiniCPM-V header fixture can be written; MiniCPM-V opens as a text body (mmproj and vision are an off cell, declared in 1a's coverage row) | `crates/metal/src/arch/qwen3moe/arena.rs`, `crates/model/src/arch/qwen35moe/fixture.rs`, the head clauses (they read the fixture's vocabulary in place of `HEAD_VOCAB`) | MiniCPM-V, Sage, Ornith = Qwen3.6's **85.6 tok/s** (M4 Pro, Q4_K_M, depth about 0) within 0.1 %: bytes a token 1.9754 / 1.9757 / 1.9757 GB against 1.9757 GB, head grid 7,755 against 7,760 groups [derived]; about 20 host lines [judgment] | a host-only diff (`git diff --stat` lists no `.metal`, `*_geom.rs` or `build.rs` line; Qwen3.6's vocabulary still makes 7,760 groups); the head tail clause at 248,144 rows on a MiniCPM-V header fixture (`clauses_head.rs:244`, `clauses_a4.rs:536`) | `lab-metal`, `lab-metal-e2e` (the qwen35moe fixture, bit-identical to base) | lead Mac; a Sage real run on the second M1 Pro is background evidence for the first user claim, not a merge blocker |
| **1a** instruments | `generate_metal --plan` (header-only; memory as the room on an idle Mac, plus a live free-memory check at open); Metal's coverage and type rows in `crates/metal`, checked against Metal-owned type rows, isolated from CUDA's routed-format rules and weight-type pins (`coverage.rs:1942`, `:1062`); the move proof's tools (§3.4): a per-pipeline hash, its determinism probe, the encode trace | `crates/metal/src/bin/generate_metal.rs`, a coverage module in `crates/metal`, a trace layer under the tracer wrapper (`session.rs:1740`), a script in `tools/` | `--plan` on Qwen3.6 Q4_K_M and MLX exits 0; on MiniCPM-V, Sage and Ornith it exits 0 once 1-0 is in (the vision off cell is printed); on UD (until 1c), lev, Clef and Qwen3-30B it exits 3 naming exactly the §1 rows. One build run twice, and again on another day, gives one hash per pipeline: the one term to measure `[unverified]`; Qwen3.6 Q4_K_M and MLX_A4 pass the Metal type rows, an unsupported Metal weight type fails by name | the plan's lists equal §1.2/1.3 for every README file; Qwen3.6's pipeline hashes and encode trace recorded as the base | `just mac-test` for the pure parts. no CUDA gate while `coverage.rs` and `qtype.rs` are not edited (`arena.rs` has no `gate-*` recipe); if the isolation changes `coverage.rs`, its owning CUDA checks join | lead Mac; the plan needs no GPU; the hash and the trace need one. Neither the lead Mac nor the M4 Pro has the Metal toolchain (2026-10-25), so they run the run-time source path and the hash is of each pipeline serialized to a binary archive, not of AIR |
| **1c** Qwen3.6 UD-Q4_K_XL (add class) | Q8_0 and Q5_K owners with the 16-value GEMM dequant; the prompt GEMM entries (`kernels/gemm.rs:503-505`, `gemm_routed.metal:340-351` and `gemm_tensor.metal:506-517` take Q4_K, Q6_K and A4 only); the F32 P1 segment (`ssm_alpha`, `ssm_beta`: `arena.rs:357-379` takes quant types only); `SharedPlan::Apart` as one common dispatch arm with no per-model branch (Qwen3.8 uses it too); the new expert lanes written to the offload doc's slot-table signature | `kernels/quant.metal`, `matvec_q8_0.metal`, `matvec_q5k.metal`, the gemm entries, `src/kernels/mod.rs` `Quant`, `arena.rs:334/618/625/993/1074/1109/1117` | UD decode ~~61-68~~ **60.8-67.4 tok/s** on the M4 Pro (without Apart 61.4-68.2; roofline 95), 0.71-0.79 × Qwen3.6's 85.6 [derived: 2.753 GB a token, 262 GB/s, c_L 0.104 ms × 40 for the upper edge, 169 GB/s for the lower; Apart adds 80 bandwidth-bound dispatches × 2.08 µs = 0.166 ms]. Prompt: UD / Qwen3.6 pp = 0.95-1.00 at P = 512 and 4096 on one Mac in one sitting [derived: a compute-bound prompt (`docs/contrib/mac.md:174`) gives 1.0; if weight-read bound, 21.167 / 22.349 GB = 0.947], so pp@4096 on the M4 Pro is 989-1,045 tok/s from the measured 1,044.5 (`README.md:103-104`); pp@512 in absolute terms is `[unverified]` (no Metal row for Qwen3.6). The term left to measure is the Q8_0 and Q5_K tile decode in the prompt GEMM | the clause pattern with `dequant_q8_0`/`dequant_q5_k` as the f64 oracle (§5.1); e2e on the fixture variants `unjoined` and `unjoinedq6` (`tools/fixture-variants.tsv:18-19`), no box sitting; the Q8_0 embed and head are outside the variants' maps (§5.3) | `lab-metal`, `lab-metal-e2e` and the variants' e2e | lead Mac; the UD real file on the second M1 Pro (22.35 GB fits its 26.8 GB working set), interleaved with Qwen3.6 |
| **1b-0** instance mechanism (a move) | the MSL header becomes a function of the `models::shape` row, generated at build time for the finite registered rows; Qwen3.6 is the single instance; K stays compile-time per instance; the router's K joins the instance key; ~~no load-time path for an unregistered shape (the named refusal is the contract and the contributor's route)~~ (superseded 2026-10-27: §3.3); the registered instance's source-compile fallback stays | `crates/metal/build.rs`, `src/kernels/{geometry,delta_geom,router_geom,experts_geom,attention_geom}.rs`, `src/library.rs` | Qwen3.6 unchanged: **85.6 / 101.2 tok/s** stand by the move proof (§3.4); the next round that touches the dispatch path re-measures. Price: `time xcrun metal -c` once × the instance count (build time) | §3.4: per-entry AIR hashes equal, encode trace equal at depth about 0 and one deep depth | `lab-metal`, `lab-metal-e2e` | lead Mac; the M4 only if a proof is not equal |
| **1b-1** Qwen3-30B-A3B (add class, about 0.6 kLOC) | GQA head 128 (an attention rewrite: the norm-rope pairing and rows, `attention.metal:20-26`) and the ungated softmax-norm router at 128; a new qwen3moe fixture writer and its `fx_ref` dump | `kernels/attention.metal`, `router.metal`, `src/kernels/attention*.rs`, chain `GqaKind::Neox128` plan rows (exist) | Qwen3-30B **81-88 tok/s** (roofline 137). Qwen3.6's attention AIR likely moves, so an M4 A/B is budgeted from the start (below) | clauses vs `oracle_attention.rs` at head 128; e2e on the new fixture vs the new `fx_ref_qwen3moe` | `lab-metal` + qwen3moe e2e | Mac + box (fixture and dump, one sitting); M4 for the A/B |
| **Seats track** (weight 0 until the user ranks it) | lev (body only: hidden 2560, GQA group 4, no `output.weight`, which `arena.rs:620-627` requires), Clef (the body; its decision head is host f32), EmbeddingGemma (the bidirectional pass); a Metal seat engine and a headless open: `bloomery_serve_metal.rs:186` builds only a `TargetEngine`, and the decide seat lives in `crates/gpu-gates/src/bin/shared/serve_seats/decide.rs` | `crates/metal/src/arch/` (a seat module), MSL for the pass and the packs | ~~lev body 64-74, Clef body 28-37 tok/s~~ no decode-rate claim: the metric is prompt throughput and request latency, `[unverified]` until the seat engine exists. Roofline only: the lev body 63.4-74.1 tok/s [derived, §3.2] | a qwen35 fixture and sets (none yet); a gemma fixture and sets | e2e per seat | Mac + box |
| **Session lift** (a move) | `session.rs` and its `count.rs`, `rows.rs`, `target.rs` become a Metal-common session over a per-arch trait, with a per-mixer two-lane state contract (§4.3); after 1c, before the second family's session | `crates/metal/src/arch/qwen3moe/{session,count,rows,target}.rs` | none: a move | encode trace equal (§3.4); `lab-metal` and `lab-metal-e2e` bit-identical | `lab-metal`, `lab-metal-e2e` | lead Mac |

| **4** Nemotron (first chain lift) | lift nemotron_h, then nemotron_h_moe, into `crates/chain` (per-kind steps only, §4.2); Mamba-2; Q5_0, Q5_1, IQ2_XXS, NVFP4; the biased-sigmoid router; ReLU²; latent MoE | `crates/chain/src/arch/{hybrid,nemotron}`, `crates/gpu/src/arch/nemotron_h*/`, MSL | Nano **66-71**, Lightning **54-58**, Bolt **36-37** tok/s | lift: `just canon-check BASE` identical + structural lines; kernels: clauses; e2e on `nemotron`, `nemotron-bolt` and a Lightning fixture | `canon-check`, the nemotron gates on the box, `lab-metal` + e2e | Mac (Nano real file fits the M4); box for the lift's gates and the dumps |
| **5** KDA + MLA | KDA (sigmoid, softplus, write gate); MLA core; Raon's glue; HcChart; the out-gate projection | lifts of raon1 and solar2; MSL | Raon **57-71 tok/s**; Solar2 about 59 tok/s on an M5 Ultra | lift: canon-check; e2e on `solar2` and a new raon1 fixture | as phase 4 | Mac + box |
| **6** MiMo, Muse | MXFP4; K192/V128 window, sinks, value scale; head-64 window; alpha-beta, softcap, expert norms | lifts of mimo2 and musespark; MSL | MiMo about 41 tok/s (M5 Ultra); Muse about 12 (M3 Ultra) | e2e on `musespark` and a new mimo2 fixture | as phase 4 | Mac + box; real files only on a contributor's Mac |
| **7** the 256/512 GB set | GLM (TokenPool, mHC), Qwen3.8 (QSA, HcGated, PLE), MiniMax (BlockMax, IQ3_S, IQ4_XS), V4.1 (compressor, hash router, Engram, Q3_K) | their lifts; MSL | GLM about 21 (M3 Ultra), Qwen3.8 about 69 (M5 Ultra), MiniMax about 16, V4.1 about 38 (M3 Ultra) tok/s | e2e on `glm5next`, `qwen38`, `minimaxm31`, `v41` (sets exist) | as phase 4 | Mac + box; with the sibling's offload, smaller Macs |

Phases 2 and 3 of the first draft are now 1b-1 and the seats track.

**Open before 1c: the UD name.** `crates/model/src/arch/qwen35moe/fixture/qwen35moe.rs:27` calls the unjoined layout
"unsloth's UD-Q4_K_M", while the README row and the header read in §1.1 say UD-Q4_K_XL (both carry `general.file_type`
15). Which file's type map 1c's owner set follows has to be settled first. This doc follows the header read of the README's
file: Q8_0 mixers, shared expert, embed and head; Q4_K gate and up (Q5_K on 1 layer); Q5_K down on 36 layers and Q6_K down on 4.

**1b-1's A/B.** An M4 same-sitting A/B on Qwen3.6 Q4_K_M, base against new build, with an A/A arm: 6 rounds per arm, judged
at the M4's own spread (0.42 % over the 3 rounds of the 2026-10-25 cross-engine sitting, `mac-m4pro-xeng`), which makes the interval ±0.54 % [derived:
0.42 × √(2/6) × t(0.975, 10 df) = 0.42 × 0.577 × 2.228]. The predicted effect is 0 (the head-256 path is not meant to
change). Inside the interval: land. Outside: bisect with the per-entry AIR hashes and the encode trace to the entry that
moved.

**The first round's prediction (1-0), as the round's card states it** [derived]: MiniCPM-V, Sage and Ornith decode at
Qwen3.6's rate, 85.6 tok/s on the M4 Pro at depth about 0 (Q4_K_M; bytes a token 1.9754 GB for MiniCPM-V and 1.9757 GB for
Sage, Ornith and Qwen3.6, a difference of −0.015 %; head grid 7,755 groups against 7,760). 1-0 does not re-measure it: the
diff is host-only and Qwen3.6's grid stays 7,760. Qwen3.6 itself: 85.6 and 101.2 tok/s stand by the move proof (no `.metal`,
geometry or `build.rs` line changes); the next round that touches the dispatch path re-measures. Three model files open
(18 weight) for about 20 host lines, and no term is left to measure.

**Phase 1 as a whole** (1-0, 1a, 1c) opens ~~five~~ four more model files: MiniCPM-V, Sage and Ornith at 85.6 tok/s, and UD at
~~61-68~~ 60.8-67.4 tok/s on the M4 Pro (~~lev at 64-74~~ lev is in the seats track), for ~~about 1,200~~ about 820 MSL and
launcher lines; 1a is tools, not kernels. Qwen3.6's 85.6 and 101.2 tok/s ~~stand without a sitting~~ stand by the move
proof. The term 1c leaves to measure is the Q8_0 and Q5_K tile decode in the prompt GEMM. 1b-0's only unknown is `time
xcrun metal -c`; the compile time of a second instance's library, the one term the first draft left to measure, is that
build-time cost.

**Chain lift** (Phase 4 on): the common owner holds per-kind steps only; a family with model-only steps (Engram, PLE, the
decision head, softcap) keeps its own walk, with no per-model hook and no one-model selector in the common owner (§4.2).

## 7. Places outside the assignment where improvement looked possible

1. `crates/metal/kernels/norm.metal:121` raises `NormQuant` for a refused RMS scale, while CUDA's same refusal raises
   `F32Product` (`crates/gpu/src/elem.rs:543`). A fault compare across backends disagrees on the site. About 2 lines
   plus a clause.
2. `crates/metal/src/kernels/head.rs:20,39` fixes the vocabulary at 248,320 with an exact-multiple assert, and
   `arena.rs:602` and `:678` refuse and size by it, which refuses MiniCPM-V's 248,144 rows. The kernel takes runtime rows
   already (`head.metal:52-54,65,83`), so the fix is host-only, about 20 lines: Phase 1-0.
3. `crates/metal/src/arch/qwen3moe/session.rs:1-30`: 2,165 lines of model-agnostic step modes (with `count.rs`,
   `rows.rs`, `target.rs`, 932 more) live in the qwen3moe arch module. Lift them to a Metal-common session before the
   second family, about 3,000 lines moved. Its own move round after 1c (§4.3).
4. `crates/model/src/arch/coverage.rs:47`: no `AVAILABLE` row names `Program::MiniMaxM3Body` (`grep -c` 0; the program
   appears only in the routed-type rule `:1881`). Either its Needs are refused by `check_with` (`:1828-1855`, which
   needs `a.programs.contains(&p)`) or another path covers them `[unverified]`. Reading this takes about 30 minutes;
   adding the rows is about 40 lines.
5. The header-reading script used for this survey (`ggufhdr.py:44`, kept outside the repo) crashes on an unknown type id (NVFP4 = 40,
   `{ty:8s}` on an int) and reads only the first shard, so split files show `tensors 0`. 1 line plus about 30 lines for
   a shard loop. A promoted `tools/` header reader would serve `--plan`'s Hub path too.
6. `crates/gpu-gates/src/record.rs` is the record-kind owner, but `crates/metal` does not depend on `gpu-gates`, so a
   Metal `--plan` print has no record owner to use. Whether the record module can move to a device-free crate is
   `[unverified]`, about 1 day.
7. `crates/metal/src/kernels/matvec.rs:91` says "when Metal implements `ChainOps`", but it does
   (`crates/metal/src/arch/qwen3moe/ops.rs:399`). A stale comment, 2 lines.
8. `crates/metal/kernels/attention.metal:20-26`: the norm-rope pairing is written for head 256 only. Head 128 (1b-1) and
   head 64 (Muse) need it head-generic. This is in scope for 1b-1, which likely moves Qwen3.6's attention AIR (the A/B
   budget in §6); it is listed so that 1b-0, a move, does not template it early.
9. `crates/metal/src/arch/qwen3moe/arena.rs:597` ties the hidden width to the head's K and the router's K in one refusal.
   Split it per kernel: about 10 lines.
10. `crates/model/src/arch/qwen35moe/fixture.rs:187`: the copy list lacks `rope.section_order`, which refuses a MiniCPM-V
    header fixture. One line and a test; scheduled in 1-0.
11. `crates/model/src/arch/qwen35moe/fixture/qwen35moe.rs:27` calls the unjoined layout UD-Q4_K_M while the README row is
    UD-Q4_K_XL (both carry `general.file_type` 15). One line; which name is right is `[unverified]`; settle before 1c.
12. `crates/metal/src/census.rs:26-45`: `free_bytes` ignores other processes. About 20 lines using the host's memory
    statistics; it precedes the tier round in the offload doc.

## Re-review (2026-10-25)

One line per finding of the adversarial re-review (S1: the phase does not hold as cut; S2: a claim is wrong or unproven; S3:
a citation). Each was checked against the code it cites before it was applied.

- S1-1 applied §0, §1.3 (shape sources), §2.2, §3.1, §6 1-0, §7 item 2: the head kernel takes runtime rows with a tail
  guard (`head.metal:52-54,65,83`, `head.rs:46-50`); only `arena.rs:602` and `:678` stop MiniCPM-V, and Sage and Ornith
  pass `arena.rs:570-630`. Kin is host-only, about 20 lines.
- S1-2 applied §1.2 (F32 row), §1.3 (Apart row and sources), §2.2, §5.3, §6 1c: UD is unjoined (`arena.rs:1074`, `:1109`,
  `:1117`), its `ssm_alpha`/`ssm_beta` are F32 (header), `ssm_out` is Q8_0 (`:993`) and the prompt GEMMs take Q4_K, Q6_K and
  A4 only. 1c is about 0.8 kLOC with pp@512/4096 predictions.
- S1-3 applied §3.4, §3.2, §3.3, §0, §6: `msl-scan` is unsound (the main library is one translation unit, `build.rs:175-176`;
  host dispatch is outside the text). The move proof is a per-entry AIR hash plus an encode trace, and "stand by the move proof" replaces
  "stand with no sitting".
- S1-4 applied §2.2 (seats rows), §6 (seats track): lev reads one prompt pass and a host label readout
  (`crates/decision/src/label.rs:1-5`), `serve-metal` has only a `TargetEngine` (`bloomery_serve_metal.rs:186`), and lev
  has no `output.weight`. The seats track runs at weight 0 on pp and request latency.
- S1-5 applied §5.3, §6 1a: Metal's rows live in `crates/metal` and are judged by `coverage::check_with`
  (`coverage.rs:1834`); `coverage.rs` is reached by 95 gates and `qtype.rs` by 230 (`tools/recipes.py reach`).
- S1-6 applied §2.2, §6 1c: model breadth first, the offload identity round after; one owner at a time for `experts.metal`,
  `arena.rs` and `session.rs`; 1c writes its lanes to the slot-table signature.
- S2-7 applied §3.1, §3.2: a runtime K puts `ib / per_slot` and `ib % per_slot` on the address path (`quant.metal:276-280`,
  `experts.metal:173-175`); c_L is 35 % of the token, so the bandwidth shadow does not cover it. K is compile-time per
  instance.
- S2-8 applied §1.3 (mixers source), §3.1: Qwen3.8 has 48 value heads and inner 6,144 (its header), a second delta
  instance.
- S2-9 applied §1.3: Clef's decision head (host f32, `crates/decision/src/head.rs:1`) and the LayerNorm row are removed;
  the activation row is five rows (columns read from the family specs); the dense row drops Nano; an Apart row is added.
- S2-10 applied §0, §1.3 "Counts": 77 needs (28 types, 49 op rows; 41 shared, 36 one-model), 15 on Metal, re-run by a
  script over the tables as they now stand.
- S2-11 applied §0, §5.3, §6 1a: memory is stated as the room on an idle Mac, with a live free-memory check at open
  (`census.rs:26-45`).
- S2-12 applied §6 (basis paragraph and the 1c row): UD is a same-Mac ratio, 0.71-0.79 × Qwen3.6, interleaved in one
  sitting; kin needs no run. The sitting itself is 1c's.
- S2-13 applied §2.2 (method bullet, row 10): the CUDA `delta.rs` and `conv.rs` hold KDA entries, so 0.62 is a floor
  ratio. A GDN-only ratio needs the files split by entry and was not computed.
- S2-14 applied §4.2: per-kind steps only; one-model steps stay in the family's walk. Corrected against the code, in
  the review's numbers: `mamba_layer` ends at `hybrid.rs:632` and `gqa_layer` at `:1030` (the review's `:635` and `:1037` run into the
  next items), and `hybrid.rs` is imported by six families but those two functions are called by three.
- S2-15 applied §4.3, §6 (session-lift row): its own move round after 1c, with a per-mixer two-lane state contract
  (`session.rs:9-13`, `SNAPS` 2; CUDA's Mamba-2 is in place: `mamba2.rs:1-4`).
- S2-16 applied §5.5: product order, fp contraction and per-output raise are added as non-mechanical places; the table
  gains `threadfence`, `sync_mask`, `match_any_sync` and the device barrier.
- S2-17 applied §5.1: the ik dot dumps are CPU kernels over `q8_2_x4` activations (`tools/ref/q8f0_ref.cpp:1-9`); the
  check is `dequant_row` and the f64 oracle, and `gate-qdot` is the CPU gate.
- S2-18 applied §0, §5.2, §5.3, §5.4 for MiniCPM-V, Sage and Ornith: no real-file set exists for their files, and the
  MiniCPM-V header fixture is refused until 1-0 adds `rope.section_order` to `qwen35moe/fixture.rs:187`. Not applied for
  Clef: `crates/refset/src/arch/qwen35/mod.rs:1-8` shows real hidden sets for the 27B Q4_K_M and Clef-Flash's Q8_0 files.
  They are not sets of the README's Q5_K_M file, and a set is refused for another file, so the text says that.
- S2-19 applied §3.3: instances are generated and compiled at build time for the finite registered rows; ~~no load-time path
  for an unregistered shape~~ (superseded 2026-10-27: an unregistered geometry compiles from source at open, §3.3); the
  price is `time xcrun metal -c` × instances.
- S2-20 applied §3.1, §3.2, §3.3: the router's K (the hidden width) joins the instance key; a `ROUTERS` row holds no
  hidden (`shape.rs:88-130`).
- S2-21 applied §2.2, §6: lev, Clef and EmbeddingGemma are a seats track at weight 0 until the user ranks it; its cost
  includes the seat engine.
- S3 applied: `arena.rs:289` is a doc comment and no 85 MB measurement is at the cited lines, so §1.1 now gives 84.2 MB as
  derived (§1.1); the new-family MSL median is 1,268 and the share 31.5 % (§0, §4.3); `gate-qdot` is the CPU gate (§5.1);
  the qwen3moe module is 5,779 lines and the `arch` directory 5,783 (§0); Metal today is 15, not 16 (§0, §1.3).

Found while applying (not in the review):
- `crates/metal/src/bin` is 30,154 lines by `wc -l`, not the 30,760 of the first draft (§0).
- K compile-time per instance and the fixture rule pull apart for a family whose fixture narrows expert ff (§3.1).
- The `unjoined` fixture variants name no `token_embd` or `output.weight` role, so the UD file's Q8_0 embed and head are
  outside them (§5.3, §6 1c).
- The first draft's text said 49 op classes over a 46-row table; the corrected tables and the text now agree (§1.3).
- The runtime holds two programs, not one: the main library and the tensor-op GEMM library (`build.rs:293-309`,
  `library.rs:17-23`); the move proof covers both (§3.4).
- Local scratch paths and a machine host name were removed from the text.

Not applied: the Clef part of S2-18 (above). Deferred: none (the measurements the findings name are the rounds' own terms in §6).

The reviews are the 2026-10-25 adversarial re-review of the first draft of this file; the lead's decisions file of the same
day set the phase order and wording where the two differ.
