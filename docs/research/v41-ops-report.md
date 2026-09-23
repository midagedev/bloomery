# DeepSeek-V4.1-Flash decode graph as tensor math — from the official reference, cross-checked against two llama.cpp-family ports

Investigation only. Nothing in any source tree was modified. Every claim cites `file:line`; derived
numbers are marked **derived**; facts about our quantized file are marked **inferred from name**
(the file itself is not present in any of the four trees).

## Source aliases (absolute paths; short names used in citations)

| alias | path |
|---|---|
| `model.py` | `/Users/hckim/repo/upstream/deepseek-v41-inference/model.py` (1309 lines) |
| `kernel.py` | `/Users/hckim/repo/upstream/deepseek-v41-inference/kernel.py` (591 lines) |
| `config.json` | `/Users/hckim/repo/upstream/deepseek-v41-inference/config.json` |
| `ik` | `/Users/hckim/repo/ik_llama.cpp/src/…` |
| `fork` | `/Users/hckim/repo/llama.cpp-fork/src/…` and `/Users/hckim/repo/llama.cpp-fork/conversion/deepseek.py` |
| `bloomery` | `/Users/hckim/repo/bloomery/crates/model/src/attn.rs`, `moe.rs` |

**Which source wins where.** For the math: the reference (`model.py`/`kernel.py`). For GGUF tensor
names of the file we have: the mainline fork (`conversion/deepseek.py` writes it; `fork`'s
`models/deepseek41.cpp` loads it). `ik` is a cross-check and — important premise correction, see
§7.1 — **this `ik` snapshot implements DeepSeek-V4 (ratios 4/128), not V4.1 (ratios 2/1)**, and has
no engram and no `deepseek41` arch at all, so it cannot load our file; its citations are still
given wherever its graph realises an op the same way.

Working-tree check performed at start: `git worktree list` (not a git repo; plain directory).

---

## 1. Hyperparameters (V4.1-Flash values)

`config.json` nests everything under `text_config`; the fork's converter promotes it flat
(`conversion/deepseek.py:1200-1215`). Field names below are `ModelArgs` (`model.py:44-136`); the
config key is cited. Where a `ModelArgs` default is used because the key is absent, marked.

### 1.1 Global / layers
| ModelArgs | value | config key |
|---|---|---|
| `vocab_size` | 129 280 | `vocab_size` (`config.json:24`) |
| `dim` | 5 120 | `hidden_size` (`config.json:25`) |
| `n_layers` | 40 (backbone) | `num_hidden_layers` (`config.json:27`) |
| `n_mtp_layers` | 3 (DSpark layers, ids 40–42) | `num_nextn_predict_layers` (`config.json:145`) |
| dense layers | **none** — every backbone block is MoE (`Block.ffn = MoE`, `model.py:929`; the ports agree: `fork/models/deepseek41.cpp:894-914` has no dense branch) | — |
| `norm_eps` | 1e-20 | `rms_norm_eps` (`config.json:37`) |
| `max_seq_len`-ish | 1 048 576 | `max_position_embeddings` (`config.json:43`) |

Layer roles from the two per-layer vectors (both length 43 = backbone + DSpark, `model.py:80-81`):
- `compress_ratios` (`config.json:60-104`): layers 0–1 = **0** (sliding window only), layers 2–19 = **2**, layers 20–39 = **1**, DSpark 40–42 = **0**.
- `kv_source_layers` = [2, 8, 14, 20] (`config.json:106-111`): the only layers that *produce* compressed KV (own a `Compressor`, `model.py:654-659`).
- `index_source_layers` = [2, 8, 14, 20, 24, 28, 32, 36] (`config.json:112-121`): the only layers that *run* an `Indexer` and publish `topk_idxs` (`model.py:655-661`, `model.py:725-737`). Every source layer is ≥ its kv source, and {2,8,14,20} are also the index-key owners (`owns_k = layer_id in kv_source_layers`, `model.py:500`).
- The fork *derives* these roles from which tensors the file carries instead of trusting metadata (`fork/models/deepseek41.cpp:212-247`).

### 1.2 Attention / MLA dims
| ModelArgs | value | source |
|---|---|---|
| `n_heads` | 64 | `num_attention_heads` (`config.json:28`) |
| `head_dim` | 512 | `head_dim` (`config.json:30`) |
| `rope_head_dim` | 64 | `qk_rope_head_dim` (`config.json:31`) |
| nope part (**derived**) | 512 − 64 = 448 | `model.py:632` |
| KV heads | **1** — one shared 512-d latent serves as K *and* V for all heads (`num_key_value_heads`, `config.json:29`; `wkv: Linear(dim, head_dim)`, `model.py:643`; `sparse_attn(q, kv, …)` uses `kv` as both, `model.py:780`; fork passes `k_all` as both k and v, `fork/models/deepseek41.cpp:796`) |
| `q_lora_rank` | 1 280 | `q_lora_rank` (`config.json:32`) |
| `o_lora_rank` | 1 024 | `o_lora_rank` (`config.json:33`) |
| `o_groups` | 8 | `o_groups` (`config.json:34`) |
| `window_size` | 128 | `sliding_window` (`config.json:59`) |
| attn sinks | per-head fp32 param `attn_sink [n_heads]` (`model.py:639`; GGUF `blk.%d.attn_sinks` `{n_head}`, `fork/models/deepseek41.cpp:131`) — new vs V2-Lite |

### 1.3 Sparse-attention indexer / compressor
| parameter | value | source |
|---|---|---|
| `index_n_heads` | 32 | `config.json:122` |
| `index_head_dim` | 128 | `config.json:123` |
| `index_topk` | 512 | `config.json:124` |
| `candidate_source_layer` | 20 | `config.json:125` |
| `candidate_topk_blocks` | 2 048 | `config.json:126` |
| `candidate_block_size` | 8 | `config.json:127` |
| compress ratios per layer | see §1.1 | |
| `compress_rope_theta` | 160 000 | `config.json:105` |

What each piece does (details in §5):
- **`Compressor`** (`model.py:429-485`): pools `compress_ratio` consecutive tokens into **one
  512-d latent** = softmax-gated weighted sum of per-token `wkv(x)` values, weights =
  `softmax(wgate(x))` over the group; RMSNorm after pooling (`model.py:475-485`). Ratio 1 is a
  plain projection with **no gate** (`model.py:461-462`) — the fork's file then carries no
  `attn_compressor_gate` for layer 20 (`fork/models/deepseek41.cpp:149-153, 249-255` and the
  graph's fallback `fork/models/deepseek41.cpp:668-671`).
- **`Indexer`** (`model.py:488-580`): a small side-attention whose **queries** are 32 fp4 heads of
  dim 128 built from the *query's* low-rank `qr`, and whose **keys** are one 128-d vector per
  compressed position, derived by `k_norm(wk(latent))` from the *un-rotated* compressor latent of
  the kv-source layer (`model.py:544-547`). Scores: `Σ_heads relu(q·k) · proj(x)·(128^-0.5 · 32^-0.5)`
  (`model.py:550-557`); per query keep the `index_topk=512` best compressed positions, re-sorted by
  position (`model.py:578-580`).
- **`select_candidate_blocks`** (`model.py:583-610`): level one of a two-level top-k — score each
  block of 8 compressed positions by its best member, keep 2 048 blocks, pin the block containing
  the newest position (`model.py:604-605`); consumed by indexers of layers > 20 (`model.py:569-575`).
- **`get_window_topk_idxs`** (`model.py:409-426`): which ring slots the sliding window attends to;
  decode = the whole 128-ring, oldest first, `-1` while filling.

### 1.4 Hyperconnections
| parameter | value | source |
|---|---|---|
| `hc_mult` | 4 | `config.json:128`; `model.py:103` |
| `hc_sinkhorn_iters` | 20 | `config.json:129` |
| `hc_eps` | 1e-6 | `config.json:130` |

`make_identity_pre_mix` (`model.py:1159-1163`): layer 0's attention collapses the copies with a
one-hot on copy 0. `SharedAttentionRuntime` (`model.py:1166-1180`): a module-global hand-me-down
slot holding `compress_kv`, `index_k`, `topk_idxs`, `candidates` — this is how "shared compressed
KV" is realised in the reference (one tensor, last writer wins, later layers read).

### 1.5 Engram
| parameter | value | source |
|---|---|---|
| `engram_layer_ids` | [1, 14] | `config.json:131-134` |
| `engram_num_embeddings` | [384 006 168, 384 016 682] rows (unpadded) | `config.json:135-138` |
| `engram_max_ngram_size` | 4 | `config.json:139` |
| `engram_vocab_size` | 16 000 000 (prime-search start) | `config.json:140` |
| `engram_n_heads` | 8 | `config.json:141` |
| `engram_head_dim` | 256 | `config.json:142` |
| `engram_pad_id` | 2 | `config.json:143` |
| `engram_compressed_vocab_size` | 99 092 | `config.json:144` |

### 1.6 DSpark (draft head)
`dspark_block_size`=5, `dspark_noise_token_id`=128 799, `dspark_target_layer_ids`=[37,38,39],
`dspark_markov_rank`=256, `dspark_n_routed_experts`=128, `dspark_n_activated_experts`=3
(`config.json:146-155`). Backbone MoE config applies to DSpark layers via `get_moe_config`
(`model.py:142-149`) except those overrides.

### 1.7 Rope
- Window-only layers (0,1 and DSpark 40-42): base `rope_theta`=10 000, **YaRN disabled**
  (`model.py:680-687` — `original_seq_len=0` passed when `compress_ratio==0`).
- Compressed layers: base `compress_rope_theta`=160 000 with YaRN: `rope_factor`=16,
  `beta_fast`=32, `beta_slow`=1, `original_seq_len`=65 536 (`config.json:44-51, 105`; `model.py:688-696`).
- YaRN formula: per-dim frequency keep/fade ramp between `corrected_dim(beta_fast)` and
  `corrected_dim(beta_slow)` (`model.py:376-386`). Ports: identical split, plus an `attn_factor`
  `1/(1+0.1·ln(1/freq_scale))` multiplier on the mscale (`fork/models/deepseek4.cpp:11-17`,
  `fork/models/deepseek41.cpp:452-461`; `ik graphs/build_deepseek4.cpp:22-28, 995-1003`).

### 1.8 Reference dtype/quant scheme (`kernel.py`)
- Storage fp8-e4m3 with **e8m0 power-of-2 scales**, 32×32 weight blocks / 32-activation blocks
  (`model.py:26-30`; `config.json:12-21` `quant_method fp8, weight_block_size [32,32], scale_fmt ue8m0`).
  The V4.1 block is 32×32 (V4 used 128×128) — the converter reads it rather than assuming
  (`conversion/deepseek.py:1191-1196, 1223-1232`).
- Experts fp4-e2m1 packed 2/byte, one e8m0 scale per 32 along K (`model.py:28, 218-224`;
  `kernel.py:477-591`). The converter repacks routed experts to ggml **MXFP4**
  (`conversion/deepseek.py:721-746`).
- Activations quantized per 32 with e8m0 scales (`act_quant`, `kernel.py:98-124`); **in-place**
  variants quantize→dequantize back to bf16, i.e. the reference *simulates* the quantization noise
  of K/compressed-KV/indexer-k while storing bf16 (`model.py:707, 760, 846-847`; `kernel.py:41-42`).
  Window K: fp8 sim, 32-block e8m0 (`model.py:707`); compressed KV: fp4 sim, **16-groups with e4m3
  scales** (`model.py:760`); indexer k/q: fp4 sim, 32-groups e8m0 (`model.py:546, 552`).
- `sparse_attn` runs bf16 (`kernel.py:392-403`), softmax scale `head_dim**-0.5` = 512^-0.5
  (`model.py:651`).

---

## 2. Decode graph, one token, as math

Shapes use V4.1-Flash numbers; batch 1, one query at absolute position `p` (`start_pos=p`,
`seqlen=1`). `[embd, n_tokens]` convention. **derived** marks shapes computed from §1.

### 2.1 Front (before layer 0)
| # | op | inputs (param → GGUF name) | output shape | ref |
|---|---|---|---|---|
| F1 | row gather `h = W_emb[t]` | `embed.weight` → `token_embd.weight` `{5120, 129280}` (`fork/models/deepseek41.cpp:122`; ik same name, `ik/llama-model.cpp:1238`) | `[5120, 1]` | `model.py:1253, 152-178` |
| F2 | (VL only) overwrite image-span rows with ViT+aligner features; text decode skips | `image_start/end/newline`, `vision.*` (exported to a separate mmproj, `conversion/deepseek.py:1240-1245`) | — | `model.py:1254-1256, 1228-1239` |
| F3 | hc expand `h4 = repeat(h, hc)` | — | `[5120, 4, 1]` **derived** | `model.py:1258` |
| F4 | `pre_mix = one-hot(copy 0)` (only for layer 0; afterwards it is the previous layer's FFN pre mix) | — | `[4, 1]` | `model.py:1260, 1159-1163` |
| F5 | engram hash ids (host side in the fork): 24 row ids per engram layer for this token, from the previous 3 tokens (see §4) | KV-cell token ids (`fork/models/deepseek41.cpp:328`) | `[24]` per engram layer | `model.py:1252`; `fork/models/deepseek41.cpp:295-355` |
| F6 | DSpark capture: at layers 37/38/39, `main_hidden_i = mean over hc copies of the stream entering layer i` (`model.py:1264-1266`); concatenated at the end (`model.py:1271`) — note it is the **attention input**, taken before the block runs | — | `[5120·3, 1]` **derived** | `model.py:1264-1271` |

### 2.2 One backbone MoE layer `i` (generic; layer-specific branches marked)

**Engram (only i ∈ {1, 14}), applied to the hc stream before anything else:**
| # | op | params → GGUF | out | ref |
|---|---|---|---|---|
| E1 | gather 24 rows: `emb = table[ids]`, dequant row-wise (fp8·per-32 scale in reference form) | `engram.embed.weight(+.scale)` → `blk.N.engram_embd.weight` `{256, ~3.84e8}` read-lazy; `blk.N.engram_wkv.weight` `{6144·?→}` see §6 | `[24, 256]` | `model.py:353`; `ParallelEngramEmbedding` `model.py:296-325`; `fork/models/deepseek41.cpp:374-379` |
| E2 | flatten + GEMM `kv = W_wkv · vec(emb)` | `engram.wkv.weight` → `blk.N.engram_wkv.weight` `{6144, 25600}` (`fork/models/deepseek41.cpp:202`) | `[25600, 1]` **derived** (4·5120 key + 5120 value) | `model.py:345, 353-354` |
| E3 | split `key = kv[:20480] → [4,5120]`, `value = kv[20480:]` | | | `model.py:354-355` |
| E4 | gate: `dot_j = Σ_d h_j·(q⊙k)_d · rstd(h_j)·rstd(key_j) · 5120^-0.5`; `gate_j = σ(copysign(√max(|dot_j|,1e-6), dot_j))` | `engram.q_weight`,`engram.k_weight` → `blk.N.engram_q.weight`, `blk.N.engram_k.weight` `{5120, 4}` each (`fork/models/deepseek41.cpp:203-204`) — only the product is used (`model.py:356`) | `[4, 1]` | `model.py:357-362` |
| E5 | `h4 += gate ⊗ value` (value shared by the 4 copies; gate per copy) | | `[5120, 4, 1]` | `model.py:365` |

Fork realisation is op-for-op the same (`fork/models/deepseek41.cpp:382-437`), with one honest
numerics note: sgn(0) gates 0.5 vs the reference's copysign 0.50025 (`fork/models/deepseek41.cpp:427-431`).

**Attention sublayer:**
| # | op | params → GGUF | out | ref |
|---|---|---|---|---|
| A1 | hc mixes for attention (consumed by the **next** sublayer — the lag): `mixes = hc_attn_fn^T · rmsnorm(flatten(h4))`; `attn_pre = σ(mix[:4]·s0+b)+ε`; `attn_post = 2σ(mix[4:8]·s1+b)`; `comb = sinkhorn(mix[8:24]→[4,4])` (row-softmax+ε, col-norm, then 19×(row,col)) | `hc_attn_fn` → `blk.N.hc_attn_fn.weight` `{20480, 24}`, `hc_attn_base` `{24}`, `hc_attn_scale` `{3}` (`fork/models/deepseek41.cpp:142-144`) | pre/post `[4,1]`, comb `[4,4,1]` | `model.py:948-955, 982`; `kernel.py:406-474` |
| A2 | collapse with the **incoming** mix: `x = Σ_j pre_mix_in[j]·h4[j]` | — | `[5120, 1]` | `model.py:957-960, 983` |
| A3 | `x = attn_norm(x)` (RMSNorm, ε=1e-20) | `attn_norm.weight` → `blk.N.attn_norm.weight` `{5120}` | `[5120, 1]` | `model.py:984, 281-293` |
| A4 | `qr = q_norm(wq_a·x)` (norm sits on the **low-rank** query, dim 1280; there is **no** norm after `wq_b` — unlike V4) | `wq_a` → `blk.N.attn_q_a.weight` `{5120,1280}`; `q_norm` → `blk.N.attn_q_a_norm.weight` `{1280}` (`fork/models/deepseek41.cpp:132-133`) | `[1280, 1]` | `model.py:770`; V4-difference noted at `fork/models/deepseek4.cpp:938` |
| A5 | `q = wq_b·qr → 64 heads × 512`; rope on last 64 dims of each head at pos p (θ=160k+YaRN if layer compresses, else θ=10k no YaRN) | `wq_b` → `blk.N.attn_q_b.weight` `{1280, 32768}` (`fork/models/deepseek41.cpp:134`) | `[512, 64, 1]` **derived** | `model.py:771-772, 680-696` |
| A6 | window KV: `kv_w = rope(kv_norm(wkv·x))` (single latent, rope tail); fp8-sim quant | `wkv` → `blk.N.attn_kv.weight` `{5120, 512}`; `kv_norm` → `blk.N.attn_kv_a_norm.weight` `{512}` (`fork/models/deepseek41.cpp:135-136`; name mapping `fork/llama-arch.cpp:528-529`) | `[512, 1]` | `model.py:705-707, 700-720` |
| A7 | ring write `win_cache[p mod 128] = kv_w`; attend over all 128 slots (oldest first, −1 while filling) | cache (not a GGUF tensor; ports: SWA K-only cache, `fork/llama-kv-cache-dsv4.cpp:1263-1270`) | `[512, 128]` | `model.py:717-720, 409-426` |
| A8 | *(only kv-source layers 2,8,14,20)* compressor state update: `kvc = wkv_c·x` (fp32), `score = wgate_c·x` (layer 20, ratio 1: no gate — plain projection); slot `p mod ratio`; if `(p+1) mod ratio == 0`: `latent = rmsnorm( Σ_g softmax(score)_g · kvc_g )` | `attn.compressor.wkv` → `blk.N.attn_compressor_kv.weight` `{5120, 512}`; `attn.compressor.wgate` → `blk.N.attn_compressor_gate.weight` `{5120, 512}`; `attn.compressor.norm` → `blk.N.attn_compressor_norm.weight` `{512}` (`fork/models/deepseek41.cpp:151-153`) | `latent [512, 1]` every ratio-th token | `model.py:437-485` |
| A9 | *(only index-source layers)* indexer (§5 for full math): query heads from `qr`, keys from the owner's `latent`; top-512 compressed positions; publishes `shared_attn.topk_idxs`; non-source compressed layers **reuse** the published ids unchanged | `attn.indexer.wq_b` → `blk.N.indexer.attn_q_b.weight` `{1280, 4096}`; `attn.indexer.weights_proj` → `blk.N.indexer.proj.weight` `{5120, 32}`; *(owners only)* `attn.indexer.wk` → `blk.N.indexer.attn_k.weight` `{512, 128}`; `attn.indexer.k_norm` → `blk.N.indexer.k_norm.weight` `{128}` (`fork/models/deepseek41.cpp:157-160`; name map `conversion/deepseek.py:1478-1479`) | ids `[≤512]` | `model.py:527-580, 722-737` |
| A10 | *(owners only)* rotate `latent` tail at position `p+1−ratio` ("a latent stands for the first token of its group"), fp4-sim, write `compress_cache[p div ratio]` | cache | `[512, ≤max_seq/ratio]` | `model.py:749-761` |
| A11 | concat KV sources: `kv_all = [win(128) ; compress(≤512)]`, ids likewise (compressed ids offset by window length) | — | `[512, ≤640]` **derived** | `model.py:774-778` |
| A12 | `o = sparse_attn(q, kv_all, sink, ids, 512^-0.5)`: per head, online softmax over gathered rows; sink enters only the denominator: `Z += exp(sink_h − m)`; `o = Σ p_j·kv_all[id_j]` | `attn_sink` → `blk.N.attn_sinks.weight` `{64}` | `[512, 64, 1]` | `model.py:780`; `kernel.py:310-403` |
| A13 | inverse rope on `o`'s last 64 dims (conjugate rotation) — removes the query's rotation from the output | — | `[512, 64, 1]` | `model.py:781, 392-406`; ports: `ggml_rope_ext_back` (`fork/models/deepseek41.cpp:483-485`, `ik graphs/build_deepseek4.cpp:1237-1240`) |
| A14 | grouped output projection A: per group g (8 heads each), `oa_g = wo_a[g]·o_g` | `wo_a` → `blk.N.attn_output_a.weight`, file shape `{4096, 1024, 8}` (group-major 3-D; reshaped at load, `fork/models/deepseek41.cpp:137-139`) | `[1024·8, 1]` **derived** | `model.py:783-787` |
| A15 | output projection B: `x_att = wo_b·oa` | `wo_b` → `blk.N.attn_output_b.weight` `{8192, 5120}` (`fork/models/deepseek41.cpp:140`) | `[5120, 1]` | `model.py:788` |
| A16 | hc post: `h'_j = attn_post_j · x_att + Σ_k comb[j,k] · h4[k]` | — | `[5120, 4, 1]` | `model.py:962-966, 986` |

**FFN sublayer (MoE — every backbone layer):**
| # | op | params → GGUF | out | ref |
|---|---|---|---|---|
| M1 | hc mixes for FFN (`hc_ffn_*`), producing `ffn_pre` (for the *next* layer's attention), `ffn_post`, `ffn_comb` | `blk.N.hc_ffn_fn.weight {20480,24}`, `hc_ffn_base {24}`, `hc_ffn_scale {3}` (`fork/models/deepseek41.cpp:145-147`) | | `model.py:989` |
| M2 | collapse with A1's `attn_pre`: `x = Σ_j attn_pre_j · h'_j`; `x = ffn_norm(x)` | `blk.N.ffn_norm.weight {5120}` | `[5120, 1]` | `model.py:990-991` |
| M3 | router: `logits = ffn_gate_inp·x` (fp32); `s = √softplus(logits)`; `idx = top6(s + bias)`; `w = s[idx]`; `w /= Σw + 1e-20`; `w *= 1.5` | `ffn.gate.weight` → `blk.N.ffn_gate_inp.weight` `{5120, 384}`; `ffn.gate.bias` → `blk.N.exp_probs_b.bias` `{384}` (bias steers selection only, never the weight — "noaux_tc"); VL variant `blk.N.exp_probs_b_vl.bias` optional (`fork/models/deepseek41.cpp:162-165`) | idx `[6]`, w `[6]` | `model.py:809-827`; `fork/llama-graph.cpp:2025-2152` |
| M4 | routed experts e ∈ idx: `silu(clamp_max(gate,10)) · clamp(±10)(up)` then `w2·(w·…)` (the reference's form; ik computes `min(silu(gate),10)`, `iqk_mul_mat.cpp:155-157` — the two differ only for gate > 10, by ≤ 4.5e-5; noted by b4plan) | `ffn.experts.E.w1/w3` → `blk.N.ffn_gate_exps/up_exps.weight` `{5120, 2304, 384}`; `w2` → `blk.N.ffn_down_exps.weight` `{2304, 5120, 384}` (MXFP4 from conversion, `conversion/deepseek.py:721-746`) | `[5120, 1]` | `model.py:841-851, 893-900` |
| M5 | shared expert (always on, unweighted): same SwiGLU shape | `ffn.shared_experts.w1/w2/w3` → `blk.N.ffn_gate/down/up_shexp.weight` `{5120↔2304}` | `[5120, 1]` | `model.py:886-887, 903` |
| M6 | `y = Σ_e w·expert_e(x) + shared(x)` | | `[5120, 1]` | `model.py:893-904` |
| M7 | hc post with `ffn_post`/`ffn_comb`; block returns `(h_next, ffn_pre)` | | `[5120, 4, 1]`, `[4,1]` | `model.py:993-994` |

### 2.3 Tail (after layer 39)
| # | op | ref |
|---|---|---|
| T1 | final collapse with the **last** FFN's pre mix: `h_out = Σ_j ffn_pre_last[j]·h[j]` — V4.1 has **no learned hc head** (unlike V4: `output_hc_*` tensors, `fork/models/deepseek41.cpp:22-24`; `fork/models/deepseek4.cpp:106-108, 459-479`) | `model.py:1268`; `fork/models/deepseek41.cpp:934-937` |
| T2 | `x = output_norm(h_out)` — `output_norm.weight {5120}` | `model.py:1269` |
| T3 | `logits = head·x` (bf16 checkpoint held fp32) — `output.weight {129280, 5120}`, untied | `model.py:1008-1017, 1269` |
| T4 | Gumbel-max sample (temperature from CLI) | `model.py:1285-1292` |

### 2.4 DSpark path (draft only — does **not** run in plain decode)
`Transformer.forward` never calls `forward_spec`; the main path only *collects* `main_hidden`
(F6) and returns it (`model.py:1271-1272`). The DSpark layers (40–42) run when the caller invokes
`forward_spec(output_ids, main_hidden, p)` (`model.py:1274-1282`), i.e. only in speculative-draft
mode:
1. `main_x = main_norm(main_proj(main_hidden))`, `main_proj: 3·5120 → 5120` (`model.py:1128-1131, 1112-1114`).
2. draft block of 5 tokens: `[t, noise, noise, noise, noise]` (`dspark_block_size`=5, noise id 128 799), embedded with the **shared** `embed` table and expanded to hc copies (`model.py:1131-1135`; shared embed/head wired at `model.py:1212-1213`).
3. 3 stacked `DSparkBlock`s — `Block` bodies with `DSparkAttention`: window-only attention (ratio 0 asserted, `model.py:1034`) where the *main* model's token at `p` is written into the draft's own window cache and the 5 draft queries attend to `[window(128) ; own 5]` (`model.py:1039-1067`; index list `get_dspark_topk_idxs`, `model.py:1020-1029`). During the `start_pos==0` call the DSpark layers only **seed** their window cache (`model.py:1123-1126`).
4. `DSparkBlock.forward_head` (last stage only): collapse with incoming pre-mix, `norm`, shared `head` full-vocab logits; then a **Markov head** autoregressively refines the 5 draft steps: `logits[i] += markov_w2 · embed_markov(ids[i])` (`ParallelEmbedding` vocab×256 + `ParallelHead` 256→vocab, `model.py:1077-1086, 1149-1153`), sampling each step; a **confidence head** `proj([x ; markov_embed]) → 1` per step (`model.py:1089-1097, 1155`). Output `(5+1 ids, logits, confidence)` (`model.py:1137-1156`).
5. Mainline names for these (only consumed by the separate DFLASH draft arch, not by deepseek41): `markov_w1`, `markov_w2`, `conf_proj` (`fork/llama-arch.cpp:699-701`; converter `conversion/deepseek.py:932-938`).

**Ports vs DSpark:** mainline's only MTP graph for DEEPSEEK4/41 is the **V3-style single block**
`graph_mtp` — asserts `n_layer_nextn == 1` (`fork/models/deepseek4.cpp:1384-1385`), input
`concat(enorm(embd), hnorm(h)) @ eh_proj` (`fork/models/deepseek4.cpp:1426-1438`), i.e. *not* the
reference's `main_proj→main_norm` on 3 concatenated target-layer means nor the Markov/confidence
heads. `ik` likewise implements the eh_proj MTP (`ik/llama-build-context.cpp:482-507`) with its
dspark companion under a different arch (`ik/llama-dflash.cpp`, tensors
`ik/llama-model.cpp:1236+` DFLASH table). Neither port runs the reference's DSpark graph for
V4.1 — treat §2.4 as reference-only math (also listed in §7).

---

## 3. Three columns against `deepseek2` (V2-Lite)

V2-Lite side from `ik graphs/build_deepseek2.cpp` and our own implementation
(`bloomery/crates/model/src/attn.rs:1-53`, `moe.rs`).

### 3.1 SAME (tensor math identical)
| op | V4.1 | V2-Lite | refs |
|---|---|---|---|
| token embed / untied output head / final RMSNorm | F1, T2–T3 | same (`token_embd`, `output_norm`, `output`) | `model.py:1201-1206`; `bloomery/attn.rs:1-22` |
| pre-sublayer RMSNorm placement (attn_norm, ffn_norm) | A3, M2 | same | `model.py:933-934`; `ik build_deepseek2.cpp` (`attn_norm`/`ffn_norm`) |
| low-rank query with norm on the lora rank | A4 (1280) | `attn_q_a(2048→1536) → attn_q_a_norm → attn_q_b(1536→3072)` | `model.py:770`; `bloomery/attn.rs:37-48` |
| rope applied to a tail slice only, pairs-as-complex, YaRN ramp formula | A5/A6 | same rope math (`precompute_freqs_cis` vs V2's yarn) | `model.py:368-406` |
| MoE: router matmul in fp32, bias for selection only (noaux_tc), top-k, shared expert added unweighted | M3–M6 | V2-Lite: softmax scores + `exp_probs_b` bias, top-k, shared expert; no renorm, scale 1.0 | `model.py:809-827`; `bloomery/moe.rs:70-85, 176-181`; `ik build_deepseek2.cpp` (`ffn_gate_inp`, `exp_probs_b`) |
| SwiGLU expert shape silu(gate)·up → down | M4 | same | `model.py:841-851`; `bloomery/moe.rs:77-84` |

### 3.2 DIFFERS (exactly how)
| op | V2-Lite | V4.1 | refs (V4.1) |
|---|---|---|---|
| KV representation | MLA latent: `attn_kv_a_mqa: 2048→576 = [latent 512 ; rope 64]`, cached per token **per layer**; per-head K/V recovered by absorbing `wk_b [128,512,16]`/`wv_b [512,512,16]` into the attention (`bloomery/attn.rs:39-51`; `ik build_deepseek2.cpp:96-130, 153-224`) | one **shared 512-d latent per layer**, K=V with no up-projection at all (`wkv: 5120→512`); rope on its last 64 dims; **per-layer ring buffer of 128** replaces the unbounded cache; a second, *compressed* cache reaches further back (§5) | `model.py:643-644, 663-679, 700-720` |
| attention kernel | weight-absorbed MLA: q·k over `[rope 64 ; nope 128]` with the nope part contracted through `wk_b` into the latent; output through `wv_b` and `attn_output [2048×2048]` | direct 512-d dot of each of 64 query heads against the single shared K; softmax with a **learned per-head sink** added to the denominator; **inverse rope** on the output tail; output through the *grouped low-rank* `wo_a/wo_b` | `model.py:780-789, 639`; `kernel.py:382-387` |
| output projection | one dense `wo` | `wo_a` block-diagonal over 8 groups (8 heads each) to rank 1024, then `wo_b` back to 5120 — einsum not Linear in the reference because of the block structure | `model.py:645-650, 783-788` |
| residual stream | `h += attn_out`, `h += ffn_out` | hyper-connections: stream is **4 copies**; each sublayer collapses with the *previous* sublayer's mix (`pre`), scales its output by `post`, and mixes the residual through a doubly-stochastic `comb` (Sinkhorn, 20 iters); mixes are computed from the rms-normed flattened stream | `model.py:907-994, 948-966` |
| where the query norm sits | norm on `q_lora`(1536) only | same placement (norm on 1280, none after `wq_b`) — but V4 *does* norm after `wq_b` (`fork/models/deepseek4.cpp:938`), so this row is a V4.1-vs-V4 difference; vs V2-Lite it is "same" | `model.py:770-771` |
| MoE scoring | `softmax` over 384? (V2-Lite: 64 experts, top-6, no renorm, scale 1.0) | `√softplus` over 384 experts, top-6, **renormalize** selected weights, ×1.5; per-layer fp32 bias; swiglu clamps ±10 | `model.py:811-826`; `config.json:52-58` |
| rope bases | single rope (10k, YaRN) for everything | **split**: window-only layers 10k/no-YaRN; compressed layers 160k with YaRN-16; compressed rows rotate at the *group's first-token position* | `model.py:680-696, 752-757` |
| activation clamps | none | `up` clamped to ±10, `gate` to ≤10 (`swiglu_limit`) — fp8/fp4 range guard from training | `model.py:845-847`; ports `fork/llama-graph.cpp:1831-1837`, `ik/llama-build-context.cpp:1192` |
| KV cache growth | O(ctx) per layer | O(128) window per layer + O(ctx/ratio) compressed rows **shared per source group** (§5) | `model.py:663-679` |

### 3.3 NEW in V4.1 (absent from V2-Lite)
1. **Sparse attention stack**: `Compressor` + `Indexer` + `select_candidate_blocks` + shared compressed KV (`model.py:429-610`) — §5.
2. **Engram** n-gram tables (`model.py:296-365`) — §4.
3. **Hyper-connections** (`model.py:907-994`) replacing the plain residual — §2.2 A1/A16.
4. **DSpark** draft head (3 layers, block-of-5, Markov + confidence heads) vs V2-Lite's absence of a draft head in our deployment (`model.py:1020-1156`) — §2.4.
5. **Attention sinks** (`model.py:639`, `kernel.py:382-387`).
6. fp4 experts + e8m0-scale fp8 elsewhere (`model.py:26-30`) vs V2-Lite bf16/fp8-int8 GGUF quants.

### 3.4 Where the ports diverge from the reference or from each other
| topic | reference | mainline fork | ik |
|---|---|---|---|
| indexer keys | `k_norm(wk(latent_pre))` from the shared 512-d latent (`model.py:544-547`) | identical (`fork/models/deepseek41.cpp:698-716`, tensors `indexer.attn_k/k_norm` required only on owners) | **V4 form**: a *second compressor* (`indexer_compressor_kv/gate/ape/norm`) pools its own 128-d latent from x (`ik graphs/build_deepseek4.cpp:1045-1050, 890-951`; names `ik/llama-model.cpp:1275-1278`) |
| compressor per-slot bias (ape) | **absent** from `Compressor` (`model.py:437-456` has only `wkv/wgate/norm`) | absent from the V4.1 loader (`fork/models/deepseek41.cpp:151-153`); present only in the V4 loader (`fork/models/deepseek4.cpp:139`) and converter map (`conversion/deepseek.py:844`) | applied (`ggml_get_rows(ape, pos%ratio)` added to the score, `ik graphs/build_deepseek4.cpp:902-904`) — conclusion: ape is a V4-only tensor; V4.1's math has no such term |
| pooling overlap | disjoint consecutive groups (`model.py:473-482`) | non-overlap for both V4.1 streams (`fork/llama-kv-cache-dsv4.cpp:1427-1434` with `!is_v41` overlap only for V4; comment at 605-607) | overlap (2·ratio window) for its CSA/LID streams (`ik/llama-dsv4.cpp:1751-1753`, `ik graphs/build_deepseek4.cpp:795-813` V4 semantics) |
| two-level candidate mask | `select_candidate_blocks` pins the partial newest block (`model.py:583-610`) | **not implemented**; measured as a no-op until compressed length > 2048·8 positions, context capped there instead (`fork/models/deepseek41.cpp:32-34`) | not implemented (V4 code has no candidates) |
| hc mix timing / head | mixes lag by one sublayer; final collapse reuses the last FFN mix (`model.py:976-994, 1268`) | same, and no `output_hc_*` tensors (`fork/models/deepseek41.cpp:19-24, 826-829, 934-937`) | V4 form: mixes consumed in the same sublayer, learned `hc_head_fn/base/scale` (`ik graphs/build_deepseek4.cpp:626-682, 1489-1501`) |
| fp8/fp4 act-quant simulation in caches | simulated on window K (fp8/32/e8m0), compressed KV (fp4/16/e4m3), indexer k/q (fp4/32/e8m0) (`model.py:707, 760, 546, 552`) | not simulated; cache stored in the user-selected `--cache-type-k` (F16 default) — a numerics simplification, silently | same (plus optional random-orthogonal **hadamard** rotation of K/q to help quantized caches: `ik/llama.cpp:8450-8463`, `ik graphs/build_deepseek4.cpp:1023-1030`; fork equivalent `k_rot` inputs `fork/models/deepseek41.cpp:754-758, 798-800`) |
| decode-time compressed gather | `sparse_attn` gathers exactly the top-512 rows (`model.py:780`) | mask-based: builds a top-k mask over the full compressed length and relies on FA sparsity (`fork/models/deepseek4.cpp:697-724`); n_kv_max hint `swa + top_k` (`fork/models/deepseek41.cpp:792-794`) | **decode fast path**: with `n_tokens==1` it materializes only the selected rows via `ggml_get_rows_ext` on the CSA cache and mask (`ik graphs/build_deepseek4.cpp:1202-1214`) — closest to the reference's intent |
| MoE weight-sum epsilon | `Σ + 1e-20` (`model.py:825`) | sum clamped to ≥ 6.1035e-5 (`fork/llama-graph.cpp:2141`) | same clamp (`ik/llama-build-context.cpp:1547`) |
| engram | `model.py:296-365` | implemented, host-side hash, tables mmap-lazy (`fork/models/deepseek41.cpp:262-438`) | **none** (no `engram` string anywhere under `ik/src`) |
| sinkhorn iters fallback | 20 (config) | reads GGUF key | defaults to **3** if the key is missing (`ik/llama-hparams.cpp:2062-2064`) |

---

## 4. The engram path in detail

### 4.1 Index formation (which tokens, which hash)
The reference imports `EngramLayout, NgramHashState` from an `engram.py` that is **not present in
the reference repo** (only `model.py`, `kernel.py`, `config.json`, `README.md` exist — verified by
directory listing). The complete, load-bearing hash construction is therefore taken from the
mainline fork's converter and graph input, which agree with each other:

- **Token map**: every vocab id is decoded to text and normalized (NFKC → NFD → strip accents →
  lowercase → collapse whitespace, with a private-use sentinel preserving a lone space), then ids
  whose normalized forms coincide collapse to one **compressed id**; the compressed vocabulary
  must be exactly 99 092 (asserted against `engram_compressed_vocab_size`,
  `conversion/deepseek.py:1137-1170, 1299-1303`).
- **Multipliers**: per engram layer, `ngram_size` odd int64 multipliers, drawn from
  `np.random.default_rng(10007·layer_id)` in `[0, 2^63/99092/2)`, forced odd
  (`conversion/deepseek.py:1173-1185`).
- **Buckets**: for each layer, for each look-back `s ∈ [1, ngram−1]`, for each of 8 heads: the
  next unused prime counting up from `engram_vocab_size − 1` = 15 999 999 (48 primes total,
  shared `seen` set keeps buckets disjoint) (`conversion/deepseek.py:1107-1134, 1312-1329`).
- **Offsets**: per layer, buckets are laid out end-to-end (`cumsum` of that layer's 24 primes)
  (`conversion/deepseek.py:1331-1337`) — the two tables' row counts (384 006 168 / 384 016 682,
  `config.json:135-138`) are exactly these prime sums **derived**.
- **Row id at decode time** (host side, per token): context = `[map(t_p), map(t_{p−1}), map(t_{p−2}), map(t_{p−3})]`
  with look-back stopping at the sequence start (padding = mapped pad id 2;
  `fork/models/deepseek41.cpp:330-340`); rolling hash
  `rolling_s = (ctx[0]·m[0]) ⊕ … ⊕ (ctx[s]·m[s])`; for each bucket `b = (s−1)·8 + h`:
  `row_b = rolling_s mod prime[layer,b] + offset[layer,b]` (`fork/models/deepseek41.cpp:342-351`,
  formula comment at 262-265). So a bucket's s-th column uses the s-order rolling hash — i.e. the
  row already encodes "the last s+1 tokens".
- The reference calls this once per forward for **all** positions (`model.py:1252`) and slices per
  engram layer (`model.py:1263`); predecessors come from the KV cells in the fork
  (`fork/models/deepseek41.cpp:327-328`, `llama-kv-cache-dsv4.cpp:2046-2047`), which is what
  makes it work in decode without a separate token history.

### 4.2 Rows and bytes per decode token
- Columns per token per engram layer: `n_hash_cols = (max_ngram−1)·n_heads = 3·8 = 24`
  (`model.py:344`); two engram layers → **48 row reads per decode token** total.
- Row = 256 values. Reference storage: fp8-e4m3 with one e8m0 scale per 32 columns **per row**
  (scale shape `[rows, 8]`, `model.py:307-310`; scale layout documented at
  `conversion/deepseek.py:1380-1383`) = 256 + 8 = **264 B/row**.
- Our GGUF: the converter re-quantizes whole tables to **Q8_0** row-block-wise on a disk-backed
  memmap (`conversion/deepseek.py:1369-1457`, qtype fixed at 1403) = 8 blocks × (2 B fp16 scale +
  32 B) = **272 B/row** → per table ≈ 384 006 168 × 272 B ≈ **104.4 GB**; both tables ≈ **208.9 GB**
  (**derived**; the converter's own log line calls one table "393 GB as float32" at
  `conversion/deepseek.py:1376-1378`).
- Minimum bytes per decode token: 48 × 272 B = **13 056 B ≈ 12.8 KiB** (**derived**).

### 4.3 Access pattern (for an NVMe-resident design)
- **Random, not contiguous**: each of the 24 buckets hashes uniformly inside a disjoint prime-sized
  range; the 24 rows for one token are ~uniformly scattered over the whole ~104 GB table and are
  distinct by construction (disjoint ranges), with collisions only across tokens/heads by chance.
- Read granularity: one 272 B contiguous row per gather; with 4 KiB pages a cold row faults 1 page
  (~6 % of rows straddle two) → worst case ≈ 48 × 4 KiB ≈ 192 KiB of page-cache misses per token
  (**derived**). Prefill reads `24·n_tokens` rows as one batched gather
  (`fork/models/deepseek41.cpp:318-354`).
- The gather is `ggml_get_rows` on the table tensor (`fork/models/deepseek41.cpp:375`).

### 4.4 How the result enters the stream
Additive-gated (§2.2 E1–E5): `h += gate ⊗ value` per hc copy, before the block's attention, only
at layers 1 and 14 (`model.py:1262-1263`); image-span tokens get their gate zeroed
(`token_mask`, `model.py:350-364`; the fork substitutes the padding token for embd batches,
`fork/models/deepseek41.cpp:308-316`).

### 4.5 How each port stores the tables
- **mainline fork**: `TENSOR_READ_LAZY` (`fork/models/deepseek41.cpp:201`), i.e. the tensor is
  *not* loaded; it stays in a host-pointer buffer over the mmap of the GGUF
  (`fork/llama-model-loader.h:72, 87-118`; `fork/llama-model-loader.cpp:1082-1108` — requires
  mmap, auto-engages for tensors > 4 GiB; `fork/llama-model.cpp:1748-1771`), so `ggml_get_rows`
  faults rows in from NVMe through the page cache, host-side only ("lazy tensors are gathered on
  the host", `fork/llama-model-loader.h:112`). Bucket bounds are validated against the table's
  row count at load (`fork/models/deepseek41.py:191-199` in deepseek41.cpp:190-199).
- **ik**: no engram support (§7.1).

---

## 5. The sparse attention path in detail

### 5.1 What is indexed, and block geometry
- **Indexed unit**: one compressed row = `compress_ratio` consecutive tokens pooled into a 512-d
  latent (§2.2 A8). Ratios: 2 for layers 2–19, 1 for layers 20–39 (§1.1). A latent "stands for the
  first token of its group": group j gets rope position `j·ratio` (`model.py:538-543, 752-757`).
- **Top-k over what**: over *compressed positions* (not blocks) — per query, the 512 best of the
  reachable compressed rows (`model.py:578-580`), where reachable = blocks whose last token the
  query has passed: `visible = (p+1) div ratio` rows (`model.py:562-567`; fork identical at
  `fork/llama-kv-cache-dsv4.cpp:499-501`). During prefill, per-query causal visibility applies
  row-wise (`model.py:563-565`).
- **Second level** (candidate blocks, layers > 20): 2 048 blocks × 8 rows = 16 384 compressed
  positions pre-filter (`model.py:583-610`, `config.json:125-127`). Mainline skips it (§3.4).

### 5.2 What the "shared compressed KV" is shared across, and the cache layout vs V2-Lite
- Only layers {2, 8, 14, 20} compute a compressor; **every layer with the same ratio reads the
  same rows** through `shared_attn.compress_kv` (`model.py:739-763, 1166-1180`; "layers sharing a
  ratio also share one compressed KV and one indexer, produced by the first", `model.py:82-84`).
  Concretely: layers 2–7 share layer 2's rows, 8–13 share 8's, 14–19 share 14's (ratio 2); layers
  20–39 share layer 20's rows (ratio 1, one latent per token). Index keys (128-d) are likewise
  computed once per kv-source layer and read by the whole ratio group (`model.py:548`,
  `model.py:500`).
- **KV layout vs V2-Lite's MLA latent cache**: V2-Lite caches one 576-d `[rope; latent]` row per
  token *per layer*, unbounded, and reconstructs per-head K/V by weight absorption. V4.1 keeps,
  per layer: (a) a **128-row ring** of raw 512-d latents (window), overwritten cyclically; and,
  per *source group*: (b) `ceil(ctx/ratio)` **compressed 512-d rows** (rotated, fp4-sim in the
  reference) and (c) `ceil(ctx/ratio)` **index-key rows of 128 d** (rotated). There is no V cache
  at all — the same latents are used as V (`model.py:780`, `kernel.py:392-403`; fork
  `build_attn_mha(q, k_all, k_all, …)`, `fork/models/deepseek41.cpp:796`).
- **Visibility during decode** for the window part: the whole ring, oldest first, `-1` for slots
  not yet written (`model.py:417-426`); for the compressed part: `visible` rows (above) filtered
  to the indexer's top-512.

### 5.3 How the ports implement it
- **mainline** (`fork/llama-kv-cache-dsv4.{h,cpp}` — the dedicated cache this task asks about):
  - `llama_kv_cache_dsv4` owns: `kv_raw` = an **iswa SWA token cache** (K-only, window 128;
    `llama-kv-cache-dsv4.cpp:1263-1270`), `kv_csa` = **one** compressed K cache holding *all*
    compressed rows for *both* ratios, sized for the finest ratio with **a slot per source layer**,
    readers aliasing their source's slot via a reuse callback (`llama-kv-cache-dsv4.cpp:1321-1324,
    1390-1392, 1403-1406`); `kv_hca` = exists but empty for V4.1 (filter returns false,
    `llama-kv-cache-dsv4.cpp:1343-1346`); `kv_lid` = index-key cache (128-d rows) with the same
    aliasing (`1416-1422, 1394-1396`).
  - Per-ratio **compressor state** (`llama_dsv4_comp_state`): a `ratio`-row ring of fp32
    `kv`/`score` per source layer, carrying the partial group across ubatches (sizes at
    `llama-kv-cache-dsv4.cpp:1424-1441`; ring geometry `state_size = ratio` for V4.1).
  - A per-ubatch **comp_plan** (`llama-kv-cache-dsv4.h:280-326`) drives everything through
    set_rows/get_rows indirection tensors: `state_pos` (= `pos mod ratio`), persist rows,
    restore/snapshot rows (device-side rollback for speculative decoding), read idxs
    (`ratio·n_blocks` entries into `[persistent_state ; current_ubatch_scratch]`), write idxs+pos,
    `n_visible`, and a padded graph width `n_kv` (multiple of 256 so masked padding rows do not
    reshape the graph every other token; `llama-kv-cache-dsv4.cpp:605-627`). Masked attention is
    then built by concatenating `[raw_k ; comp_k]` and `[raw_mask ; top_k_mask]`
    (`fork/models/deepseek41.cpp:770-797`).
  - Decode-step subtlety mirrored from the reference: at ratio 2 a token completes a group only
    every other step; on off-steps a *dummy* block write keeps the graph shape (masked out) —
    comment at `llama-kv-cache-dsv4.cpp:605-607`.
- **ik** (`ik/llama-dsv4.cpp`): same architecture of plans/indirection tensors (clearly a common
  ancestor) but keyed to V4's constants: CSA ratio 4 with 2×-width *overlap* state, HCA ratio 128,
  LID with its own compressor states; caches `csa_k` (512-d), `lid_k` (128-d), `hca_k`, plus fp32
  state rings (`ik/llama-dsv4.cpp:883-1005, 1693-1798`); per-stream (per-sequence) replication of
  every cache (`n_stream = n_seq_max`, `ik/llama-dsv4.cpp:887-947`). Per-step/rollback
  checkpointing of the state rings for speculative decode: `ik/llama-dsv4.cpp:1137-1326`.

---

## 6. Tensor table (one MoE layer + non-block tensors)

Reference param → GGUF name (fork, authoritative for our file: written by
`conversion/deepseek.py:806-878, 1470-1484` and read by `fork/models/deepseek41.cpp:102-205`;
ik's alternates from `ik/llama-model.cpp:1236-1294` and `ik/llama-load-tensors.cpp:3380-3458`
noted where they differ). Shapes in the fork loader's `ne` notation (cite given); torch-shape in
the reference column. Quant column = our file `DeepSeek-V4.1-Flash-Q3_K_M-engramQ8-tokembdBF16-attnQ8`
(**inferred from name**: base Q3_K_M mix; regex `--tensor-type` overrides for engram_embd→Q8_0,
token_embd→BF16, `attn*`→Q8_0; override machinery at `fork/llama-quant.cpp:187-195, 713-748`;
never-quantized classes at `fork/llama-quant.cpp:287-359`).

### 6.1 Non-block
| reference | GGUF (fork) | ik alt | shape (fork ne) | role | quant (inferred) |
|---|---|---|---|---|---|
| `embed.weight` | `token_embd.weight` | same | `{5120, 129280}` | token embedding | **BF16** (suffix) |
| `norm.weight` | `output_norm.weight` | same | `{5120}` | final norm | F32/BF16 (norms never quantized, `fork/llama-quant.cpp:300`) |
| `head.weight` | `output.weight` | same | `{5120, 129280}` | lm head, untied | Q3_K (unless `--output-tensor-type`) |
| — (V4 only) | `output_hc_{fn,base,scale}.weight` | `hc_head_*` / `output_hc_*` | — | **absent in V4.1** (`fork/models/deepseek41.cpp:22-24`) | — |

### 6.2 Every backbone layer (40×; DSpark layers 40–42 same block tensors, MoE counts differ)
| reference | GGUF (fork) | ik alt | shape | role | quant (inferred) |
|---|---|---|---|---|---|
| `attn_norm.weight` | `blk.N.attn_norm.weight` | same | `{5120}` | pre-attn norm | F32/BF16 |
| `attn.attn_sink` | `blk.N.attn_sinks.weight` | same | `{64}` | softmax sink per head | F32 |
| `attn.wq_a.weight` | `blk.N.attn_q_a.weight` | same | `{5120, 1280}` | q down-proj | **Q8_0** (attn) |
| `attn.q_norm.weight` | `blk.N.attn_q_a_norm.weight` | same | `{1280}` | q lora norm | F32/BF16 |
| `attn.wq_b.weight` | `blk.N.attn_q_b.weight` | same | `{1280, 32768}` | q up-proj | Q8_0 |
| `attn.wkv.weight` | `blk.N.attn_kv.weight` | `attn_kv` / `attn_kv_latent` / `attn_kv_a_mqa` (ik tries 3, `ik/llama-load-tensors.cpp:3385-3389`) | `{5120, 512}` | shared K=V latent | Q8_0 |
| `attn.kv_norm.weight` | `blk.N.attn_kv_a_norm.weight` (name map `fork/llama-arch.cpp:529`) | `attn_kv_a_norm` | `{512}` | latent norm | F32/BF16 |
| `attn.wo_a.weight` | `blk.N.attn_output_a.weight` | same | `{4096, 1024, 8}` (3-D group-major; reshape-at-load, `fork/models/deepseek41.cpp:137-139`) | grouped o proj A | Q8_0 |
| `attn.wo_b.weight` | `blk.N.attn_output_b.weight` | same | `{8192, 5120}` | o proj B | Q8_0 |
| `hc_attn_fn/base/scale` | `blk.N.hc_attn_{fn,base,scale}.weight` | same | `{20480,24}`, `{24}`, `{3}` | hc mixes (attn) | fn Q3_K/Q8; base/scale F32 (1-D → not quantized, `fork/llama-quant.cpp:292`) |
| `hc_ffn_fn/base/scale` | `blk.N.hc_ffn_{fn,base,scale}.weight` | same | as above | hc mixes (ffn) | as above |
| `ffn_norm.weight` | `blk.N.ffn_norm.weight` | same | `{5120}` | pre-ffn norm | F32/BF16 |
| `ffn.gate.weight` | `blk.N.ffn_gate_inp.weight` | same | `{5120, 384}` | router (never quantized, `fork/llama-quant.cpp:306`) | BF16 |
| `ffn.gate.bias` | `blk.N.exp_probs_b.bias` | same (`exp_probs_b`) | `{384}` fp32 | selection bias | F32 |
| `ffn.gate.bias_vl` | `blk.N.exp_probs_b_vl.bias` | — (ik: none) | `{384}` | VL routing bias (optional) | F32 |
| `ffn.experts.E.w1/w3` | `blk.N.ffn_{gate,up}_exps.weight` | same | `{5120, 2304, 384}` | routed SwiGLU | MXFP4 from conversion (`conversion/deepseek.py:721-746`); re-quant target of the Q3_K_M run (**inferred**) |
| `ffn.experts.E.w2` | `blk.N.ffn_down_exps.weight` | same | `{2304, 5120, 384}` | routed down | as above |
| `ffn.shared_experts.w1/w2/w3` | `blk.N.ffn_{gate,down,up}_shexp.weight` | same | `{5120↔2304}` | shared expert | Q3_K mix |

### 6.3 Sparse-attention tensors (layer-dependent)
| reference | GGUF (fork) | ik alt | present on | shape | role |
|---|---|---|---|---|---|
| `attn.compressor.wkv.weight` | `blk.N.attn_compressor_kv.weight` | `attn_compress_kv`/`attn_compressor_kv` | N ∈ {2,8,14,20} | `{5120, 512}` | pooled-KV projection |
| `attn.compressor.wgate.weight` | `blk.N.attn_compressor_gate.weight` | ditto `_gate` | N ∈ {2,8,14} (ratio 2 only; ratio-1 has no gate, `fork/models/deepseek41.cpp:249-255`) | `{5120, 512}` fp32 | softmax gate |
| `attn.compressor.norm.weight` | `blk.N.attn_compressor_norm.weight` | ditto `_norm` | sources | `{512}` | post-pool norm |
| *(no reference counterpart)* | `blk.N.attn_compressor_ape.weight` | same | **V4 only** — V4.1 reference has no ape and the V4.1 loader never asks for it | `{512·k, ratio}` | per-slot gate bias (V4) |
| `attn.indexer.wq_b.weight` | `blk.N.indexer.attn_q_b.weight` | same (`indexer.attn_q_b`) | N ∈ index_source {2,8,14,20,24,28,32,36} | `{1280, 4096}` | indexer queries |
| `attn.indexer.weights_proj.weight` | `blk.N.indexer.proj.weight` | same | index sources | `{5120, 32}` bf16 | head weights |
| `attn.indexer.wk.weight` | `blk.N.indexer.attn_k.weight` | same name exists in ik loader (`ik/llama-load-tensors.cpp:3423`) but ik's *graph* uses the V4 second compressor instead | N ∈ kv_source {2,8,14,20} | `{512, 128}` bf16 | latent→index key |
| `attn.indexer.k_norm.weight` | `blk.N.indexer.k_norm.weight` | same | kv sources | `{128}` | index-key norm |
| *(V4 only)* | `blk.N.indexer_compressor_{kv,gate,ape,norm}.weight` | `indexer.compress_*`/`indexer_compressor_*` | V4 | — | second compressor (not in V4.1) |

### 6.4 Engram tensors (N ∈ {1, 14} only)
| reference (HF) | GGUF (fork) | shape | role | quant |
|---|---|---|---|---|
| `engram.embed.weight` (+`.scale`) | `blk.N.engram_embd.weight` | `{256, 384 006 168 / 384 016 682}` | n-gram table | **Q8_0** (converter writes Q8_0; suffix "engramQ8" = the quant-time override) |
| `engram.wkv.weight` | `blk.N.engram_wkv.weight` | `{6144, 25600}` | rows→(key per copy, shared value) | Q3_K mix |
| `engram.q_weight` | `blk.N.engram_q.weight` | `{5120, 4}` | gate scale q | **never quantized** (`fork/llama-quant.cpp:315-318`) |
| `engram.k_weight` | `blk.N.engram_k.weight` | `{5120, 4}` | gate scale k | never quantized |
| — | GGUF metadata `deepseek41.engram.{head_count,key_length,max_ngram_size,layer_ids,multipliers,primes,offsets,token_map,pad_id}` | u32 / u64 arrays | hash constants (§4.1) | — |

### 6.5 Draft (MTP/DSpark) tensors
V3-style nextn (what the fork's loader can consume for DEEPSEEK4/41): `blk.N.nextn.eh_proj.weight
{10240, 5120}`, `nextn.enorm/hnorm {5120}`, optional `nextn.embed_tokens`, `nextn.shared_head_head`,
`nextn.shared_head_norm` (`fork/models/deepseek4.cpp:175-182`). The reference's DSpark names map to
`markov_w1/markov_w2/conf_proj` (`fork/llama-arch.cpp:699-701`) — consumed only by the separate
DFLASH draft arch (`conversion/deepseek.py:927-938`). ik: `blk.N.nextn.*` same names
(`ik/llama-model.cpp:1288-1293`).

---

## 7. What I could not determine (with coordinates)

1. **This `ik` snapshot cannot run V4.1-Flash.** Premise correction to the task brief: `ik`'s arch
   name table has only `deepseek4` (`ik/llama-arch.cpp:57`) — no `deepseek41` — and its DSV4 graph
   and cache are hard-wired to ratios 4/128 (`ik/llama-context.h:501-502`,
   `ik/llama-model.cpp:2616-2641`); the ratio branches in its graph test only those two values
   (`ik graphs/build_deepseek4.cpp:1033, 1054, 1193, 1219`), so a hypothetical ratio-1/2 model would
   silently fall through to window-only attention. It also has zero engram support (grep over
   `ik/src` and `ik/ggml/src` for "engram" returns nothing). Everything V4.1-specific therefore
   rests on the reference + the mainline fork; `ik` is used here only as a second opinion on shared
   ops (attention tail, hc, MoE, window handling). I did not check whether `ik`'s own converter
   (outside `src/`) could emit a V4-shaped GGUF from V4.1 weights.
2. **The reference's `engram.py` is missing** from the reference repo (directory has only 4 files),
   so the hash's ground truth (multiplier RNG, prime search, token-map normalization order, pad
   handling) is the fork's reconstruction (`conversion/deepseek.py:1107-1185`), which the runtime
   consumes (`fork/models/deepseek41.cpp:295-355`). No independent second implementation exists in
   these trees to cross-check against.
3. **The actual GGUF was not opened** (absent from all four trees): every "quant in our file" claim
   in §6 is inferred from the filename suffixes; the fork's override machinery supports the
   interpretation (`fork/llama-quant.cpp:187-195, 713-748`) but the exact regex patterns used in
   the quant run are not recorded anywhere in the trees.
4. **DSpark execution**: no port implements the reference's 3-layer DSpark graph (mainline's MTP is
   V3-style and asserts one nextn layer, `fork/models/deepseek4.cpp:1384-1385`; `--dspark`
   conversion is registered for `DeepseekV4ForCausalLM` only, `convert_hf_to_gguf.py:269-271`).
   Whether the V4.1-Flash HF repo's `mtp.*` tensors would even round-trip through either port's
   draft path is undeterminable from these trees.
5. **`attn.compressor.ape` in the V4.1 HF checkpoint**: the converter maps it if present
   (`conversion/deepseek.py:844`) and the V4.1 loader ignores it; whether the released V4.1
   checkpoint carries it (it would then be an unused-tensor warning) cannot be checked without the
   checkpoint. The reference math (no ape) is what §2 uses.
6. **`gate_temp`**: not in `config.json`; the reference default 1.0 applies (`model.py:67`), and
   neither port reads such a key (their router has no temperature input,
   `fork/llama-graph.cpp:2025`). If the released model used a non-1 value it is invisible here.
7. **Fork cache-type defaults at runtime** (`-c K`/`--cache-type-k` for the dsv4 caches): the
   code validates F16/BF16/Q8_0 (`ik` analogue `ik/llama-dsv4.cpp:21-23`); the fork's accepted set
   is the same family (used via `params.type_k`, `fork/llama-model.cpp:2467-2470`), but which type
   our runs will use is a runtime choice, not determined by the file.

## 8. Three ops I am least sure about

1. **The engram hash** (§4.1). Its only source in these trees is the fork's converter, written to
   satisfy the fork's own runtime; the reference module is absent. If DeepSeek's actual
   `NgramHashState` differs in any constant — the `10007·layer` RNG seed, the prime-search start
   (15 999 999), the normalization chain, or pad-id mapping — every table read returns rows that
   are *plausible garbage* (the gate would just be poorly matched), with no shape or range error to
   surface it. Confidence in the *structure* (24 buckets, disjoint prime ranges, rolling XOR) is
   high because the row counts match the prime sums exactly; confidence in the *constants* is the
   weak part.
2. **Compressed-visibility boundary arithmetic** (§5.1): `visible = (p+1) div ratio`, group j at
   rope position `j·ratio`, decode rope index `start_pos+1−ratio`, and the every-ratio-th-token
   commit. Derived from `model.py:538-543, 562-567, 752-757` and independently implemented by the
   fork (`fork/llama-kv-cache-dsv4.cpp:499-527`); `ik`'s variant pools *overlapping* windows (V4),
   so only one port cross-checks the V4.1 form. An off-by-one here shifts which historical block a
   query may see — silent, and only visible in long-context quality.
3. **MoE weight normalization edge + routing dtype** (§2.2 M3): the reference normalizes with
   `Σ+1e-20` in fp32 (`model.py:824-826`) and both ports clamp at `6.1035e-5`
   (`fork/llama-graph.cpp:2141`, `ik/llama-build-context.cpp:1547`); with `√softplus` scores all
   six selected weights are positive so the edge rarely binds, but `gate_temp` is unverified (§7.6)
   and the ports accumulate router logits with an explicit f32 hint
   (`fork/llama-graph.cpp:2026-2028`) while the reference always casts to fp32
   (`model.py:811`) — a mixed-dtype router is the single most likely source of expert-id
   disagreements at tie boundaries in a reimplementation.
