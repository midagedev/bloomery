# V4.1 op map — the port's graph against the file we hold, op by op

The engineering map for the V4.1 rounds: one row per **op of the decode graph**, at the
granularity of a kernel in `crates/gpu`, each naming the port function that emits it, the GGUF
tensors it reads with their real names and shapes, and what our engine has to do about it.

It stands on the architecture reading and does not restate it. Read those first:

- `v41-ops.md` / `v41-ops-report.md` (B0a) — the *math* of the graph, primary source DeepSeek's
  official reference. Cited here, not repeated. That report read ik **`main`**, which carries V4
  and has no `deepseek41` arch, so its ik citations describe V4.
- `v41-ports.md` / `v41-ports-report.md` — the three V4.1 ports side by side, for *serving*
  strategy (engram on NVMe, expert cache, prefetch, major-fault counts). Cited here for anything
  about how a table is served.

What this adds: the ik tree we can read now **implements V4.1**, so every op can be tied to the
node that emits it; and every tensor is named with the shape and type `docs/v41-inventory.md`
measured from the file, rather than inferred from a name.

## Sources and how claims are marked

| mark | tree / commit | note |
|---|---|---|
| `[ik]` | **our own port, PR #2455**, at `c10fbbcc`. Checked out on the Mac at `/Users/hckim/repo/upstream/v41-ports/mine` **and** on the box at `/home/user/ik_llama.cpp` — same commit, so it reads locally. | Three V4.1 commits on top of ik `main`: `0c6e934a model : add DeepSeek-V4.1 (deepseek41)`, `b63c33e1` engram prefetch, `c10fbbcc` (stream naming). Cited `file:line`; every `[ik]` claim below is from `c10fbbcc` unless a commit is named. |
| `[ref]` | `/Users/hckim/repo/upstream/deepseek-v41-inference/config.json`, read directly for this document. | |
| `[ref*]` | a claim about `model.py`/`kernel.py` that I did **not** read here; taken from `v41-ops-report.md`, which cites it. | Two hops — weaker evidence, marked so it can be re-checked. |
| `[gguf]` | `docs/v41-inventory.md` — measured from the 9 shards by `gguf-inventory` (B0b, `e63caf3`). | Ground truth for what the file contains. |
| `[derived]` | arithmetic; the steps are shown. | |

The other two ports (`skelectric`, `phylliida`) were **not** read for this document; where serving
strategy matters, `v41-ports-report.md` is the source. Where `[ik]` and `[gguf]` disagree, the file
wins. Nothing here was measured on hardware; no model was loaded.

---

## 1. Hyperparameters

`[ref]` throughout the value column; the ik column says how the loader obtains it. The loader
**derives the layer roles from which tensors the file carries** rather than trusting metadata
(`llama-hparams.cpp:2149-2176` `[ik]`), so the file and the reference are independently checkable —
and they agree (§1.2).

### 1.1 Global

| parameter | value | `[ref]` key | `[ik]` |
|---|---:|---|---|
| vocab | 129,280 | `vocab_size` | `token_embd.weight` `5120×129280` bf16 `[gguf]` |
| `n_embd` | 5,120 | `hidden_size` | |
| backbone layers | 40, **all MoE** | `num_hidden_layers` | `n_layer_dense_lead` path exists but is unused for this file `[ik]` `build_deepseek4.cpp:1571` |
| rms eps | 1e-20 | `rms_norm_eps` | |
| heads | 64 | `num_attention_heads` | |
| head dim | 512 | `head_dim` | `attn_q_b` `1280×32768` = 64×512 `[gguf]` |
| KV heads | **1** | `num_key_value_heads` | one 512-latent is K *and* V: `dsv4_build_attn(…, k_all, k_all, …)` `[ik]` `build_deepseek4.cpp:1325` |
| rope dim | 64 of 512 | `qk_rope_head_dim` | `n_rot`; `build_rope` rotates the first 64 `[ik]` `:1112` |
| `q_lora_rank` | 1,280 | `q_lora_rank` | `attn_q_a` `5120×1280` `[gguf]` |
| `o_lora_rank` / `o_groups` | 1,024 / 8 | `o_lora_rank`, `o_groups` | asserted against `attn_output_a/_b` shapes at load `[ik]` `llama-hparams.cpp:2201-2205` |
| window | 128 | `sliding_window` | `n_swa` |
| experts | 384 routed + 1 shared, 6 used | `n_routed_experts`, … | |
| router scoring | `sqrtsoftplus`, `noaux_tc`, renorm, ×1.5 | `scoring_func`, `routed_scaling_factor` | |
| SwiGLU clamp | ±10 | `swiglu_limit` | |
| rope (window layers 0–1) | θ=10,000, **no YaRN** | `rope_theta` | `use_compress_rope == false` zeroes `ext_factor`/`beta_*`/`n_ctx_orig` `[ik]` `:1093-1100` |
| rope (compressed layers) | θ=160,000, YaRN ×16, β 32/1, orig 65,536 | `compress_rope_theta`, `rope_scaling` | `dsv4_compress_rope_base` `[ik]` `:1094` |

### 1.2 Layer roles — reference and file agree

| role | `[ref]` key | `[gguf]` — blocks carrying the deciding tensor | `[ik]` rule |
|---|---|---|---|
| compressed-KV source | `kv_source_layer_ids` = 2, 8, 14, 20 | `attn_compressor_kv`: **2, 8, 14, 20** | `has_comp` → `dsv41_kv_source` `llama-hparams.cpp:2150,2154` |
| pooling gate | (ratio > 1 only) | `attn_compressor_gate`: **2, 8, 14** — layer 20 has none | at ratio 1 a softmax over one element is 1, so the file ships no gate; a missing gate at ratio ≠ 1 throws `:2166` |
| index-key owner | = kv sources | `indexer.attn_k`: **2, 8, 14, 20** | `has_idx_k` → `dsv41_index_key_source` `:2152,2155` |
| index source (runs top-k) | `index_source_layer_ids` = 2, 8, 14, 20, 24, 28, 32, 36 | `indexer.attn_q_b`, `indexer.proj`: **2, 8, 14, 20, 24, 28, 32, 36** | `has_idx_q` → `dsv41_topk_source` `:2153,2156` |
| engram site | `engram_layer_ids` = 1, 14 | `engram_embd`: **1, 14** | `engram_index(il)` `llama-hparams.h:194` |
| compress ratio | `compress_ratios`: 0 for 0–1, 2 for 2–19, 1 for 20–39 | — | two segments held in `dsv4_csa_ratio`/`dsv4_hca_ratio`; for V4.1 these are **compression segments in layer order, not roles** `[ik]` `:2181-2188` |

`engram_num_embeddings` = [384,006,168, 384,016,682] `[ref]` matches `engram_embd` rows
`256×384006168` and `256×384016682` `[gguf]` exactly.

Integrity witness for the whole role mapping: `[derived]` the q3_K tensors this table implies are
80 expert (`ffn_{gate,up}_exps` × 40) + 80 hyperconnection (`hc_{attn,ffn}_fn` × 40) + 7 compressor
(4 `_kv` + 3 `_gate`) + 4 `indexer.attn_k` + 8 `indexer.proj` = **179**, and the inventory's type
table counts exactly 179 q3_K tensors `[gguf]`. Every q3_K tensor in the file is accounted for by a
role above, and no role needs one the file lacks.

### 1.3 Indexer, hyperconnections, engram

| parameter | value | `[ref]` key |
|---|---:|---|
| indexer heads × dim | 32 × 128 | `index_n_heads`, `index_head_dim` |
| indexer top-k | 512 | `index_topk` |
| candidate blocks / block size | 2,048 / 8, source layer 20 | `candidate_*` |
| hc streams | 4 | `hc_mult` |
| Sinkhorn iterations | 20 | `hc_sinkhorn_iters` |
| hc eps | 1e-6 | `hc_eps` |
| engram max n-gram | 4 | `engram_max_ngram_size` |
| engram heads | 8 | `engram_n_heads` |
| engram key dim | 256 | `engram_head_dim` |
| engram pad id | 2 | `engram_pad_token_id` |
| engram compressed vocab | 99,092 | `engram_compressed_vocab_size` |
| DSpark | block 5, target layers 37–39, markov rank 256, 128 experts / 3 used | `dspark_*`, `num_nextn_predict_layers` = 3 |

---

## 2. Op inventory

Granularity = one kernel in `crates/gpu`. "Unchanged" means the kernel runs as it stands once the
dtype and the shape are available; "changed" means the same kernel with a different contract;
"new" means no kernel in our tree does it.

### 2.1 Unchanged — the op V2-Lite already has, used as-is

| op | our kernel | `[ik]` site | GGUF tensors `[gguf]` |
|---|---|---|---|
| attn pre-norm | `rms_norm` | `build_deepseek4.cpp:1083` | `blk.N.attn_norm.weight` `5120` f32 |
| low-rank query projection | `q8_0_gemv` | `:1086` | `blk.N.attn_q_a.weight` `5120×1280` q8_0 |
| **low-rank query norm** | `rms_norm` | `:1089` | `blk.N.attn_q_a_norm.weight` `1280` f32 |
| query up-projection | `q8_0_gemv` | `:1103` via `build_rope` | `blk.N.attn_q_b.weight` `1280×32768` q8_0 |
| latent norm | `rms_norm` | `:1121` via `build_rope`'s `norm` arg | `blk.N.attn_kv_a_norm.weight` `512` f32 |
| ffn pre-norm | `rms_norm` | `:1568` | `blk.N.ffn_norm.weight` `5120` f32 |
| residual add of shared expert | `add` | `:1637` | — |
| final norm | `rms_norm` | `:1714` | `output_norm.weight` `5120` f32 |
| output head | `q6k_gemv` | `:1716` `build_output` | `output.weight` `5120×129280` q6_K |
| sampling | `argmax` | host | — |

### 2.2 Changed — same op, different shape, dtype or math

| op | our kernel | what changes | `[ik]` site | GGUF tensors `[gguf]` |
|---|---|---|---|---|
| token embed | `embed_rows` | dtype **bf16** (we have none); result is then broadcast into 4 hc streams | `:1520-1523` | `token_embd.weight` `5120×129280` **bf16** |
| latent K/V projection | `q8_0_gemv` | output is one 512 latent that serves as K **and** V; no up-projection at all | `:1121` | `blk.N.attn_kv.weight` `5120×512` q8_0 |
| rope | `rope` | first 64 dims of a **512** head (we rotate a 64-dim tail of a 192-wide MLA head); **two bases in one model**, θ=10,000 no-YaRN on layers 0–1 and θ=160,000 YaRN×16 elsewhere | `:1112`, bases at `:1093-1100` | — |
| window KV append | `kv_norm_rope_append` | a 128-row **ring**, written by plan index rather than by position | `:1216` `dsv4_raw_cpy_k` | — |
| attention | `flash_latent*` | K and V are the same tensor; a learned per-head **sink** joins the softmax denominator; the key set is `concat(window 128, selected compressed rows)` | `:1319-1326`, `dsv4_build_attn:496-625`, sink at `:590`/`:609` | `blk.N.attn_sinks.weight` `64` f32 |
| output projection | `q8_0_gemv` | **block-diagonal in 8 groups** then a dense second stage: `wo_a` is viewed as `[4096, 1024, 8]` and batched over groups | `:1415-1436` | `blk.N.attn_output_a.weight` `4096×8192` q8_0, `blk.N.attn_output_b.weight` `8192×5120` q8_0 |
| router | `router_topk` | score = √softplus, a **selection-only** bias added before top-6, weights renormalized then ×1.5; gate weights are **bf16** | `:1609-1624` | `blk.N.ffn_gate_inp.weight` `5120×384` **bf16**, `blk.N.exp_probs_b.bias` `384` f32 |
| routed experts | `expert_gate_up_swiglu_q3k`, `moe_combine`; down is `down_add_q5_1`'s slot but **not its dtype** | SwiGLU gains a **±10 clamp**; the down projection is q4_K on 38 blocks and **q5_K on blk.0–1**, and we have no q5_K path at all (§5 row 8) | `:1609` | `blk.N.ffn_{gate,up}_exps.weight` `5120×2304×384` q3_K; `blk.N.ffn_down_exps.weight` `2304×5120×384` q4_K, **q5_K on blk.0–1** |
| shared expert | `q8_0_gemv` ×3 + `swiglu` | weights are **q8_0**, and our fused gate/up SwiGLU kernels (`gate_up_swiglu_q3k`) are q3_K-only, so the fused form does not apply; the ±10 clamp applies here too | `:1629-1634` | `blk.N.ffn_{gate,up}_shexp.weight` `5120×2304` q8_0, `blk.N.ffn_down_shexp.weight` `2304×5120` q8_0 |
| hc collapse / weighting | `weighted_sum` | weights 4 streams by a per-stream vector instead of summing expert outputs | `:666` `build_mhc_weighted_sum` | — |
| hc pre-norm | `rms_norm` | over the **flattened 20,480** (= 4×5,120), not 5,120 | `:648-649` | — |

### 2.3 New — nothing in our tree does this

| op | what it is | `[ik]` site | GGUF tensors `[gguf]` |
|---|---|---|---|
| `hc_pre` mix head | `mul_mat` to 24 values, then an affine + sigmoid + **Sinkhorn (20 iters)** producing `pre[4]`, `post[4]`, `comb[4×4]` | `:651-658`, `ggml_hc_pre` at `:654` | `blk.N.hc_{attn,ffn}_fn.weight` `20480×24` q3_K, `…_base.weight` `24` f32, `…_scale.weight` `3` f32 |
| `hc_post` residual mix | new streams = `post ⊗ sublayer_out + comb · residual` | `:1439`, `:1643` `build_mhc_post` | — |
| compressor pooling | softmax-gated weighted sum of `ratio` consecutive tokens into one 512 latent, then norm and rope | `ds4_build_comp:912-979`, `build_compressed_kv_from_state:690-732`, `ggml_ds4_comp` at `:712` | `blk.N.attn_compressor_kv.weight` `5120×512` q3_K, `…_gate.weight` `5120×512` q3_K (2, 8, 14 only), `…_norm.weight` `512` f32 |
| indexer key build | project the **un-rotated** pooled latent to 128, norm, rope, Hadamard, write to the index cache | `:1153-1172` | `blk.N.indexer.attn_k.weight` `512×128` q3_K, `blk.N.indexer.k_norm.weight` `128` f32 |
| indexer scoring | 32 heads × 128 queries from `qr`; `Σ_heads relu(q·k) · proj(x)` scaled by `(128·32)^-1/2` | `dsv4_build_lid_top_k:838-904` | `blk.N.indexer.attn_q_b.weight` `1280×4096` q8_0, `blk.N.indexer.proj.weight` `5120×32` q3_K |
| top-k select | `ggml_top_k` over the masked scores, k = 512 | `:906` | — |
| gather selected rows | `ggml_get_rows_ext` on the compressed cache and its mask (the decode, `n_tokens == 1`, path) | `:1352-1355` | — |
| inverse rope | the attention output is de-rotated: the node is built as a rope then **re-tagged** `GGML_OP_ROPE_BACK` | `:1409-1412` | — |
| engram lookup | §4 | `ds4_build_engram:985-1047` | `blk.{1,14}.engram_*` |
| engram gate | signed-sqrt of a normed q·k, then sigmoid | `:1035-1039` | `blk.{1,14}.engram_{k,q}.weight` `5120×4` bf16 |
| Hadamard transform | fixed-size butterfly on indexer q/k (and on K when `k_cache_hadamard`) | `:849`, `:1166`, `:956` | — |

Counts: **10 unchanged, 11 changed, 11 new** — 32 rows, counted from the three tables above.

---

## 3. The five features

### 3.1 Hyperconnections

Four residual streams instead of one. The embedding is copied into all four `[ik]` `:1520-1523`;
every sublayer collapses them to one vector, runs, then re-expands.

- `build_hc_pre` `:627-667`: flatten `[5120×4, nt]` → `rms_norm` → `mul_mat(hc_*_fn)` → **24
  values** → `ggml_hc_pre(mixes, hc_scale, hc_base, hc=4, sinkhorn_iters=20, eps)` → one packed
  tensor sliced into `pre[4]`, `post[4]`, `comb[4×4]` `:656-658`.
- `[derived]` the 24 columns of `hc_*_fn` `20480×24` `[gguf]` are exactly `4 + 4 + 16`; `hc_*_scale`
  `3` is one scale per group. The file's shapes and the slicing agree.
- `comb` is made doubly stochastic by Sinkhorn; `[ref]` `hc_sinkhorn_iters` = 20.
- **The lag is the V4.1 difference.** `dsv4_hc_lag = true` `[ik]` `llama-hparams.cpp:2191`: a
  sublayer collapses with the mix the **previous** sublayer produced (`pre_in`), and hands its own
  `pre` to the next `:640-641, :666`. V4 consumes its own.
- Consequence: the last FFN's mix is the one nothing consumed, so it performs the **output
  collapse** `:1700-1703`, and the file ships **no `output_hc_*` / `hc_head_*` tensors** — confirmed,
  `[gguf]` has none.
- Layer 0 gets a one-hot `[1,0,0,0]` `:1535-1538`.

### 3.2 Shared compressed KV

- Only the four source layers pool; every later layer **aliases the source's cache**
  (`dsv41_is_kv_source` gates the pooling `:1142`).
- Pooling `[ik]` `ds4_build_comp:921-951`: `wkv(x)` and `wgate(x)` per token → `ggml_ds4_comp`
  pools `ratio` consecutive rows with `softmax(score)` weights, `type = 1` (disjoint groups) for
  V4.1 since both overlaps are false `:2189-2190` → RMSNorm → rope at θ=160,000.
- Ratio 2 for layers 2–19 and ratio 1 for 20–39 `[ref]`; at ratio 1 there is no gate `[gguf]` and
  the code substitutes the value itself `:924`.
- The index-key owner derives its keys from the **pre-rope** pooled latent `:719-722, :1153-1158`.
- The top-k pick is also shared: the index source publishes it and the readers of that stream
  reuse it `:1340-1348`.
- **Decode cost is depth-independent**: the key set is `window 128 + top-k 512` = at most **640
  rows** `[derived]` `:1349-1361`. What grows with depth is the indexer's scan over compressed
  positions (128-dim keys, ctx/ratio of them).

### 3.3 Low-rank query norm

- `wq_a` → `attn_q_a_norm` (1280) → `wq_b` `[ik]` `:1086-1089, :1103`.
- V4 additionally RMS-norms **each 512 head** after `wq_b`; V4.1 does not —
  `dsv4_q_head_norm = false` `:2192`, and `build_rope` then skips the norm because the layer also
  carries no weight for it `:1107`. The file has no `attn_q_b_norm` `[gguf]`.
- So this feature is a *removal* for us, not an addition: one `rms_norm` on a 1280 vector, a
  kernel we have.

### 3.4 The engram lookup path

See §4 — it is the one feature whose cost is not arithmetic but I/O.

### 3.5 DSpark heads (draft / verify)

This is where the sources disagree with each other, and it is the largest finding.

- `[ref]` describes it: `num_nextn_predict_layers` = 3, `dspark_block_size` = 5,
  `dspark_target_layer_ids` = [37, 38, 39], `dspark_markov_rank` = 256, 128 routed experts / 3 used.
- `[ik]` **has code for it**: `LLM_TENSOR_DSPARK_MARKOV_W1/W2`, `LLM_TENSOR_DSPARK_CONF_PROJ`
  (`llama-arch.h:503-505`), `build_dspark_logits` (`llama-build-context.h:528`), loaded at
  `llama-load-tensors.cpp:2669-2672`.
- `[gguf]` **our file ships none of it**: the inventory's `mtp_nextn` tier (`*mtp*`/`*nextn*`/
  `*spark*`) is **0 tensors, 0 bytes**.
- `[ik]` the V4.1 graph refuses MTP outright: `GGML_ASSERT(!is_mtp && "V4.1 MTP graph is not
  implemented")` `build_deepseek4.cpp:1534`.
- In plain decode the backbone only *collects* what a draft head would need; the `dflash.capture`
  branch stores a per-layer mean `:1645-1648`.

**For us**: DSpark is not implementable against this file and not on the B-phase critical path.
It should be scoped out until a file that carries the tensors exists.

---

## 4. The engram lookup path in detail

### 4.1 A token becomes row ids — on the host, before the graph

`[ik]` `llama.cpp:5383-5418`, `llama_set_engram_rows`:

```
n_gram  = 4      (engram_max_ngram_size)
n_heads = 8      (engram_n_heads)
n_cols  = (n_gram - 1) * n_heads = 24          rows per site per token
ctx[0]  = token_map[current token]
ctx[s]  = token_map[token s positions back]    (pad id 2 at a sequence start, and for
                                                every s beyond the start once blocked)
rolling = ctx[0] * mult[0]
for s in 1..3:
    rolling ^= ctx[s] * mult[s]
    for h in 0..7:
        b        = (s-1)*8 + h
        idx[b]   = rolling % prime[b] + offset[b]
```

- Bucket `b` is the (n-gram size `s+1`, head `h`) pair: 2-, 3- and 4-grams × 8 heads = 24.
- `mult`, `prime`, `offset`, `token_map`, `pad` are **read from GGUF metadata**, per site —
  `model.engram_{multipliers,primes,offsets,token_map,pad_id}` `[ik]` `:5392-5395`. We do not have
  to reproduce a prime search or a seed; the risk left is the formula, which is the eight lines above.

#### 4.1.1 The metadata seam — confirmed, with the exact keys and array shapes

The existing `v41-ops.md` says the hash constants live in GGUF metadata rather than in the
reference code. **The port confirms it**, and this is the seam the `engram` round builds against.

Keys (`[ik]` `llama-arch.cpp:237-245`; the `%s` is the arch name, so `deepseek41.engram.*`) and
where each is consumed (`[ik]` `llama-hparams.cpp:2075-2112`):

| GGUF key | type | required? | length, as the loader validates it | our value |
|---|---|---|---|---|
| `deepseek41.engram.layer_ids` | u32[] | yes `:2076-2084` | `engram_n_layer`, each `< n_layer` `:2092-2095` | 2 → [1, 14] |
| `deepseek41.engram.head_count` | u32 | yes `:2085` | ≥ 1 `:2089` | 8 |
| `deepseek41.engram.key_length` | u32 | yes `:2086` | — | 256 |
| `deepseek41.engram.max_ngram_size` | u32 | yes `:2087` | ≥ 2 `:2089` | 4 |
| `deepseek41.engram.pad_id` | u32 | yes `:2088` | — | 2 |
| `deepseek41.engram.multipliers` | u64[] | yes `:2097` | `n_layer_eg × max_ngram` `:2102` | `[derived]` 2×4 = **8** |
| `deepseek41.engram.primes` | u64[] | yes `:2098` | `n_layer_eg × (max_ngram−1) × n_head` `:2101,2105` | `[derived]` 2×24 = **48** |
| `deepseek41.engram.offsets` | u64[] | yes `:2099` | same as primes `:2106` | `[derived]` 2×24 = **48** |
| `deepseek41.engram.token_map` | u32[] | yes `:2100` | indexed by token id; out-of-range → `pad` `[ik]` `llama.cpp:5396-5399` | maps 129,280 → 99,092 `[ref]` |

Notes that matter for the seam:

- None of these carries a default: the `ml.get_key` calls at `:2085-2088` have no `false`
  (optional) argument, so a file missing any of them fails to load. The constants are **file data,
  not code** — we read them, we do not derive them.
- Every prime is checked non-zero at load `:2109-2111`, because it is a divisor in the hash.
- `key_length` is its own key, so the 256 is read, not inferred from the tensor shape — though the
  two agree (`engram_embd` is `256×N` `[gguf]`).
- The per-site stride is what `llama_set_engram_rows` uses: `mult + eg*n_gram`,
  `prime + eg*n_cols`, `offset + eg*n_cols` `[ik]` `llama.cpp:5392-5394`. Site 0 is blk.1, site 1
  is blk.14, in `layer_ids` order.
- **We cannot yet read these values.** `docs/v41-inventory.md` dumps tensors only, not KV — see §6
  item 3. Extending `gguf-inventory` to print KV is the `engram` round's prerequisite, and it is
  header-only (no lease, seconds).
- The ids depend only on the token history, so **the next token's ids are computable the moment it
  is sampled** — one full step of prefetch headroom. ik uses it: it walks the pages of the rows and
  starts the reads before the graph runs `[ik]` `:5419-5429` (commit `b63c33e1`).

### 4.2 Rows and bytes per token

| quantity | value | source |
|---|---:|---|
| rows per site per token | 24 | `(4−1)×8` `[ik]` `:5390` + `[ref]` |
| sites | 2 (blocks 1, 14) | `[ref]` `engram_layer_ids`, `[gguf]` |
| **rows per token** | **48** | `[derived]` 24 × 2 |
| values per row | 256 | `[ref]` `engram_head_dim`; `engram_embd` is `256×N` `[gguf]` |
| bytes per row (q8_0) | 272 | `[derived]` 256/32 = 8 blocks × (2 B scale + 32 B) = 272 |
| **bytes per token** | **13,056 B = 12.75 KiB** | `[derived]` 48 × 272 |

The 272 is confirmed against the file rather than assumed: `104,449,677,696 / 384,006,168 = 272`
exactly `[gguf]`. A third, independent check on the row count: `engram_wkv` is `6144×25600`
`[gguf]`, and `6144 = 24 × 256` — the projection's input is exactly the 24 rows of 256 laid end to
end. **The roofline's 12.75 KiB per token stands.**

Table size: `engram` tier = 209,236,612,640 B = 194.867 GiB over 8 tensors `[gguf]`. The two
`engram_embd` tables carry all but 0.3 GiB of it and sit in a shard of their own each: the
inventory's per-tensor `shard` column is the 0-based `split.no`, so blk.1's (`shard 1`,
104,449,677,696 B) is in **`…-00002-of-00009.gguf`** and blk.14's (`shard 4`, 104,452,537,504 B) is
in **`…-00005-of-00009.gguf`** — the two 6- and 8-tensor shards, 104.6 GB and 107.2 GB on disk
`[gguf]`. A serving design can mmap those two files and nothing else.

### 4.3 What the rows become

`[ik]` `ds4_build_engram:1005-1046`:

1. `ggml_get_rows(engram_embd, rows)` → `[256, 24·nt]`, reshaped to `[6144, nt]`.
2. `engram_wkv` `6144×25600` q8_0 → `[25600, nt]`, split as **4 keys of 5,120** (one per hc stream)
   and **1 value of 5,120** shared by all four `:1012-1013`. `[derived]` 25,600 = 5,120 × (4+1).
3. Key and the stream state `x` are each RMS-normed per (token, stream) over 5,120 and scaled by
   `engram_{k,q}` `5120×4` bf16 `:1025-1033`.
4. Gate: `s = Σ(k·q)/√5120`; `gate = sigmoid(sgn(s)·√|s|)` — a **signed square root** before the
   sigmoid `:1035-1039`.
5. `out = x + value ⊗ gate`, the same value added into all four streams `:1042-1044`.

The table is touched **only** through `get_rows`, deliberately, so it can stay in the file mapping
and never be materialized `[ik]` comment at `:983-984`.

---

## 5. What our engine must add, ranked by risk

Highest first. "Generalizes" names the existing kernel and the change.

| # | item | kind | why this rank |
|---|---|---|---|
| 1 | **engram serving** — 194.9 GiB table, 48 random 272 B reads per token, prefetch one step ahead | new host subsystem + `embed_rows` generalized to a mapped table | The only item whose cost is I/O, not arithmetic. Random reads into a 195 GiB mapping: throughput is major-fault *count*, not bytes (the ports measured 41–62 → 1–11 faults/token from prefetch alone, `v41-ports.md`). It needs a design, not a kernel. Row-id computation is eight lines and the constants come from the file. |
| 2 | **attention rewrite** — K = V, one 512 latent, per-head sink in the softmax, key set = window 128 ⧺ selected 512 | new kernel; `flash_latent*` does not generalize | Our flash path is MLA: latent KV **with** an up-projection and a 64-dim rope tail. V4.1 removes the up-projection, makes K and V one tensor, and adds a learned sink term to the denominator. That is a different kernel, and it is the hot one. |
| 3 | **indexer + top-k** — 32×128 queries, `Σ relu(q·k)·proj(x)`, select 512 of ctx/ratio | new kernel + new cache | Nothing in our tree selects k of a growing set. This is also the only part of decode whose cost **grows with depth**, so it decides the deep-row numbers we are judged on. |
| 4 | **compressed-KV cache shared between layers** — source writes, readers alias, top-k published once | new host structure | Our KV is per-layer and position-addressed. This is plan-addressed, ring-buffered, shared, and has two ratio segments. An aliasing bug here is silent (wrong rows, plausible output). |
| 5 | **hyperconnections** — 4 streams, `hc_pre`/`hc_post`, Sinkhorn 20, **lagged by one sublayer** | new small kernel + a change to every residual site | The kernel is cheap; the risk is that it touches *every* sublayer boundary and the lag is easy to get wrong by one. Getting it wrong costs quality, not a crash. |
| 6 | **q8_0 in the loader** | `GgmlType` + size table only | 332 tensors / 216 GB are q8_0 and `crates/gguf/src/quant.rs:30-40` has no variant for it, but `crates/gpu` already has `q8_0_gemv` and `q8_0_gemv_heads`. Here the gap really is the loader, not the kernel. Blocks everything — B1's item. |
| 7 | **bf16 — no read path anywhere** | new dequant/read path, or a load-time decode to f32 | 45 tensors: `token_embd` (read by `embed_rows`), every `ffn_gate_inp` (the router's own gemv), and `engram_{k,q}`. `GgmlType` has no bf16 variant, so these land as `Unknown(30)`. Unlike q8_0 there is no kernel waiting — decide read-path vs decode-at-load before B1 fixes the shape of the loader. |
| 8 | **q5_K — no kernel at all** | new gemv + dequant | `blk.0/1.ffn_down_exps` are q5_K (2 tensors, 6,228,541,440 B). Our GPU path refuses the type outright (`crates/gpu/src/weights.rs:253,356` return `None`/`Err`; `crates/gguf/src/quant.rs:198` `Unsupported`), and `quant.rs:37` says the enum tag exists "for the citation table only" — only the block size (176) is known. Two blocks of the model cannot run without this. |
| 9 | **compressor pooling** — softmax-gated pool of `ratio` tokens, on 4 source layers | new kernel | Small and local; ratio 1 (layers 20–39) is a plain projection with no gate. |
| 10 | **grouped output projection** — `wo_a` as `[4096, 1024, 8]` batched, then `wo_b` | `q8_0_gemv` generalized to a batched/block-diagonal form | Mechanical. `q3k_gemv_heads` already does a batched shape. |
| 11 | **two rope bases + inverse rope** | `rope` generalized | The inverse is the same kernel with a negated angle; ik literally builds a rope node and re-tags it `ROPE_BACK`. The per-layer base selection is host bookkeeping. |
| 12 | **router and SwiGLU deltas** — √softplus, selection-only bias, ×1.5, ±10 clamp | `router_topk`, `*_swiglu_*` generalized | Small, well-specified, and testable against the oracle. |
| 13 | **Hadamard transform** on indexer q/k | new small kernel | Only needed if we keep ik's Hadamard path; the reference's own scheme is a quantization simulation `[ref*]`. Confirms, rather than adds to, the deviation `v41-ops.md` already lists as a #2455 candidate ("인덱서 Hadamard 잔존") — so treat it as a port artifact and check against the reference before building. |

Not on the list: **DSpark** (§3.5 — the file carries none of it).

---

## 6. Disagreements, gaps, and what I could not determine

1. **DSpark**: `[ref]` specifies it, `[ik]` has loader and logit code for it, `[gguf]` has zero such
   tensors, and the V4.1 graph asserts MTP off. Any V4.1 plan that budgets for a draft head against
   *this* file is budgeting for nothing.
2. **`[ik]` carries V4 machinery this file never exercises**: hash-routed layers
   (`ffn_gate_tid2eid` gathered by token id, `build_deepseek4.cpp:1589-1591`) — the file has no
   `tid2eid` tensor `[gguf]`; and the vision bias `exp_probs_b_vl`, which **is** in the file on
   every block `[gguf]` but is only read for image batches `:1586-1588`. Neither belongs in our
   decode graph.
3. **The GGUF KV metadata is not in our inventory.** `docs/v41-inventory.md` lists tensors, not
   key/value metadata, so every hyperparameter above is `[ref]` (the reference config) or derived
   from a tensor shape — **not read from our file**. The engram hash constants
   (`multipliers`, `primes`, `offsets`, `token_map`, `pad_id`) live in that metadata and I have not
   seen their values. Extending `gguf-inventory` to dump KV is a prerequisite for B3, and it is
   cheap (header-only, no lease).
4. **`ggml_hc_pre`, `ggml_ds4_comp`, `ggml_top_k`, `ggml_get_rows_ext`, `ggml_hadamard`,
   `ggml_indexer_mask` are ggml ops** whose bodies live outside `src/` (in `ggml/src`). I read
   their call sites and shapes, not their implementations — so the *contract* of each is pinned
   here, the numerics are not. `ggml_hc_pre`'s internal order (affine → sigmoid → Sinkhorn) is
   inferred from `build_hc_head:681-687`, which does the un-Sinkhorned version inline; I did not
   read the fused op.
5. **Two ratio segments, not two roles.** `dsv4_csa_ratio`/`dsv4_hca_ratio` keep V4 names but for
   V4.1 hold the file's two compression segments in layer order `[ik]` `:2181-2188`. Reading those
   names as V4 roles is the trap; the graph dispatches on the ratio.
6. **`n_swa` = 128 is `[ref]`'s `sliding_window`**; I did not confirm the loader reads it from our
   file's metadata rather than defaulting.
7. **A silent default worth checking before we copy any constant from the port.** Unlike the
   engram keys, `hyper_connection.sinkhorn_iterations` is optional in the loader and falls back to
   **3** when the file does not carry it (`[ik]` `llama-hparams.cpp:2067-2068`), while `[ref]`
   specifies **20**. `hyper_connection.epsilon` likewise falls back to `f_norm_rms_eps` (1e-20)
   instead of `[ref]`'s 1e-6 `:2070-2072`. If our file omits either key, the port is running
   different arithmetic from the reference and so would we. Reading the file's KV settles it —
   the same prerequisite as item 3.

---

## 7. Against the prior documents

Explicit, because a contradiction is a finding and not something to overwrite.

**Confirmed** (the port now supplies first-hand evidence for what B0a inferred from the reference):

| prior claim | where | status |
|---|---|---|
| engram hash constants live in GGUF metadata, not in reference code | `v41-ops.md` | **Confirmed**, with keys and array lengths — §4.1.1. The reference ships no `engram.py`; the port reads all nine keys and refuses a file missing any. |
| 48 rows × 272 B = 12.75 KiB per token | `v41-ops.md`, `roofline.md:22` | **Confirmed three independent ways** — §4.2. The figure stands. |
| attention is not MLA: one 512 latent is K and V, window 128 + top-512, depth-independent | `v41-ops.md`, `roofline.md:97` | **Confirmed** at `build_deepseek4.cpp:1325` (K and V are the same tensor) and `:1349-1361` (≤ 640 rows). |
| hyperconnection mix is consumed one sublayer late; the file ships no `output_hc_*` | `v41-ops.md` | **Confirmed** — `dsv4_hc_lag = true` `llama-hparams.cpp:2191`, output collapse at `:1700-1703`, and `[gguf]` has no such tensor. |
| layer roles: kv sources {2,8,14,20}, index sources {2,8,14,20,24,28,32,36}, engram {1,14} | `v41-ops.md` | **Confirmed from the file itself** — §1.2. The port derives them from tensor presence, so file and reference agree independently. |
| the indexer Hadamard is a #2455 deviation from the reference | `v41-ops.md` | **Still present** at `c10fbbcc` (`:849`, `:1166`). Confirmation, not a new finding. |
| next token's engram rows are computable at sampling time (one step of prefetch headroom) | `v41-ops.md`, `v41-ports.md` | **Confirmed** — ids depend only on token history, and `b63c33e1` starts the reads before the graph runs `llama.cpp:5419-5429`. |

**Corrected or sharpened:**

1. **DSpark.** `v41-ops.md` says "두 포트 모두 참조의 DSpark 그래프를 구현하지 않았다" — read as
   *the ports lack the code*. At `c10fbbcc` that is too strong: the port **does** carry DSpark
   tensors and a logit builder (`llama-arch.h:503-505`, `llama-build-context.h:528`,
   `llama-load-tensors.cpp:2669`). What is missing is elsewhere and is firmer: **our file ships
   none of those tensors** (`[gguf]` `mtp_nextn` tier = 0 tensors, 0 bytes) and the V4.1 graph
   asserts MTP off (`build_deepseek4.cpp:1534`). Same conclusion for planning — DSpark is not
   reachable — but the reason is the file, not the port.
2. **"The local ik snapshot cannot run V4.1"** (`v41-ops.md`, already struck through there). The
   tree at `/home/user/ik_llama.cpp` *is* the V4.1 port; the struck line is correct to be struck.
   Noting it so no later round re-derives the old premise from the surrounding text.
3. **Two compression ratios are segments, not V4's two roles** — §6 item 5. `v41-ops.md`
   describes the behaviour correctly ("같은 비율 그룹의 층은 그 행을 공유한다") but the port keeps
   V4's `csa`/`hca` field names for them, which is the trap when reading the code.

No contradiction found between the port and `docs/v41-inventory.md`: every tensor the V4.1 graph
reads exists in the file with a compatible shape, and the q3_K count reconciles exactly (§1.2).
The disagreements are all port-vs-file *presence* (items above, plus §6 item 2), never shape.
