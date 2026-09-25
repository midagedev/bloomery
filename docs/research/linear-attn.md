# Linear attention — one delta-rule program for Gated DeltaNet and KDA

Date: 2026-09-24. Question (plan.md 「M4 모델 계열」, the `linear` row; 「M4 다른 모델 계열」 item 6): what
one program serves the linear-attention layers of Qwen3-Next / Qwen3.5–3.8 (Gated DeltaNet, "GDN") and
GLM-5.3-Flash (Kimi Delta Attention, "KDA", 34 of 45 layers), where its state lives in this engine, and in
what order to build it. This is the design survey the kernel rounds start from; no code is written here.

Rules of this document (the form of `models-survey.md`):

- Every claim carries its source: `file:line @ tree` for code, a URL at a pinned commit for papers and
  model cards, `[header]` for a value read from a GGUF header on the box, or **[derived]** with the
  arithmetic. Nothing is filled from memory; a thing not settled from sources is in §7.
- No number here was measured. Header reads are a pure-Python walk of the GGUF key/tensor tables (no
  numpy on the box) plus, for §1.4, a pread of six small F32 tensors (≤ 32 KiB each) — no model load.
- Engine facts cite this tree at base `340579f`.

Trees read (on the box, read-only):

| id | tree | commit | what it is |
|---|---|---|---|
| LCF | `/home/user/llama.cpp-v41` | `5210c7c5ed61` (2026-09-11) | our V4.1-port fork of mainline llama.cpp; its `src/models/*` and `ggml-cuda/*` are mainline's code for these archs. Called "mainline" below |
| IK | `/home/user/ik-idxkey` | `db517b690a8c` (2026-09-23) | the idxkey oracle tree (`# build db517b69` in the V4.1 sets) |
| FLA | github.com/fla-org/flash-linear-attention | `954438d1fcb5` (2026-09-21) | the reference PyTorch/Triton implementations |

`/home/user/ik_llama.cpp` (`c10fbbcc`) was listed but not read; IK is the tree our harnesses link.

## 0. Findings that change the plan

1. **One kernel, not two.** Mainline already runs GDN and KDA through one CUDA kernel templated on
   `bool KDA` (`ggml/src/ggml-cuda/gated_delta_net.cu:4-167 @ LCF`); the only difference inside it is
   whether the decay is one scalar per head (`:84-111`) or one value per key channel (`:112-141`). IK has
   two files (`delta-net.cu`, `kda.cu`) that are copies differing in the decay line
   (`ggml/src/ggml-cuda/kda.cu:128-130 @ IK` vs `delta-net.cu:128 @ IK`). The survey's open hypothesis
   (`models-survey.md` §6, "that they can share one kernel is a hypothesis") is answered: yes, with the
   gate granularity as a const generic.
2. ~~**Neither reference has a chunked CUDA kernel.**~~ Corrected 2026-09-25 (qwennext-lit): mistral.rs has one (`mistralrs-core/src/cuda/gdn.cu:909`), exllamav3 uses FLA's chunked form, and mainline has PRs #26001 and #29353 open. Original text: Mainline's launcher carries
   `//TODO: Add chunked kernel for even faster pre-fill` (`gated_delta_net.cu:180 @ LCF`): prefill runs the
   same recurrent kernel with the token loop inside it (`:63`), or, with the fused op off, a graph of ggml
   ops (`src/models/delta-net-base.cpp:16-287 @ LCF`). IK's kernel also loops tokens (`delta-net.cu:117
   @ IK`). So the first program needs only the recurrent kernel for both decode and prefill; a chunked
   kernel is a later lever with no CUDA oracle (§2.4, §4).
3. **Decode is weight-bound, not state-bound.** Qwen3.6-35B-A3B's recurrent state is 2 MiB per linear
   layer; reading and writing it is ~12 % of that layer's weight bytes and 4.6 % of the whole step
   [derived, §2]. The step's floor on the 3090 at the file on the box is ~324 tok/s, and **540 MB of the
   2.75 GB a step reads is the Q8_0 output head** (vocab 248,320) [derived]. plan.md:650's "1.1 GB/step,
   400–500 tok/s band" assumed 3 bpw everywhere; the published Q4_K_XL keeps every attention, GDN and
   head tensor at Q8_0 (7.5 bpw effective over the active set). The file, not the kernel, sets the line.
4. **A recurrent state breaks the cut/rollback rule the engine uses today.** V4.1's rollback works
   because every cache write is position-indexed and rewritten before it is read
   (`crates/gpu-deepseek41/src/body.rs:690-713`). A delta-rule step overwrites its state in place, so a
   rejected draft position, a `cut` below `pos`, or a prefix hit needs a *saved* state. Mainline solves
   this with K snapshot slots per sequence (`n_rs_seq`, `delta-net-base.cpp:563-603 @ LCF`); the design
   in §3 copies that shape as a two-slot ping-pong, so the state one position back is always held and a
   rollback is an index flip.
5. **The V-head order differs between the two Qwen generations.** Mainline's converter reorders V heads
   from grouped to tiled order for `qwen35`/`qwen35moe` (`conversion/qwen.py:453-470 @ LCF`,
   `_LinearAttentionVReorderBase`) but not for `qwen3next`; IK encodes the consequence as
   `repeat_type = ssm_beta_alpha ? 0 : 1` (`src/llama-delta-net.cpp:549,615 @ IK`), i.e. k-head
   `h / ratio` for Qwen3-Next files and `h % n_k` for Qwen3.5+ files (`delta-net.cu:52 @ IK`). The k-head
   map is a launch argument, not a constant.

## 1. The rule, per family

### 1.1 Notation

Per token, per value head h (H_v heads), key width d_k, value width d_v (all 128 here), state
S_h ∈ ℝ^{d_k × d_v} (FLA's layout: `initial_state: [B, H, K, V]`, `fla/ops/gated_delta_rule/naive.py:33
@ FLA`). With x the layer input after the pre-norm:

    c      = W_qkv · x                               (projection, one fused or three separate)
    [q̃;k̃;ṽ] = SiLU(conv4(c))                         (depthwise causal conv, width 4, per channel)
    q, k   = l2norm(q̃_h), l2norm(k̃_h)               (per head, over d_k)
    β_h    = sigmoid(w_β · x)                         (one scalar per v-head)
    g_h    = decay (≤ 0, log space; GDN: scalar; KDA: a d_k vector)
    S'     = Diag(exp g_h) · S_h                      (GDN: exp g_h · S_h)
    u      = β_h · (v − S'ᵀ k)                        (the delta, d_v)
    S_h    = S' + k uᵀ
    o_h    = S_hᵀ (q / √d_k)
    y      = W_out · [ RMSNorm_w(o_h) ⊙ act(z_h) ]_h   (per-head RMSNorm over d_v, weight shared by heads)

This is FLA's recurrence line for line: decay first (`naive.py:54`), `b_v - (h * b_k).sum(-2)` times β
(`:56-57`), rank-1 add (`:58`), `o = q·h` with q pre-scaled by `1/√K` (`:47-48, 59`); KDA the same with a
per-channel decay (`fla/ops/kda/naive.py:61-63 @ FLA`). The paper form is the transpose:
`S_t = S_{t−1}(α_t(I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ` with S ∈ ℝ^{d_v×d_k} (arXiv 2412.06464, Gated Delta
Networks, eq. 10, https://arxiv.org/html/2412.06464), and for KDA
`S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ`, `o_t = S_tᵀ q_t` (arXiv 2510.26692, Kimi Linear,
eq. 1, https://arxiv.org/html/2510.26692), α = exp g.

Heads that share a key: when H_v > H_k, value head h reads key/query head `h / (H_v/H_k)` (FLA's
`repeat_interleave`, `fla/ops/kda/naive.py:52 @ FLA`) — or `h % H_k` in a file whose converter tiled the
V heads (finding 5).

### 1.2 The shared core and the difference table

The shared core is everything in §1.1 that the table below does not mention: the conv (width 4, SiLU
after, no bias), the q/k L2-norm, β as sigmoid of a per-head linear map, the delta update and the output
`o = Sᵀq/√d`, the per-head RMSNorm over d_v with one 128-wide weight shared by all heads, f32 state.

| | Gated DeltaNet — Qwen3-Next, Qwen3.5/3.6, Qwen3.8-Flash-Next | KDA — GLM-5.3-Flash |
|---|---|---|
| heads (H_k / H_v × d) | Qwen3-Next and Coder-Next: 16 / 32 × 128 (config `linear_num_key_heads 16`, `linear_num_value_heads 32`, `linear_key_head_dim 128` [S14]; `qwen3next.ssm.group_count 16`, `time_step_rank 32`, `state_size 128` [header, Coder-Next]). Qwen3.6-35B-A3B: 16 / 32 × 128 [S16; header]. Qwen3.8-Flash-Next: 16 / 48 × 128 [S18; header `ssm.time_step_rank 48`, `inner_size 6144`] | 64 / 64 × 128, no key sharing (`linear_attn_config.num_heads 64`, `head_dim 128` [S1]; header `glm5next.kda.head_dim 128`, `attention.head_count 64`) |
| k-head map | Qwen3.5+ files: `h % H_k` (V heads tiled by the converter, `conversion/qwen.py:453-470 @ LCF`; IK `repeat_type 1`). Qwen3-Next/Coder-Next files: `h / (H_v/H_k)` (mainline repeat-interleave `src/models/qwen3next.cpp:516-537 @ LCF`; IK `repeat_type 0`, `llama-delta-net.cpp:549,615 @ IK`) | identity |
| q/k/v projection | one `attn_qkv` [n_embd, 2·H_k·d + H_v·d] = [2048, 8192] (Qwen3.6, Coder-Next [header]); Qwen3-Next's HF `in_proj_qkvz` is split into `attn_qkv` + `attn_gate` by the converter (`conversion/qwen.py:404-435 @ LCF`); an older GGUF form `ssm_in` is still read (`qwen3next.cpp:335-391 @ LCF`) | three: `attn_q`, `attn_k`, `attn_v`, each [4096, 8192] (`src/llama-load-tensors.cpp:4560-4562 @ IK`; [header]) |
| conv weights | one `ssm_conv1d` [4, C] over the concatenated channels, C = 8192 (Qwen3.6), 10240 (Flash-Next) [header] | three `ssm_conv1d_{q,k,v}` [4, 1, 8192] (`llama-load-tensors.cpp:4563-4568 @ IK`); IK concatenates them into one [4, 24576] conv (`src/llama-kda.cpp:70-78 @ IK`) — the same op over 3·8192 channels |
| β | `ssm_beta` [n_embd, H_v] (Qwen3.5+, F32 in the Qwen3.6 file [header]); Qwen3-Next: fused `ssm_ba` [n_embd, 2·H_v], split per k-head group into β then α (`qwen3next.cpp:418-455 @ LCF`; `llm_build_delta_net::build_beta_gate`, `llama-delta-net.cpp:240-262 @ IK`) | `ssm_beta` [4096, 64] (Q8_0 [header]) |
| decay g | **scalar per v-head**: `g = ssm_a · softplus(W_α x + ssm_dt.bias)`, `ssm_a` = −exp(A_log) folded by the converter (`conversion/qwen.py:395-396 @ LCF`); graph `qwen35moe.cpp:389-400 @ LCF`. `ssm_a` [32], `ssm_dt.bias` [32], `ssm_alpha` [n_embd, 32] [header] | **per (head, key channel)**: `g = lb · sigmoid(exp(A_log) · (f_b(f_a x) + ssm_dt.bias))`, lb = `kda.gate_lower_bound` = −5.0 [header]; `llama-kda.cpp:43-62 @ IK` (the comment at `:56-57` says `ssm_a` holds −exp(A_log), so the sigmoid is taken on `−ssm_a·(f+dt)`); FLA `naive_kda_lowerbound_gate` (`fla/ops/kda/gate.py:58-92 @ FLA`). `ssm_f_a` [4096, 128], `ssm_f_b` [128, 8192], `ssm_dt.bias` [8192], `ssm_a` [64] [header]. The decay multiplies S along the **key** axis (`kda.cu:128-130 @ IK`, `fla/ops/kda/naive.py:61 @ FLA`) |
| output gate z | `attn_gate` [n_embd, H_v·d] (full rank) | low-rank `ssm_g_b(ssm_g_a x)`, [4096,128] then [128, 8192] (`llama-kda.cpp:15-19 @ IK`; [header]) |
| gate activation | SiLU for Qwen3-Next/3.5/3.6 (`qwen35moe.cpp:267-276 @ LCF`); **sigmoid** for Qwen3.8-Flash-Next ("the one numerical difference from Qwen3.5's GDN", `qwen4exp.cpp:459-468 @ LCF`; config `output_gate_type sigmoid` [S18]) | sigmoid (`llama-kda.cpp:88 @ IK`; Kimi Linear eq. 10 `Sigmoid(W_g↑ W_g↓ x) ⊙ RMSNorm(·)`) |
| output projection | `ssm_out` [H_v·d, n_embd] | `attn_output` [8192, 4096] (`llama-load-tensors.cpp:4579-4580 @ IK`) |
| L2-norm / RMS eps | `layer_norm_rms_epsilon` 1e-6 [header] | 1e-5 [header] |
| block around it | pre-norm residual; `post_attention_norm` before the FFN (`qwen35moe.cpp:181-219 @ LCF`) | mHC 4 streams, Sinkhorn 20 (`build_glm5next.cpp:446-470 @ IK`); `attn_norm` applied inside the KDA call (`llama-kda.cpp:200-203 @ IK`) |
| mainline chunk size (graph path) | 64 | 16 (`delta-net-base.cpp:61 @ LCF`, `const int CS = kda ? 16 : 64`) |

A GDN step is a KDA step whose decay vector is one value repeated d_k times; the GDN branch of mainline's
kernel is the KDA branch with that value hoisted out of the loop (`gated_delta_net.cu:85-86` vs `:118`).

### 1.3 Layer patterns

| file (box) | `general.architecture` | layers | linear / full | how the file says it |
|---|---|---:|---|---|
| `/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (22,360,456,160 B) | `qwen35moe` | 40 | 30 GDN / 10 GQA (3, 7, …, 39) | `full_attention_interval 4` [header]; rule `(i+1) % 4 != 0` is linear (`qwen35moe.cpp:20-26 @ LCF`) |
| `/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q6_K.gguf` (29,308,320,736 B) | `qwen35moe` | 40 | same | [header] |
| `/models/Qwen3-Coder-Next/Qwen3-Coder-Next-IQ4_XS.gguf` (42,676,135,968 B) | `qwen3next` | 48 | 36 GDN / 12 GQA | `full_attention_interval 4` [header]; IK hard-codes the same rule (`src/llama-hparams.cpp:603-605 @ IK`) |
| `/models/Qwen3.8-Flash-Next/…-UD-Q4_K_XL-0000{1..4}-of-00004.gguf` (4 shards) | `qwen4exp` | 48 | 36 GDN / 12 GQA with a compressed indexer | `full_attention_interval 4`, `attention.compress_ratios` 4 on the full layers [header] |
| `/models/GLM-5.3-Flash-UD-Q2_K_XL/…-0000{1,2}-of-00004.gguf` (shard 3 still `.part`, 4 absent) | `glm5next` | 45 + 1 MTP | 34 KDA / 11 nope-MLA (3, 7, …, 43) | `attention.head_count_kv` array: 0 = KDA, 1 = MLA [header]; IK reads it so (`llama-hparams.cpp:2389-2391 @ IK`); config `kda_layers` / `full_attn_layers` agree [S1] |

Full-attention layer shapes:

- **Qwen3.6-35B-A3B**: GQA 16 / 2 heads × 256, `attn_q` [2048, 8192] carries query *and* a per-head
  output gate (2 × 16 × 256), `attn_k`/`attn_v` [2048, 512], `attn_q_norm`/`attn_k_norm` [256] [header];
  the gate is `sigmoid` on the attention output before `attn_output` (`qwen35moe.cpp:313-351 @ LCF`).
  IMROPE sections [11, 11, 10, 0], 64 of 256 dims, θ 1e7 [header; `qwen35moe.cpp:322-333 @ LCF`].
- **Qwen3-Coder-Next**: GQA 16 / 2 × 256, `rope.dimension_count 64`, θ 5e6, no sections key [header].
- **Qwen3.8-Flash-Next**: GQA 24 / 2 × 256 plus an indexer (4 heads × 128, top 2048) on compress ratio 4
  [header].
- **GLM-5.3-Flash**: MLA without rope, `q_lora_rank 1536`, `kv_lora_rank 512`, `key_length_mla` 256 per head,
  64 heads; K = V = the 512-wide latent (`build_glm5next.cpp:232-379 @ IK`, `wk_b` absorbs the query into
  the latent at `:278`, `wv_b` expands after at `:361-366`), `rope.dimension_count 0` [header]; a DSA
  k-pool indexer (32 heads × 128, top 2048, kpool 4) picks whole 4-token pools plus the incomplete tail
  (`build_glm5next.cpp:59-218 @ IK`).

MoE shapes: Qwen3.6 256 experts / 8 used / 1 shared with a sigmoid shared-expert gate
`ffn_gate_inp_shexp` [2048] (`qwen35moe.cpp:516-541 @ LCF`), expert ff 512 [header]; router softmax,
renormalised, no bias, no scale in both trees (`qwen35moe.cpp:498-512 @ LCF`: `LLM_FFN_SILU, true, …,
LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`; `src/graphs/build_qwen35.cpp:47-59 @ IK`: `true, false, 0.0f,
LLM_EXPERT_GATING_FUNC_SOFTMAX`). This is the Qwen3 MoE rule at other widths (256/8). Coder-Next 512/10,
ff 512 [header]. GLM-5.3 288/8 + 1, sigmoid (`expert_gating_func 2`), scale 2.5, 3 dense lead layers
[header].

### 1.4 Values checked in the files

The two converter folds the kernels depend on, read from the tensors themselves (pread of the F32 data):

| file | tensor | n | min | max | reading |
|---|---|---:|---:|---:|---|
| Qwen3.6 Q4_K_XL | `blk.0.ssm_a` | 32 | −72.33 | −0.0186 | negative: −exp(A_log) is folded, as `conversion/qwen.py:395-396` writes |
| GLM-5.3 Q2_K_XL shard 2 | `blk.0.ssm_a` | 64 | −8.159 | −1.417 | negative: −exp(A_log) folded; `exp(A_log)` ∈ [1.42, 8.16] inside the sigmoid |
| Qwen3.6 | `blk.0.ssm_norm.weight` | 128 | 0.523 | 0.957 | a plain RMSNorm weight — the converter's `+1` for zero-centred norms skips `linear_attn.norm` (`conversion/qwen.py:401-402 @ LCF`) |
| GLM-5.3 | `blk.0.ssm_norm.weight` | 128 | 0.0723 | 0.150 | plain weight |

No converter for `glm5next` exists in either tree on the box (grep of `*.py` for `glm5next`/`GLM5NEXT`:
no hit), so the GLM fold is established by the sign above, not by source.

## 2. Decode = recurrent step; prefill = one kernel over the tokens

### 2.1 State sizes [derived]

State per linear layer = H_v · d_k · d_v · 4 B (f32, mainline allocates the recurrent memory as
`GGML_TYPE_F32` for both the conv and the state, `src/llama-model.cpp:2579-2580 @ LCF`; config
`mamba_ssm_dtype float32` [S16, S18]). Conv window = (w − 1) · C · 4 B.

| model | H_v | state / layer | linear layers | state / sequence | conv / layer | conv / sequence |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B | 32 | 32·128·128·4 = 2,097,152 B | 30 | 62.9 MB | 3·8192·4 = 98,304 B | 2.9 MB |
| Qwen3-Coder-Next | 32 | 2,097,152 B | 36 | 75.5 MB | 98,304 B | 3.5 MB |
| Qwen3.8-Flash-Next | 48 | 3,145,728 B | 36 | 113.2 MB | 3·10240·4 = 122,880 B | 4.4 MB |
| GLM-5.3-Flash | 64 | 4,194,304 B | 34 | 142.6 MB | 3·24576·4 = 294,912 B | 10.0 MB |

### 2.2 The recurrent step, per token per layer [derived]

Bytes: the state is read once and written once (mainline keeps it in registers for the whole token
loop and stores at the end, `gated_delta_net.cu:57-61, 160-166 @ LCF`): 2 × state + 2 × conv. For
Qwen3.6: 2·2,097,152 + 2·98,304 = 4,390,912 B.

FLOPs per head: decay (d_k·d_v multiplies), Sᵀk (d_k·d_v FMA), rank-1 update (d_k·d_v FMA), Sᵀq
(d_k·d_v FMA) ≈ 7·d² = 114,688 FLOP; × 32 heads = 3.67 MFLOP per layer. Intensity 3.67e6 / 4.39e6 B ≈
0.84 FLOP/B, far under any card's ridge — the step is bandwidth-bound on its own bytes.

| | per layer | 30 layers (one Qwen3.6 token) |
|---|---:|---:|
| bytes | 4.39 MB | 131.7 MB |
| at 936 GB/s (3090 peak) | 4.7 µs | 141 µs |
| at 700 GB/s (A6000, the rate AGENTS.md cites for kernels near its roof) | 6.3 µs | 188 µs |
| at 233 GB/s (DGX Spark, the spec's figure) | 18.8 µs | 565 µs |

GLM-5.3 per KDA layer: 2·4,194,304 + 2·294,912 = 8.98 MB → 12.8 µs at 700 GB/s; 34 layers 305 MB
→ 0.44 ms per token at 700 GB/s [derived].

Against the layer's weights: the Qwen3.6 linear layer's projections (`attn_qkv` + `attn_gate` +
`ssm_out` at Q8_0, `ssm_alpha`/`ssm_beta` at F32) are 1,089.2 MB over 30 layers = 36.3 MB per layer
[derived from header dims × type block sizes: Q8_0 34/32 B, F32 4 B]. The state traffic is 12 % of it.

Against a full-attention layer: a Qwen3.6 GQA layer at depth D reads D · 2 (K,V) · 2 heads · 256 · 2 B
= 2,048 · D bytes of f16 cache. At D = 4096 that is 8.39 MB, twice the linear layer's 4.39 MB; the two
are equal at D ≈ 2,144 [derived]. So the linear layers are the depth-independent part of attention and
the 10 GQA layers carry all the depth growth. (AGENTS.md's "3.4 vs 0.87 µs per cached key per step" is
the CPU engine and ik on V2-Lite, a whole 27-layer step — rig-log `log/2026-09-21.md:69` @ `143a261` — and
is not a per-layer GPU figure; the byte comparison above is the one that transfers to the card.)

### 2.3 The whole step, and the floor [derived]

Active bytes per token of the Qwen3.6 Q4_K_XL file, from the header's dims and types (experts counted at
8 of 256):

| part | bytes |
|---|---:|
| 30 GDN layers' projections | 1,089.2 MB |
| 10 GQA layers' projections | 289.7 MB |
| 40 MoE layers (router F32, 8 routed + shared) | 832.6 MB |
| output head `output.weight` Q8_0 [2048, 248320] | 540.3 MB |
| norms | 0.7 MB |
| **weights** | **2,752.6 MB** |
| recurrent state + conv, read and write | 131.7 MB |
| GQA cache at depth D | 2,048 · 10 · D B (D = 4096: 83.9 MB) |

Step bytes ≈ 2,884 MB at depth ≈ 0 and 2,968 MB at depth 4096. Floor on the 3090 at 936 GB/s: 3.08 ms
→ **324 tok/s** at depth 0, 315 tok/s at 4096; on the A6000 at 700 GB/s: 243 / 236 tok/s.

The active set is 2.95 G values (head 0.51 G, GDN 1.01 G, GQA 0.27 G, MoE 1.15 G, same count with a
weight of 1 per value), so the file averages 7.47 bits per active value. At 3 bpw on every tensor the
step would be 1.10 GB, which is plan.md:650's figure — but no published file on the box is that file.
What moves the floor: the head (540 MB = 19 % of the step) and the GDN projections (38 %) are Q8_0; a
re-quantised head and GDN block at 4–5 bpw cut ~0.8 GB [derived: 1.52 G values × (8.5 − 4.5) bits]. No
kernel change moves it by that much.

### 2.4 Prefill

- **Recurrent-in-kernel** (both references): one launch walks the tokens with S in registers
  (`gated_delta_net.cu:63-158 @ LCF`; `delta-net.cu:117-177 @ IK`). The state is read and written once
  per launch, not per token; the cost is the serial token loop. Per token a warp does two dependent
  32-lane reductions (`gated_delta_net.cu:93, 107`); at ~~an assumed 0.2–0.4 µs per token~~ (corrected 2026-09-25, qwennext-lit: the public PR #29353 numbers give a lower bound of 1.44 µs per token per layer on a 3090, 5–7× this assumption, so the recurrence is ≈ 22 ms of a 512-ubatch, not "a few percent") that is
  0.8–1.6 ms per layer for a 4,096-token prompt, 25–50 ms over 30 layers [derived, the per-token latency
  is an assumption — §7 Q4 measures it]. The projections and experts for the same prompt are
  2 · 2.95e9 · 4096 ≈ 24 TFLOP [derived] — hundreds of milliseconds on either card. The recurrent loop is a
  few percent of prefill; a chunked kernel is not the first lever.
- **Chunked** (FLA, C = 64; Kimi Linear's kernel C = 64, arXiv 2510.26692; mainline's op graph C = 64 GDN,
  16 KDA): per chunk and head, KKᵀ and QKᵀ (2·2C²d), the triangular solve (~C³), W = A·Kβ and U = A·Vβ
  (2·2C²d), inter-chunk v′ = W·S, o += Q·S and S += KᵀU (3·2Cd²). Per token ≈ 6d² + 10Cd + C² = 98,304 +
  81,920 + 4,096 ≈ 184 kFLOP per head, 1.6× the recurrent count but on tensor cores, and parallel over
  chunks within the intra-chunk part [derived]. It pays once prefill is dominated by the token loop,
  which §2.4's first bullet says it is not at these widths.

## 3. The state cache in this engine

### 3.1 Where it sits

The per-layer cache today is a field of the architecture's `ChainBody`: deepseek2 holds
`kv: Vec<DeviceTensor<u16>>`, one `[ctx_max, 576]` plane per resident layer
(`crates/gpu/src/arch/deepseek2/mod.rs:67-80`), and the flash kernels take a plane and a live-key count
(`crates/gpu/src/flash.rs:1-40`). The placement's byte function is per architecture and per layer
(`crates/model/src/placement.rs:153-155`, `KvBytes::layer_bytes(layer, ctx_max)`; V4.1's is
`crates/model/src/arch/deepseek41/kv.rs:85-100`, Qwen3's `crates/model/src/arch/qwen3moe/kv.rs`).

A linear-attention body holds, beside the GQA/MLA planes of its full layers:

    struct LinearState {               // one per linear layer
        s:    DeviceTensor<f32>,       // [SLOTS][H_v][d_k][d_v]
        conv: DeviceTensor<f32>,       // [(w - 1) + K_MAX][C]  -- a ring of conv inputs
    }

- `KvBytes::layer_bytes` for a linear layer returns `SLOTS · H_v·d_k·d_v·4 + ((w−1)+K_MAX)·C·4`,
  independent of `ctx_max`; `placement.rs` does not change (its `Role::Attention` already covers "attention,
  its compressor and its indexer", `placement.rs:36-38`, and the linear weights are attention weights).
- `d_k`, `d_v`, `H_v`, `H_k`, the k-head map, C and w are read once in the arch's `hparams.rs` from the
  keys of §1.2 and refused loudly when absent (arch-split's "계획은 기본값을 만들지 않는다").

### 3.2 What a slot can grant: `keepable` / `cut` / `rollback`

The serve slot asks the engine for the longest prefix it can keep and cuts to it
(`crates/serve/src/genloop.rs:161-182`; contract `crates/serve/src/engine.rs:72-83`). For the full
layers any `n ≤ pos` is grantable (rows are position-indexed). For a linear layer only positions whose
state is still held are: `pos` itself, and each saved snapshot.

- **Multi-turn chat needs no snapshot.** The slot never feeds back the last generated id
  (`genloop.rs:143-145`), so a follow-up turn's prompt extends `held`; the longest common prefix is `held`,
  `ask = min(common, len − 1)` (`genloop.rs:171`) equals `pos`, and the live state is the answer.
- **"Regenerate the same prompt" asks for `len − 1`** — one position back. The rollback slot of §3.3 covers
  it at no extra cost.
- **A snapshot every N tokens** costs `state/sequence` per snapshot: Qwen3.6 65.8 MB (state + conv), so at
  ctx 32,768 with N = 256 that is 128 × 65.8 MB = 8.4 GB, with N = 2,048 1.05 GB [derived]. On a 24 GB
  card holding a 22.4 GB file neither fits; on the host it is a 65.8 MB device-to-host copy per snapshot
  (~2.6 ms at 25 GB/s over PCIe [derived, the rate is an assumption]).
- **The sane policy**: grant `{pos, pos − 1, 0}` on the card (live + the rollback slot), plus one host
  snapshot at the end of each prompt prefill — the point a client that edits its last answer returns to.
  This is the V4.1 grant `{m, m−1, 0}` (`crates/gpu-gates/src/bind.rs:352-370`) with one host point added.

### 3.3 Draft speculation: `step_pair` and rollback

`step_pair(t, t1)` runs two positions in one pass and "a caller that does not keep `t1` takes it back with
`GpuModel::rollback`" (`crates/gpu/src/model.rs:806-812, 947-960`; the body hooks default to refusing,
`model.rs:187-236`). V4.1's body takes back one position because its caches need nothing undone
(`crates/gpu-deepseek41/src/body.rs:690-713`). A linear layer after the pair holds the state after `t1`;
if `t1` is rejected it needs the state after `t`.

References: mainline writes the last K = `n_rs_seq + 1` per-token states into K slots from inside the
kernel (`gated_delta_net.cu:145-157 @ LCF`, slot 0 = newest), the conv windows likewise
(`delta-net-base.cpp:497-521`), allocates `mem_size · (1 + n_rs_seq)` rows (`src/llama-memory-recurrent.cpp:101-103
@ LCF`), and a `seq_rm` of up to `n_rs_seq` trailing positions selects the snapshot by index instead of
copying (`llama-memory-recurrent.cpp:193-203`, single-use). IK saves every step of a short batch into a
per-step checkpoint buffer (`delta-net.cu:162-169 @ IK`) and restores by an async copy
(`src/llama.cpp:2368-2440 @ IK`).

Design, following mainline:

- `SLOTS = 2`, always ping-pong: the kernel reads S from slot `r` into registers and writes the state
  after the pass's last token to slot `1 − r`, then `r` flips. Slot `1 − r` (after the flip) still holds
  the state one position back, so `pos − 1` exists after every plain step. A pair pass writes the state
  after `t` to `1 − r` and the state after `t1` back to `r` (safe: `r` is already in registers) and does
  not flip: a rejected `t1` is taken back by flipping `r`, a kept one by nothing. Rollback of one position
  is a flip — no copy. Cost: a plain step writes the state once, as an in-place kernel would; a pair pass
  writes it twice, 30 × 2 MiB = 62.9 MB more on the pair pass's ~2.9 GB, ~2 % [derived]; memory: one more
  state per sequence (+62.9 MB Qwen3.6, +142.6 MB GLM).
- The conv window is a ring of the last `(w − 1) + K_MAX` inputs with a head index; rollback moves the head.
- **The capture seam.** A captured graph fixes buffer addresses, so `r` must be a device scalar the
  kernels read each replay (like `n_keys_buf`, `flash.rs:28-33`) or the body captures two graphs, one per
  parity. The device scalar keeps one graph and touches only the body; two graphs touch `GpuModel`'s
  capture bookkeeping in `crates/gpu/src/model.rs`, the one-round-at-a-time file (arch-split.md, M2 row).
  Recommend the device scalar.
- A later k > 2 draft (MTP heads) raises `SLOTS` to k; the kernel writes slot `(r + j) mod k` after token j.

### 3.4 The seam: files

Read-only judgment of what a Qwen3.6 round touches, following `docs/arch-split.md` (arch name = the GGUF
string; one `arch/<name>/` directory of six files; one `ChainBody` per architecture; kernels shape-named):

New:

- `crates/model/src/arch/qwen35moe/{hparams,names,roles,kv,place,host}.rs` — `kv.rs` carries §3.1's byte
  function; `host.rs` the host-tier spec for the 256-expert stack (host tier only when the card is short).
- `crates/gpu/src/linear/mod.rs`, `conv.rs` (conv + SiLU + L2-norm + β/g prep, one launch),
  `delta.rs` (the recurrent kernel, `DECAY: Scalar | Channel` const generic, the k-head map and `SLOTS`
  as launch arguments), `norm_gate.rs` (per-head RMSNorm × SiLU/sigmoid, one const generic) — shared by
  every GDN/KDA arch, so in `crates/gpu` beside `flash.rs`, not in an arch crate. `delta_chunk.rs` only
  when §2.4's lever is opened.
- `crates/gpu/src/arch/qwen35moe/` — the `ChainBody` (its kernels are all shared ones: GQA flash, rope,
  router, `_sel`, `linear/*`).
- `tools/ref/models/qwen35moe.sh`, `crates/gpu-gates/src/oracle/qwen35moe.rs`, gate bins
  `gate_qwen35moe_<op>.rs`.
- For GLM-5.3 later: `crates/model/src/arch/glm5next/*` and a device crate `crates/gpu-glm5next` (it adds
  device code only it uses — the KDA gate prep, the k-pool indexer — and calls `gpu-deepseek41`'s hc and
  512-wide attention cores, which S0 showed inline across crates when `pub` + `#[inline(always)]`,
  arch-split.md 「열린 것」). Entry names take a `g53_` prefix (kernel entry names are binary-global,
  nvlabs-ledger #12 via arch-split.md).

Changed:

- `crates/model/src/arch/mod.rs` — `Arch::{Qwen35moe, Qwen3next, Qwen4exp, Glm5next}` and their names
  (the `Arch` enum today: `Deepseek2, Deepseek41, Qwen3moe`, `mod.rs:27-32`).
- `crates/gpu/src/lib.rs` — `mod linear` and the arch module line.
- `crates/gpu-gates/src/bind.rs` — an engine arm for the new body with §3.2's `keepable`.
- `crates/gguf/src/quant.rs` / `placement.rs::CardFormat::of` — only for the formats §5 names.
- Not changed: `crates/gpu/src/model.rs` (with the device-scalar slot index), `crates/serve/*`,
  `placement.rs`'s arithmetic.

## 4. The kernels, and their proof

Proof classes are AGENTS.md's table ("Derive first"). Every kernel here is new arithmetic, so the class is
"float sum order or precision": the error model and a runtime gate, a host simulation of *our own*
rounding rule for bit identity (the `exact_ref --act ours` idea), and a band against the reference dump.

| kernel | reference (file:line, launch shape) | reference sum order | our gate |
|---|---|---|---|
| short conv + SiLU | mainline `ssm_conv_f32` (`ssm-conv.cu:5-58 @ LCF`): block = channels split, one thread per channel, token loop inside; `sumf = 0; sumf += x·w` for j = 0..3, then `+ b`, then SiLU (`:40-56`). IK fuses the state in/out (`ssm_conv_single_seq_f32_nc4`, `ssm-conv.cu:67-126 @ IK`): `x0·c0 + x1·c1 + x2·c2 + x3·c3` as one expression (`:117`), SiLU a separate op (`llama-delta-net.cpp:340 @ IK`) | a four-term chain from the oldest input; with FMA contraction both likely produce the same fma chain [derived, unverified — §7 Q2] | host rule bit-identical; `conv_output_raw` tap equal to IK within the crate band |
| L2-norm q, k + β/g prep | mainline: `scale(rms_norm(x, eps/n), 1/√n)` (`src/models/models.h:46-50 @ LCF`) = x·rsqrt(Σx² + eps); IK `l2_norm_f32`: x·rsqrt(max(Σx², eps²)) (`ggml-cuda/norm.cu:213 @ IK`); FLA x / √(Σx² + eps) (`fla/modules/l2norm.py:43 @ FLA`). β sigmoid, g softplus·ssm_a (GDN) or lb·sigmoid (KDA) are elementwise ops | **three eps conventions**; at eps 1e-6 and Σx² ≈ 10 the `+eps` form moves a value by ~1e-7 relative, above half an f32 ulp — the trees do not agree to the bit here [derived] | host rule; band against IK `q_conv_normed`/`k_conv_normed`, `beta`, `gate` (GDN) / `log_decay` (KDA) |
| recurrent delta step (decode and prefill) | mainline `gated_delta_net_cuda<S_v, KDA, keep_rs>` (`gated_delta_net.cu:4-167 @ LCF`): grid (H, n_seqs, S_v/4), block (32, 4) (`:181-184`); a warp owns one value column, a lane holds 4 key rows in registers; Sᵀk as 4 serial FMAs per lane then `warp_reduce_sum` (`:89-93`); S updated, then Sᵀq the same way (`:101-107`); scale applied to the output (`:110`). State stored `[v][k]`, k contiguous (`:54`). IK `delta_net_recurrent_f32<128, 256|128>` (`delta-net.cu:30-183 @ IK`): 256 threads for ≤ 8 tokens, else 128 (`:216-234`); q pre-scaled (`:120`); output `decay·(Sᵀq) + u·(k·q)` from the **old** state (`:150-152`); state **clamped to ±1e6** (`:158`), g clamped ≤ 50 (`:128`); state stored `[k][v]`, v contiguous (`:108`) | different orders and different algebra (o from S_new vs from S_old + u·(k·q)); IK's clamps are absent from FLA and mainline | host rule bit-identical (our order); state band against an f64 simulation per depth (errors compound over positions; the decay is contractive, so a band per depth 6/1024/4096 is the right shape); `new_state` / `attn_output` band against IK; the `gate-gpu-e2e` token-count pin form for the model |
| chunked delta (later) | none on CUDA in either tree (§0 finding 2); FLA's Triton `chunk.py` and mainline's op graph (`delta-net-base.cpp:16-287`) | — | banded against our own recurrent kernel on the same inputs, state at every chunk end |
| gated output norm | mainline `build_norm_gated`: `rms_norm(o) · w`, then `· SiLU(z)` (`qwen35moe.cpp:267-276`) / `· sigmoid(z)` (`qwen4exp.cpp:459-468`); IK `build_gated_output`: RMSNorm then `ggml_fused_mul_unary` (SiLU) or explicit sigmoid (`llama-delta-net.cpp:408-430 @ IK`) | the RMS reduction order of each tree's norm kernel | host rule; `attn_out_norm` band |

Choice of oracle: IK. Our harnesses link it (`tools/ref/models/qwen3moe.sh`: "`IK` … the installed dump_ref
links against it"), and it is the only tree with `glm5next`. Its ±1e6 clamp is outside FLA's and mainline's
rule; §7 Q1 prices whether it ever fires.

### 4.1 What `dump_ref` needs beyond the V4.1 taps

`dump_ref` asks for every node through `cb_eval` (`tools/ref/dump_ref.cpp:362-386, 616-617`), so the
linear layers' intermediates are already named ggml tensors. IK's names on this path
(`llama-delta-net.cpp @ IK`): `qkv_mixed`/`linear_attn_qkv_mixed` (`:159-165`), `z`, `beta`, `alpha`,
`a_softplus`, `gate` (`:273-283`), KDA `decay_raw`, `log_decay` (`llama-kda.cpp:49, 63`), `conv_states`,
`state_predelta` (`:332-333`), `conv_output_raw`, `conv_output_silu` (`:337-341`), `q_conv`, `k_conv`,
`v_conv`, `q_conv_normed`, `k_conv_normed` (`:360-376`), `q_in` … `state_in`, `delta_net_fused_raw`,
`output_tokens`, `new_state` (`:106-148`), `new_conv_states_cont` (`:388`), `ssm_state_cpy`,
`conv_state_cpy` (`:398-402`), `attn_rms_norm`, `attn_out_norm`, `final_output`, `linear_attn_out`
(`:415-427`), `ssm_output` (`:623`).

Three things differ from the V4.1 set:

1. **The state is a tap only when the state copy is not fused.** IK's CUDA backend fuses the
   `new_state → cache` copy into the op and then writes the state straight into the cache slot instead of
   the op's output tail (`ggml-cuda.cu:4234-4250 @ IK`; the scan is `ggml_delta_net_find_state_cpy`,
   `ggml.c:24264-24268`, which "stops at the first real node"). Under the every-node schedule each node is
   computed alone (`dump_ref.cpp:362-363`), so the scan sees no following copy and the tail is written
   [derived from reading — §7 Q3 verifies by checking `new_state` at step t against `state_in` at t+1].
2. **Layout.** IK's state is `[k][v]` with v contiguous, mainline's `[v][k]` (the kernel rows above). A
   gate that compares against a mainline dump transposes.
3. **Size.** A decode-step set carries `state_in` and `new_state` for every linear layer: 30 × 2 × 2 MiB
   = 126 MB for Qwen3.6 [derived]. The profile's `ref_step_variant` (the `step4` form of
   `tools/ref/models/qwen3moe.sh`) is the right shape: a quiet prefill, then one dumped step, so
   `state_in` is the prefill's final state and the step checks the prefill path too.

A `qwen35moe` profile needs, beyond the V4.1/Qwen3 fields: the model path (Q4_K_XL above), `REF_TOKENS`
from `llama-tokenize` on this file (pre `qwen35`, [header]), and a step variant with a prompt long enough
that the full layers' KV and the linear state both matter (≥ 64 tokens so a chunked path, if ever
enabled, runs a whole chunk).

## 5. Quantization of the linear layers in the files on the box

Today's card formats at base `340579f`: q3_K/q4_K/q6_K, q5_0, q5_1, q8_0, F32, BF16→F32; **F16, Q5_K,
MXFP4 and every IQ type are refused** (`crates/model/src/placement.rs:202-215`). (The spec's "MXFP4
today" does not hold at this base; the `dsmx` round builds an MXFP4 `_sel` in its own track.) Routed `_sel`
on the card exists for q3_K, q4_K, q5_0 on main; `q6k_sel.rs` exists only untracked in the `qwen3`
worktree (`git status` there: `?? crates/gpu/src/q6k_sel.rs`).

| file | linear-layer tensors | full-attention | experts | named gaps |
|---|---|---|---|---|
| Qwen3.6 Q4_K_XL | `attn_qkv`, `attn_gate`, `ssm_out` Q8_0 ×30; `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_a`, `ssm_dt`, `ssm_norm` F32 | `attn_q/k/v/output` Q8_0 | gate/up Q4_K ×39 + Q5_K ×1; down **Q5_K ×36**, Q6_K ×4; shexp Q8_0; router F32 | **Q5_K** card format + routed `_sel` (37 stacks); Q6_K `_sel` (in flight in `qwen3`) |
| Qwen3.6 Q6_K | same Q8_0 / F32 | Q8_0 | gate/up Q6_K ×40; down Q6_K ×37, **Q8_0 ×3** | Q8_0 routed `_sel`; the file (29.3 GB) does not fit a 24 GB card |
| Qwen3-Coder-Next IQ4_XS | `attn_qkv`, `attn_gate`, `ssm_ba` **IQ4_XS**; `ssm_out` **Q5_K**; conv/a/dt/norm F32 | q/k/o IQ4_XS, v Q5_K | gate/up/down **IQ4_XS**; shexp gate/up **MXFP4**, down Q6_K | IQ4_XS on every projection (card and host dots), MXFP4, Q5_K; 42.7 GB — A6000 or host tier |
| Qwen3.8-Flash-Next Q4_K_XL (shard 2 read) | `attn_qkv`, `attn_gate`, `ssm_out` Q8_0; `ssm_alpha/beta` F32 | Q8_0; indexer `k_proj`/`q_proj` BF16 | gate/up Q4_K (1 Q5_K), down **Q5_1** ×10 / Q8_0 ×2 | Q5_1 routed `_sel` (format exists, no `_sel`); PLE table `per_layer_token_embd` **IQ4_NL** [160, 320,001,536]; host tier (111 GB) |
| GLM-5.3 Q2_K_XL (shard 2 read; 3 `.part`, 4 absent) | `attn_q` Q5_K, `attn_k`/`attn_v` Q6_K, `attn_output` Q5_K (one Q6_K); `ssm_f_a/f_b/g_a/g_b/beta` Q8_0; conv/a/dt/norm F32 | MLA `attn_q_a` Q5_K, `attn_q_b`, `attn_k_b`, `attn_v_b`, `attn_kv_a_mqa` Q8_0; indexer Q8_0 / F32 | gate/up **IQ2_XS** ×18 / IQ3_XXS ×1; down **IQ3_XXS** ×17 / IQ4_XS ×2; shexp Q5_K / Q6_K | Q5_K on card for the KDA q/o projections; IQ2_XS/IQ3_XXS/IQ4_XS host dots (c765166 added host *dequant* for them, bit-identical to ggml, not a dot); the file is a host-tier model |

The linear layers' own weights are Q8_0 in every file except Coder-Next's IQ4_XS and GLM's Q5_K/Q6_K
q/k/v; the small per-head tensors (`ssm_a`, `ssm_dt`, `ssm_norm`, conv) are F32 everywhere. The F32
`ssm_alpha`/`ssm_beta` [2048, 32] of Qwen3.6 are `DevWeight::F32` on the card (`crates/gpu/src/weights.rs:71-74`).

## 6. Order and size

Sizes as in arch-split.md: S half a day or less, M one round, L several rounds.

### (a) Qwen3.6-35B-A3B whole on one card — L in total

Fit [derived]: file 22.36 GB + GQA cache at ctx 32k (10 × 32,768 × 2,048 B = 671 MB) + state and
rollback slot (2 × 65.8 MB) + context 512 MiB + scratch 64 MiB + margin 1 GiB ≈ 24.8 GB against the
3090's 24 GiB = 25.77 GB — inside by ~1 GB; the token embedding (Q8_0, 540 MB) may stay on the host as a
row gather. The A6000 holds it with room.

| phase | content | size | depends on |
|---|---|---|---|
| a0 | pieces from the `qwen3` track, **untracked in its worktree, not on main**: `flash_gqa.rs` (its `HEAD` is 128 for Qwen3; Qwen3.6 needs 256, group 8), `rope_neox.rs`, `q6k_sel.rs`, the softmax router width variant 256/8 (`route_core`) | — | `qwen3` landing |
| a1 | `arch/qwen35moe/{hparams,names,roles,kv,place,host}.rs`, `Arch` variant, loud refusals (no `linear_*` key → error naming it) | S | — |
| a2 | `linear/conv.rs`: conv + SiLU + L2-norm + β/g prep, one launch per layer | S | a1 |
| a3 | `linear/delta.rs`: the recurrent kernel, `DECAY = Scalar`, `SLOTS = 2`, k-head map argument; host-rule simulation; gate vs IK taps | M | a2 |
| a4 | `linear/norm_gate.rs` (RMSNorm × SiLU/sigmoid) — may fuse into a3's store | XS | a3 |
| a5 | Q5_K card format + routed `_sel` (37 of 40 down stacks) | M | — (parallel) |
| a6 | full-attention extras: head 256 in the GQA flash, the `attn_q` query/gate split and sigmoid output gate, IMROPE (sections [11,11,10]; hypothesis: with text positions every section gets the same position, so it equals NEOX rope over 64 dims — §7 Q5 settles it), QK-norm, the gated shared expert | S | a0 |
| a7 | `arch/qwen35moe` `ChainBody`: state store, device slot index, graph capture, `decode_pair`/`rollback` | M | a3–a6 |
| a8 | oracle profile + sets, `l_out-N` per layer, e2e token pin | S | a7 |

Comparison lines: the reference is ik and mainline on the same Q4_K_XL file (§7 Q6 measures them on the
A6000; timed numbers are taken on the A6000 per AGENTS.md, the 3090 run is a separate functional
sitting). PR #83's figures (plan.md:650: Qwen3.6-35B-A3B, EXL3 3.0 bpw, SGLang + MTP, one 3090, 252 prose
/ 352 code tok/s) are a different file at 3 bpw with a draft: at this file our floor is 324 tok/s on the
3090 without a draft (§2.3), so 252 is beatable only if the step runs above 78 % of the floor, and 352 is
not reachable without a draft or a smaller file [derived]. The MTP layer is not in this GGUF
(`block_count 40`, no `nextn` key [header]); the n-gram lookup draft (`BLOOMERY_DRAFT=lookup`) is the draft
available.

### (b) Qwen3-Coder-Next and Qwen3.8-Flash-Next — M each on top of (a)

- **Coder-Next** (`qwen3next`): the `h / ratio` k-head map (finding 5), the fused `ssm_ba` split per k-head
  group (`qwen3next.cpp:418-455`), 512/10 experts, 48 layers, no IMROPE (`rope.dimension_count 64`, no
  sections key [header]; mainline `qwen3next` uses plain NEOX — not re-read here), pre `qwen2`. The file on
  the box is IQ4_XS/MXFP4 throughout and 42.7 GB: an A6000 or host-tier model, and it waits on IQ4_XS dots.
- **Qwen3.8-Flash-Next** (`qwen4exp`): H_v 48 (ratio 3), sigmoid output gate, the low-rank hc variant
  (`hc_attn_{down,up,inject,norm}`, low rank 320 [header]), the PLE table (IQ4_NL), the QSA indexer on the
  full layers, 512/10 experts; host tier. The linear layers are (a)'s kernels with two launch values changed.

### (c) GLM-5.3-Flash — L

- KDA on (a)'s program: `DECAY = Channel` (S on top of a3), the lower-bound gate prep with the low-rank
  f/g projections (S), the three convs as one 24,576-channel conv (XS).
- The 11 MLA layers: K = V = the 512-wide latent, 64 heads, no rope, no sinks (`build_glm5next.cpp:232-379
  @ IK`). This is the shape of V4.1's attention (`crates/gpu-deepseek41/src/attn.rs:1-8`: "K = V: one latent
  row per key is both its key and its value", walked by `mma_segment_walk` at the latent width, with a
  gathering source for indexer-selected rows at `:15-20`) minus the sink fold, plus `wk_b` absorption and
  a per-head `wv_b` after (V2-Lite's pattern). M.
- The k-pool indexer: a pool key is `softmax(gate + ape)`-weighted over 4 member keys, channel by channel
  (`build_glm5next.cpp:127-137 @ IK`) — V4.1's compressor pattern at ratio 4. IK recomputes every pool's key
  every step from the cached `[key; gate]` rows (`:109-137`); a pool's key depends only on its members, so
  it can be computed once when the pool completes, as our V4.1 compressor does. Scores `relu(pooled · q)`
  weighted by `indexer.proj` and summed over 32 heads, top 2048/4 = 512 pools plus the incomplete tail
  (`:150-215`). M.
- mHC with V4.1's hc kernels (4 streams, Sinkhorn 20, eps 1e-6 [header]; `hc_*_fn` Q8_0 [16384, 24]); the
  head collapses streams by an unweighted mean (`build_glm5next.cpp:529-544`). S.
- Router sigmoid 288/8, bias, scale 2.5 (a width variant of V4.1's rule). S.
- Formats: Q5_K card format (shared with a5), IQ host dots for the experts. The file is host tier.

## 7. Open questions, each with the box job that settles it

All jobs are reads of reference dumps or reference timings; each is under 30 minutes and those that run
ik take the CPU lease (`REF_DUMP_LEASE=1` in the profiles).

| # | question | job (≤ 30 min) |
|---|---|---|
| Q1 | Does IK's ±1e6 state clamp or g ≤ 50 clamp (`delta-net.cu:128, 158`) ever fire on real text? If yes, IK is not a faithful oracle for the rule | an IK dump of Qwen3.6 (5-token oracle ids, then a 1,024-token prompt step variant): max \|S\| over every `new_state` tap and max g over `gate`. ~10 min + page-in of 22 GB |
| Q2 | Are the three trees' conv sums bit-equal (fma contraction) and how far apart are their L2-norms? | the same prompt through IK and mainline `llama-eval-callback`-style dumps; max\|diff\| of `conv_output_raw` and `k_conv_normed` per layer. Also read the CUDA flags in each tree's `build/CMakeCache.txt` (`--use_fast_math`, `-fmad`) — seconds |
| Q3 | Is `new_state` a valid tap under dump_ref's every-node schedule (the fused-copy scan, §4.1)? | a 2-step decode dump: `new_state` of step t equals `state_in` of step t+1 bit for bit |
| Q4 | Is the recurrent kernel bandwidth-bound per token at decode (4.7 µs/layer on the 3090, 6.3 on the A6000 derived) and what is the per-token latency of its prefill loop? | `nsys` of ik on Qwen3.6 on the A6000 (`tools/ref/nsys-gpu.sh` form): the delta kernel's µs per launch at decode and for a 512-token prefill |
| Q5 | Does text-only IMROPE equal NEOX rope over 64 dims bit for bit? | from the Q1 dump: `Qcur`/`Kcur` after rope against the qwen3 track's `rope_neox` host rule at `n_rot 64`, θ 1e7 |
| Q6 | The reference lines: ik and mainline tok/s on Qwen3.6 Q4_K_XL at depths 6 / 1024 / 4096 on the A6000 | `tools/ref/depth-decode.sh`-form ik run (`llama-bench -gp d,96`), each depth one lease |
| Q7 | GLM-5.3 shards 3–4: the types of the layers not in shard 2 (layers ≥ 23) | a header read when the download ends — seconds, no load |
| Q8 | Qwen3.8-Flash-Next's PLE layer: config `ple_layer_ids [2]` [S18] vs GGUF `qwen4exp.ple.layers [1]` [header] — an index convention or an off-by-one in the file | read the `qwen4exp` converter (`conversion/qwen4exp.py @ LCF`) and IK's reader; seconds |

## Sources

HF repo commits are the `sha` the HF API returned on 2026-09-24; config URLs are
`https://huggingface.co/<repo>/resolve/main/config.json`.

| id | source | commit |
|---|---|---|
| S1 | https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/config.json (`linear_attn_config`, `layer_types`, `index_*`) | eb9eb208eb0d |
| S14 | https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/resolve/main/config.json | 9c7f2fbe8446 |
| S16 | https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/main/config.json (`text_config`) | 995ad96eacd9 |
| S18 | https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/config.json (`text_config`) | de4b8e4d43b9 |
| S23 | https://huggingface.co/Qwen/Qwen3-Coder-Next/resolve/main/config.json (fetched; the GGUF header is what this document cites) | a7fbcb5c0e12 |
| P1 | Yang, Kautz, Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule", arXiv 2412.06464, §3.1 eq. 10 — https://arxiv.org/html/2412.06464 | — |
| P2 | Kimi Team, "Kimi Linear: An Expressive, Efficient Attention Architecture", arXiv 2510.26692, eq. 1 (KDA recurrence), eq. 10 (output gate), chunk size 64 — https://arxiv.org/html/2510.26692 | — |
| FLA | https://github.com/fla-org/flash-linear-attention/blob/954438d1fcb5e1bb05c22f9908de9c5c2df74ae5/ — `fla/ops/gated_delta_rule/naive.py`, `fla/ops/kda/naive.py`, `fla/ops/kda/gate.py`, `fla/modules/l2norm.py` | 954438d1 |
| LCF | box `/home/user/llama.cpp-v41` — `src/models/{delta-net-base,qwen35moe,qwen3next,qwen4exp,kimi-linear}.cpp`, `src/models/models.h`, `src/llama-memory-recurrent.cpp`, `src/llama-model.cpp`, `ggml/src/ggml-cuda/{gated_delta_net,ssm-conv}.cu`, `conversion/qwen.py` | 5210c7c5ed61 |
| IK | box `/home/user/ik-idxkey` — `src/llama-delta-net.cpp`, `src/llama-kda.cpp`, `src/graphs/{build_glm5next,build_qwen35}.cpp`, `src/llama-hparams.cpp`, `src/llama-load-tensors.cpp`, `src/llama.cpp`, `ggml/src/ggml-cuda/{delta-net,kda,ssm-conv,norm}.cu`, `ggml/src/ggml-cuda.cu`, `ggml/src/ggml.c` | db517b690a8c |
| header | box GGUF headers, read 2026-09-24 by a pure-Python walk: `/models/Qwen3.6-35B-A3B/*.gguf` (2), `/models/Qwen3-Coder-Next/Qwen3-Coder-Next-IQ4_XS.gguf`, `/models/Qwen3.8-Flash-Next/…-00001` and `-00002-of-00004.gguf`, `/models/GLM-5.3-Flash-UD-Q2_K_XL/…-00001` and `-00002-of-00004.gguf` | file sizes as listed in §1.3 |
| rig-log | `~/repo/rig-log/log/2026-09-21.md:69` (the depth-decode per-key figures) | 143a261 |
