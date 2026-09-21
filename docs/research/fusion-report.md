# Computation fusion for the m = 1 GPU decode chain (bloomery, DeepSeek-V2-Lite Q3_K_M, RTX 3090)

Research round. Every external claim carries `repo path:line` (local checkout) or a URL.
Sources pinned:
- `~/repo/upstream/v41-ports/mine` = fork of ik_llama.cpp at **c10fbbcc** (2026-09-15), remote `upstream github.com/ikawrakow/ik_llama.cpp`. It is a fork with V4.1 ports, not stock ik — I flag where a file looks fork-only.
- mainline llama.cpp `ggml/src/ggml-cuda/ggml-cuda.cu`, fetched from `raw.githubusercontent.com/ggml-org/llama.cpp/master` on 2026-09-21 (5785 lines; line numbers are of that fetch, saved at `scratchpad/llamacpp-cuda.cu`).
- vLLM `csrc/layernorm_kernels.cu` and `csrc/layernorm_quant_kernels.cu` at **tag v0.10.0** (both files have since moved on `main`; the tag is the pinnable citation).
- FlashInfer `flashinfer/rope.py` @ main.
Marked **READ** = I read the source. **CLAIMS** = a README/blog/PR title says it and I did not read the kernel. **inferred** = my arithmetic or reasoning.

---

## 1. One-paragraph answer

At m = 1 every intermediate in this chain is 8–37 KB, which is **0.01–0.05 µs of memory traffic at 800 GB/s** — so no activation-level fusion in this model saves bandwidth, and the "bytes not moved" term in the spec's formula is numerically zero for all but one candidate (the V4.1 gathered-KV buffer, §5). What fusion actually buys here is the two terms our own P0b round measured and the spec's formula omits: the **0.76 µs node gap** per launch removed, and the **fixed body cost of a small kernel** — the few µs a launch spends with a handful of warps occupying an otherwise idle GPU — which the P0b dense-FFN spike measured at a 6 : 1 ratio over the gap (17 µs body vs 2.9 µs gap for four nodes removed, `docs/gpu-design.md` P0b 측정). Applied launch by launch to `enqueue_attn` (20 launches) and `enqueue_ffn_moe` (11), the worthwhile merges are **20 → 13 and 11 → 10**, worth an estimated **≈ 17 µs per MoE layer** against a post-shape-fix layer of ≈ 154 µs (inferred) — roughly **−11 % of the step**, not a step change. The single best candidate is the MLA key path (`rope(kv_a) + rms_norm(kv_a_norm) + gather(kvr) + kv_append` → one kernel, ≈ 4.7 µs/layer, bit-identical by construction), which is also the one every production engine has already fused: mainline llama.cpp fuses `RMS_NORM+MUL+ROPE+VIEW+SET_ROWS` in a single dispatch, FlashInfer ships `mla_rope_quantize_fp8` / `rope_quantize_fp8_append_paged_kv_cache` for exactly ckv = 512 / kpe = 64, and vLLM has open PRs titled "RoPE+KV cache and dual RMSNorm fusion for MLA". Horizontal gemv merges (`attn_q ∥ attn_kv_a_mqa`, shexp ∥ routed experts) buy the **gap only** — `attn_kv_a_mqa` already runs at its roofline (0.63 µs of weight bytes inside a 1.4 µs launch, so ~0 absorbable body) — and ik does not do them either: it shares the *quantized activation buffer* across consecutive `MUL_MAT`s and still issues one `mul_mat_vec_q` per weight (`ggml-cuda.cu:2571-2600`, READ). We already share that buffer. Finally, the honest framing for the round after this one: at ≈ 154 µs/layer against a ≈ 52 µs/layer bandwidth floor, **fusion is worth ~11 % and the gemv/flash efficiency gap is worth the other ~190 %** — fusion should be taken because it is cheap and mostly bit-identical, not because it closes the distance to ik.

---

## 2. Ranked candidates

Baseline used for every "absorbed body" number, stated once so the table can be audited:

| quantity | value | source |
|---|---:|---|
| pure node gap | 0.76 µs/launch | `docs/gpu-design.md` 노드 갭 표 (700-node `touch`) |
| grid barrier (cooperative launch) | 2.4 µs | same doc, `probe::two_phase` |
| block-0 per-op sum today | 216 µs / 24 launches | spec |
| the 16 unnamed launches | 22.9 µs total → **1.43 µs each = 0.76 gap + 0.67 body** | inferred (216 − Σ named 193.1) |
| post-shape-fix assumption | flash 70.8→9, attn_norm 21.4→3, ffn_norm_quant 16.5→3, kv norm 6.0→2 | spec ("assume single-digit"); **block-0 becomes ≈ 118 µs** (inferred) |
| activation bytes at m=1 | hidden 8 KB → 0.010 µs; `f_rows` 36.9 KB → 0.046 µs; `kqvc` 32.8 KB → 0.041 µs (@800 GB/s) | inferred |

FP-order classes: **(A) address-only** — the fused kernel writes the same values to different addresses, or skips values nobody reads → bit-identical by construction. **(B) same-core concatenation** — two launches' bodies placed in one launch, same cores, same fixed reduction tree → bit-identical (the P0b contract holds, `docs/gpu-design.md` P0b). **(C) new reduction partition** → band gate only.

| # | candidate | launches merged | µs/layer (derived) | FP class | size | precedent |
|---:|---|---|---:|:--:|---|---|
| **1** | **MLA key path**: `rope(kv_a)` + `rms_norm(kv_a_norm)` + `gather(kvr)` + `kv_append` → 1 | 4 → 1 (attn #5,#6,#7,#9a) | **≈ 4.7** = 3×0.76 gap + ~2.4 body | A + B\* | ~120 lines, one new kernel | llama.cpp `ggml-cuda.cu:4135`; FlashInfer `rope.py:1286,1554`; vLLM PR #41099 — **but ik's own equivalent is disabled, §6** |
| **2** | **kqvc quantize + wv_b, lo/hi → single m=16** | 4 → 2 (attn #10a-d) | **≈ 3.0** = 2×0.76 + ~1.5 body | A + B | ~10 lines *if* m=16 is legal (see §3) | ik one `_id` launch covers all experts, `ggml-cuda.cu:3138-3150` |
| **3** | **q rope written straight into `f_rows`**: `rope(q)` + `gather(f_rope_lo)` + `gather(f_rope_hi)` → 1 | 3 → 1 (attn #4a,#8a,#8b) | **≈ 2.4** = 2×0.76 + ~1.3 body (+ rope does 3× less work) | A | ~80 lines (per-head dst stride) | llama.cpp rope+set_rows `ggml-cuda.cu:3256` |
| **4** | **`attn_norm` + `quantize(act_q)` → existing `norm_quant`** (+ optional f32 out) | 2 → 1 (attn #1,#2) | **≈ 2.2** = 0.76 + ~1.4 body | B | ~40 lines, kernel exists | our `fused.rs:354`; vLLM `layernorm_quant_kernels.cu:25` |
| **5** | **MoE `ffn_norm` + `quantize` → `norm_quant` with an f32 side-output** (router needs f32) | 2 → 1 (moe #1,#2) | **≈ 2.2** = 0.76 + ~1.4 body | B | ~40 lines (same kernel as #4) | vLLM `layernorm_quant_kernels.cu:66` |
| **6** | **`attn_output` + `add(attn_resid)` → `q4k_gemv_add`** (mirror of `down_add_q5_1`) | 2 → 1 (attn #11a,#11c) | **≈ 1.4** = 0.76 + ~0.7 body | B | ~60 lines, reuses q4k core | our `fused.rs:490`; ik `mul_mat_vec_q_biased`, `ggml-cuda.cu:2547` |
| **7** | **shexp `gate_up_swiglu` ∥ routed `expert_gate_up_swiglu`** (same `act_ffn`) | 2 → 1 (moe #5,#8) | **≈ 1.0–1.5** = 0.76 + small body | B | ~90 lines (row-map for the 9856 slots) | none exact; ik keeps them separate |
| 8 | shexp `quantize_q8_1` ∥ expert `quantize_q8` (32-value) | 2 → 1 (moe #6,#9) | ≈ 1.4 | A | ugly (two block geometries in one kernel) | none |
| 9 | shexp `down` (Q4_K) ∥ expert `down_sel` (Q5_0) | 2 → 1 (moe #7,#10) | ≈ 1.2 | B | ~110 lines, two quant bodies | none |
| 10 | whole-step CUDA graph (27 layers + head in one capture) | 27 replays → 1 | **unmeasured** — removes 26 `cuGraphLaunch` host calls/step | none (no kernel change) | write the chain first (`model.rs:1153` is an error stub) | ik replays graphs and measured +10.6 % over graphs-off (`docs/gpu-design.md` 미정) |
| ✗ | `attn_q ∥ attn_kv_a_mqa` horizontal gemv | 2 → 1 | **0.76 only** — `kv_a_mqa` is already at roofline | B | ~70 lines | **ik does NOT do this** (`ggml-cuda.cu:2571`) |
| ✗ | router gemv + `router_topk` → 1 | 2 → 1 | ≈ 0.8, but forces a single-block 512 KB gemv | C | medium | ik/mainline fuse the *post-gemv* part only, which we already have |
| ✗ | `quantize(act_ao)` into `wv_b`; `quantize_q8` into `gate_up_swiglu` | — | **impossible without a grid barrier** (2.4 µs > 3 gaps) | — | — | our own P0b/MoE rounds found the same boundary |
| ✗ | `add(attn_resid)` + `ffn_norm_quant` (vLLM `fused_add_rms_norm`) | 2 → 1 | ≈ 1.4 — **mutually exclusive with #6**, which is simpler | B | small | vLLM `layernorm_quant_kernels.cu:66` |

\* class B for the norm **only if** the fused kernel calls `elem.rs`'s `rms_scale` core with the identical warp/lane reduction partition — a fresh kernel with its own geometry is class C (band gate). State it in the spec; `norm_quant` got bit-identity because it was designed for it, not for free.

**Totals** (inferred, post-shape-fix): attention half **20 → 13 launches, ≈ 13.7 µs saved**; MoE half **11 → 10 (taking #5 and #7 only), ≈ 3.6 µs**; dense FFN half is already 4 launches and has nothing left.

**Schedule against the per-head wrapper round, not against this total.** `docs/gpu-design.md` P8a says a per-head-wrapper round already owns the 9-node pair-table gather and the m = 8 column geometry. If it lands first, **#2 and #3 (≈ 5.4 µs) are already taken** and the remaining independent savings on the attention half are **#1 + #4 + #6 ≈ 8.3 µs/layer** (20 → 17 launches). The "20 → 13" row counts #2 and #3 in full and is therefore an upper bound.

Per MoE layer ≈ **17 µs of ≈ 154 µs** (attn ≈ 96 + MoE ≈ 58, inferred from `docs/gpu-design.md` P8a 264 µs minus the shape-fix deltas and the 84.5 µs dense FFN half). Over 27 layers ≈ **0.45 ms of a ≈ 4.1 ms step → −11 %**, against a 1.42 ms bandwidth floor.

---

## 3. Per-candidate detail

### #1 — MLA key path, 4 launches → 1

Our chain today (`crates/gpu/src/model.rs:2129-2195`):

```
rope(kv_a)              kv_a → kv_s          (rotates the last 64-value column)      model.rs:2129
rms_norm(kv_a_norm)     kv_a → kv_s[0..512]  (overwrites the head; rope tail stays)  model.rs:2141
gather(kvr)             kv_s → kvr           ([k_rope | kv_compressed], CONCAT order) model.rs:2152
kv_append(pos)          kvr → kv_l[pos]      (F32 → F16)                              model.rs:2180
```

Both the rope and the norm read the same 576-value `kv_a` and write disjoint spans of `kv_s` — they are *horizontally* independent, and the gather is a pure permutation, and the append is a cast. One kernel: one CUDA block reads `kv_a`, reduces the 512 latent values for the rms scale (one fixed tree, same `elem.rs` core → class B), rotates the 64-value rope tail (elementwise → class A), and writes the row **in kvr order** as F16 into `kv_l[pos_buf]` and as f32 into `kvr` and `kv_s` for the taps (`model.rs:1317-1325` reads both; 2 × 2.3 KB = 0.006 µs, free).

Arithmetic: 3 gaps = 2.28 µs; absorbed bodies = kv norm post-fix ~2.0 minus the reduction the fused kernel still does (call it ~1.2 net) + gather body 0.67 + append body 0.67 ≈ 2.4 µs. **≈ 4.7 µs/layer × 27 = 0.13 ms/step.** Bytes: 576 f32 read once either way; the eliminated intermediate round trips are 2.3 KB each = 0.006 µs. Zero.

FP: class A for rope/gather/append; class B for the norm **provided the fused kernel reuses `elem.rs`'s `rms_scale` core with the identical warp reduction partition**. That is an implementation constraint the spec must state, not a property that comes for free — `fused::norm_quant` is bit-identical because it was built to call the same cores with the same fixed tree (`fused.rs:348-353`). A fused kernel that re-partitions the 512-value reduction across a different lane layout is class C and needs a band gate instead.

Precedent, all READ except where noted:
- mainline llama.cpp fuses exactly this shape: `ggml_cuda_can_fuse` accepts `{ RMS_NORM, MUL, ROPE, VIEW, SET_ROWS }` (`llamacpp-cuda.cu:3227,3229` and the dispatch at `:4135`), and the shorter `{ROPE, VIEW, SET_ROWS}` (`:3256`, dispatch `:3544` → `ggml_cuda_op_rope_fused`). `SET_ROWS` is the KV-cache write; this is norm + rope + cache-append in one kernel.
- ik (fork c10fbbcc) fuses two ropes into one launch — `ggml_cuda_op_rope_rope` (`ggml/src/ggml-cuda/rope.cu:1622`, kernels `rope_rope_neox` at `:1202`, `rope_rope_norm` `:1261`), dispatched at `ggml/src/ggml-cuda.cu:4121-4131`. It also has `ggml_cuda_op_fused_rms_rope_fast` (`ggml-cuda.cu:3968`) — **but that branch is disabled** (`if (false && fusion …)`), which is itself evidence the shape is tricky, not that it is wrong.
- FlashInfer ships `mla_rope_quantize_fp8` (`flashinfer/rope.py:1286`) and `rope_quantize_fp8_append_paged_kv_cache` (`:1554`) — rope + quantize + paged-KV append in one call. Docs **CLAIM** MLA support at ckv = 512, kpe = 64 (https://docs.flashinfer.ai/generated/flashinfer.page.append_paged_mla_kv_cache.html) — our exact `latent`/`rope_dims`.
- vLLM PRs #41099 and #41101, titles (**CLAIMS**, not read): "[ROCm] Add (unified) AITER RoPE+KV cache and dual RMSNorm fusion for MLA" (https://github.com/vllm-project/vllm/pull/41099).

### #2 — kqvc quantize + wv_b, lo/hi → single m = 16

Today (`model.rs:2204-2232`): `quantize_q8_1(kqvc)` into `act_kv_lo` (`Q8Act::with_k(stream, 8, latent)`, `model.rs:1923`), then `quantize_q8_1_at(kqvc, 8*512)` into `act_kv_hi`, then two `enqueue_q3k_gemv_heads` with `head_base` 0 and 8.

I could find **no constraint in the host API that forces the split**. `enqueue_q3k_gemv_heads` requires only `heads == act.m()` and `(head_base + heads) * row_stride_per_head <= w.rows()` (`model.rs:390-440`, READ); the grid is `n_rows.div_ceil(8)` blocks of 256 threads, which at 16 heads × 128 rows = 2048 rows is 256 blocks — nothing near a limit. `enqueue_quantize_q8_1_at` takes an arbitrary base offset, so `x0 = 0, m = 16` is the same bytes as the two halves concatenated (its own doc comment says the quantized bytes equal a copy-then-quantize, `lib.rs:1168-1175`). So this looks like **a 10-line change**: one `Q8Act::with_k(stream, 16, latent)`, one quantize, one gemv with `head_base = 0, heads = 16`.

Caveat I could not close: `docs/gpu-design.md` P8a records the current attention geometry as "헤드별 m=8 열 기하(가중치 2배)" and says a per-head wrapper round will replace it. If that "weights 2×" refers to this pair, the author has a reason I did not find in the code. **Ask before spending the round** — and if the per-head wrapper round lands first, this candidate disappears into it.

Arithmetic: 2 gaps = 1.52 + two small bodies ~1.5 ≈ **3.0 µs/layer**. Weight bytes unchanged: wv_b reads 16 heads × 128 rows × 220 B (K = 512 → 2 super-blocks × 110 B) = 450 KB = 0.56 µs at 800 GB/s, once either way. FP class A (quantization is per 128-value column, identical per column) + B.

### #3 — q rope written straight into `f_rows`

Today: `rope(q)` writes `q_rope_all` over **every 64-value column** of q — `s.dims.q_cols = q_rows / rope` = 3072/64 = 48 columns — and the code's own comment says "the rotated neighbours are never read" (`model.rs:2115-2118`). Then two gathers copy the 16 useful per-head rope slices into `f_rows` (`model.rs:2174-2177`). `q_rope_all` has no other consumer (grep: written at `model.rs:2126`, read only by the two gathers; the taps build `q_rope` from `f_rows`, `model.rs:1311-1315`).

So: a rope variant that takes a per-head source offset (`h*kq_head + nope`) and a per-head destination base (`h*kv_width`) writes the 16 useful columns directly into `f_rows`, and **does one third of the work** it does today. 3 → 1.

Arithmetic: 2 gaps = 1.52 + two gather bodies 2 × 0.67 = 1.34 ≈ **2.9 µs**, less whatever the rope kernel itself keeps; call it **≈ 2.4 µs/layer**. Bytes: `q_rope_all` 3072 f32 = 12 KB never written or re-read = 0.03 µs. Zero, as always at m = 1.

FP: **class A** — the same rotations written elsewhere, plus values nobody read are no longer computed. Bit-identical by construction.

**Overlap**: `docs/gpu-design.md` P8a says the pair-table gather (9 nodes) and the m = 8 column geometry are "헤드별 래퍼 라운드가 지운다" — a per-head-wrapper round already owns this. Do not double-count; this candidate is a description of what that round should do, not a second round.

### #4 / #5 — norm + quantize

`fused::enqueue_norm_quant` already exists, is used by the dense FFN half, and is **bit-identical to `rms_norm` followed by `enqueue_quantize_q8_1`** by its own contract (`crates/gpu/src/fused.rs:348-353`, READ). The attention half does not use it (`model.rs:2098-2110`), and the MoE half deliberately does not (`model.rs:2306-2313`: "the router eats the f32 normed vector and the experts eat its q8_1 form, so both must exist").

Both are closed by one change: give `norm_quant` an **optional f32 side-output**. Cost: one extra 8 KB store = 0.010 µs. That also restores `s.normed` for the taps (`model.rs:1320,1362` read it — without the side-output, fusing the attention norm breaks the `attn_norm` tap against `ref_cuda`'s `FUSED_RMS_NORM`, which is a gate, not a nicety).

Arithmetic per site: gap 0.76 + the quantize body (~1.4 µs today as one of the 16 unnamed launches, essentially all fixed cost — 2048 f32 in, 2 KB out) ≈ **2.2 µs**. #4 applies to all 27 layers, #5 to the 26 routed ones.

FP class B. Precedent: vLLM's `rms_norm_static_fp8_quant_kernel` (`csrc/layernorm_quant_kernels.cu:25` @ v0.10.0, READ) is norm + quantize in one; its `fused_add_rms_norm_static_fp8_quant_kernel` (`:66`) is residual-add + norm + quantize in one, writing the new residual back (`csrc/layernorm_kernels.cu:121-124` @ v0.10.0 shows the plain `fused_add_rms_norm` writing `residual[...] = z`). ik fuses norm with *its gain multiply* (`fused_rms_norm_f32`, `ggml/src/ggml-cuda/norm.cu:309`, dispatch `ggml-cuda.cu:3963`) but **never fuses the activation quantizer into anything** — `quantize_row_q8_1_cuda` is always its own launch (`ggml-cuda.cu:2521`, `:3604`, `:3123`, READ). Our `norm_quant` is already ahead of ik here.

### #6 — attn_output + residual add

`gpu.enqueue_gemv_q4k(attn_output) ; gpu.elem().enqueue_add(attn_out, x → ffn_inp)` (`model.rs:2236-2244`). The exact shape of `fused::enqueue_down_add_q5_1` — `y[row] = w_row · act + resid[row]` (`fused.rs:487-492`, READ) — one type over. `attn_out` then never needs materializing (2048 f32 = 8 KB = 0.01 µs; zero, but it also removes a scratch buffer).

Arithmetic: gap 0.76 + the add's body ~0.67 ≈ **1.4 µs/layer**. FP class B: `dot` then `+resid` in the store is the same f32 addition of the same two values as the separate `add` performs — the P0b contract already covers this exact transformation for `down_add`.

Precedent: ik fuses a bias add into the gemv store, `ggml_cuda_op_mul_mat_vec_q_biased` (`ggml-cuda.cu:2547-2557`, and the 3-way QKV-bias form at `:2528-2546`, READ). mainline fuses `{MUL_MAT, ADD}` (`llamacpp-cuda.cu:4078`).

**Mutually exclusive with the vLLM-style `add(attn_resid) + ffn_norm_quant`** (the table's ✗ row): there is exactly one standalone add in the chain, and #6 consumes it. #6 is preferable because it does not add a residual input to the `norm_quant` kernel that the dense FFN, the MoE half and the shexp all share.

### #7 — shexp ∥ routed experts (horizontal)

`enqueue_expert_gate_up_swiglu` (6 experts × 1408 rows through `sel`) and `enqueue_gate_up_swiglu` (shexp, 1408 rows) read the **same** `act_ffn` and both run the Q3_K gate·up·swiglu core (`model.rs:2375-2385`, `:2403-2409`). One launch over 9856 row-slots with a slot→(weight, row) map (slot < 8448 → `sel` expert, else shexp) merges them.

Honest accounting: shexp gate+up is 2 × 1408 × 880 B = 2.48 MB = **3.1 µs of real weight traffic** at 800 GB/s, so there is little fixed body to absorb — the saving is the gap plus a fraction: **≈ 1.0–1.5 µs/layer**. Activation is read once either way (it is 2 KB, in L2 in both designs). FP class B.

No production precedent found: ik keeps the shared expert as its own `MUL_MAT`/`FUSED_UP_GATE` node (`ggml-cuda.cu:4047-4051`, READ). Take it for the gap, not because anyone else does.

### #10 — whole-step graph

Not blocked by CUDA: every per-replay quantity is already a device buffer, which is the stated reason one graph serves every position (`model.rs:1216-1222`, READ). Weight and KV pointers are frozen at capture and are all load-time resident (decision 4), so a 27-layer body captures the same way a 1-layer body does.

Three real items, none of them CUDA:
1. `GpuModel::step` is an **error stub** today (`model.rs:1153-1163`) — the stage-wide chain is unwritten. This candidate is therefore "capture the chain whole when you write it", not a separate project.
2. The layer boundary is a host round trip today: `replay_layer(l, x_in: &[f32], pos)` (`model.rs:1462`). Inside one graph it becomes `l_out → x` in device memory — a ping-pong of the two scratch buffers, or one 8 KB device copy node (0.01 µs).
3. `refresh_params` (host YaRN cos/sin + four `copy_from_host`) costs **18 µs/step** and is per *step*, not per layer — a whole-step graph does not touch it. It is 1.3 % of the 1.42 ms floor; see §7.

Expected saving: 26 fewer `cuGraphLaunch` host calls per step. **I have no number for `cuGraphLaunch` host cost and cannot measure it** (no box access this round; §6). The measurement to run: capture the same N total nodes as 27 graphs replayed in a loop vs 1 graph replayed once, same body, and report the difference — the existing node-gap probe harness (`gpu-spike`) already has the shape.

---

## 4. What other engines do that we should NOT copy

1. **Cooperative-launch / grid-barrier mega-kernels** (the "persistent kernel" and megakernel designs). Our own measurement kills it: a `grid::sync()` costs **2.4 µs**, three node gaps (`docs/gpu-design.md` 단계 간 순서의 수단). Any "N launches → 1 launch + N−1 barriers" is a net loss at these launch counts. Where a cross-block boundary is genuinely needed (the 32-value quantize after swiglu, `act_ao` after wv_b), the answer is the ExLlamaV3-style per-chunk completion counter, not a barrier — and that is a *second* question, after the four-node shape stands.
2. **Pool-allocating the quantized activation inside the step.** ik does `ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool(), …)` on every `mul_mat` (`ggml-cuda.cu:2519`, `:3601`, `:3123`, READ). That is incompatible with graph capture (`graph.rs:63-67` rejects allocation in a captured body) and with decision 4. Our scratch arena is the right call; do not import the pool.
3. **A graph-level pattern matcher.** ik's fusion is ~30 `if (fusion && cgraph->nodes[i+k]->op == …)` chains in one `switch` (`ggml-cuda.cu:3720-4200`), and mainline's is `ggml_cuda_can_fuse` + `ggml_can_fuse_subgraph` with memory-range overlap checks (`llamacpp-cuda.cu:3180-3275`; note the *overlap* guards at `ggml-cuda.cu:3998-4010` in ik, which exist because the allocator may place the second rms output on top of the first's source). Both are infrastructure for a **general** graph. We hand-write 31 launches for one model. Writing the fused kernel directly is strictly cheaper and the pattern-matcher risk (silent non-fusion, aliasing bugs) does not exist for us.
4. **Horizontally merging gemvs that already run at roofline.** `attn_kv_a_mqa` reads 576 × 880 B = 507 KB = 0.63 µs at 800 GB/s inside a ~1.4 µs launch (0.76 gap + 0.64 body) — there is essentially **no body to absorb**, so the merge with `attn_q` buys 0.76 µs for a two-weight-tensor kernel. ik does not do it either: `ggml_cuda_mul_mat_q` quantizes `src1` once and then walks forward over consecutive `MUL_MAT`s that share the same `src1`, reusing `src1_quantized` but issuing **one `mul_mat_vec_q` per weight** (`ggml-cuda.cu:2571-2600`, READ). The thing it is buying — one quantization instead of N — we already have (`model.rs:2108-2114` and the design's decision 2).
5. **Fusing the router gemv into top-k.** ik and mainline both fuse the *post-gemv* chain (softmax → argsort → view → get_rows → reshape → sum_rows → div) into one `topk_moe` kernel (ik `ggml-cuda.cu:4085-4113` → `topk-moe.cu:209`; mainline `llamacpp-cuda.cu:3470-3536`, both READ). Neither pulls the gemv in, because top-k needs all logits and the gemv rows live on different blocks. **We already have the fused piece** — `router::enqueue_router_topk` is softmax + top-6 + normalize in one launch (`model.rs:2364-2372`). There is nothing here to copy; pulling the 64 × 2048 F32 gemv into a single block to close the last boundary would trade 0.76 µs for a serialized 512 KB read.
6. **Marlin / Machete-style fused dequant-GEMM and DeepGEMM.** Both are tensor-core GEMM designs for M ≥ 16 (Marlin/Machete on `mma`, DeepGEMM on Hopper TMA/WGMMA). sm_86 has no WGMMA and at m = 1 there is no tensor-core path at all; our dequant is already fused into the gemv in the mmvq style. **inferred** — I did not read either source this round (§6).
7. **ik's bias fusion into the gemv** (`mul_mat_vec_q_biased`). Useful pattern, but DeepSeek-V2-Lite has no QKV/FFN biases — there is nothing to fuse. Note it only because #6 borrows its *shape* for the residual.

---

## 5. V4.1-Flash carry-over

Where fusion matters **more** on V4.1 than on V2-Lite, in order:

**(a) The gathered-KV buffer — the one place bytes are real.** V4.1 attention gathers ≤ 640 rows of the 512+64 latent per layer (`docs/research/v41-ops.md:59`: "최대 640행에 대한 것이라 깊이에 따라 늘지 않는다"). Materializing that gather is 640 × 576 × 2 B = **737 KB written and re-read per layer** ≈ 1.8 µs/layer at 800 GB/s, **≈ 74 µs/token over 40 layers** — the first activation-level fusion in this whole report whose bytes term is not zero. ik's fork does materialize it: `k_latent_gather_rows_indexed` / `k_latent_gather_rows_q8_0_indexed` (`ggml/src/ggml-cuda/latent_attn.cu:189,209`), then `k_latent_mask_softmax_indexed` (`:231`) and cuBLAS gemms (`sgemm_strided_batched`, `:348`), then `k_latent_copy_out_indexed` (`:277`). Fusing the gather **into** the flash kernel — the indexer's top-k ids as an indirection inside the key loop, never writing the packed buffer — is the V4.1-specific candidate. It is also the shape our `flash_latent` already has (it walks `kv_l` rows directly); the work is adding an id indirection, not restructuring.

**(b) Hyper-connections: RMSNorm + Sinkhorn + residual mixing in one kernel.** V4.1 carries 4 residual copies mixed by a doubly-stochastic `comb` from 20 Sinkhorn iterations (`v41-ops.md:47`), plus a 24-dim gemv per site (`:90`). Naively that is a dozen tiny launches per layer over 4 × 2048 f32 = 32 KB — pure fixed-body cost, exactly what P0b measured as the dominant term. **SGLang fuses precisely this**: a "fused `mhc_pre_big_fuse_tilelang` path that combines RMSNorm, Sinkhorn, and residual mixing in one kernel, with PDL enabled" (LMSYS, *DeepSeek-V4 on Day 0*, 2026-04-25, https://www.lmsys.org/blog/2026-04-25-deepseek-v4/ — **CLAIMS**, verbatim quote, no speedup given, targets SM90/SM100). This is the strongest copy-this signal in the report. Note their split-K variant (`mhc_pre_gemm_sqrsum_splitk_kernel`, same source) exists specifically "improving GPU utilization at small batch sizes" — our regime.

**(c) Engram lookup-add.** 48 rows/token × 272 B = 12.75 KiB (`docs/roofline.md`), added into the stream at layers 1 and 14 (`v41-ops.md:63`). Bytes are nothing; it is a gather + a 6144 → 25600 gemv + an add (`:68`). Fuse the add into its consumer and the gather into the gemv — launch-count only, same argument as everything in §3.

**(d) 384-expert routing.** The routing path is *shape*-different (top-6 of 384 instead of 64) but *fusion*-identical: our `router_topk` and `_sel` already collapse it, and `docs/roofline.md` shows routed experts are only 34.6 % of V4.1's per-token bytes with most of them host-resident. **Fusion matters less here, not more** — the host/GPU boundary and overlap own that half of the step (roofline: host term 22.6 ms of 56.5 ms), and no kernel merge touches it.

**(e) What carries over unchanged.** Candidate #1 is the same op for the same shape — V4.1 is MLA with kv_lora 512 + rope 64, which is why FlashInfer's `mla_rope_quantize_fp8` names exactly those dimensions. Candidates #4/#5 (norm+quant) apply per layer and scale from 27 to 40 sites. Candidate #10 (whole-step graph) gets *harder*, not easier: with host-resident experts the step is no longer one device graph, and the right unit becomes one graph per GPU-resident span between host boundaries.

---

## 6. What I could not verify

- **No box access this round** — every µs in §2/§3 is derived from the spec's block-0 profile and `docs/gpu-design.md`'s measured node gap and P0b/MoE deltas. Nothing here was measured. The derivations that would most change the ranking if wrong: the "0.67 µs absorbable body per small launch" figure (inferred by subtraction from the 216 µs sum), and the post-shape-fix assumption that flash drops 70.8 → 9.
- **`cuGraphLaunch` host cost** — unknown, so candidate #10 has no number. Measurement proposed in §3.
- **The lo/hi split rationale** (candidate #2). No constraint is visible in `enqueue_q3k_gemv_heads` / `enqueue_quantize_q8_1_at`; `docs/gpu-design.md` P8a's "헤드별 m=8 열 기하(가중치 2배)" may or may not refer to it. Ask the author before spending the round.
- **TensorRT-LLM and SGLang source** — not read. SGLang claims are from the LMSYS blog (§5b) and a PR title ("[CPU] Implement fused QK Norm and RoPE kernels", sgl-project/sglang#37748, https://github.com/sgl-project/sglang/pull/37748). TensorRT-LLM: only an issue title found ("Add qknorm + rope fuse kernel", NVIDIA/TensorRT-LLM#12716) — no source read, so I make no claim about it.
- **Marlin / Machete / DeepGEMM / FlashMLA source** — not read; §4.6 is inferred from their documented target shapes (tensor-core GEMM, M ≥ 16, Hopper TMA/WGMMA) and should be treated as a reasoned dismissal, not a verified one.
- **vLLM's `main`** — `csrc/layernorm_kernels.cu` and `csrc/layernorm_quant_kernels.cu` 404 on `main` (the files moved; the repo now has `csrc/quantization/`, `csrc/moe/`, …). I pinned **v0.10.0** instead. The fusion patterns are almost certainly still present under new paths, but I did not follow them, so the citation is to the tag.
- **vLLM PRs #41099 / #41101** — titles only, from search results. Not fetched, not read.
- **ik's disabled fusion branches.** `ggml_cuda_op_fused_rms_rope_fast` and `ggml_cuda_op_fused_add_rms_norm` are both guarded by `if (false && fusion …)` (`ggml-cuda.cu:3964`, `:3972`, `:3767`). Why they are off — correctness, aliasing, or a regression — is not in the code. Worth asking upstream before we build the same shape; it is the one thing in this report that could turn candidate #1 from "bit-identical by construction" into "there is a reason nobody ships this".
- **Whether this checkout's `latent_attn.cu`, `sinkhorn.cu`, `indexer_topk.cu`, `ds4_comp.cu`, `topk-moe.cu` are fork-only.** They are present at c10fbbcc in `midagedev/ik_llama.cpp`; I did not diff against `upstream/main`.

---

## 7. Improvement opportunities outside the asked scope (report only, nothing touched)

- `crates/gpu/src/model.rs:2119-2128` — `rope(q)` rotates all 48 columns of `q` while the code's own comment says only the 16 per-head rope slices are ever read. **3× wasted work** in that kernel, independent of whether candidate #3 is taken. Small.
- `crates/gpu/src/model.rs:1216-1232` — `refresh_params` does host YaRN cos/sin math plus four `copy_from_host` per step: **18 µs/step, 1.3 % of the 1.42 ms floor** and 100 % of the host time in the step. A device kernel reading `pos_buf` would make the step a single graph launch with no host math. Medium (~80 lines + the YaRN port).
- `crates/gpu/src/lib.rs:1188-1191` — "Prepared per call for now … P8's business": `prepare_*` runs its host-side contract validation on **every** enqueue. Free under graph replay, but it is ~31 redundant host calls per layer on the eager path that every gate and tap harness uses. Small (cache per shape).
- `crates/gpu/src/model.rs:1153-1163` — `GpuModel::step` is an error stub; the stage-wide chain and the lm_head are unwritten. This is the blocking item for *any* whole-step measurement, including candidate #10 and the ik tok/s comparison. Known and planned (P8b), noted because three things in this report depend on it. Large.
- `crates/gpu/src/model.rs:1462` — `replay_layer(l, x_in: &[f32], pos)` moves the layer boundary through the **host**. Fine for the gate harness, but it is the shape that must not survive into the assembled chain. Small.
- `crates/gpu/src/model.rs:2204-2232` — the kqvc/wv_b lo/hi split appears to cost two launches for no reason visible in the host API (candidate #2). Tiny, if the author confirms.

DONE-fusion
