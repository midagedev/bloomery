# Three ik_llama.cpp ports of DeepSeek-V4.1-Flash, read side by side

Investigation only. Nothing was modified in any source tree; no git write commands were run.
Read against `/Users/hckim/repo/bloomery/docs/research/v41-ops.md` and `v41-ops-report.md` — the
reference math is taken from there and not re-derived. Claims marked **inferred** are mine, not
the trees'.

## 0. Bases, aliases, citation convention

| alias | tree | HEAD | `main` |
|---|---|---|---|
| `mine` | `/Users/hckim/repo/upstream/v41-ports/mine` | `c10fbbcc` | `3bb386eb` |
| `skel` | `/Users/hckim/repo/upstream/v41-ports/skelectric` | `2154c9fc` | `3bb386eb` |
| `phyl` | `/Users/hckim/repo/upstream/v41-ports/phylliida` | `0b9a772a` | `3bb386eb` |

`git rev-parse 3bb386eb` resolves in all three (`3bb386eb68ffee0a5dc7db21da0735d594929eeb`), so the
spec's premise holds. But **`main..HEAD` is not the port's work**: each branch also carries ik
upstream commits merged after 3bb386eb (`#2441`–`#2448` in `mine`, up to `#2474` in `skel`), which
is why a naive `diff main...HEAD --stat` credits `clip.cpp +435`, `build_glm5next.cpp +550/+692`
and `unicode.cpp +154` to the V4.1 work. The real port boundaries, used throughout below:

- `mine`: `19dfb71b..HEAD` — 3 commits, 793 insertions over 13 files.
- `skel`: `dc310244..HEAD` — 16 commits, 5324 insertions over 51 files.
- `phyl`: `fa338794~1..HEAD` — 31 commits, but only three are V4.1 (`266d13dd` converter,
  `c7f9911b` runtime M1–M3, `7b69c599` seq_rm). The other 28 are a **GLM-5.3-Flash expert-cache
  project** that predates the V4.1 port (2026-08-28 … 2026-09-08 vs the V4.1 commits on 09-13).

Citations: `mine/src/llama.cpp:5412` for code, `mine@b63c33e1` for a commit message,
`phyl/docs/expert-cache/PHASE4-STATUS.md:14` for docs.

---

## 1. Engram serving

### mine
Row ids are computed host-side in `llama_set_engram_rows` (`mine/src/llama.cpp:5380-5418`) from the
compressed-id history recovered out of the KV cells (`llama_kv_prev_tokens`,
`mine/src/llama.cpp:5751`), with all hash constants read from GGUF metadata
(`model.engram_multipliers/primes/offsets/token_map/pad_id`, `mine/src/llama.cpp:5392-5395`). The
ids go into the graph as an I32 input and **the gather itself is a graph op**:
`ggml_get_rows(layer.engram_embd, rows)` (`mine/src/graphs/build_deepseek4.cpp:1005`), with the
table forced into the host `ctx_input` buffer so it is only ever touched through `get_rows`
(`mine/src/llama-load-tensors.cpp:3466-3468`).

Prefetch is the port's whole contribution here: for each row id, page-align the 272 B row and
`posix_madvise(..., POSIX_MADV_WILLNEED)` it, issued inside `llama_set_engram_rows` — i.e. before
the graph is launched, on the decode thread, one call per row (`mine/src/llama.cpp:5419-5434`).
Both engram sites are covered by the same loop over `eg` (`mine/src/llama.cpp:5752-5754`), so the
48 rows of a token go out together. Measured, quoted verbatim from `mine@b63c33e1`:

> DeepSeek-V4.1-Flash Q3_K_M, experts of six layers on two GPUs and the rest on the CPU, six new
> 200-token greedy prompts: 41-62 major faults per token and 13.6 tok/s before, 1-11 faults and
> 18.4 tok/s after; the same prompts with the rows already cached decode 18.8-19.0 either way.

No hot-row cache, no io_uring, no O_DIRECT, no pread: pure page-cache reliance over the GGUF mmap.

### skel
Constants do **not** come from the GGUF — they are precomputed by an out-of-tree
`engram_constants.py` into a sidecar binary `<model_stem>.engram-constants.bin` loaded at context
init; if it is missing, engram is disabled with a warning rather than a failure
(`skel/src/llama-engram.h:3-15`, `skel/src/llama-engram.cpp:202-208`). Validation is real: sidecar
layer ids must match the model's, and the prime partition must cover the table exactly
(`skel/src/llama-engram.cpp:246-252`).

Serving is a deliberate **two-pass host gather** (`skel/src/llama-engram.cpp:305-471`):

1. Pass 1 hashes every row id for every token of the ubatch with no table access at all, so
   nothing can fault while the ids are being produced (`:338-406`). History is a per-sequence
   3-entry tail (`llama_engram_tail`, `skel/src/llama-engram.h:65-69`), with image tokens marked
   `DEAD` so they both hash as pad and block later lookbacks (`:346-380`).
2. All row ranges are handed to the OS at once as `posix_madvise(WILLNEED)` — explicitly "so the
   page faults go out together instead of one per row on the thread the graph is waiting on"
   (`skel/src/llama-engram.cpp:408-424`).
3. Pass 2 dequantises Q8_0 rows on the host into an F32 scratch and does **one**
   `ggml_backend_tensor_set` per engram layer (`:426-453`). The graph's input is a plain
   `F32 [24*256, n_tokens]` tensor (`skel/src/llama-engram.h:83-84`,
   `skel/src/llama-engram.cpp:281`).

The gather requires a host buffer and says so in the error (`-ot ".*engram.*=CPU"`,
`skel/src/llama-engram.cpp:435-437`). `--defer-engram` keeps the tables on the file instead of
resident (`skel/common/common.cpp:2191, 3323`; `skel/src/llama-model-loader.cpp:745-747`). There is
a debug dump of ids and dequantised values for cross-checking (`LLAMA_ENGRAM_DUMP=1`, `:386-399`,
`:455-470`).

### phyl
Constants come from the GGUF, with an explicit rule that they must never be regenerated at runtime
because the hashing has to be bit-exact with training (`phyl/src/llama-engram.h:5-8`). The hash is
the same rolling XOR (`phyl/src/llama-engram.cpp:24-61`). The gather is host-side like skel's
(`llama_engram::gather_rows`, `phyl/src/llama-engram.cpp:63-80`), driven from
`dsv41_engram_set_inputs` (`phyl/src/llama-dsv41.cpp:1017-1047`), and asserts the table is in a
host buffer (`:1036`).

**There is no prefetch of any kind** — hash, then gather straight through `traits.to_float` on the
mmap. The one mmap-level change in the tree is an anonymous-THP mode that `pread`s the whole model
file into `MADV_HUGEPAGE` memory and drops the page cache behind itself
(`phyl/src/llama-mmap.cpp:300-325`), plus a `dontneed_fragment` guard so such a mapping is never
`MADV_DONTNEED`-ed (`:411-417`); that is a whole-file loading strategy, not an engram path, and it
makes the engram tables anonymous resident memory rather than file pages. `IK_CUDA_PINNED_IO`
(`phyl@03913b64`) is a general small-copy pinned-staging path in the CUDA backend and never names
engram — **inferred**: it would incidentally cover the engram input upload.

phyl's history is a per-position vector for the whole sequence rather than a rolling tail
(`phyl/src/llama-engram.h:22-23`), which buys session save/restore of the engram state
(`phyl/src/llama.cpp:12894-12903`, `13837-13854`) at the cost of being single-sequence only and
aborting on a position gap (`phyl/src/llama-dsv41.cpp:992-1007`).

### GPU side and bytes
skel and phyl upload the *dequantised* rows: 24 × 256 × 4 B = 24,576 B per token per site, 49,152
B/token for the two sites — 3.8× the 13,056 B that were read from disk (**derived**; disk figure
from `v41-ops.md`). `mine` uploads 24 I32 ids per site (96 B/token/site) and lets `ggml_get_rows`
run on the CPU backend over the host table, so the same F32 volume still crosses to the device if
`engram_wkv` is device-resident, but it crosses as a scheduler copy of a graph tensor rather than a
`tensor_set` (**inferred** from `mine/src/graphs/build_deepseek4.cpp:1005-1010` and the `ctx_split`
placement of `engram_wkv` at `mine/src/llama-load-tensors.cpp:3470`).

**What this means for our engine.** The prefetch window we planned (issue reads right after
sampling, one full step of slack) is strictly better than all three: mine prefetches inside the
same decode call, skel batches the advice but still inside the call, phyl not at all — and mine's
own numbers (41–62 → 1–11 major faults, 13.6 → 18.4 tok/s) show the fault count *is* the decode
bottleneck at this scale, so the remaining 0.4 tok/s gap to the fully-cached case is the prize for
going earlier. Take skel's hash-everything-first/touch-nothing structure: it guarantees the advice
covers every row before any of them blocks. Keep the dequantised rows on the host and upload F32
(48 KiB/token) only if the engram `wkv` matmul is device-side; our Q8_0-aware kernels could instead
consume the 13 KiB of packed rows directly and dequantise on the device, which no port does.

---

## 2. Routed-expert placement and the expert cache

### phyl — the only expert cache, and it is a GLM-5.3 project
Design and evidence live in `phyl/docs/expert-cache/` (1350 lines across five files). The target
throughout is **GLM-5.3-Flash 321B IQ3_XXS, 42 MoE layers, 288 experts, top-8**
(`PHASE4-DYNAMIC-EXPERT-CACHE-PLAN.md:9-11`, `PHASE4-STATUS.md:69-75`). No V4.1 number appears
anywhere in `phyl/docs` (grep for `V4.1|v41|deepseek41` over `docs/` returns nothing).

- **Cache unit**: one expert's three slices (up + gate + down) in one layer, ≈ 8.56 MB at GLM's
  IQ3_XXS (`PHASE4-M3C-ADMISSION.md:11-13`). The expert index is the last dim, so the load-time
  split is a contiguous memcpy with no dequant/requant (`PHASE4-DYNAMIC-EXPERT-CACHE-PLAN.md:120`).
- **Where they live**: VRAM, `H+1` per-layer slot tensors `ffn_{up,gate,down}_exps_hot`, slot `H`
  being a **trash slot** (`phyl/src/llama-build-context.cpp:1592-1597`). The trash slot exists
  because the fast-TG `mmvq-id` kernels have no negative-id check and a `-1` would be an OOB read
  (`PHASE4-DYNAMIC-EXPERT-CACHE-PLAN.md` masked-execution section).
- **How the graph addresses hit vs miss**: two-path masked execution, both paths always built,
  masks are *values* not shapes (`phyl/src/llama-build-context.cpp:1616-1701`). `hot_ids` = slot or
  trash, `hot_mask` = 1.0/0.0, `cold_ids` = expert id on miss and `-1` on hit (native CPU skip).
  `experts = experts_hot*mask + experts_cold` is bitwise-exact against the single path because
  exactly one side is non-zero per (j, token) (`:1684-1700`). Classification moved into the graph
  as a device kernel `ggml_exp_cache_classify` reading device-resident `remap`/`pending` tables
  (`phyl/ggml/src/ggml-cuda/exp-cache-classify.cu`, wired at
  `phyl/src/llama-build-context.cpp:1623-1659`) specifically so the TG graph stays CUDA-capturable
  and there is no per-layer host sync.
- **Eviction / admission**: LRU with a **2nd-miss-in-a-64-step-window** admission filter, a
  wall-time token bucket (`--expert-cache-promote-gbps`, doc default 8 GB/s, code default lowered
  to 2 by `phyl@85e46bcf`), in-flight cap 128, per-layer pending cap `H+1−k`, candidates sorted
  hottest-first (`PHASE4-M3C-FINDINGS.md:16-24`). Promotions are published only at TG step
  boundaries after the copy fence completes (`PHASE4-STATUS.md:44-58`).
- **What is copied on a miss**: *nothing*. "A miss is computed on CPU from the host-resident full
  expert tensors … Miss bytes never cross PCIe — so promotion cannot be 'retarget the miss's HtoD
  stream'; promotion bytes are always *additional* traffic" (`PHASE4-M3C-ADMISSION.md:62-67`).
- **Warm-up**: PP is read-only — no promotion or eviction while `n_tokens > 8`, because a PP ubatch
  touches ~all experts and would thrash LRU; TG warms its own cache in 100–200 tokens
  (`PHASE4-DYNAMIC-EXPERT-CACHE-PLAN.md`, "Read-only during PP").
- **Numbers** (GLM, 19.7k context, RTX 3090): hit 0.46 at step 100, converged 0.524; TG 7.86 base →
  8.11 unstaged → **9.98 t/s** staged; the TG ≥ 12 t/s gate still fails
  (`PHASE4-STATUS.md:13-15`). Traffic model `TG ≈ 9.01/(1−hit)` validated
  (`PHASE4-STATUS.md:74`). PCIe: 26.2 GB/s pinned HtoD on a dedicated copy stream, 26.15 GB/s with
  the compute stream saturated, 14.4 GB/s pageable (`PHASE4-STATUS.md:97-101`).

V4.1 reaches this code because `build_deepseek41` calls the shared `llm_build_moe_ffn`
(`phyl/src/graphs/build_deepseek41.cpp:1393`), but the eligibility test is narrow and explicitly
shaped for glm5next — separate up/gate tensors, fused up+gate, no biases, no `fused_mmad` tail
(`phyl/src/llama-build-context.cpp:1599-1614`). **Inferred**: V4.1 satisfies it, but nothing in the
tree records it having been run.

### mine and skel
Neither has an expert cache. `grep` for `expert_cache` in `mine/src` and `skel/src` returns
nothing. Placement is stock ik `-ot` / `-ncmoe`. skel adds `--defer-experts` (defer expert mmap
residency to cut load time, `skel/common/common.cpp:2182, 3321`;
`skel/src/llama-model-loader.cpp:662`). phyl also carries `--prefetch-experts`,
`--prefetch-experts-threads` and `--prefetch-experts-ahead N` for batch-graph lookahead streaming
of mmapped experts into the page cache (`phyl/common/common.cpp:2211-2226, 3352-3357`), plus a
`GGML_MOE_PREFETCH_READAHEAD` prefault mode (`phyl@a54a12e7`).

**What this means for our engine.** The design worth taking is the *shape*, not the numbers: two
always-built paths with value-only masks, a trash slot instead of a negative id, and admission on
the second miss. The two hard lessons are recorded as failures, not theory — pageable promotion
copies cost ~27 ms/step through driver staging-lock contention with critical-path input copies
(`PHASE4-STATUS.md:16-21`), so a promotion path that is not pinned-staged is worse than no cache;
and a host classify per layer per step is expensive enough that they moved it into a device kernel
to keep graph capture. For our box the arithmetic transfers with different constants: V4.1 is
top-6 of 384 with much smaller per-expert slices than GLM's 8.56 MB, so at the same hit rate the
promotion traffic per token is lower and a 3090-sized cache covers a larger fraction of a layer.

---

## 3. Sparse attention at decode

All three inherit ik's dsv4 machinery, and — contrary to what one might assume from the "ik cannot
run V4.1" note in the earlier report — **all three kept ik's decode-time gather**, which is the
reference's intent rather than mainline's mask:

- `mine/src/graphs/build_deepseek4.cpp:1351-1381`: `if (n_tokens == 1)` →
  `ggml_get_rows_ext(comp_kv, shared_top_k, …)` and the same on the mask; otherwise
  `build_top_k_mask` over the full compressed width.
- `skel/src/graphs/build_deepseek4.cpp:1757-1812`: same, extended to dequantise on the fly when the
  cache row type is packed (`!llama_is_packed_kv_cache_type(csa_kv->type)` as the `same_type` flag).
- `phyl/src/graphs/build_deepseek41.cpp:1147-1162`: same gather, with the mask branch forced to
  full width when flash-attention is off (`dsv41_build_top_k_mask(..., !cparams.flash_attn)`).

**Indexer scoring.** The kernel is ik's fused `ggml_indexer_topk` (score + top-k in one op) taken
when `cparams.fused_idx_topk` is set — `mine:880-883`, `skel:1022`, `phyl:796-800` — with a generic
matmul + `ggml_top_k` fallback. Cost scales with the compressed length `ctx/ratio`, as the reference
says. skel is the only port that bounds the fallback's memory: it chunks the unfused lightning-
indexer score over tokens to keep the KQ buffer small (`skel@d4d30733`;
`skel/src/graphs/build_deepseek4.cpp:24`).

**Two-level candidate-block selection: skel only.** `dsv4_build_candidate_mask`
(`skel/src/graphs/build_deepseek4.cpp:1036-1094`) max-pools the causally masked index scores into
blocks with `ggml_pool_2d`, adds a pin tensor so the half-full newest block cannot be outscored by
an older full one, takes `top_k` over blocks, builds a `-inf/0` keep-mask with `ggml_set_rows`, and
repeats it back to position granularity. It returns `nullptr` when every block fits the top-k, so
the op disappears below the 2048×8 = 16,384-position threshold. `mine` and `phyl` do not implement
it (`grep candidate` finds nothing in either graph).

**Ratio-2 "the group completes every other token".** Handled by the inherited per-ratio compressor
state ring plus the per-ubatch `comp_plan` of indirection tensors; `mine@0c6e934a` states the V4.1
change as "Both plan slots take their ratio and overlap from the file (2 and 1, disjoint groups)
instead of V4's hardcoded 4-overlapping and 128", and `phyl` builds the two plans explicitly as
non-overlap with a `ratio`-row state ring (`phyl/src/llama-dsv41.cpp:1066-1071`). The graph shape
is kept constant across off-steps by the plan's write indices, not by a graph branch. skel keeps
the gate-less ratio-1 compressor honest by substituting a zero score tensor so the softmax over a
one-element group is exactly 1.0 (`skel/src/graphs/build_deepseek4.cpp:1360-1366`).

**A divergence hiding in plain sight — the indexer Hadamard.** ik's V4 rotates indexer q and k by a
random-orthogonal block Hadamard to help quantized caches. `skel` gates it off for V4.1 with a
comment ("the indexer_q hadamard is V4-only", `skel/src/graphs/build_deepseek4.cpp:1202-1207`);
`phyl` omits it ("cloned … minus the Hadamard — V4.1 carries no indexer rotation",
`phyl/src/graphs/build_deepseek41.cpp:712-713`). `mine` **keeps it on both sides** — q at
`mine/src/graphs/build_deepseek4.cpp:848-850` and k at `:1155-1166`, with the comment "the indexer
query is rotated the same way". Because the same orthogonal transform is applied to both operands
of the dot product, the scores are preserved up to floating-point error, so this is a cost
difference and a tie-break-order difference rather than a semantic one (**inferred** — I did not
find a measurement of it in any tree).

**What this means for our engine.** The gather-not-mask decision is settled three ways over: every
port that had the choice kept the gather, and the plan's 640-row decode attention is the right
target. Implement the ratio machinery as a plan of indices computed on the host per step (what all
three do) rather than as graph branches — it is what makes the "group completes every other token"
problem disappear. Defer the two-level candidate pool: only skel built it, it is inert below 16,384
compressed positions, and at ratio 1 that is 16k tokens of context. If we do build it, skel's shape
(max-pool → pin the newest block → top-k → repeat back) is a direct transcription.

---

## 4. KV cache numerics

`mine` and `phyl` do not simulate the reference's quantized caches at all: caches are the
user-selected `type_k` (F16 by default), exactly as `v41-ops-report.md` §3.4 describes for both
earlier ports.

**skel implements them, as storage types.** Three new ggml types, one per (value grid, block size,
scale dtype), documented against the reference rule rather than ik's MXFP4 rule
(`skel/ggml/src/ggml-kv-quants.h:7-33`):

| type | grid | block | scale | site |
|---|---|---|---|---|
| `GGML_TYPE_FP4_B16_E4M3` | e2m1 | 16 | e4m3 | compressed (main) KV |
| `GGML_TYPE_FP4_B32_E8M0` | e2m1 | 32 | e8m0 | indexer K **and** Q |
| `GGML_TYPE_FP8_B32_E8M0` | e4m3 | 32 | e8m0 | window KV |

Scale rules are quoted from the reference (`s = 2^ceil(log2(amax/6))` with floor `6·2^-126`;
`s = e4m3(amax/6)` with floor `6·2^-9`; `s = 2^ceil(log2(amax/448))` with floor `1e-4`), rounding is
round-nearest-even "verified against the reference's own CUDA kernels … RNE, 36/36 configurations
exact", and codes are packed 2-per-byte in the reference's nibble order so a block can be compared
byte-for-byte with the reference's output (`skel/ggml/src/ggml-kv-quants.h:18-29`). The types are
storage-only and must never be a raw operand of a compute op (`:31-33`); reads go through a
dequantising `ggml_get_rows` with a host-filled read index
(`skel/src/graphs/build_deepseek4.cpp:102-135`).

Everything is behind `--packed-kv-cache` and a parse-time gate refuses `-ctk fp4_B16_E4M3` without
it, so a cache type cannot be switched on silently (`skel/common/common.cpp:4229-4245`,
`skel/src/llama-cparams.h:49`). The compressed cache is packed by the flag independently of the raw
cache type (`skel/src/llama-dsv4.cpp:979`).

The one measured quality statement in any of the three trees is here, and it is about *Q*, not the
cache:

> D.0 measured on the real V4.1 that the round-trip changes the top-k selection for 100% of the
> tokens and 7.33% of the slots, so scoring a packed indexer K against an unrounded Q is a silent
> deviation on essentially every selection
> — `skel/src/graphs/build_deepseek4.cpp:1096-1102`

So skel also fake-quantises the indexer Q (set_rows into a scratch of the indexer-K type, get_rows
back to F32; no new op, no new type, Q is never cached), and states it is not optional
(`skel/src/graphs/build_deepseek4.cpp:1096-1115`, `skel@0b239c22`). I found no PPL or accuracy
measurement of the packed caches themselves in the tree (`tests/test-kv-quants.cpp` has no `ppl`,
`rmse` or tolerance strings; the eight KV-quant commits have empty bodies).

A separate, measured memory win is `--swa-compress` (`skel@d64cc0da`), which required wiring the
compacted window mask for V4.1: "ctx 65536 / 62000-token prompt: prefill +0.95%, decode +6.0%, raw
K 2.68 GB -> 31 MB" and, at ctx 256000 with DSpark, "raw K 10000 -> 330 MiB".

**What this means for our engine.** skel's rule table is a free, already-cross-checked
specification of the three reference quantizers — worth transcribing verbatim into our kernel
crate, including the floors and the nibble order, because byte-comparability against the reference
is a gate we can actually run. The finding that matters more is the asymmetry: quantising K without
rounding Q changes 100% of top-k selections. If we ever store the indexer keys in a packed form we
must round the query through the same grid, and our oracle's "indexer top-k ids match as integers"
gate will otherwise fail for a reason that is not a bug in our indexer.

---

## 5. Hyper-connections

All three use ik's fused op unchanged: `ggml_hc_pre(ctx0, mixes, hc_scale, hc_base, hc,
hparams.dsv4_hc_sinkhorn_iters, hparams.dsv4_hc_eps)` — `mine/src/graphs/build_deepseek4.cpp:654`,
`skel:821`, `phyl:651`. The op takes the 24-wide mix vector and produces pre/post/comb with the
Sinkhorn iteration folded in; there is no separate `ggml_sinkhorn` call in any V4.1 graph (the op
exists at `mine/ggml/src/ggml.c:10331` but is not called from the dsv4 path). **It has a CUDA
kernel**: `ggml-cuda/sinkhorn.cu` and `sinkhorn.cuh` are present in all three trees and are the only
files under `ggml/src/ggml-cuda/` that mention `hc_pre`, so on a GPU-resident sublayer the mix stays
on the device instead of forcing a host round trip 80 times per token. No port pins placement
beyond that.

The iteration count is read from the GGUF key `%s.hyper_connection.sinkhorn_iterations`
(`mine/src/llama-arch.cpp:222` and the same line in the other two) with a **fallback of 3** when the
key is absent, identically in all three (`mine/src/llama-hparams.cpp:2067-2068`,
`skel:2130-2131`, `phyl:1994-1995`); V4.1's file carries the key, so the effective value is the
reference's 20. It is logged at load (`mine/src/llama.cpp:2960`).

The V4.1-specific change is the **lag**, and it is the same in all three: attention collapses the
stream with the mix the *previous* FFN produced, and the last FFN's mix does the output collapse,
"which is why the file ships no `output_hc_*`" (`mine@0c6e934a`). No port records a cost figure for
the hc path.

**What this means for our engine.** Fusing the mix projection, the pre/post sigmoids and the 20
Sinkhorn sweeps into one op is the settled choice — three independent ports use the same fused op
and none of them profiled it separately, which is itself weak evidence that it is not on the
critical path at batch 1. Our planned "hyper-connection mix (24-wide gemv + Sinkhorn 4×4)" kernel is
the right granularity. Take the lag as a structural fact and not a flag: it changes which mix a
sublayer consumes and it is why there is no learned hc head.

---

## 6. Draft (DSpark / MTP)

**mine: none.** `mine@0c6e934a` lists under "Not implemented": "session save and restore of the
shared compressed streams (the raw window still round-trips, and the skip is logged), the MTP
graph, and the quantizer exemption for the engram gate scales". The runtime warning is at
`mine/src/llama.cpp:10948`. The only spec-related change is widening an MTP-width predicate from
`LLM_ARCH_DEEPSEEK4` to `llm_arch_is_dsv4` (`mine/src/llama-spec-features.cpp:30`).

**skel: a real DSpark drafter, as a separate architecture.** `LLM_ARCH_DFLASH` with its own window
KV cache and transition planner (`skel/src/llama-dflash.cpp` +190,
`skel/src/llama-spec-features-dflash.cpp` +68), described in `skel@c0014d02` as "DSpark draft model
(LLM_ARCH_DFLASH) + spec-decode (disabled for serving)". The draft's own window KV is packed too
(`skel@2154c9fc`). A `--override-kv general.architecture` on the target is stripped before the draft
loads, because the draft declares its own arch (`skel/common/speculative.cpp:2045-2067`).

**What must be rolled back on rejection** is the dsv4 per-step checkpoint of the compressor state
rings — shadow buffers plus per-step deltas allocated per state tensor
(`skel/src/llama-dsv4.cpp:1253-1326`, release paths at `:1212, 1237`), driven through
`llama_dsv4_spec_ckpt_save` / `_restore` (`mine/src/llama.cpp:10351-10416` in the shared code). skel
found this was silently dead for V4.1 after an upstream refactor: `common_speculative_needs_
checkpoint()` and the server's checkpoint decisions moved onto `llama_model_has_recurrent()`, which
covered `DEEPSEEK4` but not `DEEPSEEK41`, so "the dsv4 spec-checkpoint path is dead for V4.1"
(`skel@4357936b`, which fixes it and reports "32 context checkpoints created and 3 restored … zero
rewind-refused or restore failures"). Engram state is not in the checkpoint in any port.

The most instructive draft bug is `skel@e64c427f`. A previous fix re-anchored the raw-K window crop
to the last valid row, which shifted the window for *every* batch; with speculative decoding the
target decodes a multi-token verification batch and the output degraded rather than crashed —
quoted verbatim:

> needle list @8k 'QX-4417 ZP-8823 MK-1706' … re-anchored 'QX-4417 ZP-8823 MK-8823 MK-8823
> MK-1706 …' 96 tok, length … MK-8823 is not in the document - it blends the two real codes …
> The same binary without the draft is clean, and the pre-fix binary with the draft is clean:
> the regression needs the shifted window *and* a multi-token batch.

The predecessor commit `skel@d271e2f2` is the matching all-`-INFINITY` mask row → NaN softmax →
"Failed to sample token" chain, with the exact arithmetic of the 59 tokens that fell outside their
own window.

**phyl: no draft.** Its related fix is the `seq_rm` contract (`phyl@7b69c599`): the M3-era guard
reset the DSV41 stream state including the position-contiguous engram cache on *every* `seq_rm`,
including the no-op trims llama-server issues each prompt, so on turn 2 of a continuation chat the
KV prefix survived while the engram cache was wiped and the process aborted on "DSV41 engram:
position gap". The new contract is pass-through for no-op ranges, refuse for partial trims, reset
only for a full clear.

**Nobody rolls back the engram history, and two of the three designs need to.** The three history
designs differ in exactly this property:

- `mine` derives the history from the KV cells at graph-input time (`llama_kv_prev_tokens`,
  `mine/src/llama.cpp:5751`), so a KV rollback rolls the engram history back for free. This is the
  only design that needs no separate rollback.
- `skel` keeps a per-sequence 3-entry rolling tail advanced once per ubatch token in position order,
  with the only reset at `pos == 0` (`skel/src/llama-engram.cpp:352-406`). After a 5-token
  verification batch of which 2 are accepted, the tail holds three *rejected* draft ids and the next
  three decodes hash the wrong rows — silently, since every row id is in range by construction.
  I found nothing that restores it: `grep` for `engram.hist`, `engram_tail` and `.hist` over
  `skel/src/llama.cpp`, `llama-spec-features*.cpp`, `llama-dsv4.cpp`, `llama-dflash.cpp` and
  `common/speculative.cpp` returns **no matches at all** outside `llama-engram.cpp`. skel is the one
  port that runs DSpark, so this is live, not hypothetical (**inferred** that it is a defect — I
  could not run it).
- `phyl` indexes by absolute position and *skips* positions already pushed
  (`skip = engram.size - first_pos`, `phyl/src/llama-dsv41.cpp:1003-1010`), so a re-decode at a
  rejected position does not overwrite the stale id either. Moot only because phyl has no draft.

**What this means for our engine.** Our rollback set is the compressor state rings and the window
ring — and, unless the engram history is *derived* from the KV cells the way mine does it, the
engram tail as well. Deriving it is the cheaper answer: it makes one rollback owner instead of two,
and it is the single structural difference that turns skel's silent staleness into a non-issue. The
V4.1-specific trap worth copying into our test plan is skel's: a window
bug that is invisible at batch 1 and only corrupts output under a multi-token verification batch.
Any speculative-decode gate we write must run a multi-token verify batch after a deep rewind, not
just single-token decode.

---

## 7. Where the ports disagree with each other or with the reference

| topic | reference (v41-ops-report §) | mine | skel | phyl |
|---|---|---|---|---|
| MoE weight-sum epsilon | `Σ + 1e-20` (§2.2 M3) | **no epsilon at all** — plain `ggml_div` by `sum_rows`; `DEEPSEEK41` matches neither branch (`mine/src/llama-build-context.cpp:1548-1562`) | `+1e-20`, matching the reference (`skel@a59a2507`, `skel/src/llama-build-context.cpp:1563-1564`) | **no epsilon** (`phyl/src/llama-build-context.cpp:1549-1558`) |
| engram gate at `dot == 0` | `copysign` → 0.50025 (§2.2 E4) | `ggml_sgn` → gate 0.5 (`mine/.../build_deepseek4.cpp:1038-1039`) | same (`skel/.../build_deepseek4.cpp:2169-2170`) | same, and says so in a comment (`phyl/.../build_deepseek41.cpp:1276-1279`) |
| indexer Hadamard rotation | absent from V4.1 (§3.4 lists it as ik's V4 aid) | **applied to q and k** (`:848-850`, `:1155-1166`) — orthogonal on both sides, so score-preserving, but extra work and different tie order (**inferred**) | off for V4.1, explicit comment (`:1202-1207`) | omitted, explicit comment (`:712-713`) |
| compressor per-slot bias (`ape`) | absent (§3.4) | `nullptr` on the V4.1 path (`:1146`) | skipped with a comment (`:1370-1375`) | not built |
| ratio-1 gate-less compressor | plain projection, no gate (§1.3) | falls back to the **kv values themselves** as the score: `state_score = comp_wgate ? mm(comp_wgate, cur) : state_kv` (`mine/.../build_deepseek4.cpp:924`) — equivalent only because a one-element softmax is 1.0 whatever the score | zero-score substitution, with the one-element-softmax reasoning in the comment (`skel/.../build_deepseek4.cpp:1358-1366`) | non-overlap plan per ratio (`phyl/src/llama-dsv41.cpp:1066-1071`) |
| decode compressed selection | gather exactly the top-512 (§3.4) | gather at `n_tokens==1` (`:1351-1381`) | gather, dequantising if packed (`:1757-1812`) | gather (`:1147-1162`) |
| two-level candidate blocks | `select_candidate_blocks`, pin newest (§5.1) | not implemented | **implemented** (`:1036-1094`) | not implemented |
| quantized-cache simulation | fp8 window K / fp4-16 compressed KV / fp4 indexer k,q (§1.8) | none (F16) | all three, storage types + Q round-trip (`ggml-kv-quants.h:7-33`, `:1096-1115`) | none (F16) |
| Sinkhorn iterations | 20 (§1.4) | key, fallback 3 (`llama-hparams.cpp:2067`) | same (`:2130`) | same (`:1994`) |
| engram hash constants | GGUF metadata per `v41-ops.md` | GGUF (`mine/src/llama.cpp:5392-5395`) | **sidecar `.engram-constants.bin`**, GGUF carries none (`skel/src/llama-engram.h:8-10`) | GGUF, "must never be regenerated at runtime" (`phyl/src/llama-engram.h:5-8`) |
| engram pad at sequence start | lookback stops at start, pad id 2 (§4.1) | `blocked` latch, pad thereafter (`mine/src/llama.cpp:5404-5409`) | `DEAD` tail + `pad_compressed`, `pos==0` refills the tail (`skel/src/llama-engram.cpp:369-380`) | `blocked` latch on `p < shift` or `DEAD` (`phyl/src/llama-engram.cpp:43-47`) |
| engram out-of-range row | reference zero-masks (`model.py:312-321`) | not handled (ids trusted) | not handled | zero-masks, with the reason in a comment (`phyl/src/llama-engram.cpp:71-77`) |

The first row is the one that changes outputs: with `√softplus` scores all six selected weights are
positive so the sum is normally far from zero, but skel found and fixed a real case — its commit is
titled "keep the MoE weight normalisation finite when a token's gate row sums to zero". In `mine`
and `phyl` that token divides by zero.

Note also a correction to `v41-ops-report.md` §3.4: it states ik clamps the MoE weight sum at
6.1035e-5, citing `ik/llama-build-context.cpp:1547`. Checked against the common ancestor rather than
any port — `git -C mine show 3bb386eb:src/llama-build-context.cpp` puts
`if (lctx.model.arch == LLM_ARCH_LAGUNA)` at line 1546 and the clamp at 1547 (`ik@3bb386eb`). The
cited line *is* the clamp, but it is inside the LAGUNA arm, so it never applied to the DeepSeek-V4
family; the `1e-20` arm at 1550 lists BAILINGMOE2/3 and STEP35 only. skel's fix adds `DEEPSEEK41` to
the second arm; mine and phyl leave both arms unentered.

**What this means for our engine.** Three of these are integer-valued and therefore gateable by the
oracle we already planned: indexer top-k ids, engram row ids, router top-6 ids. The epsilon is not —
it is a silent NaN on a rare token, so put `Σ + 1e-20` in from the start and make the zero-sum case a
unit test rather than a discovery. Do not copy mine's Hadamard: it is inherited V4 machinery that
buys nothing when the cache is not quantized.

---

## 8. Performance facts recorded in the trees

Verbatim, with coordinates. No editorialising.

**mine@b63c33e1** (engram prefetch), the only V4.1 end-to-end tok/s in any of the three trees:
> DeepSeek-V4.1-Flash Q3_K_M, experts of six layers on two GPUs and the rest on the CPU, six new
> 200-token greedy prompts: 41-62 major faults per token and 13.6 tok/s before, 1-11 faults and
> 18.4 tok/s after; the same prompts with the rows already cached decode 18.8-19.0 either way.

**skel@d64cc0da** (`--swa-compress` on V4.1):
> 5k needle A/B, 3 runs per arm: identical output (the first pass's 1.43x prefill was a cold-page-
> cache artifact; warm arms agree within 0.5%) / ctx 65536 / 62000-token prompt: prefill +0.95%,
> decode +6.0%, raw K 2.68 GB -> 31 MB / DSpark x --swa-compress at the production entry flags
> (ctx 256000): runs, coherent, same final answer; raw K 10000 -> 330 MiB

**skel@c0014d02** (the port itself): "DeepSeek-V4.1-Flash (748.5B, MXFP4) … GGUF converter (507.9
GB, 1046 tensors)".

**skel/src/graphs/build_deepseek4.cpp:1099-1101**: "D.0 measured on the real V4.1 that the
round-trip changes the top-k selection for 100% of the tokens and 7.33% of the slots".

**phyl — all figures are GLM-5.3-Flash 321B IQ3_XXS on an RTX 3090, not V4.1:**
- `PHASE4-STATUS.md:13-15`: "base 7.86 → M3c unstaged 8.11 → **M3c staged 9.98 t/s**; hit 0.46
  @step 100 (stories converged 0.524). TG ≥ 12 t/s gate still FAILS."
- `PHASE4-STATUS.md:16-21`: "Hits always converted (−30.6 ms/step CPU at hit 0.44); pageable-source
  promotion copies clawed ~27 ms/step back via driver staging-lock serialization with critical-path
  input copies (KQ_mask sets 0.04 → 28.5 ms/step)."
- `PHASE4-STATUS.md:97-101`: "**PCIe microbenchmark: 26.2 GB/s sustained** pinned HtoD on a
  dedicated non-blocking copy stream with 8.6 MB slot-sized chunks (idle GPU), **26.15 GB/s while
  the compute stream is saturated** … Pageable memory halves it (14.4 GB/s) … Single-slice burst
  latency 0.53 ms, amortized 0.34 ms/slice at 16."
- `PHASE4-STATUS.md:105-107`: "**Capture-break cost (GGML_CUDA_DISABLE_GRAPHS=1, champion #12
  @19.7k): TG 8.76 vs 10.16 (−13.8%), PP 68.98 vs 77.3 (−10.7%).**"
- `PHASE4-STATUS.md:74`: "TG 10.16 → 14–16 t/s. Traffic model TG ≈ 9.01/(1−hit) validated."
- `PHASE4-M3C-ADMISSION.md:24-30`: classic LRU ~160 promotions/step ≈ 12 GB/s; 2nd-miss filtered
  ~113/step ≈ 8.5 GB/s; cap 8 GB/s.
- `PHASE4-STATUS.md:523-524`: "GPU: RTX 3090 24 GiB, idle between runs … RAM: one hybrid run pins
  ~105 GiB; NEVER overlap two model runs."

Placement flags recorded: `-ot ffn_.*_exps=CPU` as the whole-config baseline
(`PHASE4-M3C-ADMISSION.md:63`); `-ot 'blk.(39|40|41|42|43).ffn_*=CUDA0'` as the pre-cache champion
(`PHASE4-DYNAMIC-EXPERT-CACHE-PLAN.md`, sizing section); `-ot ".*engram.*=CPU"` required by skel's
gather (`skel/src/llama-engram.cpp:436`).

**The spec's "6–8 tok/s decode on ONE 3090 + DDR4-2933" for phyl does not appear anywhere in the
phyl tree** — no docs, no README, no commit message. See §10.

---

## 9. Ranked list

### Ten most valuable things for our engine

1. **Prefetch the engram rows before the graph, and measure major faults not just tok/s.**
   `mine/src/llama.cpp:5419-5434`, numbers in `mine@b63c33e1`. Why: it is the only V4.1 end-to-end
   measurement in any tree and it says the fault count *is* the decode bottleneck (41–62 → 1–11
   faults, 13.6 → 18.4 tok/s). Cost: near zero — our B3 engram crate already plans the read to be
   issued right after sampling, which is strictly earlier than mine's. Add a major-fault counter to
   the telemetry.
2. **Hash everything first, touch the table second.** `skel/src/llama-engram.cpp:338-424`. Why: it
   guarantees the whole advice batch is issued before any row can block, which mine's per-row
   interleaving does not. Cost: a loop split in the engram crate.
3. **The three reference cache quantizers, transcribed with their floors and nibble order.**
   `skel/ggml/src/ggml-kv-quants.h:7-33`. Why: a written, RNE-verified, byte-comparable spec of
   fp8-B32-E8M0 / fp4-B16-E4M3 / fp4-B32-E8M0 that we would otherwise re-derive from `kernel.py`.
   Cost: one kernel-crate module; it is a specification, not code to port.
4. **Round the indexer Q through the same grid as the indexer K.**
   `skel/src/graphs/build_deepseek4.cpp:1096-1115`. Why: measured — 100% of tokens and 7.33% of
   slots change top-k selection otherwise; our "indexer top-k ids match as integers" oracle gate
   would fail for a non-bug reason. Cost: one fake-quant site, only if we pack indexer keys.
5. **`Σ + 1e-20` on the MoE weight sum, plus a zero-sum unit test.** `skel@a59a2507`. Why: a real
   token was found that divides by zero; `mine` and `phyl` still do. Cost: one line and one test.
6. **Two-path masked MoE execution with value-only masks and a trash slot.**
   `phyl/src/llama-build-context.cpp:1591-1701`. Why: it makes any expert-residency scheme bitwise-
   neutral by construction (exactly one path non-zero per token-slot), which turns "did the cache
   change my output" into a non-question. Cost: moderate — our graph is ours, but the pattern is
   small.
7. **Admission on the second miss in a window, not on every miss.**
   `phyl/docs/expert-cache/PHASE4-M3C-ADMISSION.md:24-30`. Why: measured 160 → 113 promotions/step,
   and one-shot experts never enter. Cost: a per-layer windowed miss counter.
8. **Pin the promotion staging buffer.** `PHASE4-STATUS.md:16-21`. Why: pageable promotion copies
   cost ~27 ms/step through driver staging-lock contention with the critical-path input copies —
   enough to erase the entire cache benefit. Cost: a staging ring; unavoidable knowledge, cheaply
   bought here.
9. **Derive the engram history from the KV cells, and keep the compressor state rings in the
   speculative checkpoint.** `mine/src/llama.cpp:5751`; `skel@4357936b`,
   `skel/src/llama-dsv4.cpp:1253-1326`. Why: both failure modes are silence — a KV-derived history
   cannot go stale on rejection, whereas skel's and phyl's separately-owned histories can (§6), and
   skel's checkpoint path simply stopped running for V4.1 after an unrelated upstream refactor.
   Cost: design-time only, if the state-ring owner and the history owner are decided together.
10. **A multi-token verification batch after a deep rewind, in the gate.** `skel@e64c427f` and
    `skel@d271e2f2`. Why: a window-crop bug that is clean at batch 1 and produces blended, plausible
    text under a verify batch — exactly the class our numeric gates miss. Cost: one harness case.

### Five things to avoid

1. **Do not carry the indexer Hadamard into V4.1.** `mine/.../build_deepseek4.cpp:848-850, 1155-1166`
   vs the explicit exclusions in `skel:1202-1207` and `phyl:712-713`. It is a V4 aid for quantized
   caches, score-preserving and therefore pure cost when our caches are not packed.
2. **Do not divide by a bare weight sum.** `mine/src/llama-build-context.cpp:1548-1562`,
   `phyl:1549-1558`.
3. **Do not keep the engram history in a second, independently-owned structure.** Both alternatives
   to deriving it from the KV cells fail under rewind: phyl's absolute-position vector was wiped by
   a no-op server trim while the KV prefix survived, aborting on turn 2 (`phyl@7b69c599`), and its
   `skip` logic (`phyl/src/llama-dsv41.cpp:1003-1010`) will not overwrite a rejected position;
   skel's rolling tail (`skel/src/llama-engram.cpp:352-406`) is advanced by rejected draft tokens
   and nothing in that tree restores it. `mine`'s KV-derived history (`mine/src/llama.cpp:5751`) is
   the design with one rollback owner.
4. **Do not put the hash constants outside the model file.** skel's sidecar
   (`skel/src/llama-engram.h:8-10`) silently disables engram when absent
   (`skel/src/llama-engram.cpp:204-208`); phyl's comment states the opposite requirement, that the
   constants must be bit-exact with training and never regenerated
   (`phyl/src/llama-engram.h:5-8`). Read them from the file.
5. **Do not build the two-level candidate pool yet.** Only skel has it
   (`skel/.../build_deepseek4.cpp:1036-1094`), it self-disables below 2048×8 compressed positions,
   and at ratio 1 that is 16k tokens. It is graph surface with no payoff at our planned contexts.

---

## 10. What I could not determine, and my least-sure claims

**Could not determine.**

1. **phyl's "6–8 tok/s decode on one 3090 + DDR4-2933" is not in the tree.** Every performance
   figure in `phyl/docs/` is GLM-5.3-Flash; `grep -rn 'V4.1|v41|deepseek41' phyl/docs` returns
   nothing and no V4.1 commit message carries a speed. The number in the task brief must come from a
   PR or issue thread I was not allowed to fetch.
2. **Whether phyl's expert cache has ever been run against V4.1.** The eligibility predicate is
   written for the glm5next tensor shape (`phyl/src/llama-build-context.cpp:1599-1614`) and V4.1
   reaches it only through the shared `llm_build_moe_ffn` call at
   `phyl/src/graphs/build_deepseek41.cpp:1393`. Nothing records a V4.1 run.
3. **Quality impact of skel's packed KV caches.** The eight KV-quant commits have empty bodies,
   `tests/test-kv-quants.cpp` contains no PPL/tolerance assertions, and no doc in the tree reports a
   before/after. Only the *indexer-Q* effect (100% / 7.33%) is measured.
4. **Whether skel's DSpark path is usable.** `skel@c0014d02` says "spec-decode (disabled for
   serving)" and I did not find a commit re-enabling it; the later draft-window work
   (`skel@2154c9fc`) and the rewind repros suggest it is exercised in development.
5. **Engram hot-row caching.** No port caches hot rows; whether the natural row reuse across tokens
   is high enough to matter is not measured anywhere.
6. **Whether skel's engram tail staleness under a rejected draft batch is real in practice.** The
   code path is clear (§6) and nothing in the tree restores the tail, but skel runs DSpark in
   development and the rewind repros in `skel@e64c427f` pass, so either the effect is below the
   needle tests' resolution or something I did not find resets it. This is the finding in the report
   most worth a second pair of eyes.

**The three claims above I am least sure of.**

1. **That mine's Hadamard is output-preserving.** The argument is that the same orthogonal block
   transform is applied to both q (`:848-850`) and k (`:1155-1166`), so the dot product is invariant.
   The block size is confirmed identical: both sites call `hadamard_size(indexer_head_size)` and
   `indexer_head_size` = 128 is a power of two, which that function returns unchanged
   (`mine/src/llama-model.h:699-707`). What remains unverified is whether ik's `ggml_hadamard` is
   exactly orthogonal (normalised) rather than scaled, and whether the two sites see the same
   random seed if the transform is a *random*-orthogonal one as `v41-ops-report.md` §3.4 describes
   for ik. If it is scaled rather than normalised, the scores are multiplied by a constant, which
   still leaves top-k unchanged. Marked inferred in §3 and §7.
2. **The engram upload volume comparison** (§1, 49,152 B/token for skel/phyl vs the same F32 crossing
   as a scheduler copy for mine). The skel/phyl figure is direct from the input tensor shape; the
   mine figure depends on `engram_wkv` actually being device-resident (`ctx_split`,
   `mine/src/llama-load-tensors.cpp:3470`), which is a placement decision at runtime, not a fixed
   property.
3. **That `mine` and `phyl` have no MoE weight-sum epsilon.** I read the branch chain at
   `mine/src/llama-build-context.cpp:1548-1562` and the equivalent in phyl and saw `DEEPSEEK41`
   match neither arm, then the bare `ggml_div`. This also contradicts `v41-ops-report.md` §3.4's
   claim that ik clamps at 6.1035e-5, which I believe was a misread of the LAGUNA guard — but two
   readings of the same chain are not independent, and skel's commit exists precisely because
   someone hit the case, which is the strongest support for the reading.
