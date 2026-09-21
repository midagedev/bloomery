# Scheduling CPU-resident and GPU-resident work together in MoE decode

Investigation round, 2026-09-21. Labels used throughout: **READ** = I read the source at the cited
`path:line`; **MEASURED** = a number someone measured, with hardware/model/batch named; **CLAIMED** =
a README/abstract assertion I could not check; **DERIVED** = my arithmetic on top of named inputs.

**Every timeline in this report is given as a bracket over two GPU bandwidths**, because the choice
changes the answer by 2×. The **floor** bracket uses **700 GB/s** (3090 stream read, `roofline.md:41`;
our own q4k gemv measured 810 GB/s, MUL-9). The **envelope** bracket uses **247 GB/s**, which is
ik's stall-inclusive whole-pass average DERIVED in `roofline.md:76-79` — it already contains launch
gaps, attention compute and the very syncs this report proposes removing, so it is a description of
today's ik, not a property of the card. **For our engine the floor bracket is the relevant one.**

---

## 1. One-paragraph answer

The user's hypothesis is right in mechanism and smaller than it sounds at batch 1. The serial wait is
real and I verified it in source: ggml's scheduler runs graph splits in a plain `for` loop with a
`ggml_backend_synchronize` at every CPU↔GPU boundary, and the CPU backend's `graph_compute` is
synchronous on the scheduler thread — so while the host computes a layer's experts nothing is enqueued
to the GPU, and before the host can read a router output the GPU stream is drained
(`ggml-backend.cpp:2483-2531`, `:2016-2162`, `:875-895`, `:940`). ik has a parallel split executor but
it is gated on `n_backends > 2 && split_mode_graph && has_reduce` (multi-GPU tensor-parallel) and its
OpenMP arm explicitly *bails out when the CPU backend owns more than one split* — ik itself treats CPU
splits as incompatible with concurrent split execution (`:2221`, `:2233-2254`). Mainline llama.cpp has
no parallel executor at all: one serial loop, `ggml-backend.cpp:1646-1830` (fetched from `master`).
**But the idle time is not mostly sync overhead** — from our own numbers the boundary legs are ~1–2 %
of a token while the *serialisation itself* is ~40 % GPU idle and ~60 % CPU idle (§2, DERIVED from
`roofline.md:76-84`). The one system that actually overlaps intra-layer at batch 1 is KTransformers,
and it overlaps exactly one thing: the **shared expert on the GPU against the routed experts on the
CPU**, joined inside a single CUDA graph via `cudaLaunchHostFunc` host nodes (READ,
`archive/ktransformers/operators/experts.py:982-986` + `archive/csrc/ktransformers_ext/cpu_backend/cpuinfer.h:69-90`).
For V4.1 on our box that one trick is worth **−5.6 % (floor) to −10.4 % (envelope)** of a token,
because V4.1's shared expert is unusually fat (37.6 MB/layer, q8_0 — §2.1). Everything else that is
popular — expert prediction/prefetch, uploading experts over PCIe, grouping verify-batch tokens by
expert — **does not pay at batch 1 on this hardware**, and §5 shows the arithmetic for each rejection.
The two levers that do pay are (R1) overlap the shared expert + GPU-cached experts with the host expert
leg, and (R3) an expert cache that moves bytes off the host leg permanently — together **−33 % to
−41 %** at a hit rate of 0.5, which is itself the unmeasured input. (R2) in-graph wait/write-value
nodes at the CPU boundary is the enabler that lets R1 live inside one captured graph per layer instead
of forcing eager submission.

---

## 2. The timeline, with numbers

### 2.1 Byte budget per layer (DERIVED from `docs/v41-inventory.md`, formulas shown)

Per-expert bytes, typical layer (row `blk.20`, `docs/v41-inventory.md:577-583`):

| tensor | type | bytes (384 experts) | ÷ 384 = per expert |
|---|---|---:|---:|
| `blk.20.ffn_gate_exps` | q3_K | 1,946,419,200 | 5,068,800 |
| `blk.20.ffn_up_exps` | q3_K | 1,946,419,200 | 5,068,800 |
| `blk.20.ffn_down_exps` | q4_K | 2,548,039,680 | 6,635,520 |
| **one expert (gate+up+down)** | | | **16,773,120 B = 16.77 MB** |

Top-6 → **100.6 MB of routed expert bytes per layer per token** (DERIVED). Cross-check: `roofline.md:20`
counts 3.7656 GiB routed per token over 40 layers = 101.1 MB/layer — agrees to 0.5 % (layers 0–1 carry
q5_K `down`, which is larger; `v41-inventory.md:46`). I use **101.1 MB** below so the per-layer rows
sum to the whole-model rows.

Shared expert, same layer (`docs/v41-inventory.md:578-584`): `ffn_{gate,up,down}_shexp` are **q8_0**,
12,533,760 B each → **37.60 MB/layer**. Attention is 5.02 GiB over 40 layers = **134.7 MB/layer**
(`v41-inventory.md` types table + HANDOFF §7 B0b). GPU-side bytes per layer =
(dense 7.132 GiB/token − the 543 MB `output` head) ÷ 40 = **177.9 MB**, of which shexp is 37.6 and
"everything else" (attention, router, norms, hyper-connections, indexer) is **140.3 MB** (DERIVED).

> Note: the shared expert is **2.24 routed experts' worth of bytes** and it lives on the GPU. That
> single fact is why intra-layer overlap is worth more on V4.1 than on the models the literature
> measured (Mixtral has no shared expert; DeepSeek-V2-Lite's shexp is ~2 experts of a 64-expert pool).

### 2.2 Bandwidths (MEASURED, with source)

| leg | GB/s | source |
|---|---:|---|
| host DDR4-3600 ×8, STREAM triad | 147.7 | HANDOFF §1 |
| PCIe 4.0 ×16 HtoD pinned | 26.2 (26.15 with the compute stream saturated) | `docs/research/v41-ports.md:38` |
| PCIe 4.0 ×16 HtoD pageable | 14.4 | same |
| 3090 stream read (**floor bracket**) | 700 (DERIVED, 81 % of 936) | `roofline.md:41` |
| 3090, our q4k gemv / q3k gemv M=1 | 810 / 348 | HANDOFF §7 (MUL-9); `roofline.md:41` |
| 3090 effective on today's whole V4.1 pass (**envelope bracket**) | 247 (DERIVED) | `roofline.md:76-79` |

### 2.3 Fixed costs of a boundary (MEASURED on our box)

| item | µs | source |
|---|---:|---|
| eager op + `cuStreamSynchronize` floor | 5.2 | HANDOFF §7, A1d profile ("net = 동기화 바닥 5.2") |
| one small synchronous H→D copy | ~4.6 (`refresh_params` = 4 copies, 18.3 µs host) | HANDOFF §7, A1d |
| graph node gap | 0.76 | HANDOFF §7, gpu-cores |
| host submit, eager vs graph replay | 4.1 vs 1.4 per 2-launch step | HANDOFF §7, P0 |
| our CPU pool dispatch tax | 4.65 per dispatch | HANDOFF §5 (MUL-35 `pool-rate`) |

Payload at a boundary is tiny: hidden 5120 × f32 = **20.48 KB** → 0.78 µs pinned / 1.42 µs pageable
(DERIVED). With hyper-connections' 4 streams, 81.9 KB → 3.1 µs. Router ids: 6 × i32 = **24 B**,
latency only. **So batching the transfer is not a lever at our sizes**: one pinned upload per layer of
all host results versus one per expert differs by ~2 µs of copy and ~5 µs per avoided sync — the round
trip is dominated by the synchronize, not by the bytes. Budget **~7 µs per boundary, 2 per layer**.

### 2.4 Today's timeline, serial (one host-resident layer)

```
   GPU: [ attn+router+rest ]......idle.......[ shexp ][ combine ]
                        |  ids 24 B         ^
                        v  sync ~7 µs       |  result 20 KB + sync ~7 µs
   CPU: .....idle......[ 6 experts, 101.1 MB @147.7 GB/s = 0.684 ms ].....idle....
```

| leg | bytes | floor @700 GB/s | envelope @247 GB/s |
|---|---:|---:|---:|
| GPU attn + router + rest | 140.3 MB | 0.200 ms | 0.568 ms |
| boundary out + in | 20.5 KB ×2 | 0.014 ms | 0.014 ms |
| **CPU 6 routed experts** | 101.1 MB | **0.684 ms** | **0.684 ms** |
| GPU shared expert | 37.6 MB | 0.054 ms | 0.152 ms |
| **layer wall (serial)** | | **0.952 ms** | **1.418 ms** |
| **× 40 layers + head (543 MB)** | | **38.9 ms/token = 25.7 tok/s** | **58.9 ms/token = 17.0 tok/s** |

The envelope row lands on the **measured** one-pass number for this model on this box, **56.5 ms**
(`roofline.md:74`, 20 greedy prompts, draft off) — 4 % high, because the serving split puts 7 of 40
layers' routed experts on the two GPUs (`roofline.md:50-56`) and shortens those layers' host leg. That
agreement is partly by construction (247 GB/s was derived from 56.5) but the per-layer decomposition
was not, so the row sums are a real consistency check.

**Idle fractions, DERIVED directly from the measured decomposition** (`roofline.md:76-79`): of the
56.5 ms token, 22.6 ms is the host expert term and 33.9 ms is everything GPU-side.

- **GPU idle ≈ 22.6 / 56.5 = 40 % of every token.**
- **CPU idle ≈ 33.9 / 56.5 = 60 % of every token.**
- **Boundary/sync legs ≈ 80 × ~7 µs = 0.56 ms = 1.0 % of the token** (DERIVED from §2.3; 0.4–1.2 ms
  over the plausible per-boundary range 5–15 µs).

That last line is the load-bearing one. **The waste is the serialisation, not the synchronisation.**
A round that only makes the boundaries cheaper recovers at most ~2 %.

### 2.5 The same timeline under each top recommendation

All deltas below are against the **same-bracket derived baseline** from §2.4 (38.9 ms floor /
58.9 ms envelope), never against the measured 56.5 — the measured number belongs to a different
placement (7 layers of experts on GPU).

**R1 — shexp overlapped with the host leg**, batch 1, no cache. Layer wall becomes
`rest + boundary + max(host, shexp)`:

| | floor | envelope |
|---|---:|---:|
| layer wall | 0.200 + 0.014 + max(0.684, 0.054) = **0.898 ms** | 0.568 + 0.014 + max(0.684, 0.152) = **1.266 ms** |
| token | 36.7 ms (27.2 tok/s) | 52.8 ms (18.9 tok/s) |
| **vs baseline** | **−5.6 %** | **−10.4 %** |

**R3 alone — expert cache, hit 0.5, still serial.** 3 of 6 experts resident in VRAM (50.3 MB):

| | floor | envelope |
|---|---:|---:|
| layer wall | 0.200 + 0.014 + 0.342 + 0.072 + 0.054 = **0.682 ms** | 0.568 + 0.014 + 0.342 + 0.204 + 0.152 = **1.280 ms** |
| token | 28.1 ms (35.6 tok/s) | 53.4 ms (18.7 tok/s) |
| **vs baseline** | **−27.8 %** | **−9.3 %** |

**R1 + R3 — overlap *and* cache, hit 0.5.** Layer wall = `rest + boundary + max(host_remaining,
shexp + gpu_experts)`:

| | floor | envelope |
|---|---:|---:|
| layer wall | 0.200 + 0.014 + max(0.342, 0.126) = **0.556 ms** | 0.568 + 0.014 + max(0.342, 0.356) = **0.938 ms** |
| token | 23.0 ms (43.5 tok/s) | 39.7 ms (25.2 tok/s) |
| **vs baseline** | **−40.8 %** | **−32.6 %** |

Read the envelope column of that last row carefully: at hit 0.5 the GPU side (0.356 ms) has just
overtaken the remaining host side (0.342 ms). **Past roughly hit ≈ 0.5 the GPU becomes the long pole
and further promotion buys nothing.** There is a cheapest-correct cache size and it is smaller than
"as much VRAM as fits" — see §6.

**R2 alone — cheap boundaries, still serial**: −0.4 to −1.2 ms/token, **−0.7 to −2.1 %**. Its real
value is that it lets R1 live inside a captured per-layer graph (§4.9).

**DSpark verify batch k = 8**: host leg grows to 5.18 ms/layer, GPU leg barely moves.
```
   GPU: [ rest 0.20-0.57 ][ shexp 0.05-0.15 ]..........idle ~5 ms..........
   CPU: ..................[ 45.6 distinct experts, 764 MB @147.7 = 5.18 ms ]....
   layer wall ≈ 5.44 (floor) / 5.91 ms (envelope);  R1 saves 1.0 % / 2.5 %
```
Under speculation the overlap lever collapses and the cache lever grows. §6 works this through.

---

## 3. Ranked recommendations

Gain columns are `floor … envelope`, against the matching §2.4 baseline.

| # | recommendation | exact? | gain, batch-1 decode | needs from our architecture | size | risk |
|---|---|---|---|---|---|---|
| **R1** | Start the host expert leg right after the router; run **shared expert** (+ any GPU-cached routed experts) concurrently; join at combine | **exact** — same operands, same order; `moe_combine` is the only join | **−5.6 … −10.4 %** alone; **−33 … −41 %** with R3 at hit 0.5 | an async submit into our CPU pool + a join the GPU stream can wait on; `moe_combine` already exists as that join point (HANDOFF §7 "MoE 융합") | M | the join mechanism (R2); combine's parenthesisation must not move or the band gate shifts |
| **R2** | Put the CPU boundary **inside** the captured layer graph: `cuStreamWriteValue32` to hand off + `cuStreamWaitValue32` to join, on host-mapped pinned flags (graph node: `cuGraphAddBatchMemOpNode`) | exact | −0.7 … −2.1 % directly; **enables R1 without losing graph replay** — else ~25 nodes × 2.7 µs eager ≈ 68 µs/layer ≈ 2.7 ms/token penalty (DERIVED, HANDOFF §7 P0) | `Gpu` already owns a non-blocking stream and `graph.rs` owns capture/launch; two new node kinds | S–M | `CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS` unverified on the box; cuda-oxide safe wrapper absent (bindings likely present — bindgen `^cu.*`, HANDOFF §7) |
| **R3** | Expert cache in VRAM: two-path **value** masks + trash slot, admission on **second miss in a 64-step window**, **pinned** promotion staging | exact — phylliida's value-mask form is bit-identical to the single path (`v41-ports.md:32-34`) | **−9.3 … −27.8 %** alone at hit 0.5; the shared half of the −33…−41 % | `B1` placement table + per-layer slot tensors; `_sel` kernels already read expert ids from a device buffer (P9) | M–L | **hit rate for 384/top-6 is unmeasured**; pageable staging cost someone ~27 ms/step (MEASURED, `v41-ports.md:36-37`) |
| **R4** | Batch the boundary: one mapped pinned buffer per layer carrying {hidden, ids, weights} out and {result} back, allocated at load, never re-allocated | exact | ~0 time; it is a **prerequisite** for R1/R2, not a win | `Stage` already allocates all scratch at load (P8a) | S | none |
| **R5** | Two-stage host/GPU pipeline across ≥2 **independent sequences** | exact | max(22.6, 33.9) vs 56.5 = **1.67× throughput**, *no single-stream latency gain* (`roofline.md:148-156`) | the C3 scheduler | L | belongs to C3, not to B2 |

Rejected ideas and their arithmetic are in §5.

**Sequencing note:** R3 is worth more than R1 in the floor bracket, but R1+R2 is the smaller, safer
round and R3 without R1 leaves the GPU idle during the shortened host leg. Order: R4 → R2 → R1 → R3,
with the R3 hit-rate measurement (router-id replay, no GPU needed) running in parallel from the start.

---

## 4. Per-technique detail

### 4.1 ik_llama.cpp — serial, and it knows it (READ)

Paths below are `/Users/hckim/repo/upstream/v41-ports/mine/` (our PR #2455 branch, ik main `3bb386eb`).

- **The default executor is a serial loop.** `ggml/src/ggml-backend.cpp:2483-2531`:
  `for (i in splits) { copy_inputs(split); eval(split); }`. No overlap construct anywhere in it.
- **The CPU backend blocks the scheduler thread.** `ggml_backend_cpu_graph_compute` calls
  `ggml_graph_compute` inline and returns only when done (`:875-895`), and the CPU backend's
  `.synchronize` is `NULL` (`:940`) — it has nothing to synchronise because it is never async. So
  `ggml_backend_sched_eval` (`:2164-2215`) is blocking for a CPU split. **While the host computes a
  layer's experts, the host thread is inside the kernel and the GPU queue is not being fed.** The
  lead's belief is confirmed.
- **Every boundary drains the stream.** `ggml_backend_sched_copy_inputs` (`:2016-2162`) calls
  `ggml_backend_synchronize` at `:2036`, `:2047`, `:2059`, `:2083` (the ids readback), `:2149`,
  `:2154`. The `:2083` one is the router→host leg: `ggml_backend_tensor_get_async(ids)` immediately
  followed by a full backend synchronize.
- **ik's parallel executor exists and excludes us.** `:2221`:
  `if (sched->is_async && sched->n_backends > 2 && sched->split_mode_graph && sched->has_reduce)`.
  `has_reduce` is set only when the graph contains `GGML_OP_REDUCE` (`:1757`) — the multi-GPU
  tensor-parallel all-reduce. With one GPU + CPU, `n_backends == 2` and the condition is false. And
  inside it, the OpenMP arm computes `has_cpu_work` (`:2233-2245`) and **refuses the parallel path when
  the CPU backend owns more than one split** (`:2254`, `if (!has_cpu_work)`), falling back to
  `std::thread` + `std::barrier`. ik's own code treats CPU splits as hostile to concurrent split
  execution.
- **ik does have a host-side MoE prefetcher, and it is a page-cache warmer, not an overlap.**
  `ggml/src/ggml-moe-prefetch.cpp` (introduced by `6a909f4f`, "Add --prefetch-experts to stream mmap'd
  MoE experts into page cache (#2101)"); driven from `:2440-2540`. Two modes: lookahead streaming
  (`:2495-2504`, **only when `n_tokens >= 32`** — batch/prefill graphs) and selective per-node
  (`:2513-2517`) after ids are host-visible, with `MADV_COLD` afterwards (`:2535-2540`). It removes
  major faults; it does not make the GPU and CPU run at the same time.
- **ik also implements the "upload only the selected experts" path.**
  `:2053-2143`, gated on `sched->only_active_experts` (`llama.cpp:9572-9574`, CLI `--only-active-exps`,
  `common/common.cpp:4381`). It reads the ids to the host (`:2081-2083`), builds a bitset of unique
  expert ids (`:2089-2096`), and issues one `tensor_set_async` per **contiguous run** of selected
  experts (`:2110-2138`) — so the transfer is already coalesced. This is the direct upstream precedent
  for §4.8.
- **And ik states the break-even for that upload in code.** `ggml/src/ggml-cuda.cu:5201-5230`:
  `should_offload = batch_size * n_experts_active >= min_batch_size * n_experts_tot`, with
  `min_batch_size = GGML_CUDA_MIN_BATCH_OFFLOAD` default **32** (`ggml/CMakeLists.txt:130`). For V4.1
  (384 total, 6 active) that is **batch ≥ 32 × 384 / 6 = 2048 tokens** (DERIVED). ik's own heuristic
  says: never upload at decode, never at a verify batch of 8, only deep in prefill.

### 4.2 mainline llama.cpp `-ot` / `--n-cpu-moe` — strictly serial (READ)

Fetched `https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-backend.cpp`
(2026-09-21, 2513 lines). There is exactly one `ggml_backend_sched_compute_splits` (`:1646`), a plain
`for (split_id ...)` loop (`:1656`) with `ggml_backend_synchronize` at `:1667`, `:1682`, `:1690`,
`:1705`, `:1728` (ids readback), `:1786`, `:1790`, and `graph_compute_async` at `:1799`. **No parallel
executor at all** — mainline is a strict subset of ik here. So `-ot exps=CPU` and `--n-cpu-moe` in
mainline serialise every CPU↔GPU boundary, 2 per MoE layer.

### 4.3 KTransformers — the one system that really overlaps at batch 1 (READ)

Source: `kvcache-ai/ktransformers@main`, classic path now under `archive/`.

`archive/ktransformers/operators/experts.py:982-986` (`KDeepseekV3MoE.forward`; identical shape at
`:882-886` for V2 and `:1092-1094` for Mixtral):

```python
if sequence_length == 1 and hasattr(..., "submit_for_one_decode") and torch.cuda.is_current_stream_capturing():
    self.experts.generate_experts.submit_for_one_decode(hidden_states[0], topk_idx[0], topk_weight[0])
    if self.config.n_shared_experts is not None:
        y_ = self.shared_experts(identity).squeeze(0)      # GPU work, concurrent with the CPU pool
    y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
```

The mechanism (`:293-317` and `archive/csrc/ktransformers_ext/cpu_backend/cpuinfer.h:69-90`):

1. `input_tensor_cpu`, `expert_ids_cpu`, `weights_cpu`, `output_cpu` are **`pin_memory=True`** host
   tensors allocated once per CUDA-graph shape (`experts.py:267-291`). Non-blocking D→H copies fill
   them in stream order (`:297-300`).
2. `submit_with_cuda_stream` is **`cudaLaunchHostFunc(stream, func, args)`** (`cpuinfer.h:69-77`).
   The host function enqueues the MoE task into `CPUInfer`'s `TaskQueue` and returns immediately, so
   the stream advances.
3. The GPU then runs the shared expert.
4. `sync_with_cuda_stream` is a second `cudaLaunchHostFunc` calling `task_queue_->sync()`
   (`cpuinfer.h:80-90`) — it blocks the CUDA host-callback thread until the pool drains.
5. `output_cpu → output_gpu` async H→D copy, then combine (`experts.py:310-317`).

So the KTransformers claim of "one CUDA graph with CPU ops injected" is **verified as read**: the CPU
work is two *host nodes* inside the captured graph, not a graph split. Constraints that come with it
(CUDA programming model, not read in their code): a host function may not call CUDA APIs, and work
enqueued after it waits for it to return — which is exactly what makes step 4 a correct join.

What this does **not** do: it never computes routed experts on both processors for the same layer; the
GPU-side concurrent work is only the shared expert. HybriMoE's contribution is precisely to add that.

**Transferability to our box.** KTransformers' headline numbers are **CLAIMED** on Xeon + AMX + DDR5
(their AMX backend is `archive/csrc/ktransformers_ext/operators/amx/moe.hpp`; the AVX2-capable path is
the llamafile one, `.../llamafile/moe.cpp`). Do not transfer them. The *structure* transfers; the
*speed* does not, and it does not need to: our own decode is bandwidth-bound on the host (MEASURED:
16 threads is −5 % vs 32, in-dispatch 127–147 GB/s, HANDOFF §4 "체제"), and our CPU kernels already
run at 95–99 % of ik's (HANDOFF §4). AMX buys arithmetic; we do not need arithmetic at k\* ≈ 2.2 (§4.5).

### 4.4 HybriMoE — intra-layer split of the *routed* experts (MEASURED, with caveats)

arXiv 2504.05897 (DAC'25), code `github.com/PKU-SEC-Lab/HybriMoE`. Hardware: **RTX A6000 + Intel Xeon
Gold 5220R restricted to 10 cores** ("edge simulation"), PCIe; models Mixtral-8x7B-Instruct (8/2),
DeepSeek-V2-Lite-Chat (64/6), Qwen2-57B-A14B (64/8); llama.cpp 4-bit kernels. Headline **1.70×
decode / 1.33× prefill against KTransformers** — batch size not stated in the text I fetched.

- **Intra-layer:** CPU and GPU compute *different experts of the same layer concurrently*. The split is
  chosen by simulating three timelines (CPU compute, GPU compute, PCIe transfer) and picking whichever
  op finishes earliest; GPU prioritises **cached** experts and higher-load ones, CPU takes **uncached**
  and lower-load ones. This is the generalisation of R1 from "shexp only" to "shexp + cached experts",
  which is exactly what R1+R3 become.
- **Inter-layer prefetch:** predicts the next three layers' activations by reusing those layers' gating
  information. It is **prefetch-only**, so it is exact with respect to output whatever the prediction
  accuracy — a wrong prediction costs a wasted transfer, not a wrong token. (I accept "exact" on the
  structural argument plus the fetched description; I did not read their kernel to confirm no weight is
  *substituted*.)
- **Caveat that matters for us:** the 10-core Xeon makes the CPU leg artificially long and therefore
  makes CPU/GPU balancing artificially valuable. Our host has 32 cores and is **bandwidth**-limited,
  not core-limited (MEASURED, `roofline.md:130-137`). Their 1.70× does not transfer; their *scheduling
  shape* does.

### 4.5 Batching across tokens on the CPU side — the arithmetic kills it (DERIVED + MEASURED)

**How many experts does a k-token batch touch?** Expected distinct experts per layer:
`E[distinct] = 384 · (1 − (1 − 6/384)^k)` (DERIVED).

| k | max possible (6k) | E[distinct] | tokens per expert | host bytes/layer | host leg @147.7 |
|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 6.00 | 1.00 | 101.1 MB | 0.684 ms |
| 2 | 12 | 11.91 | 1.008 | 200.7 MB | 1.359 ms |
| 4 | 24 | 23.51 | 1.021 | 396.2 MB | 2.682 ms |
| 8 | 48 | 45.58 | 1.053 | 768.1 MB | 5.200 ms |

At a DSpark verify batch of 8, **each expert sees 1.05 tokens**. Grouping by expert and issuing one
GEMM per expert therefore collapses to 45.6 independent batch-1 gemvs — the grouping saves the ~5 % of
bytes that would otherwise be re-read, nothing more. This is not a model: it is what we **measured**.
WKS-36 (`roofline.md:203-227`) fitted `ms/pass(k) = 10.32 + 50.21 + 17.28·k` with ±0.36 ms residual;
the slope 17.28 ms is **34.43 %** of the pass while the routed-expert share of token bytes is
**34.56 %** — dense amortises perfectly, routed experts amortise **not at all**, and the excess over
the byte floor is 0.13 pp, i.e. zero.

**ggml already groups rows by expert** — so there is nothing to add there.
`ggml/src/ggml.c:18234-18235` allocates `matrix_row_counts[n_as]` and `matrix_rows[n_as][ne11]`;
`:18273-18285` bins every (token, slot) pair into its expert's row list; `:18306-18347` then walks one
expert at a time (`cne1 = matrix_row_counts[cur_a]`) and hands the group to `iqk_mul_mat_moe`
(`:18325-18329`). The grouping machinery is there and correct; at 1.05 rows per expert it has nothing
to group.

**The CPU throughput curve from k = 1 to 8 on our AVX2-only Zen 3 host** (MEASURED,
`roofline.md:130-137`, same-binary thread sweep 4/8/12/16/32 threads × 5 rounds): the compute-bound
slope is `W = 101.9 ms·thread/step` (≈ 12 GB/s per core, ≈ 8 Q3_K weights per cycle), the bus fills at
**~14 threads**, and the tokens-per-weight-pass at which compute catches the bus is **k\* ≈ 2.2**
(3–4 with a kernel that unpacks a row once and multiplies several columns — DERIVED, same passage).
Read together with the table above: **for routed experts the effective k per expert is 1.05 even at
k = 8, so the host never reaches its own balance point on MoE work.** The k\* ≈ 2.2 headroom is only
collectable on *dense* work — attention, shared expert, head — which on V4.1 lives on the GPU anyway.

**How much of KTransformers' gain depends on AMX?** Their AMX path (`operators/amx/moe.hpp`) and their
AMX benchmarks are prefill-shaped (`bench_moe_amx.py`); AMX raises int8 arithmetic throughput, and
decode at 1.05 tokens/expert is a pure streaming read. Their README numbers should be assumed to be
largely AMX + DDR5 + prefill; I found no decode-only AVX2 ablation. Our position is safer than theirs:
we are bandwidth-bound, so ISA is not our lever.

**Does the GPU-side hot-expert cache change the split?** Yes, and it is the only thing that does — see
R3 and §4.8. It does not change the *grouping* question; it changes how many experts are on the host leg.

### 4.6 Cross-layer pipelining and expert prediction (mostly CLAIMED, one-liners)

Exact cross-layer overlap is impossible for the reason the spec states: layer L+1's attention reads
layer L's output, which the host experts produce. What the field does instead, and where each lands:

| technique | what it does | exact? | verdict for us |
|---|---|---|---|
| **Pre-gated MoE** | retrains a preemptive gate so expert selection is decoupled from execution and transfers can overlap | **output-changing** (different trained gate) | out — we gate against a reference |
| **AdapMoE** (arXiv 2408.10284) | cross-layer prefetch + cache allocation; **adapts the number of active experts** | **output-changing** on the adaptive-top-k part; the prefetch part is exact | out for the top-k part |
| **ProMoE / MoE-Infinity / SiDA / APEX / ST-MoE** (e.g. arXiv 2608.11688, 2606.15453) | predict next-layer expert ids, prefetch weights into GPU/faster tier; ST-MoE **CLAIMS** 85 % prediction accuracy and 2.5×/2.2×/1.5× vs GPU / Adap-Gating / Pre-gated | exact if prefetch-only | **≈ 0 for us**: our expert weights are already resident in DDR4. Prediction would only answer *which expert to promote into VRAM* — an input to R3's admission policy, not a scheduling technique |
| **DAOP** (arXiv 2501.10375) | "predictive pre-calculation": computes experts from an earlier/stale hidden state | **output-changing** | out |
| **HybriMoE prefetch** | next-3-layer prediction from those layers' gates, prefetch only | exact | same as above — feeds R3 |
| **KTransformers "expert deferral" (v0.3+)** | — | — | **not verified** (§7) |

**One V4.1-specific piece of exact slack, flagged as INFERENCE.** `docs/research/v41-ops.md:46-49`
records that the hyper-connection mix is **consumed one sub-layer late** (V4 consumed it in the same
sub-layer; ik has the V4 form). If that is literally true of the reference `model.py`, then part of
sub-layer n+1's input does not depend on sub-layer n's output, and a slice of the next sub-layer's GPU
work is legally startable before the host experts return — an *exact* cross-layer overlap that the
literature does not have, because no other deployed model has hyper-connections. I did not verify this
against `model.py` and I am not proposing it as a round; I am flagging it as the one place worth a
reading pass before anyone concludes cross-layer overlap is impossible here.

### 4.7 The rest of the systems list, briefly

- **Fiddler** (arXiv 2402.07033, MEASURED): Quadro RTX 6000 24 GB + Xeon Gold 6126 (48 c, PCIe Gen3
  ×16), and RTX 6000 Ada 49 GB + Xeon Platinum 8480+ (112 c, PCIe Gen4 ×16); **Mixtral-8x7B at 16-bit**
  (>90 GB); **1.26× single-batch vs llama.cpp**, 1.30× long prefill, 11.57× beam search. Cost model:
  *if `cpu_lat(s) > gpu_lat(s) + trans_lat()` run on the GPU, else on the CPU*, where `gpu_lat(s)` is
  ~constant in the token count `s` and `cpu_lat(s)` grows with `s`. **Fiddler does not overlap** — it
  *chooses* one processor per expert. It is the reference for §4.8's break-even, not for §2's overlap.
- **MoE-Lightning, FlexGen, DeepSpeed-Inference / ZeRO-Inference, vLLM `--cpu-offload-gb`,
  HF accelerate** (CLAIMED, abstracts/docs): all throughput-oriented — large batches, layer-wise weight
  swapping, pipelined micro-batches. vLLM's CPU offload and accelerate's device map swap whole layers
  in and out serially with the forward pass. None targets batch-1 latency; none of their numbers should
  be quoted at us.
- **PowerInfer / PowerInfer-2** (CLAIMED): exploit ReLU activation sparsity in *dense* FFNs to keep hot
  neurons on the GPU. Not MoE routing; V4.1's FFN is SwiGLU with ±10 clamp (`v41-ops.md:28`), not ReLU.
  The hot/cold *idea* survives as R3; the mechanism does not.
- **SGLang + KTransformers integration** (CLAIMED): serving front end over the same KT CPU kernels —
  same overlap structure as §4.3, larger batches.

### 4.8 Chunked / dynamic placement — the break-even for our link (DERIVED, formula shown)

**Setup.** One V4.1 expert = **16.77 MB** (§2.1). The host computes it by streaming those bytes at
147.7 GB/s; the GPU can only compute it if the bytes first cross PCIe at 26.2 GB/s (pinned), then are
read from VRAM at 700 GB/s.

```
t_host_compute(E)  = bytes(E) / 147.7 GB/s  = 16.77 MB / 147.7 = 0.1136 ms
t_upload(E)        = bytes(E) /  26.2 GB/s  = 16.77 MB /  26.2 = 0.6401 ms
t_gpu_compute(E)   = bytes(E) / 700   GB/s  = 16.77 MB / 700   = 0.0240 ms
upload + gpu       = 0.6641 ms      vs      host 0.1136 ms     →  5.8× worse
```

**Uploading an expert in order to compute it once is never right on this box**, at any batch size, by
the bandwidth ratio 147.7 / 26.2 = **5.6×** — the same bytes must move either way and the host's path
is 5.6× wider. Concretely: *uploading one expert costs about as much as computing all six on the host*
(0.640 vs 0.684 ms). Fiddler's rule reaches the same place from the other side: `cpu_lat(s) >
gpu_lat(s) + trans_lat()` is only satisfiable when `s` (tokens per expert) is large enough that CPU
*compute* dominates — and §4.5 shows our tokens-per-expert is 1.05 even at k = 8. ik agrees in code:
offload only at batch ≥ 2048 for a 384/6 model (§4.1).

**The one case where uploading pays: amortisation over future tokens (= R3, the cache).**
```
payback_hits = t_upload(E) / (t_host_compute(E) − t_gpu_compute(E))
             = 0.6401 / (0.1136 − 0.0240) = 0.6401 / 0.0896 = 7.14 future hits
```
The base rate of re-picking a given expert at a given layer is 6/384 = 1.56 %/token, so 7.14 hits at
the base rate needs ≈ 457 tokens (DERIVED). That is why the admission rule must key on *evidence of
hotness*, not on a miss: phylliida's **second miss inside a 64-step window** (`v41-ports.md:36`) admits
only experts whose observed rate is ≈ 1/32 per step, ~2× the base rate, and that is the cheapest signal
available. Two measured constraints ride along: promotion staging **must be pinned** (pageable staging
serialised against the critical-path input copy in the driver's staging lock and cost **~27 ms/step** —
MEASURED, `v41-ports.md:36-37`), and the copy engine does not steal compute bandwidth (**26.15 GB/s
with the compute stream saturated** — MEASURED, same line), so a promotion is nearly free in wall time
and the whole question is VRAM thrash.

**At k = 8.** Host leg 5.200 ms/layer (§4.5). In that window the copy engine can move
`5.200 ms × 26.2 GB/s = 136.2 MB = 8.1 experts` of 45.6 → **at most 17.8 % of the layer's expert work
could migrate**, and only if those experts are wanted and VRAM has room. Marginal, positive, and it is
the same machinery as R3 — not a separate technique.

**At prefill.** Here uploading wins, because a prefill batch touches nearly all 384 experts and each
expert serves many tokens, so the transfer amortises over tokens rather than over future steps. ik's
threshold (batch ≥ 2048 for 384/6) is the upstream number; our own crossover should be re-derived once
A5 gives us a prefill GEMM and we can measure `t_gpu_compute` for a wide expert.

### 4.9 Host-side mechanics that decide whether overlap is real

**The join primitive.** Three options, ranked:

1. **Stream memory ops on host-mapped pinned memory** — `cuStreamWriteValue32` to publish "ids are
   ready" and `cuStreamWaitValue32` to block the stream until the host writes "results are ready";
   as graph nodes via `cuGraphAddBatchMemOpNode` (CUDA ≥ 11.7). The host pool spins on the mapped
   flag. This keeps **one captured graph per layer** with the CPU boundary inside it, preserving the
   1.4 µs/launch replay cost instead of 4.1 µs eager (MEASURED, HANDOFF §7 P0). Latency per leg
   ~1–3 µs (INFERENCE — not measured). **Two unverified preconditions, and they are the first thing
   the B2 round should check** (§7): the device attribute
   `CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS` on this driver, and whether cuda-oxide's bindgen
   output exposes `cuStreamWaitValue32` / `cuGraphAddBatchMemOpNode` (HANDOFF §7 records that bindings
   are generated from the regex `^cu.*` into `OUT_DIR`, so they very likely exist and only the safe
   wrapper is missing — exactly the shape of our `graph.rs` and of upstream ledger item #3).
2. **`cudaLaunchHostFunc` host nodes** — KTransformers' choice (READ, §4.3). Works today, captures into
   a graph, and is the proven-in-production form. Costs: a host-callback thread wake per node
   (INFERENCE: ~5–20 µs, not measured), and a host function **may not call CUDA APIs**, so the result
   copy must be a separate node after it.
3. **A captured spin kernel polling a mapped flag** — lowest latency, burns an SM, and deadlocks if the
   host ever waits on the GPU while the GPU waits on the host. Only as a measurement probe.

Not an option: `cudaStreamSynchronize` per boundary. That is today's design and it is what §4.1 shows
ik doing 80 times per token.

**Buffers.** ids (24 B), weights (6 × f32), hidden (20.5 KB, or 81.9 KB with 4 hyper-connection
streams) and results (20.5 KB) all live in **host-mapped pinned** buffers allocated once at `Stage`
load — our `Stage` already allocates all scratch at load with zero allocation inside `step`
(HANDOFF §7 P8a), so this is an extension, not a change of shape. Zero-copy (mapped) is right here
precisely because the payloads are tiny and the cost is the round trip, not the bytes (§2.3).

**Thread-pool wake.** Our pool's dispatch tax is **4.65 µs** (MEASURED, MUL-35), and removing an
unconditional collection mutex at chunk end was worth **+44 %** on CPU decode (HANDOFF §4). For R1 the
pool must already be *spinning* on the mapped flag when the ids land, not woken by a condvar — the
"notify 생략 + 워커별 완료 표식" change we already made (pool bench 4.56 → 1.61 µs, HANDOFF §4) is the
right precedent.

**NUMA and SMT.** The 5975WX is single-socket with 8 channels; the NUMA pinning KTransformers needs on
dual-socket Xeons is a much smaller lever here (INFERENCE — NPS setting on this box not checked). SMT:
we measured the bus filling at **~14 threads** and 16 threads being **−5 %** against 32
(`roofline.md:132`, HANDOFF §4), so the expert pool should be sized near the bus knee, not at 64
threads — and once R1 lands, the remaining cores have nothing useful to do, which is another way of
saying the host leg is bandwidth-bound and cannot be shortened by giving it more cores.

**CUDA graphs with a CPU op in the middle.** KTransformers' claim is verified (§4.3): host nodes, one
graph. ik takes the other route — it never puts CPU work in a graph; it captures one CUDA graph per
*GPU split* per `graph_compute` call, and disables capture on a list of conditions
(`ggml-cuda.cu:4472-4564` node compatibility; `:4723-4765` including `disable_due_to_gpu_arch`,
`disable_due_to_too_many_updates` at `number_consecutive_updates >= 4`,
`disable_due_to_failed_graph_capture`). We measured what graphs are worth to ik on this card: **tg96
196.2 → 216.9 tok/s, +10.6 %**, with `GGML_CUDA_DISABLE_GRAPHS=1` as the control (MEASURED,
`gpu-design.md:70`). Losing graph capture at the CPU boundary would cost us the same order; phylliida
measured **−13.8 % TG** when their expert-cache classification broke capture (MEASURED,
`v41-ports.md:33`). **This is why R2 is sequenced before R3 even though R3 is worth more.**

**GPUDirect Storage for the engram table: irrelevant.** Per token the engram reads are 48 rows ×
272 B = **12.75 KiB**, scattered across ~104 GB, worst case 48 pages / 192 KiB (`v41-ops.md:63-70`).
GDS removes a host bounce for *bandwidth*; this workload is **I/O count**, not bandwidth. The measured
lever is already known and already in the plan: our own commit took major faults per token from 41–62
to 1–11 and **13.6 → 18.4 tok/s** by prefetching the rows (MEASURED, `v41-ports.md:12`), and the row
ids for the next token are computable the instant the current token is sampled (`v41-ops.md:69-70`),
giving a full step of slack. Keep `madvise`/`io_uring` readahead at sampling time; skip GDS.

---

## 5. What not to copy, and why

| popular idea | why it does not pay here | number that kills it |
|---|---|---|
| **Upload experts over PCIe and compute them on the GPU** (Fiddler's GPU arm, llama.cpp's offload path) | the same bytes must move either way and the host path is 5.6× wider; uploading *one* expert costs what computing *all six* on the host costs | 0.640 ms upload vs 0.684 ms for the whole host leg; ik's own rule needs batch ≥ 2048 for 384/6 (§4.8, §4.1) |
| **Expert prediction / prefetch** (Pre-gated, ProMoE, MoE-Infinity, AdapMoE, SiDA, ST-MoE, APEX, HybriMoE's inter-layer arm) | these prefetch weights from *slow storage or host RAM into the GPU*. Ours are already in DDR4 and the host computes them in place — **there is nothing to prefetch**. The only residue is "which expert to promote", i.e. an input to R3's admission rule | 0 ms saved at batch 1; §4.6 |
| **Group verify-batch tokens by expert into one host GEMM** | at k = 8 each expert sees 1.05 tokens; ggml already does the grouping (`ggml.c:18234-18347`) and has nothing to group | E[distinct] = 45.6 of 48 at k = 8; measured amortisation excess 0.13 pp (§4.5) |
| **Stale-residual / predictive pre-calculation** (DAOP), **adaptive top-k** (AdapMoE), **retrained preemptive gate** (Pre-gated MoE) | **output-changing**. Our gates are band/bit identity against an ik reference; any of these needs a separate justification and a new oracle | §4.6 |
| **Two-stage host/GPU pipeline as a latency fix** | it is a **throughput** result and needs ≥2 *independent sequences*; the k tokens of a verify pass are one pass and do not qualify | 1.67× throughput, 0 % single-stream latency (`roofline.md:148-156`) |
| **AMX-derived CPU numbers** (KTransformers, and anything quoting them) | decode at 1.05 tokens/expert is a streaming read; AMX buys arithmetic we do not need, and we have none | k\* ≈ 2.2 measured, 16 threads −5 % vs 32 (§4.5) |
| **GPUDirect Storage for engram** | engram is I/O-count-bound (48 random 272 B reads/token), not bandwidth-bound | 12.75 KiB/token; the measured lever was fault count, 13.6 → 18.4 tok/s (§4.9) |
| **Making the boundaries cheaper as the goal** | sync is ~1 % of a token; the serialisation is ~40 %/60 % idle | §2.4 |

---

## 6. Interaction with speculative decoding and with the expert cache

**DSpark verify batches (k ≈ 4–8) invert the ranking.** From §4.5, at k = 8 the host expert leg is
5.200 ms/layer against a GPU leg that barely grows (dense amortises perfectly — MEASURED slope
34.43 % vs byte share 34.56 %, `roofline.md:216-218`). So:

- R1's gain **shrinks to 1.0 % (floor) / 2.5 % (envelope)**: the GPU-side overlappable work (shexp,
  0.054–0.152 ms) is a small fraction of a 5.2 ms host leg.
- R3's gain **grows**: every expert moved to VRAM removes ~0.114 ms × (tokens hitting it) from the
  critical path, and at k = 8 that is 1.05 tokens each, 45.6 experts deep.
- The already-measured ceiling stands: verification amortises only the dense share, so the best a
  k-token pass can do is **1 / 0.654 = 1.53×** (`roofline.md:224-225`), and the remaining lever is
  acceptance rate, not verify cost.
- One correctness item to carry: `skelectric` hit a window-truncation bug that was **clean at batch 1
  and produced plausible mixed text only in multi-token verify batches** (`v41-ports.md:70-71`). Any
  hybrid-boundary round that touches the verify path must run the multi-token verify gate.

**The expert cache (`docs/research/v41-ports.md:29-39`) is where R1 and R3 meet.** R1 without a cache
overlaps only the shared expert; R1 with a cache overlaps shexp *plus the cached experts*, which is
exactly HybriMoE's intra-layer scheduler (§4.4). The design inputs already recorded on that card
survive this round unchanged and are the right ones: two-path **value** masks (bit-identical to the
single path), trash slot for out-of-range ids, admission on second miss in a 64-step window,
token-bucket rate limit, read-only during prefill, **pinned** promotion staging, and device-side
classification so capture survives. Two things this round adds:

1. **The hit rate is the whole number and it is unmeasured for V4.1.** phylliida's 0.46–0.52 and
   `TG ≈ 9.01/(1−hit)` are from **GLM-5.3-Flash, 288 experts top-8** (`v41-ports.md:29,39`). V4.1 is
   384/6 — a *lower* per-expert base rate (1.56 % vs 2.78 %) and smaller slices. Expect a lower hit
   rate; measure it offline from a router-id trace before sizing VRAM. This is a cheap, GPU-free
   experiment: replay 32 prompts through the router only and count.
2. **Score the cache as "bytes removed from the host leg", not as "hit rate".** With R1 in place the
   layer wall is `rest + boundary + max(host_remaining, shexp + gpu_experts)`, so promotion stops
   paying the moment `shexp + gpu_experts ≥ host_remaining`. In the envelope bracket that crossover is
   already at **hit ≈ 0.5** (§2.5); in the floor bracket it is around hit ≈ 0.83 (DERIVED:
   `0.054 + h·0.114·6/6·(700/147.7)⁻¹` reaches `(1−h)·0.684` near h = 0.83). **There is a
   cheapest-correct cache size and it is smaller than "as much VRAM as fits"** — and which bracket we
   are in is decided by our own gemv efficiency, which makes the A6b/kernel-shape work an input to the
   cache sizing decision, not an unrelated track.

---

## 7. What I could not verify

- **The spec's given fact "~0.20 ms per layer per token" carries no model name and is a V2-Lite
  number.** V2-Lite: 6 × 4.254 MiB = 26.77 MB / 147.7 GB/s = **0.181 ms** (DERIVED; expert size from
  `roofline.md:31`). For **V4.1 the host leg is 0.684 ms/layer** (§2.1). Every piece of arithmetic in
  this report uses the V4.1 figure. The 3.8× difference changes conclusions — please correct the note
  this came from.
- `CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS` on the box's driver, and whether cuda-oxide's generated
  bindings expose `cuStreamWaitValue32` / `cuStreamWriteValue32` / `cuGraphAddBatchMemOpNode`. I could
  not touch the box and the bindings are a build-time `OUT_DIR` artifact. **First check of the B2 round.**
- Latency of `cudaLaunchHostFunc` and of a stream mem-op wait on this hardware. I quote ranges as
  INFERENCE; both are one micro-benchmark away and belong in the same B2 probe.
- **The GPU bracket itself.** 700 GB/s is a stream-read derivation and 247 GB/s is ik's whole-pass
  average; our actual per-layer GPU leg for V4.1 is unmeasured and will sit between them. Every
  percentage in this report is therefore a range, not a point.
- KTransformers' "expert deferral / skipping" in v0.3+ — not in the archived operator tree; the
  successor scheduler lives at `kt-kernel/python/experts.py` and
  `doc/en/kt-kernel/experts-sched-Tutorial.md`, which I did not read. Exact vs output-changing is **open**.
- Whether HybriMoE's prefetch ever *substitutes* a weight rather than only staging it. I concluded
  "exact" from the structural argument plus the fetched description, not from their kernel source.
- The V4.1 expert-cache hit rate — no measurement exists anywhere, including in the three ik ports
  (`v41-ports.md:10-13`).
- The hyper-connection "one sub-layer late" slack (§4.6) — recorded in our own summary, not
  re-verified against the reference `model.py` in this round.
- Fiddler's per-expert threshold constants: I have the comparison
  (`cpu_lat(s) > gpu_lat(s) + trans_lat()`) from the paper's HTML, not the numbers behind it.

---

## 8. Improvement opportunities noticed outside the asked scope (report only, nothing touched)

1. `/Users/hckim/repo/bloomery/docs/roofline.md:96-98` — states V4.1 attention is MLA
   ("kv_lora 512 + rope 64"), contradicted by `docs/research/v41-ops.md:40-44` (not MLA: one 512-d
   latent is K *and* V, ≤640 gathered rows). Stale line that survived the B0a correction; ~3 lines.
2. `/Users/hckim/repo/bloomery/docs/roofline.md:161` — "engram … 상주시켜도 처리량은 안 바뀐다 (지연만)"
   is contradicted by the measured 13.6 → 18.4 tok/s from cutting major faults
   (`docs/research/v41-ports.md:12`). At this scale fault count *is* throughput; ~2 lines.
3. `/Users/hckim/repo/bloomery/docs/orchestration.md:92` — the B2 card's gate column names "왕복 µs"
   but no idle-time metric and no sync-mechanism decision. §2.4 says the round will be judged on the
   wrong axis if it optimises the round trip instead of the overlap; ~2 lines, and this report is the
   input.
4. `/Users/hckim/repo/bloomery/docs/plan.md:63` — the B2 card's gate is "층·토큰당 호스트 항 ms 대
   roofline 하한 (3.34 GB ÷ 147.7 = 22.6 ms)". That bound is only reachable if the host leg is the
   *only* thing on the critical path, which is what R1 is for; the card should also carry a GPU-idle
   fraction so a round that hits the host bound while leaving the GPU 40 % idle does not read as green;
   ~2 lines.
5. `/Users/hckim/repo/bloomery/docs/roofline.md:50-56` — the serving placement table (A6000 4⅓ / 3090
   2⅔ / DDR4 33 blocks) predates both the "A6000 usable for development" decision (HANDOFF §7) and the
   expert-cache design; placement should become a function of the cache, not a fixed block split;
   ~5 lines.
6. `/Users/hckim/repo/upstream/v41-ports/mine/ggml/src/ggml-backend.cpp:2398` — a stray
   `printf("Recording event %d, %d\n", ...)` sits in the hot loop of the non-OpenMP parallel split
   executor (debug leftover; the OpenMP arm at `:2321` has no such printf). 1-line upstream candidate.
7. `/Users/hckim/repo/upstream/v41-ports/mine/ggml/src/ggml-backend.cpp:2019` —
   `constexpr bool k_set_sync = false;` makes every `needs_sync[...] = k_set_sync` a dead store of
   `false`, while the surrounding logic reads as if the flag were meant to be settable. Either a
   deliberate kill-switch that deserves a comment or a bug that suppresses a needed sync; worth an
   upstream question before anyone builds on this path.

DONE-hybrid
